"""Hidden Markov Model module for regime detection and sequence modeling.

Implements forward-backward algorithm, Viterbi decoding, Baum-Welch EM training,
and regime-based forecasting using discrete hidden states.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


@dataclass
class HMMResult:
    states: List[int]
    state_probabilities: List[List[float]]
    transition_matrix: List[List[float]]
    emission_params: List[dict]
    log_likelihood: float
    n_states: int
    n_iterations: int
    converged: bool
    method: str = "hmm"


@dataclass
class RegimeResult:
    regimes: List[int]
    regime_labels: List[str]
    regime_statistics: List[dict]
    transition_matrix: List[List[float]]
    regime_probabilities: List[List[float]]
    log_likelihood: float
    n_regimes: int
    method: str = "regime_detection"


class HiddenMarkovModel:
    """Discrete HMM with Gaussian emissions for regime detection.

    Implements:
    - Forward algorithm (alpha pass)
    - Backward algorithm (beta pass)
    - Baum-Welch EM training
    - Viterbi decoding
    - Regime statistics and forecasting
    """

    def __init__(self, n_states: int = 3, seed: int = 42,
                 max_iter: int = 100, tol: float = 1e-6):
        self.n_states = n_states
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol
        self.pi: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self._fitted = False

    def _initialize(self, data: np.ndarray):
        rng = np.random.RandomState(self.seed)
        n = len(data)
        self.pi = np.ones(self.n_states) / self.n_states
        self.A = rng.dirichlet(np.ones(self.n_states), self.n_states)
        quantiles = np.linspace(0, 100, self.n_states + 2)[1:-1]
        self.means = np.percentile(data, quantiles)
        self.stds = np.full(self.n_states, np.std(data) / self.n_states)

    def _emission_prob(self, x: float, state: int) -> float:
        mu = self.means[state]
        sigma = self.stds[state] + 1e-10
        return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def _emission_probs(self, x: float) -> np.ndarray:
        probs = np.array([self._emission_prob(x, s) for s in range(self.n_states)])
        return probs / (probs.sum() + 1e-300)

    def _forward(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        T = len(data)
        alpha = np.zeros((T, self.n_states))
        b0 = self._emission_probs(data[0])
        alpha[0] = self.pi * b0
        alpha[0] /= (alpha[0].sum() + 1e-300)
        log_scale = 0.0
        for t in range(1, T):
            b_t = self._emission_probs(data[t])
            for j in range(self.n_states):
                alpha[t, j] = b_t[j] * np.sum(alpha[t-1] * self.A[:, j])
            scale = alpha[t].sum() + 1e-300
            alpha[t] /= scale
            log_scale += np.log(scale + 1e-300)
        return alpha, log_scale

    def _backward(self, data: np.ndarray) -> np.ndarray:
        T = len(data)
        beta = np.zeros((T, self.n_states))
        beta[-1] = np.ones(self.n_states)
        for t in range(T - 2, -1, -1):
            b_next = self._emission_probs(data[t + 1])
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * b_next * beta[t + 1])
            beta[t] /= (beta[t].sum() + 1e-300)
        return beta

    def fit(self, data: np.ndarray) -> 'HiddenMarkovModel':
        if data.ndim > 1:
            data = data.ravel()
        if len(data) < self.n_states * 2:
            raise ValueError(f"Need at least {self.n_states * 2} data points")
        self._initialize(data)
        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            alpha, log_scale = self._forward(data)
            beta = self._backward(data)
            ll = log_scale
            gamma = alpha * beta
            gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-300)
            xi = np.zeros((len(data) - 1, self.n_states, self.n_states))
            for t in range(len(data) - 1):
                b_next = self._emission_probs(data[t + 1])
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = alpha[t, i] * self.A[i, j] * b_next[j] * beta[t + 1, j]
                xi[t] /= (xi[t].sum() + 1e-300)
            self.pi = gamma[0] / (gamma[0].sum() + 1e-300)
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.A[i, j] = xi[:, i, j].sum() / (gamma[:-1, i].sum() + 1e-300)
                self.A[i] /= (self.A[i].sum() + 1e-300)
            for s in range(self.n_states):
                w = gamma[:, s]
                w_sum = w.sum() + 1e-300
                self.means[s] = float(np.sum(w * data) / w_sum)
                diff = data - self.means[s]
                self.stds[s] = float(np.sqrt(np.sum(w * diff**2) / w_sum)) + 1e-6
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        self._fitted = True
        return self

    def decode(self, data: np.ndarray) -> HMMResult:
        if not self._fitted:
            self.fit(data)
        T = len(data)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        b0 = self._emission_probs(data[0])
        delta[0] = np.log(self.pi + 1e-300) + np.log(b0 + 1e-300)
        for t in range(1, T):
            b_t = self._emission_probs(data[t])
            for j in range(self.n_states):
                scores = delta[t-1] + np.log(self.A[:, j] + 1e-300)
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = scores[psi[t, j]] + np.log(b_t[j] + 1e-300)
            delta[t] -= np.max(delta[t])
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        alpha, log_scale = self._forward(data)
        beta = self._backward(data)
        gamma = alpha * beta
        gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-300)
        ll = log_scale
        emission_params = []
        for s in range(self.n_states):
            emission_params.append({
                "mean": float(self.means[s]),
                "std": float(self.stds[s]),
                "variance": float(self.stds[s] ** 2),
            })
        return HMMResult(
            states=states.tolist(),
            state_probabilities=gamma.tolist(),
            transition_matrix=self.A.tolist(),
            emission_params=emission_params,
            log_likelihood=float(ll),
            n_states=self.n_states,
            n_iterations=0,
            converged=True,
            method="hmm_viterbi"
        )

    def detect_regimes(self, data: np.ndarray,
                       regime_names: Optional[List[str]] = None) -> RegimeResult:
        result = self.decode(data)
        states = np.array(result.states)
        regime_stats = []
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() > 0:
                regime_data = data[mask]
                stats = {
                    "mean": float(np.mean(regime_data)),
                    "std": float(np.std(regime_data)),
                    "min": float(np.min(regime_data)),
                    "max": float(np.max(regime_data)),
                    "duration": int(mask.sum()),
                    "persistence": float(self.A[s, s]),
                    "frequency": float(mask.sum() / len(data)),
                }
            else:
                stats = {
                    "mean": float(self.means[s]),
                    "std": float(self.stds[s]),
                    "min": 0.0, "max": 0.0,
                    "duration": 0,
                    "persistence": float(self.A[s, s]),
                    "frequency": 0.0,
                }
            regime_stats.append(stats)
        if regime_names is None:
            regime_names = []
            for s in range(self.n_states):
                stats = regime_stats[s]
                if stats["mean"] > np.mean(data) + 0.5 * np.std(data):
                    regime_names.append("bull/expansion")
                elif stats["mean"] < np.mean(data) - 0.5 * np.std(data):
                    regime_names.append("bear/recession")
                else:
                    regime_names.append("neutral/range-bound")
        return RegimeResult(
            regimes=result.states,
            regime_labels=regime_names,
            regime_statistics=regime_stats,
            transition_matrix=result.transition_matrix,
            regime_probabilities=result.state_probabilities,
            log_likelihood=result.log_likelihood,
            n_regimes=self.n_states,
            method="hmm_regime"
        )

    def forecast(self, data: np.ndarray, n_ahead: int = 10,
                 n_simulations: int = 100) -> dict:
        if not self._fitted:
            self.fit(data)
        result = self.decode(data)
        current_state = result.states[-1]
        state_probs = result.state_probabilities[-1]
        rng = np.random.RandomState(self.seed)
        per_horizon_samples = {h: [] for h in range(n_ahead)}
        for _ in range(n_simulations):
            state = current_state
            for h in range(n_ahead):
                state = rng.choice(self.n_states, p=self.A[state])
                value = rng.normal(self.means[state], self.stds[state])
                per_horizon_samples[h].append(value)
        forecasts = []
        lower = []
        upper = []
        for h in range(n_ahead):
            samples_h = per_horizon_samples[h]
            if samples_h:
                forecasts.append(float(np.mean(samples_h)))
                lower.append(float(np.percentile(samples_h, 5)))
                upper.append(float(np.percentile(samples_h, 95)))
            else:
                forecasts.append(float(np.mean(data)))
                lower.append(float(np.percentile(data, 5)))
                upper.append(float(np.percentile(data, 95)))
        return {
            "forecast": forecasts,
            "confidence_lower": lower,
            "confidence_upper": upper,
            "current_state": int(current_state),
            "state_probabilities": state_probs.tolist() if hasattr(state_probs, 'tolist') else state_probs,
            "n_simulations": n_simulations,
            "method": "hmm_forecast"
        }

    def regime_change_probability(self, data: np.ndarray) -> dict:
        if not self._fitted:
            self.fit(data)
        result = self.decode(data)
        transitions = []
        for t in range(1, len(data)):
            if result.states[t] != result.states[t-1]:
                transitions.append({
                    "time": t,
                    "from_state": result.states[t-1],
                    "to_state": result.states[t],
                    "to_label": f"State {result.states[t]}",
                    "probability": float(max(result.state_probabilities[t])),
                })
        transition_matrix = self.A
        stability = {f"state_{s}": float(transition_matrix[s, s]) for s in range(self.n_states)}
        avg_durations = {}
        for s in range(self.n_states):
            durations = []
            current_dur = 0
            for t in range(len(data)):
                if result.states[t] == s:
                    current_dur += 1
                else:
                    if current_dur > 0:
                        durations.append(current_dur)
                    current_dur = 0
            if current_dur > 0:
                durations.append(current_dur)
            avg_durations[f"state_{s}"] = float(np.mean(durations)) if durations else 0.0
        return {
            "regime_changes": transitions[:20],
            "n_changes": len(transitions),
            "change_frequency": float(len(transitions) / len(data)),
            "stability": stability,
            "average_durations": avg_durations,
            "method": "hmm_regime_change"
        }

    def state_duration_analysis(self, data: np.ndarray) -> dict:
        if not self._fitted:
            self.fit(data)
        result = self.decode(data)
        states = result.states
        durations = {s: [] for s in range(self.n_states)}
        current_state = states[0]
        current_dur = 1
        for t in range(1, len(states)):
            if states[t] == current_state:
                current_dur += 1
            else:
                durations[current_state].append(current_dur)
                current_state = states[t]
                current_dur = 1
        durations[current_state].append(current_dur)
        stats = {}
        for s in range(self.n_states):
            durs = durations[s]
            if durs:
                stats[f"state_{s}"] = {
                    "mean_duration": float(np.mean(durs)),
                    "max_duration": int(max(durs)),
                    "min_duration": int(min(durs)),
                    "expected_duration": float(1.0 / max(1.0 - self.A[s, s], 1e-10)),
                    "total_time": int(sum(durs)),
                    "n_spells": len(durs),
                }
            else:
                stats[f"state_{s}"] = {
                    "mean_duration": 0.0,
                    "max_duration": 0,
                    "min_duration": 0,
                    "expected_duration": float(1.0 / max(1.0 - self.A[s, s], 1e-10)),
                    "total_time": 0,
                    "n_spells": 0,
                }
        return {
            "duration_statistics": stats,
            "method": "hmm_duration"
        }