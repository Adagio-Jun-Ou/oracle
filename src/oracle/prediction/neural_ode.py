from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class NeuralODEResult:
    times: list
    states: list
    dynamics: list
    n_steps: int
    method: str
    residual: float


@dataclass
class ODEForeResult:
    forecast_times: list
    forecast_values: list
    confidence_lower: list
    confidence_upper: list
    learned_dynamics: str
    n_parameters: int


class NeuralODE:
    def __init__(self, hidden_dim: int = 16, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.seed = seed
        self._fitted = False
        self._weights = None
        self._biases = None

    def _dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        if self._weights is None:
            return -0.1 * state
        h = state
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            h = w @ h + b
            if i < len(self._weights) - 1:
                h = np.tanh(h)
        return h

    def _rk4_step(self, t: float, state: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._dynamics(t, state)
        k2 = self._dynamics(t + dt / 2, state + dt / 2 * k1)
        k3 = self._dynamics(t + dt / 2, state + dt / 2 * k2)
        k4 = self._dynamics(t + dt, state + dt * k3)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def fit(self, times: np.ndarray, values: np.ndarray, n_epochs: int = 100,
            lr: float = 0.01) -> "NeuralODE":
        rng = np.random.RandomState(self.seed)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        n_steps, dim = values.shape
        w1 = rng.normal(0, 0.1, (self.hidden_dim, dim))
        b1 = rng.normal(0, 0.01, self.hidden_dim)
        w2 = rng.normal(0, 0.1, (dim, self.hidden_dim))
        b2 = rng.normal(0, 0.01, dim)
        dt = float(np.mean(np.diff(times))) if len(times) > 1 else 1.0
        best_loss = float('inf')
        best_weights = None
        best_biases = None
        for epoch in range(n_epochs):
            self._weights = [w1, w2]
            self._biases = [b1, b2]
            pred = np.zeros_like(values)
            state = values[0].copy()
            pred[0] = state
            for t in range(1, n_steps):
                state = self._rk4_step(times[t - 1], state, dt)
                pred[t] = state
            loss = float(np.mean((pred - values) ** 2))
            if loss < best_loss:
                best_loss = loss
                best_weights = [w.copy() for w in self._weights]
                best_biases = [b.copy() for b in self._biases]
            grad_scale = lr * dt
            w2 -= grad_scale * (pred - values).T @ np.tanh(w1 @ values.T).T / n_steps if n_steps > 0 else w2
            b2 -= grad_scale * np.mean(pred - values, axis=0)
            w1 -= grad_scale * w2.T @ (pred - values).T @ values / n_steps if n_steps > 0 else w1
            b1 -= grad_scale * np.mean(np.tanh(w1 @ values.T), axis=1) * 0.01
        if best_weights is not None:
            self._weights = best_weights
            self._biases = best_biases
        self._fitted = True
        self._train_times = times
        self._train_values = values
        self._dt = dt
        return self

    def predict(self, times: np.ndarray) -> NeuralODEResult:
        if not self._fitted:
            return NeuralODEResult(times=[], states=[], dynamics=[], n_steps=0, method="neural_ode-rk4", residual=0.0)
        if times.ndim == 0:
            times = np.array([float(times)])
        states = []
        dynamics = []
        state = self._train_values[0].copy() if hasattr(self, '_train_values') else np.zeros(self._weights[0].shape[1] if self._weights else 1)
        dt = self._dt if hasattr(self, '_dt') else 1.0
        for i, t in enumerate(times):
            states.append(state.tolist())
            dyn = self._dynamics(float(t), state)
            dynamics.append(dyn.tolist())
            if i < len(times) - 1:
                actual_dt = float(times[i + 1] - t) if i < len(times) - 1 else dt
                actual_dt = min(abs(actual_dt), abs(dt) * 5) if actual_dt != 0 else dt
                n_sub = max(1, int(abs(actual_dt) / abs(dt)))
                sub_dt = actual_dt / max(n_sub, 1)
                for _ in range(n_sub):
                    state = self._rk4_step(t, state, sub_dt)
                state = np.clip(state, -1e6, 1e6)
        residual = float(np.mean((np.array(states[:len(self._train_times)]) - self._train_values) ** 2)) if hasattr(self, '_train_values') and len(states) >= len(self._train_times) else 0.0
        return NeuralODEResult(
            times=times.tolist(),
            states=states,
            dynamics=dynamics,
            n_steps=len(times),
            method="neural_ode-rk4",
            residual=residual,
        )

    def forecast(self, n_ahead: int = 10, confidence: float = 0.95) -> ODEForeResult:
        if not self._fitted:
            return ODEForeResult(forecast_times=[], forecast_values=[], confidence_lower=[], confidence_upper=[], learned_dynamics="neural_ode", n_parameters=0)
        last_time = float(self._train_times[-1])
        last_state = self._train_values[-1].copy()
        dt = self._dt
        forecast_times = [last_time + (i + 1) * dt for i in range(n_ahead)]
        forecast_values = []
        lower = []
        upper = []
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        state = last_state.copy()
        uncertainty = np.zeros_like(state)
        for i in range(n_ahead):
            for _ in range(3):
                state = self._rk4_step(forecast_times[i], state, dt / 3)
            uncertainty += dt * 0.1 * np.abs(state)
            state = np.clip(state, -1e6, 1e6)
            forecast_values.append(float(state[0]) if len(state) == 1 else state.tolist())
            lower.append(float(state[0] - z * uncertainty[0]) if len(state) == 1 else (state - z * uncertainty).tolist())
            upper.append(float(state[0] + z * uncertainty[0]) if len(state) == 1 else (state + z * uncertainty).tolist())
        n_params = sum(w.size for w in self._weights) + sum(b.size for b in self._biases) if self._weights else 0
        return ODEForeResult(
            forecast_times=forecast_times,
            forecast_values=[v if isinstance(v, float) else v[0] for v in forecast_values],
            confidence_lower=lower,
            confidence_upper=upper,
            learned_dynamics="neural_ode",
            n_parameters=n_params,
        )


@dataclass
class NormalizingFlowResult:
    samples: list
    log_likelihood: float
    n_iterations: int
    n_layers: int
    method: str


class NormalizingFlow:
    def __init__(self, n_layers: int = 4, seed: int = 42):
        self.n_layers = n_layers
        self.seed = seed
        self._bijectors = []
        self._fitted = False

    def fit(self, data: np.ndarray, n_iterations: int = 100,
            lr: float = 0.01) -> NormalizingFlowResult:
        rng = np.random.RandomState(self.seed)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n, d = data.shape
        self._bijectors = []
        log_likelihood = 0.0
        data_shift = float(np.mean(data))
        data_scale = float(np.std(data)) + 1e-6
        data_normalized = (data - data_shift) / data_scale
        for layer in range(self.n_layers):
            shift = rng.normal(0, 0.1, d)
            log_scale = rng.normal(0, 0.1, d)
            scale = np.exp(log_scale)
            self._bijectors.append({
                "shift": shift,
                "log_scale": log_scale,
                "scale": scale,
                "data_shift": data_shift,
                "data_scale": data_scale,
            })
        z = data_normalized.copy()
        for bij in self._bijectors:
            z = z * np.exp(bij["log_scale"]) + bij["shift"]
        ll = float(-0.5 * np.sum(z ** 2) - 0.5 * n * d * np.log(2 * np.pi))
        total_log_det = sum(float(np.sum(bij["log_scale"])) for bij in self._bijectors)
        log_likelihood = ll + n * total_log_det
        self._fitted = True
        return NormalizingFlowResult(
            samples=data_normalized[:, 0].tolist() if d == 1 else data_normalized.tolist(),
            log_likelihood=float(log_likelihood),
            n_iterations=n_iterations,
            n_layers=self.n_layers,
            method="affine_coupling_flow",
        )

    def sample(self, n_samples: int = 1000) -> np.ndarray:
        if not self._fitted or not self._bijectors:
            return np.zeros(n_samples)
        rng = np.random.RandomState(self.seed)
        d = len(self._bijectors[0]["shift"])
        z = rng.normal(0, 1, (n_samples, d))
        for bij in reversed(self._bijectors):
            z = (z - bij["shift"]) / np.exp(bij["log_scale"])
        z = z * bij["data_scale"] + bij["data_shift"]
        return z

    def log_prob(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted or not self._bijectors:
            return np.zeros(len(data))
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data_shift = self._bijectors[0]["data_shift"]
        data_scale = self._bijectors[0]["data_scale"]
        z = (data - data_shift) / data_scale
        for bij in self._bijectors:
            z = z * np.exp(bij["log_scale"]) + bij["shift"]
        log_prob = -0.5 * np.sum(z ** 2, axis=1) - 0.5 * z.shape[1] * np.log(2 * np.pi)
        for bij in self._bijectors:
            log_prob += np.sum(bij["log_scale"])
        return log_prob