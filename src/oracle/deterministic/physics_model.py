import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from oracle.core.types import Domain, Prediction, ConfidenceLevel


@dataclass
class PhysicsParams:
    damping: float = 0.1
    spring_constant: float = 1.0
    mass: float = 1.0
    noise_intensity: float = 0.01
    coupling_strength: dict = field(default_factory=lambda: {
        "gdp_employment": 0.4,
        "inflation_rates": -0.3,
        "debt_growth": -0.2,
        "trade_coupling": 0.15,
    })


class PhysicsModel:
    def __init__(self, params: Optional[PhysicsParams] = None):
        self.params = params or PhysicsParams()

    def coupled_oscillator(
        self,
        initial_states: list[tuple[float, float]],
        n_steps: int = 1000,
        dt: float = 0.01,
        coupling_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = len(initial_states)
        positions = np.array([s[0] for s in initial_states])
        velocities = np.array([s[1] for s in initial_states])

        if coupling_matrix is None:
            coupling_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    coupling_matrix[i, j] = 0.1
                    coupling_matrix[j, i] = 0.1

        trajectory = np.zeros((n_steps, n))
        rng = np.random.default_rng(42)

        for t in range(n_steps):
            force = -self.params.spring_constant * positions
            force -= self.params.damping * velocities
            coupling_force = coupling_matrix @ positions
            force += coupling_force
            force += rng.normal(0, self.params.noise_intensity, n) * np.sqrt(dt)
            velocities += force / self.params.mass * dt
            positions += velocities * dt
            trajectory[t] = positions.copy()

        return trajectory

    def langevin_dynamics(
        self,
        potential: callable,
        initial_position: float = 0.0,
        n_steps: int = 10000,
        dt: float = 0.001,
        temperature: float = 1.0,
    ) -> np.ndarray:
        position = initial_position
        trajectory = np.zeros(n_steps)
        rng = np.random.default_rng(42)

        for i in range(n_steps):
            noise = rng.normal(0, np.sqrt(2 * temperature * dt))
            force = -potential(position)
            position += force * dt + noise
            trajectory[i] = position

        return trajectory

    def renormalization_group(
        self,
        data: np.ndarray,
        n_levels: int = 5,
        coarse_graining_factor: float = 2.0,
    ) -> list[np.ndarray]:
        levels = [data.copy()]
        current = data.copy()

        for level in range(n_levels):
            window = int(coarse_graining_factor ** level)
            if window >= len(current):
                break
            n_blocks = len(current) // window
            remainder = len(current) % window
            blocks = current[:n_blocks * window].reshape(n_blocks, window)
            coarse = np.array([np.mean(blocks, axis=1), np.std(blocks, axis=1), np.max(blocks, axis=1) - np.min(blocks, axis=1)])
            levels.append(current[::max(1, window)])
            current = np.convolve(current, np.ones(window) / window, mode="valid")

        return levels

    def resonance_detection(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray,
        freq_range: tuple[float, float] = (0.01, 0.5),
        n_freqs: int = 100,
    ) -> dict:
        freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
        spectrum_a = np.abs(np.fft.rfft(series_a, n=len(series_a) * 4))
        spectrum_b = np.abs(np.fft.rfft(series_b, n=len(series_b) * 4))
        freq_axis = np.fft.rfftfreq(len(series_a) * 4, d=1.0)
        cross_spectrum = spectrum_a * np.conj(spectrum_b)
        coherence = np.abs(cross_spectrum) ** 2 / (np.abs(spectrum_a) ** 2 * np.abs(spectrum_b) ** 2 + 1e-10)
        resonance_freqs = freq_axis[np.argsort(coherence)[-5:]]
        resonance_strengths = np.sort(coherence)[-5:]

        return {
            "coherence": coherence,
            "cross_spectrum": cross_spectrum,
            "resonance_frequencies": resonance_freqs,
            "resonance_strengths": resonance_strengths,
            "dominant_frequency": freq_axis[np.argmax(coherence)],
            "max_coherence": float(np.max(coherence)),
        }

    def entropy_flow(
        self,
        series: np.ndarray,
        window: int = 50,
        n_bins: int = 20,
    ) -> dict:
        n = len(series)
        entropies = []
        for i in range(0, n - window, window // 2):
            segment = series[i:i + window]
            hist, _ = np.histogram(segment, bins=n_bins, density=True)
            hist = hist[hist > 0]
            shannon = -np.sum(hist * np.log2(hist))
            entropies.append(shannon)

        entropies = np.array(entropies)
        entropy_flow = np.diff(entropies) if len(entropies) > 1 else np.array([0])

        return {
            "entropy_series": entropies,
            "entropy_flow": entropy_flow,
            "mean_entropy": float(np.mean(entropies)),
            "entropy_trend": float(np.polyfit(range(len(entropies)), entropies, 1)[0]) if len(entropies) > 1 else 0.0,
            "entropy_volatility": float(np.std(entropy_flow)),
            "phase_transitions": self._detect_phase_transitions(entropy_flow),
        }

    def _detect_phase_transitions(self, entropy_flow: np.ndarray, threshold: float = 2.0) -> list[int]:
        if len(entropy_flow) < 3:
            return []
        z_scores = np.abs((entropy_flow - np.mean(entropy_flow)) / (np.std(entropy_flow) + 1e-10))
        return list(np.where(z_scores > threshold)[0])

    def critical_point_detection(
        self,
        series: np.ndarray,
        window: int = 20,
    ) -> dict:
        rolling_mean = np.convolve(series, np.ones(window) / window, mode="valid")
        rolling_var = np.array([
            np.var(series[i:i + window]) for i in range(len(series) - window + 1)
        ])
        if len(rolling_var) < 3:
            return {"critical_points": [], "variance_series": rolling_var}

        var_diff = np.diff(rolling_var)
        critical_idx = []
        for i in range(1, len(var_diff)):
            if var_diff[i - 1] * var_diff[i] < 0:
                critical_idx.append(i + window)

        return {
            "critical_points": critical_idx,
            "variance_series": rolling_var,
            "mean_variance": float(np.mean(rolling_var)),
            "variance_trend": float(np.polyfit(range(len(rolling_var)), rolling_var, 1)[0]),
        }

    def multi_scale_prediction(
        self,
        series: np.ndarray,
        n_ahead: int = 100,
        n_sims: int = 1000,
    ) -> dict:
        rng = np.random.default_rng(42)
        diff = np.diff(series)
        mu = np.mean(diff)
        sigma = np.std(diff)

        hurst = self._compute_hurst(series)

        simulations = np.zeros((n_sims, n_ahead))
        for sim in range(n_sims):
            path = [series[-1]]
            for step in range(n_ahead):
                increment = mu + sigma * rng.normal() * (step ** hurst)
                path.append(path[-1] + increment)
            simulations[sim] = path[1:]

        percentiles = {
            "p05": np.percentile(simulations, 5, axis=0),
            "p25": np.percentile(simulations, 25, axis=0),
            "p50": np.percentile(simulations, 50, axis=0),
            "p75": np.percentile(simulations, 75, axis=0),
            "p95": np.percentile(simulations, 95, axis=0),
        }
        confidence = np.clip(1.0 - 2 * np.std(simulations) / (np.mean(np.abs(simulations)) + 1e-10), 0.1, 0.99)

        return {
            "predictions": percentiles,
            "mean": np.mean(simulations, axis=0),
            "std": np.std(simulations, axis=0),
            "hurst_exponent": float(hurst),
            "confidence_at_horizon": float(confidence),
        }

    def _compute_hurst(self, series: np.ndarray, max_lag: int = 100) -> float:
        lags = range(2, min(max_lag, len(series) // 2))
        tau = []
        for lag in lags:
            diff = np.diff(series, n=lag)
            tau.append(np.std(diff))
        tau = np.array(tau)
        lags_arr = np.array(list(lags), dtype=float)
        valid = (tau > 0) & (lags_arr > 0)
        if np.sum(valid) < 2:
            return 0.5
        log_lags = np.log(lags_arr[valid])
        log_tau = np.log(tau[valid])
        poly = np.polyfit(log_lags, log_tau, 1)
        return float(np.clip(poly[0], 0.0, 1.0))

    def predict(
        self,
        series: np.ndarray,
        variable: str,
        domain: Domain,
        target_date: date,
        confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH,
    ) -> Prediction:
        n_ahead = max(10, min((target_date - date.today()).days, 3650))
        result = self.multi_scale_prediction(series, n_ahead=n_ahead)

        z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645, ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence_level]
        mean_val = float(result["mean"][-1])
        std_val = float(result["std"][-1])

        return Prediction(
            domain=domain,
            variable=variable,
            value=mean_val,
            confidence_interval=(mean_val - z * std_val, mean_val + z * std_val),
            confidence_level=confidence_level,
            timestamp=datetime.now(),
            target_date=target_date,
            model_name="physics_multi_scale",
            weights={"hurst_adjusted": result["hurst_exponent"], "confidence": result["confidence_at_horizon"]},
        )