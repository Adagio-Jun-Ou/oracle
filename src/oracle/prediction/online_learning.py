from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class OnlineLearningState:
    weights: list
    bias: float
    learning_rate: float
    forgetting_factor: float
    n_updates: int
    cumulative_loss: float
    recent_loss: float
    drift_detected: bool
    drift_magnitude: float
    stability_index: float


@dataclass
class DriftDetectionResult:
    drift_detected: bool
    drift_type: str
    drift_magnitude: float
    drift_start_index: int
    confidence: float
    window_statistic: float
    threshold: float


class OnlineLearner:
    def __init__(self, n_features: int = 1, learning_rate: float = 0.01,
                 forgetting_factor: float = 0.995, seed: int = 42):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.forgetting_factor = forgetting_factor
        self.seed = seed
        rng = np.random.RandomState(seed)
        self.weights = rng.normal(0, 0.01, n_features)
        self.bias = 0.0
        self.n_updates = 0
        self.cumulative_loss = 0.0
        self._recent_losses = []
        self._reference_dist = None
        self._reference_mean = None
        self._reference_std = None

    def partial_fit(self, x: np.ndarray, y: float) -> OnlineLearningState:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[0] == 1:
            x = x.flatten()
        pred = float(np.dot(self.weights, x) + self.bias)
        error = y - pred
        self.weights = self.weights + self.learning_rate * error * x * self.forgetting_factor ** self.n_updates
        self.bias = self.bias + self.learning_rate * error * 0.01
        loss = float(error ** 2)
        self._recent_losses.append(loss)
        if len(self._recent_losses) > 100:
            self._recent_losses = self._recent_losses[-100:]
        self.cumulative_loss += loss
        self.n_updates += 1
        drift_detected = self._check_drift(loss)
        recent_loss = float(np.mean(self._recent_losses[-20:])) if self._recent_losses else 0.0
        long_term = self.cumulative_loss / max(1, self.n_updates)
        drift_mag = abs(recent_loss - long_term) / (long_term + 1e-10)
        stability = float(1.0 / (1.0 + drift_mag))
        return OnlineLearningState(
            weights=self.weights.tolist(),
            bias=float(self.bias),
            learning_rate=self.learning_rate,
            forgetting_factor=self.forgetting_factor,
            n_updates=self.n_updates,
            cumulative_loss=float(self.cumulative_loss),
            recent_loss=recent_loss,
            drift_detected=drift_detected,
            drift_magnitude=float(drift_mag),
            stability_index=stability,
        )

    def _check_drift(self, loss: float) -> bool:
        if self.n_updates < 20:
            return False
        recent = self._recent_losses[-20:]
        long_term = self.cumulative_loss / max(1, self.n_updates)
        if long_term < 1e-10:
            return False
        ratio = float(np.mean(recent)) / long_term
        return ratio > 2.5

    def predict(self, x: np.ndarray) -> float:
        if x.ndim > 1:
            x = x.flatten()
        return float(np.dot(self.weights, x) + self.bias)

    def predict_batch(self, X: np.ndarray) -> list:
        return [self.predict(X[i]) for i in range(X.shape[0])]

    def get_state(self) -> OnlineLearningState:
        recent_loss = float(np.mean(self._recent_losses[-20:])) if self._recent_losses else 0.0
        long_term = self.cumulative_loss / max(1, self.n_updates)
        drift_mag = abs(recent_loss - long_term) / (long_term + 1e-10)
        return OnlineLearningState(
            weights=self.weights.tolist(),
            bias=float(self.bias),
            learning_rate=self.learning_rate,
            forgetting_factor=self.forgetting_factor,
            n_updates=self.n_updates,
            cumulative_loss=float(self.cumulative_loss),
            recent_loss=recent_loss,
            drift_detected=False,
            drift_magnitude=float(drift_mag),
            stability_index=float(1.0 / (1.0 + drift_mag)),
        )

    @staticmethod
    def adwin_drift_detection(series: np.ndarray, delta: float = 0.002) -> DriftDetectionResult:
        n = len(series)
        if n < 10:
            return DriftDetectionResult(
                drift_detected=False, drift_type="none", drift_magnitude=0.0,
                drift_start_index=0, confidence=0.0,
                window_statistic=0.0, threshold=0.0,
            )
        window = list(series[:10])
        drift_idx = 0
        for i in range(10, n):
            window.append(series[i])
            while len(window) > 2:
                n_w = len(window)
                for cut in range(1, n_w):
                    w0 = window[:cut]
                    w1 = window[cut:]
                    if len(w0) < 3 or len(w1) < 3:
                        continue
                    mu0 = np.mean(w0)
                    mu1 = np.mean(w1)
                    n0 = len(w0)
                    n1 = len(w1)
                    m = 1.0 / (1.0 / n0 + 1.0 / n1)
                    eps = np.sqrt(0.5 * np.log(2 * m / delta) / m)
                    if abs(mu0 - mu1) >= eps:
                        if mu1 > mu0:
                            window = w1
                            drift_idx = i - len(w1) + 1
                        else:
                            window = w0
                        break
                else:
                    break
        if drift_idx > 0:
            w_a = series[:drift_idx]
            w_b = series[drift_idx:]
            drift_mag = float(abs(np.mean(w_b) - np.mean(w_a)) / (np.std(series) + 1e-10)) if np.std(series) > 0 else 0.0
            drift_type = "abrupt" if drift_mag > 2.0 else "gradual"
            confidence = float(min(1.0, drift_mag / 3.0))
            return DriftDetectionResult(
                drift_detected=True,
                drift_type=drift_type,
                drift_magnitude=drift_mag,
                drift_start_index=drift_idx,
                confidence=confidence,
                window_statistic=float(abs(np.mean(w_b) - np.mean(w_a))),
                threshold=float(np.sqrt(0.5 * np.log(2 * len(series) / delta) / len(series))),
            )
        return DriftDetectionResult(
            drift_detected=False, drift_type="none", drift_magnitude=0.0,
            drift_start_index=0, confidence=0.0,
            window_statistic=0.0, threshold=0.0,
        )

    @staticmethod
    def page_hinkley_drift(series: np.ndarray, threshold: float = 50.0,
                            forgetting: float = 0.005) -> DriftDetectionResult:
        n = len(series)
        if n < 5:
            return DriftDetectionResult(
                drift_detected=False, drift_type="none", drift_magnitude=0.0,
                drift_start_index=0, confidence=0.0,
                window_statistic=0.0, threshold=threshold,
            )
        cumsum = 0.0
        cumsum_min = float('inf')
        cumsum_max = float('-inf')
        mean = float(np.mean(series))
        drift_idx = 0
        for i in range(n):
            cumsum += series[i] - mean - forgetting * cumsum
            cumsum_min = min(cumsum_min, cumsum)
            cumsum_max = max(cumsum_max, cumsum)
            if cumsum - cumsum_min > threshold or cumsum_max - cumsum > threshold:
                drift_idx = i
                break
        if drift_idx > 0:
            pre_mean = float(np.mean(series[:drift_idx]))
            post_mean = float(np.mean(series[drift_idx:])) if drift_idx < n else pre_mean
            drift_mag = float(abs(post_mean - pre_mean) / (np.std(series[:drift_idx]) + 1e-10))
            return DriftDetectionResult(
                drift_detected=True,
                drift_type="abrupt" if drift_mag > 3.0 else "gradual",
                drift_magnitude=drift_mag,
                drift_start_index=drift_idx,
                confidence=float(min(1.0, drift_mag / 2.0)),
                window_statistic=float(cumsum - cumsum_min),
                threshold=threshold,
            )
        return DriftDetectionResult(
            drift_detected=False, drift_type="none", drift_magnitude=0.0,
            drift_start_index=0, confidence=0.0,
            window_statistic=float(cumsum - cumsum_min),
            threshold=threshold,
        )

    @staticmethod
    def exponential_forgetting_forecast(series: np.ndarray, halflife: float = 10.0,
                                          n_ahead: int = 5) -> dict:
        n = len(series)
        if n < 3:
            return {"forecast": [float(series[-1])] * n_ahead if n > 0 else [0.0] * n_ahead,
                    "method": "exponential_forgetting", "halflife": halflife}
        decay = np.log(2) / halflife
        weights = np.exp(-decay * np.arange(n)[::-1])
        weights = weights / weights.sum()
        ema = float(np.sum(series * weights))
        residuals = series - ema
        weighted_var = float(np.sum(weights * residuals ** 2))
        std = float(np.sqrt(weighted_var))
        slope = float((series[-1] - series[max(0, int(n - halflife))]) / max(1, halflife))
        forecasts = []
        for i in range(1, n_ahead + 1):
            f = ema + slope * i
            forecasts.append(float(f))
        lower_95 = [float(f - 1.96 * std * np.sqrt(i + 1)) for i, f in enumerate(forecasts, 1)]
        upper_95 = [float(f + 1.96 * std * np.sqrt(i + 1)) for i, f in enumerate(forecasts, 1)]
        return {
            "forecast": forecasts,
            "forecast_lower_95": lower_95,
            "forecast_upper_95": upper_95,
            "ema": float(ema),
            "trend": float(slope),
            "volatility": float(std),
            "halflife": float(halflife),
            "effective_observations": float(min(n, 3 * halflife)),
            "method": "exponential_forgetting",
        }

    @staticmethod
    def kalman_filter_1d(series: np.ndarray, process_noise: float = 1.0,
                          measurement_noise: float = 4.0) -> dict:
        n = len(series)
        if n < 2:
            return {"filtered": series.tolist(), "forecast": [0.0] * 5, "method": "kalman_1d"}
        x_est = float(series[0])
        p_est = 1.0
        filtered, gains, innovations, p_history = [], [], [], []
        for i in range(n):
            p_pred = p_est + process_noise
            k = p_pred / (p_pred + measurement_noise)
            innovation = float(series[i] - x_est)
            x_est = x_est + k * innovation
            p_est = (1 - k) * p_pred
            filtered.append(float(x_est))
            gains.append(float(k))
            innovations.append(innovation)
            p_history.append(float(p_est))
        forecasts = [float(x_est)] * 5
        forecast_std = float(np.sqrt(p_est + process_noise))
        return {
            "filtered": filtered,
            "kalman_gains": gains,
            "innovations": innovations,
            "uncertainty": p_history,
            "forecast": forecasts,
            "forecast_lower_95": [float(f - 1.96 * forecast_std * np.sqrt(i + 1)) for i, f in enumerate(forecasts)],
            "forecast_upper_95": [float(f + 1.96 * forecast_std * np.sqrt(i + 1)) for i, f in enumerate(forecasts)],
            "final_estimate": float(x_est),
            "final_uncertainty": float(p_est),
            "innovation_std": float(np.std(innovations)) if len(innovations) > 1 else 1.0,
            "method": "kalman_1d",
        }