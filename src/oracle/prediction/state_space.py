from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class KalmanSmoothResult:
    filtered_means: list
    smoothed_means: list
    filtered_covs: list
    smoothed_covs: list
    log_likelihood: float
    n_steps: int
    method: str


@dataclass
class StateSpaceResult:
    forecast: list
    forecast_lower: list
    forecast_upper: list
    components: dict
    log_likelihood: float
    n_parameters: int
    method: str


class StateSpaceModel:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._fitted = False
        self._params = None

    def fit(self, y: np.ndarray, trend: bool = True, seasonal: bool = True,
            seasonal_period: int = 12) -> "StateSpaceModel":
        n = len(y)
        self._params = {
            "level_var": float(np.var(np.diff(y)) * 0.1) if n > 1 else 0.01,
            "trend_var": float(np.var(np.diff(y, 2)) * 0.01) if n > 2 else 0.001,
            "seasonal_var": 0.001,
            "obs_var": float(np.var(y)) if n > 0 else 1.0,
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_period": seasonal_period,
        }
        self._fitted = True
        self._y = y
        return self

    def smooth(self, y: Optional[np.ndarray] = None) -> KalmanSmoothResult:
        if not self._fitted:
            return KalmanSmoothResult(filtered_means=[], smoothed_means=[], filtered_covs=[],
                                       smoothed_covs=[], log_likelihood=0.0, n_steps=0, method="kalman_smoother")
        if y is None:
            y = self._y
        n = len(y)
        p = self._params
        h = p["obs_var"]
        q_level = p["level_var"]
        q_trend = p["trend_var"] if p["trend"] else 0.0
        n_states = 2 + (p["seasonal_period"] - 1 if p["seasonal"] else 0)
        F = np.eye(n_states)
        if p["trend"]:
            F[0, 1] = 1.0
        if p["seasonal"] and n_states > 2:
            s = p["seasonal_period"]
            for j in range(3, n_states):
                F[j, j - 1] = 1.0
            F[2, 2:n_states] = [-1.0] * (n_states - 3) + [0.0] if n_states > 3 else [-1.0]
        Q = np.zeros((n_states, n_states))
        Q[0, 0] = q_level
        Q[1, 1] = q_trend
        if p["seasonal"] and n_states > 2:
            Q[2, 2] = p["seasonal_var"]
        H = np.zeros((1, n_states))
        H[0, 0] = 1.0
        if p["seasonal"] and n_states > 2:
            H[0, 2] = 1.0
        a = np.zeros(n_states)
        a[0] = float(y[0]) if n > 0 else 0.0
        P = np.eye(n_states) * 1.0
        filtered_means = []
        filtered_covs = []
        log_likelihood = 0.0
        for t in range(n):
            a_pred = F @ a
            P_pred = F @ P @ F.T + Q
            y_pred = float((H @ a_pred).item())
            S = float((H @ P_pred @ H.T).item()) + h
            if S < 1e-10:
                S = 1e-10
            K = (P_pred @ H.T) / S
            innovation = float(y[t]) - y_pred
            a = a_pred + K.flatten() * innovation
            P = P_pred - K @ H @ P_pred
            filtered_means.append(a.copy().tolist())
            filtered_covs.append(float(P[0, 0]))
            log_likelihood += -0.5 * np.log(2 * np.pi * S) - 0.5 * innovation ** 2 / S
        smoothed_means_list = [np.array(m) if not isinstance(m, np.ndarray) else m for m in filtered_means]
        smoothed_covs_list = list(filtered_covs)
        for t in range(n - 2, -1, -1):
            a_t = np.array(filtered_means[t])
            a_pred = F @ a_t
            P_t = np.eye(n_states)
            P_t[0, 0] = filtered_covs[t]
            P_pred = F @ P_t @ F.T + Q
            if P_pred[0, 0] < 1e-10:
                P_pred[0, 0] = 1e-10
            G = P_t @ F.T @ np.linalg.inv(P_pred)
            a_smooth = a_t + G @ (np.array(smoothed_means_list[t + 1]) - a_pred)
            smoothed_means_list[t] = a_smooth
            smoothed_covs_list[t] = float(P_t[0, 0] + G[0, 0] * (smoothed_covs_list[t + 1] - P_pred[0, 0]))
        smoothed_means = [m.tolist() if hasattr(m, 'tolist') else m for m in smoothed_means_list]
        smoothed_covs = smoothed_covs_list
        return KalmanSmoothResult(
            filtered_means=filtered_means,
            smoothed_means=smoothed_means,
            filtered_covs=filtered_covs,
            smoothed_covs=smoothed_covs,
            log_likelihood=float(log_likelihood),
            n_steps=n,
            method="kalman_smoother",
        )

    def forecast(self, n_ahead: int = 10, confidence: float = 0.95) -> StateSpaceResult:
        if not self._fitted:
            return StateSpaceResult(forecast=[], forecast_lower=[], forecast_upper=[],
                                     components={}, log_likelihood=0.0, n_parameters=0, method="state_space")
        smooth_result = self.smooth()
        n = len(self._y)
        p = self._params
        h = p["obs_var"]
        n_states = len(smooth_result.filtered_means[-1]) if smooth_result.filtered_means else 1
        F = np.eye(n_states)
        if p["trend"] and n_states > 1:
            F[0, 1] = 1.0
        Q = np.zeros((n_states, n_states))
        Q[0, 0] = p["level_var"]
        if n_states > 1:
            Q[1, 1] = p["trend_var"] if p["trend"] else 0.0
        H = np.zeros((1, n_states))
        H[0, 0] = 1.0
        if p["seasonal"] and n_states > 2:
            H[0, 2] = 1.0
        a = np.array(smooth_result.filtered_means[-1]) if smooth_result.filtered_means else np.zeros(n_states)
        P = np.eye(n_states) * (smooth_result.filtered_covs[-1] if smooth_result.filtered_covs else 1.0)
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        forecasts = []
        lower = []
        upper = []
        level_component = []
        trend_component = []
        for t in range(n_ahead):
            a = F @ a
            P = F @ P @ F.T + Q
            y_pred = float((H @ a).item())
            y_var = float((H @ P @ H.T).item()) + h
            forecasts.append(float(y_pred))
            lower.append(float(y_pred - z * np.sqrt(y_var)))
            upper.append(float(y_pred + z * np.sqrt(y_var)))
            level_component.append(float(a[0]))
            if n_states > 1:
                trend_component.append(float(a[1]))
        components = {"level": level_component}
        if p["trend"] and trend_component:
            components["trend"] = trend_component
        n_params = n_states * (n_states + 1) // 2 + 1
        return StateSpaceResult(
            forecast=forecasts,
            forecast_lower=lower,
            forecast_upper=upper,
            components=components,
            log_likelihood=smooth_result.log_likelihood,
            n_parameters=n_params,
            method="state_space",
        )

    @staticmethod
    def arima_state_space(y: np.ndarray, p: int = 1, d: int = 1, q: int = 1,
                          n_ahead: int = 10) -> dict:
        from scipy import stats as sp_stats
        if d > 0:
            for _ in range(d):
                y = np.diff(y)
        n = len(y)
        if n < 10:
            return {"forecast": [], "method": "arima_state_space"}
        y_mean = float(np.mean(y))
        y_centered = y - y_mean
        ar_coef = 0.5
        ma_coef = 0.3
        if p > 0:
            ar_coef = float(np.corrcoef(y_centered[:-1], y_centered[1:])[0, 1]) if np.std(y_centered) > 1e-10 else 0.0
        resid = y_centered.copy()
        if p > 0:
            resid = y_centered[1:] - ar_coef * y_centered[:-1]
        sigma2 = float(np.var(resid)) if len(resid) > 0 else 1.0
        forecasts = []
        last_val = float(y[-1])
        for t in range(n_ahead):
            next_val = y_mean + ar_coef * (last_val - y_mean) + np.sqrt(sigma2) * 0
            forecasts.append(float(next_val))
            last_val = next_val
        return {
            "forecast": forecasts,
            "ar_coefficient": ar_coef,
            "ma_coefficient": ma_coef,
            "residual_variance": sigma2,
            "order": [p, d, q],
            "method": "arima_state_space",
        }