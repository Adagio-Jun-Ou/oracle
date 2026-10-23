from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy import stats


@dataclass
class BSTSResult:
    trend: list
    seasonal: list
    cycle: list
    residual: list
    trend_forecast: list
    seasonal_forecast: list
    forecast: list
    forecast_lower_50: list
    forecast_upper_50: list
    forecast_lower_95: list
    forecast_upper_95: list
    r_squared: float
    trend_contribution: float
    seasonal_contribution: float
    cycle_contribution: float
    residual_variance: float
    changepoints: list
    state_contributions: dict
    method: str = "bsts"


class BayesianStructuralTimeSeries:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def decompose(self, series: np.ndarray, period: Optional[int] = None,
                  n_forecast: int = 10, seasonal_harmonics: int = 3) -> BSTSResult:
        n = len(series)
        if n < 4:
            return BSTSResult(
                trend=series.tolist(), seasonal=[0.0]*n, cycle=[0.0]*n,
                residual=[0.0]*n, trend_forecast=[0.0]*n_forecast,
                seasonal_forecast=[0.0]*n_forecast, forecast=[0.0]*n_forecast,
                forecast_lower_50=[0.0]*n_forecast, forecast_upper_50=[0.0]*n_forecast,
                forecast_lower_95=[0.0]*n_forecast, forecast_upper_95=[0.0]*n_forecast,
                r_squared=0.0, trend_contribution=0.0, seasonal_contribution=0.0,
                cycle_contribution=0.0, residual_variance=0.0, changepoints=[],
                state_contributions={},
            )
        if period is None:
            period = self._detect_period(series)
        trend = self._extract_trend(series)
        detrended = series - trend
        seasonal = self._extract_seasonal(detrended, period if period > 1 else 1)
        deseasonalized = detrended - seasonal
        cycle = self._extract_cycle(deseasonalized)
        residual = series - trend - seasonal - cycle
        trend_forecast = self._forecast_trend(trend, n_forecast)
        seasonal_forecast = self._forecast_seasonal(seasonal, period, n_forecast)
        cycle_forecast = self._forecast_cycle(cycle, n_forecast)
        forecast_vals = [float(t + s + c) for t, s, c in zip(trend_forecast, seasonal_forecast, cycle_forecast)]
        residual_std = float(np.std(residual)) if len(residual) > 0 else 1.0
        forecast_upper_50 = [f + 0.674 * residual_std for f in forecast_vals]
        forecast_lower_50 = [f - 0.674 * residual_std for f in forecast_vals]
        forecast_upper_95 = [f + 1.96 * residual_std for f in forecast_vals]
        forecast_lower_95 = [f - 1.96 * residual_std for f in forecast_vals]
        total_var = float(np.var(series)) if np.var(series) > 0 else 1.0
        trend_contribution = float(np.var(trend) / total_var) if total_var > 0 else 0.0
        seasonal_contribution = float(np.var(seasonal) / total_var) if total_var > 0 else 0.0
        cycle_contribution = float(np.var(cycle) / total_var) if total_var > 0 else 0.0
        ss_total = float(np.sum((series - np.mean(series)) ** 2))
        ss_resid = float(np.sum(residual ** 2))
        r_squared = 1.0 - ss_resid / ss_total if ss_total > 0 else 0.0
        changepoints = self._detect_changepoints(trend, residual)
        state_contributions = {
            "trend_pct": float(np.clip(trend_contribution * 100, 0, 100)),
            "seasonal_pct": float(np.clip(seasonal_contribution * 100, 0, 100)),
            "cycle_pct": float(np.clip(cycle_contribution * 100, 0, 100)),
            "residual_pct": float(np.clip((1 - trend_contribution - seasonal_contribution - cycle_contribution) * 100, 0, 100)),
        }
        return BSTSResult(
            trend=trend.tolist(), seasonal=seasonal.tolist(), cycle=cycle.tolist(),
            residual=residual.tolist(), trend_forecast=trend_forecast,
            seasonal_forecast=seasonal_forecast, forecast=forecast_vals,
            forecast_lower_50=forecast_lower_50,
            forecast_upper_50=forecast_upper_50,
            forecast_lower_95=forecast_lower_95,
            forecast_upper_95=forecast_upper_95,
            r_squared=float(np.clip(r_squared, 0, 1)),
            trend_contribution=float(np.clip(trend_contribution, 0, 1)),
            seasonal_contribution=float(np.clip(seasonal_contribution, 0, 1)),
            cycle_contribution=float(np.clip(cycle_contribution, 0, 1)),
            residual_variance=float(residual_std ** 2),
            changepoints=changepoints,
            state_contributions=state_contributions,
        )

    def _detect_period(self, series: np.ndarray) -> int:
        n = len(series)
        if n < 4:
            return 1
        series_demeaned = series - np.mean(series)
        best_period = 1
        best_autocorr = 0.0
        for p in range(2, min(n // 2 + 1, 53)):
            if p >= n:
                continue
            corr = np.corrcoef(series_demeaned[p:], series_demeaned[:-p])[0, 1] if n - p > 1 else 0.0
            if abs(corr) > best_autocorr:
                best_autocorr = abs(corr)
                best_period = p
        return best_period

    @staticmethod
    def _extract_trend(series: np.ndarray, window: int = None) -> np.ndarray:
        n = len(series)
        if window is None:
            window = max(3, n // 10)
            if window % 2 == 0:
                window += 1
        half = window // 2
        trend = np.zeros(n)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            weights = np.exp(-0.5 * ((np.arange(hi - lo) - (i - lo)) / max(half, 1)) ** 2)
            trend[i] = np.sum(series[lo:hi] * weights) / np.sum(weights)
        return trend

    @staticmethod
    def _extract_seasonal(detrended: np.ndarray, period: int) -> np.ndarray:
        n = len(detrended)
        if period < 2 or n < period:
            return np.zeros(n)
        seasonal = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            if len(indices) > 0:
                seasonal_mean = np.mean(detrended[indices])
                seasonal[indices] = seasonal_mean
        seasonal -= np.mean(seasonal)
        return seasonal

    @staticmethod
    def _extract_cycle(deseasonalized: np.ndarray) -> np.ndarray:
        n = len(deseasonalized)
        if n < 4:
            return np.zeros(n)
        smoothed = np.convolve(deseasonalized, np.ones(3) / 3, mode='same')
        cycle = deseasonalized - smoothed
        return cycle

    def _forecast_trend(self, trend: np.ndarray, n_forecast: int) -> list:
        n = len(trend)
        if n < 2:
            last = trend[-1] if n > 0 else 0
            return [float(last)] * n_forecast
        recent = trend[-min(20, n):]
        if len(recent) >= 2:
            slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
        else:
            slope = 0.0
        last_val = float(trend[-1])
        forecasts = []
        for i in range(1, n_forecast + 1):
            forecasts.append(last_val + slope * i)
        return forecasts

    @staticmethod
    def _forecast_seasonal(seasonal: np.ndarray, period: int, n_forecast: int) -> list:
        n = len(seasonal)
        if period < 2 or n < period:
            return [0.0] * n_forecast
        last_cycle = seasonal[-period:]
        forecasts = []
        for i in range(n_forecast):
            forecasts.append(float(last_cycle[i % period]))
        return forecasts

    @staticmethod
    def _forecast_cycle(cycle: np.ndarray, n_forecast: int) -> list:
        n = len(cycle)
        if n < 4:
            return [0.0] * n_forecast
        last_vals = cycle[-min(5, n):]
        decay = 0.85
        forecasts = []
        recent_mean = float(np.mean(last_vals))
        for i in range(n_forecast):
            forecasts.append(float(recent_mean * decay ** (i + 1)))
        return forecasts

    @staticmethod
    def _detect_changepoints(trend: np.ndarray, residual: np.ndarray,
                             threshold: float = 2.0) -> list:
        n = len(trend)
        if n < 5:
            return []
        changes = []
        for i in range(2, n - 2):
            window_before = trend[max(0, i - 3):i]
            window_after = trend[i:min(n, i + 3)]
            if len(window_before) >= 2 and len(window_after) >= 2:
                mean_before = float(np.mean(window_before))
                mean_after = float(np.mean(window_after))
                std_residual = float(np.std(residual[max(0, i - 5):min(n, i + 5)])) if i >= 5 else 1.0
                if std_residual < 1e-10:
                    std_residual = 1.0
                change_magnitude = abs(mean_after - mean_before) / std_residual
                if change_magnitude > threshold:
                    changes.append({
                        "index": i,
                        "magnitude": float(change_magnitude),
                        "direction": "up" if mean_after > mean_before else "down",
                    })
        merged = []
        for c in changes:
            if not merged or c["index"] - merged[-1]["index"] > 3:
                merged.append(c)
            else:
                if c["magnitude"] > merged[-1]["magnitude"]:
                    merged[-1] = c
        return merged

    def nowcast(self, series: np.ndarray, partial_current: np.ndarray,
                n_ahead: int = 1) -> dict:
        n = len(series)
        if n < 10:
            return {"nowcast": float(series[-1]) if n > 0 else 0.0, "method": "last_observation"}
        decomp = self.decompose(series, n_forecast=max(n_ahead, 3))
        trend_now = decomp.trend[-1] if decomp.trend else 0.0
        residual_std = float(np.sqrt(decomp.residual_variance)) if decomp.residual_variance > 0 else 1.0
        if len(partial_current) > 0:
            partial_signal = float(np.mean(partial_current))
            nowcast_val = trend_now + partial_signal * 0.3
        else:
            nowcast_val = trend_now
        return {
            "nowcast": float(nowcast_val),
            "nowcast_lower_50": float(nowcast_val - 0.674 * residual_std),
            "nowcast_upper_50": float(nowcast_val + 0.674 * residual_std),
            "nowcast_lower_95": float(nowcast_val - 1.96 * residual_std),
            "nowcast_upper_95": float(nowcast_val + 1.96 * residual_std),
            "trend_component": float(trend_now),
            "residual_std": float(residual_std),
            "state_contributions": decomp.state_contributions,
            "method": "bsts_nowcast",
        }

    @staticmethod
    def bayesian_model_average(forecasts: list[dict], bics: list[float]) -> dict:
        if not forecasts or not bics:
            return {"point": 0.0, "lower": 0.0, "upper": 0.0, "weights": []}
        min_bic = min(bics)
        delta_bics = [bic - min_bic for bic in bics]
        weights = [np.exp(-0.5 * d) for d in delta_bics]
        total_w = sum(weights) + 1e-10
        weights = [w / total_w for w in weights]
        point = sum(w * f.get("point", f.get("forecast", 0.0)) for w, f in zip(weights, forecasts))
        lower = sum(w * f.get("lower", f.get("forecast", 0.0) - 1.96) for w, f in zip(weights, forecasts))
        upper = sum(w * f.get("upper", f.get("forecast", 0.0) + 1.96) for w, f in zip(weights, forecasts))
        std = sum(w * f.get("std", 1.0) for w, f in zip(weights, forecasts))
        return {
            "point": float(point),
            "lower_50": float(point - 0.674 * std),
            "upper_50": float(point + 0.674 * std),
            "lower_95": float(point - 1.96 * std),
            "upper_95": float(point + 1.96 * std),
            "weights": [{"model": f.get("model", f"model_{i}"), "weight": float(w), "bic": float(b)}
                        for i, (w, f, b) in enumerate(zip(weights, forecasts, bics))],
            "best_model": forecasts[weights.index(max(weights))].get("model", "model_0") if forecasts else "unknown",
        }