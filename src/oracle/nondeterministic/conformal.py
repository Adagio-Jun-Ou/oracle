from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy import stats


@dataclass
class ConformalResult:
    prediction: float
    lower: float
    upper: float
    width: float
    coverage: float
    p_value: float
    conformity_scores: list = field(default_factory=list)
    method: str = "split_conformal"


class ConformalPredictor:
    def __init__(self, alpha: float = 0.1, method: str = "split"):
        self.alpha = alpha
        self.method = method
        self.calibration_scores = np.array([])
        self._fitted = False

    @staticmethod
    def _absolute_conformity(residuals: np.ndarray) -> np.ndarray:
        return np.abs(residuals)

    @staticmethod
    def _normalized_conformity(residuals: np.ndarray, uncertainties: np.ndarray) -> np.ndarray:
        denom = np.maximum(uncertainties, 1e-8)
        return np.abs(residuals) / denom

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray,
            uncertainties: Optional[np.ndarray] = None) -> "ConformalPredictor":
        if uncertainties is not None:
            self.calibration_scores = self._normalized_conformity(
                y_true - y_pred, uncertainties
            )
            self.method = "normalized"
        else:
            self.calibration_scores = self._absolute_conformity(y_true - y_pred)
            self.method = "split"
        self._fitted = True
        return self

    def predict(self, y_pred: float, uncertainty: Optional[float] = None,
                n_calib: Optional[int] = None) -> ConformalResult:
        if not self._fitted:
            raise ValueError("ConformalPredictor must be fitted before prediction")
        scores = self.calibration_scores
        n = len(scores)
        level = min(1 - self.alpha, 1.0)
        q = np.quantile(scores, level, method="higher")
        if self.method == "normalized" and uncertainty is not None:
            lower = y_pred - q * uncertainty
            upper = y_pred + q * uncertainty
            width = 2 * q * uncertainty
        else:
            lower = y_pred - q
            upper = y_pred + q
            width = 2 * q
        p_value = float(np.mean(scores >= abs(y_pred - (lower + upper) / 2 + 1e-10)))
        return ConformalResult(
            prediction=y_pred,
            lower=float(lower),
            upper=float(upper),
            width=float(width),
            coverage=1.0 - self.alpha,
            p_value=float(p_value),
            conformity_scores=scores.tolist()[:100],
            method=self.method,
        )

    def predict_batch(self, y_preds: np.ndarray,
                      uncertainties: Optional[np.ndarray] = None) -> list[ConformalResult]:
        if not self._fitted:
            raise ValueError("ConformalPredictor must be fitted before prediction")
        results = []
        scores = self.calibration_scores
        n = len(scores)
        level = min(1 - self.alpha, 1.0)
        q = np.quantile(scores, level, method="higher")
        for i, yp in enumerate(y_preds):
            if self.method == "normalized" and uncertainties is not None:
                u = uncertainties[i] if i < len(uncertainties) else 1.0
                lower = yp - q * u
                upper = yp + q * u
                width = 2 * q * u
            else:
                lower = yp - q
                upper = yp + q
                width = 2 * q
            results.append(ConformalResult(
                prediction=float(yp),
                lower=float(lower),
                upper=float(upper),
                width=float(width),
                coverage=1.0 - self.alpha,
                p_value=float(np.mean(scores >= abs(yp - (lower + upper) / 2 + 1e-10))),
                conformity_scores=scores.tolist()[:50],
                method=self.method,
            ))
        return results

    def jackknife_plus(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_new: float) -> ConformalResult:
        n = len(y_true)
        scores = np.abs(y_true - y_pred)
        q = np.quantile(scores, min(1 - self.alpha, 1.0), method="higher")
        loos = np.array([np.delete(scores, i) for i in range(n)])
        quantiles = np.array([np.quantile(l, min(1 - self.alpha, 1.0)) for l in loos])
        lower = float(max(y_new - np.max(quantiles), y_new - np.quantile(scores, 1 - self.alpha)))
        upper = float(min(y_new + np.max(quantiles), y_new + np.quantile(scores, 1 - self.alpha)))
        return ConformalResult(
            prediction=float(y_new),
            lower=lower,
            upper=upper,
            width=float(upper - lower),
            coverage=1.0 - self.alpha,
            p_value=float(np.mean(scores >= abs(y_new - (lower + upper) / 2 + 1e-10))),
            conformity_scores=scores.tolist()[:50],
            method="jackknife_plus",
        )

    @staticmethod
    def adaptive_conformal(y_true: np.ndarray, y_pred: np.ndarray,
                           alpha: float = 0.1, step_size: float = 0.01) -> dict:
        n = len(y_true)
        residuals = np.abs(y_true - y_pred)
        q = np.quantile(residuals, 1 - alpha)
        miscoverage_current = 1.0
        coverage_history = []
        q_history = [float(q)]
        for i in range(n):
            in_interval = abs(y_true[i] - y_pred[i]) <= q
            indicator = 1.0 if not in_interval else 0.0
            miscoverage_current = miscoverage_current + step_size * (indicator - miscoverage_current)
            if not in_interval:
                q = q * (1 + step_size)
            else:
                q = q * (1 - step_size * 0.1)
            q = max(q, 0.01)
            coverage_history.append(float(1.0 - miscoverage_current))
            q_history.append(float(q))
        return {
            "final_quantile": float(q),
            "final_coverage": coverage_history[-1] if coverage_history else 0.0,
            "coverage_history": coverage_history,
            "quantile_history": q_history,
            "target_coverage": 1.0 - alpha,
            "method": "adaptive_conformal",
        }

    @staticmethod
    def conformal_bands(time_series: np.ndarray, window: int = 21,
                        alpha: float = 0.1) -> dict:
        n = len(time_series)
        if n < 2 * window:
            return {"lower": [], "upper": [], "coverage": 0.0, "method": "conformal_bands"}
        bands_lower = []
        bands_upper = []
        for i in range(window, n):
            historical = time_series[max(0, i - window):i]
            rolling_mean = float(np.mean(historical))
            rolling_std = float(np.std(historical)) + 1e-8
            residuals = np.abs(historical - rolling_mean) / rolling_std
            q = float(np.quantile(residuals, 1 - alpha)) if len(residuals) > 0 else 2.0
            bands_lower.append(rolling_mean - q * rolling_std)
            bands_upper.append(rolling_mean + q * rolling_std)
        actual_in_window = time_series[window:]
        in_band = 0
        for i in range(min(len(actual_in_window), len(bands_lower))):
            if float(bands_lower[i]) <= float(actual_in_window[i]) <= float(bands_upper[i]):
                in_band += 1
        n_actual = len(actual_in_window) if len(actual_in_window) > 0 else 1
        coverage = in_band / n_actual
        return {
            "lower": bands_lower,
            "upper": bands_upper,
            "coverage": float(coverage),
            "mean_width": float(np.mean([u - l for l, u in zip(bands_lower, bands_upper)])),
            "method": "conformal_bands",
        }