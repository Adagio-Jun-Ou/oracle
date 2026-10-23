import numpy as np
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional
from scipy import stats as sp_stats
from oracle.core.types import Domain, Prediction, ConfidenceLevel


class StatisticalDeterministicModel:
    def __init__(self):
        self.methods = {
            "linear_regression": self._linear_regression,
            "polynomial_fit": self._polynomial_fit,
            "exponential_smoothing": self._exponential_smoothing,
            "arima_approx": self._arima_approx,
            "kalman_filter": self._kalman_filter,
        }

    def _linear_regression(self, y: np.ndarray, n_ahead: int, **kwargs) -> dict:
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
        future_x = np.arange(len(y), len(y) + n_ahead)
        predictions = slope * future_x + intercept
        residual_std = np.std(y - (slope * x + intercept))
        ci_upper = predictions + 1.96 * residual_std * np.sqrt(1 + 1 / len(y) + (future_x - np.mean(x)) ** 2 / (np.sum((x - np.mean(x)) ** 2) + 1e-10))
        ci_lower = predictions - 1.96 * residual_std * np.sqrt(1 + 1 / len(y) + (future_x - np.mean(x)) ** 2 / (np.sum((x - np.mean(x)) ** 2) + 1e-10))
        return {
            "predictions": predictions, "ci_upper": ci_upper, "ci_lower": ci_lower,
            "r_squared": r_value ** 2, "p_value": p_value, "slope": slope, "intercept": intercept,
        }

    def _polynomial_fit(self, y: np.ndarray, n_ahead: int, degree: int = 3, **kwargs) -> dict:
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, min(degree, len(y) - 1))
        future_x = np.arange(len(y), len(y) + n_ahead)
        predictions = np.polyval(coeffs, future_x)
        fitted = np.polyval(coeffs, x)
        residual_std = np.std(y - fitted)
        return {
            "predictions": predictions,
            "ci_upper": predictions + 1.96 * residual_std,
            "ci_lower": predictions - 1.96 * residual_std,
            "coefficients": coeffs, "residual_std": residual_std,
        }

    def _exponential_smoothing(self, y: np.ndarray, n_ahead: int, alpha: float = 0.3, beta: float = 0.1, **kwargs) -> dict:
        n = len(y)
        level = np.zeros(n + n_ahead)
        trend = np.zeros(n + n_ahead)
        level[0] = y[0]
        trend[0] = y[1] - y[0] if n > 1 else 0
        for i in range(1, n):
            level[i] = alpha * y[i] + (1 - alpha) * (level[i - 1] + trend[i - 1])
            trend[i] = beta * (level[i] - level[i - 1]) + (1 - beta) * trend[i - 1]
        predictions = np.zeros(n_ahead)
        for i in range(n_ahead):
            level[n + i] = level[n - 1] + (i + 1) * trend[n - 1]
            predictions[i] = level[n + i]
        residuals = y - level[:n]
        residual_std = np.std(residuals)
        return {
            "predictions": predictions,
            "ci_upper": predictions + 1.96 * residual_std * np.sqrt(np.arange(1, n_ahead + 1)),
            "ci_lower": predictions - 1.96 * residual_std * np.sqrt(np.arange(1, n_ahead + 1)),
            "level": level[:n], "trend": trend[:n],
        }

    def _arima_approx(self, y: np.ndarray, n_ahead: int, ar_order: int = 2, **kwargs) -> dict:
        y_detrended = np.diff(y) if len(y) > 1 else y
        ar_order = min(ar_order, len(y_detrended) - 1)
        if ar_order < 1:
            ar_order = 1
        X = np.zeros((len(y_detrended) - ar_order, ar_order))
        y_target = y_detrended[ar_order:]
        for i in range(len(y_target)):
            X[i] = y_detrended[i:i + ar_order]
        coeffs, _, _, _ = np.linalg.lstsq(X, y_target, rcond=None)
        predictions = np.zeros(n_ahead)
        history = list(y_detrended[-ar_order:])
        for i in range(n_ahead):
            pred = np.dot(history[-ar_order:], coeffs)
            predictions[i] = pred
            history.append(pred)
        predictions = np.cumsum(predictions) + y[-1]
        residual_std = float(np.std(y_target - X @ coeffs))
        return {
            "predictions": predictions,
            "ci_upper": predictions + 1.96 * residual_std * np.sqrt(np.arange(1, n_ahead + 1)),
            "ci_lower": predictions - 1.96 * residual_std * np.sqrt(np.arange(1, n_ahead + 1)),
            "ar_coefficients": coeffs, "residual_std": residual_std,
        }

    def _kalman_filter(self, y: np.ndarray, n_ahead: int, process_noise: float = 0.01, measurement_noise: float = 0.1, **kwargs) -> dict:
        n = len(y)
        x_est = np.zeros(n)
        P = np.zeros(n)
        x_est[0] = y[0]
        P[0] = 1.0
        Q = process_noise
        R = measurement_noise

        for i in range(1, n):
            x_pred = x_est[i - 1]
            P_pred = P[i - 1] + Q
            K = P_pred / (P_pred + R)
            x_est[i] = x_pred + K * (y[i] - x_pred)
            P[i] = (1 - K) * P_pred

        predictions = np.full(n_ahead, x_est[-1])
        pred_var = np.full(n_ahead, P[-1])
        for i in range(n_ahead):
            pred_var[i] += i * Q

        return {
            "predictions": predictions,
            "ci_upper": predictions + 1.96 * np.sqrt(pred_var),
            "ci_lower": predictions - 1.96 * np.sqrt(pred_var),
            "filtered": x_est, "variance": P,
        }

    def ensemble_predict(
        self,
        y: np.ndarray,
        n_ahead: int = 50,
        weights: Optional[dict] = None,
    ) -> dict:
        default_weights = {
            "linear_regression": 0.2, "polynomial_fit": 0.15,
            "exponential_smoothing": 0.25, "arima_approx": 0.25,
            "kalman_filter": 0.15,
        }
        w = weights or default_weights
        results = {}
        for name, method in self.methods.items():
            try:
                results[name] = method(y, n_ahead)
            except Exception:
                continue

        n_methods = len(results)
        if n_methods == 0:
            return self._linear_regression(y, n_ahead)

        pred_sum = np.zeros(n_ahead)
        ci_upper_sum = np.zeros(n_ahead)
        ci_lower_sum = np.zeros(n_ahead)
        total_weight = 0.0

        for name, result in results.items():
            weight = w.get(name, 1.0 / n_methods)
            pred_sum += weight * result["predictions"]
            ci_upper_sum += weight * result["ci_upper"]
            ci_lower_sum += weight * result["ci_lower"]
            total_weight += weight

        predictions = pred_sum / total_weight
        ci_upper = ci_upper_sum / total_weight
        ci_lower = ci_lower_sum / total_weight

        return {
            "predictions": predictions,
            "ci_upper": ci_upper,
            "ci_lower": ci_lower,
            "individual_results": results,
            "method_weights": {n: w.get(n, 1.0 / n_methods) for n in results},
        }

    def predict(
        self,
        y: np.ndarray,
        variable: str,
        domain: Domain,
        target_date: date,
        confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH,
        weights: Optional[dict] = None,
    ) -> Prediction:
        n_ahead = max(1, min((target_date - date.today()).days, 3650))
        result = self.ensemble_predict(y, n_ahead, weights)

        z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645, ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence_level]
        mean_val = float(result["predictions"][-1])
        std_val = float((result["ci_upper"][-1] - result["ci_lower"][-1]) / (2 * 1.96)) * z / 1.96

        return Prediction(
            domain=domain, variable=variable, value=mean_val,
            confidence_interval=(mean_val - z * std_val, mean_val + z * std_val),
            confidence_level=confidence_level, timestamp=datetime.now(),
            target_date=target_date, model_name="statistical_deterministic",
            weights=result.get("method_weights", {}),
        )