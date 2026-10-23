from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy import stats


@dataclass
class QuantileRegressionResult:
    quantile: float
    coefficients: list
    intercept: float
    predictions: list
    r_squared: float
    n_obs: int
    method: str = "pinball"


@dataclass
class HeteroscedasticInterval:
    lower: list
    upper: list
    width: list
    point_forecast: list
    coverage_probability: float
    mean_width: float
    quantile_crossing: bool = False


class QuantileRegression:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._fitted_quantiles = {}

    @staticmethod
    def _pinball_loss(y: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        error = y - y_pred
        return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))

    @staticmethod
    def _fit_quantile(X: np.ndarray, y: np.ndarray, quantile: float,
                       max_iter: int = 1000, lr: float = 0.01,
                       tol: float = 1e-6) -> tuple:
        n, p = X.shape
        beta = np.zeros(p)
        best_loss = float('inf')
        best_beta = beta.copy()
        patience = 0
        for iteration in range(max_iter):
            y_pred = X @ beta
            residuals = y - y_pred
            pos_mask = residuals >= 0
            gradient = -X.T @ (quantile * pos_mask - (1 - quantile) * ~pos_mask) / n
            beta = beta - lr * gradient / (1 + iteration * 0.001)
            loss = float(np.mean(np.maximum(quantile * residuals, (quantile - 1) * residuals)))
            if loss < best_loss:
                best_loss = loss
                best_beta = beta.copy()
                patience = 0
            else:
                patience += 1
            if patience > 50 or abs(loss - best_loss) < tol:
                break
        return best_beta, best_loss

    def fit(self, X: np.ndarray, y: np.ndarray,
            quantiles: list[float] = None,
            max_iter: int = 1000,
            lr: float = 0.01) -> dict:
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        n, p = X.shape
        X_with_intercept = np.column_stack([np.ones(n), X])
        results = {}
        for q in quantiles:
            beta, loss = self._fit_quantile(X_with_intercept, y, q, max_iter=max_iter, lr=lr)
            y_pred = X_with_intercept @ beta
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
            self._fitted_quantiles[q] = {
                "beta": beta,
                "loss": loss,
            }
            results[q] = QuantileRegressionResult(
                quantile=q,
                coefficients=beta[1:].tolist(),
                intercept=float(beta[0]),
                predictions=y_pred.tolist(),
                r_squared=float(max(0, r2)),
                n_obs=n,
            )
        return results

    def predict(self, X: np.ndarray, quantiles: list[float] = None) -> dict:
        n = X.shape[0]
        X_with_intercept = np.column_stack([np.ones(n), X])
        if quantiles is None:
            quantiles = sorted(self._fitted_quantiles.keys())
        results = {}
        for q in quantiles:
            if q in self._fitted_quantiles:
                beta = self._fitted_quantiles[q]["beta"]
                results[q] = (X_with_intercept @ beta).tolist()
            else:
                if 0.5 in self._fitted_quantiles:
                    median_pred = X_with_intercept @ self._fitted_quantiles[0.5]["beta"]
                    if 0.05 in self._fitted_quantiles and 0.95 in self._fitted_quantiles:
                        q05 = X_with_intercept @ self._fitted_quantiles[0.05]["beta"]
                        q95 = X_with_intercept @ self._fitted_quantiles[0.95]["beta"]
                        half_width = (q95 - q05) / 2
                        scale = (q - 0.5) / 0.45 if q > 0.5 else (0.5 - q) / 0.45
                        if q > 0.5:
                            results[q] = (median_pred + half_width * scale * np.sign(q - 0.5)).tolist()
                        else:
                            results[q] = (median_pred - half_width * scale).tolist()
                    else:
                        results[q] = median_pred.tolist()
        return results

    def predict_intervals(self, X: np.ndarray,
                           lower_quantile: float = 0.05,
                           upper_quantile: float = 0.95) -> HeteroscedasticInterval:
        preds = self.predict(X, quantiles=[lower_quantile, 0.5, upper_quantile])
        lower = preds.get(lower_quantile, preds.get(0.5, [0.0]))
        point = preds.get(0.5, [0.0])
        upper = preds.get(upper_quantile, preds.get(0.5, [0.0]))
        lower = [float(l) for l in lower]
        upper = [float(u) for u in upper]
        point = [float(p) for p in point]
        width = [float(u - l) for l, u in zip(lower, upper)]
        crossing = any(w < 0 for w in width)
        if crossing:
            lower, upper = point[:], point[:]
            width = [0.0] * len(point)
        return HeteroscedasticInterval(
            lower=lower,
            upper=upper,
            width=width,
            point_forecast=point,
            coverage_probability=upper_quantile - lower_quantile if hasattr(self, '_') else upper_quantile - lower_quantile,
            mean_width=float(np.mean(width)),
            quantile_crossing=crossing,
        )

    @staticmethod
    def noncrossing_quantile_adjust(predictions: dict) -> dict:
        sorted_q = sorted(predictions.keys())
        n = len(predictions[sorted_q[0]]) if sorted_q else 0
        adjusted = {q: list(p) for q, p in predictions.items()}
        for i in range(n):
            for j in range(1, len(sorted_q)):
                q_low = sorted_q[j - 1]
                q_high = sorted_q[j]
                if adjusted[q_high][i] < adjusted[q_low][i]:
                    avg = (adjusted[q_low][i] + adjusted[q_high][i]) / 2
                    adjusted[q_low][i] = avg
                    adjusted[q_high][i] = avg
        return adjusted

    @staticmethod
    def conditional_var(predictions: dict, confidence: float = 0.9) -> dict:
        lower_q = (1 - confidence) / 2
        upper_q = 1 - lower_q
        pred_lower = predictions.get(lower_q, predictions.get(0.05, []))
        pred_upper = predictions.get(upper_q, predictions.get(0.95, []))
        if not pred_lower or not pred_upper:
            return {"conditional_var": [], "mean_var": 0.0}
        width = [float(u - l) for l, u in zip(pred_lower, pred_upper)]
        mean_w = float(np.mean(width)) if width else 0.0
        var_est = [float(w ** 2 / (2 * 1.96 ** 2)) for w in width]
        return {
            "conditional_var": var_est,
            "mean_var": float(np.mean(var_est)),
            "width": width,
            "mean_width": mean_w,
            "heteroscedasticity_ratio": float(np.std(width) / (np.mean(width) + 1e-10)),
        }

    @staticmethod
    def quantile_feature_importance(fitted_results: dict) -> dict:
        importance = {}
        quantiles = sorted(fitted_results.keys())
        if not quantiles:
            return importance
        n_features = len(fitted_results[quantiles[0]].coefficients)
        for feat_idx in range(n_features):
            coef_range = []
            for q in quantiles:
                coef = fitted_results[q].coefficients[feat_idx] if feat_idx < len(fitted_results[q].coefficients) else 0
                coef_range.append(coef)
            variability = float(np.std(coef_range))
            mean_abs = float(np.mean(np.abs(coef_range)))
            importance[f"feature_{feat_idx}"] = {
                "variability": variability,
                "mean_abs_coefficient": mean_abs,
                "quantile_coefficients": {str(q): fitted_results[q].coefficients[feat_idx]
                                          for q in quantiles
                                          if feat_idx < len(fitted_results[q].coefficients)},
                "heteroscedastic_influence": variability / (mean_abs + 1e-10),
            }
        return importance

    @staticmethod
    def backtest_coverage(y_true: np.ndarray, predictions: dict,
                          quantiles: list[float] = None) -> dict:
        if quantiles is None:
            quantiles = sorted(predictions.keys())
        results = {}
        for q in quantiles:
            if q not in predictions:
                continue
            below = sum(1 for i, p in enumerate(predictions[q]) if i < len(y_true) and y_true[i] <= p)
            n = min(len(y_true), len(predictions[q]))
            empirical_q = below / n if n > 0 else 0.0
            results[q] = {
                "nominal": q,
                "empirical": float(empirical_q),
                "bias": float(empirical_q - q),
                "abs_bias": float(abs(empirical_q - q)),
                "n": n,
            }
        return results