import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from itertools import combinations
from scipy import stats as sp_stats
from scipy.signal import correlate
from oracle.core.types import TimeSeries, Correlation, CorrelationMatrix, Domain


class CorrelationEngine:
    def __init__(self, min_samples: int = 20, significance_level: float = 0.05):
        self.min_samples = min_samples
        self.significance_level = significance_level

    def pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]
        if len(x_clean) < self.min_samples:
            return 0.0, 1.0
        r, p = sp_stats.pearsonr(x_clean, y_clean)
        return float(r), float(p)

    def spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]
        if len(x_clean) < self.min_samples:
            return 0.0, 1.0
        r, p = sp_stats.spearmanr(x_clean, y_clean)
        return float(r), float(p)

    def cross_correlation_with_lag(self, x: np.ndarray, y: np.ndarray, max_lag: int = 50) -> dict:
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]
        n = len(x_clean)
        max_lag = min(max_lag, n // 2)
        if n < self.min_samples:
            return {"max_correlation": 0.0, "best_lag": 0, "correlations": np.zeros(2 * max_lag + 1)}

        x_norm = (x_clean - np.mean(x_clean)) / (np.std(x_clean) + 1e-10)
        y_norm = (y_clean - np.mean(y_clean)) / (np.std(y_clean) + 1e-10)
        correlation = correlate(x_norm, y_norm, mode="full") / n
        lags = np.arange(-max_lag, max_lag + 1)
        mid = len(correlation) // 2
        corr_slice = correlation[mid - max_lag:mid + max_lag + 1]

        best_lag_idx = np.argmax(np.abs(corr_slice))
        best_lag = lags[best_lag_idx]
        max_corr = corr_slice[best_lag_idx]

        return {
            "max_correlation": float(max_corr),
            "best_lag": int(best_lag),
            "correlations": corr_slice,
            "lags": lags,
            "positive_lag_means_x_leads": float(max_corr) > 0 and best_lag > 0,
        }

    def granger_causality(self, x: np.ndarray, y: np.ndarray, max_lag: int = 10) -> dict:
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]
        n = len(x_clean)
        if n < max_lag + 5:
            return {"f_statistic": 0.0, "p_value": 1.0, "causality": False}

        best_lag = 1
        best_f = 0.0
        best_p = 1.0

        for lag in range(1, min(max_lag + 1, n // 3)):
            y_lagged = np.column_stack([y_clean[lag:], np.column_stack([y_clean[i:-lag + i] if i > 0 else y_clean[:-lag] for i in range(lag)])] if lag > 0 else [y_clean[:-lag]])
            x_lagged = np.column_stack([x_clean[i:-lag + i] if i < lag else x_clean[:-lag] for i in range(lag)])

            try:
                ssr_restricted = np.sum((y_clean[lag:] - np.mean(y_clean[lag:])) ** 2)
                X_full = np.column_stack([np.ones(len(y_clean[lag:])), y_clean[:-lag].reshape(-1, 1), x_clean[:-lag].reshape(-1, 1)])
                beta = np.linalg.lstsq(X_full, y_clean[lag:], rcond=None)[0]
                residuals_full = y_clean[lag:] - X_full @ beta
                ssr_unrestricted = np.sum(residuals_full ** 2)

                f_stat = ((ssr_restricted - ssr_unrestricted) / lag) / (ssr_unrestricted / (n - 2 * lag - 1))
                p_val = 1 - sp_stats.f.cdf(f_stat, lag, n - 2 * lag - 1)

                if f_stat > best_f:
                    best_f = f_stat
                    best_p = p_val
                    best_lag = lag
            except Exception:
                continue

        return {
            "f_statistic": float(best_f),
            "p_value": float(best_p),
            "best_lag": best_lag,
            "causality": best_p < self.significance_level and best_f > 3.0,
            "direction": "x_causes_y" if best_p < self.significance_level else "no_causality",
        }

    def partial_correlation(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x_c, y_c, z_c = x[mask], y[mask], z[mask]
        if len(x_c) < self.min_samples:
            return 0.0
        beta_xz = np.polyfit(z_c, x_c, 1)
        beta_yz = np.polyfit(z_c, y_c, 1)
        res_x = x_c - np.polyval(beta_xz, z_c)
        res_y = y_c - np.polyval(beta_yz, z_c)
        r, _ = sp_stats.pearsonr(res_x, res_y)
        return float(r)

    def rolling_correlation(self, x: np.ndarray, y: np.ndarray, window: int = 60) -> np.ndarray:
        n = len(x)
        if n < window:
            r, _ = sp_stats.pearsonr(x, y) if n > 2 else (0.0, 1.0)
            return np.array([r])
        correlations = np.zeros(n - window + 1)
        for i in range(n - window + 1):
            r, _ = sp_stats.pearsonr(x[i:i + window], y[i:i + window])
            correlations[i] = r
        return correlations

    def build_correlation_matrix(self, series_dict: dict[str, np.ndarray]) -> CorrelationMatrix:
        names = list(series_dict.keys())
        n = len(names)
        if n < 2:
            return CorrelationMatrix(variables=names, matrix=np.eye(1))

        min_len = min(len(v) for v in series_dict.values())
        aligned = {k: v[:min_len] for k, v in series_dict.items()}

        corr_matrix = np.eye(n)
        p_matrix = np.zeros((n, n))
        lag_matrix = np.zeros((n, n))

        for i, j in combinations(range(n), 2):
            r, p = self.pearson_correlation(aligned[names[i]], aligned[names[j]])
            corr_matrix[i, j] = corr_matrix[j, i] = r
            p_matrix[i, j] = p_matrix[j, i] = p

            cross = self.cross_correlation_with_lag(aligned[names[i]], aligned[names[j]], max_lag=min(20, min_len // 4))
            lag_matrix[i, j] = cross.get("best_lag", 0)
            lag_matrix[j, i] = -cross.get("best_lag", 0)

        return CorrelationMatrix(
            variables=names, matrix=corr_matrix, p_values=p_matrix, lag_matrix=lag_matrix,
        )

    def discover_cross_domain_patterns(
        self, series_dict: dict[str, np.ndarray],
        domains: Optional[dict[str, Domain]] = None,
    ) -> list[Correlation]:
        corr_matrix = self.build_correlation_matrix(series_dict)
        strong_correlations = corr_matrix.get_strong_correlations(threshold=0.3)

        results = []
        for corr in strong_correlations:
            if domains:
                corr.domain_a = domains.get(corr.source_a, Domain.ALL)
                corr.domain_b = domains.get(corr.source_b, Domain.ALL)

            x = series_dict.get(corr.source_a, np.array([]))
            y = series_dict.get(corr.source_b, np.array([]))
            if len(x) > 0 and len(y) > 0 and len(x) == len(y):
                gc = self.granger_causality(x, y)
                corr.metadata = {
                    "granger_causality": gc.get("causality", False),
                    "granger_p_value": gc.get("p_value", 1.0),
                    "best_lag": gc.get("best_lag", 0),
                }

            results.append(corr)

        return results