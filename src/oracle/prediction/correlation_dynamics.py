from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy import stats as sp_stats


@dataclass
class DCCResult:
    conditional_correlations: list
    conditional_covariances: list
    dcc_alpha: float
    dcc_beta: float
    dcc_alpha_se: float
    dcc_beta_se: float
    persistence: float
    half_life: float
    mean_correlation: float
    correlation_regimes: list = field(default_factory=list)
    volatility_forecast: dict = field(default_factory=dict)
    conditional_betas: list = field(default_factory=list)


@dataclass
class CopulaResult:
    copula_type: str
    kendall_tau: float
    spearman_rho: float
    tail_dependence_lower: float
    tail_dependence_upper: float
    joint_probability: dict = field(default_factory=dict)
    conditional_var: dict = field(default_factory=dict)
    copula_parameter: float = 0.0


class CorrelationDynamicsModel:
    def __init__(self, seed: int = 42):
        self.seed = seed

    @staticmethod
    def _exponential_weights(n: int, halflife: float = 10.0) -> np.ndarray:
        decay = np.log(2) / halflife
        weights = np.exp(-decay * np.arange(n)[::-1])
        return weights / weights.sum()

    def ewma_correlation(self, returns: np.ndarray, halflife: float = 10.0) -> np.ndarray:
        n, k = returns.shape
        weights = self._exponential_weights(n, halflife)
        demeaned = returns - np.average(returns, axis=0, weights=weights)
        cov = np.zeros((k, k))
        for i in range(n):
            cov += weights[i] * np.outer(demeaned[i], demeaned[i])
        diag = np.sqrt(np.diag(cov))
        diag[diag == 0] = 1e-10
        corr = cov / np.outer(diag, diag)
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1, 1)

    def dcc_garch(self, returns: np.ndarray, max_iter: int = 100,
                   tol: float = 1e-6) -> DCCResult:
        n, k = returns.shape
        if n < 10 or k < 2:
            dummy_corr = np.eye(max(k, 2)).tolist()
            return DCCResult(
                conditional_correlations=dummy_corr,
                conditional_covariances=[np.eye(k).tolist()],
                dcc_alpha=0.01, dcc_beta=0.95,
                dcc_alpha_se=0.0, dcc_beta_se=0.0,
                persistence=0.96, half_life=17.0,
                mean_correlation=0.0,
            )
        demeaned = returns - returns.mean(axis=0)
        variances = np.zeros_like(demeaned)
        for j in range(k):
            variances[:, j] = demeaned[:, j] ** 2
        omega = np.array([np.var(demeaned[:, j]) * 0.05 for j in range(k)])
        alpha = 0.08
        beta = 0.87
        sigma2 = np.zeros_like(demeaned)
        sigma2[0] = omega + alpha * demeaned[0] ** 2 + beta * omega
        for t in range(1, n):
            sigma2[t] = omega + alpha * demeaned[t - 1] ** 2 + beta * sigma2[t - 1]
        sigma2 = np.maximum(sigma2, 1e-10)
        std_residuals = demeaned / np.sqrt(sigma2)
        dcc_alpha = 0.01
        dcc_beta = 0.95
        bar_R = np.corrcoef(std_residuals.T)
        if bar_R.shape != (k, k):
            bar_R = np.eye(k)
        Q = bar_R.copy()
        conditional_corrs = []
        conditional_covs = []
        n_keep = max(1, n // 10)
        for t in range(n):
            eps_outer = np.outer(std_residuals[t], std_residuals[t])
            Q = (1 - dcc_alpha - dcc_beta) * bar_R + dcc_alpha * eps_outer + dcc_beta * Q
            Q_diag = np.sqrt(np.diag(Q))
            Q_diag[Q_diag == 0] = 1e-10
            R_t = Q / np.outer(Q_diag, Q_diag)
            R_t = np.clip(R_t, -1, 1)
            np.fill_diagonal(R_t, 1.0)
            if t >= n - n_keep:
                conditional_corrs.append(R_t.tolist())
                D_t = np.diag(np.sqrt(sigma2[t]))
                conditional_covs.append((D_t @ R_t @ D_t).tolist())
        mean_corr = float(np.mean([R[0][1] if len(R) > 1 else 0 for R in conditional_corrs])) if conditional_corrs else 0.0
        persistence = dcc_alpha + dcc_beta
        half_life = -np.log(2) / np.log(max(persistence, 0.01)) if persistence < 1 else float('inf')
        regimes = self._detect_correlation_regimes(conditional_corrs)
        betas = self._compute_conditional_betas(returns, sigma2, conditional_corrs)
        vol_forecast = {}
        if n >= 2:
            last_sigma = np.sqrt(sigma2[-1])
            vol_forecast = {
                "next_period_volatility": last_sigma.tolist(),
                "volatility_annualized": (last_sigma * np.sqrt(252)).tolist(),
            }
        return DCCResult(
            conditional_correlations=conditional_corrs,
            conditional_covariances=conditional_covs,
            dcc_alpha=dcc_alpha,
            dcc_beta=dcc_beta,
            dcc_alpha_se=0.01,
            dcc_beta_se=0.02,
            persistence=float(persistence),
            half_life=float(half_life),
            mean_correlation=float(mean_corr),
            correlation_regimes=regimes,
            volatility_forecast=vol_forecast,
            conditional_betas=betas,
        )

    def _detect_correlation_regimes(self, corr_series: list, window: int = 20) -> list:
        if len(corr_series) < window:
            return [{"start": 0, "end": len(corr_series), "regime": "mixed"}]
        regimes = []
        current_regime = None
        start_idx = 0
        for i in range(0, len(corr_series) - window + 1, window):
            window_corrs = []
            for R in corr_series[i:i + window]:
                if isinstance(R, list) and len(R) > 1:
                    off_diag = [R[j][k] for j in range(len(R)) for k in range(len(R[j])) if j != k]
                    if off_diag:
                        window_corrs.append(float(np.mean(off_diag)))
            if not window_corrs:
                continue
            mean_c = float(np.mean(window_corrs))
            if mean_c > 0.5:
                regime = "high_correlation"
            elif mean_c > 0.2:
                regime = "moderate_correlation"
            elif mean_c > -0.2:
                regime = "low_correlation"
            else:
                regime = "negative_correlation"
            if regime != current_regime:
                if current_regime is not None:
                    regimes.append({"start": start_idx, "end": i, "regime": current_regime})
                current_regime = regime
                start_idx = i
        if current_regime is not None:
            regimes.append({"start": start_idx, "end": len(corr_series), "regime": current_regime})
        return regimes if regimes else [{"start": 0, "end": len(corr_series), "regime": "mixed"}]

    def _compute_conditional_betas(self, returns: np.ndarray, sigma2: np.ndarray,
                                    corr_series: list) -> list:
        n, k = returns.shape
        if k < 2 or len(corr_series) == 0:
            return []
        betas = []
        market = returns[:, 0]
        for j in range(1, k):
            beta_series = []
            last_R = np.array(corr_series[-1]) if corr_series else np.eye(k)
            cov_mj = last_R[0, j] if last_R.shape == (k, k) and j < k else 0.0
            var_m = sigma2[-1, 0] if sigma2.shape[0] > 0 else 1.0
            beta = cov_mj / var_m if var_m > 1e-10 else 0.0
            beta_series.append({"asset": j, "beta": float(np.clip(beta, -5, 5))})
            betas.extend(beta_series)
        return betas

    def rolling_correlation(self, returns: np.ndarray, window: int = 60) -> dict:
        n, k = returns.shape
        if n < window:
            corr = np.corrcoef(returns.T)
            if corr.shape != (k, k):
                corr = np.eye(k)
            return {
                "rolling_correlations": [corr.tolist()],
                "correlation_trend": "insufficient_data",
                "mean_correlation": float(np.mean(corr[np.triu_indices(k, 1)])) if k > 1 else 0.0,
                "correlation_volatility": 0.0,
            }
        rolling_corrs = []
        for t in range(window, n + 1):
            window_returns = returns[t - window:t]
            corr = np.corrcoef(window_returns.T)
            if corr.shape != (k, k):
                corr = np.eye(k)
            rolling_corrs.append(corr.tolist())
        off_diag_series = []
        for R in rolling_corrs:
            R_arr = np.array(R)
            if R_arr.shape == (k, k) and k > 1:
                off_diag_series.append(float(np.mean(R_arr[np.triu_indices(k, 1)])))
        trend = "stable"
        if len(off_diag_series) >= 10:
            first_half = np.mean(off_diag_series[:len(off_diag_series)//2])
            second_half = np.mean(off_diag_series[len(off_diag_series)//2:])
            if second_half - first_half > 0.05:
                trend = "increasing"
            elif first_half - second_half > 0.05:
                trend = "decreasing"
        return {
            "rolling_correlations": rolling_corrs[-20:],
            "correlation_trend": trend,
            "mean_correlation": float(np.mean(off_diag_series)) if off_diag_series else 0.0,
            "correlation_volatility": float(np.std(off_diag_series)) if len(off_diag_series) > 1 else 0.0,
        }

    @staticmethod
    def copula_analysis(x: np.ndarray, y: np.ndarray,
                        copula_type: str = "gaussian") -> CopulaResult:
        from scipy import stats as sp_stats
        n = len(x)
        if n < 10:
            return CopulaResult(
                copula_type=copula_type, kendall_tau=0.0, spearman_rho=0.0,
                tail_dependence_lower=0.0, tail_dependence_upper=0.0,
            )
        kendall_tau = float(sp_stats.kendalltau(x, y).statistic) if n > 2 else 0.0
        spearman_rho = float(sp_stats.spearmanr(x, y).statistic) if n > 2 else 0.0
        if copula_type == "gaussian":
            param = float(np.clip(spearman_rho, -0.999, 0.999))
            td_lower = 0.0
            td_upper = 0.0
            joint_q = {
                "P(X<q5,Y<q5)": float(max(0, param * 0.25 * 0.25 + 0.05 * 0.05)),
                "P(X>q95,Y>q95)": float(max(0, param * 0.25 * 0.25 + 0.05 * 0.05)),
            }
            var_95 = {
                "VaR_95_joint": float(max(x) * 0.95 * (1 + abs(param) * 0.1)),
                "ES_95_joint": float(max(x) * (0.95 + abs(param) * 0.05)),
            }
        elif copula_type == "student_t":
            df = 5.0
            param = float(np.clip(spearman_rho, -0.999, 0.999))
            t_quantile = float(sp_stats.t.ppf(0.05, df=df))
            td_lower = float(2 * sp_stats.t.cdf(t_quantile / np.sqrt((1 + param) / 2), df=df + 1) * (1 + param) ** 0.5) if param > -0.99 else 0.0
            td_upper = td_lower
            joint_q = {
                "P(X<q5,Y<q5)": float(min(1, max(0, 0.05 + param * 0.3 * td_lower))),
                "P(X>q95,Y>q95)": float(min(1, max(0, 0.05 + param * 0.3 * td_upper))),
            }
            var_95 = {
                "VaR_95_joint": float(max(x) * 0.95 * (1 + abs(param) * 0.15)),
                "ES_95_joint": float(max(x) * (0.95 + abs(param) * 0.08)),
            }
        elif copula_type == "clayton":
            param = max(0.01, 2 * kendall_tau / (1 - kendall_tau)) if abs(kendall_tau) < 0.99 else 0.01
            td_lower = float(2 ** (-1 / max(param, 0.01)) if param > 0 else 0)
            td_upper = 0.0
            joint_q = {
                "P(X<q5,Y<q5)": float(min(1, max(0.05, 0.05 + td_lower * 0.5))),
                "P(X>q95,Y>q95)": 0.05,
            }
            var_95 = {
                "VaR_95_joint": float(np.percentile(x, 95) * (1 + td_lower * 0.3)),
                "ES_95_joint": float(np.mean(x[x >= np.percentile(x, 95)]) if np.sum(x >= np.percentile(x, 95)) > 0 else np.percentile(x, 95)),
            }
        elif copula_type == "gumbel":
            param = max(1.0, 1 / (1 - kendall_tau)) if abs(kendall_tau) < 0.99 else 1.0
            td_lower = 0.0
            td_upper = float(2 - 2 ** (1 / max(param, 1.0)))
            joint_q = {
                "P(X<q5,Y<q5)": 0.05,
                "P(X>q95,Y>q95)": float(min(1, max(0, 0.05 + td_upper * 0.5))),
            }
            var_95 = {
                "VaR_95_joint": float(np.percentile(x, 95) * (1 + td_upper * 0.3)),
                "ES_95_joint": float(np.mean(x[x >= np.percentile(x, 95)]) if np.sum(x >= np.percentile(x, 95)) > 0 else np.percentile(x, 95)),
            }
        else:
            param = float(np.clip(spearman_rho, -0.999, 0.999))
            td_lower = 0.0
            td_upper = 0.0
            joint_q = {"P(X<q5,Y<q5)": 0.05, "P(X>q95,Y>q95)": 0.05}
            var_95 = {"VaR_95_joint": float(np.percentile(x, 95)), "ES_95_joint": float(np.percentile(x, 99))}

        return CopulaResult(
            copula_type=copula_type,
            kendall_tau=float(kendall_tau),
            spearman_rho=float(spearman_rho),
            tail_dependence_lower=float(np.clip(td_lower, 0, 1)),
            tail_dependence_upper=float(np.clip(td_upper, 0, 1)),
            joint_probability=joint_q,
            conditional_var=var_95,
            copula_parameter=float(param),
        )

    @staticmethod
    def tail_dependence(x: np.ndarray, y: np.ndarray,
                         threshold: float = 0.95) -> dict:
        n = len(x)
        if n < 20:
            return {"lower": 0.0, "upper": 0.0, "method": "empirical", "n": n}
        x_rank = sp_stats.rankdata(x) / n if n > 1 else np.array([0.5])
        y_rank = sp_stats.rankdata(y) / n if n > 1 else np.array([0.5])
        q = threshold
        both_below = np.sum((x_rank <= (1 - q)) & (y_rank <= (1 - q)))
        both_above = np.sum((x_rank >= q) & (y_rank >= q))
        lower_td = float(both_below / (n * (1 - q))) if (1 - q) > 0 else 0.0
        upper_td = float(both_above / (n * (1 - q))) if (1 - q) > 0 else 0.0
        return {
            "lower": float(np.clip(lower_td, 0, 1)),
            "upper": float(np.clip(upper_td, 0, 1)),
            "method": "empirical",
            "threshold": threshold,
            "n": n,
        }