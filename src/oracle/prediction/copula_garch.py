"""Copula-GARCH module for conditional dependency modeling.

Implements GARCH(1,1) volatility modeling with copula-based dependency
structure for multivariate time series with time-varying correlations
and tail dependencies.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import math


@dataclass
class GARCHResult:
    conditional_variance: List[float]
    standardized_residuals: List[float]
    omega: float
    alpha: float
    beta: float
    persistence: float
    unconditional_variance: float
    log_likelihood: float
    n_observations: int
    converged: bool
    method: str = "garch_1_1"


@dataclass
class CopulaGARCHResult:
    marginal_results: List[dict]
    copula_rho: float
    copula_type: str
    joint_log_likelihood: float
    conditional_correlation: float
    tail_dependence_lower: float
    tail_dependence_upper: float
    forecast_correlation: float
    method: str = "copula_garch"


def _garch_likelihood(params: np.ndarray, residuals: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    T = len(residuals)
    h = np.zeros(T)
    h[0] = omega / (1 - alpha - beta) if (1 - alpha - beta) > 1e-10 else omega
    for t in range(1, T):
        h[t] = omega + alpha * residuals[t-1]**2 + beta * h[t-1]
        if h[t] <= 0:
            h[t] = 1e-6
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + residuals**2 / h)
    return -ll


def _fit_garch(residuals: np.ndarray, max_iter: int = 50) -> dict:
    omega = float(np.var(residuals)) * 0.1
    alpha = 0.1
    beta = 0.85
    best_params = np.array([omega, alpha, beta])
    best_ll = _garch_likelihood(best_params, residuals)
    lr = 0.01
    for iteration in range(max_iter):
        eps = 1e-5
        grad = np.zeros(3)
        for i in range(3):
            params_plus = best_params.copy()
            params_plus[i] += eps
            params_minus = best_params.copy()
            params_minus[i] -= eps
            grad[i] = (_garch_likelihood(params_plus, residuals) -
                       _garch_likelihood(params_minus, residuals)) / (2 * eps)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm
        for lr_try in [lr, lr * 0.5, lr * 0.1]:
            new_params = best_params - lr_try * grad * 0.01
            new_params[0] = max(new_params[0], 1e-6)
            new_params[1] = max(min(new_params[1], 0.5), 0.001)
            new_params[2] = max(min(new_params[2], 0.99), 0.01)
            if new_params[1] + new_params[2] < 0.999:
                new_ll = _garch_likelihood(new_params, residuals)
                if new_ll < best_ll:
                    best_params = new_params
                    best_ll = new_ll
                    break
    omega, alpha, beta = best_params
    T = len(residuals)
    h = np.zeros(T)
    h[0] = omega / (1 - alpha - beta + 1e-10)
    std_resid = np.zeros(T)
    for t in range(1, T):
        h[t] = omega + alpha * residuals[t-1]**2 + beta * h[t-1]
        h[t] = max(h[t], 1e-10)
    h[0] = max(h[0], 1e-10)
    std_resid = residuals / np.sqrt(np.maximum(h, 1e-10))
    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "conditional_variance": h.tolist(),
        "standardized_residuals": std_resid.tolist(),
        "persistence": float(alpha + beta),
        "unconditional_variance": float(omega / (1 - alpha - beta + 1e-10)),
        "log_likelihood": float(-best_ll),
        "converged": True,
    }


def _gaussian_copula_density(u: np.ndarray, v: np.ndarray, rho: float) -> float:
    z_u = _norm_ppf(np.clip(u, 1e-6, 1 - 1e-6))
    z_v = _norm_ppf(np.clip(v, 1e-6, 1 - 1e-6))
    rho_c = max(min(rho, 0.999), -0.999)
    exponent = -(rho_c**2 * (z_u**2 + z_v**2) - 2 * rho_c * z_u * z_v) / (2 * (1 - rho_c**2))
    return exponent - 0.5 * np.log(1 - rho_c**2)


def _norm_ppf(p: float) -> float:
    if isinstance(p, np.ndarray):
        return np.array([_norm_ppf_scalar(pi) for pi in p.flat]).reshape(p.shape)
    return _norm_ppf_scalar(p)


def _norm_ppf_scalar(p: float) -> float:
    a = [0, -3.969683028665376e+01, 2.209460984269744e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [0, -5.447609879822406e+01, 1.618151413137677e+02,
         -1.361608557649085e+02, 5.398178106577912e+01,
         -7.836600133987812e+00, 1.0]
    c = [0, -5.504726025014842e-06, 2.238721602868431e-03,
         2.774539701477588e-03, 4.639405927022524e-02,
         1.349770164393436e-01, 2.003626029201024e-02,
         -5.276524048649425e-02, 1.865286584927251e-01]
    d = [0, 3.855624362566842e-02, 2.244025673527610e+01,
         5.725252090829401e+01, 5.266993034617726e+01,
         1.680035314086803e+01]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / \
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q / \
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+b[6])
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / \
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)


class GARCHModel:
    """GARCH(1,1) model for volatility estimation."""

    def __init__(self, omega: float = 0.01, alpha: float = 0.1,
                 beta: float = 0.85):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

    def fit(self, returns: np.ndarray, max_iter: int = 50) -> GARCHResult:
        result = _fit_garch(returns, max_iter)
        self.unconditional_variance = result["unconditional_variance"]
        return GARCHResult(
            conditional_variance=result["conditional_variance"],
            standardized_residuals=result["standardized_residuals"],
            omega=result["omega"],
            alpha=result["alpha"],
            beta=result["beta"],
            persistence=result["persistence"],
            unconditional_variance=result["unconditional_variance"],
            log_likelihood=result["log_likelihood"],
            n_observations=len(returns),
            converged=result["converged"],
        )

    def forecast(self, residuals: np.ndarray, n_ahead: int = 10) -> dict:
        h_last = self.omega + self.alpha * residuals[-1]**2 + self.beta * np.var(residuals)
        forecast_var = []
        for i in range(n_ahead):
            h_forecast = self.unconditional_variance + (self.alpha + self.beta)**i * (h_last - self.unconditional_variance) if hasattr(self, 'unconditional_variance') else h_last
            forecast_var.append(float(h_forecast) if isinstance(h_forecast, (int, float, np.floating)) else float(h_last))
            h_last = self.omega + (self.alpha + self.beta) * h_last
        return {
            "forecast_variance": forecast_var,
            "forecast_volatility": [float(np.sqrt(v)) for v in forecast_var],
            "method": "garch_forecast"
        }


class CopulaGARCH:
    """Copula-GARCH model for multivariate dependency with GARCH marginals.

    Fits GARCH(1,1) to each marginal series, transforms to uniform via
    empirical CDF, then estimates copula dependency structure.
    Supports Gaussian, Clayton, Gumbel, and Frank copulas.
    """

    def __init__(self, copula_type: str = "auto", seed: int = 42):
        self.copula_type = copula_type
        self.seed = seed
        self.marginal_models: List[dict] = []
        self.copula_rho: float = 0.0
        self._fitted = False

    def _empirical_cdf(self, x: np.ndarray) -> np.ndarray:
        sorted_x = np.sort(x)
        ranks = np.searchsorted(sorted_x, x) + 1
        return ranks / (len(x) + 1)

    def _kendall_tau(self, u: np.ndarray, v: np.ndarray) -> float:
        n = len(u)
        concordant = 0
        discordant = 0
        for i in range(min(n, 500)):
            for j in range(i + 1, min(n, 500)):
                if (u[i] - u[j]) * (v[i] - v[j]) > 0:
                    concordant += 1
                else:
                    discordant += 1
        total = concordant + discordant
        return (concordant - discordant) / max(total, 1)

    def _clayton_copula_ll(self, u: np.ndarray, v: np.ndarray, theta: float) -> float:
        if theta <= 0:
            return -1e10
        n = len(u)
        ll = 0.0
        for i in range(n):
            ui, vi = max(u[i], 1e-6), max(v[i], 1e-6)
            ll += (theta + 1) * math.log(max(ui**(-theta) + vi**(-theta) - 1, 1e-10)) - \
                  (theta + 1) * math.log(ui * vi) - (1/theta + 2) * math.log(ui**(-theta) + vi**(-theta) - 1 + 1e-10)
        return ll / n

    def _gumbel_copula_ll(self, u: np.ndarray, v: np.ndarray, theta: float) -> float:
        if theta < 1:
            return -1e10
        n = len(u)
        ll = 0.0
        for i in range(n):
            ui, vi = max(u[i], 1e-6), max(v[i], 1e-6)
            a = (-math.log(ui))**theta + (-math.log(vi))**theta
            ll += math.log(theta + 1) + (theta - 1) * math.log(max(a, 1e-10)) - \
                  a**(1/theta) + (1/theta - 1) * math.log(max(a**(1/theta), 1e-10)) - \
                  (theta - 1) * math.log(ui * vi)
        return ll / n

    def _frank_copula_ll(self, u: np.ndarray, v: np.ndarray, theta: float) -> float:
        n = len(u)
        ll = 0.0
        for i in range(n):
            ui, vi = max(u[i], 1e-6), max(v[i], 1e-6)
            ex = math.exp(-theta) - 1
            if abs(ex) < 1e-10:
                return -1e10
            numerator = theta * math.exp(-theta * (ui + vi)) * ex
            denominator = (1 + ex * (math.exp(-theta * ui) - 1) * (math.exp(-theta * vi) - 1) / max(ex, 1e-10))**2
            if denominator < 1e-10:
                return -1e10
            ll += math.log(max(numerator / denominator, 1e-10))
        return ll / n

    def _select_copula(self, u: np.ndarray, v: np.ndarray) -> Tuple[str, float, float]:
        tau = self._kendall_tau(u, v)
        best_name = "gaussian"
        best_ll = -1e10
        best_param = self.copula_rho
        rho = np.corrcoef(_norm_ppf(np.clip(u, 1e-6, 1-1e-6)),
                          _norm_ppf(np.clip(v, 1e-6, 1-1e-6)))[0, 1]
        rho = max(min(rho, 0.999), -0.999)
        best_ll = -0.5 * np.sum(np.log(1 - rho**2))
        best_param = rho
        if tau > 0:
            theta_clayton = max(2 * tau / (1 - tau), 0.01)
            ll_clayton = self._clayton_copula_ll(u, v, theta_clayton)
            if ll_clayton > best_ll:
                best_ll = ll_clayton
                best_name = "clayton"
                best_param = theta_clayton
            theta_gumbel = max(1 / (1 - tau), 1.001)
            ll_gumbel = self._gumbel_copula_ll(u, v, theta_gumbel)
            if ll_gumbel > best_ll:
                best_ll = ll_gumbel
                best_name = "gumbel"
                best_param = theta_gumbel
        theta_frank = 3.0 * tau if abs(tau) < 0.9 else 5.0
        if abs(theta_frank) > 0.01:
            ll_frank = self._frank_copula_ll(u, v, theta_frank)
            if ll_frank > best_ll:
                best_ll = ll_frank
                best_name = "frank"
                best_param = theta_frank
        return best_name, best_param, best_ll

    def fit(self, data: np.ndarray) -> CopulaGARCHResult:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_series = data.shape[1] if data.ndim > 1 else 2
        if n_series == 1:
            data = np.column_stack([data.ravel(), np.random.RandomState(self.seed).normal(0, 1, len(data))])
            n_series = 2
        marginal_results = []
        std_residuals = []
        for i in range(n_series):
            col = data[:, i]
            garch_result = _fit_garch(np.diff(col) if len(col) > 10 else col)
            marginal_results.append(garch_result)
            std_residuals.append(np.array(garch_result["standardized_residuals"]))
        min_len = min(len(sr) for sr in std_residuals)
        std_residuals = [sr[:min_len] for sr in std_residuals]
        u = self._empirical_cdf(std_residuals[0])
        v = self._empirical_cdf(std_residuals[1])
        rho = np.corrcoef(std_residuals[0], std_residuals[1])[0, 1]
        rho = max(min(rho, 0.999), -0.999)
        if self.copula_type == "auto":
            copula_name, copula_param, copula_ll = self._select_copula(u, v)
        elif self.copula_type == "gaussian":
            copula_name = "gaussian"
            copula_param = rho
            copula_ll = 0.0
        elif self.copula_type == "clayton":
            tau = self._kendall_tau(u, v)
            copula_name = "clayton"
            copula_param = max(2 * tau / (1 - tau), 0.01)
            copula_ll = self._clayton_copula_ll(u, v, copula_param)
        elif self.copula_type == "gumbel":
            tau = self._kendall_tau(u, v)
            copula_name = "gumbel"
            copula_param = max(1 / max(1 - tau, 0.01), 1.001)
            copula_ll = self._gumbel_copula_ll(u, v, copula_param)
        elif self.copula_type == "frank":
            tau = self._kendall_tau(u, v)
            copula_name = "frank"
            copula_param = max(3.0 * tau, 0.01)
            copula_ll = self._frank_copula_ll(u, v, copula_param)
        else:
            copula_name = "gaussian"
            copula_param = rho
            copula_ll = 0.0
        lower_td = 0.0
        upper_td = 0.0
        if copula_name == "clayton":
            if copula_param > 0:
                lower_td = 2 ** (-1 / copula_param)
        elif copula_name == "gumbel":
            if copula_param > 1:
                upper_td = 2 - 2 ** (1 / copula_param)
        elif copula_name == "gaussian":
            lower_td = 2 * _norm_ppf_2(-math.sqrt((1 + rho) / 2)) if rho > -1 else 0
            upper_td = lower_td
        self.copula_rho = copula_param
        self._fitted = True
        self.marginal_models = marginal_results
        joint_ll = sum(r.get("log_likelihood", 0) for r in marginal_results) + copula_ll
        return CopulaGARCHResult(
            marginal_results=marginal_results,
            copula_rho=copula_param,
            copula_type=copula_name,
            joint_log_likelihood=float(joint_ll),
            conditional_correlation=float(rho),
            tail_dependence_lower=float(lower_td),
            tail_dependence_upper=float(upper_td),
            forecast_correlation=float(rho),
            method=f"copula_garch_{copula_name}"
        )

    def conditional_forecast(self, data: np.ndarray,
                            n_ahead: int = 10,
                            n_simulations: int = 1000) -> dict:
        if not self._fitted:
            self.fit(data)
        rng = np.random.RandomState(self.seed)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_series = min(data.shape[1] if data.ndim > 1 else 2, len(self.marginal_models))
        forecasts = {f"series_{i}": [] for i in range(n_series)}
        for i in range(n_series):
            col = data[:, i] if n_series > 1 else data.ravel()
            garch = self.marginal_models[i]
            h_last = garch["conditional_variance"][-1] if garch["conditional_variance"] else np.var(np.diff(col))
            vol_forecast = []
            for h in range(n_ahead):
                h_next = garch["omega"] + (garch["alpha"] + garch["beta"]) * h_last + \
                         garch["alpha"] * 0 + garch["beta"] * (h_last - garch["unconditional_variance"])
                vol_forecast.append(float(np.sqrt(abs(h_next))))
                h_last = h_next
            forecasts[f"series_{i}"] = {
                "forecast_volatility": vol_forecast,
                "forecast_returns": [float(rng.normal(0, v)) for v in vol_forecast],
                "forecast_levels": [float(col[-1] * (1 + r)) for r in
                                   rng.normal(0, vol_forecast[0] if vol_forecast else 0.01, n_ahead)],
            }
        rho = self.conditional_correlation if hasattr(self, 'conditional_correlation') else 0.5
        simulations = []
        for _ in range(n_simulations):
            z1 = rng.normal(0, 1)
            z2 = rho * z1 + math.sqrt(1 - rho**2) * rng.normal(0, 1)
            simulations.append((z1, z2))
        return {
            "forecasts": forecasts,
            "n_ahead": n_ahead,
            "n_simulations": n_simulations,
            "conditional_correlation": float(self.copula_rho if self.copula_type != "gaussian" else 0.5),
            "copula_type": self.copula_type if hasattr(self, 'copula_type') else "gaussian",
            "method": "copula_garch_forecast"
        }

    def risk_metrics(self, data: np.ndarray, confidence: float = 0.95) -> dict:
        if not self._fitted:
            self.fit(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        risks = {}
        for i, model in enumerate(self.marginal_models):
            cond_var = np.array(model["conditional_variance"])
            std_resid = np.array(model["standardized_residuals"])
            vol = float(np.sqrt(cond_var[-1])) if len(cond_var) > 0 else float(np.std(data[:, i] if data.shape[1] > i else data.ravel()))
            returns = data[:, i] if data.shape[1] > i else data.ravel()
            var_95 = float(-vol * 1.645)
            var_99 = float(-vol * 2.326)
            cvar_95 = float(-vol * 2.06)
            cvar_99 = float(-vol * 2.67)
            risks[f"series_{i}"] = {
                "VaR_95": var_95,
                "VaR_99": var_99,
                "CVaR_95": cvar_95,
                "CVaR_99": cvar_99,
                "conditional_volatility": vol,
                "persistence": model["persistence"],
                "unconditional_variance": model["unconditional_variance"],
            }
        joint_var = float(-np.sqrt(sum(r["conditional_volatility"]**2 for r in risks.values())) * 1.645 *
                         np.sqrt(1 + abs(self.copula_rho)))
        risks["joint"] = {
            "VaR_95": joint_var,
            "copula_type": self.copula_type if hasattr(self, 'copula_type') else "gaussian",
            "copula_parameter": float(self.copula_rho),
            "tail_dependence_lower": 0.0,
            "tail_dependence_upper": 0.0,
        }
        return risks


def _norm_ppf_2(x: float) -> float:
    return _norm_ppf(max(min(x, 0.999), 0.001))