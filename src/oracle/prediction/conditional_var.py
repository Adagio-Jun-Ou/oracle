from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ComponentVaRResult:
    total_var: float
    component_vars: dict
    marginal_vars: dict
    incremental_vars: dict
    percent_contributions: dict
    conditional_var: float
    method: str


@dataclass
class FactorDecompositionResult:
    systematic_var: float
    idiosyncratic_var: float
    factor_betas: dict
    factor_contributions: dict
    r_squared: float
    n_factors: int


@dataclass
class StressVaRResult:
    baseline_var: float
    stressed_var: float
    var_increase: float
    stressed_returns: dict
    scenario_name: str
    factor_shocks: dict


class ConditionalVaRDecomposition:
    def __init__(self, confidence_level: float = 0.95, seed: int = 42):
        self.confidence_level = confidence_level
        self.seed = seed

    def component_var(self, returns: np.ndarray, weights: np.ndarray) -> ComponentVaRResult:
        n_assets = returns.shape[1] if returns.ndim > 1 else 1
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        portfolio_returns = returns @ weights
        z = {0.90: 1.282, 0.95: 1.645, 0.975: 1.96, 0.99: 2.326}.get(
            self.confidence_level, 1.645)
        mu = np.mean(returns, axis=0)
        cov_mat = np.cov(returns.T)
        if cov_mat.ndim == 0:
            cov_mat = np.array([[float(cov_mat)]])
        sigma_p = float(np.sqrt(weights @ cov_mat @ weights))
        total_var = float(-np.sum(weights * mu) + z * sigma_p)
        comp_vars = {}
        marg_vars = {}
        incr_vars = {}
        mu_p = float(np.sum(weights * mu))
        for i in range(n_assets):
            beta_i = float((cov_mat @ weights)[i] / max(sigma_p ** 2, 1e-10))
            comp_vars[f"asset_{i}"] = float(-weights[i] * mu[i] + weights[i] * beta_i * sigma_p * z)
            marg_vars[f"asset_{i}"] = float(beta_i * (cov_mat @ weights)[i])
            base_weights = weights.copy()
            base_weights[i] = 0
            if base_weights.sum() > 1e-10:
                base_weights = base_weights / base_weights.sum()
            base_sigma = float(np.sqrt(base_weights @ cov_mat @ base_weights))
            base_var = float(-np.sum(base_weights * mu) + z * base_sigma)
            incr_vars[f"asset_{i}"] = float(total_var - base_var)
        total_comp = sum(comp_vars.values()) + 1e-10
        pct_contribs = {k: v / total_comp for k, v in comp_vars.items()}
        cvar_level = float(np.percentile(portfolio_returns, (1 - self.confidence_level) * 100))
        return ComponentVaRResult(
            total_var=total_var,
            component_vars=comp_vars,
            marginal_vars=marg_vars,
            incremental_vars=incr_vars,
            percent_contributions=pct_contribs,
            conditional_var=float(-cvar_level),
            method="parametric",
        )

    def factor_decomposition(self, returns: np.ndarray, factor_returns: np.ndarray,
                              portfolio_weights: np.ndarray) -> FactorDecompositionResult:
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        if factor_returns.ndim == 1:
            factor_returns = factor_returns.reshape(-1, 1)
        n_factors = factor_returns.shape[1]
        factor_betas = {}
        factor_contribs = {}
        systematic_var = 0.0
        idio_var = 0.0
        cov_mat = np.cov(returns.T)
        if cov_mat.ndim == 0:
            cov_mat = np.array([[float(cov_mat)]])
        sigma_p_sq = float(portfolio_weights @ cov_mat @ portfolio_weights)
        residual_var = 0.0
        for f in range(n_factors):
            f_col = factor_returns[:, f]
            port_returns = returns @ portfolio_weights
            beta = float(np.cov(port_returns, f_col)[0, 1] / max(np.var(f_col), 1e-10))
            factor_betas[f"factor_{f}"] = beta
            f_var = float(np.var(f_col) * beta ** 2)
            factor_contribs[f"factor_{f}"] = f_var
            systematic_var += f_var
            residual_var += f_var
        r_sq = min(1.0, systematic_var / max(sigma_p_sq, 1e-10))
        idio_var = max(0, sigma_p_sq - systematic_var)
        return FactorDecompositionResult(
            systematic_var=systematic_var,
            idiosyncratic_var=idio_var,
            factor_betas=factor_betas,
            factor_contributions=factor_contribs,
            r_squared=max(0, r_sq),
            n_factors=n_factors,
        )

    def stress_var(self, returns: np.ndarray, weights: np.ndarray,
                   factor_shocks: dict) -> StressVaRResult:
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        z = {0.90: 1.282, 0.95: 1.645, 0.975: 1.96, 0.99: 2.326}.get(
            self.confidence_level, 1.645)
        mu = np.mean(returns, axis=0)
        cov_mat = np.cov(returns.T)
        if cov_mat.ndim == 0:
            cov_mat = np.array([[float(cov_mat)]])
        sigma_p = float(np.sqrt(weights @ cov_mat @ weights))
        baseline_var = float(-np.sum(weights * mu) + z * sigma_p)
        stressed_mu = mu.copy()
        for key, shock in factor_shocks.items():
            idx = int(key.replace("factor_", "").replace("asset_", ""))
            if idx < len(stressed_mu):
                stressed_mu[idx] += shock
        stressed_returns = {f"asset_{i}": float(stressed_mu[i]) for i in range(len(stressed_mu))}
        stressed_portfolio_mean = float(np.sum(weights * stressed_mu))
        stressed_var = float(-stressed_portfolio_mean + z * sigma_p)
        return StressVaRResult(
            baseline_var=baseline_var,
            stressed_var=stressed_var,
            var_increase=float(stressed_var - baseline_var),
            stressed_returns=stressed_returns,
            scenario_name="custom_stress",
            factor_shocks=factor_shocks,
        )

    @staticmethod
    def marginal_var_contribution(returns: np.ndarray, weights: np.ndarray,
                                   confidence_level: float = 0.95) -> dict:
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        z = {0.90: 1.282, 0.95: 1.645, 0.975: 1.96, 0.99: 2.326}.get(
            confidence_level, 1.645)
        cov_mat = np.cov(returns.T)
        if cov_mat.ndim == 0:
            cov_mat = np.array([[float(cov_mat)]])
        sigma_p = float(np.sqrt(weights @ cov_mat @ weights))
        marginal = (cov_mat @ weights) / max(sigma_p, 1e-10)
        contributions = {f"asset_{i}": float(z * marginal[i]) for i in range(len(weights))}
        return {
            "marginal_var_contributions": contributions,
            "total_portfolio_sigma": sigma_p,
            "confidence_level": confidence_level,
        }