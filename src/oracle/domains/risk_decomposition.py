import numpy as np
from typing import Optional


class RiskDecomposition:

    def __init__(self):
        self.name = "risk_decomposition"

    @staticmethod
    def component_var(
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> dict:
        n_assets = returns_matrix.shape[1]
        portfolio_returns = returns_matrix @ weights
        if method == "historical":
            sorted_returns = np.sort(portfolio_returns)
            idx = int((1 - confidence) * len(sorted_returns))
            portfolio_var = -sorted_returns[idx]
        elif method == "parametric":
            mu = np.mean(portfolio_returns)
            sigma = np.std(portfolio_returns, ddof=1)
            from scipy.stats import norm
            portfolio_var = -(mu - norm.ppf(1 - confidence) * sigma)
        else:
            portfolio_var = 0.0
        component_vars = []
        marginal_vars = []
        for i in range(n_assets):
            marginal = np.cov(returns_matrix[:, i], portfolio_returns)[0, 1]
            marginal_vars.append(round(float(marginal), 6))
            comp_var = weights[i] * marginal
            component_vars.append(round(float(comp_var), 6))
        total_marginal = sum(marginal_vars)
        pct_contributions = []
        for i in range(n_assets):
            if total_marginal != 0:
                pct = (weights[i] * marginal_vars[i]) / (sum(w * m for w, m in zip(weights, marginal_vars)))
            else:
                pct = 0.0
            pct_contributions.append(round(float(pct) * 100, 2))
        return {
            "portfolio_var": round(float(portfolio_var), 6),
            "component_var": component_vars,
            "marginal_var": marginal_vars,
            "pct_contribution": pct_contributions,
            "weights": [round(float(w), 4) for w in weights],
            "confidence": confidence,
            "method": method,
        }

    @staticmethod
    def factor_attribution(
        returns_matrix: np.ndarray,
        factor_returns: np.ndarray,
        weights: np.ndarray,
        factor_names: Optional[list] = None,
    ) -> dict:
        n_periods = returns_matrix.shape[0]
        n_assets = returns_matrix.shape[1]
        n_factors = factor_returns.shape[1]
        if factor_names is None:
            factor_names = [f"Factor_{i+1}" for i in range(n_factors)]
        portfolio_returns = returns_matrix @ weights
        X = np.column_stack([np.ones(n_periods), factor_returns])
        beta, residuals, rank, sv = np.linalg.lstsq(X, portfolio_returns, rcond=None)
        alpha = float(beta[0])
        factor_betas = beta[1:]
        factor_contributions = []
        for i in range(n_factors):
            mean_factor = np.mean(factor_returns[:, i])
            contribution = factor_betas[i] * mean_factor
            factor_contributions.append(round(float(contribution), 6))
        explained = np.sum(factor_contributions)
        total_return = float(np.mean(portfolio_returns))
        residual_contrib = total_return - alpha - explained
        r_squared = 1.0 - np.var(residuals) / np.var(portfolio_returns) if len(residuals) > 0 and np.var(portfolio_returns) > 0 else 0.0
        return {
            "alpha": round(alpha, 6),
            "total_period_return": round(total_return, 6),
            "factor_betas": {name: round(float(b), 6) for name, b in zip(factor_names, factor_betas)},
            "factor_contributions": {name: round(float(c), 6) for name, c in zip(factor_names, factor_contributions)},
            "residual_contribution": round(float(residual_contrib), 6),
            "r_squared": round(float(r_squared), 4),
            "factor_names": factor_names,
        }

    @staticmethod
    def risk_budget(
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> dict:
        n_assets = returns_matrix.shape[1]
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix]])
        portfolio_var = float(weights @ cov_matrix @ weights)
        portfolio_vol = float(np.sqrt(portfolio_var))
        marginal_contribs = cov_matrix @ weights
        risk_contribs = weights * marginal_contribs
        if portfolio_var > 0:
            pct_risk = risk_contribs / portfolio_var
        else:
            pct_risk = np.zeros(n_assets)
        portfolio_return = float(np.mean(returns_matrix @ weights))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0.0
        return {
            "portfolio_volatility": round(portfolio_vol, 6),
            "portfolio_variance": round(float(portfolio_var), 6),
            "asset_risk_contribution": [round(float(rc), 6) for rc in risk_contribs],
            "pct_risk_contribution": [round(float(p), 4) for p in pct_risk],
            "weights": [round(float(w), 4) for w in weights],
            "portfolio_return": round(portfolio_return, 6),
            "sharpe_ratio": round(sharpe, 4),
        }

    @staticmethod
    def tracking_error(
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        annualize_factor: int = 252,
    ) -> dict:
        active_returns = portfolio_returns - benchmark_returns
        te = float(np.std(active_returns, ddof=1) * np.sqrt(annualize_factor))
        tracking_error_bps = te * 10000
        ir = float(np.mean(active_returns) / (np.std(active_returns, ddof=1))) if np.std(active_returns, ddof=1) > 0 else 0.0
        annualized_ir = ir * np.sqrt(annualize_factor)
        return {
            "tracking_error": round(te, 6),
            "tracking_error_bps": round(tracking_error_bps, 2),
            "information_ratio": round(ir, 4),
            "annualized_ir": round(annualized_ir, 4),
            "mean_active_return": round(float(np.mean(active_returns)), 6),
            "active_return_volatility": round(float(np.std(active_returns, ddof=1)), 6),
        }

    @staticmethod
    def stress_decomposition(
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        scenarios: Optional[dict] = None,
    ) -> dict:
        if scenarios is None:
            scenarios = {
                "market_crash": {"equity_shock": -0.20, "vol_mult": 3.0},
                "rate_hike": {"rate_shock_bps": 200},
                "credit_crisis": {"credit_spread_bps": 300},
                "liquidity_crisis": {"liquidity_discount": 0.10},
            }
        portfolio_returns = returns_matrix @ weights
        mu = np.mean(portfolio_returns)
        sigma = np.std(portfolio_returns, ddof=1)
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix]])
        results = {}
        for name, shock in scenarios.items():
            if "equity_shock" in shock:
                shocked = mu + shock["equity_shock"]
                shocked_sigma = sigma * shock.get("vol_mult", 1.0)
            elif "rate_shock_bps" in shock:
                rate_move = -shock["rate_shock_bps"] / 10000
                shocked = mu + rate_move * 0.5
                shocked_sigma = sigma * 1.5
            elif "credit_spread_bps" in shock:
                spread = shock["credit_spread_bps"] / 10000
                shocked = mu - spread
                shocked_sigma = sigma * 2.0
            elif "liquidity_discount" in shock:
                discount = shock["liquidity_discount"]
                shocked = mu - discount
                shocked_sigma = sigma * 2.5
            else:
                shocked = mu
                shocked_sigma = sigma
            results[name] = {
                "stressed_return": round(float(shocked), 6),
                "stressed_vol": round(float(shocked_sigma), 6),
                "loss_vs_baseline": round(float(shocked - mu), 6),
                "vol_increase": round(float((shocked_sigma / sigma - 1) * 100) if sigma > 0 else 0, 2),
            }
        return {
            "baseline_return": round(float(mu), 6),
            "baseline_vol": round(float(sigma), 6),
            "scenarios": results,
        }