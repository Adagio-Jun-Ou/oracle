import numpy as np
from scipy.special import erfinv
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PortfolioResult:
    optimal_weights: dict[str, float]
    optimal_return: float
    optimal_volatility: float
    optimal_sharpe: float
    conservative_weights: dict[str, float]
    conservative_return: float
    conservative_volatility: float
    efficient_frontier: list[dict] = field(default_factory=list)
    individual_returns: dict[str, float] = field(default_factory=dict)
    individual_volatilities: dict[str, float] = field(default_factory=dict)
    n_simulations: int = 10000
    risk_free_rate: float = 0.05


class PortfolioOptimizer:
    def __init__(
        self,
        n_simulations: int = 10000,
        risk_free_rate: float = 0.05,
        seed: int = 42,
    ):
        self.n_simulations = n_simulations
        self.risk_free_rate = risk_free_rate
        self.seed = seed

    def optimize(
        self,
        returns_matrix: np.ndarray,
        asset_names: list[str],
        n_simulations: Optional[int] = None,
        risk_free_rate: Optional[float] = None,
    ) -> PortfolioResult:
        if n_simulations is None:
            n_simulations = self.n_simulations
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        n_assets = returns_matrix.shape[1]
        mean_returns = np.mean(returns_matrix, axis=0) * 252
        cov_matrix = np.cov(returns_matrix, rowvar=False) * 252

        np.random.seed(self.seed)

        best_sharpe = -np.inf
        best_weights_arr = np.ones(n_assets) / n_assets
        best_return = float(np.mean(mean_returns))
        best_vol = float(np.sqrt(np.mean(np.diag(cov_matrix))))

        min_var_vol = np.inf
        min_var_weights = np.ones(n_assets) / n_assets

        frontier = []
        for _ in range(n_simulations):
            weights = np.random.dirichlet(np.ones(n_assets))
            port_return = float(np.dot(weights, mean_returns))
            port_vol = float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))
            sharpe = (port_return - risk_free_rate) / (port_vol + 1e-10)

            frontier.append({
                "return": round(port_return, 4),
                "volatility": round(port_vol, 4),
                "sharpe": round(sharpe, 4),
            })

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights_arr = weights.copy()
                best_return = port_return
                best_vol = port_vol

            if port_vol < min_var_vol:
                min_var_vol = port_vol
                min_var_weights = weights.copy()

        optimal_weights = {name: round(float(best_weights_arr[i]), 4) for i, name in enumerate(asset_names)}
        conservative_weights = {name: round(float(min_var_weights[i]), 4) for i, name in enumerate(asset_names)}
        individual_returns = {name: round(float(mean_returns[i]), 4) for i, name in enumerate(asset_names)}
        individual_vols = {name: round(float(np.sqrt(cov_matrix[i][i])), 4) for i, name in enumerate(asset_names)}

        frontier_sample = frontier[::max(1, len(frontier) // 200)][:200]

        return PortfolioResult(
            optimal_weights=optimal_weights,
            optimal_return=round(best_return, 4),
            optimal_volatility=round(best_vol, 4),
            optimal_sharpe=round(best_sharpe, 4),
            conservative_weights=conservative_weights,
            conservative_return=round(float(np.dot(min_var_weights, mean_returns)), 4),
            conservative_volatility=round(min_var_vol, 4),
            efficient_frontier=frontier_sample,
            individual_returns=individual_returns,
            individual_volatilities=individual_vols,
            n_simulations=n_simulations,
            risk_free_rate=risk_free_rate,
        )

    def optimize_from_prices(
        self,
        price_data: dict[str, np.ndarray],
        n_simulations: Optional[int] = None,
        risk_free_rate: Optional[float] = None,
    ) -> PortfolioResult:
        names = list(price_data.keys())
        if len(names) < 2:
            raise ValueError("Need at least 2 assets for portfolio optimization")

        returns_dict = {}
        for name, prices in price_data.items():
            if len(prices) < 10:
                continue
            rets = np.diff(prices) / prices[:-1]
            returns_dict[name] = rets

        valid_names = list(returns_dict.keys())
        if len(valid_names) < 2:
            raise ValueError("Need at least 2 assets with sufficient data")

        min_len = min(len(returns_dict[n]) for n in valid_names)
        returns_matrix = np.column_stack([returns_dict[n][:min_len] for n in valid_names])

        result = self.optimize(
            returns_matrix=returns_matrix,
            asset_names=valid_names,
            n_simulations=n_simulations,
            risk_free_rate=risk_free_rate,
        )
        return result

    def capm_expected_return(
        self,
        asset_return: float,
        market_return: float,
        risk_free_rate: float,
        beta: float,
    ) -> float:
        return risk_free_rate + beta * (market_return - risk_free_rate)

    def beta(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
    ) -> float:
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 1.0
        cov_matrix = np.cov(asset_returns, market_returns)
        market_var = np.var(market_returns, ddof=1)
        if market_var == 0:
            return 1.0
        return float(cov_matrix[0, 1] / market_var)

    def information_ratio(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 0.0
        excess = portfolio_returns - benchmark_returns
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)
        if std_excess == 0:
            return 0.0
        return float(mean_excess / std_excess)

    def sortino_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
    ) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns * periods_per_year - risk_free_rate
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float(np.mean(excess)) if np.mean(excess) > 0 else 0.0
        downside_std = np.std(downside, ddof=1) * np.sqrt(periods_per_year)
        if downside_std == 0:
            return 0.0
        return float(np.mean(excess) / downside_std)

    def max_drawdown(self, prices: np.ndarray) -> float:
        if len(prices) < 2:
            return 0.0
        peak = prices[0]
        max_dd = 0.0
        for price in prices:
            if price > peak:
                peak = price
            dd = (price - peak) / peak
            if dd < max_dd:
                max_dd = dd
        return float(max_dd)

    def calmar_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
    ) -> float:
        if len(returns) < 10:
            return 0.0
        annual_return = float(np.mean(returns) * periods_per_year)
        prices = np.cumprod(1 + returns) * 100
        mdd = self.max_drawdown(prices)
        if abs(mdd) < 1e-10:
            return 0.0
        return float((annual_return - risk_free_rate) / abs(mdd))

    def var_historical(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        periods_per_year: int = 252,
    ) -> float:
        if len(returns) < 10:
            return 0.0
        sorted_returns = np.sort(returns)
        index = int((1 - confidence) * len(sorted_returns))
        index = max(0, min(index, len(sorted_returns) - 1))
        return float(sorted_returns[index] * np.sqrt(periods_per_year))

    def cvar_historical(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        periods_per_year: int = 252,
    ) -> float:
        if len(returns) < 10:
            return 0.0
        sorted_returns = np.sort(returns)
        index = int((1 - confidence) * len(sorted_returns))
        index = max(0, min(index, len(sorted_returns) - 1))
        tail = sorted_returns[:index + 1]
        if len(tail) == 0:
            return 0.0
        return float(np.mean(tail) * np.sqrt(periods_per_year))

    def kelly_criterion(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
    ) -> dict:
        if len(returns) < 20:
            return {"kelly_fraction": 0.0, "half_kelly": 0.0, "quarter_kelly": 0.0, "expected_growth_rate": 0.0}

        mu = float(np.mean(returns) * periods_per_year)
        sigma2 = float(np.var(returns, ddof=1) * periods_per_year)

        if sigma2 < 1e-10:
            return {"kelly_fraction": 0.0, "half_kelly": 0.0, "quarter_kelly": 0.0, "expected_growth_rate": 0.0}

        excess_mu = mu - risk_free_rate
        kelly = excess_mu / sigma2
        half_kelly = kelly / 2
        quarter_kelly = kelly / 4
        growth_rate = risk_free_rate + kelly * excess_mu - (kelly ** 2 * sigma2) / 2

        return {
            "kelly_fraction": round(float(np.clip(kelly, -2.0, 2.0)), 4),
            "half_kelly": round(float(np.clip(half_kelly, -1.0, 1.0)), 4),
            "quarter_kelly": round(float(np.clip(quarter_kelly, -0.5, 0.5)), 4),
            "expected_growth_rate": round(float(growth_rate), 4),
            "annualized_return": round(float(mu), 4),
            "annualized_variance": round(float(sigma2), 4),
        }

    def fibonacci_retracement(
        self,
        prices: np.ndarray,
        levels: Optional[list[float]] = None,
    ) -> dict:
        if len(prices) < 2:
            return {"error": "Insufficient data"}

        if levels is None:
            levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

        high = float(np.max(prices))
        low = float(np.min(prices))
        diff = high - low

        if diff < 1e-10:
            return {"error": "Price range too small"}

        retracements = {}
        for level in levels:
            price = high - diff * level
            retracements[f"level_{level:.3f}"] = round(price, 4)

        current_price = float(prices[-1])
        position_in_range = (current_price - low) / diff if diff > 0 else 0.5

        extension_levels = [1.272, 1.414, 1.618, 2.0, 2.618]
        extensions = {}
        for ext in extension_levels:
            ext_price = high + diff * (ext - 1.0)
            extensions[f"extension_{ext:.3f}"] = round(ext_price, 4)

        trend = "uptrend" if prices[-1] > prices[0] else "downtrend"
        key_levels = {
            "support_236": round(high - diff * 0.236, 4),
            "support_382": round(high - diff * 0.382, 4),
            "support_500": round(high - diff * 0.50, 4),
            "support_618": round(high - diff * 0.618, 4),
            "resistance_1618": round(high + diff * 0.618, 4),
        }

        nearest_level = "0.500"
        min_dist = float('inf')
        for level_name, level_price in retracements.items():
            dist = abs(current_price - level_price)
            if dist < min_dist:
                min_dist = dist
                nearest_level = level_name

        return {
            "high": round(high, 4),
            "low": round(low, 4),
            "current_price": round(current_price, 4),
            "range_pct": round(position_in_range, 4),
            "trend": trend,
            "retracements": retracements,
            "extensions": extensions,
            "key_levels": key_levels,
            "nearest_level": nearest_level,
            "nearest_price": round(retracements[nearest_level], 4),
        }

    def rolling_sharpe(
        self,
        returns: np.ndarray,
        window: int = 63,
        risk_free_rate: float = 0.05,
    ) -> list:
        if len(returns) < window:
            return []
        annual_factor = 252
        sharpes = []
        for i in range(window, len(returns)):
            window_returns = returns[i - window:i]
            mean_ret = float(np.mean(window_returns)) * annual_factor
            std_ret = float(np.std(window_returns, ddof=1)) * np.sqrt(annual_factor)
            sharpe = (mean_ret - risk_free_rate) / (std_ret + 1e-10)
            sharpes.append(round(sharpe, 4))
        return sharpes

    def downside_risk(
        self,
        returns: np.ndarray,
        target_return: float = 0.0,
        periods_per_year: int = 252,
    ) -> dict:
        if len(returns) < 2:
            return {"downside_deviation": 0.0, "downside_frequency": 0.0}

        downside = returns[returns < target_return]
        downside_freq = len(downside) / len(returns)
        if len(downside) == 0:
            return {"downside_deviation": 0.0, "downside_frequency": round(float(downside_freq), 4)}

        downside_dev = float(np.std(downside - target_return, ddof=1) * np.sqrt(periods_per_year))
        return {
            "downside_deviation": round(downside_dev, 4),
            "downside_frequency": round(float(downside_freq), 4),
            "avg_downside_loss": round(float(np.mean(downside)), 6),
            "max_downside_loss": round(float(np.min(downside)), 6),
        }

    def correlation_analysis(
        self,
        returns_matrix: np.ndarray,
        asset_names: list[str],
    ) -> dict:
        if returns_matrix.shape[1] != len(asset_names) or returns_matrix.shape[0] < 10:
            return {"error": "Insufficient data"}

        corr = np.corrcoef(returns_matrix, rowvar=False)
        pairs = []
        for i in range(len(asset_names)):
            for j in range(i + 1, len(asset_names)):
                pairs.append({
                    "pair": f"{asset_names[i]}_{asset_names[j]}",
                    "correlation": round(float(corr[i][j]), 4),
                    "strength": "strong_positive" if corr[i][j] > 0.7 else "moderate_positive" if corr[i][j] > 0.3 else "weak_positive" if corr[i][j] > 0 else "weak_negative" if corr[i][j] > -0.3 else "moderate_negative" if corr[i][j] > -0.7 else "strong_negative",
                })

        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "correlation_matrix": {asset_names[i]: {asset_names[j]: round(float(corr[i][j]), 4) for j in range(len(asset_names))} for i in range(len(asset_names))},
            "top_pairs": pairs[:5],
            "diversification_score": round(float(1.0 - np.mean(np.abs(corr[np.triu_indices(len(asset_names), k=1)]))), 4),
            "avg_correlation": round(float(np.mean(np.abs(corr[np.triu_indices(len(asset_names), k=1)]))), 4),
        }

    def black_litterman(
        self,
        market_weights: np.ndarray,
        covariance_matrix: np.ndarray,
        views: dict[int, float],
        view_confidence: float = 0.5,
        tau: float = 0.025,
        risk_aversion: float = 2.5,
    ) -> dict:
        n = len(market_weights)
        cov = np.array(covariance_matrix)
        if cov.shape != (n, n):
            raise ValueError(f"Covariance matrix shape {cov.shape} doesn't match {n} assets")

        implied_returns = risk_aversion * cov @ market_weights

        P = np.zeros((len(views), n))
        Q = np.zeros(len(views))
        for idx, (asset_idx, view_return) in enumerate(views.items()):
            P[idx, asset_idx] = 1.0
            Q[idx] = view_return

        omega = np.diag(np.full(len(views), 1.0 / view_confidence)) if len(views) > 0 else np.eye(0)
        omega = np.eye(len(views)) * (1.0 / max(view_confidence, 0.01)) if len(views) > 0 else np.zeros((0, 0))

        tau_cov = tau * cov
        if len(views) > 0:
            bl_returns = np.linalg.inv(
                np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(omega) @ P
            ) @ (np.linalg.inv(tau_cov) @ implied_returns + P.T @ np.linalg.inv(omega) @ Q)
        else:
            bl_returns = implied_returns

        bl_weights = np.linalg.inv(risk_aversion * cov) @ bl_returns
        bl_weights = np.maximum(bl_weights, 0)
        bl_weights = bl_weights / bl_weights.sum()

        asset_names = [f"Asset_{i}" for i in range(n)]
        return {
            "implied_returns": {asset_names[i]: round(float(implied_returns[i]), 6) for i in range(n)},
            "bl_returns": {asset_names[i]: round(float(bl_returns[i]), 6) for i in range(n)},
            "bl_weights": {asset_names[i]: round(float(bl_weights[i]), 4) for i in range(n)},
            "market_weights": {asset_names[i]: round(float(market_weights[i]), 4) for i in range(n)},
            "view_adjustment": {asset_names[i]: round(float(bl_returns[i] - implied_returns[i]), 6) for i in range(n)},
        }

    def historical_var_cvar(
        self,
        returns: np.ndarray,
        confidence_levels: list[float] = None,
        portfolio_weights: np.ndarray = None,
        n_simulations: int = 10000,
    ) -> dict:
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.975, 0.99]
        returns = np.array(returns)
        if len(returns) < 30:
            return {"error": "insufficient_data"}

        if portfolio_weights is not None:
            if len(portfolio_weights) == 1:
                portfolio_returns = returns
            else:
                portfolio_returns = returns @ portfolio_weights
        else:
            portfolio_returns = returns

        results = {}
        mu = float(np.mean(portfolio_returns))
        sigma = float(np.std(portfolio_returns, ddof=1))

        for cl in confidence_levels:
            hist_var = float(np.percentile(portfolio_returns, (1 - cl) * 100))
            hist_cvar = float(np.mean(portfolio_returns[portfolio_returns <= hist_var]))

            parametric_var = mu - sigma * float(np.sqrt(2) * erfinv(2 * cl - 1)) if cl > 0.5 else 0
            parametric_cvar = mu - sigma * float(np.sqrt(2) * erfinv(2 * cl - 1)) + sigma / (np.sqrt(2 * np.pi) * cl) * np.exp(-((np.sqrt(2) * erfinv(2 * cl - 1)) ** 2) / 2)

            mc_returns = np.random.normal(mu, sigma, n_simulations)
            mc_var = float(np.percentile(mc_returns, (1 - cl) * 100))
            mc_cvar = float(np.mean(mc_returns[mc_returns <= mc_var]))

            results[f"var_{int(cl * 100)}"] = {
                "historical": round(abs(hist_var), 6),
                "parametric": round(abs(parametric_var), 6),
                "monte_carlo": round(abs(mc_var), 6),
                "annualized_pct": round(abs(hist_var) * 252 * 100, 2),
            }
            results[f"cvar_{int(cl * 100)}"] = {
                "historical": round(abs(hist_cvar), 6),
                "parametric": round(abs(parametric_cvar), 6),
                "monte_carlo": round(abs(mc_cvar), 6),
                "annualized_pct": round(abs(hist_cvar) * 252 * 100, 2),
            }

        tail_events = int(np.sum(portfolio_returns < np.percentile(portfolio_returns, 5)))
        return {
            "methods": results,
            "summary": {
                "daily_var_95": results.get("var_95", {}).get("historical", 0),
                "daily_cvar_95": results.get("cvar_95", {}).get("historical", 0),
                "annualized_var_95_pct": results.get("var_95", {}).get("annualized_pct", 0),
                "annualized_cvar_95_pct": results.get("cvar_95", {}).get("annualized_pct", 0),
                "max_daily_loss": round(float(abs(np.min(portfolio_returns))), 6),
                "tail_events_5pct": tail_events,
                "n_observations": len(portfolio_returns),
            },
        }

    def risk_parity(
        self,
        returns_matrix: np.ndarray,
        asset_names: list[str] = None,
        n_iterations: int = 1000,
    ) -> dict:
        n_assets = returns_matrix.shape[1]
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]

        cov = np.cov(returns_matrix.T)
        if n_assets == 1:
            return {asset_names[0]: 1.0}

        weights = np.ones(n_assets) / n_assets
        step_size = 0.01

        for _ in range(n_iterations):
            port_var = weights @ cov @ weights
            marginal_risk = cov @ weights / np.sqrt(port_var) if port_var > 0 else np.ones(n_assets) / n_assets
            risk_contributions = weights * marginal_risk
            total_risk = np.sum(risk_contributions)
            if total_risk == 0:
                break
            target_risk = total_risk / n_assets
            adjustments = (target_risk - risk_contributions) / total_risk
            weights = weights + step_size * adjustments
            weights = np.maximum(weights, 0.001)
            weights = weights / np.sum(weights)

        port_var_final = weights @ cov @ weights
        marginal_risk_final = cov @ weights / np.sqrt(port_var_final) if port_var_final > 0 else np.ones(n_assets)
        risk_contributions_final = weights * marginal_risk_final

        return {
            "weights": {asset_names[i]: round(float(weights[i]), 4) for i in range(n_assets)},
            "risk_contribution": {asset_names[i]: round(float(risk_contributions_final[i] / np.sum(risk_contributions_final)), 4) for i in range(n_assets)},
            "portfolio_volatility": round(float(np.sqrt(port_var_final) * np.sqrt(252) * 100), 2),
            "portfolio_return_ann": round(float(np.mean(returns_matrix @ weights) * 252 * 100), 2),
        }

    def rolling_drawdown(
        self,
        prices: np.ndarray,
        window: int = 21,
    ) -> dict:
        if len(prices) < window + 1:
            return {"error": "insufficient_data", "rolling_drawdown": [], "current_drawdown": 0, "max_drawdown": 0}

        rolling_dd = []
        global_peak = prices[0]
        global_max_dd = 0.0

        for i in range(len(prices)):
            if prices[i] > global_peak:
                global_peak = prices[i]
            dd = (prices[i] - global_peak) / global_peak
            if dd < global_max_dd:
                global_max_dd = dd

            if i >= window:
                local_peak = prices[i - window]
                for j in range(i - window + 1, i + 1):
                    if prices[j] > local_peak:
                        local_peak = prices[j]
                local_dd = float((prices[i] - local_peak) / local_peak) if local_peak > 0 else 0.0
                rolling_dd.append({"day": i, "drawdown_pct": round(local_dd * 100, 3)})

        current_dd = float((prices[-1] - global_peak) / global_peak) if global_peak > 0 else 0.0

        recovery_periods = []
        in_dd = False
        dd_start = 0
        for i in range(1, len(prices)):
            local_peak = prices[0]
            for j in range(i):
                if prices[j] > local_peak:
                    local_peak = prices[j]
            if prices[i] < local_peak and not in_dd:
                in_dd = True
                dd_start = i
            elif prices[i] >= local_peak and in_dd:
                recovery_periods.append(i - dd_start)
                in_dd = False

        return {
            "rolling_drawdown": rolling_dd[-120:] if len(rolling_dd) > 120 else rolling_dd,
            "current_drawdown_pct": round(current_dd * 100, 2),
            "max_drawdown_pct": round(global_max_dd * 100, 2),
            "avg_recovery_days": round(float(np.mean(recovery_periods)), 1) if recovery_periods else 0,
            "max_recovery_days": max(recovery_periods) if recovery_periods else 0,
            "n_drawdown_periods": len(recovery_periods),
        }

    def shrinkage_covariance(
        self,
        returns_matrix: np.ndarray,
        shrinkage_target: str = "identity",
        asset_names: list[str] = None,
    ) -> dict:
        n, p = returns_matrix.shape
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(p)]

        S = np.cov(returns_matrix.T)
        if shrinkage_target == "identity":
            mu = np.trace(S) / p
            F = mu * np.eye(p)
        else:
            F = np.outer(np.diag(S), np.ones(p)) * np.eye(p)
            np.fill_diagonal(F, np.diag(S))

        delta = S - F
        denom = np.sum(delta ** 2)

        if denom == 0:
            shrinkage_intensity = 0.0
        else:
            X = returns_matrix - returns_matrix.mean(axis=0)
            gamma = 0.0
            for k in range(n):
                diff = np.outer(X[k], X[k]) - delta
                gamma += np.sum(diff ** 2)
            gamma /= (n ** 2 * denom)
            alpha = max(0, min(1, gamma))

        shrinkage_intensity = alpha if 'alpha' in dir() else 0.0
        shrunk = (1 - shrinkage_intensity) * S + shrinkage_intensity * F

        return {
            "sample_covariance": {asset_names[i]: {asset_names[j]: round(float(S[i][j]), 6) for j in range(p)} for i in range(p)},
            "shrunk_covariance": {asset_names[i]: {asset_names[j]: round(float(shrunk[i][j]), 6) for j in range(p)} for i in range(p)},
            "shrinkage_intensity": round(float(shrinkage_intensity), 4),
            "n_observations": n,
            "n_assets": p,
        }

    def mean_variance_optimize(
        self,
        returns_matrix: np.ndarray,
        asset_names: list[str] = None,
        risk_free_rate: float = 0.05,
        target_return: float = None,
    ) -> dict:
        n, p = returns_matrix.shape
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(p)]

        if p == 1:
            ret = float(np.mean(returns_matrix) * 252)
            vol = float(np.std(returns_matrix, ddof=1) * np.sqrt(252))
            return {
                "tangency_portfolio": {asset_names[0]: 1.0},
                "tangency_return": round(ret, 4),
                "tangency_volatility": round(vol, 4),
                "tangency_sharpe": round((ret - risk_free_rate) / vol, 4) if vol > 0 else 0,
                "minimum_variance_portfolio": {asset_names[0]: 1.0},
                "min_var_return": round(ret, 4),
                "min_var_volatility": round(vol, 4),
                "efficient_frontier": [{"return": round(ret, 4), "volatility": round(vol, 4)}],
                "expected_returns": {asset_names[0]: round(ret, 4)},
            }

        mu = np.mean(returns_matrix, axis=0) * 252
        cov = np.cov(returns_matrix.T)
        cov_inv = np.linalg.inv(cov)

        ones = np.ones(p)
        a = ones @ cov_inv @ ones
        b = ones @ cov_inv @ mu
        c = mu @ cov_inv @ mu
        det = a * c - b ** 2

        if det == 0:
            w_tan = ones / p
        else:
            w_tan = (c * cov_inv @ ones - b * cov_inv @ mu) / det

        w_tan = np.maximum(w_tan, 0)
        w_tan = w_tan / w_tan.sum()

        tan_return = float(w_tan @ mu)
        tan_vol = float(np.sqrt(w_tan @ cov @ w_tan * 252))
        tan_sharpe = (tan_return - risk_free_rate) / tan_vol if tan_vol > 0 else 0

        w_minvar = cov_inv @ ones
        w_minvar = np.maximum(w_minvar, 0)
        w_minvar = w_minvar / w_minvar.sum()

        min_var_return = float(w_minvar @ mu)
        min_var_vol = float(np.sqrt(w_minvar @ cov @ w_minvar * 252))

        if target_return is not None:
            lam = (target_return - b / a) / (c - b ** 2 / a) if det != 0 else 0
            w_target = w_minvar + lam * (cov_inv @ mu - (b / a) * cov_inv @ ones)
            w_target = np.maximum(w_target, 0)
            if w_target.sum() > 0:
                w_target = w_target / w_target.sum()
            target_port_return = float(w_target @ mu)
            target_port_vol = float(np.sqrt(w_target @ cov @ w_target * 252))
        else:
            w_target = None
            target_port_return = None
            target_port_vol = None

        frontier_returns = np.linspace(min_var_return, tan_return * 1.5, 50)
        frontier_vols = []
        for r in frontier_returns:
            lam = (r - b / a) / (c - b ** 2 / a) if det != 0 and a > 0 else 0
            w = w_minvar + lam * (cov_inv @ mu - (b / a) * cov_inv @ ones) if det != 0 else w_minvar
            w = np.maximum(w, 0)
            if w.sum() > 0:
                w = w / w.sum()
            vol = float(np.sqrt(w @ cov @ w * 252))
            frontier_vols.append(vol)

        result = {
            "tangency_portfolio": {asset_names[i]: round(float(w_tan[i]), 4) for i in range(p)},
            "tangency_return": round(tan_return, 4),
            "tangency_volatility": round(tan_vol, 4),
            "tangency_sharpe": round(tan_sharpe, 4),
            "minimum_variance_portfolio": {asset_names[i]: round(float(w_minvar[i]), 4) for i in range(p)},
            "min_var_return": round(min_var_return, 4),
            "min_var_volatility": round(min_var_vol, 4),
            "efficient_frontier": [
                {"return": round(float(r), 4), "volatility": round(float(v), 4)}
                for r, v in zip(frontier_returns, frontier_vols)
            ],
            "expected_returns": {asset_names[i]: round(float(mu[i]), 4) for i in range(p)},
        }

        if target_return is not None and w_target is not None:
            result["target_portfolio"] = {asset_names[i]: round(float(w_target[i]), 4) for i in range(p)}
            result["target_return"] = round(target_port_return, 4)
            result["target_volatility"] = round(target_port_vol, 4)

        return result