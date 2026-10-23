import pytest
import numpy as np
from oracle.optimization.portfolio import PortfolioOptimizer, PortfolioResult


class TestPortfolioOptimizer:
    def setup_method(self):
        self.optimizer = PortfolioOptimizer(n_simulations=500, seed=42)
        np.random.seed(42)
        self.returns_matrix = np.random.randn(252, 4) * 0.01 + 0.0003
        self.asset_names = ["sp500", "gold", "treasury", "bitcoin"]
        self.price_data = {
            "sp500": np.cumprod(1 + np.random.randn(100) * 0.01 + 0.0003) * 100,
            "gold": np.cumprod(1 + np.random.randn(100) * 0.008) * 100,
            "treasury": np.cumprod(1 + np.random.randn(100) * 0.002 + 0.0001) * 100,
        }

    def test_init(self):
        opt = PortfolioOptimizer()
        assert opt.n_simulations == 10000
        assert opt.risk_free_rate == 0.05
        assert opt.seed == 42

    def test_init_custom(self):
        opt = PortfolioOptimizer(n_simulations=5000, risk_free_rate=0.03, seed=123)
        assert opt.n_simulations == 5000
        assert opt.risk_free_rate == 0.03

    def test_optimize_basic(self):
        result = self.optimizer.optimize(self.returns_matrix, self.asset_names)
        assert isinstance(result, PortfolioResult)
        assert len(result.optimal_weights) == 4
        assert abs(sum(result.optimal_weights.values()) - 1.0) < 0.01
        assert result.optimal_return != 0 or result.optimal_volatility != 0
        assert isinstance(result.optimal_sharpe, float)
        assert result.n_simulations == 500

    def test_optimal_weights_sum_to_one(self):
        result = self.optimizer.optimize(self.returns_matrix, self.asset_names)
        total = sum(result.optimal_weights.values())
        assert abs(total - 1.0) < 0.02

    def test_conservative_weights(self):
        result = self.optimizer.optimize(self.returns_matrix, self.asset_names)
        assert len(result.conservative_weights) == 4
        total = sum(result.conservative_weights.values())
        assert abs(total - 1.0) < 0.02

    def test_efficient_frontier(self):
        result = self.optimizer.optimize(self.returns_matrix, self.asset_names)
        assert len(result.efficient_frontier) > 0
        assert len(result.efficient_frontier) <= 200
        for point in result.efficient_frontier:
            assert "return" in point
            assert "volatility" in point
            assert "sharpe" in point

    def test_individual_metrics(self):
        result = self.optimizer.optimize(self.returns_matrix, self.asset_names)
        assert len(result.individual_returns) == 4
        assert len(result.individual_volatilities) == 4

    def test_optimize_from_prices(self):
        result = self.optimizer.optimize_from_prices(self.price_data)
        assert isinstance(result, PortfolioResult)
        assert len(result.optimal_weights) >= 2

    def test_optimize_from_prices_insufficient_data(self):
        with pytest.raises(ValueError):
            self.optimizer.optimize_from_prices({"a": np.array([1.0, 2.0])})

    def test_optimize_from_prices_single_asset(self):
        with pytest.raises(ValueError):
            self.optimizer.optimize_from_prices({"a": np.array([1.0] * 10)})

    def test_capm_expected_return(self):
        ret = self.optimizer.capm_expected_return(0.1, 0.08, 0.05, 1.2)
        expected = 0.05 + 1.2 * (0.08 - 0.05)
        assert abs(ret - expected) < 1e-10

    def test_beta(self):
        asset = np.array([0.01, -0.02, 0.03, 0.01, -0.01])
        market = np.array([0.02, -0.01, 0.02, 0.01, 0.00])
        beta = self.optimizer.beta(asset, market)
        assert isinstance(beta, float)
        assert beta > 0

    def test_beta_mismatched_lengths(self):
        assert self.optimizer.beta(np.array([0.01]), np.array([0.01, 0.02])) == 1.0

    def test_information_ratio(self):
        port = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        bench = np.array([0.005, 0.015, -0.005, 0.02, 0.005])
        ir = self.optimizer.information_ratio(port, bench)
        assert isinstance(ir, float)

    def test_information_ratio_mismatched(self):
        assert self.optimizer.information_ratio(np.array([0.01]), np.array([0.01, 0.02])) == 0.0

    def test_sortino_ratio(self):
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.005, -0.015, 0.01])
        sr = self.optimizer.sortino_ratio(returns)
        assert isinstance(sr, float)

    def test_sortino_all_positive(self):
        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02])
        sr = self.optimizer.sortino_ratio(returns)
        assert isinstance(sr, float)

    def test_max_drawdown(self):
        prices = np.array([100, 110, 105, 115, 95, 120])
        mdd = self.optimizer.max_drawdown(prices)
        assert mdd < 0
        expected = (95 - 115) / 115
        assert abs(mdd - expected) < 0.05

    def test_max_drawdown_always_up(self):
        prices = np.array([100, 110, 120, 130, 140])
        mdd = self.optimizer.max_drawdown(prices)
        assert mdd == 0.0

    def test_calmar_ratio(self):
        returns = np.random.randn(100) * 0.01 + 0.0003
        cr = self.optimizer.calmar_ratio(returns)
        assert isinstance(cr, float)

    def test_var_historical(self):
        returns = np.random.randn(252) * 0.02
        var = self.optimizer.var_historical(returns)
        assert isinstance(var, float)

    def test_var_confidence_levels(self):
        returns = np.random.randn(252) * 0.02
        var_95 = self.optimizer.var_historical(returns, confidence=0.95)
        var_99 = self.optimizer.var_historical(returns, confidence=0.99)
        assert var_99 <= var_95

    def test_cvar_historical(self):
        returns = np.random.randn(252) * 0.02
        cvar = self.optimizer.cvar_historical(returns)
        assert isinstance(cvar, float)
        var = self.optimizer.var_historical(returns)
        assert cvar <= var

    def test_short_data_returns_zero(self):
        assert self.optimizer.var_historical(np.array([0.01])) == 0.0
        assert self.optimizer.cvar_historical(np.array([0.01])) == 0.0
        assert self.optimizer.calmar_ratio(np.array([0.01])) == 0.0

    def test_custom_n_simulations(self):
        result = self.optimizer.optimize(self.returns_matrix, self.asset_names, n_simulations=100)
        assert result.n_simulations == 100

    def test_custom_risk_free_rate(self):
        result = self.optimizer.optimize(
            self.returns_matrix, self.asset_names, risk_free_rate=0.03
        )
        assert result.risk_free_rate == 0.03

    def test_kelly_criterion(self):
        returns = np.random.randn(252) * 0.02 + 0.0005
        kelly = self.optimizer.kelly_criterion(returns)
        assert "kelly_fraction" in kelly
        assert "half_kelly" in kelly
        assert "quarter_kelly" in kelly
        assert "expected_growth_rate" in kelly
        assert "annualized_return" in kelly
        assert "annualized_variance" in kelly
        assert abs(kelly["half_kelly"] - kelly["kelly_fraction"] / 2) < 0.01

    def test_kelly_short_data(self):
        returns = np.array([0.01])
        kelly = self.optimizer.kelly_criterion(returns)
        assert kelly["kelly_fraction"] == 0.0
        assert kelly["half_kelly"] == 0.0

    def test_kelly_positive_returns(self):
        returns = np.random.randn(252) * 0.01 + 0.001
        kelly = self.optimizer.kelly_criterion(returns)
        assert kelly["kelly_fraction"] != 0 or kelly["expected_growth_rate"] >= 0

    def test_fibonacci_retracement(self):
        prices = np.array([100, 105, 110, 108, 112, 115, 113, 118, 120, 117])
        fib = self.optimizer.fibonacci_retracement(prices)
        assert "high" in fib
        assert "low" in fib
        assert "current_price" in fib
        assert "retracements" in fib
        assert "extensions" in fib
        assert "trend" in fib
        assert fib["high"] >= fib["low"]
        assert fib["retracements"]["level_0.000"] == fib["high"]

    def test_fibonacci_custom_levels(self):
        prices = np.array([100, 110, 105, 115, 108, 120])
        fib = self.optimizer.fibonacci_retracement(prices, levels=[0.0, 0.382, 0.618, 1.0])
        assert "level_0.000" in fib["retracements"]
        assert "level_0.382" in fib["retracements"]
        assert "level_0.618" in fib["retracements"]
        assert "level_1.000" in fib["retracements"]

    def test_fibonacci_key_levels(self):
        prices = np.cumprod(1 + np.random.randn(100) * 0.01 + 0.0003) * 100
        fib = self.optimizer.fibonacci_retracement(prices)
        assert "key_levels" in fib
        assert "support_382" in fib["key_levels"]
        assert "support_618" in fib["key_levels"]

    def test_fibonacci_short_data(self):
        prices = np.array([100.0])
        fib = self.optimizer.fibonacci_retracement(prices)
        assert "error" in fib

    def test_rolling_sharpe(self):
        returns = np.random.randn(252) * 0.02 + 0.001
        sharpes = self.optimizer.rolling_sharpe(returns, window=63)
        assert isinstance(sharpes, list)
        assert len(sharpes) == 252 - 63

    def test_rolling_sharpe_short_data(self):
        returns = np.random.randn(10) * 0.01
        sharpes = self.optimizer.rolling_sharpe(returns)
        assert sharpes == []

    def test_downside_risk(self):
        returns = np.random.randn(252) * 0.02
        ds = self.optimizer.downside_risk(returns)
        assert "downside_deviation" in ds
        assert "downside_frequency" in ds
        assert "avg_downside_loss" in ds
        assert "max_downside_loss" in ds

    def test_downside_risk_all_positive(self):
        returns = np.abs(np.random.randn(252)) * 0.01
        ds = self.optimizer.downside_risk(returns)
        assert ds["downside_deviation"] == 0.0

    def test_downside_risk_short_data(self):
        returns = np.array([0.01])
        ds = self.optimizer.downside_risk(returns)
        assert ds["downside_deviation"] == 0.0

    def test_correlation_analysis(self):
        np.random.seed(42)
        returns_matrix = np.random.randn(100, 3)
        names = ["A", "B", "C"]
        result = self.optimizer.correlation_analysis(returns_matrix, names)
        assert "correlation_matrix" in result
        assert "top_pairs" in result
        assert "diversification_score" in result
        assert "avg_correlation" in result
        assert len(result["top_pairs"]) <= 3

    def test_correlation_analysis_insufficient_data(self):
        result = self.optimizer.correlation_analysis(np.array([[1, 2]]), ["A", "B"])
        assert "error" in result

    def test_black_litterman_basic(self):
        np.random.seed(42)
        n = 4
        market_weights = np.array([0.4, 0.3, 0.2, 0.1])
        cov = np.eye(n) * 0.04
        cov[0, 1] = cov[1, 0] = 0.01
        cov[0, 2] = cov[2, 0] = 0.005
        views = {0: 0.10, 2: 0.05}
        result = self.optimizer.black_litterman(market_weights, cov, views, view_confidence=0.5)
        assert "implied_returns" in result
        assert "bl_returns" in result
        assert "bl_weights" in result
        assert "market_weights" in result
        assert "view_adjustment" in result
        assert len(result["bl_weights"]) == n
        assert abs(sum(result["bl_weights"].values()) - 1.0) < 0.1

    def test_black_litterman_no_views(self):
        n = 3
        market_weights = np.array([0.5, 0.3, 0.2])
        cov = np.eye(n) * 0.03
        result = self.optimizer.black_litterman(market_weights, cov, {}, view_confidence=0.5)
        assert "bl_returns" in result
        assert len(result["bl_weights"]) == n
        for k, v in result["view_adjustment"].items():
            assert v == 0 or v == pytest.approx(0, abs=1e-10)

    def test_historical_var_cvar_basic(self):
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01
        result = self.optimizer.historical_var_cvar(returns)
        assert "methods" in result
        assert "summary" in result
        assert "var_95" in result["methods"]
        assert "cvar_95" in result["methods"]
        assert "daily_var_95" in result["summary"]
        assert "daily_cvar_95" in result["summary"]
        assert result["summary"]["n_observations"] == 252

    def test_historical_var_cvar_with_weights(self):
        np.random.seed(42)
        returns_matrix = np.random.randn(252, 3) * 0.01
        weights = np.array([0.5, 0.3, 0.2])
        result = self.optimizer.historical_var_cvar(returns_matrix, portfolio_weights=weights)
        assert "methods" in result
        assert result["summary"]["n_observations"] == 252

    def test_historical_var_cvar_custom_confidence(self):
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01
        result = self.optimizer.historical_var_cvar(returns, confidence_levels=[0.95, 0.99])
        assert "var_95" in result["methods"]
        assert "var_99" in result["methods"]
        assert "cvar_95" in result["methods"]
        assert "cvar_99" in result["methods"]

    def test_historical_var_cvar_insufficient_data(self):
        returns = np.array([0.01, 0.02])
        result = self.optimizer.historical_var_cvar(returns)
        assert "error" in result

    def test_risk_parity_basic(self):
        np.random.seed(42)
        returns_matrix = np.random.randn(252, 4) * 0.01
        result = self.optimizer.risk_parity(returns_matrix, ["A", "B", "C", "D"])
        assert "weights" in result
        assert "risk_contribution" in result
        assert "portfolio_volatility" in result
        assert "portfolio_return_ann" in result
        assert len(result["weights"]) == 4
        assert abs(sum(result["weights"].values()) - 1.0) < 0.1

    def test_risk_parity_single_asset(self):
        returns_matrix = np.random.randn(252, 1) * 0.01
        result = self.optimizer.risk_parity(returns_matrix, ["A"])
        assert "A" in result
        assert result["A"] == 1.0

    def test_risk_parity_equal_risk_contribution(self):
        np.random.seed(42)
        n = 3
        returns_matrix = np.random.randn(500, n) * 0.01
        result = self.optimizer.risk_parity(returns_matrix, ["A", "B", "C"])
        rc = list(result["risk_contribution"].values())
        for rc_val in rc:
            assert abs(rc_val - 1.0 / n) < 0.15

    def test_rolling_drawdown_basic(self):
        np.random.seed(42)
        prices = np.cumprod(1 + np.random.randn(100) * 0.02) * 100
        result = self.optimizer.rolling_drawdown(prices)
        assert "rolling_drawdown" in result
        assert "current_drawdown_pct" in result
        assert "max_drawdown_pct" in result
        assert "avg_recovery_days" in result
        assert "max_recovery_days" in result
        assert len(result["rolling_drawdown"]) > 0

    def test_rolling_drawdown_insufficient_data(self):
        prices = np.array([100.0, 101.0])
        result = self.optimizer.rolling_drawdown(prices)
        assert "error" in result

    def test_rolling_drawdown_always_up(self):
        prices = np.arange(100, 200, 1, dtype=float)
        result = self.optimizer.rolling_drawdown(prices, window=5)
        assert result["current_drawdown_pct"] == 0.0
        assert result["max_drawdown_pct"] == 0.0

    def test_shrinkage_covariance_basic(self):
        np.random.seed(42)
        returns_matrix = np.random.randn(100, 3) * 0.01
        result = self.optimizer.shrinkage_covariance(returns_matrix, asset_names=["A", "B", "C"])
        assert "sample_covariance" in result
        assert "shrunk_covariance" in result
        assert "shrinkage_intensity" in result
        assert "n_observations" in result
        assert "n_assets" in result
        assert result["n_assets"] == 3

    def test_shrinkage_covariance_custom_names(self):
        np.random.seed(42)
        returns_matrix = np.random.randn(50, 2) * 0.01
        result = self.optimizer.shrinkage_covariance(
            returns_matrix, shrinkage_target="diagonal", asset_names=["X", "Y"]
        )
        assert "X" in result["sample_covariance"]
        assert result["n_assets"] == 2

    def test_mean_variance_optimize_basic(self):
        np.random.seed(42)
        returns_matrix = np.random.randn(252, 4) * 0.01 + 0.0003
        result = self.optimizer.mean_variance_optimize(returns_matrix, ["SPY", "GLD", "TLT", "QQQ"])
        assert "tangency_portfolio" in result
        assert "minimum_variance_portfolio" in result
        assert "tangency_return" in result
        assert "tangency_volatility" in result
        assert "tangency_sharpe" in result
        assert "min_var_return" in result
        assert "min_var_volatility" in result
        assert "efficient_frontier" in result
        assert len(result["efficient_frontier"]) == 50
        assert sum(result["tangency_portfolio"].values()) == pytest.approx(1.0, abs=0.05)

    def test_mean_variance_optimize_with_target(self):
        np.random.seed(42)
        returns_matrix = np.random.randn(252, 3) * 0.01 + 0.0003
        result = self.optimizer.mean_variance_optimize(
            returns_matrix, ["A", "B", "C"], target_return=0.10
        )
        assert "target_portfolio" in result
        assert result["target_return"] is not None
        assert result["target_volatility"] is not None

    def test_mean_variance_optimize_single_asset(self):
        np.random.seed(42)
        returns_matrix = np.random.randn(100, 1) * 0.01 + 0.0003
        result = self.optimizer.mean_variance_optimize(returns_matrix, ["A"])
        assert result["tangency_portfolio"]["A"] == 1.0

    def test_rolling_drawdown_different_windows(self):
        np.random.seed(42)
        prices = np.cumprod(1 + np.random.randn(200) * 0.02) * 100
        for window in [5, 21, 63]:
            result = self.optimizer.rolling_drawdown(prices, window=window)
            assert "rolling_drawdown" in result
            assert "max_drawdown_pct" in result

    def test_shrinkage_covariance_identity_vs_diagonal(self):
        np.random.seed(42)
        returns_matrix = np.random.randn(100, 4) * 0.01
        result_id = self.optimizer.shrinkage_covariance(returns_matrix, shrinkage_target="identity")
        result_diag = self.optimizer.shrinkage_covariance(returns_matrix, shrinkage_target="diagonal")
        assert result_id["n_assets"] == 4
        assert result_diag["n_assets"] == 4
        assert "shrinkage_intensity" in result_id
        assert "shrinkage_intensity" in result_diag

    def test_mean_variance_two_assets(self):
        np.random.seed(42)
        returns_matrix = np.random.randn(200, 2) * 0.01 + np.array([0.0003, 0.0005])
        result = self.optimizer.mean_variance_optimize(returns_matrix, ["A", "B"])
        assert "tangency_portfolio" in result
        assert abs(sum(result["tangency_portfolio"].values()) - 1.0) < 0.15
        assert len(result["efficient_frontier"]) == 50

    def test_black_litterman_single_view(self):
        np.random.seed(42)
        n = 3
        market_weights = np.array([0.5, 0.3, 0.2])
        cov = np.eye(n) * 0.03 + np.eye(n) * 0.005
        views = {0: 0.08}
        result = self.optimizer.black_litterman(market_weights, cov, views, view_confidence=0.9)
        assert "bl_returns" in result
        assert result["bl_returns"]["Asset_0"] > result["implied_returns"]["Asset_0"]

    def test_historical_var_cvar_multiple_confidence(self):
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02
        result = self.optimizer.historical_var_cvar(returns, confidence_levels=[0.90, 0.95, 0.99])
        assert "var_90" in result["methods"]
        assert "var_95" in result["methods"]
        assert "var_99" in result["methods"]
        assert result["summary"]["n_observations"] == 500