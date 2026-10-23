import pytest
import numpy as np
from oracle.domains.yield_curve import YieldCurveModel
from oracle.domains.risk_decomposition import RiskDecomposition
from oracle.domains.valuation import ValuationModels


class TestYieldCurveModel:

    def setup_method(self):
        self.ycm = YieldCurveModel()

    def test_nelson_siegel_basic(self):
        result = self.ycm.nelson_siegel()
        assert len(result["rates"]) == 10
        assert all(r > 0 for r in result["rates"])
        assert result["model"] == "nelson_siegel"

    def test_nelson_siegel_long_rate(self):
        result = self.ycm.nelson_siegel(beta0=0.05, beta1=-0.02)
        assert abs(result["long_rate"] - 5.0) < 0.01

    def test_nelson_siegel_short_rate(self):
        result = self.ycm.nelson_siegel(beta0=0.05, beta1=-0.02)
        assert abs(result["short_rate"] - 3.0) < 0.01

    def test_nelson_siegel_custom_maturities(self):
        result = self.ycm.nelson_siegel(maturities=[1, 2, 5, 10])
        assert len(result["rates"]) == 4

    def test_nelson_siegel_svensson(self):
        result = self.ycm.nelson_siegel_svensson()
        assert len(result["rates"]) == 10
        assert result["model"] == "nelson_siegel_svensson"

    def test_nelson_siegel_svensson_extra_param(self):
        ns = self.ycm.nelson_siegel()
        nss = self.ycm.nelson_siegel_svensson(beta3=-0.01)
        assert ns["rates"] != nss["rates"]

    def test_fit_nelson_siegel(self):
        maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
        rates = [3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.3, 4.4, 4.5, 4.5]
        rates_decimal = [r / 100 for r in rates]
        result = self.ycm.fit_nelson_siegel(maturities, rates_decimal)
        assert result["success"]
        assert result["rmse_bps"] < 100

    def test_spot_rates_from_bonds(self):
        prices = [990, 980, 970, 960]
        coupons = [0.02, 0.03, 0.035, 0.04]
        maturities = [0.5, 1.0, 2.0, 5.0]
        result = self.ycm.spot_rates_from_bonds(prices, coupons, maturities)
        assert len(result["spot_rates"]) == 4

    def test_forward_rates(self):
        spot_rates = [0.03, 0.035, 0.04, 0.045]
        maturities = [1, 2, 3, 5]
        result = self.ycm.forward_rates(spot_rates, maturities)
        assert len(result["forward_rates"]) == 3
        assert result["forward_maturities"][0] == "1y-2y"

    def test_forward_rates_increasing(self):
        spot_rates = [0.03, 0.04, 0.05]
        maturities = [1, 2, 3]
        result = self.ycm.forward_rates(spot_rates, maturities)
        assert result["forward_rates"][0] > spot_rates[1] * 100

    def test_yield_curve_scenario_parallel_shift(self):
        result = self.ycm.yield_curve_scenario(shift_bps=100)
        changed = [c > 0 for c in result["changes_bps"]]
        assert all(changed)

    def test_yield_curve_scenario_steepening(self):
        result = self.ycm.yield_curve_scenario(steepen_bps=50)
        assert len(result["changes_bps"]) > 0

    def test_yield_curve_scenario_butterfly(self):
        result = self.ycm.yield_curve_scenario(butterfly_bps=100)
        assert len(result["changes_bps"]) > 0

    def test_yield_curve_scenario_base_unchanged(self):
        result = self.ycm.yield_curve_scenario(shift_bps=0, steepen_bps=0, twist_bps=0, butterfly_bps=0)
        assert all(abs(c) < 0.01 for c in result["changes_bps"])


class TestRiskDecomposition:

    def setup_method(self):
        self.rd = RiskDecomposition()
        np.random.seed(42)
        self.n_periods = 252
        self.n_assets = 3
        self.returns = np.random.randn(self.n_periods, self.n_assets) * 0.01 + 0.0003
        self.weights = np.array([0.4, 0.35, 0.25])

    def test_component_var_historical(self):
        result = self.rd.component_var(self.returns, self.weights, 0.95, "historical")
        assert "portfolio_var" in result
        assert len(result["component_var"]) == 3
        assert result["confidence"] == 0.95

    def test_component_var_parametric(self):
        result = self.rd.component_var(self.returns, self.weights, 0.95, "parametric")
        assert "portfolio_var" in result

    def test_component_var_pct_sum(self):
        result = self.rd.component_var(self.returns, self.weights)
        total_pct = sum(result["pct_contribution"])
        assert abs(total_pct - 100.0) < 1.0

    def test_factor_attribution(self):
        factor_returns = np.random.randn(self.n_periods, 3) * 0.01
        factor_names = ["Market", "Size", "Value"]
        result = self.rd.factor_attribution(self.returns[:, :1], factor_returns, np.array([1.0]), factor_names)
        assert "alpha" in result
        assert "factor_betas" in result
        assert "r_squared" in result
        assert len(result["factor_betas"]) == 3

    def test_risk_budget(self):
        result = self.rd.risk_budget(self.returns, self.weights)
        assert result["portfolio_volatility"] > 0
        assert len(result["asset_risk_contribution"]) == 3
        pct_sum = sum(result["pct_risk_contribution"])
        assert abs(pct_sum - 1.0) < 0.01

    def test_tracking_error(self):
        benchmark = np.random.randn(self.n_periods) * 0.01 + 0.0003
        portfolio = benchmark + 0.0001 + np.random.randn(self.n_periods) * 0.002
        result = self.rd.tracking_error(portfolio, benchmark)
        assert result["tracking_error"] > 0
        assert result["tracking_error_bps"] > 0

    def test_stress_decomposition(self):
        result = self.rd.stress_decomposition(self.returns, self.weights)
        assert "baseline_return" in result
        assert "market_crash" in result["scenarios"]
        assert "rate_hike" in result["scenarios"]
        assert "credit_crisis" in result["scenarios"]
        assert "liquidity_crisis" in result["scenarios"]

    def test_stress_decomposition_losses(self):
        result = self.rd.stress_decomposition(self.returns, self.weights)
        for name, scenario in result["scenarios"].items():
            assert "stressed_return" in scenario
            assert "loss_vs_baseline" in scenario

    def test_component_var_equal_weights(self):
        equal = np.ones(self.n_assets) / self.n_assets
        result = self.rd.component_var(self.returns, equal)
        assert len(result["weights"]) == self.n_assets


class TestValuationModels:

    def setup_method(self):
        self.vm = ValuationModels()

    def test_ddm_basic(self):
        result = self.vm.dividend_discount_model(2.0, 0.05, 0.10)
        assert result["intrinsic_value"] > 0
        assert result["terminal_value_pct"] > 0

    def test_ddm_higher_growth_higher_value(self):
        low = self.vm.dividend_discount_model(2.0, 0.03, 0.10)
        high = self.vm.dividend_discount_model(2.0, 0.07, 0.10)
        assert high["intrinsic_value"] > low["intrinsic_value"]

    def test_ddm_higher_discount_lower_value(self):
        low = self.vm.dividend_discount_model(2.0, 0.05, 0.08)
        high = self.vm.dividend_discount_model(2.0, 0.05, 0.12)
        assert low["intrinsic_value"] > high["intrinsic_value"]

    def test_ddm_multi_stage(self):
        result = self.vm.dividend_discount_model(2.0, 0.08, 0.10, stages=2, years_per_stage=[5, 5])
        assert result["intrinsic_value"] > 0
        assert result["stages"] == 2

    def test_ddm_signal_undervalued(self):
        result = self.vm.dividend_discount_model(2.0, 0.05, 0.10)
        # With market_price < intrinsic_value
        low_price = result["intrinsic_value"] * 0.5
        assert result["intrinsic_value"] > low_price

    def test_dcf_basic(self):
        result = self.vm.discounted_cash_flow(1000, [0.10, 0.08, 0.06, 0.04, 0.03], 0.02, 0.10)
        assert result["enterprise_value"] > 0
        assert result["per_share_value"] > 0

    def test_dcf_with_debt(self):
        nodebt = self.vm.discounted_cash_flow(1000, [0.05], 0.02, 0.10, net_debt=0, shares_outstanding=100)
        withdebt = self.vm.discounted_cash_flow(1000, [0.05], 0.02, 0.10, net_debt=5000, shares_outstanding=100)
        assert nodebt["per_share_value"] > withdebt["per_share_value"]

    def test_dcf_terminal_value_pct(self):
        result = self.vm.discounted_cash_flow(1000, [0.10, 0.08, 0.06, 0.04, 0.03], 0.02, 0.10)
        assert 0 < result["terminal_value_pct"] < 100

    def test_residual_income_basic(self):
        result = self.vm.residual_income_model(50, 5, 0.10, [0.08, 0.06, 0.04, 0.02])
        assert result["intrinsic_value"] > 0
        assert len(result["residual_incomes"]) == 4

    def test_residual_income_higher_cost_lower_value(self):
        low_cost = self.vm.residual_income_model(50, 5, 0.08, [0.05])
        high_cost = self.vm.residual_income_model(50, 5, 0.12, [0.05])
        assert low_cost["intrinsic_value"] >= high_cost["intrinsic_value"]

    def test_eva_basic(self):
        result = self.vm.eva_model(200, 1000, 0.10, [0.05, 0.04, 0.03, 0.02, 0.01])
        assert result["enterprise_value"] > 0
        assert result["current_eva"] > 0

    def test_eva_spread_positive(self):
        result = self.vm.eva_model(200, 1000, 0.10, [0.05])
        assert result["current_roic"] > 10.0

    def test_sensitivity_analysis_basic(self):
        result = self.vm.SensitivityAnalysis()
        assert result["base_value"] > 0
        assert "growth_rate" in result["sensitivities"]
        assert "discount_rate" in result["sensitivities"]

    def test_sensitivity_growth_impact(self):
        result = self.vm.SensitivityAnalysis()
        growth_results = result["sensitivities"]["growth_rate"]
        values = [r["value"] for r in growth_results if r["value"] is not None]
        assert values[-1] > values[0]

    def test_dcf_year_fcfs(self):
        result = self.vm.discounted_cash_flow(1000, [0.10, 0.08, 0.06, 0.04, 0.03], 0.02, 0.10)
        assert len(result["year_fcfs"]) == 5
        assert result["year_fcfs"][0]["year"] == 1