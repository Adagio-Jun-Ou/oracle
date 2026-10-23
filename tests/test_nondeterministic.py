import pytest
import numpy as np
from datetime import date
from oracle.nondeterministic.buffett_value import BuffettValueEngine, CompanyFundamentals, MoatAssessment
from oracle.nondeterministic.monte_carlo import MonteCarloEngine
from oracle.nondeterministic.bayesian import BayesianEngine
from oracle.nondeterministic.ml_ensemble import MLEnsemble
from oracle.core.types import Domain, ConfidenceLevel


class TestBuffettValueEngine:
    def setup_method(self):
        self.engine = BuffettValueEngine()

    def test_assess_moat(self):
        company = CompanyFundamentals(
            gross_margin=0.60, roe=0.25, operating_margin=0.30,
            revenue_growth_5yr=0.12, roic=0.20, net_margin=0.18,
        )
        moat = self.engine.assess_moat(company)
        assert isinstance(moat, MoatAssessment)
        assert 0 <= moat.overall_moat_score <= 1
        assert moat.moat_durability_years > 0

    def test_dcf_valuation(self):
        company = CompanyFundamentals(
            earnings_per_share=5.0, free_cash_flow_per_share=4.0,
            share_count_mm=1000.0, earnings_growth_5yr=0.08, roe=0.15,
        )
        value = self.engine.dcf_valuation(company)
        assert value > 0

    def test_graham_number(self):
        company = CompanyFundamentals(earnings_per_share=5.0, book_value_per_share=40.0)
        graham = self.engine.graham_number(company)
        assert graham > 0
        expected = np.sqrt(22.5 * 5.0 * 40.0)
        assert abs(graham - expected) < 1.0

    def test_earnings_power_value(self):
        company = CompanyFundamentals(
            earnings_per_share=5.0, operating_margin=0.20, roe=0.15, roic=0.12,
        )
        epv = self.engine.earnings_power_value(company)
        assert epv > 0

    def test_full_valuation_undervalued(self):
        company = CompanyFundamentals(
            earnings_per_share=8.0, book_value_per_share=60.0,
            free_cash_flow_per_share=7.0, dividends_per_share=2.0,
            earnings_growth_5yr=0.12, roe=0.22, roic=0.18,
            debt_to_equity=0.3, gross_margin=0.55, operating_margin=0.28,
            net_margin=0.17, payout_ratio=0.25, market_price=100.0,
        )
        result = self.engine.full_valuation(company)
        assert result.intrinsic_value > 0
        assert result.quality_score > 0
        assert result.recommendation in ("strong_buy", "buy", "hold", "sell")

    def test_full_valuation_overvalued(self):
        company = CompanyFundamentals(
            earnings_per_share=2.0, book_value_per_share=15.0,
            free_cash_flow_per_share=1.0, earnings_growth_5yr=0.02,
            roe=0.06, roic=0.04, debt_to_equity=2.0,
            gross_margin=0.20, operating_margin=0.05, net_margin=0.02,
            market_price=200.0,
        )
        result = self.engine.full_valuation(company)
        assert result.margin_of_safety < 0
        assert result.recommendation == "sell"

    def test_predict(self):
        company = CompanyFundamentals(earnings_per_share=5.0, market_price=100.0)
        pred = self.engine.predict(company, date(2030, 1, 1), ConfidenceLevel.HIGH)
        assert pred.domain == Domain.FINANCE
        assert pred.confidence_interval[0] <= pred.value


class TestMonteCarloEngine:
    def setup_method(self):
        self.engine = MonteCarloEngine(n_simulations=1000, seed=42)

    def test_gbm(self):
        result = self.engine.geometric_brownian_motion(s0=5000, mu=0.08, sigma=0.2, n_steps=252)
        assert result.paths.shape == (1000, 253)
        assert result.mean[-1] > 0
        assert result.probability_above(0) == 1.0
        assert 0 < result.probability_above(5000) < 1

    def test_gbm_var(self):
        result = self.engine.geometric_brownian_motion(s0=100, mu=0.08, sigma=0.2, n_steps=252)
        var = result.value_at_risk(0.95)
        assert var > 0
        es = result.expected_shortfall(0.95)
        assert es <= var

    def test_mean_reverting(self):
        result = self.engine.mean_reverting(x0=0.05, theta=0.3, mu=0.04, sigma=0.02, n_steps=252)
        assert result.paths.shape[0] == 1000
        assert result.model_type == "ou_mean_reverting"

    def test_jump_diffusion(self):
        result = self.engine.jump_diffusion(s0=100, mu=0.08, sigma=0.2, n_steps=252)
        assert result.paths.shape[0] == 1000
        assert result.model_type == "merton_jump_diffusion"

    def test_regime_switching(self):
        result = self.engine.regime_switching(s0=100, n_steps=252)
        assert result.paths.shape[0] == 1000
        assert result.model_type == "regime_switching"

    def test_max_drawdown(self):
        result = self.engine.geometric_brownian_motion(s0=100, mu=0.08, sigma=0.3, n_steps=252)
        dd = result.max_drawdown()
        assert 0 <= dd <= 1

    def test_predict(self):
        pred = self.engine.predict(100.0, "test", Domain.FINANCE, date(2030, 1, 1))
        assert pred.model_name.startswith("monte_carlo")
        assert pred.confidence_interval[0] <= pred.value


class TestBayesianEngine:
    def setup_method(self):
        self.engine = BayesianEngine()

    def test_normal_normal_conjugate(self):
        data = np.random.normal(0.5, 0.3, 50)
        result = self.engine.conjugate_normal_normal(data, prior_mu=0.0, prior_sigma=1.0, known_sigma=0.3)
        assert abs(result.mean - np.mean(data)) < 1.0
        assert result.std > 0
        assert result.credible_interval[0] < result.credible_interval[1]

    def test_beta_binomial(self):
        result = self.engine.conjugate_beta_binomial(successes=70, trials=100)
        assert 0.5 < result.mean < 0.9
        assert result.std > 0

    def test_gamma_poisson(self):
        counts = np.random.poisson(5, 100)
        result = self.engine.conjugate_gamma_poisson(counts)
        assert result.mean > 0

    def test_model_comparison(self):
        data = np.random.normal(0.5, 0.5, 50)
        result = self.engine.model_comparison(data, null_mu=0.0)
        assert "bayes_factor" in result
        assert isinstance(result["favor_null"], (bool, np.bool_))

    def test_change_point(self):
        data = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(3, 1, 50)])
        result = self.engine.bayesian_change_point(data)
        assert "change_points" in result

    def test_hierarchical(self):
        groups = [np.random.normal(mu, 0.5, 30) for mu in [0, 1, 2]]
        result = self.engine.hierarchical_model(groups)
        assert "shrunk_means" in result
        assert len(result["shrunk_means"]) == 3

    def test_predict(self):
        data = np.random.normal(0.5, 0.3, 100)
        pred = self.engine.predict(data, "test", Domain.ECONOMICS, date(2030, 1, 1))
        assert pred.model_name == "bayesian_normal"


class TestMLEnsemble:
    def setup_method(self):
        self.model = MLEnsemble(n_estimators=50, random_state=42)
        np.random.seed(42)
        self.trend_data = np.cumsum(np.random.normal(0.01, 0.05, 200))

    def test_create_features(self):
        features = self.model.create_features(self.trend_data, lag=5)
        assert features.values.shape[0] > 0
        assert features.values.shape[1] == len(features.names)

    def test_fit_predict(self):
        features = self.model.create_features(self.trend_data, lag=5)
        self.model.fit(features.values, features.target)
        assert self.model.is_fitted
        result = self.model.predict_with_uncertainty(features.values)
        assert len(result.predictions) == features.values.shape[0]

    def test_ensemble_predict(self):
        features = self.model.create_features(self.trend_data, lag=5)
        self.model.fit(features.values, features.target)
        result = self.model.predict_with_uncertainty(features.values)
        assert hasattr(result, "predictions")
        assert hasattr(result, "model_weights")
        assert len(result.model_weights) > 0
        assert len(result.model_weights) > 0

    def test_predict(self):
        pred = self.model.predict(self.trend_data, "test_var", Domain.ECONOMICS, date(2030, 1, 1))
        assert pred.model_name == "ml_ensemble"