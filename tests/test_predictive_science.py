import pytest
import numpy as np
from oracle.nondeterministic.conformal import ConformalPredictor, ConformalResult
from oracle.nondeterministic.causal import CausalInference, CausalEffect, CounterfactualResult
from oracle.nondeterministic.probabilistic import ProbabilisticForecaster, ProbabilisticForecast, CRPSResult
from oracle.nondeterministic.quantile_regression import QuantileRegression, QuantileRegressionResult, HeteroscedasticInterval
from oracle.prediction.bsts import BayesianStructuralTimeSeries, BSTSResult
from oracle.prediction.correlation_dynamics import CorrelationDynamicsModel, DCCResult, CopulaResult
from oracle.prediction.shapley import ShapleyAttribution, ShapleyResult, ShapleyValue
from oracle.prediction.online_learning import OnlineLearner, OnlineLearningState, DriftDetectionResult
from oracle.prediction.kan import KolmogorovArnoldNetwork, KANResult, KANLayerResult


@pytest.fixture
def rng():
    return np.random.RandomState(42)


class TestConformalPrediction:
    def test_split_conformal_basic(self):
        cp = ConformalPredictor(alpha=0.1)
        y_true = np.random.RandomState(42).normal(0, 1, 100)
        y_pred = y_true + np.random.RandomState(43).normal(0, 0.5, 100)
        cp.fit(y_true, y_pred)
        result = cp.predict(0.0)
        assert isinstance(result, ConformalResult)
        assert result.prediction == 0.0
        assert result.lower < 0.0
        assert result.upper > 0.0
        assert result.coverage == 0.9
        assert result.width > 0
        assert result.method == "split"

    def test_normalized_conformal(self):
        cp = ConformalPredictor(alpha=0.1)
        rng = np.random.RandomState(42)
        y_true = rng.normal(0, 1, 100)
        y_pred = y_true + rng.normal(0, 0.5, 100)
        uncertainties = np.abs(y_pred - y_true) + 0.1
        cp.fit(y_true, y_pred, uncertainties=uncertainties)
        result = cp.predict(0.0, uncertainty=0.5)
        assert result.method == "normalized"
        assert result.lower < result.upper

    def test_predict_batch(self):
        cp = ConformalPredictor(alpha=0.1)
        rng = np.random.RandomState(42)
        y_true = rng.normal(0, 1, 100)
        y_pred = y_true + rng.normal(0, 0.3, 100)
        cp.fit(y_true, y_pred)
        results = cp.predict_batch(np.array([0.0, 1.0, -1.0]))
        assert len(results) == 3
        for r in results:
            assert r.lower < r.upper

    def test_jackknife_plus(self):
        cp = ConformalPredictor(alpha=0.1)
        rng = np.random.RandomState(42)
        y_true = rng.normal(0, 1, 50)
        y_pred = y_true + rng.normal(0, 0.3, 50)
        result = cp.jackknife_plus(y_true, y_pred, 0.0)
        assert isinstance(result, ConformalResult)
        assert result.method == "jackknife_plus"
        assert result.lower < result.upper

    def test_adaptive_conformal(self):
        result = ConformalPredictor.adaptive_conformal(
            np.random.RandomState(42).normal(0, 1, 200),
            np.random.RandomState(42).normal(0, 1, 200),
            alpha=0.1,
        )
        assert "final_quantile" in result
        assert "final_coverage" in result
        assert "coverage_history" in result
        assert "method" in result
        assert result["method"] == "adaptive_conformal"

    def test_conformal_bands(self):
        rng = np.random.RandomState(42)
        ts = np.cumsum(rng.normal(0, 1, 100))
        result = ConformalPredictor.conformal_bands(ts, window=20, alpha=0.1)
        assert "lower" in result
        assert "upper" in result
        assert "coverage" in result
        assert result["method"] == "conformal_bands"

    def test_coverage_property(self):
        rng = np.random.RandomState(42)
        y_true = rng.normal(0, 1, 1000)
        y_pred = y_true + rng.normal(0, 0.3, 1000)
        cp = ConformalPredictor(alpha=0.1)
        cp.fit(y_true[:500], y_pred[:500])
        covered = 0
        for i in range(500, 1000):
            result = cp.predict(y_pred[i])
            if result.lower <= y_true[i] <= result.upper:
                covered += 1
        coverage = covered / 500
        assert coverage > 0.80


class TestCausalInference:
    def test_ate_basic(self):
        ci = CausalInference(seed=42)
        rng = np.random.RandomState(42)
        y_treated = rng.normal(1.0, 1.0, 100)
        y_control = rng.normal(0.0, 1.0, 100)
        result = ci.average_treatment_effect(y_treated, y_control, treatment="treatment", outcome="outcome")
        assert isinstance(result, CausalEffect)
        assert result.ate > 0
        assert result.direction == "positive"
        assert len(result.assumptions) > 0

    def test_ate_with_covariates(self):
        ci = CausalInference(seed=42)
        rng = np.random.RandomState(42)
        n = 200
        x = rng.normal(0, 1, (n, 3))
        treatment = rng.binomial(1, 0.5, n)
        y = 2.0 * treatment + 0.5 * x[:, 0] + rng.normal(0, 0.5, n)
        y_treated = y[treatment == 1]
        y_control = y[treatment == 0]
        x_treated = x[treatment == 1]
        x_control = x[treatment == 0]
        result = ci.average_treatment_effect(y_treated, y_control, x_treated, x_control)
        assert abs(result.ate - 2.0) < 1.5
        assert result.ite_std > 0

    def test_sensitivity_analysis(self):
        ci = CausalInference(seed=42)
        rng = np.random.RandomState(42)
        y_t = rng.normal(1.0, 1.0, 200)
        y_c = rng.normal(0.0, 1.0, 200)
        result = ci.average_treatment_effect(y_t, y_c)
        assert "gamma_1.0" in result.sensitivity
        assert "robustness_threshold" in result.sensitivity

    def test_instrumental_variable(self):
        ci = CausalInference(seed=42)
        rng = np.random.RandomState(42)
        n = 500
        z = rng.normal(0, 1, n)
        d = (0.5 * z + rng.normal(0, 1, n) > 0).astype(float)
        y = 2.0 * d + rng.normal(0, 1, n)
        result = ci.instrumental_variable(y, d, z, treatment="education", outcome="income")
        assert isinstance(result, CausalEffect)
        assert result.ate != 0

    def test_difference_in_differences(self):
        ci = CausalInference(seed=42)
        rng = np.random.RandomState(42)
        y_pre_t = rng.normal(100, 5, 50)
        y_post_t = rng.normal(110, 5, 50)
        y_pre_c = rng.normal(100, 5, 50)
        y_post_c = rng.normal(105, 5, 50)
        result = ci.difference_in_differences(y_pre_t, y_post_t, y_pre_c, y_post_c)
        assert result.ate > 0
        assert "parallel_trends" in str(result.assumptions).lower()

    def test_counterfactual(self):
        ci = CausalInference(seed=42)
        factual = {"gdp_growth": 0.03, "unemployment": 0.06, "inflation": 0.02}
        interventions = [
            {"gdp_growth": 0.05},
            {"unemployment": 0.03},
            {"inflation": 0.06},
        ]
        def outcome_fn(gdp_growth, unemployment, inflation, **kwargs):
            return gdp_growth * 100 - unemployment * 50 - inflation * 30
        results = ci.counterfactual_analysis(factual, interventions, outcome_fn=outcome_fn)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, CounterfactualResult)
            assert r.treatment_effect != 0

    def test_mediation(self):
        ci = CausalInference(seed=42)
        rng = np.random.RandomState(42)
        n = 200
        treatment = rng.binomial(1, 0.5, n)
        mediator = 0.5 * treatment + rng.normal(0, 1, n)
        y = 0.3 * treatment + 0.7 * mediator + rng.normal(0, 0.5, n)
        result = ci.mediation_analysis(y, treatment, mediator)
        assert "total_effect" in result
        assert "direct_effect" in result
        assert "indirect_effect" in result
        assert "proportion_mediated" in result

    def test_propensity_score_matching(self):
        ci = CausalInference(seed=42)
        rng = np.random.RandomState(42)
        n = 300
        x = rng.normal(0, 1, (n, 3))
        treatment = (0.5 * x[:, 0] + rng.normal(0, 1, n) > 0).astype(float)
        y = 1.5 * treatment + 0.3 * x[:, 0] + rng.normal(0, 1, n)
        result = ci.propensity_score_matching(y, treatment, x)
        assert isinstance(result, CausalEffect)
        assert result.ate > 0

    def test_backdoor_adjustment(self):
        ci = CausalInference(seed=42)
        ci.define_causal_graph([
            ("X", "Y", 0.7),
            ("C", "X", 0.3),
            ("C", "Y", 0.4),
        ], confounders=[("C", ["X", "Y"])])
        result = ci.backdoor_adjustment("X", "Y")
        assert result["backdoor_criterion_satisfied"]
        assert result["adjustment_needed"]


class TestProbabilisticForecasting:
    def test_gaussian_forecast(self):
        pf = ProbabilisticForecaster()
        result = pf.forecast_gaussian(100.0, 10.0)
        assert isinstance(result, ProbabilisticForecast)
        assert result.mean == 100.0
        assert result.std == 10.0
        assert result.lower_95 < result.lower_50 < result.point_forecast
        assert result.upper_50 < result.upper_95
        assert result.distribution == "gaussian"

    def test_gaussian_forecast_with_observation(self):
        pf = ProbabilisticForecaster()
        result = pf.forecast_gaussian(100.0, 10.0, observation=105.0)
        assert result.crps is not None
        assert result.crps > 0
        assert result.pit is not None
        assert 0 < result.pit < 1

    def test_t_distribution_forecast(self):
        pf = ProbabilisticForecaster()
        result = pf.forecast_t_distribution(100.0, 10.0, df=5.0)
        assert result.distribution == "student_t"
        assert result.lower_95 < result.upper_95
        assert result.kurtosis > 3.0

    def test_skewnorm_forecast(self):
        pf = ProbabilisticForecaster()
        result = pf.forecast_skewnorm(100.0, 10.0, skewness=0.5)
        assert result.distribution == "skew_normal"
        assert abs(result.skewness - 0.5) < 0.5

    def test_ensemble_probabilistic(self):
        pf = ProbabilisticForecaster()
        f1 = pf.forecast_gaussian(100.0, 10.0)
        f2 = pf.forecast_gaussian(102.0, 8.0)
        f3 = pf.forecast_t_distribution(101.0, 12.0, df=5.0)
        result = pf.ensemble_probabilistic([f1, f2, f3], weights=[0.5, 0.3, 0.2])
        assert result.distribution == "ensemble"
        assert 99 < result.mean < 103
        assert result.std > 0

    def test_crps_gaussian(self):
        result = ProbabilisticForecaster.crps_gaussian(100.0, 100.0, 10.0)
        assert result > 0
        result_zero = ProbabilisticForecaster.crps_gaussian(100.0, 100.0, 1.0)
        assert result_zero < result

    def test_crps_evaluation(self):
        pf = ProbabilisticForecaster()
        rng = np.random.RandomState(42)
        observations = rng.normal(0, 1, 50)
        means = rng.normal(0, 0.5, 50)
        stds = np.abs(rng.normal(1, 0.2, 50))
        result = pf.evaluate_forecasts(observations, means, stds)
        assert isinstance(result, CRPSResult)
        assert result.crps > 0
        assert result.brier_score >= 0
        assert result.log_score > 0
        assert "q_10" in result.quantile_scores
        assert "q_50" in result.quantile_scores

    def test_pit_histogram(self):
        rng = np.random.RandomState(42)
        observations = rng.normal(0, 1, 100)
        means = np.zeros(100)
        stds = np.ones(100)
        result = ProbabilisticForecaster.pit_histogram(observations, means, stds)
        assert "calibrated" in result
        assert "histogram" in result

    def test_calibration_curve(self):
        rng = np.random.RandomState(42)
        observed = rng.binomial(1, 0.5, 100).astype(float)
        predicted = rng.uniform(0, 1, 100)
        result = ProbabilisticForecaster.calibration_curve(observed, predicted)
        assert "calibration_data" in result
        assert "expected_calibration_error" in result
        assert result["expected_calibration_error"] >= 0


class TestBSTS:
    def test_decompose_basic(self):
        bsts = BayesianStructuralTimeSeries(seed=42)
        rng = np.random.RandomState(42)
        series = np.cumsum(rng.normal(0, 1, 100)) + 10 * np.sin(np.arange(100) * 2 * np.pi / 12)
        result = bsts.decompose(series, period=12)
        assert isinstance(result, BSTSResult)
        assert len(result.trend) == 100
        assert len(result.seasonal) == 100
        assert len(result.cycle) == 100
        assert len(result.residual) == 100
        assert len(result.forecast) == 10

    def test_decompose_forecast_intervals(self):
        bsts = BayesianStructuralTimeSeries(seed=42)
        rng = np.random.RandomState(42)
        series = np.cumsum(rng.normal(0, 1, 150))
        result = bsts.decompose(series, period=12, n_forecast=5)
        for i in range(5):
            assert result.forecast_lower_95[i] <= result.forecast_lower_50[i]
            assert result.forecast_lower_50[i] <= result.forecast[i]
            assert result.forecast[i] <= result.forecast_upper_50[i]
            assert result.forecast_upper_50[i] <= result.forecast_upper_95[i]

    def test_state_contributions(self):
        bsts = BayesianStructuralTimeSeries(seed=42)
        rng = np.random.RandomState(42)
        series = 100 + np.cumsum(rng.normal(0, 1, 200))
        result = bsts.decompose(series, period=52)
        assert "trend_pct" in result.state_contributions
        assert "seasonal_pct" in result.state_contributions
        total = sum(result.state_contributions.values())
        assert 80 < total < 120

    def test_changepoint_detection(self):
        bsts = BayesianStructuralTimeSeries(seed=42)
        series = np.concatenate([np.zeros(50), np.ones(50) * 5])
        result = bsts.decompose(series)
        assert isinstance(result.changepoints, list)

    def test_nowcast(self):
        bsts = BayesianStructuralTimeSeries(seed=42)
        rng = np.random.RandomState(42)
        series = np.cumsum(rng.normal(0, 1, 100))
        partial = np.array([1.0, 2.0, 3.0])
        result = bsts.nowcast(series, partial)
        assert "nowcast" in result
        assert "nowcast_lower_95" in result
        assert "nowcast_upper_95" in result

    def test_bayesian_model_average(self):
        forecasts = [
            {"point": 100.0, "forecast": 100.0, "lower": 90.0, "upper": 110.0, "std": 5.0, "model": "model_a"},
            {"point": 105.0, "forecast": 105.0, "lower": 95.0, "upper": 115.0, "std": 6.0, "model": "model_b"},
        ]
        bics = [100.0, 120.0]
        result = BayesianStructuralTimeSeries.bayesian_model_average(forecasts, bics)
        assert "point" in result
        assert "weights" in result
        assert len(result["weights"]) == 2
        assert result["weights"][0]["weight"] > result["weights"][1]["weight"]

    def test_short_series(self):
        bsts = BayesianStructuralTimeSeries(seed=42)
        series = np.array([1.0, 2.0, 3.0])
        result = bsts.decompose(series)
        assert len(result.trend) == 3


class TestCorrelationDynamics:
    def test_dcc_garch_basic(self):
        cdm = CorrelationDynamicsModel(seed=42)
        rng = np.random.RandomState(42)
        returns = rng.normal(0, 0.02, (100, 3))
        result = cdm.dcc_garch(returns)
        assert isinstance(result, DCCResult)
        assert 0 < result.dcc_alpha < 1
        assert 0 < result.dcc_beta < 1
        assert result.persistence > 0
        assert result.half_life > 0

    def test_dcc_mean_correlation(self):
        cdm = CorrelationDynamicsModel(seed=42)
        rng = np.random.RandomState(42)
        c1 = rng.normal(0, 1, 100)
        returns = np.column_stack([c1, c1 + rng.normal(0, 0.1, 100), rng.normal(0, 1, 100)])
        result = cdm.dcc_garch(returns)
        assert result.mean_correlation > 0

    def test_rolling_correlation(self):
        cdm = CorrelationDynamicsModel(seed=42)
        rng = np.random.RandomState(42)
        returns = rng.normal(0, 0.02, (200, 3))
        result = cdm.rolling_correlation(returns, window=30)
        assert "rolling_correlations" in result
        assert "correlation_trend" in result
        assert "mean_correlation" in result

    def test_copula_gaussian(self):
        cdm = CorrelationDynamicsModel()
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 500)
        y = 0.7 * x + rng.normal(0, 0.5, 500)
        result = cdm.copula_analysis(x, y, "gaussian")
        assert isinstance(result, CopulaResult)
        assert result.copula_type == "gaussian"
        assert abs(result.spearman_rho) > 0.5
        assert result.joint_probability is not None
        assert result.conditional_var is not None

    def test_copula_student_t(self):
        cdm = CorrelationDynamicsModel()
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 500)
        y = 0.7 * x + rng.normal(0, 0.5, 500)
        result = cdm.copula_analysis(x, y, "student_t")
        assert result.tail_dependence_lower > 0
        assert result.tail_dependence_upper > 0

    def test_copula_clayton(self):
        cdm = CorrelationDynamicsModel()
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 500)
        y = 0.5 * x + rng.normal(0, 1, 500)
        result = cdm.copula_analysis(x, y, "clayton")
        assert result.copula_type == "clayton"
        assert result.tail_dependence_lower > 0

    def test_copula_gumbel(self):
        cdm = CorrelationDynamicsModel()
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 500)
        y = 0.5 * x + rng.normal(0, 1, 500)
        result = cdm.copula_analysis(x, y, "gumbel")
        assert result.copula_type == "gumbel"
        assert result.copula_parameter >= 1.0

    def test_tail_dependence(self):
        cdm = CorrelationDynamicsModel()
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 500)
        y = 0.8 * x + rng.normal(0, 0.3, 500)
        result = cdm.tail_dependence(x, y, threshold=0.95)
        assert "lower" in result
        assert "upper" in result
        assert 0 <= result["lower"] <= 1


class TestShapleyAttribution:
    def test_explain_basic(self):
        def predict_fn(x):
            return x[:, 0] * 2 + x[:, 1] * 0.5 + x[:, 2] * 0.1

        sa = ShapleyAttribution(seed=42)
        x = np.array([3.0, 1.0, 2.0])
        result = sa.explain(predict_fn, x, ["feature_a", "feature_b", "feature_c"], n_samples=200)
        assert isinstance(result, ShapleyResult)
        assert len(result.shapley_values) == 3
        assert result.base_value is not None
        assert result.prediction is not None
        assert len(result.top_features) > 0
        assert "feature_a" in result.top_features

    def test_feature_importance_global(self):
        def predict_fn(x):
            return x[:, 0] * 3 + x[:, 1] * 1 + x[:, 2] * 0.1

        sa = ShapleyAttribution(seed=42)
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 3))
        result = sa.feature_importance_global(predict_fn, X, ["a", "b", "c"], n_samples=50)
        assert "feature_importance" in result
        assert result["feature_importance"]["a"] > result["feature_importance"]["c"]

    def test_counterfactual_explanation(self):
        def predict_fn(x):
            return x[:, 0] * 2 + x[:, 1] * 3

        sa = ShapleyAttribution(seed=42)
        x = np.array([1.0, 1.0])
        result = sa.counterfactual_explanation(predict_fn, x, ["a", "b"], target_value=10.0)
        assert "feature_changes" in result
        assert "original_prediction" in result
        assert "counterfactual_prediction" in result

    def test_confidence_score(self):
        def predict_fn(x):
            return x[:, 0] * 5

        sa = ShapleyAttribution(seed=42)
        x = np.array([2.0])
        result = sa.explain(predict_fn, x, ["feature_1"], n_samples=100)
        assert 0 <= result.confidence <= 1


class TestWebAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from oracle.web.app import app
        return TestClient(app)

    def test_conformal_endpoint(self, client):
        resp = client.get("/api/conformal/bands")
        assert resp.status_code == 200

    def test_causal_endpoint(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200

    def test_probabilistic_forecast_endpoint(self, client):
        resp = client.get("/api/probabilistic/forecast?mean=100&std=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["point_forecast"] == 100.0
        assert data["distribution"] == "gaussian"

    def test_probabilistic_forecast_t(self, client):
        resp = client.get("/api/probabilistic/forecast?mean=100&std=10&distribution=student_t")
        assert resp.status_code == 200
        data = resp.json()
        assert data["distribution"] == "student_t"

    def test_probabilistic_evaluate_endpoint(self, client):
        resp = client.get("/api/probabilistic/evaluate")
        assert resp.status_code == 200
        data = resp.json()
        assert "crps" in data

    def test_bsts_endpoint(self, client):
        resp = client.get("/api/bsts/decompose")
        assert resp.status_code == 200
        data = resp.json()
        assert "trend" in data
        assert "forecast" in data

    def test_dcc_endpoint(self, client):
        resp = client.get("/api/dcc/correlation")
        assert resp.status_code == 200
        data = resp.json()
        assert "mean_correlation" in data

    def test_copula_endpoint(self, client):
        resp = client.get("/api/copula/analyze")
        assert resp.status_code == 200
        data = resp.json()
        assert "spearman_rho" in data

    def test_shapley_endpoint(self, client):
        resp = client.get("/api/shapley/explain")
        assert resp.status_code == 200
        data = resp.json()
        assert "shapley_values" in data

    def test_drift_detection_endpoint(self, client):
        resp = client.get("/api/drift-detection")
        assert resp.status_code == 200
        data = resp.json()
        assert "drift_detected" in data

    def test_online_forecast_endpoint(self, client):
        resp = client.get("/api/online-forecast")
        assert resp.status_code == 200
        data = resp.json()
        assert "exponential_forgetting" in data


class TestQuantileRegression:
    def test_fit_basic(self):
        rng = np.random.RandomState(42)
        n = 100
        X = rng.normal(0, 1, (n, 2))
        y = 2 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5 + 0.3 * np.abs(X[:, 0]), n)
        qr = QuantileRegression(seed=42)
        results = qr.fit(X, y, quantiles=[0.1, 0.5, 0.9])
        assert 0.1 in results
        assert 0.5 in results
        assert 0.9 in results
        for q, result in results.items():
            assert isinstance(result, QuantileRegressionResult)
            assert result.quantile == q
            assert result.n_obs == n

    def test_predict_intervals(self):
        rng = np.random.RandomState(42)
        n = 200
        X = rng.normal(0, 1, (n, 3))
        y = X[:, 0] * 3 + rng.normal(0, 1, n)
        qr = QuantileRegression(seed=42)
        qr.fit(X, y)
        X_new = rng.normal(0, 1, (20, 3))
        intervals = qr.predict_intervals(X_new)
        assert isinstance(intervals, HeteroscedasticInterval)
        assert len(intervals.lower) == 20
        assert len(intervals.upper) == 20
        assert not intervals.quantile_crossing or intervals.mean_width == 0

    def test_noncrossing_adjustment(self):
        rng = np.random.RandomState(42)
        preds = {
            0.1: [1, 2, 3],
            0.5: [2, 3, 4],
            0.9: [5, 4, 3],
        }
        adjusted = QuantileRegression.noncrossing_quantile_adjust(preds)
        for i in range(3):
            assert adjusted[0.1][i] <= adjusted[0.5][i]
            assert adjusted[0.5][i] <= adjusted[0.9][i]

    def test_conditional_var(self):
        preds = {0.05: [1.0] * 10, 0.95: [3.0] * 10}
        result = QuantileRegression.conditional_var(preds, confidence=0.9)
        assert "conditional_var" in result
        assert "mean_var" in result
        assert result["heteroscedasticity_ratio"] >= 0

    def test_backtest_coverage(self):
        rng = np.random.RandomState(42)
        y = rng.normal(0, 1, 100)
        preds = {0.05: y - 1.65, 0.5: y, 0.95: y + 1.65}
        result = QuantileRegression.backtest_coverage(y, preds)
        assert 0.05 in result
        assert 0.5 in result
        assert 0.95 in result
        assert abs(result[0.05]["empirical"] - 0.05) < 0.25

    def test_feature_importance(self):
        rng = np.random.RandomState(42)
        n = 100
        X = rng.normal(0, 1, (n, 2))
        y = 2 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n)
        qr = QuantileRegression(seed=42)
        results = qr.fit(X, y)
        importance = QuantileRegression.quantile_feature_importance(results)
        assert "feature_0" in importance
        assert "feature_1" in importance


class TestOnlineLearning:
    def test_partial_fit(self):
        ol = OnlineLearner(n_features=3, learning_rate=0.01)
        rng = np.random.RandomState(42)
        for _ in range(50):
            x = rng.normal(0, 1, 3)
            y = 2 * x[0] + 0.5 * x[1]
            state = ol.partial_fit(x, y)
            assert isinstance(state, OnlineLearningState)
        assert state.n_updates == 50
        assert len(state.weights) == 3

    def test_drift_detection(self):
        rng = np.random.RandomState(42)
        series = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100)])
        result = OnlineLearner.adwin_drift_detection(series)
        assert isinstance(result, DriftDetectionResult)
        if result.drift_detected:
            assert result.drift_start_index > 0
            assert result.drift_type in ("abrupt", "gradual", "none")

    def test_page_hinkley(self):
        rng = np.random.RandomState(42)
        series = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])
        result = OnlineLearner.page_hinkley_drift(series, threshold=30.0)
        assert isinstance(result, DriftDetectionResult)

    def test_exponential_forgetting(self):
        rng = np.random.RandomState(42)
        series = np.cumsum(rng.normal(0, 1, 200))
        result = OnlineLearner.exponential_forgetting_forecast(series, halflife=10.0, n_ahead=5)
        assert "forecast" in result
        assert "ema" in result
        assert "trend" in result
        assert len(result["forecast"]) == 5
        assert "forecast_lower_95" in result
        assert "forecast_upper_95" in result

    def test_kalman_filter(self):
        rng = np.random.RandomState(42)
        series = rng.normal(10, 2, 100)
        result = OnlineLearner.kalman_filter_1d(series)
        assert "filtered" in result
        assert len(result["filtered"]) == 100
        assert "forecast" in result
        assert "kalman_gains" in result
        assert "innovations" in result

    def test_online_predict(self):
        ol = OnlineLearner(n_features=2, learning_rate=0.01)
        rng = np.random.RandomState(42)
        for _ in range(100):
            x = rng.normal(0, 1, 2)
            y = 3 * x[0] - 1 * x[1]
            ol.partial_fit(x, y)
        test_x = np.array([1.0, 0.0])
        pred = ol.predict(test_x)
        assert abs(pred - 3.0) < 3.0

    def test_batch_predict(self):
        ol = OnlineLearner(n_features=2, learning_rate=0.01)
        rng = np.random.RandomState(42)
        for _ in range(50):
            x = rng.normal(0, 1, 2)
            y = x[0]
            ol.partial_fit(x, y)
        X = rng.normal(0, 1, (10, 2))
        preds = ol.predict_batch(X)
        assert len(preds) == 10


class TestKAN:
    def test_kan_fit_and_predict(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 3))
        y = 2 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, 100)
        kan = KolmogorovArnoldNetwork(grid_size=5, spline_order=3)
        kan.fit(X, y, max_iter=200, lr=0.01)
        preds = kan.predict(X)
        assert len(preds) == 100
        assert np.all(np.isfinite(preds))

    def test_kan_explain(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (80, 4))
        y = X[:, 0] ** 2 + rng.normal(0, 0.3, 80)
        kan = KolmogorovArnoldNetwork(grid_size=5, spline_order=3)
        kan.fit(X, y, max_iter=200)
        result = kan.explain(X)
        assert isinstance(result, KANResult)
        assert isinstance(result.prediction, float)
        assert isinstance(result.feature_importance, dict)
        assert len(result.feature_importance) > 0
        assert isinstance(result.contributions, list)
        assert result.n_parameters > 0

    def test_kan_symbolic_regression(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 3))
        y = 2 * X[:, 0] ** 2 + np.sin(X[:, 1])
        result = KolmogorovArnoldNetwork.symbolic_regression(X, y)
        assert "best_expressions" in result
        assert "r_squared_approx" in result
        assert "n_candidates_evaluated" in result
        assert len(result["best_expressions"]) > 0

    def test_kan_kolmogorov_decomposition(self):
        rng = np.random.RandomState(42)
        f_values = rng.normal(0, 1, 100)
        result = KolmogorovArnoldNetwork.kolmogorov_decomposition(f_values, n_inputs=2)
        assert "inner_functions" in result
        assert "outer_function" in result
        assert "r_squared" in result
        assert len(result["inner_functions"]) > 0

    def test_kan_symbolic_regression_nonlinear(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 2))
        y = np.exp(np.clip(X[:, 0], -5, 5)) + np.abs(X[:, 1])
        result = KolmogorovArnoldNetwork.symbolic_regression(X, y, n_sweeps=3)
        assert len(result["best_expressions"]) > 0
        assert result["r_squared_approx"] >= 0

    def test_kan_unfitted_predict(self):
        kan = KolmogorovArnoldNetwork()
        X = np.array([[1.0, 2.0, 3.0]])
        preds = kan.predict(X)
        assert len(preds) == 1
        assert preds[0] == 0.0

    def test_kan_unfitted_explain(self):
        kan = KolmogorovArnoldNetwork()
        X = np.array([[1.0, 2.0, 3.0]])
        result = kan.explain(X)
        assert result.prediction == 0.0
        assert result.contributions == []
        assert result.feature_importance == {}

    def test_kan_decomposition_small_input(self):
        f_values = np.array([1.0, 2.0, 3.0])
        result = KolmogorovArnoldNetwork.kolmogorov_decomposition(f_values, n_inputs=2)
        assert "inner_functions" in result
        assert result["inner_functions"] == []

    def test_kan_grid_size_variation(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (50, 2))
        y = X[:, 0] + rng.normal(0, 0.3, 50)
        kan = KolmogorovArnoldNetwork(grid_size=3, spline_order=2)
        kan.fit(X, y, max_iter=100)
        preds = kan.predict(X)
        assert len(preds) == 50