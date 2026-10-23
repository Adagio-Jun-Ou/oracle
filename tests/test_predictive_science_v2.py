import pytest
import numpy as np
from oracle.prediction.advanced_ensemble import (
    AdvancedEnsemble, EnsembleResult, StackingResult,
    BayesianModelAveragingResult, SuperLearnerResult,
)
from oracle.prediction.conditional_var import (
    ConditionalVaRDecomposition, ComponentVaRResult,
    FactorDecompositionResult, StressVaRResult,
)
from oracle.nondeterministic.causal_advanced import (
    DoubleMachineLearning, DMLResult, RDDResult,
    IV2SLSResult, DifferenceInDifferencesResult,
)
from oracle.prediction.bayesian_optimization import (
    BayesianOptimizer, AcquisitionResult, OptimizationResult, HyperparameterResult,
)
from oracle.prediction.gaussian_process import GaussianProcessPredictor, GPPrediction
from oracle.nondeterministic.extreme_value import ExtremeValueTheory, GEVResult, GPDRresult, TailDependenceResult
from oracle.prediction.wavelet import WaveletDecomposition, WaveletResult, MultiResolutionResult


@pytest.fixture
def rng():
    return np.random.RandomState(42)


class TestAdvancedEnsemble:
    def test_crps_weighting(self, rng):
        n = 200
        actuals = rng.normal(0, 1, n)
        preds = np.column_stack([actuals + rng.normal(0, 0.3, n) for _ in range(4)])
        result = AdvancedEnsemble.crps_weighting(preds, actuals)
        assert "weights" in result
        assert "crps_scores" in result
        assert "weighted_prediction" in result
        assert len(result["weights"]) == 4
        assert abs(sum(result["weights"].values()) - 1.0) < 0.01

    def test_optimal_combination(self, rng):
        n = 200
        actuals = rng.normal(0, 1, n)
        preds = np.column_stack([actuals + rng.normal(0, 0.3 * (i + 1), n) for i in range(3)])
        result = AdvancedEnsemble.optimal_combination(preds, actuals)
        assert "weights" in result
        assert "combined_mse" in result
        assert result["combined_mse"] >= 0

    def test_stacking(self, rng):
        n = 200
        actuals = rng.normal(0, 1, n)
        preds = np.column_stack([actuals + rng.normal(0, 0.3, n) for _ in range(4)])
        result = AdvancedEnsemble.stacking(preds, actuals, n_folds=5)
        assert isinstance(result, StackingResult)
        assert isinstance(result.prediction, float)
        assert len(result.meta_coefficients) == 4
        assert 0 <= result.oof_score <= 1.5

    def test_bayesian_model_averaging(self, rng):
        n = 200
        actuals = rng.normal(0, 1, n)
        preds = np.column_stack([actuals + rng.normal(0, 0.5 * (i + 1), n) for i in range(4)])
        result = AdvancedEnsemble.bayesian_model_averaging(preds, actuals)
        assert isinstance(result, BayesianModelAveragingResult)
        assert isinstance(result.prediction, float)
        assert abs(sum(result.posterior_probabilities.values()) - 1.0) < 0.01
        assert result.model_uncertainty >= 0

    def test_super_learner(self, rng):
        n = 200
        actuals = rng.normal(0, 1, n)
        preds = np.column_stack([actuals + rng.normal(0, 0.3 * (i + 1), n) for i in range(4)])
        result = AdvancedEnsemble.super_learner(preds, actuals)
        assert isinstance(result, SuperLearnerResult)
        assert isinstance(result.prediction, float)
        assert len(result.weights) == 4
        assert abs(sum(result.weights) - 1.0) < 0.01
        assert result.discrete_winner.startswith("model_")

    def test_diversity_analysis(self, rng):
        preds = np.column_stack([rng.normal(0, 1 + 0.5 * i, 200) for i in range(4)])
        result = AdvancedEnsemble.diversity_analysis(preds)
        assert "diversity_score" in result
        assert "avg_correlation" in result
        assert "disagreement" in result
        assert 0 <= result["diversity_score"] <= 1

    def test_diversity_single_model(self, rng):
        preds = rng.normal(0, 1, (200, 1))
        result = AdvancedEnsemble.diversity_analysis(preds)
        assert result["diversity_score"] == 0.0

    def test_forecast_reconciliation(self):
        forecasts = np.array([10.0, 20.0, 30.0])
        result = AdvancedEnsemble.forecast_reconciliation(forecasts)
        assert "base_forecasts" in result
        assert "reconciled_forecasts" in result
        assert "method" in result


class TestConditionalVaR:
    def test_component_var(self, rng):
        n = 252
        n_assets = 5
        returns = rng.normal(0.0005, 0.02, (n, n_assets))
        weights = np.ones(n_assets) / n_assets
        cvd = ConditionalVaRDecomposition(confidence_level=0.95)
        result = cvd.component_var(returns, weights)
        assert isinstance(result, ComponentVaRResult)
        assert result.total_var > 0
        assert len(result.component_vars) == n_assets
        assert abs(sum(result.percent_contributions.values()) - 1.0) < 0.05

    def test_factor_decomposition(self, rng):
        n = 252
        returns = rng.normal(0.0005, 0.02, (n, 5))
        factor_returns = rng.normal(0, 0.01, (n, 3))
        weights = np.ones(5) / 5
        cvd = ConditionalVaRDecomposition(confidence_level=0.95)
        result = cvd.factor_decomposition(returns, factor_returns, weights)
        assert isinstance(result, FactorDecompositionResult)
        assert result.systematic_var >= 0
        assert result.idiosyncratic_var >= 0
        assert 0 <= result.r_squared <= 1

    def test_stress_var(self, rng):
        returns = rng.normal(0.0005, 0.02, (252, 5))
        weights = np.ones(5) / 5
        cvd = ConditionalVaRDecomposition(confidence_level=0.95)
        shocks = {"factor_0": -0.05, "factor_1": 0.02}
        result = cvd.stress_var(returns, weights, shocks)
        assert isinstance(result, StressVaRResult)
        assert result.stressed_var > result.baseline_var

    def test_marginal_var_contribution(self, rng):
        returns = rng.normal(0.0005, 0.02, (252, 5))
        weights = np.ones(5) / 5
        result = ConditionalVaRDecomposition.marginal_var_contribution(returns, weights)
        assert "marginal_var_contributions" in result
        assert len(result["marginal_var_contributions"]) == 5


class TestDoubleML:
    def test_dml_fit(self, rng):
        n = 200
        X = rng.normal(0, 1, (n, 3))
        D = (X[:, 0] > 0).astype(float)
        Y = 2.0 * D + X[:, 0] + rng.normal(0, 0.5, n)
        dml = DoubleMachineLearning(n_cross_fits=2, seed=42)
        result = dml.fit(Y, D, X, n_cross_fits=2)
        assert isinstance(result, DMLResult)
        assert isinstance(result.treatment_effect, float)
        assert result.treatment_effect > 0.5
        assert 0 <= result.p_value <= 1

    def test_rdd(self, rng):
        n = 200
        X = rng.normal(0, 1, n)
        Y = 3.0 * (X >= 0).astype(float) + X + rng.normal(0, 0.5, n)
        result = DoubleMachineLearning.regression_discontinuity(Y, X, cutoff=0.0)
        assert isinstance(result, RDDResult)
        assert result.treatment_effect > 0
        assert result.bandwidth > 0

    def test_iv_2sls(self, rng):
        n = 200
        Z = rng.normal(0, 1, n)
        D = 0.5 * Z + rng.normal(0, 0.5, n)
        Y = 3.0 * D + rng.normal(0, 0.5, n)
        result = DoubleMachineLearning.iv_2sls(Y, D, Z)
        assert isinstance(result, IV2SLSResult)
        assert result.treatment_effect > 1.0
        assert result.weak_instrument_test in ("strong", "weak")

    def test_did(self, rng):
        n = 50
        Y_t_pre = rng.normal(10, 1, n)
        Y_t_post = rng.normal(13, 1, n)
        Y_c_pre = rng.normal(10, 1, n)
        Y_c_post = rng.normal(11, 1, n)
        result = DoubleMachineLearning.difference_in_differences(Y_t_pre, Y_t_post, Y_c_pre, Y_c_post)
        assert isinstance(result, DifferenceInDifferencesResult)
        assert result.treatment_effect > 0


class TestBayesianOptimizer:
    def test_expected_improvement(self, rng):
        X = rng.uniform(-3, 3, (20, 1))
        y = -X[:, 0] ** 2
        X_candidates = np.linspace(-3, 3, 20).reshape(-1, 1)
        opt = BayesianOptimizer()
        result = opt.expected_improvement(X, y, X_candidates)
        assert isinstance(result, AcquisitionResult)
        assert len(result.x_next) == 1
        assert result.method == "expected_improvement"

    def test_ucb(self, rng):
        X = rng.uniform(-3, 3, (20, 1))
        y = -X[:, 0] ** 2
        X_candidates = np.linspace(-3, 3, 20).reshape(-1, 1)
        opt = BayesianOptimizer()
        result = opt.upper_confidence_bound(X, y, X_candidates)
        assert isinstance(result, AcquisitionResult)
        assert result.method == "ucb"

    def test_probability_of_improvement(self, rng):
        X = rng.uniform(-3, 3, (20, 1))
        y = -X[:, 0] ** 2
        X_candidates = np.linspace(-3, 3, 20).reshape(-1, 1)
        opt = BayesianOptimizer()
        result = opt.probability_of_improvement(X, y, X_candidates)
        assert isinstance(result, AcquisitionResult)
        assert result.method == "probability_of_improvement"

    def test_optimize(self):
        def objective(x):
            return -((x[0] - 1.0) ** 2)
        opt = BayesianOptimizer(seed=42)
        result = opt.optimize(objective, bounds=[(-5.0, 5.0)], n_iterations=8, acquisition="ei")
        assert isinstance(result, OptimizationResult)
        assert len(result.best_x) == 1
        assert result.best_value > -1.5
        assert len(result.convergence_history) > 0

    def test_hyperparameter_optimization(self):
        param_space = {"lr": (0.001, 0.1), "depth": [3, 5, 7]}
        def objective(params):
            return -abs(params.get("lr", 0.01) - 0.05) + params.get("depth", 5) * 0.1
        opt = BayesianOptimizer(seed=42)
        result = opt.hyperparameter_optimization(param_space, objective, n_trials=10)
        assert isinstance(result, HyperparameterResult)
        assert len(result.all_trials) == 10
        assert "lr" in result.param_importance


class TestGaussianProcess:
    def test_fit_predict(self, rng):
        X = np.linspace(-5, 5, 30).reshape(-1, 1)
        y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 30)
        gp = GaussianProcessPredictor(kernel="rbf", length_scale=2.0)
        gp.fit(X, y)
        X_test = np.linspace(-4, 4, 10).reshape(-1, 1)
        result = gp.predict(X_test)
        assert isinstance(result, GPPrediction)
        assert len(result.mean) == 10
        assert len(result.std) == 10
        assert len(result.confidence_lower) == 10
        assert len(result.confidence_upper) == 10

    def test_kernel_optimization(self, rng):
        X = np.linspace(-5, 5, 30).reshape(-1, 1)
        y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 30)
        gp = GaussianProcessPredictor(seed=42)
        result = gp.optimize_kernel(X, y)
        assert "best_length_scale" in result
        assert "best_variance" in result
        assert "log_likelihood" in result

    def test_temporal_forecast(self, rng):
        times = np.arange(50, dtype=float)
        values = 100 + np.cumsum(rng.normal(0, 1, 50))
        result = GaussianProcessPredictor.temporal_forecast(times, values, n_ahead=5)
        assert "forecast_mean" in result
        assert "forecast_lower" in result
        assert "forecast_upper" in result
        assert len(result["forecast_mean"]) == 5

    def test_unfitted_predict(self):
        gp = GaussianProcessPredictor()
        X_test = np.linspace(-3, 3, 5).reshape(-1, 1)
        result = gp.predict(X_test)
        assert result.mean == []
        assert result.n_training_points == 0

    def test_matern_kernel(self, rng):
        X = np.linspace(-5, 5, 20).reshape(-1, 1)
        y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 20)
        gp = GaussianProcessPredictor(kernel="matern", length_scale=2.0)
        gp.fit(X, y)
        X_test = np.linspace(-4, 4, 5).reshape(-1, 1)
        result = gp.predict(X_test)
        assert len(result.mean) == 5

    def test_periodic_kernel(self, rng):
        X = np.linspace(-5, 5, 20).reshape(-1, 1)
        y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 20)
        gp = GaussianProcessPredictor(kernel="periodic", length_scale=2.0)
        gp.fit(X, y)
        X_test = np.linspace(-4, 4, 5).reshape(-1, 1)
        result = gp.predict(X_test)
        assert len(result.mean) == 5


class TestExtremeValueTheory:
    def test_gev_fit(self, rng):
        data = rng.gumbel(0, 1, 200)
        result = ExtremeValueTheory.gev_fit(data)
        assert isinstance(result, GEVResult)
        assert "10_year" in result.return_levels
        assert "50_year" in result.return_levels
        assert result.scale > 0

    def test_gpd_fit(self, rng):
        data = rng.normal(0, 1, 200) ** 2
        result = ExtremeValueTheory.gpd_fit(data, threshold_quantile=0.95)
        assert isinstance(result, GPDRresult)
        assert result.n_exceedances > 0

    def test_pot_analysis(self, rng):
        data = rng.normal(0, 1, 500)
        result = ExtremeValueTheory.pot_analysis(data, threshold_quantile=0.95)
        assert "threshold" in result
        assert "simulated_95_max" in result
        assert result["n_exceedances"] > 0

    def test_tail_dependence(self, rng):
        x = rng.normal(0, 1, 200)
        y = 0.7 * x + rng.normal(0, 0.5, 200)
        result = ExtremeValueTheory.tail_dependence(x, y)
        assert isinstance(result, TailDependenceResult)
        assert 0 <= result.upper_tail_dependence <= 1
        assert 0 <= result.lower_tail_dependence <= 1

    def test_hill_estimator(self, rng):
        data = rng.pareto(1.5, 200)
        result = ExtremeValueTheory.hill_estimator(data)
        assert "tail_index" in result
        assert result["tail_index"] > 0

    def test_block_maxima(self, rng):
        data = rng.gumbel(0, 1, 500)
        result = ExtremeValueTheory.block_maxima(data, block_size=21)
        assert isinstance(result, GEVResult)
        assert result.scale > 0

    def test_gpd_insufficient_data(self, rng):
        data = rng.normal(0, 1, 10)
        result = ExtremeValueTheory.gpd_fit(data, threshold_quantile=0.99)
        assert result.method == "gpd_insufficient"


class TestWaveletDecomposition:
    def test_haar_decompose(self, rng):
        t = np.linspace(0, 4 * np.pi, 128)
        signal = np.sin(t) + rng.normal(0, 0.1, 128)
        wd = WaveletDecomposition(wavelet="haar")
        result = wd.decompose(signal, levels=3)
        assert isinstance(result, WaveletResult)
        assert result.levels == 3
        assert len(result.details) == 3
        assert "approximation" in result.energy_by_level

    def test_d4_decompose(self, rng):
        signal = rng.normal(0, 1, 128)
        wd = WaveletDecomposition(wavelet="d4")
        result = wd.decompose(signal, levels=2)
        assert isinstance(result, WaveletResult)
        assert result.wavelet == "d4"

    def test_mra(self, rng):
        signal = rng.normal(0, 1, 256)
        result = WaveletDecomposition.multi_resolution_analysis(signal, levels=4)
        assert isinstance(result, MultiResolutionResult)
        assert "approximation" in result.components
        assert result.total_energy > 0

    def test_change_points(self, rng):
        signal = np.concatenate([rng.normal(0, 1, 128), rng.normal(3, 1, 128)])
        result = WaveletDecomposition.detect_change_points(signal)
        assert "change_points" in result
        assert result["n_detected"] >= 0