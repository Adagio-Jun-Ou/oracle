import pytest
import numpy as np
from oracle.prediction.vine_copula import VineCopulaModel, VineCopulaResult, VineSimulationResult, TailDepMapResult
from oracle.prediction.neural_ode import NeuralODE, NeuralODEResult, ODEForeResult, NormalizingFlow, NormalizingFlowResult
from oracle.nondeterministic.survival import SurvivalAnalysis, SurvivalResult, CoxResult, KaplanMeierResult
from oracle.prediction.information_theory import InformationTheoreticPredictor, TransferEntropyResult, MutualInfoResult, GrangerCausalityResult
from oracle.prediction.topological import TopologicalDataAnalysis, PersistenceResult, TopologicalFeatureResult, ShapeDescriptor
from oracle.prediction.state_space import StateSpaceModel, KalmanSmoothResult, StateSpaceResult


@pytest.fixture
def rng():
    return np.random.RandomState(42)


class TestVineCopula:
    def test_fit(self, rng):
        data = rng.normal(0, 1, (100, 3))
        data[:, 1] = 0.7 * data[:, 0] + rng.normal(0, 0.5, 100)
        vc = VineCopulaModel()
        result = vc.fit(data)
        assert isinstance(result, VineCopulaResult)
        assert result.n_pairs > 0
        assert len(result.pair_copulas) > 0

    def test_tail_dependence_map(self, rng):
        data = rng.normal(0, 1, (150, 3))
        data[:, 1] = 0.6 * data[:, 0] + rng.normal(0, 0.5, 150)
        result = VineCopulaModel.tail_dependence_map(data)
        assert isinstance(result, TailDepMapResult)
        assert len(result.upper_tail) > 0
        assert len(result.lower_tail) > 0

    def test_conditional_simulation(self):
        result = VineCopulaModel.conditional_simulation("clayton", theta=2.0, n_samples=100)
        assert "conditional_mean" in result
        assert "conditional_quantiles" in result
        assert result["copula_type"] == "clayton"


class TestNeuralODE:
    def test_fit_and_predict(self, rng):
        times = np.linspace(0, 10, 50)
        values = np.sin(times) + rng.normal(0, 0.1, 50)
        node = NeuralODE(hidden_dim=8, seed=42)
        node.fit(times, values, n_epochs=30, lr=0.01)
        result = node.predict(times[:10])
        assert isinstance(result, NeuralODEResult)
        assert len(result.times) == 10

    def test_forecast(self, rng):
        times = np.linspace(0, 10, 50)
        values = np.sin(times) + rng.normal(0, 0.1, 50)
        node = NeuralODE(hidden_dim=8, seed=42)
        node.fit(times, values, n_epochs=30)
        result = node.forecast(n_ahead=5)
        assert isinstance(result, ODEForeResult)
        assert len(result.forecast_values) == 5
        assert len(result.confidence_lower) == 5

    def test_unfitted_predict(self):
        node = NeuralODE()
        result = node.predict(np.array([0.0, 1.0]))
        assert result.n_steps == 0


class TestNormalizingFlow:
    def test_fit(self, rng):
        data = rng.normal(0, 1, 200)
        nf = NormalizingFlow(n_layers=4, seed=42)
        result = nf.fit(data)
        assert isinstance(result, NormalizingFlowResult)
        assert result.n_layers == 4

    def test_sample(self, rng):
        data = rng.normal(0, 1, 200)
        nf = NormalizingFlow(n_layers=4, seed=42)
        nf.fit(data)
        samples = nf.sample(100)
        assert len(samples) == 100

    def test_log_prob(self, rng):
        data = rng.normal(0, 1, 200)
        nf = NormalizingFlow(n_layers=4, seed=42)
        nf.fit(data)
        log_probs = nf.log_prob(data[:10])
        assert len(log_probs) == 10


class TestSurvivalAnalysis:
    def test_kaplan_meier(self, rng):
        times = rng.exponential(10, 200)
        events = rng.binomial(1, 0.7, 200)
        result = SurvivalAnalysis.kaplan_meier(times, events)
        assert isinstance(result, KaplanMeierResult)
        assert result.median_survival >= 0
        assert len(result.survival) > 0

    def test_cox_regression(self, rng):
        X = rng.normal(0, 1, (200, 3))
        times = rng.exponential(10, 200)
        events = rng.binomial(1, 0.7, 200)
        result = SurvivalAnalysis.cox_regression(X, times, events)
        assert isinstance(result, CoxResult)
        assert len(result.coefficients) == 3
        assert result.concordance >= 0

    def test_exponential_survival(self, rng):
        times = rng.exponential(10, 200)
        events = rng.binomial(1, 0.7, 200)
        result = SurvivalAnalysis.exponential_survival(times, events)
        assert isinstance(result, SurvivalResult)
        assert result.median_survival > 0
        assert result.mean_survival > 0

    def test_weibull_survival(self, rng):
        times = rng.exponential(10, 200)
        events = rng.binomial(1, 0.7, 200)
        result = SurvivalAnalysis.weibull_survival(times, events)
        assert isinstance(result, SurvivalResult)
        assert result.median_survival > 0
        assert result.method == "weibull"


class TestInformationTheory:
    def test_transfer_entropy(self, rng):
        source = rng.normal(0, 1, 200)
        target = 0.7 * source[:-1] + rng.normal(0, 0.5, 199)
        result = InformationTheoreticPredictor.transfer_entropy(source[:-1], target)
        assert isinstance(result, TransferEntropyResult)
        assert result.transfer_entropy >= 0

    def test_mutual_information(self, rng):
        x = rng.normal(0, 1, 200)
        y = 0.7 * x + rng.normal(0, 0.5, 200)
        result = InformationTheoreticPredictor.mutual_information(x, y)
        assert isinstance(result, MutualInfoResult)
        assert result.mutual_information >= 0
        assert 0 <= result.normalized_mi <= 1

    def test_granger_causality(self, rng):
        x = rng.normal(0, 1, 200)
        y = np.zeros(200)
        y[1:] = 0.5 * x[:-1] + rng.normal(0, 0.3, 199)
        result = InformationTheoreticPredictor.granger_causality(x, y)
        assert isinstance(result, GrangerCausalityResult)
        assert result.f_statistic >= 0

    def test_conditional_entropy(self, rng):
        x = rng.normal(0, 1, 200)
        y = 0.6 * x + rng.normal(0, 0.5, 200)
        result = InformationTheoreticPredictor.conditional_entropy(x, y)
        assert result.conditional_entropy >= 0
        assert result.mutual_information >= 0

    def test_active_info_storage(self, rng):
        series = rng.normal(0, 1, 200)
        result = InformationTheoreticPredictor.active_info_storage(series)
        assert "ais" in result
        assert result["method"] == "active_info_storage"


class TestTopologicalDataAnalysis:
    def test_vietoris_rips(self, rng):
        theta = np.linspace(0, 2 * np.pi, 30)
        data = np.column_stack([np.cos(theta), np.sin(theta)])
        tda = TopologicalDataAnalysis()
        result = tda.vietoris_rips_persistence(data, max_dim=1)
        assert isinstance(result, PersistenceResult)
        assert result.n_features > 0
        assert "dim_0" in result.betti_numbers

    def test_shape_descriptor(self, rng):
        data = rng.normal(0, 1, (50, 3))
        result = TopologicalDataAnalysis.shape_descriptor(data)
        assert isinstance(result, ShapeDescriptor)
        assert result.connected_components >= 1

    def test_time_series_persistence(self, rng):
        series = np.cumsum(rng.normal(0, 1, 100))
        result = TopologicalDataAnalysis.time_series_persistence(series)
        assert result.n_features >= 0


class TestStateSpaceModel:
    def test_fit_and_smooth(self, rng):
        y = 100 + np.cumsum(rng.normal(0, 1, 100))
        ssm = StateSpaceModel()
        ssm.fit(y)
        result = ssm.smooth()
        assert isinstance(result, KalmanSmoothResult)
        assert result.n_steps == 100
        assert len(result.filtered_means) == 100

    def test_forecast(self, rng):
        y = 100 + np.cumsum(rng.normal(0, 1, 100))
        ssm = StateSpaceModel()
        ssm.fit(y)
        result = ssm.forecast(n_ahead=10)
        assert isinstance(result, StateSpaceResult)
        assert len(result.forecast) == 10
        assert len(result.forecast_lower) == 10

    def test_arima_state_space(self, rng):
        y = 100 + np.cumsum(rng.normal(0, 1, 100))
        result = StateSpaceModel.arima_state_space(y, p=1, d=1, q=1, n_ahead=10)
        assert "forecast" in result
        assert "ar_coefficient" in result
        assert result["method"] == "arima_state_space"