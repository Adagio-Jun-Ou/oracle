"""Tests for RNN, HMM, and Copula-GARCH predictive science modules."""
import pytest
import numpy as np


class TestRNN:
    def test_rnn_predictor_init(self):
        from oracle.prediction.rnn import RNNPredictor
        rnn = RNNPredictor(hidden_dim=16, seed=42)
        assert rnn.hidden_dim == 16
        assert rnn.model is None

    def test_rnn_fit_and_forecast(self):
        from oracle.prediction.rnn import RNNPredictor
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.normal(0, 1, 80))
        rnn = RNNPredictor(hidden_dim=8, sequence_length=10, seed=42)
        rnn.fit(data, n_epochs=10)
        result = rnn.forecast(n_ahead=5)
        assert len(result.predictions) == 5
        assert result.method == "lstm"
        assert result.n_epochs == 10
        assert len(result.loss_history) > 0

    def test_rnn_result_dataclass(self):
        from oracle.prediction.rnn import RNNResult
        result = RNNResult(
            predictions=[1.0, 2.0, 3.0],
            hidden_states=[[0.1, 0.2]],
            loss_history=[0.5, 0.3],
            n_epochs=2,
            gradient_norm=0.01,
            sequence_length=10,
        )
        assert len(result.predictions) == 3
        assert result.method == "lstm"

    def test_lstm_network_init(self):
        from oracle.prediction.rnn import LSTMNetwork
        net = LSTMNetwork(input_dim=1, hidden_dim=16, seed=42)
        assert net.hidden_dim == 16
        assert net.W_ih.shape[0] == 64
        assert net.W_out.shape[0] == 1

    def test_lstm_forward(self):
        from oracle.prediction.rnn import LSTMNetwork
        net = LSTMNetwork(input_dim=1, hidden_dim=16, seed=42)
        x = np.random.randn(20, 1)
        outputs, hidden, cell, gates = net._forward(x)
        assert len(hidden) == 20
        assert len(cell) == 20
        assert hidden[0].shape == (16,)

    def test_rnn_hidden_state_analysis(self):
        from oracle.prediction.rnn import RNNPredictor
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.normal(0, 1, 80))
        rnn = RNNPredictor(hidden_dim=8, seed=42)
        rnn.fit(data, n_epochs=10)
        result = rnn.hidden_state_analysis(data)
        assert "hidden_norms" in result
        assert "mean_norm" in result
        assert result["n_timesteps"] > 0

    def test_rnn_forecast_not_fitted_raises(self):
        from oracle.prediction.rnn import RNNPredictor
        rnn = RNNPredictor()
        with pytest.raises(ValueError):
            rnn.forecast(n_ahead=5)


class TestHMM:
    def test_hmm_init(self):
        from oracle.prediction.hmm import HiddenMarkovModel
        hmm = HiddenMarkovModel(n_states=3, seed=42)
        assert hmm.n_states == 3
        assert not hmm._fitted

    def test_hmm_fit(self):
        from oracle.prediction.hmm import HiddenMarkovModel
        np.random.seed(42)
        data = np.concatenate([np.random.normal(0, 1, 50),
                               np.random.normal(3, 1, 50),
                               np.random.normal(-1, 0.5, 50)])
        hmm = HiddenMarkovModel(n_states=3, seed=42)
        hmm.fit(data)
        assert hmm._fitted
        assert hmm.pi is not None
        assert hmm.A.shape == (3, 3)

    def test_hmm_decode(self):
        from oracle.prediction.hmm import HiddenMarkovModel
        np.random.seed(42)
        data = np.concatenate([np.random.normal(0, 1, 50),
                               np.random.normal(3, 1, 50),
                               np.random.normal(-1, 0.5, 50)])
        hmm = HiddenMarkovModel(n_states=3, seed=42)
        result = hmm.decode(data)
        assert len(result.states) == 150
        assert len(result.state_probabilities) == 150
        assert result.n_states == 3
        assert result.method == "hmm_viterbi"

    def test_hmm_regime_detection(self):
        from oracle.prediction.hmm import HiddenMarkovModel
        np.random.seed(42)
        data = np.concatenate([np.random.normal(0, 1, 50),
                               np.random.normal(3, 1, 50),
                               np.random.normal(-1, 0.5, 50)])
        hmm = HiddenMarkovModel(n_states=3, seed=42)
        result = hmm.detect_regimes(data)
        assert len(result.regimes) == 150
        assert result.n_regimes == 3
        assert len(result.regime_labels) == 3
        assert len(result.regime_statistics) == 3

    def test_hmm_forecast(self):
        from oracle.prediction.hmm import HiddenMarkovModel
        np.random.seed(42)
        data = np.concatenate([np.random.normal(0, 1, 50),
                               np.random.normal(3, 1, 50)])
        hmm = HiddenMarkovModel(n_states=3, seed=42)
        result = hmm.forecast(data, n_ahead=5)
        assert len(result["forecast"]) == 5
        assert len(result["confidence_lower"]) == 5
        assert len(result["confidence_upper"]) == 5
        assert "current_state" in result

    def test_hmm_regime_change(self):
        from oracle.prediction.hmm import HiddenMarkovModel
        np.random.seed(42)
        data = np.concatenate([np.random.normal(0, 1, 50),
                               np.random.normal(3, 1, 50)])
        hmm = HiddenMarkovModel(n_states=3, seed=42)
        result = hmm.regime_change_probability(data)
        assert "n_changes" in result
        assert "stability" in result
        assert "average_durations" in result

    def test_hmm_duration_analysis(self):
        from oracle.prediction.hmm import HiddenMarkovModel
        np.random.seed(42)
        data = np.concatenate([np.random.normal(0, 1, 50),
                               np.random.normal(3, 1, 50)])
        hmm = HiddenMarkovModel(n_states=3, seed=42)
        result = hmm.state_duration_analysis(data)
        assert "duration_statistics" in result
        assert result["method"] == "hmm_duration"

    def test_hmm_insufficient_data(self):
        from oracle.prediction.hmm import HiddenMarkovModel
        hmm = HiddenMarkovModel(n_states=5, seed=42)
        with pytest.raises(ValueError):
            hmm.fit(np.array([1.0, 2.0, 3.0]))

    def test_hmm_result_dataclass(self):
        from oracle.prediction.hmm import HMMResult, RegimeResult
        r = HMMResult(
            states=[0, 1, 0],
            state_probabilities=[[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]],
            transition_matrix=[[0.7, 0.3], [0.4, 0.6]],
            emission_params=[{"mean": 0, "std": 1}],
            log_likelihood=-100.0,
            n_states=2,
            n_iterations=20,
            converged=True,
        )
        assert len(r.states) == 3


class TestCopulaGARCH:
    def test_garch_model_init(self):
        from oracle.prediction.copula_garch import GARCHModel
        garch = GARCHModel(omega=0.01, alpha=0.1, beta=0.85)
        assert garch.omega == 0.01
        assert garch.alpha == 0.1

    def test_garch_fit(self):
        from oracle.prediction.copula_garch import GARCHModel
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        garch = GARCHModel()
        result = garch.fit(returns)
        assert len(result.conditional_variance) > 0
        assert len(result.standardized_residuals) > 0
        assert result.omega > 0
        assert 0 <= result.alpha <= 1
        assert 0 <= result.beta <= 1
        assert result.n_observations == 100

    def test_copula_garch_init(self):
        from oracle.prediction.copula_garch import CopulaGARCH
        cg = CopulaGARCH(copula_type="auto", seed=42)
        assert cg.copula_type == "auto"
        assert not cg._fitted

    def test_copula_garch_fit_bivariate(self):
        from oracle.prediction.copula_garch import CopulaGARCH
        np.random.seed(42)
        n = 150
        returns1 = np.random.normal(0, 0.02, n)
        returns2 = 0.5 * returns1 + np.random.normal(0, 0.015, n)
        data = np.column_stack([returns1, returns2])
        cg = CopulaGARCH(copula_type="auto", seed=42)
        result = cg.fit(data)
        assert result.copula_type in ["gaussian", "clayton", "gumbel", "frank"]
        assert -1 <= result.conditional_correlation <= 1
        assert 0 <= result.tail_dependence_lower <= 1
        assert 0 <= result.tail_dependence_upper <= 1

    def test_copula_garch_fit_clayton(self):
        from oracle.prediction.copula_garch import CopulaGARCH
        np.random.seed(42)
        n = 150
        returns1 = np.random.normal(0, 0.02, n)
        returns2 = 0.7 * returns1 + np.random.normal(0, 0.01, n)
        data = np.column_stack([returns1, returns2])
        cg = CopulaGARCH(copula_type="clayton", seed=42)
        result = cg.fit(data)
        assert result.copula_type == "clayton"

    def test_copula_garch_risk_metrics(self):
        from oracle.prediction.copula_garch import CopulaGARCH
        np.random.seed(42)
        n = 150
        returns1 = np.random.normal(0, 0.02, n)
        returns2 = 0.5 * returns1 + np.random.normal(0, 0.015, n)
        data = np.column_stack([returns1, returns2])
        cg = CopulaGARCH(copula_type="auto", seed=42)
        cg.fit(data)
        risks = cg.risk_metrics(data)
        assert "series_0" in risks
        assert "series_1" in risks
        assert "joint" in risks
        assert "VaR_95" in risks["series_0"]

    def test_copula_garch_result_dataclass(self):
        from oracle.prediction.copula_garch import CopulaGARCHResult
        r = CopulaGARCHResult(
            marginal_results=[],
            copula_rho=0.5,
            copula_type="gaussian",
            joint_log_likelihood=-100.0,
            conditional_correlation=0.6,
            tail_dependence_lower=0.1,
            tail_dependence_upper=0.15,
            forecast_correlation=0.6,
        )
        assert r.copula_type == "gaussian"
        assert r.copula_rho == 0.5

    def test_empirical_cdf(self):
        from oracle.prediction.copula_garch import CopulaGARCH
        cg = CopulaGARCH(seed=42)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        u = cg._empirical_cdf(x)
        assert len(u) == 5
        assert 0 < u[0] < 1
        assert u[-1] > u[0]

    def test_kendall_tau(self):
        from oracle.prediction.copula_garch import CopulaGARCH
        cg = CopulaGARCH(seed=42)
        u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        v = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
        tau = cg._kendall_tau(u, v)
        assert -1 <= tau <= 1