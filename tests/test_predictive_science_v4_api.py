"""Web API tests for RNN, HMM, Copula-GARCH, and TFT endpoints."""
import pytest
from fastapi.testclient import TestClient
from oracle.web.app import app

client = TestClient(app)


class TestRNNAPI:
    def test_rnn_forecast(self):
        resp = client.post("/api/rnn/forecast")
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert "loss_history" in data
        assert "method" in data
        assert data["method"] == "lstm"

    def test_rnn_forecast_params(self):
        resp = client.post("/api/rnn/forecast?n_epochs=20&n_ahead=5&hidden_dim=8")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 5

    def test_rnn_hidden_state(self):
        resp = client.post("/api/rnn/hidden-state")
        assert resp.status_code == 200
        data = resp.json()
        assert "hidden_norms" in data
        assert "mean_norm" in data


class TestHMMAPI:
    def test_hmm_detect_regimes(self):
        resp = client.post("/api/hmm/detect-regimes")
        assert resp.status_code == 200
        data = resp.json()
        assert "regimes" in data
        assert "regime_labels" in data
        assert "regime_statistics" in data
        assert "transition_matrix" in data
        assert "n_regimes" in data

    def test_hmm_detect_regimes_params(self):
        resp = client.post("/api/hmm/detect-regimes?n_states=2&n_samples=100")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_regimes"] == 2

    def test_hmm_forecast(self):
        resp = client.post("/api/hmm/forecast?n_states=3&n_ahead=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "forecast" in data
        assert len(data["forecast"]) == 5
        assert "current_state" in data

    def test_hmm_regime_change(self):
        resp = client.post("/api/hmm/regime-change?n_states=3")
        assert resp.status_code == 200
        data = resp.json()
        assert "n_changes" in data
        assert "stability" in data


class TestCopulaGARCHAPI:
    def test_copula_garch_fit(self):
        resp = client.post("/api/copula-garch/fit")
        assert resp.status_code == 200
        data = resp.json()
        assert "copula_type" in data
        assert "copula_rho" in data
        assert "conditional_correlation" in data
        assert "tail_dependence_lower" in data
        assert "tail_dependence_upper" in data

    def test_copula_garch_risk(self):
        resp = client.post("/api/copula-garch/risk")
        assert resp.status_code == 200
        data = resp.json()
        assert "series_0" in data
        assert "series_1" in data
        assert "joint" in data
        assert "VaR_95" in data["series_0"]


class TestTFTAPI:
    def test_tft_forecast(self):
        resp = client.post("/api/tft/forecast")
        assert resp.status_code == 200
        data = resp.json()
        assert "forecast" in data
        assert "forecast_lower" in data
        assert "forecast_upper" in data
        assert "method" in data
        assert data["method"] == "transformer_tft"

    def test_tft_forecast_params(self):
        resp = client.post("/api/tft/forecast?n_epochs=10&n_ahead=5&d_model=8")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["forecast"]) == 5

    def test_tft_interpret(self):
        resp = client.post("/api/tft/interpret")
        assert resp.status_code == 200
        data = resp.json()
        assert "variable_importance" in data
        assert "n_layers" in data


class TestRLAPI:
    def test_q_learning(self):
        resp = client.post("/api/rl/q-learning")
        assert resp.status_code == 200
        data = resp.json()
        assert "optimal_policy" in data
        assert "optimal_values" in data
        assert data["method"] == "q_learning"

    def test_bandit(self):
        resp = client.post("/api/rl/bandit")
        assert resp.status_code == 200
        data = resp.json()
        assert "arm_values" in data
        assert "optimal_arm" in data

    def test_bandit_thompson(self):
        resp = client.post("/api/rl/bandit?method=thompson")
        assert resp.status_code == 200
        data = resp.json()
        assert "method" in data
        assert "thompson" in data["method"]

    def test_policy_gradient(self):
        resp = client.post("/api/rl/policy-gradient")
        assert resp.status_code == 200
        data = resp.json()
        assert "optimal_actions" in data
        assert data["method"] == "policy_gradient"