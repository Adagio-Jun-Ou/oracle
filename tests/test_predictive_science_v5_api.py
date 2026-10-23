"""Web API tests for Koopman, ABM, and Symbolic Regression endpoints."""
import pytest
from fastapi.testclient import TestClient
from oracle.web.app import app

client = TestClient(app)


class TestKoopmanAPI:
    def test_koopman_dmd(self):
        resp = client.post("/api/koopman/dmd")
        assert resp.status_code == 200
        data = resp.json()
        assert "n_modes" in data
        assert data["method"] == "koopman_dmd"

    def test_koopman_edmd(self):
        resp = client.post("/api/koopman/edmd")
        assert resp.status_code == 200
        data = resp.json()
        assert "n_observables" in data
        assert "koopman_edmd" in data["method"]


class TestABMAPI:
    def test_abm_market(self):
        resp = client.post("/api/abm/market")
        assert resp.status_code == 200
        data = resp.json()
        assert "statistics" in data
        assert "emergent_patterns" in data
        assert data["method"] == "abm_market"

    def test_abm_epidemic(self):
        resp = client.post("/api/abm/epidemic")
        assert resp.status_code == 200
        data = resp.json()
        assert "statistics" in data
        assert "susceptible" in data

    def test_abm_opinion(self):
        resp = client.post("/api/abm/opinion")
        assert resp.status_code == 200
        data = resp.json()
        assert "statistics" in data
        assert "polarization" in data


class TestSymbolicRegressionAPI:
    def test_symbolic_fit(self):
        resp = client.post("/api/symbolic-regression/fit")
        assert resp.status_code == 200
        data = resp.json()
        assert "expression" in data
        assert "r_squared" in data
        assert "rmse" in data

    def test_symbolic_fit_params(self):
        resp = client.post("/api/symbolic-regression/fit?population_size=20&n_generations=10&max_depth=3")
        assert resp.status_code == 200
        data = resp.json()
        assert "expression" in data