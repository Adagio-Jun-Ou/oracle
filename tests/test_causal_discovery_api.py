"""Web API tests for Causal Discovery endpoints."""
import pytest
from fastapi.testclient import TestClient
from oracle.web.app import app

client = TestClient(app)


class TestCausalDiscoveryAPI:
    def test_pc_algorithm(self):
        resp = client.post("/api/causal-discovery/pc")
        assert resp.status_code == 200
        data = resp.json()
        assert "n_edges" in data
        assert "n_directed" in data
        assert "method" in data
        assert data["method"] == "pc_algorithm"

    def test_pc_algorithm_params(self):
        resp = client.post("/api/causal-discovery/pc?n_vars=3&n_samples=200&alpha=0.1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alpha"] == 0.1

    def test_ges_algorithm(self):
        resp = client.post("/api/causal-discovery/ges")
        assert resp.status_code == 200
        data = resp.json()
        assert "n_edges" in data
        assert "bic_score" in data
        assert data["method"] == "ges_algorithm"

    def test_ges_algorithm_params(self):
        resp = client.post("/api/causal-discovery/ges?n_vars=3&penalty=2.0")
        assert resp.status_code == 200
        data = resp.json()
        assert "edges" in data

    def test_compare_algorithms(self):
        resp = client.post("/api/causal-discovery/compare")
        assert resp.status_code == 200
        data = resp.json()
        assert "pc_result" in data
        assert "ges_result" in data
        assert "agreement" in data
        assert "agreement_rate" in data["agreement"]