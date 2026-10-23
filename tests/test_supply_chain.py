import pytest
import numpy as np
from oracle.domains.supply_chain import SupplyChainRiskModel, SupplyChainState


class TestSupplyChainRiskModel:

    def setup_method(self):
        self.model = SupplyChainRiskModel()

    def test_assess_risk_default(self):
        result = self.model.assess_risk()
        assert "overall_risk" in result
        assert "risk_level" in result
        assert "concentration_risk" in result
        assert "disruption_risk" in result
        assert "resilience_score" in result
        assert "vulnerabilities" in result
        assert "recommendations" in result
        assert 0 <= result["overall_risk"] <= 1
        assert result["risk_level"] in ("low", "moderate", "high", "critical")

    def test_assess_risk_high_risk(self):
        state = SupplyChainState(
            supplier_concentration=0.80,
            single_source_dependency=0.70,
            logistics_disruption_prob=0.40,
            alternate_supplier_availability=0.20,
            visibility_score=0.25,
            lead_time_variability=0.40,
            cyber_exposure=0.50,
        )
        result = self.model.assess_risk(state)
        assert result["overall_risk"] > 0.40
        assert result["risk_level"] in ("high", "critical")
        assert len(result["vulnerabilities"]) >= 3

    def test_assess_risk_low_risk(self):
        state = SupplyChainState(
            supplier_concentration=0.15,
            single_source_dependency=0.10,
            logistics_disruption_prob=0.05,
            alternate_supplier_availability=0.90,
            visibility_score=0.85,
            inventory_buffer_days=60,
        )
        result = self.model.assess_risk(state)
        assert result["overall_risk"] < 0.40
        assert result["risk_level"] in ("low", "moderate")

    def test_disruption_simulation_default(self):
        result = self.model.disruption_simulation()
        assert "disruption_type" in result
        assert "revenue_impact" in result
        assert "avg_recovery_days" in result
        assert "cascade_risk" in result
        assert result["disruption_type"] == "supplier_failure"
        assert "p50" in result["revenue_impact"]
        assert result["avg_recovery_days"] > 0

    def test_disruption_simulation_types(self):
        for dtype in ["supplier_failure", "logistics_disruption", "cyber_attack", "natural_disaster", "geopolitical_shock", "quality_failure"]:
            result = self.model.disruption_simulation(disruption_type=dtype, n_simulations=100)
            assert result["disruption_type"] == dtype
            assert result["revenue_impact"]["p50"] > 0

    def test_disruption_simulation_resilient(self):
        state = SupplyChainState(
            alternate_supplier_availability=0.90,
            inventory_buffer_days=90,
            visibility_score=0.85,
            supplier_concentration=0.15,
        )
        result = self.model.disruption_simulation(state, duration_days=30, n_simulations=500)
        state_fragile = SupplyChainState(
            alternate_supplier_availability=0.20,
            inventory_buffer_days=7,
            visibility_score=0.25,
            supplier_concentration=0.80,
        )
        result_fragile = self.model.disruption_simulation(state_fragile, duration_days=30, n_simulations=500)
        assert result_fragile["revenue_impact"]["p50"] >= result["revenue_impact"]["p50"]

    def test_tier_mapping_default(self):
        result = self.model.tier_mapping()
        assert "n_tiers" in result
        assert "tiers" in result
        assert "total_suppliers" in result
        assert "visibility_by_tier" in result
        assert "concentration_hhi" in result
        assert result["n_tiers"] == 3

    def test_tier_mapping_custom(self):
        result = self.model.tier_mapping(n_tiers=4, n_suppliers=30)
        assert result["n_tiers"] == 4
        assert "tier_1" in result["tiers"]
        assert "tier_4" in result["tiers"]

    def test_vulnerability_identification(self):
        state = SupplyChainState(
            supplier_concentration=0.70,
            single_source_dependency=0.60,
            logistics_disruption_prob=0.30,
            cyber_exposure=0.50,
            geopolitical_route_risk=0.40,
            inventory_buffer_days=5,
        )
        result = self.model.assess_risk(state)
        vuln_areas = [v["area"] for v in result["vulnerabilities"]]
        assert "supplier_concentration" in vuln_areas
        assert "single_source_dependency" in vuln_areas
        assert "low_inventory_buffer" in vuln_areas

    def test_reproducibility(self):
        result1 = self.model.disruption_simulation(seed=42, n_simulations=100)
        result2 = self.model.disruption_simulation(seed=42, n_simulations=100)
        assert result1["revenue_impact"]["p50"] == result2["revenue_impact"]["p50"]


class TestSupplyChainWebAPI:

    def setup_method(self):
        from oracle.web.app import app
        from fastapi.testclient import TestClient
        self.client = TestClient(app)

    def test_risk_endpoint(self):
        response = self.client.post("/api/supply-chain/risk", json={})
        assert response.status_code == 200
        data = response.json()
        assert "overall_risk" in data
        assert "risk_level" in data
        assert "resilience_score" in data

    def test_risk_endpoint_custom(self):
        response = self.client.post("/api/supply-chain/risk", json={
            "supplier_concentration": 0.80,
            "single_source_dependency": 0.70,
            "alternate_supplier_availability": 0.20,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["overall_risk"] > 0.40

    def test_disruption_endpoint(self):
        response = self.client.post("/api/supply-chain/disruption", json={
            "supplier_concentration": 0.40,
        })
        assert response.status_code == 200
        data = response.json()
        assert "revenue_impact" in data
        assert "avg_recovery_days" in data
        assert "cascade_risk" in data

    def test_tier_mapping_endpoint(self):
        response = self.client.post("/api/supply-chain/tier-mapping", json={
            "supplier_concentration": 0.30,
            "visibility_score": 0.70,
        })
        assert response.status_code == 200
        data = response.json()
        assert "n_tiers" in data
        assert "tiers" in data
        assert "concentration_hhi" in data

    def test_status_includes_supply_chain(self):
        response = self.client.get("/api/status")
        data = response.json()
        endpoints = data["api_endpoints"]
        assert "/api/supply-chain/risk" in endpoints
        assert "/api/supply-chain/disruption" in endpoints
        assert "/api/supply-chain/tier-mapping" in endpoints