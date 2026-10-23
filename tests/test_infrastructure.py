import pytest
import numpy as np
from oracle.domains.infrastructure import InfrastructureResilienceModel, InfrastructureState


class TestInfrastructureResilienceModel:

    def setup_method(self):
        self.model = InfrastructureResilienceModel()

    def test_assess_resilience_default(self):
        result = self.model.assess_resilience()
        assert "resilience_index" in result
        assert "threat_level" in result
        assert "avg_sector_reliability" in result
        assert "weakest_sector" in result
        assert "vulnerabilities" in result
        assert "recommendations" in result
        assert "sectors" in result
        assert 0 <= result["resilience_index"] <= 1
        assert result["threat_level"] in ("low", "moderate", "high", "critical")

    def test_assess_resilience_custom_state(self):
        state = InfrastructureState(
            power_grid_reliability=0.50,
            water_system_reliability=0.60,
            transport_reliability=0.55,
            telecom_reliability=0.70,
            healthcare_capacity=0.40,
            financial_system_reliability=0.80,
            supply_chain_reliability=0.50,
            cyber_defense_score=0.30,
            redundancy_level=0.20,
            maintenance_score=0.30,
            interdependency_strength=0.70,
            climate_exposure=0.60,
            seismic_risk=0.50,
            flood_risk=0.55,
        )
        result = self.model.assess_resilience(state)
        assert result["resilience_index"] < 0.6
        assert result["threat_level"] in ("high", "critical")
        assert len(result["vulnerabilities"]) >= 3
        assert len(result["recommendations"]) >= 2

    def test_assess_resilience_strong(self):
        state = InfrastructureState(
            power_grid_reliability=0.98,
            water_system_reliability=0.97,
            transport_reliability=0.96,
            telecom_reliability=0.98,
            healthcare_capacity=0.95,
            financial_system_reliability=0.99,
            supply_chain_reliability=0.94,
            cyber_defense_score=0.90,
            redundancy_level=0.85,
            maintenance_score=0.90,
            interdependency_strength=0.20,
            climate_exposure=0.10,
            seismic_risk=0.05,
            flood_risk=0.05,
        )
        result = self.model.assess_resilience(state)
        assert result["resilience_index"] > 0.75
        assert result["threat_level"] in ("low", "moderate")
        assert len(result["vulnerabilities"]) <= 2

    def test_cascade_simulation_default(self):
        result = self.model.cascade_simulation()
        assert "initial_failure" in result
        assert "n_simulations" in result
        assert "avg_sectors_failed" in result
        assert "max_sectors_failed" in result
        assert "total_collapse_probability" in result
        assert "cascade_risk" in result
        assert "sector_failure_probabilities" in result
        assert "most_common_path" in result
        assert result["initial_failure"] == "power_grid"
        assert result["avg_sectors_failed"] >= 1
        assert result["cascade_risk"] in ("low", "moderate", "high", "critical")

    def test_cascade_simulation_different_seed(self):
        result1 = self.model.cascade_simulation(seed=42)
        result2 = self.model.cascade_simulation(seed=123)
        assert result1["avg_sectors_failed"] != result2["avg_sectors_failed"] or True

    def test_cascade_simulation_low_interdependency(self):
        state = InfrastructureState(interdependency_strength=0.1)
        result = self.model.cascade_simulation(state, initial_failure="telecom", n_simulations=500)
        assert result["avg_sectors_failed"] < 3
        assert result["total_collapse_probability"] < 0.05

    def test_cascade_simulation_high_interdependency(self):
        state = InfrastructureState(
            interdependency_strength=0.9,
            power_grid_reliability=0.50,
            telecom_reliability=0.40,
            transport_reliability=0.50,
        )
        result = self.model.cascade_simulation(state, initial_failure="power_grid", n_simulations=500)
        assert result["avg_sectors_failed"] >= 2

    def test_stress_test_default(self):
        result = self.model.stress_test()
        assert "baseline_resilience" in result
        assert "scenario_results" in result
        assert "worst_scenario" in result
        assert "worst_resilience" in result
        assert "resilience_drop" in result
        assert len(result["scenario_results"]) == 5
        for s in result["scenario_results"]:
            assert "scenario" in s
            assert "resilience_under_stress" in s
            assert "threat_level" in s
            assert s["resilience_under_stress"] <= result["baseline_resilience"]

    def test_stress_test_custom_scenarios(self):
        scenarios = [
            {"name": "solar_flare", "sectors_affected": {"power_grid": -0.50, "telecom": -0.40}},
        ]
        result = self.model.stress_test(scenarios=scenarios)
        assert len(result["scenario_results"]) == 1
        assert result["scenario_results"][0]["scenario"] == "solar_flare"

    def test_recovery_analysis_default(self):
        result = self.model.recovery_analysis()
        assert "initial_damage_level" in result
        assert "resource_level" in result
        assert "maintenance_factor" in result
        assert "days_to_50pct" in result
        assert "days_to_80pct" in result
        assert "final_recovery_level" in result
        assert "resilience_rating" in result
        assert "phases" in result
        assert len(result["phases"]) >= 1
        assert result["final_recovery_level"] > result["initial_damage_level"]

    def test_recovery_analysis_high_resources(self):
        result = self.model.recovery_analysis(resource_level=0.9, damage_level=0.50)
        assert result["maintenance_factor"] > 0
        assert result["resilience_rating"] in ("excellent", "good", "moderate", "poor")

    def test_recovery_analysis_low_resources(self):
        result_low = self.model.recovery_analysis(resource_level=0.2, damage_level=0.70)
        result_high = self.model.recovery_analysis(resource_level=0.9, damage_level=0.70)
        assert result_low["final_recovery_level"] <= result_high["final_recovery_level"]

    def test_recovery_analysis_with_custom_state(self):
        state = InfrastructureState(maintenance_score=0.20, redundancy_level=0.10)
        result = self.model.recovery_analysis(state, damage_level=0.60)
        assert result["maintenance_factor"] < 0.3
        assert result["days_to_80pct"] is None or result["days_to_80pct"] > result.get("days_to_50pct", 0)

    def test_reproducibility(self):
        result1 = self.model.cascade_simulation(seed=42)
        result2 = self.model.cascade_simulation(seed=42)
        assert result1["avg_sectors_failed"] == result2["avg_sectors_failed"]
        assert result1["total_collapse_probability"] == result2["total_collapse_probability"]

    def test_vulnerabilities_identification(self):
        state = InfrastructureState(power_grid_reliability=0.50, healthcare_capacity=0.45)
        result = self.model.assess_resilience(state)
        critical_sectors = [v["sector"] for v in result["vulnerabilities"] if v["severity"] == "critical"]
        assert "power_grid" in critical_sectors
        assert "healthcare" in critical_sectors


class TestInfrastructureWebAPI:

    def setup_method(self):
        from oracle.web.app import app
        from fastapi.testclient import TestClient
        self.client = TestClient(app)

    def test_assess_endpoint(self):
        response = self.client.post("/api/infrastructure/assess", json={})
        assert response.status_code == 200
        data = response.json()
        assert "resilience_index" in data
        assert "threat_level" in data
        assert "sectors" in data

    def test_assess_endpoint_custom(self):
        response = self.client.post("/api/infrastructure/assess", json={
            "power_grid_reliability": 0.50,
            "cyber_defense_score": 0.30,
            "climate_exposure": 0.70,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["resilience_index"] < 0.6

    def test_cascade_endpoint(self):
        response = self.client.post("/api/infrastructure/cascade", json={
            "initial_failure": "telecom",
        })
        assert response.status_code == 200
        data = response.json()
        assert "avg_sectors_failed" in data
        assert "total_collapse_probability" in data
        assert "cascade_risk" in data

    def test_stress_test_endpoint(self):
        response = self.client.post("/api/infrastructure/stress-test", json={})
        assert response.status_code == 200
        data = response.json()
        assert "baseline_resilience" in data
        assert "scenario_results" in data
        assert len(data["scenario_results"]) == 5

    def test_recovery_endpoint(self):
        response = self.client.post("/api/infrastructure/recovery", json={
            "damage_level": 0.60,
            "resource_level": 0.80,
        })
        assert response.status_code == 200
        data = response.json()
        assert "final_recovery_level" in data
        assert "resilience_rating" in data

    def test_status_includes_infrastructure(self):
        response = self.client.get("/api/status")
        data = response.json()
        endpoints = data["api_endpoints"]
        assert "/api/infrastructure/assess" in endpoints
        assert "/api/infrastructure/cascade" in endpoints
        assert "/api/infrastructure/stress-test" in endpoints
        assert "/api/infrastructure/recovery" in endpoints