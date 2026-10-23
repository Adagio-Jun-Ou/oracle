import pytest
from oracle.domains.cyber import CyberThreatModel, CyberState


class TestCyberThreatModel:

    def setup_method(self):
        self.model = CyberThreatModel()

    def test_assess_threats_defaults(self):
        result = self.model.assess_threats()
        assert "overall_risk_score" in result
        assert "risk_level" in result
        assert "base_threat" in result
        assert "vulnerability_index" in result
        assert "threat_landscape" in result
        assert "top_threats" in result
        assert "attack_surface" in result
        assert "resilience_score" in result
        assert "recommendations" in result
        assert 0 <= result["overall_risk_score"] <= 1
        assert result["risk_level"] in ["minimal", "low", "moderate", "high", "critical"]

    def test_assess_threats_custom_state(self):
        state = CyberState(
            threat_level=0.9,
            ransomware_frequency=0.8,
            nation_state_activity=0.7,
            zero_day_likelihood=0.6,
            ai_threat_capability=0.8,
            incident_response_capacity=0.2,
            regulatory_compliance=0.3,
            workforce_skill_level=0.2,
            patch_velocity=0.1,
        )
        result = self.model.assess_threats(state)
        assert result["overall_risk_score"] > 0.5
        assert result["risk_level"] in ["high", "critical"]
        assert result["vulnerability_index"] > 0.5

    def test_assess_threats_low_risk(self):
        state = CyberState(
            threat_level=0.1,
            ransomware_frequency=0.05,
            nation_state_activity=0.1,
            zero_day_likelihood=0.05,
            ai_threat_capability=0.05,
            incident_response_capacity=0.9,
            regulatory_compliance=0.95,
            workforce_skill_level=0.9,
            patch_velocity=0.95,
        )
        result = self.model.assess_threats(state)
        assert result["overall_risk_score"] < 0.4
        assert result["risk_level"] in ["minimal", "low", "moderate"]

    def test_threat_landscape_keys(self):
        result = self.model.assess_threats()
        expected_keys = [
            "ransomware", "nation_state_apt", "supply_chain_attack",
            "cloud_exploit", "iot_compromise", "social_engineering",
            "ai_enhanced_attack", "insider_threat", "zero_day", "ddos",
        ]
        for key in expected_keys:
            assert key in result["threat_landscape"]
            assert 0 <= result["threat_landscape"][key] <= 1

    def test_top_threats_sorted(self):
        result = self.model.assess_threats()
        threats = result["top_threats"]
        assert len(threats) <= 8
        for t in threats:
            assert "threat" in t
            assert "probability" in t
            assert "severity" in t
            assert 0 <= t["probability"] <= 1
            assert 0 <= t["severity"] <= 1

    def test_recommendations_generated(self):
        state = CyberState(
            incident_response_capacity=0.2,
            workforce_skill_level=0.2,
            patch_velocity=0.1,
        )
        result = self.model.assess_threats(state)
        assert len(result["recommendations"]) >= 2

    def test_simulate_attack_scenario_ransomware(self):
        state = CyberState(threat_level=0.5, incident_response_capacity=0.5)
        result = self.model.simulate_attack_scenario("ransomware", state=state, duration_days=90)
        assert result["attack_vector"] == "ransomware"
        assert 0 <= result["probability"] <= 1
        assert result["expected_containment_days"] > 0
        assert result["expected_recovery_days"] > result["expected_containment_days"]
        assert 0 <= result["data_loss_pct"] <= 1
        assert result["financial_impact_m"] > 0
        assert 0 <= result["reputation_impact"] <= 1
        assert len(result["timeline"]) > 0

    def test_simulate_attack_scenario_all_vectors(self):
        vectors = [
            "ransomware", "phishing", "zero_day", "supply_chain",
            "ddos", "insider_threat", "apt", "iot_exploit",
            "cloud_misconfiguration", "social_engineering",
        ]
        state = CyberState()
        for vector in vectors:
            result = self.model.simulate_attack_scenario(vector, state=state)
            assert result["attack_vector"] == vector
            assert result["probability"] >= 0

    def test_simulate_attack_scenario_low_defense(self):
        state = CyberState(
            threat_level=0.8,
            incident_response_capacity=0.1,
            workforce_skill_level=0.1,
            patch_velocity=0.05,
        )
        result = self.model.simulate_attack_scenario("ransomware", state=state)
        assert result["expected_containment_days"] > 5
        assert result["data_loss_pct"] > 0.1

    def test_simulate_attack_scenario_high_defense(self):
        state = CyberState(
            threat_level=0.3,
            incident_response_capacity=0.9,
            workforce_skill_level=0.95,
            patch_velocity=0.95,
        )
        result = self.model.simulate_attack_scenario("ransomware", state=state)
        assert result["expected_containment_days"] < 20

    def test_predict_defaults(self):
        result = self.model.predict()
        assert "years_ahead" in result
        assert "future_risk_score" in result
        assert "attack_surface_trend" in result
        assert "ai_threat_growth" in result
        assert "defense_improvement" in result
        assert "quantum_threat_horizon" in result
        assert "supply_chain_threat_growth" in result
        assert "regulatory_pressure" in result
        assert result["years_ahead"] == 5
        assert 0 <= result["future_risk_score"] <= 1

    def test_predict_custom_years(self):
        for years in [1, 3, 5, 10, 20]:
            result = self.model.predict(years_ahead=years)
            assert result["years_ahead"] == years
            assert 0 <= result["future_risk_score"] <= 1

    def test_predict_ai_threat_growth(self):
        result = self.model.predict(years_ahead=10)
        assert result["ai_threat_growth"] > 0.3

    def test_predict_quantum_threat_horizon(self):
        result_5 = self.model.predict(years_ahead=5)
        result_15 = self.model.predict(years_ahead=15)
        assert result_15["quantum_threat_horizon"] > result_5["quantum_threat_horizon"]

    def test_set_state(self):
        self.model.set_state(
            threat_level=0.7,
            nation_state_activity=0.8,
            ai_threat_capability=0.6,
        )
        assert self.model.state is not None
        assert self.model.state.threat_level == 0.7
        assert self.model.state.nation_state_activity == 0.8
        assert self.model.state.ai_threat_capability == 0.6

    def test_set_state_returns_self(self):
        result = self.model.set_state(threat_level=0.5)
        assert result is self.model

    def test_assess_with_set_state(self):
        self.model.set_state(
            threat_level=0.8,
            ransomware_frequency=0.7,
            incident_response_capacity=0.9,
        )
        result = self.model.assess_threats()
        assert result["overall_risk_score"] > 0
        assert result["resilience_score"] > 0.5

    def test_recommendations_ai_threat(self):
        state = CyberState(ai_threat_capability=0.5)
        result = self.model.assess_threats(state)
        assert any("AI" in r for r in result["recommendations"])

    def test_recommendations_ransomware(self):
        state = CyberState(ransomware_frequency=0.5)
        result = self.model.assess_threats(state)
        assert any("ransomware" in r.lower() for r in result["recommendations"])

    def test_recommendations_compliance(self):
        state = CyberState(regulatory_compliance=0.3)
        result = self.model.assess_threats(state)
        assert any("framework" in r.lower() or "NIST" in r for r in result["recommendations"])

    def test_assess_resilience_equals_one_minus_vulnerability(self):
        result = self.model.assess_threats()
        assert abs(result["resilience_score"] + result["vulnerability_index"] - 1.0) < 0.01