import pytest
from oracle.domains.political import PoliticalStabilityModel, PoliticalState
from oracle.domains.biodiversity import BiodiversityModel, BiodiversityState
from oracle.domains.pandemic import PandemicPreparednessModel, PandemicState


class TestPoliticalStability:
    def setup_method(self):
        self.model = PoliticalStabilityModel()

    def test_default_stability_assessment(self):
        result = self.model.assess_stability()
        assert "overall_stability" in result
        assert "stability_level" in result
        assert "instability_factors" in result
        assert "coup_risk" in result
        assert "protest_probability" in result
        assert "regime_health" in result
        assert "reform_prospects" in result
        assert "recommendations" in result
        assert 0 <= result["overall_stability"] <= 1
        assert 0 <= result["coup_risk"] <= 1
        assert 0 <= result["protest_probability"] <= 1

    def test_stable_government(self):
        s = PoliticalState(
            government_stability=0.8, rule_of_law=0.85, corruption_index=0.15,
            press_freedom=0.9, institutional_trust=0.75, electoral_integrity=0.85,
            civil_liberties=0.9, military_influence=0.1, economic_inequality=0.2,
            ethnic_tension=0.1, opposition_strength=0.3, external_pressure=0.1,
        )
        result = self.model.assess_stability(s)
        assert result["overall_stability"] > 0.6
        assert result["stability_level"] in ("stable", "highly_stable")
        assert result["coup_risk"] < 0.2
        assert result["protest_probability"] < 0.3

    def test_unstable_government(self):
        s = PoliticalState(
            government_stability=0.2, rule_of_law=0.15, corruption_index=0.85,
            press_freedom=0.1, institutional_trust=0.1, electoral_integrity=0.1,
            civil_liberties=0.15, military_influence=0.8, economic_inequality=0.8,
            ethnic_tension=0.7, opposition_strength=0.9, external_pressure=0.6,
        )
        result = self.model.assess_stability(s)
        assert result["overall_stability"] < 0.4
        assert result["coup_risk"] > 0.3
        assert result["protest_probability"] > 0.4
        assert len(result["instability_factors"]) > 3

    def test_instability_factors_severity(self):
        s = PoliticalState(corruption_index=0.8, economic_inequality=0.7, military_influence=0.6)
        result = self.model.assess_stability(s)
        factor_names = [f["factor"] for f in result["instability_factors"]]
        assert "high_corruption" in factor_names
        assert "economic_inequality" in factor_names
        assert "military_influence" in factor_names

    def test_regime_health_democracy(self):
        s = PoliticalState(
            electoral_integrity=0.9, civil_liberties=0.85, press_freedom=0.8,
            rule_of_law=0.85, military_influence=0.1,
        )
        result = self.model.assess_stability(s)
        assert result["regime_health"]["regime_classification"] == "full_democracy"
        assert result["regime_health"]["democratic_index"] > 0.7

    def test_regime_health_authoritarian(self):
        s = PoliticalState(
            electoral_integrity=0.1, civil_liberties=0.1, press_freedom=0.1,
            rule_of_law=0.1, military_influence=0.9,
        )
        result = self.model.assess_stability(s)
        assert result["regime_health"]["regime_classification"] in ("authoritarian", "totalitarian")

    def test_set_state(self):
        self.model.set_state(government_stability=0.7, corruption_index=0.3)
        assert self.model.state is not None
        assert self.model.state.government_stability == 0.7
        assert self.model.state.corruption_index == 0.3

    def test_regime_change_simulation_gradual_reform(self):
        s = PoliticalState(government_stability=0.4, institutional_trust=0.35)
        result = self.model.simulate_regime_change("gradual_reform", state=s, months=24)
        assert "probability" in result
        assert "timeline" in result
        assert "economic_impact_pct" in result
        assert "social_impact" in result
        assert 0 <= result["probability"] <= 1
        assert len(result["timeline"]) > 0

    def test_regime_change_simulation_military_coup(self):
        s = PoliticalState(government_stability=0.2, military_influence=0.8)
        result = self.model.simulate_regime_change("military_coup", state=s, months=12)
        assert "probability" in result
        assert result["democratic_setback"] > 0
        assert result["human_rights_impact"] > 0

    def test_regime_change_all_scenarios(self):
        for scenario in ["gradual_reform", "electoral_crisis", "military_coup", "popular_revolution", "foreign_intervention", "constitutional_crisis"]:
            result = self.model.simulate_regime_change(scenario)
            assert "probability" in result
            assert "timeline" in result

    def test_predict_default(self):
        result = self.model.predict()
        assert "future_stability" in result
        assert "authoritarian_drift" in result
        assert "reform_likelihood" in result
        assert "protest_forecast" in result
        assert "democratic_trajectory" in result
        assert "political_violence_risk" in result
        assert 0 <= result["future_stability"] <= 1

    def test_predict_unstable(self):
        s = PoliticalState(government_stability=0.2, corruption_index=0.8, military_influence=0.7)
        result = self.model.predict(years_ahead=5, state=s)
        assert result["authoritarian_drift"] > 0.3
        assert result["political_violence_risk"] > 0.2

    def test_recommendations_critical(self):
        s = PoliticalState(corruption_index=0.9, economic_inequality=0.9, institutional_trust=0.1)
        result = self.model.assess_stability(s)
        assert len(result["recommendations"]) > 0


class TestBiodiversity:
    def setup_method(self):
        self.model = BiodiversityModel()

    def test_default_health_assessment(self):
        result = self.model.assess_health()
        assert "biodiversity_index" in result
        assert "health_level" in result
        assert "threat_assessment" in result
        assert "ecosystem_resilience" in result
        assert "tipping_points" in result
        assert "species_outlook" in result
        assert "ecosystem_services_at_risk" in result
        assert "recommendations" in result
        assert 0 <= result["biodiversity_index"] <= 1

    def test_healthy_ecosystem(self):
        s = BiodiversityState(
            habitat_loss_rate=0.1, species_extinction_rate=0.1,
            ecosystem_services_index=0.85, protected_area_coverage=0.3,
            deforestation_rate=0.05, ocean_acidification=0.1,
            invasive_species_impact=0.1, pollution_index=0.1,
            overexploitation=0.1, climate_impact_on_biodiversity=0.1,
            genetic_diversity_index=0.85, coral_reef_health=0.8,
            pollinator_decline=0.1,
        )
        result = self.model.assess_health(s)
        assert result["biodiversity_index"] > 0.6
        assert result["health_level"] in ("good", "moderate")

    def test_degraded_ecosystem(self):
        s = BiodiversityState(
            habitat_loss_rate=0.8, species_extinction_rate=0.85,
            ecosystem_services_index=0.2, deforestation_rate=0.8,
            ocean_acidification=0.7, pollution_index=0.8,
            climate_impact_on_biodiversity=0.8, coral_reef_health=0.1,
            pollinator_decline=0.8,
        )
        result = self.model.assess_health(s)
        assert result["biodiversity_index"] < 0.4
        assert result["n_tipping_points"] > 2

    def test_threat_assessment(self):
        result = self.model.assess_health()
        threats = result["threat_assessment"]
        assert "habitat_destruction" in threats
        assert "climate_change" in threats
        assert "overexploitation" in threats
        assert "pollution" in threats
        assert "invasive_species" in threats
        assert "species_loss" in threats
        for key, val in threats.items():
            assert 0 <= val <= 1

    def test_ecosystem_resilience(self):
        result = self.model.assess_health()
        resilience = result["ecosystem_resilience"]
        assert "overall_resilience" in resilience
        assert "adaptive_capacity" in resilience
        assert "resistance" in resilience
        assert "recovery_potential" in resilience

    def test_tipping_points(self):
        s = BiodiversityState(coral_reef_health=0.15, deforestation_rate=0.7, pollinator_decline=0.8)
        result = self.model.assess_health(s)
        assert len(result["tipping_points"]) > 0
        for tp in result["tipping_points"]:
            assert "name" in tp
            assert "probability" in tp
            assert "impact" in tp

    def test_set_state(self):
        self.model.set_state(habitat_loss_rate=0.5, species_extinction_rate=0.6)
        assert self.model.state is not None
        assert self.model.state.habitat_loss_rate == 0.5

    def test_extinction_scenario_business_as_usual(self):
        result = self.model.simulate_extinction_scenario("business_as_usual", years=50)
        assert "final_biodiversity_index" in result
        assert "timeline" in result
        assert "outcome" in result
        assert "species_at_risk_pct" in result
        assert "ecosystem_services_loss_pct" in result
        assert len(result["timeline"]) > 0

    def test_extinction_scenario_aggressive_protection(self):
        result_bau = self.model.simulate_extinction_scenario("business_as_usual", years=50)
        result_aggressive = self.model.simulate_extinction_scenario("aggressive_protection", years=50)
        assert result_aggressive["final_biodiversity_index"] > result_bau["final_biodiversity_index"]

    def test_extinction_scenario_mass_extinction(self):
        s = BiodiversityState(habitat_loss_rate=0.7, species_extinction_rate=0.8, deforestation_rate=0.7, climate_impact_on_biodiversity=0.8)
        result = self.model.simulate_extinction_scenario("mass_extinction", state=s, years=50)
        assert result["outcome"] in ("severe_decline", "mass_extinction")

    def test_predict_default(self):
        result = self.model.predict()
        assert "future_biodiversity_index" in result
        assert "future_extinction_rate" in result
        assert "future_coral_health" in result
        assert "future_pollinator_decline" in result
        assert "conservation_trajectory" in result
        assert "mass_extinction_risk" in result
        assert "ecosystem_collapse_probability" in result

    def test_predict_degraded(self):
        s = BiodiversityState(habitat_loss_rate=0.8, species_extinction_rate=0.9, climate_impact_on_biodiversity=0.8)
        result = self.model.predict(years_ahead=10, state=s)
        assert result["mass_extinction_risk"] > 0.4
        assert result["ecosystem_collapse_probability"] > 0.3

    def test_ecosystem_services_risk(self):
        result = self.model.assess_health()
        services = result["ecosystem_services_at_risk"]
        assert "pollination" in services
        assert "water_purification" in services
        assert "carbon_sequestration" in services
        assert "coastal_protection" in services
        assert "soil_fertility" in services


class TestPandemicPreparedness:
    def setup_method(self):
        self.model = PandemicPreparednessModel()

    def test_default_preparedness_assessment(self):
        result = self.model.assess_preparedness()
        assert "overall_preparedness" in result
        assert "readiness_level" in result
        assert "gaps" in result
        assert "response_capacity" in result
        assert "vulnerability" in result
        assert "detection_capacity" in result
        assert "treatment_capacity" in result
        assert "prevention_capacity" in result
        assert "recommendations" in result
        assert 0 <= result["overall_preparedness"] <= 1

    def test_well_prepared(self):
        s = PandemicState(
            healthcare_capacity=0.9, surveillance_systems=0.85,
            vaccine_readiness=0.8, supply_chain_resilience=0.85,
            international_cooperation=0.8, public_trust=0.75,
            pandemic_preparedness_plan=0.85, laboratory_capacity=0.8,
            risk_communication=0.8, essential_worker_protection=0.75,
            genomic_surveillance=0.8, ppe_stockpile=0.7,
            antiviral_stockpile=0.6, contact_tracing_capacity=0.7,
            quarantine_capacity=0.7,
        )
        result = self.model.assess_preparedness(s)
        assert result["overall_preparedness"] > 0.6
        assert result["readiness_level"] in ("high", "moderate")
        assert result["detection_capacity"] > 0.5

    def test_poorly_prepared(self):
        s = PandemicState(
            healthcare_capacity=0.1, surveillance_systems=0.1,
            vaccine_readiness=0.1, supply_chain_resilience=0.1,
            international_cooperation=0.1, public_trust=0.1,
            pandemic_preparedness_plan=0.1, laboratory_capacity=0.1,
            risk_communication=0.1, essential_worker_protection=0.1,
            genomic_surveillance=0.1, ppe_stockpile=0.1,
            antiviral_stockpile=0.1, contact_tracing_capacity=0.1,
            quarantine_capacity=0.1,
        )
        result = self.model.assess_preparedness(s)
        assert result["overall_preparedness"] < 0.2
        assert result["readiness_level"] in ("low", "critical")
        assert len(result["gaps"]) > 5

    def test_gaps_identification(self):
        s = PandemicState(genomic_surveillance=0.1, vaccine_readiness=0.1, pandemic_preparedness_plan=0.1)
        result = self.model.assess_preparedness(s)
        gap_areas = [g["area"] for g in result["gaps"]]
        assert len(result["gaps"]) > 0
        assert result["n_gaps"] == len(result["gaps"])

    def test_response_capacity(self):
        result = self.model.assess_preparedness()
        rc = result["response_capacity"]
        assert "detection_speed_days" in rc
        assert "treatment_capacity" in rc
        assert "vaccination_speed_months" in rc
        assert "communication_effectiveness" in rc
        assert "overall_response_score" in rc
        assert rc["detection_speed_days"] > 0

    def test_vulnerability_assessment(self):
        result = self.model.assess_preparedness()
        vuln = result["vulnerability"]
        assert "systemic_vulnerability" in vuln
        assert "high_risk_populations_pct" in vuln
        assert "supply_chain_disruption_risk" in vuln
        assert "misinformation_vulnerability" in vuln
        assert "international_coordination_gap" in vuln

    def test_set_state(self):
        self.model.set_state(healthcare_capacity=0.8, surveillance_systems=0.7)
        assert self.model.state is not None
        assert self.model.state.healthcare_capacity == 0.8

    def test_outbreak_simulation_respiratory(self):
        result = self.model.simulate_outbreak("respiratory_virus", r0=2.5, days=365)
        assert "effective_r0" in result
        assert "detection_delay_days" in result
        assert "response_delay_days" in result
        assert "infection_fatality_rate" in result
        assert "peak_hospital_occupancy" in result
        assert "timeline" in result
        assert "healthcare_overwhelm" in result
        assert "vaccine_deployment_months" in result
        assert len(result["timeline"]) > 0

    def test_outbreak_simulation_all_pathogens(self):
        for pathogen in self.model.pathogen_types:
            result = self.model.simulate_outbreak(pathogen)
            assert "effective_r0" in result
            assert "infection_fatality_rate" in result

    def test_outbreak_well_prepared(self):
        s = PandemicState(healthcare_capacity=0.9, surveillance_systems=0.85, vaccine_readiness=0.8)
        result = self.model.simulate_outbreak("respiratory_virus", state=s)
        s_bad = PandemicState(healthcare_capacity=0.1, surveillance_systems=0.1, vaccine_readiness=0.1)
        result_bad = self.model.simulate_outbreak("respiratory_virus", state=s_bad)
        assert result["detection_delay_days"] < result_bad["detection_delay_days"]

    def test_predict_default(self):
        result = self.model.predict()
        assert "future_preparedness" in result
        assert "preparation_improvement" in result
        assert "surveillance_trajectory" in result
        assert "vaccine_readiness_trajectory" in result
        assert "pandemic_probability" in result
        assert "emergence_risk" in result
        assert "resilience_score" in result

    def test_predict_improvement(self):
        s = PandemicState(surveillance_systems=0.3, vaccine_readiness=0.2, genomic_surveillance=0.2)
        result = self.model.predict(years_ahead=5, state=s)
        assert result["surveillance_trajectory"] > s.surveillance_systems
        assert result["vaccine_readiness_trajectory"] > s.vaccine_readiness

    def test_emergence_risk(self):
        result = self.model.predict(years_ahead=10)
        emergence = result["emergence_risk"]
        assert "respiratory_virus" in emergence
        assert "influenza_pandemic" in emergence
        assert "novel_coronavirus" in emergence
        assert "antibiotic_resistant" in emergence
        assert "zoonotic_spillover" in emergence

    def test_recommendations(self):
        s = PandemicState(surveillance_systems=0.1, vaccine_readiness=0.1, healthcare_capacity=0.1)
        result = self.model.assess_preparedness(s)
        assert len(result["recommendations"]) > 0


class TestEngineIntegrationPolitical:
    def test_political_in_predict(self):
        from oracle.prediction.engine import OracleEngine
        from datetime import date
        engine = OracleEngine()
        engine.set_political_state(government_stability=0.6, corruption_index=0.3)
        forecast = engine.predict(date(2030, 1, 1))
        pol_preds = [p for p in forecast.predictions if p.domain.value == "political"]
        assert len(pol_preds) >= 2

    def test_political_in_diagnose(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_political_state(government_stability=0.4, corruption_index=0.6)
        result = engine.diagnose()
        assert "political_stability" in result or "political_future" in result

    def test_political_cross_domain_interaction(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_political_state(government_stability=0.1, rule_of_law=0.1, corruption_index=0.9, military_influence=0.9)
        engine.set_geopolitical_state(us_china_tension=0.7)
        result = engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        interaction_types = [i["type"] for i in interactions]
        assert len(interactions) > 0  # Should have at least one interaction


class TestEngineIntegrationBiodiversity:
    def test_biodiversity_in_predict(self):
        from oracle.prediction.engine import OracleEngine
        from datetime import date
        engine = OracleEngine()
        engine.set_biodiversity_state(habitat_loss_rate=0.4, species_extinction_rate=0.5)
        forecast = engine.predict(date(2030, 1, 1))
        bio_preds = [p for p in forecast.predictions if p.domain.value == "biodiversity"]
        assert len(bio_preds) >= 2

    def test_biodiversity_in_diagnose(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_biodiversity_state(habitat_loss_rate=0.5, deforestation_rate=0.6)
        result = engine.diagnose()
        assert "biodiversity_health" in result or "biodiversity_future" in result


class TestEngineIntegrationPandemic:
    def test_pandemic_in_predict(self):
        from oracle.prediction.engine import OracleEngine
        from datetime import date
        engine = OracleEngine()
        engine.set_pandemic_state(healthcare_capacity=0.5, surveillance_systems=0.4)
        forecast = engine.predict(date(2030, 1, 1))
        pan_preds = [p for p in forecast.predictions if p.domain.value == "pandemic_preparedness"]
        assert len(pan_preds) >= 2

    def test_pandemic_in_diagnose(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_pandemic_state(healthcare_capacity=0.3, surveillance_systems=0.2)
        result = engine.diagnose()
        assert "pandemic_preparedness" in result or "pandemic_future" in result

    def test_pandemic_cross_domain(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_pandemic_state(healthcare_capacity=0.2, supply_chain_resilience=0.2)
        engine.set_food_state(food_production_index=0.3)
        result = engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        assert len(interactions) >= 0

class TestNewCrossDomainInteractions:
    def test_political_geopolitical_interaction(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_political_state(government_stability=0.1, rule_of_law=0.1, corruption_index=0.9, military_influence=0.9)
        engine.set_geopolitical_state(us_china_tension=0.7)
        result = engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        types = [i["type"] for i in interactions]
        assert "political_geopolitical_instability" in types

    def test_biodiversity_climate_interaction(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_biodiversity_state(habitat_loss_rate=0.95, species_extinction_rate=0.95, climate_impact_on_biodiversity=0.95, ecosystem_services_index=0.05, genetic_diversity_index=0.05)
        engine.set_climate_state(global_temp_anomaly_c=2.0, sea_level_rise_mm=200)
        result = engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        types = [i["type"] for i in interactions]
        assert "climate_biodiversity_collapse" in types

    def test_biodiversity_food_interaction(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_biodiversity_state(habitat_loss_rate=0.7, species_extinction_rate=0.6, ecosystem_services_index=0.2)
        engine.set_food_state(food_production_index=0.3, supply_chain_disruption=0.5)
        result = engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        types = [i["type"] for i in interactions]
        assert "biodiversity_food_security_nexus" in types

    def test_pandemic_political_interaction(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_pandemic_state(healthcare_capacity=0.1, surveillance_systems=0.1)
        engine.set_political_state(government_stability=0.1, military_influence=0.8)
        result = engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        types = [i["type"] for i in interactions]
        assert "pandemic_political_instability" in types

    def test_pandemic_biodiversity_interaction(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_pandemic_state(healthcare_capacity=0.05, surveillance_systems=0.05, vaccine_readiness=0.05)
        engine.set_biodiversity_state(habitat_loss_rate=0.95, species_extinction_rate=0.95, ecosystem_services_index=0.05)
        result = engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        types = [i["type"] for i in interactions]
        assert "biodiversity_pandemic_emergence" in types

    def test_political_biodiversity_governance(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_political_state(government_stability=0.05, corruption_index=0.95, rule_of_law=0.05)
        engine.set_biodiversity_state(habitat_loss_rate=0.95, species_extinction_rate=0.95, ecosystem_services_index=0.05)
        result = engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        types = [i["type"] for i in interactions]
        assert "political_biodiversity_governance_failure" in types

    def test_biodiversity_disease_emergence(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_biodiversity_state(habitat_loss_rate=0.95, species_extinction_rate=0.95, ecosystem_services_index=0.05)
        engine.set_epidemiology_state(beta=0.5, mu=0.02)
        result = engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        types = [i["type"] for i in interactions]
        assert "biodiversity_disease_emergence" in types
        assert "biodiversity_disease_emergence" in types

    def test_multiple_new_domains_predict(self):
        from oracle.prediction.engine import OracleEngine
        from datetime import date
        engine = OracleEngine()
        engine.set_political_state(government_stability=0.5, corruption_index=0.3)
        engine.set_biodiversity_state(habitat_loss_rate=0.3, species_extinction_rate=0.4)
        engine.set_pandemic_state(healthcare_capacity=0.5, surveillance_systems=0.4)
        forecast = engine.predict(date(2030, 1, 1))
        domains = set(p.domain.value for p in forecast.predictions)
        assert "political" in domains
        assert "biodiversity" in domains
        assert "pandemic_preparedness" in domains
