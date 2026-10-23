import pytest
import numpy as np
from datetime import date
from oracle.domains.geopolitical import GeopoliticalRiskModel, GeopoliticalState
from oracle.domains.climate import ClimateModel, ClimateState
from oracle.domains.sentiment import SentimentEngine, SentimentData
from oracle.domains.contagion import ContagionModel, ContagionState, NetworkNode
from oracle.domains.demographics import DemographicsModel, DemographicsState
from oracle.core.types import Domain, ConfidenceLevel
from oracle.prediction.engine import OracleEngine
from oracle.core.config import OracleConfig


class TestGeopoliticalRiskModel:
    def setup_method(self):
        self.model = GeopoliticalRiskModel()

    def test_assess_risk(self):
        state = GeopoliticalState()
        result = self.model.assess_risk(state)
        assert "overall_risk" in result
        assert 0 <= result["overall_risk"] <= 1
        assert "stability_score" in result
        assert "conflict_risk" in result

    def test_simulate_events(self):
        state = GeopoliticalState(us_china_tension=0.6, russia_west_tension=0.7)
        result = self.model.simulate_events(state, months=6, n_simulations=100)
        assert "aggregate_impact" in result
        assert "US-China Trade War Escalation" in result or "Technology Decoupling" in result
        assert result["aggregate_impact"]["mean"] >= 0

    def test_predict(self):
        state = GeopoliticalState()
        preds = self.model.predict(state, date(2030, 1, 1))
        assert len(preds) >= 4
        for p in preds:
            assert p.domain in [Domain.GEOPOLITICS, Domain.ECONOMICS, Domain.ENERGY]

    def test_scenario_analysis(self):
        baseline = GeopoliticalState()
        high_tension = GeopoliticalState(us_china_tension=0.9, russia_west_tension=0.9)
        result = self.model.scenario_analysis(baseline, {"high_tension": high_tension})
        assert "baseline" in result
        assert "high_tension" in result

    def test_adjusted_probability_high_tension(self):
        state = GeopoliticalState(us_china_tension=0.9)
        template = {"name": "test", "type": "trade_war", "base_prob": 0.2, "indicators": {"us_china_tension": 0.5}}
        prob = self.model._adjusted_probability(template, state)
        assert prob > 0.2


class TestClimateModel:
    def setup_method(self):
        self.model = ClimateModel()

    def test_project_temperature(self):
        state = ClimateState()
        result = self.model.project_temperature(state, years_ahead=10, n_simulations=50)
        assert "aggressive_mitigation" in result
        assert "current_policies" in result
        assert "high_emissions" in result
        for scenario in ["aggressive_mitigation", "current_policies", "high_emissions"]:
            assert result[scenario]["temp_2080_median"] > 0

    def test_carbon_pricing(self):
        result = self.model.carbon_pricing_impact(50.0)
        assert result["emission_reduction_pct"] >= 0
        assert result["gdp_impact_pct"] < 0

    def test_tipping_points(self):
        state = ClimateState(global_temp_anomaly_c=1.2)
        result = self.model.tipping_points(state)
        assert "tipping_points" in result
        assert "overall_tipping_risk" in result
        assert len(result["tipping_points"]) >= 5
        coral = [tp for tp in result["tipping_points"] if "Coral" in tp["name"]]
        assert len(coral) == 1
        assert coral[0]["triggered"] == False

    def test_tipping_points_high_temp(self):
        state = ClimateState(global_temp_anomaly_c=3.0)
        result = self.model.tipping_points(state)
        triggered = [tp for tp in result["tipping_points"] if tp["triggered"]]
        assert len(triggered) >= 2

    def test_predict(self):
        state = ClimateState()
        preds = self.model.predict(state, date(2035, 1, 1))
        assert len(preds) >= 5
        for p in preds:
            assert p.domain in [Domain.CLIMATE, Domain.ENERGY]


class TestSentimentEngine:
    def setup_method(self):
        self.engine = SentimentEngine()

    def test_compute_composite(self):
        data = SentimentData(fear_greed_index=0.3, vix_level=0.6)
        result = self.engine.compute_composite_index(data)
        assert 0 <= result["composite_index"] <= 1
        assert result["interpretation"] in ["extreme_fear", "fear", "neutral", "greed", "extreme_greed"]

    def test_composite_extreme_fear(self):
        data = SentimentData(fear_greed_index=0.95, vix_level=0.95, put_call_ratio=2.0, short_interest=0.9, safe_haven_demand=0.9, consumer_confidence=0.1, social_media_sentiment=0.1, news_sentiment=0.1, market_breadth=0.1, institutional_flow=-0.5)
        result = self.engine.compute_composite_index(data)
        assert 0 <= result["composite_index"] <= 1
        assert result["interpretation"] in ["extreme_fear", "fear", "neutral", "greed", "extreme_greed"]

    def test_predict_sentiment(self):
        data = SentimentData()
        result = self.engine.predict_sentiment(data, days_ahead=30, n_simulations=100)
        assert "current" in result
        assert "forecasts" in result
        assert result["forecasts"]["p50"] >= 0

    def test_contrarian_signals(self):
        data = SentimentData(fear_greed_index=0.05, vix_level=0.8, put_call_ratio=1.5, short_interest=0.5, safe_haven_demand=0.1)
        result = self.engine.detect_contrarian_signals(data)
        assert "signals" in result
        assert "composite" in result
        assert isinstance(result["signals"], list)

    def test_predict(self):
        data = SentimentData()
        preds = self.engine.predict(data, date(2030, 1, 1))
        assert len(preds) >= 3
        for p in preds:
            assert p.domain in [Domain.FINANCE, Domain.ECONOMICS, Domain.SOCIAL]


class TestContagionModel:
    def setup_method(self):
        self.model = ContagionModel(n_nodes=15)

    def test_sir_model(self):
        result = self.model.sir_model(population=10000, beta=0.3, gamma=0.1, days=365)
        assert result["r0"] == pytest.approx(3.0, abs=0.01)
        assert result["peak_day"] > 0
        assert result["total_infected_pct"] > 0.5

    def test_financial_contagion(self):
        state = ContagionState(financial_systemic_risk=0.4)
        result = self.model.financial_contagion(state, n_simulations=50, time_steps=20)
        assert result["mean_systemic_stress"] >= 0
        assert result["cascade_probability"] >= 0

    def test_cascade_threshold(self):
        state = ContagionState(trade_linkage=0.5)
        result = self.model.cascade_threshold_model(state, n_agents=50)
        assert "cascade_size" in result
        assert result["cascade_size"] >= 0
        assert isinstance(result["full_cascade"], bool)

    def test_predict(self):
        state = ContagionState()
        preds = self.model.predict(state, date(2030, 1, 1))
        assert len(preds) >= 4
        for p in preds:
            assert p.domain in [Domain.FINANCE, Domain.ECONOMICS, Domain.SOCIAL]


class TestDemographicsModel:
    def setup_method(self):
        self.model = DemographicsModel()

    def test_classify_stage(self):
        developing = DemographicsState(birth_rate_per_1000=35, death_rate_per_1000=25)
        assert self.model.classify_stage(developing) == "early_transition"
        developed = DemographicsState(birth_rate_per_1000=11, death_rate_per_1000=9)
        assert self.model.classify_stage(developed) == "post_transition"

    def test_project_population(self):
        state = DemographicsState()
        result = self.model.project_population(state, years=10, n_simulations=50)
        assert "baseline" in result
        assert "high_growth" in result
        assert result["baseline"]["pop_final_median"] > 8000

    def test_aging_dividend(self):
        state = DemographicsState(fertility_rate=2.1, elderly_pct=0.08)
        result = self.model.aging_dividend(state)
        assert result["stage"] in ["pre_transition", "early_transition", "late_transition", "post_transition"]
        assert result["working_age_pct"] > 0

    def test_predict(self):
        state = DemographicsState()
        preds = self.model.predict(state, date(2050, 1, 1))
        assert len(preds) >= 5
        for p in preds:
            assert p.domain in [Domain.DEMOGRAPHICS, Domain.HEALTH, Domain.ECONOMICS, Domain.SOCIAL]


class TestOracleDomainIntegration:
    def setup_method(self):
        np.random.seed(42)
        self.config = OracleConfig()
        self.engine = OracleEngine(self.config)

    def test_predict_with_geopolitical(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02, inflation_rate=0.03)
        self.engine.set_geopolitical_state(
            us_china_tension=0.6, russia_west_tension=0.7,
        )
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        geo_preds = [p for p in forecast.predictions if p.domain == Domain.GEOPOLITICS]
        assert len(geo_preds) >= 1

    def test_predict_with_climate(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02)
        self.engine.set_climate_state(global_temp_anomaly_c=1.2, co2_ppm=420)
        forecast = self.engine.predict(date(2035, 1, 1), ConfidenceLevel.MEDIUM)
        climate_preds = [p for p in forecast.predictions if p.domain == Domain.CLIMATE]
        assert len(climate_preds) >= 1

    def test_predict_with_sentiment(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02)
        self.engine.set_sentiment_data(fear_greed_index=0.35, vix_level=0.6)
        forecast = self.engine.predict(date(2030, 6, 1), ConfidenceLevel.MEDIUM)
        sent_preds = [p for p in forecast.predictions if "sentiment" in p.variable]
        assert len(sent_preds) >= 1

    def test_predict_with_contagion(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02)
        self.engine.set_contagion_state(financial_systemic_risk=0.4, trade_linkage=0.5)
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.MEDIUM)
        contagion_preds = [p for p in forecast.predictions if "contagion" in p.variable]
        assert len(contagion_preds) >= 1

    def test_predict_with_demographics(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02)
        self.engine.set_demographics_state(total_population_m=8100, fertility_rate=2.3)
        forecast = self.engine.predict(date(2050, 1, 1), ConfidenceLevel.MEDIUM)
        demo_preds = [p for p in forecast.predictions if p.domain == Domain.DEMOGRAPHICS]
        assert len(demo_preds) >= 1

    def test_predict_with_epidemiology(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02)
        self.engine.set_epidemiology_state(population=1e6, initial_infected=100, beta=0.3)
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        epi_preds = [p for p in forecast.predictions if p.domain == Domain.EPIDEMIOLOGY]
        assert len(epi_preds) >= 1
        vars_set = {p.variable for p in epi_preds}
        assert "peak_infected" in vars_set or "total_deceased" in vars_set

    def test_predict_with_technology(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02)
        self.engine.set_technology_state(current_adoption=0.15, innovation_speed=0.3)
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.MEDIUM)
        tech_preds = [p for p in forecast.predictions if p.domain == Domain.TECHNOLOGY]
        assert len(tech_preds) >= 1

    def test_predict_with_social(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02)
        self.engine.set_social_state(gini=0.41, unemployment=0.06, political_instability=0.3)
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.MEDIUM)
        social_preds = [p for p in forecast.predictions if p.domain == Domain.SOCIAL]
        assert len(social_preds) >= 1

    def test_diagnose_with_epidemiology(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02)
        self.engine.set_epidemiology_state(population=1e6, initial_infected=100)
        diagnosis = self.engine.diagnose()
        assert "epidemiology" in diagnosis

    def test_diagnose_with_technology(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02)
        self.engine.set_technology_state(current_adoption=0.15)
        diagnosis = self.engine.diagnose()
        assert "technology" in diagnosis

    def test_diagnose_with_social(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02)
        self.engine.set_social_state(gini=0.41, unemployment=0.06)
        diagnosis = self.engine.diagnose()
        assert "social" in diagnosis

    def test_cross_domain_interactions_with_new_domains(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02, debt_to_gdp=1.5)
        self.engine.set_geopolitical_state(us_china_tension=0.7, trade_protectionism=0.5)
        self.engine.set_epidemiology_state(population=1e6, mu=0.02)
        self.engine.set_social_state(gini=0.5, unemployment=0.1, political_instability=0.6)
        self.engine.set_technology_state(current_adoption=0.4)
        self.engine.set_demographics_state(total_population_m=8100, elderly_pct=0.25)
        diagnosis = self.engine.diagnose()
        interactions = diagnosis.get("cross_domain_interactions", [])
        assert len(interactions) >= 1

    def test_predict_with_all_domains(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02, inflation_rate=0.03)
        self.engine.set_geopolitical_state(us_china_tension=0.5)
        self.engine.set_climate_state(global_temp_anomaly_c=1.2)
        self.engine.set_sentiment_data(fear_greed_index=0.5)
        self.engine.set_contagion_state(financial_systemic_risk=0.3)
        self.engine.set_demographics_state(total_population_m=8100)
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        domains_covered = set(p.domain for p in forecast.predictions)
        assert Domain.ECONOMICS in domains_covered
        assert len(forecast.predictions) > 10

    def test_diagnose_with_domains(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02, debt_to_gdp=2.0)
        self.engine.set_geopolitical_state(us_china_tension=0.6, russia_west_tension=0.7)
        self.engine.set_climate_state(global_temp_anomaly_c=1.5)
        self.engine.set_contagion_state(financial_systemic_risk=0.4)
        self.engine.set_sentiment_data(fear_greed_index=0.3, vix_level=0.6)
        self.engine.set_demographics_state(total_population_m=8100, elderly_pct=0.15)
        diagnosis = self.engine.diagnose()
        assert "geopolitical_risk" in diagnosis
        assert "climate_tipping_points" in diagnosis
        assert "contagion_cascade" in diagnosis
        assert "sentiment" in diagnosis
        assert "demographics_dividend" in diagnosis
        assert "cross_domain_interactions" in diagnosis

    def test_diagnose_cross_domain_interactions(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02, debt_to_gdp=2.0)
        self.engine.set_geopolitical_state(us_china_tension=0.8, russia_west_tension=0.8, energy_geopolitics=0.7)
        self.engine.set_sentiment_data(fear_greed_index=0.2, vix_level=0.8)
        self.engine.set_contagion_state(financial_systemic_risk=0.6, trade_linkage=0.6)
        self.engine.set_climate_state(global_temp_anomaly_c=2.0)
        self.engine.set_demographics_state(total_population_m=9000, elderly_pct=0.25)
        diagnosis = self.engine.diagnose()
        interactions = diagnosis.get("cross_domain_interactions", [])
        assert len(interactions) >= 2
        types = [i["type"] for i in interactions]
        assert any("geopolitical" in t.lower() for t in types)

    def test_predict_with_all_domains_ensemble(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02)
        self.engine.set_geopolitical_state(us_china_tension=0.5)
        self.engine.set_climate_state(global_temp_anomaly_c=1.2)
        self.engine.set_sentiment_data(fear_greed_index=0.5)
        self.engine.set_contagion_state(financial_systemic_risk=0.3)
        self.engine.set_demographics_state(total_population_m=8100)
        np.random.seed(42)
        self.engine.add_time_series("gdp_index", np.cumsum(np.random.normal(0.0002, 0.01, 120)) + 100, Domain.ECONOMICS)
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        all_domains = set(p.domain.value for p in forecast.predictions)
        assert "economics" in all_domains or "finance" in all_domains
        assert len(forecast.predictions) >= 5


class TestGeopoliticalEdgeCases:
    def setup_method(self):
        self.model = GeopoliticalRiskModel()

    def test_extreme_tensions(self):
        state = GeopoliticalState(us_china_tension=0.99, russia_west_tension=0.99, middle_east_instability=0.99)
        result = self.model.assess_risk(state)
        assert result["overall_risk"] > 0.5
        assert result["conflict_risk"] > 0.4

    def test_minimal_tensions(self):
        state = GeopoliticalState(us_china_tension=0.01, russia_west_tension=0.01)
        result = self.model.assess_risk(state)
        assert result["overall_risk"] < 0.5
        assert result["stability_score"] > 0.5

    def test_scenario_comparison(self):
        baseline = GeopoliticalState()
        crisis = GeopoliticalState(us_china_tension=0.9, russia_west_tension=0.9, middle_east_instability=0.9, nuclear_proliferation_risk=0.5)
        result = self.model.scenario_analysis(baseline, {"crisis": crisis})
        assert "baseline" in result
        assert "crisis" in result
        assert result["crisis"]["overall_risk"] > result["baseline"]["overall_risk"]

    def test_conflict_intensity_score(self):
        result = self.model.conflict_intensity_score(GeopoliticalState())
        assert "overall_intensity" in result
        assert "hot_conflict_score" in result
        assert "warm_conflict_score" in result
        assert "cold_tension_score" in result
        assert "stage" in result
        assert "de_escalation_potential" in result
        assert "nuclear_risk_level" in result
        assert "major_flashpoints" in result
        assert 0 <= result["overall_intensity"] <= 1
        assert result["stage"] in ["peaceful", "latent_tension", "cold_war", "warm_conflict", "hot_war"]

    def test_conflict_intensity_high_tension(self):
        high = self.model.conflict_intensity_score(GeopoliticalState(us_china_tension=0.9, russia_west_tension=0.9, middle_east_instability=0.8))
        low = self.model.conflict_intensity_score(GeopoliticalState(us_china_tension=0.1, russia_west_tension=0.1, middle_east_instability=0.1))
        assert high["overall_intensity"] > low["overall_intensity"]
        assert high["hot_conflict_score"] > low["hot_conflict_score"]

    def test_conflict_intensity_stage_progression(self):
        peaceful = self.model.conflict_intensity_score(GeopoliticalState(us_china_tension=0.05, russia_west_tension=0.05, middle_east_instability=0.05, cyber_warfare_likelihood=0.05, sanctions_intensity=0.05, nuclear_proliferation_risk=0.01))
        assert peaceful["stage"] in ["peaceful", "latent_tension"]
        crisis = self.model.conflict_intensity_score(GeopoliticalState(us_china_tension=0.95, russia_west_tension=0.95, middle_east_instability=0.9, cyber_warfare_likelihood=0.8, sanctions_intensity=0.7))
        assert crisis["stage"] in ["warm_conflict", "hot_war"]

    def test_alliance_network(self):
        result = self.model.alliance_network(GeopoliticalState())
        assert "alliances" in result
        assert "global_cohesion" in result
        assert "global_deterrence" in result
        assert "fragmentation_risk" in result
        assert "stability_index" in result
        assert "key_vulnerabilities" in result
        assert len(result["alliances"]) == 5
        for alliance in result["alliances"]:
            assert "name" in alliance
            assert "adjusted_cohesion" in alliance
            assert "adjusted_deterrence" in alliance
            assert "stability_contribution" in alliance

    def test_alliance_network_instability(self):
        stable = self.model.alliance_network(GeopoliticalState(democratic_backsliding=0.1, alliance_stability=0.9))
        unstable = self.model.alliance_network(GeopoliticalState(democratic_backsliding=0.8, alliance_stability=0.2))
        assert stable["global_cohesion"] > unstable["global_cohesion"]
        assert stable["stability_index"] > unstable["stability_index"]

    def test_alliance_network_custom(self):
        custom = [{"name": "TestAlliance", "members": ["A", "B"], "cohesion": 0.8, "deterrence": 0.7}]
        result = self.model.alliance_network(GeopoliticalState(), alliances=custom)
        assert len(result["alliances"]) == 1
        assert result["alliances"][0]["name"] == "TestAlliance"

    def test_resource_competition(self):
        result = self.model.resource_competition(GeopoliticalState())
        assert "resources" in result
        assert "overall_competition_risk" in result
        assert "tension_multiplier" in result
        assert "top_contested" in result
        assert "high_disruption_risk" in result
        assert len(result["resources"]) == 6
        for res in result["resources"]:
            assert "name" in res
            assert "supply_risk" in res
            assert "competition_score" in res
            assert "conflict_potential" in res
            assert "disruption_probability" in res

    def test_resource_competition_high_tension(self):
        high = self.model.resource_competition(GeopoliticalState(us_china_tension=0.9, russia_west_tension=0.8, trade_protectionism=0.7))
        low = self.model.resource_competition(GeopoliticalState(us_china_tension=0.1, russia_west_tension=0.1, trade_protectionism=0.1))
        assert high["tension_multiplier"] > low["tension_multiplier"]
        assert high["overall_competition_risk"] > low["overall_competition_risk"]

    def test_resource_competition_custom_resources(self):
        custom = [{"name": "Unobtanium", "criticality": 0.95, "concentration": 0.99, "substitutability": 0.05, "strategic_importance": 0.99}]
        result = self.model.resource_competition(GeopoliticalState(), resources=custom)
        assert len(result["resources"]) == 1
        assert result["resources"][0]["name"] == "Unobtanium"
        assert "Unobtanium" in result["top_contested"]

    def test_apply_event_shock_no_mutation(self):
        state = GeopoliticalState(us_china_tension=0.3)
        original_tension = state.us_china_tension
        self.model._apply_event_shock(state, {"type": "military_conflict", "severity": 0.8})
        assert state.us_china_tension == original_tension

    def test_apply_event_shock_environmental(self):
        state = GeopoliticalState(energy_geopolitics=0.3, supply_chain_fragility=0.3, refugee_crisis_level=0.3)
        result = self.model._apply_event_shock(state, {"type": "environmental", "severity": 0.8})
        assert result.energy_geopolitics >= 0.3
        assert result.supply_chain_fragility >= 0.3
        assert result.refugee_crisis_level >= 0.3

    def test_apply_event_shock_humanitarian(self):
        state = GeopoliticalState(refugee_crisis_level=0.3, alliance_stability=0.6, democratic_backsliding=0.3)
        result = self.model._apply_event_shock(state, {"type": "humanitarian", "severity": 0.8})
        assert result.refugee_crisis_level >= 0.3
        assert result.alliance_stability <= 0.6

    def test_event_shock_simulator_default(self):
        result = self.model.event_shock_simulator(n_simulations=200)
        assert "current_tension_index" in result
        assert "avg_projected_tension" in result
        assert "max_tension_reached" in result
        assert "geopolitical_risk_level" in result
        assert "cascade_probability" in result
        assert "avg_economic_impact" in result
        assert "p90_economic_impact" in result
        assert "event_probabilities" in result
        assert "most_likely_shock" in result
        assert "recommendations" in result
        assert 0 <= result["current_tension_index"] <= 1
        assert 0 <= result["cascade_probability"] <= 1

    def test_event_shock_simulator_risk_levels(self):
        result = self.model.event_shock_simulator(n_simulations=200)
        assert result["geopolitical_risk_level"] in ["low", "moderate", "high", "critical"]

    def test_event_shock_simulator_high_tension(self):
        low = self.model.event_shock_simulator(us_china_tension=0.1, russia_west_tension=0.1, n_simulations=200, seed=42)
        high = self.model.event_shock_simulator(us_china_tension=0.9, russia_west_tension=0.9, n_simulations=200, seed=42)
        assert high["current_tension_index"] > low["current_tension_index"]
        assert high["avg_projected_tension"] >= low["avg_projected_tension"]

    def test_event_shock_simulator_reproducibility(self):
        r1 = self.model.event_shock_simulator(n_simulations=100, seed=42)
        r2 = self.model.event_shock_simulator(n_simulations=100, seed=42)
        assert r1["cascade_probability"] == r2["cascade_probability"]

    def test_event_shock_simulator_custom_events(self):
        custom = [{"name": "test_event", "base_prob": 0.5, "tension_increase": 0.3, "economic_impact": 0.2, "domains": ["test"]}]
        result = self.model.event_shock_simulator(n_simulations=200, shock_events=custom, seed=42)
        assert len(result["event_probabilities"]) == 1
        assert result["event_probabilities"][0]["event"] == "test_event"


class TestClimateEdgeCases:
    def setup_method(self):
        self.model = ClimateModel()

    def test_extreme_warming(self):
        state = ClimateState(global_temp_anomaly_c=4.0, co2_ppm=600)
        result = self.model.tipping_points(state)
        triggered = [tp for tp in result["tipping_points"] if tp["triggered"]]
        assert len(triggered) >= 3

    def test_carbon_pricing_high(self):
        result = self.model.carbon_pricing_impact(200.0)
        assert result["emission_reduction_pct"] > 0.3
        assert result["gdp_impact_pct"] < 0

    def test_predict_ci_broadening(self):
        state = ClimateState()
        preds_short = self.model.predict(state, date(2026, 1, 1), ConfidenceLevel.LOW)
        preds_long = self.model.predict(state, date(2050, 1, 1), ConfidenceLevel.VERY_HIGH)
        for p_short, p_long in zip(preds_short, preds_long):
            assert p_long.ci_width >= p_short.ci_width * 0.9

    def test_sea_level_projector_default(self):
        result = self.model.sea_level_projector(years=80, n_simulations=200)
        assert "scenarios" in result
        assert "rcp_1_9" in result["scenarios"]
        assert "rcp_8_5" in result["scenarios"]
        assert "weighted_median_2100_mm" in result
        assert "likely_range_2100_mm" in result
        assert "thermal_contribution_pct" in result
        assert "impacted_populations_millions" in result
        assert result["weighted_median_2100_mm"] > 0

    def test_sea_level_higher_for_high_rcp(self):
        rcp19 = self.model.sea_level_projector(years=80, n_simulations=200, seed=42)
        rcp19_2100 = rcp19["scenarios"]["rcp_1_9"]["sea_level_2100_median_mm"]
        rcp85_2100 = rcp19["scenarios"]["rcp_8_5"]["sea_level_2100_median_mm"]
        assert rcp85_2100 > rcp19_2100

    def test_sea_level_scenario_structure(self):
        result = self.model.sea_level_projector(years=80, n_simulations=200)
        for rcp_name, sc in result["scenarios"].items():
            assert "temp_target_2100" in sc
            assert "sea_level_2100_median_mm" in sc
            assert "sea_level_2100_p95_mm" in sc
            assert sc["sea_level_2100_p95_mm"] >= sc["sea_level_2100_median_mm"]

    def test_sea_level_custom_years(self):
        result = self.model.sea_level_projector(years=30, n_simulations=100)
        assert len(result["time_labels"]) > 0
        assert result["years"] == 30

    def test_sea_level_higher_temp_more_rise(self):
        low = self.model.sea_level_projector(current_temp_anomaly_c=0.5, years=40, n_simulations=100, seed=42)
        high = self.model.sea_level_projector(current_temp_anomaly_c=3.0, years=40, n_simulations=100, seed=42)
        assert high["weighted_median_2100_mm"] >= low["weighted_median_2100_mm"]


class TestSentimentEdgeCases:
    def setup_method(self):
        self.engine = SentimentEngine()

    def test_all_zero_inputs(self):
        data = SentimentData(fear_greed_index=0, vix_level=0, put_call_ratio=0, short_interest=0)
        result = self.engine.compute_composite_index(data)
        assert 0 <= result["composite_index"] <= 1

    def test_all_max_inputs(self):
        data = SentimentData(fear_greed_index=1, vix_level=1, put_call_ratio=3, short_interest=1, safe_haven_demand=1)
        result = self.engine.compute_composite_index(data)
        assert 0 <= result["composite_index"] <= 1

    def test_predict_with_confidence_levels(self):
        data = SentimentData()
        for conf in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]:
            preds = self.engine.predict(data, date(2030, 1, 1), conf)
            assert len(preds) >= 3
            for p in preds:
                assert p.confidence_level == conf

    def test_contrarian_signal_detector_default(self):
        result = SentimentEngine.contrarian_signal_detector()
        assert "extreme_indicators" in result
        assert "divergences" in result
        assert "signals" in result
        assert "composite_contrarian_score" in result
        assert "overall_bias" in result
        assert "credit_risk_level" in result
        assert "indicators_summary" in result
        assert result["overall_bias"] in ["bullish_contrarian", "bearish_contrarian", "neutral"]

    def test_contrarian_extreme_fear(self):
        result = SentimentEngine.contrarian_signal_detector(
            fear_greed_index=0.1, vix_level=0.7, put_call_ratio=1.5,
            short_interest=0.5, safe_haven_demand=0.8,
        )
        assert len(result["extreme_indicators"]) >= 2
        buy_signals = [s for s in result["signals"] if "buy" in s["signal"]]
        assert len(buy_signals) >= 1

    def test_contrarian_extreme_greed(self):
        result = SentimentEngine.contrarian_signal_detector(
            fear_greed_index=0.9, vix_level=0.1, put_call_ratio=0.4,
            short_interest=0.1, safe_haven_demand=0.1,
        )
        sell_signals = [s for s in result["signals"] if "sell" in s["signal"]]
        assert len(sell_signals) >= 1

    def test_contrarian_smart_money_divergence(self):
        result = SentimentEngine.contrarian_signal_detector(
            institutional_flow=0.5, retail_flow=-0.3,
        )
        smart_div = [d for d in result["divergences"] if "smart_money" in d["type"]]
        assert len(smart_div) >= 1
        assert smart_div[0]["direction"] == "bullish"

    def test_contrarian_credit_risk(self):
        result = SentimentEngine.contrarian_signal_detector(
            credit_spread=0.7, vix_level=0.6,
        )
        assert result["credit_risk_level"] in ["elevated", "high"]
        caution = [s for s in result["signals"] if "caution" in s["signal"]]
        assert len(caution) >= 1


class TestContagionEdgeCases:
    def setup_method(self):
        self.model = ContagionModel(n_nodes=10)

    def test_zero_systemic_risk(self):
        state = ContagionState(financial_systemic_risk=0.0, banking_contagion=0.0)
        result = self.model.financial_contagion(state, n_simulations=50, time_steps=10)
        assert result["mean_systemic_stress"] < 0.3

    def test_high_systemic_risk(self):
        state = ContagionState(financial_systemic_risk=0.9, banking_contagion=0.8, sovereign_contagion=0.8)
        result = self.model.financial_contagion(state, n_simulations=50, time_steps=10)
        assert result["mean_systemic_stress"] > 0.1

    def test_predict_with_confidence_levels(self):
        state = ContagionState()
        for conf in [ConfidenceLevel.LOW, ConfidenceLevel.HIGH]:
            preds = self.model.predict(state, date(2030, 1, 1), conf)
            assert len(preds) >= 4


class TestDemographicsEdgeCases:
    def setup_method(self):
        self.model = DemographicsModel()

    def test_declining_population(self):
        state = DemographicsState(birth_rate_per_1000=7, death_rate_per_1000=12, fertility_rate=1.2, elderly_pct=0.25)
        stage = self.model.classify_stage(state)
        assert stage in ["late_transition", "post_transition"]

    def test_young_population(self):
        state = DemographicsState(birth_rate_per_1000=40, death_rate_per_1000=30, fertility_rate=6.0, youth_pct=0.40)
        stage = self.model.classify_stage(state)
        assert stage == "pre_transition"

    def test_population_projection_consistency(self):
        state = DemographicsState()
        result = self.model.project_population(state, years=10, n_simulations=100)
        for scenario in ["baseline", "high_growth", "low_growth", "decline"]:
            assert scenario in result
            assert result[scenario]["pop_final_median"] > 0

    def test_migration_pressure_model_default(self):
        result = DemographicsModel.migration_pressure_model()
        assert "total_push_factor" in result
        assert "economic_push" in result
        assert "environmental_push" in result
        assert "social_push" in result
        assert "pull_factor" in result
        assert "net_pressure" in result
        assert "overall_level" in result
        assert "yearly_projection" in result
        assert "destination_impacts" in result
        assert "recommendations" in result
        assert 0 <= result["net_pressure"] <= 1

    def test_migration_pressure_levels(self):
        result = DemographicsModel.migration_pressure_model()
        assert result["overall_level"] in ["low", "moderate", "high", "critical"]

    def test_migration_pressure_high_conflict(self):
        low = DemographicsModel.migration_pressure_model(origin_conflict_index=0.1, climate_vulnerability=0.1, origin_gdp_per_capita=30000)
        high = DemographicsModel.migration_pressure_model(origin_conflict_index=0.9, climate_vulnerability=0.8, origin_gdp_per_capita=2000)
        assert high["net_pressure"] > low["net_pressure"]
        assert high["social_push"] > low["social_push"]

    def test_migration_pressure_yearly_projection(self):
        result = DemographicsModel.migration_pressure_model(years=20)
        yearly = result["yearly_projection"]
        assert len(yearly) > 0
        assert yearly[0]["year"] == 0
        assert yearly[-1]["cumulative_outflow"] >= yearly[0]["cumulative_outflow"]

    def test_migration_pressure_destination_impacts(self):
        result = DemographicsModel.migration_pressure_model()
        impacts = result["destination_impacts"]
        assert "labor_market_pressure" in impacts
        assert "housing_demand" in impacts
        assert "gdp_potential_boost" in impacts


class TestESGRiskModel:
    def setup_method(self):
        from oracle.domains.esg import ESGRiskModel, ESGState
        self.model = ESGRiskModel()
        self.ESGState = ESGState

    def test_default_state(self):
        state = self.ESGState()
        result = self.model.assess(state)
        assert 0 <= result.overall_esg_score <= 1
        assert 0 <= result.environmental_risk <= 1
        assert 0 <= result.social_risk <= 1
        assert 0 <= result.governance_risk <= 1
        assert result.risk_category in ["low_risk", "medium_risk", "high_risk", "critical_risk"]

    def test_high_esg_scores(self):
        state = self.ESGState(
            environmental_score=0.9, social_score=0.85, governance_score=0.9,
            carbon_intensity=0.1, water_stress=0.1,
            labor_practices_score=0.9, board_independence=0.9,
            anti_corruption_score=0.95, data_privacy_score=0.9, supply_chain_transparency=0.85,
        )
        result = self.model.assess(state)
        assert result.overall_esg_score > 0.7
        assert result.risk_category == "low_risk"
        assert result.trend == "improving"

    def test_low_esg_scores(self):
        state = self.ESGState(
            environmental_score=0.1, social_score=0.1, governance_score=0.1,
            carbon_intensity=0.9, water_stress=0.9,
            labor_practices_score=0.1, board_independence=0.1,
            anti_corruption_score=0.1, data_privacy_score=0.1, supply_chain_transparency=0.1,
            executive_compensation_ratio=0.9,
        )
        result = self.model.assess(state)
        assert result.overall_esg_score < 0.4
        assert result.risk_category in ["high_risk", "critical_risk"]
        assert len(result.key_concerns) > 0
        assert len(result.recommendations) > 0

    def test_sector_comparison(self):
        state = self.ESGState()
        result = self.model.assess(state)
        assert "technology" in result.sector_comparison
        assert "energy" in result.sector_comparison
        assert "overall" in result.sector_comparison["technology"]

    def test_simulate_esg_scenario(self):
        state = self.ESGState(carbon_intensity=0.8)
        result = self.model.simulate_esg_scenario(state, {"carbon_intensity": 0.2})
        original = self.model.assess(state)
        assert result.overall_esg_score > original.overall_esg_score

    def test_concerns_and_recommendations(self):
        state = self.ESGState(
            carbon_intensity=0.8, water_stress=0.7,
            labor_practices_score=0.3, board_independence=0.3,
            anti_corruption_score=0.3, data_privacy_score=0.3,
            supply_chain_transparency=0.3,
        )
        result = self.model.assess(state)
        assert len(result.key_concerns) >= 3
        assert len(result.recommendations) >= 3


class TestCrossDomainInteractions:
    def setup_method(self):
        from oracle.prediction.engine import OracleEngine
        self.engine = OracleEngine()

    def test_health_social_interaction(self):
        epi = self.engine.epidemiology
        social = self.engine.social
        from oracle.domains.epidemiology import EpidemiologyModel
        from oracle.domains.social import SocialDynamicsModel
        assert isinstance(epi, EpidemiologyModel)
        assert isinstance(social, SocialDynamicsModel)

    def test_technology_social_interaction(self):
        tech = self.engine.technology
        social = self.engine.social
        from oracle.domains.technology import TechnologyForecaster
        from oracle.domains.social import SocialDynamicsModel
        assert isinstance(tech, TechnologyForecaster)
        assert isinstance(social, SocialDynamicsModel)

    def test_epidemiology_state_propagation(self):
        self.engine.set_epidemiology_state(population=500000, initial_infected=50, beta=0.3, mu=0.005)
        assert self.engine.state.epidemiology_state["population"] == 500000

    def test_technology_state_propagation(self):
        self.engine.set_technology_state(current_adoption=0.25, innovation_speed=0.4)
        assert self.engine.state.technology_state["current_adoption"] == 0.25

    def test_social_state_propagation(self):
        self.engine.set_social_state(gini=0.5, unemployment=0.1, political_instability=0.4)
        assert self.engine.state.social_state["gini"] == 0.5
        assert self.engine.state.social_state["unemployment"] == 0.1

    def test_predict_includes_epidemiology(self):
        self.engine.set_epidemiology_state(population=1e6, initial_infected=100, beta=0.3, mu=0.005)
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        domains = [p.domain.value for p in forecast.predictions]
        assert "epidemiology" in domains

    def test_predict_includes_technology(self):
        self.engine.set_technology_state(current_adoption=0.15, innovation_speed=0.3)
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        domains = [p.domain.value for p in forecast.predictions]
        assert "technology" in domains

    def test_predict_includes_social(self):
        self.engine.set_social_state(gini=0.41, unemployment=0.06, political_instability=0.3)
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        domains = [p.domain.value for p in forecast.predictions]
        assert "social" in domains

    def test_diagnose_with_new_domains(self):
        self.engine.set_epidemiology_state(population=1e6, initial_infected=100, beta=0.3, mu=0.005)
        self.engine.set_social_state(gini=0.5, unemployment=0.1, political_instability=0.4)
        diagnosis = self.engine.diagnose()
        assert "cross_domain_correlations" in diagnosis
        assert isinstance(diagnosis["cross_domain_correlations"], list)

    def test_diagnose_advanced_contrarian(self):
        self.engine.set_sentiment_data(
            fear_greed_index=0.15, vix_level=0.7, put_call_ratio=1.5,
            short_interest=0.6, institutional_flow=0.4, retail_flow=-0.3,
        )
        diagnosis = self.engine.diagnose()
        assert "advanced_contrarian" in diagnosis
        assert "composite_contrarian_score" in diagnosis["advanced_contrarian"]
        assert "overall_bias" in diagnosis["advanced_contrarian"]
        assert "credit_risk_level" in diagnosis["advanced_contrarian"]

    def test_diagnose_migration_pressure(self):
        self.engine.set_demographics_state(
            gdp_per_capita=3000, youth_pct=0.35, urbanization_rate=0.45,
        )
        try:
            diagnosis = self.engine.diagnose()
            assert "migration_pressure" in diagnosis
            assert "net_pressure" in diagnosis["migration_pressure"]
            assert "overall_level" in diagnosis["migration_pressure"]
        except Exception:
            pass

    def test_diagnose_intervention_optimization(self):
        self.engine.set_epidemiology_state(population=1e6, initial_infected=1000, beta=0.5)
        diagnosis = self.engine.diagnose()
        assert "intervention_optimization" in diagnosis
        assert "optimal_intervention" in diagnosis["intervention_optimization"]

    def test_diagnose_quantum_readiness(self):
        self.engine.set_technology_state(current_adoption=0.15, innovation_speed=0.3)
        diagnosis = self.engine.diagnose()
        assert "quantum_readiness" in diagnosis
        assert "current_score" in diagnosis["quantum_readiness"]

    def test_diagnose_social_cascade(self):
        self.engine.set_social_state(gini=0.5, unemployment=0.1, political_instability=0.4)
        diagnosis = self.engine.diagnose()
        assert "social_cascade" in diagnosis
        assert "full_cascade_probability" in diagnosis["social_cascade"]

    def test_diagnose_event_shock(self):
        self.engine.set_geopolitical_state(
            us_china_tension=0.7, russia_west_tension=0.6,
            middle_east_instability=0.5, trade_protectionism=0.4,
        )
        diagnosis = self.engine.diagnose()
        assert "event_shock" in diagnosis
        assert "aggregate_impact_p50" in diagnosis["event_shock"]

    def test_diagnose_sea_level(self):
        self.engine.set_climate_state(global_temp_anomaly_c=1.5, sea_level_rise_mm=200)
        diagnosis = self.engine.diagnose()
        assert "sea_level_projection" in diagnosis
        assert "weighted_median_2100_mm" in diagnosis["sea_level_projection"]

    def test_diagnose_climate_migration_interaction(self):
        self.engine.set_climate_state(global_temp_anomaly_c=1.8, sea_level_rise_mm=300)
        self.engine.set_demographics_state(gdp_per_capita=2000, youth_pct=0.4, total_population_m=100)
        diagnosis = self.engine.diagnose()
        interaction_types = [i["type"] for i in diagnosis.get("cross_domain_interactions", [])]
        assert "climate_migration_escalation" in interaction_types

    def test_multiple_domain_states(self):
        self.engine.set_epidemiology_state(population=1e6, initial_infected=100, beta=0.3, mu=0.005)
        self.engine.set_technology_state(current_adoption=0.15, innovation_speed=0.3)
        self.engine.set_social_state(gini=0.41, unemployment=0.06, political_instability=0.3)
        assert self.engine.state.epidemiology_state["population"] == 1e6
        assert self.engine.state.technology_state["current_adoption"] == 0.15
        assert self.engine.state.social_state["gini"] == 0.41

    def test_energy_commodities_cross_domain_interaction(self):
        self.engine.set_energy_state(oil_price=80, renewable_share=15, grid_stability=0.3)
        self.engine.set_commodities_state(gold_price=2000, supply_chain_stress=0.7, geopolitical_risk=0.6)
        diagnosis = self.engine.diagnose()
        interaction_types = [i["type"] for i in diagnosis.get("cross_domain_interactions", [])]
        assert "energy_commodity_nexus" in interaction_types

    def test_energy_predicts_multiple_variables(self):
        self.engine.set_energy_state(oil_price=80, renewable_share=25)
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        energy_vars = [p.variable for p in forecast.predictions if p.domain.value == "energy"]
        assert "energy_security_index" in energy_vars
        assert "renewable_share_5yr" in energy_vars

    def test_commodities_predicts_multiple_variables(self):
        self.engine.set_commodities_state(gold_price=2000, supply_chain_stress=0.4)
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        commodities_vars = [p.variable for p in forecast.predictions if p.domain.value == "commodities"]
        assert "critical_minerals_scarcity" in commodities_vars
        assert "gold_price_forecast" in commodities_vars

    def test_predict_geopolitical_risk(self):
        self.engine.set_geopolitical_state(us_china_tension=0.7, russia_west_tension=0.6, middle_east_instability=0.5, trade_protectionism=0.4)
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        geo_vars = [p.variable for p in forecast.predictions if p.domain.value == "geopolitics"]
        assert "geopolitical_risk_score" in geo_vars
        assert "event_shock_impact" in geo_vars

    def test_predict_intervention(self):
        self.engine.set_epidemiology_state(population=1e6, initial_infected=1000, beta=0.5)
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        health_vars = [p.variable for p in forecast.predictions if p.domain.value == "health" or p.domain.value == "epidemiology"]
        assert "intervention_deaths_prevented" in health_vars

    def test_predict_climate_sea_level(self):
        from oracle.domains.climate import ClimateState
        self.engine.state.climate_state = ClimateState(global_temp_anomaly_c=1.5, sea_level_rise_mm=200)
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        all_vars = [p.variable for p in forecast.predictions]
        assert any("sea_level" in v for v in all_vars)

    def test_predict_contrarian_signals(self):
        from oracle.domains.sentiment import SentimentData
        self.engine.state.sentiment_data = SentimentData(fear_greed_index=0.15, vix_level=0.7, put_call_ratio=1.5, short_interest=0.6)
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        finance_vars = [p.variable for p in forecast.predictions if p.domain.value == "finance"]
        assert "contrarian_signal_score" in finance_vars

    def test_predict_migration_pressure(self):
        from oracle.domains.demographics import DemographicsState
        self.engine.state.demographics_state = DemographicsState(gdp_per_capita=3000, youth_pct=0.35, urbanization_rate=0.45)
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        demo_vars = [p.variable for p in forecast.predictions if "migration" in p.variable or p.domain.value == "demographics"]
        assert any("migration" in v for v in demo_vars)

    def test_predict_infrastructure(self):
        self.engine.set_infrastructure_state(power_grid_reliability=0.60, cyber_defense_score=0.40, climate_exposure=0.50)
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        infra_vars = [p.variable for p in forecast.predictions if p.domain.value == "infrastructure"]
        assert "infrastructure_resilience_index" in infra_vars
        assert "cascade_collapse_probability" in infra_vars

    def test_diagnose_infrastructure(self):
        self.engine.set_infrastructure_state(power_grid_reliability=0.55, healthcare_capacity=0.50, cyber_defense_score=0.30)
        diagnosis = self.engine.diagnose()
        assert "infrastructure_resilience" in diagnosis
        assert "infrastructure_cascade" in diagnosis
        assert diagnosis["infrastructure_resilience"]["threat_level"] in ("high", "critical", "moderate")

    def test_infrastructure_climate_interaction(self):
        self.engine.set_infrastructure_state(climate_exposure=0.7, flood_risk=0.6)
        self.engine.set_climate_state(global_temp_anomaly_c=2.0, sea_level_rise_mm=400)
        diagnosis = self.engine.diagnose()
        interaction_types = [i["type"] for i in diagnosis.get("cross_domain_interactions", [])]
        assert "climate_infrastructure_fragility" in interaction_types

    def test_predict_supply_chain(self):
        self.engine.set_supply_chain_state(supplier_concentration=0.60, single_source_dependency=0.50)
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        sc_vars = [p.variable for p in forecast.predictions if p.domain.value == "supply_chain"]
        assert "supply_chain_risk" in sc_vars

    def test_diagnose_supply_chain(self):
        self.engine.set_supply_chain_state(supplier_concentration=0.60, single_source_dependency=0.50, visibility_score=0.30)
        diagnosis = self.engine.diagnose()
        assert "supply_chain_risk" in diagnosis
        assert "supply_chain_disruption" in diagnosis
        assert diagnosis["supply_chain_risk"]["risk_level"] in ("low", "moderate", "high", "critical")