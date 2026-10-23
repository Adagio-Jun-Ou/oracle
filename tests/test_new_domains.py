import pytest
from datetime import date
from oracle.domains.space import SpaceModel, SpaceState
from oracle.domains.mental_health import MentalHealthModel, MentalHealthState
from oracle.domains.housing import HousingModel, HousingState
from oracle.prediction.engine import OracleEngine, OracleState
from oracle.core.types import Domain


class TestSpaceModel:
    def setup_method(self):
        self.model = SpaceModel()

    def test_space_state_defaults(self):
        state = SpaceState()
        assert state.orbital_debris_count == 25000.0
        assert state.active_satellites == 7500.0
        assert state.kessler_syndrome_risk == 0.15
        assert state.space_governance_index == 0.35

    def test_assess_orbital_environment_defaults(self):
        result = self.model.assess_orbital_environment()
        assert "collision_probability" in result
        assert "kessler_syndrome_risk" in result
        assert "kessler_status" in result
        assert "leo_congestion" in result
        assert "orbital_environment" in result
        assert "sustainable_orbit" in result
        assert 0 <= result["collision_probability"] <= 1

    def test_assess_orbital_environment_high_debris(self):
        state = SpaceState(orbital_debris_count=100000, active_satellites=50000)
        result = self.model.assess_orbital_environment(state)
        assert result["kessler_status"] in ("imminent", "elevated", "moderate", "low")
        assert result["collision_probability"] > 0

    def test_assess_orbital_environment_low_risk(self):
        state = SpaceState(orbital_debris_count=5000, active_satellites=2000, kessler_syndrome_risk=0.05)
        result = self.model.assess_orbital_environment(state)
        assert result["kessler_status"] == "low"
        assert result["recommendations"] is not None

    def test_simulate_kessler_default(self):
        result = self.model.simulate_kessler_scenario()
        assert "scenario" in result
        assert "kessler_triggered" in result
        assert "timeline" in result
        assert len(result["timeline"]) == 30

    def test_simulate_kessler_active_removal(self):
        result = self.model.simulate_kessler_scenario(scenario="active_removal")
        assert result["scenario"] == "active_removal"
        assert len(result["timeline"]) == 30

    def test_simulate_kessler_mega_constellation(self):
        result = self.model.simulate_kessler_scenario(scenario="mega_constellation")
        assert result["final_active_satellites"] > 0

    def test_predict_defaults(self):
        result = self.model.predict(years_ahead=10)
        assert "future_debris_count" in result
        assert "future_active_satellites" in result
        assert "future_kessler_risk" in result
        assert "future_governance_index" in result
        assert "future_commercial_revenue_billion" in result
        assert "orbital_sustainability_trend" in result

    def test_predict_with_state(self):
        state = SpaceState(orbital_debris_count=50000, active_satellites=15000, kessler_syndrome_risk=0.4)
        result = self.model.predict(years_ahead=5, state=state)
        assert result["future_kessler_risk"] >= state.kessler_syndrome_risk or result["orbital_sustainability_trend"] == "improving"

    def test_set_state(self):
        model = self.model.set_state(orbital_debris_count=60000, active_satellites=20000)
        assert model.state.orbital_debris_count == 60000
        assert model.state.active_satellites == 20000


class TestMentalHealthModel:
    def setup_method(self):
        self.model = MentalHealthModel()

    def test_mental_health_state_defaults(self):
        state = MentalHealthState()
        assert state.depression_prevalence == 0.05
        assert state.anxiety_prevalence == 0.04
        assert state.suicide_rate_per_100k == 10.5
        assert state.treatment_gap_pct == 0.55

    def test_assess_population_health_defaults(self):
        result = self.model.assess_population_health()
        assert "overall_burden" in result
        assert "burden_level" in result
        assert "system_capacity" in result
        assert "capacity_level" in result
        assert "treatment_access_rate" in result
        assert "youth_mental_health_risk" in result
        assert "workplace_impact_score" in result
        assert "risk_factors" in result
        assert "recommendations" in result

    def test_assess_crisis_level(self):
        state = MentalHealthState(
            depression_prevalence=0.30, anxiety_prevalence=0.25,
            suicide_rate_per_100k=40.0, social_isolation_index=0.85,
            substance_abuse_rate=0.20, treatment_gap_pct=0.7, stigma_index=0.7,
        )
        result = self.model.assess_population_health(state)
        assert result["burden_level"] in ("critical", "high")
        assert result["overall_burden"] > 0.3

    def test_assess_strong_system(self):
        state = MentalHealthState(
            treatment_gap_pct=0.2, stigma_index=0.2,
            mental_health_budget_pct=0.06, social_isolation_index=0.2,
        )
        result = self.model.assess_population_health(state)
        assert result["capacity_level"] in ("strong", "moderate")
        assert result["system_capacity"] > 0.4

    def test_assess_recommendations(self):
        state = MentalHealthState(treatment_gap_pct=0.7, stigma_index=0.6, social_isolation_index=0.6)
        result = self.model.assess_population_health(state)
        assert len(result["recommendations"]) >= 2

    def test_simulate_crisis_scenario(self):
        result = self.model.simulate_crisis_scenario()
        assert "scenario" in result
        assert "timeline" in result
        assert len(result["timeline"]) == 5
        assert result["final_depression_prevalence"] > 0

    def test_simulate_pandemic_aftermath(self):
        result = self.model.simulate_crisis_scenario(scenario="pandemic_aftermath")
        assert result["scenario"] == "pandemic_aftermath"
        assert result["treatment_gap_trajectory"] in ("widening", "narrowing")

    def test_simulate_reform(self):
        result = self.model.simulate_crisis_scenario(scenario="reform")
        assert result["treatment_gap_trajectory"] in ("widening", "narrowing")
        assert result["population_impact_increase_pct"] >= 0

    def test_predict_defaults(self):
        result = self.model.predict(years_ahead=10)
        assert "future_depression_prevalence" in result
        assert "future_anxiety_prevalence" in result
        assert "future_suicide_rate" in result
        assert "future_treatment_gap" in result
        assert "trajectory" in result

    def test_predict_trajectory_improving(self):
        state = MentalHealthState(
            treatment_gap_pct=0.3, stigma_index=0.3,
            mental_health_budget_pct=0.07, social_isolation_index=0.3,
        )
        result = self.model.predict(years_ahead=10, state=state)
        assert result["trajectory"] == "improving"

    def test_set_state(self):
        model = self.model.set_state(depression_prevalence=0.12, anxiety_prevalence=0.10)
        assert model.state.depression_prevalence == 0.12
        assert model.state.anxiety_prevalence == 0.10


class TestHousingModel:
    def setup_method(self):
        self.model = HousingModel()

    def test_housing_state_defaults(self):
        state = HousingState()
        assert state.median_home_price == 350000.0
        assert state.median_income == 70000.0
        assert state.mortgage_rate == 0.065
        assert state.vacancy_rate == 0.05

    def test_assess_market_defaults(self):
        result = self.model.assess_market()
        assert "market_health_score" in result
        assert "health_label" in result
        assert "affordability_index" in result
        assert "price_to_income_ratio" in result
        assert "monthly_payment_burden" in result
        assert "rental_burden" in result
        assert "bubble_risk" in result
        assert "recommendations" in result

    def test_assess_crisis_market(self):
        state = HousingState(
            median_home_price=800000, median_income=50000,
            vacancy_rate=0.01, mortgage_rate=0.03,
        )
        result = self.model.assess_market(state)
        assert result["health_label"] in ("crisis", "stressed")
        assert result["price_to_income_ratio"] > 10

    def test_assess_healthy_market(self):
        state = HousingState(
            median_home_price=250000, median_income=90000,
            vacancy_rate=0.08, mortgage_rate=0.05,
            rent_to_income_ratio=0.22, housing_supply_elasticity=0.7,
        )
        result = self.model.assess_market(state)
        assert result["health_label"] in ("healthy", "moderate")
        assert result["affordability_index"] > 0.4

    def test_simulate_scenario(self):
        result = self.model.simulate_housing_scenario()
        assert "scenario" in result
        assert "timeline" in result
        assert len(result["timeline"]) == 5
        assert result["final_median_price"] > 0

    def test_simulate_crash(self):
        result = self.model.simulate_housing_scenario(scenario="crash")
        assert result["scenario"] == "crash"
        assert result["final_price_to_income"] > 0

    def test_simulate_reform(self):
        result = self.model.simulate_housing_scenario(scenario="reform")
        assert result["affordability_trajectory"] == "improving"

    def test_predict_defaults(self):
        result = self.model.predict(years_ahead=5)
        assert "future_median_price" in result
        assert "future_median_income" in result
        assert "future_price_to_income" in result
        assert "future_affordability" in result
        assert "future_bubble_risk" in result
        assert "market_outlook" in result

    def test_predict_with_stressed_state(self):
        state = HousingState(median_home_price=600000, median_income=50000, mortgage_rate=0.08)
        result = self.model.predict(years_ahead=5, state=state)
        assert "market_outlook" in result
        assert result["future_price_to_income"] > 0

    def test_set_state(self):
        model = self.model.set_state(median_home_price=500000, median_income=80000)
        assert model.state.median_home_price == 500000
        assert model.state.median_income == 80000


class TestEngineNewDomains:
    def setup_method(self):
        self.engine = OracleEngine()

    def test_space_state_setting(self):
        self.engine.set_space_state(orbital_debris_count=50000, active_satellites=15000, kessler_syndrome_risk=0.4)
        assert self.engine.state.space_state is not None
        assert self.engine.state.space_state.orbital_debris_count == 50000

    def test_mental_health_state_setting(self):
        self.engine.set_mental_health_state(depression_prevalence=0.12, anxiety_prevalence=0.10)
        assert self.engine.state.mental_health_state is not None
        assert self.engine.state.mental_health_state.depression_prevalence == 0.12

    def test_housing_state_setting(self):
        self.engine.set_housing_state(median_home_price=500000, median_income=75000)
        assert self.engine.state.housing_state is not None
        assert self.engine.state.housing_state.median_home_price == 500000

    def test_diagnose_with_space(self):
        self.engine.set_space_state(kessler_syndrome_risk=0.5)
        result = self.engine.diagnose()
        assert "orbital_environment" in result
        assert "space_future" in result

    def test_diagnose_with_mental_health(self):
        self.engine.set_mental_health_state(depression_prevalence=0.15)
        result = self.engine.diagnose()
        assert "mental_health" in result
        assert "mental_health_future" in result

    def test_diagnose_with_housing(self):
        self.engine.set_housing_state(median_home_price=600000, median_income=50000)
        result = self.engine.diagnose()
        assert "housing_market" in result
        assert "housing_future" in result

    def test_cross_domain_space_cyber(self):
        self.engine.set_space_state(kessler_syndrome_risk=0.5, space_military_index=0.6, broadband_satellite_share=0.15)
        self.engine.set_cyber_state(ai_threat_capability=0.5, nation_state_activity=0.6)
        result = self.engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        interaction_types = [i["type"] for i in interactions]
        assert "space_military_debris_cascade" in interaction_types

    def test_cross_domain_mental_health_economic(self):
        self.engine.set_mental_health_state(depression_prevalence=0.20, anxiety_prevalence=0.18, social_isolation_index=0.8)
        self.engine.set_economic_state(debt_to_gdp=1.5, gdp_growth_rate=0.01)
        result = self.engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        interaction_types = [i["type"] for i in interactions]
        assert "mental_health_economic_burden" in interaction_types

    def test_cross_domain_housing_demographics(self):
        self.engine.set_housing_state(median_home_price=600000, median_income=50000, vacancy_rate=0.02)
        self.engine.set_demographics_state(urbanization_rate=0.85)
        result = self.engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        interaction_types = [i["type"] for i in interactions]
        assert "housing_demographics_affordability_crisis" in interaction_types

    def test_predict_with_new_domains(self):
        self.engine.set_space_state()
        self.engine.set_mental_health_state()
        self.engine.set_housing_state()
        forecast = self.engine.predict(date(2028, 1, 1))
        domain_values = [p.domain for p in forecast.predictions]
        assert Domain.SPACE in domain_values
        assert Domain.MENTAL_HEALTH in domain_values
        assert Domain.HOUSING in domain_values