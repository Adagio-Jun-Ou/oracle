import pytest
from datetime import date
from oracle.prediction.engine import OracleEngine
from oracle.core.types import Domain


class TestFullDomainIntegration:
    def setup_method(self):
        self.engine = OracleEngine()

    def test_predict_all_domains(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02, debt_to_gdp=1.2)
        self.engine.set_geopolitical_state(
            us_china_tension=0.5, russia_west_tension=0.6,
            middle_east_instability=0.4, trade_protectionism=0.3,
            cyber_warfare_likelihood=0.3, energy_geopolitics=0.4,
        )
        self.engine.set_climate_state(
            global_temp_anomaly_c=1.2, sea_level_rise_mm=100,
            extreme_weather_index=0.5,
        )
        self.engine.set_sentiment_data(
            fear_greed_index=45, vix_level=18,
        )
        self.engine.set_demographics_state(
            total_population_m=8000, elderly_pct=0.18,
        )
        self.engine.set_epidemiology_state(population=1e6, initial_infected=100, beta=0.3)
        self.engine.set_technology_state(current_adoption=0.15, innovation_speed=0.3)
        self.engine.set_social_state(gini=0.41, unemployment=0.06, political_instability=0.3)
        self.engine.set_energy_state(oil_price=75.0, renewable_share=20.0)
        self.engine.set_commodities_state(gold_price=1900.0, supply_chain_stress=0.3)
        self.engine.set_infrastructure_state(power_grid_reliability=0.85, water_system_reliability=0.90)
        self.engine.set_supply_chain_state(supplier_concentration=0.40)
        self.engine.set_cyber_state(threat_level=0.4, ransomware_frequency=0.3)
        self.engine.set_water_state(freshwater_availability_m3_per_capita=5000.0, water_stress_index=0.4)
        self.engine.set_food_state(climate_impact_on_crops=0.3, supply_chain_disruption=0.2)
        self.engine.set_political_state(government_stability=0.5, electoral_integrity=0.7)
        self.engine.set_biodiversity_state(genetic_diversity_index=0.5, habitat_loss_rate=0.02)
        self.engine.set_pandemic_state(healthcare_capacity=0.5, surveillance_systems=0.6)
        self.engine.set_space_state(orbital_debris_count=30000, kessler_syndrome_risk=0.3)
        self.engine.set_mental_health_state(depression_prevalence=0.05, anxiety_prevalence=0.04)
        self.engine.set_housing_state(median_home_price=350000.0, median_income=70000.0)
        self.engine.set_seismic_state(earthquake_rate_monthly=10, avg_magnitude=4.5)
        self.engine.set_solar_state(sunspot_number=80, flare_probability=0.15)
        self.engine.set_oceanic_state(enso_index=0.5, sea_surface_temp_anomaly_c=0.8)
        self.engine.set_neo_state(neo_count_tracked=30000, detection_rate_pct=0.95)
        self.engine.set_atmospheric_state(polar_vortex_strength=0.5, stratospheric_warming_risk=0.1)
        self.engine.set_paleoclimate_state(co2_ppmv=420, ice_volume_fraction=0.01)
        self.engine.set_cosmic_radiation_state(gcr_flux_relative=1.0, solar_modulation_potential_gv=0.5)
        self.engine.set_gravitational_state(lunar_tidal_force_relative=1.0)
        self.engine.set_compound_hazard_state(seismic_hazard=0.3, solar_storm_risk=0.2)

        forecast = self.engine.predict(date(2035, 1, 1))

        domains_found = set(p.domain for p in forecast.predictions)
        expected_domains = {
            Domain.ECONOMICS, Domain.GEOPOLITICS, Domain.CLIMATE,
            Domain.DEMOGRAPHICS, Domain.EPIDEMIOLOGY,
            Domain.TECHNOLOGY, Domain.SOCIAL, Domain.ENERGY, Domain.COMMODITIES,
            Domain.INFRASTRUCTURE, Domain.CYBER, Domain.WATER, Domain.FOOD,
            Domain.POLITICAL, Domain.BIODIVERSITY, Domain.PANDEMIC_PREPAREDNESS,
            Domain.SPACE, Domain.MENTAL_HEALTH, Domain.HOUSING,
            Domain.SEISMIC, Domain.SOLAR, Domain.OCEANIC,
            Domain.NEAR_EARTH_OBJECT, Domain.ATMOSPHERIC,
            Domain.PALEOCLIMATOLOGY, Domain.COSMIC_RADIATION,
            Domain.GRAVITATIONAL, Domain.COMPOUND_HAZARD,
        }
        missing = expected_domains - domains_found
        assert len(missing) <= 5, f"Too many domains missing from predict: {missing}"
        assert len(forecast.predictions) > 30

    def test_diagnose_all_domains(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02, debt_to_gdp=1.2)
        self.engine.set_geopolitical_state(
            us_china_tension=0.5, russia_west_tension=0.6,
            middle_east_instability=0.4, trade_protectionism=0.3,
            cyber_warfare_likelihood=0.3, energy_geopolitics=0.4,
        )
        self.engine.set_climate_state(
            global_temp_anomaly_c=1.2, sea_level_rise_mm=100,
            extreme_weather_index=0.5,
        )
        self.engine.set_infrastructure_state(power_grid_reliability=0.85)
        self.engine.set_cyber_state(threat_level=0.4)
        self.engine.set_water_state(water_stress_index=0.4)
        self.engine.set_food_state(climate_impact_on_crops=0.3)
        self.engine.set_political_state(government_stability=0.5)
        self.engine.set_biodiversity_state(genetic_diversity_index=0.5)
        self.engine.set_seismic_state(earthquake_rate_monthly=10, avg_magnitude=4.5)
        self.engine.set_solar_state(sunspot_number=80)
        self.engine.set_compound_hazard_state(seismic_hazard=0.3, solar_storm_risk=0.2)

        result = self.engine.diagnose()

        expected_keys = [
            "cycle_diagnosis", "geopolitical_risk",
            "climate_tipping_points", "infrastructure_resilience",
            "cyber_threat", "water_security", "food_security",
            "political_stability", "biodiversity_health",
            "seismic_hazard", "solar_weather",
            "compound_hazard", "world_state",
            "cross_domain_interactions",
        ]
        found = [k for k in expected_keys if k in result]
        assert len(found) >= 8, f"Missing diagnose keys: {set(expected_keys) - set(found)}"
        assert "world_state" in result
        assert result["world_state"]["overall_stability"] > 0
        assert len(result.get("cross_domain_interactions", [])) > 0

    def test_fate_cross_domain_interactions(self):
        self.engine.set_fate_state(
            co2_ppm=440.0,
            economic_inequality_gini=0.7,
            governance_effectiveness=0.3,
            technological_dependency=0.8,
            biodiversity_integrity_index=0.3,
            resource_remaining_pct=35.0,
            ai_capability_index=0.35,
            nuclear_warheads=13000,
            nuclear_proliferation_risk=0.4,
            water_stress_global=0.6,
            food_system_resilience=0.4,
            information_integrity=0.3,
            climate_commitment_c=1.8,
            energy_consumption_tw=20.0,
        )
        self.engine.set_geopolitical_state(us_china_tension=0.7, nuclear_proliferation_risk=0.4)
        self.engine.set_climate_state(global_temp_anomaly_c=1.8, sea_level_rise_mm=150)
        self.engine.set_energy_state(oil_price=100, renewable_share=18)
        self.engine.set_commodities_state(gold_price=2000, supply_chain_stress=0.6)
        self.engine.set_cyber_state(threat_level=0.5, ai_threat_capability=0.5)
        self.engine.set_biodiversity_state(genetic_diversity_index=0.3, habitat_loss_rate=0.05)
        self.engine.set_political_state(government_stability=0.3)
        self.engine.set_mental_health_state(depression_prevalence=0.08, anxiety_prevalence=0.06)
        self.engine.set_social_state(gini=0.7, unemployment=0.08, political_instability=0.5)
        self.engine.set_food_state(climate_impact_on_crops=0.5, supply_chain_disruption=0.4)
        self.engine.set_water_state(water_stress_index=0.6)
        self.engine.set_infrastructure_state(power_grid_reliability=0.5)
        self.engine.set_technology_state(current_adoption=0.35, innovation_speed=0.3)
        self.engine.set_economic_state(gdp_growth_rate=0.01, debt_to_gdp=1.8)
        self.engine.set_pandemic_state(healthcare_capacity=0.3, surveillance_systems=0.4)
        result = self.engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        fate_types = [i["type"] for i in interactions if "fate_" in i["type"]]
        assert len(fate_types) >= 3, f"Expected >= 3 fate interactions, got {len(fate_types)}: {fate_types}"

    def test_prescience_predict_in_forecast(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02, debt_to_gdp=1.2)
        self.engine.set_geopolitical_state(us_china_tension=0.5, russia_west_tension=0.5)
        self.engine.set_climate_state(global_temp_anomaly_c=1.5, sea_level_rise_mm=100)
        self.engine.set_seismic_state(earthquake_rate_monthly=10, avg_magnitude=4.5)
        self.engine.set_solar_state(sunspot_number=80, flare_probability=0.15)
        forecast = self.engine.predict(date(2035, 1, 1))
        domains_found = set(p.domain for p in forecast.predictions)
        assert Domain.PRESCIENCE in domains_found, f"PRESCIENCE domain not in forecast predictions: {domains_found}"
        prescience_preds = [p for p in forecast.predictions if p.domain == Domain.PRESCIENCE]
        prescience_vars = [p.variable for p in prescience_preds]
        assert "prescience_score" in prescience_vars, f"prescience_score not found: {prescience_vars}"

    def test_prescience_cross_domain_interactions(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02, debt_to_gdp=1.5)
        self.engine.set_geopolitical_state(us_china_tension=0.7, russia_west_tension=0.6)
        self.engine.set_climate_state(global_temp_anomaly_c=1.8, sea_level_rise_mm=150)
        self.engine.set_pandemic_state(healthcare_capacity=0.3, surveillance_systems=0.3, supply_chain_resilience=0.3)
        self.engine.set_infrastructure_state(power_grid_reliability=0.5)
        self.engine.set_seismic_state(earthquake_rate_monthly=10, avg_magnitude=4.5)
        self.engine.set_solar_state(sunspot_number=80, flare_probability=0.15)
        self.engine.set_cyber_state(threat_level=0.5)
        self.engine.set_water_state(water_stress_index=0.5)
        self.engine.set_food_state(climate_impact_on_crops=0.5, supply_chain_disruption=0.3)
        self.engine.set_political_state(government_stability=0.3)
        self.engine.set_biodiversity_state(genetic_diversity_index=0.4, habitat_loss_rate=0.04)
        self.engine.set_supply_chain_state(supplier_concentration=0.5)
        self.engine.set_mental_health_state(depression_prevalence=0.07, anxiety_prevalence=0.05)
        self.engine.set_housing_state(median_home_price=400000, median_income=60000)
        result = self.engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        prescience_types = [i["type"] for i in interactions if "prescience_" in i["type"]]
        assert len(prescience_types) >= 1, f"Expected >= 1 prescience interaction, got {len(prescience_types)}: {prescience_types}"

    def test_personal_fate_cross_domain_interactions(self):
        from oracle.domains.fate import PersonalFateState
        self.engine.set_personal_fate_state(
            age=55,
            genetic_risk_score=0.8,
            socioeconomic_origin=0.2,
            habit_strength=0.9,
            financial_safety_net=0.1,
            exercise_habit=0.1,
            diet_quality=0.2,
            sleep_quality=0.2,
            neuroplasticity=0.2,
            cognitive_reserve=0.2,
            geographic_mobility=0.1,
            purpose_clarity=0.1,
        )
        self.engine.set_mental_health_state(depression_prevalence=0.15, anxiety_prevalence=0.12)
        self.engine.set_fate_state(economic_inequality_gini=0.7, biotech_capability=0.2)
        self.engine.set_economic_state(gdp_growth_rate=0.01, debt_to_gdp=1.5)
        self.engine.set_social_state(gini=0.7, unemployment=0.08, political_instability=0.5)
        result = self.engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        personal_fate_types = [i["type"] for i in interactions if "personal_fate_" in i["type"]]
        assert len(personal_fate_types) >= 1, f"Expected >= 1 personal_fate interaction, got {len(personal_fate_types)}: {personal_fate_types}"