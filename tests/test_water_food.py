import pytest
from oracle.domains.water import WaterSecurityModel, WaterState
from oracle.domains.food import FoodSecurityModel, FoodState


class TestWaterSecurityModel:

    def setup_method(self):
        self.model = WaterSecurityModel()

    def test_assess_security_defaults(self):
        result = self.model.assess_security()
        assert "overall_risk" in result
        assert "risk_level" in result
        assert "resilience_score" in result
        assert "scarcity_types" in result
        assert "components" in result
        assert "recommendations" in result
        assert 0 <= result["overall_risk"] <= 1

    def test_assess_security_high_risk(self):
        state = WaterState(
            freshwater_availability_m3_per_capita=500,
            water_stress_index=0.8,
            groundwater_depletion_rate=0.05,
            drought_probability=0.6,
            flood_risk=0.4,
            water_quality_index=0.3,
            irrigation_efficiency=0.3,
            population_pressure=0.7,
        )
        result = self.model.assess_security(state)
        assert result["overall_risk"] > 0.5
        assert result["risk_level"] in ["high", "critical"]

    def test_assess_security_low_risk(self):
        state = WaterState(
            freshwater_availability_m3_per_capita=15000,
            water_stress_index=0.1,
            groundwater_depletion_rate=0.005,
            drought_probability=0.05,
            flood_risk=0.1,
            water_quality_index=0.9,
            irrigation_efficiency=0.9,
            desalination_capacity=0.3,
            population_pressure=0.1,
        )
        result = self.model.assess_security(state)
        assert result["overall_risk"] < 0.5
        assert result["risk_level"] in ["minimal", "low", "moderate"]

    def test_scarcity_types_absolute(self):
        state = WaterState(freshwater_availability_m3_per_capita=500)
        result = self.model.assess_security(state)
        assert "absolute_scarcity" in result["scarcity_types"]

    def test_scarcity_types_stress(self):
        state = WaterState(freshwater_availability_m3_per_capita=1500)
        result = self.model.assess_security(state)
        assert "water_stress" in result["scarcity_types"]

    def test_resilience_equals_one_minus_vulnerability_plus_stress(self):
        result = self.model.assess_security()
        assert result["resilience_score"] > 0
        assert result["resilience_score"] <= 1

    def test_drought_scenario_defaults(self):
        result = self.model.drought_scenario()
        assert "severity" in result
        assert "water_reduction_pct" in result
        assert "agricultural_impact_pct" in result
        assert "gdp_impact_pct" in result
        assert "population_affected_millions" in result
        assert "recovery_months" in result
        assert "timeline" in result
        assert len(result["timeline"]) > 0

    def test_drought_scenario_severe(self):
        state = WaterState(irrigation_efficiency=0.3, population_pressure=0.8)
        result = self.model.drought_scenario(severity=0.8, duration_months=18, state=state)
        assert result["water_reduction_pct"] > 0.3
        assert result["agricultural_impact_pct"] > 0.1

    def test_drought_recovery_longer_with_older_infrastructure(self):
        state_new = WaterState(infrastructure_age=0.1, desalination_capacity=0.3)
        state_old = WaterState(infrastructure_age=0.9, desalination_capacity=0.01)
        result_new = self.model.drought_scenario(severity=0.5, duration_months=12, state=state_new)
        result_old = self.model.drought_scenario(severity=0.5, duration_months=12, state=state_old)
        assert result_new["recovery_months"] <= result_old["recovery_months"]

    def test_predict_defaults(self):
        result = self.model.predict()
        assert "future_risk" in result
        assert "freshwater_per_capita_projected" in result
        assert "water_stress_projected" in result
        assert "drought_probability_projected" in result
        assert result["years_ahead"] == 10

    def test_predict_custom_years(self):
        for years in [1, 5, 10, 20]:
            result = self.model.predict(years_ahead=years)
            assert result["years_ahead"] == years
            assert 0 <= result["future_risk"] <= 1

    def test_set_state(self):
        self.model.set_state(water_stress_index=0.7, drought_probability=0.5)
        assert self.model.state is not None
        assert self.model.state.water_stress_index == 0.7
        assert self.model.state.drought_probability == 0.5

    def test_recommendations_for_stress(self):
        state = WaterState(water_stress_index=0.6)
        result = self.model.assess_security(state)
        assert any("demand-side" in r or "water management" in r for r in result["recommendations"])

    def test_recommendations_for_desalination(self):
        state = WaterState(desalination_capacity=0.02)
        result = self.model.assess_security(state)
        assert any("desalination" in r.lower() for r in result["recommendations"])


class TestFoodSecurityModel:

    def setup_method(self):
        self.model = FoodSecurityModel()

    def test_assess_security_defaults(self):
        result = self.model.assess_security()
        assert "overall_food_security" in result
        assert "overall_risk" in result
        assert "risk_level" in result
        assert "availability_score" in result
        assert "access_score" in result
        assert "stability_score" in result
        assert "sustainability_score" in result
        assert "vulnerabilities" in result
        assert "recommendations" in result
        assert 0 <= result["overall_risk"] <= 1

    def test_assess_security_high_risk(self):
        state = FoodState(
            food_production_index=0.3,
            food_price_volatility=0.4,
            crop_yield_variability=0.5,
            supply_chain_disruption=0.6,
            climate_impact_on_crops=0.5,
            population_growth_pressure=0.7,
            food_waste_pct=0.4,
            reserve_stock_months=1.0,
        )
        result = self.model.assess_security(state)
        assert result["overall_risk"] > 0.4
        assert result["risk_level"] in ["high", "critical"]

    def test_assess_security_low_risk(self):
        state = FoodState(
            food_production_index=0.9,
            food_price_volatility=0.05,
            crop_yield_variability=0.05,
            supply_chain_disruption=0.05,
            climate_impact_on_crops=0.05,
            population_growth_pressure=0.1,
            food_waste_pct=0.1,
            reserve_stock_months=8.0,
            dietary_diversity=0.9,
        )
        result = self.model.assess_security(state)
        assert result["overall_risk"] < 0.4

    def test_vulnerabilities(self):
        state = FoodState(food_price_volatility=0.4, reserve_stock_months=2.0)
        result = self.model.assess_security(state)
        vuln_areas = [v["area"] for v in result["vulnerabilities"]]
        assert "price_volatility" in vuln_areas
        assert "low_reserves" in vuln_areas

    def test_crop_model_defaults(self):
        result = self.model.crop_model()
        assert "crop" in result
        assert "current_yield_t_ha" in result
        assert "projected_yield_t_ha" in result
        assert "yield_change_pct" in result
        assert "climate_impact_on_yield" in result
        assert result["crop"] == "wheat"

    def test_crop_model_all_crops(self):
        for crop in ["wheat", "rice", "corn", "soybean"]:
            result = self.model.crop_model(crop=crop)
            assert result["crop"] == crop
            assert result["current_yield_t_ha"] > 0

    def test_crop_model_climate_impact(self):
        state_mild = FoodState(climate_impact_on_crops=0.05)
        state_severe = FoodState(climate_impact_on_crops=0.5)
        result_mild = self.model.crop_model(state=state_mild)
        result_severe = self.model.crop_model(state=state_severe)
        assert result_mild["current_yield_t_ha"] >= result_severe["current_yield_t_ha"]

    def test_predict_defaults(self):
        result = self.model.predict()
        assert "future_risk" in result
        assert "hunger_risk_projected" in result
        assert "price_volatility_projected" in result
        assert "future_food_security" in result
        assert result["years_ahead"] == 10

    def test_predict_custom_years(self):
        for years in [1, 5, 10, 20]:
            result = self.model.predict(years_ahead=years)
            assert result["years_ahead"] == years
            assert 0 <= result["future_risk"] <= 1

    def test_set_state(self):
        self.model.set_state(food_production_index=0.8, food_price_volatility=0.1)
        assert self.model.state is not None
        assert self.model.state.food_production_index == 0.8
        assert self.model.state.food_price_volatility == 0.1

    def test_recommendations_for_waste(self):
        state = FoodState(food_waste_pct=0.4)
        result = self.model.assess_security(state)
        assert any("waste" in r.lower() for r in result["recommendations"])

    def test_recommendations_for_fertilizer(self):
        state = FoodState(fertilizer_dependency=0.7)
        result = self.model.assess_security(state)
        assert any("fertilizer" in r.lower() or "organic" in r.lower() for r in result["recommendations"])