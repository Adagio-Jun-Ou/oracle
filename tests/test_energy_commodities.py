import pytest
import numpy as np
from oracle.domains.energy import EnergyModel, EnergyState
from oracle.domains.commodities import CommoditiesModel, CommoditiesState


class TestEnergyState:
    def test_defaults(self):
        state = EnergyState()
        assert state.oil_price_usd == 75.0
        assert state.renewable_share_pct == 20.0
        assert state.nuclear_share_pct == 10.0
        assert state.coal_share_pct == 27.0
        assert state.grid_stability == 0.7

    def test_custom(self):
        state = EnergyState(oil_price_usd=100, renewable_share_pct=35.0, grid_stability=0.9)
        assert state.oil_price_usd == 100
        assert state.renewable_share_pct == 35.0
        assert state.grid_stability == 0.9


class TestEnergyModel:
    def setup_method(self):
        self.em = EnergyModel()

    def test_energy_mix_forecast(self):
        result = self.em.energy_mix_forecast(years=5)
        assert "yearly_forecast" in result
        assert len(result["yearly_forecast"]) == 6
        assert result["scenario"] == "current_policies"

    def test_energy_mix_announced_pledges(self):
        result = self.em.energy_mix_forecast(years=5, scenario="announced_pledges")
        assert result["renewable_share_final"] > 20

    def test_energy_mix_net_zero(self):
        result = self.em.energy_mix_forecast(years=10, scenario="net_zero_2050")
        assert result["co2_reduction_pct"] > 0
        assert result["renewable_share_final"] > 30

    def test_energy_transition_cost_default(self):
        result = self.em.energy_transition_cost()
        assert "capex" in result
        assert result["capex"]["total_capex_bn"] > 0
        assert result["net_cost_bn"] >= 0
        assert result["feasibility"] in ["achievable", "challenging", "extremely_ambitious"]

    def test_energy_transition_cost_ambitious(self):
        result = self.em.energy_transition_cost(years=5, target_renewable_pct=80.0)
        assert result["gap_pct"] > 50
        assert result["feasibility"] in ["challenging", "extremely_ambitious"]

    def test_energy_transition_cost_zero_gap(self):
        result = self.em.energy_transition_cost(target_renewable_pct=20.0)
        assert result["gap_pct"] == 0

    def test_grid_storage_requirements_default(self):
        result = self.em.grid_storage_requirements()
        assert "storage_requirements" in result
        assert result["storage_requirements"]["required_gw"] > 0
        assert result["storage_requirements"]["required_gwh"] > 0
        assert "cost_estimate" in result
        assert result["cost_estimate"]["total_storage_cost_bn"] > 0

    def test_grid_storage_high_renewable(self):
        result = self.em.grid_storage_requirements(renewable_target_pct=80.0)
        assert result["storage_requirements"]["required_gw"] > 100

    def test_energy_security_index(self):
        result = self.em.energy_security_index()
        assert "overall_security_index" in result
        assert "risk_level" in result
        assert "components" in result
        assert result["risk_level"] in ["low", "moderate", "high"]
        assert "supply_diversity" in result["components"]

    def test_energy_security_high_renewables(self):
        state = EnergyState(renewable_share_pct=50, grid_stability=0.9, strategic_reserve_days=180)
        result = self.em.energy_security_index(state)
        assert result["overall_security_index"] > 0.5

    def test_energy_security_low_renewables(self):
        state = EnergyState(renewable_share_pct=5, grid_stability=0.3, strategic_reserve_days=30)
        result = self.em.energy_security_index(state)
        assert result["overall_security_index"] < 0.6

    def test_security_vulnerabilities(self):
        state = EnergyState(renewable_share_pct=5, strategic_reserve_days=30, coal_share_pct=50, grid_stability=0.3)
        result = self.em.energy_security_index(state)
        assert len(result["vulnerabilities"]) >= 2
        assert len(result["recommendations"]) >= 2

    def test_carbon_budget_tracker(self):
        result = self.em.carbon_budget_tracker()
        assert "remaining_budget_gtco2" in result
        assert "years_at_current_rate" in result
        assert "budget_deadline_year" in result
        assert result["budget_deadline_year"] > 2025

    def test_carbon_budget_scenarios(self):
        result = self.em.carbon_budget_tracker(budget_gtco2=500)
        assert "scenarios" in result
        assert "1.5c" in result["scenarios"]
        assert "2.0c" in result["scenarios"]
        assert result["scenarios"]["1.5c"]["years_left"] < result["scenarios"]["2.0c"]["years_left"]

    def test_price_volatility(self):
        result = self.em.price_volatility_analysis()
        assert "volatility" in result
        assert "oil" in result["volatility"]
        assert "natural_gas" in result["volatility"]
        assert result["volatility"]["oil"]["annualized_volatility"] > 0
        assert result["volatility"]["natural_gas"]["annualized_volatility"] > 0
        assert "volatility_level" in result["volatility"]["oil"]
        assert "volatility_index" in result["volatility"]["oil"]

    def test_energy_mix_co2_decreasing(self):
        result = self.em.energy_mix_forecast(years=10, scenario="net_zero_2050")
        first_co2 = result["yearly_forecast"][0]["co2_intensity"]
        last_co2 = result["yearly_forecast"][-1]["co2_intensity"]
        assert last_co2 < first_co2


class TestCommoditiesState:
    def test_defaults(self):
        state = CommoditiesState()
        assert state.gold_price == 1900.0
        assert state.copper_price == 8500.0
        assert state.supply_chain_stress == 0.3


class TestCommoditiesModel:
    def setup_method(self):
        self.cm = CommoditiesModel()

    def test_price_forecast_gold(self):
        result = self.cm.price_forecast("gold")
        assert "current_price" in result
        assert "percentiles" in result
        assert result["commodity"] == "gold"
        assert result["percentiles"]["p50"] > 0

    def test_price_forecast_copper(self):
        result = self.cm.price_forecast("copper")
        assert result["commodity"] == "copper"
        assert result["risk_metrics"]["var_95_daily"] > 0

    def test_price_forecast_confidence_levels(self):
        low = self.cm.price_forecast("gold", confidence="low")
        high = self.cm.price_forecast("gold", confidence="high")
        assert low["percentiles"]["p10"] > 0
        assert high["percentiles"]["p90"] > 0

    def test_price_forecast_with_state(self):
        state = CommoditiesState(gold_price=2000, geopolitical_risk=0.7)
        result = self.cm.price_forecast("gold", state, days=126)
        assert result["expected_return_annualized"] != 0

    def test_supply_demand_copper(self):
        result = self.cm.supply_demand_analysis("copper")
        assert "supply" in result
        assert "demand" in result
        assert isinstance(result["supply"], dict)
        assert isinstance(result["demand"], dict)
        assert "deficit_pct" in result
        assert "supply_stress_index" in result
        assert result["commodity"] == "copper"

    def test_supply_demand_gold(self):
        result = self.cm.supply_demand_analysis("gold")
        assert result["commodity"] == "gold"
        assert isinstance(result["supply"], dict)
        assert result["supply"]["production_level"] > 0
        assert isinstance(result["demand"], dict)
        assert result["demand"]["consumption_growth"] >= 0

    def test_supply_demand_with_stress(self):
        state = CommoditiesState(supply_chain_stress=0.8, geopolitical_risk=0.9)
        result = self.cm.supply_demand_analysis("lithium", state)
        assert result["supply_stress_index"] > 0.3
        assert result["stress_level"] in ["low", "moderate", "high"]

    def test_correlation_matrix(self):
        result = self.cm.correlation_matrix()
        assert "commodities" in result
        assert "correlation_matrix" in result
        assert len(result["commodities"]) == 8
        assert "safe_haven_commodities" in result
        assert "growth_sensitive" in result
        assert "tech_metal" in result

    def test_correlation_gold_self(self):
        result = self.cm.correlation_matrix()
        assert result["correlation_matrix"]["gold"]["gold"] == 1.0

    def test_correlation_copper_oil(self):
        result = self.cm.correlation_matrix()
        assert result["correlation_matrix"]["copper"]["oil"] > 0

    def test_critical_minerals_outlook(self):
        result = self.cm.critical_minerals_outlook(years=10)
        assert "minerals" in result
        assert isinstance(result["minerals"], dict)
        assert len(result["minerals"]) == 6
        assert "top_critical" in result
        assert result["top_critical"][0] in ["Lithium", "Rare Earths", "Cobalt"]

    def test_critical_minerals_scarcity(self):
        result = self.cm.critical_minerals_outlook(years=10)
        for key, m in result["minerals"].items():
            assert "scarcity_score" in m
            assert "risk_level" in m
            assert m["risk_level"] in ["low", "moderate", "high", "critical"]

    def test_invalid_commodity_returns_gold(self):
        result = self.cm.price_forecast("unknown_commodity")
        assert result["commodity"] == "gold"

    def test_forecast_days(self):
        result = self.cm.price_forecast("gold", days=126)
        assert result["forecast_days"] == 126

    def test_engine_integration_predict(self):
        from oracle.prediction.engine import OracleEngine
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        engine = OracleEngine()
        engine.set_energy_state(oil_price=80, renewable_share=25)
        forecast = engine.predict(date(2030, 1, 1), ConfidenceLevel.HIGH)
        domains = [p.domain.value for p in forecast.predictions]
        assert "energy" in domains

    def test_engine_integration_diagnose(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_energy_state(oil_price=80, renewable_share=25)
        engine.set_commodities_state(gold_price=2000, supply_chain_stress=0.4)
        diagnosis = engine.diagnose()
        assert "energy" in diagnosis
        assert "commodities" in diagnosis

    def test_commodity_index_default(self):
        result = self.cm.commodity_index()
        assert "index_value" in result
        assert result["index_value"] > 0
        assert result["trend"] in ["bullish", "neutral", "bearish"]
        assert "sector_breakdown" in result

    def test_commodity_index_stressed(self):
        state = CommoditiesState(gold_price=2200, supply_chain_stress=0.8, geopolitical_risk=0.7, inflation_expectation=0.05)
        result = self.cm.commodity_index(state)
        assert result["index_value"] > 100
        assert result["weighted_change_pct"] > 0

    def test_seasonality_analysis_oil(self):
        result = self.cm.seasonality_analysis("oil")
        assert "avg_monthly_returns" in result
        assert len(result["avg_monthly_returns"]) == 12
        assert "best_month" in result
        assert "worst_month" in result
        assert result["seasonality_level"] in ["strong", "moderate", "weak"]

    def test_seasonality_analysis_gold(self):
        result = self.cm.seasonality_analysis("gold", years_back=3)
        assert result["commodity"] == "gold"
        assert len(result["avg_monthly_returns"]) == 12

    def test_seasonality_analysis_invalid_commodity(self):
        result = self.cm.seasonality_analysis("diamonds")
        assert result["commodity"] == "oil"