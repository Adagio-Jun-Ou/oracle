from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import math


@dataclass
class FoodState:
    food_production_index: float = 0.70
    food_price_volatility: float = 0.15
    crop_yield_variability: float = 0.20
    fertilizer_dependency: float = 0.60
    water_for_agriculture: float = 0.70
    arable_land_pct: float = 0.11
    supply_chain_disruption: float = 0.25
    climate_impact_on_crops: float = 0.20
    population_growth_pressure: float = 0.30
    dietary_diversity: float = 0.55
    food_waste_pct: float = 0.30
    reserve_stock_months: float = 3.0
    trade_dependency: float = 0.25
    biofuel_competition: float = 0.10


class FoodSecurityModel:
    def __init__(self):
        self.state: Optional[FoodState] = None

    def set_state(self, **kwargs) -> "FoodSecurityModel":
        self.state = FoodState(**kwargs)
        return self

    def assess_security(self, state: Optional[FoodState] = None) -> Dict[str, Any]:
        s = state or self.state or FoodState()
        availability = (
            s.food_production_index * 0.30
            + (1.0 - s.food_price_volatility) * 0.15
            + (1.0 - s.crop_yield_variability) * 0.15
            + s.arable_land_pct * 0.10
            + (s.reserve_stock_months / 12.0) * 0.15
            + (1.0 - s.trade_dependency) * 0.15
        )
        access = (
            s.dietary_diversity * 0.30
            + (1.0 - s.population_growth_pressure) * 0.25
            + (1.0 - s.food_waste_pct) * 0.25
            + s.food_production_index * 0.20
        )
        stability = (
            (1.0 - s.food_price_volatility) * 0.25
            + (1.0 - s.supply_chain_disruption) * 0.25
            + (s.reserve_stock_months / 6.0) * 0.25
            + (1.0 - s.climate_impact_on_crops) * 0.25
        )
        sustainability = (
            s.water_for_agriculture * 0.25
            + (1.0 - s.fertilizer_dependency) * 0.25
            + (1.0 - s.biofuel_competition) * 0.25
            + (1.0 - s.food_waste_pct) * 0.25
        )
        overall_security = availability * 0.30 + access * 0.25 + stability * 0.25 + sustainability * 0.20
        overall_risk = 1.0 - overall_security
        if overall_risk > 0.6:
            risk_level = "critical"
        elif overall_risk > 0.4:
            risk_level = "high"
        elif overall_risk > 0.25:
            risk_level = "moderate"
        elif overall_risk > 0.1:
            risk_level = "low"
        else:
            risk_level = "secure"
        vulnerabilities = []
        if s.food_price_volatility > 0.3:
            vulnerabilities.append({"area": "price_volatility", "severity": "high", "description": "Extreme food price volatility destabilizing access"})
        if s.crop_yield_variability > 0.3:
            vulnerabilities.append({"area": "crop_instability", "severity": "high", "description": "High crop yield variability from climate and environmental factors"})
        if s.fertilizer_dependency > 0.6:
            vulnerabilities.append({"area": "fertilizer_dependency", "severity": "moderate", "description": "Heavy reliance on imported/synthetic fertilizers"})
        if s.water_for_agriculture < 0.5:
            vulnerabilities.append({"area": "water_scarcity", "severity": "high", "description": "Insufficient water resources for agricultural production"})
        if s.supply_chain_disruption > 0.4:
            vulnerabilities.append({"area": "supply_chain", "severity": "moderate", "description": "Food supply chain vulnerable to trade and logistics disruptions"})
        if s.reserve_stock_months < 3:
            vulnerabilities.append({"area": "low_reserves", "severity": "high", "description": "Insufficient food reserves for crisis resilience"})
        if s.food_waste_pct > 0.3:
            vulnerabilities.append({"area": "food_waste", "severity": "moderate", "description": "Significant food waste reducing effective supply"})
        if s.biofuel_competition > 0.2:
            vulnerabilities.append({"area": "biofuel_competition", "severity": "low", "description": "Agricultural land diverted from food to fuel production"})
        recommendations = []
        if overall_risk > 0.4:
            recommendations.append("Diversify food import sources and establish strategic reserve stockpiles")
        if s.crop_yield_variability > 0.2:
            recommendations.append("Invest in climate-resilient crop varieties and precision agriculture technology")
        if s.food_waste_pct > 0.25:
            recommendations.append("Implement food waste reduction programs: cold chain, improved storage, consumer education")
        if s.fertilizer_dependency > 0.5:
            recommendations.append("Develop organic/regenerative farming practices and local fertilizer production")
        if s.water_for_agriculture < 0.6:
            recommendations.append("Deploy water-efficient irrigation systems and drought-resistant crop varieties")
        if s.food_price_volatility > 0.2:
            recommendations.append("Strengthen social safety nets and price stabilization mechanisms")
        if s.reserve_stock_months < 4:
            recommendations.append("Build strategic grain reserves to minimum 6-month supply coverage")
        return {
            "overall_food_security": float(min(1.0, overall_security)),
            "overall_risk": float(overall_risk),
            "risk_level": risk_level,
            "availability_score": float(availability),
            "access_score": float(access),
            "stability_score": float(stability),
            "sustainability_score": float(sustainability),
            "vulnerabilities": vulnerabilities,
            "n_vulnerabilities": len(vulnerabilities),
            "recommendations": recommendations,
        }

    def crop_model(
        self,
        crop: str = "wheat",
        years_ahead: int = 10,
        state: Optional[FoodState] = None,
    ) -> Dict[str, Any]:
        s = state or self.state or FoodState()
        crop_params = {
            "wheat": {"base_yield_t_ha": 3.5, "water_need": 0.7, "fertilizer_need": 0.6, "drought_sensitivity": 0.7, "temp_optimal": 18},
            "rice": {"base_yield_t_ha": 4.5, "water_need": 0.9, "fertilizer_need": 0.7, "drought_sensitivity": 0.8, "temp_optimal": 25},
            "corn": {"base_yield_t_ha": 5.5, "water_need": 0.75, "fertilizer_need": 0.65, "drought_sensitivity": 0.6, "temp_optimal": 22},
            "soybean": {"base_yield_t_ha": 2.5, "water_need": 0.6, "fertilizer_need": 0.3, "drought_sensitivity": 0.5, "temp_optimal": 24},
        }
        params = crop_params.get(crop, crop_params["wheat"])
        climate_impact_yield = 1.0 - s.climate_impact_on_crops * params["drought_sensitivity"] * 0.3
        water_impact_yield = min(1.0, s.water_for_agriculture / max(0.01, params["water_need"]))
        fertilizer_impact_yield = min(1.0, s.fertilizer_dependency * 0.7 + (1.0 - s.fertilizer_dependency) * 1.0)
        base_yield = params["base_yield_t_ha"]
        current_yield = base_yield * climate_impact_yield * water_impact_yield * fertilizer_impact_yield
        future_climate_decline = s.climate_impact_on_crops * years_ahead * 0.01
        future_water_decline = max(0.5, (1.0 - s.population_growth_pressure * 0.02)) if s.water_for_agriculture < 0.6 else 1.0
        technology_gain = min(1.15, 1.0 + years_ahead * 0.005)
        future_yield = base_yield * max(0.3, climate_impact_yield - future_climate_decline) * min(1.0, water_impact_yield * future_water_decline) * technology_gain
        yield_change_pct = (future_yield - current_yield) / max(0.01, current_yield) * 100
        water_required = params["water_need"] * base_yield * (1.0 + years_ahead * 0.01)
        price_pressure = min(1.0, s.food_price_volatility + s.population_growth_pressure * 0.3 + (1.0 - current_yield / base_yield) * 0.2)
        return {
            "crop": crop,
            "years_ahead": years_ahead,
            "current_yield_t_ha": round(current_yield, 2),
            "projected_yield_t_ha": round(future_yield, 2),
            "yield_change_pct": round(yield_change_pct, 1),
            "climate_impact_on_yield": round(climate_impact_yield, 3),
            "water_sufficiency": round(water_impact_yield, 3),
            "fertilizer_impact": round(min(1.0, fertilizer_impact_yield), 3),
            "water_requirement_index": round(water_required, 2),
            "price_pressure": round(price_pressure, 3),
            "drought_sensitivity": round(params["drought_sensitivity"], 3),
        }

    def predict(self, years_ahead: int = 10, state: Optional[FoodState] = None) -> Dict[str, Any]:
        s = state or self.state or FoodState()
        production_decline = max(0.0, 1.0 - s.food_production_index * (1.0 - s.climate_impact_on_crops * 0.3) * (1.0 - s.crop_yield_variability * 0.2) - years_ahead * 0.005)
        price_vol_future = min(1.0, s.food_price_volatility + s.population_growth_pressure * years_ahead * 0.01 + s.climate_impact_on_crops * 0.1)
        hunger_increase = min(1.0, (1.0 - s.food_production_index) * 0.4 + s.population_growth_pressure * 0.3 + price_vol_future * 0.3)
        future_reserves = max(0.5, s.reserve_stock_months - years_ahead * 0.05 + s.food_production_index * years_ahead * 0.03)
        future_waste = max(0.1, s.food_waste_pct - years_ahead * 0.003)
        future_diversity = min(1.0, s.dietary_diversity + years_ahead * 0.005 * s.food_production_index)
        future_security = (
            s.food_production_index * 0.30
            + (1.0 - price_vol_future) * 0.20
            + min(1.0, future_reserves / 6.0) * 0.20
            + future_diversity * 0.15
            + (1.0 - future_waste) * 0.15
        )
        future_risk = 1.0 - min(1.0, future_security)
        return {
            "years_ahead": years_ahead,
            "future_food_security": float(min(1.0, future_security)),
            "future_risk": float(future_risk),
            "production_outlook": float(max(0, production_decline)),
            "price_volatility_projected": float(price_vol_future),
            "hunger_risk_projected": float(hunger_increase),
            "projected_reserves_months": float(min(24, max(0.5, future_reserves))),
            "food_waste_projected": float(future_waste),
            "dietary_diversity_projected": float(future_diversity),
        }