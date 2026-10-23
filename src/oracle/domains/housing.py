from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class HousingState:
    median_home_price: float = 350000.0
    median_income: float = 70000.0
    mortgage_rate: float = 0.065
    vacancy_rate: float = 0.05
    construction_permits_index: float = 0.55
    rent_to_income_ratio: float = 0.30
    homelessness_per_100k: float = 17.0
    housing_affordability_index: float = 0.45
    urbanization_rate: float = 0.83
    zoning_restrictiveness: float = 0.6
    housing_supply_elasticity: float = 0.4


class HousingModel:

    def __init__(self):
        self.name = "housing_model"

    def set_state(self, **kwargs) -> "HousingModel":
        current = HousingState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, float(v))
        self.state = current
        return self

    def assess_market(self, state: Optional[HousingState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", HousingState())
        price_to_income = state.median_home_price / max(1.0, state.median_income)
        affordability = 1.0 - min(1.0, (price_to_income - 3.0) / 7.0)
        monthly_payment = state.median_home_price * (state.mortgage_rate / 12) * (1 + state.mortgage_rate / 12) ** 360 / ((1 + state.mortgage_rate / 12) ** 360 - 1)
        monthly_income = state.median_income / 12
        payment_burden = monthly_payment / max(1.0, monthly_income)
        supply_pressure = 1.0 - state.vacancy_rate
        construction_urgency = max(0.0, supply_pressure - state.housing_supply_elasticity)
        rental_stress = state.rent_to_income_ratio
        if rental_stress > 0.4:
            rental_burden = "severe"
        elif rental_stress > 0.3:
            rental_burden = "moderate"
        elif rental_stress > 0.25:
            rental_burden = "mild"
        else:
            rental_burden = "manageable"
        homelessness_impact = state.homelessness_per_100k / 50.0
        bubble_risk = 0.0
        if price_to_income > 6:
            bubble_risk += 0.3
        elif price_to_income > 5:
            bubble_risk += 0.2
        elif price_to_income > 4:
            bubble_risk += 0.1
        if state.mortgage_rate < 0.04:
            bubble_risk += 0.2
        if state.vacancy_rate < 0.03:
            bubble_risk += 0.15
        if state.zoning_restrictiveness > 0.6:
            bubble_risk += 0.1
        bubble_risk = min(1.0, bubble_risk)
        market_health_score = (
            affordability * 0.30
            + (1.0 - bubble_risk) * 0.25
            + (1.0 - rental_stress) * 0.20
            + state.housing_supply_elasticity * 0.15
            + state.construction_permits_index * 0.10
        )
        market_health_score = min(1.0, max(0.0, market_health_score))
        if market_health_score > 0.7:
            health_label = "healthy"
        elif market_health_score > 0.5:
            health_label = "moderate"
        elif market_health_score > 0.3:
            health_label = "stressed"
        else:
            health_label = "crisis"
        recommendations = []
        if affordability < 0.4:
            recommendations.append("Implement inclusionary zoning and affordable housing mandates")
        if bubble_risk > 0.4:
            recommendations.append("Consider macroprudential measures: LTV caps, stress testing, foreign buyer taxes")
        if rental_stress > 0.35:
            recommendations.append("Strengthen tenant protections and rent stabilization policies")
        if state.homelessness_per_100k > 20:
            recommendations.append("Expand Housing First programs and supportive housing inventory")
        if state.vacancy_rate < 0.03:
            recommendations.append("Accelerate housing construction through zoning reform and permitting streamlining")
        if state.zoning_restrictiveness > 0.5:
            recommendations.append("Reform single-family zoning to allow missing-middle housing types")
        return sanitize_nan({
            "market_health_score": round(float(market_health_score), 4),
            "health_label": health_label,
            "affordability_index": round(float(affordability), 4),
            "price_to_income_ratio": round(float(price_to_income), 2),
            "monthly_payment_burden": round(float(payment_burden), 4),
            "rental_burden": rental_burden,
            "bubble_risk": round(float(bubble_risk), 4),
            "housing_supply_pressure": round(float(supply_pressure), 4),
            "construction_urgecy": round(float(construction_urgency), 4),
            "homelessness_impact_score": round(float(homelessness_impact), 4),
            "recommendations": recommendations[:6],
        })

    def simulate_housing_scenario(self, scenario: str = "gradual", state: Optional[HousingState] = None, years: int = 5) -> dict:
        if state is None:
            state = getattr(self, "state", HousingState())
        scenarios = {
            "gradual": {"price_growth": 1.04, "income_growth": 1.03, "rate_change": 0.00, "construction_growth": 1.02, "homelessness_growth": 1.02},
            "boom": {"price_growth": 1.10, "income_growth": 1.04, "rate_change": -0.01, "construction_growth": 1.08, "homelessness_growth": 1.01},
            "crash": {"price_growth": 0.92, "income_growth": 0.98, "rate_change": 0.02, "construction_growth": 0.85, "homelessness_growth": 1.15},
            "affordability_crisis": {"price_growth": 1.08, "income_growth": 1.02, "rate_change": 0.01, "construction_growth": 1.01, "homelessness_growth": 1.10},
            "reform": {"price_growth": 1.02, "income_growth": 1.03, "rate_change": -0.005, "construction_growth": 1.15, "homelessness_growth": 0.95},
        }
        params = scenarios.get(scenario, scenarios["gradual"])
        timeline = []
        price = state.median_home_price
        income = state.median_income
        rate = state.mortgage_rate
        permits = state.construction_permits_index
        homeless = state.homelessness_per_100k
        for year in range(years):
            price *= params["price_growth"]
            income *= params["income_growth"]
            rate += params["rate_change"]
            permits *= params["construction_growth"] - 1.0
            permits = min(1.0, max(0.1, permits + 0.03))
            homeless *= params["homelessness_growth"]
            pti = price / max(1.0, income)
            mp = price * (rate / 12) * (1 + rate / 12) ** 360 / ((1 + rate / 12) ** 360 - 1) if rate > 0 else price / 360
            mi = income / 12
            burden = mp / max(1.0, mi)
            timeline.append({
                "year": year + 1,
                "median_price": round(float(price), 0),
                "median_income": round(float(income), 0),
                "mortgage_rate": round(float(rate), 4),
                "price_to_income": round(float(pti), 2),
                "payment_burden": round(float(burden), 4),
                "homelessness_per_100k": round(float(homeless), 1),
            })
        return sanitize_nan({
            "scenario": scenario,
            "final_median_price": round(float(price), 0),
            "final_price_to_income": round(float(price / max(1.0, income)), 2),
            "final_mortgage_rate": round(float(rate), 4),
            "final_homelessness_per_100k": round(float(homeless), 1),
            "affordability_trajectory": "worsening" if params["price_growth"] > params["income_growth"] else "improving",
            "timeline": timeline,
        })

    def predict(self, years_ahead: int = 5, state: Optional[HousingState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", HousingState())
        future_price = state.median_home_price * (1.04 ** years_ahead)
        future_income = state.median_income * (1.03 ** years_ahead)
        future_rate = state.mortgage_rate + 0.002 * years_ahead
        future_vacancy = max(0.02, state.vacancy_rate + 0.001 * years_ahead * (1 - state.housing_supply_elasticity))
        future_homeless = state.homelessness_per_100k * (1.02 ** years_ahead)
        future_pti = future_price / max(1.0, future_income)
        future_affordability = 1.0 - min(1.0, (future_pti - 3.0) / 7.0)
        future_bubble = min(1.0, 0.1 * years_ahead / 5.0 + (future_pti - 4.0) * 0.1)
        future_bubble = max(0.0, future_bubble)
        if future_affordability < 0.3:
            outlook = "severe_crisis"
        elif future_affordability < 0.5:
            outlook = "deteriorating"
        elif future_affordability < 0.7:
            outlook = "stable"
        else:
            outlook = "improving"
        return sanitize_nan({
            "future_median_price": round(float(future_price), 0),
            "future_median_income": round(float(future_income), 0),
            "future_mortgage_rate": round(float(min(0.12, future_rate)), 4),
            "future_price_to_income": round(float(future_pti), 2),
            "future_affordability": round(float(future_affordability), 4),
            "future_bubble_risk": round(float(future_bubble), 4),
            "future_vacancy_rate": round(float(future_vacancy), 4),
            "future_homelessness_per_100k": round(float(future_homeless), 1),
            "market_outlook": outlook,
            "affordability_trend": "worsening" if future_pti > state.median_home_price / max(1.0, state.median_income) else "improving",
        })


def sanitize_nan(obj):
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj