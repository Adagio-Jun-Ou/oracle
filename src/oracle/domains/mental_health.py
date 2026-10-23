from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MentalHealthState:
    depression_prevalence: float = 0.05
    anxiety_prevalence: float = 0.04
    suicide_rate_per_100k: float = 10.5
    treatment_gap_pct: float = 0.55
    mental_health_budget_pct: float = 0.02
    social_isolation_index: float = 0.45
    substance_abuse_rate: float = 0.08
    youth_mental_health_index: float = 0.45
    workplace_stress_index: float = 0.55
    digital_wellbeing_index: float = 0.40
    crisis_hotline_volume: float = 0.5
    stigma_index: float = 0.50


class MentalHealthModel:

    def __init__(self):
        self.name = "mental_health_model"

    def set_state(self, **kwargs) -> "MentalHealthModel":
        current = MentalHealthState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, float(v))
        self.state = current
        return self

    def assess_population_health(self, state: Optional[MentalHealthState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", MentalHealthState())
        burden = (
            state.depression_prevalence * 0.30
            + state.anxiety_prevalence * 0.25
            + state.suicide_rate_per_100k / 50.0 * 0.20
            + state.social_isolation_index * 0.15
            + state.substance_abuse_rate * 0.10
        )
        burden = min(1.0, burden)
        if burden > 0.6:
            burden_level = "critical"
        elif burden > 0.4:
            burden_level = "high"
        elif burden > 0.25:
            burden_level = "moderate"
        else:
            burden_level = "low"
        treatment_access = 1.0 - state.treatment_gap_pct
        system_capacity = (
            treatment_access * 0.35
            + (1.0 - state.stigma_index) * 0.25
            + state.mental_health_budget_pct / 0.05 * 0.20
            + (1.0 - state.social_isolation_index) * 0.20
        )
        system_capacity = min(1.0, max(0.0, system_capacity))
        if system_capacity > 0.7:
            capacity_level = "strong"
        elif system_capacity > 0.5:
            capacity_level = "moderate"
        elif system_capacity > 0.3:
            capacity_level = "limited"
        else:
            capacity_level = "inadequate"
        youth_risk = 1.0 - state.youth_mental_health_index
        workplace_impact = state.workplace_stress_index * 0.6 + (1.0 - state.digital_wellbeing_index) * 0.4
        digital_risk = 1.0 - state.digital_wellbeing_index
        risk_factors = [
            {"factor": "social_isolation", "severity": round(float(state.social_isolation_index), 3)},
            {"factor": "workplace_stress", "severity": round(float(state.workplace_stress_index), 3)},
            {"factor": "substance_abuse", "severity": round(float(state.substance_abuse_rate), 3)},
            {"factor": "digital_overload", "severity": round(float(digital_risk), 3)},
            {"factor": "treatment_gap", "severity": round(float(state.treatment_gap_pct), 3)},
            {"factor": "stigma", "severity": round(float(state.stigma_index), 3)},
        ]
        risk_factors.sort(key=lambda x: x["severity"], reverse=True)
        recommendations = []
        if state.treatment_gap_pct > 0.5:
            recommendations.append("Expand community-based mental health services and telehealth access")
        if state.stigma_index > 0.4:
            recommendations.append("Launch anti-stigma campaigns and normalize mental health conversations")
        if state.social_isolation_index > 0.4:
            recommendations.append("Invest in social connection programs and community spaces")
        if state.youth_mental_health_index < 0.5:
            recommendations.append("Scale school-based mental health screening and early intervention programs")
        if state.workplace_stress_index > 0.5:
            recommendations.append("Implement workplace mental health standards and burnout prevention")
        if digital_risk > 0.4:
            recommendations.append("Develop digital wellbeing guidelines and screen time awareness programs")
        if state.substance_abuse_rate > 0.07:
            recommendations.append("Integrate substance abuse treatment with mental health services")
        return sanitize_nan({
            "overall_burden": round(float(burden), 4),
            "burden_level": burden_level,
            "system_capacity": round(float(system_capacity), 4),
            "capacity_level": capacity_level,
            "treatment_access_rate": round(float(treatment_access), 4),
            "youth_mental_health_risk": round(float(youth_risk), 4),
            "workplace_impact_score": round(float(workplace_impact), 4),
            "digital_wellbeing_risk": round(float(digital_risk), 4),
            "population_affected_millions": round(float((state.depression_prevalence + state.anxiety_prevalence) * 8000), 1),
            "risk_factors": risk_factors[:5],
            "recommendations": recommendations[:6],
        })

    def simulate_crisis_scenario(self, scenario: str = "gradual_deterioration", state: Optional[MentalHealthState] = None, years: int = 5) -> dict:
        if state is None:
            state = getattr(self, "state", MentalHealthState())
        scenarios = {
            "gradual_deterioration": {"depression_growth": 1.03, "anxiety_growth": 1.04, "isolation_growth": 1.02, "budget_change": 0.99, "recession_impact": 0.0},
            "economic_recession": {"depression_growth": 1.08, "anxiety_growth": 1.10, "isolation_growth": 1.05, "budget_change": 0.95, "recession_impact": 0.15},
            "pandemic_aftermath": {"depression_growth": 1.15, "anxiety_growth": 1.12, "isolation_growth": 1.08, "budget_change": 1.05, "recession_impact": 0.10},
            "social_media_crisis": {"depression_growth": 1.06, "anxiety_growth": 1.08, "isolation_growth": 1.06, "budget_change": 1.01, "recession_impact": 0.0},
            "youth_crisis": {"depression_growth": 1.10, "anxiety_growth": 1.12, "isolation_growth": 1.04, "budget_change": 1.02, "recession_impact": 0.05},
        }
        params = scenarios.get(scenario, scenarios["gradual_deterioration"])
        timeline = []
        depression = state.depression_prevalence
        anxiety = state.anxiety_prevalence
        isolation = state.social_isolation_index
        budget = state.mental_health_budget_pct
        suicide = state.suicide_rate_per_100k
        for year in range(years):
            depression *= params["depression_growth"]
            anxiety *= params["anxiety_growth"]
            isolation *= params["isolation_growth"]
            budget *= params["budget_change"]
            suicide += params["recession_impact"] * 1.5 + (depression - 0.05) * 20
            timeline.append({
                "year": year + 1,
                "depression_prevalence": round(float(min(0.30, depression)), 4),
                "anxiety_prevalence": round(float(min(0.25, anxiety)), 4),
                "suicide_rate_per_100k": round(float(suicide), 1),
                "social_isolation_index": round(float(min(1.0, isolation)), 4),
                "mental_health_budget_pct": round(float(max(0.005, budget)), 4),
            })
        return sanitize_nan({
            "scenario": scenario,
            "final_depression_prevalence": round(float(min(0.30, depression)), 4),
            "final_anxiety_prevalence": round(float(min(0.25, anxiety)), 4),
            "final_suicide_rate": round(float(suicide), 1),
            "treatment_gap_trajectory": "widening" if params["budget_change"] < 1.0 else "narrowing",
            "population_impact_increase_pct": round(float((depression / max(0.01, state.depression_prevalence) - 1) * 100), 1),
            "timeline": timeline,
        })

    def predict(self, years_ahead: int = 10, state: Optional[MentalHealthState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", MentalHealthState())
        future_depression = min(0.25, state.depression_prevalence * (1.01 ** years_ahead) + 0.005 * years_ahead)
        future_anxiety = min(0.20, state.anxiety_prevalence * (1.015 ** years_ahead) + 0.003 * years_ahead)
        future_suicide = state.suicide_rate_per_100k * (1.005 ** years_ahead)
        future_treatment_gap = max(0.2, state.treatment_gap_pct - 0.01 * years_ahead)
        future_stigma = max(0.15, state.stigma_index - 0.015 * years_ahead)
        future_budget = min(0.10, state.mental_health_budget_pct + 0.002 * years_ahead)
        future_isolation = min(0.7, state.social_isolation_index + 0.005 * years_ahead)
        future_youth = max(0.3, state.youth_mental_health_index - 0.005 * years_ahead)
        future_digital = min(0.7, state.digital_wellbeing_index * 0.98 ** years_ahead + 0.02 * years_ahead)
        if future_treatment_gap < 0.3:
            trajectory = "improving"
        elif future_treatment_gap < 0.5:
            trajectory = "stabilizing"
        else:
            trajectory = "worsening"
        return sanitize_nan({
            "future_depression_prevalence": round(float(future_depression), 4),
            "future_anxiety_prevalence": round(float(future_anxiety), 4),
            "future_suicide_rate": round(float(future_suicide), 1),
            "future_treatment_gap": round(float(future_treatment_gap), 4),
            "future_stigma_index": round(float(future_stigma), 4),
            "future_budget_pct": round(float(future_budget), 4),
            "future_social_isolation": round(float(future_isolation), 4),
            "future_youth_index": round(float(future_youth), 4),
            "future_digital_wellbeing": round(float(future_digital), 4),
            "trajectory": trajectory,
            "affected_population_millions": round(float((future_depression + future_anxiety) * 8200), 1),
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