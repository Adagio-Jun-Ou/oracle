from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, date
import numpy as np


def sanitize_nan(obj):
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or abs(obj) == float('inf'):
            return None
        return obj
    if isinstance(obj, (np.floating, np.float64)):
        v = float(obj)
        if v != v or abs(v) == float('inf'):
            return None
        return v
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@dataclass
class PersonProfile:
    name: str = ""
    age: int = 35
    biological_sex: str = "male"
    ethnicity: str = "general"
    height_cm: float = 175.0
    weight_kg: float = 75.0
    education_years: int = 16
    occupation: str = "professional"
    annual_income: float = 75000.0
    net_worth: float = 150000.0
    debt_total: float = 50000.0
    marital_status: str = "married"
    n_children: int = 0
    children_ages: list = field(default_factory=list)
    exercise_minutes_weekly: int = 90
    sleep_hours_avg: float = 7.0
    diet_quality: float = 0.5
    smoking_status: str = "never"
    alcohol_drinks_weekly: int = 3
    chronic_conditions: list = field(default_factory=list)
    medications: list = field(default_factory=list)
    family_history: dict = field(default_factory=dict)
    stress_level: float = 0.4
    relationship_quality: float = 0.7
    job_satisfaction: float = 0.6
    life_satisfaction: float = 0.6
    social_network_size: int = 12
    close_friends: int = 3
    community_involvement: float = 0.3
    purpose_score: float = 0.5
    resilience_score: float = 0.6
    risk_tolerance: float = 0.5
    career_growth_rate: float = 0.03
    savings_rate: float = 0.15
    investment_risk_profile: str = "moderate"


@dataclass
class FamilyMemberProfile:
    relation: str = "spouse"
    name: str = ""
    age: int = 33
    biological_sex: str = "female"
    health_status: float = 0.8
    income_contribution: float = 0.5
    emotional_support: float = 0.7
    caregiving_burden: float = 0.2
    shared_activities: float = 0.5
    conflict_level: float = 0.2
    dependency_level: float = 0.3


@dataclass
class SocialContactProfile:
    contact_type: str = "friend"
    name: str = ""
    closeness: float = 0.6
    frequency_contact_monthly: float = 4.0
    influence_score: float = 0.4
    emotional_valence: float = 0.7
    professional_value: float = 0.3
    health_influence: float = 0.3
    age: int = 35
    reliability: float = 0.6


CAREER_STAGES = {
    "early_career": {"age_range": (22, 30), "growth_rate": 0.08, "volatility": 0.3, "peak_probability": 0.05},
    "career_building": {"age_range": (30, 40), "growth_rate": 0.05, "volatility": 0.2, "peak_probability": 0.15},
    "mid_career": {"age_range": (40, 50), "growth_rate": 0.03, "volatility": 0.15, "peak_probability": 0.30},
    "senior": {"age_range": (50, 60), "growth_rate": 0.01, "volatility": 0.1, "peak_probability": 0.25},
    "pre_retirement": {"age_range": (60, 67), "growth_rate": -0.02, "volatility": 0.08, "peak_probability": 0.10},
    "retirement": {"age_range": (67, 100), "growth_rate": -0.04, "volatility": 0.05, "peak_probability": 0.0},
}

HEALTH_MILESTONES = [
    {"age": 30, "event": "Peak bone mass", "type": "physiological", "impact": 0.05},
    {"age": 35, "event": "Fertility decline begins", "type": "reproductive", "impact": 0.08},
    {"age": 40, "event": "Vision presbyopia onset", "type": "sensory", "impact": 0.03},
    {"age": 45, "event": "Cancer screening threshold", "type": "medical", "impact": 0.06},
    {"age": 50, "event": "Cardiovascular risk acceleration", "type": "cardiovascular", "impact": 0.10},
    {"age": 55, "event": "Menopause / testosterone decline", "type": "hormonal", "impact": 0.08},
    {"age": 60, "event": "Osteoarthritis onset risk", "type": "musculoskeletal", "impact": 0.07},
    {"age": 65, "event": "Cognitive decline detectable", "type": "neurological", "impact": 0.10},
    {"age": 70, "event": "Fall risk threshold", "type": "musculoskeletal", "impact": 0.08},
    {"age": 75, "event": "Multi-morbidity acceleration", "type": "medical", "impact": 0.12},
    {"age": 80, "event": "Disability onset probability >50%", "type": "functional", "impact": 0.15},
    {"age": 85, "event": "Frailty syndrome threshold", "type": "geriatric", "impact": 0.20},
]

LIFE_EVENT_TEMPLATES = {
    "career_advancement": {
        "name": "Career Advancement",
        "base_probability_annual": 0.08,
        "age_modifier": lambda a: 1.5 if 30 <= a <= 45 else 0.8 if 45 < a <= 55 else 0.3,
        "impact": {"income": 0.15, "satisfaction": 0.10, "stress": 0.08, "health": -0.02},
        "cascade": ["increased_savings", "housing_upgrade", "status_boost"],
    },
    "job_loss": {
        "name": "Job Loss / Layoff",
        "base_probability_annual": 0.04,
        "age_modifier": lambda a: 1.5 if a > 50 else 0.7 if a < 30 else 1.0,
        "impact": {"income": -0.40, "satisfaction": -0.30, "stress": 0.25, "health": -0.08, "relationship": -0.10},
        "cascade": ["financial_stress", "relocation", "career_shift"],
    },
    "marriage": {
        "name": "Marriage / Long-term Partnership",
        "base_probability_annual": 0.02,
        "age_modifier": lambda a: 3.0 if 25 <= a <= 35 else 1.0 if 35 < a <= 45 else 0.3,
        "impact": {"income": 0.10, "satisfaction": 0.15, "health": 0.08, "relationship": 0.25, "social": 0.15},
        "cascade": ["housing_change", "family_planning", "combined_finances"],
    },
    "divorce": {
        "name": "Divorce / Separation",
        "base_probability_annual": 0.015,
        "age_modifier": lambda a: 2.0 if 35 <= a <= 50 else 0.5,
        "impact": {"income": -0.30, "satisfaction": -0.25, "health": -0.12, "relationship": -0.30, "social": -0.10},
        "cascade": ["financial_rebuilding", "housing_change", "custody_arrangement", "social_network_shift"],
    },
    "child_birth": {
        "name": "Child Born",
        "base_probability_annual": 0.01,
        "age_modifier": lambda a: 5.0 if 28 <= a <= 35 else 2.0 if 35 < a <= 40 else 0.3,
        "impact": {"income": -0.05, "satisfaction": 0.08, "stress": 0.15, "health": -0.03, "relationship": -0.05, "purpose": 0.15},
        "cascade": ["housing_need_increase", "education_planning", "life_restructuring"],
    },
    "health_crisis": {
        "name": "Major Health Crisis",
        "base_probability_annual": 0.02,
        "age_modifier": lambda a: 0.3 if a < 40 else 1.0 if 40 <= a < 60 else 2.5 if 60 <= a < 75 else 5.0,
        "impact": {"income": -0.20, "satisfaction": -0.25, "health": -0.30, "stress": 0.20, "relationship": -0.10},
        "cascade": ["medical_costs", "lifestyle_change", "career_interruption"],
    },
    "parent_illness": {
        "name": "Parent/elder Family Member Illness",
        "base_probability_annual": 0.03,
        "age_modifier": lambda a: 0.5 if a < 40 else 2.0 if 40 <= a < 55 else 3.0 if 55 <= a else 1.5,
        "impact": {"stress": 0.15, "income": -0.10, "satisfaction": -0.10, "time": -0.15},
        "cascade": ["caregiving_burden", "family_financial_impact", "inheritance_planning"],
    },
    "financial_windfall": {
        "name": "Financial Windfall (inheritance/sale)",
        "base_probability_annual": 0.01,
        "age_modifier": lambda a: 0.5 if a < 40 else 1.5 if 40 <= a < 60 else 3.0,
        "impact": {"income": 0.30, "satisfaction": 0.10, "stress": -0.05, "net_worth": 0.25},
        "cascade": ["investment_opportunity", "debt_reduction", "lifestyle_inflation_risk"],
    },
    "relocation": {
        "name": "Major Relocation",
        "base_probability_annual": 0.03,
        "age_modifier": lambda a: 2.0 if 22 <= a <= 35 else 1.0 if 35 < a <= 50 else 0.3,
        "impact": {"social": -0.15, "income": 0.05, "stress": 0.10, "satisfaction": 0.02},
        "cascade": ["housing_change", "new_social_network", "career_opportunity"],
    },
    "retirement": {
        "name": "Retirement",
        "base_probability_annual": 0.005,
        "age_modifier": lambda a: 0.1 if a < 60 else 0.5 if 60 <= a < 65 else 5.0 if 65 <= a < 70 else 10.0,
        "impact": {"income": -0.40, "stress": -0.15, "satisfaction": 0.05, "social": -0.10, "health": 0.03, "purpose": -0.10},
        "cascade": ["lifestyle_adjustment", "social_network_shift", "health_regimen"],
    },
    "friendship_loss": {
        "name": "Close Friendship Loss (drift/death)",
        "base_probability_annual": 0.05,
        "age_modifier": lambda a: 0.8 if a < 50 else 2.0 if 50 <= a < 70 else 4.0,
        "impact": {"social": -0.10, "satisfaction": -0.05, "health": -0.03, "loneliness": 0.08},
        "cascade": ["social_network_contraction", "emotional_impact"],
    },
    "major_purchase_home": {
        "name": "Home Purchase",
        "base_probability_annual": 0.02,
        "age_modifier": lambda a: 3.0 if 28 <= a <= 40 else 1.0 if 40 < a <= 55 else 0.3,
        "impact": {"debt": 0.30, "satisfaction": 0.10, "stability": 0.15, "net_worth": 0.05},
        "cascade": ["location_commitment", "maintenance_burden", "equity_building"],
    },
    "legal_issue": {
        "name": "Legal Issue / Lawsuit",
        "base_probability_annual": 0.005,
        "age_modifier": lambda a: 1.0,
        "impact": {"income": -0.20, "stress": 0.20, "satisfaction": -0.15, "health": -0.05},
        "cascade": ["financial_drain", "reputation_impact", "time_cost"],
    },
}

FAMILY_DYNAMICS_PATTERNS = {
    "nuclear_family_stress": {
        "name": "Nuclear Family Stress Cycle",
        "trigger": "combined_financial_time_pressure",
        "probability_modifier": lambda p: 1.0 + p.stress_level + (1.0 - p.relationship_quality) + (1.0 - p.financial_safety_net()),
        "impact": {"marital_satisfaction": -0.15, "parenting_quality": -0.10, "children_outcome": -0.05},
        "description": "Financial and time pressure creates marital strain, reducing parenting quality and child outcomes",
    },
    "sandwich_generation": {
        "name": "Sandwich Generation Squeeze",
        "trigger": "simultaneous_eldercare_childcare",
        "probability_modifier": lambda p: 3.0 if 40 <= p.age <= 55 and p.n_children > 0 else 0.5,
        "impact": {"stress": 0.25, "income": -0.10, "health": -0.10, "satisfaction": -0.15, "career": -0.08},
        "description": "Caring for aging parents while raising children creates compound burden on health, finances, and career",
    },
    "intergenerational_trauma": {
        "name": "Intergenerational Pattern Repetition",
        "trigger": "family_history_of_adversity",
        "probability_modifier": lambda p: 1.5 + len(p.chronic_conditions) * 0.2 + (1.0 - p.resilience_score) * 0.5,
        "impact": {"parenting_replication": 0.25, "health_heritability": 0.15, "behavioral_pattern": 0.20},
        "description": "Without intervention, family patterns of conflict, addiction, or illness replicate across generations",
    },
    "family_financial_shock": {
        "name": "Family Financial Shock Cascade",
        "trigger": "income_loss_or_medical_bills",
        "probability_modifier": lambda p: 1.0 + (1.0 - p.savings_rate / 0.3) + p.debt_total / max(1, p.annual_income) * 0.5,
        "impact": {"housing_stability": -0.20, "children_education": -0.10, "marital_stress": 0.15, "health": -0.08},
        "description": "Financial shock cascades through housing, education, relationship, and health outcomes for entire family",
    },
    "family_health_event": {
        "name": "Family Health Event Ripple",
        "trigger": "member_diagnosis",
        "probability_modifier": lambda members: 0.1 + sum((1.0 - m.health_status) for m in members) / max(1, len(members)) * 0.5,
        "impact": {"caregiving_burden": 0.20, "financial_impact": -0.15, "emotional_impact": -0.15, "schedule_disruption": 0.15},
        "description": "Serious illness in one family member creates caregiving burden, financial stress, and emotional impact across family",
    },
}

SOCIAL_NETWORK_EFFECTS = {
    "health_contagion": {
        "name": "Health Behavior Contagion",
        "effect": "obesity, smoking, and exercise habits spread through social networks with 3 degrees of influence",
        "strength": 0.25,
        "mechanism": "Social norms and shared activities shape individual health behavior unconsciously",
    },
    "emotional_contagion": {
        "name": "Emotional Contagion",
        "effect": "Depression and anxiety propagate through close relationships at 0.3 correlation",
        "strength": 0.30,
        "mechanism": "Mirror neurons and empathic bonding create shared emotional states",
    },
    "career_network": {
        "name": "Career Network Effect",
        "effect": "70-80% of jobs are found through personal connections; network quality predicts career trajectory",
        "strength": 0.35,
        "mechanism": "Information asymmetry and trust-based referrals dominate hiring decisions",
    },
    "longevity_network": {
        "name": "Social Network Longevity Effect",
        "effect": "Each close social connection adds 0.5-2 years of life expectancy; isolation is equivalent to smoking 15 cigarettes/day",
        "strength": 0.20,
        "mechanism": "Social integration regulates HPA axis, immune function, and health behavior",
    },
    "financial_contagion": {
        "name": "Financial Behavior Contagion",
        "effect": "Spending, saving, and investment behaviors are strongly influenced by reference group",
        "strength": 0.20,
        "mechanism": "Social comparison and keeping-up effects drive financial decisions",
    },
    "cultural_transmission": {
        "name": "Cultural Value Transmission",
        "effect": "Values, beliefs, and worldviews transmit through social networks with 60-80% fidelity",
        "strength": 0.25,
        "mechanism": "Socialization and identity formation occur through network reference groups",
    },
}


class PersonalLifeModel:

    def __init__(self, seed: int = 42):
        self.name = "personal_life_model"
        self.seed = seed

    def predict_career_trajectory(self, person: PersonProfile, years: int = 20) -> dict:
        rng = np.random.RandomState(self.seed)
        current_stage = "early_career"
        for stage_name, stage_data in CAREER_STAGES.items():
            lo, hi = stage_data["age_range"]
            if lo <= person.age <= hi:
                current_stage = stage_name
                break

        stage_info = CAREER_STAGES.get(current_stage, CAREER_STAGES["mid_career"])
        base_growth = stage_info["growth_rate"]
        education_bonus = person.education_years / 20.0 * 0.02
        satisfaction_bonus = person.job_satisfaction * 0.01
        growth_rate = base_growth + education_bonus + satisfaction_bonus
        income_trajectory = [person.annual_income]
        current_income = person.annual_income

        trajectory = []
        for yr in range(1, years + 1):
            future_age = person.age + yr
            future_stage = current_stage
            for sname, sdata in CAREER_STAGES.items():
                lo, hi = sdata["age_range"]
                if lo <= future_age <= hi:
                    future_stage = sname
                    break
            fs = CAREER_STAGES.get(future_stage, stage_info)
            growth = fs["growth_rate"] + education_bonus
            shock = rng.normal(0, fs["volatility"] * current_income * 0.1)
            promotion = rng.random() < fs["peak_probability"] * 0.1
            promo_bonus = current_income * 0.15 if promotion else 0
            current_income = max(person.annual_income * 0.3, current_income * (1 + growth) + shock + promo_bonus)
            income_trajectory.append(round(current_income, 0))
            trajectory.append({
                "year": yr,
                "age": future_age,
                "projected_income": round(current_income, 0),
                "stage": future_stage,
                "promotion_probability": round(fs["peak_probability"] * 0.1, 3),
            })

        peak_income = max(income_trajectory)
        peak_year = income_trajectory.index(peak_income)

        return sanitize_nan({
            "current_stage": current_stage,
            "base_growth_rate": round(growth_rate, 4),
            "current_income": person.annual_income,
            "projected_peak_income": round(peak_income, 0),
            "years_to_peak": peak_year,
            "retirement_income_estimate": round(income_trajectory[-1] if income_trajectory else 0, 0),
            "lifetime_earnings_estimate": round(sum(income_trajectory), 0),
            "trajectory": trajectory[:10],
        })

    def predict_health_trajectory(self, person: PersonProfile, years: int = 30) -> dict:
        rng = np.random.RandomState(self.seed)

        base_le = 79.0
        sex_adj = -5.0 if person.biological_sex == "male" else 0.0
        bmi = person.weight_kg / ((person.height_cm / 100) ** 2)
        bmi_adj = -3.0 if bmi > 30 else -1.0 if bmi > 25 else 0.5 if 18.5 <= bmi <= 24.9 else -1.0
        smoke_adj = -10.0 if person.smoking_status == "current" else -3.0 if person.smoking_status == "former" else 0.0
        alcohol_adj = -2.0 if person.alcohol_drinks_weekly > 14 else -0.5 if person.alcohol_drinks_weekly > 7 else 0.0
        exercise_adj = 3.0 if person.exercise_minutes_weekly > 150 else 1.5 if person.exercise_minutes_weekly > 75 else -1.0
        sleep_adj = 1.0 if 7 <= person.sleep_hours_avg <= 9 else -2.0 if person.sleep_hours_avg < 6 else -0.5
        diet_adj = 2.0 if person.diet_quality > 0.7 else 0.0 if person.diet_quality > 0.4 else -2.0
        chronic_adj = -2.0 * len(person.chronic_conditions)
        stress_adj = -1.5 if person.stress_level > 0.6 else 0.5 if person.stress_level < 0.3 else 0.0
        social_adj = 1.5 if person.close_friends >= 5 else 0.5 if person.close_friends >= 3 else -1.5
        purpose_adj = 2.0 if person.purpose_score > 0.7 else 0.0 if person.purpose_score > 0.4 else -2.0
        relationship_adj = 1.5 if person.relationship_quality > 0.7 else 0.0 if person.relationship_quality > 0.4 else -1.5
        income_adj = 2.0 if person.annual_income > 100000 else 0.0 if person.annual_income > 40000 else -2.0

        adjusted_le = base_le + sex_adj + bmi_adj + smoke_adj + alcohol_adj + exercise_adj + sleep_adj + diet_adj + chronic_adj + stress_adj + social_adj + purpose_adj + relationship_adj + income_adj
        adjusted_le = max(40, min(105, adjusted_le))

        healthspan = adjusted_le - 10 - (3 if person.chronic_conditions else 0) - (2 if person.stress_level > 0.6 else 0)

        upcoming_milestones = []
        for ms in HEALTH_MILESTONES:
            if ms["age"] > person.age and ms["age"] <= person.age + years:
                prob = min(1.0, ms["impact"] * 3 + 0.3)
                upcoming_milestones.append({
                    "age": ms["age"],
                    "event": ms["event"],
                    "type": ms["type"],
                    "probability": round(prob, 3),
                    "impact": ms["impact"],
                    "years_until": ms["age"] - person.age,
                })

        condition_risks = {}
        fh = person.family_history
        base_risks = {
            "cardiovascular_disease": 0.40, "cancer": 0.35, "alzheimers_dementia": 0.12,
            "diabetes_type2": 0.25, "depression": 0.20, "stroke": 0.15,
            "chronic_kidney_disease": 0.10, "copd": 0.08, "osteoporosis": 0.15,
        }
        for condition, base_prob in base_risks.items():
            risk = base_prob
            if fh.get(condition, False):
                risk *= 1.8
            if condition == "cardiovascular_disease":
                risk += 0.1 if bmi > 30 else 0.05 if bmi > 25 else 0
                risk += 0.1 if person.smoking_status == "current" else 0
            elif condition == "diabetes_type2":
                risk += 0.15 if bmi > 30 else 0.05 if bmi > 25 else 0
            elif condition == "depression":
                risk += 0.1 if person.stress_level > 0.6 else 0
                risk -= 0.05 if person.close_friends >= 3 else 0
            elif condition == "cancer":
                risk += 0.05 if person.smoking_status == "current" else 0
            risk = min(0.95, max(0.01, risk))
            condition_risks[condition] = round(risk, 3)

        health_score_by_year = []
        current_score = 0.85 if not person.chronic_conditions else 0.65
        for yr in range(1, years + 1):
            age = person.age + yr
            age_decline = 0.003 * max(0, age - 40) / 10
            lifestyle_effect = (exercise_adj + diet_adj + sleep_adj) * 0.005
            current_score = max(0.1, min(1.0, current_score - age_decline + lifestyle_effect + rng.normal(0, 0.01)))
            health_score_by_year.append({"year": yr, "age": age, "health_score": round(float(current_score), 3)})

        return sanitize_nan({
            "life_expectancy_adjusted": round(adjusted_le, 1),
            "healthspan_estimate": round(max(adjusted_le * 0.8, healthspan), 1),
            "current_bmi": round(bmi, 1),
            "bmi_category": "underweight" if bmi < 18.5 else "normal" if bmi < 25 else "overweight" if bmi < 30 else "obese",
            "upcoming_health_milestones": upcoming_milestones,
            "condition_risks": condition_risks,
            "top_3_risks": sorted(condition_risks.items(), key=lambda x: x[1], reverse=True)[:3],
            "modifiable_factors": {
                "exercise": {"current_min_weekly": person.exercise_minutes_weekly, "recommended": 150, "gap": max(0, 150 - person.exercise_minutes_weekly)},
                "sleep": {"current_hours": person.sleep_hours_avg, "recommended": "7-9", "quality": "good" if 7 <= person.sleep_hours_avg <= 9 else "poor"},
                "diet": {"current_score": person.diet_quality, "target": 0.8},
                "weight": {"current_bmi": round(bmi, 1), "target_bmi": 22.0, "adjustment_kg": round(person.weight_kg - 22 * (person.height_cm / 100) ** 2, 1)},
            },
            "health_trajectory": health_score_by_year[:10],
        })

    def predict_financial_trajectory(self, person: PersonProfile, years: int = 30) -> dict:
        rng = np.random.RandomState(self.seed)

        career = self.predict_career_trajectory(person, years=years)
        income_series = [t["projected_income"] for t in career.get("trajectory", [])]
        if not income_series:
            income_series = [person.annual_income * (1 + person.career_growth_rate) ** y for y in range(years)]

        savings_rate = person.savings_rate
        initial_savings = person.net_worth
        current_savings = initial_savings
        debt = person.debt_total
        savings_trajectory = [current_savings]

        for yr in range(years):
            income = income_series[yr] if yr < len(income_series) else income_series[-1] if income_series else person.annual_income
            annual_saving = income * savings_rate
            debt_payment = debt * 0.08
            debt = max(0, debt - debt_payment)
            interest = current_savings * 0.05
            current_savings += annual_saving + interest - (debt_payment if yr < 10 else 0)
            savings_trajectory.append(round(current_savings, 0))

        retirement_target = person.annual_income * 25
        fire_number = person.annual_income * 300 / savings_rate if savings_rate > 0 else float('inf')

        net_worth_by_decade = {}
        for yr in [10, 20, 30]:
            idx = min(yr, len(savings_trajectory) - 1)
            net_worth_by_decade[f"year_{yr}"] = round(savings_trajectory[idx], 0)

        financial_risks = []
        if person.debt_total > person.annual_income * 0.5:
            financial_risks.append({"risk": "debt_overhang", "severity": "high", "description": "Debt exceeds 50% of annual income"})
        if savings_rate < 0.10:
            financial_risks.append({"risk": "insufficient_savings", "severity": "high", "description": "Savings rate below 10%"})
        if person.annual_income < 50000:
            financial_risks.append({"risk": "income_insufficiency", "severity": "medium", "description": "Income below median"})
        if person.net_worth < 0:
            financial_risks.append({"risk": "negative_net_worth", "severity": "high", "description": "Net worth is negative"})

        return sanitize_nan({
            "current_net_worth": person.net_worth,
            "current_debt": person.debt_total,
            "savings_rate": savings_rate,
            "projected_net_worth": net_worth_by_decade,
            "retirement_target": round(retirement_target, 0),
            "fire_number": round(fire_number, 0) if fire_number != float('inf') else None,
            "years_to_retirement_readiness": round(retirement_target / max(1, (person.annual_income * savings_rate * 0.05)), 1) if savings_rate > 0 else None,
            "financial_risks": financial_risks,
            "recommendations": self._financial_recommendations(person, savings_rate, debt),
        })

    def predict_relationship_trajectory(self, person: PersonProfile, years: int = 20) -> dict:
        rng = np.random.RandomState(self.seed)

        current_quality = person.relationship_quality
        current_satisfaction = person.life_satisfaction

        if person.marital_status == "married":
            divorce_annual_prob = 0.015 + (1.0 - current_quality) * 0.03
            satisfaction_trend = current_quality * 0.01 - 0.005
        elif person.marital_status == "single":
            partnership_prob = 0.05 if 25 <= person.age <= 40 else 0.02 if 40 < person.age <= 55 else 0.01
            satisfaction_trend = -0.003 if person.close_friends < 2 else 0.001
        else:
            divorce_annual_prob = 0.02
            satisfaction_trend = 0.001

        n_children = person.n_children
        child_milestones = []
        for i, child_age in enumerate(person.children_ages):
            future_events = []
            if child_age + years >= 18 and child_age < 18:
                future_events.append({"age": 18, "event": "Child reaches adulthood", "impact": "reduced_financial_burden"})
            if child_age + years >= 22 and child_age < 22:
                future_events.append({"age": 22, "event": "Child college/completion", "impact": "major_financial_milestone"})
            if future_events:
                child_milestones.append({"child": i + 1, "events": future_events})

        relationship_events = []
        for yr in range(1, years + 1):
            future_age = person.age + yr
            if person.marital_status == "married" and rng.random() < divorce_annual_prob:
                relationship_events.append({"year": yr, "age": future_age, "event": "divorce_risk_elevated", "probability": round(divorce_annual_prob, 3)})
            if person.marital_status == "single" and rng.random() < partnership_prob * 0.1:
                relationship_events.append({"year": yr, "age": future_age, "event": "partnership_formation", "probability": round(partnership_prob, 3)})
            if future_age in [50, 65, 75]:
                relationship_events.append({"year": yr, "age": future_age, "event": "social_network_contraction", "probability": 0.3, "impact": "-1 to -3 close connections"})

        return sanitize_nan({
            "current_marital_status": person.marital_status,
            "relationship_quality": person.relationship_quality,
            "close_friends": person.close_friends,
            "social_network_size": person.social_network_size,
            "estimated_divorce_probability_annual": round(divorce_annual_prob, 3) if person.marital_status == "married" else 0,
            "child_milestones": child_milestones,
            "predicted_relationship_events": relationship_events[:8],
            "loneliness_risk": round(max(0, 0.3 - person.close_friends * 0.05 + person.age * 0.002), 3),
            "social_support_adequacy": "adequate" if person.close_friends >= 3 else "at_risk" if person.close_friends >= 1 else "isolated",
            "recommendations": self._relationship_recommendations(person),
        })

    def predict_personal_timeline(self, person: PersonProfile, years: int = 20) -> dict:
        rng = np.random.RandomState(self.seed)

        likely_events = []
        for event_id, template in LIFE_EVENT_TEMPLATES.items():
            age_mod = template["age_modifier"](person.age)
            adjusted_prob = min(0.5, template["base_probability_annual"] * age_mod)
            period_prob = 1.0 - (1.0 - adjusted_prob) ** years
            if period_prob > 0.02:
                expected_year = min(years, int(1.0 / (adjusted_prob + 0.001)))
                likely_events.append({
                    "event_id": event_id,
                    "name": template["name"],
                    "annual_probability": round(adjusted_prob, 4),
                    "period_probability": round(period_prob, 4),
                    "expected_year": expected_year,
                    "impact": template["impact"],
                    "cascade_effects": template.get("cascade", []),
                })

        likely_events.sort(key=lambda e: e["period_probability"], reverse=True)

        return sanitize_nan({
            "person_age": person.age,
            "prediction_years": years,
            "probable_life_events": likely_events[:15],
            "top_5_likely_events": likely_events[:5],
            "total_event_burden": round(sum(e["period_probability"] for e in likely_events), 3),
        })

    def assess_family_dynamics(self, person: PersonProfile, family_members: list = None) -> dict:
        if family_members is None:
            family_members = []

        active_patterns = []
        for pattern_id, pattern in FAMILY_DYNAMICS_PATTERNS.items():
            try:
                modifier = pattern["probability_modifier"](person)
            except Exception:
                modifier = 1.0
            intensity = min(2.0, max(0.0, modifier))
            if intensity > 0.8:
                active_patterns.append({
                    "pattern_id": pattern_id,
                    "name": pattern["name"],
                    "intensity": round(intensity, 3),
                    "impact": pattern["impact"],
                    "description": pattern["description"],
                })

        family_health_summary = {"average_health": 0.7, "at_risk_members": []}
        if family_members:
            health_scores = [m.health_status for m in family_members]
            family_health_summary["average_health"] = round(float(np.mean(health_scores)), 3)
            for m in family_members:
                if m.health_status < 0.5:
                    family_health_summary["at_risk_members"].append({
                        "relation": m.relation,
                        "health_status": m.health_status,
                        "caregiving_burden": m.caregiving_burden,
                    })

        financial_interdependence = 0.5
        if family_members:
            total_income_fraction = sum(m.income_contribution for m in family_members)
            financial_interdependence = max(0.3, 1.0 - abs(0.5 - total_income_fraction))

        children_outlook = []
        for i, child_age in enumerate(person.children_ages):
            child_outlook = {
                "child_number": i + 1,
                "current_age": child_age,
                "education_stage": "preschool" if child_age < 5 else "elementary" if child_age < 11 else "secondary" if child_age < 18 else "adulthood",
                "years_to_independence": max(0, 18 - child_age),
                "estimated_education_cost_remaining": round(max(0, (18 - child_age)) * 15000 + max(0, 22 - child_age) * 30000, 0),
            }
            children_outlook.append(child_outlook)

        return sanitize_nan({
            "family_size": 1 + len(family_members),
            "active_patterns": active_patterns,
            "family_health_summary": family_health_summary,
            "financial_interdependence": round(financial_interdependence, 3),
            "children_outlook": children_outlook,
            "overall_family_stability": round(0.5 + person.relationship_quality * 0.2 - person.stress_level * 0.1 + (0.1 if not person.chronic_conditions else -0.1), 3),
            "recommendations": self._family_recommendations(person, family_members),
        })

    def assess_social_network(self, person: PersonProfile, contacts: list = None) -> dict:
        if contacts is None:
            contacts = []

        network_health = {
            "size": len(contacts) + person.social_network_size,
            "avg_closeness": 0.5,
            "avg_influence": 0.4,
            "emotional_balance": 0.5,
            "professional_value": 0.4,
            "health_influence": 0.5,
        }
        if contacts:
            network_health["avg_closeness"] = round(float(np.mean([c.closeness for c in contacts])), 3)
            network_health["avg_influence"] = round(float(np.mean([c.influence_score for c in contacts])), 3)
            network_health["emotional_balance"] = round(float(np.mean([c.emotional_valence for c in contacts])), 3)
            network_health["professional_value"] = round(float(np.mean([c.professional_value for c in contacts])), 3)
            network_health["health_influence"] = round(float(np.mean([c.health_influence for c in contacts])), 3)

        active_effects = []
        for effect_id, effect_data in SOCIAL_NETWORK_EFFECTS.items():
            if effect_id == "health_contagion" and network_health["health_influence"] > 0.4:
                active_effects.append({"effect": effect_id, "name": effect_data["name"], "mechanism": effect_data["mechanism"], "strength": effect_data["strength"], "direction": "positive" if person.exercise_minutes_weekly > 100 else "negative"})
            elif effect_id == "emotional_contagion" and network_health["emotional_balance"] < 0.5:
                active_effects.append({"effect": effect_id, "name": effect_data["name"], "mechanism": effect_data["mechanism"], "strength": effect_data["strength"], "direction": "negative"})
            elif effect_id == "career_network" and network_health["professional_value"] > 0.4:
                active_effects.append({"effect": effect_id, "name": effect_data["name"], "mechanism": effect_data["mechanism"], "strength": effect_data["strength"], "direction": "positive"})
            elif effect_id == "longevity_network":
                direction = "positive" if person.close_friends >= 3 else "negative"
                active_effects.append({"effect": effect_id, "name": effect_data["name"], "mechanism": effect_data["mechanism"], "strength": effect_data["strength"], "direction": direction})
            elif effect_id == "financial_contagion":
                active_effects.append({"effect": effect_id, "name": effect_data["name"], "mechanism": effect_data["mechanism"], "strength": effect_data["strength"], "direction": "positive" if person.savings_rate > 0.15 else "negative"})

        positive_contacts = sum(1 for c in contacts if c.emotional_valence > 0.6) if contacts else person.close_friends
        negative_contacts = sum(1 for c in contacts if c.emotional_valence < 0.3) if contacts else max(0, person.social_network_size - person.close_friends)

        return sanitize_nan({
            "network_size": network_health["size"],
            "close_connections": person.close_friends,
            "positive_negative_ratio": round(max(0.1, positive_contacts / max(1, negative_contacts)), 2),
            "network_health": network_health,
            "active_network_effects": active_effects,
            "network_adequacy": "strong" if person.close_friends >= 5 else "adequate" if person.close_friends >= 3 else "at_risk" if person.close_friends >= 1 else "isolated",
            "loneliness_score": round(max(0, 1.0 - person.close_friends * 0.15 - person.community_involvement * 0.2), 3),
            "network_reliability": round(float(np.mean([c.reliability for c in contacts])), 3) if contacts else 0.5,
            "recommendations": self._social_recommendations(person, contacts),
        })

    def predict_comprehensive_future(self, person: PersonProfile, family_members: list = None, contacts: list = None, years: int = 20) -> dict:
        career = self.predict_career_trajectory(person, years=years)
        health = self.predict_health_trajectory(person, years=years)
        financial = self.predict_financial_trajectory(person, years=years)
        relationship = self.predict_relationship_trajectory(person, years=years)
        timeline = self.predict_personal_timeline(person, years=years)
        family = self.assess_family_dynamics(person, family_members)
        network = self.assess_social_network(person, contacts)

        overall_outlook = (
            min(1.0, person.life_satisfaction) * 0.2 +
            min(1.0, person.relationship_quality) * 0.15 +
            health.get("life_expectancy_adjusted", 79) / 100 * 0.15 +
            min(1.0, person.job_satisfaction) * 0.1 +
            person.purpose_score * 0.1 +
            person.resilience_score * 0.1 +
            min(1.0, person.close_friends / 5) * 0.1 +
            (0.5 if person.savings_rate > 0.15 else 0.2) * 0.1
        )
        overall_outlook = float(np.clip(overall_outlook, 0.1, 1.0))

        if overall_outlook > 0.7:
            outlook_label = "flourishing"
        elif overall_outlook > 0.55:
            outlook_label = "stable_with_potential"
        elif overall_outlook > 0.4:
            outlook_label = "managing"
        else:
            outlook_label = "at_risk"

        return sanitize_nan({
            "person": {
                "age": person.age,
                "occupation": person.occupation,
                "marital_status": person.marital_status,
                "n_children": person.n_children,
            },
            "overall_outlook_score": round(overall_outlook, 3),
            "overall_outlook_label": outlook_label,
            "career": career,
            "health": health,
            "financial": financial,
            "relationship": relationship,
            "timeline": timeline,
            "family": family,
            "network": network,
            "key_concerns": self._extract_concerns(health, financial, relationship, network),
            "top_opportunities": self._extract_opportunities(career, health, financial),
        })

    def _financial_recommendations(self, person: PersonProfile, savings_rate: float, debt: float) -> list:
        recs = []
        if savings_rate < 0.15:
            recs.append("Increase savings rate to at least 15% — automate contributions before discretionary spending")
        if debt > person.annual_income * 0.3:
            recs.append("Prioritize high-interest debt elimination — avalanche method: pay minimum on all, extra on highest rate")
        if person.net_worth < person.annual_income:
            recs.append("Build emergency fund to cover 6 months expenses before aggressive investing")
        if person.annual_income < 80000:
            recs.append("Focus on income growth: skills, certifications, or career moves that increase earning power")
        if not recs:
            recs.append("Financial position is solid — focus on diversification and tax-advantaged accounts")
        return recs[:5]

    def _relationship_recommendations(self, person: PersonProfile) -> list:
        recs = []
        if person.close_friends < 3:
            recs.append("Critical: build at least 3 close friendships — join clubs, volunteer, or reconnect with old friends")
        if person.relationship_quality < 0.5:
            recs.append("Relationship quality is low — consider couples counseling or relationship skills training (Gottman method)")
        if person.social_network_size < 8:
            recs.append("Expand social network through community involvement, professional groups, or hobby communities")
        if person.loneliness_index if hasattr(person, 'loneliness_index') else person.close_friends < 2:
            recs.append("Loneliness is a health risk equivalent to smoking 15 cigarettes/day — prioritize social connection")
        if not recs:
            recs.append("Social connections are healthy — maintain and deepen existing relationships")
        return recs[:5]

    def _family_recommendations(self, person: PersonProfile, family_members: list) -> list:
        recs = []
        if person.n_children > 0 and person.stress_level > 0.6:
            recs.append("High stress is affecting family dynamics — stress management is a parenting intervention")
        if family_members and any(m.health_status < 0.5 for m in family_members):
            recs.append("At-risk family member needs care plan — establish support network before crisis")
        if person.n_children > 0 and person.savings_rate < 0.15:
            recs.append("Financial buffer is essential for family stability — aim for 6-month emergency fund")
        if person.age >= 40 and person.n_children > 0:
            recs.append("Update estate planning: will, guardianship, life insurance — protect family against worst case")
        if not recs:
            recs.append("Family dynamics are healthy — continue open communication and shared activities")
        return recs[:5]

    def _social_recommendations(self, person: PersonProfile, contacts: list) -> list:
        recs = []
        if person.close_friends < 3:
            recs.append("Critical health risk: fewer than 3 close connections — invest in deeper friendships immediately")
        if contacts and any(c.emotional_valence < 0.3 for c in contacts):
            recs.append("Some relationships are net-negative — consider setting boundaries with chronically negative contacts")
        if person.community_involvement < 0.3:
            recs.append("Low community involvement — volunteer, join local groups, or attend community events monthly")
        if not recs:
            recs.append("Social network appears healthy — nurture key relationships and diversify connections")
        return recs[:4]

    def _extract_concerns(self, health: dict, financial: dict, relationship: dict, network: dict) -> list:
        concerns = []
        if health.get("life_expectancy_adjusted", 79) < 70:
            concerns.append({"area": "health", "concern": f"Life expectancy adjusted to {health.get('life_expectancy_adjusted', 'N/A')} — well below national average"})
        for risk_name, risk_val in health.get("top_3_risks", []):
            if risk_val > 0.3:
                concerns.append({"area": "health", "concern": f"{risk_name}: {risk_val:.0%} lifetime risk"})
        for risk in financial.get("financial_risks", []):
            concerns.append({"area": "financial", "concern": f"{risk['risk']}: {risk['description']}"})
        if network.get("loneliness_score", 0) > 0.5:
            concerns.append({"area": "social", "concern": "High loneliness score — social isolation health risk"})
        if relationship.get("estimated_divorce_probability_annual", 0) > 0.03:
            concerns.append({"area": "relationship", "concern": "Elevated divorce risk — relationship intervention recommended"})
        return concerns[:8]

    def _extract_opportunities(self, career: dict, health: dict, financial: dict) -> list:
        opportunities = []
        if career.get("years_to_peak", 0) > 5:
            opportunities.append({"area": "career", "opportunity": f"Income growth expected for {career.get('years_to_peak', 0)} more years — invest aggressively now"})
        savings_rate = financial.get("savings_rate", 0)
        if savings_rate < 0.20:
            opportunities.append({"area": "financial", "opportunity": f"Increasing savings from {savings_rate:.0%} to 20% would add significant retirement safety margin"})
        modifiable = health.get("modifiable_factors", {})
        if modifiable.get("exercise", {}).get("gap", 0) > 60:
            opportunities.append({"area": "health", "opportunity": "Adding 60+ min/week of exercise would significantly improve healthspan"})
        return opportunities[:5]