"""Life Trajectory Engines — Year-by-year future projection for each life domain.

Evidence bases:
- Health: GBD 2019, Framingham Heart Study, Nurses' Health Study, WHI, UK Biobank
- Career: BLS National Longitudinal Surveys, Chetty et al. mobility, ACS income data
- Relationships: National Survey of Family Growth, General Social Survey, CDC marriage/divorce data
- Finance: SCF (Survey of Consumer Finances), Federal Reserve data, compound interest math
- Cognition: Rotterdam Study, Mayo Clinic Study of Aging, NIA-AA criteria for MCI/dementia
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class YearTrajectory:
    year: int
    age: int
    value: float
    confidence: float
    phase: str


@dataclass
class InterventionEffect:
    name: str
    value_added: float
    confidence: float
    time_to_effect_years: float
    evidence: str


@dataclass
class HealthTrajectoryResult:
    life_expectancy: float
    healthspan: float
    biological_age: float
    yearly_trajectory: List[YearTrajectory]
    disease_onset_probabilities: Dict[str, Dict[str, float]]
    interventions: List[InterventionEffect]
    narrative: str


@dataclass
class CareerTrajectoryResult:
    peak_income: float
    peak_income_age: int
    total_career_earnings: float
    yearly_income: List[YearTrajectory]
    promotion_probabilities: Dict[int, float]
    job_loss_probabilities: Dict[int, float]
    industry_outlook: str
    interventions: List[InterventionEffect]
    narrative: str


@dataclass
class RelationshipTrajectoryResult:
    marriage_probability_by_age: Dict[int, float]
    divorce_probability_by_age: Dict[int, float]
    children_probability: float
    social_network_size_trajectory: List[YearTrajectory]
    loneliness_trajectory: List[YearTrajectory]
    relationship_satisfaction_trajectory: List[YearTrajectory]
    interventions: List[InterventionEffect]
    narrative: str


@dataclass
class FinancialTrajectoryResult:
    net_worth_at_65: float
    fire_age: Optional[int]
    debt_free_age: Optional[int]
    yearly_net_worth: List[YearTrajectory]
    yearly_income: List[YearTrajectory]
    yearly_savings: List[YearTrajectory]
    recession_impact_scenario: Dict[str, float]
    interventions: List[InterventionEffect]
    narrative: str


@dataclass
class CognitiveTrajectoryResult:
    peak_cognitive_age: int
    cognitive_decline_start_age: int
    mci_probability_by_85: float
    dementia_probability_by_85: float
    yearly_cognitive: List[YearTrajectory]
    brain_health_interventions: List[InterventionEffect]
    narrative: str


DISEASE_ONSET_BASE = {
    "cardiovascular": {40: 0.02, 50: 0.06, 60: 0.15, 70: 0.30, 80: 0.50, 85: 0.70},
    "cancer": {40: 0.01, 50: 0.03, 60: 0.08, 70: 0.18, 80: 0.30, 85: 0.40},
    "diabetes_type2": {30: 0.01, 40: 0.04, 50: 0.10, 60: 0.20, 70: 0.30, 80: 0.35},
    "dementia": {60: 0.01, 65: 0.02, 70: 0.05, 75: 0.12, 80: 0.25, 85: 0.45, 90: 0.60},
    "copd": {40: 0.01, 50: 0.03, 60: 0.06, 70: 0.10, 80: 0.15},
    "stroke": {45: 0.01, 55: 0.04, 65: 0.10, 75: 0.20, 85: 0.35},
    "depression": {20: 0.08, 30: 0.10, 40: 0.09, 50: 0.08, 60: 0.07, 70: 0.08, 80: 0.10},
    "arthritis": {40: 0.05, 50: 0.15, 60: 0.30, 70: 0.45, 80: 0.60},
}

INTERVENTION_EFFECTS = {
    "exercise_150": {"le_added": 3.5, "healthspan_added": 4.0, "cognitive_benefit": 0.12, "time_years": 0.5,
                     "evidence": "AHA 2022, Moore et al. PLOS Medicine 2012: 150 min/wk adds 3.5 yr LE"},
    "exercise_300": {"le_added": 5.0, "healthspan_added": 6.0, "cognitive_benefit": 0.18, "time_years": 1.0,
                     "evidence": "Kelly et al. JAMA Intern Med 2014: 300 min/wk adds 5 yr LE"},
    "sleep_7_9": {"le_added": 2.0, "healthspan_added": 2.5, "cognitive_benefit": 0.15, "time_years": 0.1,
                  "evidence": "Cappuccio et al. Sleep 2010: 7-9h optimal; <6h adds 12% mortality"},
    "med_diet": {"le_added": 2.5, "healthspan_added": 3.0, "cognitive_benefit": 0.10, "time_years": 1.0,
                 "evidence": "Estruch et al. NEJM 2018: Mediterranean diet reduces CV events 30%"},
    "quit_smoking": {"le_added": 10.0, "healthspan_added": 8.0, "cognitive_benefit": 0.05, "time_years": 5.0,
                     "evidence": "Jha et al. NEJM 2013: quitting at 30 adds 10 yr; at 40 adds 9; at 50 adds 6"},
    "social_3_friends": {"le_added": 2.5, "healthspan_added": 3.0, "cognitive_benefit": 0.08, "time_years": 0.5,
                        "evidence": "Holt-Lunstad et al. PLOS Medicine 2010: 3+ friends reduces mortality 50%"},
    "purpose": {"le_added": 2.0, "healthspan_added": 2.5, "cognitive_benefit": 0.12, "time_years": 2.0,
                "evidence": "Alimujiang et al. JAMA Network Open 2019: purpose reduces all-cause mortality 15-30%"},
    "stress_mgmt": {"le_added": 1.5, "healthspan_added": 2.0, "cognitive_benefit": 0.10, "time_years": 0.25,
                    "evidence": "Goyal et al. JAMA Intern Med 2014: meditation reduces anxiety, cortisol, inflammation"},
    "weight_loss": {"le_added": 2.0, "healthspan_added": 3.0, "cognitive_benefit": 0.05, "time_years": 2.0,
                    "evidence": "Losing BMI from 35 to 25 reduces diabetes risk 60%, CV risk 30%"},
    "cognitive_training": {"le_added": 0.5, "healthspan_added": 1.0, "cognitive_benefit": 0.15, "time_years": 1.0,
                          "evidence": "ACTIVE trial JAMA 2006: cognitive training improves processing speed, delays decline"},
}

INCOME_GROWTH_RATES = {
    12: {"early": 0.02, "mid": 0.015, "late": 0.005, "peak_age": 48, "decline_start": 55},
    14: {"early": 0.03, "mid": 0.02, "late": 0.01, "peak_age": 45, "decline_start": 52},
    16: {"early": 0.04, "mid": 0.035, "late": 0.015, "peak_age": 42, "decline_start": 50},
    18: {"early": 0.06, "mid": 0.05, "late": 0.02, "peak_age": 40, "decline_start": 48},
    20: {"early": 0.08, "mid": 0.06, "late": 0.025, "peak_age": 38, "decline_start": 45},
}

MARRIAGE_RATE_BY_AGE = {
    20: 0.015, 22: 0.03, 25: 0.06, 28: 0.08, 30: 0.07,
    33: 0.05, 35: 0.04, 38: 0.03, 40: 0.02, 45: 0.01, 50: 0.005, 60: 0.002,
}

DIVORCE_RATE_BY_AGE = {
    25: 0.04, 28: 0.035, 30: 0.03, 33: 0.025, 35: 0.02,
    38: 0.018, 40: 0.015, 45: 0.012, 50: 0.01, 55: 0.008, 60: 0.005,
}

SOCIAL_NETWORK_DECAY = 0.03


class HealthTrajectoryEngine:
    def __init__(self, n_simulations: int = 500):
        self.n_simulations = n_simulations

    def project(self, age: int = 30, sex: str = "male",
                exercise_min: int = 90, sleep_hours: float = 7.0,
                diet_quality: float = 0.5, smoking: str = "never",
                alcohol: int = 3, bmi: float = 24.0,
                chronic_conditions: List[str] = None,
                family_history: Dict[str, str] = None) -> HealthTrajectoryResult:
        chronic_conditions = chronic_conditions or []
        family_history = family_history or {}
        rng = np.random.RandomState(42)
        base_le = 80.0 if sex == "female" else 76.0
        le_modifiers = self._compute_le_modifiers(sex, exercise_min, sleep_hours, diet_quality, smoking, alcohol, bmi, chronic_conditions, family_history)
        le = base_le + sum(le_modifiers.values())
        le = max(40, min(110, le))
        healthspan_ratio = 0.78 + (exercise_min / 300) * 0.1 + (diet_quality - 0.5) * 0.06 + (0.2 if smoking == "never" else 0)
        healthspan_ratio = max(0.55, min(0.92, healthspan_ratio))
        healthspan = le * healthspan_ratio
        bio_age = self._compute_bio_age(age, exercise_min, sleep_hours, diet_quality, smoking, bmi, chronic_conditions, family_history)
        trajectory = []
        health = 0.85 if age < 40 else max(0.5, 0.85 - (age - 40) * 0.008)
        for yr in range(min(60, max(1, int(le - age)))):
            proj_age = age + yr
            health *= (1 - 0.005 * max(0, (proj_age - 45) / 10))
            if smoking == "current":
                health *= 0.998
            if exercise_min >= 150:
                health = min(1.0, health * 1.002)
            if proj_age >= 65:
                health *= 0.995
            health = max(0.05, min(1.0, health))
            conf = max(0.3, 0.85 - yr * 0.008)
            phase = "youth" if proj_age < 30 else "building" if proj_age < 45 else "peak" if proj_age < 55 else "maintenance" if proj_age < 70 else "decline" if proj_age < 85 else "late"
            trajectory.append(YearTrajectory(year=2026 + yr, age=proj_age, value=round(float(health), 4), confidence=round(float(conf), 4), phase=phase))
        disease_probs = {}
        for disease, onset in DISEASE_ONSET_BASE.items():
            disease_probs[disease] = {}
            mod = 1.0
            if disease == "cardiovascular":
                if exercise_min >= 150: mod *= 0.7
                if smoking == "current": mod *= 1.8
                if bmi > 30: mod *= 1.3
            elif disease == "cancer":
                if smoking == "current": mod *= 2.5
                if alcohol > 14: mod *= 1.3
            elif disease == "diabetes_type2":
                if bmi > 30: mod *= 3.0
                if exercise_min < 60: mod *= 1.5
            elif disease == "dementia":
                if exercise_min >= 150: mod *= 0.7
                if sleep_hours < 6: mod *= 1.3
            elif disease == "copd":
                if smoking == "current": mod *= 5.0
            for onset_age, base_p in onset.items():
                if onset_age >= age:
                    disease_probs[disease][str(onset_age)] = round(float(np.clip(base_p * mod, 0, 0.95)), 3)
        interventions = []
        intervention_defs = [
            ("exercise_150", exercise_min < 150),
            ("exercise_300", exercise_min < 300),
            ("sleep_7_9", sleep_hours < 7 or sleep_hours > 9),
            ("med_diet", diet_quality < 0.6),
            ("quit_smoking", smoking == "current"),
            ("social_3_friends", True),
            ("purpose", True),
            ("stress_mgmt", True),
            ("weight_loss", bmi > 30),
            ("cognitive_training", True),
        ]
        for intv_name, recommended in intervention_defs:
            if recommended:
                eff = INTERVENTION_EFFECTS[intv_name]
                interventions.append(InterventionEffect(
                    name=intv_name, value_added=eff["le_added"],
                    confidence=0.7, time_to_effect_years=eff["time_years"],
                    evidence=eff["evidence"]))
        if bio_age > age + 3:
            narrative = f"Biological age {bio_age:.0f} exceeds chronological {age} — you are aging faster than your years. The highest-ROI interventions are exercise, sleep, and social connection."
        elif bio_age < age - 2:
            narrative = f"Biological age {bio_age:.0f} is younger than {age} — you have a health advantage that compounds over decades."
        else:
            narrative = f"Biological age {bio_age:.0f} aligns with chronological {age} — neither advantage nor crisis, but the trajectory depends on the next decade."
        return HealthTrajectoryResult(
            life_expectancy=round(float(le), 1), healthspan=round(float(healthspan), 1),
            biological_age=round(float(bio_age), 1), yearly_trajectory=trajectory,
            disease_onset_probabilities=disease_probs, interventions=interventions, narrative=narrative)

    def _compute_le_modifiers(self, sex, exercise, sleep, diet, smoking, alcohol, bmi, chronic, family) -> Dict[str, float]:
        mods = {}
        mods["exercise"] = 3.5 if exercise >= 150 else 1.0 if exercise >= 60 else -1.0
        mods["sleep"] = 1.0 if 7 <= sleep <= 9 else -2.0 if sleep < 6 else -0.5
        mods["diet"] = 2.5 if diet > 0.7 else 0.5 if diet > 0.4 else -1.5
        mods["smoking"] = -10.0 if smoking == "current" else -3.0 if smoking == "former" else 0
        mods["alcohol"] = 0.5 if alcohol <= 7 else -1.0 if alcohol <= 14 else -3.0
        mods["bmi"] = -3.0 if bmi > 30 else -0.5 if bmi > 25 else 0.5 if 18.5 <= bmi <= 25 else -1.5
        mods["social"] = 2.5
        mods["purpose"] = 2.0
        mods["sex"] = 4.0 if sex == "female" else 0
        cond_mods = {"diabetes": -7.0, "hypertension": -5.0, "heart_disease": -10.0,
                     "cancer_survivor": -8.0, "depression": -3.0, "copd": -6.0, "obesity": -3.0}
        mods["chronic"] = sum(cond_mods.get(c, 0) for c in chronic) if chronic else 0
        fam_mods = {"alzheimer": -2.0, "heart_disease": -1.5, "cancer": -1.0, "diabetes": -0.5, "stroke": -1.5, "longevity": 4.0}
        mods["family"] = sum(fam_mods.get(k, 0) if v == "positive" else -fam_mods.get(k, 0) for k, v in family.items()) if family else 0
        return mods

    def _compute_bio_age(self, age, exercise, sleep, diet, smoking, bmi, chronic, family):
        bio = float(age)
        if exercise >= 150: bio -= 3.0
        elif exercise >= 60: bio -= 1.5
        else: bio += 2.0
        if 7 <= sleep <= 9: bio -= 1.0
        elif sleep < 6: bio += 2.5
        if diet > 0.7: bio -= 1.5
        elif diet < 0.3: bio += 2.0
        if smoking == "current": bio += 5.0
        elif smoking == "former": bio += 1.5
        if bmi > 30: bio += 2.5
        elif bmi > 25: bio += 0.5
        elif bmi < 18.5: bio += 1.5
        for c in chronic:
            bio += 2.0
        for k, v in (family or {}).items():
            if v == "positive":
                fam_impact = {"alzheimer": 2.0, "heart_disease": 1.5, "cancer": 1.0, "diabetes": 0.5}
                bio += fam_impact.get(k, 0)
            elif v == "negative" and k == "longevity":
                bio -= 4.0
        return max(18.0, bio)


class CareerTrajectoryEngine:
    def __init__(self):
        pass

    def project(self, age: int = 30, education_years: int = 16,
                current_income: float = 65000, industry: str = "professional",
                satisfaction: float = 0.6, growth_rate: float = 0.03) -> CareerTrajectoryResult:
        edu_key = min((k for k in INCOME_GROWTH_RATES if k >= education_years), default=max(INCOME_GROWTH_RATES))
        growth = INCOME_GROWTH_RATES[edu_key]
        income = current_income
        incomes = []
        peak_income = current_income
        peak_age = age
        promotion_probs = {}
        job_loss_probs = {}
        for yr in range(min(45, max(5, 65 - age))):
            proj_age = age + yr
            if proj_age < growth["peak_age"]:
                rate = growth["early"] * (1 + (satisfaction - 0.5) * 0.3)
            elif proj_age < growth["decline_start"]:
                rate = growth["mid"] * (1 + (satisfaction - 0.5) * 0.2)
            elif proj_age < 65:
                rate = growth["late"]
            else:
                rate = -0.02
            income *= (1 + rate)
            if proj_age < 30:
                promotion_probs[proj_age] = round(float(np.clip(0.08 + satisfaction * 0.04, 0.01, 0.2)), 3)
            elif proj_age < 40:
                promotion_probs[proj_age] = round(float(np.clip(0.06 + satisfaction * 0.03, 0.01, 0.15)), 3)
            elif proj_age < 50:
                promotion_probs[proj_age] = round(float(np.clip(0.04 + satisfaction * 0.02, 0.01, 0.10)), 3)
            else:
                promotion_probs[proj_age] = round(float(np.clip(0.02, 0.005, 0.05)), 3)
            if proj_age < 30:
                job_loss_probs[proj_age] = 0.04
            elif proj_age < 50:
                job_loss_probs[proj_age] = 0.035
            elif proj_age < 60:
                job_loss_probs[proj_age] = 0.05
            else:
                job_loss_probs[proj_age] = 0.08
            peak_income = max(peak_income, income)
            if income >= peak_income:
                peak_age = proj_age
            conf = max(0.3, 0.8 - yr * 0.01)
            phase = "early" if proj_age < 35 else "building" if proj_age < 45 else "peak" if proj_age < 55 else "maturity" if proj_age < 65 else "decline"
            incomes.append(YearTrajectory(year=2026 + yr, age=proj_age, value=round(float(income), 0), confidence=round(float(conf), 4), phase=phase))
        total_earnings = float(sum(i.value for i in incomes))
        industry_map = {"professional": "Steady growth, above-average earnings, AI disruption risk in middle management", "tech": "High growth, high volatility, age discrimination above 45", "healthcare": "Stable, recession-proof, growing demand, burnout risk", "finance": "Peak-heavy, bonus-driven, early exit common above 50", "education": "Low growth, high stability, low variance", "trades": "Moderate growth, skill-appreciation after 40, physical decline after 55", "creative": "High variance, portfolio career, age-agnostic"}
        outlook = industry_map.get(industry, "Moderate growth, industry-dependent stability")
        interventions = [
            InterventionEffect(name="continuous_learning", value_added=0.05, confidence=0.8, time_to_effect_years=1.0, evidence="Each additional year of education increases lifetime earnings 8-13% (Card 1999)"),
            InterventionEffect(name="career_switch_risk", value_added=0.10, confidence=0.6, time_to_effect_years=2.0, evidence="Career switchers at 30-35 see 15% income gain within 5 years; post-45 see 10% loss on average"),
            InterventionEffect(name="negotiation_training", value_added=0.08, confidence=0.7, time_to_effect_years=0.5, evidence="Negotiating initial offer compounds to $1M+ over career (Gerchick 2022)"),
            InterventionEffect(name="side_income_diversification", value_added=0.03, confidence=0.5, time_to_effect_years=1.0, evidence="Multiple income streams reduce career disruption impact by 40%"),
        ]
        narrative = f"Education {education_years} years, starting income ${current_income:,.0f}. "
        if education_years >= 16:
            narrative += "College degree trajectory typically peaks mid-40s with 3-4% annual growth. "
        elif education_years >= 14:
            narrative += "Some college trajectory — growth rate 2-3%, often limited by credential ceiling. "
        else:
            narrative += "High school trajectory — steady but slower growth, limited by credential and network constraints. "
        narrative += f"Estimated peak income: ${peak_income:,.0f} at age {peak_age}. Total career earnings: ${total_earnings:,.0f}."
        return CareerTrajectoryResult(
            peak_income=round(float(peak_income), 0), peak_income_age=peak_age,
            total_career_earnings=round(float(total_earnings), 0), yearly_income=incomes,
            promotion_probabilities=promotion_probs, job_loss_probabilities=job_loss_probs,
            industry_outlook=outlook, interventions=interventions, narrative=narrative)


class RelationshipTrajectoryEngine:
    def __init__(self):
        pass

    def project(self, age: int = 30, sex: str = "male",
                marital_status: str = "single", close_friends: int = 4,
                loneliness: float = 0.3, relationship_quality: float = 0.6,
                n_children: int = 0) -> RelationshipTrajectoryResult:
        rng = np.random.RandomState(42)
        marriage_probs = {}
        current_marriage_p = 0.0
        for target_age, base_rate in MARRIAGE_RATE_BY_AGE.items():
            if target_age >= age:
                mod = 1.0
                if sex == "male" and target_age < 35: mod *= 0.7
                if loneliness > 0.6: mod *= 0.6
                if close_friends >= 5: mod *= 1.3
                marriage_probs[target_age] = round(float(np.clip(base_rate * mod, 0.001, 0.15)), 4)
        divorce_probs = {}
        current_married = marital_status in ("married", "cohabiting")
        for target_age, base_rate in DIVORCE_RATE_BY_AGE.items():
            if target_age >= age and current_married:
                mod = 1.0
                if relationship_quality < 0.4: mod *= 2.0
                elif relationship_quality > 0.7: mod *= 0.5
                if loneliness > 0.5: mod *= 1.5
                divorce_probs[target_age] = round(float(np.clip(base_rate * mod, 0.001, 0.08)), 4)
            elif target_age >= age:
                divorce_probs[target_age] = 0.001
        children_p = 0.0
        if n_children == 0:
            if 25 <= age <= 35:
                children_p = 0.15
            elif 35 < age <= 40:
                children_p = 0.08
            elif 40 < age <= 45:
                children_p = 0.03
        elif n_children == 1:
            children_p = 0.12 if age < 38 else 0.03
        else:
            children_p = 0.02
        if sex == "male" and age > 50: children_p *= 0.3
        if sex == "female" and age > 40: children_p *= 0.1
        social_net = []
        loneliness_t = []
        satisfaction_t = []
        friends = float(close_friends)
        lonely = loneliness
        quality = relationship_quality
        for yr in range(min(50, max(10, 80 - age))):
            proj_age = age + yr
            friends *= (1 - SOCIAL_NETWORK_DECAY)
            if proj_age < 30:
                friends += 0.8
            elif proj_age < 50:
                friends += 0.2
            else:
                friends -= 0.3
            if 60 <= proj_age < 75:
                friends -= 0.15
            if marital_status in ("married", "cohabiting"):
                lonely = max(0.1, lonely * 0.95 + 0.01 * (1 - quality))
            else:
                lonely = min(0.9, lonely * 1.02 + 0.01)
            quality += (0.5 - quality) * 0.01
            quality = max(0.1, min(0.95, quality))
            lonely = max(0.05, min(0.95, lonely))
            friends = max(0, min(30, friends))
            conf = max(0.3, 0.75 - yr * 0.007)
            social_net.append(YearTrajectory(year=2026 + yr, age=proj_age, value=round(float(friends), 1), confidence=round(float(conf), 3), phase="active"))
            loneliness_t.append(YearTrajectory(year=2026 + yr, age=proj_age, value=round(float(lonely), 3), confidence=round(float(conf), 3), phase="stable" if lonely < 0.4 else "elevated"))
            satisfaction_t.append(YearTrajectory(year=2026 + yr, age=proj_age, value=round(float(quality), 3), confidence=round(float(conf), 3), phase="satisfied" if quality > 0.6 else "strained"))
        interventions = [
            InterventionEffect(name="join_community", value_added=2.5, confidence=0.8, time_to_effect_years=0.5, evidence="Community involvement reduces loneliness 40% and adds 2.5 years LE (Holt-Lunstad 2010)"),
            InterventionEffect(name="relationship_investment", value_added=1.5, confidence=0.75, time_to_effect_years=1.0, evidence="Marriage/partnership reduces mortality 10-15%; quality matters more than status"),
            InterventionEffect(name="therapy_counseling", value_added=1.0, confidence=0.7, time_to_effect_years=0.5, evidence="Couples therapy improves satisfaction 70-80% of cases (Gottman Institute)"),
            InterventionEffect(name="expand_social_network", value_added=2.0, confidence=0.6, time_to_effect_years=1.0, evidence="Each close friend adds 0.5-2 years of life; network diversity is key"),
        ]
        narrative = f"Age {age}, {sex}, currently {marital_status} with {close_friends} close friends and loneliness at {loneliness:.0%}."
        if lonely > 0.5:
            narrative += " High loneliness is equivalent to smoking 15 cigarettes/day for mortality. Social connection is your most urgent intervention."
        elif lonely > 0.3:
            narrative += " Moderate loneliness — vulnerable to social decay in mid-life. Investing in relationships now compounds for decades."
        else:
            narrative += " Social baseline is healthy — maintain and deepen connections as the single most protective factor."
        return RelationshipTrajectoryResult(
            marriage_probability_by_age=marriage_probs, divorce_probability_by_age=divorce_probs,
            children_probability=round(float(children_p), 3),
            social_network_size_trajectory=social_net, loneliness_trajectory=loneliness_t,
            relationship_satisfaction_trajectory=satisfaction_t, interventions=interventions, narrative=narrative)


class FinancialTrajectoryEngine:
    def __init__(self):
        pass

    def project(self, age: int = 30, income: float = 65000, net_worth: float = 50000,
                debt: float = 30000, savings_rate: float = 0.10,
                education_years: int = 16, risk_tolerance: float = 0.5,
                expected_return: float = 0.07) -> FinancialTrajectoryResult:
        edu_key = min((k for k in INCOME_GROWTH_RATES if k >= education_years), default=max(INCOME_GROWTH_RATES))
        growth = INCOME_GROWTH_RATES[edu_key]
        current_income = income
        current_nw = net_worth - debt
        yearly_nw = []
        yearly_inc = []
        yearly_sav = []
        for yr in range(min(50, max(5, 65 - age + 5))):
            proj_age = age + yr
            if proj_age < growth["peak_age"]:
                rate = growth["early"]
            elif proj_age < growth["decline_start"]:
                rate = growth["mid"]
            elif proj_age < 65:
                rate = growth["late"]
            elif proj_age < 68:
                rate = -0.02
            else:
                rate = -0.04
            current_income *= (1 + rate)
            if proj_age >= 65:
                current_income = min(current_income, income * 0.6)
            savings = current_income * max(0.05, savings_rate)
            investment_return = expected_return * risk_tolerance + 0.02 * (1 - risk_tolerance)
            current_nw = current_nw * (1 + investment_return) + savings
            conf = max(0.3, 0.8 - yr * 0.01)
            phase = "accumulation" if proj_age < 45 else "peak_earning" if proj_age < 55 else "pre_retirement" if proj_age < 65 else "distribution"
            yearly_nw.append(YearTrajectory(year=2026 + yr, age=proj_age, value=round(float(current_nw), 0), confidence=round(float(conf), 4), phase=phase))
            yearly_inc.append(YearTrajectory(year=2026 + yr, age=proj_age, value=round(float(current_income), 0), confidence=round(float(conf), 4), phase=phase))
            yearly_sav.append(YearTrajectory(year=2026 + yr, age=proj_age, value=round(float(savings), 0), confidence=round(float(conf), 4), phase=phase))
        nw_65 = next((y.value for y in yearly_nw if y.age >= 65), current_nw)
        fire_target = income * 25
        fire_age = None
        accum = max(0, net_worth - debt)
        for y in yearly_nw:
            if y.value >= fire_target and fire_age is None:
                fire_age = y.age
        debt_free_age = None
        remaining_debt = debt
        for y in yearly_sav:
            remaining_debt *= 1.05
            remaining_debt -= y.value * 0.3
            if remaining_debt <= 0 and debt_free_age is None:
                debt_free_age = y.age
                break
        recession_impact = {"mild": round(float(income * 0.05), 0), "moderate": round(float(income * 0.15), 0),
                           "severe": round(float(income * 0.30), 0), "pandemic": round(float(income * 0.40), 0)}
        interventions = [
            InterventionEffect(name="increase_savings_to_15", value_added=max(0, 0.15 - savings_rate) * income * 30, confidence=0.95, time_to_effect_years=1.0, evidence="Compound interest: saving 15% vs 10% for 30 years = $500K+ difference"),
            InterventionEffect(name="increase_savings_to_20", value_added=max(0, 0.20 - savings_rate) * income * 30, confidence=0.92, time_to_effect_years=1.0, evidence="FIRE math: 20% savings rate → FI in ~30 years; 50% → ~15 years"),
            InterventionEffect(name="eliminate_high_interest_debt", value_added=debt * 0.15 * 10, confidence=0.98, time_to_effect_years=2.0, evidence="Average credit card APR 24%; paying $100/mo instead of min saves $3K+ over life of debt"),
            InterventionEffect(name="increase_income_10pct", value_added=income * 0.10 * 30, confidence=0.6, time_to_effect_years=2.0, evidence="10% raise compounds to $600K+ over career"),
            InterventionEffect(name="rebalance_portfolio", value_added=current_nw * 0.005 * 30, confidence=0.7, time_to_effect_years=0.5, evidence="Proper asset allocation adds 0.5-1% annual return; 0.5% × 30 years = 15%+ more wealth"),
        ]
        narrative = f"Starting at ${income:,.0f}/yr, ${net_worth:,.0f} net worth, ${debt:,.0f} debt, saving {savings_rate:.0%}."
        if debt > income * 0.5:
            narrative += f" Debt-to-income ratio is {debt/income:.1%} — prioritize debt elimination (highest-ROI financial move)."
        if savings_rate < 0.15:
            narrative += f" Savings rate {savings_rate:.0%} is below the 15% minimum — you need to increase by {(0.15-savings_rate)*100:.0f} percentage points."
        if fire_age:
            narrative += f" FIRE (Financial Independence, Retire Early) achievable at age {fire_age}."
        else:
            narrative += " FIRE not achievable at current savings rate — increase to 20%+ to reach independence."
        return FinancialTrajectoryResult(
            net_worth_at_65=round(float(nw_65), 0), fire_age=fire_age, debt_free_age=debt_free_age,
            yearly_net_worth=yearly_nw, yearly_income=yearly_inc, yearly_savings=yearly_sav,
            recession_impact_scenario=recession_impact, interventions=interventions, narrative=narrative)


class CognitiveTrajectoryEngine:
    def __init__(self):
        pass

    def project(self, age: int = 30, education_years: int = 16,
                exercise_min: int = 90, sleep_hours: float = 7.0,
                cognitive_reserve: float = 0.5, purpose_score: float = 0.5,
                loneliness: float = 0.3, family_history: Dict[str, str] = None) -> CognitiveTrajectoryResult:
        family_history = family_history or {}
        peak_age = 29
        decline_start = 55
        mci_85 = 0.15
        dementia_85 = 0.10
        cognitive_reserve += education_years / 40
        if exercise_min >= 150:
            cognitive_reserve += 0.15
        elif exercise_min >= 90:
            cognitive_reserve += 0.08
        if 7 <= sleep_hours <= 9:
            cognitive_reserve += 0.08
        elif sleep_hours < 6:
            cognitive_reserve -= 0.12
        if purpose_score > 0.7:
            cognitive_reserve += 0.10
        elif purpose_score < 0.3:
            cognitive_reserve -= 0.08
        if loneliness > 0.6:
            cognitive_reserve -= 0.10
        if family_history.get("alzheimer") == "positive":
            mci_85 *= 3.0
            dementia_85 *= 3.0
            decline_start -= 5
        cognitive_reserve = max(0.1, min(1.0, cognitive_reserve))
        mci_85 = min(0.80, mci_85 * (1.5 - cognitive_reserve))
        dementia_85 = min(0.60, dementia_85 * (1.5 - cognitive_reserve))
        trajectory = []
        for yr in range(min(60, max(10, 90 - age))):
            proj_age = age + yr
            if proj_age < peak_age:
                cog = 0.95 + (0.05 * (proj_age - 18) / max(1, peak_age - 18))
            elif proj_age < decline_start:
                cog = 1.0 - (proj_age - peak_age) * 0.001
            else:
                decline_rate = 0.004 * (1 + (1 - cognitive_reserve) * 2)
                years_since = proj_age - decline_start
                cog = 1.0 - decline_start * 0.001 - decline_rate * years_since
                if proj_age > 70:
                    cog -= (proj_age - 70) * 0.003
            cog = max(0.1, min(1.0, cog))
            conf = max(0.3, 0.85 - yr * 0.008)
            phase = "development" if proj_age < 30 else "peak" if proj_age < decline_start else "early_decline" if proj_age < 70 else "accelerated_decline"
            trajectory.append(YearTrajectory(year=2026 + yr, age=proj_age, value=round(float(cog), 4), confidence=round(float(conf), 4), phase=phase))
        brain_interventions = [
            InterventionEffect(name="exercise_150_min", value_added=0.15, confidence=0.8, time_to_effect_years=0.5, evidence="Exercise increases BDNF 200-300%, reduces Alzheimer risk 45% (Podewils et al.)"),
            InterventionEffect(name="sleep_optimization", value_added=0.12, confidence=0.85, time_to_effect_years=0.1, evidence="Glymphatic clearance during sleep removes amyloid-beta; <6h sleep = 1.68x Alzheimer risk (Xie et al. Science 2013)"),
            InterventionEffect(name="cognitive_training", value_added=0.10, confidence=0.75, time_to_effect_years=1.0, evidence="ACTIVE trial: cognitive training improves processing speed, reasoning, delays IADL decline"),
            InterventionEffect(name="mediterranean_diet", value_added=0.08, confidence=0.7, time_to_effect_years=2.0, evidence="MIND diet reduces Alzheimer risk 53% (Morris et al. Alzheimer's & Dementia 2015)"),
            InterventionEffect(name="social_engagement", value_added=0.08, confidence=0.7, time_to_effect_years=1.0, evidence="Social isolation increases dementia risk 50% (Holwerda et al. J Neurol Neurosurg Psychiatry 2014)"),
            InterventionEffect(name="learn_new_skills", value_added=0.12, confidence=0.6, time_to_effect_years=2.0, evidence="Bilingualism delays dementia 4-5 years; musical training builds cognitive reserve (Bialystok et al.)"),
        ]
        if cognitive_reserve < 0.4:
            narrative = f"Cognitive reserve at {cognitive_reserve:.0%} is LOW — this is your brain's resilience buffer against decline. The three highest-ROI investments are exercise (BDNF production), sleep (glymphatic clearance), and learning new skills (neural plasticity)."
        elif cognitive_reserve < 0.6:
            narrative = f"Cognitive reserve at {cognitive_reserve:.0%} is MODERATE — decent but improvable. Your brain will maintain function until ~{decline_start}, then decline accelerates. Every unit of reserve delays dementia onset by 3-5 years."
        else:
            narrative = f"Cognitive reserve at {cognitive_reserve:.0%} is STRONG — your lifestyle and education have built substantial neural backup. Cognitive decline onset estimated at {decline_start}, delayed by {int((cognitive_reserve - 0.3) * 10)} years compared to low-reserve individuals."
        return CognitiveTrajectoryResult(
            peak_cognitive_age=peak_age, cognitive_decline_start_age=decline_start,
            mci_probability_by_85=round(float(mci_85), 3), dementia_probability_by_85=round(float(dementia_85), 3),
            yearly_cognitive=trajectory, brain_health_interventions=brain_interventions, narrative=narrative)