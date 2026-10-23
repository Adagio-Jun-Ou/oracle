from dataclasses import dataclass, field
from typing import Optional
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
class FateState:
    human_population_bn: float = 8.1
    energy_consumption_tw: float = 18.0
    gdp_world_trillion: float = 105.0
    co2_ppm: float = 420.0
    nuclear_warheads: float = 12500.0
    ai_capability_index: float = 0.15
    biodiversity_integrity_index: float = 0.35
    social_cohesion_index: float = 0.45
    technological_dependency: float = 0.75
    resource_remaining_pct: float = 50.0
    climate_commitment_c: float = 1.5
    space_capability: float = 0.1
    biotech_capability: float = 0.2
    governance_effectiveness: float = 0.4
    information_integrity: float = 0.4
    water_stress_global: float = 0.4
    food_system_resilience: float = 0.5
    pandemic_preparedness: float = 0.35
    economic_inequality_gini: float = 0.65
    nuclear_proliferation_risk: float = 0.3
    asteroid_impact_risk: float = 0.0001
    solar_catastrophe_risk: float = 0.012
    supervolcano_probability: float = 0.0007
    gamma_ray_burst_risk: float = 0.0000005


@dataclass
class PersonalFateState:
    age: float = 35.0
    genetic_risk_score: float = 0.3
    socioeconomic_origin: float = 0.5
    education_level: float = 0.5
    health_baseline: float = 0.7
    habit_strength: float = 0.5
    social_capital: float = 0.5
    neuroplasticity: float = 0.6
    trauma_load: float = 0.2
    addiction_vulnerability: float = 0.3
    epigenetic_stress: float = 0.3
    cognitive_reserve: float = 0.5
    relationship_stability: float = 0.5
    financial_safety_net: float = 0.4
    geographic_mobility: float = 0.5
    talent_alignment: float = 0.4
    purpose_clarity: float = 0.3
    intergenerational_mobility: float = 0.4
    chronic_disease_risk: float = 0.3
    mental_health_baseline: float = 0.6
    sleep_quality: float = 0.5
    exercise_habit: float = 0.4
    diet_quality: float = 0.5
    loneliness_index: float = 0.3
    adverse_childhood_experiences: float = 0.2
    zip_code_destiny: float = 0.5


PERSONAL_DETERMINANTS = {
    "genetic": [
        {"name": "APOE4 Alzheimer risk", "contribution": 0.05, "modifiable": False, "description": "Genetic predisposition to late-onset Alzheimer — reduces cognitive reserve ceiling"},
        {"name": "BRCA1/2 cancer susceptibility", "contribution": 0.03, "modifiable": False, "description": "Hereditary cancer risk — significantly elevated lifetime probability"},
        {"name": "MTHFR methylation", "contribution": 0.02, "modifiable": True, "description": "Methylation efficiency affects detox, mood, cardiovascular health — supplementable"},
        {"name": "ACTN3 fast-twitch", "contribution": 0.01, "modifiable": False, "description": "Athletic predisposition toward power vs endurance — influences physical health ceiling"},
        {"name": "HLA immune profile", "contribution": 0.03, "modifiable": False, "description": "Immune system genetics — autoimmune susceptibility and infection resistance"},
        {"name": "Serotonin transporter (5-HTTLPR)", "contribution": 0.04, "modifiable": True, "description": "Stress resilience gene — short allele increases depression risk under adversity"},
        {"name": "Dopamine receptor D2 (DRD2)", "contribution": 0.03, "modifiable": True, "description": "Reward pathway genetics — addiction vulnerability and motivation baseline"},
        {"name": "Telomere length", "contribution": 0.05, "modifiable": True, "description": "Cellular aging clock — shortened telomeres predict earlier disease and mortality"},
    ],
    "epigenetic": [
        {"name": "Childhood adversity methylation", "contribution": 0.06, "modifiable": True, "description": "ACEs permanently alter gene expression — inflammation, cortisol, and immune genes modified for life"},
        {"name": "Nutritional epigenetics", "contribution": 0.03, "modifiable": True, "description": "Diet alters gene expression through methylation and histone modification — leafy greens, B12, folate"},
        {"name": "Stress-induced methylation", "contribution": 0.04, "modifiable": True, "description": "Chronic stress modifies NR3C1 methylation — permanently alters HPA axis reactivity"},
        {"name": "Exercise-induced expression", "contribution": 0.03, "modifiable": True, "description": "Physical activity activates beneficial gene programs — BDNF, IGF-1, anti-inflammatory cascades"},
        {"name": "Intergenerational trauma", "contribution": 0.04, "modifiable": True, "description": "Parental trauma affects offspring epigenetics for 3+ generations — cortisol, metabolism, behavior"},
    ],
    "behavioral": [
        {"name": "Habit loops (basal ganglia)", "contribution": 0.15, "modifiable": True, "description": "40%+ of daily actions are automatic — habit neurology makes deliberate change extremely costly"},
        {"name": "Loss aversion bias", "contribution": 0.08, "modifiable": True, "description": "Kahneman: losses hurt 2x more than gains feel good — creates status quo bias and risk avoidance"},
        {"name": "Temporal discounting", "contribution": 0.10, "modifiable": True, "description": "Future rewards are systematically undervalued — explains obesity, debt, under-saving"},
        {"name": "Confirmation bias", "contribution": 0.06, "modifiable": True, "description": "Information that confirms existing beliefs is preferentially processed — creates ideological lock-in"},
        {"name": "Social conformity pressure", "contribution": 0.07, "modifiable": True, "description": "Asch conformity: 75% will deny obvious truth to conform — peer groups wire your destiny"},
        {"name": "Dunning-Kruger metacognition gap", "contribution": 0.04, "modifiable": True, "description": "Incompetent cannot perceive own incompetence — creates permanent skill plateaus without external feedback"},
        {"name": "Willpower depletion (ego depletion)", "contribution": 0.05, "modifiable": True, "description": "Self-control is a finite daily resource — decisions drain it, creating afternoon vulnerability"},
        {"name": "Negativity bias", "contribution": 0.06, "modifiable": True, "description": "Negative experiences are 3x more impactful than positive ones — shapes risk perception permanently"},
    ],
    "structural": [
        {"name": "Zip code destiny", "contribution": 0.20, "modifiable": False, "description": "Birthplace ZIP code predicts 60%+ of life outcomes — school quality, air quality, social networks, opportunity"},
        {"name": "Intergenerational income elasticity", "contribution": 0.15, "modifiable": False, "description": "In US, ~50% of income advantage transmits to children — your parents' income is your strongest predictor"},
        {"name": "Education pipeline lock-in", "contribution": 0.12, "modifiable": False, "description": "Age-8 reading level predicts college graduation — early tracking creates irreversible path divergence"},
        {"name": "Healthcare access gradient", "contribution": 0.08, "modifiable": False, "description": "Life expectancy varies 15-20 years by county — access to care is structurally determined"},
        {"name": "Social capital inheritance", "contribution": 0.10, "modifiable": True, "description": "Parents' social networks become yours — jobs, opportunities, and mentorship transfer intergenerationally"},
        {"name": "Algorithmic determination", "contribution": 0.05, "modifiable": True, "description": "Credit scores, hiring algorithms, content feeds increasingly determine life paths"},
    ],
}

LIFE_TRAJECTORY_ARCHETYPES = {
    "momentum_continuation": {
        "name": "Momentum Continuation",
        "description": "Most lives follow the trajectory set by age 25 — habits, networks, and skills compound along established paths",
        "probability_range": (0.65, 0.80),
        "modifiable": False,
        "key_factors": ["habit_strength", "social_capital", "socioeconomic_origin"],
    },
    "inflection_point": {
        "name": "Inflection Point",
        "description": "Major life events (marriage, job loss, health crisis, relocation) create bifurcation points where fate branches",
        "probability_range": (0.15, 0.25),
        "modifiable": True,
        "key_factors": ["trauma_load", "geographic_mobility", "relationship_stability"],
    },
    "deliberate_rewrite": {
        "name": "Deliberate Rewrite",
        "description": "Conscious, sustained effort to override default trajectory — requires neuroplasticity, strategy, and rare persistence",
        "probability_range": (0.05, 0.10),
        "modifiable": True,
        "key_factors": ["neuroplasticity", "purpose_clarity", "cognitive_reserve"],
    },
    "structural_escape": {
        "name": "Structural Escape",
        "description": "Escaping structural disadvantage through exceptional talent, mentorship, or institutional intervention — rare but possible",
        "probability_range": (0.02, 0.05),
        "modifiable": True,
        "key_factors": ["talent_alignment", "social_capital", "intergenerational_mobility"],
    },
    "cascade_decline": {
        "name": "Cascade Decline",
        "description": "Negative feedback loops compound — addiction, debt, health decline, social isolation accelerate each other",
        "probability_range": (0.05, 0.15),
        "modifiable": True,
        "key_factors": ["addiction_vulnerability", "chronic_disease_risk", "loneliness_index"],
    },
}

MODIFIABLE_FACTORS = {
    "exercise": {"impact": 0.12, "difficulty": 0.4, "time_to_effect_years": 0.5, "description": "Regular exercise adds 3-7 years of life and dramatically reduces chronic disease risk"},
    "sleep": {"impact": 0.10, "difficulty": 0.3, "time_to_effect_years": 0.1, "description": "7-9 hours of quality sleep is the single highest-ROI health intervention"},
    "nutrition": {"impact": 0.10, "difficulty": 0.5, "time_to_effect_years": 1.0, "description": "Mediterranean-style diet reduces all-cause mortality by 20-30%"},
    "social_connection": {"impact": 0.15, "difficulty": 0.6, "time_to_effect_years": 0.5, "description": "Loneliness is equivalent to smoking 15 cigarettes/day — social connection is a health vital sign"},
    "stress_management": {"impact": 0.08, "difficulty": 0.5, "time_to_effect_years": 0.5, "description": "Meditation, therapy, and stress reduction lower cortisol and inflammation biomarkers measurably"},
    "education_continuous": {"impact": 0.07, "difficulty": 0.6, "time_to_effect_years": 3.0, "description": "Each additional year of education increases lifespan by 1-2 years through cognitive reserve and income"},
    "financial_planning": {"impact": 0.06, "difficulty": 0.4, "time_to_effect_years": 5.0, "description": "Compound interest and emergency savings provide structural resilience against fate's random shocks"},
    "purpose_meaning": {"impact": 0.10, "difficulty": 0.7, "time_to_effect_years": 2.0, "description": "Ikigai/purpose reduces all-cause mortality by 15-30% — meaning is not optional for longevity"},
    "relationship_investment": {"impact": 0.09, "difficulty": 0.5, "time_to_effect_years": 1.0, "description": "Marriage/partnership reduces mortality by 10-15% — relationship quality matters more than quantity"},
    "addiction_abstinence": {"impact": 0.08, "difficulty": 0.8, "time_to_effect_years": 2.0, "description": "Eliminating substance addiction is the single most impactful behavioral change for affected individuals"},
}

UNMODIFIABLE_FACTORS = {
    "genetics": {"impact": 0.20, "description": "Approximately 20% of lifespan variation is heritable — you cannot choose your genes"},
    "birth_era": {"impact": 0.15, "description": "Born in the right decade? Technology, medicine, and economy determine vast swaths of possibility space"},
    "birthplace": {"impact": 0.25, "description": "Country of birth is the single strongest predictor of life outcomes — 25% structural determination"},
    "early_childhood": {"impact": 0.18, "description": "First 5 years of brain development create irreversible neural architecture — ACEs are mostly fated"},
    "parental_income": {"impact": 0.20, "description": "Parents' income quintile predicts yours with 40-50% accuracy in the US — class reproduction is powerful"},
    "gender_race": {"impact": 0.10, "description": "Structural discrimination creates unequal playing fields — wage gaps, health disparities, violence risk"},
}


CIVILIZATIONAL_TRAJECTORIES = {
    "kardashev_i": {
        "name": "Kardashev Type I Achievement",
        "description": "Civilization harnesses all energy reaching Earth from the Sun (~10^16 W) and controls planetary-scale processes",
        "probability_per_year": 0.005,
        "required_energy_tw": 1e7,
        "current_progress": 0.73,
        "timeline_years": "100-200",
        "domains": ["energy", "technology", "infrastructure"],
    },
    "kardashev_ii": {
        "name": "Kardashev Type II Achievement",
        "description": "Civilization harnesses total stellar energy output (~10^26 W) via Dyson sphere or similar megastructure",
        "probability_per_year": 1e-6,
        "required_energy_tw": 1e17,
        "current_progress": 1e-10,
        "timeline_years": "10,000-1,000,000",
        "domains": ["energy", "technology", "space"],
    },
    "great_filter": {
        "name": "The Great Filter",
        "description": "Civilization encounters an existential barrier that prevents transition to multi-planetary or interstellar civilization",
        "probability_per_year": 0.001,
        "current_progress": 0.0,
        "timeline_years": "Unknown — could be now or millennia away",
        "domains": ["all"],
    },
    "extinction_earth_bound": {
        "name": "Earth-Bound Extinction",
        "description": "Civilization collapses and species goes extinct without achieving multi-planetary status",
        "probability_per_year": 0.0003,
        "current_progress": 0.0,
        "timeline_years": "100-10,000 years if current trajectory continues",
        "domains": ["infrastructure", "environment", "geopolitical"],
    },
    "transcension": {
        "name": "Transcension Hypothesis",
        "description": "Advanced civilizations compress inward rather than expand outward, creating inner space rather than outer space",
        "probability_per_year": 1e-5,
        "current_progress": 0.0,
        "timeline_years": "Unknown — speculative",
        "domains": ["technology", "consciousness"],
    },
    "multi_planetary": {
        "name": "Multi-Planetary Civilization",
        "description": "Self-sustaining colonies on Mars, Moon, or orbital habitats — not dependent on single planet",
        "probability_per_year": 0.002,
        "current_progress": 0.01,
        "timeline_years": "50-200",
        "domains": ["technology", "space", "energy"],
    },
}

THERMODYNAMIC_CONSTRAINTS = {
    "entropy_increase": {
        "name": "Entropy Always Increases",
        "description": "Second law of thermodynamics — all ordered systems trend toward disorder, limiting complexity maintenance",
        "implication": "Civilization must continuously expend energy to maintain order; efficiency has fundamental limits",
        "timeline": "eternal",
        "certainty": 1.0,
    },
    "energy_return_on_investment": {
        "name": "EROI Decline",
        "description": "Energy returned on energy invested declines as easiest resources are consumed first",
        "implication": "Net energy available for civilization decreases over time unless new sources are found",
        "timeline": "decades",
        "certainty": 0.95,
    },
    "carrying_capacity": {
        "name": "Carrying Capacity Limits",
        "description": "Earth has finite carrying capacity for any species — exponential growth eventually encounters hard limits",
        "implication": "Population growth must stabilize; overshoot leads to die-off",
        "timeline": "centuries",
        "certainty": 0.99,
    },
    "cosmic_horizon": {
        "name": "Observable Universe Finite",
        "description": "The observable universe has finite energy and resources (~10^80 atoms, ~10^12 galaxies)",
        "implication": "Even a Type III civilization faces fundamental resource constraints",
        "timeline": "billions of years",
        "certainty": 1.0,
    },
    "heat_death": {
        "name": "Heat Death (Big Freeze)",
        "description": "Universe expands forever, entropy reaches maximum, all structure dissolves",
        "implication": "Ultimate fate of all information and complexity — no escape from thermodynamic endgame",
        "timeline": "10^100 years",
        "certainty": 0.99,
    },
}

RESOURCE_DEPLETION_CURVES = {
    "oil": {"peak_year": 2025, "decline_rate": 0.03, "remaining_pct": 0.45, "critical_threshold": 0.1, "substitute_readiness": 0.6},
    "natural_gas": {"peak_year": 2035, "decline_rate": 0.02, "remaining_pct": 0.55, "critical_threshold": 0.15, "substitute_readiness": 0.5},
    "coal": {"peak_year": 2040, "decline_rate": 0.02, "remaining_pct": 0.65, "critical_threshold": 0.2, "substitute_readiness": 0.7},
    "phosphorus": {"peak_year": 2040, "decline_rate": 0.015, "remaining_pct": 0.50, "critical_threshold": 0.1, "substitute_readiness": 0.2},
    "rare_earth": {"peak_year": 2050, "decline_rate": 0.025, "remaining_pct": 0.60, "critical_threshold": 0.15, "substitute_readiness": 0.3},
    "fresh_water": {"peak_year": 2030, "decline_rate": 0.01, "remaining_pct": 0.40, "critical_threshold": 0.1, "substitute_readiness": 0.4},
    "topsoil": {"peak_year": 2020, "decline_rate": 0.008, "remaining_pct": 0.50, "critical_threshold": 0.2, "substitute_readiness": 0.15},
    "biodiversity": {"peak_year": 2000, "decline_rate": 0.02, "remaining_pct": 0.30, "critical_threshold": 0.15, "substitute_readiness": 0.05},
    "fish_stocks": {"peak_year": 1990, "decline_rate": 0.03, "remaining_pct": 0.25, "critical_threshold": 0.1, "substitute_readiness": 0.35},
    "uranium": {"peak_year": 2060, "decline_rate": 0.01, "remaining_pct": 0.70, "critical_threshold": 0.2, "substitute_readiness": 0.8},
    "lithium": {"peak_year": 2055, "decline_rate": 0.02, "remaining_pct": 0.65, "critical_threshold": 0.15, "substitute_readiness": 0.3},
    "copper": {"peak_year": 2045, "decline_rate": 0.018, "remaining_pct": 0.55, "critical_threshold": 0.15, "substitute_readiness": 0.4},
}

DEMOGRAPHIC_DESTINIES = {
    "population_peak": {"year_range": (2050, 2100), "peak_bn": (9.5, 11.0), "probability": 0.85, "description": "Global population peaks this century then declines"},
    "aging_crisis": {"year_range": (2030, 2060), "peak_dependency_ratio": (0.6, 0.8), "probability": 0.9, "description": "Working-age population insufficient to support elderly — structural economic strain"},
    "urbanization_saturation": {"year_range": (2040, 2070), "saturation_pct": (0.75, 0.85), "probability": 0.95, "description": "Urbanization reaches saturation — rural depopulation completes in developed nations"},
    "longevity_escape_velocity": {"year_range": (2050, 2100), "probability": 0.15, "description": "Medical technology extends healthy lifespan faster than aging advances"},
    "demographic_transition_completion": {"year_range": (2070, 2120), "probability": 0.9, "description": "All nations complete demographic transition — fertility stabilizes below replacement globally"},
    "population_decline_acceleration": {"year_range": (2080, 2150), "probability": 0.6, "description": "Post-peak population decline accelerates as fertility drops below replacement in most nations"},
}

BIOLOGICAL_DESTINIES = {
    "antibiotic_resistance_crisis": {"year_range": (2025, 2050), "probability": 0.7, "description": "Post-antibiotic era — common infections become untreatable, surgery risk increases dramatically"},
    "genetic_engineering_maturity": {"year_range": (2030, 2060), "probability": 0.85, "description": "Gene editing becomes routine — disease elimination and enhancement become viable"},
    "synthetic_biology_risk": {"year_range": (2025, 2050), "probability": 0.3, "description": "Engineered pathogen or accidental release creates novel biological threat"},
    "brain_computer_interface": {"year_range": (2040, 2080), "probability": 0.6, "description": "Direct brain-computer interfaces become reliable — cognition augmented by AI"},
    "longevity_divergence": {"year_range": (2040, 2080), "probability": 0.5, "description": "Life expectancy gaps widen between those with and without access to advanced medicine"},
    "digital_consciousness": {"year_range": (2080, 2200), "probability": 0.05, "description": "Consciousness becomes uploadable/duplicable — fundamental questions of identity and rights emerge"},
}

COSMOLOGICAL_TIMELINE = [
    {"event": "Current era — Anthropocene", "time": "now", "significance": "Civilization exists; all outcomes remain possible", "probability": 1.0},
    {"event": "AI capability singularity window", "time": "2030-2070", "significance": "Artificial general intelligence may emerge, transforming civilization", "probability": 0.35},
    {"event": "Nuclear war probability window", "time": "2025-2100", "significance": "Peak nuclear conflict risk during geopolitical transition", "probability": 0.15},
    {"event": "Climate tipping cascade window", "time": "2030-2100", "significance": "Multiple climate tipping points likely to activate", "probability": 0.5},
    {"event": "Population peak", "time": "2050-2100", "significance": "Human population reaches maximum then declines", "probability": 0.85},
    {"event": "Kardashev Type I achievement window", "time": "2100-2300", "significance": "Civilization controls all planetary energy", "probability": 0.3},
    {"event": "Andromeda-Milky Way collision", "time": "4.5 billion years", "significance": "Galaxies merge — solar system likely undisturbed", "probability": 1.0},
    {"event": "Sun becomes red giant", "time": "5 billion years", "significance": "Earth becomes uninhabitable — oceans evaporate", "probability": 1.0},
    {"event": "Sun white dwarf", "time": "8 billion years", "significance": "Solar system enters final phase", "probability": 1.0},
    {"event": "Proton decay (if exists)", "time": "10^34 years", "significance": "All matter potentially unstable — end of atomic era", "probability": 0.5},
    {"event": "Heat death", "time": "10^100 years", "significance": "Maximum entropy — no structure, no information, no complexity", "probability": 0.99},
]


class FateEngine:

    def __init__(self, seed: int = 42):
        self.name = "fate_engine"
        self.seed = seed

    def assess_civilizational_trajectory(self, state: FateState = None) -> dict:
        if state is None:
            state = FateState()
        rng = np.random.RandomState(self.seed)

        energy_kardashev = np.clip(np.log10(max(1, state.energy_consumption_tw)) / 16.0, 0, 1)
        tech_progress = np.clip((state.ai_capability_index + state.biotech_capability + state.space_capability) / 3, 0, 1)
        social_capacity = np.clip(state.social_cohesion_index * state.governance_effectiveness, 0, 1)
        resource_base = np.clip(state.resource_remaining_pct / 100.0, 0, 1)
        environmental_constraint = np.clip(1.0 - state.climate_commitment_c / 4.0, 0, 1)
        existential_risk = np.clip(
            state.nuclear_proliferation_risk * 0.4 +
            (1.0 - state.pandemic_preparedness) * 0.2 +
            state.asteroid_impact_risk * 100 * 0.05 +
            state.solar_catastrophe_risk * 10 * 0.05 +
            state.ai_capability_index * 0.2 +
            (1.0 - state.information_integrity) * 0.1,
            0, 1
        )
        trajectory_score = (
            energy_kardashev * 0.2 +
            tech_progress * 0.25 +
            social_capacity * 0.2 +
            resource_base * 0.15 +
            environmental_constraint * 0.1 +
            (1.0 - existential_risk) * 0.1
        )
        trajectory_score = float(np.clip(trajectory_score, 0, 1))

        if trajectory_score > 0.7:
            trajectory_label = "ascending"
        elif trajectory_score > 0.5:
            trajectory_label = "plateau"
        elif trajectory_score > 0.35:
            trajectory_label = "declining"
        else:
            trajectory_label = "existential_risk"

        trajectories = []
        for tid, tdef in CIVILIZATIONAL_TRAJECTORIES.items():
            prob_modifier = trajectory_score if tid == "kardashev_i" else (1.0 - trajectory_score if tid == "extinction_earth_bound" else 0.5)
            trajectories.append({
                "id": tid,
                "name": tdef["name"],
                "description": tdef["description"],
                "current_progress": tdef["current_progress"],
                "timeline_years": tdef["timeline_years"],
                "relevance": round(float(np.clip(prob_modifier, 0, 1)), 3),
            })

        return sanitize_nan({
            "trajectory_score": round(trajectory_score, 3),
            "trajectory_label": trajectory_label,
            "kardashev_scale_current": round(float(np.log10(max(1, state.energy_consumption_tw * 1e12)) / 16 + 0.73), 3),
            "energy_progress_to_type_i_pct": round(energy_kardashev * 100, 1),
            "technology_capacity": round(tech_progress, 3),
            "social_capacity": round(social_capacity, 3),
            "resource_base": round(resource_base, 3),
            "environmental_constraint": round(environmental_constraint, 3),
            "existential_risk_composite": round(existential_risk, 3),
            "trajectories": trajectories,
            "fate_commitment": round(float(np.clip(1.0 - trajectory_score + existential_risk * 0.5, 0, 1)), 3),
            "recommendations": self._trajectory_recommendations(trajectory_score, existential_risk, social_capacity),
        })

    def assess_thermodynamic_fate(self, state: FateState = None) -> dict:
        if state is None:
            state = FateState()
        rng = np.random.RandomState(self.seed)

        eroi_current = np.clip(30.0 * state.resource_remaining_pct / 50.0, 1, 100)
        eroi_declining = eroi_current < 10
        carrying_capacity_usage = np.clip(state.human_population_bn / 10.0, 0, 2)
        overshoot = carrying_capacity_usage > 1.0
        energy_efficiency = np.clip(state.energy_consumption_tw / (state.gdp_world_trillion * 0.17), 0.5, 3.0)

        entropy_pressure = np.clip(
            (1.0 - state.resource_remaining_pct / 100.0) * 0.4 +
            state.human_population_bn / 15.0 * 0.3 +
            state.co2_ppm / 1000.0 * 0.3,
            0, 1
        )

        constraints = []
        for cid, cdef in THERMODYNAMIC_CONSTRAINTS.items():
            bound = "hard" if cdef["certainty"] >= 0.99 else "soft"
            active = False
            if cid == "entropy_increase":
                active = True
            elif cid == "energy_return_on_investment":
                active = eroi_current < 15
            elif cid == "carrying_capacity":
                active = overshoot
            elif cid == "cosmic_horizon":
                active = False
            elif cid == "heat_death":
                active = False

            constraints.append({
                "id": cid,
                "name": cdef["name"],
                "description": cdef["description"],
                "implication": cdef["implication"],
                "timeline": cdef["timeline"],
                "certainty": cdef["certainty"],
                "bound": bound,
                "currently_active": active,
            })

        active_count = sum(1 for c in constraints if c["currently_active"])
        hard_count = sum(1 for c in constraints if c["bound"] == "hard")

        return sanitize_nan({
            "thermodynamic_pressure": round(entropy_pressure, 3),
            "pressure_label": "critical" if entropy_pressure > 0.7 else "high" if entropy_pressure > 0.5 else "moderate" if entropy_pressure > 0.3 else "manageable",
            "eroi_current": round(float(eroi_current), 1),
            "eroi_threshold": 10,
            "eroi_declining": eroi_declining,
            "carrying_capacity_usage": round(float(carrying_capacity_usage), 3),
            "overshoot": overshoot,
            "energy_intensity": round(float(energy_efficiency), 3),
            "n_active_constraints": active_count,
            "n_hard_constraints": hard_count,
            "constraints": constraints,
            "implication": "Thermodynamic constraints are binding — energy efficiency and resource substitution are the only paths forward" if entropy_pressure > 0.5 else "Thermodynamic pressure is manageable but requires long-term planning",
            "recommendations": self._thermodynamic_recommendations(entropy_pressure, eroi_current, overshoot),
        })

    def assess_resource_depletion(self, state: FateState = None, years_ahead: int = 50) -> dict:
        if state is None:
            state = FateState()
        rng = np.random.RandomState(self.seed)

        depletions = []
        critical_resources = []
        substitute_gap_resources = []
        total_pressure = 0.0

        for rid, rdef in RESOURCE_DEPLETION_CURVES.items():
            remaining = rdef["remaining_pct"] * state.resource_remaining_pct / 50.0
            remaining = max(0.01, min(1.0, remaining))
            years_to_peak = max(0, rdef["peak_year"] - 2025)
            years_since_peak = max(0, 2025 - rdef["peak_year"])
            decline_years = years_since_peak + years_ahead
            future_remaining = max(0.01, remaining * np.exp(-rdef["decline_rate"] * years_ahead))
            depletion_pressure = 1.0 - remaining

            if remaining < rdef["critical_threshold"]:
                critical_resources.append(rid)
            if rdef["substitute_readiness"] < 0.3:
                substitute_gap_resources.append(rid)

            total_pressure += depletion_pressure * (1.0 - rdef["substitute_readiness"])

            depletions.append({
                "resource": rid,
                "remaining_pct": round(remaining * 100, 1),
                "future_remaining_pct": round(future_remaining * 100, 1),
                "peak_year": rdef["peak_year"],
                "years_past_peak": years_since_peak,
                "decline_rate_annual": rdef["decline_rate"],
                "critical_threshold_pct": rdef["critical_threshold"] * 100,
                "substitute_readiness": rdef["substitute_readiness"],
                "depletion_pressure": round(float(depletion_pressure), 3),
                "status": "critical" if remaining < rdef["critical_threshold"] else "depleting" if remaining < 0.3 else "moderate" if remaining < 0.5 else "stable",
            })

        depletions.sort(key=lambda x: x["depletion_pressure"], reverse=True)
        total_pressure /= len(RESOURCE_DEPLETION_CURVES)

        return sanitize_nan({
            "overall_depletion_pressure": round(float(total_pressure), 3),
            "pressure_label": "critical" if total_pressure > 0.5 else "high" if total_pressure > 0.35 else "moderate" if total_pressure > 0.2 else "manageable",
            "n_critical_resources": len(critical_resources),
            "n_substitute_gap": len(substitute_gap_resources),
            "critical_resources": critical_resources,
            "substitute_gap_resources": substitute_gap_resources,
            "depletions": depletions[:6],
            "years_ahead": years_ahead,
            "recommendations": self._depletion_recommendations(total_pressure, critical_resources, substitute_gap_resources),
        })

    def assess_demographic_destiny(self, state: FateState = None) -> dict:
        if state is None:
            state = FateState()
        rng = np.random.RandomState(self.seed)

        destinies = []
        for did, ddef in DEMOGRAPHIC_DESTINIES.items():
            prob_modifier = 1.0
            if did == "population_peak":
                prob_modifier = np.clip(state.human_population_bn / 10.0, 0.5, 1.2)
            elif did == "aging_crisis":
                prob_modifier = np.clip(1.0 - state.social_cohesion_index + 0.5, 0.5, 1.5)
            elif did == "urbanization_saturation":
                prob_modifier = 1.0
            elif did == "longevity_escape_velocity":
                prob_modifier = np.clip(state.biotech_capability * 3, 0, 1.5)
            elif did == "demographic_transition_completion":
                prob_modifier = np.clip(state.gdp_world_trillion / 150.0, 0.5, 1.3)
            elif did == "population_decline_acceleration":
                prob_modifier = np.clip(1.0 - state.resource_remaining_pct / 100.0 + 0.3, 0.5, 1.5)

            adjusted_prob = np.clip(ddef["probability"] * prob_modifier, 0, 1)
            destinies.append({
                "id": did,
                "description": ddef["description"],
                "probability": round(float(adjusted_prob), 3),
                "year_range": ddef["year_range"],
                "inevitability": "near_certain" if adjusted_prob > 0.8 else "likely" if adjusted_prob > 0.5 else "possible" if adjusted_prob > 0.2 else "unlikely",
            })

        destinies.sort(key=lambda x: x["probability"], reverse=True)

        dependency_ratio = np.clip(
            (1.0 - state.social_cohesion_index) * 0.5 + state.economic_inequality_gini * 0.3 + 0.2,
            0.2, 0.9
        )

        return sanitize_nan({
            "dependency_ratio_outlook": round(float(dependency_ratio), 3),
            "population_bn": state.human_population_bn,
            "inequality_gini": state.economic_inequality_gini,
            "destinies": destinies,
            "most_likely_destiny": destinies[0]["id"] if destinies else "unknown",
            "recommendations": self._demographic_recommendations(state, dependency_ratio),
        })

    def assess_biological_destiny(self, state: FateState = None) -> dict:
        if state is None:
            state = FateState()
        rng = np.random.RandomState(self.seed)

        destinies = []
        for bid, bdef in BIOLOGICAL_DESTINIES.items():
            prob_modifier = 1.0
            if bid == "antibiotic_resistance_crisis":
                prob_modifier = np.clip(1.0 - state.pandemic_preparedness + 0.5, 0.5, 1.5)
            elif bid == "genetic_engineering_maturity":
                prob_modifier = np.clip(state.biotech_capability * 2 + 0.5, 0.5, 1.5)
            elif bid == "synthetic_biology_risk":
                prob_modifier = np.clip(state.biotech_capability * 2, 0.3, 1.5)
            elif bid == "brain_computer_interface":
                prob_modifier = np.clip(state.ai_capability_index * 2 + 0.3, 0.3, 1.5)
            elif bid == "longevity_divergence":
                prob_modifier = np.clip(state.economic_inequality_gini + 0.3, 0.5, 1.5)
            elif bid == "digital_consciousness":
                prob_modifier = np.clip(state.ai_capability_index * 3, 0, 1.0)

            adjusted_prob = np.clip(bdef["probability"] * prob_modifier, 0, 1)
            destinies.append({
                "id": bid,
                "description": bdef["description"],
                "probability": round(float(adjusted_prob), 3),
                "year_range": bdef["year_range"],
                "inevitability": "near_certain" if adjusted_prob > 0.8 else "likely" if adjusted_prob > 0.5 else "possible" if adjusted_prob > 0.2 else "speculative",
            })

        destinies.sort(key=lambda x: x["probability"], reverse=True)

        bio_advancement = np.clip((state.biotech_capability + state.ai_capability_index) / 2, 0, 1)
        bio_risk = np.clip(
            (1.0 - state.pandemic_preparedness) * 0.4 +
            state.biotech_capability * 0.3 +
            (1.0 - state.information_integrity) * 0.3,
            0, 1
        )

        return sanitize_nan({
            "biological_advancement_index": round(float(bio_advancement), 3),
            "biological_risk_index": round(float(bio_risk), 3),
            "biodiversity_integrity": state.biodiversity_integrity_index,
            "destinies": destinies,
            "most_likely_destiny": destinies[0]["id"] if destinies else "unknown",
            "recommendations": self._biological_recommendations(bio_advancement, bio_risk, state),
        })

    def assess_deterministic_commitments(self, state: FateState = None) -> dict:
        if state is None:
            state = FateState()
        rng = np.random.RandomState(self.seed)

        commitments = []
        commitments.append({
            "name": "Climate Commitment",
            "description": "Current CO2 concentration commits Earth to ~1.5-2°C warming regardless of future emissions",
            "committed_warming_c": round(state.climate_commitment_c, 1),
            "irreversibility": 0.95,
            "timeline": "decades to centuries",
            "probability": 0.99,
        })
        commitments.append({
            "name": "Sea Level Rise Commitment",
            "description": "Thermal inertia and ice sheet dynamics commit to multiple meters of sea level rise",
            "committed_rise_m": round(0.3 + state.climate_commitment_c * 0.8, 1),
            "irreversibility": 0.9,
            "timeline": "centuries to millennia",
            "probability": 0.95,
        })
        commitments.append({
            "name": "Biodiversity Loss Commitment",
            "description": f"Extinction debt from current habitat loss: additional species committed to extinction",
            "committed_extinction_fraction": round(float(np.clip(1.0 - state.biodiversity_integrity_index, 0.1, 0.9)), 2),
            "irreversibility": 1.0,
            "timeline": "decades to centuries",
            "probability": 0.95,
        })
        commitments.append({
            "name": "Nuclear Arsenal Commitment",
            "description": f"Current nuclear warhead count maintains existential risk indefinitely unless verifiably dismantled",
            "warhead_count": int(state.nuclear_warheads),
            "irreversibility": 0.6,
            "timeline": "ongoing until disarmament",
            "probability": 1.0,
        })
        commitments.append({
            "name": "Population Momentum Commitment",
            "description": "Demographic inertia commits to continued population growth even if fertility drops below replacement",
            "peak_population_bn": round(state.human_population_bn * 1.2, 1),
            "irreversibility": 0.8,
            "timeline": "2-4 decades",
            "probability": 0.85,
        })
        commitments.append({
            "name": "Infrastructure Lock-in",
            "description": "Existing fossil fuel infrastructure commits to future emissions unless stranded assets are accepted",
            "estimated_committed_emissions_gt_co2": round(float(500 + state.co2_ppm * 0.5), 0),
            "irreversibility": 0.7,
            "timeline": "decades",
            "probability": 0.8,
        })
        commitments.append({
            "name": "AI Trajectory Commitment",
            "description": "AI capability development has momentum — current trajectory commits to increasing AI autonomy",
            "current_capability": round(state.ai_capability_index, 2),
            "irreversibility": 0.5,
            "timeline": "5-20 years",
            "probability": 0.7,
        })
        commitments.append({
            "name": "Topsoil Loss Commitment",
            "description": "Accelerating topsoil loss commits to reduced agricultural capacity",
            "irreversibility": 0.85,
            "timeline": "centuries for natural regeneration",
            "probability": 0.9,
        })
        commitments.append({
            "name": "Ocean Acidification Commitment",
            "description": "Absorbed CO2 has committed oceans to continued acidification",
            "committed_ph_drop": round(float(np.clip(0.05 + state.co2_ppm * 0.00005, 0.05, 0.3)), 3),
            "irreversibility": 0.85,
            "timeline": "centuries",
            "probability": 0.95,
        })

        commitments.sort(key=lambda x: x.get("irreversibility", 0), reverse=True)

        total_irreversibility = float(np.mean([c.get("irreversibility", 0) for c in commitments]))
        high_irreversibility = [c for c in commitments if c.get("irreversibility", 0) >= 0.9]

        return sanitize_nan({
            "n_commitments": len(commitments),
            "avg_irreversibility": round(total_irreversibility, 3),
            "n_high_irreversibility": len(high_irreversibility),
            "high_irreversibility_commitments": [c["name"] for c in high_irreversibility],
            "commitments": commitments,
            "fate_coupling": round(float(np.clip(total_irreversibility * (1.0 - state.resource_remaining_pct / 200.0), 0, 1)), 3),
            "recommendations": self._commitment_recommendations(total_irreversibility, high_irreversibility, state),
        })

    def assess_cosmological_timeline(self, state: FateState = None) -> dict:
        if state is None:
            state = FateState()
        rng = np.random.RandomState(self.seed)

        timeline = []
        for event in COSMOLOGICAL_TIMELINE:
            prob_modifier = 1.0
            if "nuclear" in event["event"].lower():
                prob_modifier = np.clip(state.nuclear_proliferation_risk * 2 + 0.5, 0.5, 2.0)
            elif "climate" in event["event"].lower():
                prob_modifier = np.clip(state.climate_commitment_c, 0.5, 1.5)
            elif "AI" in event["event"] or "singularity" in event["event"].lower():
                prob_modifier = np.clip(state.ai_capability_index * 3 + 0.5, 0.5, 2.0)

            adjusted_prob = np.clip(event["probability"] * prob_modifier, 0, 1)
            timeline.append({
                "event": event["event"],
                "time": event["time"],
                "significance": event["significance"],
                "probability": round(float(adjusted_prob), 3),
                "fate_category": "hard_determinism" if adjusted_prob > 0.9 else "probable_determinism" if adjusted_prob > 0.5 else "contingent",
            })

        near_term = [e for e in timeline if e["fate_category"] != "hard_determinism" or "billion" not in e["time"]]
        existential = [e for e in timeline if "nuclear" in e["event"].lower() or "existential" in e.get("significance", "").lower() or e.get("probability", 0) < 0.2]

        return sanitize_nan({
            "timeline": timeline,
            "near_term_fates": [e["event"] for e in near_term[:5]],
            "existential_risks": [e["event"] for e in existential],
            "long_term_trajectory": "Type I achievable" if state.energy_consumption_tw > 1e4 else "extinction risk elevated" if state.nuclear_proliferation_risk > 0.5 else "uncertain — contingent on choices",
            "cosmological_awareness": round(float(np.clip(state.space_capability + state.ai_capability_index / 4, 0, 1)), 3),
            "recommendations": self._cosmological_recommendations(state),
        })

    def predict_fate(self, state: FateState = None) -> dict:
        if state is None:
            state = FateState()

        trajectory = self.assess_civilizational_trajectory(state)
        thermodynamic = self.assess_thermodynamic_fate(state)
        depletion = self.assess_resource_depletion(state)
        demographic = self.assess_demographic_destiny(state)
        biological = self.assess_biological_destiny(state)
        commitment = self.assess_deterministic_commitments(state)
        cosmological = self.assess_cosmological_timeline(state)

        fate_score = np.clip(
            trajectory.get("fate_commitment", 0.5) * 0.3 +
            thermodynamic.get("thermodynamic_pressure", 0.3) * 0.2 +
            depletion.get("overall_depletion_pressure", 0.3) * 0.15 +
            (1.0 - demographic.get("destinies", [{}])[0].get("probability", 0.5) if demographic.get("destinies") else 0.5) * 0.1 +
            biological.get("biological_risk_index", 0.3) * 0.1 +
            commitment.get("fate_coupling", 0.5) * 0.15,
            0, 1
        )

        if fate_score < 0.3:
            fate_label = "open_possibility"
        elif fate_score < 0.5:
            fate_label = "constrained_freedom"
        elif fate_score < 0.7:
            fate_label = "probable_destiny"
        elif fate_score < 0.85:
            fate_label = "near_determinism"
        else:
            fate_label = "locked_in"

        return sanitize_nan({
            "fate_score": round(float(fate_score), 3),
            "fate_label": fate_label,
            "trajectory": trajectory,
            "thermodynamic": thermodynamic,
            "depletion": depletion,
            "demographic": demographic,
            "biological": biological,
            "commitment": commitment,
            "cosmological": cosmological,
            "key_fates": [
                commitment.get("high_irreversibility_commitments", [])[:3],
                [d["id"] for d in demographic.get("destinies", [])[:2] if d.get("probability", 0) > 0.7],
                [b["id"] for b in biological.get("destinies", [])[:2] if b.get("probability", 0) > 0.5],
            ],
            "escape_velocities": self._escape_velocities(state, trajectory),
            "recommendations": self._fate_recommendations(fate_score, trajectory, thermodynamic, commitment),
        })

    def generate_fate_scenario(self, scenario: str = "default", state: FateState = None, years: int = 100) -> dict:
        if state is None:
            state = FateState()
        rng = np.random.RandomState(self.seed)

        scenarios = {
            "default": {"stability_base": 0.6, "shock_frequency": 0.03, "domain_modifiers": {}},
            "collapse": {"stability_base": 0.3, "shock_frequency": 0.08, "domain_modifiers": {"social_cohesion": -0.3, "governance": -0.3, "resources": -0.2}},
            "resilience": {"stability_base": 0.8, "shock_frequency": 0.01, "domain_modifiers": {"social_cohesion": 0.2, "governance": 0.2, "resources": 0.1}},
            "singularitarian": {"stability_base": 0.5, "shock_frequency": 0.05, "domain_modifiers": {"ai_capability": 0.5, "biotech": 0.3, "social_cohesion": -0.2}},
            "extinction": {"stability_base": 0.1, "shock_frequency": 0.15, "domain_modifiers": {"nuclear": 0.5, "pandemic": 0.3, "resources": -0.4}},
        }

        config = scenarios.get(scenario, scenarios["default"])
        base_stability = config["stability_base"]
        stability = base_stability
        trajectory = []
        rng = np.random.RandomState(self.seed)

        for yr in range(min(years, 500)):
            shock = rng.random() < config["shock_frequency"]
            if shock:
                stability -= rng.uniform(0.05, 0.2)
            else:
                stability += rng.uniform(0.001, 0.01)
            stability = float(np.clip(stability, 0, 1))
            trajectory.append({"year": 2025 + yr, "stability": round(stability, 3)})

        final_stability = trajectory[-1]["stability"] if trajectory else base_stability
        min_stability = min(t["stability"] for t in trajectory) if trajectory else base_stability

        if min_stability < 0.05:
            outcome = "civilizational_collapse"
        elif min_stability < 0.2:
            outcome = "near_extinction"
        elif min_stability < 0.4:
            outcome = "severe_crisis"
        elif final_stability > 0.7:
            outcome = "sustainable_advancement"
        elif final_stability > 0.5:
            outcome = "muddling_through"
        else:
            outcome = "gradual_decline"

        return sanitize_nan({
            "scenario": scenario,
            "years_simulated": len(trajectory),
            "initial_stability": round(base_stability, 3),
            "final_stability": round(final_stability, 3),
            "minimum_stability": round(min_stability, 3),
            "outcome": outcome,
            "trajectory_sample": trajectory[::max(1, len(trajectory) // 20)],
            "n_shocks": sum(1 for t in trajectory if t["stability"] < trajectory[max(0, trajectory.index(t) - 1)]["stability"] - 0.04) if len(trajectory) > 1 else 0,
        })

    def assess_personal_destiny(self, state: PersonalFateState = None) -> dict:
        if state is None:
            state = PersonalFateState()
        rng = np.random.RandomState(self.seed)

        genetic_determinism = np.clip(
            state.genetic_risk_score * 0.3 +
            (1.0 - state.health_baseline) * 0.2 +
            state.addiction_vulnerability * 0.2 +
            state.chronic_disease_risk * 0.15 +
            (1.0 - state.neuroplasticity) * 0.15,
            0, 1
        )

        structural_determinism = np.clip(
            (1.0 - state.socioeconomic_origin) * 0.3 +
            (1.0 - state.intergenerational_mobility) * 0.25 +
            (1.0 - state.zip_code_destiny) * 0.2 +
            (1.0 - state.financial_safety_net) * 0.15 +
            (1.0 - state.education_level) * 0.1,
            0, 1
        )

        behavioral_determinism = np.clip(
            state.habit_strength * 0.25 +
            state.loneliness_index * 0.2 +
            (1.0 - state.purpose_clarity) * 0.2 +
            (1.0 - state.talent_alignment) * 0.2 +
            state.adverse_childhood_experiences * 0.15,
            0, 1
        )

        epigenetic_determinism = np.clip(
            state.epigenetic_stress * 0.35 +
            state.trauma_load * 0.3 +
            state.adverse_childhood_experiences * 0.2 +
            (1.0 - state.diet_quality) * 0.15,
            0, 1
        )

        total_determinism = float(np.clip(
            genetic_determinism * 0.25 +
            structural_determinism * 0.30 +
            behavioral_determinism * 0.25 +
            epigenetic_determinism * 0.20,
            0, 1
        ))

        agency_score = float(np.clip(1.0 - total_determinism + 
            state.neuroplasticity * 0.1 +
            state.cognitive_reserve * 0.1 +
            state.purpose_clarity * 0.1 +
            state.geographic_mobility * 0.05 +
            state.relationship_stability * 0.05,
            0, 1))

        if total_determinism > 0.7:
            destiny_label = "heavily_determined"
        elif total_determinism > 0.5:
            destiny_label = "largely_constrained"
        elif total_determinism > 0.35:
            destiny_label = "partially_destined"
        elif total_determinism > 0.2:
            destiny_label = "significant_agency"
        else:
            destiny_label = "open_trajectory"

        life_expectancy_modifier = np.clip(
            state.health_baseline * 0.2 +
            state.exercise_habit * 0.1 +
            state.diet_quality * 0.1 +
            state.sleep_quality * 0.1 +
            (1.0 - state.chronic_disease_risk) * 0.15 +
            (1.0 - state.addiction_vulnerability) * 0.1 +
            state.mental_health_baseline * 0.1 +
            (1.0 - state.loneliness_index) * 0.15,
            0.3, 1.2
        )
        base_life_expectancy = 73.0
        adjusted_life_expectancy = base_life_expectancy * life_expectancy_modifier

        determinants = {}
        for category, factors in PERSONAL_DETERMINANTS.items():
            cat_determinism = np.mean([f["contribution"] for f in factors])
            modifiable_pct = np.mean([1.0 if f["modifiable"] else 0.0 for f in factors])
            determinants[category] = {
                "factors": factors,
                "aggregate_determinism": round(float(cat_determinism), 3),
                "modifiable_pct": round(float(modifiable_pct * 100), 1),
            }

        trajectory_archetypes = []
        for aid, adef in LIFE_TRAJECTORY_ARCHETYPES.items():
            prob_range = adef["probability_range"]
            key_factors_vals = [getattr(state, f, 0.5) for f in adef["key_factors"]]
            avg_factor = float(np.mean(key_factors_vals))
            prob = float(np.clip(
                (prob_range[0] + prob_range[1]) / 2 * (1 + (0.5 - avg_factor) * 0.5),
                prob_range[0], prob_range[1]
            ))
            trajectory_archetypes.append({
                "id": aid,
                "name": adef["name"],
                "description": adef["description"],
                "personal_probability": round(prob, 3),
                "modifiable": adef["modifiable"],
                "key_factors": adef["key_factors"],
            })

        free_will_domains = []
        modifiable_factors = sorted(MODIFIABLE_FACTORS.items(), key=lambda x: x[1]["impact"], reverse=True)
        for fname, fdef in modifiable_factors:
            current = getattr(state, fname if hasattr(state, fname) else "exercise_habit", 0.5)
            free_will_domains.append({
                "factor": fname,
                "impact": fdef["impact"],
                "difficulty": fdef["difficulty"],
                "time_to_effect_years": fdef["time_to_effect_years"],
                "description": fdef["description"],
                "roi": round(fdef["impact"] / max(0.1, fdef["difficulty"]), 2),
                "current_investment": round(float(current), 2),
            })

        locked_in_factors = sorted(UNMODIFIABLE_FACTORS.items(), key=lambda x: x[1]["impact"], reverse=True)

        return sanitize_nan({
            "total_determinism": round(float(total_determinism), 3),
            "destiny_label": destiny_label,
            "agency_score": round(agency_score, 3),
            "genetic_determinism": round(float(genetic_determinism), 3),
            "structural_determinism": round(float(structural_determinism), 3),
            "behavioral_determinism": round(float(behavioral_determinism), 3),
            "epigenetic_determinism": round(float(epigenetic_determinism), 3),
            "adjusted_life_expectancy": round(float(adjusted_life_expectancy), 1),
            "determinants": determinants,
            "trajectory_archetypes": trajectory_archetypes,
            "modifiable_factors": free_will_domains[:10],
            "locked_in_factors": [{"name": k, "impact": v["impact"], "description": v["description"]} for k, v in locked_in_factors],
            "highest_roi_interventions": [f["factor"] for f in sorted(free_will_domains, key=lambda x: x["roi"], reverse=True)[:5]],
            "recommendations": self._personal_recommendations(total_determinism, agency_score, state),
        })

    def assess_mortality(self, state: PersonalFateState = None) -> dict:
        if state is None:
            state = PersonalFateState()
        rng = np.random.RandomState(self.seed)

        baseline_le = 73.0
        modifiers = {
            "exercise": (state.exercise_habit - 0.5) * 6,
            "sleep": (state.sleep_quality - 0.5) * 4,
            "diet": (state.diet_quality - 0.5) * 5,
            "social": (state.relationship_stability - 0.5) * 3,
            "loneliness": (0.5 - state.loneliness_index) * 4,
            "genetic": (0.5 - state.genetic_risk_score) * 5,
            "chronic_disease": (0.5 - state.chronic_disease_risk) * 7,
            "mental_health": (state.mental_health_baseline - 0.5) * 3,
            "addiction": (0.5 - state.addiction_vulnerability) * 4,
            "childhood_aces": (0.5 - state.adverse_childhood_experiences) * 3,
        }

        adjusted_le = baseline_le + sum(modifiers.values())
        adjusted_le = max(40, min(110, adjusted_le))

        cause_probabilities = {
            "cardiovascular_disease": round(float(np.clip(0.30 + (1 - state.exercise_habit) * 0.15 + (1 - state.diet_quality) * 0.1 + state.chronic_disease_risk * 0.1, 0, 1)), 3),
            "cancer": round(float(np.clip(0.22 + state.genetic_risk_score * 0.1 + (1 - state.diet_quality) * 0.05, 0, 1)), 3),
            "neurodegenerative": round(float(np.clip(0.08 + (1 - state.cognitive_reserve) * 0.06 + (1 - state.exercise_habit) * 0.04, 0, 1)), 3),
            "respiratory": round(float(np.clip(0.06 + state.chronic_disease_risk * 0.05, 0, 1)), 3),
            "accident_injury": round(float(np.clip(0.05 + (1 - state.sleep_quality) * 0.03, 0, 1)), 3),
            "mental_health_related": round(float(np.clip(0.04 + (1 - state.mental_health_baseline) * 0.08 + state.loneliness_index * 0.05, 0, 1)), 3),
            "infectious_disease": round(float(np.clip(0.03 + (1 - state.health_baseline) * 0.04, 0, 1)), 3),
            "other": 0.22,
        }

        telomere_proxy = float(np.clip(
            state.health_baseline * 0.3 +
            state.exercise_habit * 0.2 +
            state.sleep_quality * 0.2 +
            (1 - state.stress_level if hasattr(state, 'stress_level') else 1 - state.epigenetic_stress) * 0.15 +
            state.diet_quality * 0.15,
            0.2, 1.0
        ))

        biological_age = state.age + (1.0 - telomere_proxy) * 10
        healthspan = adjusted_le * telomere_proxy * 0.8

        morbidity_compounding = float(np.clip(
            state.chronic_disease_risk * 0.3 +
            (1 - state.health_baseline) * 0.2 +
            state.addiction_vulnerability * 0.2 +
            (1 - state.exercise_habit) * 0.15 +
            (1 - state.sleep_quality) * 0.15,
            0, 1
        ))

        return sanitize_nan({
            "life_expectancy_adjusted": round(adjusted_le, 1),
            "biological_age": round(biological_age, 1),
            "chronological_age": state.age,
            "healthspan_estimate": round(healthspan, 1),
            "telomere_proxy": round(telomere_proxy, 3),
            "morbidity_compounding": round(morbidity_compounding, 3),
            "cause_probabilities": cause_probabilities,
            "modifiable_years_gained": round(sum(max(0, v) for v in modifiers.values()), 1),
            "modifiable_years_lost": round(sum(min(0, v) for v in modifiers.values()), 1),
            "recommendations": self._mortality_recommendations(adjusted_le, state, modifiers),
        })

    def assess_habit_destiny(self, state: PersonalFateState = None) -> dict:
        if state is None:
            state = PersonalFateState()
        rng = np.random.RandomState(self.seed)

        habit_inertia = float(np.clip(
            state.habit_strength * 0.4 +
            (1 - state.neuroplasticity) * 0.3 +
            (1 - state.purpose_clarity) * 0.2 +
            state.loneliness_index * 0.1,
            0, 1
        ))

        change_probability = float(np.clip(
            (1 - habit_inertia) * 0.5 +
            state.neuroplasticity * 0.2 +
            state.cognitive_reserve * 0.15 +
            state.social_capital * 0.15,
            0.05, 0.95
        ))

        annual_habit_reversion = float(np.clip(
            state.habit_strength * 0.6 +
            state.addiction_vulnerability * 0.2 +
            (1 - state.purpose_clarity) * 0.2,
            0.1, 0.9
        ))

        keystone_habits = [
            {"habit": "sleep", "impact": 0.25, "current": round(state.sleep_quality, 2), "reversion_risk": round(float(np.clip(1 - state.sleep_quality * 0.5, 0.1, 0.9)), 3), "description": "Sleep is the keystone habit — it modulates every other behavior"},
            {"habit": "exercise", "impact": 0.20, "current": round(state.exercise_habit, 2), "reversion_risk": round(float(np.clip(1 - state.exercise_habit * 0.5, 0.1, 0.9)), 3), "description": "Exercise creates upward spirals — energy, mood, discipline, health compound"},
            {"habit": "nutrition", "impact": 0.15, "current": round(state.diet_quality, 2), "reversion_risk": round(float(np.clip(1 - state.diet_quality * 0.5, 0.1, 0.9)), 3), "description": "Diet shapes gut microbiome which shapes cravings — a self-reinforcing loop"},
            {"habit": "social_connection", "impact": 0.20, "current": round(state.relationship_stability, 2), "reversion_risk": round(float(np.clip(state.loneliness_index * 0.7, 0.1, 0.9)), 3), "description": "Social isolation feeds on itself — loneliness begets withdrawal begets more loneliness"},
            {"habit": "purpose", "impact": 0.20, "current": round(state.purpose_clarity, 2), "reversion_risk": round(float(np.clip(1 - state.purpose_clarity * 0.5, 0.1, 0.9)), 3), "description": "Purpose creates coherence — without it, habits drift toward comfort-seeking"},
        ]

        compounding_loops = []
        if state.exercise_habit > 0.5 and state.diet_quality > 0.5 and state.sleep_quality > 0.5:
            compounding_loops.append({
                "name": "virtuous_health_compounding",
                "description": "Exercise + nutrition + sleep create upward spiral — each reinforces the others",
                "strength": round(float(np.clip((state.exercise_habit + state.diet_quality + state.sleep_quality) / 3, 0, 1)), 3),
                "direction": "positive",
            })
        if state.loneliness_index > 0.5 and state.addiction_vulnerability > 0.3:
            compounding_loops.append({
                "name": "vicious_isolation_compounding",
                "description": "Loneliness → substance use → worse relationships → more loneliness — downward spiral",
                "strength": round(float(np.clip(state.loneliness_index * 0.5 + state.addiction_vulnerability * 0.5, 0, 1)), 3),
                "direction": "negative",
            })
        if state.purpose_clarity > 0.5 and state.social_capital > 0.4:
            compounding_loops.append({
                "name": "purpose_network_compounding",
                "description": "Purpose attracts aligned people → stronger network → more opportunities → clearer purpose",
                "strength": round(float(np.clip(state.purpose_clarity * 0.5 + state.social_capital * 0.5, 0, 1)), 3),
                "direction": "positive",
            })
        if state.habit_strength > 0.6 and (1 - state.purpose_clarity) > 0.5:
            compounding_loops.append({
                "name": "autopilot_drift",
                "description": "Strong habits without purpose → life on autopilot → habitual mediocrity → reinforced habits",
                "strength": round(float(np.clip(state.habit_strength * 0.6 + (1 - state.purpose_clarity) * 0.4, 0, 1)), 3),
                "direction": "negative",
            })

        return sanitize_nan({
            "habit_inertia": round(habit_inertia, 3),
            "inertia_label": "extreme" if habit_inertia > 0.7 else "high" if habit_inertia > 0.5 else "moderate" if habit_inertia > 0.3 else "low",
            "change_probability": round(change_probability, 3),
            "annual_habit_reversion": round(annual_habit_reversion, 3),
            "keystone_habits": keystone_habits,
            "compounding_loops": compounding_loops,
            "years_trapped_by_habits": round(float(annual_habit_reversion * 20), 0),
            "recommendations": self._habit_recommendations(habit_inertia, change_probability, state),
        })

    def assess_socioeconomic_path(self, state: PersonalFateState = None) -> dict:
        if state is None:
            state = PersonalFateState()
        rng = np.random.RandomState(self.seed)

        origin_trap = float(np.clip(
            (1 - state.socioeconomic_origin) * 0.4 +
            (1 - state.intergenerational_mobility) * 0.3 +
            (1 - state.zip_code_destiny) * 0.2 +
            (1 - state.financial_safety_net) * 0.1,
            0, 1
        ))

        mobility_potential = float(np.clip(
            state.education_level * 0.25 +
            state.social_capital * 0.2 +
            state.talent_alignment * 0.2 +
            state.geographic_mobility * 0.15 +
            state.cognitive_reserve * 0.1 +
            state.neuroplasticity * 0.1,
            0, 1
        ))

        income_trajectory = float(np.clip(
            state.socioeconomic_origin * 0.3 +
            state.education_level * 0.2 +
            state.social_capital * 0.15 +
            state.talent_alignment * 0.15 +
            state.health_baseline * 0.1 +
            mobility_potential * 0.1,
            0, 1
        ))

        if origin_trap > 0.7:
            path_label = "structural_trap"
        elif origin_trap > 0.5:
            path_label = "constrained_path"
        elif origin_trap > 0.3:
            path_label = "partially_free"
        else:
            path_label = "open_field"

        percentile_proxy = round(float(np.clip(income_trajectory * 100, 5, 99)), 0)

        return sanitize_nan({
            "origin_trap_index": round(origin_trap, 3),
            "path_label": path_label,
            "mobility_potential": round(mobility_potential, 3),
            "income_trajectory": round(income_trajectory, 3),
            "income_percentile_estimate": percentile_proxy,
            "structural_factors": {
                "birth_zip_contribution": round(float((1 - state.zip_code_destiny) * 0.25), 3),
                "parental_income_contribution": round(float((1 - state.socioeconomic_origin) * 0.30), 3),
                "education_contribution": round(float(state.education_level * 0.20), 3),
                "social_capital_contribution": round(float(state.social_capital * 0.15), 3),
                "mobility_contribution": round(float(state.geographic_mobility * 0.10), 3),
            },
            "probability_of_class_escape": round(float(np.clip(mobility_potential * (1 - origin_trap * 0.5), 0.01, 0.95)), 3),
            "probability_of_class_decline": round(float(np.clip((1 - income_trajectory) * origin_trap * 0.5, 0.01, 0.6)), 3),
            "recommendations": self._socioeconomic_recommendations(origin_trap, mobility_potential, state),
        })

    def predict_personal_fate(self, state: PersonalFateState = None) -> dict:
        if state is None:
            state = PersonalFateState()

        destiny = self.assess_personal_destiny(state)
        mortality = self.assess_mortality(state)
        habits = self.assess_habit_destiny(state)
        socioeconomic = self.assess_socioeconomic_path(state)

        total_destiny = destiny.get("total_determinism", 0.5)
        agency = destiny.get("agency_score", 0.5)

        if total_destiny > 0.7:
            fate_type = "predetermined"
        elif total_destiny > 0.5:
            fate_type = "heavily_constrained"
        elif total_destiny > 0.35:
            fate_type = "partially_destined"
        elif total_destiny > 0.2:
            fate_type = "significant_agency"
        else:
            fate_type = "open_horizon"

        return sanitize_nan({
            "fate_type": fate_type,
            "total_determinism": total_destiny,
            "agency_score": agency,
            "personal_destiny": destiny,
            "mortality": mortality,
            "habit_destiny": habits,
            "socioeconomic_path": socioeconomic,
            "immutable_pct": round((0.20 + 0.25 + 0.18 + 0.15) / 4 * 100, 0),
            "modifiable_pct": round(100 - (0.20 + 0.25 + 0.18 + 0.15) / 4 * 100, 0),
            "highest_leverage_actions": destiny.get("highest_roi_interventions", [])[:5],
            "recommendations": self._personal_recommendations(total_destiny, agency, state),
        })

    def _personal_recommendations(self, determinism: float, agency: float, state: PersonalFateState) -> list:
        recs = []
        if determinism > 0.6:
            recs.append("Your fate is heavily determined — focus on the few high-leverage choices that remain: sleep, exercise, social connection, and purpose")
        if agency > 0.5:
            recs.append("You have significant agency — invest in neuroplasticity through novelty, learning, and deliberate practice")
        if state.loneliness_index > 0.5:
            recs.append("Loneliness is a structural health risk equivalent to smoking — prioritize relationship investment")
        if state.adverse_childhood_experiences > 0.4:
            recs.append("ACEs create epigenetic determination but trauma-informed therapy can partially reverse methylation patterns")
        if state.purpose_clarity < 0.3:
            recs.append("Without purpose, habits drift toward comfort — finding ikigai is not optional, it is the engine of agency")
        if state.sleep_quality < 0.4:
            recs.append("Sleep is the single highest-ROI health intervention — improving sleep unlocks cascading improvements in every domain")
        if determinism < 0.4:
            recs.append("Your fate is relatively open — deliberate practice, strategic relocation, and relationship curation have maximum impact now")
        return recs[:5]

    def _mortality_recommendations(self, le: float, state: PersonalFateState, modifiers: dict) -> list:
        recs = []
        biggest_gains = sorted(modifiers.items(), key=lambda x: x[1], reverse=True)
        top = biggest_gains[:3]
        for name, gain in top:
            if gain > 0:
                recs.append(f"Improving {name} could add ~{gain:.1f} years to life expectancy")
        if state.chronic_disease_risk > 0.5:
            recs.append("Chronic disease risk is elevated — preventive screening and lifestyle modification are urgent")
        return recs[:5]

    def _habit_recommendations(self, inertia: float, change_prob: float, state: PersonalFateState) -> list:
        recs = []
        if inertia > 0.6:
            recs.append("Habit inertia is extremely high — start with keystone habits (sleep first, then exercise) rather than trying to change everything")
        if change_prob < 0.3:
            recs.append("Change probability is low — environment design (removing triggers, adding friction) is more effective than willpower")
        if state.loneliness_index > 0.4:
            recs.append("Social isolation feeds habit loops — joining communities that embody desired habits leverages social conformity")
        if state.purpose_clarity < 0.4:
            recs.append("Without purpose, habit change is 3x harder — clarify values and meaning before attempting behavioral overhaul")
        return recs[:5]

    def _socioeconomic_recommendations(self, trap: float, mobility: float, state: PersonalFateState) -> list:
        recs = []
        if trap > 0.6:
            recs.append("Structural trap is severe — geographic mobility and education are the highest-leverage escape vectors")
        if state.social_capital < 0.4:
            recs.append("Social capital is low — mentoring relationships are the single strongest predictor of class mobility")
        if state.talent_alignment < 0.4:
            recs.append("Talent-alignment gap is significant — discover strengths first (VIA survey, StrengthsFinder) then build career around them")
        if mobility > 0.5 and trap > 0.4:
            recs.append("High mobility potential but constrained by origin — strategic relocation and network-building can unlock trapped potential")
        return recs[:5]

    def _trajectory_recommendations(self, score: float, existential_risk: float, social_capacity: float) -> list:
        recs = []
        if score < 0.4:
            recs.append("CRITICAL: Civilizational trajectory is declining — immediate action needed on energy, governance, and social cohesion")
        if existential_risk > 0.5:
            recs.append("Existential risk is elevated — nuclear disarmament and pandemic preparedness are top priorities")
        if social_capacity < 0.3:
            recs.append("Social capacity critically low — governance reform and social cohesion building are essential")
        if score > 0.6:
            recs.append("Trajectory is positive — invest in space capability and biotech to extend civilizational reach")
        if not recs:
            recs.append("Trajectory is moderate — focus on resource substitution and climate mitigation to improve prospects")
        return recs[:5]

    def _thermodynamic_recommendations(self, pressure: float, eroi: float, overshoot: bool) -> list:
        recs = []
        if pressure > 0.5:
            recs.append("CRITICAL: Thermodynamic pressure exceeds sustainable levels — radical efficiency gains and resource substitution needed")
        if eroi < 10:
            recs.append("EROI below threshold — energy returned on investment declining toward viability limit; accelerate renewable transition")
        if overshoot:
            recs.append("Population exceeding carrying capacity — managed de-growth or rapid efficiency gains required")
        if not recs:
            recs.append("Thermodynamic constraints are manageable but require long-term planning for resource transitions")
        return recs[:5]

    def _depletion_recommendations(self, pressure: float, critical: list, substitute_gap: list) -> list:
        recs = []
        if pressure > 0.4:
            recs.append("Resource depletion pressure is high — accelerate circular economy and substitution strategies")
        if critical:
            recs.append(f"Critical resources approaching depletion: {', '.join(critical[:3])} — emergency substitution programs needed")
        if substitute_gap:
            recs.append(f"No adequate substitutes for: {', '.join(substitute_gap[:3])} — R&D investment urgent")
        if not recs:
            recs.append("Resource depletion is manageable — continue monitoring and investing in substitution")
        return recs[:5]

    def _demographic_recommendations(self, state: FateState, dependency_ratio: float) -> list:
        recs = []
        if state.human_population_bn > 10:
            recs.append("Population approaching carrying capacity estimates — voluntary family planning and education access critical")
        if dependency_ratio > 0.6:
            recs.append("High dependency ratio — automation, immigration reform, and extended working life needed to maintain productivity")
        if state.economic_inequality_gini > 0.6:
            recs.append("Extreme inequality undermines social stability and demographic resilience — progressive reform needed")
        return recs[:5]

    def _biological_recommendations(self, advancement: float, risk: float, state: FateState) -> list:
        recs = []
        if risk > 0.5:
            recs.append("Elevated biological risk — strengthen pandemic preparedness and synthetic biology governance")
        if state.pandemic_preparedness < 0.4:
            recs.append("Pandemic preparedness dangerously low — invest in surveillance, rapid response, and PPE stockpiles")
        if advancement > 0.5:
            recs.append("Biotechnology capability is significant — ensure governance frameworks keep pace with capability")
        return recs[:5]

    def _commitment_recommendations(self, irreversibility: float, high_irrev: list, state: FateState) -> list:
        recs = []
        if irreversibility > 0.7:
            recs.append("High average irreversibility — many committed changes cannot be reversed; focus on preventing new commitments")
        if state.climate_commitment_c > 1.5:
            recs.append("Climate commitment exceeds 1.5°C — aggressive mitigation needed to prevent further lock-in")
        if state.biodiversity_integrity_index < 0.3:
            recs.append("Biodiversity integrity critically low — extinction debt is irreversible; maximize conservation immediately")
        if len(high_irrev) > 3:
            recs.append(f"{len(high_irrev)} high-irreversibility commitments identified — these represent locked-in fate; work to minimize additional irreversible commitments")
        return recs[:5]

    def _cosmological_recommendations(self, state: FateState) -> list:
        recs = []
        if state.nuclear_proliferation_risk > 0.4:
            recs.append("Nuclear risk elevated on cosmological timescale — only existential risk with agency to prevent")
        if state.ai_capability_index > 0.3:
            recs.append("AI capability advancing — ensure alignment before capabilities exceed governance frameworks")
        recs.append("Civilization's long-term fate depends on choices made in the next century — the Great Filter may be ahead or behind")
        return recs[:5]

    def _escape_velocities(self, state: FateState, trajectory: dict) -> list:
        escapes = []
        energy_progress = trajectory.get("energy_progress_to_type_i_pct", 73)
        if energy_progress < 50:
            escapes.append({"name": "Energy Revolution", "description": "Achieve fusion or space-based solar to bypass fossil fuel constraints", "feasibility": round(float(np.clip(state.ai_capability_index * 2 + 0.2, 0, 1)), 3), "impact": 0.9})
        if state.biodiversity_integrity_index < 0.4:
            escapes.append({"name": "Biosphere Restoration", "description": "Reverse biodiversity decline through大规模 conservation and restoration", "feasibility": round(float(np.clip(state.governance_effectiveness * 1.5, 0, 1)), 3), "impact": 0.8})
        if state.nuclear_proliferation_risk > 0.3:
            escapes.append({"name": "Nuclear Disarmament", "description": "Verifiable nuclear disarmament to remove existential risk", "feasibility": round(float(np.clip(state.social_cohesion_index * 2, 0, 1)), 3), "impact": 1.0})
        if state.resource_remaining_pct < 40:
            escapes.append({"name": "Resource Substitution", "description": "Develop substitutes for critical depleting resources", "feasibility": round(float(np.clip(state.ai_capability_index + state.biotech_capability, 0, 1)), 3), "impact": 0.7})
        escapes.append({"name": "Multi-Planetary Expansion", "description": "Establish self-sustaining colonies on Mars or orbital habitats", "feasibility": round(float(np.clip(state.space_capability * 3 + state.ai_capability_index * 0.5, 0, 1)), 3), "impact": 0.95})
        escapes.sort(key=lambda x: x["impact"], reverse=True)
        return escapes

    def _fate_recommendations(self, score: float, trajectory: dict, thermodynamic: dict, commitment: dict) -> list:
        recs = []
        if score > 0.7:
            recs.append("Fate is heavily determined — many irreversible commitments are locked in; focus on adaptation rather than prevention")
        elif score > 0.5:
            recs.append("Significant deterministic constraints exist — prioritize reversible decisions and maintain optionality")
        elif score > 0.3:
            recs.append("Fate is moderately determined — the window for shaping destiny remains open but narrowing")
        else:
            recs.append("Fate remains largely open — current choices have maximal impact on civilizational trajectory")
        if trajectory.get("existential_risk_composite", 0) > 0.4:
            recs.append("Existential risk is the highest-priority constraint — nuclear disarmament and pandemic preparedness are fate-shaping")
        if thermodynamic.get("thermodynamic_pressure", 0) > 0.4:
            recs.append("Thermodynamic constraints are binding — efficiency and resource substitution are not optional but mandatory")
        if commitment.get("n_high_irreversibility", 0) > 3:
            recs.append("Multiple high-irreversibility commitments identified — focus on preventing new irreversible commitments")
        return recs[:5]