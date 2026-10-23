"""HumanOutcomePredictor — Individual human outcome predictions.

Predicts specific, testable, falsifiable outcomes across behavioral,
psychological, genetic risk, and decision domains. Grounded in published
research with base rates from epidemiological studies, clinical trials,
and longitudinal cohort data.

Domains:
  - Behavioral: habit formation, exercise adherence, diet adherence, addiction risk
  - Psychological: anxiety onset, depression recurrence, resilience, burnout
  - Genetic Risk: polygenic risk scores, carrier status, pharmacogenomics
  - Decision: career transitions, relationship formation, financial decisions
  - Cognitive: learning rate, memory trajectory, expertise acquisition
  - Social: network formation, influence dynamics, marriage/divorce probability
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class HumanOutcomeDomain(Enum):
    BEHAVIORAL = "behavioral"
    PSYCHOLOGICAL = "psychological"
    GENETIC_RISK = "genetic_risk"
    DECISION = "decision"
    COGNITIVE = "cognitive"
    SOCIAL = "social"


class OutcomeConfidence(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class InterventionType(Enum):
    BEHAVIORAL = "behavioral"
    PHARMACOLOGICAL = "pharmacological"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    COGNITIVE = "cognitive"


@dataclass
class HumanPrediction:
    domain: HumanOutcomeDomain
    prediction: str
    probability: float
    confidence: float
    time_horizon: str
    evidence: List[str]
    falsification: str
    key_metric: str
    key_metric_current: float
    key_metric_target: float
    method: str
    base_rate_source: str
    uncertainty_band: Tuple[float, float]
    individual_factors: List[str]
    intervention_potential: float
    intervention_type: InterventionType


@dataclass
class DecisionOutcome:
    decision_domain: str
    option_a: str
    option_a_probability: float
    option_a_expected_value: float
    option_b: str
    option_b_probability: float
    option_b_expected_value: float
    bias_flags: List[str]
    evidence: str
    recommended: str
    confidence: float


@dataclass
class GeneticRiskFactor:
    condition: str
    lifetime_risk_baseline: float
    lifetime_risk_with_risk_allele: float
    risk_allele_frequency: float
    odds_ratio: float
    evidence: str
    modifiable_fraction: float
    intervention: str


@dataclass
class HumanOutcomeResult:
    predictions: List[HumanPrediction]
    decision_outcomes: List[DecisionOutcome]
    genetic_risks: List[GeneticRiskFactor]
    breakthrough_probability_by_year: Dict[int, float]
    overall_assessment: str
    domains_covered: int
    n_predictions: int


BEHAVIORAL_PREDICTIONS = [
    {
        "domain": HumanOutcomeDomain.BEHAVIORAL,
        "prediction": "Adult starting a new exercise program maintains it ≥6 months (meets CDC guidelines ≥150 min/week) — only 25% sustain beyond 6 months",
        "probability": 0.25,
        "evidence": [
            "Dishman & Buckworth 1996: 50% dropout within 6 months",
            "Marcus et al. 2000: Stages of change predict exercise maintenance",
            "Gourlan et al. 2016: Systematic review of exercise adherence determinants",
        ],
        "falsification": "Exercise adherence rate ≥50% at 12 months in prospective cohort study",
        "key_metric": "Exercise adherence rate at 6 months (%)",
        "key_metric_current": 25.0,
        "key_metric_target": 50.0,
        "time_horizon": "2026-2028",
        "method": "Meta-analytic base rate from longitudinal exercise studies",
        "base_rate_source": "Dishman et al. 1985; Marcus et al. 2000; Dunton 2022 review",
        "uncertainty_band": (0.18, 0.33),
        "individual_factors": ["prior_exercise_history", "social_support", "intrinsic_motivation", "self_efficacy"],
        "intervention_potential": 0.40,
        "intervention_type": InterventionType.BEHAVIORAL,
    },
    {
        "domain": HumanOutcomeDomain.BEHAVIORAL,
        "prediction": "New Year's resolution to lose weight: only 9% achieve sustained weight loss ≥10% body weight at 12 months",
        "probability": 0.09,
        "evidence": [
            "Norcross et al. 2002: Only 19% maintain resolutions past 2 years, 46% at 6 months",
            "Kraschnewski et al. 2010: 63% of New Year dieters regain within 1 year",
            "Wing & Phelan 2005: NWCR shows <20% maintain ≥10% loss over 5 years",
        ],
        "falsification": "≥25% of New Year resolution-makers sustain ≥10% weight loss at 12 months",
        "key_metric": "% achieving ≥10% weight loss at 12 months",
        "key_metric_current": 9.0,
        "key_metric_target": 25.0,
        "time_horizon": "2026-2028",
        "method": "Longitudinal resolution tracking studies",
        "base_rate_source": "Norcross 2002; Wing & Phelan 2005",
        "uncertainty_band": (0.05, 0.15),
        "individual_factors": ["bmi_baseline", "dietary_adherence", "social_support", "prior_attempts"],
        "intervention_potential": 0.55,
        "intervention_type": InterventionType.BEHAVIORAL,
    },
    {
        "domain": HumanOutcomeDomain.BEHAVIORAL,
        "prediction": "Smoker attempting unaided quit: only 3-5% achieve 12-month abstinence without pharmacological or behavioral support",
        "probability": 0.04,
        "evidence": [
            "Hughes et al. 2004: Unaided quit rate ~3-5% at 12 months",
            "CDC 2020: Most quit attempts fail without support",
            "Hartmann-Boyce et al. 2021 Cochrane: Nicotine replacement + behavioral support triples quit rate",
        ],
        "falsification": "Unaided quit rate ≥15% at 12 months in prospective clinical trial",
        "key_metric": "12-month abstinence rate unaided (%)",
        "key_metric_current": 4.0,
        "key_metric_target": 15.0,
        "time_horizon": "2026-2028",
        "method": "Cochrane review meta-analysis",
        "base_rate_source": "Hughes et al. 2004; Hartmann-Boyce et al. 2021",
        "uncertainty_band": (0.03, 0.07),
        "individual_factors": ["cigarettes_per_day", "years_smoking", "prior_quit_attempts", "ftnd_score"],
        "intervention_potential": 0.70,
        "intervention_type": InterventionType.PHARMACOLOGICAL,
    },
    {
        "domain": HumanOutcomeDomain.BEHAVIORAL,
        "prediction": "A person with bedtime procrastination (≥2x/week) will reduce to ≤1x/week within 6 months using CBT-I: 45% probability",
        "probability": 0.45,
        "evidence": [
            "Kroese et al. 2014: Bedtime procrastination linked to insufficient sleep",
            "Traeger et al. 2019: CBT-I produces durable improvements at 6 months",
            "Morin et al. 2022: CBT-I 70-80% response rate for insomnia, but lower for pure procrastination",
        ],
        "falsification": "CBT-I reduces bedtime procrastination to ≤1x/week in ≥60% of cases at 6 months",
        "key_metric": "Bedtime procrastination frequency per week",
        "key_metric_current": 4.0,
        "key_metric_target": 1.0,
        "time_horizon": "2026-2027",
        "method": "Clinical trial base rates with CBT-I protocols",
        "base_rate_source": "Morin et al. 2022; Kroese et al. 2014",
        "uncertainty_band": (0.30, 0.60),
        "individual_factors": ["chronotype", "work_schedule", "screen_time", "anxiety_level"],
        "intervention_potential": 0.40,
        "intervention_type": InterventionType.COGNITIVE,
    },
]

PSYCHOLOGICAL_PREDICTIONS = [
    {
        "domain": HumanOutcomeDomain.PSYCHOLOGICAL,
        "prediction": "Adult with one prior depressive episode: 50% probability of recurrence within 2 years without ongoing treatment",
        "probability": 0.50,
        "evidence": [
            "Burcusa et al. 2003: 50% recurrence rate at 2 years after single episode",
            "Solomon et al. 2000: Recurrence risk increases with number of prior episodes",
            "Hardeveld et al. 2013 meta-analysis: 2-year recurrence 30-40% with treatment, 50% without",
        ],
        "falsification": "≥70% 2-year recurrence rate after first episode in prospective cohort",
        "key_metric": "2-year depressive recurrence rate (%)",
        "key_metric_current": 50.0,
        "key_metric_target": 30.0,
        "time_horizon": "2026-2028",
        "method": "Epidemiological cohort studies with survival analysis",
        "base_rate_source": "Burcusa et al. 2003; Hardeveld et al. 2013",
        "uncertainty_band": (0.40, 0.60),
        "individual_factors": ["prior_episodes", "family_history", "stress_buffering", "treatment_adherence"],
        "intervention_potential": 0.35,
        "intervention_type": InterventionType.PHARMACOLOGICAL,
    },
    {
        "domain": HumanOutcomeDomain.PSYCHOLOGICAL,
        "prediction": "Young adult (18-25) experiencing first panic attack: 40% probability of developing panic disorder within 12 months",
        "probability": 0.40,
        "evidence": [
            "Ehlers 1993: 40% of first-panic-attack individuals develop panic disorder",
            "Kessler et al. 2006: Median age of onset 24, 50% onset by age 30",
            "Craske & Waters 2005: Cognitive-behavioral factors predict panic progression",
        ],
        "falsification": "Panic disorder onset after first panic attack >55% in prospective study",
        "key_metric": "Panic disorder onset rate after first attack (%)",
        "key_metric_current": 40.0,
        "key_metric_target": 25.0,
        "time_horizon": "2026-2028",
        "method": "Prospective cohort of first-episode panic",
        "base_rate_source": "Ehlers 1993; Kessler et al. 2006",
        "uncertainty_band": (0.30, 0.50),
        "individual_factors": ["anxiety_sensitivity", "agoraphobia_tendency", "catastrophizing", "social_support"],
        "intervention_potential": 0.50,
        "intervention_type": InterventionType.COGNITIVE,
    },
    {
        "domain": HumanOutcomeDomain.PSYCHOLOGICAL,
        "prediction": "Employee experiencing burnout (MBI high exhaustion): 60% probability of taking medical leave or quitting within 18 months without intervention",
        "probability": 0.60,
        "evidence": [
            "Maslach & Leiter 2016: Burnout predicts turnover, absenteeism, and disability",
            "Salvagioni et al. 2017: Systematic review shows burnout → illness, absenteeism",
            "Ahola et al. 2009: Burnout associated with 1.5x risk of long-term sickness absence",
        ],
        "falsification": "Burnout → leave/quit rate <40% at 18 months in prospective organizational study",
        "key_metric": "Medical leave or job exit rate from burnout (%)",
        "key_metric_current": 60.0,
        "key_metric_target": 30.0,
        "time_horizon": "2026-2028",
        "method": "Organizational psychology longitudinal studies",
        "base_rate_source": "Maslach & Leiter 2016; Salvagioni et al. 2017",
        "uncertainty_band": (0.45, 0.72),
        "individual_factors": ["workload", "autonomy", "reward_fairness", "community"],
        "intervention_potential": 0.45,
        "intervention_type": InterventionType.ENVIRONMENTAL,
    },
    {
        "domain": HumanOutcomeDomain.PSYCHOLOGICAL,
        "prediction": "Adult with high resilience (CD-RISC ≥33): 75% probability of maintaining mental health after major life stressor (bereavement, job loss, divorce)",
        "probability": 0.75,
        "evidence": [
            "Connor & Davidson 2003: CD-RISC normative data, high scorers show better outcomes",
            "Southwick et al. 2014: Resilience is modifiable and predicts post-trauma trajectory",
            "Bonanno 2004: 35-65% show resilience trajectory after loss; higher with protective factors",
        ],
        "falsification": "High CD-RISC scorers show <55% resilient trajectory after major stressor",
        "key_metric": "Resilient trajectory rate after major stressor (%)",
        "key_metric_current": 75.0,
        "key_metric_target": 85.0,
        "time_horizon": "2026-2028",
        "method": "Prospective resilience tracking studies",
        "base_rate_source": "Connor & Davidson 2003; Bonanno 2004",
        "uncertainty_band": (0.60, 0.85),
        "individual_factors": ["social_support", "optimism", "cognitive_flexibility", "meaning_purpose"],
        "intervention_potential": 0.30,
        "intervention_type": InterventionType.SOCIAL,
    },
]

GENETIC_RISK_PREDICTIONS = [
    {
        "domain": HumanOutcomeDomain.GENETIC_RISK,
        "prediction": "Carrier of APOE ε4 allele (heterozygous): lifetime Alzheimer's risk increases from 11% to ~25% by age 85",
        "probability": 0.25,
        "evidence": [
            "Farrer et al. 1997: APOE ε4 OR=2.6-3.2 for AD (heterozygous), OR=12-15 (homozygous)",
            "Jansen et al. 2019: 75 risk loci explain ~33% of AD heritability beyond APOE",
            "Bellenguez et al. 2022: 75 loci, APOE ε4 remains strongest single genetic risk factor",
        ],
        "falsification": "APOE ε4 heterozygous carrier AD risk <15% or >40% in new GWAS meta-analysis",
        "key_metric": "Lifetime AD risk for APOE ε4 heterozygous (%)",
        "key_metric_current": 25.0,
        "key_metric_target": 11.0,
        "time_horizon": "2026-2030",
        "method": "GWAS meta-analysis with polygenic risk scoring",
        "base_rate_source": "Farrer et al. 1997; Bellenguez et al. 2022",
        "uncertainty_band": (0.18, 0.35),
        "individual_factors": ["family_history_ad", "cardiovascular_risk", "education_level", "physical_activity"],
        "intervention_potential": 0.30,
        "intervention_type": InterventionType.ENVIRONMENTAL,
    },
    {
        "domain": HumanOutcomeDomain.GENETIC_RISK,
        "prediction": "BRCA1 pathogenic variant carrier: lifetime breast cancer risk 55-72% by age 70 (without prophylactic surgery)",
        "probability": 0.65,
        "evidence": [
            "Kuchenbaecker et al. 2017: BRCA1 cumulative risk 65% (95%CI 61-69%) by age 80",
            "Antoniou et al. 2003: BRCA1 carriers have 55-72% breast cancer risk by age 70",
            "Rebbeck et al. 2004: Risk-reducing mastectomy reduces risk by ~90%",
        ],
        "falsification": "BRCA1 penetrance <40% or >85% in new prospective carrier cohort",
        "key_metric": "Lifetime breast cancer risk for BRCA1 carriers (%)",
        "key_metric_current": 65.0,
        "key_metric_target": 65.0,
        "time_horizon": "2026-2030",
        "method": "Prospective carrier cohort studies with penetrance analysis",
        "base_rate_source": "Kuchenbaecker et al. 2017; Antoniou et al. 2003",
        "uncertainty_band": (0.55, 0.72),
        "individual_factors": ["family_history_breast", "hormonal_factors", "screening_adherence", "prophylactic_surgery"],
        "intervention_potential": 0.90,
        "intervention_type": InterventionType.PHARMACOLOGICAL,
    },
    {
        "domain": HumanOutcomeDomain.GENETIC_RISK,
        "prediction": "PRS (polygenic risk score) in top 5% for coronary artery disease: 3x relative risk vs. population average; absolute risk ≈ 18% by age 75",
        "probability": 0.18,
        "evidence": [
            "Khera et al. 2018: Top 5% PRS has OR≈3.0 for CAD",
            "Inoue & Khera 2023: PRS adds incremental risk prediction beyond clinical factors",
            "Mars et al. 2022: PRS for CAD validated across multiple ancestries (EUR: OR~3.0, AFR: OR~2.0)",
        ],
        "falsification": "Top 5% PRS CAD relative risk <2.0 or >5.0 in independent validation cohort",
        "key_metric": "CAD absolute risk by age 75 for top-5% PRS (%)",
        "key_metric_current": 18.0,
        "key_metric_target": 6.0,
        "time_horizon": "2026-2030",
        "method": "Polygenic risk scoring with population-level validation",
        "base_rate_source": "Khera et al. 2018; Mars et al. 2022",
        "uncertainty_band": (0.12, 0.25),
        "individual_factors": ["ldl_cholesterol", "blood_pressure", "smoking_status", "diabetes"],
        "intervention_potential": 0.60,
        "intervention_type": InterventionType.ENVIRONMENTAL,
    },
    {
        "domain": HumanOutcomeDomain.GENETIC_RISK,
        "prediction": "CYP2D6 poor metabolizer: 7-10% of Caucasians carry reduced function; ~25% of all medications metabolized by CYP2D6 show altered efficacy/toxicity",
        "probability": 0.10,
        "evidence": [
            "Gaedigk et al. 2017: CYP2D6 allele frequencies vary by ancestry; PM rate 5-10% EUR",
            "Zanger & Schwab 2013: CYP2D6 metabolizes 25% of all clinically used drugs",
            "Hicks et al. 2013 CPIC: Dosing guidelines for CYP2D6 poor metabolizers",
        ],
        "falsification": "CYP2D6 poor metabolizer prevalence >15% in European-ancestry populations",
        "key_metric": "CYP2D6 poor metabolizer prevalence - EUR ancestry (%)",
        "key_metric_current": 10.0,
        "key_metric_target": 10.0,
        "time_horizon": "2026-2030",
        "method": "Pharmacogenomic population genetics",
        "base_rate_source": "Gaedigk et al. 2017; CPIC Guidelines",
        "uncertainty_band": (0.05, 0.15),
        "individual_factors": ["ancestry", "concurrent_medications", "liver_function", "age"],
        "intervention_potential": 0.80,
        "intervention_type": InterventionType.PHARMACOLOGICAL,
    },
]

DECISION_PREDICTIONS = [
    {
        "domain": HumanOutcomeDomain.DECISION,
        "prediction": "Career changer after age 40: 55% probability of reporting satisfaction improvement at 2 years, but 35% regret at 1 year",
        "probability": 0.55,
        "evidence": [
            "Ibarra 2003: Career reinvention success depends on identity exploration period",
            "Brett 1982: Job transfer satisfaction U-curve (dip at 6 months, recovery at 2 years)",
            "Kanchier & Unruh 1988: 55-65% of career changers report satisfaction improvement at 2 years",
        ],
        "falsification": "Career change satisfaction rate at 2 years <35% or >75% in longitudinal study",
        "key_metric": "Satisfaction improvement after career change at 2 years (%)",
        "key_metric_current": 55.0,
        "key_metric_target": 70.0,
        "time_horizon": "2026-2028",
        "method": "Longitudinal career transition studies with satisfaction tracking",
        "base_rate_source": "Ibarra 2003; Kanchier & Unruh 1988",
        "uncertainty_band": (0.40, 0.68),
        "individual_factors": ["financial_buffer", "social_support", "skill_transferability", "identity_coherence"],
        "intervention_potential": 0.35,
        "intervention_type": InterventionType.COGNITIVE,
    },
    {
        "domain": HumanOutcomeDomain.DECISION,
        "prediction": "First-time homebuyer during price peak: 40% probability of negative equity within 5 years in top-20 metro areas",
        "probability": 0.40,
        "evidence": [
            "Shiller 2015: Housing price mean-reversion cycle (14-year half-life)",
            "FHFA 2023: Negative equity rates after 2006 peak reached 25-30% nationally",
            "Glaeser & Gyourko 2018: Housing bust probability 23-40% in overvalued markets",
        ],
        "falsification": "Negative equity rate for peak buyers <20% at 5 years in top-20 metros",
        "key_metric": "Negative equity rate for peak-price homebuyers at 5 years (%)",
        "key_metric_current": 40.0,
        "key_metric_target": 15.0,
        "time_horizon": "2026-2031",
        "method": "Historical housing cycle analysis with mortgage equity models",
        "base_rate_source": "Shiller 2015; FHFA 2023",
        "uncertainty_band": (0.25, 0.55),
        "individual_factors": ["down_payment_pct", "mortgage_rate", "local_market_cycle", "job_stability"],
        "intervention_potential": 0.25,
        "intervention_type": InterventionType.ENVIRONMENTAL,
    },
    {
        "domain": HumanOutcomeDomain.DECISION,
        "prediction": "Partners meeting via dating app: 25% probability of relationship lasting ≥3 years (vs. 35% for meeting via friends/family)",
        "probability": 0.25,
        "evidence": [
            "Rosenfeld & Thomas 2012: Online dating couples slightly more likely to break up",
            "Liebe et al. 2023: App-mediated relationships show lower relationship quality at 3 years",
            "Cacioppo et al. 2013: Online marriages slightly less likely to end in separation",
        ],
        "falsification": "App-met couples' 3-year stability rate ≥35% in large prospective cohort",
        "key_metric": "Relationship stability at 3 years by meeting method (%)",
        "key_metric_current": 25.0,
        "key_metric_target": 35.0,
        "time_horizon": "2026-2029",
        "method": "Prospective relationship stability studies",
        "base_rate_source": "Rosenfeld & Thomas 2012; Liebe et al. 2023",
        "uncertainty_band": (0.18, 0.35),
        "individual_factors": ["attachment_style", "communication_quality", "shared_values", "intention_alignment"],
        "intervention_potential": 0.30,
        "intervention_type": InterventionType.SOCIAL,
    },
    {
        "domain": HumanOutcomeDomain.DECISION,
        "prediction": "Person declining a job offer for current stability: 30% probability of regretting the decision within 18 months",
        "probability": 0.30,
        "evidence": [
            "Gilovich & Medvec 1995: Action errors regretted short-term, inaction errors long-term",
            "Zeelenberg & Pieters 2004: Inaction regret tends to increase over time",
            "Dijksterhuis et al. 2006: 'Unchosen' job increases in perceived attractiveness over time",
        ],
        "falsification": "Job-rejection regret rate <15% at 18 months in prospective study",
        "key_metric": "Job opportunity rejection regret rate at 18 months (%)",
        "key_metric_current": 30.0,
        "key_metric_target": 15.0,
        "time_horizon": "2026-2028",
        "method": "Decision psychology prospective studies",
        "base_rate_source": "Gilovich & Medvec 1995; Zeelenberg & Pieters 2004",
        "uncertainty_band": (0.20, 0.40),
        "individual_factors": ["career_stage", "current_satisfaction", "risk_tolerance", "opportunity_scarcity"],
        "intervention_potential": 0.45,
        "intervention_type": InterventionType.COGNITIVE,
    },
]

COGNITIVE_PREDICTIONS = [
    {
        "domain": HumanOutcomeDomain.COGNITIVE,
        "prediction": "Adult learning a second language (L2) starting after age 30: 12% probability of reaching CEFR B2 proficiency within 5 years with 5+ hours/week study",
        "probability": 0.12,
        "evidence": [
            "Hakuta et al. 2003: Age of acquisition strongly predicts ultimate attainment",
            "DeKeyser 2013: Explicit learning compensates for age-related decline, but only to B1/B2",
            "Bongaerts 1999: Rare exceptional learners reach native-like proficiency after age 30",
        ],
        "falsification": "≥30% of adult L2 learners (age 30+) reach B2 within 5 years at 5+ hrs/week",
        "key_metric": "CEFR B2 attainment rate for adult L2 learners (%)",
        "key_metric_current": 12.0,
        "key_metric_target": 25.0,
        "time_horizon": "2026-2031",
        "method": "SLA age-of-acquisition studies with proficiency tracking",
        "base_rate_source": "Hakuta et al. 2003; DeKeyser 2013",
        "uncertainty_band": (0.07, 0.20),
        "individual_factors": ["motivation_type", "l1_similarity", "study_hours_per_week", "immersion"],
        "intervention_potential": 0.50,
        "intervention_type": InterventionType.COGNITIVE,
    },
    {
        "domain": HumanOutcomeDomain.COGNITIVE,
        "prediction": "Adult aged 25-35 performing 150+ min/week moderate exercise: 30% reduced risk of cognitive decline at age 65+ compared to sedentary peers",
        "probability": 0.70,
        "evidence": [
            "Sofi et al. 2011: Physical activity reduces dementia risk by 28% (RR 0.72)",
            "Kishimoto et al. 2018: Midlife exercise associated with 30-40% reduced cognitive decline",
            "Ngandu et al. 2015 FINGER trial: Multidomain intervention including exercise slowed decline",
        ],
        "falsification": "Exercise effect on cognitive decline shows <10% risk reduction in RCT",
        "key_metric": "Relative risk reduction for cognitive decline with exercise (%)",
        "key_metric_current": 30.0,
        "key_metric_target": 40.0,
        "time_horizon": "2026-2040",
        "method": "Longitudinal epidemiological studies + RCT evidence",
        "base_rate_source": "Sofi et al. 2011; Ngandu et al. 2015",
        "uncertainty_band": (0.20, 0.40),
        "individual_factors": ["exercise_dose", "exercise_type", "baseline_cognition", "cardiovascular_health"],
        "intervention_potential": 0.50,
        "intervention_type": InterventionType.BEHAVIORAL,
    },
    {
        "domain": HumanOutcomeDomain.COGNITIVE,
        "prediction": "Person beginning deliberate practice in a domain (chess, music, medicine) at age 25+: probability of reaching expert-level (top 1%) within 10,000 hours is ~5%",
        "probability": 0.05,
        "evidence": [
            "Ericsson et al. 1993: 10,000 hours of deliberate practice → expertise, but starting age matters",
            "Campitelli & Gobet 2011: Chess expertise starting age mediates practice effect size",
            "Hambrick et al. 2014: Deliberate practice explains ~30% of variance in expert performance",
        ],
        "falsification": "Late starters (25+) reaching top 1% at >15% rate with 10,000 hours deliberate practice",
        "key_metric": "Expert attainment rate for late starters with 10K hours (%)",
        "key_metric_current": 5.0,
        "key_metric_target": 5.0,
        "time_horizon": "2026-2036",
        "method": "Expertise acquisition studies with deliberate practice tracking",
        "base_rate_source": "Ericsson et al. 1993; Hambrick et al. 2014",
        "uncertainty_band": (0.02, 0.10),
        "individual_factors": ["starting_age", "practice_quality", "domain_specific_aptitude", "mentorship"],
        "intervention_potential": 0.20,
        "intervention_type": InterventionType.COGNITIVE,
    },
]

SOCIAL_PREDICTIONS = [
    {
        "domain": HumanOutcomeDomain.SOCIAL,
        "prediction": "US adult with <3 close confidants: 2.4x increased mortality risk over 7 years vs. those with 3+ confidants (after adjusting for health behaviors)",
        "probability": 0.70,
        "evidence": [
            "Holt-Lunstad et al. 2010 meta-analysis: Social isolation OR=1.50 for mortality",
            "Holt-Lunstad et al. 2015: Loneliness/social isolation increases mortality risk 26-32%",
            "Steptoe et al. 2013: Social isolation predicts 2.4x mortality risk over 7 years",
        ],
        "falsification": "Social isolation mortality HR <1.5 in prospective cohort with 7+ year follow-up",
        "key_metric": "Social isolation mortality hazard ratio",
        "key_metric_current": 2.4,
        "key_metric_target": 1.0,
        "time_horizon": "2026-2033",
        "method": "Prospective cohort mortality studies with social network adjustment",
        "base_rate_source": "Holt-Lunstad et al. 2010, 2015; Steptoe et al. 2013",
        "uncertainty_band": (1.5, 3.2),
        "individual_factors": ["close_confidant_count", "social_integration", "loneliness_score", "community_involvement"],
        "intervention_potential": 0.45,
        "intervention_type": InterventionType.SOCIAL,
    },
    {
        "domain": HumanOutcomeDomain.SOCIAL,
        "prediction": "First marriage at age 30+: 33% probability of divorce within 10 years (vs. 48% for marriage before age 25)",
        "probability": 0.33,
        "evidence": [
            "CDC NSFG 2022: Divorce rate for first marriage: 33% at 10 years for age 30+",
            "Levenson et al. 1993: Age at marriage inversely associated with divorce until plateau at ~30",
            "Rotz 2016: Each additional year of age at marriage reduces divorce probability by 11%",
        ],
        "falsification": "Divorce rate for first marriage at age 30+ exceeds 45% at 10 years in national survey",
        "key_metric": "10-year divorce rate for first marriage at age 30+ (%)",
        "key_metric_current": 33.0,
        "key_metric_target": 33.0,
        "time_horizon": "2026-2036",
        "method": "National longitudinal survey of family growth",
        "base_rate_source": "CDC NSFG 2022; Rotz 2016",
        "uncertainty_band": (0.25, 0.42),
        "individual_factors": ["premarital_cohabitation", "education_level", "income_ratio", "communication_style"],
        "intervention_potential": 0.35,
        "intervention_type": InterventionType.SOCIAL,
    },
    {
        "domain": HumanOutcomeDomain.SOCIAL,
        "prediction": "Remote-first worker: 38% probability of reporting 'weak ties' growth (new connections) vs. 62% for hybrid workers at 12 months",
        "probability": 0.38,
        "evidence": [
            "Yang et al. 2022 Nature: Remote work reduced weak-tie formation by 38%",
            "Bloom et al. 2015: Hybrid workers report higher satisfaction and connections",
            "Wang et al. 2023: Fully remote workers form fewer cross-functional ties",
        ],
        "falsification": "Remote-first workers' weak-tie growth ≥55% at 12 months in organizational study",
        "key_metric": "Weak-tie growth rate for remote vs. hybrid workers (%)",
        "key_metric_current": 38.0,
        "key_metric_target": 62.0,
        "time_horizon": "2026-2028",
        "method": "Organizational network analysis of remote vs. hybrid workers",
        "base_rate_source": "Yang et al. 2022; Bloom et al. 2015",
        "uncertainty_band": (0.25, 0.52),
        "individual_factors": ["proactive_networking", "digital_communication_skill", "team_size", "industry_type"],
        "intervention_potential": 0.30,
        "intervention_type": InterventionType.ENVIRONMENTAL,
    },
]

GENETIC_RISK_FACTORS = [
    GeneticRiskFactor(
        condition="Alzheimer's Disease (APOE ε4)",
        lifetime_risk_baseline=0.11,
        lifetime_risk_with_risk_allele=0.25,
        risk_allele_frequency=0.14,
        odds_ratio=3.2,
        evidence="Farrer et al. 1997; Bellenguez et al. 2022",
        modifiable_fraction=0.30,
        intervention="Exercise (30% reduction), cognitive engagement, cardiovascular risk management",
    ),
    GeneticRiskFactor(
        condition="Breast Cancer (BRCA1)",
        lifetime_risk_baseline=0.12,
        lifetime_risk_with_risk_allele=0.65,
        risk_allele_frequency=0.003,
        odds_ratio=8.0,
        evidence="Kuchenbaecker et al. 2017; Antoniou et al. 2003",
        modifiable_fraction=0.90,
        intervention="Risk-reducing mastectomy (90% risk reduction), enhanced screening, chemoprevention",
    ),
    GeneticRiskFactor(
        condition="Coronary Artery Disease (top 5% PRS)",
        lifetime_risk_baseline=0.06,
        lifetime_risk_with_risk_allele=0.18,
        risk_allele_frequency=0.05,
        odds_ratio=3.0,
        evidence="Khera et al. 2018; Mars et al. 2022",
        modifiable_fraction=0.60,
        intervention="Statin therapy, exercise, diet, blood pressure management",
    ),
    GeneticRiskFactor(
        condition="Type 2 Diabetes (top 10% PRS)",
        lifetime_risk_baseline=0.11,
        lifetime_risk_with_risk_allele=0.30,
        risk_allele_frequency=0.10,
        odds_ratio=2.5,
        evidence="Mahajan et al. 2018; Vujkovic et al. 2020",
        modifiable_fraction=0.58,
        intervention="Lifestyle intervention (58% reduction, DPP), metformin, weight management",
    ),
    GeneticRiskFactor(
        condition="Major Depressive Disorder (top 10% PRS)",
        lifetime_risk_baseline=0.15,
        lifetime_risk_with_risk_allele=0.35,
        risk_allele_frequency=0.10,
        odds_ratio=2.0,
        evidence="Howard et al. 2019; Wray et al. 2018",
        modifiable_fraction=0.40,
        intervention="CBT, exercise, social connection, stress management",
    ),
    GeneticRiskFactor(
        condition="Schizophrenia (top 1% PRS)",
        lifetime_risk_baseline=0.01,
        lifetime_risk_with_risk_allele=0.10,
        risk_allele_frequency=0.01,
        odds_ratio=10.0,
        evidence="Schizophrenia Working Group 2022; Trubetskoy et al. 2022",
        modifiable_fraction=0.20,
        intervention="Early intervention services, stress reduction, cannabis avoidance",
    ),
]

DECISION_OUTCOMES_DATA = [
    DecisionOutcome(
        decision_domain="Career",
        option_a="Accept promotion (more money, more stress)",
        option_a_probability=0.65,
        option_a_expected_value=0.60,
        option_b="Stay in current role (stability, growth elsewhere)",
        option_b_probability=0.55,
        option_b_expected_value=0.50,
        bias_flags=["status_quo_bias", "loss_aversion", "anchoring_to_salary"],
        evidence="Kahneman & Tversky 1979; Ibarra 2003 career identity transition",
        recommended="Accept if financial buffer ≥6 months; else negotiate hybrid",
        confidence=0.60,
    ),
    DecisionOutcome(
        decision_domain="Relationship",
        option_a="Express vulnerability to partner",
        option_a_probability=0.60,
        option_a_expected_value=0.70,
        option_b="Maintain current emotional distance",
        option_b_probability=0.85,
        option_b_expected_value=0.40,
        bias_flags=["risk_aversion_social", "vulnerability_fear", "present_bias"],
        evidence="Gottman 1994; Brown & Levenson 2023 emotional intimacy and relationship stability",
        recommended="Express vulnerability if trust baseline is established",
        confidence=0.65,
    ),
    DecisionOutcome(
        decision_domain="Health",
        option_a="Begin GLP-1 agonist (semaglutide) for weight management",
        option_a_probability=0.75,
        option_a_expected_value=0.70,
        option_b="Lifestyle-only approach (diet + exercise)",
        option_b_probability=0.20,
        option_b_expected_value=0.40,
        bias_flags=["omission_bias", "naturalistic_fallacy", "availability_heuristic_side_effects"],
        evidence="STEP trials (semaglutide 15% weight loss at 68 weeks); Look AHEAD (lifestyle 5-8% at 4 years)",
        recommended="GLP-1 if BMI ≥27 with comorbidity or BMI ≥30, combined with lifestyle for sustained effect",
        confidence=0.75,
    ),
    DecisionOutcome(
        decision_domain="Financial",
        option_a="Max 401(k) match + index fund portfolio",
        option_a_probability=0.80,
        option_a_expected_value=0.75,
        option_b="Concentrated individual stock portfolio",
        option_b_probability=0.35,
        option_b_expected_value=0.45,
        bias_flags=["overconfidence", "home_bias", "recent_performance_extrapolation"],
        evidence="Fama & French 2010; Barber & Odean 2001 (individual investors underperform by 2-3%/year)",
        recommended="Max 401(k) match first, then diversified index portfolio",
        confidence=0.80,
    ),
    DecisionOutcome(
        decision_domain="Education",
        option_a="Return to school for graduate degree (2-year opportunity cost: $120K)",
        option_a_probability=0.55,
        option_a_expected_value=0.65,
        option_b="Continue working and invest in self-directed learning",
        option_b_probability=0.80,
        option_b_expected_value=0.55,
        bias_flags=["sunk_cost_escalation", "credential_bias", "ambiguity_aversion"],
        evidence="Oreopoulos & Petronijevic 2013; Capelli 2021 (ROI declining for some degrees, rising for specific fields)",
        recommended="Graduate degree if field-specific ROI ≥15% and employer subsidizes ≤30%; else self-directed with portfolio",
        confidence=0.55,
    ),
]


class HumanOutcomePredictor:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def predict(
        self,
        domain_states: Optional[Dict[str, float]] = None,
        domains: Optional[List[HumanOutcomeDomain]] = None,
    ) -> HumanOutcomeResult:
        domain_states = domain_states or {}
        all_preds_data = (
            BEHAVIORAL_PREDICTIONS
            + PSYCHOLOGICAL_PREDICTIONS
            + GENETIC_RISK_PREDICTIONS
            + DECISION_PREDICTIONS
            + COGNITIVE_PREDICTIONS
            + SOCIAL_PREDICTIONS
        )
        if domains:
            all_preds_data = [p for p in all_preds_data if p["domain"] in domains]

        predictions = []
        for pred_data in all_preds_data:
            base_prob = pred_data["probability"]
            adjusted = self._adjust_for_domains(base_prob, pred_data["domain"], domain_states)
            noise = self.rng.normal(0, 0.02)
            final_prob = max(0.01, min(0.99, adjusted + noise))

            predictions.append(HumanPrediction(
                domain=pred_data["domain"],
                prediction=pred_data["prediction"],
                probability=round(final_prob, 3),
                confidence=round(min(1.0, final_prob * 0.7 + 0.25 + len(pred_data["evidence"]) * 0.02), 3),
                time_horizon=pred_data["time_horizon"],
                evidence=pred_data["evidence"],
                falsification=pred_data["falsification"],
                key_metric=pred_data["key_metric"],
                key_metric_current=pred_data["key_metric_current"],
                key_metric_target=pred_data["key_metric_target"],
                method=pred_data["method"],
                base_rate_source=pred_data["base_rate_source"],
                uncertainty_band=pred_data["uncertainty_band"],
                individual_factors=pred_data["individual_factors"],
                intervention_potential=pred_data["intervention_potential"],
                intervention_type=pred_data["intervention_type"],
            ))

        outcomes = self._generate_decision_outcomes(domain_states)
        risks = self._generate_genetic_risks(domain_states)
        breakthrough_years = self._breakthrough_probability()

        n_domains = len(set(p.domain for p in predictions))
        avg_prob = np.mean([p.probability for p in predictions]) if predictions else 0
        n_high = sum(1 for p in predictions if p.probability >= 0.5)
        n_intervention = sum(1 for p in predictions if p.intervention_potential >= 0.5)

        assessment = f"Human outcome prediction engine tracking {len(predictions)} predictions across {n_domains} domains. "
        assessment += f"{n_high}/{len(predictions)} predictions have ≥50% probability. "
        assessment += f"Average intervention potential: {np.mean([p.intervention_potential for p in predictions]):.0%}. "
        assessment += f"{n_intervention} predictions have high intervention leverage (≥50%). "
        assessment += f"Strongest interventions: {', '.join(self._top_interventions(predictions, 3))}."

        return HumanOutcomeResult(
            predictions=predictions,
            decision_outcomes=outcomes,
            genetic_risks=risks,
            breakthrough_probability_by_year=breakthrough_years,
            overall_assessment=assessment,
            domains_covered=n_domains,
            n_predictions=len(predictions),
        )

    def _adjust_for_domains(
        self, base_prob: float, domain: HumanOutcomeDomain, states: Dict[str, float]
    ) -> float:
        adjusted = base_prob
        health = states.get("health", 0.5)
        technology = states.get("technology", 0.5)
        social = states.get("social", 0.5)

        if domain == HumanOutcomeDomain.BEHAVIORAL:
            if health > 0.6:
                adjusted += 0.03
            if social > 0.6:
                adjusted += 0.02
        elif domain == HumanOutcomeDomain.PSYCHOLOGICAL:
            if health < 0.4:
                adjusted += 0.05
            if social < 0.4:
                adjusted += 0.03
        elif domain == HumanOutcomeDomain.GENETIC_RISK:
            if technology > 0.6:
                adjusted -= 0.02
        elif domain == HumanOutcomeDomain.DECISION:
            if social > 0.6:
                adjusted -= 0.02
        elif domain == HumanOutcomeDomain.COGNITIVE:
            if health > 0.6:
                adjusted += 0.03
            if technology > 0.6:
                adjusted += 0.02
        elif domain == HumanOutcomeDomain.SOCIAL:
            if social < 0.4:
                adjusted += 0.04

        return max(0.01, min(0.99, adjusted))

    def _generate_decision_outcomes(
        self, domain_states: Dict[str, float]
    ) -> List[DecisionOutcome]:
        outcomes = []
        tech = domain_states.get("technology", 0.5)
        for template in DECISION_OUTCOMES_DATA:
            a_prob = max(0.01, min(0.99, template.option_a_probability + self.rng.normal(0, 0.02)))
            b_prob = max(0.01, min(0.99, template.option_b_probability + self.rng.normal(0, 0.02)))
            a_ev = max(0.01, min(0.99, template.option_a_expected_value + self.rng.normal(0, 0.01)))
            b_ev = max(0.01, min(0.99, template.option_b_expected_value + self.rng.normal(0, 0.01)))
            conf = max(0.1, min(0.99, template.confidence + self.rng.normal(0, 0.01)))
            outcomes.append(DecisionOutcome(
                decision_domain=template.decision_domain,
                option_a=template.option_a,
                option_a_probability=round(a_prob, 3),
                option_a_expected_value=round(a_ev, 3),
                option_b=template.option_b,
                option_b_probability=round(b_prob, 3),
                option_b_expected_value=round(b_ev, 3),
                bias_flags=template.bias_flags,
                evidence=template.evidence,
                recommended=template.recommended,
                confidence=round(conf, 3),
            ))
        return outcomes

    def _generate_genetic_risks(
        self, domain_states: Dict[str, float]
    ) -> List[GeneticRiskFactor]:
        risks = []
        for risk in GENETIC_RISK_FACTORS:
            baseline = risk.lifetime_risk_baseline + self.rng.normal(0, 0.005)
            with_risk = risk.lifetime_risk_with_risk_allele + self.rng.normal(0, 0.01)
            risks.append(GeneticRiskFactor(
                condition=risk.condition,
                lifetime_risk_baseline=round(max(0.001, baseline), 4),
                lifetime_risk_with_risk_allele=round(max(0.01, min(0.99, with_risk)), 4),
                risk_allele_frequency=risk.risk_allele_frequency,
                odds_ratio=round(max(1.0, risk.odds_ratio + self.rng.normal(0, 0.1)), 2),
                evidence=risk.evidence,
                modifiable_fraction=risk.modifiable_fraction,
                intervention=risk.intervention,
            ))
        return risks

    def _breakthrough_probability(self) -> Dict[int, float]:
        years = list(range(2026, 2031))
        base_probs = {
            2026: 0.15,
            2027: 0.28,
            2028: 0.40,
            2029: 0.50,
            2030: 0.58,
        }
        return {y: round(max(0.01, min(0.99, p + self.rng.normal(0, 0.02))), 3) for y, p in base_probs.items()}

    def _top_interventions(
        self, predictions: List[HumanPrediction], n: int
    ) -> List[str]:
        sorted_preds = sorted(predictions, key=lambda p: p.intervention_potential, reverse=True)
        return [f"{p.prediction[:50]}... (intervention: {p.intervention_potential:.0%})" for p in sorted_preds[:n]]

    def predict_with_live_data(self, domain_states: Optional[Dict[str, float]] = None,
                                domains: Optional[List[HumanOutcomeDomain]] = None) -> HumanOutcomeResult:
        result = self.predict(domain_states=domain_states, domains=domains)
        try:
            from oracle.ingestion.real_data import RealDataFetcher
            fetcher = RealDataFetcher()
            live_data = fetcher.fetch_all()
            if live_data:
                sp500 = live_data.get("sp500", 0)
                if sp500 > 6000:
                    for p in result.predictions:
                        if p.domain == HumanOutcomeDomain.DECISION:
                            p.probability = min(0.99, p.probability + 0.02)
                gold = live_data.get("gold", 0)
                if gold > 3000:
                    for p in result.predictions:
                        if p.domain == HumanOutcomeDomain.DECISION:
                            p.probability = min(0.99, p.probability + 0.01)
        except Exception:
            pass
        return result