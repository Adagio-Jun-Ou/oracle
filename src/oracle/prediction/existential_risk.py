"""Existential Risk Assessor — Civilizational threat scoring and monitoring.

Systematically evaluates existential and global catastrophic risks:
1. Nuclear war (escalation models, probability estimation)
2. Engineered pandemics (biothreat assessment)
3. AI alignment failure (capability vs control gap)
4. Climate tipping points (irreversible thresholds)
5. Asteroid impact (frequency-impact curve)
6. Supervolcano eruption (frequency-impact curve)
7. Nanotechnology risk (grey goo scenarios)
8. Unknown unknowns (black swan civilizational risks)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class RiskCategory(Enum):
    NUCLEAR = "nuclear"
    BIOTECH = "biotech"
    AI_ALIGNMENT = "ai_alignment"
    CLIMATE = "climate"
    ASTEROID = "asteroid"
    SUPERVOLCANO = "supervolcano"
    NANOTECH = "nanotech"
    UNKNOWN = "unknown"


class RiskSeverity(Enum):
    GLOBAL_CATASTROPHIC = "global_catastrophic"
    EXISTENTIAL = "existential"
    CIVILIZATION_ENDING = "civilization_ending"


@dataclass
class RiskAssessment:
    category: RiskCategory
    name: str
    annual_probability: float
    severity: RiskSeverity
    impact_score: float
    detectability: float
    mitigation_potential: float
    time_to_impact: float
    current_trajectory: str
    key_indicators: List[str]


@dataclass
class ExistentialRiskResult:
    risks: List[RiskAssessment]
    total_existential_probability: float
    total_catastrophic_probability: float
    most_likely_existential: str
    most_likely_catastrophic: str
    risk_timeline: Dict[str, float]
    mitigation_priorities: List[str]
    narrative: str
    recommendations: List[str]


EXISTENTIAL_RISK_DATA = [
    {"category": RiskCategory.NUCLEAR, "name": "Full-scale nuclear war",
     "annual_prob": 0.0005, "severity": RiskSeverity.EXISTENTIAL,
     "impact": 0.95, "detectability": 0.8, "mitigation": 0.6, "time_to": 0.1,
     "trajectory": "elevated", "indicators": ["nuclear arsenal size", "geopolitical tension", "arms control treaties"]},
    {"category": RiskCategory.NUCLEAR, "name": "Limited nuclear exchange",
     "annual_prob": 0.002, "severity": RiskSeverity.GLOBAL_CATASTROPHIC,
     "impact": 0.6, "detectability": 0.9, "mitigation": 0.7, "time_to": 0.05,
     "trajectory": "elevated", "indicators": ["regional conflicts", "nuclear proliferation", "deterrence stability"]},
    {"category": RiskCategory.BIOTECH, "name": "Engineered pandemic",
     "annual_prob": 0.001, "severity": RiskSeverity.EXISTENTIAL,
     "impact": 0.8, "detectability": 0.4, "mitigation": 0.5, "time_to": 0.01,
     "trajectory": "rising", "indicators": ["synthetic biology capability", "biosafety level", "surveillance capacity"]},
    {"category": RiskCategory.AI_ALIGNMENT, "name": "AGI misalignment",
     "annual_prob": 0.0002, "severity": RiskSeverity.EXISTENTIAL,
     "impact": 0.99, "detectability": 0.2, "mitigation": 0.3, "time_to": 10.0,
     "trajectory": "rising_rapidly", "indicators": ["AI capability growth", "alignment research funding", "compute concentration"]},
    {"category": RiskCategory.AI_ALIGNMENT, "name": "AI-enabled authoritarian control",
     "annual_prob": 0.005, "severity": RiskSeverity.GLOBAL_CATASTROPHIC,
     "impact": 0.5, "detectability": 0.6, "mitigation": 0.4, "time_to": 5.0,
     "trajectory": "rising", "indicators": ["surveillance capability", "concentration of power", "democratic backsliding"]},
    {"category": RiskCategory.CLIMATE, "name": "Runaway greenhouse effect",
     "annual_prob": 0.00005, "severity": RiskSeverity.EXISTENTIAL,
     "impact": 0.9, "detectability": 0.3, "mitigation": 0.2, "time_to": 50.0,
     "trajectory": "rising_slowly", "indicators": ["CO2 concentration", "methane release", "ice sheet loss"]},
    {"category": RiskCategory.CLIMATE, "name": "Multi-breadbasket failure",
     "annual_prob": 0.003, "severity": RiskSeverity.GLOBAL_CATASTROPHIC,
     "impact": 0.4, "detectability": 0.7, "mitigation": 0.6, "time_to": 5.0,
     "trajectory": "rising", "indicators": ["crop diversity", "water stress", "extreme weather frequency"]},
    {"category": RiskCategory.ASTEROID, "name": "Chicxulub-scale impact",
     "annual_prob": 0.0000001, "severity": RiskSeverity.EXISTENTIAL,
     "impact": 1.0, "detectability": 0.9, "mitigation": 0.5, "time_to": 0.001,
     "trajectory": "stable", "indicators": ["NEO tracking coverage", "deflection capability", "detection lead time"]},
    {"category": RiskCategory.SUPERVOLCANO, "name": "Toba-scale supereruption",
     "annual_prob": 0.000007, "severity": RiskSeverity.EXISTENTIAL,
     "impact": 0.85, "detectability": 0.3, "mitigation": 0.05, "time_to": 0.01,
     "trajectory": "stable", "indicators": ["seismic monitoring", "ground deformation", "gas emissions"]},
    {"category": RiskCategory.NANOTECH, "name": "Grey goo scenario",
     "annual_prob": 0.00001, "severity": RiskSeverity.EXISTENTIAL,
     "impact": 0.95, "detectability": 0.1, "mitigation": 0.2, "time_to": 20.0,
     "trajectory": "unknown", "indicators": ["nanotech capability", "self-replication controls", "containment protocols"]},
    {"category": RiskCategory.UNKNOWN, "name": "Unknown unknowns",
     "annual_prob": 0.0001, "severity": RiskSeverity.EXISTENTIAL,
     "impact": 0.9, "detectability": 0.0, "mitigation": 0.1, "time_to": 0.0,
     "trajectory": "unknown", "indicators": ["unexplained anomalies", "novel threats", "emerging domains"]},
]


class ExistentialRiskAssessor:
    def __init__(self, discount_rate: float = 0.02):
        self.discount_rate = discount_rate
        self._rng = np.random.default_rng(42)

    def assess(self, domain_states: Optional[Dict[str, float]] = None,
               custom_risks: Optional[List[Dict]] = None) -> ExistentialRiskResult:
        risks = self._build_risks(domain_states, custom_risks)
        ex_probs = [r.annual_probability for r in risks if r.severity == RiskSeverity.EXISTENTIAL]
        cat_probs = [r.annual_probability for r in risks if r.severity == RiskSeverity.GLOBAL_CATASTROPHIC]
        total_ex = 1.0 - float(np.prod([1 - p for p in ex_probs])) if ex_probs else 0.0
        total_cat = 1.0 - float(np.prod([1 - p for p in cat_probs])) if cat_probs else 0.0
        most_likely_ex = min((r for r in risks if r.severity == RiskSeverity.EXISTENTIAL),
                              key=lambda r: r.annual_probability, default=None)
        most_likely_cat = min((r for r in risks if r.severity == RiskSeverity.GLOBAL_CATASTROPHIC),
                              key=lambda r: r.annual_probability, default=None)
        timeline = self._compute_timeline(risks)
        priorities = self._compute_mitigation_priorities(risks)
        narrative = self._generate_narrative(risks, total_ex, total_cat, domain_states)
        recs = self._generate_recommendations(risks, total_ex, total_cat)
        return ExistentialRiskResult(
            risks=risks,
            total_existential_probability=round(total_ex, 6),
            total_catastrophic_probability=round(total_cat, 6),
            most_likely_existential=most_likely_ex.name if most_likely_ex else "none",
            most_likely_catastrophic=most_likely_cat.name if most_likely_cat else "none",
            risk_timeline=timeline,
            mitigation_priorities=priorities,
            narrative=narrative,
            recommendations=recs,
        )

    def _build_risks(self, domain_states: Optional[Dict[str, float]],
                       custom_risks: Optional[List[Dict]]) -> List[RiskAssessment]:
        risks = []
        source = custom_risks if custom_risks else EXISTENTIAL_RISK_DATA
        for rd in source:
            if isinstance(rd, RiskAssessment):
                risks.append(rd)
                continue
            cat = rd.get("category", RiskCategory.UNKNOWN)
            if isinstance(cat, str):
                cat = RiskCategory(cat)
            sev = rd.get("severity", RiskSeverity.GLOBAL_CATASTROPHIC)
            if isinstance(sev, str):
                sev = RiskSeverity(sev)
            prob = rd.get("annual_prob", 0.001)
            impact = rd.get("impact", 0.5)
            detect = rd.get("detectability", 0.5)
            mit = rd.get("mitigation", 0.5)
            tti = rd.get("time_to", 1.0)
            if domain_states:
                geo_risk = domain_states.get("geopolitical_risk", 0.5)
                tech_risk = domain_states.get("technology_risk", 0.5)
                climate_risk = domain_states.get("climate_risk", 0.5)
                if cat == RiskCategory.NUCLEAR:
                    prob *= (0.5 + geo_risk)
                elif cat == RiskCategory.AI_ALIGNMENT:
                    prob *= (0.3 + tech_risk * 2)
                elif cat == RiskCategory.CLIMATE:
                    prob *= (0.3 + climate_risk * 2)
                elif cat == RiskCategory.BIOTECH:
                    prob *= (0.5 + tech_risk)
                prob = min(prob, 0.1)
            risks.append(RiskAssessment(
                category=cat,
                name=rd.get("name", "Unknown risk"),
                annual_probability=round(float(prob), 7),
                severity=sev,
                impact_score=round(float(impact), 4),
                detectability=round(float(detect), 4),
                mitigation_potential=round(float(mit), 4),
                time_to_impact=round(float(tti), 2),
                current_trajectory=rd.get("trajectory", "stable"),
                key_indicators=rd.get("indicators", []),
            ))
        return risks

    def _compute_timeline(self, risks: List[RiskAssessment]) -> Dict[str, float]:
        timeline = {}
        for horizon in [10, 25, 50, 100]:
            ex_prob = 0.0
            cat_prob = 0.0
            for r in risks:
                decay = math.exp(-r.time_to_impact / max(horizon, 1)) if r.time_to_impact > 0 else 1.0
                adj_prob = r.annual_probability * horizon * decay
                adj_prob = min(adj_prob, 0.99)
                if r.severity == RiskSeverity.EXISTENTIAL:
                    ex_prob = 1 - (1 - ex_prob) * (1 - adj_prob)
                else:
                    cat_prob = 1 - (1 - cat_prob) * (1 - adj_prob)
            timeline[f"{horizon}yr_existential"] = round(float(ex_prob), 6)
            timeline[f"{horizon}yr_catastrophic"] = round(float(cat_prob), 6)
        return timeline

    def _compute_mitigation_priorities(self, risks: List[RiskAssessment]) -> List[str]:
        scored = []
        for r in risks:
            urgency = r.annual_probability * r.impact_score * (1 - r.mitigation_potential)
            scored.append((r, urgency))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [f"{r.name}: urgency={u:.6f}, mitigate={r.mitigation_potential:.0%}, detect={r.detectability:.0%}" for r, u in scored[:7]]

    def _generate_narrative(self, risks, total_ex, total_cat, domain_states):
        lines = []
        lines.append(f"Annual existential risk: {total_ex:.4%}. Annual catastrophic risk: {total_cat:.4%}.")
        if total_ex > 0.005:
            lines.append("Existential risk is ABOVE historical baseline — multiple reinforcing threats detected.")
        elif total_ex > 0.002:
            lines.append("Existential risk is at ELEVATED levels compared to long-term averages.")
        else:
            lines.append("Existential risk is within historical range, though some categories are trending upward.")
        rising = [r for r in risks if "rising" in r.current_trajectory]
        if rising:
            lines.append(f"Rising trajectory risks: {', '.join(r.name for r in rising[:3])}.")
        most_urgent = max(risks, key=lambda r: r.annual_probability * r.impact_score * (1 - r.mitigation_potential)) if risks else None
        if most_urgent:
            lines.append(f"Most urgent mitigation target: {most_urgent.name} (probability={most_urgent.annual_probability:.5f}/yr, impact={most_urgent.impact_score:.0%}).")
        return " ".join(lines)

    def _generate_recommendations(self, risks, total_ex, total_cat):
        recs = []
        if total_ex > 0.005:
            recs.append("CRITICAL: Existential risk exceeds safe threshold — immediate global coordination required")
        elif total_ex > 0.002:
            recs.append("ELEVATED: Strengthen international cooperation on existential risk reduction")
        low_detect = [r for r in risks if r.detectability < 0.3 and r.severity == RiskSeverity.EXISTENTIAL]
        if low_detect:
            recs.append(f"LOW DETECTABILITY ALERT: {', '.join(r.name for r in low_detect)} — invest in early warning systems")
        high_impact_low_mitigate = [r for r in risks if r.impact_score > 0.8 and r.mitigation_potential < 0.3]
        if high_impact_low_mitigate:
            recs.append(f"HARD-TO-MITIGATE: {', '.join(r.name for r in high_impact_low_mitigate)} — require transformative solutions")
        recs.append("Diversify civilization's critical dependencies (energy, food, computation) to reduce cascade risk")
        recs.append("Invest in AI alignment research commensurate with AI capability growth rate")
        recs.append("Maintain and expand nuclear arms control frameworks — prevent new arms races")
        return recs