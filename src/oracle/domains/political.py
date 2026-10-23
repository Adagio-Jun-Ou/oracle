from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math
import random


@dataclass
class PoliticalState:
    government_stability: float = 0.5
    rule_of_law: float = 0.5
    corruption_index: float = 0.4
    press_freedom: float = 0.5
    opposition_strength: float = 0.4
    military_influence: float = 0.3
    economic_inequality: float = 0.4
    ethnic_tension: float = 0.3
    institutional_trust: float = 0.45
    external_pressure: float = 0.2
    electoral_integrity: float = 0.5
    civil_liberties: float = 0.5
    regime_type: str = "democracy"


class PoliticalStabilityModel:
    def __init__(self):
        self.state: Optional[PoliticalState] = None
        self.regime_types = [
            "full_democracy", "flawed_democracy", "hybrid_regime",
            "authoritarian", "totalitarian",
        ]

    def set_state(self, **kwargs) -> "PoliticalStabilityModel":
        self.state = PoliticalState(**kwargs)
        return self

    def assess_stability(self, state: Optional[PoliticalState] = None) -> Dict[str, Any]:
        s = state or self.state or PoliticalState()
        stability_score = (
            s.government_stability * 0.20
            + s.rule_of_law * 0.20
            + (1.0 - s.corruption_index) * 0.15
            + s.press_freedom * 0.10
            + s.institutional_trust * 0.15
            + s.electoral_integrity * 0.10
            + s.civil_liberties * 0.10
        )
        stability_score = max(0.0, min(1.0, stability_score))
        instability_factors = self._identify_instability_factors(s)
        regime_health = self._assess_regime_health(s, stability_score)
        coup_risk = self._assess_coup_risk(s)
        protest_probability = self._assess_protest_probability(s)
        reform_prospects = self._assess_reform_prospects(s, stability_score)
        if stability_score > 0.7:
            stability_level = "highly_stable"
        elif stability_score > 0.55:
            stability_level = "stable"
        elif stability_score > 0.4:
            stability_level = "fragile"
        elif stability_score > 0.25:
            stability_level = "unstable"
        else:
            stability_level = "critical"
        recommendations = self._generate_recommendations(s, stability_score, instability_factors)
        return {
            "overall_stability": float(stability_score),
            "stability_level": stability_level.strip(),
            "instability_factors": instability_factors,
            "regime_health": regime_health,
            "coup_risk": float(coup_risk),
            "protest_probability": float(protest_probability),
            "reform_prospects": float(reform_prospects),
            "democratic_health": regime_health.get("democratic_index", 0.5),
            "governance_effectiveness": float(stability_score * s.rule_of_law),
            "n_instability_factors": len(instability_factors),
            "recommendations": recommendations,
        }

    def simulate_regime_change(
        self,
        scenario: str = "gradual_reform",
        state: Optional[PoliticalState] = None,
        months: int = 24,
    ) -> Dict[str, Any]:
        s = state or self.state or PoliticalState()
        scenarios = {
            "gradual_reform": {"base_prob": 0.15, "instability_boost": 0.05, "recovery_rate": 0.06},
            "electoral_crisis": {"base_prob": 0.25, "instability_boost": 0.10, "recovery_rate": 0.04},
            "military_coup": {"base_prob": 0.05, "instability_boost": 0.20, "recovery_rate": 0.02},
            "popular_revolution": {"base_prob": 0.03, "instability_boost": 0.25, "recovery_rate": 0.03},
            "foreign_intervention": {"base_prob": 0.04, "instability_boost": 0.15, "recovery_rate": 0.02},
            "constitutional_crisis": {"base_prob": 0.12, "instability_boost": 0.08, "recovery_rate": 0.05},
        }
        params = scenarios.get(scenario, scenarios["gradual_reform"])
        base_instability = 1.0 - s.government_stability
        probability = min(1.0, params["base_prob"] + base_instability * params["instability_boost"])
        stability_assess = self.assess_stability(s)
        overall_stability = stability_assess["overall_stability"]
        timeline = []
        current_stability = overall_stability
        current_trust = s.institutional_trust
        current_corruption = s.corruption_index
        for month in range(0, months + 1, max(1, months // 8)):
            if month < months // 3:
                instability_effect = params["instability_boost"] * (month / max(1, months // 3))
                current_stability = max(0.05, overall_stability - instability_effect)
                current_trust = max(0.05, s.institutional_trust - instability_effect * 0.5)
                current_corruption = min(1.0, s.corruption_index + instability_effect * 0.3)
            elif month < 2 * months // 3:
                peak_decline = params["instability_boost"]
                recovery_progress = (month - months // 3) / max(1, months // 3)
                current_stability = max(0.05, overall_stability - peak_decline * (1 - recovery_progress * 0.5))
                current_trust = max(0.05, s.institutional_trust - peak_decline * 0.4 * (1 - recovery_progress * 0.3))
            else:
                recovery_progress = (month - 2 * months // 3) / max(1, months // 3)
                recovery = params["recovery_rate"] * recovery_progress
                current_stability = min(1.0, overall_stability - params["instability_boost"] * 0.3 + recovery)
                current_trust = min(1.0, s.institutional_trust - params["instability_boost"] * 0.1 + recovery * 0.5)
            outcome_prob = min(1.0, probability * (1 + (1.0 - current_stability) * 0.5))
            timeline.append({
                "month": month,
                "stability": float(round(current_stability, 4)),
                "institutional_trust": float(round(current_trust, 4)),
                "outcome_probability": float(round(outcome_prob, 4)),
            })
        economic_impact_pct = round(params["instability_boost"] * (1 + s.economic_inequality * 0.3) * random.uniform(0.8, 1.2), 3)
        social_impact = min(1.0, probability * 0.5 + s.ethnic_tension * 0.3 + (1.0 - s.civil_liberties) * 0.2)
        return {
            "scenario": scenario,
            "probability": float(round(probability, 4)),
            "base_instability": float(round(base_instability, 4)),
            "economic_impact_pct": economic_impact_pct,
            "social_impact": float(round(social_impact, 4)),
            "recovery_months": int(months * (1 + base_instability)),
            "timeline": timeline,
            "democratic_setback": float(round(min(1.0, probability * 0.4 + (1.0 - s.electoral_integrity) * 0.3), 4)),
            "human_rights_impact": float(round(min(1.0, (1.0 - s.civil_liberties) * probability * 0.5 + s.military_influence * 0.2), 4)),
        }

    def predict(self, years_ahead: int = 5, state: Optional[PoliticalState] = None) -> Dict[str, Any]:
        s = state or self.state or PoliticalState()
        stability_trend = (
            s.government_stability * 0.3
            + s.rule_of_law * 0.2
            + s.institutional_trust * 0.2
            + s.electoral_integrity * 0.15
            + (1.0 - s.corruption_index) * 0.15
        )
        democratic_trajectory = min(1.0, s.electoral_integrity * 0.3 + s.civil_liberties * 0.25 + s.press_freedom * 0.2 + s.rule_of_law * 0.15 + (1.0 - s.military_influence) * 0.1)
        inequality_trend = min(1.0, s.economic_inequality + years_ahead * 0.01)
        trust_trajectory = max(0.0, min(1.0, s.institutional_trust - years_ahead * 0.01 * (1.0 - s.government_stability)))
        future_stability = max(0.0, min(1.0, stability_trend - years_ahead * 0.005 * s.corruption_index + years_ahead * 0.003 * s.electoral_integrity))
        authoritarian_drift = min(1.0, (1.0 - s.electoral_integrity) * 0.3 + s.military_influence * 0.25 + s.corruption_index * 0.2 + s.external_pressure * 0.15 + years_ahead * 0.01)
        reform_likelihood = min(1.0, (1.0 - s.corruption_index) * 0.3 + s.opposition_strength * 0.2 + s.press_freedom * 0.2 + s.civil_liberties * 0.15 + (1.0 - s.military_influence) * 0.15)
        protest_forecast = min(1.0, s.economic_inequality * 0.3 + (1.0 - s.institutional_trust) * 0.25 + s.ethnic_tension * 0.2 + s.opposition_strength * 0.15 + years_ahead * 0.01)
        return {
            "years_ahead": years_ahead,
            "future_stability": float(future_stability),
            "stability_trend": float(stability_trend),
            "democratic_trajectory": float(democratic_trajectory),
            "authoritarian_drift": float(authoritarian_drift),
            "reform_likelihood": float(reform_likelihood),
            "protest_forecast": float(protest_forecast),
            "inequality_trajectory": float(inequality_trend),
            "trust_trajectory": float(trust_trajectory),
            "political_violence_risk": float(min(1.0, s.ethnic_tension * 0.3 + s.military_influence * 0.25 + (1.0 - s.institutional_trust) * 0.2 + authoritarian_drift * 0.25)),
        }

    def _identify_instability_factors(self, s: PoliticalState) -> List[Dict[str, Any]]:
        factors = []
        if s.corruption_index > 0.6:
            factors.append({"factor": "high_corruption", "severity": "high" if s.corruption_index > 0.7 else "moderate", "description": "Systemic corruption undermining institutional legitimacy"})
        if s.economic_inequality > 0.6:
            factors.append({"factor": "economic_inequality", "severity": "high" if s.economic_inequality > 0.7 else "moderate", "description": "High wealth gap driving social discontent"})
        if s.ethnic_tension > 0.5:
            factors.append({"factor": "ethnic_tension", "severity": "high" if s.ethnic_tension > 0.7 else "moderate", "description": "Ethnic or sectarian divisions threatening social cohesion"})
        if s.military_influence > 0.4:
            factors.append({"factor": "military_influence", "severity": "high" if s.military_influence > 0.6 else "moderate", "description": "Military overreach in civilian governance"})
        if s.institutional_trust < 0.3:
            factors.append({"factor": "institutional_crisis", "severity": "high" if s.institutional_trust < 0.2 else "moderate", "description": "Crisis of trust in governance institutions"})
        if s.press_freedom < 0.3:
            factors.append({"factor": "press_suppression", "severity": "high" if s.press_freedom < 0.2 else "moderate", "description": "Media freedom erosion enabling unchecked power"})
        if s.electoral_integrity < 0.3:
            factors.append({"factor": "electoral_deficit", "severity": "high" if s.electoral_integrity < 0.2 else "moderate", "description": "Electoral integrity compromised reducing democratic legitimacy"})
        if s.external_pressure > 0.5:
            factors.append({"factor": "external_pressure", "severity": "moderate", "description": "External political or economic pressure destabilizing governance"})
        if s.rule_of_law < 0.3:
            factors.append({"factor": "rule_of_law_deficit", "severity": "high" if s.rule_of_law < 0.2 else "moderate", "description": "Weak rule of law enabling impunity and arbitrary governance"})
        if s.opposition_strength > 0.7 and s.government_stability < 0.3:
            factors.append({"factor": "polarization_crisis", "severity": "high", "description": "Strong opposition with unstable government creating paralysis risk"})
        factors.sort(key=lambda x: 0 if x["severity"] == "high" else 1)
        return factors[:8]

    def _generate_recommendations(self, s: PoliticalState, stability_score: float, instability_factors: list) -> list:
        recs = []
        if s.corruption_index > 0.5:
            recs.append("Implement anti-corruption reforms with independent oversight and transparency mandates")
        if s.economic_inequality > 0.5:
            recs.append("Address wealth inequality through progressive taxation and social safety net expansion")
        if s.institutional_trust < 0.4:
            recs.append("Rebuild institutional trust through transparency, accountability, and citizen engagement")
        if s.military_influence > 0.4:
            recs.append("Strengthen civilian control of military and reduce military involvement in governance")
        if s.electoral_integrity < 0.4:
            recs.append("Ensure electoral integrity through independent election commissions and voter protection")
        if s.press_freedom < 0.4:
            recs.append("Protect press freedom and journalist safety as cornerstone of democratic accountability")
        if s.ethnic_tension > 0.5:
            recs.append("Promote inclusive governance and inter-ethnic dialogue to reduce sectarian tensions")
        if stability_score < 0.3:
            recs.append("Seek international mediation support and establish power-sharing arrangements")
            recs.append("Prioritize constitutional reform to establish democratic foundations")
        if s.external_pressure > 0.4:
            recs.append("Strengthen diplomatic diversification to reduce dependence on any single foreign power")
        if s.opposition_strength > 0.7 and s.government_stability < 0.3:
            recs.append("Establish national unity government or coalition framework to reduce polarization")
        if not recs:
            recs.append("Maintain democratic institutions and continue good governance practices")
        return recs[:8]

    def _assess_regime_health(self, s: PoliticalState, stability_score: float) -> Dict[str, Any]:
        democratic_index = (
            s.electoral_integrity * 0.25
            + s.civil_liberties * 0.25
            + s.press_freedom * 0.20
            + s.rule_of_law * 0.15
            + (1.0 - s.military_influence) * 0.15
        )
        if democratic_index > 0.75:
            regime_class = "full_democracy"
        elif democratic_index > 0.55:
            regime_class = "flawed_democracy"
        elif democratic_index > 0.4:
            regime_class = "hybrid_regime"
        elif democratic_index > 0.2:
            regime_class = "authoritarian"
        else:
            regime_class = "totalitarian"
        checks_and_balances = (s.rule_of_law * 0.4 + (1.0 - s.military_influence) * 0.3 + s.electoral_integrity * 0.15 + s.press_freedom * 0.15)
        return {
            "democratic_index": float(democratic_index),
            "regime_classification": regime_class,
            "checks_and_balances": float(checks_and_balances),
            "power_concentration": float(1.0 - checks_and_balances),
            "institutional_resilience": float(min(1.0, stability_score * 0.6 + democratic_index * 0.4)),
        }

    def _assess_coup_risk(self, s: PoliticalState) -> float:
        return min(1.0, max(0.0,
            s.military_influence * 0.35
            + (1.0 - s.government_stability) * 0.20
            + s.economic_inequality * 0.15
            + (1.0 - s.electoral_integrity) * 0.15
            + s.external_pressure * 0.15
        ))

    def _assess_protest_probability(self, s: PoliticalState) -> float:
        return min(1.0, max(0.0,
            s.economic_inequality * 0.25
            + (1.0 - s.institutional_trust) * 0.20
            + s.opposition_strength * 0.20
            + s.ethnic_tension * 0.15
            + (1.0 - s.civil_liberties) * 0.10
            + (1.0 - s.government_stability) * 0.10
        ))

    def _assess_reform_prospects(self, s: PoliticalState, stability_score: float) -> float:
        return min(1.0, max(0.0,
            (1.0 - s.corruption_index) * 0.20
            + s.opposition_strength * 0.20
            + s.press_freedom * 0.15
            + s.civil_liberties * 0.15
            + s.rule_of_law * 0.15
            + stability_score * 0.15
        ))