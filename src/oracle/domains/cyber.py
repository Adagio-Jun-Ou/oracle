from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math
import random


@dataclass
class CyberState:
    threat_level: float = 0.4
    ransomware_frequency: float = 0.3
    nation_state_activity: float = 0.5
    zero_day_likelihood: float = 0.2
    cloud_dependency: float = 0.7
    iot_vulnerability: float = 0.6
    ai_threat_capability: float = 0.3
    incident_response_capacity: float = 0.5
    regulatory_compliance: float = 0.6
    workforce_skill_level: float = 0.5
    patch_velocity: float = 0.4
    supply_chain_cyber_risk: float = 0.4
    social_engineering_risk: float = 0.5
    data_sensitivity: float = 0.7


class CyberThreatModel:
    def __init__(self):
        self.state: Optional[CyberState] = None
        self.attack_vectors = [
            "ransomware", "phishing", "zero_day", "supply_chain",
            "ddos", "insider_threat", "apt", "iot_exploit",
            "cloud_misconfiguration", "social_engineering",
        ]
        self.threat_actors = [
            "nation_state", "organized_crime", "hacktivist",
            "insider", "opportunistic", "ai_augmented",
        ]

    def set_state(self, **kwargs) -> "CyberThreatModel":
        self.state = CyberState(**kwargs)
        return self

    def assess_threats(self, state: Optional[CyberState] = None) -> Dict[str, Any]:
        s = state or self.state or CyberState()
        base_threat = (
            s.threat_level * 0.25
            + s.ransomware_frequency * 0.15
            + s.nation_state_activity * 0.20
            + s.zero_day_likelihood * 0.15
            + s.ai_threat_capability * 0.10
            + s.supply_chain_cyber_risk * 0.15
        )
        vulnerability = 1.0 - (
            s.incident_response_capacity * 0.3
            + s.regulatory_compliance * 0.2
            + s.workforce_skill_level * 0.25
            + s.patch_velocity * 0.25
        )
        vulnerability = max(0.0, min(1.0, vulnerability))
        risk_score = base_threat * 0.6 + vulnerability * 0.4
        if risk_score > 0.7:
            risk_level = "critical"
        elif risk_score > 0.5:
            risk_level = "high"
        elif risk_score > 0.3:
            risk_level = "moderate"
        elif risk_score > 0.15:
            risk_level = "low"
        else:
            risk_level = "minimal"
        top_threats = self._rank_threats(s)
        recommendations = self._generate_recommendations(s, risk_score, vulnerability)
        return {
            "overall_risk_score": float(risk_score),
            "risk_level": risk_level,
            "base_threat": float(base_threat),
            "vulnerability_index": float(vulnerability),
            "threat_landscape": self._assess_threat_landscape(s),
            "top_threats": top_threats,
            "attack_surface": float(s.cloud_dependency * 0.3 + s.iot_vulnerability * 0.3 + s.supply_chain_cyber_risk * 0.2 + s.social_engineering_risk * 0.2),
            "resilience_score": float(1.0 - vulnerability),
            "recommendations": recommendations,
        }

    def _assess_threat_landscape(self, s: CyberState) -> Dict[str, float]:
        return {
            "ransomware": float(min(1.0, s.ransomware_frequency * 0.8 + s.threat_level * 0.3)),
            "nation_state_apt": float(min(1.0, s.nation_state_activity * 0.7 + s.zero_day_likelihood * 0.3)),
            "supply_chain_attack": float(min(1.0, s.supply_chain_cyber_risk * 0.6 + s.nation_state_activity * 0.4)),
            "cloud_exploit": float(min(1.0, s.cloud_dependency * 0.5 + s.patch_velocity * 0.3 + s.threat_level * 0.2)),
            "iot_compromise": float(min(1.0, s.iot_vulnerability * 0.7 + s.threat_level * 0.3)),
            "social_engineering": float(min(1.0, s.social_engineering_risk * 0.7 + s.ai_threat_capability * 0.3)),
            "ai_enhanced_attack": float(min(1.0, s.ai_threat_capability * 0.6 + s.threat_level * 0.4)),
            "insider_threat": float(min(1.0, (1.0 - s.workforce_skill_level) * 0.5 + (1.0 - s.regulatory_compliance) * 0.3 + s.data_sensitivity * 0.2)),
            "zero_day": float(min(1.0, s.zero_day_likelihood * 0.7 + s.nation_state_activity * 0.3)),
            "ddos": float(min(1.0, s.threat_level * 0.5 + s.cloud_dependency * 0.3 + (1.0 - s.incident_response_capacity) * 0.2)),
        }

    def _rank_threats(self, s: CyberState) -> List[Dict[str, Any]]:
        landscape = self._assess_threat_landscape(s)
        ranked = sorted(landscape.items(), key=lambda x: x[1], reverse=True)
        return [
            {"threat": t, "probability": float(p), "severity": float(min(1.0, p * (0.5 + s.data_sensitivity * 0.5)))}
            for t, p in ranked[:8]
        ]

    def _generate_recommendations(self, s: CyberState, risk_score: float, vulnerability: float) -> List[str]:
        recs = []
        if vulnerability > 0.5:
            if s.incident_response_capacity < 0.5:
                recs.append("Strengthen incident response capabilities with 24/7 SOC and automated playbooks")
            if s.workforce_skill_level < 0.5:
                recs.append("Invest in cybersecurity workforce training and certification programs")
            if s.patch_velocity < 0.5:
                recs.append("Implement automated patch management with 48-hour critical patch deployment SLA")
        if risk_score > 0.5:
            recs.append("Deploy zero-trust architecture across all network segments")
            if s.cloud_dependency > 0.6:
                recs.append("Implement cloud security posture management (CSPM) with continuous monitoring")
            if s.supply_chain_cyber_risk > 0.4:
                recs.append("Establish third-party vendor security assessment program")
        if s.ai_threat_capability > 0.3:
            recs.append("Deploy AI-powered threat detection and response systems")
            recs.append("Implement adversarial AI testing for existing ML security models")
        if s.ransomware_frequency > 0.3:
            recs.append("Deploy ransomware-specific defenses: immutable backups, air-gapped recovery")
        if s.nation_state_activity > 0.4:
            recs.append("Enhance threat intelligence sharing with ISACs and government agencies")
        if s.regulatory_compliance < 0.6:
            recs.append("Align security program with NIST CSF, ISO 27001, or SOC 2 framework")
        if not recs:
            recs.append("Maintain current security posture with continuous monitoring")
        return recs[:8]

    def simulate_attack_scenario(
        self,
        attack_vector: str = "ransomware",
        state: Optional[CyberState] = None,
        duration_days: int = 90,
    ) -> Dict[str, Any]:
        s = state or self.state or CyberState()
        vector_params = {
            "ransomware": {"base_prob": 0.15, "impact_scale": 0.8, "spread_rate": 0.4},
            "phishing": {"base_prob": 0.30, "impact_scale": 0.5, "spread_rate": 0.6},
            "zero_day": {"base_prob": 0.05, "impact_scale": 0.9, "spread_rate": 0.3},
            "supply_chain": {"base_prob": 0.08, "impact_scale": 0.7, "spread_rate": 0.2},
            "ddos": {"base_prob": 0.20, "impact_scale": 0.4, "spread_rate": 0.7},
            "insider_threat": {"base_prob": 0.10, "impact_scale": 0.7, "spread_rate": 0.3},
            "apt": {"base_prob": 0.07, "impact_scale": 0.85, "spread_rate": 0.15},
            "iot_exploit": {"base_prob": 0.12, "impact_scale": 0.5, "spread_rate": 0.5},
            "cloud_misconfiguration": {"base_prob": 0.18, "impact_scale": 0.6, "spread_rate": 0.4},
            "social_engineering": {"base_prob": 0.25, "impact_scale": 0.5, "spread_rate": 0.55},
        }
        params = vector_params.get(attack_vector, vector_params["ransomware"])
        probability = min(
            1.0,
            params["base_prob"]
            + s.threat_level * 0.2
            + params["spread_rate"] * (1.0 - s.patch_velocity) * 0.2
            + s.ai_threat_capability * 0.1,
        )
        containment_time = max(
            1,
            int(
                duration_days * 0.1 * (1.0 - s.incident_response_capacity)
                + duration_days * 0.05 * (1.0 - s.workforce_skill_level)
            ),
        )
        recovery_time = max(
            containment_time + 1,
            int(containment_time * (1.5 - s.regulatory_compliance * 0.5) + duration_days * 0.05 * (1.0 - s.patch_velocity)),
        )
        data_loss_pct = min(
            1.0,
            params["impact_scale"] * (1.0 - s.incident_response_capacity) * s.data_sensitivity,
        )
        financial_impact_m = round(
            params["impact_scale"]
            * data_loss_pct
            * (1.0 + s.cloud_dependency * 0.3)
            * random.uniform(0.5, 2.0)
            * 10,
            1,
        )
        reputation_impact = min(1.0, params["impact_scale"] * probability * 0.5)
        stages = []
        for day in range(0, min(duration_days, containment_time + 10), max(1, containment_time // 5)):
            progress = min(1.0, day / max(1, containment_time))
            damage = params["impact_scale"] * progress
            recovery = min(1.0, (day - containment_time) / max(1, recovery_time - containment_time)) if day > containment_time else 0.0
            stages.append({
                "day": day,
                "damage_level": float(min(1.0, damage)),
                "containment_progress": float(progress),
                "recovery_progress": float(max(0.0, recovery)),
            })
        return {
            "attack_vector": attack_vector,
            "probability": float(probability),
            "impact_scale": float(params["impact_scale"]),
            "expected_containment_days": containment_time,
            "expected_recovery_days": recovery_time,
            "data_loss_pct": float(data_loss_pct),
            "financial_impact_m": financial_impact_m,
            "reputation_impact": float(reputation_impact),
            "timeline": stages,
        }

    def predict(self, years_ahead: int = 5, state: Optional[CyberState] = None) -> Dict[str, Any]:
        s = state or self.state or CyberState()
        attack_surface_trend = min(1.0, s.cloud_dependency * 0.02 + s.iot_vulnerability * 0.015 + years_ahead * 0.03)
        ai_threat_growth = min(1.0, s.ai_threat_capability + years_ahead * 0.05)
        defense_improvement = min(1.0, s.incident_response_capacity + years_ahead * 0.03)
        quantum_threat = min(1.0, 0.05 + years_ahead * 0.02 + (1 if years_ahead > 8 else 0) * 0.15)
        supply_chain_threat = min(1.0, s.supply_chain_cyber_risk + years_ahead * 0.02)
        regulatory_pressure = min(1.0, s.regulatory_compliance + years_ahead * 0.04)
        future_risk = min(1.0, attack_surface_trend * 0.3 + ai_threat_growth * 0.3 + (1.0 - defense_improvement) * 0.2 + supply_chain_threat * 0.1 + quantum_threat * 0.1)
        return {
            "years_ahead": years_ahead,
            "future_risk_score": float(future_risk),
            "attack_surface_trend": float(attack_surface_trend),
            "ai_threat_growth": float(ai_threat_growth),
            "defense_improvement": float(defense_improvement),
            "quantum_threat_horizon": float(quantum_threat),
            "supply_chain_threat_growth": float(supply_chain_threat),
            "regulatory_pressure": float(regulatory_pressure),
        }