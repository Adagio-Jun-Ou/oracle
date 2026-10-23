from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import math
import random


@dataclass
class PandemicState:
    healthcare_capacity: float = 0.5
    surveillance_systems: float = 0.4
    vaccine_readiness: float = 0.3
    supply_chain_resilience: float = 0.4
    international_cooperation: float = 0.5
    public_trust: float = 0.45
    pandemic_preparedness_plan: float = 0.3
    laboratory_capacity: float = 0.4
    risk_communication: float = 0.35
    essential_worker_protection: float = 0.3
    genomic_surveillance: float = 0.25
    ppe_stockpile: float = 0.35
    antiviral_stockpile: float = 0.2
    contact_tracing_capacity: float = 0.3
    quarantine_capacity: float = 0.35


class PandemicPreparednessModel:
    def __init__(self):
        self.state: Optional[PandemicState] = None
        self.pathogen_types = [
            "respiratory_virus", "influenza", "coronavirus",
            "hemorrhagic_fever", "antibiotic_resistant_bacteria",
            "novel_pathogen", "vector_borne", "zoonotic_spillover",
        ]

    def set_state(self, **kwargs) -> "PandemicPreparednessModel":
        self.state = PandemicState(**kwargs)
        return self

    def assess_preparedness(self, state: Optional[PandemicState] = None) -> Dict[str, Any]:
        s = state or self.state or PandemicState()
        overall_preparedness = (
            s.healthcare_capacity * 0.15
            + s.surveillance_systems * 0.12
            + s.vaccine_readiness * 0.12
            + s.supply_chain_resilience * 0.08
            + s.international_cooperation * 0.08
            + s.public_trust * 0.08
            + s.pandemic_preparedness_plan * 0.10
            + s.laboratory_capacity * 0.07
            + s.risk_communication * 0.05
            + s.essential_worker_protection * 0.05
            + s.genomic_surveillance * 0.05
            + s.ppe_stockpile * 0.03
            + s.antiviral_stockpile * 0.02
            + s.contact_tracing_capacity * 0.03
            + s.quarantine_capacity * 0.02
        )
        overall_preparedness = max(0.0, min(1.0, overall_preparedness))
        if overall_preparedness > 0.65:
            readiness_level = "high"
        elif overall_preparedness > 0.45:
            readiness_level = "moderate"
        elif overall_preparedness > 0.25:
            readiness_level = "low"
        else:
            readiness_level = "critical"
        gaps = self._identify_gaps(s, overall_preparedness)
        response_capacity = self._assess_response_capacity(s)
        vulnerability_assessment = self._assess_vulnerability(s)
        recommendations = self._generate_recommendations(s, overall_preparedness, gaps)
        return {
            "overall_preparedness": float(overall_preparedness),
            "readiness_level": readiness_level,
            "gaps": gaps,
            "n_gaps": len(gaps),
            "response_capacity": response_capacity,
            "vulnerability": vulnerability_assessment,
            "detection_capacity": float(s.surveillance_systems * 0.4 + s.genomic_surveillance * 0.35 + s.laboratory_capacity * 0.25),
            "treatment_capacity": float(s.healthcare_capacity * 0.5 + s.antiviral_stockpile * 0.2 + s.quarantine_capacity * 0.3),
            "prevention_capacity": float(s.vaccine_readiness * 0.35 + s.pandemic_preparedness_plan * 0.30 + s.international_cooperation * 0.20 + s.contact_tracing_capacity * 0.15),
            "recommendations": recommendations,
        }

    def simulate_outbreak(
        self,
        pathogen_type: str = "respiratory_virus",
        state: Optional[PandemicState] = None,
        r0: float = 2.5,
        days: int = 365,
    ) -> Dict[str, Any]:
        s = state or self.state or PandemicState()
        pathogen_params = {
            "respiratory_virus": {"r0_range": (2.0, 4.0), "cfr_range": (0.005, 0.03), "spread_speed": 0.7},
            "influenza": {"r0_range": (1.3, 2.0), "cfr_range": (0.001, 0.02), "spread_speed": 0.6},
            "coronavirus": {"r0_range": (2.5, 6.0), "cfr_range": (0.005, 0.05), "spread_speed": 0.65},
            "hemorrhagic_fever": {"r0_range": (1.5, 2.5), "cfr_range": (0.3, 0.9), "spread_speed": 0.3},
            "antibiotic_resistant_bacteria": {"r0_range": (1.1, 1.5), "cfr_range": (0.1, 0.3), "spread_speed": 0.4},
            "novel_pathogen": {"r0_range": (2.0, 5.0), "cfr_range": (0.01, 0.1), "spread_speed": 0.8},
            "vector_borne": {"r0_range": (1.0, 3.0), "cfr_range": (0.01, 0.1), "spread_speed": 0.4},
            "zoonotic_spillover": {"r0_range": (1.5, 4.0), "cfr_range": (0.02, 0.2), "spread_speed": 0.5},
        }
        params = pathogen_params.get(pathogen_type, pathogen_params["respiratory_virus"])
        effective_r0 = max(0.5, r0 * (1 + params["spread_speed"] * 0.3) * (1 - s.surveillance_systems * 0.2) * (1 - s.contact_tracing_capacity * 0.15))
        detection_delay = max(1, int(30 * (1 - s.surveillance_systems * 0.6) * (1 - s.genomic_surveillance * 0.3)))
        response_delay = max(1, int(14 * (1 - s.pandemic_preparedness_plan * 0.3) * (1 - s.risk_communication * 0.2)))
        ifr = params["cfr_range"][0] + (params["cfr_range"][1] - params["cfr_range"][0]) * (1 - s.healthcare_capacity * 0.5)
        ifr *= (1 - s.essential_worker_protection * 0.1)
        ifr = min(1.0, max(0.0001, ifr))
        peak_day = max(10, int(days * 0.25 * (1 - s.international_cooperation * 0.2)))
        total_cases_pct = min(0.8, (1 - 1 / effective_r0) * (1 - s.vaccine_readiness * 0.3) * (1 - s.quarantine_capacity * 0.1))
        hospital_occupancy_peak = min(1.5, total_cases_pct * ifr * 100 / max(0.01, s.healthcare_capacity * 0.3))
        economic_impact_gdp = round(min(25, total_cases_pct * 10 * (1 - s.supply_chain_resilience * 0.3) * random.uniform(0.8, 1.2)), 1)
        timeline = []
        for day in range(0, days + 1, max(1, days // 10)):
            if day < detection_delay:
                cumulative_pct = total_cases_pct * 0.01 * (day / max(1, detection_delay))
            else:
                day_scaled = (day - detection_delay) / max(1, days - detection_delay)
                cumulative_pct = total_cases_pct / (1 + math.exp(-10 * (day_scaled - 0.4)))
            daily_new_pct = total_cases_pct * 0.02 * effective_r0 * day_scaled * math.exp(-5 * day_scaled) if day > detection_delay else 0
            daily_new_pct = max(0, daily_new_pct)
            hospitalized_pct = cumulative_pct * ifr * 0.2
            timeline.append({
                "day": day,
                "cumulative_pct": float(round(min(1.0, cumulative_pct) * 100, 2)),
                "daily_new_pct": float(round(min(0.05, daily_new_pct) * 100, 3)),
                "hospitalized_pct": float(round(min(1.0, hospitalized_pct) * 100, 3)),
            })
        return {
            "pathogen_type": pathogen_type,
            "effective_r0": float(round(effective_r0, 2)),
            "detection_delay_days": detection_delay,
            "response_delay_days": response_delay,
            "infection_fatality_rate": float(round(ifr, 4)),
            "peak_hospital_occupancy": float(round(min(1.0, hospital_occupancy_peak), 3)),
            "total_populations_infected_pct": float(round(total_cases_pct * 100, 1)),
            "economic_impact_gdp_pct": economic_impact_gdp,
            "timeline": timeline,
            "healthcare_overwhelm": bool(hospital_occupancy_peak > s.healthcare_capacity * 2),
            "vaccine_deployment_months": int(max(3, 18 * (1 - s.vaccine_readiness * 0.5))),
            "supply_chain_disruption_severity": float(round(min(1.0, (1 - s.supply_chain_resilience) * 0.7 + total_cases_pct * 0.3), 3)),
        }

    def predict(self, years_ahead: int = 5, state: Optional[PandemicState] = None) -> Dict[str, Any]:
        s = state or self.state or PandemicState()
        current_preparedness = self.assess_preparedness(s)["overall_preparedness"]
        surveillance_trajectory = min(1.0, s.surveillance_systems + years_ahead * 0.02)
        vaccine_trajectory = min(1.0, s.vaccine_readiness + years_ahead * 0.03)
        genomic_trajectory = min(1.0, s.genomic_surveillance + years_ahead * 0.025)
        cooperation_trajectory = min(1.0, s.international_cooperation + years_ahead * 0.01)
        trust_trajectory = max(0.1, min(1.0, s.public_trust + years_ahead * 0.005 * (1 if s.public_trust > 0.4 else -0.5)))
        future_preparedness = min(1.0, (
            s.healthcare_capacity * 0.12
            + surveillance_trajectory * 0.12
            + vaccine_trajectory * 0.12
            + min(1.0, s.supply_chain_resilience + years_ahead * 0.01) * 0.08
            + cooperation_trajectory * 0.08
            + trust_trajectory * 0.08
            + min(1.0, s.pandemic_preparedness_plan + years_ahead * 0.02) * 0.10
            + min(1.0, s.laboratory_capacity + years_ahead * 0.01) * 0.07
            + min(1.0, s.risk_communication + years_ahead * 0.01) * 0.05
            + min(1.0, s.essential_worker_protection + years_ahead * 0.01) * 0.05
            + genomic_trajectory * 0.05
            + min(1.0, s.ppe_stockpile + years_ahead * 0.005) * 0.03
            + min(1.0, s.antiviral_stockpile + years_ahead * 0.005) * 0.02
            + min(1.0, s.contact_tracing_capacity + years_ahead * 0.01) * 0.03
            + min(1.0, s.quarantine_capacity + years_ahead * 0.005) * 0.02
        ))
        pandemic_probability = min(1.0, 0.1 + years_ahead * 0.02 * (1 + (1 - s.surveillance_systems) * 0.3))
        emergence_risk = self._assess_emergence_risk(s, years_ahead)
        return {
            "years_ahead": years_ahead,
            "future_preparedness": float(future_preparedness),
            "preparation_improvement": float(round(future_preparedness - current_preparedness, 4)),
            "surveillance_trajectory": float(surveillance_trajectory),
            "vaccine_readiness_trajectory": float(vaccine_trajectory),
            "genomic_surveillance_trajectory": float(genomic_trajectory),
            "cooperation_trajectory": float(cooperation_trajectory),
            "trust_trajectory": float(trust_trajectory),
            "pandemic_probability": float(pandemic_probability),
            "emergence_risk": emergence_risk,
            "resilience_score": float(min(1.0, future_preparedness * 0.6 + s.international_cooperation * 0.2 + s.supply_chain_resilience * 0.2)),
        }

    def _identify_gaps(self, s: PandemicState, preparedness: float) -> List[Dict[str, Any]]:
        gaps = []
        thresholds = {
            "healthcare_capacity": 0.5, "surveillance_systems": 0.5, "vaccine_readiness": 0.4,
            "supply_chain_resilience": 0.5, "public_trust": 0.5, "pandemic_preparedness_plan": 0.4,
            "laboratory_capacity": 0.5, "risk_communication": 0.5, "genomic_surveillance": 0.4,
            "ppe_stockpile": 0.4, "contact_tracing": 0.4, "quarantine_capacity": 0.4,
        }
        labels = {
            "healthcare_capacity": ("Healthcare surge capacity", "critical"),
            "surveillance_systems": ("Disease surveillance networks", "critical"),
            "vaccine_readiness": ("Vaccine development and deployment", "high"),
            "supply_chain_resilience": ("Medical supply chain resilience", "high"),
            "public_trust": ("Public trust and compliance", "moderate"),
            "pandemic_preparedness_plan": ("Pandemic preparedness planning", "critical"),
            "laboratory_capacity": ("Diagnostic laboratory capacity", "high"),
            "risk_communication": ("Risk communication systems", "moderate"),
            "genomic_surveillance": ("Genomic sequencing surveillance", "high"),
            "ppe_stockpile": ("PPE stockpile adequacy", "moderate"),
            "contact_tracing": ("Contact tracing infrastructure", "moderate"),
            "quarantine_capacity": ("Quarantine and isolation capacity", "moderate"),
        }
        for attr, threshold in thresholds.items():
            val = getattr(s, attr, 0.5)
            if val < threshold:
                label, severity = labels.get(attr, (attr, "moderate"))
                actual_severity = severity if val < threshold * 0.5 else "moderate"
                gap_pct = (threshold - val) / threshold
                gaps.append({
                    "area": label,
                    "severity": actual_severity,
                    "gap_pct": float(round(gap_pct * 100, 1)),
                    "current_level": float(val),
                    "required_level": float(threshold),
                })
        gaps.sort(key=lambda x: 0 if x["severity"] == "critical" else 1 if x["severity"] == "high" else 2)
        return gaps[:8]

    def _assess_response_capacity(self, s: PandemicState) -> Dict[str, Any]:
        detection_speed = 1.0 - s.surveillance_systems * 0.5 - s.genomic_surveillance * 0.3 - s.laboratory_capacity * 0.2
        treatment_capacity = s.healthcare_capacity * 0.4 + s.antiviral_stockpile * 0.2 + s.quarantine_capacity * 0.2 + s.essential_worker_protection * 0.2
        vaccination_speed = s.vaccine_readiness * 0.5 + s.supply_chain_resilience * 0.2 + s.international_cooperation * 0.2 + s.public_trust * 0.1
        communication_effectiveness = s.risk_communication * 0.4 + s.public_trust * 0.3 + s.international_cooperation * 0.3
        return {
            "detection_speed_days": int(max(1, 30 * detection_speed)),
            "treatment_capacity": float(treatment_capacity),
            "vaccination_speed_months": int(max(3, 18 * (1 - vaccination_speed * 0.5))),
            "communication_effectiveness": float(communication_effectiveness),
            "overall_response_score": float((1 - detection_speed) * 0.3 + treatment_capacity * 0.3 + vaccination_speed * 0.2 + communication_effectiveness * 0.2),
        }

    def _assess_vulnerability(self, s: PandemicState) -> Dict[str, Any]:
        systemic_vulnerability = 1.0 - (
            s.healthcare_capacity * 0.25
            + s.supply_chain_resilience * 0.20
            + s.pandemic_preparedness_plan * 0.20
            + s.international_cooperation * 0.15
            + s.public_trust * 0.10
            + s.essential_worker_protection * 0.10
        )
        systemic_vulnerability = max(0.0, min(1.0, systemic_vulnerability))
        return {
            "systemic_vulnerability": float(systemic_vulnerability),
            "high_risk_populations_pct": float(round(min(80, 25 + (1 - s.healthcare_capacity) * 20 + (1 - s.essential_worker_protection) * 15), 1)),
            "supply_chain_disruption_risk": float(round(1 - s.supply_chain_resilience, 3)),
            "misinformation_vulnerability": float(round(1 - s.risk_communication * 0.5 - s.public_trust * 0.5, 3)),
            "international_coordination_gap": float(round(1 - s.international_cooperation, 3)),
        }

    def _assess_emergence_risk(self, s: PandemicState, years_ahead: int) -> Dict[str, float]:
        base_risk = 0.1 + years_ahead * 0.015
        return {
            "respiratory_virus": float(min(1.0, base_risk * 1.5 * (1 + years_ahead * 0.01))),
            "influenza_pandemic": float(min(1.0, base_risk * 0.8 * (1 + years_ahead * 0.005))),
            "novel_coronavirus": float(min(1.0, base_risk * 0.6 * (1 + years_ahead * 0.008))),
            "hemorrhagic_fever": float(min(1.0, base_risk * 0.3 * (1 + years_ahead * 0.003))),
            "antibiotic_resistant": float(min(1.0, base_risk * 1.2 * (1 + years_ahead * 0.015))),
            "zoonotic_spillover": float(min(1.0, base_risk * 1.0 * (1 + years_ahead * 0.01 * (1 + (1 - s.surveillance_systems) * 0.3)))),
            "vector_borne_expansion": float(min(1.0, base_risk * 0.5 * (1 + years_ahead * 0.007))),
        }

    def _generate_recommendations(self, s: PandemicState, preparedness: float, gaps: List[Dict[str, Any]]) -> List[str]:
        recs = []
        if s.surveillance_systems < 0.5:
            recs.append("Strengthen disease surveillance networks with real-time data sharing and AI-powered early warning systems")
        if s.genomic_surveillance < 0.4:
            recs.append("Expand genomic sequencing capacity for pathogen detection and variant tracking")
        if s.vaccine_readiness < 0.5:
            recs.append("Invest in mRNA platform technology and pre-positioned manufacturing capacity for rapid vaccine deployment")
        if s.healthcare_capacity < 0.5:
            recs.append("Build healthcare surge capacity with stockpiled equipment, trained reserve staff, and facility conversion plans")
        if s.public_trust < 0.5:
            recs.append("Rebuild public trust through transparent risk communication, community engagement, and consistent policy messaging")
        if s.international_cooperation < 0.5:
            recs.append("Strengthen international health cooperation through WHO reform, pathogen access agreements, and joint preparedness exercises")
        if s.supply_chain_resilience < 0.5:
            recs.append("Diversify medical supply chains and establish strategic stockpiles of essential medicines and equipment")
        if s.pandemic_preparedness_plan < 0.4:
            recs.append("Develop and regularly exercise comprehensive pandemic preparedness plans at national and subnational levels")
        if s.ppe_stockpile < 0.5:
            recs.append("Maintain strategic national stockpiles of PPE with 90-day supply minimum and rotation protocols")
        if s.essential_worker_protection < 0.4:
            recs.append("Establish priority protection protocols for essential workers including healthcare, food supply, and utilities")
        if s.risk_communication < 0.4:
            recs.append("Develop pre-tested risk communication frameworks with message templates for different outbreak scenarios")
        if not recs:
            recs.append("Maintain current preparedness level and invest in emerging pathogen detection technologies")
        return recs[:8]