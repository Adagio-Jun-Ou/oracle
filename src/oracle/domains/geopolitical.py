import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from oracle.core.types import Domain, Prediction, ConfidenceLevel


@dataclass
class GeopoliticalEvent:
    name: str
    event_type: str
    probability: float
    impact_severity: float
    affected_domains: list[Domain] = field(default_factory=list)
    duration_months: float = 12.0
    reversibility: float = 0.5
    key_indicators: dict = field(default_factory=dict)


@dataclass
class GeopoliticalState:
    us_china_tension: float = 0.4
    russia_west_tension: float = 0.6
    middle_east_instability: float = 0.5
    trade_protectionism: float = 0.35
    nuclear_proliferation_risk: float = 0.15
    cyber_warfare_likelihood: float = 0.3
    energy_geopolitics: float = 0.4
    democratic_backsliding: float = 0.3
    refugee_crisis_level: float = 0.35
    supply_chain_fragility: float = 0.4
    sanctions_intensity: float = 0.3
    alliance_stability: float = 0.6
    timestamp: date = field(default_factory=date.today)


EVENT_TEMPLATES: list[dict] = [
    {"name": "Taiwan Strait Crisis", "type": "military_conflict", "base_prob": 0.05,
     "severity": 0.9, "domains": [Domain.GEOPOLITICS, Domain.ECONOMICS, Domain.FINANCE],
     "indicators": {"us_china_tension": 0.7, "military_posturing": True}},
    {"name": "Russia-NATO Escalation", "type": "military_conflict", "base_prob": 0.08,
     "severity": 0.85, "domains": [Domain.GEOPOLITICS, Domain.ENERGY, Domain.ECONOMICS],
     "indicators": {"russia_west_tension": 0.7, "energy_geopolitics": 0.6}},
    {"name": "Middle East Regional War", "type": "military_conflict", "base_prob": 0.12,
     "severity": 0.7, "domains": [Domain.GEOPOLITICS, Domain.ENERGY, Domain.COMMODITIES],
     "indicators": {"middle_east_instability": 0.6}},
    {"name": "US-China Trade War Escalation", "type": "trade_war", "base_prob": 0.25,
     "severity": 0.6, "domains": [Domain.ECONOMICS, Domain.FINANCE, Domain.TECHNOLOGY],
     "indicators": {"us_china_tension": 0.5, "trade_protectionism": 0.5}},
    {"name": "Global Supply Chain Disruption", "type": "economic_shock", "base_prob": 0.20,
     "severity": 0.5, "domains": [Domain.ECONOMICS, Domain.COMMODITIES],
     "indicators": {"supply_chain_fragility": 0.5}},
    {"name": "Major Cyber Attack on Infrastructure", "type": "cyber", "base_prob": 0.15,
     "severity": 0.6, "domains": [Domain.TECHNOLOGY, Domain.GEOPOLITICS],
     "indicators": {"cyber_warfare_likelihood": 0.5}},
    {"name": "European Union Fragmentation", "type": "political_instability", "base_prob": 0.10,
     "severity": 0.5, "domains": [Domain.ECONOMICS, Domain.GEOPOLITICS],
     "indicators": {"democratic_backsliding": 0.5, "alliance_stability": 0.4}},
    {"name": "Oil Supply Shock", "type": "energy_crisis", "base_prob": 0.15,
     "severity": 0.65, "domains": [Domain.ENERGY, Domain.ECONOMICS, Domain.COMMODITIES],
     "indicators": {"middle_east_instability": 0.6, "energy_geopolitics": 0.6}},
    {"name": "Pandemic Recurrence", "type": "health_crisis", "base_prob": 0.12,
     "severity": 0.55, "domains": [Domain.HEALTH, Domain.ECONOMICS, Domain.SOCIAL],
     "indicators": {"supply_chain_fragility": 0.5}},
    {"name": "Nuclear Proliferation Event", "type": "nuclear", "base_prob": 0.03,
     "severity": 0.95, "domains": [Domain.GEOPOLITICS, Domain.HEALTH],
     "indicators": {"nuclear_proliferation_risk": 0.4}},
    {"name": "Climate Treaty Collapse", "type": "environmental", "base_prob": 0.18,
     "severity": 0.45, "domains": [Domain.CLIMATE, Domain.ENERGY, Domain.ECONOMICS],
     "indicators": {"trade_protectionism": 0.4, "energy_geopolitics": 0.5}},
    {"name": "Democratic Backlash / Populist Wave", "type": "political_instability", "base_prob": 0.22,
     "severity": 0.4, "domains": [Domain.GEOPOLITICS, Domain.SOCIAL],
     "indicators": {"democratic_backsliding": 0.5, "alliance_stability": 0.5}},
    {"name": "Sanctions Cascade (Russia/Iran/China)", "type": "sanctions", "base_prob": 0.20,
     "severity": 0.5, "domains": [Domain.ECONOMICS, Domain.GEOPOLITICS],
     "indicators": {"sanctions_intensity": 0.5, "us_china_tension": 0.4}},
    {"name": "Refugee Crisis Amplification", "type": "humanitarian", "base_prob": 0.16,
     "severity": 0.4, "domains": [Domain.SOCIAL, Domain.GEOPOLITICS],
     "indicators": {"refugee_crisis_level": 0.5, "middle_east_instability": 0.5}},
    {"name": "Technology Decoupling", "type": "trade_war", "base_prob": 0.30,
     "severity": 0.55, "domains": [Domain.TECHNOLOGY, Domain.ECONOMICS],
     "indicators": {"us_china_tension": 0.5, "trade_protectionism": 0.5}},
]

INTERACTION_MATRIX = {
    ("us_china_tension", "russia_west_tension"): 0.15,
    ("us_china_tension", "trade_protectionism"): 0.35,
    ("russia_west_tension", "energy_geopolitics"): 0.30,
    ("middle_east_instability", "energy_geopolitics"): 0.40,
    ("middle_east_instability", "refugee_crisis_level"): 0.30,
    ("trade_protectionism", "supply_chain_fragility"): 0.25,
    ("cyber_warfare_likelihood", "us_china_tension"): 0.20,
    ("democratic_backsliding", "alliance_stability"): -0.25,
    ("sanctions_intensity", "russia_west_tension"): 0.20,
    ("nuclear_proliferation_risk", "russia_west_tension"): 0.15,
    ("supply_chain_fragility", "trade_protectionism"): 0.20,
    ("refugee_crisis_level", "democratic_backsliding"): 0.15,
}


class GeopoliticalRiskModel:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.event_templates = EVENT_TEMPLATES
        self.interactions = INTERACTION_MATRIX

    def assess_risk(self, state: GeopoliticalState) -> dict:
        overall = np.mean([
            state.us_china_tension, state.russia_west_tension,
            state.middle_east_instability, state.trade_protectionism,
            state.cyber_warfare_likelihood, state.supply_chain_fragility,
            state.sanctions_intensity,
        ])
        stability_score = np.mean([state.alliance_stability, 1 - state.democratic_backsliding, 1 - state.supply_chain_fragility])
        conflict_risk = np.mean([state.us_china_tension, state.russia_west_tension, state.middle_east_instability])
        economic_impact = self._economic_impact(state)
        triggered = self._propagate_interactions(state)
        triggered.update({
            "us_china_tension": state.us_china_tension,
            "russia_west_tension": state.russia_west_tension,
            "middle_east_instability": state.middle_east_instability,
            "trade_protectionism": state.trade_protectionism,
        })
        return {
            "overall_risk": float(overall),
            "stability_score": float(stability_score),
            "conflict_risk": float(conflict_risk),
            "economic_impact_score": float(economic_impact),
            "adjusted_tensions": triggered,
        }

    def simulate_events(
        self, state: GeopoliticalState, months: int = 12, n_simulations: int = 1000,
    ) -> dict:
        events_triggered = {t["name"]: np.zeros(n_simulations) for t in self.event_templates}
        total_impact = np.zeros(n_simulations)
        escalated = state

        for _ in range(n_simulations):
            s = GeopoliticalState(
                us_china_tension=state.us_china_tension,
                russia_west_tension=state.russia_west_tension,
                middle_east_instability=state.middle_east_instability,
                trade_protectionism=state.trade_protectionism,
                nuclear_proliferation_risk=state.nuclear_proliferation_risk,
                cyber_warfare_likelihood=state.cyber_warfare_likelihood,
                energy_geopolitics=state.energy_geopolitics,
                democratic_backsliding=state.democratic_backsliding,
                refugee_crisis_level=state.refugee_crisis_level,
                supply_chain_fragility=state.supply_chain_fragility,
                sanctions_intensity=state.sanctions_intensity,
                alliance_stability=state.alliance_stability,
            )
            sim_impact = 0.0
            for month in range(months):
                for template in self.event_templates:
                    adjusted_prob = self._adjusted_probability(template, s)
                    if self.rng.random() < adjusted_prob:
                        events_triggered[template["name"]][_] += 1
                        sim_impact += template["severity"] * adjusted_prob
                        s = self._apply_event_shock(s, template)
            total_impact[_] = sim_impact

        result = {}
        for name, counts in events_triggered.items():
            result[name] = {
                "probability": float(np.mean(counts > 0)),
                "expected_occurrences": float(np.mean(counts)),
                "max_occurrences": int(np.max(counts)),
            }
        result["aggregate_impact"] = {
            "mean": float(np.mean(total_impact)),
            "p95": float(np.percentile(total_impact, 95)),
            "p99": float(np.percentile(total_impact, 99)),
            "worst_case": float(np.max(total_impact)),
        }
        return result

    def predict(
        self, state: GeopoliticalState, target_date: date,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    ) -> list[Prediction]:
        risk = self.assess_risk(state)
        years_ahead = max(0.25, (target_date - state.timestamp).days / 365.25)
        decay = np.exp(-0.1 * years_ahead)

        z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645,
             ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence]
        base_std = 0.15

        predictions = []
        metrics = {
            "geopolitical_risk_index": (risk["overall_risk"], Domain.GEOPOLITICS),
            "conflict_probability": (risk["conflict_risk"], Domain.GEOPOLITICS),
            "stability_score": (risk["stability_score"], Domain.GEOPOLITICS),
            "economic_impact": (risk["economic_impact_score"], Domain.ECONOMICS),
            "supply_chain_risk": (state.supply_chain_fragility, Domain.ECONOMICS),
            "trade_protectionism": (state.trade_protectionism, Domain.ECONOMICS),
            "energy_geopolitics_risk": (state.energy_geopolitics, Domain.ENERGY),
        }

        for name, (value, domain) in metrics.items():
            adj_value = value * (1 + (1 - decay) * 0.1 * self.rng.normal(0, 1))
            adj_std = base_std * (1 + 0.3 * (years_ahead - 0.25))
            ci = (max(0, adj_value - z * adj_std), min(1, adj_value + z * adj_std))
            predictions.append(Prediction(
                domain=domain, variable=f"geopolitical_{name}", value=round(adj_value, 4),
                confidence_interval=(round(ci[0], 4), round(ci[1], 4)),
                confidence_level=confidence, timestamp=datetime.now(),
                target_date=target_date, model_name="geopolitical_risk",
                weights={"simulation": 0.6, "expert": 0.4},
                signals_used=list(risk["adjusted_tensions"].keys())[:5],
            ))

        return predictions

    def scenario_analysis(self, state: GeopoliticalState, scenarios: dict[str, GeopoliticalState]) -> dict:
        results = {}
        baseline = self.assess_risk(state)
        results["baseline"] = baseline
        for name, scenario in scenarios.items():
            results[name] = self.assess_risk(scenario)
        return results

    def _adjusted_probability(self, template: dict, state: GeopoliticalState) -> float:
        base_prob = template["base_prob"]
        severity_multiplier = 1.0
        indicators = template.get("indicators", {})
        for indicator, threshold in indicators.items():
            tension = getattr(state, indicator, 0.3)
            if isinstance(threshold, bool):
                if threshold:
                    severity_multiplier *= (1 + 0.3 * tension)
            else:
                if tension > threshold:
                    severity_multiplier *= (1 + 0.2 * (tension - threshold))
        overall_tension = np.mean([
            state.us_china_tension, state.russia_west_tension,
            state.middle_east_instability, state.trade_protectionism,
        ])
        prob = base_prob * severity_multiplier * (1 + overall_tension)
        return float(np.clip(prob, 0, 0.95))

    def _apply_event_shock(self, state: GeopoliticalState, template: dict) -> GeopoliticalState:
        from dataclasses import replace as dc_replace
        s = dc_replace(state)
        shock = template["severity"] * 0.05
        event_type = template["type"]
        if event_type == "military_conflict":
            s = dc_replace(s, us_china_tension=min(1, s.us_china_tension + shock * 0.5),
                           russia_west_tension=min(1, s.russia_west_tension + shock * 0.3),
                           energy_geopolitics=min(1, s.energy_geopolitics + shock * 0.4))
        elif event_type == "trade_war":
            s = dc_replace(s, trade_protectionism=min(1, s.trade_protectionism + shock * 0.6),
                           supply_chain_fragility=min(1, s.supply_chain_fragility + shock * 0.4))
        elif event_type == "cyber":
            s = dc_replace(s, cyber_warfare_likelihood=min(1, s.cyber_warfare_likelihood + shock * 0.5))
        elif event_type == "energy_crisis":
            s = dc_replace(s, energy_geopolitics=min(1, s.energy_geopolitics + shock * 0.7))
        elif event_type == "sanctions":
            s = dc_replace(s, sanctions_intensity=min(1, s.sanctions_intensity + shock * 0.6),
                           trade_protectionism=min(1, s.trade_protectionism + shock * 0.3))
        elif event_type == "health_crisis":
            s = dc_replace(s, supply_chain_fragility=min(1, s.supply_chain_fragility + shock * 0.5),
                           democratic_backsliding=min(1, s.democratic_backsliding + shock * 0.2))
        elif event_type == "political_instability":
            s = dc_replace(s, democratic_backsliding=min(1, s.democratic_backsliding + shock * 0.5),
                           alliance_stability=max(0, s.alliance_stability - shock * 0.4))
        elif event_type == "environmental":
            s = dc_replace(s, energy_geopolitics=min(1, s.energy_geopolitics + shock * 0.3),
                           supply_chain_fragility=min(1, s.supply_chain_fragility + shock * 0.4),
                           refugee_crisis_level=min(1, s.refugee_crisis_level + shock * 0.5))
        elif event_type == "humanitarian":
            s = dc_replace(s, refugee_crisis_level=min(1, s.refugee_crisis_level + shock * 0.6),
                           alliance_stability=max(0, s.alliance_stability - shock * 0.3),
                           democratic_backsliding=min(1, s.democratic_backsliding + shock * 0.2))
        s = dc_replace(s, alliance_stability=max(0, min(1, s.alliance_stability)))
        return s

    def _propagate_interactions(self, state: GeopoliticalState) -> dict:
        state_dict = {
            "us_china_tension": state.us_china_tension,
            "russia_west_tension": state.russia_west_tension,
            "middle_east_instability": state.middle_east_instability,
            "trade_protectionism": state.trade_protectionism,
            "nuclear_proliferation_risk": state.nuclear_proliferation_risk,
            "cyber_warfare_likelihood": state.cyber_warfare_likelihood,
            "energy_geopolitics": state.energy_geopolitics,
            "democratic_backsliding": state.democratic_backsliding,
            "refugee_crisis_level": state.refugee_crisis_level,
            "supply_chain_fragility": state.supply_chain_fragility,
            "sanctions_intensity": state.sanctions_intensity,
            "alliance_stability": state.alliance_stability,
        }
        adjusted = dict(state_dict)
        for (var_a, var_b), strength in self.interactions.items():
            if var_a in adjusted and var_b in adjusted:
                adjustment = strength * adjusted[var_a] * 0.1
                adjusted[var_b] = float(np.clip(adjusted[var_b] + adjustment, 0, 1))
        return adjusted

    def _economic_impact(self, state: GeopoliticalState) -> float:
        components = {
            "conflict": 0.3 * np.mean([state.us_china_tension, state.russia_west_tension, state.middle_east_instability]),
            "trade": 0.25 * state.trade_protectionism,
            "energy": 0.2 * state.energy_geopolitics,
            "supply": 0.15 * state.supply_chain_fragility,
            "cyber": 0.1 * state.cyber_warfare_likelihood,
        }
        return float(np.clip(sum(components.values()), 0, 1))

    def conflict_intensity_score(
        self,
        state: GeopoliticalState,
        escalation_factors: Optional[dict] = None,
    ) -> dict:
        hot_conflicts = np.mean([state.us_china_tension, state.russia_west_tension, state.middle_east_instability])
        warm_conflicts = np.mean([state.cyber_warfare_likelihood, state.sanctions_intensity, state.trade_protectionism])
        cold_tensions = np.mean([state.nuclear_proliferation_risk, state.energy_geopolitics, state.democratic_backsliding])
        if escalation_factors:
            for key, val in escalation_factors.items():
                if key == "military_posturing":
                    hot_conflicts = min(1, hot_conflicts + val * 0.3)
                elif key == "diplomatic_failure":
                    warm_conflicts = min(1, warm_conflicts + val * 0.2)
                elif key == "arms_race":
                    cold_tensions = min(1, cold_tensions + val * 0.25)
        overall = 0.5 * hot_conflicts + 0.3 * warm_conflicts + 0.2 * cold_tensions
        if overall > 0.7:
            stage = "hot_war"
        elif overall > 0.5:
            stage = "warm_conflict"
        elif overall > 0.3:
            stage = "cold_war"
        elif overall > 0.15:
            stage = "latent_tension"
        else:
            stage = "peaceful"
        de_escalation_potential = (
            state.alliance_stability * 0.3
            + (1 - state.democratic_backsliding) * 0.2
            + (1 - state.trade_protectionism) * 0.2
            + (1 - state.nuclear_proliferation_risk) * 0.15
            + (1 - state.refugee_crisis_level) * 0.15
        )
        return {
            "overall_intensity": round(float(overall), 4),
            "hot_conflict_score": round(float(hot_conflicts), 4),
            "warm_conflict_score": round(float(warm_conflicts), 4),
            "cold_tension_score": round(float(cold_tensions), 4),
            "stage": stage,
            "de_escalation_potential": round(float(de_escalation_potential), 4),
            "nuclear_risk_level": "critical" if state.nuclear_proliferation_risk > 0.5 else "elevated" if state.nuclear_proliferation_risk > 0.25 else "moderate" if state.nuclear_proliferation_risk > 0.1 else "low",
            "major_flashpoints": [
                {"region": "US-China", "intensity": round(state.us_china_tension, 3)},
                {"region": "Russia-West", "intensity": round(state.russia_west_tension, 3)},
                {"region": "Middle East", "intensity": round(state.middle_east_instability, 3)},
            ],
        }

    def alliance_network(
        self,
        state: GeopoliticalState,
        alliances: Optional[list] = None,
    ) -> dict:
        if alliances is None:
            alliances = [
                {"name": "NATO", "members": ["US", "UK", "Germany", "France", "Canada"], "cohesion": 0.7, "deterrence": 0.8},
                {"name": "QUAD", "members": ["US", "Japan", "India", "Australia"], "cohesion": 0.5, "deterrence": 0.6},
                {"name": "AUKUS", "members": ["US", "UK", "Australia"], "cohesion": 0.75, "deterrence": 0.7},
                {"name": "BRICS", "members": ["China", "Russia", "India", "Brazil", "South Africa"], "cohesion": 0.35, "deterrence": 0.4},
                {"name": "Shanghai Cooperation", "members": ["China", "Russia", "Central Asian states"], "cohesion": 0.4, "deterrence": 0.45},
            ]
        results = []
        for alliance in alliances:
            cohesion = alliance["cohesion"]
            deterrence = alliance["deterrence"]
            effectiveness = cohesion * (1 - state.democratic_backsliding * 0.3) * (1 - state.refugee_crisis_level * 0.1)
            deterrence_effective = deterrence * effectiveness * (1 - state.nuclear_proliferation_risk * 0.2)
            results.append({
                "name": alliance["name"],
                "members": alliance["members"],
                "base_cohesion": round(cohesion, 3),
                "base_deterrence": round(deterrence, 3),
                "adjusted_cohesion": round(float(min(1, effectiveness)), 3),
                "adjusted_deterrence": round(float(min(1, deterrence_effective)), 3),
                "stability_contribution": round(float(min(1, (effectiveness + deterrence_effective) / 2)), 3),
            })
        global_cohesion = np.mean([r["adjusted_cohesion"] for r in results])
        global_deterrence = np.mean([r["adjusted_deterrence"] for r in results])
        fragmentation = 1 - global_cohesion
        return {
            "alliances": results,
            "global_cohesion": round(float(global_cohesion), 4),
            "global_deterrence": round(float(global_deterrence), 4),
            "fragmentation_risk": round(float(fragmentation), 4),
            "stability_index": round(float((global_cohesion + global_deterrence) / 2), 4),
            "key_vulnerabilities": [
                v for v in [
                    "democratic_backsliding" if state.democratic_backsliding > 0.4 else None,
                    "nuclear_proliferation" if state.nuclear_proliferation_risk > 0.2 else None,
                    "refugee_crisis" if state.refugee_crisis_level > 0.4 else None,
                    "alliance_fracture" if state.alliance_stability < 0.4 else None,
                ] if v is not None
            ],
        }

    def resource_competition(
        self,
        state: GeopoliticalState,
        resources: Optional[list] = None,
    ) -> dict:
        if resources is None:
            resources = [
                {"name": "Rare Earth Minerals", "criticality": 0.9, "concentration": 0.85, "substitutability": 0.2, "strategic_importance": 0.95},
                {"name": "Semiconductors", "criticality": 0.95, "concentration": 0.75, "substitutability": 0.15, "strategic_importance": 0.98},
                {"name": "Oil & Gas", "criticality": 0.7, "concentration": 0.6, "substitutability": 0.3, "strategic_importance": 0.8},
                {"name": "Lithium", "criticality": 0.75, "concentration": 0.7, "substitutability": 0.25, "strategic_importance": 0.85},
                {"name": "Fresh Water", "criticality": 0.8, "concentration": 0.4, "substitutability": 0.1, "strategic_importance": 0.7},
                {"name": "Food/Grain", "criticality": 0.85, "concentration": 0.5, "substitutability": 0.2, "strategic_importance": 0.75},
            ]
        tension_multiplier = 1 + (state.us_china_tension + state.russia_west_tension + state.trade_protectionism) / 3 * 0.5
        results = []
        for res in resources:
            supply_risk = res["concentration"] * (1 - res["substitutability"]) * res["criticality"]
            competition_score = res["criticality"] * res["strategic_importance"] * res["concentration"] * tension_multiplier
            conflict_potential = min(1, res["concentration"] * res["criticality"] * (1 - res["substitutability"]) * (state.energy_geopolitics * 0.4 + state.us_china_tension * 0.3 + state.russia_west_tension * 0.3))
            disruption_probability = min(1, supply_risk * (1 + state.supply_chain_fragility * 0.5))
            results.append({
                "name": res["name"],
                "supply_risk": round(float(supply_risk), 4),
                "competition_score": round(float(min(1, competition_score)), 4),
                "conflict_potential": round(float(conflict_potential), 4),
                "disruption_probability": round(float(disruption_probability), 4),
                "strategic_importance": round(res["strategic_importance"], 3),
            })
        results.sort(key=lambda x: x["competition_score"], reverse=True)
        overall_risk = np.mean([r["competition_score"] for r in results])
        return {
            "resources": results,
            "overall_competition_risk": round(float(overall_risk), 4),
            "tension_multiplier": round(float(tension_multiplier), 4),
            "top_contested": [r["name"] for r in results[:3]],
            "high_disruption_risk": [r["name"] for r in results if r["disruption_probability"] > 0.4],
        }

    @staticmethod
    def event_shock_simulator(
        us_china_tension: float = 0.5,
        russia_west_tension: float = 0.6,
        middle_east_instability: float = 0.5,
        trade_protectionism: float = 0.35,
        n_simulations: int = 1000,
        shock_events: Optional[list] = None,
        seed: int = 42,
    ) -> dict:
        np.random.seed(seed)
        if shock_events is None:
            shock_events = [
                {"name": "Taiwan_crisis", "base_prob": 0.05, "tension_increase": 0.4, "economic_impact": 0.3, "domains": ["technology", "trade", "military"]},
                {"name": "Russia_NATO_escalation", "base_prob": 0.08, "tension_increase": 0.5, "economic_impact": 0.25, "domains": ["energy", "military", "trade"]},
                {"name": "Middle_East_war", "base_prob": 0.10, "tension_increase": 0.3, "economic_impact": 0.35, "domains": ["energy", "refugees", "trade"]},
                {"name": "Cyber_attack_critical_infra", "base_prob": 0.12, "tension_increase": 0.2, "economic_impact": 0.15, "domains": ["technology", "finance", "infrastructure"]},
                {"name": "Trade_war_escalation", "base_prob": 0.15, "tension_increase": 0.25, "economic_impact": 0.20, "domains": ["trade", "supply_chain", "technology"]},
                {"name": "Nuclear_proliferation_event", "base_prob": 0.03, "tension_increase": 0.6, "economic_impact": 0.40, "domains": ["military", "diplomacy", "energy"]},
                {"name": "Pandemic_reemergence", "base_prob": 0.07, "tension_increase": 0.15, "economic_impact": 0.30, "domains": ["health", "trade", "social"]},
                {"name": "Resource_nationalism", "base_prob": 0.10, "tension_increase": 0.20, "economic_impact": 0.25, "domains": ["energy", "commodities", "trade"]},
            ]
        current_tension = (us_china_tension + russia_west_tension + middle_east_instability + trade_protectionism) / 4
        total_shocks = 0
        shock_counts = {e["name"]: 0 for e in shock_events}
        cascade_count = 0
        max_tension_reached = current_tension
        economic_impacts = []
        for _ in range(n_simulations):
            sim_tension = current_tension
            sim_econ_impact = 0
            sim_shocks = []
            for event in shock_events:
                adjusted_prob = event["base_prob"] * (1 + sim_tension * 0.5)
                if np.random.random() < adjusted_prob:
                    sim_shocks.append(event["name"])
                    shock_counts[event["name"]] = shock_counts.get(event["name"], 0) + 1
                    sim_tension = min(1.0, sim_tension + event["tension_increase"] * (1 - sim_tension))
                    sim_econ_impact += event["economic_impact"]
                    if len(sim_shocks) >= 2:
                        for cascade_event in shock_events:
                            if cascade_event["name"] not in sim_shocks:
                                cascade_prob = cascade_event["base_prob"] * (1 + sim_tension * 1.5)
                                if np.random.random() < cascade_prob * 0.5:
                                    sim_shocks.append(cascade_event["name"])
                                    shock_counts[cascade_event["name"]] = shock_counts.get(cascade_event["name"], 0) + 1
                                    sim_tension = min(1.0, sim_tension + cascade_event["tension_increase"] * 0.5)
                                    sim_econ_impact += cascade_event["economic_impact"] * 0.5
            if len(sim_shocks) >= 2:
                cascade_count += 1
            max_tension_reached = max(max_tension_reached, sim_tension)
            total_shocks += len(sim_shocks)
            economic_impacts.append(sim_econ_impact)
        avg_tension_reached = current_tension + (max_tension_reached - current_tension) * 0.5
        shock_probs = {name: count / n_simulations for name, count in shock_counts.items()}
        sorted_events = sorted(shock_probs.items(), key=lambda x: x[1], reverse=True)
        cascade_prob = cascade_count / n_simulations
        avg_econ_impact = float(np.mean(economic_impacts)) if economic_impacts else 0
        p90_econ_impact = float(np.percentile(economic_impacts, 90)) if economic_impacts else 0
        geopolitical_risk_level = "critical" if avg_tension_reached > 0.8 else "high" if avg_tension_reached > 0.6 else "moderate" if avg_tension_reached > 0.4 else "low"
        return {
            "current_tension_index": round(float(current_tension), 4),
            "avg_projected_tension": round(float(avg_tension_reached), 4),
            "max_tension_reached": round(float(max_tension_reached), 4),
            "geopolitical_risk_level": geopolitical_risk_level,
            "cascade_probability": round(float(cascade_prob), 4),
            "avg_economic_impact": round(float(avg_econ_impact), 4),
            "p90_economic_impact": round(float(p90_econ_impact), 4),
            "event_probabilities": [{"event": name, "probability": round(prob, 4)} for name, prob in sorted_events],
            "most_likely_shock": sorted_events[0][0] if sorted_events else None,
            "avg_shocks_per_year": round(float(total_shocks / n_simulations), 2),
            "n_simulations": n_simulations,
            "recommendations": [
                f"Monitor {sorted_events[0][0].replace('_', ' ')} — {sorted_events[0][1]*100:.0f}% annual probability" if sorted_events else "Low risk environment",
                f"Cascade risk: {cascade_prob*100:.0f}% — {'High' if cascade_prob > 0.2 else 'Moderate' if cascade_prob > 0.1 else 'Low'} multi-event correlation",
                f"P90 economic impact: {p90_econ_impact*100:.0f}% GDP — {'Severe' if p90_econ_impact > 0.5 else 'Moderate' if p90_econ_impact > 0.2 else 'Manageable'} stress scenario",
            ],
        }