from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CompoundHazardState:
    seismic_hazard: float = 0.3
    volcanic_risk: float = 0.1
    solar_storm_risk: float = 0.2
    hurricane_risk: float = 0.3
    flood_risk: float = 0.3
    drought_risk: float = 0.2
    pandemic_risk: float = 0.15
    cyber_attack_risk: float = 0.3
    supply_chain_fragility: float = 0.3
    infrastructure_resilience: float = 0.6
    economic_vulnerability: float = 0.3
    social_cohesion: float = 0.5
    governance_capacity: float = 0.6
    climate_stress: float = 0.4
    biodiversity_collapse_risk: float = 0.3


COMPOUND_SCENARIOS = {
    "earthquake_tsunami_nuclear": {
        "name": "Earthquake-Tsunami-Nuclear Cascade",
        "triggers": ["seismic_hazard"],
        "amplifiers": ["infrastructure_resilience", "economic_vulnerability"],
        "description": "Major earthquake triggers tsunami causing nuclear accident — compound cascading failure across physical, energy, and environmental domains",
    },
    "solar_storm_grid_cyber": {
        "name": "Solar Storm-Grid Collapse-Cyber Attack",
        "triggers": ["solar_storm_risk"],
        "amplifiers": ["infrastructure_resilience", "cyber_attack_risk", "economic_vulnerability"],
        "description": "Carrington-class solar storm causes grid collapse, creating opportunity for cyber attack on crippled infrastructure",
    },
    "pandemic_supply_cyber": {
        "name": "Pandemic-Supply Chain-Cyber Cascade",
        "triggers": ["pandemic_risk"],
        "amplifiers": ["supply_chain_fragility", "cyber_attack_risk", "social_cohesion"],
        "description": "Pandemic disrupts supply chains while cyber attacks target healthcare and logistics systems",
    },
    "drought_flood_food_rebellion": {
        "name": "Drought-Flood-Food Insecurity-Social Unrest",
        "triggers": ["drought_risk", "flood_risk"],
        "amplifiers": ["climate_stress", "economic_vulnerability", "social_cohesion", "governance_capacity"],
        "description": "Concurrent drought and flooding causes widespread crop failure, food insecurity triggers social unrest and governance failure",
    },
    "volcanic_winter_pandemic": {
        "name": "Volcanic Winter-Pandemic Compound",
        "triggers": ["volcanic_risk"],
        "amplifiers": ["climate_stress", "pandemic_risk", "economic_vulnerability"],
        "description": "Major eruption causes volcanic winter, weakening populations and triggering pandemic while crushing economic output",
    },
    "biodiversity_flood_drought": {
        "name": "Ecosystem Collapse-Climate Extreme Cascade",
        "triggers": ["biodiversity_collapse_risk"],
        "amplifiers": ["drought_risk", "flood_risk", "climate_stress"],
        "description": "Biodiversity collapse reduces natural flood/drought buffering, compounding climate extremes",
    },
    "hurricane_flood_infra_cyber": {
        "name": "Hurricane-Flood-Infrastructure-Cyber Cascade",
        "triggers": ["hurricane_risk", "flood_risk"],
        "amplifiers": ["infrastructure_resilience", "cyber_attack_risk", "economic_vulnerability"],
        "description": "Major hurricane floods critical infrastructure, cyber attack exploits emergency response weakness",
    },
    "multi_hazard_synchronous": {
        "name": "Synchronous Multi-Hazard Event",
        "triggers": ["seismic_hazard", "flood_risk", "solar_storm_risk"],
        "amplifiers": ["infrastructure_resilience", "supply_chain_fragility", "social_cohesion", "economic_vulnerability"],
        "description": "Multiple natural hazards strike simultaneously — earthquake + flooding + solar storm creates compounding cascading failures",
    },
}


class CompoundHazardModel:

    def __init__(self):
        self.name = "compound_hazard_model"

    def set_state(self, **kwargs) -> "CompoundHazardModel":
        current = CompoundHazardState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, float(v))
        self.state = current
        return self

    def assess_compound_risk(self, state: Optional[CompoundHazardState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", CompoundHazardState())

        resilience = (
            state.infrastructure_resilience * 0.25
            + state.governance_capacity * 0.25
            + state.social_cohesion * 0.20
            + (1.0 - state.supply_chain_fragility) * 0.15
            + (1.0 - state.economic_vulnerability) * 0.15
        )
        resilience = min(1.0, max(0.0, resilience))

        hazard_burden = (
            state.seismic_hazard * 0.12
            + state.volcanic_risk * 0.08
            + state.solar_storm_risk * 0.08
            + state.hurricane_risk * 0.10
            + state.flood_risk * 0.10
            + state.drought_risk * 0.08
            + state.pandemic_risk * 0.08
            + state.cyber_attack_risk * 0.08
            + state.climate_stress * 0.10
            + state.biodiversity_collapse_risk * 0.08
            + state.supply_chain_fragility * 0.05
            + state.economic_vulnerability * 0.05
        )
        hazard_burden = min(1.0, max(0.0, hazard_burden))

        compound_index = hazard_burden * (1.5 - resilience * 0.5)
        compound_index = min(1.0, max(0.0, compound_index))

        if compound_index < 0.15:
            level = "minimal"
        elif compound_index < 0.30:
            level = "low"
        elif compound_index < 0.45:
            level = "moderate"
        elif compound_index < 0.60:
            level = "elevated"
        elif compound_index < 0.75:
            level = "high"
        else:
            level = "critical"

        top_hazards = sorted([
            ("seismic", state.seismic_hazard),
            ("volcanic", state.volcanic_risk),
            ("solar_storm", state.solar_storm_risk),
            ("hurricane", state.hurricane_risk),
            ("flood", state.flood_risk),
            ("drought", state.drought_risk),
            ("pandemic", state.pandemic_risk),
            ("cyber", state.cyber_attack_risk),
            ("climate", state.climate_stress),
            ("biodiversity", state.biodiversity_collapse_risk),
        ], key=lambda x: x[1], reverse=True)

        worst_pair_prob = self._pair_coupling(state, top_hazards[0][0], top_hazards[1][0])
        worst_triple_prob = self._triple_coupling(state, top_hazards[0][0], top_hazards[1][0], top_hazards[2][0])

        systemic_collapse_prob = (
            compound_index * 0.3
            + worst_triple_prob * 0.25
            + (1.0 - resilience) * 0.25
            + (1.0 - state.governance_capacity) * 0.1
            + state.economic_vulnerability * 0.1
        )
        systemic_collapse_prob = min(0.5, max(0.0, systemic_collapse_prob))

        recommendations = []
        if compound_index > 0.5:
            recommendations.append("Critical compound risk — multiple hazard domains simultaneously elevated")
        if resilience < 0.3:
            recommendations.append("Low system resilience — invest in infrastructure hardening and governance capacity")
        if worst_pair_prob > 0.1:
            recommendations.append(f"High coupling between {top_hazards[0][0]} and {top_hazards[1][0]} — prepare for cascading failure")
        if state.cyber_attack_risk > 0.5 and state.infrastructure_resilience < 0.5:
            recommendations.append("Cyber-physical cascade risk — harden critical infrastructure against cyber exploitation during physical crises")
        if state.pandemic_risk > 0.3 and state.supply_chain_fragility > 0.4:
            recommendations.append("Pandemic-supply compound — diversify supply chains and build strategic reserves")

        return sanitize_nan({
            "compound_hazard_index": compound_index,
            "hazard_level": level,
            "resilience_score": resilience,
            "hazard_burden": hazard_burden,
            "top_hazards": [{"hazard": h, "score": round(s, 3)} for h, s in top_hazards[:5]],
            "worst_pair_probability": worst_pair_prob,
            "worst_triple_probability": worst_triple_prob,
            "systemic_collapse_probability": systemic_collapse_prob,
            "recommendations": recommendations[:5],
        })

    def simulate_scenario(self, scenario: str = "earthquake_tsunami_nuclear",
                          years: int = 10, n_simulations: int = 500,
                          state: Optional[CompoundHazardState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", CompoundHazardState())

        config = COMPOUND_SCENARIOS.get(scenario, COMPOUND_SCENARIOS["earthquake_tsunami_nuclear"])

        trigger_prods = 1.0
        for t in config["triggers"]:
            val = getattr(state, t, 0.3)
            trigger_prods *= val

        annual_trigger_prob = min(0.3, trigger_prods * 0.5)

        amplifier_effect = 1.0
        for a in config["amplifiers"]:
            val = getattr(state, a, 0.5)
            if a in ("infrastructure_resilience", "governance_capacity", "social_cohesion"):
                amplifier_effect *= (1.5 - val * 0.5)
            else:
                amplifier_effect *= (0.5 + val * 0.5)

        cascade_prob = annual_trigger_prob * amplifier_effect
        cascade_prob = min(0.5, cascade_prob)

        np.random.seed(42)
        sim_results = []
        for _ in range(n_simulations):
            triggered = np.random.random() < cascade_prob
            if triggered:
                n_cascades = np.random.poisson(2) + 1
                total_impact = min(1.0, np.random.beta(2, 5) * amplifier_effect * (1 + n_cascades * 0.15))
                recovery_years = max(1, int(np.random.exponential(3) * (1 + total_impact)))
            else:
                n_cascades = 0
                total_impact = 0.0
                recovery_years = 0
            sim_results.append({
                "triggered": bool(triggered),
                "cascades": int(n_cascades),
                "impact": float(total_impact),
                "recovery_years": int(recovery_years),
            })

        n_triggered = sum(1 for s in sim_results if s["triggered"])
        triggered_results = [s for s in sim_results if s["triggered"]]
        if triggered_results:
            mean_impact = np.mean([s["impact"] for s in triggered_results])
            mean_cascades = np.mean([s["cascades"] for s in triggered_results])
            mean_recovery = np.mean([s["recovery_years"] for s in triggered_results])
            p95_impact = float(np.percentile([s["impact"] for s in triggered_results], 95))
        else:
            mean_impact = 0.0
            mean_cascades = 0.0
            mean_recovery = 0.0
            p95_impact = 0.0

        period_prob = 1.0 - (1.0 - cascade_prob) ** years

        timeline = []
        for y in range(1, years + 1):
            annual_prob = cascade_prob
            cumulative_no_event = (1.0 - cascade_prob) ** y
            timeline.append({
                "year": y,
                "annual_probability": round(annual_prob, 4),
                "cumulative_probability": round(1.0 - cumulative_no_event, 4),
                "expected_impact_if_triggered": round(float(mean_impact), 3),
            })

        return sanitize_nan({
            "scenario": scenario,
            "scenario_name": config["name"],
            "description": config["description"],
            "annual_trigger_probability": round(float(annual_trigger_prob), 4),
            "cascade_amplification": round(float(amplifier_effect), 3),
            "annual_cascade_probability": round(float(cascade_prob), 4),
            f"period_{years}yr_probability": round(float(period_prob), 4),
            "simulations": n_simulations,
            "triggered_count": n_triggered,
            "mean_impact_given_trigger": round(float(mean_impact), 3),
            "p95_impact_given_trigger": round(float(p95_impact), 3),
            "mean_cascade_steps": round(float(mean_cascades), 1),
            "mean_recovery_years": round(float(mean_recovery), 1),
            "timeline": timeline,
        })

    def multi_hazard_simultaneous(self, n_hazards: int = 3,
                                   years: int = 10,
                                   n_simulations: int = 1000,
                                   state: Optional[CompoundHazardState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", CompoundHazardState())

        hazard_rates = {
            "seismic": state.seismic_hazard * 0.02,
            "volcanic": state.volcanic_risk * 0.01,
            "solar_storm": state.solar_storm_risk * 0.03,
            "hurricane": state.hurricane_risk * 0.15,
            "flood": state.flood_risk * 0.10,
            "drought": state.drought_risk * 0.05,
            "pandemic": state.pandemic_risk * 0.02,
            "cyber": state.cyber_attack_risk * 0.10,
        }

        np.random.seed(42)
        simultaneous_counts = []
        for _ in range(n_simulations):
            active = 0
            for hz, rate in hazard_rates.items():
                if np.random.random() < rate:
                    active += 1
            simultaneous_counts.append(active)

        probs = {}
        for k in range(n_hazards + 1):
            count = sum(1 for c in simultaneous_counts if c >= k)
            probs[f"{k}_or_more"] = round(count / n_simulations, 4)

        mean_simultaneous = float(np.mean(simultaneous_counts))
        max_simultaneous = int(max(simultaneous_counts))

        resilience = (
            state.infrastructure_resilience * 0.3
            + state.governance_capacity * 0.3
            + state.social_cohesion * 0.2
            + (1.0 - state.economic_vulnerability) * 0.2
        )

        multi_year_probs = {}
        for k in range(n_hazards + 1):
            p_k = probs.get(f"{k}_or_more", 0)
            multi_year_probs[k] = round(1.0 - (1.0 - p_k) ** years, 4)

        return sanitize_nan({
            "n_hazards_threshold": n_hazards,
            "mean_simultaneous": round(mean_simultaneous, 2),
            "max_simultaneous_observed": max_simultaneous,
            "simultaneous_1yr_probabilities": probs,
            f"simultaneous_{years}yr_probabilities": multi_year_probs,
            "resilience_capacity": round(float(resilience), 3),
            "most_likely_hazard": max(hazard_rates, key=hazard_rates.get),
            "highest_rate": round(float(max(hazard_rates.values())), 4),
        })

    def _pair_coupling(self, state, hazard_a: str, hazard_b: str) -> float:
        coupling_matrix = {
            ("seismic", "flood"): 0.15,
            ("seismic", "volcanic"): 0.10,
            ("hurricane", "flood"): 0.25,
            ("drought", "flood"): 0.05,
            ("solar_storm", "cyber"): 0.12,
            ("pandemic", "cyber"): 0.08,
            ("climate", "drought"): 0.20,
            ("climate", "flood"): 0.15,
            ("biodiversity", "drought"): 0.10,
            ("biodiversity", "flood"): 0.08,
        }
        key1 = (hazard_a, hazard_b)
        key2 = (hazard_b, hazard_a)
        coupling = coupling_matrix.get(key1, coupling_matrix.get(key2, 0.02))
        val_a = getattr(state, hazard_a, 0.3) if hasattr(state, hazard_a) else 0.3
        val_b = getattr(state, hazard_b, 0.3) if hasattr(state, hazard_b) else 0.3
        return min(0.5, coupling * min(val_a, val_b) * 2)

    def _triple_coupling(self, state, a: str, b: str, c: str) -> float:
        pair_ab = self._pair_coupling(state, a, b)
        val_c = getattr(state, c, 0.3) if hasattr(state, c) else 0.3
        return min(0.3, pair_ab * val_c * 0.5)


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