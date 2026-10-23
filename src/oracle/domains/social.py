import numpy as np
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class SocialPrediction:
    unrest_probability: float = 0.0
    polarization_index: float = 0.0
    migration_pressure: float = 0.0
    inequality_gini: float = 0.0
    youth_bulge: float = 0.0
    regime_stability: float = 0.0
    social_cohesion: float = 0.0


class SocialDynamicsModel:

    def __init__(self):
        self.name = "social_dynamics_model"

    @staticmethod
    def unrest_prediction(
        gini_coefficient: float = 0.41,
        unemployment_rate: float = 0.06,
        inflation_rate: float = 0.03,
        youth_unemployment: float = 0.12,
        food_price_index: float = 1.0,
        political_instability: float = 0.3,
        social_media_penetration: float = 0.7,
        regime_repression: float = 0.3,
        ethnic_fractionalization: float = 0.4,
        urbanization_rate: float = 0.55,
        gdp_per_capita: float = 15000,
        education_index: float = 0.7,
        press_freedom: float = 0.6,
    ) -> dict:
        economic_distress = (
            0.25 * (gini_coefficient / 0.5) +
            0.20 * (unemployment_rate / 0.15) +
            0.15 * (inflation_rate / 0.10) +
            0.15 * (youth_unemployment / 0.25) +
            0.10 * (food_price_index / 1.5) +
            0.15 * (1 - min(gdp_per_capita / 50000, 1.0))
        )
        political_risk = (
            0.30 * political_instability +
            0.20 * regime_repression +
            0.20 * (1 - press_freedom) +
            0.15 * ethnic_fractionalization +
            0.15 * (1 - min(education_index, 1.0))
        )
        mobilization_factor = (
            0.40 * social_media_penetration +
            0.30 * urbanization_rate +
            0.30 * (1 - regime_repression)
        )
        unrest_prob = 1 / (1 + np.exp(-(economic_distress * 0.5 + political_risk * 0.35 + mobilization_factor * 0.15 - 0.5)))
        if unrest_prob > 0.7:
            level = "critical"
            triggers = ["mass_protests", "regime_crisis", "systemic_breakdown"]
        elif unrest_prob > 0.45:
            level = "high"
            triggers = ["widespread_protests", "political_polarization", "economic_grievances"]
        elif unrest_prob > 0.25:
            level = "moderate"
            triggers = ["localized_protests", "sector_discontent", "rising_polarization"]
        elif unrest_prob > 0.10:
            level = "low"
            triggers = ["sporadic_unrest", "social_tension"]
        else:
            level = "minimal"
            triggers = ["stable_conditions"]
        cohesion = max(0, 1 - unrest_prob * 1.5 - political_instability * 0.3)
        return {
            "unrest_probability": round(float(unrest_prob), 4),
            "unrest_level": level,
            "likely_triggers": triggers,
            "economic_distress": round(float(economic_distress), 4),
            "political_risk": round(float(political_risk), 4),
            "mobilization_factor": round(float(mobilization_factor), 4),
            "social_cohesion": round(float(cohesion), 4),
            "indicators": {
                "gini": round(gini_coefficient, 3),
                "unemployment": round(unemployment_rate, 3),
                "inflation": round(inflation_rate, 3),
                "youth_unemployment": round(youth_unemployment, 3),
                "food_price_index": round(food_price_index, 2),
                "political_instability": round(political_instability, 3),
                "regime_repression": round(regime_repression, 3),
                "social_media": round(social_media_penetration, 3),
                "ethnic_fractionalization": round(ethnic_fractionalization, 3),
            },
        }

    @staticmethod
    def migration_prediction(
        origin_gdp_per_capita: float = 5000,
        destination_gdp_per_capita: float = 50000,
        origin_conflict: float = 0.3,
        climate_risk: float = 0.2,
        origin_population: float = 50e6,
        border_openness: float = 0.5,
        diaspora_existing: float = 0.1,
        language_similarity: float = 0.3,
        colonial_tie: float = 0.0,
    ) -> dict:
        wage_gap = max(0, np.log(destination_gdp_per_capita / max(origin_gdp_per_capita, 100)))
        push_factor = (
            0.30 * origin_conflict +
            0.25 * (1 - min(origin_gdp_per_capita / 20000, 1.0)) +
            0.20 * climate_risk +
            0.25 * (1 - min(origin_gdp_per_capita / 10000, 1.0))
        )
        pull_factor = (
            0.25 * border_openness +
            0.20 * diaspora_existing +
            0.20 * language_similarity +
            0.15 * colonial_tie +
            0.20 * min(wage_gap / 3, 1.0)
        )
        migration_pressure = push_factor * 0.6 + (1 - push_factor) * pull_factor * 0.4
        base_flow = origin_population * 0.001 * migration_pressure
        projected_5yr = base_flow * 5 * (1 + climate_risk * 0.5)
        if migration_pressure > 0.6:
            level = "mass_exodus"
        elif migration_pressure > 0.4:
            level = "high"
        elif migration_pressure > 0.2:
            level = "moderate"
        else:
            level = "low"
        destination_impact = {
            "labor_market": round(float(min(migration_pressure * 1.2, 0.8)), 3),
            "housing_pressure": round(float(min(migration_pressure * 1.5, 0.9)), 3),
            "cultural_tension": round(float(min(migration_pressure * 0.8 * (1 - language_similarity), 0.7)), 3),
            "fiscal_burden": round(float(min(migration_pressure * 0.7, 0.6)), 3),
            "gdp_boost": round(float(migration_pressure * 0.3), 3),
        }
        return {
            "migration_pressure": round(float(migration_pressure), 4),
            "migration_level": level,
            "push_factors": round(float(push_factor), 4),
            "pull_factors": round(float(pull_factor), 4),
            "wage_gap_log": round(float(wage_gap), 3),
            "projected_flow_5yr": round(float(projected_5yr), 0),
            "flow_as_pct_origin": round(float(projected_5yr / origin_population * 100), 3),
            "destination_impact": destination_impact,
        }

    @staticmethod
    def polarization_index(
        income_inequality: float = 0.41,
        political_polarization: float = 0.6,
        media_fragmentation: float = 0.7,
        trust_in_institutions: float = 0.3,
        ideological_distance: float = 0.5,
        echo_chamber_effect: float = 0.6,
    ) -> dict:
        polarization = (
            0.20 * income_inequality +
            0.25 * political_polarization +
            0.15 * media_fragmentation +
            0.15 * (1 - trust_in_institutions) +
            0.15 * ideological_distance +
            0.10 * echo_chamber_effect
        )
        fragmentation = (
            0.30 * media_fragmentation +
            0.25 * echo_chamber_effect +
            0.20 * (1 - trust_in_institutions) +
            0.15 * ideological_distance +
            0.10 * income_inequality
        )
        if polarization > 0.7:
            state = "deeply_divided"
            risk = "civil_conflict", "democratic_backsliding"
        elif polarization > 0.5:
            state = "polarized"
            risk = "institutional_paralysis", "populist_surge"
        elif polarization > 0.3:
            state = "moderately_polarized"
            risk = "legislative_gridlock", "social_tension"
        else:
            state = "relatively_unified"
            risk = "low_social_risk"
        return {
            "polarization_index": round(float(polarization), 4),
            "fragmentation_index": round(float(fragmentation), 4),
            "state": state,
            "primary_risks": list(risk) if isinstance(risk, tuple) else [risk],
            "components": {
                "income_inequality": round(income_inequality, 3),
                "political_polarization": round(political_polarization, 3),
                "media_fragmentation": round(media_fragmentation, 3),
                "trust_deficit": round(1 - trust_in_institutions, 3),
                "ideological_distance": round(ideological_distance, 3),
                "echo_chamber": round(echo_chamber_effect, 3),
            },
        }

    @staticmethod
    def wealth_power_shift(
        regions: Optional[List[dict]] = None,
        metrics: Optional[List[str]] = None,
        horizon: int = 2035,
    ) -> dict:
        current_year = 2025
        years_ahead = horizon - current_year
        if regions is None:
            regions = [
                {"name": "North America", "wealth": 0.34, "tech": 0.40, "pop": 0.05, "growth": 0.02, "stability": 0.85},
                {"name": "Europe", "wealth": 0.22, "tech": 0.20, "pop": 0.07, "growth": 0.015, "stability": 0.80},
                {"name": "East Asia", "wealth": 0.25, "tech": 0.30, "pop": 0.20, "growth": 0.04, "stability": 0.65},
                {"name": "South Asia", "wealth": 0.05, "tech": 0.06, "pop": 0.26, "growth": 0.06, "stability": 0.50},
                {"name": "Middle East", "wealth": 0.04, "tech": 0.03, "pop": 0.04, "growth": 0.03, "stability": 0.40},
                {"name": "Africa", "wealth": 0.02, "tech": 0.02, "pop": 0.18, "growth": 0.05, "stability": 0.35},
                {"name": "Latin America", "wealth": 0.05, "tech": 0.04, "pop": 0.08, "growth": 0.025, "stability": 0.55},
            ]
        if metrics is None:
            metrics = ["gdp_share", "tech_share", "influence_score"]
        results = []
        for r in regions:
            growth = r["growth"] * (1 if r["stability"] > 0.6 else 0.7)
            future_wealth = r["wealth"] * (1 + growth * years_ahead)
            future_tech = r["tech"] * (1 + r["growth"] * 1.5 * years_ahead * r["stability"])
            influence = (future_wealth * 0.4 + future_tech * 0.35 + r["stability"] * 0.25) * r["pop"] * 20
            current_influence = (r["wealth"] * 0.4 + r["tech"] * 0.35 + r["stability"] * 0.25) * r["pop"] * 20
            results.append({
                "name": r["name"],
                "current_influence": round(float(current_influence), 4),
                "projected_influence": round(float(influence), 4),
                "shift": round(float(influence - current_influence), 4),
                "direction": "rising" if influence > current_influence * 1.05 else "declining" if influence < current_influence * 0.95 else "stable",
                "projected_wealth_share": round(float(future_wealth), 4),
                "projected_tech_share": round(float(future_tech), 4),
            })
        results.sort(key=lambda x: x["projected_influence"], reverse=True)
        return {
            "horizon": horizon,
            "regions": results,
            "biggest_gainer": max(results, key=lambda x: x["shift"]),
            "biggest_loser": min(results, key=lambda x: x["shift"]),
        }

    @staticmethod
    def civil_unrest_scenario(
        base_unrest: float = 0.3,
        trigger_events: Optional[List[dict]] = None,
    ) -> dict:
        if trigger_events is None:
            trigger_events = [
                {"event": "food_price_spike", "probability": 0.3, "impact": 0.4},
                {"event": "election_dispute", "probability": 0.2, "impact": 0.6},
                {"event": "economic_crisis", "probability": 0.15, "impact": 0.7},
                {"event": "social_media_viral_event", "probability": 0.4, "impact": 0.3},
                {"event": "police_violence_incident", "probability": 0.15, "impact": 0.5},
                {"event": "labor_strike_wave", "probability": 0.25, "impact": 0.35},
            ]
        scenarios = {"baseline": round(base_unrest, 3)}
        for n_events in range(1, 4):
            from itertools import combinations
            max_impact = 0
            for combo in combinations(trigger_events, n_events):
                prob = np.prod([e["probability"] for e in combo])
                impact = base_unrest + sum(e["impact"] for e in combo) * (1 - base_unrest) * 0.5
                if impact > max_impact:
                    max_impact = impact
            scenarios[f"{n_events}_event_cascade"] = round(min(float(max_impact), 0.95), 3)
        worst_case = base_unrest
        for event in trigger_events:
            worst_case = worst_case + event["impact"] * (1 - worst_case) * 0.6
        worst_case = min(worst_case, 0.98)
        scenarios["worst_case"] = round(float(worst_case), 3)
        return {
            "base_unrest": round(base_unrest, 3),
            "scenarios": scenarios,
            "trigger_events": trigger_events,
            "recommendation": "Monitor key indicators: food prices, social media sentiment, regime signals" if base_unrest > 0.3 else "Low risk environment — maintain resilience",
        }

    @staticmethod
    def civil_unrest_cascade(
        n_groups: int = 8,
        base_activation: float = 0.2,
        connectivity: float = 0.3,
        threshold_spread: float = 0.25,
        n_simulations: int = 500,
        seed: int = 42,
    ) -> dict:
        np.random.seed(seed)
        group_names = ["urban_youth", "labor_unions", "ethnic_minorities", "students", "religious_groups", "rural_poor", "middle_class", "military_factions"][:n_groups]
        total_activated = []
        cascade_sizes = []
        n_full_cascade = 0
        first_cascade = None
        for sim in range(n_simulations):
            activated = set()
            initially_activated = set()
            for i in range(n_groups):
                if np.random.random() < base_activation:
                    activated.add(i)
                    initially_activated.add(i)
            step = 0
            while True:
                newly_activated = set()
                for i in range(n_groups):
                    if i in activated:
                        continue
                    n_active_neighbors = 0
                    for j in range(n_groups):
                        if j != i and np.random.random() < connectivity and j in activated:
                            n_active_neighbors += 1
                    frac = n_active_neighbors / max(n_groups - 1, 1)
                    if frac >= threshold_spread or (n_active_neighbors >= 2 and np.random.random() < frac):
                        newly_activated.add(i)
                if not newly_activated:
                    break
                activated.update(newly_activated)
                step += 1
                if step > 20:
                    break
            frac_activated = len(activated) / n_groups
            total_activated.append(frac_activated)
            cascade_sizes.append(len(activated))
            if len(activated) == n_groups:
                n_full_cascade += 1
                if first_cascade is None:
                    first_cascade = step
        avg_activation = float(np.mean(total_activated))
        avg_cascade_size = float(np.mean(cascade_sizes))
        full_cascade_prob = n_full_cascade / n_simulations
        group_risk = []
        sim_rng = np.random.RandomState(seed)
        for i in range(n_groups):
            influence_count = 0
            for _ in range(200):
                can_spread = 0
                for j in range(n_groups):
                    if j != i and sim_rng.random() < connectivity:
                        can_spread += 1
                influence_count += can_spread
            group_risk.append({
                "group": group_names[i] if i < len(group_names) else f"group_{i}",
                "connectivity_degree": round(influence_count / 200, 2),
                "risk_contribution": round(avg_activation * (1 + influence_count / (200 * n_groups)), 3),
            })
        group_risk.sort(key=lambda x: x["risk_contribution"], reverse=True)
        tipping_point = round(1 - threshold_spread, 3)
        return {
            "n_groups": n_groups,
            "base_activation": base_activation,
            "connectivity": connectivity,
            "threshold": threshold_spread,
            "n_simulations": n_simulations,
            "avg_activation_frac": round(avg_activation, 4),
            "avg_cascade_size": round(avg_cascade_size, 2),
            "full_cascade_probability": round(full_cascade_prob, 4),
            "full_cascade_steps_to_complete": first_cascade,
            "tipping_point_indicator": tipping_point,
            "group_risk_ranking": group_risk,
            "cascade_risk_level": "critical" if full_cascade_prob > 0.3 else "high" if avg_activation > 0.6 else "moderate" if avg_activation > 0.3 else "low",
            "intervention_points": [
                f"Reduce connectivity between {group_risk[0]['group']} and {group_risk[1]['group']}" if len(group_risk) > 1 else "Reduce base activation",
                "Address grievances of most connected groups first",
                "Increase threshold for spread through institutional resilience",
            ][:3],
        }