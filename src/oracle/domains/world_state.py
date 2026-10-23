from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class WorldStateAssessment:
    overall_stability: float = 0.5
    human_health_index: float = 0.5
    natural_forces_index: float = 0.5
    infrastructure_integrity: float = 0.5
    governance_effectiveness: float = 0.5
    economic_resilience: float = 0.5
    ecological_integrity: float = 0.5
    technological_readiness: float = 0.5
    social_cohesion: float = 0.5
    compound_hazard_burden: float = 0.3
    climate_commitment: float = 0.5


DOMAIN_WEIGHTS = {
    "geopolitical": 0.08,
    "climate": 0.10,
    "economic": 0.10,
    "sentiment": 0.04,
    "epidemiology": 0.06,
    "technology": 0.06,
    "social": 0.06,
    "energy": 0.06,
    "commodities": 0.04,
    "infrastructure": 0.08,
    "supply_chain": 0.04,
    "cyber": 0.04,
    "water": 0.05,
    "food": 0.05,
    "political": 0.04,
    "biodiversity": 0.05,
    "pandemic": 0.05,
    "space": 0.02,
    "mental_health": 0.04,
    "housing": 0.03,
    "seismic": 0.04,
    "solar": 0.04,
    "oceanic": 0.05,
    "neo": 0.02,
    "atmospheric": 0.04,
    "ecological": 0.05,
    "paleoclimate": 0.03,
    "cosmic_radiation": 0.02,
    "gravitational": 0.02,
}

CLUSTER_DEFINITIONS = {
    "human_health": ["epidemiology", "pandemic", "mental_health", "food", "water"],
    "natural_forces": ["seismic", "solar", "oceanic", "neo", "atmospheric", "cosmic_radiation", "gravitational", "paleoclimate"],
    "infrastructure_integrity": ["infrastructure", "supply_chain", "cyber", "energy"],
    "governance_effectiveness": ["political", "geopolitical", "social"],
    "economic_resilience": ["economic", "sentiment", "commodities", "housing"],
    "ecological_integrity": ["biodiversity", "ecological", "climate", "oceanic"],
    "technological_readiness": ["technology", "space"],
    "social_cohesion": ["social", "mental_health", "housing", "political"],
}

THREAT_LEVELS = {
    (0.0, 0.2): "existential_crisis",
    (0.2, 0.35): "severe_danger",
    (0.35, 0.5): "elevated_threat",
    (0.5, 0.65): "moderate_concern",
    (0.65, 0.8): "guarded_stability",
    (0.8, 1.01): "favorable_trajectory",
}


class WorldStateModel:

    def __init__(self):
        self.name = "world_state_model"

    def assess_world_state(self, domain_scores: dict[str, float]) -> dict:
        overall = 0.0
        total_weight = 0.0
        for domain, score in domain_scores.items():
            w = DOMAIN_WEIGHTS.get(domain, 0.02)
            overall += score * w
            total_weight += w
        if total_weight > 0:
            overall = overall / total_weight
        overall = min(1.0, max(0.0, overall))

        clusters = {}
        for cluster_name, domains in CLUSTER_DEFINITIONS.items():
            vals = [domain_scores.get(d, 0.5) for d in domains if d in domain_scores]
            if vals:
                clusters[cluster_name] = round(float(np.mean(vals)), 3)
            else:
                clusters[cluster_name] = 0.5

        threat_level = "moderate_concern"
        for (lo, hi), level in THREAT_LEVELS.items():
            if lo <= overall < hi:
                threat_level = level
                break

        worst_domains = sorted(domain_scores.items(), key=lambda x: x[1])[:5]
        best_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        imbalances = []
        if len(domain_scores) >= 4:
            vals = list(domain_scores.values())
            std = float(np.std(vals))
            mean = float(np.mean(vals))
            if std > 0.15:
                imbalances.append("High inter-domain variance — systemic imbalances creating fragility")
            if mean < 0.4:
                imbalances.append("Multi-domain distress — systemic degradation across most indicators")
            if clusters.get("natural_forces", 0.5) - clusters.get("economic_resilience", 0.5) > 0.2:
                imbalances.append("Natural forces outpacing economic adaptation capacity")
            if clusters.get("human_health", 0.5) < 0.35:
                imbalances.append("Human health cluster in critical territory — compound disease vulnerability")

        recommendations = []
        if overall < 0.4:
            recommendations.append("Systemic crisis: multiple domains simultaneously degraded — emergency cross-domain coordination required")
        if clusters.get("infrastructure_integrity", 0.5) < 0.4:
            recommendations.append("Infrastructure integrity critical — harden physical and digital systems against cascade failures")
        if clusters.get("ecological_integrity", 0.5) < 0.35:
            recommendations.append("Ecological integrity collapsing — irreversible tipping point risk demands immediate conservation action")
        if clusters.get("governance_effectiveness", 0.5) < 0.35:
            recommendations.append("Governance failure undermining response capacity across all domains — institutional reform urgent")
        if clusters.get("natural_forces", 0.5) > 0.6:
            recommendations.append("Elevated natural hazard environment — ensure disaster preparedness and early warning systems operational")
        if not recommendations:
            recommendations.append("World state broadly stable — maintain monitoring and continue building resilience across domains")

        compound_burden = 0.0
        n_active = 0
        for domain, score in domain_scores.items():
            if score < 0.4:
                compound_burden += (0.4 - score)
                n_active += 1
        if n_active >= 3:
            compound_burden *= 1.2
        compound_burden = min(1.0, compound_burden)

        return sanitize_nan({
            "overall_stability": round(overall, 3),
            "threat_level": threat_level,
            "n_domains_assessed": len(domain_scores),
            "clusters": clusters,
            "compound_hazard_burden": round(compound_burden, 3),
            "climate_commitment": round(clusters.get("ecological_integrity", 0.5), 3),
            "worst_domains": [{"domain": d, "score": round(s, 3)} for d, s in worst_domains],
            "best_domains": [{"domain": d, "score": round(s, 3)} for d, s in best_domains],
            "imbalances": imbalances[:5],
            "recommendations": recommendations[:5],
        })

    def stability_trajectory(self, domain_scores: dict[str, float],
                              years: int = 50,
                              n_simulations: int = 500) -> dict:
        base_stability = 0.0
        total_weight = 0.0
        for domain, score in domain_scores.items():
            w = DOMAIN_WEIGHTS.get(domain, 0.02)
            base_stability += score * w
            total_weight += w
        if total_weight > 0:
            base_stability /= total_weight
        base_stability = min(1.0, max(0.0, base_stability))

        n_distressed = sum(1 for v in domain_scores.values() if v < 0.4)
        n_crisis = sum(1 for v in domain_scores.values() if v < 0.25)

        rng = np.random.RandomState(42)
        trajectories = []
        collapse_events = 0

        for _ in range(n_simulations):
            stability = base_stability
            annual_drift = 0.002
            if n_distressed > 5:
                annual_drift -= 0.003
            if n_crisis > 2:
                annual_drift -= 0.005
            trajectory = []
            for yr in range(years):
                shock = rng.normal(0, 0.02)
                if rng.random() < 0.02:
                    shock -= rng.uniform(0.05, 0.15)
                stability += annual_drift + shock
                stability = min(1.0, max(0.0, stability))
                trajectory.append(round(stability, 3))
                if stability < 0.1:
                    collapse_events += 1
                    break
            trajectories.append(trajectory)

        final_stabilities = [t[-1] for t in trajectories]
        min_stabilities = [min(t) for t in trajectories]

        timeline = []
        step = max(1, years // 20)
        for yr in range(0, years, step):
            vals = [t[yr] for t in trajectories if yr < len(t)]
            if vals:
                timeline.append({
                    "year": yr,
                    "mean_stability": round(float(np.mean(vals)), 3),
                    "p10_stability": round(float(np.percentile(vals, 10)), 3),
                    "p90_stability": round(float(np.percentile(vals, 90)), 3),
                })

        return sanitize_nan({
            "base_stability": round(base_stability, 3),
            "years_simulated": years,
            "n_simulations": n_simulations,
            "mean_final_stability": round(float(np.mean(final_stabilities)), 3),
            "p10_final_stability": round(float(np.percentile(final_stabilities, 10)), 3),
            "worst_case_stability": round(float(np.min(min_stabilities)), 3),
            "collapse_probability": round(collapse_events / n_simulations, 4),
            "timeline": timeline,
        })

    def domain_radar(self, domain_scores: dict[str, float]) -> dict:
        radar = []
        for domain, score in sorted(domain_scores.items(), key=lambda x: x[1]):
            w = DOMAIN_WEIGHTS.get(domain, 0.02)
            danger = max(0, 1.0 - score)
            radar.append({
                "domain": domain,
                "score": round(score, 3),
                "danger": round(danger, 3),
                "weight": w,
                "weighted_danger": round(danger * w, 4),
            })
        total_weighted_danger = sum(r["weighted_danger"] for r in radar)
        return sanitize_nan({
            "domains": radar,
            "total_weighted_danger": round(total_weighted_danger, 3),
            "most_dangerous_domain": min(radar, key=lambda x: x["score"])["domain"] if radar else "none",
            "safest_domain": max(radar, key=lambda x: x["score"])["domain"] if radar else "none",
        })


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