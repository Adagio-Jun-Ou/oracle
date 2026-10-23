from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class EcologicalState:
    biodiversity_index: float = 0.6
    species_richness_per_km2: float = 50.0
    trophic_integrity: float = 0.5
    keystone_species_ratio: float = 0.3
    ecosystem_connectivity: float = 0.5
    habitat_fragmentation: float = 0.4
    invasive_species_impact: float = 0.3
    pollution_load: float = 0.3
    climate_stress: float = 0.4
    overexploitation_rate: float = 0.3
    adaptive_capacity: float = 0.4
    genetic_diversity_index: float = 0.5
    evolutionary_pressure: float = 0.3
    extinction_debt: float = 0.4


class EcologicalModel:

    def __init__(self):
        self.name = "ecological_model"

    def set_state(self, **kwargs) -> "EcologicalModel":
        current = EcologicalState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, float(v))
        self.state = current
        return self

    def assess_ecosystem(self, state: Optional[EcologicalState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", EcologicalState())

        trophic_health = state.trophic_integrity * 0.3 + state.keystone_species_ratio * 0.3 + state.ecosystem_connectivity * 0.2 + state.adaptive_capacity * 0.2
        trophic_health = min(1.0, max(0.0, trophic_health))

        habitat_quality = max(0.0, 1.0 - state.habitat_fragmentation) * 0.3 + state.biodiversity_index * 0.4 + state.genetic_diversity_index * 0.15 + (1.0 - state.invasive_species_impact) * 0.15
        habitat_quality = min(1.0, max(0.0, habitat_quality))

        resilience = (state.adaptive_capacity * 0.3 + state.genetic_diversity_index * 0.25 + min(1.0, 1.0 - state.extinction_debt) * 0.25 + trophic_health * 0.2)
        resilience = min(1.0, max(0.0, resilience))

        stress_score = (state.climate_stress * 0.3 + state.pollution_load * 0.25 + state.overexploitation_rate * 0.2 + state.invasive_species_impact * 0.15 + state.evolutionary_pressure * 0.1)
        stress_score = min(1.0, max(0.0, stress_score))

        composite = trophic_health * 0.25 + habitat_quality * 0.25 + resilience * 0.3 + (1.0 - stress_score) * 0.2
        composite = min(1.0, max(0.0, composite))

        if composite > 0.7:
            level = "thriving"
        elif composite > 0.5:
            level = "stable"
        elif composite > 0.3:
            level = "stressed"
        else:
            level = "critical"

        keystone_risk = max(0.0, 1.0 - state.keystone_species_ratio) * (1.0 + state.habitat_fragmentation * 0.5)
        keystone_risk = min(1.0, keystone_risk)

        cascade_potential = state.trophic_integrity * 0.0 + state.habitat_fragmentation * 0.4 + state.invasive_species_impact * 0.3 + state.extinction_debt * 0.3
        cascade_potential = min(1.0, max(0.0, cascade_potential))

        adaptations = min(1.0, state.adaptive_capacity * state.genetic_diversity_index * (1.0 + 0.2 * state.evolutionary_pressure))
        adaptation_rate = adaptations * 0.05

        threshold_extinctions = []
        if state.keystone_species_ratio < 0.15:
            threshold_extinctions.append({"ecosystem": "apex_predator_collapse", "probability": keystone_risk, "impact": "high"})
        if state.habitat_fragmentation > 0.6:
            threshold_extinctions.append({"ecosystem": "forested_fragmentation", "probability": state.habitat_fragmentation * 0.7, "impact": "medium"})
        if state.climate_stress > 0.7:
            threshold_extinctions.append({"ecosystem": "coral_reef_bleaching", "probability": state.climate_stress * 0.8, "impact": "high"})
        if state.invasive_species_impact > 0.5:
            threshold_extinctions.append({"ecosystem": "native_species_displacement", "probability": state.invasive_species_impact * 0.6, "impact": "medium"})

        recommendations = []
        if trophic_health < 0.4:
            recommendations.append("Protect keystone species and restore top predator populations")
        if habitat_quality < 0.4:
            recommendations.append("Create wildlife corridors to reduce habitat fragmentation")
        if resilience < 0.4:
            recommendations.append("Enhance genetic diversity through conservation breeding programs")
        if stress_score > 0.6:
            recommendations.append("Reduce pollution load and overexploitation pressure")
        if cascade_potential > 0.5:
            recommendations.append("Monitor trophic cascades — ecosystem collapse risk elevated")
        if state.evolutionary_pressure > 0.5:
            recommendations.append("Accelerated adaptation support — maintain genetic variation reservoirs")
        if len(recommendations) > 5:
            recommendations = recommendations[:5]

        return sanitize_nan({
            "ecosystem_health_index": composite,
            "health_level": level,
            "trophic_integrity": trophic_health,
            "habitat_quality": habitat_quality,
            "resilience_score": resilience,
            "stress_score": stress_score,
            "keystone_risk": keystone_risk,
            "cascade_potential": cascade_potential,
            "adaptation_rate": adaptation_rate,
            "threshold_extinctions": threshold_extinctions,
            "recommendations": recommendations,
        })

    def simulate_trophic_cascade(self, keystone_removed: str = "apex_predator", severity: float = 0.8, years: int = 10) -> dict:
        cascade_effects = {
            "apex_predator": {
                "mesopredator_release": severity * 0.7,
                "herbivore_decline": severity * 0.5,
                "vegetation_change": severity * 0.6,
                "biodiversity_loss": severity * 0.4,
            },
            "pollinator": {
                "plant_reproduction_decline": severity * 0.8,
                "crop_yield_impact": severity * 0.6,
                "bird_population_decline": severity * 0.3,
                "biodiversity_loss": severity * 0.5,
            },
            "coral": {
                "fish_habitat_loss": severity * 0.8,
                "coastal_protection_decline": severity * 0.7,
                "tourism_loss_billion": severity * 5.0,
                "biodiversity_loss": severity * 0.7,
            },
            "forest_canopy": {
                "understory_growth": severity * 0.6,
                "carbon_sequestration_decline": severity * 0.5,
                "watershed_impact": severity * 0.4,
                "biodiversity_loss": severity * 0.3,
            },
        }
        effects = cascade_effects.get(keystone_removed, cascade_effects["apex_predator"])

        timeline = []
        bdi = 0.6
        for y in range(1, years + 1):
            drop = effects["biodiversity_loss"] * (1.0 - np.exp(-0.3 * y))
            bdi_y = max(0.05, bdi * (1.0 - drop * 0.5))
            timeline.append({
                "year": y,
                "biodiversity_index": bdi_y,
                "species_lost_pct": drop * 30.0,
                "ecosystem_stability": max(0.0, 1.0 - drop * 0.8),
                "recovery_potential": max(0.0, 0.3 * (1.0 - severity) * np.exp(-0.05 * y)),
            })

        recovery_years = int(15 + (severity * 20))
        if keystone_removed in ("coral",):
            recovery_years = int(30 + (severity * 40))

        return sanitize_nan({
            "scenario": f"trophic_cascade_{keystone_removed}",
            "keystone_removed": keystone_removed,
            "severity": severity,
            "effects": effects,
            "timeline": timeline,
            "projected_biodiversity_loss_pct": effects["biodiversity_loss"] * 30.0,
            "recovery_years": recovery_years,
            "recovery_possible": severity < 0.7,
        })

    def predict(self, years_ahead: int = 10, state: Optional[EcologicalState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", EcologicalState())

        climate_pressure_rate = state.climate_stress * 0.02
        biodiversity_decline_rate = state.extinction_debt * 0.01 + state.habitat_fragmentation * 0.005

        future_biodiversity = max(0.05, state.biodiversity_index - biodiversity_decline_rate * years_ahead)
        future_genetic = max(0.05, state.genetic_diversity_index - 0.005 * years_ahead + state.adaptive_capacity * 0.003 * years_ahead)
        future_trophic = max(0.05, state.trophic_integrity - 0.01 * years_ahead * state.habitat_fragmentation)

        adaptation_gain = state.adaptive_capacity * state.genetic_diversity_index * 0.01 * years_ahead
        future_adaptive = min(1.0, state.adaptive_capacity + adaptation_gain * 0.3)

        extinction_rate = state.extinction_debt * (1.0 + climate_pressure_rate * years_ahead) * 0.001
        cumulative_extinctions_pct = (1.0 - (1.0 - extinction_rate) ** years_ahead) * 100.0

        if state.evolutionary_pressure > 0.5:
            future_adaptive += 0.05 * years_ahead * 0.1

        regime = "adaptive" if future_adaptive > 0.5 else ("stressed" if future_biodiversity > 0.3 else "collapse")

        return sanitize_nan({
            "years_ahead": years_ahead,
            "future_biodiversity_index": future_biodiversity,
            "future_genetic_diversity": future_genetic,
            "future_trophic_integrity": future_trophic,
            "future_adaptive_capacity": min(1.0, future_adaptive),
            "projected_extinction_rate": extinction_rate,
            "cumulative_extinctions_pct": cumulative_extinctions_pct,
            "ecosystem_regime": regime,
            "tipping_point_risk": min(1.0, (1.0 - future_adaptive) * (1.0 - future_biodiversity)),
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