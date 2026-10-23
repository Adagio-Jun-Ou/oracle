from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SeismicState:
    earthquake_rate_monthly: float = 12.0
    avg_magnitude: float = 5.2
    max_magnitude_30d: float = 6.0
    seismic_energy_release_30d: float = 1e16
    depth_avg_km: float = 35.0
    fault_stress_accumulation: float = 0.5
    volcanic_activity_index: float = 0.3
    gps_strain_rate: float = 0.02
    radon_anomaly: float = 0.2
    electromagnetic_anomaly: float = 0.15
    swarm_activity: float = 0.2
    subduction_zone_stress: float = 0.5
    transform_fault_slip_rate: float = 0.02


class SeismicModel:

    def __init__(self):
        self.name = "seismic_model"

    def set_state(self, **kwargs) -> "SeismicModel":
        current = SeismicState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, float(v))
        self.state = current
        return self

    def assess_seismic_hazard(self, state: Optional[SeismicState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", SeismicState())
        energy_score = min(1.0, np.log10(max(1, state.seismic_energy_release_30d)) / 18.0)
        stress_score = state.fault_stress_accumulation * 0.35 + state.subduction_zone_stress * 0.30 + state.swarm_activity * 0.20 + state.gps_strain_rate * 5.0 * 0.15
        stress_score = min(1.0, stress_score)
        precursor_score = state.radon_anomaly * 0.3 + state.electromagnetic_anomaly * 0.3 + state.swarm_activity * 0.25 + (1.0 - state.depth_avg_km / 100.0) * 0.15
        precursor_score = min(1.0, max(0.0, precursor_score))
        composite_hazard = energy_score * 0.35 + stress_score * 0.35 + precursor_score * 0.30
        if composite_hazard > 0.7:
            hazard_level = "critical"
        elif composite_hazard > 0.5:
            hazard_level = "high"
        elif composite_hazard > 0.3:
            hazard_level = "moderate"
        else:
            hazard_level = "low"
        m7_probability_30d = 1.0 - np.exp(-0.03 * composite_hazard * 3)
        m8_probability_365d = 1.0 - np.exp(-0.005 * composite_hazard * 5)
        volcanic_risk = state.volcanic_activity_index > 0.5
        recommendations = []
        if composite_hazard > 0.5:
            recommendations.append("Elevated seismic hazard — review building codes and emergency response plans")
        if state.fault_stress_accumulation > 0.6:
            recommendations.append("High fault stress accumulation — increased earthquake probability on locked faults")
        if state.swarm_activity > 0.4:
            recommendations.append("Earthquake swarm activity detected — monitor for foreshock sequences")
        if state.volcanic_activity_index > 0.5:
            recommendations.append("Volcanic activity elevated — monitor for eruption precursors (gas, deformation, tremor)")
        if state.radon_anomaly > 0.4 or state.electromagnetic_anomaly > 0.4:
            recommendations.append("Geochemical/electromagnetic precursors detected — potential earthquake preparation zone")
        return sanitize_nan({
            "composite_hazard": round(float(composite_hazard), 4),
            "hazard_level": hazard_level,
            "energy_release_score": round(float(energy_score), 4),
            "fault_stress_score": round(float(stress_score), 4),
            "precursor_score": round(float(precursor_score), 4),
            "m7_probability_30d": round(float(m7_probability_30d), 4),
            "m8_probability_365d": round(float(m8_probability_365d), 6),
            "volcanic_risk": volcanic_risk,
            "volcanic_activity_index": round(float(state.volcanic_activity_index), 4),
            "recommendations": recommendations[:5],
        })

    def simulate_earthquake_scenario(self, magnitude: float = 7.5, depth_km: float = 15.0,
                                      population_exposed_m: float = 5.0, building_quality: float = 0.5) -> dict:
        intensity = max(4.0, magnitude * 1.5 - depth_km * 0.02)
        pga = 10.0 ** (0.3 * intensity - 1.5)
        damage_ratio = min(0.9, pga * (1.0 - building_quality * 0.7) * 0.15)
        fatalities = population_exposed_m * 1e6 * damage_ratio * 0.01 * (1.0 - building_quality * 0.5)
        fatalities = min(fatalities, population_exposed_m * 1e6 * 0.1)
        economic_loss_billion = population_exposed_m * damage_ratio * 50.0 * (1.0 + magnitude / 10.0)
        displacement_m = population_exposed_m * damage_ratio * 0.3
        infrastructure_damage_pct = min(1.0, damage_ratio * 1.5)
        tsunami_probability = 0.0
        if depth_km < 70 and magnitude >= 7.0:
            tsunami_probability = min(1.0, (magnitude - 6.5) * 0.2 * (1.0 - depth_km / 200.0))
        secondary_effects = []
        if magnitude > 6.5:
            secondary_effects.append("landslides")
        if tsunami_probability > 0.1:
            secondary_effects.append("tsunami")
        if damage_ratio > 0.2:
            secondary_effects.append("infrastructure_cascade")
        if magnitude > 7.0:
            secondary_effects.append("aftershock_sequence")
        recovery_years = max(1.0, damage_ratio * 10.0 * (1.0 - building_quality * 0.5))
        return sanitize_nan({
            "magnitude": magnitude,
            "depth_km": depth_km,
            "estimated_intensity_mmi": round(float(intensity), 1),
            "pga_g": round(float(pga), 4),
            "damage_ratio": round(float(damage_ratio), 4),
            "estimated_fatalities": round(float(fatalities), 0),
            "economic_loss_billion_usd": round(float(economic_loss_billion), 1),
            "displacement_millions": round(float(displacement_m), 2),
            "infrastructure_damage_pct": round(float(infrastructure_damage_pct), 4),
            "tsunami_probability": round(float(tsunami_probability), 4),
            "secondary_effects": secondary_effects,
            "recovery_years": round(float(recovery_years), 1),
        })

    def predict(self, years_ahead: int = 5, state: Optional[SeismicState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", SeismicState())
        stress_growth = state.fault_stress_accumulation + 0.02 * years_ahead
        stress_growth = min(1.0, stress_growth)
        m6_expected_count = state.earthquake_rate_monthly * 12 * years_ahead * (1.0 + stress_growth * 0.3) * 0.08
        m7_expected_count = m6_expected_count * 0.1
        m8_probability = 1.0 - np.exp(-0.003 * years_ahead * stress_growth * 5)
        future_volcanic = min(1.0, state.volcanic_activity_index + 0.01 * years_ahead * 0.5)
        return sanitize_nan({
            "future_fault_stress": round(float(stress_growth), 4),
            "expected_m6_count": round(float(m6_expected_count), 1),
            "expected_m7_count": round(float(m7_expected_count), 2),
            "m8_probability_5yr": round(float(m8_probability), 4),
            "future_volcanic_index": round(float(future_volcanic), 4),
            "seismic_regime": "heightened" if stress_growth > 0.7 else "elevated" if stress_growth > 0.5 else "background",
        })


def sanitize_nan(obj):
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj