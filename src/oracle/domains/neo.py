from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class NearEarthObjectState:
    neo_count_tracked: float = 34000.0
    neo_count_hazardous: float = 2300.0
    avg_close_approach_distance_ld: float = 15.0
    largest_approach_diameter_m: float = 50.0
    detection_rate_pct: float = 0.60
    torino_scale_max: float = 0.0
    cumulative_impact_probability_100yr: float = 0.0002
    kinetic_energy_typical_mt: float = 10.0
    deflection_readiness: float = 0.05
    asteroid_mining_interest: float = 0.1


class NearEarthObjectModel:

    def __init__(self):
        self.name = "neo_model"

    def set_state(self, **kwargs) -> "NearEarthObjectModel":
        current = NearEarthObjectState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, float(v))
        self.state = current
        return self

    def assess_impact_risk(self, state: Optional[NearEarthObjectState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", NearEarthObjectState())
        detection_gap = 1.0 - state.detection_rate_pct
        undetected_hazardous = state.neo_count_hazardous * detection_gap
        closest_approach_risk = 1.0 / max(0.1, state.avg_close_approach_distance_ld)
        torino_risk = min(1.0, state.torino_scale_max / 10.0)
        cumulative_risk = state.cumulative_impact_probability_100yr
        city_killer_probability_50yr = cumulative_risk * 0.5
        global_catastrophe_probability_100yr = cumulative_risk * 0.1
        composite = (
            torino_risk * 0.30
            + detection_gap * 0.25
            + closest_approach_risk * 0.15
            + cumulative_risk * 100 * 0.20
            + (1.0 - state.deflection_readiness) * 0.10
        )
        composite = min(1.0, max(0.0, composite))
        if composite > 0.5:
            level = "critical"
        elif composite > 0.3:
            level = "elevated"
        elif composite > 0.1:
            level = "moderate"
        else:
            level = "low"
        event_types = []
        if state.kinetic_energy_typical_mt > 100:
            event_types.append({"type": "regional_devastation", "diameter_km_est": round(float(state.largest_approach_diameter_m / 1000), 2), "return_period_years": 700000})
        if state.kinetic_energy_typical_mt > 100000:
            event_types.append({"type": "global_mass_extinction", "diameter_km_est": round(float(state.largest_approach_diameter_m / 1000), 2), "return_period_years": 100000000})
        event_types.append({"type": "city_killer", "diameter_m_est": 40, "return_period_years": 500})
        event_types.append({"type": "tunguska_class", "diameter_m_est": 60, "return_period_years": 1000})
        recommendations = []
        if detection_gap > 0.4:
            recommendations.append("Expand NEO survey — significant asteroid population remains undetected")
        if state.deflection_readiness < 0.2:
            recommendations.append("Invest in deflection technology development (kinetic impactor, gravity tractor, nuclear standoff)")
        if torino_risk > 0.1:
            recommendations.append("Object on Torino scale — detailed orbital analysis and deflection planning warranted")
        if state.asteroid_mining_interest > 0.2:
            recommendations.append("Near-Earth asteroids offer mining opportunities that fund deflection infrastructure")
        return sanitize_nan({
            "composite_risk": round(float(composite), 4),
            "risk_level": level,
            "detection_gap_pct": round(float(detection_gap * 100), 1),
            "undetected_hazardous_estimate": round(float(undetected_hazardous), 0),
            "torino_scale_current": round(float(state.torino_scale_max), 1),
            "city_killer_probability_50yr": round(float(city_killer_probability_50yr), 6),
            "global_catastrophe_probability_100yr": round(float(global_catastrophe_probability_100yr), 7),
            "deflection_readiness": round(float(state.deflection_readiness), 4),
            "potential_impact_events": event_types,
            "recommendations": recommendations[:4],
        })

    def simulate_impact_scenario(self, diameter_m: float = 200.0, velocity_kms: float = 20.0,
                                   density_kgm3: float = 3000.0, land_impact: bool = True) -> dict:
        radius = diameter_m / 2.0
        volume = (4.0 / 3.0) * np.pi * radius ** 3
        mass_kg = volume * density_kgm3
        velocity_ms = velocity_kms * 1000.0
        energy_j = 0.5 * mass_kg * velocity_ms ** 2
        energy_mt = energy_j / 4.184e18
        crater_diameter_km = 0.0
        if land_impact:
            crater_diameter_km = 1.8 * (energy_mt ** 0.3) * (density_kgm3 / 2500.0) ** (1.0 / 3.0)
        else:
            crater_diameter_km = 0.0
        if diameter_m < 50:
            event_class = "airburst"
            description = "High-altitude airburst causing limited damage"
        elif diameter_m < 200:
            event_class = "local_devastation"
            description = "Local/regional devastation, Tunguska-class"
        elif diameter_m < 2000:
            event_class = "regional_catastrophe"
            description = "Regional catastrophe, significant climate effects"
        else:
            event_class = "global_mass_extinction"
            description = "Global mass extinction event, civilization-threatening"
        tsunami_height_m = 0.0
        if not land_impact and diameter_m > 100:
            tsunami_height_m = min(300, diameter_m * 0.3 * velocity_kms / 20.0)
        nuclear_equivalent = energy_mt
        earthquake_equivalent = energy_mt * 80000.0 / 1.0
        return sanitize_nan({
            "diameter_m": diameter_m,
            "velocity_kms": velocity_kms,
            "mass_kg": round(float(mass_kg), 0),
            "kinetic_energy_mt_tnt": round(float(energy_mt), 2),
            "event_class": event_class,
            "description": description,
            "crater_diameter_km": round(float(crater_diameter_km), 2),
            "tsunami_height_m": round(float(tsunami_height_m), 1),
            "nuclear_bomb_equivalent": round(float(nuclear_equivalent), 1),
            "earthquake_magnitude_equivalent": round(float(min(10.0, np.log10(max(1, earthquake_equivalent)) * 1.5 + 3.0)), 1),
            "land_impact": land_impact,
        })

    def predict(self, years_ahead: int = 50, state: Optional[NearEarthObjectState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", NearEarthObjectState())
        detection_improvement = min(0.95, state.detection_rate_pct + 0.01 * years_ahead)
        future_deflection = min(1.0, state.deflection_readiness + 0.005 * years_ahead)
        impact_prob = 1.0 - (1.0 - state.cumulative_impact_probability_100yr) ** (years_ahead / 100.0)
        city_killer_in_period = 1.0 - np.exp(-years_ahead / 500.0)
        return sanitize_nan({
            "years_ahead": years_ahead,
            "future_detection_rate": round(float(detection_improvement), 4),
            "future_deflection_readiness": round(float(future_deflection), 4),
            "impact_probability_period": round(float(impact_prob), 6),
            "city_killer_probability_period": round(float(city_killer_in_period), 4),
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