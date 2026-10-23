from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SpaceState:
    orbital_debris_count: float = 25000.0
    active_satellites: float = 7500.0
    space_budget_billion: float = 80.0
    launch_frequency_per_year: float = 200.0
    kessler_syndrome_risk: float = 0.15
    space_governance_index: float = 0.35
    commercial_space_revenue_billion: float = 400.0
    space_military_index: float = 0.4
    broadband_satellite_share: float = 0.08
    asteroid_mining_readiness: float = 0.05
    space_station_occupancy: float = 0.7
    solar_storm_risk: float = 0.2


class SpaceModel:

    def __init__(self):
        self.name = "space_model"

    def set_state(self, **kwargs) -> "SpaceModel":
        current = SpaceState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, float(v))
        self.state = current
        return self

    def assess_orbital_environment(self, state: Optional[SpaceState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", SpaceState())
        density = state.active_satellites / 1000.0
        debris_density = state.orbital_debris_count / 25000.0
        collision_prob = 1.0 - np.exp(-0.1 * debris_density * density)
        kessler_threshold = state.kessler_syndrome_risk
        if kessler_threshold > 0.5:
            kessler_status = "imminent"
        elif kessler_threshold > 0.3:
            kessler_status = "elevated"
        elif kessler_threshold > 0.15:
            kessler_status = "moderate"
        else:
            kessler_status = "low"
        debris_growth = state.launch_frequency_per_year * 0.02 - state.orbital_debris_count * 0.001
        sustainable_debris = debris_growth < 0
        orbital_decay = min(1.0, state.orbital_debris_count / 50000.0)
        leo_congestion = min(1.0, state.active_satellites / 15000.0)
        recommendations = []
        if kessler_threshold > 0.3:
            recommendations.append("Prioritize active debris removal programs")
        if leo_congestion > 0.5:
            recommendations.append("Implement LEO traffic management and coordination")
        if state.space_governance_index < 0.4:
            recommendations.append("Strengthen international space governance frameworks")
        if state.space_military_index > 0.4:
            recommendations.append("Establish norms against weaponization of space")
        if not sustainable_debris:
            recommendations.append("Accelerate debris mitigation guidelines compliance")
        if collision_prob > 0.3:
            recommendations.append("Invest in automated collision avoidance systems")
        return sanitize_nan({
            "collision_probability": round(float(collision_prob), 4),
            "debris_density_metric": round(float(debris_density), 4),
            "kessler_syndrome_risk": round(float(kessler_threshold), 4),
            "kessler_status": kessler_status,
            "orbital_environment": "critical" if orbital_decay > 0.7 else "degraded" if orbital_decay > 0.4 else "stable",
            "leo_congestion": round(float(leo_congestion), 4),
            "debris_growth_trajectory": round(float(debris_growth), 4),
            "sustainable_orbit": sustainable_debris,
            "recommendations": recommendations[:5],
        })

    def simulate_kessler_scenario(self, scenario: str = "moderate", state: Optional[SpaceState] = None, years: int = 30) -> dict:
        if state is None:
            state = getattr(self, "state", SpaceState())
        scenarios = {
            "business_as_usual": {"launch_growth": 1.08, "debris_multiplier": 1.05, "mitigation": 0.01, "active_removal": 0.0},
            "moderate": {"launch_growth": 1.12, "debris_multiplier": 1.08, "mitigation": 0.02, "active_removal": 0.0},
            "mega_constellation": {"launch_growth": 1.25, "debris_multiplier": 1.12, "mitigation": 0.01, "active_removal": 0.0},
            "active_removal": {"launch_growth": 1.08, "debris_multiplier": 1.03, "mitigation": 0.05, "active_removal": 50.0},
            "kessler_trigger": {"launch_growth": 1.05, "debris_multiplier": 1.30, "mitigation": 0.01, "active_removal": 0.0},
        }
        params = scenarios.get(scenario, scenarios["moderate"])
        debris = float(state.orbital_debris_count)
        satellites = float(state.active_satellites)
        timeline = []
        kessler_triggered = False
        for year in range(years):
            growth = params["launch_growth"] ** year
            satellites = satellites * growth
            new_debris = debris * (params["debris_multiplier"] - 1) + satellites * 0.005
            removed = params["active_removal"] + debris * params["mitigation"]
            debris = debris + new_debris - removed
            if not kessler_triggered and debris > 50000:
                kessler_triggered = True
            timeline.append({
                "year": year + 1,
                "debris_count": round(debris, 0),
                "active_satellites": round(satellites, 0),
                "collision_rate_per_satellite": round(float(1.0 - np.exp(-debris / 250000.0)), 6),
            })
        return sanitize_nan({
            "scenario": scenario,
            "kessler_triggered": kessler_triggered,
            "final_debris_count": round(debris, 0),
            "final_active_satellites": round(satellites, 0),
            "peak_collision_rate": max(t["collision_rate_per_satellite"] for t in timeline),
            "timeline": timeline,
        })

    def predict(self, years_ahead: int = 10, state: Optional[SpaceState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", SpaceState())
        satellite_growth = 1.10 ** years_ahead
        debris_growth_factor = 1.05 ** years_ahead - state.orbital_debris_count * 0.005 * years_ahead
        future_debris = state.orbital_debris_count * debris_growth_factor
        future_satellites = state.active_satellites * satellite_growth
        future_kessler = min(1.0, state.kessler_syndrome_risk + 0.02 * years_ahead)
        future_governance = min(1.0, state.space_governance_index + 0.005 * years_ahead)
        future_revenue = state.commercial_space_revenue_billion * (1.08 ** years_ahead)
        future_broadband = min(0.5, state.broadband_satellite_share + 0.02 * years_ahead)
        future_mining = min(1.0, state.asteroid_mining_readiness + 0.01 * years_ahead)
        future_military = min(1.0, state.space_military_index + 0.01 * years_ahead)
        return sanitize_nan({
            "future_debris_count": round(float(future_debris), 0),
            "future_active_satellites": round(float(future_satellites), 0),
            "future_kessler_risk": round(float(future_kessler), 4),
            "future_governance_index": round(float(future_governance), 4),
            "future_commercial_revenue_billion": round(float(future_revenue), 1),
            "future_broadband_share": round(float(future_broadband), 4),
            "future_asteroid_mining_readiness": round(float(future_mining), 4),
            "future_military_index": round(float(future_military), 4),
            "space_economy_growth_pct": round(float((1.08 ** years_ahead - 1) * 100), 1),
            "orbital_sustainability_trend": "improving" if future_kessler < state.kessler_syndrome_risk else "degrading",
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