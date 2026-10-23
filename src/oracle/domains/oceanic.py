from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class OceanicState:
    enso_index: float = 0.5
    pdo_index: float = 0.2
    amo_index: float = 0.1
    sea_surface_temp_anomaly_c: float = 0.8
    ocean_heat_content_700m: float = 0.5
    thermohaline_circulation_strength: float = 0.7
    arctic_sea_ice_extent_mkm2: float = 4.5
    atlantic_hurricane_sst_threshold: float = 26.5
    coral_bleaching_alert: float = 0.3
    fisheries_productivity_index: float = 0.6
    sea_level_trend_mm_yr: float = 3.4
    ocean_acidification_ph: float = 8.07


class OceanicModel:

    def __init__(self):
        self.name = "oceanic_model"

    def set_state(self, **kwargs) -> "OceanicModel":
        current = OceanicState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, float(v))
        self.state = current
        return self

    def assess_ocean_state(self, state: Optional[OceanicState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", OceanicState())
        enso_phase = "elnino" if state.enso_index > 0.5 else "lanina" if state.enso_index < -0.5 else "neutral"
        enso_strength = abs(state.enso_index)
        hurricane_fuel = max(0.0, (state.sea_surface_temp_anomaly_c + state.atlantic_hurricane_sst_threshold - 26.0) / 5.0)
        hurricane_fuel = min(1.0, hurricane_fuel)
        if state.enso_index > 0.5:
            hurricane_suppression = 0.8
        elif state.enso_index < -0.5:
            hurricane_suppression = 1.3
        else:
            hurricane_suppression = 1.0
        expected_hurricanes = hurricane_fuel * 15.0 * hurricane_suppression
        thermohaline_risk = 0.0
        if state.thermohaline_circulation_strength < 0.5:
            thermohaline_risk = (0.5 - state.thermohaline_circulation_strength) * 2.0
        ice_risk = 0.0
        if state.arctic_sea_ice_extent_mkm2 < 3.0:
            ice_risk = (3.0 - state.arctic_sea_ice_extent_mkm2) / 3.0
        coral_risk = state.coral_bleaching_alert
        fisheries_stress = 1.0 - state.fisheries_productivity_index
        composite = (
            enso_strength * 0.20
            + hurricane_fuel * 0.20
            + thermohaline_risk * 0.20
            + ice_risk * 0.15
            + coral_risk * 0.10
            + fisheries_stress * 0.15
        )
        composite = min(1.0, composite)
        if composite > 0.6:
            level = "critical"
        elif composite > 0.4:
            level = "elevated"
        elif composite > 0.2:
            level = "moderate"
        else:
            level = "stable"
        recommendations = []
        if enso_strength > 0.8:
            ename = "El Nino" if state.enso_index > 0 else "La Nina"
            recommendations.append(f"Strong {ename} conditions — prepare for global teleconnection impacts on weather, agriculture, and fisheries")
        if expected_hurricanes > 15:
            recommendations.append(f"Active hurricane season expected (~{expected_hurricanes:.0f} storms) — coastal infrastructure preparedness critical")
        if thermohaline_risk > 0.3:
            recommendations.append("Thermohaline circulation weakening — potential European climate shift and tropical rain belt migration")
        if state.arctic_sea_ice_extent_mkm2 < 2.0:
            recommendations.append("Arctic ice-free summer approaching — accelerate Arctic shipping governance and polar bear conservation")
        if state.ocean_acidification_ph < 8.0:
            recommendations.append("Ocean acidification below threshold — coral and shellfish ecosystems at critical risk")
        return sanitize_nan({
            "composite_ocean_risk": round(float(composite), 4),
            "risk_level": level,
            "enso_phase": enso_phase,
            "enso_strength": round(float(enso_strength), 4),
            "expected_hurricane_season": round(float(expected_hurricanes), 1),
            "thermohaline_circulation_risk": round(float(thermohaline_risk), 4),
            "arctic_ice_risk": round(float(ice_risk), 4),
            "coral_bleaching_risk": round(float(coral_risk), 4),
            "fisheries_stress": round(float(fisheries_stress), 4),
            "ocean_acidification_ph": round(float(state.ocean_acidification_ph), 3),
            "recommendations": recommendations[:5],
        })

    def simulate_enso_transition(self, current_phase: str = "elnino", months: int = 24) -> dict:
        if current_phase == "elnino":
            start_index = 1.5
        elif current_phase == "lanina":
            start_index = -1.5
        else:
            start_index = 0.2
        timeline = []
        idx = start_index
        for m in range(months):
            decay = -idx * 0.08
            noise = np.random.normal(0, 0.1) * 0.5
            idx = idx + decay + noise
            idx = np.clip(idx, -2.5, 2.5)
            phase = "elnino" if idx > 0.5 else "lanina" if idx < -0.5 else "neutral"
            timeline.append({
                "month": m + 1,
                "nino34_index": round(float(idx), 3),
                "phase": phase,
            })
        final = timeline[-1]["phase"]
        return sanitize_nan({
            "start_phase": current_phase,
            "final_phase": final,
            "timeline": timeline,
        })

    def predict(self, years_ahead: int = 5, state: Optional[OceanicState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", OceanicState())
        future_sst = state.sea_surface_temp_anomaly_c + 0.02 * years_ahead
        future_ice = max(0.0, state.arctic_sea_ice_extent_mkm2 - 0.08 * years_ahead)
        future_ph = state.ocean_acidification_ph - 0.002 * years_ahead
        future_thermohaline = max(0.1, state.thermohaline_circulation_strength - 0.02 * years_ahead)
        future_sea_level = state.sea_level_trend_mm_yr * years_ahead + 0.5 * years_ahead
        el_nino_probability_next_year = 0.35
        if abs(state.enso_index) > 1.0:
            el_nino_probability_next_year = 0.6 if state.enso_index > 0 else 0.15
        amoc_collapse_risk = max(0.0, (0.5 - future_thermohaline) * 2.0) if future_thermohaline < 0.5 else 0.0
        return sanitize_nan({
            "future_sst_anomaly_c": round(float(future_sst), 2),
            "future_arctic_ice_mkm2": round(float(future_ice), 2),
            "future_ocean_ph": round(float(future_ph), 3),
            "future_thermohaline": round(float(future_thermohaline), 4),
            "cumulative_sea_level_rise_mm": round(float(future_sea_level), 1),
            "el_nino_probability_next_year": round(float(el_nino_probability_next_year), 4),
            "amoc_collapse_risk": round(float(amoc_collapse_risk), 4),
            "ice_free_summer_probability": round(float(min(1.0, max(0.0, (5.0 - future_ice) / 5.0))), 4),
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