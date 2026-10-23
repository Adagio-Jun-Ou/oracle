from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SolarState:
    sunspot_number: float = 80.0
    solar_flux_10_7cm: float = 130.0
    flare_probability: float = 0.3
    cme_probability: float = 0.2
    solar_wind_speed_kms: float = 400.0
    interplanetary_magnetic_field_bt: float = 5.0
    kp_index: float = 3.0
    dst_index: float = -20.0
    proton_flux: float = 0.1
    aurora_latitude: float = 65.0
    solar_cycle_phase: float = 0.6
    f10_7_variance: float = 20.0


class SolarModel:

    def __init__(self):
        self.name = "solar_model"

    def set_state(self, **kwargs) -> "SolarModel":
        current = SolarState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, float(v))
        self.state = current
        return self

    def assess_solar_weather(self, state: Optional[SolarState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", SolarState())
        x_flare_prob = state.flare_probability * (state.sunspot_number / 200.0)
        x_flare_prob = min(1.0, max(0.0, x_flare_prob))
        cme_impact_prob = state.cme_probability * 0.3
        geomagnetic_storm = min(1.0, max(0.0, (state.kp_index / 9.0)))
        radiation_storm = min(1.0, state.proton_flux * 10.0)
        grid_risk = 0.0
        if state.dst_index < -300:
            grid_risk = 0.7
        elif state.dst_index < -200:
            grid_risk = 0.4
        elif state.dst_index < -100:
            grid_risk = 0.15
        elif state.dst_index < -50:
            grid_risk = 0.05
        satellite_risk = min(1.0, (state.proton_flux * 5.0 + (state.kp_index / 9.0) * 0.3))
        gps_degradation = min(1.0, state.kp_index / 9.0 * 0.8)
        composite_risk = x_flare_prob * 0.25 + cme_impact_prob * 0.25 + geomagnetic_storm * 0.20 + radiation_storm * 0.15 + grid_risk * 0.15
        if composite_risk > 0.6:
            level = "extreme"
        elif composite_risk > 0.4:
            level = "high"
        elif composite_risk > 0.2:
            level = "moderate"
        else:
            level = "quiet"
        recommendations = []
        if grid_risk > 0.3:
            recommendations.append("Carrington-class event risk — prepare grid infrastructure for geomagnetic induced currents")
        if satellite_risk > 0.3:
            recommendations.append("Radiation environment hazardous — satellite operators should safe-mode sensitive electronics")
        if x_flare_prob > 0.3:
            recommendations.append("Elevated X-flare probability — increased HF radio blackout risk and radiation exposure")
        if gps_degradation > 0.3:
            recommendations.append("GPS degradation likely — aviation and precision navigation should prepare fallbacks")
        if state.kp_index > 6:
            recommendations.append("G3+ geomagnetic storm — aurora visible at mid-latitudes, power grid monitoring advised")
        return sanitize_nan({
            "composite_risk": round(float(composite_risk), 4),
            "risk_level": level,
            "x_flare_probability": round(float(x_flare_prob), 4),
            "cme_impact_probability": round(float(cme_impact_prob), 4),
            "geomagnetic_storm_intensity": round(float(geomagnetic_storm), 4),
            "radiation_storm_level": round(float(radiation_storm), 4),
            "grid_infrastructure_risk": round(float(grid_risk), 4),
            "satellite_risk": round(float(satellite_risk), 4),
            "gps_degradation": round(float(gps_degradation), 4),
            "aurora_latitude": round(float(state.aurora_latitude), 1),
            "recommendations": recommendations[:5],
        })

    def simulate_carrington_event(self, grid_hardening: float = 0.3, satellite_shielding: float = 0.2,
                                   backup_power_days: float = 3.0) -> dict:
        dst_nadir = -800.0
        grid_damage_pct = min(1.0, (1.0 - grid_hardening) * 0.4)
        transformers_at_risk = int(grid_damage_pct * 2000)
        restoration_days = max(30, int(backup_power_days + grid_damage_pct * 365))
        satellite_loss_pct = min(0.8, (1.0 - satellite_shielding) * 0.3)
        gps_outage_days = int(7 - satellite_shielding * 5)
        gps_outage_days = max(1, gps_outage_days)
        hf_radio_blackout_hours = 48
        economic_impact_trillion = grid_damage_pct * 2.0 + satellite_loss_pct * 0.5 + 0.5
        return sanitize_nan({
            "scenario": "carrington_class",
            "dst_nadir_nT": dst_nadir,
            "grid_damage_pct": round(float(grid_damage_pct), 4),
            "transformers_at_risk": transformers_at_risk,
            "full_restoration_days": restoration_days,
            "satellite_loss_pct": round(float(satellite_loss_pct), 4),
            "gps_outage_days": gps_outage_days,
            "hf_radio_blackout_hours": hf_radio_blackout_hours,
            "economic_impact_trillion_usd": round(float(economic_impact_trillion), 2),
            "population_affected_billions": round(float(grid_damage_pct * 3.0 + 0.5), 2),
        })

    def predict(self, years_ahead: int = 5, state: Optional[SolarState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", SolarState())
        cycle_phase = state.solar_cycle_phase
        years_in_cycle = cycle_phase * 11.0
        future_phase = (years_in_cycle + years_ahead) / 11.0
        future_phase_mod = future_phase % 1.0
        if future_phase_mod < 0.3 or future_phase_mod > 0.8:
            future_sunspot = 30.0
            activity_regime = "solar_minimum"
        elif 0.35 < future_phase_mod < 0.65:
            future_sunspot = 150.0 + 50.0 * np.sin(future_phase_mod * np.pi * 2)
            activity_regime = "solar_maximum"
        else:
            future_sunspot = 80.0
            activity_regime = "transition"
        carrington_probability = 1.0 - np.exp(-0.01 * years_ahead)
        future_flux = future_sunspot * 1.6 + 67.0
        return sanitize_nan({
            "future_sunspot_number": round(float(future_sunspot), 1),
            "future_solar_flux": round(float(future_flux), 1),
            "activity_regime": activity_regime,
            "carrington_probability": round(float(carrington_probability), 4),
            "years_to_next_maximum": round(float(max(0, (0.45 - future_phase_mod) * 11.0 if future_phase_mod < 0.45 else (1.45 - future_phase_mod) * 11.0)), 1),
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