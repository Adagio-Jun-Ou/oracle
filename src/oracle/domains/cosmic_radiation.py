from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CosmicRadiationState:
    gcr_flux_relative: float = 1.0
    solar_modulation_potential_gv: float = 0.5
    geomagnetic_shielding_eff: float = 0.7
    atmospheric_shielding_eff: float = 0.85
    svensmark_cloud_nucleation_factor: float = 0.3
    neutron_monitor_count_rate: float = 200.0
    cosmic_ray_dose_rate_usv_hr: float = 0.35
    space_weather_shielding: float = 0.5
    magnetosphere_compression: float = 0.3
    ionization_rate_cm3_s: float = 2.0


class CosmicRadiationModel:

    def __init__(self):
        self.name = "cosmic_radiation_model"

    def set_state(self, **kwargs) -> "CosmicRadiationModel":
        current = CosmicRadiationState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, float(v))
        self.state = current
        return self

    def assess_radiation(self, state: Optional[CosmicRadiationState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", CosmicRadiationState())

        effective_flux = state.gcr_flux_relative * (1.0 - state.solar_modulation_potential_gv * 0.3)
        effective_flux = max(0.0, effective_flux)

        ground_level_flux = effective_flux * (1.0 - state.geomagnetic_shielding_eff * 0.6) * (1.0 - state.atmospheric_shielding_eff * 0.8)
        ground_level_flux = max(0.0, min(2.0, ground_level_flux))

        aviation_flux = effective_flux * (1.0 - state.geomagnetic_shielding_eff * 0.4) * (1.0 - state.atmospheric_shielding_eff * 0.3)
        aviation_flux = max(0.0, min(3.0, aviation_flux))

        svensmark_effect = state.svensmark_cloud_nucleation_factor * effective_flux * 0.15
        svensmark_effect = max(0.0, min(0.3, svensmark_effect))

        composite_risk = (
            ground_level_flux * 0.25
            + aviation_flux * 0.20
            + (1.0 - state.magnetosphere_compression) * 0.15
            + svensmark_effect * 0.15
            + state.cosmic_ray_dose_rate_usv_hr * 0.25
            + (1.0 - state.space_weather_shielding) * 0.10
        )
        composite_risk = min(1.0, max(0.0, composite_risk))

        if composite_risk < 0.2:
            risk_level = "low"
        elif composite_risk < 0.4:
            risk_level = "moderate"
        elif composite_risk < 0.6:
            risk_level = "elevated"
        elif composite_risk < 0.8:
            risk_level = "high"
        else:
            risk_level = "extreme"

        health_risk = state.cosmic_ray_dose_rate_usv_hr * ground_level_flux * 0.5
        health_risk = min(1.0, health_risk)

        aviation_exposure = aviation_flux * 0.4 + (1.0 - state.space_weather_shielding) * 0.2
        aviation_exposure = min(1.0, aviation_exposure)

        electronics_risk = (effective_flux * 0.3 + state.ionization_rate_cm3_s * 0.05
                           + (1.0 - state.space_weather_shielding) * 0.3)
        electronics_risk = min(1.0, electronics_risk)

        solar_cycle_phase = "solar_minimum" if state.solar_modulation_potential_gv < 0.4 else (
            "solar_maximum" if state.solar_modulation_potential_gv > 0.7 else "transitional"
        )

        recommendations = []
        if composite_risk > 0.5:
            recommendations.append("Elevated cosmic radiation - consider aviation route adjustments")
        if aviation_exposure > 0.4:
            recommendations.append("High aviation radiation exposure - polar routes particularly affected")
        if svensmark_effect > 0.1:
            recommendations.append("GCR-cloud nucleation coupling may influence regional cloud cover")
        if electronics_risk > 0.4:
            recommendations.append("Increased single-event effects risk for satellites and aviation electronics")
        if state.magnetosphere_compression > 0.5:
            recommendations.append("Compressed magnetosphere reducing geomagnetic shielding significantly")

        return sanitize_nan({
            "composite_risk": composite_risk,
            "risk_level": risk_level,
            "effective_gcr_flux": effective_flux,
            "ground_level_flux": ground_level_flux,
            "aviation_flux": aviation_flux,
            "svensmark_cloud_effect": svensmark_effect,
            "health_risk": health_risk,
            "aviation_risk": aviation_exposure,
            "electronics_seu_risk": electronics_risk,
            "solar_cycle_phase": solar_cycle_phase,
            "neutron_monitor_normalized": state.neutron_monitor_count_rate / 200.0,
            "dose_rate_primary": state.cosmic_ray_dose_rate_usv_hr,
            "recommendations": recommendations[:5],
        })

    def simulate_solar_minimum(self, duration_years: int = 3, severity: float = 0.8) -> dict:
        timeline = []
        current_flux = 1.0 + severity * 0.5
        current_dose = 0.35 + severity * 0.2
        current_ionization = 2.0 + severity * 1.5

        for year in range(1, duration_years + 1):
            phase_progress = year / duration_years
            flux = 1.0 + current_flux * (1.0 + 0.1 * np.sin(np.pi * phase_progress))
            dose = current_dose * (1.0 + 0.15 * np.sin(np.pi * phase_progress))
            cloud_effect = severity * 0.15 * flux * state_factor(year, duration_years)
            aviation_extra = (flux - 1.0) * 0.3
            timeline.append({
                "year": year,
                "gcr_flux_relative": round(flux, 3),
                "dose_rate_usv_hr": round(dose, 3),
                "ionization_rate_cm3_s": round(current_ionization * (0.9 + 0.2 * np.sin(np.pi * phase_progress)), 3),
                "svensmark_cloud_increase_pct": round(cloud_effect * 100, 1),
                "aviation_dose_increase_pct": round(aviation_extra * 100, 1),
                "neutron_monitor_pct_increase": round((flux - 1.0) * 100, 1),
            })

        peak_flux = max(t["gcr_flux_relative"] for t in timeline)
        avg_dose_increase = np.mean([t["dose_rate_usv_hr"] for t in timeline]) / 0.35 - 1.0
        cancer_risk_increase = peak_flux * 0.0003 * duration_years
        cloud_cover_change = np.mean([t["svensmark_cloud_increase_pct"] for t in timeline])

        return sanitize_nan({
            "scenario": "grand_solar_minimum",
            "duration_years": duration_years,
            "severity": severity,
            "peak_gcr_flux": round(peak_flux, 3),
            "average_dose_increase_pct": round(avg_dose_increase * 100, 1),
            "cumulative_cancer_risk_increase": round(cancer_risk_increase, 5),
            "cloud_cover_change_pct": round(cloud_cover_change, 1),
            "cooling_potential_c": round(severity * 0.1 * duration_years / max(1, duration_years), 2),
            "aviation_grounding_risk": "moderate" if severity > 0.5 else "low",
            "timeline": timeline,
        })

    def simulate_cme_shielding(self, cme_intensity: float = 0.7, recovery_hours: int = 48) -> dict:
        shielded_flux = 0.05 + (1.0 - cme_intensity) * 0.3
        peak_suppression = cme_intensity * 0.8
        dose_suppression = peak_suppression * 0.6

        recovery_phases = []
        for hour in [0, 6, 12, 24, 48, 72]:
            if hour <= recovery_hours:
                progress = hour / recovery_hours
                recovery = 1.0 - np.exp(-3 * progress)
                current_flux = shielded_flux + (1.0 - shielded_flux) * recovery
                current_suppression = peak_suppression * (1.0 - recovery)
            else:
                current_flux = 1.0
                current_suppression = 0.0
            recovery_phases.append({
                "hour": hour,
                "gcr_flux_relative": round(current_flux, 3),
                "dose_suppression_pct": round(current_suppression * 100, 1),
            })

        return sanitize_nan({
            "scenario": "cme_gcr_suppression",
            "cme_intensity": cme_intensity,
            "peak_gcr_suppression_pct": round(peak_suppression * 100, 1),
            "peak_dose_suppression_pct": round(dose_suppression * 100, 1),
            "recovery_hours": recovery_hours,
            "ground_level_dose_reduction_pct": round(dose_suppression * 70, 1),
            "aviation_dose_reduction_pct": round(dose_suppression * 50, 1),
            "forbush_decrease_pct": round(peak_suppression * 100, 1),
            "recovery_timeline": recovery_phases,
        })

    def predict(self, years_ahead: int = 10, state: Optional[CosmicRadiationState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", CosmicRadiationState())

        cycle_period = 11.0
        phase_in_cycle = (years_ahead % cycle_period) / cycle_period
        solar_min_prob = np.exp(-5 * ((phase_in_cycle - 0.0) ** 2)) + np.exp(-5 * ((phase_in_cycle - 1.0) ** 2))
        solar_max_prob = np.exp(-5 * ((phase_in_cycle - 0.5) ** 2))

        future_flux = state.gcr_flux_relative * (1.0 + solar_min_prob * 0.4 - solar_max_prob * 0.3)
        future_flux = max(0.3, min(2.5, future_flux))

        future_dose = state.cosmic_ray_dose_rate_usv_hr * future_flux
        future_ionization = state.ionization_rate_cm3_s * future_flux

        future_svensmark = state.svensmark_cloud_nucleation_factor * future_flux * 0.15

        future_modulation = 0.5 + 0.2 * np.sin(2 * np.pi * years_ahead / cycle_period)
        future_modulation = np.clip(future_modulation, 0.1, 0.9)

        geomagnetic_decay = 0.001 * years_ahead
        future_geo_shielding = max(0.3, state.geomagnetic_shielding_eff - geomagnetic_decay * 0.05)

        if future_flux > 1.3:
            radiation_regime = "grand_solar_minimum"
        elif future_flux > 1.1:
            radiation_regime = "solar_minimum_elevated"
        elif future_flux < 0.7:
            radiation_regime = "solar_maximum_suppressed"
        elif future_flux < 0.9:
            radiation_regime = "transitional_low"
        else:
            radiation_regime = "near_average"

        return sanitize_nan({
            "years_ahead": years_ahead,
            "future_gcr_flux": round(future_flux, 3),
            "future_dose_rate_usv_hr": round(future_dose, 3),
            "future_ionization_rate": round(future_ionization, 3),
            "future_svensmark_effect": round(future_svensmark, 3),
            "future_solar_modulation_gv": round(future_modulation, 3),
            "future_geomagnetic_shielding": round(future_geo_shielding, 3),
            "radiation_regime": radiation_regime,
            "cancer_risk_increase_cumulative": round(max(0, future_flux - 1.0) * 0.0003 * years_ahead, 5),
            "cloud_nucleation_change_pct": round(future_svensmark * 100, 1),
        })


def state_factor(year, total):
    return 0.5 + 0.5 * np.sin(np.pi * year / max(total, 1))


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