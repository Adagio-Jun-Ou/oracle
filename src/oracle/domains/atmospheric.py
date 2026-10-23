from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AtmosphericState:
    jet_stream_latitude_offset: float = 0.0
    jet_stream_speed_kts: float = 130.0
    polar_vortex_strength: float = 0.6
    stratospheric_warming_risk: float = 0.15
    ozone_column_du: float = 300.0
    uv_index_anomaly: float = 0.0
    geomagnetic_field_strength_nt: float = 45000.0
    geomagnetic_delta_nt: float = 0.0
    atmospheric_circulation_index: float = 0.5
    hadley_cell_extent_deg: float = 30.0
    rossby_wave_amplitude: float = 0.4
    atmospheric_co2_ppm: float = 420.0
    methane_ppb: float = 1900.0
    aerosol_optical_depth: float = 0.12


class AtmosphericModel:

    def __init__(self):
        self.name = "atmospheric_model"

    def set_state(self, **kwargs) -> "AtmosphericModel":
        current = AtmosphericState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, float(v))
        self.state = current
        return self

    def assess_atmospheric_state(self, state: Optional[AtmosphericState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", AtmosphericState())

        jet_anomaly = abs(state.jet_stream_latitude_offset) / 15.0
        jet_anomaly = min(1.0, jet_anomaly)
        vortex_instability = max(0.0, 1.0 - state.polar_vortex_strength)
        ssw_risk = state.stratospheric_warming_risk

        ozone_deficit = max(0.0, (300.0 - state.ozone_column_du) / 150.0)
        ozone_deficit = min(1.0, ozone_deficit)
        uv_hazard = max(0.0, state.uv_index_anomaly)

        geo_magnitude = abs(state.geomagnetic_delta_nt) / 500.0
        geo_magnitude = min(1.0, geo_magnitude)

        circulation_disruption = state.rossby_wave_amplitude * 0.4 + jet_anomaly * 0.3 + vortex_instability * 0.3
        circulation_disruption = min(1.0, circulation_disruption)

        greenhouse_pressure = (state.atmospheric_co2_ppm - 280.0) / 500.0
        greenhouse_pressure = min(1.0, max(0.0, greenhouse_pressure))
        methane_contribution = state.methane_ppb / 5000.0
        methane_contribution = min(1.0, methane_contribution)

        composite = (
            circulation_disruption * 0.25
            + ssw_risk * 0.15
            + ozone_deficit * 0.15
            + uv_hazard * 0.10
            + geo_magnitude * 0.10
            + greenhouse_pressure * 0.15
            + methane_contribution * 0.10
        )
        composite = min(1.0, max(0.0, composite))

        if composite > 0.7:
            level = "critical"
        elif composite > 0.5:
            level = "elevated"
        elif composite > 0.3:
            level = "moderate"
        else:
            level = "stable"

        hadley_expansion = max(0.0, (state.hadley_cell_extent_deg - 30.0) / 15.0)
        hadley_expansion = min(1.0, hadley_expansion)

        jet_shift_direction = "poleward" if state.jet_stream_latitude_offset > 0 else "equatorward"

        recommendations = []
        if ozone_deficit > 0.3:
            recommendations.append("Monitor UV radiation levels; issue advisories for affected regions")
        if ssw_risk > 0.3:
            recommendations.append("Prepare for polar vortex disruption and extreme cold events")
        if hadley_expansion > 0.3:
            recommendations.append("Assess subtropical drying impacts on agriculture and water")
        if geo_magnitude > 0.3:
            recommendations.append("Track geomagnetic anomalies for navigation and grid impacts")
        if greenhouse_pressure > 0.5:
            recommendations.append("Atmospheric greenhouse load warrants emissions reduction")
        if state.aerosol_optical_depth > 0.3:
            recommendations.append("High aerosol load may mask warming; monitor air quality")
        if len(recommendations) > 5:
            recommendations = recommendations[:5]

        return sanitize_nan({
            "composite_atmospheric_risk": composite,
            "risk_level": level,
            "circulation_disruption": circulation_disruption,
            "jet_stream_status": jet_shift_direction,
            "jet_stream_anomaly_score": jet_anomaly,
            "polar_vortex_status": "weakened" if vortex_instability > 0.4 else "stable",
            "ssw_risk_score": ssw_risk,
            "ozone_deficit_pct": ozone_deficit * 100.0,
            "uv_hazard_score": uv_hazard,
            "geomagnetic_anomaly_score": geo_magnitude,
            "hadley_cell_expansion": hadley_expansion,
            "greenhouse_pressure": greenhouse_pressure,
            "recommendations": recommendations,
        })

    def simulate_ssw_scenario(self, vortex_strength: float = 0.2, ssw_intensity: float = 0.8, duration_days: int = 30) -> dict:
        cold_outbreak_probability = ssw_intensity * 0.7 + (1.0 - vortex_strength) * 0.3
        cold_outbreak_probability = min(1.0, cold_outbreak_probability)

        temp_anomaly_c_northern = -ssw_intensity * 25.0 * (1.0 - vortex_strength)
        temp_anomaly_c_northern = max(-40.0, temp_anomaly_c_northern)

        jet_reversal_probability = ssw_intensity * 0.6
        jet_reversal_probability = min(1.0, jet_reversal_probability)

        winter_storm_amplification = 1.0 + ssw_intensity * 1.5 * (1.0 - vortex_strength)

        recovery_days = int(30 + duration_days * (1.0 - vortex_strength))

        affected_population_m = ssw_intensity * 500.0 + abs(temp_anomaly_c_northern) * 20.0
        affected_population_m = min(2000.0, affected_population_m)

        energy_demand_increase_pct = abs(temp_anomaly_c_northern) * 2.0
        energy_demand_increase_pct = min(100.0, energy_demand_increase_pct)

        ozone_impact_du = ssw_intensity * 30.0

        return sanitize_nan({
            "scenario": "sudden_stratospheric_warming",
            "cold_outbreak_probability": cold_outbreak_probability,
            "temperature_anomaly_northern_c": temp_anomaly_c_northern,
            "jet_reversal_probability": jet_reversal_probability,
            "winter_storm_amplification": winter_storm_amplification,
            "recovery_days": recovery_days,
            "affected_population_millions": affected_population_m,
            "energy_demand_increase_pct": energy_demand_increase_pct,
            "ozone_impact_du": ozone_impact_du,
            "duration_days": duration_days,
        })

    def simulate_ozone_scenario(self, ozone_column_du: float = 150.0, duration_months: int = 6) -> dict:
        ozone_deficit_pct = max(0.0, (300.0 - ozone_column_du) / 300.0 * 100.0)
        uv_increase_pct = ozone_deficit_pct * 1.5
        uv_increase_pct = min(300.0, uv_increase_pct)

        skin_cancer_risk_increase = uv_increase_pct * 0.5
        skin_cancer_risk_increase = min(100.0, skin_cancer_risk_increase)

        crop_damage_pct = uv_increase_pct * 0.3
        crop_damage_pct = min(50.0, crop_damage_pct)

        phytoplankton_impact_pct = uv_increase_pct * 0.4
        phytoplankton_impact_pct = min(60.0, phytoplankton_impact_pct)

        recovery_years = max(5, int(ozone_deficit_pct / 2.0))

        affected_regions = []
        if ozone_column_du < 200:
            affected_regions = ["antarctic", "southern_ocean", "patagonia", "new_zealand"]
        elif ozone_column_du < 250:
            affected_regions = ["antarctic", "southern_ocean", "patagonia"]
        else:
            affected_regions = ["antarctic", "southern_ocean"]

        return sanitize_nan({
            "scenario": "ozone_depletion",
            "ozone_deficit_pct": ozone_deficit_pct,
            "uv_increase_pct": uv_increase_pct,
            "skin_cancer_risk_increase_pct": skin_cancer_risk_increase,
            "crop_damage_pct": crop_damage_pct,
            "phytoplankton_impact_pct": phytoplankton_impact_pct,
            "recovery_years": recovery_years,
            "affected_regions": affected_regions,
            "duration_months": duration_months,
        })

    def predict(self, years_ahead: int = 10, state: Optional[AtmosphericState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", AtmosphericState())

        co2_growth = 2.5
        methane_growth_pct = 0.005
        future_co2 = state.atmospheric_co2_ppm + co2_growth * years_ahead
        future_methane = state.methane_ppb * (1.0 + methane_growth_pct) ** years_ahead

        ozone_recovery_rate = 0.5
        future_ozone = min(300.0, state.ozone_column_du + ozone_recovery_rate * years_ahead * (1.0 if state.ozone_column_du < 300.0 else 0.0))

        geo_decay_rate = 0.995
        future_geomag = state.geomagnetic_field_strength_nt * geo_decay_rate ** (years_ahead * 10)
        geo_trend = "weakening" if future_geomag < state.geomagnetic_field_strength_nt else "stable"

        hadley_rate = 0.2
        future_hadley = state.hadley_cell_extent_deg + hadley_rate * years_ahead

        ssw_freq_increase = 1.0 + years_ahead * 0.02
        vortex_trend = "weakening" if state.polar_vortex_strength < 0.5 else "variable"

        future_jet_shift = state.jet_stream_latitude_offset + years_ahead * 0.1
        jet_trend = "poleward_shift" if future_jet_shift > 0 else "equatorward_shift"

        aerosol_trend = "declining" if state.aerosol_optical_depth > 0.1 else "stable"

        return sanitize_nan({
            "future_co2_ppm": future_co2,
            "future_methane_ppb": future_methane,
            "future_ozone_du": future_ozone,
            "geomagnetic_trend": geo_trend,
            "future_hadley_extent_deg": future_hadley,
            "vortex_trend": vortex_trend,
            "jet_stream_trend": jet_trend,
            "ssw_frequency_multiplier": ssw_freq_increase,
            "aerosol_trend": aerosol_trend,
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