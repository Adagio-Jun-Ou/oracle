from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class GravitationalState:
    lunar_tidal_force_relative: float = 1.0
    solar_tidal_force_relative: float = 0.46
    planetary_alignment_index: float = 0.2
    earth_moon_distance_km: float = 384400.0
    tidal_period_hours: float = 12.42
    tidal_range_variation_pct: float = 5.0
    solid_earth_tide_amplitude_cm: float = 30.0
    ocean_tide_amplitude_m: float = 1.0
    resonance_frequency_hz: float = 0.0001
    rotational_energy_dissipation_pw: float = 3.7
    crustal_stress_tidal_mpa: float = 0.3


class GravitationalModel:

    def __init__(self):
        self.name = "gravitational_model"

    def set_state(self, **kwargs) -> "GravitationalModel":
        current = GravitationalState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, float(v))
        self.state = current
        return self

    def assess_tidal_forces(self, state: Optional[GravitationalState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", GravitationalState())

        combined_tidal = state.lunar_tidal_force_relative + state.solar_tidal_force_relative * 0.46
        combined_tidal = max(0.0, combined_tidal)

        alignment_force = state.planetary_alignment_index * combined_tidal * 0.15
        alignment_force = min(0.5, alignment_force)

        perigee_factor = (384400.0 / max(state.earth_moon_distance_km, 350000.0)) ** 3
        perigee_factor = min(2.0, perigee_factor)

        effective_tidal = combined_tidal * perigee_factor * (1.0 + alignment_force)

        spring_tide_amplification = 1.0 + state.solar_tidal_force_relative * 0.46 * state.lunar_tidal_force_relative
        neap_tide_suppression = max(0.2, 1.0 - state.solar_tidal_force_relative * 0.3)

        seismic_triggering = (
            effective_tidal * 0.3
            + state.crustal_stress_tidal_mpa * 0.2
            + state.planetary_alignment_index * 0.15
            + state.solid_earth_tide_amplitude_cm * 0.003
        )
        seismic_triggering = min(1.0, max(0.0, seismic_triggering))

        volcanic_triggering = (
            effective_tidal * 0.15
            + state.crustal_stress_tidal_mpa * 0.25
            + state.planetary_alignment_index * 0.1
        )
        volcanic_triggering = min(1.0, max(0.0, volcanic_triggering))

        coastal_flood_risk = (
            state.ocean_tide_amplitude_m * 0.3
            + state.tidal_range_variation_pct * 0.01
            + effective_tidal * 0.3
        )
        coastal_flood_risk = min(1.0, max(0.0, coastal_flood_risk))

        rotational_deceleration = state.rotational_energy_dissipation_pw * 0.001 * effective_tidal
        day_lengthening_us_per_century = rotational_deceleration * 2.3

        composite = (
            seismic_triggering * 0.3
            + volcanic_triggering * 0.2
            + coastal_flood_risk * 0.25
            + alignment_force * 0.15
            + effective_tidal * 0.1
        )
        composite = min(1.0, max(0.0, composite))

        if composite < 0.2:
            level = "minimal"
        elif composite < 0.4:
            level = "low"
        elif composite < 0.6:
            level = "moderate"
        elif composite < 0.8:
            level = "elevated"
        else:
            level = "high"

        recommendations = []
        if seismic_triggering > 0.4:
            recommendations.append("Enhanced tidal stress - slightly elevated seismic triggering probability")
        if coastal_flood_risk > 0.4:
            recommendations.append("King tide conditions - coastal flooding risk increased")
        if state.planetary_alignment_index > 0.5:
            recommendations.append("Notable planetary alignment - minor gravitational perturbation")
        if volcanic_triggering > 0.3:
            recommendations.append("Tidal loading may slightly influence volcanic activity")
        if perigee_factor > 1.3:
            recommendations.append("Perigee proximity amplifying tidal forces")

        return sanitize_nan({
            "composite_tidal_index": composite,
            "level": level,
            "effective_tidal_force": round(effective_tidal, 3),
            "perigee_amplification": round(perigee_factor, 3),
            "spring_tide_amplification": round(spring_tide_amplification, 3),
            "neap_tide_suppression": round(neap_tide_suppression, 3),
            "seismic_triggering_probability": seismic_triggering,
            "volcanic_triggering_probability": volcanic_triggering,
            "coastal_flood_risk": coastal_flood_risk,
            "planetary_alignment_force": alignment_force,
            "rotational_deceleration_pw": round(rotational_deceleration, 4),
            "day_lengthening_us_per_century": round(day_lengthening_us_per_century, 2),
            "recommendations": recommendations[:5],
        })

    def simulate_alignment(self, n_planets: int = 5, alignment_strength: float = 0.7, days: int = 30) -> dict:
        planet_tides = {
            "Jupiter": 0.0000047,
            "Venus": 0.0000054,
            "Mars": 0.0000002,
            "Mercury": 0.00000004,
            "Saturn": 0.0000005,
        }
        planets = list(planet_tides.keys())[:n_planets]
        alignment_forces = {p: planet_tides[p] * alignment_strength * 1e6 for p in planets}

        total_planetary = sum(alignment_forces.values())

        timeline = []
        for day in range(0, days + 1, max(1, days // 10)):
            lunar_phase = 0.5 * (1.0 + np.cos(2 * np.pi * day / 29.53))
            tidal_lunar = 1.0 + 0.18 * lunar_phase
            tidal_solar = 0.46 + alignment_strength * 0.05 * np.sin(2 * np.pi * day / days)
            combined = (tidal_lunar + tidal_solar) * (1.0 + alignment_strength * 0.02)
            seismic_prob = combined * 0.05 * alignment_strength
            timeline.append({
                "day": day,
                "lunar_tidal_relative": round(tidal_lunar, 3),
                "combined_tidal": round(combined, 3),
                "seismic_probability_multiplier": round(1.0 + seismic_prob, 3),
            })

        peak_seismic_multiplier = max(t["seismic_probability_multiplier"] for t in timeline)

        return sanitize_nan({
            "scenario": "planetary_alignment",
            "n_planets": n_planets,
            "alignment_strength": alignment_strength,
            "planetary_tidal_forces_ne": alignment_forces,
            "total_planetary_tide_ne": round(total_planetary, 4),
            "lunar_tidal_comparison": round(total_planetary / 4.39e-1, 6),
            "peak_seismic_probability_multiplier": round(peak_seismic_multiplier, 3),
            "coastal_flood_risk_increase_pct": round(alignment_strength * 3.0, 1),
            "volcanic_triggering_increase_pct": round(alignment_strength * 1.5, 1),
            "timeline": timeline,
        })

    def simulate_perigee_spring(self, perigee_distance_km: float = 356500.0, duration_days: int = 7) -> dict:
        perigee_factor = (384400.0 / max(perigee_distance_km, 340000.0)) ** 3

        king_tide_amplification = perigee_factor * 1.18
        coastal_flood_multiplier = min(3.0, perigee_factor * 0.8)
        seismic_stress_mpa = 0.3 * perigee_factor

        timeline = []
        for day in range(duration_days + 1):
            phase = np.cos(2 * np.pi * day / 14.76) * 0.5 + 0.5
            tide_height = 1.0 * king_tide_amplification * (0.8 + 0.2 * phase)
            timeline.append({
                "day": day,
                "tide_height_m": round(tide_height, 2),
                "tidal_range_pct_above_mean": round((king_tide_amplification - 1.0) * 100 * phase, 1),
                "crustal_stress_mpa": round(seismic_stress_mpa * (0.5 + 0.5 * phase), 3),
            })

        return sanitize_nan({
            "scenario": "perigee_spring_tide",
            "perigee_distance_km": perigee_distance_km,
            "perigee_amplification": round(perigee_factor, 3),
            "king_tide_amplification": round(king_tide_amplification, 3),
            "coastal_flood_multiplier": round(coastal_flood_multiplier, 3),
            "seismic_stress_mpa": round(seismic_stress_mpa, 3),
            "volcanic_loading_mpa": round(seismic_stress_mpa * 0.5, 3),
            "day_lengthening_us": round(2.3 * perigee_factor / 1.0, 3),
            "timeline": timeline,
        })

    def predict(self, years_ahead: int = 10, state: Optional[GravitationalState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", GravitationalState())

        lunar_recession_km = years_ahead * 3.82
        future_moon_distance = state.earth_moon_distance_km + lunar_recession_km

        future_lunar_tidal = state.lunar_tidal_force_relative * (384400.0 / future_moon_distance) ** 3

        future_tidal_range_variation = state.tidal_range_variation_pct * (1.0 + 0.01 * years_ahead)
        future_solid_earth_tide = state.solid_earth_tide_amplitude_cm * (384400.0 / future_moon_distance) ** 3
        future_ocean_tide = state.ocean_tide_amplitude_m * (384400.0 / future_moon_distance) ** 3

        future_dissipation = state.rotational_energy_dissipation_pw * (384400.0 / future_moon_distance) ** 6

        future_day_lengthening = years_ahead * 2.3

        alignment_cycles = years_ahead / 179.0
        major_alignment_prob = min(1.0, alignment_cycles * 0.05)

        seismic_contribution = (
            future_lunar_tidal * 0.3 + state.crustal_stress_tidal_mpa * 0.2
        ) * (1.0 + alignment_cycles * 0.02)

        if years_ahead < 20:
            regime = "current_cycle"
        elif years_ahead < 100:
            regime = "millennial_variation"
        elif years_ahead < 1000:
            regime = "orbital_evolution"
        else:
            regime = "deep_time_recession"

        return sanitize_nan({
            "years_ahead": years_ahead,
            "future_moon_distance_km": round(future_moon_distance, 1),
            "future_lunar_tidal_force": round(future_lunar_tidal, 4),
            "future_tidal_range_pct": round(future_tidal_range_variation, 1),
            "future_solid_earth_tide_cm": round(future_solid_earth_tide, 1),
            "future_ocean_tide_m": round(future_ocean_tide, 3),
            "future_rotational_dissipation_pw": round(future_dissipation, 3),
            "cumulative_day_lengthening_us": round(future_day_lengthening, 1),
            "major_alignment_probability": round(major_alignment_prob, 3),
            "seismic_contribution_index": round(seismic_contribution, 3),
            "tidal_regime": regime,
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