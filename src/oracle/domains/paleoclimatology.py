from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PaleoclimateState:
    milankovitch_obliquity_deg: float = 23.45
    milankovitch_eccentricity: float = 0.0167
    milankovitch_precession: float = 0.0167
    ice_volume_fraction: float = 0.03
    co2_ppmv: float = 420.0
    global_temp_anomaly_c: float = 1.2
    solar_luminosity_fraction: float = 1.0
    volcanic_aerosol_loading: float = 0.05
    thermohaline_circulation_strength: float = 0.7
    vegetation_cover_fraction: float = 0.30
    methane_ppbv: float = 1900.0
    albedo: float = 0.30
    sea_level_m_above_present: float = 0.0


class PaleoclimatologyModel:

    def __init__(self):
        self.name = "paleoclimatology_model"

    def set_state(self, **kwargs) -> "PaleoclimatologyModel":
        current = PaleoclimateState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, float(v))
        self.state = current
        return self

    def assess_milankovitch(self, state: Optional[PaleoclimateState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", PaleoclimateState())

        obliquity_high = abs(state.milankovitch_obliquity_deg - 22.1) / (24.5 - 22.1)
        eccentricity_effect = state.milankovitch_eccentricity / 0.05
        precession_effect = state.milankovitch_precession / 0.058

        insolation_variability = (
            obliquity_high * 0.4 + eccentricity_effect * 0.3 + precession_effect * 0.3
        )
        insolation_variability = min(1.0, max(0.0, insolation_variability))

        if insolation_variability < 0.25:
            phase = "favorable_interglacial"
        elif insolation_variability < 0.50:
            phase = "transitional_warming"
        elif insolation_variability < 0.75:
            phase = "transitional_cooling"
        else:
            phase = "favorable_glacial"

        ice_threshold = 0.08 + state.co2_ppmv / 5000.0
        glaciation_risk = max(0.0, state.ice_volume_fraction - ice_threshold * 0.5) * 2.0
        glaciation_risk += insolation_variability * 0.3
        glaciation_risk += (1.0 - state.thermohaline_circulation_strength) * 0.2
        glaciation_risk = min(1.0, max(0.0, glaciation_risk))

        deglaciation_risk = 0.0
        if state.co2_ppmv > 450:
            deglaciation_risk += (state.co2_ppmv - 350) / 1000.0
        deglaciation_risk += state.global_temp_anomaly_c / 5.0
        deglaciation_risk += (1.0 - state.ice_volume_fraction) * 0.3
        deglaciation_risk = min(1.0, max(0.0, deglaciation_risk))

        cycle_forcing = (
            obliquity_high * 0.35 + eccentricity_effect * 0.30 + precession_effect * 0.35
        )

        radiative_forcing = (
            np.log(state.co2_ppmv / 280.0) * 5.35
            + np.log(state.methane_ppbv / 700.0) * 0.6
            - state.volcanic_aerosol_loading * 1.5
            + (1.0 - state.albedo) * 0.3 * state.solar_luminosity_fraction
        )

        next_glaciation_kyr = max(5, 50 - glaciation_risk * 40 + insolation_variability * 10)

        recommendations = []
        if glaciation_risk > 0.5:
            recommendations.append("High glaciation risk - insolation favorable for ice growth")
        if deglaciation_risk > 0.5:
            recommendations.append("High deglaciation risk - CO2 forcing dominates orbital forcing")
        if state.thermohaline_circulation_strength < 0.3:
            recommendations.append("Thermohaline circulation weakened - may trigger abrupt cooling")
        if state.volcanic_aerosol_loading > 0.3:
            recommendations.append("High volcanic aerosol loading - temporary cooling forcing")
        if state.ice_volume_fraction > 0.15:
            recommendations.append("Elevated ice volume - positive albedo feedback active")

        return sanitize_nan({
            "insolation_variability": insolation_variability,
            "orbital_phase": phase,
            "obliquity_contribution": obliquity_high,
            "eccentricity_contribution": eccentricity_effect,
            "precession_contribution": precession_effect,
            "glaciation_risk": glaciation_risk,
            "deglaciation_risk": deglaciation_risk,
            "cycle_forcing": cycle_forcing,
            "radiative_forcing_wm2": radiative_forcing,
            "next_glaciation_estimate_kyr": next_glaciation_kyr,
            "ice_volume_assessment": "expanding" if glaciation_risk > 0.5 else ("retreating" if deglaciation_risk > 0.5 else "stable"),
            "recommendations": recommendations[:5],
        })

    def simulate_analog(self, analog: str = "eemian", years: int = 5000) -> dict:
        analogs = {
            "eemian": {
                "name": "Eemian Interglacial (130-115 kya)",
                "co2_ppmv": 290, "temp_anomaly_c": 2.0, "sea_level_m": 6.0,
                "ice_volume_fraction": 0.02, "vegetation_fraction": 0.35,
                "obliquity_deg": 24.2, "eccentricity": 0.04,
            },
            "mid_holocene": {
                "name": "Mid-Holocene Optimum (8-6 kya)",
                "co2_ppmv": 265, "temp_anomaly_c": 0.7, "sea_level_m": 2.0,
                "ice_volume_fraction": 0.025, "vegetation_fraction": 0.32,
                "obliquity_deg": 24.1, "eccentricity": 0.02,
            },
            "younger_dryas": {
                "name": "Younger Dryas (12.9-11.7 kya)",
                "co2_ppmv": 240, "temp_anomaly_c": -3.0, "sea_level_m": -40.0,
                "ice_volume_fraction": 0.20, "vegetation_fraction": 0.20,
                "obliquity_deg": 24.0, "eccentricity": 0.02,
            },
            "last_glacial_maximum": {
                "name": "Last Glacial Maximum (26-19 kya)",
                "co2_ppmv": 180, "temp_anomaly_c": -6.0, "sea_level_m": -120.0,
                "ice_volume_fraction": 0.30, "vegetation_fraction": 0.15,
                "obliquity_deg": 23.2, "eccentricity": 0.02,
            },
            "petm": {
                "name": "Paleocene-Eocene Thermal Maximum (56 Mya)",
                "co2_ppmv": 1000, "temp_anomaly_c": 8.0, "sea_level_m": 20.0,
                "ice_volume_fraction": 0.0, "vegetation_fraction": 0.45,
                "obliquity_deg": 23.5, "eccentricity": 0.03,
            },
            "mid_pliocene": {
                "name": "Mid-Pliocene Warm Period (3.3 Mya)",
                "co2_ppmv": 400, "temp_anomaly_c": 3.0, "sea_level_m": 15.0,
                "ice_volume_fraction": 0.01, "vegetation_fraction": 0.38,
                "obliquity_deg": 23.8, "eccentricity": 0.02,
            },
        }

        config = analogs.get(analog, analogs["eemian"])
        co2_rate = (420.0 - config["co2_ppmv"]) / max(years, 1)
        temp_rate = config["temp_anomaly_c"] / max(years / 2, 1)
        sea_rate = config["sea_level_m"] / max(years / 2, 1)

        timeline = []
        n_points = min(20, years // max(years // 20, 1))
        for i in range(n_points + 1):
            t = int(i * years / n_points)
            progress = i / n_points if n_points > 0 else 0
            co2 = config["co2_ppmv"] + co2_rate * t
            temp = config["temp_anomaly_c"] * (1.0 - progress * 0.3) + temp_rate * t * 0.5
            sea = config["sea_level_m"] * (1.0 - progress * 0.2) + sea_rate * t * 0.3
            ice = config["ice_volume_fraction"] * (1.0 + progress * 0.1)
            veg = config["vegetation_fraction"] * (1.0 - progress * 0.15)
            timeline.append({
                "year_offset": t,
                "co2_ppmv": max(150, min(1500, co2)),
                "temp_anomaly_c": round(temp, 2),
                "sea_level_m": round(sea, 1),
                "ice_volume_fraction": max(0.0, min(0.5, ice)),
                "vegetation_fraction": max(0.05, min(0.5, veg)),
            })

        divergence = abs(420.0 - config["co2_ppmv"])
        similarity = max(0.0, 1.0 - divergence / 1200.0)

        analog_temp = config["temp_anomaly_c"]
        current_relevance = "high" if abs(analog_temp - 1.2) < 2.0 else ("moderate" if abs(analog_temp - 1.2) < 4.0 else "low")

        lessons = []
        if analog == "petm":
            lessons.append("Carbon release rate was ~0.5 GtC/yr; current rate is ~10 GtC/yr (20x faster)")
            lessons.append("Ocean acidification led to deep-sea extinction; recovery took 100,000+ years")
            lessons.append("Temperature spike persisted for ~200,000 years")
        elif analog == "mid_pliocene":
            lessons.append("CO2 similar to today produced 3C warming and 15m sea level rise")
            lessons.append("Vegetation shifted poleward significantly")
            lessons.append("Ice sheets were minimal even with CO2 at 400 ppm")
        elif analog == "eemian":
            lessons.append("Natural warming reached 2C with only 290 ppm CO2")
            lessons.append("Sea levels 6-9m higher than present")
            lessons.append("Orbital configuration was more favorable for warming")
        elif analog == "last_glacial_maximum":
            lessons.append("6C cooler than present with 180 ppm CO2")
            lessons.append("Ice sheets covered much of North America and Europe")
            lessons.append("Vegetation reduced to 15% global cover")
        elif analog == "younger_dryas":
            lessons.append("Abrupt cooling of 5-10C within decades")
            lessons.append("Likely triggered by freshwater pulse disrupting thermohaline circulation")
            lessons.append("Analog for potential AMOC disruption scenarios")
        elif analog == "mid_holocene":
            lessons.append("Slightly warmer than present with lower CO2")
            lessons.append("African Humid Period: Sahara was green")
            lessons.append("Orbital forcing was key driver of climate optimum")

        return sanitize_nan({
            "analog": analog,
            "analog_name": config["name"],
            "timeline": timeline,
            "co2_divergence_ppmv": round(divergence, 1),
            "similarity_score": similarity,
            "current_relevance": current_relevance,
            "key_lessons": lessons,
            "projected_equilibrium_temp_c": round(analog_temp * similarity + 1.2 * (1 - similarity), 2),
            "projected_equilibrium_sea_level_m": round(config["sea_level_m"] * similarity * 0.5, 1),
        })

    def predict(self, years_ahead: int = 1000, state: Optional[PaleoclimateState] = None) -> dict:
        if state is None:
            state = getattr(self, "state", PaleoclimateState())

        milankovitch_cycle_period = 41000
        precession_period = 26000
        eccentricity_period = 100000

        obliquity_change = 0.5 * np.sin(2 * np.pi * years_ahead / milankovitch_cycle_period)
        future_obliquity = np.clip(state.milankovitch_obliquity_deg + obliquity_change * 0.5, 22.0, 24.5)

        eccentricity_change = 0.01 * np.sin(2 * np.pi * years_ahead / eccentricity_period)
        future_eccentricity = np.clip(state.milankovitch_eccentricity + eccentricity_change, 0.001, 0.06)

        precession_change = np.sin(2 * np.pi * years_ahead / precession_period)

        co2_rate = 0.002
        future_co2 = state.co2_ppmv + co2_rate * years_ahead
        if years_ahead > 200:
            future_co2 = state.co2_ppmv + (future_co2 - state.co2_ppmv) * 0.3

        future_temp = state.global_temp_anomaly_c + (
            np.log(future_co2 / 280.0) * 3.0
            - np.log(state.co2_ppmv / 280.0) * 3.0
        ) + obliquity_change * 0.5 + precession_change * 0.3

        thermal_expansion = future_temp * 0.2
        ice_melt = max(0, future_temp - 0.5) * 0.5
        future_sea_level = state.sea_level_m_above_present + thermal_expansion + ice_melt

        future_ice = max(0.0, state.ice_volume_fraction - max(0, future_temp) * 0.01)
        if future_temp < 0:
            future_ice = min(0.4, state.ice_volume_fraction + abs(future_temp) * 0.02)

        future_vegetation = max(0.05, min(0.45, state.vegetation_cover_fraction + future_temp * 0.01))

        if future_temp > 3.0:
            climate_regime = "hothouse"
        elif future_temp > 1.5:
            climate_regime = "warm_interglacial"
        elif future_temp > 0.0:
            climate_regime = "interglacial"
        elif future_temp > -2.0:
            climate_regime = "cool_interglacial"
        elif future_temp > -5.0:
            climate_regime = "glacial"
        else:
            climate_regime = "deep_glacial"

        tipping_risk = 0.0
        if future_co2 > 450:
            tipping_risk += (future_co2 - 400) / 1000.0
        if state.thermohaline_circulation_strength < 0.4:
            tipping_risk += 0.2
        if future_temp > 2.0:
            tipping_risk += 0.15
        tipping_risk = min(1.0, tipping_risk)

        return sanitize_nan({
            "years_ahead": years_ahead,
            "future_obliquity_deg": round(future_obliquity, 3),
            "future_eccentricity": round(future_eccentricity, 5),
            "future_co2_ppmv": round(future_co2, 1),
            "future_temp_anomaly_c": round(future_temp, 2),
            "future_sea_level_m": round(future_sea_level, 1),
            "future_ice_volume_fraction": round(future_ice, 4),
            "future_vegetation_cover": round(future_vegetation, 3),
            "climate_regime": climate_regime,
            "tipping_point_risk": round(tipping_risk, 3),
            "orbital_phase": "cooling" if obliquity_change < -0.1 else ("warming" if obliquity_change > 0.1 else "neutral"),
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