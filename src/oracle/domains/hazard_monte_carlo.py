from dataclasses import dataclass
from typing import Optional
import numpy as np


HAZARD_DISTRIBUTIONS = {
    "earthquake": {"magnitude_mean": 5.5, "magnitude_std": 0.8, "annual_rate": 12.0, "depth_mean": 30.0, "depth_std": 15.0},
    "volcanic_eruption": {"vei_mean": 2.0, "vei_std": 0.8, "annual_rate": 1.5, "duration_days_mean": 30, "duration_days_std": 60},
    "solar_storm": {"kp_mean": 4.0, "kp_std": 1.5, "annual_rate_major": 0.5, "carrington_rate_per_century": 0.12},
    "hurricane": {"category_mean": 1.5, "category_std": 0.8, "annual_rate_atlantic": 12.0, "landfall_probability": 0.25},
    "flood": {"return_period_100yr": 0.01, "annual_rate_major": 0.02, "depth_m_mean": 1.0, "depth_m_std": 0.5},
    "drought": {"duration_months_mean": 6, "duration_months_std": 4, "annual_rate": 0.08, "severity_mean": 0.3, "severity_std": 0.15},
    "tsunami": {"height_m_mean": 2.0, "height_m_std": 3.0, "rate_given_m8": 0.3, "annual_rate_global": 0.02},
    "asteroid_impact": {"rate_1m_per_year": 0.5, "rate_10m_per_year": 0.02, "rate_100m_per_million_years": 0.001},
}


@dataclass
class HazardEvent:
    hazard_type: str
    magnitude: float
    year: int
    damage_fraction: float
    economic_loss_billion: float
    fatalities: int
    recovery_years: int


class NaturalHazardMonteCarlo:

    def __init__(self, seed: Optional[int] = None):
        self.name = "natural_hazard_monte_carlo"
        self.seed = seed

    def simulate_portfolio(self, hazards: list[str] = None,
                           years: int = 50,
                           n_simulations: int = 1000,
                           exposure_billion: float = 100.0,
                           resilience_factor: float = 0.6,
                           interaction_enabled: bool = True) -> dict:
        if hazards is None:
            hazards = ["earthquake", "volcanic_eruption", "solar_storm", "hurricane", "flood", "drought"]

        rng = np.random.RandomState(self.seed)
        all_events: list[list[HazardEvent]] = []
        total_losses = []
        max_annual_losses = []
        recovery_trajectories = []

        for sim in range(n_simulations):
            sim_events: list[HazardEvent] = []
            annual_losses = np.zeros(years)
            current_resilience = resilience_factor

            for yr in range(years):
                year_events = []
                for hz in hazards:
                    dist = HAZARD_DISTRIBUTIONS.get(hz, {})
                    annual_rate = dist.get("annual_rate", dist.get("annual_rate_major", dist.get("annual_rate_atlantic", 0.1)))

                    if hz == "hurricane":
                        n_events = rng.poisson(annual_rate)
                        for _ in range(n_events):
                            if rng.random() < dist.get("landfall_probability", 0.25):
                                cat = max(1, min(5, rng.normal(dist["category_mean"], dist["category_std"])))
                                cat = int(round(cat))
                                damage = min(1.0, (cat / 5.0) ** 2 * 0.3 * (1.5 - current_resilience * 0.5))
                                loss = damage * exposure_billion
                                fatalities = int(cat ** 2 * 10 * rng.uniform(0.5, 2.0))
                                event = HazardEvent("hurricane", cat, yr, damage, loss, fatalities, max(1, int(cat * 0.5)))
                                year_events.append(event)
                    elif hz == "solar_storm":
                        rate = dist.get("annual_rate_major", 0.5)
                        if rng.random() < rate:
                            kp = max(1, min(9, rng.normal(dist["kp_mean"], dist["kp_std"])))
                            if kp >= 7:
                                damage = min(1.0, (kp / 9.0) ** 2 * 0.15 * (1.5 - current_resilience * 0.5))
                                loss = damage * exposure_billion * 0.3
                                event = HazardEvent("solar_storm", kp, yr, damage, loss, 0, 2)
                                year_events.append(event)
                    elif hz == "flood":
                        major_rate = dist.get("annual_rate_major", 0.02)
                        if rng.random() < major_rate:
                            depth = max(0.1, rng.normal(dist["depth_m_mean"], dist["depth_m_std"]))
                            damage = min(1.0, depth * 0.2 * (1.5 - current_resilience * 0.5))
                            loss = damage * exposure_billion * 0.2
                            fatalities = int(depth * 50 * rng.uniform(0.5, 2.0))
                            event = HazardEvent("flood", depth, yr, damage, loss, fatalities, max(1, int(depth)))
                            year_events.append(event)
                    elif hz == "drought":
                        rate = dist.get("annual_rate", 0.08)
                        if rng.random() < rate:
                            dur = max(1, rng.normal(dist["duration_months_mean"], dist["duration_months_std"]))
                            sev = min(1.0, max(0.0, rng.normal(dist["severity_mean"], dist["severity_std"])))
                            damage = sev * 0.1 * (1.5 - current_resilience * 0.5)
                            loss = damage * exposure_billion * 0.05
                            event = HazardEvent("drought", sev, yr, damage, loss, 0, max(1, int(dur / 12)))
                            year_events.append(event)
                    else:
                        n_events = rng.poisson(annual_rate)
                        for _ in range(n_events):
                            mag = rng.normal(dist.get("magnitude_mean", 5.5), dist.get("magnitude_std", 0.8))
                            if mag >= 5.0:
                                damage = min(1.0, (mag / 9.0) ** 3 * 0.2 * (1.5 - current_resilience * 0.5))
                                loss = damage * exposure_billion * 0.1
                                if hz == "earthquake" and mag >= 7:
                                    fatalities = int(10 ** (mag - 5) * rng.uniform(0.1, 1.0))
                                elif hz == "volcanic_eruption":
                                    vei = max(1, min(8, int(round(mag))))
                                    fatalities = int(10 ** (vei - 2) * rng.uniform(0.01, 0.5))
                                else:
                                    fatalities = int(loss * rng.uniform(0, 10))
                                event = HazardEvent(hz, mag, yr, damage, loss, fatalities, max(1, int(mag / 3)))
                                year_events.append(event)

                if interaction_enabled and len(year_events) >= 2:
                    max_damage_evt = max(year_events, key=lambda e: e.damage_fraction)
                    for evt in year_events:
                        if evt is not max_damage_evt:
                            compound = min(1.5, 1.0 + max_damage_evt.damage_fraction * 0.3)
                            evt.damage_fraction = min(1.0, evt.damage_fraction * compound)
                            evt.economic_loss_billion = evt.economic_loss_billion * compound

                year_loss = sum(e.economic_loss_billion for e in year_events)
                year_damage = min(1.0, sum(e.damage_fraction for e in year_events))
                annual_losses[yr] = year_loss

                current_resilience = max(0.2, current_resilience - year_damage * 0.05 + 0.01)

                sim_events.extend(year_events)

            total_loss = float(np.sum(annual_losses))
            max_annual = float(np.max(annual_losses)) if len(annual_losses) > 0 else 0
            total_losses.append(total_loss)
            max_annual_losses.append(max_annual)
            all_events.append(sim_events)

            trajectory = np.zeros(years)
            for e in sim_events:
                yr_idx = min(e.year, years - 1)
                for r in range(min(e.recovery_years, years - yr_idx)):
                    trajectory[yr_idx + r] += e.damage_fraction * (1.0 - r / max(e.recovery_years, 1)) * 0.1
            recovery_trajectories.append(trajectory)

        total_losses_arr = np.array(total_losses)
        max_annual_arr = np.array(max_annual_losses)

        event_counts = {}
        for sim_events in all_events:
            for e in sim_events:
                event_counts[e.hazard_type] = event_counts.get(e.hazard_type, 0) + 1

        mean_trajectory = np.mean(recovery_trajectories, axis=0)

        var_95 = float(np.percentile(total_losses_arr, 5))
        cvar_95 = float(np.mean(total_losses_arr[total_losses_arr <= var_95])) if np.sum(total_losses_arr <= var_95) > 0 else 0.0

        return sanitize_nan({
            "years": years,
            "n_simulations": n_simulations,
            "exposure_billion": exposure_billion,
            "hazards_modeled": hazards,
            "mean_total_loss_billion": round(float(np.mean(total_losses_arr)), 2),
            "median_total_loss_billion": round(float(np.median(total_losses_arr)), 2),
            "p95_total_loss_billion": round(float(np.percentile(total_losses_arr, 95)), 2),
            "max_annual_loss_mean_billion": round(float(np.mean(max_annual_arr)), 2),
            "max_annual_loss_p95_billion": round(float(np.percentile(max_annual_arr, 95)), 2),
            "var_95_billion": round(var_95, 2),
            "cvar_95_billion": round(float(np.mean(total_losses_arr[total_losses_arr <= float(np.percentile(total_losses_arr, 5))])), 2),
            "total_events_by_type": {k: int(v) for k, v in sorted(event_counts.items(), key=lambda x: x[1], reverse=True)},
            "interaction_enabled": interaction_enabled,
            "mean_recovery_trajectory": [round(float(v), 4) for v in mean_trajectory[::max(1, years // 20)]],
        })

    def earthquake_scenarios(self, magnitude_range: tuple = (6.0, 9.0),
                             n_steps: int = 7,
                             n_simulations: int = 200,
                             exposure_billion: float = 100.0) -> dict:
        rng = np.random.RandomState(self.seed)
        magnitudes = np.linspace(magnitude_range[0], magnitude_range[1], n_steps)
        scenarios = []

        for mag in magnitudes:
            losses = []
            fatalities_list = []
            for _ in range(n_simulations):
                intensity = min(12, max(1, (mag - 1.5) * 1.5 + rng.normal(0, 0.5)))
                damage = min(1.0, (intensity / 12) ** 2 * 0.4 * rng.uniform(0.5, 1.5))
                loss = damage * exposure_billion
                fatalities = int(10 ** (mag - 5) * rng.uniform(0.1, 2.0))
                losses.append(loss)
                fatalities_list.append(fatalities)

            scenarios.append({
                "magnitude": round(float(mag), 1),
                "mean_loss_billion": round(float(np.mean(losses)), 2),
                "median_loss_billion": round(float(np.median(losses)), 2),
                "p95_loss_billion": round(float(np.percentile(losses, 95)), 2),
                "mean_fatalities": int(float(np.mean(fatalities_list))),
                "p95_fatalities": int(float(np.percentile(fatalities_list, 95))),
                "loss_std_billion": round(float(np.std(losses)), 2),
            })

        return sanitize_nan({
            "hazard": "earthquake",
            "scenarios": scenarios,
        })

    def hurricane_season(self, n_simulations: int = 500,
                         el_nino_factor: float = 1.0,
                         climate_warming_factor: float = 1.1) -> dict:
        rng = np.random.RandomState(self.seed)
        seasons = []

        for _ in range(n_simulations):
            base_rate = HAZARD_DISTRIBUTIONS["hurricane"]["annual_rate_atlantic"]
            adjusted_rate = base_rate * el_nino_factor * climate_warming_factor
            n_storms = rng.poisson(adjusted_rate)
            landfalls = sum(1 for _ in range(n_storms) if rng.random() < HAZARD_DISTRIBUTIONS["hurricane"]["landfall_probability"])
            categories = [max(1, min(5, int(round(rng.normal(HAZARD_DISTRIBUTIONS["hurricane"]["category_mean"], HAZARD_DISTRIBUTIONS["hurricane"]["category_std"]))))) for _ in range(landfalls)]
            major = sum(1 for c in categories if c >= 3)
            total_loss = sum(c ** 2 * 5 * rng.uniform(0.5, 2.0) for c in categories)
            seasons.append({
                "n_storms": n_storms,
                "landfalls": landfalls,
                "major_hurricanes": major,
                "total_loss_billion": round(total_loss, 1),
            })

        n_storms_all = [s["n_storms"] for s in seasons]
        landfalls_all = [s["landfalls"] for s in seasons]
        major_all = [s["major_hurricanes"] for s in seasons]
        losses_all = [s["total_loss_billion"] for s in seasons]

        return sanitize_nan({
            "el_nino_factor": el_nino_factor,
            "climate_warming_factor": climate_warming_factor,
            "mean_storms": round(float(np.mean(n_storms_all)), 1),
            "mean_landfalls": round(float(np.mean(landfalls_all)), 1),
            "mean_major": round(float(np.mean(major_all)), 1),
            "mean_loss_billion": round(float(np.mean(losses_all)), 1),
            "p95_loss_billion": round(float(np.percentile(losses_all, 95)), 1),
            "prob_major_landfall": round(float(np.mean([1 if m > 0 else 0 for m in major_all])), 3),
            "prob_cat5_landfall": round(float(np.mean([1 if any(s["major_hurricanes"] > 0 for s in seasons) else 0 for _ in [0]])), 3),
        })

    def tsunami_scenarios(self, trigger_magnitude_range: tuple = (7.5, 9.2),
                           n_steps: int = 5,
                           n_simulations: int = 200,
                           coastal_exposure_billion: float = 50.0) -> dict:
        rng = np.random.RandomState(self.seed)
        magnitudes = np.linspace(trigger_magnitude_range[0], trigger_magnitude_range[1], n_steps)
        scenarios = []
        for mag in magnitudes:
            losses = []
            fatalities_list = []
            for _ in range(n_simulations):
                dist = HAZARD_DISTRIBUTIONS["tsunami"]
                height = max(0.5, rng.normal(dist["height_m_mean"], dist["height_m_std"]) * (mag / 8.0) ** 1.5)
                damage = min(1.0, (height / 20.0) * rng.uniform(0.3, 1.0))
                loss = damage * coastal_exposure_billion
                runup = height * rng.uniform(1.0, 3.0)
                fatalities = int(height * 200 * rng.uniform(0.1, 3.0))
                losses.append(loss)
                fatalities_list.append(fatalities)
            scenarios.append({
                "trigger_magnitude": round(float(mag), 1),
                "mean_wave_height_m": round(float(np.mean([max(0.5, rng.normal(dist["height_m_mean"], dist["height_m_std"]) * (mag / 8.0) ** 1.5) for _ in range(n_simulations)])), 1),
                "mean_loss_billion": round(float(np.mean(losses)), 2),
                "p95_loss_billion": round(float(np.percentile(losses, 95)), 2),
                "mean_fatalities": int(float(np.mean(fatalities_list))),
                "p95_fatalities": int(float(np.percentile(fatalities_list, 95))),
            })
        return sanitize_nan({"hazard": "tsunami", "scenarios": scenarios})

    def volcanic_winter_portfolio(self, vei_min: int = 5,
                                   years: int = 50,
                                   n_simulations: int = 500,
                                   global_gdp_trillion: float = 105.0) -> dict:
        rng = np.random.RandomState(self.seed)
        results = []
        for _ in range(n_simulations):
            annual_rate = 0.003 if vei_min >= 6 else 0.01
            triggered = rng.random() < (1.0 - (1.0 - annual_rate) ** years)
            if triggered:
                vei = max(vei_min, min(8, int(rng.normal(vei_min + 0.5, 0.5))))
                cooling_c = rng.uniform(0.3, 1.5) * (vei - 4) ** 1.2
                duration_years = max(1, int(rng.exponential(2) * (vei - 3)))
                gdp_impact_pct = cooling_c * rng.uniform(0.5, 2.0)
                crop_loss_pct = cooling_c * rng.uniform(1.0, 5.0)
                total_loss = global_gdp_trillion * (gdp_impact_pct / 100.0) * duration_years * 0.3
                recovery_years = max(1, int(duration_years * 1.5))
                results.append({
                    "triggered": True,
                    "vei": vei,
                    "cooling_c": round(float(cooling_c), 2),
                    "duration_years": duration_years,
                    "gdp_impact_pct": round(float(gdp_impact_pct), 2),
                    "crop_loss_pct": round(float(crop_loss_pct), 2),
                    "total_loss_trillion": round(float(total_loss), 2),
                    "recovery_years": recovery_years,
                })
            else:
                results.append({"triggered": False, "vei": 0, "total_loss_trillion": 0.0})
        triggered_results = [r for r in results if r["triggered"]]
        n_triggered = len(triggered_results)
        if triggered_results:
            mean_vei = float(np.mean([r["vei"] for r in triggered_results]))
            mean_cooling = float(np.mean([r["cooling_c"] for r in triggered_results]))
            mean_loss = float(np.mean([r["total_loss_trillion"] for r in triggered_results]))
            p95_loss = float(np.percentile([r["total_loss_trillion"] for r in triggered_results], 95))
            mean_crop = float(np.mean([r["crop_loss_pct"] for r in triggered_results]))
        else:
            mean_vei = 0
            mean_cooling = 0
            mean_loss = 0
            p95_loss = 0
            mean_crop = 0
        period_prob = 1.0 - (1.0 - (0.003 if vei_min >= 6 else 0.01)) ** years
        return sanitize_nan({
            "hazard": "volcanic_winter",
            "vei_threshold": vei_min,
            f"period_{years}yr_probability": round(period_prob, 4),
            "n_simulations": n_simulations,
            "triggered_count": n_triggered,
            "mean_vei_given_trigger": round(mean_vei, 1),
            "mean_cooling_c": round(mean_cooling, 2),
            "mean_gdp_loss_trillion": round(mean_loss, 2),
            "p95_gdp_loss_trillion": round(p95_loss, 2),
            "mean_crop_loss_pct": round(mean_crop, 1),
        })

    def drought_portfolio(self, years: int = 50,
                           n_simulations: int = 500,
                           agricultural_gdp_billion: float = 4000.0) -> dict:
        rng = np.random.RandomState(self.seed)
        dist = HAZARD_DISTRIBUTIONS["drought"]
        results = []
        for _ in range(n_simulations):
            annual_losses = []
            for yr in range(years):
                if rng.random() < dist["annual_rate"]:
                    dur_months = max(1, rng.normal(dist["duration_months_mean"], dist["duration_months_std"]))
                    sev = min(1.0, max(0.0, rng.normal(dist["severity_mean"], dist["severity_std"])))
                    climate_factor = 1.0 + yr / years * 0.5
                    adjusted_sev = min(1.0, sev * climate_factor)
                    loss = adjusted_sev * agricultural_gdp_billion * 0.02 * (dur_months / 12.0)
                    annual_losses.append(loss)
                else:
                    annual_losses.append(0)
            total_loss = sum(annual_losses)
            n_drought_years = sum(1 for l in annual_losses if l > 0)
            max_annual = max(annual_losses) if annual_losses else 0
            results.append({"total_loss_billion": total_loss, "n_drought_years": n_drought_years, "max_annual_billion": max_annual})
        total_losses = [r["total_loss_billion"] for r in results]
        drought_years = [r["n_drought_years"] for r in results]
        max_annuals = [r["max_annual_billion"] for r in results]
        return sanitize_nan({
            "hazard": "drought",
            "years": years,
            "mean_total_loss_billion": round(float(np.mean(total_losses)), 1),
            "p95_total_loss_billion": round(float(np.percentile(total_losses, 95)), 1),
            "mean_drought_years": round(float(np.mean(drought_years)), 1),
            "mean_max_annual_billion": round(float(np.mean(max_annuals)), 1),
            "climate_warming_factor": 1.5,
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