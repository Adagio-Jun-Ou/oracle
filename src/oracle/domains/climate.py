import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from oracle.core.types import Domain, Prediction, ConfidenceLevel


@dataclass
class ClimateScenario:
    name: str
    temp_increase_by_2100: float
    description: str = ""
    assumptions: dict = field(default_factory=dict)


@dataclass
class ClimateState:
    global_temp_anomaly_c: float = 1.2
    co2_ppm: float = 420.0
    sea_level_rise_mm: float = 100.0
    arctic_ice_extent_mkm: float = 3.5
    renewable_share: float = 0.30
    carbon_price_usd: float = 50.0
    methane_ppb: float = 1900.0
    deforestation_rate: float = 0.01
    ocean_ph: float = 8.1
    extreme_weather_index: float = 0.4
    net_zero_pledge_coverage: float = 0.65
    fossil_fuel_subsidies_b: float = 577.0
    timestamp: date = field(default_factory=date.today)


RCP_SCENARIOS = {
    "rcp_1_9": ClimateScenario("RCP 1.9 (SSP1-1.9)", 1.5,
        "Paris Agreement 1.5C target pathway. Extreme mitigation.",
        {"co2_2100": 350, "mitigation_cost": "high", "probability": 0.05}),
    "rcp_2_6": ClimateScenario("RCP 2.6 (SSP1-2.6)", 2.0,
        "Paris Agreement 2C pathway. Strong mitigation.",
        {"co2_2100": 400, "mitigation_cost": "moderate", "probability": 0.15}),
    "rcp_4_5": ClimateScenario("RCP 4.5 (SSP2-4.5)", 2.8,
        "Moderate mitigation. Most likely median pathway.",
        {"co2_2100": 538, "mitigation_cost": "moderate", "probability": 0.35}),
    "rcp_6_0": ClimateScenario("RCP 6.0 (SSP4-6.0)", 3.6,
        "Weak mitigation. Continued high emissions.",
        {"co2_2100": 670, "mitigation_cost": "low", "probability": 0.30}),
    "rcp_8_5": ClimateScenario("RCP 8.5 (SSP5-8.5)", 4.4,
        "Business as usual. Very high emissions.",
        {"co2_2100": 936, "mitigation_cost": "minimal", "probability": 0.15}),
}

CARBON_BUDGET_REMAINING_GT = 400.0
WARMING_PER_DOUBLING_CO2 = 3.0
OCEAN_HEAT_CAPACITY_YR = 40.0


class ClimateModel:
    def __init__(self, climate_sensitivity: float = 3.0, seed: int = 42):
        self.climate_sensitivity = climate_sensitivity
        self.rng = np.random.default_rng(seed)
        self.scenarios = RCP_SCENARIOS

    def project_temperature(
        self, state: ClimateState, years_ahead: int = 30, n_simulations: int = 1000,
    ) -> dict:
        pathways = {
            "aggressive_mitigation": {"co2_growth": 0.0, "methane_growth": -0.005, "renewable_ramp": 0.03},
            "moderate_action": {"co2_growth": 0.005, "methane_growth": 0.001, "renewable_ramp": 0.02},
            "current_policies": {"co2_growth": 0.012, "methane_growth": 0.003, "renewable_ramp": 0.015},
            "high_emissions": {"co2_growth": 0.02, "methane_growth": 0.005, "renewable_ramp": 0.01},
        }

        results = {}
        for pathway_name, params in pathways.items():
            temps = np.zeros((n_simulations, years_ahead + 1))
            co2 = np.zeros((n_simulations, years_ahead + 1))
            sea = np.zeros((n_simulations, years_ahead + 1))

            for sim in range(n_simulations):
                t = 0.0
                co2_level = state.co2_ppm
                sea_level = state.sea_level_rise_mm
                co2_forcing_base = 5.35

                for yr in range(years_ahead + 1):
                    temps[sim, yr] = state.global_temp_anomaly_c + t
                    co2[sim, yr] = co2_level
                    sea[sim, yr] = sea_level

                    co2_growth_noise = self.rng.normal(params["co2_growth"], 0.003)
                    co2_level *= (1 + co2_growth_noise)
                    co2_forcing = co2_forcing_base * np.log(co2_level / 280.0)
                    temp_response = co2_forcing / (self.climate_sensitivity * 5.35 / 3.0)
                    ocean_lag = 1.0 - np.exp(-1.0 / OCEAN_HEAT_CAPACITY_YR)
                    t += temp_response * ocean_lag * 0.01 + self.rng.normal(0, 0.005)
                    sea_level += 3.3 * (1 + t * 0.5) + self.rng.normal(0, 0.2)

            results[pathway_name] = {
                "temperature": temps,
                "co2": co2,
                "sea_level": sea,
                "temp_2080_median": float(np.median(temps[:, -1])),
                "temp_2080_p95": float(np.percentile(temps[:, -1], 95)),
                "temp_2080_p05": float(np.percentile(temps[:, -1], 5)),
                "sea_level_2080_median": float(np.median(sea[:, -1])),
                "sea_level_2080_p95": float(np.percentile(sea[:, -1], 95)),
                "carbon_budget_exceed_prob": float(np.mean(co2[:, -1] > 450)),
            }
        return results

    def carbon_pricing_impact(
        self, carbon_price_usd: float, gdp_trillion: float = 105.0,
        price_elasticity: float = -0.8,
    ) -> dict:
        emission_reduction = 1 - np.exp(price_elasticity * np.log(1 + carbon_price_usd / 50))
        emission_reduction = max(0, min(1, emission_reduction))
        gdp_impact = -0.01 * carbon_price_usd / 50 * (1 - emission_reduction * 0.4)
        investment_shift = carbon_price_usd / 100 * 0.5
        renewable_acceleration = min(0.8, carbon_price_usd / 200 + 0.1)

        return {
            "emission_reduction_pct": float(emission_reduction * 100),
            "gdp_impact_pct": float(gdp_impact * 100),
            "investment_shift_trillion": float(investment_shift * gdp_trillion * 0.05),
            "renewable_share_acceleration": float(renewable_acceleration),
            "carbon_revenue_billion": float(carbon_price_usd * 35 * emission_reduction),
        }

    def tipping_points(self, state: ClimateState) -> dict:
        tipping_points_list = [
            {"name": "Greenland Ice Sheet Collapse", "threshold": 2.7, "current": state.global_temp_anomaly_c,
             "impact": "7m sea level rise", "reversibility": "very_low", "timescale": "1000+ years"},
            {"name": "West Antarctic Ice Sheet", "threshold": 2.0, "current": state.global_temp_anomaly_c,
             "impact": "3-5m sea level rise", "reversibility": "very_low", "timescale": "2000+ years"},
            {"name": "Amazon Rainforest Dieback", "threshold": 3.5, "current": state.global_temp_anomaly_c,
             "impact": "Massive CO2 release, biodiversity loss", "reversibility": "low", "timescale": "50-200 years"},
            {"name": "Permafrost Carbon Release", "threshold": 2.0, "current": state.global_temp_anomaly_c,
             "impact": "100-250 GtC over 50-100 years", "reversibility": "very_low", "timescale": "100+ years"},
            {"name": "Atlantic Meridional Overturning (AMOC)", "threshold": 4.0, "current": state.global_temp_anomaly_c,
             "impact": "European cooling, monsoon disruption", "reversibility": "very_low", "timescale": "100+ years"},
            {"name": "Coral Reef Die-off", "threshold": 1.5, "current": state.global_temp_anomaly_c,
             "impact": "Marine ecosystem collapse, fisheries", "reversibility": "low", "timescale": "10-50 years"},
            {"name": "Boreal Forest Shift", "threshold": 4.0, "current": state.global_temp_anomaly_c,
             "impact": "Albedo change, carbon release", "reversibility": "low", "timescale": "50-100 years"},
            {"name": "Methane Hydrate Destabilization", "threshold": 3.5, "current": state.global_temp_anomaly_c,
             "impact": "Runaway warming potential", "reversibility": "none", "timescale": "50-500 years"},
        ]

        results = []
        for tp in tipping_points_list:
            proximity = (state.global_temp_anomaly_c / tp["threshold"]) if tp["threshold"] > 0 else 99.0
            triggered = state.global_temp_anomaly_c >= tp["threshold"]
            results.append({
                "name": tp["name"],
                "threshold_c": tp["threshold"],
                "proximity": float(min(proximity, 1.5)),
                "triggered": triggered,
                "probability_by_2100": float(min(proximity ** 3, 0.95)) if not triggered else 1.0,
                "impact": tp["impact"],
                "reversibility": tp["reversibility"],
                "timescale": tp["timescale"],
            })

        overall_risk = float(np.mean([r["proximity"] for r in results]))
        return {
            "tipping_points": results,
            "overall_tipping_risk": overall_risk,
            "n_triggered": sum(1 for r in results if r["triggered"]),
            "n_close": sum(1 for r in results if 0.7 < r["proximity"] < 1.0),
        }

    def predict(
        self, state: ClimateState, target_date: date,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    ) -> list[Prediction]:
        years_ahead = max(0.25, (target_date - state.timestamp).days / 365.25)
        z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645,
             ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence]

        co2_growth = 0.012 * (1 - state.renewable_share * 0.5)
        projected_co2 = state.co2_ppm * (1 + co2_growth) ** years_ahead
        projected_temp = state.global_temp_anomaly_c + 0.03 * years_ahead * (1 + projected_co2 / 280 - 1) / 10
        sea_rise = state.sea_level_rise_mm + 3.3 * years_ahead * (1 + projected_temp * 0.2)
        arctic_decline = max(0, state.arctic_ice_extent_mkm - 0.05 * years_ahead * (1 + projected_temp * 0.1))
        ph_decline = state.ocean_ph - 0.002 * years_ahead

        predictions = []
        metrics = [
            ("temperature_anomaly_c", projected_temp, 0.3 * years_ahead, Domain.CLIMATE),
            ("co2_ppm", projected_co2, 10 * years_ahead, Domain.CLIMATE),
            ("sea_level_rise_mm", sea_rise, 30 * years_ahead, Domain.CLIMATE),
            ("arctic_ice_mkm", arctic_decline, 0.5 * years_ahead, Domain.CLIMATE),
            ("extreme_weather_index", min(1, state.extreme_weather_index + 0.01 * years_ahead * projected_temp),
             0.1 * years_ahead, Domain.CLIMATE),
            ("renewable_share_projection", min(1, state.renewable_share + 0.01 * years_ahead * (1 + state.carbon_price_usd / 100)),
             0.05 * years_ahead, Domain.ENERGY),
            ("carbon_budget_remaining_gt", max(0, CARBON_BUDGET_REMAINING_GT - (projected_co2 - 280) * 2.12),
             50 * years_ahead, Domain.CLIMATE),
            ("ocean_ph", ph_decline, 0.01 * years_ahead, Domain.CLIMATE),
        ]

        for name, value, std, domain in metrics:
            ci = (max(0, value - z * std), value + z * std)
            predictions.append(Prediction(
                domain=domain, variable=f"climate_{name}", value=round(float(value), 4),
                confidence_interval=(round(float(ci[0]), 4), round(float(ci[1]), 4)),
                confidence_level=confidence, timestamp=datetime.now(),
                target_date=target_date, model_name="climate_model",
                weights={"rcp_pathway": 0.4, "energy_balance": 0.3, "carbon_cycle": 0.3},
                signals_used=["co2", "temperature", "sea_level", "arctic_ice"],
            ))

        return predictions

    @staticmethod
    def sea_level_projector(
        current_temp_anomaly_c: float = 1.2,
        current_sea_level_mm: float = 100.0,
        thermal_expansion_rate: float = 1.3,
        ice_contribution_rate: float = 1.1,
        acceleration_factor: float = 0.04,
        years: int = 80,
        n_simulations: int = 1000,
        seed: int = 42,
    ) -> dict:
        np.random.seed(seed)
        rcp_params = {
            "rcp_1_9": {"temp_2100": 1.5, "prob": 0.05},
            "rcp_2_6": {"temp_2100": 2.0, "prob": 0.15},
            "rcp_4_5": {"temp_2100": 2.8, "prob": 0.35},
            "rcp_6_0": {"temp_2100": 3.6, "prob": 0.30},
            "rcp_8_5": {"temp_2100": 4.4, "prob": 0.15},
        }
        scenarios = {}
        for rcp_name, params in rcp_params.items():
            target_temp = params["temp_2100"]
            temp_rate = max(0.001, (target_temp - current_temp_anomaly_c) / years)
            projections = np.zeros((n_simulations, years + 1))
            for sim in range(n_simulations):
                t = current_temp_anomaly_c
                sea = current_sea_level_mm
                for yr in range(years + 1):
                    projections[sim, yr] = sea
                    temp_increment = temp_rate + np.random.normal(0, 0.005)
                    t += temp_increment
                    thermal = thermal_expansion_rate * (1 + 0.002 * t)
                    ice = ice_contribution_rate * (1 + acceleration_factor * max(0, t - 1.5))
                    total_rate = thermal + ice + np.random.normal(0, 0.3)
                    sea += total_rate
            median_curve = np.median(projections, axis=0)
            scenarios[rcp_name] = {
                "temp_target_2100": target_temp,
                "sea_level_2050_median_mm": round(float(median_curve[25]), 1),
                "sea_level_2100_median_mm": round(float(median_curve[years]), 1),
                "sea_level_2100_p95_mm": round(float(np.percentile(projections[:, years], 95)), 1),
                "sea_level_2100_p05_mm": round(float(np.percentile(projections[:, years], 5)), 1),
                "yearly_median": [round(float(median_curve[i]), 1) for i in range(0, years + 1, 10)],
                "probability_weight": params["prob"],
            }
        weighted_2100 = sum(s["sea_level_2100_median_mm"] * s["probability_weight"] for s in scenarios.values())
        yearly_medians = {rcp: s["yearly_median"] for rcp, s in scenarios.items()}
        time_labels = list(range(0, years + 1, 10))
        ice_sheet_contrib = 0.5 * (current_temp_anomaly_c / 2.0) ** 2
        thermal_contrib_pct = round(min(100, thermal_expansion_rate / (thermal_expansion_rate + ice_contribution_rate + ice_sheet_contrib) * 100), 1)
        return {
            "current_temp_anomaly_c": current_temp_anomaly_c,
            "current_sea_level_mm": current_sea_level_mm,
            "years": years,
            "n_simulations": n_simulations,
            "scenarios": scenarios,
            "weighted_median_2100_mm": round(float(weighted_2100), 1),
            "likely_range_2100_mm": [
                round(float(min(s["sea_level_2100_p05_mm"] for s in scenarios.values())), 1),
                round(float(max(s["sea_level_2100_p95_mm"] for s in scenarios.values())), 1),
            ],
            "thermal_contribution_pct": thermal_contrib_pct,
            "ice_contribution_pct": round(100 - thermal_contrib_pct, 1),
            "yearly_medians_by_scenario": yearly_medians,
            "time_labels": time_labels,
            "impacted_populations_millions": round(float(weighted_2100 / 1000 * 150), 0),
            "coastal_cities_at_risk": round(float(weighted_2100 / 100), 0),
        }