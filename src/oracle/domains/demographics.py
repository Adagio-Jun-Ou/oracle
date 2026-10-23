import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from oracle.core.types import Domain, Prediction, ConfidenceLevel


@dataclass
class DemographicsState:
    total_population_m: float = 8100.0
    birth_rate_per_1000: float = 18.0
    death_rate_per_1000: float = 8.0
    net_migration_per_1000: float = 3.0
    median_age: float = 30.5
    dependency_ratio: float = 0.54
    urbanization_rate: float = 0.57
    female_labor_force_pct: float = 0.47
    gdp_per_capita: float = 12500.0
    education_index: float = 0.65
    health_expenditure_pct: float = 0.10
    fertility_rate: float = 2.3
    life_expectancy: float = 73.0
    elderly_pct: float = 0.10
    youth_pct: float = 0.26
    timestamp: date = field(default_factory=date.today)


DEMOGRAPHIC_TRANSITION_STAGES = {
    "pre_transition": {"birth_rate": (30, 50), "death_rate": (20, 40), "growth": (0.01, 0.02)},
    "early_transition": {"birth_rate": (25, 35), "death_rate": (10, 20), "growth": (0.015, 0.03)},
    "late_transition": {"birth_rate": (15, 25), "death_rate": (6, 10), "growth": (0.005, 0.015)},
    "post_transition": {"birth_rate": (8, 15), "death_rate": (8, 12), "growth": (-0.005, 0.005)},
}


class DemographicsModel:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def classify_stage(self, state: DemographicsState) -> str:
        br = state.birth_rate_per_1000
        dr = state.death_rate_per_1000
        if dr > 25:
            return "pre_transition"
        elif br > 25 or (br > 20 and dr > 10):
            return "early_transition"
        elif br > 15 or dr > 10:
            return "late_transition"
        else:
            return "post_transition"

    def project_population(
        self, state: DemographicsState, years: int = 30, n_simulations: int = 1000,
    ) -> dict:
        scenarios = {
            "high_growth": {"birth_rate_change": 0.02, "death_rate_change": -0.01, "migration_change": 0.001},
            "baseline": {"birth_rate_change": 0.005, "death_rate_change": -0.005, "migration_change": 0.0},
            "low_growth": {"birth_rate_change": -0.01, "death_rate_change": 0.0, "migration_change": -0.001},
            "decline": {"birth_rate_change": -0.02, "death_rate_change": 0.005, "migration_change": -0.002},
        }

        results = {}
        for scenario_name, params in scenarios.items():
            populations = np.zeros((n_simulations, years + 1))
            birth_rates = np.zeros((n_simulations, years + 1))
            death_rates = np.zeros((n_simulations, years + 1))
            median_ages = np.zeros((n_simulations, years + 1))
            dependency_ratios = np.zeros((n_simulations, years + 1))

            for sim in range(n_simulations):
                pop = state.total_population_m
                br = state.birth_rate_per_1000 / 1000
                dr = state.death_rate_per_1000 / 1000
                mr = state.net_migration_per_1000 / 1000
                ma = state.median_age
                dep = state.dependency_ratio

                populations[sim, 0] = pop
                birth_rates[sim, 0] = br * 1000
                death_rates[sim, 0] = dr * 1000
                median_ages[sim, 0] = ma
                dependency_ratios[sim, 0] = dep

                for yr in range(1, years + 1):
                    aging = 0.1 + params["birth_rate_change"] * 5
                    ma += aging * 0.1 + self.rng.normal(0, 0.05)
                    br += params["birth_rate_change"] / 1000 + self.rng.normal(0, 0.002 / 1000)
                    dr += params["death_rate_change"] / 1000 + self.rng.normal(0, 0.001 / 1000)
                    mr_change = params["migration_change"] / 1000 + self.rng.normal(0, 0.001 / 1000)

                    br = max(0.005, min(0.05, br))
                    dr = max(0.005, min(0.03, dr))

                    growth = br - dr + mr + mr_change
                    pop = pop * (1 + growth)

                    elderly_pct = min(0.40, (ma - 25) / 100 + state.elderly_pct)
                    youth_pct = max(0.10, state.youth_pct - (ma - 30) / 300)
                    dep = (youth_pct + elderly_pct) / (1 - youth_pct - elderly_pct + 0.01)

                    populations[sim, yr] = pop
                    birth_rates[sim, yr] = br * 1000
                    death_rates[sim, yr] = dr * 1000
                    median_ages[sim, yr] = ma
                    dependency_ratios[sim, yr] = dep

            results[scenario_name] = {
                "population": populations,
                "birth_rate": birth_rates,
                "death_rate": death_rates,
                "median_age": median_ages,
                "dependency_ratio": dependency_ratios,
                "pop_final_median": float(np.median(populations[:, -1])),
                "pop_final_p05": float(np.percentile(populations[:, -1], 5)),
                "pop_final_p95": float(np.percentile(populations[:, -1], 95)),
                "pop_peak_median": float(np.max(np.median(populations, axis=0))),
                "median_age_final": float(np.median(median_ages[:, -1])),
                "dependency_final": float(np.median(dependency_ratios[:, -1])),
            }

        return results

    def aging_dividend(
        self, state: DemographicsState, years: int = 20,
    ) -> dict:
        stage = self.classify_stage(state)
        if stage == "pre_transition":
            dividend_years = 0
            dividend_start = years + 10
        elif stage == "early_transition":
            dividend_years = int(years * 0.4)
            dividend_start = 5
        elif stage == "late_transition":
            dividend_years = int(years * 0.6)
            dividend_start = 0
        else:
            dividend_years = 0
            dividend_start = 0

        working_age_pct = max(0.55, 1 - state.dependency_ratio)
        gdp_bonus = 0
        if dividend_years > 0:
            gdp_bonus = working_age_pct * (1 - state.elderly_pct) * 0.5

        peak_dependency = state.dependency_ratio * (1 - gdp_bonus * 0.3)
        aging_burden = state.elderly_pct * (1 + state.fertility_rate / 2.1 - 1)

        return {
            "stage": stage,
            "demographic_dividend_years": dividend_years,
            "dividend_starts_in": dividend_start,
            "working_age_pct": float(working_age_pct),
            "gdp_growth_bonus_pct": float(gdp_bonus * 100),
            "peak_dependency_ratio": float(peak_dependency),
            "aging_burden_index": float(min(aging_burden, 1)),
            "pension_sustainability": float(max(0, 1 - state.elderly_pct * 3)),
            "healthcare_pressure": float(min(1, state.elderly_pct * 5 + (1 - state.health_expenditure_pct) * 0.3)),
        }

    def predict(
        self, state: DemographicsState, target_date: date,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    ) -> list[Prediction]:
        years_ahead = max(0.5, (target_date - state.timestamp).days / 365.25)
        z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645,
             ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence]

        br_change = -0.05 * years_ahead
        dr_change = -0.01 * years_ahead
        aging = 0.12 * years_ahead

        metrics = [
            ("population_billion", state.total_population_m / 1000 * (1 + (state.birth_rate_per_1000 - state.death_rate_per_1000) / 1000) ** years_ahead,
             state.total_population_m / 1000 * 0.02 * years_ahead, Domain.DEMOGRAPHICS),
            ("birth_rate_per_1000", max(8, state.birth_rate_per_1000 + br_change),
             3 * years_ahead / 10, Domain.DEMOGRAPHICS),
            ("fertility_rate", max(1.0, state.fertility_rate - 0.01 * years_ahead),
             0.2, Domain.DEMOGRAPHICS),
            ("life_expectancy", min(90, state.life_expectancy + 0.15 * years_ahead),
             2 * years_ahead ** 0.5, Domain.HEALTH),
            ("median_age", min(55, state.median_age + aging),
             2 * years_ahead ** 0.3, Domain.DEMOGRAPHICS),
            ("dependency_ratio", min(1, state.dependency_ratio + 0.01 * years_ahead * state.elderly_pct * 5),
             0.05 * years_ahead, Domain.ECONOMICS),
            ("urbanization_rate", min(0.95, state.urbanization_rate + 0.005 * years_ahead),
             0.03, Domain.SOCIAL),
        ]

        return [
            Prediction(
                domain=domain, variable=f"demographic_{name}",
                value=round(float(val), 4),
                confidence_interval=(round(float(max(0, val - z * std)), 4), round(float(val + z * std), 4)),
                confidence_level=confidence, timestamp=datetime.now(),
                target_date=target_date, model_name="demographics_model",
                weights={"cohort": 0.4, "transition": 0.3, "momentum": 0.3},
            ) for name, val, std, domain in metrics
        ]

    @staticmethod
    def migration_pressure_model(
        origin_population_m: float = 50.0,
        origin_gdp_per_capita: float = 5000,
        destination_gdp_per_capita: float = 50000,
        origin_conflict_index: float = 0.3,
        climate_vulnerability: float = 0.2,
        youth_bulge_pct: float = 0.25,
        urbanization_rate: float = 0.55,
        water_stress: float = 0.3,
        food_security_index: float = 0.6,
        unemployment_rate: float = 0.1,
        political_instability: float = 0.3,
        border_permeability: float = 0.5,
        diaspora_network: float = 0.1,
        years: int = 20,
    ) -> dict:
        economic_push = (
            0.25 * (1 - min(origin_gdp_per_capita / 50000, 1.0)) +
            0.20 * unemployment_rate +
            0.15 * (1 - food_security_index) +
            0.10 * water_stress
        )
        environmental_push = (
            0.35 * climate_vulnerability +
            0.25 * water_stress +
            0.20 * (1 - food_security_index) +
            0.20 * climate_vulnerability * (1 - food_security_index)
        )
        social_push = (
            0.30 * origin_conflict_index +
            0.25 * political_instability +
            0.20 * youth_bulge_pct +
            0.15 * (1 - min(urbanization_rate, 1.0)) +
            0.10 * max(0, origin_conflict_index - 0.3)
        )
        total_push = economic_push * 0.4 + environmental_push * 0.25 + social_push * 0.35
        wage_gain = np.log(max(destination_gdp_per_capita, 100) / max(origin_gdp_per_capita, 100))
        pull_factor = (
            0.35 * (wage_gain / 3) +
            0.25 * border_permeability +
            0.20 * diaspora_network +
            0.20 * (1 - political_instability * 0.5)
        )
        net_pressure = total_push * 0.6 + pull_factor * 0.4
        base_annual_outflow = origin_population_m * 1e6 * net_pressure * 0.002
        yearly_projection = []
        cumulative = 0
        for yr in range(years + 1):
            climate_factor = 1 + climate_vulnerability * yr * 0.03
            conflict_factor = 1 + origin_conflict_index * max(0, yr - 5) * 0.01
            youth_factor = 1 - max(0, youth_bulge_pct - 0.05 * yr) * 0.5
            adjusted_outflow = base_annual_outflow * climate_factor * conflict_factor * youth_factor
            cumulative += adjusted_outflow
            pressure_level = "mass_exodus" if net_pressure * climate_factor > 0.7 else "high" if net_pressure * climate_factor > 0.5 else "moderate" if net_pressure * climate_factor > 0.3 else "low"
            yearly_projection.append({
                "year": yr,
                "annual_outflow": round(float(adjusted_outflow), 0),
                "cumulative_outflow": round(float(cumulative), 0),
                "cumulative_as_pct_origin": round(float(cumulative / (origin_population_m * 1e6) * 100), 2),
                "pressure_level": pressure_level,
            })
        destination_impacts = {
            "labor_market_pressure": round(float(min(net_pressure * 1.2, 0.8)), 3),
            "housing_demand": round(float(min(net_pressure * 1.5, 0.9)), 3),
            "cultural_integration_stress": round(float(min(net_pressure * 0.7 * (1 - diaspora_network), 0.7)), 3),
            "fiscal_burden": round(float(min(net_pressure * 0.6, 0.6)), 3),
            "gdp_potential_boost": round(float(min(net_pressure * 0.3, 0.4)), 3),
        }
        if net_pressure > 0.6:
            overall = "critical"
        elif net_pressure > 0.45:
            overall = "high"
        elif net_pressure > 0.3:
            overall = "moderate"
        else:
            overall = "low"
        return {
            "origin_population_m": origin_population_m,
            "total_push_factor": round(float(total_push), 4),
            "economic_push": round(float(economic_push), 4),
            "environmental_push": round(float(environmental_push), 4),
            "social_push": round(float(social_push), 4),
            "pull_factor": round(float(pull_factor), 4),
            "net_pressure": round(float(net_pressure), 4),
            "overall_level": overall,
            "wage_gap_log": round(float(wage_gain), 3),
            "base_annual_outflow": round(float(base_annual_outflow), 0),
            "20yr_cumulative_outflow": round(float(cumulative), 0),
            "20yr_pct_of_origin": round(float(cumulative / (origin_population_m * 1e6) * 100), 2),
            "yearly_projection": yearly_projection[::5],
            "destination_impacts": destination_impacts,
            "key_drivers": sorted([
                ("economic", economic_push),
                ("environmental", environmental_push),
                ("social", social_push),
            ], key=lambda x: x[1], reverse=True),
            "recommendations": [
                f"Primary driver: {['economic', 'environmental', 'social'][[economic_push, environmental_push, social_push].index(max(economic_push, environmental_push, social_push))]} factors",
                f"Climate amplification: {'High' if climate_vulnerability > 0.5 else 'Moderate' if climate_vulnerability > 0.3 else 'Low'} over {years}yr horizon",
                f"Youth bulge {'declining' if youth_bulge_pct > 0.2 else 'stable'} — {'expect reduced pressure' if youth_bulge_pct > 0.2 else 'stable migration pressure'}",
            ],
        }