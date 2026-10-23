import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnergyState:
    oil_price_usd: float = 75.0
    natural_gas_price: float = 3.0
    renewable_share_pct: float = 20.0
    nuclear_share_pct: float = 10.0
    coal_share_pct: float = 27.0
    energy_demand_growth: float = 0.02
    carbon_price_usd: float = 25.0
    grid_stability: float = 0.7
    opec_production_mbd: float = 28.0
    strategic_reserve_days: float = 90.0
    energy_intensity: float = 5.0


class EnergyModel:
    def __init__(self):
        self.energy_sources = [
            {"name": "Oil", "share": 0.31, "growth": -0.01, "volatility": 0.25, "co2_intensity": 0.73},
            {"name": "Natural Gas", "share": 0.24, "growth": 0.01, "volatility": 0.20, "co2_intensity": 0.41},
            {"name": "Coal", "share": 0.27, "growth": -0.03, "volatility": 0.15, "co2_intensity": 0.96},
            {"name": "Nuclear", "share": 0.10, "growth": 0.005, "volatility": 0.05, "co2_intensity": 0.0},
            {"name": "Renewables", "share": 0.08, "growth": 0.08, "volatility": 0.10, "co2_intensity": 0.0},
        ]
        self.transition_scenarios = {
            "current_policies": {"renewable_target_2030": 0.25, "carbon_price_trend": 0.02, "fossil_subsidy_change": 0.0},
            "announced_pledges": {"renewable_target_2030": 0.40, "carbon_price_trend": 0.05, "fossil_subsidy_change": -0.03},
            "net_zero_2050": {"renewable_target_2030": 0.60, "carbon_price_trend": 0.10, "fossil_subsidy_change": -0.08},
        }

    def energy_mix_forecast(self, state: Optional[EnergyState] = None, years: int = 10, scenario: str = "current_policies") -> dict:
        if state is None:
            state = EnergyState()
        params = self.transition_scenarios.get(scenario, self.transition_scenarios["current_policies"])
        current_shares = {s["name"]: s["share"] for s in self.energy_sources}
        renewable_share = state.renewable_share_pct / 100.0
        nuclear_share = state.nuclear_share_pct / 100.0
        coal_share = state.coal_share_pct / 100.0
        fossil_share = 1.0 - renewable_share - nuclear_share
        yearly_forecast = []
        r_share = renewable_share
        n_share = nuclear_share
        c_share = coal_share
        for y in range(years + 1):
            r_share = min(r_share + params["renewable_target_2030"] / 10 * r_share * 0.5 + 0.005, 0.85)
            n_share = min(n_share + 0.002, 0.30)
            fossil_share_yr = max(1.0 - r_share - n_share, 0.05)
            c_share = max(c_share * (1 - abs(params["fossil_subsidy_change"])), 0.02)
            oil_gas_share = fossil_share_yr - c_share
            if oil_gas_share < 0.05:
                oil_gas_share = 0.05
                c_share = max(fossil_share_yr - 0.05, 0.01)
            yearly_forecast.append({
                "year": 2025 + y,
                "renewable": round(r_share, 4),
                "nuclear": round(n_share, 4),
                "fossil": round(fossil_share_yr, 4),
                "coal": round(c_share, 4),
                "co2_intensity": round(r_share * 0 + n_share * 0 + c_share * 0.96 + oil_gas_share * 0.55, 4),
            })
        initial_fossil = yearly_forecast[0]["fossil"]
        final_fossil = yearly_forecast[-1]["fossil"]
        fossil_decline_pct = round((initial_fossil - final_fossil) / initial_fossil * 100, 1) if initial_fossil > 0 else 0
        co2_trajectory = [m["co2_intensity"] for m in yearly_forecast]
        current_co2 = co2_trajectory[0]
        final_co2 = co2_trajectory[-1]
        reduction_pct = ((current_co2 - final_co2) / current_co2 * 100) if current_co2 > 0 else 0
        return {
            "scenario": scenario,
            "years": years,
            "yearly_forecast": yearly_forecast,
            "current_co2_intensity": round(current_co2, 4),
            "final_co2_intensity": round(final_co2, 4),
            "co2_reduction_pct": round(reduction_pct, 2),
            "renewable_share_final": round(r_share * 100, 1),
            "fossil_decline_pct": fossil_decline_pct,
            "fossil_fraction_remaining": round((1 - r_share - n_share) * 100, 1),
            "transition_adequacy": round(min(r_share / params["renewable_target_2030"], 1.0), 3),
        }

    def price_volatility_analysis(self, state: Optional[EnergyState] = None, days: int = 252) -> dict:
        if state is None:
            state = EnergyState()
        np.random.seed(42)
        oil_returns = np.random.normal(0.0001, 0.02, days)
        gas_returns = np.random.normal(0.0002, 0.035, days)
        oil_prices = state.oil_price_usd * np.exp(np.cumsum(oil_returns))
        gas_prices = state.natural_gas_price * np.exp(np.cumsum(gas_returns))
        oil_vol = float(np.std(oil_returns) * np.sqrt(252))
        gas_vol = float(np.std(gas_returns) * np.sqrt(252))
        oil_var_95 = float(state.oil_price_usd * oil_vol * 1.645 * np.sqrt(5 / 252))
        gas_var_95 = float(state.natural_gas_price * gas_vol * 1.645 * np.sqrt(5 / 252))
        oil_30d_vol = round(float(np.std(oil_returns[:min(30, days)]) * np.sqrt(252) * 100), 1)
        gas_30d_vol = round(float(np.std(gas_returns[:min(30, days)]) * np.sqrt(252) * 100), 1)
        def vol_label(vol_pct):
            if vol_pct < 15:
                return "low"
            elif vol_pct < 30:
                return "moderate"
            else:
                return "high"
        oil_vol_pct = round(oil_vol * 100, 1)
        gas_vol_pct = round(gas_vol * 100, 1)
        return {
            "volatility": {
                "oil": {
                    "current_price": round(state.oil_price_usd, 2),
                    "annualized_volatility": oil_vol_pct,
                    "price_volatility_30d": round(oil_30d_vol / 100, 4),
                    "volatility_index": round(oil_vol, 3),
                    "volatility_level": vol_label(oil_vol_pct),
                    "var_95_weekly": round(oil_var_95, 2),
                    "price_range_1yr": {
                        "low_10": round(float(oil_prices[np.int64(days * 0.1)]), 2) if days > 10 else round(state.oil_price_usd, 2),
                        "median": round(float(np.median(oil_prices)), 2),
                        "high_90": round(float(oil_prices[np.int64(days * 0.9)]), 2) if days > 10 else round(state.oil_price_usd, 2),
                    },
                },
                "natural_gas": {
                    "current_price": round(state.natural_gas_price, 2),
                    "annualized_volatility": gas_vol_pct,
                    "price_volatility_30d": round(gas_30d_vol / 100, 4),
                    "volatility_index": round(gas_vol, 3),
                    "volatility_level": vol_label(gas_vol_pct),
                    "var_95_weekly": round(gas_var_95, 2),
                },
            },
            "correlation_oil_gas": round(float(np.corrcoef(oil_returns, gas_returns)[0, 1]), 3),
        }

    def energy_security_index(self, state: Optional[EnergyState] = None) -> dict:
        if state is None:
            state = EnergyState()
        import_dependency = max(0, 1.0 - state.grid_stability)
        diversity_score = 0.0
        shares = [state.renewable_share_pct / 100, state.nuclear_share_pct / 100, state.coal_share_pct / 100]
        total = sum(shares) if sum(shares) > 0 else 1.0
        fracs = [s / total for s in shares]
        hhi = sum(f ** 2 for f in fracs)
        diversity_score = max(0, 1.0 - hhi)
        reserve_score = min(state.strategic_reserve_days / 180.0, 1.0)
        renewable_security = min(state.renewable_share_pct / 50.0, 1.0)
        import_score = 1.0 - import_dependency * 0.5
        overall_security = (diversity_score * 0.3 + reserve_score * 0.25 + renewable_security * 0.25 + import_score * 0.2)
        risk_level = "low" if overall_security > 0.7 else "moderate" if overall_security > 0.4 else "high"
        vulnerabilities = []
        if state.strategic_reserve_days < 60:
            vulnerabilities.append("Low strategic petroleum reserve")
        if state.renewable_share_pct < 15:
            vulnerabilities.append("Low renewable penetration")
        if import_dependency > 0.5:
            vulnerabilities.append("High import dependency")
        if state.grid_stability < 0.5:
            vulnerabilities.append("Grid instability risk")
        if state.coal_share_pct > 40:
            vulnerabilities.append("High coal dependency")
        infrastructure_resilience = round(min(state.grid_stability, 1.0), 3)
        demand_management = round(min(1.0 - state.energy_intensity / 15.0, 1.0), 3)
        geopolitical_stability = round(min(import_score, 1.0), 3)
        strategic_reserve = round(reserve_score, 3)
        return {
            "overall_score": round(overall_security, 3),
            "overall_security_index": round(overall_security, 3),
            "security_level": risk_level,
            "risk_level": risk_level,
            "components": {
                "supply_diversity": round(diversity_score, 3),
                "infrastructure_resilience": infrastructure_resilience,
                "demand_management": demand_management,
                "geopolitical_stability": geopolitical_stability,
                "strategic_reserve": strategic_reserve,
                "reserve_adequacy": round(reserve_score, 3),
                "renewable_resilience": round(renewable_security, 3),
                "import_independence": round(import_score, 3),
            },
            "vulnerabilities": vulnerabilities,
            "risks": vulnerabilities,
            "recommendations": self._energy_security_recommendations(overall_security, vulnerabilities),
        }

    def _energy_security_recommendations(self, security: float, vulnerabilities: list) -> list:
        recs = []
        if security < 0.4:
            recs.append("Critical: Diversify energy sources urgently")
        if "Low strategic petroleum reserve" in vulnerabilities:
            recs.append("Increase strategic petroleum reserves to 90+ days")
        if "Low renewable penetration" in vulnerabilities:
            recs.append("Accelerate renewable deployment timeline")
        if "High import dependency" in vulnerabilities:
            recs.append("Develop domestic energy production capacity")
        if "Grid instability risk" in vulnerabilities:
            recs.append("Invest in grid modernization and storage")
        if "High coal dependency" in vulnerabilities:
            recs.append("Plan coal phase-out with just transition framework")
        if security > 0.7:
            recs.append("Maintain current energy security posture")
        return recs

    def carbon_budget_tracker(self, state: Optional[EnergyState] = None, budget_gtco2: float = 500.0) -> dict:
        if state is None:
            state = EnergyState()
        current_emissions_gtco2 = 36.8
        years_remaining = budget_gtco2 / current_emissions_gtco2 if current_emissions_gtco2 > 0 else 0
        annual_reduction_needed = current_emissions_gtco2 - (budget_gtco2 / max(25, years_remaining)) if years_remaining > 0 else current_emissions_gtco2
        reduction_pct = (annual_reduction_needed / current_emissions_gtco2 * 100) if current_emissions_gtco2 > 0 else 0
        carbon_price_adequacy = min(state.carbon_price_usd / 100.0, 1.0)
        return {
            "remaining_budget_gtco2": round(budget_gtco2, 1),
            "current_annual_emissions_gtco2": round(current_emissions_gtco2, 1),
            "years_at_current_rate": round(years_remaining, 1),
            "annual_reduction_needed_gtco2": round(max(annual_reduction_needed, 0), 2),
            "reduction_pct_annual": round(reduction_pct, 1),
            "budget_deadline_year": round(2025 + years_remaining),
            "carbon_price_adequacy": round(carbon_price_adequacy, 3),
            "on_track": annual_reduction_needed < current_emissions_gtco2 * 0.05,
            "scenarios": {
                "1.5c": {
                    "budget_gtco2": 400,
                    "remaining_budget_gtco2": 400,
                    "years_left": round(400 / current_emissions_gtco2, 1),
                    "years_at_current_rate": round(400 / current_emissions_gtco2, 1),
                    "budget_used_pct": round((current_emissions_gtco2 * (2025 - 2020)) / 400, 3) if budget_gtco2 > 0 else 0,
                },
                "2.0c": {
                    "budget_gtco2": 1150,
                    "remaining_budget_gtco2": 1150,
                    "years_left": round(1150 / current_emissions_gtco2, 1),
                    "years_at_current_rate": round(1150 / current_emissions_gtco2, 1),
                    "budget_used_pct": round((current_emissions_gtco2 * (2025 - 2020)) / 1150, 3),
                },
                "2.5c": {
                    "budget_gtco2": 2300,
                    "remaining_budget_gtco2": 2300,
                    "years_left": round(2300 / current_emissions_gtco2, 1),
                    "years_at_current_rate": round(2300 / current_emissions_gtco2, 1),
                    "budget_used_pct": round((current_emissions_gtco2 * (2025 - 2020)) / 2300, 3),
                },
            },
            "recommendation": "Aggressively transition to renewables and increase carbon pricing to meet the 1.5C target." if not (annual_reduction_needed < current_emissions_gtco2 * 0.05) else "Current trajectory is within budget bounds but continued reduction efforts needed.",
        }

    def energy_transition_cost(self, state: Optional[EnergyState] = None, years: int = 10, target_renewable_pct: float = 60.0) -> dict:
        if state is None:
            state = EnergyState()
        current_gw_renewable = 3500.0
        current_gw_nuclear = 440.0
        cost_per_gw_solar = 800e6
        cost_per_gw_wind = 1400e6
        cost_per_gw_nuclear = 6000e6
        cost_per_gw_storage = 400e6
        cost_grid_upgrade_per_pct = 50e9
        current_renewable_pct = state.renewable_share_pct
        gap_pct = max(target_renewable_pct - current_renewable_pct, 0)
        annual_growth_rate = 0.08
        required_gw_renewable = current_gw_renewable * ((1 + annual_growth_rate) ** years - 1) * (gap_pct / 40.0)
        required_gw_storage = required_gw_renewable * 0.25
        grid_upgrade_pct = gap_pct * 0.6
        nuclear_addition_gw = current_gw_nuclear * 0.1 * (years / 10.0)
        solar_share = 0.55
        wind_share = 0.45
        solar_cost = required_gw_renewable * solar_share * cost_per_gw_solar
        wind_cost = required_gw_renewable * wind_share * cost_per_gw_wind
        nuclear_cost = nuclear_addition_gw * cost_per_gw_nuclear
        storage_cost = required_gw_storage * cost_per_gw_storage
        grid_cost = grid_upgrade_pct * cost_grid_upgrade_per_pct
        total_cost = solar_cost + wind_cost + nuclear_cost + storage_cost + grid_cost
        carbon_revenue = state.carbon_price_usd * 36.8 * years * 1e6 * 0.01
        fossil_savings = (1 - (current_renewable_pct / 100.0)) * 500e9 * years * 0.02 * (gap_pct / 100.0)
        net_cost = max(total_cost - carbon_revenue - fossil_savings, 0)
        return {
            "target_renewable_pct": target_renewable_pct,
            "current_renewable_pct": current_renewable_pct,
            "gap_pct": round(gap_pct, 1),
            "years": years,
            "capex": {
                "solar_gw": round(required_gw_renewable * solar_share, 0),
                "wind_gw": round(required_gw_renewable * wind_share, 0),
                "nuclear_gw": round(nuclear_addition_gw, 0),
                "storage_gw": round(required_gw_storage, 0),
                "solar_cost_bn": round(solar_cost / 1e9, 1),
                "wind_cost_bn": round(wind_cost / 1e9, 1),
                "nuclear_cost_bn": round(nuclear_cost / 1e9, 1),
                "storage_cost_bn": round(storage_cost / 1e9, 1),
                "grid_upgrade_cost_bn": round(grid_cost / 1e9, 1),
                "total_capex_bn": round(total_cost / 1e9, 1),
            },
            "revenue_offsets": {
                "carbon_pricing_revenue_bn": round(carbon_revenue / 1e9, 1),
                "fossil_savings_bn": round(fossil_savings / 1e9, 1),
            },
            "net_cost_bn": round(net_cost / 1e9, 1),
            "cost_per_ton_co2_avoided": round(net_cost / (36.8 * gap_pct / 100.0 * years * 1e9), 2) if gap_pct > 0 else 0,
            "feasibility": "achievable" if gap_pct < 30 and years >= 10 else "challenging" if gap_pct < 50 else "extremely_ambitious",
            "recommendations": self._transition_recommendations(gap_pct, years, net_cost),
        }

    def _transition_recommendations(self, gap_pct: float, years: int, net_cost: float) -> list:
        recs = []
        if gap_pct > 40:
            recs.append("Consider extending timeline beyond 10 years for feasibility")
        if years < 8 and gap_pct > 20:
            recs.append("Accelerate deployment with streamlined permitting")
        if net_cost > 5000e9:
            recs.append("Leverage green bonds and international climate finance")
        recs.append("Prioritize grid storage to manage intermittency")
        recs.append("Phase nuclear additions to provide baseload stability")
        return recs

    def grid_storage_requirements(self, state: Optional[EnergyState] = None, renewable_target_pct: float = 50.0) -> dict:
        if state is None:
            state = EnergyState()
        current_renewable_pct = state.renewable_share_pct
        intermittency_factor = 0.3 if current_renewable_pct < 25 else 0.4 if current_renewable_pct < 40 else 0.5
        peak_demand_gw = 800.0
        base_storage_hours = 4.0
        additional_hours = (renewable_target_pct - current_renewable_pct) * 0.08
        total_storage_hours = base_storage_hours + max(additional_hours, 0)
        required_gw = peak_demand_gw * (renewable_target_pct / 100.0) * intermittency_factor
        required_gwh = required_gw * total_storage_hours
        lithium_requirement_kt = required_gwh * 0.8
        cost_per_gwh_liion = 150e6
        cost_per_gwh_flow = 100e6
        liion_share = 0.85
        flow_share = 0.15
        liion_cost = required_gwh * liion_share * cost_per_gwh_liion
        flow_cost = required_gwh * flow_share * cost_per_gwh_flow
        total_cost = liion_cost + flow_cost
        return {
            "current_renewable_pct": current_renewable_pct,
            "target_renewable_pct": renewable_target_pct,
            "intermittency_factor": round(intermittency_factor, 2),
            "storage_requirements": {
                "required_gw": round(required_gw, 1),
                "required_gwh": round(required_gwh, 1),
                "storage_hours": round(total_storage_hours, 1),
                "liion_gwh": round(required_gwh * liion_share, 1),
                "flow_battery_gwh": round(required_gwh * flow_share, 1),
            },
            "material_requirements": {
                "lithium_kt": round(lithium_requirement_kt, 0),
                "lithium_supply_years": round(89000 / lithium_requirement_kt, 1) if lithium_requirement_kt > 0 else 0,
            },
            "cost_estimate": {
                "liion_cost_bn": round(liion_cost / 1e9, 1),
                "flow_battery_cost_bn": round(flow_cost / 1e9, 1),
                "total_storage_cost_bn": round(total_cost / 1e9, 1),
            },
            "grid_upgrade_needed": current_renewable_pct < 30,
            "peak_demand_accommodation": round(required_gw / peak_demand_gw * 100, 1),
            "recommendations": [
                "Deploy 4-hour lithium-ion for daily cycling",
                "Add flow batteries for long-duration storage",
                "Upgrade transmission for renewable curtailment reduction",
                "Consider pumped hydro where geography permits",
            ][:3 if required_gw < 200 else 4],
        }