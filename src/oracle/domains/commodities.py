import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CommoditiesState:
    gold_price: float = 1900.0
    copper_price: float = 8500.0
    wheat_price: float = 280.0
    lithium_price: float = 15000.0
    rare_earth_index: float = 100.0
    supply_chain_stress: float = 0.3
    geopolitical_risk: float = 0.4
    inflation_expectation: float = 0.025


class CommoditiesModel:
    def __init__(self):
        self.commodities = {
            "gold": {"price_usd": 1900, "volatility": 0.15, "correlation_inflation": 0.6, "safe_haven": True, "unit": "USD/oz"},
            "copper": {"price_usd": 8500, "volatility": 0.25, "correlation_growth": 0.7, "safe_haven": False, "unit": "USD/ton"},
            "oil": {"price_usd": 75, "volatility": 0.30, "correlation_growth": 0.5, "safe_haven": False, "unit": "USD/bbl"},
            "wheat": {"price_usd": 280, "volatility": 0.20, "correlation_inflation": 0.4, "safe_haven": False, "unit": "USD/bushel"},
            "lithium": {"price_usd": 15000, "volatility": 0.35, "correlation_tech": 0.8, "safe_haven": False, "unit": "USD/ton"},
            "rare_earth": {"price_usd": 100, "volatility": 0.25, "correlation_tech": 0.7, "safe_haven": False, "unit": "index"},
            "iron_ore": {"price_usd": 110, "volatility": 0.20, "correlation_growth": 0.6, "safe_haven": False, "unit": "USD/ton"},
            "natural_gas": {"price_usd": 3.0, "volatility": 0.35, "correlation_weather": 0.4, "safe_haven": False, "unit": "USD/MMBtu"},
        }

    def price_forecast(self, commodity: str, state: Optional[CommoditiesState] = None, days: int = 252, n_scenarios: int = 1000, confidence: str = "medium") -> dict:
        if state is None:
            state = CommoditiesState()
        commodity = commodity.lower()
        if commodity not in self.commodities:
            commodity = "gold"
        comm = self.commodities[commodity]
        np.random.seed(42)
        vol = comm["volatility"]
        drift = 0.0001
        if comm.get("safe_haven"):
            drift += state.geopolitical_risk * 0.0003
            drift += state.inflation_expectation * 0.5
        else:
            drift -= state.supply_chain_stress * 0.0002
        scenarios = np.random.normal(drift / 252, vol / np.sqrt(252), (n_scenarios, days))
        paths = comm["price_usd"] * np.exp(np.cumsum(scenarios, axis=1))
        conf_map = {"low": 10, "medium": 25, "high": 40, "very_high": 49}
        ci = conf_map.get(confidence, 25)
        p10 = float(np.percentile(paths[:, -1], ci))
        p50 = float(np.percentile(paths[:, -1], 50))
        p90 = float(np.percentile(paths[:, -1], 100 - ci))
        max_drawdown = float(np.max(np.maximum.accumulate(paths, axis=1) - paths) / np.maximum.accumulate(paths, axis=1).max())
        expected_return_pct = round((p50 / comm["price_usd"] - 1), 4)
        volatility_pct = round(vol, 4)
        return {
            "commodity": commodity,
            "current_price": comm["price_usd"],
            "unit": comm["unit"],
            "forecast_days": days,
            "scenarios_run": n_scenarios,
            "expected_return_pct": expected_return_pct,
            "volatility_pct": volatility_pct,
            "percentiles": {
                "p10": round(p10, 2),
                "p25": round(float(np.percentile(paths[:, -1], 25)), 2),
                "p50": round(p50, 2),
                "p75": round(float(np.percentile(paths[:, -1], 75)), 2),
                "p90": round(p90, 2),
            },
            "expected_return_annualized": round((p50 / comm["price_usd"] - 1) * 100, 2),
            "risk_metrics": {
                "var_95_daily": round(comm["price_usd"] * vol / np.sqrt(252) * 1.645, 2),
                "cvar_95_daily": round(comm["price_usd"] * vol / np.sqrt(252) * 2.06, 2),
                "max_drawdown_pct": round(max_drawdown * 100, 1),
            },
        }

    def supply_demand_analysis(self, commodity: str, state: Optional[CommoditiesState] = None) -> dict:
        if state is None:
            state = CommoditiesState()
        commodity = commodity.lower()
        commodity_data = {
            "gold": {"annual_supply_mt": 3600, "annual_demand_mt": 4400, "stock_to_flow": 65, "recycling_rate": 0.28},
            "copper": {"annual_supply_mt": 21000000, "annual_demand_mt": 25000000, "stock_to_flow": 35, "recycling_rate": 0.35},
            "oil": {"annual_supply_mbd": 100000, "annual_demand_mbd": 99000, "stock_to_flow": 8, "recycling_rate": 0.0},
            "wheat": {"annual_supply_mt": 780, "annual_demand_mt": 790, "stock_to_flow": 3, "recycling_rate": 0.0},
            "lithium": {"annual_supply_mt": 130000, "annual_demand_mt": 150000, "stock_to_flow": 12, "recycling_rate": 0.05},
            "rare_earth": {"annual_supply_mt": 300000, "annual_demand_mt": 290000, "stock_to_flow": 15, "recycling_rate": 0.10},
            "iron_ore": {"annual_supply_mt": 2500000000, "annual_demand_mt": 2400000000, "stock_to_flow": 5, "recycling_rate": 0.25},
            "natural_gas": {"annual_supply_bcm": 4000, "annual_demand_bcm": 3950, "stock_to_flow": 5, "recycling_rate": 0.0},
        }
        comm = commodity_data.get(commodity, commodity_data["gold"])
        supply_keys = [k for k in comm.keys() if "supply" in k]
        demand_keys = [k for k in comm.keys() if "demand" in k]
        supply_val = comm[supply_keys[0]] if supply_keys else 0
        demand_val = comm[demand_keys[0]] if demand_keys else 0
        deficit = (demand_val - supply_val) / demand_val if demand_val > 0 else 0
        stress = state.supply_chain_stress * 0.5 + abs(deficit) * 0.3 + state.geopolitical_risk * 0.2
        supply_normalized = min(supply_val / max(demand_val, 1), 1.5) if demand_val > 0 else 0
        industrial_demand = min(supply_normalized, 1.0)
        strategic_factor = min(comm["stock_to_flow"] / 50.0, 1.0)
        extraction_cost = round(min(state.geopolitical_risk * 0.3 + state.supply_chain_stress * 0.4 + 0.3, 1.0), 3)
        outlook_descriptors = {
            "gold": ("Gold demand driven by safe-haven flows and central bank purchases.", "Prices likely to remain elevated with geopolitical uncertainty."),
            "copper": ("Copper demand rising with electrification and green transition.", "Supply tightness expected to support prices."),
            "oil": ("Oil demand plateauing as renewable adoption accelerates.", "Price volatility expected to remain elevated."),
            "wheat": ("Wheat demand stable with climate risk to supply.", "Weather events could spike prices seasonally."),
            "lithium": ("Lithium demand surging with EV and battery storage growth.", "Long-term supply deficit unless new mines come online."),
            "rare_earth": ("Rare earth demand driven by tech and defense sectors.", "Concentrated supply creates persistent risk."),
            "iron_ore": ("Iron ore demand tracking global construction cycle.", "Chinese demand outlook is key variable."),
            "natural_gas": ("Natural gas demand as transition fuel bridging coal-to-renewables.", "Price highly seasonal and weather-dependent."),
        }
        desc = outlook_descriptors.get(commodity, ("Commodity outlook uncertain.", "Price direction depends on macro factors."))
        return {
            "commodity": commodity,
            "supply": {
                "production_level": round(supply_normalized, 3),
                "reserve_level": round(comm["stock_to_flow"] / 100.0, 3),
                "extraction_cost": extraction_cost,
                "annual_supply": supply_val,
            },
            "demand": {
                "consumption_growth": round(min(abs(deficit), 1.0), 3),
                "industrial_demand": round(industrial_demand, 3),
                "strategic_stockpile": round(strategic_factor, 3),
                "annual_demand": demand_val,
            },
            "deficit_pct": round(deficit * 100, 1),
            "stock_to_flow": comm["stock_to_flow"],
            "recycling_rate": round(comm["recycling_rate"] * 100, 1),
            "supply_stress_index": round(stress, 3),
            "stress_level": "high" if stress > 0.6 else "moderate" if stress > 0.3 else "low",
            "geopolitical_risk_factor": round(state.geopolitical_risk, 3),
            "market_outlook": desc[0],
            "price_outlook": desc[1],
        }

    def correlation_matrix(self, state: Optional[CommoditiesState] = None) -> dict:
        np.random.seed(42)
        n = 252
        base_corr = np.array([
            [1.00, 0.15, -0.10, 0.05, 0.05, 0.10, 0.10, -0.20],
            [0.15, 1.00, 0.40, 0.10, 0.55, 0.45, 0.60, 0.20],
            [-0.10, 0.40, 1.00, 0.10, 0.15, 0.20, 0.05, 0.50],
            [0.05, 0.10, 0.10, 1.00, 0.05, 0.05, 0.10, 0.15],
            [0.05, 0.55, 0.15, 0.05, 1.00, 0.70, 0.20, 0.10],
            [0.10, 0.45, 0.20, 0.05, 0.70, 1.00, 0.30, 0.10],
            [0.10, 0.60, 0.05, 0.10, 0.20, 0.30, 1.00, 0.05],
            [-0.20, 0.20, 0.50, 0.15, 0.10, 0.10, 0.05, 1.00],
        ])
        names = ["gold", "copper", "oil", "wheat", "lithium", "rare_earth", "iron_ore", "natural_gas"]
        corr_dict = {}
        matrix_2d = []
        for i, name_i in enumerate(names):
            corr_dict[name_i] = {}
            row = []
            for j, name_j in enumerate(names):
                val = round(float(base_corr[i, j]), 3)
                corr_dict[name_i][name_j] = val
                row.append(val)
            matrix_2d.append(row)
        return {
            "commodities": names,
            "matrix": matrix_2d,
            "correlation_matrix": corr_dict,
            "safe_haven_commodities": ["gold"],
            "growth_sensitive": ["copper", "iron_ore", "oil"],
            "tech_metal": ["lithium", "rare_earth"],
        }

    def critical_minerals_outlook(self, state: Optional[CommoditiesState] = None, years: int = 10) -> dict:
        if state is None:
            state = CommoditiesState()
        minerals = [
            {"name": "Lithium", "current_demand_growth": 0.20, "supply_growth": 0.12, "price_trend_annual": 0.08, "substitution_difficulty": 0.8, "recycling_rate": 0.05},
            {"name": "Cobalt", "current_demand_growth": 0.12, "supply_growth": 0.06, "price_trend_annual": 0.05, "substitution_difficulty": 0.6, "recycling_rate": 0.15},
            {"name": "Nickel", "current_demand_growth": 0.08, "supply_growth": 0.05, "price_trend_annual": 0.03, "substitution_difficulty": 0.3, "recycling_rate": 0.30},
            {"name": "Rare Earths", "current_demand_growth": 0.15, "supply_growth": 0.08, "price_trend_annual": 0.06, "substitution_difficulty": 0.9, "recycling_rate": 0.10},
            {"name": "Graphite", "current_demand_growth": 0.18, "supply_growth": 0.10, "price_trend_annual": 0.04, "substitution_difficulty": 0.4, "recycling_rate": 0.08},
            {"name": "Manganese", "current_demand_growth": 0.06, "supply_growth": 0.04, "price_trend_annual": 0.02, "substitution_difficulty": 0.3, "recycling_rate": 0.20},
        ]
        outlook = []
        for m in minerals:
            gap = m["current_demand_growth"] - m["supply_growth"]
            scarcity = gap * m["substitution_difficulty"] * (1 - m["recycling_rate"])
            year_demand_multiple = (1 + m["current_demand_growth"]) ** years
            year_supply_multiple = (1 + m["supply_growth"]) ** years
            deficit_pct = max(0, (year_demand_multiple - year_supply_multiple) / year_supply_multiple) * 100
            outlook.append({
                "mineral": m["name"],
                "demand_growth_annual_pct": round(m["current_demand_growth"] * 100, 1),
                "supply_growth_annual_pct": round(m["supply_growth"] * 100, 1),
                "supply_gap_pct": round(gap * 100, 1),
                "price_trend_annual_pct": round(m["price_trend_annual"] * 100, 1),
                "substitution_difficulty": round(m["substitution_difficulty"], 2),
                "recycling_rate_pct": round(m["recycling_rate"] * 100, 1),
                f"demand_x{years}": round(year_demand_multiple, 2),
                f"supply_x{years}": round(year_supply_multiple, 2),
                f"projected_deficit_pct": round(deficit_pct, 1),
                "scarcity_score": round(scarcity * 10, 2),
                "risk_level": "critical" if scarcity > 0.1 else "high" if scarcity > 0.05 else "moderate" if scarcity > 0.02 else "low",
            })
        outlook.sort(key=lambda x: x["scarcity_score"], reverse=True)
        minerals_dict = {}
        for m in outlook:
            key = m["mineral"].lower().replace(" ", "_")
            minerals_dict[key] = {
                "scarcity_score": round(m["scarcity_score"] / 10, 3),
                "supply_risk": round(m["substitution_difficulty"] * (1 - m["recycling_rate_pct"] / 100), 3),
                "price_trend": "up" if m["price_trend_annual_pct"] > 3 else "stable" if m["price_trend_annual_pct"] > 1 else "down",
                "demand_growth_annual_pct": m["demand_growth_annual_pct"],
                "supply_growth_annual_pct": m["supply_growth_annual_pct"],
                "supply_gap_pct": m["supply_gap_pct"],
                "price_trend_annual_pct": m["price_trend_annual_pct"],
                "substitution_difficulty": m["substitution_difficulty"],
                "recycling_rate_pct": m["recycling_rate_pct"],
                "risk_level": m["risk_level"],
            }
        most_critical = outlook[0]["mineral"] if outlook else "N/A"
        avg_scarcity = sum(m["scarcity_score"] for m in outlook) / len(outlook) / 10 if outlook else 0
        recs = []
        if avg_scarcity > 0.05:
            recs.append("Invest in alternative material R&D")
        if any(m["scarcity_score"] > 0.5 for m in outlook):
            recs.append("Diversify supply chains for high-scarcity minerals")
        if avg_scarcity > 0.08:
            recs.append("Accelerate recycling infrastructure development")
        recs.append("Monitor geopolitical risks in key mining regions")
        return {
            "minerals": minerals_dict,
            "minerals_list": outlook,
            "years_ahead": years,
            "top_critical": [m["mineral"] for m in outlook[:3]],
            "most_critical": most_critical.lower().replace(" ", "_"),
            "avg_scarcity": round(avg_scarcity, 3),
            "recommendations": recs,
            "geopolitical_risk_factor": round(state.geopolitical_risk, 3),
        }

    def commodity_index(self, state: Optional[CommoditiesState] = None) -> dict:
        if state is None:
            state = CommoditiesState()
        weights = {
            "gold": 0.15,
            "copper": 0.20,
            "oil": 0.20,
            "wheat": 0.05,
            "lithium": 0.10,
            "rare_earth": 0.08,
            "iron_ore": 0.12,
            "natural_gas": 0.10,
        }
        base_prices = {name: data["price_usd"] for name, data in self.commodities.items()}
        current_prices = {
            "gold": state.gold_price,
            "copper": getattr(state, "copper_price", 8500.0),
            "oil": base_prices["oil"],
            "wheat": getattr(state, "wheat_price", 280.0),
            "lithium": getattr(state, "lithium_price", 15000.0),
            "rare_earth": base_prices["rare_earth"],
            "iron_ore": base_prices["iron_ore"],
            "natural_gas": base_prices["natural_gas"],
        }
        price_changes = {}
        for name, current in current_prices.items():
            base = base_prices[name]
            if base > 0:
                price_changes[name] = (current - base) / base
            else:
                price_changes[name] = 0.0
        stress_adj = state.supply_chain_stress * 0.02
        geo_adj = state.geopolitical_risk * 0.015
        inflation_adj = state.inflation_expectation * 0.4
        weighted_change = sum(price_changes.get(n, 0) * weights.get(n, 0) for n in self.commodities)
        index_value = 100.0 * (1 + weighted_change + stress_adj + geo_adj + inflation_adj)
        sector_breakdown = {}
        for name in self.commodities:
            sector_breakdown[name] = {
                "weight": weights.get(name, 0),
                "price_change_pct": round(price_changes.get(name, 0) * 100, 2),
                "contribution": round(price_changes.get(name, 0) * weights.get(name, 0) * 100, 3),
            }
        trend = "bullish" if index_value > 105 else "neutral" if index_value > 95 else "bearish"
        return {
            "index_value": round(index_value, 2),
            "trend": trend,
            "weighted_change_pct": round(weighted_change * 100, 2),
            "adjustments": {
                "supply_chain_stress_contribution": round(stress_adj * 100, 2),
                "geopolitical_risk_contribution": round(geo_adj * 100, 2),
                "inflation_contribution": round(inflation_adj * 100, 2),
            },
            "sector_breakdown": sector_breakdown,
            "top_mover": max(sector_breakdown, key=lambda k: abs(sector_breakdown[k]["price_change_pct"])),
            "interpretation": f"Commodity index at {round(index_value, 1)} indicates {trend} broad market conditions. Supply chain stress adds {round(stress_adj*100, 2)}%, geopolitical risk adds {round(geo_adj*100, 2)}%.",
        }

    def seasonality_analysis(self, commodity: str, years_back: int = 5) -> dict:
        commodity = commodity.lower()
        if commodity not in self.commodities:
            commodity = "oil"
        comm = self.commodities[commodity]
        np.random.seed(42)
        seasonal_patterns = {
            "gold": [0.02, 0.01, 0.02, 0.01, -0.01, -0.02, 0.01, 0.02, 0.03, 0.02, 0.03, 0.02],
            "copper": [0.01, 0.02, 0.03, 0.02, 0.01, -0.01, -0.02, -0.01, 0.01, 0.02, 0.02, 0.01],
            "oil": [-0.01, -0.02, 0.02, 0.03, 0.04, 0.03, 0.02, -0.01, -0.02, -0.01, 0.00, -0.01],
            "wheat": [0.00, -0.02, 0.02, 0.03, 0.01, -0.03, -0.04, -0.02, 0.01, 0.02, 0.01, 0.00],
            "lithium": [0.01, 0.01, 0.02, 0.02, 0.01, 0.00, -0.01, -0.01, 0.01, 0.02, 0.02, 0.01],
            "rare_earth": [0.01, 0.00, 0.01, 0.02, 0.01, 0.00, -0.01, -0.01, 0.00, 0.01, 0.01, 0.01],
            "iron_ore": [0.02, 0.03, 0.02, 0.01, 0.00, -0.02, -0.03, -0.01, 0.01, 0.02, 0.02, 0.01],
            "natural_gas": [0.04, 0.03, 0.01, -0.01, -0.03, -0.04, -0.02, 0.00, 0.02, 0.03, 0.04, 0.05],
        }
        base_pattern = seasonal_patterns.get(commodity, [0.0] * 12)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_returns = []
        for y in range(years_back * 12):
            month_idx = y % 12
            noise = np.random.normal(0, comm["volatility"] / np.sqrt(12))
            monthly_returns.append(base_pattern[month_idx] / 12 + noise)
        monthly_returns = np.array(monthly_returns)
        avg_monthly = []
        for m in range(12):
            month_vals = monthly_returns[m::12]
            avg_monthly.append(round(float(np.mean(month_vals)), 4))
        best_month_idx = int(np.argmax(avg_monthly))
        worst_month_idx = int(np.argmin(avg_monthly))
        total_return = float(np.prod(1 + monthly_returns) - 1)
        seasonal_strength = float(np.std(avg_monthly)) / (float(np.std(monthly_returns)) + 1e-10)
        monthly_prices = [round(comm["price_usd"] * (1 + r), 2) for r in avg_monthly]
        return {
            "commodity": commodity,
            "unit": comm["unit"],
            "months": months,
            "avg_monthly_returns": [round(r * 100, 2) for r in avg_monthly],
            "avg_monthly_prices": monthly_prices,
            "best_month": months[best_month_idx],
            "best_month_avg_return_pct": round(avg_monthly[best_month_idx] * 100, 2),
            "worst_month": months[worst_month_idx],
            "worst_month_avg_return_pct": round(avg_monthly[worst_month_idx] * 100, 2),
            "seasonal_strength": round(seasonal_strength, 3),
            "total_return_pct": round(total_return * 100, 2),
            "seasonality_level": "strong" if seasonal_strength > 0.3 else "moderate" if seasonal_strength > 0.15 else "weak",
            "pattern_description": f"{commodity.title()} tends to peak in {months[best_month_idx]} and bottom in {months[worst_month_idx]} with {('strong' if seasonal_strength > 0.3 else 'moderate')} seasonality",
            "years_analyzed": years_back,
        }