import numpy as np
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
from oracle.prediction.engine import OracleEngine
from oracle.core.types import Forecast, Prediction, ConfidenceLevel, Domain

logger = logging.getLogger(__name__)


@dataclass
class TimeSlice:
    target_date: date
    forecast: Forecast
    confidence_level: ConfidenceLevel
    years_ahead: float
    summary: dict = field(default_factory=dict)


class TimeSlider:
    def __init__(self, engine: OracleEngine, start_date: date = date(2020, 1, 1),
                 end_date: date = date(2040, 12, 31)):
        self.engine = engine
        self.start_date = start_date
        self.end_date = end_date
        self._slices: dict[date, TimeSlice] = {}

    def slide_to(self, target_date: date, confidence: ConfidenceLevel = ConfidenceLevel.HIGH) -> TimeSlice:
        if target_date in self._slices:
            cached = self._slices[target_date]
            if cached.confidence_level == confidence:
                return cached

        years_ahead = (target_date - date.today()).days / 365.25
        forecast = self.engine.predict(target_date, confidence)

        summary = self._compute_summary(forecast, target_date)

        ts = TimeSlice(
            target_date=target_date,
            forecast=forecast,
            confidence_level=confidence,
            years_ahead=years_ahead,
            summary=summary,
        )
        self._slices[target_date] = ts
        return ts

    def slide_range(
        self, start: date, end: date, step_days: int = 90,
        confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    ) -> list[TimeSlice]:
        results = []
        current = start
        while current <= end:
            results.append(self.slide_to(current, confidence))
            current += timedelta(days=step_days)
        return results

    def quick_scan(
        self, intervals: list[int] = None, confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    ) -> list[TimeSlice]:
        if intervals is None:
            intervals = [30, 90, 180, 365, 730, 1825, 3650, 7300]
        today = date.today()
        results = []
        for days in intervals:
            target = today + timedelta(days=days)
            if target <= self.end_date:
                results.append(self.slide_to(target, confidence))
        return results

    def scenario_analysis(
        self, scenarios: dict[str, dict], target_date: date,
        confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    ) -> dict[str, TimeSlice]:
        results = {}
        base_state = self.engine.state.economic_state
        base_macro = self.engine.state.macro_state

        for name, params in scenarios.items():
            if base_state:
                state_params = {
                    "gdp_growth_rate": base_state.gdp_growth_rate,
                    "debt_to_gdp": base_state.debt_to_gdp,
                    "interest_rate": base_state.interest_rate,
                    "money_supply_growth": base_state.money_supply_growth,
                    "credit_growth": base_state.credit_growth,
                    "productivity_growth": base_state.productivity_growth,
                    "inflation_rate": base_state.inflation_rate,
                    "unemployment_rate": base_state.unemployment_rate,
                    "stock_valuation": base_state.stock_valuation,
                    "capacity_utilization": base_state.capacity_utilization,
                }
                state_params.update(params)
                self.engine.set_economic_state(**state_params)

            try:
                results[name] = self.slide_to(target_date, confidence)
            except Exception as e:
                logger.warning(f"Scenario '{name}' prediction failed: {e}")
                results[name] = TimeSlice(
                    date=target_date,
                    predictions=[],
                    domain_summaries={},
                    risk_assessment={},
                    summary=f"Prediction failed for scenario '{name}': {e}",
                )

        try:
            if base_state:
                self.engine.set_economic_state(
                    gdp_growth_rate=base_state.gdp_growth_rate,
                    debt_to_gdp=base_state.debt_to_gdp,
                    interest_rate=base_state.interest_rate,
                    money_supply_growth=base_state.money_supply_growth,
                    credit_growth=base_state.credit_growth,
                    productivity_growth=base_state.productivity_growth,
                    inflation_rate=base_state.inflation_rate,
                    unemployment_rate=base_state.unemployment_rate,
                    stock_valuation=base_state.stock_valuation,
                    capacity_utilization=base_state.capacity_utilization,
                )
        except Exception as e:
            logger.warning(f"Failed to restore base state after scenario analysis: {e}")

        return results

    def domain_scenario_analysis(
        self,
        domain_scenarios: dict[str, dict],
        target_date: date,
        confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    ) -> dict[str, TimeSlice]:
        domain_presets = {
            "geopolitical_crisis": {
                "geopolitical": {"us_china_tension": 0.9, "russia_west_tension": 0.9, "middle_east_instability": 0.9},
                "sentiment": {"fear_greed_index": 0.15, "vix_level": 0.9},
            },
            "geopolitical_peace": {
                "geopolitical": {"us_china_tension": 0.1, "russia_west_tension": 0.1, "middle_east_instability": 0.1},
                "sentiment": {"fear_greed_index": 0.8, "vix_level": 0.1},
            },
            "climate_acceleration": {
                "climate": {"global_temp_anomaly_c": 2.5, "co2_ppm": 500},
                "contagion": {"financial_systemic_risk": 0.6, "trade_linkage": 0.7},
            },
            "climate_action": {
                "climate": {"global_temp_anomaly_c": 0.8, "co2_ppm": 380},
            },
            "financial_crisis": {
                "contagion": {"financial_systemic_risk": 0.8, "banking_contagion": 0.7, "sovereign_contagion": 0.6},
                "sentiment": {"fear_greed_index": 0.1, "vix_level": 0.9, "put_call_ratio": 1.8},
            },
            "demographic_decline": {
                "demographics": {"birth_rate_per_1000": 7, "death_rate_per_1000": 14, "elderly_pct": 0.30, "fertility_rate": 1.1},
            },
        }

        results = {}
        for name, overrides in domain_scenarios.items():
            preset = domain_presets.get(name, overrides)
            setters = {
                "geopolitical": self.engine.set_geopolitical_state,
                "climate": self.engine.set_climate_state,
                "sentiment": self.engine.set_sentiment_data,
                "contagion": self.engine.set_contagion_state,
                "demographics": self.engine.set_demographics_state,
            }
            applied = {}
            for domain_key, params in preset.items():
                if domain_key in setters and isinstance(params, dict):
                    setters[domain_key](**params)
                    applied[domain_key] = params

            results[name] = self.slide_to(target_date, confidence)

            for domain_key in applied:
                setters_to_reset = {
                    "geopolitical": self.engine.set_geopolitical_state,
                    "climate": self.engine.set_climate_state,
                    "sentiment": self.engine.set_sentiment_data,
                    "contagion": self.engine.set_contagion_state,
                    "demographics": self.engine.set_demographics_state,
                }
                setters_to_reset[domain_key]()

        return results

    def get_timeline(self) -> list[TimeSlice]:
        return sorted(self._slices.values(), key=lambda ts: ts.target_date)

    def _compute_summary(self, forecast: Forecast, target_date: date) -> dict:
        summary = {
            "target_date": target_date.isoformat(),
            "years_ahead": (target_date - date.today()).days / 365.25,
            "n_predictions": len(forecast.predictions),
            "domains": {},
            "ensemble_values": {},
            "risk_assessment": {},
        }

        domain_preds = {}
        for p in forecast.predictions:
            domain_preds.setdefault(p.domain.value, []).append(p)

        for domain, preds in domain_preds.items():
            if domain not in summary["domains"]:
                summary["domains"][domain] = {}

            for p in preds:
                if p.model_name == "oracle_ensemble":
                    summary["ensemble_values"][f"{domain}:{p.variable}"] = {
                        "value": p.value,
                        "ci_lower": p.confidence_interval[0],
                        "ci_upper": p.confidence_interval[1],
                        "ci_width": round(p.confidence_interval[1] - p.confidence_interval[0], 6),
                    }
                summary["domains"][domain][f"{p.variable} ({p.model_name})"] = {
                    "value": p.value,
                    "ci": p.confidence_interval,
                }

        ensemble_preds = [p for p in forecast.predictions if p.model_name == "oracle_ensemble"]
        for p in ensemble_preds:
            ci_width = p.confidence_interval[1] - p.confidence_interval[0]
            relative_uncertainty = ci_width / abs(p.value) if abs(p.value) > 1e-10 else float("inf")
            if relative_uncertainty > 1.0:
                risk = "HIGH"
            elif relative_uncertainty > 0.5:
                risk = "MEDIUM"
            else:
                risk = "LOW"
            summary["risk_assessment"][f"{p.domain.value}:{p.variable}"] = {
                "risk_level": risk,
                "relative_uncertainty": round(relative_uncertainty, 4),
                "ci_width": round(ci_width, 6),
            }

        return summary