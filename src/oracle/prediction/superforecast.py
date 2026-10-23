"""Superforecasting Engine — Base Rate Anchoring, Inside/Outside View, Bayesian Updating, Extremizing.

Core Insight: Research by Philip Tetlock showed that "superforecasters" outperform
experts by: (1) anchoring on base rates, (2) combining inside and outside views,
(3) updating incrementally with Bayes' rule, and (4) extremizing toward the edges
when evidence is strong. This engine operationalizes those principles.

Implements:
- BaseRateDatabase: Reference class forecasting with historical base rates
- SuperforecastEngine: Combines inside/outside view with Bayesian updating and extremizing
- ForecastDecomposition: Decomposes forecast into base rate, inside view, adjustment, extremized
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class BaseRateCategory(Enum):
    RECESSION_1Y = "recession_1y"
    RECESSION_5Y = "recession_5y"
    FINANCIAL_CRISIS = "financial_crisis"
    GEOPOLITICAL_CONFLICT = "geopolitical_conflict"
    PANDEMIC = "pandemic"
    HOUSING_CRASH = "housing_crash"
    CURRENCY_CRISIS = "currency_crisis"
    SOVEREIGN_DEFAULT = "sovereign_default"
    STOCK_CRASH_20PCT = "stock_crash_20pct"
    INFLATION_ABOVE_5 = "inflation_above_5"
    DEFLATION = "deflation"
    CLIMATE_TIPPING = "climate_tipping"
    TECHNOLOGY_BREAKTHROUGH = "technology_breakthrough"
    REGIME_CHANGE = "regime_change"
    SOCIAL_REVOLUTION = "social_revolution"
    ENERGY_CRISIS = "energy_crisis"
    CYBER_ATTACK_MAJOR = "cyber_attack_major"


BASE_RATE_DATA = {
    BaseRateCategory.RECESSION_1Y: {"annual_rate": 0.12, "n_years_data": 75, "n_events": 9, "confidence": 0.8},
    BaseRateCategory.RECESSION_5Y: {"annual_rate": 0.45, "n_years_data": 75, "n_events": 9, "confidence": 0.75},
    BaseRateCategory.FINANCIAL_CRISIS: {"annual_rate": 0.03, "n_years_data": 100, "n_events": 3, "confidence": 0.6},
    BaseRateCategory.GEOPOLITICAL_CONFLICT: {"annual_rate": 0.15, "n_years_data": 75, "n_events": 11, "confidence": 0.5},
    BaseRateCategory.PANDEMIC: {"annual_rate": 0.02, "n_years_data": 100, "n_events": 2, "confidence": 0.4},
    BaseRateCategory.HOUSING_CRASH: {"annual_rate": 0.04, "n_years_data": 50, "n_events": 2, "confidence": 0.55},
    BaseRateCategory.CURRENCY_CRISIS: {"annual_rate": 0.05, "n_years_data": 50, "n_events": 3, "confidence": 0.6},
    BaseRateCategory.STOCK_CRASH_20PCT: {"annual_rate": 0.08, "n_years_data": 95, "n_events": 8, "confidence": 0.7},
    BaseRateCategory.INFLATION_ABOVE_5: {"annual_rate": 0.15, "n_years_data": 75, "n_events": 11, "confidence": 0.75},
    BaseRateCategory.DEFLATION: {"annual_rate": 0.03, "n_years_data": 75, "n_events": 2, "confidence": 0.5},
    BaseRateCategory.TECHNOLOGY_BREAKTHROUGH: {"annual_rate": 0.2, "n_years_data": 30, "n_events": 6, "confidence": 0.4},
    BaseRateCategory.REGIME_CHANGE: {"annual_rate": 0.1, "n_years_data": 50, "n_events": 5, "confidence": 0.45},
    BaseRateCategory.ENERGY_CRISIS: {"annual_rate": 0.08, "n_years_data": 50, "n_events": 4, "confidence": 0.55},
    BaseRateCategory.CYBER_ATTACK_MAJOR: {"annual_rate": 0.1, "n_years_data": 20, "n_events": 2, "confidence": 0.35},
}


@dataclass
class ForecastDecomposition:
    category: str
    base_rate: float
    outside_view_probability: float
    inside_view_probability: float
    combined_prior: float
    evidence_strength: float
    posterior_after_evidence: float
    extremized_probability: float
    extremizing_factor: float
    method: str = "superforecasting"


@dataclass
class SuperforecastResult:
    category: str
    final_probability: float
    decomposition: ForecastDecomposition
    confidence_interval: Tuple[float, float]
    updates_log: List[Dict]
    brier_score_estimate: float
    recommendations: List[str]
    method: str = "superforecasting"


class BaseRateDatabase:
    """Reference class forecasting with historical base rates."""

    def get_base_rate(self, category: BaseRateCategory) -> Dict:
        data = BASE_RATE_DATA.get(category, {})
        if not data:
            return {"annual_rate": 0.1, "n_years_data": 50, "n_events": 5, "confidence": 0.3}
        return data

    def get_adjusted_rate(self, category: BaseRateCategory, horizon_years: float = 1.0) -> float:
        data = self.get_base_rate(category)
        annual = data.get("annual_rate", 0.1)
        n_years = data.get("n_years_data", 50)
        confidence = data.get("confidence", 0.5)
        horizon_rate = 1 - (1 - annual) ** horizon_years
        shrinkage = 0.3 * (1 - confidence)
        adjusted = annual * (1 - shrinkage) + 0.5 * shrinkage
        return max(0.01, min(0.99, adjusted))

    def list_categories(self) -> List[str]:
        return [c.value for c in BaseRateCategory]


class SuperforecastEngine:
    """Combines inside/outside view with Bayesian updating and extremizing."""

    def __init__(self):
        self.base_rates = BaseRateDatabase()
        self.update_history: List[Dict] = []

    def _bayes_update(self, prior: float, likelihood_if_event: float,
                      likelihood_if_no_event: float) -> float:
        numerator = prior * likelihood_if_event
        denominator = numerator + (1 - prior) * likelihood_if_no_event
        if denominator < 1e-10:
            return 0.99 if numerator > 0 else 0.01
        return max(0.01, min(0.99, numerator / denominator))

    def _extremize(self, probability: float, evidence_strength: float,
                    extremizing_factor: float = 2.2) -> float:
        if evidence_strength < 0.1:
            return probability
        odds = probability / (1 - probability + 1e-10)
        log_odds = math.log(odds + 1e-10)
        extremized_log_odds = log_odds * extremizing_factor
        extremized_odds = math.exp(extremized_log_odds)
        return extremized_odds / (1 + extremized_odds)

    def forecast(self, category,
                  inside_view_prob: float = 0.5,
                  evidence_signals: Optional[List[Dict]] = None,
                  horizon_years: float = 1.0) -> SuperforecastResult:
        if isinstance(category, str):
            category = BaseRateCategory(category)
        base_data = self.base_rates.get_base_rate(category)
        base_rate = base_data.get("annual_rate", 0.1)
        base_confidence = base_data.get("confidence", 0.5)
        outside_view = self.base_rates.get_adjusted_rate(category, horizon_years)
        combined_prior = 0.6 * outside_view + 0.4 * inside_view_prob
        posterior = combined_prior
        updates = []
        if evidence_signals:
            for signal in evidence_signals:
                strength = signal.get("strength", 0.5)
                direction = signal.get("direction", "toward")
                source = signal.get("source", "unknown")
                if direction == "toward":
                    likelihood_if = 0.5 + 0.5 * strength
                    likelihood_if_not = 0.5 - 0.25 * strength
                else:
                    likelihood_if = 0.5 - 0.25 * strength
                    likelihood_if_not = 0.5 + 0.5 * strength
                posterior = self._bayes_update(posterior, likelihood_if, likelihood_if_not)
                updates.append({
                    "source": source, "direction": direction,
                    "strength": round(strength, 3),
                    "prior_before": round(combined_prior, 4),
                    "posterior_after": round(posterior, 4),
                })
        evidence_strength = min(1.0, len(updates) * 0.15 + abs(inside_view_prob - outside_view) * 2)
        extremized = self._extremize(posterior, evidence_strength)
        ci_width = 2 * math.sqrt(extremized * (1 - extremized)) * (1 - base_confidence * 0.5)
        ci_lower = max(0.01, extremized - ci_width)
        ci_upper = min(0.99, extremized + ci_width)
        brier_estimate = (extremized - (1 if extremized > 0.5 else 0)) ** 2 + 0.05
        recs = []
        if abs(extremized - base_rate) > 0.15:
            recs.append(f"Forecast {extremized:.0%} deviates significantly from {base_rate:.0%} base rate — verify evidence")
        if evidence_strength < 0.3:
            recs.append("Weak evidence — anchor more heavily on base rate")
        if inside_view_prob > 0.8 and base_rate < 0.2:
            recs.append("Inside view very optimistic vs base rate — check for overconfidence")
        decomposition = ForecastDecomposition(
            category=category.value,
            base_rate=round(base_rate, 4),
            outside_view_probability=round(outside_view, 4),
            inside_view_probability=round(inside_view_prob, 4),
            combined_prior=round(combined_prior, 4),
            evidence_strength=round(evidence_strength, 4),
            posterior_after_evidence=round(posterior, 4),
            extremized_probability=round(extremized, 4),
            extremizing_factor=2.2,
        )
        self.update_history.extend(updates)
        return SuperforecastResult(
            category=category.value,
            final_probability=round(extremized, 4),
            decomposition=decomposition,
            confidence_interval=(round(ci_lower, 4), round(ci_upper, 4)),
            updates_log=updates,
            brier_score_estimate=round(brier_estimate, 4),
            recommendations=recs,
        )