"""Collective Intelligence Aggregator — Wisdom-of-crowds with confidence weighting.

Aggregates predictions from multiple sources using advanced weighting schemes:
1. Simple averaging (unweighted baseline)
2. Confidence-weighted averaging
3. Past-performance (Brier score) weighted averaging
4. Extremized means (Tetlock-style)
5. Logarithmic opinion pool (geometric mean of odds)
6. Bayesian model averaging
7. Decorrelation-weighted (penalize correlated forecasts)

Key insight: The wisdom of crowds works best when forecasts are:
- Diverse (independent errors)
- Weighted by past accuracy (disciplined forecasters)
- Extremized (pulled toward 0 or 1, away from 50/50)
- Calibrated (past predictions match realized frequencies)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    SIMPLE_AVERAGE = "simple_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    EXTREMIZED = "extremized"
    LOGARITHMIC_POOL = "logarithmic_pool"
    BAYESIAN_MODEL_AVERAGE = "bayesian_model_average"
    DECORRELATION_WEIGHTED = "decorrelation_weighted"
    BEST = "best"


@dataclass
class ForecasterProfile:
    name: str
    expertise: List[str] = field(default_factory=list)
    brier_scores: List[float] = field(default_factory=list)
    calibration_slope: float = 1.0
    overconfidence: float = 0.0
    n_forecasts: int = 0
    weight: float = 1.0


@dataclass
class AggregatedForecast:
    event: str
    probability: float
    confidence_interval: Tuple[float, float]
    method: AggregationMethod
    n_forecasters: int
    individual_forecasts: Dict[str, float]
    individual_weights: Dict[str, float]
    extremizing_factor: float
    diversity_index: float
    is_extremized: bool
    recommended_action: str


class CollectiveIntelligenceAggregator:
    def __init__(self, extremizing_range: Tuple[float, float] = (1.1, 1.5)):
        self.extremizing_range = extremizing_range
        self.forecasters: Dict[str, ForecasterProfile] = {}
        self.forecast_history: List[Dict] = []

    def register_forecaster(self, name: str, expertise: Optional[List[str]] = None,
                              initial_weight: float = 1.0) -> ForecasterProfile:
        profile = ForecasterProfile(
            name=name,
            expertise=expertise or [],
            weight=initial_weight,
        )
        self.forecasters[name] = profile
        return profile

    def update_forecaster_performance(self, name: str, forecast_prob: float,
                                        realized_outcome: float) -> None:
        if name not in self.forecasters:
            self.register_forecaster(name)
        brier = (forecast_prob - realized_outcome) ** 2
        self.forecasters[name].brier_scores.append(brier)
        self.forecasters[name].n_forecasts += 1
        n = self.forecasters[name].n_forecasts
        if n > 5:
            past_probs = [f["probability"] for f in self.forecast_history
                         if f.get("forecaster") == name and f.get("realized") is not None]
            past_outcomes = [f["realized"] for f in self.forecast_history
                           if f.get("forecaster") == name and f.get("realized") is not None]
            if past_probs and past_outcomes:
                probs = np.array(past_probs)
                outcomes = np.array(past_outcomes)
                self.forecasters[name].calibration_slope = float(np.polyfit(probs, outcomes, 1)[0]) if len(probs) > 2 else 1.0
                predicted_yes = np.mean(probs)
                realized_yes = np.mean(outcomes)
                self.forecasters[name].overconfidence = float(predicted_yes - realized_yes)

    def aggregate(self, event: str, forecasts: Dict[str, float],
                  method: AggregationMethod = AggregationMethod.BEST,
                  domain: Optional[str] = None) -> AggregatedForecast:
        if not forecasts:
            return AggregatedForecast(
                event=event, probability=0.5, confidence_interval=(0.4, 0.6),
                method=method, n_forecasters=0, individual_forecasts={},
                individual_weights={}, extremizing_factor=1.0,
                diversity_index=0.0, is_extremized=False,
                recommended_action="Insufficient forecasts — gather more expert input",
            )
        for name in forecasts:
            if name not in self.forecasters:
                self.register_forecaster(name)
        diversity = self._compute_diversity(forecasts)
        if method == AggregationMethod.BEST:
            method = self._select_best_method(forecasts, diversity)
        if method == AggregationMethod.SIMPLE_AVERAGE:
            prob, weights = self._simple_average(forecasts)
            ext_factor = 1.0
            is_ext = False
        elif method == AggregationMethod.CONFIDENCE_WEIGHTED:
            prob, weights = self._confidence_weighted(forecasts, domain)
            ext_factor = 1.0
            is_ext = False
        elif method == AggregationMethod.PERFORMANCE_WEIGHTED:
            prob, weights = self._performance_weighted(forecasts)
            ext_factor = 1.0
            is_ext = False
        elif method == AggregationMethod.EXTREMIZED:
            prob, weights = self._extremized(forecasts, domain)
            ext_factor = self._compute_extremizing_factor(forecasts)
            is_ext = True
        elif method == AggregationMethod.LOGARITHMIC_POOL:
            prob, weights = self._logarithmic_pool(forecasts)
            ext_factor = 1.0
            is_ext = False
        elif method == AggregationMethod.BAYESIAN_MODEL_AVERAGE:
            prob, weights = self._bayesian_model_average(forecasts)
            ext_factor = 1.0
            is_ext = False
        elif method == AggregationMethod.DECORRELATION_WEIGHTED:
            prob, weights = self._decorrelation_weighted(forecasts)
            ext_factor = 1.0
            is_ext = False
        else:
            prob, weights = self._simple_average(forecasts)
            ext_factor = 1.0
            is_ext = False
        prob = max(0.01, min(0.99, prob))
        n = len(forecasts)
        ci_width = 1.96 * math.sqrt(prob * (1 - prob) / max(n, 1)) * (1 + diversity * 0.5)
        ci_lower = max(0.01, prob - ci_width)
        ci_upper = min(0.99, prob + ci_width)
        action = self._recommend_action(prob, diversity, method)
        for name, f_prob in forecasts.items():
            self.forecast_history.append({
                "forecaster": name,
                "event": event,
                "probability": f_prob,
                "aggregated_probability": round(prob, 4),
                "method": method.value,
                "realized": None,
            })
        if len(self.forecast_history) > 5000:
            self.forecast_history = self.forecast_history[-5000:]
        return AggregatedForecast(
            event=event, probability=round(prob, 4),
            confidence_interval=(round(ci_lower, 4), round(ci_upper, 4)),
            method=method, n_forecasters=n,
            individual_forecasts={k: round(v, 4) for k, v in forecasts.items()},
            individual_weights={k: round(v, 4) for k, v in weights.items()},
            extremizing_factor=round(ext_factor, 4),
            diversity_index=round(diversity, 4),
            is_extremized=is_ext,
            recommended_action=action,
        )

    def aggregate_multiple_events(self, events: Dict[str, Dict[str, float]],
                                    method: AggregationMethod = AggregationMethod.BEST,
                                    domain: Optional[str] = None) -> Dict[str, AggregatedForecast]:
        return {event: self.aggregate(event, forecasts, method, domain)
                for event, forecasts in events.items()}

    def resolve_forecast(self, event: str, realized_outcome: float) -> int:
        """Record the realized outcome for a past forecast, enabling calibration updates.

        Args:
            event: The event name that was forecasted.
            realized_outcome: 1.0 if the event occurred, 0.0 if it did not.

        Returns:
            Number of forecast entries resolved.
        """
        resolved = 0
        for entry in self.forecast_history:
            if entry.get("event") == event and entry.get("realized") is None:
                entry["realized"] = realized_outcome
                name = entry["forecaster"]
                f_prob = entry["probability"]
                self.update_forecaster_performance(name, f_prob, realized_outcome)
                resolved += 1
        return resolved

    def _simple_average(self, forecasts: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        n = len(forecasts)
        weights = {name: 1.0 / n for name in forecasts}
        prob = sum(forecasts.values()) / n
        return prob, weights

    def _confidence_weighted(self, forecasts: Dict[str, float],
                              domain: Optional[str] = None) -> Tuple[float, Dict[str, float]]:
        weights = {}
        for name in forecasts:
            forecaster = self.forecasters.get(name, ForecasterProfile(name=name))
            expertise_bonus = 1.0
            if domain and forecaster.expertise:
                expertise_bonus = 1.5 if domain in forecaster.expertise else 0.7
            n_bonus = min(1.0, forecaster.n_forecasts / 20.0) if forecaster.n_forecasts > 0 else 0.5
            brier_penalty = 1.0
            if forecaster.brier_scores:
                avg_brier = np.mean(forecaster.brier_scores[-10:])
                brier_penalty = max(0.1, 1.0 - avg_brier)
            weight = expertise_bonus * (0.5 + 0.5 * n_bonus) * brier_penalty
            weights[name] = weight
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}
        prob = sum(forecasts[name] * weights[name] for name in forecasts)
        return prob, weights

    def _performance_weighted(self, forecasts: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        weights = {}
        for name in forecasts:
            forecaster = self.forecasters.get(name, ForecasterProfile(name=name))
            if forecaster.brier_scores and len(forecaster.brier_scores) >= 3:
                avg_brier = float(np.mean(forecaster.brier_scores[-20:]))
                weight = max(0.01, (1.0 - avg_brier) ** 2)
            else:
                weight = 0.5
            weights[name] = weight
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}
        prob = sum(forecasts[name] * weights[name] for name in forecasts)
        return prob, weights

    def _extremized(self, forecasts: Dict[str, float],
                     domain: Optional[str] = None) -> Tuple[float, Dict[str, float]]:
        base_prob, weights = self._confidence_weighted(forecasts, domain)
        n = len(forecasts)
        diversity = self._compute_diversity(forecasts)
        ext_factor = 1.0 + 0.4 * math.log(n) / math.log(max(n, 2)) * diversity
        ext_factor = max(self.extremizing_range[0], min(self.extremizing_range[1], ext_factor))
        if base_prob > 0.5:
            ext_prob = 0.5 + ext_factor * (base_prob - 0.5)
        else:
            ext_prob = 0.5 - ext_factor * (0.5 - base_prob)
        return ext_prob, weights

    def _logarithmic_pool(self, forecasts: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        log_odds = []
        weights = {}
        eps = 0.01
        for name, prob in forecasts.items():
            p = max(eps, min(1 - eps, prob))
            log_odds.append(math.log(p / (1 - p)))
            weights[name] = 1.0 / len(forecasts)
        avg_log_odds = np.mean(log_odds)
        pooled_prob = 1.0 / (1.0 + math.exp(-avg_log_odds))
        return pooled_prob, weights

    def _bayesian_model_average(self, forecasts: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        weights = {}
        for name in forecasts:
            forecaster = self.forecasters.get(name, ForecasterProfile(name=name))
            if forecaster.brier_scores:
                likelihood = math.exp(-np.mean(forecaster.brier_scores[-10:]) * 5)
                prior = 0.5
                posterior = likelihood * prior
            else:
                posterior = 0.5
            weights[name] = posterior
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}
        prob = sum(forecasts[name] * weights[name] for name in forecasts)
        return prob, weights

    def _decorrelation_weighted(self, forecasts: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        n = len(forecasts)
        if n <= 2:
            return self._simple_average(forecasts)
        names = list(forecasts.keys())
        values = np.array([forecasts[n] for n in names])
        mean_prob = np.mean(values)
        diversity = self._compute_diversity(forecasts)
        weights = {}
        for name in names:
            forecaster = self.forecasters.get(name, ForecasterProfile(name=name))
            uniqueness = max(0.1, 1.0 - abs(forecasts[name] - mean_prob) / max(0.5, math.sqrt(max(mean_prob * (1 - mean_prob), 0.01)) * 2))
            expertise = max(0.3, 1.0 - float(np.mean(forecaster.brier_scores[-10:]))) if forecaster.brier_scores else 0.5
            weights[name] = uniqueness * expertise
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}
        prob = sum(forecasts[name] * weights[name] for name in forecasts)
        return prob, weights

    def _compute_diversity(self, forecasts: Dict[str, float]) -> float:
        if len(forecasts) < 2:
            return 0.0
        values = np.array(list(forecasts.values()))
        std = np.std(values)
        max_diversity = np.sqrt(0.25)
        return min(1.0, std / max(max_diversity, 1e-10))

    def _compute_extremizing_factor(self, forecasts: Dict[str, float]) -> float:
        n = len(forecasts)
        diversity = self._compute_diversity(forecasts)
        base = 1.0 + 0.2 * math.log(max(n, 1))
        return min(self.extremizing_range[1], base * diversity)

    def _select_best_method(self, forecasts: Dict[str, float],
                              diversity: float) -> AggregationMethod:
        has_performance = any(
            self.forecasters.get(name, ForecasterProfile(name=name)).brier_scores
            for name in forecasts
        )
        n = len(forecasts)
        if has_performance and n >= 3:
            return AggregationMethod.EXTREMIZED
        elif diversity > 0.3 and n >= 3:
            return AggregationMethod.LOGARITHMIC_POOL
        elif has_performance:
            return AggregationMethod.PERFORMANCE_WEIGHTED
        elif n >= 5:
            return AggregationMethod.DECORRELATION_WEIGHTED
        return AggregationMethod.CONFIDENCE_WEIGHTED

    def _recommend_action(self, prob: float, diversity: float,
                            method: AggregationMethod) -> str:
        if prob > 0.8:
            return f"High confidence ({prob:.0%}) with {diversity:.0%} forecast diversity — strong signal for YES"
        elif prob > 0.6:
            return f"Moderate confidence ({prob:.0%}) — lean YES with {diversity:.0%} diversity"
        elif prob > 0.4:
            return f"Uncertain ({prob:.0%}) — near coin flip, gather more expert input"
        elif prob > 0.2:
            return f"Moderate confidence ({1-prob:.0%}) — lean NO with {diversity:.0%} diversity"
        else:
            return f"High confidence ({1-prob:.0%}) — strong signal for NO"