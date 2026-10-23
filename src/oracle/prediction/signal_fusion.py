"""Signal Fusion Engine — Combine signals from multiple modules into unified assessments.

Applies principled signal fusion from multiple prediction modules:
1. Bayesian pooling of independent signals
2. Reliability-weighted averaging
3. Correlation-adjusted combination (reduces double-counting)
4. Consensus detection (signals pointing the same way)
5. Conflict detection (signals pointing opposite ways)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class SignalType(Enum):
    DIRECTION = "direction"
    PROBABILITY = "probability"
    RANKING = "ranking"


class FusionMethod(Enum):
    BAYESIAN_POOL = "bayesian_pool"
    RELIABILITY_WEIGHTED = "reliability_weighted"
    CORRELATION_ADJUSTED = "correlation_adjusted"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"


@dataclass
class ModuleSignal:
    module: str
    variable: str
    probability: float
    confidence: float
    direction: float
    reliability: float = 0.7
    weight: float = 1.0


@dataclass
class FusionResult:
    variable: str
    fused_probability: float
    fused_confidence: float
    method: FusionMethod
    n_signals: int
    consensus: float
    conflict: float
    individual_signals: Dict[str, float]
    reliability_weights: Dict[str, float]


MODULE_RELIABILITIES: Dict[str, float] = {
    "superforecast": 0.85, "analogical_reasoning": 0.70, "precursor_pattern": 0.72,
    "cross_impact": 0.68, "prediction_market": 0.75, "collective_intelligence": 0.73,
    "probabilistic_network": 0.78, "delphi": 0.70, "morphological": 0.55,
    "regime_shift": 0.65, "butterfly_effect": 0.60, "temporal_causal": 0.67,
    "horizon_scanning": 0.58, "black_swan": 0.55, "precondition": 0.65,
}


class SignalFusionEngine:
    def __init__(self, module_reliabilities: Optional[Dict[str, float]] = None,
                 default_reliability: float = 0.6):
        self.reliabilities = module_reliabilities or dict(MODULE_RELIABILITIES)
        self.default_reliability = default_reliability
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}

    def fuse(self, signals: List[ModuleSignal],
             method: FusionMethod = FusionMethod.BAYESIAN_POOL) -> Optional[FusionResult]:
        if not signals:
            return None
        variable = signals[0].variable
        for s in signals:
            s.reliability = self.reliabilities.get(s.module, self.default_reliability)
            s.weight = s.confidence * s.reliability
        if method == FusionMethod.BAYESIAN_POOL:
            return self._bayesian_pool(signals, variable)
        elif method == FusionMethod.RELIABILITY_WEIGHTED:
            return self._reliability_weighted(signals, variable)
        elif method == FusionMethod.CORRELATION_ADJUSTED:
            return self._correlation_adjusted(signals, variable)
        elif method == FusionMethod.MEDIAN:
            return self._median_fusion(signals, variable)
        elif method == FusionMethod.TRIMMED_MEAN:
            return self._trimmed_mean(signals, variable)
        return self._bayesian_pool(signals, variable)

    def fuse_multiple(self, signal_groups: Dict[str, List[ModuleSignal]],
                       method: FusionMethod = FusionMethod.BAYESIAN_POOL) -> Dict[str, FusionResult]:
        results = {}
        for variable, signals in signal_groups.items():
            result = self.fuse(signals, method)
            if result:
                results[variable] = result
        return results

    def detect_consensus(self, signals: List[ModuleSignal]) -> float:
        if len(signals) < 2:
            return 1.0
        directions = [s.direction for s in signals]
        mean_dir = np.mean(directions)
        if abs(mean_dir) < 0.01:
            return 0.0
        agreement = sum(1 for d in directions if np.sign(d) == np.sign(mean_dir)) / len(directions)
        return round(agreement, 4)

    def detect_conflict(self, signals: List[ModuleSignal]) -> float:
        if len(signals) < 2:
            return 0.0
        probs = [s.probability for s in signals]
        std = np.std(probs)
        directions = [s.direction for s in signals]
        pos = sum(1 for d in directions if d > 0.1)
        neg = sum(1 for d in directions if d < -0.1)
        dir_conflict = min(pos, neg) / max(pos + neg, 1)
        total_conflict = 0.5 * dir_conflict + 0.5 * min(1.0, std * 5)
        return round(total_conflict, 4)

    def set_correlation(self, module_a: str, module_b: str, correlation: float) -> None:
        self.correlation_matrix[(module_a, module_b)] = correlation
        self.correlation_matrix[(module_b, module_a)] = correlation

    def _bayesian_pool(self, signals: List[ModuleSignal], variable: str) -> FusionResult:
        prior_log_odds = 0.0
        log_odds_sum = prior_log_odds
        for s in signals:
            p = max(0.01, min(0.99, s.probability))
            reliability = s.reliability
            signal_strength = abs(s.direction) * reliability
            log_odds = math.log(p / (1 - p))
            log_odds_sum += log_odds * signal_strength
        fused_p = 1.0 / (1.0 + math.exp(-log_odds_sum))
        fused_p = max(0.01, min(0.99, fused_p))
        individual = {s.module: round(s.probability, 4) for s in signals}
        weights = {s.module: round(s.weight, 4) for s in signals}
        confidence = self._compute_confidence(signals, fused_p)
        consensus = self.detect_consensus(signals)
        conflict = self.detect_conflict(signals)
        return FusionResult(
            variable=variable, fused_probability=round(fused_p, 4),
            fused_confidence=round(confidence, 4),
            method=FusionMethod.BAYESIAN_POOL, n_signals=len(signals),
            consensus=consensus, conflict=conflict,
            individual_signals=individual, reliability_weights=weights,
        )

    def _reliability_weighted(self, signals: List[ModuleSignal], variable: str) -> FusionResult:
        if not signals:
            return None
        weights = [s.weight for s in signals]
        total_weight = sum(weights) or 1.0
        probs = [s.probability for s in signals]
        fused_p = sum(p * w for p, w in zip(probs, weights)) / total_weight
        individual = {s.module: round(s.probability, 4) for s in signals}
        weights_dict = {s.module: round(s.weight, 4) for s in signals}
        confidence = self._compute_confidence(signals, fused_p)
        consensus = self.detect_consensus(signals)
        conflict = self.detect_conflict(signals)
        return FusionResult(
            variable=variable, fused_probability=round(fused_p, 4),
            fused_confidence=round(confidence, 4),
            method=FusionMethod.RELIABILITY_WEIGHTED, n_signals=len(signals),
            consensus=consensus, conflict=conflict,
            individual_signals=individual, reliability_weights=weights_dict,
        )

    def _correlation_adjusted(self, signals: List[ModuleSignal], variable: str) -> FusionResult:
        n = len(signals)
        if n < 2:
            return self._reliability_weighted(signals, variable)
        avg_corr = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                key = (signals[i].module, signals[j].module)
                avg_corr += self.correlation_matrix.get(key, 0.3)
        avg_corr = avg_corr / max(n * (n - 1) / 2, 1)
        effective_n = n / (1 + (n - 1) * avg_corr)
        weights = [s.weight for s in signals]
        total_weight = sum(weights) or 1.0
        probs = [s.probability for s in signals]
        simple_fused = sum(p * w for p, w in zip(probs, weights)) / total_weight
        shrinkage = min(1.0, effective_n / max(n, 1))
        prior = 0.5
        fused_p = shrinkage * simple_fused + (1 - shrinkage) * prior
        individual = {s.module: round(s.probability, 4) for s in signals}
        weights_dict = {s.module: round(s.weight, 4) for s in signals}
        confidence = self._compute_confidence(signals, fused_p) * shrinkage
        consensus = self.detect_consensus(signals)
        conflict = self.detect_conflict(signals)
        return FusionResult(
            variable=variable, fused_probability=round(fused_p, 4),
            fused_confidence=round(confidence, 4),
            method=FusionMethod.CORRELATION_ADJUSTED, n_signals=len(signals),
            consensus=consensus, conflict=conflict,
            individual_signals=individual, reliability_weights=weights_dict,
        )

    def _median_fusion(self, signals: List[ModuleSignal], variable: str) -> FusionResult:
        probs = [s.probability for s in signals]
        fused_p = float(np.median(probs))
        individual = {s.module: round(s.probability, 4) for s in signals}
        weights_dict = {s.module: round(s.weight, 4) for s in signals}
        confidence = self._compute_confidence(signals, fused_p)
        consensus = self.detect_consensus(signals)
        conflict = self.detect_conflict(signals)
        return FusionResult(
            variable=variable, fused_probability=round(fused_p, 4),
            fused_confidence=round(confidence, 4),
            method=FusionMethod.MEDIAN, n_signals=len(signals),
            consensus=consensus, conflict=conflict,
            individual_signals=individual, reliability_weights=weights_dict,
        )

    def _trimmed_mean(self, signals: List[ModuleSignal], variable: str) -> FusionResult:
        probs = sorted([s.probability for s in signals])
        trim = max(1, len(probs) // 4)
        if len(probs) > 2 * trim:
            trimmed = probs[trim:-trim]
        else:
            trimmed = probs
        fused_p = float(np.mean(trimmed)) if trimmed else 0.5
        individual = {s.module: round(s.probability, 4) for s in signals}
        weights_dict = {s.module: round(s.weight, 4) for s in signals}
        confidence = self._compute_confidence(signals, fused_p)
        consensus = self.detect_consensus(signals)
        conflict = self.detect_conflict(signals)
        return FusionResult(
            variable=variable, fused_probability=round(fused_p, 4),
            fused_confidence=round(confidence, 4),
            method=FusionMethod.TRIMMED_MEAN, n_signals=len(signals),
            consensus=consensus, conflict=conflict,
            individual_signals=individual, reliability_weights=weights_dict,
        )

    def _compute_confidence(self, signals: List[ModuleSignal], fused_p: float) -> float:
        if len(signals) < 2:
            return signals[0].confidence if signals else 0.5
        probs = [s.probability for s in signals]
        std = np.std(probs)
        agreement = 1.0 - min(1.0, std * 5)
        avg_confidence = np.mean([s.confidence for s in signals])
        avg_reliability = np.mean([s.reliability for s in signals])
        n_bonus = min(0.2, len(signals) * 0.02)
        return min(0.95, agreement * 0.4 + avg_confidence * 0.3 + avg_reliability * 0.2 + n_bonus)