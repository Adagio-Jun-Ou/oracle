"""Multi-objective fitness evaluation for prediction genome evolution.

Evaluates prediction strategies against historical outcomes using:
- Brier score (calibration)
- Calibration slope/intercept
- Confidence reliability
- Cascade prediction accuracy
- Cross-domain synthesis score
- Spread/coverage/direction metrics
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from oracle.evolution.map_elites import BehavioralDescriptor, FitnessScore


@dataclass
class FitnessComponent:
    name: str
    value: float
    weight: float
    contribution: float = 0.0
    description: str = ""

    def __post_init__(self):
        self.contribution = self.value * self.weight


@dataclass
class FitnessResult:
    fitness: FitnessScore
    components: List[FitnessComponent] = field(default_factory=list)
    behavioral_descriptor: Optional[BehavioralDescriptor] = None
    predictions: Dict[str, float] = field(default_factory=dict)
    outcomes: Dict[str, float] = field(default_factory=dict)
    evaluation_time: float = 0.0
    genome_id: str = ""
    domain: str = "economic"


@dataclass
class BacktestWindow:
    start_time: float = 0.0
    end_time: float = 0.0
    predictions: Dict[str, float] = field(default_factory=dict)
    outcomes: Dict[str, float] = field(default_factory=dict)
    domain: str = ""


@dataclass
class OutcomeRecord:
    timestamp: float
    domain: str
    predicted: float
    actual: float
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class OutcomeTracker:
    """Tracks prediction outcomes for fitness evaluation."""

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._records: List[OutcomeRecord] = []
        self._by_domain: Dict[str, List[OutcomeRecord]] = defaultdict(list)

    def record(self, domain: str, predicted: float, actual: float,
               confidence: float = 0.5, metadata: Optional[Dict] = None) -> None:
        rec = OutcomeRecord(
            timestamp=time.time(), domain=domain,
            predicted=np.clip(predicted, 0.01, 0.99),
            actual=np.clip(actual, 0.0, 1.0),
            confidence=confidence, metadata=metadata or {}
        )
        self._records.append(rec)
        self._by_domain[domain].append(rec)
        if len(self._records) > self.max_history:
            oldest = self._records.pop(0)
            if oldest.domain in self._by_domain:
                domain_list = self._by_domain[oldest.domain]
                if domain_list and domain_list[0].timestamp == oldest.timestamp:
                    domain_list.pop(0)

    def get_records(self, domain: Optional[str] = None,
                    since: float = 0.0) -> List[OutcomeRecord]:
        if domain:
            records = self._by_domain.get(domain, [])
            return [r for r in records if r.timestamp >= since]
        return [r for r in self._records if r.timestamp >= since]

    def get_recent(self, n: int = 100, domain: Optional[str] = None) -> List[OutcomeRecord]:
        records = self.get_records(domain)
        return records[-n:]

    def compute_brier(self, domain: Optional[str] = None) -> float:
        records = self.get_recent(500, domain)
        if not records:
            return 0.25
        brier_sum = sum((r.predicted - r.actual) ** 2 for r in records)
        return brier_sum / len(records)

    def compute_calibration(self, domain: Optional[str] = None) -> float:
        records = self.get_recent(500, domain)
        if len(records) < 5:
            return 0.5
        bins = defaultdict(list)
        for r in records:
            bin_idx = min(int(r.predicted * 10), 9)
            bins[bin_idx].append(r)
        total_error = 0.0
        for bin_idx, bin_records in bins.items():
            avg_pred = np.mean([r.predicted for r in bin_records])
            avg_actual = np.mean([r.actual for r in bin_records])
            total_error += abs(avg_pred - avg_actual) * len(bin_records)
        return max(0, 1.0 - total_error / len(records))

    def compute_confidence_reliability(self, domain: Optional[str] = None) -> float:
        records = self.get_recent(500, domain)
        if len(records) < 5:
            return 0.5
        accuracy = np.mean([1.0 if abs(r.predicted - r.actual) < 0.2 else 0.0 for r in records])
        avg_confidence = np.mean([r.confidence for r in records])
        return 1.0 - abs(accuracy - avg_confidence)

    def compute_direction_accuracy(self, domain: Optional[str] = None) -> float:
        records = self.get_recent(200, domain)
        if len(records) < 5:
            return 0.5
        correct = 0
        for i in range(1, len(records)):
            pred_dir = records[i].predicted - records[i - 1].predicted
            actual_dir = records[i].actual - records[i - 1].actual
            if pred_dir * actual_dir > 0:
                correct += 1
            elif pred_dir == 0 and actual_dir == 0:
                correct += 1
        return correct / max(len(records) - 1, 1)

    def compute_cascade_accuracy(self, domain: Optional[str] = None) -> float:
        records = self.get_recent(200, domain)
        if len(records) < 10:
            return 0.5
        predictions = np.array([r.predicted for r in records])
        actuals = np.array([r.actual for r in records])
        if len(predictions) < 3:
            return 0.5
        pred_changes = np.diff(predictions)
        actual_changes = np.diff(actuals)
        extreme_mask = np.abs(actual_changes) > 0.1
        if np.sum(extreme_mask) < 2:
            return 0.5
        pred_extreme = pred_changes[extreme_mask]
        actual_extreme = actual_changes[extreme_mask]
        correlation = np.corrcoef(pred_extreme, actual_extreme)[0, 1] if len(pred_extreme) > 1 else 0.0
        return float(max(0, (correlation + 1) / 2)) if not np.isnan(correlation) else 0.5

    def compute_spread(self, domain: Optional[str] = None) -> float:
        records = self.get_recent(200, domain)
        if len(records) < 5:
            return 0.5
        preds = [r.predicted for r in records]
        return float(np.std(preds)) * 2

    def compute_coverage(self, domain: Optional[str] = None) -> float:
        records = self.get_recent(200, domain)
        if not records:
            return 0.0
        predicates = set()
        for r in records:
            bin_idx = int(r.predicted * 10)
            predicates.add(bin_idx)
        return len(predicates) / 10.0

    def summary(self) -> Dict[str, Any]:
        domains = list(self._by_domain.keys())
        return {
            "total_records": len(self._records),
            "domains": domains,
            "domain_counts": {d: len(r) for d, r in self._by_domain.items()},
            "overall_brier": round(self.compute_brier(), 4),
            "overall_calibration": round(self.compute_calibration(), 4)
        }


class MultiObjectiveFitness:
    """Multi-objective fitness evaluation combining multiple metrics."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        outcome_tracker: Optional[OutcomeTracker] = None,
        bd_names: Optional[List[str]] = None,
        bd_ranges: Optional[List[Tuple[float, float]]] = None,
        seed: int = 42,
    ):
        self.weights = weights or {
            "synthesis": 0.30, "calibration": 0.20, "cascade": 0.15,
            "confidence": 0.12, "spread": 0.10, "coverage": 0.08,
            "direction": 0.05
        }
        self.tracker = outcome_tracker or OutcomeTracker()
        self.bd_names = bd_names or ["synthesis_score", "cascade_richness"]
        self.bd_ranges = bd_ranges or [(0.0, 1.0), (0.0, 1.0)]
        self.rng = np.random.RandomState(seed)
        self._fitness_cache: Dict[str, FitnessResult] = {}
        self._max_cache_size = 10000

    def evaluate(
        self,
        genome_predictions: Dict[str, float],
        domain_states: Dict[str, float],
        domain: str = "economic",
        genome_id: str = "",
    ) -> FitnessResult:
        start_time = time.time()
        brier = self.tracker.compute_brier(domain)
        calibration = self.tracker.compute_calibration(domain)
        confidence = self.tracker.compute_confidence_reliability(domain)
        cascade = self.tracker.compute_cascade_accuracy(domain)
        direction = self.tracker.compute_direction_accuracy(domain)
        spread = self.tracker.compute_spread(domain)
        coverage = self.tracker.compute_coverage(domain)
        brier_inverted = max(0, 1.0 - brier)
        synthesis = float(np.mean([brier_inverted, calibration, confidence, cascade, direction]))
        fitness = FitnessScore(
            composite=0.0,
            brier=round(brier_inverted, 4),
            calibration=round(calibration, 4),
            confidence=round(confidence, 4),
            cascade=round(cascade, 4),
            spread=round(spread, 4),
            coverage=round(coverage, 4),
            direction=round(direction, 4),
            synthesis=round(synthesis, 4)
        )
        fitness.composite = round(fitness.weighted_sum(self.weights), 4)
        bd = BehavioralDescriptor(
            values=[synthesis, cascade],
            names=self.bd_names
        )
        components = [
            FitnessComponent("brier_inverted", brier_inverted, self.weights.get("synthesis", 0.3) * 0.3, description="Inverse Brier score"),
            FitnessComponent("calibration", calibration, self.weights.get("calibration", 0.2)),
            FitnessComponent("confidence", confidence, self.weights.get("confidence", 0.12)),
            FitnessComponent("cascade", cascade, self.weights.get("cascade", 0.15)),
            FitnessComponent("direction", direction, self.weights.get("direction", 0.05)),
            FitnessComponent("spread", spread, self.weights.get("spread", 0.10)),
            FitnessComponent("coverage", coverage, self.weights.get("coverage", 0.08)),
            FitnessComponent("synthesis", synthesis, self.weights.get("synthesis", 0.3) * 0.7),
        ]
        result = FitnessResult(
            fitness=fitness, components=components,
            behavioral_descriptor=bd,
            predictions=genome_predictions,
            domain=domain,
            evaluation_time=time.time() - start_time,
            genome_id=genome_id
        )
        if genome_id:
            if len(self._fitness_cache) >= self._max_cache_size:
                keys = list(self._fitness_cache.keys())
                for old_key in keys[:len(keys) // 4]:
                    del self._fitness_cache[old_key]
            self._fitness_cache[genome_id] = result
        return result

    def evaluate_against_outcomes(
        self,
        predictions: Dict[str, float],
        outcomes: Dict[str, float],
        domain: str = "economic",
        genome_id: str = "",
    ) -> FitnessResult:
        """Evaluate fitness by directly comparing predictions to known outcomes."""
        if not predictions or not outcomes:
            fitness = FitnessScore(synthesis=0.5, calibration=0.5, cascade=0.5)
            fitness.composite = 0.5
            return FitnessResult(fitness=fitness, genome_id=genome_id, domain=domain)
        matched = {k: (predictions[k], outcomes[k]) for k in predictions if k in outcomes}
        if not matched:
            fitness = FitnessScore(synthesis=0.5, calibration=0.5, cascade=0.5)
            fitness.composite = 0.5
            return FitnessResult(fitness=fitness, genome_id=genome_id, domain=domain)
        pred_arr = np.array([v[0] for v in matched.values()])
        actual_arr = np.array([v[1] for v in matched.values()])
        brier = float(np.mean((pred_arr - actual_arr) ** 2))
        brier_inv = max(0, 1.0 - brier)
        bins_pred = np.clip((pred_arr * 10).astype(int), 0, 9)
        calibration_error = 0.0
        bin_counts = np.zeros(10)
        for b in range(10):
            mask = bins_pred == b
            if np.sum(mask) > 0:
                avg_pred = np.mean(pred_arr[mask])
                avg_actual = np.mean(actual_arr[mask])
                calibration_error += abs(avg_pred - avg_actual) * np.sum(mask)
                bin_counts[b] = np.sum(mask)
        calibration = max(0, 1.0 - calibration_error / max(len(pred_arr), 1))
        if len(pred_arr) > 1:
            pred_diff = np.diff(pred_arr)
            actual_diff = np.diff(actual_arr)
            direction = float(np.mean((pred_diff * actual_diff) > 0))
            cascade_corr = np.corrcoef(pred_arr, actual_arr)[0, 1]
            cascade = max(0, float((cascade_corr + 1) / 2)) if not np.isnan(cascade_corr) else 0.5
        else:
            direction = 0.5
            cascade = 0.5
        confidence = max(0, 1.0 - abs(np.mean(pred_arr) - np.mean(actual_arr)))
        spread = float(np.std(pred_arr)) * 2
        coverage_bins = len(set(bins_pred))
        coverage = coverage_bins / 10.0
        synthesis = float(np.mean([brier_inv, calibration, confidence, cascade, direction]))

        fitness = FitnessScore(
            composite=0.0,
            brier=round(brier_inv, 4), calibration=round(calibration, 4),
            confidence=round(confidence, 4), cascade=round(cascade, 4),
            synthesis=round(synthesis, 4), spread=round(spread, 4),
            coverage=round(coverage, 4), direction=round(direction, 4)
        )
        fitness.composite = round(fitness.weighted_sum(self.weights), 4)
        bd = BehavioralDescriptor(values=[synthesis, cascade], names=self.bd_names)
        return FitnessResult(
            fitness=fitness, behavioral_descriptor=bd,
            predictions=predictions, outcomes=outcomes,
            evaluation_time=0.001, genome_id=genome_id,
            domain=domain
        )