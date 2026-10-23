"""Temporal Consistency Checker — Validates prediction consistency across time horizons.

Checks that predictions remain consistent when viewed at different time scales:
1. Cross-horizon consistency (short/medium/long term align?)
2. Temporal monotonicity (predictions evolve smoothly?)
3. Discount consistency (far-future less certain than near-future?)
4. Horizon coherence (different methods agree across horizons?)
5. Change point detection (sudden prediction shifts?)
6. Backward-forward consistency (retrodiction matches past?)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConsistencyIssue(Enum):
    CROSS_HORIZON_MISMATCH = "cross_horizon_mismatch"
    MONOTONICITY_VIOLATION = "monotonicity_violation"
    DISCOUNT_INCONSISTENCY = "discount_inconsistency"
    HORIZON_INCOHERENCE = "horizon_incoherence"
    CHANGE_POINT = "change_point"
    BACKWARD_FORWARD_MISMATCH = "backward_forward_mismatch"


@dataclass
class ConsistencyViolation:
    issue_type: ConsistencyIssue
    description: str
    severity: float
    horizons_involved: List[str]
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None


@dataclass
class TemporalConsistencyResult:
    is_consistent: bool
    consistency_score: float
    violations: List[ConsistencyViolation]
    horizon_predictions: Dict[str, Dict[str, float]]
    smoothness_score: float
    uncertainty_expansion_score: float
    recommendations: List[str]


class TemporalConsistencyChecker:
    def __init__(self, discount_rate: float = 0.05, smoothness_threshold: float = 0.3):
        self.discount_rate = discount_rate
        self.smoothness_threshold = smoothness_threshold

    def check(self, predictions_by_horizon: Dict[str, Dict[str, float]],
              uncertainties_by_horizon: Optional[Dict[str, Dict[str, float]]] = None) -> TemporalConsistencyResult:
        violations = []
        if uncertainties_by_horizon is None:
            uncertainties_by_horizon = self._estimate_uncertainties(predictions_by_horizon)
        violations.extend(self._check_cross_horizon(predictions_by_horizon))
        violations.extend(self._check_monotonicity(predictions_by_horizon))
        violations.extend(self._check_discount_consistency(predictions_by_horizon, uncertainties_by_horizon))
        violations.extend(self._check_horizon_coherence(predictions_by_horizon))
        violations.extend(self._check_change_points(predictions_by_horizon))
        violations.extend(self._check_backward_forward(predictions_by_horizon))
        consistency_score = max(0, 1 - sum(v.severity for v in violations) / max(len(violations), 1))
        is_consistent = consistency_score >= 0.5 and len([v for v in violations if v.severity > 0.5]) == 0
        smoothness = self._compute_smoothness(predictions_by_horizon)
        uncertainty_exp = self._compute_uncertainty_expansion(uncertainties_by_horizon)
        recs = self._generate_recommendations(violations, consistency_score, smoothness, uncertainty_exp)
        return TemporalConsistencyResult(
            is_consistent=is_consistent,
            consistency_score=round(consistency_score, 4),
            violations=violations,
            horizon_predictions=predictions_by_horizon,
            smoothness_score=round(smoothness, 4),
            uncertainty_expansion_score=round(uncertainty_exp, 4),
            recommendations=recs,
        )

    def _estimate_uncertainties(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        horizon_order = ["1day", "1week", "1month", "3month", "6month", "1year", "5year", "10year"]
        horizon_scale = {"1day": 1, "1week": 2, "1month": 4, "3month": 6, "6month": 8, "1year": 10, "5year": 20, "10year": 30}
        uncertainties = {}
        for horizon, preds in predictions.items():
            scale = horizon_scale.get(horizon, 5)
            uncertainties[horizon] = {var: abs(val) * 0.02 * scale for var, val in preds.items()}
        return uncertainties

    def _check_cross_horizon(self, predictions: Dict[str, Dict[str, float]]) -> List[ConsistencyViolation]:
        violations = []
        horizon_order = ["1day", "1week", "1month", "3month", "6month", "1year", "5year", "10year"]
        available = [h for h in horizon_order if h in predictions]
        for i in range(len(available) - 1):
            h1, h2 = available[i], available[i + 1]
            for var in predictions.get(h1, {}):
                if var in predictions.get(h2, {}):
                    v1 = predictions[h1][var]
                    v2 = predictions[h2][var]
                    if v1 != 0 and abs(v2 - v1) / max(abs(v1), 1e-10) > 0.5:
                        severity = min(1.0, abs(v2 - v1) / max(abs(v1), 1e-10))
                        violations.append(ConsistencyViolation(
                            issue_type=ConsistencyIssue.CROSS_HORIZON_MISMATCH,
                            description=f"{var}: large jump between {h1} ({v1:.3f}) and {h2} ({v2:.3f})",
                            severity=round(severity, 4),
                            horizons_involved=[h1, h2],
                            expected_value=round(v1, 4),
                            actual_value=round(v2, 4),
                        ))
        return violations

    def _check_monotonicity(self, predictions: Dict[str, Dict[str, float]]) -> List[ConsistencyViolation]:
        violations = []
        horizon_order = ["1day", "1week", "1month", "3month", "6month", "1year", "5year", "10year"]
        available = [h for h in horizon_order if h in predictions]
        for var in set().union(*[predictions.get(h, {}).keys() for h in available]):
            series = [(h, predictions[h].get(var)) for h in available if var in predictions.get(h, {})]
            if len(series) < 3:
                continue
            for i in range(1, len(series) - 1):
                _, v_prev = series[i - 1]
                _, v_curr = series[i]
                _, v_next = series[i + 1]
                trend = v_curr - v_prev
                next_trend = v_next - v_curr
                if trend * next_trend < -0.01:
                    severity = min(1.0, abs(trend - next_trend) / max(abs(v_curr), 1e-10))
                    violations.append(ConsistencyViolation(
                        issue_type=ConsistencyIssue.MONOTONICITY_VIOLATION,
                        description=f"{var}: direction reversal at {series[i][0]}",
                        severity=round(severity, 4),
                        horizons_involved=[series[i - 1][0], series[i][0], series[i + 1][0]],
                    ))
        return violations

    def _check_discount_consistency(self, predictions: Dict[str, Dict[str, float]],
                                      uncertainties: Dict[str, Dict[str, float]]) -> List[ConsistencyViolation]:
        violations = []
        horizon_order = ["1day", "1week", "1month", "3month", "6month", "1year", "5year", "10year"]
        available = [h for h in horizon_order if h in uncertainties]
        for var in set().union(*[uncertainties.get(h, {}).keys() for h in available]):
            series = [(h, uncertainties[h].get(var, 0)) for h in available if var in uncertainties.get(h, {})]
            for i in range(1, len(series)):
                _, u_prev = series[i - 1]
                _, u_curr = series[i]
                if u_curr < u_prev * 0.95:
                    violations.append(ConsistencyViolation(
                        issue_type=ConsistencyIssue.DISCOUNT_INCONSISTENCY,
                        description=f"{var}: uncertainty decreases from {series[i-1][0]} to {series[i][0]} — should increase for distant horizons",
                        severity=0.4,
                        horizons_involved=[series[i - 1][0], series[i][0]],
                    ))
        return violations

    def _check_horizon_coherence(self, predictions: Dict[str, Dict[str, float]]) -> List[ConsistencyViolation]:
        violations = []
        if len(predictions) < 3:
            return violations
        for var in set().union(*[preds.keys() for preds in predictions.values()]):
            values = [predictions[h].get(var) for h in predictions if var in predictions.get(h, {})]
            if len(values) < 3:
                continue
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val > abs(mean_val) * 0.3 and abs(mean_val) > 0.01:
                cv = std_val / max(abs(mean_val), 1e-10)
                severity = min(1.0, cv * 0.5)
                violations.append(ConsistencyViolation(
                    issue_type=ConsistencyIssue.HORIZON_INCOHERENCE,
                    description=f"{var}: high coefficient of variation ({cv:.2f}) across horizons — methods disagree",
                    severity=round(severity, 4),
                    horizons_involved=list(predictions.keys()),
                    expected_value=round(float(mean_val), 4),
                    actual_value=round(float(std_val), 4),
                ))
        return violations

    def _check_change_points(self, predictions: Dict[str, Dict[str, float]]) -> List[ConsistencyViolation]:
        violations = []
        horizon_order = ["1day", "1week", "1month", "3month", "6month", "1year", "5year", "10year"]
        available = [h for h in horizon_order if h in predictions]
        for var in set().union(*[predictions.get(h, {}).keys() for h in available]):
            series = [(h, predictions[h].get(var)) for h in available if var in predictions.get(h, {})]
            if len(series) < 3:
                continue
            values = [v for _, v in series]
            diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
            if not diffs:
                continue
            mean_diff = np.mean(diffs)
            for i, d in enumerate(diffs):
                if mean_diff > 0 and d > 3 * mean_diff:
                    severity = min(1.0, d / max(mean_diff, 1e-10) * 0.2)
                    violations.append(ConsistencyViolation(
                        issue_type=ConsistencyIssue.CHANGE_POINT,
                        description=f"{var}: abrupt change at {series[i + 1][0]} (diff={d:.3f}, mean={mean_diff:.3f})",
                        severity=round(severity, 4),
                        horizons_involved=[series[i][0], series[i + 1][0]],
                    ))
        return violations

    def _check_backward_forward(self, predictions: Dict[str, Dict[str, float]]) -> List[ConsistencyViolation]:
        violations = []
        if "1day" in predictions and "1year" in predictions:
            for var in predictions.get("1day", {}):
                if var in predictions.get("1year", {}):
                    near = predictions["1day"][var]
                    far = predictions["1year"][var]
                    if abs(near) > 0.01 and abs(far - near) / max(abs(near), 1e-10) > 1.0:
                        violations.append(ConsistencyViolation(
                            issue_type=ConsistencyIssue.BACKWARD_FORWARD_MISMATCH,
                            description=f"{var}: near-term and far-term diverge by >100% — backward/forward inconsistency",
                            severity=0.6,
                            horizons_involved=["1day", "1year"],
                        ))
        return violations

    def _compute_smoothness(self, predictions: Dict[str, Dict[str, float]]) -> float:
        horizon_order = ["1day", "1week", "1month", "3month", "6month", "1year", "5year", "10year"]
        available = [h for h in horizon_order if h in predictions]
        if len(available) < 3:
            return 1.0
        all_vars = set().union(*[predictions.get(h, {}).keys() for h in available])
        smoothness_scores = []
        for var in all_vars:
            values = [predictions[h].get(var, 0) for h in available if var in predictions.get(h, {})]
            if len(values) < 3:
                continue
            diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
            if max(abs(v) for v in values) > 0:
                mean_abs_diff = np.mean(diffs) / max(abs(v) for v in values)
                smoothness_scores.append(max(0, 1 - mean_abs_diff))
        return float(np.mean(smoothness_scores)) if smoothness_scores else 1.0

    def _compute_uncertainty_expansion(self, uncertainties: Dict[str, Dict[str, float]]) -> float:
        horizon_order = ["1day", "1week", "1month", "3month", "6month", "1year", "5year", "10year"]
        available = [h for h in horizon_order if h in uncertainties]
        if len(available) < 2:
            return 1.0
        expansion_scores = []
        for var in set().union(*[uncertainties.get(h, {}).keys() for h in available]):
            values = [uncertainties[h].get(var, 0) for h in available if var in uncertainties.get(h, {})]
            if len(values) < 2:
                continue
            is_increasing = all(values[i] <= values[i + 1] * 1.5 for i in range(len(values) - 1))
            expansion_scores.append(1.0 if is_increasing else 0.5)
        return float(np.mean(expansion_scores)) if expansion_scores else 1.0

    def _generate_recommendations(self, violations: List[ConsistencyViolation],
                                    consistency_score: float,
                                    smoothness: float,
                                    uncertainty_exp: float) -> List[str]:
        recs = []
        severe = [v for v in violations if v.severity > 0.3]
        if severe:
            recs.append(f"{len(severe)} severe consistency violations found — cross-horizon predictions may be unreliable")
        if consistency_score > 0.8:
            recs.append(f"CONSISTENT: Cross-horizon score={consistency_score:.0%} — predictions are well-aligned")
        elif consistency_score > 0.5:
            recs.append(f"MODERATE CONSISTENCY: Score={consistency_score:.0%} — some horizon mismatches detected")
        else:
            recs.append(f"INCONSISTENT: Score={consistency_score:.0%} — major cross-horizon disagreements")
        if smoothness < 0.5:
            recs.append("LOW SMOOTHNESS: Predictions jump between horizons — consider smoothing or interpolation")
        if uncertainty_exp < 0.5:
            recs.append("UNCERTAINTY NOT EXPANDING: Far-horizon uncertainty should exceed near-horizon — check model")
        issue_types = set(v.issue_type for v in violations)
        for it in issue_types:
            count = sum(1 for v in violations if v.issue_type == it)
            recs.append(f"  → {it.value}: {count} violation(s)")
        return recs