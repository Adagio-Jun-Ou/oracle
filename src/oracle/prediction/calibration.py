"""Confidence Calibration System for Oracle predictions.

Tracks prediction accuracy over time, calibrates confidence intervals
based on historical performance, and auto-adjusts CI widths per domain.

Key insight: Most prediction models are overconfident (CIs too narrow).
This system measures calibration error and applies corrections.

Implements:
- CalibrationTracker: Records forecast vs actual outcomes
- CalibrationCurve: Reliability diagram, sharpness analysis, recalibration
- DomainCalibration: Per-domain calibration with Brier scores
- PredictionAuditor: Version, lineage, hash, and drift detection for predictions
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import date, datetime
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ForecastAudit:
    prediction_id: str
    prediction_hash: str
    domain: str
    variable: str
    predicted_value: float
    ci_lower: float
    ci_upper: float
    confidence_level: str
    model_name: str
    methodology: str
    timestamp: str
    target_date: str
    state_hash: str
    lineage: List[str] = field(default_factory=list)
    version: str = "1.0"

    def verify_hash(self) -> bool:
        payload = f"{self.domain}:{self.variable}:{self.predicted_value}:{self.ci_lower}:{self.ci_upper}:{self.model_name}:{self.state_hash}"
        expected = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return expected == self.prediction_hash


@dataclass
class CalibrationRecord:
    domain: str
    variable: str
    predicted_value: float
    ci_lower: float
    ci_upper: float
    actual_value: float
    timestamp: datetime
    target_date: date
    model_name: str
    hit: bool
    error: float
    relative_error: float
    ci_width: float


@dataclass
class CalibrationCurveResult:
    expected_proportions: List[float]
    observed_proportions: List[float]
    bin_centers: List[float]
    calibration_error: float
    sharpness: float
    resolution: float
    reliability: float
    n_bins: int
    n_observations: int
    method: str = "calibration_curve"


@dataclass
class DomainCalibrationResult:
    domain: str
    brier_score: float
    log_score: float
    mean_absolute_error: float
    mean_squared_error: float
    coverage_50: float
    coverage_80: float
    coverage_95: float
    mean_ci_width: float
    overconfidence_ratio: float
    n_forecasts: int
    method: str = "domain_calibration"


class CalibrationTracker:
    """Records forecast/outcome pairs and computes calibration metrics."""

    def __init__(self):
        self.records: List[CalibrationRecord] = []
        self._domain_records: Dict[str, List[CalibrationRecord]] = {}

    def record(self, domain: str, variable: str, predicted_value: float,
               ci_lower: float, ci_upper: float, actual_value: float,
               model_name: str = "unknown",
               timestamp: Optional[datetime] = None,
               target_date: Optional[date] = None) -> CalibrationRecord:
        hit = ci_lower <= actual_value <= ci_upper
        error = actual_value - predicted_value
        ci_width = ci_upper - ci_lower
        relative_error = abs(error) / max(abs(predicted_value), 1e-10)
        record = CalibrationRecord(
            domain=domain, variable=variable, predicted_value=predicted_value,
            ci_lower=ci_lower, ci_upper=ci_upper, actual_value=actual_value,
            timestamp=timestamp or datetime.now(), target_date=target_date or date.today(),
            model_name=model_name, hit=hit, error=error,
            relative_error=relative_error, ci_width=ci_width,
        )
        self.records.append(record)
        if domain not in self._domain_records:
            self._domain_records[domain] = []
        self._domain_records[domain].append(record)
        return record

    def calibration_curve(self, domain: Optional[str] = None,
                           n_bins: int = 10) -> CalibrationCurveResult:
        records = self._domain_records.get(domain, self.records) if domain else self.records
        if len(records) < n_bins:
            n_bins = max(1, len(records))
        hits = np.array([r.hit for r in records])
        ci_widths = np.array([r.ci_width for r in records])
        values = np.array([r.predicted_value for r in records])
        observed = np.array([1.0 if r.hit else 0.0 for r in records])
        sorted_indices = np.argsort(values)
        sorted_obs = observed[sorted_indices]
        sorted_widths = ci_widths[sorted_indices]
        bins = np.array_split(sorted_obs, n_bins)
        bin_centers = []
        observed_proportions = []
        expected_proportions = []
        for i, bin_data in enumerate(bins):
            if len(bin_data) == 0:
                continue
            observed_proportions.append(float(np.mean(bin_data)))
            expected_proportions.append((i + 0.5) / n_bins)
            bin_centers.append((i + 0.5) / n_bins)
        if not observed_proportions:
            return CalibrationCurveResult(
                expected_proportions=[], observed_proportions=[], bin_centers=[],
                calibration_error=1.0, sharpness=0.0, resolution=0.0,
                reliability=1.0, n_bins=0, n_observations=len(records))
        observed_arr = np.array(observed_proportions)
        expected_arr = np.array(expected_proportions)
        calibration_error = float(np.mean((observed_arr - expected_arr) ** 2))
        mean_width = float(np.mean(sorted_widths))
        mean_obs = float(np.mean(observed_arr))
        resolution = float(np.var(observed_arr))
        reliability = calibration_error
        return CalibrationCurveResult(
            expected_proportions=expected_proportions,
            observed_proportions=observed_proportions,
            bin_centers=bin_centers,
            calibration_error=round(calibration_error, 6),
            sharpness=round(mean_width, 6),
            resolution=round(resolution, 6),
            reliability=round(reliability, 6),
            n_bins=len(observed_proportions),
            n_observations=len(records),
        )

    def domain_calibration(self, domain: str) -> DomainCalibrationResult:
        records = self._domain_records.get(domain, [])
        if not records:
            return DomainCalibrationResult(
                domain=domain, brier_score=1.0, log_score=0.0,
                mean_absolute_error=0.0, mean_squared_error=0.0,
                coverage_50=0.0, coverage_80=0.0, coverage_95=0.0,
                mean_ci_width=0.0, overconfidence_ratio=1.0, n_forecasts=0)
        errors = np.array([r.error for r in records])
        relative_errors = np.array([r.relative_error for r in records])
        hits = np.array([r.hit for r in records])
        ci_widths = np.array([r.ci_width for r in records])
        predicted = np.array([r.predicted_value for r in records])
        actual = np.array([r.actual_value for r in records])
        mse = float(np.mean(errors ** 2))
        mae = float(np.mean(np.abs(errors)))
        brier = float(np.mean((relative_errors) ** 2))
        log_vals = -np.log(np.maximum(np.clip(relative_errors, 1e-10, 1.0), 1e-10))
        log_score = float(np.mean(log_vals))
        ci_50 = np.array([r.ci_lower + 0.25 * (r.ci_upper - r.ci_lower) <= r.actual_value <=
                           r.ci_upper - 0.25 * (r.ci_upper - r.ci_lower) for r in records])
        coverage_50 = float(np.mean([1.0 if abs(r.error) <= 0.674 * r.ci_width / 2 else 0.0
                                      for r in records]))
        coverage_80 = float(np.mean([1.0 if abs(r.error) <= 1.282 * r.ci_width / 2 else 0.0
                                      for r in records]))
        coverage_95 = float(np.mean([1.0 if abs(r.error) <= 1.96 * r.ci_width / 2 else 0.0
                                      for r in records]))
        ideal_width_95 = float(2 * 1.96 * np.std(errors))
        overconfidence = ideal_width_95 / max(float(np.mean(ci_widths)), 1e-10)
        return DomainCalibrationResult(
            domain=domain, brier_score=round(brier, 6), log_score=round(log_score, 6),
            mean_absolute_error=round(mae, 6), mean_squared_error=round(mse, 6),
            coverage_50=round(coverage_50, 4), coverage_80=round(coverage_80, 4),
            coverage_95=round(coverage_95, 4), mean_ci_width=round(float(np.mean(ci_widths)), 6),
            overconfidence_ratio=round(overconfidence, 4), n_forecasts=len(records))

    def calibrate_ci(self, domain: str, ci_lower: float, ci_upper: float,
                      confidence_level: str = "HIGH") -> Tuple[float, float]:
        records = self._domain_records.get(domain, [])
        if len(records) < 10:
            return (ci_lower, ci_upper)
        errors = np.array([r.error for r in records])
        ci_widths = np.array([r.ci_width for r in records])
        observed_coverage = np.mean([1.0 if r.hit else 0.0 for r in records])
        if confidence_level == "HIGH":
            target_coverage = 0.95
        elif confidence_level == "MEDIUM":
            target_coverage = 0.80
        else:
            target_coverage = 0.50
        if observed_coverage > 0.05 and observed_coverage < 0.99:
            correction_factor = target_coverage / max(observed_coverage, 0.01)
            correction_factor = max(0.5, min(correction_factor, 3.0))
        else:
            correction_factor = 1.0
        center = (ci_lower + ci_upper) / 2
        half_width = (ci_upper - ci_lower) / 2 * correction_factor
        return (center - half_width, center + half_width)

    def summary(self) -> Dict:
        total = len(self.records)
        if total == 0:
            return {"total_forecasts": 0, "domains": [], "overall_coverage": 0.0}
        overall_coverage = sum(1 for r in self.records if r.hit) / total
        domain_summaries = {}
        for domain, records in self._domain_records.items():
            domain_hit = sum(1 for r in records if r.hit) / len(records)
            mae = float(np.mean([abs(r.error) for r in records]))
            domain_summaries[domain] = {
                "n_forecasts": len(records), "coverage": round(domain_hit, 4),
                "mae": round(mae, 6),
            }
        return {
            "total_forecasts": total, "overall_coverage": round(overall_coverage, 4),
            "domains": domain_summaries,
            "mean_ci_width": round(float(np.mean([r.ci_width for r in self.records])), 6),
            "median_relative_error": round(float(np.median([r.relative_error for r in self.records])), 6),
        }


class PredictionAuditor:
    """Version, lineage, hash, and drift detection for predictions."""

    def __init__(self):
        self.audits: List[ForecastAudit] = []

    def create_audit(self, domain: str, variable: str, predicted_value: float,
                      ci_lower: float, ci_upper: float, confidence_level: str,
                      model_name: str, methodology: str = "",
                      state_hash: str = "", lineage: Optional[List[str]] = None) -> ForecastAudit:
        payload = f"{domain}:{variable}:{predicted_value}:{ci_lower}:{ci_upper}:{model_name}:{state_hash}"
        pred_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        audit = ForecastAudit(
            prediction_id=f"P-{len(self.audits)+1:06d}",
            prediction_hash=pred_hash, domain=domain, variable=variable,
            predicted_value=predicted_value, ci_lower=ci_lower, ci_upper=ci_upper,
            confidence_level=confidence_level, model_name=model_name,
            methodology=methodology,
            timestamp=datetime.now().isoformat(),
            target_date=date.today().isoformat(),
            state_hash=state_hash,
            lineage=lineage or [],
        )
        self.audits.append(audit)
        return audit

    def detect_drift(self, domain: str, variable: str,
                      window: int = 5) -> Dict:
        relevant = [a for a in self.audits
                     if a.domain == domain and a.variable == variable]
        if len(relevant) < window * 2:
            return {"drift_detected": False, "reason": "insufficient_data",
                    "n_observations": len(relevant)}
        recent = relevant[-window:]
        baseline = relevant[-2*window:-window]
        recent_mean = np.mean([a.predicted_value for a in recent])
        baseline_mean = np.mean([a.predicted_value for a in baseline])
        recent_std = np.std([a.predicted_value for a in recent]) if len(recent) > 1 else 1.0
        drift_magnitude = abs(recent_mean - baseline_mean) / max(recent_std, 1e-10)
        ci_widening = np.mean([a.ci_upper - a.ci_lower for a in recent]) / max(
            np.mean([a.ci_upper - a.ci_lower for a in baseline]), 1e-10)
        return {
            "drift_detected": drift_magnitude > 2.0,
            "drift_magnitude": round(drift_magnitude, 4),
            "recent_mean": round(recent_mean, 6),
            "baseline_mean": round(baseline_mean, 6),
            "ci_widening_ratio": round(ci_widening, 4),
            "drift_direction": "up" if recent_mean > baseline_mean else "down",
            "n_observations": len(relevant),
        }

    def lineage_trace(self, prediction_id: str) -> Dict:
        for audit in self.audits:
            if audit.prediction_id == prediction_id:
                return {
                    "prediction_id": audit.prediction_id,
                    "prediction_hash": audit.prediction_hash,
                    "domain": audit.domain,
                    "variable": audit.variable,
                    "predicted_value": audit.predicted_value,
                    "model_name": audit.model_name,
                    "methodology": audit.methodology,
                    "timestamp": audit.timestamp,
                    "state_hash": audit.state_hash,
                    "lineage": audit.lineage,
                    "version": audit.version,
                }
        return {"error": f"Prediction {prediction_id} not found"}

    def verify_integrity(self) -> Dict:
        verified = sum(1 for a in self.audits if a.verify_hash())
        return {
            "total_audits": len(self.audits),
            "verified_hashes": verified,
            "hash_integrity_rate": verified / max(len(self.audits), 1),
        }