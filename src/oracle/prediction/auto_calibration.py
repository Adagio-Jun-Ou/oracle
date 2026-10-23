"""Auto-Calibration Engine — Continuous self-correction from prediction tracking.

The Oracle's most powerful capability: learning from its own mistakes. Every prediction
is tracked, every outcome recorded. Over time, the calibration curves reveal systematic
biases (overconfidence, underconfidence, domain-specific drift) and auto-correct them.

Based on:
- Tetlock (2015): Superforecasting — calibration is the #1 indicator of forecaster quality
- Kahneman & Tversky (1979): Prospect theory — systematic biases in probability judgment
- Armstrong (2001): Principles of Forecasting — combining + calibration beats raw models

Key features:
1. Per-domain calibration curves (predicted vs observed frequency)
2. Brier score decomposition (reliability, resolution, uncertainty)
3. Overconfidence/underconfidence detection
4. Domain-specific bias correction
5. Time-varying calibration (drift detection)
6. Auto-correction factors applied to raw predictions
7. Forecasting skill score (vs naive baseline)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import date, timedelta
import logging
import math

logger = logging.getLogger(__name__)


class CalibrationBias(Enum):
    OVERCONFIDENT = "overconfident"
    UNDERCONFIDENT = "underconfident"
    WELL_CALIBRATED = "well_calibrated"
    EXTREME_BIAS = "extreme_bias"
    CENTRAL_BIAS = "central_bias"


class DriftDirection(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


@dataclass
class CalibrationBin:
    predicted_range: Tuple[float, float]
    n_predictions: int
    observed_frequency: float
    predicted_mean: float
    calibration_error: float
    overconfidence_ratio: float


@dataclass
class DomainCalibration:
    domain: str
    n_predictions: int
    n_resolved: int
    brier_score: float
    reliability: float
    resolution: float
    uncertainty: float
    skill_score: float
    bias: CalibrationBias
    bias_magnitude: float
    correction_factor: float
    recent_mae: float
    drift: DriftDirection
    calibration_curve: List[CalibrationBin]


@dataclass
class AutoCalibrationReport:
    overall_brier: float
    overall_skill: float
    overall_bias: CalibrationBias
    domain_calibrations: Dict[str, DomainCalibration]
    global_correction: Dict[str, float]
    drift_trend: DriftDirection
    improvement_rate: float
    recommendations: List[str]
    narrative: str


class AutoCalibrationEngine:
    def __init__(self, window_days: int = 90):
        self.window_days = window_days
        self._records: List[Dict] = []

    def record(self, domain: str, predicted: float, actual: Optional[float] = None,
               confidence: float = 0.8, horizon_days: int = 30, method: str = "",
               date_str: Optional[str] = None) -> str:
        pid = f"cal_{domain}_{len(self._records):06d}"
        self._records.append({
            "id": pid, "domain": domain, "predicted": predicted,
            "actual": actual, "confidence": confidence,
            "horizon_days": horizon_days, "method": method,
            "date": date_str or date.today().isoformat(),
        })
        return pid

    def resolve(self, pid: str, actual: float) -> None:
        for r in self._records:
            if r["id"] == pid:
                r["actual"] = actual
                return

    def calibrate(self, domain: str) -> DomainCalibration:
        domain_records = [r for r in self._records if r["domain"] == domain]
        resolved = [r for r in domain_records if r.get("actual") is not None]
        n_total = len(domain_records)
        n_resolved = len(resolved)
        if n_resolved < 3:
            return DomainCalibration(
                domain=domain, n_predictions=n_total, n_resolved=n_resolved,
                brier_score=0.25, reliability=0.0, resolution=0.0,
                uncertainty=0.25, skill_score=0.0,
                bias=CalibrationBias.WELL_CALIBRATED, bias_magnitude=0.0,
                correction_factor=1.0, recent_mae=0.0,
                drift=DriftDirection.STABLE, calibration_curve=[],
            )
        preds = np.array([r["predicted"] for r in resolved])
        actuals = np.array([r["actual"] for r in resolved])
        brier = float(np.mean((preds - actuals) ** 2))
        n_bins = min(10, max(3, n_resolved // 5))
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins = []
        for i in range(n_bins):
            mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
            n_in_bin = int(np.sum(mask))
            if n_in_bin == 0:
                continue
            obs_freq = float(np.mean(actuals[mask]))
            pred_mean = float(np.mean(preds[mask]))
            cal_error = abs(obs_freq - pred_mean)
            over_ratio = obs_freq / max(pred_mean, 0.01)
            bins.append(CalibrationBin(
                predicted_range=(round(float(bin_edges[i]), 2), round(float(bin_edges[i + 1]), 2)),
                n_predictions=n_in_bin,
                observed_frequency=round(obs_freq, 4),
                predicted_mean=round(pred_mean, 4),
                calibration_error=round(cal_error, 4),
                overconfidence_ratio=round(float(over_ratio), 4),
            ))
        obs_mean = float(np.mean(actuals))
        obs_var = float(np.var(actuals))
        expected = preds
        reliability = float(np.mean((expected - actuals) ** 2))
        resolution = float(np.var([b.observed_frequency for b in bins])) if len(bins) > 1 else 0.0
        uncertainty = float(obs_var)
        skill = 1 - brier / max(uncertainty, 1e-10) if uncertainty > 1e-10 else 0.0
        skill = max(0.0, min(1.0, skill))
        bias, bias_mag = self._detect_bias(preds, actuals)
        correction = self._compute_correction(preds, actuals, bias, bias_mag)
        drift = self._detect_drift(resolved)
        recent = resolved[-min(10, len(resolved)):]
        recent_mae = float(np.mean([abs(r["predicted"] - r["actual"]) for r in recent]))
        return DomainCalibration(
            domain=domain, n_predictions=n_total, n_resolved=n_resolved,
            brier_score=round(brier, 4),
            reliability=round(reliability, 4),
            resolution=round(resolution, 4),
            uncertainty=round(uncertainty, 4),
            skill_score=round(skill, 4),
            bias=bias, bias_magnitude=round(bias_mag, 4),
            correction_factor=round(correction, 4),
            recent_mae=round(recent_mae, 4),
            drift=drift, calibration_curve=bins,
        )

    def calibrate_all(self) -> AutoCalibrationReport:
        domains = set(r["domain"] for r in self._records)
        domain_cals = {}
        for domain in domains:
            domain_cals[domain] = self.calibrate(domain)
        resolved = [r for r in self._records if r.get("actual") is not None]
        if not resolved:
            return AutoCalibrationReport(
                overall_brier=0.25, overall_skill=0.0,
                overall_bias=CalibrationBias.WELL_CALIBRATED,
                domain_calibrations=domain_cals, global_correction={},
                drift_trend=DriftDirection.STABLE, improvement_rate=0.0,
                recommendations=["Make predictions and track outcomes to calibrate"],
                narrative="No resolved predictions yet. Start making forecasts.",
            )
        all_preds = np.array([r["predicted"] for r in resolved])
        all_actuals = np.array([r["actual"] for r in resolved])
        overall_brier = float(np.mean((all_preds - all_actuals) ** 2))
        overall_skill = 1 - overall_brier / max(float(np.var(all_actuals)), 1e-10)
        overall_skill = max(0.0, min(1.0, overall_skill))
        overall_bias, overall_mag = self._detect_bias(all_preds, all_actuals)
        global_correction = {}
        for domain, cal in domain_cals.items():
            global_correction[domain] = cal.correction_factor
        drift_counts = {"improving": 0, "stable": 0, "degrading": 0}
        for cal in domain_cals.values():
            drift_counts[cal.drift.value] += 1
        overall_drift = DriftDirection.STABLE
        if drift_counts["improving"] > drift_counts["degrading"]:
            overall_drift = DriftDirection.IMPROVING
        elif drift_counts["degrading"] > drift_counts["improving"]:
            overall_drift = DriftDirection.DEGRADING
        n_total = len(resolved)
        if n_total >= 20:
            first_half = resolved[:n_total // 2]
            second_half = resolved[n_total // 2:]
            e1 = np.mean([abs(r["predicted"] - r["actual"]) for r in first_half])
            e2 = np.mean([abs(r["predicted"] - r["actual"]) for r in second_half])
            improvement = float((e1 - e2) / max(e1, 1e-10))
        else:
            improvement = 0.0
        recs = self._generate_recommendations(domain_cals, overall_brier, overall_skill)
        narrative = self._generate_narrative(domain_cals, overall_brier, overall_skill, overall_bias)
        return AutoCalibrationReport(
            overall_brier=round(overall_brier, 4),
            overall_skill=round(overall_skill, 4),
            overall_bias=overall_bias,
            domain_calibrations=domain_cals,
            global_correction=global_correction,
            drift_trend=overall_drift,
            improvement_rate=round(improvement, 4),
            recommendations=recs,
            narrative=narrative,
        )

    def apply_correction(self, domain: str, predicted: float) -> float:
        cal = self.calibrate(domain)
        corrected = predicted * cal.correction_factor
        if cal.bias == CalibrationBias.EXTREME_BIAS:
            corrected = 0.5 + (predicted - 0.5) * cal.correction_factor
        elif cal.bias == CalibrationBias.CENTRAL_BIAS:
            corrected = 0.5 + (predicted - 0.5) / max(cal.correction_factor, 0.1)
        return max(0.01, min(0.99, corrected))

    def _detect_bias(self, preds: np.ndarray, actuals: np.ndarray) -> Tuple[CalibrationBias, float]:
        if len(preds) < 3:
            return CalibrationBias.WELL_CALIBRATED, 0.0
        errors = preds - actuals
        mean_error = float(np.mean(errors))
        high_pred = preds > 0.7
        low_pred = preds < 0.3
        central_pred = (preds >= 0.3) & (preds <= 0.7)
        extreme_avg = float(np.mean(errors[high_pred | low_pred])) if np.any(high_pred | low_pred) else 0
        central_avg = float(np.mean(errors[central_pred])) if np.any(central_pred) else 0
        if abs(mean_error) < 0.05 and abs(extreme_avg) < 0.1:
            return CalibrationBias.WELL_CALIBRATED, float(abs(mean_error))
        if abs(extreme_avg) > 0.15 and abs(extreme_avg) > abs(central_avg):
            bias = CalibrationBias.OVERCONFIDENT if extreme_avg > 0 else CalibrationBias.EXTREME_BIAS
            return bias, float(abs(extreme_avg))
        if abs(central_avg) > 0.1 and abs(central_avg) > abs(extreme_avg):
            return CalibrationBias.CENTRAL_BIAS, float(abs(central_avg))
        if mean_error > 0.05:
            return CalibrationBias.OVERCONFIDENT, float(abs(mean_error))
        if mean_error < -0.05:
            return CalibrationBias.UNDERCONFIDENT, float(abs(mean_error))
        return CalibrationBias.WELL_CALIBRATED, float(abs(mean_error))

    def _compute_correction(self, preds: np.ndarray, actuals: np.ndarray,
                           bias: CalibrationBias, magnitude: float) -> float:
        if len(preds) < 5:
            return 1.0
        if bias == CalibrationBias.OVERCONFIDENT:
            return 1.0 / (1.0 + magnitude)
        elif bias == CalibrationBias.UNDERCONFIDENT:
            return 1.0 + magnitude * 0.5
        elif bias == CalibrationBias.EXTREME_BIAS:
            return max(0.3, 1.0 - magnitude)
        elif bias == CalibrationBias.CENTRAL_BIAS:
            return min(2.0, 1.0 + magnitude)
        return 1.0

    def _detect_drift(self, resolved: List[Dict]) -> DriftDirection:
        if len(resolved) < 10:
            return DriftDirection.STABLE
        n = len(resolved)
        half = n // 2
        first_mae = np.mean([abs(r["predicted"] - r["actual"]) for r in resolved[:half]])
        second_mae = np.mean([abs(r["predicted"] - r["actual"]) for r in resolved[half:]])
        if second_mae < first_mae * 0.85:
            return DriftDirection.IMPROVING
        elif second_mae > first_mae * 1.15:
            return DriftDirection.DEGRADING
        return DriftDirection.STABLE

    def _generate_recommendations(self, cals: Dict[str, DomainCalibration],
                                  brier: float, skill: float) -> List[str]:
        recs = []
        if brier > 0.3:
            recs.append(f"Overall Brier score {brier:.3f} is above 0.3 — significant prediction improvement needed")
        if skill < 0.1:
            recs.append("Skill score near zero — predictions barely better than chance. Review models.")
        overconf = [d for d, c in cals.items() if c.bias in (CalibrationBias.OVERCONFIDENT, CalibrationBias.EXTREME_BIAS)]
        if overconf:
            recs.append(f"Overconfident domains: {', '.join(overconf)} — widen confidence intervals by 20-50%")
        underconf = [d for d, c in cals.items() if c.bias == CalibrationBias.UNDERCONFIDENT]
        if underconf:
            recs.append(f"Underconfident domains: {', '.join(underconf)} — predictions are too conservative; trust signals more")
        degrading = [d for d, c in cals.items() if c.drift == DriftDirection.DEGRADING]
        if degrading:
            recs.append(f"Degrading domains: {', '.join(degrading)} — models losing accuracy; recalibrate immediately")
        improving = [d for d, c in cals.items() if c.drift == DriftDirection.IMPROVING]
        if improving:
            recs.append(f"Improving domains: {', '.join(improving)} — recent accuracy is improving")
        return recs[:7]

    def _generate_narrative(self, cals: Dict[str, DomainCalibration],
                            brier: float, skill: float, bias: CalibrationBias) -> str:
        n_resolved = sum(c.n_resolved for c in cals.values())
        parts = [f"Oracle calibration report: {n_resolved} predictions resolved."]
        parts.append(f"Overall Brier: {brier:.3f}, Skill: {skill:.2%}, Bias: {bias.value}.")
        best = min(cals.items(), key=lambda x: x[1].brier_score) if cals else None
        worst = max(cals.items(), key=lambda x: x[1].brier_score) if cals else None
        if best:
            parts.append(f"Best domain: {best[0]} (Brier {best[1].brier_score:.3f}).")
        if worst:
            parts.append(f"Worst domain: {worst[0]} (Brier {worst[1].brier_score:.3f}).")
        return " ".join(parts)