"""Ensemble Belief Calibrator — Calibrates ensemble predictions against observed frequencies.

Combines multiple belief sources and calibrates them:
1. Binned calibration (reliability diagrams)
2. Isotonic regression calibration
3. Platt scaling
4. Temperature scaling
5. Ensemble calibration (combining calibrators)
6. Calibration trend analysis (drift detection)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CalibratorType(Enum):
    BINNED = "binned"
    ISOTONIC = "isotonic"
    PLATT = "platt"
    TEMPERATURE = "temperature"
    ENSEMBLE_CAL = "ensemble_cal"


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    predicted_prob: float
    observed_freq: float
    count: int
    deviation: float


@dataclass
class CalibrationCurve:
    bins: List[CalibrationBin]
    ece: float
    mce: float
    brier_score: float
    reliability: float


@dataclass
class CalibrationDrift:
    period: str
    ece_before: float
    ece_after: float
    drift_magnitude: float
    is_significant: bool


@dataclass
class EnsembleCalibrationResult:
    original_predictions: List[float]
    calibrated_predictions: List[float]
    calibration_curve: CalibrationCurve
    best_calibrator: CalibratorType
    calibrator_outputs: Dict[str, List[float]]
    drift_results: List[CalibrationDrift]
    confidence_adjustment: float
    recommendations: List[str]


class EnsembleBeliefCalibrator:
    def __init__(self, n_bins: int = 10, significance_level: float = 0.05):
        self.n_bins = n_bins
        self.significance_level = significance_level
        self._history: List[Tuple[float, bool]] = []

    def calibrate(self, predictions: List[float],
                   outcomes: Optional[List[bool]] = None,
                   method: CalibratorType = CalibratorType.ENSEMBLE_CAL) -> EnsembleCalibrationResult:
        if outcomes is None:
            outcomes = [p > 0.5 for p in predictions]
        for p, o in zip(predictions, outcomes):
            self._history.append((p, o))
        curve = self._compute_binned_calibration(predictions, outcomes)
        binned_cal = self._binned_calibrate(predictions, curve)
        isotonic_cal = self._isotonic_calibrate(predictions, outcomes)
        platt_cal = self._platt_calibrate(predictions, outcomes)
        temp_cal = self._temperature_calibrate(predictions, outcomes)
        calibrator_outputs = {
            CalibratorType.BINNED.value: binned_cal,
            CalibratorType.ISOTONIC.value: isotonic_cal,
            CalibratorType.PLATT.value: platt_cal,
            CalibratorType.TEMPERATURE.value: temp_cal,
        }
        if method == CalibratorType.ENSEMBLE_CAL:
            ensemble_cal = self._ensemble_calibrate(calibrator_outputs)
            calibrator_outputs[CalibratorType.ENSEMBLE_CAL.value] = ensemble_cal
            best_cal = ensemble_cal
            best_type = CalibratorType.ENSEMBLE_CAL
        else:
            best_cal = calibrator_outputs.get(method.value, binned_cal)
            best_type = method
        best_curve = self._compute_binned_calibration(best_cal, outcomes)
        drift = self._detect_drift()
        conf_adj = self._compute_confidence_adjustment(best_curve)
        recs = self._generate_recommendations(best_curve, drift, best_type)
        return EnsembleCalibrationResult(
            original_predictions=[round(p, 4) for p in predictions],
            calibrated_predictions=[round(p, 4) for p in best_cal],
            calibration_curve=best_curve,
            best_calibrator=best_type,
            calibrator_outputs={k: [round(v, 4) for v in vals] for k, vals in calibrator_outputs.items()},
            drift_results=drift,
            confidence_adjustment=round(conf_adj, 4),
            recommendations=recs,
        )

    def _compute_binned_calibration(self, predictions: List[float],
                                      outcomes: List[bool]) -> CalibrationCurve:
        preds = np.array(predictions)
        outs = np.array(outcomes, dtype=float)
        n_bins = self.n_bins
        bins = []
        bin_edges = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == 0:
                mask = (preds >= lo) & (preds <= hi)
            else:
                mask = (preds > lo) & (preds <= hi)
            count = int(mask.sum())
            if count > 0:
                pred_mean = float(preds[mask].mean())
                obs_freq = float(outs[mask].mean())
            else:
                pred_mean = (lo + hi) / 2
                obs_freq = 0.0
            deviation = abs(pred_mean - obs_freq)
            bins.append(CalibrationBin(
                lower=round(float(lo), 4),
                upper=round(float(hi), 4),
                predicted_prob=round(pred_mean, 4),
                observed_freq=round(obs_freq, 4),
                count=count,
                deviation=round(deviation, 4),
            ))
        ece = 0.0
        mce = 0.0
        total = max(len(preds), 1)
        for b in bins:
            weight = b.count / total
            ece += weight * b.deviation
            if b.deviation > mce:
                mce = b.deviation
        brier = float(np.mean((preds - outs) ** 2)) if len(preds) > 0 else 0.0
        reliability = max(0, 1 - ece * 2)
        return CalibrationCurve(
            bins=bins,
            ece=round(ece, 4),
            mce=round(mce, 4),
            brier_score=round(brier, 4),
            reliability=round(reliability, 4),
        )

    def _binned_calibrate(self, predictions: List[float],
                            curve: CalibrationCurve) -> List[float]:
        calibrated = []
        for p in predictions:
            cal_p = p
            for b in curve.bins:
                if b.lower <= p <= b.upper and b.count > 0:
                    cal_p = b.observed_freq
                    break
            calibrated.append(max(0.01, min(0.99, cal_p)))
        return calibrated

    def _isotonic_calibrate(self, predictions: List[float],
                              outcomes: List[bool]) -> List[float]:
        if len(predictions) == 0:
            return list(predictions)
        pairs = sorted(zip(predictions, outcomes), key=lambda x: x[0])
        sorted_preds = [p for p, _ in pairs]
        sorted_outs = [float(o) for _, o in pairs]
        n = len(sorted_preds)
        values = np.array(sorted_outs, dtype=float)
        counts = np.ones(n)
        active = list(range(n))
        changed = True
        while changed:
            changed = False
            new_active = []
            i = 0
            while i < len(active):
                if i + 1 < len(active) and values[active[i]] > values[active[i + 1]]:
                    total_count = counts[active[i]] + counts[active[i + 1]]
                    values[active[i]] = (values[active[i]] * counts[active[i]] + values[active[i + 1]] * counts[active[i + 1]]) / total_count
                    counts[active[i]] = total_count
                    counts[active[i + 1]] = 0
                    new_active.append(active[i])
                    i += 2
                    changed = True
                else:
                    new_active.append(active[i])
                    i += 1
            active = new_active
        boundaries = []
        idx = 0
        for ai in active:
            block_end = idx + counts[ai] - 1
            boundaries.append((idx, block_end, values[ai]))
            idx += counts[ai]
        val_map = {}
        for pidx in range(n):
            cal_val = 0.5
            for start, end, val in boundaries:
                if start <= pidx <= end:
                    cal_val = val
                    break
            val_map[sorted_preds[pidx]] = cal_val
        calibrated = []
        for p in predictions:
            cal = val_map.get(p, p)
            calibrated.append(max(0.01, min(0.99, cal)))
        return calibrated

    def _platt_calibrate(self, predictions: List[float],
                           outcomes: List[bool]) -> List[float]:
        preds = np.array(predictions)
        outs = np.array(outcomes, dtype=float)
        a, b = 0.0, 0.0
        lr = 0.1
        for _ in range(100):
            z = a * preds + b
            cal = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
            error = cal - outs
            da = float(np.mean(error * preds))
            db = float(np.mean(error))
            a -= lr * da
            b -= lr * db
            if abs(da) < 1e-6 and abs(db) < 1e-6:
                break
        z = a * preds + b
        calibrated = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        return [max(0.01, min(0.99, float(c))) for c in calibrated]

    def _temperature_calibrate(self, predictions: List[float],
                                 outcomes: List[bool]) -> List[float]:
        preds = np.array(predictions)
        outs = np.array(outcomes, dtype=float)
        temp = 1.0
        lr = 0.1
        for _ in range(200):
            z = np.log(np.clip(preds, 1e-10, 1 - 1e-10) / np.clip(1 - preds, 1e-10, 1 - 1e-10))
            z_scaled = z / max(temp, 0.01)
            cal = 1.0 / (1.0 + np.exp(-np.clip(z_scaled, -20, 20)))
            nll = -float(np.mean(outs * np.log(np.clip(cal, 1e-10, 1)) + (1 - outs) * np.log(np.clip(1 - cal, 1e-10, 1))))
            grad = 0.0
            for i in range(len(preds)):
                grad += (cal[i] - outs[i]) * z[i] / (max(temp, 0.01) ** 2)
            grad /= max(len(preds), 1)
            temp_new = temp - lr * grad
            z_new = z / max(temp_new, 0.01)
            cal_new = 1.0 / (1.0 + np.exp(-np.clip(z_new, -20, 20)))
            nll_new = -float(np.mean(outs * np.log(np.clip(cal_new, 1e-10, 1)) + (1 - outs) * np.log(np.clip(1 - cal_new, 1e-10, 1))))
            if nll_new < nll:
                temp = max(0.1, min(5.0, temp_new))
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break
        z = np.log(np.clip(preds, 1e-10, 1 - 1e-10) / np.clip(1 - preds, 1e-10, 1 - 1e-10))
        z_scaled = z / max(temp, 0.01)
        calibrated = 1.0 / (1.0 + np.exp(-np.clip(z_scaled, -20, 20)))
        return [max(0.01, min(0.99, float(c))) for c in calibrated]

    def _ensemble_calibrate(self, calibrator_outputs: Dict[str, List[float]]) -> List[float]:
        methods = list(calibrator_outputs.values())
        n = len(methods[0]) if methods else 0
        if n == 0:
            return []
        result = []
        for i in range(n):
            vals = [m[i] for m in methods if i < len(m)]
            result.append(sum(vals) / max(len(vals), 1))
        return result

    def _detect_drift(self) -> List[CalibrationDrift]:
        if len(self._history) < 20:
            return []
        mid = len(self._history) // 2
        first_half = self._history[:mid]
        second_half = self._history[mid:]
        preds1 = [p for p, _ in first_half]
        outs1 = [o for _, o in first_half]
        preds2 = [p for p, _ in second_half]
        outs2 = [o for _, o in second_half]
        ece1 = self._quick_ece(preds1, outs1)
        ece2 = self._quick_ece(preds2, outs2)
        drift_mag = abs(ece2 - ece1)
        is_sig = drift_mag > 0.05
        return [CalibrationDrift(
            period=f"first_{mid}_vs_last_{len(self._history)-mid}",
            ece_before=round(ece1, 4),
            ece_after=round(ece2, 4),
            drift_magnitude=round(drift_mag, 4),
            is_significant=is_sig,
        )]

    def _quick_ece(self, predictions: List[float], outcomes: List[bool]) -> float:
        if not predictions:
            return 0.0
        preds = np.array(predictions)
        outs = np.array(outcomes, dtype=float)
        ece = 0.0
        total = len(preds)
        bin_edges = np.linspace(0, 1, 6)
        for i in range(5):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (preds >= lo) & (preds <= hi) if i == 0 else (preds > lo) & (preds <= hi)
            count = int(mask.sum())
            if count > 0:
                ece += (count / total) * abs(float(preds[mask].mean()) - float(outs[mask].mean()))
        return ece

    def _compute_confidence_adjustment(self, curve: CalibrationCurve) -> float:
        if curve.ece < 0.05:
            return 1.0
        elif curve.ece < 0.10:
            return 0.8
        elif curve.ece < 0.20:
            return 0.6
        else:
            return 0.4

    def _generate_recommendations(self, curve: CalibrationCurve,
                                    drift: List[CalibrationDrift],
                                    best_type: CalibratorType) -> List[str]:
        recs = []
        if curve.ece < 0.05:
            recs.append("WELL CALIBRATED: ECE < 0.05 — predictions are reliable")
        elif curve.ece < 0.10:
            recs.append(f"MODERATE CALIBRATION: ECE={curve.ece:.3f} — consider calibrating before high-stakes decisions")
        else:
            recs.append(f"POOR CALIBRATION: ECE={curve.ece:.3f} — calibrate urgently, predictions over/underconfident")
        if curve.mce > 0.3:
            recs.append(f"HIGH MCE ({curve.mce:.3f}): worst bin has significant miscalibration")
        recs.append(f"Best calibrator: {best_type.value} (Brier={curve.brier_score:.4f}, reliability={curve.reliability:.2f})")
        if drift:
            for d in drift:
                if d.is_significant:
                    recs.append(f"DRIFT DETECTED: ECE changed from {d.ece_before:.3f} to {d.ece_after:.3f} — recalibrate")
        return recs