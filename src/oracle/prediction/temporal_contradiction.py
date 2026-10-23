import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TurningPointType(Enum):
    ACCELERATION = "acceleration"
    DECELERATION = "deceleration"
    REVERSAL = "reversal"
    OVERSHOOT = "overshoot"
    CORRECTION = "correction"


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


@dataclass
class TurningPointReport:
    variable: str
    short_term_direction: str
    long_term_direction: str
    divergence_score: float
    turning_point_type: str
    probability: float
    estimated_timing_months: float
    confidence: float

    def to_dict(self) -> Dict:
        return {
            "variable": self.variable,
            "short_term_direction": self.short_term_direction,
            "long_term_direction": self.long_term_direction,
            "divergence_score": self.divergence_score,
            "turning_point_type": self.turning_point_type,
            "probability": round(self.probability, 4),
            "estimated_timing_months": round(self.estimated_timing_months, 2),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class TemporalContradictionResult:
    turning_points: List[TurningPointReport]
    n_divergences: int
    max_divergence: float
    mean_divergence: float
    system_divergence_score: float
    most_critical_variable: str
    method: str = "temporal_contradiction"

    def to_dict(self) -> Dict:
        return {
            "turning_points": [tp.to_dict() for tp in self.turning_points],
            "n_divergences": self.n_divergences,
            "max_divergence": round(self.max_divergence, 4),
            "mean_divergence": round(self.mean_divergence, 4),
            "system_divergence_score": round(self.system_divergence_score, 4),
            "most_critical_variable": self.most_critical_variable,
            "method": self.method,
        }


class TemporalContradictionDetector:
    def __init__(self, short_term_horizon: int = 3, long_term_horizon: int = 24,
                 divergence_threshold: float = 0.3, min_historical_length: int = 30):
        self.short_term_horizon = short_term_horizon
        self.long_term_horizon = long_term_horizon
        self.divergence_threshold = divergence_threshold
        self.min_historical_length = min_historical_length

    @staticmethod
    def _direction(value: float, baseline: float = 0.0) -> Direction:
        if value > baseline + 1e-8:
            return Direction.UP
        elif value < baseline - 1e-8:
            return Direction.DOWN
        return Direction.FLAT

    def divergence_score(self, variable: str, short_val: float, long_val: float) -> float:
        short_dir = self._direction(short_val)
        long_dir = self._direction(long_val)

        if short_dir == long_dir:
            magnitude = abs(short_val - long_val)
            max_val = max(abs(short_val), abs(long_val), 1e-10)
            relative_gap = magnitude / max_val
            return min(relative_gap * 0.3, 0.3)

        direction_penalty = 0.5 if (short_dir == Direction.FLAT or long_dir == Direction.FLAT) else 0.0

        short_mag = abs(short_val)
        long_mag = abs(long_val)
        max_mag = max(short_mag, long_mag, 1e-10)
        magnitude_factor = (short_mag + long_mag) / (2.0 * max_mag)

        opposing = 1.0 if (short_dir == Direction.UP and long_dir == Direction.DOWN) or \
                           (short_dir == Direction.DOWN and long_dir == Direction.UP) else 0.6

        score = 0.5 + direction_penalty * 0.1 + magnitude_factor * 0.3 + opposing * 0.2
        return float(np.clip(score, 0.0, 1.0))

    def classify_turning_point(self, divergences: List[Dict]) -> List[TurningPointType]:
        classifications = []
        for d in divergences:
            short_dir = d.get("short_term_direction", "flat")
            long_dir = d.get("long_term_direction", "flat")
            score = d.get("divergence_score", 0.0)
            short_mag = abs(d.get("short_val", 0.0))
            long_mag = abs(d.get("long_val", 0.0))

            if short_dir == "up" and long_dir == "down":
                if short_mag > long_mag * 2.0:
                    classifications.append(TurningPointType.OVERSHOOT)
                else:
                    classifications.append(TurningPointType.REVERSAL)
            elif short_dir == "down" and long_dir == "up":
                if short_mag > long_mag * 2.0:
                    classifications.append(TurningPointType.OVERSHOOT)
                else:
                    classifications.append(TurningPointType.REVERSAL)
            elif short_dir == "up" and long_dir == "up":
                if short_mag > long_mag * 1.5:
                    classifications.append(TurningPointType.ACCELERATION)
                else:
                    classifications.append(TurningPointType.DECELERATION)
            elif short_dir == "down" and long_dir == "down":
                if short_mag > long_mag * 1.5:
                    classifications.append(TurningPointType.DECELERATION)
                else:
                    classifications.append(TurningPointType.ACCELERATION)
            elif short_dir == "flat" and long_dir != "flat":
                if long_dir == "up":
                    classifications.append(TurningPointType.CORRECTION)
                else:
                    classifications.append(TurningPointType.CORRECTION)
            elif long_dir == "flat" and short_dir != "flat":
                classifications.append(TurningPointType.ACCELERATION)
            else:
                classifications.append(TurningPointType.CORRECTION)

        return classifications

    def turning_point_probability(self, divergences: List[Dict],
                                   historical_data: Optional[Dict[str, np.ndarray]] = None) -> List[float]:
        probabilities = []

        for d in divergences:
            score = d.get("divergence_score", 0.0)
            base_prob = score ** 1.5

            variable = d.get("variable", "")
            if historical_data is not None and variable in historical_data:
                history = np.asarray(historical_data[variable], dtype=float).ravel()
                if len(history) >= self.min_historical_length:
                    n = len(history)
                    half = n // 2
                    windows = []
                    for start in range(0, n - self.long_term_horizon, self.short_term_horizon):
                        end_short = min(start + self.short_term_horizon, n)
                        end_long = min(start + self.long_term_horizon, n)
                        seg_short = history[start:end_short]
                        seg_long = history[start:end_long]
                        if len(seg_short) > 1 and len(seg_long) > 1:
                            slope_short = float(np.polyfit(np.arange(len(seg_short)), seg_short, 1)[0])
                            slope_long = float(np.polyfit(np.arange(len(seg_long)), seg_long, 1)[0])
                            windows.append({"short_slope": slope_short, "long_slope": slope_long})

                    if windows:
                        turning_points_observed = 0
                        for i in range(1, len(windows)):
                            prev_short_dir = self._direction(windows[i - 1]["short_slope"])
                            prev_long_dir = self._direction(windows[i - 1]["long_slope"])
                            curr_short_dir = self._direction(windows[i]["short_slope"])
                            curr_long_dir = self._direction(windows[i]["long_slope"])

                            if (prev_short_dir != prev_long_dir and
                                    curr_short_dir == curr_long_dir):
                                turning_points_observed += 1
                            elif (prev_short_dir == prev_long_dir and
                                  curr_short_dir != curr_long_dir):
                                turning_points_observed += 1

                        historical_rate = turning_points_observed / max(len(windows) - 1, 1)
                        adjustment = 1.0 + 0.3 * min(historical_rate, 1.0)
                        base_prob = min(base_prob * adjustment, 1.0)

            probabilities.append(float(np.clip(base_prob, 0.0, 1.0)))

        return probabilities

    def _estimate_timing(self, divergence_score: float, short_val: float,
                          long_val: float, tp_type: TurningPointType) -> float:
        base_timings = {
            TurningPointType.REVERSAL: 6.0,
            TurningPointType.OVERSHOOT: 3.0,
            TurningPointType.ACCELERATION: 9.0,
            TurningPointType.DECELERATION: 8.0,
            TurningPointType.CORRECTION: 5.0,
        }
        base = base_timings.get(tp_type, 6.0)

        speed_factor = 1.0 - 0.5 * divergence_score
        magnitude_factor = 1.0
        max_mag = max(abs(short_val), abs(long_val), 1e-10)
        if max_mag > 1.0:
            magnitude_factor = 0.7
        elif max_mag < 0.1:
            magnitude_factor = 1.3

        return max(1.0, base * speed_factor * magnitude_factor)

    def _compute_confidence(self, divergence_score: float, probability: float,
                             historical_data: Optional[Dict[str, np.ndarray]] = None,
                             variable: str = "") -> float:
        base = divergence_score * 0.4 + probability * 0.4 + 0.2

        if historical_data is not None and variable in historical_data:
            history = np.asarray(historical_data[variable], dtype=float).ravel()
            if len(history) >= self.min_historical_length:
                data_bonus = min(len(history) / 200.0, 0.15)
                base += data_bonus

        return float(np.clip(base, 0.0, 1.0))

    def detect(self, short_term_forecasts: List[Dict], long_term_forecasts: List[Dict],
               historical_data: Optional[Dict[str, np.ndarray]] = None) -> TemporalContradictionResult:
        short_by_var = {f["variable"]: f for f in short_term_forecasts if "variable" in f}
        long_by_var = {f["variable"]: f for f in long_term_forecasts if "variable" in f}

        common_vars = set(short_by_var.keys()) & set(long_by_var.keys())
        if not common_vars:
            logger.warning("No common variables between short-term and long-term forecasts")
            return TemporalContradictionResult(
                turning_points=[], n_divergences=0, max_divergence=0.0,
                mean_divergence=0.0, system_divergence_score=0.0,
                most_critical_variable="",
            )

        raw_divergences = []
        for var in common_vars:
            sf = short_by_var[var]
            lf = long_by_var[var]
            short_val = sf.get("change_rate", sf.get("value", 0.0))
            long_val = lf.get("change_rate", lf.get("value", 0.0))
            div_score = self.divergence_score(var, short_val, long_val)

            if div_score >= self.divergence_threshold:
                raw_divergences.append({
                    "variable": var,
                    "short_val": short_val,
                    "long_val": long_val,
                    "short_term_direction": self._direction(short_val).value,
                    "long_term_direction": self._direction(long_val).value,
                    "divergence_score": div_score,
                })

        raw_divergences.sort(key=lambda x: -x["divergence_score"])

        tp_types = self.classify_turning_point(raw_divergences)
        tp_probs = self.turning_point_probability(raw_divergences, historical_data)

        reports = []
        for i, d in enumerate(raw_divergences):
            tp_type = tp_types[i] if i < len(tp_types) else TurningPointType.CORRECTION
            prob = tp_probs[i] if i < len(tp_probs) else d["divergence_score"]
            timing = self._estimate_timing(d["divergence_score"], d["short_val"], d["long_val"], tp_type)
            confidence = self._compute_confidence(d["divergence_score"], prob, historical_data, d["variable"])

            reports.append(TurningPointReport(
                variable=d["variable"],
                short_term_direction=d["short_term_direction"],
                long_term_direction=d["long_term_direction"],
                divergence_score=round(d["divergence_score"], 4),
                turning_point_type=tp_type.value,
                probability=round(prob, 4),
                estimated_timing_months=round(timing, 2),
                confidence=round(confidence, 4),
            ))

        n_div = len(reports)
        max_div = max((r.divergence_score for r in reports), default=0.0)
        mean_div = float(np.mean([r.divergence_score for r in reports])) if reports else 0.0
        system_score = float(np.clip(mean_div * np.log1p(n_div) * 0.5 + max_div * 0.5, 0.0, 1.0))
        most_critical = reports[0].variable if reports else ""

        logger.info(
            "Temporal contradiction detection: %d divergences, max=%.3f, system=%.3f",
            n_div, max_div, system_score,
        )

        return TemporalContradictionResult(
            turning_points=reports,
            n_divergences=n_div,
            max_divergence=round(max_div, 4),
            mean_divergence=round(mean_div, 4),
            system_divergence_score=round(system_score, 4),
            most_critical_variable=most_critical,
        )