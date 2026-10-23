import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EmergenceType(Enum):
    CRITICAL_SLOWING = "critical_slowing"
    VARIANCE_INCREASE = "variance_increase"
    FLICKERING = "flickering"
    CORRELATION_LENGTH = "correlation_length"
    COMBINED = "combined"


@dataclass
class SignalResult:
    signal_type: str
    value: float
    normalized: float
    threshold: float
    triggered: bool

    def to_dict(self) -> Dict:
        return {
            "signal_type": self.signal_type,
            "value": round(self.value, 6),
            "normalized": round(self.normalized, 4),
            "threshold": round(self.threshold, 4),
            "triggered": self.triggered,
        }


@dataclass
class EmergenceReport:
    variable: str
    emergence_type: str
    signal_strength: float
    distance_to_transition: float
    confidence: float
    early_warning_days_estimate: float
    signals: List[SignalResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "variable": self.variable,
            "emergence_type": self.emergence_type,
            "signal_strength": round(self.signal_strength, 4),
            "distance_to_transition": round(self.distance_to_transition, 4),
            "confidence": round(self.confidence, 4),
            "early_warning_days_estimate": round(self.early_warning_days_estimate, 1),
            "signals": [s.to_dict() for s in self.signals],
        }


@dataclass
class EmergenceDetectionResult:
    reports: List[EmergenceReport]
    n_emergence_signals: int
    max_signal_strength: float
    mean_signal_strength: float
    system_emergence_probability: float
    most_critical_variable: str
    method: str = "emergence_detector"

    def to_dict(self) -> Dict:
        return {
            "reports": [r.to_dict() for r in self.reports],
            "n_emergence_signals": self.n_emergence_signals,
            "max_signal_strength": round(self.max_signal_strength, 4),
            "mean_signal_strength": round(self.mean_signal_strength, 4),
            "system_emergence_probability": round(self.system_emergence_probability, 4),
            "most_critical_variable": self.most_critical_variable,
            "method": self.method,
        }


class EmergenceDetector:
    def __init__(self, window_size: int = 20, ar_lag: int = 1,
                 flickering_threshold: float = 1.5,
                 correlation_threshold: float = 0.6,
                 slowdown_threshold: float = 0.7,
                 variance_threshold: float = 1.5,
                 min_series_length: int = 30):
        self.window_size = window_size
        self.ar_lag = ar_lag
        self.flickering_threshold = flickering_threshold
        self.correlation_threshold = correlation_threshold
        self.slowdown_threshold = slowdown_threshold
        self.variance_threshold = variance_threshold
        self.min_series_length = min_series_length

    def detect_critical_slowing(self, series: np.ndarray) -> SignalResult:
        s = np.asarray(series, dtype=float).ravel()
        if len(s) < self.min_series_length:
            return SignalResult(
                signal_type="critical_slowing",
                value=0.0, normalized=0.0,
                threshold=self.slowdown_threshold, triggered=False,
            )

        n_windows = max(len(s) // self.window_size, 2)
        ar1_values = []

        for i in range(n_windows):
            start = i * self.window_size
            end = min(start + self.window_size, len(s))
            window = s[start:end]
            if len(window) < self.ar_lag + 2:
                continue

            x = window[:-self.ar_lag]
            y = window[self.ar_lag:]
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]

            x_centered = x - np.mean(x)
            y_centered = y - np.mean(y)
            denom = np.sum(x_centered ** 2)
            if denom < 1e-12:
                ar1_values.append(0.0)
                continue
            beta = float(np.sum(x_centered * y_centered) / denom)
            ar1_values.append(float(np.clip(beta, -1.0, 1.0)))

        if len(ar1_values) < 2:
            return SignalResult(
                signal_type="critical_slowing",
                value=ar1_values[-1] if ar1_values else 0.0,
                normalized=0.0,
                threshold=self.slowdown_threshold, triggered=False,
            )

        mean_ar1 = float(np.mean(ar1_values))
        trend = float(ar1_values[-1] - ar1_values[0])
        normalized = float(np.clip(abs(mean_ar1) + max(trend, 0.0), 0.0, 1.0))

        return SignalResult(
            signal_type="critical_slowing",
            value=mean_ar1,
            normalized=normalized,
            threshold=self.slowdown_threshold,
            triggered=normalized >= self.slowdown_threshold,
        )

    def detect_variance_increase(self, series: np.ndarray) -> SignalResult:
        s = np.asarray(series, dtype=float).ravel()
        if len(s) < self.min_series_length:
            return SignalResult(
                signal_type="variance_increase",
                value=0.0, normalized=0.0,
                threshold=self.variance_threshold, triggered=False,
            )

        n_windows = max(len(s) // self.window_size, 2)
        variances = []

        for i in range(n_windows):
            start = i * self.window_size
            end = min(start + self.window_size, len(s))
            window = s[start:end]
            if len(window) < 3:
                continue
            variances.append(float(np.var(window, ddof=1)))

        if len(variances) < 2:
            return SignalResult(
                signal_type="variance_increase",
                value=variances[0] if variances else 0.0,
                normalized=0.0,
                threshold=self.variance_threshold, triggered=False,
            )

        early_var = float(np.mean(variances[:len(variances) // 2]))
        late_var = float(np.mean(variances[len(variances) // 2:]))

        if early_var < 1e-12:
            ratio = float(np.clip(late_var / 1e-12, 0.0, 100.0))
        else:
            ratio = late_var / early_var

        normalized = float(np.clip((ratio - 1.0) / (self.variance_threshold - 1.0 + 1e-10), 0.0, 1.0))

        return SignalResult(
            signal_type="variance_increase",
            value=float(ratio),
            normalized=normalized,
            threshold=self.variance_threshold,
            triggered=ratio >= self.variance_threshold,
        )

    def detect_flickering(self, series: np.ndarray, threshold: Optional[float] = None) -> SignalResult:
        s = np.asarray(series, dtype=float).ravel()
        th = threshold if threshold is not None else self.flickering_threshold
        if len(s) < self.min_series_length:
            return SignalResult(
                signal_type="flickering",
                value=0.0, normalized=0.0,
                threshold=th, triggered=False,
            )

        mean_val = float(np.mean(s))
        std_val = float(np.std(s, ddof=1))
        if std_val < 1e-12:
            return SignalResult(
                signal_type="flickering",
                value=0.0, normalized=0.0,
                threshold=th, triggered=False,
            )

        z = (s - mean_val) / std_val
        crossings = int(np.sum(np.abs(np.diff(np.sign(z))) > 0))

        expected_crossings = len(s) * 0.5
        actual_ratio = crossings / max(expected_crossings, 1.0)

        rolling_means = []
        half_w = max(self.window_size // 2, 2)
        for i in range(half_w, len(s) - half_w):
            window = s[i - half_w:i + half_w + 1]
            rolling_means.append(float(np.mean(window)))

        if len(rolling_means) < 2:
            normalized = float(np.clip(actual_ratio / th, 0.0, 1.0))
            return SignalResult(
                signal_type="flickering",
                value=float(actual_ratio),
                normalized=normalized,
                threshold=th,
                triggered=actual_ratio >= th,
            )

        rm = np.array(rolling_means)
        above = rm > mean_val
        state_changes = int(np.sum(np.diff(above.astype(int)) != 0))
        state_change_rate = state_changes / max(len(rolling_means) - 1, 1.0)

        expected_change_rate = 0.5
        flicker_intensity = state_change_rate / max(expected_change_rate, 1e-10)
        combined = float(0.5 * actual_ratio / max(th, 1e-10) + 0.5 * flicker_intensity)
        normalized = float(np.clip(combined, 0.0, 1.0))

        return SignalResult(
            signal_type="flickering",
            value=float(actual_ratio),
            normalized=normalized,
            threshold=th,
            triggered=normalized >= 0.5,
        )

    def detect_correlation_length(self, series_matrix: np.ndarray) -> SignalResult:
        m = np.asarray(series_matrix, dtype=float)
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        if m.shape[0] < self.min_series_length or m.shape[1] < 2:
            return SignalResult(
                signal_type="correlation_length",
                value=0.0, normalized=0.0,
                threshold=self.correlation_threshold, triggered=False,
            )

        n_obs, n_vars = m.shape
        n_windows = max(n_obs // self.window_size, 2)
        mean_cors = []

        for w in range(n_windows):
            start = w * self.window_size
            end = min(start + self.window_size, n_obs)
            window = m[start:end, :]
            if window.shape[0] < 3:
                continue

            cm = np.corrcoef(window, rowvar=False)
            if cm.shape[0] < 2:
                continue

            idx = np.triu_indices(cm.shape[0], k=1)
            off_diag = cm[idx]
            off_diag = off_diag[~np.isnan(off_diag)]
            if len(off_diag) > 0:
                mean_cors.append(float(np.mean(np.abs(off_diag))))

        if len(mean_cors) < 2:
            return SignalResult(
                signal_type="correlation_length",
                value=mean_cors[0] if mean_cors else 0.0,
                normalized=0.0,
                threshold=self.correlation_threshold, triggered=False,
            )

        early_cor = float(np.mean(mean_cors[:len(mean_cors) // 2]))
        late_cor = float(np.mean(mean_cors[len(mean_cors) // 2:]))
        cor_change = late_cor - early_cor
        normalized = float(np.clip(abs(cor_change) / (1.0 - self.correlation_threshold + 1e-10), 0.0, 1.0))

        return SignalResult(
            signal_type="correlation_length",
            value=float(cor_change),
            normalized=normalized,
            threshold=self.correlation_threshold,
            triggered=late_cor >= self.correlation_threshold,
        )

    def emergence_score(self, all_signals: List[SignalResult]) -> float:
        if not all_signals:
            return 0.0

        weights = {
            "critical_slowing": 0.30,
            "variance_increase": 0.25,
            "flickering": 0.25,
            "correlation_length": 0.20,
        }

        weighted_sum = 0.0
        total_weight = 0.0
        for signal in all_signals:
            w = weights.get(signal.signal_type, 0.1)
            effective = signal.normalized if signal.triggered else signal.normalized * 0.3
            weighted_sum += w * effective
            total_weight += w

        if total_weight < 1e-10:
            return 0.0

        n_triggered = sum(1 for s in all_signals if s.triggered)
        if n_triggered >= 3:
            synergistic = 1.0 + 0.15 * (n_triggered - 2)
        elif n_triggered >= 2:
            synergistic = 1.05
        else:
            synergistic = 1.0

        raw = (weighted_sum / total_weight) * synergistic
        return float(np.clip(raw, 0.0, 1.0))

    def _estimate_distance_to_transition(self, signals: List[SignalResult]) -> float:
        n_triggered = sum(1 for s in signals if s.triggered)
        if n_triggered == 0:
            return 1.0
        if n_triggered == 1:
            return 0.7
        if n_triggered == 2:
            return 0.4
        if n_triggered == 3:
            return 0.2
        return 0.05

    def _estimate_early_warning_days(self, signals: List[SignalResult],
                                      distance: float) -> float:
        if distance > 0.8:
            return 365.0
        if distance > 0.5:
            return 180.0
        if distance > 0.3:
            return 90.0
        if distance > 0.15:
            return 30.0
        return 7.0

    def _compute_confidence(self, signals: List[SignalResult],
                             series_length: int) -> float:
        if not signals:
            return 0.0
        n_triggered = sum(1 for s in signals if s.triggered)
        signal_conf = n_triggered / max(len(signals), 1)
        strength_conf = float(np.mean([s.normalized for s in signals]))
        data_conf = min(series_length / 200.0, 0.2)
        return float(np.clip(0.4 * signal_conf + 0.4 * strength_conf + 0.2 + data_conf, 0.0, 1.0))

    def _classify_emergence_type(self, signals: List[SignalResult]) -> EmergenceType:
        n_triggered = sum(1 for s in signals if s.triggered)
        if n_triggered >= 3:
            return EmergenceType.COMBINED
        best_signal = max(signals, key=lambda s: s.normalized) if signals else None
        if best_signal is None:
            return EmergenceType.COMBINED
        type_map = {
            "critical_slowing": EmergenceType.CRITICAL_SLOWING,
            "variance_increase": EmergenceType.VARIANCE_INCREASE,
            "flickering": EmergenceType.FLICKERING,
            "correlation_length": EmergenceType.CORRELATION_LENGTH,
        }
        return type_map.get(best_signal.signal_type, EmergenceType.COMBINED)

    def detect(self, variables: Dict[str, np.ndarray],
               cross_matrix: Optional[np.ndarray] = None) -> EmergenceDetectionResult:
        reports = []

        for var_name, series in variables.items():
            s = np.asarray(series, dtype=float).ravel()
            if len(s) < self.min_series_length:
                logger.debug("Skipping variable %s: series too short (%d)", var_name, len(s))
                continue

            signals = [
                self.detect_critical_slowing(s),
                self.detect_variance_increase(s),
                self.detect_flickering(s),
            ]

            if cross_matrix is not None:
                signals.append(self.detect_correlation_length(cross_matrix))

            score = self.emergence_score(signals)
            distance = self._estimate_distance_to_transition(signals)
            early_days = self._estimate_early_warning_days(signals, distance)
            confidence = self._compute_confidence(signals, len(s))
            emergence_type = self._classify_emergence_type(signals)

            reports.append(EmergenceReport(
                variable=var_name,
                emergence_type=emergence_type.value,
                signal_strength=round(score, 4),
                distance_to_transition=round(distance, 4),
                confidence=round(confidence, 4),
                early_warning_days_estimate=round(early_days, 1),
                signals=signals,
            ))

        reports.sort(key=lambda r: -r.signal_strength)

        n_signals = sum(1 for r in reports if r.signal_strength > 0.3)
        max_strength = max((r.signal_strength for r in reports), default=0.0)
        mean_strength = float(np.mean([r.signal_strength for r in reports])) if reports else 0.0
        system_prob = float(np.clip(
            mean_strength * np.log1p(n_signals) * 0.5 + max_strength * 0.5, 0.0, 1.0
        ))
        most_critical = reports[0].variable if reports else ""

        logger.info(
            "Emergence detection: %d signals, max_strength=%.3f, system_prob=%.3f",
            n_signals, max_strength, system_prob,
        )

        return EmergenceDetectionResult(
            reports=reports,
            n_emergence_signals=n_signals,
            max_signal_strength=round(max_strength, 4),
            mean_signal_strength=round(mean_strength, 4),
            system_emergence_probability=round(system_prob, 4),
            most_critical_variable=most_critical,
        )

    def detect_single(self, variable: str, series: np.ndarray,
                       cross_matrix: Optional[np.ndarray] = None) -> Optional[EmergenceReport]:
        s = np.asarray(series, dtype=float).ravel()
        if len(s) < self.min_series_length:
            return None

        signals = [
            self.detect_critical_slowing(s),
            self.detect_variance_increase(s),
            self.detect_flickering(s),
        ]

        if cross_matrix is not None:
            signals.append(self.detect_correlation_length(cross_matrix))

        score = self.emergence_score(signals)
        distance = self._estimate_distance_to_transition(signals)
        early_days = self._estimate_early_warning_days(signals, distance)
        confidence = self._compute_confidence(signals, len(s))
        emergence_type = self._classify_emergence_type(signals)

        return EmergenceReport(
            variable=variable,
            emergence_type=emergence_type.value,
            signal_strength=round(score, 4),
            distance_to_transition=round(distance, 4),
            confidence=round(confidence, 4),
            early_warning_days_estimate=round(early_days, 1),
            signals=signals,
        )