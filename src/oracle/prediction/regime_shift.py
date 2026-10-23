"""Regime Shift Predictor — Structural break early warning system.

Detects impending regime shifts in complex systems using multiple early-warning
signals drawn from complex systems theory:

1. CUSUM/EWMA change detection — cumulative sum control charts
2. Variance ratio test — increasing variance signals approaching bifurcation
3. Autocorrelation increase — critical slowing down (lag-1 AR coefficient)
4. Skewness shift — flickering between alternative stable states
5. Spatial/temporal correlation length — increasing coherence before transition
6. Return rate from perturbation — recovery slows near tipping points
7. Spectral reddening — power spectrum shifts toward low frequencies

Based on Scheffer et al. (2009) "Early-warning signals for critical transitions"
and Dakos et al. (2012) "Methods for detecting early warnings of critical
transitions in time series."
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class ShiftType(Enum):
    GRADUAL = "gradual"
    ABRUPT = "abrupt"
    OSCILLATORY = "oscillatory"
    COLLAPSE = "collapse"
    RECOVERY = "recovery"


class WarningSignal(Enum):
    VARIANCE_INCREASE = "variance_increase"
    AUTOCORRELATION_INCREASE = "autocorrelation_increase"
    SKEWNESS_SHIFT = "skewness_shift"
    KURTOSIS_INCREASE = "kurtosis_increase"
    SPECTRAL_REDDENING = "spectral_reddening"
    CUSUM_SIGNAL = "cusum_signal"
    RETURN_RATE_SLOWING = "return_rate_slowing"
    CORRELATION_INCREASE = "correlation_increase"


@dataclass
class EarlyWarningResult:
    signal: WarningSignal
    detected: bool
    strength: float
    raw_value: float
    baseline_value: float
    p_value: float
    description: str


@dataclass
class RegimeShiftWarning:
    variable: str
    shift_type: ShiftType
    probability: float
    estimated_horizon_days: Optional[int]
    warning_signals: List[EarlyWarningResult]
    n_signals_triggered: int
    confidence: float
    recommendation: str


@dataclass
class RegimeShiftReport:
    warnings: List[RegimeShiftWarning]
    overall_risk: float
    most_likely_shift_type: ShiftType
    system_stability: float
    recommendations: List[str]


class RegimeShiftPredictor:
    def __init__(self, window_size: int = 30, significance_level: float = 0.05):
        self.window_size = window_size
        self.significance_level = significance_level

    def detect_variance_increase(self, series: np.ndarray) -> EarlyWarningResult:
        if len(series) < 2 * self.window_size:
            return EarlyWarningResult(
                signal=WarningSignal.VARIANCE_INCREASE, detected=False,
                strength=0.0, raw_value=0.0, baseline_value=0.0,
                p_value=1.0, description="Insufficient data",
            )
        recent = series[-self.window_size:]
        baseline = series[:-self.window_size][-self.window_size:]
        recent_var = float(np.var(recent, ddof=1))
        baseline_var = float(np.var(baseline, ddof=1))
        if baseline_var < 1e-15:
            ratio = 1.0
        else:
            ratio = recent_var / baseline_var
        n1, n2 = len(baseline), len(recent)
        f_stat = recent_var / max(baseline_var, 1e-15)
        df1, df2 = n1 - 1, n2 - 1
        try:
            from scipy.stats import f as f_dist
            p_value = 1.0 - f_dist.cdf(f_stat, df1, df2) if f_stat >= 1.0 else f_dist.cdf(f_stat, df1, df2)
        except ImportError:
            p_value = max(0.01, min(0.99, 1.0 / (1.0 + f_stat)))
        detected = ratio > 1.5 and p_value < self.significance_level
        strength = min(1.0, max(0, (ratio - 1.0) / 3.0))
        return EarlyWarningResult(
            signal=WarningSignal.VARIANCE_INCREASE, detected=detected,
            strength=round(strength, 4), raw_value=round(recent_var, 4),
            baseline_value=round(baseline_var, 4), p_value=round(p_value, 4),
            description=f"Variance ratio {ratio:.2f} — recent variance exceeds baseline" if detected else "No significant variance increase",
        )

    def detect_autocorrelation_increase(self, series: np.ndarray) -> EarlyWarningResult:
        if len(series) < 2 * self.window_size:
            return EarlyWarningResult(
                signal=WarningSignal.AUTOCORRELATION_INCREASE, detected=False,
                strength=0.0, raw_value=0.0, baseline_value=0.0,
                p_value=1.0, description="Insufficient data",
            )
        recent = series[-self.window_size:]
        baseline = series[:-self.window_size][-self.window_size:]
        recent_ac = self._lag1_autocorrelation(recent)
        baseline_ac = self._lag1_autocorrelation(baseline)
        diff = recent_ac - baseline_ac
        detected = recent_ac > 0.5 and diff > 0.1
        strength = min(1.0, max(0, recent_ac))
        p_value = max(0.01, min(0.99, 1.0 - abs(diff) * 5))
        return EarlyWarningResult(
            signal=WarningSignal.AUTOCORRELATION_INCREASE, detected=detected,
            strength=round(strength, 4), raw_value=round(recent_ac, 4),
            baseline_value=round(baseline_ac, 4), p_value=round(p_value, 4),
            description=f"AR(1) coeff {recent_ac:.3f} vs baseline {baseline_ac:.3f} — critical slowing detected" if detected else "No autocorrelation increase",
        )

    def detect_skewness_shift(self, series: np.ndarray) -> EarlyWarningResult:
        if len(series) < 2 * self.window_size:
            return EarlyWarningResult(
                signal=WarningSignal.SKEWNESS_SHIFT, detected=False,
                strength=0.0, raw_value=0.0, baseline_value=0.0,
                p_value=1.0, description="Insufficient data",
            )
        recent = series[-self.window_size:]
        baseline = series[:-self.window_size][-self.window_size:]
        recent_skew = float(self._skewness(recent))
        baseline_skew = float(self._skewness(baseline))
        shift = abs(recent_skew - baseline_skew)
        detected = shift > 0.5 or abs(recent_skew) > 1.0
        strength = min(1.0, max(0, shift / 2.0))
        p_value = max(0.01, min(0.99, 1.0 - shift * 3))
        return EarlyWarningResult(
            signal=WarningSignal.SKEWNESS_SHIFT, detected=detected,
            strength=round(strength, 4), raw_value=round(recent_skew, 4),
            baseline_value=round(baseline_skew, 4), p_value=round(p_value, 4),
            description=f"Skewness shift {baseline_skew:.2f} → {recent_skew:.2f} — flickering between states" if detected else "No skewness shift",
        )

    def detect_kurtosis_increase(self, series: np.ndarray) -> EarlyWarningResult:
        if len(series) < 2 * self.window_size:
            return EarlyWarningResult(
                signal=WarningSignal.KURTOSIS_INCREASE, detected=False,
                strength=0.0, raw_value=0.0, baseline_value=0.0,
                p_value=1.0, description="Insufficient data",
            )
        recent = series[-self.window_size:]
        baseline = series[:-self.window_size][-self.window_size:]
        recent_kurt = float(self._excess_kurtosis(recent))
        baseline_kurt = float(self._excess_kurtosis(baseline))
        diff = recent_kurt - baseline_kurt
        detected = recent_kurt > 5.0 or diff > 2.0
        strength = min(1.0, max(0, diff / 5.0))
        p_value = max(0.01, min(0.99, 1.0 - abs(diff) * 2))
        return EarlyWarningResult(
            signal=WarningSignal.KURTOSIS_INCREASE, detected=detected,
            strength=round(strength, 4), raw_value=round(recent_kurt, 4),
            baseline_value=round(baseline_kurt, 4), p_value=round(p_value, 4),
            description=f"Kurtosis {recent_kurt:.2f} vs baseline {baseline_kurt:.2f} — fat tails emerging" if detected else "No kurtosis increase",
        )

    def detect_spectral_reddening(self, series: np.ndarray) -> EarlyWarningResult:
        if len(series) < 64:
            return EarlyWarningResult(
                signal=WarningSignal.SPECTRAL_REDDENING, detected=False,
                strength=0.0, raw_value=0.0, baseline_value=0.0,
                p_value=1.0, description="Insufficient data for spectral analysis",
            )
        fft_vals = np.fft.rfft(series - np.mean(series))
        power = np.abs(fft_vals) ** 2
        if len(power) < 4:
            return EarlyWarningResult(
                signal=WarningSignal.SPECTRAL_REDDENING, detected=False,
                strength=0.0, raw_value=0.0, baseline_value=0.0,
                p_value=1.0, description="Insufficient frequency resolution",
            )
        freqs = np.fft.rfftfreq(len(series))
        low_freq_power = np.sum(power[:max(2, len(power) // 4)])
        high_freq_power = np.sum(power[max(2, len(power) // 4):])
        total_power = low_freq_power + high_freq_power
        if total_power < 1e-15:
            ratio = 1.0
        else:
            ratio = low_freq_power / max(high_freq_power, 1e-15)
        detected = ratio > 4.0
        strength = min(1.0, max(0, (ratio - 1.0) / 10.0))
        p_value = max(0.01, min(0.99, 1.0 / (1.0 + ratio * 0.1)))
        return EarlyWarningResult(
            signal=WarningSignal.SPECTRAL_REDDENING, detected=detected,
            strength=round(strength, 4), raw_value=round(ratio, 4),
            baseline_value=1.0, p_value=round(p_value, 4),
            description=f"Spectral ratio {ratio:.2f} — power shifting to low frequencies" if detected else "No spectral reddening",
        )

    def detect_cusum(self, series: np.ndarray, threshold: float = 5.0) -> EarlyWarningResult:
        if len(series) < self.window_size:
            return EarlyWarningResult(
                signal=WarningSignal.CUSUM_SIGNAL, detected=False,
                strength=0.0, raw_value=0.0, baseline_value=0.0,
                p_value=1.0, description="Insufficient data",
            )
        mean = np.mean(series[:self.window_size])
        std = max(np.std(series[:self.window_size], ddof=1), 1e-10)
        cusum_pos = np.zeros(len(series))
        cusum_neg = np.zeros(len(series))
        k = 0.5 * std
        for i in range(1, len(series)):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + (series[i] - mean) - k)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - (series[i] - mean) - k)
        max_pos = float(np.max(cusum_pos))
        max_neg = float(np.max(cusum_neg))
        signal_value = max(max_pos, max_neg)
        h = threshold * std
        detected = signal_value > h
        strength = min(1.0, max(0, signal_value / max(h, 1e-10)))
        z_score = signal_value / std
        p_value = max(0.01, min(0.99, math.exp(-z_score * 0.5)))
        return EarlyWarningResult(
            signal=WarningSignal.CUSUM_SIGNAL, detected=detected,
            strength=round(strength, 4), raw_value=round(signal_value, 4),
            baseline_value=0.0, p_value=round(p_value, 4),
            description=f"CUSUM signal {signal_value:.2f} exceeds threshold {h:.2f}" if detected else "No CUSUM signal detected",
        )

    def detect_return_rate_slowing(self, series: np.ndarray) -> EarlyWarningResult:
        if len(series) < 2 * self.window_size:
            return EarlyWarningResult(
                signal=WarningSignal.RETURN_RATE_SLOWING, detected=False,
                strength=0.0, raw_value=0.0, baseline_value=0.0,
                p_value=1.0, description="Insufficient data",
            )
        recent = series[-self.window_size:]
        baseline = series[:-self.window_size][-self.window_size:]
        recent_rate = self._estimate_return_rate(recent)
        baseline_rate = self._estimate_return_rate(baseline)
        if baseline_rate < 1e-10:
            ratio = 2.0
        else:
            ratio = baseline_rate / max(recent_rate, 1e-10)
        detected = ratio > 1.5
        strength = min(1.0, max(0, (ratio - 1.0) / 3.0))
        p_value = max(0.01, min(0.99, 1.0 / (1.0 + (ratio - 1.0) * 2)))
        return EarlyWarningResult(
            signal=WarningSignal.RETURN_RATE_SLOWING, detected=detected,
            strength=round(strength, 4), raw_value=round(recent_rate, 4),
            baseline_value=round(baseline_rate, 4), p_value=round(p_value, 4),
            description=f"Return rate {recent_rate:.4f} vs baseline {baseline_rate:.4f} — recovery slowing" if detected else "Return rate stable",
        )

    def detect_correlation_increase(self, data_matrix: np.ndarray) -> EarlyWarningResult:
        if data_matrix.shape[0] < 10 or data_matrix.shape[1] < 2:
            return EarlyWarningResult(
                signal=WarningSignal.CORRELATION_INCREASE, detected=False,
                strength=0.0, raw_value=0.0, baseline_value=0.0,
                p_value=1.0, description="Insufficient data",
            )
        n = data_matrix.shape[0]
        half = n // 2
        early_data = data_matrix[:half]
        late_data = data_matrix[half:]
        early_corr = self._mean_abs_corr(early_data)
        late_corr = self._mean_abs_corr(late_data)
        diff = late_corr - early_corr
        detected = diff > 0.1 and late_corr > 0.5
        strength = min(1.0, max(0, diff / 0.5))
        p_value = max(0.01, min(0.99, 1.0 - abs(diff) * 5))
        return EarlyWarningResult(
            signal=WarningSignal.CORRELATION_INCREASE, detected=detected,
            strength=round(strength, 4), raw_value=round(late_corr, 4),
            baseline_value=round(early_corr, 4), p_value=round(p_value, 4),
            description=f"Cross-correlation {late_corr:.3f} vs baseline {early_corr:.3f} — coherence increasing" if detected else "No correlation increase",
        )

    def detect_regime_shift(self, variable: str, series: np.ndarray,
                             data_matrix: Optional[np.ndarray] = None) -> RegimeShiftWarning:
        signals = []
        signals.append(self.detect_variance_increase(series))
        signals.append(self.detect_autocorrelation_increase(series))
        signals.append(self.detect_skewness_shift(series))
        signals.append(self.detect_kurtosis_increase(series))
        signals.append(self.detect_spectral_reddening(series))
        signals.append(self.detect_cusum(series))
        signals.append(self.detect_return_rate_slowing(series))
        if data_matrix is not None:
            signals.append(self.detect_correlation_increase(data_matrix))
        n_triggered = sum(1 for s in signals if s.detected)
        avg_strength = np.mean([s.strength for s in signals]) if signals else 0
        probability = min(1.0, n_triggered / max(len(signals), 1) * 1.5)
        probability = probability * 0.7 + avg_strength * 0.3
        probability = min(1.0, probability)
        shift_type = self._classify_shift_type(signals)
        horizon = self._estimate_horizon(signals, n_triggered)
        confidence = min(0.9, 0.3 + n_triggered * 0.1 + sum(s.strength for s in signals if s.detected) * 0.1)
        return RegimeShiftWarning(
            variable=variable,
            shift_type=shift_type,
            probability=round(probability, 4),
            estimated_horizon_days=horizon,
            warning_signals=signals,
            n_signals_triggered=n_triggered,
            confidence=round(confidence, 4),
            recommendation=self._generate_shift_recommendation(shift_type, n_triggered),
        )

    def detect_all(self, time_series_data: Dict[str, np.ndarray],
                    cross_sectional: Optional[Dict[str, np.ndarray]] = None) -> RegimeShiftReport:
        warnings = []
        for var_name, series in time_series_data.items():
            arr = np.asarray(series, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) < 2 * self.window_size:
                continue
            matrix = None
            if cross_sectional is not None:
                aligned = []
                for name, s in cross_sectional.items():
                    s_arr = np.asarray(s, dtype=float)
                    if len(s_arr) == len(arr):
                        aligned.append(s_arr)
                if len(aligned) >= 2:
                    matrix = np.column_stack(aligned)
            warning = self.detect_regime_shift(var_name, arr, matrix)
            warnings.append(warning)
        high_risk = [w for w in warnings if w.probability > 0.5]
        overall_risk = np.mean([w.probability for w in warnings]) if warnings else 0.0
        shift_counts = {}
        for w in warnings:
            shift_counts[w.shift_type] = shift_counts.get(w.shift_type, 0) + 1
        most_likely = max(shift_counts, key=shift_counts.get) if shift_counts else ShiftType.GRADUAL
        stability = 1.0 - overall_risk
        recs = []
        if high_risk:
            recs.append(f"IMMEDIATE: {len(high_risk)} variable(s) showing regime shift signals")
            recs.append("Reduce exposure to domains showing variance + autocorrelation increase")
            recs.append("Implement circuit breakers and position limits")
        else:
            recs.append("Monitor early warning indicators for changes")
        recs.append("Track cross-variable correlation as system coherence measure")
        recs.append("Prepare contingency plans for identified shift types")
        return RegimeShiftReport(
            warnings=warnings,
            overall_risk=round(overall_risk, 4),
            most_likely_shift_type=most_likely,
            system_stability=round(stability, 4),
            recommendations=recs,
        )

    def _lag1_autocorrelation(self, arr: np.ndarray) -> float:
        if len(arr) < 3:
            return 0.0
        mean = np.mean(arr)
        if np.var(arr) < 1e-15:
            return 0.0
        ac = np.corrcoef(arr[:-1], arr[1:])[0, 1]
        return float(ac) if np.isfinite(ac) else 0.0

    def _skewness(self, arr: np.ndarray) -> float:
        n = len(arr)
        if n < 3:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        if std < 1e-15:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))

    def _excess_kurtosis(self, arr: np.ndarray) -> float:
        n = len(arr)
        if n < 4:
            return 0.0
        mean = np.mean(arr)
        var = np.var(arr, ddof=1)
        if var < 1e-15:
            return 0.0
        m4 = np.mean(((arr - mean) / np.sqrt(var)) ** 4)
        return float(m4 - 3.0)

    def _estimate_return_rate(self, arr: np.ndarray) -> float:
        if len(arr) < 5:
            return 1.0
        mean = np.mean(arr)
        deviations = arr - mean
        recoveries = []
        for i in range(1, len(deviations)):
            if deviations[i - 1] != 0:
                recovery_rate = abs(deviations[i]) / max(abs(deviations[i - 1]), 1e-10)
                recoveries.append(recovery_rate)
        return float(np.mean(recoveries)) if recoveries else 0.5

    def _mean_abs_corr(self, data: np.ndarray) -> float:
        if data.shape[1] < 2:
            return 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(data.T)
        corr = np.nan_to_num(corr, nan=0.0)
        n = corr.shape[0]
        off_diag = corr[np.triu_indices(n, k=1)]
        return float(np.mean(np.abs(off_diag))) if len(off_diag) > 0 else 0.0

    def _classify_shift_type(self, signals: List[EarlyWarningResult]) -> ShiftType:
        sig_map = {s.signal: s for s in signals}
        if sig_map.get(WarningSignal.CUSUM_SIGNAL, EarlyWarningResult(WarningSignal.CUSUM_SIGNAL, False, 0, 0, 0, 1, "")).detected:
            return ShiftType.ABRUPT
        if sig_map.get(WarningSignal.SKEWNESS_SHIFT, EarlyWarningResult(WarningSignal.SKEWNESS_SHIFT, False, 0, 0, 0, 1, "")).detected:
            return ShiftType.OSCILLATORY
        if sig_map.get(WarningSignal.VARIANCE_INCREASE, EarlyWarningResult(WarningSignal.VARIANCE_INCREASE, False, 0, 0, 0, 1, "")).detected:
            ac_signal = sig_map.get(WarningSignal.AUTOCORRELATION_INCREASE, EarlyWarningResult(WarningSignal.AUTOCORRELATION_INCREASE, False, 0, 0, 0, 1, ""))
            if ac_signal.strength > 0.7:
                return ShiftType.COLLAPSE
        return ShiftType.GRADUAL

    def _estimate_horizon(self, signals: List[EarlyWarningResult], n_triggered: int) -> Optional[int]:
        if n_triggered == 0:
            return None
        avg_strength = np.mean([s.strength for s in signals if s.detected]) if any(s.detected for s in signals) else 0
        if avg_strength > 0.7:
            return max(7, int(30 * (1 - avg_strength)))
        elif avg_strength > 0.4:
            return max(14, int(90 * (1 - avg_strength)))
        else:
            return max(30, int(180 * (1 - avg_strength)))

    def _generate_shift_recommendation(self, shift_type: ShiftType, n_signals: int) -> str:
        if shift_type == ShiftType.ABRUPT:
            return f"ABRUPT shift likely ({n_signals} signals). Implement circuit breakers immediately."
        elif shift_type == ShiftType.COLLAPSE:
            return f"COLLAPSE risk ({n_signals} signals). Reduce leverage, increase reserves."
        elif shift_type == ShiftType.OSCILLATORY:
            return f"OSCILLATORY regime ({n_signals} signals). Prepare for volatile but bounded swings."
        elif shift_type == ShiftType.RECOVERY:
            return f"RECOVERY signals ({n_signals} signals). Gradual position rebuilding possible."
        return f"GRADUAL shift developing ({n_signals} signals). Monitor and adjust incrementally."