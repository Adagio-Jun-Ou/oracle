import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from oracle.core.types import TimeSeries


@dataclass
class CycleInfo:
    dominant_period: float
    dominant_frequency: float
    strength: float
    phase: float
    harmonics: list[dict] = field(default_factory=list)
    significance: float = 0.0


@dataclass
class CycleDecomposition:
    cycles: list[CycleInfo]
    trend: np.ndarray
    residual: np.ndarray
    total_variance: float
    explained_variance_ratio: float


class CycleDetector:
    def __init__(self, min_period: int = 3, max_period: Optional[int] = None,
                 significance_threshold: float = 0.1):
        self.min_period = min_period
        self.max_period = max_period
        self.significance_threshold = significance_threshold

    def detect_cycles(self, series: np.ndarray, n_cycles: int = 5) -> list[CycleInfo]:
        values = series[~np.isnan(series)] if np.any(np.isnan(series)) else series
        if len(values) < 2 * self.min_period:
            return []

        values_detrended = values - np.mean(values)
        window = min(256, len(values_detrended))
        freqs, psd = scipy_signal.welch(values_detrended, nperseg=window)

        peak_indices, properties = scipy_signal.find_peaks(psd, height=np.max(psd) * self.significance_threshold)
        if len(peak_indices) == 0:
            return []

        peak_powers = psd[peak_indices]
        sorted_idx = np.argsort(peak_powers)[::-1]
        top_peaks = peak_indices[sorted_idx[:n_cycles]]

        total_power = np.sum(psd)
        cycles = []
        for idx in top_peaks:
            if idx == 0 or freqs[idx] <= 0:
                continue
            period = 1.0 / freqs[idx]
            if period < self.min_period or period > len(values) / 2:
                continue

            cycle_power = psd[idx]
            relative_power = cycle_power / total_power

            amp = np.sqrt(2 * cycle_power / np.sum(np.isfinite(values_detrended)))
            phase_angle = np.angle(np.fft.fft(values_detrended)[idx]) if idx < len(np.fft.fft(values_detrended)) else 0

            harmonics = []
            for h in range(2, 5):
                h_freq = freqs[idx] * h
                if h_freq > freqs[-1]:
                    break
                h_idx = np.argmin(np.abs(freqs - h_freq))
                if abs(freqs[h_idx] - h_freq) < freqs[1]:
                    harmonics.append({
                        "harmonic": h,
                        "frequency": float(freqs[h_idx]),
                        "period": float(1.0 / freqs[h_idx]) if freqs[h_idx] > 0 else 0,
                        "power": float(psd[h_idx] / total_power),
                    })

            cycles.append(CycleInfo(
                dominant_period=float(period),
                dominant_frequency=float(freqs[idx]),
                strength=float(relative_power),
                phase=float(phase_angle),
                harmonics=harmonics,
                significance=float(np.max(psd) * 0.1 < cycle_power),
            ))

        return sorted(cycles, key=lambda c: c.strength, reverse=True)

    def decompose(self, series: np.ndarray, n_components: int = 3) -> CycleDecomposition:
        values = series.copy()
        n = len(values)

        if n < 2 * self.min_period:
            return CycleDecomposition(
                cycles=[], trend=values, residual=np.zeros_like(values),
                total_variance=float(np.var(values)),
                explained_variance_ratio=0.0,
            )

        trend = np.convolve(values, np.ones(12) / 12, mode="same")
        detrended = values - trend

        cycles_info = self.detect_cycles(detrended, n_cycles=n_components)
        total_var = float(np.var(values))

        explained = np.zeros_like(values)
        for cycle in cycles_info[:n_components]:
            t = np.arange(n)
            amplitude = np.sqrt(2 * cycle.strength * total_var)
            cycle_component = amplitude * np.sin(2 * np.pi * cycle.dominant_frequency * t + cycle.phase)
            explained += cycle_component

        residual = values - trend - explained
        explained_var = float(np.var(explained))

        return CycleDecomposition(
            cycles=cycles_info[:n_components],
            trend=trend,
            residual=residual,
            total_variance=total_var,
            explained_variance_ratio=explained_var / total_var if total_var > 0 else 0.0,
        )

    def forecast_cycles(self, series: np.ndarray, n_ahead: int = 50) -> np.ndarray:
        decomposition = self.decompose(series)
        n = len(series)
        t_future = np.arange(n, n + n_ahead)

        trend_slope = float(np.polyfit(np.arange(n), decomposition.trend, 1)[0]) if n > 1 else 0.0
        trend_forecast = decomposition.trend[-1] + trend_slope * np.arange(1, n_ahead + 1)

        cycle_forecast = np.zeros(n_ahead)
        total_var = decomposition.total_variance
        for cycle in decomposition.cycles:
            amplitude = np.sqrt(2 * cycle.strength * total_var)
            cycle_forecast += amplitude * np.sin(2 * np.pi * cycle.dominant_frequency * t_future + cycle.phase)

        return trend_forecast + cycle_forecast

    def detect_regime_cycles(self, series: np.ndarray, window: int = 60) -> list[dict]:
        n = len(series)
        if n < 2 * window:
            return []

        regimes = []
        for i in range(0, n - window, window // 2):
            segment = series[i:i + window]
            cycles = self.detect_cycles(segment, n_cycles=2)
            if cycles:
                regimes.append({
                    "start_idx": i,
                    "end_idx": i + window,
                    "dominant_period": cycles[0].dominant_period,
                    "strength": cycles[0].strength,
                    "phase": cycles[0].phase,
                })

        return regimes