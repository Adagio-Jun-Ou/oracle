from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class WaveletResult:
    approximation: list
    details: list
    levels: int
    wavelet: str
    energy_by_level: dict
    reconstructed_error: float


@dataclass
class MultiResolutionResult:
    components: dict
    dominant_levels: list
    variance_decomposition: dict
    total_energy: float
    method: str


class WaveletDecomposition:
    def __init__(self, wavelet: str = "haar", max_level: int = 5):
        self.wavelet = wavelet
        self.max_level = max_level

    @staticmethod
    def _haar_forward(signal: np.ndarray) -> tuple:
        n = len(signal)
        if n < 2:
            return signal, np.array([])
        if n % 2 == 1:
            signal = np.append(signal, signal[-1])
            n = len(signal)
        approx = np.array([(signal[2 * i] + signal[2 * i + 1]) / np.sqrt(2) for i in range(n // 2)])
        detail = np.array([(signal[2 * i] - signal[2 * i + 1]) / np.sqrt(2) for i in range(n // 2)])
        return approx, detail

    @staticmethod
    def _d4_forward(signal: np.ndarray) -> tuple:
        n = len(signal)
        h0 = (1 + np.sqrt(3)) / (4 * np.sqrt(2))
        h1 = (3 + np.sqrt(3)) / (4 * np.sqrt(2))
        h2 = (3 - np.sqrt(3)) / (4 * np.sqrt(2))
        h3 = (1 - np.sqrt(3)) / (4 * np.sqrt(2))
        g0 = h3
        g1 = -h2
        g2 = h1
        g3 = -h0
        if n % 2 == 1:
            signal = np.append(signal, signal[-1])
            n = len(signal)
        approx = np.zeros(n // 2)
        detail = np.zeros(n // 2)
        for i in range(n // 2):
            for k in range(4):
                idx = (2 * i + k) % n
                approx[i] += h0 * signal[idx] if k == 0 else (h1 * signal[idx] if k == 1 else (h2 * signal[idx] if k == 2 else h3 * signal[idx]))
                detail[i] += g0 * signal[idx] if k == 0 else (g1 * signal[idx] if k == 1 else (g2 * signal[idx] if k == 2 else g3 * signal[idx]))
        return approx, detail

    def decompose(self, signal: np.ndarray, levels: Optional[int] = None) -> WaveletResult:
        signal = signal.copy().astype(float)
        n = len(signal)
        levels = levels or min(self.max_level, int(np.log2(max(n, 2))))
        levels = min(levels, int(np.log2(max(n, 2))))
        if levels < 1:
            levels = 1
        details = []
        approx = signal.copy()
        for level in range(levels):
            if len(approx) < 2:
                break
            if self.wavelet == "haar":
                approx, detail = self._haar_forward(approx)
            else:
                approx, detail = self._d4_forward(approx)
            details.append(detail.tolist())
        energy_by_level = {}
        for i, d in enumerate(details):
            energy_by_level[f"level_{i + 1}"] = float(np.sum(np.array(d) ** 2))
        energy_by_level["approximation"] = float(np.sum(approx ** 2))
        reconstructed = self._reconstruct(approx, details)
        error = float(np.mean((signal - reconstructed) ** 2)) if len(reconstructed) == len(signal) else float('inf')
        return WaveletResult(
            approximation=approx.tolist(),
            details=details,
            levels=len(details),
            wavelet=self.wavelet,
            energy_by_level=energy_by_level,
            reconstructed_error=error,
        )

    def _reconstruct(self, approx: np.ndarray, details: list) -> np.ndarray:
        current = approx.copy()
        for i in range(len(details) - 1, -1, -1):
            detail = np.array(details[i])
            n = len(current)
            reconstructed = np.zeros(2 * n)
            for j in range(n):
                if j < len(detail):
                    reconstructed[2 * j] = (current[j] + detail[j]) / np.sqrt(2) if j < len(detail) else current[j] / np.sqrt(2)
                    reconstructed[2 * j + 1] = (current[j] - detail[j]) / np.sqrt(2) if j < len(detail) else current[j] / np.sqrt(2)
                else:
                    reconstructed[2 * j] = current[j]
                    reconstructed[2 * j + 1] = current[j]
            current = reconstructed
        return current

    @staticmethod
    def denoise(signal: np.ndarray, threshold: float = 0.1,
                wavelet: str = "haar", levels: Optional[int] = None) -> dict:
        wd = WaveletDecomposition(wavelet=wavelet)
        result = wd.decompose(signal, levels=levels)
        denoised_details = []
        threshold_value = threshold * np.std(signal)
        for i, d in enumerate(result.details):
            d_arr = np.array(d)
            denoised = np.where(np.abs(d_arr) > threshold_value, d_arr, 0.0)
            denoised_details.append(denoised.tolist())
        approx_arr = np.array(result.approximation)
        reconstructed = wd._reconstruct(approx_arr, denoised_details)
        reconstructed = reconstructed[:len(signal)]
        noise_removed = float(np.sum(np.abs(np.array(signal[:len(reconstructed)]) - reconstructed)))
        snr = float(10 * np.log10(np.var(signal) / max(noise_removed ** 2 / len(signal), 1e-10)))
        return {
            "denoised_signal": reconstructed.tolist(),
            "original_length": len(signal),
            "denoised_length": len(reconstructed),
            "threshold": threshold,
            "threshold_value": float(threshold_value),
            "noise_energy_removed": noise_removed,
            "snr_db": snr,
            "levels": result.levels,
        }

    @staticmethod
    def multi_resolution_analysis(signal: np.ndarray, levels: int = 5,
                                   wavelet: str = "haar") -> MultiResolutionResult:
        wd = WaveletDecomposition(wavelet=wavelet, max_level=levels)
        result = wd.decompose(signal, levels=levels)
        components = {"approximation": result.approximation}
        for i, d in enumerate(result.details):
            components[f"detail_level_{i + 1}"] = d
        total_energy = sum(result.energy_by_level.values())
        variance_decomp = {k: v / max(total_energy, 1e-10) for k, v in result.energy_by_level.items()}
        threshold = 0.05
        dominant = [k for k, v in variance_decomp.items() if v > threshold]
        return MultiResolutionResult(
            components=components,
            dominant_levels=dominant,
            variance_decomposition=variance_decomp,
            total_energy=total_energy,
            method=f"wavelet_mra_{wavelet}",
        )

    @staticmethod
    def detect_change_points(signal: np.ndarray, wavelet: str = "haar",
                             threshold: float = 2.0) -> dict:
        wd = WaveletDecomposition(wavelet=wavelet, max_level=4)
        result = wd.decompose(signal)
        change_points = []
        for level, d in enumerate(result.details):
            d_arr = np.abs(np.array(d))
            mean_d = np.mean(d_arr) + 1e-10
            std_d = np.std(d_arr) + 1e-10
            exceeds = np.where(d_arr > mean_d + threshold * std_d)[0]
            for idx in exceeds:
                original_idx = int(idx * (2 ** (level + 1)))
                if original_idx < len(signal):
                    change_points.append({
                        "index": original_idx,
                        "level": level + 1,
                        "magnitude": float(d_arr[idx]),
                        "z_score": float((d_arr[idx] - mean_d) / std_d),
                    })
        change_points.sort(key=lambda x: x["magnitude"], reverse=True)
        return {
            "change_points": change_points[:20],
            "n_detected": len(change_points),
            "threshold": threshold,
            "wavelet": wavelet,
        }