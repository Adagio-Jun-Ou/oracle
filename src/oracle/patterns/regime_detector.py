import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeSegment:
    start_idx: int
    end_idx: int
    mean_value: float
    volatility: float
    trend: float
    label: str
    characteristics: dict = field(default_factory=dict)


@dataclass
class RegimeResult:
    regimes: list[RegimeSegment]
    n_regimes: int
    transition_matrix: np.ndarray
    regime_labels: np.ndarray
    current_regime: str
    regime_probabilities: dict


class RegimeDetector:
    def __init__(self, n_regimes: int = 3, window: int = 20):
        self.n_regimes = n_regimes
        self.window = window

    def _compute_features(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        w = min(self.window, n // 2)
        if n < w:
            return np.column_stack([series, np.zeros(n), np.gradient(series)])

        rolling_mean = np.convolve(series, np.ones(w) / w, mode="valid")
        rolling_std = np.array([np.std(series[max(0, i - w):i + 1]) for i in range(n)])
        returns = np.diff(series, prepend=series[0])
        gradient = np.gradient(series)
        abs_returns = np.abs(returns)

        min_len = min(len(rolling_mean), len(rolling_std), len(gradient))
        features = np.column_stack([
            series[:min_len],
            rolling_mean[:min_len] if len(rolling_mean) >= min_len else np.interp(np.arange(min_len), np.arange(len(rolling_mean)), rolling_mean),
            rolling_std[:min_len],
            gradient[:min_len],
            abs_returns[:min_len],
        ])

        mask = ~np.any(np.isnan(features) | np.isinf(features), axis=1)
        features = features[mask]

        return features

    def detect_hmm_regimes(self, series: np.ndarray, n_regimes: Optional[int] = None) -> RegimeResult:
        n_r = n_regimes or self.n_regimes
        features = self._compute_features(series)
        n = len(features)

        if n < n_r * 5:
            n_r = max(2, n // 10)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_r, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)

        series_aligned = series[:n] if len(series) >= n else np.pad(series, (0, n - len(series)), mode='edge')
        regime_means = []
        regime_stds = []
        regime_names = []
        for i in range(n_r):
            mask = labels == i
            if np.sum(mask) > 0:
                regime_means.append(float(np.mean(series_aligned[mask])))
                regime_stds.append(float(np.std(series_aligned[mask])))
            else:
                regime_means.append(0.0)
                regime_stds.append(1.0)

        sorted_idx = np.argsort(regime_means)
        regime_means_arr = np.array(regime_means)
        regime_stds_arr = np.array(regime_stds)
        name_map = {int(old): int(new) for old, new in zip(sorted_idx, range(n_r))}
        regime_names_map = {0: "bear", 1: "neutral", 2: "bull"}
        labels = np.array([name_map.get(int(l), 0) for l in labels])

        segments = []
        current_label = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                segment_series = series_aligned[start:i]
                try:
                    slope = float(np.polyfit(range(len(segment_series)), segment_series, 1)[0])
                except np.linalg.LinAlgError:
                    slope = 0.0
                r = int(current_label)
                segments.append(RegimeSegment(
                    start_idx=start, end_idx=i,
                    mean_value=float(np.mean(segment_series)),
                    volatility=float(np.std(segment_series)),
                    trend=slope,
                    label=regime_names_map.get(r, f"regime_{r}"),
                    characteristics={"mean": float(regime_means_arr[r]) if r < len(regime_means_arr) else 0.0, "std": float(regime_stds_arr[r]) if r < len(regime_stds_arr) else 1.0},
                ))
                current_label = labels[i]
                start = i

        if start < len(labels):
            segment_series = series_aligned[start:]
            try:
                slope = float(np.polyfit(range(len(segment_series)), segment_series, 1)[0])
            except np.linalg.LinAlgError:
                slope = 0.0
            r = int(current_label)
            segments.append(RegimeSegment(
                start_idx=start, end_idx=len(labels),
                mean_value=float(np.mean(segment_series)),
                volatility=float(np.std(segment_series)),
                trend=slope,
                label=regime_names_map.get(r, f"regime_{r}"),
                characteristics={"mean": float(regime_means_arr[r]) if r < len(regime_means_arr) else 0.0, "std": float(regime_stds_arr[r]) if r < len(regime_stds_arr) else 1.0},
            ))

        trans_matrix = np.zeros((n_r, n_r))
        for i in range(1, len(labels)):
            trans_matrix[labels[i - 1]][labels[i]] += 1
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        trans_matrix = np.where(row_sums > 0, trans_matrix / row_sums, trans_matrix)

        current_regime_idx = int(labels[-1])
        current_regime_name = regime_names_map.get(current_regime_idx, f"regime_{current_regime_idx}")

        regime_probs = {}
        if trans_matrix.shape[0] > current_regime_idx:
            for j in range(n_r):
                name = regime_names_map.get(j, f"regime_{j}")
                regime_probs[name] = float(trans_matrix[current_regime_idx, j])

        return RegimeResult(
            regimes=segments, n_regimes=n_r, transition_matrix=trans_matrix,
            regime_labels=labels, current_regime=current_regime_name,
            regime_probabilities=regime_probs,
        )

    def forecast_regime_probability(
        self, series: np.ndarray, steps_ahead: int = 12,
    ) -> dict:
        result = self.detect_hmm_regimes(series)
        current_idx = int(result.regime_labels[-1])

        probs = result.transition_matrix[current_idx]
        forecast_probs = {f"step_{i}": {} for i in range(steps_ahead)}

        current_probs = probs.copy()
        for step in range(steps_ahead):
            for regime_idx in range(result.transition_matrix.shape[0]):
                regime_name = {0: "bear", 1: "neutral", 2: "bull"}.get(regime_idx, f"regime_{regime_idx}")
                forecast_probs[f"step_{step}"][regime_name] = float(current_probs[regime_idx])
            current_probs = current_probs @ result.transition_matrix

        return {
            "current_regime": result.current_regime,
            "transition_matrix": result.transition_matrix.tolist(),
            "forecast_probabilities": forecast_probs,
            "regime_characteristics": {
                seg.label: {"mean": seg.mean_value, "volatility": seg.volatility, "trend": seg.trend}
                for seg in result.regimes
            },
        }