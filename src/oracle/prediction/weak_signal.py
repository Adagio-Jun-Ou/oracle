from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class AmplifiedSignal:
    signal_type: str
    strength: float
    source_domain: str
    lead_time_estimate: float
    confidence: float
    description: str


@dataclass
class WeakSignalReport:
    signals: List[AmplifiedSignal] = field(default_factory=list)
    total_signal_count: int = 0
    urgency_rank: List[str] = field(default_factory=list)
    combined_strength: float = 0.0
    dominant_signal_type: str = ""
    domains_affected: List[str] = field(default_factory=list)

    def sort_by_urgency(self) -> None:
        def _urgency_key(s: AmplifiedSignal) -> float:
            return -(s.strength * s.confidence / max(s.lead_time_estimate, 0.1))
        self.signals.sort(key=_urgency_key)
        self.urgency_rank = [s.signal_type for s in self.signals]
        if self.signals:
            self.dominant_signal_type = self.signals[0].signal_type
            self.combined_strength = float(np.mean([s.strength for s in self.signals]))
            self.domains_affected = sorted(set(s.source_domain for s in self.signals))
        self.total_signal_count = len(self.signals)


class WeakSignalDetector:
    """Scans domain state histories for weak signals that precede regime changes.

    Detects faint early signals in noise before they become visible trends using
    entropy shift detection, cross-domain correlation drift, and multi-scale
    anomaly decomposition.
    """

    def __init__(self, entropy_window: int = 20, correlation_window: int = 30,
                 n_bins: int = 8, drift_threshold_sigma: float = 2.0,
                 min_series_length: int = 10):
        self.entropy_window = entropy_window
        self.correlation_window = correlation_window
        self.n_bins = n_bins
        self.drift_threshold_sigma = drift_threshold_sigma
        self.min_series_length = min_series_length

    @staticmethod
    def _sample_entropy(series: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
        n = len(series)
        if n < m + 2:
            return 0.0
        if r is None:
            r = 0.2 * np.std(series)
        if r == 0:
            return 0.0

        def _maxdist(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.max(np.abs(a - b)))

        def _phi(k: int) -> float:
            count = 0
            for i in range(n - k + 1):
                for j in range(i + 1, n - k + 1):
                    if _maxdist(series[i:i + k], series[j:j + k]) <= r:
                        count += 1
            if count == 0:
                return 0.0
            return count / ((n - k + 1) * (n - k))

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        if phi_m == 0 or phi_m1 == 0:
            return 0.0
        return -np.log(phi_m1 / phi_m)

    def _rolling_sample_entropy(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        out = np.full(n, np.nan)
        if n < self.min_series_length:
            return out
        for i in range(self.entropy_window, n + 1):
            window = series[i - self.entropy_window:i]
            ent = self._sample_entropy(window)
            out[i - 1] = ent
        return out

    @staticmethod
    def _histogram_entropy(series: np.ndarray, n_bins: int = 8) -> float:
        if len(series) < 3:
            return 0.0
        counts, _ = np.histogram(series, bins=n_bins)
        probs = counts.astype(float) / counts.sum()
        probs = probs[probs > 0]
        return -float(np.sum(probs * np.log(probs)))

    def _rolling_histogram_entropy(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        out = np.full(n, np.nan)
        if n < self.min_series_length:
            return out
        for i in range(self.entropy_window, n + 1):
            window = series[i - self.entropy_window:i]
            out[i - 1] = self._histogram_entropy(window, n_bins=self.n_bins)
        return out

    def detect_entropy_shift(self, series: np.ndarray,
                              method: str = "histogram") -> Dict:
        """Compute rolling sample entropy; falling entropy = approaching regime change.

        Args:
            series: 1-D numeric time series array.
            method: 'histogram' for histogram-based probability estimation,
                    'sample' for approximate sample entropy.

        Returns:
            Dictionary with entropy_series, is_dropping, drop_magnitude,
            drop_rate, current_entropy, baseline_entropy, z_score, description.
        """
        series = np.asarray(series, dtype=float).ravel()
        result: Dict = {
            "entropy_series": np.array([]),
            "is_dropping": False,
            "drop_magnitude": 0.0,
            "drop_rate": 0.0,
            "current_entropy": np.nan,
            "baseline_entropy": np.nan,
            "z_score": 0.0,
            "description": "Insufficient data for entropy analysis",
        }

        if len(series) < self.min_series_length:
            return result

        if method == "sample":
            ent = self._rolling_sample_entropy(series)
        else:
            ent = self._rolling_histogram_entropy(series)

        result["entropy_series"] = ent

        valid = ent[~np.isnan(ent)]
        if len(valid) < 4:
            return result

        baseline = float(np.nanmean(valid[:len(valid) // 2]))
        current = float(valid[-1])
        result["baseline_entropy"] = baseline
        result["current_entropy"] = current

        std_ent = float(np.nanstd(valid))
        if std_ent == 0:
            std_ent = 1e-10

        z = (current - baseline) / std_ent
        result["z_score"] = z

        tail_len = min(len(valid) // 2, max(5, len(valid) // 4))
        tail = valid[-tail_len:]
        head = valid[:tail_len]
        drop_rate = float(np.mean(np.diff(tail))) if len(tail) > 1 else 0.0
        drop_mag = float(np.mean(head)) - float(np.mean(tail))

        result["drop_magnitude"] = drop_mag
        result["drop_rate"] = drop_rate

        if z < -self.drift_threshold_sigma:
            result["is_dropping"] = True
            result["description"] = (
                f"Entropy dropped {abs(z):.2f} sigma below baseline — "
                f"pattern formation suggests approaching regime change"
            )
        elif drop_rate < -0.05:
            result["is_dropping"] = True
            result["description"] = (
                f"Entropy declining at rate {drop_rate:.4f}/step — "
                f"early signal of emerging structure"
            )
        else:
            result["description"] = (
                f"Entropy stable (z={z:.2f}); no regime shift signal detected"
            )

        return result

    def detect_correlation_drift(self, domain_series_dict: Dict[str, np.ndarray]) -> Dict:
        """Track cross-domain correlation changes over time.

        Computes rolling pairwise correlations between all domain series and
        flags pairs whose recent correlation deviates more than
        drift_threshold_sigma standard deviations from the historical mean.

        Args:
            domain_series_dict: Mapping of domain name to 1-D numeric array.
                All series must share the same length.

        Returns:
            Dictionary with drift_pairs, correlation_matrix, rolling_correlations,
            all_drifts, description.
        """
        domains = sorted(domain_series_dict.keys())
        n_domains = len(domains)
        result: Dict = {
            "drift_pairs": [],
            "correlation_matrix": np.eye(max(n_domains, 1)),
            "rolling_correlations": {},
            "all_drifts": [],
            "description": "Insufficient data for correlation drift analysis",
        }

        if n_domains < 2:
            return result

        series_list = [np.asarray(domain_series_dict[d], dtype=float).ravel() for d in domains]
        min_len = min(len(s) for s in series_list)
        if min_len < self.correlation_window + 5:
            return result

        series_list = [s[:min_len] for s in series_list]
        matrix = np.zeros((n_domains, n_domains))
        for i in range(n_domains):
            for j in range(n_domains):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    si, sj = series_list[i], series_list[j]
                    si_m, sj_m = si - si.mean(), sj - sj.mean()
                    denom = np.std(si) * np.std(sj) * len(si)
                    if denom == 0:
                        matrix[i, j] = 0.0
                    else:
                        matrix[i, j] = float(np.dot(si_m, sj_m) / denom)
        result["correlation_matrix"] = matrix

        n_windows = min_len - self.correlation_window + 1
        if n_windows < 3:
            return result

        drift_pairs = []
        all_drifts = []

        for i in range(n_domains):
            for j in range(i + 1, n_domains):
                si, sj = series_list[i], series_list[j]
                rolling_corr = np.zeros(n_windows)
                for w in range(n_windows):
                    wi = si[w:w + self.correlation_window]
                    wj = sj[w:w + self.correlation_window]
                    wi_m, wj_m = wi - wi.mean(), wj - wj.mean()
                    denom = np.std(wi) * np.std(wj) * len(wi)
                    if denom == 0:
                        rolling_corr[w] = 0.0
                    else:
                        rolling_corr[w] = float(np.dot(wi_m, wj_m) / denom)

                pair_key = f"{domains[i]}-{domains[j]}"
                result["rolling_correlations"][pair_key] = rolling_corr

                mean_corr = float(np.mean(rolling_corr))
                std_corr = float(np.std(rolling_corr))
                if std_corr == 0:
                    std_corr = 1e-10

                current_corr = float(rolling_corr[-1])
                z = abs(current_corr - mean_corr) / std_corr

                drift_info = {
                    "pair": pair_key,
                    "domains": (domains[i], domains[j]),
                    "current_correlation": current_corr,
                    "mean_correlation": mean_corr,
                    "std_correlation": std_corr,
                    "z_score": z,
                    "is_drifting": z > self.drift_threshold_sigma,
                    "direction": "increased" if current_corr > mean_corr else "decreased",
                }
                all_drifts.append(drift_info)

                if z > self.drift_threshold_sigma:
                    drift_pairs.append({
                        "pair": pair_key,
                        "current": current_corr,
                        "mean": mean_corr,
                        "z_score": z,
                        "direction": drift_info["direction"],
                    })

        result["drift_pairs"] = drift_pairs
        result["all_drifts"] = all_drifts

        if drift_pairs:
            directions = [p["direction"] for p in drift_pairs]
            increasing = sum(1 for d in directions if d == "increased")
            decreasing = len(drift_pairs) - increasing
            result["description"] = (
                f"{len(drift_pairs)} domain pair(s) show correlation drift "
                f"({increasing} increasing, {decreasing} decreasing) — "
                f"structural relationships changing"
            )
        else:
            result["description"] = "No significant correlation drift detected across domain pairs"

        return result

    def detect_multi_scale_anomaly(self, series: np.ndarray,
                                   scales: Optional[List[int]] = None) -> Dict:
        """Detect anomalies at multiple time scales using moving-average decomposition.

        At each scale, the series is decomposed into a trend (moving average at that
        scale) and residual. Anomalies are flagged where the absolute residual exceeds
        a threshold based on the residual distribution at that scale.

        Args:
            series: 1-D numeric time series array.
            scales: List of integer window sizes for multi-scale decomposition.
                Defaults to [4, 8, 16, 32].

        Returns:
            Dictionary with anomalies_by_scale, combined_anomaly_score,
            anomaly_indices, scales_used, description.
        """
        series = np.asarray(series, dtype=float).ravel()
        if scales is None:
            scales = [4, 8, 16, 32]

        result: Dict = {
            "anomalies_by_scale": {},
            "combined_anomaly_score": np.zeros(len(series)),
            "anomaly_indices": [],
            "scales_used": scales,
            "description": "Insufficient data for multi-scale anomaly detection",
        }

        if len(series) < max(scales) + 2:
            return result

        combined_score = np.zeros(len(series))
        any_anomaly = False

        for scale in scales:
            if scale >= len(series):
                continue

            trend = np.full(len(series), np.nan)
            for i in range(scale - 1, len(series)):
                trend[i] = float(np.mean(series[i - scale + 1:i + 1]))

            valid_mask = ~np.isnan(trend)
            residual = np.full(len(series), np.nan)
            residual[valid_mask] = series[valid_mask] - trend[valid_mask]

            valid_residuals = residual[valid_mask]
            if len(valid_residuals) < 3:
                continue

            res_mean = float(np.mean(valid_residuals))
            res_std = float(np.std(valid_residuals))
            if res_std == 0:
                res_std = 1e-10

            z_residuals = np.full(len(series), 0.0)
            z_residuals[valid_mask] = (residual[valid_mask] - res_mean) / res_std

            anomaly_mask = np.abs(z_residuals) > self.drift_threshold_sigma
            anomaly_indices = np.where(anomaly_mask)[0].tolist()
            scale_anomaly_score = np.abs(z_residuals) / max(self.drift_threshold_sigma, 1.0)
            scale_anomaly_score = np.clip(scale_anomaly_score, 0.0, 1.0)

            result["anomalies_by_scale"][scale] = {
                "trend": trend,
                "residual": residual,
                "z_residuals": z_residuals,
                "anomaly_indices": anomaly_indices,
                "max_z": float(np.max(np.abs(z_residuals))),
                "n_anomalies": len(anomaly_indices),
            }

            weight = 1.0 / scale
            combined_score += scale_anomaly_score * weight

            if anomaly_indices:
                any_anomaly = True

        total_weight = sum(1.0 / s for s in scales if s < len(series))
        if total_weight > 0:
            combined_score /= total_weight

        result["combined_anomaly_score"] = combined_score
        high_anomaly_mask = combined_score > 0.7
        result["anomaly_indices"] = np.where(high_anomaly_mask)[0].tolist()

        total_anomalies = sum(
            info["n_anomalies"] for info in result["anomalies_by_scale"].values()
        )
        if any_anomaly:
            result["description"] = (
                f"Multi-scale anomalies detected: {total_anomalies} point(s) across "
                f"{len(result['anomalies_by_scale'])} scale(s) — "
                f"weak signal present at multiple time horizons"
            )
        else:
            result["description"] = "No multi-scale anomalies detected — series appears stable"

        return result

    def amplify(self, domain_values: Dict[str, float],
                domain_histories: Dict[str, np.ndarray]) -> WeakSignalReport:
        """Combine all detection methods into an amplified signal report.

        Runs entropy shift detection, correlation drift detection, and
        multi-scale anomaly detection across all provided domains, then
        assembles detected weak signals into a unified report sorted by urgency.

        Args:
            domain_values: Current scalar values per domain (used for descriptions).
            domain_histories: Mapping of domain name to historical 1-D series.

        Returns:
            WeakSignalReport with all detected weak signals sorted by urgency
            (strength * confidence / lead_time).
        """
        report = WeakSignalReport()
        signals: List[AmplifiedSignal] = []

        for domain, history in domain_histories.items():
            history = np.asarray(history, dtype=float).ravel()
            if len(history) < self.min_series_length:
                logger.debug("Domain %s skipped: insufficient history (%d points)", domain, len(history))
                continue

            ent_result = self.detect_entropy_shift(history, method="histogram")
            if ent_result["is_dropping"]:
                strength = min(abs(ent_result["z_score"]) / 3.0, 1.0)
                drop_rate = abs(ent_result.get("drop_rate", 0.0))
                if drop_rate > 0:
                    lead_time = min(1.0 / drop_rate, 100.0)
                else:
                    lead_time = 20.0
                confidence = min(0.5 + abs(ent_result["z_score"]) * 0.1, 0.95)
                signals.append(AmplifiedSignal(
                    signal_type="entropy_shift",
                    strength=strength,
                    source_domain=domain,
                    lead_time_estimate=lead_time,
                    confidence=confidence,
                    description=ent_result["description"],
                ))

            ms_result = self.detect_multi_scale_anomaly(history)
            n_scales_with_anomaly = sum(
                1 for s, info in ms_result["anomalies_by_scale"].items()
                if info["n_anomalies"] > 0
            )
            if n_scales_with_anomaly >= 2:
                max_score = float(np.max(ms_result["combined_anomaly_score"]))
                strength = min(max_score, 1.0)
                smallest_anomaly_scale = min(
                    (s for s, info in ms_result["anomalies_by_scale"].items()
                     if info["n_anomalies"] > 0),
                    default=32,
                )
                lead_time = float(smallest_anomaly_scale) * 2.0
                confidence = min(0.4 + 0.1 * n_scales_with_anomaly, 0.9)
                signals.append(AmplifiedSignal(
                    signal_type="multi_scale_anomaly",
                    strength=strength,
                    source_domain=domain,
                    lead_time_estimate=lead_time,
                    confidence=confidence,
                    description=ms_result["description"],
                ))

        if len(domain_histories) >= 2:
            corr_result = self.detect_correlation_drift(domain_histories)
            for dp in corr_result["drift_pairs"]:
                strength = min(dp["z_score"] / (2.0 * self.drift_threshold_sigma), 1.0)
                direction = dp["direction"]
                lead_time = 15.0 / max(dp["z_score"], 1.0)
                confidence = min(0.4 + dp["z_score"] * 0.08, 0.9)
                dom_pair = dp["pair"]
                signals.append(AmplifiedSignal(
                    signal_type=f"correlation_drift_{direction}",
                    strength=strength,
                    source_domain=dom_pair,
                    lead_time_estimate=lead_time,
                    confidence=confidence,
                    description=(
                        f"Correlation between {dom_pair} {direction} "
                        f"(z={dp['z_score']:.2f}) — structural relationship changing"
                    ),
                ))

        report.signals = signals
        report.sort_by_urgency()
        logger.info(
            "Weak signal amplification complete: %d signals detected across %d domains",
            len(signals), len(domain_histories),
        )
        return report