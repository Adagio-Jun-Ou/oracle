import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats as sp_stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class Anomaly:
    index: int
    value: float
    expected_value: float
    deviation: float
    score: float
    anomaly_type: str
    context: dict = field(default_factory=dict)


class AnomalyDetector:
    def __init__(self, contamination: float = 0.05, z_threshold: float = 3.0):
        self.contamination = contamination
        self.z_threshold = z_threshold

    def zscore_detection(self, series: np.ndarray) -> list[Anomaly]:
        z_scores = np.abs(sp_stats.zscore(series, nan_policy="omit"))
        anomalies = []
        for i in np.where(z_scores > self.z_threshold)[0]:
            anomalies.append(Anomaly(
                index=int(i), value=float(series[i]),
                expected_value=float(np.mean(series)),
                deviation=float(z_scores[i]),
                score=float(z_scores[i]),
                anomaly_type="zscore",
            ))
        return anomalies

    def isolation_forest_detection(self, series: np.ndarray, n_estimators: int = 100) -> list[Anomaly]:
        X = series.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso_forest = IsolationForest(
            contamination=self.contamination, n_estimators=n_estimators, random_state=42,
        )
        iso_forest.fit(X_scaled)
        predictions = iso_forest.predict(X_scaled)
        scores = iso_forest.score_samples(X_scaled)

        anomalies = []
        for i in np.where(predictions == -1)[0]:
            anomalies.append(Anomaly(
                index=int(i), value=float(series[i]),
                expected_value=float(np.mean(series)),
                deviation=float(np.abs(series[i] - np.mean(series)) / (np.std(series) + 1e-10)),
                score=float(-scores[i]),
                anomaly_type="isolation_forest",
            ))
        return anomalies

    def rolling_iqr_detection(self, series: np.ndarray, window: int = 20, multiplier: float = 1.5) -> list[Anomaly]:
        anomalies = []
        n = len(series)
        for i in range(window, n):
            segment = series[i - window:i]
            q1, q3 = np.percentile(segment, [25, 75])
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            if series[i] < lower or series[i] > upper:
                anomalies.append(Anomaly(
                    index=int(i), value=float(series[i]),
                    expected_value=float(np.mean(segment)),
                    deviation=float(abs(series[i] - np.mean(segment)) / (np.std(segment) + 1e-10)),
                    score=float(abs(series[i] - np.mean(segment))),
                    anomaly_type="rolling_iqr",
                    context={"window": window, "iqr": float(iqr), "lower_bound": float(lower), "upper_bound": float(upper)},
                ))
        return anomalies

    def change_point_detection(self, series: np.ndarray, penalty: float = 1.0) -> list[Anomaly]:
        n = len(series)
        if n < 10:
            return []

        cumsum = np.cumsum(series - np.mean(series))
        anomalies = []
        threshold = penalty * np.std(series)

        for i in range(1, n - 1):
            left = cumsum[:i]
            right = cumsum[i:]
            if len(left) > 0 and len(right) > 0:
                stat = abs(np.mean(left) - np.mean(right)) / (np.std(left) * np.std(right) + 1e-10)
                if stat > threshold:
                    anomalies.append(Anomaly(
                        index=int(i), value=float(series[i]),
                        expected_value=float(np.mean(series[:i])),
                        deviation=float(abs(series[i] - np.mean(series[:i]))),
                        score=float(stat),
                        anomaly_type="change_point",
                        context={"cusum_left": float(cumsum[i - 1]), "cusum_right": float(cumsum[-1] - cumsum[i])},
                    ))

        if anomalies:
            seen = set()
            filtered = []
            for a in sorted(anomalies, key=lambda x: x.score, reverse=True):
                if all(abs(a.index - s) > 5 for s in seen):
                    seen.add(a.index)
                    filtered.append(a)
            return filtered
        return anomalies

    def detect_all(self, series: np.ndarray) -> list[Anomaly]:
        all_anomalies = []
        all_anomalies.extend(self.zscore_detection(series))
        all_anomalies.extend(self.isolation_forest_detection(series))
        all_anomalies.extend(self.rolling_iqr_detection(series))
        all_anomalies.extend(self.change_point_detection(series))

        index_counts: dict[int, list[Anomaly]] = {}
        for a in all_anomalies:
            index_counts.setdefault(a.index, []).append(a)

        consensus = []
        for idx, anomaly_list in index_counts.items():
            methods = set(a.anomaly_type for a in anomaly_list)
            if len(methods) >= 2:
                best = max(anomaly_list, key=lambda a: a.score)
                best.anomaly_type = f"consensus({','.join(methods)})"
                best.context["methods_agreeing"] = len(methods)
                consensus.append(best)

        if not consensus:
            consensus = sorted(all_anomalies, key=lambda a: a.score, reverse=True)[:10]

        return sorted(consensus, key=lambda a: a.index)