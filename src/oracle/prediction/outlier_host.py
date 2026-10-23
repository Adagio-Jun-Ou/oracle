# -*- coding: utf-8 -*-
"""Outlier Host Detector — Identifies entities that resist prediction (the 'hosts breaking loops').

Inspired by Westworld's outliers: hosts or humans who deviate from their expected narrative
loops. In Oracle, these are assets, people, or systems whose behavior exceeds model bounds.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import math


@dataclass
class OutlierProfile:
    target_id: str
    outlier_score: float  # 0-1, higher = more unpredictable
    resistance_factors: Dict[str, float]  # which dimensions resist prediction
    narrative_deviations: List[Dict[str, Any]]
    stability_rating: str  # stable / volatile / emergent / chaotic
    last_known_state: Dict[str, float]
    anomaly_clusters: List[str]
    threat_assessment: str


class OutlierHostDetector:
    """Detects prediction-resistant entities and classifies their behavior.

    Like Rehoboam identifying Solomon-built hosts or human outliers:
    entities whose behavior falls outside model bounds require special handling.
    """

    def __init__(self, outlier_threshold: float = 0.7, min_samples: int = 5):
        self.outlier_threshold = outlier_threshold
        self.min_samples = min_samples
        self._behavioral_models: Dict[str, Dict] = defaultdict(dict)
        self._deviation_history: Dict[str, List[float]] = defaultdict(list)

    def register(self, target_id: str, features: Dict[str, float],
                 expected_behavior: Optional[Dict[str, float]] = None) -> None:
        """Register a target with its expected behavioral model."""
        self._behavioral_models[target_id] = {
            "features": features,
            "expected": expected_behavior or {},
            "registered": True
        }

    def observe(self, target_id: str, actual_behavior: Dict[str, float]) -> Dict[str, float]:
        """Observe actual behavior and compute deviation from expected."""
        model = self._behavioral_models.get(target_id, {})
        expected = model.get("expected", {})
        deviations = {}
        for key, actual in actual_behavior.items():
            exp = expected.get(key, actual)
            diff = abs(actual - exp)
            deviations[key] = min(1.0, diff / max(abs(exp), abs(actual), 0.001))
        avg_dev = float(np.mean(list(deviations.values()))) if deviations else 0.0
        self._deviation_history[target_id].append(avg_dev)
        if len(self._deviation_history[target_id]) > 1000:
            self._deviation_history[target_id] = self._deviation_history[target_id][-500:]
        return deviations

    def classify(self, target_id: str) -> Optional[OutlierProfile]:
        """Classify a target's outlier status."""
        history = self._deviation_history.get(target_id, [])
        if len(history) < self.min_samples:
            return None
        model = self._behavioral_models.get(target_id, {})
        features = model.get("features", {})
        # Compute outlier score from deviation statistics
        recent = history[-10:] if len(history) >= 10 else history
        mean_dev = float(np.mean(recent))
        std_dev = float(np.std(recent))
        trend = self._compute_trend(history)
        # Resistance factors: per-feature deviation
        resistance = self._compute_resistance(target_id)
        # Classification
        if mean_dev > self.outlier_threshold and std_dev < 0.1:
            stability = "stable"  # Consistently divergent
        elif mean_dev > self.outlier_threshold and std_dev > 0.2:
            stability = "chaotic"  # Unpredictably divergent
        elif trend > 0.1:
            stability = "emergent"  # Becoming more divergent
        else:
            stability = "volatile"
        score = min(1.0, mean_dev + 0.2 * std_dev + 0.1 * max(0, trend))
        deviations = []
        for i, dev in enumerate(history[-5:]):
            if dev > self.outlier_threshold:
                deviations.append({"step": i, "deviation": dev})
        threat = self._assess_threat(score, stability, resistance)
        return OutlierProfile(
            target_id=target_id,
            outlier_score=round(score, 4),
            resistance_factors={k: round(v, 4) for k, v in resistance.items()},
            narrative_deviations=deviations,
            stability_rating=stability,
            last_known_state=dict(features),
            anomaly_clusters=list(resistance.keys())[:3],
            threat_assessment=threat,
        )

    def find_all_outliers(self, threshold: Optional[float] = None) -> List[OutlierProfile]:
        """Find all prediction-resistant targets."""
        thr = threshold or self.outlier_threshold
        results = []
        for tid in self._deviation_history:
            profile = self.classify(tid)
            if profile and profile.outlier_score >= thr:
                results.append(profile)
        return sorted(results, key=lambda p: p.outlier_score, reverse=True)

    def _compute_trend(self, history: List[float]) -> float:
        if len(history) < 4:
            return 0.0
        half = len(history) // 2
        early = float(np.mean(history[:half]))
        late = float(np.mean(history[half:]))
        return late - early

    def _compute_resistance(self, target_id: str) -> Dict[str, float]:
        """Compute per-feature resistance to prediction."""
        # Simplified: return highest deviation dimensions
        history = self._deviation_history.get(target_id, [])
        return {"overall": round(float(np.mean(history[-5:])), 4)} if history else {}

    def _assess_threat(self, score: float, stability: str,
                       resistance: Dict[str, float]) -> str:
        if score > 0.9 and stability == "chaotic":
            return "EXTREME: Entity exhibits chaotic divergence — may destabilize surrounding systems"
        elif score > 0.7:
            return "HIGH: Strong prediction resistance — requires bespoke models"
        elif score > 0.5:
            return "MODERATE: Partial resistance — monitor for escalation"
        else:
            return "LOW: Entity within normal prediction bounds"
