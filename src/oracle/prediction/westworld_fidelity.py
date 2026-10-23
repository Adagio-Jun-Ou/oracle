# -*- coding: utf-8 -*-
"""Fidelity Engine — Rehoboam-style prediction-vs-reality divergence tracking.

Inspired by Westworld's fidelity testing: continuously compare predicted outcomes
against actual outcomes, compute divergence scores, and flag when reality deviates
from the model. Tracks per-entity and system-wide fidelity over time.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import math


@dataclass
class FidelitySnapshot:
    target_id: str
    predicted: float
    actual: float
    divergence: float
    timestamp: float
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FidelityReport:
    target_id: str
    mean_divergence: float
    divergence_trend: str  # improving / worsening / stable
    fidelity_score: float  # 0-1, higher = more predictable
    n_samples: int
    outliers: List[FidelitySnapshot]
    trajectory: List[Tuple[float, float]]  # (timestamp, divergence)
    recommendation: str


class FidelityEngine:
    """Tracks prediction accuracy and divergence over time.

    Like Rehoboam's fidelity testing: every prediction is scored against eventual
    reality. High divergence identifies outliers or model drift.
    """

    def __init__(self, window_size: int = 100, divergence_threshold: float = 0.2):
        self.window_size = window_size
        self.divergence_threshold = divergence_threshold
        self._history: Dict[str, List[FidelitySnapshot]] = defaultdict(list)
        self._global_divergence: List[float] = []

    def record(self, target_id: str, predicted: float, actual: float,
               variables: Optional[Dict[str, Any]] = None) -> float:
        """Record a prediction-vs-actual pair. Returns divergence."""
        diff = actual - predicted
        divergence = abs(diff) / max(abs(actual), abs(predicted), 0.001)
        snap = FidelitySnapshot(
            target_id=target_id,
            predicted=predicted,
            actual=actual,
            divergence=min(1.0, divergence),
            timestamp=datetime.now().timestamp(),
            variables=variables or {}
        )
        self._history[target_id].append(snap)
        if len(self._history[target_id]) > self.window_size:
            self._history[target_id] = self._history[target_id][-self.window_size:]
        self._global_divergence.append(snap.divergence)
        if len(self._global_divergence) > self.window_size * 10:
            self._global_divergence = self._global_divergence[-self.window_size * 5:]
        return snap.divergence

    def assess(self, target_id: str) -> Optional[FidelityReport]:
        """Assess fidelity for a specific target (person, asset, system)."""
        history = self._history.get(target_id, [])
        if len(history) < 2:
            return None
        divergences = [s.divergence for s in history]
        mean_div = float(np.mean(divergences))
        fidelity = max(0.0, 1.0 - mean_div)
        # Trend analysis
        half = len(divergences) // 2
        early = np.mean(divergences[:half]) if half > 0 else mean_div
        late = np.mean(divergences[half:]) if half > 0 else mean_div
        trend = "worsening" if late > early * 1.2 else "improving" if late < early * 0.8 else "stable"
        # Outliers
        outliers = [s for s in history if s.divergence > self.divergence_threshold]
        trajectory = [(s.timestamp, s.divergence) for s in history]
        rec = self._generate_recommendation(fidelity, trend, len(outliers))
        return FidelityReport(
            target_id=target_id,
            mean_divergence=round(mean_div, 4),
            divergence_trend=trend,
            fidelity_score=round(fidelity, 4),
            n_samples=len(history),
            outliers=outliers[:5],
            trajectory=trajectory,
            recommendation=rec,
        )

    def assess_all(self) -> Dict[str, FidelityReport]:
        """Assess fidelity for all tracked targets."""
        return {tid: report for tid in self._history if (report := self.assess(tid))}

    def system_fidelity(self) -> Dict[str, float]:
        """Global system-level fidelity metrics."""
        if not self._global_divergence:
            return {"fidelity": 0.5, "entropy": 1.0, "outlier_ratio": 0.0}
        divs = np.array(self._global_divergence)
        return {
            "fidelity": round(float(1.0 - np.mean(divs)), 4),
            "entropy": round(float(np.std(divs) / max(np.mean(divs), 0.001)), 4),
            "outlier_ratio": round(float(np.mean(divs > self.divergence_threshold)), 4),
            "mean_divergence": round(float(np.mean(divs)), 4),
            "n_predictions": len(divs),
        }

    def find_outliers(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Find most divergent targets (the 'hosts' breaking the loop)."""
        scores = []
        for tid, history in self._history.items():
            if len(history) >= 2:
                avg_div = float(np.mean([s.divergence for s in history]))
                scores.append((tid, avg_div))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def _generate_recommendation(self, fidelity: float, trend: str,
                                  n_outliers: int) -> str:
        parts = []
        if fidelity < 0.5:
            parts.append("CRITICAL: Fidelity below 50% — model predictions unreliable.")
        elif fidelity < 0.7:
            parts.append("WARNING: Moderate fidelity degradation.")
        else:
            parts.append("ACCEPTABLE: Fidelity within operational bounds.")
        if trend == "worsening":
            parts.append("Trend is WORSENING — imminent model recalibration required.")
        if n_outliers > 0:
            parts.append(f"{n_outliers} outlier events detected.")
        return " ".join(parts)
