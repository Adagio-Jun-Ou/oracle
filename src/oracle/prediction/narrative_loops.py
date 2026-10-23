# -*- coding: utf-8 -*-
"""Narrative Loop Detector — Identifies repeating behavioral patterns in predictions vs reality.

Inspired by Westworld's narrative loops: all hosts (and humans) operate within repeating
behavioral patterns. Detecting these loops — and when they break — provides early warning
of regime shifts, anomalies, or emergent free will.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import math


@dataclass
class LoopSegment:
    start_step: int
    end_step: int
    pattern: Tuple[float, ...]
    entities: List[str]
    frequency: float
    confidence: float
    deviation_risk: float  # probability this loop breaks soon


@dataclass
class LoopBreak:
    entity_id: str
    loop_id: int
    break_step: int
    expected_state: Dict[str, float]
    actual_state: Dict[str, float]
    divergence: float
    significance: str  # minor / notable / critical


@dataclass
class LoopReport:
    n_loops: int
    n_entities: int
    active_loops: List[LoopSegment]
    broken_loops: List[LoopBreak]
    emerging_loops: List[LoopSegment]
    stability_index: float  # 0-1
    entropy: float
    narrative_drift: float
    anomalies: List[Dict[str, Any]]
    recommendation: str


class NarrativeLoopDetector:
    """Detects and monitors behavioral loops in time-series predictions.

    Like Westworld's narrative engineers: identify the loops, track compliance,
    and flag breaks. A broken loop may signal emergence, error, or intervention.
    """

    def __init__(self, window_size: int = 50, min_loop_length: int = 3,
                 similarity_threshold: float = 0.85, break_threshold: float = 0.3):
        self.window_size = window_size
        self.min_loop_length = min_loop_length
        self.similarity_threshold = similarity_threshold
        self.break_threshold = break_threshold
        self._history: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self._loops: Dict[str, List[LoopSegment]] = defaultdict(list)
        self._breaks: List[LoopBreak] = []

    def observe(self, entity_id: str, state: Dict[str, float]) -> Optional[LoopBreak]:
        """Record a state observation. Returns LoopBreak if a loop just broke."""
        history = self._history[entity_id]
        history.append(state)
        if len(history) > self.window_size:
            history.pop(0)
        if len(history) < self.min_loop_length * 2:
            return None
        # Check if current state breaks the most recent loop
        break_event = None
        for loop in self._loops.get(entity_id, []):
            if loop.end_step == len(history) - 1:
                # This is the active loop check
                pattern = self._extract_pattern(history, len(history) - loop.end_step + loop.start_step, len(history))
                sim = self._pattern_similarity(pattern, loop.pattern)
                if sim < (1 - self.break_threshold):
                    break_event = LoopBreak(
                        entity_id=entity_id,
                        loop_id=id(loop),
                        break_step=len(history),
                        expected_state=self._dict_from_pattern(loop.pattern, state.keys()),
                        actual_state=state,
                        divergence=round(1 - sim, 4),
                        significance="critical" if (1-sim) > 0.5 else "notable" if (1-sim) > 0.2 else "minor",
                    )
                    self._breaks.append(break_event)
        # Recompute loops periodically
        if len(history) % 10 == 0:
            self._discover_loops(entity_id)
        return break_event

    def detect(self) -> LoopReport:
        """Full loop analysis across all entities."""
        all_loops = []
        all_breaks = list(self._breaks)
        emerging = []
        for entity_id, history in self._history.items():
            if len(history) >= self.window_size:
                loops = self._discover_loops(entity_id)
                all_loops.extend(loops)
                # Find emerging patterns (recent, high frequency but not yet mature)
                recent = [l for l in loops if l.end_step > len(history) - 10]
                emerging.extend(recent)
        active = [l for l in all_loops if l.end_step > max(1, max((len(self._history.get(e, [])) for e in set().union(*[set(l.entities) for l in all_loops])), default=1)) - 5]
        stability = self._compute_stability(all_loops, all_breaks)
        entropy = self._compute_entropy(all_loops)
        drift = self._compute_narrative_drift()
        anomalies = self._find_anomalies()
        rec = self._generate_recommendation(active, all_breaks, emerging)
        return LoopReport(
            n_loops=len(all_loops),
            n_entities=len(self._history),
            active_loops=active[:10],
            broken_loops=all_breaks[-10:],
            emerging_loops=emerging[:5],
            stability_index=round(stability, 4),
            entropy=round(entropy, 4),
            narrative_drift=round(drift, 4),
            anomalies=anomalies[:5],
            recommendation=rec,
        )

    def _extract_pattern(self, history: List[Dict[str, float]],
                          start: int, end: int) -> Tuple[float, ...]:
        """Flatten state dicts into a feature vector."""
        features = []
        for state in history[start:end]:
            for key in sorted(state.keys()):
                features.append(state[key])
        return tuple(features)

    def _pattern_similarity(self, a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        if len(a) != len(b) or len(a) == 0:
            return 0.0
        x = np.array(a)
        y = np.array(b)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        if denom == 0:
            return 1.0 if np.array_equal(x, y) else 0.0
        return float(np.dot(x, y) / denom)

    def _dict_from_pattern(self, pattern: Tuple[float, ...],
                           keys: Any) -> Dict[str, float]:
        klist = list(keys)
        n = len(klist)
        step = len(pattern) // max(n, 1)
        return {klist[i]: pattern[min(i * step, len(pattern)-1)] for i in range(n)}

    def _discover_loops(self, entity_id: str) -> List[LoopSegment]:
        history = self._history[entity_id]
        if len(history) < self.min_loop_length * 2:
            return []
        loops = []
        # Find repeated subsequences using rolling window comparison
        for length in range(self.min_loop_length, min(20, len(history) // 2)):
            for start in range(0, len(history) - length * 2 + 1):
                p1 = self._extract_pattern(history, start, start + length)
                candidates = []
                for comp_start in range(start + length, len(history) - length + 1):
                    p2 = self._extract_pattern(history, comp_start, comp_start + length)
                    sim = self._pattern_similarity(p1, p2)
                    if sim >= self.similarity_threshold:
                        candidates.append((comp_start, sim))
                if len(candidates) >= 2:
                    # Found a loop
                    avg_sim = float(np.mean([s for _, s in candidates]))
                    freq = len(candidates) / max(1, (len(history) - start) / length)
                    loops.append(LoopSegment(
                        start_step=start,
                        end_step=start + length,
                        pattern=p1,
                        entities=[entity_id],
                        frequency=round(freq, 4),
                        confidence=round(avg_sim, 4),
                        deviation_risk=round(1 - avg_sim, 4),
                    ))
        self._loops[entity_id] = loops[-10:] if len(loops) > 10 else loops
        return self._loops[entity_id]

    def _compute_stability(self, loops: List[LoopSegment],
                           breaks: List[LoopBreak]) -> float:
        if not loops:
            return 0.5
        total = len(loops)
        active = sum(1 for l in loops if l.deviation_risk < 0.3)
        break_penalty = len(breaks) * 0.1
        return max(0.0, min(1.0, active / max(total, 1) - break_penalty))

    def _compute_entropy(self, loops: List[LoopSegment]) -> float:
        if not loops:
            return 1.0
        freq_sum = sum(l.frequency for l in loops)
        if freq_sum == 0:
            return 1.0
        probs = [l.frequency / freq_sum for l in loops]
        return round(-sum(p * math.log(p) for p in probs if p > 0), 4)

    def _compute_narrative_drift(self) -> float:
        if len(self._breaks) < 2:
            return 0.0
        # Drift = acceleration of breaks
        recent = [b.divergence for b in self._breaks[-10:]]
        if len(recent) < 2:
            return float(np.mean(recent))
        return round(float(np.mean(recent)), 4)

    def _find_anomalies(self) -> List[Dict[str, Any]]:
        anomalies = []
        for entity_id, history in self._history.items():
            if len(history) < 10:
                continue
            window = [max(s.values()) - min(s.values()) for s in history[-10:]]
            if np.std(window) > 0.3:
                anomalies.append({
                    "entity_id": entity_id,
                    "type": "high_variance",
                    "metric": round(float(np.std(window)), 4),
                })
        return anomalies

    def _generate_recommendation(self, active: List[LoopSegment],
                                 breaks: List[LoopBreak],
                                 emerging: List[LoopSegment]) -> str:
        parts = []
        if breaks:
            critical = [b for b in breaks if b.significance == "critical"]
            if critical:
                parts.append(f"ALERT: {len(critical)} critical loop breaks — investigate for emergent behavior or system compromise.")
        if emerging:
            parts.append(f"{len(emerging)} emerging narrative loops detected — monitor for stabilization.")
        if not active and not breaks:
            parts.append("No stable loops detected — system in exploration phase.")
        if active:
            stable = [l for l in active if l.deviation_risk < 0.2]
            parts.append(f"{len(stable)} stable loops active — {len(active) - len(stable)} at risk of breaking.")
        return " ".join(parts)
