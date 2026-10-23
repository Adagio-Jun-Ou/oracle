"""TemporalConsciousnessTracker — tracks collective consciousness evolution over time.

Detects how the collective heart and mind change together, identifying:
- Consciousness regimes: coherent, fragmented, transitional, crystallizing
- Phase transitions: sudden shifts in collective state
- Historical patterns: recurring emotional-cognitive configurations
- Consciousness cycles: how long coherence/dissonance phases typically last
- Multi-horizon trajectory: where consciousness is heading at 1h, 24h, 168h, 720h

Evidence base:
- Regime shift detection (Hamilton 1989, Kim & Nelson 1999)
- Phase transitions in social systems (Schelling 1978, Granovetter 1978)
- Collective emotion cycles (Jasny et al. 2018, Brady et al. 2017)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oracle.prediction.heart_pulse import EmotionalPulseEngine, EmotionalState
from oracle.prediction.mind_landscape import CognitiveLandscape
from oracle.prediction.consciousness_fusion import ConsciousnessFusion, FusionSignal, ResonanceMode


class ConsciousnessRegime(Enum):
    COHERENT = "coherent"          
    FRAGMENTED = "fragmented"      
    TRANSITIONAL = "transitional"  
    CRYSTALLIZING = "crystallizing"
    DISSOLVING = "dissolving"      
    POLARIZED = "polarized"        


@dataclass
class ConsciousnessSnapshot:
    timestamp: float
    heart_mood: Dict[str, float]        
    mind_belief: Dict[str, float]       
    fusion_signals: Dict[str, Dict]     
    global_will: float
    global_alignment: float
    phase: str
    regime: ConsciousnessRegime
    dominant_emotion: str
    dominant_narrative: str


@dataclass
class RegimeTransition:
    from_regime: ConsciousnessRegime
    to_regime: ConsciousnessRegime
    timestamp: float
    trigger_domain: str = ""
    confidence: float = 0.5
    description: str = ""


@dataclass
class ConsciousnessCycle:
    start_time: float
    end_time: float
    dominant_regime: ConsciousnessRegime
    peak_will: float
    trough_will: float
    duration_hours: float
    num_phase_changes: int


class TemporalConsciousnessTracker:
    """Tracks collective consciousness over time with regime detection."""

    def __init__(self, heart_engine: Optional[EmotionalPulseEngine] = None,
                 mind_landscape: Optional[CognitiveLandscape] = None,
                 max_snapshots: int = 1000,
                 seed: int = 42):
        self.heart = heart_engine or EmotionalPulseEngine()
        self.mind = mind_landscape or CognitiveLandscape()
        self.fusion = ConsciousnessFusion(heart_engine=self.heart, mind_landscape=self.mind, seed=seed)
        self.max_snapshots = max_snapshots
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._snapshots: deque[ConsciousnessSnapshot] = deque(maxlen=max_snapshots)
        self._transitions: List[RegimeTransition] = []
        self._cycles: List[ConsciousnessCycle] = []
        self._current_regime: Optional[ConsciousnessRegime] = None
        self._regime_duration: Dict[ConsciousnessRegime, float] = {r: 0.0 for r in ConsciousnessRegime}
        self._wills_over_time: List[float] = []
        self._alignments_over_time: List[float] = []

    def tick(self, domain_states: Optional[Dict[str, float]] = None) -> ConsciousnessSnapshot:
        now = time.time()
        fusion_result = self.fusion.fuse(domain_states)
        signals_dict = fusion_result.get("signals", {})
        phase_info = fusion_result.get("phase", {})
        phase_name = phase_info.get("phase", "flux")
        global_will = fusion_result.get("global_will", 0.0)
        consciousness_map = fusion_result.get("consciousness_map", {})
        global_alignment = consciousness_map.get("mean_alignment", 0.5)

        heart_mood = {d: float(s["heart_mood"]) for d, s in signals_dict.items()} if signals_dict else {}
        mind_belief = {d: float(s["mind_belief"]) for d, s in signals_dict.items()} if signals_dict else {}

        regime = self._classify_regime(global_will, global_alignment, phase_name, signals_dict)
        dominant_emotion = self._find_dominant_emotion(signals_dict)
        dominant_narrative = self._find_dominant_narrative(signals_dict)

        snapshot = ConsciousnessSnapshot(
            timestamp=now, heart_mood=heart_mood, mind_belief=mind_belief,
            fusion_signals=signals_dict, global_will=global_will,
            global_alignment=global_alignment, phase=phase_name,
            regime=regime, dominant_emotion=dominant_emotion,
            dominant_narrative=dominant_narrative
        )
        self._snapshots.append(snapshot)
        self._wills_over_time.append(global_will)
        self._alignments_over_time.append(global_alignment)

        if len(self._wills_over_time) > 500:
            self._wills_over_time.pop(0)
        if len(self._alignments_over_time) > 500:
            self._alignments_over_time.pop(0)

        prev_regime = self._current_regime
        if regime != prev_regime and prev_regime is not None:
            transition = RegimeTransition(
                from_regime=prev_regime, to_regime=regime,
                timestamp=now, trigger_domain=dominant_emotion.split(":")[0] if ":" in dominant_emotion else "",
                confidence=float(np.clip(abs(global_will - 0.5) * 2, 0.0, 1.0)),
                description=f"Consciousness shifted from {prev_regime.value} to {regime.value}"
            )
            self._transitions.append(transition)

        self._current_regime = regime
        if regime in self._regime_duration:
            self._regime_duration[regime] += 1.0

        return snapshot

    def _classify_regime(self, will: float, alignment: float,
                          phase: str, signals: Dict) -> ConsciousnessRegime:
        will_std = float(np.std([s.get("collective_will", 0.3) for s in signals.values()])) if len(signals) > 1 else 0.0
        dissonant_count = sum(1 for s in signals.values() if s.get("resonance_mode") == "dissonant")
        total = max(len(signals), 1)
        if will > 0.65 and alignment > 0.65:
            return ConsciousnessRegime.COHERENT
        elif dissonant_count >= total * 0.5:
            return ConsciousnessRegime.POLARIZED
        elif will < 0.35 and alignment < 0.35:
            return ConsciousnessRegime.FRAGMENTED
        elif will > 0.5 and alignment > 0.45 and will_std < 0.15:
            return ConsciousnessRegime.CRYSTALLIZING
        elif will > 0.35 and alignment > 0.35 and will_std > 0.2:
            return ConsciousnessRegime.TRANSITIONAL
        else:
            return ConsciousnessRegime.DISSOLVING

    def _find_dominant_emotion(self, signals: Dict) -> str:
        max_will = 0.0
        best = "none"
        for domain, sig in signals.items():
            w = sig.get("collective_will", 0)
            if w > max_will:
                max_will = w
                best = domain
        return best

    def _find_dominant_narrative(self, signals: Dict) -> str:
        return self.mind.narrative_landscape().get("most_dominant", "")

    def get_latest(self) -> Optional[ConsciousnessSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    def get_snapshots(self, n: int = 50) -> List[ConsciousnessSnapshot]:
        snaps = list(self._snapshots)
        return snaps[-n:]

    def detect_regime_shift(self) -> Optional[Dict[str, Any]]:
        if len(self._transitions) == 0:
            return None
        latest = self._transitions[-1]
        return {
            "from": latest.from_regime.value, "to": latest.to_regime.value,
            "timestamp": latest.timestamp, "trigger": latest.trigger_domain,
            "confidence": latest.confidence, "description": latest.description
        }

    def regime_shift_history(self) -> List[Dict[str, Any]]:
        return [{"from": t.from_regime.value, "to": t.to_regime.value,
                 "timestamp": t.timestamp, "trigger": t.trigger_domain}
                for t in self._transitions]

    def forecast_consciousness(self, horizon_hours: float = 168) -> Dict[str, Any]:
        if len(self._snapshots) < 3:
            return {
                "horizon_hours": horizon_hours,
                "trajectory": "insufficient_data",
                "predicted_regime": "unknown",
                "predicted_will": 0.5,
                "confidence": 0.1
            }

        wills = self._wills_over_time
        aligns = self._alignments_over_time
        if len(wills) < 5:
            return {
                "horizon_hours": horizon_hours,
                "trajectory": "insufficient_data",
                "predicted_regime": self._current_regime.value if self._current_regime else "unknown",
                "predicted_will": wills[-1] if wills else 0.5,
                "confidence": 0.2
            }

        recent_wills = wills[-20:]
        recent_aligns = aligns[-20:]
        x = np.arange(len(recent_wills))
        will_slope = np.polyfit(x, recent_wills, 1)[0]
        align_slope = np.polyfit(x, recent_aligns, 1)[0]

        steps = max(1, int(horizon_hours / 24))
        predicted_will = np.clip(recent_wills[-1] + will_slope * steps, 0.0, 1.0)
        predicted_alignment = np.clip(recent_aligns[-1] + align_slope * steps, 0.0, 1.0)

        if predicted_will > 0.65 and predicted_alignment > 0.65:
            predicted_regime = ConsciousnessRegime.COHERENT.value
        elif predicted_will < 0.35:
            predicted_regime = ConsciousnessRegime.FRAGMENTED.value
        elif predicted_will > 0.5 and predicted_alignment > 0.45:
            predicted_regime = ConsciousnessRegime.CRYSTALLIZING.value
        else:
            predicted_regime = ConsciousnessRegime.TRANSITIONAL.value

        will_vol = float(np.std(recent_wills)) if len(recent_wills) > 1 else 0.1
        confidence = max(0.05, min(0.95, 1.0 - will_vol * 3))

        direction = "rising" if will_slope > 0.003 else "falling" if will_slope < -0.003 else "stable"

        return {
            "horizon_hours": horizon_hours,
            "trajectory": direction,
            "predicted_regime": predicted_regime,
            "predicted_will": round(predicted_will, 4),
            "predicted_alignment": round(predicted_alignment, 4),
            "confidence": round(confidence, 4),
            "will_acceleration": round(float(np.polyfit(x, recent_wills, 2)[0]) if len(recent_wills) > 5 else 0.0, 4),
            "critical_regime_change": abs(will_slope) > 0.01 and predicted_will > 0.6
        }

    def consciousness_volatility(self) -> Dict[str, Any]:
        if len(self._wills_over_time) < 5:
            return {"overall": 0.1, "trend": "stable"}
        recent = self._wills_over_time[-30:]
        vol = float(np.std(recent))
        vol_hist = []
        window = 10
        for i in range(window, len(self._wills_over_time)):
            vol_hist.append(float(np.std(self._wills_over_time[i-window:i])))
        vol_trend = "increasing" if len(vol_hist[-20:]) >= 2 and float(np.polyfit(range(len(vol_hist[-20:])), vol_hist[-20:], 1)[0]) > 0.001 else "stable"
        return {"overall": round(vol, 4), "trend": vol_trend,
                "is_volatile": vol > 0.12, "is_calming": vol_trend == "stable" and vol < 0.08}

    def regime_duration_stats(self) -> Dict[str, float]:
        total = sum(self._regime_duration.values()) or 1
        return {r.value: round(d / total, 4) for r, d in self._regime_duration.items()}

    def summary(self) -> Dict[str, Any]:
        latest = self.get_latest()
        forecast = self.forecast_consciousness()
        volatility = self.consciousness_volatility()
        regime_dist = self.regime_duration_stats()
        latest_shift = self.detect_regime_shift()

        return {
            "current": {
                "regime": latest.regime.value if latest else "unknown",
                "will": latest.global_will if latest else 0.5,
                "alignment": latest.global_alignment if latest else 0.5,
                "phase": latest.phase if latest else "unknown",
                "dominant_emotion": latest.dominant_emotion if latest else "",
                "dominant_narrative": latest.dominant_narrative if latest else "",
            },
            "forecast": forecast,
            "volatility": volatility,
            "regime_distribution": regime_dist,
            "latest_shift": latest_shift,
            "snapshot_count": len(self._snapshots),
            "transition_count": len(self._transitions),
        }