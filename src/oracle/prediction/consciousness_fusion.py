"""ConsciousnessFusion — heart-mind integration engine.

Fuses the collective emotional state (heart) with the collective cognitive
state (mind) to detect emergent collective consciousness phenomena:

1. Resonance/Synchronization: when hearts and minds align across domains
2. Dissonance: when what people feel contradicts what they think
3. Collective Will: emergent intention from aligned heart-mind
4. Tipping Points: when coherence crosses critical threshold
5. Latent Consensus: beliefs not yet expressed but converging
6. Phase Transitions: sudden shifts in collective state

Key insight: heart-mind alignment predicts coordinated action.
Dissonance predicts surprise/volatility. Resonance predicts momentum.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oracle.prediction.heart_pulse import EmotionalPulseEngine, EmotionalState
from oracle.prediction.mind_landscape import CognitiveLandscape


class ResonanceMode(Enum):
    ALIGNED = "aligned"             
    DISSONANT = "dissonant"         
    EMERGING = "emerging"           
    DISSOLVING = "dissolving"       
    CHAOTIC = "chaotic"             
    CRYSTALLIZING = "crystallizing"  


@dataclass
class FusionSignal:
    domain: str
    heart_mood: float       
    mind_belief: float      
    alignment: float        
    resonance_mode: ResonanceMode
    collective_will: float  
    tipping_risk: float     
    confidence: float = 0.5
    
    @property
    def is_at_tipping_point(self) -> bool:
        return self.tipping_risk > 0.7
    
    @property
    def will_strength(self) -> str:
        if self.collective_will > 0.75:
            return "dominant"
        elif self.collective_will > 0.5:
            return "significant"
        elif self.collective_will > 0.3:
            return "emerging"
        return "weak"


@dataclass
class ConsciousCluster:
    cluster_id: str
    members: List[str]             
    center_heart: List[float]      
    center_mind: List[float]       
    coherence: float               
    size: int
    shared_narratives: List[str]


class ConsciousnessFusion:
    def __init__(self, heart_engine: Optional[EmotionalPulseEngine] = None,
                 mind_landscape: Optional[CognitiveLandscape] = None,
                 seed: int = 42):
        self.heart = heart_engine or EmotionalPulseEngine()
        self.mind = mind_landscape or CognitiveLandscape()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._fusion_history: Dict[str, List[FusionSignal]] = {}
        self._critical_mass: float = 0.25
        self._dissonance_threshold: float = 0.3
        self._resonance_threshold: float = 0.6

    def fuse(self, domain_states: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        self.mind.update_all_beliefs(domain_states)
        domains = self._collect_domains()
        fusion_signals = {}
        for domain in domains:
            heart_state = self.heart.pulse(domain, domain_states)
            belief_state = self.mind.collective_belief_state().get(domain, 0.5)
            heart_mood = heart_state.composite_mood
            alignment = 1.0 - abs(heart_mood - belief_state)
            collective_will = min(1.0, (alignment * heart_state.arousal * 0.7 + 
                                        (1.0 - abs(heart_mood - 0.5)) * 0.3))
            coherence_avg = float(np.mean([p.coherence for p in heart_state.pulses.values()])) if heart_state.pulses else 0.5
            tipping_risk = self._compute_tipping_risk(alignment, collective_will, coherence_avg)
            mode = self._classify_resonance(alignment, heart_mood, belief_state, tipping_risk)
            signal = FusionSignal(
                domain=domain, heart_mood=round(heart_mood, 4),
                mind_belief=round(belief_state, 4),
                alignment=round(alignment, 4),
                resonance_mode=mode,
                collective_will=round(collective_will, 4),
                tipping_risk=round(tipping_risk, 4),
                confidence=round(heart_state.arousal * alignment, 4)
            )
            fusion_signals[domain] = signal
            hist = self._fusion_history.get(domain, [])
            hist.append(signal)
            if len(hist) > 200:
                hist.pop(0)
            self._fusion_history[domain] = hist
        consciousness_map = self._assemble_consciousness_map(fusion_signals)
        global_will = self._compute_global_will(consciousness_map)
        phase = self._detect_phase(consciousness_map)
        return {
            "signals": {d: self._signal_to_dict(s) for d, s in fusion_signals.items()},
            "consciousness_map": consciousness_map,
            "global_will": round(global_will, 4),
            "phase": phase,
            "has_tipping_point": any(s.is_at_tipping_point for s in fusion_signals.values()),
            "tipping_domains": [d for d, s in fusion_signals.items() if s.is_at_tipping_point],
            "resonant_domains": [d for d, s in fusion_signals.items() if s.resonance_mode == ResonanceMode.ALIGNED],
            "dissonant_domains": [d for d, s in fusion_signals.items() if s.resonance_mode == ResonanceMode.DISSONANT],
            "collective_insight": self._synthesize_insight(fusion_signals, global_will, phase)
        }

    def _collect_domains(self) -> List[str]:
        heart_domains = set(self.heart._states.keys())
        mind_domains = set(b.domain for b in self.mind._beliefs.values())
        return sorted(heart_domains | mind_domains)

    def _compute_tipping_risk(self, alignment: float, will: float, coherence: float) -> float:
        if alignment > 0.6 and will > 0.5 and coherence > 0.4:
            base = 0.4
            base += (alignment - 0.6) * 1.5
            base += (will - 0.5) * 1.0
            base += (coherence - 0.4) * 1.0
        elif alignment < 0.3 and will < 0.3:
            base = 0.3
            base += (0.3 - alignment) * 1.0
        else:
            base = 0.2
        return float(np.clip(base, 0.0, 0.95))

    def _classify_resonance(self, alignment: float, heart_mood: float,
                              mind_belief: float, tipping_risk: float) -> ResonanceMode:
        if alignment > self._resonance_threshold:
            if tipping_risk > 0.6:
                return ResonanceMode.CRYSTALLIZING
            return ResonanceMode.ALIGNED
        elif alignment < self._dissonance_threshold:
            return ResonanceMode.DISSONANT
        elif tipping_risk > 0.5:
            return ResonanceMode.EMERGING
        elif alignment > 0.4:
            return ResonanceMode.DISSOLVING
        return ResonanceMode.CHAOTIC

    def _assemble_consciousness_map(self, signals: Dict[str, FusionSignal]) -> Dict[str, Any]:
        if not signals:
            return {"dominant": "none", "regions": {}}
        max_will = max(signals.values(), key=lambda s: s.collective_will)
        regions = {}
        for domain, sig in signals.items():
            regions[domain] = {
                "resonance": sig.resonance_mode.value,
                "will": sig.collective_will,
                "tipping": sig.tipping_risk,
                "alignment": sig.alignment
            }
        return {
            "dominant": max_will.domain,
            "dominant_will": max_will.collective_will,
            "mean_will": round(float(np.mean([s.collective_will for s in signals.values()])), 4),
            "mean_alignment": round(float(np.mean([s.alignment for s in signals.values()])), 4),
            "resonance_distribution": {
                mode.value: sum(1 for s in signals.values() if s.resonance_mode == mode)
                for mode in ResonanceMode
            },
            "regions": regions
        }

    def _compute_global_will(self, consciousness_map: Dict) -> float:
        domains = consciousness_map.get("regions", {})
        if not domains:
            return 0.3
        wills = [r["will"] for r in domains.values()]
        mean_will = float(np.mean(wills)) if wills else 0.3
        alignment_bonus = consciousness_map.get("mean_alignment", 0.5) * 0.2
        return float(np.clip(mean_will + alignment_bonus, 0.0, 1.0))

    def _detect_phase(self, consciousness_map: Dict) -> Dict[str, Any]:
        regions = consciousness_map.get("regions", {})
        dist = consciousness_map.get("resonance_distribution", {})
        aligned_count = dist.get("aligned", 0) + dist.get("crystallizing", 0)
        dissonant_count = dist.get("dissonant", 0)
        total = max(len(regions), 1)
        if aligned_count >= total * 0.5:
            phase_name = "coherence"
            description = "Collective heart and mind are aligning — coordinated action imminent"
        elif dissonant_count >= total * 0.4:
            phase_name = "fragmentation"
            description = "What people feel and what they think are pulling apart — volatility ahead"
        elif aligned_count > dissonant_count:
            phase_name = "convergence"
            description = "Collective consciousness is gathering but not yet unified"
        elif dissonant_count > aligned_count and total > 2:
            phase_name = "divergence"
            description = "Hearts and minds are dispersing — collective direction unclear"
        else:
            phase_name = "flux"
            description = "Collective consciousness is in transition"
        return {"phase": phase_name, "description": description,
                "stability": round(1.0 - float(np.std([r.get("will", 0.5) for r in regions.values()])), 4) if regions else 0.5}

    def _signal_to_dict(self, signal: FusionSignal) -> Dict[str, Any]:
        return {
            "heart_mood": signal.heart_mood, "mind_belief": signal.mind_belief,
            "alignment": signal.alignment, "resonance_mode": signal.resonance_mode.value,
            "collective_will": signal.collective_will, "tipping_risk": signal.tipping_risk,
            "confidence": signal.confidence, "will_strength": signal.will_strength,
            "at_tipping_point": signal.is_at_tipping_point
        }

    def _synthesize_insight(self, signals: Dict[str, FusionSignal],
                             global_will: float, phase: Dict) -> str:
        phase_name = phase.get("phase", "flux")
        tipping_domains = [d for d, s in signals.items() if s.is_at_tipping_point]
        parts = []
        if phase_name == "coherence":
            parts.append("The collective consciousness is unified")
        elif phase_name == "fragmentation":
            parts.append("The collective consciousness is fractured")
        elif phase_name == "convergence":
            parts.append("The collective consciousness is converging")
        elif phase_name == "divergence":
            parts.append("The collective consciousness is diverging")
        else:
            parts.append("The collective consciousness is in flux")
        if tipping_domains:
            parts.append(f"with tipping points detected in: {', '.join(tipping_domains)}")
        if global_will > 0.6:
            parts.append("— coordinated action is likely within 48 hours")
        return " ".join(parts) + "."

    def detect_dissonance_events(self) -> List[Dict[str, Any]]:
        events = []
        for domain, history in self._fusion_history.items():
            if len(history) < 3:
                continue
            recent = history[-3:]
            alignment_trend = np.mean([s.alignment for s in recent])
            if alignment_trend < 0.3 and all(s.resonance_mode == ResonanceMode.DISSONANT for s in recent):
                events.append({
                    "type": "dissonance_break",
                    "domain": domain,
                    "severity": round(0.7 + (0.3 - alignment_trend) * 2, 3),
                    "expected_resolution": "convergence_or_conflict",
                    "description": f"Dissonance in {domain}: hearts and minds diverging"
                })
            if alignment_trend > 0.7 and all(s.resonance_mode in [ResonanceMode.ALIGNED, ResonanceMode.CRYSTALLIZING] for s in recent):
                events.append({
                    "type": "resonance_surge",
                    "domain": domain,
                    "intensity": round(alignment_trend, 3),
                    "expected_resolution": "coordinated_movement",
                    "description": f"Resonance in {domain}: hearts and minds aligning"
                })
        return events

    def predict_consciousness_shift(self, horizon_hours: float = 48) -> Dict[str, Any]:
        fusion = self.fuse()
        signals = fusion.get("signals", {})
        predictions = {}
        for domain, sig_dict in signals.items():
            hist = self._fusion_history.get(domain, [])
            if len(hist) < 3:
                predictions[domain] = {"trajectory": "stable", "confidence": 0.3}
                continue
            alignments = [s.alignment for s in hist[-20:]]
            wills = [s.collective_will for s in hist[-20:]]
            if len(alignments) < 3:
                predictions[domain] = {"trajectory": "stable", "confidence": 0.3}
                continue
            x = np.arange(len(alignments))
            align_slope = np.polyfit(x, alignments, 1)[0]
            will_slope = np.polyfit(x, wills, 1)[0]
            projected_align = np.clip(alignments[-1] + align_slope * horizon_hours / 24, 0, 1)
            projected_will = np.clip(wills[-1] + will_slope * horizon_hours / 24, 0, 1)
            if align_slope > 0.003 and will_slope > 0.002:
                trajectory = "strengthening"
            elif align_slope < -0.003 and will_slope < -0.002:
                trajectory = "weakening"
            elif abs(align_slope) < 0.002 and abs(will_slope) < 0.002:
                trajectory = "stable"
            else:
                trajectory = "mixed"
            predictions[domain] = {
                "trajectory": trajectory,
                "projected_alignment": round(projected_align, 4),
                "projected_will": round(projected_will, 4),
                "confidence": round(np.clip(1.0 - np.std(wills) * 2, 0.1, 0.9), 4)
            }
        return {"predictions": predictions, "horizon_hours": horizon_hours,
                "global_phase": fusion.get("phase", {}),
                "global_will": fusion.get("global_will", 0.0)}