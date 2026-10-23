"""EmotionalPulseEngine — the collective heart of humanity.

Measures what billions of people FEEL across 6 primary emotional axes,
how emotions spread through contagion networks, and maps emotional states
to specific domains (economic, geopolitical, health, social, technology).

Key insight: emotion precedes action. Fear/greed ratios predict markets.
Anger/calm ratios predict unrest. Hope/despair ratios predict investment.
Trust/distrust ratios predict institutional stability.

Evidence base:
- Plutchik's wheel of emotions (1980), Ekman universal emotions (1972)
- Emotional contagion via social networks (Kramer et al., 2014, PNAS)
- VIX as fear index (Whaley, 1993), consumer sentiment (Katona, 1951)
- Affective forecasting (Wilson & Gilbert, 2003)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class EmotionalAxis(Enum):
    FEAR_GREED = "fear_greed"        
    HOPE_DESPAIR = "hope_despair"    
    TRUST_DISTRUST = "trust_distrust"
    ANGER_CALM = "anger_calm"        
    LOVE_HATE = "love_hate"          
    SURPRISE_ANTICIPATION = "surprise_anticipation"  


EMOTION_NAMES: Dict[str, str] = {
    "fear_greed": "Fear/Greed", "hope_despair": "Hope/Despair",
    "trust_distrust": "Trust/Distrust", "anger_calm": "Anger/Calm",
    "love_hate": "Love/Hate", "surprise_anticipation": "Surprise/Anticipation"
}

EMOTION_POLARITIES: Dict[str, Tuple[str, str]] = {
    "fear_greed": ("fear", "greed"), "hope_despair": ("despair", "hope"),
    "trust_distrust": ("distrust", "trust"), "anger_calm": ("calm", "anger"),
    "love_hate": ("hate", "love"), "surprise_anticipation": ("surprise", "anticipation")
}


@dataclass
class EmotionalPulse:
    axis: str
    intensity: float          
    polarity: float           
    coherence: float          
    velocity: float           
    acceleration: float       
    entropy: float            
    
    @property
    def net_valence(self) -> float:
        return self.intensity * self.polarity
    
    @property
    def is_escalating(self) -> bool:
        return self.velocity > 0.01 and self.intensity > 0.4
    
    @property
    def is_deescalating(self) -> bool:
        return self.velocity < -0.01


@dataclass
class EmotionalContagion:
    source_domain: str
    target_domain: str
    source_axis: str
    coupling_strength: float
    lag_hours: float = 24.0
    evidence: str = ""


@dataclass
class EmotionalState:
    domain: str
    pulses: Dict[str, EmotionalPulse] = field(default_factory=dict)
    composite_mood: float = 0.5
    arousal: float = 0.5
    timestamp: float = 0.0
    evidence: Dict[str, str] = field(default_factory=dict)


DOMAIN_EMOTION_PROFILES: Dict[str, Dict[str, float]] = {
    "economic": {"fear_greed": 0.8, "hope_despair": 0.7, "trust_distrust": 0.6, "anger_calm": 0.3,
                 "love_hate": 0.2, "surprise_anticipation": 0.4},
    "geopolitical": {"fear_greed": 0.6, "hope_despair": 0.4, "trust_distrust": 0.8, "anger_calm": 0.7,
                     "love_hate": 0.3, "surprise_anticipation": 0.5},
    "health": {"fear_greed": 0.7, "hope_despair": 0.8, "trust_distrust": 0.5, "anger_calm": 0.2,
               "love_hate": 0.4, "surprise_anticipation": 0.3},
    "social": {"fear_greed": 0.4, "hope_despair": 0.5, "trust_distrust": 0.7, "anger_calm": 0.8,
               "love_hate": 0.9, "surprise_anticipation": 0.3},
    "technology": {"fear_greed": 0.5, "hope_despair": 0.6, "trust_distrust": 0.3, "anger_calm": 0.2,
                   "love_hate": 0.2, "surprise_anticipation": 0.9},
    "monetary": {"fear_greed": 0.9, "hope_despair": 0.5, "trust_distrust": 0.8, "anger_calm": 0.2,
                 "love_hate": 0.1, "surprise_anticipation": 0.6},
}

DOMAIN_EMOTION_BASELINES: Dict[str, Dict[str, float]] = {
    "economic": {"fear_greed": -0.15, "hope_despair": 0.2, "trust_distrust": 0.1, "anger_calm": -0.1,
                 "love_hate": 0.05, "surprise_anticipation": -0.1},
    "geopolitical": {"fear_greed": 0.1, "hope_despair": 0.0, "trust_distrust": 0.05, "anger_calm": -0.1,
                     "love_hate": -0.05, "surprise_anticipation": -0.05},
}

CONTAGION_NETWORK: List[EmotionalContagion] = [
    EmotionalContagion("economic", "monetary", "fear_greed", 0.85, 1, "VIX-VIX correlation"),
    EmotionalContagion("economic", "social", "hope_despair", 0.6, 48, "Unemployment→mood, Clark & Oswald 1994"),
    EmotionalContagion("geopolitical", "economic", "fear_greed", 0.7, 12, "Geopolitical risk→markets, Caldara & Iacoviello 2022"),
    EmotionalContagion("geopolitical", "social", "anger_calm", 0.8, 24, "Conflict→public anger, Gallup World Poll"),
    EmotionalContagion("social", "geopolitical", "anger_calm", 0.65, 72, "Civil unrest→policy response"),
    EmotionalContagion("health", "economic", "hope_despair", 0.5, 168, "Pandemic→recession, IMF WEO"),
    EmotionalContagion("monetary", "economic", "trust_distrust", 0.75, 8, "Fed credibility→markets, Binder 2017"),
    EmotionalContagion("technology", "economic", "surprise_anticipation", 0.6, 48, "Tech breakthroughs→investment"),
]


class EmotionalPulseEngine:
    def __init__(self, seed: int = 42, use_real_data: bool = False):
        self.seed = seed
        self.use_real_data = use_real_data
        self.rng = np.random.RandomState(seed)
        self._states: Dict[str, EmotionalState] = {}
        self._history: Dict[str, List[EmotionalState]] = {}
        self._contagion_traces: Dict[str, List[float]] = {}
        self._last_update: float = 0.0
        self._initialize_states()
        if use_real_data:
            self._ingest_real_sentiment()

    def _initialize_states(self) -> None:
        now = time.time()
        for domain in DOMAIN_EMOTION_PROFILES:
            pulses = {}
            for axis_str in [e.value for e in EmotionalAxis]:
                profile_weight = DOMAIN_EMOTION_PROFILES.get(domain, {}).get(axis_str, 0.5)
                baseline = DOMAIN_EMOTION_BASELINES.get(domain, {}).get(axis_str, 0.0)
                intensity = 0.3 + 0.4 * profile_weight
                polarity = max(-1.0, min(1.0, baseline + self.rng.normal(0, 0.15)))
                pulses[axis_str] = EmotionalPulse(
                    axis=axis_str, intensity=round(intensity, 4),
                    polarity=round(polarity, 4), coherence=0.5,
                    velocity=0.0, acceleration=0.0, entropy=0.3
                )
            self._states[domain] = EmotionalState(
                domain=domain, pulses=pulses,
                composite_mood=0.5, arousal=0.5, timestamp=now
            )
            self._history[domain] = [self._states[domain]]

    def _ingest_real_sentiment(self) -> None:
        pass

    def pulse(self, domain: str, domain_states: Optional[Dict[str, float]] = None) -> EmotionalState:
        now = time.time()
        state = self._states.get(domain)
        if state is None:
            state = EmotionalState(domain=domain, timestamp=now)
            self._states[domain] = state
        
        dt = max(1, now - state.timestamp) / 3600.0
        domain_data = domain_states or {}
        
        for axis_str in [e.value for e in EmotionalAxis]:
            pulse = state.pulses.get(axis_str)
            if pulse is None:
                base_pulse = self._pulse_from_domain_data(domain, axis_str, domain_data)
                state.pulses[axis_str] = base_pulse
                continue
            
            new_intensity = self._update_intensity(pulse, domain, axis_str, domain_data, dt)
            new_polarity = self._update_polarity(pulse, domain, axis_str, domain_data, dt)
            new_coherence = self._update_coherence(pulse, dt)
            new_velocity = (new_intensity - pulse.intensity) / max(dt, 1)
            new_acceleration = (new_velocity - pulse.velocity) / max(dt, 1) if dt > 0 else 0.0
            hist_vals = self._contagion_traces.get(f"{domain}_{axis_str}", [pulse.net_valence])
            new_entropy = self._compute_entropy(hist_vals)
            
            state.pulses[axis_str] = EmotionalPulse(
                axis=axis_str, intensity=round(new_intensity, 4),
                polarity=round(new_polarity, 4), coherence=round(new_coherence, 4),
                velocity=round(new_velocity, 4), acceleration=round(new_acceleration, 4),
                entropy=round(new_entropy, 4)
            )
            trace_key = f"{domain}_{axis_str}"
            if trace_key not in self._contagion_traces:
                self._contagion_traces[trace_key] = []
            self._contagion_traces[trace_key].append(pulse.net_valence)
            if len(self._contagion_traces[trace_key]) > 200:
                self._contagion_traces[trace_key].pop(0)

        self._apply_contagion(domain, dt)
        
        intensities = [p.intensity for p in state.pulses.values()]
        net_valences = [p.net_valence for p in state.pulses.values()]
        state.composite_mood = round(float(np.clip(np.mean(net_valences) * 0.5 + 0.5, 0.0, 1.0)), 4)
        state.arousal = round(float(np.clip(np.mean(intensities), 0.0, 1.0)), 4)
        state.timestamp = now
        
        hist = self._history.get(domain, [])
        hist.append(state)
        if len(hist) > 200:
            hist.pop(0)
        self._history[domain] = hist
        
        return state

    def _pulse_from_domain_data(self, domain: str, axis_str: str,
                                 domain_data: Dict[str, float]) -> EmotionalPulse:
        profile = DOMAIN_EMOTION_PROFILES.get(domain, {})
        weight = profile.get(axis_str, 0.5)
        baseline = DOMAIN_EMOTION_BASELINES.get(domain, {}).get(axis_str, 0.0)
        domain_val = domain_data.get(domain, 0.5)
        
        if axis_str == "fear_greed":
            pol = 0.6 * domain_val - 0.3
        elif axis_str == "hope_despair":
            pol = 0.7 * domain_val - 0.15
        elif axis_str == "trust_distrust":
            pol = 0.5 * domain_val - 0.25
        elif axis_str == "anger_calm":
            pol = -0.4 * domain_val + 0.1
        elif axis_str == "love_hate":
            pol = 0.3 * domain_val - 0.1
        elif axis_str == "surprise_anticipation":
            pol = 0.2 * domain_val - 0.05
        else:
            pol = baseline
        
        return EmotionalPulse(
            axis=axis_str,
            intensity=round(0.3 + 0.4 * weight + self.rng.normal(0, 0.05), 4),
            polarity=round(max(-1.0, min(1.0, pol)), 4),
            coherence=0.5, velocity=0.0, acceleration=0.0, entropy=0.3
        )

    def _update_intensity(self, pulse: EmotionalPulse, domain: str, axis_str: str,
                           domain_data: Dict[str, float], dt: float) -> float:
        base = pulse.intensity
        decay = 0.001 * dt
        noise = self.rng.normal(0, 0.01)
        
        domain_val = domain_data.get(domain, 0.5)
        signal_drive = 0.02 * abs(domain_val - 0.5)
        
        profile_weight = DOMAIN_EMOTION_PROFILES.get(domain, {}).get(axis_str, 0.5)
        attractor = 0.3 + 0.4 * profile_weight
        attraction = 0.01 * (attractor - base) * dt / 24
        
        return np.clip(base - decay + signal_drive + attraction + noise, 0.05, 0.95)

    def _update_polarity(self, pulse: EmotionalPulse, domain: str, axis_str: str,
                          domain_data: Dict[str, float], dt: float) -> float:
        base = pulse.polarity
        decay = 0.002 * base * dt / 24
        noise = self.rng.normal(0, 0.02)
        
        domain_val = domain_data.get(domain, 0.5)
        if axis_str == "fear_greed":
            signal = 0.5 * (domain_val - 0.5)
        elif axis_str == "hope_despair":
            signal = 0.4 * (domain_val - 0.5)
        elif axis_str == "trust_distrust":
            signal = 0.3 * (domain_val - 0.5)
        elif axis_str == "anger_calm":
            signal = -0.4 * (domain_val - 0.5)
        else:
            signal = 0.1 * (domain_val - 0.5)
        
        return np.clip(base - decay + signal * dt / 24 + noise, -1.0, 1.0)

    def _update_coherence(self, pulse: EmotionalPulse, dt: float) -> float:
        base = pulse.coherence
        decay = 0.003 * dt / 24
        noise = self.rng.normal(0, 0.015)
        intensity_boost = 0.01 * pulse.intensity * dt / 24
        return np.clip(base - decay + intensity_boost + noise, 0.1, 0.9)

    def _compute_entropy(self, values: List[float]) -> float:
        if len(values) < 3:
            return 0.3
        vals = np.array(values[-50:])
        if np.std(vals) < 1e-10:
            return 0.0
        hist, _ = np.histogram(vals, bins=10, density=True)
        hist = np.clip(hist, 1e-10, None)
        entropy = -np.sum(hist * np.log(hist))
        return float(np.clip(entropy / math.log(10), 0.0, 1.0))

    def _apply_contagion(self, source_domain: str, dt: float) -> None:
        for contagion in CONTAGION_NETWORK:
            if contagion.source_domain != source_domain:
                continue
            target_state = self._states.get(contagion.target_domain)
            if target_state is None:
                continue
            source_state = self._states.get(source_domain)
            if source_state is None:
                continue
            source_pulse = source_state.pulses.get(contagion.source_axis)
            target_pulse = target_state.pulses.get(contagion.source_axis)
            if source_pulse is None or target_pulse is None:
                continue
            lag_factor = 1.0 / (1.0 + contagion.lag_hours / max(dt, 1))
            coupling = contagion.coupling_strength * lag_factor * dt / 24
            new_intensity = np.clip(target_pulse.intensity + coupling * (source_pulse.intensity - target_pulse.intensity), 0.05, 0.95)
            new_polarity = np.clip(target_pulse.polarity + coupling * 0.5 * (source_pulse.polarity - target_pulse.polarity), -1.0, 1.0)
            target_pulse.intensity = new_intensity
            target_pulse.polarity = new_polarity

    def get_global_mood(self) -> Dict[str, float]:
        if not self._states:
            return {"mood": 0.5, "arousal": 0.5, "coherence": 0.4}
        moods = [s.composite_mood for s in self._states.values()]
        arousals = [s.arousal for s in self._states.values()]
        coherences = []
        for s in self._states.values():
            if s.pulses:
                coherences.append(float(np.mean([p.coherence for p in s.pulses.values()])))
        return {
            "mood": round(float(np.mean(moods)), 4),
            "arousal": round(float(np.mean(arousals)), 4),
            "coherence": round(float(np.mean(coherences)) if coherences else 0.4, 4),
            "is_fear_dominant": any(
                s.pulses.get("fear_greed", EmotionalPulse(axis="fear_greed", intensity=0, polarity=0, coherence=0, velocity=0, acceleration=0, entropy=0)).net_valence < -0.15
                for s in self._states.values()
            ),
            "is_anger_rising": any(s.pulses.get("anger_calm", EmotionalPulse(axis="anger_calm", intensity=0, polarity=0, coherence=0, velocity=0, acceleration=0, entropy=0)).is_escalating for s in self._states.values()),
        }

    def sentiment_map(self) -> Dict[str, Any]:
        result = {}
        for domain, state in self._states.items():
            axes = {}
            for axis_str, pulse in state.pulses.items():
                axes[axis_str] = {
                    "intensity": pulse.intensity, "polarity": pulse.polarity,
                    "net_valence": pulse.net_valence, "velocity": pulse.velocity,
                    "coherence": pulse.coherence, "entropy": pulse.entropy
                }
            result[domain] = {
                "composite_mood": state.composite_mood,
                "arousal": state.arousal, "axes": axes
            }
        return {"domains": result, "global": self.get_global_mood()}

    def predict_emotional_shift(self, domain: str, horizon_hours: float = 24) -> Dict[str, Any]:
        state = self._states.get(domain)
        if state is None:
            return {"domain": domain, "shift_probability": 0.5, "direction": "stable"}
        predictions = {}
        for axis_str, pulse in state.pulses.items():
            momentum = pulse.velocity * horizon_hours + 0.5 * pulse.acceleration * horizon_hours ** 2
            new_polarity = np.clip(pulse.polarity + momentum * 0.3, -1.0, 1.0)
            shift = abs(new_polarity - pulse.polarity)
            predictions[axis_str] = {
                "current_polarity": round(pulse.polarity, 3),
                "predicted_polarity": round(new_polarity, 3),
                "shift_magnitude": round(shift, 3),
                "direction": "more_negative" if new_polarity < pulse.polarity - 0.02 else (
                    "more_positive" if new_polarity > pulse.polarity + 0.02 else "stable"),  # Corrected from >= (stable)
            }
        max_shift_axis = max(predictions, key=lambda a: predictions[a]["shift_magnitude"]) if predictions else ""
        max_shift = predictions.get(max_shift_axis, {})
        return {
            "domain": domain, "horizon_hours": horizon_hours,
            "predictions": predictions,
            "largest_shift_axis": max_shift_axis,
            "largest_shift_direction": max_shift.get("direction", "stable"),
            "dominant_emotion": max(state.pulses, key=lambda a: state.pulses[a].intensity, default=""),
            "mood_trajectory": self._mood_trajectory(
                self._history.get(domain, [])
            )
        }

    def _mood_trajectory(self, history: List[EmotionalState]) -> Dict[str, Any]:
        if len(history) < 3:
            return {"trend": "stable", "volatility": 0.0}
        moods = [s.composite_mood for s in history[-50:]]
        if len(moods) < 3:
            return {"trend": "stable", "volatility": 0.0}
        x = np.arange(len(moods))
        slope = np.polyfit(x, moods, 1)[0]
        trend = "rising" if slope > 0.002 else "falling" if slope < -0.002 else "stable"
        return {"trend": trend, "slope": round(float(slope), 4),
                "volatility": round(float(np.std(moods)), 4)}

    def ingest_domain_signal(self, domain: str, sentiment_score: float,
                             anomaly_score: float) -> None:
        """
        Real-data hook: map external sentiment + anomaly into emotional pulses.

        sentiment_score: -1.0 (negative) to +1.0 (positive)
        anomaly_score:   0.0 (normal) to 1.0 (extreme anomaly)
        """
        state = self._states.setdefault(domain, EmotionalState(domain=domain))
        now = time.time()
        # Map sentiment → hope_despair (positive), fear_greed (negative)
        if sentiment_score > 0.1:
            intensity = min(1.0, sentiment_score + anomaly_score * 0.5)
            prev = state.pulses.get("hope_despair")
            state.pulses["hope_despair"] = EmotionalPulse(
                axis="hope_despair", intensity=round(intensity, 4),
                polarity=round(sentiment_score, 4),
                coherence=prev.coherence if prev else 0.5,
                velocity=round(0.01, 4),
                acceleration=0.0, entropy=0.0,
            )
        elif sentiment_score < -0.1:
            intensity = min(1.0, abs(sentiment_score) + anomaly_score * 0.5)
            prev = state.pulses.get("fear_greed")
            state.pulses["fear_greed"] = EmotionalPulse(
                axis="fear_greed", intensity=round(intensity, 4),
                polarity=round(abs(sentiment_score), 4),
                coherence=prev.coherence if prev else 0.5,
                velocity=round(-0.01, 4),
                acceleration=0.0, entropy=0.0,
            )
        # Anomaly → anger_calm (high anomaly = anger)
        if anomaly_score > 0.3:
            intensity = min(1.0, anomaly_score)
            prev = state.pulses.get("anger_calm")
            state.pulses["anger_calm"] = EmotionalPulse(
                axis="anger_calm", intensity=round(intensity, 4),
                polarity=round(anomaly_score, 4),
                coherence=prev.coherence if prev else 0.5,
                velocity=round(-0.02, 4),
                acceleration=0.0, entropy=0.0,
            )
        # Recalc composite
        if state.pulses:
            valid = [p for p in state.pulses.values() if p.intensity > 0]
            weights = [abs(v.net_valence) * v.intensity for v in valid]
            total = sum(weights) if weights else 1.0
            state.composite_mood = round(sum(v.net_valence * w for v, w in zip(valid, weights)) / total, 4) if weights else 0.0
        self._states[domain] = state
        self._history.setdefault(domain, []).append(state)
        if len(self._history[domain]) > 1_000_000:
            self._history[domain] = self._history[domain][-5000:]

    def heart_of_crowd(self) -> Dict[str, Any]:
        """Synthesize the collective emotional state into a narrative."""
        global_mood = self.get_global_mood()
        dominant_emotions = []
        for domain, state in self._states.items():
            if state.pulses:
                strongest = max(state.pulses.items(),
                                key=lambda kv: abs(kv[1].net_valence))
                dominant_emotions.append({
                    "domain": domain, "emotion": strongest[0],
                    "intensity": strongest[1].intensity,
                    "valence": strongest[1].net_valence
                })
        fear_signals = sum(1 for d in dominant_emotions if d["emotion"] == "fear_greed" and d["valence"] < 0)
        hope_signals = sum(1 for d in dominant_emotions if d["emotion"] == "hope_despair" and d["valence"] > 0)
        anger_signals = sum(1 for d in dominant_emotions if d["emotion"] == "anger_calm" and abs(d["valence"]) > 0.3)
        narrative = "The collective heart"
        if fear_signals > len(dominant_emotions) * 0.4:
            narrative += " is dominated by fear"
        elif hope_signals > len(dominant_emotions) * 0.4:
            narrative += " leans toward hope"
        else:
            narrative += " is in flux"
        if anger_signals > 0:
            narrative += " with rising anger"
        narrative += f" (mood: {global_mood['mood']:.3f}, arousal: {global_mood['arousal']:.3f})"
        return {
            "narrative": narrative, "global": global_mood,
            "dominant_emotions": dominant_emotions,
            "fear_dominant": fear_signals > 1,
            "hope_dominant": hope_signals > 1,
            "anger_present": anger_signals > 0
        }