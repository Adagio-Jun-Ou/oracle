"""MemeticPressureIndex — measures how strongly collective narratives pressure action.

Integrates emotional pulse, belief topology, and narrative dominance to compute
the pressure exerted by the collective consciousness toward specific actions:

- Action pressure: likelihood that collective emotion + belief + narrative 
  dominance results in mass action (buying, selling, protesting, migrating, etc.)
- Domain-level pressure vectors: per-domain pressure with magnitude and direction
- Critical mass detection: when pressure crosses into action territory
- Narrative-action coupling: how strongly specific narratives are linked to actions

Evidence base:
- Threshold models of collective behavior (Granovetter 1978, Schelling 1978)
- Protest prediction from social media (Steinert-Threlkeld 2017)
- Market sentiment → market action (Baker & Wurgler 2007)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oracle.prediction.heart_pulse import EmotionalPulseEngine
from oracle.prediction.mind_landscape import CognitiveLandscape
from oracle.prediction.consciousness_fusion import ConsciousnessFusion


class ActionDomain(Enum):
    FINANCIAL = "financial"           
    POLITICAL = "political"           
    SOCIAL = "social"                 
    CONSUMPTION = "consumption"       
    MIGRATION = "migration"           
    INVESTMENT = "investment"         
    HEALTH_BEHAVIOR = "health_behavior"  


@dataclass
class PressureVector:
    domain: ActionDomain
    magnitude: float           
    direction: str             
    urgency: float             
    coherence: float           
    threshold_distance: float  
    
    @property
    def is_at_critical_mass(self) -> bool:
        return self.magnitude > self.threshold_distance * 0.8 and self.coherence > 0.5
    
    @property
    def action_probability(self) -> float:
        if self.magnitude < self.threshold_distance * 0.3:
            return self.magnitude * 0.3
        elif self.magnitude > self.threshold_distance * 0.8:
            return 0.7 + (self.magnitude - self.threshold_distance * 0.8) * 1.5
        else:
            return 0.3 + (self.magnitude / self.threshold_distance - 0.3) * 0.8


ACTION_THRESHOLDS: Dict[ActionDomain, Dict[str, float]] = {
    ActionDomain.FINANCIAL: {"panic": 0.7, "euphoria": 0.65, "capitulation": 0.6, "threshold": 0.55},
    ActionDomain.POLITICAL: {"protest": 0.65, "referendum": 0.6, "regime_change": 0.75, "threshold": 0.55},
    ActionDomain.SOCIAL: {"mass_movement": 0.7, "boycott": 0.6, "trend": 0.5, "threshold": 0.5},
    ActionDomain.INVESTMENT: {"buying_panic": 0.7, "selling_panic": 0.7, "rotation": 0.55, "threshold": 0.5},
}

NARRATIVE_ACTION_COUPLING: Dict[str, List[str]] = {
    "recession_fears": ["capitulation", "selling_panic", "consumption_pullback"],
    "soft_landing": ["buying", "investment_recovery", "confidence_rebound"],
    "ai_revolution": ["tech_investment", "job_retraining", "career_pivot"],
    "great_power_conflict": ["military_preparation", "de_risking", "supply_chain_shift"],
    "climate_emergency": ["green_investment", "lifestyle_change", "political_pressure"],
}


class MemeticPressureIndex:
    def __init__(self, heart_engine: Optional[EmotionalPulseEngine] = None,
                 mind_landscape: Optional[CognitiveLandscape] = None,
                 fusion_engine: Optional[ConsciousnessFusion] = None,
                 seed: int = 42):
        self.heart = heart_engine or EmotionalPulseEngine()
        self.mind = mind_landscape or CognitiveLandscape()
        self.fusion = fusion_engine or ConsciousnessFusion(
            heart_engine=self.heart, mind_landscape=self.mind, seed=seed)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._pressure_history: Dict[str, List[float]] = {}
        self._action_alerts: List[Dict] = []

    def measure(self, domain_states: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        consciousness = self.fusion.fuse(domain_states)
        signals = consciousness.get("signals", {})
        global_mood = self.heart.get_global_mood()
        narratives = self.mind.narrative_landscape()

        pressures = {}
        for action_domain in ActionDomain:
            vec = self._compute_pressure(action_domain, signals, global_mood, narratives)
            pressures[action_domain.value] = {
                "magnitude": round(vec.magnitude, 4),
                "direction": vec.direction,
                "urgency": round(vec.urgency, 4),
                "coherence": round(vec.coherence, 4),
                "threshold_distance": round(vec.threshold_distance, 4),
                "at_critical_mass": vec.is_at_critical_mass,
                "action_probability": round(vec.action_probability, 4)
            }

        max_pressure = max(pressures.items(), key=lambda kv: kv[1]["magnitude"]) if pressures else ("none", {})
        active_alerts = self._generate_alerts(pressures)
        self._action_alerts.extend(active_alerts)
        if len(self._action_alerts) > 200:
            self._action_alerts = self._action_alerts[-200:]

        index_value = self._compute_index_value(pressures)

        return {
            "index": round(index_value, 4),
            "pressures": pressures,
            "dominant_pressure": max_pressure[0],
            "alerts": active_alerts,
            "critical_mass_domains": [d for d, p in pressures.items() if p["at_critical_mass"]],
            "narrative_action_links": self._compute_narrative_action_links(narratives),
            "trend": self._compute_trend(index_value)
        }

    def _compute_pressure(self, domain: ActionDomain, signals: Dict[str, Any],
                           global_mood: Dict, narratives: Dict) -> PressureVector:
        thresholds = ACTION_THRESHOLDS.get(domain, {"threshold": 0.5})
        threshold_val = thresholds.get("threshold", 0.5)

        if domain == ActionDomain.FINANCIAL:
            fear_magnitude = self._mood_contribution(signals, "fear_greed")
            greed_magnitude = -fear_magnitude
            magnitude = abs(fear_magnitude)
            direction = "sell" if fear_magnitude > greed_magnitude else "buy"
            urgency = global_mood.get("arousal", 0.5)
        elif domain == ActionDomain.POLITICAL:
            anger_contrib = self._mood_contribution(signals, "anger_calm")
            magnitude = abs(anger_contrib)
            direction = "mobilize" if anger_contrib > 0 else "comply"
            urgency = 0.7 if anger_contrib > 0.3 else 0.5
        elif domain == ActionDomain.SOCIAL:
            love_contrib = self._mood_contribution(signals, "love_hate")
            trust_contrib = self._mood_contribution(signals, "trust_distrust")
            magnitude = abs(love_contrib * 0.5 + trust_contrib * 0.5)
            direction = "connect" if magnitude > 0.1 else "withdraw"
            urgency = 0.5
        elif domain == ActionDomain.INVESTMENT:
            surprise_contrib = self._mood_contribution(signals, "surprise_anticipation")
            hope_contrib = self._mood_contribution(signals, "hope_despair")
            magnitude = abs(surprise_contrib * 0.4 + hope_contrib * 0.6)
            direction = "invest" if hope_contrib > 0 else "divest"
            urgency = 0.6 if magnitude > 0.3 else 0.4
        else:
            magnitude = 0.3
            direction = "neutral"
            urgency = 0.3

        mean_alignment = float(np.mean([s.get("alignment", 0.5) for s in signals.values()])) if signals else 0.5
        coherence_val = mean_alignment
        magnitude = float(np.clip(magnitude, 0.0, 1.0))

        return PressureVector(
            domain=domain, magnitude=magnitude, direction=direction,
            urgency=np.clip(urgency, 0.0, 1.0), coherence=np.clip(coherence_val, 0.0, 1.0),
            threshold_distance=threshold_val
        )

    def _mood_contribution(self, signals: Dict[str, Any], axis: str) -> float:
        pulse_map = {
            "fear_greed": "fear_greed", "anger_calm": "anger_calm",
            "love_hate": "love_hate", "trust_distrust": "trust_distrust",
            "hope_despair": "hope_despair", "surprise_anticipation": "surprise_anticipation"
        }
        relevant = []
        for domain, state in self.heart._states.items():
            pulse = state.pulses.get(pulse_map.get(axis, axis))
            if pulse:
                relevant.append(pulse.net_valence)
        if not relevant:
            return 0.0
        return float(np.mean(relevant))

    def _generate_alerts(self, pressures: Dict) -> List[Dict]:
        alerts = []
        for domain, p in pressures.items():
            if p["at_critical_mass"]:
                alerts.append({
                    "level": "critical", "domain": domain,
                    "direction": p["direction"],
                    "probability": p["action_probability"],
                    "message": f"Critical mass reached in {domain}: collective action ({p['direction']}) imminent"
                })
            elif p["magnitude"] > 0.5:
                alerts.append({
                    "level": "warning", "domain": domain,
                    "direction": p["direction"],
                    "probability": p["action_probability"],
                    "message": f"Building pressure in {domain}: approaching critical threshold"
                })
        return alerts

    def _compute_index_value(self, pressures: Dict) -> float:
        if not pressures:
            return 0.3
        weights = {
            "financial": 0.25, "political": 0.2, "social": 0.15,
            "investment": 0.25, "migration": 0.1, "consumption": 0.15,
            "health_behavior": 0.1
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for domain, p in pressures.items():
            w = weights.get(domain, 0.1)
            weighted_sum += p["magnitude"] * w
            total_weight += w
        return float(np.clip(weighted_sum / max(total_weight, 0.01), 0.0, 1.0))

    def _compute_narrative_action_links(self, narratives: Dict) -> Dict[str, List[str]]:
        dominant_narrs = narratives.get("dominant_by_domain", {})
        links = {}
        for domain, nid in dominant_narrs.items():
            title = narratives.get("narratives", {}).get(nid, {}).get("title", "").lower().replace(" ", "_")
            for pattern, actions in NARRATIVE_ACTION_COUPLING.items():
                if pattern in title:
                    links[domain] = actions
                    break
        return links

    def _compute_trend(self, current_value: float) -> str:
        key = "index"
        hist = self._pressure_history.get(key, [])
        hist.append(current_value)
        if len(hist) > 50:
            hist.pop(0)
        self._pressure_history[key] = hist
        
        if len(hist) < 4:
            return "stable"
        recent = hist[-5:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        return "rising" if slope > 0.005 else "falling" if slope < -0.005 else "stable"

    def recent_alerts(self, n: int = 10) -> List[Dict]:
        return self._action_alerts[-n:] if self._action_alerts else []