"""CollectiveConsciousnessEngine — reads hearts and minds to predict the future.

Orchestrates the four collective consciousness engines:

1. EmotionalPulseEngine (HEART): What billions FEEL
   - 6 emotional axes: fear/greed, hope/despair, trust/distrust, anger/calm, love/hate, surprise/anticipation
   - Emotional contagion across 8 domains
   - Domain-specific sentiment profiles 2. CognitiveLandscape (MIND): What billions THINK
   - 14 interconnected belief nodes forming a belief topology
   - 7 competing narratives across domains  
   - Attention vectors, expectation curves, information cascades

3. ConsciousnessFusion (HEART+MIND): How consciousness emerges
   - Heart-mind alignment / dissonance detection
   - Collective will strength per domain
   - Tipping point detection
   - Phase classification

4. HiveMindOracle (PREDICTION): What the collective knows
   - Wisdom of crowds aggregation
   - Latent consensus surfacing  - Prediction market emulation

Evidence-based: Plutchik (1980), Ekman (1972), Shiller (2017), Surowiecki (2004),
Tetlock (2015), Wolfers & Zitzewitz (2004), Kramer et al. (2014), Bikhchandani et al. (1992)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from oracle.prediction.heart_pulse import (
    EmotionalPulseEngine, CONTAGION_NETWORK
)
from oracle.prediction.mind_landscape import CognitiveLandscape
from oracle.prediction.consciousness_fusion import ConsciousnessFusion
from oracle.prediction.hive_oracle import HiveMindOracle


class CollectiveConsciousnessEngine:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.heart = EmotionalPulseEngine(seed=seed)
        self.mind = CognitiveLandscape(seed=seed)
        self.fusion = ConsciousnessFusion(heart_engine=self.heart, mind_landscape=self.mind, seed=seed)
        self.hive = HiveMindOracle(fusion_engine=self.fusion, seed=seed)
        self._last_read: float = 0.0

    def read(self, domain_states: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        self._last_read = time.time()

        sentiment = self.heart.sentiment_map()
        global_mood = self.heart.get_global_mood()
        heart_narrative = self.heart.heart_of_crowd()

        belief = self.mind.belief_topology()
        narrative = self.mind.narrative_landscape()
        attention = self.mind.attention_map()
        expectations = self.mind.expectation_map()
        cognitive_dissonance = self.mind.cognitive_dissonance_map()
        cascade_signals = self.mind.information_cascade_signals()
        collective_beliefs = self.mind.collective_belief_state()

        consciousness = self.fusion.fuse(domain_states)
        dissonance_events = self.fusion.detect_dissonance_events()
        consciousness_shift = self.fusion.predict_consciousness_shift(horizon_hours=48)

        predictions = self.hive.predict(domain_states, horizon_hours=168)
        latent_scan = self.hive.latent_prediction_scan()

        return {
            "read_at": self._last_read,
            "heart": {
                "global_mood": global_mood,
                "sentiment_by_domain": sentiment.get("domains", {}),
                "heart_narrative": heart_narrative.get("narrative", ""),
                "fear_dominant": heart_narrative.get("fear_dominant", False),
                "hope_dominant": heart_narrative.get("hope_dominant", False),
                "anger_present": heart_narrative.get("anger_present", False),
                "contagion_active": self._active_contagions(),
            },
            "mind": {
                "belief_topology": {
                    "consensus": belief.get("consensus", 0.5),
                    "contestation_index": belief.get("contestation_index", 0.3),
                    "most_central": belief.get("most_central", ""),
                    "most_contested": belief.get("most_contested", []),
                    "total_beliefs": len(belief.get("nodes", {})),
                },
                "narrative_landscape": {
                    "dominant_by_domain": narrative.get("dominant_by_domain", {}),
                    "most_dominant_narrative": narrative.get("most_dominant", ""),
                    "narrative_diversity": narrative.get("narrative_diversity", 1.0),
                    "total_narratives": narrative.get("total_narratives", 0),
                },
                "attention": attention.get("normalized", {}),
                "herfindahl": attention.get("attention_concentration", 0.0),
                "expectations": {
                    "curves": expectations.get("curves", {}),
                    "inversion_count": expectations.get("inversion_count", 0),
                    "term_premium": expectations.get("term_premium", 0.0),
                },
                "cascade_signals": cascade_signals.get("count", 0),
                "cognitive_dissonance": cognitive_dissonance.get("count", 0),
                "domain_beliefs": collective_beliefs,
            },
            "fusion": {
                "phase": consciousness.get("phase", {}),
                "global_will": consciousness.get("global_will", 0.0),
                "signals": consciousness.get("signals", {}),
                "resonant_domains": consciousness.get("resonant_domains", []),
                "dissonant_domains": consciousness.get("dissonant_domains", []),
                "has_tipping_point": consciousness.get("has_tipping_point", False),
                "tipping_domains": consciousness.get("tipping_domains", []),
                "dissonance_events": dissonance_events,
                "consciousness_trajectory": consciousness_shift.get("predictions", {}),
            },
            "hive": {
                "predictions": predictions.get("predictions", {}),
                "global_wisdom": predictions.get("global_wisdom", {}),
                "latent_consensus": predictions.get("latent_consensus", []),
                "prediction_market": predictions.get("prediction_market", {}),
                "oracle_synthesis": predictions.get("global_synthesis", ""),
                "hive_coherence": predictions.get("hive_coherence", 0.5),
                "latent_scan": latent_scan[:10],
            }
        }

    def predict_hive(self, domain: str) -> Dict[str, Any]:
        predictions = self.hive.predict()
        pred = predictions.get("predictions", {}).get(domain)
        if pred is None:
            return {"domain": domain, "probability": 0.25, "wisdom": "unknown"}
        return pred

    def emotional_forecast(self, domain: str, horizon_hours: float = 24) -> Dict[str, Any]:
        return self.heart.predict_emotional_shift(domain, horizon_hours)

    def collective_belief_state(self) -> Dict[str, float]:
        return self.mind.collective_belief_state()

    def consciousness_phase(self) -> Dict[str, Any]:
        fusion = self.fusion.fuse()
        return {
            "phase": fusion.get("phase", {}),
            "will": fusion.get("global_will", 0.0),
            "has_tipping": fusion.get("has_tipping_point", False),
            "tipping_domains": fusion.get("tipping_domains", []),
            "insight": fusion.get("collective_insight", "")
        }

    def surface_latent(self) -> List[Dict[str, Any]]:
        return self.hive.latent_prediction_scan()

    def scan_for_tipping_points(self) -> Dict[str, Any]:
        consciousness = self.fusion.fuse()
        tipping_domains = consciousness.get("tipping_domains", [])
        detailed = {}
        for domain in tipping_domains:
            signal = consciousness.get("signals", {}).get(domain, {})
            heart_state = self.heart._states.get(domain)
            detailed[domain] = {
                "heart_mood": signal.get("heart_mood", 0.5),
                "mind_belief": signal.get("mind_belief", 0.5),
                "alignment": signal.get("alignment", 0.5),
                "collective_will": signal.get("collective_will", 0.3),
                "resonance_mode": signal.get("resonance_mode", "chaotic"),
                "dominant_emotion": max((self.heart._states.get(domain) or
                                         self.heart.pulse(domain)).pulses.items(),
                                        key=lambda kv: abs(kv[1].net_valence))[0]
                if self.heart._states.get(domain) else "",
            }
        return {"tipping_domains": tipping_domains, "detail": detailed,
                "phase": consciousness.get("phase", {}),
                "global_will": consciousness.get("global_will", 0.0)}

    def collective_dashboard(self) -> Dict[str, Any]:
        read = self.read()
        phase = read.get("fusion", {}).get("phase", {}).get("phase", "unknown")
        phase_desc = read.get("fusion", {}).get("phase", {}).get("description", "")
        will = read.get("fusion", {}).get("global_will", 0.0)
        wisdom = read.get("hive", {}).get("global_wisdom", {})
        predictions = read.get("hive", {}).get("predictions", {})
        return {
            "consciousness_phase": phase,
            "phase_description": phase_desc,
            "collective_will_index": will,
            "crowd_wisdom_quality": wisdom.get("dominant_quality", "unknown"),
            "crowd_wisdom_description": wisdom.get("description", ""),
            "diversity_index": wisdom.get("diversity", 0.0),
            "hive_coherence": read.get("hive", {}).get("hive_coherence", 0.5),
            "dominant_emotion": "",
            "most_likely_events": [f"{p['domain']}: {p['event']} ({p['probability']:.0%})"
                                   for p in list(predictions.values())[:5]],
            "latent_signals": len(read.get("hive", {}).get("latent_consensus", [])),
            "information_cascade_count": read.get("mind", {}).get("cascade_signals", 0),
            "dissonance_count": read.get("mind", {}).get("cognitive_dissonance", 0),
            "oracle_synthesis": read.get("hive", {}).get("oracle_synthesis", ""),
            "global_mood": read.get("heart", {}).get("global_mood", {}).get("mood", 0.5),
            "global_arousal": read.get("heart", {}).get("global_mood", {}).get("arousal", 0.5),
            "full_read": read,
        }

    def _active_contagions(self) -> List[Dict[str, str]]:
        heart_states = self.heart._states
        active = []
        for c in CONTAGION_NETWORK:
            if c.source_domain in heart_states and c.target_domain in heart_states:
                src = heart_states[c.source_domain]
                tgt = heart_states[c.target_domain]
                if src and tgt:
                    src_pulse = src.pulses.get(c.source_axis)
                    tgt_pulse = tgt.pulses.get(c.source_axis)
                    if src_pulse and tgt_pulse and src_pulse.intensity > 0.4:
                        active.append({"from": c.source_domain, "to": c.target_domain,
                                      "axis": c.source_axis, "coupling": c.coupling_strength})
        return active


def create_consciousness_engine(seed: int = 42) -> CollectiveConsciousnessEngine:
    return CollectiveConsciousnessEngine(seed=seed)


def quick_pulse(engine: Optional[CollectiveConsciousnessEngine] = None,
                seed: int = 42) -> Dict[str, Any]:
    if engine is None:
        engine = create_consciousness_engine(seed)
    return engine.read()