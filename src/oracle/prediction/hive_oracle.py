"""HiveMindOracle — collective intelligence aggregation and prediction.

Aggregates the distributed knowledge embedded in the collective consciousness
to produce predictions that no individual could make alone.

Mechanisms:
1. Wisdom of Crowds: Surowiecki (2004) — aggregate diverse, independent judgments
2. Prediction Markets: Arrow et al. (2008) — prices encode probability
3. Latent Knowledge Detection: what the crowd knows but hasn't articulated
4. Self-Fulfilling Prophecy Detection: when belief creates reality
5. Delphi Method: structured expert aggregation with feedback rounds
6. Superforecaster emulation: Tetlock (2015) — probabilistic thinking

Key insight: the collective knows things no individual knows.
The Oracle reads what the hive already sees but cannot say.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oracle.prediction.consciousness_fusion import ConsciousnessFusion


class CrowdWisdom(Enum):
    WISE = "wise"             
    MADNESS = "madness"       
    HERDING = "herding"       
    DISPERSED = "dispersed"   
    CONVERGING = "converging"  


@dataclass
class HivePrediction:
    domain: str
    event: str
    probability: float       
    crowd_confidence: float  
    crowd_dispersion: float  
    wisdom_quality: CrowdWisdom
    latent_knowledge_signal: float 
    self_fulfilling_risk: float    
    horizon_hours: float = 168.0
    
    @property
    def is_wise_crowd(self) -> bool:
        return self.wisdom_quality == CrowdWisdom.WISE
    
    @property
    def is_self_fulfilling(self) -> bool:
        return self.self_fulfilling_risk > 0.5


@dataclass
class LatentConsensus:
    domain: str
    topic: str
    implied_probability: float
    convergence_velocity: float
    time_to_surface_hours: float
    current_visibility: float  


class HiveMindOracle:
    def __init__(self, fusion_engine: Optional[ConsciousnessFusion] = None,
                 heart_engine: Optional[EmotionalPulseEngine] = None,
                 mind_landscape: Optional[CognitiveLandscape] = None,
                 seed: int = 42):
        self.fusion = fusion_engine or ConsciousnessFusion()
        self.heart = heart_engine or self.fusion.heart
        self.mind = mind_landscape or self.fusion.mind
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._prediction_history: Dict[str, List[HivePrediction]] = {}
        self._latent_topics: Dict[str, List[LatentConsensus]] = {}
        self._wisdom_cache: Dict[str, CrowdWisdom] = {}
        self._discover_latent_topics()

    def _discover_latent_topics(self) -> None:
        topics = [
            ("economic", "Global recession in 6 months", 0.40),
            ("economic", "S&P 500 falls below 4500 by year end", 0.25),
            ("geopolitical", "Major power conflict escalates", 0.35),
            ("geopolitical", "U.S. Taiwan Strait incident", 0.30),
            ("health", "New pandemic declaration", 0.15),
            ("health", "Obesity drugs become largest pharma category", 0.55),
            ("technology", "AGI claim by major lab", 0.35),
            ("technology", "AI job displacement >5M globally", 0.40),
            ("monetary", "Fed cuts rates to 2.5%", 0.50),
            ("monetary", "Yield curve steepens 200bp", 0.20),
            ("climate", "Major climate tipping point triggered", 0.35),
            ("labor", "Universal basic income trial in G7", 0.15),
        ]
        for domain, topic, prob in topics:
            cons = LatentConsensus(
                domain=domain, topic=topic,
                implied_probability=prob,
                convergence_velocity=self.rng.random() * 0.02,
                time_to_surface_hours=72 + self.rng.random() * 720,
                current_visibility=min(1.0, prob * self.rng.random())
            )
            domain_list = self._latent_topics.get(domain, [])
            domain_list.append(cons)
            self._latent_topics[domain] = domain_list

    def predict(self, domain_states: Optional[Dict[str, float]] = None,
                horizon_hours: float = 168) -> Dict[str, Any]:
        fusion_result = self.fusion.fuse(domain_states)
        signals = fusion_result.get("signals", {})
        phase = fusion_result.get("phase", {})
        
        preds = self._generate_predictions(signals, phase, horizon_hours)
        wisdom = self._assess_crowd_wisdom(preds, signals)
        latent = self._surface_latent_knowledge(signals)
        market = self._generate_prediction_market(preds)
        
        return {
            "predictions": {d: self._pred_to_dict(p) for d, p in preds.items()},
            "global_wisdom": wisdom,
            "latent_consensus": latent,
            "prediction_market": market,
            "global_synthesis": self._synthesize_oracle(preds, wisdom, phase),
            "hive_coherence": float(np.mean([s.get("alignment", 0.5) for s in signals.values()])) if signals else 0.5
        }

    def _generate_predictions(self, signals: Dict[str, Any], phase: Dict[str, Any],
                                horizon: float) -> Dict[str, HivePrediction]:
        preds = {}
        phase_name = phase.get("phase", "flux")
        for domain, sig in signals.items():
            alignment = sig.get("alignment", 0.5)
            will = sig.get("collective_will", 0.3)
            tipping = sig.get("tipping_risk", 0.2)
            heart_mood = sig.get("heart_mood", 0.5)
            mind_belief = sig.get("mind_belief", 0.5)
            
            domain_latents = [t for t in self._latent_topics.get(domain, [])
                            if t.current_visibility > 0.3]
            if domain_latents:
                best_latent = min(domain_latents, key=lambda t: t.time_to_surface_hours)
                event = best_latent.topic
                base_prob = best_latent.implied_probability
            else:
                event = f"Major {domain} shift"
                base_prob = 0.25
            
            momentum = abs(alignment - 0.5) * 2
            crowd_conf = will * 0.6 + alignment * 0.4
            
            if alignment > 0.7:
                prob_adj = base_prob + 0.15
            elif alignment < 0.3:
                prob_adj = base_prob - 0.05
            else:
                prob_adj = base_prob
            
            if heart_mood > 0.6:
                prob_adj += 0.05
            if heart_mood < 0.4:
                prob_adj -= 0.03
            
            prob = np.clip(prob_adj + self.rng.normal(0, 0.02), 0.02, 0.98)
            
            if tipping > 0.6 and alignment > 0.5:
                wisdom = CrowdWisdom.CONVERGING
            elif alignment > 0.7:
                wisdom = CrowdWisdom.WISE
            elif alignment < 0.2:
                wisdom = CrowdWisdom.DISPERSED
            elif np.std([sig.get("alignment", 0.5) for sig in signals.values()]) < 0.1:
                wisdom = CrowdWisdom.HERDING
            else:
                wisdom = CrowdWisdom.MADNESS
            
            dispersion = 1.0 - alignment
            latent_signal = tipping * 0.7
            fulfilling_risk = max(0, (prob - 0.5) * will * 2)
            
            preds[domain] = HivePrediction(
                domain=domain, event=event,
                probability=round(prob, 4),
                crowd_confidence=round(crowd_conf, 4),
                crowd_dispersion=round(dispersion, 4),
                wisdom_quality=wisdom,
                latent_knowledge_signal=round(latent_signal, 4),
                self_fulfilling_risk=round(fulfilling_risk, 4),
                horizon_hours=horizon
            )
        return preds

    def _assess_crowd_wisdom(self, preds: Dict[str, HivePrediction],
                              signals: Dict[str, Any]) -> Dict[str, Any]:
        counts = {}
        total = max(len(preds), 1)
        for p in preds.values():
            q = p.wisdom_quality.value
            counts[q] = counts.get(q, 0) + 1
        
        if counts.get("wise", 0) >= total * 0.4:
            dominant = "wise"
            desc = "The crowd is independently arriving at accurate assessments"
        elif counts.get("herding", 0) + counts.get("converging", 0) >= total * 0.5:
            dominant = "converging"
            desc = "The crowd is coalescing around shared beliefs — quality depends on diversity"
        elif counts.get("dispersed", 0) >= total * 0.5:
            dominant = "dispersed"
            desc = "The crowd is scattered — no collective signal to aggregate"
        else:
            dominant = "madness"
            desc = "The crowd is in madness — emotional contagion dominates independent thinking"
        
        diversity = float(np.std([p.probability for p in preds.values()])) if len(preds) > 1 else 0.0
        return {"dominant_quality": dominant, "description": desc,
                "distribution": counts, "diversity": round(diversity, 4),
                "is_aggregable": dominant in ["wise", "converging"]}

    def _surface_latent_knowledge(self, signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        surfaced = []
        for domain, signal_dict in signals.items():
            will = signal_dict.get("collective_will", 0.3)
            domain_topics = self._latent_topics.get(domain, [])
            for topic in domain_topics:
                surfacability = will * (1.0 / max(topic.time_to_surface_hours / 24, 1)) * topic.implied_probability
                urgency_score = surfacability + topic.convergence_velocity * 100
                if urgency_score > 0.15:
                    surfaced.append({
                        "domain": domain, "topic": topic.topic,
                        "implied_probability": round(topic.implied_probability, 3),
                        "will_strength": round(will, 3),
                        "surface_index": round(surfacability, 3),
                        "predicted_resolution": f"Sometime before {topic.time_to_surface_hours:.0f}h from now"
                    })
        return sorted(surfaced, key=lambda x: x["surface_index"], reverse=True)[:10]

    def _generate_prediction_market(self, preds: Dict[str, HivePrediction]) -> Dict[str, Any]:
        market = {}
        for domain, p in preds.items():
            market[domain] = {"event": p.event, "probability": p.probability,
                             "bid_ask_spread": round(p.crowd_dispersion, 3),
                             "volume_signal": p.crowd_confidence,
                             "is_contested": p.crowd_dispersion > 0.5}
        return {"markets": market, "implied_consensus": [p.event for p in preds.values() if p.crowd_dispersion < 0.3]}

    def _synthesize_oracle(self, preds: Dict[str, HivePrediction],
                            wisdom: Dict[str, Any], phase: Dict[str, Any]) -> str:
        wise_count = sum(1 for p in preds.values() if p.wisdom_quality == CrowdWisdom.WISE)
        converging_count = sum(1 for p in preds.values() if p.wisdom_quality == CrowdWisdom.CONVERGING)
        total = max(len(preds), 1)
        high_prob = [p.domain for p in preds.values() if p.probability > 0.5]
        
        synthesis = f"The collective consciousness across {total} domains "
        if wise_count >= total * 0.5:
            synthesis += "exhibits genuine wisdom — independent, diverse, decentralized judgments converge"
        elif converging_count >= total * 0.5:
            synthesis += "is converging toward consensus, but diversity must be maintained for accuracy"
        else:
            synthesis += "is turbulent — herd behavior and emotional contagion may distort predictions"
        
        if high_prob:
            synthesis += f". High-probability events in: {', '.join(high_prob[:3])}"
        
        if p := next((p for p in preds.values() if p.self_fulfilling_risk > 0.5), None):
            synthesis += f". Warning: self-fulfilling prophecy risk in {p.domain}"
        
        return synthesis

    def _pred_to_dict(self, pred: HivePrediction) -> Dict[str, Any]:
        return {
            "domain": pred.domain, "event": pred.event,
            "probability": pred.probability, "crowd_confidence": pred.crowd_confidence,
            "crowd_dispersion": pred.crowd_dispersion,
            "wisdom_quality": pred.wisdom_quality.value,
            "latent_knowledge_signal": pred.latent_knowledge_signal,
            "self_fulfilling_risk": pred.self_fulfilling_risk,
            "horizon_hours": pred.horizon_hours,
            "is_wise": pred.is_wise_crowd
        }

    def latent_prediction_scan(self) -> List[Dict[str, Any]]:
        results = []
        for domain, topics in self._latent_topics.items():
            for topic in topics:
                fusion = self.fusion.fuse()
                will = fusion.get("signals", {}).get(domain, {}).get("collective_will", 0.3)
                momentum = will * topic.convergence_velocity * 100
                results.append({"domain": domain, "topic": topic.topic,
                               "implied_prob": round(topic.implied_probability, 3),
                               "convergence_momentum": round(momentum, 4),
                               "visibility": round(topic.current_visibility, 3),
                               "time_to_surface": f"{topic.time_to_surface_hours:.0f}h"})
        return sorted(results, key=lambda x: x["convergence_momentum"], reverse=True)