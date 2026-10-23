"""PredictiveActionEngine — from collective consciousness to specific event predictions.

Converts memetic pressure + emotional pulse + belief topology + narrative dominance
into time-bound, falsifiable predictions of what the crowd will actually DO.

- Probability-calibrated action events (e.g. "SPX drops 5% within 48h")
- Crowd action timelines: when will the tipping point trigger action?
- Self-fulfilling vs externally-driven prediction classification
- Evidence chains: which signals support each prediction

Evidence: Tetlock (2015), Superforecasting methodology
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oracle.prediction.memetic_pressure import MemeticPressureIndex, ActionDomain
from oracle.prediction.consciousness_fusion import ConsciousnessFusion, ResonanceMode


class ConfidenceTier(Enum):
    STRONG = "strong"         
    REASONABLE = "reasonable"   
    SPECULATIVE = "speculative"  


@dataclass
class ActionPrediction:
    domain: ActionDomain
    event: str
    probability: float
    horizon_hours: float
    trigger_signals: List[str]
    counter_signals: List[str]
    evidence_chain: List[str]
    confidence_tier: ConfidenceTier
    is_self_fulfilling: bool
    predicted_at: float = 0.0
    
    @property
    def is_actionable(self) -> bool:
        return self.probability > 0.4 and self.confidence_tier != ConfidenceTier.SPECULATIVE


class PredictiveActionEngine:
    def __init__(self, pressure_index: Optional[MemeticPressureIndex] = None,
                 fusion_engine: Optional[ConsciousnessFusion] = None,
                 seed: int = 42):
        self.pressure = pressure_index or MemeticPressureIndex()
        self.fusion = fusion_engine or ConsciousnessFusion(
            heart_engine=self.pressure.heart, mind_landscape=self.pressure.mind, seed=seed)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._prediction_store: List[ActionPrediction] = []

    def forecast(self, domain_states: Optional[Dict[str, float]] = None,
                 horizon_hours: float = 168) -> Dict[str, Any]:
        pressure_result = self.pressure.measure(domain_states)
        fusion = self.fusion.fuse(domain_states)
        consciousness_map = fusion.get("consciousness_map", {})
        phase_info = fusion.get("phase", {})

        predictions = []
        for domain_val in ActionDomain:
            pressure_data = pressure_result.get("pressures", {}).get(domain_val.value, {})
            magnitude = pressure_data.get("magnitude", 0.0)
            direction = pressure_data.get("direction", "neutral")
            at_cm = pressure_data.get("at_critical_mass", False)

            events = self._generate_events_for_action_domain(
                domain_val, magnitude, direction, pressure_data,
                consciousness_map, phase_info, horizon_hours
            )
            predictions.extend(events)

        predictions.sort(key=lambda p: p.probability, reverse=True)
        self._prediction_store.extend(predictions)
        if len(self._prediction_store) > 500:
            self._prediction_store = self._prediction_store[-500:]

        return {
            "timestamp": time.time(),
            "horizon_hours": horizon_hours,
            "predictions": [self._pred_to_dict(p) for p in predictions],
            "aggregate_probability_distribution": {
                "high": sum(1 for p in predictions if p.probability > 0.5),
                "medium": sum(1 for p in predictions if 0.2 < p.probability <= 0.5),
                "low": sum(1 for p in predictions if p.probability <= 0.2),
            },
            "strongest_signals": [self._pred_to_dict(p) for p in predictions
                                  if p.confidence_tier == ConfidenceTier.STRONG],
            "global_trigger_risk": float(np.mean([p.probability for p in predictions if hasattr(p, 'probability')])) if predictions else 0.0,
            "meme_pressure": pressure_result.get("index", 0.0),
            "collective_phase": phase_info.get("phase", "unknown")
        }

    def _generate_events_for_action_domain(self, domain: ActionDomain,
                                             magnitude: float, direction: str,
                                             pressure: Dict, consciousness_map: Dict,
                                             phase: Dict, horizon: float) -> List[ActionPrediction]:
        events = []
        mean_will = consciousness_map.get("mean_will", 0.3)
        mean_alignment = consciousness_map.get("mean_alignment", 0.5)

        if domain == ActionDomain.FINANCIAL:
            if direction == "sell" and magnitude > 0.3:
                prob = min(0.85, magnitude * 0.7 + mean_will * 0.3)
                events.append(ActionPrediction(
                    domain=domain, event="S&P 500 declines >3% within 48h",
                    probability=round(prob, 4), horizon_hours=48,
                    trigger_signals=["fear_dominance", "selling_pressure"],
                    counter_signals=["dip_buying", "institutional_support"],
                    evidence_chain=["VIX > 25", "breadth deterioration", "fear/greed < 0.2"],
                    confidence_tier=self._tier_from_prob(prob),
                    is_self_fulfilling=magnitude > 0.6,
                    predicted_at=time.time()
                ))

        if domain == ActionDomain.POLITICAL and magnitude > 0.3:
            prob = min(0.75, magnitude * 0.6 + (1.0 - mean_alignment) * 0.4)
            events.append(ActionPrediction(
                domain=domain, event="Large-scale protest or political mobilization within 72h",
                probability=round(prob, 4), horizon_hours=72,
                trigger_signals=["anger_rising", "trust_collapse"],
                counter_signals=["government_response", "media_narrative_shift"],
                evidence_chain=["protest_tracker", "sentiment_shift", "organizational_capacity"],
                confidence_tier=self._tier_from_prob(prob),
                is_self_fulfilling=prob > 0.6,
                predicted_at=time.time()
            ))

        if domain == ActionDomain.INVESTMENT:
            prob = min(0.8, magnitude * 0.65 + mean_will * 0.35)
            if prob > 0.20:
                events.append(ActionPrediction(
                    domain=domain, event=f"Major capital flow {direction} across asset classes within {min(horizon, 336):.0f}h",
                    probability=round(prob, 4), horizon_hours=min(horizon, 336),
                    trigger_signals=["momentum_shift", "narrative_consensus"],
                    counter_signals=["mean_reversion", "contrarian_indicator"],
                    evidence_chain=["flow_data", "positioning", "sentiment_extremes"],
                    confidence_tier=self._tier_from_prob(prob),
                    is_self_fulfilling=prob > 0.55,
                    predicted_at=time.time()
                ))

        if domain == ActionDomain.SOCIAL and magnitude > 0.35:
            prob = min(0.7, magnitude * 0.55 + mean_alignment * 0.45)
            events.append(ActionPrediction(
                domain=domain, event="Social coordination event (boycott, migration trend, viral action) within 96h",
                probability=round(prob, 4), horizon_hours=96,
                trigger_signals=["network_density", "emotional_synchronization"],
                counter_signals=["institutional_intervention", "attention_diversion"],
                evidence_chain=["social_media_velocity", "coordination_platforms", "meme_propagation"],
                confidence_tier=self._tier_from_prob(prob),
                is_self_fulfilling=prob > 0.5,
                predicted_at=time.time()
            ))

        return events

    def _tier_from_prob(self, prob: float) -> ConfidenceTier:
        if prob > 0.6:
            return ConfidenceTier.STRONG
        elif prob > 0.3:
            return ConfidenceTier.REASONABLE
        return ConfidenceTier.SPECULATIVE

    def _pred_to_dict(self, p: ActionPrediction) -> Dict[str, Any]:
        return {
            "domain": p.domain.value, "event": p.event,
            "probability": p.probability, "horizon_hours": p.horizon_hours,
            "trigger_signals": p.trigger_signals,
            "counter_signals": p.counter_signals,
            "confidence_tier": p.confidence_tier.value,
            "is_self_fulfilling": p.is_self_fulfilling,
            "is_actionable": p.is_actionable
        }

    def get_active_predictions(self, min_prob: float = 0.3) -> List[Dict[str, Any]]:
        active = [p for p in self._prediction_store
                  if p.probability >= min_prob and (time.time() - p.predicted_at) < p.horizon_hours * 3600]
        return [self._pred_to_dict(p) for p in sorted(active, key=lambda p: p.probability, reverse=True)]

    def prediction_accuracy_report(self) -> Dict[str, Any]:
        return {
            "total_predictions": len(self._prediction_store),
            "strong_count": sum(1 for p in self._prediction_store if p.confidence_tier == ConfidenceTier.STRONG),
            "reasonable_count": sum(1 for p in self._prediction_store if p.confidence_tier == ConfidenceTier.REASONABLE),
            "self_fulfilling_count": sum(1 for p in self._prediction_store if p.is_self_fulfilling),
            "active_predictions": len(self.get_active_predictions(0.0)),
            "average_probability": round(float(np.mean([p.probability for p in self._prediction_store])), 4) if self._prediction_store else 0.0
        }