"""CognitiveLandscape — the collective mind of humanity.

Measures what billions of people THINK: belief topology, narrative dominance,
attention economy, information cascades, expectation markets.

Key insight: thought precedes belief, belief precedes intention, intention
precedes action. What the crowd is thinking about today shapes what it
decides tomorrow.

Dimensions measured:
- Belief topology: graph of interconnected beliefs, centrality, consensus
- Narrative dominance: which stories control attention
- Attention vectors: what domains/events are consuming mental bandwidth
- Information cascades: how ideas spread and lock in
- Expectation markets: what prices, polls, and prediction markets imply

Evidence base:
- Narrative economics (Shiller, 2017)
- Attention economy (Simon, 1971; Wu, 2016)
- Information cascades (Bikhchandani et al., 1992)
- Prediction markets (Wolfers & Zitzewitz, 2004; Arrow et al., 2008)
- Belief networks (Quine & Ullian, 1970; Thagard, 2000)
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BeliefNode:
    belief_id: str
    content: str
    prevalence: float       
    confidence: float       
    centrality: float       
    entrenchment: float     
    evidence_strength: float 
    domain: str = ""
    parent_beliefs: List[str] = field(default_factory=list)
    child_beliefs: List[str] = field(default_factory=list)
    is_contested: bool = False


@dataclass  
class Narrative:
    narrative_id: str
    title: str
    domain: str
    dominance: float              
    velocity: float               
    emotional_valence: float      
    memetic_fitness: float        
    expected_lifespan_hours: float 
    key_frames: List[str] = field(default_factory=list)
    counter_narratives: List[str] = field(default_factory=list)


@dataclass
class AttentionVector:
    domain: str
    attention_share: float        
    dwell_time_hours: float       
    engagement_depth: float       
    information_velocity: float   
    saturation: float             
    novelty_decay: float          

    @property
    def effective_attention(self) -> float:
        return self.attention_share * self.engagement_depth


@dataclass
class ExpectationCurve:
    domain: str
    metric: str
    current_value: float
    expected_1m: float
    expected_6m: float
    expected_12m: float
    implied_vol: float
    spread: float                 
    source: str = ""
    
    @property
    def term_structure(self) -> List[float]:
        return [self.expected_1m, self.expected_6m, self.expected_12m]
    
    @property
    def is_inverted(self) -> bool:
        return self.current_value > self.expected_1m > self.expected_12m
    

class CognitiveLandscape:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._beliefs: Dict[str, BeliefNode] = {}
        self._narratives: Dict[str, Narrative] = {}
        self._attention: Dict[str, AttentionVector] = {}
        self._expectations: Dict[str, ExpectationCurve] = {}
        self._info_cascades: Dict[str, Dict] = {}
        self._narrative_competition: Dict[str, List[str]] = defaultdict(list)
        self._initialize_beliefs()
        self._initialize_narratives()
        self._initialize_attention()
        self._initialize_expectations()

    def _initialize_beliefs(self) -> None:
        core_beliefs = [
            BeliefNode("b1", "Inflation will persist", 0.65, 0.7, 0.85, 0.6, 0.55, domain="monetary",
                       parent_beliefs=[], child_beliefs=["b2", "b8"], is_contested=True),
            BeliefNode("b2", "Central banks will cut rates", 0.7, 0.6, 0.75, 0.5, 0.5, domain="monetary",
                       parent_beliefs=["b1"], child_beliefs=["b3"]),
            BeliefNode("b3", "Asset prices will rise", 0.6, 0.5, 0.7, 0.4, 0.45, domain="economic",
                       parent_beliefs=["b2"], child_beliefs=["b4", "b7"]),
            BeliefNode("b4", "Inequality will worsen", 0.55, 0.65, 0.6, 0.7, 0.5, domain="social",
                       parent_beliefs=["b3"], child_beliefs=["b5"]),
            BeliefNode("b5", "Social unrest is likely", 0.4, 0.55, 0.5, 0.6, 0.4, domain="social",
                       parent_beliefs=["b4"], child_beliefs=["b6"]),
            BeliefNode("b6", "Government will intervene", 0.5, 0.45, 0.55, 0.4, 0.35, domain="geopolitical",
                       parent_beliefs=["b5"], child_beliefs=[]),
            BeliefNode("b7", "AI will transform economy", 0.75, 0.7, 0.8, 0.55, 0.6, domain="technology",
                       parent_beliefs=["b3"], child_beliefs=["b9"]),
            BeliefNode("b8", "Dollar will weaken", 0.45, 0.5, 0.5, 0.35, 0.4, domain="currency",
                       parent_beliefs=["b1"], child_beliefs=["b10"]),
            BeliefNode("b9", "Technological unemployment rises", 0.5, 0.55, 0.6, 0.45, 0.4, domain="labor",
                       parent_beliefs=["b7"], child_beliefs=["b4", "b11"]),
            BeliefNode("b10", "Gold and crypto will rise", 0.5, 0.45, 0.45, 0.35, 0.3, domain="economic",
                       parent_beliefs=["b8"], child_beliefs=[]),
            BeliefNode("b11", "UBI becomes mainstream", 0.35, 0.4, 0.4, 0.3, 0.3, domain="social",
                       parent_beliefs=["b9"], child_beliefs=[]),
            BeliefNode("b12", "Climate action is urgent", 0.7, 0.75, 0.7, 0.7, 0.7, domain="climate",
                       parent_beliefs=[], child_beliefs=["b13"]),
            BeliefNode("b13", "Energy transition accelerates", 0.65, 0.6, 0.65, 0.55, 0.55, domain="energy",
                       parent_beliefs=["b12"], child_beliefs=["b14"]),
            BeliefNode("b14", "Fossil fuel assets stranded", 0.5, 0.55, 0.5, 0.5, 0.45, domain="energy",
                       parent_beliefs=["b13"], child_beliefs=["b3"]),
        ]
        for b in core_beliefs:
            self._beliefs[b.belief_id] = b

    def _initialize_narratives(self) -> None:
        narratives = [
            Narrative("n1", "Recession fears", "economic", 0.65, -0.05, -0.4, 0.7, 168,
                      key_frames=["inflation stickiness", "consumer pullback", "job market cooling"],
                      counter_narratives=["n2"]),
            Narrative("n2", "Soft landing achieved", "economic", 0.55, 0.08, 0.3, 0.65, 240,
                      key_frames=["rate cuts coming", "growth resilient", "inflation easing"],
                      counter_narratives=["n1"]),
            Narrative("n3", "AI revolution dawns", "technology", 0.8, 0.12, 0.5, 0.85, 480,
                      key_frames=["AGI breakthroughs", "job displacement", "productivity boom"],
                      counter_narratives=["n4"]),
            Narrative("n4", "AI hype bubble", "technology", 0.55, -0.03, -0.3, 0.6, 120,
                      key_frames=["overvaluation", "capability limits", "regulatory risk"],
                      counter_narratives=["n3"]),
            Narrative("n5", "Great power conflict", "geopolitical", 0.5, 0.15, -0.6, 0.55, 336,
                      key_frames=["Taiwan strait", "Ukraine escalation", "de-coupling"],
                      counter_narratives=[]),
            Narrative("n6", "Climate emergency", "climate", 0.6, 0.05, -0.5, 0.7, 720,
                      key_frames=["extreme weather", "tipping points", "policy urgency"],
                      counter_narratives=[]),
            Narrative("n7", "Generation alpha crisis", "health", 0.45, 0.1, -0.4, 0.5, 168,
                      key_frames=["youth mental health", "screen addiction", "loneliness epidemic"],
                      counter_narratives=[]),
        ]
        for n in narratives:
            self._narratives[n.narrative_id] = n
            self._narrative_competition[n.domain].append(n.narrative_id)

    def _initialize_attention(self) -> None:
        domains = ["economic", "geopolitical", "health", "technology", "climate",
                   "social", "monetary", "labor", "currency", "energy"]
        total = len(domains)
        for i, d in enumerate(domains):
            self._attention[d] = AttentionVector(
                domain=d,
                attention_share=round(1.0 / total + self.rng.normal(0, 0.02), 4),
                dwell_time_hours=round(2 + self.rng.normal(0, 1), 2),
                engagement_depth=round(0.4 + self.rng.random() * 0.4, 4),
                information_velocity=round(0.3 + self.rng.random() * 0.5, 4),
                saturation=round(0.2 + self.rng.random() * 0.4, 4),
                novelty_decay=round(0.1 + self.rng.random() * 0.3, 4)
            )

    def _initialize_expectations(self) -> None:
        self._expectations["sp500"] = ExpectationCurve("economic", "SP500", 5500, 5520, 5700, 6000, 0.15, 0.08, "options")
        self._expectations["vix"] = ExpectationCurve("volatility", "VIX", 18, 20, 22, 24, 0.4, 0.15, "futures")
        self._expectations["fed_rate"] = ExpectationCurve("monetary", "Fed Funds", 4.35, 4.25, 3.75, 3.50, 0.3, 0.1, "Fed funds futures")
        self._expectations["cpi"] = ExpectationCurve("inflation", "CPI YoY", 3.0, 2.8, 2.5, 2.3, 0.2, 0.05, "TIPS breakevens")
        self._expectations["unemployment"] = ExpectationCurve("labor", "Unemployment", 4.0, 4.1, 4.3, 4.5, 0.15, 0.03, "forecasts")
        self._expectations["oil"] = ExpectationCurve("energy", "WTI Crude", 70, 68, 72, 75, 0.25, 0.1, "futures")
        self._expectations["ai_adoption"] = ExpectationCurve("technology", "AI Adoption %", 35, 40, 55, 70, 0.35, 0.2, "surveys")

    def update_belief(self, belief_id: str, domain_states: Optional[Dict[str, float]] = None) -> None:
        belief = self._beliefs.get(belief_id)
        if belief is None:
            return
        states = domain_states or {}
        domain_val = states.get(belief.domain, 0.5)
        momentum = states.get("momentum", 0.0)
        belief.prevalence = np.clip(belief.prevalence + 0.03 * (domain_val - belief.prevalence) + self.rng.normal(0, 0.01), 0.05, 0.95)
        belief.confidence = np.clip(belief.confidence + 0.02 * momentum + self.rng.normal(0, 0.01), 0.1, 0.9)
        belief.entrenchment = np.clip(belief.entrenchment + 0.005 * (belief.prevalence - belief.entrenchment), 0.1, 0.9)
        belief.evidence_strength = np.clip(belief.evidence_strength + self.rng.normal(0, 0.02), 0.1, 0.9)

    def update_all_beliefs(self, domain_states: Optional[Dict[str, float]] = None) -> None:
        for bid in list(self._beliefs.keys()):
            self.update_belief(bid, domain_states)
        self._compute_centralities()

    def _compute_centralities(self) -> None:
        in_degree = defaultdict(int)
        for b in self._beliefs.values():
            for child in b.child_beliefs:
                in_degree[child] += 1
        total = max(len(self._beliefs), 1)
        for bid, b in self._beliefs.items():
            b.centrality = min(0.95, max(0.05, (len(b.child_beliefs) + in_degree[bid]) / (2 * total) + 0.3))

    def belief_topology(self) -> Dict[str, Any]:
        nodes = {}
        for bid, b in self._beliefs.items():
            nodes[bid] = {
                "content": b.content, "domain": b.domain,
                "prevalence": b.prevalence, "confidence": b.confidence,
                "centrality": b.centrality, "entrenchment": b.entrenchment,
                "contested": b.is_contested,
                "children": b.child_beliefs, "parents": b.parent_beliefs
            }
        contestation_index = sum(1 for b in self._beliefs.values() if b.is_contested) / max(len(self._beliefs), 1)
        consensus = 1.0 - np.std([b.prevalence for b in self._beliefs.values()]) * 2
        return {"nodes": nodes, "contestation_index": round(contestation_index, 4),
                "consensus": round(max(0, consensus), 4),
                "most_central": max(self._beliefs.values(), key=lambda b: b.centrality).belief_id if self._beliefs else "",
                "most_contested": [bid for bid, b in self._beliefs.items() if b.is_contested and b.centrality > 0.5]}

    def narrative_landscape(self) -> Dict[str, Any]:
        result = {}
        for nid, n in self._narratives.items():
            result[nid] = {"title": n.title, "domain": n.domain, "dominance": n.dominance,
                           "velocity": n.velocity, "emotional_valence": n.emotional_valence,
                           "memetic_fitness": n.memetic_fitness, "lifespan": n.expected_lifespan_hours,
                           "counter_narratives": n.counter_narratives}
        dominant_by_domain = {}
        for domain, nids in self._narrative_competition.items():
            dom_narrs = {nid: self._narratives[nid].dominance for nid in nids if nid in self._narratives}
            if dom_narrs:
                dominant_by_domain[domain] = max(dom_narrs, key=dom_narrs.get)
        return {"narratives": result, "dominant_by_domain": dominant_by_domain,
                "total_narratives": len(self._narratives),
                "most_dominant": max(self._narratives.values(), key=lambda n: n.dominance).narrative_id if self._narratives else "",
                "narrative_diversity": len(self._narratives) / max(len(self._narrative_competition), 1)}

    def attention_map(self) -> Dict[str, Any]:
        vectors = {d: {"share": v.attention_share, "dwell": v.dwell_time_hours,
                       "engagement": v.engagement_depth, "velocity": v.information_velocity,
                       "saturation": v.saturation, "effective": v.effective_attention}
                   for d, v in self._attention.items()}
        total_attention = sum(v.attention_share for v in self._attention.values()) if self._attention else 1.0
        normalized = {d: round(v.attention_share / max(total_attention, 1e-10), 4)
                      for d, v in self._attention.items()}
        return {"vectors": vectors, "normalized": normalized,
                "total_attention": round(total_attention, 4),
                "attention_concentration": self._herfindahl([v.attention_share for v in self._attention.values()]) if self._attention else 0.0}

    def _herfindahl(self, shares: List[float]) -> float:
        total = max(sum(shares), 1e-10)
        return float(np.sum([(s / total) ** 2 for s in shares]))

    def expectation_map(self) -> Dict[str, Any]:
        curves = {}
        for eid, e in self._expectations.items():
            curves[eid] = {"domain": e.domain, "metric": e.metric, "current": e.current_value,
                           "expected_1m": e.expected_1m, "expected_6m": e.expected_6m,
                           "expected_12m": e.expected_12m, "vol": e.implied_vol,
                           "spread": e.spread, "is_inverted": e.is_inverted, "source": e.source}
        return {"curves": curves, "inversion_count": sum(1 for e in self._expectations.values() if e.is_inverted),
                "term_premium": round(sum(e.expected_12m - e.expected_1m for e in self._expectations.values()) / max(len(self._expectations), 1), 2)}

    def information_cascade_signals(self) -> Dict[str, Any]:
        signals = []
        for nid, n in self._narratives.items():
            if n.dominance > 0.6 and n.velocity > 0.05:
                signals.append({"type": "informational_cascade", "narrative_id": nid,
                                "narrative": n.title, "dominance": n.dominance,
                                "velocity": n.velocity, "risk": "informational_herding"})
            if n.velocity > 0.1 and n.memetic_fitness > 0.7:
                signals.append({"type": "viral_spread", "narrative_id": nid,
                                "narrative": n.title, "velocity": n.velocity,
                                "memetic_fitness": n.memetic_fitness, "risk": "belief_lock_in"})
        return {"signals": signals, "count": len(signals)}

    def cognitive_dissonance_map(self) -> Dict[str, Any]:
        dissonances = []
        for nid, n in self._narratives.items():
            if n.emotional_valence < -0.5 and n.dominance > 0.4:
                narratives_in_domain = [cn for cn in n.counter_narratives if cn in self._narratives]
                for cn in narratives_in_domain:
                    cn_obj = self._narratives[cn]
                    if cn_obj.emotional_valence > 0 and cn_obj.dominance > 0.3:
                        gap = n.dominance - cn_obj.dominance
                        dissonances.append({"negative_narrative": n.title,
                                           "positive_narrative": cn_obj.title,
                                           "dominance_gap": round(gap, 3),
                                           "domain": n.domain})
        return {"dissonances": dissonances, "count": len(dissonances)}

    def collective_belief_state(self) -> Dict[str, float]:
        domains = set(b.domain for b in self._beliefs.values())
        result = {}
        for domain in domains:
            beliefs = [b for b in self._beliefs.values() if b.domain == domain]
            if beliefs:
                result[domain] = round(float(np.mean([b.prevalence * b.confidence for b in beliefs])), 4)
        return result