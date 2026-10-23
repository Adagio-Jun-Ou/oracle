"""NarrativeEvolutionEngine — natural selection of ideas and memetic competition.

Models how narratives evolve, compete, spread, and die in the collective mind.
Treats stories as organisms in a memetic ecosystem:

- Memetic fitness: how well a narrative survives and reproduces across minds
- Narrative selection: which stories dominate based on emotional resonance, simplicity, novelty
- Mutation: how stories change as they spread (Chinese whispers, embellishment, simplification)
- Cross-domain hopping: how narratives jump between economic, geopolitical, health domains
- Narrative lifecycle: birth → growth → peak → decay → death or immortalization

Evidence base:
- Memetics (Dawkins 1976, Blackmore 1999)
- Narrative economics (Shiller 2017, 2019)
- Viral media spread (Vosoughi et al. 2018, Science)
- Information cascades (Bikhchandani et al. 1992)
- Cultural evolution (Boyd & Richerson 1985, Mesoudi 2011)
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MemeLifecycle(Enum):
    GESTATION = "gestation"          
    GROWING = "growing"              
    PEAK = "peak"                    
    DECAYING = "decaying"            
    DEAD = "dead"                    
    IMMORTALIZED = "immortalized"    


@dataclass
class Meme:
    meme_id: str
    content: str
    domain: str
    memetic_fitness: float           
    spread_rate: float               
    mutation_rate: float             
    emotional_resonance: float       
    cognitive_simplicity: float      
    novelty: float                   
    authority_boost: float           
    age_hours: float = 0.0
    prevalence: float = 0.0          
    lifecycle: MemeLifecycle = MemeLifecycle.GESTATION
    variant_count: int = 0
    parent_meme_id: str = ""
    cross_domain_potential: Dict[str, float] = field(default_factory=dict)
    
    @property
    def composite_fitness(self) -> float:
        return (self.memetic_fitness * 0.3 + self.emotional_resonance * 0.25 +
                self.cognitive_simplicity * 0.2 + self.novelty * 0.15 +
                self.authority_boost * 0.1)
    
    @property
    def is_viral(self) -> bool:
        return self.spread_rate > 0.5 and self.composite_fitness > 0.7


@dataclass
class MemeticEcosystem:
    domain: str
    memes: Dict[str, Meme] = field(default_factory=dict)
    carrying_capacity: int = 20
    competition_intensity: float = 0.0
    novelty_threshold: float = 0.3
    homogeneity: float = 0.5


class NarrativeEvolutionEngine:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._ecosystems: Dict[str, MemeticEcosystem] = {}
        self._global_meme_counter: int = 0
        self._meme_extinction_events: List[Dict] = []
        self._meme_mutation_log: List[Dict] = []
        self._initialize_default_memes()

    def _initialize_default_memes(self) -> None:
        defaults = [
            {"domain": "economic", "content": "Inflation is here to stay",
             "fitness": 0.6, "spread": 0.4, "emotion": 0.5, "simplicity": 0.7, "novelty": 0.3, "authority": 0.6},
            {"domain": "economic", "content": "The Fed has it under control",
             "fitness": 0.55, "spread": 0.35, "emotion": 0.4, "simplicity": 0.8, "novelty": 0.2, "authority": 0.5},
            {"domain": "technology", "content": "AI will replace all jobs",
             "fitness": 0.7, "spread": 0.6, "emotion": 0.65, "simplicity": 0.75, "novelty": 0.5, "authority": 0.5},
            {"domain": "geopolitical", "content": "We are entering a new Cold War",
             "fitness": 0.55, "spread": 0.45, "emotion": 0.6, "simplicity": 0.7, "novelty": 0.2, "authority": 0.6},
            {"domain": "health", "content": "The next pandemic is inevitable",
             "fitness": 0.45, "spread": 0.3, "emotion": 0.55, "simplicity": 0.8, "novelty": 0.1, "authority": 0.5},
        ]
        for d in defaults:
            self._ecosystems.setdefault(d["domain"], MemeticEcosystem(domain=d["domain"]))
            self._generate_meme(d["domain"], d["content"], d["fitness"], d["spread"],
                               d["emotion"], d["simplicity"], d["novelty"], d["authority"])

    def _generate_meme(self, domain: str, content: str, fitness: float = 0.5,
                       spread: float = 0.3, emotion: float = 0.5,
                       simplicity: float = 0.5, novelty: float = 0.3,
                       authority: float = 0.3) -> Meme:
        self._global_meme_counter += 1
        meme_id = f"meme_{self._global_meme_counter:04d}"
        cross_potential = {}
        for other_domain in ["economic", "geopolitical", "health", "technology", "social"]:
            if other_domain != domain:
                cross_potential[other_domain] = self.rng.random() * 0.3
        meme = Meme(
            meme_id=meme_id, content=content, domain=domain,
            memetic_fitness=fitness, spread_rate=spread, mutation_rate=self.rng.random() * 0.1,
            emotional_resonance=emotion, cognitive_simplicity=simplicity,
            novelty=novelty, authority_boost=authority, prevalence=0.05
        )
        ecosystem = self._ecosystems.get(domain)
        if ecosystem:
            ecosystem.memes[meme_id] = meme
        return meme

    def evolve(self, domain: str, steps: int = 1) -> List[Meme]:
        ecosystem = self._ecosystems.get(domain)
        if ecosystem is None:
            ecosystem = MemeticEcosystem(domain=domain)
            self._ecosystems[domain] = ecosystem
        
        for _ in range(steps):
            self._step_ecosystem(ecosystem)
            self._apply_selection_pressure(ecosystem)
        
        return sorted(ecosystem.memes.values(), key=lambda m: m.prevalence, reverse=True)

    def _step_ecosystem(self, eco: MemeticEcosystem) -> None:
        for meme in list(eco.memes.values()):
            meme.age_hours += 1.0
            
            growth = meme.spread_rate * meme.prevalence * (1.0 - meme.prevalence) * 0.1
            decay = 0.001 * meme.age_hours * (1.0 - meme.memetic_fitness)
            meme.prevalence = np.clip(meme.prevalence + growth - decay, 0.0, 0.95)
            
            if meme.prevalence > 0.5 and meme.lifecycle == MemeLifecycle.GESTATION:
                meme.lifecycle = MemeLifecycle.GROWING
            elif meme.prevalence > 0.7:
                meme.lifecycle = MemeLifecycle.PEAK
            elif meme.prevalence < 0.1 and meme.age_hours > 50:
                meme.lifecycle = MemeLifecycle.DECAYING
            elif meme.prevalence < 0.01 and meme.age_hours > 100:
                meme.lifecycle = MemeLifecycle.DEAD
            
            if self.rng.random() < meme.mutation_rate:
                self._mutate_meme(meme, eco)
            
            if self.rng.random() < 0.01 and any(p > 0.3 for p in meme.cross_domain_potential.values()):
                self._cross_domain_hop(meme)

    def _apply_selection_pressure(self, eco: MemeticEcosystem) -> None:
        dead_ids = [mid for mid, m in eco.memes.items() if m.lifecycle == MemeLifecycle.DEAD]
        for mid in dead_ids:
            meme = eco.memes.pop(mid)
            self._meme_extinction_events.append({
                "meme_id": mid, "content": meme.content, "domain": eco.domain,
                "age_hours": meme.age_hours, "peak_prevalence": meme.prevalence
            })
        
        if len(eco.memes) > eco.carrying_capacity:
            sorted_memes = sorted(eco.memes.items(), key=lambda kv: kv[1].prevalence)
            to_kill = sorted_memes[:len(sorted_memes) - eco.carrying_capacity]
            for mid, meme in to_kill:
                meme.lifecycle = MemeLifecycle.DEAD
                eco.memes.pop(mid)
        
        if self.rng.random() < 0.05:
            eco.carrying_capacity = max(5, int(eco.carrying_capacity + self.rng.normal(0, 2)))
        
        alive = len(eco.memes)
        eco.competition_intensity = np.clip(alive / max(eco.carrying_capacity, 1), 0.0, 1.0)

    def _mutate_meme(self, meme: Meme, eco: MemeticEcosystem) -> None:
        meme.variant_count += 1
        mutation_type = self.rng.choice(["drift", "embellishment", "simplification", "polarize"])
        if mutation_type == "drift":
            meme.emotional_resonance = np.clip(meme.emotional_resonance + self.rng.normal(0, 0.05), 0.0, 1.0)
            meme.memetic_fitness = np.clip(meme.memetic_fitness + self.rng.normal(0, 0.03), 0.0, 1.0)
        elif mutation_type == "embellishment":
            meme.memetic_fitness = np.clip(meme.memetic_fitness + 0.05 * self.rng.random(), 0.0, 1.0)
        elif mutation_type == "simplification":
            meme.cognitive_simplicity = np.clip(meme.cognitive_simplicity + 0.03, 0.0, 1.0)
            meme.spread_rate = np.clip(meme.spread_rate + 0.02, 0.0, 0.8)
        elif mutation_type == "polarize":
            meme.emotional_resonance = np.clip(meme.emotional_resonance + 0.1 * (2 * self.rng.random() - 1), 0.0, 1.0)
        self._meme_mutation_log.append({
            "meme_id": meme.meme_id, "type": mutation_type,
            "new_fitness": meme.memetic_fitness, "new_emotion": meme.emotional_resonance,
            "variant_count": meme.variant_count
        })

    def _cross_domain_hop(self, source_meme: Meme) -> None:
        target_domains = [d for d, p in source_meme.cross_domain_potential.items()
                         if p > 0.3 and d != source_meme.domain]
        if not target_domains:
            return
        target = self.rng.choice(target_domains)
        new_content = source_meme.content.replace(source_meme.domain, target)
        self._generate_meme(
            domain=target, content=new_content,
            fitness=source_meme.memetic_fitness * 0.7,
            spread=source_meme.spread_rate * 0.5,
            emotion=source_meme.emotional_resonance * 0.8,
            simplicity=source_meme.cognitive_simplicity,
            novelty=source_meme.novelty + 0.3,
            authority=source_meme.authority_boost * 0.5
        )

    def get_dominant_memes(self, domain: str, n: int = 5) -> List[Dict[str, Any]]:
        eco = self._ecosystems.get(domain)
        if eco is None:
            return []
        sorted_memes = sorted(eco.memes.values(), key=lambda m: m.prevalence, reverse=True)
        return [{
            "meme_id": m.meme_id, "content": m.content,
            "prevalence": round(m.prevalence, 4),
            "fitness": round(m.memetic_fitness, 4),
            "lifecycle": m.lifecycle.value,
            "spread_rate": round(m.spread_rate, 4),
            "emotional_resonance": round(m.emotional_resonance, 4)
        } for m in sorted_memes[:n]]

    def ecosystem_stats(self) -> Dict[str, Any]:
        stats = {}
        for domain, eco in self._ecosystems.items():
            stats[domain] = {
                "population": len(eco.memes),
                "carrying_capacity": eco.carrying_capacity,
                "competition": round(eco.competition_intensity, 4),
                "mean_prevalence": round(float(np.mean([m.prevalence for m in eco.memes.values()])), 4) if eco.memes else 0.0,
                "gestation": sum(1 for m in eco.memes.values() if m.lifecycle == MemeLifecycle.GESTATION),
                "growing": sum(1 for m in eco.memes.values() if m.lifecycle == MemeLifecycle.GROWING),
                "peak": sum(1 for m in eco.memes.values() if m.lifecycle == MemeLifecycle.PEAK),
                "decaying": sum(1 for m in eco.memes.values() if m.lifecycle == MemeLifecycle.DECAYING),
            }
        return {"ecosystems": stats, "total_extinctions": len(self._meme_extinction_events),
                "total_mutations": len(self._meme_mutation_log),
                "total_memes_alive": sum(len(eco.memes) for eco in self._ecosystems.values())}

    def predict_memetic_wave(self, domain: str, horizon_hours: float = 168) -> Dict[str, Any]:
        eco = self._ecosystems.get(domain)
        if eco is None or not eco.memes:
            return {"domain": domain, "prediction": "no memes in ecosystem"}

        future_memes = []
        for meme in eco.memes.values():
            projected_prev = meme.prevalence * np.exp(meme.spread_rate * horizon_hours / 24)
            future_prev = np.clip(projected_prev, 0.0, 0.95)
            future_memes.append({
                "meme_id": meme.meme_id, "content": meme.content,
                "current_prevalence": round(meme.prevalence, 3),
                "projected_prevalence": round(future_prev, 3),
                "growth": round(future_prev - meme.prevalence, 3),
                "is_rising": future_prev > meme.prevalence + 0.05,
                "is_peaking": meme.lifecycle == MemeLifecycle.PEAK,
                "will_go_viral": meme.is_viral and future_prev > 0.6
            })

        future_memes.sort(key=lambda x: x["projected_prevalence"], reverse=True)
        rising_count = sum(1 for m in future_memes if m["is_rising"])
        viral_count = sum(1 for m in future_memes if m["will_go_viral"])

        return {
            "domain": domain,
            "horizon_hours": horizon_hours,
            "projected_dominant": future_memes[0]["content"] if future_memes else "",
            "rising_narrative_count": rising_count,
            "potential_viral_count": viral_count,
            "ecosystem_growth": rising_count > len(eco.memes) * 0.3,
            "top_memes": future_memes[:5]
        }