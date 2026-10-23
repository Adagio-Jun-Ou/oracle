"""Prediction genome: evolvable prediction strategy representation.

Each genome encodes a prediction strategy as a set of parameterized functions
that can be mutated, crossed over, and evaluated. The genome serves as the
genotype that MAP-Elites evolves across behavioral descriptors.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class MutationType(Enum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    LLM_DRIVEN = "llm_driven"
    CATASTROPHIC = "catastrophic"
    GENE_DUPLICATION = "gene_duplication"


@dataclass
class EvolvableFunction:
    name: str
    params: Dict[str, float]
    param_bounds: Dict[str, Tuple[float, float]]
    domain: str = ""
    description: str = ""
    mutation_rate: float = 0.1
    mutation_type: MutationType = MutationType.GAUSSIAN
    source: str = "baseline"

    def mutate(self, rng: Optional[random.Random] = None,
               sigma: float = 0.15) -> EvolvableFunction:
        rng = rng or random.Random()
        new_params = {}
        for key, value in self.params.items():
            lo, hi = self.param_bounds.get(key, (0.0, 1.0))
            if rng.random() < self.mutation_rate:
                if self.mutation_type == MutationType.GAUSSIAN:
                    delta = rng.gauss(0, sigma * (hi - lo))
                    new_val = np.clip(value + delta, lo, hi)
                elif self.mutation_type == MutationType.UNIFORM:
                    new_val = rng.uniform(lo, hi)
                elif self.mutation_type == MutationType.POLYNOMIAL:
                    eta = 1.0 + rng.random() * 4.0
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    new_val = np.clip(value + delta * (hi - lo), lo, hi)
                elif self.mutation_type == MutationType.CATASTROPHIC:
                    new_val = rng.uniform(lo, hi)
                else:
                    new_val = value + rng.gauss(0, sigma * (hi - lo))
                    new_val = np.clip(new_val, lo, hi)
                new_params[key] = float(new_val)
            else:
                new_params[key] = float(value)
        return EvolvableFunction(
            name=self.name, params=new_params,
            param_bounds=copy.deepcopy(self.param_bounds),
            domain=self.domain, description=self.description,
            mutation_rate=self.mutation_rate, mutation_type=self.mutation_type,
            source="mutated"
        )

    def crossover(self, other: EvolvableFunction,
                  rng: Optional[random.Random] = None) -> EvolvableFunction:
        rng = rng or random.Random()
        if self.name != other.name:
            return copy.deepcopy(self)
        new_params = {}
        for key in self.params:
            if key not in other.params:
                new_params[key] = self.params[key]
                continue
            alpha = rng.random()
            new_val = alpha * self.params[key] + (1 - alpha) * other.params[key]
            lo, hi = self.param_bounds.get(key, (0.0, 1.0))
            new_params[key] = float(np.clip(new_val, lo, hi))
        return EvolvableFunction(
            name=self.name, params=new_params,
            param_bounds=copy.deepcopy(self.param_bounds),
            domain=self.domain, description=self.description,
            mutation_rate=self.mutation_rate, mutation_type=self.mutation_type,
            source="crossover"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "params": self.params,
            "param_bounds": {k: list(v) for k, v in self.param_bounds.items()},
            "domain": self.domain, "description": self.description,
            "mutation_rate": self.mutation_rate,
            "mutation_type": self.mutation_type.value,
            "source": self.source
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EvolvableFunction:
        return cls(
            name=d["name"], params=d["params"],
            param_bounds={k: tuple(v) for k, v in d["param_bounds"].items()},
            domain=d.get("domain", ""), description=d.get("description", ""),
            mutation_rate=d.get("mutation_rate", 0.1),
            mutation_type=MutationType(d.get("mutation_type", "gaussian")),
            source=d.get("source", "baseline")
        )


@dataclass
class PredictionGenome:
    genome_id: str = ""
    functions: Dict[str, EvolvableFunction] = field(default_factory=dict)
    generation: int = 0
    fitness: float = 0.0
    behavioral_descriptor: List[float] = field(default_factory=list)
    birth_time: float = 0.0
    parent_ids: List[str] = field(default_factory=list)
    origin: str = "initial"

    def __post_init__(self):
        if not self.genome_id:
            content = json.dumps({k: v.params for k, v in sorted(self.functions.items())},
                                 sort_keys=True)
            self.genome_id = hashlib.md5(content.encode()).hexdigest()[:12]

    def mutate(self, rng: Optional[random.Random] = None,
               sigma: float = 0.15,
               mutation_rate_override: float = 0.3) -> PredictionGenome:
        rng = rng or random.Random()
        new_funcs = {}
        for name, func in self.functions.items():
            original_mr = func.mutation_rate
            try:
                func.mutation_rate = mutation_rate_override
                new_funcs[name] = func.mutate(rng, sigma)
            finally:
                func.mutation_rate = original_mr
        child = PredictionGenome(
            functions=new_funcs, generation=self.generation + 1,
            parent_ids=[self.genome_id], origin="mutation",
            behavioral_descriptor=[]
        )
        return child

    def crossover(self, other: PredictionGenome,
                  rng: Optional[random.Random] = None) -> PredictionGenome:
        rng = rng or random.Random()
        all_keys = sorted(set(self.functions.keys()) | set(other.functions.keys()))
        new_funcs = {}
        for key in all_keys:
            if key in self.functions and key in other.functions:
                if rng.random() < 0.5:
                    new_funcs[key] = self.functions[key].crossover(other.functions[key], rng)
                else:
                    new_funcs[key] = other.functions[key].crossover(self.functions[key], rng)
            elif key in self.functions:
                new_funcs[key] = copy.deepcopy(self.functions[key])
            else:
                new_funcs[key] = copy.deepcopy(other.functions[key])
        child = PredictionGenome(
            functions=new_funcs, generation=max(self.generation, other.generation) + 1,
            parent_ids=[self.genome_id, other.genome_id], origin="crossover",
            behavioral_descriptor=[]
        )
        return child

    def apply_function(self, domain_states: Dict[str, float]) -> Dict[str, float]:
        results = {}
        for name, func in self.functions.items():
            if func.domain and func.domain not in domain_states:
                continue
            base = domain_states.get(func.domain, 0.5)
            adjustment = 0.0
            for pname, pval in func.params.items():
                if pname == "base_weight":
                    adjustment += pval * base
                elif pname == "confidence_modifier":
                    adjustment += pval * abs(base - 0.5) * 2
                elif pname == "trend_weight":
                    adjustment += pval * 0.1
                elif pname == "anchoring_strength":
                    adjustment += pval * (0.5 - base) * 0.2
                elif pname == "decay_rate":
                    adjustment -= pval * 0.05
                elif pname == "interaction_strength":
                    adjustment += pval * 0.1
                else:
                    adjustment += pval * 0.01
            results[name] = np.clip(base + adjustment, 0.01, 0.99)
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "functions": {k: v.to_dict() for k, v in self.functions.items()},
            "generation": self.generation, "fitness": self.fitness,
            "behavioral_descriptor": self.behavioral_descriptor,
            "birth_time": self.birth_time,
            "parent_ids": self.parent_ids, "origin": self.origin
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PredictionGenome:
        genome = cls(
            genome_id=d.get("genome_id", ""),
            functions={k: EvolvableFunction.from_dict(v) for k, v in d["functions"].items()},
            generation=d.get("generation", 0),
            fitness=d.get("fitness", 0.0),
            behavioral_descriptor=d.get("behavioral_descriptor", []),
            birth_time=d.get("birth_time", 0.0),
            parent_ids=d.get("parent_ids", []),
            origin=d.get("origin", "initial")
        )
        return genome


def _adjust_probability_params() -> Dict[str, Tuple[float, float]]:
    return {
        "base_weight": (0.0, 2.0),
        "confidence_modifier": (0.0, 1.5),
        "anchoring_strength": (0.0, 1.0),
        "decay_rate": (0.0, 0.5),
        "trend_weight": (0.0, 1.0),
        "interaction_strength": (0.0, 1.0),
    }


def _cascade_params() -> Dict[str, Tuple[float, float]]:
    return {
        "base_weight": (0.0, 1.5),
        "decay_rate": (0.01, 0.8),
        "interaction_strength": (0.0, 1.0),
    }


def _calibration_params() -> Dict[str, Tuple[float, float]]:
    return {
        "base_weight": (0.5, 1.5),
        "confidence_modifier": (0.0, 1.0),
        "anchoring_strength": (0.0, 1.0),
    }


def _confidence_params() -> Dict[str, Tuple[float, float]]:
    return {
        "base_weight": (0.3, 1.5),
        "confidence_modifier": (0.0, 1.5),
        "trend_weight": (0.0, 1.0),
    }


def _synthesis_params() -> Dict[str, Tuple[float, float]]:
    return {
        "base_weight": (0.3, 2.0),
        "confidence_modifier": (0.0, 1.5),
        "anchoring_strength": (0.0, 1.0),
        "interaction_strength": (0.0, 1.5),
        "trend_weight": (0.0, 1.0),
        "decay_rate": (0.0, 0.3),
    }


DOMAIN_GENOMES: Dict[str, Dict[str, Dict]] = {
    "economic": {
        "adjust_probability_for_domain_states": {
            "params": {"base_weight": 1.0, "confidence_modifier": 0.3, "anchoring_strength": 0.2, "decay_rate": 0.05, "trend_weight": 0.5, "interaction_strength": 0.4},
            "bounds": _adjust_probability_params(),
            "domain": "economic"
        },
        "compute_cascade_probability": {
            "params": {"base_weight": 0.8, "decay_rate": 0.3, "interaction_strength": 0.5},
            "bounds": _cascade_params(),
            "domain": "economic"
        },
        "calibrate_probability": {
            "params": {"base_weight": 1.0, "confidence_modifier": 0.5, "anchoring_strength": 0.3},
            "bounds": _calibration_params(),
            "domain": "economic"
        },
        "compute_confidence": {
            "params": {"base_weight": 0.8, "confidence_modifier": 0.4, "trend_weight": 0.3},
            "bounds": _confidence_params(),
            "domain": "economic"
        },
        "synthesize_cross_domain_signals": {
            "params": {"base_weight": 1.0, "confidence_modifier": 0.5, "anchoring_strength": 0.3, "interaction_strength": 0.6, "trend_weight": 0.4, "decay_rate": 0.1},
            "bounds": _synthesis_params(),
            "domain": "economic"
        }
    },
    "geopolitical": {
        "adjust_probability_for_domain_states": {
            "params": {"base_weight": 0.9, "confidence_modifier": 0.5, "anchoring_strength": 0.4, "decay_rate": 0.08, "trend_weight": 0.3, "interaction_strength": 0.6},
            "bounds": _adjust_probability_params(),
            "domain": "geopolitical"
        },
        "compute_cascade_probability": {
            "params": {"base_weight": 0.9, "decay_rate": 0.35, "interaction_strength": 0.6},
            "bounds": _cascade_params(),
            "domain": "geopolitical"
        },
        "calibrate_probability": {
            "params": {"base_weight": 1.0, "confidence_modifier": 0.6, "anchoring_strength": 0.4},
            "bounds": _calibration_params(),
            "domain": "geopolitical"
        },
        "compute_confidence": {
            "params": {"base_weight": 0.7, "confidence_modifier": 0.5, "trend_weight": 0.4},
            "bounds": _confidence_params(),
            "domain": "geopolitical"
        },
        "synthesize_cross_domain_signals": {
            "params": {"base_weight": 0.9, "confidence_modifier": 0.6, "anchoring_strength": 0.4, "interaction_strength": 0.7, "trend_weight": 0.3, "decay_rate": 0.12},
            "bounds": _synthesis_params(),
            "domain": "geopolitical"
        }
    },
    "health": {
        "adjust_probability_for_domain_states": {
            "params": {"base_weight": 1.1, "confidence_modifier": 0.2, "anchoring_strength": 0.3, "decay_rate": 0.04, "trend_weight": 0.6, "interaction_strength": 0.3},
            "bounds": _adjust_probability_params(),
            "domain": "health"
        },
        "compute_cascade_probability": {
            "params": {"base_weight": 0.7, "decay_rate": 0.25, "interaction_strength": 0.4},
            "bounds": _cascade_params(),
            "domain": "health"
        },
        "calibrate_probability": {
            "params": {"base_weight": 0.9, "confidence_modifier": 0.4, "anchoring_strength": 0.5},
            "bounds": _calibration_params(),
            "domain": "health"
        },
        "compute_confidence": {
            "params": {"base_weight": 0.9, "confidence_modifier": 0.3, "trend_weight": 0.5},
            "bounds": _confidence_params(),
            "domain": "health"
        },
        "synthesize_cross_domain_signals": {
            "params": {"base_weight": 1.1, "confidence_modifier": 0.4, "anchoring_strength": 0.5, "interaction_strength": 0.4, "trend_weight": 0.5, "decay_rate": 0.08},
            "bounds": _synthesis_params(),
            "domain": "health"
        }
    }
}


class GenomeRegistry:
    def __init__(self):
        self._genomes: Dict[str, PredictionGenome] = {}
        self._best_per_domain: Dict[str, PredictionGenome] = {}

    def register(self, genome: PredictionGenome) -> None:
        self._genomes[genome.genome_id] = genome

    def get(self, genome_id: str) -> Optional[PredictionGenome]:
        return self._genomes.get(genome_id)

    def best_for_domain(self, domain: str) -> Optional[PredictionGenome]:
        return self._best_per_domain.get(domain)

    def update_best(self, domain: str, genome: PredictionGenome) -> None:
        current = self._best_per_domain.get(domain)
        if current is None or genome.fitness > current.fitness:
            self._best_per_domain[domain] = genome

    def top_k(self, k: int = 5) -> List[PredictionGenome]:
        sorted_genomes = sorted(self._genomes.values(), key=lambda g: g.fitness, reverse=True)
        return sorted_genomes[:k]

    def all_genomes(self) -> List[PredictionGenome]:
        return list(self._genomes.values())

    def prune(self, max_size: int = 500) -> int:
        if len(self._genomes) <= max_size:
            return 0
        sorted_genomes = sorted(self._genomes.values(), key=lambda g: g.fitness, reverse=True)
        to_keep = {g.genome_id for g in sorted_genomes[:max_size]}
        pruned = len(self._genomes) - max_size
        self._genomes = {gid: g for gid, g in self._genomes.items() if gid in to_keep}
        return pruned


def create_default_genome(domain: str = "economic") -> PredictionGenome:
    domain_funcs: Dict[str, EvolvableFunction] = {}
    domain_config = DOMAIN_GENOMES.get(domain, DOMAIN_GENOMES["economic"])
    for func_name, config in domain_config.items():
        domain_funcs[func_name] = EvolvableFunction(
            name=func_name, params=dict(config["params"]),
            param_bounds=config["bounds"], domain=config["domain"],
            description=f"Prediction function for {config['domain']}",
            source="baseline"
        )
    genome = PredictionGenome(functions=domain_funcs, origin=f"baseline_{domain}")
    return genome


def create_all_domain_genomes() -> Dict[str, PredictionGenome]:
    return {domain: create_default_genome(domain) for domain in DOMAIN_GENOMES}


def create_random_genome(domain: str = "economic",
                         rng: Optional[random.Random] = None) -> PredictionGenome:
    rng = rng or random.Random()
    genome = create_default_genome(domain)
    for func in genome.functions.values():
        for key in func.params:
            lo, hi = func.param_bounds[key]
            func.params[key] = rng.uniform(lo, hi)
    genome.origin = "random"
    genome.genome_id = ""
    genome.__post_init__()
    return genome