"""LLM-driven mutation and crossover operators for prediction genomes.

Uses cloud LLMs (GLM-5.1:cloud, etc.) via Ollama Cloud to:
- Generate novel prediction strategies from natural language
- Mutate genome parameters guided by fitness feedback
- Inject domain knowledge as new genome functions
- Perform semantic crossover between successful strategies
"""
from __future__ import annotations

import copy
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oracle.evolution.genome import (
    EvolvableFunction, MutationType, PredictionGenome, create_default_genome
)

logger = logging.getLogger(__name__)


class MutationStrategy(Enum):
    PARAMETER_PERTURBATION = "parameter_perturbation"
    GAUSSIAN_ADAPTATION = "gaussian_adaptation"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    LLM_GUIDED = "llm_guided"
    HYBRID = "hybrid"


@dataclass
class MutationResult:
    genome: PredictionGenome
    strategy: str
    mutation_details: Dict[str, Any] = field(default_factory=dict)
    llm_prompt: str = ""
    llm_response: str = ""
    success: bool = True


@dataclass
class CrossoverResult:
    child: PredictionGenome
    parent1_id: str
    parent2_id: str
    crossover_points: List[str] = field(default_factory=list)
    method: str = "uniform"


@dataclass
class InjectionResult:
    genome: PredictionGenome
    domain: str
    concept: str
    llm_prompt: str = ""
    llm_response: str = ""
    success: bool = True


class LLMMutator:
    """LLM-guided mutation and crossover for prediction genomes."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "glm-5.1:cloud",
        api_key: Optional[str] = None,
        seed: int = 42,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        fallback_to_heuristic: bool = True,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.fallback_to_heuristic = fallback_to_heuristic
        self.rng = random.Random(seed)
        self._mutation_count = 0
        self._llm_call_count = 0
        self._llm_failure_count = 0

    def _call_llm(self, prompt: str) -> Optional[str]:
        if not self.base_url:
            self._llm_failure_count += 1
            return None
        try:
            import urllib.request
            import urllib.error
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            url = f"{self.base_url.rstrip('/')}/api/generate"
            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature, "seed": self.seed}
            }).encode()
            req = urllib.request.Request(url, data=payload, headers=headers)
            timeout_tuple = (self.connect_timeout, self.timeout)
            with urllib.request.urlopen(req, timeout=timeout_tuple) as resp:
                data = json.loads(resp.read().decode())
                self._llm_call_count += 1
                return data.get("response", "")
        except Exception as e:
            logger.debug(f"LLM call failed: {e}")
            self._llm_failure_count += 1
            return None

    def mutate(self, genome: PredictionGenome,
               strategy: MutationStrategy = MutationStrategy.HYBRID,
               sigma: float = 0.15) -> MutationResult:
        if strategy == MutationStrategy.LLM_GUIDED:
            return self._llm_mutate(genome)
        elif strategy == MutationStrategy.GAUSSIAN_ADAPTATION:
            return self._gaussian_mutate(genome, sigma)
        elif strategy == MutationStrategy.DIFFERENTIAL_EVOLUTION:
            return self._de_mutate(genome)
        elif strategy == MutationStrategy.PARAMETER_PERTURBATION:
            return self._perturbation_mutate(genome, sigma)
        else:
            if self.rng.random() < 0.3:
                result = self._llm_mutate(genome)
                if result.success:
                    return result
            return self._gaussian_mutate(genome, sigma)

    def _gaussian_mutate(self, genome: PredictionGenome, sigma: float = 0.15) -> MutationResult:
        child = genome.mutate(self.rng, sigma)
        mutated_params = {}
        for name, func in child.functions.items():
            original = genome.functions.get(name)
            if original:
                diff = {k: round(child.functions[name].params[k] - original.params[k], 4)
                        for k in original.params}
                mutated_params[name] = diff
        self._mutation_count += 1
        return MutationResult(
            genome=child, strategy="gaussian",
            mutation_details={"mutated_params": mutated_params, "sigma": sigma}
        )

    def _perturbation_mutate(self, genome: PredictionGenome, sigma: float = 0.1) -> MutationResult:
        child = genome.mutate(self.rng, sigma)
        self._mutation_count += 1
        return MutationResult(
            genome=child, strategy="perturbation",
            mutation_details={"sigma": sigma}
        )

    def _de_mutate(self, genome: PredictionGenome) -> MutationResult:
        child = PredictionGenome(
            generation=genome.generation + 1,
            parent_ids=[genome.genome_id], origin="de_mutation"
        )
        for name, func in genome.functions.items():
            perturb_scale = 0.5 + self.rng.random()
            new_params = {}
            for key, val in func.params.items():
                lo, hi = func.param_bounds.get(key, (0.0, 1.0))
                delta = self.rng.gauss(0, perturb_scale * (hi - lo) * 0.3)
                new_params[key] = float(np.clip(val + delta, lo, hi))
            child.functions[name] = EvolvableFunction(
                name=name, params=new_params,
                param_bounds=copy.deepcopy(func.param_bounds),
                domain=func.domain, description=func.description,
                mutation_rate=func.mutation_rate,
                mutation_type=func.mutation_type, source="de_mutation"
            )
        self._mutation_count += 1
        return MutationResult(
            genome=child, strategy="differential_evolution",
            mutation_details={"perturb_scale": perturb_scale}
        )

    def _llm_mutate(self, genome: PredictionGenome) -> MutationResult:
        current_params = {name: {k: round(v, 3) for k, v in func.params.items()}
                          for name, func in genome.functions.items()}
        fitness_info = f"Current fitness: {genome.fitness:.4f}, generation: {genome.generation}"
        prompt = f"""You are an expert prediction system optimizer. Given the current prediction strategy parameters and its fitness score, suggest improved parameter values.

Current parameters (JSON):
{json.dumps(current_params, indent=2)}

{fitness_info}

Rules:
1. Only modify parameters that would improve prediction accuracy
2. Keep values within reasonable ranges
3. Return ONLY a JSON object with the same structure, with modified values
4. Make small adjustments (5-15% changes)
5. Focus on calibration and cascade accuracy

Response (JSON only):"""

        response = self._call_llm(prompt)
        if response:
            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    suggested = json.loads(response[json_start:json_end])
                    child = self._apply_llm_suggestion(genome, suggested)
                    if child:
                        self._mutation_count += 1
                        return MutationResult(
                            genome=child, strategy="llm_guided",
                            mutation_details={"llm_modified": list(suggested.keys())},
                            llm_prompt=prompt, llm_response=response
                        )
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"Failed to parse LLM response: {e}")

        if self.fallback_to_heuristic:
            return self._gaussian_mutate(genome, 0.2)
        return MutationResult(
            genome=genome, strategy="llm_failed",
            success=False
        )

    def _apply_llm_suggestion(self, genome: PredictionGenome,
                               suggested: Dict[str, Any]) -> Optional[PredictionGenome]:
        child = PredictionGenome(
            generation=genome.generation + 1,
            parent_ids=[genome.genome_id], origin="llm_mutation"
        )
        for name, func in genome.functions.items():
            if name in suggested and isinstance(suggested[name], dict):
                new_params = dict(func.params)
                for key in suggested[name]:
                    if key in func.params:
                        lo, hi = func.param_bounds.get(key, (0.0, 1.0))
                        try:
                            val = float(suggested[name][key])
                            new_params[key] = float(np.clip(val, lo, hi))
                        except (ValueError, TypeError):
                            continue
                child.functions[name] = EvolvableFunction(
                    name=name, params=new_params,
                    param_bounds=copy.deepcopy(func.param_bounds),
                    domain=func.domain, description=func.description,
                    mutation_rate=func.mutation_rate,
                    mutation_type=func.mutation_type, source="llm_mutation"
                )
            else:
                child.functions[name] = copy.deepcopy(func)
        return child

    def crossover(self, parent1: PredictionGenome, parent2: PredictionGenome,
                   method: str = "uniform") -> CrossoverResult:
        if method == "uniform":
            child = parent1.crossover(parent2, self.rng)
        elif method == "single_point":
            child = self._single_point_crossover(parent1, parent2)
        elif method == "blend":
            child = self._blend_crossover(parent1, parent2)
        else:
            child = parent1.crossover(parent2, self.rng)
        crossover_points = [name for name in child.functions
                            if name in parent1.functions and name in parent2.functions]
        return CrossoverResult(
            child=child, parent1_id=parent1.genome_id,
            parent2_id=parent2.genome_id,
            crossover_points=crossover_points, method=method
        )

    def _single_point_crossover(self, p1: PredictionGenome, p2: PredictionGenome) -> PredictionGenome:
        keys = sorted(set(p1.functions.keys()) | set(p2.functions.keys()))
        point = self.rng.randint(1, max(len(keys) - 1, 1)) if len(keys) > 1 else 0
        child = PredictionGenome(
            generation=max(p1.generation, p2.generation) + 1,
            parent_ids=[p1.genome_id, p2.genome_id], origin="single_point_crossover"
        )
        for i, key in enumerate(keys):
            if i < point:
                if key in p1.functions:
                    child.functions[key] = copy.deepcopy(p1.functions[key])
            else:
                if key in p2.functions:
                    child.functions[key] = copy.deepcopy(p2.functions[key])
        return child

    def _blend_crossover(self, p1: PredictionGenome, p2: PredictionGenome) -> PredictionGenome:
        child = PredictionGenome(
            generation=max(p1.generation, p2.generation) + 1,
            parent_ids=[p1.genome_id, p2.genome_id], origin="blend_crossover"
        )
        all_keys = sorted(set(p1.functions.keys()) | set(p2.functions.keys()))
        for key in all_keys:
            if key in p1.functions and key in p2.functions:
                f1, f2 = p1.functions[key], p2.functions[key]
                new_params = {}
                for pname in f1.params:
                    if pname in f2.params:
                        lo = min(f1.params[pname], f2.params[pname])
                        hi = max(f1.params[pname], f2.params[pname])
                        alpha = 0.5
                        spread = (hi - lo) * alpha
                        new_params[pname] = float(np.clip(
                            self.rng.uniform(lo - spread, hi + spread),
                            f1.param_bounds.get(pname, (0.0, 1.0))[0],
                            f1.param_bounds.get(pname, (0.0, 1.0))[1]
                        ))
                    else:
                        new_params[pname] = f1.params[pname]
                child.functions[key] = EvolvableFunction(
                    name=key, params=new_params,
                    param_bounds=copy.deepcopy(f1.param_bounds),
                    domain=f1.domain, description=f1.description,
                    mutation_rate=f1.mutation_rate,
                    mutation_type=f1.mutation_type, source="blend_crossover"
                )
            elif key in p1.functions:
                child.functions[key] = copy.deepcopy(p1.functions[key])
            else:
                child.functions[key] = copy.deepcopy(p2.functions[key])
        return child

    def inject_novel_strategy(self, domain: str = "economic",
                               context: str = "") -> InjectionResult:
        prompt = f"""You are designing a novel prediction strategy for the '{domain}' domain.
{context if context else f'Consider current {domain} conditions and prediction challenges.'}

Design a prediction strategy with parameter values for these functions:
1. adjust_probability_for_domain_states — adjusts probability based on domain states
2. compute_cascade_probability — computes cascade propagation probability
3. calibrate_probability — calibrates raw probability estimates
4. compute_confidence — computes prediction confidence
5. synthesize_cross_domain_signals — combines signals from multiple domains

For each function, provide these parameters as JSON:
- base_weight (0.0-2.0): weight of base domain value
- confidence_modifier (0.0-1.5): how much confidence affects output
- anchoring_strength (0.0-1.0): base rate anchoring
- decay_rate (0.0-0.5): temporal discount factor
- trend_weight (0.0-1.0): importance of trend direction
- interaction_strength (0.0-1.5): cross-domain interaction weight

Return ONLY a JSON object like:
{{"adjust_probability_for_domain_states": {{"base_weight": X, ...}}, ...}}"""

        response = self._call_llm(prompt)
        if response:
            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    suggested = json.loads(response[json_start:json_end])
                    genome = create_default_genome(domain)
                    for func_name, params in suggested.items():
                        if func_name in genome.functions and isinstance(params, dict):
                            for key, val in params.items():
                                if key in genome.functions[func_name].params:
                                    lo, hi = genome.functions[func_name].param_bounds.get(key, (0.0, 1.0))
                                    try:
                                        genome.functions[func_name].params[key] = float(np.clip(val, lo, hi))
                                    except (ValueError, TypeError):
                                        continue
                    genome.origin = "llm_injection"
                    genome.genome_id = ""
                    genome.__post_init__()
                    return InjectionResult(
                        genome=genome, domain=domain,
                        concept="llm_generated",
                        llm_prompt=prompt, llm_response=response
                    )
            except (json.JSONDecodeError, ValueError):
                pass

        genome = create_default_genome(domain)
        for func in genome.functions.values():
            for key in func.params:
                lo, hi = func.param_bounds[key]
                func.params[key] = self.rng.uniform(lo, hi)
        genome.origin = "heuristic_injection"
        genome.genome_id = ""
        genome.__post_init__()
        return InjectionResult(
            genome=genome, domain=domain, concept="heuristic_random",
            success=True
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "total_mutations": self._mutation_count,
            "llm_calls": self._llm_call_count,
            "llm_failures": self._llm_failure_count,
            "llm_success_rate": max(0, (self._llm_call_count - self._llm_failure_count)) / max(self._llm_call_count, 1)
        }