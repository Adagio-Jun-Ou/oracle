# -*- coding: utf-8 -*-
"""The Cradle — Branching-future sandbox simulation engine.

Inspired by Westworld's Cradle: simulate thousands of branching futures from a
single present state, identify which branches lead to desirable vs. catastrophic
outcomes, and return the probability distribution over future states.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
from collections import defaultdict
import math
import random


class BranchOutcome(Enum):
    COLLAPSE = "collapse"
    DECLINE = "decline"
    STAGNATION = "stagnation"
    GROWTH = "growth"
    GOLDEN_AGE = "golden_age"


@dataclass
class Branch:
    branch_id: str
    trajectory: List[Dict[str, float]]
    final_state: Dict[str, float]
    outcome: BranchOutcome
    probability: float
    path_description: str
    critical_juncture: Optional[int] = None


@dataclass
class CradleResult:
    n_branches: int
    n_generations: int
    outcome_distribution: Dict[str, float]
    most_likely_outcome: str
    optimal_branches: List[Branch]
    catastrophic_branches: List[Branch]
    critical_junctures: List[Dict[str, Any]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    recommendation: str


class CradleSimulator:
    """Simulates branching futures from current state.

    Like Rehoboam's predictive capacity: every decision branches into probabilities.
    The Cradle runs Monte Carlo tree search through future space.
    """

    def __init__(self, n_branches: int = 1000, max_generations: int = 20,
                 branch_factor: float = 0.3, random_seed: int = 42):
        self.n_branches = n_branches
        self.max_generations = max_generations
        self.branch_factor = branch_factor  # probability of branching per step
        self.rng = random.Random(random_seed)
        self.np_rng = np.random.default_rng(random_seed)

    def simulate(self, initial_state: Dict[str, float],
                 transition_fn: Optional[Callable] = None,
                 constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> CradleResult:
        """Simulate branching futures from initial state."""
        branches = []
        for i in range(self.n_branches):
            branch = self._run_branch(initial_state, transition_fn, constraints, branch_id=f"B{i}")
            branches.append(branch)
        # Analyze outcomes
        outcomes = defaultdict(list)
        for b in branches:
            outcomes[b.outcome.value].append(b)
        total = len(branches)
        distribution = {k: len(v) / total for k, v in outcomes.items()}
        most_likely = max(distribution, key=distribution.get)
        # Optimal branches (growth/golden_age)
        optimal = sorted(
            [b for b in branches if b.outcome in (BranchOutcome.GROWTH, BranchOutcome.GOLDEN_AGE)],
            key=lambda b: b.probability, reverse=True
        )[:10]
        # Catastrophic
        catastrophic = sorted(
            [b for b in branches if b.outcome in (BranchOutcome.COLLAPSE, BranchOutcome.DECLINE)],
            key=lambda b: b.probability, reverse=True
        )[:10]
        # Critical junctures: where futures diverged most
        junctures = self._find_critical_junctures(branches)
        # Confidence intervals for key variables
        ci = self._compute_confidence_intervals(branches)
        rec = self._generate_recommendation(distribution, optimal, catastrophic)
        return CradleResult(
            n_branches=total,
            n_generations=self.max_generations,
            outcome_distribution={k: round(v, 4) for k, v in distribution.items()},
            most_likely_outcome=most_likely,
            optimal_branches=optimal,
            catastrophic_branches=catastrophic,
            critical_junctures=junctures[:5],
            confidence_intervals=ci,
            recommendation=rec,
        )

    def _run_branch(self, initial_state: Dict[str, float],
                     transition_fn: Optional[Callable],
                     constraints: Optional[Dict[str, Tuple[float, float]]],
                     branch_id: str) -> Branch:
        trajectory = [dict(initial_state)]
        state = dict(initial_state)
        critical_juncture = None
        for gen in range(self.max_generations):
            if transition_fn:
                next_state = transition_fn(state, gen)
            else:
                next_state = self._default_transition(state)
            if constraints:
                for key, (lo, hi) in constraints.items():
                    if key in next_state:
                        next_state[key] = float(np.clip(next_state[key], lo, hi))
            trajectory.append(next_state)
            state = next_state
            # Branching factor: random perturbation
            if self.rng.random() < self.branch_factor:
                state = {k: v + self.np_rng.normal(0, 0.05) for k, v in state.items()}
                if critical_juncture is None:
                    critical_juncture = gen
        # Classify outcome
        outcome = self._classify_outcome(state, trajectory)
        prob = self._compute_path_probability(trajectory)
        path_desc = self._describe_path(trajectory, outcome)
        return Branch(
            branch_id=branch_id,
            trajectory=trajectory,
            final_state=state,
            outcome=outcome,
            probability=round(prob, 6),
            path_description=path_desc,
            critical_juncture=critical_juncture,
        )

    def _default_transition(self, state: Dict[str, float]) -> Dict[str, float]:
        """Default transition: damped random walk with momentum."""
        result = {}
        for key, val in state.items():
            momentum = self.np_rng.normal(0, 0.02)
            result[key] = float(np.clip(val + momentum, 0.0, 1.0))
        return result

    def _classify_outcome(self, final: Dict[str, float],
                          trajectory: List[Dict[str, float]]) -> BranchOutcome:
        avg_val = np.mean(list(final.values()))
        trend = final.get("trend", avg_val - np.mean(list(trajectory[0].values())))
        if avg_val < 0.2 or trend < -0.3:
            return BranchOutcome.COLLAPSE
        elif avg_val < 0.4 or trend < -0.1:
            return BranchOutcome.DECLINE
        elif avg_val > 0.8 and trend > 0.1:
            return BranchOutcome.GOLDEN_AGE
        elif avg_val > 0.6 or trend > 0.05:
            return BranchOutcome.GROWTH
        return BranchOutcome.STAGNATION

    def _compute_path_probability(self, trajectory: List[Dict[str, float]]) -> float:
        """Compute path probability based on smoothness."""
        if len(trajectory) < 2:
            return 0.5
        diffs = []
        for i in range(1, len(trajectory)):
            for key in trajectory[i]:
                diffs.append(abs(trajectory[i][key] - trajectory[i-1].get(key, 0)))
        mean_diff = np.mean(diffs) if diffs else 0.1
        return math.exp(-mean_diff * 5)

    def _describe_path(self, trajectory: List[Dict[str, float]],
                       outcome: BranchOutcome) -> str:
        start_avg = np.mean(list(trajectory[0].values()))
        end_avg = np.mean(list(trajectory[-1].values()))
        direction = "↑" if end_avg > start_avg else "↓" if end_avg < start_avg else "→"
        return f"{direction} {outcome.value}: {start_avg:.2f} → {end_avg:.2f} over {len(trajectory)} steps"

    def _find_critical_junctures(self, branches: List[Branch]) -> List[Dict[str, Any]]:
        junctures = defaultdict(list)
        for b in branches:
            if b.critical_juncture is not None:
                junctures[b.critical_juncture].append(b.outcome.value)
        result = []
        for gen, outcomes in sorted(junctures.items()):
            counts = defaultdict(int)
            for o in outcomes:
                counts[o] += 1
            total = len(outcomes)
            result.append({
                "generation": gen,
                "n_divergences": total,
                "outcome_distribution": {k: round(v/total, 3) for k, v in counts.items()},
                "entropy": round(-sum((c/total) * math.log(c/total) for c in counts.values()), 3),
            })
        return result

    def _compute_confidence_intervals(self, branches: List[Branch]) -> Dict[str, Tuple[float, float]]:
        if not branches:
            return {}
        all_keys = set()
        for b in branches:
            all_keys.update(b.final_state.keys())
        ci = {}
        for key in all_keys:
            vals = sorted([b.final_state.get(key, 0.5) for b in branches])
            if len(vals) >= 10:
                ci[key] = (round(float(np.percentile(vals, 5)), 4),
                           round(float(np.percentile(vals, 95)), 4))
            else:
                ci[key] = (round(min(vals), 4), round(max(vals), 4))
        return ci

    def _generate_recommendation(self, distribution: Dict[str, float],
                                  optimal: List[Branch],
                                  catastrophic: List[Branch]) -> str:
        collapse_p = distribution.get("collapse", 0)
        decline_p = distribution.get("decline", 0)
        growth_p = distribution.get("growth", 0) + distribution.get("golden_age", 0)
        parts = []
        if collapse_p > 0.1:
            parts.append(f"CRITICAL: {collapse_p:.1%} collapse probability — immediate intervention required.")
        if decline_p > 0.2:
            parts.append(f"WARNING: {decline_p:.1%} decline probability — reduce risk exposure.")
        if growth_p > 0.3:
            parts.append(f"OPPORTUNITY: {growth_p:.1%} growth/golden-age paths identified.")
        if optimal and catastrophic:
            parts.append(f"Key insight: {len(optimal)} optimal paths vs {len(catastrophic)} catastrophic — focus on widening the gap.")
        if not parts:
            parts.append("System stable. Maintain current trajectory.")
        return " ".join(parts)
