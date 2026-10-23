# -*- coding: utf-8 -*-
"""Optimal Path Optimizer — Multi-objective future trajectory optimization.

Inspired by Rehoboam's function in Westworld: compute the globally optimal path
through future state space that maximizes desired outcomes while minimizing risk.
Uses Pareto optimization over multi-objective trajectories with constraint handling.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
import math
import random


class ObjectiveDirection(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class Objective:
    name: str
    direction: ObjectiveDirection
    weight: float
    transform: Optional[str] = None  # "log", "sqrt"


@dataclass
class Constraint:
    name: str
    lhs_var: str
    comparator: str  # "<=", ">=", "=="
    rhs: float


@dataclass
class Path:
    trajectory: List[Dict[str, float]]
    objective_values: Dict[str, float]
    constraint_satisfaction: Dict[str, bool]
    dominance_score: float  # number of paths this one dominates
    regret: float  # distance from utopia point
    robustness: float  # sensitivity to perturbation
    path_id: str = ""


@dataclass
class OptimalPathResult:
    pareto_frontier: List[Path]
    utopia_point: Dict[str, float]
    anti_utopia: Dict[str, float]
    recommended_paths: List[Path]
    nash_point: Dict[str, float]
    kalai_smorodinsky_point: Dict[str, float]
    equity_score: float
    robustness_analysis: Dict[str, float]
    recommendation: str


class OptimalPathOptimizer:
    """Compute optimal future trajectories via Pareto optimization.

    Like Rehoboam computing the optimal path for humanity: maximize growth,
    minimize conflict, subject to resource constraints. Identifies the Nash
    point (stable equilibrium), Kalai-Smorodinsky (fairness), and
    Pareto-optimal frontiers.
    """

    def __init__(self, objectives: List[Objective],
                 constraints: List[Constraint],
                 n_paths: int = 500,
                 max_steps: int = 15,
                 seed: int = 42):
        self.objectives = objectives
        self.constraints = constraints
        self.n_paths = n_paths
        self.max_steps = max_steps
        self.rng = random.Random(seed)

    def optimize(self, initial_state: Dict[str, float],
                 transition_fn: Optional[Callable] = None) -> OptimalPathResult:
        """Generate and rank multi-objective paths."""
        paths = []
        for i in range(self.n_paths):
            path = self._generate_path(initial_state, transition_fn, path_id=f"P{i}")
            paths.append(path)
        # Normalize objectives
        normalized = self._normalize_objectives(paths)
        # Pareto frontier
        pareto = self._compute_pareto_frontier(normalized)
        # Utopia / Anti-utopia
        utopia = self._compute_utopia(pareto)
        anti = self._compute_anti_utopia(pareto)
        # Regret for each path
        for p in pareto:
            p.regret = self._compute_regret(p, utopia, anti)
        # Robustness: perturb and re-evaluate
        for p in pareto:
            p.robustness = self._evaluate_robustness(p, initial_state, transition_fn)
        # Recommended paths: low regret + high robustness
        pareto.sort(key=lambda p: (p.regret, -p.robustness))
        recommended = pareto[:5]
        # Bargaining solutions
        nash = self._compute_nash(pareto)
        ks = self._compute_kalai_smorodinsky(pareto, utopia, anti)
        equity = self._compute_equity(pareto)
        robustness = self._compute_global_robustness(pareto)
        rec = self._generate_recommendation(recommended, nash, ks)
        return OptimalPathResult(
            pareto_frontier=pareto[:15],
            utopia_point={k: round(v, 4) for k, v in utopia.items()},
            anti_utopia={k: round(v, 4) for k, v in anti.items()},
            recommended_paths=recommended,
            nash_point={k: round(v, 4) for k, v in nash.items()},
            kalai_smorodinsky_point={k: round(v, 4) for k, v in ks.items()},
            equity_score=round(equity, 4),
            robustness_analysis={k: round(v, 4) for k, v in robustness.items()},
            recommendation=rec,
        )

    def _generate_path(self, initial: Dict[str, float],
                       transition_fn: Optional[Callable],
                       path_id: str) -> Path:
        trajectory = [dict(initial)]
        state = dict(initial)
        for _ in range(self.max_steps):
            if transition_fn:
                state = transition_fn(state)
            else:
                state = self._default_transition(state)
            trajectory.append(dict(state))
        obj_vals = self._evaluate_objectives(trajectory)
        constraints = self._evaluate_constraints(trajectory)
        return Path(
            trajectory=trajectory,
            objective_values=obj_vals,
            constraint_satisfaction=constraints,
            dominance_score=0.0,
            regret=0.0,
            robustness=0.0,
            path_id=path_id,
        )

    def _default_transition(self, state: Dict[str, float]) -> Dict[str, float]:
        result = {}
        for k, v in state.items():
            delta = self.rng.gauss(0, 0.03)
            result[k] = float(np.clip(v + delta, 0.0, 1.0))
        return result

    def _evaluate_objectives(self, trajectory: List[Dict[str, float]]) -> Dict[str, float]:
        values = {}
        final = trajectory[-1]
        for obj in self.objectives:
            val = final.get(obj.name, 0.5)
            if obj.transform == "log":
                val = math.log(max(val, 0.001))
            elif obj.transform == "sqrt":
                val = math.sqrt(max(val, 0))
            if obj.direction == ObjectiveDirection.MINIMIZE:
                val = -val  # Invert for maximization
            values[obj.name] = val
        return values

    def _evaluate_constraints(self, trajectory: List[Dict[str, float]]) -> Dict[str, bool]:
        satisfied = {}
        final = trajectory[-1]
        for c in self.constraints:
            lhs = final.get(c.lhs_var, 0.5)
            if c.comparator == "<=":
                satisfied[c.name] = lhs <= c.rhs
            elif c.comparator == ">=":
                satisfied[c.name] = lhs >= c.rhs
            else:
                satisfied[c.name] = abs(lhs - c.rhs) < 0.05
        return satisfied

    def _normalize_objectives(self, paths: List[Path]) -> List[Path]:
        obj_names = [o.name for o in self.objectives]
        for name in obj_names:
            vals = [p.objective_values.get(name, 0) for p in paths]
            lo, hi = min(vals), max(vals)
            for p in paths:
                v = p.objective_values.get(name, 0)
                p.objective_values[name] = (v - lo) / max(hi - lo, 0.001)
        return paths

    def _compute_pareto_frontier(self, paths: List[Path]) -> List[Path]:
        obj_names = [o.name for o in self.objectives]
        pareto = []
        for p in paths:
            dominated = False
            for other in paths:
                if other is not p:
                    if all(other.objective_values.get(o, 0) >= p.objective_values.get(o, 0) for o in obj_names) and \
                       any(other.objective_values.get(o, 0) > p.objective_values.get(o, 0) for o in obj_names):
                        dominated = True
                        break
            if not dominated:
                pareto.append(p)
        for p in pareto:
            p.dominance_score = sum(
                1 for other in paths if all(
                    p.objective_values.get(o, 0) >= other.objective_values.get(o, 0) for o in obj_names
                ) and any(
                    p.objective_values.get(o, 0) > other.objective_values.get(o, 0) for o in obj_names
                )
            )
        return pareto

    def _compute_utopia(self, paths: List[Path]) -> Dict[str, float]:
        return {obj.name: max(p.objective_values.get(obj.name, 0) for p in paths) for obj in self.objectives}

    def _compute_anti_utopia(self, paths: List[Path]) -> Dict[str, float]:
        return {obj.name: min(p.objective_values.get(obj.name, 0) for p in paths) for obj in self.objectives}

    def _compute_regret(self, path: Path, utopia: Dict[str, float],
                        anti: Dict[str, float]) -> float:
        distance = 0
        for name in self.objectives:
            v = path.objective_values.get(name.name, 0)
            ideal = utopia.get(name.name, 1)
            worst = anti.get(name.name, 0)
            distance += ((ideal - v) / max(ideal - worst, 0.001)) ** 2
        return round(math.sqrt(distance), 4)

    def _evaluate_robustness(self, path: Path, initial: Dict[str, float],
                             transition_fn: Optional[Callable],
                             n_perturb: int = 3) -> float:
        """Re-simulate with perturbations and measure objective variance."""
        obj_names = [o.name for o in self.objectives]
        variances = []
        for _ in range(n_perturb):
            perturbed = {k: float(np.clip(v + self.rng.gauss(0, 0.05), 0, 1)) for k, v in initial.items()}
            test_path = self._generate_path(perturbed, transition_fn, "")
            for name in obj_names:
                vals = [path.objective_values.get(name, 0),
                        test_path.objective_values.get(name, 0)]
                variances.append(float(np.var(vals)))
        return max(0.0, 1.0 - float(np.mean(variances)) * 5)

    def _compute_nash(self, paths: List[Path]) -> Dict[str, float]:
        """Product of normalized objectives — maximizing equal gains."""
        obj_names = [o.name for o in self.objectives]
        best = max(paths, key=lambda p: math.prod(
            [p.objective_values.get(n, 0.001) for n in obj_names]
        ))
        return {n: best.objective_values.get(n, 0) for n in obj_names}

    def _compute_kalai_smorodinsky(self, paths: List[Path],
                                    utopia: Dict[str, float],
                                    anti: Dict[str, float]) -> Dict[str, float]:
        """Fairness-maximizing point: equal proportional gains from worst."""
        obj_names = [o.name for o in self.objectives]
        best = max(paths, key=lambda p: min(
            [(p.objective_values.get(n, 0) - anti.get(n, 0)) / max(utopia.get(n, 1) - anti.get(n, 0), 0.001)
             for n in obj_names]
        ))
        return {n: best.objective_values.get(n, 0) for n in obj_names}

    def _compute_equity(self, paths: List[Path]) -> float:
        obj_names = [o.name for o in self.objectives]
        if len(paths) < 2:
            return 0.5
        vals = [[p.objective_values.get(n, 0) for n in obj_names] for p in paths]
        means = [float(np.mean([v[i] for v in vals])) for i in range(len(obj_names))]
        stds = [float(np.std([v[i] for v in vals])) for i in range(len(obj_names))]
        return float(np.mean([1 - min(s/m, 1) for m, s in zip(means, stds)] if any(means) else [0.5]))

    def _compute_global_robustness(self, paths: List[Path]) -> Dict[str, float]:
        return {
            "mean_robustness": round(float(np.mean([p.robustness for p in paths])), 4),
            "min_robustness": round(float(np.min([p.robustness for p in paths])), 4),
            "constraint_satisfaction_rate": round(sum(1 for p in paths if all(p.constraint_satisfaction.values())) / max(len(paths), 1), 4),
        }

    def _generate_recommendation(self, recommended: List[Path],
                                  nash: Dict[str, float],
                                  ks: Dict[str, float]) -> str:
        parts = []
        if recommended:
            parts.append(f"RECOMMENDATION: {len(recommended)} optimal paths identified.")
            best = recommended[0]
            parts.append(f"Best path {best.path_id}: regret={best.regret:.3f}, robustness={best.robustness:.1%}.")
        if all(nash.get(k, 0) > 0.5 for k in nash):
            parts.append("Nash equilibrium is favorable — stable outcomes achievable.")
        else:
            parts.append("Nash equilibrium is mixed — requires bargaining.")
        ks_values = list(ks.values())
        if ks_values and float(np.mean(ks_values)) > 0.5:
            parts.append("Fairness-optimized (Kalai-Smorodinsky) point is strong.")
        return " ".join(parts)
