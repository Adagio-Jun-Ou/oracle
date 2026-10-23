"""Robust Optimizer — Minimax and distributionally robust optimization for predictions.

Finds predictions that perform well under the worst-case distribution shift:
1. Minimax optimization (worst-case over uncertainty set)
2. Distributionally robust (Wasserstein ball around empirical distribution)
3. Robust Bayesian (prior-robust posterior)
4. Pareto-robust (multi-objective with robustness as objective)
5. Scenario-robust (optimize across multiple scenarios)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RobustMethod(Enum):
    MINIMAX = "minimax"
    DISTRIBUTIONALLY_ROBUST = "distributionally_robust"
    ROBUST_BAYESIAN = "robust_bayesian"
    PARETO_ROBUST = "pareto_robust"
    SCENARIO_ROBUST = "scenario_robust"


@dataclass
class UncertaintySet:
    center: Dict[str, float]
    radius: float
    variables: List[str]
    bounds: Dict[str, Tuple[float, float]]


@dataclass
class RobustSolution:
    decision: Dict[str, float]
    worst_case_loss: float
    nominal_loss: float
    robustness_premium: float
    method: RobustMethod
    scenario_losses: Dict[str, float]


@dataclass
class ParetoRobustSolution:
    solutions: List[RobustSolution]
    pareto_front: List[Tuple[float, float]]
    knee_point: Optional[RobustSolution]


@dataclass
class RobustOptimizationResult:
    best_solution: RobustSolution
    all_solutions: List[RobustSolution]
    robustness_score: float
    nominal_performance: float
    worst_case_performance: float
    price_of_robustness: float
    recommendations: List[str]


class RobustOptimizer:
    def __init__(self, n_scenarios: int = 50, n_iterations: int = 200,
                 uncertainty_radius: float = 0.1):
        self.n_scenarios = n_scenarios
        self.n_iterations = n_iterations
        self.uncertainty_radius = uncertainty_radius
        self._rng = np.random.default_rng(42)

    def optimize(self, predictions: Dict[str, float],
                  loss_function: Optional[object] = None,
                  method: RobustMethod = RobustMethod.MINIMAX,
                  constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> RobustOptimizationResult:
        if loss_function is None:
            loss_function = self._default_loss
        uncertainty_set = self._build_uncertainty_set(predictions, constraints)
        if method == RobustMethod.MINIMAX:
            solution = self._minimax_optimize(predictions, uncertainty_set, loss_function)
        elif method == RobustMethod.DISTRIBUTIONALLY_ROBUST:
            solution = self._dro_optimize(predictions, uncertainty_set, loss_function)
        elif method == RobustMethod.ROBUST_BAYESIAN:
            solution = self._robust_bayesian_optimize(predictions, uncertainty_set, loss_function)
        elif method == RobustMethod.SCENARIO_ROBUST:
            solution = self._scenario_robust_optimize(predictions, uncertainty_set, loss_function)
        else:
            solution = self._pareto_robust_optimize(predictions, uncertainty_set, loss_function)
        all_solutions = self._multi_method_optimize(predictions, uncertainty_set, loss_function)
        nominal_loss = self._eval_loss(predictions, predictions, loss_function)
        por = abs(solution.worst_case_loss - nominal_loss) / max(abs(nominal_loss), 0.01)
        recs = self._generate_recommendations(solution, nominal_loss, por)
        return RobustOptimizationResult(
            best_solution=solution,
            all_solutions=all_solutions,
            robustness_score=round(max(0, 1.0 - por), 4),
            nominal_performance=round(nominal_loss, 4),
            worst_case_performance=round(solution.worst_case_loss, 4),
            price_of_robustness=round(por, 4),
            recommendations=recs,
        )

    def _build_uncertainty_set(self, predictions: Dict[str, float],
                                 constraints: Optional[Dict[str, Tuple[float, float]]]) -> UncertaintySet:
        bounds = {}
        for var, val in predictions.items():
            lo = val * (1 - self.uncertainty_radius)
            hi = val * (1 + self.uncertainty_radius)
            if constraints and var in constraints:
                lo = max(lo, constraints[var][0])
                hi = min(hi, constraints[var][1])
            bounds[var] = (lo, hi)
        return UncertaintySet(
            center=dict(predictions),
            radius=self.uncertainty_radius,
            variables=list(predictions.keys()),
            bounds=bounds,
        )

    def _default_loss(self, decision: Dict[str, float], realization: Dict[str, float]) -> float:
        loss = 0.0
        for var in decision:
            if var in realization:
                diff = decision.get(var, 0) - realization.get(var, 0)
                loss += diff * diff
        return loss

    def _eval_loss(self, decision: Dict[str, float], realization: Dict[str, float],
                    loss_fn) -> float:
        if callable(loss_fn):
            try:
                return float(loss_fn(decision, realization))
            except Exception:
                return self._default_loss(decision, realization)
        return self._default_loss(decision, realization)

    def _sample_scenarios(self, uncertainty_set: UncertaintySet) -> List[Dict[str, float]]:
        scenarios = []
        for _ in range(self.n_scenarios):
            s = {}
            for var in uncertainty_set.variables:
                lo, hi = uncertainty_set.bounds.get(var, (0, 1))
                s[var] = self._rng.uniform(lo, hi)
            scenarios.append(s)
        return scenarios

    def _minimax_optimize(self, predictions: Dict[str, float],
                            uncertainty_set: UncertaintySet,
                            loss_fn) -> RobustSolution:
        best_decision = None
        best_worst_loss = float('inf')
        best_nominal = 0.0
        best_scenario_losses = {}
        center = uncertainty_set.center
        for i in range(self.n_iterations):
            decision = {}
            for var in uncertainty_set.variables:
                lo, hi = uncertainty_set.bounds.get(var, (0, 1))
                t = i / max(self.n_iterations - 1, 1)
                noise = self._rng.uniform(-0.05, 0.05)
                val = center[var] + (1 - 2 * t) * self.uncertainty_radius * center.get(var, 0.5) * (1 + noise)
                val = max(lo, min(hi, val))
                decision[var] = val
            scenarios = self._sample_scenarios(uncertainty_set)
            losses = {}
            worst_loss = -float('inf')
            for idx, sc in enumerate(scenarios):
                l = self._eval_loss(decision, sc, loss_fn)
                losses[f"scenario_{idx}"] = l
                if l > worst_loss:
                    worst_loss = l
            if worst_loss < best_worst_loss:
                best_worst_loss = worst_loss
                best_decision = decision
                best_nominal = self._eval_loss(decision, center, loss_fn)
                best_scenario_losses = losses
        if best_decision is None:
            best_decision = dict(predictions)
            best_worst_loss = 0.0
            best_nominal = 0.0
        return RobustSolution(
            decision={k: round(v, 4) for k, v in best_decision.items()},
            worst_case_loss=round(best_worst_loss, 4),
            nominal_loss=round(best_nominal, 4),
            robustness_premium=round(abs(best_worst_loss - best_nominal), 4),
            method=RobustMethod.MINIMAX,
            scenario_losses={k: round(v, 4) for k, v in best_scenario_losses.items()},
        )

    def _dro_optimize(self, predictions: Dict[str, float],
                        uncertainty_set: UncertaintySet,
                        loss_fn) -> RobustSolution:
        best_decision = None
        best_worst_loss = float('inf')
        best_nominal = 0.0
        best_scenario_losses = {}
        center = uncertainty_set.center
        n_dRO_samples = max(20, self.n_scenarios // 2)
        for i in range(self.n_iterations):
            decision = {}
            for var in uncertainty_set.variables:
                lo, hi = uncertainty_set.bounds.get(var, (0, 1))
                t = i / max(self.n_iterations - 1, 1)
                val = center[var] * (1 + self.uncertainty_radius * (0.5 - t) * 2)
                val = max(lo, min(hi, val))
                decision[var] = val
            losses = {}
            worst_loss = -float('inf')
            for j in range(n_dRO_samples):
                sc = {}
                for var in uncertainty_set.variables:
                    center_val = center[var]
                    radius_val = center_val * self.uncertainty_radius
                    sc[var] = center_val + self._rng.uniform(-radius_val, radius_val)
                l = self._eval_loss(decision, sc, loss_fn)
                lam = 1.0 / (n_dRO_samples * self.uncertainty_radius + 1e-10)
                penalized_loss = l + lam * max(0, l - self._eval_loss(decision, center, loss_fn))
                losses[f"dro_{j}"] = penalized_loss
                if penalized_loss > worst_loss:
                    worst_loss = penalized_loss
            if worst_loss < best_worst_loss:
                best_worst_loss = worst_loss
                best_decision = decision
                best_nominal = self._eval_loss(decision, center, loss_fn)
                best_scenario_losses = losses
        if best_decision is None:
            best_decision = dict(predictions)
            best_worst_loss = 0.0
            best_nominal = 0.0
        return RobustSolution(
            decision={k: round(v, 4) for k, v in best_decision.items()},
            worst_case_loss=round(best_worst_loss, 4),
            nominal_loss=round(best_nominal, 4),
            robustness_premium=round(abs(best_worst_loss - best_nominal), 4),
            method=RobustMethod.DISTRIBUTIONALLY_ROBUST,
            scenario_losses={k: round(v, 4) for k, v in best_scenario_losses.items()},
        )

    def _robust_bayesian_optimize(self, predictions: Dict[str, float],
                                    uncertainty_set: UncertaintySet,
                                    loss_fn) -> RobustSolution:
        center = uncertainty_set.center
        prior_center = dict(center)
        posterior_samples = self.n_scenarios
        best_decision = None
        best_worst_loss = float('inf')
        best_nominal = 0.0
        best_scenario_losses = {}
        for i in range(self.n_iterations):
            decision = {}
            for var in uncertainty_set.variables:
                lo, hi = uncertainty_set.bounds.get(var, (0, 1))
                blend = 0.5 + 0.5 * (i / max(self.n_iterations - 1, 1))
                val = prior_center.get(var, 0.5) * (1 - blend) + center[var] * blend
                val = max(lo, min(hi, val))
                decision[var] = val
            losses = {}
            worst_loss = -float('inf')
            for j in range(posterior_samples):
                sc = {}
                for var in uncertainty_set.variables:
                    sc[var] = self._rng.normal(center[var], center[var] * self.uncertainty_radius * 0.5)
                    lo, hi = uncertainty_set.bounds.get(var, (0, 1))
                    sc[var] = max(lo, min(hi, sc[var]))
                l = self._eval_loss(decision, sc, loss_fn)
                losses[f"bayes_{j}"] = l
                if l > worst_loss:
                    worst_loss = l
            if worst_loss < best_worst_loss:
                best_worst_loss = worst_loss
                best_decision = decision
                best_nominal = self._eval_loss(decision, center, loss_fn)
                best_scenario_losses = losses
        if best_decision is None:
            best_decision = dict(predictions)
            best_worst_loss = 0.0
            best_nominal = 0.0
        return RobustSolution(
            decision={k: round(v, 4) for k, v in best_decision.items()},
            worst_case_loss=round(best_worst_loss, 4),
            nominal_loss=round(best_nominal, 4),
            robustness_premium=round(abs(best_worst_loss - best_nominal), 4),
            method=RobustMethod.ROBUST_BAYESIAN,
            scenario_losses={k: round(v, 4) for k, v in best_scenario_losses.items()},
        )

    def _scenario_robust_optimize(self, predictions: Dict[str, float],
                                    uncertainty_set: UncertaintySet,
                                    loss_fn) -> RobustSolution:
        center = uncertainty_set.center
        named_scenarios = {
            "optimistic": {k: v * (1 + self.uncertainty_radius) for k, v in center.items()},
            "nominal": dict(center),
            "pessimistic": {k: v * (1 - self.uncertainty_radius) for k, v in center.items()},
            "tail_risk": {k: v * (1 - 2 * self.uncertainty_radius) for k, v in center.items()},
            "stress": {k: v * (1 - 3 * self.uncertainty_radius) for k, v in center.items()},
        }
        for sc_name in named_scenarios:
            for var in named_scenarios[sc_name]:
                if var in uncertainty_set.bounds:
                    lo, hi = uncertainty_set.bounds[var]
                    named_scenarios[sc_name][var] = max(lo, min(hi, named_scenarios[sc_name][var]))
        best_decision = None
        best_avg_loss = float('inf')
        best_worst = 0.0
        best_nominal = 0.0
        best_scenario_losses = {}
        for i in range(self.n_iterations):
            decision = {}
            for var in uncertainty_set.variables:
                lo, hi = uncertainty_set.bounds.get(var, (0, 1))
                t = i / max(self.n_iterations - 1, 1)
                val = center[var] * (1 - self.uncertainty_radius + 2 * self.uncertainty_radius * t)
                val = max(lo, min(hi, val))
                decision[var] = val
            losses = {}
            for sc_name, sc in named_scenarios.items():
                losses[sc_name] = self._eval_loss(decision, sc, loss_fn)
            avg_loss = sum(losses.values()) / max(len(losses), 1)
            worst = max(losses.values()) if losses else 0
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                best_decision = decision
                best_worst = worst
                best_nominal = losses.get("nominal", 0)
                best_scenario_losses = losses
        if best_decision is None:
            best_decision = dict(predictions)
            best_worst = 0.0
            best_nominal = 0.0
            best_scenario_losses = {}
        return RobustSolution(
            decision={k: round(v, 4) for k, v in best_decision.items()},
            worst_case_loss=round(best_worst, 4),
            nominal_loss=round(best_nominal, 4),
            robustness_premium=round(abs(best_worst - best_nominal), 4),
            method=RobustMethod.SCENARIO_ROBUST,
            scenario_losses={k: round(v, 4) for k, v in best_scenario_losses.items()},
        )

    def _pareto_robust_optimize(self, predictions: Dict[str, float],
                                  uncertainty_set: UncertaintySet,
                                  loss_fn) -> RobustSolution:
        center = uncertainty_set.center
        solutions = []
        pareto_front = []
        for alpha in np.linspace(0, 1, 20):
            decision = {}
            for var in uncertainty_set.variables:
                lo, hi = uncertainty_set.bounds.get(var, (0, 1))
                nominal_val = center[var]
                robust_val = center[var] * (1 - self.uncertainty_radius * alpha)
                robust_val = max(lo, min(hi, robust_val))
                decision[var] = robust_val
            nom_loss = self._eval_loss(decision, center, loss_fn)
            scenarios = self._sample_scenarios(uncertainty_set)[:10]
            worst_loss = max(self._eval_loss(decision, sc, loss_fn) for sc in scenarios) if scenarios else nom_loss
            sol = RobustSolution(
                decision={k: round(v, 4) for k, v in decision.items()},
                worst_case_loss=round(worst_loss, 4),
                nominal_loss=round(nom_loss, 4),
                robustness_premium=round(abs(worst_loss - nom_loss), 4),
                method=RobustMethod.PARETO_ROBUST,
                scenario_losses={},
            )
            solutions.append(sol)
            pareto_front.append((round(nom_loss, 4), round(worst_loss, 4)))
        if solutions:
            best_idx = min(range(len(solutions)),
                          key=lambda i: solutions[i].worst_case_loss * 0.6 + solutions[i].nominal_loss * 0.4)
            return solutions[best_idx]
        return RobustSolution(
            decision={k: round(v, 4) for k, v in predictions.items()},
            worst_case_loss=0.0, nominal_loss=0.0, robustness_premium=0.0,
            method=RobustMethod.PARETO_ROBUST, scenario_losses={},
        )

    def _multi_method_optimize(self, predictions: Dict[str, float],
                                 uncertainty_set: UncertaintySet,
                                 loss_fn) -> List[RobustSolution]:
        solutions = []
        for method in [RobustMethod.MINIMAX, RobustMethod.DISTRIBUTIONALLY_ROBUST,
                        RobustMethod.ROBUST_BAYESIAN, RobustMethod.SCENARIO_ROBUST]:
            if method == RobustMethod.MINIMAX:
                sol = self._minimax_optimize(predictions, uncertainty_set, loss_fn)
            elif method == RobustMethod.DISTRIBUTIONALLY_ROBUST:
                sol = self._dro_optimize(predictions, uncertainty_set, loss_fn)
            elif method == RobustMethod.ROBUST_BAYESIAN:
                sol = self._robust_bayesian_optimize(predictions, uncertainty_set, loss_fn)
            else:
                sol = self._scenario_robust_optimize(predictions, uncertainty_set, loss_fn)
            solutions.append(sol)
        return solutions

    def _generate_recommendations(self, solution: RobustSolution,
                                   nominal_loss: float,
                                   price_of_robustness: float) -> List[str]:
        recs = []
        if price_of_robustness > 0.5:
            recs.append(f"HIGH PRICE OF ROBUSTNESS ({price_of_robustness:.0%}): robust solution significantly sacrifices nominal performance")
        elif price_of_robustness > 0.2:
            recs.append(f"MODERATE PRICE OF ROBUSTNESS ({price_of_robustness:.0%}): acceptable trade-off between performance and safety")
        else:
            recs.append(f"LOW PRICE OF ROBUSTNESS ({price_of_robustness:.0%}): robust solution nearly matches nominal performance")
        if solution.worst_case_loss > 2 * nominal_loss:
            recs.append("WORST CASE is 2x+ nominal — high tail risk, consider reducing exposure")
        if solution.robustness_premium > nominal_loss:
            recs.append("Robustness premium exceeds nominal loss — uncertainty dominates the decision")
        recs.append(f"Method: {solution.method.value} — worst case {solution.worst_case_loss:.4f}, nominal {nominal_loss:.4f}")
        return recs