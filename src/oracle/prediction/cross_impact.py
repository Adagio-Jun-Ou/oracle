"""Cross-Impact Analysis — Probabilistic cross-impact matrices for scenario generation.

Implements the cross-impact method developed by Theodore Gordon and Olaf Helmer
at RAND: estimate how the occurrence of one event affects the probability of
other events, then solve the resulting system for self-consistent probabilities.

Key innovations:
1. Cross-impact matrix with conditional probability adjustments
2. Iterative convergence solver for self-consistent probabilities
3. Scenario generation from consistent probability sets
4. Sensitivity analysis on cross-impact coefficients
5. Monte Carlo cross-impact simulation
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import logging
import math
import itertools

logger = logging.getLogger(__name__)


class ImpactDirection(Enum):
    INHIBITING = "inhibiting"
    NEUTRAL = "neutral"
    FACILITATING = "facilitating"


class EventStatus(Enum):
    DID_OCCUR = "did_occur"
    DID_NOT_OCCUR = "did_not_occur"
    UNCERTAIN = "uncertain"


@dataclass
class CrossImpactEvent:
    name: str
    initial_probability: float
    description: str = ""
    domain: str = ""


@dataclass
class CrossImpactPair:
    event_i: str
    event_j: str
    impact_value: float
    direction: ImpactDirection
    rationale: str = ""


@dataclass
class ScenarioProbability:
    events: Dict[str, bool]
    joint_probability: float
    description: str


@dataclass
class CrossImpactResult:
    events: List[CrossImpactEvent]
    initial_probabilities: Dict[str, float]
    revised_probabilities: Dict[str, float]
    probability_changes: Dict[str, float]
    cross_impact_matrix: Dict[str, Dict[str, float]]
    top_scenarios: List[ScenarioProbability]
    most_surprising: List[str]
    key_drivers: List[str]
    key_inhibitors: List[str]
    convergence_iterations: int
    sensitivity_ranking: Dict[str, float]


DEFAULT_EVENTS: List[CrossImpactEvent] = [
    CrossImpactEvent("recession", 0.20, "Major economic recession", "economic"),
    CrossImpactEvent("high_inflation", 0.25, "Persistent high inflation (>5%)", "economic"),
    CrossImpactEvent("geopolitical_crisis", 0.15, "Major geopolitical crisis", "geopolitical"),
    CrossImpactEvent("tech_breakthrough", 0.30, "Significant technology breakthrough", "technology"),
    CrossImpactEvent("climate_disaster", 0.20, "Major climate-related disaster", "climate"),
    CrossImpactEvent("pandemic_resurgence", 0.10, "Pandemic resurgence", "health"),
    CrossImpactEvent("financial_crisis", 0.12, "Banking/financial system crisis", "financial"),
    CrossImpactEvent("trade_war", 0.25, "Escalating trade war", "economic"),
    CrossImpactEvent("social_unrest", 0.18, "Major social unrest", "social"),
    CrossImpactEvent("energy_crisis", 0.15, "Severe energy supply crisis", "energy"),
    CrossImpactEvent("ai_regulation", 0.40, "Major AI regulation enacted", "technology"),
    CrossImpactEvent("political_instability", 0.22, "Political instability in major nations", "political"),
]

DEFAULT_IMPACTS: List[CrossImpactPair] = [
    CrossImpactPair("recession", "financial_crisis", 0.6, ImpactDirection.FACILITATING, "Recession triggers financial stress"),
    CrossImpactPair("recession", "social_unrest", 0.4, ImpactDirection.FACILITATING, "Economic hardship breeds unrest"),
    CrossImpactPair("recession", "trade_war", 0.3, ImpactDirection.FACILITATING, "Recession increases protectionism"),
    CrossImpactPair("high_inflation", "recession", 0.3, ImpactDirection.FACILITATING, "Inflation can trigger recession"),
    CrossImpactPair("high_inflation", "social_unrest", 0.35, ImpactDirection.FACILITATING, "Inflation erodes purchasing power"),
    CrossImpactPair("geopolitical_crisis", "trade_war", 0.5, ImpactDirection.FACILITATING, "Geopolitical tension increases trade barriers"),
    CrossImpactPair("geopolitical_crisis", "energy_crisis", 0.4, ImpactDirection.FACILITATING, "Conflicts disrupt energy supplies"),
    CrossImpactPair("geopolitical_crisis", "recession", 0.25, ImpactDirection.FACILITATING, "Geopolitical shocks hurt growth"),
    CrossImpactPair("tech_breakthrough", "recession", 0.2, ImpactDirection.INHIBITING, "Tech growth offsets recession"),
    CrossImpactPair("tech_breakthrough", "social_unrest", 0.15, ImpactDirection.INHIBITING, "Tech creates opportunities"),
    CrossImpactPair("tech_breakthrough", "ai_regulation", 0.35, ImpactDirection.FACILITATING, "Breakthroughs prompt regulation"),
    CrossImpactPair("climate_disaster", "social_unrest", 0.45, ImpactDirection.FACILITATING, "Disaster response failures"),
    CrossImpactPair("climate_disaster", "political_instability", 0.3, ImpactDirection.FACILITATING, "Government failures"),
    CrossImpactPair("climate_disaster", "recession", 0.25, ImpactDirection.FACILITATING, "Economic destruction"),
    CrossImpactPair("pandemic_resurgence", "recession", 0.5, ImpactDirection.FACILITATING, "Pandemic hurts economy"),
    CrossImpactPair("pandemic_resurgence", "social_unrest", 0.4, ImpactDirection.FACILITATING, "Lockdown fatigue"),
    CrossImpactPair("financial_crisis", "recession", 0.55, ImpactDirection.FACILITATING, "Financial crisis causes recession"),
    CrossImpactPair("financial_crisis", "social_unrest", 0.35, ImpactDirection.FACILITATING, "Financial pain → unrest"),
    CrossImpactPair("trade_war", "recession", 0.3, ImpactDirection.FACILITATING, "Trade disruption hurts growth"),
    CrossImpactPair("trade_war", "high_inflation", 0.25, ImpactDirection.FACILITATING, "Tariffs increase prices"),
    CrossImpactPair("social_unrest", "political_instability", 0.4, ImpactDirection.FACILITATING, "Unrest destabilizes"),
    CrossImpactPair("social_unrest", "recession", 0.15, ImpactDirection.FACILITATING, "Unrest hurts business"),
    CrossImpactPair("energy_crisis", "recession", 0.35, ImpactDirection.FACILITATING, "Energy costs drag economy"),
    CrossImpactPair("energy_crisis", "social_unrest", 0.3, ImpactDirection.FACILITATING, "Energy poverty"),
    CrossImpactPair("political_instability", "trade_war", 0.25, ImpactDirection.FACILITATING, "Instability → protectionism"),
    CrossImpactPair("political_instability", "social_unrest", 0.3, ImpactDirection.FACILITATING, "Instability → protests"),
    CrossImpactPair("ai_regulation", "tech_breakthrough", 0.15, ImpactDirection.INHIBITING, "Regulation slows innovation"),
]


class CrossImpactAnalyzer:
    def __init__(self, events: Optional[List[CrossImpactEvent]] = None,
                 impacts: Optional[List[CrossImpactPair]] = None):
        self.events = events or DEFAULT_EVENTS.copy()
        self.impacts = impacts or DEFAULT_IMPACTS.copy()
        self.event_names = [e.name for e in self.events]
        self._matrix: Optional[np.ndarray] = None

    def add_event(self, event: CrossImpactEvent) -> None:
        self.events.append(event)
        self.event_names.append(event.name)
        self._matrix = None

    def add_impact(self, impact: CrossImpactPair) -> None:
        self.impacts.append(impact)
        self._matrix = None

    def build_matrix(self) -> Dict[str, Dict[str, float]]:
        n = len(self.event_names)
        matrix = np.zeros((n, n))
        event_idx = {name: i for i, name in enumerate(self.event_names)}
        for impact in self.impacts:
            if impact.event_i in event_idx and impact.event_j in event_idx:
                i = event_idx[impact.event_i]
                j = event_idx[impact.event_j]
                value = impact.impact_value
                if impact.direction == ImpactDirection.INHIBITING:
                    value = -abs(value)
                matrix[i][j] = value
        self._matrix = matrix
        result = {}
        for i, name_i in enumerate(self.event_names):
            result[name_i] = {}
            for j, name_j in enumerate(self.event_names):
                if matrix[i][j] != 0:
                    result[name_i][name_j] = round(float(matrix[i][j]), 4)
        return result

    def solve(self, max_iterations: int = 50, tolerance: float = 0.001) -> CrossImpactResult:
        if self._matrix is None:
            self.build_matrix()
        n = len(self.event_names)
        initial_probs = np.array([e.initial_probability for e in self.events])
        revised = initial_probs.copy()
        if self._matrix is None:
            self.build_matrix()
        matrix = self._matrix
        converged = False
        iterations = 0
        for iteration in range(max_iterations):
            old_probs = revised.copy()
            for j in range(n):
                adjustment = 0.0
                incoming = 0
                for i in range(n):
                    if i == j:
                        continue
                    if matrix[i][j] > 0:
                        adjustment += matrix[i][j] * old_probs[i] * (1 - old_probs[j])
                        incoming += 1
                    elif matrix[i][j] < 0:
                        adjustment += matrix[i][j] * old_probs[i] * old_probs[j]
                        incoming += 1
                dampening = 0.2 / max(incoming, 1)
                step = adjustment * dampening
                max_step = 0.1
                step = max(-max_step, min(max_step, step))
                revised[j] = old_probs[j] + step
                revised[j] = max(0.01, min(0.99, revised[j]))
            max_change = np.max(np.abs(revised - old_probs))
            if max_change < tolerance:
                converged = True
                iterations = iteration + 1
                break
            iterations = iteration + 1
        if not converged:
            iterations = max_iterations
        prob_changes = {name: round(float(revised[i] - initial_probs[i]), 4)
                       for i, name in enumerate(self.event_names)}
        initial_dict = {name: round(float(initial_probs[i]), 4) for i, name in enumerate(self.event_names)}
        revised_dict = {name: round(float(revised[i]), 4) for i, name in enumerate(self.event_names)}
        matrix_dict = self.build_matrix()
        scenarios = self._generate_scenarios(revised, n_scenarios=5)
        surprising = self._find_surprising(revised, initial_probs)
        drivers, inhibitors = self._find_key_factors(matrix)
        sensitivity = self._sensitivity_analysis(revised)
        return CrossImpactResult(
            events=self.events, initial_probabilities=initial_dict,
            revised_probabilities=revised_dict, probability_changes=prob_changes,
            cross_impact_matrix=matrix_dict, top_scenarios=scenarios,
            most_surprising=surprising, key_drivers=drivers, key_inhibitors=inhibitors,
            convergence_iterations=iterations, sensitivity_ranking=sensitivity,
        )

    def _generate_scenarios(self, probs: np.ndarray, n_scenarios: int = 5) -> List[ScenarioProbability]:
        n = len(self.event_names)
        if n > 15:
            scenarios = []
            for _ in range(n_scenarios * 10):
                realization = np.random.random(n) < probs
                log_prob = 0.0
                for j in range(n):
                    p = probs[j] if realization[j] else (1 - probs[j])
                    log_prob += math.log(max(p, 1e-10))
                scenarios.append((realization, math.exp(log_prob)))
            scenarios.sort(key=lambda x: x[1], reverse=True)
            unique = []
            seen = set()
            for real, prob in scenarios:
                key = tuple(real)
                if key not in seen:
                    seen.add(key)
                    unique.append(ScenarioProbability(
                        events={self.event_names[i]: bool(real[i]) for i in range(n)},
                        joint_probability=round(prob, 6),
                        description=self._describe_scenario(real),
                    ))
                if len(unique) >= n_scenarios:
                    break
            return unique
        scenarios = []
        for combo in itertools.product([True, False], repeat=n):
            log_prob = 0.0
            for j, occurs in enumerate(combo):
                p = probs[j] if occurs else (1 - probs[j])
                log_prob += math.log(max(p, 1e-10))
            scenarios.append((combo, math.exp(log_prob), combo))
        scenarios.sort(key=lambda x: x[1], reverse=True)
        top = scenarios[:n_scenarios]
        return [ScenarioProbability(
            events={self.event_names[i]: bool(combo[i]) for i in range(n)},
            joint_probability=round(prob, 6),
            description=self._describe_scenario(np.array(combo)),
        ) for combo, prob, _ in top]

    def _describe_scenario(self, realization: np.ndarray) -> str:
        occurring = [self.event_names[i] for i in range(len(realization)) if realization[i]]
        not_occurring = [self.event_names[i] for i in range(len(realization)) if not realization[i]]
        parts = []
        if occurring:
            parts.append(f"Occurs: {', '.join(occurring[:3])}")
        if not_occurring:
            parts.append(f"Does not: {', '.join(not_occurring[:3])}")
        return "; ".join(parts) if parts else "Empty scenario"

    def _find_surprising(self, revised: np.ndarray, initial: np.ndarray) -> List[str]:
        surprising = []
        for i, name in enumerate(self.event_names):
            change = revised[i] - initial[i]
            if abs(change) > 0.05:
                surprising.append((name, float(initial[i]), float(revised[i]), float(change)))
        surprising.sort(key=lambda x: abs(x[3]), reverse=True)
        return [f"{x[0]}: {x[1]:.2f} → {x[2]:.2f} ({x[3]:+.2f})" for x in surprising[:5]]

    def _find_key_factors(self, matrix: np.ndarray) -> Tuple[List[str], List[str]]:
        if matrix is None:
            return [], []
        drivers = []
        inhibitors = []
        for i, name in enumerate(self.event_names):
            pos_impact = float(np.sum(np.maximum(matrix[i], 0)))
            neg_impact = float(np.sum(np.minimum(matrix[i], 0)))
            if pos_impact > 0.3:
                drivers.append(f"{name} (+{pos_impact:.2f})")
            if neg_impact < -0.3:
                inhibitors.append(f"{name} ({neg_impact:.2f})")
        drivers.sort(key=lambda x: float(x.split("+")[1].rstrip(")")), reverse=True)
        inhibitors.sort(key=lambda x: float(x.split("(")[1].rstrip(")")))
        return drivers[:5], inhibitors[:5]

    def _sensitivity_analysis(self, probs: np.ndarray) -> Dict[str, float]:
        if self._matrix is None:
            return {}
        n = len(self.event_names)
        sensitivity = {}
        for i, name in enumerate(self.event_names):
            row_impact = float(np.sum(np.abs(self._matrix[i])))
            col_impact = float(np.sum(np.abs(self._matrix[:, i])))
            total = row_impact + col_impact
            sensitivity[name] = round(total / max(n, 1), 4)
        return dict(sorted(sensitivity.items(), key=lambda x: x[1], reverse=True))

    def monte_carlo(self, n_simulations: int = 1000) -> Dict[str, float]:
        if self._matrix is None:
            self.build_matrix()
        result = self.solve()
        probs = np.array([result.revised_probabilities.get(name, 0.5) for name in self.event_names])
        co_occurrence = np.zeros(len(self.event_names))
        for _ in range(n_simulations):
            realization = np.random.random(len(probs)) < probs
            co_occurrence += realization
        return {name: round(float(co_occurrence[i] / n_simulations), 4) for i, name in enumerate(self.event_names)}