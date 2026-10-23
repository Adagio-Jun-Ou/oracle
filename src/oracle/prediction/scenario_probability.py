"""Scenario Probability Mapper — Maps narrative scenarios to probability distributions.

Converts qualitative scenarios into quantitative probability distributions:
1. Scenario tree construction (branching factor, depth)
2. Probability assignment (base rates, conditional, joint)
3. Scenario interaction (mutually exclusive, independent, reinforcing)
4. Sensitivity analysis (which assumptions drive probability?)
5. Consequence scoring (impact x probability = expected value)
6. Scenario aggregation (envelope distribution)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ScenarioRelation(Enum):
    INDEPENDENT = "independent"
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    REINFORCING = "reinforcing"
    OPPOSING = "opposing"
    CONDITIONAL = "conditional"


@dataclass
class ScenarioNode:
    id: str
    name: str
    description: str
    probability: float
    impact: float
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None


@dataclass
class ScenarioPath:
    nodes: List[str]
    joint_probability: float
    expected_impact: float
    description: str


@dataclass
class ScenarioProbabilityResult:
    scenarios: Dict[str, ScenarioNode]
    paths: List[ScenarioPath]
    total_probability: float
    expected_impact: float
    top_scenario: str
    top_probability: float
    sensitivity: Dict[str, float]
    probability_distribution: Dict[str, float]
    recommendations: List[str]


DEFAULT_SCENARIOS: List[Dict] = [
    {"id": "base_case", "name": "Base Case", "description": "Most likely continuation of current trends",
     "probability": 0.45, "impact": 0.5},
    {"id": "upside", "name": "Upside Scenario", "description": "Faster growth, favorable conditions",
     "probability": 0.20, "impact": 0.8},
    {"id": "downside", "name": "Downside Scenario", "description": "Slower growth, adverse conditions",
     "probability": 0.25, "impact": 0.3},
    {"id": "crisis", "name": "Crisis Scenario", "description": "Major disruption, systemic shock",
     "probability": 0.07, "impact": 0.1},
    {"id": "breakout", "name": "Breakout Scenario", "description": "Unexpected positive transformation",
     "probability": 0.03, "impact": 0.95},
]


class ScenarioProbabilityMapper:
    def __init__(self, n_sensitivity_samples: int = 100):
        self.n_sensitivity_samples = n_sensitivity_samples
        self._rng = np.random.default_rng(42)

    def map_scenarios(self, predictions: Dict[str, float],
                      custom_scenarios: Optional[List[Dict]] = None,
                      relations: Optional[Dict[Tuple[str, str], ScenarioRelation]] = None) -> ScenarioProbabilityResult:
        scenarios = self._build_scenarios(predictions, custom_scenarios)
        if relations is None:
            relations = {}
        paths = self._compute_paths(scenarios, relations)
        sensitivity = self._sensitivity_analysis(predictions, scenarios)
        prob_dist = self._compute_distribution(predictions, scenarios)
        total_prob = sum(s.probability for s in scenarios.values())
        expected_impact = sum(s.probability * s.impact for s in scenarios.values())
        top_id = max(scenarios, key=lambda k: scenarios[k].probability) if scenarios else "base_case"
        top_scenario = scenarios.get(top_id, scenarios.get("base_case", ScenarioNode("base_case", "Base", "", 0.5, 0.5)))
        recs = self._generate_recommendations(scenarios, paths, expected_impact, sensitivity, prob_dist)
        return ScenarioProbabilityResult(
            scenarios=scenarios,
            paths=paths,
            total_probability=round(min(total_prob, 1.5), 4),
            expected_impact=round(expected_impact, 4),
            top_scenario=top_scenario.name,
            top_probability=round(top_scenario.probability, 4),
            sensitivity={k: round(v, 4) for k, v in sensitivity.items()},
            probability_distribution={k: round(v, 4) for k, v in prob_dist.items()},
            recommendations=recs,
        )

    def _build_scenarios(self, predictions: Dict[str, float],
                          custom_scenarios: Optional[List[Dict]]) -> Dict[str, ScenarioNode]:
        source = custom_scenarios if custom_scenarios else DEFAULT_SCENARIOS
        scenarios = {}
        for i, sc in enumerate(source):
            if isinstance(sc, dict):
                prob = sc.get("probability", 0.2)
                impact = sc.get("impact", 0.5)
                if predictions:
                    avg_pred = float(np.mean(list(predictions.values())))
                    prob = round(prob * (0.5 + avg_pred) / 1.0, 4)
                    prob = min(0.95, max(0.01, prob))
                node = ScenarioNode(
                    id=sc.get("id", f"scenario_{i}"),
                    name=sc.get("name", f"Scenario {i}"),
                    description=sc.get("description", ""),
                    probability=prob,
                    impact=impact,
                )
                scenarios[node.id] = node
            elif isinstance(sc, ScenarioNode):
                scenarios[sc.id] = sc
        return scenarios

    def _compute_paths(self, scenarios: Dict[str, ScenarioNode],
                        relations: Dict[Tuple[str, str], ScenarioRelation]) -> List[ScenarioPath]:
        paths = []
        for sid, scenario in scenarios.items():
            joint_prob = scenario.probability
            for (id1, id2), relation in relations.items():
                if id1 == sid:
                    if relation == ScenarioRelation.MUTUALLY_EXCLUSIVE:
                        joint_prob = 0.0
                    elif relation == ScenarioRelation.REINFORCING:
                        other_prob = scenarios[id2].probability if id2 in scenarios else 0
                        joint_prob *= (1 + other_prob * 0.3)
                    elif relation == ScenarioRelation.OPPOSING:
                        other_prob = scenarios[id2].probability if id2 in scenarios else 0
                        joint_prob *= (1 - other_prob * 0.5)
                elif id2 == sid:
                    if relation == ScenarioRelation.MUTUALLY_EXCLUSIVE:
                        joint_prob = 0.0
                    elif relation == ScenarioRelation.REINFORCING:
                        other_prob = scenarios[id1].probability if id1 in scenarios else 0
                        joint_prob *= (1 + other_prob * 0.3)
                    elif relation == ScenarioRelation.OPPOSING:
                        other_prob = scenarios[id1].probability if id1 in scenarios else 0
                        joint_prob *= (1 - other_prob * 0.5)
            joint_prob = min(1.0, max(0.0, joint_prob))
            expected_impact = joint_prob * scenario.impact
            paths.append(ScenarioPath(
                nodes=[sid],
                joint_probability=round(joint_prob, 4),
                expected_impact=round(expected_impact, 4),
                description=scenario.description,
            ))
        for i, sid1 in enumerate(scenarios):
            for sid2 in list(scenarios.keys())[i + 1:]:
                pair_key = (sid1, sid2)
                relation = relations.get(pair_key, ScenarioRelation.INDEPENDENT)
                if relation == ScenarioRelation.CONDITIONAL:
                    cond_prob = scenarios[sid1].probability * 0.5
                    paths.append(ScenarioPath(
                        nodes=[sid1, sid2],
                        joint_probability=round(cond_prob * scenarios[sid2].probability, 4),
                        expected_impact=round(cond_prob * scenarios[sid2].probability * scenarios[sid2].impact, 4),
                        description=f"Conditional: {scenarios[sid1].name} → {scenarios[sid2].name}",
                    ))
        return paths

    def _sensitivity_analysis(self, predictions: Dict[str, float],
                                scenarios: Dict[str, ScenarioNode]) -> Dict[str, float]:
        sensitivity = {}
        for var, val in predictions.items():
            base_expected = sum(s.probability * s.impact for s in scenarios.values())
            perturbed_expected = 0.0
            for _ in range(self.n_sensitivity_samples):
                noise = val * 0.1 * self._rng.normal()
                perturbed_scenarios = {}
                for sid, s in scenarios.items():
                    perturbed_prob = s.probability * (1 + noise * 0.5)
                    perturbed_prob = min(0.99, max(0.01, perturbed_prob))
                    perturbed_scenarios[sid] = ScenarioNode(s.id, s.name, s.description, perturbed_prob, s.impact)
                perturbed_expected += sum(s.probability * s.impact for s in perturbed_scenarios.values())
            perturbed_expected /= self.n_sensitivity_samples
            sensitivity[var] = abs(perturbed_expected - base_expected) / max(abs(base_expected), 1e-10)
        return sensitivity

    def _compute_distribution(self, predictions: Dict[str, float],
                               scenarios: Dict[str, ScenarioNode]) -> Dict[str, float]:
        distribution = {}
        for sid, s in scenarios.items():
            distribution[s.name] = s.probability
        total = sum(distribution.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            distribution = {k: v / total for k, v in distribution.items()}
        return distribution

    def _generate_recommendations(self, scenarios: Dict[str, ScenarioNode],
                                   paths: List[ScenarioPath],
                                   expected_impact: float,
                                   sensitivity: Dict[str, float],
                                   prob_dist: Dict[str, float]) -> List[str]:
        recs = []
        top = max(scenarios.values(), key=lambda s: s.probability) if scenarios else None
        if top:
            recs.append(f"Most likely: {top.name} at {top.probability:.0%} probability (impact={top.impact:.2f})")
        worst = min(scenarios.values(), key=lambda s: s.impact) if scenarios else None
        if worst:
            ev_worst = worst.probability * worst.impact
            recs.append(f"Highest risk: {worst.name} — expected loss={ev_worst:.3f}")
        best = max(scenarios.values(), key=lambda s: s.impact) if scenarios else None
        if best:
            ev_best = best.probability * best.impact
            recs.append(f"Best opportunity: {best.name} — expected gain={ev_best:.3f}")
        if sensitivity:
            top_sens = max(sensitivity, key=sensitivity.get)
            recs.append(f"Key sensitivity: {top_sens} drives the most scenario variation")
        total_prob = sum(s.probability for s in scenarios.values())
        if total_prob > 1.2:
            recs.append("SCENARIO OVERLAP: Total probability > 120% — scenarios may not be mutually exclusive")
        elif total_prob < 0.8:
            recs.append("SCENARIO GAP: Total probability < 80% — consider adding missing scenarios")
        recs.append(f"Expected impact across all scenarios: {expected_impact:.3f}")
        return recs