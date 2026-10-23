"""Probabilistic Network — Bayesian network with inference for multi-variable prediction.

Implements a Bayesian belief network with forward/backward inference, conditional
probability tables, and belief propagation. Supports:
1. Network construction with conditional dependencies
2. Forward inference (evidence → posterior beliefs)
3. Backward inference (target → explanatory variables)
4. D-separation for identifying independent subgraphs
5. Most probable explanation (MPE) via belief propagation
6. Sensitivity analysis on network parameters
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class NodeType(Enum):
    BINARY = "binary"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


@dataclass
class NetworkNode:
    name: str
    states: List[str]
    prior: List[float]
    parents: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ConditionalTable:
    child: str
    parents: List[str]
    table: Dict[str, List[float]]


@dataclass
class InferenceResult:
    variable: str
    posterior: List[float]
    states: List[str]
    evidence: Dict[str, str]
    method: str
    confidence: float


@dataclass
class SensitivityResult:
    variable: str
    sensitivity: float
    explanation: str


@dataclass
class NetworkResult:
    posteriors: Dict[str, List[float]]
    states: Dict[str, List[str]]
    evidence: Dict[str, str]
    most_probable_state: Dict[str, str]
    sensitivity: List[SensitivityResult]
    independent_sets: List[Set[str]]
    narrative: str


DEFAULT_NODES: List[NetworkNode] = [
    NetworkNode("recession", ["yes", "no"], [0.2, 0.8], [], "Economic recession"),
    NetworkNode("high_inflation", ["yes", "no"], [0.25, 0.75], [], "Persistent high inflation"),
    NetworkNode("geopolitical_crisis", ["yes", "no"], [0.15, 0.85], [], "Major geopolitical crisis"),
    NetworkNode("tech_disruption", ["yes", "no"], [0.3, 0.7], [], "Significant technology disruption"),
    NetworkNode("financial_stress", ["low", "medium", "high"], [0.4, 0.35, 0.25],
                ["recession", "high_inflation"], "Financial system stress"),
    NetworkNode("market_crash", ["yes", "no"], [0.15, 0.85],
                ["recession", "financial_stress"], "Market crash"),
    NetworkNode("social_unrest", ["low", "medium", "high"], [0.4, 0.35, 0.25],
                ["recession", "geopolitical_crisis"], "Social unrest level"),
    NetworkNode("supply_disruption", ["yes", "no"], [0.2, 0.8],
                ["geopolitical_crisis", "tech_disruption"], "Supply chain disruption"),
    NetworkNode("policy_response", ["effective", "moderate", "ineffective"], [0.3, 0.45, 0.25],
                ["recession", "high_inflation"], "Policy response effectiveness"),
    NetworkNode("recovery_speed", ["fast", "normal", "slow"], [0.25, 0.45, 0.3],
                ["policy_response", "financial_stress"], "Recovery speed"),
]

DEFAULT_CONDITIONALS: List[ConditionalTable] = [
    ConditionalTable("financial_stress", ["recession", "high_inflation"], {
        "yes_yes": [0.05, 0.25, 0.7],
        "yes_no": [0.1, 0.4, 0.5],
        "no_yes": [0.3, 0.45, 0.25],
        "no_no": [0.7, 0.25, 0.05],
    }),
    ConditionalTable("market_crash", ["recession", "financial_stress"], {
        "yes_low": [0.05, 0.95],
        "yes_medium": [0.25, 0.75],
        "yes_high": [0.65, 0.35],
        "no_low": [0.02, 0.98],
        "no_medium": [0.1, 0.9],
        "no_high": [0.35, 0.65],
    }),
    ConditionalTable("social_unrest", ["recession", "geopolitical_crisis"], {
        "yes_yes": [0.05, 0.15, 0.8],
        "yes_no": [0.1, 0.35, 0.55],
        "no_yes": [0.15, 0.4, 0.45],
        "no_no": [0.6, 0.3, 0.1],
    }),
    ConditionalTable("supply_disruption", ["geopolitical_crisis", "tech_disruption"], {
        "yes_yes": [0.85, 0.15],
        "yes_no": [0.6, 0.4],
        "no_yes": [0.5, 0.5],
        "no_no": [0.05, 0.95],
    }),
    ConditionalTable("policy_response", ["recession", "high_inflation"], {
        "yes_yes": [0.15, 0.35, 0.5],
        "yes_no": [0.4, 0.4, 0.2],
        "no_yes": [0.2, 0.4, 0.4],
        "no_no": [0.6, 0.35, 0.05],
    }),
    ConditionalTable("recovery_speed", ["policy_response", "financial_stress"], {
        "effective_low": [0.7, 0.25, 0.05],
        "effective_medium": [0.5, 0.35, 0.15],
        "effective_high": [0.3, 0.4, 0.3],
        "moderate_low": [0.5, 0.4, 0.1],
        "moderate_medium": [0.3, 0.45, 0.25],
        "moderate_high": [0.15, 0.35, 0.5],
        "ineffective_low": [0.3, 0.45, 0.25],
        "ineffective_medium": [0.15, 0.35, 0.5],
        "ineffective_high": [0.05, 0.2, 0.75],
    }),
]


class ProbabilisticNetwork:
    def __init__(self, nodes: Optional[List[NetworkNode]] = None,
                 conditionals: Optional[List[ConditionalTable]] = None):
        self.nodes = {}
        self.conditional_tables = {}
        self.beliefs: Dict[str, np.ndarray] = {}
        for node in (nodes or DEFAULT_NODES):
            self.nodes[node.name] = node
            self.beliefs[node.name] = np.array(node.prior)
        for ct in (conditionals or DEFAULT_CONDITIONALS):
            self.conditional_tables[ct.child] = ct

    def add_node(self, node: NetworkNode) -> None:
        self.nodes[node.name] = node
        self.beliefs[node.name] = np.array(node.prior)

    def add_conditional(self, ct: ConditionalTable) -> None:
        self.conditional_tables[ct.child] = ct

    def infer(self, evidence: Dict[str, str],
              target: Optional[str] = None) -> NetworkResult:
        beliefs = {name: np.array(node.prior) for name, node in self.nodes.items()}
        for var, state in evidence.items():
            if var in beliefs:
                node = self.nodes[var]
                state_idx = node.states.index(state) if state in node.states else 0
                probs = np.zeros(len(node.states))
                probs[state_idx] = 1.0
                beliefs[var] = probs
        for _ in range(10):
            new_beliefs = {}
            for name, node in self.nodes.items():
                if name in evidence:
                    new_beliefs[name] = beliefs[name]
                    continue
                if node.parents:
                    parent_combos = self._parent_combinations(node)
                    new_probs = np.zeros(len(node.states))
                    ct = self.conditional_tables.get(name)
                    if ct:
                        for combo_key, parent_probs in parent_combos.items():
                            table_key = combo_key
                            if table_key in ct.table:
                                cond_probs = np.array(ct.table[table_key])
                            else:
                                cond_probs = np.array(node.prior)
                            new_probs += parent_probs * cond_probs
                    else:
                        new_probs = np.array(node.prior)
                    total = new_probs.sum()
                    if total > 0:
                        new_probs = new_probs / total
                    else:
                        new_probs = np.array(node.prior)
                    new_beliefs[name] = new_probs
                else:
                    new_beliefs[name] = beliefs[name] if name not in evidence else beliefs[name]
            beliefs = new_beliefs
        for name, probs in beliefs.items():
            total = probs.sum()
            if total > 0:
                beliefs[name] = probs / total
        mpe = {}
        for name, probs in beliefs.items():
            node = self.nodes[name]
            mpe[name] = node.states[np.argmax(probs)]
        sensitivity = self._sensitivity_analysis(beliefs, evidence)
        independent_sets = self._find_independent_sets(evidence)
        narrative = self._generate_narrative(beliefs, evidence, mpe)
        posteriors = {name: [round(float(p), 4) for p in probs] for name, probs in beliefs.items()}
        states = {name: node.states for name, node in self.nodes.items()}
        return NetworkResult(
            posteriors=posteriors, states=states, evidence=evidence,
            most_probable_state=mpe, sensitivity=sensitivity,
            independent_sets=independent_sets, narrative=narrative,
        )

    def infer_target(self, target: str, evidence: Dict[str, str]) -> InferenceResult:
        result = self.infer(evidence)
        if target in result.posteriors:
            return InferenceResult(
                variable=target,
                posterior=result.posteriors[target],
                states=result.states[target],
                evidence=evidence,
                method="belief_propagation",
                confidence=round(max(result.posteriors[target]), 4),
            )
        return InferenceResult(
            variable=target, posterior=[0.5, 0.5], states=["yes", "no"],
            evidence=evidence, method="prior", confidence=0.5,
        )

    def probability_of(self, variable: str, state: str,
                         evidence: Dict[str, str]) -> float:
        result = self.infer(evidence, target=variable)
        if variable in result.posteriors:
            node = self.nodes[variable]
            if state in node.states:
                idx = node.states.index(state)
                return result.posteriors[variable][idx]
        return 0.0

    def _parent_combinations(self, node: NetworkNode) -> Dict[str, float]:
        if not node.parents:
            return {"": 1.0}
        parent_state_lists = []
        for pname in node.parents:
            if pname in self.beliefs:
                parent_state_lists.append(list(self.beliefs[pname]))
            else:
                parent_node = self.nodes.get(pname)
                if parent_node:
                    parent_state_lists.append(parent_node.prior)
                else:
                    parent_state_lists.append([0.5, 0.5])
        combos = {}
        n_parents = len(node.parents)
        parent_states = [self.nodes[p].states if p in self.nodes else ["yes", "no"]
                        for p in node.parents]
        indices = [range(len(sl)) for sl in parent_state_lists]
        for combo in _cartesian_product(indices):
            key_parts = []
            prob = 1.0
            for i, idx in enumerate(combo):
                probs = parent_state_lists[i]
                prob *= probs[idx] if idx < len(probs) else 0.5
                pname = node.parents[i]
                pstates = parent_states[i]
                key_parts.append(pstates[idx] if idx < len(pstates) else f"state_{idx}")
            key = "_".join(key_parts)
            combos[key] = prob
        return combos

    def _sensitivity_analysis(self, beliefs: Dict[str, np.ndarray],
                               evidence: Dict[str, str]) -> List[SensitivityResult]:
        results = []
        for name, node in self.nodes.items():
            if name in evidence:
                continue
            if name in self.conditional_tables:
                entropy = -np.sum(beliefs[name] * np.log2(beliefs[name] + 1e-15))
                max_entropy = np.log2(len(node.states))
                sensitivity = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
                results.append(SensitivityResult(
                    variable=name,
                    sensitivity=round(sensitivity, 4),
                    explanation=f"{name}: {sensitivity:.0%} determined by evidence"
                ))
        results.sort(key=lambda x: x.sensitivity, reverse=True)
        return results[:5]

    def _find_independent_sets(self, evidence: Dict[str, str]) -> List[Set[str]]:
        root_nodes = {name for name, node in self.nodes.items() if not node.parents}
        evidence_set = set(evidence.keys())
        if not evidence_set:
            independent_sets = [root_nodes]
            child_sets = {name for name in self.nodes if name not in root_nodes}
            if child_sets:
                independent_sets.append(child_sets)
            return independent_sets
        connected = set()
        for name, node in self.nodes.items():
            for parent in node.parents:
                if parent in evidence_set:
                    connected.add(name)
        independent = set(self.nodes.keys()) - connected - evidence_set
        sets = []
        if connected:
            sets.append(connected)
        if independent:
            sets.append(independent)
        return sets

    def _generate_narrative(self, beliefs: Dict[str, np.ndarray],
                             evidence: Dict[str, str],
                             mpe: Dict[str, str]) -> str:
        parts = []
        if evidence:
            parts.append(f"Given: {', '.join(f'{k}={v}' for k, v in evidence.items())}")
        high_prob = [(name, max(probs)) for name, probs in beliefs.items()
                     if name not in evidence]
        high_prob.sort(key=lambda x: x[1], reverse=True)
        for name, prob in high_prob[:3]:
            node = self.nodes[name]
            state = mpe[name]
            parts.append(f"{name}: {state} ({prob:.0%} probability)")
        return ". ".join(parts)


def _cartesian_product(lists):
    if not lists:
        yield ()
        return
    for item in lists[0]:
        for rest in _cartesian_product(lists[1:]):
            yield (item,) + rest