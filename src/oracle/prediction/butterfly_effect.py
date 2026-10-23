import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from collections import defaultdict, deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

VALID_MECHANISMS = {"contagion", "feedback_loop", "threshold", "amplification", "damping", "cascade"}

DOMAINS = [
    "economic", "geopolitical", "climate", "financial", "technology",
    "social", "energy", "health", "pandemic", "cyber",
    "supply_chain", "political", "demographic", "environment", "trade",
]

DOMAIN_CONNECTIONS = [
    ("economic", "financial", 0.85, "contagion"),
    ("economic", "trade", 0.80, "cascade"),
    ("economic", "political", 0.65, "feedback_loop"),
    ("economic", "social", 0.70, "cascade"),
    ("economic", "energy", 0.60, "threshold"),
    ("economic", "supply_chain", 0.75, "cascade"),
    ("economic", "demographic", 0.40, "amplification"),
    ("financial", "economic", 0.80, "contagion"),
    ("financial", "technology", 0.55, "amplification"),
    ("financial", "political", 0.60, "feedback_loop"),
    ("financial", "cyber", 0.50, "threshold"),
    ("financial", "energy", 0.45, "cascade"),
    ("geopolitical", "political", 0.90, "feedback_loop"),
    ("geopolitical", "trade", 0.85, "cascade"),
    ("geopolitical", "energy", 0.75, "threshold"),
    ("geopolitical", "cyber", 0.60, "amplification"),
    ("geopolitical", "health", 0.40, "cascade"),
    ("geopolitical", "social", 0.55, "amplification"),
    ("geopolitical", "supply_chain", 0.70, "cascade"),
    ("geopolitical", "technology", 0.50, "feedback_loop"),
    ("climate", "environment", 0.95, "amplification"),
    ("climate", "energy", 0.70, "threshold"),
    ("climate", "agriculture", 0.80, "cascade"),
    ("climate", "health", 0.60, "cascade"),
    ("climate", "economic", 0.55, "cascade"),
    ("climate", "social", 0.50, "amplification"),
    ("climate", "supply_chain", 0.65, "cascade"),
    ("climate", "trade", 0.45, "threshold"),
    ("technology", "cyber", 0.80, "amplification"),
    ("technology", "economic", 0.60, "feedback_loop"),
    ("technology", "social", 0.65, "amplification"),
    ("technology", "energy", 0.50, "threshold"),
    ("technology", "health", 0.55, "cascade"),
    ("technology", "financial", 0.55, "amplification"),
    ("technology", "supply_chain", 0.60, "cascade"),
    ("social", "political", 0.70, "feedback_loop"),
    ("social", "health", 0.60, "cascade"),
    ("social", "demographic", 0.75, "amplification"),
    ("social", "economic", 0.55, "feedback_loop"),
    ("social", "pandemic", 0.65, "threshold"),
    ("social", "technology", 0.50, "feedback_loop"),
    ("energy", "economic", 0.70, "threshold"),
    ("energy", "geopolitical", 0.65, "cascade"),
    ("energy", "trade", 0.60, "cascade"),
    ("energy", "health", 0.30, "damping"),
    ("energy", "environment", 0.75, "amplification"),
    ("energy", "supply_chain", 0.55, "cascade"),
    ("health", "social", 0.60, "feedback_loop"),
    ("health", "pandemic", 0.90, "cascade"),
    ("health", "economic", 0.65, "cascade"),
    ("health", "demographic", 0.70, "amplification"),
    ("health", "technology", 0.50, "feedback_loop"),
    ("pandemic", "health", 0.95, "cascade"),
    ("pandemic", "economic", 0.80, "cascade"),
    ("pandemic", "social", 0.75, "contagion"),
    ("pandemic", "supply_chain", 0.70, "cascade"),
    ("pandemic", "trade", 0.60, "cascade"),
    ("pandemic", "political", 0.55, "feedback_loop"),
    ("pandemic", "technology", 0.40, "feedback_loop"),
    ("cyber", "technology", 0.75, "feedback_loop"),
    ("cyber", "financial", 0.70, "cascade"),
    ("cyber", "economic", 0.55, "contagion"),
    ("cyber", "political", 0.50, "amplification"),
    ("cyber", "supply_chain", 0.45, "cascade"),
    ("supply_chain", "economic", 0.70, "cascade"),
    ("supply_chain", "trade", 0.75, "feedback_loop"),
    ("supply_chain", "energy", 0.50, "threshold"),
    ("supply_chain", "technology", 0.55, "amplification"),
    ("supply_chain", "social", 0.40, "cascade"),
    ("political", "economic", 0.60, "feedback_loop"),
    ("political", "geopolitical", 0.85, "feedback_loop"),
    ("political", "social", 0.65, "amplification"),
    ("political", "trade", 0.70, "cascade"),
    ("political", "environment", 0.40, "threshold"),
    ("demographic", "social", 0.70, "amplification"),
    ("demographic", "economic", 0.55, "cascade"),
    ("demographic", "health", 0.60, "cascade"),
    ("demographic", "political", 0.45, "feedback_loop"),
    ("environment", "climate", 0.90, "feedback_loop"),
    ("environment", "health", 0.65, "cascade"),
    ("environment", "energy", 0.55, "threshold"),
    ("environment", "economic", 0.50, "cascade"),
    ("environment", "social", 0.45, "amplification"),
    ("trade", "economic", 0.75, "feedback_loop"),
    ("trade", "geopolitical", 0.60, "cascade"),
    ("trade", "supply_chain", 0.70, "feedback_loop"),
    ("trade", "political", 0.55, "cascade"),
]


@dataclass
class Perturbation:
    variable: str
    magnitude: float
    source_domain: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PropagationStep:
    from_domain: str
    to_domain: str
    amplification: float
    mechanism: str
    cumulative_impact: float


@dataclass
class PropagationTrace:
    perturbation: Perturbation
    steps: List[PropagationStep]
    total_amplification: float
    butterfly_index: float
    reached_domains: List[str]
    max_step_impact: float


@dataclass
class CriticalNode:
    domain: str
    amplification_score: float
    out_weight: float
    in_weight: float
    betweenness: float
    sensitivity: float


@dataclass
class AmplificationPath:
    path: List[str]
    cumulative_amplification: float
    mechanisms: List[str]
    length: int


@dataclass
class EarlyWarning:
    variable: str
    domain: str
    current_impact: float
    amplification_potential: float
    threshold_proximity: float
    severity: str
    connected_domains: List[str]


MECHANISM_AMPLIFICATION = {
    "contagion": 1.15,
    "feedback_loop": 1.35,
    "threshold": 1.50,
    "amplification": 1.25,
    "damping": 0.75,
    "cascade": 1.20,
}


class DomainNetwork:
    def __init__(self):
        self._adjacency: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)
        self._domains: Set[str] = set()
        self._node_in_weights: Dict[str, float] = defaultdict(float)
        self._node_out_weights: Dict[str, float] = defaultdict(float)

    def add_connection(self, from_domain: str, to_domain: str, weight: float, mechanism: str) -> None:
        if mechanism not in VALID_MECHANISMS:
            raise ValueError(f"Invalid mechanism: {mechanism}. Must be one of {VALID_MECHANISMS}")
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be in [0, 1], got {weight}")
        self._adjacency[from_domain].append((to_domain, weight, mechanism))
        self._domains.add(from_domain)
        self._domains.add(to_domain)
        self._node_out_weights[from_domain] += weight
        self._node_in_weights[to_domain] += weight

    @property
    def domains(self) -> Set[str]:
        return self._domains.copy()

    @property
    def n_connections(self) -> int:
        return sum(len(v) for v in self._adjacency.values())

    def get_neighbors(self, domain: str) -> List[Tuple[str, float, str]]:
        return self._adjacency.get(domain, [])

    def propagate(self, perturbation: Perturbation, max_steps: int = 10) -> List[PropagationStep]:
        steps: List[PropagationStep] = []
        visited: Set[str] = {perturbation.source_domain}
        queue: deque = deque()
        neighbors = self.get_neighbors(perturbation.source_domain)
        for to_domain, weight, mechanism in neighbors:
            mech_amp = MECHANISM_AMPLIFICATION.get(mechanism, 1.0)
            effective_amp = weight * mech_amp
            cumulative = perturbation.magnitude * effective_amp
            step = PropagationStep(
                from_domain=perturbation.source_domain,
                to_domain=to_domain,
                amplification=effective_amp,
                mechanism=mechanism,
                cumulative_impact=cumulative,
            )
            steps.append(step)
            if to_domain not in visited:
                visited.add(to_domain)
                queue.append((to_domain, cumulative, 1))

        while queue and len(steps) < max_steps * 5:
            current_domain, incoming_impact, depth = queue.popleft()
            if depth >= max_steps:
                continue
            neighbors = self.get_neighbors(current_domain)
            for to_domain, weight, mechanism in neighbors:
                if to_domain in visited and depth > 2:
                    continue
                mech_amp = MECHANISM_AMPLIFICATION.get(mechanism, 1.0)
                effective_amp = weight * mech_amp
                decay = 0.85 ** depth
                cumulative = incoming_impact * effective_amp * decay
                step = PropagationStep(
                    from_domain=current_domain,
                    to_domain=to_domain,
                    amplification=effective_amp * decay,
                    mechanism=mechanism,
                    cumulative_impact=cumulative,
                )
                steps.append(step)
                if to_domain not in visited:
                    visited.add(to_domain)
                    queue.append((to_domain, cumulative, depth + 1))

        return steps

    def _compute_betweenness(self) -> Dict[str, float]:
        betweenness: Dict[str, float] = defaultdict(float)
        domain_list = list(self._domains)
        for source in domain_list:
            for target in domain_list:
                if source == target:
                    continue
                paths = self._find_shortest_paths(source, target)
                for path in paths:
                    for node in path[1:-1]:
                        betweenness[node] += 1.0 / len(paths)
        n = len(domain_list)
        if n > 2:
            for k in betweenness:
                betweenness[k] /= ((n - 1) * (n - 2))
        return dict(betweenness)

    def _find_shortest_paths(self, source: str, target: str) -> List[List[str]]:
        if source not in self._domains or target not in self._domains:
            return []
        queue = deque([(source, [source])])
        visited = {source: 0}
        paths = []
        min_len = float('inf')
        while queue:
            node, path = queue.popleft()
            if len(path) > min_len:
                continue
            if node == target:
                if len(path) < min_len:
                    min_len = len(path)
                    paths = [path]
                elif len(path) == min_len:
                    paths.append(path)
                continue
            for neighbor, _, _ in self._adjacency.get(node, []):
                if len(path) < min_len and (neighbor not in visited or visited[neighbor] >= len(path)):
                    visited[neighbor] = len(path)
                    queue.append((neighbor, path + [neighbor]))
        return paths if paths else [[source]]

    def compute_butterfly_index(self, perturbation: Perturbation) -> float:
        steps = self.propagate(perturbation, max_steps=10)
        if not steps:
            return 1.0
        total_cumulative = sum(abs(s.cumulative_impact) for s in steps)
        initial = abs(perturbation.magnitude)
        if initial < 1e-12:
            return 1.0
        return total_cumulative / initial

    def find_critical_nodes(self) -> List[CriticalNode]:
        betweenness = self._compute_betweenness()
        nodes: List[CriticalNode] = []
        for domain in self._domains:
            out_w = self._node_out_weights.get(domain, 0.0)
            in_w = self._node_in_weights.get(domain, 0.0)
            bw = betweenness.get(domain, 0.0)
            sensitivity = self.sensitivity_analysis(domain)
            amp_score = (out_w * 0.4) + (bw * 0.3) + (sensitivity * 0.3)
            nodes.append(CriticalNode(
                domain=domain,
                amplification_score=amp_score,
                out_weight=out_w,
                in_weight=in_w,
                betweenness=bw,
                sensitivity=sensitivity,
            ))
        nodes.sort(key=lambda n: n.amplification_score, reverse=True)
        return nodes

    def sensitivity_analysis(self, variable: str) -> float:
        if variable not in self._domains:
            return 0.0
        test_perturbation = Perturbation(
            variable=f"test_{variable}",
            magnitude=1.0,
            source_domain=variable,
        )
        butterfly_idx = self.compute_butterfly_index(test_perturbation)
        out_w = self._node_out_weights.get(variable, 0.0)
        n_incoming = len([1 for src_edges in self._adjacency.values()
                          for dst, _, _ in src_edges if dst == variable])
        inbound_sensitivity = n_incoming * 0.05
        return min(1.0, butterfly_idx * 0.3 + out_w * 0.4 + inbound_sensitivity * 0.3)


class ButterflyTracker:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.network = DomainNetwork()
        self._perturbation_history: List[Perturbation] = []
        self._trace_history: List[PropagationTrace] = []
        for from_d, to_d, weight, mechanism in DOMAIN_CONNECTIONS:
            self.network.add_connection(from_d, to_d, weight, mechanism)

    def trace_perturbation(self, variable: str, magnitude: float,
                           source_domain: Optional[str] = None) -> PropagationTrace:
        if source_domain is None:
            source_domain = variable.split("_")[0] if "_" in variable else variable
        if source_domain not in self.network.domains:
            best = min(self.network.domains,
                       key=lambda d: self._domain_distance(d, source_domain))
            source_domain = best
        perturbation = Perturbation(
            variable=variable,
            magnitude=magnitude,
            source_domain=source_domain,
        )
        steps = self.network.propagate(perturbation, max_steps=10)
        total_amp = abs(magnitude)
        for s in steps:
            total_amp += abs(s.cumulative_impact)
        initial = abs(magnitude)
        butterfly_idx = total_amp / initial if initial > 1e-12 else 1.0
        reached = list({s.to_domain for s in steps} | {source_domain})
        max_impact = max((abs(s.cumulative_impact) for s in steps), default=0.0)
        trace = PropagationTrace(
            perturbation=perturbation,
            steps=steps,
            total_amplification=total_amp,
            butterfly_index=butterfly_idx,
            reached_domains=reached,
            max_step_impact=max_impact,
        )
        self._perturbation_history.append(perturbation)
        self._trace_history.append(trace)
        return trace

    def butterfly_score(self, variable: str) -> float:
        test = self.trace_perturbation(variable, magnitude=1.0)
        raw = test.butterfly_index
        score = 1.0 - 1.0 / (1.0 + raw / 5.0)
        return min(1.0, max(0.0, score))

    def find_amplification_paths(self, threshold: float = 1.5) -> List[AmplificationPath]:
        paths: List[AmplificationPath] = []
        for domain in self.network.domains:
            self._dfs_paths(domain, [], 1.0, [], threshold, paths, max_depth=6, visited=set())
        return paths

    def _dfs_paths(self, current: str, path: List[str], cum_amp: float,
                   mechanisms: List[str], threshold: float,
                   results: List[AmplificationPath], max_depth: int,
                   visited: Set[str]) -> None:
        path = path + [current]
        if len(path) > 1 and cum_amp >= threshold:
            results.append(AmplificationPath(
                path=list(path),
                cumulative_amplification=cum_amp,
                mechanisms=list(mechanisms),
                length=len(path) - 1,
            ))
        if len(path) > max_depth:
            return
        visited_copy = visited | {current}
        for to_domain, weight, mechanism in self.network.get_neighbors(current):
            if to_domain in visited_copy:
                continue
            mech_amp = MECHANISM_AMPLIFICATION.get(mechanism, 1.0)
            new_amp = cum_amp * weight * mech_amp
            new_mechs = mechanisms + [mechanism]
            self._dfs_paths(to_domain, path, new_amp, new_mechs, threshold,
                            results, max_depth, visited_copy)

    def cascade_potential(self) -> float:
        critical_nodes = self.network.find_critical_nodes()
        if not critical_nodes:
            return 0.0
        top_5 = critical_nodes[:5]
        avg_amp = np.mean([n.amplification_score for n in top_5])
        avg_out = np.mean([n.out_weight for n in top_5])
        avg_bw = np.mean([n.betweenness for n in top_5])
        n_domain_connections = self.network.n_connections
        n_domains = len(self.network.domains)
        connectivity = n_domain_connections / (n_domains * (n_domains - 1)) if n_domains > 1 else 0
        raw = avg_amp * 0.4 + min(1.0, avg_out / 5.0) * 0.3 + min(1.0, avg_bw * 10) * 0.15 + min(1.0, connectivity * 2) * 0.15
        return min(1.0, max(0.0, raw))

    def get_early_warnings(self, threshold_proximity: float = 0.7) -> List[EarlyWarning]:
        warnings: List[EarlyWarning] = []
        critical_nodes = self.network.find_critical_nodes()
        for node in critical_nodes:
            if node.sensitivity < 0.3:
                continue
            sensitivity = node.sensitivity
            amplification_potential = node.out_weight * sensitivity
            proximity = min(1.0, amplification_potential / 3.0)
            if proximity >= threshold_proximity * 0.5:
                neighbor_domains = [d for d, _, _ in self.network.get_neighbors(node.domain)]
                if sensitivity > 0.8:
                    severity = "critical"
                elif sensitivity > 0.6:
                    severity = "high"
                elif sensitivity > 0.4:
                    severity = "medium"
                else:
                    severity = "low"
                warnings.append(EarlyWarning(
                    variable=f"{node.domain}_perturbation",
                    domain=node.domain,
                    current_impact=node.in_weight,
                    amplification_potential=amplification_potential,
                    threshold_proximity=proximity,
                    severity=severity,
                    connected_domains=neighbor_domains,
                ))
        warnings.sort(key=lambda w: w.threshold_proximity, reverse=True)
        return warnings

    @staticmethod
    def _domain_distance(a: str, b: str) -> float:
        if a == b:
            return 0.0
        if a[:4] == b[:4]:
            return 0.3
        common = len(set(a) & set(b))
        total = len(set(a) | set(b))
        return 1.0 - (common / max(total, 1))