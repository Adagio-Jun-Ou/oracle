import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChainNode:
    variable: str
    base_probability: float
    value_range: Tuple[float, float] = (0.0, 1.0)

    def __hash__(self):
        return hash(self.variable)

    def __eq__(self, other):
        if isinstance(other, ChainNode):
            return self.variable == other.variable
        return NotImplemented


@dataclass
class ChainEdge:
    source: str
    target: str
    conditional_prob: float
    condition_range: Tuple[float, float] = (0.0, 1.0)

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "conditional_prob": round(self.conditional_prob, 4),
            "condition_range": list(self.condition_range),
        }


@dataclass
class PathStep:
    variable: str
    probability: float
    cumulative_probability: float
    value_range: Tuple[float, float]

    def to_dict(self) -> Dict:
        return {
            "variable": self.variable,
            "probability": round(self.probability, 4),
            "cumulative_probability": round(self.cumulative_probability, 4),
            "value_range": list(self.value_range),
        }


@dataclass
class ChainResult:
    chain_nodes: List[str]
    chain_probabilities: List[float]
    joint_probability: float
    most_probable_path: List[PathStep]
    critical_links: List[str]

    def to_dict(self) -> Dict:
        return {
            "chain_nodes": self.chain_nodes,
            "chain_probabilities": [round(p, 4) for p in self.chain_probabilities],
            "joint_probability": round(self.joint_probability, 6),
            "most_probable_path": [p.to_dict() for p in self.most_probable_path],
            "critical_links": self.critical_links,
        }


DEFAULT_NODES = [
    ChainNode("us_gdp_growth", 0.70, (0.0, 1.0)),
    ChainNode("fed_interest_rate", 0.55, (0.0, 1.0)),
    ChainNode("us_inflation", 0.50, (0.0, 1.0)),
    ChainNode("unemployment_rate", 0.45, (0.0, 1.0)),
    ChainNode("sp500_returns", 0.50, (0.0, 1.0)),
    ChainNode("vix_volatility", 0.40, (0.0, 1.0)),
    ChainNode("oil_price", 0.45, (0.0, 1.0)),
    ChainNode("usd_strength", 0.50, (0.0, 1.0)),
    ChainNode("china_gdp_growth", 0.60, (0.0, 1.0)),
    ChainNode("geopolitical_tension", 0.35, (0.0, 1.0)),
    ChainNode("global_trade_volume", 0.55, (0.0, 1.0)),
    ChainNode("eurozone_stability", 0.50, (0.0, 1.0)),
    ChainNode("gold_price", 0.45, (0.0, 1.0)),
    ChainNode("tech_sector_growth", 0.60, (0.0, 1.0)),
    ChainNode("emerging_market_stress", 0.40, (0.0, 1.0)),
]


DEFAULT_EDGES = [
    ChainEdge("us_gdp_growth", "fed_interest_rate", 0.65, (0.6, 1.0)),
    ChainEdge("us_gdp_growth", "sp500_returns", 0.70, (0.5, 1.0)),
    ChainEdge("us_gdp_growth", "unemployment_rate", -0.60, (0.5, 1.0)),
    ChainEdge("fed_interest_rate", "us_inflation", -0.55, (0.5, 1.0)),
    ChainEdge("fed_interest_rate", "usd_strength", 0.50, (0.4, 1.0)),
    ChainEdge("fed_interest_rate", "emerging_market_stress", 0.45, (0.4, 1.0)),
    ChainEdge("us_inflation", "fed_interest_rate", 0.60, (0.4, 1.0)),
    ChainEdge("us_inflation", "gold_price", 0.50, (0.4, 1.0)),
    ChainEdge("unemployment_rate", "us_gdp_growth", -0.40, (0.4, 1.0)),
    ChainEdge("unemployment_rate", "vix_volatility", 0.50, (0.4, 1.0)),
    ChainEdge("sp500_returns", "vix_volatility", -0.65, (0.3, 0.7)),
    ChainEdge("vix_volatility", "gold_price", 0.55, (0.5, 1.0)),
    ChainEdge("oil_price", "us_inflation", 0.45, (0.4, 1.0)),
    ChainEdge("oil_price", "geopolitical_tension", 0.40, (0.5, 1.0)),
    ChainEdge("china_gdp_growth", "global_trade_volume", 0.60, (0.4, 1.0)),
    ChainEdge("china_gdp_growth", "emerging_market_stress", -0.45, (0.4, 1.0)),
    ChainEdge("geopolitical_tension", "oil_price", 0.55, (0.4, 1.0)),
    ChainEdge("geopolitical_tension", "vix_volatility", 0.45, (0.3, 1.0)),
    ChainEdge("global_trade_volume", "us_gdp_growth", 0.50, (0.4, 1.0)),
    ChainEdge("usd_strength", "emerging_market_stress", 0.50, (0.4, 1.0)),
    ChainEdge("eurozone_stability", "usd_strength", -0.40, (0.3, 0.7)),
    ChainEdge("tech_sector_growth", "sp500_returns", 0.55, (0.4, 1.0)),
    ChainEdge("tech_sector_growth", "us_gdp_growth", 0.40, (0.4, 1.0)),
    ChainEdge("emerging_market_stress", "vix_volatility", 0.40, (0.4, 1.0)),
    ChainEdge("gold_price", "usd_strength", -0.50, (0.4, 1.0)),
]


class ForesightChainBuilder:
    def __init__(self):
        self.nodes: Dict[str, ChainNode] = {}
        self.edges: List[ChainEdge] = []
        self.outgoing: Dict[str, List[ChainEdge]] = defaultdict(list)
        self.incoming: Dict[str, List[ChainEdge]] = defaultdict(list)

        for node in DEFAULT_NODES:
            self.nodes[node.variable] = node

        for edge in DEFAULT_EDGES:
            self.edges.append(edge)
            self.outgoing[edge.source].append(edge)
            self.incoming[edge.target].append(edge)

    def add_node(self, variable: str, base_probability: float,
                 value_range: Tuple[float, float] = (0.0, 1.0)) -> ChainNode:
        node = ChainNode(
            variable=variable,
            base_probability=float(np.clip(base_probability, 0.0, 1.0)),
            value_range=value_range,
        )
        self.nodes[variable] = node
        return node

    def add_conditional(self, source: str, target: str,
                        conditional_prob: float,
                        condition_range: Tuple[float, float] = (0.0, 1.0)) -> ChainEdge:
        if source not in self.nodes:
            self.add_node(source, 0.5)
        if target not in self.nodes:
            self.add_node(target, 0.5)

        edge = ChainEdge(
            source=source,
            target=target,
            conditional_prob=float(np.clip(conditional_prob, -1.0, 1.0)),
            condition_range=condition_range,
        )
        self.edges.append(edge)
        self.outgoing[source].append(edge)
        self.incoming[target].append(edge)
        return edge

    def _compute_conditional_probability(self, target: str,
                                          parent_values: Dict[str, float]) -> float:
        if target not in self.nodes:
            return 0.5

        base = self.nodes[target].base_probability

        if target not in self.incoming or not self.incoming[target]:
            return base

        logit_base = np.log(base / max(1.0 - base, 1e-10) + 1e-10)

        total_shift = 0.0
        for edge in self.incoming[target]:
            if edge.source not in parent_values:
                continue

            val = parent_values[edge.source]
            low, high = edge.condition_range
            if not (low <= val <= high):
                continue

            weight = edge.conditional_prob
            total_shift += abs(weight) * 0.3

            if weight > 0:
                total_shift += weight * 0.1 * val
            else:
                total_shift -= abs(weight) * 0.1 * val

        adjusted_logit = logit_base + total_shift
        prob = 1.0 / (1.0 + np.exp(-adjusted_logit))

        return float(np.clip(prob, 0.01, 0.99))

    def compute_chain(self, start_variable: str, start_value: float,
                       max_depth: int = 5) -> ChainResult:
        if start_variable not in self.nodes:
            logger.warning("Start variable %s not found in chain", start_variable)
            return ChainResult(
                chain_nodes=[], chain_probabilities=[],
                joint_probability=0.0, most_probable_path=[], critical_links=[],
            )

        visited: Set[str] = set()
        chain_nodes: List[str] = []
        chain_probs: List[float] = []
        path_steps: List[PathStep] = []
        parent_values: Dict[str, float] = {}

        queue = deque()
        queue.append((start_variable, start_value, 0))
        visited.add(start_variable)

        cumulative = 1.0

        while queue:
            var, val, depth = queue.popleft()
            if depth > max_depth:
                continue

            node = self.nodes[var]
            cond_prob = self._compute_conditional_probability(var, parent_values)

            if depth == 0:
                cond_prob = float(np.clip(val, 0.01, 0.99))
                parent_values[var] = val
            else:
                parent_values[var] = cond_prob

            cumulative *= cond_prob
            chain_nodes.append(var)
            chain_probs.append(cond_prob)
            path_steps.append(PathStep(
                variable=var,
                probability=round(cond_prob, 4),
                cumulative_probability=round(cumulative, 6),
                value_range=node.value_range,
            ))

            if var in self.outgoing:
                next_edges = sorted(
                    [e for e in self.outgoing[var] if e.target not in visited],
                    key=lambda e: -abs(e.conditional_prob),
                )
                for edge in next_edges[:3]:
                    if edge.target not in visited and edge.target in self.nodes:
                        visited.add(edge.target)
                        queue.append((edge.target, 0.0, depth + 1))

        critical = self.critical_chain_links()

        logger.info(
            "Forward chain from %s: %d nodes, joint_prob=%.6f",
            start_variable, len(chain_nodes), cumulative,
        )

        return ChainResult(
            chain_nodes=chain_nodes,
            chain_probabilities=chain_probs,
            joint_probability=round(cumulative, 6),
            most_probable_path=path_steps,
            critical_links=critical,
        )

    def compute_backward_chain(self, target_variable: str,
                                target_value: float,
                                max_depth: int = 5) -> ChainResult:
        if target_variable not in self.nodes:
            logger.warning("Target variable %s not found in chain", target_variable)
            return ChainResult(
                chain_nodes=[], chain_probabilities=[],
                joint_probability=0.0, most_probable_path=[], critical_links=[],
            )

        visited: Set[str] = set()
        chain_nodes: List[str] = []
        chain_probs: List[float] = []
        path_steps: List[PathStep] = []
        parent_values: Dict[str, float] = {}

        queue = deque()
        queue.append((target_variable, target_value, 0))
        visited.add(target_variable)

        cumulative = 1.0

        while queue:
            var, val, depth = queue.popleft()
            if depth > max_depth:
                continue

            node = self.nodes[var]

            if depth == 0:
                cond_prob = float(np.clip(val, 0.01, 0.99))
                parent_values[var] = val
            else:
                incoming = self.incoming.get(var, [])
                if incoming:
                    cond_prob = self._compute_conditional_probability(var, parent_values)
                else:
                    cond_prob = node.base_probability
                parent_values[var] = cond_prob

            cumulative *= cond_prob
            chain_nodes.append(var)
            chain_probs.append(cond_prob)
            path_steps.append(PathStep(
                variable=var,
                probability=round(cond_prob, 4),
                cumulative_probability=round(cumulative, 6),
                value_range=node.value_range,
            ))

            if var in self.incoming:
                prev_edges = sorted(
                    [e for e in self.incoming[var] if e.source not in visited],
                    key=lambda e: -abs(e.conditional_prob),
                )
                for edge in prev_edges[:3]:
                    if edge.source not in visited and edge.source in self.nodes:
                        visited.add(edge.source)
                        queue.append((edge.source, 0.0, depth + 1))

        critical = self.critical_chain_links()

        logger.info(
            "Backward chain to %s: %d nodes, joint_prob=%.6f",
            target_variable, len(chain_nodes), cumulative,
        )

        return ChainResult(
            chain_nodes=chain_nodes,
            chain_probabilities=chain_probs,
            joint_probability=round(cumulative, 6),
            most_probable_path=path_steps,
            critical_links=critical,
        )

    def critical_chain_links(self, threshold: float = 0.4) -> List[str]:
        link_impacts: Dict[str, float] = {}

        for edge in self.edges:
            impact = abs(edge.conditional_prob)
            src_out = len(self.outgoing.get(edge.source, []))
            tgt_in = len(self.incoming.get(edge.target, []))
            centrality = 1.0 + 0.1 * src_out + 0.1 * tgt_in
            link_impacts[f"{edge.source}->{edge.target}"] = impact * centrality

        sorted_links = sorted(link_impacts.items(), key=lambda x: -x[1])
        result = [link for link, impact in sorted_links if impact >= threshold]

        if not result and sorted_links:
            result = [sorted_links[0][0]]

        return result

    def all_nodes(self) -> List[ChainNode]:
        return list(self.nodes.values())

    def all_edges(self) -> List[ChainEdge]:
        return list(self.edges)

    def Monte_Carlo_chain(self, start_variable: str, start_value: float,
                            n_simulations: int = 1000,
                            max_depth: int = 5) -> Dict[str, float]:
        variable_probs: Dict[str, float] = defaultdict(float)
        variable_counts: Dict[str, int] = defaultdict(int)

        for _ in range(n_simulations):
            visited: Set[str] = set()
            queue = deque()
            queue.append((start_variable, start_value, 0))
            visited.add(start_variable)
            parent_vals: Dict[str, float] = {}
            active_prob = 1.0

            while queue:
                var, val, depth = queue.popleft()
                if depth > max_depth:
                    continue

                if depth == 0:
                    p = float(np.clip(val, 0.01, 0.99))
                    parent_vals[var] = val
                    activated = np.random.random() < p
                else:
                    p = self._compute_conditional_probability(var, parent_vals)
                    parent_vals[var] = p
                    activated = np.random.random() < p

                variable_counts[var] += 1
                if activated:
                    variable_probs[var] += 1.0
                    active_prob *= p

                    if var in self.outgoing:
                        next_edges = sorted(
                            [e for e in self.outgoing[var] if e.target not in visited],
                            key=lambda e: -abs(e.conditional_prob),
                        )
                        for edge in next_edges[:3]:
                            if edge.target not in visited and edge.target in self.nodes:
                                visited.add(edge.target)
                                queue.append((edge.target, 0.0, depth + 1))

        result = {}
        for var in variable_counts:
            if variable_counts[var] > 0:
                result[var] = variable_probs[var] / variable_counts[var]
            else:
                result[var] = 0.0

        return result