"""Cascading Failure Forecaster — multi-step failure cascade trees with propagation probabilities.

Builds directed graphs of failure propagation paths and simulates cascades using
percolation theory and contagion models. Monte Carlo simulation estimates cascade
reach, critical paths, domain-level failure probability, and systemic risk.

Implements:
- FailureNode: represents a failure event
- FailureEdge: propagation link between failures
- CascadeTree: directed graph with simulation
- CascadeResult: result dataclass
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class FailureNode:
    name: str
    domain: str
    probability: float
    impact_severity: float

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, FailureNode):
            return self.name == other.name
        return NotImplemented


@dataclass
class FailureEdge:
    source: str
    target: str
    propagation_probability: float
    time_delay_months: float
    mechanism: str


@dataclass
class CascadeResult:
    trigger: str
    cascade_paths: List[List[str]]
    total_probability: float
    n_domains_affected: int
    max_depth: int
    mean_impact: float
    risk_level: str

    def to_dict(self) -> Dict:
        return {
            "trigger": self.trigger,
            "cascade_paths": self.cascade_paths,
            "total_probability": self.total_probability,
            "n_domains_affected": self.n_domains_affected,
            "max_depth": self.max_depth,
            "mean_impact": round(self.mean_impact, 4),
            "risk_level": self.risk_level,
        }


class CascadeTree:
    def __init__(self):
        self.nodes: Dict[str, FailureNode] = {}
        self.edges: List[FailureEdge] = []
        self.adjacency: Dict[str, List[FailureEdge]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[FailureEdge]] = defaultdict(list)

    def add_node(self, failure: FailureNode) -> None:
        self.nodes[failure.name] = failure

    def add_edge(self, source: str, target: str, probability: float,
                 delay: float, mechanism: str) -> None:
        if source not in self.nodes or target not in self.nodes:
            logger.warning("Edge references unknown node: %s -> %s", source, target)
            return
        probability = float(np.clip(probability, 0.0, 1.0))
        edge = FailureEdge(
            source=source,
            target=target,
            propagation_probability=probability,
            time_delay_months=delay,
            mechanism=mechanism,
        )
        self.edges.append(edge)
        self.adjacency[source].append(edge)
        self.reverse_adjacency[target].append(edge)

    def simulate_cascade(self, trigger_node: str, n_simulations: int = 1000,
                         rng: Optional[np.random.Generator] = None) -> CascadeResult:
        if trigger_node not in self.nodes:
            raise ValueError(f"Unknown trigger node: {trigger_node}")
        if rng is None:
            rng = np.random.default_rng()

        all_paths: List[List[str]] = []
        total_activated_count = 0
        max_depth_observed = 0
        impact_sum = 0.0
        domains_affected_set: Set[str] = set()
        cascades_reaching_beyond_trigger = 0

        for _ in range(n_simulations):
            activated: Set[str] = {trigger_node}
            frontier = deque([(trigger_node, 1)])
            current_paths: Dict[str, List[str]] = {trigger_node: [trigger_node]}
            local_max_depth = 0

            while frontier:
                node_name, depth = frontier.popleft()
                local_max_depth = max(local_max_depth, depth)
                for edge in self.adjacency.get(node_name, []):
                    if edge.target not in activated:
                        if rng.random() < edge.propagation_probability:
                            activated.add(edge.target)
                            current_path = current_paths.get(node_name, [node_name])
                            current_paths[edge.target] = current_path + [edge.target]
                            frontier.append((edge.target, depth + 1))

            if len(activated) > 1:
                cascades_reaching_beyond_trigger += 1

            for node_name in activated:
                node = self.nodes[node_name]
                domains_affected_set.add(node.domain)
                impact_sum += node.impact_severity * node.probability

            activated_paths = [p for name, p in current_paths.items() if len(p) > 1]
            if not activated_paths:
                longest = [trigger_node]
                for name, p in current_paths.items():
                    if len(p) > len(longest):
                        longest = p
                activated_paths = [longest]

            all_paths.extend(activated_paths)
            total_activated_count += len(activated)
            max_depth_observed = max(max_depth_observed, local_max_depth)

        total_probability = cascades_reaching_beyond_trigger / n_simulations
        mean_impact = impact_sum / n_simulations
        n_domains = len(domains_affected_set)

        if total_probability < 0.2:
            risk_level = "LOW"
        elif total_probability < 0.4:
            risk_level = "MODERATE"
        elif total_probability < 0.6:
            risk_level = "HIGH"
        elif total_probability < 0.8:
            risk_level = "VERY HIGH"
        else:
            risk_level = "CRITICAL"

        path_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        for p in all_paths:
            path_counts[tuple(p)] += 1
        top_paths = sorted(path_counts.items(), key=lambda x: -x[1])[:20]
        representative_paths = [list(k) for k, _ in top_paths]

        logger.info(
            "Cascade from %s: P=%.3f, domains=%d, max_depth=%d, risk=%s",
            trigger_node, total_probability, n_domains, max_depth_observed, risk_level,
        )

        return CascadeResult(
            trigger=trigger_node,
            cascade_paths=representative_paths,
            total_probability=round(total_probability, 4),
            n_domains_affected=n_domains,
            max_depth=max_depth_observed,
            mean_impact=round(mean_impact, 4),
            risk_level=risk_level,
        )

    def critical_paths(self, top_n: int = 5) -> List[Dict]:
        all_source_nodes = set(e.source for e in self.edges)
        source_only = all_source_nodes - set(e.target for e in self.edges)
        root_nodes = source_only if source_only else all_source_nodes

        all_paths: List[Tuple[List[str], float, float]] = []

        for root in root_nodes:
            stack = [(root, [root], 1.0, 0.0, {root})]
            while stack:
                current, path, cumulative_prob, cumulative_delay, visited = stack.pop()
                out_edges = self.adjacency.get(current, [])
                if not out_edges:
                    if len(path) > 1:
                        all_paths.append((path, cumulative_prob, cumulative_delay))
                    continue
                has_unvisited = False
                for edge in out_edges:
                    if edge.target in visited:
                        continue
                    new_prob = cumulative_prob * edge.propagation_probability
                    new_delay = cumulative_delay + edge.time_delay_months
                    if new_prob < 1e-6:
                        continue
                    has_unvisited = True
                    new_path = path + [edge.target]
                    new_visited = visited | {edge.target}
                    stack.append((edge.target, new_path, new_prob, new_delay, new_visited))
                if not has_unvisited and len(path) > 1:
                    all_paths.append((path, cumulative_prob, cumulative_delay))

        scored = []
        for path, cum_prob, cum_delay in all_paths:
            total_impact = sum(self.nodes[n].impact_severity for n in path)
            domains = len(set(self.nodes[n].domain for n in path))
            score = cum_prob * total_impact * (1 + 0.1 * domains)
            path_mechanisms = []
            for i in range(len(path) - 1):
                for edge in self.adjacency.get(path[i], []):
                    if edge.target == path[i + 1]:
                        path_mechanisms.append(edge.mechanism)
                        break
            scored.append({
                "path": path,
                "cumulative_probability": round(cum_prob, 6),
                "cumulative_delay_months": round(cum_delay, 2),
                "domains_crossed": domains,
                "total_impact": round(total_impact, 4),
                "risk_score": round(score, 4),
                "mechanisms": path_mechanisms,
            })

        scored.sort(key=lambda x: -x["risk_score"])
        return scored[:top_n]

    def cascade_probability(self, target_domain: str,
                            n_simulations: int = 2000) -> Dict[str, float]:
        domain_nodes = [
            n.name for n in self.nodes.values() if n.domain == target_domain
        ]
        if not domain_nodes:
            logger.warning("No nodes in domain: %s", target_domain)
            return {}

        root_candidates = set(self.nodes.keys()) - set(
            e.target for e in self.edges
        )
        if not root_candidates:
            root_candidates = set(self.nodes.keys())

        trigger_probs: Dict[str, float] = {}
        for trigger in root_candidates:
            n_hit = 0
            rng = np.random.default_rng()
            for _ in range(n_simulations):
                activated: Set[str] = {trigger}
                frontier = deque([trigger])
                while frontier:
                    current = frontier.popleft()
                    for edge in self.adjacency.get(current, []):
                        if edge.target not in activated:
                            if rng.random() < edge.propagation_probability:
                                activated.add(edge.target)
                                frontier.append(edge.target)
                if activated & set(domain_nodes):
                    n_hit += 1
            trigger_probs[trigger] = round(n_hit / n_simulations, 4)

        return trigger_probs

    def systemic_risk_score(self, n_simulations: int = 2000) -> Dict:
        all_domains = list(set(n.domain for n in self.nodes.values()))
        root_candidates = set(self.nodes.keys()) - set(
            e.target for e in self.edges
        )
        if not root_candidates:
            root_candidates = set(self.nodes.keys())

        rng = np.random.default_rng()
        cascade_sizes: List[int] = []
        domain_hits: Dict[str, int] = defaultdict(int)
        node_hits: Dict[str, int] = defaultdict(int)
        total_sims = n_simulations * max(len(root_candidates), 1)

        for trigger in root_candidates:
            for _ in range(n_simulations):
                activated: Set[str] = {trigger}
                frontier = deque([trigger])
                while frontier:
                    current = frontier.popleft()
                    for edge in self.adjacency.get(current, []):
                        if edge.target not in activated:
                            if rng.random() < edge.propagation_probability:
                                activated.add(edge.target)
                                frontier.append(edge.target)
                cascade_sizes.append(len(activated))
                for node_name in activated:
                    node_hits[node_name] += 1
                    domain_hits[self.nodes[node_name].domain] += 1

        if not cascade_sizes:
            return {"systemic_risk_score": 0.0, "risk_level": "UNKNOWN"}

        mean_cascade = np.mean(cascade_sizes)
        max_cascade = np.max(cascade_sizes)
        p_large_cascade = np.mean(np.array(cascade_sizes) > len(self.nodes) * 0.3)
        percolation_threshold = mean_cascade / len(self.nodes) if self.nodes else 0.0

        activated_nodes = [n for n in self.nodes if n in node_hits]
        weighted_cascade = (
            sum(self.nodes[n].impact_severity for n in activated_nodes)
            / max(len(activated_nodes), 1)
        )

        systemic_score = (
            percolation_threshold * 40.0
            + p_large_cascade * 30.0
            + min(mean_cascade / len(self.nodes), 1.0) * 20.0
            + min(weighted_cascade / (len(self.nodes) * 5.0), 1.0) * 10.0
        )
        systemic_score = float(np.clip(systemic_score, 0.0, 100.0))

        if systemic_score < 20:
            risk_level = "LOW"
        elif systemic_score < 40:
            risk_level = "MODERATE"
        elif systemic_score < 60:
            risk_level = "HIGH"
        elif systemic_score < 80:
            risk_level = "VERY HIGH"
        else:
            risk_level = "CRITICAL"

        domain_vulnerability = []
        for domain in all_domains:
            domain_node_count = sum(
                1 for n in self.nodes.values() if n.domain == domain
            )
            hit_rate = domain_hits.get(domain, 0) / max(total_sims, 1)
            domain_vulnerability.append({
                "domain": domain,
                "hit_rate": round(hit_rate, 4),
                "vulnerability_score": round(hit_rate * 100, 2),
            })
        domain_vulnerability.sort(key=lambda x: -x["vulnerability_score"])

        node_criticality = []
        for node_name, hit_count in sorted(node_hits.items(), key=lambda x: -x[1]):
            node = self.nodes[node_name]
            criticality = (hit_count / max(total_sims, 1)) * node.impact_severity
            node_criticality.append({
                "node": node_name,
                "domain": node.domain,
                "activation_rate": round(hit_count / max(total_sims, 1), 4),
                "impact_severity": node.impact_severity,
                "criticality_score": round(criticality, 4),
            })

        super_spreader_scores = {}
        for node_name in self.nodes:
            out_degree = len(self.adjacency.get(node_name, []))
            if out_degree > 0:
                avg_prop = np.mean([
                    e.propagation_probability
                    for e in self.adjacency[node_name]
                ])
                downstream = self._count_downstream(node_name)
                super_spreader_scores[node_name] = (
                    out_degree * avg_prop * np.log1p(downstream)
                )

        top_spreaders = sorted(
            super_spreader_scores.items(), key=lambda x: -x[1]
        )[:5]

        return {
            "systemic_risk_score": round(systemic_score, 2),
            "risk_level": risk_level,
            "mean_cascade_size": round(float(mean_cascade), 2),
            "max_cascade_size": int(max_cascade),
            "p_large_cascade": round(float(p_large_cascade), 4),
            "percolation_ratio": round(float(percolation_threshold), 4),
            "domain_vulnerability": domain_vulnerability,
            "node_criticality": node_criticality[:10],
            "super_spreaders": [
                {"node": n, "score": round(s, 4)} for n, s in top_spreaders
            ],
        }

    def _count_downstream(self, node_name: str) -> int:
        visited: Set[str] = set()
        stack = [node_name]
        count = 0
        while stack:
            current = stack.pop()
            for edge in self.adjacency.get(current, []):
                if edge.target not in visited:
                    visited.add(edge.target)
                    stack.append(edge.target)
                    count += 1
        return count

    @classmethod
    def build_default(cls) -> "CascadeTree":
        tree = cls()

        nodes = [
            FailureNode("banking_crisis", "economics", 0.15, 9.0),
            FailureNode("credit_freeze", "economics", 0.25, 7.0),
            FailureNode("trade_disruption", "economics", 0.2, 6.5),
            FailureNode("supply_chain_failure", "supply_chain", 0.3, 8.0),
            FailureNode("social_unrest", "social", 0.25, 7.5),
            FailureNode("oil_price_shock", "energy", 0.2, 7.0),
            FailureNode("inflation_spike", "economics", 0.25, 6.0),
            FailureNode("interest_rate_hike", "economics", 0.2, 5.5),
            FailureNode("recession", "economics", 0.15, 9.5),
            FailureNode("pandemic_outbreak", "health", 0.1, 9.0),
            FailureNode("climate_disaster", "climate", 0.2, 8.5),
            FailureNode("food_shortage", "supply_chain", 0.2, 8.0),
            FailureNode("cyber_attack_infra", "technology", 0.15, 8.0),
            FailureNode("power_grid_failure", "energy", 0.1, 9.0),
            FailureNode("geopolitical_conflict", "geopolitics", 0.2, 8.5),
            FailureNode("refugee_crisis", "social", 0.2, 7.0),
            FailureNode("sovereign_default", "economics", 0.1, 9.5),
            FailureNode("currency_crisis", "economics", 0.15, 8.0),
        ]

        for n in nodes:
            tree.add_node(n)

        edges = [
            ("banking_crisis", "credit_freeze", 0.8, 1, "liquidity_evaporation"),
            ("credit_freeze", "trade_disruption", 0.6, 2, "trade_finance_collapse"),
            ("trade_disruption", "supply_chain_failure", 0.7, 3, "logistics_breakdown"),
            ("supply_chain_failure", "social_unrest", 0.4, 4, "shortage_anger"),
            ("oil_price_shock", "inflation_spike", 0.7, 2, "cost_push_inflation"),
            ("inflation_spike", "interest_rate_hike", 0.8, 3, "central_bank_response"),
            ("interest_rate_hike", "recession", 0.6, 6, "demand_destruction"),
            ("pandemic_outbreak", "supply_chain_failure", 0.6, 2, "workforce_and_lockdown"),
            ("climate_disaster", "food_shortage", 0.5, 3, "crop_destruction"),
            ("food_shortage", "social_unrest", 0.6, 2, "hunger_protests"),
            ("geopolitical_conflict", "oil_price_shock", 0.6, 2, "supply_disruption_fear"),
            ("cyber_attack_infra", "power_grid_failure", 0.5, 1, "scada_compromise"),
            ("power_grid_failure", "supply_chain_failure", 0.5, 1, "manufacturing_halt"),
            ("geopolitical_conflict", "trade_disruption", 0.5, 2, "sanctions_embargo"),
            ("recession", "social_unrest", 0.5, 4, "unemployment_despair"),
            ("pandemic_outbreak", "social_unrest", 0.4, 3, "lockdown_fatigue"),
            ("climate_disaster", "refugee_crisis", 0.6, 3, "displacement"),
            ("refugee_crisis", "social_unrest", 0.4, 3, "integration_strain"),
            ("banking_crisis", "sovereign_default", 0.3, 6, "bailout_fiscal_drain"),
            ("sovereign_default", "currency_crisis", 0.7, 2, "investor_flight"),
            ("currency_crisis", "inflation_spike", 0.6, 1, "import_cost_surge"),
            ("currency_crisis", "banking_crisis", 0.3, 3, "foreign_debt_unserviceable"),
            ("interest_rate_hike", "sovereign_default", 0.3, 4, "debt_servicing_burden"),
            ("recession", "banking_crisis", 0.4, 5, "loan_defaults_spike"),
            ("power_grid_failure", "social_unrest", 0.3, 2, "basic_services_loss"),
            ("cyber_attack_infra", "banking_crisis", 0.35, 1, "financial_infra_compromise"),
            ("social_unrest", "geopolitical_conflict", 0.3, 4, "regional_destabilization"),
            ("food_shortage", "refugee_crisis", 0.4, 3, "famine_migration"),
        ]

        for source, target, prob, delay, mech in edges:
            tree.add_edge(source, target, prob, delay, mech)

        return tree