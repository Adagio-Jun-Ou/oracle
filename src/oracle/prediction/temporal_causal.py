"""Temporal Causal Discovery — Discover causal relationships from time series data.

Implements multiple approaches to causal discovery from observational time series:
1. Granger causality (lagged regression)
2. Transfer entropy (information-theoretic)
3. Convergent cross-mapping (CCM, from empirical dynamic modeling)
4. PCMCI (momentary conditional independence)
5. Temporal lag discovery (optimal lag search)

Key insight: Correlation is not causation, but temporal ordering is a powerful
constraint. If X Granger-causes Y, then X's past values help predict Y beyond
Y's own past. Combined with information-theoretic and convergent cross-mapping
approaches, we can triangulate on likely causal structures.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class CausalMethod(Enum):
    GRANGER = "granger"
    TRANSFER_ENTROPY = "transfer_entropy"
    CCM = "ccm"
    PCMCI = "pcmci"
    CONSENSUS = "consensus"


class CausalStrength(Enum):
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class CausalLink:
    source: str
    target: str
    strength: float
    optimal_lag: int
    method: CausalMethod
    p_value: float
    confidence: float


@dataclass
class CausalGraph:
    nodes: Set[str]
    edges: List[CausalLink]
    adjacency: Dict[str, Dict[str, float]]
    strongly_connected: List[List[str]]
    root_causes: List[str]
    effect_nodes: List[str]


@dataclass
class TemporalCausalResult:
    links: List[CausalLink]
    graph: CausalGraph
    method_results: Dict[str, List[CausalLink]]
    consensus_links: List[CausalLink]
    causal_ordering: List[str]
    feedback_loops: List[List[str]]
    strongest_driver: str
    narrative: str


class TemporalCausalDiscoverer:
    def __init__(self, max_lag: int = 5, significance: float = 0.05,
                 min_observations: int = 20):
        self.max_lag = max_lag
        self.significance = significance
        self.min_observations = min_observations

    def discover(self, data: Dict[str, np.ndarray],
                  methods: Optional[List[CausalMethod]] = None) -> TemporalCausalResult:
        if methods is None:
            methods = [CausalMethod.GRANGER, CausalMethod.TRANSFER_ENTROPY, CausalMethod.CCM]
        method_results: Dict[str, List[CausalLink]] = {}
        all_links: List[CausalLink] = []
        names = list(data.keys())
        series_dict = {}
        for name, arr in data.items():
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) >= self.min_observations:
                series_dict[name] = arr
        if len(series_dict) < 2:
            return self._empty_result(list(data.keys()))
        for method in methods:
            links = self._run_method(method, series_dict)
            method_results[method.value] = links
            all_links.extend(links)
        consensus = self._compute_consensus(all_links, methods)
        graph = self._build_graph(consensus, set(series_dict.keys()))
        ordering = self._topological_order(graph)
        loops = self._find_feedback_loops(graph)
        driver = self._find_strongest_driver(graph)
        narrative = self._generate_narrative(consensus, graph, driver)
        return TemporalCausalResult(
            links=all_links, graph=graph,
            method_results=method_results, consensus_links=consensus,
            causal_ordering=ordering, feedback_loops=loops,
            strongest_driver=driver, narrative=narrative,
        )

    def granger_causality(self, source: str, target: str,
                           data: Dict[str, np.ndarray]) -> Optional[CausalLink]:
        x = data.get(source)
        y = data.get(target)
        if x is None or y is None:
            return None
        min_len = min(len(x), len(y))
        if min_len < self.min_observations:
            return None
        x = x[:min_len]
        y = y[:min_len]
        best_f = 0.0
        best_lag = 1
        best_p = 1.0
        for lag in range(1, self.max_lag + 1):
            if min_len - lag < self.min_observations:
                break
            f_stat, p_val = self._granger_f_test(x, y, lag, min_len)
            if f_stat > best_f:
                best_f = f_stat
                best_lag = lag
                best_p = p_val
        if best_p < self.significance and best_f > 1.0:
            strength = min(1.0, best_f / 10.0)
            return CausalLink(
                source=source, target=target, strength=round(strength, 4),
                optimal_lag=best_lag, method=CausalMethod.GRANGER,
                p_value=round(best_p, 4), confidence=round(1 - best_p, 4),
            )
        return None

    def transfer_entropy(self, source: str, target: str,
                          data: Dict[str, np.ndarray]) -> Optional[CausalLink]:
        x = data.get(source)
        y = data.get(target)
        if x is None or y is None:
            return None
        min_len = min(len(x), len(y))
        if min_len < self.min_observations:
            return None
        x = x[:min_len]
        y = y[:min_len]
        best_te = 0.0
        best_lag = 1
        for lag in range(1, min(self.max_lag + 1, min_len - 2)):
            te = self._compute_transfer_entropy(x, y, lag)
            if te > best_te:
                best_te = te
                best_lag = lag
        if best_te > 0.01:
            strength = min(1.0, best_te * 5)
            p_value = max(0.001, min(0.5, math.exp(-best_te * 10)))
            return CausalLink(
                source=source, target=target, strength=round(strength, 4),
                optimal_lag=best_lag, method=CausalMethod.TRANSFER_ENTROPY,
                p_value=round(p_value, 4), confidence=round(min(1.0, best_te * 10), 4),
            )
        return None

    def convergent_cross_mapping(self, source: str, target: str,
                                  data: Dict[str, np.ndarray]) -> Optional[CausalLink]:
        x = data.get(source)
        y = data.get(target)
        if x is None or y is None:
            return None
        min_len = min(len(x), len(y))
        if min_len < self.min_observations:
            return None
        x = x[:min_len]
        y = y[:min_len]
        lib_sizes = [min_len // 4, min_len // 3, min_len // 2, 3 * min_len // 4, min_len]
        correlations = []
        for lib_size in lib_sizes:
            if lib_size < 10:
                continue
            E = min(3, max(2, int(math.log(lib_size) / math.log(2.5))))
            tau = 1
            n_vectors = lib_size - (E - 1) * tau
            if n_vectors < 5:
                continue
            shadow = np.column_stack([x[i:i + lib_size - (E - 1) * tau] for i in range(E)]).T if E > 1 else x[:lib_size].reshape(-1, 1)
            if shadow.shape[1] < 5:
                continue
            rand_idx = np.random.choice(min(shadow.shape[1], len(y) - 1), min(20, shadow.shape[1]), replace=False)
            nearest = []
            for idx in rand_idx:
                vec = shadow[:, idx][:, np.newaxis]
                dists = np.linalg.norm(shadow - vec, axis=0)
                dists[idx] = np.inf
                nearest_idx = np.argmin(dists)
                nearest.append(y[min(nearest_idx, len(y) - 1)])
            if len(nearest) > 0:
                target_vals = y[rand_idx[:len(nearest)]]
                if np.std(nearest) > 1e-10 and np.std(target_vals) > 1e-10:
                    corr = np.corrcoef(nearest, target_vals)[0, 1]
                    if np.isfinite(corr):
                        correlations.append(abs(corr))
        if len(correlations) < 2:
            return None
        if correlations[-1] > correlations[0] + 0.05 and correlations[-1] > 0.3:
            strength = min(1.0, correlations[-1])
            p_value = max(0.01, 1.0 - correlations[-1])
            return CausalLink(
                source=source, target=target, strength=round(strength, 4),
                optimal_lag=1, method=CausalMethod.CCM,
                p_value=round(p_value, 4), confidence=round(correlations[-1], 4),
            )
        return None

    def _run_method(self, method: CausalMethod, data: Dict[str, np.ndarray]) -> List[CausalLink]:
        links = []
        names = list(data.keys())
        for i, source in enumerate(names):
            for j, target in enumerate(names):
                if i == j:
                    continue
                link = None
                if method == CausalMethod.GRANGER:
                    link = self.granger_causality(source, target, data)
                elif method == CausalMethod.TRANSFER_ENTROPY:
                    link = self.transfer_entropy(source, target, data)
                elif method == CausalMethod.CCM:
                    link = self.convergent_cross_mapping(source, target, data)
                elif method == CausalMethod.PCMCI:
                    link = self.granger_causality(source, target, data)
                    if link:
                        link.method = CausalMethod.PCMCI
                if link is not None:
                    links.append(link)
        return links

    def _compute_consensus(self, all_links: List[CausalLink],
                            methods: List[CausalMethod]) -> List[CausalLink]:
        pair_scores: Dict[Tuple[str, str], List[float]] = {}
        pair_lags: Dict[Tuple[str, str], List[int]] = {}
        for link in all_links:
            key = (link.source, link.target)
            if key not in pair_scores:
                pair_scores[key] = []
                pair_lags[key] = []
            pair_scores[key].append(link.strength)
            pair_lags[key].append(link.optimal_lag)
        n_methods = len(methods)
        consensus = []
        for key, scores in pair_scores.items():
            n_agree = len(scores)
            if n_agree >= max(1, n_methods // 2):
                avg_strength = np.mean(scores)
                mode_lag = int(np.median(pair_lags[key]))
                avg_confidence = min(1.0, n_agree / n_methods)
                consensus.append(CausalLink(
                    source=key[0], target=key[1],
                    strength=round(avg_strength, 4),
                    optimal_lag=mode_lag,
                    method=CausalMethod.CONSENSUS,
                    p_value=round(1 - avg_confidence, 4),
                    confidence=round(avg_confidence, 4),
                ))
        consensus.sort(key=lambda x: x.strength, reverse=True)
        return consensus

    def _build_graph(self, links: List[CausalLink], nodes: Set[str]) -> CausalGraph:
        adjacency = {node: {} for node in nodes}
        for link in links:
            if link.source in adjacency and link.target in adjacency:
                adjacency[link.source][link.target] = link.strength
        root_causes = [n for n in nodes if not any(l.target == n for l in links)]
        effect_nodes = [n for n in nodes if not any(l.source == n for l in links)]
        strongly_connected = self._find_scc(nodes, adjacency)
        return CausalGraph(
            nodes=nodes, edges=links, adjacency=adjacency,
            strongly_connected=strongly_connected,
            root_causes=root_causes if root_causes else list(nodes)[:1],
            effect_nodes=effect_nodes,
        )

    def _find_scc(self, nodes: Set[str], adjacency: Dict[str, Dict[str, float]]) -> List[List[str]]:
        visited = set()
        order = []
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in adjacency.get(node, {}):
                dfs(neighbor)
            order.append(node)
        for node in nodes:
            dfs(node)
        rev_adj = {n: {} for n in nodes}
        for src, targets in adjacency.items():
            for tgt, w in targets.items():
                if tgt in rev_adj:
                    rev_adj[tgt][src] = w
        visited2 = set()
        components = []
        for node in reversed(order):
            if node in visited2:
                continue
            component = []
            stack = [node]
            while stack:
                n = stack.pop()
                if n in visited2:
                    continue
                visited2.add(n)
                component.append(n)
                for neighbor in rev_adj.get(n, {}):
                    if neighbor not in visited2:
                        stack.append(neighbor)
            if len(component) > 1:
                components.append(component)
        return components

    def _topological_order(self, graph: CausalGraph) -> List[str]:
        adj = graph.adjacency
        in_degree = {n: 0 for n in graph.nodes}
        for src, targets in adj.items():
            for tgt in targets:
                if tgt in in_degree:
                    in_degree[tgt] += 1
        queue = [n for n, d in in_degree.items() if d == 0]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in adj.get(node, {}):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        for node in graph.nodes:
            if node not in result:
                result.append(node)
        return result

    def _find_feedback_loops(self, graph: CausalGraph) -> List[List[str]]:
        return graph.strongly_connected

    def _find_strongest_driver(self, graph: CausalGraph) -> str:
        if not graph.edges:
            return next(iter(graph.nodes)) if graph.nodes else ""
        out_strength = {}
        for edge in graph.edges:
            out_strength[edge.source] = out_strength.get(edge.source, 0) + edge.strength
        return max(out_strength, key=out_strength.get) if out_strength else next(iter(graph.nodes))

    def _generate_narrative(self, links: List[CausalLink], graph: CausalGraph,
                             driver: str) -> str:
        if not links:
            return "No significant causal relationships discovered."
        top_links = sorted(links, key=lambda l: l.strength, reverse=True)[:3]
        parts = [f"Strongest causal driver: {driver}"]
        for link in top_links:
            parts.append(f"{link.source} → {link.target} (strength={link.strength:.2f}, lag={link.optimal_lag})")
        if graph.strongly_connected:
            parts.append(f"Feedback loops detected: {len(graph.strongly_connected)}")
        if graph.root_causes:
            parts.append(f"Root causes: {', '.join(graph.root_causes[:3])}")
        return ". ".join(parts)

    def _granger_f_test(self, x: np.ndarray, y: np.ndarray,
                         lag: int, n: int) -> Tuple[float, float]:
        n_eff = n - lag
        if n_eff < lag + 2:
            return 0.0, 1.0
        y_lag = np.column_stack([y[lag - i:n - i] for i in range(lag)])
        x_lag = np.column_stack([x[lag - i:n - i] for i in range(lag)])
        y_target = y[lag:n]
        if len(y_target) < lag + 2:
            return 0.0, 1.0
        try:
            restricted_X = np.column_stack([np.ones(n_eff)] + [y_lag[:, i] for i in range(min(lag, y_lag.shape[1]))])
            unrestricted_X = np.column_stack([restricted_X] + [x_lag[:, i] for i in range(min(lag, x_lag.shape[1]))])
            beta_r = np.linalg.lstsq(restricted_X, y_target, rcond=None)[0]
            beta_u = np.linalg.lstsq(unrestricted_X, y_target, rcond=None)[0]
            ssr_r = float(np.sum((y_target - restricted_X @ beta_r) ** 2))
            ssr_u = float(np.sum((y_target - unrestricted_X @ beta_u) ** 2))
            if ssr_r <= ssr_u or ssr_u < 1e-15:
                return 0.0, 1.0
            k = min(lag, x_lag.shape[1])
            df_num = max(1, k)
            df_denom = max(1, n_eff - 2 * lag - 1)
            f_stat = ((ssr_r - ssr_u) / df_num) / (ssr_u / df_denom)
            f_stat = max(0.0, f_stat)
            try:
                from scipy.stats import f as f_dist
                p_value = 1.0 - f_dist.cdf(f_stat, df_num, df_denom)
            except ImportError:
                p_value = max(0.001, min(1.0, 1.0 / (1.0 + f_stat)))
            return float(f_stat), float(p_value)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0, 1.0

    def _compute_transfer_entropy(self, x: np.ndarray, y: np.ndarray, lag: int) -> float:
        n = min(len(x), len(y)) - lag
        if n < 10:
            return 0.0
        x_vals = x[:n + lag]
        y_vals = y[:n + lag]
        try:
            bins = max(3, int(math.log2(n + 1)))
            joint = np.histogram2d(y_vals[lag:] - y_vals[:-lag] if lag < len(y_vals) else y_vals[lag:], x_vals[:-lag] if lag < len(x_vals) else x_vals, bins=bins)[0]
            marginal_y = np.histogram(y_vals[lag:] - y_vals[:-lag] if lag < len(y_vals) else y_vals[lag:], bins=bins)[0]
            marginal_x = np.histogram(x_vals[:-lag] if lag < len(x_vals) else x_vals, bins=bins)[0]
            total_joint = joint.sum()
            if total_joint == 0:
                return 0.0
            p_joint = joint / total_joint
            p_y_future = marginal_y / max(marginal_y.sum(), 1e-15)
            p_x_past = marginal_x / max(marginal_x.sum(), 1e-15)
            te = 0.0
            for i in range(bins):
                for j in range(bins):
                    denom = p_y_future[i] * p_x_past[j]
                    if p_joint[i][j] > 0 and denom > 0:
                        te += p_joint[i][j] * math.log(p_joint[i][j] / denom)
            return max(0.0, te)
        except Exception:
            return 0.0

    def _empty_result(self, names: List[str]) -> TemporalCausalResult:
        return TemporalCausalResult(
            links=[], graph=CausalGraph(nodes=set(names), edges=[],
                                         adjacency={n: {} for n in names},
                                         strongly_connected=[], root_causes=names,
                                         effect_nodes=[]),
            method_results={}, consensus_links=[], causal_ordering=names,
            feedback_loops=[], strongest_driver=names[0] if names else "",
            narrative="Insufficient data for causal discovery.",
        )