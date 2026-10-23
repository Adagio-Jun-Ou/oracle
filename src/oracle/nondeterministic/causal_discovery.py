"""Causal Discovery module for learning causal graph structure from data.

Implements:
- PC algorithm (constraint-based): conditional independence testing for skeleton,
  orientation rules for edge direction
- GES (score-based): greedy equivalence search with BIC scoring
- Causal graph representation and validation
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set


@dataclass
class CausalEdge:
    from_node: int
    to_node: int
    edge_type: str  # "directed", "undirected", "bidirected"
    confidence: float = 1.0


@dataclass
class CausalGraph:
    n_nodes: int
    edges: List[CausalEdge]
    adjacency: List[List[int]]
    node_names: List[str]
    method: str = "pc_algorithm"

    def get_parents(self, node: int) -> List[int]:
        return [e.from_node for e in self.edges if e.to_node == node and e.edge_type == "directed"]

    def get_children(self, node: int) -> List[int]:
        return [e.to_node for e in self.edges if e.from_node == node and e.edge_type == "directed"]

    def get_neighbors(self, node: int) -> List[int]:
        neighbors = set()
        for e in self.edges:
            if e.from_node == node:
                neighbors.add(e.to_node)
            if e.to_node == node:
                neighbors.add(e.from_node)
        return list(neighbors)

    def to_adjacency_matrix(self) -> np.ndarray:
        adj = np.zeros((self.n_nodes, self.n_nodes))
        for e in self.edges:
            if e.edge_type == "directed":
                adj[e.from_node][e.to_node] = 1
            elif e.edge_type == "undirected":
                adj[e.from_node][e.to_node] = 0.5
                adj[e.to_node][e.from_node] = 0.5
            elif e.edge_type == "bidirected":
                adj[e.from_node][e.to_node] = 1
                adj[e.to_node][e.from_node] = 1
        return adj

    def is_dag(self) -> bool:
        adj = self.to_adjacency_matrix()
        n = self.n_nodes
        visited = [0] * n
        rec_stack = [0] * n

        def dfs(v):
            visited[v] = 1
            rec_stack[v] = 1
            for u in range(n):
                if adj[v][u] > 0.3 and u != v:
                    if not visited[u]:
                        if dfs(u):
                            return True
                    elif rec_stack[u]:
                        return True
            rec_stack[v] = 0
            return False

        for i in range(n):
            if not visited[i]:
                if dfs(i):
                    return False
        return True

    def topological_sort(self) -> List[int]:
        adj = self.to_adjacency_matrix()
        n = self.n_nodes
        in_degree = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j] > 0.5 and adj[j][i] < 0.5:
                    in_degree[j] += 1
        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for j in range(n):
                if adj[node][j] > 0.5 and adj[j][node] < 0.5:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)
        return order


@dataclass
class CausalDiscoveryResult:
    graph: CausalGraph
    n_edges: int
    n_directed: int
    n_undirected: int
    independence_tests: int
    alpha: float
    log_likelihood: float
    bic_score: float
    method: str = "pc_algorithm"


class PCAlgorithm:
    """PC algorithm for constraint-based causal discovery.

    Phase 1: Learn skeleton via conditional independence tests
    Phase 2: Orient edges using v-structures and orientation rules
    """

    def __init__(self, alpha: float = 0.05, max_cond_set: int = 3,
                 test_type: str = "partial_corr"):
        self.alpha = alpha
        self.max_cond_set = max_cond_set
        self.test_type = test_type
        self._n_tests = 0

    def _partial_corr_test(self, data: np.ndarray, i: int, j: int,
                           cond_set: List[int]) -> float:
        n = data.shape[0]
        if len(cond_set) == 0:
            r = np.corrcoef(data[:, i], data[:, j])[0, 1]
        elif len(cond_set) == 1:
            k = cond_set[0]
            ri = np.corrcoef(data[:, i], data[:, k])[0, 1]
            rj = np.corrcoef(data[:, j], data[:, k])[0, 1]
            denom = np.sqrt((1 - ri**2) * (1 - rj**2))
            r = (np.corrcoef(data[:, i], data[:, j])[0, 1] - ri * rj) / max(denom, 1e-10)
        else:
            cols = [i, j] + list(cond_set)
            sub = data[:, cols]
            cov = np.cov(sub, rowvar=False)
            try:
                prec = np.linalg.inv(cov)
                r = -prec[0, 1] / np.sqrt(max(prec[0, 0] * prec[1, 1], 1e-10))
            except np.linalg.LinAlgError:
                r = 0.0
        r = max(min(r, 0.999), -0.999)
        self._n_tests += 1
        import math
        z = 0.5 * math.log((1 + r) / (1 - r + 1e-10))
        df = max(n - len(cond_set) - 3, 1)
        p_value = 2.0 * (1.0 - self._norm_cdf(abs(z) * math.sqrt(df)))
        return p_value

    def _norm_cdf(self, x: float) -> float:
        import math
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    def _ci_test(self, data: np.ndarray, i: int, j: int,
                 cond_set: List[int]) -> bool:
        p = self._partial_corr_test(data, i, j, cond_set)
        return p > self.alpha

    def fit(self, data: np.ndarray,
            node_names: Optional[List[str]] = None) -> CausalDiscoveryResult:
        n_vars = data.shape[1]
        if node_names is None:
            node_names = [f"X{i}" for i in range(n_vars)]
        self._n_tests = 0
        adj = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        sep_sets: Dict[Tuple[int, int], Set[int]] = {}
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                sep_sets[(i, j)] = set()
        for depth in range(min(self.max_cond_set + 1, n_vars)):
            removed = []
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if adj[i, j] == 0:
                        continue
                    neighbors_i = [k for k in range(n_vars) if adj[i, k] > 0 and k != j]
                    from itertools import combinations
                    cond_candidates = list(combinations(neighbors_i, min(depth + 1, len(neighbors_i))))
                    if depth + 1 <= len(neighbors_i):
                        cond_candidates = list(combinations(neighbors_i, depth + 1))
                    for cond in cond_candidates:
                        if len(cond) > self.max_cond_set:
                            continue
                        if self._ci_test(data, i, j, list(cond)):
                            adj[i, j] = 0
                            adj[j, i] = 0
                            sep_sets[(i, j)] = set(cond)
                            removed.append((i, j))
                            break
                    if (i, j) not in removed:
                        neighbors_j = [k for k in range(n_vars) if adj[j, k] > 0 and k != i]
                        cond_candidates_j = list(combinations(neighbors_j, min(depth + 1, len(neighbors_j))))
                        if depth + 1 <= len(neighbors_j):
                            cond_candidates_j = list(combinations(neighbors_j, depth + 1))
                        for cond in cond_candidates_j:
                            if len(cond) > self.max_cond_set:
                                continue
                            if self._ci_test(data, j, i, list(cond)):
                                adj[i, j] = 0
                                adj[j, i] = 0
                                sep_sets[(i, j)] = set(cond)
                                removed.append((i, j))
                                break
            if not removed and depth > 0:
                break
        edge_types = np.full((n_vars, n_vars), "", dtype=object)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj[i, j] > 0:
                    edge_types[i][j] = "undirected"
                    edge_types[j][i] = "undirected"
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj[i, j] == 0:
                    continue
                for k in range(n_vars):
                    if k == i or k == j:
                        continue
                    if adj[i, k] > 0 and adj[j, k] > 0:
                        if k in sep_sets.get((min(i, j), max(i, j)), set()):
                            continue
                        if edge_types[i][k] == "undirected":
                            edge_types[i][k] = "directed"
                        if edge_types[j][k] == "undirected":
                            edge_types[j][k] = "directed"
                        if edge_types[k][i] == "undirected":
                            edge_types[k][i] = ""
                        if edge_types[k][j] == "undirected":
                            edge_types[k][j] = ""
                        break
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                if adj[i, j] == 0:
                    continue
                if edge_types[j][i] == "directed" and edge_types[i][j] == "undirected":
                    edge_types[i][j] = ""
                if edge_types[i][j] == "undirected" and edge_types[j][i] == "undirected":
                    continue
                if edge_types[i][j] == "directed" and edge_types[j][i] == "directed":
                    edge_types[i][j] = "bidirected"
        edges = []
        adj_list = [[0] * n_vars for _ in range(n_vars)]
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                if edge_types[i][j] == "directed":
                    edges.append(CausalEdge(from_node=i, to_node=j, edge_type="directed", confidence=0.8))
                    adj_list[i][j] = 1
                elif edge_types[i][j] == "undirected" and i < j:
                    edges.append(CausalEdge(from_node=i, to_node=j, edge_type="undirected", confidence=0.5))
                    adj_list[i][j] = 1
                    adj_list[j][i] = 1
                elif edge_types[i][j] == "bidirected" and i < j:
                    edges.append(CausalEdge(from_node=i, to_node=j, edge_type="bidirected", confidence=0.3))
                    adj_list[i][j] = 1
                    adj_list[j][i] = 1
        n_directed = sum(1 for e in edges if e.edge_type == "directed")
        n_undirected = sum(1 for e in edges if e.edge_type == "undirected")
        ll = -0.5 * n_vars * data.shape[0] * np.log(2 * np.pi)
        try:
            cov = np.cov(data, rowvar=False)
            sign, logdet = np.linalg.slogdet(cov)
            if sign > 0:
                ll -= 0.5 * data.shape[0] * logdet
        except Exception:
            ll = 0.0
        k = len(edges)
        bic = -2 * ll + k * np.log(data.shape[0])
        graph = CausalGraph(
            n_nodes=n_vars,
            edges=edges,
            adjacency=adj_list,
            node_names=node_names,
            method="pc_algorithm"
        )
        return CausalDiscoveryResult(
            graph=graph,
            n_edges=len(edges),
            n_directed=n_directed,
            n_undirected=n_undirected,
            independence_tests=self._n_tests,
            alpha=self.alpha,
            log_likelihood=float(ll),
            bic_score=float(bic),
            method="pc_algorithm"
        )


class GESAlgorithm:
    """Greedy Equivalence Search for score-based causal discovery.

    Phase 1: Forward step — greedily add edges that improve BIC
    Phase 2: Backward step — greedily remove edges that improve BIC
    """

    def __init__(self, penalty: float = 1.0, max_parents: int = 3):
        self.penalty = penalty
        self.max_parents = max_parents

    def _bic_local(self, data: np.ndarray, child: int,
                   parents: List[int]) -> float:
        n = data.shape[0]
        y = data[:, child]
        if len(parents) == 0:
            rss = np.sum((y - np.mean(y))**2)
            k = 1
        else:
            X = data[:, parents]
            X = np.column_stack([np.ones(n), X])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                rss = np.sum((y - X @ beta)**2)
            except Exception:
                rss = np.sum((y - np.mean(y))**2)
            k = len(parents) + 1
        sigma2 = rss / max(n, 1)
        ll = -n / 2 * np.log(2 * np.pi * max(sigma2, 1e-10)) - n / 2
        bic = -2 * ll + self.penalty * k * np.log(n)
        return bic

    def fit(self, data: np.ndarray,
            node_names: Optional[List[str]] = None) -> CausalDiscoveryResult:
        n_vars = data.shape[1]
        if node_names is None:
            node_names = [f"X{i}" for i in range(n_vars)]
        parents = {i: [] for i in range(n_vars)}
        current_bic = sum(self._bic_local(data, i, []) for i in range(n_vars))
        improved = True
        n_forward_steps = 0
        while improved:
            improved = False
            best_delta = 0
            best_add = None
            for child in range(n_vars):
                for candidate in range(n_vars):
                    if candidate == child or candidate in parents[child]:
                        continue
                    if len(parents[child]) >= self.max_parents:
                        continue
                    new_parents = parents[child] + [candidate]
                    if self._would_create_cycle(parents, child, candidate, n_vars):
                        continue
                    old_bic = self._bic_local(data, child, parents[child])
                    new_bic = self._bic_local(data, child, new_parents)
                    delta = new_bic - old_bic
                    if delta < best_delta:
                        best_delta = delta
                        best_add = (child, candidate)
            if best_add is not None:
                child, parent = best_add
                parents[child].append(parent)
                current_bic += best_delta
                improved = True
                n_forward_steps += 1
        improved = True
        n_backward_steps = 0
        while improved:
            improved = False
            best_delta = 0
            best_remove = None
            for child in range(n_vars):
                for parent in parents[child]:
                    new_parents = [p for p in parents[child] if p != parent]
                    old_bic = self._bic_local(data, child, parents[child])
                    new_bic = self._bic_local(data, child, new_parents)
                    delta = new_bic - old_bic
                    if delta < best_delta:
                        best_delta = delta
                        best_remove = (child, parent)
            if best_remove is not None:
                child, parent = best_remove
                parents[child].remove(parent)
                current_bic += best_delta
                improved = True
                n_backward_steps += 1
        edges = []
        adj_list = [[0] * n_vars for _ in range(n_vars)]
        n_edges = 0
        for child in range(n_vars):
            for parent in parents[child]:
                edges.append(CausalEdge(from_node=parent, to_node=child,
                                        edge_type="directed", confidence=0.7))
                adj_list[parent][child] = 1
                n_edges += 1
        graph = CausalGraph(
            n_nodes=n_vars,
            edges=edges,
            adjacency=adj_list,
            node_names=node_names,
            method="ges_algorithm"
        )
        return CausalDiscoveryResult(
            graph=graph,
            n_edges=n_edges,
            n_directed=n_edges,
            n_undirected=0,
            independence_tests=0,
            alpha=0.0,
            log_likelihood=-current_bic / 2,
            bic_score=float(current_bic),
            method="ges_algorithm"
        )

    def _would_create_cycle(self, parents: dict, child: int,
                            candidate: int, n_vars: int) -> bool:
        visited = set()
        stack = [candidate]
        while stack:
            node = stack.pop()
            if node == child:
                return True
            if node in visited:
                continue
            visited.add(node)
            for p in parents.get(node, []):
                stack.append(p)
        return False


class CausalDiscoveryEngine:
    """Unified causal discovery engine supporting multiple algorithms."""

    def __init__(self, algorithm: str = "pc", alpha: float = 0.05,
                 penalty: float = 1.0):
        self.algorithm = algorithm
        self.alpha = alpha
        self.penalty = penalty

    def fit(self, data: np.ndarray,
            node_names: Optional[List[str]] = None) -> CausalDiscoveryResult:
        if self.algorithm == "pc":
            algo = PCAlgorithm(alpha=self.alpha)
        elif self.algorithm == "ges":
            algo = GESAlgorithm(penalty=self.penalty)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        return algo.fit(data, node_names)

    def compare_algorithms(self, data: np.ndarray,
                           node_names: Optional[List[str]] = None) -> dict:
        pc_algo = PCAlgorithm(alpha=self.alpha)
        ges_algo = GESAlgorithm(penalty=self.penalty)
        pc_result = pc_algo.fit(data, node_names)
        ges_result = ges_algo.fit(data, node_names)
        pc_edges = {(e.from_node, e.to_node) for e in pc_result.graph.edges if e.edge_type == "directed"}
        ges_edges = {(e.from_node, e.to_node) for e in ges_result.graph.edges if e.edge_type == "directed"}
        common = pc_edges & ges_edges
        pc_only = pc_edges - ges_edges
        ges_only = ges_edges - pc_edges
        return {
            "pc_result": {
                "n_edges": pc_result.n_edges,
                "n_directed": pc_result.n_directed,
                "bic": pc_result.bic_score,
                "independence_tests": pc_result.independence_tests,
            },
            "ges_result": {
                "n_edges": ges_result.n_edges,
                "n_directed": ges_result.n_directed,
                "bic": ges_result.bic_score,
            },
            "agreement": {
                "common_edges": len(common),
                "pc_only_edges": len(pc_only),
                "ges_only_edges": len(ges_only),
                "agreement_rate": len(common) / max(len(pc_edges | ges_edges), 1),
            },
            "method": "causal_discovery_comparison"
        }