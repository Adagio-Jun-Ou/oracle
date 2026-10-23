from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class PersistenceResult:
    birth_times: list
    death_times: list
    persistence_scores: list
    n_features: int
    betti_numbers: dict
    max_persistence: float
    method: str


@dataclass
class TopologicalFeatureResult:
    bottleneck_distance: float
    wasserstein_distance: float
    persistence_entropy: float
    betti_curve: list
    landscape: list
    method: str


@dataclass
class ShapeDescriptor:
    connected_components: int
    loops: int
    voids: int
    total_persistence: float
    mean_persistence: float
    method: str


class TopologicalDataAnalysis:
    def __init__(self, max_dim: int = 1, seed: int = 42):
        self.max_dim = max_dim
        self.seed = seed

    @staticmethod
    def vietoris_rips_persistence(data: np.ndarray, max_dim: int = 1,
                                   max_edge_length: Optional[float] = None) -> PersistenceResult:
        from scipy.spatial.distance import pdist, squareform
        n = data.shape[0]
        if n < 3:
            return PersistenceResult(birth_times=[], death_times=[], persistence_scores=[],
                                      n_features=0, betti_numbers={}, max_persistence=0.0, method="vietoris_rips")
        dist_matrix = squareform(pdist(data))
        if max_edge_length is None:
            max_edge_length = float(np.percentile(dist_matrix, 90))
        filtration_values = np.unique(dist_matrix[dist_matrix > 0])
        filtration_values = filtration_values[filtration_values <= max_edge_length]
        filtration_values = np.sort(filtration_values)
        if len(filtration_values) > 200:
            filtration_values = np.percentile(dist_matrix[dist_matrix > 0],
                                                np.linspace(0, 100, 200))
            filtration_values = np.sort(np.unique(filtration_values))
        births_dim0 = []
        deaths_dim0 = []
        visited = set()
        component_map = {i: i for i in range(n)}
        for r in filtration_values:
            edges = np.argwhere((dist_matrix <= r) & (dist_matrix > 0))
            for e in edges:
                i, j = int(e[0]), int(e[1])
                if i >= j:
                    continue
                root_i = component_map[i]
                root_j = component_map[j]
                if root_i != root_j and (root_i, root_j) not in visited:
                    visited.add((root_i, root_j))
                    deaths_dim0.append(float(r))
                    for k in range(n):
                        if component_map[k] == root_j:
                            component_map[k] = root_i
        births_dim0 = [0.0] * len(deaths_dim0)
        if len(births_dim0) == 0:
            births_dim0 = [0.0]
            deaths_dim0 = [max_edge_length]
        elif len(deaths_dim0) < n:
            births_dim0.append(0.0)
            deaths_dim0.append(float(max_edge_length))
        birth_dim1 = []
        death_dim1 = []
        if max_dim >= 1 and n >= 4:
            visited_triangles = set()
            for r_idx, r in enumerate(filtration_values):
                edges_r = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        if dist_matrix[i, j] <= r:
                            edges_r.add((i, j))
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(j + 1, n):
                            tri = (i, j, k)
                            if tri not in visited_triangles:
                                if (i, j) in edges_r and (i, k) in edges_r and (j, k) in edges_r:
                                    visited_triangles.add(tri)
                                    tri_edges = [(i, j), (i, k), (j, k)]
                                    max_edge = max(dist_matrix[e[0], e[1]] for e in tri_edges)
                                    for e in tri_edges:
                                        if dist_matrix[e[0], e[1]] == max_edge:
                                            birth_dim1.append(float(dist_matrix[e[0], e[1]]))
                                            break
                                    death_dim1.append(float(r))
        birth_times = births_dim0 + birth_dim1
        death_times = deaths_dim0 + death_dim1
        death_times_capped = [min(d, float(max_edge_length)) for d in death_times]
        persistence_scores = [d - b for b, d in zip(birth_times, death_times_capped)]
        betti_numbers = {}
        for dim in range(max_dim + 1):
            if dim == 0:
                betti_numbers["dim_0"] = max(1, len(births_dim0) - len(deaths_dim0))
            elif dim == 1:
                betti_numbers["dim_1"] = max(0, len(birth_dim1) - len(death_dim1))
        return PersistenceResult(
            birth_times=birth_times,
            death_times=death_times_capped,
            persistence_scores=persistence_scores,
            n_features=len(birth_times),
            betti_numbers=betti_numbers,
            max_persistence=float(max(persistence_scores)) if persistence_scores else 0.0,
            method="vietoris_rips",
        )

    @staticmethod
    def compare_persistence(result1: PersistenceResult, result2: PersistenceResult,
                            p: int = 1) -> TopologicalFeatureResult:
        scores1 = result1.persistence_scores
        scores2 = result2.persistence_scores
        diag1 = list(zip(result1.birth_times, result1.death_times))
        diag2 = list(zip(result2.birth_times, result2.death_times))
        max_len = max(len(diag1), len(diag2))
        while len(diag1) < max_len:
            diag1.append((0, 0))
        while len(diag2) < max_len:
            diag2.append((0, 0))
        if p == float('inf'):
            bottleneck = float(max(abs(b1 - b2) + abs(d1 - d2)
                                   for (b1, d1), (b2, d2) in zip(diag1, diag2)))
        else:
            wasserstein = float(sum((abs(b1 - b2) ** p + abs(d1 - d2) ** p) ** (1 / p)
                                     for (b1, d1), (b2, d2) in zip(diag1, diag2)) ** (1 / p))
            bottleneck = wasserstein
        wasserstein_dist = bottleneck
        if p == 1:
            wasserstein_dist = float(sum(abs(b1 - b2) + abs(d1 - d2)
                                          for (b1, d1), (b2, d2) in zip(diag1, diag2)))
        scores_combined = scores1 + scores2
        total = sum(scores_combined) + 1e-10
        probs = [s / total for s in scores_combined if s > 0]
        entropy = -sum(p * np.log(max(p, 1e-10)) for p in probs) if probs else 0.0
        betti_curve = []
        max_filter = 10
        for t in np.linspace(0, max_filter, 20):
            b0 = sum(1 for b, d in diag1 if b <= t < d)
            betti_curve.append(b0)
        landscape = sorted(scores1, reverse=True)[:20]
        return TopologicalFeatureResult(
            bottleneck_distance=float(bottleneck),
            wasserstein_distance=float(wasserstein_dist),
            persistence_entropy=float(entropy),
            betti_curve=betti_curve,
            landscape=landscape,
            method="topological_comparison",
        )

    @staticmethod
    def shape_descriptor(data: np.ndarray) -> ShapeDescriptor:
        tda = TopologicalDataAnalysis(max_dim=1)
        result = tda.vietoris_rips_persistence(data, max_dim=1)
        total_persistence = float(sum(result.persistence_scores))
        mean_persistence = float(np.mean(result.persistence_scores)) if result.persistence_scores else 0.0
        cc = result.betti_numbers.get("dim_0", 1)
        loops = result.betti_numbers.get("dim_1", 0)
        return ShapeDescriptor(
            connected_components=max(1, cc),
            loops=loops,
            voids=0,
            total_persistence=total_persistence,
            mean_persistence=mean_persistence,
            method="shape_descriptor",
        )

    @staticmethod
    def time_series_persistence(series: np.ndarray, embedding_dim: int = 3,
                                 delay: int = 1) -> PersistenceResult:
        n = len(series)
        if n < embedding_dim * delay + 5:
            return PersistenceResult(birth_times=[], death_times=[], persistence_scores=[],
                                      n_features=0, betti_numbers={}, max_persistence=0.0, method="takens_embedding")
        m = n - (embedding_dim - 1) * delay
        embedded = np.zeros((m, embedding_dim))
        for i in range(m):
            for d in range(embedding_dim):
                embedded[i, d] = series[i + d * delay]
        return TopologicalDataAnalysis.vietoris_rips_persistence(embedded, max_dim=1)