"""CVT-MAP-Elites archive for quality-diversity evolution of prediction strategies.

Implements Centroidal Voronoi Tessellation MAP-Elites (CVT-MAP-Elites) with:
- Behavioral descriptors: (synthesis_score, cascade_richness) per domain
- Multi-objective fitness: Brier + calibration + confidence + cascade
- Archive statistics: QD score, coverage, uniformity, reliability
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class BehavioralDescriptor:
    """Multi-dimensional behavioral descriptor for niche placement."""
    def __init__(self, values: List[float], names: Optional[List[str]] = None):
        self.values = np.array(values, dtype=np.float64)
        self.names = names or [f"bd_{i}" for i in range(len(values))]

    def distance_to(self, other: BehavioralDescriptor) -> float:
        return float(np.linalg.norm(self.values - other.values))

    def to_cell_index(self, grid_sizes: List[int], ranges: List[Tuple[float, float]]) -> Tuple[int, ...]:
        indices = []
        for v, gs, (lo, hi) in zip(self.values, grid_sizes, ranges):
            norm = np.clip((v - lo) / max(hi - lo, 1e-10), 0.0, 1.0)
            indices.append(int(norm * (gs - 1)))
        return tuple(indices)

    def __repr__(self):
        return f"BD({dict(zip(self.names, [round(v, 4) for v in self.values]))})"


@dataclass
class FitnessScore:
    composite: float = 0.0
    brier: float = 0.0
    calibration: float = 0.0
    confidence: float = 0.0
    cascade: float = 0.0
    spread: float = 0.0
    coverage: float = 0.0
    direction: float = 0.0
    synthesis: float = 0.0

    def weighted_sum(self, weights: Optional[Dict[str, float]] = None) -> float:
        w = weights or {
            "synthesis": 0.30, "calibration": 0.20, "cascade": 0.15,
            "confidence": 0.12, "spread": 0.10, "coverage": 0.08,
            "direction": 0.05
        }
        return (
            w.get("synthesis", 0.3) * self.synthesis +
            w.get("calibration", 0.2) * self.calibration +
            w.get("cascade", 0.15) * self.cascade +
            w.get("confidence", 0.12) * self.confidence +
            w.get("spread", 0.10) * self.spread +
            w.get("coverage", 0.08) * self.coverage +
            w.get("direction", 0.05) * self.direction
        )


@dataclass
class Elite:
    genome_id: str
    behavioral_descriptor: BehavioralDescriptor
    fitness: FitnessScore
    cell_index: Tuple[int, ...]
    generation: int = 0
    birth_time: float = 0.0
    origin: str = "initial"
    parent_ids: List[str] = field(default_factory=list)
    evaluations: int = 1

    def update_fitness(self, new_fitness: FitnessScore, alpha: float = 0.3) -> None:
        blended = FitnessScore()
        for attr in ["composite", "brier", "calibration", "confidence", "cascade",
                      "spread", "coverage", "direction", "synthesis"]:
            old_val = getattr(self.fitness, attr)
            new_val = getattr(new_fitness, attr)
            setattr(blended, attr, alpha * new_val + (1 - alpha) * old_val)
        self.fitness = blended
        self.evaluations += 1


def _default_fitness_weights() -> Dict[str, float]:
    return {
        "synthesis": 0.30, "calibration": 0.20, "cascade": 0.15,
        "confidence": 0.12, "spread": 0.10, "coverage": 0.08,
        "direction": 0.05
    }


@dataclass
class QDMetrics:
    qd_score: float = 0.0
    coverage: float = 0.0
    uniformity: float = 0.0
    reliability: float = 0.0
    entropy: float = 0.0
    max_fitness: float = 0.0
    mean_fitness: float = 0.0
    num_elites: int = 0
    num_cells_occupied: int = 0
    total_cells: int = 0
    generation: int = 0


class CVTMAPElites:
    """CVT-MAP-Elites archive for quality-diversity evolution."""

    def __init__(
        self,
        bd_dimensions: int = 2,
        bd_ranges: Optional[List[Tuple[float, float]]] = None,
        grid_sizes: Optional[List[int]] = None,
        fitness_weights: Optional[Dict[str, float]] = None,
        n_centroids: int = 100,
        seed: int = 42,
    ):
        self.bd_dimensions = bd_dimensions
        self.bd_ranges = bd_ranges or [(0.0, 1.0), (0.0, 1.0)]
        self.grid_sizes = grid_sizes or [10, 10]
        self.fitness_weights = fitness_weights or _default_fitness_weights()
        self.n_centroids = n_centroids
        self.rng = np.random.RandomState(seed)

        self.archive: Dict[Tuple[int, ...], Elite] = {}
        self.centroids: Optional[np.ndarray] = None
        self._init_centroids()
        self._total_cells = int(np.prod(self.grid_sizes))
        self._generation = 0
        self._history: List[QDMetrics] = []
        self._max_history = 5000
        self._genome_archive: Dict[str, Any] = {}

    def _init_centroids(self) -> None:
        if self.n_centroids > 0 and self.bd_dimensions > 0:
            points = self.rng.rand(self.n_centroids, self.bd_dimensions)
            for d, (lo, hi) in enumerate(self.bd_ranges):
                points[:, d] = lo + points[:, d] * (hi - lo)
            self.centroids = points
        else:
            self.centroids = None

    def _find_cell(self, bd: BehavioralDescriptor) -> Tuple[int, ...]:
        if self.centroids is not None and len(self.centroids) > 0:
            dists = np.linalg.norm(self.centroids - bd.values[:self.bd_dimensions], axis=1)
            closest = int(np.argmin(dists))
            row = closest // self.grid_sizes[1] if len(self.grid_sizes) > 1 else 0
            col = closest % self.grid_sizes[1] if len(self.grid_sizes) > 1 else closest
            return (row, col) if len(self.grid_sizes) > 1 else (closest,)
        return bd.to_cell_index(self.grid_sizes, self.bd_ranges)

    def add_elite(self, genome_id: str, bd: BehavioralDescriptor,
                  fitness: FitnessScore, generation: int = 0,
                  origin: str = "initial", parent_ids: Optional[List[str]] = None,
                  genome_data: Optional[Any] = None) -> bool:
        cell_index = self._find_cell(bd)
        fitness_composite = fitness.weighted_sum(self.fitness_weights)
        fitness.composite = fitness_composite
        elite = Elite(
            genome_id=genome_id, behavioral_descriptor=bd,
            fitness=fitness, cell_index=cell_index,
            generation=generation, birth_time=time.time(),
            origin=origin, parent_ids=parent_ids or []
        )
        if cell_index in self.archive:
            existing = self.archive[cell_index]
            if fitness_composite > existing.fitness.composite:
                elite.evaluations = existing.evaluations + 1
                self.archive[cell_index] = elite
                if genome_data is not None:
                    self._genome_archive[genome_id] = genome_data
                return True
            else:
                existing.update_fitness(fitness)
                return False
        else:
            self.archive[cell_index] = elite
            if genome_data is not None:
                self._genome_archive[genome_id] = genome_data
            return True

    def get_elite(self, cell_index: Tuple[int, ...]) -> Optional[Elite]:
        return self.archive.get(cell_index)

    def get_random_elite(self) -> Optional[Elite]:
        if not self.archive:
            return None
        keys = list(self.archive.keys())
        idx = self.rng.randint(0, len(keys))
        return self.archive[keys[idx]]

    def get_best_elite(self) -> Optional[Elite]:
        if not self.archive:
            return None
        return max(self.archive.values(), key=lambda e: e.fitness.composite)

    def get_elites_near_bd(self, bd: BehavioralDescriptor, k: int = 5) -> List[Elite]:
        if not self.archive:
            return []
        elites = list(self.archive.values())
        elites.sort(key=lambda e: e.behavioral_descriptor.distance_to(bd))
        return elites[:k]

    def compute_qd_metrics(self) -> QDMetrics:
        if not self.archive:
            return QDMetrics()
        elites = list(self.archive.values())
        fitnesses = [e.fitness.composite for e in elites]
        max_fit = max(fitnesses)
        mean_fit = float(np.mean(fitnesses))
        active_count = len(elites)
        all_cells = int(np.prod(self.grid_sizes))
        coverage = active_count / max(all_cells, 1)
        qd = sum(max(0, f) for f in fitnesses)
        if active_count > 1:
            fitness_arr = np.array(fitnesses)
            mu = np.mean(fitness_arr)
            spread = float(np.std(fitness_arr))
            probs = np.maximum(fitness_arr - min(fitness_arr) + 1e-10, 1e-10)
            probs /= probs.sum()
            entropy = float(-np.sum(probs * np.log(probs)))
            mean_prob = probs.mean()
            uniformity = 1.0 / (1.0 + float(np.std(probs - mean_prob)))
        else:
            spread = 0.0
            entropy = 0.0
            uniformity = 1.0
        reliable = sum(1 for e in elites if e.evaluations >= 3) / max(active_count, 1)
        metrics = QDMetrics(
            qd_score=qd, coverage=coverage, uniformity=uniformity,
            reliability=reliable, entropy=entropy,
            max_fitness=max_fit, mean_fitness=mean_fit,
            num_elites=active_count, num_cells_occupied=active_count,
            total_cells=all_cells, generation=self._generation
        )
        self._history.append(metrics)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        return metrics

    def tournament_select(self, tournament_size: int = 3) -> Optional[Elite]:
        if not self.archive or len(self.archive) < 1:
            return None
        elites = list(self.archive.values())
        candidates = [elites[self.rng.randint(0, len(elites))] for _ in range(min(tournament_size, len(elites)))]
        return max(candidates, key=lambda e: e.fitness.composite)

    def novelty_select(self, k: int = 5) -> Optional[Elite]:
        if not self.archive or len(self.archive) < 2:
            return self.get_random_elite()
        elites = list(self.archive.values())
        bds = np.array([e.behavioral_descriptor.values for e in elites])
        distances = np.sum((bds[:, None] - bds[None, :]) ** 2, axis=-1)
        novelty_scores = np.mean(distances, axis=1)
        most_novel_idx = int(np.argmax(novelty_scores))
        return elites[most_novel_idx]

    def archive_stats(self) -> Dict[str, Any]:
        metrics = self.compute_qd_metrics()
        return {
            "total_cells": self._total_cells,
            "occupied_cells": metrics.num_cells_occupied,
            "coverage": round(metrics.coverage, 4),
            "qd_score": round(metrics.qd_score, 2),
            "max_fitness": round(metrics.max_fitness, 4),
            "mean_fitness": round(metrics.mean_fitness, 4),
            "entropy": round(metrics.entropy, 4),
            "uniformity": round(metrics.uniformity, 4),
            "reliability": round(metrics.reliability, 4),
            "generation": self._generation
        }

    def evolve_generation(self) -> None:
        self._generation += 1

    def export_archive(self) -> Dict[str, Any]:
        return {
            "generation": self._generation,
            "grid_sizes": self.grid_sizes,
            "bd_ranges": self.bd_ranges,
            "fitness_weights": self.fitness_weights,
            "elites": {
                str(k): {
                    "genome_id": e.genome_id,
                    "bd": e.behavioral_descriptor.values.tolist(),
                    "fitness": {
                        "composite": e.fitness.composite,
                        "brier": e.fitness.brier,
                        "calibration": e.fitness.calibration,
                        "confidence": e.fitness.confidence,
                        "cascade": e.fitness.cascade,
                        "synthesis": e.fitness.synthesis,
                        "spread": e.fitness.spread,
                        "coverage": e.fitness.coverage,
                        "direction": e.fitness.direction
                    },
                    "generation": e.generation,
                    "origin": e.origin,
                    "evaluations": e.evaluations
                }
                for k, e in self.archive.items()
            },
            "metrics_history": [
                {"generation": m.generation, "qd_score": m.qd_score,
                 "coverage": m.coverage, "max_fitness": m.max_fitness}
                for m in self._history[-100:]
            ]
        }

    def project_2d(self, method: str = "pca") -> Dict[str, Any]:
        if not self.archive or len(self.archive) < 2:
            return {"points": [], "method": method}
        elites = list(self.archive.values())
        bds = np.array([e.behavioral_descriptor.values for e in elites])
        fitnesses = np.array([e.fitness.composite for e in elites])
        if method == "pca" and bds.shape[1] > 2:
            from numpy.linalg import svd
            centered = bds - bds.mean(axis=0)
            if min(centered.shape) < 2:
                points_2d = bds[:, :2]
            else:
                U, S, Vt = svd(centered, full_matrices=False)
                points_2d = centered @ Vt[:2].T
        elif method == "random" and bds.shape[1] > 2:
            proj = self.rng.randn(bds.shape[1], 2)
            proj /= np.linalg.norm(proj, axis=1, keepdims=True)
            points_2d = bds @ proj
        else:
            points_2d = bds[:, :2]
        return {
            "points": points_2d.tolist(),
            "fitnesses": fitnesses.tolist(),
            "genome_ids": [e.genome_id for e in elites],
            "method": method
        }