"""Learning Simulation module for population learning, cultural evolution, and skill transfer.

Implements:
- Population learning with knowledge diffusion on networks
- Cultural evolution with trait transmission and mutation
- Skill cascade models for technology adoption
- Social learning strategies (conformist, success-biased, random)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable


@dataclass
class LearningResult:
    knowledge_history: List[float]
    skill_distribution: List[float]
    learning_rate: float
    convergence_generation: int
    n_learners: int
    strategy: str
    method: str = "population_learning"


@dataclass
class CulturalEvolutionResult:
    trait_frequencies: Dict[str, List[float]]
    diversity_history: List[float]
    n_trait_types: int
    dominant_trait: str
    generations: int
    method: str = "cultural_evolution"


@dataclass
class SkillCascadeResult:
    adoption_curve: List[float]
    skill_levels: Dict[str, List[float]]
    cascade_threshold: float
    peak_adoption: float
    time_to_peak: int
    method: str = "skill_cascade"


class PopulationLearning:
    """Simulate knowledge diffusion through a population on a network.

    Agents learn from neighbors via social learning strategies:
    - Conformist: copy majority
    - Success-biased: copy most successful
    - Random: copy random neighbor
    """

    def __init__(self, n_learners: int = 100, network_type: str = "small_world",
                 initial_knowledge_mean: float = 0.3, initial_knowledge_std: float = 0.15,
                 learning_rate: float = 0.1, seed: int = 42):
        self.n_learners = n_learners
        self.network_type = network_type
        self.initial_knowledge_mean = initial_knowledge_mean
        self.initial_knowledge_std = initial_knowledge_std
        self.learning_rate = learning_rate
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.knowledge: Optional[np.ndarray] = None
        self.adjacency: Optional[np.ndarray] = None
        self._fitted = False

    def _build_network(self) -> np.ndarray:
        n = self.n_learners
        adj = np.zeros((n, n))
        if self.network_type == "random":
            p = 0.1
            for i in range(n):
                for j in range(i + 1, n):
                    if self.rng.random() < p:
                        adj[i, j] = 1
                        adj[j, i] = 1
        elif self.network_type == "small_world":
            k = max(2, int(0.1 * n))
            for i in range(n):
                for j in range(1, k // 2 + 1):
                    adj[i, (i + j) % n] = 1
                    adj[(i + j) % n, i] = 1
            for i in range(n):
                for j in range(n):
                    if adj[i, j] == 1 and self.rng.random() < 0.1:
                        adj[i, j] = 0
                        adj[j, i] = 0
                        new_j = self.rng.randint(0, n)
                        while new_j == i:
                            new_j = self.rng.randint(0, n)
                        adj[i, new_j] = 1
                        adj[new_j, i] = 1
        elif self.network_type == "scale_free":
            m = max(2, int(0.05 * n))
            degrees = np.zeros(n)
            for i in range(m + 1):
                for j in range(i + 1, m + 1):
                    adj[i, j] = 1
                    adj[j, i] = 1
                    degrees[i] += 1
                    degrees[j] += 1
            for new_node in range(m + 1, n):
                probs = degrees[:new_node] / max(degrees[:new_node].sum(), 1)
                targets = self.rng.choice(new_node, size=min(m, new_node), replace=False, p=probs)
                for t in targets:
                    adj[new_node, t] = 1
                    adj[t, new_node] = 1
                    degrees[new_node] += 1
                    degrees[t] += 1
        else:
            p = 0.15
            for i in range(n):
                for j in range(i + 1, n):
                    if self.rng.random() < p:
                        adj[i, j] = 1
                        adj[j, i] = 1
        for i in range(n):
            if adj[i].sum() == 0:
                j = self.rng.randint(0, n)
                while j == i:
                    j = self.rng.randint(0, n)
                adj[i, j] = 1
                adj[j, i] = 1
        return adj

    def simulate(self, n_generations: int = 50, strategy: str = "success_biased") -> LearningResult:
        self.knowledge = np.clip(
            self.rng.normal(self.initial_knowledge_mean, self.initial_knowledge_std, self.n_learners),
            0, 1
        )
        self.adjacency = self._build_network()
        knowledge_history = [float(np.mean(self.knowledge))]
        convergence_gen = n_generations
        for gen in range(n_generations):
            new_knowledge = self.knowledge.copy()
            for i in range(self.n_learners):
                neighbors = np.where(self.adjacency[i] > 0)[0]
                if len(neighbors) == 0:
                    continue
                if strategy == "conformist":
                    neighbor_knowledge = self.knowledge[neighbors]
                    median_knowledge = float(np.median(neighbor_knowledge))
                    new_knowledge[i] += self.learning_rate * (median_knowledge - self.knowledge[i])
                elif strategy == "success_biased":
                    best_neighbor = neighbors[np.argmax(self.knowledge[neighbors])]
                    new_knowledge[i] += self.learning_rate * (self.knowledge[best_neighbor] - self.knowledge[i])
                elif strategy == "random":
                    random_neighbor = self.rng.choice(neighbors)
                    new_knowledge[i] += self.learning_rate * (self.knowledge[random_neighbor] - self.knowledge[i])
                else:
                    random_neighbor = self.rng.choice(neighbors)
                    new_knowledge[i] += self.learning_rate * (self.knowledge[random_neighbor] - self.knowledge[i])
                new_knowledge[i] += self.rng.normal(0, 0.01)
                new_knowledge[i] = np.clip(new_knowledge[i], 0, 1)
            self.knowledge = new_knowledge
            knowledge_history.append(float(np.mean(self.knowledge)))
            if gen > 0 and abs(knowledge_history[-1] - knowledge_history[-2]) < 1e-5:
                convergence_gen = gen
                break
        return LearningResult(
            knowledge_history=knowledge_history,
            skill_distribution=self.knowledge.tolist(),
            learning_rate=self.learning_rate,
            convergence_generation=convergence_gen,
            n_learners=self.n_learners,
            strategy=strategy,
        )


class CulturalEvolution:
    """Simulate cultural trait evolution through transmission, mutation, and selection.

    Models trait frequency changes over generations with:
    - Vertical transmission (parent to child)
    - Horizontal transmission (peer to peer)
    - Mutation (innovation)
    - Selection pressure
    """

    def __init__(self, n_traits: int = 5, population_size: int = 200,
                 mutation_rate: float = 0.01, selection_strength: float = 0.1,
                 seed: int = 42):
        self.n_traits = n_traits
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.selection_strength = selection_strength
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def simulate(self, n_generations: int = 100) -> CulturalEvolutionResult:
        trait_labels = [f"trait_{i}" for i in range(self.n_traits)]
        frequencies = {label: [float(self.rng.uniform(0.05, 0.3))] for label in trait_labels}
        diversity_history = []
        for gen in range(n_generations):
            current_freqs = np.array([frequencies[label][-1] for label in trait_labels])
            current_freqs = current_freqs / max(current_freqs.sum(), 1e-10)
            fitness = 1.0 + self.selection_strength * (current_freqs - current_freqs.mean())
            selected_freqs = current_freqs * fitness
            selected_freqs = selected_freqs / max(selected_freqs.sum(), 1e-10)
            for i, label in enumerate(trait_labels):
                mutation_effect = self.rng.normal(0, self.mutation_rate)
                new_freq = selected_freqs[i] + mutation_effect
                new_freq = max(0.001, min(new_freq, 0.999))
                frequencies[label].append(float(new_freq))
            total = sum(frequencies[label][-1] for label in trait_labels)
            for label in trait_labels:
                frequencies[label][-1] /= max(total, 1e-10)
            current_diversity = 0.0
            for label in trait_labels:
                f = frequencies[label][-1]
                if f > 0:
                    current_diversity -= f * np.log(f)
            diversity_history.append(float(current_diversity))
        dominant_idx = np.argmax([frequencies[label][-1] for label in trait_labels])
        return CulturalEvolutionResult(
            trait_frequencies=frequencies,
            diversity_history=diversity_history,
            n_trait_types=self.n_traits,
            dominant_trait=trait_labels[dominant_idx],
            generations=n_generations,
        )


class SkillCascade:
    """Model skill cascade / technology adoption through a population.

    Implements Bass diffusion model with network effects:
    - Innovators adopt spontaneously
    - Imitators adopt after seeing others
    - Network effects accelerate adoption
    """

    def __init__(self, population_size: int = 1000,
                 innovation_rate: float = 0.01,
                 imitation_rate: float = 0.3,
                 network_effect: float = 0.1,
                 n_skills: int = 3, seed: int = 42):
        self.population_size = population_size
        self.innovation_rate = innovation_rate
        self.imitation_rate = imitation_rate
        self.network_effect = network_effect
        self.n_skills = n_skills
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def simulate(self, n_periods: int = 50) -> SkillCascadeResult:
        skill_names = [f"skill_{i}" for i in range(self.n_skills)]
        adoption = np.zeros((self.n_skills, self.population_size))
        skill_levels = {name: [float(adoption[i].mean())] for i, name in enumerate(skill_names)}
        adoption_curve = [0.0]
        cascade_threshold = self.innovation_rate + self.imitation_rate * 0.1
        for t in range(n_periods):
            for s in range(self.n_skills):
                current_adopters = adoption[s].sum()
                adoption_fraction = current_adopters / self.population_size
                for i in range(self.population_size):
                    if adoption[s, i] == 0:
                        p_adopt = self.innovation_rate + self.imitation_rate * adoption_fraction
                        p_adopt += self.network_effect * adoption_fraction ** 2
                        p_adopt = min(p_adopt, 0.95)
                        if self.rng.random() < p_adopt:
                            adoption[s, i] = 1
                            for s2 in range(self.n_skills):
                                if s2 != s and adoption[s2, i] == 0:
                                    if self.rng.random() < 0.3:
                                        adoption[s2, i] = 1
            total_adopted = sum(1 for i in range(self.population_size) if any(adoption[s, i] > 0 for s in range(self.n_skills)))
            adoption_curve.append(float(total_adopted / self.population_size))
            for s, name in enumerate(skill_names):
                skill_levels[name].append(float(adoption[s].mean()))
        peak_idx = int(np.argmax(adoption_curve))
        return SkillCascadeResult(
            adoption_curve=adoption_curve,
            skill_levels=skill_levels,
            cascade_threshold=float(cascade_threshold),
            peak_adoption=float(max(adoption_curve)),
            time_to_peak=peak_idx,
        )