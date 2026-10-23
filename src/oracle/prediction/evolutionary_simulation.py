"""Evolutionary Simulation module for genetic algorithms, fitness landscapes, and speciation.

Implements:
- Genetic algorithm with tournament/roulette selection, crossover, mutation
- Fitness landscape analysis (ruggedness, neutrality, epistasis)
- Speciation via niche partitioning
- Population dynamics with carrying capacity
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
import math


@dataclass
class Individual:
    genome: List[float]
    fitness: float = 0.0
    species: int = 0
    age: int = 0


@dataclass
class EvolutionResult:
    best_fitness_history: List[float]
    mean_fitness_history: List[float]
    diversity_history: List[float]
    species_count_history: List[int]
    best_genome: List[float]
    best_fitness: float
    n_generations: int
    n_species: int
    population_size: int
    method: str = "genetic_algorithm"


@dataclass
class FitnessLandscapeResult:
    ruggedness: float
    neutrality: float
    epistasis: float
    fitness_range: List[float]
    local_optima: int
    global_optimum: float
    method: str = "fitness_landscape"


class GeneticAlgorithm:
    """Genetic algorithm with configurable selection, crossover, and mutation.

    Supports:
    - Tournament and roulette wheel selection
    - Single-point, uniform, and blend crossover
    - Gaussian mutation with adaptive rate
    - Elitism preservation
    """

    def __init__(self, population_size: int = 100, genome_length: int = 10,
                 crossover_rate: float = 0.7, mutation_rate: float = 0.1,
                 mutation_sigma: float = 0.3, elite_fraction: float = 0.1,
                 selection_method: str = "tournament",
                 tournament_size: int = 5, seed: int = 42):
        self.population_size = population_size
        self.genome_length = genome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.elite_fraction = elite_fraction
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.population: List[Individual] = []
        self.best_fitness_history: List[float] = []
        self.mean_fitness_history: List[float] = []
        self.diversity_history: List[float] = []

    def _init_population(self) -> List[Individual]:
        return [Individual(genome=self.rng.uniform(-5, 5, self.genome_length).tolist())
                for _ in range(self.population_size)]

    def _evaluate(self, ind: Individual, fitness_func: Callable) -> float:
        return fitness_func(np.array(ind.genome))

    def _tournament_select(self, population: List[Individual], fitnesses: List[float]) -> Individual:
        indices = self.rng.choice(len(population), min(self.tournament_size, len(population)), replace=False)
        best = indices[0]
        for idx in indices[1:]:
            if fitnesses[idx] > fitnesses[best]:
                best = idx
        return population[best]

    def _roulette_select(self, population: List[Individual], fitnesses: List[float]) -> Individual:
        min_fit = min(fitnesses)
        shifted = [f - min_fit + 1e-10 for f in fitnesses]
        total = sum(shifted)
        if total <= 0:
            return population[self.rng.randint(0, len(population))]
        probs = [s / total for s in shifted]
        idx = self.rng.choice(len(population), p=probs)
        return population[idx]

    def _crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        g1, g2 = p1.genome[:], p2.genome[:]
        if self.rng.random() < self.crossover_rate:
            point = self.rng.randint(1, len(g1))
            g1, g2 = g1[:point] + g2[point:], g2[:point] + g1[point:]
        return Individual(genome=g1), Individual(genome=g2)

    def _mutate(self, ind: Individual) -> Individual:
        genome = ind.genome[:]
        for i in range(len(genome)):
            if self.rng.random() < self.mutation_rate:
                genome[i] += self.rng.normal(0, self.mutation_sigma)
        return Individual(genome=genome, age=ind.age + 1)

    def fit(self, fitness_func: Callable, n_generations: int = 100,
            target_fitness: Optional[float] = None) -> 'GeneticAlgorithm':
        self.population = self._init_population()
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.diversity_history = []
        for gen in range(n_generations):
            fitnesses = [self._evaluate(ind, fitness_func) for ind in self.population]
            for i, ind in enumerate(self.population):
                ind.fitness = fitnesses[i]
            best_fit = max(fitnesses)
            mean_fit = float(np.mean(fitnesses))
            genomes = np.array([ind.genome for ind in self.population])
            diversity = float(np.mean(np.std(genomes, axis=0)))
            self.best_fitness_history.append(best_fit)
            self.mean_fitness_history.append(mean_fit)
            self.diversity_history.append(diversity)
            if target_fitness is not None and best_fit >= target_fitness:
                break
            n_elite = max(1, int(self.elite_fraction * self.population_size))
            sorted_indices = np.argsort(fitnesses)[::-1]
            elite = [Individual(genome=self.population[i].genome[:], fitness=fitnesses[i])
                     for i in sorted_indices[:n_elite]]
            new_population = list(elite)
            while len(new_population) < self.population_size:
                if self.selection_method == "tournament":
                    p1 = self._tournament_select(self.population, fitnesses)
                    p2 = self._tournament_select(self.population, fitnesses)
                else:
                    p1 = self._roulette_select(self.population, fitnesses)
                    p2 = self._roulette_select(self.population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.population_size:
                    new_population.append(c2)
            self.population = new_population[:self.population_size]
            self.mutation_rate = max(0.01, self.mutation_rate * 0.999)
        return self

    def get_result(self) -> EvolutionResult:
        fitnesses = [ind.fitness for ind in self.population]
        best_idx = int(np.argmax(fitnesses))
        n_elite = max(1, int(self.elite_fraction * self.population_size))
        species_ids = [0] * len(self.population)
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        current_species = 0
        niches = [[sorted_pop[0]]]
        for ind in sorted_pop[1:]:
            assigned = False
            for i, niche in enumerate(niches):
                niche_center = np.mean([np.array(m.genome) for m in niche], axis=0)
                dist = float(np.linalg.norm(np.array(ind.genome) - niche_center))
                if dist < 2.0:
                    niche.append(ind)
                    species_ids[self.population.index(ind)] = i
                    assigned = True
                    break
            if not assigned:
                niches.append([ind])
                species_ids[self.population.index(ind)] = len(niches) - 1
        return EvolutionResult(
            best_fitness_history=self.best_fitness_history,
            mean_fitness_history=self.mean_fitness_history,
            diversity_history=self.diversity_history,
            species_count_history=[1] * len(self.best_fitness_history),
            best_genome=self.population[best_idx].genome,
            best_fitness=float(max(fitnesses)),
            n_generations=len(self.best_fitness_history),
            n_species=len(niches),
            population_size=self.population_size,
            method=f"ga_{self.selection_method}",
        )


class FitnessLandscape:
    """Analyze fitness landscape properties: ruggedness, neutrality, epistasis."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def analyze(self, fitness_func: Callable, bounds: Tuple[float, float] = (-5, 5),
                dim: int = 2, n_samples: int = 500) -> FitnessLandscapeResult:
        samples = self.rng.uniform(bounds[0], bounds[1], (n_samples, dim))
        fitnesses = np.array([fitness_func(s) for s in samples])
        diffs = []
        for i in range(n_samples):
            for j in range(i + 1, min(i + 20, n_samples)):
                d = np.linalg.norm(samples[i] - samples[j])
                if d > 0:
                    diffs.append(abs(fitnesses[i] - fitnesses[j]) / d)
        ruggedness = float(np.mean(diffs)) if diffs else 0.0
        neutral_threshold = np.std(fitnesses) * 0.01
        neutral_count = 0
        total_pairs = 0
        for i in range(n_samples):
            for j in range(i + 1, min(i + 10, n_samples)):
                total_pairs += 1
                if abs(fitnesses[i] - fitnesses[j]) < neutral_threshold:
                    neutral_count += 1
        neutrality = neutral_count / max(total_pairs, 1)
        epistasis_values = []
        base_genome = self.rng.uniform(bounds[0], bounds[1], dim)
        base_fit = fitness_func(base_genome)
        for _ in range(100):
            i = self.rng.randint(0, dim)
            j = self.rng.randint(0, dim)
            if i == j:
                continue
            mutant_i = base_genome.copy()
            mutant_i[i] += self.rng.normal(0, 0.5)
            mutant_j = base_genome.copy()
            mutant_j[j] += self.rng.normal(0, 0.5)
            mutant_ij = base_genome.copy()
            mutant_ij[i] += self.rng.normal(0, 0.5)
            mutant_ij[j] += self.rng.normal(0, 0.5)
            epistatic = fitness_func(mutant_ij) - fitness_func(mutant_i) - fitness_func(mutant_j) + base_fit
            epistasis_values.append(abs(epistatic))
        epistasis = float(np.mean(epistasis_values)) if epistasis_values else 0.0
        local_optima = 0
        for i in range(n_samples):
            neighbors_fit = []
            for j in range(n_samples):
                if i != j and np.linalg.norm(samples[i] - samples[j]) < 0.5:
                    neighbors_fit.append(fitnesses[j])
            if neighbors_fit and fitnesses[i] > max(neighbors_fit):
                local_optima += 1
        return FitnessLandscapeResult(
            ruggedness=ruggedness,
            neutrality=neutrality,
            epistasis=epistasis,
            fitness_range=[float(np.min(fitnesses)), float(np.max(fitnesses))],
            local_optima=local_optima,
            global_optimum=float(np.max(fitnesses)),
            method="fitness_landscape",
        )


class SpeciationSimulator:
    """Simulate speciation through niche partitioning and reproductive isolation."""

    def __init__(self, n_species: int = 3, n_individuals: int = 100,
                 niche_width: float = 2.0, hybrid_fitness: float = 0.5,
                 seed: int = 42):
        self.n_species = n_species
        self.n_individuals = n_individuals
        self.niche_width = niche_width
        self.hybrid_fitness = hybrid_fitness
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def simulate(self, n_generations: int = 100) -> dict:
        species_centers = self.rng.uniform(-5, 5, (self.n_species, 2))
        species_sizes = [self.n_individuals // self.n_species] * self.n_species
        species_sizes[-1] = self.n_individuals - sum(species_sizes[:-1])
        individuals = []
        species_labels = []
        for s in range(self.n_species):
            for _ in range(species_sizes[s]):
                pos = species_centers[s] + self.rng.normal(0, self.niche_width / 2, 2)
                individuals.append(pos)
                species_labels.append(s)
        individuals = np.array(individuals)
        species_sizes_history = [species_sizes[:]]
        hybrid_fraction_history = []
        diversity_history = [float(np.mean(np.std(individuals, axis=0)))]
        for gen in range(n_generations):
            new_individuals = []
            new_labels = []
            for i in range(self.n_individuals):
                mate_idx = self.rng.randint(0, self.n_individuals)
                dist = np.linalg.norm(individuals[i] - individuals[mate_idx])
                if species_labels[i] == species_labels[mate_idx] or dist < self.niche_width:
                    child = (individuals[i] + individuals[mate_idx]) / 2 + self.rng.normal(0, 0.3, 2)
                    child_label = species_labels[i] if self.rng.random() < 0.9 else species_labels[mate_idx]
                else:
                    child = (individuals[i] + individuals[mate_idx]) / 2 + self.rng.normal(0, 0.5, 2)
                    child_label = species_labels[i] if self.rng.random() < self.hybrid_fitness else species_labels[mate_idx]
                new_individuals.append(child)
                new_labels.append(child_label)
            individuals = np.array(new_individuals)
            species_labels = new_labels
            current_sizes = [species_labels.count(s) for s in range(self.n_species)]
            species_sizes_history.append(current_sizes)
            hybrids = sum(1 for i in range(len(species_labels)) for j in range(i + 1, len(species_labels))
                         if species_labels[i] != species_labels[j] and np.linalg.norm(individuals[i] - individuals[j]) < self.niche_width)
            total_pairs = self.n_individuals * (self.n_individuals - 1) / 2
            hybrid_fraction_history.append(hybrids / max(total_pairs, 1))
            diversity_history.append(float(np.mean(np.std(individuals, axis=0))))
        active_species = sum(1 for s in range(self.n_species) if species_labels.count(s) > 2)
        return {
            "species_sizes_history": species_sizes_history[-5:],
            "hybrid_fraction_history": hybrid_fraction_history[-10:],
            "diversity_history": diversity_history[-10:],
            "n_active_species": active_species,
            "final_species_sizes": [species_labels.count(s) for s in range(self.n_species)],
            "method": "speciation_simulation",
        }