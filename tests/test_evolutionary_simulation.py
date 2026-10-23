"""Tests for evolutionary simulation module."""
import pytest
import numpy as np
from oracle.prediction.evolutionary_simulation import (
    GeneticAlgorithm, FitnessLandscape, SpeciationSimulator,
    EvolutionResult, FitnessLandscapeResult
)


def sphere(x):
    return -float(np.sum(x**2))


def rastrigin(x):
    return -(10 * len(x) + float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x))))


class TestGeneticAlgorithm:
    def test_init(self):
        ga = GeneticAlgorithm(population_size=20, genome_length=5, seed=42)
        assert ga.population_size == 20
        assert ga.genome_length == 5
        assert ga.mutation_rate == 0.1

    def test_fit_sphere(self):
        ga = GeneticAlgorithm(population_size=30, genome_length=5, mutation_rate=0.15,
                              selection_method="tournament", seed=42)
        ga.fit(sphere, n_generations=50)
        result = ga.get_result()
        assert isinstance(result, EvolutionResult)
        assert result.best_fitness > -10.0
        assert result.n_generations > 0
        assert result.population_size == 30
        assert len(result.best_genome) == 5
        assert len(result.best_fitness_history) > 0

    def test_fit_roulette(self):
        ga = GeneticAlgorithm(population_size=30, genome_length=3,
                              selection_method="roulette", seed=42)
        ga.fit(sphere, n_generations=30)
        result = ga.get_result()
        assert result.best_fitness > -5.0
        assert "roulette" in result.method

    def test_convergence(self):
        ga = GeneticAlgorithm(population_size=50, genome_length=3, seed=42)
        ga.fit(sphere, n_generations=100, target_fitness=-0.1)
        result = ga.get_result()
        assert result.best_fitness > -0.5

    def test_diversity_decreases(self):
        ga = GeneticAlgorithm(population_size=50, genome_length=5, seed=42)
        ga.fit(sphere, n_generations=50)
        result = ga.get_result()
        if len(result.diversity_history) > 5:
            assert result.diversity_history[-1] <= result.diversity_history[0] + 0.5


class TestFitnessLandscape:
    def test_analyze_sphere(self):
        fl = FitnessLandscape(seed=42)
        result = fl.analyze(sphere, dim=2, n_samples=200)
        assert isinstance(result, FitnessLandscapeResult)
        assert result.ruggedness > 0
        assert 0 <= result.neutrality <= 1
        assert result.epistasis >= 0
        assert len(result.fitness_range) == 2
        assert result.global_optimum > result.fitness_range[0]

    def test_analyze_rastrigin(self):
        fl = FitnessLandscape(seed=42)
        result = fl.analyze(rastrigin, dim=2, n_samples=200)
        assert result.local_optima > 0
        assert result.ruggedness > 0

    def test_method_field(self):
        fl = FitnessLandscape(seed=42)
        result = fl.analyze(sphere, dim=2, n_samples=100)
        assert result.method == "fitness_landscape"


class TestSpeciationSimulator:
    def test_simulate(self):
        sim = SpeciationSimulator(n_species=3, n_individuals=50, seed=42)
        result = sim.simulate(n_generations=20)
        assert "species_sizes_history" in result
        assert "diversity_history" in result
        assert "n_active_species" in result
        assert result["n_active_species"] >= 1
        assert len(result["final_species_sizes"]) == 3

    def test_method(self):
        sim = SpeciationSimulator(seed=42)
        result = sim.simulate(n_generations=10)
        assert result["method"] == "speciation_simulation"

    def test_diversity_nonempty(self):
        sim = SpeciationSimulator(n_species=2, n_individuals=30, seed=42)
        result = sim.simulate(n_generations=20)
        assert len(result["diversity_history"]) > 0