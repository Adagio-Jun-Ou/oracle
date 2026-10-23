"""Tests for learning simulation module."""
import pytest
import numpy as np
from oracle.prediction.learning_simulation import (
    PopulationLearning, CulturalEvolution, SkillCascade,
    LearningResult, CulturalEvolutionResult, SkillCascadeResult
)


class TestPopulationLearning:
    def test_init(self):
        pl = PopulationLearning(n_learners=50, seed=42)
        assert pl.n_learners == 50
        assert pl.learning_rate == 0.1

    def test_simulate_success_biased(self):
        pl = PopulationLearning(n_learners=50, network_type="small_world", seed=42)
        result = pl.simulate(n_generations=30, strategy="success_biased")
        assert isinstance(result, LearningResult)
        assert len(result.knowledge_history) > 0
        assert result.n_learners == 50
        assert result.strategy == "success_biased"
        assert result.knowledge_history[-1] > result.knowledge_history[0]

    def test_simulate_conformist(self):
        pl = PopulationLearning(n_learners=30, seed=42)
        result = pl.simulate(n_generations=20, strategy="conformist")
        assert result.strategy == "conformist"
        assert len(result.skill_distribution) == 30

    def test_simulate_random(self):
        pl = PopulationLearning(n_learners=30, network_type="random", seed=42)
        result = pl.simulate(n_generations=20, strategy="random")
        assert result.strategy == "random"
        assert result.n_learners == 30

    def test_network_types(self):
        for nt in ["small_world", "scale_free", "random"]:
            pl = PopulationLearning(n_learners=30, network_type=nt, seed=42)
            result = pl.simulate(n_generations=10, strategy="success_biased")
            assert len(result.knowledge_history) > 0

    def test_method_field(self):
        pl = PopulationLearning(seed=42)
        result = pl.simulate(n_generations=5)
        assert result.method == "population_learning"


class TestCulturalEvolution:
    def test_init(self):
        ce = CulturalEvolution(n_traits=3, seed=42)
        assert ce.n_traits == 3

    def test_simulate(self):
        ce = CulturalEvolution(n_traits=5, seed=42)
        result = ce.simulate(n_generations=50)
        assert isinstance(result, CulturalEvolutionResult)
        assert result.n_trait_types == 5
        assert result.dominant_trait.startswith("trait_")
        assert len(result.diversity_history) == 50
        for key in result.trait_frequencies:
            assert len(result.trait_frequencies[key]) == 51

    def test_method_field(self):
        ce = CulturalEvolution(seed=42)
        result = ce.simulate(n_generations=10)
        assert result.method == "cultural_evolution"

    def test_convergence(self):
        ce = CulturalEvolution(n_traits=3, seed=42)
        result = ce.simulate(n_generations=200)
        assert result.generations == 200


class TestSkillCascade:
    def test_init(self):
        sc = SkillCascade(n_skills=2, seed=42)
        assert sc.n_skills == 2

    def test_simulate(self):
        sc = SkillCascade(n_skills=3, seed=42)
        result = sc.simulate(n_periods=30)
        assert isinstance(result, SkillCascadeResult)
        assert result.peak_adoption > 0
        assert result.time_to_peak >= 0
        assert result.cascade_threshold > 0
        assert len(result.adoption_curve) == 31

    def test_method_field(self):
        sc = SkillCascade(seed=42)
        result = sc.simulate(n_periods=10)
        assert result.method == "skill_cascade"

    def test_adoption_increases(self):
        sc = SkillCascade(innovation_rate=0.02, imitation_rate=0.4, seed=42)
        result = sc.simulate(n_periods=50)
        assert result.adoption_curve[-1] > result.adoption_curve[0]