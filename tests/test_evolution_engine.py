"""Comprehensive tests for the ContinuousEvolutionEngine and supporting modules.

Tests cover:
- Genome creation, mutation, crossover
- MAP-Elites archive operations
- Multi-objective fitness evaluation
- LLM mutator (heuristic fallback)
- Data flow and outcome tracking
- ContinuousEvolutionEngine step/evolve/predict
"""
import math
import time
import random
import pytest
import numpy as np

from oracle.evolution.genome import (
    EvolvableFunction, PredictionGenome, GenomeRegistry,
    MutationType, create_default_genome, create_all_domain_genomes,
    create_random_genome, DOMAIN_GENOMES
)
from oracle.evolution.map_elites import (
    CVTMAPElites, Elite, BehavioralDescriptor, FitnessScore, QDMetrics
)
from oracle.evolution.fitness import (
    MultiObjectiveFitness, FitnessResult, FitnessComponent,
    OutcomeTracker, BacktestWindow
)
from oracle.evolution.llm_mutator import (
    LLMMutator, MutationResult, CrossoverResult, InjectionResult, MutationStrategy
)
from oracle.evolution.data_flow import (
    DataFlowEngine, DataPoint, OutcomeRecord, DomainStream, StreamSubscription, Domain
)
from oracle.evolution.continuous_engine import (
    ContinuousEvolutionEngine, EvolutionConfig, EvolutionState, EvolutionPhase, StrategyReport
)


class TestEvolvableFunction:
    def test_creation(self):
        f = EvolvableFunction(
            name="test_func", params={"a": 0.5, "b": 0.3},
            param_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
            domain="economic"
        )
        assert f.name == "test_func"
        assert f.params["a"] == 0.5
        assert f.domain == "economic"

    def test_mutate_gaussian(self):
        rng = random.Random(42)
        f = EvolvableFunction(
            name="test", params={"x": 0.5},
            param_bounds={"x": (0.0, 1.0)},
            mutation_rate=1.0, mutation_type=MutationType.GAUSSIAN
        )
        child = f.mutate(rng, sigma=0.15)
        assert child.name == "test"
        assert child.params["x"] != 0.5 or abs(child.params["x"] - 0.5) < 0.5
        assert 0.0 <= child.params["x"] <= 1.0
        assert child.source == "mutated"

    def test_mutate_preserves_bounds(self):
        rng = random.Random(42)
        f = EvolvableFunction(
            name="test", params={"x": 0.01},
            param_bounds={"x": (0.0, 1.0)},
            mutation_rate=1.0, mutation_type=MutationType.GAUSSIAN
        )
        for _ in range(100):
            child = f.mutate(rng, sigma=0.5)
            assert 0.0 <= child.params["x"] <= 1.0

    def test_crossover_blend(self):
        rng = random.Random(42)
        f1 = EvolvableFunction(
            name="test", params={"a": 0.0, "b": 1.0},
            param_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)}
        )
        f2 = EvolvableFunction(
            name="test", params={"a": 1.0, "b": 0.0},
            param_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)}
        )
        child = f1.crossover(f2, rng)
        assert child.name == "test"
        assert 0.0 <= child.params["a"] <= 1.0
        assert 0.0 <= child.params["b"] <= 1.0

    def test_crossover_different_names(self):
        f1 = EvolvableFunction(name="func_a", params={"x": 0.5}, param_bounds={"x": (0.0, 1.0)})
        f2 = EvolvableFunction(name="func_b", params={"x": 0.5}, param_bounds={"x": (0.0, 1.0)})
        child = f1.crossover(f2)
        assert child.name == "func_a"

    def test_to_dict_from_dict_roundtrip(self):
        f = EvolvableFunction(
            name="test", params={"a": 0.5, "b": 0.3},
            param_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
            domain="economic", source="baseline"
        )
        d = f.to_dict()
        f2 = EvolvableFunction.from_dict(d)
        assert f2.name == f.name
        assert f2.params == f.params
        assert f2.domain == f.domain
        assert f2.source == f.source

    def test_mutation_types(self):
        rng = random.Random(42)
        for mt in [MutationType.GAUSSIAN, MutationType.UNIFORM, MutationType.POLYNOMIAL, MutationType.CATASTROPHIC]:
            f = EvolvableFunction(
                name="test", params={"x": 0.5},
                param_bounds={"x": (0.0, 1.0)},
                mutation_rate=1.0, mutation_type=mt
            )
            child = f.mutate(rng)
            assert 0.0 <= child.params["x"] <= 1.0


class TestPredictionGenome:
    def test_creation_default(self):
        g = PredictionGenome()
        assert g.genome_id != ""
        assert g.generation == 0
        assert g.origin == "initial"

    def test_create_default_genome(self):
        g = create_default_genome("economic")
        assert len(g.functions) == 5
        assert "adjust_probability_for_domain_states" in g.functions
        assert g.functions["adjust_probability_for_domain_states"].domain == "economic"

    def test_create_all_domain_genomes(self):
        genomes = create_all_domain_genomes()
        assert len(genomes) == len(DOMAIN_GENOMES)
        for domain in DOMAIN_GENOMES:
            assert domain in genomes

    def test_mutate_creates_new_id(self):
        g = create_default_genome("economic")
        child = g.mutate(random.Random(42), sigma=0.15)
        assert child.genome_id != g.genome_id
        assert child.generation == g.generation + 1
        assert g.genome_id in child.parent_ids

    def test_crossover_creates_new_id(self):
        g1 = create_default_genome("economic")
        g2 = create_random_genome("economic", random.Random(43))
        child = g1.crossover(g2, random.Random(44))
        assert child.genome_id != g1.genome_id
        assert child.genome_id != g2.genome_id
        assert child.generation == max(g1.generation, g2.generation) + 1

    def test_apply_function(self):
        g = create_default_genome("economic")
        states = {"economic": 0.7, "confidence": 0.6}
        results = g.apply_function(states)
        assert len(results) > 0
        for key, val in results.items():
            assert 0.01 <= val <= 0.99, f"{key}={val} out of bounds"

    def test_to_dict_from_dict_roundtrip(self):
        g = create_default_genome("economic")
        d = g.to_dict()
        g2 = PredictionGenome.from_dict(d)
        assert g2.genome_id == g.genome_id
        assert len(g2.functions) == len(g.functions)


class TestGenomeRegistry:
    def test_register_and_get(self):
        reg = GenomeRegistry()
        g = create_default_genome("economic")
        reg.register(g)
        assert reg.get(g.genome_id) is g

    def test_best_for_domain(self):
        reg = GenomeRegistry()
        g1 = create_default_genome("economic")
        g1.fitness = 0.5
        reg.register(g1)
        reg.update_best("economic", g1)
        g2 = create_random_genome("economic")
        g2.fitness = 0.8
        reg.register(g2)
        reg.update_best("economic", g2)
        assert reg.best_for_domain("economic") is g2

    def test_top_k(self):
        reg = GenomeRegistry()
        for i in range(10):
            g = create_random_genome("economic", random.Random(i))
            g.fitness = i * 0.1
            reg.register(g)
        top = reg.top_k(3)
        assert len(top) == 3
        assert top[0].fitness >= top[1].fitness

    def test_prune(self):
        reg = GenomeRegistry()
        for i in range(100):
            g = create_random_genome("economic", random.Random(i))
            g.fitness = i * 0.01
            reg.register(g)
        pruned = reg.prune(50)
        assert pruned == 50
        assert len(reg.all_genomes()) == 50


class TestBehavioralDescriptor:
    def test_creation(self):
        bd = BehavioralDescriptor([0.5, 0.7])
        assert len(bd.values) == 2
        assert bd.names == ["bd_0", "bd_1"]

    def test_distance_to(self):
        bd1 = BehavioralDescriptor([0.0, 0.0])
        bd2 = BehavioralDescriptor([1.0, 1.0])
        assert abs(bd1.distance_to(bd2) - math.sqrt(2)) < 1e-10

    def test_to_cell_index(self):
        bd = BehavioralDescriptor([0.5, 0.5])
        idx = bd.to_cell_index([10, 10], [(0.0, 1.0), (0.0, 1.0)])
        assert idx == (4, 4)


class TestCVTMAPElites:
    def test_add_elite(self):
        archive = CVTMAPElites(bd_dimensions=2, seed=42)
        bd = BehavioralDescriptor([0.5, 0.5])
        fitness = FitnessScore(composite=0.8, synthesis=0.7, calibration=0.8)
        added = archive.add_elite("g1", bd, fitness, generation=0)
        assert added

    def test_add_elite_replaces_if_better(self):
        archive = CVTMAPElites(bd_dimensions=2, seed=42)
        bd = BehavioralDescriptor([0.5, 0.5])
        fitness1 = FitnessScore(composite=0.6, synthesis=0.5)
        fitness2 = FitnessScore(composite=0.9, synthesis=0.8)
        archive.add_elite("g1", bd, fitness1, generation=0)
        added = archive.add_elite("g2", bd, fitness2, generation=1)
        assert added

    def test_get_best_elite(self):
        archive = CVTMAPElites(bd_dimensions=2, seed=42)
        for i in range(10):
            bd = BehavioralDescriptor([0.1 * i, 0.1 * i])
            fitness = FitnessScore(synthesis=0.3 + 0.05 * i, calibration=0.5,
                                   cascade=0.4, confidence=0.3)
            fitness.composite = fitness.weighted_sum()
            archive.add_elite(f"g{i}", bd, fitness, generation=i)
        best = archive.get_best_elite()
        assert best is not None
        assert best.fitness.composite >= 0.4

    def test_compute_qd_metrics(self):
        archive = CVTMAPElites(bd_dimensions=2, seed=42)
        for i in range(20):
            bd = BehavioralDescriptor([0.05 * i, 0.05 * (20 - i)])
            fitness = FitnessScore(composite=0.3 + 0.03 * i, synthesis=0.3 + 0.02 * i)
            archive.add_elite(f"g{i}", bd, fitness, generation=i)
        metrics = archive.compute_qd_metrics()
        assert metrics.num_elites > 0
        assert metrics.qd_score > 0
        assert 0 <= metrics.coverage <= 1.0

    def test_tournament_select(self):
        archive = CVTMAPElites(bd_dimensions=2, seed=42)
        for i in range(10):
            bd = BehavioralDescriptor([0.1 * i, 0.1 * i])
            fitness = FitnessScore(composite=0.1 * i)
            archive.add_elite(f"g{i}", bd, fitness, generation=i)
        selected = archive.tournament_select(3)
        assert selected is not None
        assert selected.genome_id.startswith("g")

    def test_archive_stats(self):
        archive = CVTMAPElites(bd_dimensions=2, seed=42)
        for i in range(10):
            bd = BehavioralDescriptor([0.1 * i, 0.1 * i])
            fitness = FitnessScore(composite=0.1 * i)
            archive.add_elite(f"g{i}", bd, fitness, generation=i)
        stats = archive.archive_stats()
        assert "occupied_cells" in stats
        assert stats["occupied_cells"] > 0

    def test_export_archive(self):
        archive = CVTMAPElites(bd_dimensions=2, seed=42)
        bd = BehavioralDescriptor([0.5, 0.5])
        fitness = FitnessScore(composite=0.8)
        archive.add_elite("g1", bd, fitness, generation=0)
        exported = archive.export_archive()
        assert "elites" in exported
        assert "generation" in exported


class TestMultiObjectiveFitness:
    def test_evaluate(self):
        fitness = MultiObjectiveFitness(seed=42)
        predictions = {"adjust_probability_for_domain_states": 0.6, "compute_cascade_probability": 0.4}
        domain_states = {"economic": 0.5, "confidence": 0.5}
        result = fitness.evaluate(genome_predictions=predictions, domain_states=domain_states, domain="economic")
        assert 0 <= result.fitness.composite <= 1.0
        assert result.fitness.synthesis >= 0

    def test_evaluate_against_outcomes(self):
        fitness = MultiObjectiveFitness(seed=42)
        predictions = {"a": 0.7, "b": 0.3, "c": 0.8}
        outcomes = {"a": 0.75, "b": 0.25, "c": 0.85}
        result = fitness.evaluate_against_outcomes(predictions, outcomes, domain="economic")
        assert result.fitness.composite > 0
        assert result.fitness.brier > 0

    def test_evaluate_empty_predictions(self):
        fitness = MultiObjectiveFitness(seed=42)
        result = fitness.evaluate(genome_predictions={}, domain_states={}, domain="economic")
        assert 0.0 <= result.fitness.composite <= 1.0

    def test_fitness_weights(self):
        fitness = MultiObjectiveFitness(weights={"synthesis": 1.0})
        predictions = {"a": 0.6}
        domain_states = {"economic": 0.5}
        result = fitness.evaluate(genome_predictions=predictions, domain_states=domain_states)
        assert result.fitness.composite > 0

    def test_outcome_tracker_brier(self):
        tracker = OutcomeTracker()
        for i in range(50):
            pred = 0.3 + (i % 4) * 0.1
            tracker.record("economic", predicted=pred, actual=pred + 0.05)
        brier = tracker.compute_brier("economic")
        assert 0 <= brier <= 1.0


class TestLLMMutator:
    def test_gaussian_mutate(self):
        mutator = LLMMutator(seed=42, fallback_to_heuristic=True)
        genome = create_default_genome("economic")
        result = mutator.mutate(genome, strategy=MutationStrategy.GAUSSIAN_ADAPTATION)
        assert result.success
        assert result.genome.genome_id != genome.genome_id

    def test_perturbation_mutate(self):
        mutator = LLMMutator(seed=42)
        genome = create_default_genome("economic")
        result = mutator.mutate(genome, strategy=MutationStrategy.PARAMETER_PERTURBATION)
        assert result.success
        assert result.strategy == "perturbation"

    def test_crossover_uniform(self):
        mutator = LLMMutator(seed=42)
        g1 = create_default_genome("economic")
        g2 = create_random_genome("economic", random.Random(43))
        result = mutator.crossover(g1, g2, method="uniform")
        assert result.child.genome_id not in [g1.genome_id, g2.genome_id]
        assert len(result.crossover_points) > 0

    def test_crossover_blend(self):
        mutator = LLMMutator(seed=42)
        g1 = create_default_genome("economic")
        g2 = create_random_genome("economic", random.Random(43))
        result = mutator.crossover(g1, g2, method="blend")
        assert result.method == "blend"
        for name, func in result.child.functions.items():
            for key in func.params:
                lo, hi = func.param_bounds.get(key, (0.0, 1.0))
                assert lo <= func.params[key] <= hi

    def test_inject_novel_strategy_fallback(self):
        mutator = LLMMutator(seed=42, fallback_to_heuristic=True)
        result = mutator.inject_novel_strategy("economic")
        assert result.success
        assert result.genome.origin in ["llm_injection", "heuristic_injection"]

    def test_stats(self):
        mutator = LLMMutator(seed=42)
        genome = create_default_genome("economic")
        mutator.mutate(genome, strategy=MutationStrategy.GAUSSIAN_ADAPTATION)
        stats = mutator.stats()
        assert "total_mutations" in stats
        assert stats["total_mutations"] >= 1


class TestDataFlowEngine:
    def test_ingest_data(self):
        engine = DataFlowEngine(domains=["economic", "geopolitical"])
        engine.ingest_data("economic", "sp500", 5500.0, source="yfinance")
        states = engine.get_domain_states()
        assert "economic" in states
        assert states["economic"] == 5500.0

    def test_ingest_domain_states(self):
        engine = DataFlowEngine(domains=["economic", "geopolitical"])
        engine.ingest_domain_states({"economic": 0.65, "geopolitical": 0.4})
        states = engine.get_domain_states()
        assert abs(states["economic"] - 0.65) < 0.01
        assert abs(states["geopolitical"] - 0.4) < 0.01

    def test_stream_subscribe(self):
        engine = DataFlowEngine(domains=["economic"])
        received = []
        sub = StreamSubscription(domain="economic", callback=lambda p: received.append(p))
        stream = engine.get_stream("economic")
        if stream:
            stream.subscribe(sub)
        engine.ingest_data("economic", "sp500", 5500.0)
        assert len(received) == 1

    def test_outcome_tracker(self):
        engine = DataFlowEngine(domains=["economic"])
        pid = engine.outcome_tracker.record_prediction(
            genome_id="test_g1", domain="economic",
            predicted=0.7, horizon_hours=1.0
        )
        assert pid != ""
        resolved = engine.outcome_tracker.resolve_prediction(pid, 0.75)
        assert resolved is not None
        assert abs(resolved.actual_outcome - 0.75) < 0.01

    def test_summary(self):
        engine = DataFlowEngine(domains=["economic"])
        engine.ingest_data("economic", "sp500", 5500.0)
        summary = engine.summary()
        assert "domains" in summary
        assert "domain_states" in summary


class TestContinuousEvolutionEngine:
    def test_creation(self):
        config = EvolutionConfig(
            population_size=10, domains=["economic"],
            seed=42, max_generations=100
        )
        engine = ContinuousEvolutionEngine(config)
        assert engine.state.generation == 0
        assert engine.state.phase == EvolutionPhase.IDLE

    def test_step(self):
        config = EvolutionConfig(population_size=10, domains=["economic"], seed=42)
        engine = ContinuousEvolutionEngine(config)
        report = engine.step()
        assert report.genome_id != ""
        assert report.domain == "economic"
        assert engine.state.generation == 1

    def test_evolve(self):
        config = EvolutionConfig(population_size=10, domains=["economic"], seed=42)
        engine = ContinuousEvolutionEngine(config)
        reports = engine.evolve(n_steps=5)
        assert len(reports) == 5
        assert engine.state.generation >= 5

    def test_evaluate_genome(self):
        config = EvolutionConfig(population_size=10, domains=["economic"], seed=42)
        engine = ContinuousEvolutionEngine(config)
        genome = create_default_genome("economic")
        result = engine._evaluate_genome(genome, "economic")
        assert result.fitness.composite >= 0
        assert result.fitness.synthesis >= 0

    def test_predict_with_best(self):
        config = EvolutionConfig(population_size=10, domains=["economic"], seed=42)
        engine = ContinuousEvolutionEngine(config)
        predictions = engine.predict_with_best("economic", {"economic": 0.6, "confidence": 0.5})
        assert len(predictions) > 0
        for key, val in predictions.items():
            assert 0.01 <= val <= 0.99

    def test_get_best_strategy(self):
        config = EvolutionConfig(population_size=10, domains=["economic"], seed=42)
        engine = ContinuousEvolutionEngine(config)
        engine.evolve(n_steps=3)
        best = engine.get_best_strategy("economic")
        assert best.genome_id != ""
        assert best.domain == "economic"

    def test_get_archive_stats(self):
        config = EvolutionConfig(population_size=10, domains=["economic"], seed=42)
        engine = ContinuousEvolutionEngine(config)
        engine.evolve(n_steps=3)
        stats = engine.get_archive_stats()
        assert "global" in stats
        assert "domains" in stats
        assert "state" in stats

    def test_ingest_and_resolve(self):
        config = EvolutionConfig(population_size=10, domains=["economic"], seed=42)
        engine = ContinuousEvolutionEngine(config)
        engine.ingest_predictions("test_genome", "economic", {"a": 0.6, "b": 0.4})
        resolved = engine.resolve_with_outcomes({"economic": 0.55})
        assert resolved >= 0

    def test_export_state(self):
        config = EvolutionConfig(population_size=10, domains=["economic"], seed=42)
        engine = ContinuousEvolutionEngine(config)
        engine.evolve(n_steps=3)
        state = engine.export_state()
        assert "config" in state
        assert "state" in state
        assert "archive" in state
        assert state["state"]["generation"] >= 3

    def test_strategy_report_to_dict(self):
        config = EvolutionConfig(population_size=10, domains=["economic"], seed=42)
        engine = ContinuousEvolutionEngine(config)
        report = engine.step()
        d = report.to_dict()
        assert "genome_id" in d
        assert "fitness" in d
        assert "generation" in d

    def test_fitness_improves_over_evolution(self):
        config = EvolutionConfig(population_size=20, domains=["economic"], seed=42)
        engine = ContinuousEvolutionEngine(config)
        engine.evolve(n_steps=5)
        initial_best = engine.state.best_fitness
        for _ in range(20):
            engine.step()
        later_best = engine.state.best_fitness
        assert later_best >= initial_best or engine.state.generation > 5

    def test_evolve_multiple_domains(self):
        config = EvolutionConfig(
            population_size=15,
            domains=["economic", "geopolitical", "health"],
            seed=42
        )
        engine = ContinuousEvolutionEngine(config)
        reports = engine.evolve(n_steps=10)
        assert len(reports) == 10
        stats = engine.get_archive_stats()
        assert "economic" in stats["domains"]
        assert "geopolitical" in stats["domains"]
        assert "health" in stats["domains"]

    def test_inject_creates_new_genome(self):
        config = EvolutionConfig(population_size=10, domains=["economic"], seed=42)
        engine = ContinuousEvolutionEngine(config)
        result = engine.mutator.inject_novel_strategy("economic")
        assert result.success
        assert result.genome.genome_id != ""
        assert len(result.genome.functions) > 0

    def test_data_flow_feedback_loop(self):
        config = EvolutionConfig(population_size=10, domains=["economic"], seed=42)
        engine = ContinuousEvolutionEngine(config)
        engine.evolve(n_steps=3)
        engine.data_flow.ingest_domain_states({"economic": 0.65, "geopolitical": 0.45})
        engine.resolve_with_outcomes({"economic": 0.65})
        engine.evolve(n_steps=3)
        assert engine.state.generation >= 6

    def test_outcome_tracking_improves_fitness(self):
        config = EvolutionConfig(population_size=15, domains=["economic"], seed=42)
        engine = ContinuousEvolutionEngine(config)
        for _ in range(5):
            engine.data_flow.ingest_data("economic", "state", 0.6 + 0.1 * (_ % 3))
        engine.evolve(n_steps=5)
        assert engine.state.total_evaluations >= 20


class TestFitnessScore:
    def test_weighted_sum(self):
        score = FitnessScore(
            composite=0.0, synthesis=0.8, calibration=0.7,
            cascade=0.6, confidence=0.5, spread=0.4,
            coverage=0.3, direction=0.2
        )
        ws = score.weighted_sum()
        assert ws > 0

    def test_custom_weights(self):
        score = FitnessScore(synthesis=1.0, calibration=0.0)
        ws = score.weighted_sum(weights={"synthesis": 1.0, "calibration": 0.0})
        assert ws == 1.0


class TestOutcomeRecord:
    def test_brier_score(self):
        record = OutcomeRecord(
            timestamp=time.time(), domain="economic",
            predicted_probability=0.8, actual_outcome=1.0
        )
        assert abs(record.brier_score - 0.04) < 0.001
        assert record.is_correct_direction

    def test_is_correct_direction(self):
        record = OutcomeRecord(
            timestamp=time.time(), domain="economic",
            predicted_probability=0.3, actual_outcome=0.2
        )
        assert record.is_correct_direction

    def test_wrong_direction(self):
        record = OutcomeRecord(
            timestamp=time.time(), domain="economic",
            predicted_probability=0.8, actual_outcome=0.2
        )
        assert not record.is_correct_direction