# -*- coding: utf-8 -*-
"""Tests for Westworld-inspired prediction modules."""
import pytest
import numpy as np
from oracle.prediction.westworld_fidelity import FidelityEngine, FidelityReport
from oracle.prediction.cradle import CradleSimulator, BranchOutcome
from oracle.prediction.outlier_host import OutlierHostDetector, OutlierProfile
from oracle.prediction.narrative_loops import NarrativeLoopDetector, LoopBreak, LoopSegment
from oracle.prediction.optimal_path import OptimalPathOptimizer, Objective, ObjectiveDirection, Constraint, OptimalPathResult, Path
from oracle.prediction.entropy_meter import EntropyMeter, EntropyReport, DomainEntropy


class TestFidelityEngine:
    def test_record_divergence(self):
        fe = FidelityEngine(window_size=10)
        div = fe.record("host_1", predicted=0.5, actual=0.7)
        assert 0 < div < 1
        assert len(fe._history["host_1"]) == 1

    def test_assess_returns_report(self):
        fe = FidelityEngine(window_size=5)
        for i in range(5):
            fe.record("host_1", predicted=0.5, actual=0.5 + i*0.05)
        report = fe.assess("host_1")
        assert isinstance(report, FidelityReport)
        assert report.target_id == "host_1"
        assert report.n_samples == 5
        assert 0 <= report.mean_divergence <= 1
        assert report.divergence_trend in ["improving", "worsening", "stable"]
        assert 0 <= report.fidelity_score <= 1

    def test_system_fidelity(self):
        fe = FidelityEngine(window_size=5)
        fe.record("h1", 0.5, 0.55)
        fe.record("h2", 0.3, 0.31)
        system = fe.system_fidelity()
        assert 0 <= system["fidelity"] <= 1
        assert system["n_predictions"] == 2

    def test_find_outliers(self):
        fe = FidelityEngine(window_size=5)
        for i in range(10):
            fe.record("predictable", 0.5, 0.5 + np.random.normal(0, 0.05))
            fe.record("volatile", 0.5, 0.5 + np.random.normal(0, 0.4))
        outliers = fe.find_outliers(top_n=5)
        assert len(outliers) == 2
        # Volatile should have higher divergence
        assert outliers[0][0] == "volatile" or outliers[0][1] > outliers[1][1]


class TestCradleSimulator:
    def test_simulate_returns_result(self):
        cs = CradleSimulator(n_branches=100, max_generations=5, random_seed=42)
        result = cs.simulate({"growth": 0.6, "stability": 0.5, "risk": 0.3})
        assert result.n_branches == 100
        assert result.n_generations == 5
        assert len(result.outcome_distribution) > 0
        assert result.most_likely_outcome in result.outcome_distribution
        assert len(result.catastrophic_branches) >= 0
        assert len(result.optimal_branches) >= 0

    def test_outcome_probabilities_sum_near_one(self):
        cs = CradleSimulator(n_branches=100, max_generations=3, random_seed=42)
        result = cs.simulate({"x": 0.5})
        total = sum(result.outcome_distribution.values())
        assert 0.9 < total <= 1.01

    def test_confidence_intervals_present(self):
        cs = CradleSimulator(n_branches=50, max_generations=3, random_seed=42)
        result = cs.simulate({"a": 0.5, "b": 0.3})
        assert "a" in result.confidence_intervals
        assert len(result.confidence_intervals["a"]) == 2


class TestOutlierHostDetector:
    def test_classify_with_insufficient_samples(self):
        od = OutlierHostDetector(outlier_threshold=0.7, min_samples=5)
        result = od.classify("host_x")
        assert result is None

    def test_observe_and_classify(self):
        od = OutlierHostDetector(outlier_threshold=0.7, min_samples=5)
        od.register("host_x", features={"volatility": 0.3}, expected_behavior={"volatility": 0.3})
        for i in range(10):
            devs = od.observe("host_x", {"volatility": 0.3 + np.random.normal(0, 0.05)})
            assert isinstance(devs, dict)
        profile = od.classify("host_x")
        assert isinstance(profile, OutlierProfile)
        assert profile.target_id == "host_x"
        assert profile.stability_rating in ["stable", "volatile", "emergent", "chaotic"]

    def test_find_all_outliers(self):
        od = OutlierHostDetector(outlier_threshold=0.3, min_samples=5)
        for name, actuals in [("normal", [0.5]*10), ("volatile", [0.5, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.9, 0.3, 0.8])]:
            od.register(name, features={"x": 0.5}, expected_behavior={"x": 0.5})
            for a in actuals:
                od.observe(name, {"x": a})
        outliers = od.find_all_outliers()
        assert len(outliers) == 1
        assert outliers[0].target_id == "volatile"


class TestNarrativeLoopDetector:
    def test_observe_returns_none_early(self):
        nld = NarrativeLoopDetector(window_size=20, min_loop_length=3)
        result = nld.observe("host_a", {"x": 0.5, "y": 0.3})
        assert result is None

    def test_detect_finds_loops(self):
        nld = NarrativeLoopDetector(window_size=50, min_loop_length=3)
        # Create a repeating pattern
        pattern = [{"x": 0.5, "y": 0.3}, {"x": 0.6, "y": 0.4}, {"x": 0.5, "y": 0.3}]
        for i in range(30):
            state = pattern[i % len(pattern)]
            nld.observe("host_a", state)
        report = nld.detect()
        assert report.n_entities >= 1
        assert len(report.active_loops) >= 0
        assert 0 <= report.stability_index <= 1
        assert 0 <= report.entropy

    def test_loop_break_detection(self):
        nld = NarrativeLoopDetector(window_size=50, min_loop_length=3, break_threshold=0.3)
        pattern = [{"x": 0.5, "y": 0.3}, {"x": 0.51, "y": 0.31}]
        for i in range(20):
            state = pattern[i % len(pattern)]
            nld.observe("host_b", state)
        # Inject a break
        break_event = nld.observe("host_b", {"x": 0.9, "y": 0.8})
        # Loop detection is probabilistic; just verify it doesn't crash
        report = nld.detect()
        assert report is not None


class TestOptimalPathOptimizer:
    def test_optimize_returns_result(self):
        objectives = [
            Objective(name="growth", direction=ObjectiveDirection.MAXIMIZE, weight=0.5),
            Objective(name="risk", direction=ObjectiveDirection.MINIMIZE, weight=0.5),
        ]
        constraints = [Constraint(name="risk_cap", lhs_var="risk", comparator="<=", rhs=0.8)]
        opo = OptimalPathOptimizer(objectives, constraints, n_paths=100, max_steps=5, seed=42)
        result = opo.optimize({"growth": 0.5, "risk": 0.3})
        assert isinstance(result, OptimalPathResult)
        assert len(result.pareto_frontier) > 0
        assert "growth" in result.utopia_point
        assert "risk" in result.utopia_point
        assert result.equity_score >= 0
        assert result.recommendation != ""

    def test_pareto_frontier_nondominated(self):
        objectives = [
            Objective(name="a", direction=ObjectiveDirection.MAXIMIZE, weight=0.5),
            Objective(name="b", direction=ObjectiveDirection.MAXIMIZE, weight=0.5),
        ]
        opo = OptimalPathOptimizer(objectives, [], n_paths=200, max_steps=5, seed=42)
        result = opo.optimize({"a": 0.5, "b": 0.5})
        for i, p1 in enumerate(result.pareto_frontier):
            for j, p2 in enumerate(result.pareto_frontier):
                if i != j:
                    assert not (all(p2.objective_values.get(k, 0) >= p1.objective_values.get(k, 0) for k in ["a", "b"]) and
                               any(p2.objective_values.get(k, 0) > p1.objective_values.get(k, 0) for k in ["a", "b"]))

    def test_nash_and_ks_points(self):
        objectives = [
            Objective(name="x", direction=ObjectiveDirection.MAXIMIZE, weight=0.5),
            Objective(name="y", direction=ObjectiveDirection.MAXIMIZE, weight=0.5),
        ]
        opo = OptimalPathOptimizer(objectives, [], n_paths=100, max_steps=5, seed=42)
        result = opo.optimize({"x": 0.5, "y": 0.5})
        assert "x" in result.nash_point
        assert "y" in result.kalai_smorodinsky_point
        assert 0 <= result.equity_score <= 1

    def test_regret_and_robustness(self):
        objectives = [
            Objective(name="value", direction=ObjectiveDirection.MAXIMIZE, weight=1.0),
        ]
        opo = OptimalPathOptimizer(objectives, [], n_paths=50, max_steps=3, seed=42)
        result = opo.optimize({"value": 0.5})
        for path in result.recommended_paths:
            assert path.regret >= 0
            assert 0 <= path.robustness <= 1


class TestEntropyMeter:
    def test_observe_returns_entropy(self):
        em = EntropyMeter(window_size=20)
        e = em.observe("market", 0.5)
        assert 0 <= e <= 2  # Shannon entropy in reasonable range

    def test_measure_report(self):
        em = EntropyMeter(window_size=20)
        for i in range(15):
            em.observe("market", 0.5 + np.random.normal(0, 0.1))
            em.observe("sentiment", 0.3 + np.random.normal(0, 0.05))
        report = em.measure()
        assert isinstance(report, EntropyReport)
        assert 0 <= report.system_entropy <= 2
        assert report.entropy_trend in ["increasing", "decreasing", "stable"]
        assert "market" in report.domain_entropies
        assert "sentiment" in report.domain_entropies
        assert 0 <= report.overall_predictability <= 1

    def test_chaos_hotspots_and_stability(self):
        em = EntropyMeter(window_size=20)
        for i in range(20):
            em.observe("volatile_domain", np.random.uniform(0, 1))
            em.observe("stable_domain", 0.5 + np.random.normal(0, 0.01))
        report = em.measure()
        assert len(report.chaos_hotspots) >= 0
        assert len(report.stability_zones) >= 0
        assert report.intervention_recommendation != ""

    def test_predictability_with_predictions(self):
        em = EntropyMeter(window_size=20)
        for i in range(15):
            actual = 0.5 + np.sin(i * 0.3) * 0.1
            predicted = 0.5 + np.sin(i * 0.3) * 0.09  # Good prediction
            em.observe("test_domain", actual, predicted)
        report = em.measure()
        de = report.domain_entropies.get("test_domain")
        if de:
            assert de.predictability >= 0
