import pytest
import numpy as np
from datetime import datetime
from oracle.prediction.butterfly_effect import (
    Perturbation, PropagationStep, PropagationTrace, DomainNetwork,
    ButterflyTracker, CriticalNode, AmplificationPath, EarlyWarning,
    MECHANISM_AMPLIFICATION, DOMAIN_CONNECTIONS, DOMAINS, VALID_MECHANISMS,
)


class TestPerturbation:
    def test_default_timestamp(self):
        p = Perturbation(variable="interest_rate", magnitude=0.5, source_domain="financial")
        assert p.variable == "interest_rate"
        assert p.magnitude == 0.5
        assert p.source_domain == "financial"
        assert isinstance(p.timestamp, datetime)

    def test_custom_timestamp(self):
        ts = datetime(2025, 1, 1)
        p = Perturbation(variable="x", magnitude=1.0, source_domain="economic", timestamp=ts)
        assert p.timestamp == ts


class TestPropagationStep:
    def test_fields(self):
        step = PropagationStep(
            from_domain="economic", to_domain="financial",
            amplification=1.2, mechanism="contagion", cumulative_impact=0.6,
        )
        assert step.from_domain == "economic"
        assert step.to_domain == "financial"
        assert step.amplification == 1.2
        assert step.mechanism == "contagion"
        assert step.cumulative_impact == 0.6


class TestDomainNetwork:
    def test_add_connection(self):
        net = DomainNetwork()
        net.add_connection("a", "b", 0.5, "contagion")
        assert "a" in net.domains
        assert "b" in net.domains
        assert net.n_connections == 1

    def test_add_connection_invalid_mechanism(self):
        net = DomainNetwork()
        with pytest.raises(ValueError, match="Invalid mechanism"):
            net.add_connection("a", "b", 0.5, "invalid_mech")

    def test_add_connection_invalid_weight(self):
        net = DomainNetwork()
        with pytest.raises(ValueError, match="Weight must be"):
            net.add_connection("a", "b", 1.5, "contagion")
        with pytest.raises(ValueError, match="Weight must be"):
            net.add_connection("a", "b", -0.1, "cascade")

    def test_propagate_basic(self):
        net = DomainNetwork()
        net.add_connection("a", "b", 0.8, "contagion")
        net.add_connection("b", "c", 0.6, "cascade")
        p = Perturbation(variable="test", magnitude=1.0, source_domain="a")
        steps = net.propagate(p, max_steps=5)
        assert len(steps) >= 2
        assert steps[0].from_domain == "a"
        assert steps[0].to_domain == "b"
        reached = {s.to_domain for s in steps}
        assert "b" in reached
        assert "c" in reached

    def test_propagate_max_steps(self):
        net = DomainNetwork()
        for i in range(15):
            net.add_connection(f"d{i}", f"d{i+1}", 0.9, "cascade")
        p = Perturbation(variable="test", magnitude=1.0, source_domain="d0")
        steps = net.propagate(p, max_steps=3)
        assert len(steps) >= 3

    def test_propagate_isolated_domain(self):
        net = DomainNetwork()
        net.add_connection("a", "b", 0.5, "contagion")
        p = Perturbation(variable="test", magnitude=1.0, source_domain="c")
        steps = net.propagate(p)
        assert len(steps) == 0

    def test_compute_butterfly_index(self):
        net = DomainNetwork()
        net.add_connection("a", "b", 0.9, "amplification")
        net.add_connection("b", "c", 0.9, "amplification")
        p = Perturbation(variable="test", magnitude=1.0, source_domain="a")
        idx = net.compute_butterfly_index(p)
        assert idx > 1.0

    def test_compute_butterfly_index_zero_magnitude(self):
        net = DomainNetwork()
        net.add_connection("a", "b", 0.8, "cascade")
        p = Perturbation(variable="test", magnitude=0.0, source_domain="a")
        idx = net.compute_butterfly_index(p)
        assert idx == 1.0

    def test_compute_butterfly_index_no_neighbors(self):
        net = DomainNetwork()
        p = Perturbation(variable="test", magnitude=1.0, source_domain="z")
        idx = net.compute_butterfly_index(p)
        assert idx == 1.0

    def test_find_critical_nodes(self):
        net = DomainNetwork()
        net.add_connection("hub", "a", 0.9, "cascade")
        net.add_connection("hub", "b", 0.8, "amplification")
        net.add_connection("hub", "c", 0.7, "contagion")
        net.add_connection("a", "b", 0.3, "damping")
        nodes = net.find_critical_nodes()
        assert len(nodes) == 4
        assert nodes[0].domain == "hub"
        assert nodes[0].out_weight > 0

    def test_sensitivity_analysis(self):
        net = DomainNetwork()
        net.add_connection("a", "b", 0.9, "amplification")
        net.add_connection("a", "c", 0.8, "cascade")
        net.add_connection("b", "a", 0.7, "feedback_loop")
        sens = net.sensitivity_analysis("a")
        assert 0.0 <= sens <= 1.0

    def test_sensitivity_analysis_unknown_domain(self):
        net = DomainNetwork()
        assert net.sensitivity_analysis("nonexistent") == 0.0

    def test_damping_reduces_amplification(self):
        net = DomainNetwork()
        net.add_connection("a", "b", 0.5, "damping")
        p = Perturbation(variable="test", magnitude=1.0, source_domain="a")
        steps = net.propagate(p)
        assert len(steps) >= 1
        assert steps[0].amplification < 1.0

    def test_mechanism_amplification_factors(self):
        assert MECHANISM_AMPLIFICATION["damping"] < 1.0
        assert MECHANISM_AMPLIFICATION["threshold"] > 1.0
        assert MECHANISM_AMPLIFICATION["feedback_loop"] > 1.0


class TestDomainNetworkPrePopulated:
    def test_domain_count(self):
        bt = ButterflyTracker()
        assert len(bt.network.domains) >= 15

    def test_connection_count(self):
        bt = ButterflyTracker()
        assert bt.network.n_connections >= 30

    def test_all_domains_connected(self):
        bt = ButterflyTracker()
        for domain in DOMAINS:
            neighbors = bt.network.get_neighbors(domain)
            assert len(neighbors) > 0, f"{domain} has no outgoing connections"

    def test_domain_connections_data(self):
        assert len(DOMAIN_CONNECTIONS) >= 30
        for from_d, to_d, w, m in DOMAIN_CONNECTIONS:
            assert 0.0 <= w <= 1.0
            assert m in VALID_MECHANISMS


class TestButterflyTracker:
    def test_init(self):
        bt = ButterflyTracker()
        assert bt.network is not None
        assert len(bt.network.domains) >= 15

    def test_trace_perturbation(self):
        bt = ButterflyTracker()
        trace = bt.trace_perturbation("interest_rate_hike", magnitude=0.5, source_domain="financial")
        assert isinstance(trace, PropagationTrace)
        assert trace.perturbation.variable == "interest_rate_hike"
        assert len(trace.steps) > 0
        assert len(trace.reached_domains) >= 2
        assert trace.butterfly_index > 1.0

    def test_trace_perturbation_auto_domain(self):
        bt = ButterflyTracker()
        trace = bt.trace_perturbation("economic_recession", magnitude=1.0)
        assert trace.perturbation.source_domain in bt.network.domains

    def test_trace_perturbation_unknown_domain(self):
        bt = ButterflyTracker()
        trace = bt.trace_perturbation("test_var", magnitude=1.0, source_domain="nonexistent_domain")
        assert len(trace.reached_domains) >= 1

    def test_butterfly_score(self):
        bt = ButterflyTracker()
        score = bt.butterfly_score("geopolitical")
        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_butterfly_score_high_connectivity(self):
        bt = ButterflyTracker()
        score_geopolitical = bt.butterfly_score("geopolitical")
        assert score_geopolitical > 0.5

    def test_butterfly_score_reproducible(self):
        bt1 = ButterflyTracker(seed=42)
        bt2 = ButterflyTracker(seed=42)
        assert bt1.butterfly_score("economic") == bt2.butterfly_score("economic")

    def test_find_amplification_paths(self):
        bt = ButterflyTracker()
        paths = bt.find_amplification_paths(threshold=0.5)
        assert isinstance(paths, list)
        for p in paths:
            assert isinstance(p, AmplificationPath)
            assert len(p.path) >= 2
            assert p.cumulative_amplification >= 0.5

    def test_find_amplification_paths_strict_threshold(self):
        bt = ButterflyTracker()
        paths_low = bt.find_amplification_paths(threshold=0.5)
        paths_high = bt.find_amplification_paths(threshold=2.0)
        assert len(paths_low) >= len(paths_high)

    def test_cascade_potential(self):
        bt = ButterflyTracker()
        cp = bt.cascade_potential()
        assert 0.0 <= cp <= 1.0
        assert cp > 0.0

    def test_cascade_potential_dense_network(self):
        bt = ButterflyTracker()
        cp = bt.cascade_potential()
        assert cp > 0.5

    def test_get_early_warnings(self):
        bt = ButterflyTracker()
        warnings = bt.get_early_warnings()
        assert isinstance(warnings, list)
        for w in warnings:
            assert isinstance(w, EarlyWarning)
            assert w.severity in ("critical", "high", "medium", "low")
            assert 0.0 <= w.threshold_proximity <= 1.0
            assert len(w.connected_domains) > 0

    def test_get_early_warnings_sorted(self):
        bt = ButterflyTracker()
        warnings = bt.get_early_warnings()
        for i in range(len(warnings) - 1):
            assert warnings[i].threshold_proximity >= warnings[i + 1].threshold_proximity

    def test_perturbation_history(self):
        bt = ButterflyTracker()
        bt.trace_perturbation("a", 1.0, "economic")
        bt.trace_perturbation("b", 0.5, "climate")
        assert len(bt._perturbation_history) == 2
        assert len(bt._trace_history) == 2


class TestMechanismEffects:
    def test_feedback_loop_amplifies(self):
        net = DomainNetwork()
        net.add_connection("a", "b", 0.8, "feedback_loop")
        p = Perturbation(variable="test", magnitude=1.0, source_domain="a")
        steps = net.propagate(p)
        if steps:
            assert steps[0].amplification > 0.8

    def test_threshold_max_amplification(self):
        assert MECHANISM_AMPLIFICATION["threshold"] == max(MECHANISM_AMPLIFICATION.values())

    def test_damping_min_amplification(self):
        assert MECHANISM_AMPLIFICATION["damping"] == min(MECHANISM_AMPLIFICATION.values())


class TestEdgeCases:
    def test_very_small_magnitude(self):
        bt = ButterflyTracker()
        trace = bt.trace_perturbation("tiny", magnitude=0.001, source_domain="economic")
        assert trace.butterfly_index > 0

    def test_very_large_magnitude(self):
        bt = ButterflyTracker()
        trace = bt.trace_perturbation("huge", magnitude=100.0, source_domain="economic")
        assert trace.butterfly_index > 0

    def test_single_domain_network(self):
        net = DomainNetwork()
        p = Perturbation(variable="test", magnitude=1.0, source_domain="a")
        steps = net.propagate(p)
        assert len(steps) == 0

    def test_cyclic_network(self):
        net = DomainNetwork()
        net.add_connection("a", "b", 0.8, "contagion")
        net.add_connection("b", "c", 0.7, "cascade")
        net.add_connection("c", "a", 0.6, "feedback_loop")
        p = Perturbation(variable="test", magnitude=1.0, source_domain="a")
        steps = net.propagate(p, max_steps=5)
        assert len(steps) >= 2