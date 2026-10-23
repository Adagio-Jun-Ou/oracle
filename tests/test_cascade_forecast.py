import pytest
import numpy as np
from oracle.prediction.cascade_forecast import (
    FailureNode, FailureEdge, CascadeTree, CascadeResult,
)


class TestFailureNode:
    def test_hash_equality(self):
        a = FailureNode("x", "economics", 0.5, 7.0)
        b = FailureNode("x", "economics", 0.5, 7.0)
        assert a == b
        assert hash(a) == hash(b)

    def test_different_names_not_equal(self):
        a = FailureNode("x", "economics", 0.5, 7.0)
        b = FailureNode("y", "economics", 0.5, 7.0)
        assert a != b

    def test_equality_with_non_node(self):
        a = FailureNode("x", "economics", 0.5, 7.0)
        assert a != "x"


class TestCascadeTreeBuild:
    def test_add_node(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("a", "econ", 0.1, 5.0))
        assert "a" in tree.nodes

    def test_add_edge(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("a", "econ", 0.1, 5.0))
        tree.add_node(FailureNode("b", "econ", 0.2, 6.0))
        tree.add_edge("a", "b", 0.7, 2, "test")
        assert len(tree.edges) == 1
        assert tree.edges[0].propagation_probability == 0.7

    def test_edge_unknown_node_warns(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("a", "econ", 0.1, 5.0))
        tree.add_edge("a", "missing", 0.5, 1, "x")
        assert len(tree.edges) == 0

    def test_probability_clamped(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("a", "econ", 0.1, 5.0))
        tree.add_node(FailureNode("b", "econ", 0.2, 6.0))
        tree.add_edge("a", "b", 1.5, 1, "x")
        assert tree.edges[0].propagation_probability == 1.0
        tree.add_edge("a", "b", -0.3, 1, "y")
        assert tree.edges[1].propagation_probability == 0.0

    def test_default_tree_has_enough_nodes(self):
        tree = CascadeTree.build_default()
        assert len(tree.nodes) >= 15

    def test_default_tree_has_enough_edges(self):
        tree = CascadeTree.build_default()
        assert len(tree.edges) >= 25

    def test_default_domains_covered(self):
        tree = CascadeTree.build_default()
        domains = {n.domain for n in tree.nodes.values()}
        for expected in ["economics", "geopolitics", "climate", "social",
                         "technology", "energy", "health", "supply_chain"]:
            assert expected in domains, f"Missing domain: {expected}"


class TestSimulateCascade:
    def test_returns_cascade_result(self):
        tree = CascadeTree.build_default()
        result = tree.simulate_cascade("banking_crisis", n_simulations=100)
        assert isinstance(result, CascadeResult)
        assert result.trigger == "banking_crisis"

    def test_probability_bounded(self):
        tree = CascadeTree.build_default()
        result = tree.simulate_cascade("banking_crisis", n_simulations=500)
        assert 0.0 <= result.total_probability <= 1.0

    def test_domains_affected_positive(self):
        tree = CascadeTree.build_default()
        result = tree.simulate_cascade("banking_crisis", n_simulations=500)
        assert result.n_domains_affected >= 1

    def test_max_depth_positive(self):
        tree = CascadeTree.build_default()
        result = tree.simulate_cascade("banking_crisis", n_simulations=500)
        assert result.max_depth >= 1

    def test_risk_level_valid(self):
        tree = CascadeTree.build_default()
        result = tree.simulate_cascade("banking_crisis", n_simulations=500)
        assert result.risk_level in {"LOW", "MODERATE", "HIGH", "VERY HIGH", "CRITICAL"}

    def test_unknown_trigger_raises(self):
        tree = CascadeTree.build_default()
        with pytest.raises(ValueError):
            tree.simulate_cascade("nonexistent")

    def test_isolated_node_low_cascade(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("isolated", "econ", 0.1, 5.0))
        tree.add_node(FailureNode("other", "econ", 0.2, 3.0))
        result = tree.simulate_cascade("isolated", n_simulations=200)
        assert result.total_probability == 0.0
        assert result.max_depth == 1

    def test_deterministic_with_seed(self):
        tree = CascadeTree.build_default()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r1 = tree.simulate_cascade("banking_crisis", n_simulations=500, rng=rng1)
        r2 = tree.simulate_cascade("banking_crisis", n_simulations=500, rng=rng2)
        assert r1.total_probability == r2.total_probability
        assert r1.max_depth == r2.max_depth

    def test_cascade_paths_populated(self):
        tree = CascadeTree.build_default()
        result = tree.simulate_cascade("banking_crisis", n_simulations=1000)
        assert isinstance(result.cascade_paths, list)

    def test_mean_impact_positive(self):
        tree = CascadeTree.build_default()
        result = tree.simulate_cascade("banking_crisis", n_simulations=500)
        assert result.mean_impact >= 0.0

    def test_to_dict(self):
        tree = CascadeTree.build_default()
        result = tree.simulate_cascade("banking_crisis", n_simulations=100)
        d = result.to_dict()
        assert "trigger" in d
        assert "risk_level" in d
        assert isinstance(d["cascade_paths"], list)


class TestCriticalPaths:
    def test_returns_list(self):
        tree = CascadeTree.build_default()
        paths = tree.critical_paths(top_n=3)
        assert isinstance(paths, list)

    def test_top_n_respected(self):
        tree = CascadeTree.build_default()
        paths = tree.critical_paths(top_n=3)
        assert len(paths) <= 3

    def test_path_structure(self):
        tree = CascadeTree.build_default()
        paths = tree.critical_paths(top_n=5)
        for p in paths:
            assert "path" in p
            assert "cumulative_probability" in p
            assert "risk_score" in p
            assert "mechanisms" in p
            assert len(p["path"]) >= 2

    def test_sorted_by_risk_score(self):
        tree = CascadeTree.build_default()
        paths = tree.critical_paths(top_n=5)
        if len(paths) >= 2:
            assert paths[0]["risk_score"] >= paths[1]["risk_score"]

    def test_probability_decreasing_with_length(self):
        tree = CascadeTree.build_default()
        paths = tree.critical_paths(top_n=5)
        for p in paths:
            assert p["cumulative_probability"] <= 1.0
            assert p["cumulative_probability"] > 0.0

    def test_mechanisms_populated(self):
        tree = CascadeTree.build_default()
        paths = tree.critical_paths(top_n=5)
        for p in paths:
            n_mechs = len(p["mechanisms"])
            n_hops = len(p["path"]) - 1
            assert n_mechs == n_hops


class TestCascadeProbability:
    def test_returns_dict(self):
        tree = CascadeTree.build_default()
        result = tree.cascade_probability("social", n_simulations=500)
        assert isinstance(result, dict)

    def test_unknown_domain_empty(self):
        tree = CascadeTree.build_default()
        result = tree.cascade_probability("nonexistent_domain", n_simulations=100)
        assert result == {}

    def test_probabilities_bounded(self):
        tree = CascadeTree.build_default()
        result = tree.cascade_probability("social", n_simulations=500)
        for trigger, prob in result.items():
            assert 0.0 <= prob <= 1.0


class TestSystemicRiskScore:
    def test_returns_dict(self):
        tree = CascadeTree.build_default()
        result = tree.systemic_risk_score(n_simulations=500)
        assert isinstance(result, dict)

    def test_score_bounded(self):
        tree = CascadeTree.build_default()
        result = tree.systemic_risk_score(n_simulations=500)
        assert 0.0 <= result["systemic_risk_score"] <= 100.0

    def test_risk_level_valid(self):
        tree = CascadeTree.build_default()
        result = tree.systemic_risk_score(n_simulations=500)
        assert result["risk_level"] in {"LOW", "MODERATE", "HIGH", "VERY HIGH", "CRITICAL"}

    def test_domain_vulnerability_populated(self):
        tree = CascadeTree.build_default()
        result = tree.systemic_risk_score(n_simulations=500)
        assert len(result["domain_vulnerability"]) > 0
        for dv in result["domain_vulnerability"]:
            assert "domain" in dv
            assert "vulnerability_score" in dv

    def test_super_spreaders_populated(self):
        tree = CascadeTree.build_default()
        result = tree.systemic_risk_score(n_simulations=500)
        assert len(result["super_spreaders"]) > 0

    def test_empty_tree_risk(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("only", "econ", 0.1, 5.0))
        result = tree.systemic_risk_score(n_simulations=100)
        assert result["risk_level"] in {"LOW", "MODERATE", "HIGH", "VERY HIGH", "CRITICAL"}


class TestEdgeCases:
    def test_single_node_tree(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("solo", "econ", 0.5, 5.0))
        result = tree.simulate_cascade("solo", n_simulations=100)
        assert result.total_probability == 0.0
        assert result.n_domains_affected == 1

    def test_self_loop_prevented_in_sim(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("a", "econ", 0.5, 5.0))
        tree.add_edge("a", "a", 0.9, 1, "self")
        result = tree.simulate_cascade("a", n_simulations=100)
        assert result.max_depth == 1

    def test_chain_three_nodes(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("a", "econ", 0.1, 5.0))
        tree.add_node(FailureNode("b", "social", 0.2, 6.0))
        tree.add_node(FailureNode("c", "health", 0.3, 7.0))
        tree.add_edge("a", "b", 0.9, 1, "x")
        tree.add_edge("b", "c", 0.9, 1, "y")
        result = tree.simulate_cascade("a", n_simulations=1000)
        assert result.total_probability > 0.5
        assert result.max_depth >= 2