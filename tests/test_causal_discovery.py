"""Tests for Causal Discovery module."""
import pytest
import numpy as np


class TestPCAlgorithm:
    def test_basic_chain(self):
        from oracle.nondeterministic.causal_discovery import PCAlgorithm
        np.random.seed(42)
        n = 500
        x1 = np.random.normal(0, 1, n)
        x2 = 0.8 * x1 + np.random.normal(0, 0.3, n)
        x3 = 0.7 * x2 + np.random.normal(0, 0.3, n)
        data = np.column_stack([x1, x2, x3])
        pc = PCAlgorithm(alpha=0.05, max_cond_set=2)
        result = pc.fit(data, ["X1", "X2", "X3"])
        assert result.n_edges >= 1
        assert result.method == "pc_algorithm"
        assert result.independence_tests > 0

    def test_fork_structure(self):
        from oracle.nondeterministic.causal_discovery import PCAlgorithm
        np.random.seed(42)
        n = 500
        x1 = np.random.normal(0, 1, n)
        x2 = 0.7 * x1 + np.random.normal(0, 0.3, n)
        x3 = 0.6 * x1 + np.random.normal(0, 0.4, n)
        data = np.column_stack([x1, x2, x3])
        pc = PCAlgorithm(alpha=0.05)
        result = pc.fit(data)
        assert result.n_edges >= 1
        assert result.graph.n_nodes == 3

    def test_independent_vars(self):
        from oracle.nondeterministic.causal_discovery import PCAlgorithm
        np.random.seed(42)
        n = 200
        data = np.random.normal(0, 1, (n, 3))
        pc = PCAlgorithm(alpha=0.01)
        result = pc.fit(data)
        assert result.n_edges == 0 or result.n_edges <= 2

    def test_alpha_sensitivity(self):
        from oracle.nondeterministic.causal_discovery import PCAlgorithm
        np.random.seed(42)
        n = 300
        x1 = np.random.normal(0, 1, n)
        x2 = 0.5 * x1 + np.random.normal(0, 0.5, n)
        x3 = 0.4 * x2 + np.random.normal(0, 0.5, n)
        data = np.column_stack([x1, x2, x3])
        result_lax = PCAlgorithm(alpha=0.2).fit(data)
        result_strict = PCAlgorithm(alpha=0.005).fit(data)
        assert result_lax.independence_tests > 0
        assert result_strict.independence_tests > 0


class TestGESAlgorithm:
    def test_basic_chain(self):
        from oracle.nondeterministic.causal_discovery import GESAlgorithm
        np.random.seed(42)
        n = 500
        x1 = np.random.normal(0, 1, n)
        x2 = 0.8 * x1 + np.random.normal(0, 0.3, n)
        x3 = 0.7 * x2 + np.random.normal(0, 0.3, n)
        data = np.column_stack([x1, x2, x3])
        ges = GESAlgorithm(penalty=1.0, max_parents=3)
        result = ges.fit(data, ["X1", "X2", "X3"])
        assert result.n_edges >= 1
        assert result.method == "ges_algorithm"
        assert result.n_directed >= 1

    def test_penalty_sensitivity(self):
        from oracle.nondeterministic.causal_discovery import GESAlgorithm
        np.random.seed(42)
        n = 300
        x1 = np.random.normal(0, 1, n)
        x2 = 0.5 * x1 + np.random.normal(0, 0.5, n)
        data = np.column_stack([x1, x2])
        result_low = GESAlgorithm(penalty=0.5).fit(data)
        result_high = GESAlgorithm(penalty=5.0).fit(data)
        assert result_low.n_edges >= result_high.n_edges

    def test_independent_vars(self):
        from oracle.nondeterministic.causal_discovery import GESAlgorithm
        np.random.seed(42)
        n = 200
        data = np.random.normal(0, 1, (n, 3))
        ges = GESAlgorithm(penalty=2.0)
        result = ges.fit(data)
        assert result.n_edges <= 1


class TestCausalGraph:
    def test_causal_graph_basic(self):
        from oracle.nondeterministic.causal_discovery import CausalGraph, CausalEdge
        edges = [
            CausalEdge(0, 1, "directed"),
            CausalEdge(1, 2, "directed"),
        ]
        graph = CausalGraph(n_nodes=3, edges=edges, adjacency=[[0,1,0],[0,0,1],[0,0,0]], node_names=["A","B","C"])
        assert graph.get_parents(2) == [1]
        assert graph.get_children(0) == [1]
        assert 1 in graph.get_neighbors(0)

    def test_is_dag(self):
        from oracle.nondeterministic.causal_discovery import CausalGraph, CausalEdge
        edges = [CausalEdge(0, 1, "directed"), CausalEdge(1, 2, "directed")]
        graph = CausalGraph(n_nodes=3, edges=edges, adjacency=[[0,1,0],[0,0,1],[0,0,0]], node_names=["A","B","C"])
        assert graph.is_dag()

    def test_is_not_dag(self):
        from oracle.nondeterministic.causal_discovery import CausalGraph, CausalEdge
        edges = [CausalEdge(0, 1, "directed"), CausalEdge(1, 0, "directed")]
        graph = CausalGraph(n_nodes=2, edges=edges, adjacency=[[0,1],[1,0]], node_names=["A","B"])
        assert not graph.is_dag()

    def test_topological_sort(self):
        from oracle.nondeterministic.causal_discovery import CausalGraph, CausalEdge
        edges = [CausalEdge(0, 1, "directed"), CausalEdge(1, 2, "directed")]
        graph = CausalGraph(n_nodes=3, edges=edges, adjacency=[[0,1,0],[0,0,1],[0,0,0]], node_names=["A","B","C"])
        order = graph.topological_sort()
        assert order.index(0) < order.index(1)
        assert order.index(1) < order.index(2)

    def test_adjacency_matrix(self):
        from oracle.nondeterministic.causal_discovery import CausalGraph, CausalEdge
        edges = [CausalEdge(0, 1, "directed")]
        graph = CausalGraph(n_nodes=2, edges=edges, adjacency=[[0,1],[0,0]], node_names=["A","B"])
        adj = graph.to_adjacency_matrix()
        assert adj[0][1] == 1
        assert adj[1][0] == 0


class TestCausalDiscoveryEngine:
    def test_compare_algorithms(self):
        from oracle.nondeterministic.causal_discovery import CausalDiscoveryEngine
        np.random.seed(42)
        n = 300
        x1 = np.random.normal(0, 1, n)
        x2 = 0.7 * x1 + np.random.normal(0, 0.3, n)
        x3 = 0.5 * x2 + np.random.normal(0, 0.4, n)
        data = np.column_stack([x1, x2, x3])
        engine = CausalDiscoveryEngine(alpha=0.05)
        result = engine.compare_algorithms(data)
        assert "pc_result" in result
        assert "ges_result" in result
        assert "agreement" in result
        assert 0 <= result["agreement"]["agreement_rate"] <= 1

    def test_pc_via_engine(self):
        from oracle.nondeterministic.causal_discovery import CausalDiscoveryEngine
        np.random.seed(42)
        n = 300
        x1 = np.random.normal(0, 1, n)
        x2 = 0.6 * x1 + np.random.normal(0, 0.4, n)
        data = np.column_stack([x1, x2])
        engine = CausalDiscoveryEngine(algorithm="pc")
        result = engine.fit(data)
        assert result.method == "pc_algorithm"

    def test_ges_via_engine(self):
        from oracle.nondeterministic.causal_discovery import CausalDiscoveryEngine
        np.random.seed(42)
        n = 300
        x1 = np.random.normal(0, 1, n)
        x2 = 0.6 * x1 + np.random.normal(0, 0.4, n)
        data = np.column_stack([x1, x2])
        engine = CausalDiscoveryEngine(algorithm="ges")
        result = engine.fit(data)
        assert result.method == "ges_algorithm"

    def test_unknown_algorithm(self):
        from oracle.nondeterministic.causal_discovery import CausalDiscoveryEngine
        engine = CausalDiscoveryEngine(algorithm="unknown")
        with pytest.raises(ValueError):
            engine.fit(np.random.normal(0, 1, (100, 2)))