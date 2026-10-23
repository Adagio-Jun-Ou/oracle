"""Tests for Koopman, ABM, and Symbolic Regression modules."""
import pytest
import numpy as np


class TestDMD:
    def test_dmd_init(self):
        from oracle.prediction.koopman import DynamicModeDecomposition
        dmd = DynamicModeDecomposition()
        assert not dmd._fitted

    def test_dmd_fit_1d(self):
        from oracle.prediction.koopman import DynamicModeDecomposition
        np.random.seed(42)
        t = np.linspace(0, 10, 50)
        X = np.array([np.sin(t), np.cos(t)])
        dmd = DynamicModeDecomposition()
        dmd.fit(X)
        assert dmd._fitted
        assert dmd.modes is not None
        assert dmd.eigenvalues is not None

    def test_dmd_get_result(self):
        from oracle.prediction.koopman import DynamicModeDecomposition
        np.random.seed(42)
        t = np.linspace(0, 10, 50)
        X = np.array([np.sin(t), np.cos(t)])
        dmd = DynamicModeDecomposition()
        dmd.fit(X)
        result = dmd.get_result()
        assert result.n_modes > 0
        assert len(result.eigenvalues) > 0
        assert result.method == "exact_dmd"

    def test_dmd_forecast(self):
        from oracle.prediction.koopman import DynamicModeDecomposition
        np.random.seed(42)
        t = np.linspace(0, 10, 50)
        X = np.array([np.sin(t), np.cos(t)])
        dmd = DynamicModeDecomposition()
        dmd.fit(X)
        forecast = dmd.forecast(n_ahead=5)
        assert forecast.shape[0] == 2
        assert forecast.shape[1] == 5

    def test_dmd_classify_modes(self):
        from oracle.prediction.koopman import DynamicModeDecomposition
        np.random.seed(42)
        X = np.random.randn(3, 30)
        dmd = DynamicModeDecomposition()
        dmd.fit(X)
        modes = dmd.classify_modes()
        assert "stable" in modes
        assert "unstable" in modes
        assert "marginal" in modes

    def test_dmd_insufficient_data(self):
        from oracle.prediction.koopman import DynamicModeDecomposition
        dmd = DynamicModeDecomposition()
        with pytest.raises(ValueError):
            dmd.fit(np.ones((2, 2)))

    def test_dmd_not_fitted_raises(self):
        from oracle.prediction.koopman import DynamicModeDecomposition
        dmd = DynamicModeDecomposition()
        with pytest.raises(ValueError):
            dmd.forecast(n_ahead=5)


class TestExtendedDMD:
    def test_edmd_init(self):
        from oracle.prediction.koopman import ExtendedDMD
        edmd = ExtendedDMD(poly_degree=2)
        assert not edmd._fitted

    def test_edmd_fit(self):
        from oracle.prediction.koopman import ExtendedDMD
        np.random.seed(42)
        t = np.linspace(0, 5, 40)
        X = np.array([np.sin(t), np.cos(t)])
        edmd = ExtendedDMD(poly_degree=2)
        edmd.fit(X)
        assert edmd._fitted

    def test_edmd_get_result(self):
        from oracle.prediction.koopman import ExtendedDMD
        np.random.seed(42)
        X = np.random.randn(2, 40)
        edmd = ExtendedDMD(poly_degree=2)
        edmd.fit(X)
        result = edmd.get_result()
        assert result.n_observables > 2
        assert result.method == "koopman_edmd"

    def test_edmd_forecast(self):
        from oracle.prediction.koopman import ExtendedDMD
        np.random.seed(42)
        X = np.random.randn(2, 40)
        edmd = ExtendedDMD(poly_degree=2)
        edmd.fit(X)
        fc = edmd.forecast(X, n_ahead=5)
        assert fc.shape[1] == 5

    def test_edmd_koopman_modes(self):
        from oracle.prediction.koopman import ExtendedDMD
        np.random.seed(42)
        X = np.random.randn(2, 40)
        edmd = ExtendedDMD(poly_degree=2)
        edmd.fit(X)
        modes = edmd.koopman_modes()
        assert "modes" in modes
        assert modes["n_modes"] > 0


class TestKoopmanOperator:
    def test_koopman_dmd(self):
        from oracle.prediction.koopman import KoopmanOperator
        np.random.seed(42)
        t = np.linspace(0, 10, 50)
        X = np.array([np.sin(t), np.cos(t)])
        ko = KoopmanOperator(method="dmd")
        ko.fit(X)
        result = ko.forecast(n_ahead=5)
        assert "forecast" in result
        assert result["method"] == "koopman_dmd"

    def test_koopman_edmd(self):
        from oracle.prediction.koopman import KoopmanOperator
        np.random.seed(42)
        X = np.random.randn(2, 40)
        ko = KoopmanOperator(method="edmd")
        ko.fit(X)
        result = ko.forecast(n_ahead=5)
        assert "forecast" in result
        assert result["method"] == "koopman_edmd"

    def test_koopman_stability(self):
        from oracle.prediction.koopman import KoopmanOperator
        np.random.seed(42)
        X = np.random.randn(2, 40)
        ko = KoopmanOperator(method="dmd")
        ko.fit(X)
        stability = ko.analyze_stability()
        assert "n_stable_modes" in stability


class TestABM:
    def test_abm_market(self):
        from oracle.prediction.agent_based import AgentBasedModel
        abm = AgentBasedModel(n_agents=30, seed=42)
        result = abm.simulate_market(n_steps=20)
        assert "price" in result.time_series
        assert len(result.time_series["price"]) == 21
        assert result.method in ("abm_market", "abm")
        assert result.n_agents == 30

    def test_abm_epidemic(self):
        from oracle.prediction.agent_based import AgentBasedModel
        abm = AgentBasedModel(n_agents=50, seed=42)
        result = abm.simulate_epidemic(n_steps=30)
        assert "susceptible" in result.time_series
        assert "infected" in result.time_series
        assert "recovered" in result.time_series
        assert result.n_agents == 50

    def test_abm_opinion(self):
        from oracle.prediction.agent_based import AgentBasedModel
        abm = AgentBasedModel(n_agents=40, seed=42)
        result = abm.simulate_opinion(n_steps=30)
        assert "polarization" in result.time_series
        assert "consensus" in result.time_series
        assert result.n_agents == 40

    def test_abm_network_types(self):
        from oracle.prediction.agent_based import AgentBasedModel
        for net_type in ["random", "scale_free", "small_world"]:
            abm = AgentBasedModel(n_agents=30, network_type=net_type, seed=42)
            abm._init_agents("market")
            adj = abm._build_network()
            assert adj.shape == (30, 30)
            assert np.allclose(adj, adj.T)

    def test_abm_result_dataclass(self):
        from oracle.prediction.agent_based import SimulationResult
        result = SimulationResult(
            time_series={"x": [1.0, 2.0]},
            agent_states=[0.5, 0.3],
            agent_groups=[0, 1],
            statistics={"mean": 0.4},
            n_steps=2,
            n_agents=2,
            emergent_patterns=["test"],
        )
        assert result.method in ("abm_market", "abm")

    def test_abm_agent_dataclass(self):
        from oracle.prediction.agent_based import Agent
        agent = Agent(id=0, state=0.5, wealth=100.0)
        assert agent.id == 0
        assert agent.wealth == 100.0

    def test_abm_patterns(self):
        from oracle.prediction.agent_based import AgentBasedModel
        abm = AgentBasedModel(n_agents=50, seed=42)
        result = abm.simulate_market(n_steps=50)
        assert isinstance(result.emergent_patterns, list)


class TestSymbolicRegression:
    def test_sr_init(self):
        from oracle.prediction.symbolic_regression import SymbolicRegressor
        sr = SymbolicRegressor(population_size=20, n_generations=5)
        assert sr.population_size == 20
        assert sr.n_generations == 5

    def test_sr_fit_linear(self):
        from oracle.prediction.symbolic_regression import SymbolicRegressor
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 2.0 * X[:, 0] + 1.0
        sr = SymbolicRegressor(population_size=30, n_generations=10, max_depth=3, seed=42)
        sr.fit(X, y)
        assert sr.best_individual is not None
        result = sr.get_result()
        assert result.best_expression != ""
        assert result.best_r_squared > -1

    def test_sr_fit_nonlinear(self):
        from oracle.prediction.symbolic_regression import SymbolicRegressor
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = np.sin(X[:, 0]) + 0.5
        sr = SymbolicRegressor(population_size=30, n_generations=10, max_depth=4, seed=42)
        sr.fit(X, y)
        result = sr.get_result()
        assert result.best_rmse > 0
        assert result.pareto_front is not None

    def test_sr_predict(self):
        from oracle.prediction.symbolic_regression import SymbolicRegressor
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = X[:, 0] * 2
        sr = SymbolicRegressor(population_size=20, n_generations=5, seed=42)
        sr.fit(X, y)
        pred = sr.predict(X)
        assert len(pred) == 50

    def test_sr_not_fitted_raises(self):
        from oracle.prediction.symbolic_regression import SymbolicRegressor
        sr = SymbolicRegressor()
        with pytest.raises(ValueError):
            sr.predict(np.ones((10, 1)))
        with pytest.raises(ValueError):
            sr.get_result()

    def test_node_operations(self):
        from oracle.prediction.symbolic_regression import Node
        const = Node(value=3.0)
        var = Node(var_idx=0)
        add = Node(func_name='add', children=[const, var])
        X = np.array([[1.0], [2.0], [3.0]])
        result = add.evaluate(X)
        assert len(result) == 3
        assert abs(result[0] - 4.0) < 0.01

    def test_node_complexity(self):
        from oracle.prediction.symbolic_regression import Node
        const = Node(value=1.0)
        var = Node(var_idx=0)
        add = Node(func_name='add', children=[const, var])
        assert add.complexity() == 3

    def test_node_str(self):
        from oracle.prediction.symbolic_regression import Node
        const = Node(value=2.0)
        var = Node(var_idx=0)
        add = Node(func_name='add', children=[const, var])
        s = str(add)
        assert '+' in s or 'add' in s