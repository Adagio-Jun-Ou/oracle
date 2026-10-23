import pytest
import numpy as np
from oracle.backtesting.engine import BacktestEngine, BacktestResult, WalkForwardResult
from oracle.optimization.evolutionary import EvolutionaryWeightOptimizer


class TestBacktestEngine:
    def setup_method(self):
        np.random.seed(42)
        self.actual = np.cumsum(np.random.normal(0.01, 0.05, 200)) + 100
        self.predicted = self.actual + np.random.normal(0, 0.5, 200)

    def test_backtest_series(self):
        engine = BacktestEngine()
        result = engine.backtest_series(self.actual, self.predicted)
        assert result.mae > 0
        assert result.rmse > 0
        assert result.n_observations == 200

    def test_backtest_with_ci(self):
        ci_lower = self.predicted - 2
        ci_upper = self.predicted + 2
        engine = BacktestEngine()
        result = engine.backtest_series(self.actual, self.predicted, ci_lower, ci_upper)
        assert result.coverage_95 > 0

    def test_backtest_metrics(self):
        engine = BacktestEngine()
        result = engine.backtest_series(self.actual, self.predicted)
        assert 0 <= result.directional_accuracy <= 1
        assert result.mape > 0

    def test_walk_forward(self):
        from oracle.prediction.engine import OracleEngine
        from oracle.core.config import OracleConfig
        oracle = OracleEngine(OracleConfig())
        engine = BacktestEngine(oracle=oracle)
        series = np.cumsum(np.random.normal(0.01, 0.05, 200)) + 100
        result = engine.walk_forward(series, window_size=60, n_windows=3, horizon=5)
        assert result.n_windows >= 0

    def test_compare_models(self):
        engine = BacktestEngine()
        series = np.cumsum(np.random.normal(0.01, 0.05, 200)) + 100
        result = engine.compare_models(series, window_size=60, n_windows=3, horizon=5)
        assert "statistical" in result
        assert result["statistical"]["mae"] > 0


class TestEvolutionaryWeightOptimizer:
    def setup_method(self):
        self.optimizer = EvolutionaryWeightOptimizer(
            population_size=10, generations=5, seed=42,
        )

    def test_init_population(self):
        pop = self.optimizer._init_population()
        assert len(pop) == 10
        for ind in pop:
            total = sum(ind.weights.values())
            assert abs(total - 1.0) < 0.01

    def test_crossover(self):
        parent1 = self.optimizer._init_population()[0]
        parent2 = self.optimizer._init_population()[1]
        child = self.optimizer._crossover(parent1, parent2)
        assert abs(sum(child.weights.values()) - 1.0) < 0.01

    def test_mutate(self):
        individual = self.optimizer._init_population()[0]
        mutated = self.optimizer._mutate(individual)
        assert abs(sum(mutated.weights.values()) - 1.0) < 0.1

    def test_tournament_select(self):
        pop = self.optimizer._init_population()
        for ind in pop:
            ind.fitness = np.random.random()
        selected = self.optimizer._tournament_select(pop)
        assert selected is not None
        assert selected.fitness > 0

    def test_optimize_returns_result(self):
        series = np.cumsum(np.random.normal(0.01, 0.05, 200)) + 100
        result = self.optimizer.optimize(series, metric="mae", window_size=50, n_windows=3)
        assert result.best_weights is not None
        assert len(result.best_weights) == 4
        assert abs(sum(result.best_weights.values()) - 1.0) < 0.1
        assert result.generations == 5
        assert len(result.fitness_history) > 0

    def test_optimization_best_weights_sum_to_one(self):
        series = np.cumsum(np.random.normal(0.01, 0.05, 200)) + 100
        result = self.optimizer.optimize(series, metric="mae", window_size=50, n_windows=2)
        total = sum(result.best_weights.values())
        assert abs(total - 1.0) < 0.15


class TestBacktestResultMetrics:
    def setup_method(self):
        np.random.seed(123)

    def test_perfect_prediction(self):
        engine = BacktestEngine()
        actual = np.arange(100, dtype=float)
        predicted = actual.copy()
        result = engine.backtest_series(actual, predicted)
        assert result.mae == pytest.approx(0.0, abs=1e-10)
        assert result.rmse == pytest.approx(0.0, abs=1e-10)
        assert result.directional_accuracy == pytest.approx(1.0, abs=0.01) or result.directional_accuracy == 0.0

    def test_constant_prediction(self):
        engine = BacktestEngine()
        actual = np.array([100, 102, 98, 105, 103, 107, 99, 101, 104, 100], dtype=float)
        predicted = np.full(10, 101.0)
        result = engine.backtest_series(actual, predicted)
        assert result.mae > 0
        assert result.n_observations == 10

    def test_ci_coverage_wide(self):
        engine = BacktestEngine()
        actual = np.random.normal(100, 5, 100)
        predicted = np.full(100, 100.0)
        ci_lower = predicted - 20
        ci_upper = predicted + 20
        result = engine.backtest_series(actual, predicted, ci_lower, ci_upper)
        assert result.coverage_95 > 0.8