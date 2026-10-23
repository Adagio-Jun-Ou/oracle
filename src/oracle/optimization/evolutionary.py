import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from oracle.core.config import OracleConfig
from oracle.core.types import Domain
from oracle.prediction.engine import OracleEngine
from oracle.backtesting.engine import BacktestEngine


@dataclass
class Individual:
    weights: dict
    fitness: float = -float("inf")
    backtest_result: Optional[dict] = None


@dataclass
class OptimizationResult:
    best_weights: dict
    best_fitness: float
    best_mae: float
    best_directional_accuracy: float
    generations: int
    population_size: int
    fitness_history: list = field(default_factory=list)
    all_results: list = field(default_factory=list)


class EvolutionaryWeightOptimizer:
    def __init__(
        self,
        population_size: int = 30,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_fraction: float = 0.2,
        seed: int = 42,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.rng = np.random.default_rng(seed)
        self.weight_keys = [
            "deterministic", "stochastic", "ml_ensemble", "bayesian",
        ]

    def optimize(
        self,
        series: np.ndarray,
        metric: str = "mae",
        window_size: int = 60,
        n_windows: int = 5,
        verbose: bool = False,
    ) -> OptimizationResult:
        population = self._init_population()
        fitness_history = []
        all_results = []
        best_individual = None
        best_fitness = -float("inf")

        backtester = BacktestEngine()

        for gen in range(self.generations):
            for individual in population:
                if individual.fitness == -float("inf"):
                    fitness = self._evaluate_fitness(
                        individual.weights, series, metric, window_size, n_windows, backtester
                    )
                    individual.fitness = fitness
                    individual.backtest_result = backtester.backtest_series(
                        series[-n_windows:], series[-n_windows:]
                    ) if len(series) >= n_windows else None

            population.sort(key=lambda x: x.fitness, reverse=True)
            fitness_history.append(population[0].fitness)

            if population[0].fitness > best_fitness:
                best_fitness = population[0].fitness
                best_individual = Individual(
                    weights=dict(population[0].weights),
                    fitness=population[0].fitness,
                )

            if verbose and gen % 10 == 0:
                print(f"  Gen {gen}: best_fitness={best_fitness:.6f}, weights={population[0].weights}")

            n_elite = max(2, int(self.elite_fraction * self.population_size))
            new_population = [Individual(weights=dict(ind.weights), fitness=ind.fitness) for ind in population[:n_elite]]

            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                child.fitness = -float("inf")
                new_population.append(child)

            population = new_population

        if best_individual is None:
            best_individual = Individual(weights={k: 0.25 for k in self.weight_keys}, fitness=0.0)

        return OptimizationResult(
            best_weights=best_individual.weights,
            best_fitness=best_fitness,
            best_mae=1.0 / (best_fitness + 1e-10) if best_fitness > 0 else float("inf"),
            best_directional_accuracy=0.5,
            generations=self.generations,
            population_size=self.population_size,
            fitness_history=fitness_history,
            all_results=all_results,
        )

    def _init_population(self) -> list[Individual]:
        population = []
        defaults = Individual(
            weights={"deterministic": 0.30, "stochastic": 0.30, "ml_ensemble": 0.25, "bayesian": 0.15}
        )
        population.append(defaults)
        for _ in range(self.population_size - 1):
            raw = self.rng.dirichlet(np.ones(len(self.weight_keys)))
            weights = {k: float(v) for k, v in zip(self.weight_keys, raw)}
            population.append(Individual(weights=weights))
        return population

    def _evaluate_fitness(
        self, weights: dict, series: np.ndarray, metric: str,
        window_size: int, n_windows: int, backtester: BacktestEngine,
    ) -> float:
        config = OracleConfig(ensemble_weights=weights)
        engine = OracleEngine(config)
        engine.add_time_series("eval_series", series[:window_size], Domain.ECONOMICS)

        predictions = []
        actuals = []
        step_size = max(1, (len(series) - window_size) // n_windows)

        for i in range(min(n_windows, (len(series) - window_size) // step_size)):
            train_end = window_size + i * step_size
            test_start = train_end
            test_end = min(test_start + step_size, len(series))

            if test_end <= test_start:
                continue

            from oracle.deterministic.statistical_model import StatisticalDeterministicModel
            model = StatisticalDeterministicModel()
            train = series[i * step_size:train_end]
            test = series[test_start:test_end]

            if len(train) < 10 or len(test) < 1:
                continue

            try:
                pred = model.ensemble_predict(train, n_ahead=len(test))
                if metric == "mae":
                    actuals_arr = test
                    pred_arr = pred["predictions"][:len(test)] if len(pred["predictions"]) >= len(test) else np.full(len(test), pred["predictions"][-1])
                    min_len = min(len(actuals_arr), len(pred_arr))
                    error = np.mean(np.abs(actuals_arr[:min_len] - pred_arr[:min_len]))
                    fitness = 1.0 / (error + 1e-10)
                else:
                    error = np.mean(np.abs(test - np.mean(train)))
                    fitness = 1.0 / (error + 1e-10)
                predictions.append(fitness)
            except Exception:
                predictions.append(0.01)

        return float(np.mean(predictions)) if predictions else 0.01

    def _tournament_select(self, population: list[Individual], k: int = 3) -> Individual:
        contestants = list(self.rng.choice(len(population), min(k, len(population)), replace=False))
        return max([population[i] for i in contestants], key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        if self.rng.random() > self.crossover_rate:
            return Individual(weights=dict(parent1.weights))

        alpha = self.rng.random()
        child_weights = {}
        for key in self.weight_keys:
            w1 = parent1.weights.get(key, 0.25)
            w2 = parent2.weights.get(key, 0.25)
            child_weights[key] = alpha * w1 + (1 - alpha) * w2

        total = sum(child_weights.values())
        if total > 0:
            child_weights = {k: v / total for k, v in child_weights.items()}

        return Individual(weights=child_weights)

    def _mutate(self, individual: Individual) -> Individual:
        weights = dict(individual.weights)
        if self.rng.random() < self.mutation_rate:
            key = self.rng.choice(self.weight_keys)
            noise = self.rng.normal(0, 0.1)
            weights[key] = max(0.01, weights.get(key, 0.25) + noise)
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        return Individual(weights=weights)