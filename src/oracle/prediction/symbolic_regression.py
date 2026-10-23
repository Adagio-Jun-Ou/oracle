"""Symbolic Regression module for interpretable equation discovery.

Implements:
- Genetic programming-based symbolic regression
- Pareto front optimization (complexity vs accuracy)
- Expression simplification and evaluation
- Built-in function set ( arithmetic, trig, exp, log)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import math
import copy


@dataclass
class ExpressionResult:
    expression: str
    fitness: float
    complexity: int
    r_squared: float
    rmse: float
    mae: float
    method: str = "symbolic_regression"


@dataclass
class SymbolicResult:
    best_expression: str
    best_fitness: float
    best_r_squared: float
    best_rmse: float
    pareto_front: List[Dict]
    n_generations: int
    population_size: int
    method: str = "genetic_symbolic"


FUNCTIONS = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / (y + 1e-10 * np.sign(y + 1e-20)),
    'sin': lambda x: np.sin(x),
    'cos': lambda x: np.cos(x),
    'exp': lambda x: np.exp(np.clip(x, -20, 20)),
    'log': lambda x: np.log(np.abs(x) + 1e-10),
    'sqrt': lambda x: np.sqrt(np.abs(x)),
    'abs': lambda x: np.abs(x),
    'neg': lambda x: -x,
    'square': lambda x: x ** 2,
    'cube': lambda x: x ** 3,
}


class Node:
    def __init__(self, func_name: str = None, value: float = None,
                 var_idx: int = None, children: list = None):
        self.func_name = func_name
        self.value = value
        self.var_idx = var_idx
        self.children = children or []
        self.is_terminal = func_name is None and var_idx is None or (func_name is None and value is not None)
        self.is_variable = var_idx is not None
        self.is_constant = value is not None and var_idx is None

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.is_constant:
            return np.full(X.shape[0] if X.ndim > 1 else len(X), self.value)
        if self.is_variable:
            if X.ndim == 1:
                return X.copy()
            return X[:, self.var_idx].copy()
        func = FUNCTIONS.get(self.func_name)
        if func is None:
            return np.zeros(X.shape[0] if X.ndim > 1 else len(X))
        arity = len(self.children)
        if arity == 1:
            return func(self.children[0].evaluate(X))
        elif arity == 2:
            left = self.children[0].evaluate(X)
            right = self.children[1].evaluate(X)
            return func(left, right)
        return np.zeros(X.shape[0] if X.ndim > 1 else len(X))

    def complexity(self) -> int:
        if self.is_constant or self.is_variable:
            return 1
        return 1 + sum(c.complexity() for c in self.children)

    def __str__(self) -> str:
        if self.is_constant:
            return f"{self.value:.4f}" if abs(self.value) != int(abs(self.value)) else f"{int(self.value)}"
        if self.is_variable:
            return f"x{self.var_idx}" if self.var_idx is not None else "x"
        arity = len(self.children)
        if arity == 1:
            return f"{self.func_name}({self.children[0]})"
        elif arity == 2:
            op_map = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
            op = op_map.get(self.func_name, self.func_name)
            return f"({self.children[0]} {op} {self.children[1]})"
        return self.func_name

    def copy(self) -> 'Node':
        return copy.deepcopy(self)


class SymbolicRegressor:
    """Genetic programming symbolic regression.

    Evolves expression trees to discover interpretable mathematical
    relationships between variables. Supports Pareto front optimization
    balancing accuracy vs complexity.
    """

    def __init__(self, population_size: int = 100, n_generations: int = 50,
                tournament_size: int = 5, crossover_rate: float = 0.7,
                 mutation_rate: float = 0.1, max_depth: int = 5,
                 parsimony_coefficient: float = 0.01, seed: int = 42):
        self.population_size = population_size
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_individual: Optional[Node] = None
        self.best_fitness: float = float('inf')
        self.pareto_front: List[Dict] = []

    def _random_terminal(self, n_vars: int) -> Node:
        if self.rng.random() < 0.5 and n_vars > 0:
            return Node(var_idx=self.rng.randint(0, n_vars))
        return Node(value=self.rng.uniform(-5, 5))

    def _random_function(self, depth: int, n_vars: int) -> Node:
        if depth >= self.max_depth:
            return self._random_terminal(n_vars)
        if self.rng.random() < 0.3:
            return self._random_terminal(n_vars)
        unary = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'neg', 'square']
        binary = ['add', 'sub', 'mul', 'div']
        if self.rng.random() < 0.5:
            func = self.rng.choice(unary)
            child = self._random_function(depth + 1, n_vars)
            return Node(func_name=func, children=[child])
        else:
            func = self.rng.choice(binary)
            left = self._random_function(depth + 1, n_vars)
            right = self._random_function(depth + 1, n_vars)
            return Node(func_name=func, children=[left, right])

    def _evaluate_fitness(self, tree: Node, X: np.ndarray, y: np.ndarray) -> float:
        try:
            pred = tree.evaluate(X)
            if len(pred) != len(y):
                return float('inf')
            if not np.all(np.isfinite(pred)):
                return float('inf')
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            penalty = self.parsimony_coefficient * tree.complexity()
            return rmse + penalty
        except Exception:
            return float('inf')

    def _tournament(self, population: List[Node], fitnesses: List[float]) -> int:
        indices = self.rng.choice(len(population), min(self.tournament_size, len(population)), replace=False)
        best_idx = indices[0]
        for idx in indices[1:]:
            if fitnesses[idx] < fitnesses[best_idx]:
                best_idx = idx
        return best_idx

    def _crossover(self, parent1: Node, parent2: Node) -> Node:
        child = parent1.copy()
        nodes1 = self._get_all_nodes(child)
        nodes2 = self._get_all_nodes(parent2)
        if not nodes1 or not nodes2:
            return child
        node1 = self.rng.choice(nodes1)
        node2 = self.rng.choice(nodes2)
        replacement = node2.copy()
        if node1 is child:
            return replacement
        parent_found = self._replace_subtree(child, node1, replacement)
        return child

    def _replace_subtree(self, tree: Node, target: Node, replacement: Node) -> bool:
        for i, child in enumerate(tree.children):
            if child is target:
                tree.children[i] = replacement
                return True
            if self._replace_subtree(child, target, replacement):
                return True
        return False

    def _mutate(self, tree: Node, n_vars: int) -> Node:
        nodes = self._get_all_nodes(tree)
        if not nodes:
            return tree
        node = self.rng.choice(nodes)
        new_subtree = self._random_function(0, n_vars)
        if node is tree:
            return new_subtree
        self._replace_subtree(tree, node, new_subtree)
        return tree

    def _get_all_nodes(self, tree: Node) -> List[Node]:
        nodes = [tree]
        for child in tree.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SymbolicRegressor':
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_vars = X.shape[1]
        population = [self._random_function(0, n_vars) for _ in range(self.population_size)]
        fitnesses = [self._evaluate_fitness(ind, X, y) for ind in population]
        best_idx = int(np.argmin(fitnesses))
        self.best_individual = population[best_idx].copy()
        self.best_fitness = fitnesses[best_idx]
        pareto_set = {(ind.complexity(), fitnesses[i]): {'expression': str(ind), 'fitness': fitnesses[i], 'complexity': ind.complexity()}
                     for i, ind in enumerate(population) if np.isfinite(fitnesses[i])}
        for gen in range(self.n_generations):
            new_population = []
            elite_idx = int(np.argmin(fitnesses))
            new_population.append(population[elite_idx].copy())
            while len(new_population) < self.population_size:
                if self.rng.random() < self.crossover_rate:
                    p1_idx = self._tournament(population, fitnesses)
                    p2_idx = self._tournament(population, fitnesses)
                    child = self._crossover(population[p1_idx], population[p2_idx])
                else:
                    p_idx = self._tournament(population, fitnesses)
                    child = population[p_idx].copy()
                if self.rng.random() < self.mutation_rate:
                    child = self._mutate(child, n_vars)
                new_population.append(child)
            population = new_population
            fitnesses = [self._evaluate_fitness(ind, X, y) for ind in population]
            valid_fitnesses = [f for f in fitnesses if np.isfinite(f)]
            if not valid_fitnesses:
                continue
            best_gen_idx = int(np.argmin(fitnesses))
            if fitnesses[best_gen_idx] < self.best_fitness:
                self.best_individual = population[best_gen_idx].copy()
                self.best_fitness = fitnesses[best_gen_idx]
            for i, ind in enumerate(population):
                if np.isfinite(fitnesses[i]):
                    key = (ind.complexity(), fitnesses[i])
                    pareto_set[key] = {'expression': str(ind), 'fitness': fitnesses[i], 'complexity': ind.complexity()}
        self.pareto_front = sorted(pareto_set.values(), key=lambda x: x['complexity'])[:20]
        self._X = X
        self._y = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_individual is None:
            raise ValueError("Model not fitted")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.best_individual.evaluate(X)

    def get_result(self) -> SymbolicResult:
        if self.best_individual is None:
            raise ValueError("Model not fitted")
        pred = self.best_individual.evaluate(self._X)
        y = self._y
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1 - ss_res / max(ss_tot, 1e-10) if ss_tot > 1e-10 else 0.0
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        return SymbolicResult(
            best_expression=str(self.best_individual),
            best_fitness=self.best_fitness,
            best_r_squared=r_squared,
            best_rmse=rmse,
            pareto_front=self.pareto_front[:10],
            n_generations=self.n_generations,
            population_size=self.population_size,
        )

    def get_expression_result(self) -> ExpressionResult:
        result = self.get_result()
        mae = float(np.mean(np.abs(self._y - self.best_individual.evaluate(self._X))))
        return ExpressionResult(
            expression=result.best_expression,
            fitness=result.best_fitness,
            complexity=result.pareto_front[0]['complexity'] if result.pareto_front else self.best_individual.complexity(),
            r_squared=result.best_r_squared,
            rmse=result.best_rmse,
            mae=mae,
        )