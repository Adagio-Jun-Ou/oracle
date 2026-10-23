from dataclasses import dataclass, field
from typing import Optional, Callable, List
import numpy as np


@dataclass
class AcquisitionResult:
    x_next: list
    acquisition_values: list
    best_acquisition: float
    method: str


@dataclass 
class OptimizationResult:
    best_x: list
    best_value: float
    n_iterations: int
    convergence_history: list
    acquisition_function: str
    surrogate_model: str


@dataclass
class HyperparameterResult:
    best_params: dict
    best_score: float
    all_trials: list
    param_importance: dict
    n_trials: int
    method: str


class BayesianOptimizer:
    def __init__(self, n_restarts: int = 5, seed: int = 42):
        self.n_restarts = n_restarts
        self.seed = seed

    @staticmethod
    def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0,
                    variance: float = 1.0) -> np.ndarray:
        sq_dist = np.sum(X1 ** 2, axis=1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2 ** 2, axis=1)
        return variance * np.exp(-0.5 * sq_dist / (length_scale ** 2))

    def _gp_predict(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, length_scale: float = 1.0,
                    noise: float = 1e-6) -> tuple:
        K = self._rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(len(X_train))
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        except np.linalg.LinAlgError:
            K += 1e-3 * np.eye(len(X_train))
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        K_s = self._rbf_kernel(X_train, X_test, length_scale)
        K_ss = self._rbf_kernel(X_test, X_test, length_scale)
        mu = K_s.T @ alpha
        v = np.linalg.solve(L, K_s)
        K_ss_diag = np.array([K_ss[i, i] for i in range(len(X_test))])
        sigma = np.sqrt(np.maximum(K_ss_diag - np.sum(v ** 2, axis=0), 1e-10))
        return mu, sigma

    def expected_improvement(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_candidates: np.ndarray, best_y: Optional[float] = None) -> AcquisitionResult:
        if best_y is None:
            best_y = float(np.max(y_train))
        mu, sigma = self._gp_predict(X_train, y_train, X_candidates)
        sigma = np.maximum(sigma, 1e-10)
        xi = 0.01
        z = (mu - best_y - xi) / sigma
        from scipy import stats as sp_stats
        ei = (mu - best_y - xi) * sp_stats.norm.cdf(z) + sigma * sp_stats.norm.pdf(z)
        ei[sigma < 1e-10] = 0.0
        best_idx = int(np.argmax(ei))
        return AcquisitionResult(
            x_next=X_candidates[best_idx].tolist(),
            acquisition_values=[float(v) for v in ei],
            best_acquisition=float(ei[best_idx]),
            method="expected_improvement",
        )

    def upper_confidence_bound(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_candidates: np.ndarray, beta: float = 2.0) -> AcquisitionResult:
        mu, sigma = self._gp_predict(X_train, y_train, X_candidates)
        ucb = mu + beta * sigma
        best_idx = int(np.argmax(ucb))
        return AcquisitionResult(
            x_next=X_candidates[best_idx].tolist(),
            acquisition_values=[float(v) for v in ucb],
            best_acquisition=float(ucb[best_idx]),
            method="ucb",
        )

    def probability_of_improvement(self, X_train: np.ndarray, y_train: np.ndarray,
                                    X_candidates: np.ndarray, best_y: Optional[float] = None) -> AcquisitionResult:
        if best_y is None:
            best_y = float(np.max(y_train))
        mu, sigma = self._gp_predict(X_train, y_train, X_candidates)
        sigma = np.maximum(sigma, 1e-10)
        from scipy import stats as sp_stats
        pi = sp_stats.norm.cdf((mu - best_y) / sigma)
        best_idx = int(np.argmax(pi))
        return AcquisitionResult(
            x_next=X_candidates[best_idx].tolist(),
            acquisition_values=[float(v) for v in pi],
            best_acquisition=float(pi[best_idx]),
            method="probability_of_improvement",
        )

    def optimize(self, objective: Callable, bounds: list, n_iterations: int = 20,
                  acquisition: str = "ei", seed: Optional[int] = None) -> OptimizationResult:
        rng = np.random.RandomState(seed or self.seed)
        dim = len(bounds)
        n_initial = min(5, max(2, dim + 1))
        X_observed = np.array([[rng.uniform(b[0], b[1]) for b in bounds] for _ in range(n_initial)])
        y_observed = np.array([objective(x) for x in X_observed])
        X_candidates = np.array([[rng.uniform(b[0], b[1]) for b in bounds] for _ in range(100)])
        convergence = [float(np.max(y_observed))]
        for i in range(n_iterations):
            if acquisition == "ei":
                acq_result = self.expected_improvement(X_observed, y_observed, X_candidates)
            elif acquisition == "ucb":
                acq_result = self.upper_confidence_bound(X_observed, y_observed, X_candidates)
            else:
                acq_result = self.probability_of_improvement(X_observed, y_observed, X_candidates)
            x_next = np.array(acq_result.x_next).reshape(1, -1)
            if x_next.shape[1] != dim:
                x_next = np.array([[np.mean(acq_result.x_next)] * dim])
            y_next = objective(x_next[0])
            X_observed = np.vstack([X_observed, x_next])
            y_observed = np.append(y_observed, y_next)
            convergence.append(float(np.max(y_observed)))
            X_candidates = np.array([[rng.uniform(b[0], b[1]) for b in bounds] for _ in range(100)])
        best_idx = int(np.argmax(y_observed))
        return OptimizationResult(
            best_x=X_observed[best_idx].tolist(),
            best_value=float(y_observed[best_idx]),
            n_iterations=n_iterations + n_initial,
            convergence_history=convergence,
            acquisition_function=acquisition,
            surrogate_model="gp_rbf",
        )

    def hyperparameter_optimization(self, param_space: dict, objective: Callable,
                                     n_trials: int = 30, method: str = "bayesian") -> HyperparameterResult:
        rng = np.random.RandomState(self.seed)
        param_names = list(param_space.keys())
        bounds = []
        for name in param_names:
            space = param_space[name]
            if isinstance(space, list):
                bounds.append((0, len(space) - 1))
            elif isinstance(space, tuple):
                bounds.append((space[0], space[1]))
            else:
                bounds.append((0, 1))
        X_trials = []
        y_trials = []
        all_trials = []
        for trial in range(n_trials):
            params = {}
            x_vec = []
            for i, name in enumerate(param_names):
                space = param_space[name]
                if isinstance(space, list):
                    idx = rng.randint(0, len(space))
                    params[name] = space[idx]
                    x_vec.append(idx / max(len(space) - 1, 1))
                elif isinstance(space, tuple):
                    val = rng.uniform(space[0], space[1])
                    params[name] = val
                    x_vec.append(val)
                else:
                    x_vec.append(rng.uniform(0, 1))
                    params[name] = space
            score = objective(params)
            X_trials.append(x_vec)
            y_trials.append(score)
            all_trials.append({"params": params, "score": score})
        X_trials = np.array(X_trials)
        y_trials = np.array(y_trials)
        best_idx = int(np.argmax(y_trials))
        param_importance = {}
        for i, name in enumerate(param_names):
            corr = abs(float(np.corrcoef(X_trials[:, i], y_trials)[0, 1]))
            param_importance[name] = max(0, min(1, corr))
        return HyperparameterResult(
            best_params=all_trials[best_idx]["params"],
            best_score=float(y_trials[best_idx]),
            all_trials=all_trials,
            param_importance=param_importance,
            n_trials=n_trials,
            method=method,
        )