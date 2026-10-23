from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class GPPrediction:
    mean: list
    std: list
    confidence_lower: list
    confidence_upper: list
    kernel_params: dict
    n_training_points: int


@dataclass
class KernelParams:
    length_scale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-6
    period: float = 1.0


class GaussianProcessPredictor:
    def __init__(self, kernel: str = "rbf", length_scale: float = 1.0,
                 variance: float = 1.0, noise: float = 1e-6, seed: int = 42):
        self.kernel = kernel
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise
        self.seed = seed
        self._X_train = None
        self._y_train = None
        self._L = None
        self._alpha = None

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel == "rbf":
            sq_dist = np.sum(X1 ** 2, axis=1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2 ** 2, axis=1)
            return self.variance * np.exp(-0.5 * sq_dist / (self.length_scale ** 2))
        elif self.kernel == "matern":
            from scipy.special import gamma as scipy_gamma, kv
            sq_dist = np.sum(X1 ** 2, axis=1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2 ** 2, axis=1)
            sqrt_dist = np.sqrt(np.maximum(sq_dist, 1e-10))
            nu = 1.5
            scaled_dist = np.minimum(sqrt_dist * np.sqrt(2 * nu) / self.length_scale, 100)
            K = (2 ** (1 - nu) / scipy_gamma(nu)) * (scaled_dist ** nu) * kv(nu, scaled_dist)
            K[~np.isfinite(K)] = 0.0
            K = self.variance * K + 1e-8 * np.eye(K.shape[0]) if K.shape[0] == K.shape[1] else self.variance * K
            return np.nan_to_num(K, nan=self.variance, posinf=self.variance, neginf=0.0)
        elif self.kernel == "periodic":
            sq_dist = np.sum(X1 ** 2, axis=1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2 ** 2, axis=1)
            dist = np.sqrt(sq_dist + 1e-10)
            return self.variance * np.exp(-2 * np.sin(np.pi * dist / self.length_scale) ** 2)
        else:
            sq_dist = np.sum(X1 ** 2, axis=1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2 ** 2, axis=1)
            return self.variance * np.exp(-0.5 * sq_dist / (self.length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessPredictor":
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._X_train = X.copy()
        self._y_train = y.copy()
        K = self._compute_kernel(X, X) + self.noise * np.eye(len(X))
        try:
            self._L = np.linalg.cholesky(K)
            self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        except np.linalg.LinAlgError:
            K += 1e-3 * np.eye(len(X))
            self._L = np.linalg.cholesky(K)
            self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        return self

    def predict(self, X_new: np.ndarray, return_std: bool = True) -> GPPrediction:
        if self._X_train is None:
            return GPPrediction(mean=[], std=[], confidence_lower=[], confidence_upper=[],
                               kernel_params={}, n_training_points=0)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        K_s = self._compute_kernel(self._X_train, X_new)
        mu = K_s.T @ self._alpha
        v = np.linalg.solve(self._L, K_s)
        K_ss_diag = self.variance * np.ones(len(X_new))
        sigma = np.sqrt(np.maximum(K_ss_diag - np.sum(v ** 2, axis=0), 1e-10))
        return GPPrediction(
            mean=[float(m) for m in mu],
            std=[float(s) for s in sigma],
            confidence_lower=[float(m - 1.96 * s) for m, s in zip(mu, sigma)],
            confidence_upper=[float(m + 1.96 * s) for m, s in zip(mu, sigma)],
            kernel_params={"kernel": self.kernel, "length_scale": self.length_scale,
                          "variance": self.variance, "noise": self.noise},
            n_training_points=len(self._y_train),
        )

    def optimize_kernel(self, X: np.ndarray, y: np.ndarray, n_iter: int = 10) -> dict:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        best_ll = -np.inf
        best_ls = self.length_scale
        best_var = self.variance
        best_noise = self.noise
        rng = np.random.RandomState(self.seed)
        ls_candidates = np.logspace(-1, 1, 5)
        var_candidates = np.logspace(-1, 1, 5)
        for ls in ls_candidates:
            for var in var_candidates:
                self.length_scale = ls
                self.variance = var
                self.noise = 1e-4
                K = self._compute_kernel(X, X) + self.noise * np.eye(len(X))
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    ll = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(y) * np.log(2 * np.pi)
                    if ll > best_ll:
                        best_ll = ll
                        best_ls = ls
                        best_var = var
                        best_noise = 1e-4
                except np.linalg.LinAlgError:
                    continue
        self.length_scale = best_ls
        self.variance = best_var
        self.noise = best_noise
        return {"best_length_scale": best_ls, "best_variance": best_var,
                "best_noise": best_noise, "log_likelihood": float(best_ll)}

    @staticmethod
    def spatial_interpolation(coords: np.ndarray, values: np.ndarray,
                               grid_resolution: int = 20) -> dict:
        min_x, max_x = float(coords[:, 0].min()), float(coords[:, 0].max())
        min_y, max_y = float(coords[:, 1].min()), float(coords[:, 1].max())
        margin = 0.1 * max(max_x - min_x, max_y - min_y)
        x_grid = np.linspace(min_x - margin, max_x + margin, grid_resolution)
        y_grid = np.linspace(min_y - margin, max_y + margin, grid_resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        ls = max(max_x - min_x, max_y - min_y) / 3 + 1e-10
        gp = GaussianProcessPredictor(kernel="rbf", length_scale=ls)
        gp.fit(coords, values)
        pred = gp.predict(grid_points)
        z_grid = np.array(pred.mean).reshape(grid_resolution, grid_resolution)
        std_grid = np.array(pred.std).reshape(grid_resolution, grid_resolution)
        return {
            "x_grid": x_grid.tolist(),
            "y_grid": y_grid.tolist(),
            "z_grid": z_grid.tolist(),
            "std_grid": std_grid.tolist(),
            "mean_prediction": float(np.mean(values)),
            "max_prediction": float(np.max(pred.mean)),
            "min_prediction": float(np.min(pred.mean)),
        }

    @staticmethod
    def temporal_forecast(times: np.ndarray, values: np.ndarray,
                          n_ahead: int = 10, kernel: str = "rbf") -> dict:
        if times.ndim == 1:
            X = times.reshape(-1, 1)
        else:
            X = times
        t_range = float(X[-1, 0] - X[0, 0])
        ls = max(t_range / 5, 1e-10)
        gp = GaussianProcessPredictor(kernel=kernel, length_scale=ls)
        gp.fit(X, values)
        last_t = float(X[-1, 0])
        dt = float(np.mean(np.diff(X[:, 0]))) if len(X) > 1 else 1.0
        future_X = np.array([[last_t + (i + 1) * dt] for i in range(n_ahead)])
        pred = gp.predict(future_X)
        return {
            "forecast_mean": pred.mean,
            "forecast_std": pred.std,
            "forecast_lower": pred.confidence_lower,
            "forecast_upper": pred.confidence_upper,
            "horizon": n_ahead,
            "kernel": kernel,
        }