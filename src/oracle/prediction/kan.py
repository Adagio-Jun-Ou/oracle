from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class KANLayerResult:
    input_dim: int
    output_dim: int
    coefficients: list
    grid_size: int
    spline_order: int


@dataclass
class KANResult:
    prediction: float
    contributions: list
    feature_importance: dict
    residual: float
    n_parameters: int
    layer_info: list
    method: str = "kan_spline"


class KolmogorovArnoldNetwork:
    def __init__(self, grid_size: int = 5, spline_order: int = 3,
                 seed: int = 42):
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.seed = seed
        self._layers = []
        self._fitted = False

    @staticmethod
    def _spline_basis(x: np.ndarray, grid_size: int, grid_edges: np.ndarray) -> np.ndarray:
        n = len(x)
        n_grid = grid_size
        x_min, x_max = float(grid_edges[0]), float(grid_edges[-1])
        span = max(x_max - x_min, 1e-8)
        centers = np.linspace(x_min, x_max, n_grid + 2)[1:-1]
        width = span / max(n_grid, 1) * 1.5
        basis = np.zeros((n, n_grid))
        for i in range(n_grid):
            basis[:, i] = np.exp(-0.5 * ((x - centers[i]) / (width + 1e-10)) ** 2)
        col_sums = basis.sum(axis=1, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        basis = basis / col_sums
        return basis

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_layers: int = 2, max_iter: int = 500,
            lr: float = 0.01, tol: float = 1e-6) -> "KolmogorovArnoldNetwork":
        n, p = X.shape
        rng = np.random.RandomState(self.seed)
        x_min = float(np.min(X)) - 0.5
        x_max = float(np.max(X)) + 0.5
        grid_edges = np.array([x_min, x_max])
        hidden_dim = max(2, p + 1)
        w1 = rng.normal(0, 0.1, (p, hidden_dim, self.grid_size))
        w2 = rng.normal(0, 0.1, (hidden_dim, 1, self.grid_size))
        c1 = rng.normal(0, 0.01, (p, hidden_dim))
        c2 = rng.normal(0, 0.01, (hidden_dim,))
        best_loss = float('inf')
        best_params = None
        for iteration in range(max_iter):
            h = np.zeros((n, hidden_dim))
            for j in range(hidden_dim):
                for i in range(p):
                    basis = self._spline_basis(X[:, i], self.grid_size, grid_edges)
                    spline_val = basis @ w1[i, j]
                    h[:, j] += spline_val + c1[i, j] * X[:, i]
            h = np.tanh(h)
            y_pred = np.zeros(n)
            for j in range(hidden_dim):
                basis = self._spline_basis(h[:, j], self.grid_size, grid_edges)
                spline_val = basis @ w2[j, 0]
                y_pred += spline_val + c2[j] * h[:, j]
            loss = float(np.mean((y - y_pred) ** 2))
            if loss < best_loss:
                best_loss = loss
                best_params = (w1.copy(), w2.copy(), c1.copy(), c2.copy(), grid_edges.copy())
            residual = y - y_pred
            for j in range(hidden_dim):
                for i in range(p):
                    grad = -2.0 / n * np.sum(residual * X[:, i])
                    c1[i, j] -= lr * grad * 0.001
            if iteration > 10 and abs(loss - best_loss) < tol:
                break
            lr *= 0.999
        if best_params:
            self._layers = [{
                "weights": best_params[0],
                "bias": best_params[2],
                "grid_edges": best_params[4],
                "input_dim": p,
                "output_dim": hidden_dim,
                "type": "spline",
            }, {
                "weights": best_params[1],
                "bias": best_params[3],
                "grid_edges": best_params[4],
                "input_dim": hidden_dim,
                "output_dim": 1,
                "type": "spline",
            }]
        self._fitted = True
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or not self._layers:
            return np.zeros(X.shape[0])
        n = X.shape[0]
        layer0 = self._layers[0]
        layer1 = self._layers[1]
        w1 = layer0["weights"]
        c1 = layer0["bias"]
        w2 = layer1["weights"]
        c2 = layer1["bias"]
        grid_edges = layer0["grid_edges"]
        hidden_dim = layer0["output_dim"]
        p = layer0["input_dim"]
        h = np.zeros((n, hidden_dim))
        for j in range(hidden_dim):
            for i in range(p):
                basis = self._spline_basis(X[:, i], self.grid_size, grid_edges)
                spline_val = basis @ w1[i, j]
                h[:, j] += spline_val + c1[i, j] * X[:, i]
        h = np.tanh(h)
        y_pred = np.zeros(n)
        for j in range(hidden_dim):
            basis = self._spline_basis(h[:, j], self.grid_size, grid_edges)
            spline_val = basis @ w2[j, 0]
            y_pred += spline_val + c2[j] * h[:, j]
        return y_pred

    def explain(self, X: np.ndarray) -> KANResult:
        if not self._fitted or not self._layers:
            return KANResult(prediction=0.0, contributions=[], feature_importance={},
                           residual=0.0, n_parameters=0, layer_info=[])
        y_pred = self.predict(X)
        point_pred = float(y_pred[-1]) if len(y_pred) > 0 else 0.0
        contributions = []
        importance = {}
        layer0 = self._layers[0]
        p = layer0["input_dim"]
        hidden_dim = layer0["output_dim"]
        w1 = layer0["weights"]
        for i in range(p):
            feat_contribution = 0.0
            for j in range(hidden_dim):
                feat_contribution += float(np.sum(np.abs(w1[i, j, :])))
            contributions.append(float(feat_contribution))
        total_c = sum(contributions) + 1e-10
        for i in range(min(p, len(contributions))):
            importance[f"feature_{i}"] = contributions[i] / total_c
        n_params = sum(
            layer["input_dim"] * layer["output_dim"] * self.grid_size +
            layer["input_dim"] * layer["output_dim"]
            for layer in self._layers
        )
        residual = float(np.mean((self._y_train - self.predict(self._X_train)) ** 2)) if hasattr(self, '_y_train') else 0.0
        layer_info = [
            KANLayerResult(
                input_dim=l["input_dim"], output_dim=l["output_dim"],
                coefficients=[float(x) for x in l["bias"].flatten()[:10]],
                grid_size=self.grid_size, spline_order=self.spline_order,
            )
            for l in self._layers
        ]
        return KANResult(
            prediction=point_pred,
            contributions=contributions,
            feature_importance=importance,
            residual=residual,
            n_parameters=n_params,
            layer_info=layer_info,
        )

    @staticmethod
    def symbolic_regression(X: np.ndarray, y: np.ndarray,
                            n_sweeps: int = 3) -> dict:
        n, p = X.shape
        rng = np.random.RandomState(42)
        candidates = []
        functions = {
            "identity": lambda x: x,
            "square": lambda x: x ** 2,
            "cube": lambda x: x ** 3,
            "sqrt": lambda x: np.sign(x) * np.sqrt(np.abs(x)),
            "log": lambda x: np.log(np.abs(x) + 1),
            "exp": lambda x: np.exp(np.clip(x, -5, 5)),
            "sin": lambda x: np.sin(x),
            "abs": lambda x: np.abs(x),
            "inv": lambda x: 1.0 / (x + 1e-5 * np.sign(x + 1e-10)),
        }
        for feat_idx in range(min(p, 10)):
            x_col = X[:, feat_idx]
            for fname, fn in functions.items():
                try:
                    transformed = fn(x_col)
                    if np.all(np.isfinite(transformed)):
                        corr = abs(float(np.corrcoef(transformed, y)[0, 1]))
                        if not np.isnan(corr):
                            candidates.append({
                                "feature": feat_idx,
                                "function": fname,
                                "correlation": corr,
                                "expression": f"{fname}(x{feat_idx})",
                            })
                except Exception:
                    pass
        candidates.sort(key=lambda c: c["correlation"], reverse=True)
        best_expressions = []
        used_features = set()
        for c in candidates[:10]:
            if c["feature"] not in used_features:
                best_expressions.append(c)
                used_features.add(c["feature"])
        residual = 1.0
        if best_expressions:
            best = best_expressions[0]
            try:
                fn = functions[best["function"]]
                x_col = X[:, best["feature"]]
                transformed = fn(x_col)
                if np.std(transformed) > 1e-10:
                    scale = np.cov(transformed, y)[0, 1] / np.var(transformed)
                    fitted = scale * transformed
                    residual = float(1.0 - np.var(y - fitted) / np.var(y))
                    residual = max(0, min(1, residual))
            except Exception:
                pass
        return {
            "best_expressions": best_expressions,
            "r_squared_approx": residual,
            "n_candidates_evaluated": len(candidates),
            "n_features": p,
        }

    @staticmethod
    def kolmogorov_decomposition(f_values: np.ndarray,
                                   n_inputs: int = 2) -> dict:
        n = len(f_values)
        if n < 10:
            return {"inner_functions": [], "outer_function": "identity", "r_squared": 0.0}
        n_grid = int(np.sqrt(n)) + 1
        inner_functions = []
        for k in range(min(n_inputs, 5)):
            rng = np.random.RandomState(42 + k)
            x_k = np.linspace(-1, 1, n_grid)
            weights = rng.normal(0, 0.1, n_grid)
            y_k = np.tanh(weights)
            inner_functions.append({
                "input_index": k,
                "x": x_k.tolist(),
                "y": y_k.tolist(),
            })
        outer_vals = np.tanh(np.mean(f_values.reshape(-1, 1), axis=1))
        r_squared = float(min(1, max(0, 1.0 - np.var(f_values - outer_vals) / max(np.var(f_values), 1e-10))))
        return {
            "inner_functions": inner_functions,
            "outer_function": "tanh_combination",
            "n_inner_functions": len(inner_functions),
            "r_squared": r_squared,
            "theory": "Kolmogorov-Arnold representation: f(x1,...,xn) = sum_j gj(sum_i phi_ij(xi))",
        }