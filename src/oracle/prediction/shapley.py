from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from itertools import combinations
import warnings


@dataclass
class ShapleyValue:
    feature: str
    value: float
    abs_value: float
    contribution_pct: float


@dataclass
class ShapleyResult:
    shapley_values: list
    base_value: float
    prediction: float
    feature_importance: dict
    interaction_effects: dict
    top_features: list
    explanation: str
    confidence: float


class ShapleyAttribution:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def explain(self, predict_fn, x: np.ndarray, feature_names: list[str],
                baseline: Optional[np.ndarray] = None,
                n_samples: int = 1000,
                n_features_max: int = 15) -> ShapleyResult:
        n_features = len(feature_names)
        if baseline is None:
            baseline = np.zeros(n_features) if x.ndim == 1 else np.zeros(x.shape[1])
        if x.ndim > 1:
            x = x.flatten()
        if baseline.ndim > 1:
            baseline = baseline.flatten()
        n_features = min(n_features, len(feature_names), n_features_max)
        rng = np.random.RandomState(self.seed)
        base_result = predict_fn(baseline.reshape(1, -1))
        base_value = float(base_result[0] if hasattr(base_result, '__len__') else base_result)
        x_result = predict_fn(x.reshape(1, -1))
        prediction = float(x_result[0] if hasattr(x_result, '__len__') else x_result)
        shapley = self._compute_shapley_kernel(predict_fn, x, baseline, feature_names[:n_features],
                                                n_samples, rng)
        total_abs = sum(abs(v) for v in shapley.values()) + 1e-10
        shapley_values = []
        for fname, val in zip(feature_names[:n_features], shapley.values()):
            shapley_values.append(ShapleyValue(
                feature=fname,
                value=float(val),
                abs_value=float(abs(val)),
                contribution_pct=float(abs(val) / total_abs * 100),
            ))
        shapley_values.sort(key=lambda s: s.abs_value, reverse=True)
        feature_importance = {s.feature: s.contribution_pct for s in shapley_values}
        interaction_effects = self._compute_interactions(predict_fn, x, baseline,
                                                         feature_names[:n_features],
                                                         min(5, n_features), rng)
        top_features = [s.feature for s in shapley_values[:5]]
        explanation = self._generate_explanation(shapley_values, base_value, prediction)
        confidence = self._compute_confidence(shapley_values)
        return ShapleyResult(
            shapley_values=shapley_values,
            base_value=float(base_value),
            prediction=float(prediction),
            feature_importance=feature_importance,
            interaction_effects=interaction_effects,
            top_features=top_features,
            explanation=explanation,
            confidence=float(confidence),
        )

    @staticmethod
    def _compute_shapley_kernel(predict_fn, x: np.ndarray, baseline: np.ndarray,
                                  feature_names: list[str],
                                  n_samples: int, rng: np.random.RandomState) -> dict:
        n = len(feature_names)
        shapley = {f: 0.0 for f in feature_names}
        all_features = list(range(n))
        for _ in range(n_samples):
            perm = rng.permutation(n)
            base_raw = predict_fn(baseline.reshape(1, -1))
            prev_pred = float(base_raw[0] if hasattr(base_raw, '__len__') else base_raw)
            current_x = baseline.copy()
            for idx in perm:
                current_x[idx] = x[idx]
                raw = predict_fn(current_x.reshape(1, -1))
                new_pred = float(raw[0] if hasattr(raw, '__len__') else raw)
                marginal = new_pred - prev_pred
                shapley[feature_names[idx]] += marginal
                prev_pred = new_pred
        return {f: v / n_samples for f, v in shapley.items()}

    @staticmethod
    def _compute_interactions(predict_fn, x: np.ndarray, baseline: np.ndarray,
                               feature_names: list[str], n_interactions: int,
                               rng: np.random.RandomState) -> dict:
        interactions = {}
        n = len(feature_names)
        pairs = list(combinations(range(min(n_interactions, n)), 2))[:10]
        for i, j in pairs:
            mid = baseline.copy()
            mid[i] = x[i]
            mid_x = baseline.copy()
            mid_x[j] = x[j]
            both = baseline.copy()
            both[i] = x[i]
            both[j] = x[j]
            try:
                f_base = float(predict_fn(baseline.reshape(1, -1))[0])
                f_i = float(predict_fn(mid.reshape(1, -1))[0])
                f_j = float(predict_fn(mid_x.reshape(1, -1))[0])
                f_both = float(predict_fn(both.reshape(1, -1))[0])
                interaction = f_both - f_i - f_j + f_base
                interactions[f"{feature_names[i]} x {feature_names[j]}"] = float(interaction)
            except Exception:
                interactions[f"{feature_names[i]} x {feature_names[j]}"] = 0.0
        return interactions

    @staticmethod
    def _generate_explanation(shapley_values: list, base_value: float,
                               prediction: float) -> str:
        top3 = shapley_values[:3]
        parts = [f"{s.feature}({'+' if s.value > 0 else ''}{s.value:.3f})" for s in top3]
        direction = "higher" if prediction > base_value else "lower"
        magnitude = abs(prediction - base_value)
        return f"Prediction {prediction:.2f} is {direction} than baseline {base_value:.2f} by {magnitude:.2f}. Top drivers: {', '.join(parts)}"

    @staticmethod
    def _compute_confidence(shapley_values: list) -> float:
        if not shapley_values:
            return 0.0
        abs_values = [s.abs_value for s in shapley_values]
        total = sum(abs_values) + 1e-10
        concentration = sum((v / total) ** 2 for v in abs_values)
        return float(min(1.0, concentration * len(shapley_values)))

    def feature_importance_global(self, predict_fn, X: np.ndarray,
                                   feature_names: list[str],
                                   n_samples: int = 500) -> dict:
        n, p = X.shape
        rng = np.random.RandomState(self.seed)
        mean_x = np.mean(X, axis=0)
        importance = {f: 0.0 for f in feature_names}
        for _ in range(n_samples):
            idx = rng.randint(0, n)
            x_sample = X[idx]
            perm = rng.permutation(p)
            mean_raw = predict_fn(mean_x.reshape(1, -1))
            base_pred = float(mean_raw[0] if hasattr(mean_raw, '__len__') else mean_raw)
            current = mean_x.copy()
            for feat_idx in perm:
                current[feat_idx] = x_sample[feat_idx]
                cur_raw = predict_fn(current.reshape(1, -1))
                new_pred = float(cur_raw[0] if hasattr(cur_raw, '__len__') else cur_raw)
                importance[feature_names[feat_idx]] += abs(new_pred - base_pred)
                base_pred = new_pred
        for f in importance:
            importance[f] /= n_samples
        total = sum(importance.values()) + 1e-10
        importance = {f: v / total * 100 for f, v in importance.items()}
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return {
            "feature_importance": sorted_importance,
            "top_features": list(sorted_importance.keys())[:5],
            "method": "shapley_permutation",
            "n_samples": n_samples,
        }

    @staticmethod
    def counterfactual_explanation(predict_fn, x: np.ndarray, feature_names: list[str],
                                    target_value: float, feature_constraints: dict = None,
                                    n_iterations: int = 100) -> dict:
        n = len(feature_names)
        if feature_constraints is None:
            feature_constraints = {}
        current = x.copy()
        cur_raw = predict_fn(current.reshape(1, -1))
        current_pred = float(cur_raw[0] if hasattr(cur_raw, '__len__') else cur_raw)
        changes = {}
        lr = 0.01
        best_x = current.copy()
        best_loss = abs(current_pred - target_value)
        for iteration in range(n_iterations):
            current_pred = float(predict_fn(current.reshape(1, -1))[0])
            loss = abs(current_pred - target_value)
            if loss < best_loss:
                best_loss = loss
                best_x = current.copy()
            if loss < 0.01 * abs(target_value):
                break
            gradient = np.zeros(n)
            eps = 1e-4
            for i in range(n):
                if feature_names[i] in feature_constraints:
                    continue
                x_plus = current.copy()
                x_plus[i] += eps
                x_minus = current.copy()
                x_minus[i] -= eps
                f_plus = float(predict_fn(x_plus.reshape(1, -1))[0])
                f_minus = float(predict_fn(x_minus.reshape(1, -1))[0])
                gradient[i] = (f_plus - f_minus) / (2 * eps)
            gradient = gradient / (np.linalg.norm(gradient) + 1e-10)
            direction = np.sign(target_value - current_pred)
            current = current + direction * lr * gradient * (1 - iteration / n_iterations)
            for i, fname in enumerate(feature_names):
                if fname in feature_constraints:
                    current[i] = feature_constraints[fname]
        final_pred = float(predict_fn(best_x.reshape(1, -1))[0])
        for i, fname in enumerate(feature_names):
            change = float(best_x[i] - x[i])
            if abs(change) > 1e-6:
                changes[fname] = {
                    "original": float(x[i]),
                    "counterfactual": float(best_x[i]),
                    "change": change,
                    "change_pct": float(change / (abs(x[i]) + 1e-10) * 100),
                }
        return {
            "original_prediction": float(current_pred if hasattr(current_pred, '__float__') else 0),
            "counterfactual_prediction": float(final_pred),
            "target_value": float(target_value),
            "achieved": bool(abs(final_pred - target_value) < 0.1 * abs(target_value + 1e-10)),
            "feature_changes": changes,
            "n_features_changed": len(changes),
            "total_change_magnitude": float(np.sum(np.abs(best_x - x))),
        }