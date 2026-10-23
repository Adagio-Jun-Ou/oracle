import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from oracle.core.types import Domain, Prediction, ConfidenceLevel


@dataclass
class FeatureSet:
    names: list[str]
    values: np.ndarray
    target: np.ndarray
    dates: Optional[np.ndarray] = None


@dataclass
class EnsembleResult:
    predictions: np.ndarray
    confidence_intervals: np.ndarray
    model_weights: dict
    individual_predictions: dict
    feature_importance: dict
    cross_validation_scores: dict
    out_of_sample_r2: float = 0.0


class MLEnsemble:
    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models: dict = {}
        self.model_weights: dict = {}
        self.feature_importance_: dict = {}
        self.is_fitted = False

    def _create_models(self) -> dict:
        return {
            "random_forest": RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=15,
                min_samples_split=5, random_state=self.random_state, n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=min(self.n_estimators, 300), max_depth=5,
                learning_rate=0.05, subsample=0.8, random_state=self.random_state,
            ),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=0.01, max_iter=5000),
            "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
            "adaboost": AdaBoostRegressor(
                n_estimators=100, learning_rate=0.1, random_state=self.random_state,
            ),
            "mlp": MLPRegressor(
                hidden_layer_sizes=(64, 32, 16), max_iter=500,
                learning_rate="adaptive", early_stopping=True,
                random_state=self.random_state,
            ),
        }

    def create_features(self, series: np.ndarray, lag: int = 5) -> FeatureSet:
        n = len(series)
        if n < lag + 1:
            lag = max(1, n - 1)
        X_list = []
        names = []
        for i in range(1, lag + 1):
            X_list.append(series[lag - i:n - i])
            names.append(f"lag_{i}")

        if n > 20:
            window = min(10, n - lag)
            rolling_mean = np.convolve(series, np.ones(window) / window, mode="valid")[:n - lag]
            if len(rolling_mean) < n - lag:
                rolling_mean = np.pad(rolling_mean, (n - lag - len(rolling_mean), 0), mode="edge")
            X_list.append(rolling_mean[-(n - lag):])
            names.append("rolling_mean")

            rolling_std = np.array([np.std(series[max(0, j - window):j]) for j in range(lag, n)])
            X_list.append(rolling_std)
            names.append("rolling_std")

        momentum = np.diff(series[lag:n])
        if len(momentum) < n - lag:
            momentum = np.pad(momentum, (n - lag - len(momentum), 0), mode="edge")
        X_list.append(momentum[:n - lag])
        names.append("momentum")

        X = np.column_stack(X_list)
        y = series[lag:]

        return FeatureSet(names=names, values=X, target=y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLEnsemble":
        models = self._create_models()
        X_scaled = self.scaler.fit_transform(X)

        scores = {}
        tscv = TimeSeriesSplit(n_splits=min(5, len(y) // 10))
        for name, model in models.items():
            try:
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    if len(train_idx) < 5 or len(val_idx) < 2:
                        continue
                    m = type(model)(**model.get_params())
                    m.fit(X_scaled[train_idx], y[train_idx])
                    score = m.score(X_scaled[val_idx], y[val_idx])
                    cv_scores.append(max(score, -1.0))
                scores[name] = np.mean(cv_scores) if cv_scores else 0.0
            except Exception:
                scores[name] = 0.0

        total_score = sum(max(s, 0) for s in scores.values()) + 1e-10
        weights = {name: max(score, 0) / total_score for name, score in scores.items()}

        fitted_models = {}
        importances = {}
        for name, model in models.items():
            if weights.get(name, 0) > 0:
                try:
                    model.fit(X_scaled, y)
                    fitted_models[name] = model
                    if hasattr(model, "feature_importances_"):
                        importances[name] = model.feature_importances_
                    elif hasattr(model, "coef_"):
                        importances[name] = np.abs(model.coef_)
                except Exception:
                    weights[name] = 0.0

        total_w = sum(weights.values()) + 1e-10
        weights = {k: v / total_w for k, v in weights.items()}

        self.models = fitted_models
        self.model_weights = weights
        self.feature_importance_ = importances
        self.is_fitted = True
        return self

    def predict_with_uncertainty(
        self, X: np.ndarray, n_bootstrap: int = 200,
    ) -> EnsembleResult:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler.transform(X)
        individual_preds = {}
        for name, model in self.models.items():
            try:
                individual_preds[name] = model.predict(X_scaled)
            except Exception:
                continue

        ensemble_pred = np.zeros(len(X))
        for name, pred in individual_preds.items():
            ensemble_pred += self.model_weights.get(name, 0) * pred

        rng = np.random.default_rng(self.random_state)
        bootstrap_preds = np.zeros((n_bootstrap, len(X)))
        for i in range(n_bootstrap):
            sample_idx = rng.choice(len(X), size=len(X), replace=True)
            sample_pred = np.zeros(len(X))
            total_w = 0
            for name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled[sample_idx])
                    w = self.model_weights.get(name, 0)
                    sample_pred += w * pred
                    total_w += w
                except Exception:
                    continue
            if total_w > 0:
                sample_pred /= total_w
            bootstrap_preds[i] = sample_pred

        pred_std = np.std(bootstrap_preds, axis=0)
        ci_lower = ensemble_pred - 1.96 * pred_std
        ci_upper = ensemble_pred + 1.96 * pred_std

        avg_importance = {}
        if self.feature_importance_:
            all_imp = list(self.feature_importance_.values())
            avg_imp = np.mean(all_imp, axis=0)
            for i, name in enumerate(self.feature_importance_.keys()):
                avg_importance[name] = float(avg_imp[i]) if i < len(avg_imp) else 0.0

        cv_scores = {name: float(w) for name, w in self.model_weights.items()}

        return EnsembleResult(
            predictions=ensemble_pred,
            confidence_intervals=np.column_stack([ci_lower, ci_upper]),
            model_weights=self.model_weights,
            individual_predictions=individual_preds,
            feature_importance=avg_importance,
            cross_validation_scores=cv_scores,
        )

    def predict(
        self, series: np.ndarray, variable: str, domain: Domain,
        target_date: date, confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH,
        n_ahead: int = 30,
    ) -> Prediction:
        features = self.create_features(series)
        if not self.is_fitted:
            self.fit(features.values, features.target)

        n_ahead = min(n_ahead, len(series) // 2)
        last_window = series[-max(5, len(series) // 5):]
        X_forecast = np.column_stack([last_window[i:i + len(features.names)].mean() * np.ones(1) for i in range(min(3, len(features.names)))])
        if X_forecast.shape[1] < len(features.names):
            padding = np.ones((1, len(features.names) - X_forecast.shape[1])) * np.mean(series)
            X_forecast = np.column_stack([X_forecast, padding])

        result = self.predict_with_uncertainty(features.values)

        z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645,
             ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence_level]

        mean_val = float(result.predictions[-1])
        std_val = float(np.std(result.predictions)) if len(result.predictions) > 1 else abs(mean_val) * 0.1

        return Prediction(
            domain=domain, variable=variable, value=round(mean_val, 4),
            confidence_interval=(round(mean_val - z * std_val, 4), round(mean_val + z * std_val, 4)),
            confidence_level=confidence_level, timestamp=datetime.now(),
            target_date=target_date, model_name="ml_ensemble",
            weights=self.model_weights,
        )