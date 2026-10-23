from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np


@dataclass
class EnsembleResult:
    prediction: float
    prediction_interval: tuple
    weights: dict
    method: str
    individual_predictions: dict
    accuracy_scores: dict
    diversity_score: float
    n_models: int


@dataclass
class StackingResult:
    prediction: float
    meta_coefficients: list
    base_predictions: dict
    meta_model_type: str
    cv_folds: int
    oof_score: float


@dataclass
class BayesianModelAveragingResult:
    prediction: float
    posterior_probabilities: dict
    marginal_likelihoods: dict
    prior_probabilities: dict
    bayes_factors: dict
    model_uncertainty: float
    n_models: int


@dataclass
class SuperLearnerResult:
    prediction: float
    weights: list
    model_names: list
    cv_risks: list
    discrete_winner: str
    method: str = "super_learner"


class AdvancedEnsemble:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._fitted = False
        self._base_models = {}
        self._meta_weights = None

    @staticmethod
    def crps_weighting(predictions: np.ndarray, actuals: np.ndarray) -> dict:
        n_models = predictions.shape[1]
        crps_scores = np.zeros(n_models)
        for m in range(n_models):
            preds = predictions[:, m]
            errors = preds - actuals
            crps_scores[m] = float(np.mean(np.abs(errors) - 0.5 * np.abs(errors)))
            if np.std(preds) > 1e-10:
                crps_scores[m] = float(np.mean((errors) ** 2 - 0.5 * np.var(preds)))
        inv_crps = 1.0 / (crps_scores + 1e-10)
        weights = inv_crps / inv_crps.sum()
        weighted_pred = float(np.average(predictions.mean(axis=0), weights=weights))
        return {
            "weights": {f"model_{m}": float(weights[m]) for m in range(n_models)},
            "crps_scores": {f"model_{m}": float(crps_scores[m]) for m in range(n_models)},
            "weighted_prediction": weighted_pred,
            "method": "crps_weighting",
        }

    @staticmethod
    def optimal_combination(predictions: np.ndarray, actuals: np.ndarray,
                            constrain_nonneg: bool = True) -> dict:
        n, m = predictions.shape
        H = predictions.T @ predictions / n
        h = predictions.T @ actuals / n
        try:
            weights = np.linalg.solve(H + 1e-6 * np.eye(m), h)
        except np.linalg.LinAlgError:
            weights = np.ones(m) / m
        if constrain_nonneg:
            weights = np.maximum(weights, 0)
        w_sum = weights.sum()
        if abs(w_sum) > 1e-10:
            weights = weights / w_sum
        else:
            weights = np.ones(m) / m
        combined = predictions @ weights
        mse = float(np.mean((combined - actuals) ** 2))
        return {
            "weights": {f"model_{i}": float(weights[i]) for i in range(m)},
            "combined_mse": mse,
            "method": "optimal_combination",
        }

    @staticmethod
    def stacking(predictions: np.ndarray, actuals: np.ndarray,
                 n_folds: int = 5, meta_type: str = "ridge") -> StackingResult:
        n, m = predictions.shape
        rng = np.random.RandomState(42)
        fold_ids = rng.randint(0, n_folds, n)
        oof_preds = np.zeros((n, m))
        for fold in range(n_folds):
            train_mask = fold_ids != fold
            val_mask = fold_ids == fold
            if not np.any(val_mask):
                continue
            for mod in range(m):
                train_preds = predictions[train_mask, mod]
                train_actuals = actuals[train_mask]
                if np.std(train_preds) < 1e-10:
                    oof_preds[val_mask, mod] = np.mean(train_actuals)
                else:
                    cov = np.cov(train_preds, train_actuals)
                    if cov[0, 0] > 1e-10:
                        beta = cov[0, 1] / cov[0, 0]
                        alpha = np.mean(train_actuals) - beta * np.mean(train_preds)
                        oof_preds[val_mask, mod] = alpha + beta * predictions[val_mask, mod]
                    else:
                        oof_preds[val_mask, mod] = np.mean(train_actuals)
        if meta_type == "ridge":
            lam = 1.0
            H = oof_preds.T @ oof_preds + lam * np.eye(m)
            h = oof_preds.T @ actuals
            try:
                meta_coefs = np.linalg.solve(H, h)
            except np.linalg.LinAlgError:
                meta_coefs = np.ones(m) / m
        else:
            meta_coefs = np.ones(m) / m
        meta_pred = float(np.mean(oof_preds @ meta_coefs))
        oof_full = oof_preds @ meta_coefs
        oof_score = float(1 - np.mean((actuals - oof_full) ** 2) / max(np.var(actuals), 1e-10))
        base_preds = {f"model_{i}": float(np.mean(predictions[:, i])) for i in range(m)}
        return StackingResult(
            prediction=meta_pred,
            meta_coefficients=[float(c) for c in meta_coefs],
            base_predictions=base_preds,
            meta_model_type=meta_type,
            cv_folds=n_folds,
            oof_score=max(0, oof_score),
        )

    @staticmethod
    def bayesian_model_averaging(predictions: np.ndarray, actuals: np.ndarray,
                                   prior_probs: Optional[np.ndarray] = None) -> BayesianModelAveragingResult:
        n, m = predictions.shape
        if prior_probs is None:
            prior_probs = np.ones(m) / m
        else:
            prior_probs = prior_probs / prior_probs.sum()
        log_marginal_likelihoods = np.zeros(m)
        sigma2_prior = 1.0
        for mod in range(m):
            residuals = actuals - predictions[:, mod]
            sse = float(np.sum(residuals ** 2))
            log_ml = -0.5 * n * np.log(2 * np.pi * sigma2_prior) - sse / (2 * sigma2_prior)
            log_marginal_likelihoods[mod] = log_ml
        log_ml_max = np.max(log_marginal_likelihoods)
        log_weights = log_marginal_likelihoods + np.log(prior_probs) - log_ml_max
        weights_unnorm = np.exp(log_weights)
        posterior_probs = weights_unnorm / weights_unnorm.sum()
        bma_prediction = float(np.sum(predictions @ posterior_probs))
        bayes_factors = {}
        for i in range(m):
            for j in range(i + 1, m):
                bf = np.exp(log_marginal_likelihoods[i] - log_marginal_likelihoods[j])
                bayes_factors[f"model_{i}_vs_{j}"] = float(bf)
        model_uncertainty = float(max(0, -np.sum(posterior_probs * np.log(posterior_probs + 1e-10))))
        return BayesianModelAveragingResult(
            prediction=bma_prediction,
            posterior_probabilities={f"model_{i}": float(posterior_probs[i]) for i in range(m)},
            marginal_likelihoods={f"model_{i}": float(np.exp(log_marginal_likelihoods[i])) for i in range(m)},
            prior_probabilities={f"model_{i}": float(prior_probs[i]) for i in range(m)},
            bayes_factors=bayes_factors,
            model_uncertainty=model_uncertainty,
            n_models=m,
        )

    @staticmethod
    def super_learner(predictions: np.ndarray, actuals: np.ndarray,
                      n_folds: int = 5) -> SuperLearnerResult:
        n, m = predictions.shape
        rng = np.random.RandomState(42)
        fold_ids = rng.randint(0, n_folds, n)
        oof_preds = np.zeros((n, m))
        for fold in range(n_folds):
            val_mask = fold_ids == fold
            train_mask = ~val_mask
            if not np.any(val_mask):
                continue
            for mod in range(m):
                oof_preds[val_mask, mod] = predictions[val_mask, mod]
        cv_risks = []
        for mod in range(m):
            risk = float(np.mean((oof_preds[:, mod] - actuals) ** 2))
            cv_risks.append(risk)
        min_risk_idx = int(np.argmin(cv_risks))
        inv_risks = 1.0 / (np.array(cv_risks) + 1e-10)
        sl_weights = inv_risks / inv_risks.sum()
        discrete_winner = f"model_{min_risk_idx}"
        sl_prediction = float(np.mean(oof_preds @ sl_weights))
        model_names = [f"model_{i}" for i in range(m)]
        return SuperLearnerResult(
            prediction=sl_prediction,
            weights=[float(w) for w in sl_weights],
            model_names=model_names,
            cv_risks=[float(r) for r in cv_risks],
            discrete_winner=discrete_winner,
        )

    @staticmethod
    def diversity_analysis(predictions: np.ndarray) -> dict:
        n, m = predictions.shape
        if m < 2:
            return {"diversity_score": 0.0, "pairwise_correlations": {},
                    "disagreement": 0.0, "n_models": m}
        corr_matrix = np.corrcoef(predictions.T)
        upper_tri = []
        for i in range(m):
            for j in range(i + 1, m):
                upper_tri.append(corr_matrix[i, j])
        avg_corr = float(np.mean(upper_tri)) if upper_tri else 0.0
        diversity_score = float(max(0, min(1, 1.0 - avg_corr)))
        disagreement = float(np.mean(np.std(predictions, axis=1)))
        pairwise = {}
        for i in range(m):
            for j in range(i + 1, m):
                pairwise[f"model_{i}_vs_{j}"] = float(corr_matrix[i, j])
        return {
            "diversity_score": diversity_score,
            "pairwise_correlations": pairwise,
            "avg_correlation": avg_corr,
            "disagreement": disagreement,
            "n_models": m,
        }

    @staticmethod
    def forecast_reconciliation(base_forecasts: np.ndarray,
                                  summing_matrix: Optional[np.ndarray] = None) -> dict:
        n = len(base_forecasts)
        if summing_matrix is None:
            total = float(np.sum(base_forecasts))
            reconciled = base_forecasts.copy()
            bottom_sum = float(np.sum(base_forecasts))
            if abs(bottom_sum) > 1e-10:
                scale = total / bottom_sum
                reconciled = base_forecasts * scale
            return {
                "base_forecasts": base_forecasts.tolist(),
                "reconciled_forecasts": reconciled.tolist(),
                "total_base": float(np.sum(base_forecasts)),
                "total_reconciled": float(np.sum(reconciled)),
                "method": "bottom_up",
            }
        S = summing_matrix
        n_upper = S.shape[0]
        n_bottom = S.shape[1]
        bottom_forecasts = base_forecasts[n_upper:]
        reconciled_all = S @ bottom_forecasts
        reconciled_full = np.concatenate([reconciled_all, bottom_forecasts])
        return {
            "base_forecasts": base_forecasts.tolist(),
            "reconciled_forecasts": reconciled_full.tolist(),
            "method": "bottom_up_matrix",
        }