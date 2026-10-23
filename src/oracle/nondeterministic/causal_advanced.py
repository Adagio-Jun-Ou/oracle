import math

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class DMLResult:
    treatment_effect: float
    standard_error: float
    confidence_interval: tuple
    p_value: float
    method: str
    n_cross_fits: int
    first_stage_r2: dict


@dataclass
class RDDResult:
    treatment_effect: float
    standard_error: float
    confidence_interval: tuple
    p_value: float
    bandwidth: float
    kernel: str
    left_mean: float
    right_mean: float
    jump_estimate: float
    method: str


@dataclass
class IV2SLSResult:
    treatment_effect: float
    standard_error: float
    confidence_interval: tuple
    p_value: float
    first_stage_f: float
    weak_instrument_test: str
    method: str


@dataclass
class DifferenceInDifferencesResult:
    treatment_effect: float
    standard_error: float
    confidence_interval: tuple
    p_value: float
    parallel_trend_test: dict
    method: str


class DoubleMachineLearning:
    def __init__(self, n_cross_fits: int = 2, seed: int = 42):
        self.n_cross_fits = n_cross_fits
        self.seed = seed

    def fit(self, Y: np.ndarray, D: np.ndarray, X: np.ndarray,
            n_cross_fits: Optional[int] = None) -> DMLResult:
        n_cross_fits = n_cross_fits or self.n_cross_fits
        n = len(Y)
        rng = np.random.RandomState(self.seed)
        fold_ids = rng.randint(0, n_cross_fits, n)
        residuals_Y = np.zeros(n)
        residuals_D = np.zeros(n)
        first_stage_r2 = {"outcome_model": 0.0, "treatment_model": 0.0}
        for fold in range(n_cross_fits):
            train = fold_ids != fold
            test = fold_ids == fold
            X_train, X_test = X[train], X[test]
            Y_train, D_train = Y[train], D[train]
            X_with_const_train = np.column_stack([np.ones(X_train.shape[0]), X_train])
            X_with_const_test = np.column_stack([np.ones(X_test.shape[0]), X_test])
            gram = X_with_const_train.T @ X_with_const_train + 1e-6 * np.eye(X_with_const_train.shape[1])
            beta_Y = np.linalg.solve(gram, X_with_const_train.T @ Y_train)
            beta_D = np.linalg.solve(gram, X_with_const_train.T @ D_train)
            Y_pred = X_with_const_test @ beta_Y
            D_pred = X_with_const_test @ beta_D
            residuals_Y[test] = Y[test] - Y_pred
            residuals_D[test] = D[test] - D_pred
            ss_res_Y = np.sum((Y[test] - Y_pred) ** 2)
            ss_tot_Y = np.sum((Y[test] - np.mean(Y[test])) ** 2) + 1e-10
            ss_res_D = np.sum((D[test] - D_pred) ** 2)
            ss_tot_D = np.sum((D[test] - np.mean(D[test])) ** 2) + 1e-10
            first_stage_r2["outcome_model"] += 1 - ss_res_Y / ss_tot_Y
            first_stage_r2["treatment_model"] += 1 - ss_res_D / ss_tot_D
        first_stage_r2["outcome_model"] /= n_cross_fits
        first_stage_r2["treatment_model"] /= n_cross_fits
        theta = float(np.sum(residuals_Y * residuals_D) / (np.sum(residuals_D ** 2) + 1e-10))
        se = float(np.sqrt(np.mean((residuals_Y - theta * residuals_D) ** 2) / (np.sum(residuals_D ** 2) + 1e-10)))
        ci_lower = theta - 1.96 * se
        ci_upper = theta + 1.96 * se
        p_value = 2 * (1 - _normal_cdf(abs(theta) / (se + 1e-10)))
        return DMLResult(
            treatment_effect=theta,
            standard_error=se,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            p_value=float(p_value),
            method="double_ml_cross_fit",
            n_cross_fits=n_cross_fits,
            first_stage_r2=first_stage_r2,
        )

    @staticmethod
    def regression_discontinuity(Y: np.ndarray, X: np.ndarray,
                                  cutoff: float = 0.0, bandwidth: Optional[float] = None,
                                  kernel: str = "triangular") -> RDDResult:
        if bandwidth is None:
            std = float(np.std(X))
            bandwidth = std * 0.5
        mask_left = X < cutoff
        mask_right = X >= cutoff
        n_left = np.sum(mask_left)
        n_right = np.sum(mask_right)
        if n_left < 3 or n_right < 3:
            bandwidth = float(np.std(X))
            mask_left = X < cutoff
            mask_right = X >= cutoff
        X_centered = X - cutoff
        if kernel == "triangular":
            weights = np.maximum(0, 1 - np.abs(X_centered) / bandwidth)
        elif kernel == "uniform":
            weights = (np.abs(X_centered) <= bandwidth).astype(float)
        else:
            weights = np.exp(-0.5 * (X_centered / bandwidth) ** 2)
        within_band = np.abs(X_centered) <= bandwidth * 2
        X_w = X_centered[within_band]
        Y_w = Y[within_band]
        D_w = (X_w >= 0).astype(float)
        W_w = weights[within_band]
        X_design = np.column_stack([np.ones(len(X_w)), X_w, D_w, X_w * D_w])
        W_diag = np.diag(W_w)
        try:
            gram = X_design.T @ W_diag @ X_design + 1e-6 * np.eye(X_design.shape[1])
            beta = np.linalg.solve(gram, X_design.T @ W_diag @ Y_w)
        except np.linalg.LinAlgError:
            beta = np.array([np.mean(Y_w), 0.0, 0.0, 0.0])
        treatment_effect = float(beta[2])
        resid = Y_w - X_design @ beta
        se = float(np.sqrt(np.sum(W_w * resid ** 2) / (max(np.sum(W_w > 0) - 4, 1))))
        left_mean = float(np.mean(Y[mask_left])) if n_left > 0 else 0.0
        right_mean = float(np.mean(Y[mask_right])) if n_right > 0 else 0.0
        ci_lower = treatment_effect - 1.96 * se
        ci_upper = treatment_effect + 1.96 * se
        p_value = 2 * (1 - _normal_cdf(abs(treatment_effect) / (se + 1e-10)))
        return RDDResult(
            treatment_effect=treatment_effect,
            standard_error=se,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            p_value=float(p_value),
            bandwidth=bandwidth,
            kernel=kernel,
            left_mean=left_mean,
            right_mean=right_mean,
            jump_estimate=treatment_effect,
            method="local_linear_rdd",
        )

    @staticmethod
    def iv_2sls(Y: np.ndarray, D: np.ndarray, Z: np.ndarray,
                X: Optional[np.ndarray] = None) -> IV2SLSResult:
        if X is None:
            X = np.ones((len(Y), 1))
        else:
            X = np.column_stack([np.ones(len(Y)), X])
        Z_aug = np.column_stack([Z, X])
        gram_D = Z_aug.T @ Z_aug + 1e-6 * np.eye(Z_aug.shape[1])
        D_hat = Z_aug @ np.linalg.solve(gram_D, Z_aug.T @ D)
        first_stage_f = float((np.var(D_hat) / max(np.var(D - D_hat), 1e-10)))
        D_hat_aug = np.column_stack([D_hat, X])
        gram_Y = D_hat_aug.T @ D_hat_aug + 1e-6 * np.eye(D_hat_aug.shape[1])
        beta_2sls = np.linalg.solve(gram_Y, D_hat_aug.T @ Y)
        treatment_effect = float(beta_2sls[0])
        resid = Y - D_hat_aug @ beta_2sls
        n = len(Y)
        k = D_hat_aug.shape[1]
        se = float(np.sqrt(np.sum(resid ** 2) / ((n - k) * (D_hat_aug[:, 0].T @ D_hat_aug[:, 0]) + 1e-10)))
        ci_lower = treatment_effect - 1.96 * se
        ci_upper = treatment_effect + 1.96 * se
        p_value = 2 * (1 - _normal_cdf(abs(treatment_effect) / (se + 1e-10)))
        weak_test = "strong" if first_stage_f > 10 else "weak"
        return IV2SLSResult(
            treatment_effect=treatment_effect,
            standard_error=se,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            p_value=float(p_value),
            first_stage_f=first_stage_f,
            weak_instrument_test=weak_test,
            method="iv_2sls",
        )

    @staticmethod
    def difference_in_differences(Y_treated_pre: np.ndarray, Y_treated_post: np.ndarray,
                                   Y_control_pre: np.ndarray, Y_control_post: np.ndarray) -> DifferenceInDifferencesResult:
        treated_diff = float(np.mean(Y_treated_post) - np.mean(Y_treated_pre))
        control_diff = float(np.mean(Y_control_post) - np.mean(Y_control_pre))
        att = treated_diff - control_diff
        var_treated = np.var(Y_treated_post - Y_treated_pre) / len(Y_treated_post) if len(Y_treated_post) > 1 else 0.01
        var_control = np.var(Y_control_post - Y_control_pre) / len(Y_control_post) if len(Y_control_post) > 1 else 0.01
        se = float(np.sqrt(var_treated + var_control))
        ci_lower = att - 1.96 * se
        ci_upper = att + 1.96 * se
        p_value = 2 * (1 - _normal_cdf(abs(att) / (se + 1e-10)))
        parallel_trend = {
            "treated_pre_trend": float(np.mean(Y_treated_pre)),
            "control_pre_trend": float(np.mean(Y_control_pre)),
            "pre_difference": float(np.mean(Y_treated_pre) - np.mean(Y_control_pre)),
            "parallel_assumption": abs(np.mean(Y_treated_pre) - np.mean(Y_control_pre)) < 2 * se,
        }
        return DifferenceInDifferencesResult(
            treatment_effect=att,
            standard_error=se,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            p_value=float(p_value),
            parallel_trend_test=parallel_trend,
            method="did",
        )


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))