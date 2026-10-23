import math
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class SurvivalResult:
    survival_function: list
    hazard_rate: list
    median_survival: float
    mean_survival: float
    confidence_lower: list
    confidence_upper: list
    n_events: int
    n_censored: int
    method: str


@dataclass
class CoxResult:
    coefficients: dict
    hazard_ratios: dict
    confidence_intervals: dict
    p_values: dict
    concordance: float
    log_likelihood: float
    n_events: int
    method: str


@dataclass
class KaplanMeierResult:
    times: list
    survival: list
    confidence_lower: list
    confidence_upper: list
    n_at_risk: list
    n_events: list
    median_survival: float
    method: str


class SurvivalAnalysis:
    def __init__(self, seed: int = 42):
        self.seed = seed

    @staticmethod
    def kaplan_meier(times: np.ndarray, events: np.ndarray,
                     conf_type: str = "log-log") -> KaplanMeierResult:
        sorted_idx = np.argsort(times)
        times_sorted = times[sorted_idx]
        events_sorted = events[sorted_idx]
        n = len(times_sorted)
        unique_times = np.unique(times_sorted)
        survival = 1.0
        survival_list = [1.0]
        time_list = [0.0]
        n_at_risk_list = [n]
        n_events_list = [0]
        var_log_surv = 0.0
        lower_list = [1.0]
        upper_list = [1.0]
        for t in unique_times:
            at_risk = np.sum(times_sorted >= t)
            d = np.sum(events_sorted[times_sorted == t])
            if at_risk > 0:
                survival *= (1 - d / at_risk)
                var_log_surv += d / (at_risk * (at_risk - d)) if at_risk > d else 0
            survival_list.append(float(survival))
            time_list.append(float(t))
            n_at_risk_list.append(int(at_risk))
            n_events_list.append(int(d))
            z = 1.96
            log_surv = np.log(max(survival, 1e-10))
            se_log = np.sqrt(max(var_log_surv, 0))
            lower = float(np.exp(log_surv - z * se_log)) if se_log > 0 else float(survival)
            upper = float(np.exp(log_surv + z * se_log)) if se_log > 0 else float(survival)
            lower_list.append(max(0, lower))
            upper_list.append(min(1, upper))
        median = 0.0
        for i in range(len(survival_list)):
            if survival_list[i] <= 0.5:
                if i > 0:
                    t_prev = time_list[i - 1]
                    t_curr = time_list[i]
                    s_prev = survival_list[i - 1]
                    s_curr = survival_list[i]
                    if s_prev > 0.5 and s_prev != s_curr:
                        median = t_prev + (0.5 - s_prev) / (s_curr - s_prev) * (t_curr - t_prev)
                    else:
                        median = t_curr
                break
        return KaplanMeierResult(
            times=time_list,
            survival=survival_list,
            confidence_lower=lower_list,
            confidence_upper=upper_list,
            n_at_risk=n_at_risk_list,
            n_events=n_events_list,
            median_survival=float(median),
            method="kaplan_meier",
        )

    @staticmethod
    def cox_regression(X: np.ndarray, times: np.ndarray, events: np.ndarray,
                       n_iterations: int = 100, lr: float = 0.01) -> CoxResult:
        n, p = X.shape
        rng = np.random.RandomState(42)
        beta = rng.normal(0, 0.01, p)
        sorted_idx = np.argsort(times)
        X_sorted = X[sorted_idx]
        times_sorted = times[sorted_idx]
        events_sorted = events[sorted_idx]
        n_events = int(np.sum(events_sorted))
        log_likelihood = 0.0
        for iteration in range(n_iterations):
            risk_scores = np.exp(X_sorted @ beta)
            risk_sum = np.cumsum(risk_scores[::-1])[::-1]
            gradient = np.zeros(p)
            ll = 0.0
            for i in range(n):
                if events_sorted[i] == 1:
                    gradient += X_sorted[i] - np.sum(X_sorted * risk_scores[:, np.newaxis], axis=0) / max(risk_sum[i], 1e-10)
                    ll += np.log(max(risk_scores[i], 1e-10)) - np.log(max(risk_sum[i], 1e-10))
            beta += lr * gradient / n
            log_likelihood = ll
        coefficients = {f"var_{i}": float(beta[i]) for i in range(p)}
        hazard_ratios = {f"var_{i}": float(np.exp(beta[i])) for i in range(p)}
        se = np.sqrt(np.maximum(1 / n, np.abs(beta) * 0.1))
        confidence_intervals = {f"var_{i}": (float(np.exp(beta[i] - 1.96 * se[i])),
                                                float(np.exp(beta[i] + 1.96 * se[i]))) for i in range(p)}
        p_values = {f"var_{i}": float(2 * (1 - 0.5 * (1 + math.erf(abs(beta[i]) / (se[i] + 1e-10) / np.sqrt(2))))) for i in range(p)}
        concordant = 0
        discordant = 0
        for i in range(min(n, 200)):
            for j in range(i + 1, min(n, 200)):
                if events_sorted[i] == 1 and events_sorted[j] == 1:
                    risk_i = float(X_sorted[i] @ beta)
                    risk_j = float(X_sorted[j] @ beta)
                    if (times_sorted[i] < times_sorted[j] and risk_i > risk_j) or \
                       (times_sorted[i] > times_sorted[j] and risk_i < risk_j):
                        concordant += 1
                    elif times_sorted[i] != times_sorted[j]:
                        discordant += 1
        concordance = concordant / max(concordant + discordant, 1)
        return CoxResult(
            coefficients=coefficients,
            hazard_ratios=hazard_ratios,
            confidence_intervals=confidence_intervals,
            p_values=p_values,
            concordance=float(concordance),
            log_likelihood=float(log_likelihood),
            n_events=n_events,
            method="cox_partial_likelihood",
        )

    @staticmethod
    def exponential_survival(times: np.ndarray, events: np.ndarray) -> SurvivalResult:
        n_events = int(np.sum(events))
        total_time = float(np.sum(times))
        if n_events == 0:
            return SurvivalResult(survival_function=[], hazard_rate=[], median_survival=0.0,
                                   mean_survival=0.0, confidence_lower=[], confidence_upper=[],
                                   n_events=0, n_censored=len(times), method="exponential")
        lambda_rate = n_events / max(total_time, 1e-10)
        mean_survival = 1.0 / max(lambda_rate, 1e-10)
        median_survival = np.log(2) / max(lambda_rate, 1e-10)
        t_grid = np.linspace(0, max(times) * 1.5, 50)
        survival = np.exp(-lambda_rate * t_grid)
        hazard_rate_arr = np.full_like(t_grid, lambda_rate)
        se_lambda = lambda_rate / np.sqrt(max(n_events, 1))
        z = 1.96
        lower = np.exp(-(lambda_rate + z * se_lambda) * t_grid)
        upper = np.exp(-(lambda_rate - z * se_lambda) * t_grid)
        return SurvivalResult(
            survival_function=survival.tolist(),
            hazard_rate=hazard_rate_arr.tolist(),
            median_survival=float(median_survival),
            mean_survival=float(mean_survival),
            confidence_lower=lower.tolist(),
            confidence_upper=upper.tolist(),
            n_events=n_events,
            n_censored=int(len(times) - n_events),
            method="exponential",
        )

    @staticmethod
    def weibull_survival(times: np.ndarray, events: np.ndarray) -> SurvivalResult:
        n_events = int(np.sum(events))
        if n_events == 0:
            return SurvivalResult(survival_function=[], hazard_rate=[], median_survival=0.0,
                                   mean_survival=0.0, confidence_lower=[], confidence_upper=[],
                                   n_events=0, n_censored=len(times), method="weibull")
        mean_t = float(np.mean(times[events == 1])) if n_events > 0 else float(np.mean(times))
        std_t = float(np.std(times[events == 1])) if n_events > 1 else float(np.std(times))
        k = max(1.0, (std_t / max(mean_t, 1e-10)) ** (-1.086)) if std_t > 0 else 1.0
        lam = mean_t / float(math.gamma(1 + 1 / k)) if k > 0 else mean_t
        t_grid = np.linspace(0, max(times) * 1.5, 50)
        survival = np.exp(-(t_grid / max(lam, 1e-10)) ** k)
        hazard_rate = (k / max(lam, 1e-10)) * (t_grid / max(lam, 1e-10)) ** (k - 1)
        median = lam * (np.log(2)) ** (1 / k) if k > 0 else 0.0
        mean_val = lam * float(math.gamma(1 + 1 / k)) if k > 0 else 0.0
        se = 0.1 * np.ones_like(t_grid)
        lower = np.exp(-(t_grid / max(lam * 0.9, 1e-10)) ** k)
        upper = np.exp(-(t_grid / max(lam * 1.1, 1e-10)) ** k)
        return SurvivalResult(
            survival_function=survival.tolist(),
            hazard_rate=hazard_rate.tolist(),
            median_survival=float(median),
            mean_survival=float(mean_val),
            confidence_lower=lower.tolist(),
            confidence_upper=upper.tolist(),
            n_events=n_events,
            n_censored=int(len(times) - n_events),
            method="weibull",
        )