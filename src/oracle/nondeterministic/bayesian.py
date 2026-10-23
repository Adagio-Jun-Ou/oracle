import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from scipy import stats as sp_stats
from oracle.core.types import Domain, Prediction, ConfidenceLevel


@dataclass
class Prior:
    distribution: str = "normal"
    params: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 1.0})


@dataclass
class PosteriorResult:
    distribution: str = ""
    params: dict = field(default_factory=dict)
    mean: float = 0.0
    std: float = 1.0
    credible_interval: tuple[float, float] = (0.0, 0.0)
    bayes_factor: float = 1.0
    prior_predictive: float = 0.0
    posterior_predictive: float = 0.0
    evidence: float = 0.0


class BayesianEngine:
    def __init__(self, prior_strength: float = 1.0):
        self.prior_strength = prior_strength

    def conjugate_normal_normal(
        self, data: np.ndarray, prior_mu: float = 0.0, prior_sigma: float = 1.0,
        known_sigma: float = 0.5,
    ) -> PosteriorResult:
        n = len(data)
        xbar = np.mean(data)
        prior_precision = 1.0 / prior_sigma ** 2
        data_precision = n / known_sigma ** 2

        posterior_precision = prior_precision * self.prior_strength + data_precision
        posterior_mu = (prior_precision * self.prior_strength * prior_mu + data_precision * xbar) / posterior_precision
        posterior_sigma = 1.0 / np.sqrt(posterior_precision)

        ci_95 = (posterior_mu - 1.96 * posterior_sigma, posterior_mu + 1.96 * posterior_sigma)

        return PosteriorResult(
            distribution="normal",
            params={"mu": posterior_mu, "sigma": posterior_sigma},
            mean=posterior_mu,
            std=posterior_sigma,
            credible_interval=ci_95,
            bayes_factor=self._compute_bayes_factor(data, prior_mu, prior_sigma, posterior_mu, posterior_sigma, known_sigma),
            posterior_predictive=posterior_mu,
            evidence=float(np.exp(np.clip(-0.5 * ((xbar - prior_mu) ** 2) / (prior_sigma ** 2 + known_sigma ** 2 / n), -700, 700))),
        )

    def conjugate_beta_binomial(
        self, successes: int, trials: int, prior_alpha: float = 1.0, prior_beta: float = 1.0,
    ) -> PosteriorResult:
        posterior_alpha = prior_alpha + successes
        posterior_beta = prior_beta + (trials - successes)

        mean = posterior_alpha / (posterior_alpha + posterior_beta)
        std = np.sqrt((posterior_alpha * posterior_beta) / ((posterior_alpha + posterior_beta) ** 2 * (posterior_alpha + posterior_beta + 1)))

        ci_95 = (
            float(sp_stats.beta.ppf(0.025, posterior_alpha, posterior_beta)),
            float(sp_stats.beta.ppf(0.975, posterior_alpha, posterior_beta)),
        )

        return PosteriorResult(
            distribution="beta",
            params={"alpha": posterior_alpha, "beta": posterior_beta},
            mean=float(mean),
            std=float(std),
            credible_interval=ci_95,
            bayes_factor=float((sp_stats.beta.pdf(successes / max(trials, 1), posterior_alpha, posterior_beta)) /
                              max(sp_stats.beta.pdf(0.5, prior_alpha, prior_beta), 1e-10)),
        )

    def conjugate_gamma_poisson(
        self, counts: np.ndarray, prior_alpha: float = 1.0, prior_beta: float = 1.0,
    ) -> PosteriorResult:
        n = len(counts)
        total = np.sum(counts)
        posterior_alpha = prior_alpha + total
        posterior_beta = prior_beta + n

        mean = posterior_alpha / posterior_beta
        std = np.sqrt(posterior_alpha / posterior_beta ** 2)

        ci_95 = (
            float(sp_stats.gamma.ppf(0.025, posterior_alpha, scale=1.0 / posterior_beta)),
            float(sp_stats.gamma.ppf(0.975, posterior_alpha, scale=1.0 / posterior_beta)),
        )

        return PosteriorResult(
            distribution="gamma",
            params={"alpha": posterior_alpha, "beta": posterior_beta},
            mean=float(mean),
            std=float(std),
            credible_interval=ci_95,
        )

    def model_comparison(
        self, data: np.ndarray, null_mu: float = 0.0, alt_mu_range: tuple[float, float] = (-2.0, 2.0),
    ) -> dict:
        n = len(data)
        xbar = np.mean(data)
        se = np.std(data, ddof=1) / np.sqrt(n)

        null_loglike = np.sum(sp_stats.norm.logpdf(data, null_mu, se))
        alt_mu = np.linspace(alt_mu_range[0], alt_mu_range[1], 1000)
        alt_loglikes = np.array([np.sum(sp_stats.norm.logpdf(data, mu, se)) for mu in alt_mu])
        best_alt_loglike = np.max(alt_loglikes)

        log_bf = best_alt_loglike - null_loglike
        bf = float(np.exp(np.clip(log_bf, -700, 700)))

        return {
            "null_hypothesis_mu": null_mu,
            "alternative_mu_range": alt_mu_range,
            "bayes_factor": bf,
            "log_bayes_factor": float(log_bf),
            "best_alternative_mu": float(alt_mu[np.argmax(alt_loglikes)]),
            "favor_null": bf < 1.0 / 3.0,
            "favor_alternative": bf > 3.0,
            "inconclusive": 1.0 / 3.0 <= bf <= 3.0,
        }

    def bayesian_change_point(
        self, data: np.ndarray, max_change_points: int = 5,
    ) -> dict:
        n = len(data)
        if n < 3:
            return {"change_points": [], "probabilities": np.array([])}

        log_probs = np.zeros((max_change_points + 1, n))
        log_probs[0, :] = sp_stats.norm.logpdf(data, np.mean(data), np.std(data) + 1e-10)

        for k in range(1, max_change_points + 1):
            for t in range(k, n):
                left_mean = np.mean(data[:t])
                right_mean = np.mean(data[t:])
                left_std = np.std(data[:t]) + 1e-10
                right_std = np.std(data[t:]) + 1e-10
                log_probs[k, t] = (
                    np.sum(sp_stats.norm.logpdf(data[:t], left_mean, left_std)) +
                    np.sum(sp_stats.norm.logpdf(data[t:], right_mean, right_std))
                )

        best_k = int(np.argmax(np.max(log_probs, axis=1)))
        if best_k == 0:
            return {"change_points": [], "probabilities": np.array([])}

        change_points = []
        remaining = data.copy()
        offset = 0
        for k in range(best_k):
            if len(remaining) < 3:
                break
            best_idx = 1
            best_score = -np.inf
            for i in range(1, len(remaining) - 1):
                left = remaining[:i]
                right = remaining[i:]
                score = abs(np.mean(left) - np.mean(right)) / (np.std(left) + np.std(right) + 1e-10)
                if score > best_score:
                    best_score = score
                    best_idx = i
            change_points.append(offset + best_idx)
            remaining = remaining[best_idx:]

        return {
            "change_points": change_points,
            "n_segments": best_k + 1,
            "probabilities": np.exp(np.clip(log_probs[best_k, :] - np.max(log_probs[best_k, :]), -700, 700)),
        }

    def sequential_bayes(
        self, prior: PosteriorResult, new_data: np.ndarray, known_sigma: float = 0.5,
    ) -> PosteriorResult:
        result = self.conjugate_normal_normal(
            new_data, prior_mu=prior.mean, prior_sigma=prior.std, known_sigma=known_sigma
        )
        return result

    def hierarchical_model(
        self, groups_data: list[np.ndarray], prior_mu: float = 0.0, prior_sigma: float = 5.0,
    ) -> dict:
        n_groups = len(groups_data)
        group_means = np.array([np.mean(g) for g in groups_data])
        group_stds = np.array([np.std(g, ddof=1) + 1e-10 for g in groups_data])
        group_sizes = np.array([len(g) for g in groups_data])

        overall_mean = np.mean(group_means)
        between_var = np.var(group_means, ddof=1) if n_groups > 1 else 1.0
        within_var = np.mean(group_stds ** 2)

        shrinkage = between_var / (between_var + within_var / group_sizes)
        shrunk_means = shrinkage * group_means + (1 - shrinkage) * overall_mean

        posterior_vars = 1.0 / (group_sizes / within_var + 1.0 / (between_var + 1e-10))
        posterior_stds = np.sqrt(posterior_vars)

        return {
            "group_means": group_means.tolist(),
            "shrunk_means": shrunk_means.tolist(),
            "overall_mean": float(overall_mean),
            "between_group_variance": float(between_var),
            "within_group_variance": float(within_var),
            "shrinkage_factors": shrinkage.tolist(),
            "posterior_stds": posterior_stds.tolist(),
            "credible_intervals": [
                (float(m - 1.96 * s), float(m + 1.96 * s))
                for m, s in zip(shrunk_means, posterior_stds)
            ],
        }

    def predict(
        self, data: np.ndarray, variable: str, domain: Domain,
        target_date: date, confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH,
        prior_mu: float = 0.0, prior_sigma: float = 1.0,
    ) -> Prediction:
        posterior = self.conjugate_normal_normal(
            data, prior_mu=prior_mu, prior_sigma=prior_sigma, known_sigma=float(np.std(data) + 1e-10)
        )
        z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645,
             ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence_level]

        return Prediction(
            domain=domain, variable=variable, value=round(posterior.mean, 4),
            confidence_interval=(round(posterior.mean - z * posterior.std, 4),
                                 round(posterior.mean + z * posterior.std, 4)),
            confidence_level=confidence_level, timestamp=datetime.now(),
            target_date=target_date, model_name="bayesian_normal",
            weights={"prior_strength": self.prior_strength, "bayes_factor": posterior.bayes_factor},
        )

    def _compute_bayes_factor(
        self, data: np.ndarray, prior_mu: float, prior_sigma: float,
        posterior_mu: float, posterior_sigma: float, known_sigma: float,
    ) -> float:
        n = len(data)
        xbar = np.mean(data)
        se = known_sigma / np.sqrt(n)
        null_ll = np.sum(sp_stats.norm.logpdf(data, prior_mu, known_sigma))
        alt_se = np.sqrt(se ** 2 + posterior_sigma ** 2)
        alt_ll = np.sum(sp_stats.norm.logpdf(data, posterior_mu, known_sigma))
        log_bf = alt_ll - null_ll
        if log_bf > 700:
            return float('inf')
        elif log_bf < -700:
            return 0.0
        return float(np.exp(log_bf))