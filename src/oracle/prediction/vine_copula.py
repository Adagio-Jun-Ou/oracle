from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class VineCopulaResult:
    log_likelihood: float
    aic: float
    bic: float
    n_pairs: int
    pair_copulas: list
    robjects: list


@dataclass
class VineSimulationResult:
    samples: np.ndarray
    n_samples: int
    n_dim: int
    method: str


@dataclass
class TailDepMapResult:
    upper_tail: dict
    lower_tail: dict
    kendall_tau: dict
    spearman_rho: dict


class VineCopulaModel:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._fitted = False
        self._pair_params = {}

    @staticmethod
    def _clayton_cdf(u: float, v: float, theta: float) -> float:
        if abs(theta) < 1e-10:
            return u * v
        return max(1e-15, (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta))

    @staticmethod
    def _gumbel_cdf(u: float, v: float, theta: float) -> float:
        if abs(theta) < 1e-10:
            return u * v
        lu = (-np.log(max(u, 1e-15))) ** theta
        lv = (-np.log(max(v, 1e-15))) ** theta
        return np.exp(-((lu + lv) ** (1 / theta)))

    @staticmethod
    def _frank_cdf(u: float, v: float, theta: float) -> float:
        if abs(theta) < 1e-10:
            return u * v
        a = np.exp(-theta * u) - 1
        b = np.exp(-theta * v) - 1
        c = np.exp(-theta) - 1
        return -1 / theta * np.log(1 + a * b / c)

    @staticmethod
    def _gaussian_cdf(u: float, v: float, rho: float) -> float:
        from scipy import stats as sp_stats
        z1 = sp_stats.norm.ppf(max(u, 1e-10))
        z2 = sp_stats.norm.ppf(max(v, 1e-10))
        return float(sp_stats.multivariate_normal.cdf([z1, z2], mean=[0, 0], cov=[[1, rho], [rho, 1]]))

    @staticmethod
    def _student_t_cdf(u: float, v: float, rho: float, df: float = 4) -> float:
        from scipy import stats as sp_stats
        t1 = sp_stats.t.ppf(max(u, 1e-10), df)
        t2 = sp_stats.t.ppf(max(v, 1e-10), df)
        cov = np.array([[1, rho], [rho, 1]]) * (df - 2) / df
        cov = np.maximum(cov, np.eye(2) * 1e-6)
        try:
            return float(sp_stats.multivariate_normal.cdf([t1, t2], mean=[0, 0], cov=cov))
        except Exception:
            return u * v

    @staticmethod
    def _kendall_tau_from_matrix(data: np.ndarray, i: int, j: int) -> float:
        n = data.shape[0]
        if n < 4:
            return 0.0
        x = data[:, i]
        y = data[:, j]
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        concordant = 0
        discordant = 0
        max_pairs = min(n * (n - 1) // 2, 10000)
        rng = np.random.RandomState(42)
        for _ in range(max_pairs):
            a, b = rng.randint(0, n, 2)
            if a == b:
                continue
            sign_xy = (x[a] - x[b]) * (y[a] - y[b])
            if sign_xy > 0:
                concordant += 1
            elif sign_xy < 0:
                discordant += 1
        total = concordant + discordant
        if total == 0:
            return 0.0
        return (concordant - discordant) / total

    def fit(self, data: np.ndarray, family: str = "auto") -> VineCopulaResult:
        n, d = data.shape
        pair_copulas = []
        log_lik = 0.0
        rng = np.random.RandomState(self.seed)
        copula_families = ["clayton", "gumbel", "frank", "gaussian"]
        for i in range(d):
            for j in range(i + 1, d):
                tau = self._kendall_tau_from_matrix(data, i, j)
                if abs(tau) < 0.01:
                    self._pair_params[(i, j)] = {"family": "independence", "theta": 0.0}
                    pair_copulas.append({"i": i, "j": j, "family": "independence", "theta": 0.0, "tau": float(tau)})
                    continue
                best_family = "frank"
                best_theta = 0.0
                best_ll = -np.inf
                for fam in copula_families:
                    if fam == "clayton":
                        if tau > 0:
                            theta = max(2 * tau / (1 - tau), 0.01)
                            theta = min(theta, 20.0)
                        else:
                            continue
                    elif fam == "gumbel":
                        if tau > 0:
                            theta = max(1 / (1 - tau), 1.01)
                            theta = min(theta, 20.0)
                        else:
                            continue
                    elif fam == "frank":
                        theta = 4 * tau / (1 - abs(tau)) if abs(tau) > 0.01 else 0.5
                        theta = np.clip(theta, -30, 30)
                    elif fam == "gaussian":
                        theta = np.clip(tau * 1.5, -0.95, 0.95)
                    else:
                        continue
                    ll = self._compute_pair_log_likelihood(data[:, i], data[:, j], fam, theta)
                    if ll > best_ll:
                        best_ll = ll
                        best_family = fam
                        best_theta = theta
                self._pair_params[(i, j)] = {"family": best_family, "theta": best_theta}
                pair_copulas.append({"i": i, "j": j, "family": best_family, "theta": float(best_theta), "tau": float(tau)})
                log_lik += best_ll
        k = len(pair_copulas)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik
        self._fitted = True
        return VineCopulaResult(
            log_likelihood=float(log_lik),
            aic=float(aic),
            bic=float(bic),
            n_pairs=k,
            pair_copulas=pair_copulas,
            robjects=[],
        )

    def _compute_pair_log_likelihood(self, u: np.ndarray, v: np.ndarray,
                                       family: str, theta: float) -> float:
        u_ranks = (np.argsort(np.argsort(u)) + 1) / (len(u) + 1)
        v_ranks = (np.argsort(np.argsort(v)) + 1) / (len(v) + 1)
        u_ranks = np.clip(u_ranks, 0.001, 0.999)
        v_ranks = np.clip(v_ranks, 0.001, 0.999)
        n = len(u_ranks)
        log_lik = 0.0
        for k in range(0, n, max(1, n // 50)):
            ui, vi = float(u_ranks[k]), float(v_ranks[k])
            if family == "clayton" and theta > 0:
                c_val = self._clayton_cdf(ui, vi, theta)
                prob = max(c_val, 1e-10)
                log_lik += np.log(prob)
            elif family == "gumbel":
                c_val = self._gumbel_cdf(ui, vi, theta)
                prob = max(c_val, 1e-10)
                log_lik += np.log(prob)
            elif family == "frank":
                c_val = self._frank_cdf(ui, vi, theta)
                prob = max(c_val, 1e-10)
                log_lik += np.log(prob)
            elif family == "gaussian":
                log_lik += -0.5 * theta ** 2 * (1 - theta ** 2)
        n_sampled = len(range(0, n, max(1, n // 50)))
        return float(log_lik * (n / max(n_sampled, 1)))

    def simulate(self, n_samples: int = 1000) -> VineSimulationResult:
        if not self._fitted or not self._pair_params:
            return VineSimulationResult(
                samples=np.zeros((n_samples, 1)),
                n_samples=n_samples, n_dim=1, method="vine_copula",
            )
        rng = np.random.RandomState(self.seed)
        d = max(max(pair) for pair in self._pair_params) + 1
        samples = rng.normal(0, 1, (n_samples, d))
        for (i, j), params in self._pair_params.items():
            family = params["family"]
            theta = params["theta"]
            if family == "independence" or abs(theta) < 1e-10:
                pass
            elif family == "clayton":
                u = rng.uniform(0, 1, n_samples)
                v_dependent = (u ** (-theta) + rng.uniform(0, 1, n_samples) ** (-theta) - 1) ** (-1 / theta)
                samples[:, i] = samples[:, i] + theta * v_dependent
            elif family == "gaussian":
                rho = np.clip(theta, -0.95, 0.95)
                samples[:, j] = rho * samples[:, i] + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n_samples)
        return VineSimulationResult(
            samples=samples, n_samples=n_samples, n_dim=d, method="vine_copula",
        )

    @staticmethod
    def tail_dependence_map(data: np.ndarray, threshold: float = 0.9) -> TailDepMapResult:
        n, d = data.shape
        u_data = np.zeros_like(data)
        for col in range(d):
            ranks = (np.argsort(np.argsort(data[:, col])) + 1) / (n + 1)
            u_data[:, col] = ranks
        upper_tail = {}
        lower_tail = {}
        kendall_tau = {}
        spearman_rho = {}
        for i in range(d):
            for j in range(i + 1, d):
                key = f"var_{i}_var_{j}"
                u_thresh = np.quantile(u_data[:, i], threshold)
                v_thresh = np.quantile(u_data[:, j], threshold)
                upper_joint = np.sum((u_data[:, i] > u_thresh) & (u_data[:, j] > v_thresh))
                upper_marginal = np.sum(u_data[:, i] > u_thresh)
                upper_tail[key] = float(upper_joint / max(upper_marginal, 1))
                low_u_thresh = np.quantile(u_data[:, i], 1 - threshold)
                low_v_thresh = np.quantile(u_data[:, j], 1 - threshold)
                lower_joint = np.sum((u_data[:, i] < low_u_thresh) & (u_data[:, j] < low_v_thresh))
                lower_marginal = np.sum(u_data[:, i] < low_u_thresh)
                lower_tail[key] = float(lower_joint / max(lower_marginal, 1))
                if np.std(data[:, i]) > 1e-10 and np.std(data[:, j]) > 1e-10:
                    from scipy import stats as sp_stats
                    kendall_tau[key] = float(sp_stats.kendalltau(data[:, i], data[:, j]).correlation)
                else:
                    kendall_tau[key] = 0.0
                from scipy import stats as sp_stats
                spearman_rho[key] = float(sp_stats.spearmanr(data[:, i], data[:, j]).correlation)
        return TailDepMapResult(
            upper_tail=upper_tail,
            lower_tail=lower_tail,
            kendall_tau=kendall_tau,
            spearman_rho=spearman_rho,
        )

    @staticmethod
    def conditional_simulation(copula_type: str = "clayton", theta: float = 2.0,
                               n_samples: int = 1000, condition_var: int = 0,
                               condition_value: float = 0.5) -> dict:
        rng = np.random.RandomState(42)
        if copula_type == "clayton" and theta > 0:
            u_cond = np.full(n_samples, condition_value)
            v_random = rng.uniform(0, 1, n_samples)
            v_samples = (u_cond ** (-theta) + v_random ** (-theta) - 1) ** (-1 / theta)
            v_samples = np.clip(v_samples, 0.001, 0.999)
        elif copula_type == "gumbel" and theta > 1:
            u_cond = np.full(n_samples, condition_value)
            v_random = rng.uniform(0, 1, n_samples)
            lu = (-np.log(u_cond)) ** theta
            lv = (-np.log(np.clip(v_random, 0.001, 0.999))) ** theta
            v_samples = np.exp(-((lu + lv) ** (1 / theta)))
            v_samples = np.clip(v_samples, 0.001, 0.999)
        elif copula_type == "frank":
            u_cond = np.full(n_samples, condition_value)
            v_random = rng.uniform(0, 1, n_samples)
            a = 1 + (np.exp(-theta * u_cond) - 1) * (np.exp(-theta * v_random) - 1) / (np.exp(-theta) - 1)
            v_samples = -1 / theta * np.log(np.clip(a, 1e-10, None))
            v_samples = np.clip(v_samples, 0.001, 0.999)
        else:
            rho = np.clip(theta, -0.95, 0.95)
            v_samples = rng.normal(rho * condition_value, np.sqrt(1 - rho ** 2), n_samples)
            v_samples = (1 + sp_stats_norm_cdf(v_samples)) / 2
            v_samples = np.clip(v_samples, 0.001, 0.999)
        return {
            "conditioned_variable": int(condition_var),
            "condition_value": float(condition_value),
            "copula_type": copula_type,
            "theta": float(theta),
            "conditional_mean": float(np.mean(v_samples)),
            "conditional_std": float(np.std(v_samples)),
            "conditional_quantiles": {
                "5%": float(np.percentile(v_samples, 5)),
                "25%": float(np.percentile(v_samples, 25)),
                "50%": float(np.percentile(v_samples, 50)),
                "75%": float(np.percentile(v_samples, 75)),
                "95%": float(np.percentile(v_samples, 95)),
            },
            "n_samples": n_samples,
        }


def sp_stats_norm_cdf(x):
    import math
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(lambda v: math.erf(v / math.sqrt(2)))(x))