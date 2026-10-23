from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class GEVResult:
    shape: float
    scale: float
    location: float
    return_levels: dict
    method: str
    aic: float = 0.0
    log_likelihood: float = 0.0


@dataclass
class GPDRresult:
    shape: float
    scale: float
    threshold: float
    n_exceedances: int
    tail_index: float
    return_levels: dict
    method: str


@dataclass
class TailDependenceResult:
    upper_tail_dependence: float
    lower_tail_dependence: float
    copula_type: str
    kendall_tau: float


class ExtremeValueTheory:
    def __init__(self, seed: int = 42):
        self.seed = seed

    @staticmethod
    def gev_fit(data: np.ndarray, method: str = "mle") -> GEVResult:
        data = np.sort(data)
        n = len(data)
        mu = float(np.mean(data))
        sigma = float(np.std(data))
        shape = 0.1
        for _ in range(100):
            z = (data - mu) / max(sigma, 1e-10)
            if abs(shape) < 1e-8:
                g = np.exp(-z)
            else:
                t = np.maximum(1 + shape * z, 1e-10)
                g = np.maximum(t, 1e-10) ** (-1 / shape)
            ll = np.sum(np.log(g + 1e-10)) - n * np.log(max(sigma, 1e-10))
            if abs(shape) < 1e-8:
                grad_mu = np.sum(z - 1) / max(sigma, 1e-10)
                grad_sigma = -n / max(sigma, 1e-10)
            else:
                t = 1 + shape * z
                valid = t > 0
                grad_mu = np.sum((1 + shape) / (t[valid] + 1e-10) - 1) / max(sigma, 1e-10)
                grad_sigma = -n / max(sigma, 1e-10)
            mu += 0.001 * grad_mu * sigma
            sigma = max(abs(sigma + 0.001 * grad_sigma * sigma), 0.01)
        return_levels = {}
        for period in [10, 25, 50, 100]:
            if abs(shape) < 1e-8:
                rl = mu - sigma * np.log(-np.log(1 - 1 / period))
            else:
                rl = mu + sigma / shape * ((-np.log(1 - 1 / period)) ** (-shape) - 1)
            return_levels[f"{period}_year"] = float(rl)
        log_likelihood = float(-n * np.log(max(sigma, 1e-10)) - np.sum(np.maximum(1 + shape * (data - mu) / max(sigma, 1e-10), 1e-10)))
        aic = 2 * 3 - 2 * log_likelihood
        return GEVResult(
            shape=float(shape),
            scale=float(sigma),
            location=float(mu),
            return_levels=return_levels,
            method="gev_mle",
            aic=float(aic),
            log_likelihood=log_likelihood,
        )

    @staticmethod
    def gpd_fit(data: np.ndarray, threshold_quantile: float = 0.95) -> GPDRresult:
        threshold = float(np.quantile(data, threshold_quantile))
        exceedances = data[data > threshold] - threshold
        n_exceed = len(exceedances)
        if n_exceed < 5:
            return GPDRresult(
                shape=0.0, scale=0.0, threshold=threshold,
                n_exceedances=n_exceed, tail_index=0.0,
                return_levels={}, method="gpd_insufficient",
            )
        mean_exc = float(np.mean(exceedances))
        variance_exc = float(np.var(exceedances))
        if variance_exc < 1e-10:
            shape = 0.0
            scale = mean_exc
        else:
            shape = 0.5 * (1 - mean_exc ** 2 / variance_exc)
            scale = mean_exc * (1 - shape) / 2
        n = len(data)
        return_levels = {}
        for period in [10, 25, 50, 100]:
            p = 1 / period
            m = n / max(n_exceed, 1)
            if abs(shape) < 1e-8:
                rl = threshold + scale * m * p
            else:
                rl = threshold + scale / shape * ((m * p) ** shape - 1)
            return_levels[f"{period}_year"] = float(rl)
        tail_index = 1 / shape if abs(shape) > 1e-8 else float('inf')
        return GPDRresult(
            shape=float(shape),
            scale=float(scale),
            threshold=float(threshold),
            n_exceedances=n_exceed,
            tail_index=float(tail_index) if tail_index != float('inf') else 999.0,
            return_levels=return_levels,
            method="gpd_pwm",
        )

    @staticmethod
    def pot_analysis(data: np.ndarray, threshold_quantile: float = 0.95,
                    n_simulations: int = 10000, seed: int = 42) -> dict:
        threshold = float(np.quantile(data, threshold_quantile))
        exceedances = data[data > threshold] - threshold
        n_exceed = len(exceedances)
        if n_exceed < 5:
            return {"error": "insufficient_exceedances", "n_exceedances": n_exceed}
        gpd = ExtremeValueTheory.gpd_fit(data, threshold_quantile)
        rng = np.random.RandomState(seed)
        sim_maxes = np.zeros(n_simulations)
        n_total = len(data)
        rate = n_exceed / n_total
        for i in range(n_simulations):
            n_events = rng.poisson(rate * n_total)
            if n_events > 0 and gpd.scale > 0:
                if abs(gpd.shape) < 1e-8:
                    sim_exc = rng.exponential(gpd.scale, n_events)
                else:
                    sim_exc = rng.exponential(gpd.scale, n_events)
                    sim_exc = gpd.scale / gpd.shape * ((1 - rng.uniform(0, 1, n_events)) ** (-gpd.shape) - 1)
                sim_maxes[i] = threshold + np.max(sim_exc)
            else:
                sim_maxes[i] = threshold
        return {
            "threshold": threshold,
            "n_exceedances": n_exceed,
            "exceedance_rate": float(rate),
            "gpd_shape": gpd.shape,
            "gpd_scale": gpd.scale,
            "return_levels": gpd.return_levels,
            "simulated_mean_max": float(np.mean(sim_maxes)),
            "simulated_95_max": float(np.percentile(sim_maxes, 95)),
            "simulated_99_max": float(np.percentile(sim_maxes, 99)),
            "method": "pot_simulation",
        }

    @staticmethod
    def tail_dependence(x: np.ndarray, y: np.ndarray,
                        threshold: float = 0.95) -> TailDependenceResult:
        n = len(x)
        x_thresh = np.quantile(x, threshold)
        y_thresh = np.quantile(y, threshold)
        x_exceed = x > x_thresh
        y_exceed = y > y_thresh
        n_xy = float(np.sum(x_exceed & y_exceed))
        n_x = float(np.sum(x_exceed))
        n_y = float(np.sum(y_exceed))
        upper_td = min(n_xy / max(n_x, 1), 1.0) if n_x > 0 else 0.0
        x_below = x < np.quantile(x, 1 - threshold)
        y_below = y < np.quantile(y, 1 - threshold)
        n_xy_low = float(np.sum(x_below & y_below))
        n_x_low = float(np.sum(x_below))
        lower_td = min(n_xy_low / max(n_x_low, 1), 1.0) if n_x_low > 0 else 0.0
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, min(i + 100, n)):
                if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                    concordant += 1
                else:
                    discordant += 1
        total = concordant + discordant
        kendall_tau = (concordant - discordant) / max(total, 1)
        return TailDependenceResult(
            upper_tail_dependence=float(upper_td),
            lower_tail_dependence=float(lower_td),
            copula_type="empirical",
            kendall_tau=float(kendall_tau),
        )

    @staticmethod
    def hill_estimator(data: np.ndarray, k: Optional[int] = None) -> dict:
        sorted_data = np.sort(data)[::-1]
        n = len(sorted_data)
        if k is None:
            k = min(int(np.sqrt(n)), n // 2)
        if k < 2 or k >= n:
            k = max(2, n // 4)
        tail = sorted_data[:k]
        hill_est = float(np.mean(np.log(tail / sorted_data[k])))
        hill_data = []
        for ki in range(2, min(k, n)):
            tail_ki = sorted_data[:ki]
            gamma = float(np.mean(np.log(tail_ki / sorted_data[ki])))
            se = float(np.std(np.log(tail_ki / sorted_data[ki]))) / np.sqrt(ki) if ki > 1 else 0
            hill_data.append({"k": ki, "tail_index": gamma, "se": se})
        return {
            "tail_index": hill_est,
            "k": k,
            "shape_parameter": 1.0 / hill_est if abs(hill_est) > 1e-10 else float('inf'),
            "hill_plot_data": hill_data[:50],
            "method": "hill",
        }

    @staticmethod
    def block_maxima(data: np.ndarray, block_size: int = 21) -> GEVResult:
        n = len(data)
        n_blocks = max(n // block_size, 1)
        maxima = np.array([np.max(data[i * block_size:(i + 1) * block_size])
                          for i in range(n_blocks) if i * block_size < n])
        return ExtremeValueTheory.gev_fit(maxima)