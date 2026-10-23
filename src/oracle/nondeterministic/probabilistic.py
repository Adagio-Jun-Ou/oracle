from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy import stats


@dataclass
class ProbabilisticForecast:
    point_forecast: float
    lower_50: float
    upper_50: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float
    median: float
    mean: float
    std: float
    skewness: float
    kurtosis: float
    distribution: str
    crps: Optional[float] = None
    pit: Optional[float] = None
    calibration_score: Optional[float] = None


@dataclass
class CRPSResult:
    crps: float
    crpss: float
    reliability: float
    resolution: float
    uncertainty: float
    brier_score: float
    log_score: float
    quantile_scores: dict = field(default_factory=dict)


class ProbabilisticForecaster:
    def __init__(self, seed: int = 42):
        self.seed = seed

    @staticmethod
    def crps_gaussian(observation: float, mean: float, std: float) -> float:
        z = (observation - mean) / std if std > 0 else 0.0
        return float(std * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)))

    @staticmethod
    def crps_empirical(observation: float, forecast_samples: np.ndarray) -> float:
        sorted_samples = np.sort(forecast_samples)
        n = len(sorted_samples)
        cdf_at_obs = float(np.searchsorted(sorted_samples, observation) / n)
        ecs = float(np.mean(np.abs(sorted_samples - observation)))
        below = sorted_samples[sorted_samples <= observation]
        above_cdf = 1.0 - cdf_at_obs
        below_cdf = cdf_at_obs
        return float(ecs - np.mean(sorted_samples) * (below_cdf - above_cdf) + observation * (below_cdf - above_cdf)) if n > 0 else 0.0

    @staticmethod
    def crps_quantiles(observation: float, quantiles: dict) -> float:
        q_levels = sorted(quantiles.keys())
        q_values = [quantiles[q] for q in q_levels]
        crps = 0.0
        for i, q in enumerate(q_levels):
            indicator = 1.0 if observation <= q_values[i] else 0.0
            penalty = (indicator - q) ** 2
            if i < len(q_levels) - 1:
                weight = q_levels[i + 1] - q
            else:
                weight = 1.0 - q
            crps += penalty * weight
        return float(crps / len(q_levels)) if len(q_levels) > 0 else 0.0

    @staticmethod
    def brier_score(observation: float, probability_forecast: float) -> float:
        return float((probability_forecast - observation) ** 2)

    @staticmethod
    def log_score(observation: float, mean: float, std: float) -> float:
        if std <= 0:
            return 0.0
        return float(-stats.norm.logpdf(observation, loc=mean, scale=std))

    @staticmethod
    def pit_histogram(observations: np.ndarray, means: np.ndarray,
                       stds: np.ndarray, n_bins: int = 10) -> dict:
        n = len(observations)
        pit_values = []
        for i in range(n):
            if stds[i] > 0:
                pit = stats.norm.cdf(observations[i], loc=means[i], scale=stds[i])
                pit_values.append(float(pit))
            else:
                pit_values.append(0.5)
        hist, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))
        expected = n / n_bins
        chi_sq = float(np.sum((hist - expected) ** 2 / expected)) if expected > 0 else 0.0
        p_value = float(1 - stats.chi2.cdf(chi_sq, df=n_bins - 1))
        uniform = abs(hist - expected) < 2 * np.sqrt(expected)
        calibrated = float(np.mean(uniform)) > 0.5
        return {
            "pit_values": pit_values,
            "histogram": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "expected_per_bin": float(expected),
            "chi_squared": float(chi_sq),
            "p_value": p_value,
            "calibrated": bool(calibrated),
            "overforecasting": bool(np.mean(pit_values) < 0.45),
            "underforecasting": bool(np.mean(pit_values) > 0.55),
        }

    def evaluate_forecasts(self, observations: np.ndarray, means: np.ndarray,
                            stds: np.ndarray, baseline_mean: Optional[float] = None) -> CRPSResult:
        n = len(observations)
        crps_values = []
        for i in range(n):
            crps = self.crps_gaussian(observations[i], means[i], stds[i])
            crps_values.append(crps)
        avg_crps = float(np.mean(crps_values))
        if baseline_mean is None:
            baseline_mean = float(np.mean(means))
        baseline_std = float(np.mean(stds))
        baseline_crps_list = []
        for i in range(n):
            baseline_crps_list.append(self.crps_gaussian(observations[i], baseline_mean, baseline_std))
        baseline_avg_crps = float(np.mean(baseline_crps_list))
        crpss = 1.0 - avg_crps / baseline_avg_crps if baseline_avg_crps > 0 else 0.0
        reliability = float(np.mean([(m - o) ** 2 for m, o in zip(means, observations)]))
        spread_skill = float(np.mean([s ** 2 for s in stds]))
        resolution = float(np.var(means))
        uncertainty = float(np.var(observations))
        brier_scores = []
        log_scores = []
        for i in range(n):
            event = 1.0 if observations[i] > means[i] else 0.0
            prob = 1.0 - stats.norm.cdf(means[i], loc=means[i], scale=stds[i]) if stds[i] > 0 else 0.5
            brier_scores.append(self.brier_score(event, prob))
            log_scores.append(self.log_score(observations[i], means[i], stds[i]))
        quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantile_scores = {}
        for q in quantile_levels:
            q_forecasts = stats.norm.ppf(q, loc=means, scale=stds)
            q_score = float(np.mean([
                2 * (q if observations[i] <= q_forecasts[i] else (1 - q)) * abs(observations[i] - q_forecasts[i])
                for i in range(n)
            ]))
            quantile_scores[f"q_{int(q*100)}"] = q_score
        return CRPSResult(
            crps=float(avg_crps),
            crpss=float(crpss),
            reliability=float(reliability),
            resolution=float(resolution),
            uncertainty=float(uncertainty),
            brier_score=float(np.mean(brier_scores)),
            log_score=float(np.mean(log_scores)),
            quantile_scores=quantile_scores,
        )

    def forecast_gaussian(self, mean: float, std: float,
                           observation: Optional[float] = None) -> ProbabilisticForecast:
        median = mean
        skewness = 0.0
        kurtosis = 3.0
        lower_50 = float(stats.norm.ppf(0.25, loc=mean, scale=std))
        upper_50 = float(stats.norm.ppf(0.75, loc=mean, scale=std))
        lower_80 = float(stats.norm.ppf(0.10, loc=mean, scale=std))
        upper_80 = float(stats.norm.ppf(0.90, loc=mean, scale=std))
        lower_95 = float(stats.norm.ppf(0.025, loc=mean, scale=std))
        upper_95 = float(stats.norm.ppf(0.975, loc=mean, scale=std))
        crps = None
        pit = None
        calibration_score = None
        if observation is not None:
            crps = self.crps_gaussian(observation, mean, std)
            pit = float(stats.norm.cdf(observation, loc=mean, scale=std))
            calibration_score = 1.0 - abs(pit - 0.5) * 2
        return ProbabilisticForecast(
            point_forecast=float(mean), lower_50=lower_50, upper_50=upper_50,
            lower_80=lower_80, upper_80=upper_80, lower_95=lower_95, upper_95=upper_95,
            median=float(median), mean=float(mean), std=float(std),
            skewness=skewness, kurtosis=kurtosis, distribution="gaussian",
            crps=crps, pit=pit, calibration_score=calibration_score,
        )

    def forecast_t_distribution(self, mean: float, std: float, df: float = 5.0,
                                 observation: Optional[float] = None) -> ProbabilisticForecast:
        t_std = std * np.sqrt(df / (df - 2)) if df > 2 else std
        median = mean
        skewness = 0.0
        kurtosis = 3.0 + 6.0 / (df - 4) if df > 4 else 6.0
        lower_50 = float(mean + stats.t.ppf(0.25, df=df) * std)
        upper_50 = float(mean + stats.t.ppf(0.75, df=df) * std)
        lower_80 = float(mean + stats.t.ppf(0.10, df=df) * std)
        upper_80 = float(mean + stats.t.ppf(0.90, df=df) * std)
        lower_95 = float(mean + stats.t.ppf(0.025, df=df) * std)
        upper_95 = float(mean + stats.t.ppf(0.975, df=df) * std)
        crps = None
        pit = None
        calibration_score = None
        if observation is not None:
            pit = float(stats.t.cdf((observation - mean) / std, df=df))
            crps = float(std * self._t_crps_approx((observation - mean) / std, df))
            calibration_score = 1.0 - abs(pit - 0.5) * 2
        return ProbabilisticForecast(
            point_forecast=float(mean), lower_50=lower_50, upper_50=upper_50,
            lower_80=lower_80, upper_80=upper_80, lower_95=lower_95, upper_95=upper_95,
            median=float(median), mean=float(mean), std=float(t_std),
            skewness=skewness, kurtosis=float(kurtosis), distribution="student_t",
            crps=crps, pit=pit, calibration_score=calibration_score,
        )

    @staticmethod
    def _t_crps_approx(z: float, df: float) -> float:
        return float(abs(z) * (2 * stats.t.cdf(z, df=df) - 1) +
                     2 * stats.t.pdf(z, df=df) * (df + z ** 2) / (df - 1))

    def forecast_skewnorm(self, mean: float, std: float, skewness: float = 0.0,
                           observation: Optional[float] = None) -> ProbabilisticForecast:
        alpha = skewness * 4.0 / (1 + skewness ** 2) ** 0.5 if abs(skewness) < 10 else 4.0
        delta = alpha / np.sqrt(1 + alpha ** 2)
        omega = std / np.sqrt(1 - 2 * delta ** 2 / np.pi)
        xi = mean - omega * delta * np.sqrt(2 / np.pi)
        median = xi + omega * (delta * np.sqrt(2 / np.pi) + stats.norm.pdf(delta) * 0.5)
        actual_mean = xi + omega * delta * np.sqrt(2 / np.pi)
        actual_std = omega * np.sqrt(1 - 2 * delta ** 2 / np.pi)
        actual_kurtosis = 2 * (delta ** 4) * (np.pi - 3) ** 2 / (2 * np.pi - delta ** 2 * (3 * np.pi - 8)) ** 2 + 3 if abs(delta) < 0.99 else 3.0
        lower_50 = float(stats.norm.ppf(0.25, loc=mean, scale=std) + skewness * 0.3 * std)
        upper_50 = float(stats.norm.ppf(0.75, loc=mean, scale=std) + skewness * 0.3 * std)
        lower_80 = float(stats.norm.ppf(0.10, loc=mean, scale=std) + skewness * 0.5 * std)
        upper_80 = float(stats.norm.ppf(0.90, loc=mean, scale=std) + skewness * 0.5 * std)
        lower_95 = float(stats.norm.ppf(0.025, loc=mean, scale=std) + skewness * 0.8 * std)
        upper_95 = float(stats.norm.ppf(0.975, loc=mean, scale=std) + skewness * 0.8 * std)
        crps = None
        pit_val = None
        calibration_score = None
        if observation is not None:
            crps = self.crps_gaussian(observation, mean, std)
            pit_val = float(stats.norm.cdf(observation, loc=mean, scale=std))
            calibration_score = 1.0 - abs(pit_val - 0.5) * 2
        return ProbabilisticForecast(
            point_forecast=float(mean), lower_50=lower_50, upper_50=upper_50,
            lower_80=lower_80, upper_80=upper_80, lower_95=lower_95, upper_95=upper_95,
            median=float(median), mean=float(actual_mean), std=float(actual_std),
            skewness=skewness, kurtosis=float(actual_kurtosis), distribution="skew_normal",
            crps=crps, pit=pit_val, calibration_score=calibration_score,
        )

    def ensemble_probabilistic(self, forecasts: list[ProbabilisticForecast],
                                weights: Optional[list[float]] = None,
                                observation: Optional[float] = None) -> ProbabilisticForecast:
        n = len(forecasts)
        if n == 0:
            return self.forecast_gaussian(0.0, 1.0, observation)
        if weights is None:
            weights = [1.0 / n] * n
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        mean = sum(f.mean * w for f, w in zip(forecasts, weights))
        variance = sum(w * (f.std ** 2 + (f.mean - mean) ** 2) for f, w in zip(forecasts, weights))
        std = float(np.sqrt(max(variance, 1e-10)))
        median = sum(f.median * w for f, w in zip(forecasts, weights))
        skew = sum(w * f.skewness for f, w in zip(forecasts, weights))
        kurt = sum(w * f.kurtosis for f, w in zip(forecasts, weights))
        lower_50 = sum(f.lower_50 * w for f, w in zip(forecasts, weights))
        upper_50 = sum(f.upper_50 * w for f, w in zip(forecasts, weights))
        lower_80 = sum(f.lower_80 * w for f, w in zip(forecasts, weights))
        upper_80 = sum(f.upper_80 * w for f, w in zip(forecasts, weights))
        lower_95 = sum(f.lower_95 * w for f, w in zip(forecasts, weights))
        upper_95 = sum(f.upper_95 * w for f, w in zip(forecasts, weights))
        crps = None
        pit_val = None
        calibration = None
        if observation is not None:
            crps = self.crps_gaussian(observation, mean, std)
            pit_val = float(stats.norm.cdf(observation, loc=mean, scale=std))
            calibration = 1.0 - abs(pit_val - 0.5) * 2
        return ProbabilisticForecast(
            point_forecast=float(mean), lower_50=float(lower_50), upper_50=float(upper_50),
            lower_80=float(lower_80), upper_80=float(upper_80),
            lower_95=float(lower_95), upper_95=float(upper_95),
            median=float(median), mean=float(mean), std=float(std),
            skewness=float(skew), kurtosis=float(kurt), distribution="ensemble",
            crps=crps, pit=pit_val, calibration_score=calibration,
        )

    @staticmethod
    def calibration_curve(observed: np.ndarray, predicted_probs: np.ndarray,
                          n_bins: int = 10) -> dict:
        bins = np.linspace(0, 1, n_bins + 1)
        calibration_data = []
        for i in range(n_bins):
            mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
            if mask.sum() > 0:
                observed_freq = float(np.mean(observed[mask]))
                predicted_mean = float(np.mean(predicted_probs[mask]))
                count = int(mask.sum())
                calibration_data.append({
                    "predicted": predicted_mean,
                    "observed": observed_freq,
                    "count": count,
                    "bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                })
        ece = float(np.mean([abs(d["predicted"] - d["observed"]) * d["count"]
                             for d in calibration_data])) / max(1, sum(d["count"] for d in calibration_data))
        mce = float(max(abs(d["predicted"] - d["observed"]) for d in calibration_data)) if calibration_data else 0.0
        return {
            "calibration_data": calibration_data,
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "n_bins": n_bins,
        }