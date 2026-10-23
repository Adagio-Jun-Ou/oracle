import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Callable
from oracle.core.types import Domain, Prediction, ConfidenceLevel


@dataclass
class SimulationResult:
    paths: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    percentiles: dict
    n_simulations: int
    n_steps: int
    model_type: str = ""
    metadata: dict = field(default_factory=dict)

    def probability_above(self, threshold: float) -> float:
        return float(np.mean(self.paths[:, -1] > threshold))

    def probability_below(self, threshold: float) -> float:
        return float(np.mean(self.paths[:, -1] < threshold))

    def value_at_risk(self, confidence: float = 0.95) -> float:
        return float(np.percentile(self.paths[:, -1], (1 - confidence) * 100))

    def expected_shortfall(self, confidence: float = 0.95) -> float:
        var = self.value_at_risk(confidence)
        return float(np.mean(self.paths[self.paths[:, -1] <= var, -1]))

    def max_drawdown(self) -> float:
        cummax = np.maximum.accumulate(self.paths, axis=1)
        drawdown = (cummax - self.paths) / np.maximum(cummax, 1e-10)
        return float(np.max(drawdown))


class MonteCarloEngine:
    def __init__(self, n_simulations: int = 10000, seed: int = 42):
        self.n_simulations = n_simulations
        self.seed = seed

    def geometric_brownian_motion(
        self, s0: float, mu: float, sigma: float, n_steps: int = 252,
        dt: float = 1.0 / 252, n_sims: Optional[int] = None,
    ) -> SimulationResult:
        n = n_sims or self.n_simulations
        rng = np.random.default_rng(self.seed)
        z = rng.standard_normal((n, n_steps))
        increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        log_returns = np.cumsum(increments, axis=1)
        paths = s0 * np.exp(np.column_stack([np.zeros(n), log_returns]))

        return SimulationResult(
            paths=paths,
            mean=np.mean(paths, axis=0),
            std=np.std(paths, axis=0),
            percentiles={
                "p01": np.percentile(paths, 1, axis=0),
                "p05": np.percentile(paths, 5, axis=0),
                "p25": np.percentile(paths, 25, axis=0),
                "p50": np.percentile(paths, 50, axis=0),
                "p75": np.percentile(paths, 75, axis=0),
                "p95": np.percentile(paths, 95, axis=0),
                "p99": np.percentile(paths, 99, axis=0),
            },
            n_simulations=n, n_steps=n_steps,
            model_type="gbm",
            metadata={"s0": s0, "mu": mu, "sigma": sigma},
        )

    def mean_reverting(
        self, x0: float, theta: float, mu: float, sigma: float,
        n_steps: int = 252, dt: float = 1.0 / 252, n_sims: Optional[int] = None,
    ) -> SimulationResult:
        n = n_sims or self.n_simulations
        rng = np.random.default_rng(self.seed)
        paths = np.zeros((n, n_steps + 1))
        paths[:, 0] = x0

        for t in range(1, n_steps + 1):
            z = rng.standard_normal(n)
            dx = theta * (mu - paths[:, t - 1]) * dt + sigma * np.sqrt(dt) * z
            paths[:, t] = paths[:, t - 1] + dx

        return SimulationResult(
            paths=paths, mean=np.mean(paths, axis=0), std=np.std(paths, axis=0),
            percentiles={
                "p01": np.percentile(paths, 1, axis=0), "p05": np.percentile(paths, 5, axis=0),
                "p25": np.percentile(paths, 25, axis=0), "p50": np.percentile(paths, 50, axis=0),
                "p75": np.percentile(paths, 75, axis=0), "p95": np.percentile(paths, 95, axis=0),
                "p99": np.percentile(paths, 99, axis=0),
            },
            n_simulations=n, n_steps=n_steps,
            model_type="ou_mean_reverting",
            metadata={"x0": x0, "theta": theta, "mu": mu, "sigma": sigma},
        )

    def jump_diffusion(
        self, s0: float, mu: float, sigma: float, lambda_jump: float = 0.1,
        mu_jump: float = -0.02, sigma_jump: float = 0.05,
        n_steps: int = 252, dt: float = 1.0 / 252, n_sims: Optional[int] = None,
    ) -> SimulationResult:
        n = n_sims or self.n_simulations
        rng = np.random.default_rng(self.seed)
        paths = np.zeros((n, n_steps + 1))
        paths[:, 0] = s0

        for t in range(1, n_steps + 1):
            z = rng.standard_normal(n)
            n_jumps = rng.poisson(lambda_jump * dt, n)
            jump_sizes = np.sum(rng.normal(mu_jump, sigma_jump, (n, max(n_jumps.max() + 1, 1)))[:, :max(n_jumps.max() + 1, 1)] * (np.arange(max(n_jumps.max() + 1, 1))[np.newaxis, :] < n_jumps[:, np.newaxis]), axis=1) if n_jumps.max() > 0 else np.zeros(n)
            drift = (mu - 0.5 * sigma ** 2 - lambda_jump * (np.exp(mu_jump + 0.5 * sigma_jump ** 2) - 1)) * dt
            diffusion = sigma * np.sqrt(dt) * z
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion + jump_sizes)

        return SimulationResult(
            paths=paths, mean=np.mean(paths, axis=0), std=np.std(paths, axis=0),
            percentiles={
                "p05": np.percentile(paths, 5, axis=0), "p25": np.percentile(paths, 25, axis=0),
                "p50": np.percentile(paths, 50, axis=0), "p75": np.percentile(paths, 75, axis=0),
                "p95": np.percentile(paths, 95, axis=0),
            },
            n_simulations=n, n_steps=n_steps,
            model_type="merton_jump_diffusion",
            metadata={"s0": s0, "mu": mu, "sigma": sigma, "lambda_jump": lambda_jump},
        )

    def regime_switching(
        self, s0: float, n_steps: int = 252, dt: float = 1.0 / 252,
        regimes: Optional[list[dict]] = None, transition_matrix: Optional[np.ndarray] = None,
        n_sims: Optional[int] = None,
    ) -> SimulationResult:
        n = n_sims or self.n_simulations
        if regimes is None:
            regimes = [
                {"mu": 0.12, "sigma": 0.15, "name": "bull"},
                {"mu": -0.02, "sigma": 0.30, "name": "bear"},
                {"mu": 0.04, "sigma": 0.12, "name": "neutral"},
            ]
        if transition_matrix is None:
            transition_matrix = np.array([
                [0.90, 0.05, 0.05],
                [0.10, 0.80, 0.10],
                [0.10, 0.05, 0.85],
            ])

        n_regimes = len(regimes)
        rng = np.random.default_rng(self.seed)
        paths = np.zeros((n, n_steps + 1))
        paths[:, 0] = s0
        current_regime = rng.choice(n_regimes, n)
        regime_history = np.zeros((n, n_steps + 1), dtype=int)
        regime_history[:, 0] = current_regime

        for t in range(1, n_steps + 1):
            z = rng.standard_normal(n)
            probs = transition_matrix[current_regime]
            cumulative = np.cumsum(probs, axis=1)
            rands = rng.random(n)
            new_regime = np.argmax(cumulative > rands[:, np.newaxis], axis=1)
            current_regime = new_regime
            regime_history[:, t] = current_regime
            for k in range(n_regimes):
                mask = current_regime == k
                mu_r = regimes[k]["mu"]
                sigma_r = regimes[k]["sigma"]
                paths[mask, t] = paths[mask, t - 1] * np.exp(
                    (mu_r - 0.5 * sigma_r ** 2) * dt + sigma_r * np.sqrt(dt) * z[mask]
                )

        return SimulationResult(
            paths=paths, mean=np.mean(paths, axis=0), std=np.std(paths, axis=0),
            percentiles={
                "p05": np.percentile(paths, 5, axis=0), "p25": np.percentile(paths, 25, axis=0),
                "p50": np.percentile(paths, 50, axis=0), "p75": np.percentile(paths, 75, axis=0),
                "p95": np.percentile(paths, 95, axis=0),
            },
            n_simulations=n, n_steps=n_steps,
            model_type="regime_switching",
            metadata={"regimes": [r["name"] for r in regimes]},
        )

    def copula_simulation(
        self, marginals: list[dict], correlation_matrix: np.ndarray,
        n_steps: int = 252, n_sims: Optional[int] = None,
    ) -> SimulationResult:
        n = n_sims or self.n_simulations
        n_assets = len(marginals)
        rng = np.random.default_rng(self.seed)
        L = np.linalg.cholesky(correlation_matrix)
        z = rng.standard_normal((n, n_steps, n_assets))
        correlated = np.einsum("ij,tsj->tsi", L, z)
        all_paths = np.zeros((n_assets, n, n_steps + 1))

        for k, m in enumerate(marginals):
            all_paths[k, :, 0] = m.get("s0", 100)
            for t in range(1, n_steps + 1):
                dt = 1.0 / 252
                mu = m.get("mu", 0.08)
                sigma = m.get("sigma", 0.2)
                all_paths[k, :, t] = all_paths[k, :, t - 1] * np.exp(
                    (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * correlated[:, t - 1, k]
                )

        combined = np.sum(all_paths, axis=0) / n_assets

        return SimulationResult(
            paths=combined, mean=np.mean(combined, axis=0), std=np.std(combined, axis=0),
            percentiles={
                "p05": np.percentile(combined, 5, axis=0), "p50": np.percentile(combined, 50, axis=0),
                "p95": np.percentile(combined, 95, axis=0),
            },
            n_simulations=n, n_steps=n_steps,
            model_type="copula_gaussian",
        )

    AVAILABLE_MODELS = ["gbm", "mean_reverting", "regime_switching", "jump_diffusion", "copula"]

    def predict(
        self, current_value: float, variable: str, domain: Domain,
        target_date: date, confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH,
        model: str = "gbm", mu: float = 0.08, sigma: float = 0.20,
    ) -> Prediction:
        days_ahead = max(1, (target_date - date.today()).days)
        n_steps = min(days_ahead, 3650)
        dt = days_ahead / n_steps / 365.25

        if model == "gbm":
            result = self.geometric_brownian_motion(current_value, mu, sigma, n_steps, dt)
        elif model == "mean_reverting":
            result = self.mean_reverting(current_value, theta=0.5, mu=current_value, sigma=sigma * 0.5, n_steps=n_steps, dt=dt)
        elif model == "regime_switching":
            result = self.regime_switching(s0=current_value, n_steps=n_steps, dt=dt)
        elif model == "jump_diffusion":
            result = self.jump_diffusion(s0=current_value, mu=mu, sigma=sigma, n_steps=n_steps, dt=dt)
        elif model == "copula":
            result = self.geometric_brownian_motion(current_value, mu, sigma, n_steps, dt)
        else:
            result = self.geometric_brownian_motion(current_value, mu, sigma, n_steps, dt)

        z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645,
             ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence_level]

        mean_val = float(result.mean[-1])
        std_val = float(result.std[-1])

        return Prediction(
            domain=domain, variable=variable, value=round(mean_val, 2),
            confidence_interval=(round(mean_val - z * std_val, 2), round(mean_val + z * std_val, 2)),
            confidence_level=confidence_level, timestamp=datetime.now(),
            target_date=target_date, model_name=f"monte_carlo_{model}",
            weights={"model": model, "n_simulations": self.n_simulations},
            methodology="monte_carlo_simulation",
            uncertainty_type="stochastic",
        )