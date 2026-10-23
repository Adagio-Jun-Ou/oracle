"""Entropic Forecastability Index — Measures how predictable each domain actually is.

Shannon entropy applied to prediction distributions: some domains have fundamentally
lower entropy (more predictable) and others higher (less predictable). This module
quantifies the theoretical limits of predictability.

Based on:
- Shannon (1948) A Mathematical Theory of Communication
- Bialek et al. (2001) Predictability, complexity, and learning
- Crutchfield & Feldman (2003) Regularities unseen, randomness observed
- Del Sole & Tippett (2007) Predictability: past, present, future
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class ForecastabilityClass(Enum):
    HIGHLY_PREDICTABLE = "highly_predictable"
    PREDICTABLE = "predictable"
    MODERATELY_PREDICTABLE = "moderately_predictable"
    WEAKLY_PREDICTABLE = "weakly_predictable"
    UNPREDICTABLE = "unpredictable"


class EntropySource(Enum):
    THERMODYNAMIC = "thermodynamic"
    INFORMATION_THEORETIC = "information_theoretic"
    ALGORITHMIC = "algorithmic"
    CHAOTIC = "chaotic"
    STOCHASTIC = "stochastic"


@dataclass
class DomainForecastability:
    domain: str
    entropy: float
    forecastability_index: float
    forecastability_class: ForecastabilityClass
    kolmogorov_complexity_estimate: float
    mutual_information_rate: float
    predictability_horizon_days: int
    decay_rate: float
    signal_to_noise_ratio: float
    top_entropy_sources: List[EntropySource]
    irreducible_uncertainty: float
    recommendation: str


@dataclass
class ForecastabilityReport:
    domains: Dict[str, DomainForecastability]
    overall_forecastability: float
    most_predictable: List[str]
    least_predictable: List[str]
    information_theoretic_limit: float
    practical_prediction_ceiling: float
    omniscience_gap: float
    narrative: str


DOMAIN_ENTROPY_BASELINES = {
    "monetary": {"entropy": 0.45, "horizon": 180, "decay": 0.003, "snr": 2.5},
    "inflation": {"entropy": 0.55, "horizon": 120, "decay": 0.004, "snr": 1.8},
    "equity": {"entropy": 0.68, "horizon": 22, "decay": 0.02, "snr": 1.2},
    "volatility": {"entropy": 0.78, "horizon": 5, "decay": 0.05, "snr": 0.8},
    "bonds": {"entropy": 0.42, "horizon": 60, "decay": 0.005, "snr": 3.0},
    "commodity": {"entropy": 0.65, "horizon": 14, "decay": 0.03, "snr": 1.0},
    "crypto": {"entropy": 0.85, "horizon": 3, "decay": 0.1, "snr": 0.5},
    "currency": {"entropy": 0.60, "horizon": 30, "decay": 0.015, "snr": 1.5},
    "geopolitical": {"entropy": 0.75, "horizon": 7, "decay": 0.04, "snr": 0.7},
    "sentiment": {"entropy": 0.70, "horizon": 10, "decay": 0.025, "snr": 0.9},
    "climate": {"entropy": 0.50, "horizon": 90, "decay": 0.002, "snr": 2.0},
    "technology": {"entropy": 0.72, "horizon": 14, "decay": 0.02, "snr": 1.1},
    "economic": {"entropy": 0.55, "horizon": 60, "decay": 0.004, "snr": 1.8},
    "health": {"entropy": 0.60, "horizon": 30, "decay": 0.01, "snr": 1.4},
    "labor": {"entropy": 0.48, "horizon": 90, "decay": 0.003, "snr": 2.2},
}

ENTROPY_SOURCE_MAP = {
    "equity": [EntropySource.CHAOTIC, EntropySource.STOCHASTIC, EntropySource.INFORMATION_THEORETIC],
    "volatility": [EntropySource.CHAOTIC, EntropySource.STOCHASTIC],
    "geopolitical": [EntropySource.THERMODYNAMIC, EntropySource.STOCHASTIC],
    "crypto": [EntropySource.CHAOTIC, EntropySource.STOCHASTIC, EntropySource.ALGORITHMIC],
    "climate": [EntropySource.THERMODYNAMIC, EntropySource.CHAOTIC],
    "monetary": [EntropySource.INFORMATION_THEORETIC, EntropySource.STOCHASTIC],
    "inflation": [EntropySource.STOCHASTIC, EntropySource.INFORMATION_THEORETIC],
    "commodity": [EntropySource.THERMODYNAMIC, EntropySource.STOCHASTIC],
}

FORECASTABILITY_THRESHOLDS = {
    ForecastabilityClass.HIGHLY_PREDICTABLE: 0.75,
    ForecastabilityClass.PREDICTABLE: 0.60,
    ForecastabilityClass.MODERATELY_PREDICTABLE: 0.45,
    ForecastabilityClass.WEAKLY_PREDICTABLE: 0.30,
}


class EntropicForecastabilityIndex:
    def __init__(self, baselines: Optional[Dict] = None):
        self.baselines = baselines or DOMAIN_ENTROPY_BASELINES

    def compute_domain(self, domain: str,
                       domain_states: Optional[Dict[str, float]] = None,
                       time_series: Optional[np.ndarray] = None) -> DomainForecastability:
        baseline = self.baselines.get(domain, {"entropy": 0.6, "horizon": 14, "decay": 0.02, "snr": 1.0})
        entropy = baseline["entropy"]
        horizon = baseline["horizon"]
        decay = baseline["decay"]
        snr = baseline["snr"]
        if time_series is not None and len(time_series) > 20:
            ts_entropy = self._compute_time_series_entropy(time_series)
            entropy = 0.6 * entropy + 0.4 * ts_entropy
        if domain_states:
            state_val = domain_states.get(domain, 0.5)
            extremeness = abs(state_val - 0.5) * 2
            entropy *= (1 + extremeness * 0.15)
            entropy = min(1.0, entropy)
        forecastability = max(0.0, 1.0 - entropy)
        complexity = entropy * (1 + 1.0 / max(snr, 0.1)) * 0.5
        mi_rate = forecastability * (1 - math.exp(-snr * 0.5))
        irreducible = entropy * 0.3 / max(snr, 0.1)
        if forecastability >= FORECASTABILITY_THRESHOLDS[ForecastabilityClass.HIGHLY_PREDICTABLE]:
            fclass = ForecastabilityClass.HIGHLY_PREDICTABLE
        elif forecastability >= FORECASTABILITY_THRESHOLDS[ForecastabilityClass.PREDICTABLE]:
            fclass = ForecastabilityClass.PREDICTABLE
        elif forecastability >= FORECASTABILITY_THRESHOLDS[ForecastabilityClass.MODERATELY_PREDICTABLE]:
            fclass = ForecastabilityClass.MODERATELY_PREDICTABLE
        elif forecastability >= FORECASTABILITY_THRESHOLDS[ForecastabilityClass.WEAKLY_PREDICTABLE]:
            fclass = ForecastabilityClass.WEAKLY_PREDICTABLE
        else:
            fclass = ForecastabilityClass.UNPREDICTABLE
        sources = ENTROPY_SOURCE_MAP.get(domain, [EntropySource.STOCHASTIC])
        rec = self._make_recommendation(domain, fclass, horizon, irreducible)
        return DomainForecastability(
            domain=domain,
            entropy=round(float(entropy), 4),
            forecastability_index=round(float(forecastability), 4),
            forecastability_class=fclass,
            kolmogorov_complexity_estimate=round(float(complexity), 4),
            mutual_information_rate=round(float(mi_rate), 4),
            predictability_horizon_days=horizon,
            decay_rate=round(float(decay), 4),
            signal_to_noise_ratio=round(float(snr), 4),
            top_entropy_sources=sources,
            irreducible_uncertainty=round(float(irreducible), 4),
            recommendation=rec,
        )

    def compute_all(self, domain_states: Optional[Dict[str, float]] = None,
                     time_series_map: Optional[Dict[str, np.ndarray]] = None) -> ForecastabilityReport:
        domains = {}
        all_domains = set(self.baselines.keys())
        if domain_states:
            all_domains |= set(domain_states.keys())
        if time_series_map:
            all_domains |= set(time_series_map.keys())
        for domain in all_domains:
            ts = time_series_map.get(domain) if time_series_map else None
            domains[domain] = self.compute_domain(domain, domain_states, ts)
        fi_values = [d.forecastability_index for d in domains.values()]
        overall_fi = float(np.mean(fi_values)) if fi_values else 0.5
        most_pred = sorted(domains.keys(), key=lambda d: -domains[d].forecastability_index)[:5]
        least_pred = sorted(domains.keys(), key=lambda d: domains[d].forecastability_index)[:5]
        it_limit = float(np.mean([d.irreducible_uncertainty for d in domains.values()])) if domains else 0.3
        practical_ceiling = overall_fi * 0.85
        omniscience_gap = 1.0 - practical_ceiling
        narrative = self._generate_narrative(domains, overall_fi, most_pred, least_pred)
        return ForecastabilityReport(
            domains=domains,
            overall_forecastability=round(overall_fi, 4),
            most_predictable=most_pred,
            least_predictable=least_pred,
            information_theoretic_limit=round(it_limit, 4),
            practical_prediction_ceiling=round(practical_ceiling, 4),
            omniscience_gap=round(omniscience_gap, 4),
            narrative=narrative,
        )

    def _compute_time_series_entropy(self, ts: np.ndarray) -> float:
        arr = np.asarray(ts, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 10:
            return 0.6
        diffs = np.diff(arr)
        if np.std(diffs) < 1e-10:
            return 0.1
        n_bins = min(20, int(np.sqrt(len(arr))))
        hist, _ = np.histogram(diffs, bins=n_bins, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        entropy /= np.log(n_bins)
        return min(1.0, max(0.0, float(entropy)))

    def _make_recommendation(self, domain: str, fclass: ForecastabilityClass,
                             horizon: int, irreducible: float) -> str:
        if fclass == ForecastabilityClass.HIGHLY_PREDICTABLE:
            return f"{domain}: Highly predictable (horizon: {horizon}d). Confident forecasts warranted. Monitor for regime changes."
        elif fclass == ForecastabilityClass.PREDICTABLE:
            return f"{domain}: Predictable (horizon: {horizon}d). Forecasts useful with calibrated confidence intervals."
        elif fclass == ForecastabilityClass.MODERATELY_PREDICTABLE:
            return f"{domain}: Moderately predictable (horizon: {horizon}d). Use ensemble methods; present as probability ranges."
        elif fclass == ForecastabilityClass.WEAKLY_PREDICTABLE:
            return f"{domain}: Weakly predictable (horizon: {horizon}d). Emphasize scenarios over point forecasts. {irreducible:.0%} uncertainty is irreducible."
        else:
            return f"{domain}: Unpredictable (horizon: {horizon}d). Acknowledge fundamental uncertainty. Focus on resilience over prediction."

    def _generate_narrative(self, domains, overall_fi, most_pred, least_pred) -> str:
        fi_map = {d.domain: d.forecastability_index for d in domains.values()}
        parts = [f"Overall forecastability: {overall_fi:.0%} (practical ceiling: {overall_fi * 0.85:.0%})."]
        if most_pred:
            pred_str = ", ".join(f"{d} ({fi_map[d]:.0%})" for d in most_pred[:3])
            parts.append(f"Most predictable: {pred_str}.")
        if least_pred:
            unpred_str = ", ".join(f"{d} ({fi_map[d]:.0%})" for d in least_pred[:3])
            parts.append(f"Least predictable: {unpred_str}.")
        parts.append("The Oracle's omniscience is bounded by information-theoretic limits.")
        parts.append("Some futures are genuinely unknowable — wisdom lies in knowing which.")
        return " ".join(parts)