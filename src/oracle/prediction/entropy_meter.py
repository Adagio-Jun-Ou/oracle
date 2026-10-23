# -*- coding: utf-8 -*-
"""Entropy Monitor — System-wide chaos and unpredictability measurement.

Inspired by Rehoboam's entropy monitoring: measures the overall chaos in the
prediction system. High entropy = predictions unreliable, interventions needed.
Low entropy = predictions stable, system operating within bounds.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import math


@dataclass
class DomainEntropy:
    domain: str
    shannon_entropy: float
    kolmogorov_estimate: float  # estimated Kolmogorov complexity
    volatility: float
    predictability: float  # 1 - entropy
    regime_stability: float


@dataclass
class EntropyReport:
    timestamp: float
    system_entropy: float
    domain_entropies: Dict[str, DomainEntropy]
    entropy_trend: str  # increasing / decreasing / stable
    chaos_hotspots: List[str]
    stability_zones: List[str]
    overall_predictability: float
    phase_space_volume: float
    intervention_recommendation: str


class EntropyMeter:
    """Monitors system-wide entropy across all prediction domains.

    Tracks Shannon entropy, volatility, regime stability, and predictability
    for each domain. Detects chaos hotspots and stability zones.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._series: Dict[str, List[float]] = defaultdict(list)
        self._predictions: Dict[str, List[float]] = defaultdict(list)

    def observe(self, domain: str, actual: float,
                predicted: Optional[float] = None) -> float:
        """Record an observation. Returns local entropy."""
        self._series[domain].append(actual)
        if predicted is not None:
            self._predictions[domain].append(predicted)
        if len(self._series[domain]) > self.window_size:
            self._series[domain].pop(0)
        if domain in self._predictions and len(self._predictions[domain]) > self.window_size:
            self._predictions[domain].pop(0)
        return self._local_entropy(domain)

    def measure(self) -> EntropyReport:
        """Full system entropy measurement."""
        domain_entropies = {}
        for domain in self._series:
            de = self._compute_domain_entropy(domain)
            domain_entropies[domain] = de
        system_entropy = float(np.mean([de.shannon_entropy for de in domain_entropies.values()])) if domain_entropies else 0.5
        trend = self._compute_trend()
        hotspots = [d for d, de in domain_entropies.items() if de.shannon_entropy > 0.7]
        stable = [d for d, de in domain_entropies.items() if de.shannon_entropy < 0.3]
        predictability = float(1 - system_entropy)
        volume = self._estimate_phase_volume(domain_entropies)
        rec = self._recommendation(system_entropy, hotspots, stable, trend)
        return EntropyReport(
            timestamp=float(np.datetime64("now").astype(float)) if hasattr(np, 'datetime64') else 0,
            system_entropy=round(system_entropy, 4),
            domain_entropies=domain_entropies,
            entropy_trend=trend,
            chaos_hotspots=hotspots,
            stability_zones=stable,
            overall_predictability=round(predictability, 4),
            phase_space_volume=round(volume, 4),
            intervention_recommendation=rec,
        )

    def _local_entropy(self, domain: str) -> float:
        series = self._series.get(domain, [])
        if len(series) < 3:
            return 0.5
        values = np.array(series)
        n_bins = min(10, len(values))
        hist, _ = np.histogram(values, bins=n_bins, density=False)
        total = hist.sum()
        if total == 0:
            return 0.5
        probs = hist[hist > 0] / total
        raw = float(-np.sum(probs * np.log(probs)))
        max_entropy = math.log(max(n_bins, 2))
        return raw / max_entropy if max_entropy > 0 else 0.5

    def _compute_domain_entropy(self, domain: str) -> DomainEntropy:
        series = np.array(self._series.get(domain, []))
        preds = np.array(self._predictions.get(domain, []))
        shannon = self._local_entropy(domain)
        # Kolmogorov estimate: compression ratio
        if len(series) > 10:
            diffs = np.diff(series)
            kolmogorov = float(np.std(diffs) / max(np.mean(np.abs(series)), 0.001))
        else:
            kolmogorov = 0.5
        volatility = float(np.std(series)) if len(series) > 2 else 0.1
        # Predictability = 1 - normalized prediction error
        if len(preds) > 0 and len(series) >= len(preds):
            actual_slice = series[-len(preds):]
            errors = np.abs(actual_slice - preds[:len(actual_slice)])
            predictability = float(1 - np.mean(errors) / max(np.mean(np.abs(actual_slice)), 0.001))
            predictability = max(0.0, min(1.0, predictability))
        else:
            predictability = max(0.0, 1 - shannon)
        # Regime stability: how often does direction change?
        if len(series) > 3:
            changes = np.diff(np.sign(np.diff(series)))
            regime_stability = float(1 - np.mean(np.abs(changes)) / max(len(changes), 1))
        else:
            regime_stability = 0.5
        return DomainEntropy(
            domain=domain,
            shannon_entropy=round(shannon, 4),
            kolmogorov_estimate=round(kolmogorov, 4),
            volatility=round(volatility, 4),
            predictability=round(predictability, 4),
            regime_stability=round(max(0, regime_stability), 4),
        )

    def _compute_trend(self) -> str:
        all_entropy = []
        for domain, series in self._series.items():
            if len(series) > 10:
                first = self._local_entropy_from_slice(series[:len(series)//2])
                second = self._local_entropy_from_slice(series[len(series)//2:])
                all_entropy.append(second - first)
        if not all_entropy:
            return "stable"
        mean_delta = float(np.mean(all_entropy))
        if mean_delta > 0.05:
            return "increasing"
        elif mean_delta < -0.05:
            return "decreasing"
        return "stable"

    def _local_entropy_from_slice(self, series: list) -> float:
        if len(series) < 3:
            return 0.5
        values = np.array(series)
        n_bins = min(10, len(values))
        hist, _ = np.histogram(values, bins=n_bins, density=False)
        total = hist.sum()
        if total == 0:
            return 0.5
        probs = hist[hist > 0] / total
        raw = float(-np.sum(probs * np.log(probs))) if len(probs) > 0 else 0.0
        max_entropy = math.log(max(n_bins, 2))
        return raw / max_entropy if max_entropy > 0 else 0.5

    def _estimate_phase_volume(self, entropies: Dict[str, DomainEntropy]) -> float:
        """Estimate the volume of the phase space being explored."""
        if not entropies:
            return 0.5
        vols = []
        for de in entropies.values():
            vols.append(de.volatility * de.kolmogorov_estimate)
        return float(np.mean(vols)) if vols else 0.5

    def _recommendation(self, entropy: float, hotspots: List[str],
                        stable: List[str], trend: str) -> str:
        parts = []
        if entropy > 0.8:
            parts.append("CRITICAL: System entropy exceeds 80% — predictions unreliable.")
        elif entropy > 0.6:
            parts.append("ELEVATED: High entropy detected — increase monitoring frequency.")
        elif entropy < 0.3:
            parts.append("LOW: System stable — predictions highly reliable.")
        else:
            parts.append("MODERATE: Normal operating entropy.")
        if hotspots:
            parts.append(f"Chaos hotspots: {', '.join(hotspots[:3])}.")
        if trend == "increasing":
            parts.append("Entropy TRENDING UP — system destabilizing.")
        elif trend == "decreasing":
            parts.append("Entropy trending DOWN — system stabilizing.")
        if stable:
            parts.append(f"Stable zones: {', '.join(stable[:3])}.")
        return " ".join(parts)
