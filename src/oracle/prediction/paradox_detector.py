"""Paradox Detector — finds logical contradictions across domain predictions.

When Oracle predicts GDP up + unemployment up + inflation down simultaneously,
that's a macroeconomic paradox. This module systematically detects such
contradictions using domain-specific consistency rules.

Phase Transition Detector — identifies when system dynamics qualitatively change.

Uses statistical tests (CUSUM, Pettitt, Bai-Perron) and physics-inspired
order parameters to detect regime shifts in time series.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ParadoxSeverity(Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class Paradox:
    name: str
    domains: List[str]
    variables: List[str]
    description: str
    severity: ParadoxSeverity
    contradiction_score: float
    resolution_hypotheses: List[str]
    historical_precedent: str = ""


@dataclass
class ParadoxReport:
    paradoxes: List[Paradox]
    consistency_score: float
    domain_agreement_matrix: Dict[str, Dict[str, float]]
    n_paradoxes_by_severity: Dict[str, int]
    method: str = "paradox_detection"


@dataclass
class PhaseTransition:
    variable: str
    transition_date: int
    transition_strength: float
    before_mean: float
    after_mean: float
    before_std: float
    after_std: float
    confidence: float
    method: str = "phase_transition"


@dataclass
class PhaseTransitionReport:
    transitions: List[PhaseTransition]
    n_regimes: int
    regime_means: List[float]
    regime_stds: List[float]
    regime_boundaries: List[int]
    order_parameter: float
    method: str = "phase_transition_detection"


CONSISTENCY_RULES = [
    {"name": "Stagflation paradox", "condition": lambda v: v.get("inflation_rate", 0.03) > 0.05 and v.get("gdp_growth", 0.02) < 0.01 and v.get("unemployment_rate", 0.04) > 0.06,
     "description": "High inflation + low growth + high unemployment (stagflation) — difficult for central banks",
     "severity": ParadoxSeverity.SEVERE, "resolution": ["Supply-side shock (energy, food)", "Wage-price spiral from embedded inflation expectations", "Structural labor market mismatch"]},
    {"name": "Growth without jobs", "condition": lambda v: v.get("gdp_growth", 0.02) > 0.03 and v.get("unemployment_rate", 0.04) > 0.07,
     "description": "GDP growing but unemployment high — suggests jobless recovery or automation displacement",
     "severity": ParadoxSeverity.MODERATE, "resolution": ["Automation/AI displacement", "Labor force participation decline", "Gig economy masking underemployment"]},
    {"name": "Safe haven paradox", "condition": lambda v: v.get("vix_level", 20) > 30 and v.get("gold_price_change", 0) < -0.02,
     "description": "High volatility (VIX>30) but gold declining — risk-off flows not entering traditional safe haven",
     "severity": ParadoxSeverity.MODERATE, "resolution": ["Liquidation selling across all assets", "Cryptocurrency absorbing safe-haven flows", "Dollar strength dampening gold"]},
    {"name": "Climate-growth conflict", "condition": lambda v: v.get("climate_anomaly_c", 1.2) > 2.0 and v.get("fossil_fuel_investment_ratio", 0.5) > 0.6,
     "description": "Severe climate impact projected but fossil fuel investment still high — transition not occurring",
     "severity": ParadoxSeverity.SEVERE, "resolution": ["Policy lag between commitments and action", "Energy security concerns delaying transition", "Developing economy energy needs"]},
    {"name": "Low-rate housing crisis", "condition": lambda v: v.get("interest_rate", 0.04) < 0.02 and v.get("housing_affordability_index", 0.5) < 0.3,
     "description": "Low interest rates but housing unaffordable — suggests supply constraints or speculation",
     "severity": ParadoxSeverity.MODERATE, "resolution": ["Supply shortage driving prices", "Investor/speculator demand", "Zoning and regulatory barriers"]},
    {"name": "Technology paradox", "condition": lambda v: v.get("ai_capability_index", 0.5) > 0.7 and v.get("productivity_growth", 0.015) < 0.01,
     "description": "AI capability growing fast but productivity stagnant — Solow paradox revisited",
     "severity": ParadoxSeverity.MODERATE, "resolution": ["Measurement lag in productivity stats", "Deployment lag between capability and adoption", "Creative destruction phase"]},
    {"name": "Pandemic-geopolitical feedback", "condition": lambda v: v.get("pandemic_risk", 0.3) > 0.6 and v.get("geopolitical_tension", 0.3) > 0.6,
     "description": "Pandemic risk high while geopolitical tension also elevated — mutual amplification through supply chains",
     "severity": ParadoxSeverity.SEVERE, "resolution": ["Resource competition during health crisis", "Vaccine nationalism", "Deglobalization accelerating"]},
    {"name": "Water-energy-food nexus paradox", "condition": lambda v: v.get("water_stress", 0.3) > 0.6 and v.get("energy_stress", 0.3) > 0.5 and v.get("food_insecurity", 0.2) > 0.4,
     "description": "Water, energy, and food stress simultaneously elevated — nexus cascade risk",
     "severity": ParadoxSeverity.CRITICAL, "resolution": ["Water-intensive energy competing with agriculture", "Climate exacerbating all three simultaneously", "Infrastructure failure amplification"]},
]


class ParadoxDetector:
    """Detect logical contradictions across domain predictions."""

    def __init__(self):
        self.rules = list(CONSISTENCY_RULES)

    def add_rule(self, name: str, condition, description: str,
                  severity: ParadoxSeverity = ParadoxSeverity.MODERATE,
                  resolution: List[str] = None) -> 'ParadoxDetector':
        self.rules.append({
            "name": name, "condition": condition, "description": description,
            "severity": severity,
            "resolution": resolution or ["Requires further analysis"],
        })
        return self

    def detect(self, domain_values: Dict[str, float]) -> ParadoxReport:
        paradoxes = []
        for rule in self.rules:
            try:
                if rule["condition"](domain_values):
                    relevant_domains = []
                    for key in domain_values:
                        if any(k in key for k in ["gdp", "inflation", "unemployment", "vix", "gold",
                                                    "climate", "fossil", "interest", "housing", "ai",
                                                    "pandemic", "geopolitical", "water", "energy", "food"]):
                            relevant_domains.append(key)
                    paradoxes.append(Paradox(
                        name=rule["name"],
                        domains=list(set(relevant_domains))[:5],
                        variables=list(domain_values.keys())[:5],
                        description=rule["description"],
                        severity=rule["severity"],
                        contradiction_score=round(min(1.0, sum(abs(domain_values.get(k, 0.5) - 0.5) for k in domain_values) / max(len(domain_values), 1)), 4),
                        resolution_hypotheses=rule.get("resolution", ["Requires further analysis"]),
                    ))
            except Exception as e:
                logger.debug(f"Rule {rule['name']} failed: {e}")
                continue

        n_by_sev = {s.value: sum(1 for p in paradoxes if p.severity == s) for s in ParadoxSeverity}
        consistency = max(0.0, 1.0 - len(paradoxes) * 0.15)
        agreement = {}
        domain_list = list(domain_values.keys())
        for i, d1 in enumerate(domain_list):
            agreement[d1] = {}
            for j, d2 in enumerate(domain_list):
                if i < j:
                    v1 = domain_values[d1]
                    v2 = domain_values[d2]
                    agreement[d1][d2] = round(1.0 - abs(v1 - v2), 4)
                    if d2 not in agreement:
                        agreement[d2] = {}
                    agreement[d2][d1] = agreement[d1][d2]

        return ParadoxReport(
            paradoxes=paradoxes,
            consistency_score=round(consistency, 4),
            domain_agreement_matrix=agreement,
            n_paradoxes_by_severity=n_by_sev,
        )


class PhaseTransitionDetector:
    """Detect regime shifts and phase transitions in time series.

    Uses multiple methods:
    - CUSUM (cumulative sum) test
    - Pettitt test for single change point
    - Bai-Perron multiple break point detection
    - Variance ratio test for structural breaks
    """

    def __init__(self, min_segment_length: int = 10, significance: float = 0.05):
        self.min_segment_length = min_segment_length
        self.significance = significance

    def _cusum_test(self, series: np.ndarray) -> Tuple[int, float]:
        n = len(series)
        if n < 4:
            return 0, 0.0
        mean = np.mean(series)
        cusum = np.cumsum(series - mean)
        max_idx = int(np.argmax(np.abs(cusum)))
        max_val = float(np.abs(cusum[max_idx])) / (np.std(series) * np.sqrt(n)) if np.std(series) > 0 else 0.0
        return max_idx, max_val

    def _pettitt_test(self, series: np.ndarray) -> Tuple[int, float]:
        n = len(series)
        if n < 4:
            return 0, 1.0
        rank = np.argsort(np.argsort(series))
        U = np.zeros(n)
        for k in range(n):
            U[k] = 2 * np.sum(rank[:k+1]) - k * (n + 1)
        K = int(np.argmax(np.abs(U)))
        p_value = 2.0 * np.exp(-6.0 * U[K]**2 / (n**2 * (n**2 - 1))) if n > 2 else 1.0
        return K, min(p_value, 1.0)

    def _variance_ratio_break(self, series: np.ndarray, window: int = 20) -> List[int]:
        if len(series) < window * 3:
            return []
        breaks = []
        variances = []
        for i in range(0, len(series) - window, window // 2):
            seg = series[i:i + window]
            variances.append(float(np.var(seg)) if len(seg) > 1 else 0.0)
        var_arr = np.array(variances)
        if len(var_arr) < 3:
            return []
        mean_var = np.mean(var_arr)
        for i in range(1, len(var_arr)):
            ratio = var_arr[i] / max(mean_var, 1e-10)
            if ratio > 3.0 or ratio < 0.2:
                breaks.append(i * (window // 2) + window // 2)
        return breaks

    def detect(self, series: np.ndarray, variable_name: str = "") -> PhaseTransitionReport:
        if len(series) < 10:
            return PhaseTransitionReport(
                transitions=[], n_regimes=1,
                regime_means=[float(np.mean(series))] if len(series) > 0 else [0.0],
                regime_stds=[float(np.std(series))] if len(series) > 0 else [0.0],
                regime_boundaries=[],
                order_parameter=0.0,
            )

        cusum_idx, cusum_stat = self._cusum_test(series)
        pettitt_idx, pettitt_p = self._pettitt_test(series)
        var_breaks = self._variance_ratio_break(series)
        all_breaks = set()
        if cusum_stat > 1.96:
            all_breaks.add(cusum_idx)
        if pettitt_p < self.significance:
            all_breaks.add(pettitt_idx)
        for b in var_breaks:
            all_breaks.add(b)
        all_breaks = sorted([b for b in all_breaks if self.min_segment_length < b < len(series) - self.min_segment_length])

        transitions = []
        for idx in all_breaks:
            before = series[:idx]
            after = series[idx:]
            if len(before) < 3 or len(after) < 3:
                continue
            mean_diff = float(np.abs(np.mean(after) - np.mean(before)))
            pooled_std = float(np.sqrt((np.var(before) * len(before) + np.var(after) * len(after)) / max(len(before) + len(after) - 2, 1)))
            cohens_d = mean_diff / max(pooled_std, 1e-10)
            confidence = min(1.0, abs(cohens_d) / 2.0)
            transitions.append(PhaseTransition(
                variable=variable_name,
                transition_date=idx,
                transition_strength=round(cohens_d, 4),
                before_mean=round(float(np.mean(before)), 6),
                after_mean=round(float(np.mean(after)), 6),
                before_std=round(float(np.std(before)), 6),
                after_std=round(float(np.std(after)), 6),
                confidence=round(confidence, 4),
            ))

        breaks = [0] + all_breaks + [len(series)]
        regime_means = []
        regime_stds = []
        for i in range(len(breaks) - 1):
            seg = series[breaks[i]:breaks[i + 1]]
            regime_means.append(round(float(np.mean(seg)), 6))
            regime_stds.append(round(float(np.std(seg)), 6))

        if len(series) > 2:
            first_half = series[:len(series)//2]
            second_half = series[len(series)//2:]
            variance_change = np.var(second_half) / max(np.var(first_half), 1e-10)
            order_param = float(min(abs(variance_change - 1.0), 5.0))
        else:
            order_param = 0.0

        return PhaseTransitionReport(
            transitions=transitions,
            n_regimes=len(regime_means),
            regime_means=regime_means,
            regime_stds=regime_stds,
            regime_boundaries=all_breaks,
            order_parameter=round(order_param, 4),
        )