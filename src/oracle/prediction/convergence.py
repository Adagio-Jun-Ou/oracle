"""Predictive Convergence Engine — Multi-model agreement detection and synthesis.

Core insight: When multiple independent prediction methods converge on the same
outcome, the probability of that outcome is dramatically higher than any single
model suggests (Laplace's principle of insufficient reason + Condorcet jury theorem).
Conversely, when models disagree, the disagreement itself IS the signal — it
reveals structural uncertainty that no single model can see.

This engine:
1. Collects predictions from ALL Oracle modules for each variable
2. Detects convergence clusters (models agreeing beyond chance)
3. Computes Condorcet-adjusted convergence probabilities
4. Identifies divergence zones (where models fundamentally disagree)
5. Maps model reliability per domain (which modules are most accurate where)
6. Produces convergence-weighted final predictions with uncertainty bounds
7. Detects "false convergence" (models agreeing because they share assumptions)

Evidence:
- Condorcet (1785) jury theorem: N independent voters > 50% competent → majority
  probability approaches 1.0 as N→∞
- Clemen (1989): "Combining forecasts: Adjusting weights improves accuracy"
- Batte & Nau (1989): Diverse model combination beats best individual model
- Tetlock (2015): Superforecasters' edge comes from aggregating diverse viewpoints
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class ConvergenceStrength(Enum):
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    ORACULAR = "oracular"


class DivergenceType(Enum):
    ASSUMPTION_SHARED = "assumption_shared"
    MODEL_FAMILY_SPLIT = "model_family_split"
    DATA_INTERPRETATION = "data_interpretation"
    HORIZON_DISAGREEMENT = "horizon_disagreement"
    REGIME_DISAGREEMENT = "regime_disagreement"


class FalseConvergenceType(Enum):
    SHARED_DATA_SOURCE = "shared_data_source"
    SHARED_ASSUMPTIONS = "shared_assumptions"
    COLLISIONAL_CONTAGION = "informational_contagion"
    HERDING = "herding"


@dataclass
class ModelPrediction:
    model_name: str
    model_family: str
    prediction: float
    confidence: float
    domain_reliability: float
    assumptions: List[str]


@dataclass
class ConvergenceCluster:
    variable: str
    mean_prediction: float
    std_prediction: float
    n_models_agreeing: int
    n_total_models: int
    condorcet_probability: float
    strength: ConvergenceStrength
    false_convergence_risk: float
    false_convergence_types: List[FalseConvergenceType]
    model_names: List[str]


@dataclass
class DivergenceZone:
    variable: str
    predictions: List[float]
    spread: float
    direction_split: Dict[str, int]
    divergence_type: DivergenceType
    structural_uncertainty: float
    narrative: str


@dataclass
class ConvergenceResult:
    variable: str
    final_prediction: float
    lower_bound: float
    upper_bound: float
    convergence_cluster: Optional[ConvergenceCluster]
    divergence_zone: Optional[DivergenceZone]
    model_weight_map: Dict[str, float]
    effective_n_models: float
    information_value: float
    recommendation: str


@dataclass
class GlobalConvergenceReport:
    results: Dict[str, ConvergenceResult]
    n_convergent: int
    n_divergent: int
    n_oracular: int
    overall_convergence_score: float
    false_convergence_alerts: List[str]
    omniscience_gaps: List[str]
    model_reliability_map: Dict[str, Dict[str, float]]


SHARED_ASSUMPTION_GROUPS = {
    "efficient_market": ["signal_fusion", "superforecast", "ensemble_calibrator"],
    "behavioral": ["analogical_reasoning", "morphological", "delphi"],
    "structural": ["cross_impact", "causal_event", "cascade_forecast"],
    "extreme_tail": ["black_swan", "existential_risk", "stress_testing"],
    "regime_based": ["regime_shift", "regime_assess", "butterfly_effect"],
    "data_driven": ["daily_oracle", "ts_forecast", "prediction_tracker"],
}


class PredictiveConvergenceEngine:
    def __init__(self, model_reliability: Optional[Dict[str, Dict[str, float]]] = None):
        self.model_reliability = model_reliability or {}
        self._history: List[Dict] = []

    def compute_condorcet(self, n_agree: int, n_total: int,
                          avg_competence: float = 0.6) -> float:
        if n_total < 2 or n_agree < 1:
            return 0.0
        p = min(avg_competence, 0.95)
        k = n_agree
        n = n_total
        prob = 0.0
        for i in range(k, n + 1):
            prob += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        return min(1.0, prob)

    def detect_false_convergence(self, models: List[ModelPrediction]) -> Tuple[float, List[FalseConvergenceType]]:
        if len(models) < 2:
            return 0.0, []
        risks = []
        total_risk = 0.0
        assumption_sets: Dict[str, Set[str]] = {}
        model_assumptions: Dict[str, Set[str]] = {}
        for m in models:
            model_assumptions[m.model_name] = set(m.assumptions) if m.assumptions else set()
        for group_name, group_models in SHARED_ASSUMPTION_GROUPS.items():
            matching = [m for m in models if m.model_name in group_models]
            if len(matching) >= 2:
                overlap = sum(1 for m in matching) / len(models)
                if overlap > 0.5:
                    risks.append(FalseConvergenceType.SHARED_ASSUMPTIONS)
                    total_risk += overlap * 0.3
        families = {}
        for m in models:
            families.setdefault(m.model_family, []).append(m.model_name)
        for family, members in families.items():
            if len(members) >= 3:
                risks.append(FalseConvergenceType.COLLISIONAL_CONTAGION)
                total_risk += 0.15
        predictions = [m.prediction for m in models]
        if np.std(predictions) < 0.01 and len(models) >= 3:
            risks.append(FalseConvergenceType.HERDING)
            total_risk += 0.25
        source_families = set(m.model_family for m in models)
        if len(source_families) <= 2 and len(models) >= 4:
            risks.append(FalseConvergenceType.SHARED_DATA_SOURCE)
            total_risk += 0.2
        return min(1.0, total_risk), list(set(risks))

    def classify_convergence(self, cluster: ConvergenceCluster) -> ConvergenceStrength:
        if cluster.false_convergence_risk > 0.6:
            return ConvergenceStrength.WEAK
        if cluster.condorcet_probability > 0.95 and cluster.n_models_agreeing >= 4:
            return ConvergenceStrength.ORACULAR
        if cluster.condorcet_probability > 0.85 and cluster.n_models_agreeing >= 3:
            return ConvergenceStrength.STRONG
        if cluster.condorcet_probability > 0.7:
            return ConvergenceStrength.MODERATE
        if cluster.condorcet_probability > 0.55:
            return ConvergenceStrength.WEAK
        return ConvergenceStrength.NONE

    def detect_divergence(self, variable: str, models: List[ModelPrediction]) -> Optional[DivergenceZone]:
        if len(models) < 2:
            return None
        predictions = [m.prediction for m in models]
        spread = float(np.std(predictions))
        if spread < 0.05:
            return None
        direction_counts = {"above_0.5": 0, "below_0.5": 0, "near_0.5": 0}
        for p in predictions:
            if p > 0.6:
                direction_counts["above_0.5"] += 1
            elif p < 0.4:
                direction_counts["below_0.5"] += 1
            else:
                direction_counts["near_0.5"] += 1
        families_above = set(m.model_family for m in models if m.prediction > 0.5)
        families_below = set(m.model_family for m in models if m.prediction < 0.5)
        if families_above and families_below and families_above != families_below:
            div_type = DivergenceType.MODEL_FAMILY_SPLIT
        elif spread > 0.3:
            div_type = DivergenceType.REGIME_DISAGREEMENT
        elif spread > 0.15:
            div_type = DivergenceType.DATA_INTERPRETATION
        else:
            div_type = DivergenceType.HORIZON_DISAGREEMENT
        structural_u = min(1.0, spread * 2)
        narrative = f"Models disagree on {variable}: spread={spread:.3f}"
        if direction_counts["above_0.5"] > 0 and direction_counts["below_0.5"] > 0:
            narrative += f" — {direction_counts['above_0.5']} bullish, {direction_counts['below_0.5']} bearish"
        return DivergenceZone(
            variable=variable,
            predictions=predictions,
            spread=round(spread, 4),
            direction_split=direction_counts,
            divergence_type=div_type,
            structural_uncertainty=round(structural_u, 4),
            narrative=narrative,
        )

    def compute_weights(self, models: List[ModelPrediction], variable: str) -> Dict[str, float]:
        if not models:
            return {}
        raw_weights = {}
        for m in models:
            domain_rel = self.model_reliability.get(m.model_name, {}).get(variable, m.domain_reliability)
            raw_weights[m.model_name] = m.confidence * domain_rel
        total = sum(raw_weights.values())
        if total < 1e-10:
            return {m.model_name: 1.0 / len(models) for m in models}
        return {k: v / total for k, v in raw_weights.items()}

    def converge(self, predictions: Dict[str, List[ModelPrediction]],
                 convergence_threshold: float = 0.05) -> GlobalConvergenceReport:
        results = {}
        n_convergent = 0
        n_divergent = 0
        n_oracular = 0
        false_convergence_alerts = []
        omniscience_gaps = []
        for variable, models in predictions.items():
            if not models:
                continue
            weights = self.compute_weights(models, variable)
            weighted_pred = sum(m.prediction * weights.get(m.model_name, 1.0 / len(models))
                               for m in models)
            preds = [m.prediction for m in models]
            std = float(np.std(preds))
            convergence_cluster = None
            divergence_zone = None
            if std <= convergence_threshold:
                agreeing = [m for m in models if abs(m.prediction - weighted_pred) <= convergence_threshold * 2]
                avg_comp = np.mean([m.confidence * m.domain_reliability for m in agreeing]) if agreeing else 0.5
                condorcet = self.compute_condorcet(len(agreeing), len(models), float(avg_comp))
                false_risk, false_types = self.detect_false_convergence(agreeing)
                cluster = ConvergenceCluster(
                    variable=variable,
                    mean_prediction=round(float(weighted_pred), 4),
                    std_prediction=round(std, 4),
                    n_models_agreeing=len(agreeing),
                    n_total_models=len(models),
                    condorcet_probability=round(condorcet, 4),
                    strength=ConvergenceStrength.NONE,
                    false_convergence_risk=round(false_risk, 4),
                    false_convergence_types=false_types,
                    model_names=[m.model_name for m in agreeing],
                )
                cluster.strength = self.classify_convergence(cluster)
                convergence_cluster = cluster
                if cluster.strength in (ConvergenceStrength.STRONG, ConvergenceStrength.ORACULAR):
                    n_convergent += 1
                    if cluster.strength == ConvergenceStrength.ORACULAR:
                        n_oracular += 1
                if false_risk > 0.4:
                    false_convergence_alerts.append(
                        f"{variable}: {false_risk:.0%} false convergence risk ({', '.join(t.value for t in false_types)})"
                    )
            else:
                divergence_zone = self.detect_divergence(variable, models)
                n_divergent += 1
                if divergence_zone and divergence_zone.structural_uncertainty > 0.5:
                    omniscience_gaps.append(
                        f"{variable}: structural uncertainty {divergence_zone.structural_uncertainty:.0%} — {divergence_zone.divergence_type.value}"
                    )
            weights_array = list(weights.values())
            eff_n = 1.0 / max(sum(w ** 2 for w in weights_array), 1e-10) if weights_array else 1.0
            ci_width = std * (1 + 1.0 / max(eff_n, 1))
            info_value = min(1.0, eff_n / 5.0 * (1.0 - std / max(std + 0.1, 0.1)))
            rec = self._make_recommendation(convergence_cluster, divergence_zone, info_value)
            results[variable] = ConvergenceResult(
                variable=variable,
                final_prediction=round(float(weighted_pred), 4),
                lower_bound=round(max(0.0, float(weighted_pred) - ci_width), 4),
                upper_bound=round(min(1.0, float(weighted_pred) + ci_width), 4),
                convergence_cluster=convergence_cluster,
                divergence_zone=divergence_zone,
                model_weight_map={k: round(v, 4) for k, v in weights.items()},
                effective_n_models=round(float(eff_n), 2),
                information_value=round(float(info_value), 4),
                recommendation=rec,
            )
        overall = n_convergent / max(len(results), 1) if results else 0.0
        return GlobalConvergenceReport(
            results=results,
            n_convergent=n_convergent,
            n_divergent=n_divergent,
            n_oracular=n_oracular,
            overall_convergence_score=round(overall, 4),
            false_convergence_alerts=false_convergence_alerts[:10],
            omniscience_gaps=omniscience_gaps[:10],
            model_reliability_map=self.model_reliability,
        )

    def _make_recommendation(self, cluster: Optional[ConvergenceCluster],
                             divergence: Optional[DivergenceZone],
                             info_value: float) -> str:
        if cluster and cluster.strength == ConvergenceStrength.ORACULAR:
            return f"ORACULAR convergence ({cluster.n_models_agreeing}/{cluster.n_total_models} models) — act with conviction"
        if cluster and cluster.strength == ConvergenceStrength.STRONG:
            if cluster.false_convergence_risk > 0.4:
                return f"STRONG convergence but false convergence risk {cluster.false_convergence_risk:.0%} — verify independence"
            return f"STRONG convergence ({cluster.n_models_agreeing} models agree) — high confidence"
        if cluster and cluster.strength == ConvergenceStrength.MODERATE:
            return f"MODERATE convergence — proceed but monitor for divergence"
        if divergence:
            return f"DIVERGENCE detected ({divergence.divergence_type.value}) — structural uncertainty {divergence.structural_uncertainty:.0%}, reduce position size"
        if info_value < 0.3:
            return "LOW information value — gather more data before deciding"
        return "WEAK signal — no clear convergence or divergence"

    def update_reliability(self, model_name: str, domain: str, actual_outcome: float,
                           predicted: float) -> float:
        error = abs(predicted - actual_outcome)
        new_rel = max(0.1, 1.0 - error)
        if model_name not in self.model_reliability:
            self.model_reliability[model_name] = {}
        old = self.model_reliability[model_name].get(domain, 0.5)
        self.model_reliability[model_name][domain] = round(0.7 * old + 0.3 * new_rel, 4)
        return self.model_reliability[model_name][domain]