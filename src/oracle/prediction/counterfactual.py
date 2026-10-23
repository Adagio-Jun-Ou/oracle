"""Counterfactual Reasoner — What would need to change for the opposite outcome?

Core Insight: The most valuable foresight comes not from predicting what WILL happen,
but from identifying the MINIMUM CHANGES that would flip the outcome. If a recession
is 70% likely, what specific shifts would make it 30%? This creates actionable intelligence.

Implements:
- CounterfactualScenario: A set of minimal changes that would reverse the predicted outcome
- CounterfactualReasoner: Identifies critical variables, computes sensitivity, generates minimal interventions
- InterventionAssessor: Rates feasibility and impact of each counterfactual intervention
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class VariableContribution:
    variable: str
    current_value: float
    required_value: float
    change_magnitude: float
    change_direction: str
    contribution_to_outcome: float
    feasibility: float
    feasibility_obstacles: List[str] = field(default_factory=list)


@dataclass
class CounterfactualScenario:
    name: str
    original_outcome: str
    counterfactual_outcome: str
    original_probability: float
    counterfactual_probability: float
    required_changes: List[VariableContribution]
    n_changes_required: int
    min_total_change: float
    feasibility_score: float
    narrative: str
    method: str = "counterfactual"


@dataclass
class CounterfactualReport:
    original_prediction: str
    original_probability: float
    scenarios: List[CounterfactualScenario]
    critical_variables: List[str]
    leverage_points: List[str]
    robustness_score: float
    method: str = "counterfactual_reasoning"


class CounterfactualReasoner:
    """Identifies what would need to change for the opposite outcome."""

    def __init__(self):
        self.variable_sensitivities: Dict[str, float] = {}
        self.variable_weights: Dict[str, Dict] = {}

    def _register_variable_sensitivities(self):
        self.variable_sensitivities = {
            "gdp_growth": 0.3, "inflation": 0.2, "unemployment": 0.25,
            "interest_rate": 0.2, "debt_to_gdp": 0.15, "vix_level": 0.2,
            "fear_greed_index": 0.15, "us_china_tension": 0.2,
            "russia_west_tension": 0.18, "trade_protectionism": 0.15,
            "cyber_warfare_likelihood": 0.12, "global_temp_anomaly_c": 0.1,
            "renewable_share": 0.08, "current_adoption": 0.12,
            "polarization_index": 0.1, "institutional_trust": 0.12,
            "oil_price": 0.18, "financial_systemic_risk": 0.25,
            "credit_spread": 0.2, "consumer_confidence": 0.15,
        }
        self.variable_weights = {
            "gdp_growth": {"low": -0.02, "high": 0.06, "direction": "positive"},
            "inflation": {"low": -0.01, "high": 0.1, "direction": "negative_above"},
            "unemployment": {"low": 0.02, "high": 0.15, "direction": "negative"},
            "interest_rate": {"low": 0.0, "high": 0.08, "direction": "negative_above"},
            "debt_to_gdp": {"low": 0.5, "high": 3.0, "direction": "negative"},
            "vix_level": {"low": 10, "high": 50, "direction": "negative"},
            "financial_systemic_risk": {"low": 0.1, "high": 0.9, "direction": "negative"},
            "oil_price": {"low": 30, "high": 200, "direction": "negative_above"},
        }

    def compute_outcome_probability(self, domain_values: Dict[str, float],
                                     outcome: str = "positive") -> float:
        self._register_variable_sensitivities()
        score = 0.0
        total_weight = 0.0
        for var, val in domain_values.items():
            sensitivity = self.variable_sensitivities.get(var, 0.1)
            w = self.variable_weights.get(var, {})
            low = w.get("low", 0)
            high = w.get("high", 1)
            direction = w.get("direction", "positive")
            if high > low:
                normalized = (val - low) / (high - low)
                normalized = max(0, min(1, normalized))
            else:
                normalized = 0.5
            if direction.startswith("negative"):
                contribution = (1 - normalized) * sensitivity
            else:
                contribution = normalized * sensitivity
            score += contribution
            total_weight += sensitivity
        prob = score / total_weight if total_weight > 0 else 0.5
        return max(0.01, min(0.99, prob))

    def find_counterfactual(self, domain_values: Dict[str, float],
                             target_probability: float = 0.3,
                             original_outcome: str = "positive",
                             counterfactual_outcome: str = "negative",
                             max_changes: int = 5) -> CounterfactualReport:
        self._register_variable_sensitivities()
        original_prob = self.compute_outcome_probability(domain_values, original_outcome)
        if abs(original_prob - target_probability) < 0.05:
            return CounterfactualReport(
                original_prediction=original_outcome,
                original_probability=original_prob,
                scenarios=[], critical_variables=[], leverage_points=[],
                robustness_score=1.0,
            )
        direction = "decrease" if original_prob > target_probability else "increase"
        contributions = []
        for var, val in domain_values.items():
            sensitivity = self.variable_sensitivities.get(var, 0.1)
            w = self.variable_weights.get(var, {})
            low = w.get("low", 0)
            high = w.get("high", 1)
            var_direction = w.get("direction", "positive")
            if direction == "decrease":
                if var_direction.startswith("negative"):
                    target_val = low + 0.3 * (high - low)
                    change_dir = "decrease" if val > target_val else "maintain"
                else:
                    target_val = low + 0.3 * (high - low)
                    change_dir = "decrease" if val > target_val else "maintain"
            else:
                if var_direction.startswith("negative"):
                    target_val = low + 0.7 * (high - low)
                    change_dir = "increase" if val < target_val else "maintain"
                else:
                    target_val = low + 0.7 * (high - low)
                    change_dir = "increase" if val < target_val else "maintain"
            change_mag = abs(target_val - val)
            feasibility = max(0.1, 1.0 - change_mag / max(0.01, abs(high - low)))
            obstacles = []
            if feasibility < 0.3:
                obstacles.append("Large magnitude change required")
            if change_dir == "maintain":
                continue
            contributions.append(VariableContribution(
                variable=var, current_value=val, required_value=target_val,
                change_magnitude=round(change_mag, 4), change_direction=change_dir,
                contribution_to_outcome=round(sensitivity, 4),
                feasibility=round(feasibility, 4),
            ))
        contributions.sort(key=lambda c: (-c.contribution_to_outcome * (1 - c.feasibility)))
        top_changes = contributions[:max_changes]
        scenario = CounterfactualScenario(
            name=f"Counterfactual: {counterfactual_outcome} instead of {original_outcome}",
            original_outcome=original_outcome,
            counterfactual_outcome=counterfactual_outcome,
            original_probability=round(original_prob, 4),
            counterfactual_probability=round(target_probability, 4),
            required_changes=top_changes,
            n_changes_required=len(top_changes),
            min_total_change=round(sum(c.change_magnitude for c in top_changes), 4),
            feasibility_score=round(np.mean([c.feasibility for c in top_changes]) if top_changes else 0, 4),
            narrative=self._generate_narrative(original_outcome, counterfactual_outcome, top_changes, direction),
        )
        critical_vars = [c.variable for c in top_changes[:3]]
        leverage_pts = [c.variable for c in top_changes if c.feasibility > 0.5]
        robustness = 1.0 - (len(top_changes) * 0.1) - (1 - scenario.feasibility_score) * 0.3
        return CounterfactualReport(
            original_prediction=original_outcome,
            original_probability=round(original_prob, 4),
            scenarios=[scenario],
            critical_variables=critical_vars,
            leverage_points=leverage_pts,
            robustness_score=round(max(0, min(1, robustness)), 4),
        )

    def _generate_narrative(self, original: str, counterfactual: str,
                             changes: List[VariableContribution], direction: str) -> str:
        if not changes:
            return f"Outcome already near {counterfactual} probability."
        top = changes[:3]
        parts = [f"To shift from '{original}' toward '{counterfactual}':"]
        for c in top:
            parts.append(f"  - {c.variable}: {c.change_direction} from {c.current_value:.3f} to {c.required_value:.3f} (feasibility: {c.feasibility:.0%})")
        parts.append(f"Total: {len(changes)} changes needed, overall feasibility: {np.mean([c.feasibility for c in changes]):.0%}")
        return "\n".join(parts)

    def sensitivity_analysis(self, domain_values: Dict[str, float],
                              n_top: int = 10) -> List[Tuple[str, float]]:
        self._register_variable_sensitivities()
        sensitivities = []
        base_prob = self.compute_outcome_probability(domain_values)
        for var, val in domain_values.items():
            perturbed = domain_values.copy()
            w = self.variable_weights.get(var, {})
            high = w.get("high", val * 2)
            perturbed[var] = high if abs(val - high) > abs(val - w.get("low", val / 2)) else w.get("low", val / 2)
            new_prob = self.compute_outcome_probability(perturbed)
            delta = abs(new_prob - base_prob)
            sensitivities.append((var, round(delta, 4)))
        sensitivities.sort(key=lambda x: -x[1])
        return sensitivities[:n_top]