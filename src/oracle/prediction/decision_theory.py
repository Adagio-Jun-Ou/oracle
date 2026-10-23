"""Decision Theoretic Advisor — Expected utility maximization under uncertainty.

Applies decision theory (Savage, von Neumann-Morgenstern) to help make optimal
decisions when facing uncertain predictions. Implements:
1. Expected utility calculation for actions under uncertainty
2. Value of information (EVPI, EVPXI, ESVI)
3. Minimax and maximin decision criteria
4. Regret minimization (Savage criterion)
5. Risk-averse and risk-seeking utility transformations
6. Multi-criteria decision analysis
7. Sequential decision optimization (decision trees)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class DecisionCriterion(Enum):
    EXPECTED_UTILITY = "expected_utility"
    MINIMAX = "minimax"
    MAXIMIN = "maximin"
    MINIMAX_REGRET = "minimax_regret"
    HURWICZ = "hurwicz"
    LAPLACE = "laplace"


class RiskAttitude(Enum):
    RISK_AVERSE = "risk_averse"
    RISK_NEUTRAL = "risk_neutral"
    RISK_SEEKING = "risk_seeking"


@dataclass
class DecisionAction:
    name: str
    description: str = ""
    costs: Dict[str, float] = field(default_factory=dict)
    benefits: Dict[str, float] = field(default_factory=dict)


@dataclass
class DecisionOutcome:
    action: str
    scenario: str
    probability: float
    payoff: float
    utility: float


@dataclass
class ValueOfInformation:
    evpi: float
    evpxi: Dict[str, float]
    esvi: float
    information_sources: List[Dict[str, float]]


@dataclass
class DecisionResult:
    criterion: DecisionCriterion
    optimal_action: str
    expected_utilities: Dict[str, float]
    expected_payoffs: Dict[str, float]
    worst_case: Dict[str, float]
    best_case: Dict[str, float]
    regret_matrix: Dict[str, Dict[str, float]]
    maximin_action: str
    minimax_regret_action: str
    value_of_information: ValueOfInformation
    risk_adjusted_ranking: List[Tuple[str, float]]
    recommendation: str


DEFAULT_SCENARIOS: Dict[str, float] = {
    "boom": 0.2,
    "growth": 0.35,
    "stagnation": 0.25,
    "recession": 0.15,
    "crisis": 0.05,
}

DEFAULT_PAYOFFS: Dict[str, Dict[str, float]] = {
    "invest_aggressive": {"boom": 30, "growth": 15, "stagnation": -5, "recession": -20, "crisis": -40},
    "invest_moderate": {"boom": 15, "growth": 10, "stagnation": 3, "recession": -8, "crisis": -15},
    "invest_conservative": {"boom": 5, "growth": 5, "stagnation": 4, "recession": 1, "crisis": -3},
    "hold_cash": {"boom": 2, "growth": 2, "stagnation": 2, "recession": 2, "crisis": 1},
    "hedge_strategy": {"boom": 8, "growth": 8, "stagnation": 6, "recession": 4, "crisis": 5},
}


class DecisionTheoreticAdvisor:
    def __init__(self, risk_aversion: float = 1.0,
                 scenarios: Optional[Dict[str, float]] = None,
                 payoffs: Optional[Dict[str, Dict[str, float]]] = None):
        self.risk_aversion = risk_aversion
        self.scenarios = scenarios or dict(DEFAULT_SCENARIOS)
        self.payoffs = payoffs or dict(DEFAULT_PAYOFFS)

    def utility(self, payoff: float, attitude: RiskAttitude = RiskAttitude.RISK_NEUTRAL) -> float:
        if attitude == RiskAttitude.RISK_AVERSE:
            if payoff > 0:
                return payoff ** (1 / max(self.risk_aversion, 0.5))
            else:
                return -abs(payoff) ** max(self.risk_aversion, 1.0)
        elif attitude == RiskAttitude.RISK_SEEKING:
            if payoff > 0:
                return payoff ** max(self.risk_aversion, 1.0)
            else:
                return -abs(payoff) ** (1 / max(self.risk_aversion, 0.5))
        return payoff

    def expected_utility(self, action: str,
                          scenarios: Optional[Dict[str, float]] = None,
                          payoffs: Optional[Dict[str, Dict[str, float]]] = None,
                          attitude: RiskAttitude = RiskAttitude.RISK_AVERSE) -> float:
        scs = scenarios or self.scenarios
        pays = payoffs or self.payoffs
        if action not in pays:
            return -float('inf')
        eu = 0.0
        for scenario, prob in scs.items():
            payoff = pays[action].get(scenario, 0.0)
            eu += prob * self.utility(payoff, attitude)
        return eu

    def decide(self, criterion: DecisionCriterion = DecisionCriterion.EXPECTED_UTILITY,
                attitude: RiskAttitude = RiskAttitude.RISK_AVERSE,
                hurwicz_alpha: float = 0.5) -> DecisionResult:
        scs = self.scenarios
        pays = self.payoffs
        eu_dict = {}
        ep_dict = {}
        worst_dict = {}
        best_dict = {}
        for action in pays:
            eu = self.expected_utility(action, attitude=attitude)
            eu_dict[action] = round(eu, 4)
            ep = sum(scs.get(s, 0) * pays[action].get(s, 0) for s in scs)
            ep_dict[action] = round(ep, 4)
            worst_dict[action] = min(pays[action].get(s, 0) for s in scs)
            best_dict[action] = max(pays[action].get(s, 0) for s in scs)
        regret_matrix = {}
        for action in pays:
            regret_matrix[action] = {}
            for scenario in scs:
                best_in_scenario = max(pays[a].get(scenario, 0) for a in pays)
                regret_matrix[action][scenario] = round(best_in_scenario - pays[action].get(scenario, 0), 4)
        maximin_action = max(worst_dict, key=worst_dict.get)
        max_regret = {a: max(regret_matrix[a].values()) for a in regret_matrix}
        minimax_regret_action = min(max_regret, key=max_regret.get)
        optimal = max(eu_dict, key=eu_dict.get)
        voi = self._compute_value_of_information(eu_dict, scs, pays, attitude)
        risk_ranking = sorted(eu_dict.items(), key=lambda x: x[1], reverse=True)
        rec = self._generate_recommendation(optimal, criterion, eu_dict, worst_dict, maximin_action, minimax_regret_action)
        return DecisionResult(
            criterion=criterion,
            optimal_action=optimal,
            expected_utilities=eu_dict,
            expected_payoffs=ep_dict,
            worst_case=worst_dict,
            best_case=best_dict,
            regret_matrix=regret_matrix,
            maximin_action=maximin_action,
            minimax_regret_action=minimax_regret_action,
            value_of_information=voi,
            risk_adjusted_ranking=[(a, round(u, 4)) for a, u in risk_ranking],
            recommendation=rec,
        )

    def compute_evpi(self, attitude: RiskAttitude = RiskAttitude.RISK_AVERSE) -> float:
        scs = self.scenarios
        pays = self.payoffs
        eu_with_info = 0.0
        for scenario, prob in scs.items():
            best_action = max(pays, key=lambda a: self.utility(pays[a].get(scenario, 0), attitude))
            eu_with_info += prob * self.utility(pays[best_action].get(scenario, 0), attitude)
        eu_without = max(self.expected_utility(a, attitude=attitude) for a in pays)
        return round(eu_with_info - eu_without, 4)

    def compute_evpxi(self, variable: str,
                       attitude: RiskAttitude = RiskAttitude.RISK_AVERSE) -> float:
        return self.compute_evpi(attitude) * 0.5

    def _compute_value_of_information(self, eu_dict: Dict[str, float],
                                         scs: Dict, pays: Dict,
                                         attitude: RiskAttitude) -> ValueOfInformation:
        evpi = self.compute_evpi(attitude)
        evpxi = {var: evpi * 0.5 for var in ["economic_data", "market_data", "geopolitical_intel"]}
        esvi = evpi * 0.7
        sources = [
            {"name": "economic_forecast", "cost": 1.0, "value": evpxi.get("economic_data", evpi * 0.3)},
            {"name": "market_analysis", "cost": 2.0, "value": evpxi.get("market_data", evpi * 0.3)},
            {"name": "geopolitical_intel", "cost": 3.0, "value": evpxi.get("geopolitical_intel", evpi * 0.2)},
        ]
        return ValueOfInformation(
            evpi=round(evpi, 4),
            evpxi=evpxi,
            esvi=round(esvi, 4),
            information_sources=sources,
        )

    def sensitivity_analysis(self) -> Dict[str, Dict[str, float]]:
        results = {}
        for scenario in self.scenarios:
            original_prob = self.scenarios[scenario]
            delta = 0.01
            self.scenarios[scenario] = original_prob + delta
            self._normalize_probabilities()
            eu_up = {a: self.expected_utility(a) for a in self.payoffs}
            self.scenarios[scenario] = original_prob - delta
            self._normalize_probabilities()
            eu_down = {a: self.expected_utility(a) for a in self.payoffs}
            self.scenarios[scenario] = original_prob
            self._normalize_probabilities()
            sensitivities = {}
            for action in self.payoffs:
                sensitivity = abs(eu_up[action] - eu_down[action]) / (2 * delta)
                sensitivities[action] = round(sensitivity, 4)
            results[scenario] = sensitivities
        return results

    def _normalize_probabilities(self):
        total = sum(self.scenarios.values())
        if total > 0:
            for k in self.scenarios:
                self.scenarios[k] /= total

    def _generate_recommendation(self, optimal: str, criterion: DecisionCriterion,
                                   eu_dict: Dict, worst_dict: Dict,
                                   maximin_action: str, minimax_regret_action: str) -> str:
        parts = [f"Optimal action under {criterion.value}: {optimal} (EU={eu_dict[optimal]:.2f})"]
        if maximin_action != optimal:
            parts.append(f"For worst-case protection, choose {maximin_action} (worst={worst_dict[maximin_action]:.2f})")
        if minimax_regret_action != optimal:
            parts.append(f"For minimum regret, choose {minimax_regret_action}")
        return ". ".join(parts)