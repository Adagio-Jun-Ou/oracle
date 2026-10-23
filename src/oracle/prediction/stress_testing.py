"""Stress Testing Engine — Systematic scenario stress testing for predictions.

Applies systematic stress scenarios to prediction models:
1. Historical stress scenarios (2008 GFC, 2020 COVID, 1987 Black Monday, etc.)
2. Reverse stress testing (what scenario would break this prediction?)
3. Sensitivity-based stress (vary most impactful parameters)
4. Tail risk stress (extreme quantile scenarios)
5. Compound stress (multiple simultaneous stresses)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class StressScenarioType(Enum):
    HISTORICAL = "historical"
    REVERSE = "reverse"
    SENSITIVITY = "sensitivity"
    TAIL_RISK = "tail_risk"
    COMPOUND = "compound"


@dataclass
class StressScenario:
    name: str
    scenario_type: StressScenarioType
    description: str
    parameter_shocks: Dict[str, float]
    probability: float
    severity: float
    historical_precedent: Optional[str] = None


@dataclass
class StressTestResult:
    scenario: StressScenario
    original_prediction: Dict[str, float]
    stressed_prediction: Dict[str, float]
    impact: Dict[str, float]
    max_impact_variable: str
    max_impact_magnitude: float
    vulnerability_score: float
    passed: bool


@dataclass
class StressTestReport:
    results: List[StressTestResult]
    overall_vulnerability: float
    most_vulnerable_variables: List[str]
    worst_case_impact: float
    worst_case_scenario: str
    resilience_score: float
    recommendations: List[str]


HISTORICAL_SCENARIOS: List[Dict] = [
    {"name": "2008_financial_crisis", "description": "Global financial crisis: credit freeze, market crash, recession",
     "shocks": {"gdp_growth": -0.04, "unemployment": 0.10, "stock_market": -0.50, "credit_spread": 0.05, "housing": -0.30},
     "probability": 0.05, "severity": 0.95, "precedent": "2008-2009 Global Financial Crisis"},
    {"name": "covid_19_shock", "description": "Pandemic: simultaneous supply/demand shock, policy response",
     "shocks": {"gdp_growth": -0.05, "unemployment": 0.08, "stock_market": -0.35, "supply_disruption": 0.8, "policy_response": 0.9},
     "probability": 0.08, "severity": 0.85, "precedent": "COVID-19 Pandemic 2020"},
    {"name": "black_monday_1987", "description": "Sudden market crash driven by portfolio insurance and program trading",
     "shocks": {"stock_market": -0.25, "volatility": 0.80, "liquidity": -0.70},
     "probability": 0.03, "severity": 0.70, "precedent": "Black Monday October 1987"},
    {"name": "stagflation_shock", "description": "Oil shock + loose monetary policy = stagflation",
     "shocks": {"inflation": 0.08, "gdp_growth": -0.02, "unemployment": 0.05, "oil_price": 0.60},
     "probability": 0.10, "severity": 0.65, "precedent": "1970s Stagflation"},
    {"name": "tech_bubble_burst", "description": "Technology sector collapse with spillover",
     "shocks": {"stock_market": -0.40, "tech_sector": -0.70, "venture_capital": -0.60, "sentiment": -0.50},
     "probability": 0.06, "severity": 0.70, "precedent": "Dot-com Bubble 2000"},
    {"name": "sovereign_debt_crisis", "description": "Government debt crisis with banking nexus",
     "shocks": {"sovereign_yield": 0.05, "banking_stress": 0.80, "gdp_growth": -0.03, "fiscal_austerity": 0.6},
     "probability": 0.04, "severity": 0.80, "precedent": "European Debt Crisis 2010"},
    {"name": "geopolitical_shock", "description": "Major geopolitical disruption (war, trade war, sanctions)",
     "shocks": {"geopolitical_risk": 0.90, "trade_volume": -0.30, "oil_price": 0.40, "uncertainty": 0.80},
     "probability": 0.12, "severity": 0.60, "precedent": "Various geopolitical shocks"},
    {"name": "climate_disaster", "description": "Major climate-related infrastructure disruption",
     "shocks": {"infrastructure_damage": 0.70, "agriculture": -0.30, "insurance_losses": 0.80, "displacement": 0.40},
     "probability": 0.08, "severity": 0.75, "precedent": "Increasing climate disasters"},
]


class StressTestingEngine:
    def __init__(self, scenarios: Optional[List[Dict]] = None):
        self.scenarios: List[StressScenario] = []
        source = scenarios if scenarios is not None else HISTORICAL_SCENARIOS
        for s in source:
            if isinstance(s, StressScenario):
                self.scenarios.append(s)
            else:
                self.scenarios.append(StressScenario(
                    name=s["name"],
                    scenario_type=StressScenarioType.HISTORICAL if s.get("precedent") else StressScenarioType.COMPOUND,
                    description=s["description"],
                    parameter_shocks=s["shocks"],
                    probability=s["probability"],
                    severity=s["severity"],
                    historical_precedent=s.get("precedent"),
                ))

    def stress_test(self, predictions: Dict[str, float],
                     sensitivities: Optional[Dict[str, float]] = None,
                     threshold: float = 0.5) -> StressTestReport:
        results = []
        for scenario in self.scenarios:
            result = self._apply_stress(predictions, scenario, threshold)
            results.append(result)
        results.extend(self._sensitivity_stress(predictions, sensitivities, threshold))
        results.extend(self._reverse_stress(predictions, threshold))
        results.extend(self._tail_risk_stress(predictions, threshold))
        vulnerabilities = self._compute_vulnerabilities(results)
        worst = min(results, key=lambda r: r.vulnerability_score) if results else None
        resilience = 1.0 - (worst.vulnerability_score if worst else 0.5)
        recs = self._generate_recommendations(results, worst)
        return StressTestReport(
            results=results,
            overall_vulnerability=round(float(np.mean([r.vulnerability_score for r in results])) if results else 0.5, 4),
            most_vulnerable_variables=vulnerabilities[:5],
            worst_case_impact=round(worst.max_impact_magnitude, 4) if worst else 0.0,
            worst_case_scenario=worst.scenario.name if worst else "none",
            resilience_score=round(max(0, resilience), 4),
            recommendations=recs,
        )

    def _apply_stress(self, predictions: Dict[str, float],
                       scenario: StressScenario,
                       threshold: float) -> StressTestResult:
        stressed = {}
        impact = {}
        for var, pred_val in predictions.items():
            shock = scenario.parameter_shocks.get(var, 0.0)
            stressed_val = pred_val * (1 + shock)
            stressed_val = max(-1.0, min(2.0, stressed_val))
            stressed[var] = round(stressed_val, 4)
            impact[var] = round(abs(stressed_val - pred_val), 4)
        max_var = max(impact, key=impact.get) if impact else "none"
        max_mag = impact.get(max_var, 0.0)
        vulnerability = min(1.0, max_mag * scenario.severity)
        passed = vulnerability < threshold
        return StressTestResult(
            scenario=scenario,
            original_prediction={k: round(v, 4) for k, v in predictions.items()},
            stressed_prediction=stressed,
            impact=impact,
            max_impact_variable=max_var,
            max_impact_magnitude=round(max_mag, 4),
            vulnerability_score=round(vulnerability, 4),
            passed=passed,
        )

    def _sensitivity_stress(self, predictions: Dict[str, float],
                              sensitivities: Optional[Dict[str, float]],
                              threshold: float) -> List[StressTestResult]:
        results = []
        if sensitivities is None:
            for var in list(predictions.keys())[:5]:
                scenario = StressScenario(
                    name=f"sensitivity_{var}",
                    scenario_type=StressScenarioType.SENSITIVITY,
                    description=f"Sensitivity stress test: {var} shocked by 30%",
                    parameter_shocks={var: 0.3},
                    probability=0.2, severity=0.5,
                )
                results.append(self._apply_stress(predictions, scenario, threshold))
        else:
            for var, sens in list(sensitivities.items())[:10]:
                shock_val = float(sens) * 2.0
                scenario = StressScenario(
                    name=f"sensitivity_{var}",
                    scenario_type=StressScenarioType.SENSITIVITY,
                    description=f"Sensitivity stress: {var} shocked by {shock_val:.0%} (2x sensitivity)",
                    parameter_shocks={var: shock_val},
                    probability=0.15,
                    severity=min(1.0, abs(shock_val)),
                )
                results.append(self._apply_stress(predictions, scenario, threshold))
        return results

    def _reverse_stress(self, predictions: Dict[str, float],
                         threshold: float) -> List[StressTestResult]:
        results = []
        for var in list(predictions.keys())[:5]:
            break_shock = -(0.5 + predictions.get(var, 0.5))
            scenario = StressScenario(
                name=f"reverse_break_{var}",
                scenario_type=StressScenarioType.REVERSE,
                description=f"Reverse stress: What would break {var}?",
                parameter_shocks={var: break_shock},
                probability=0.02, severity=0.9,
            )
            results.append(self._apply_stress(predictions, scenario, threshold))
        return results

    def _tail_risk_stress(self, predictions: Dict[str, float],
                            threshold: float) -> List[StressTestResult]:
        results = []
        for var in list(predictions.keys())[:5]:
            for pct in [3, 1, 0.5]:
                shock = -0.3 * (3.0 / pct)
                scenario = StressScenario(
                    name=f"tail_risk_{var}_{pct}pct",
                    scenario_type=StressScenarioType.TAIL_RISK,
                    description=f"Tail risk: {var} at {pct}th percentile stress ({shock:.0%} shock)",
                    parameter_shocks={var: shock},
                    probability=pct / 100, severity=0.9,
                )
                results.append(self._apply_stress(predictions, scenario, threshold))
        return results

    def _compute_vulnerabilities(self, results: List[StressTestResult]) -> List[str]:
        var_impact: Dict[str, float] = {}
        for r in results:
            if r.max_impact_variable != "none":
                var_impact[r.max_impact_variable] = var_impact.get(r.max_impact_variable, 0) + r.max_impact_magnitude
        sorted_vars = sorted(var_impact.items(), key=lambda x: x[1], reverse=True)
        return [v[0] for v in sorted_vars]

    def _generate_recommendations(self, results: List[StressTestResult],
                                   worst: Optional[StressTestResult]) -> List[str]:
        recs = []
        failed = [r for r in results if not r.passed]
        if failed:
            recs.append(f"STRESS TESTING: {len(failed)}/{len(results)} scenarios exceed vulnerability threshold")
        if worst:
            recs.append(f"WORST CASE: {worst.scenario.name} — vulnerability {worst.vulnerability_score:.0%}, impact on {worst.max_impact_variable}")
        vulnerabilities = self._compute_vulnerabilities(results)
        if vulnerabilities:
            recs.append(f"TOP VULNERABILITY: {vulnerabilities[0]} is most impacted across scenarios")
        historical = [r for r in results if r.scenario.scenario_type == StressScenarioType.HISTORICAL]
        if historical:
            worst_hist = max(historical, key=lambda r: r.vulnerability_score)
            recs.append(f"HISTORICAL STRESS: Worst historical scenario is {worst_hist.scenario.name} (severity={worst_hist.scenario.severity:.0%})")
        resilience = 1.0 - (worst.vulnerability_score if worst else 0.5)
        if resilience < 0.3:
            recs.append("LOW RESILIENCE: System is highly vulnerable to stress — strengthen risk management")
        elif resilience < 0.6:
            recs.append("MODERATE RESILIENCE: Some vulnerabilities identified — address top risks")
        else:
            recs.append("GOOD RESILIENCE: System handles most stress scenarios adequately")
        return recs