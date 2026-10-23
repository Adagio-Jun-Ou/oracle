"""Portfolio Stress Tester v2 — Holistic portfolio vulnerability assessment.

Combines signals from scenario_generator, ts_forecast, regime_assess, and
anomaly_correlator into a unified portfolio stress assessment framework.

Like Mr. House running war-game simulations on every possible future,
this engine tests portfolios across scenarios, regime transitions, and
cross-domain anomaly cascades simultaneously.

Key features:
  - Multi-scenario portfolio stress testing (5 historical + custom)
  - Regime-aware stress (stresses differ by bull/bear/transition/crisis)
  - Cascade-aware stress (anomaly cascade amplification)
  - Cross-domain correlation stress (correlated drawdowns)
  - Vulnerability scoring per asset class and overall
  - Recovery time estimation
  - Worst-case compound scenario
  - Hedging recommendations based on vulnerabilities
  - Risk budget reallocation suggestions
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import date
import numpy as np

logger = logging.getLogger(__name__)


class StressScenarioType(Enum):
    HISTORICAL_2008 = "historical_2008"
    HISTORICAL_2020 = "historical_2020"
    HISTORICAL_1973 = "historical_1973"
    HISTORICAL_2000 = "historical_2000"
    HISTORICAL_1929 = "historical_1929"
    STAGFLATION = "stagflation"
    RATE_SHOCK = "rate_shock"
    GEOPOLITICAL_CRISIS = "geopolitical_crisis"
    CURRENCY_CRISIS = "currency_crisis"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    COMPOUND_WORST = "compound_worst"
    CUSTOM = "custom"


class VulnerabilityLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AssetClassStress:
    asset_class: str
    allocation: float
    stress_returns: Dict[str, float]
    max_drawdown: float
    vulnerability_score: float
    vulnerability_level: VulnerabilityLevel
    recovery_months: int


@dataclass
class HedgingRecommendation:
    hedge_type: str
    target_asset: str
    protection_level: float
    estimated_cost: float
    description: str


@dataclass
class StressResult:
    scenario: StressScenarioType
    portfolio_return: float
    portfolio_drawdown: float
    worst_day: float
    worst_week: float
    worst_month: float
    recovery_months: int
    asset_stresses: List[AssetClassStress]
    cascade_amplification: float
    regime_adjustment: float
    vulnerability_score: float
    vulnerability_level: VulnerabilityLevel


@dataclass
class PortfolioStressReport:
    portfolio_value: float
    total_allocation: float
    scenarios: List[StressResult]
    compound_worst: StressResult
    overall_vulnerability: VulnerabilityLevel
    overall_vulnerability_score: float
    most_vulnerable_asset: str
    hedging_recommendations: List[HedgingRecommendation]
    risk_reallocation: Dict[str, float]
    narrative: str
    evidence: List[str]


HISTORICAL_SCENARIOS = {
    StressScenarioType.HISTORICAL_2008: {
        "name": "2008 Global Financial Crisis",
        "equity": -0.37, "bonds": 0.05, "real_estate": -0.34,
        "commodities": -0.36, "crypto": -0.80, "cash": 0.02,
        "volatility_spike": 3.5, "correlation_spike": 0.85,
        "recovery_months": 54,
        "evidence": "S&P 500 -37%, MSCI World -42%, REITs -34%, Oil -36%",
    },
    StressScenarioType.HISTORICAL_2020: {
        "name": "2020 COVID-19 Crash & Recovery",
        "equity": -0.34, "bonds": 0.08, "real_estate": -0.22,
        "commodities": -0.30, "crypto": -0.50, "cash": 0.01,
        "volatility_spike": 4.0, "correlation_spike": 0.90,
        "recovery_months": 5,
        "evidence": "S&P 500 -34% (Feb-Mar 2020), VIX 82.69, fastest recovery in history",
    },
    StressScenarioType.HISTORICAL_1973: {
        "name": "1973-74 Stagflation Crisis",
        "equity": -0.43, "bonds": -0.10, "real_estate": -0.18,
        "commodities": 0.25, "crypto": -0.80, "cash": -0.02,
        "volatility_spike": 2.0, "correlation_spike": 0.70,
        "recovery_months": 72,
        "evidence": "S&P 500 -43%, stagflation, oil embargo, Nixon shock",
    },
    StressScenarioType.HISTORICAL_2000: {
        "name": "2000 Dot-Com Crash",
        "equity": -0.49, "bonds": 0.10, "real_estate": 0.05,
        "commodities": -0.10, "crypto": -0.80, "cash": 0.03,
        "volatility_spike": 2.5, "correlation_spike": 0.60,
        "recovery_months": 56,
        "evidence": "NASDAQ -78%, S&P 500 -49%, tech bubble burst",
    },
    StressScenarioType.HISTORICAL_1929: {
        "name": "1929 Great Depression",
        "equity": -0.89, "bonds": -0.05, "real_estate": -0.30,
        "commodities": -0.45, "crypto": -0.95, "cash": -0.10,
        "volatility_spike": 5.0, "correlation_spike": 0.95,
        "recovery_months": 150,
        "evidence": "DJIA -89%, GDP -30%, unemployment 25%",
    },
}

SYNTHETIC_SCENARIOS = {
    StressScenarioType.STAGFLATION: {
        "name": "Stagflation (high inflation + recession)",
        "equity": -0.25, "bonds": -0.15, "real_estate": -0.12,
        "commodities": 0.15, "crypto": -0.40, "cash": -0.05,
        "volatility_spike": 2.5, "correlation_spike": 0.75,
        "recovery_months": 36,
        "evidence": "CPI > 6%, GDP decline, unemployment rising (Blanchard et al. 2015)",
    },
    StressScenarioType.RATE_SHOCK: {
        "name": "Interest Rate Shock (+300bps)",
        "equity": -0.20, "bonds": -0.15, "real_estate": -0.22,
        "commodities": -0.08, "crypto": -0.35, "cash": 0.04,
        "volatility_spike": 2.0, "correlation_spike": 0.65,
        "recovery_months": 24,
        "evidence": "Fed tightening cycle impact (Bernanke & Kuttner 2005)",
    },
    StressScenarioType.GEOPOLITICAL_CRISIS: {
        "name": "Geopolitical Crisis (war/sanctions)",
        "equity": -0.18, "bonds": 0.03, "real_estate": -0.10,
        "commodities": 0.20, "crypto": -0.25, "cash": 0.02,
        "volatility_spike": 3.0, "correlation_spike": 0.80,
        "recovery_months": 18,
        "evidence": "Caldara & Iacoviello (2022) GPR Index impact study",
    },
    StressScenarioType.CURRENCY_CRISIS: {
        "name": "Currency Crisis (emerging market contagion)",
        "equity": -0.22, "bonds": -0.08, "real_estate": -0.15,
        "commodities": 0.10, "crypto": -0.30, "cash": 0.01,
        "volatility_spike": 3.5, "correlation_spike": 0.85,
        "recovery_months": 30,
        "evidence": "Kaminsky & Reinhart (1999) currency crisis patterns",
    },
    StressScenarioType.LIQUIDITY_CRISIS: {
        "name": "Liquidity Crisis (credit freeze)",
        "equity": -0.30, "bonds": -0.05, "real_estate": -0.25,
        "commodities": -0.20, "crypto": -0.50, "cash": 0.01,
        "volatility_spike": 4.5, "correlation_spike": 0.90,
        "recovery_months": 48,
        "evidence": "Brunnermeier (2009) liquidity spiral; TED spread > 300bps",
    },
}

ALL_SCENARIOS = {**HISTORICAL_SCENARIOS, **SYNTHETIC_SCENARIOS}

REGIME_STRESS_MULTIPLIER = {
    "bull": 0.8, "recovery": 0.9, "transition": 1.0,
    "uncertain": 1.1, "bear": 1.3, "crisis": 1.5,
}

ASSET_CLASSES = ["equity", "bonds", "real_estate", "commodities", "crypto", "cash"]


class PortfolioStressTester:
    """Holistic portfolio stress testing engine."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def stress_test(self, allocations: Dict[str, float],
                    portfolio_value: float = 1000000,
                    scenarios: Optional[List[StressScenarioType]] = None,
                    regime: Optional[str] = None,
                    cascade_amplification: float = 1.0,
                    ) -> PortfolioStressReport:
        total = sum(allocations.values())
        if total == 0:
            normalized = {k: 1.0 / len(allocations) for k in allocations}
        elif abs(total - 1.0) > 1e-9:
            normalized = {k: v / total for k, v in allocations.items()}
        else:
            normalized = dict(allocations)
        if scenarios is None:
            scenarios = [StressScenarioType.HISTORICAL_2008, StressScenarioType.HISTORICAL_2020,
                        StressScenarioType.STAGFLATION, StressScenarioType.RATE_SHOCK,
                        StressScenarioType.GEOPOLITICAL_CRISIS, StressScenarioType.LIQUIDITY_CRISIS,
                        StressScenarioType.HISTORICAL_1929]
        regime_mult = REGIME_STRESS_MULTIPLIER.get(regime or "uncertain", 1.0)
        results = []
        for scenario_type in scenarios:
            result = self._stress_scenario(scenario_type, normalized, portfolio_value,
                                           regime_mult, cascade_amplification)
            results.append(result)
        worst = min(results, key=lambda r: r.portfolio_return)
        compound = self._compound_worst(normalized, portfolio_value, regime_mult, cascade_amplification)
        all_scores = [r.vulnerability_score for r in results] + [compound.vulnerability_score]
        avg_vulnerability = float(np.mean(all_scores))
        vuln_level = self._vulnerability_level(avg_vulnerability)
        most_vulnerable = self._find_most_vulnerable(results)
        hedges = self._generate_hedges(normalized, results, compound)
        reallocation = self._suggest_reallocation(normalized, results)
        narrative = self._generate_narrative(results, compound, vuln_level, most_vulnerable)
        evidence = self._collect_evidence(results, scenarios)
        return PortfolioStressReport(
            portfolio_value=portfolio_value,
            total_allocation=total,
            scenarios=results,
            compound_worst=compound,
            overall_vulnerability=vuln_level,
            overall_vulnerability_score=round(avg_vulnerability, 4),
            most_vulnerable_asset=most_vulnerable,
            hedging_recommendations=hedges,
            risk_reallocation=reallocation,
            narrative=narrative,
            evidence=evidence[:8],
        )

    def _stress_scenario(self, scenario_type: StressScenarioType,
                         allocations: Dict[str, float],
                         portfolio_value: float,
                         regime_mult: float,
                         cascade_amplification: float) -> StressResult:
        scenario_data = ALL_SCENARIOS.get(scenario_type, ALL_SCENARIOS[StressScenarioType.HISTORICAL_2008])
        adj_returns = {}
        for asset in ASSET_CLASSES:
            base_return = scenario_data.get(asset, -0.10)
            adj_return = base_return * regime_mult * cascade_amplification
            adj_returns[asset] = adj_return
        asset_stresses = []
        for asset in ASSET_CLASSES:
            alloc = allocations.get(asset, 0.0)
            if alloc <= 0:
                continue
            stress_ret = adj_returns.get(asset, -0.10)
            drawdown = abs(min(0, stress_ret))
            vuln_score = min(1.0, drawdown * alloc * 2)
            asset_stresses.append(AssetClassStress(
                asset_class=asset, allocation=alloc,
                stress_returns={scenario_type.value: round(stress_ret, 4)},
                max_drawdown=round(drawdown, 4),
                vulnerability_score=round(vuln_score, 4),
                vulnerability_level=self._vulnerability_level(vuln_score),
                recovery_months=int(scenario_data.get("recovery_months", 24) * (1 + vuln_score) / 2),
            ))
        portfolio_return = sum(allocations.get(asset, 0) * adj_returns.get(asset, -0.10)
                              for asset in ASSET_CLASSES)
        volatility_spike = scenario_data.get("volatility_spike", 2.0)
        worst_day = portfolio_return * 0.04 * volatility_spike
        worst_week = portfolio_return * 0.20
        worst_month = portfolio_return * 0.60
        recovery_months = int(scenario_data.get("recovery_months", 24) * regime_mult)
        raw_vuln = min(1.0, abs(min(0, portfolio_return)) * 1.5)
        regime_adj = (regime_mult - 1.0) * 0.1
        cascade_adj = (cascade_amplification - 1.0) * 0.15
        vulnerability_score = min(1.0, raw_vuln + regime_adj + cascade_adj)
        return StressResult(
            scenario=scenario_type,
            portfolio_return=round(portfolio_return, 4),
            portfolio_drawdown=round(abs(min(0, portfolio_return)), 4),
            worst_day=round(worst_day, 4),
            worst_week=round(worst_week, 4),
            worst_month=round(worst_month, 4),
            recovery_months=recovery_months,
            asset_stresses=asset_stresses,
            cascade_amplification=round(cascade_amplification, 4),
            regime_adjustment=round(regime_mult, 4),
            vulnerability_score=round(vulnerability_score, 4),
            vulnerability_level=self._vulnerability_level(vulnerability_score),
        )

    def _compound_worst(self, allocations: Dict[str, float],
                       portfolio_value: float,
                       regime_mult: float,
                       cascade_amplification: float) -> StressResult:
        worst_returns = {}
        for asset in ASSET_CLASSES:
            worst_ret = -1.0
            for scenario_data in ALL_SCENARIOS.values():
                ret = scenario_data.get(asset, -0.10) * regime_mult * cascade_amplification
                if ret < worst_ret:
                    worst_ret = ret
            worst_returns[asset] = worst_ret
        asset_stresses = []
        for asset in ASSET_CLASSES:
            alloc = allocations.get(asset, 0.0)
            if alloc <= 0:
                continue
            stress_ret = worst_returns[asset]
            drawdown = abs(min(0, stress_ret))
            vuln_score = min(1.0, drawdown * alloc * 2)
            asset_stresses.append(AssetClassStress(
                asset_class=asset, allocation=alloc,
                stress_returns={"compound_worst": round(stress_ret, 4)},
                max_drawdown=round(drawdown, 4),
                vulnerability_score=round(vuln_score, 4),
                vulnerability_level=self._vulnerability_level(vuln_score),
                recovery_months=72,
            ))
        portfolio_return = sum(allocations.get(asset, 0) * worst_returns.get(asset, -0.10)
                              for asset in ASSET_CLASSES)
        vulnerability_score = min(1.0, abs(min(0, portfolio_return)) * 1.5 + 0.1)
        return StressResult(
            scenario=StressScenarioType.COMPOUND_WORST,
            portfolio_return=round(portfolio_return, 4),
            portfolio_drawdown=round(abs(min(0, portfolio_return)), 4),
            worst_day=round(portfolio_return * 0.15 * 4.0, 4),
            worst_week=round(portfolio_return * 0.4, 4),
            worst_month=round(portfolio_return * 0.85, 4),
            recovery_months=120,
            asset_stresses=asset_stresses,
            cascade_amplification=round(cascade_amplification, 4),
            regime_adjustment=round(regime_mult, 4),
            vulnerability_score=round(vulnerability_score, 4),
            vulnerability_level=self._vulnerability_level(vulnerability_score),
        )

    def _vulnerability_level(self, score: float) -> VulnerabilityLevel:
        if score >= 0.7:
            return VulnerabilityLevel.CRITICAL
        elif score >= 0.5:
            return VulnerabilityLevel.HIGH
        elif score >= 0.3:
            return VulnerabilityLevel.MODERATE
        else:
            return VulnerabilityLevel.LOW

    def _find_most_vulnerable(self, results: List[StressResult]) -> str:
        asset_max_vuln = {}
        for result in results:
            for as_ in result.asset_stresses:
                if as_.asset_class not in asset_max_vuln or as_.vulnerability_score > asset_max_vuln[as_.asset_class]:
                    asset_max_vuln[as_.asset_class] = as_.vulnerability_score
        if not asset_max_vuln:
            return "equity"
        return max(asset_max_vuln, key=asset_max_vuln.get)

    def _generate_hedges(self, allocations: Dict[str, float],
                         results: List[StressResult],
                         compound: StressResult) -> List[HedgingRecommendation]:
        hedges = []
        equity_alloc = allocations.get("equity", 0)
        bonds_alloc = allocations.get("bonds", 0)
        re_alloc = allocations.get("real_estate", 0)
        if equity_alloc > 0.3:
            hedges.append(HedgingRecommendation(
                hedge_type="put_options", target_asset="equity",
                protection_level=0.85,
                estimated_cost=round(equity_alloc * 0.02, 4),
                description="ATM put options on equity positions (SPY puts, ~2% annual cost)",
            ))
        if bonds_alloc > 0.2:
            hedges.append(HedgingRecommendation(
                hedge_type="duration_hedging", target_asset="bonds",
                protection_level=0.70,
                estimated_cost=round(bonds_alloc * 0.005, 4),
                description="Interest rate futures to hedge bond duration risk (~0.5% annual cost)",
            ))
        if compound.vulnerability_score > 0.5:
            hedges.append(HedgingRecommendation(
                hedge_type="tail_hedge", target_asset="portfolio",
                protection_level=0.90,
                estimated_cost=round(0.01, 4),
                description="Tail risk hedge via OTM put spreads or VIX calls (~1% annual cost)",
            ))
        if re_alloc > 0.1:
            hedges.append(HedgingRecommendation(
                hedge_type="reit_puts", target_asset="real_estate",
                protection_level=0.75,
                estimated_cost=round(re_alloc * 0.015, 4),
                description="REIT put options or reduction of real estate allocation (~1.5% annual cost)",
            ))
        if equity_alloc > 0.3 and bonds_alloc > 0.2:
            hedges.append(HedgingRecommendation(
                hedge_type="gold_allocation", target_asset="portfolio",
                protection_level=0.60,
                estimated_cost=round(0.005, 4),
                description="5-10% gold allocation as crisis hedge (low correlation during stress)",
            ))
        return hedges

    def _suggest_reallocation(self, allocations: Dict[str, float],
                              results: List[StressResult]) -> Dict[str, float]:
        worst_drawdown = {asset: 0.0 for asset in ASSET_CLASSES}
        for result in results:
            for as_ in result.asset_stresses:
                if as_.max_drawdown > worst_drawdown.get(as_.asset_class, 0):
                    worst_drawdown[as_.asset_class] = as_.max_drawdown
        total_vuln = sum(worst_drawdown.get(a, 0) * allocations.get(a, 0) for a in ASSET_CLASSES)
        suggested = {}
        for asset in ASSET_CLASSES:
            current = allocations.get(asset, 0.0)
            vuln_weight = worst_drawdown.get(asset, 0.0) * current
            if total_vuln > 0:
                risk_contribution = vuln_weight / total_vuln
                suggested[asset] = round(max(0.0, current * (1 - 0.3 * risk_contribution)), 4)
            else:
                suggested[asset] = round(current, 4)
        total_suggested = sum(suggested.values())
        if total_suggested > 0 and total_suggested != 1.0:
            suggested = {k: round(v / total_suggested, 4) for k, v in suggested.items()}
        return suggested

    def _generate_narrative(self, results: List[StressResult],
                            compound: StressResult,
                            vuln_level: VulnerabilityLevel,
                            most_vulnerable: str) -> str:
        worst_scenario = min(results, key=lambda r: r.portfolio_return)
        avg_return = float(np.mean([r.portfolio_return for r in results]))
        avg_recovery = float(np.mean([r.recovery_months for r in results]))
        parts = [
            f"Portfolio vulnerability: {vuln_level.value}.",
            f"Worst scenario: {worst_scenario.scenario.value} ({worst_scenario.portfolio_return:.1%} return).",
            f"Compound worst case: {compound.portfolio_return:.1%}.",
            f"Average stress return: {avg_return:.1%}.",
            f"Most vulnerable: {most_vulnerable}.",
            f"Average recovery: {avg_recovery:.0f} months.",
        ]
        if compound.vulnerability_score > 0.7:
            parts.append("CRITICAL: Portfolio highly vulnerable to extreme scenarios. Tail hedging recommended.")
        elif compound.vulnerability_score > 0.5:
            parts.append("ELEVATED: Portfolio vulnerability above average. Consider reducing concentration.")
        else:
            parts.append("MODERATE: Portfolio resilience acceptable but improvements possible.")
        return " ".join(parts)

    def _collect_evidence(self, results: List[StressResult],
                          scenarios: List[StressScenarioType]) -> List[str]:
        evidence = []
        for scenario_type in scenarios:
            data = ALL_SCENARIOS.get(scenario_type)
            if data and "evidence" in data:
                evidence.append(f"{data['evidence']}")
        return evidence