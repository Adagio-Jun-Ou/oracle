"""Tests for PortfolioStressTester — holistic portfolio vulnerability assessment."""
import pytest
import numpy as np
from oracle.prediction.portfolio_stress import (
    PortfolioStressTester, StressScenarioType, VulnerabilityLevel,
    StressResult, PortfolioStressReport, AssetClassStress, HedgingRecommendation,
    HISTORICAL_SCENARIOS, SYNTHETIC_SCENARIOS, ALL_SCENARIOS,
    ASSET_CLASSES, REGIME_STRESS_MULTIPLIER,
)


@pytest.fixture
def tester():
    return PortfolioStressTester(seed=42)


BALANCED_PORTFOLIO = {
    "equity": 0.50, "bonds": 0.25, "real_estate": 0.10,
    "commodities": 0.05, "crypto": 0.03, "cash": 0.07,
}

CONSERVATIVE_PORTFOLIO = {
    "equity": 0.20, "bonds": 0.50, "real_estate": 0.05,
    "commodities": 0.02, "crypto": 0.00, "cash": 0.23,
}

AGGRESSIVE_PORTFOLIO = {
    "equity": 0.70, "bonds": 0.10, "real_estate": 0.10,
    "commodities": 0.03, "crypto": 0.07, "cash": 0.00,
}


class TestBasicStress:
    def test_balanced_portfolio(self, tester):
        report = tester.stress_test(BALANCED_PORTFOLIO)
        assert len(report.scenarios) >= 5
        assert report.portfolio_value == 1000000
        assert 0 < report.overall_vulnerability_score <= 1
        assert isinstance(report.overall_vulnerability, VulnerabilityLevel)

    def test_conservative_portfolio(self, tester):
        report = tester.stress_test(CONSERVATIVE_PORTFOLIO)
        assert report.overall_vulnerability_score < 0.9

    def test_aggressive_portfolio(self, tester):
        report = tester.stress_test(AGGRESSIVE_PORTFOLIO)
        assert report.overall_vulnerability_score > 0

    def test_single_scenario(self, tester):
        report = tester.stress_test(
            BALANCED_PORTFOLIO,
            scenarios=[StressScenarioType.HISTORICAL_2008],
        )
        assert len(report.scenarios) == 1
        assert report.scenarios[0].scenario == StressScenarioType.HISTORICAL_2008

    def test_all_historical_scenarios(self, tester):
        report = tester.stress_test(
            BALANCED_PORTFOLIO,
            scenarios=list(HISTORICAL_SCENARIOS.keys()),
        )
        assert len(report.scenarios) == 5


class TestScenarioData:
    def test_historical_scenarios_exist(self):
        assert len(HISTORICAL_SCENARIOS) == 5

    def test_synthetic_scenarios_exist(self):
        assert len(SYNTHETIC_SCENARIOS) == 5

    def test_all_scenarios_complete(self):
        for name, data in ALL_SCENARIOS.items():
            assert "equity" in data
            assert "bonds" in data
            assert "recovery_months" in data
            assert "evidence" in data

    def test_regime_multipliers(self):
        assert "bull" in REGIME_STRESS_MULTIPLIER
        assert "crisis" in REGIME_STRESS_MULTIPLIER
        assert REGIME_STRESS_MULTIPLIER["crisis"] > REGIME_STRESS_MULTIPLIER["bull"]


class TestStressResults:
    def test_portfolio_return_negative(self, tester):
        report = tester.stress_test(BALANCED_PORTFOLIO)
        for scenario in report.scenarios:
            assert scenario.portfolio_return < 0

    def test_drawdown_correct(self, tester):
        report = tester.stress_test(BALANCED_PORTFOLIO)
        for scenario in report.scenarios:
            if scenario.portfolio_return < 0:
                assert scenario.portfolio_drawdown > 0
                assert abs(scenario.portfolio_drawdown - abs(scenario.portfolio_return)) < 0.01

    def test_worst_periods(self, tester):
        report = tester.stress_test(BALANCED_PORTFOLIO)
        for scenario in report.scenarios:
            if scenario.portfolio_return < 0:
                assert abs(scenario.worst_day) <= abs(scenario.worst_week)
                assert abs(scenario.worst_week) <= abs(scenario.worst_month)

    def test_recovery_months_positive(self, tester):
        report = tester.stress_test(BALANCED_PORTFOLIO)
        for scenario in report.scenarios:
            assert scenario.recovery_months > 0

    def test_asset_stresses(self, tester):
        report = tester.stress_test(BALANCED_PORTFOLIO)
        for scenario in report.scenarios:
            assert len(scenario.asset_stresses) >= 3
            for as_ in scenario.asset_stresses:
                assert 0 <= as_.vulnerability_score <= 1


class TestRegimeEffects:
    def test_bull_regime_less_stress(self, tester):
        report_bull = tester.stress_test(BALANCED_PORTFOLIO, regime="bull")
        report_bear = tester.stress_test(BALANCED_PORTFOLIO, regime="bear")
        bull_returns = [s.portfolio_return for s in report_bull.scenarios]
        bear_returns = [s.portfolio_return for s in report_bear.scenarios]
        assert np.mean(bull_returns) > np.mean(bear_returns)

    def test_crisis_regime_worst(self, tester):
        report_normal = tester.stress_test(BALANCED_PORTFOLIO, regime="uncertain")
        report_crisis = tester.stress_test(BALANCED_PORTFOLIO, regime="crisis")
        assert report_crisis.overall_vulnerability_score >= report_normal.overall_vulnerability_score


class TestCascadeAmplification:
    def test_cascade_increases_stress(self, tester):
        report_normal = tester.stress_test(BALANCED_PORTFOLIO, cascade_amplification=1.0)
        report_cascade = tester.stress_test(BALANCED_PORTFOLIO, cascade_amplification=1.3)
        normal_worst = min(s.portfolio_return for s in report_normal.scenarios)
        cascade_worst = min(s.portfolio_return for s in report_cascade.scenarios)
        assert cascade_worst <= normal_worst

    def test_cascade_in_vulnerability(self, tester):
        report_normal = tester.stress_test(BALANCED_PORTFOLIO, cascade_amplification=1.0)
        report_cascade = tester.stress_test(BALANCED_PORTFOLIO, cascade_amplification=1.5)
        assert report_cascade.overall_vulnerability_score >= report_normal.overall_vulnerability_score


class TestCompoundWorst:
    def test_compound_worst_in_report(self, tester):
        report = tester.stress_test(BALANCED_PORTFOLIO)
        assert report.compound_worst is not None
        assert report.compound_worst.scenario == StressScenarioType.COMPOUND_WORST

    def test_compound_worse_than_individual(self, tester):
        report = tester.stress_test(BALANCED_PORTFOLIO)
        worst_individual = min(s.portfolio_return for s in report.scenarios)
        assert report.compound_worst.portfolio_return <= worst_individual + 0.01


class TestVulnerability:
    def test_vulnerability_levels(self, tester):
        report = tester.stress_test(BALANCED_PORTFOLIO)
        assert report.overall_vulnerability in VulnerabilityLevel
        for scenario in report.scenarios:
            assert scenario.vulnerability_level in VulnerabilityLevel

    def test_most_vulnerable_asset(self, tester):
        report = tester.stress_test(BALANCED_PORTFOLIO)
        assert report.most_vulnerable_asset in ASSET_CLASSES

    def test_conservative_less_vulnerable(self, tester):
        r_cons = tester.stress_test(CONSERVATIVE_PORTFOLIO)
        r_agg = tester.stress_test(AGGRESSIVE_PORTFOLIO)
        assert r_cons.overall_vulnerability_score <= r_agg.overall_vulnerability_score + 0.05


class TestHedging:
    def test_hedges_generated(self, tester):
        report = tester.stress_test(BALANCED_PORTFOLIO)
        assert len(report.hedging_recommendations) >= 1
        for hedge in report.hedging_recommendations:
            assert hedge.hedge_type
            assert hedge.target_asset
            assert hedge.protection_level > 0
            assert hedge.estimated_cost > 0

    def test_equity_hedge_for_high_equity(self, tester):
        report = tester.stress_test(AGGRESSIVE_PORTFOLIO)
        hedge_types = [h.hedge_type for h in report.hedging_recommendations]
        assert "put_options" in hedge_types


class TestReallocation:
    def test_reallocation_suggestions(self, tester):
        report = tester.stress_test(BALANCED_PORTFOLIO)
        assert len(report.risk_reallocation) >= 3
        total = sum(report.risk_reallocation.values())
        assert abs(total - 1.0) < 0.05

    def test_reallocation_reduces_risk(self, tester):
        report = tester.stress_test(AGGRESSIVE_PORTFOLIO)
        original_equity = AGGRESSIVE_PORTFOLIO.get("equity", 0)
        suggested_equity = report.risk_reallocation.get("equity", 1)
        if original_equity > 0.3:
            assert suggested_equity <= original_equity + 0.05


class TestNarrative:
    def test_narrative_not_empty(self, tester):
        report = tester.stress_test(BALANCED_PORTFOLIO)
        assert report.narrative
        assert len(report.narrative) > 50

    def test_evidence_collected(self, tester):
        report = tester.stress_test(
            BALANCED_PORTFOLIO,
            scenarios=[StressScenarioType.HISTORICAL_2008, StressScenarioType.HISTORICAL_2020],
        )
        assert len(report.evidence) >= 2