"""Tests for ScenarioGenerator — personal finance Monte Carlo."""
import math
import pytest
from oracle.prediction.scenario_generator import (
    ScenarioGenerator, ScenarioInput, ScenarioType, ScenarioResult,
    HISTORICAL_RETURNS, HISTORICAL_INFLATION, STRESS_SCENARIOS, TAX_BRACKETS_2024
)


@pytest.fixture
def gen():
    return ScenarioGenerator(seed=42, n_simulations=500)


@pytest.fixture
def default_input():
    return ScenarioInput()


class TestRetirement:
    def test_basic_retirement(self, gen, default_input):
        result = gen.retirement(default_input)
        assert result.scenario_type == ScenarioType.RETIREMENT
        assert 0 <= result.success_probability <= 1
        assert result.median_outcome > 0
        assert result.p10 > 0
        assert result.p90 > result.p10
        assert len(result.recommendations) >= 2

    def test_retirement_with_high_savings(self, gen):
        inp = ScenarioInput(current_savings=500000, monthly_contribution=3000, age=40, retirement_age=60)
        result = gen.retirement(inp)
        assert result.success_probability > 0.5
        assert result.median_outcome > 500000

    def test_retirement_fire_age(self, gen):
        inp = ScenarioInput(current_savings=1000000, monthly_contribution=5000, retirement_spending=30000)
        result = gen.retirement(inp)
        assert result.key_metrics.get("fire_age") is not None
        assert "FIRE" in result.narrative

    def test_retirement_low_savings(self, gen):
        inp = ScenarioInput(current_savings=0, monthly_contribution=100, age=55, retirement_age=65)
        result = gen.retirement(inp)
        assert result.success_probability < 0.5

    def test_retirement_projections(self, gen, default_input):
        result = gen.retirement(default_input)
        assert len(result.projections) > 0
        assert result.projections[0].age == default_input.age + 1


class TestDebtPayoff:
    def test_basic_debt_payoff(self, gen, default_input):
        result = gen.debt_payoff(default_input)
        assert result.scenario_type == ScenarioType.DEBT_PAYOFF
        assert result.years_to_goal > 0
        assert result.key_metrics["months_to_payoff"] > 0
        assert result.key_metrics["total_interest_paid"] > 0

    def test_no_debt(self, gen):
        inp = ScenarioInput(debt_balance=0)
        result = gen.debt_payoff(inp)
        assert result.success_probability == 1.0
        assert result.narrative == "You are debt-free."

    def test_high_interest_debt(self, gen):
        inp = ScenarioInput(debt_balance=30000, debt_interest_rate=0.20, debt_monthly_payment=800)
        result = gen.debt_payoff(inp)
        assert result.key_metrics["interest_pct_of_debt"] > 20
        assert any("High interest" in r for r in result.recommendations)

    def test_low_interest_debt(self, gen):
        inp = ScenarioInput(debt_balance=10000, debt_interest_rate=0.03, debt_monthly_payment=500)
        result = gen.debt_payoff(inp)
        assert result.years_to_goal > 0


class TestHousePurchase:
    def test_basic_house_purchase(self, gen, default_input):
        result = gen.house_purchase(default_input)
        assert result.scenario_type == ScenarioType.HOUSE_PURCHASE
        assert result.key_metrics["down_payment"] > 0
        assert result.key_metrics["monthly_payment_piti"] > 0
        assert result.key_metrics["dti_ratio"] > 0

    def test_affordable_house(self, gen):
        inp = ScenarioInput(annual_income=200000, home_price=400000, monthly_contribution=3000, current_savings=200000)
        result = gen.house_purchase(inp)
        assert result.key_metrics["dti_ratio"] < 0.36

    def test_unaffordable_house(self, gen):
        inp = ScenarioInput(annual_income=50000, home_price=800000, monthly_contribution=500, current_savings=10000)
        result = gen.house_purchase(inp)
        assert result.key_metrics["dti_ratio"] > 0.28

    def test_low_down_payment(self, gen):
        inp = ScenarioInput(down_payment_pct=0.05, home_price=400000)
        result = gen.house_purchase(inp)
        assert any("PMI" in r for r in result.recommendations)


class TestCollegeSavings:
    def test_basic_college(self, gen, default_input):
        result = gen.college_savings(default_input)
        assert result.scenario_type == ScenarioType.COLLEGE_SAVINGS
        assert result.goal_amount > 0
        assert 0 <= result.success_probability <= 1

    def test_college_near_term(self, gen):
        inp = ScenarioInput(years_to_college=2, current_savings=100000, monthly_contribution=2000)
        result = gen.college_savings(inp)
        assert result.success_probability >= 0

    def test_college_far_term(self, gen):
        inp = ScenarioInput(years_to_college=18, current_savings=0, monthly_contribution=500)
        result = gen.college_savings(inp)
        assert result.goal_amount > result.key_metrics["today_cost"]


class TestEmergencyFund:
    def test_basic_emergency(self, gen, default_input):
        result = gen.emergency_fund(default_input)
        assert result.scenario_type == ScenarioType.EMERGENCY_FUND
        assert result.key_metrics["months_covered"] > 0
        assert result.key_metrics["target_6mo"] > 0

    def test_well_funded_emergency(self, gen):
        inp = ScenarioInput(current_savings=100000, monthly_expenses=3000)
        result = gen.emergency_fund(inp)
        assert result.key_metrics["months_covered"] > 6

    def test_underfunded_emergency(self, gen):
        inp = ScenarioInput(current_savings=1000, monthly_expenses=5000)
        result = gen.emergency_fund(inp)
        assert result.key_metrics["months_covered"] < 3


class TestPortfolioGrowth:
    def test_basic_portfolio(self, gen, default_input):
        result = gen.portfolio_growth(default_input)
        assert result.scenario_type == ScenarioType.PORTFOLIO_GROWTH
        assert result.median_outcome > 0
        assert result.key_metrics["median_cagr"] is not None

    def test_aggressive_portfolio(self, gen):
        inp = ScenarioInput(stock_allocation=0.90, bond_allocation=0.10, current_savings=100000)
        result = gen.portfolio_growth(inp)
        assert result.key_metrics["allocation_stocks"] == 0.90

    def test_conservative_portfolio(self, gen):
        inp = ScenarioInput(stock_allocation=0.30, bond_allocation=0.60, current_savings=100000)
        result = gen.portfolio_growth(inp)
        assert result.key_metrics["allocation_stocks"] == 0.30


class TestStressTest:
    def test_2008_stress(self, gen, default_input):
        result = gen.stress_test(default_input, "2008_financial_crisis")
        assert result.scenario_type == ScenarioType.PORTFOLIO_GROWTH
        assert result.median_outcome > 0
        assert "2008" in result.narrative

    def test_2020_stress(self, gen, default_input):
        result = gen.stress_test(default_input, "2020_covid")
        assert result.median_outcome > 0

    def test_stagflation_stress(self, gen, default_input):
        result = gen.stress_test(default_input, "1973_stagflation")
        assert result.median_outcome > 0

    def test_all_stress_scenarios(self, gen, default_input):
        for name in STRESS_SCENARIOS:
            result = gen.stress_test(default_input, name)
            assert result.median_outcome > 0
            assert len(result.recommendations) > 0


class TestTaxCalculation:
    def test_tax_on_income(self, gen):
        assert gen._tax_on_income(50000) > 0
        assert gen._tax_on_income(50000) < 50000 * 0.25

    def test_tax_brackets(self, gen):
        tax_10k = gen._tax_on_income(10000)
        tax_100k = gen._tax_on_income(100000)
        tax_500k = gen._tax_on_income(500000)
        assert tax_10k < tax_100k < tax_500k
        assert tax_10k < 10000 * 0.10 + 100
        assert tax_100k > 100000 * 0.15


class TestReproducibility:
    def test_same_seed_same_result(self):
        g1 = ScenarioGenerator(seed=42, n_simulations=500)
        g2 = ScenarioGenerator(seed=42, n_simulations=500)
        inp = ScenarioInput()
        r1 = g1.retirement(inp)
        r2 = g2.retirement(inp)
        assert abs(r1.median_outcome - r2.median_outcome) < 1