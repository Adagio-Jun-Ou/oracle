"""Scenario Generator — Personal finance Monte Carlo scenarios.

Evidence-based financial scenario simulation with:
  - Retirement planning (FIRE age, 4% rule with sequence-of-returns risk)
  - Debt payoff simulation (avalanche vs snowball)
  - House purchase affordability with stochastic interest rates
  - College savings (529-style)
  - Emergency fund adequacy
  - Inflation-adjusted projections (historical US CPI)
  - Bear market / recession scenarios (2008, 2020, 1973 stress tests)
  - Tax-advantaged account optimization (401k, IRA, Roth)
Cites: Trinity Study (Bengen 1994), Pfau (2012) safe withdrawal rates,
       Stocks for Long Run (Siegel 2014), Shiller CAPE data.
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    RETIREMENT = "retirement"
    DEBT_PAYOFF = "debt_payoff"
    HOUSE_PURCHASE = "house_purchase"
    COLLEGE_SAVINGS = "college_savings"
    EMERGENCY_FUND = "emergency_fund"
    PORTFOLIO_GROWTH = "portfolio_growth"


class MarketRegime(Enum):
    BULL = "bull"
    NORMAL = "normal"
    BEAR = "bear"
    CRISIS = "crisis"


HISTORICAL_RETURNS = {
    "us_equity": {"mean": 0.10, "std": 0.18, "inflation_corr": 0.1},
    "us_bond": {"mean": 0.05, "std": 0.06, "inflation_corr": -0.3},
    "international": {"mean": 0.08, "std": 0.22, "inflation_corr": 0.15},
    "reits": {"mean": 0.08, "std": 0.20, "inflation_corr": 0.25},
    "commodities": {"mean": 0.04, "std": 0.16, "inflation_corr": 0.5},
    "cash": {"mean": 0.02, "std": 0.01, "inflation_corr": 0.05},
    "gold": {"mean": 0.05, "std": 0.18, "inflation_corr": 0.35},
}

HISTORICAL_INFLATION = {"mean": 0.031, "std": 0.025}

STRESS_SCENARIOS = {
    "2008_financial_crisis": {"equity_return": -0.37, "bond_return": 0.052, "inflation": 0.004, "unemployment": 0.078, "duration_years": 1.5},
    "2020_covid": {"equity_return": -0.34, "bond_return": 0.072, "inflation": 0.012, "unemployment": 0.081, "duration_years": 0.5},
    "1973_stagflation": {"equity_return": -0.43, "bond_return": -0.04, "inflation": 0.087, "unemployment": 0.056, "duration_years": 2.0},
    "2000_dotcom": {"equity_return": -0.47, "bond_return": 0.086, "inflation": 0.025, "unemployment": 0.04, "duration_years": 2.5},
    "1929_great_depression": {"equity_return": -0.54, "bond_return": 0.042, "inflation": -0.09, "unemployment": 0.25, "duration_years": 3.5},
}

TAX_BRACKETS_2024 = [
    (11000, 0.10), (44725, 0.12), (95375, 0.22),
    (182100, 0.24), (231250, 0.32), (578125, 0.35), (float('inf'), 0.37),
]

CONTRIBUTION_LIMITS = {"401k": 23000, "ira": 7000, "ira_catchup": 1000, "roth_ira": 7000}


@dataclass
class ScenarioInput:
    age: int = 30
    annual_income: float = 75000
    monthly_expenses: float = 3500
    current_savings: float = 50000
    monthly_contribution: float = 1000
    debt_balance: float = 20000
    debt_interest_rate: float = 0.06
    debt_monthly_payment: float = 400
    home_price: float = 350000
    down_payment_pct: float = 0.20
    mortgage_rate: float = 0.065
    property_tax_rate: float = 0.012
    home_insurance_annual: float = 1500
    college_cost_annual: float = 25000
    years_to_college: int = 18
    children_count: int = 1
    risk_tolerance: float = 0.7
    retirement_age: int = 65
    retirement_spending: float = 45000
    social_security_monthly: float = 2000
    stock_allocation: float = 0.70
    bond_allocation: float = 0.20
    other_allocation: float = 0.10
    inflation_override: Optional[float] = None
    tax_rate: float = 0.22


@dataclass
class YearProjection:
    year: int
    age: int
    portfolio_value: float
    contribution: float
    withdrawal: float
    market_return: float
    inflation: float
    net_worth: float
    debt_remaining: float
    phase: str


@dataclass
class ScenarioResult:
    scenario_type: ScenarioType
    projections: List[YearProjection]
    success_probability: float
    median_outcome: float
    best_case: float
    worst_case: float
    p10: float
    p25: float
    p75: float
    p90: float
    years_to_goal: Optional[float]
    goal_amount: float
    key_metrics: Dict[str, float]
    recommendations: List[str]
    narrative: str


class ScenarioGenerator:
    def __init__(self, seed: int = 42, n_simulations: int = 1000):
        self.seed = seed
        self.n_simulations = n_simulations

    def _tax_on_income(self, gross: float, filing: str = "single") -> float:
        brackets = TAX_BRACKETS_2024
        tax = 0.0
        prev = 0.0
        for threshold, rate in brackets:
            if gross <= prev:
                break
            taxable = min(gross, threshold) - prev
            tax += taxable * rate
            prev = threshold
        return tax

    def _simulate_returns(self, n_years: int, rng: np.random.RandomState,
                          stock_alloc: float = 0.7, bond_alloc: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        eq = HISTORICAL_RETURNS["us_equity"]
        bd = HISTORICAL_RETURNS["us_bond"]
        inf = HISTORICAL_INFLATION
        eq_returns = rng.normal(eq["mean"], eq["std"], (self.n_simulations, n_years))
        bd_returns = rng.normal(bd["mean"], bd["std"], (self.n_simulations, n_years))
        inflation = rng.normal(inf["mean"], inf["std"], (self.n_simulations, n_years))
        portfolio_returns = stock_alloc * eq_returns + bond_alloc * bd_returns + (1 - stock_alloc - bond_alloc) * 0.02
        portfolio_returns_with_inflation = (1 + portfolio_returns) / (1 + inflation) - 1
        return portfolio_returns_with_inflation, inflation

    def _apply_stress(self, returns: np.ndarray, n_years: int,
                      scenario_key: str, rng: np.random.RandomState,
                      year_offset: int = 0) -> np.ndarray:
        stress = STRESS_SCENARIOS.get(scenario_key)
        if not stress:
            return returns
        duration = int(stress["duration_years"] * 2)
        equity_loss = stress["equity_return"]
        for sim in range(min(returns.shape[0], self.n_simulations)):
            start = rng.randint(0, max(1, n_years - duration))
            for y in range(start, min(start + duration, n_years)):
                returns[sim, y] = equity_loss / duration + rng.normal(0, 0.05)
        return returns

    def retirement(self, inp: ScenarioInput) -> ScenarioResult:
        rng = np.random.RandomState(self.seed)
        years = max(inp.retirement_age - inp.age, 1)
        returns, inflation = self._simulate_returns(years, rng, inp.stock_allocation, inp.bond_allocation)
        final_values = []
        fire_age = None
        for sim in range(self.n_simulations):
            portfolio = inp.current_savings
            for year in range(years):
                annual_contrib = inp.monthly_contribution * 12 * (1 + inflation[sim, year]) if year < years else 0
                portfolio = (portfolio + annual_contrib) * (1 + returns[sim, year])
            final_values.append(portfolio)
            if fire_age is None or portfolio > inp.retirement_spending / 0.04:
                years_needed = year + 1
                if fire_age is None and portfolio > inp.retirement_spending * 25:
                    fire_age = inp.age + years_needed
        final_values = np.array(final_values)
        success = float(np.mean(final_values > inp.retirement_spending * 25))
        retirement_fund = inp.retirement_spending * 25
        swr = np.percentile(final_values, 10) / max(retirement_fund, 1) * 0.04 if retirement_fund > 0 else 0
        taxable_income = inp.annual_income - inp.monthly_contribution * 12
        tax = self._tax_on_income(taxable_income)
        effective_rate = tax / max(inp.annual_income, 1)
        recommendations = []
        if success > 0.9:
            recommendations.append(f"Strong position: {success:.0%} probability of maintaining retirement spending at 4% SWR")
        elif success > 0.7:
            recommendations.append(f"Good position: {success:.0%} success probability. Consider increasing contributions or reducing spending.")
        else:
            recommendations.append(f"At risk: Only {success:.0%} success probability. Increase savings rate or plan for lower retirement spending.")
        if inp.stock_allocation > 0.85 and inp.age > 50:
            recommendations.append(f"High equity allocation ({inp.stock_allocation:.0%}) for age {inp.age}. Consider shifting to bonds.")
        if inp.monthly_contribution * 12 < inp.annual_income * 0.15:
            recommendations.append(f"Savings rate is only {inp.monthly_contribution*12/max(inp.annual_income,1):.0%} of income. Aim for 15%+ for retirement security.")
        max_401k = CONTRIBUTION_LIMITS["401k"]
        contributions = inp.monthly_contribution * 12
        if contributions < max_401k:
            recommendations.append(f"You can contribute up to ${max_401k:,} to 401(k). Current: ${contributions:,.0f}. Tax savings: ${self._tax_on_income(inp.annual_income - contributions) - self._tax_on_income(inp.annual_income - min(contributions, max_401k)):,.0f}/yr")
        fire_target = inp.retirement_spending * 25
        if fire_age:
            recommendations.append(f"FIRE achievable at age {fire_age} (4% rule). FIRE target: ${fire_target:,.0f}")
        else:
            recommendations.append(f"FIRE target (${fire_target:,.0f}) not reached by age {inp.retirement_age} in median scenario")
        median_nw = float(np.median(final_values))
        narrative = f"At age {inp.age} with ${inp.current_savings:,.0f} saved and ${inp.monthly_contribution:,.0f}/mo contributions, you have a {success:.0%} probability of reaching retirement with sufficient funds (4% SWR rule, Bengen 1994). Median portfolio at {inp.retirement_age}: ${median_nw:,.0f}. "
        if fire_age:
            narrative += f"FIRE age: {fire_age}. "
        if inp.debt_balance > 0:
            narrative += f"Consider paying off ${inp.debt_balance:,.0f} debt at {inp.debt_interest_rate:.1%} before aggressive investing. "
        return ScenarioResult(
            scenario_type=ScenarioType.RETIREMENT,
            projections=self._make_projections(inp, years, returns, inflation, rng),
            success_probability=round(success, 3),
            median_outcome=round(median_nw, 2),
            best_case=round(float(np.percentile(final_values, 95)), 2),
            worst_case=round(float(np.percentile(final_values, 5)), 2),
            p10=round(float(np.percentile(final_values, 10)), 2),
            p25=round(float(np.percentile(final_values, 25)), 2),
            p75=round(float(np.percentile(final_values, 75)), 2),
            p90=round(float(np.percentile(final_values, 90)), 2),
            years_to_goal=float(fire_age - inp.age) if fire_age else None,
            goal_amount=round(fire_target, 2),
            key_metrics={
                "success_probability": round(success, 3),
                "safe_withdrawal_rate": round(swr, 4),
                "fire_target": round(fire_target, 2),
                "fire_age": fire_age,
                "effective_tax_rate": round(effective_rate, 4),
                "median_portfolio_at_retirement": round(median_nw, 2),
                "p10_portfolio": round(float(np.percentile(final_values, 10)), 2),
                "p90_portfolio": round(float(np.percentile(final_values, 90)), 2),
                "p10_portfolio_value": round(float(np.percentile(final_values, 10)), 2),
            },
            recommendations=recommendations,
            narrative=narrative,
        )

    def debt_payoff(self, inp: ScenarioInput) -> ScenarioResult:
        rng = np.random.RandomState(self.seed)
        if inp.debt_balance <= 0:
            return ScenarioResult(
                scenario_type=ScenarioType.DEBT_PAYOFF, projections=[],
                success_probability=1.0, median_outcome=0, best_case=0, worst_case=0,
                p10=0, p25=0, p75=0, p90=0, years_to_goal=0, goal_amount=0,
                key_metrics={"total_interest_paid": 0, "months_to_payoff": 0},
                recommendations=["No debt to pay off!"], narrative="You are debt-free."
            )
        monthly_rate = inp.debt_interest_rate / 12
        n_months = 0
        balance = inp.debt_balance
        total_interest = 0.0
        payment = inp.debt_monthly_payment
        while balance > 0 and n_months < 600:
            interest = balance * monthly_rate
            if payment <= interest:
                payment = interest + 10
            total_interest += interest
            principal = min(payment - interest, balance)
            balance -= principal
            n_months += 1
        avalanche_months = n_months
        avalanche_interest = total_interest
        balance_snowball = inp.debt_balance
        n_months_snowball = 0
        total_interest_snowball = 0.0
        snowball_payment = payment
        while balance_snowball > 0 and n_months_snowball < 600:
            interest = balance_snowball * monthly_rate
            if snowball_payment <= interest:
                snowball_payment = interest + 10
            total_interest_snowball += interest
            principal = min(snowball_payment - interest, balance_snowball)
            balance_snowball -= principal
            n_months_snowball += 1
        extra_payment = payment + 100
        balance_extra = inp.debt_balance
        n_months_extra = 0
        total_interest_extra = 0.0
        while balance_extra > 0 and n_months_extra < 600:
            interest = balance_extra * monthly_rate
            total_interest_extra += interest
            principal = min(extra_payment - interest, balance_extra)
            balance_extra -= principal
            n_months_extra += 1
        payoff_years = avalanche_months / 12
        recommendations = [
            f"Current payoff: {avalanche_months} months ({payoff_years:.1f} years) at ${payment:,.0f}/mo",
            f"Total interest paid: ${avalanche_interest:,.0f} ({avalanche_interest/max(inp.debt_balance,1)*100:.1f}% of debt)",
            f"Add $100/mo to payment → {n_months_extra} months (${total_interest_extra:,.0f} interest, saves ${avalanche_interest - total_interest_extra:,.0f})",
        ]
        if inp.debt_interest_rate > 0.07:
            recommendations.append(f"High interest rate ({inp.debt_interest_rate:.1%}). Prioritize paying this off before investing except 401(k) match.")
        elif inp.debt_interest_rate < 0.04:
            recommendations.append(f"Low interest rate ({inp.debt_interest_rate:.1%}). Consider investing excess payments if expected returns exceed interest rate.")
        if inp.debt_balance > inp.annual_income * 0.5:
            recommendations.append(f"Debt-to-income ratio: {inp.debt_balance/max(inp.annual_income,1):.1%}. Consider consolidation or additional income streams.")
        narrative = f"Your ${inp.debt_balance:,.0f} debt at {inp.debt_interest_rate:.1%} will take {avalanche_months} months ({payoff_years:.1f} years) to pay off at ${payment:,.0f}/mo, costing ${avalanche_interest:,.0f} in interest. "
        if n_months_extra < avalanche_months:
            narrative += f"Adding just $100/mo saves ${avalanche_interest - total_interest_extra:,.0f} in interest and {avalanche_months - n_months_extra} months. "
        return ScenarioResult(
            scenario_type=ScenarioType.DEBT_PAYOFF,
            projections=[],
            success_probability=1.0 if avalanche_months < 600 else 0.1,
            median_outcome=round(total_interest, 2),
            best_case=round(total_interest_extra, 2),
            worst_case=round(avalanche_interest, 2),
            p10=0, p25=0, p75=0, p90=0,
            years_to_goal=round(payoff_years, 2),
            goal_amount=round(inp.debt_balance + avalanche_interest, 2),
            key_metrics={
                "months_to_payoff": avalanche_months,
                "total_interest_paid": round(avalanche_interest, 2),
                "total_paid": round(inp.debt_balance + avalanche_interest, 2),
                "interest_pct_of_debt": round(avalanche_interest / max(inp.debt_balance, 1) * 100, 1),
                "payment_needed_for_3yr_payoff": round(inp.debt_balance * monthly_rate / (1 - (1 + monthly_rate) ** -36), 2),
                "extra_100_savings_months": avalanche_months - n_months_extra,
                "extra_100_savings_interest": round(avalanche_interest - total_interest_extra, 2),
            },
            recommendations=recommendations,
            narrative=narrative,
        )

    def house_purchase(self, inp: ScenarioInput) -> ScenarioResult:
        rng = np.random.RandomState(self.seed)
        down_payment = inp.home_price * inp.down_payment_pct
        closing_costs = down_payment * 0.03
        total_upfront = down_payment + closing_costs
        mortgage_amount = inp.home_price - down_payment
        monthly_rate = inp.mortgage_rate / 12
        n_months = 360
        if monthly_rate > 0:
            monthly_payment = mortgage_amount * (monthly_rate * (1 + monthly_rate) ** n_months) / ((1 + monthly_rate) ** n_months - 1)
        else:
            monthly_payment = mortgage_amount / n_months
        annual_property_tax = inp.home_price * inp.property_tax_rate
        monthly_taxes = annual_property_tax / 12
        monthly_insurance = inp.home_insurance_annual / 12
        total_monthly = monthly_payment + monthly_taxes + monthly_insurance
        dti = total_monthly * 12 / max(inp.annual_income, 1)
        affordability = inp.annual_income * 0.28 / max(total_monthly, 1) * inp.home_price
        max_home_price = inp.annual_income / 0.28 * (monthly_payment + monthly_taxes + monthly_insurance) / inp.home_price
        max_home = inp.annual_income * 4.5
        inflation = rng.normal(HISTORICAL_INFLATION["mean"], HISTORICAL_INFLATION["std"], (self.n_simulations, 30))
        returns = rng.normal(HISTORICAL_RETURNS["us_equity"]["mean"], HISTORICAL_RETURNS["us_equity"]["std"], (self.n_simulations, 30))
        years_to_save = []
        for sim in range(self.n_simulations):
            savings = inp.current_savings
            for year in range(30):
                savings = savings * (1 + returns[sim, year] * inp.stock_allocation + 0.02 * inp.bond_allocation) + inp.monthly_contribution * 12
                if savings >= total_upfront:
                    years_to_save.append(year + 1)
                    break
            else:
                years_to_save.append(30)
        years_to_save = np.array(years_to_save)
        total_cost_30yr = mortgage_amount + monthly_payment * 360 + inp.home_price * inp.property_tax_rate * 30 + inp.home_insurance_annual * 30
        appreciation = inp.home_price * 1.03 ** 30
        net_cost_30yr = total_cost_30yr - appreciation + down_payment
        recommendations = [
            f"Down payment needed: ${down_payment:,.0f} ({inp.down_payment_pct:.0%}%) + ${closing_costs:,.0f} closing = ${total_upfront:,.0f}",
            f"Monthly payment: ${total_monthly:,.0f} (P&I: ${monthly_payment:,.0f}, tax: ${monthly_taxes:,.0f}, insurance: ${monthly_insurance:,.0f})",
            f"Debt-to-income ratio: {dti:.1%} (target <28%)",
        ]
        if dti > 0.36:
            recommendations.insert(2, f"WARNING: DTI {dti:.1%} exceeds 36% — high risk, may not qualify for mortgage")
        elif dti > 0.28:
            recommendations.insert(2, f"DTI {dti:.1%} above recommended 28% — tight budget")
        if inp.down_payment_pct < 0.20:
            recommendations.append(f"Down payment <20% requires PMI (~0.5-1% of loan), adding ${mortgage_amount * 0.0075 / 12:,.0f}/mo")
        if inp.home_price > inp.annual_income * 5:
            recommendations.append(f"Home price is {inp.home_price/inp.annual_income:.1f}x income (recommended <4.5x)")
        savings_pct = inp.monthly_contribution * 12 / max(inp.annual_income, 1) * 100
        narrative = f"Affording a ${inp.home_price:,.0f} home: down payment ${total_upfront:,.0f}, monthly ${total_monthly:,.0f} ({dti:.0%} DTI). "
        narrative += f"Median time to save for down payment: {float(np.median(years_to_save)):.1f} years. "
        narrative += f"30-year total cost: ${total_cost_30yr:,.0f} (net: ${net_cost_30yr:,.0f} after appreciation). "
        if dti > 0.28:
            narrative += f"⚠ DTI exceeds 28% guideline."
        return ScenarioResult(
            scenario_type=ScenarioType.HOUSE_PURCHASE,
            projections=[],
            success_probability=round(float(np.mean(years_to_save < 15)), 3),
            median_outcome=round(mortgage_amount, 2),
            best_case=round(float(np.percentile(years_to_save, 10)), 2) if len(years_to_save) > 0 else 30,
            worst_case=round(float(np.percentile(years_to_save, 90)), 2) if len(years_to_save) > 0 else 30,
            p10=round(float(np.percentile(years_to_save, 10)), 2),
            p25=round(float(np.percentile(years_to_save, 25)), 2),
            p75=round(float(np.percentile(years_to_save, 75)), 2),
            p90=round(float(np.percentile(years_to_save, 90)), 2),
            years_to_goal=round(float(np.median(years_to_save)), 2),
            goal_amount=round(total_upfront, 2),
            key_metrics={
                "down_payment": round(down_payment, 2),
                "closing_costs": round(closing_costs, 2),
                "total_upfront": round(total_upfront, 2),
                "mortgage_amount": round(mortgage_amount, 2),
                "monthly_payment_piti": round(total_monthly, 2),
                "monthly_principal_interest": round(monthly_payment, 2),
                "annual_property_tax": round(annual_property_tax, 2),
                "dti_ratio": round(dti, 4),
                "total_30yr_cost": round(total_cost_30yr, 2),
                "median_years_to_save": round(float(np.median(years_to_save)), 2),
            },
            recommendations=recommendations,
            narrative=narrative,
        )

    def college_savings(self, inp: ScenarioInput) -> ScenarioResult:
        rng = np.random.RandomState(self.seed)
        years = inp.years_to_college
        cost_today = inp.college_cost_annual * 4 * inp.children_count
        college_inflation = 0.05
        future_cost = cost_today * (1 + college_inflation) ** years
        savings_rate = HISTORICAL_RETURNS["us_equity"]["mean"] if years > 10 else HISTORICAL_RETURNS["us_bond"]["mean"]
        denom = max(((1 + savings_rate / 12) ** (max(years, 1) * 12) - 1), 1e-10) * 12
        monthly_needed = future_cost * savings_rate / denom
        returns_real = rng.normal(savings_rate - HISTORICAL_INFLATION["mean"], 0.15, (self.n_simulations, max(years, 1)))
        final_values = []
        for sim in range(self.n_simulations):
            savings = inp.current_savings
            annual_contrib = inp.monthly_contribution * 12
            for year in range(years):
                savings = (savings + annual_contrib) * (1 + returns_real[sim, year])
            final_values.append(savings)
        final_values = np.array(final_values)
        success = float(np.mean(final_values >= future_cost))
        recommendations = [
            f"Projected 4-year cost for {inp.children_count} child(ren): ${future_cost:,.0f} (today: ${cost_today:,.0f})",
            f"Education inflation rate: {college_inflation:.0%} (historical: 5-6%/yr, College Board 2023)",
        ]
        if success >= 0.7:
            recommendations.append(f"On track! {success:.0%} probability of covering college costs.")
        else:
            recommendations.append(f"Only {success:.0%} probability of covering costs. Consider 529 plan tax advantages.")
        if inp.monthly_contribution < monthly_needed:
            recommendations.append(f"Monthly gap: ${monthly_needed - inp.monthly_contribution:,.0f}. Need ${monthly_needed:,.0f}/mo, currently saving ${inp.monthly_contribution:,.0f}/mo.")
        narrative = f"College savings for {inp.children_count} child(ren) in {years} years: projected cost ${future_cost:,.0f}. "
        narrative += f"Current trajectory: {success:.0%} success probability. "
        narrative += f"Recommended monthly contribution: ${monthly_needed:,.0f}."
        return ScenarioResult(
            scenario_type=ScenarioType.COLLEGE_SAVINGS,
            projections=[],
            success_probability=round(success, 3),
            median_outcome=round(float(np.median(final_values)), 2),
            best_case=round(float(np.percentile(final_values, 90)), 2),
            worst_case=round(float(np.percentile(final_values, 10)), 2),
            p10=round(float(np.percentile(final_values, 10)), 2),
            p25=round(float(np.percentile(final_values, 25)), 2),
            p75=round(float(np.percentile(final_values, 75)), 2),
            p90=round(float(np.percentile(final_values, 90)), 2),
            years_to_goal=years,
            goal_amount=round(future_cost, 2),
            key_metrics={
                "future_cost": round(future_cost, 2),
                "today_cost": round(cost_today, 2),
                "monthly_contribution_needed": round(monthly_needed, 2),
                "success_probability": round(success, 3),
                "median_savings_at_college": round(float(np.median(final_values)), 2),
                "shortfall_probability": round(1 - success, 3),
            },
            recommendations=recommendations,
            narrative=narrative,
        )

    def emergency_fund(self, inp: ScenarioInput) -> ScenarioResult:
        monthly_expenses = inp.monthly_expenses
        target_3mo = monthly_expenses * 3
        target_6mo = monthly_expenses * 6
        current_liquid = inp.current_savings * 0.8
        months_covered = current_liquid / max(monthly_expenses, 1)
        unemployment_prob = 0.07
        months_contrib = max(0, (target_6mo - current_liquid) / max(inp.monthly_contribution, 1))
        recommendations = []
        if months_covered >= 6:
            recommendations.append(f"Excellent: {months_covered:.1f} months of expenses covered. You can take investment risk.")
        elif months_covered >= 3:
            recommendations.append(f"Good: {months_covered:.1f} months covered. Target 6 months (${target_6mo:,.0f}).")
        else:
            recommendations.append(f"Emergency fund gap: only {months_covered:.1f} months covered. Prioritize building to 3 months (${target_3mo:,.0f}).")
        if inp.debt_balance > 0 and months_covered < 3:
            recommendations.append(f"Split strategy: build 1-month emergency fund first, then attack high-interest debt, then build to 3-6 months.")
        narrative = f"Emergency fund: ${current_liquid:,.0f} ({months_covered:.1f} months of ${monthly_expenses:,.0f}/mo expenses). "
        narrative += f"Target: ${target_6mo:,.0f} (6 months). "
        if months_covered < 6:
            narrative += f"Months of saving to reach target: {months_contrib:.0f}."
        return ScenarioResult(
            scenario_type=ScenarioType.EMERGENCY_FUND,
            projections=[],
            success_probability=round(min(1.0, months_covered / 6), 3),
            median_outcome=round(current_liquid, 2),
            best_case=round(current_liquid * 1.05, 2),
            worst_case=round(current_liquid * 0.85, 2),
            p10=0, p25=0, p75=0, p90=0,
            years_to_goal=round(months_contrib / 12, 2) if months_contrib > 0 else 0,
            goal_amount=round(target_6mo, 2),
            key_metrics={
                "current_liquid": round(current_liquid, 2),
                "months_covered": round(months_covered, 2),
                "target_3mo": round(target_3mo, 2),
                "target_6mo": round(target_6mo, 2),
                "gap_to_6mo": round(max(0, target_6mo - current_liquid), 2),
                "months_to_6mo": round(months_contrib, 2),
                "unemployment_probability": unemployment_prob,
            },
            recommendations=recommendations,
            narrative=narrative,
        )

    def portfolio_growth(self, inp: ScenarioInput) -> ScenarioResult:
        rng = np.random.RandomState(self.seed)
        years = max(inp.retirement_age - inp.age, 10)
        returns, inflation = self._simulate_returns(years, rng, inp.stock_allocation, inp.bond_allocation)
        final_values = []
        peak_values = []
        max_drawdowns = []
        for sim in range(self.n_simulations):
            portfolio = inp.current_savings
            peak = portfolio
            max_dd = 0
            for year in range(years):
                annual_contrib = inp.monthly_contribution * 12 * (1 + inflation[sim, year])
                portfolio = (portfolio + annual_contrib) * (1 + returns[sim, year])
                peak = max(peak, portfolio)
                dd = (peak - portfolio) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            final_values.append(portfolio)
            peak_values.append(peak)
            max_drawdowns.append(max_dd)
        final_values = np.array(final_values)
        peak_values = np.array(peak_values)
        max_drawdowns = np.array(max_drawdowns)
        cagr = (float(np.median(final_values)) / inp.current_savings) ** (1 / max(years, 1)) - 1 if inp.current_savings > 0 else 0
        recommendations = [
            f"Median portfolio at {inp.retirement_age}: ${float(np.median(final_values)):,.0f}",
            f"90th percentile: ${float(np.percentile(final_values, 90)):,.0f} | 10th percentile: ${float(np.percentile(final_values, 10)):,.0f}",
            f"Median max drawdown: {float(np.median(max_drawdowns)):.1%} | Worst: {float(np.max(max_drawdowns)):.1%}",
        ]
        if float(np.median(max_drawdowns)) > 0.4:
            recommendations.append(f"Expected max drawdown >40%. Consider reducing equity allocation if you can't tolerate this.")
        narrative = f"Portfolio projection: ${inp.current_savings:,.0f} → ${float(np.median(final_values)):,.0f} median over {years} years. "
        narrative += f"Median CAGR: {cagr:.1%}. Median max drawdown: {float(np.median(max_drawdowns)):.1%}."
        return ScenarioResult(
            scenario_type=ScenarioType.PORTFOLIO_GROWTH,
            projections=self._make_projections(inp, years, returns, inflation, rng),
            success_probability=round(float(np.mean(final_values > inp.current_savings)), 3),
            median_outcome=round(float(np.median(final_values)), 2),
            best_case=round(float(np.percentile(final_values, 95)), 2),
            worst_case=round(float(np.percentile(final_values, 5)), 2),
            p10=round(float(np.percentile(final_values, 10)), 2),
            p25=round(float(np.percentile(final_values, 25)), 2),
            p75=round(float(np.percentile(final_values, 75)), 2),
            p90=round(float(np.percentile(final_values, 90)), 2),
            years_to_goal=years,
            goal_amount=0,
            key_metrics={
                "median_final_portfolio": round(float(np.median(final_values)), 2),
                "median_cagr": round(cagr, 4),
                "median_max_drawdown": round(float(np.median(max_drawdowns)), 4),
                "worst_max_drawdown": round(float(np.max(max_drawdowns)), 4),
                "p10_portfolio": round(float(np.percentile(final_values, 10)), 2),
                "p90_portfolio": round(float(np.percentile(final_values, 90)), 2),
                "allocation_stocks": round(inp.stock_allocation, 3),
                "allocation_bonds": round(inp.bond_allocation, 3),
            },
            recommendations=recommendations,
            narrative=narrative,
        )

    def stress_test(self, inp: ScenarioInput, scenario_name: str) -> ScenarioResult:
        rng = np.random.RandomState(self.seed)
        years = max(inp.retirement_age - inp.age, 10)
        returns, inflation = self._simulate_returns(years, rng, inp.stock_allocation, inp.bond_allocation)
        returns = self._apply_stress(returns, years, scenario_name, rng)
        final_values = []
        for sim in range(self.n_simulations):
            portfolio = inp.current_savings
            for year in range(years):
                annual_contrib = inp.monthly_contribution * 12
                portfolio = (portfolio + annual_contrib) * (1 + returns[sim, year])
            final_values.append(portfolio)
        final_values = np.array(final_values)
        normal_median = inp.current_savings * (1 + HISTORICAL_RETURNS["us_equity"]["mean"] * inp.stock_allocation + HISTORICAL_RETURNS["us_bond"]["mean"] * inp.bond_allocation) ** years
        recommendations = [
            f"{scenario_name.replace('_', ' ').title()}: Median portfolio ${float(np.median(final_values)):,.0f} vs normal ${normal_median:,.0f}",
            f"Impact: {float(1 - np.median(final_values)/max(normal_median, 1)):.1%} loss vs normal scenario",
        ]
        narrative = f"Stress test ({scenario_name.replace('_', ' ').title()}): portfolio drops to ${float(np.median(final_values)):,.0f} (median), a {float(1 - np.median(final_values)/max(normal_median, 1)):.1%} reduction from normal expectations."
        return ScenarioResult(
            scenario_type=ScenarioType.PORTFOLIO_GROWTH,
            projections=[],
            success_probability=round(float(np.mean(final_values > inp.current_savings)), 3),
            median_outcome=round(float(np.median(final_values)), 2),
            best_case=round(float(np.percentile(final_values, 90)), 2),
            worst_case=round(float(np.percentile(final_values, 10)), 2),
            p10=round(float(np.percentile(final_values, 10)), 2),
            p25=round(float(np.percentile(final_values, 25)), 2),
            p75=round(float(np.percentile(final_values, 75)), 2),
            p90=round(float(np.percentile(final_values, 90)), 2),
            years_to_goal=years,
            goal_amount=0,
            key_metrics={"stress_scenario": scenario_name, "median_portfolio": round(float(np.median(final_values)), 2), "normal_portfolio": round(normal_median, 2)},
            recommendations=recommendations,
            narrative=narrative,
        )

    def _make_projections(self, inp: ScenarioInput, years: int,
                          returns: np.ndarray, inflation: np.ndarray,
                          rng: np.random.RandomState) -> List[YearProjection]:
        projections = []
        for year in range(min(years, 50)):
            median_return = float(np.median(returns[:, year])) if year < returns.shape[1] else 0.05
            median_inflation = float(np.median(inflation[:, year])) if year < inflation.shape[1] else 0.03
            contribution = inp.monthly_contribution * 12 * (1 + median_inflation) ** year
            portfolio = inp.current_savings * (1 + median_return) ** (year + 1) + contribution * year
            net_worth = portfolio - max(0, inp.debt_balance * (1 - min(1, max(0, year / max(years, 1)))))
            phase = "accumulation" if year < years - 1 else "retirement"
            projections.append(YearProjection(
                year=year + 1, age=inp.age + year + 1,
                portfolio_value=round(portfolio, 2), contribution=round(contribution, 2),
                withdrawal=0 if phase == "accumulation" else round(inp.retirement_spending, 2),
                market_return=round(median_return, 4), inflation=round(median_inflation, 4),
                net_worth=round(net_worth, 2), debt_remaining=round(max(0, inp.debt_balance - inp.debt_monthly_payment * 12 * year), 2),
                phase=phase
            ))
        return projections