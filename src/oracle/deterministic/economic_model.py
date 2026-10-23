import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from oracle.core.types import Domain, Prediction, ConfidenceLevel


@dataclass
class MacroState:
    gdp: float = 25.0
    debt: float = 30.0
    interest_rate: float = 0.04
    inflation: float = 0.02
    unemployment: float = 0.04
    productivity_index: float = 100.0
    money_supply: float = 20.0
    population_growth: float = 0.01
    savings_rate: float = 0.18


class EconomicDeterministicModel:
    def __init__(self):
        self.params = {
            "capital_share": 0.3,
            "labor_share": 0.7,
            "depreciation_rate": 0.05,
            "tfp_growth": 0.015,
            "inflation_target": 0.02,
            "natural_unemployment": 0.04,
            "phillips_slope": 0.5,
            "taylor_rule_inflation_weight": 1.5,
            "taylor_rule_output_weight": 0.5,
            "fiscal_multiplier": 1.5,
            "velocity_of_money": 1.6,
        }

    def solow_growth(
        self, capital: float, labor: float, tfp: float = 1.0, alpha: float = 0.3
    ) -> float:
        return tfp * (capital ** alpha) * (labor ** (1 - alpha))

    def taylor_rule(
        self, inflation: float, output_gap: float, equilibrium_rate: float = 0.02
    ) -> float:
        pi_weight = self.params["taylor_rule_inflation_weight"]
        y_weight = self.params["taylor_rule_output_weight"]
        return equilibrium_rate + pi_weight * (inflation - self.params["inflation_target"]) + y_weight * output_gap

    def phillips_curve(self, unemployment: float, expected_inflation: float = 0.02) -> float:
        u_star = self.params["natural_unemployment"]
        alpha = self.params["phillips_slope"]
        return expected_inflation - alpha * (unemployment - u_star)

    def fisher_equation(self, nominal_rate: float, expected_inflation: float) -> float:
        return (1 + nominal_rate) / (1 + expected_inflation) - 1

    def debt_dynamics(
        self, debt_gdp: float, primary_balance: float, real_growth: float, real_rate: float
    ) -> float:
        return (real_rate - real_growth) * debt_gdp - primary_balance

    def quantity_theory_of_money(self, money_supply: float, velocity: float, real_gdp: float) -> float:
        return money_supply * velocity / real_gdp

    def is_lm_model(
        self, money_supply: float, fiscal_shock: float = 0.0
    ) -> dict:
        alpha = 0.5
        beta = 0.3
        interest_rate = (alpha * money_supply / 100 - beta * fiscal_shock) * 0.01
        output = money_supply * 0.5 + fiscal_shock * self.params["fiscal_multiplier"]
        return {"interest_rate": interest_rate, "output": output}

    def simulate(
        self, state: MacroState, years: int = 10, steps: int = 120
    ) -> list[MacroState]:
        dt = years / steps
        rng = np.random.default_rng(12345)
        states = [state]
        current = MacroState(
            gdp=state.gdp, debt=state.debt, interest_rate=state.interest_rate,
            inflation=state.inflation, unemployment=state.unemployment,
            productivity_index=state.productivity_index, money_supply=state.money_supply,
            population_growth=state.population_growth, savings_rate=state.savings_rate,
        )

        for i in range(steps):
            tfp_growth = self.params["tfp_growth"] + rng.normal(0, 0.002)
            current.productivity_index *= (1 + tfp_growth * dt)
            current.gdp *= (1 + tfp_growth * dt + rng.normal(0, 0.005) * dt)
            output_gap = (current.gdp / (current.gdp * (1 + self.params["tfp_growth"])) - 1)
            suggested_rate = self.taylor_rule(current.inflation, output_gap)
            current.interest_rate += (suggested_rate - current.interest_rate) * 0.1 * dt
            current.interest_rate = np.clip(current.interest_rate, -0.02, 0.25)
            current.inflation = self.phillips_curve(current.unemployment, current.inflation * 0.7 + 0.02 * 0.3)
            current.inflation += rng.normal(0, 0.002) * dt
            current.inflation = np.clip(current.inflation, -0.05, 0.20)
            current.unemployment += rng.normal(0, 0.001) * np.sqrt(dt)
            current.unemployment = np.clip(current.unemployment, 0.02, 0.30)
            primary_balance = current.savings_rate - current.interest_rate * (current.debt / current.gdp)
            debt_change = self.debt_dynamics(current.debt / current.gdp, primary_balance, tfp_growth, current.interest_rate - current.inflation)
            current.debt = current.debt * (1 + debt_change * dt)
            current.money_supply *= (1 + current.inflation * dt + rng.normal(0, 0.01) * dt)

            states.append(MacroState(
                gdp=current.gdp, debt=current.debt, interest_rate=current.interest_rate,
                inflation=current.inflation, unemployment=current.unemployment,
                productivity_index=current.productivity_index,
                money_supply=current.money_supply,
                population_growth=current.population_growth,
                savings_rate=current.savings_rate,
            ))
        return states

    def predict(
        self,
        state: MacroState,
        target_date: date,
        confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH,
    ) -> list[Prediction]:
        years = (target_date - date.today()).days / 365.25
        sims = self.simulate(state, years=max(1, int(years)), steps=max(12, int(years * 12)))
        z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645, ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence_level]

        variables = [
            ("gdp", Domain.ECONOMICS),
            ("interest_rate", Domain.ECONOMICS),
            ("inflation", Domain.ECONOMICS),
            ("unemployment", Domain.ECONOMICS),
            ("debt_to_gdp_ratio", Domain.ECONOMICS),
            ("productivity_index", Domain.ECONOMICS),
        ]

        predictions = []
        tail = sims[-13:] if len(sims) > 13 else sims
        for var_name, domain in variables:
            if var_name == "debt_to_gdp_ratio":
                values = [s.debt / max(s.gdp, 0.01) for s in tail]
            else:
                values = [getattr(s, var_name) for s in tail]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values)) if len(values) > 1 else abs(mean_val) * 0.1
            predictions.append(Prediction(
                domain=domain, variable=var_name, value=mean_val,
                confidence_interval=(mean_val - z * std_val, mean_val + z * std_val),
                confidence_level=confidence_level, timestamp=datetime.now(),
                target_date=target_date, model_name="economic_deterministic",
                weights={"solow": 0.3, "taylor": 0.25, "phillips": 0.25, "debt_dynamics": 0.2},
            ))
        return predictions