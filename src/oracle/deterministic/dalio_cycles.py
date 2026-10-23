import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
from enum import Enum
from oracle.core.types import (
    EconomicCycle, Domain, Prediction, ConfidenceLevel, TimeSeries, DataPoint
)


class CyclePhase(Enum):
    EARLY_EXPANSION = "early_expansion"
    MID_EXPANSION = "mid_expansion"
    LATE_EXPANSION = "late_expansion"
    PEAK = "peak"
    EARLY_RECESSION = "early_recession"
    MID_RECESSION = "mid_recession"
    LATE_RECESSION = "late_recession"
    TROUGH = "trough"


@dataclass
class DebtCycleState:
    gdp_growth_rate: float = 0.02
    debt_to_gdp: float = 1.0
    interest_rate: float = 0.03
    money_supply_growth: float = 0.05
    credit_growth: float = 0.04
    productivity_growth: float = 0.015
    inflation_rate: float = 0.02
    unemployment_rate: float = 0.05
    stock_valuation: float = 1.0
    capacity_utilization: float = 0.80
    timestamp: date = field(default_factory=date.today)


@dataclass
class CycleDiagnosis:
    phase: CyclePhase
    short_term_cycle_position: float
    long_term_cycle_position: float
    productivity_trend_position: float
    debt_sustainability_score: float
    bubble_risk_score: float
    deleveraging_risk_score: float
    policy_effectiveness: float
    key_signals: dict = field(default_factory=dict)
    historical_analogues: list[str] = field(default_factory=list)
    confidence: float = 0.5


class DalioCycleEngine:
    def __init__(self):
        self.short_term_debt_cycle_period = (5, 8)
        self.long_term_debt_cycle_period = (75, 100)
        self.productivity_cycle_period = (50, 75)
        self.historical_events = self._init_historical_events()
        self._state_stack: list[DebtCycleState] = []

    def _init_historical_events(self) -> list[dict]:
        return [
            {"name": "US Great Depression", "start": 1929, "end": 1939, "type": "deleveraging_deflationary", "gdp_decline": -0.30, "debt_peak": 2.7, "unemployment_peak": 0.25},
            {"name": "US Post-WWII Boom", "start": 1946, "end": 1966, "type": "expansion_productive", "gdp_growth": 0.04, "debt_trough": 0.3},
            {"name": "US Stagflation 70s", "start": 1970, "end": 1982, "type": "stagflation", "gdp_decline": -0.05, "inflation_peak": 0.14, "oil_shock": True},
            {"name": "Japan Lost Decade", "start": 1990, "end": 2005, "type": "deleveraging_deflationary", "gdp_decline": -0.15, "debt_peak": 2.4},
            {"name": "US Dot-Com Bubble", "start": 1998, "end": 2002, "type": "bubble_burst", "gdp_decline": -0.05, "valuation_peak": 3.0},
            {"name": "US Housing Crisis / GFC", "start": 2007, "end": 2012, "type": "deleveraging_deflationary", "gdp_decline": -0.05, "debt_peak": 3.5, "unemployment_peak": 0.10},
            {"name": "Euro Debt Crisis", "start": 2010, "end": 2015, "type": "sovereign_debt", "debt_peak": 4.0},
            {"name": "COVID Crash & Recovery", "start": 2020, "end": 2022, "type": "exogenous_shock", "gdp_decline": -0.10, "stimulus": True},
            {"name": "US Inflation Surge", "start": 2021, "end": 2024, "type": "inflationary_cycle", "inflation_peak": 0.09},
            {"name": "US 1944 Bretton Woods", "start": 1944, "end": 1971, "type": "monetary_paradigm", "paradigm": "gold_standard"},
            {"name": "Nixon Shock / Fiat Era", "start": 1971, "end": 2020, "type": "monetary_paradigm", "paradigm": "fiat_dollar"},
        ]

    def diagnose_cycle(self, state: DebtCycleState) -> CycleDiagnosis:
        short_position = self._compute_short_cycle_position(state)
        long_position = self._compute_long_cycle_position(state)
        productivity_position = self._compute_productivity_position(state)
        phase = self._determine_phase(state, short_position, long_position)
        debt_score = self._debt_sustainability(state)
        bubble_score = self._bubble_risk(state)
        deleveraging_score = self._deleveraging_risk(state)
        policy_score = self._policy_effectiveness(state)

        analogues = self._find_historical_analogues(state, phase)

        key_signals = {
            "debt_to_gdp": state.debt_to_gdp,
            "interest_rate": state.interest_rate,
            "credit_growth": state.credit_growth,
            "capacity_utilization": state.capacity_utilization,
            "inflation_rate": state.inflation_rate,
            "valuation_percentile": state.stock_valuation,
            "funding_spread": max(0, state.interest_rate - state.gdp_growth_rate),
        }

        confidence = self._compute_confidence(state, key_signals)

        return CycleDiagnosis(
            phase=phase,
            short_term_cycle_position=short_position,
            long_term_cycle_position=long_position,
            productivity_trend_position=productivity_position,
            debt_sustainability_score=debt_score,
            bubble_risk_score=bubble_score,
            deleveraging_risk_score=deleveraging_score,
            policy_effectiveness=policy_score,
            key_signals=key_signals,
            historical_analogues=[a["name"] for a in analogues],
            confidence=confidence,
        )

    def simulate_cycle_forward(
        self,
        state: DebtCycleState,
        years: float = 10.0,
        steps: int = 120,
        perturbation: float = 0.01,
    ) -> list[DebtCycleState]:
        dt = years / steps
        states = [state]
        current = DebtCycleState(
            gdp_growth_rate=state.gdp_growth_rate,
            debt_to_gdp=state.debt_to_gdp,
            interest_rate=state.interest_rate,
            money_supply_growth=state.money_supply_growth,
            credit_growth=state.credit_growth,
            productivity_growth=state.productivity_growth,
            inflation_rate=state.inflation_rate,
            unemployment_rate=state.unemployment_rate,
            stock_valuation=state.stock_valuation,
            capacity_utilization=state.capacity_utilization,
            timestamp=state.timestamp,
        )

        rng = np.random.default_rng(42)

        for i in range(steps):
            diagnosis = self.diagnose_cycle(current)

            gdp_accel = self._gdp_acceleration(current, diagnosis, dt)
            debt_accel = self._debt_acceleration(current, diagnosis, dt)
            rate_accel = self._rate_acceleration(current, diagnosis, dt)
            inflation_accel = self._inflation_acceleration(current, diagnosis, dt)
            unemploy_accel = self._unemployment_acceleration(current, diagnosis, dt)
            stock_accel = self._stock_acceleration(current, diagnosis, dt)

            current.gdp_growth_rate += gdp_accel * dt + rng.normal(0, perturbation * dt)
            current.debt_to_gdp += debt_accel * dt + rng.normal(0, perturbation * dt)
            current.interest_rate += rate_accel * dt + rng.normal(0, perturbation * 0.5 * dt)
            current.inflation_rate += inflation_accel * dt + rng.normal(0, perturbation * 0.5 * dt)
            current.unemployment_rate += unemploy_accel * dt + rng.normal(0, perturbation * dt)
            current.stock_valuation += stock_accel * dt + rng.normal(0, perturbation * 2 * dt)
            current.capacity_utilization += (state.capacity_utilization - current.capacity_utilization) * 0.01 * dt + rng.normal(0, perturbation * 0.5 * dt)
            current.credit_growth += (state.credit_growth * 0.5 - current.credit_growth) * 0.05 * dt
            current.money_supply_growth += rng.normal(0, perturbation * 0.3 * dt)

            current.gdp_growth_rate = np.clip(current.gdp_growth_rate, -0.10, 0.10)
            current.debt_to_gdp = max(0.0, current.debt_to_gdp)
            current.interest_rate = np.clip(current.interest_rate, -0.02, 0.25)
            current.inflation_rate = np.clip(current.inflation_rate, -0.05, 0.20)
            current.unemployment_rate = np.clip(current.unemployment_rate, 0.02, 0.30)
            current.stock_valuation = max(0.1, current.stock_valuation)

            new_state = DebtCycleState(
                gdp_growth_rate=current.gdp_growth_rate,
                debt_to_gdp=current.debt_to_gdp,
                interest_rate=current.interest_rate,
                money_supply_growth=current.money_supply_growth,
                credit_growth=current.credit_growth,
                productivity_growth=current.productivity_growth,
                inflation_rate=current.inflation_rate,
                unemployment_rate=current.unemployment_rate,
                stock_valuation=current.stock_valuation,
                capacity_utilization=current.capacity_utilization,
                timestamp=state.timestamp + timedelta(days=int(365.25 * dt * (i + 1))),
            )
            states.append(new_state)
            current = new_state

        return states

    def predict_from_cycle(
        self,
        state: DebtCycleState,
        target_date: date,
        confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH,
    ) -> list[Prediction]:
        years_ahead = max(0.5, (target_date - state.timestamp).days / 365.25)
        states = self.simulate_cycle_forward(state, years=years_ahead, steps=max(12, int(years_ahead * 12)))

        predictions = []
        variables = [
            ("gdp_growth_rate", Domain.ECONOMICS),
            ("debt_to_gdp", Domain.ECONOMICS),
            ("interest_rate", Domain.ECONOMICS),
            ("inflation_rate", Domain.ECONOMICS),
            ("unemployment_rate", Domain.ECONOMICS),
            ("stock_valuation", Domain.FINANCE),
        ]

        for var_name, domain in variables:
            values = [getattr(s, var_name) for s in states[-13:] if hasattr(s, var_name)]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values) if len(values) > 1 else abs(mean_val) * 0.1
                z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645, ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence_level]
                ci = (mean_val - z * std_val, mean_val + z * std_val)

                predictions.append(Prediction(
                    domain=domain,
                    variable=var_name,
                    value=float(mean_val),
                    confidence_interval=ci,
                    confidence_level=confidence_level,
                    timestamp=datetime.now(),
                    target_date=target_date,
                    model_name="dalio_cycle",
                    weights={"deterministic": 1.0},
                    signals_used=["debt_cycle", "productivity", "monetary_policy"],
                    correlations_used=["debt_gdp", "inflation_unemployment"],
                ))

        return predictions

    def _compute_short_cycle_position(self, state: DebtCycleState) -> float:
        if state.interest_rate < 0.02:
            return 0.8
        elif state.interest_rate > 0.06:
            return 0.9
        spread = state.credit_growth - state.gdp_growth_rate
        if spread > 0.03:
            return 0.3
        elif spread < -0.01:
            return 0.7
        return 0.5

    def _compute_long_cycle_position(self, state: DebtCycleState) -> float:
        if state.debt_to_gdp < 1.0:
            return 0.2
        elif state.debt_to_gdp < 2.0:
            return 0.5
        elif state.debt_to_gdp < 3.0:
            return 0.7
        elif state.debt_to_gdp < 4.0:
            return 0.85
        else:
            return 0.95

    def _compute_productivity_position(self, state: DebtCycleState) -> float:
        return 1.0 - min(state.productivity_growth / 0.04, 1.0)

    def _determine_phase(self, state: DebtCycleState, short_pos: float, long_pos: float) -> CyclePhase:
        if long_pos > 0.85:
            if short_pos > 0.7:
                return CyclePhase.LATE_EXPANSION
            return CyclePhase.LATE_RECESSION
        if state.unemployment_rate < 0.04 and state.capacity_utilization > 0.85:
            if state.inflation_rate > 0.03:
                return CyclePhase.LATE_EXPANSION
            return CyclePhase.MID_EXPANSION
        if state.unemployment_rate > 0.07:
            if state.gdp_growth_rate < 0:
                return CyclePhase.MID_RECESSION
            return CyclePhase.EARLY_RECESSION
        if state.gdp_growth_rate > 0.03:
            if state.unemployment_rate > 0.05:
                return CyclePhase.EARLY_EXPANSION
            return CyclePhase.MID_EXPANSION
        if state.inflation_rate > 0.05 and state.interest_rate < 0.04:
            return CyclePhase.PEAK
        if state.gdp_growth_rate < 0.01 and state.unemployment_rate < 0.06:
            return CyclePhase.TROUGH
        return CyclePhase.MID_EXPANSION

    def _gdp_acceleration(self, state: DebtCycleState, diag: CycleDiagnosis, dt: float) -> float:
        base = state.productivity_growth
        cycle_effect = -0.01 * (state.stock_valuation - 1.0)
        debt_drag = -0.005 * max(0, state.debt_to_gdp - 2.5)
        rate_drag = -0.02 * max(0, state.interest_rate - 0.04)
        return base + cycle_effect + debt_drag + rate_drag

    def _debt_acceleration(self, state: DebtCycleState, diag: CycleDiagnosis, dt: float) -> float:
        if diag.phase in (CyclePhase.EARLY_EXPANSION, CyclePhase.MID_EXPANSION):
            credit_accel = state.credit_growth * 0.1
        else:
            credit_accel = -state.credit_growth * 0.05
        gdp_effect = -state.gdp_growth_rate * state.debt_to_gdp * 0.3
        return credit_accel + gdp_effect

    def _rate_acceleration(self, state: DebtCycleState, diag: CycleDiagnosis, dt: float) -> float:
        inflation_pressure = (state.inflation_rate - 0.02) * 0.5
        unemploy_relief = -(state.unemployment_rate - 0.05) * 0.2
        return inflation_pressure + unemploy_relief

    def _inflation_acceleration(self, state: DebtCycleState, diag: CycleDiagnosis, dt: float) -> float:
        wage_pressure = (state.capacity_utilization - 0.80) * 0.05
        money_pressure = state.money_supply_growth * 0.3
        rate_relief = -(state.interest_rate - 0.03) * 0.1
        productivity_relief = -state.productivity_growth * 0.5
        return wage_pressure + money_pressure + rate_relief + productivity_relief

    def _unemployment_acceleration(self, state: DebtCycleState, diag: CycleDiagnosis, dt: float) -> float:
        gdp_effect = -state.gdp_growth_rate * 0.5
        cycle_effect = -0.005 if diag.phase in (CyclePhase.EARLY_EXPANSION, CyclePhase.MID_EXPANSION) else 0.005
        return gdp_effect + cycle_effect

    def _stock_acceleration(self, state: DebtCycleState, diag: CycleDiagnosis, dt: float) -> float:
        gdp_effect = state.gdp_growth_rate * 2.0
        rate_effect = -state.interest_rate * 0.5
        sentiment = 0.1 if diag.phase in (CyclePhase.EARLY_EXPANSION, CyclePhase.MID_EXPANSION) else -0.05
        reversion = -(state.stock_valuation - 1.0) * 0.02
        return gdp_effect + rate_effect + sentiment + reversion

    def _debt_sustainability(self, state: DebtCycleState) -> float:
        interest_burden = state.debt_to_gdp * state.interest_rate
        gdp_covers = state.gdp_growth_rate / max(interest_burden, 0.001)
        savings_rate = 0.18
        primary_balance = savings_rate - interest_burden
        score = 1.0
        if state.debt_to_gdp > 2.5:
            score -= 0.3
        if state.debt_to_gdp > 3.5:
            score -= 0.3
        if primary_balance < 0:
            score -= 0.2
        if state.interest_rate > state.gdp_growth_rate:
            score -= 0.2
        return max(0.0, min(1.0, score))

    def _bubble_risk(self, state: DebtCycleState) -> float:
        score = 0.0
        score += min(state.stock_valuation / 5.0, 0.25)
        if state.credit_growth > 0.06:
            score += 0.2
        if state.capacity_utilization > 0.90:
            score += 0.15
        if state.interest_rate < 0.02:
            score += 0.2
        if state.inflation_rate < 0.01:
            score += 0.1
        spread = state.credit_growth - state.gdp_growth_rate
        if spread > 0.03:
            score += 0.1
        return min(1.0, max(0.0, score))

    def _deleveraging_risk(self, state: DebtCycleState) -> float:
        score = 0.0
        if state.debt_to_gdp > 3.0:
            score += 0.3
        if state.interest_rate > state.gdp_growth_rate:
            score += 0.25
        if state.credit_growth < 0:
            score += 0.25
        if state.unemployment_rate > 0.08:
            score += 0.2
        return min(1.0, max(0.0, score))

    def _policy_effectiveness(self, state: DebtCycleState) -> float:
        score = 0.5
        if state.interest_rate > 0.01:
            rate_room = min((state.interest_rate - 0.01) / 0.04, 0.25)
            score += rate_room
        if state.debt_to_gdp < 2.5:
            score += 0.15
        if state.money_supply_growth > 0.02 and state.inflation_rate < 0.04:
            score += 0.1
        return min(1.0, max(0.0, score))

    def _find_historical_analogues(self, state: DebtCycleState, phase: CyclePhase) -> list[dict]:
        scored_events = []
        for event in self.historical_events:
            similarity = 0.0
            if event.get("debt_peak") and abs(state.debt_to_gdp - event["debt_peak"]) < 1.0:
                similarity += 0.3 * (1.0 - abs(state.debt_to_gdp - event["debt_peak"]))
            if event.get("gdp_decline") and state.gdp_growth_rate < 0.02:
                similarity += 0.2
            if event.get("inflation_peak") and abs(state.inflation_rate - event.get("inflation_peak", 0.05)) < 0.03:
                similarity += 0.2
            if event.get("unemployment_peak") and abs(state.unemployment_rate - event.get("unemployment_peak", 0.10)) < 0.04:
                similarity += 0.15
            if "stagflation" in event.get("type", "") and state.inflation_rate > 0.04 and state.gdp_growth_rate < 0.02:
                similarity += 0.3
            if "bubble" in event.get("type", "") and state.stock_valuation > 2.0:
                similarity += 0.3
            if "deleveraging" in event.get("type", "") and state.debt_to_gdp > 2.5:
                similarity += 0.3
            scored_events.append((event, similarity))
        scored_events.sort(key=lambda x: x[1], reverse=True)
        return [e for e, s in scored_events[:5] if s > 0.1]

    def _compute_confidence(self, state: DebtCycleState, signals: dict) -> float:
        data_quality = 0.5
        if signals.get("debt_to_gdp", 0) > 0:
            data_quality += 0.1
        if abs(signals.get("funding_spread", 0)) > 0:
            data_quality += 0.1
        if signals.get("capacity_utilization", 0) > 0:
            data_quality += 0.1
        consistency = 0.3
        if state.gdp_growth_rate > 0 and state.unemployment_rate < 0.10:
            consistency += 0.2
        if state.debt_to_gdp > 1.0:
            consistency += 0.1
        return min(1.0, max(0.1, data_quality + consistency))