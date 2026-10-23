import pytest
import numpy as np
from datetime import date, timedelta
from oracle.deterministic.dalio_cycles import DalioCycleEngine, DebtCycleState, CyclePhase
from oracle.deterministic.economic_model import EconomicDeterministicModel, MacroState
from oracle.deterministic.physics_model import PhysicsModel, PhysicsParams
from oracle.deterministic.statistical_model import StatisticalDeterministicModel
from oracle.core.types import Domain, ConfidenceLevel


class TestDalioCycleEngine:
    def setup_method(self):
        self.engine = DalioCycleEngine()

    def test_diagnose_expansion(self):
        state = DebtCycleState(
            gdp_growth_rate=0.035, debt_to_gdp=0.8, interest_rate=0.02,
            money_supply_growth=0.06, credit_growth=0.08, productivity_growth=0.02,
            inflation_rate=0.02, unemployment_rate=0.035, stock_valuation=1.2,
            capacity_utilization=0.82,
        )
        diag = self.engine.diagnose_cycle(state)
        assert diag.confidence > 0
        assert 0 <= diag.debt_sustainability_score <= 1
        assert 0 <= diag.bubble_risk_score <= 1
        assert 0 <= diag.deleveraging_risk_score <= 1

    def test_diagnose_recession(self):
        state = DebtCycleState(
            gdp_growth_rate=-0.02, debt_to_gdp=2.5, interest_rate=0.06,
            money_supply_growth=0.02, credit_growth=-0.02, productivity_growth=0.01,
            inflation_rate=0.05, unemployment_rate=0.09, stock_valuation=0.7,
            capacity_utilization=0.68,
        )
        diag = self.engine.diagnose_cycle(state)
        assert diag.deleveraging_risk_score > 0.3
        assert diag.bubble_risk_score < 0.5

    def test_simulate_forward(self):
        state = DebtCycleState()
        states = self.engine.simulate_cycle_forward(state, years=10, steps=120)
        assert len(states) == 121
        for s in states:
            assert -0.10 <= s.gdp_growth_rate <= 0.10
            assert s.debt_to_gdp >= 0
            assert -0.02 <= s.interest_rate <= 0.25

    def test_predict_from_cycle(self):
        state = DebtCycleState()
        preds = self.engine.predict_from_cycle(state, date(2030, 1, 1), ConfidenceLevel.HIGH)
        assert len(preds) >= 3
        for p in preds:
            assert p.confidence_interval[0] <= p.value <= p.confidence_interval[1]

    def test_high_debt_scenario(self):
        state = DebtCycleState(debt_to_gdp=4.0, interest_rate=0.08, gdp_growth_rate=0.005)
        diag = self.engine.diagnose_cycle(state)
        assert diag.deleveraging_risk_score > 0.3
        assert diag.debt_sustainability_score < 0.7

    def test_bubble_scenario(self):
        state = DebtCycleState(
            stock_valuation=3.0, credit_growth=0.10, interest_rate=0.01,
            inflation_rate=0.005, capacity_utilization=0.93,
        )
        diag = self.engine.diagnose_cycle(state)
        assert diag.bubble_risk_score > 0.3


class TestEconomicDeterministicModel:
    def setup_method(self):
        self.model = EconomicDeterministicModel()

    def test_solow_growth(self):
        output = self.model.solow_growth(capital=100, labor=50, tfp=1.0, alpha=0.3)
        assert output > 0

    def test_taylor_rule(self):
        rate = self.model.taylor_rule(inflation=0.04, output_gap=0.01)
        assert rate > 0.02

    def test_phillips_curve(self):
        inflation = self.model.phillips_curve(unemployment=0.03)
        assert inflation > 0

    def test_fisher_equation(self):
        real = self.model.fisher_equation(nominal_rate=0.05, expected_inflation=0.02)
        assert real < 0.05

    def test_debt_dynamics(self):
        change = self.model.debt_dynamics(debt_gdp=1.0, primary_balance=0.02, real_growth=0.02, real_rate=0.01)
        assert isinstance(change, float)

    def test_simulate(self):
        state = MacroState()
        states = self.model.simulate(state, years=5, steps=60)
        assert len(states) == 61
        for s in states:
            assert s.interest_rate >= -0.02
            assert s.unemployment >= 0.02

    def test_predict(self):
        state = MacroState()
        preds = self.model.predict(state, date(2030, 1, 1), ConfidenceLevel.MEDIUM)
        assert len(preds) >= 3
        for p in preds:
            assert p.confidence_interval[0] <= p.value


class TestPhysicsModel:
    def setup_method(self):
        self.model = PhysicsModel()

    def test_coupled_oscillator(self):
        states = [(1.0, 0.0), (0.5, 0.1)]
        trajectory = self.model.coupled_oscillator(states, n_steps=100)
        assert trajectory.shape == (100, 2)

    def test_langevin_dynamics(self):
        trajectory = self.model.langevin_dynamics(
            potential=lambda x: x**2, initial_position=0.0, n_steps=1000,
        )
        assert len(trajectory) == 1000

    def test_renormalization_group(self):
        data = np.random.randn(256)
        levels = self.model.renormalization_group(data, n_levels=3)
        assert len(levels) >= 3

    def test_resonance_detection(self):
        t = np.linspace(0, 10, 500)
        a = np.sin(2 * np.pi * t)
        b = np.sin(2 * np.pi * t + 0.5)
        result = self.model.resonance_detection(a, b)
        assert "coherence" in result
        assert result["max_coherence"] >= 0

    def test_entropy_flow(self):
        series = np.random.randn(500)
        result = self.model.entropy_flow(series)
        assert result["mean_entropy"] > 0

    def test_critical_point_detection(self):
        series = np.concatenate([np.random.randn(100), np.random.randn(100) + 3])
        result = self.model.critical_point_detection(series)
        assert "critical_points" in result

    def test_hurst_computation(self):
        series = np.cumsum(np.random.randn(500))
        result = self.model.multi_scale_prediction(series, n_ahead=50, n_sims=100)
        assert 0 < result["hurst_exponent"] <= 1

    def test_predict(self):
        series = np.cumsum(np.random.randn(200))
        pred = self.model.predict(series, "test_var", Domain.ECONOMICS, date(2030, 1, 1))
        assert pred.variable == "test_var"
        assert pred.confidence_interval[0] <= pred.value <= pred.confidence_interval[1]


class TestStatisticalDeterministicModel:
    def setup_method(self):
        self.model = StatisticalDeterministicModel()
        np.random.seed(42)
        self.trend_data = np.cumsum(np.random.normal(0.01, 0.05, 200))
        self.noisy_data = np.random.normal(0, 1, 200)

    def test_linear_regression(self):
        result = self.model._linear_regression(self.trend_data, n_ahead=10)
        assert len(result["predictions"]) == 10
        assert result["r_squared"] >= 0

    def test_polynomial_fit(self):
        result = self.model._polynomial_fit(self.trend_data, n_ahead=10)
        assert len(result["predictions"]) == 10

    def test_exponential_smoothing(self):
        result = self.model._exponential_smoothing(self.trend_data, n_ahead=10)
        assert len(result["predictions"]) == 10

    def test_arima_approx(self):
        result = self.model._arima_approx(self.trend_data, n_ahead=10)
        assert len(result["predictions"]) == 10

    def test_kalman_filter(self):
        result = self.model._kalman_filter(self.trend_data, n_ahead=10)
        assert len(result["predictions"]) == 10

    def test_ensemble_predict(self):
        result = self.model.ensemble_predict(self.trend_data, n_ahead=20)
        assert "predictions" in result
        assert "ci_upper" in result
        assert "ci_lower" in result

    def test_predict(self):
        pred = self.model.predict(self.trend_data, "test", Domain.ECONOMICS, date(2030, 1, 1))
        assert pred.model_name == "statistical_deterministic"
        assert pred.confidence_interval[0] <= pred.value