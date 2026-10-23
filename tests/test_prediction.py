import pytest
import numpy as np
from datetime import date, timedelta
from oracle.prediction.engine import OracleEngine, OracleState
from oracle.prediction.timeslider import TimeSlider, TimeSlice
from oracle.core.config import OracleConfig
from oracle.core.types import Domain, ConfidenceLevel


class TestOracleEngine:
    def setup_method(self):
        self.engine = OracleEngine()

    def test_set_economic_state(self):
        self.engine.set_economic_state(
            gdp_growth_rate=0.025, debt_to_gdp=1.2, interest_rate=0.045,
        )
        assert self.engine.state.economic_state is not None
        assert abs(self.engine.state.economic_state.gdp_growth_rate - 0.025) < 1e-10

    def test_set_company(self):
        self.engine.set_company(ticker="AAPL", eps=6.5, price=185.0)
        assert self.engine.state.company_fundamentals is not None
        assert self.engine.state.company_fundamentals.ticker == "AAPL"

    def test_add_time_series(self):
        data = np.cumsum(np.random.randn(100))
        self.engine.add_time_series("test", data, Domain.ECONOMICS)
        assert "test" in self.engine.state.raw_series

    def test_predict_economic(self):
        self.engine.set_economic_state()
        forecast = self.engine.predict(date(2030, 1, 1), ConfidenceLevel.MEDIUM)
        assert len(forecast.predictions) > 0
        assert any(p.model_name == "oracle_ensemble" for p in forecast.predictions)

    def test_predict_with_company(self):
        self.engine.set_economic_state()
        self.engine.set_company(ticker="TEST", price=100.0)
        forecast = self.engine.predict(date(2030, 1, 1))
        assert any(p.domain == Domain.FINANCE for p in forecast.predictions)

    def test_predict_with_time_series(self):
        np.random.seed(42)
        self.engine.set_economic_state()
        self.engine.add_time_series("gdp", np.cumsum(np.random.randn(120)), Domain.ECONOMICS)
        forecast = self.engine.predict(date(2030, 1, 1))
        assert len(forecast.predictions) > 0

    def test_diagnose(self):
        self.engine.set_economic_state()
        diagnosis = self.engine.diagnose()
        assert "cycle_diagnosis" in diagnosis
        assert diagnosis["cycle_diagnosis"] is not None
        assert "phase" in diagnosis["cycle_diagnosis"]

    def test_diagnose_with_series(self):
        np.random.seed(42)
        self.engine.set_economic_state()
        self.engine.add_time_series("test_series", np.random.randn(200), Domain.ECONOMICS)
        diagnosis = self.engine.diagnose()
        assert isinstance(diagnosis, dict)


class TestTimeSlider:
    def setup_method(self):
        self.engine = OracleEngine()
        self.engine.set_economic_state()
        self.slider = TimeSlider(self.engine)

    def test_slide_to(self):
        ts = self.slider.slide_to(date(2030, 1, 1))
        assert isinstance(ts, TimeSlice)
        assert ts.target_date == date(2030, 1, 1)
        assert ts.years_ahead > 0

    def test_slide_range(self):
        results = self.slider.slide_range(date(2026, 1, 1), date(2030, 1, 1), step_days=365)
        assert len(results) >= 2

    def test_quick_scan(self):
        results = self.slider.quick_scan([30, 365, 1825])
        assert len(results) == 3

    def test_scenario_analysis(self):
        scenarios = {
            "bull": {"gdp_growth_rate": 0.04, "inflation_rate": 0.02},
            "bear": {"gdp_growth_rate": -0.01, "inflation_rate": 0.06},
        }
        results = self.slider.scenario_analysis(scenarios, date(2030, 1, 1))
        assert "bull" in results
        assert "bear" in results

    def test_get_timeline(self):
        self.slider.slide_to(date(2027, 1, 1))
        self.slider.slide_to(date(2030, 1, 1))
        timeline = self.slider.get_timeline()
        assert len(timeline) >= 2

    def test_summary(self):
        ts = self.slider.slide_to(date(2030, 1, 1))
        assert "target_date" in ts.summary
        assert "years_ahead" in ts.summary