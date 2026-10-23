import pytest
import numpy as np
from datetime import date, timedelta
from oracle.core.config import OracleConfig
from oracle.core.types import (
    Domain, DataSource, TimeSeries, DataPoint, TimeInterval,
    Prediction, Forecast, ConfidenceLevel, EconomicCycle,
    Signal, SignalType, Correlation, CorrelationMatrix,
)


class TestOracleConfig:
    def test_default_config(self):
        config = OracleConfig()
        assert config.monte_carlo_simulations == 10000
        assert config.buffett_margin_of_safety == 0.25
        assert config.correlation_threshold == 0.3

    def test_custom_config(self):
        config = OracleConfig(monte_carlo_simulations=5000, buffett_margin_of_safety=0.3)
        assert config.monte_carlo_simulations == 5000
        assert config.buffett_margin_of_safety == 0.3


class TestDomain:
    def test_all_domains_exist(self):
        domains = [Domain.ECONOMICS, Domain.FINANCE, Domain.GEOPOLITICS,
                   Domain.TECHNOLOGY, Domain.CLIMATE, Domain.DEMOGRAPHICS,
                   Domain.HEALTH, Domain.ENERGY, Domain.COMMODITIES, Domain.SOCIAL]
        assert len(domains) == 10


class TestCorrelation:
    def test_correlation_strength(self):
        c = Correlation(source_a="gdp", source_b="stock", domain_a=Domain.ECONOMICS,
                        domain_b=Domain.FINANCE, correlation=0.85)
        assert c.strength == "strong"
        assert c.direction == "positive"

    def test_negative_correlation(self):
        c = Correlation(source_a="gdp", source_b="unemployment", domain_a=Domain.ECONOMICS,
                        domain_b=Domain.ECONOMICS, correlation=-0.6)
        assert c.direction == "negative"
        assert c.strength == "moderate"

    def test_weak_correlation(self):
        c = Correlation(source_a="a", source_b="b", domain_a=Domain.ALL,
                        domain_b=Domain.ALL, correlation=0.2)
        assert c.strength == "weak"


class TestSignal:
    def test_signal_creation(self):
        s = Signal(name="gdp_growth", domain=Domain.ECONOMICS, signal_type=SignalType.LEADING,
                   value=0.025, timestamp=date.today(), source=DataSource.FRED)
        assert s.name == "gdp_growth"
        assert s.decay_weight == 1.0

    def test_signal_decay(self):
        s = Signal(name="rate", domain=Domain.ECONOMICS, signal_type=SignalType.COINCIDENT,
                   value=0.05, timestamp=date.today() - timedelta(days=100), source=DataSource.FRED)
        decayed = s.decay(date.today(), rate=0.05)
        assert 0 < decayed < 1


class TestCorrelationMatrix:
    def test_strong_correlations(self):
        names = ["a", "b", "c"]
        matrix = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, -0.6], [0.3, -0.6, 1.0]])
        p_values = np.ones_like(matrix) * 0.01
        cm = CorrelationMatrix(variables=names, matrix=matrix, p_values=p_values)
        strong = cm.get_strong_correlations(threshold=0.5)
        assert len(strong) == 2


class TestPrediction:
    def test_prediction_creation(self):
        p = Prediction(
            domain=Domain.ECONOMICS, variable="gdp_growth", value=0.025,
            confidence_interval=(0.01, 0.04), confidence_level=ConfidenceLevel.HIGH,
            timestamp=date.today(), target_date=date(2030, 1, 1), model_name="test",
        )
        assert p.ci_width == pytest.approx(0.03)
        assert p.ci_midpoint == pytest.approx(0.025)


class TestForecast:
    def test_forecast_filtering(self):
        p1 = Prediction(domain=Domain.ECONOMICS, variable="gdp", value=0.02,
                         confidence_interval=(0.01, 0.03), confidence_level=ConfidenceLevel.HIGH,
                         timestamp=date.today(), target_date=date(2030, 1, 1), model_name="test")
        p2 = Prediction(domain=Domain.FINANCE, variable="stock", value=100,
                         confidence_interval=(80, 120), confidence_level=ConfidenceLevel.HIGH,
                         timestamp=date.today(), target_date=date(2030, 1, 1), model_name="test")
        f = Forecast(predictions=[p1, p2], time_range=(date.today(), date(2030, 1, 1)))
        econ = f.get_predictions_for_domain(Domain.ECONOMICS)
        assert len(econ) == 1
        assert econ[0].variable == "gdp"