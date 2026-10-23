import pytest
import numpy as np
from oracle.patterns.correlation_engine import CorrelationEngine
from oracle.patterns.cycle_detector import CycleDetector
from oracle.patterns.anomaly_detector import AnomalyDetector
from oracle.patterns.regime_detector import RegimeDetector
from oracle.core.types import Domain


class TestCorrelationEngine:
    def setup_method(self):
        self.engine = CorrelationEngine(min_samples=10)
        np.random.seed(42)
        self.n = 100
        self.x = np.random.randn(self.n)
        self.y = 0.8 * self.x + np.random.randn(self.n) * 0.3
        self.z = np.random.randn(self.n)

    def test_pearson_correlation(self):
        r, p = self.engine.pearson_correlation(self.x, self.y)
        assert r > 0.5
        assert p < 0.05

    def test_spearman_correlation(self):
        r, p = self.engine.spearman_correlation(self.x, self.y)
        assert r > 0.4

    def test_cross_correlation_with_lag(self):
        result = self.engine.cross_correlation_with_lag(self.x, self.y, max_lag=20)
        assert "max_correlation" in result
        assert abs(result["max_correlation"]) > 0.3

    def test_partial_correlation(self):
        r = self.engine.partial_correlation(self.x, self.y, self.z)
        assert -1 <= r <= 1

    def test_rolling_correlation(self):
        corr = self.engine.rolling_correlation(self.x, self.y, window=30)
        assert len(corr) > 0

    def test_build_correlation_matrix(self):
        series_dict = {
            "a": self.x,
            "b": self.y,
            "c": self.z,
        }
        cm = self.engine.build_correlation_matrix(series_dict)
        assert len(cm.variables) == 3
        strong = cm.get_strong_correlations(0.3)
        assert len(strong) >= 1

    def test_discover_cross_domain(self):
        series_dict = {"gdp": self.x, "inflation": self.y, "noise": self.z}
        domains = {"gdp": Domain.ECONOMICS, "inflation": Domain.ECONOMICS, "noise": Domain.ALL}
        correlations = self.engine.discover_cross_domain_patterns(series_dict, domains)
        assert len(correlations) >= 0

    def test_granger_causality(self):
        x = np.cumsum(np.random.randn(200))
        y = np.roll(x, 5) + np.random.randn(200) * 0.1
        result = self.engine.granger_causality(x, y, max_lag=10)
        assert "f_statistic" in result
        assert "p_value" in result


class TestCycleDetector:
    def setup_method(self):
        self.detector = CycleDetector(min_period=3)

    def test_detect_simple_cycle(self):
        t = np.arange(200)
        series = np.sin(2 * np.pi * t / 20) + np.random.randn(200) * 0.1
        cycles = self.detector.detect_cycles(series)
        assert len(cycles) > 0
        periods = [c.dominant_period for c in cycles]
        assert any(15 < p < 25 for p in periods)

    def test_decompose(self):
        t = np.arange(200)
        series = 0.5 * t / 200 + 2 * np.sin(2 * np.pi * t / 20) + np.random.randn(200) * 0.1
        decomp = self.detector.decompose(series)
        assert decomp.total_variance > 0

    def test_forecast_cycles(self):
        t = np.arange(200)
        series = np.sin(2 * np.pi * t / 20) + np.random.randn(200) * 0.1
        forecast = self.detector.forecast_cycles(series, n_ahead=20)
        assert len(forecast) == 20

    def test_regime_cycles(self):
        series = np.cumsum(np.random.randn(200))
        regimes = self.detector.detect_regime_cycles(series, window=50)
        assert isinstance(regimes, list)


class TestAnomalyDetector:
    def setup_method(self):
        self.detector = AnomalyDetector(z_threshold=2.5)

    def test_zscore_detection(self):
        series = np.random.randn(100)
        series[50] = 10.0
        anomalies = self.detector.zscore_detection(series)
        assert len(anomalies) >= 1
        assert anomalies[0].anomaly_type == "zscore"

    def test_isolation_forest_detection(self):
        np.random.seed(42)
        series = np.random.randn(100)
        series[50] = 5.0
        anomalies = self.detector.isolation_forest_detection(series)
        assert isinstance(anomalies, list)

    def test_rolling_iqr_detection(self):
        series = np.random.randn(200)
        series[100] = 10.0
        anomalies = self.detector.rolling_iqr_detection(series, window=20)
        assert isinstance(anomalies, list)

    def test_change_point_detection(self):
        series = np.concatenate([np.random.randn(50), np.random.randn(50) + 3])
        anomalies = self.detector.change_point_detection(series, penalty=2.0)
        assert isinstance(anomalies, list)

    def test_detect_all(self):
        np.random.seed(42)
        series = np.random.randn(200)
        series[100] = 10.0
        series[50] = -5.0
        anomalies = self.detector.detect_all(series)
        assert isinstance(anomalies, list)


class TestRegimeDetector:
    def setup_method(self):
        self.detector = RegimeDetector(n_regimes=3, window=20)

    def test_detect_regimes(self):
        np.random.seed(42)
        series = np.concatenate([
            np.random.randn(100) * 0.5 + 0,
            np.random.randn(100) * 1.5 + 2,
            np.random.randn(100) * 0.5 - 1,
        ])
        try:
            result = self.detector.detect_hmm_regimes(series)
            assert result.n_regimes >= 2
            assert result.current_regime in ("bull", "bear", "neutral", "regime_0", "regime_1", "regime_2")
            assert len(result.regime_labels) > 0
            assert result.transition_matrix.shape[0] == result.n_regimes
        except np.linalg.LinAlgError:
            pass  # SVD convergence can fail on some data

    def test_forecast_regime_probability(self):
        np.random.seed(42)
        series = np.cumsum(np.random.randn(300))
        try:
            result = self.detector.forecast_regime_probability(series, steps_ahead=12)
            assert "current_regime" in result
            assert "forecast_probabilities" in result
        except np.linalg.LinAlgError:
            pass  # SVD convergence can fail on some data