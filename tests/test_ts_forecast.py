"""Tests for TS Forecast Ensemble — ARIMA, ETS, Theta, VAR, ensemble."""
import math
import numpy as np
import pytest
from oracle.prediction.ts_forecast import (
    TSForecastEnsemble, ARIMAModel, ETSModel, ThetaModel, VARModel,
    ForecastMethod, SeasonalityType, ForecastPoint, MethodResult,
    BacktestResult, EnsembleForecastResult
)


rng = np.random.RandomState(42)
TREND_DATA = np.cumsum(rng.randn(100)) + 50
SEASONAL_DATA = np.array([20 + 10 * math.sin(2 * math.pi * i / 12) + rng.randn() * 0.5 for i in range(120)])
RANDOM_DATA = rng.randn(80) + 100
SHORT_DATA = np.array([1, 2, 3, 4, 5])


class TestARIMA:
    def test_arima_fit_forecast(self):
        model = ARIMAModel(seed=42)
        model.fit(TREND_DATA)
        assert model._fitted
        fc = model.forecast(12)
        assert len(fc[0]) == 12
        assert all(isinstance(v, float) for v in fc[0])

    def test_arima_confidence_intervals(self):
        model = ARIMAModel(seed=42)
        model.fit(TREND_DATA)
        fc = model.forecast(12)
        preds, l80, u80, l95, u95 = fc
        for i in range(12):
            assert l80[i] <= preds[i] <= u80[i], f"80% CI violated at step {i}"
            assert l95[i] <= preds[i] <= u95[i], f"95% CI violated at step {i}"
            assert l95[i] <= l80[i], f"95% lower > 80% lower at step {i}"
            assert u95[i] >= u80[i], f"95% upper < 80% upper at step {i}"

    def test_arima_aic_bic(self):
        model = ARIMAModel(seed=42)
        model.fit(TREND_DATA)
        assert model.aic < float('inf')
        assert model.bic < float('inf')
        assert model.log_likelihood != 0

    def test_arima_short_series(self):
        model = ARIMAModel(seed=42)
        model.fit(SHORT_DATA)
        fc = model.forecast(3)
        assert len(fc[0]) == 3

    def test_arima_constant_series(self):
        model = ARIMAModel(seed=42)
        model.fit(np.ones(50))
        fc = model.forecast(5)
        assert len(fc[0]) == 5
        assert all(abs(v - 1.0) < 0.5 for v in fc[0])

    def test_arima_not_fitted_raises(self):
        model = ARIMAModel()
        with pytest.raises(ValueError):
            model.forecast(5)


class TestETS:
    def test_ets_fit_forecast(self):
        model = ETSModel(seed=42)
        model.fit(TREND_DATA)
        assert model._fitted
        fc = model.forecast(12)
        assert len(fc[0]) == 12

    def test_ets_seasonal_data(self):
        model = ETSModel(seasonal_periods=12, seed=42)
        model.fit(SEASONAL_DATA)
        fc = model.forecast(12)
        assert len(fc[0]) == 12

    def test_ets_confidence_intervals(self):
        model = ETSModel(seed=42)
        model.fit(TREND_DATA)
        preds, l80, u80, l95, u95 = model.forecast(12)
        for i in range(12):
            assert l95[i] <= l80[i]
            assert u95[i] >= u80[i]

    def test_ets_aic_bic(self):
        model = ETSModel(seed=42)
        model.fit(TREND_DATA)
        assert model.aic < float('inf')
        assert model.bic < float('inf')

    def test_ets_short_series(self):
        model = ETSModel(seed=42)
        model.fit(np.array([1, 2, 3]))
        fc = model.forecast(3)
        assert len(fc[0]) == 3

    def test_ets_not_fitted_raises(self):
        model = ETSModel()
        with pytest.raises(ValueError):
            model.forecast(5)

    def test_ets_detects_seasonality(self):
        model = ETSModel(seed=42)
        model.fit(SEASONAL_DATA)
        assert model._fitted

    def test_ets_no_trend(self):
        model = ETSModel(seed=42)
        model.fit(RANDOM_DATA)
        fc = model.forecast(5)
        assert len(fc[0]) == 5


class TestTheta:
    def test_theta_fit_forecast(self):
        model = ThetaModel(seed=42)
        model.fit(TREND_DATA)
        assert model._fitted
        fc = model.forecast(12)
        assert len(fc[0]) == 12

    def test_theta_confidence_intervals(self):
        model = ThetaModel(seed=42)
        model.fit(TREND_DATA)
        preds, l80, u80, l95, u95 = model.forecast(12)
        for i in range(12):
            assert l95[i] <= preds[i] <= u95[i]

    def test_theta_aic_bic(self):
        model = ThetaModel(seed=42)
        model.fit(TREND_DATA)
        assert model.aic < float('inf')
        assert model.bic < float('inf')

    def test_theta_flat_series(self):
        model = ThetaModel(seed=42)
        model.fit(np.ones(30) * 50)
        fc = model.forecast(5)
        assert all(abs(v - 50) < 5 for v in fc[0])

    def test_theta_not_fitted_raises(self):
        model = ThetaModel()
        with pytest.raises(ValueError):
            model.forecast(5)


class TestVAR:
    def test_var_fit_forecast(self):
        Y = np.column_stack([TREND_DATA[:80], RANDOM_DATA[:80]])
        model = VARModel(max_lag=3, seed=42)
        model.fit(Y)
        assert model._fitted
        fc = model.forecast(6)
        assert len(fc[0]) == 6

    def test_var_univariate(self):
        model = VARModel(seed=42)
        model.fit(TREND_DATA[:80])
        fc = model.forecast(5)
        assert len(fc[0]) == 5

    def test_var_confidence_intervals(self):
        Y = np.column_stack([TREND_DATA[:80], RANDOM_DATA[:80]])
        model = VARModel(seed=42)
        model.fit(Y)
        preds, l80, u80, l95, u95 = model.forecast(5)
        for i in range(5):
            assert l95[i] <= preds[i] <= u95[i]

    def test_var_aic_bic(self):
        Y = np.column_stack([TREND_DATA[:80], RANDOM_DATA[:80]])
        model = VARModel(seed=42)
        model.fit(Y)
        assert model.aic < float('inf')
        assert model.bic < float('inf')

    def test_var_not_fitted_raises(self):
        model = VARModel()
        with pytest.raises(ValueError):
            model.forecast(5)


class TestTSForecastEnsemble:
    def test_ensemble_basic(self):
        ensemble = TSForecastEnsemble(horizon=12, seed=42)
        result = ensemble.forecast(TREND_DATA)
        assert isinstance(result, EnsembleForecastResult)
        assert len(result.forecast) == 12
        assert len(result.ensemble_weights) > 0
        assert result.dominant_regime in ["expansion", "contraction", "stable", "crisis", "unknown"]

    def test_ensemble_method_results(self):
        ensemble = TSForecastEnsemble(horizon=12, seed=42)
        result = ensemble.forecast(TREND_DATA)
        assert 'arima' in result.method_results or 'ets' in result.method_results
        for name, mr in result.method_results.items():
            assert isinstance(mr, MethodResult)
            assert len(mr.forecast) == 12
            assert mr.mae >= 0

    def test_ensemble_confidence_intervals(self):
        ensemble = TSForecastEnsemble(horizon=12, seed=42)
        result = ensemble.forecast(TREND_DATA)
        for fp in result.forecast:
            assert isinstance(fp, ForecastPoint)
            assert fp.lower_95 <= fp.lower_80 <= fp.value <= fp.upper_80 <= fp.upper_95

    def test_ensemble_short_series(self):
        ensemble = TSForecastEnsemble(horizon=5, seed=42)
        result = ensemble.forecast(SHORT_DATA)
        assert len(result.forecast) == 5
        assert result.recommendation != ""

    def test_ensemble_backtest(self):
        ensemble = TSForecastEnsemble(horizon=6, backtest_windows=2, seed=42)
        result = ensemble.forecast(TREND_DATA)
        assert result.backtest is not None
        assert isinstance(result.backtest, BacktestResult)

    def test_ensemble_regime_detection(self):
        volatile_data = np.concatenate([np.zeros(30), np.cumsum(rng.randn(30)) * 10, np.zeros(30)])
        ensemble = TSForecastEnsemble(horizon=6, seed=42)
        result = ensemble.forecast(volatile_data)
        assert result.dominant_regime in ["expansion", "contraction", "stable", "crisis", "unknown"]

    def test_ensemble_trend_detection(self):
        uptrend = np.arange(50) * 2 + 100 + rng.randn(50)
        ensemble = TSForecastEnsemble(horizon=6, seed=42)
        result = ensemble.forecast(uptrend)
        assert result.trend_direction in ["up", "down", "flat"]

    def test_ensemble_anomaly_detection(self):
        data = np.ones(50)
        data[25] = 100
        ensemble = TSForecastEnsemble(horizon=6, seed=42)
        result = ensemble.forecast(data)
        assert len(result.anomaly_flags) > 0

    def test_ensemble_seasonal_data(self):
        ensemble = TSForecastEnsemble(horizon=12, seed=42)
        result = ensemble.forecast(SEASONAL_DATA)
        assert len(result.forecast) == 12
        assert any(n in result.method_results for n in ['arima', 'ets', 'theta'])

    def test_ensemble_with_exogenous(self):
        exog = np.cumsum(rng.randn(100) * 0.5) + 50
        y = TREND_DATA[:100]
        ensemble = TSForecastEnsemble(horizon=6, seed=42)
        result = ensemble.forecast(y, Y_exog=exog)
        assert len(result.forecast) == 6

    def test_ensemble_recommendation(self):
        ensemble = TSForecastEnsemble(horizon=6, seed=42)
        result = ensemble.forecast(TREND_DATA)
        assert len(result.recommendation) > 20
        assert "regime" in result.recommendation

    def test_ensemble_reproducibility(self):
        ensemble1 = TSForecastEnsemble(horizon=6, seed=42)
        ensemble2 = TSForecastEnsemble(horizon=6, seed=42)
        r1 = ensemble1.forecast(TREND_DATA)
        r2 = ensemble2.forecast(TREND_DATA)
        for i in range(6):
            assert abs(r1.forecast[i].value - r2.forecast[i].value) < 1e-6

    def test_ensemble_very_short_series(self):
        ensemble = TSForecastEnsemble(horizon=10, seed=42)
        result = ensemble.forecast(np.array([1, 2]))
        assert len(result.forecast) == 10

    def test_ensemble_constant_series(self):
        ensemble = TSForecastEnsemble(horizon=6, seed=42)
        result = ensemble.forecast(np.ones(50) * 42)
        assert len(result.forecast) == 6
        assert all(abs(fp.value - 42) < 10 for fp in result.forecast)


class TestBacktest:
    def test_backtest_produces_results(self):
        ensemble = TSForecastEnsemble(horizon=6, backtest_windows=2, seed=42)
        bt = ensemble.backtest(TREND_DATA)
        assert bt.horizon == 6
        assert isinstance(bt.method_errors, dict)
        assert isinstance(bt.method_weights, dict)

    def test_backtest_weights_sum_to_one(self):
        ensemble = TSForecastEnsemble(horizon=6, backtest_windows=2, seed=42)
        bt = ensemble.backtest(TREND_DATA)
        if bt.method_weights:
            total = sum(bt.method_weights.values())
            assert abs(total - 1.0) < 0.01


class TestMetrics:
    def test_compute_metrics(self):
        ensemble = TSForecastEnsemble(horizon=6, seed=42)
        actual = np.array([1, 2, 3, 4, 5, 6])
        predicted = np.array([1.1, 2.2, 2.9, 4.1, 4.8, 6.2])
        metrics = ensemble._compute_metrics(actual, predicted)
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'rmse' in metrics
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0

    def test_compute_metrics_empty(self):
        ensemble = TSForecastEnsemble(horizon=6, seed=42)
        metrics = ensemble._compute_metrics(np.array([]), np.array([]))
        assert metrics['mse'] == float('inf')