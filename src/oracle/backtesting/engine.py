import numpy as np
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
from oracle.core.config import OracleConfig
from oracle.core.types import Domain, Prediction, ConfidenceLevel


@dataclass
class BacktestResult:
    predictions: list[dict] = field(default_factory=list)
    actuals: list[float] = field(default_factory=list)
    predicted: list[float] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    directional_accuracy: float = 0.0
    hit_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    coverage_95: float = 0.0
    coverage_99: float = 0.0
    avg_ci_width: float = 0.0
    calibration_score: float = 0.0
    n_observations: int = 0
    start_date: str = ""
    end_date: str = ""


@dataclass
class WalkForwardResult:
    windows: list[BacktestResult] = field(default_factory=list)
    aggregate: Optional[BacktestResult] = None
    n_windows: int = 0
    train_ratio: float = 0.7
    window_size: int = 60


class BacktestEngine:
    def __init__(self, oracle=None):
        self.oracle = oracle

    def backtest_series(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        ci_lower: Optional[np.ndarray] = None,
        ci_upper: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        n = min(len(actual), len(predicted))
        actual = actual[:n]
        predicted = predicted[:n]
        errors = actual - predicted
        abs_errors = np.abs(errors)
        pct_errors = np.abs(errors) / (np.abs(actual) + 1e-10)

        actual_direction = np.diff(actual) > 0
        pred_direction = np.diff(predicted) > 0
        if len(actual_direction) > 0:
            directional_accuracy = float(np.mean(actual_direction == pred_direction))
        else:
            directional_accuracy = 0.0

        returns = np.diff(predicted) / (np.abs(predicted[:-1]) + 1e-10)
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10)) if len(returns) > 1 else 0.0

        cummax = np.maximum.accumulate(predicted)
        drawdowns = (cummax - predicted) / (cummax + 1e-10)
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        coverage_95 = 0.0
        coverage_99 = 0.0
        avg_ci_width = 0.0
        calibration = 0.0
        if ci_lower is not None and ci_upper is not None:
            ci_lower = ci_lower[:n]
            ci_upper = ci_upper[:n]
            in_95 = (actual >= ci_lower) & (actual <= ci_upper)
            coverage_95 = float(np.mean(in_95))
            coverage_99 = coverage_95
            avg_ci_width = float(np.mean(ci_upper - ci_lower))
            z_scores = (actual - predicted) / ((ci_upper - ci_lower) / (2 * 1.96) + 1e-10)
            calibration = float(1 - np.mean(np.abs(z_scores) > 1.96))

        return BacktestResult(
            actuals=actual.tolist(),
            predicted=predicted.tolist(),
            errors=errors.tolist(),
            mae=float(np.mean(abs_errors)),
            rmse=float(np.sqrt(np.mean(errors ** 2))),
            mape=float(np.mean(pct_errors)) * 100,
            directional_accuracy=directional_accuracy,
            hit_rate=directional_accuracy,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            coverage_95=coverage_95,
            coverage_99=coverage_99,
            avg_ci_width=avg_ci_width,
            calibration_score=calibration,
            n_observations=n,
        )

    def walk_forward(
        self,
        series: np.ndarray,
        window_size: int = 60,
        n_windows: int = 10,
        horizon: int = 5,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    ) -> WalkForwardResult:
        actuals_all = []
        predicted_all = []
        ci_lower_all = []
        ci_upper_all = []
        windows = []

        total_avail = len(series) - window_size - horizon
        if total_avail < n_windows:
            n_windows_total = max(1, total_avail)
        else:
            n_windows_total = n_windows

        step = max(1, total_avail // n_windows_total) if total_avail > 0 else 1

        for i in range(n_windows_total):
            start = i * step
            end_train = start + window_size
            end_test = end_train + horizon

            if end_test > len(series):
                break

            train_data = series[start:end_train]
            test_actual = series[end_train:end_test]

            self.oracle.add_time_series("backtest_series", train_data, Domain.ECONOMICS)

            target = date.today() + timedelta(days=horizon * 30)
            try:
                forecast = self.oracle.predict(target, confidence)
                stat_preds = [p for p in forecast.predictions
                              if p.model_name == "statistical_deterministic" and p.variable == "backtest_series"]
                if stat_preds:
                    pred_val = stat_preds[0].value
                    ci_l = stat_preds[0].confidence_interval[0]
                    ci_u = stat_preds[0].confidence_interval[1]
                else:
                    pred_val = float(np.mean(train_data[-horizon:]))
                    ci_l = pred_val - np.std(train_data) * 1.96
                    ci_u = pred_val + np.std(train_data) * 1.96

                predicted_all.extend([pred_val] * len(test_actual))
                actuals_all.extend(test_actual.tolist())
                ci_lower_all.extend([ci_l] * len(test_actual))
                ci_upper_all.extend([ci_u] * len(test_actual))
            except Exception:
                continue

            self.oracle.state.raw_series.pop("backtest_series", None)

        if len(actuals_all) > 0:
            result = self.backtest_series(
                np.array(actuals_all), np.array(predicted_all),
                np.array(ci_lower_all), np.array(ci_upper_all),
            )
        else:
            result = BacktestResult()

        return WalkForwardResult(
            windows=windows,
            aggregate=result,
            n_windows=n_windows_total,
            train_ratio=window_size / (window_size + horizon),
            window_size=window_size,
        )

    def compare_models(
        self,
        series: np.ndarray,
        window_size: int = 60,
        n_windows: int = 5,
        horizon: int = 5,
    ) -> dict:
        from oracle.deterministic.statistical_model import StatisticalDeterministicModel
        from oracle.core.types import ConfidenceLevel
        from oracle.prediction.engine import OracleEngine
        from oracle.nondeterministic.monte_carlo import MonteCarloEngine
        from oracle.nondeterministic.bayesian import BayesianEngine

        models = {
            "statistical": StatisticalDeterministicModel(),
            "monte_carlo": MonteCarloEngine(n_simulations=500),
            "bayesian": BayesianEngine(),
        }

        results = {}
        for name, model in models.items():
            actuals = []
            predicted = []
            step_size = max(1, (len(series) - window_size - horizon) // n_windows)

            for i in range(0, min(n_windows, len(series) - window_size - horizon), 1):
                start = i * step_size if step_size > 0 else i
                end_train = start + window_size
                end_test = end_train + horizon

                if end_test > len(series):
                    break

                train = series[start:end_train]
                test = series[end_train:end_test]

                try:
                    target = date.today() + timedelta(days=horizon * 30)
                    pred = model.predict(train, f"{name}_test", Domain.ECONOMICS, target, ConfidenceLevel.MEDIUM)
                    predicted.append(pred.value)
                    actuals.append(float(np.mean(test)))
                except Exception:
                    continue

            if len(actuals) >= 2:
                actual_arr = np.array(actuals)
                pred_arr = np.array(predicted)
                errors = actual_arr - pred_arr
                results[name] = {
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(errors ** 2))),
                    "mape": float(np.mean(np.abs(errors) / (np.abs(actual_arr) + 1e-10))) * 100,
                    "directional_accuracy": float(np.mean((np.diff(actual_arr) > 0) == (np.diff(pred_arr) > 0))) if len(actual_arr) > 1 else 0,
                    "n_observations": len(actuals),
                }

        if results:
            best_model = min(results, key=lambda k: results[k].get("mae", float("inf")))
            results["best_model"] = best_model

        return results