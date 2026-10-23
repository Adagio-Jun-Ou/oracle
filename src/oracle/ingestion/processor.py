import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional
from scipy import signal as scipy_signal
from scipy.stats import zscore
from oracle.core.types import TimeSeries, DataPoint, Domain, DataSource, TimeInterval


class DataProcessor:
    def __init__(self, outlier_z_threshold: float = 3.0, seasonal_period: int = 12):
        self.outlier_z_threshold = outlier_z_threshold
        self.seasonal_period = seasonal_period

    def normalize(self, series: TimeSeries, method: str = "zscore") -> TimeSeries:
        values = np.array([d.value for d in series.data])
        if len(values) == 0:
            return series
        if method == "zscore":
            normalized = zscore(values, nan_policy="omit")
        elif method == "minmax":
            mn, mx = np.nanmin(values), np.nanmax(values)
            normalized = (values - mn) / (mx - mn) if mx > mn else np.zeros_like(values)
        elif method == "log":
            normalized = np.log1p(np.abs(values)) * np.sign(values)
        else:
            normalized = values

        return self._rebuild_series(series, normalized.tolist())

    def detrend(self, series: TimeSeries, method: str = "linear") -> TimeSeries:
        values = np.array([d.value for d in series.data])
        if len(values) < 3:
            return series
        if method == "linear":
            coeffs = np.polyfit(range(len(values)), values, 1)
            trend = np.polyval(coeffs, range(len(values)))
            detrended = values - trend
        elif method == "diff":
            detrended = np.diff(values, prepend=values[0])
        else:
            detrended = values
        return self._rebuild_series(series, detrended.tolist())

    def remove_outliers(self, series: TimeSeries) -> TimeSeries:
        values = np.array([d.value for d in series.data])
        if len(values) < 3:
            return series
        z = np.abs(zscore(values, nan_policy="omit"))
        mask = z < self.outlier_z_threshold
        cleaned = np.where(mask, values, np.interp(np.arange(len(values)), np.where(mask)[0], values[mask]))
        return self._rebuild_series(series, cleaned.tolist())

    def resample(self, series: TimeSeries, target_freq: TimeInterval) -> TimeSeries:
        values = np.array([d.value for d in series.data])
        timestamps = [d.timestamp for d in series.data]
        if len(values) < 2:
            return series

        freq_map = {
            TimeInterval.DAILY: "D",
            TimeInterval.WEEKLY: "W",
            TimeInterval.MONTHLY: "ME",
            TimeInterval.QUARTERLY: "QE",
            TimeInterval.YEARLY: "YE",
        }
        target_pd_freq = freq_map.get(target_freq, "ME")

        pd_series = pd.Series(values, index=pd.DatetimeIndex(timestamps))
        resampled = pd_series.resample(target_pd_freq).mean().interpolate(method="linear")

        new_points = [
            DataPoint(
                timestamp=ts.to_pydatetime(),
                value=float(val),
                source=series.source,
                domain=series.domain,
                variable=series.variable,
            )
            for ts, val in resampled.items() if not np.isnan(val)
        ]

        return TimeSeries(
            variable=series.variable,
            domain=series.domain,
            source=series.source,
            data=new_points,
            frequency=target_freq,
        )

    def compute_seasonal_decomposition(self, series: TimeSeries) -> dict:
        values = np.array([d.value for d in series.data])
        if len(values) < 2 * self.seasonal_period:
            return {"trend": values, "seasonal": np.zeros_like(values), "residual": np.zeros_like(values)}

        period = self.seasonal_period
        if len(values) > period * 2:
            result = {}
            trend = np.convolve(values, np.ones(period) / period, mode="same")
            detrended = values - trend
            seasonal = np.zeros_like(values)
            for i in range(period):
                seasonal[i::period] = np.nanmean(detrended[i::period])
            residual = values - trend - seasonal
            result = {"trend": trend, "seasonal": seasonal, "residual": residual}
            return result
        return {"trend": values, "seasonal": np.zeros_like(values), "residual": np.zeros_like(values)}

    def compute_power_spectrum(self, series: TimeSeries) -> dict:
        values = np.array([d.value for d in series.data])
        if len(values) < 4:
            return {"frequencies": np.array([]), "power": np.array([])}
        freqs, psd = scipy_signal.welch(values, nperseg=min(len(values), 256))
        return {"frequencies": freqs, "power": psd}

    def compute_rolling_stats(self, series: TimeSeries, window: int = 12) -> dict:
        values = np.array([d.value for d in series.data])
        if len(values) < window:
            return {"mean": values, "std": np.zeros_like(values), "skew": np.zeros_like(values)}
        df = pd.Series(values)
        rolling = df.rolling(window, min_periods=1)
        return {
            "mean": rolling.mean().values,
            "std": rolling.std().fillna(0).values,
            "skew": rolling.skew().fillna(0).values,
        }

    def _rebuild_series(self, original: TimeSeries, new_values: list[float]) -> TimeSeries:
        new_points = []
        for dp, val in zip(original.data, new_values):
            new_points.append(DataPoint(
                timestamp=dp.timestamp,
                value=float(val),
                source=dp.source,
                domain=dp.domain,
                variable=dp.variable,
            ))
        return TimeSeries(
            variable=original.variable,
            domain=original.domain,
            source=original.source,
            data=new_points,
            frequency=original.frequency,
        )