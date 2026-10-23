import numpy as np
from typing import Optional, Tuple


class TechnicalAnalysis:

    def __init__(self):
        self.name = "technical_analysis"

    @staticmethod
    def sma(data: np.ndarray, period: int = 20) -> np.ndarray:
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        cumsum = np.cumsum(data)
        result[period - 1:] = (cumsum[period - 1:] - np.concatenate([[0], cumsum[:-period]])) / period
        return result

    @staticmethod
    def ema(data: np.ndarray, period: int = 20) -> np.ndarray:
        if len(data) == 0:
            return np.array([])
        multiplier = 2.0 / (period + 1)
        result = np.empty(len(data))
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        if len(data) < period + 1:
            return np.full(len(data), np.nan)
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.zeros(len(data))
        avg_loss = np.zeros(len(data))
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
        rsi_values = 100.0 - (100.0 / (1.0 + rs))
        rsi_values[:period] = np.nan
        return rsi_values

    @staticmethod
    def macd(
        data: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict:
        ta = TechnicalAnalysis()
        ema_fast = ta.ema(data, fast)
        ema_slow = ta.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ta.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return {
            "macd_line": macd_line.tolist(),
            "signal_line": signal_line.tolist(),
            "histogram": histogram.tolist(),
            "fast_period": fast,
            "slow_period": slow,
            "signal_period": signal,
        }

    @staticmethod
    def bollinger_bands(
        data: np.ndarray,
        period: int = 20,
        num_std: float = 2.0,
    ) -> dict:
        ta = TechnicalAnalysis()
        middle = ta.sma(data, period)
        rolling_std = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            rolling_std[i] = np.std(data[i - period + 1:i + 1])
        upper = middle + num_std * rolling_std
        lower = middle - num_std * rolling_std
        bandwidth = np.where(middle != 0, (upper - lower) / middle, np.nan)
        percent_b = np.where((upper - lower) != 0, (data - lower) / (upper - lower), np.nan)
        return {
            "upper": upper.tolist(),
            "middle": middle.tolist(),
            "lower": lower.tolist(),
            "bandwidth": bandwidth.tolist(),
            "percent_b": percent_b.tolist(),
            "period": period,
            "num_std": num_std,
        }

    @staticmethod
    def stochastic(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3,
    ) -> dict:
        n = len(close)
        k_values = np.full(n, np.nan)
        for i in range(k_period - 1, n):
            highest = np.max(high[i - k_period + 1:i + 1])
            lowest = np.min(low[i - k_period + 1:i + 1])
            if highest != lowest:
                k_values[i] = 100.0 * (close[i] - lowest) / (highest - lowest)
            else:
                k_values[i] = 50.0
        ta = TechnicalAnalysis()
        d_values = ta.sma(k_values, d_period)
        return {
            "k_percent": k_values.tolist(),
            "d_percent": d_values.tolist() if isinstance(d_values, np.ndarray) else d_values,
            "k_period": k_period,
            "d_period": d_period,
        }

    @staticmethod
    def atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        n = len(close)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        ta = TechnicalAnalysis()
        return ta.ema(tr, period)

    @staticmethod
    def adx(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> dict:
        n = len(close)
        if n < period * 2:
            return {
                "adx": [np.nan] * n,
                "plus_di": [np.nan] * n,
                "minus_di": [np.nan] * n,
                "period": period,
            }
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        ta = TechnicalAnalysis()
        atr_vals = ta.atr(high, low, close, period)
        smoothed_plus_dm = ta.ema(plus_dm, period)
        smoothed_minus_dm = ta.ema(minus_dm, period)
        plus_di = np.where(atr_vals != 0, 100.0 * smoothed_plus_dm / atr_vals, 0.0)
        minus_di = np.where(atr_vals != 0, 100.0 * smoothed_minus_dm / atr_vals, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            dx = np.where((plus_di + minus_di) != 0, 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)
        adx_vals = ta.ema(dx, period)
        return {
            "adx": adx_vals.tolist(),
            "plus_di": plus_di.tolist(),
            "minus_di": minus_di.tolist(),
            "period": period,
        }

    @staticmethod
    def ichimoku(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26,
    ) -> dict:
        n = len(close)
        ta = TechnicalAnalysis()

        def midpoint(h, l, period):
            mp = np.full(n, np.nan)
            for i in range(period - 1, n):
                mp[i] = (np.max(h[i - period + 1:i + 1]) + np.min(l[i - period + 1:i + 1])) / 2
            return mp

        tenkan = midpoint(high, low, tenkan_period)
        kijun = midpoint(high, low, kijun_period)
        senkou_a = np.where(~np.isnan(tenkan) & ~np.isnan(kijun), (tenkan + kijun) / 2, np.nan)
        senkou_b = midpoint(high, low, senkou_b_period)
        chikou = close
        return {
            "tenkan": tenkan.tolist(),
            "kijun": kijun.tolist(),
            "senkou_a": senkou_a.tolist(),
            "senkou_b": senkou_b.tolist(),
            "chikou": chikou.tolist(),
            "tenkan_period": tenkan_period,
            "kijun_period": kijun_period,
            "senkou_b_period": senkou_b_period,
        }

    @staticmethod
    def vwap(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray:
        typical_price = (high + low + close) / 3.0
        cum_tp_vol = np.cumsum(typical_price * volume)
        cum_vol = np.cumsum(volume)
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.where(cum_vol != 0, cum_tp_vol / cum_vol, typical_price)
        return vwap

    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        result = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                result[i] = result[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                result[i] = result[i - 1] - volume[i]
            else:
                result[i] = result[i - 1]
        return result

    @staticmethod
    def williams_r(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        n = len(close)
        wr = np.full(n, np.nan)
        for i in range(period - 1, n):
            highest = np.max(high[i - period + 1:i + 1])
            lowest = np.min(low[i - period + 1:i + 1])
            if highest != lowest:
                wr[i] = -100.0 * (highest - close[i]) / (highest - lowest)
            else:
                wr[i] = -50.0
        return wr

    @staticmethod
    def cci(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 20,
    ) -> np.ndarray:
        tp = (high + low + close) / 3.0
        n = len(close)
        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            window = tp[i - period + 1:i + 1]
            mean_val = np.mean(window)
            mean_dev = np.mean(np.abs(window - mean_val))
            if mean_dev != 0:
                result[i] = (tp[i] - mean_val) / (0.015 * mean_dev)
            else:
                result[i] = 0.0
        return result

    @staticmethod
    def parabolic_sar(
        high: np.ndarray,
        low: np.ndarray,
        af_start: float = 0.02,
        af_max: float = 0.20,
        af_step: float = 0.02,
    ) -> dict:
        n = len(high)
        if n < 2:
            return {
                "sar": low.tolist(),
                "trend": [1] * n,
                "af": [af_start] * n,
            }
        sar = np.zeros(n)
        trend = np.ones(n, dtype=int)
        af = np.zeros(n)
        ep = np.zeros(n)
        trend[0] = 1
        sar[0] = low[0]
        af[0] = af_start
        ep[0] = high[0]
        for i in range(1, n):
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
            if trend[i - 1] == 1:
                sar[i] = min(sar[i], low[i - 1])
                if i >= 2:
                    sar[i] = min(sar[i], low[i - 2])
            else:
                sar[i] = max(sar[i], high[i - 1])
                if i >= 2:
                    sar[i] = max(sar[i], high[i - 2])
            if trend[i - 1] == 1:
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i - 1]
                    ep[i] = low[i]
                    af[i] = af_start
                else:
                    trend[i] = 1
                    if high[i] > ep[i - 1]:
                        ep[i] = high[i]
                        af[i] = min(af[i - 1] + af_step, af_max)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]
            else:
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i - 1]
                    ep[i] = high[i]
                    af[i] = af_start
                else:
                    trend[i] = -1
                    if low[i] < ep[i - 1]:
                        ep[i] = low[i]
                        af[i] = min(af[i - 1] + af_step, af_max)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]
        return {
            "sar": sar.tolist(),
            "trend": trend.tolist(),
            "af": af.tolist(),
        }

    @staticmethod
    def generate_signals(
        data: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        close: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None,
    ) -> dict:
        ta = TechnicalAnalysis()
        signals = {
            "sma_20": ta.sma(data, 20).tolist(),
            "sma_50": ta.sma(data, 50).tolist(),
            "sma_200": ta.sma(data, 200).tolist(),
            "ema_12": ta.ema(data, 12).tolist(),
            "ema_26": ta.ema(data, 26).tolist(),
            "rsi_14": ta.rsi(data, 14).tolist(),
            "macd_data": ta.macd(data),
        }
        if high is not None and low is not None and close is not None:
            bb = ta.bollinger_bands(data, 20, 2.0)
            signals["bollinger"] = bb
            stoch = ta.stochastic(high, low, close)
            signals["stochastic"] = stoch
            atr_vals = ta.atr(high, low, close, 14)
            if isinstance(atr_vals, np.ndarray):
                signals["atr"] = atr_vals.tolist()
            else:
                signals["atr"] = atr_vals
            wr = ta.williams_r(high, low, close)
            signals["williams_r"] = wr.tolist()
            cci_vals = ta.cci(high, low, close)
            signals["cci"] = cci_vals.tolist()
            if volume is not None:
                vwap_vals = ta.vwap(high, low, close, volume)
                signals["vwap"] = vwap_vals.tolist()
        return signals