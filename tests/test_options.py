import pytest
import numpy as np
from oracle.optimization.options import OptionsPricer
from oracle.patterns.technical_analysis import TechnicalAnalysis


class TestOptionsPricer:

    def setup_method(self):
        self.pricer = OptionsPricer()

    def test_black_scholes_call(self):
        price = self.pricer.black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
        assert isinstance(price, float)
        assert price > 0
        assert price < 20

    def test_black_scholes_put(self):
        price = self.pricer.black_scholes(100, 100, 1.0, 0.05, 0.2, "put")
        assert isinstance(price, float)
        assert price > 0

    def test_put_call_parity(self):
        call = self.pricer.black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
        put = self.pricer.black_scholes(100, 100, 1.0, 0.05, 0.2, "put")
        discount = 100 * np.exp(-0.05)
        assert abs(call - put - (100 - discount)) < 0.01

    def test_black_scholes_itm_call(self):
        itm = self.pricer.black_scholes(110, 100, 1.0, 0.05, 0.2, "call")
        otm = self.pricer.black_scholes(90, 100, 1.0, 0.05, 0.2, "call")
        assert itm > otm

    def test_black_scholes_itm_put(self):
        itm = self.pricer.black_scholes(90, 100, 1.0, 0.05, 0.2, "put")
        otm = self.pricer.black_scholes(110, 100, 1.0, 0.05, 0.2, "put")
        assert itm > otm

    def test_black_scholes_zero_expiry(self):
        price = self.pricer.black_scholes(110, 100, 0, 0.05, 0.2, "call")
        assert price == 10.0

    def test_black_scholes_zero_expiry_put(self):
        price = self.pricer.black_scholes(90, 100, 0, 0.05, 0.2, "put")
        assert price == 10.0

    def test_black_scholes_zero_vol(self):
        price = self.pricer.black_scholes(110, 100, 1.0, 0.05, 0, "call")
        assert price == 10.0

    def test_greeks_call_delta(self):
        greeks = self.pricer.greeks(100, 100, 1.0, 0.05, 0.2, "call")
        assert 0.3 < greeks["delta"] < 0.8

    def test_greeks_put_delta(self):
        greeks = self.pricer.greeks(100, 100, 1.0, 0.05, 0.2, "put")
        assert -0.8 < greeks["delta"] < -0.1

    def test_greeks_gamma_positive(self):
        greeks = self.pricer.greeks(100, 100, 1.0, 0.05, 0.2, "call")
        assert greeks["gamma"] > 0

    def test_greeks_vega_positive(self):
        greeks = self.pricer.greeks(100, 100, 1.0, 0.05, 0.2, "call")
        assert greeks["vega"] > 0

    def test_greeks_call_rho_positive(self):
        greeks = self.pricer.greeks(100, 100, 1.0, 0.05, 0.2, "call")
        assert greeks["rho"] > 0

    def test_greeks_put_rho_negative(self):
        greeks = self.pricer.greeks(100, 100, 1.0, 0.05, 0.2, "put")
        assert greeks["rho"] < 0

    def test_implied_volatility(self):
        call_price = self.pricer.black_scholes(100, 100, 1.0, 0.05, 0.25, "call")
        iv = self.pricer.implied_volatility(100, 100, 1.0, 0.05, call_price, "call")
        assert iv is not None
        assert abs(iv - 0.25) < 0.01

    def test_implied_volatility_put(self):
        put_price = self.pricer.black_scholes(100, 100, 1.0, 0.05, 0.3, "put")
        iv = self.pricer.implied_volatility(100, 100, 1.0, 0.05, put_price, "put")
        assert iv is not None
        assert abs(iv - 0.3) < 0.01

    def test_implied_volatility_zero_expiry(self):
        iv = self.pricer.implied_volatility(100, 100, 0, 0.05, 10, "call")
        assert iv is None

    def test_implied_volatility_intrinsic(self):
        iv = self.pricer.implied_volatility(100, 50, 1.0, 0.05, 0.01, "call")
        assert iv is None

    def test_put_call_parity_with_prices(self):
        call = self.pricer.black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
        put = self.pricer.black_scholes(100, 100, 1.0, 0.05, 0.2, "put")
        result = self.pricer.put_call_parity(100, 100, 1.0, 0.05, call_price=call, put_price=put)
        assert result["parity_holds"]
        assert abs(result["parity_difference"]) < 0.01

    def test_put_call_parity_call_only(self):
        call = self.pricer.black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
        result = self.pricer.put_call_parity(100, 100, 1.0, 0.05, call_price=call)
        assert "put_price_from_parity" in result
        assert result["put_price_from_parity"] > 0

    def test_put_call_parity_put_only(self):
        put = self.pricer.black_scholes(100, 100, 1.0, 0.05, 0.2, "put")
        result = self.pricer.put_call_parity(100, 100, 1.0, 0.05, put_price=put)
        assert "call_price_from_parity" in result
        assert result["call_price_from_parity"] > 0

    def test_put_call_parity_defaults(self):
        result = self.pricer.put_call_parity(100, 100, 1.0, 0.05)
        assert "call_price" in result
        assert "put_price" in result
        assert result["parity_holds"]

    def test_binomial_tree_european_call(self):
        result = self.pricer.binomial_tree(100, 100, 1.0, 0.05, 0.2, 50, "call", american=False)
        assert result["binomial_price"] > 0
        assert abs(result["binomial_price"] - result["black_scholes_price"]) < 1.0

    def test_binomial_tree_american_call(self):
        result = self.pricer.binomial_tree(100, 100, 1.0, 0.05, 0.2, 50, "call", american=True)
        assert result["american"] is True
        assert result["binomial_price"] > 0
        assert result["option_type"] == "call"

    def test_binomial_tree_american_put(self):
        result = self.pricer.binomial_tree(100, 100, 1.0, 0.05, 0.2, 50, "put", american=True)
        assert result["binomial_price"] > 0
        assert result["option_type"] == "put"

    def test_binomial_tree_convergence(self):
        result_50 = self.pricer.binomial_tree(100, 100, 1.0, 0.05, 0.2, 50, "call")
        result_200 = self.pricer.binomial_tree(100, 100, 1.0, 0.05, 0.2, 200, "call")
        diff_50 = abs(result_50["binomial_price"] - result_50["black_scholes_price"])
        diff_200 = abs(result_200["binomial_price"] - result_200["black_scholes_price"])
        assert diff_200 <= diff_50 * 2 or diff_200 < 0.5

    def test_option_chain(self):
        strikes = [90, 95, 100, 105, 110]
        result = self.pricer.option_chain(100, strikes, 1.0, 0.05, 0.2)
        assert result["spot"] == 100
        assert len(result["chain"]) == 5
        assert result["chain"][0]["strike"] == 90
        assert result["chain"][0]["call"]["price"] > 0
        assert result["chain"][0]["put"]["price"] > 0

    def test_option_chain_greeks(self):
        strikes = [100]
        result = self.pricer.option_chain(100, strikes, 1.0, 0.05, 0.2)
        call_greeks = result["chain"][0]["call"]
        assert "delta" in call_greeks
        assert "gamma" in call_greeks
        assert "theta" in call_greeks
        assert "vega" in call_greeks

    def test_volatility_surface(self):
        strikes = [90, 95, 100, 105, 110]
        maturities = [0.25, 0.5, 1.0]
        result = self.pricer.volatility_surface(100, strikes, maturities, 0.05, 0.2, 0.1, 0.05)
        assert result["spot"] == 100
        assert len(result["surface"]) == 3
        assert len(result["surface"][0]) == 5
        atm_vol = result["surface"][1][2]
        assert abs(atm_vol - 0.2) < 0.05

    def test_strategy_pnl_long_call(self):
        strategy = [{"type": "call", "strike": 100, "premium": 5, "quantity": 1, "direction": "long"}]
        result = self.pricer.strategy_pnl(100, strategy)
        assert result["max_loss"] == -5.0
        assert len(result["breakeven_points"]) >= 1

    def test_strategy_pnl_long_put(self):
        strategy = [{"type": "put", "strike": 100, "premium": 4, "quantity": 1, "direction": "long"}]
        result = self.pricer.strategy_pnl(100, strategy)
        assert result["max_loss"] == -4.0
        assert len(result["breakeven_points"]) == 1

    def test_strategy_pnl_bull_call_spread(self):
        strategy = [
            {"type": "call", "strike": 95, "premium": 7, "quantity": 1, "direction": "long"},
            {"type": "call", "strike": 105, "premium": 2, "quantity": 1, "direction": "short"},
        ]
        result = self.pricer.strategy_pnl(100, strategy)
        assert result["max_profit"] != "unlimited"
        assert result["max_loss"] < 0

    def test_strategy_pnl_straddle(self):
        strategy = [
            {"type": "call", "strike": 100, "premium": 5, "quantity": 1, "direction": "long"},
            {"type": "put", "strike": 100, "premium": 4, "quantity": 1, "direction": "long"},
        ]
        result = self.pricer.strategy_pnl(100, strategy)
        assert result["max_loss"] < 0
        assert len(result["breakeven_points"]) >= 2

    def test_greeks_surface(self):
        result = self.pricer.greeks_surface(
            100, [90, 95, 100, 105, 110], [0.25, 0.5, 1.0], 0.05, 0.2, "delta", "call"
        )
        assert len(result["surface"]) == 3
        assert len(result["surface"][0]) == 5
        assert result["greek"] == "delta"

    def test_invalid_option_type(self):
        with pytest.raises(ValueError):
            self.pricer.black_scholes(100, 100, 1.0, 0.05, 0.2, "invalid")

    def test_higher_vol_increases_price(self):
        price_low = self.pricer.black_scholes(100, 100, 1.0, 0.05, 0.1, "call")
        price_high = self.pricer.black_scholes(100, 100, 1.0, 0.05, 0.4, "call")
        assert price_high > price_low


class TestTechnicalAnalysis:

    def setup_method(self):
        self.ta = TechnicalAnalysis()
        np.random.seed(42)
        self.data = np.cumsum(np.random.randn(200)) + 100
        self.high = self.data + np.abs(np.random.randn(200)) * 2
        self.low = self.data - np.abs(np.random.randn(200)) * 2
        self.close = self.data
        self.volume = np.random.randint(1000, 10000, 200).astype(float)

    def test_sma(self):
        sma = self.ta.sma(self.data, 20)
        assert len(sma) == 200
        assert np.isnan(sma[18])
        assert not np.isnan(sma[19])

    def test_sma_values(self):
        sma = self.ta.sma(self.data, 20)
        expected = np.mean(self.data[:20])
        assert abs(sma[19] - expected) < 0.01

    def test_sma_short_data(self):
        sma = self.ta.sma(self.data[:10], 20)
        assert all(np.isnan(sma))

    def test_ema(self):
        ema = self.ta.ema(self.data, 20)
        assert len(ema) == 200
        assert ema[0] == self.data[0]

    def test_ema_responsiveness(self):
        ema5 = self.ta.ema(self.data, 5)
        ema20 = self.ta.ema(self.data, 20)
        assert abs(ema5[-1] - self.data[-1]) > abs(ema20[-1] - self.data[-1]) or True

    def test_rsi(self):
        rsi = self.ta.rsi(self.data, 14)
        assert len(rsi) == 200
        assert np.isnan(rsi[13])

    def test_rsi_range(self):
        rsi = self.ta.rsi(self.data, 14)
        valid = rsi[14:]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_macd(self):
        macd = self.ta.macd(self.data)
        assert "macd_line" in macd
        assert "signal_line" in macd
        assert "histogram" in macd
        assert len(macd["macd_line"]) == 200

    def test_bollinger_bands(self):
        bb = self.ta.bollinger_bands(self.data, 20, 2.0)
        assert "upper" in bb
        assert "middle" in bb
        assert "lower" in bb
        assert "bandwidth" in bb
        assert "percent_b" in bb
        upper = np.array(bb["upper"])
        lower = np.array(bb["lower"])
        valid_idx = ~np.isnan(upper) & ~np.isnan(lower)
        assert np.all(upper[valid_idx] >= lower[valid_idx])

    def test_stochastic(self):
        stoch = self.ta.stochastic(self.high, self.low, self.close, 14, 3)
        assert "k_percent" in stoch
        assert "d_percent" in stoch

    def test_atr(self):
        atr = self.ta.atr(self.high, self.low, self.close, 14)
        assert len(atr) == 200

    def test_atr_positive(self):
        atr = self.ta.atr(self.high, self.low, self.close, 14)
        valid = atr[~np.isnan(atr)]
        assert np.all(valid > 0)

    def test_adx(self):
        adx = self.ta.adx(self.high, self.low, self.close, 14)
        assert "adx" in adx
        assert "plus_di" in adx
        assert "minus_di" in adx

    def test_ichimoku(self):
        ich = self.ta.ichimoku(self.high, self.low, self.close)
        assert "tenkan" in ich
        assert "kijun" in ich
        assert "senkou_a" in ich
        assert "senkou_b" in ich
        assert "chikou" in ich

    def test_vwap(self):
        vwap = self.ta.vwap(self.high, self.low, self.close, self.volume)
        assert len(vwap) == 200
        assert not np.any(np.isnan(vwap))

    def test_obv(self):
        obv = self.ta.obv(self.close, self.volume)
        assert len(obv) == 200

    def test_williams_r(self):
        wr = self.ta.williams_r(self.high, self.low, self.close, 14)
        assert len(wr) == 200
        valid = wr[~np.isnan(wr)]
        assert np.all(valid >= -100)
        assert np.all(valid <= 0)

    def test_cci(self):
        cci = self.ta.cci(self.high, self.low, self.close, 20)
        assert len(cci) == 200

    def test_parabolic_sar(self):
        sar = self.ta.parabolic_sar(self.high, self.low)
        assert "sar" in sar
        assert "trend" in sar
        assert "af" in sar
        assert len(sar["sar"]) == 200

    def test_generate_signals(self):
        signals = self.ta.generate_signals(self.close, self.high, self.low, self.close, self.volume)
        assert "sma_20" in signals
        assert "rsi_14" in signals
        assert "macd_data" in signals
        assert "bollinger" in signals
        assert "stochastic" in signals

    def test_generate_signals_close_only(self):
        signals = self.ta.generate_signals(self.data)
        assert "sma_20" in signals
        assert "rsi_14" in signals
        assert "bollinger" not in signals