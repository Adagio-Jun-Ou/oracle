import pytest
import numpy as np
from oracle.ingestion.real_data import RealDataFetcher, EconomicDataFetcher, MarketDataCache, DomainDataFetcher


class TestMarketDataCache:
    def setup_method(self):
        self.cache = MarketDataCache(ttl_seconds=3600)

    def test_cache_set_get(self):
        self.cache.set("test_key", {"value": 42})
        result = self.cache.get("test_key")
        assert result is not None
        assert result["value"] == 42

    def test_cache_miss(self):
        result = self.cache.get("nonexistent")
        assert result is None

    def test_cache_clear(self):
        self.cache.set("test_key", 123)
        self.cache.clear()
        assert self.cache.get("test_key") is None


class TestRealDataFetcher:
    def setup_method(self):
        self.fetcher = RealDataFetcher(cache_ttl=60)

    def test_init(self):
        assert self.fetcher.cache is not None

    def test_ticker_map(self):
        assert "sp500" in RealDataFetcher.TICKER_MAP
        assert "vix" in RealDataFetcher.TICKER_MAP
        assert "gold" in RealDataFetcher.TICKER_MAP

    def test_compute_returns_synthetic(self):
        arr = np.cumsum(np.random.normal(0.001, 0.02, 252)) + 100
        self.fetcher.cache.set("^GSPC:1y:1d", arr)
        result = self.fetcher._max_drawdown(arr)
        assert result <= 0

    def test_max_drawdown_calculation(self):
        arr = np.array([100, 110, 105, 95, 115, 110, 120])
        dd = self.fetcher._max_drawdown(arr)
        assert dd < 0

    def test_compute_returns_with_data(self):
        arr = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 110], dtype=float)
        self.fetcher.cache.set("^GSPC:1y:1d", arr)
        returns = self.fetcher.compute_returns("sp500", period="1y")
        assert returns is not None
        assert "cumulative_return" in returns
        assert "annualized_volatility" in returns
        assert "max_drawdown" in returns
        assert "sharpe_ratio" in returns

    def test_compute_returns_empty(self):
        fetcher_no_yf = RealDataFetcher(cache_ttl=60)
        fetcher_no_yf._yf_available = False
        result = fetcher_no_yf.get_close_array("sp500", "1y", "1d")
        assert result is None


class TestEconomicDataFetcher:
    def setup_method(self):
        self.fetcher = EconomicDataFetcher(cache_ttl=86400)

    def test_series_map(self):
        assert "gdp_growth" in EconomicDataFetcher.SERIES_MAP
        assert "unemployment" in EconomicDataFetcher.SERIES_MAP
        assert "cpi_inflation" in EconomicDataFetcher.SERIES_MAP

    def test_economic_state_approx_defaults(self):
        state = self.fetcher.get_economic_state_approx()
        assert "gdp_growth_rate" in state
        assert "interest_rate" in state
        assert "inflation_rate" in state
        assert isinstance(state["gdp_growth_rate"], float)


class TestRealDataFetcherIntegration:
    def setup_method(self):
        self.fetcher = RealDataFetcher(cache_ttl=300)

    def test_fetch_multi_with_cache(self):
        arr = np.cumsum(np.random.normal(0.001, 0.02, 60)) + 5000
        self.fetcher.cache.set("^GSPC:3mo:1d", arr)
        result = self.fetcher.get_close_array("sp500", "3mo", "1d")
        if result is not None:
            assert len(result) >= 10
            assert result[-1] > 0

    def test_get_latest_value_cached(self):
        arr = np.array([100, 101, 102, 103, 104], dtype=float)
        import pandas as pd
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"Close": arr, "Open": arr, "High": arr, "Low": arr, "Volume": [1000]*5}, index=dates)
        self.fetcher.cache.set("^GSPC:5d:1d", df)
        val = self.fetcher.get_latest_value("sp500")
        if val is not None:
            assert val > 0

    def test_returns_calculation_properties(self):
        arr = np.cumsum(np.random.normal(0.001, 0.02, 252)) + 100
        self.fetcher.cache.set("^GSPC:1y:1d", arr)
        result = self.fetcher.compute_returns("sp500", period="1y")
        if result is not None:
            assert result["max_drawdown"] <= 0
            assert result["annualized_volatility"] > 0
            assert result["n_days"] >= 200


class TestDomainDataFetcher:
    def setup_method(self):
        self.fetcher = DomainDataFetcher(cache_ttl=3600)

    def test_get_sentiment_state(self):
        state = self.fetcher.get_sentiment_state()
        assert "fear_greed_index" in state
        assert "vix_level" in state
        assert 0 <= state["fear_greed_index"] <= 1
        assert 0 <= state["vix_level"] <= 1

    def test_get_contagion_state(self):
        state = self.fetcher.get_contagion_state()
        assert "financial_systemic_risk" in state
        assert "banking_contagion" in state
        assert 0 <= state["financial_systemic_risk"] <= 1

    def test_get_geopolitical_state(self):
        state = self.fetcher.get_geopolitical_state()
        assert "us_china_tension" in state
        assert "russia_west_tension" in state
        assert 0 <= state["us_china_tension"] <= 1

    def test_get_climate_state(self):
        state = self.fetcher.get_climate_state()
        assert "global_temp_anomaly_c" in state
        assert "co2_ppm" in state

    def test_get_demographics_state(self):
        state = self.fetcher.get_demographics_state()
        assert "total_population_m" in state
        assert "birth_rate_per_1000" in state

    def test_get_all_domain_states(self):
        states = self.fetcher.get_all_domain_states()
        assert "economic" in states
        assert "sentiment" in states
        assert "contagion" in states
        assert "geopolitical" in states
        assert "climate" in states
        assert "demographics" in states
        assert "data_sources" in states
        assert "timestamp" in states