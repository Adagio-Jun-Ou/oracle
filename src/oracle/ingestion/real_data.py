import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional
import logging
import time

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class MarketDataCache:
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: dict = {}
        self._timestamps: dict = {}
        self.ttl = ttl_seconds

    def get(self, key: str):
        if key in self._cache:
            if time.time() - self._timestamps.get(key, 0) < self.ttl:
                return self._cache[key]
        return None

    def set(self, key: str, value):
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def clear(self):
        self._cache.clear()
        self._timestamps.clear()


class RealDataFetcher:
    TICKER_MAP = {
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "dji": "^DJI",
        "vix": "^VIX",
        "gold": "GC=F",
        "oil": "CL=F",
        "bitcoin": "BTC-USD",
        "usd_index": "DX-Y.NYB",
        "treasury_10y": "^TNX",
        "treasury_2y": "^IRX",
        "russell_2000": "^RUT",
    }

    def __init__(self, cache_ttl: int = 3600):
        self.cache = MarketDataCache(ttl_seconds=cache_ttl)
        self._yf_available = YFINANCE_AVAILABLE

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def fetch_series(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}:{period}:{interval}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        if not self._yf_available:
            return None

        ticker = self.TICKER_MAP.get(symbol, symbol)
        for attempt in range(self.MAX_RETRIES):
            try:
                data = yf.download(ticker, period=period, interval=interval, progress=False)
                if data is None or len(data) == 0:
                    if attempt < self.MAX_RETRIES - 1:
                        logger.info(f"Empty data for {symbol}, retry {attempt+1}/{self.MAX_RETRIES}")
                        time.sleep(self.RETRY_DELAY * (attempt + 1))
                        continue
                    return None

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                self.cache.set(cache_key, data)
                return data
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} (attempt {attempt+1}/{self.MAX_RETRIES}): {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    return None
        return None

    def get_close_array(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[np.ndarray]:
        data = self.fetch_series(symbol, period, interval)
        if data is None or "Close" not in data.columns:
            return None
        close = data["Close"].dropna()
        if len(close) == 0:
            return None
        try:
            if isinstance(close.iloc[-1], (pd.Series, np.ndarray)):
                close = close.iloc[:, 0] if hasattr(close.iloc[-1], 'iloc') else close
        except (IndexError, KeyError):
            pass
        values = close.values.astype(float)
        valid_mask = np.isfinite(values)
        if not np.any(valid_mask):
            return None
        return values[valid_mask]

    def get_latest_value(self, symbol: str) -> Optional[float]:
        data = self.fetch_series(symbol, period="5d", interval="1d")
        if data is None or len(data) == 0:
            return None
        close = data["Close"].dropna()
        if len(close) == 0:
            return None
        val = close.iloc[-1]
        if hasattr(val, 'item'):
            return float(val.item())
        return float(val)

    def fetch_multi(self, symbols: list[str], period: str = "1y", interval: str = "1d") -> dict[str, np.ndarray]:
        results = {}
        if self._yf_available and len(symbols) > 0:
            yf_tickers = [self.TICKER_MAP.get(s, s) for s in symbols]
            try:
                data = yf.download(yf_tickers, period=period, interval=interval, progress=False)
                if data is not None and len(data) > 0:
                    for symbol, ticker in zip(symbols, yf_tickers):
                        try:
                            if len(symbols) == 1:
                                close = data["Close"].dropna()
                            else:
                                close = data["Close"][ticker].dropna()
                            if len(close) > 0:
                                results[symbol] = close.values.astype(float)
                        except (KeyError, TypeError):
                            pass
            except Exception as e:
                logger.warning(f"Multi-fetch failed: {e}")

        for symbol in symbols:
            if symbol not in results:
                arr = self.get_close_array(symbol, period, interval)
                if arr is not None:
                    results[symbol] = arr

        return results

    def get_economic_snapshot(self) -> dict:
        snapshot = {}
        tickers = {
            "sp500": "^GSPC", "nasdaq": "^IXIC", "vix": "^VIX",
            "gold": "GC=F", "treasury_10y": "^TNX",
            "oil": "CL=F", "bitcoin": "BTC-USD",
            "usd_index": "DX-Y.NYB",
        }

        for name, ticker in tickers.items():
            val = self.get_latest_value(ticker)
            if val is not None:
                snapshot[name] = val

        return snapshot

    def compute_returns(self, symbol: str, period: str = "1y") -> Optional[dict]:
        arr = self.get_close_array(symbol, period, "1d")
        if arr is None or len(arr) < 2:
            return None

        returns = np.diff(arr) / arr[:-1]
        cumulative_return = (arr[-1] / arr[0]) - 1
        volatility = float(np.std(returns) * np.sqrt(252))

        return {
            "current": float(arr[-1]),
            "period_start": float(arr[0]),
            "cumulative_return": float(cumulative_return),
            "annualized_volatility": float(volatility),
            "max_drawdown": float(self._max_drawdown(arr)),
            "sharpe_ratio": float(cumulative_return / volatility) if volatility > 0 else 0.0,
            "daily_returns_mean": float(np.mean(returns)),
            "daily_returns_std": float(np.std(returns)),
            "n_days": len(arr),
        }

    def _max_drawdown(self, arr: np.ndarray) -> float:
        peak = arr[0]
        max_dd = 0.0
        for val in arr:
            if val > peak:
                peak = val
            dd = (val - peak) / peak
            if dd < max_dd:
                max_dd = dd
        return max_dd


class EconomicDataFetcher:
    FRED_API_BASE = "https://api.stlouisfed.org/fred"

    SERIES_MAP = {
        "gdp_growth": "A191RL1Q225SBEA",
        "unemployment": "UNRATE",
        "cpi_inflation": "CPIAUCSL",
        "federal_funds_rate": "FEDFUNDS",
        "m2_money_supply": "M2SL",
        "treasury_10y": "DGS10",
        "treasury_2y": "DGS2",
        "consumer_confidence": "UMCSENT",
        "industrial_production": "INDPRO",
        "housing_starts": "HOUST",
        "retail_sales": "RSAFS",
        "debt_to_gdp": "GFDEBTN",
        "snp_500": "SP500",
        "vix": "VIXCLS",
    }

    def __init__(self, fred_api_key: Optional[str] = None, cache_ttl: int = 86400):
        self.api_key = fred_api_key
        self.cache = MarketDataCache(ttl_seconds=cache_ttl)

    def fetch_fred_series(self, series_id: str, observation_start: str = "2020-01-01") -> Optional[pd.DataFrame]:
        cache_key = f"fred:{series_id}:{observation_start}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        if not REQUESTS_AVAILABLE or not self.api_key:
            return None

        try:
            url = f"{self.FRED_API_BASE}/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": observation_start,
                "sort_order": "desc",
                "limit": 120,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            observations = data.get("observations", [])
            if not observations:
                return None

            df = pd.DataFrame(observations)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df[["date", "value"]].dropna().sort_values("date")

            self.cache.set(cache_key, df)
            return df
        except Exception as e:
            logger.warning(f"FRED fetch failed for {series_id}: {e}")
            return None

    def get_latest_values(self, series_ids: Optional[list[str]] = None) -> dict[str, float]:
        if series_ids is None:
            series_ids = list(self.SERIES_MAP.keys())

        results = {}
        for name in series_ids:
            fred_id = self.SERIES_MAP.get(name, name)
            df = self.fetch_fred_series(fred_id)
            if df is not None and len(df) > 0:
                results[name] = float(df["value"].iloc[-1])

        return results

    def get_economic_state_approx(self) -> dict:
        market_fetcher = RealDataFetcher(cache_ttl=3600)
        snapshot = market_fetcher.get_economic_snapshot()

        state = {
            "gdp_growth_rate": 0.025,
            "debt_to_gdp": 1.2,
            "interest_rate": 0.045,
            "inflation_rate": 0.03,
            "unemployment_rate": 0.04,
            "stock_valuation": 1.0,
            "capacity_utilization": 0.78,
        }

        if "sp500" in snapshot:
            state["sp500"] = snapshot["sp500"]
        if "vix" in snapshot:
            state["vix"] = snapshot["vix"]
            state["market_fear"] = min(snapshot["vix"] / 40.0, 1.0)
        if "treasury_10y" in snapshot:
            state["interest_rate"] = snapshot["treasury_10y"] / 100.0
            state["debt_to_gdp"] = 1.23
        if "gold" in snapshot:
            state["gold_price"] = snapshot["gold"]
        if "nasdaq" in snapshot:
            state["nasdaq"] = snapshot["nasdaq"]

        fred_data = self.get_latest_values(["unemployment", "cpi_inflation", "federal_funds_rate"])
        if "unemployment" in fred_data:
            state["unemployment_rate"] = fred_data["unemployment"] / 100.0
        if "cpi_inflation" in fred_data:
            state["inflation_rate"] = fred_data["cpi_inflation"] / 100.0
        if "federal_funds_rate" in fred_data:
            state["interest_rate"] = fred_data["federal_funds_rate"] / 100.0

        return state


class DomainDataFetcher:
    def __init__(self, cache_ttl: int = 86400):
        self.market = RealDataFetcher(cache_ttl=cache_ttl)
        self.econ = EconomicDataFetcher(cache_ttl=cache_ttl)
        self.cache = MarketDataCache(ttl_seconds=cache_ttl)

    def get_sentiment_state(self) -> dict:
        snapshot = self.market.get_economic_snapshot()
        vix = snapshot.get("vix", 20.0)
        vix_normalized = min(max(vix / 40.0, 0.0), 1.0)

        sp500_data = self.market.get_close_array("sp500", period="3mo", interval="1d")
        if sp500_data is not None and len(sp500_data) > 20:
            returns = np.diff(sp500_data) / sp500_data[:-1]
            neg_days = np.sum(returns < 0) / len(returns)
            put_call_proxy = 0.5 + 0.3 * neg_days + 0.2 * vix_normalized
            market_breadth = 1.0 - neg_days * 0.5
        else:
            put_call_proxy = 0.7
            market_breadth = 0.5

        return {
            "fear_greed_index": max(0, min(1, 1.0 - vix_normalized)),
            "vix_level": vix_normalized,
            "put_call_ratio": put_call_proxy,
            "consumer_confidence": 0.55 if vix < 20 else 0.4 if vix < 30 else 0.25,
            "social_media_sentiment": 0.5 - 0.1 * vix_normalized,
            "news_sentiment": 0.5 - 0.15 * vix_normalized,
            "market_breadth": market_breadth,
            "institutional_flow": 0.5 - 0.1 * vix_normalized,
            "retail_flow": 0.5 + 0.1 * vix_normalized,
            "short_interest": 0.15 + 0.2 * vix_normalized,
            "safe_haven_demand": 0.2 + 0.4 * vix_normalized,
            "credit_spread": 0.15 + 0.15 * vix_normalized,
        }

    def get_contagion_state(self) -> dict:
        snapshot = self.market.get_economic_snapshot()
        vix = snapshot.get("vix", 20.0)
        vix_norm = min(max(vix / 40.0, 0.0), 1.0)

        sp500_returns = self.market.compute_returns("sp500", period="6mo")
        if sp500_returns:
            vol = sp500_returns["annualized_volatility"]
            max_dd = abs(sp500_returns["max_drawdown"])
        else:
            vol = 0.18
            max_dd = 0.10

        systemic_risk = min(1.0, vix_norm * 0.6 + vol * 1.5 + max_dd * 2.0)

        return {
            "financial_systemic_risk": systemic_risk,
            "banking_contagion": systemic_risk * 0.8,
            "sovereign_contagion": systemic_risk * 0.7,
            "trade_linkage": 0.5,
            "capital_flow_volatility": 0.3 + 0.4 * vol,
            "currency_stress": 0.2 + 0.3 * vix_norm,
            "supply_chain_coupling": 0.4,
            "info_propagation_rate": 0.6 + 0.2 * vix_norm,
        }

    def get_geopolitical_state(self) -> dict:
        snapshot = self.market.get_economic_snapshot()
        oil_price = snapshot.get("oil", 75.0)
        gold_price = snapshot.get("gold", 2000.0)
        usd_index = snapshot.get("usd_index", 100.0)

        oil_stress = min(1.0, max(0.0, (oil_price - 60) / 80)) if oil_price > 60 else 0.2
        gold_stress = min(1.0, max(0.0, (gold_price - 1800) / 600)) if gold_price > 1800 else 0.2
        dollar_dominance = 0.5 + 0.3 * max(0, min(1, (usd_index - 95) / 20))

        return {
            "us_china_tension": min(1.0, 0.4 + oil_stress * 0.2),
            "russia_west_tension": min(1.0, 0.6 + gold_stress * 0.1),
            "middle_east_instability": min(1.0, 0.4 + oil_stress * 0.3),
            "trade_protectionism": min(1.0, 0.3 + dollar_dominance * 0.2),
            "nuclear_proliferation_risk": 0.15,
            "cyber_warfare_likelihood": 0.35,
            "energy_geopolitics": min(1.0, 0.4 + oil_stress * 0.4),
            "democratic_backsliding": 0.3,
            "refugee_crisis_level": 0.35,
            "supply_chain_fragility": min(1.0, 0.35 + oil_stress * 0.2),
            "sanctions_intensity": min(1.0, 0.3 + gold_stress * 0.15),
            "alliance_stability": max(0.2, 0.6 - dollar_dominance * 0.15),
        }

    def get_climate_state(self) -> dict:
        return {
            "global_temp_anomaly_c": 1.45,
            "co2_ppm": 425.0,
            "sea_level_rise_mm": 3.7,
            "arctic_ice_extent_mkm": 4.2,
            "renewable_share": 0.22,
            "carbon_price_usd": 55.0,
            "methane_ppb": 1920.0,
            "deforestation_rate": 0.28,
            "ocean_ph": 8.07,
            "extreme_weather_index": 0.47,
            "net_zero_pledge_coverage": 0.70,
            "fossil_fuel_subsidies_b": 700.0,
        }

    def get_demographics_state(self) -> dict:
        return {
            "total_population_m": 8200.0,
            "birth_rate_per_1000": 18.0,
            "death_rate_per_1000": 7.8,
            "net_migration_per_1000": 3.0,
            "median_age": 30.5,
            "dependency_ratio": 0.54,
            "urbanization_rate": 0.57,
            "female_labor_force_pct": 0.47,
            "gdp_per_capita": 12500.0,
            "education_index": 0.65,
            "health_expenditure_pct": 0.10,
            "fertility_rate": 2.3,
            "life_expectancy": 73.0,
            "elderly_pct": 0.10,
            "youth_pct": 0.26,
        }

    def get_all_domain_states(self) -> dict:
        result = {
            "economic": self.econ.get_economic_state_approx(),
            "sentiment": self.get_sentiment_state(),
            "contagion": self.get_contagion_state(),
            "geopolitical": self.get_geopolitical_state(),
            "climate": self.get_climate_state(),
            "demographics": self.get_demographics_state(),
            "data_sources": {
                "market_data": "yfinance (real-time)",
                "economic": "FRED API (when key available) + market-based estimates",
                "sentiment": "VIX-derived + market-based estimates",
                "contagion": "Market volatility-derived estimates",
                "geopolitical": "Baseline estimates (no real-time source)",
                "climate": "Baseline estimates (NOAA API available with key)",
                "demographics": "Baseline estimates (World Bank available)",
            },
            "timestamp": datetime.now().isoformat(),
        }
        return result