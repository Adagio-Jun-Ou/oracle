from datetime import date
from typing import Optional, Protocol
from oracle.core.types import TimeSeries, DataSource, Domain, DataPoint, TimeInterval
from oracle.ingestion.fetcher import DataSourceProtocol


class BaseSource:
    def __init__(self, base_url: str = "", api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key


class FREDSource(BaseSource, DataSourceProtocol):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("https://api.stlouisfed.org/fred", api_key)
        self.series_map = {
            "gdp": "GDP",
            "unemployment": "UNRATE",
            "inflation_cpi": "CPIAUCSL",
            "federal_funds_rate": "FEDFUNDS",
            "m2_money_supply": "M2SL",
            "treasury_10y": "DGS10",
            "treasury_2y": "DGS2",
            "consumer_confidence": "UMCSENT",
            "housing_starts": "HOUST",
            "industrial_production": "INDPRO",
            "retail_sales": "RSAFS",
            "debt_to_gdp": "GFDEBTN",
            "corporate_profits": "CP",
            "snp_500": "SP500",
            "vix": "VIXCLS",
        }

    async def fetch(self, variable: str, start: date, end: date) -> TimeSeries:
        series_id = self.series_map.get(variable, variable.upper())
        return TimeSeries(
            variable=variable,
            domain=Domain.ECONOMICS,
            source=DataSource.FRED,
            data=[],
            frequency=TimeInterval.MONTHLY,
        )


class YahooFinanceSource(BaseSource, DataSourceProtocol):
    def __init__(self):
        super().__init__("https://query1.finance.yahoo.com/v8/finance/chart")
        self.tickers = {
            "sp500": "^GSPC",
            "nasdaq": "^IXIC",
            "dow_jones": "^DJI",
            "russell_2000": "^RUT",
            "vix": "^VIX",
            "gold": "GC=F",
            "oil": "CL=F",
            "bitcoin_btc": "BTC-USD",
            "usd_index": "DX-Y.NYB",
            "treasury_yield_10y": "^TNX",
        }

    async def fetch(self, variable: str, start: date, end: date) -> TimeSeries:
        ticker = self.tickers.get(variable, variable)
        return TimeSeries(
            variable=variable,
            domain=Domain.FINANCE,
            source=DataSource.YAHOO_FINANCE,
            data=[],
            frequency=TimeInterval.DAILY,
        )


class WorldBankSource(BaseSource, DataSourceProtocol):
    def __init__(self):
        super().__init__("https://api.worldbank.org/v2")
        self.indicators = {
            "population": "SP.POP.TOTL",
            "gdp_per_capita": "NY.GDP.PCAP.CD",
            "life_expectancy": "SP.DYN.LE00.IN",
            "gini_index": "SI.POV.GINI",
            "co2_emissions": "EN.ATM.CO2E.KT",
            "trade_percent_gdp": "NE.TRD.GNFS.ZS",
            "foreign_direct_investment": "BX.KLT.DINV.WD.GD.ZS",
        }

    async def fetch(self, variable: str, start: date, end: date) -> TimeSeries:
        indicator = self.indicators.get(variable, variable)
        return TimeSeries(
            variable=variable,
            domain=Domain.ECONOMICS,
            source=DataSource.WORLD_BANK,
            data=[],
            frequency=TimeInterval.YEARLY,
        )


class NOAASource(BaseSource, DataSourceProtocol):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("https://www.ncdc.noaa.gov/cdo-web/api/v2", api_key)

    async def fetch(self, variable: str, start: date, end: date) -> TimeSeries:
        return TimeSeries(
            variable=variable,
            domain=Domain.CLIMATE,
            source=DataSource.NOAA,
            data=[],
            frequency=TimeInterval.MONTHLY,
        )


class WHOSource(BaseSource, DataSourceProtocol):
    def __init__(self):
        super().__init__("https://ghoapi.azureedge.net/api")

    async def fetch(self, variable: str, start: date, end: date) -> TimeSeries:
        return TimeSeries(
            variable=variable,
            domain=Domain.HEALTH,
            source=DataSource.WHO,
            data=[],
            frequency=TimeInterval.YEARLY,
        )


class GenericAPISource(BaseSource, DataSourceProtocol):
    def __init__(self, base_url: str, source: DataSource = DataSource.CUSTOM, domain: Domain = Domain.ALL):
        super().__init__(base_url)
        self._source = source
        self._domain = domain

    async def fetch(self, variable: str, start: date, end: date) -> TimeSeries:
        return TimeSeries(
            variable=variable,
            domain=self._domain,
            source=self._source,
            data=[],
            frequency=TimeInterval.DAILY,
        )