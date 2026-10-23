from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, Protocol
import asyncio
import numpy as np
import pandas as pd
from oracle.core.config import OracleConfig
from oracle.core.types import (
    DataSource, Domain, DataPoint, TimeSeries, TimeInterval,
)


class DataSourceProtocol(Protocol):
    async def fetch(self, variable: str, start: date, end: date) -> TimeSeries: ...


@dataclass
class IngestionRequest:
    variable: str
    source: DataSource
    domain: Domain
    start_date: date
    end_date: date = field(default_factory=date.today)
    frequency: TimeInterval = TimeInterval.MONTHLY


class DataFetcher:
    def __init__(self, config: Optional[OracleConfig] = None):
        self.config = config or OracleConfig()
        self._cache: dict[str, TimeSeries] = {}
        self._sources: dict[DataSource, DataSourceProtocol] = {}

    def register_source(self, source: DataSource, handler: DataSourceProtocol):
        self._sources[source] = handler

    async def fetch(self, request: IngestionRequest) -> TimeSeries:
        cache_key = f"{request.source.value}:{request.variable}:{request.start_date}:{request.end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if request.source in self._sources:
            series = await self._sources[request.source].fetch(
                request.variable, request.start_date, request.end_date
            )
        else:
            series = self._generate_synthetic(request)

        self._cache[cache_key] = series
        return series

    async def fetch_batch(self, requests: list[IngestionRequest]) -> list[TimeSeries]:
        tasks = [self.fetch(r) for r in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _generate_synthetic(self, request: IngestionRequest) -> TimeSeries:
        np.random.seed(hash(request.variable) % (2**31))
        n_days = (request.end_date - request.start_date).days
        freq_map = {
            TimeInterval.DAILY: 1,
            TimeInterval.WEEKLY: 7,
            TimeInterval.MONTHLY: 30,
            TimeInterval.QUARTERLY: 90,
            TimeInterval.YEARLY: 365,
        }
        freq = freq_map.get(request.frequency, 30)
        n_points = max(n_days // freq, 10)

        trend = np.linspace(0, np.random.uniform(0.5, 3.0), n_points)
        cycle_amp = np.random.uniform(0.05, 0.3)
        cycle_period = np.random.uniform(20, 80)
        cycle = cycle_amp * np.sin(2 * np.pi * np.arange(n_points) / cycle_period)
        noise = np.random.normal(0, 0.02, n_points)
        regime_shift = np.zeros(n_points)
        if n_points > 50:
            shift_point = np.random.randint(n_points // 3, 2 * n_points // 3)
            regime_shift[shift_point:] = np.random.uniform(-0.3, 0.3)

        values = np.exp(trend + cycle + noise + regime_shift)

        timestamps = [
            datetime.combine(request.start_date + timedelta(days=i * freq), datetime.min.time())
            for i in range(n_points)
        ]

        data_points = [
            DataPoint(
                timestamp=ts,
                value=float(v),
                source=request.source,
                domain=request.domain,
                variable=request.variable,
            )
            for ts, v in zip(timestamps, values)
        ]

        return TimeSeries(
            variable=request.variable,
            domain=request.domain,
            source=request.source,
            data=data_points,
            frequency=request.frequency,
        )