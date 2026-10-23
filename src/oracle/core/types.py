from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum, auto
from typing import Optional, Any
import numpy as np


class Domain(Enum):
    ECONOMICS = "economics"
    FINANCE = "finance"
    GEOPOLITICS = "geopolitics"
    TECHNOLOGY = "technology"
    CLIMATE = "climate"
    DEMOGRAPHICS = "demographics"
    HEALTH = "health"
    EPIDEMIOLOGY = "epidemiology"
    ENERGY = "energy"
    COMMODITIES = "commodities"
    SOCIAL = "social"
    INFRASTRUCTURE = "infrastructure"
    SUPPLY_CHAIN = "supply_chain"
    CYBER = "cyber"
    WATER = "water"
    FOOD = "food"
    POLITICAL = "political"
    BIODIVERSITY = "biodiversity"
    PANDEMIC_PREPAREDNESS = "pandemic_preparedness"
    SPACE = "space"
    MENTAL_HEALTH = "mental_health"
    HOUSING = "housing"
    SEISMIC = "seismic"
    SOLAR = "solar"
    OCEANIC = "oceanic"
    NEAR_EARTH_OBJECT = "near_earth_object"
    ATMOSPHERIC = "atmospheric"
    ECOLOGICAL = "ecological"
    PALEOCLIMATOLOGY = "paleoclimatology"
    COSMIC_RADIATION = "cosmic_radiation"
    GRAVITATIONAL = "gravitational"
    COMPOUND_HAZARD = "compound_hazard"
    WORLD_STATE = "world_state"
    PRESCIENCE = "prescience"
    FATE = "fate"
    PERSONAL_LIFE = "personal_life"
    EVOLUTIONARY = "evolutionary"
    LEARNING = "learning"
    PHYSICAL_CONSTANTS = "physical_constants"
    AI_INTEGRATION = "ai_integration"
    ESG = "esg"
    YIELD_CURVE = "yield_curve"
    RISK_DECOMPOSITION = "risk_decomposition"
    VALUATION = "valuation"
    PORTFOLIO = "portfolio"
    OPTIONS = "options"
    TECHNICAL_ANALYSIS = "technical_analysis"
    BACKTESTING = "backtesting"
    SENTIMENT = "sentiment"
    CONTAGION = "contagion"
    CALIBRATION = "calibration"
    NARRATIVE = "narrative"
    STRESS_TESTING = "stress_testing"
    ALL = "all"


class DataSource(Enum):
    FRED = "fred"
    WORLD_BANK = "world_bank"
    YAHOO_FINANCE = "yahoo_finance"
    WIKIPEDIA = "wikipedia"
    OPENALEX = "openalex"
    NOAA = "noaa"
    WHO = "who"
    CIA_FACTBOOK = "cia_factbook"
    GISTEMP = "gistemp"
    CUSTOM = "custom"


class TimeInterval(Enum):
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"
    QUARTERLY = "3M"
    YEARLY = "1Y"
    DECADE = "10Y"
    CENTURY = "100Y"


class ConfidenceLevel(Enum):
    LOW = 0.68
    MEDIUM = 0.90
    HIGH = 0.95
    VERY_HIGH = 0.99


class EconomicCycle(Enum):
    EXPANSION = "expansion"
    PEAK = "peak"
    RECESSION = "recession"
    TROUGH = "trough"
    RECOVERY = "recovery"


class SignalType(Enum):
    LEADING = "leading"
    COINCIDENT = "coincident"
    LAGGING = "lagging"


@dataclass
class Signal:
    name: str
    domain: Domain
    signal_type: SignalType
    value: float
    timestamp: datetime
    source: DataSource
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)
    decay_weight: float = 1.0

    def decay(self, current_time: datetime, rate: float = 0.05) -> float:
        dt = (current_time - self.timestamp).total_seconds() / 86400.0
        self.decay_weight = np.exp(-rate * dt)
        return self.decay_weight


@dataclass
class Correlation:
    source_a: str
    source_b: str
    domain_a: Domain
    domain_b: Domain
    correlation: float
    lag_days: float = 0.0
    p_value: float = 1.0
    strength: str = "weak"
    direction: str = "positive"
    discovered_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.correlation = float(np.clip(self.correlation, -1, 1))
        self.direction = "positive" if self.correlation >= 0 else "negative"
        abs_corr = abs(self.correlation)
        if abs_corr >= 0.7:
            self.strength = "strong"
        elif abs_corr >= 0.4:
            self.strength = "moderate"
        else:
            self.strength = "weak"


@dataclass
class Prediction:
    domain: Domain
    variable: str
    value: float
    confidence_interval: tuple[float, float]
    confidence_level: ConfidenceLevel
    timestamp: datetime
    target_date: date
    model_name: str
    weights: dict = field(default_factory=dict)
    signals_used: list[str] = field(default_factory=list)
    correlations_used: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    methodology: str = ""
    uncertainty_type: str = "frequentist"
    data_sources: list[str] = field(default_factory=list)
    reliability_horizon_days: float = 365.0

    @property
    def ci_width(self) -> float:
        return self.confidence_interval[1] - self.confidence_interval[0]

    @property
    def ci_midpoint(self) -> float:
        return (self.confidence_interval[0] + self.confidence_interval[1]) / 2


@dataclass
class Forecast:
    predictions: list[Prediction]
    generated_at: datetime = field(default_factory=datetime.now)
    time_range: tuple[date, date] = (date.today(), date.today())
    domains: list[Domain] = field(default_factory=list)
    model_versions: dict = field(default_factory=dict)
    ensemble_weights_used: dict = field(default_factory=dict)
    error_log: list[str] = field(default_factory=list)
    failed_domains: list[str] = field(default_factory=list)
    computation_time_ms: float = 0.0

    @property
    def start_date(self) -> date:
        return self.time_range[0]

    @property
    def end_date(self) -> date:
        return self.time_range[1]

    def get_predictions_for_date(self, d: date) -> list[Prediction]:
        return [p for p in self.predictions if p.target_date == d]

    def get_predictions_for_domain(self, domain: Domain) -> list[Prediction]:
        return [p for p in self.predictions if p.domain == domain]


@dataclass
class DataPoint:
    timestamp: datetime
    value: float
    source: DataSource
    domain: Domain
    variable: str
    metadata: dict = field(default_factory=dict)


@dataclass
class TimeSeries:
    variable: str
    domain: Domain
    source: DataSource
    data: list[DataPoint] = field(default_factory=list)
    frequency: TimeInterval = TimeInterval.MONTHLY

    def to_array(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.data:
            return np.array([]), np.array([])
        sorted_data = sorted(self.data, key=lambda d: d.timestamp)
        timestamps = np.array([d.timestamp.timestamp() for d in sorted_data])
        values = np.array([d.value for d in sorted_data])
        return timestamps, values

    @property
    def mean(self) -> float:
        if not self.data:
            return 0.0
        return float(np.mean([d.value for d in self.data]))

    @property
    def std(self) -> float:
        if not self.data:
            return 0.0
        return float(np.std([d.value for d in self.data]))


@dataclass
class CorrelationMatrix:
    variables: list[str]
    matrix: np.ndarray
    p_values: np.ndarray = field(default_factory=lambda: np.ones((1, 1)))
    lag_matrix: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))

    def get_strong_correlations(self, threshold: float = 0.5) -> list[Correlation]:
        results = []
        n = len(self.variables)
        for i in range(n):
            for j in range(i + 1, n):
                corr = self.matrix[i, j]
                if abs(corr) >= threshold:
                    results.append(Correlation(
                        source_a=self.variables[i],
                        source_b=self.variables[j],
                        domain_a=Domain.ALL,
                        domain_b=Domain.ALL,
                        correlation=corr,
                        lag_days=self.lag_matrix[i, j] if self.lag_matrix.shape == (n, n) else 0.0,
                        p_value=self.p_values[i, j] if self.p_values.shape == (n, n) else 1.0,
                    ))
        return results