"""Continuous data ingestion and outcome tracking for the evolution engine.

Provides:
- DomainStream: streams of domain-specific data points
- OutcomeTracker integration with DailyOracle real data
- Backtesting windows for fitness evaluation
- DataFlowEngine: orchestrates data flow between sources and the evolution loop
"""
from __future__ import annotations

import time
import threading
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Domain(Enum):
    ECONOMIC = "economic"
    GEOPOLITICAL = "geopolitical"
    HEALTH = "health"
    TECHNOLOGY = "technology"
    CLIMATE = "climate"
    SOCIAL = "social"
    MONETARY = "monetary"
    INFLATION = "inflation"
    LABOR = "labor"
    CURRENCY = "currency"


@dataclass
class DataPoint:
    timestamp: float
    domain: str
    metric: str
    value: float
    source: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeRecord:
    timestamp: float
    domain: str
    predicted_probability: float
    actual_outcome: float
    confidence: float = 0.5
    genome_id: str = ""
    horizon_hours: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def brier_score(self) -> float:
        return (self.predicted_probability - self.actual_outcome) ** 2

    @property
    def is_correct_direction(self) -> bool:
        return (self.predicted_probability > 0.5) == (self.actual_outcome > 0.5)


@dataclass
class StreamSubscription:
    domain: str
    callback: Callable[[DataPoint], None]
    min_confidence: float = 0.0
    filter_metrics: Optional[List[str]] = None


class DomainStream:
    """Stream of data points for a specific domain."""

    def __init__(self, domain: str, max_buffer: int = 10000):
        self.domain = domain
        self.max_buffer = max_buffer
        self._buffer: deque[DataPoint] = deque(maxlen=max_buffer)
        self._subscribers: List[StreamSubscription] = []
        self._stats: Dict[str, Any] = {
            "total_received": 0, "total_processed": 0,
            "last_update": 0.0, "avg_value": 0.0
        }

    def push(self, point: DataPoint) -> int:
        self._buffer.append(point)
        self._stats["total_received"] += 1
        self._stats["last_update"] = time.time()
        self._stats["avg_value"] = (
            self._stats["avg_value"] * (self._stats["total_received"] - 1) + point.value
        ) / self._stats["total_received"]
        notified = 0
        for sub in self._subscribers:
            if sub.domain != self.domain and sub.domain != "*":
                continue
            if point.confidence < sub.min_confidence:
                continue
            if sub.filter_metrics and point.metric not in sub.filter_metrics:
                continue
            try:
                sub.callback(point)
                notified += 1
            except Exception as e:
                logger.debug(f"Subscriber callback error: {e}")
        self._stats["total_processed"] += 1
        return notified

    def subscribe(self, subscription: StreamSubscription) -> None:
        self._subscribers.append(subscription)

    def get_recent(self, n: int = 100, metric: Optional[str] = None) -> List[DataPoint]:
        points = list(self._buffer)
        if metric:
            points = [p for p in points if p.metric == metric]
        return points[-n:]

    def get_values(self, n: int = 100, metric: Optional[str] = None) -> np.ndarray:
        points = self.get_recent(n, metric)
        if not points:
            return np.array([])
        return np.array([p.value for p in points])

    def get_state(self) -> float:
        recent = self.get_recent(10)
        if not recent:
            return 0.5
        return float(np.mean([p.value for p in recent]))


class OutcomeTracker:
    """Tracks prediction outcomes for fitness evaluation."""

    def __init__(self, max_history: int = 50000, resolution_window: float = 168.0,
                 max_pending: int = 10000):
        self.max_history = max_history
        self.max_pending = max_pending
        self.resolution_window = resolution_window
        self._records: List[OutcomeRecord] = []
        self._pending: Dict[str, OutcomeRecord] = {}
        self._by_domain: Dict[str, List[OutcomeRecord]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_prediction(self, genome_id: str, domain: str,
                           predicted: float, horizon_hours: float = 24.0,
                           confidence: float = 0.5,
                           metadata: Optional[Dict] = None) -> str:
        prediction_id = f"{genome_id}_{domain}_{int(time.time())}"
        record = OutcomeRecord(
            timestamp=time.time(), domain=domain,
            predicted_probability=np.clip(predicted, 0.01, 0.99),
            actual_outcome=-1.0,
            confidence=confidence,
            genome_id=genome_id,
            horizon_hours=horizon_hours,
            metadata=metadata or {}
        )
        with self._lock:
            if len(self._pending) >= self.max_pending:
                oldest_key = next(iter(self._pending))
                del self._pending[oldest_key]
            self._pending[prediction_id] = record
        return prediction_id

    def resolve_prediction(self, prediction_id: str, actual_outcome: float) -> Optional[OutcomeRecord]:
        with self._lock:
            record = self._pending.pop(prediction_id, None)
        if record is None:
            return None
        record.actual_outcome = np.clip(actual_outcome, 0.0, 1.0)
        with self._lock:
            self._records.append(record)
            self._by_domain[record.domain].append(record)
            if len(self._records) > self.max_history:
                oldest = self._records.pop(0)
                if oldest.domain in self._by_domain:
                    domain_list = self._by_domain[oldest.domain]
                    if domain_list and domain_list[0].timestamp == oldest.timestamp:
                        domain_list.pop(0)
        return record

    def resolve_all_pending(self, domain_states: Dict[str, float]) -> int:
        resolved = 0
        with self._lock:
            pending_ids = list(self._pending.keys())
        for pid in pending_ids:
            record = self._pending.get(pid)
            if record is None:
                continue
            domain = record.domain
            actual = domain_states.get(domain, 0.5)
            if record.horizon_hours > 0:
                elapsed = (time.time() - record.timestamp) / 3600
                if elapsed < record.horizon_hours * 0.5:
                    continue
            self.resolve_prediction(pid, actual)
            resolved += 1
        return resolved

    def get_brier_score(self, domain: Optional[str] = None, last_n: int = 500) -> float:
        records = self.get_resolved(domain, last_n)
        if not records:
            return 0.25
        return float(np.mean([r.brier_score for r in records]))

    def get_direction_accuracy(self, domain: Optional[str] = None, last_n: int = 200) -> float:
        records = self.get_resolved(domain, last_n)
        if len(records) < 2:
            return 0.5
        correct = sum(1 for r in records if r.is_correct_direction)
        return correct / len(records)

    def get_calibration(self, domain: Optional[str] = None, last_n: int = 500) -> Dict[str, float]:
        records = self.get_resolved(domain, last_n)
        if len(records) < 10:
            return {"slope": 1.0, "intercept": 0.0, "rmse": 0.5}
        preds = np.array([r.predicted_probability for r in records])
        actuals = np.array([r.actual_outcome for r in records])
        if np.std(preds) < 1e-10:
            return {"slope": 0.0, "intercept": float(np.mean(actuals)), "rmse": float(np.sqrt(np.mean((preds - actuals) ** 2)))}
        slope = float(np.corrcoef(preds, actuals)[0, 1]) * np.std(actuals) / max(np.std(preds), 1e-10)
        intercept = float(np.mean(actuals) - slope * np.mean(preds))
        rmse = float(np.sqrt(np.mean((preds * slope + intercept - actuals) ** 2)))
        return {"slope": slope, "intercept": intercept, "rmse": rmse}

    def get_resolved(self, domain: Optional[str] = None, last_n: int = 500) -> List[OutcomeRecord]:
        with self._lock:
            if domain:
                records = self._by_domain.get(domain, [])
            else:
                records = self._records
            resolved = [r for r in records if r.actual_outcome >= 0]
            return resolved[-last_n:]

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._records)
            pending = len(self._pending)
            domains = list(self._by_domain.keys())
        return {
            "total_resolved": total,
            "pending_predictions": pending,
            "domains": domains,
            "brier_score": round(self.get_brier_score(), 4),
            "direction_accuracy": round(self.get_direction_accuracy(), 4)
        }


class DataFlowEngine:
    """Orchestrates continuous data flow into the evolution engine."""

    def __init__(self, domains: Optional[List[str]] = None, max_buffer: int = 10000):
        self.domains = domains or [d.value for d in Domain]
        self.max_buffer = max_buffer
        self._streams: Dict[str, DomainStream] = {
            d: DomainStream(d, max_buffer) for d in self.domains
        }
        self.outcome_tracker = OutcomeTracker()
        self._running = False
        self._last_data_time: float = 0.0
        self._data_hooks: List[Callable] = []
        self._domain_cache: Dict[str, float] = {}

    def register_hook(self, hook: Callable[[Dict[str, float]], None]) -> None:
        self._data_hooks.append(hook)

    def ingest_data(self, domain: str, metric: str, value: float,
                    source: str = "", confidence: float = 1.0,
                    metadata: Optional[Dict] = None) -> None:
        point = DataPoint(
            timestamp=time.time(), domain=domain,
            metric=metric, value=value,
            source=source, confidence=confidence,
            metadata=metadata or {}
        )
        stream = self._streams.get(domain)
        if stream:
            stream.push(point)
        self._domain_cache[domain] = value
        self._last_data_time = time.time()
        for hook in self._data_hooks:
            try:
                hook(self._domain_cache)
            except Exception as e:
                logger.debug(f"Data hook error: {e}")

    def ingest_domain_states(self, states: Dict[str, float],
                             source: str = "daily_oracle") -> None:
        for domain, value in states.items():
            self.ingest_data(domain, "state", value, source=source)
        resolved = self.outcome_tracker.resolve_all_pending(states)

    def get_domain_states(self) -> Dict[str, float]:
        states = {}
        for domain, stream in self._streams.items():
            state = stream.get_state()
            if state != 0.5 or len(stream._buffer) > 0:
                states[domain] = state
        states.update(self._domain_cache)
        return states

    def get_stream(self, domain: str) -> Optional[DomainStream]:
        return self._streams.get(domain)

    def get_recent_values(self, domain: str, n: int = 100) -> np.ndarray:
        stream = self._streams.get(domain)
        if stream:
            return stream.get_values(n)
        return np.array([])

    def summary(self) -> Dict[str, Any]:
        return {
            "domains": self.domains,
            "last_data_time": self._last_data_time,
            "total_data_points": sum(len(s._buffer) for s in self._streams.values()),
            "domain_states": self.get_domain_states(),
            "outcome_tracker": self.outcome_tracker.summary()
        }