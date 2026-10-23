"""Unified Regime Assessment — Regime Detection v2.

Combines TSForecast's regime flags, DailyOracle's regime classification,
and AnomalyCorrelator's risk levels into a single composite regime view.

Key features:
  - Multi-source regime fusion (TS forecast + daily oracle + anomaly)
  - Regime probability distribution (bull/bear/transition/uncertain)
  - Regime stability/transience scoring
  - Regime shift early warning (divergence between sources)
  - Historical regime transition matrix
  - Composite confidence score
  - Narrative regime description with evidence
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import date, datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    BULL = "bull"
    BEAR = "bear"
    TRANSITION = "transition"
    UNCERTAIN = "uncertain"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class RegimeSource(Enum):
    TS_FORECAST = "ts_forecast"
    DAILY_ORACLE = "daily_oracle"
    ANOMALY_CORRELATOR = "anomaly_correlator"
    MARKET_DATA = "market_data"
    MACRO_DATA = "macro_data"


class RegimeStability(Enum):
    STABLE = "stable"
    MODERATE = "moderate"
    TRANSIENT = "transient"
    VOLATILE = "volatile"


@dataclass
class SourceRegimeSignal:
    source: RegimeSource
    regime: RegimeType
    confidence: float
    score: float
    details: Optional[Dict[str, float]] = None


@dataclass
class RegimeTransition:
    from_regime: RegimeType
    to_regime: RegimeType
    probability: float
    historical_frequency: float
    days_typical: int


@dataclass
class RegimeAssessment:
    composite_regime: RegimeType
    regime_probabilities: Dict[str, float]
    regime_score: float
    confidence: float
    stability: RegimeStability
    stability_score: float
    source_signals: List[SourceRegimeSignal]
    source_divergence: float
    regime_shift_risk: float
    shift_direction: Optional[str]
    shift_evidence: List[str]
    transitions: List[RegimeTransition]
    narrative: str
    timestamp: str


# Historical transition probabilities (bull → bear, bear → recovery, etc.)
REGIME_TRANSITION_MATRIX = {
    (RegimeType.BULL, RegimeType.BULL): 0.75,
    (RegimeType.BULL, RegimeType.TRANSITION): 0.15,
    (RegimeType.BULL, RegimeType.BEAR): 0.08,
    (RegimeType.BULL, RegimeType.CRISIS): 0.02,
    (RegimeType.BEAR, RegimeType.BEAR): 0.60,
    (RegimeType.BEAR, RegimeType.TRANSITION): 0.20,
    (RegimeType.BEAR, RegimeType.RECOVERY): 0.15,
    (RegimeType.BEAR, RegimeType.CRISIS): 0.05,
    (RegimeType.TRANSITION, RegimeType.BULL): 0.35,
    (RegimeType.TRANSITION, RegimeType.BEAR): 0.25,
    (RegimeType.TRANSITION, RegimeType.TRANSITION): 0.30,
    (RegimeType.TRANSITION, RegimeType.CRISIS): 0.10,
    (RegimeType.CRISIS, RegimeType.BEAR): 0.30,
    (RegimeType.CRISIS, RegimeType.RECOVERY): 0.50,
    (RegimeType.CRISIS, RegimeType.TRANSITION): 0.15,
    (RegimeType.CRISIS, RegimeType.CRISIS): 0.05,
    (RegimeType.RECOVERY, RegimeType.BULL): 0.55,
    (RegimeType.RECOVERY, RegimeType.RECOVERY): 0.30,
    (RegimeType.RECOVERY, RegimeType.TRANSITION): 0.10,
    (RegimeType.RECOVERY, RegimeType.BEAR): 0.05,
}

# Regime scoring guidelines
REGIME_SCORE_MAP = {
    RegimeType.BULL: lambda s: 0.5 + 0.5 * s,
    RegimeType.BEAR: lambda s: 0.5 - 0.5 * s,
    RegimeType.TRANSITION: lambda s: 0.5 + 0.2 * (s - 0.5),
    RegimeType.CRISIS: lambda s: max(0, 0.3 - 0.3 * s),
    RegimeType.RECOVERY: lambda s: 0.5 + 0.4 * s,
    RegimeType.UNCERTAIN: lambda s: 0.5,
}


class UnifiedRegimeAssessor:
    """Unified regime assessment combining multiple sources."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._regime_history: List[Tuple[str, RegimeType, float]] = []

    def assess(self, ts_forecast_regime: Optional[str] = None,
               ts_forecast_score: Optional[float] = None,
               daily_oracle_regime: Optional[str] = None,
               daily_oracle_score: Optional[float] = None,
               anomaly_risk_level: Optional[str] = None,
               anomaly_risk_score: Optional[float] = None,
               market_trend: Optional[str] = None,
               market_score: Optional[float] = None,
               vix_level: Optional[float] = None,
               yield_spread: Optional[float] = None,
               ) -> RegimeAssessment:
        source_signals = []
        if ts_forecast_regime:
            r = self._parse_regime(ts_forecast_regime)
            conf = abs(ts_forecast_score or 0.5)
            source_signals.append(SourceRegimeSignal(
                source=RegimeSource.TS_FORECAST, regime=r,
                confidence=min(1.0, abs(conf)), score=conf))
        if daily_oracle_regime:
            r = self._parse_regime(daily_oracle_regime)
            conf = abs(daily_oracle_score or 0.5)
            source_signals.append(SourceRegimeSignal(
                source=RegimeSource.DAILY_ORACLE, regime=r,
                confidence=min(1.0, abs(conf)), score=conf))
        if anomaly_risk_level:
            r = self._risk_to_regime(anomaly_risk_level)
            conf = abs(anomaly_risk_score or 0.5)
            source_signals.append(SourceRegimeSignal(
                source=RegimeSource.ANOMALY_CORRELATOR, regime=r,
                confidence=min(1.0, abs(conf)), score=conf))
        if market_trend:
            r = self._parse_regime(market_trend)
            conf = abs(market_score or 0.5)
            source_signals.append(SourceRegimeSignal(
                source=RegimeSource.MARKET_DATA, regime=r,
                confidence=min(1.0, abs(conf)), score=conf))
        composite, probabilities = self._compute_composite(source_signals)
        score = self._compute_score(composite, probabilities)
        confidence = self._compute_confidence(source_signals, probabilities)
        stability = self._compute_stability(source_signals)
        stability_score = self._stability_score(stability)
        divergence = self._compute_divergence(source_signals)
        shift_risk, shift_dir, shift_evidence = self._detect_shift_risk(
            source_signals, vix_level, yield_spread)
        transitions = self._compute_transitions(composite)
        narrative = self._generate_narrative(
            composite, confidence, stability, divergence,
            shift_risk, shift_dir, shift_evidence,
            len(source_signals))
        return RegimeAssessment(
            composite_regime=composite,
            regime_probabilities=probabilities,
            regime_score=round(score, 4),
            confidence=round(confidence, 4),
            stability=stability,
            stability_score=round(stability_score, 4),
            source_signals=source_signals,
            source_divergence=round(divergence, 4),
            regime_shift_risk=round(shift_risk, 4),
            shift_direction=shift_dir,
            shift_evidence=shift_evidence[:5],
            transitions=transitions,
            narrative=narrative,
            timestamp=datetime.now().isoformat(),
        )

    def _parse_regime(self, s: str) -> RegimeType:
        s_lower = s.lower().strip()
        for r in RegimeType:
            if r.value in s_lower or r.name.lower() in s_lower:
                return r
        return RegimeType.UNCERTAIN

    def _risk_to_regime(self, risk_level: str) -> RegimeType:
        mapping = {
            "minimal": RegimeType.BULL,
            "low": RegimeType.BULL,
            "moderate": RegimeType.TRANSITION,
            "high": RegimeType.BEAR,
            "critical": RegimeType.CRISIS,
        }
        return mapping.get(risk_level.lower(), RegimeType.UNCERTAIN)

    def _compute_composite(self, signals: List[SourceRegimeSignal]) -> Tuple[RegimeType, Dict[str, float]]:
        if not signals:
            return RegimeType.UNCERTAIN, {r.value: 0.0 for r in RegimeType}
        votes: Dict[str, float] = {r.value: 0.0 for r in RegimeType}
        for signal in signals:
            regime_name = signal.regime.value
            votes[regime_name] = votes.get(regime_name, 0) + signal.confidence * abs(signal.score + 0.5) / 1.5
        total = sum(votes.values())
        if total > 0:
            probabilities = {k: v / total for k, v in votes.items()}
        else:
            probabilities = {k: 0.0 for k in votes}
        best = max(probabilities, key=probabilities.get)
        return self._parse_regime(best), probabilities

    def _compute_score(self, regime: RegimeType, probabilities: Dict[str, float]) -> float:
        score_fn = REGIME_SCORE_MAP.get(regime, lambda s: 0.5)
        confidence = probabilities.get(regime.value, 0.5)
        return score_fn(confidence)

    def _compute_confidence(self, signals: List[SourceRegimeSignal],
                            probabilities: Dict[str, float]) -> float:
        if not signals:
            return 0.0
        avg_conf = np.mean([s.confidence for s in signals])
        max_prob = max(probabilities.values())
        agreement = max_prob
        return float(min(1.0, avg_conf * 0.5 + agreement * 0.5))

    def _compute_stability(self, signals: List[SourceRegimeSignal]) -> RegimeStability:
        if len(signals) < 2:
            return RegimeStability.MODERATE
        regimes = [s.regime for s in signals]
        unique = len(set(regimes))
        if unique == 1:
            return RegimeStability.STABLE
        elif unique == 2:
            return RegimeStability.MODERATE
        elif unique == 3:
            return RegimeStability.TRANSIENT
        else:
            return RegimeStability.VOLATILE

    def _stability_score(self, stability: RegimeStability) -> float:
        mapping = {
            RegimeStability.STABLE: 0.85,
            RegimeStability.MODERATE: 0.60,
            RegimeStability.TRANSIENT: 0.35,
            RegimeStability.VOLATILE: 0.15,
        }
        return mapping.get(stability, 0.5)

    def _compute_divergence(self, signals: List[SourceRegimeSignal]) -> float:
        if len(signals) < 2:
            return 0.0
        regime_values = {RegimeType.BULL: 1, RegimeType.RECOVERY: 0.8,
                         RegimeType.TRANSITION: 0.5, RegimeType.UNCERTAIN: 0.5,
                         RegimeType.BEAR: 0.2, RegimeType.CRISIS: 0.0}
        values = [regime_values.get(s.regime, 0.5) for s in signals]
        return float(np.std(values))

    def _detect_shift_risk(self, signals: List[SourceRegimeSignal],
                           vix: Optional[float] = None,
                           yield_spread: Optional[float] = None) -> Tuple[float, Optional[str], List[str]]:
        evidence = []
        if not signals:
            return 0.0, None, evidence
        regimes = [s.regime for s in signals]
        unique = list(set(regimes))
        if len(unique) <= 1:
            shift_risk = 0.1
            shift_dir = None
        elif len(unique) <= 2:
            bull_present = any(r in (RegimeType.BULL, RegimeType.RECOVERY) for r in regimes)
            bear_present = any(r in (RegimeType.BEAR, RegimeType.CRISIS) for r in regimes)
            transition_present = RegimeType.TRANSITION in regimes
            if bull_present and transition_present:
                shift_risk = 0.4
                shift_dir = "bull_to_transition"
                evidence.append("Bull regime with transition signals — potential top formation")
            elif bear_present and transition_present:
                shift_risk = 0.5
                shift_dir = "bear_to_transition"
                evidence.append("Bear regime with transition signals — potential bottom formation")
            elif bull_present and bear_present:
                shift_risk = 0.6
                shift_dir = "bull_to_bear"
                evidence.append("Conflicting bull/bear signals — high regime uncertainty")
            else:
                shift_risk = 0.3
                shift_dir = None
        else:
            shift_risk = 0.7
            shift_dir = "multiple_signals"
            evidence.append("Multiple conflicting regimes detected — elevated transition probability")
        if vix is not None and vix > 25:
            shift_risk = min(1.0, shift_risk + 0.15)
            evidence.append(f"Elevated VIX ({vix:.0f}) supports shift risk")
        if yield_spread is not None and yield_spread < 0:
            shift_risk = min(1.0, shift_risk + 0.2)
            evidence.append("Yield curve inversion supports bearish shift risk")
        return shift_risk, shift_dir, evidence

    def _compute_transitions(self, current: RegimeType) -> List[RegimeTransition]:
        transitions = []
        for target in RegimeType:
            key = (current, target)
            prob = REGIME_TRANSITION_MATRIX.get(key, 0.0)
            if prob > 0:
                transitions.append(RegimeTransition(
                    from_regime=current, to_regime=target,
                    probability=prob, historical_frequency=prob,
                    days_typical=self._typical_days(current, target)))
        return sorted(transitions, key=lambda t: t.probability, reverse=True)

    def _typical_days(self, frm: RegimeType, to: RegimeType) -> int:
        estimates = {
            (RegimeType.BULL, RegimeType.TRANSITION): 45,
            (RegimeType.BULL, RegimeType.BEAR): 90,
            (RegimeType.BEAR, RegimeType.TRANSITION): 60,
            (RegimeType.BEAR, RegimeType.RECOVERY): 120,
            (RegimeType.TRANSITION, RegimeType.BULL): 30,
            (RegimeType.TRANSITION, RegimeType.BEAR): 45,
            (RegimeType.TRANSITION, RegimeType.CRISIS): 15,
            (RegimeType.CRISIS, RegimeType.RECOVERY): 90,
            (RegimeType.CRISIS, RegimeType.BEAR): 60,
            (RegimeType.RECOVERY, RegimeType.BULL): 180,
        }
        return estimates.get((frm, to), 60)

    def _generate_narrative(self, regime: RegimeType, confidence: float,
                            stability: RegimeStability, divergence: float,
                            shift_risk: float, shift_dir: Optional[str],
                            shift_evidence: List[str],
                            n_sources: int) -> str:
        regime_names = {
            RegimeType.BULL: "Bull", RegimeType.BEAR: "Bear",
            RegimeType.TRANSITION: "Transition", RegimeType.CRISIS: "Crisis",
            RegimeType.RECOVERY: "Recovery", RegimeType.UNCERTAIN: "Uncertain",
        }
        name = regime_names.get(regime, "Unknown")
        parts = [f"Composite regime: {name} ({confidence:.0%} confidence)"]
        if stability == RegimeStability.STABLE:
            parts.append("All sources in agreement — stable regime assessment")
        elif stability == RegimeStability.MODERATE:
            parts.append("Minor divergence between sources — moderate stability")
        elif stability == RegimeStability.TRANSIENT:
            parts.append("Significant divergence detected — potentially transient regime")
        else:
            parts.append("High source divergence — regime classification volatile")
        if shift_risk > 0.5:
            dir_str = shift_dir.replace("_", " ") if shift_dir else "unknown"
            parts.append(f"Elevated shift risk ({shift_risk:.0%}) toward {dir_str}")
            if shift_evidence:
                parts.append(f"Evidence: {'; '.join(shift_evidence[:3])}")
        elif shift_risk > 0.3:
            parts.append(f"Moderate shift risk ({shift_risk:.0%})")
        else:
            parts.append(f"Low shift risk ({shift_risk:.0%})")
        parts.append(f"Assessment based on {n_sources} source(s), divergence: {divergence:.3f}")
        return " — ".join(parts)