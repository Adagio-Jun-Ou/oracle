"""Precursor Pattern Engine — Historical Precursor Sequence Matching for Early Warning.

Core Insight: Major regime changes (recessions, crises, tipping points) are preceded
by characteristic sequences of precursor events. This engine encodes historical
precursor patterns and matches current conditions against them to provide
LEAD TIME before regime changes occur.

Implements:
- PrecursorPattern: A sequence of events/signals that historically precede a regime change
- PrecursorPatternLibrary: Curated library of known precursor patterns across domains
- PrecursorMatcher: Matches current domain states against library patterns with similarity scoring
- RegimeChangeWarning: Early warning with estimated lead time and confidence
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    RECESSION = "recession"
    FINANCIAL_CRISIS = "financial_crisis"
    INFLATION_SPIKE = "inflation_spike"
    DEFLATIONARY_TRAP = "deflationary_trap"
    GEOPOLITICAL_ESCALATION = "geopolitical_escalation"
    CLIMATE_TIPPING = "climate_tipping"
    TECHNOLOGY_DISRUPTION = "technology_disruption"
    PANDEMIC_OUTBREAK = "pandemic_outbreak"
    SOCIAL_UNREST = "social_unrest"
    CURRENCY_CRISIS = "currency_crisis"
    SOVEREIGN_DEBT_CRISIS = "sovereign_debt_crisis"
    HOUSING_CRASH = "housing_crash"
    ENERGY_CRISIS = "energy_crisis"
    REGIME_CHANGE = "regime_change"


class Phase(Enum):
    NORMAL = "normal"
    EARLY_PRECURSOR = "early_precursor"
    MID_PRECURSOR = "mid_precursor"
    LATE_PRECURSOR = "late_precursor"
    IMMINENT = "imminent"
    ACTIVE = "active"


@dataclass
class PrecursorStep:
    order: int
    description: str
    domain_signals: Dict[str, float]
    typical_lead_time_months: float
    signal_direction: str = "above"
    threshold: float = 0.5


@dataclass
class PrecursorPattern:
    name: str
    target_regime: RegimeType
    steps: List[PrecursorStep]
    historical_frequency: float = 0.3
    historical_accuracy: float = 0.6
    domains: List[str] = field(default_factory=list)
    description: str = ""
    last_occurrence_year: Optional[int] = None
    n_historical_occurrences: int = 3
    false_positive_rate: float = 0.25

    def match_score(self, domain_values: Dict[str, float]) -> Tuple[float, Phase, Dict]:
        if not domain_values:
            return 0.0, Phase.NORMAL, {}
        matched_steps = 0
        step_scores = {}
        for step in self.steps:
            step_score = 0.0
            for signal_name, signal_level in step.domain_signals.items():
                val = domain_values.get(signal_name)
                if val is not None:
                    if step.signal_direction == "above":
                        ratio = val / signal_level if signal_level > 0 else 0
                        step_score = max(step_score, min(1.0, ratio))
                    elif step.signal_direction == "below":
                        ratio = signal_level / val if val > 0 else 0
                        step_score = max(step_score, min(1.0, ratio))
                    else:
                        proximity = 1.0 - min(1.0, abs(val - signal_level) / max(0.01, abs(signal_level)))
                        step_score = max(step_score, proximity)
            step_scores[step.order] = step_score
            if step_score >= step.threshold:
                matched_steps += 1
        total_steps = len(self.steps)
        overall_score = matched_steps / total_steps if total_steps > 0 else 0
        if matched_steps == 0:
            phase = Phase.NORMAL
        elif matched_steps <= total_steps * 0.33:
            phase = Phase.EARLY_PRECURSOR
        elif matched_steps <= total_steps * 0.66:
            phase = Phase.MID_PRECURSOR
        elif matched_steps < total_steps:
            phase = Phase.LATE_PRECURSOR
        else:
            phase = Phase.IMMINENT
        return overall_score, phase, step_scores

    def estimated_lead_time(self, phase: Phase) -> float:
        if phase == Phase.NORMAL:
            return float("inf")
        elif phase == Phase.EARLY_PRECURSOR:
            lead_steps = [s for s in self.steps if s.order > len(self.steps) * 0.33]
            return sum(s.typical_lead_time_months for s in lead_steps) / max(1, len(lead_steps))
        elif phase == Phase.MID_PRECURSOR:
            lead_steps = [s for s in self.steps if s.order > len(self.steps) * 0.66]
            return sum(s.typical_lead_time_months for s in lead_steps) / max(1, len(lead_steps))
        elif phase == Phase.LATE_PRECURSOR:
            last_steps = [s for s in self.steps if s.order >= len(self.steps) - 1]
            return sum(s.typical_lead_time_months for s in last_steps) / max(1, len(last_steps))
        return 0.0


@dataclass
class RegimeChangeWarning:
    pattern_name: str
    target_regime: RegimeType
    phase: Phase
    match_score: float
    confidence: float
    estimated_lead_time_months: float
    matched_signals: Dict[str, float]
    next_expected_signals: List[str]
    historical_precedent: str
    recommended_actions: List[str]
    risk_level: str = "moderate"

    def __post_init__(self):
        if self.risk_level == "moderate":
            if self.phase in (Phase.LATE_PRECURSOR, Phase.IMMINENT):
                self.risk_level = "critical"
            elif self.phase == Phase.MID_PRECURSOR:
                self.risk_level = "high"
            elif self.phase == Phase.EARLY_PRECURSOR:
                self.risk_level = "moderate"


class PrecursorPatternLibrary:
    """Curated library of known precursor patterns from historical analysis."""

    def __init__(self):
        self.patterns: List[PrecursorPattern] = []
        self._build_library()

    def _build_library(self):
        self.patterns.append(PrecursorPattern(
            name="recession_precursor",
            target_regime=RegimeType.RECESSION,
            steps=[
                PrecursorStep(1, "Yield curve inversion", {"interest_rate_10y": 0.035, "interest_rate_2y": 0.04}, 18, "above", 0.6),
                PrecursorStep(2, "Rising unemployment claims", {"unemployment": 0.05}, 12, "above", 0.5),
                PrecursorStep(3, "Declining consumer confidence", {"consumer_confidence": 85}, 8, "below", 0.5),
                PrecursorStep(4, "Manufacturing contraction", {"manufacturing_pmi": 48}, 5, "below", 0.6),
                PrecursorStep(5, "Credit spread widening", {"credit_spread": 0.02}, 3, "above", 0.7),
            ],
            historical_frequency=0.15, historical_accuracy=0.7,
            domains=["economics", "sentiment"], description="Classic recession precursor sequence",
            last_occurrence_year=2020, n_historical_occurrences=8, false_positive_rate=0.2,
        ))

        self.patterns.append(PrecursorPattern(
            name="financial_crisis_precursor",
            target_regime=RegimeType.FINANCIAL_CRISIS,
            steps=[
                PrecursorStep(1, "Rapid credit expansion", {"credit_growth": 0.1}, 24, "above", 0.5),
                PrecursorStep(2, "Rising leverage ratios", {"leverage_ratio": 3.0}, 18, "above", 0.5),
                PrecursorStep(3, "Asset price inflation", {"asset_price_ratio": 1.5}, 12, "above", 0.5),
                PrecursorStep(4, "Liquidity deterioration", {"liquidity_stress": 0.4}, 6, "above", 0.6),
                PrecursorStep(5, "Counterparty risk spike", {"counterparty_risk": 0.5}, 2, "above", 0.7),
                PrecursorStep(6, "Confidence collapse", {"confidence_index": 30}, 1, "below", 0.8),
            ],
            historical_frequency=0.08, historical_accuracy=0.65,
            domains=["economics", "contagion", "sentiment"], description="2008-style financial crisis precursor",
            last_occurrence_year=2008, n_historical_occurrences=5, false_positive_rate=0.3,
        ))

        self.patterns.append(PrecursorPattern(
            name="geopolitical_crisis_precursor",
            target_regime=RegimeType.GEOPOLITICAL_ESCALATION,
            steps=[
                PrecursorStep(1, "Diplomatic breakdown", {"diplomatic_index": 0.3}, 12, "below", 0.5),
                PrecursorStep(2, "Military buildup", {"military_readiness": 0.6}, 6, "above", 0.6),
                PrecursorStep(3, "Alliance fragmentation", {"alliance_cohesion": 0.4}, 4, "below", 0.5),
                PrecursorStep(4, "Resource competition intensification", {"resource_competition": 0.7}, 2, "above", 0.7),
            ],
            historical_frequency=0.1, historical_accuracy=0.55,
            domains=["geopolitical"], description="Geopolitical escalation precursor",
            last_occurrence_year=2022, n_historical_occurrences=12, false_positive_rate=0.35,
        ))

        self.patterns.append(PrecursorPattern(
            name="climate_tipping_precursor",
            target_regime=RegimeType.CLIMATE_TIPPING,
            steps=[
                PrecursorStep(1, "Accelerating temperature anomaly", {"temp_anomaly_c": 1.5}, 60, "above", 0.5),
                PrecursorStep(2, "Ice sheet acceleration", {"ice_loss_rate": 300}, 36, "above", 0.5),
                PrecursorStep(3, "Permafrost feedback activation", {"permafrost_release": 0.5}, 24, "above", 0.6),
                PrecursorStep(4, "Extreme event clustering", {"extreme_event_frequency": 3.0}, 12, "above", 0.6),
                PrecursorStep(5, "Carbon sink failure", {"carbon_sink_efficiency": 0.5}, 6, "below", 0.7),
            ],
            historical_frequency=0.02, historical_accuracy=0.4,
            domains=["climate"], description="Climate tipping point precursor",
            n_historical_occurrences=0, false_positive_rate=0.5,
        ))

        self.patterns.append(PrecursorPattern(
            name="technology_disruption_precursor",
            target_regime=RegimeType.TECHNOLOGY_DISRUPTION,
            steps=[
                PrecursorStep(1, "Research breakthrough cluster", {"research_breakthrough_index": 0.6}, 36, "above", 0.4),
                PrecursorStep(2, "Early adopter acceleration", {"adoption_rate": 0.15}, 18, "above", 0.5),
                PrecursorStep(3, "Incumbent defensive moves", {"incumbent_defense_index": 0.5}, 12, "above", 0.5),
                PrecursorStep(4, "Infrastructure buildup", {"infrastructure_readiness": 0.3}, 6, "above", 0.6),
                PrecursorStep(5, "Market shift signals", {"market_shift_velocity": 0.5}, 3, "above", 0.7),
            ],
            historical_frequency=0.12, historical_accuracy=0.5,
            domains=["technology"], description="Technology disruption precursor",
            n_historical_occurrences=6, false_positive_rate=0.4,
        ))

        self.patterns.append(PrecursorPattern(
            name="inflation_spike_precursor",
            target_regime=RegimeType.INFLATION_SPIKE,
            steps=[
                PrecursorStep(1, "Commodity price surge", {"commodity_index": 1.2}, 12, "above", 0.5),
                PrecursorStep(2, "Wage pressure buildup", {"wage_growth": 0.05}, 9, "above", 0.5),
                PrecursorStep(3, "Supply chain bottleneck", {"supply_chain_stress": 0.6}, 6, "above", 0.6),
                PrecursorStep(4, "Money supply expansion", {"money_supply_growth": 0.1}, 4, "above", 0.6),
                PrecursorStep(5, "Expectations unanchoring", {"inflation_expectations": 0.04}, 2, "above", 0.7),
            ],
            historical_frequency=0.12, historical_accuracy=0.65,
            domains=["economics"], description="Inflation spike precursor sequence",
            n_historical_occurrences=7, false_positive_rate=0.25,
        ))

        self.patterns.append(PrecursorPattern(
            name="social_unrest_precursor",
            target_regime=RegimeType.SOCIAL_UNREST,
            steps=[
                PrecursorStep(1, "Rising inequality", {"gini_coefficient": 0.45}, 18, "above", 0.5),
                PrecursorStep(2, "Institutional trust erosion", {"institutional_trust": 0.3}, 12, "below", 0.5),
                PrecursorStep(3, "Polarization acceleration", {"polarization_index": 0.6}, 6, "above", 0.6),
                PrecursorStep(4, "Economic hardship trigger", {"economic_hardship_index": 0.6}, 2, "above", 0.7),
            ],
            historical_frequency=0.2, historical_accuracy=0.55,
            domains=["social"], description="Social unrest precursor pattern",
            n_historical_occurrences=15, false_positive_rate=0.3,
        ))

    def get_pattern(self, name: str) -> Optional[PrecursorPattern]:
        for p in self.patterns:
            if p.name == name:
                return p
        return None

    def find_by_regime(self, regime: RegimeType) -> List[PrecursorPattern]:
        return [p for p in self.patterns if p.target_regime == regime]


class PrecursorMatcher:
    """Matches current domain states against library patterns for early warning."""

    def __init__(self, library: Optional[PrecursorPatternLibrary] = None):
        self.library = library or PrecursorPatternLibrary()
        self.warnings: List[RegimeChangeWarning] = []

    def scan(self, domain_values: Dict[str, float]) -> List[RegimeChangeWarning]:
        warnings = []
        for pattern in self.library.patterns:
            score, phase, step_scores = pattern.match_score(domain_values)
            if phase == Phase.NORMAL:
                continue
            base_conf = pattern.historical_accuracy * score
            fpr_penalty = pattern.false_positive_rate * (1 - score)
            confidence = max(0.0, min(1.0, base_conf - fpr_penalty))
            lead_time = pattern.estimated_lead_time(phase)
            next_signals = []
            for step in pattern.steps:
                if step_scores.get(step.order, 0) < step.threshold:
                    next_signals.append(step.description)
                    if len(next_signals) >= 3:
                        break
            precedent = f"Occurred {pattern.n_historical_occurrences} times historically"
            if pattern.last_occurrence_year:
                precedent += f", last in {pattern.last_occurrence_year}"
            actions = []
            if phase == Phase.EARLY_PRECURSOR:
                actions = ["Monitor closely", "Increase observation frequency", "Update contingency plans"]
            elif phase == Phase.MID_PRECURSOR:
                actions = ["Activate early warning protocols", "Review exposure to regime change", "Begin hedging positions"]
            elif phase == Phase.LATE_PRECURSOR:
                actions = ["Implement protective measures", "Reduce exposure", "Prepare response teams", "Alert stakeholders"]
            elif phase == Phase.IMMINENT:
                actions = ["Execute contingency plans", "Maximum protection mode", "Continuous monitoring"]

            warning = RegimeChangeWarning(
                pattern_name=pattern.name,
                target_regime=pattern.target_regime,
                phase=phase,
                match_score=round(score, 4),
                confidence=round(confidence, 4),
                estimated_lead_time_months=round(lead_time, 1),
                matched_signals=step_scores,
                next_expected_signals=next_signals,
                historical_precedent=precedent,
                recommended_actions=actions,
            )
            warnings.append(warning)
        warnings.sort(key=lambda w: (w.phase.value, -w.match_score))
        self.warnings = warnings
        return warnings

    def highest_risk_warning(self) -> Optional[RegimeChangeWarning]:
        active = [w for w in self.warnings if w.phase != Phase.NORMAL]
        if not active:
            return None
        return max(active, key=lambda w: (w.confidence, w.match_score))

    def threat_summary(self) -> Dict:
        if not self.warnings:
            return {"n_warnings": 0, "max_phase": "normal", "highest_risk": None, "by_regime": {}}
        by_regime = {}
        for w in self.warnings:
            regime = w.target_regime.value
            if regime not in by_regime:
                by_regime[regime] = []
            by_regime[regime].append({"pattern": w.pattern_name, "phase": w.phase.value,
                                      "score": w.match_score, "confidence": w.confidence,
                                      "lead_months": w.estimated_lead_time_months})
        phase_order = {Phase.NORMAL: 0, Phase.EARLY_PRECURSOR: 1, Phase.MID_PRECURSOR: 2,
                       Phase.LATE_PRECURSOR: 3, Phase.IMMINENT: 4, Phase.ACTIVE: 5}
        max_phase = max(self.warnings, key=lambda w: phase_order.get(w.phase, 0)).phase.value
        highest = self.highest_risk_warning()
        return {
            "n_warnings": len(self.warnings),
            "max_phase": max_phase,
            "highest_risk": highest.pattern_name if highest else None,
            "highest_risk_confidence": highest.confidence if highest else 0,
            "by_regime": by_regime,
        }