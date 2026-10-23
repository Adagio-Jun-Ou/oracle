"""Emergent Narrative Engine — generates human-readable narratives from quantitative forecasts.

Uses NO LLM. Instead, it maps quantitative predictions to narrative templates,
detects storylines (crisis, recovery, boom, stagnation), identifies protagonists
(domains driving the story), and crafts coherent multi-paragraph narratives.

Outside-the-box approach: narratives emerge from data patterns, not text generation.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StoryArc(Enum):
    CRISIS = "crisis"
    RECOVERY = "recovery"
    BOOM = "boom"
    STAGNATION = "stagnation"
    TIPPING_POINT = "tipping_point"
    BLACK_SWAN = "black_swan"
    RESILIENT_GROWTH = "resilient_growth"
    CASCADE_FAILURE = "cascade_failure"


class Severity(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"


@dataclass
class NarrativeSegment:
    headline: str
    body: str
    domain: str
    severity: Severity
    confidence: float
    timeline: str
    key_numbers: Dict[str, float]


@dataclass
class EmergentNarrative:
    title: str
    story_arc: StoryArc
    segments: List[NarrativeSegment]
    protagonists: List[str]
    antagonists: List[str]
    plot_twist: Optional[str]
    resolution_scenarios: List[Dict]
    overall_severity: Severity
    narrative_confidence: float
    method: str = "emergent_narrative"


DOMAIN_VOCABULARY = {
    "economics": {
        "up": ["expansion", "growth", "recovery", "boom", "surge", "rally"],
        "down": ["contraction", "recession", "downturn", "slump", "crisis", "crash"],
        "stable": ["stability", "plateau", "steady state", "equilibrium"],
        "metrics": ["GDP growth", "debt-to-GDP", "inflation", "unemployment", "interest rates"],
        "actors": ["central banks", "governments", "consumers", "corporations", "investors"],
    },
    "geopolitics": {
        "up": ["escalation", "tension rise", "conflict intensification", "confrontation"],
        "down": ["de-escalation", "detente", "diplomatic breakthrough", "peace process"],
        "stable": ["status quo", "frozen conflict", "managed competition"],
        "metrics": ["tension index", "conflict probability", "alliance strength", "trade disruption"],
        "actors": ["nation-states", "alliances", "insurgents", "diplomats", "military"],
    },
    "climate": {
        "up": ["warming acceleration", "extreme events", "tipping approach", "irreversible change"],
        "down": ["mitigation progress", "emissions decline", "green transition", "adaptation"],
        "stable": ["moderate warming", "managed transition", "steady increase"],
        "metrics": ["temperature anomaly", "sea level rise", "CO2 concentration", "extreme events"],
        "actors": ["governments", "industry", "climate scientists", "activists"],
    },
    "technology": {
        "up": ["breakthrough", "disruption", "adoption surge", "innovation wave"],
        "down": ["stagnation", "bubble burst", "regulatory barrier", "adoption plateau"],
        "stable": ["incremental progress", "steady adoption", "maturation"],
        "metrics": ["adoption rate", "capability index", "disruption probability", "innovation speed"],
        "actors": ["tech companies", "regulators", "researchers", "early adopters"],
    },
    "health": {
        "up": ["outbreak", "pandemic escalation", "health crisis", "systemic strain"],
        "down": ["containment", "recovery", "vaccination success", "resilience"],
        "stable": ["managed risk", "baseline health", "endemic equilibrium"],
        "metrics": ["R0", "hospital capacity", "vaccination rate", "mortality rate"],
        "actors": ["public health agencies", "healthcare workers", "pharmaceutical companies", "populations"],
    },
}


class NarrativeEngine:
    """Generate emergent narratives from quantitative forecast data."""

    def __init__(self):
        self.arc_thresholds = {
            StoryArc.CRISIS: {"risk": 0.7, "n_domains_above": 3},
            StoryArc.RECOVERY: {"improvement": 0.3, "n_domains_improving": 3},
            StoryArc.BOOM: {"growth": 0.6, "risk": 0.3},
            StoryArc.STAGNATION: {"variance": 0.1, "n_flat": 5},
            StoryArc.TIPPING_POINT: {"acceleration": 0.5, "nonlinearity": 0.7},
            StoryArc.BLACK_SWAN: {"tail_prob": 0.01, "n_extreme": 3},
            StoryArc.RESILIENT_GROWTH: {"growth": 0.4, "risk_managed": 0.6},
            StoryArc.CASCADE_FAILURE: {"cascade_depth": 3, "amplification": 2.0},
        }

    def _classify_domain(self, domain: str, values: Dict[str, float],
                          thresholds: Optional[Dict] = None) -> Tuple[str, float]:
        if not values:
            return "stable", 0.5
        avg = np.mean(list(values.values()))
        std = np.std(list(values.values())) if len(values) > 1 else 0.0
        if thresholds:
            high = thresholds.get("high", 0.7)
            low = thresholds.get("low", 0.3)
        else:
            high, low = 0.7, 0.3
        risk_domains = {"geopolitics", "cyber", "pandemic", "seismic",
                         "climate", "water", "food"}
        if domain in risk_domains:
            if avg > high:
                return "worsening", avg
            elif avg < low:
                return "improving", avg
            else:
                return "stable", avg
        else:
            growth_domains = {"technology", "energy", "economics"}
            if domain in growth_domains:
                if avg > 0.02 or (domain == "economics" and avg > 0.6):
                    return "strengthening", avg
                elif avg < -0.02 or avg < 0.4:
                    return "weakening", avg
                else:
                    return "stable", avg
            if avg > high:
                return "elevated", avg
            elif avg < low:
                return "low", avg
            return "moderate", avg

    def _detect_arc(self, domain_states: Dict[str, Dict],
                     risk_scores: Optional[Dict[str, float]] = None) -> StoryArc:
        n_worsening = 0
        n_improving = 0
        total_risk = 0.0
        severe_count = 0
        for domain, vals in domain_states.items():
            direction, magnitude = self._classify_domain(domain, vals)
            if direction in ("worsening", "elevated", "strengthening"):
                if direction == "worsening" or (direction == "strengthening" and domain != "technology"):
                    n_worsening += 1
                total_risk += magnitude
                if magnitude > 0.7:
                    severe_count += 1
            elif direction in ("improving", "low"):
                n_improving += 1
        n_domains = len(domain_states) or 1
        avg_risk = total_risk / max(n_domains, 1)
        if severe_count >= 3 and avg_risk > 0.7:
            return StoryArc.CASCADE_FAILURE
        elif severe_count >= 2 and n_worsening >= 3:
            return StoryArc.CRISIS
        elif n_improving >= 3 and avg_risk < 0.4:
            return StoryArc.RECOVERY
        elif n_worsening <= 1 and avg_risk < 0.3:
            return StoryArc.BOOM
        elif n_worsening <= 1 and avg_risk < 0.5:
            return StoryArc.RESILIENT_GROWTH
        elif severe_count >= 1 and avg_risk > 0.6:
            return StoryArc.TIPPING_POINT
        elif avg_risk > 0.8 and n_worsening >= 4:
            return StoryArc.BLACK_SWAN
        else:
            return StoryArc.STAGNATION

    def _severity(self, risk: float) -> Severity:
        if risk >= 0.9:
            return Severity.EXTREME
        elif risk >= 0.7:
            return Severity.CRITICAL
        elif risk >= 0.5:
            return Severity.HIGH
        elif risk >= 0.3:
            return Severity.MODERATE
        return Severity.LOW

    def _generate_headline(self, arc: StoryArc, domain_states: Dict) -> str:
        templates = {
            StoryArc.CRISIS: "Multi-domain crisis emerging as {worst_domain} deteriorates — {n_affected} domains under severe pressure",
            StoryArc.RECOVERY: "Signs of recovery across {n_improving} domains as systemic risk declines",
            StoryArc.BOOM: "Conditions favorable for growth: {n_stable} domains stable, risk elevated in only {n_worsening}",
            StoryArc.STAGNATION: "Prolonged stagnation risk: most domains show flat trajectories with limited upside",
            StoryArc.TIPPING_POINT: "{worst_domain} approaching critical threshold — cascading effects possible",
            StoryArc.BLACK_SWAN: "Extreme tail scenario: {n_affected} domains simultaneously under severe stress",
            StoryArc.RESILIENT_GROWTH: "Resilient expansion: growth continues despite elevated risk in {n_worsening} domains",
            StoryArc.CASCADE_FAILURE: "Cascade failure risk: {worst_domain} trigger could propagate through {n_affected} domains",
        }
        worst_domain = "global risk"
        n_affected = 0
        n_improving = 0
        n_stable = 0
        n_worsening = 0
        for domain, vals in domain_states.items():
            direction, mag = self._classify_domain(domain, vals)
            if direction in ("worsening", "elevated"):
                n_affected += 1
                n_worsening += 1
                if not worst_domain or worst_domain == "global risk":
                    worst_domain = domain.replace("_", " ")
            elif direction in ("improving", "low"):
                n_improving += 1
            else:
                n_stable += 1
        return templates.get(arc, "Multi-domain outlook uncertain").format(
            worst_domain=worst_domain, n_affected=n_affected,
            n_improving=n_improving, n_stable=n_stable, n_worsening=n_worsening)

    def _generate_segment(self, domain: str, direction: str,
                           magnitude: float, values: Dict[str, float]) -> NarrativeSegment:
        vocab = DOMAIN_VOCABULARY.get(domain, DOMAIN_VOCABULARY.get("economics", {}))
        direction_key = "up" if direction in ("worsening", "elevated", "strengthening") else "down" if direction in ("improving", "low", "weakening") else "stable"
        terms = vocab.get(direction_key, ["change", "shift", "movement"])
        term = terms[0] if terms else "change"
        metrics = vocab.get("metrics", ["indicators"])
        key_metric = metrics[0] if metrics else "index"
        actors = vocab.get("actors", ["stakeholders"])
        actor = actors[0] if actors else "actors"

        headline = f"{domain.replace('_', ' ').title()}: {term.capitalize()} detected"
        scale = "significant" if magnitude > 0.6 else "moderate" if magnitude > 0.3 else "slight"
        body = f"The {domain.replace('_', ' ')} domain shows a {scale} {term}. "
        if magnitude > 0.7:
            body += f"Critical threshold approaching as {key_metric} reaches {magnitude:.1%}. "
            body += f"{actor.capitalize()} face increasing pressure to respond. "
        elif magnitude > 0.4:
            body += f"{key_metric.capitalize()} trending at {magnitude:.1%}, above moderate thresholds. "
            body += f"Monitoring {actor} responses for acceleration signals. "
        else:
            body += f"{key_metric.capitalize()} at {magnitude:.1%}, within expected range. "
            body += f"No immediate action required from {actor}. "

        if values:
            top_key = max(values, key=values.get) if values else ""
            top_val = values.get(top_key, 0)
            body += f"Key driver: {top_key} ({top_val:.2f})."

        severity = self._severity(magnitude)
        confidence = min(0.95, 0.5 + magnitude * 0.3)
        if direction in ("worsening", "elevated") and magnitude > 0.5:
            timeline = "1-3 months"
        elif direction in ("improving",) and magnitude < 0.3:
            timeline = "6-12 months"
        else:
            timeline = "3-6 months"

        return NarrativeSegment(
            headline=headline, body=body, domain=domain,
            severity=severity, confidence=round(confidence, 2),
            timeline=timeline, key_numbers=values,
        )

    def generate_narrative(self, domain_states: Dict[str, Dict[str, float]],
                            risk_scores: Optional[Dict[str, float]] = None,
                            cross_domain_effects: Optional[Dict[str, Dict[str, float]]] = None) -> EmergentNarrative:
        arc = self._detect_arc(domain_states, risk_scores)
        segments = []
        protagonists = []
        antagonists = []
        for domain, values in domain_states.items():
            direction, magnitude = self._classify_domain(domain, values)
            segment = self._generate_segment(domain, direction, magnitude, values)
            segments.append(segment)
            if magnitude > 0.6:
                if direction in ("improving", "low", "strengthening"):
                    protagonists.append(domain.replace("_", " "))
                else:
                    antagonists.append(domain.replace("_", " "))

        if not protagonists:
            protagonists = ["systemic resilience"]
        if not antagonists and arc in (StoryArc.CRISIS, StoryArc.CASCADE_FAILURE, StoryArc.BLACK_SWAN):
            antagonists = ["compound risk"]

        plot_twist = None
        if cross_domain_effects:
            max_amp = 0.0
            max_pair = ("", "")
            for src, targets in cross_domain_effects.items():
                for tgt, amp in targets.items():
                    if amp > max_amp:
                        max_amp = amp
                        max_pair = (src, tgt)
            if max_amp > 0.3:
                plot_twist = f"Unexpected amplification: {max_pair[0].replace('_', ' ')} → {max_pair[1].replace('_', ' ')} link ({max_amp:.1%} amplification) creates feedback loop"

        resolution_scenarios = []
        if arc in (StoryArc.CRISIS, StoryArc.CASCADE_FAILURE, StoryArc.BLACK_SWAN):
            resolution_scenarios = [
                {"name": "Coordinated intervention", "probability": 0.15, "description": "Multi-domain policy response contains cascade"},
                {"name": "Partial containment", "probability": 0.35, "description": "Some domains stabilize while others deteriorate further"},
                {"name": "Full escalation", "probability": 0.50, "description": "Cascading failures continue across all domains"},
            ]
        elif arc in (StoryArc.RECOVERY, StoryArc.RESILIENT_GROWTH):
            resolution_scenarios = [
                {"name": "Sustained recovery", "probability": 0.45, "description": "Recovery accelerates as risk domains continue improving"},
                {"name": "False dawn", "probability": 0.30, "description": "Initial improvement reverses as structural risks persist"},
                {"name": "Stabilization", "probability": 0.25, "description": "Growth moderates to steady state"},
            ]
        else:
            resolution_scenarios = [
                {"name": "Gradual shift", "probability": 0.40, "description": "Slow transition from current state"},
                {"name": "Sudden break", "probability": 0.25, "description": "External shock disrupts equilibrium"},
                {"name": "Continued flat", "probability": 0.35, "description": "Status quo persists"},
            ]

        overall_severity = self._severity(
            max(m for _, m in [self._classify_domain(d, v) for d, v in domain_states.items()])
            if domain_states else 0.5)
        narrative_confidence = 0.5 + 0.1 * min(len(domain_states), 10) / 10

        return EmergentNarrative(
            title=self._generate_headline(arc, domain_states),
            story_arc=arc,
            segments=segments,
            protagonists=protagonists[:3],
            antagonists=antagonists[:3],
            plot_twist=plot_twist,
            resolution_scenarios=resolution_scenarios,
            overall_severity=overall_severity,
            narrative_confidence=round(narrative_confidence, 2),
        )