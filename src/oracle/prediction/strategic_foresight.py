"""Strategic Foresight Integrator — Combines all prediction modules into unified foresight.

The capstone module that integrates signals from ALL other prediction modules into
a single coherent strategic foresight assessment. This is the "conductor" that:

1. Collects signals from all modules (precursor patterns, analogies, black swans, etc.)
2. Weights signals by module reliability and relevance
3. Identifies convergent signals (multiple modules agree)
4. Flags divergent signals (modules disagree — richest learning opportunity)
5. Produces a Unified Foresight Assessment with calibrated probability ranges
6. Generates strategic recommendations with action timelines

This module embodies the principle that no single prediction method is sufficient —
the future is best anticipated by triangulating across many independent methods.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CONVERGENT = "convergent"


class SignalDirection(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


@dataclass
class ModuleSignal:
    module: str
    variable: str
    direction: SignalDirection
    strength: float
    probability: float
    confidence: float
    description: str


@dataclass
class ConvergentSignal:
    variable: str
    contributing_modules: List[str]
    direction: SignalDirection
    combined_strength: float
    combined_probability: float
    confidence: float
    description: str


@dataclass
class DivergentSignal:
    variable: str
    module_positions: Dict[str, SignalDirection]
    disagreement_magnitude: float
    learning_opportunity: str


@dataclass
class ForesightAssessment:
    overall_outlook: str
    probability_range: Tuple[float, float]
    convergent_signals: List[ConvergentSignal]
    divergent_signals: List[DivergentSignal]
    key_risks: List[str]
    key_opportunities: List[str]
    strategic_recommendations: List[str]
    action_timeline: Dict[str, int]
    confidence: float
    module_coverage: float
    narrative: str


MODULE_WEIGHTS: Dict[str, float] = {
    "superforecast": 0.9,
    "analogical_reasoning": 0.7,
    "precursor_pattern": 0.75,
    "black_swan": 0.6,
    "regime_shift": 0.7,
    "collective_intelligence": 0.8,
    "morphological": 0.5,
    "cross_impact": 0.7,
    "temporal_causal": 0.65,
    "foresight_chain": 0.6,
    "cascade_forecast": 0.55,
    "butterfly_effect": 0.6,
    "information_horizon": 0.5,
    "horizon_scanning": 0.55,
    "probabilistic_network": 0.75,
    "delphi": 0.7,
    "counterfactual": 0.65,
    "weak_signal": 0.45,
    "emergence_detector": 0.5,
    "temporal_contradiction": 0.55,
    "prediction_market": 0.7,
    "quantum_uncertainty": 0.4,
    "decision_theory": 0.6,
}


class StrategicForesightIntegrator:
    def __init__(self, module_weights: Optional[Dict[str, float]] = None):
        self.module_weights = module_weights or dict(MODULE_WEIGHTS)
        self.signals: List[ModuleSignal] = []

    def add_signal(self, module: str, variable: str,
                    direction: SignalDirection, strength: float,
                    probability: float, confidence: float,
                    description: str = "") -> None:
        self.signals.append(ModuleSignal(
            module=module, variable=variable, direction=direction,
            strength=strength, probability=probability, confidence=confidence,
            description=description,
        ))

    def collect_from_engine(self, engine) -> None:
        """Collect signals from all engine modules."""
        self.signals = []
        try:
            if hasattr(engine, 'precursor_matcher') and hasattr(engine, 'state'):
                self._collect_precursor_signals(engine)
        except Exception:
            pass
        try:
            if hasattr(engine, 'cross_impact'):
                self._collect_cross_impact_signals(engine)
        except Exception:
            pass
        try:
            if hasattr(engine, 'probabilistic_network'):
                self._collect_network_signals(engine)
        except Exception:
            pass
        try:
            if hasattr(engine, 'horizon_scanner'):
                self._collect_horizon_signals(engine)
        except Exception:
            pass

    def integrate(self, domain: str = "global") -> ForesightAssessment:
        convergent = self._find_convergent_signals()
        divergent = self._find_divergent_signals()
        risks = self._extract_risks(convergent, divergent)
        opportunities = self._extract_opportunities(convergent)
        prob_range = self._compute_probability_range(convergent)
        overall = self._compute_overall_outlook(convergent, prob_range)
        recs = self._generate_recommendations(convergent, divergent, risks, opportunities)
        timeline = self._generate_timeline(convergent)
        coverage = self._compute_coverage()
        confidence = self._compute_overall_confidence(convergent)
        narrative = self._generate_narrative(domain, overall, convergent, divergent)
        return ForesightAssessment(
            overall_outlook=overall,
            probability_range=prob_range,
            convergent_signals=convergent,
            divergent_signals=divergent,
            key_risks=risks,
            key_opportunities=opportunities,
            strategic_recommendations=recs,
            action_timeline=timeline,
            confidence=round(confidence, 4),
            module_coverage=round(coverage, 4),
            narrative=narrative,
        )

    def _collect_precursor_signals(self, engine) -> None:
        try:
            library = engine.precursor_matcher.library
            for pattern in library.patterns[:3]:
                self.add_signal("precursor_pattern", pattern.regime_type.value,
                                SignalDirection.NEGATIVE, 0.6, 0.3, 0.6,
                                f"Precursor pattern: {pattern.name}")
        except Exception:
            pass

    def _collect_cross_impact_signals(self, engine) -> None:
        try:
            result = engine.cross_impact.solve()
            for name, change in result.probability_changes.items():
                if abs(change) > 0.05:
                    direction = SignalDirection.NEGATIVE if change > 0 and name in ("recession", "financial_crisis", "social_unrest") else SignalDirection.POSITIVE
                    self.add_signal("cross_impact", name, direction,
                                    min(1.0, abs(change) * 5), result.revised_probabilities.get(name, 0.5), 0.7,
                                    f"Cross-impact: {name} revised by {change:+.2f}")
        except Exception:
            pass

    def _collect_network_signals(self, engine) -> None:
        try:
            result = engine.probabilistic_network.infer({})
            for name, probs in result.posteriors.items():
                max_idx = np.argmax(probs)
                max_prob = probs[max_idx]
                node = engine.probabilistic_network.nodes.get(name)
                if node and max_prob > 0.5:
                    state = node.states[max_idx]
                    dirn = SignalDirection.NEGATIVE if state in ("yes", "high", "crisis") else SignalDirection.POSITIVE
                    self.add_signal("probabilistic_network", name, dirn,
                                    max_prob, max_prob, 0.75,
                                    f"Network inference: {name} most likely {state} ({max_prob:.0%})")
        except Exception:
            pass

    def _collect_horizon_signals(self, engine) -> None:
        try:
            result = engine.horizon_scanner.scan(min_velocity=0.5)
            for signal in result.high_velocity_signals[:3]:
                self.add_signal("horizon_scanning", signal.name, SignalDirection.UNCERTAIN,
                                signal.velocity, signal.probability, 0.5,
                                f"High-velocity signal: {signal.name} (v={signal.velocity:.2f})")
        except Exception:
            pass

    def _find_convergent_signals(self) -> List[ConvergentSignal]:
        var_signals: Dict[str, List[ModuleSignal]] = {}
        for sig in self.signals:
            if sig.variable not in var_signals:
                var_signals[sig.variable] = []
            var_signals[sig.variable].append(sig)
        convergent = []
        for var, sigs in var_signals.items():
            if len(sigs) >= 2:
                directions = [s.direction for s in sigs]
                dir_counts = {}
                for d in directions:
                    dir_counts[d] = dir_counts.get(d, 0) + 1
                dominant_dir = max(dir_counts, key=dir_counts.get)
                if dir_counts[dominant_dir] >= 2:
                    avg_strength = np.mean([s.strength * self.module_weights.get(s.module, 0.5) for s in sigs])
                    avg_prob = np.mean([s.probability for s in sigs])
                    avg_conf = np.mean([s.confidence for s in sigs])
                    bonus = min(0.3, len(sigs) * 0.05)
                    convergent.append(ConvergentSignal(
                        variable=var,
                        contributing_modules=[s.module for s in sigs],
                        direction=dominant_dir,
                        combined_strength=round(min(1.0, avg_strength + bonus), 4),
                        combined_probability=round(avg_prob, 4),
                        confidence=round(min(0.95, avg_conf + bonus), 4),
                        description=f"{len(sigs)} modules converge on {var} ({dominant_dir.value})",
                    ))
        convergent.sort(key=lambda c: c.combined_strength * c.confidence, reverse=True)
        return convergent

    def _find_divergent_signals(self) -> List[DivergentSignal]:
        var_signals: Dict[str, List[ModuleSignal]] = {}
        for sig in self.signals:
            if sig.variable not in var_signals:
                var_signals[sig.variable] = []
            var_signals[sig.variable].append(sig)
        divergent = []
        for var, sigs in var_signals.items():
            if len(sigs) >= 2:
                directions = set(s.direction for s in sigs)
                if len(directions) > 1:
                    positions = {s.module: s.direction for s in sigs}
                    magnitude = np.std([s.probability for s in sigs])
                    divergent.append(DivergentSignal(
                        variable=var,
                        module_positions=positions,
                        disagreement_magnitude=round(magnitude, 4),
                        learning_opportunity=f"Modules disagree on {var} — investigate underlying assumptions",
                    ))
        divergent.sort(key=lambda d: d.disagreement_magnitude, reverse=True)
        return divergent[:5]

    def _extract_risks(self, convergent: List[ConvergentSignal],
                         divergent: List[DivergentSignal]) -> List[str]:
        risks = []
        for c in convergent[:5]:
            if c.direction in (SignalDirection.NEGATIVE, SignalDirection.UNCERTAIN):
                risks.append(f"{c.variable}: {c.description} (P={c.combined_probability:.0%}, confidence={c.confidence:.0%})")
        for d in divergent[:2]:
            risks.append(f"{d.variable}: {d.learning_opportunity}")
        return risks[:7]

    def _extract_opportunities(self, convergent: List[ConvergentSignal]) -> List[str]:
        opportunities = []
        for c in convergent[:5]:
            if c.direction == SignalDirection.POSITIVE:
                opportunities.append(f"{c.variable}: {c.description} (P={c.combined_probability:.0%})")
        return opportunities[:5]

    def _compute_probability_range(self, convergent: List[ConvergentSignal]) -> Tuple[float, float]:
        if convergent:
            probs = [c.combined_probability for c in convergent]
            mean_prob = np.mean(probs)
            std_prob = np.std(probs) if len(probs) > 1 else 0.1
            lower = max(0.01, mean_prob - 1.96 * std_prob)
            upper = min(0.99, mean_prob + 1.96 * std_prob)
        else:
            lower, upper = 0.2, 0.8
        return (round(lower, 4), round(upper, 4))

    def _compute_overall_outlook(self, convergent: List[ConvergentSignal],
                                   prob_range: Tuple[float, float]) -> str:
        neg = sum(1 for c in convergent if c.direction == SignalDirection.NEGATIVE)
        pos = sum(1 for c in convergent if c.direction == SignalDirection.POSITIVE)
        if neg > pos * 2:
            return "bearish"
        elif pos > neg * 2:
            return "bullish"
        elif neg > pos:
            return "cautious"
        elif pos > neg:
            return "optimistic"
        return "neutral"

    def _generate_recommendations(self, convergent: List[ConvergentSignal],
                                    divergent: List[DivergentSignal],
                                    risks: List[str], opportunities: List[str]) -> List[str]:
        recs = []
        if risks:
            recs.append(f"RISK MANAGEMENT: Address top risk — {risks[0]}")
        if opportunities:
            recs.append(f"OPPORTUNITY: Capitalize on — {opportunities[0]}")
        if divergent:
            recs.append(f"RESEARCH: Resolve {len(divergent)} divergent signals through targeted investigation")
        if convergent:
            recs.append(f"CONFIDENCE: {len(convergent)} convergent signals provide actionability")
        recs.append("Monitor all convergent signals for strengthening or weakening")
        return recs

    def _generate_timeline(self, convergent: List[ConvergentSignal]) -> Dict[str, int]:
        timeline = {}
        for c in convergent[:5]:
            days = max(7, int(365 * (1 - c.combined_probability)))
            timeline[c.variable] = days
        return timeline

    def _compute_coverage(self) -> float:
        active_modules = set(s.module for s in self.signals)
        total_modules = len(self.module_weights)
        return len(active_modules) / max(total_modules, 1)

    def _compute_overall_confidence(self, convergent: List[ConvergentSignal]) -> float:
        if not convergent:
            return 0.2
        return min(0.9, np.mean([c.confidence for c in convergent]) + 0.1 * min(len(convergent), 5))

    def _generate_narrative(self, domain: str, outlook: str,
                              convergent: List[ConvergentSignal],
                              divergent: List[DivergentSignal]) -> str:
        parts = [f"Strategic foresight assessment for {domain}: overall outlook is {outlook}."]
        if convergent:
            parts.append(f"{len(convergent)} convergent signals identified across {len(set(m for c in convergent for m in c.contributing_modules))} modules.")
        if divergent:
            parts.append(f"{len(divergent)} divergent signals warrant further investigation.")
        top = convergent[0] if convergent else None
        if top:
            parts.append(f"Strongest convergent signal: {top.variable} ({top.direction.value}, strength={top.combined_strength:.2f}).")
        return " ".join(parts)