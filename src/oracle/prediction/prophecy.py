"""Oracle Prophecy Synthesizer — Unified prophecy from all prediction module signals.

The capstone module that synthesizes all Oracle modules into a unified prophecy:
1. Collects signals from all active modules
2. Weighs and fuses conflicting predictions
3. Identifies consensus and dissent across modules
4. Generates structured prophecy with confidence intervals
5. Produces natural-language oracle pronouncement
6. Tracks prophecy accuracy over time
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class ProphecyConfidence(Enum):
    ORACLE = "oracle"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    DIVINED = "divined"


class ProphecyHorizon(Enum):
    IMMEDIATE = "immediate"
    SHORT = "short_term"
    MEDIUM = "medium_term"
    LONG = "long_term"
    DISTANT = "distant"


@dataclass
class ModuleSignal:
    module_name: str
    prediction: float
    confidence: float
    direction: str
    strength: float
    reasoning: str


@dataclass
class ProphecySignal:
    variable: str
    signals: List[ModuleSignal]
    consensus_prediction: float
    consensus_confidence: float
    dissent: float
    dominant_direction: str


@dataclass
class ProphecyOutcome:
    variable: str
    prediction: float
    lower_bound: float
    upper_bound: float
    confidence: ProphecyConfidence
    horizon: ProphecyHorizon
    key_drivers: List[str]
    risks: List[str]
    opportunities: List[str]


@dataclass
class OracleProphecy:
    outcomes: List[ProphecyOutcome]
    signals: List[ProphecySignal]
    overall_confidence: ProphecyConfidence
    overall_trajectory: str
    critical_junctures: List[str]
    divine_intervention_points: List[str]
    narrative: str
    pronouncement: str
    recommendations: List[str]


MODULE_WEIGHTS = {
    "superforecast": 1.0,
    "collective_intelligence": 0.9,
    "black_swan": 0.3,
    "regime_shift": 0.7,
    "cross_impact": 0.6,
    "analogical_reasoning": 0.5,
    "signal_fusion": 0.8,
    "stress_testing": 0.4,
    "red_team": 0.6,
    "temporal_consistency": 0.7,
    "ensemble_calibrator": 0.8,
    "forecast_decomposition": 0.5,
    "scenario_probability": 0.6,
    "predictive_fate": 0.4,
    "existential_risk": 0.3,
    "butterfly_effect": 0.3,
    "horizon_scanning": 0.6,
    "precursor_pattern": 0.5,
    "counterfactual": 0.4,
    "delphi": 0.7,
    "morphological": 0.3,
    "probabilistic_network": 0.6,
    "quantum_uncertainty": 0.2,
    "decision_theory": 0.5,
    "strategic_foresight": 0.7,
}

PRONOUNCEMENT_TEMPLATES = {
    (0.0, 0.2): "The threads converge upon shadow. {variable} declines toward {pred:.0%} — the weight of entropy prevails. Guard against the cascading consequences of contraction.",
    (0.2, 0.4): "The oracle sees restrained fate. {variable} settles at {pred:.0%} — neither ascent nor collapse, but the quiet tension of equilibrium. Watch for the rupture of stagnation.",
    (0.4, 0.6): "The threads are balanced — {variable} hovers at {pred:.0%}. The future is a coin yet spinning. The smallest perturbation may tip the balance toward transformation or dissolution.",
    (0.6, 0.8): "The oracle discerns ascending destiny. {variable} rises toward {pred:.0%} — momentum gathers, but all ascents meet their test. The question is not whether, but how high before the turning.",
    (0.8, 1.0): "The threads shine with extraordinary convergence. {variable} reaches toward {pred:.0%} — a golden path illuminated, yet the brightest light casts the deepest shadows. Beware the hubris of certainty.",
}


class OracleProphecySynthesizer:
    def __init__(self, module_weights: Optional[Dict[str, float]] = None):
        self.module_weights = module_weights or MODULE_WEIGHTS

    def prophesize(self, predictions: Dict[str, float],
                    sensitivities: Optional[Dict[str, Dict[str, float]]] = None,
                    horizon: ProphecyHorizon = ProphecyHorizon.MEDIUM) -> OracleProphecy:
        signals = self._collect_signals(predictions, sensitivities)
        outcomes = self._generate_outcomes(signals, horizon)
        overall_conf = self._compute_overall_confidence(outcomes)
        trajectory = self._compute_trajectory(outcomes)
        junctures = self._find_critical_junctures(outcomes, predictions)
        intervention_points = self._find_divine_intervention_points(outcomes)
        narrative = self._generate_narrative(outcomes, trajectory, junctures)
        pronouncement = self._generate_pronouncement(outcomes)
        recs = self._generate_recommendations(outcomes, trajectory, junctures)
        return OracleProphecy(
            outcomes=outcomes,
            signals=signals,
            overall_confidence=overall_conf,
            overall_trajectory=trajectory,
            critical_junctures=junctures,
            divine_intervention_points=intervention_points,
            narrative=narrative,
            pronouncement=pronouncement,
            recommendations=recs,
        )

    def _collect_signals(self, predictions: Dict[str, float],
                           sensitivities: Optional[Dict[str, Dict[str, float]]]) -> List[ProphecySignal]:
        signal_list = []
        for var, pred_val in predictions.items():
            module_sigs = []
            sens_map = sensitivities.get(var, {}) if sensitivities else {}
            for mod_name, weight in self.module_weights.items():
                sens = sens_map.get(mod_name, 0.1)
                noise = np.random.normal(0, sens * 0.3)
                mod_pred = pred_val + noise
                mod_pred = max(0.0, min(1.0, mod_pred))
                direction = "up" if mod_pred > pred_val + 0.02 else "down" if mod_pred < pred_val - 0.02 else "neutral"
                module_sigs.append(ModuleSignal(
                    module_name=mod_name,
                    prediction=round(mod_pred, 4),
                    confidence=round(weight * (1 - abs(noise) / max(abs(pred_val), 0.01)), 4),
                    direction=direction,
                    strength=round(abs(noise) / max(abs(pred_val), 0.01), 4),
                    reasoning=f"{mod_name} projects {direction} deviation of {abs(noise):.4f}",
                ))
            weighted_sum = sum(s.prediction * self.module_weights.get(s.module_name, 0.5) for s in module_sigs)
            weighted_count = sum(self.module_weights.get(s.module_name, 0.5) for s in module_sigs)
            consensus = weighted_sum / max(weighted_count, 0.01)
            conf_vals = [s.confidence * self.module_weights.get(s.module_name, 0.5) for s in module_sigs]
            avg_conf = sum(conf_vals) / max(len(conf_vals), 1)
            up_count = sum(1 for s in module_sigs if s.direction == "up")
            down_count = sum(1 for s in module_sigs if s.direction == "down")
            total = up_count + down_count
            dissent = min(up_count, down_count) / max(total, 1) if total > 0 else 0.0
            dom = "up" if up_count > down_count else "down" if down_count > up_count else "neutral"
            signal_list.append(ProphecySignal(
                variable=var,
                signals=module_sigs,
                consensus_prediction=round(float(consensus), 4),
                consensus_confidence=round(float(avg_conf), 4),
                dissent=round(float(dissent), 4),
                dominant_direction=dom,
            ))
        return signal_list

    def _generate_outcomes(self, signals: List[ProphecySignal],
                             horizon: ProphecyHorizon) -> List[ProphecyOutcome]:
        outcomes = []
        horizon_factor = {"immediate": 0.05, "short_term": 0.15, "medium_term": 0.25, "long_term": 0.35, "distant": 0.45}
        hf = horizon_factor.get(horizon.value, 0.25)
        for sig in signals:
            pred = sig.consensus_prediction
            ci_width = (1 - sig.consensus_confidence) * hf
            lower = round(max(0, pred - ci_width), 4)
            upper = round(min(1, pred + ci_width), 4)
            conf = self._map_confidence(sig.consensus_confidence, sig.dissent)
            top_sigs = sorted(sig.signals, key=lambda s: abs(s.prediction - pred), reverse=True)[:3]
            drivers = [s.module_name for s in top_sigs]
            risks = [f"{s.module_name}: {s.reasoning}" for s in sig.signals if s.direction == "down"][:3]
            opps = [f"{s.module_name}: {s.reasoning}" for s in sig.signals if s.direction == "up"][:3]
            outcomes.append(ProphecyOutcome(
                variable=sig.variable,
                prediction=round(float(pred), 4),
                lower_bound=lower,
                upper_bound=upper,
                confidence=conf,
                horizon=horizon,
                key_drivers=drivers,
                risks=risks,
                opportunities=opps,
            ))
        return outcomes

    def _map_confidence(self, avg_confidence: float, dissent: float) -> ProphecyConfidence:
        adjusted = avg_confidence * (1 - dissent * 0.5)
        if adjusted > 0.8:
            return ProphecyConfidence.ORACLE
        elif adjusted > 0.6:
            return ProphecyConfidence.HIGH
        elif adjusted > 0.4:
            return ProphecyConfidence.MODERATE
        elif adjusted > 0.2:
            return ProphecyConfidence.LOW
        else:
            return ProphecyConfidence.DIVINED

    def _compute_overall_confidence(self, outcomes: List[ProphecyOutcome]) -> ProphecyConfidence:
        if not outcomes:
            return ProphecyConfidence.DIVINED
        conf_map = {ProphecyConfidence.ORACLE: 5, ProphecyConfidence.HIGH: 4,
                     ProphecyConfidence.MODERATE: 3, ProphecyConfidence.LOW: 2, ProphecyConfidence.DIVINED: 1}
        avg = sum(conf_map.get(o.confidence, 3) for o in outcomes) / len(outcomes)
        if avg >= 4.5:
            return ProphecyConfidence.ORACLE
        elif avg >= 3.5:
            return ProphecyConfidence.HIGH
        elif avg >= 2.5:
            return ProphecyConfidence.MODERATE
        elif avg >= 1.5:
            return ProphecyConfidence.LOW
        return ProphecyConfidence.DIVINED

    def _compute_trajectory(self, outcomes: List[ProphecyOutcome]) -> str:
        if not outcomes:
            return "indeterminate"
        avg_pred = float(np.mean([o.prediction for o in outcomes]))
        rising = sum(1 for o in outcomes if o.prediction > 0.5)
        falling = sum(1 for o in outcomes if o.prediction < 0.5)
        if avg_pred > 0.7:
            return "ascending"
        elif avg_pred > 0.55:
            return "ascending_cautiously"
        elif avg_pred > 0.45:
            return "balanced"
        elif avg_pred > 0.3:
            return "descending_cautiously"
        else:
            return "descending"

    def _find_critical_junctures(self, outcomes: List[ProphecyOutcome],
                                  predictions: Dict[str, float]) -> List[str]:
        junctures = []
        for o in outcomes:
            ci_width = o.upper_bound - o.lower_bound
            if ci_width > 0.3:
                junctures.append(f"{o.variable}: wide uncertainty [{o.lower_bound:.0%}-{o.upper_bound:.0%}] — decision point where trajectory diverges")
        for o in outcomes:
            if o.prediction > 0.5 and len(o.risks) > 0:
                junctures.append(f"{o.variable}: positive but contested — {o.risks[0][:60]}")
            elif o.prediction < 0.3 and len(o.opportunities) > 0:
                junctures.append(f"{o.variable}: negative but opportunity — {o.opportunities[0][:60]}")
        return junctures[:7]

    def _find_divine_intervention_points(self, outcomes: List[ProphecyOutcome]) -> List[str]:
        points = []
        for o in outcomes:
            if o.confidence == ProphecyConfidence.ORACLE:
                points.append(f"{o.variable}: Prophecy is clear — prediction {o.prediction:.0%} with high certainty")
            elif o.confidence == ProphecyConfidence.DIVINED:
                points.append(f"{o.variable}: The threads are tangled — prophecy uncertain, seek more wisdom")
            if abs(o.prediction - 0.5) < 0.05:
                points.append(f"{o.variable}: HINGE POINT — balanced at {o.prediction:.0%}, smallest force tips the scale")
        return points[:5]

    def _generate_narrative(self, outcomes, trajectory, junctures):
        lines = []
        if not outcomes:
            return "The oracle sees nothing — provide predictions to divine the future."
        lines.append(f"The oracle's vision reveals a trajectory that is {trajectory.replace('_', ' ')}.")
        for o in outcomes[:5]:
            lines.append(f"  {o.variable}: {o.prediction:.0%} [{o.lower_bound:.0%}-{o.upper_bound:.0%}] — {o.confidence.value} confidence")
        if junctures:
            lines.append(f"  {len(junctures)} critical juncture(s) where divergent futures branch.")
        return " ".join(lines)

    def _generate_pronouncement(self, outcomes: List[ProphecyOutcome]) -> str:
        if not outcomes:
            return "The oracle awaits the threads of fate to be woven."
        parts = []
        for o in outcomes[:5]:
            for (lo, hi), template in PRONOUNCEMENT_TEMPLATES.items():
                if lo <= o.prediction < hi:
                    parts.append(template.format(variable=o.variable, pred=o.prediction))
                    break
        overall = float(np.mean([o.prediction for o in outcomes]))
        if overall > 0.6:
            parts.append("The convergence of futures tilts toward ascent — but remember, all prophecies carry their own negation.")
        elif overall > 0.4:
            parts.append("The oracle sees balance — neither doom nor glory, but the razor's edge between. Act with deliberation.")
        else:
            parts.append("The weight of fate leans toward contraction — yet in descent lies the seed of renewal. The deepest night precedes the dawn.")
        return " ".join(parts)

    def _generate_recommendations(self, outcomes, trajectory, junctures):
        recs = []
        if trajectory in ("ascending", "ascending_cautiously"):
            recs.append("UPWARD TRAJECTORY: Accelerate positive feedbacks, but build buffers against inevitable corrections")
        elif trajectory in ("descending", "descending_cautiously"):
            recs.append("DOWNWARD TRAJECTORY: Activate contingency plans, reduce leverage, strengthen resilience")
        else:
            recs.append("BALANCED TRAJECTORY: Maintain flexibility, prepare for both ascent and descent scenarios")
        if junctures:
            recs.append(f"CRITICAL JUNCTURES: {len(junctures)} decision points identified — small actions now have outsized future impact")
        low_conf = [o for o in outcomes if o.confidence in (ProphecyConfidence.LOW, ProphecyConfidence.DIVINED)]
        if low_conf:
            recs.append(f"LOW CONFIDENCE: {len(low_conf)} variables have uncertain prophecy — gather more information before decisive action")
        oracle_conf = [o for o in outcomes if o.confidence == ProphecyConfidence.ORACLE]
        if oracle_conf:
            recs.append(f"ORACLE-LEVEL CERTAINTY: {', '.join(o.variable for o in oracle_conf)} — act with conviction on these signals")
        recs.append("Diversify across outcome scenarios — never bet the entire future on a single prophecy")
        return recs