"""Red Team Forecaster — Adversarial challenge of predictions from multiple angles.

Systematically attacks predictions to find weaknesses:
1. Bias decomposition (cognitive biases that may inflate predictions)
2. Base rate neglect check (are historical base rates properly weighted?)
3. Adversarial scenarios (construct worst-case disconfirming scenarios)
4. Assumption rupture (what if key assumptions are wrong?)
5. Contrarian argument generation (steel-manned opposing cases)
6. Overconfidence detection (calibration check on confidence intervals)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChallengeType(Enum):
    BIAS_DECOMPOSITION = "bias_decomposition"
    BASE_RATE_NEGLECT = "base_rate_neglect"
    ADVERSARIAL_SCENARIO = "adversarial_scenario"
    ASSUMPTION_RUPTURE = "assumption_rupture"
    CONTRARIAN_ARGUMENT = "contrarian_argument"
    OVERCONFIDENCE = "overconfidence"


@dataclass
class BiasFinding:
    bias_name: str
    description: str
    estimated_bias_magnitude: float
    adjustment_direction: str
    adjusted_value: Optional[float] = None


@dataclass
class AdversarialChallenge:
    challenge_type: ChallengeType
    title: str
    description: str
    severity: float
    plausibility: float
    impact_on_prediction: float
    counterargument: str


@dataclass
class AssumptionRupture:
    assumption: str
    rupture_scenario: str
    probability: float
    impact: float
    detection_signals: List[str]


@dataclass
class RedTeamResult:
    challenges: List[AdversarialChallenge]
    biases_found: List[BiasFinding]
    assumption_ruptures: List[AssumptionRupture]
    overall_vulnerability: float
    adjusted_prediction: Dict[str, float]
    confidence_adjustment: float
    key_weaknesses: List[str]
    recommendations: List[str]


COGNITIVE_BIASES = [
    {"name": "anchoring", "description": "Prediction anchored to initial value, insufficient adjustment",
     "magnitude_range": (0.05, 0.15), "direction": "toward_anchor"},
    {"name": "overconfidence", "description": "Confidence intervals too narrow, underestimated uncertainty",
     "magnitude_range": (0.10, 0.30), "direction": "narrow_ci"},
    {"name": "recency_bias", "description": "Overweight recent events, underweight historical base rates",
     "magnitude_range": (0.05, 0.20), "direction": "toward_recent"},
    {"name": "confirmation_bias", "description": "Seeking confirming evidence, ignoring disconfirming data",
     "magnitude_range": (0.08, 0.25), "direction": "toward_belief"},
    {"name": "availability_bias", "description": "Vivid/recent events overweighted vs. statistically likely ones",
     "magnitude_range": (0.05, 0.15), "direction": "toward_available"},
    {"name": "optimism_bias", "description": "Systematic overestimation of positive outcomes",
     "magnitude_range": (0.10, 0.25), "direction": "upward"},
    {"name": "status_quo_bias", "description": "Assumption that current conditions will persist",
     "magnitude_range": (0.05, 0.20), "direction": "toward_current"},
    {"name": "sunk_cost", "description": "Continuing a course of action due to prior investment",
     "magnitude_range": (0.03, 0.10), "direction": "toward_prior"},
    {"name": "groupthink", "description": "Convergence toward consensus without adequate dissent",
     "magnitude_range": (0.05, 0.15), "direction": "toward_consensus"},
    {"name": "narrative_fallacy", "description": "Forcing causal narrative onto random/coincidental events",
     "magnitude_range": (0.05, 0.15), "direction": "toward_story"},
]

BASE_RATE_DATA = {
    "gdp_recession": {"annual_prob": 0.12, "10yr_prob": 0.65, "description": "Historical US recession frequency"},
    "market_crash_20pct": {"annual_prob": 0.05, "10yr_prob": 0.35, "description": "S&P 20%+ decline"},
    "banking_crisis": {"annual_prob": 0.03, "10yr_prob": 0.25, "description": "Systemic banking crisis"},
    "pandemic": {"annual_prob": 0.02, "10yr_prob": 0.15, "description": "Major pandemic (post-1918 scale)"},
    "sovereign_default": {"annual_prob": 0.04, "10yr_prob": 0.30, "description": "Emerging market sovereign default"},
    "war_major": {"annual_prob": 0.01, "10yr_prob": 0.10, "description": "Major interstate war"},
    "civil_war": {"annual_prob": 0.02, "10yr_prob": 0.15, "description": "Civil war onset"},
    "hyperinflation": {"annual_prob": 0.01, "10yr_prob": 0.08, "description": "Hyperinflation (>50%/mo)"},
    "currency_crisis": {"annual_prob": 0.05, "10yr_prob": 0.35, "description": "Currency crisis/crash"},
    "oil_shock": {"annual_prob": 0.04, "10yr_prob": 0.30, "description": "Oil price doubling in 1 year"},
    "tech_bubble": {"annual_prob": 0.03, "10yr_prob": 0.25, "description": "Sector-wide tech correction >30%"},
    "housing_crash": {"annual_prob": 0.03, "10yr_prob": 0.20, "description": "National housing price decline >20%"},
}


class RedTeamForecaster:
    def __init__(self, n_adversarial_scenarios: int = 10):
        self.n_adversarial_scenarios = n_adversarial_scenarios
        self._rng = np.random.default_rng(42)

    def red_team(self, predictions: Dict[str, float],
                  confidence: float = 0.9,
                  assumptions: Optional[List[str]] = None,
                  base_rates: Optional[Dict[str, float]] = None) -> RedTeamResult:
        biases = self._decompose_biases(predictions, confidence)
        base_rate_findings = self._check_base_rate_neglect(predictions, base_rates)
        adversarial = self._generate_adversarial_scenarios(predictions)
        ruptures = self._rupture_assumptions(predictions, assumptions)
        contrarian = self._generate_contrarian_arguments(predictions)
        overconfidence = self._detect_overconfidence(predictions, confidence)
        all_challenges = adversarial + contrarian + overconfidence
        all_challenges.extend(base_rate_findings)
        adjusted = self._compute_adjusted_prediction(predictions, biases, all_challenges)
        vulnerability = self._compute_vulnerability(all_challenges, biases)
        confidence_adj = self._adjust_confidence(confidence, vulnerability)
        key_weaknesses = self._extract_key_weaknesses(all_challenges, biases, ruptures)
        recs = self._generate_recommendations(vulnerability, key_weaknesses, confidence_adj)
        return RedTeamResult(
            challenges=all_challenges,
            biases_found=biases,
            assumption_ruptures=ruptures,
            overall_vulnerability=round(vulnerability, 4),
            adjusted_prediction=adjusted,
            confidence_adjustment=round(confidence_adj, 4),
            key_weaknesses=key_weaknesses,
            recommendations=recs,
        )

    def _decompose_biases(self, predictions: Dict[str, float],
                            confidence: float) -> List[BiasFinding]:
        findings = []
        for bias_info in COGNITIVE_BIASES:
            lo, hi = bias_info["magnitude_range"]
            magnitude = self._rng.uniform(lo, hi)
            if magnitude > 0.08:
                adjusted_vals = {}
                for var, val in predictions.items():
                    if bias_info["direction"] in ("upward", "toward_anchor", "toward_belief"):
                        adjusted_vals[var] = val * (1 - magnitude)
                    elif bias_info["direction"] in ("narrow_ci",):
                        adjusted_vals[var] = val
                    else:
                        adjusted_vals[var] = val * (1 + magnitude * 0.5)
                findings.append(BiasFinding(
                    bias_name=bias_info["name"],
                    description=bias_info["description"],
                    estimated_bias_magnitude=round(float(magnitude), 4),
                    adjustment_direction=bias_info["direction"],
                    adjusted_value=adjusted_vals,
                ))
        return findings

    def _check_base_rate_neglect(self, predictions: Dict[str, float],
                                   base_rates: Optional[Dict[str, float]]) -> List[AdversarialChallenge]:
        challenges = []
        rates = base_rates if base_rates is not None else {}
        for event, data in BASE_RATE_DATA.items():
            if event in rates:
                user_rate = rates[event]
                historical_rate = data["annual_prob"]
            else:
                continue
            if abs(user_rate - historical_rate) > 0.1:
                severity = min(1.0, abs(user_rate - historical_rate) * 2)
                direction = "overestimating" if user_rate > historical_rate else "underestimating"
                challenges.append(AdversarialChallenge(
                    challenge_type=ChallengeType.BASE_RATE_NEGLECT,
                    title=f"Base rate neglect: {event}",
                    description=f"Prediction is {direction} {event} (user: {user_rate:.0%} vs historical: {historical_rate:.0%})",
                    severity=round(severity, 4),
                    plausibility=0.7,
                    impact_on_prediction=round(abs(user_rate - historical_rate), 4),
                    counterargument=f"Historical base rate for {event} is {historical_rate:.0%}/year — {data['description']}",
                ))
        return challenges

    def _generate_adversarial_scenarios(self, predictions: Dict[str, float]) -> List[AdversarialChallenge]:
        challenges = []
        variables = list(predictions.keys())
        for i in range(self.n_adversarial_scenarios):
            if not variables:
                break
            var = variables[i % len(variables)]
            val = predictions[var]
            shock = -0.2 * (i + 1) / self.n_adversarial_scenarios
            scenarios = [
                (f"adversarial_{var}_down", f"Suppose {var} drops by {abs(shock):.0%} due to unforeseen factors", abs(shock), 0.3 + 0.05 * i),
                (f"adversarial_{var}_structural", f"Structural break in {var} relationship invalidates current model", 0.5, 0.1 + 0.03 * i),
            ]
            for name, desc, impact, plaus in scenarios:
                challenges.append(AdversarialChallenge(
                    challenge_type=ChallengeType.ADVERSARIAL_SCENARIO,
                    title=name,
                    description=desc,
                    severity=round(min(1.0, impact * val if val > 0 else impact), 4),
                    plausibility=round(min(1.0, plaus), 4),
                    impact_on_prediction=round(impact, 4),
                    counterargument=f"If {var} trajectory reverses, the prediction becomes unreliable",
                ))
        return challenges[: self.n_adversarial_scenarios]

    def _rupture_assumptions(self, predictions: Dict[str, float],
                               assumptions: Optional[List[str]]) -> List[AssumptionRupture]:
        ruptures = []
        default_assumptions = [
            "Continued economic growth",
            "Stable geopolitical order",
            "No major pandemic",
            "Functional supply chains",
            "Stable monetary policy",
        ]
        source = assumptions if assumptions else default_assumptions
        for i, assumption in enumerate(source):
            prob = 0.02 + 0.03 * i
            impact = 0.3 + 0.1 * i
            rupture_scenarios = {
                "Continued economic growth": "Global recession triggered by debt crisis or trade war",
                "Stable geopolitical order": "Major power conflict or alliance fragmentation",
                "No major pandemic": "Novel pathogen with high R0 and significant mortality",
                "Functional supply chains": "Cascading supply chain failure from port closures or cyber attack",
                "Stable monetary policy": "Central bank credibility crisis or policy error",
            }
            detection_signals = [
                "Leading economic indicators deteriorating",
                "Geopolitical tension escalation metrics rising",
                "Public health surveillance signals increasing",
                "Supply chain stress indices rising above threshold",
                "Central bank communication divergence increasing",
            ]
            ruptures.append(AssumptionRupture(
                assumption=assumption,
                rupture_scenario=rupture_scenarios.get(assumption, f"Key assumption '{assumption}' no longer holds"),
                probability=round(prob, 4),
                impact=round(min(1.0, impact), 4),
                detection_signals=detection_signals[i % len(detection_signals): i % len(detection_signals) + 2] or [detection_signals[0]],
            ))
        return ruptures

    def _generate_contrarian_arguments(self, predictions: Dict[str, float]) -> List[AdversarialChallenge]:
        challenges = []
        for var, val in list(predictions.items())[:5]:
            if val > 0.5:
                title = f"contrarian_low_{var}"
                desc = f"Steel-manned argument for why {var} is overestimated"
                impact = val * 0.3
                counter = f"Historical evidence suggests {var} tends to mean-revert"
            elif val < 0.3:
                title = f"contrarian_high_{var}"
                desc = f"Steel-manned argument for why {var} is underestimated"
                impact = (1 - val) * 0.2
                counter = f"Missing positive factors could drive {var} higher"
            else:
                title = f"contrarian_neutral_{var}"
                desc = f"Argument that {var} could swing either way based on overlooked factors"
                impact = 0.15
                counter = f"Both upside and downside risks exist for {var}"
            challenges.append(AdversarialChallenge(
                challenge_type=ChallengeType.CONTRARIAN_ARGUMENT,
                title=title,
                description=desc,
                severity=round(min(1.0, impact), 4),
                plausibility=0.4,
                impact_on_prediction=round(impact, 4),
                counterargument=counter,
            ))
        return challenges

    def _detect_overconfidence(self, predictions: Dict[str, float],
                                 confidence: float) -> List[AdversarialChallenge]:
        challenges = []
        if confidence > 0.9:
            adj = (confidence - 0.9) / 0.1
            challenges.append(AdversarialChallenge(
                challenge_type=ChallengeType.OVERCONFIDENCE,
                title="overconfidence_detected",
                description=f"Reported confidence {confidence:.0%} is above 90th percentile — likely miscalibrated",
                severity=round(adj, 4),
                plausibility=0.6,
                impact_on_prediction=round(adj * 0.2, 4),
                counterargument="Empirical evidence shows forecasters with >90% confidence are typically overconfident by 15-25%",
            ))
        if confidence > 0.7 and len(predictions) > 3:
            challenges.append(AdversarialChallenge(
                challenge_type=ChallengeType.OVERCONFIDENCE,
                title="multiple_variable_overconfidence",
                description=f"High confidence ({confidence:.0%}) across {len(predictions)} variables suggests systematic overconfidence",
                severity=round(min(1.0, 0.3), 4),
                plausibility=0.5,
                impact_on_prediction=0.15,
                counterargument="Confidence should decrease with the number of independent predictions",
            ))
        return challenges

    def _compute_adjusted_prediction(self, predictions: Dict[str, float],
                                       biases: List[BiasFinding],
                                       challenges: List[AdversarialChallenge]) -> Dict[str, float]:
        adjustments = {var: 0.0 for var in predictions}
        for bias in biases:
            if bias.adjusted_value:
                for var in predictions:
                    if var in bias.adjusted_value:
                        adj = bias.adjusted_value[var] - predictions[var]
                        adjustments[var] += adj * 0.3
        high_severity = [c for c in challenges if c.severity > 0.3]
        if high_severity:
            shrinkage = min(0.3, len(high_severity) * 0.05)
            for var in predictions:
                adjustments[var] -= predictions[var] * shrinkage * 0.1
        adjusted = {}
        for var, val in predictions.items():
            adjusted[var] = round(max(-1.0, min(2.0, val + adjustments[var])), 4)
        return adjusted

    def _compute_vulnerability(self, challenges: List[AdversarialChallenge],
                                biases: List[BiasFinding]) -> float:
        if not challenges and not biases:
            return 0.0
        challenge_severity = 0.0
        if challenges:
            challenge_severity = float(np.mean([c.severity for c in challenges]))
        bias_severity = 0.0
        if biases:
            bias_severity = float(np.mean([b.estimated_bias_magnitude for b in biases]))
        return min(1.0, challenge_severity * 0.6 + bias_severity * 0.4)

    def _adjust_confidence(self, confidence: float, vulnerability: float) -> float:
        penalty = vulnerability * 0.3
        return max(0.1, min(0.95, confidence - penalty))

    def _extract_key_weaknesses(self, challenges: List[AdversarialChallenge],
                                  biases: List[BiasFinding],
                                  ruptures: List[AssumptionRupture]) -> List[str]:
        weaknesses = []
        high_challenges = sorted(challenges, key=lambda c: c.severity, reverse=True)[:3]
        for c in high_challenges:
            if c.severity > 0.3:
                weaknesses.append(f"[{c.challenge_type.value}] {c.title}: severity={c.severity:.0%}")
        high_biases = sorted(biases, key=lambda b: b.estimated_bias_magnitude, reverse=True)[:2]
        for b in high_biases:
            weaknesses.append(f"[bias] {b.bias_name}: magnitude={b.estimated_bias_magnitude:.0%}")
        high_ruptures = sorted(ruptures, key=lambda r: r.probability * r.impact, reverse=True)[:2]
        for r in high_ruptures:
            weaknesses.append(f"[assumption] {r.assumption}: rupture risk={r.probability * r.impact:.1%}")
        return weaknesses

    def _generate_recommendations(self, vulnerability: float,
                                    key_weaknesses: List[str],
                                    confidence_adj: float) -> List[str]:
        recs = []
        if vulnerability > 0.5:
            recs.append("HIGH VULNERABILITY: Prediction is fragile — significantly decrease confidence and broaden intervals")
        elif vulnerability > 0.3:
            recs.append("MODERATE VULNERABILITY: Address identified weaknesses before relying on this prediction")
        else:
            recs.append("LOW VULNERABILITY: Prediction passes most red-team challenges")
        if key_weaknesses:
            recs.append(f"Top weakness: {key_weaknesses[0]}")
        recs.append(f"Adjusted confidence: {confidence_adj:.0%} (from original)")
        recs.append("Apply inside-view + outside-view blend to reduce cognitive biases")
        recs.append("Seek disconfirming evidence with same rigor as confirming evidence")
        return recs