"""Delphi Method Simulator — Structured expert judgment aggregation.

Simulates the Delphi method: iterative rounds of expert estimation with
controlled feedback between rounds. Experts provide estimates, receive
statistical feedback about the group's responses, and revise.

Key innovations:
1. Multi-round iterative convergence (3+ rounds)
2. Controlled feedback (quartiles + reasons, not raw estimates)
3. Expert calibration from historical accuracy
4. Dissent tracking (stable minority opinions may be right)
5. Confidence-rated estimates
6. Structured disagreement resolution
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class ExpertiseLevel(Enum):
    NOVICE = "novice"
    COMPETENT = "competent"
    EXPERT = "expert"
    MASTER = "master"


class ConvergenceStatus(Enum):
    CONVERGING = "converging"
    STABLE = "stable"
    DIVERGING = "diverging"
    POLARIZED = "polarized"


@dataclass
class SimulatedExpert:
    name: str
    expertise: ExpertiseLevel
    domain: str
    accuracy: float
    bias: float
    confidence_calibration: float
    openness_to_evidence: float
    current_estimate: Optional[float] = None
    estimate_history: List[float] = field(default_factory=list)
    confidence: float = 0.7
    rationale: str = ""


@dataclass
class DelphiRound:
    round_number: int
    estimates: Dict[str, float]
    confidences: Dict[str, float]
    rationales: Dict[str, str]
    median: float
    q1: float
    q3: float
    iqr: float
    convergence_score: float


@dataclass
class DelphiResult:
    question: str
    n_rounds: int
    final_median: float
    final_iqr: float
    convergence_status: ConvergenceStatus
    rounds: List[DelphiRound]
    stable_dissenters: List[str]
    confidence_intervals: Tuple[float, float]
    recommended_estimate: float
    uncertainty: float
    narrative: str


@dataclass
class DelphiPanel:
    experts: List[SimulatedExpert]
    question: str
    domain: str = ""


EXPERT_TEMPLATES: List[Dict] = [
    {"name": "dr_economics", "expertise": ExpertiseLevel.EXPERT, "domain": "economic", "accuracy": 0.72, "bias": 0.03, "confidence_calibration": 0.8, "openness": 0.7},
    {"name": "prof_finance", "expertise": ExpertiseLevel.EXPERT, "domain": "financial", "accuracy": 0.68, "bias": -0.02, "confidence_calibration": 0.75, "openness": 0.65},
    {"name": "analyst_geopolitical", "expertise": ExpertiseLevel.COMPETENT, "domain": "geopolitical", "accuracy": 0.6, "bias": 0.05, "confidence_calibration": 0.7, "openness": 0.6},
    {"name": "dr_technology", "expertise": ExpertiseLevel.EXPERT, "domain": "technology", "accuracy": 0.65, "bias": 0.08, "confidence_calibration": 0.65, "openness": 0.8},
    {"name": "prof_climate", "expertise": ExpertiseLevel.MASTER, "domain": "climate", "accuracy": 0.78, "bias": -0.01, "confidence_calibration": 0.85, "openness": 0.75},
    {"name": "analyst_social", "expertise": ExpertiseLevel.COMPETENT, "domain": "social", "accuracy": 0.55, "bias": 0.04, "confidence_calibration": 0.6, "openness": 0.7},
    {"name": "dr_health", "expertise": ExpertiseLevel.EXPERT, "domain": "health", "accuracy": 0.7, "bias": -0.03, "confidence_calibration": 0.75, "openness": 0.65},
    {"name": "prof_energy", "expertise": ExpertiseLevel.EXPERT, "domain": "energy", "accuracy": 0.67, "bias": 0.02, "confidence_calibration": 0.7, "openness": 0.6},
    {"name": "analyst_policy", "expertise": ExpertiseLevel.COMPETENT, "domain": "political", "accuracy": 0.58, "bias": 0.06, "confidence_calibration": 0.65, "openness": 0.55},
    {"name": "dr_cyber", "expertise": ExpertiseLevel.EXPERT, "domain": "cyber", "accuracy": 0.62, "bias": -0.04, "confidence_calibration": 0.7, "openness": 0.75},
    {"name": "novice_dynamics", "expertise": ExpertiseLevel.NOVICE, "domain": "general", "accuracy": 0.4, "bias": 0.1, "confidence_calibration": 0.4, "openness": 0.9},
    {"name": "master_systemic", "expertise": ExpertiseLevel.MASTER, "domain": "systemic", "accuracy": 0.8, "bias": 0.0, "confidence_calibration": 0.9, "openness": 0.8},
]


class DelphiMethodSimulator:
    def __init__(self, n_rounds: int = 3, convergence_threshold: float = 0.1,
                 min_experts: int = 3):
        self.n_rounds = n_rounds
        self.convergence_threshold = convergence_threshold
        self.min_experts = min_experts

    def create_panel(self, domain: Optional[str] = None,
                      n_experts: int = 8) -> DelphiPanel:
        experts = []
        templates = EXPERT_TEMPLATES.copy()
        if domain:
            domain_matches = [t for t in templates if t["domain"] == domain]
            other = [t for t in templates if t["domain"] != domain]
            selected = domain_matches[:max(0, min(len(domain_matches), n_experts - 2))]
            remaining = other[:max(0, n_experts - len(selected))]
            selected = selected + remaining
        else:
            selected = templates[:n_experts]
        for t in selected:
            experts.append(SimulatedExpert(
                name=t["name"], expertise=t["expertise"], domain=t["domain"],
                accuracy=t["accuracy"], bias=t["bias"],
                confidence_calibration=t["confidence_calibration"],
                openness_to_evidence=t["openness"],
            ))
        return DelphiPanel(experts=experts, question="", domain=domain or "")

    def run_delphi(self, question: str, true_value: float = 0.5,
                    domain: Optional[str] = None,
                    panel: Optional[DelphiPanel] = None) -> DelphiResult:
        if panel is None:
            panel = self.create_panel(domain)
        panel.question = question
        rounds: List[DelphiRound] = []
        all_estimates = {e.name: [] for e in panel.experts}
        for expert in panel.experts:
            initial = true_value + expert.bias + float(np.random.normal(0, 0.15 * (1 - expert.accuracy)))
            initial = float(max(0.01, min(0.99, initial)))
            expert.current_estimate = initial
            expert.estimate_history.append(initial)
            all_estimates[expert.name].append(initial)
        for round_num in range(self.n_rounds):
            estimates = {e.name: e.current_estimate for e in panel.experts if e.current_estimate is not None}
            confidences = {e.name: e.confidence for e in panel.experts}
            rationales = {e.name: e.rationale for e in panel.experts}
            values = list(estimates.values())
            median = float(np.median(values))
            q1 = float(np.percentile(values, 25))
            q3 = float(np.percentile(values, 75))
            iqr = q3 - q1
            if round_num > 0:
                prev = rounds[-1]
                conv = 1.0 - min(1.0, abs(iqr - prev.iqr) / max(prev.iqr, 0.01))
            else:
                conv = 0.0
            round_result = DelphiRound(
                round_number=round_num + 1, estimates=estimates,
                confidences=confidences, rationales=rationales,
                median=median, q1=q1, q3=q3, iqr=iqr,
                convergence_score=round(conv, 4),
            )
            rounds.append(round_result)
            if round_num < self.n_rounds - 1:
                for expert in panel.experts:
                    self._revise_estimate(expert, median, q1, q3, true_value, round_num)
        final_median = rounds[-1].median
        final_iqr = rounds[-1].iqr
        convergence = self._assess_convergence(rounds)
        dissenters = self._identify_dissenters(panel.experts, final_median)
        n_valid = sum(1 for e in panel.experts if e.current_estimate is not None)
        std_approx = final_iqr / 1.349
        ci_width = 1.96 * std_approx / math.sqrt(max(n_valid, 1))
        ci_lower = max(0.01, final_median - ci_width)
        ci_upper = min(0.99, final_median + ci_width)
        recommended = self._compute_recommended(rounds, panel.experts)
        uncertainty = final_iqr / 2
        narrative = self._generate_narrative(question, rounds, convergence, dissenters)
        return DelphiResult(
            question=question, n_rounds=self.n_rounds,
            final_median=round(final_median, 4), final_iqr=round(final_iqr, 4),
            convergence_status=convergence, rounds=rounds,
            stable_dissenters=dissenters,
            confidence_intervals=(round(ci_lower, 4), round(ci_upper, 4)),
            recommended_estimate=round(recommended, 4),
            uncertainty=round(uncertainty, 4), narrative=narrative,
        )

    def _revise_estimate(self, expert: SimulatedExpert, median: float,
                          q1: float, q3: float, true_value: float,
                          round_num: int) -> None:
        current = expert.current_estimate or 0.5
        if current < q1:
            feedback_pressure = (q1 - current) * expert.openness_to_evidence * 0.3
        elif current > q3:
            feedback_pressure = (q3 - current) * expert.openness_to_evidence * 0.3
        else:
            feedback_pressure = (median - current) * expert.openness_to_evidence * 0.1
        accuracy_pull = (true_value - current) * expert.accuracy * 0.05
        noise = float(np.random.normal(0, max(0.01, 0.05 * (1 - expert.openness_to_evidence))))
        revision = current + feedback_pressure + accuracy_pull + noise
        revision = float(max(0.01, min(0.99, revision)))
        expert.current_estimate = revision
        expert.estimate_history.append(revision)
        expert.confidence = min(0.95, expert.confidence + 0.05 * expert.confidence_calibration)

    def _assess_convergence(self, rounds: List[DelphiRound]) -> ConvergenceStatus:
        if len(rounds) < 2:
            return ConvergenceStatus.CONVERGING
        iqr_trend = [r.iqr for r in rounds]
        if iqr_trend[-1] < self.convergence_threshold:
            return ConvergenceStatus.STABLE
        recent_change = iqr_trend[-1] - iqr_trend[-2] if len(iqr_trend) >= 2 else 0
        if recent_change < -0.02:
            return ConvergenceStatus.CONVERGING
        elif recent_change > 0.02:
            return ConvergenceStatus.DIVERGING
        if len(iqr_trend) >= 3:
            if iqr_trend[-1] > 0.3 and abs(iqr_trend[-1] - iqr_trend[-2]) < 0.01:
                return ConvergenceStatus.POLARIZED
        return ConvergenceStatus.STABLE

    def _identify_dissenters(self, experts: List[SimulatedExpert],
                               median: float) -> List[str]:
        dissenters = []
        for expert in experts:
            if expert.current_estimate is not None:
                if abs(expert.current_estimate - median) > 0.2:
                    if len(expert.estimate_history) >= 2:
                        recent_change = abs(expert.estimate_history[-1] - expert.estimate_history[-2])
                        if recent_change < 0.05:
                            dissenters.append(expert.name)
        return dissenters

    def _compute_recommended(self, rounds: List[DelphiRound],
                               experts: List[SimulatedExpert]) -> float:
        final = rounds[-1]
        expertise_weights = []
        for expert in experts:
            weight = {"novice": 0.5, "competent": 1.0, "expert": 1.5, "master": 2.0}.get(expert.expertise.value, 1.0)
            weight *= (0.5 + 0.5 * expert.accuracy)
            expertise_weights.append(weight)
        total_weight = sum(expertise_weights) or 1.0
        weighted_sum = 0
        for i, expert in enumerate(experts):
            if expert.current_estimate is not None:
                weighted_sum += expert.current_estimate * expertise_weights[i]
        return weighted_sum / total_weight

    def _generate_narrative(self, question: str, rounds: List[DelphiRound],
                              convergence: ConvergenceStatus,
                              dissenters: List[str]) -> str:
        n = len(rounds)
        final = rounds[-1]
        parts = [f"Delphi process ({n} rounds) on: {question}"]
        parts.append(f"Final median estimate: {final.median:.2f} (IQR: {final.iqr:.2f})")
        if convergence == ConvergenceStatus.STABLE:
            parts.append("Panel reached stable consensus.")
        elif convergence == ConvergenceStatus.CONVERGING:
            parts.append("Panel is converging but has not fully stabilized.")
        elif convergence == ConvergenceStatus.DIVERGING:
            parts.append("Panel estimates are diverging — fundamental disagreement.")
        elif convergence == ConvergenceStatus.POLARIZED:
            parts.append("Panel is polarized into distinct camps.")
        if dissenters:
            parts.append(f"Stable dissenters: {', '.join(dissenters)} — their persistence may signal overlooked factors.")
        return " ".join(parts)

    def run_multi_question(self, questions: Dict[str, float],
                              domain: Optional[str] = None) -> Dict[str, DelphiResult]:
        return {q: self.run_delphi(q, tv, domain) for q, tv in questions.items()}