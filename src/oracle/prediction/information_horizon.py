import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class QuestionCategory(Enum):
    DAILY_MARKET_MOVE = "daily_market_move"
    QUARTERLY_EARNINGS = "quarterly_earnings"
    RECESSION_ONSET = "recession_onset"
    GEOPOLITICAL_CRISIS = "geopolitical_crisis"
    CLIMATE_TIPPING = "climate_tipping"
    TECHNOLOGY_DISRUPTION = "technology_disruption"
    PANDEMIC_OUTCOME = "pandemic_outcome"
    ELECTION_OUTCOME = "election_outcome"
    REGULATORY_CHANGE = "regulatory_change"
    HOUSING_MARKET_SHIFT = "housing_market_shift"


RESOLUTION_PROFILES: Dict[QuestionCategory, Dict] = {
    QuestionCategory.DAILY_MARKET_MOVE: {
        "min_months": 0.03,
        "max_months": 0.07,
        "typical_months": 0.03,
        "base_certainty": 0.10,
        "information_velocity": 0.90,
        "resolution_shape": "exponential",
    },
    QuestionCategory.QUARTERLY_EARNINGS: {
        "min_months": 3.0,
        "max_months": 3.0,
        "typical_months": 3.0,
        "base_certainty": 0.05,
        "information_velocity": 0.80,
        "resolution_shape": "step",
    },
    QuestionCategory.RECESSION_ONSET: {
        "min_months": 6.0,
        "max_months": 18.0,
        "typical_months": 12.0,
        "base_certainty": 0.15,
        "information_velocity": 0.40,
        "resolution_shape": "sigmoid",
    },
    QuestionCategory.GEOPOLITICAL_CRISIS: {
        "min_months": 1.0,
        "max_months": 24.0,
        "typical_months": 6.0,
        "base_certainty": 0.10,
        "information_velocity": 0.30,
        "resolution_shape": "sigmoid",
    },
    QuestionCategory.CLIMATE_TIPPING: {
        "min_months": 120.0,
        "max_months": 600.0,
        "typical_months": 300.0,
        "base_certainty": 0.05,
        "information_velocity": 0.05,
        "resolution_shape": "linear",
    },
    QuestionCategory.TECHNOLOGY_DISRUPTION: {
        "min_months": 12.0,
        "max_months": 60.0,
        "typical_months": 30.0,
        "base_certainty": 0.10,
        "information_velocity": 0.25,
        "resolution_shape": "sigmoid",
    },
    QuestionCategory.PANDEMIC_OUTCOME: {
        "min_months": 3.0,
        "max_months": 12.0,
        "typical_months": 6.0,
        "base_certainty": 0.10,
        "information_velocity": 0.50,
        "resolution_shape": "sigmoid",
    },
    QuestionCategory.ELECTION_OUTCOME: {
        "min_months": 1.0,
        "max_months": 6.0,
        "typical_months": 3.0,
        "base_certainty": 0.10,
        "information_velocity": 0.60,
        "resolution_shape": "step",
    },
    QuestionCategory.REGULATORY_CHANGE: {
        "min_months": 6.0,
        "max_months": 36.0,
        "typical_months": 18.0,
        "base_certainty": 0.08,
        "information_velocity": 0.20,
        "resolution_shape": "sigmoid",
    },
    QuestionCategory.HOUSING_MARKET_SHIFT: {
        "min_months": 6.0,
        "max_months": 24.0,
        "typical_months": 12.0,
        "base_certainty": 0.10,
        "information_velocity": 0.35,
        "resolution_shape": "sigmoid",
    },
}


def _certainty_curve(
    months: float,
    profile: Dict,
    data_quality: float,
) -> float:
    shape = profile.get("resolution_shape", "sigmoid")
    base = profile["base_certainty"]
    velocity = profile["information_velocity"] * max(data_quality, 0.01)
    typical = profile["typical_months"]
    uncertainty_mass = 1.0 - base

    effective_typical = typical / max(velocity, 0.01)

    if shape == "exponential":
        rate = 5.0 / max(effective_typical, 0.001)
        certainty = base + uncertainty_mass * (1.0 - math.exp(-rate * months))
    elif shape == "step":
        midpoint = effective_typical * 0.95
        steepness = 20.0 * max(velocity, 0.1) + 2.0
        certainty = base + uncertainty_mass / (1.0 + math.exp(-steepness * (months - midpoint)))
    elif shape == "linear":
        max_m = profile["max_months"] / max(velocity, 0.01)
        rate = 1.0 / max(max_m, 1.0)
        certainty = base + uncertainty_mass * min(rate * months, 1.0)
    else:
        midpoint = effective_typical * 0.5
        steepness = 8.0 / max(effective_typical, 0.01)
        certainty = base + uncertainty_mass / (1.0 + math.exp(-steepness * (months - midpoint)))

    return min(max(certainty, 0.0), 1.0)


def _find_time_for_certainty(
    target_certainty: float,
    profile: Dict,
    data_quality: float,
    max_months: float = 10000.0,
) -> Optional[float]:
    if target_certainty <= profile["base_certainty"]:
        return 0.0

    low = 0.0
    high = max_months
    for _ in range(200):
        mid = (low + high) / 2.0
        c = _certainty_curve(mid, profile, data_quality)
        if abs(c - target_certainty) < 1e-7:
            return mid
        if c < target_certainty:
            low = mid
        else:
            high = mid

    final_c = _certainty_curve(high, profile, data_quality)
    if final_c >= target_certainty - 1e-5:
        return high
    return None


@dataclass
class CertaintyHorizon:
    question: str
    category: str
    current_certainty: float
    resolution_times: Dict[float, float] = field(default_factory=dict)
    information_velocity: float = 0.0
    recommendation: str = ""

    def time_to_threshold(self, threshold: float) -> Optional[float]:
        if threshold in self.resolution_times:
            return self.resolution_times[threshold]
        for t in sorted(self.resolution_times.keys()):
            if t >= threshold:
                return self.resolution_times[t]
        return None

    def is_resolvable_before(self, months: float) -> bool:
        for threshold, t in self.resolution_times.items():
            if t <= months:
                return True
        return False

    def certainty_at_time(self, months: float) -> float:
        if not self.resolution_times:
            return self.current_certainty
        sorted_thresholds = sorted(self.resolution_times.keys(), reverse=True)
        for threshold in sorted_thresholds:
            if months >= self.resolution_times[threshold]:
                return threshold
        return self.current_certainty


@dataclass
class DecisionTimingResult:
    optimal_decision_time: float
    wait_value: float
    current_info_value: float
    diminishing_returns_point: float
    net_value_of_waiting: float
    risk_of_waiting: float
    recommendation: str
    uncertainty_breakdown: List[Dict] = field(default_factory=list)

    def should_wait(self) -> bool:
        return self.net_value_of_waiting > 0 and self.optimal_decision_time > 0


class InformationHorizonMapper:
    def __init__(
        self,
        custom_profiles: Optional[Dict[QuestionCategory, Dict]] = None,
        default_data_quality: float = 0.7,
        decay_rate: float = 0.02,
    ):
        self.profiles = dict(RESOLUTION_PROFILES)
        if custom_profiles:
            for cat, profile in custom_profiles.items():
                self.profiles[cat] = {**self.profiles.get(cat, {}), **profile}
        self.default_data_quality = default_data_quality
        self.decay_rate = decay_rate
        self._cache: Dict[str, CertaintyHorizon] = {}

    def estimate_resolution_time(
        self,
        question_category: QuestionCategory,
        current_data_quality: Optional[float] = None,
    ) -> Dict[str, float]:
        dq = current_data_quality if current_data_quality is not None else self.default_data_quality
        dq = float(np.clip(dq, 0.01, 1.0))
        profile = self.profiles.get(question_category, self.profiles[QuestionCategory.GEOPOLITICAL_CRISIS])

        result: Dict[str, float] = {
            "p50_time": _find_time_for_certainty(0.50, profile, dq) or float("inf"),
            "p80_time": _find_time_for_certainty(0.80, profile, dq) or float("inf"),
            "p95_time": _find_time_for_certainty(0.95, profile, dq) or float("inf"),
            "typical_resolution": profile["typical_months"],
            "min_resolution": profile["min_months"],
            "max_resolution": profile["max_months"],
            "information_velocity": profile["information_velocity"] * dq,
            "current_certainty": profile["base_certainty"],
        }

        logger.info(
            "Resolution estimate for %s (dq=%.2f): p50=%.1fm p80=%.1fm p95=%.1fm",
            question_category.value, dq,
            result["p50_time"], result["p80_time"], result["p95_time"],
        )
        return result

    def certainty_timeline(
        self,
        uncertainties: List[Dict],
    ) -> List[CertaintyHorizon]:
        horizons: List[CertaintyHorizon] = []

        for unc in uncertainties:
            question = unc.get("question", "unknown")
            cat_str = unc.get("category", "geopolitical_crisis")
            dq = unc.get("data_quality", self.default_data_quality)
            dq = float(np.clip(dq, 0.01, 1.0))

            try:
                category = QuestionCategory(cat_str)
            except ValueError:
                logger.warning("Unknown category %s, falling back to geopolitical_crisis", cat_str)
                category = QuestionCategory.GEOPOLITICAL_CRISIS

            profile = self.profiles[category]
            velocity = profile["information_velocity"] * dq
            base_certainty = profile["base_certainty"]

            resolution_times: Dict[float, float] = {}
            for threshold in [0.50, 0.80, 0.95, 0.99]:
                t = _find_time_for_certainty(threshold, profile, dq)
                if t is not None:
                    resolution_times[threshold] = round(t, 3)

            current_certainty = _certainty_curve(0.0, profile, dq)

            if current_certainty >= 0.95:
                recommendation = "Sufficient certainty already — decide now"
            elif not resolution_times or min(resolution_times.values(), default=float("inf")) > 120:
                recommendation = "Very long horizon — incremental decisions recommended"
            else:
                p50 = resolution_times.get(0.50, float("inf"))
                if p50 < 1:
                    recommendation = "Rapid resolution expected — brief wait highly valuable"
                elif p50 < 6:
                    recommendation = "Moderate resolution horizon — short wait valuable"
                elif p50 < 24:
                    recommendation = "Extended resolution horizon — staged decisions advisable"
                else:
                    recommendation = "Very long resolution horizon — decide with current info, plan to revisit"

            horizon = CertaintyHorizon(
                question=question,
                category=cat_str,
                current_certainty=round(current_certainty, 4),
                resolution_times=resolution_times,
                information_velocity=round(velocity, 4),
                recommendation=recommendation,
            )
            horizons.append(horizon)
            self._cache[question] = horizon

        logger.info("Computed certainty timelines for %d uncertainties", len(horizons))
        return horizons

    def decision_timing_score(
        self,
        decision_deadline: float,
        uncertainties: List[Dict],
        decision_cost_of_delay: float = 0.01,
    ) -> DecisionTimingResult:
        horizons = self.certainty_timeline(uncertainties)

        if not horizons:
            return DecisionTimingResult(
                optimal_decision_time=0.0,
                wait_value=0.0,
                current_info_value=0.0,
                diminishing_returns_point=0.0,
                net_value_of_waiting=0.0,
                risk_of_waiting=0.0,
                recommendation="No uncertainties to consider — decide now",
            )

        effective_deadline = min(decision_deadline, 1200.0)
        n_points = 200
        time_points = np.linspace(0, effective_deadline, n_points)
        info_values = np.zeros(n_points)

        for i, t in enumerate(time_points):
            for horizon in horizons:
                cat_str = horizon.category
                try:
                    cat = QuestionCategory(cat_str)
                except ValueError:
                    cat = QuestionCategory.GEOPOLITICAL_CRISIS

                dq = self.default_data_quality
                for unc in uncertainties:
                    if unc.get("question") == horizon.question:
                        dq = unc.get("data_quality", self.default_data_quality)
                        dq = float(np.clip(dq, 0.01, 1.0))
                        break

                profile = self.profiles[cat]
                certainty_now = _certainty_curve(t, profile, dq)
                certainty_0 = _certainty_curve(0.0, profile, dq)
                certainty_gain = certainty_now - certainty_0
                info_values[i] += certainty_gain * horizon.information_velocity

        info_values /= max(len(horizons), 1)

        delay_costs = self.decay_rate * time_points * decision_cost_of_delay
        net_values = info_values - delay_costs

        best_idx = int(np.argmax(net_values))
        optimal_time = float(time_points[best_idx])

        diminishing_idx = n_points - 1
        for i in range(1, n_points):
            marginal_info = info_values[i] - info_values[i - 1]
            marginal_cost = delay_costs[i] - delay_costs[i - 1]
            if marginal_info < marginal_cost and info_values[i] > 0:
                diminishing_idx = i
                break
        diminishing_point = float(time_points[diminishing_idx])

        current_value = float(info_values[0])
        wait_value = float(info_values[best_idx]) - current_value
        risk = float(self.decay_rate * optimal_time * decision_cost_of_delay)

        uncertainty_breakdown: List[Dict] = []
        for horizon in horizons:
            p50 = horizon.resolution_times.get(0.50, float("inf"))
            uncertainty_breakdown.append({
                "question": horizon.question,
                "category": horizon.category,
                "current_certainty": horizon.current_certainty,
                "p50_months": round(p50, 2),
                "information_velocity": horizon.information_velocity,
                "resolvable_by_deadline": p50 <= decision_deadline,
            })

        if optimal_time <= 0.1:
            recommendation = "Decide now — no significant information gain from waiting"
        elif optimal_time >= decision_deadline * 0.9:
            recommendation = "Wait until near deadline — information accumulates slowly"
        elif diminishing_point < decision_deadline:
            recommendation = f"Wait ~{optimal_time:.1f} months for key information, then decide before {diminishing_point:.1f}m"
        else:
            recommendation = f"Optimal decision at ~{optimal_time:.1f} months"

        result = DecisionTimingResult(
            optimal_decision_time=round(optimal_time, 3),
            wait_value=round(wait_value, 4),
            current_info_value=round(current_value, 4),
            diminishing_returns_point=round(diminishing_point, 3),
            net_value_of_waiting=round(float(net_values[best_idx]), 4),
            risk_of_waiting=round(risk, 4),
            recommendation=recommendation,
            uncertainty_breakdown=uncertainty_breakdown,
        )

        logger.info(
            "Decision timing: optimal=%.1fm, wait_value=%.3f, risk=%.3f, %s",
            optimal_time, wait_value, risk, recommendation,
        )
        return result

    def value_of_information(
        self,
        uncertainty: Dict,
        decision_context: Dict,
    ) -> Dict:
        question = uncertainty.get("question", "unknown")
        cat_str = uncertainty.get("category", "geopolitical_crisis")
        dq = uncertainty.get("data_quality", self.default_data_quality)
        dq = float(np.clip(dq, 0.01, 1.0))

        try:
            category = QuestionCategory(cat_str)
        except ValueError:
            category = QuestionCategory.GEOPOLITICAL_CRISIS

        profile = self.profiles[category]

        stake = decision_context.get("stake", 100.0)
        reversible = decision_context.get("reversible", True)
        alternatives = decision_context.get("alternatives", 2)
        deadline = decision_context.get("deadline_months", 12.0)
        cost_of_wrong = decision_context.get("cost_of_wrong_decision", stake * 0.5)
        cost_of_delay = decision_context.get("cost_of_delay_per_month", stake * 0.01)

        base_certainty = profile["base_certainty"]
        current_ev = -cost_of_wrong * (1.0 - base_certainty)

        certainty_if_wait = _certainty_curve(deadline, profile, dq)
        ev_if_wait = -cost_of_wrong * (1.0 - certainty_if_wait) - cost_of_delay * deadline

        p50_time = _find_time_for_certainty(0.50, profile, dq) or float("inf")
        certainty_at_p50 = _certainty_curve(min(p50_time, deadline), profile, dq)
        ev_at_p50 = -cost_of_wrong * (1.0 - certainty_at_p50) - cost_of_delay * min(p50_time, deadline)

        voi_full = current_ev - ev_if_wait
        voi_at_p50 = current_ev - ev_at_p50

        if not reversible:
            reversibility_discount = 0.5
        else:
            reversibility_discount = 1.0

        voi_adjusted = voi_full * reversibility_discount

        if alternatives > 1:
            voi_adjusted *= 1.0 + 0.1 * math.log2(max(alternatives, 2))

        expected_improvement = certainty_if_wait - base_certainty
        cost_ratio = (cost_of_delay * deadline) / max(cost_of_wrong * expected_improvement, 1e-10)

        if voi_adjusted > 0 and cost_ratio < 1.0:
            recommendation = "Waiting for information is worthwhile — high VoI relative to delay cost"
        elif voi_adjusted > cost_of_delay * deadline:
            recommendation = "Moderate VoI — consider short wait for key resolution"
        else:
            recommendation = "Low VoI — delay costs outweigh information gains, decide now"

        result = {
            "question": question,
            "category": cat_str,
            "value_of_information_full": round(voi_adjusted, 4),
            "value_of_information_p50": round(voi_at_p50, 4),
            "current_expected_value": round(current_ev, 4),
            "wait_expected_value": round(ev_if_wait, 4),
            "p50_expected_value": round(ev_at_p50, 4),
            "current_certainty": round(base_certainty, 4),
            "certainty_at_deadline": round(certainty_if_wait, 4),
            "cost_of_wrong_decision": round(cost_of_wrong, 4),
            "delay_cost_total": round(cost_of_delay * deadline, 4),
            "cost_ratio": round(cost_ratio, 4),
            "reversibility_factor": round(reversibility_discount, 4),
            "recommendation": recommendation,
        }

        logger.info(
            "VoI for '%s': %.2f (adjusted), current_ev=%.2f, wait_ev=%.2f, %s",
            question[:40], voi_adjusted, current_ev, ev_if_wait, recommendation[:50],
        )
        return result

    def batch_value_of_information(
        self,
        uncertainties: List[Dict],
        decision_context: Dict,
    ) -> Dict:
        individual_results: List[Dict] = []
        total_voi = 0.0
        best_voi_question = ""
        best_voi_value = -float("inf")

        for unc in uncertainties:
            result = self.value_of_information(unc, decision_context)
            individual_results.append(result)
            voi = result["value_of_information_full"]
            total_voi += voi
            if voi > best_voi_value:
                best_voi_value = voi
                best_voi_question = result["question"]

        deadline = decision_context.get("deadline_months", 12.0)
        resolved_by_deadline = 0
        for unc in uncertainties:
            cat_str = unc.get("category", "geopolitical_crisis")
            dq = float(np.clip(unc.get("data_quality", self.default_data_quality), 0.01, 1.0))
            try:
                category = QuestionCategory(cat_str)
            except ValueError:
                category = QuestionCategory.GEOPOLITICAL_CRISIS
            profile = self.profiles[category]
            p80 = _find_time_for_certainty(0.80, profile, dq) or float("inf")
            if p80 <= deadline:
                resolved_by_deadline += 1

        priorities = sorted(
            individual_results,
            key=lambda r: r["value_of_information_full"],
            reverse=True,
        )

        return {
            "total_voi": round(total_voi, 4),
            "average_voi": round(total_voi / max(len(uncertainties), 1), 4),
            "best_voi_question": best_voi_question,
            "best_voi_value": round(best_voi_value, 4),
            "n_uncertainties": len(uncertainties),
            "n_resolved_by_deadline": resolved_by_deadline,
            "resolution_fraction": round(resolved_by_deadline / max(len(uncertainties), 1), 4),
            "prioritized": [p["question"] for p in priorities],
            "individual_results": individual_results,
        }