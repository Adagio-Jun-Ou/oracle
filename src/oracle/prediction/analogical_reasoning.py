"""Analogical Reasoning Engine — Historical analog matching for prediction.

Finds relevant historical precedents for current situations and uses their
outcomes to inform predictions. Based on the principle that "history doesn't
repeat, but it rhymes" — structural similarities between situations can
provide calibrated probability estimates.

Key innovations:
1. Multi-dimensional feature matching (not just surface similarity)
2. Structural similarity detection (same causal mechanism, different surface)
3. Temporal decay (recent analogs weighted more)
4. Outcome distance adjustment (how close was the analog's outcome?)
5. Anti-analog detection (when current situation INVERTS a historical pattern)
6. Analog ensemble (weighted combination of multiple matches)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class AnalogQuality(Enum):
    EXACT = "exact"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    ANTI_ANALOG = "anti_analog"


@dataclass
class HistoricalEvent:
    name: str
    year: int
    domain: str
    features: Dict[str, float]
    outcome: Dict[str, float]
    outcome_probability: float
    description: str
    causal_mechanism: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class AnalogMatch:
    event: HistoricalEvent
    similarity: float
    feature_matches: Dict[str, float]
    structural_match: bool
    quality: AnalogQuality
    outcome_adjustment: float
    weight: float
    divergence_points: List[str]


@dataclass
class AnalogicalPrediction:
    target: str
    probability: float
    confidence: float
    analogs: List[AnalogMatch]
    anti_analogs: List[AnalogMatch]
    base_rate_from_analogs: float
    structural_probability: float
    ensemble_probability: float
    key_similarities: List[str]
    key_divergences: List[str]
    narrative: str


HISTORICAL_EVENTS: List[HistoricalEvent] = [
    HistoricalEvent("2008 Financial Crisis", 2008, "financial",
        {"leverage": 0.9, "housing_bubble": 0.85, "credit_growth": 0.8, "derivatives_complexity": 0.9,
         "regulatory_gaps": 0.8, "sentiment_euphoria": 0.75, "liquidity_risk": 0.85, "interconnectedness": 0.8},
        {"market_crash": 0.95, "recession_depth": 0.85, "recovery_speed": 0.4},
        0.95, "Subprime mortgage crisis triggered by housing bubble collapse and amplified by derivatives",
        "credit_bubble_burst", ["financial_crisis", "recession", "housing", "derivatives"]),
    HistoricalEvent("Dot-com Bubble", 2000, "financial",
        {"leverage": 0.5, "housing_bubble": 0.2, "credit_growth": 0.6, "derivatives_complexity": 0.4,
         "regulatory_gaps": 0.5, "sentiment_euphoria": 0.9, "liquidity_risk": 0.5, "interconnectedness": 0.4},
        {"market_crash": 0.8, "recession_depth": 0.5, "recovery_speed": 0.6},
        0.8, "Technology speculation driven by irrational exuberance and novel valuation metrics",
        "speculative_bubble_burst", ["tech", "bubble", "speculation", "valuation"]),
    HistoricalEvent("1970s Stagflation", 1975, "economic",
        {"leverage": 0.4, "inflation": 0.9, "unemployment": 0.8, "energy_shock": 0.9,
         "monetary_overshoot": 0.7, "productivity_slowdown": 0.6, "wage_price_spiral": 0.85, "policy_uncertainty": 0.7},
        {"recession_depth": 0.6, "inflation_persistence": 0.9, "recovery_speed": 0.2},
        0.7, "Oil shock combined with loose monetary policy and structural economic weaknesses",
        "supply_shock_stagflation", ["stagflation", "oil_shock", "inflation", "recession"]),
    HistoricalEvent("COVID-19 Pandemic", 2020, "pandemic",
        {"uncertainty": 0.95, "supply_disruption": 0.9, "demand_shock": 0.85, "policy_response": 0.9,
         "global_coordination": 0.5, "information_asymmetry": 0.8, "behavioral_change": 0.9, "fiscal_expansion": 0.85},
        {"market_crash": 0.7, "recession_depth": 0.9, "recovery_speed": 0.7},
        0.85, "Novel coronavirus causing simultaneous supply and demand shock with unprecedented policy response",
        "pandemic_shock", ["pandemic", "supply_shock", "demand_shock", "policy_response"]),
    HistoricalEvent("1997 Asian Financial Crisis", 1997, "financial",
        {"leverage": 0.85, "currency_mismatch": 0.9, "capital_flow_reversal": 0.95, "bank_weakness": 0.8,
         "current_account_deficit": 0.85, "fixed_exchange_rate": 0.9, "contagion": 0.8, "imf_intervention": 0.7},
        {"market_crash": 0.85, "currency_devaluation": 0.9, "recovery_speed": 0.5},
        0.85, "Sudden capital flow reversal exposing currency mismatches and banking weaknesses",
        "sudden_stop_crisis", ["asian_crisis", "currency", "capital_flows", "contagion"]),
    HistoricalEvent("2010 European Debt Crisis", 2010, "financial",
        {"leverage": 0.8, "sovereign_debt": 0.95, "bank_sovereign_nexus": 0.9, "monetary_union_constraint": 0.85,
         "current_account_deficit": 0.7, "political_fragmentation": 0.7, "austerity": 0.8, "contagion": 0.75},
        {"market_crash": 0.7, "recession_depth": 0.7, "recovery_speed": 0.3},
        0.75, "Sovereign debt crisis amplified by bank-sovereign doom loop and monetary union constraints",
        "sovereign_debt_crisis", ["europe", "sovereign_debt", "austerity", "contagion"]),
    HistoricalEvent("Arab Spring", 2011, "geopolitical",
        {"political_instability": 0.9, "youth_unemployment": 0.85, "food_prices": 0.8, "social_media": 0.7,
         "corruption": 0.85, "authoritarian_rigidity": 0.8, "regional_spillover": 0.8, "military_defection": 0.7},
        {"regime_change": 0.7, "civil_conflict": 0.6, "democratic_transition": 0.3},
        0.7, "Popular uprising triggered by economic hardship, corruption, and enabled by social media",
        "popular_uprising", ["arab_spring", "regime_change", "social_media"]),
    HistoricalEvent("1914 July Crisis", 1914, "geopolitical",
        {"alliance_commitments": 0.95, "military_planning_rigidity": 0.9, "mobilization_pressure": 0.9,
         "miscommunication": 0.85, "leadership_vacuum": 0.7, "arms_race": 0.8, "nationalist_sentiment": 0.8, "crisis_escalation": 0.95},
        {"major_war": 0.95, "regime_change": 0.8, "economic_devastation": 0.9},
        0.95, "Alliance commitments and military planning leading to unintended escalation",
        "alliance_cascade_war", ["wwi", "alliance", "escalation", "military"]),
    HistoricalEvent("Chernobyl Disaster", 1986, "technological",
        {"technology_complexity": 0.9, "regulatory_capture": 0.8, "operator_error": 0.85, "design_flaw": 0.9,
         "information_suppression": 0.85, "safety_culture_failure": 0.9, "cascade_failure": 0.8, "environmental_impact": 0.95},
        {"technological_disaster": 0.95, "regulatory_overhaul": 0.8, "public_trust_loss": 0.9},
        0.9, "Nuclear accident caused by design flaws, operator errors, and regulatory failures",
        "technology_cascade_failure", ["nuclear", "disaster", "technology_failure"]),
    HistoricalEvent("1998 LTCM Crisis", 1998, "financial",
        {"leverage": 0.95, "model_risk": 0.9, "liquidity_mismatch": 0.85, "correlation_breakdown": 0.9,
         "concentration_risk": 0.85, "counterparty_risk": 0.8, "market_stress": 0.7, "hubris": 0.8},
        {"market_disruption": 0.8, "bailout": 0.7, "recovery_speed": 0.6},
        0.8, "Highly leveraged fund brought down by correlation breakdown and liquidity crisis",
        "model_failure_crisis", ["ltcm", "model_risk", "leverage", "correlation_breakdown"]),
    HistoricalEvent("Suez Crisis 1956", 1956, "geopolitical",
        {"nationalism": 0.8, "imperial_retreat": 0.7, "superpower_pressure": 0.85, "energy_chokepoint": 0.8,
         "alliance_tension": 0.7, "military_miscalculation": 0.7, "diplomatic_isolation": 0.6, "economic_sanctions": 0.6},
        {"strategic_defeat": 0.7, "power_realignment": 0.8, "energy_disruption": 0.5},
        0.7, "Failed military intervention exposing declining imperial power and rising superpower influence",
        "imperial_overstretch", ["suez", "imperial_decline", "energy_chokepoint"]),
    HistoricalEvent("1987 Black Monday", 1987, "financial",
        {"leverage": 0.6, "program_trading": 0.85, "portfolio_insurance": 0.9, "valution_stretch": 0.7,
         "liquidity_evaporation": 0.85, "contagion": 0.8, "regulatory_gap": 0.6, "sentiment_reversal": 0.8},
        {"market_crash": 0.95, "recession_depth": 0.1, "recovery_speed": 0.9},
        0.85, "Portfolio insurance and program trading creating feedback loop driving rapid crash",
        "algorithmic_cascade", ["black_monday", "program_trading", "feedback_loop"]),
    HistoricalEvent("Fukushima 2011", 2011, "technological",
        {"natural_hazard": 0.95, "cascading_failure": 0.9, "safety_culture": 0.6, "regulatory_failure": 0.7,
         "design_basis_exceedance": 0.95, "crisis_communication": 0.5, "evacuation_failure": 0.7, "environmental_contamination": 0.9},
        {"technological_disaster": 0.9, "energy_policy_change": 0.8, "public_trust_loss": 0.85},
        0.85, "Earthquake and tsunami exceeding design basis causing nuclear meltdown",
        "compound_natural_technological", ["fukushima", "nuclear", "tsunami", "compound_hazard"]),
    HistoricalEvent("Brexit Referendum 2016", 2016, "geopolitical",
        {"political_populism": 0.85, "economic_discontent": 0.7, "immigration_anxiety": 0.75,
         "elite_disconnect": 0.8, "information_asymmetry": 0.7, "institutional_rigidity": 0.6,
         "media_echo_chamber": 0.75, "surprise_factor": 0.8},
        {"political_shock": 0.9, "currency_devaluation": 0.7, "trade_disruption": 0.6},
        0.75, "Populist backlash against elite consensus and institutional inertia",
        "populist_backlash", ["brexit", "populism", "referendum", "surprise"]),
    HistoricalEvent("Global Financial Crisis Preview 2007", 2007, "financial",
        {"credit_spread_widening": 0.8, "housing_stress": 0.85, "bank_leverage": 0.85,
         "derivatives_opaqueness": 0.9, "liquidity_stress": 0.7, "rating_agency_failure": 0.85,
         "regulatory_oversight_gap": 0.8, "securitization_complexity": 0.85},
        {"market_crash": 0.85, "recession": 0.8, "banking_crisis": 0.9},
        0.85, "Pre-crisis conditions: widening credit spreads and housing stress signals",
        "credit_stress_early_warning", ["financial_crisis", "credit_stress", "early_warning"]),
]


class AnalogicalReasoningEngine:
    def __init__(self, historical_events: Optional[List[HistoricalEvent]] = None):
        self.events = historical_events or HISTORICAL_EVENTS.copy()
        self.feature_weights: Dict[str, float] = {}

    def add_event(self, event: HistoricalEvent) -> None:
        self.events.append(event)

    def find_analogs(self, current_features: Dict[str, float],
                      domain: Optional[str] = None,
                      n_analogs: int = 5) -> List[AnalogMatch]:
        scored = []
        for event in self.events:
            if domain and event.domain != domain:
                continue
            similarity, feature_matches = self._compute_similarity(current_features, event.features)
            structural = self._check_structural_match(current_features, event)
            quality = self._classify_quality(similarity, structural)
            divergence = self._find_divergences(current_features, event.features)
            weight = self._compute_weight(similarity, event, structural)
            adjustment = self._outcome_adjustment(similarity, structural)
            scored.append(AnalogMatch(
                event=event, similarity=round(similarity, 4),
                feature_matches={k: round(v, 4) for k, v in feature_matches.items()},
                structural_match=structural, quality=quality,
                outcome_adjustment=round(adjustment, 4), weight=round(weight, 4),
                divergence_points=divergence,
            ))
        scored.sort(key=lambda x: x.similarity, reverse=True)
        return scored[:n_analogs]

    def predict(self, target: str, current_features: Dict[str, float],
                 domain: Optional[str] = None,
                 n_analogs: int = 5) -> AnalogicalPrediction:
        analogs = self.find_analogs(current_features, domain, n_analogs)
        anti_analogs = [a for a in self.find_analogs(
            {k: 1.0 - v for k, v in current_features.items()}, domain, 3
        ) if a.quality != AnalogQuality.ANTI_ANALOG]
        for a in anti_analogs:
            a.quality = AnalogQuality.ANTI_ANALOG
            a.weight = a.weight * 0.5
        if not analogs:
            return AnalogicalPrediction(
                target=target, probability=0.5, confidence=0.1,
                analogs=[], anti_analogs=anti_analogs,
                base_rate_from_analogs=0.5,
                structural_probability=0.5,
                ensemble_probability=0.5,
                key_similarities=[], key_divergences=[],
                narrative="No strong historical analogs found — high uncertainty",
            )
        base_rate = self._compute_base_rate(analogs)
        structural_prob = self._compute_structural_probability(analogs)
        ensemble_prob = self._compute_ensemble_probability(analogs, anti_analogs)
        key_sims = self._extract_key_similarities(analogs[:3], current_features)
        key_divs = self._extract_key_divergences(analogs[:3], current_features)
        narrative = self._generate_narrative(target, analogs[:3], anti_analogs[:1], ensemble_prob)
        confidence = min(0.9, max(0.1, np.mean([a.similarity for a in analogs[:3]]) * 0.6 + len(analogs) * 0.05))
        return AnalogicalPrediction(
            target=target, probability=round(ensemble_prob, 4),
            confidence=round(confidence, 4), analogs=analogs,
            anti_analogs=anti_analogs,
            base_rate_from_analogs=round(base_rate, 4),
            structural_probability=round(structural_prob, 4),
            ensemble_probability=round(ensemble_prob, 4),
            key_similarities=key_sims, key_divergences=key_divs,
            narrative=narrative,
        )

    def _compute_similarity(self, current: Dict[str, float],
                              historical: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        common_keys = set(current.keys()) & set(historical.keys())
        if not common_keys:
            return 0.0, {}
        feature_matches = {}
        weighted_diffs = []
        for key in common_keys:
            c_val = current.get(key, 0.5)
            h_val = historical.get(key, 0.5)
            diff = abs(c_val - h_val)
            match_score = 1.0 - diff
            feature_matches[key] = match_score
            feature_importance = self.feature_weights.get(key, 1.0)
            weighted_diffs.append(diff * feature_importance)
        similarity = 1.0 - np.mean(weighted_diffs)
        return max(0.0, min(1.0, similarity)), feature_matches

    def _check_structural_match(self, current: Dict[str, float],
                                 event: HistoricalEvent) -> bool:
        causal_keywords = set(event.causal_mechanism.lower().replace("_", " ").split())
        current_high = set()
        for k, v in current.items():
            if v > 0.7:
                current_high.update(k.lower().replace("_", " ").split())
        overlap = len(current_high & causal_keywords)
        return overlap >= 2

    def _classify_quality(self, similarity: float, structural: bool) -> AnalogQuality:
        if similarity > 0.85 and structural:
            return AnalogQuality.EXACT
        elif similarity > 0.7:
            return AnalogQuality.STRONG
        elif similarity > 0.5:
            return AnalogQuality.MODERATE
        elif similarity > 0.3:
            return AnalogQuality.WEAK
        return AnalogQuality.WEAK

    def _find_divergences(self, current: Dict[str, float],
                           historical: Dict[str, float]) -> List[str]:
        common_keys = set(current.keys()) & set(historical.keys())
        divergences = []
        for key in common_keys:
            diff = abs(current.get(key, 0.5) - historical.get(key, 0.5))
            if diff > 0.3:
                divergences.append(f"{key}: {current.get(key, 0.5):.2f} vs {historical.get(key, 0.5):.2f}")
        return divergences[:5]

    def _compute_weight(self, similarity: float, event: HistoricalEvent,
                         structural: bool) -> float:
        base_weight = similarity ** 2
        from datetime import datetime
        age = max(1, datetime.now().year - event.year)
        recency_decay = math.exp(-0.01 * age)
        structural_bonus = 1.5 if structural else 1.0
        return base_weight * recency_decay * structural_bonus

    def _outcome_adjustment(self, similarity: float, structural: bool) -> float:
        base = similarity * 0.5
        if structural:
            base *= 1.3
        return min(1.0, base)

    def _compute_base_rate(self, analogs: List[AnalogMatch]) -> float:
        if not analogs:
            return 0.5
        weights = [a.weight for a in analogs]
        outcomes = [a.event.outcome_probability for a in analogs]
        total_weight = sum(weights) or 1.0
        return sum(w * o for w, o in zip(weights, outcomes)) / total_weight

    def _compute_structural_probability(self, analogs: List[AnalogMatch]) -> float:
        structural_matches = [a for a in analogs if a.structural_match]
        if not structural_matches:
            return self._compute_base_rate(analogs) * 0.8 + 0.1
        weights = [a.weight for a in structural_matches]
        outcomes = [a.event.outcome_probability for a in structural_matches]
        total_weight = sum(weights) or 1.0
        return sum(w * o for w, o in zip(weights, outcomes)) / total_weight

    def _compute_ensemble_probability(self, analogs: List[AnalogMatch],
                                        anti_analogs: List[AnalogMatch]) -> float:
        if not analogs:
            return 0.5
        analog_weight_sum = sum(a.weight for a in analogs) or 1.0
        analog_prob = sum(a.weight * a.event.outcome_probability for a in analogs) / analog_weight_sum
        if hasattr(analogs[0], 'outcome_adjustment'):
            adjustments = [a.outcome_adjustment for a in analogs]
            avg_adj = float(np.mean(adjustments))
            analog_prob = analog_prob * (0.7 + 0.3 * avg_adj)
        if anti_analogs:
            anti_weight_sum = sum(a.weight for a in anti_analogs) or 1.0
            anti_prob = sum(a.weight * (1.0 - a.event.outcome_probability) for a in anti_analogs) / anti_weight_sum
            analog_prob = analog_prob * 0.8 + (1.0 - anti_prob) * 0.2
        return max(0.01, min(0.99, analog_prob))

    def _extract_key_similarities(self, analogs: List[AnalogMatch],
                                    current: Dict[str, float]) -> List[str]:
        sims = []
        seen = set()
        for analog in analogs:
            for feat, score in analog.feature_matches.items():
                if score > 0.7 and feat not in seen:
                    sims.append(f"{feat} (similar to {analog.event.name}: {score:.0%})")
                    seen.add(feat)
                if len(sims) >= 5:
                    break
        return sims

    def _extract_key_divergences(self, analogs: List[AnalogMatch],
                                   current: Dict[str, float]) -> List[str]:
        divs = []
        for analog in analogs:
            for div in analog.divergence_points[:2]:
                if div not in divs:
                    divs.append(f"{div} (divergence from {analog.event.name})")
            if len(divs) >= 5:
                break
        return divs

    def _generate_narrative(self, target: str, analogs: List[AnalogMatch],
                              anti_analogs: List[AnalogMatch], prob: float) -> str:
        if not analogs:
            return f"No strong historical precedent for {target} — high uncertainty. Proceed with caution."
        top = analogs[0]
        lines = [f"Situation most analogous to {top.event.name} ({top.event.year}) with {top.similarity:.0%} feature similarity."]
        if len(analogs) > 1:
            lines.append(f"Second analog: {analogs[1].event.name} ({analogs[1].similarity:.0%} similarity).")
        if top.structural_match:
            lines.append(f"Structural causal mechanism match: {top.event.causal_mechanism}")
        if anti_analogs:
            lines.append(f"Anti-analog warning: {anti_analogs[0].event.name} suggests opposite outcome possible.")
        prob_label = "likely" if prob > 0.65 else ("unlikely" if prob < 0.35 else "uncertain")
        lines.append(f"Predicted outcome: {prob} ({prob_label}).")
        return " ".join(lines)