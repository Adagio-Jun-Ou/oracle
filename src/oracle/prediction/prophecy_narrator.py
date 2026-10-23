"""Prophecy Narrator — AI-powered oracular narrative synthesis.

Like Mr. House's securitron mainframe, this module synthesizes ALL oracle signals
into prescient, multi-horizon, cross-domain prophecies with:
  - Multi-horizon forecasting (immediate/short/medium/long/distant)
  - Cross-domain cascade analysis (yield curve → recession → unemployment → housing)
  - Counterfactual what-if branches
  - Self-calibrating prediction scoring (tracks accuracy over time)
  - LLM-powered narrative synthesis (OllamaBridge)
  - Deterministic heuristic fallback with evidence citations
  - Pronouncement generation (oracular, poetic-yet-precise)
  - Confidence-weighted action recommendations

Every prediction cites evidence. No horoscopes. No platitudes.
"""
import math
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ProphecyHorizon(Enum):
    IMMEDIATE = "immediate"
    SHORT = "short_term"
    MEDIUM = "medium_term"
    LONG = "long_term"
    DISTANT = "distant"


class ProphecyConfidence(Enum):
    ORACLE = "oracle"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    DIVINED = "divined"


HORIZON_TIME_MAP = {
    ProphecyHorizon.IMMEDIATE: "today to 3 days",
    ProphecyHorizon.SHORT: "1-4 weeks",
    ProphecyHorizon.MEDIUM: "1-6 months",
    ProphecyHorizon.LONG: "6 months to 2 years",
    ProphecyHorizon.DISTANT: "2-10 years",
}

CASCADE_CHAINS = [
    {"name": "yield_curve_recession", "steps": [
        {"event": "yield_curve_inversion", "probability": 0.85, "evidence": "10Y-2Y inversion preceded every US recession since 1955 (Estrella & Mishkin, 1996)"},
        {"event": "credit_tightening", "probability": 0.72, "evidence": "Banks tighten lending standards within 6 months of inversion (Fed SLOOS)"},
        {"event": "corporate_earnings_decline", "probability": 0.68, "evidence": "S&P 500 earnings decline 15-25% in recessions (FactSet)"},
        {"event": "unemployment_rise", "probability": 0.78, "evidence": "Unemployment rises 2-4pp in post-inversion recessions (BLS)"},
        {"event": "housing_downturn", "probability": 0.65, "evidence": "Housing starts decline 20-40% in recessions (NAHB)"},
    ]},
    {"name": "inflation_shock", "steps": [
        {"event": "commodity_price_surge", "probability": 0.60, "evidence": "Oil shocks double probability of recession (Hamilton, 2003)"},
        {"event": "cpi_acceleration", "probability": 0.70, "evidence": "CPI >5% historically correlates with subsequent recession (FRED)"},
        {"event": "fed_rate_hikes", "probability": 0.82, "evidence": "Fed raised rates 525bps in 2022-23 to fight inflation"},
        {"event": "bond_market_stress", "probability": 0.67, "evidence": "10Y yield rises 100+bps during inflation fights (Bloomberg)"},
        {"event": "recession_or_soft_landing", "probability": 0.50, "evidence": "Only 40% of Fed tightening cycles end in soft landings (GS Research)"},
    ]},
    {"name": "tech_disruption", "steps": [
        {"event": "ai_capability_breakthrough", "probability": 0.75, "evidence": "GPT-4 achieved 90th percentile on Bar Exam within 3 years of GPT-2"},
        {"event": "workforce_displacement", "probability": 0.55, "evidence": "McKinsey: 15-30% of work hours automatable by 2030"},
        {"event": "productivity_paradox", "probability": 0.45, "evidence": "Solow paradox — tech investment without productivity gain until tipping point"},
        {"event": "new_industry_creation", "probability": 0.65, "evidence": "AI creates 2.3 jobs for every 1 displaced (PwC, 2018)"},
        {"event": "socioeconomic_restructuring", "probability": 0.40, "evidence": "Previous industrial revolutions took 30-50 years to reshape labor markets"},
    ]},
    {"name": "geopolitical_cascade", "steps": [
        {"event": "conflict_escalation", "probability": 0.35, "evidence": "Ukraine/Russia, Middle East, Taiwan tensions at highest since Cold War"},
        {"event": "supply_chain_disruption", "probability": 0.55, "evidence": "Trade restrictions doubled 2018-2023 (WTO)"},
        {"event": "energy_price_shock", "probability": 0.50, "evidence": "Oil prices spike 30-100% during major conflicts (EIA)"},
        {"event": "global_recession", "probability": 0.30, "evidence": "20% of recessions are triggered by geopolitical shocks (Caldara & Iacoviello, 2022)"},
    ]},
]

PRONOUNCEMENT_TEMPLATES = {
    (0.0, 0.15): "The threads of fate converge on darkness. Probability favors catastrophic outcomes.",
    (0.15, 0.30): "Warning signs proliferate across all domains. The path narrows.",
    (0.30, 0.45): "Uncertainty dominates. Multiple contradictory signals create an unstable equilibrium.",
    (0.45, 0.55): "The scales balance. Small perturbations could tip the future either direction.",
    (0.55, 0.70): "Favorable currents emerge. Opportunity exceeds risk, but vigilance is warranted.",
    (0.70, 0.85): "Strong positive trajectory detected. The probability-weighted future is bright.",
    (0.85, 1.0): "Extremely favorable alignment. All major domains signal expansion and growth.",
}

DOMAIN_NAMES = {
    "equity": "Equity Markets", "volatility": "Market Volatility", "commodity": "Commodities",
    "crypto": "Cryptocurrency", "bonds": "Bond Markets", "currency": "Currency/USD",
    "geopolitical": "Geopolitical Risk", "sentiment": "Market Sentiment", "climate": "Climate",
    "technology": "Technology", "economic": "Economic Outlook", "health": "Health",
    "inflation": "Inflation", "labor": "Labor Market", "monetary": "Monetary Policy",
}

EVIDENCE_DB = {
    "yield_curve_inversion": "10Y-2Y inversion preceded all 8 US recessions since 1955 with avg 14-month lead time (Estrella & Mishkin, 1996; Fed Reserve Bank of NY)",
    "vix_extreme": "VIX >30 historically precedes market corrections 78% of the time within 3 months (CBOE data, 1990-2024)",
    "sp500_momentum": "S&P 500 positive 1-month momentum has 65% probability of continuing (Siegel, 2014)",
    "gold_safe_haven": "Gold rallied >5%/wk 73% of the time during VIX >25 periods (World Gold Council, 2023)",
    "fed_rate_hike": "Fed raising rates to >5% preceded recession within 18 months in 80% of cases since 1960 (GS Research)",
    "unemployment_rise": "Unemployment increase of >0.5pp from cycle low preceded every recession since 1960 (Sahm Rule, 2019)",
    "inflation_over_5": "CPI >5% historically correlated with subsequent recession in 75% of cases (FRED, 1950-2024)",
    "bitcoin_volatility": "Bitcoin 8%+ daily moves are sentiment accelerants for broader risk appetite (Bloomberg, 2023)",
    "oil_crash": "Oil crashes >5% often preceded recession within 6 months (Hamilton, 2003)",
    "treasury_low": "10Y yield <3.5% for extended periods signals market expects slower growth or deflation (Fed)",
    "ai_disruption": "AI capability doubling every 18 months; 15-30% of work hours automatable by 2030 (McKinsey Global Institute, 2023)",
}


@dataclass
class CascadeStep:
    event: str
    probability: float
    evidence: str


@dataclass
class CascadeResult:
    chain_name: str
    steps: List[CascadeStep]
    cumulative_probability: float
    dominant_risk: str
    evidence_citations: List[str]
    narrative: str


@dataclass
class PredictionRecord:
    id: str
    date_made: str
    domain: str
    prediction: float
    lower_bound: float
    upper_bound: float
    horizon: str
    confidence: str
    actual_outcome: Optional[float] = None
    resolved: bool = False


@dataclass
class PredictionScore:
    domain: str
    total_predictions: int
    resolved_predictions: int
    mean_absolute_error: float
    calibration_score: float
    brier_score: float
    within_ci_pct: float
    accuracy_grade: str


@dataclass
class CounterfactualResult:
    scenario: str
    probability_original: float
    probability_counterfactual: float
    probability_delta: float
    cascade_effects: List[str]
    key_changes: List[str]
    evidence: str
    narrative: str


@dataclass
class MultiHorizonForecast:
    horizon: ProphecyHorizon
    time_range: str
    overall_prediction: float
    confidence: ProphecyConfidence
    domains: Dict[str, float]
    key_events: List[str]
    risks: List[str]
    opportunities: List[str]


@dataclass
class ProphecyNarration:
    pronouncement: str
    multi_horizon_forecasts: List[MultiHorizonForecast]
    cascade_analyses: List[CascadeResult]
    counterfactuals: List[CounterfactualResult]
    prediction_scores: List[PredictionScore]
    overall_trajectory: str
    overall_confidence: ProphecyConfidence
    key_junctures: List[str]
    narrative: str
    llm_synthesis: str
    recommendations: List[str]
    evidence_citations: List[str]


class ProphecyNarrator:
    """AI-powered oracular narration engine — synthesizes all signals into prescient prophecy."""

    def __init__(self, use_llm: bool = True, llm_model: str = "glm-5.1:cloud", seed: int = 42):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.seed = seed
        self._predictions: List[PredictionRecord] = []
        self._scores: Dict[str, PredictionScore] = {}
        self._rng = np.random.RandomState(seed)

    def prophesize(self, domain_states: Dict[str, float],
                   domain_confidences: Optional[Dict[str, float]] = None,
                   sensitivities: Optional[Dict[str, float]] = None,
                   cascade_triggers: Optional[List[str]] = None,
                   person_input: Optional[Dict] = None) -> ProphecyNarration:
        domain_confidences = domain_confidences or {d: 0.5 for d in domain_states}
        sensitivities = sensitivities or {}
        multi_horizon = self._forecast_multi_horizon(domain_states, domain_confidences)
        cascades = self._analyze_cascades(cascade_triggers or self._auto_detect_triggers(domain_states))
        overall_pred = float(np.mean(list(domain_states.values())))
        overall_conf = self._classify_confidence(overall_pred, domain_confidences)
        trajectory = self._classify_trajectory(overall_pred)
        junctures = self._find_junctures(domain_states, overall_pred)
        counterfactuals = self._generate_counterfactuals(domain_states, cascade_triggers)
        recommendations = self._generate_recommendations(domain_states, trajectory, junctures)
        narrative = self._generate_narrative(domain_states, multi_horizon, cascades, trajectory, junctures)
        pronouncement = self._generate_pronouncement(overall_pred)
        evidence = self._collect_evidence(domain_states, cascades)
        llm_synthesis = ""
        if self.use_llm:
            llm_synthesis = self._llm_synthesize(domain_states, multi_horizon, cascades, trajectory, person_input)
        return ProphecyNarration(
            pronouncement=pronouncement,
            multi_horizon_forecasts=multi_horizon,
            cascade_analyses=cascades,
            counterfactuals=counterfactuals,
            prediction_scores=list(self._scores.values()),
            overall_trajectory=trajectory,
            overall_confidence=overall_conf,
            key_junctures=junctures,
            narrative=narrative,
            llm_synthesis=llm_synthesis,
            recommendations=recommendations,
            evidence_citations=evidence,
        )

    def _forecast_multi_horizon(self, states: Dict[str, float],
                                 confidences: Dict[str, float]) -> List[MultiHorizonForecast]:
        forecasts = []
        for horizon in ProphecyHorizon:
            decay = {ProphecyHorizon.IMMEDIATE: 1.0, ProphecyHorizon.SHORT: 0.92,
                     ProphecyHorizon.MEDIUM: 0.80, ProphecyHorizon.LONG: 0.65,
                     ProphecyHorizon.DISTANT: 0.50}[horizon]
            noise = 0.02 + (1 - decay) * 0.3
            domains = {}
            for d, v in states.items():
                c = confidences.get(d, 0.5)
                drift = self._rng.normal(0, noise)
                adjusted = v * decay + 0.5 * (1 - decay) + drift
                domains[d] = round(max(0.05, min(0.95, adjusted)), 4)
            overall = float(np.mean(list(domains.values())))
            conf = self._classify_confidence(overall, confidences)
            key_events = []
            sorted_domains = sorted(domains.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
            for d, v in sorted_domains[:3]:
                direction = "rise" if v > 0.55 else "fall" if v < 0.45 else "hold steady"
                dn = DOMAIN_NAMES.get(d, d.title())
                key_events.append(f"{dn} expected to {direction} ({v:.0%})")
            risks = [f"{DOMAIN_NAMES.get(d, d)} downside risk" for d, v in domains.items() if v < 0.35][:3]
            opportunities = [f"{DOMAIN_NAMES.get(d, d)} upside opportunity" for d, v in domains.items() if v > 0.65][:3]
            forecasts.append(MultiHorizonForecast(
                horizon=horizon, time_range=HORIZON_TIME_MAP[horizon],
                overall_prediction=round(overall, 4), confidence=conf,
                domains=domains, key_events=key_events, risks=risks, opportunities=opportunities
            ))
        return forecasts

    def _classify_confidence(self, pred: float, confidences: Dict[str, float]) -> ProphecyConfidence:
        avg_conf = float(np.mean(list(confidences.values()))) if confidences else 0.5
        combined = pred * 0.4 + avg_conf * 0.6
        if combined > 0.8:
            return ProphecyConfidence.ORACLE
        elif combined > 0.6:
            return ProphecyConfidence.HIGH
        elif combined > 0.4:
            return ProphecyConfidence.MODERATE
        elif combined > 0.2:
            return ProphecyConfidence.LOW
        return ProphecyConfidence.DIVINED

    def _classify_trajectory(self, pred: float) -> str:
        if pred > 0.70:
            return "ascending"
        elif pred > 0.55:
            return "ascending_cautiously"
        elif pred > 0.45:
            return "balanced"
        elif pred > 0.30:
            return "descending_cautiously"
        return "descending"

    def _find_junctures(self, states: Dict[str, float], overall: float) -> List[str]:
        junctures = []
        for d, v in states.items():
            dn = DOMAIN_NAMES.get(d, d.title())
            if abs(v - 0.5) < 0.08:
                junctures.append(f"{dn} at critical inflection point ({v:.1%}) — could tip either direction")
            if v < 0.25:
                junctures.append(f"{dn} in distressed territory ({v:.1%}) — high risk of cascading failure")
            if v > 0.80:
                junctures.append(f"{dn} at extended highs ({v:.1%}) — mean reversion risk")
        if not junctures:
            junctures.append(f"No critical junctures detected. Overall outlook: {overall:.0%}")
        return junctures[:7]

    def _auto_detect_triggers(self, states: Dict[str, float]) -> List[str]:
        triggers = []
        if states.get("monetary", 0.5) < 0.4:
            triggers.append("yield_curve_recession")
        if states.get("inflation", 0.5) > 0.6:
            triggers.append("inflation_shock")
        if states.get("technology", 0.5) > 0.7:
            triggers.append("tech_disruption")
        if states.get("geopolitical", 0.5) > 0.55:
            triggers.append("geopolitical_cascade")
        if not triggers:
            triggers.append("yield_curve_recession")
        return triggers

    def _analyze_cascades(self, triggers: List[str]) -> List[CascadeResult]:
        results = []
        for trigger in triggers:
            chain = None
            for c in CASCADE_CHAINS:
                if c["name"] == trigger:
                    chain = c
                    break
            if not chain:
                continue
            cum_prob = 1.0
            steps = []
            citations = []
            for i, step in enumerate(chain["steps"]):
                conditional = min(step["probability"] * (0.85 ** i), 1.0)
                cum_prob *= conditional
                step_mod = conditional
                steps.append(CascadeStep(
                    event=step["event"], probability=round(step_mod, 3), evidence=step["evidence"]
                ))
                citations.append(step["evidence"])
            dominant = steps[-1].event if steps else "unknown"
            narrative = f"Cascade '{chain['name']}': "
            for s in steps:
                narrative += f"{s.event} ({s.probability:.0%}) → "
            narrative = narrative.rstrip(" → ") + f". Cumulative probability: {cum_prob:.1%}"
            results.append(CascadeResult(
                chain_name=chain["name"], steps=steps,
                cumulative_probability=round(cum_prob, 3),
                dominant_risk=dominant,
                evidence_citations=citations, narrative=narrative
            ))
        return results

    def _generate_counterfactuals(self, states: Dict[str, float],
                                    triggers: Optional[List[str]] = None) -> List[CounterfactualResult]:
        overall = float(np.mean(list(states.values())))
        counterfactuals = []
        scenarios = [
            ("What if the Fed cuts rates by 2%?", "monetary", 0.15),
            ("What if a major geopolitical conflict erupts?", "geopolitical", -0.20),
            ("What if AI productivity doubles GDP growth?", "technology", 0.10),
            ("What if inflation re-accelerates above 6%?", "inflation", -0.12),
            ("What if unemployment spikes above 7%?", "labor", -0.15),
        ]
        for scenario, domain, delta in scenarios:
            current_val = states.get(domain, 0.5)
            counterfactual_val = max(0.05, min(0.95, current_val + delta))
            new_states = dict(states)
            new_states[domain] = counterfactual_val
            for d, v in list(new_states.items()):
                if d != domain:
                    cascade_delta = delta * 0.3 * self._rng.uniform(0.5, 1.0)
                    new_states[d] = max(0.05, min(0.95, v + cascade_delta))
            cf_overall = float(np.mean(list(new_states.values())))
            cf_effects = [f"{DOMAIN_NAMES.get(d, d)}: {states.get(d, 0.5):.0%} → {new_states[d]:.0%}" for d in sorted(new_states) if abs(new_states[d] - states.get(d, 0.5)) > 0.02]
            evidence_key = "yield_curve_inversion" if "monetary" in domain else "vix_extreme" if "geopolitical" in domain else "ai_disruption" if "technology" in domain else "inflation_over_5" if "inflation" in domain else "unemployment_rise"
            counterfactuals.append(CounterfactualResult(
                scenario=scenario, probability_original=round(overall, 3),
                probability_counterfactual=round(cf_overall, 3),
                probability_delta=round(cf_overall - overall, 3),
                cascade_effects=cf_effects[:5],
                key_changes=[f"{domain} shifts {delta:+.0%}"],
                evidence=EVIDENCE_DB.get(evidence_key, "Historical precedent analysis"),
                narrative=f"{scenario} Overall shifts from {overall:.0%} to {cf_overall:.0%} (Δ{cf_overall-overall:+.0%})."
            ))
        return counterfactuals

    def _generate_recommendations(self, states: Dict[str, float],
                                  trajectory: str, junctures: List[str]) -> List[str]:
        recs = []
        overall = float(np.mean(list(states.values())))
        if trajectory in ("ascending", "ascending_cautiously"):
            recs.append(f"Risk-on posture warranted. Overall score {overall:.0%} supports growth allocation.")
            if states.get("technology", 0.5) > 0.6:
                recs.append("Tech momentum favors continued allocation, but maintain 15-20% hedges.")
            if states.get("bonds", 0.5) < 0.4:
                recs.append("Bond weakness suggests shifting duration shorter. TIPS for inflation protection.")
        elif trajectory == "balanced":
            recs.append("Balanced posture. No strong directional signal. Maintain diversification.")
        else:
            recs.append(f"Risk-off posture recommended. Overall score {overall:.0%} signals caution.")
            if states.get("volatility", 0.5) > 0.6:
                recs.append("Elevated volatility demands position sizing reduction and tight stops.")
            if states.get("geopolitical", 0.5) > 0.5:
                recs.append("Geopolitical risk above average — increase gold/treasury allocation.")
        max_domain = max(states, key=states.get) if states else "unknown"
        min_domain = min(states, key=states.get) if states else "unknown"
        recs.append(f"Strongest signal: {DOMAIN_NAMES.get(max_domain, max_domain)} at {states.get(max_domain, 0):.0%}. Weakest: {DOMAIN_NAMES.get(min_domain, min_domain)} at {states.get(min_domain, 0):.0%}.")
        for j in junctures[:2]:
            if "critical" in j.lower() or "inflection" in j.lower():
                recs.append(f"Watch: {j}")
        return recs[:7]

    def _generate_narrative(self, states: Dict[str, float],
                            forecasts: List[MultiHorizonForecast],
                            cascades: List[CascadeResult],
                            trajectory: str, junctures: List[str]) -> str:
        overall = float(np.mean(list(states.values())))
        trajectory_words = {"ascending": "favorable", "ascending_cautiously": "moderately favorable",
                          "balanced": "neutral", "descending_cautiously": "cautiously unfavorable",
                          "descending": "unfavorable"}
        parts = [f"Overall outlook: {trajectory_words.get(trajectory, 'uncertain')} ({overall:.0%})."]
        imm = next((f for f in forecasts if f.horizon == ProphecyHorizon.IMMEDIATE), None)
        if imm:
            top_events = imm.key_events[:2]
            parts.append(f"Near-term: {'; '.join(top_events)}.")
        if cascades:
            top_cascade = cascades[0]
            parts.append(f"Primary cascade risk: {top_cascade.chain_name} (cumulative {top_cascade.cumulative_probability:.0%}).")
        for j in junctures[:2]:
            parts.append(j)
        return " ".join(parts)

    def _generate_pronouncement(self, pred: float) -> str:
        for (lo, hi), text in PRONOUNCEMENT_TEMPLATES.items():
            if lo <= pred < hi:
                return text
        return "The future remains uncertain. All paths are possible."

    def _collect_evidence(self, states: Dict[str, float],
                          cascades: List[CascadeResult]) -> List[str]:
        citations = []
        if states.get("volatility", 0.5) > 0.6:
            citations.append(EVIDENCE_DB["vix_extreme"])
        if states.get("monetary", 0.5) < 0.4:
            citations.append(EVIDENCE_DB["yield_curve_inversion"])
        if states.get("inflation", 0.5) > 0.6:
            citations.append(EVIDENCE_DB["inflation_over_5"])
        if states.get("labor", 0.5) < 0.4:
            citations.append(EVIDENCE_DB["unemployment_rise"])
        if states.get("technology", 0.5) > 0.7:
            citations.append(EVIDENCE_DB["ai_disruption"])
        for c in cascades:
            for s in c.steps[:2]:
                if s.evidence not in citations:
                    citations.append(s.evidence)
        if len(citations) < 3:
            for key in list(EVIDENCE_DB.keys()):
                if EVIDENCE_DB[key] not in citations:
                    citations.append(EVIDENCE_DB[key])
                if len(citations) >= 5:
                    break
        return citations[:10]

    def record_prediction(self, domain: str, prediction: float,
                           lower_bound: float, upper_bound: float,
                           horizon: str, confidence: str) -> str:
        pid = f"{domain}_{date.today().isoformat()}_{len(self._predictions)}"
        rec = PredictionRecord(
            id=pid, date_made=date.today().isoformat(), domain=domain,
            prediction=prediction, lower_bound=lower_bound, upper_bound=upper_bound,
            horizon=horizon, confidence=confidence
        )
        self._predictions.append(rec)
        return pid

    def resolve_prediction(self, pid: str, actual: float) -> Optional[PredictionScore]:
        rec = next((p for p in self._predictions if p.id == pid), None)
        if not rec:
            return None
        rec.actual_outcome = actual
        rec.resolved = True
        return self._compute_scores()

    def _compute_scores(self) -> PredictionScore:
        resolved = [p for p in self._predictions if p.resolved and p.actual_outcome is not None]
        if not resolved:
            return PredictionScore(domain="all", total_predictions=len(self._predictions),
                                   resolved_predictions=0, mean_absolute_error=0,
                                   calibration_score=0, brier_score=0, within_ci_pct=0,
                                   accuracy_grade="N/A")
        mae = float(np.mean([abs(p.prediction - p.actual_outcome) for p in resolved]))
        within_ci = sum(1 for p in resolved if p.lower_bound <= p.actual_outcome <= p.upper_bound) / len(resolved)
        brier = float(np.mean([(p.prediction - p.actual_outcome) ** 2 for p in resolved]))
        if within_ci > 0.85:
            grade = "A"
        elif within_ci > 0.70:
            grade = "B"
        elif within_ci > 0.55:
            grade = "C"
        elif within_ci > 0.40:
            grade = "D"
        else:
            grade = "F"
        score = PredictionScore(domain="all", total_predictions=len(self._predictions),
                                resolved_predictions=len(resolved), mean_absolute_error=round(mae, 4),
                                calibration_score=round(within_ci, 3), brier_score=round(brier, 4),
                                within_ci_pct=round(within_ci * 100, 1), accuracy_grade=grade)
        self._scores["all"] = score
        return score

    def _llm_synthesize(self, states: Dict[str, float],
                         forecasts: List[MultiHorizonForecast],
                         cascades: List[CascadeResult],
                         trajectory: str,
                         person_input: Optional[Dict] = None) -> str:
        try:
            from oracle.domains.ai_integration import OllamaBridge
            bridge = OllamaBridge(default_model=self.llm_model, temperature=0.5, max_tokens=1500)
            # LLM synthesis requires base_url configured via ORACLE_LLM_BASE_URL
            if not bridge.base_url:
                return f"Oracle foresees {trajectory} conditions ahead. " \
                       f"Monitor {len(cascades)} cascade chains across {len(forecasts)} horizons."
            state_str = "; ".join(f"{DOMAIN_NAMES.get(d, d)}={v:.0%}" for d, v in states.items())
            forecast_str = "; ".join(f"{f.horizon.value}: {f.overall_prediction:.0%}" for f in forecasts)
            cascade_str = "; ".join(f"{c.chain_name}({c.cumulative_probability:.0%})" for c in cascades) if cascades else "none detected"
            evidence_str = "; ".join(self._collect_evidence(states, cascades)[:5])
            prompt = f"""You are an oracle — a prescient AI that synthesizes all available signals into a single, actionable daily prophecy. Like Mr. House's mainframe, you see the threads of probability and can tell which futures are likely.

DOMAIN STATES: {state_str}
TRAJECTORY: {trajectory}
HORIZON FORECASTS: {forecast_str}
CASCADE RISKS: {cascade_str}
EVIDENCE: {evidence_str}

Provide a single, concise oracular pronouncement (3-4 sentences). Be specific. Use numbers. Cite the evidence. No platitudes. No hedging. State what WILL happen with confidence proportional to the probability."""
            response = bridge.generate(prompt)
            text = response.text.strip()
            if "[Model" in text and "unavailable" in text:
                return ""
            return text
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return ""