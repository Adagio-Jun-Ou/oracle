"""Blind Spot Detector — Identifies what the Oracle CANNOT see.

The greatest danger in prediction is not what we get wrong, but what we don't
even think to predict. This module systematically surfaces blind spots:

1. Unmodeled Domains: domains with no model coverage
2. Assumption Dependencies: hidden assumptions shared across models
3. Edge Case Blindness: regions where no training data exists
4. Temporal Blind Spots: time horizons where all models degrade
5. Correlation Blindness: relationships between domains we never model together
6. Narrative Traps: stories so compelling they exclude contradictory evidence

Evidence:
- Kahneman (2011): "What You See Is All There Is" (WYSIATI) bias
- Tetlock (2015): Foxes (many models) outperform hedgehogs (one big idea) because
  they see more of the possibility space
- Taleb (2007): Black swans come from what we don't know we don't know
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BlindSpotCategory(Enum):
    UNMODELED_DOMAIN = "unmodeled_domain"
    SHARED_ASSUMPTION = "shared_assumption"
    EDGE_CASE = "edge_case"
    TEMPORAL_HORIZON = "temporal_horizon"
    CORRELATION_GAP = "correlation_gap"
    NARRATIVE_TRAP = "narrative_trap"
    DATA_DESERT = "data_desert"
    INTERACTION_BLINDNESS = "interaction_blindness"


class BlindSpotSeverity(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BlindSpot:
    category: BlindSpotCategory
    severity: BlindSpotSeverity
    description: str
    domains: List[str]
    affected_models: List[str]
    probability_impact: float
    recommendation: str
    evidence: str


@dataclass
class BlindSpotReport:
    total_blind_spots: int
    critical_count: int
    blind_spots: List[BlindSpot]
    unmodeled_domains: List[str]
    shared_assumptions: List[str]
    coverage_score: float
    omniscience_gap: float
    narrative: str


ORACLE_DOMAINS = {
    "equity", "volatility", "commodity", "crypto", "bonds", "currency",
    "geopolitical", "sentiment", "climate", "technology", "economic",
    "health", "inflation", "labor", "monetary",
}

ORACLE_MODELS = {
    "causal_event", "signal_fusion", "cross_impact", "black_swan", "regime_shift",
    "superforecast", "daily_oracle", "cascade_forecast", "butterfly_effect",
    "analogical_reasoning", "delphi", "morphological", "collective_intelligence",
    "horizon_scanning", "precursor_pattern", "counterfactual", "prophecy",
    "stress_testing", "red_team", "ensemble_calibrator", "temporal_consistency",
    "forecast_decomposition", "scenario_probability",
}

KNOWN_INTERACTIONS = {
    ("equity", "volatility"): True,
    ("equity", "economic"): True,
    ("geopolitical", "commodity"): True,
    ("inflation", "monetary"): True,
    ("labor", "inflation"): True,
    ("sentiment", "equity"): True,
    ("bonds", "inflation"): True,
    ("currency", "economic"): True,
    ("technology", "equity"): True,
    ("crypto", "sentiment"): True,
    ("climate", "commodity"): True,
    ("health", "economic"): True,
    ("volatility", "bonds"): True,
    ("geopolitical", "volatility"): True,
}

NARRATIVE_TRAPS = {
    "bullish_bias": "Markets tend to rise over time, anchoring forecasters to optimistic scenarios",
    "recency_bias": "Recent events are overweighted, extrapolating current trends indefinitely",
    "normalcy_bias": "Assuming tomorrow will resemble today, underestimating regime changes",
    "expert_overconfidence": "Domain experts overestimate their certainty within specialty",
    "single_scenario": "Focusing on one outcome rather than the full distribution",
    "anchoring": "Latching onto initial estimates and adjusting insufficiently",
    "confirmation_seeking": "Seeking evidence that confirms existing beliefs",
    "precision_illusion": "Confusing precise numbers with accurate predictions",
}


class BlindSpotDetector:
    def __init__(self, covered_domains: Optional[Set[str]] = None,
                 model_domain_map: Optional[Dict[str, Set[str]]] = None):
        self.covered_domains = covered_domains or ORACLE_DOMAINS
        self.model_domain_map = model_domain_map or {}

    def detect(self, predictions: Optional[Dict[str, Dict]] = None,
               domain_states: Optional[Dict[str, float]] = None,
               active_models: Optional[List[str]] = None) -> BlindSpotReport:
        blind_spots = []
        active = set(active_models or ORACLE_MODELS)
        unmodeled = self._find_unmodeled_domains(predictions, domain_states)
        shared = self._find_shared_assumptions(active)
        edge_cases = self._find_edge_cases(domain_states)
        temporal = self._find_temporal_blind_spots()
        corr_gaps = self._find_correlation_gaps(predictions, domain_states)
        narratives = self._find_narrative_traps(predictions)
        data_deserts = self._find_data_deserts(predictions)
        interaction_blind = self._find_interaction_blindness(domain_states)
        blind_spots.extend(unmodeled)
        blind_spots.extend(shared)
        blind_spots.extend(edge_cases)
        blind_spots.extend(temporal)
        blind_spots.extend(corr_gaps)
        blind_spots.extend(narratives)
        blind_spots.extend(data_deserts)
        blind_spots.extend(interaction_blind)
        blind_spots.sort(key=lambda b: -b.probability_impact)
        critical = sum(1 for b in blind_spots if b.severity == BlindSpotSeverity.CRITICAL)
        coverage = self._compute_coverage(active, self.covered_domains)
        omniscience_gap = 1.0 - coverage + sum(b.probability_impact for b in blind_spots[:5]) * 0.1
        omniscience_gap = min(1.0, max(0.0, omniscience_gap))
        narrative = self._generate_narrative(blind_spots, unmodeled, coverage)
        return BlindSpotReport(
            total_blind_spots=len(blind_spots),
            critical_count=critical,
            blind_spots=blind_spots[:20],
            unmodeled_domains=[b.description for b in unmodeled],
            shared_assumptions=[b.description for b in shared],
            coverage_score=round(coverage, 4),
            omniscience_gap=round(omniscience_gap, 4),
            narrative=narrative,
        )

    def _find_unmodeled_domains(self, predictions, domain_states) -> List[BlindSpot]:
        spots = []
        known_silos = {
            "supply_chain": "Global supply chain disruptions (semiconductor, shipping, rare earths) — no dedicated model",
            "regulatory": "Regulatory regime changes (antitrust, privacy, emissions) — no dedicated model",
            "demographic": "Demographic shifts (aging, migration, birth rate collapse) — partially modeled",
            "cyber": "Cyber risk (infrastructure attacks, data breaches) — no dedicated model",
            "social_cohesion": "Social cohesion and polarization trends — no dedicated model",
            "pandemic": "Pandemic recurrence and pathogen evolution — no dedicated model",
            "space": "Space weather, satellite disruption, Kessler syndrome — no model",
            "institutional": "Institutional trust erosion, democratic backsliding — partially modeled",
            "psychological": "Collective psychological states (fear, euphoria, fatigue) — partially modeled",
        }
        covered = set()
        if domain_states:
            covered = set(domain_states.keys())
        for domain, desc in known_silos.items():
            if domain not in covered:
                severity = BlindSpotSeverity.HIGH if domain in ("cyber", "pandemic", "supply_chain") else BlindSpotSeverity.MODERATE
                spots.append(BlindSpot(
                    category=BlindSpotCategory.UNMODELED_DOMAIN,
                    severity=severity,
                    description=desc,
                    domains=[domain],
                    affected_models=list(ORACLE_MODELS)[:3],
                    probability_impact=0.15 if severity == BlindSpotSeverity.HIGH else 0.08,
                    recommendation=f"Develop {domain} prediction model or add {domain} domain states",
                    evidence="Systematic absence in prediction coverage",
                ))
        return spots

    def _find_shared_assumptions(self, active_models: Set[str]) -> List[BlindSpot]:
        spots = []
        for group_name, models in SHARED_ASSUMPTION_GROUPS.items():
            overlap = len(active_models & models)
            if overlap >= 3:
                spots.append(BlindSpot(
                    category=BlindSpotCategory.SHARED_ASSUMPTION,
                    severity=BlindSpotSeverity.HIGH if overlap >= 4 else BlindSpotSeverity.MODERATE,
                    description=f"{overlap} models share '{group_name}' assumptions (efficiency, rationality, data source) — false convergence risk",
                    domains=[group_name],
                    affected_models=list(active_models & models),
                    probability_impact=0.12,
                    recommendation=f"Add models with independent assumptions for {group_name} to diversify",
                    evidence=f"Models: {', '.join(active_models & models)} share common root assumptions",
                ))
        return spots

    def _find_edge_cases(self, domain_states) -> List[BlindSpot]:
        if not domain_states:
            return []
        spots = []
        for domain, value in domain_states.items():
            if value > 0.85 or value < 0.15:
                severity = BlindSpotSeverity.CRITICAL if value > 0.95 or value < 0.05 else BlindSpotSeverity.HIGH
                direction = "extreme high" if value > 0.85 else "extreme low"
                spots.append(BlindSpot(
                    category=BlindSpotCategory.EDGE_CASE,
                    severity=severity,
                    description=f"{domain} at {value:.2f} ({direction}) — models may not generalize to this region",
                    domains=[domain],
                    affected_models=list(ORACLE_MODELS)[:5],
                    probability_impact=0.2 if severity == BlindSpotSeverity.CRITICAL else 0.1,
                    recommendation=f"Apply scenario analysis for extreme {domain} — stress test all models",
                    evidence="Most models trained/calibrated on moderate data; extreme tails are poorly estimated",
                ))
        return spots

    def _find_temporal_blind_spots(self) -> List[BlindSpot]:
        spots = []
        horizons = [
            (7, "Weekly horizon", "Most models calibrated monthly+; weekly noise dominates signal", 0.06),
            (1, "Daily horizon", "Near-term dominated by noise; no model captures intraday dynamics", 0.1),
            (365 * 3, "3+ year horizon", "Structural breaks invalidate long-range assumptions", 0.15),
            (365 * 5, "5+ year horizon", "Fundamental uncertainty dominates; regime shifts likely", 0.2),
        ]
        for days, label, desc, impact in horizons:
            spots.append(BlindSpot(
                category=BlindSpotCategory.TEMPORAL_HORIZON,
                severity=BlindSpotSeverity.MODERATE if days > 30 else BlindSpotSeverity.HIGH,
                description=f"{label}: {desc}",
                domains=["temporal"],
                affected_models=list(ORACLE_MODELS)[:3],
                probability_impact=impact,
                recommendation=f"Increase model diversity at {label.lower()} or use broader confidence intervals",
                evidence="Forecast accuracy degrades ~sqrt(horizon); structural breaks compound uncertainty",
            ))
        return spots

    def _find_correlation_gaps(self, predictions, domain_states) -> List[BlindSpot]:
        spots = []
        domain_set = set(domain_states.keys()) if domain_states else self.covered_domains
        covered_pairs = set()
        for (a, b), exists in KNOWN_INTERACTIONS.items():
            if exists:
                covered_pairs.add((a, b))
                covered_pairs.add((b, a))
        all_pairs = set()
        domains = list(domain_set)
        for i, d1 in enumerate(domains):
            for d2 in domains[i + 1:]:
                all_pairs.add((d1, d2))
        uncovered = all_pairs - covered_pairs
        if len(uncovered) > 0:
            sample = list(uncovered)[:5]
            pairs_str = ", ".join(f"{a}-{b}" for a, b in sample)
            coverage_ratio = len(covered_pairs) / max(len(all_pairs) * 2, 1)
            spots.append(BlindSpot(
                category=BlindSpotCategory.CORRELATION_GAP,
                severity=BlindSpotSeverity.MODERATE if coverage_ratio > 0.2 else BlindSpotSeverity.HIGH,
                description=f"{len(uncovered)} domain pairs lack modeled interactions (e.g., {pairs_str})",
                domains=[d for pair in sample for d in pair],
                affected_models=["cross_impact", "signal_fusion", "cascade_forecast"],
                probability_impact=0.08 * len(uncovered) / max(len(all_pairs), 1),
                recommendation=f"Model interactions for top-risk pairs; currently only {coverage_ratio:.0%} coverage",
                evidence=f"Only {len(covered_pairs)} of {len(all_pairs)*2} possible domain interactions are modeled",
            ))
        return spots

    def _find_narrative_traps(self, predictions) -> List[BlindSpot]:
        spots = []
        for trap_name, desc in NARRATIVE_TRAPS.items():
            spots.append(BlindSpot(
                category=BlindSpotCategory.NARRATIVE_TRAP,
                severity=BlindSpotSeverity.MODERATE,
                description=f"{trap_name}: {desc}",
                domains=["cognitive"],
                affected_models=["prophecy", "daily_oracle", "superforecast"],
                probability_impact=0.09,
                recommendation=f"Actively seek disconfirming evidence for {trap_name}; use RedTeam module",
                evidence="Behavioral economics: cognitive biases systematically distort probability judgments",
            ))
        return spots[:5]

    def _find_data_deserts(self, predictions) -> List[BlindSpot]:
        spots = []
        deserts = [
            ("real_time_geopolitical", "No real-time geopolitical event feed — relying on periodic updates"),
            ("corporate_earnings", "No earnings data pipeline — equity models use proxy data"),
            ("social_media_sentiment", "No social media sentiment API — sentiment models use market proxies"),
            ("satellite_imagery", "No satellite/sensor data — economic activity estimates are lagging"),
            ("pandemic_surveillance", "No wastewater/hospital admission data — pandemic models use lagging indicators"),
        ]
        for name, desc in deserts:
            spots.append(BlindSpot(
                category=BlindSpotCategory.DATA_DESERT,
                severity=BlindSpotSeverity.MODERATE,
                description=desc,
                domains=[name.split("_")[0]],
                affected_models=["daily_oracle", "signal_fusion"],
                probability_impact=0.06,
                recommendation=f"Integrate {name} data source for improved signal quality",
                evidence="Data deserts create systematic underestimation of relevant domain changes",
            ))
        return spots[:4]

    def _find_interaction_blindness(self, domain_states) -> List[BlindSpot]:
        if not domain_states:
            return []
        spots = []
        extreme_domains = [d for d, v in domain_states.items() if v > 0.7 or v < 0.3]
        for d in extreme_domains:
            spots.append(BlindSpot(
                category=BlindSpotCategory.INTERACTION_BLINDNESS,
                severity=BlindSpotSeverity.HIGH,
                description=f"Extreme {d} state may trigger nonlinear interactions with other domains that are not modeled",
                domains=[d],
                affected_models=["cross_impact", "cascade_forecast"],
                probability_impact=0.1,
                recommendation=f"Run stress scenarios for {d}-originated cascades across all domains",
                evidence="Nonlinear regime transitions are most likely when one domain is at extreme values",
            ))
        return spots[:3]

    def _compute_coverage(self, active_models: Set[str], covered_domains: Set[str]) -> float:
        domain_model_count = {}
        for model in active_models:
            for domain in covered_domains:
                domain_model_count[domain] = domain_model_count.get(domain, 0) + 1
        total_possible = len(covered_domains) * len(ORACLE_MODELS)
        total_actual = sum(domain_model_count.values())
        coverage = total_actual / max(total_possible, 1)
        min_per_domain = min(domain_model_count.values()) if domain_model_count else 0
        coverage *= 0.7 + 0.3 * min(max(min_per_domain / 3, 0), 1)
        return min(1.0, coverage)

    def _generate_narrative(self, blind_spots, unmodeled, coverage) -> str:
        crit = [b for b in blind_spots if b.severity == BlindSpotSeverity.CRITICAL]
        high = [b for b in blind_spots if b.severity == BlindSpotSeverity.HIGH]
        parts = [f"Oracle omniscience gap: {1 - coverage:.0%} of prediction space unmodeled."]
        if crit:
            parts.append(f"CRITICAL: {len(crit)} blind spots — {', '.join(b.domains[0] for b in crit[:3])}.")
        if high:
            parts.append(f"HIGH: {len(high)} significant gaps detected.")
        if unmodeled:
            parts.append(f"Unmodeled domains ({len(unmodeled)}): no prediction models cover these areas.")
        parts.append(f"Total blind spots: {len(blind_spots)} — the Oracle's sight has limits.")
        return " ".join(parts)


SHARED_ASSUMPTION_GROUPS = {
    "efficient_market": {"signal_fusion", "superforecast", "ensemble_calibrator", "ts_forecast"},
    "behavioral": {"analogical_reasoning", "morphological", "delphi"},
    "structural": {"cross_impact", "causal_event", "cascade_forecast", "butterfly_effect"},
    "extreme_tail": {"black_swan", "existential_risk", "stress_testing"},
    "regime_based": {"regime_shift", "regime_assess", "butterfly_effect"},
    "data_driven": {"daily_oracle", "ts_forecast", "prediction_tracker"},
}