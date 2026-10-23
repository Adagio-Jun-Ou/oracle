"""Black Swan Detector — Extreme tail risk + outlier detection beyond EVT.

Identifies potential Black Swan events using multiple complementary approaches:
1. Statistical extreme value theory (GPD tail fitting, POT method)
2. Structural vulnerability analysis (interconnected fragility)
3. Narrative surprise detection (events outside the model space)
4. Fat-tail diagnostics (kurtosis, power law exponent, tail index)
5. Dragon-King detection (outliers even beyond power law tails)

Key insight: Black Swans aren't just rare events — they're events outside the
imagination of the model space. This detector looks for structural conditions
that make system-surprising events more likely, not just statistical outliers.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class SwanCategory(Enum):
    GREY_SWAN = "grey_swan"
    KNOWN_UNKNOWN = "known_unknown"
    BLACK_SWAN = "black_swan"
    DRAGON_KING = "dragon_king"
    WILD_CARD = "wild_card"


class VulnerabilityAxis(Enum):
    LEVERAGE = "leverage"
    INTERCONNECTEDNESS = "interconnectedness"
    COMPLEXITY = "complexity"
    OPACITY = "opacity"
    FEEDBACK_STRENGTH = "feedback_strength"
    FRAGILITY = "fragility"


@dataclass
class TailDiagnostic:
    distribution_type: str
    tail_index: float
    kurtosis: float
    power_law_exponent: Optional[float]
    max_to_mean_ratio: float
    gpd_shape: Optional[float]
    is_fat_tailed: bool
    is_power_law: bool
    dragon_king_score: float


@dataclass
class StructuralVulnerability:
    axis: VulnerabilityAxis
    score: float
    description: str
    contributing_factors: List[str]


@dataclass
class BlackSwanReport:
    overall_probability: float
    category: SwanCategory
    tail_diagnostics: Dict[str, TailDiagnostic]
    structural_vulnerabilities: List[StructuralVulnerability]
    surprise_potential: float
    dragon_king_candidates: List[str]
    wild_card_domains: List[str]
    recommendations: List[str]
    confidence: float


class BlackSwanDetector:
    def __init__(self, threshold_percentile: float = 95.0, min_observations: int = 30):
        self.threshold_percentile = threshold_percentile
        self.min_observations = min_observations
        self._historical_surprises: List[Dict] = []

    def diagnose_tails(self, data: Dict[str, np.ndarray]) -> Dict[str, TailDiagnostic]:
        diagnostics = {}
        for name, series in data.items():
            if len(series) < self.min_observations:
                continue
            arr = np.asarray(series, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) < 5:
                continue
            kurt = self._kurtosis(arr)
            tail_idx = self._tail_index(arr)
            max_ratio = np.max(np.abs(arr)) / max(np.mean(np.abs(arr)), 1e-10)
            is_fat = kurt > 3.0
            power_exp = self._power_law_exponent(arr) if len(arr) > 20 else None
            is_pl = power_exp is not None and power_exp < 5.0
            gpd_shape = self._gpd_fit(arr)
            dk_score = self._dragon_king_score(arr)
            dist_type = "normal"
            if is_pl:
                dist_type = "power_law"
            elif is_fat:
                dist_type = "fat_tailed"
            elif abs(kurt - 3.0) < 0.5:
                dist_type = "normal"
            else:
                dist_type = "thin_tailed"
            diagnostics[name] = TailDiagnostic(
                distribution_type=dist_type,
                tail_index=round(tail_idx, 4),
                kurtosis=round(kurt, 4),
                power_law_exponent=round(power_exp, 4) if power_exp else None,
                max_to_mean_ratio=round(max_ratio, 4),
                gpd_shape=round(gpd_shape, 4) if gpd_shape else None,
                is_fat_tailed=is_fat,
                is_power_law=is_pl,
                dragon_king_score=round(dk_score, 4),
            )
        return diagnostics

    def assess_structural_vulnerability(self, domain_states: Dict[str, Dict]) -> List[StructuralVulnerability]:
        vulnerabilities = []
        leverage_score = self._compute_leverage(domain_states)
        vulnerabilities.append(StructuralVulnerability(
            axis=VulnerabilityAxis.LEVERAGE, score=round(leverage_score, 4),
            description="System leverage and debt amplification potential",
            contributing_factors=self._leverage_factors(domain_states),
        ))
        interconnect_score = self._compute_interconnectedness(domain_states)
        vulnerabilities.append(StructuralVulnerability(
            axis=VulnerabilityAxis.INTERCONNECTEDNESS, score=round(interconnect_score, 4),
            description="Cross-system contagion and cascade potential",
            contributing_factors=self._interconnect_factors(domain_states),
        ))
        complexity_score = self._compute_complexity(domain_states)
        vulnerabilities.append(StructuralVulnerability(
            axis=VulnerabilityAxis.COMPLEXITY, score=round(complexity_score, 4),
            description="System complexity and emergent behavior potential",
            contributing_factors=self._complexity_factors(domain_states),
        ))
        opacity_score = self._compute_opacity(domain_states)
        vulnerabilities.append(StructuralVulnerability(
            axis=VulnerabilityAxis.OPACITY, score=round(opacity_score, 4),
            description="Hidden risks and information asymmetry",
            contributing_factors=self._opacity_factors(domain_states),
        ))
        feedback_score = self._compute_feedback(domain_states)
        vulnerabilities.append(StructuralVulnerability(
            axis=VulnerabilityAxis.FEEDBACK_STRENGTH, score=round(feedback_score, 4),
            description="Positive feedback loops and self-reinforcing dynamics",
            contributing_factors=self._feedback_factors(domain_states),
        ))
        fragility_score = self._compute_fragility(domain_states)
        vulnerabilities.append(StructuralVulnerability(
            axis=VulnerabilityAxis.FRAGILITY, score=round(fragility_score, 4),
            description="Sensitivity to stress and convexity of outcomes",
            contributing_factors=self._fragility_factors(domain_states),
        ))
        return vulnerabilities

    def detect(
        self,
        time_series_data: Dict[str, np.ndarray],
        domain_states: Dict[str, Dict],
        known_risks: Optional[List[str]] = None,
    ) -> BlackSwanReport:
        tail_diags = self.diagnose_tails(time_series_data)
        struct_vulns = self.assess_structural_vulnerability(domain_states)
        avg_vuln = np.mean([v.score for v in struct_vulns]) if struct_vulns else 0.5
        fat_tail_count = sum(1 for d in tail_diags.values() if d.is_fat_tailed)
        total_series = max(len(tail_diags), 1)
        fat_tail_ratio = fat_tail_count / total_series
        dragon_candidates = [name for name, d in tail_diags.items() if d.dragon_king_score > 0.7]
        surprise = self._compute_surprise_potential(tail_diags, struct_vulns)
        overall_prob = self._compute_overall_probability(avg_vuln, fat_tail_ratio, surprise)
        category = self._classify_swan(overall_prob, avg_vuln, surprise, dragon_candidates)
        wild_cards = self._identify_wild_cards(domain_states, known_risks or [])
        recs = self._generate_recommendations(category, struct_vulns, dragon_candidates)
        return BlackSwanReport(
            overall_probability=round(overall_prob, 4),
            category=category,
            tail_diagnostics=tail_diags,
            structural_vulnerabilities=struct_vulns,
            surprise_potential=round(surprise, 4),
            dragon_king_candidates=dragon_candidates,
            wild_card_domains=wild_cards,
            recommendations=recs,
            confidence=round(self._compute_confidence(tail_diags, struct_vulns), 4),
        )

    def _kurtosis(self, arr: np.ndarray) -> float:
        n = len(arr)
        if n < 4:
            return 3.0
        mean = np.mean(arr)
        var = np.var(arr, ddof=1)
        if var < 1e-15:
            return 3.0
        m4 = np.mean((arr - mean) ** 4)
        kurt = (m4 / (var ** 2)) - 3.0
        return float(kurt)

    def _tail_index(self, arr: np.ndarray) -> float:
        abs_arr = np.abs(arr)
        sorted_arr = np.sort(abs_arr)[::-1]
        n = len(sorted_arr)
        if n < 10:
            return 1.0
        threshold = np.percentile(abs_arr, self.threshold_percentile)
        exceedances = sorted_arr[sorted_arr > threshold] - threshold
        if len(exceedances) < 2:
            return 1.0
        hill_estimator = np.mean(np.log(exceedances / threshold))
        tail_idx = 1.0 / max(hill_estimator, 0.01)
        return min(float(tail_idx), 10.0)

    def _power_law_exponent(self, arr: np.ndarray) -> Optional[float]:
        abs_arr = np.abs(arr)
        abs_arr = abs_arr[abs_arr > 0]
        if len(abs_arr) < 20:
            return None
        x_min = np.percentile(abs_arr, 90)
        tail = abs_arr[abs_arr >= x_min]
        if len(tail) < 5:
            return None
        n = len(tail)
        alpha = 1.0 + n / (np.sum(np.log(tail / x_min)))
        return float(alpha)

    def _gpd_fit(self, arr: np.ndarray) -> Optional[float]:
        abs_arr = np.sort(np.abs(arr))[::-1]
        threshold_idx = max(1, int(len(abs_arr) * (1 - self.threshold_percentile / 100)))
        exceedances = abs_arr[:threshold_idx] - abs_arr[threshold_idx] if threshold_idx < len(abs_arr) else abs_arr - np.min(abs_arr)
        if len(exceedances) < 3:
            return None
        mean_exc = np.mean(exceedances)
        var_exc = np.var(exceedances, ddof=1)
        if var_exc < 1e-15:
            return 0.0
        xi = 0.5 * (1 - (mean_exc ** 2) / var_exc)
        return float(xi)

    def _dragon_king_score(self, arr: np.ndarray) -> float:
        sorted_arr = np.sort(np.abs(arr))[::-1]
        n = len(sorted_arr)
        if n < 10:
            return 0.0
        top3 = np.sum(sorted_arr[:3])
        total = np.sum(sorted_arr)
        concentration = top3 / total if total > 0 else 0
        expected_concentration = 3.0 / n
        ratio = concentration / max(expected_concentration, 0.01)
        if ratio > 10:
            return min(1.0, ratio / 20.0)
        elif ratio > 5:
            return min(1.0, 0.5 + ratio / 30.0)
        elif ratio > 2:
            return min(1.0, ratio / 8.0)
        return min(1.0, ratio / 10.0)

    def _compute_leverage(self, states: Dict) -> float:
        score = 0.4
        econ = states.get("economic", states.get("economics", {}))
        if isinstance(econ, dict):
            debt_gdp = float(econ.get("debt_to_gdp", econ.get("debt_ratio", 0.5)))
            score += min(0.3, debt_gdp * 0.2)
            credit = float(econ.get("credit_growth", econ.get("credit_expansion", 0.05)))
            score += min(0.15, abs(credit) * 2)
        return min(1.0, score)

    def _compute_interconnectedness(self, states: Dict) -> float:
        n_domains = len(states)
        base = min(1.0, n_domains / 20.0)
        trade = states.get("trade", states.get("supply_chain", {}))
        if isinstance(trade, dict):
            trade_vol = float(trade.get("global_trade_volume", trade.get("trade_dependency", 0.5)))
            base = max(base, min(1.0, base + trade_vol * 0.3))
        return min(1.0, base)

    def _compute_complexity(self, states: Dict) -> float:
        score = 0.3
        tech = states.get("technology", states.get("tech", {}))
        if isinstance(tech, dict):
            disruption = float(tech.get("disruption_rate", tech.get("innovation_velocity", 0.3)))
            score += min(0.3, disruption * 0.5)
        return min(1.0, score)

    def _compute_opacity(self, states: Dict) -> float:
        score = 0.3
        financial = states.get("financial", states.get("economic", {}))
        if isinstance(financial, dict):
            derivative_ratio = float(financial.get("derivative_ratio", financial.get("shadow_banking", 0.4)))
            score += min(0.3, derivative_ratio * 0.5)
        return min(1.0, score)

    def _compute_feedback(self, states: Dict) -> float:
        score = 0.3
        sentiment = states.get("sentiment", {})
        if isinstance(sentiment, dict):
            herding = float(sentiment.get("herding_index", sentiment.get("momentum_score", 0.4)))
            score += min(0.3, herding * 0.5)
        return min(1.0, score)

    def _compute_fragility(self, states: Dict) -> float:
        base = 0.35
        risks = [float(v.get("risk_score", v.get("severity", 0.3))) for v in states.values() if isinstance(v, dict) and any(k in v for k in ["risk_score", "severity"])]
        if risks:
            max_risk = max(risks)
            base += min(0.3, max_risk * 0.4)
        return min(1.0, base)

    def _leverage_factors(self, states: Dict) -> List[str]:
        factors = ["debt levels", "credit expansion", "margin requirements"]
        econ = states.get("economic", states.get("economics", {}))
        if isinstance(econ, dict):
            if float(econ.get("debt_to_gdp", econ.get("debt_ratio", 0))) > 2.5:
                factors.append("extreme debt-to-GDP ratio")
        return factors

    def _interconnect_factors(self, states: Dict) -> List[str]:
        return ["global supply chains", "financial contagion channels", "trade dependencies", "information networks"]

    def _complexity_factors(self, states: Dict) -> List[str]:
        return ["technology discontinuity", "regulatory overlap", "multi-domain coupling"]

    def _opacity_factors(self, states: Dict) -> List[str]:
        return ["shadow banking", "OTC derivatives", "off-balance-sheet entities", "emergent risks"]

    def _feedback_factors(self, states: Dict) -> List[str]:
        return ["margin calls", "bank runs", "liquidity spirals", "herding behavior"]

    def _fragility_factors(self, states: Dict) -> List[str]:
        return ["concentrated positions", "convex loss profiles", "low reserves", "linear risk models"]

    def _compute_surprise_potential(self, tail_diags: Dict[str, TailDiagnostic], vulns: List[StructuralVulnerability]) -> float:
        fat_tail_ratio = sum(1 for d in tail_diags.values() if d.is_fat_tailed) / max(len(tail_diags), 1)
        avg_kurt = np.mean([d.kurtosis for d in tail_diags.values()]) if tail_diags else 3.0
        kurt_surprise = min(1.0, max(0, avg_kurt - 3.0) / 10.0)
        avg_vuln = np.mean([v.score for v in vulns]) if vulns else 0.5
        surprise = 0.3 * fat_tail_ratio + 0.3 * kurt_surprise + 0.4 * avg_vuln
        return min(1.0, surprise)

    def _compute_overall_probability(self, avg_vuln: float, fat_tail_ratio: float, surprise: float) -> float:
        prob = 0.25 * avg_vuln + 0.25 * fat_tail_ratio + 0.5 * surprise
        return min(1.0, max(0.0, prob))

    def _classify_swan(self, prob: float, vuln: float, surprise: float, dk_candidates: List[str]) -> SwanCategory:
        if dk_candidates:
            return SwanCategory.DRAGON_KING
        if prob > 0.7 and surprise > 0.7:
            return SwanCategory.BLACK_SWAN
        elif prob > 0.5 and vuln > 0.6:
            return SwanCategory.KNOWN_UNKNOWN
        elif prob > 0.3:
            return SwanCategory.GREY_SWAN
        return SwanCategory.WILD_CARD

    def _identify_wild_cards(self, states: Dict, known_risks: List[str]) -> List[str]:
        wild_cards = []
        risk_domains = {
            "pandemic": "novel pathogen with unusual transmission pattern",
            "cyber": "cascading cyber infrastructure failure",
            "climate": "abrupt climate regime shift",
            "geopolitical": "nuclear escalation or state collapse",
            "financial": "systemic liquidity crisis",
            "technology": "AI capability surprise",
            "space": "major solar storm or asteroid impact",
            "energy": "energy infrastructure cascade failure",
        }
        for domain, desc in risk_domains.items():
            if domain not in known_risks:
                domain_state = states.get(domain, {})
                if isinstance(domain_state, dict):
                    severity = float(domain_state.get("risk_score", domain_state.get("severity", 0.1)))
                    if severity > 0.3:
                        wild_cards.append(f"{domain}: {desc}")
        if not wild_cards:
            wild_cards.append("unmodeled systemic risk in interconnected domains")
        return wild_cards[:5]

    def _generate_recommendations(self, category: SwanCategory, vulns: List[StructuralVulnerability], dk_candidates: List[str]) -> List[str]:
        recs = []
        if dk_candidates:
            recs.append(f"DRAGON KING ALERT: {', '.join(dk_candidates)} shows extreme concentration — prepare for outliers beyond statistical models")
        if category == SwanCategory.BLACK_SWAN:
            recs.append("BLACK SWAN conditions detected — focus on resilience over prediction")
            recs.append("Build optionality: maintain reserves, diversify, avoid concentrated bets")
        elif category == SwanCategory.KNOWN_UNKNOWN:
            recs.append("KNOWN UNKNOWN conditions — prepare for specific tail scenarios")
        elif category == SwanCategory.GREY_SWAN:
            recs.append("GREY SWAN conditions — tail risks elevated but not extreme")
        high_vulns = sorted(vulns, key=lambda v: v.score, reverse=True)[:2]
        for v in high_vulns:
            recs.append(f"Reduce {v.axis.value}: {v.description} (score: {v.score:.2f})")
        recs.append("Stress-test all major assumptions with 10x adverse scenarios")
        recs.append("Monitor early-warning indicators for regime change signals")
        return recs

    def _compute_confidence(self, tail_diags: Dict, vulns: List) -> float:
        n_series = len(tail_diags)
        n_domains = len([v for v in vulns])
        data_confidence = min(1.0, n_series / 10.0) * 0.5
        domain_confidence = min(1.0, n_domains / 6.0) * 0.3
        base_confidence = 0.2
        quality = sum(1 for d in tail_diags.values() if d.is_fat_tailed is not None)
        quality_confidence = min(1.0, quality / max(n_series, 1)) * 0.2
        return min(0.85, data_confidence + domain_confidence + base_confidence + quality_confidence)