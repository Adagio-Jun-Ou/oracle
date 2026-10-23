"""Data Quality Scoring System for Oracle Prediction Engine.

Evaluates domain state completeness, accuracy, and recency to:
- Score data quality per domain on a 0-1 scale
- Adjust prediction confidence intervals based on data quality
- Track data source provenance and staleness
- Provide actionable data quality recommendations

Implements:
- DataQualityScorer: Main scoring engine with completeness, accuracy, recency metrics
- DomainQualityReport: Per-domain quality assessment with recommendations
- QualityAdjustedForecaster: Adjusts CI widths based on data quality scores
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class DomainQualityReport:
    domain: str
    overall_score: float
    completeness_score: float
    accuracy_score: float
    recency_score: float
    missing_fields: List[str]
    stale_fields: List[str]
    suspicious_values: List[str]
    recommendations: List[str]
    confidence_multiplier: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class DataSourceProvenance:
    source: str
    domain: str
    last_updated: datetime
    update_frequency_hours: float
    reliability: float

    @property
    def staleness_hours(self) -> float:
        return (datetime.now() - self.last_updated).total_seconds() / 3600

    @property
    def is_stale(self) -> bool:
        return self.staleness_hours > self.update_frequency_hours * 2


DOMAIN_FIELD_REGISTRY = {
    "geopolitical": {
        "required": ["us_china_tension", "russia_west_tension", "middle_east_instability", "trade_protectionism", "energy_geopolitics", "cyber_warfare_likelihood"],
        "ranges": {"us_china_tension": (0, 1), "russia_west_tension": (0, 1), "middle_east_instability": (0, 1),
                   "trade_protectionism": (0, 1), "energy_geopolitics": (0, 1), "cyber_warfare_likelihood": (0, 1)},
        "expected_update_hours": 24.0,
    },
    "climate": {
        "required": ["global_temp_anomaly_c", "co2_concentration_ppm", "sea_level_rise_mm", "extreme_weather_index", "ice_sheet_loss_gt", "renewable_share"],
        "ranges": {"global_temp_anomaly_c": (-1, 5), "co2_concentration_ppm": (280, 600), "sea_level_rise_mm": (0, 500),
                   "extreme_weather_index": (0, 1), "ice_sheet_loss_gt": (0, 10000), "renewable_share": (0, 100)},
        "expected_update_hours": 168.0,
    },
    "sentiment": {
        "required": ["fear_greed_index", "vix_level", "put_call_ratio", "short_interest", "institutional_flow", "retail_flow", "market_breadth"],
        "ranges": {"fear_greed_index": (0, 100), "vix_level": (5, 80), "put_call_ratio": (0.3, 2.0),
                   "short_interest": (0, 0.5), "institutional_flow": (-1, 1), "retail_flow": (-1, 1), "market_breadth": (-1, 1)},
        "expected_update_hours": 4.0,
    },
    "contagion": {
        "required": ["financial_systemic_risk", "bank_stress_index", "sovereign_risk_index", "liquidity_stress", "correlation_spike"],
        "ranges": {"financial_systemic_risk": (0, 1), "bank_stress_index": (0, 1), "sovereign_risk_index": (0, 1),
                   "liquidity_stress": (0, 1), "correlation_spike": (0, 1)},
        "expected_update_hours": 24.0,
    },
    "demographics": {
        "required": ["total_population_m", "birth_rate", "death_rate", "elderly_pct", "urbanization_rate", "gdp_per_capita"],
        "ranges": {"total_population_m": (1, 10000), "birth_rate": (0.5, 5), "death_rate": (0.3, 3),
                   "elderly_pct": (0.01, 0.5), "urbanization_rate": (0.1, 1.0), "gdp_per_capita": (500, 200000)},
        "expected_update_hours": 720.0,
    },
    "epidemiology": {
        "required": ["beta", "gamma", "mu", "vaccination_rate", "hospital_capacity"],
        "ranges": {"beta": (0.1, 3.0), "gamma": (0.05, 0.5), "mu": (0.001, 0.05),
                   "vaccination_rate": (0, 1), "hospital_capacity": (0, 1)},
        "expected_update_hours": 24.0,
    },
    "technology": {
        "required": ["current_adoption", "ai_capability", "quantum_readiness", "disruption_probability"],
        "ranges": {"current_adoption": (0, 1), "ai_capability": (0, 1), "quantum_readiness": (0, 1), "disruption_probability": (0, 1)},
        "expected_update_hours": 168.0,
    },
    "social": {
        "required": ["unrest_probability", "polarization_index", "gini_coefficient", "institutional_trust"],
        "ranges": {"unrest_probability": (0, 1), "polarization_index": (0, 1), "gini_coefficient": (0.2, 0.7), "institutional_trust": (0, 1)},
        "expected_update_hours": 168.0,
    },
    "energy": {
        "required": ["renewable_share", "oil_price", "demand_growth", "security_index"],
        "ranges": {"renewable_share": (0, 100), "oil_price": (10, 300), "demand_growth": (-0.05, 0.1), "security_index": (0, 1)},
        "expected_update_hours": 24.0,
    },
    "economics": {
        "required": ["gdp_growth", "inflation", "unemployment", "interest_rate", "debt_to_gdp"],
        "ranges": {"gdp_growth": (-0.1, 0.15), "inflation": (-0.02, 0.2), "unemployment": (0, 0.3),
                   "interest_rate": (-0.01, 0.2), "debt_to_gdp": (0.1, 4.0)},
        "expected_update_hours": 24.0,
    },
}


class DataQualityScorer:
    """Scores domain data quality on completeness, accuracy, and recency."""

    def __init__(self):
        self.reports: Dict[str, DomainQualityReport] = {}
        self.provenance: Dict[str, DataSourceProvenance] = {}

    def score_completeness(self, domain: str, state: Dict) -> Tuple[float, List[str]]:
        registry = DOMAIN_FIELD_REGISTRY.get(domain, {})
        if not registry:
            return 1.0, []
        required = registry.get("required", [])
        if not required:
            return 1.0, []
        present = 0
        missing = []
        for field in required:
            val = state.get(field)
            if val is not None and (isinstance(val, (int, float)) and not (np.isnan(val) if isinstance(val, float) else False)):
                present += 1
            else:
                missing.append(field)
        score = present / len(required) if required else 1.0
        return score, missing

    def score_accuracy(self, domain: str, state: Dict) -> Tuple[float, List[str]]:
        registry = DOMAIN_FIELD_REGISTRY.get(domain, {})
        if not registry:
            return 1.0, []
        ranges = registry.get("ranges", {})
        if not ranges:
            return 1.0, []
        suspicious = []
        in_range = 0
        total = 0
        for field, (low, high) in ranges.items():
            val = state.get(field)
            if val is not None and isinstance(val, (int, float)):
                total += 1
                if low <= val <= high:
                    in_range += 1
                else:
                    suspicious.append(f"{field}={val} outside [{low}, {high}]]")
        score = in_range / total if total > 0 else 1.0
        return score, suspicious

    def score_recency(self, domain: str, last_updated: Optional[datetime] = None) -> Tuple[float, List[str]]:
        registry = DOMAIN_FIELD_REGISTRY.get(domain, {})
        if not registry:
            return 1.0, []
        expected_hours = registry.get("expected_update_hours", 168.0)
        stale_fields = []
        if self.provenance.get(domain):
            prov = self.provenance[domain]
            staleness = prov.staleness_hours
            ratio = staleness / expected_hours
            if ratio > 2:
                stale_fields.append(f"{domain} data {staleness:.0f}h old (expected {expected_hours:.0f}h)")
        elif last_updated:
            staleness = (datetime.now() - last_updated).total_seconds() / 3600
            ratio = staleness / expected_hours
            if ratio > 2:
                stale_fields.append(f"{domain} data {staleness:.0f}h old")
        score = max(0.0, 1.0 - 0.5 * max(0, (len(stale_fields)) / max(1, len(stale_fields) + 1)))
        return score, stale_fields

    def score_domain(self, domain: str, state: Dict,
                     last_updated: Optional[datetime] = None) -> DomainQualityReport:
        completeness, missing = self.score_completeness(domain, state)
        accuracy, suspicious = self.score_accuracy(domain, state)
        recency, stale = self.score_recency(domain, last_updated)
        overall = 0.4 * completeness + 0.35 * accuracy + 0.25 * recency
        confidence_multiplier = 0.5 + 0.5 * overall
        recommendations = []
        if missing:
            recommendations.append(f"Populate missing fields: {', '.join(missing[:3])}")
        if suspicious:
            recommendations.append(f"Verify out-of-range values: {', '.join(suspicious[:3])}")
        if stale:
            recommendations.append(f"Update stale data: {', '.join(stale)}")
        if completeness < 0.7:
            recommendations.append(f"Critical: {domain} data completeness at {completeness:.0%} — predictions unreliable")
        if accuracy < 0.5:
            recommendations.append(f"Warning: {domain} data accuracy at {accuracy:.0%} — consider data validation")

        report = DomainQualityReport(
            domain=domain,
            overall_score=round(overall, 4),
            completeness_score=round(completeness, 4),
            accuracy_score=round(accuracy, 4),
            recency_score=round(recency, 4),
            missing_fields=missing,
            stale_fields=stale,
            suspicious_values=suspicious,
            recommendations=recommendations,
            confidence_multiplier=round(confidence_multiplier, 4),
        )
        self.reports[domain] = report
        return report

    def score_all_domains(self, domain_states: Dict[str, Dict],
                          last_updates: Optional[Dict[str, datetime]] = None) -> Dict[str, DomainQualityReport]:
        results = {}
        for domain, state in domain_states.items():
            lu = last_updates.get(domain) if last_updates else None
            results[domain] = self.score_domain(domain, state, lu)
        return results

    def global_quality_score(self) -> float:
        if not self.reports:
            return 1.0
        scores = [r.overall_score for r in self.reports.values()]
        return float(np.mean(scores)) if scores else 1.0

    def adjust_confidence_interval(self, domain: str, ci_lower: float, ci_upper: float) -> Tuple[float, float]:
        report = self.reports.get(domain)
        if not report:
            return ci_lower, ci_upper
        multiplier = report.confidence_multiplier
        center = (ci_lower + ci_upper) / 2
        half_width = (ci_upper - ci_lower) / 2
        adjusted_half = half_width / multiplier
        return round(center - adjusted_half, 6), round(center + adjusted_half, 6)

    def quality_summary(self) -> Dict:
        if not self.reports:
            return {"global_quality": 1.0, "domains": {}, "recommendations": []}
        domain_scores = {d: {"overall": r.overall_score, "completeness": r.completeness_score,
                             "accuracy": r.accuracy_score, "recency": r.recency_score,
                             "confidence_multiplier": r.confidence_multiplier} for d, r in self.reports.items()}
        all_recos = []
        for r in self.reports.values():
            all_recos.extend(r.recommendations)
        return {
            "global_quality": round(self.global_quality_score(), 4),
            "domains": domain_scores,
            "recommendations": list(set(all_recos)),
            "n_domains": len(self.reports),
            "n_low_quality": sum(1 for r in self.reports.values() if r.overall_score < 0.5),
        }


class QualityAdjustedForecaster:
    """Adjusts forecast confidence intervals based on data quality scores."""

    def __init__(self, scorer: DataQualityScorer):
        self.scorer = scorer

    def adjust_prediction(self, domain: str, value: float,
                          ci_lower: float, ci_upper: float,
                          confidence_level: float = 0.95) -> Dict:
        adj_lower, adj_upper = self.scorer.adjust_confidence_interval(domain, ci_lower, ci_upper)
        report = self.scorer.reports.get(domain)
        quality_score = report.overall_score if report else 1.0
        multiplier = report.confidence_multiplier if report else 1.0
        return {
            "domain": domain,
            "point_estimate": value,
            "original_ci": (ci_lower, ci_upper),
            "adjusted_ci": (adj_lower, adj_upper),
            "quality_score": quality_score,
            "confidence_multiplier": multiplier,
            "adjustment_factor": round(1 / multiplier, 4) if multiplier > 0 else float("inf"),
            "note": f"CI widened by {round((1/multiplier - 1)*100, 1)}% due to data quality ({quality_score:.0%})" if multiplier < 1 else "No adjustment needed",
        }

    def adjust_forecast_set(self, predictions: List[Dict]) -> List[Dict]:
        adjusted = []
        for pred in predictions:
            domain = pred.get("domain", "unknown")
            value = pred.get("value", pred.get("point_estimate", 0))
            ci_lower = pred.get("confidence_interval", (None, None))[0] or pred.get("ci_lower", value - 0.1)
            ci_upper = pred.get("confidence_interval", (None, None))[1] or pred.get("ci_upper", value + 0.1)
            adj = self.adjust_prediction(domain, value, ci_lower, ci_upper)
            adjusted.append({**pred, "quality_adjusted_ci": adj["adjusted_ci"],
                           "quality_score": adj["quality_score"],
                           "confidence_multiplier": adj["confidence_multiplier"]})
        return adjusted