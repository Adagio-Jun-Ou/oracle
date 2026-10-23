"""Anomaly Correlator — Cross-domain anomaly correlation engine.

Detects when anomalies across different domains are correlated,
identifying compound signals that are stronger than individual anomalies.

Like Mr. House correlating seismic data from multiple sensor arrays,
this engine finds the hidden connections between domain disruptions.

Key features:
  - Cross-domain anomaly correlation (Pearson, Spearman, Kendall)
  - Temporal lag correlation (anomaly in domain A precedes domain B)
  - Cascade risk scoring (multi-domain anomaly chains)
  - Known correlation patterns (yield curve inversion + volatility, etc.)
  - Emerging correlation detection (new patterns not in known set)
  - Anomaly cluster identification (groups of co-occurring anomalies)
  - Predictive amplification (correlated anomalies amplify prediction shifts)
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import date, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class AnomalySeverity(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class CorrelationType(Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    LAGGED = "lagged"


class CascadeRiskLevel(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    domain: str
    metric: str
    value: float
    baseline: float
    z_score: float
    severity: AnomalySeverity
    date: str
    direction: str  # "above" or "below"
    description: str = ""


@dataclass
class DomainCorrelation:
    domain_a: str
    domain_b: str
    correlation: float
    correlation_type: CorrelationType
    lag_days: int
    confidence: float
    historical_frequency: float
    description: str = ""


@dataclass
class AnomalyCluster:
    anomalies: List[AnomalyEvent]
    domains_involved: List[str]
    cluster_score: float
    cascade_risk: CascadeRiskLevel
    narrative: str
    predictive_shift: float


@dataclass
class EmergingCorrelation:
    domain_a: str
    domain_b: str
    recent_correlation: float
    historical_correlation: float
    divergence: float
    is_emerging: bool
    description: str


@dataclass
class CascadePathway:
    steps: List[str]
    cumulative_probability: float
    cumulative_amplification: float
    risk_level: CascadeRiskLevel
    narrative: str


@dataclass
class AnomalyCorrelationResult:
    anomalies: List[AnomalyEvent]
    correlations: List[DomainCorrelation]
    clusters: List[AnomalyCluster]
    emerging: List[EmergingCorrelation]
    cascade_pathways: List[CascadePathway]
    overall_risk_level: CascadeRiskLevel
    overall_risk_score: float
    predictive_shift: float
    narrative: str
    evidence: List[str]


KNOWN_CORRELATION_PATTERNS = [
    {
        "domains": ["equity", "volatility"],
        "correlation": -0.7,
        "lag_days": 0,
        "description": "Equity-volatility inverse correlation (VIX effect). Rising volatility precedes equity declines.",
        "evidence": "Black (1976) leverage effect; French, Schwert & Stambaugh (1987)",
    },
    {
        "domains": ["inflation", "bonds"],
        "correlation": -0.6,
        "lag_days": 5,
        "description": "Rising inflation erodes bond yields with ~5 day lag. TIPS breakeven widens.",
        "evidence": "Fisher (1930) effect; Campbell & Shiller (1991)",
    },
    {
        "domains": ["labor", "equity"],
        "correlation": 0.5,
        "lag_days": 30,
        "description": "Low unemployment → rising wages → margin pressure → equity drag. Phillips curve dynamic.",
        "evidence": "BLS Employment-Cost Index; Fed Beige Book correlations",
    },
    {
        "domains": ["monetary", "equity"],
        "correlation": -0.4,
        "lag_days": 15,
        "description": "Tightening monetary policy (rate hikes) precedes equity corrections by ~2 weeks.",
        "evidence": "Bernanke & Kuttner (2005); Fed SLOOS survey",
    },
    {
        "domains": ["geopolitical", "commodity"],
        "correlation": 0.6,
        "lag_days": 3,
        "description": "Geopolitical risk spikes drive commodity (esp. oil, gold) surges within 3 days.",
        "evidence": "Caldara & Iacoviello (2022) GPR Index; Baffes et al. (2022)",
    },
    {
        "domains": ["volatility", "bonds"],
        "correlation": 0.5,
        "lag_days": 7,
        "description": "Volatility spikes precede flight-to-quality bond rallies by ~1 week.",
        "evidence": "Baele, Bekaert & Inghelbrecht (2010); MOVE Index correlations",
    },
    {
        "domains": ["sentiment", "equity"],
        "correlation": 0.55,
        "lag_days": 2,
        "description": "Sentiment shifts precede equity moves by 1-2 days (behavioral finance effect).",
        "evidence": "Baker & Wurgler (2006); Da, Engelberg & Gao (2015)",
    },
    {
        "domains": ["inflation", "labor"],
        "correlation": 0.6,
        "lag_days": 20,
        "description": "Low unemployment → wage inflation with ~1 month lag (Phillips curve).",
        "evidence": "Phillips (1958); Blanchard, Cerutti & Summers (2015)",
    },
    {
        "domains": ["monetary", "inflation"],
        "correlation": 0.4,
        "lag_days": 180,
        "description": "Monetary expansion precedes inflation by ~6 months (monetary transmission lag).",
        "evidence": "Friedman (1961); Christiano, Eichenbaum & Evans (1999)",
    },
    {
        "domains": ["technology", "sentiment"],
        "correlation": 0.5,
        "lag_days": 5,
        "description": "Technology breakthrough events boost market sentiment within a week.",
        "evidence": "Novy-Marx (2013); Shiller (2000) irrational exuberance",
    },
    {
        "domains": ["climate", "commodity"],
        "correlation": 0.4,
        "lag_days": 14,
        "description": "Climate disruptions (droughts, floods) affect agricultural commodity prices within 2 weeks.",
        "evidence": "Lesk, Rowhani & Ramankutty (2016); NOAA climate-commerce studies",
    },
    {
        "domains": ["geopolitical", "volatility"],
        "correlation": 0.55,
        "lag_days": 1,
        "description": "Geopolitical events produce immediate volatility spikes (within 1 day).",
        "evidence": "Caldara & Iacoviello (2022); VIX-GPR correlation studies",
    },
]

DOMAIN_THRESHOLDS = {
    "equity": {"low": 1.0, "moderate": 1.5, "high": 2.0, "critical": 3.0},
    "volatility": {"low": 1.0, "moderate": 1.5, "high": 2.0, "critical": 2.5},
    "bonds": {"low": 1.0, "moderate": 1.5, "high": 2.0, "critical": 3.0},
    "commodity": {"low": 1.0, "moderate": 2.0, "high": 2.5, "critical": 3.5},
    "inflation": {"low": 1.0, "moderate": 1.5, "high": 2.0, "critical": 3.0},
    "labor": {"low": 1.0, "moderate": 1.5, "high": 2.0, "critical": 3.0},
    "monetary": {"low": 1.0, "moderate": 1.5, "high": 2.0, "critical": 2.5},
    "geopolitical": {"low": 1.0, "moderate": 1.5, "high": 2.5, "critical": 3.5},
    "sentiment": {"low": 1.0, "moderate": 1.5, "high": 2.5, "critical": 3.0},
    "technology": {"low": 1.0, "moderate": 1.5, "high": 2.5, "critical": 3.5},
    "climate": {"low": 1.0, "moderate": 2.0, "high": 2.5, "critical": 3.5},
    "health": {"low": 1.0, "moderate": 1.5, "high": 2.0, "critical": 3.0},
    "crypto": {"low": 1.0, "moderate": 1.5, "high": 2.0, "critical": 3.0},
    "currency": {"low": 1.0, "moderate": 1.5, "high": 2.0, "critical": 3.0},
}


class AnomalyCorrelator:
    """Cross-domain anomaly correlation engine."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._anomaly_history: Dict[str, List[AnomalyEvent]] = {}

    def _severity(self, domain: str, z_score: float) -> AnomalySeverity:
        thresholds = DOMAIN_THRESHOLDS.get(domain, DOMAIN_THRESHOLDS["equity"])
        z_abs = abs(z_score)
        if z_abs >= thresholds["critical"]:
            return AnomalySeverity.CRITICAL
        elif z_abs >= thresholds["high"]:
            return AnomalySeverity.HIGH
        elif z_abs >= thresholds["moderate"]:
            return AnomalySeverity.MODERATE
        elif z_abs >= thresholds["low"]:
            return AnomalySeverity.LOW
        return AnomalySeverity.LOW

    def detect_anomalies(self, domain_states: Dict[str, float],
                         baselines: Optional[Dict[str, float]] = None,
                         std_devs: Optional[Dict[str, float]] = None) -> List[AnomalyEvent]:
        baselines = baselines or {k: 0.5 for k in domain_states}
        std_devs = std_devs or {k: 0.15 for k in domain_states}
        anomalies = []
        for domain, value in domain_states.items():
            baseline = baselines.get(domain, 0.5)
            sd = std_devs.get(domain, 0.15)
            if sd < 1e-10:
                continue
            z_score = (value - baseline) / sd
            direction = "above" if value > baseline else "below"
            severity = self._severity(domain, z_score)
            if abs(z_score) >= 1.0:
                event = AnomalyEvent(
                    domain=domain, metric="state", value=round(value, 4),
                    baseline=round(baseline, 4), z_score=round(z_score, 4),
                    severity=severity, date=date.today().isoformat(),
                    direction=direction,
                    description=f"{domain} {direction} baseline by {abs(z_score):.1f}σ",
                )
                anomalies.append(event)
                self._anomaly_history.setdefault(domain, []).append(event)
        return sorted(anomalies, key=lambda a: abs(a.z_score), reverse=True)

    def _pearson(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 3:
            return 0.0
        std_x, std_y = np.std(x), np.std(y)
        if std_x < 1e-10 or std_y < 1e-10:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def _spearman(self, x: np.ndarray, y: np.ndarray) -> float:
        n = len(x)
        if n < 3:
            return 0.0
        rank_x = np.argsort(np.argsort(x)) + 1
        rank_y = np.argsort(np.argsort(y)) + 1
        return self._pearson(rank_x.astype(float), rank_y.astype(float))

    def correlate_domains(self, domain_series: Dict[str, np.ndarray],
                          lag_range: Tuple[int, int] = (0, 10)) -> List[DomainCorrelation]:
        correlations = []
        domains = sorted(domain_series.keys())
        for i, da in enumerate(domains):
            for db in domains[i + 1:]:
                series_a = domain_series[da]
                series_b = domain_series[db]
                min_len = min(len(series_a), len(series_b))
                if min_len < 5:
                    continue
                a = series_a[-min_len:]
                b = series_b[-min_len:]
                pearson = self._pearson(a, b)
                spearman = self._spearman(a, b)
                best_corr = pearson
                best_type = CorrelationType.PEARSON
                if abs(spearman) > abs(pearson):
                    best_corr = spearman
                    best_type = CorrelationType.SPEARMAN
                best_lag = 0
                best_lag_corr = best_corr
                if min_len > lag_range[1] + 3:
                    for lag in range(lag_range[0], min(lag_range[1], min_len // 3)):
                        if lag == 0:
                            continue
                        shifted_a = a[:-lag] if lag > 0 else a[-lag:]
                        shifted_b = b[lag:] if lag > 0 else b[:lag]
                        min_shifted = min(len(shifted_a), len(shifted_b))
                        if min_shifted < 3:
                            continue
                        c = self._pearson(shifted_a[-min_shifted:], shifted_b[-min_shifted:])
                        if abs(c) > abs(best_lag_corr):
                            best_lag_corr = c
                            best_lag = lag
                if abs(best_lag_corr) > abs(best_corr):
                    best_corr = best_lag_corr
                    best_type = CorrelationType.LAGGED
                confidence = min(1.0, abs(best_corr) * min_len / 20)
                known = next((p for p in KNOWN_CORRELATION_PATTERNS
                             if set([da, db]) == set(p["domains"])), None)
                desc = known["description"] if known else f"Empirical {da}-{db} correlation"
                freq = abs(best_corr) if known else abs(best_corr) * 0.5
                correlations.append(DomainCorrelation(
                    domain_a=da, domain_b=db, correlation=round(best_corr, 4),
                    correlation_type=best_type, lag_days=best_lag,
                    confidence=round(confidence, 4), historical_frequency=round(freq, 4),
                    description=desc))
        return sorted(correlations, key=lambda c: abs(c.correlation), reverse=True)

    def find_clusters(self, anomalies: List[AnomalyEvent],
                      correlations: List[DomainCorrelation]) -> List[AnomalyCluster]:
        if not anomalies:
            return []
        domain_anomalies = {}
        for a in anomalies:
            domain_anomalies.setdefault(a.domain, []).append(a)
        domain_graph = {}
        for c in correlations:
            if abs(c.correlation) >= 0.3:
                domain_graph.setdefault(c.domain_a, set()).add(c.domain_b)
                domain_graph.setdefault(c.domain_b, set()).add(c.domain_a)
        visited = set()
        clusters = []
        for domain in domain_anomalies:
            if domain in visited:
                continue
            cluster_domains = set()
            stack = [domain]
            while stack:
                d = stack.pop()
                if d in visited:
                    continue
                visited.add(d)
                cluster_domains.add(d)
                for neighbor in domain_graph.get(d, set()):
                    if neighbor not in visited and neighbor in domain_anomalies:
                        stack.append(neighbor)
            if len(cluster_domains) < 2:
                continue
            cluster_anomalies = []
            for d in cluster_domains:
                cluster_anomalies.extend(domain_anomalies.get(d, []))
            max_z = max((abs(a.z_score) for a in cluster_anomalies), default=0)
            n_domains = len(cluster_domains)
            cluster_score = min(1.0, max_z * n_domains / 10)
            if cluster_score >= 0.7:
                cascade_risk = CascadeRiskLevel.CRITICAL
            elif cluster_score >= 0.5:
                cascade_risk = CascadeRiskLevel.HIGH
            elif cluster_score >= 0.3:
                cascade_risk = CascadeRiskLevel.MODERATE
            elif cluster_score >= 0.15:
                cascade_risk = CascadeRiskLevel.LOW
            else:
                cascade_risk = CascadeRiskLevel.MINIMAL
            domains_str = ", ".join(sorted(cluster_domains))
            severity_counts = {}
            for a in cluster_anomalies:
                severity_counts[a.severity.value] = severity_counts.get(a.severity.value, 0) + 1
            narrative = f"Anomaly cluster across {domains_str}. "
            if severity_counts.get("critical", 0) > 0:
                narrative += f"Contains {severity_counts['critical']} critical anomalies. "
            elif severity_counts.get("high", 0) > 0:
                narrative += f"Contains {severity_counts['high']} high-severity anomalies. "
            predictive_shift = cluster_score * sum(abs(a.z_score) for a in cluster_anomalies) / max(len(cluster_anomalies), 1) * 0.1
            clusters.append(AnomalyCluster(
                anomalies=cluster_anomalies,
                domains_involved=sorted(cluster_domains),
                cluster_score=round(cluster_score, 4),
                cascade_risk=cascade_risk,
                narrative=narrative,
                predictive_shift=round(predictive_shift, 4),
            ))
        return sorted(clusters, key=lambda c: c.cluster_score, reverse=True)

    def detect_emerging(self, domain_series: Dict[str, np.ndarray],
                         recent_window: int = 20) -> List[EmergingCorrelation]:
        emerging = []
        domains = sorted(domain_series.keys())
        for i, da in enumerate(domains):
            for db in domains[i + 1:]:
                full_a = domain_series[da]
                full_b = domain_series[db]
                if len(full_a) < recent_window * 2 or len(full_b) < recent_window * 2:
                    continue
                hist_corr = self._pearson(full_a[:-recent_window], full_b[:-recent_window])
                recent_corr = self._pearson(full_a[-recent_window:], full_b[-recent_window:])
                divergence = recent_corr - hist_corr
                is_emerging = abs(divergence) > 0.2 and abs(recent_corr) > 0.3
                known = next((p for p in KNOWN_CORRELATION_PATTERNS
                             if set([da, db]) == set(p["domains"])), None)
                desc = ""
                if is_emerging:
                    if divergence > 0:
                        desc = f"Emerging positive correlation between {da} and {db} (recent: {recent_corr:.2f}, historical: {hist_corr:.2f}). "
                    else:
                        desc = f"Emerging negative correlation between {da} and {db} (recent: {recent_corr:.2f}, historical: {hist_corr:.2f}). "
                    if known:
                        desc += f"This aligns with known pattern: {known['description'][:80]}"
                    else:
                        desc += "Novel correlation pattern — no known historical precedent."
                elif known and abs(hist_corr) > 0.3:
                    desc = f"Known {da}-{db} pattern holding (correlation: {recent_corr:.2f})."
                emerging.append(EmergingCorrelation(
                    domain_a=da, domain_b=db,
                    recent_correlation=round(recent_corr, 4),
                    historical_correlation=round(hist_corr, 4),
                    divergence=round(divergence, 4),
                    is_emerging=is_emerging,
                    description=desc,
                ))
        return sorted([e for e in emerging if e.is_emerging or e.description],
                       key=lambda e: abs(e.divergence), reverse=True)

    def generate_cascade_pathways(self, anomalies: List[AnomalyEvent],
                                  correlations: List[DomainCorrelation]) -> List[CascadePathway]:
        if not anomalies or not correlations:
            return []
        anomaly_domains = {a.domain for a in anomalies}
        strong_corrs = [c for c in correlations
                        if abs(c.correlation) >= 0.3 and
                        c.domain_a in anomaly_domains and c.domain_b in anomaly_domains]
        if not strong_corrs:
            return []
        visited_starts = set()
        pathways = []
        sorted_anomalies = sorted(anomalies, key=lambda a: abs(a.z_score), reverse=True)
        for seed_anomaly in sorted_anomalies[:5]:
            if seed_anomaly.domain in visited_starts:
                continue
            visited_starts.add(seed_anomaly.domain)
            steps = [seed_anomaly.domain]
            prob = 0.7
            amplification = abs(seed_anomaly.z_score) * 0.1
            current_domain = seed_anomaly.domain
            visited_domains = {current_domain}
            for _ in range(5):
                next_corrs = [c for c in strong_corrs
                             if (c.domain_a == current_domain or c.domain_b == current_domain)
                             and (c.domain_a not in visited_domains or c.domain_b not in visited_domains)]
                if not next_corrs:
                    break
                next_corr = max(next_corrs, key=lambda c: abs(c.correlation))
                next_domain = next_corr.domain_b if next_corr.domain_a == current_domain else next_corr.domain_a
                if next_domain in visited_domains:
                    break
                steps.append(next_domain)
                prob *= abs(next_corr.correlation)
                amplification += abs(next_corr.correlation) * 0.1
                visited_domains.add(next_domain)
                current_domain = next_domain
                if prob < 0.05:
                    break
            if len(steps) < 2:
                continue
            steps_str = " → ".join(steps)
            if prob >= 0.3:
                risk = CascadeRiskLevel.HIGH
            elif prob >= 0.15:
                risk = CascadeRiskLevel.MODERATE
            elif prob >= 0.05:
                risk = CascadeRiskLevel.LOW
            else:
                risk = CascadeRiskLevel.MINIMAL
            severity_desc = "critical" if any(a.severity == AnomalySeverity.CRITICAL for a in anomalies if a.domain in steps) else "elevated"
            narrative = f"Cascade pathway: {steps_str}. Cumulative probability: {prob:.1%}. "
            narrative += f"Severity: {severity_desc}. Amplification factor: {amplification:.2f}x."
            pathways.append(CascadePathway(
                steps=steps,
                cumulative_probability=round(prob, 4),
                cumulative_amplification=round(amplification, 4),
                risk_level=risk,
                narrative=narrative,
            ))
        return sorted(pathways, key=lambda p: p.cumulative_probability, reverse=True)[:5]

    def analyze(self, domain_states: Dict[str, float],
                domain_series: Optional[Dict[str, np.ndarray]] = None,
                baselines: Optional[Dict[str, float]] = None,
                std_devs: Optional[Dict[str, float]] = None) -> AnomalyCorrelationResult:
        anomalies = self.detect_anomalies(domain_states, baselines, std_devs)
        correlations = []
        clusters = []
        emerging = []
        cascades = []
        if domain_series:
            correlations = self.correlate_domains(domain_series)
            clusters = self.find_clusters(anomalies, correlations)
            if len(domain_series.get(next(iter(domain_series)), np.array([]))) >= 40:
                emerging = self.detect_emerging(domain_series)
        if anomalies and correlations:
            cascades = self.generate_cascade_pathways(anomalies, correlations)
        elif anomalies:
            anomaly_domains = {a.domain for a in anomalies}
            for pattern in KNOWN_CORRELATION_PATTERNS:
                if len(anomaly_domains & set(pattern["domains"])) >= 2:
                    correlated_domains = anomaly_domains & set(pattern["domains"])
                    correlation = DomainCorrelation(
                        domain_a=pattern["domains"][0],
                        domain_b=pattern["domains"][1],
                        correlation=pattern["correlation"],
                        correlation_type=CorrelationType.PEARSON,
                        lag_days=pattern["lag_days"],
                        confidence=0.7,
                        historical_frequency=abs(pattern["correlation"]),
                        description=pattern["description"],
                    )
                    correlations.append(correlation)
            if correlations:
                clusters = self.find_clusters(anomalies, correlations)
                cascades = self.generate_cascade_pathways(anomalies, correlations)
        n_critical = sum(1 for a in anomalies if a.severity == AnomalySeverity.CRITICAL)
        n_high = sum(1 for a in anomalies if a.severity == AnomalySeverity.HIGH)
        overall_score = n_critical * 1.0 + n_high * 0.5 + len(clusters) * 0.3
        overall_score = min(1.0, overall_score / 5)
        if overall_score >= 0.7:
            risk_level = CascadeRiskLevel.CRITICAL
        elif overall_score >= 0.5:
            risk_level = CascadeRiskLevel.HIGH
        elif overall_score >= 0.3:
            risk_level = CascadeRiskLevel.MODERATE
        elif overall_score >= 0.15:
            risk_level = CascadeRiskLevel.LOW
        else:
            risk_level = CascadeRiskLevel.MINIMAL
        predictive_shift = sum(c.predictive_shift for c in clusters) if clusters else overall_score * 0.1
        narrative_parts = []
        if n_critical > 0:
            narrative_parts.append(f"{n_critical} critical anomalies detected")
        if n_high > 0:
            narrative_parts.append(f"{n_high} high-severity anomalies")
        if clusters:
            narrative_parts.append(f"{len(clusters)} anomaly cluster(s)")
        if cascades:
            narrative_parts.append(f"{len(cascades)} cascade pathway(s)")
        if emerging:
            n_emerging = sum(1 for e in emerging if e.is_emerging)
            if n_emerging:
                narrative_parts.append(f"{n_emerging} emerging correlation(s)")
        narrative = "; ".join(narrative_parts) if narrative_parts else "No significant cross-domain anomaly correlations detected"
        evidence = [p["evidence"] for p in KNOWN_CORRELATION_PATTERNS
                    if len({a.domain for a in anomalies} & set(p["domains"])) >= 2]
        if not evidence:
            evidence = ["Empirical domain state analysis"]
        return AnomalyCorrelationResult(
            anomalies=anomalies, correlations=correlations, clusters=clusters,
            emerging=emerging, cascade_pathways=cascades,
            overall_risk_level=risk_level, overall_risk_score=round(overall_score, 4),
            predictive_shift=round(predictive_shift, 4), narrative=narrative,
            evidence=evidence[:5],
        )