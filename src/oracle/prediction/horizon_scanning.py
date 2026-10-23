"""Horizon Scanning — Systematic weak signal and emerging issue detection.

Scans across domains for emerging trends, weak signals, and future issues
that may not yet be recognized but could have significant impact. Implements
a structured horizon scanning methodology used by foresight organizations
like the UK Government Office for Science and Singapore's Centre for Strategic
Futures.

Key innovations:
1. Multi-domain signal collection with taxonomy
2. Signal velocity tracking (acceleration detection)
3. Cross-domain signal correlation
4. Signal clustering into coherent themes
5. Horizon classification (near/far/macro)
6. Maturity assessment (nascent → emerging → established)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


class SignalMaturity(Enum):
    NASCENT = "nascent"
    EMERGING = "emerging"
    DEVELOPING = "developing"
    ESTABLISHED = "established"
    MAINSTREAM = "mainstream"


class HorizonLevel(Enum):
    HORIZON_1 = "horizon_1"
    HORIZON_2 = "horizon_2"
    HORIZON_3 = "horizon_3"


class SignalCategory(Enum):
    TECHNOLOGY = "technology"
    SOCIAL = "social"
    ECONOMIC = "economic"
    POLITICAL = "political"
    ENVIRONMENTAL = "environmental"
    SCIENTIFIC = "scientific"
    DEMOGRAPHIC = "demographic"
    CULTURAL = "cultural"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class WeakSignal:
    name: str
    category: SignalCategory
    domain: str
    maturity: SignalMaturity
    horizon: HorizonLevel
    impact_potential: float
    probability: float
    velocity: float
    description: str
    indicators: List[str]
    cross_references: List[str]


@dataclass
class SignalCluster:
    theme: str
    signals: List[str]
    coherence: float
    combined_impact: float
    combined_probability: float
    description: str


@dataclass
class HorizonScanResult:
    signals: List[WeakSignal]
    clusters: List[SignalCluster]
    category_distribution: Dict[str, int]
    maturity_distribution: Dict[str, int]
    horizon_distribution: Dict[str, int]
    high_velocity_signals: List[WeakSignal]
    cross_domain_connections: List[Tuple[str, str, float]]
    emerging_themes: List[str]
    recommendations: List[str]


PRE_POPULATED_SIGNALS: List[WeakSignal] = [
    WeakSignal("Quantum computing advantage", SignalCategory.TECHNOLOGY, "computing",
               SignalMaturity.NASCENT, HorizonLevel.HORIZON_3, 0.8, 0.15, 0.3,
               "Quantum computers approaching practical advantage for specific problems",
               ["qubit count growth", "error correction progress", "algorithm development"],
               ["cryptography", "drug_discovery"]),
    WeakSignal("Synthetic biology commoditization", SignalCategory.SCIENTIFIC, "biotech",
               SignalMaturity.EMERGING, HorizonLevel.HORIZON_2, 0.7, 0.25, 0.5,
               "Gene synthesis and organism design becoming accessible and cheap",
               ["gene synthesis cost decline", "DIY bio community growth", "regulatory gaps"],
               ["pandemic_risk", "agriculture"]),
    WeakSignal("AI agent autonomy", SignalCategory.TECHNOLOGY, "ai",
               SignalMaturity.EMERGING, HorizonLevel.HORIZON_2, 0.85, 0.35, 0.7,
               "Autonomous AI agents capable of independent goal pursuit",
               ["agent framework releases", "autonomous tool usage", "multi-agent coordination"],
               ["cybersecurity", "employment"]),
    WeakSignal("Climate tipping point cascade", SignalCategory.ENVIRONMENTAL, "climate",
               SignalMaturity.DEVELOPING, HorizonLevel.HORIZON_1, 0.9, 0.2, 0.4,
               "Multiple climate tipping points approaching simultaneously",
               ["ice sheet acceleration", "permafrost emissions", "ocean circulation slowdown"],
               ["agriculture", "migration"]),
    WeakSignal("Digital currency adoption", SignalCategory.ECONOMIC, "finance",
               SignalMaturity.DEVELOPING, HorizonLevel.HORIZON_1, 0.6, 0.4, 0.6,
               "CBDC and digital currencies reaching critical mass",
               ["CBDC pilot programs", "stablecoin growth", "regulatory frameworks"],
               ["monetary_policy", "banking"]),
    WeakSignal("Neurotechnology consumer adoption", SignalCategory.TECHNOLOGY, "neurotech",
               SignalMaturity.NASCENT, HorizonLevel.HORIZON_3, 0.7, 0.1, 0.2,
               "Brain-computer interfaces becoming consumer products",
               ["Neuralink progress", "EEG consumer devices", "cognitive enhancement demand"],
               ["privacy", "healthcare"]),
    WeakSignal("Space economy emergence", SignalCategory.ECONOMIC, "space",
               SignalMaturity.EMERGING, HorizonLevel.HORIZON_2, 0.65, 0.3, 0.45,
               "Commercial space activities creating new economic sector",
               ["launch cost decline", "satellite mega-constellations", "space manufacturing R&D"],
               ["telecommunications", "earth_observation"]),
    WeakSignal("Demographic inversion", SignalCategory.DEMOGRAPHIC, "population",
               SignalMaturity.DEVELOPING, HorizonLevel.HORIZON_1, 0.7, 0.5, 0.3,
               "Population decline in developed nations creating structural shifts",
               ["fertility rate decline", "aging population data", "immigration pressure"],
               ["healthcare", "pensions"]),
    WeakSignal("Autonomous weapons proliferation", SignalCategory.SECURITY, "defense",
               SignalMaturity.EMERGING, HorizonLevel.HORIZON_2, 0.8, 0.35, 0.55,
               "Autonomous weapons systems spreading globally",
               ["drone swarm tests", "AI targeting systems", "arms race dynamics"],
               ["geopolitical", "ethics"]),
    WeakSignal("Post-antibiotic era signals", SignalCategory.SCIENTIFIC, "health",
               SignalMaturity.DEVELOPING, HorizonLevel.HORIZON_1, 0.85, 0.3, 0.35,
               "Antibiotic resistance reaching critical thresholds",
               ["AMR mortality increases", "phage therapy research", "last-resort antibiotic failures"],
               ["pandemic", "agriculture"]),
    WeakSignal("Mental health crisis acceleration", SignalCategory.SOCIAL, "health",
               SignalMaturity.ESTABLISHED, HorizonLevel.HORIZON_1, 0.75, 0.6, 0.7,
               "Global mental health crisis accelerating across demographics",
               ["youth depression rates", "workplace burnout metrics", "suicide data"],
               ["economy", "social_cohesion"]),
    WeakSignal("Energy storage breakthrough", SignalCategory.TECHNOLOGY, "energy",
               SignalMaturity.EMERGING, HorizonLevel.HORIZON_2, 0.7, 0.25, 0.5,
               "Battery and storage technology approaching transformation threshold",
               ["solid-state battery progress", "grid storage deployment", "hydrogen cost curves"],
               ["renewable_energy", "transportation"]),
    WeakSignal("Geopolitical multipolarity", SignalCategory.POLITICAL, "geopolitics",
               SignalMaturity.DEVELOPING, HorizonLevel.HORIZON_1, 0.75, 0.55, 0.4,
               "Shift from unipolar to multipolar world order accelerating",
               ["alliance shifting", "institutional reform", "currency diversification"],
               ["trade", "security"]),
    WeakSignal("Algorithmic governance expansion", SignalCategory.POLITICAL, "governance",
               SignalMaturity.NASCENT, HorizonLevel.HORIZON_3, 0.65, 0.15, 0.3,
               "Government policy increasingly determined by algorithmic systems",
               ["AI regulatory sandboxes", "automated compliance", "predictive policing"],
               ["democracy", "privacy"]),
    WeakSignal("Biodiversity collapse indicators", SignalCategory.ENVIRONMENTAL, "ecology",
               SignalMaturity.DEVELOPING, HorizonLevel.HORIZON_1, 0.85, 0.45, 0.35,
               "Ecosystem collapse signals accelerating across biomes",
               ["species extinction rates", "ecosystem service decline", "tipping point indicators"],
               ["agriculture", "health"]),
]


class HorizonScanner:
    def __init__(self, signals: Optional[List[WeakSignal]] = None):
        self.signals = signals if signals else [s for s in PRE_POPULATED_SIGNALS
                                                 if isinstance(s, WeakSignal)]

    def add_signal(self, signal: WeakSignal) -> None:
        self.signals.append(signal)

    def scan(self, min_velocity: float = 0.0,
              categories: Optional[List[SignalCategory]] = None,
              horizons: Optional[List[HorizonLevel]] = None) -> HorizonScanResult:
        filtered = self.signals
        if min_velocity > 0:
            filtered = [s for s in filtered if s.velocity >= min_velocity]
        if categories:
            filtered = [s for s in filtered if s.category in categories]
        if horizons:
            filtered = [s for s in filtered if s.horizon in horizons]
        cat_dist = defaultdict(int)
        mat_dist = defaultdict(int)
        hor_dist = defaultdict(int)
        for s in filtered:
            cat_dist[s.category.value] += 1
            mat_dist[s.maturity.value] += 1
            hor_dist[s.horizon.value] += 1
        high_vel = sorted([s for s in filtered if s.velocity > 0.4],
                          key=lambda s: s.velocity, reverse=True)
        clusters = self._cluster_signals(filtered)
        cross_domain = self._find_cross_domain(filtered)
        themes = self._extract_emerging_themes(clusters)
        recs = self._generate_recommendations(filtered, high_vel, clusters)
        return HorizonScanResult(
            signals=filtered,
            clusters=clusters,
            category_distribution=dict(cat_dist),
            maturity_distribution=dict(mat_dist),
            horizon_distribution=dict(hor_dist),
            high_velocity_signals=high_vel,
            cross_domain_connections=cross_domain,
            emerging_themes=themes,
            recommendations=recs,
        )

    def track_signal(self, name: str, velocity_readings: List[float]) -> Dict:
        if len(velocity_readings) < 2:
            return {"name": name, "acceleration": 0.0, "trend": "insufficient_data"}
        velocities = np.array(velocity_readings)
        x = np.arange(len(velocities))
        if len(x) >= 2:
            slope = np.polyfit(x, velocities, 1)[0]
        else:
            slope = 0.0
        acceleration = float(velocities[-1] - velocities[0]) / max(len(velocities) - 1, 1)
        trend = "accelerating" if slope > 0.05 else ("decelerating" if slope < -0.05 else "stable")
        return {
            "name": name,
            "acceleration": round(acceleration, 4),
            "slope": round(float(slope), 4),
            "trend": trend,
            "current_velocity": round(float(velocities[-1]), 4),
        }

    def assess_impact_trajectory(self, signal: WeakSignal,
                                   time_steps: int = 10) -> List[Dict]:
        trajectory = []
        impact = signal.impact_potential
        velocity = signal.velocity
        for t in range(time_steps):
            growth = velocity * (1 + 0.1 * t) * impact
            impact = min(1.0, impact + growth * 0.1)
            probability = min(1.0, signal.probability + velocity * 0.05 * t)
            trajectory.append({
                "time_step": t,
                "projected_impact": round(impact, 4),
                "projected_probability": round(probability, 4),
                "momentum": round(impact * probability, 4),
            })
        return trajectory

    def _cluster_signals(self, signals: List[WeakSignal]) -> List[SignalCluster]:
        cross_ref_map = defaultdict(set)
        for s in signals:
            for ref in s.cross_references:
                cross_ref_map[ref].add(s.name)
        clusters = []
        used = set()
        for ref, members in cross_ref_map.items():
            cluster_signals = [s for s in signals if s.name in members and s.name not in used]
            if len(cluster_signals) >= 2:
                avg_impact = np.mean([s.impact_potential for s in cluster_signals])
                avg_prob = np.mean([s.probability for s in cluster_signals])
                coherence = len(cluster_signals) / max(len(members), 1)
                for s in cluster_signals:
                    used.add(s.name)
                clusters.append(SignalCluster(
                    theme=ref.replace("_", " ").title(),
                    signals=[s.name for s in cluster_signals],
                    coherence=round(coherence, 4),
                    combined_impact=round(float(avg_impact), 4),
                    combined_probability=round(float(avg_prob), 4),
                    description=f"Cluster around {ref}: {', '.join(s.name for s in cluster_signals)}",
                ))
        clusters.sort(key=lambda c: c.combined_impact * c.combined_probability, reverse=True)
        return clusters[:10]

    def _find_cross_domain(self, signals: List[WeakSignal]) -> List[Tuple[str, str, float]]:
        connections = []
        seen = set()
        for s1 in signals:
            for s2 in signals:
                if s1.name == s2.name:
                    continue
                overlap = len(set(s1.cross_references) & set(s2.cross_references))
                if overlap > 0:
                    key = tuple(sorted([s1.name, s2.name]))
                    if key not in seen:
                        strength = overlap / max(len(s1.cross_references), len(s2.cross_references), 1)
                        connections.append((s1.name, s2.name, round(strength, 4)))
                        seen.add(key)
        connections.sort(key=lambda x: x[2], reverse=True)
        return connections[:10]

    def _extract_emerging_themes(self, clusters: List[SignalCluster]) -> List[str]:
        return [c.theme for c in clusters[:5]]

    def _generate_recommendations(self, signals: List[WeakSignal],
                                    high_vel: List[WeakSignal],
                                    clusters: List[SignalCluster]) -> List[str]:
        recs = []
        if high_vel:
            recs.append(f"MONITOR: {len(high_vel)} high-velocity signals require attention — top: {high_vel[0].name}")
        if clusters:
            top_cluster = clusters[0]
            recs.append(f"EMERGING THEME: {top_cluster.theme} — {len(top_cluster.signals)} related signals, combined impact {top_cluster.combined_impact:.2f}")
        h3_signals = [s for s in signals if s.horizon == HorizonLevel.HORIZON_3]
        if h3_signals:
            recs.append(f"FAR HORIZON: {len(h3_signals)} Horizon 3 signals that may reshape landscape long-term")
        established = [s for s in signals if s.maturity in (SignalMaturity.ESTABLISHED, SignalMaturity.MAINSTREAM)]
        if established:
            recs.append(f"ESTABLISHED: {len(established)} signals already in mainstream — track for unexpected acceleration")
        nascent = [s for s in signals if s.maturity == SignalMaturity.NASCENT]
        if nascent:
            recs.append(f"WATCH LIST: {len(nascent)} nascent signals with high uncertainty — early positioning opportunity")
        return recs