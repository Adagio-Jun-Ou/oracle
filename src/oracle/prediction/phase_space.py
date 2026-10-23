"""Phase Space Mapper — State space topology for prediction.

Maps the high-dimensional state space of all domain variables onto a lower-
dimensional phase space. Identifies attractor basins, saddle points, and 
bifurcation boundaries where the system can transition between qualitatively
different futures.

Key concepts:
- Attractor basins: regions of phase space the system naturally tends toward
- Saddle points: unstable equilibria where small perturbations change trajectory
- Bifurcation boundaries: surfaces where system behavior qualitatively changes
- Escape routes: paths from one attractor to another (regime change trajectories)

Based on:
- Strogatz (1994) Nonlinear Dynamics and Chaos
- Scheffer (2009) Critical Transitions in Nature and Society
- May (1977) Thresholds and breakpoints in ecosystems
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class AttractorType(Enum):
    STABLE_EQUILIBRIUM = "stable_equilibrium"
    LIMIT_CYCLE = "limit_cycle"
    STRANGE_ATTRACTOR = "strange_attractor"
    SADDLE_POINT = "saddle_point"
    UNSTABLE = "unstable"


class BifurcationType(Enum):
    SADDLE_NODE = "saddle_node"
    TRANSCRITICAL = "transcritical"
    PITCHFORK = "pitchfork"
    HOPF = "hopf"
    SNAPSHOT = "snapshot"


@dataclass
class AttractorBasin:
    name: str
    center: Dict[str, float]
    radius: float
    attractor_type: AttractorType
    stability: float
    domains: List[str]
    probability: float
    narrative: str


@dataclass
class SaddlePoint:
    location: Dict[str, float]
    eigenvalues: List[float]
    instability_index: float
    transition_paths: List[str]
    narrative: str


@dataclass
class BifurcationBoundary:
    boundary_type: BifurcationType
    domains: List[str]
    threshold_values: Dict[str, float]
    distance_from_current: float
    probability_of_crossing: float
    consequence: str
    narrative: str


@dataclass
class EscapeRoute:
    from_attractor: str
    to_attractor: str
    trigger_domains: List[str]
    required_shift: Dict[str, float]
    probability: float
    time_horizon_days: int
    narrative: str


@dataclass
class PhaseSpaceResult:
    current_state_vector: Dict[str, float]
    current_attractor: str
    distance_to_nearest_boundary: float
    attractors: List[AttractorBasin]
    saddle_points: List[SaddlePoint]
    bifurcation_boundaries: List[BifurcationBoundary]
    escape_routes: List[EscapeRoute]
    phase_space_dimension: int
    lyapunov_exponent: float
    resilience: float
    narrative: str


ATTRACTOR_CATALOG = {
    "goldilocks": {
        "center": {"equity": 0.6, "volatility": 0.25, "inflation": 0.4, "geopolitical": 0.3,
                   "technology": 0.6, "sentiment": 0.6, "economic": 0.6, "monetary": 0.5},
        "radius": 0.15,
        "stability": 0.8,
        "probability": 0.35,
        "narrative": "Goldilocks economy — moderate growth, low volatility, stable geopolitics. The system's preferred state.",
    },
    "late_cycle_euphoria": {
        "center": {"equity": 0.8, "volatility": 0.15, "inflation": 0.4, "geopolitical": 0.25,
                   "technology": 0.75, "sentiment": 0.8, "economic": 0.7, "monetary": 0.4},
        "radius": 0.12,
        "stability": 0.5,
        "probability": 0.20,
        "narrative": "Late-cycle euphoria — high asset prices, low perceived risk, but fragility building beneath the surface.",
    },
    "stagflation": {
        "center": {"equity": 0.3, "volatility": 0.6, "inflation": 0.75, "geopolitical": 0.5,
                   "technology": 0.4, "sentiment": 0.25, "economic": 0.25, "monetary": 0.7},
        "radius": 0.15,
        "stability": 0.6,
        "probability": 0.12,
        "narrative": "Stagflation — stagnant growth, high inflation, policy dilemma. The system is trapped in an uncomfortable equilibrium.",
    },
    "crisis": {
        "center": {"equity": 0.15, "volatility": 0.85, "inflation": 0.5, "geopolitical": 0.7,
                   "technology": 0.3, "sentiment": 0.1, "economic": 0.1, "monetary": 0.8},
        "radius": 0.20,
        "stability": 0.4,
        "probability": 0.08,
        "narrative": "Crisis — systemic breakdown, fear-dominated, self-reinforcing panic spiral. Deep attractor once entered.",
    },
    "recovery": {
        "center": {"equity": 0.45, "volatility": 0.35, "inflation": 0.35, "geopolitical": 0.35,
                   "technology": 0.55, "sentiment": 0.45, "economic": 0.45, "monetary": 0.4},
        "radius": 0.15,
        "stability": 0.55,
        "probability": 0.15,
        "narrative": "Recovery — post-crisis rebuilding, cautious optimism, improving fundamentals but scar tissue remains.",
    },
    "innovation_surge": {
        "center": {"equity": 0.65, "volatility": 0.35, "inflation": 0.3, "geopolitical": 0.2,
                   "technology": 0.85, "sentiment": 0.7, "economic": 0.6, "monetary": 0.35},
        "radius": 0.12,
        "stability": 0.45,
        "probability": 0.10,
        "narrative": "Innovation surge — technological disruption accelerating, old models breaking, new wealth creation.",
    },
}


class PhaseSpaceMapper:
    def __init__(self, attractors: Optional[Dict] = None):
        self.attractors = attractors or ATTRACTOR_CATALOG

    def map_phase_space(self, domain_states: Dict[str, float]) -> PhaseSpaceResult:
        all_domains = sorted(set(domain_states.keys()) |
                            set(k for a in self.attractors.values() for k in a["center"]))
        state_vector = {}
        for d in all_domains:
            state_vector[d] = domain_states.get(d, 0.5)
        attractor_basins = self._find_attractors(state_vector)
        named_attr = {a.name: a for a in attractor_basins}
        current = min(attractor_basins, key=lambda a: a.radius)
        current_attr = current.name
        saddle_points = self._find_saddle_points(state_vector, attractor_basins)
        bifurcation_boundaries = self._find_bifurcations(state_vector, attractor_basins)
        escape_routes = self._find_escape_routes(state_vector, attractor_basins)
        nearest_boundary = min((b.distance_from_current for b in bifurcation_boundaries), default=1.0)
        lyapunov = self._estimate_lyapunov(state_vector, attractor_basins)
        resilience = self._compute_resilience(state_vector, current, bifurcation_boundaries)
        narrative = self._generate_narrative(current, bifurcation_boundaries, escape_routes, resilience)
        return PhaseSpaceResult(
            current_state_vector={k: round(v, 4) for k, v in state_vector.items()},
            current_attractor=current_attr,
            distance_to_nearest_boundary=round(nearest_boundary, 4),
            attractors=attractor_basins,
            saddle_points=saddle_points,
            bifurcation_boundaries=bifurcation_boundaries,
            escape_routes=escape_routes,
            phase_space_dimension=len(all_domains),
            lyapunov_exponent=round(lyapunov, 4),
            resilience=round(resilience, 4),
            narrative=narrative,
        )

    def _find_attractors(self, state_vector: Dict[str, float]) -> List[AttractorBasin]:
        basins = []
        for name, config in self.attractors.items():
            center = config["center"]
            radius = config["radius"]
            stability = config["stability"]
            probability = config["probability"]
            narrative = config["narrative"]
            common_domains = set(state_vector.keys()) & set(center.keys())
            if not common_domains:
                continue
            dist = math.sqrt(sum(
                (state_vector.get(d, 0.5) - center.get(d, 0.5)) ** 2
                for d in common_domains
            ) / len(common_domains))
            attraction_prob = math.exp(-dist ** 2 / (2 * radius ** 2))
            attractor_type = AttractorType.STABLE_EQUILIBRIUM
            if name in ("crisis", "stagflation"):
                attractor_type = AttractorType.LIMIT_CYCLE
            elif name == "innovation_surge":
                attractor_type = AttractorType.STRANGE_ATTRACTOR
            basins.append(AttractorBasin(
                name=name,
                center=center,
                radius=radius,
                attractor_type=attractor_type,
                stability=stability,
                domains=list(common_domains),
                probability=round(float(attraction_prob * probability), 4),
                narrative=narrative,
            ))
        total_prob = sum(b.probability for b in basins) or 1.0
        for b in basins:
            b.probability = round(b.probability / total_prob, 4)
        return sorted(basins, key=lambda b: -b.probability)

    def _find_saddle_points(self, state_vector, basins) -> List[SaddlePoint]:
        saddles = []
        if len(basins) < 2:
            return saddles
        for i in range(min(len(basins), 5)):
            for j in range(i + 1, min(len(basins), 5)):
                b1, b2 = basins[i], basins[j]
                common = set(b1.center.keys()) & set(b2.center.keys()) & set(state_vector.keys())
                if not common:
                    continue
                saddle_location = {}
                for d in common:
                    saddle_location[d] = (b1.center.get(d, 0.5) + b2.center.get(d, 0.5)) / 2
                eigenvalues = []
                for d in common:
                    diff = b2.center.get(d, 0.5) - b1.center.get(d, 0.5)
                    eigenvalues.append(float(diff * b1.stability * b2.stability))
                instability = float(np.mean([abs(e) for e in eigenvalues]))
                probability = (b1.probability + b2.probability) / 2 * instability
                transition = f"{b1.name} → {b2.name}"
                saddles.append(SaddlePoint(
                    location=saddle_location,
                    eigenvalues=[round(e, 4) for e in eigenvalues],
                    instability_index=round(float(instability), 4),
                    transition_paths=[transition],
                    narrative=f"Saddle point between {b1.name} and {b2.name}: small perturbation can push system to either attractor",
                ))
        return sorted(saddles, key=lambda s: -s.instability_index)[:5]

    def _find_bifurcations(self, state_vector, basins) -> List[BifurcationBoundary]:
        boundaries = []
        for basin in basins:
            common = set(basin.center.keys()) & set(state_vector.keys())
            if not common:
                continue
            for domain in common:
                current_val = state_vector.get(domain, 0.5)
                center_val = basin.center.get(domain, 0.5)
                distance = abs(current_val - center_val)
                threshold = center_val + basin.radius * (1 if current_val > center_val else -1)
                is_approaching = (current_val > threshold and current_val < threshold + 0.15) or \
                                 (current_val < threshold and current_val > threshold - 0.15)
                near_boundary = distance > basin.radius * 0.7
                if near_boundary or is_approaching:
                    bif_type = BifurcationType.SADDLE_NODE
                    if domain in ("volatility", "sentiment"):
                        bif_type = BifurcationType.HOPF
                    elif domain in ("monetary", "inflation"):
                        bif_type = BifurcationType.TRANSCRITICAL
                    elif domain == "technology":
                        bif_type = BifurcationType.PITCHFORK
                    boundaries.append(BifurcationBoundary(
                        boundary_type=bif_type,
                        domains=[domain],
                        threshold_values={domain: round(threshold, 3)},
                        distance_from_current=round(distance, 4),
                        probability_of_crossing=round(min(0.5, 0.1 + distance * 0.3), 4) if near_boundary else 0.05,
                        consequence=f"Crossing {domain} threshold shifts system toward {basin.name} attractor",
                        narrative=f"{domain} approaching {basin.name} boundary at {threshold:.2f} (distance: {distance:.2f})",
                    ))
        return sorted(boundaries, key=lambda b: b.distance_from_current)[:8]

    def _find_escape_routes(self, state_vector, basins) -> List[EscapeRoute]:
        routes = []
        if len(basins) < 2:
            return routes
        for i, from_b in enumerate(basins[:4]):
            for to_b in basins:
                if to_b.name == from_b.name:
                    continue
                common = set(from_b.center.keys()) & set(to_b.center.keys()) & set(state_vector.keys())
                if not common:
                    continue
                shifts = {}
                trigger_domains = []
                for d in common:
                    shift = to_b.center.get(d, 0.5) - from_b.center.get(d, 0.5)
                    shifts[d] = round(shift, 3)
                    if abs(shift) > 0.15:
                        trigger_domains.append(d)
                prob = from_b.probability * to_b.stability * 0.3 / max(len(basins), 1)
                prob = min(0.25, max(0.01, prob))
                horizon = int(30 + 180 * (1 - to_b.stability))
                routes.append(EscapeRoute(
                    from_attractor=from_b.name,
                    to_attractor=to_b.name,
                    trigger_domains=trigger_domains[:3],
                    required_shift=shifts,
                    probability=round(prob, 4),
                    time_horizon_days=horizon,
                    narrative=f"Transition from {from_b.name} to {to_b.name} requires shift in {', '.join(trigger_domains[:2])} over {horizon} days",
                ))
        return sorted(routes, key=lambda r: -r.probability)[:6]

    def _estimate_lyapunov(self, state_vector, basins) -> float:
        if not basins:
            return 0.0
        primary = basins[0]
        common = set(primary.center.keys()) & set(state_vector.keys())
        if not common:
            return 0.0
        dist = math.sqrt(sum(
            (state_vector.get(d, 0.5) - primary.center.get(d, 0.5)) ** 2
            for d in common
        ) / len(common))
        return -primary.stability * dist + 0.1 * (1 - primary.stability)

    def _compute_resilience(self, state_vector, current_attractor,
                           boundaries) -> float:
        resilience = current_attractor.stability
        near_boundary_count = sum(1 for b in boundaries if b.distance_from_current < 0.15)
        resilience -= near_boundary_count * 0.1
        return max(0.0, min(1.0, resilience))

    def _generate_narrative(self, current, boundaries, escapes, resilience) -> str:
        parts = [f"The system sits within the **{current.name}** attractor basin."]
        if resilience > 0.7:
            parts.append(f"Resilience is HIGH ({resilience:.0%}) — the system is stable within this attractor.")
        elif resilience > 0.4:
            parts.append(f"Resilience is MODERATE ({resilience:.0%}) — perturbations could shift the trajectory.")
        else:
            parts.append(f"Resilience is LOW ({resilience:.0%}) — the system is near a phase transition boundary.")
        if boundaries:
            nearest = boundaries[0]
            parts.append(f"Nearest bifurcation: {nearest.narrative}")
        if escapes:
            top_escape = escapes[0]
            parts.append(f"Most likely regime change: {top_escape.narrative}")
        return " ".join(parts)