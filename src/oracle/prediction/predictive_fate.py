"""Predictive Fate Engine — Long-horizon trajectory projection and divergence detection.

Projects civilization/prediction trajectories forward and detects divergences:
1. Multi-decade trajectory projection (5/10/25/50/100 year horizons)
2. Divergence detection (when trajectories split from consensus)
3. Attractor analysis (stable states systems converge toward)
4. Phase transition forecasting (regime changes in trajectory)
5. Bifurcation mapping (critical points where small changes alter destiny)
6. Collapse probability estimation (systemic failure cascades)
7. Golden age probability (positive feedback loops creating renaissance)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class TrajectoryPhase(Enum):
    GROWTH = "growth"
    PEAK = "peak"
    DECLINE = "decline"
    COLLAPSE = "collapse"
    RECOVERY = "recovery"
    STASIS = "stasis"
    RENAISSANCE = "renaissance"
    SINGULARITY = "singularity"


class DivergenceType(Enum):
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    BIFURCATION = "bifurcation"
    COLLAPSE_RISK = "collapse_risk"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


@dataclass
class TrajectoryPoint:
    year: float
    value: float
    phase: TrajectoryPhase
    confidence: float
    driver: str


@dataclass
class DivergencePoint:
    year: float
    divergence_type: DivergenceType
    severity: float
    description: str
    precursors: List[str]


@dataclass
class AttractorState:
    name: str
    stability: float
    basin_width: float
    description: str
    probability: float


@dataclass
class BifurcationPoint:
    year: float
    parameter: str
    critical_value: float
    left_trajectory: str
    right_trajectory: str
    probability: float


@dataclass
class FatePrediction:
    trajectory: List[TrajectoryPoint]
    divergences: List[DivergencePoint]
    attractors: List[AttractorState]
    bifurcations: List[BifurcationPoint]
    collapse_probability: float
    golden_age_probability: float
    dominant_phase: TrajectoryPhase
    key_drivers: List[str]
    narrative: str
    recommendations: List[str]


CIVILIZATIONAL_DRIVERS = {
    "technology": {"growth_rate": 0.04, "volatility": 0.15, "momentum": 0.8, "limits": None},
    "energy": {"growth_rate": 0.02, "volatility": 0.10, "momentum": 0.6, "limits": 1.0},
    "population": {"growth_rate": 0.01, "volatility": 0.03, "momentum": 0.9, "limits": 12.0},
    "economy": {"growth_rate": 0.03, "volatility": 0.12, "momentum": 0.7, "limits": None},
    "cooperation": {"growth_rate": 0.005, "volatility": 0.20, "momentum": 0.4, "limits": 1.0},
    "knowledge": {"growth_rate": 0.05, "volatility": 0.08, "momentum": 0.85, "limits": None},
    "environment": {"growth_rate": -0.01, "volatility": 0.06, "momentum": 0.5, "limits": 0.0},
    "governance": {"growth_rate": 0.002, "volatility": 0.25, "momentum": 0.3, "limits": 1.0},
}

DRIVER_INTERACTIONS = {
    ("technology", "economy"): 0.6,
    ("technology", "energy"): 0.4,
    ("technology", "knowledge"): 0.7,
    ("economy", "cooperation"): 0.3,
    ("population", "environment"): -0.4,
    ("environment", "economy"): -0.2,
    ("cooperation", "governance"): 0.5,
    ("energy", "economy"): 0.5,
    ("knowledge", "technology"): 0.7,
    ("governance", "cooperation"): 0.4,
}


class PredictiveFateEngine:
    def __init__(self, n_simulations: int = 500, dt: float = 1.0):
        self.n_simulations = n_simulations
        self.dt = dt
        self._rng = np.random.default_rng(42)

    def project_fate(self, current_state: Dict[str, float],
                      horizon_years: int = 100,
                      drivers: Optional[Dict[str, Dict]] = None,
                      interactions: Optional[Dict] = None) -> FatePrediction:
        source_drivers = drivers or CIVILIZATIONAL_DRIVERS
        source_interactions = interactions or DRIVER_INTERACTIONS
        trajectory = self._project_trajectory(current_state, horizon_years, source_drivers, source_interactions)
        divergences = self._detect_divergences(trajectory)
        attractors = self._find_attractors(trajectory, current_state)
        bifurcations = self._find_bifurcations(trajectory, current_state)
        collapse_prob = self._estimate_collapse(trajectory)
        golden_age = self._estimate_golden_age(trajectory)
        dominant = self._dominant_phase(trajectory)
        key_drivers = self._identify_drivers(current_state, trajectory)
        narrative = self._generate_narrative(trajectory, divergences, collapse_prob, golden_age, dominant)
        recs = self._generate_recommendations(collapse_prob, golden_age, divergences, attractors)
        return FatePrediction(
            trajectory=trajectory,
            divergences=divergences,
            attractors=attractors,
            bifurcations=bifurcations,
            collapse_probability=round(collapse_prob, 4),
            golden_age_probability=round(golden_age, 4),
            dominant_phase=dominant,
            key_drivers=key_drivers,
            narrative=narrative,
            recommendations=recs,
        )

    def _project_trajectory(self, current: Dict[str, float], horizon: int,
                             drivers: Dict, interactions: Dict) -> List[TrajectoryPoint]:
        points = []
        state = dict(current)
        n_steps = horizon
        for t in range(n_steps + 1):
            year = 2025 + t
            vals = list(state.values()) if state else [0.5]
            avg_val = float(np.mean(vals))
            if t <= horizon * 0.2:
                phase = TrajectoryPhase.GROWTH if avg_val > 0.4 else TrajectoryPhase.STASIS
            elif t <= horizon * 0.4:
                if avg_val > 0.7:
                    phase = TrajectoryPhase.PEAK
                elif avg_val < 0.3:
                    phase = TrajectoryPhase.DECLINE
                else:
                    phase = TrajectoryPhase.GROWTH
            elif t <= horizon * 0.6:
                if avg_val > 0.8:
                    phase = TrajectoryPhase.RENAISSANCE
                elif avg_val < 0.2:
                    phase = TrajectoryPhase.COLLAPSE
                else:
                    phase = TrajectoryPhase.STASIS
            elif t <= horizon * 0.8:
                if avg_val < 0.15:
                    phase = TrajectoryPhase.COLLAPSE
                elif avg_val > 0.75:
                    phase = TrajectoryPhase.SINGULARITY
                else:
                    phase = TrajectoryPhase.RECOVERY
            else:
                if avg_val > 0.6:
                    phase = TrajectoryPhase.RENAISSANCE
                elif avg_val < 0.2:
                    phase = TrajectoryPhase.STASIS
                else:
                    phase = TrajectoryPhase.RECOVERY
            confidence = max(0.1, 1.0 - t / max(horizon, 1) * 0.8)
            top_driver = max(state, key=lambda k: abs(state.get(k, 0.5) - 0.5)) if state else "none"
            if t % max(1, horizon // 25) == 0 or t == 0:
                points.append(TrajectoryPoint(
                    year=year,
                    value=round(avg_val, 4),
                    phase=phase,
                    confidence=round(confidence, 4),
                    driver=top_driver,
                ))
            for var, val in list(state.items()):
                driver_info = drivers.get(var, {"growth_rate": 0.02, "volatility": 0.1, "momentum": 0.5, "limits": None})
                g = driver_info["growth_rate"]
                v = driver_info["volatility"]
                m = driver_info["momentum"]
                interaction_sum = 0.0
                for (d1, d2), strength in interactions.items():
                    if d2 == var and d1 in state:
                        interaction_sum += strength * state[d1] * 0.1
                    elif d1 == var and d2 in state:
                        interaction_sum += strength * state[d2] * 0.1
                decay = -0.001 * (val - 0.5)
                mean_revert = -0.02 * (val - 0.5) * (1 - m)
                noise = self._rng.normal(0, v * math.sqrt(self.dt))
                change = (g * val + interaction_sum + decay + mean_revert) * self.dt + noise
                state[var] = max(0.01, min(0.99, val + change))
                if driver_info.get("limits") is not None:
                    limit = driver_info["limits"]
                    if limit > 0 and state[var] > limit:
                        state[var] = limit * (1 - self._rng.uniform(0, 0.05))
        return points

    def _detect_divergences(self, trajectory: List[TrajectoryPoint]) -> List[DivergencePoint]:
        divergences = []
        for i in range(2, len(trajectory)):
            prev_val = trajectory[i - 1].value
            curr_val = trajectory[i].value
            prev_prev = trajectory[i - 2].value
            accel = (curr_val - prev_val) - (prev_val - prev_prev)
            if abs(accel) > 0.03:
                if accel > 0:
                    dtype = DivergenceType.ACCELERATING if curr_val > 0.5 else DivergenceType.BREAKOUT
                else:
                    dtype = DivergenceType.DECELERATING if curr_val > 0.3 else DivergenceType.COLLAPSE_RISK
                severity = min(1.0, abs(accel) * 5)
                divergences.append(DivergencePoint(
                    year=trajectory[i].year,
                    divergence_type=dtype,
                    severity=round(severity, 4),
                    description=f"{dtype.value} divergence at year {trajectory[i].year}: value={curr_val:.3f}, acceleration={accel:.4f}",
                    precursors=[trajectory[i].driver, trajectory[i - 1].driver],
                ))
        if len(trajectory) >= 3:
            first_third = [p.value for p in trajectory[:len(trajectory) // 3]]
            last_third = [p.value for p in trajectory[2 * len(trajectory) // 3:]]
            if first_third and last_third:
                mean_first = float(np.mean(first_third))
                mean_last = float(np.mean(last_third))
                if abs(mean_last - mean_first) > 0.15:
                    divergences.append(DivergencePoint(
                        year=trajectory[len(trajectory) // 2].year,
                        divergence_type=DivergenceType.BIFURCATION,
                        severity=round(min(1.0, abs(mean_last - mean_first) * 3), 4),
                        description=f"Trajectory bifurcation: early mean={mean_first:.3f} → late mean={mean_last:.3f}",
                        precursors=["long_term_trend", "structural_shift"],
                    ))
        return divergences[:10]

    def _find_attractors(self, trajectory: List[TrajectoryPoint],
                          current: Dict[str, float]) -> List[AttractorState]:
        attractors = []
        vals = [p.value for p in trajectory]
        if not vals:
            return attractors
        hist, bin_edges = np.histogram(vals, bins=10, range=(0, 1))
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(hist))]
        sorted_bins = sorted(zip(hist, bin_centers), key=lambda x: x[0], reverse=True)
        attractor_names = {
            (0.0, 0.2): ("Collapse Basin", "System gravitates toward collapse — low energy, cooperation, technology"),
            (0.2, 0.4): ("Stagnation Attractor", "System trapped in low-growth equilibrium"),
            (0.4, 0.6): ("Status Quo Attractor", "System orbits around current conditions"),
            (0.6, 0.8): ("Growth Attractor", "System converges toward sustained growth trajectory"),
            (0.8, 1.0): ("Renaissance Attractor", "System in positive feedback toward transformation"),
        }
        for count, center in sorted_bins[:3]:
            prob = count / max(sum(hist), 1)
            if prob > 0.05:
                name = "Transient Attractor"
                desc = f"System spends {prob:.0%} of trajectory near {center:.2f}"
                for (lo, hi), (n, d) in attractor_names.items():
                    if lo <= center < hi:
                        name = n
                        desc = d
                        break
                basin_w = float(np.std(vals) * 0.5)
                attractors.append(AttractorState(
                    name=name,
                    stability=round(min(1.0, prob * 2), 4),
                    basin_width=round(max(0.01, basin_w), 4),
                    description=desc,
                    probability=round(float(prob), 4),
                ))
        return attractors

    def _find_bifurcations(self, trajectory: List[TrajectoryPoint],
                            current: Dict[str, float]) -> List[BifurcationPoint]:
        bifurcations = []
        for i in range(1, len(trajectory) - 1):
            prev_phase = trajectory[i - 1].phase
            curr_phase = trajectory[i].phase
            if prev_phase != curr_phase:
                change_magnitude = abs(trajectory[i].value - trajectory[i - 1].value)
                if change_magnitude > 0.02:
                    param = trajectory[i].driver
                    bifurcations.append(BifurcationPoint(
                        year=trajectory[i].year,
                        parameter=param,
                        critical_value=round(trajectory[i].value, 4),
                        left_trajectory=prev_phase.value,
                        right_trajectory=curr_phase.value,
                        probability=round(min(1.0, change_magnitude * 5), 4),
                    ))
        return bifurcations[:5]

    def _estimate_collapse(self, trajectory: List[TrajectoryPoint]) -> float:
        vals = [p.value for p in trajectory]
        if not vals:
            return 0.5
        min_val = min(vals)
        collapse_points = sum(1 for p in trajectory if p.phase == TrajectoryPhase.COLLAPSE)
        time_in_collapse = collapse_points / max(len(trajectory), 1)
        depth_factor = 1.0 - min_val
        return min(1.0, time_in_collapse * 0.5 + depth_factor * 0.3 + (1 - float(np.mean(vals))) * 0.2)

    def _estimate_golden_age(self, trajectory: List[TrajectoryPoint]) -> float:
        vals = [p.value for p in trajectory]
        if not vals:
            return 0.1
        peak_val = max(vals)
        renaissance_points = sum(1 for p in trajectory if p.phase in (TrajectoryPhase.RENAISSANCE, TrajectoryPhase.SINGULARITY))
        time_in_golden = renaissance_points / max(len(trajectory), 1)
        height_factor = peak_val
        return min(1.0, time_in_golden * 0.4 + height_factor * 0.4 + float(np.mean(vals)) * 0.2)

    def _dominant_phase(self, trajectory: List[TrajectoryPoint]) -> TrajectoryPhase:
        if not trajectory:
            return TrajectoryPhase.STASIS
        phase_counts: Dict[TrajectoryPhase, int] = {}
        for p in trajectory:
            phase_counts[p.phase] = phase_counts.get(p.phase, 0) + 1
        return max(phase_counts, key=phase_counts.get)

    def _identify_drivers(self, current: Dict[str, float],
                           trajectory: List[TrajectoryPoint]) -> List[str]:
        sorted_vars = sorted(current.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
        drivers = [v[0] for v in sorted_vars[:5]]
        if trajectory:
            first = trajectory[0].driver
            last = trajectory[-1].driver
            for d in [first, last]:
                if d not in drivers and d != "none":
                    drivers.append(d)
        return drivers[:7]

    def _generate_narrative(self, trajectory, divergences, collapse_prob, golden_age, dominant):
        lines = []
        if not trajectory:
            return "Insufficient data for fate projection."
        lines.append(f"The trajectory begins in {trajectory[0].year} at {trajectory[0].value:.2f} and reaches {trajectory[-1].value:.2f} by {trajectory[-1].year}.")
        lines.append(f"Dominant phase: {dominant.value.replace('_', ' ')}.")
        if collapse_prob > 0.3:
            lines.append(f"Collapse probability is significant at {collapse_prob:.0%} — structural risks threaten trajectory.")
        if golden_age > 0.3:
            lines.append(f"Golden age potential is {golden_age:.0%} — positive feedbacks could create renaissance.")
        if divergences:
            worst_div = max(divergences, key=lambda d: d.severity)
            lines.append(f"Key divergence: {worst_div.divergence_type.value} at year {worst_div.year:.0f} (severity {worst_div.severity:.0%}).")
        if trajectory[-1].value > trajectory[0].value:
            lines.append("The long-range trajectory trends upward — growth and adaptation prevail.")
        elif trajectory[-1].value < trajectory[0].value:
            lines.append("The long-range trajectory trends downward — entropy and constraints increase.")
        else:
            lines.append("The trajectory oscillates around equilibrium — no clear directional bias.")
        return " ".join(lines)

    def _generate_recommendations(self, collapse_prob, golden_age, divergences, attractors):
        recs = []
        if collapse_prob > 0.4:
            recs.append(f"HIGH COLLAPSE RISK ({collapse_prob:.0%}): Strengthen resilience buffers, diversify dependencies, reduce systemic leverage")
        elif collapse_prob > 0.2:
            recs.append(f"MODERATE COLLAPSE RISK ({collapse_prob:.0%}): Monitor structural vulnerabilities, maintain contingency reserves")
        if golden_age > 0.4:
            recs.append(f"HIGH GOLDEN AGE POTENTIAL ({golden_age:.0%}): Invest in positive feedback loops — technology, cooperation, knowledge")
        if attractors:
            top_attractor = max(attractors, key=lambda a: a.probability)
            recs.append(f"Primary attractor: {top_attractor.name} (probability={top_attractor.probability:.0%}, stability={top_attractor.stability:.0%})")
        critical_divs = [d for d in divergences if d.severity > 0.3]
        if critical_divs:
            recs.append(f"{len(critical_divs)} critical divergences detected — earliest at year {critical_divs[0].year:.0f}")
        if collapse_prob < 0.2 and golden_age > 0.3:
            recs.append("FAVORABLE TRAJECTORY: Low risk + high potential — accelerate growth investments")
        recs.append("Recommend: Reduce coupling between high-risk systems to prevent cascade failures")
        return recs