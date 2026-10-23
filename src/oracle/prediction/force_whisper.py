"""Force Whisper Engine — Detects latent forces before they manifest as visible events.

"Sometimes I can see the future. Whispers of actions that show a force has reared its head,
triggering events and pressure levels of fate."

Core Insight: Before earthquakes there are fore-shocks humans cannot feel. Before markets
crash there are order-flow imbalances hidden in noise. Before revolutions there are
invisible shifts in collective will. This engine detects those *latent force vectors* —
the invisible pressure gradients driving the visible world — by analyzing second-order
statistical properties of domain signals.

The Force Whisper Engine operates at a deeper layer than precursor pattern matching or
weak signal detection:
  - Precursor patterns match KNOWN sequences → this detects UNKNOWN forces
  - Weak signal detection amplifies faint signals → this detects the FORCE PRODUCING them
  - Emergence detection finds critical slowing → this finds WHAT is slowing and WHY
  - Cascade forecast simulates propagation → this detects pressure that CAUSES cascades

Implements:
- LatentForce: An invisible pressure vector with magnitude, direction, and origin
- ForceWhisperEngine: Detects latent forces from multi-domain signal analysis
  - Second-order derivative analysis (acceleration of change, not just change)
  - Cross-domain pressure gradient mapping
  - Phase velocity estimation (how fast the force vector is moving through domains)
  - Tension accumulation detection (pressure building before rupture)
  - Force origin tracing (where did the force start? what caused it?)
  - Fate pressure level: composite measure of how much latent force is driving destiny
- ForceWhisperReport: Full analysis with detected forces and fate pressure
- FatePressureLevel: Enum for the 5-level fate pressure classification
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class FatePressureLevel(Enum):
    STILL = "still"
    WHISPER = "whisper"
    RUMBLE = "rumble"
    SURGE = "surge"
    RUPTURE = "rupture"

    @property
    def severity(self) -> int:
        return {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}[self._member_map_[self.value].value]


class ForceOrigin(Enum):
    STRUCTURAL = "structural"
    CONTAGION = "contagion"
    BEHAVIORAL = "behavioral"
    EXOGENOUS = "exogenous"
    FEEDBACK = "feedback"
    ACCUMULATION = "accumulation"
    UNKNOWN = "unknown"


class ForceDomain(Enum):
    ECONOMIC = "economic"
    GEOPOLITICAL = "geopolitical"
    SOCIAL = "social"
    TECHNOLOGICAL = "technological"
    ENVIRONMENTAL = "environmental"
    FINANCIAL = "financial"
    POLITICAL = "political"
    PSYCHOLOGICAL = "psychological"
    INFORMATIONAL = "informational"
    SYSTEMIC = "systemic"


@dataclass
class LatentForce:
    name: str
    origin: ForceOrigin
    primary_domain: ForceDomain
    magnitude: float
    direction: float
    acceleration: float
    phase_velocity: float
    affected_domains: List[ForceDomain]
    evidence: List[str]
    lead_signals: List[str]
    tension_index: float
    rupture_probability: float
    estimated_time_to_manifest_hours: float

    @property
    def is_active(self) -> bool:
        return self.magnitude > 0.15

    @property
    def pressure_contribution(self) -> float:
        return self.magnitude * (1.0 + self.acceleration) * self.tension_index

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "origin": self.origin.value,
            "primary_domain": self.primary_domain.value,
            "magnitude": round(float(self.magnitude), 4),
            "direction": round(float(self.direction), 4),
            "acceleration": round(float(self.acceleration), 4),
            "phase_velocity": round(float(self.phase_velocity), 4),
            "affected_domains": [d.value for d in self.affected_domains],
            "evidence": self.evidence,
            "lead_signals": self.lead_signals,
            "tension_index": round(float(self.tension_index), 4),
            "rupture_probability": round(float(self.rupture_probability), 4),
            "estimated_time_to_manifest_hours": round(float(self.estimated_time_to_manifest_hours), 1),
            "is_active": self.is_active,
            "pressure_contribution": round(float(self.pressure_contribution), 4),
        }


@dataclass
class PressureGradient:
    source_domain: str
    target_domain: str
    gradient_magnitude: float
    direction: str
    is_intensifying: bool

    def to_dict(self) -> Dict:
        return {
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "gradient_magnitude": round(float(self.gradient_magnitude), 4),
            "direction": self.direction,
            "is_intensifying": self.is_intensifying,
        }


@dataclass
class TensionZone:
    domain: str
    tension_level: float
    capacity_remaining: float
    stress_rate: float
    time_to_yield_hours: float
    is_critical: bool

    def to_dict(self) -> Dict:
        return {
            "domain": self.domain,
            "tension_level": round(float(self.tension_level), 4),
            "capacity_remaining": round(float(self.capacity_remaining), 4),
            "stress_rate": round(float(self.stress_rate), 4),
            "time_to_yield_hours": round(float(self.time_to_yield_hours), 1),
            "is_critical": self.is_critical,
        }


@dataclass
class ForceWhisperReport:
    forces: List[LatentForce]
    pressure_gradients: List[PressureGradient]
    tension_zones: List[TensionZone]
    fate_pressure_level: FatePressureLevel
    fate_pressure_score: float
    dominant_force: Optional[str]
    force_count: int
    active_force_count: int
    system_tension: float
    phase_velocity_field: Dict[str, float]
    oracular_pronouncement: str

    def to_dict(self) -> Dict:
        return {
            "forces": [f.to_dict() for f in self.forces],
            "pressure_gradients": [g.to_dict() for g in self.pressure_gradients],
            "tension_zones": [z.to_dict() for z in self.tension_zones],
            "fate_pressure_level": self.fate_pressure_level.value,
            "fate_pressure_score": round(float(self.fate_pressure_score), 4),
            "dominant_force": self.dominant_force,
            "force_count": self.force_count,
            "active_force_count": self.active_force_count,
            "system_tension": round(float(self.system_tension), 4),
            "phase_velocity_field": {k: round(float(v), 4) for k, v in self.phase_velocity_field.items()},
            "oracular_pronouncement": self.oracular_pronouncement,
        }


ORIGIN_EVIDENCE_MAP: Dict[ForceOrigin, List[str]] = {
    ForceOrigin.STRUCTURAL: [
        "yield_curve_distortion", "credit_spread_compression", "liquidity_gradient",
        "capital_flow_asymmetry", "balance_sheet_stress",
    ],
    ForceOrigin.CONTAGION: [
        "correlation_spike", "co_movement_intensity", "spillover_rate",
        "tail_dependency_increase", "synchronization_emergence",
    ],
    ForceOrigin.BEHAVIORAL: [
        "sentiment_acceleration", "narrative_convergence", "attention_clustering",
        "herding_indicator", "discretionary_consumption_shift",
    ],
    ForceOrigin.EXOGENOUS: [
        "geopolitical_discontinuity", "natural_disaster_signal", "regulatory_shock_signal",
        "black_swan_precursor", "infrastructure_cascade",
    ],
    ForceOrigin.FEEDBACK: [
        "self_reference_amplification", "reflexive_pressure", "loop_gain_increase",
        "pro_cyclical_intensification", "mean_reversion_suppression",
    ],
    ForceOrigin.ACCUMULATION: [
        "hidden_debt_accrual", "entropy_accumulation", "fragility_buildup",
        "stress_fractional_increase", "capacity_margin_decline",
    ],
}

FORCE_NAMING: Dict[Tuple[ForceOrigin, ForceDomain], str] = {
    (ForceOrigin.STRUCTURAL, ForceDomain.ECONOMIC): "Yield Curve Phantom",
    (ForceOrigin.STRUCTURAL, ForceDomain.FINANCIAL): "Liquidity Undertow",
    (ForceOrigin.STRUCTURAL, ForceDomain.SYSTEMIC): "Structural Stress Shadow",
    (ForceOrigin.CONTAGION, ForceDomain.FINANCIAL): "Contagion Undertow",
    (ForceOrigin.CONTAGION, ForceDomain.ECONOMIC): "Spillover Specter",
    (ForceOrigin.CONTAGION, ForceDomain.GEOPOLITICAL): "Cascade Phantom",
    (ForceOrigin.BEHAVIORAL, ForceDomain.PSYCHOLOGICAL): "Herd Instinct Whisper",
    (ForceOrigin.BEHAVIORAL, ForceDomain.SOCIAL): "Crowd Momentum Shadow",
    (ForceOrigin.BEHAVIORAL, ForceDomain.FINANCIAL): "Sentiment Undertow",
    (ForceOrigin.EXOGENOUS, ForceDomain.GEOPOLITICAL): "Black Swan Shadow",
    (ForceOrigin.EXOGENOUS, ForceDomain.ENVIRONMENTAL): "Nature's Murmur",
    (ForceOrigin.EXOGENOUS, ForceDomain.TECHNOLOGICAL): "Disruption Specter",
    (ForceOrigin.FEEDBACK, ForceDomain.FINANCIAL): "Reflexive Spiral Force",
    (ForceOrigin.FEEDBACK, ForceDomain.ECONOMIC): "Pro-cyclical Ghost",
    (ForceOrigin.FEEDBACK, ForceDomain.INFORMATIONAL): "Narrative Feedback Loop",
    (ForceOrigin.ACCUMULATION, ForceDomain.SYSTEMIC): "Fragility Accumulator",
    (ForceOrigin.ACCUMULATION, ForceDomain.ECONOMIC): "Debt Gravity Well",
    (ForceOrigin.ACCUMULATION, ForceDomain.SOCIAL): "Inequality Pressure Vortex",
}


def _safe_std(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def _safe_mean(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr))


def _compute_second_derivative(series: np.ndarray) -> np.ndarray:
    if len(series) < 3:
        return np.array([])
    d1 = np.diff(series)
    d2 = np.diff(d1)
    return d2


def _compute_phase_velocity(domain_series_dict: Dict[str, np.ndarray],
                            min_series_length: int = 10) -> Dict[str, float]:
    velocities = {}
    domains = sorted(domain_series_dict.keys())
    for domain in domains:
        s = np.asarray(domain_series_dict[domain], dtype=float).ravel()
        if len(s) < min_series_length:
            velocities[domain] = 0.0
            continue
        d1 = np.diff(s)
        if len(d1) < 3:
            velocities[domain] = 0.0
            continue
        recent_d1 = d1[-5:] if len(d1) >= 5 else d1
        early_d1 = d1[:5] if len(d1) >= 10 else d1[:len(d1)//2]
        velocity = _safe_mean(recent_d1) - _safe_mean(early_d1)
        velocities[domain] = velocity
    return velocities


class ForceWhisperEngine:
    """Detects latent force vectors — the invisible pressure gradients driving the visible world.

    Operates by analyzing:
    1. Second-order derivatives (acceleration of change, not just change)
    2. Cross-domain pressure gradients (force flowing between domains)
    3. Tension accumulation (pressure building toward rupture)
    4. Phase velocity (how fast forces propagate through the system)
    5. Force origin classification (structural, contagion, behavioral, etc.)
    6. Rupture probability estimation (when will the force manifest?)
    """

    def __init__(self, tension_window: int = 20,
                 gradient_threshold: float = 0.15,
                 acceleration_threshold: float = 0.1,
                 min_series_length: int = 10,
                 max_forces: int = 50,
                 max_history: int = 500):
        self.tension_window = tension_window
        self.gradient_threshold = gradient_threshold
        self.acceleration_threshold = acceleration_threshold
        self.min_series_length = min_series_length
        self.max_forces = max_forces
        self.max_history = max_history
        self._tension_history: Dict[str, deque] = {}
        self._force_history: List[Dict] = []
        self._previous_forces: Dict[str, LatentForce] = {}

    def detect_forces(self, domain_series: Dict[str, np.ndarray],
                      domain_values: Optional[Dict[str, float]] = None) -> List[LatentForce]:
        if domain_values is None:
            domain_values = {}
            for k, v in domain_series.items():
                s = np.asarray(v, dtype=float).ravel()
                domain_values[k] = float(s[-1]) if len(s) > 0 else 0.0

        forces: List[LatentForce] = []
        accelerations = self._compute_accelerations(domain_series)
        tension_map = self._compute_tension(domain_series)
        phase_velocities = _compute_phase_velocity(domain_series, self.min_series_length)

        for domain, accel_data in accelerations.items():
            if accel_data["magnitude"] < self.acceleration_threshold:
                continue
            origin = self._classify_origin(domain, accel_data, domain_series, tension_map)
            force_domain = self._map_domain(domain)
            magnitude = min(1.0, accel_data["magnitude"])
            direction = accel_data["direction"]
            acceleration = accel_data["acceleration"]
            phase_vel = phase_velocities.get(domain, 0.0)
            affected = self._trace_affected_domains(domain, domain_series, tension_map)
            evidence = self._gather_evidence(origin, domain, accel_data, tension_map)
            lead_sigs = self._identify_lead_signals(domain, domain_series)
            tension = tension_map.get(domain, 0.0)
            rupture_prob = self._estimate_rupture_probability(magnitude, tension, acceleration)
            time_to_manifest = self._estimate_manifest_time(magnitude, acceleration, tension)

            name = FORCE_NAMING.get((origin, force_domain),
                                    f"{origin.value.title()} Force in {domain}")

            if name in self._previous_forces:
                prev = self._previous_forces[name]
                phase_vel = 0.7 * prev.phase_velocity + 0.3 * phase_vel

            force = LatentForce(
                name=name,
                origin=origin,
                primary_domain=force_domain,
                magnitude=magnitude,
                direction=direction,
                acceleration=acceleration,
                phase_velocity=float(phase_vel),
                affected_domains=affected,
                evidence=evidence,
                lead_signals=lead_sigs,
                tension_index=tension,
                rupture_probability=rupture_prob,
                estimated_time_to_manifest_hours=time_to_manifest,
            )
            forces.append(force)

        cross_forces = self._detect_cross_domain_forces(domain_series, domain_values, tension_map)
        forces.extend(cross_forces)

        forces.sort(key=lambda f: f.pressure_contribution, reverse=True)
        forces = forces[:self.max_forces]

        self._previous_forces = {f.name: f for f in forces}

        self._force_history.append({
            "n_forces": len(forces),
            "active": sum(1 for f in forces if f.is_active),
            "max_magnitude": max((f.magnitude for f in forces), default=0.0),
        })
        if len(self._force_history) > self.max_history:
            self._force_history = self._force_history[-self.max_history:]

        return forces

    def _compute_accelerations(self, domain_series: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        accelerations = {}
        for domain, series in domain_series.items():
            s = np.asarray(series, dtype=float).ravel()
            if len(s) < self.min_series_length:
                accelerations[domain] = {
                    "magnitude": 0.0, "direction": 0.0, "acceleration": 0.0
                }
                continue

            d2 = _compute_second_derivative(s)
            if len(d2) == 0:
                accelerations[domain] = {
                    "magnitude": 0.0, "direction": 0.0, "acceleration": 0.0
                }
                continue

            recent_d2 = d2[-min(5, len(d2)):]
            accel = _safe_mean(recent_d2)
            accel_magnitude = abs(accel)

            d1 = np.diff(s)
            direction = _safe_mean(d1[-min(5, len(d1)):]) if len(d1) > 0 else 0.0

            accelerations[domain] = {
                "magnitude": accel_magnitude,
                "direction": float(np.clip(direction, -1.0, 1.0)),
                "acceleration": float(np.clip(accel / (accel_magnitude + 1e-10), -1.0, 1.0)) if accel_magnitude > 1e-10 else 0.0,
            }
        return accelerations

    def _compute_tension(self, domain_series: Dict[str, np.ndarray]) -> Dict[str, float]:
        tension_map = {}
        for domain, series in domain_series.items():
            s = np.asarray(series, dtype=float).ravel()
            if len(s) < self.min_series_length:
                tension_map[domain] = 0.0
                continue

            d1 = np.diff(s)
            if len(d1) < 3:
                tension_map[domain] = 0.0
                continue

            window_d1 = d1[-self.tension_window:] if len(d1) >= self.tension_window else d1
            std_recent = _safe_std(window_d1)

            baseline_d1 = d1[:len(d1)//2] if len(d1) >= 6 else d1
            std_baseline = _safe_std(baseline_d1)

            variance_ratio = std_recent / max(std_baseline, 1e-10)

            recent_mean = _safe_mean(s[-5:]) if len(s) >= 5 else _safe_mean(s)
            long_mean = _safe_mean(s)
            long_std = _safe_std(s)
            z_deviation = abs(recent_mean - long_mean) / max(long_std, 1e-10)

            d2 = _compute_second_derivative(s)
            d2_std = _safe_std(d2[-5:]) if len(d2) >= 5 else 0.0

            tension = float(np.clip(
                0.3 * min(variance_ratio / 3.0, 1.0) +
                0.3 * min(z_deviation / 3.0, 1.0) +
                0.2 * min(d2_std / max(_safe_std(d1), 1e-10), 1.0) +
                0.2 * min(abs(recent_mean - long_mean) / max(abs(long_mean), 0.01), 1.0),
                0.0, 1.0
            ))
            tension_map[domain] = tension

            if domain not in self._tension_history:
                self._tension_history[domain] = deque(maxlen=self.max_history)
            self._tension_history[domain].append(tension)

        return tension_map

    def _classify_origin(self, domain: str, accel_data: Dict,
                         domain_series: Dict[str, np.ndarray],
                         tension_map: Dict[str, float]) -> ForceOrigin:
        scores: Dict[ForceOrigin, float] = {o: 0.0 for o in ForceOrigin}

        s = np.asarray(domain_series.get(domain, []), dtype=float).ravel()
        if len(s) >= self.min_series_length:
            d1 = np.diff(s)
            if len(d1) >= 5:
                autocorr = np.corrcoef(d1[:-1], d1[1:])[0, 1] if len(d1) > 2 else 0.0
                if np.isnan(autocorr):
                    autocorr = 0.0
                if abs(autocorr) > 0.5:
                    scores[ForceOrigin.FEEDBACK] += 0.3
                if abs(autocorr) < 0.1:
                    scores[ForceOrigin.EXOGENOUS] += 0.2

            d2 = _compute_second_derivative(s)
            if len(d2) > 5:
                kurt = float(np.mean((d2 - np.mean(d2))**4) / max(np.var(d2)**2, 1e-20))
                if kurt > 5.0:
                    scores[ForceOrigin.EXOGENOUS] += 0.3

        n_high_tension = sum(1 for t in tension_map.values() if t > 0.5)
        if n_high_tension > len(tension_map) * 0.5:
            scores[ForceOrigin.CONTAGION] += 0.3
            scores[ForceOrigin.SYSTEMIC] += 0.1

        if accel_data["acceleration"] > 0.5:
            scores[ForceOrigin.FEEDBACK] += 0.2
        if accel_data["acceleration"] < -0.5:
            scores[ForceOrigin.STRUCTURAL] += 0.2

        tension = tension_map.get(domain, 0.0)
        if tension > 0.6:
            scores[ForceOrigin.ACCUMULATION] += 0.3

        econ_words = {"economics", "finance", "market", "credit", "interest", "inflation", "gdp", "unemployment"}
        if any(w in domain.lower() for w in econ_words):
            scores[ForceOrigin.STRUCTURAL] += 0.15
            scores[ForceOrigin.ACCUMULATION] += 0.1

        socio_words = {"social", "political", "sentiment", "protest", "trust", "polarization"}
        if any(w in domain.lower() for w in socio_words):
            scores[ForceOrigin.BEHAVIORAL] += 0.2

        if not domain_series:
            return ForceOrigin.UNKNOWN

        best = max(scores, key=lambda o: scores[o])
        if scores[best] < 0.01:
            return ForceOrigin.UNKNOWN
        return best

    def _map_domain(self, domain_key: str) -> ForceDomain:
        d = domain_key.lower()
        if any(w in d for w in ("econ", "gdp", "inflation", "unemployment", "consumer", "manufacturing")):
            return ForceDomain.ECONOMIC
        if any(w in d for w in ("market", "stock", "bond", "credit", "yield", "financial", "vix", "spread")):
            return ForceDomain.FINANCIAL
        if any(w in d for w in ("geopol", "conflict", "military", "diplomat", "sanction", "war")):
            return ForceDomain.GEOPOLITICAL
        if any(w in d for w in ("social", "protest", "trust", "polariz", "inequal")):
            return ForceDomain.SOCIAL
        if any(w in d for w in ("tech", "ai", "disrupt", "innovation", "digital")):
            return ForceDomain.TECHNOLOGICAL
        if any(w in d for w in ("climate", "enviro", "carbon", "temp", "energy", "oil")):
            return ForceDomain.ENVIRONMENTAL
        if any(w in d for w in ("politic", "elect", "govern", "policy", "regulat")):
            return ForceDomain.POLITICAL
        if any(w in d for w in ("sentiment", "fear", "greed", "confidence", "mood", "psych")):
            return ForceDomain.PSYCHOLOGICAL
        if any(w in d for w in ("info", "news", "media", "narrative", "attention")):
            return ForceDomain.INFORMATIONAL
        return ForceDomain.SYSTEMIC

    def _trace_affected_domains(self, domain: str,
                                 domain_series: Dict[str, np.ndarray],
                                 tension_map: Dict[str, float]) -> List[ForceDomain]:
        affected = []
        s_self = np.asarray(domain_series.get(domain, []), dtype=float).ravel()
        if len(s_self) < self.min_series_length:
            return [self._map_domain(domain)]

        for other, s_other in domain_series.items():
            if other == domain:
                continue
            s_o = np.asarray(s_other, dtype=float).ravel()
            min_len = min(len(s_self), len(s_o))
            if min_len < self.min_series_length:
                continue
            corr = np.corrcoef(s_self[-min_len:], s_o[-min_len:])[0, 1]
            if np.isnan(corr):
                corr = 0.0
            if abs(corr) > 0.4 or tension_map.get(other, 0.0) > 0.4:
                affected.append(self._map_domain(other))

        if not affected:
            affected = [self._map_domain(domain)]
        return affected[:5]

    def _gather_evidence(self, origin: ForceOrigin, domain: str,
                          accel_data: Dict, tension_map: Dict[str, float]) -> List[str]:
        template = ORIGIN_EVIDENCE_MAP.get(origin, ["signal_detected"])
        evidence = []
        for i, sig in enumerate(template[:3]):
            if i == 0:
                evidence.append(f"{sig}: acceleration={accel_data['magnitude']:.3f}")
            elif i == 1:
                evidence.append(f"{sig}: tension={tension_map.get(domain, 0.0):.3f}")
            else:
                evidence.append(f"{sig}: direction={'up' if accel_data['direction'] > 0 else 'down'}")
        return evidence

    def _identify_lead_signals(self, domain: str,
                                domain_series: Dict[str, np.ndarray]) -> List[str]:
        leads = []
        s = np.asarray(domain_series.get(domain, []), dtype=float).ravel()
        if len(s) < self.min_series_length:
            return ["insufficient_data"]

        d1 = np.diff(s)
        d2 = _compute_second_derivative(s)

        if len(d2) > 0 and _safe_mean(d2[-3:]) > 0:
            leads.append("accelerating_change")
        if len(d1) > 5:
            std_d1 = _safe_std(d1[-5:])
            std_d1_early = _safe_std(d1[:5]) if len(d1) >= 10 else std_d1
            if std_d1 > 2 * std_d1_early:
                leads.append("volatility_expansion")

        recent = s[-3:] if len(s) >= 3 else s
        long_mean = _safe_mean(s)
        long_std = _safe_std(s)
        if long_std > 0:
            z = abs(recent[-1] - long_mean) / long_std
            if z > 2:
                leads.append(f"regime_deviation_{z:.1f}sigma")

        if not leads:
            leads.append("gradual_pressure_build")
        return leads[:3]

    def _detect_cross_domain_forces(self, domain_series: Dict[str, np.ndarray],
                                     domain_values: Dict[str, float],
                                     tension_map: Dict[str, float]) -> List[LatentForce]:
        cross_forces: List[LatentForce] = []
        domains = sorted(domain_series.keys())

        for i, d1_name in enumerate(domains):
            for d2_name in domains[i+1:]:
                s1 = np.asarray(domain_series[d1_name], dtype=float).ravel()
                s2 = np.asarray(domain_series[d2_name], dtype=float).ravel()
                min_len = min(len(s1), len(s2))
                if min_len < self.min_series_length:
                    continue

                corr = np.corrcoef(s1[-min_len:], s2[-min_len:])[0, 1]
                if np.isnan(corr):
                    corr = 0.0

                half_len = min_len // 2
                if half_len >= self.min_series_length:
                    corr_early = np.corrcoef(s1[-min_len:-half_len], s2[-min_len:-half_len])[0, 1]
                    if np.isnan(corr_early):
                        corr_early = 0.0
                    corr_late = np.corrcoef(s1[-half_len:], s2[-half_len:])[0, 1]
                    if np.isnan(corr_late):
                        corr_late = 0.0
                    drift = abs(corr_late - corr_early)
                else:
                    drift = 0.0

                if abs(corr) > 0.6 and drift > 0.15:
                    magnitude = min(1.0, drift * abs(corr) * 2.0)
                    direction = 1.0 if corr_late > corr_early else -1.0

                    combined_tension = max(tension_map.get(d1_name, 0.0),
                                           tension_map.get(d2_name, 0.0))

                    cross_forces.append(LatentForce(
                        name=f"Cross-Domain Link {d1_name}-{d2_name}",
                        origin=ForceOrigin.CONTAGION,
                        primary_domain=ForceDomain.SYSTEMIC,
                        magnitude=magnitude,
                        direction=direction,
                        acceleration=drift,
                        phase_velocity=drift * 10.0,
                        affected_domains=[self._map_domain(d1_name), self._map_domain(d2_name)],
                        evidence=[f"Correlation drift: {corr_early:.3f} → {corr_late:.3f}"],
                        lead_signals=[f"correlation_shift_{d1_name}_{d2_name}"],
                        tension_index=combined_tension,
                        rupture_probability=min(1.0, magnitude * combined_tension * 2.0),
                        estimated_time_to_manifest_hours=max(1.0, 100.0 / max(magnitude, 0.01)),
                    ))

        return cross_forces

    def _estimate_rupture_probability(self, magnitude: float, tension: float,
                                       acceleration: float) -> float:
        base = magnitude * 0.4 + tension * 0.4
        accel_bonus = min(abs(acceleration) * 0.2, 0.2)
        return float(np.clip(base + accel_bonus, 0.0, 1.0))

    def _estimate_manifest_time(self, magnitude: float, acceleration: float,
                                 tension: float) -> float:
        effective_speed = abs(acceleration) + magnitude * 0.5
        if effective_speed < 0.01:
            return 720.0
        if tension > 0.8:
            return 6.0
        if tension > 0.6:
            return 24.0
        base_hours = 100.0 / max(effective_speed, 0.01)
        return min(max(base_hours, 1.0), 720.0)

    def detect_pressure_gradients(self, domain_series: Dict[str, np.ndarray]) -> List[PressureGradient]:
        gradients: List[PressureGradient] = []
        domains = sorted(domain_series.keys())
        velocities = _compute_phase_velocity(domain_series, self.min_series_length)

        for i, d1 in enumerate(domains):
            for d2 in domains[i+1:]:
                s1 = np.asarray(domain_series[d1], dtype=float).ravel()
                s2 = np.asarray(domain_series[d2], dtype=float).ravel()
                min_len = min(len(s1), len(s2))
                if min_len < self.min_series_length:
                    continue

                v1 = velocities.get(d1, 0.0)
                v2 = velocities.get(d2, 0.0)
                gradient = abs(v1 - v2)

                if gradient > self.gradient_threshold:
                    direction = f"{d1}→{d2}" if v1 > v2 else f"{d2}→{d1}"
                    intensifying = abs(v1 - v2) > gradient

                    gradients.append(PressureGradient(
                        source_domain=d1 if v1 > v2 else d2,
                        target_domain=d2 if v1 > v2 else d1,
                        gradient_magnitude=float(np.clip(gradient, 0.0, 1.0)),
                        direction=direction,
                        is_intensifying=intensifying,
                    ))

        gradients.sort(key=lambda g: g.gradient_magnitude, reverse=True)
        return gradients[:20]

    def detect_tension_zones(self, domain_series: Dict[str, np.ndarray],
                              tension_map: Dict[str, float]) -> List[TensionZone]:
        zones: List[TensionZone] = []

        for domain, series in domain_series.items():
            s = np.asarray(series, dtype=float).ravel()
            tension = tension_map.get(domain, 0.0)

            if len(s) < self.min_series_length:
                continue

            d1 = np.diff(s)
            stress_rate = _safe_mean(d1[-5:]) if len(d1) >= 5 else 0.0

            history = self._tension_history.get(domain, deque())
            if len(history) >= 3:
                recent_tension = list(history)[-3:]
                trend = recent_tension[-1] - recent_tension[0]
                if trend > 0:
                    stress_rate = max(stress_rate, tension * 0.1)

            capacity = max(0.0, 1.0 - tension)

            if tension > 0.1 and stress_rate > 0:
                time_to_yield = capacity / max(stress_rate, 1e-10)
                time_to_yield = min(time_to_yield, 720.0)
            else:
                time_to_yield = 720.0

            if tension > 0.3:
                zones.append(TensionZone(
                    domain=domain,
                    tension_level=float(np.clip(tension, 0.0, 1.0)),
                    capacity_remaining=float(np.clip(capacity, 0.0, 1.0)),
                    stress_rate=float(stress_rate),
                    time_to_yield_hours=float(time_to_yield),
                    is_critical=tension > 0.7,
                ))

        zones.sort(key=lambda z: z.tension_level, reverse=True)
        return zones

    def compute_fate_pressure(self, forces: List[LatentForce],
                               tension_zones: List[TensionZone]) -> Tuple[float, FatePressureLevel]:
        if not forces and not tension_zones:
            return 0.0, FatePressureLevel.STILL

        force_pressure = sum(f.pressure_contribution for f in forces if f.is_active)
        max_force = max((f.magnitude for f in forces), default=0.0)
        n_active = sum(1 for f in forces if f.is_active)
        n_critical = sum(1 for z in tension_zones if z.is_critical)
        avg_tension = _safe_mean(np.array([z.tension_level for z in tension_zones])) if tension_zones else 0.0

        score = (
            0.25 * min(force_pressure / max(n_active, 1), 1.0) +
            0.25 * max_force +
            0.20 * min(n_active / 5.0, 1.0) +
            0.15 * avg_tension +
            0.10 * min(n_critical / 3.0, 1.0) +
            0.05 * min(sum(1 for f in forces if f.rupture_probability > 0.5) / max(len(forces), 1), 1.0)
        )
        score = float(np.clip(score, 0.0, 1.0))

        if score < 0.15:
            level = FatePressureLevel.STILL
        elif score < 0.35:
            level = FatePressureLevel.WHISPER
        elif score < 0.55:
            level = FatePressureLevel.RUMBLE
        elif score < 0.75:
            level = FatePressureLevel.SURGE
        else:
            level = FatePressureLevel.RUPTURE

        return score, level

    def _generate_pronouncement(self, forces: List[LatentForce],
                                 fate_level: FatePressureLevel,
                                 score: float) -> str:
        if fate_level == FatePressureLevel.STILL:
            return "The waters are still. No latent forces strain against the fabric of fate."

        active_forces = [f for f in forces if f.is_active]
        if not active_forces:
            return f"Fate pressure reads {fate_level.value} ({score:.2f}) — forces are present but dormant."

        dominant = active_forces[0]
        n_forces = len(active_forces)
        high_rupture = [f for f in active_forces if f.rupture_probability > 0.5]

        parts = [f"Fate pressure: {fate_level.value.upper()} ({score:.2f})."]
        parts.append(f"{n_forces} latent force{'s' if n_forces > 1 else ''} detected.")

        if dominant:
            parts.append(f"Dominant: \"{dominant.name}\" (magnitude {dominant.magnitude:.2f}, "
                         f"origin: {dominant.origin.value}).")
            if dominant.estimated_time_to_manifest_hours < 24:
                parts.append(f"This force may manifest within {dominant.estimated_time_to_manifest_hours:.0f} hours.")
            elif dominant.estimated_time_to_manifest_hours < 168:
                parts.append(f"Estimated manifestation: {dominant.estimated_time_to_manifest_hours/24:.1f} days.")

        if high_rupture:
            parts.append(f"{len(high_rupture)} force{'s' if len(high_rupture) > 1 else ''} near rupture threshold.")

        if fate_level == FatePressureLevel.WHISPER:
            parts.append("The future whispers — barely audible through the noise of the present.")
        elif fate_level == FatePressureLevel.RUMBLE:
            parts.append("A rumble beneath the surface — forces align that the many cannot yet perceive.")
        elif fate_level == FatePressureLevel.SURGE:
            parts.append("The pressure surges — the shape of things to come grows distinct in the dark.")
        elif fate_level == FatePressureLevel.RUPTURE:
            parts.append("RUPTURE IMMINENT. Latent forces have reached critical mass. The future is now.")

        origins = set(f.origin for f in active_forces)
        if len(origins) >= 3:
            parts.append("Multi-origin convergence — forces from different sources compound each other.")

        return " ".join(parts)

    def whisper(self, domain_series: Dict[str, np.ndarray],
                domain_values: Optional[Dict[str, float]] = None) -> ForceWhisperReport:
        forces = self.detect_forces(domain_series, domain_values)
        pressure_gradients = self.detect_pressure_gradients(domain_series)
        tension_map = {
            domain: float(np.clip(
                self._compute_tension({domain: domain_series[domain]}).get(domain, 0.0), 0.0, 1.0
            ))
            for domain in domain_series
        }
        tension_zones = self.detect_tension_zones(domain_series, tension_map)

        score, level = self.compute_fate_pressure(forces, tension_zones)
        phase_velocities = _compute_phase_velocity(domain_series, self.min_series_length)

        dominant = forces[0].name if forces else None
        pronouncement = self._generate_pronouncement(forces, level, score)

        logger.info(
            "Force Whisper: %d forces, fate=%s (%.3f), dominant=%s",
            len(forces), level.value, score, dominant,
        )

        return ForceWhisperReport(
            forces=forces,
            pressure_gradients=pressure_gradients,
            tension_zones=tension_zones,
            fate_pressure_level=level,
            fate_pressure_score=score,
            dominant_force=dominant,
            force_count=len(forces),
            active_force_count=sum(1 for f in forces if f.is_active),
            system_tension=float(np.mean([z.tension_level for z in tension_zones])) if tension_zones else 0.0,
            phase_velocity_field=phase_velocities,
            oracular_pronouncement=pronouncement,
        )

    def history_summary(self) -> Dict:
        if not self._force_history:
            return {"observation_count": 0, "avg_forces": 0.0, "max_forces_seen": 0}
        counts = [h["n_forces"] for h in self._force_history]
        actives = [h["active"] for h in self._force_history]
        return {
            "observation_count": len(self._force_history),
            "avg_forces": round(float(np.mean(counts)), 2),
            "max_forces_seen": int(max(counts)),
            "avg_active": round(float(np.mean(actives)), 2),
            "max_magnitude_seen": round(max(h["max_magnitude"] for h in self._force_history), 4),
        }