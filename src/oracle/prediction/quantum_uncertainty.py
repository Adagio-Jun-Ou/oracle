"""Quantum Uncertainty Mapper — Heisenberg-inspired uncertainty modeling for predictions.

Applies quantum mechanical principles to prediction uncertainty:
1. Uncertainty principle: precision on one variable limits precision on correlated variable
2. Superposition: predictions exist in probability distributions, not point estimates
3. Entanglement: correlated variables affect each other's uncertainty
4. Decoherence: predictions become more certain over time as information accumulates
5. Quantum tunneling: low-probability outcomes can occur through "barrier penetration"
6. Measurement collapse: observing a prediction changes the system state

This is a metaphorical application — we don't literally use quantum mechanics, but
the conceptual framework provides powerful tools for understanding prediction limits.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class UncertaintyType(Enum):
    ALEATORY = "aleatory"
    EPISTEMIC = "epistemic"
    ONTOLOGICAL = "ontological"


@dataclass
class UncertaintyBundle:
    variable: str
    mean: float
    std: float
    uncertainty_type: UncertaintyType
    complementary_variable: Optional[str]
    complementary_uncertainty: Optional[float]
    measurement_limit: float
    description: str


@dataclass
class EntanglementPair:
    var_a: str
    var_b: str
    correlation: float
    mutual_uncertainty: float
    joint_constraint: str


@dataclass
class TunnelingEvent:
    barrier: str
    barrier_height: float
    tunneling_probability: float
    energy_estimate: float
    path_description: str


@dataclass
class DecoherenceState:
    variable: str
    initial_uncertainty: float
    current_uncertainty: float
    information_rate: float
    time_to_certainty: Optional[float]
    collapse_triggers: List[str]


@dataclass
class QuantumUncertaintyResult:
    uncertainties: List[UncertaintyBundle]
    entanglements: List[EntanglementPair]
    tunneling_events: List[TunnelingEvent]
    decoherence_states: List[DecoherenceState]
    superposition_states: Dict[str, List[float]]
    total_uncertainty: float
    irreducible_uncertainty: float
    knowable_fraction: float
    recommendations: List[str]


DEFAULT_CONSTRAINTS: Dict[Tuple[str, str], float] = {
    ("inflation", "growth"): 0.3,
    ("unemployment", "wage_growth"): 0.4,
    ("interest_rates", "asset_prices"): 0.5,
    ("risk", "return"): 0.7,
    ("stability", "innovation"): 0.35,
    ("equity", "efficiency"): 0.4,
    ("privacy", "security"): 0.5,
    ("freedom", "safety"): 0.45,
    ("diversity", "coherence"): 0.3,
    ("speed", "accuracy"): 0.6,
}


class QuantumUncertaintyMapper:
    def __init__(self, h_bar: float = 0.1, decoherence_rate: float = 0.02):
        self.h_bar = h_bar
        self.decoherence_rate = decoherence_rate
        self.constraints = dict(DEFAULT_CONSTRAINTS)
        self.observed: Dict[str, float] = {}
        self.decoherence_history: Dict[str, List[float]] = {}

    def map_uncertainty(self, predictions: Dict[str, Tuple[float, float]],
                         correlations: Optional[Dict[Tuple[str, str], float]] = None) -> QuantumUncertaintyResult:
        uncertainties = []
        for var, (mean, std) in predictions.items():
            comp_var, comp_unc, meas_limit = self._find_complementary(var, std, correlations)
            utype = self._classify_uncertainty(var, std)
            uncertainties.append(UncertaintyBundle(
                variable=var, mean=mean, std=std,
                uncertainty_type=utype,
                complementary_variable=comp_var,
                complementary_uncertainty=comp_unc,
                measurement_limit=meas_limit,
                description=f"{var}: μ={mean:.2f}, σ={std:.2f} — {utype.value} uncertainty"
            ))
        entanglements = self._find_entanglements(predictions, correlations)
        tunneling = self._find_tunneling_events(predictions)
        decoherence = self._compute_decoherence(predictions)
        superposition = self._compute_superposition(predictions)
        total_unc = np.mean([u.std for u in uncertainties]) if uncertainties else 0.5
        irreducible = self._compute_irreducible(uncertainties, entanglements)
        knowable = 1.0 - (irreducible / max(total_unc, 0.01))
        recs = self._generate_recommendations(uncertainties, entanglements, tunneling, irreducible)
        return QuantumUncertaintyResult(
            uncertainties=uncertainties,
            entanglements=entanglements,
            tunneling_events=tunneling,
            decoherence_states=decoherence,
            superposition_states=superposition,
            total_uncertainty=round(total_unc, 4),
            irreducible_uncertainty=round(irreducible, 4),
            knowable_fraction=round(max(0, min(1, knowable)), 4),
            recommendations=recs,
        )

    def observe(self, variable: str, value: float, precision: float) -> Dict:
        self.observed[variable] = value
        complementary = None
        for (a, b), limit in self.constraints.items():
            if a == variable:
                complementary = b
                complementary_precision = max(0.01, self.h_bar / precision) if precision > 0 else 1.0
                break
            elif b == variable:
                complementary = a
                complementary_precision = max(0.01, self.h_bar / precision) if precision > 0 else 1.0
                break
        effect = {
            "observed": variable,
            "value": value,
            "precision": precision,
            "collapse": True,
            "complementary_affected": complementary,
        }
        if complementary:
            effect["complementary_uncertainty_increase"] = round(complementary_precision, 4)
        return effect

    def compute_uncertainty_principle(self, var_a: str, var_b: str,
                                        delta_a: float, delta_b: float) -> Dict:
        key = (var_a, var_b)
        rev_key = (var_b, var_a)
        limit = self.constraints.get(key, self.constraints.get(rev_key, 0.5))
        product = delta_a * delta_b
        saturated = abs(product - limit) < limit * 0.2
        violated = product < limit
        min_delta_b = limit / max(delta_a, 0.01)
        min_delta_a = limit / max(delta_b, 0.01)
        return {
            "var_a": var_a,
            "var_b": var_b,
            "delta_a": round(delta_a, 4),
            "delta_b": round(delta_b, 4),
            "product": round(product, 4),
            "heisenberg_limit": round(limit, 4),
            "saturated": saturated,
            "violated": violated,
            "minimum_delta_b_given_a": round(min_delta_b, 4),
            "minimum_delta_a_given_b": round(min_delta_a, 4),
        }

    def _find_complementary(self, var: str, std: float,
                              correlations: Optional[Dict] = None) -> Tuple[Optional[str], Optional[float], float]:
        for (a, b), limit in self.constraints.items():
            if a == var:
                min_std_b = max(0.01, self.h_bar / max(std, 0.01))
                return b, min_std_b, limit
            elif b == var:
                min_std_a = max(0.01, self.h_bar / max(std, 0.01))
                return a, min_std_a, limit
        if correlations:
            for (a, b), corr in correlations.items():
                if a == var and abs(corr) > 0.5:
                    return b, std * abs(corr) * 0.3, abs(corr) * 0.3
                elif b == var and abs(corr) > 0.5:
                    return a, std * abs(corr) * 0.3, abs(corr) * 0.3
        return None, None, 0.0

    def _classify_uncertainty(self, var: str, std: float) -> UncertaintyType:
        if std < 0.05:
            return UncertaintyType.ALEATORY
        elif std < 0.2:
            return UncertaintyType.EPISTEMIC
        return UncertaintyType.ONTOLOGICAL

    def _find_entanglements(self, predictions: Dict[str, Tuple[float, float]],
                              correlations: Optional[Dict] = None) -> List[EntanglementPair]:
        entanglements = []
        seen = set()
        all_pairs = list(self.constraints.keys())
        if correlations:
            all_pairs.extend(correlations.keys())
        for pair in all_pairs:
            a, b = pair[0], pair[1]
            key = tuple(sorted([a, b]))
            if key in seen or a not in predictions or b not in predictions:
                continue
            seen.add(key)
            corr = 0.5
            if correlations and pair in correlations:
                corr = correlations[pair]
            elif (b, a) in self.constraints:
                corr = self.constraints.get((b, a), 0.5)
            else:
                corr = self.constraints.get(pair, 0.5)
            std_a = predictions[a][1]
            std_b = predictions[b][1]
            mutual_unc = abs(corr) * math.sqrt(std_a ** 2 + std_b ** 2) / 2
            constraint_type = "Heisenberg-type" if abs(corr) > 0.3 else "loose"
            entanglements.append(EntanglementPair(
                var_a=a, var_b=b, correlation=round(corr, 4),
                mutual_uncertainty=round(mutual_unc, 4),
                joint_constraint=constraint_type,
            ))
        return entanglements

    def _find_tunneling_events(self, predictions: Dict[str, Tuple[float, float]]) -> List[TunnelingEvent]:
        events = []
        for var, (mean, std) in predictions.items():
            if std > 0.3:
                barrier_height = abs(mean) + 2 * std
                tunnel_prob = math.exp(-2 * barrier_height / max(std, 0.01))
                if tunnel_prob > 0.001:
                    events.append(TunnelingEvent(
                        barrier=f"{var}_extreme_outcome",
                        barrier_height=round(barrier_height, 4),
                        tunneling_probability=round(tunnel_prob, 6),
                        energy_estimate=round(std, 4),
                        path_description=f"Low-probability extreme outcome for {var} through statistical tunneling (P={tunnel_prob:.4f})",
                    ))
        events.sort(key=lambda e: e.tunneling_probability, reverse=True)
        return events[:5]

    def _compute_decoherence(self, predictions: Dict[str, Tuple[float, float]]) -> List[DecoherenceState]:
        states = []
        for var, (mean, std) in predictions.items():
            initial_std = std
            current_std = std * math.exp(-self.decoherence_rate * 10)
            info_rate = self.decoherence_rate * std
            if current_std > 0.01:
                ttc = -math.log(0.01 / max(std, 0.001)) / max(self.decoherence_rate, 0.001)
            else:
                ttc = 0.0
            triggers = [f"data_arrival_{var}", "expert_update", "market_observation"]
            states.append(DecoherenceState(
                variable=var,
                initial_uncertainty=round(initial_std, 4),
                current_uncertainty=round(current_std, 4),
                information_rate=round(info_rate, 4),
                time_to_certainty=round(ttc, 2) if ttc < 100 else None,
                collapse_triggers=triggers,
            ))
        return states

    def _compute_superposition(self, predictions: Dict[str, Tuple[float, float]]) -> Dict[str, List[float]]:
        states = {}
        for var, (mean, std) in predictions.items():
            n_outcomes = max(3, min(7, int(1 + std * 10)))
            outcomes = list(np.linspace(mean - 2 * std, mean + 2 * std, n_outcomes))
            states[var] = [round(float(o), 4) for o in outcomes]
        return states

    def _compute_irreducible(self, uncertainties: List[UncertaintyBundle],
                               entanglements: List[EntanglementPair]) -> float:
        individual = np.mean([u.std for u in uncertainties]) if uncertainties else 0.5
        entanglement_penalty = sum(e.mutual_uncertainty for e in entanglements) / max(len(entanglements), 1)
        irreducible = individual * (1 - 0.5 * min(1, entanglement_penalty))
        return irreducible

    def _generate_recommendations(self, uncertainties: List[UncertaintyBundle],
                                    entanglements: List[EntanglementPair],
                                    tunneling: List[TunnelingEvent],
                                    irreducible: float) -> List[str]:
        recs = []
        ontological = [u for u in uncertainties if u.uncertainty_type == UncertaintyType.ONTOLOGICAL]
        if ontological:
            recs.append(f"ONTOLOGICAL UNCERTAINTY: {len(ontological)} variables have fundamental unpredictability — no amount of data resolves this")
        epistemic = [u for u in uncertainties if u.uncertainty_type == UncertaintyType.EPISTEMIC]
        if epistemic:
            recs.append(f"EPISTEMIC UNCERTAINTY: {len(epistemic)} variables could be reduced with more information — prioritize data collection")
        if entanglements:
            heisenberg = [e for e in entanglements if "Heisenberg" in e.joint_constraint]
            if heisenberg:
                recs.append(f"HEISENBERG CONSTRAINTS: {len(heisenberg)} variable pairs have trade-off precision limits — cannot know both precisely")
        if tunneling:
            recs.append(f"QUANTUM TUNNELING: {len(tunneling)} low-probability extreme outcomes possible — prepare for statistical black swans")
        recs.append(f"IRREDUCIBLE UNCERTAINTY: {irreducible:.2f} — minimum achievable uncertainty after all information is gathered")
        recs.append("Design decisions robust to the uncertainty level rather than seeking false precision")
        return recs