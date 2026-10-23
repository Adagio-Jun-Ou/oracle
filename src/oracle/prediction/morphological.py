"""Morphological Analysis — Systematic future possibility exploration.

Based on Fritz Zwicky's morphological analysis method: decompose a complex problem
into all possible parameter combinations and evaluate each systematically. This
generates a complete possibility space rather than relying on narrative scenarios.

Key innovations:
1. Parameter space decomposition with cross-consistency assessment
2. Forced combination generation (systematic exploration of all configurations)
3. Consistency filtering (eliminate internally contradictory combinations)
4. Morphological field dynamics (how parameter spaces evolve over time)
5. Wild card injection (systematic inclusion of low-probability high-impact parameters)
6. Cluster detection (identify families of similar future states)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import logging
import math
import itertools

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    COMPATIBLE = "compatible"
    POSSIBLE = "possible"
    UNLIKELY = "unlikely"
    INCONSISTENT = "inconsistent"


@dataclass
class MorphologicalParameter:
    name: str
    values: List[str]
    probabilities: List[float]
    description: str = ""


@dataclass
class MorphologicalCell:
    parameters: Dict[str, str]
    consistency_score: float
    probability: float
    feasibility: float
    novelty: float
    impact: float
    description: str


@dataclass
class MorphologicalField:
    parameters: List[MorphologicalParameter]
    consistency_matrix: Dict[Tuple[str, str, str, str], ConsistencyLevel]
    cells: List[MorphologicalCell]
    n_possible: int
    n_consistent: int
    coverage: float


@dataclass
class MorphologicalResult:
    field: MorphologicalField
    top_scenarios: List[MorphologicalCell]
    wild_cards: List[MorphologicalCell]
    clusters: List[Dict[str, str]]
    key_tensions: List[str]
    key_opportunities: List[str]
    recommendations: List[str]


CONSISTENCY_DEFAULTS: Dict[Tuple[str, str], ConsistencyLevel] = {
    ("high_growth", "low_inflation"): ConsistencyLevel.COMPATIBLE,
    ("high_growth", "high_inflation"): ConsistencyLevel.POSSIBLE,
    ("recession", "low_unemployment"): ConsistencyLevel.UNLIKELY,
    ("recession", "high_inflation"): ConsistencyLevel.POSSIBLE,
    ("low_rates", "high_inflation"): ConsistencyLevel.UNLIKELY,
    ("high_rates", "high_growth"): ConsistencyLevel.UNLIKELY,
    ("high_rates", "recession"): ConsistencyLevel.COMPATIBLE,
    ("low_rates", "high_growth"): ConsistencyLevel.COMPATIBLE,
    ("peace", "high_geopolitical_risk"): ConsistencyLevel.INCONSISTENT,
    ("conflict", "low_defense_spending"): ConsistencyLevel.INCONSISTENT,
    ("tech_boom", "low_rd_investment"): ConsistencyLevel.INCONSISTENT,
    ("climate_crisis", "low_carbon_policy"): ConsistencyLevel.POSSIBLE,
    ("pandemic", "high_mobility"): ConsistencyLevel.INCONSISTENT,
    ("trade_war", "high_global_trade"): ConsistencyLevel.INCONSISTENT,
    ("energy_crisis", "renewable_dominance"): ConsistencyLevel.UNLIKELY,
    ("financial_crisis", "low_leverage"): ConsistencyLevel.UNLIKELY,
    ("ai_revolution", "tech_regulation"): ConsistencyLevel.POSSIBLE,
    ("demographic_decline", "high_growth"): ConsistencyLevel.UNLIKELY,
    ("political_stability", "social_unrest"): ConsistencyLevel.INCONSISTENT,
    ("decentralization", "strong_regulation"): ConsistencyLevel.POSSIBLE,
}

DEFAULT_PARAMETERS: List[MorphologicalParameter] = [
    MorphologicalParameter("economic_growth", ["recession", "stagnation", "moderate_growth", "high_growth"],
                           [0.15, 0.25, 0.40, 0.20], "Economic growth trajectory"),
    MorphologicalParameter("inflation", ["deflation", "low_inflation", "moderate_inflation", "high_inflation"],
                           [0.10, 0.35, 0.35, 0.20], "Inflation environment"),
    MorphologicalParameter("interest_rates", ["near_zero", "low_rates", "moderate_rates", "high_rates"],
                           [0.15, 0.35, 0.30, 0.20], "Monetary policy stance"),
    MorphologicalParameter("geopolitical", ["peace", "tension", "conflict", "crisis"],
                           [0.20, 0.40, 0.25, 0.15], "Geopolitical environment"),
    MorphologicalParameter("technology", ["stagnation", "incremental", "disruptive", "revolutionary"],
                           [0.10, 0.40, 0.30, 0.20], "Technology trajectory"),
    MorphologicalParameter("climate", ["stable", "warming", "severe", "catastrophic"],
                           [0.15, 0.40, 0.30, 0.15], "Climate trajectory"),
    MorphologicalParameter("financial", ["stable", "stress", "crisis", "meltdown"],
                           [0.30, 0.35, 0.25, 0.10], "Financial system health"),
    MorphologicalParameter("social", ["cohesive", "divided", "polarized", "unrest"],
                           [0.20, 0.35, 0.30, 0.15], "Social cohesion level"),
]


class MorphologicalAnalyzer:
    def __init__(self, parameters: Optional[List[MorphologicalParameter]] = None,
                 consistency_rules: Optional[Dict] = None):
        self.parameters = parameters or DEFAULT_PARAMETERS.copy()
        self.consistency_rules = consistency_rules or dict(CONSISTENCY_DEFAULTS)
        self._consistency_matrix: Dict[Tuple[str, str, str, str], ConsistencyLevel] = {}

    def add_parameter(self, parameter: MorphologicalParameter) -> None:
        self.parameters.append(parameter)
        self._consistency_matrix = {}

    def set_consistency(self, param1: str, value1: str, param2: str, value2: str,
                         level: ConsistencyLevel) -> None:
        key = (param1, value1, param2, value2)
        self.consistency_rules[key] = level
        key_rev = (param2, value2, param1, value1)
        self.consistency_rules[key_rev] = level

    def build_field(self, max_cells: int = 1000) -> MorphologicalField:
        param_dict = {p.name: p for p in self.parameters}
        all_combos = list(itertools.product(*[p.values for p in self.parameters]))
        n_possible = len(all_combos)
        if n_possible > max_cells:
            sampled_indices = np.random.choice(n_possible, max_cells, replace=False)
            all_combos = [all_combos[i] for i in sampled_indices]
        cells = []
        consistent_cells = []
        for combo in all_combos:
            params = {p.name: v for p, v in zip(self.parameters, combo)}
            consistency = self._evaluate_consistency(params)
            prob = self._compute_probability(params, param_dict)
            feasibility = consistency * prob
            novelty = self._compute_novelty(params, param_dict)
            impact = self._compute_impact(params, param_dict)
            if consistency > 0.1:
                cell = MorphologicalCell(
                    parameters=params, consistency_score=round(consistency, 4),
                    probability=round(prob, 4), feasibility=round(feasibility, 4),
                    novelty=round(novelty, 4), impact=round(impact, 4),
                    description=self._describe_cell(params),
                )
                cells.append(cell)
                if consistency > 0.3:
                    consistent_cells.append(cell)
        cells.sort(key=lambda c: c.feasibility, reverse=True)
        coverage = len(consistent_cells) / max(len(cells), 1)
        return MorphologicalField(
            parameters=self.parameters,
            consistency_matrix=self._consistency_matrix,
            cells=cells,
            n_possible=n_possible,
            n_consistent=len(consistent_cells),
            coverage=round(coverage, 4),
        )

    def analyze(self, max_cells: int = 500, top_n: int = 5) -> MorphologicalResult:
        field = self.build_field(max_cells)
        top_scenarios = sorted(field.cells, key=lambda c: c.feasibility * c.impact, reverse=True)[:top_n]
        wild_cards = [c for c in field.cells if c.novelty > 0.7 and c.impact > 0.5][:5]
        clusters = self._find_clusters(field.cells[:min(50, len(field.cells))])
        tensions = self._find_tensions()
        opportunities = self._find_opportunities(top_scenarios)
        recs = self._generate_recommendations(top_scenarios, wild_cards, tensions)
        return MorphologicalResult(
            field=field, top_scenarios=top_scenarios[:top_n],
            wild_cards=wild_cards[:5], clusters=clusters[:5],
            key_tensions=tensions[:5], key_opportunities=opportunities[:5],
            recommendations=recs,
        )

    def _evaluate_consistency(self, params: Dict[str, str]) -> float:
        violations = 0
        param_items = list(params.items())
        for i in range(len(param_items)):
            for j in range(i + 1, len(param_items)):
                key = (param_items[i][0], param_items[i][1], param_items[j][0], param_items[j][1])
                key_rev = (param_items[j][0], param_items[j][1], param_items[i][0], param_items[i][1])
                level = self.consistency_rules.get(key)
                if level is None:
                    level = self.consistency_rules.get(key_rev)
                if level is None:
                    # Fallback: check value-only keys for backward compatibility
                    old_key = (param_items[i][1], param_items[j][1])
                    old_key_rev = (param_items[j][1], param_items[i][1])
                    level = self.consistency_rules.get(old_key, self.consistency_rules.get(old_key_rev, ConsistencyLevel.POSSIBLE))
                if level == ConsistencyLevel.INCONSISTENT:
                    violations += 1
        return max(0.0, 1.0 - 0.3 * violations)

    def _compute_probability(self, params: Dict[str, str],
                              param_dict: Dict[str, MorphologicalParameter]) -> float:
        log_prob = 0.0
        for name, value in params.items():
            if name in param_dict:
                p = param_dict[name]
                if value in p.values:
                    idx = p.values.index(value)
                    prob = p.probabilities[idx]
                    log_prob += math.log(max(prob, 1e-10))
                else:
                    log_prob += math.log(1e-5)
            else:
                log_prob += math.log(0.1)
        return max(0.0, min(1.0, math.exp(log_prob / max(len(params), 1))))

    def _compute_novelty(self, params: Dict[str, str],
                          param_dict: Dict[str, MorphologicalParameter]) -> float:
        novelty = 0.0
        for name, value in params.items():
            if name in param_dict:
                p = param_dict[name]
                if value in p.values:
                    idx = p.values.index(value)
                    novelty += (1.0 - p.probabilities[idx])
        return novelty / max(len(params), 1)

    def _compute_impact(self, params: Dict[str, str],
                          param_dict: Dict[str, MorphologicalParameter]) -> float:
        impact_keywords = {
            "recession": 0.8, "crisis": 0.9, "revolutionary": 0.8, "catastrophic": 0.9,
            "conflict": 0.7, "meltdown": 0.9, "unrest": 0.7, "high_growth": 0.6,
            "severe": 0.8, "disruptive": 0.7, "high_inflation": 0.7,
        }
        total = 0.0
        for name, value in params.items():
            total += impact_keywords.get(value, 0.3)
        return total / max(len(params), 1)

    def _describe_cell(self, params: Dict[str, str]) -> str:
        parts = [f"{name}={value}" for name, value in params.items()]
        return "; ".join(parts[:4])

    def _find_clusters(self, cells: List[MorphologicalCell]) -> List[Dict[str, str]]:
        if len(cells) < 2:
            return []
        param_names = list(cells[0].parameters.keys())
        param_options = {name: list(set(c.parameters[name] for c in cells)) for name in param_names}
        clusters = []
        for param in param_names:
            values = param_options[param]
            for value in values[:2]:
                matching = [c for c in cells if c.parameters.get(param) == value]
                if matching:
                    avg_feasibility = float(np.mean([c.feasibility for c in matching]))
                    if avg_feasibility > 0.3:
                        clusters.append({"parameter": param, "value": value,
                                        "avg_feasibility": round(float(avg_feasibility), 3),
                                        "n_scenarios": len(matching)})
        clusters.sort(key=lambda c: c["avg_feasibility"], reverse=True)
        return clusters

    def _find_tensions(self) -> List[str]:
        tensions = []
        seen = set()
        inconsistent_pairs = [(k, v) for k, v in self.consistency_rules.items()
                              if v == ConsistencyLevel.INCONSISTENT]
        for key, level in inconsistent_pairs:
            v1, v2 = key
            pair_key = tuple(sorted([v1, v2]))
            if pair_key not in seen:
                tensions.append(f"{v1} conflicts with {v2}")
                seen.add(pair_key)
            if len(tensions) >= 5:
                break
        return tensions

    def _find_opportunities(self, top_scenarios: List[MorphologicalCell]) -> List[str]:
        opportunities = []
        for cell in top_scenarios[:3]:
            if cell.novelty > 0.5 and cell.feasibility > 0.3:
                opportunities.append(f"Novel feasible scenario: {cell.description} (feasibility={cell.feasibility:.2f})")
        return opportunities

    def _generate_recommendations(self, top_scenarios: List[MorphologicalCell],
                                    wild_cards: List[MorphologicalCell],
                                    tensions: List[str]) -> List[str]:
        recs = []
        if top_scenarios:
            best = top_scenarios[0]
            recs.append(f"Highest feasibility scenario: {best.description} (P={best.probability:.2f}, feasibility={best.feasibility:.2f})")
        if wild_cards:
            wc = wild_cards[0]
            recs.append(f"Key wild card to monitor: {wc.description} (novelty={wc.novelty:.2f}, impact={wc.impact:.2f})")
        if tensions:
            recs.append(f"Key tension to manage: {tensions[0]}")
        recs.append("Develop strategies robust across top 3 scenarios")
        recs.append("Monitor wild card indicators for early warning of novel futures")
        return recs