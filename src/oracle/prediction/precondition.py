"""Precondition Detector — Identifies necessary preconditions for predicted events.

Detects whether the necessary (but not sufficient) conditions for an event
are in place. Many predictions fail because preconditions are missing — this
module checks systematically. Implements:
1. Precondition trees: AND/OR trees of necessary conditions
2. Completion scoring: how many preconditions are met
3. Blocking factor detection: which missing condition is the bottleneck
4. Temporal sequencing: which preconditions must occur in order
5. Minimum time estimation: earliest possible fulfillment date
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class LogicType(Enum):
    AND = "and"
    OR = "or"


class ConditionStatus(Enum):
    MET = "met"
    PARTIAL = "partial"
    MISSING = "missing"
    IMPOSSIBLE = "impossible"


@dataclass
class Precondition:
    name: str
    description: str
    status: ConditionStatus
    probability: float
    estimated_time_days: Optional[int]
    blocking: bool
    children: List['Precondition'] = field(default_factory=list)
    logic: LogicType = LogicType.AND


@dataclass
class PreconditionTree:
    event: str
    description: str
    root: Precondition
    completion_score: float
    blocking_factors: List[str]
    earliest_date_days: Optional[int]
    probability_achievement: float


@dataclass
class PreconditionReport:
    trees: List[PreconditionTree]
    overall_readiness: float
    most_likely_events: List[str]
    blocked_events: List[str]
    near_completion: List[str]
    recommendations: List[str]


DEFAULT_TREES: Dict[str, Dict] = {
    "recession": {
        "description": "Economic recession preconditions",
        "children": [
            {"name": "credit_expansion", "description": "Sustained credit growth above GDP", "status": "partial", "probability": 0.6, "time": 90, "blocking": False},
            {"name": "yield_curve_inversion", "description": "Yield curve inversion sustained", "status": "missing", "probability": 0.3, "time": 180, "blocking": True},
            {"name": "leverage_peak", "description": "Peak leverage ratios in financial system", "status": "partial", "probability": 0.5, "time": 120, "blocking": False},
            {"name": "confidence_decline", "description": "Declining business/consumer confidence", "status": "partial", "probability": 0.4, "time": 60, "blocking": False},
        ],
        "logic": "or",
    },
    "financial_crisis": {
        "description": "Financial system crisis preconditions",
        "children": [
            {"name": "excessive_leverage", "description": "System-wide leverage above 3x GDP", "status": "met", "probability": 0.8, "time": 0, "blocking": False},
            {"name": "liquidity_mismatch", "description": "Widespread maturity transformation", "status": "partial", "probability": 0.6, "time": 60, "blocking": False},
            {"name": "regulatory_gaps", "description": "Shadow banking regulatory gaps", "status": "met", "probability": 0.75, "time": 0, "blocking": False},
            {"name": "asset_bubble", "description": "Overvalued asset class", "status": "partial", "probability": 0.5, "time": 90, "blocking": False},
            {"name": "trigger_event", "description": "Catalyzing shock event", "status": "missing", "probability": 0.2, "time": 365, "blocking": True},
        ],
        "logic": "and",
    },
    "geopolitical_crisis": {
        "description": "Major geopolitical crisis preconditions",
        "children": [
            {"name": "alliance_commitments", "description": "Active alliance obligations", "status": "met", "probability": 0.7, "time": 0, "blocking": False},
            {"name": "territorial_dispute", "description": "Active territorial disputes", "status": "met", "probability": 0.65, "time": 0, "blocking": False},
            {"name": "military_buildup", "description": "Military mobilization/escalation", "status": "partial", "probability": 0.4, "time": 60, "blocking": False},
            {"name": "diplomatic_failure", "description": "Diplomatic channels broken", "status": "missing", "probability": 0.2, "time": 90, "blocking": True},
        ],
        "logic": "and",
    },
    "tech_breakthrough": {
        "description": "Major technology breakthrough preconditions",
        "children": [
            {"name": "research_investment", "description": "Sustained R&D investment", "status": "met", "probability": 0.8, "time": 0, "blocking": False},
            {"name": "talent_availability", "description": "Critical mass of domain expertise", "status": "met", "probability": 0.75, "time": 0, "blocking": False},
            {"name": "computational_resources", "description": "Adequate computing infrastructure", "status": "partial", "probability": 0.6, "time": 30, "blocking": False},
            {"name": "scientific_foundation", "description": "Key scientific insight discovered", "status": "missing", "probability": 0.25, "time": 180, "blocking": True},
        ],
        "logic": "and",
    },
    "climate_tipping_point": {
        "description": "Climate tipping point preconditions",
        "children": [
            {"name": "temperature_threshold", "description": "Global temperature above critical level", "status": "partial", "probability": 0.5, "time": 365, "blocking": False},
            {"name": "permafrost_feedback", "description": "Permafrost methane release accelerating", "status": "partial", "probability": 0.4, "time": 730, "blocking": False},
            {"name": "ice_sheet_collapse", "description": "Ice sheet destabilization started", "status": "missing", "probability": 0.2, "time": 1825, "blocking": True},
            {"name": "ocean_circulation_change", "description": "AMOC or other circulation weakening", "status": "partial", "probability": 0.35, "time": 1095, "blocking": False},
        ],
        "logic": "or",
    },
}


class PreconditionDetector:
    def __init__(self, trees: Optional[Dict[str, Dict]] = None):
        self.tree_defs = trees or dict(DEFAULT_TREES)

    def build_tree(self, event_name: str) -> PreconditionTree:
        tree_def = self.tree_defs.get(event_name, {})
        if not tree_def:
            root = Precondition(event_name, "Unknown event", ConditionStatus.MISSING, 0.0, None, True)
            return PreconditionTree(event_name, "", root, 0.0, [event_name], None, 0.0)
        children = []
        for child_def in tree_def.get("children", []):
            status = ConditionStatus(child_def.get("status", "missing"))
            child = Precondition(
                name=child_def["name"],
                description=child_def["description"],
                status=status,
                probability=child_def.get("probability", 0.5),
                estimated_time_days=child_def.get("time"),
                blocking=child_def.get("blocking", False),
            )
            children.append(child)
        logic = LogicType(tree_def.get("logic", "and"))
        root = Precondition(
            name=event_name,
            description=tree_def.get("description", ""),
            status=ConditionStatus.PARTIAL,
            probability=0.5,
            estimated_time_days=None,
            blocking=False,
            children=children,
            logic=logic,
        )
        completion = self._compute_completion(root)
        blocking = self._find_blocking(root)
        earliest = self._compute_earliest(root)
        prob = self._compute_probability(root)
        root.status = self._status_from_completion(completion)
        root.probability = prob
        return PreconditionTree(
            event=event_name,
            description=tree_def.get("description", ""),
            root=root,
            completion_score=round(completion, 4),
            blocking_factors=blocking,
            earliest_date_days=earliest,
            probability_achievement=round(prob, 4),
        )

    def detect(self) -> PreconditionReport:
        trees = []
        for event_name in self.tree_defs:
            trees.append(self.build_tree(event_name))
        readiness = np.mean([t.completion_score for t in trees]) if trees else 0.0
        likely = sorted(trees, key=lambda t: t.probability_achievement, reverse=True)[:3]
        blocked = [t.event for t in trees if t.blocking_factors]
        near = [t.event for t in trees if t.completion_score > 0.6]
        recs = self._recommend(trees, near, blocked)
        return PreconditionReport(
            trees=trees,
            overall_readiness=round(readiness, 4),
            most_likely_events=[e.event for e in likely],
            blocked_events=blocked,
            near_completion=near,
            recommendations=recs,
        )

    def check_event(self, event: str) -> Optional[PreconditionTree]:
        if event in self.tree_defs:
            return self.build_tree(event)
        return None

    def update_condition(self, event: str, condition_name: str,
                          status: ConditionStatus, probability: float) -> None:
        if event in self.tree_defs:
            for child in self.tree_defs[event].get("children", []):
                if child["name"] == condition_name:
                    child["status"] = status.value
                    child["probability"] = probability

    def _compute_completion(self, node: Precondition) -> float:
        if not node.children:
            status_scores = {
                ConditionStatus.MET: 1.0,
                ConditionStatus.PARTIAL: 0.5,
                ConditionStatus.MISSING: 0.0,
                ConditionStatus.IMPOSSIBLE: 0.0,
            }
            return status_scores.get(node.status, 0.0)
        child_scores = [self._compute_completion(c) for c in node.children]
        if node.logic == LogicType.AND:
            return min(child_scores) if child_scores else 0.0
        else:
            return max(child_scores) if child_scores else 0.0

    def _find_blocking(self, node: Precondition) -> List[str]:
        blocking = []
        for child in node.children:
            if child.blocking and child.status in (ConditionStatus.MISSING, ConditionStatus.IMPOSSIBLE):
                blocking.append(child.name)
            sub_blocking = self._find_blocking(child)
            blocking.extend(sub_blocking)
        return blocking

    def _compute_earliest(self, node: Precondition) -> Optional[int]:
        if not node.children:
            if node.status == ConditionStatus.MET:
                return 0
            return node.estimated_time_days
        times = [self._compute_earliest(c) for c in node.children if self._compute_earliest(c) is not None]
        if not times:
            return None
        if node.logic == LogicType.AND:
            return max(times)
        else:
            return min(times)

    def _compute_probability(self, node: Precondition) -> float:
        if not node.children:
            return node.probability
        if node.logic == LogicType.AND:
            return float(np.prod([self._compute_probability(c) for c in node.children]))
        else:
            return 1.0 - float(np.prod([1 - self._compute_probability(c) for c in node.children]))

    def _status_from_completion(self, completion: float) -> ConditionStatus:
        if completion >= 0.9:
            return ConditionStatus.MET
        elif completion >= 0.4:
            return ConditionStatus.PARTIAL
        elif completion > 0:
            return ConditionStatus.MISSING
        return ConditionStatus.IMPOSSIBLE

    def _recommend(self, trees: List[PreconditionTree],
                     near: List[str], blocked: List[str]) -> List[str]:
        recs = []
        if near:
            recs.append(f"NEAR COMPLETION: {', '.join(near)} events have most preconditions met — highest risk")
        if blocked:
            recs.append(f"BLOCKED: {', '.join(blocked)} events have blocking preconditions — watch for blocker resolution")
        for tree in trees[:3]:
            if tree.blocking_factors:
                recs.append(f"  {tree.event}: resolve {tree.blocking_factors[0]} to unblock (P={tree.probability_achievement:.0%})")
        recs.append("Monitor precondition status changes as early warning indicators")
        return recs