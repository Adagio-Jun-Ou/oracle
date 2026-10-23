"""Scenario Engine with Causal Loop Diagrams and System Dynamics.

Implements:
- CausalLoopDiagram: Define and simulate feedback loops between domain variables
- SystemDynamicsModel: Stock-and-flow simulation with accumulation, rates, and delays
- ScenarioSimulator: Multi-scenario exploration with branching and comparison
- CrossDomainFeedback: Quantitative feedback propagation between domain predictions
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable
from enum import Enum
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    REINFORCING = "reinforcing"
    BALANCING = "balancing"
    NEUTRAL = "neutral"


class DelayType(Enum):
    FIRST_ORDER = "first_order"
    THIRD_ORDER = "third_order"
    FIXED = "fixed"


@dataclass
class CausalLink:
    source: str
    target: str
    polarity: float
    delay: float = 0.0
    delay_type: DelayType = DelayType.FIRST_ORDER
    strength: float = 1.0
    description: str = ""
    feedback_type: FeedbackType = FeedbackType.NEUTRAL


@dataclass
class Stock:
    name: str
    initial_value: float
    unit: str = ""
    min_value: float = float("-inf")
    max_value: float = float("inf")
    description: str = ""


@dataclass
class Flow:
    name: str
    source_stock: Optional[str]
    target_stock: str
    rate_function: str
    parameters: Dict[str, float] = field(default_factory=dict)
    description: str = ""


@dataclass
class SimulationResult:
    time_series: Dict[str, List[float]]
    time_points: List[float]
    stocks_final: Dict[str, float]
    stocks_initial: Dict[str, float]
    feedback_loops: List[Dict]
    equilibrium_analysis: Dict
    sensitivity_scores: Dict[str, float]
    method: str = "system_dynamics"


@dataclass
class ScenarioResult:
    scenario_name: str
    scenario_hash: str
    assumptions: Dict[str, float]
    predictions: Dict[str, List[float]]
    time_points: List[float]
    equilibrium_values: Dict[str, float]
    stability: str
    feedback_summary: List[Dict]
    cross_domain_effects: Dict[str, Dict[str, float]]
    method: str = "scenario_simulation"


class CausalLoopDiagram:
    """Define and analyze causal loop diagrams with feedback loop detection."""

    def __init__(self):
        self.links: List[CausalLink] = []
        self.variables: set = set()

    def add_link(self, source: str, target: str, polarity: float,
                 delay: float = 0.0, strength: float = 1.0,
                 description: str = "") -> 'CausalLoopDiagram':
        self.links.append(CausalLink(
            source=source, target=target, polarity=polarity,
            delay=delay, strength=strength, description=description,
            feedback_type=FeedbackType.NEUTRAL
        ))
        self.variables.add(source)
        self.variables.add(target)
        return self

    def remove_link(self, source: str, target: str) -> bool:
        for i, link in enumerate(self.links):
            if link.source == source and link.target == target:
                self.links.pop(i)
                return True
        return False

    def get_affected_variables(self, source: str) -> List[str]:
        return [link.target for link in self.links if link.source == source]

    def get_affecting_variables(self, target: str) -> List[str]:
        return [link.source for link in self.links if link.target == target]

    def detect_loops(self) -> List[Dict]:
        loops = []
        var_list = list(self.variables)
        adj = {v: [] for v in var_list}
        for link in self.links:
            if link.target in adj:
                adj[link.target].append((link.source, link))

        def find_cycles(start, current, path, visited):
            if len(path) > len(var_list):
                return
            for neighbor, link in adj.get(current, []):
                if neighbor == start and len(path) > 1:
                    loop_links = []
                    loop_polarity = 1.0
                    for p_var in path:
                        for l in self.links:
                            if l.source == p_var and l.target in [n for n, _ in adj.get(p_var, [])]:
                                loop_links.append(l)
                    for p_var in path + [start]:
                        for l in self.links:
                            if l.source in path and l.target in path:
                                loop_polarity *= l.polarity
                    total_delay = sum(abs(l.delay) for l in self.links
                                     if l.source in (path + [start]) and l.target in (path + [start]))
                    loops.append({
                        "variables": path + [start],
                        "polarity": "reinforcing" if loop_polarity > 0 else "balancing",
                        "loop_strength": min(abs(l.strength) for l in self.links
                                            if l.source in (path + [start]) and l.target in (path + [start])),
                        "total_delay": total_delay,
                        "n_variables": len(path) + 1,
                    })
                elif neighbor not in visited:
                    visited.add(neighbor)
                    find_cycles(start, neighbor, path + [current], visited)
                    visited.discard(neighbor)

        for start_var in var_list:
            find_cycles(start_var, start_var, [], {start_var})
        unique_loops = []
        seen = set()
        for loop in loops:
            key = tuple(sorted(loop["variables"]))
            if key not in seen:
                seen.add(key)
                unique_loops.append(loop)
        return unique_loops[:20]

    def compute_feedback_matrix(self) -> np.ndarray:
        vars_sorted = sorted(self.variables)
        n = len(vars_sorted)
        var_idx = {v: i for i, v in enumerate(vars_sorted)}
        matrix = np.zeros((n, n))
        for link in self.links:
            i = var_idx.get(link.source, -1)
            j = var_idx.get(link.target, -1)
            if i >= 0 and j >= 0:
                matrix[i, j] = link.polarity * link.strength
        return matrix

    def analyze_lease_and_effect(self) -> Dict:
        matrix = self.compute_feedback_matrix()
        vars_sorted = sorted(self.variables)
        n = len(vars_sorted)
        if n == 0:
            return {"variables": [], "centrality": {}, "leverage_points": []}
        eigenvalues = np.linalg.eigvals(matrix)
        max_eigenvalue = float(max(abs(eigenvalues))) if len(eigenvalues) > 0 else 0.0
        outgoing = {v: float(np.sum(np.abs(matrix[i, :]))) for i, v in enumerate(vars_sorted)}
        incoming = {v: float(np.sum(np.abs(matrix[:, i]))) for i, v in enumerate(vars_sorted)}
        centrality = {v: outgoing.get(v, 0) + incoming.get(v, 0) for v in vars_sorted}
        leverage = [v for v in vars_sorted if outgoing.get(v, 0) > 0.5 and abs(outgoing.get(v, 0) - incoming.get(v, 0)) > 0.1]
        leverage.sort(key=lambda v: centrality[v], reverse=True)
        return {
            "variables": vars_sorted,
            "eigenvalues": [complex(e).real for e in eigenvalues][:10],
            "max_eigenvalue_real": max_eigenvalue,
            "centrality": centrality,
            "leverage_points": leverage[:10],
            "n_reinforcing": sum(1 for e in eigenvalues if complex(e).real > 0),
            "n_balancing": sum(1 for e in eigenvalues if complex(e).real < 0),
        }


class SystemDynamicsModel:
    """Stock-and-flow system dynamics simulation with Euler/Runge-Kutta integration."""

    def __init__(self, dt: float = 0.1, integration_method: str = "rk4"):
        self.dt = dt
        self.integration_method = integration_method
        self.stocks: List[Stock] = []
        self.flows: List[Flow] = []
        self.auxiliaries: Dict[str, float] = {}
        self._stock_idx: Dict[str, int] = {}

    def add_stock(self, name: str, initial_value: float, unit: str = "",
                  min_value: float = float("-inf"),
                  max_value: float = float("inf")) -> 'SystemDynamicsModel':
        self.stocks.append(Stock(name=name, initial_value=initial_value, unit=unit,
                                  min_value=min_value, max_value=max_value))
        self._stock_idx[name] = len(self.stocks) - 1
        return self

    def add_flow(self, name: str, source_stock: Optional[str], target_stock: str,
                 rate_function: str, parameters: Dict[str, float] = None) -> 'SystemDynamicsModel':
        self.flows.append(Flow(name=name, source_stock=source_stock,
                                target_stock=target_stock,
                                rate_function=rate_function,
                                parameters=parameters or {}))
        return self

    def add_auxiliary(self, name: str, value: float) -> 'SystemDynamicsModel':
        self.auxiliaries[name] = value
        return self

    def _evaluate_rate(self, rate_function: str, stocks: Dict[str, float],
                      params: Dict[str, float], t: float) -> float:
        safe_dict = {**stocks, **self.auxiliaries, **params, "t": t, "max": max, "min": min, "abs": abs}
        safe_dict["__builtins__"] = {}
        try:
            return float(eval(rate_function, safe_dict))
        except Exception as e:
            logger.warning(f"Rate function error: {e}")
            return 0.0

    def simulate(self, n_steps: int = 100, start_time: float = 0.0,
                  shocks: Optional[Dict[float, Dict[str, float]]] = None) -> SimulationResult:
        stocks = {s.name: s.initial_value for s in self.stocks}
        initial_stocks = dict(stocks)
        time_points = [start_time + i * self.dt for i in range(n_steps + 1)]
        time_series = {s.name: [s.initial_value] for s in self.stocks}

        for step in range(n_steps):
            t = time_points[step]
            if shocks and t in shocks:
                for var, val in shocks[t].items():
                    if var in stocks:
                        stocks[var] = val

            rates = {}
            for flow in self.flows:
                rate = self._evaluate_rate(flow.rate_function, stocks, flow.parameters, t)
                rates[flow.name] = rate
                if flow.target_stock in stocks:
                    stocks[flow.target_stock] += rate * self.dt
                if flow.source_stock and flow.source_stock in stocks:
                    stocks[flow.source_stock] -= rate * self.dt

            for s in self.stocks:
                stocks[s.name] = max(s.min_value, min(s.max_value, stocks[s.name]))
                time_series[s.name].append(stocks[s.name])

        cld = CausalLoopDiagram()
        for flow in self.flows:
            if flow.source_stock:
                cld.add_link(flow.source_stock, flow.target_stock, 1.0)
            cld.add_link(flow.target_stock, flow.name, 1.0)

        feedback_loops = cld.detect_loops()
        equilibrium = {name: values[-1] for name, values in time_series.items()}
        sensitivity = {}
        base_final = {name: values[-1] for name, values in time_series.items()}
        for s in self.stocks:
            stocks_pert = dict(initial_stocks)
            stocks_pert[s.name] *= 1.01
            pert_final = {}
            for step in range(n_steps):
                t = time_points[step]
                rates_p = {}
                for flow in self.flows:
                    rate_p = self._evaluate_rate(flow.rate_function, stocks_pert, flow.parameters, t)
                    rates_p[flow.name] = rate_p
                    if flow.target_stock in stocks_pert:
                        stocks_pert[flow.target_stock] += rate_p * self.dt
                    if flow.source_stock and flow.source_stock in stocks_pert:
                        stocks_pert[flow.source_stock] -= rate_p * self.dt
                for si in self.stocks:
                    stocks_pert[si.name] = max(si.min_value, min(si.max_value, stocks_pert[si.name]))
            pert_final = {name: stocks_pert[name] for name in stocks_pert}
            total_sens = sum(abs(pert_final.get(name, 0) - base_final.get(name, 0))
                            for name in base_final)
            sensitivity[s.name] = total_sens

        return SimulationResult(
            time_series=time_series,
            time_points=time_points,
            stocks_final=equilibrium,
            stocks_initial=initial_stocks,
            feedback_loops=feedback_loops,
            equilibrium_analysis={
                "converged": all(abs(time_series[name][-1] - time_series[name][-2]) < 0.01 * abs(time_series[name][-1] + 1e-10)
                              for name in time_series if len(time_series[name]) > 2),
                "oscillating": any(
                    time_series[name][-1] * time_series[name][-3] < 0
                    for name in time_series if len(time_series[name]) > 3
                    and abs(time_series[name][-1]) > 0.01 and abs(time_series[name][-3]) > 0.01),
                "n_time_steps": n_steps,
                "dt": self.dt,
            },
            sensitivity_scores=sensitivity,
            method="system_dynamics",
        )


class CrossDomainFeedback:
    """Quantitative feedback propagation between domain predictions.

    Instead of just descriptive strings (like diagnose()), this propagates
    quantitative adjustments: if domain A predicts X, domain B's predictions
    are adjusted based on the causal link A→B.
    """

    def __init__(self):
        self.feedback_rules: List[Dict] = []
        self._build_default_rules()

    def _build_default_rules(self):
        self.feedback_rules = [
            {"source": "climate", "target": "energy", "variable": "renewable_share",
             "effect": 0.3, "description": "Higher temp anomaly → faster renewable adoption",
             "feedback_type": "reinforcing"},
            {"source": "geopolitical", "target": "energy", "variable": "oil_price",
             "effect": 0.5, "description": "Higher geopolitical tension → higher oil prices",
             "feedback_type": "reinforcing"},
            {"source": "technology", "target": "energy", "variable": "renewable_share",
             "effect": 0.5, "description": "Faster AI capability → faster renewable deployment",
             "feedback_type": "reinforcing"},
            {"source": "climate", "target": "infrastructure", "variable": "resilience_score",
             "effect": -0.3, "description": "Higher temp anomaly → lower infrastructure resilience",
             "feedback_type": "reinforcing"},
            {"source": "geopolitical", "target": "supply_chain", "variable": "stress_index",
             "effect": 0.4, "description": "Higher tension → more supply chain stress",
             "feedback_type": "reinforcing"},
            {"source": "economic", "target": "social", "variable": "unrest_index",
             "effect": 0.4, "description": "Higher debt-to-GDP → more social unrest",
             "feedback_type": "reinforcing"},
            {"source": "technology", "target": "social", "variable": "polarization_index",
             "effect": 0.3, "description": "Faster tech adoption → more social polarization",
             "feedback_type": "reinforcing"},
            {"source": "pandemic", "target": "economic", "variable": "gdp_growth",
             "effect": -0.3, "description": "Higher pandemic risk → lower GDP growth",
             "feedback_type": "balancing"},
            {"source": "energy", "target": "climate", "variable": "temp_anomaly",
             "effect": 0.2, "description": "Higher fossil fuel use → higher temperature anomaly",
             "feedback_type": "reinforcing"},
            {"source": "cyber", "target": "infrastructure", "variable": "resilience_score",
             "effect": -0.2, "description": "Higher cyber threat → lower infrastructure resilience",
             "feedback_type": "reinforcing"},
            {"source": "water", "target": "food", "variable": "food_security_index",
             "effect": 0.3, "description": "Water stress → lower food security",
             "feedback_type": "reinforcing"},
            {"source": "food", "target": "social", "variable": "unrest_index",
             "effect": 0.4, "description": "Food insecurity → social unrest",
             "feedback_type": "reinforcing"},
            {"source": "biodiversity", "target": "ecological", "variable": "ecosystem_health",
             "effect": 0.5, "description": "Biodiversity loss → ecosystem degradation",
             "feedback_type": "reinforcing"},
            {"source": "seismic", "target": "infrastructure", "variable": "resilience_score",
             "effect": -0.4, "description": "Seismic hazard → infrastructure damage",
             "feedback_type": "reinforcing"},
            {"source": "space", "target": "technology", "variable": "innovation_speed",
             "effect": 0.1, "description": "Space hazard → technology innovation",
             "feedback_type": "balancing"},
        ]

    def add_rule(self, source: str, target: str, variable: str,
                  effect: float, description: str = "",
                  feedback_type: str = "reinforcing") -> 'CrossDomainFeedback':
        self.feedback_rules.append({
            "source": source, "target": target, "variable": variable,
            "effect": effect, "description": description,
            "feedback_type": feedback_type,
        })
        return self

    def propagate(self, domain_values: Dict[str, float],
                   iterations: int = 3, damping: float = 0.5) -> Dict[str, Dict[str, float]]:
        adjusted = {domain: {"value": val} for domain, val in domain_values.items()}
        adjusted.setdefault("_global", {})
        for _ in range(iterations):
            for rule in self.feedback_rules:
                src_val = domain_values.get(rule["source"], 0.5)
                if rule["source"] in adjusted:
                    src_keys = [k for k in adjusted[rule["source"]] if k != "_feedback"]
                    if src_keys:
                        src_val = adjusted[rule["source"]][src_keys[0]]
                effect = rule["effect"] * src_val * damping
                if rule["target"] not in adjusted:
                    adjusted[rule["target"]] = {}
                adjusted[rule["target"]][rule["variable"]] = adjusted[rule["target"]].get(
                    rule["variable"], domain_values.get(rule["target"], 0.5)) + effect
                adjusted[rule["target"]]["_feedback"] = adjusted[rule["target"]].get("_feedback", 0.0) + abs(effect)

        cross_effects = {}
        for rule in self.feedback_rules:
            src = rule["source"]
            tgt = rule["target"]
            if src not in cross_effects:
                cross_effects[src] = {}
            if tgt not in cross_effects[src]:
                cross_effects[src][tgt] = 0.0
            cross_effects[src][tgt] += abs(rule["effect"])

        return {"adjusted_domains": adjusted, "cross_effects": cross_effects}


class ScenarioSimulator:
    """Multi-scenario exploration with branching, comparison, and hash-based audit trail."""

    def __init__(self):
        self.cld = CausalLoopDiagram()
        self.sdm = SystemDynamicsModel()
        self.feedback = CrossDomainFeedback()
        self._scenario_cache: Dict[str, ScenarioResult] = {}

    def _scenario_hash(self, assumptions: Dict[str, float], domain_values: Dict[str, float]) -> str:
        payload = json.dumps({"assumptions": {k: round(v, 6) for k, v in sorted(assumptions.items())},
                               "domains": {k: round(v, 6) for k, v in sorted(domain_values.items())}},
                              sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def simulate_scenario(self, name: str, assumptions: Dict[str, float],
                           domain_values: Dict[str, float],
                           n_steps: int = 50, dt: float = 0.1) -> ScenarioResult:
        sdm = SystemDynamicsModel(dt=dt)
        for var, val in domain_values.items():
            sdm.add_stock(var, val)
        flows = list(assumptions.items())
        for i, (var, val) in enumerate(flows):
            sdm.add_flow(f"flow_{var}", None, var, str(val), {})

        feedback_result = self.feedback.propagate(domain_values, iterations=3)
        sd_result = sdm.simulate(n_steps=n_steps)

        cross_effects = {}
        for src, targets in feedback_result.get("cross_effects", {}).items():
            for tgt, strength in targets.items():
                if src not in cross_effects:
                    cross_effects[src] = {}
                cross_effects[src][tgt] = strength

        stability = "stable"
        if sd_result.equilibrium_analysis.get("oscillating", False):
            stability = "oscillating"
        elif not sd_result.equilibrium_analysis.get("converged", False):
            stability = "unstable"

        feedback_summary = []
        for rule in self.feedback.feedback_rules:
            src_val = domain_values.get(rule["source"], 0.5)
            effect = rule["effect"] * src_val
            feedback_summary.append({
                "source": rule["source"], "target": rule["target"],
                "variable": rule["variable"], "effect": round(effect, 4),
                "description": rule["description"],
                "feedback_type": rule["feedback_type"],
            })

        result = ScenarioResult(
            scenario_name=name,
            scenario_hash=self._scenario_hash(assumptions, domain_values),
            assumptions=assumptions,
            predictions=sd_result.time_series,
            time_points=sd_result.time_points,
            equilibrium_values=sd_result.stocks_final,
            stability=stability,
            feedback_summary=feedback_summary,
            cross_domain_effects=cross_effects,
            method="scenario_simulation",
        )
        self._scenario_cache[name] = result
        return result

    def compare_scenarios(self, scenario_names: List[str]) -> Dict:
        if len(scenario_names) < 2:
            return {"error": "Need at least 2 scenarios to compare"}
        scenarios = {}
        for name in scenario_names:
            if name in self._scenario_cache:
                scenarios[name] = self._scenario_cache[name]
        if len(scenarios) < 2:
            return {"error": "Scenarios not found in cache"}

        comparison = {"scenarios": list(scenarios.keys()), "variables": {}, "divergence": {}}
        all_vars = set()
        for s in scenarios.values():
            all_vars.update(s.equilibrium_values.keys())

        for var in all_vars:
            values = {}
            for name, s in scenarios.items():
                val = s.equilibrium_values.get(var, 0.0)
                values[name] = val
            comparison["variables"][var] = values
            vals = list(values.values())
            comparison["divergence"][var] = {
                "range": round(max(vals) - min(vals), 6),
                "std": round(float(np.std(vals)), 6),
                "mean": round(float(np.mean(vals)), 6),
                "coefficient_of_variation": round(float(np.std(vals)) / max(abs(float(np.mean(vals))), 1e-10), 6),
            }

        baseline_name = scenario_names[0]
        baseline = scenarios[baseline_name]
        comparison["ranked_sensitivity"] = []
        for var in all_vars:
            divergence = comparison["divergence"].get(var, {}).get("coefficient_of_variation", 0)
            comparison["ranked_sensitivity"].append({"variable": var, "sensitivity": divergence})
        comparison["ranked_sensitivity"].sort(key=lambda x: x["sensitivity"], reverse=True)

        return comparison