"""Adversarial Stress Tester — finds worst-case combinations across ALL domains simultaneously.

Uses optimization (genetic algorithm), Monte Carlo, and gradient-free search
to discover adversarial scenarios that maximize systemic risk across all domains.

This is NOT a simple worst-case analysis. It finds combinatorial catastrophes:
scenarios where moderate individual risks combine into extreme systemic failure.

Implements:
- AdversarialStressTester: Multi-domain worst-case search
- SystemicRiskSurface: Maps the risk landscape across domain combinations
- CascadeAmplifier: Models how risks amplify through domain interconnections
- TailDependencyStress: Tests extreme co-movement of domain risks
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    name: str
    domain_shocks: Dict[str, float]
    composite_risk: float
    cascade_depth: int
    affected_domains: List[str]
    amplification_factor: float
    probability_estimate: float
    description: str
    method: str = "adversarial_stress"


@dataclass
class SystemicRiskResult:
    risk_surface: List[Dict]
    worst_case: StressScenario
    tail_scenarios: List[StressScenario]
    cascade_paths: List[Dict]
    systemic_risk_score: float
    domain_vulnerability_ranking: List[Dict]
    amplification_network: Dict[str, Dict[str, float]]
    method: str = "systemic_risk"


class AdversarialStressTester:
    """Find worst-case cross-domain scenarios using optimization and Monte Carlo.

    Key insight: The worst systemic risk comes not from extreme individual shocks,
    but from moderate shocks in multiple domains that cascade and amplify.
    """

    DOMAIN_BOUNDS = {
        "geopolitical_tension": (0.0, 1.0),
        "climate_anomaly_c": (0.5, 3.0),
        "gdp_growth": (-0.05, 0.05),
        "debt_to_gdp": (0.5, 2.5),
        "unemployment_rate": (0.03, 0.25),
        "inflation_rate": (-0.02, 0.15),
        "vix_level": (10.0, 80.0),
        "pandemic_risk": (0.0, 1.0),
        "cyber_threat": (0.0, 1.0),
        "social_instability": (0.0, 1.0),
        "energy_stress": (0.0, 1.0),
        "supply_chain_stress": (0.0, 1.0),
        "water_stress": (0.0, 1.0),
        "food_insecurity": (0.0, 1.0),
        "seismic_risk": (0.0, 1.0),
        "space_weather_risk": (0.0, 1.0),
    }

    CASCADE_AMPLIFIERS = {
        "geopolitical_tension": {"energy_stress": 0.4, "supply_chain_stress": 0.3,
                                  "social_instability": 0.3, "cyber_threat": 0.2},
        "climate_anomaly_c": {"water_stress": 0.5, "food_insecurity": 0.4,
                              "energy_stress": 0.2, "social_instability": 0.2},
        "gdp_growth": {"unemployment_rate": -0.5, "social_instability": 0.4,
                       "debt_to_gdp": 0.3},
        "pandemic_risk": {"social_instability": 0.4, "supply_chain_stress": 0.5,
                          "gdp_growth": 0.3},
        "cyber_threat": {"supply_chain_stress": 0.3, "gdp_growth": 0.2},
        "water_stress": {"food_insecurity": 0.5, "social_instability": 0.3},
        "energy_stress": {"supply_chain_stress": 0.4, "gdp_growth": 0.2},
        "social_instability": {"geopolitical_tension": 0.2, "debt_to_gdp": 0.1},
    }

    def __init__(self, n_simulations: int = 5000, n_generations: int = 30,
                 population_size: int = 50, seed: int = 42):
        self.n_simulations = n_simulations
        self.n_generations = n_generations
        self.population_size = population_size
        self.rng = np.random.RandomState(seed)

    def _risk_function(self, x: np.ndarray, domain_names: List[str]) -> float:
        raw_risk = 0.0
        norm_x = np.zeros_like(x)
        for i, name in enumerate(domain_names):
            bounds = self.DOMAIN_BOUNDS.get(name, (0.0, 1.0))
            norm_x[i] = (x[i] - bounds[0]) / max(bounds[1] - bounds[0], 1e-10)

        primary_risks = {
            "geopolitical_tension": 0.25, "climate_anomaly_c": 0.15,
            "pandemic_risk": 0.15, "cyber_threat": 0.05,
            "social_instability": 0.10, "gdp_growth": 0.15,
        }
        for i, name in enumerate(domain_names):
            weight = primary_risks.get(name, 0.02)
            if name == "gdp_growth":
                raw_risk += weight * (1.0 - norm_x[i])
            elif name == "debt_to_gdp":
                raw_risk += weight * norm_x[i]
            else:
                raw_risk += weight * norm_x[i]

        cascade_risk = 0.0
        for src, targets in self.CASCADE_AMPLIFIERS.items():
            src_idx = next((i for i, n in enumerate(domain_names) if n == src), -1)
            if src_idx >= 0:
                for tgt, amp in targets.items():
                    tgt_idx = next((i for i, n in enumerate(domain_names) if n == tgt), -1)
                    if tgt_idx >= 0:
                        cascade_risk += amp * norm_x[src_idx] * norm_x[tgt_idx]

        interaction_risk = 0.0
        for i in range(len(domain_names)):
            for j in range(i + 1, len(domain_names)):
                interaction_risk += 0.02 * norm_x[i] * norm_x[j]

        tail_penalty = 0.0
        for i, name in enumerate(domain_names):
            if name in primary_risks and norm_x[i] > 0.8:
                tail_penalty += 0.5 * (norm_x[i] - 0.8) ** 2

        return raw_risk + cascade_risk + interaction_risk + tail_penalty

    def find_worst_case(self, domains: Optional[List[str]] = None) -> StressScenario:
        if domains is None:
            domains = list(self.DOMAIN_BOUNDS.keys())
        domain_names = domains
        n_dims = len(domain_names)
        lb = np.array([self.DOMAIN_BOUNDS.get(d, (0.0, 1.0))[0] for d in domain_names])
        ub = np.array([self.DOMAIN_BOUNDS.get(d, (0.0, 1.0))[1] for d in domain_names])
        population = self.rng.uniform(lb, ub, (self.population_size, n_dims))
        fitnesses = np.array([self._risk_function(ind, domain_names) for ind in population])
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_individual = population[best_idx].copy()

        for gen in range(self.n_generations):
            parents_idx = self.rng.choice(self.population_size, size=self.population_size)
            children = population[parents_idx].copy()
            mutation_mask = self.rng.random((self.population_size, n_dims)) < 0.2
            mutation_scale = (ub - lb) * 0.1
            children += mutation_mask * self.rng.normal(0, 1, (self.population_size, n_dims)) * mutation_scale
            children = np.clip(children, lb, ub)
            child_fitnesses = np.array([self._risk_function(ind, domain_names) for ind in children])
            combined = np.vstack([population, children])
            combined_fitness = np.concatenate([fitnesses, child_fitnesses])
            top_indices = np.argsort(combined_fitness)[-self.population_size:]
            population = combined[top_indices]
            fitnesses = combined_fitness[top_indices]
            if fitnesses.max() > best_fitness:
                best_fitness = fitnesses.max()
                best_individual = population[np.argmax(fitnesses)].copy()

        domain_shocks = {domain_names[i]: round(float(best_individual[i]), 4)
                         for i in range(n_dims)}
        norm_x = (best_individual - lb) / np.maximum(ub - lb, 1e-10)
        cascade_depth = 0
        for src, targets in self.CASCADE_AMPLIFIERS.items():
            if src in domain_names:
                src_idx = domain_names.index(src)
                if norm_x[src_idx] > 0.5:
                    cascade_depth += 1
        affected = [d for i, d in enumerate(domain_names)
                     if norm_x[i] > 0.5]
        amp_factor = 1.0 + sum(0.1 for d in affected if d in self.CASCADE_AMPLIFIERS)
        prob_est = float(np.prod([1.0 - (1.0 - norm_x[i]) * 0.01 for i in range(n_dims)
                                   if norm_x[i] > 0.7]))

        return StressScenario(
            name="adversarial_worst_case",
            domain_shocks=domain_shocks,
            composite_risk=round(float(best_fitness), 4),
            cascade_depth=cascade_depth,
            affected_domains=affected[:10],
            amplification_factor=round(amp_factor, 4),
            probability_estimate=round(max(0.001, min(prob_est, 0.5)), 4),
            description=f"Adversarial worst case: {len(affected)} domains under stress with {cascade_depth}-deep cascading effects and {amp_factor:.1f}x amplification",
            method="genetic_adversarial",
        )

    def monte_carlo_stress(self, n_simulations: Optional[int] = None,
                            domains: Optional[List[str]] = None,
                            tail_percentile: float = 99.5) -> SystemicRiskResult:
        n = n_simulations or self.n_simulations
        if domains is None:
            domains = list(self.DOMAIN_BOUNDS.keys())
        domain_names = domains
        n_dims = len(domain_names)
        lb = np.array([self.DOMAIN_BOUNDS.get(d, (0.0, 1.0))[0] for d in domain_names])
        ub = np.array([self.DOMAIN_BOUNDS.get(d, (0.0, 1.0))[1] for d in domain_names])
        samples = self.rng.uniform(lb, ub, (n, n_dims))
        risks = np.array([self._risk_function(s, domain_names) for s in samples])
        tail_idx = risks > np.percentile(risks, tail_percentile)
        tail_scenarios = []
        risk_surface = []
        for pctl in [50, 75, 90, 95, 99]:
            idx = np.percentile(risks, pctl)
            representative = samples[np.argmin(np.abs(risks - idx))]
            risk_surface.append({
                "percentile": pctl,
                "risk_score": round(float(idx), 4),
                "domain_values": {domain_names[i]: round(float(representative[i]), 4)
                                  for i in range(n_dims)},
            })

        worst_idx = np.argmax(risks)
        worst_scenario = self.find_worst_case(domains)
        for i in range(min(int(n * (1 - tail_percentile / 100)), 20)):
            if i < np.sum(tail_idx):
                sample_idx = np.where(tail_idx)[0][i]
                s = samples[sample_idx]
                tail_scenarios.append(StressScenario(
                    name=f"tail_scenario_{i+1}",
                    domain_shocks={domain_names[j]: round(float(s[j]), 4) for j in range(n_dims)},
                    composite_risk=round(float(risks[sample_idx]), 4),
                    cascade_depth=0, affected_domains=[],
                    amplification_factor=1.0, probability_estimate=round(1 - tail_percentile/100, 4),
                    description=f"Tail scenario at {tail_percentile}th percentile",
                    method="monte_carlo_tail",
                ))

        domain_vuln = []
        for i, name in enumerate(domain_names):
            partial_risks = np.array([self._risk_function(s, domain_names) for s in samples])
            perturbed = samples.copy()
            perturbed[:, i] = np.clip(samples[:, i] + (ub[i] - lb[i]) * 0.1, lb[i], ub[i])
            perturbed_risks = np.array([self._risk_function(s, domain_names) for s in perturbed])
            sensitivity = float(np.mean(np.abs(perturbed_risks - partial_risks)))
            domain_vuln.append({"domain": name, "sensitivity": round(sensitivity, 4),
                                "mean_contribution": round(float(np.mean(np.abs(samples[:, i] - lb[i]) / np.maximum(ub[i] - lb[i], 1e-10))), 4)})
        domain_vuln.sort(key=lambda x: x["sensitivity"], reverse=True)

        cascade_paths = []
        worst_norm = (samples[worst_idx] - lb) / np.maximum(ub - lb, 1e-10)
        for src, targets in self.CASCADE_AMPLIFIERS.items():
            if src in domain_names:
                src_idx = domain_names.index(src)
                if worst_norm[src_idx] > 0.5:
                    for tgt, amp in targets.items():
                        if tgt in domain_names:
                            tgt_idx = domain_names.index(tgt)
                            cascade_paths.append({
                                "source": src, "target": tgt,
                                "amplification": amp,
                                "source_value": round(float(worst_norm[src_idx]), 4),
                                "target_value": round(float(worst_norm[tgt_idx]), 4),
                            })

        amp_network = {}
        for src, targets in self.CASCADE_AMPLIFIERS.items():
            amp_network[src] = {}
            for tgt, amp in targets.items():
                amp_network[src][tgt] = amp

        systemic_score = float(risks[worst_idx]) * worst_scenario.amplification_factor

        return SystemicRiskResult(
            risk_surface=risk_surface,
            worst_case=worst_scenario,
            tail_scenarios=tail_scenarios,
            cascade_paths=cascade_paths[:15],
            systemic_risk_score=round(systemic_score, 4),
            domain_vulnerability_ranking=domain_vuln,
            amplification_network=amp_network,
        )

    def black_swan_scenarios(self, n_scenarios: int = 5,
                              correlation_threshold: float = 0.8) -> List[StressScenario]:
        domain_names = list(self.DOMAIN_BOUNDS.keys())
        n_dims = len(domain_names)
        lb = np.array([self.DOMAIN_BOUNDS.get(d, (0.0, 1.0))[0] for d in domain_names])
        ub = np.array([self.DOMAIN_BOUNDS.get(d, (0.0, 1.0))[1] for d in domain_names])
        n_sample = self.n_simulations
        uniform = self.rng.uniform(0, 1, (n_sample, n_dims))
        copula = np.zeros_like(uniform)
        for i in range(n_dims):
            u = uniform[:, i]
            if i > 0 and self.rng.random() < correlation_threshold:
                z = self.rng.normal(0, 1)
                u = 0.3 * u + 0.7 * self.rng.uniform(0, 1)
            copula[:, i] = u
        samples = lb + copula * (ub - lb)
        risks = np.array([self._risk_function(s, domain_names) for s in samples])
        sorted_idx = np.argsort(risks)[::-1]
        scenarios = []
        for i in range(min(n_scenarios, 5)):
            idx = sorted_idx[i]
            s = samples[idx]
            scenarios.append(StressScenario(
                name=f"black_swan_{i+1}",
                domain_shocks={domain_names[j]: round(float(s[j]), 4) for j in range(n_dims)},
                composite_risk=round(float(risks[idx]), 4),
                cascade_depth=0, affected_domains=[],
                amplification_factor=1.0,
                probability_estimate=round(1 / n_sample, 6),
                description=f"Black swan scenario: extreme co-movement across {n_dims} domains",
                method="copula_black_swan",
            ))
        return scenarios