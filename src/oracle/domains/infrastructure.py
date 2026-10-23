import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class InfrastructureState:
    power_grid_reliability: float = 0.85
    water_system_reliability: float = 0.90
    transport_reliability: float = 0.80
    telecom_reliability: float = 0.88
    healthcare_capacity: float = 0.70
    financial_system_reliability: float = 0.92
    supply_chain_reliability: float = 0.75
    cyber_defense_score: float = 0.60
    redundancy_level: float = 0.40
    maintenance_score: float = 0.65
    interdependency_strength: float = 0.50
    climate_exposure: float = 0.30
    seismic_risk: float = 0.20
    flood_risk: float = 0.25


class InfrastructureResilienceModel:

    def __init__(self):
        self.name = "infrastructure_resilience_model"

    def assess_resilience(
        self,
        state: Optional[InfrastructureState] = None,
        **kwargs,
    ) -> dict:
        if state is None:
            state = InfrastructureState(**kwargs)

        sectors = {
            "power_grid": state.power_grid_reliability,
            "water_system": state.water_system_reliability,
            "transport": state.transport_reliability,
            "telecom": state.telecom_reliability,
            "healthcare": state.healthcare_capacity,
            "financial": state.financial_system_reliability,
            "supply_chain": state.supply_chain_reliability,
        }

        redundancy_score = state.redundancy_level * 0.3 + state.maintenance_score * 0.3 + state.cyber_defense_score * 0.4
        avg_reliability = float(np.mean(list(sectors.values())))
        min_reliability = float(min(sectors.values()))
        resilience_index = (
            0.30 * avg_reliability +
            0.20 * min_reliability +
            0.20 * redundancy_score +
            0.15 * (1 - state.interdependency_strength) +
            0.15 * (1 - max(state.climate_exposure, state.seismic_risk, state.flood_risk))
        )

        vulnerabilities = []
        for name, score in sectors.items():
            if score < 0.70:
                vulnerabilities.append({"sector": name, "reliability": round(score, 3), "severity": "critical"})
            elif score < 0.80:
                vulnerabilities.append({"sector": name, "reliability": round(score, 3), "severity": "high"})
            elif score < 0.90:
                vulnerabilities.append({"sector": name, "reliability": round(score, 3), "severity": "moderate"})

        if state.interdependency_strength > 0.6:
            vulnerabilities.append({"sector": "interdependency_cascade", "reliability": round(1 - state.interdependency_strength, 3), "severity": "high"})

        hazard_exposure = max(state.climate_exposure, state.seismic_risk, state.flood_risk)
        threat_level = "low" if resilience_index > 0.80 else "moderate" if resilience_index > 0.65 else "high" if resilience_index > 0.50 else "critical"

        recommendations = []
        weak_sectors = sorted(sectors.items(), key=lambda x: x[1])[:3]
        for name, score in weak_sectors:
            if score < 0.80:
                recommendations.append(f"Critical: Reinforce {name.replace('_', ' ')} infrastructure (current: {score:.0%})")
        if state.redundancy_level < 0.50:
            recommendations.append("Increase system redundancy to prevent single-point failures")
        if state.cyber_defense_score < 0.70:
            recommendations.append("Strengthen cyber defense posture across critical infrastructure")
        if state.interdependency_strength > 0.60:
            recommendations.append("Reduce inter-system coupling to limit cascade potential")
        if hazard_exposure > 0.30:
            recommendations.append("Harden physical infrastructure against environmental hazards")

        return {
            "resilience_index": round(resilience_index, 4),
            "threat_level": threat_level,
            "avg_sector_reliability": round(avg_reliability, 4),
            "weakest_sector": min(sectors.items(), key=lambda x: x[1])[0],
            "weakest_sector_reliability": round(min_reliability, 4),
            "redundancy_score": round(redundancy_score, 4),
            "hazard_exposure": round(hazard_exposure, 4),
            "vulnerabilities": vulnerabilities,
            "recommendations": recommendations,
            "sectors": {k: round(v, 3) for k, v in sectors.items()},
        }

    def cascade_simulation(
        self,
        state: Optional[InfrastructureState] = None,
        initial_failure: str = "power_grid",
        n_simulations: int = 1000,
        seed: int = 42,
        **kwargs,
    ) -> dict:
        if state is None:
            state = InfrastructureState(**kwargs)
        np.random.seed(seed)

        sectors = {
            "power_grid": state.power_grid_reliability,
            "water_system": state.water_system_reliability,
            "transport": state.transport_reliability,
            "telecom": state.telecom_reliability,
            "healthcare": state.healthcare_capacity,
            "financial": state.financial_system_reliability,
            "supply_chain": state.supply_chain_reliability,
        }

        dependents = {
            "power_grid": ["water_system", "telecom", "healthcare", "financial", "transport"],
            "water_system": ["healthcare", "supply_chain"],
            "transport": ["supply_chain", "healthcare"],
            "telecom": ["financial", "healthcare", "supply_chain"],
            "healthcare": [],
            "financial": ["supply_chain"],
            "supply_chain": [],
        }

        n_failed_list = []
        cascade_paths = []
        affected_sectors_counts = {s: 0 for s in sectors}

        for _ in range(n_simulations):
            failed = {initial_failure}
            queue = [initial_failure]
            path = [initial_failure]

            while queue:
                current = queue.pop(0)
                for dep in dependents.get(current, []):
                    if dep not in failed:
                        casc_prob = state.interdependency_strength * (1 - sectors[dep])
                        if np.random.random() < casc_prob:
                            failed.add(dep)
                            queue.append(dep)
                            path.append(dep)

            n_failed_list.append(len(failed))
            cascade_paths.append(path)
            for s in failed:
                affected_sectors_counts[s] += 1

        avg_failed = float(np.mean(n_failed_list))
        max_failed = int(max(n_failed_list))
        total_collapse_prob = float(sum(1 for n in n_failed_list if n == len(sectors)) / n_simulations)

        sector_failure_probs = {s: round(c / n_simulations, 4) for s, c in affected_sectors_counts.items()}

        most_common_path = max(set(tuple(p) for p in cascade_paths), key=lambda p: cascade_paths.count(list(p)))

        return {
            "initial_failure": initial_failure,
            "n_simulations": n_simulations,
            "avg_sectors_failed": round(avg_failed, 2),
            "max_sectors_failed": max_failed,
            "total_collapse_probability": round(total_collapse_prob, 4),
            "cascade_risk": "critical" if total_collapse_prob > 0.1 else "high" if total_collapse_prob > 0.05 else "moderate" if avg_failed > 2 else "low",
            "sector_failure_probabilities": sector_failure_probs,
            "most_common_path": list(most_common_path),
            "recommendations": [
                f"Primary failure point: {initial_failure} — ensure robust backup systems",
                f"Most vulnerable cascade target: {max(sector_failure_probs, key=sector_failure_probs.get)}",
                f"Average cascading failures: {avg_failed:.1f} sectors per event",
            ] + (["Total system collapse risk significant — emergency protocols needed"] if total_collapse_prob > 0.05 else []),
        }

    def stress_test(
        self,
        state: Optional[InfrastructureState] = None,
        scenarios: Optional[list] = None,
        **kwargs,
    ) -> dict:
        if state is None:
            state = InfrastructureState(**kwargs)

        if scenarios is None:
            scenarios = [
                {"name": "cyber_attack", "sectors_affected": {"telecom": -0.30, "financial": -0.20, "power_grid": -0.10}},
                {"name": "earthquake", "sectors_affected": {"transport": -0.40, "water_system": -0.30, "power_grid": -0.25}},
                {"name": "extreme_weather", "sectors_affected": {"power_grid": -0.35, "transport": -0.25, "water_system": -0.20}},
                {"name": "pandemic", "sectors_affected": {"healthcare": -0.30, "supply_chain": -0.25, "transport": -0.15}},
                {"name": "supply_crisis", "sectors_affected": {"supply_chain": -0.35, "financial": -0.10, "healthcare": -0.15}},
            ]

        results = []
        for scenario in scenarios:
            mod_state = InfrastructureState(
                power_grid_reliability=max(0.1, state.power_grid_reliability + scenario["sectors_affected"].get("power_grid", 0)),
                water_system_reliability=max(0.1, state.water_system_reliability + scenario["sectors_affected"].get("water_system", 0)),
                transport_reliability=max(0.1, state.transport_reliability + scenario["sectors_affected"].get("transport", 0)),
                telecom_reliability=max(0.1, state.telecom_reliability + scenario["sectors_affected"].get("telecom", 0)),
                healthcare_capacity=max(0.1, state.healthcare_capacity + scenario["sectors_affected"].get("healthcare", 0)),
                financial_system_reliability=max(0.1, state.financial_system_reliability + scenario["sectors_affected"].get("financial", 0)),
                supply_chain_reliability=max(0.1, state.supply_chain_reliability + scenario["sectors_affected"].get("supply_chain", 0)),
                redundancy_level=state.redundancy_level,
                maintenance_score=state.maintenance_score,
                cyber_defense_score=state.cyber_defense_score,
                interdependency_strength=state.interdependency_strength,
                climate_exposure=state.climate_exposure,
                seismic_risk=state.seismic_risk,
                flood_risk=state.flood_risk,
            )
            resilience = self.assess_resilience(mod_state)
            cascade = self.cascade_simulation(mod_state, initial_failure=list(scenario["sectors_affected"].keys())[0], n_simulations=200)
            results.append({
                "scenario": scenario["name"],
                "resilience_under_stress": resilience["resilience_index"],
                "threat_level": resilience["threat_level"],
                "weakest_sector": resilience["weakest_sector"],
                "cascade_collapse_prob": cascade["total_collapse_probability"],
                "avg_cascade_failures": cascade["avg_sectors_failed"],
            })

        baseline = self.assess_resilience(state)
        worst = min(results, key=lambda r: r["resilience_under_stress"])

        return {
            "baseline_resilience": baseline["resilience_index"],
            "scenario_results": results,
            "worst_scenario": worst["scenario"],
            "worst_resilience": worst["resilience_under_stress"],
            "resilience_drop": round(baseline["resilience_index"] - worst["resilience_under_stress"], 4),
            "recommendations": [
                f"Worst case scenario: {worst['scenario']} (resilience drops to {worst['resilience_under_stress']:.2f})",
                f"Prepare mitigation for {worst['scenario']} — reinforce affected sectors",
            ],
        }

    def recovery_analysis(
        self,
        state: Optional[InfrastructureState] = None,
        damage_level: float = 0.40,
        days: int = 90,
        resource_level: float = 0.60,
        seed: int = 42,
        **kwargs,
    ) -> dict:
        if state is None:
            state = InfrastructureState(**kwargs)
        np.random.seed(seed)

        maintenance_factor = state.maintenance_score * 0.6 + state.redundancy_level * 0.4
        base_recovery_rate = 0.02 * (1 + resource_level) * (1 + maintenance_factor)

        daily_recovery = np.zeros(days)
        current_damage = damage_level

        for d in range(days):
            recovery_speed = base_recovery_rate * (1 + 0.5 * current_damage)
            noise = np.random.normal(0, 0.005)
            actual_recovery = max(0, recovery_speed + noise)
            current_damage = max(0, current_damage - actual_recovery)
            daily_recovery[d] = 1 - current_damage

        days_to_50 = None
        days_to_80 = None
        days_to_95 = None
        for d, r in enumerate(daily_recovery):
            if r >= 0.50 and days_to_50 is None:
                days_to_50 = d
            if r >= 0.80 and days_to_80 is None:
                days_to_80 = d
            if r >= 0.95 and days_to_95 is None:
                days_to_95 = d

        final_recovery = float(daily_recovery[-1])
        phases = []
        if days_to_50:
            phases.append(f"Emergency phase: {days_to_50} days to 50% recovery")
        if days_to_80:
            phases.append(f"Stabilization phase: {days_to_80} days to 80% recovery")
        if days_to_95:
            phases.append(f"Full recovery: {days_to_95} days to 95% recovery")
        else:
            phases.append(f"Full recovery: >{days} days (requires extended resources)")

        resilience_rating = "excellent" if final_recovery > 0.95 else "good" if final_recovery > 0.85 else "moderate" if final_recovery > 0.70 else "poor"

        return {
            "initial_damage_level": damage_level,
            "resource_level": resource_level,
            "maintenance_factor": round(maintenance_factor, 3),
            "base_recovery_rate": round(base_recovery_rate, 5),
            "days_to_50pct": days_to_50,
            "days_to_80pct": days_to_80,
            "days_to_95pct": days_to_95,
            "final_recovery_level": round(final_recovery, 4),
            "resilience_rating": resilience_rating,
            "phases": phases,
            "recommendations": [
                f"Resource allocation: {'adequate' if resource_level > 0.5 else 'insufficient'} — {'maintain' if resource_level > 0.5 else 'increase'} emergency stocks",
                f"Maintenance readiness: {'strong' if maintenance_factor > 0.6 else 'weak'} — {'sustain' if maintenance_factor > 0.6 else 'improve'} preventive maintenance",
            ],
        }