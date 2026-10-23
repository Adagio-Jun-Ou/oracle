import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class SupplyChainState:
    supplier_concentration: float = 0.40
    logistics_disruption_prob: float = 0.15
    inventory_buffer_days: float = 30.0
    demand_volatility: float = 0.20
    lead_time_variability: float = 0.25
    single_source_dependency: float = 0.30
    geopolitical_route_risk: float = 0.20
    quality_failure_rate: float = 0.05
    cyber_exposure: float = 0.25
    regulatory_risk: float = 0.15
    commodity_price_volatility: float = 0.30
    labor_disruption_risk: float = 0.10
    alternate_supplier_availability: float = 0.60
    visibility_score: float = 0.50


class SupplyChainRiskModel:

    def __init__(self):
        self.name = "supply_chain_risk_model"

    def assess_risk(
        self,
        state: Optional[SupplyChainState] = None,
        **kwargs,
    ) -> dict:
        if state is None:
            state = SupplyChainState(**kwargs)

        concentration_risk = (
            0.30 * state.supplier_concentration +
            0.20 * state.single_source_dependency +
            0.15 * (1 - state.alternate_supplier_availability) +
            0.10 * (1 - state.visibility_score) +
            0.25 * state.logistics_disruption_prob
        )

        disruption_risk = (
            0.25 * state.logistics_disruption_prob +
            0.20 * state.lead_time_variability +
            0.20 * state.demand_volatility +
            0.15 * state.geopolitical_route_risk +
            0.10 * state.labor_disruption_risk +
            0.10 * state.regulatory_risk
        )

        resilience_score = (
            0.25 * state.alternate_supplier_availability +
            0.20 * min(state.inventory_buffer_days / 90, 1.0) +
            0.20 * state.visibility_score +
            0.15 * (1 - state.supplier_concentration) +
            0.20 * (1 - state.single_source_dependency)
        )

        overall_risk = (
            0.35 * concentration_risk +
            0.35 * disruption_risk +
            0.30 * (1 - resilience_score)
        )

        risk_level = "low" if overall_risk < 0.25 else "moderate" if overall_risk < 0.45 else "high" if overall_risk < 0.65 else "critical"

        vulnerabilities = []
        if state.supplier_concentration > 0.50:
            vulnerabilities.append({"area": "supplier_concentration", "score": round(state.supplier_concentration, 3), "severity": "high"})
        if state.single_source_dependency > 0.40:
            vulnerabilities.append({"area": "single_source_dependency", "score": round(state.single_source_dependency, 3), "severity": "high"})
        if state.logistics_disruption_prob > 0.20:
            vulnerabilities.append({"area": "logistics_disruption", "score": round(state.logistics_disruption_prob, 3), "severity": "medium"})
        if state.lead_time_variability > 0.30:
            vulnerabilities.append({"area": "lead_time_variability", "score": round(state.lead_time_variability, 3), "severity": "medium"})
        if state.cyber_exposure > 0.40:
            vulnerabilities.append({"area": "cyber_exposure", "score": round(state.cyber_exposure, 3), "severity": "medium"})
        if state.geopolitical_route_risk > 0.30:
            vulnerabilities.append({"area": "geopolitical_route_risk", "score": round(state.geopolitical_route_risk, 3), "severity": "high"})
        if state.quality_failure_rate > 0.08:
            vulnerabilities.append({"area": "quality_failure_rate", "score": round(state.quality_failure_rate, 3), "severity": "medium"})
        if state.inventory_buffer_days < 14:
            vulnerabilities.append({"area": "low_inventory_buffer", "score": round(state.inventory_buffer_days, 1), "severity": "high"})
        if state.visibility_score < 0.40:
            vulnerabilities.append({"area": "low_visibility", "score": round(state.visibility_score, 3), "severity": "medium"})

        recommendations = []
        if state.alternate_supplier_availability < 0.50:
            recommendations.append("Certify alternate suppliers for critical components to reduce single-source risk")
        if state.inventory_buffer_days < 21:
            recommendations.append(f"Increase safety stock from {state.inventory_buffer_days:.0f} to 30+ days for critical SKUs")
        if state.visibility_score < 0.60:
            recommendations.append("Implement end-to-end supply chain visibility platform (IoT + blockchain)")
        if state.supplier_concentration > 0.40:
            recommendations.append("Diversify supplier base across geographies to reduce concentration risk")
        if state.cyber_exposure > 0.30:
            recommendations.append("Strengthen supply chain cybersecurity with vendor risk assessments and monitoring")
        if state.lead_time_variability > 0.25:
            recommendations.append("Stabilize lead times through nearshoring and dual-sourcing strategies")
        if state.geopolitical_route_risk > 0.25:
            recommendations.append("Map alternate logistics routes and pre-position safety stock near demand centers")

        return {
            "overall_risk": round(overall_risk, 4),
            "risk_level": risk_level,
            "concentration_risk": round(concentration_risk, 4),
            "disruption_risk": round(disruption_risk, 4),
            "resilience_score": round(resilience_score, 4),
            "vulnerabilities": vulnerabilities,
            "recommendations": recommendations,
            "n_vulnerabilities": len(vulnerabilities),
            "critical_vulnerabilities": len([v for v in vulnerabilities if v["severity"] == "high"]),
        }

    def disruption_simulation(
        self,
        state: Optional[SupplyChainState] = None,
        disruption_type: str = "supplier_failure",
        duration_days: int = 30,
        n_simulations: int = 1000,
        seed: int = 42,
        **kwargs,
    ) -> dict:
        if state is None:
            state = SupplyChainState(**kwargs)
        np.random.seed(seed)

        disruption_profiles = {
            "supplier_failure": {"base_impact": 0.40, "recovery_rate": 0.03, "spread_prob": 0.20},
            "logistics_disruption": {"base_impact": 0.30, "recovery_rate": 0.05, "spread_prob": 0.15},
            "cyber_attack": {"base_impact": 0.35, "recovery_rate": 0.04, "spread_prob": 0.25},
            "natural_disaster": {"base_impact": 0.50, "recovery_rate": 0.02, "spread_prob": 0.30},
            "geopolitical_shock": {"base_impact": 0.45, "recovery_rate": 0.02, "spread_prob": 0.35},
            "quality_failure": {"base_impact": 0.25, "recovery_rate": 0.06, "spread_prob": 0.10},
        }

        profile = disruption_profiles.get(disruption_type, disruption_profiles["supplier_failure"])

        revenue_impacts = np.zeros(n_simulations)
        recovery_times = np.zeros(n_simulations)
        max_disruptions_list = np.zeros(n_simulations)

        for sim in range(n_simulations):
            concentration_factor = 1 + state.supplier_concentration
            visibility_penalty = 1 + (1 - state.visibility_score)
            resilience_factor = state.alternate_supplier_availability * 0.3 + min(state.inventory_buffer_days / 90, 1.0) * 0.3 + 0.4

            impact = profile["base_impact"] * concentration_factor * visibility_penalty
            impact *= (1 + np.random.normal(0, 0.1))
            impact /= resilience_factor
            impact = min(impact, 1.0)

            days = np.random.exponential(duration_days / 2)
            recovery = days / (1 + profile["recovery_rate"] * 100 * resilience_factor)
            recovery_times[sim] = recovery

            n_spread = 0
            current_intensity = impact
            for d in range(int(min(recovery, 365))):
                if np.random.random() < profile["spread_prob"] * current_intensity:
                    n_spread += 1
                    current_intensity *= 0.9
            max_disruptions_list[sim] = n_spread

            daily_revenue_loss = impact * (1 - np.exp(-profile["recovery_rate"] * np.arange(int(days))))
            revenue_impacts[sim] = np.sum(daily_revenue_loss) / 365

        p10 = float(np.percentile(revenue_impacts, 10))
        p50 = float(np.percentile(revenue_impacts, 50))
        p90 = float(np.percentile(revenue_impacts, 90))
        avg_recovery = float(np.mean(recovery_times))
        avg_disruptions = float(np.mean(max_disruptions_list))

        return {
            "disruption_type": disruption_type,
            "duration_days": duration_days,
            "n_simulations": n_simulations,
            "revenue_impact": {
                "p10": round(p10, 4),
                "p50": round(p50, 4),
                "p90": round(p90, 4),
                "mean": round(float(np.mean(revenue_impacts)), 4),
            },
            "avg_recovery_days": round(avg_recovery, 1),
            "avg_secondary_disruptions": round(avg_disruptions, 2),
            "worst_case_impact": round(float(np.max(revenue_impacts)), 4),
            "cascade_risk": round(float(np.mean(max_disruptions_list > 2)), 4),
            "recommendations": [
                f"For {disruption_type}: avg recovery = {avg_recovery:.0f} days, avg revenue impact = {p50:.1%}",
                f"Secondary cascade probability: {float(np.mean(max_disruptions_list > 2)):.1%}",
                f"Ensure {'supplier redundancy' if disruption_type == 'supplier_failure' else 'logistics diversification' if disruption_type == 'logistics_disruption' else 'cyber hardening' if disruption_type == 'cyber_attack' else 'geographic diversification'}",
            ],
        }

    def tier_mapping(
        self,
        state: Optional[SupplyChainState] = None,
        n_tiers: int = 3,
        n_suppliers: int = 20,
        seed: int = 42,
        **kwargs,
    ) -> dict:
        if state is None:
            state = SupplyChainState(**kwargs)
        np.random.seed(seed)

        tiers = {}
        suppliers_per_tier = max(n_suppliers // n_tiers, 2)
        total_critical = 0

        for tier in range(1, n_tiers + 1):
            n_sup = suppliers_per_tier + (1 if tier == 1 else 0)
            reliability = np.clip(0.95 - 0.10 * (tier - 1) + np.random.normal(0, 0.03, n_sup), 0.5, 0.99)
            lead_time = np.clip(7 + tier * 10 + np.random.normal(0, 3, n_sup), 1, 60)

            n_critical = int(np.sum(reliability < 0.80))
            total_critical += n_critical

            tiers[f"tier_{tier}"] = {
                "suppliers": n_sup,
                "avg_reliability": round(float(np.mean(reliability)), 3),
                "avg_lead_time_days": round(float(np.mean(lead_time)), 1),
                "critical_suppliers": n_critical,
                "single_source_risk": round(state.single_source_dependency * (1 + 0.1 * tier), 3),
            }

        visibility_per_tier = max(0.2, state.visibility_score - 0.15 * (n_tiers - 1))

        return {
            "n_tiers": n_tiers,
            "total_suppliers": n_suppliers,
            "total_critical": total_critical,
            "visibility_by_tier": {
                f"tier_{i+1}": round(min(1.0, state.visibility_score * (0.85 ** i)), 3)
                for i in range(n_tiers)
            },
            "tiers": tiers,
            "overall_visibility": round(state.visibility_score, 3),
            "concentration_hhi": round(state.supplier_concentration ** 2 * 10000, 1),
            "recommendations": [
                f"Supply chain has {total_critical} critical suppliers across {n_tiers} tiers",
                f"Visibility drops to {state.visibility_score * (0.85 ** (n_tiers-1)):.0%} at tier {n_tiers} — invest in tier-N mapping",
                f"HHI concentration index: {state.supplier_concentration ** 2 * 10000:.0f} ({'highly concentrated' if state.supplier_concentration > 0.5 else 'moderately concentrated' if state.supplier_concentration > 0.25 else 'diversified'})",
            ] + (["Urgent: Reduce single-source dependency before tier-1 disruption cascades"] if state.single_source_dependency > 0.4 else []),
        }