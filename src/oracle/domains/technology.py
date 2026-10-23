import numpy as np
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class TechPrediction:
    technology: str = ""
    current_adoption: float = 0.0
    predicted_adoption_1y: float = 0.0
    predicted_adoption_5y: float = 0.0
    saturation: float = 0.0
    years_to_50pct: float = 0.0
    disruption_score: float = 0.0
    phase: str = ""
    confidence: float = 0.0


class TechnologyForecaster:

    def __init__(self):
        self.name = "technology_forecaster"

    @staticmethod
    def s_curve_adoption(
        technology: str = "AI",
        current_adoption: float = 0.15,
        market_potential: float = 1.0,
        innovation_speed: float = 0.3,
        resistance: float = 0.1,
        years: int = 20,
    ) -> dict:
        L = market_potential
        k = innovation_speed
        x0 = -np.log(current_adoption / (L - current_adoption)) / k if current_adoption > 0 and current_adoption < L else 5.0
        t = np.arange(0, years + 1)
        adoption = L / (1 + np.exp(-k * (t - x0)))
        adoption = np.clip(adoption * (1 - resistance * np.exp(-0.2 * t)), 0, L)
        years_to_50 = None
        for i, a in enumerate(adoption):
            if a >= 0.5 * L:
                years_to_50 = float(i)
                break
        if years_to_50 is None:
            years_to_50 = float("inf")
        yearly = [round(float(a), 4) for a in adoption]
        phase = "early" if current_adoption < 0.16 * L else "growth" if current_adoption < 0.5 * L else "maturity" if current_adoption < 0.84 * L else "saturation"
        growth_velocity = [0.0] + [round(float(yearly[i] - yearly[i - 1]), 4) for i in range(1, len(yearly))]
        return {
            "technology": technology,
            "current_adoption": round(float(current_adoption), 4),
            "predicted_adoption_1y": round(float(adoption[1]), 4),
            "predicted_adoption_5y": round(float(adoption[min(5, years)]), 4),
            "saturation": round(float(L), 4),
            "years_to_50pct": round(years_to_50, 1),
            "innovation_speed": round(float(k), 4),
            "resistance": round(float(resistance), 4),
            "phase": phase,
            "yearly_adoption": yearly,
            "yearly_velocity": growth_velocity,
            "years": list(range(years + 1)),
        }

    @staticmethod
    def ai_capability_forecast(
        current_capability: float = 0.35,
        compute_growth_rate: float = 0.4,
        algorithm_efficiency_gain: float = 0.15,
        data_multiplier: float = 1.5,
        regulatory_slowdown: float = 0.05,
        years: int = 10,
    ) -> dict:
        capabilities = []
        cap = current_capability
        for year in range(years + 1):
            compute_boost = compute_growth_rate * (1.4 ** year) / (1 + 0.1 * year)
            algo_boost = algorithm_efficiency_gain * (1 - 0.3 * min(year / 5, 1))
            data_boost = 0.05 * np.log(1 + data_multiplier * year)
            regulatory_drag = regulatory_slowdown * min(year / 5, 1)
            total_boost = compute_boost + algo_boost + data_boost - regulatory_drag
            capabilities.append(round(float(cap), 4))
            cap = min(cap + total_boost * (1 - cap), 0.99)
        milestones = []
        thresholds = [
            (0.50, "Narrow superintelligence in most domains"),
            (0.65, "Creative and strategic AI assistance"),
            (0.80, "Autonomous AI research"),
            (0.90, "Near-AGI capability"),
            (0.95, "Full AGI-level performance"),
        ]
        for threshold, label in thresholds:
            for y, c in enumerate(capabilities):
                if c >= threshold:
                    milestones.append({"year": y, "capability": round(float(c), 3), "milestone": label})
                    break
        return {
            "current_capability": round(float(current_capability), 3),
            "predicted_capabilities": capabilities,
            "years": list(range(years + 1)),
            "milestones_reached": milestones,
            "compute_growth_rate": round(float(compute_growth_rate), 3),
            "algorithm_efficiency_gain": round(float(algorithm_efficiency_gain), 3),
            "regulatory_slowdown": round(float(regulatory_slowdown), 3),
            "years_to_near_agi": next((y for y, c in enumerate(capabilities) if c >= 0.80), None),
        }

    @staticmethod
    def disruption_analysis(
        industry: str = "technology",
        technologies: Optional[List[dict]] = None,
        timeline_years: int = 10,
    ) -> dict:
        if technologies is None:
            technologies = [
                {"name": "AI/ML", "impact": 0.9, "probability": 0.85, "speed": 0.7, "displacement": 0.4},
                {"name": "Quantum Computing", "impact": 0.7, "probability": 0.3, "speed": 0.3, "displacement": 0.2},
                {"name": "Blockchain/Web3", "impact": 0.5, "probability": 0.5, "speed": 0.5, "displacement": 0.15},
                {"name": "Biotechnology", "impact": 0.8, "probability": 0.6, "speed": 0.4, "displacement": 0.25},
                {"name": "Clean Energy", "impact": 0.85, "probability": 0.9, "speed": 0.6, "displacement": 0.35},
                {"name": "Robotics/Automation", "impact": 0.75, "probability": 0.7, "speed": 0.5, "displacement": 0.3},
            ]
        for tech in technologies:
            score = tech["impact"] * tech["probability"] * tech["speed"]
            tech["disruption_score"] = round(float(score), 4)
            tech["years_to_impact"] = round(float(timeline_years * (1 - tech["speed"])), 1)
            tech["job_displacement_millions"] = round(float(tech["displacement"] * 150 * tech["probability"]), 1)
        sorted_techs = sorted(technologies, key=lambda x: x["disruption_score"], reverse=True)
        total_disruption = sum(t["disruption_score"] for t in sorted_techs) / len(sorted_techs)
        winners = [t["name"] for t in sorted_techs[:3]]
        losers_sectors = ["Fossil Fuels", "Manual Labor", "Traditional Banking", "Print Media"]
        return {
            "industry": industry,
            "technologies": sorted_techs,
            "total_disruption_score": round(float(total_disruption), 4),
            "top_disruptors": winners,
            "displaced_sectors": losers_sectors[:3],
            "timeline_years": timeline_years,
        }

    @staticmethod
    def technology_adoption_comparison(
        technologies: Optional[List[dict]] = None,
    ) -> dict:
        if technologies is None:
            technologies = [
                {"name": "Internet", "current": 0.92, "speed": 0.5, "potential": 1.0},
                {"name": "Smartphones", "current": 0.85, "speed": 0.6, "potential": 0.95},
                {"name": "AI Assistants", "current": 0.15, "speed": 0.35, "potential": 0.9},
                {"name": "Electric Vehicles", "current": 0.18, "speed": 0.25, "potential": 0.8},
                {"name": "CRISPR/Gene Edit", "current": 0.03, "speed": 0.15, "potential": 0.6},
                {"name": "Quantum Computing", "current": 0.01, "speed": 0.1, "potential": 0.4},
                {"name": "Brain-Computer Interface", "current": 0.005, "speed": 0.08, "potential": 0.3},
            ]
        tf = TechnologyForecaster()
        results = []
        for tech in technologies:
            curve = tf.s_curve_adoption(
                technology=tech["name"],
                current_adoption=tech["current"],
                market_potential=tech["potential"],
                innovation_speed=tech["speed"],
            )
            results.append({
                "name": tech["name"],
                "current": tech["current"],
                "speed": tech["speed"],
                "5yr_adoption": curve["predicted_adoption_5y"],
                "phase": curve["phase"],
                "years_to_50pct": curve["years_to_50pct"],
            })
        return {"technologies": results}

    @staticmethod
    def power_shift_forecast(
        regions: Optional[List[dict]] = None,
        horizon_years: int = 10,
    ) -> dict:
        if regions is None:
            regions = [
                {"name": "US", "gdp_share": 0.24, "military_share": 0.35, "tech_share": 0.40, "pop_share": 0.04, "growth_rate": 0.022, "stability": 0.8},
                {"name": "China", "gdp_share": 0.19, "military_share": 0.15, "tech_share": 0.28, "pop_share": 0.18, "growth_rate": 0.045, "stability": 0.6},
                {"name": "EU", "gdp_share": 0.17, "military_share": 0.12, "tech_share": 0.15, "pop_share": 0.06, "growth_rate": 0.015, "stability": 0.75},
                {"name": "India", "gdp_share": 0.04, "military_share": 0.04, "tech_share": 0.06, "pop_share": 0.18, "growth_rate": 0.065, "stability": 0.5},
                {"name": "Russia", "gdp_share": 0.02, "military_share": 0.05, "tech_share": 0.02, "pop_share": 0.02, "growth_rate": 0.01, "stability": 0.4},
            ]
        composite_indices = []
        for r in regions:
            w_gdp, w_mil, w_tech, w_stab = 0.3, 0.2, 0.25, 0.25
            current_composite = (
                w_gdp * r["gdp_share"] +
                w_mil * r["military_share"] +
                w_tech * r["tech_share"] +
                w_stab * r["stability"] * 0.3
            )
            growth_factor = 1 + r["growth_rate"] * horizon_years * (0.5 if r["stability"] < 0.5 else 1.0)
            future_composite = current_composite * growth_factor
            shift = future_composite - current_composite
            composite_indices.append({
                "name": r["name"],
                "current_composite": round(float(current_composite), 4),
                "projected_composite": round(float(future_composite), 4),
                "power_shift": round(float(shift), 4),
                "direction": "rising" if shift > 0.005 else "declining" if shift < -0.005 else "stable",
                "gdp_share": r["gdp_share"],
                "tech_share": r["tech_share"],
                "growth_rate": r["growth_rate"],
                "stability": r["stability"],
            })
        composite_indices.sort(key=lambda x: x["projected_composite"], reverse=True)
        return {
            "horizon_years": horizon_years,
            "regions": composite_indices,
            "biggest_shift": max(composite_indices, key=lambda x: abs(x["power_shift"])),
            "projected_leaders": [r["name"] for r in composite_indices[:3]],
        }

    @staticmethod
    def quantum_readiness(
        current_qubits: int = 1000,
        error_rate: float = 0.001,
        coherence_time_us: float = 100,
        gate_fidelity: float = 0.999,
        classical_speedup_threshold: float = 0.1,
        investment_billion: float = 5.0,
        talent_pool_size: int = 5000,
        regulatory_support: float = 0.5,
        years: int = 15,
    ) -> dict:
        qubit_growth_rate = 1.5
        error_reduction_rate = 0.85
        fidelity_improvement_rate = 0.995
        coherence_improvement_rate = 1.3
        yearly = []
        q = current_qubits
        err = error_rate
        coh = coherence_time_us
        fid = gate_fidelity
        for y in range(years + 1):
            fault_tolerant = q >= 1000 and err < 1e-4 and fid >= 0.9999
            practical_advantage = q >= 100 and err < 0.01 and fid >= 0.999 and coh >= 50
            research_capable = q >= 50 and err < 0.05 and fid >= 0.99
            if fault_tolerant:
                readiness = "fault_tolerant"
                score = min(1.0, 0.4 + 0.2 * np.log10(max(q, 1)) / 4 + 0.2 * (1 - err / 0.001) + 0.2 * (fid - 0.999) / 0.001)
            elif practical_advantage:
                readiness = "practical_advantage"
                score = min(0.8, 0.2 + 0.2 * np.log10(max(q, 1)) / 3 + 0.2 * (1 - err / 0.01) + 0.2 * (coh / 1000))
            elif research_capable:
                readiness = "research_capable"
                score = min(0.5, 0.1 + 0.1 * np.log10(max(q, 1)) / 2 + 0.15 * (1 - err / 0.05) + 0.15 * (fid - 0.99) / 0.01)
            else:
                readiness = "early_development"
                score = max(0.0, 0.05 + 0.05 * np.log10(max(q, 1)) + 0.05 * (1 - err))
            talent_factor = min(talent_pool_size * (1.1 ** y) / 50000, 1.0)
            investment_factor = min(investment_billion * (1.15 ** y) / 100, 1.0)
            regulatory_factor = regulatory_support
            composite = score * 0.5 + talent_factor * 0.15 + investment_factor * 0.2 + regulatory_factor * 0.15
            composite = min(float(composite), 1.0)
            yearly.append({
                "year": y,
                "logical_qubits_est": int(q * (1 - err) ** 10),
                "error_rate": float(err),
                "coherence_us": float(coh),
                "gate_fidelity": float(fid),
                "readiness_level": readiness,
                "composite_score": round(composite, 4),
            })
            q = int(q * qubit_growth_rate)
            err = err * error_reduction_rate
            coh = coh * coherence_improvement_rate
            fid = min(fid + (1 - fid) * (1 - fidelity_improvement_rate) * 0.1, 0.999999)
        milestones = []
        thresholds = [
            (0.2, "Quantum simulation of small molecules"),
            (0.35, "Quantum advantage for optimization problems"),
            (0.5, "Cryptographic relevance (RSA-2048 threat)"),
            (0.65, "Drug discovery acceleration"),
            (0.8, "Fault-tolerant general quantum computing"),
        ]
        for threshold, label in thresholds:
            for entry in yearly:
                if entry["composite_score"] >= threshold:
                    milestones.append({"year": entry["year"], "score": entry["composite_score"], "milestone": label})
                    break
        current_score = yearly[0]["composite_score"]
        final_score = yearly[-1]["composite_score"]
        applications = [
            {"name": "Cryptography", "impact": 0.9, "timeline_years": max(1, next((e["year"] for e in yearly if e["composite_score"] >= 0.5), years))},
            {"name": "Drug Discovery", "impact": 0.8, "timeline_years": max(1, next((e["year"] for e in yearly if e["composite_score"] >= 0.35), years))},
            {"name": "Financial Optimization", "impact": 0.7, "timeline_years": max(1, next((e["year"] for e in yearly if e["composite_score"] >= 0.35), years))},
            {"name": "Materials Science", "impact": 0.75, "timeline_years": max(1, next((e["year"] for e in yearly if e["composite_score"] >= 0.2), years))},
            {"name": "Machine Learning", "impact": 0.6, "timeline_years": max(1, next((e["year"] for e in yearly if e["composite_score"] >= 0.5), years))},
        ]
        risk_assessment = {
            "quantum_threat_to_crypto": round(float(current_score), 3),
            "years_to_rsa_threat": next((e["year"] for e in yearly if e["composite_score"] >= 0.5), None),
            "mitigation_urgency": "critical" if current_score > 0.3 else "high" if current_score > 0.15 else "moderate" if current_score > 0.05 else "low",
            "post_quantum_readiness": round(float(min(regulatory_support + investment_billion / 20, 1.0)), 3),
        }
        return {
            "current_qubits": current_qubits,
            "current_error_rate": error_rate,
            "current_coherence_us": coherence_time_us,
            "current_gate_fidelity": gate_fidelity,
            "investment_billion": investment_billion,
            "talent_pool_size": talent_pool_size,
            "regulatory_support": regulatory_support,
            "years": years,
            "current_readiness": yearly[0]["readiness_level"],
            "current_score": round(current_score, 4),
            "projected_score_5yr": round(yearly[min(5, years)]["composite_score"], 4),
            "projected_score_final": round(final_score, 4),
            "yearly_projection": yearly,
            "milestones": milestones,
            "applications": applications,
            "risk_assessment": risk_assessment,
        }