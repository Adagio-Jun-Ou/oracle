from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math


@dataclass
class WaterState:
    freshwater_availability_m3_per_capita: float = 5000.0
    water_stress_index: float = 0.4
    groundwater_depletion_rate: float = 0.02
    drought_probability: float = 0.2
    flood_risk: float = 0.25
    water_quality_index: float = 0.65
    irrigation_efficiency: float = 0.55
    desalination_capacity: float = 0.05
    transboundary_dependency: float = 0.2
    industrial_water_use_pct: float = 0.22
    agricultural_water_use_pct: float = 0.70
    climate_impact_factor: float = 0.3
    population_pressure: float = 0.4
    infrastructure_age: float = 0.5


class WaterSecurityModel:
    def __init__(self):
        self.state: Optional[WaterState] = None

    def set_state(self, **kwargs) -> "WaterSecurityModel":
        self.state = WaterState(**kwargs)
        return self

    def assess_security(self, state: Optional[WaterState] = None) -> Dict[str, Any]:
        s = state or self.state or WaterState()
        scarcity_score = 1.0 - min(1.0, s.freshwater_availability_m3_per_capita / 10000.0)
        stress_composite = (
            s.water_stress_index * 0.25
            + scarcity_score * 0.20
            + s.groundwater_depletion_rate * 0.15
            + s.drought_probability * 0.15
            + s.flood_risk * 0.10
            + (1.0 - s.water_quality_index) * 0.10
            + s.transboundary_dependency * 0.05
        )
        vuln_composite = (
            (1.0 - s.irrigation_efficiency) * 0.20
            + s.population_pressure * 0.20
            + s.climate_impact_factor * 0.20
            + s.infrastructure_age * 0.15
            + (1.0 - s.desalination_capacity) * 0.15
            + s.industrial_water_use_pct * 0.10
        )
        overall_risk = min(1.0, stress_composite * 0.6 + vuln_composite * 0.4)
        if overall_risk > 0.7:
            risk_level = "critical"
        elif overall_risk > 0.5:
            risk_level = "high"
        elif overall_risk > 0.3:
            risk_level = "moderate"
        elif overall_risk > 0.15:
            risk_level = "low"
        else:
            risk_level = "minimal"
        resilience = min(1.0, s.irrigation_efficiency * 0.3 + s.desalination_capacity * 0.2 + s.water_quality_index * 0.3 + (1.0 - s.infrastructure_age) * 0.2)
        scarcity_types = []
        if s.freshwater_availability_m3_per_capita < 1000:
            scarcity_types.append("absolute_scarcity")
        elif s.freshwater_availability_m3_per_capita < 1700:
            scarcity_types.append("water_stress")
        if s.drought_probability > 0.3:
            scarcity_types.append("drought_prone")
        if s.groundwater_depletion_rate > 0.03:
            scarcity_types.append("groundwater_crisis")
        recommendations = []
        if s.water_stress_index > 0.4:
            recommendations.append("Implement demand-side water management: pricing reform, efficiency standards, leakage reduction")
        if s.irrigation_efficiency < 0.6:
            recommendations.append("Deploy precision irrigation (drip/trickle) to reduce agricultural water waste")
        if s.desalination_capacity < 0.1:
            recommendations.append("Invest in desalination infrastructure for coastal population centers")
        if s.groundwater_depletion_rate > 0.02:
            recommendations.append("Establish groundwater monitoring and managed aquifer recharge programs")
        if s.water_quality_index < 0.6:
            recommendations.append("Upgrade water treatment infrastructure and enforce discharge regulations")
        if s.flood_risk > 0.3:
            recommendations.append("Develop green infrastructure and early warning systems for flood management")
        if s.transboundary_dependency > 0.3:
            recommendations.append("Negotiate transboundary water cooperation agreements and data-sharing protocols")
        if s.climate_impact_factor > 0.4:
            recommendations.append("Integrate climate adaptation into water resource planning: drought reserves, flood resilience")
        return {
            "overall_risk": float(overall_risk),
            "risk_level": risk_level,
            "water_stress_composite": float(stress_composite),
            "vulnerability_composite": float(vuln_composite),
            "resilience_score": float(resilience),
            "scarcity_types": scarcity_types,
            "freshwater_per_capita": s.freshwater_availability_m3_per_capita,
            "per_capita_status": "absolute_scarcity" if s.freshwater_availability_m3_per_capita < 1000 else "water_stress" if s.freshwater_availability_m3_per_capita < 1700 else "sufficient",
            "components": {
                "scarcity": float(scarcity_score),
                "water_stress": float(s.water_stress_index),
                "groundwater_depletion": float(s.groundwater_depletion_rate),
                "drought_risk": float(s.drought_probability),
                "flood_risk": float(s.flood_risk),
                "quality deficit": float(1.0 - s.water_quality_index),
                "transboundary_risk": float(s.transboundary_dependency),
            },
            "recommendations": recommendations,
        }

    def drought_scenario(
        self,
        severity: float = 0.5,
        duration_months: int = 12,
        state: Optional[WaterState] = None,
    ) -> Dict[str, Any]:
        s = state or self.state or WaterState()
        base_drought = s.drought_probability
        effective_drought_prob = min(1.0, base_drought + severity * 0.5)
        water_reduction = min(0.8, severity * (1.0 - s.irrigation_efficiency) * 0.6 + severity * 0.2)
        agricultural_impact_pct = min(0.9, water_reduction * 0.7 * (s.agricultural_water_use_pct / 0.7))
        industrial_impact_pct = min(0.6, water_reduction * 0.3 * (s.industrial_water_use_pct / 0.22))
        gdp_impact_pct = agricultural_impact_pct * 0.04 + industrial_impact_pct * 0.02
        population_affected_m = round(s.population_pressure * water_reduction * 50, 1)
        recovery_months = max(1, int(duration_months * (0.5 + s.infrastructure_age * 0.5) * (1.1 - s.desalination_capacity * 0.5)))
        phases = []
        total_months = duration_months + recovery_months
        for month in range(1, min(total_months, 24) + 1):
            if month <= duration_months:
                phase_progress = min(1.0, month / (duration_months * 0.3))
                water_available = max(0.1, 1.0 - water_reduction * phase_progress)
            else:
                recovery_progress = (month - duration_months) / max(1, recovery_months)
                water_available = min(1.0, (1.0 - water_reduction) + water_reduction * recovery_progress)
            phases.append({
                "month": month,
                "water_available_pct": float(min(1.0, water_available)),
                "phase": "drought" if month <= duration_months else "recovery",
            })
        return {
            "severity": float(severity),
            "duration_months": duration_months,
            "effective_drought_probability": float(effective_drought_prob),
            "water_reduction_pct": float(water_reduction),
            "agricultural_impact_pct": float(agricultural_impact_pct),
            "industrial_impact_pct": float(industrial_impact_pct),
            "gdp_impact_pct": float(gdp_impact_pct),
            "population_affected_millions": population_affected_m,
            "recovery_months": recovery_months,
            "timeline": phases,
        }

    def predict(self, years_ahead: int = 10, state: Optional[WaterState] = None) -> Dict[str, Any]:
        s = state or self.state or WaterState()
        pop_growth_pressure = min(1.0, s.population_pressure + years_ahead * 0.01)
        climate_worsening = min(1.0, s.climate_impact_factor + years_ahead * 0.02)
        freshwater_decline = max(0, s.freshwater_availability_m3_per_capita - years_ahead * (s.groundwater_depletion_rate * 1000 + s.population_pressure * 50))
        future_stress = min(1.0, s.water_stress_index + pop_growth_pressure * 0.15 + climate_worsening * 0.1)
        future_drought = min(1.0, s.drought_probability + climate_worsening * 0.15)
        future_flood = min(1.0, s.flood_risk + climate_worsening * 0.1)
        desalination_growth = min(1.0, s.desalination_capacity + years_ahead * 0.005)
        efficiency_gain = min(1.0, s.irrigation_efficiency + years_ahead * 0.01)
        future_quality = min(1.0, s.water_quality_index + years_ahead * 0.005 - climate_worsening * 0.02)
        future_quality = max(0, future_quality)
        future_risk = min(1.0, future_stress * 0.3 + (1.0 - freshwater_decline / max(1, s.freshwater_availability_m3_per_capita)) * 0.2 + future_drought * 0.2 + (1.0 - future_quality) * 0.15 + (1.0 - efficiency_gain) * 0.15)
        return {
            "years_ahead": years_ahead,
            "future_risk": float(future_risk),
            "freshwater_per_capita_projected": float(max(0, freshwater_decline)),
            "water_stress_projected": float(future_stress),
            "drought_probability_projected": float(future_drought),
            "flood_risk_projected": float(future_flood),
            "desalination_capacity_projected": float(desalination_growth),
            "irrigation_efficiency_projected": float(efficiency_gain),
            "water_quality_projected": float(future_quality),
        }