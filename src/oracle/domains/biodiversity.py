from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import math
import random


@dataclass
class BiodiversityState:
    habitat_loss_rate: float = 0.3
    species_extinction_rate: float = 0.4
    ecosystem_services_index: float = 0.55
    protected_area_coverage: float = 0.15
    deforestation_rate: float = 0.3
    ocean_acidification: float = 0.3
    invasive_species_impact: float = 0.25
    pollution_index: float = 0.35
    overexploitation: float = 0.3
    climate_impact_on_biodiversity: float = 0.35
    genetic_diversity_index: float = 0.5
    coral_reef_health: float = 0.5
    pollinator_decline: float = 0.4


class BiodiversityModel:
    def __init__(self):
        self.state: Optional[BiodiversityState] = None
        self.threat_categories = [
            "habitat_loss", "overexploitation", "pollution",
            "invasive_species", "climate_change", "disease",
        ]
        self.ecosystem_types = [
            "tropical_forest", "temperate_forest", "grassland",
            "freshwater", "marine", "wetland", "tundra", "dryland",
        ]

    def set_state(self, **kwargs) -> "BiodiversityModel":
        self.state = BiodiversityState(**kwargs)
        return self

    def assess_health(self, state: Optional[BiodiversityState] = None) -> Dict[str, Any]:
        s = state or self.state or BiodiversityState()
        biodiversity_index = (
            (1.0 - s.habitat_loss_rate) * 0.20
            + (1.0 - s.species_extinction_rate) * 0.20
            + s.ecosystem_services_index * 0.15
            + s.genetic_diversity_index * 0.10
            + (1.0 - s.deforestation_rate) * 0.10
            + (1.0 - s.ocean_acidification) * 0.10
            + (1.0 - s.pollution_index) * 0.08
            + (1.0 - s.overexploitation) * 0.07
        )
        biodiversity_index = max(0.0, min(1.0, biodiversity_index))
        threat_assessment = self._assess_threats(s)
        ecosystem_resilience = self._assess_ecosystem_resilience(s)
        tipping_points = self._identify_tipping_points(s)
        if biodiversity_index > 0.7:
            health_level = "good"
        elif biodiversity_index > 0.5:
            health_level = "moderate"
        elif biodiversity_index > 0.3:
            health_level = "poor"
        elif biodiversity_index > 0.15:
            health_level = "critical"
        else:
            health_level = "collapsed"
        recommendations = self._generate_recommendations(s, biodiversity_index, threat_assessment)
        return {
            "biodiversity_index": float(biodiversity_index),
            "health_level": health_level,
            "threat_assessment": threat_assessment,
            "ecosystem_resilience": ecosystem_resilience,
            "tipping_points": tipping_points,
            "n_tipping_points": len(tipping_points),
            "species_outlook": self._species_outlook(s, biodiversity_index),
            "ecosystem_services_at_risk": self._ecosystem_services_risk(s),
            "recommendations": recommendations,
        }

    def simulate_extinction_scenario(
        self,
        scenario: str = "business_as_usual",
        state: Optional[BiodiversityState] = None,
        years: int = 50,
    ) -> Dict[str, Any]:
        s = state or self.state or BiodiversityState()
        scenarios = {
            "business_as_usual": {"habitat_loss_accel": 0.003, "climate_worsening": 0.005, "extinction_accel": 0.004},
            "moderate_conservation": {"habitat_loss_accel": 0.001, "climate_worsening": 0.003, "extinction_accel": 0.002},
            "aggressive_protection": {"habitat_loss_accel": -0.002, "climate_worsening": 0.001, "extinction_accel": -0.001},
            "climate_crisis": {"habitat_loss_accel": 0.005, "climate_worsening": 0.008, "extinction_accel": 0.006},
            "mass_extinction": {"habitat_loss_accel": 0.008, "climate_worsening": 0.010, "extinction_accel": 0.008},
        }
        params = scenarios.get(scenario, scenarios["business_as_usual"])
        base_biodiversity = self.assess_health(s)["biodiversity_index"]
        timeline = []
        current_biodiversity = base_biodiversity
        current_extinction = s.species_extinction_rate
        current_habitat = s.habitat_loss_rate
        for year in range(0, years + 1, max(1, years // 10)):
            current_biodiversity = max(0.01, min(1.0,
                current_biodiversity
                - params["habitat_loss_accel"] * (1 + current_extinction * 0.5)
                - params["extinction_accel"] * 0.5
            ))
            current_extinction = min(1.0, current_extinction + params["extinction_accel"])
            current_habitat = min(1.0, max(0.0, current_habitat + params["habitat_loss_accel"]))
            species_lost_pct = min(95, (1.0 - current_biodiversity) * 80 + current_extinction * 15)
            ecosystem_collapse_pct = min(90, (1.0 - current_biodiversity) * 60 + current_habitat * 20)
            timeline.append({
                "year": year,
                "biodiversity_index": float(round(current_biodiversity, 4)),
                "extinction_rate": float(round(current_extinction, 4)),
                "habitat_integrity": float(round(1.0 - current_habitat, 4)),
                "species_lost_pct": float(round(species_lost_pct, 1)),
                "ecosystem_collapse_pct": float(round(ecosystem_collapse_pct, 1)),
            })
        final_biodiversity = current_biodiversity
        if final_biodiversity > 0.6:
            outcome = "stable"
        elif final_biodiversity > 0.4:
            outcome = "declining"
        elif final_biodiversity > 0.2:
            outcome = "severe_decline"
        else:
            outcome = "mass_extinction"
        return {
            "scenario": scenario,
            "years_projected": years,
            "final_biodiversity_index": float(round(final_biodiversity, 4)),
            "biodiversity_loss_pct": float(round((base_biodiversity - final_biodiversity) * 100, 1)),
            "outcome": outcome,
            "timeline": timeline,
            "species_at_risk_pct": float(round(min(95, (1.0 - final_biodiversity) * 70 + current_extinction * 20), 1)),
            "ecosystem_services_loss_pct": float(round(min(90, (1.0 - final_biodiversity) * 65), 1)),
            "economic_impact_gdp_pct": float(round(min(25, (1.0 - final_biodiversity) * 15 + current_habitat * 5), 1)),
        }

    def predict(self, years_ahead: int = 10, state: Optional[BiodiversityState] = None) -> Dict[str, Any]:
        s = state or self.state or BiodiversityState()
        degradation_trend = (
            s.habitat_loss_rate * 0.25
            + s.species_extinction_rate * 0.20
            + s.climate_impact_on_biodiversity * 0.20
            + s.deforestation_rate * 0.15
            + s.pollution_index * 0.10
            + s.overexploitation * 0.10
        )
        future_biodiversity = max(0.01, min(1.0, 1.0 - degradation_trend - years_ahead * 0.008))
        future_extinction = min(1.0, s.species_extinction_rate + years_ahead * 0.01)
        future_coral = max(0.01, s.coral_reef_health - years_ahead * 0.015 * (1 + s.ocean_acidification + s.climate_impact_on_biodiversity * 0.5))
        future_pollinator = max(0.01, s.pollinator_decline + years_ahead * 0.008 * s.climate_impact_on_biodiversity)
        future_ecosystem_services = max(0.01, s.ecosystem_services_index - years_ahead * 0.005 * degradation_trend)
        future_genetic = max(0.01, s.genetic_diversity_index - years_ahead * 0.006 * s.species_extinction_rate)
        protected_effectiveness = min(1.0, s.protected_area_coverage * 3 + 0.2)
        conservation_trajectory = min(1.0, protected_effectiveness * 0.4 + (1.0 - s.overexploitation) * 0.3 + (1.0 - s.deforestation_rate) * 0.3)
        return {
            "years_ahead": years_ahead,
            "future_biodiversity_index": float(future_biodiversity),
            "future_extinction_rate": float(future_extinction),
            "future_coral_health": float(future_coral),
            "future_pollinator_decline": float(future_pollinator),
            "future_ecosystem_services": float(future_ecosystem_services),
            "future_genetic_diversity": float(future_genetic),
            "conservation_trajectory": float(conservation_trajectory),
            "degradation_trend": float(degradation_trend),
            "mass_extinction_risk": float(min(1.0, degradation_trend * 1.5 + years_ahead * 0.005)),
            "ecosystem_collapse_probability": float(min(1.0, (1.0 - future_biodiversity) * 0.7 + future_extinction * 0.3)),
        }

    def _assess_threats(self, s: BiodiversityState) -> Dict[str, float]:
        return {
            "habitat_destruction": float(min(1.0, s.habitat_loss_rate * 0.7 + s.deforestation_rate * 0.3)),
            "climate_change": float(min(1.0, s.climate_impact_on_biodiversity * 0.8 + s.ocean_acidification * 0.2)),
            "overexploitation": float(min(1.0, s.overexploitation * 0.8 + s.pollution_index * 0.2)),
            "pollution": float(min(1.0, s.pollution_index * 0.7 + s.ocean_acidification * 0.3)),
            "invasive_species": float(min(1.0, s.invasive_species_impact * 0.8 + s.habitat_loss_rate * 0.2)),
            "species_loss": float(min(1.0, s.species_extinction_rate * 0.7 + s.genetic_diversity_index * 0.15 + (1.0 - s.genetic_diversity_index) * 0.15)),
        }

    def _assess_ecosystem_resilience(self, s: BiodiversityState) -> Dict[str, Any]:
        resilience = (
            s.ecosystem_services_index * 0.25
            + s.genetic_diversity_index * 0.20
            + (1.0 - s.habitat_loss_rate) * 0.20
            + (1.0 - s.climate_impact_on_biodiversity) * 0.15
            + s.protected_area_coverage * 0.10
            + s.coral_reef_health * 0.10
        )
        adaptive_capacity = min(1.0, s.genetic_diversity_index * 0.4 + (1.0 - s.species_extinction_rate) * 0.3 + s.ecosystem_services_index * 0.3)
        return {
            "overall_resilience": float(resilience),
            "adaptive_capacity": float(adaptive_capacity),
            "resistance": float(min(1.0, (1.0 - s.habitat_loss_rate) * 0.5 + (1.0 - s.overexploitation) * 0.3 + (1.0 - s.pollution_index) * 0.2)),
            "recovery_potential": float(min(1.0, s.protected_area_coverage * 0.4 + s.genetic_diversity_index * 0.3 + (1.0 - s.species_extinction_rate) * 0.3)),
        }

    def _identify_tipping_points(self, s: BiodiversityState) -> List[Dict[str, Any]]:
        tipping = []
        if s.coral_reef_health < 0.3:
            tipping.append({"name": "coral_reef_collapse", "probability": float(min(1.0, 1.0 - s.coral_reef_health + s.ocean_acidification * 0.3)), "impact": "catastrophic", "description": "Coral reef ecosystem collapse affecting 25% of marine species"})
        if s.deforestation_rate > 0.5:
            tipping.append({"name": "amazon_tipping_point", "probability": float(min(1.0, s.deforestation_rate * 0.7 + s.climate_impact_on_biodiversity * 0.3)), "impact": "catastrophic", "description": "Tropical forest dieback converting forest to savanna"})
        if s.pollinator_decline > 0.5:
            tipping.append({"name": "pollinator_collapse", "probability": float(min(1.0, s.pollinator_decline * 0.6 + s.climate_impact_on_biodiversity * 0.2 + s.habitat_loss_rate * 0.2)), "impact": "severe", "description": "Pollinator collapse threatening 75% of food crops"})
        if s.species_extinction_rate > 0.6:
            tipping.append({"name": "sixth_mass_extinction", "probability": float(min(1.0, s.species_extinction_rate * 0.8 + s.habitat_loss_rate * 0.2)), "impact": "catastrophic", "description": "Mass extinction event exceeding background rate by 1000x"})
        if s.ocean_acidification > 0.5:
            tipping.append({"name": "ocean_acidification_threshold", "probability": float(min(1.0, s.ocean_acidification * 0.7 + s.climate_impact_on_biodiversity * 0.3)), "impact": "severe", "description": "Ocean acidification reaching threshold for shell-forming organisms"})
        if s.genetic_diversity_index < 0.3:
            tipping.append({"name": "genetic_bottleneck", "probability": float(min(1.0, (1.0 - s.genetic_diversity_index) * 0.7 + s.species_extinction_rate * 0.3)), "impact": "moderate", "description": "Loss of genetic diversity reducing species adaptability"})
        return tipping[:6]

    def _species_outlook(self, s: BiodiversityState, biodiversity_index: float) -> str:
        if biodiversity_index > 0.7:
            return "stable"
        elif biodiversity_index > 0.5:
            return "declining"
        elif biodiversity_index > 0.3:
            return "rapidly_declining"
        elif biodiversity_index > 0.15:
            return "severe_decline"
        else:
            return "imminent_collapse"

    def _ecosystem_services_risk(self, s: BiodiversityState) -> Dict[str, float]:
        return {
            "pollination": float(min(1.0, 1.0 - s.pollinator_decline * 0.7 + s.habitat_loss_rate * 0.3)),
            "water_purification": float(min(1.0, 1.0 - s.pollution_index * 0.6 - s.deforestation_rate * 0.4)),
            "carbon_sequestration": float(min(1.0, 1.0 - s.deforestation_rate * 0.7 - s.habitat_loss_rate * 0.3)),
            "coastal_protection": float(min(1.0, s.coral_reef_health * 0.7 + (1.0 - s.habitat_loss_rate) * 0.3)),
            "soil_fertility": float(min(1.0, 1.0 - s.deforestation_rate * 0.5 - s.pollution_index * 0.3 - s.overexploitation * 0.2)),
        }

    def _generate_recommendations(self, s: BiodiversityState, biodiversity_index: float, threat_assessment: Dict[str, float]) -> List[str]:
        recs = []
        if s.habitat_loss_rate > 0.4:
            recs.append("Expand protected areas to at least 30% of land and sea (CBD 30x30 target)")
        if s.deforestation_rate > 0.4:
            recs.append("Implement zero-deforestation supply chain commitments and verification systems")
        if s.species_extinction_rate > 0.5:
            recs.append("Establish endangered species recovery programs with captive breeding and habitat restoration")
        if s.pollinator_decline > 0.4:
            recs.append("Restrict neonicotinoid pesticides and establish pollinator corridors in agricultural landscapes")
        if s.coral_reef_health < 0.4:
            recs.append("Reduce local stressors on coral reefs (overfishing, pollution) and establish marine protected areas")
        if s.pollution_index > 0.4:
            recs.append("Implement circular economy policies and regulate industrial discharge standards")
        if s.climate_impact_on_biodiversity > 0.4:
            recs.append("Establish climate refugia and assisted migration programs for climate-vulnerable species")
        if s.overexploitation > 0.4:
            recs.append("Enforce science-based harvest limits and establish sustainable use certification")
        if s.ocean_acidification > 0.4:
            recs.append("Accelerate CO2 emission reductions and protect blue carbon ecosystems (mangroves, seagrass)")
        if s.genetic_diversity_index < 0.4:
            recs.append("Support seed banks and genetic resource conservation programs")
        if not recs:
            recs.append("Maintain current conservation efforts and strengthen monitoring systems")
        return recs[:8]