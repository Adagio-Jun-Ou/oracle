import pytest
import numpy as np
from oracle.domains.technology import TechnologyForecaster, TechPrediction


class TestTechPrediction:
    def test_defaults(self):
        pred = TechPrediction()
        assert pred.technology == ""
        assert pred.current_adoption == 0.0
        assert pred.disruption_score == 0.0
        assert pred.phase == ""

    def test_custom(self):
        pred = TechPrediction(technology="AI", current_adoption=0.15, phase="growth")
        assert pred.technology == "AI"
        assert pred.current_adoption == 0.15
        assert pred.phase == "growth"


class TestTechnologyForecasterInit:
    def test_name(self):
        tf = TechnologyForecaster()
        assert tf.name == "technology_forecaster"


class TestSCurveAdoption:
    def test_basic_adoption(self):
        tf = TechnologyForecaster()
        result = tf.s_curve_adoption(technology="AI", current_adoption=0.15, years=20)
        assert result["technology"] == "AI"
        assert result["current_adoption"] == 0.15
        assert "predicted_adoption_1y" in result
        assert "predicted_adoption_5y" in result
        assert "saturation" in result
        assert "years_to_50pct" in result
        assert "phase" in result
        assert "yearly_adoption" in result
        assert "yearly_velocity" in result
        assert len(result["yearly_adoption"]) == 21

    def test_adoption_increases(self):
        tf = TechnologyForecaster()
        result = tf.s_curve_adoption(current_adoption=0.1, years=20)
        yearly = result["yearly_adoption"]
        assert yearly[0] < yearly[-1]

    def test_growth_phase(self):
        tf = TechnologyForecaster()
        result = tf.s_curve_adoption(current_adoption=0.3, market_potential=1.0)
        assert result["phase"] in ["early", "growth", "maturity", "saturation"]

    def test_early_phase(self):
        tf = TechnologyForecaster()
        result = tf.s_curve_adoption(current_adoption=0.01, market_potential=1.0)
        assert result["phase"] == "early"

    def test_saturation_phase(self):
        tf = TechnologyForecaster()
        result = tf.s_curve_adoption(current_adoption=0.9, market_potential=1.0, innovation_speed=0.3)
        assert result["phase"] == "saturation"

    def test_custom_market_potential(self):
        tf = TechnologyForecaster()
        result = tf.s_curve_adoption(current_adoption=0.3, market_potential=0.8)
        assert result["saturation"] == 0.8

    def test_resistance_slows_adoption(self):
        tf = TechnologyForecaster()
        no_resist = tf.s_curve_adoption(current_adoption=0.1, years=10, resistance=0.0)
        high_resist = tf.s_curve_adoption(current_adoption=0.1, years=10, resistance=0.5)
        assert no_resist["predicted_adoption_5y"] >= high_resist["predicted_adoption_5y"]

    def test_velocity_starts_and_increases(self):
        tf = TechnologyForecaster()
        result = tf.s_curve_adoption(current_adoption=0.1, years=10)
        assert len(result["yearly_velocity"]) == 11

    def test_years_to_50pct(self):
        tf = TechnologyForecaster()
        result = tf.s_curve_adoption(current_adoption=0.05, innovation_speed=0.3, market_potential=1.0, years=50)
        assert result["years_to_50pct"] is not None


class TestAICapabilityForecast:
    def test_basic_forecast(self):
        tf = TechnologyForecaster()
        result = tf.ai_capability_forecast(current_capability=0.35, years=10)
        assert "current_capability" in result
        assert "predicted_capabilities" in result
        assert "years" in result
        assert "milestones_reached" in result
        assert "years_to_near_agi" in result
        assert len(result["predicted_capabilities"]) == 11
        assert len(result["years"]) == 11

    def test_capability_increases(self):
        tf = TechnologyForecaster()
        result = tf.ai_capability_forecast(current_capability=0.2, years=10)
        caps = result["predicted_capabilities"]
        assert caps[-1] > caps[0]

    def test_milestones(self):
        tf = TechnologyForecaster()
        result = tf.ai_capability_forecast(current_capability=0.2, compute_growth_rate=0.5, years=15)
        assert isinstance(result["milestones_reached"], list)

    def test_regulatory_slowdown(self):
        tf = TechnologyForecaster()
        fast = tf.ai_capability_forecast(regulatory_slowdown=0.0, years=10)
        slow = tf.ai_capability_forecast(regulatory_slowdown=0.3, years=10)
        assert fast["predicted_capabilities"][-1] >= slow["predicted_capabilities"][-1]

    def test_compute_growth_effect(self):
        tf = TechnologyForecaster()
        low = tf.ai_capability_forecast(compute_growth_rate=0.2, years=10)
        high = tf.ai_capability_forecast(compute_growth_rate=0.6, years=10)
        assert high["predicted_capabilities"][-1] >= low["predicted_capabilities"][-1]

    def test_capability_bounded(self):
        tf = TechnologyForecaster()
        result = tf.ai_capability_forecast(current_capability=0.5, years=20)
        for cap in result["predicted_capabilities"]:
            assert cap <= 1.0

    def test_custom_years(self):
        tf = TechnologyForecaster()
        result = tf.ai_capability_forecast(years=5)
        assert len(result["predicted_capabilities"]) == 6


class TestDisruptionAnalysis:
    def test_default_technologies(self):
        tf = TechnologyForecaster()
        result = tf.disruption_analysis()
        assert "technologies" in result
        assert "total_disruption_score" in result
        assert "top_disruptors" in result
        assert "displaced_sectors" in result
        assert len(result["technologies"]) == 6

    def test_disruption_scores(self):
        tf = TechnologyForecaster()
        result = tf.disruption_analysis()
        for tech in result["technologies"]:
            assert "disruption_score" in tech
            assert "years_to_impact" in tech
            assert "job_displacement_millions" in tech
            assert tech["disruption_score"] > 0

    def test_custom_industry(self):
        tf = TechnologyForecaster()
        result = tf.disruption_analysis(industry="healthcare", timeline_years=5)
        assert result["industry"] == "healthcare"
        assert result["timeline_years"] == 5

    def test_custom_technologies(self):
        tf = TechnologyForecaster()
        custom_techs = [
            {"name": "TestTech", "impact": 0.8, "probability": 0.7, "speed": 0.5, "displacement": 0.3}
        ]
        result = tf.disruption_analysis(technologies=custom_techs)
        assert len(result["technologies"]) == 1
        assert result["technologies"][0]["name"] == "TestTech"

    def test_sorted_by_disruption(self):
        tf = TechnologyForecaster()
        result = tf.disruption_analysis()
        scores = [t["disruption_score"] for t in result["technologies"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_disruptors(self):
        tf = TechnologyForecaster()
        result = tf.disruption_analysis()
        assert len(result["top_disruptors"]) == 3


class TestTechnologyAdoptionComparison:
    def test_default_comparison(self):
        tf = TechnologyForecaster()
        result = tf.technology_adoption_comparison()
        assert "technologies" in result
        assert len(result["technologies"]) == 7

    def test_comparison_structure(self):
        tf = TechnologyForecaster()
        result = tf.technology_adoption_comparison()
        for tech in result["technologies"]:
            assert "name" in tech
            assert "current" in tech
            assert "5yr_adoption" in tech
            assert "phase" in tech
            assert "years_to_50pct" in tech

    def test_custom_technologies(self):
        tf = TechnologyForecaster()
        custom = [
            {"name": "TestTech", "current": 0.5, "speed": 0.4, "potential": 0.9}
        ]
        result = tf.technology_adoption_comparison(technologies=custom)
        assert len(result["technologies"]) == 1
        assert result["technologies"][0]["name"] == "TestTech"


class TestPowerShiftForecast:
    def test_default_regions(self):
        tf = TechnologyForecaster()
        result = tf.power_shift_forecast()
        assert "regions" in result
        assert "biggest_shift" in result
        assert "projected_leaders" in result
        assert len(result["regions"]) == 5

    def test_region_structure(self):
        tf = TechnologyForecaster()
        result = tf.power_shift_forecast()
        for region in result["regions"]:
            assert "name" in region
            assert "current_composite" in region
            assert "projected_composite" in region
            assert "power_shift" in region
            assert "direction" in region

    def test_direction_values(self):
        tf = TechnologyForecaster()
        result = tf.power_shift_forecast()
        for region in result["regions"]:
            assert region["direction"] in ["rising", "declining", "stable"]

    def test_sorted_by_projected_composite(self):
        tf = TechnologyForecaster()
        result = tf.power_shift_forecast()
        compos = [r["projected_composite"] for r in result["regions"]]
        assert compos == sorted(compos, reverse=True)

    def test_custom_horizon(self):
        tf = TechnologyForecaster()
        result = tf.power_shift_forecast(horizon_years=20)
        assert result["horizon_years"] == 20

    def test_custom_regions(self):
        tf = TechnologyForecaster()
        custom = [
            {"name": "TestRegion", "gdp_share": 0.3, "military_share": 0.2, "tech_share": 0.25, "pop_share": 0.1, "growth_rate": 0.03, "stability": 0.7}
        ]
        result = tf.power_shift_forecast(regions=custom)
        assert len(result["regions"]) == 1
        assert result["regions"][0]["name"] == "TestRegion"

    def test_projected_leaders(self):
        tf = TechnologyForecaster()
        result = tf.power_shift_forecast()
        assert len(result["projected_leaders"]) == 3

    def test_adoption_zero_resistance(self):
        tf = TechnologyForecaster()
        result = tf.s_curve_adoption(technology="TestTech", current_adoption=0.01, innovation_speed=0.5, resistance=0.0, years=10)
        assert result["predicted_adoption_5y"] > 0.05

    def test_adoption_max_resistance(self):
        tf = TechnologyForecaster()
        result = tf.s_curve_adoption(technology="TestTech", current_adoption=0.01, innovation_speed=0.5, resistance=0.9, years=10)
        assert result["predicted_adoption_5y"] < 0.5

    def test_ai_forecast_zero_start(self):
        tf = TechnologyForecaster()
        result = tf.ai_capability_forecast(current_capability=0.0, compute_growth_rate=0.4, years=5)
        assert len(result["predicted_capabilities"]) == 6

    def test_ai_forecast_high_compute(self):
        tf = TechnologyForecaster()
        low = tf.ai_capability_forecast(current_capability=0.1, compute_growth_rate=0.2, years=5)
        high = tf.ai_capability_forecast(current_capability=0.1, compute_growth_rate=0.8, years=5)
        assert high["predicted_capabilities"][-1] > low["predicted_capabilities"][-1]

    def test_disruption_all_industries(self):
        tf = TechnologyForecaster()
        for industry in ["technology", "healthcare", "finance", "energy", "manufacturing"]:
            result = tf.disruption_analysis(industry=industry, timeline_years=10)
            assert result["industry"] == industry
            assert len(result["technologies"]) > 0

    def test_adoption_comparison_unique_phases(self):
        tf = TechnologyForecaster()
        result = tf.technology_adoption_comparison()
        phases = set(t["phase"] for t in result["technologies"])
        assert len(phases) >= 2

    def test_power_shift_declining_region(self):
        tf = TechnologyForecaster()
        result = tf.power_shift_forecast()
        directions = [r["direction"] for r in result["regions"]]
        assert "declining" in directions or "rising" in directions


class TestQuantumReadiness:
    def test_default_quantum(self):
        tf = TechnologyForecaster()
        result = tf.quantum_readiness()
        assert "current_qubits" in result
        assert "current_error_rate" in result
        assert "current_coherence_us" in result
        assert "current_gate_fidelity" in result
        assert "current_readiness" in result
        assert "current_score" in result
        assert "projected_score_5yr" in result
        assert "projected_score_final" in result
        assert "yearly_projection" in result
        assert "milestones" in result
        assert "applications" in result
        assert "risk_assessment" in result
        assert 0 <= result["current_score"] <= 1

    def test_yearly_projection_length(self):
        tf = TechnologyForecaster()
        result = tf.quantum_readiness(years=15)
        assert len(result["yearly_projection"]) == 16

    def test_readiness_levels(self):
        tf = TechnologyForecaster()
        result = tf.quantum_readiness()
        for entry in result["yearly_projection"]:
            assert entry["readiness_level"] in ["early_development", "research_capable", "practical_advantage", "fault_tolerant"]

    def test_score_increases_over_time(self):
        tf = TechnologyForecaster()
        result = tf.quantum_readiness(years=15)
        scores = [e["composite_score"] for e in result["yearly_projection"]]
        assert scores[-1] >= scores[0]

    def test_more_investment_higher_score(self):
        tf = TechnologyForecaster()
        low = tf.quantum_readiness(investment_billion=1.0, years=10)
        high = tf.quantum_readiness(investment_billion=20.0, years=10)
        assert high["projected_score_final"] >= low["projected_score_final"]

    def test_applications_structure(self):
        tf = TechnologyForecaster()
        result = tf.quantum_readiness()
        for app in result["applications"]:
            assert "name" in app
            assert "impact" in app
            assert "timeline_years" in app

    def test_risk_assessment(self):
        tf = TechnologyForecaster()
        result = tf.quantum_readiness()
        risk = result["risk_assessment"]
        assert "quantum_threat_to_crypto" in risk
        assert "years_to_rsa_threat" in risk
        assert "mitigation_urgency" in risk
        assert "post_quantum_readiness" in risk
        assert risk["mitigation_urgency"] in ["low", "moderate", "high", "critical"]

    def test_milestones_progress(self):
        tf = TechnologyForecaster()
        result = tf.quantum_readiness(years=20)
        if result["milestones"]:
            for m in result["milestones"]:
                assert "year" in m
                assert "score" in m
                assert "milestone" in m

    def test_custom_years(self):
        tf = TechnologyForecaster()
        result = tf.quantum_readiness(years=5)
        assert len(result["yearly_projection"]) == 6
        assert result["yearly_projection"][0]["year"] == 0
        assert result["yearly_projection"][-1]["year"] == 5