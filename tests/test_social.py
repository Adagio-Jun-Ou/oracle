import pytest
import numpy as np
from oracle.domains.social import SocialDynamicsModel, SocialPrediction


class TestSocialPrediction:
    def test_defaults(self):
        pred = SocialPrediction()
        assert pred.unrest_probability == 0.0
        assert pred.polarization_index == 0.0
        assert pred.migration_pressure == 0.0
        assert pred.inequality_gini == 0.0
        assert pred.youth_bulge == 0.0
        assert pred.regime_stability == 0.0
        assert pred.social_cohesion == 0.0

    def test_custom(self):
        pred = SocialPrediction(unrest_probability=0.7, polarization_index=0.6)
        assert pred.unrest_probability == 0.7
        assert pred.polarization_index == 0.6


class TestSocialDynamicsModelInit:
    def test_name(self):
        model = SocialDynamicsModel()
        assert model.name == "social_dynamics_model"


class TestUnrestPrediction:
    def test_default_unrest(self):
        sdm = SocialDynamicsModel()
        result = sdm.unrest_prediction()
        assert "unrest_probability" in result
        assert "unrest_level" in result
        assert "likely_triggers" in result
        assert "economic_distress" in result
        assert "political_risk" in result
        assert "mobilization_factor" in result
        assert "social_cohesion" in result
        assert "indicators" in result
        assert 0 <= result["unrest_probability"] <= 1

    def test_unrest_levels(self):
        sdm = SocialDynamicsModel()
        low = sdm.unrest_prediction(gini_coefficient=0.25, unemployment_rate=0.03, political_instability=0.1, youth_unemployment=0.05)
        high = sdm.unrest_prediction(gini_coefficient=0.6, unemployment_rate=0.15, political_instability=0.8, youth_unemployment=0.3)
        assert high["unrest_probability"] > low["unrest_probability"]
        assert low["unrest_level"] in ["minimal", "low", "moderate", "high", "critical"]
        assert high["unrest_level"] in ["low", "moderate", "high", "critical"]

    def test_unrest_severity_mapping(self):
        sdm = SocialDynamicsModel()
        critical = sdm.unrest_prediction(gini_coefficient=0.7, unemployment_rate=0.25, political_instability=0.9, youth_unemployment=0.5, regime_repression=0.8)
        assert critical["unrest_level"] in ["high", "critical"]

    def test_minimal_unrest(self):
        sdm = SocialDynamicsModel()
        result = sdm.unrest_prediction(gini_coefficient=0.2, unemployment_rate=0.02, political_instability=0.05, youth_unemployment=0.03, food_price_index=0.5, social_media_penetration=0.2, regime_repression=0.05, education_index=0.95, press_freedom=0.95, gdp_per_capita=80000)
        assert result["unrest_probability"] < 0.5

    def test_indicators_returned(self):
        sdm = SocialDynamicsModel()
        result = sdm.unrest_prediction()
        ind = result["indicators"]
        assert "gini" in ind
        assert "unemployment" in ind
        assert "inflation" in ind
        assert "youth_unemployment" in ind
        assert "food_price_index" in ind
        assert "political_instability" in ind
        assert "regime_repression" in ind
        assert "social_media" in ind
        assert "ethnic_fractionalization" in ind

    def test_economic_distress_components(self):
        sdm = SocialDynamicsModel()
        result = sdm.unrest_prediction()
        assert result["economic_distress"] > 0
        assert result["political_risk"] > 0
        assert result["mobilization_factor"] > 0

    def test_custom_params(self):
        sdm = SocialDynamicsModel()
        result = sdm.unrest_prediction(gini_coefficient=0.5, unemployment_rate=0.1)
        assert result["indicators"]["gini"] == 0.5
        assert result["indicators"]["unemployment"] == 0.1


class TestMigrationPrediction:
    def test_default_migration(self):
        sdm = SocialDynamicsModel()
        result = sdm.migration_prediction()
        assert "migration_pressure" in result
        assert "migration_level" in result
        assert "push_factors" in result
        assert "pull_factors" in result
        assert "wage_gap_log" in result
        assert "projected_flow_5yr" in result
        assert "flow_as_pct_origin" in result
        assert "destination_impact" in result
        assert 0 <= result["migration_pressure"] <= 1

    def test_migration_levels(self):
        sdm = SocialDynamicsModel()
        result = sdm.migration_prediction()
        assert result["migration_level"] in ["low", "moderate", "high", "mass_exodus"]

    def test_high_conflict_migration(self):
        sdm = SocialDynamicsModel()
        low = sdm.migration_prediction(origin_conflict=0.1, origin_gdp_per_capita=20000)
        high = sdm.migration_prediction(origin_conflict=0.9, origin_gdp_per_capita=1000)
        assert high["migration_pressure"] > low["migration_pressure"]
        assert high["push_factors"] > low["push_factors"]

    def test_destination_impact(self):
        sdm = SocialDynamicsModel()
        result = sdm.migration_prediction()
        impact = result["destination_impact"]
        assert "labor_market" in impact
        assert "housing_pressure" in impact
        assert "cultural_tension" in impact
        assert "fiscal_burden" in impact
        assert "gdp_boost" in impact

    def test_wage_gap_effect(self):
        sdm = SocialDynamicsModel()
        small_gap = sdm.migration_prediction(origin_gdp_per_capita=20000, destination_gdp_per_capita=30000)
        large_gap = sdm.migration_prediction(origin_gdp_per_capita=1000, destination_gdp_per_capita=60000)
        assert large_gap["migration_pressure"] > small_gap["migration_pressure"]

    def test_projected_flow(self):
        sdm = SocialDynamicsModel()
        result = sdm.migration_prediction()
        assert result["projected_flow_5yr"] > 0
        assert result["flow_as_pct_origin"] > 0

    def test_custom_params(self):
        sdm = SocialDynamicsModel()
        result = sdm.migration_prediction(climate_risk=0.5, border_openness=0.2)
        assert isinstance(result["migration_pressure"], float)


class TestPolarizationIndex:
    def test_default_polarization(self):
        sdm = SocialDynamicsModel()
        result = sdm.polarization_index()
        assert "polarization_index" in result
        assert "fragmentation_index" in result
        assert "state" in result
        assert "primary_risks" in result
        assert "components" in result
        assert 0 <= result["polarization_index"] <= 1
        assert 0 <= result["fragmentation_index"] <= 1

    def test_polarization_states(self):
        sdm = SocialDynamicsModel()
        result = sdm.polarization_index()
        assert result["state"] in ["deeply_divided", "polarized", "moderately_polarized", "relatively_unified"]

    def test_high_polarization(self):
        sdm = SocialDynamicsModel()
        result = sdm.polarization_index(income_inequality=0.6, political_polarization=0.9, media_fragmentation=0.9, trust_in_institutions=0.1, ideological_distance=0.8, echo_chamber_effect=0.9)
        assert result["polarization_index"] > 0.5
        assert result["state"] in ["deeply_divided", "polarized"]

    def test_low_polarization(self):
        sdm = SocialDynamicsModel()
        result = sdm.polarization_index(income_inequality=0.2, political_polarization=0.2, media_fragmentation=0.2, trust_in_institutions=0.9, ideological_distance=0.2, echo_chamber_effect=0.2)
        assert result["polarization_index"] < 0.4
        assert result["state"] in ["moderately_polarized", "relatively_unified"]

    def test_components(self):
        sdm = SocialDynamicsModel()
        result = sdm.polarization_index()
        comp = result["components"]
        assert "income_inequality" in comp
        assert "political_polarization" in comp
        assert "media_fragmentation" in comp
        assert "trust_deficit" in comp
        assert "ideological_distance" in comp
        assert "echo_chamber" in comp

    def test_risks_are_lists(self):
        sdm = SocialDynamicsModel()
        result = sdm.polarization_index()
        assert isinstance(result["primary_risks"], list)


class TestWealthPowerShift:
    def test_default_shift(self):
        sdm = SocialDynamicsModel()
        result = sdm.wealth_power_shift()
        assert "horizon" in result
        assert "regions" in result
        assert "biggest_gainer" in result
        assert "biggest_loser" in result
        assert len(result["regions"]) == 7

    def test_region_structure(self):
        sdm = SocialDynamicsModel()
        result = sdm.wealth_power_shift()
        for r in result["regions"]:
            assert "name" in r
            assert "current_influence" in r
            assert "projected_influence" in r
            assert "shift" in r
            assert "direction" in r

    def test_directions(self):
        sdm = SocialDynamicsModel()
        result = sdm.wealth_power_shift()
        directions = [r["direction"] for r in result["regions"]]
        assert all(d in ["rising", "declining", "stable"] for d in directions)

    def test_custom_horizon(self):
        sdm = SocialDynamicsModel()
        result = sdm.wealth_power_shift(horizon=2050)
        assert result["horizon"] == 2050

    def test_custom_regions(self):
        sdm = SocialDynamicsModel()
        custom = [
            {"name": "TestRegion", "wealth": 0.3, "tech": 0.2, "pop": 0.1, "growth": 0.04, "stability": 0.7}
        ]
        result = sdm.wealth_power_shift(regions=custom)
        assert len(result["regions"]) == 1
        assert result["regions"][0]["name"] == "TestRegion"

    def test_biggest_gainer_and_loser(self):
        sdm = SocialDynamicsModel()
        result = sdm.wealth_power_shift()
        assert result["biggest_gainer"]["shift"] >= 0
        assert result["biggest_loser"]["shift"] <= result["biggest_gainer"]["shift"]

    def test_sorted_by_projected_influence(self):
        sdm = SocialDynamicsModel()
        result = sdm.wealth_power_shift()
        influences = [r["projected_influence"] for r in result["regions"]]
        assert influences == sorted(influences, reverse=True)


class TestCivilUnrestScenario:
    def test_default_scenario(self):
        sdm = SocialDynamicsModel()
        result = sdm.civil_unrest_scenario()
        assert "base_unrest" in result
        assert "scenarios" in result
        assert "trigger_events" in result
        assert "recommendation" in result
        assert result["base_unrest"] == 0.3

    def test_scenarios_structure(self):
        sdm = SocialDynamicsModel()
        result = sdm.civil_unrest_scenario()
        assert "baseline" in result["scenarios"]
        assert "1_event_cascade" in result["scenarios"]
        assert "2_event_cascade" in result["scenarios"]
        assert "3_event_cascade" in result["scenarios"]
        assert "worst_case" in result["scenarios"]

    def test_escalation(self):
        sdm = SocialDynamicsModel()
        result = sdm.civil_unrest_scenario()
        assert result["scenarios"]["baseline"] <= result["scenarios"]["worst_case"]

    def test_trigger_events(self):
        sdm = SocialDynamicsModel()
        result = sdm.civil_unrest_scenario()
        for event in result["trigger_events"]:
            assert "event" in event
            assert "probability" in event
            assert "impact" in event
            assert 0 <= event["probability"] <= 1
            assert 0 <= event["impact"] <= 1

    def test_custom_base_unrest(self):
        sdm = SocialDynamicsModel()
        low = sdm.civil_unrest_scenario(base_unrest=0.1)
        high = sdm.civil_unrest_scenario(base_unrest=0.7)
        assert high["scenarios"]["baseline"] > low["scenarios"]["baseline"]

    def test_custom_triggers(self):
        sdm = SocialDynamicsModel()
        custom = [
            {"event": "test_event", "probability": 0.5, "impact": 0.6}
        ]
        result = sdm.civil_unrest_scenario(base_unrest=0.3, trigger_events=custom)
        assert len(result["trigger_events"]) == 1
        assert result["trigger_events"][0]["event"] == "test_event"

    def test_recommendation(self):
        sdm = SocialDynamicsModel()
        low = sdm.civil_unrest_scenario(base_unrest=0.1)
        high = sdm.civil_unrest_scenario(base_unrest=0.5)
        assert isinstance(low["recommendation"], str)
        assert isinstance(high["recommendation"], str)

    def test_worst_case_bounded(self):
        sdm = SocialDynamicsModel()
        result = sdm.civil_unrest_scenario(base_unrest=0.9)
        assert result["scenarios"]["worst_case"] <= 0.98

    def test_unrest_extreme_gini(self):
        sdm = SocialDynamicsModel()
        result = sdm.unrest_prediction(gini_coefficient=0.9, unemployment_rate=0.3, political_instability=0.8)
        low_gini = sdm.unrest_prediction(gini_coefficient=0.2, unemployment_rate=0.02, political_instability=0.1)
        assert result["unrest_probability"] > low_gini["unrest_probability"]

    def test_unrest_low_risk(self):
        sdm = SocialDynamicsModel()
        result = sdm.unrest_prediction(gini_coefficient=0.2, unemployment_rate=0.02, political_instability=0.1)
        assert result["unrest_probability"] < 0.7

    def test_migration_with_colonial_tie(self):
        sdm = SocialDynamicsModel()
        result = sdm.migration_prediction(
            origin_gdp_per_capita=3000, destination_gdp_per_capita=50000,
            colonial_tie=0.8, language_similarity=0.9
        )
        assert result["migration_pressure"] > 0

    def test_polarization_echo_chamber(self):
        sdm = SocialDynamicsModel()
        result = sdm.polarization_index(
            income_inequality=0.4, political_polarization=0.5,
            echo_chamber_effect=0.9
        )
        assert result["polarization_index"] > 0.3

    def test_wealth_shift_horizon_range(self):
        sdm = SocialDynamicsModel()
        near = sdm.wealth_power_shift(horizon=2030)
        far = sdm.wealth_power_shift(horizon=2050)
        assert len(near["regions"]) == len(far["regions"])

    def test_civil_unrest_zero_base(self):
        sdm = SocialDynamicsModel()
        result = sdm.civil_unrest_scenario(base_unrest=0.0)
        assert result["scenarios"]["baseline"] <= 0.1

class TestCivilUnrestCascade:
    def test_default_cascade(self):
        sdm = SocialDynamicsModel()
        result = sdm.civil_unrest_cascade()
        assert "n_groups" in result
        assert "base_activation" in result
        assert "connectivity" in result
        assert "threshold" in result
        assert "n_simulations" in result
        assert "avg_activation_frac" in result
        assert "avg_cascade_size" in result
        assert "full_cascade_probability" in result
        assert "tipping_point_indicator" in result
        assert "group_risk_ranking" in result
        assert "cascade_risk_level" in result
        assert "intervention_points" in result
        assert 0 <= result["avg_activation_frac"] <= 1
        assert 0 <= result["full_cascade_probability"] <= 1

    def test_cascade_risk_levels(self):
        sdm = SocialDynamicsModel()
        result = sdm.civil_unrest_cascade()
        assert result["cascade_risk_level"] in ["low", "moderate", "high", "critical"]

    def test_group_risk_ranking(self):
        sdm = SocialDynamicsModel()
        result = sdm.civil_unrest_cascade()
        assert len(result["group_risk_ranking"]) > 0
        for g in result["group_risk_ranking"]:
            assert "group" in g
            assert "connectivity_degree" in g
            assert "risk_contribution" in g

    def test_high_connectivity_more_cascade(self):
        sdm = SocialDynamicsModel()
        low = sdm.civil_unrest_cascade(connectivity=0.1, n_simulations=200)
        high = sdm.civil_unrest_cascade(connectivity=0.8, n_simulations=200)
        assert high["avg_activation_frac"] >= low["avg_activation_frac"]

    def test_high_activation_more_cascade(self):
        sdm = SocialDynamicsModel()
        low = sdm.civil_unrest_cascade(base_activation=0.05, n_simulations=200)
        high = sdm.civil_unrest_cascade(base_activation=0.6, n_simulations=200)
        assert high["avg_activation_frac"] >= low["avg_activation_frac"]

    def test_custom_groups(self):
        sdm = SocialDynamicsModel()
        result = sdm.civil_unrest_cascade(n_groups=5, n_simulations=100)
        assert result["n_groups"] == 5

    def test_intervention_points(self):
        sdm = SocialDynamicsModel()
        result = sdm.civil_unrest_cascade()
        assert len(result["intervention_points"]) > 0

    def test_reproducibility(self):
        sdm = SocialDynamicsModel()
        r1 = sdm.civil_unrest_cascade(seed=42, n_simulations=100)
        r2 = sdm.civil_unrest_cascade(seed=42, n_simulations=100)
        assert r1["avg_activation_frac"] == r2["avg_activation_frac"]

    def test_tipping_point(self):
        sdm = SocialDynamicsModel()
        result = sdm.civil_unrest_cascade(threshold_spread=0.3)
        assert result["tipping_point_indicator"] == 0.7


    def test_unrest_all_factors_contribute(self):
        sdm = SocialDynamicsModel()
        result = sdm.unrest_prediction(
            gini_coefficient=0.5, unemployment_rate=0.1,
            inflation_rate=0.05, youth_unemployment=0.2,
            food_price_index=1.5, political_instability=0.4,
            social_media_penetration=0.8, regime_repression=0.5,
            ethnic_fractionalization=0.6, urbanization_rate=0.7,
            gdp_per_capita=8000, education_index=0.5, press_freedom=0.3
        )
        assert result["economic_distress"] > 0
        assert result["political_risk"] > 0
        assert result["mobilization_factor"] > 0