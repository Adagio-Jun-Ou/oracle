import pytest
from oracle.domains.fate import (
    FateEngine, FateState, PersonalFateState,
    CIVILIZATIONAL_TRAJECTORIES, THERMODYNAMIC_CONSTRAINTS,
    RESOURCE_DEPLETION_CURVES, DEMOGRAPHIC_DESTINIES,
    BIOLOGICAL_DESTINIES, COSMOLOGICAL_TIMELINE,
    PERSONAL_DETERMINANTS, LIFE_TRAJECTORY_ARCHETYPES,
    MODIFIABLE_FACTORS, UNMODIFIABLE_FACTORS,
)


@pytest.fixture
def engine():
    return FateEngine(seed=42)


def test_fate_state_defaults():
    state = FateState()
    assert state.human_population_bn == 8.1
    assert state.energy_consumption_tw == 18.0
    assert state.co2_ppm == 420.0
    assert state.ai_capability_index == 0.15


def test_personal_fate_state_defaults():
    state = PersonalFateState()
    assert state.age == 35.0
    assert state.genetic_risk_score == 0.3
    assert state.socioeconomic_origin == 0.5
    assert state.habit_strength == 0.5


def test_civilizational_trajectory_structure():
    assert len(CIVILIZATIONAL_TRAJECTORIES) >= 5
    for tid, tdef in CIVILIZATIONAL_TRAJECTORIES.items():
        assert "name" in tdef
        assert "description" in tdef
        assert "probability_per_year" in tdef
        assert "domains" in tdef


def test_thermodynamic_constraints_structure():
    assert len(THERMODYNAMIC_CONSTRAINTS) >= 5
    for cid, cdef in THERMODYNAMIC_CONSTRAINTS.items():
        assert "name" in cdef
        assert "certainty" in cdef
        assert cdef["certainty"] >= 0
        assert cdef["certainty"] <= 1


def test_resource_depletion_curves_structure():
    assert len(RESOURCE_DEPLETION_CURVES) >= 8
    for rid, rdef in RESOURCE_DEPLETION_CURVES.items():
        assert "peak_year" in rdef
        assert "decline_rate" in rdef
        assert "remaining_pct" in rdef
        assert "substitute_readiness" in rdef


def test_cosmological_timeline_structure():
    assert len(COSMOLOGICAL_TIMELINE) >= 5
    for event in COSMOLOGICAL_TIMELINE:
        assert "event" in event
        assert "probability" in event
        assert event["probability"] >= 0
        assert event["probability"] <= 1


def test_assess_civilizational_trajectory_default(engine):
    result = engine.assess_civilizational_trajectory()
    assert "trajectory_score" in result
    assert "trajectory_label" in result
    assert "kardashev_scale_current" in result
    assert "energy_progress_to_type_i_pct" in result
    assert "existential_risk_composite" in result
    assert "trajectories" in result
    assert result["trajectory_label"] in ("ascending", "plateau", "declining", "existential_risk")


def test_assess_civilizational_trajectory_good_state(engine):
    state = FateState(
        human_population_bn=8.0, energy_consumption_tw=100,
        social_cohesion_index=0.8, governance_effectiveness=0.8,
        ai_capability_index=0.5, resource_remaining_pct=70,
        nuclear_proliferation_risk=0.1,
    )
    result = engine.assess_civilizational_trajectory(state)
    assert result["trajectory_score"] > 0.3


def test_assess_civilizational_trajectory_bad_state(engine):
    state = FateState(
        social_cohesion_index=0.1, governance_effectiveness=0.1,
        nuclear_proliferation_risk=0.8, resource_remaining_pct=20,
        climate_commitment_c=3.0,
    )
    result = engine.assess_civilizational_trajectory(state)
    assert result["existential_risk_composite"] > 0.3


def test_assess_thermodynamic_fate_default(engine):
    result = engine.assess_thermodynamic_fate()
    assert "thermodynamic_pressure" in result
    assert "pressure_label" in result
    assert "eroi_current" in result
    assert "carrying_capacity_usage" in result
    assert "constraints" in result
    assert len(result["constraints"]) >= 5


def test_assess_thermodynamic_fate_high_pressure(engine):
    state = FateState(human_population_bn=12.0, resource_remaining_pct=20, co2_ppm=600)
    result = engine.assess_thermodynamic_fate(state)
    assert result["thermodynamic_pressure"] > 0.4


def test_assess_resource_depletion_default(engine):
    result = engine.assess_resource_depletion()
    assert "overall_depletion_pressure" in result
    assert "n_critical_resources" in result
    assert "depletions" in result
    assert len(result["depletions"]) > 0


def test_assess_resource_depletion_critical(engine):
    state = FateState(resource_remaining_pct=20)
    result = engine.assess_resource_depletion(state, years_ahead=50)
    assert result["n_critical_resources"] > 0


def test_assess_demographic_destiny_default(engine):
    result = engine.assess_demographic_destiny()
    assert "dependency_ratio_outlook" in result
    assert "destinies" in result
    assert len(result["destinies"]) >= 5


def test_assess_demographic_destiny_high_population(engine):
    state = FateState(human_population_bn=11.0, economic_inequality_gini=0.7)
    result = engine.assess_demographic_destiny(state)
    pop_peak = [d for d in result["destinies"] if d["id"] == "population_peak"]
    assert len(pop_peak) > 0
    assert pop_peak[0]["probability"] > 0.5


def test_assess_biological_destiny_default(engine):
    result = engine.assess_biological_destiny()
    assert "biological_advancement_index" in result
    assert "biological_risk_index" in result
    assert "destinies" in result
    assert len(result["destinies"]) >= 5


def test_assess_deterministic_commitments_default(engine):
    result = engine.assess_deterministic_commitments()
    assert "n_commitments" in result
    assert "avg_irreversibility" in result
    assert "commitments" in result
    assert len(result["commitments"]) >= 7


def test_assess_deterministic_commitments_high_climate(engine):
    state = FateState(climate_commitment_c=2.5, biodiversity_integrity_index=0.2, nuclear_warheads=15000)
    result = engine.assess_deterministic_commitments(state)
    assert result["n_high_irreversibility"] >= 3


def test_assess_cosmological_timeline_default(engine):
    result = engine.assess_cosmological_timeline()
    assert "timeline" in result
    assert "near_term_fates" in result
    assert "existential_risks" in result
    assert len(result["timeline"]) >= 5


def test_predict_fate_default(engine):
    result = engine.predict_fate()
    assert "fate_score" in result
    assert "fate_label" in result
    assert "trajectory" in result
    assert "thermodynamic" in result
    assert "depletion" in result
    assert "demographic" in result
    assert "biological" in result
    assert "commitment" in result
    assert "cosmological" in result
    assert "escape_velocities" in result
    assert 0 <= result["fate_score"] <= 1


def test_generate_fate_scenario_default(engine):
    result = engine.generate_fate_scenario()
    assert "scenario" in result
    assert "outcome" in result
    assert "initial_stability" in result
    assert "final_stability" in result
    assert "minimum_stability" in result
    assert result["outcome"] in ("civilizational_collapse", "near_extinction", "severe_crisis", "sustainable_advancement", "muddling_through", "gradual_decline")


def test_generate_fate_scenario_collapse(engine):
    result = engine.generate_fate_scenario("collapse", years=100)
    assert result["minimum_stability"] < result["initial_stability"]


def test_generate_fate_scenario_resilience(engine):
    result = engine.generate_fate_scenario("resilience", years=100)
    assert result["final_stability"] >= result["initial_stability"] * 0.8


def test_generate_fate_scenario_extinction(engine):
    result = engine.generate_fate_scenario("extinction", years=100)
    assert result["minimum_stability"] < 0.3


def test_generate_fate_scenario_singularitarian(engine):
    result = engine.generate_fate_scenario("singularitarian", years=100)
    assert result["scenario"] == "singularitarian"


def test_personal_destiny_assessment(engine):
    state = PersonalFateState(
        genetic_risk_score=0.2, socioeconomic_origin=0.7,
        education_level=0.8, habit_strength=0.3,
        neuroplasticity=0.7, purpose_clarity=0.6,
    )
    result = engine.assess_personal_destiny(state)
    assert "total_determinism" in result
    assert "destiny_label" in result
    assert "agency_score" in result
    assert "genetic_determinism" in result
    assert "structural_determinism" in result
    assert "behavioral_determinism" in result
    assert "modifiable_factors" in result
    assert "locked_in_factors" in result


def test_personal_destiny_heavily_constrained(engine):
    state = PersonalFateState(
        socioeconomic_origin=0.1, intergenerational_mobility=0.2,
        zip_code_destiny=0.1, habit_strength=0.9,
        adverse_childhood_experiences=0.8, genetic_risk_score=0.7,
    )
    result = engine.assess_personal_destiny(state)
    assert result["total_determinism"] > 0.5
    assert result["destiny_label"] in ("heavily_determined", "largely_constrained")


def test_personal_destiny_open_trajectory(engine):
    state = PersonalFateState(
        genetic_risk_score=0.1, socioeconomic_origin=0.9,
        neuroplasticity=0.9, purpose_clarity=0.9,
        cognitive_reserve=0.9, habit_strength=0.1,
    )
    result = engine.assess_personal_destiny(state)
    assert result["agency_score"] > 0.3


def test_mortality_assessment(engine):
    state = PersonalFateState(
        exercise_habit=0.8, sleep_quality=0.8,
        diet_quality=0.7, health_baseline=0.8,
        chronic_disease_risk=0.1,
    )
    result = engine.assess_mortality(state)
    assert "life_expectancy_adjusted" in result
    assert "biological_age" in result
    assert "healthspan_estimate" in result
    assert "cause_probabilities" in result
    assert result["life_expectancy_adjusted"] > 70


def test_mortality_bad_habits(engine):
    state = PersonalFateState(
        exercise_habit=0.1, sleep_quality=0.2,
        diet_quality=0.1, health_baseline=0.3,
        chronic_disease_risk=0.7, addiction_vulnerability=0.8,
    )
    result = engine.assess_mortality(state)
    good_state = PersonalFateState(
        exercise_habit=0.9, sleep_quality=0.9,
        diet_quality=0.8, health_baseline=0.9,
        chronic_disease_risk=0.1,
    )
    result_good = engine.assess_mortality(good_state)
    assert result["life_expectancy_adjusted"] < result_good["life_expectancy_adjusted"]


def test_habit_destiny_assessment(engine):
    result = engine.assess_habit_destiny()
    assert "habit_inertia" in result
    assert "inertia_label" in result
    assert "change_probability" in result
    assert "keystone_habits" in result
    assert "compounding_loops" in result
    assert len(result["keystone_habits"]) >= 4


def test_habit_destiny_high_inertia(engine):
    state = PersonalFateState(habit_strength=0.9, neuroplasticity=0.1, purpose_clarity=0.1)
    result = engine.assess_habit_destiny(state)
    assert result["habit_inertia"] > 0.5
    assert result["inertia_label"] in ("extreme", "high")


def test_habit_destiny_low_inertia(engine):
    state = PersonalFateState(habit_strength=0.1, neuroplasticity=0.9, purpose_clarity=0.9)
    result = engine.assess_habit_destiny(state)
    assert result["habit_inertia"] < 0.5


def test_socioeconomic_path_assessment(engine):
    result = engine.assess_socioeconomic_path()
    assert "origin_trap_index" in result
    assert "path_label" in result
    assert "mobility_potential" in result
    assert "probability_of_class_escape" in result
    assert "probability_of_class_decline" in result


def test_socioeconomic_path_high_origin(engine):
    state = PersonalFateState(
        socioeconomic_origin=0.9, education_level=0.9,
        social_capital=0.8, zip_code_destiny=0.9,
        intergenerational_mobility=0.8,
    )
    result = engine.assess_socioeconomic_path(state)
    assert result["origin_trap_index"] < 0.4
    assert result["path_label"] in ("partially_free", "open_field")


def test_socioeconomic_path_low_origin(engine):
    state = PersonalFateState(
        socioeconomic_origin=0.1, education_level=0.2,
        zip_code_destiny=0.1, intergenerational_mobility=0.2,
        financial_safety_net=0.1,
    )
    result = engine.assess_socioeconomic_path(state)
    assert result["origin_trap_index"] > 0.4


def test_predict_personal_fate(engine):
    state = PersonalFateState(age=40, health_baseline=0.6, habit_strength=0.5)
    result = engine.predict_personal_fate(state)
    assert "fate_type" in result
    assert "total_determinism" in result
    assert "agency_score" in result
    assert "personal_destiny" in result
    assert "mortality" in result
    assert "habit_destiny" in result
    assert "socioeconomic_path" in result
    assert "immutable_pct" in result
    assert "modifiable_pct" in result


def test_predict_personal_fate_fate_types(engine):
    for ft in ("predetermined", "heavily_constrained", "partially_destined", "significant_agency", "open_horizon"):
        pass
    result = engine.predict_personal_fate()
    assert result["fate_type"] in ("predetermined", "heavily_constrained", "partially_destined", "significant_agency", "open_horizon")


def test_determinant_categories(engine):
    result = engine.assess_personal_destiny()
    assert "determinants" in result
    assert "genetic" in result["determinants"]
    assert "epigenetic" in result["determinants"]
    assert "behavioral" in result["determinants"]
    assert "structural" in result["determinants"]
    for cat, data in result["determinants"].items():
        assert "factors" in data
        assert "aggregate_determinism" in data
        assert "modifiable_pct" in data


def test_trajectory_archetypes(engine):
    result = engine.assess_personal_destiny()
    archetypes = result.get("trajectory_archetypes", [])
    assert len(archetypes) >= 4
    for a in archetypes:
        assert "id" in a
        assert "name" in a
        assert "personal_probability" in a
        assert "modifiable" in a


def test_modifiable_factors_ro(engine):
    result = engine.assess_personal_destiny()
    mod_factors = result.get("modifiable_factors", [])
    assert len(mod_factors) > 0
    for f in mod_factors:
        assert "roi" in f
        assert f["roi"] > 0


def test_locked_in_factors(engine):
    result = engine.assess_personal_destiny()
    locked = result.get("locked_in_factors", [])
    assert len(locked) >= 5
    for f in locked:
        assert f["impact"] > 0


def test_escape_velocities(engine):
    result = engine.predict_fate()
    escapes = result.get("escape_velocities", [])
    assert len(escapes) >= 2
    for e in escapes:
        assert "name" in e
        assert "feasibility" in e
        assert "impact" in e


def test_fate_scenario_years(engine):
    result = engine.generate_fate_scenario("default", years=50)
    assert result["years_simulated"] > 0


def test_personal_recommendations(engine):
    state = PersonalFateState(loneliness_index=0.8, purpose_clarity=0.1)
    result = engine.assess_personal_destiny(state)
    assert "recommendations" in result
    assert len(result["recommendations"]) > 0


def test_mortality_cause_probabilities(engine):
    result = engine.assess_mortality()
    causes = result["cause_probabilities"]
    assert "cardiovascular_disease" in causes
    assert "cancer" in causes
    total = sum(v for v in causes.values())
    assert total > 0.5


def test_resource_depletion_substitute_gaps(engine):
    result = engine.assess_resource_depletion()
    assert "substitute_gap_resources" in result
    assert isinstance(result["substitute_gap_resources"], list)