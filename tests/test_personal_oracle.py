"""Tests for Personal Oracle — evidence-based personal future forecasting."""
import pytest
from oracle.prediction.personal_oracle import (
    PersonalOracle, PersonInput, PersonalOracleResult, YearForecast,
    InflectionPoint, MortalityBreakdown, LifePhase, EventCategory,
)


def _person(**kwargs):
    defaults = dict(age=30, sex="male", height_cm=175, weight_kg=75,
                    exercise_minutes_weekly=120, sleep_hours=7.5,
                    diet_quality=0.6, smoking="never", alcohol_drinks_weekly=3,
                    close_friends=4, loneliness=0.3, purpose_score=0.5,
                    stress_level=0.4, annual_income=65000, net_worth=50000,
                    savings_rate=0.10, education_years=16)
    defaults.update(kwargs)
    return PersonInput(**defaults)


class TestPersonalOracleCore:
    def test_initialization(self):
        oracle = PersonalOracle(n_simulations=100)
        assert oracle.n_simulations == 100

    def test_divine_returns_result(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person())
        assert isinstance(result, PersonalOracleResult)

    def test_life_expectancy_reasonable(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person(age=30))
        assert 60 <= result.life_expectancy <= 100

    def test_life_expectancy_ci(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person())
        lo, hi = result.life_expectancy_ci
        assert lo <= result.life_expectancy <= hi
        assert hi > lo

    def test_healthspan_less_than_lifespan(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person())
        assert result.healthspan <= result.life_expectancy

    def test_biological_age(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person(age=30))
        assert result.biological_age >= 18

    def test_timeline_not_empty(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person())
        assert len(result.year_by_year) > 0

    def test_timeline_survival_decreasing(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person())
        for i in range(1, len(result.year_by_year)):
            assert result.year_by_year[i].survival_probability <= result.year_by_year[i - 1].survival_probability + 0.001

    def test_timeline_phases(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person(age=25))
        phases = set(y.phase for y in result.year_by_year)
        assert len(phases) >= 2

    def test_timeline_events(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person(age=30))
        all_events = [e for y in result.year_by_year for e in y.events]
        assert len(all_events) > 0

    def test_inflection_points(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person(age=30))
        assert isinstance(result.inflection_points, list)

    def test_mortality_breakdown(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person())
        assert len(result.mortality_breakdown) > 0
        for m in result.mortality_breakdown:
            assert isinstance(m, MortalityBreakdown)
            assert 0 <= m.lifetime_probability <= 1

    def test_determinism_agency(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person())
        assert 0 <= result.determinism_score <= 1
        assert 0 <= result.agency_score <= 1

    def test_highest_leverage_actions(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person())
        assert len(result.highest_leverage_actions) > 0

    def test_monte_carlo_summary(self):
        oracle = PersonalOracle(n_simulations=100)
        result = oracle.divine(_person())
        mc = result.monte_carlo_summary
        assert mc["n_simulations"] == 100
        assert mc["p5_life_expectancy"] <= mc["median_life_expectancy"] <= mc["p95_life_expectancy"]

    def test_oracle_reading_not_empty(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person())
        assert len(result.oracle_reading) > 50

    def test_recommendations(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person())
        assert len(result.recommendations) > 0

    def test_financial_trajectory(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person())
        assert "peak_income_estimate" in result.financial_trajectory_summary


class TestSmokingEffects:
    def test_smoker_lower_le(self):
        oracle = PersonalOracle(n_simulations=50)
        non_smoker = oracle.divine(_person(smoking="never"))
        smoker = oracle.divine(_person(smoking="current"))
        assert smoker.life_expectancy < non_smoker.life_expectancy

    def test_former_smoker_intermediate(self):
        oracle = PersonalOracle(n_simulations=50)
        never = oracle.divine(_person(smoking="never"))
        former = oracle.divine(_person(smoking="former"))
        current = oracle.divine(_person(smoking="current"))
        assert former.life_expectancy <= never.life_expectancy + 1
        assert former.life_expectancy >= current.life_expectancy

    def test_smoker_cancer_risk_elevated(self):
        oracle = PersonalOracle(n_simulations=50)
        non = oracle.divine(_person(smoking="never"))
        smk = oracle.divine(_person(smoking="current"))
        non_cancer = [m for m in non.mortality_breakdown if m.cause == "cancer"][0]
        smk_cancer = [m for m in smk.mortality_breakdown if m.cause == "cancer"][0]
        assert smk_cancer.lifetime_probability > non_cancer.lifetime_probability


class TestExerciseEffects:
    def test_more_exercise_higher_le(self):
        oracle = PersonalOracle(n_simulations=50)
        sedentary = oracle.divine(_person(exercise_minutes_weekly=0))
        active = oracle.divine(_person(exercise_minutes_weekly=180))
        assert active.life_expectancy > sedentary.life_expectancy

    def test_exercise_appears_in_leverage(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person(exercise_minutes_weekly=30))
        has_exercise = any("exercise" in a["action"].lower() for a in result.highest_leverage_actions)
        assert has_exercise


class TestSocialEffects:
    def test_loneliness_reduces_le(self):
        oracle = PersonalOracle(n_simulations=50)
        connected = oracle.divine(_person(loneliness=0.1, close_friends=8))
        isolated = oracle.divine(_person(loneliness=0.8, close_friends=1))
        assert connected.life_expectancy > isolated.life_expectancy

    def test_loneliness_appears_in_recommendations(self):
        oracle = PersonalOracle(n_simulations=50)
        result = oracle.divine(_person(loneliness=0.7, close_friends=1))
        has_social = any("social" in r.lower() or "friend" in r.lower() or "isolat" in r.lower() for r in result.recommendations)
        assert has_social


class TestBiologicalAge:
    def test_healthy_younger(self):
        oracle = PersonalOracle(n_simulations=10)
        healthy = oracle.divine(_person(exercise_minutes_weekly=180, sleep_hours=8, diet_quality=0.8, stress_level=0.2))
        unhealthy = oracle.divine(_person(exercise_minutes_weekly=0, sleep_hours=5, diet_quality=0.2, stress_level=0.9, smoking="current"))
        assert healthy.biological_age < unhealthy.biological_age

    def test_smoker_bio_age_older(self):
        oracle = PersonalOracle(n_simulations=10)
        result = oracle.divine(_person(age=30, smoking="current"))
        assert result.biological_age > 30


class TestSexEffects:
    def test_female_higher_le(self):
        oracle = PersonalOracle(n_simulations=50)
        male = oracle.divine(_person(sex="male"))
        female = oracle.divine(_person(sex="female"))
        assert female.life_expectancy >= male.life_expectancy


class TestChronicConditions:
    def test_diabetes_reduces_le(self):
        oracle = PersonalOracle(n_simulations=50)
        healthy = oracle.divine(_person(chronic_conditions=[]))
        diabetic = oracle.divine(_person(chronic_conditions=["diabetes"]))
        assert diabetic.life_expectancy < healthy.life_expectancy

    def test_multiple_conditions_compound(self):
        oracle = PersonalOracle(n_simulations=50)
        single = oracle.divine(_person(chronic_conditions=["diabetes"]))
        multi = oracle.divine(_person(chronic_conditions=["diabetes", "hypertension", "depression"]))
        assert multi.life_expectancy < single.life_expectancy


class TestFamilyHistory:
    def test_alzheimer_family_history(self):
        oracle = PersonalOracle(n_simulations=50)
        no_history = oracle.divine(_person(family_history={}))
        with_history = oracle.divine(_person(family_history={"alzheimer": "positive"}))
        neuro_no = [m for m in no_history.mortality_breakdown if m.cause == "neurodegenerative"][0]
        neuro_yes = [m for m in with_history.mortality_breakdown if m.cause == "neurodegenerative"][0]
        assert neuro_yes.lifetime_probability >= neuro_no.lifetime_probability

    def test_longevity_family_history(self):
        oracle = PersonalOracle(n_simulations=50)
        normal = oracle.divine(_person(family_history={}))
        longevity = oracle.divine(_person(family_history={"longevity": "negative"}))
        assert longevity.life_expectancy > normal.life_expectancy


class TestFinancialTrajectory:
    def test_high_saver_better_net_worth(self):
        oracle = PersonalOracle(n_simulations=50)
        low_save = oracle.divine(_person(savings_rate=0.05))
        high_save = oracle.divine(_person(savings_rate=0.25))
        assert high_save.financial_trajectory_summary["savings_at_retirement"] > low_save.financial_trajectory_summary["savings_at_retirement"]

    def test_education_affects_growth(self):
        oracle = PersonalOracle(n_simulations=50)
        hs = oracle.divine(_person(education_years=12))
        college = oracle.divine(_person(education_years=16))
        assert college.financial_trajectory_summary["peak_income_estimate"] > hs.financial_trajectory_summary["peak_income_estimate"]


class TestMonteCarlo:
    def test_distribution_spread(self):
        oracle = PersonalOracle(n_simulations=500)
        result = oracle.divine(_person())
        mc = result.monte_carlo_summary
        spread = mc["p95_life_expectancy"] - mc["p5_life_expectancy"]
        assert spread > 5

    def test_high_risk_narrower(self):
        oracle = PersonalOracle(n_simulations=200)
        low = oracle.divine(_person(smoking="never", exercise_minutes_weekly=180, chronic_conditions=[]))
        high = oracle.divine(_person(smoking="current", exercise_minutes_weekly=0, chronic_conditions=["heart_disease", "diabetes"]))
        assert high.monte_carlo_summary["probability_die_before_65"] >= low.monte_carlo_summary["probability_die_before_65"]


class TestLifePhases:
    def test_young_adult(self):
        oracle = PersonalOracle(n_simulations=10)
        result = oracle.divine(_person(age=22))
        assert result.year_by_year[0].phase == LifePhase.YOUNG_ADULT

    def test_career_building(self):
        oracle = PersonalOracle(n_simulations=10)
        result = oracle.divine(_person(age=30))
        assert result.year_by_year[0].phase == LifePhase.CAREER_BUILDING

    def test_elder(self):
        oracle = PersonalOracle(n_simulations=10)
        result = oracle.divine(_person(age=82))
        assert result.year_by_year[0].phase == LifePhase.ELDER


class TestOracleReading:
    def test_contains_age(self):
        oracle = PersonalOracle(n_simulations=10)
        result = oracle.divine(_person(age=30))
        assert "30" in result.oracle_reading

    def test_smoker_reading_warns(self):
        oracle = PersonalOracle(n_simulations=10)
        result = oracle.divine(_person(smoking="current"))
        assert "cigarette" in result.oracle_reading.lower() or "smok" in result.oracle_reading.lower()

    def test_lonely_reading_warns(self):
        oracle = PersonalOracle(n_simulations=10)
        result = oracle.divine(_person(loneliness=0.9, close_friends=1))
        assert "isolat" in result.oracle_reading.lower() or "lonel" in result.oracle_reading.lower() or "social" in result.oracle_reading.lower()


class TestEdgeCases:
    def test_very_old(self):
        oracle = PersonalOracle(n_simulations=10)
        result = oracle.divine(_person(age=85))
        assert result.life_expectancy >= 85

    def test_very_young(self):
        oracle = PersonalOracle(n_simulations=10)
        result = oracle.divine(_person(age=18))
        assert result.life_expectancy > 50

    def test_very_high_income(self):
        oracle = PersonalOracle(n_simulations=10)
        result = oracle.divine(_person(annual_income=500000))
        assert result.financial_trajectory_summary["peak_income_estimate"] > 500000

    def test_zero_net_worth(self):
        oracle = PersonalOracle(n_simulations=10)
        result = oracle.divine(_person(net_worth=0, debt=100000))
        assert result.financial_trajectory_summary["debt_to_income_ratio"] > 1.0

    def test_all_chronic_conditions(self):
        oracle = PersonalOracle(n_simulations=10)
        result = oracle.divine(_person(chronic_conditions=["diabetes", "hypertension", "heart_disease",
                                                             "cancer_survivor", "depression", "anxiety",
                                                             "asthma", "copd", "obesity", "autoimmune"]))
        assert result.life_expectancy < 70