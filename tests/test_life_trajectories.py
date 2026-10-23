"""Tests for Life Trajectory Engines."""
import pytest
from oracle.prediction.life_trajectories import (
    HealthTrajectoryEngine, CareerTrajectoryEngine, RelationshipTrajectoryEngine,
    FinancialTrajectoryEngine, CognitiveTrajectoryEngine,
)


class TestHealthTrajectory:
    def test_basic_projection(self):
        ht = HealthTrajectoryEngine(n_simulations=100)
        r = ht.project(age=30)
        assert 40 <= r.life_expectancy <= 110
        assert 30 <= r.healthspan <= 100
        assert 18 <= r.biological_age <= 80
        assert len(r.yearly_trajectory) > 0
        assert len(r.disease_onset_probabilities) > 0
        assert len(r.interventions) > 0
        assert len(r.narrative) > 20

    def test_smoker_lower_le(self):
        ht = HealthTrajectoryEngine()
        nonsmoker = ht.project(smoking="never")
        smoker = ht.project(smoking="current")
        assert nonsmoker.life_expectancy > smoker.life_expectancy

    def test_exercise_effect(self):
        ht = HealthTrajectoryEngine()
        sedentary = ht.project(exercise_min=0)
        active = ht.project(exercise_min=180)
        assert active.life_expectancy > sedentary.life_expectancy

    def test_bmi_effect(self):
        ht = HealthTrajectoryEngine()
        normal = ht.project(bmi=22)
        obese = ht.project(bmi=35)
        assert normal.life_expectancy > obese.life_expectancy

    def test_female_higher_le(self):
        ht = HealthTrajectoryEngine()
        male = ht.project(sex="male")
        female = ht.project(sex="female")
        assert female.life_expectancy >= male.life_expectancy

    def test_chronic_conditions(self):
        ht = HealthTrajectoryEngine()
        healthy = ht.project(chronic_conditions=[])
        sick = ht.project(chronic_conditions=["diabetes", "hypertension"])
        assert healthy.life_expectancy > sick.life_expectancy

    def test_family_history(self):
        ht = HealthTrajectoryEngine()
        baseline = ht.project(family_history={})
        with_alz = ht.project(family_history={"alzheimer": "positive"})
        assert baseline.life_expectancy >= with_alz.life_expectancy

    def test_disease_onset_ages(self):
        ht = HealthTrajectoryEngine()
        r = ht.project(age=30)
        assert "cardiovascular" in r.disease_onset_probabilities
        assert "cancer" in r.disease_onset_probabilities
        assert "dementia" in r.disease_onset_probabilities

    def test_interventions_recommended(self):
        ht = HealthTrajectoryEngine()
        r = ht.project(exercise_min=30, sleep_hours=5.5, smoking="current", bmi=32)
        intv_names = [i.name for i in r.interventions]
        assert "exercise_150" in intv_names
        assert "quit_smoking" in intv_names
        assert "sleep_7_9" in intv_names
        assert "weight_loss" in intv_names

    def test_trajectory_decreasing_health(self):
        ht = HealthTrajectoryEngine()
        r = ht.project(age=30)
        health_vals = [y.value for y in r.yearly_trajectory]
        assert health_vals[-1] < health_vals[0]


class TestCareerTrajectory:
    def test_basic_projection(self):
        ct = CareerTrajectoryEngine()
        r = ct.project(age=30, current_income=65000)
        assert r.peak_income > 65000
        assert r.peak_income_age > 30
        assert r.total_career_earnings > 1000000
        assert len(r.yearly_income) > 0
        assert len(r.promotion_probabilities) > 0
        assert len(r.job_loss_probabilities) > 0

    def test_education_affects_growth(self):
        ct = CareerTrajectoryEngine()
        hs = ct.project(education_years=12, current_income=35000)
        college = ct.project(education_years=16, current_income=65000)
        phd = ct.project(education_years=20, current_income=90000)
        assert phd.peak_income > college.peak_income
        assert college.peak_income > hs.peak_income

    def test_satisfaction_effect(self):
        ct = CareerTrajectoryEngine()
        low = ct.project(satisfaction=0.3)
        high = ct.project(satisfaction=0.8)
        assert high.peak_income >= low.peak_income

    def test_industry_outlook(self):
        ct = CareerTrajectoryEngine()
        for industry in ["professional", "tech", "healthcare", "finance"]:
            r = ct.project(industry=industry)
            assert len(r.industry_outlook) > 10

    def test_promotion_decreases_with_age(self):
        ct = CareerTrajectoryEngine()
        r = ct.project(age=25)
        early_probas = [r.promotion_probabilities[a] for a in sorted(r.promotion_probabilities.keys())[:5]]
        later_probas = [r.promotion_probabilities[a] for a in sorted(r.promotion_probabilities.keys())[-5:]]
        assert sum(early_probas) / max(len(early_probas), 1) >= sum(later_probas) / max(len(later_probas), 1)


class TestRelationshipTrajectory:
    def test_basic_projection(self):
        rt = RelationshipTrajectoryEngine()
        r = rt.project(age=30)
        assert len(r.marriage_probability_by_age) > 0
        assert len(r.divorce_probability_by_age) > 0
        assert 0 <= r.children_probability <= 1
        assert len(r.social_network_size_trajectory) > 0
        assert len(r.loneliness_trajectory) > 0
        assert len(r.interventions) > 0

    def test_married_has_divorce_risk(self):
        rt = RelationshipTrajectoryEngine()
        r = rt.project(age=30, marital_status="married")
        divorce_sum = sum(r.divorce_probability_by_age.values())
        assert divorce_sum > 0

    def test_single_has_marriage_prob(self):
        rt = RelationshipTrajectoryEngine()
        r = rt.project(age=25, marital_status="single")
        assert len(r.marriage_probability_by_age) > 0

    def test_loneliness_affects_narrative(self):
        rt = RelationshipTrajectoryEngine()
        lonely = rt.project(age=30, loneliness=0.8, close_friends=1)
        social = rt.project(age=30, loneliness=0.2, close_friends=8)
        assert "urgent" in lonely.narrative.lower() or "lonely" in lonely.narrative.lower() or "loneliness" in lonely.narrative.lower()

    def test_social_network_decline(self):
        rt = RelationshipTrajectoryEngine()
        r = rt.project(age=30, close_friends=10)
        early_friends = r.social_network_size_trajectory[0].value
        later_friends = r.social_network_size_trajectory[-1].value
        assert later_friends < early_friends


class TestFinancialTrajectory:
    def test_basic_projection(self):
        ft = FinancialTrajectoryEngine()
        r = ft.project(age=30, income=65000, net_worth=50000)
        assert r.net_worth_at_65 > 0
        assert len(r.yearly_net_worth) > 0
        assert len(r.yearly_income) > 0
        assert len(r.yearly_savings) > 0
        assert len(r.interventions) > 0

    def test_high_saver_better_nw(self):
        ft = FinancialTrajectoryEngine()
        low = ft.project(savings_rate=0.05)
        high = ft.project(savings_rate=0.25)
        assert high.net_worth_at_65 > low.net_worth_at_65

    def test_fire_age(self):
        ft = FinancialTrajectoryEngine()
        r = ft.project(age=25, income=100000, savings_rate=0.50, net_worth=100000)
        assert r.fire_age is not None or r.net_worth_at_65 > 0

    def test_recession_scenario(self):
        ft = FinancialTrajectoryEngine()
        r = ft.project()
        assert "mild" in r.recession_impact_scenario
        assert "severe" in r.recession_impact_scenario

    def test_debt_burden(self):
        ft = FinancialTrajectoryEngine()
        no_debt = ft.project(debt=0)
        high_debt = ft.project(debt=100000)
        assert no_debt.net_worth_at_65 > high_debt.net_worth_at_65


class TestCognitiveTrajectory:
    def test_basic_projection(self):
        ct = CognitiveTrajectoryEngine()
        r = ct.project(age=30)
        assert r.peak_cognitive_age == 29
        assert r.cognitive_decline_start_age > 30
        assert 0 <= r.mci_probability_by_85 <= 1
        assert 0 <= r.dementia_probability_by_85 <= 1
        assert len(r.yearly_cognitive) > 0
        assert len(r.brain_health_interventions) > 0

    def test_education_builds_reserve(self):
        ct = CognitiveTrajectoryEngine()
        low_edu = ct.project(education_years=12)
        high_edu = ct.project(education_years=20)
        assert high_edu.dementia_probability_by_85 < low_edu.dementia_probability_by_85

    def test_exercise_delays_decline(self):
        ct = CognitiveTrajectoryEngine()
        sedentary = ct.project(exercise_min=0)
        active = ct.project(exercise_min=180)
        assert active.cognitive_decline_start_age >= sedentary.cognitive_decline_start_age

    def test_family_history_alzheimer(self):
        ct = CognitiveTrajectoryEngine()
        baseline = ct.project(family_history={})
        with_alz = ct.project(family_history={"alzheimer": "positive"})
        assert with_alz.mci_probability_by_85 > baseline.mci_probability_by_85
        assert with_alz.dementia_probability_by_85 > baseline.dementia_probability_by_85

    def test_cognitive_peak_then_decline(self):
        ct = CognitiveTrajectoryEngine()
        r = ct.project(age=20)
        peak_idx = max(range(len(r.yearly_cognitive)), key=lambda i: r.yearly_cognitive[i].value)
        decline_after = r.yearly_cognitive[peak_idx + 10:]
        if decline_after:
            assert all(d.value <= r.yearly_cognitive[peak_idx].value + 0.01 for d in decline_after[5:])

    def test_cognitive_values_in_range(self):
        ct = CognitiveTrajectoryEngine()
        r = ct.project()
        for y in r.yearly_cognitive:
            assert 0.1 <= y.value <= 1.05
            assert 0.1 <= y.confidence <= 1.0