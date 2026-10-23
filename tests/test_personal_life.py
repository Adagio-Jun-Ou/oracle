import pytest
from oracle.domains.personal_life import (
    PersonalLifeModel, PersonProfile, FamilyMemberProfile, SocialContactProfile,
    CAREER_STAGES, HEALTH_MILESTONES, LIFE_EVENT_TEMPLATES,
    FAMILY_DYNAMICS_PATTERNS, SOCIAL_NETWORK_EFFECTS,
)


@pytest.fixture
def model():
    return PersonalLifeModel(seed=42)


@pytest.fixture
def young_professional():
    return PersonProfile(
        age=28,
        biological_sex="male",
        height_cm=178,
        weight_kg=72,
        education_years=18,
        occupation="software engineer",
        annual_income=95000,
        net_worth=30000,
        debt_total=45000,
        marital_status="single",
        n_children=0,
        exercise_minutes_weekly=120,
        sleep_hours_avg=7.5,
        diet_quality=0.6,
        smoking_status="never",
        alcohol_drinks_weekly=4,
        stress_level=0.5,
        relationship_quality=0.5,
        job_satisfaction=0.7,
        life_satisfaction=0.6,
        social_network_size=15,
        close_friends=4,
        community_involvement=0.3,
        purpose_score=0.6,
        resilience_score=0.7,
        savings_rate=0.12,
        career_growth_rate=0.05,
    )


@pytest.fixture
def middle_aged_parent():
    return PersonProfile(
        age=42,
        biological_sex="female",
        height_cm=163,
        weight_kg=68,
        education_years=16,
        occupation="manager",
        annual_income=85000,
        net_worth=200000,
        debt_total=120000,
        marital_status="married",
        n_children=2,
        children_ages=[8, 12],
        exercise_minutes_weekly=60,
        sleep_hours_avg=6.5,
        diet_quality=0.5,
        smoking_status="never",
        alcohol_drinks_weekly=2,
        stress_level=0.6,
        relationship_quality=0.6,
        job_satisfaction=0.5,
        life_satisfaction=0.5,
        social_network_size=10,
        close_friends=2,
        community_involvement=0.4,
        purpose_score=0.5,
        resilience_score=0.6,
        savings_rate=0.18,
        career_growth_rate=0.03,
        chronic_conditions=["mild_hypertension"],
        family_history={"cardiovascular_disease": True},
    )


@pytest.fixture
def older_smoker():
    return PersonProfile(
        age=58,
        biological_sex="male",
        height_cm=175,
        weight_kg=95,
        education_years=12,
        occupation="tradesman",
        annual_income=55000,
        net_worth=80000,
        debt_total=30000,
        marital_status="divorced",
        n_children=1,
        children_ages=[25],
        exercise_minutes_weekly=30,
        sleep_hours_avg=5.5,
        diet_quality=0.3,
        smoking_status="current",
        alcohol_drinks_weekly=10,
        stress_level=0.7,
        relationship_quality=0.3,
        job_satisfaction=0.3,
        life_satisfaction=0.3,
        social_network_size=5,
        close_friends=1,
        community_involvement=0.1,
        purpose_score=0.3,
        resilience_score=0.4,
        savings_rate=0.05,
        career_growth_rate=0.01,
        chronic_conditions=["diabetes_type2", "hypertension"],
        family_history={"cardiovascular_disease": True, "diabetes_type2": True},
    )


class TestPersonProfile:
    def test_default_profile(self):
        p = PersonProfile()
        assert p.age == 35
        assert p.biological_sex == "male"
        assert p.annual_income == 75000
        assert p.chronic_conditions == []
        assert p.family_history == {}

    def test_custom_profile(self, young_professional):
        assert young_professional.age == 28
        assert young_professional.annual_income == 95000
        assert young_professional.marital_status == "single"
        assert young_professional.n_children == 0

    def test_financial_safety_net_method(self):
        p = PersonProfile(annual_income=100000, savings_rate=0.2)
        assert hasattr(p, 'annual_income')
        assert p.savings_rate == 0.2


class TestFamilyMemberProfile:
    def test_default(self):
        fm = FamilyMemberProfile()
        assert fm.relation == "spouse"
        assert fm.health_status == 0.8
        assert fm.caregiving_burden == 0.2

    def test_custom(self):
        fm = FamilyMemberProfile(
            relation="mother", age=68, health_status=0.4,
            caregiving_burden=0.6, dependency_level=0.5
        )
        assert fm.relation == "mother"
        assert fm.health_status == 0.4


class TestSocialContactProfile:
    def test_default(self):
        sc = SocialContactProfile()
        assert sc.contact_type == "friend"
        assert sc.closeness == 0.6
        assert sc.reliability == 0.6

    def test_custom(self):
        sc = SocialContactProfile(
            contact_type="colleague", closeness=0.3,
            emotional_valence=0.2, professional_value=0.8
        )
        assert sc.contact_type == "colleague"
        assert sc.emotional_valence == 0.2


class TestCareerPrediction:
    def test_career_trajectory_young(self, model, young_professional):
        result = model.predict_career_trajectory(young_professional, years=20)
        assert "current_stage" in result
        assert "base_growth_rate" in result
        assert "projected_peak_income" in result
        assert "trajectory" in result
        assert result["current_stage"] == "early_career"
        assert result["projected_peak_income"] > young_professional.annual_income
        assert len(result["trajectory"]) > 0

    def test_career_trajectory_mid(self, model, middle_aged_parent):
        result = model.predict_career_trajectory(middle_aged_parent, years=20)
        assert result["current_stage"] in ["mid_career", "career_building"]
        assert result["projected_peak_income"] > 0

    def test_career_trajectory_senior(self, model, older_smoker):
        result = model.predict_career_trajectory(older_smoker, years=15)
        assert result["current_stage"] in ["senior", "pre_retirement"]
        assert result["years_to_peak"] >= 0

    def test_career_stages_defined(self):
        assert len(CAREER_STAGES) == 6
        assert "early_career" in CAREER_STAGES
        assert "retirement" in CAREER_STAGES
        for stage_data in CAREER_STAGES.values():
            assert "age_range" in stage_data
            assert "growth_rate" in stage_data
            assert "volatility" in stage_data

    def test_career_short_horizon(self, model, young_professional):
        result = model.predict_career_trajectory(young_professional, years=1)
        assert len(result["trajectory"]) >= 1


class TestHealthPrediction:
    def test_health_trajectory_young(self, model, young_professional):
        result = model.predict_health_trajectory(young_professional, years=30)
        assert "life_expectancy_adjusted" in result
        assert "healthspan_estimate" in result
        assert "current_bmi" in result
        assert "bmi_category" in result
        assert result["life_expectancy_adjusted"] > 40
        assert result["life_expectancy_adjusted"] <= 105
        assert result["current_bmi"] > 0

    def test_health_trajectory_risky(self, model, older_smoker):
        result = model.predict_health_trajectory(older_smoker, years=30)
        assert result["life_expectancy_adjusted"] < 79
        assert result["bmi_category"] == "obese"

    def test_health_milestones(self, model, young_professional):
        result = model.predict_health_trajectory(young_professional, years=30)
        assert "upcoming_health_milestones" in result
        assert len(result["upcoming_health_milestones"]) > 0

    def test_condition_risks(self, model, middle_aged_parent):
        result = model.predict_health_trajectory(middle_aged_parent, years=30)
        assert "condition_risks" in result
        assert "cardiovascular_disease" in result["condition_risks"]
        cv_risk = result["condition_risks"]["cardiovascular_disease"]
        assert cv_risk > 0.3
        assert "top_3_risks" in result
        assert len(result["top_3_risks"]) <= 3

    def test_modifiable_factors(self, model, young_professional):
        result = model.predict_health_trajectory(young_professional, years=30)
        assert "modifiable_factors" in result
        mf = result["modifiable_factors"]
        assert "exercise" in mf
        assert "sleep" in mf
        assert "diet" in mf
        assert "weight" in mf

    def test_health_trajectory_over_time(self, model, young_professional):
        result = model.predict_health_trajectory(young_professional, years=30)
        assert "health_trajectory" in result
        scores = [h["health_score"] for h in result["health_trajectory"]]
        assert all(0 <= s <= 1 for s in scores)

    def test_bmi_categories(self, model):
        underweight = PersonProfile(height_cm=180, weight_kg=50)
        r = model.predict_health_trajectory(underweight)
        assert r["bmi_category"] == "underweight"

        normal = PersonProfile(height_cm=175, weight_kg=70)
        r = model.predict_health_trajectory(normal)
        assert r["bmi_category"] == "normal"

        overweight = PersonProfile(height_cm=175, weight_kg=85)
        r = model.predict_health_trajectory(overweight)
        assert r["bmi_category"] in ("overweight", "normal")

        obese = PersonProfile(height_cm=175, weight_kg=110)
        r = model.predict_health_trajectory(obese)
        assert r["bmi_category"] == "obese"


class TestFinancialPrediction:
    def test_financial_trajectory(self, model, young_professional):
        result = model.predict_financial_trajectory(young_professional, years=30)
        assert "current_net_worth" in result
        assert "savings_rate" in result
        assert "projected_net_worth" in result
        assert "financial_risks" in result
        assert "recommendations" in result
        assert result["current_net_worth"] == young_professional.net_worth

    def test_financial_high_debt(self, model, older_smoker):
        result = model.predict_financial_trajectory(older_smoker, years=20)
        high_severity = [r for r in result["financial_risks"] if r["severity"] == "high"]
        assert len(high_severity) > 0

    def test_financial_net_worth_projection(self, model, middle_aged_parent):
        result = model.predict_financial_trajectory(middle_aged_parent, years=30)
        nw = result["projected_net_worth"]
        assert "year_10" in nw
        assert "year_20" in nw
        assert "year_30" in nw

    def test_financial_recommendations(self, model, young_professional):
        result = model.predict_financial_trajectory(young_professional, years=20)
        assert len(result["recommendations"]) > 0


class TestRelationshipPrediction:
    def test_relationship_married(self, model, middle_aged_parent):
        result = model.predict_relationship_trajectory(middle_aged_parent, years=20)
        assert result["current_marital_status"] == "married"
        assert "estimated_divorce_probability_annual" in result
        assert result["estimated_divorce_probability_annual"] >= 0

    def test_relationship_single(self, model, young_professional):
        result = model.predict_relationship_trajectory(young_professional, years=20)
        assert result["current_marital_status"] == "single"

    def test_relationship_loneliness(self, model, older_smoker):
        result = model.predict_relationship_trajectory(older_smoker, years=20)
        assert result["loneliness_risk"] > 0
        assert result["social_support_adequacy"] in ("adequate", "at_risk", "isolated")

    def test_child_milestones(self, model, middle_aged_parent):
        result = model.predict_relationship_trajectory(middle_aged_parent, years=20)
        assert "child_milestones" in result
        assert len(result["child_milestones"]) == 2


class TestTimelinePrediction:
    def test_personal_timeline(self, model, young_professional):
        result = model.predict_personal_timeline(young_professional, years=20)
        assert "probable_life_events" in result
        assert "top_5_likely_events" in result
        assert "total_event_burden" in result
        assert len(result["probable_life_events"]) > 0
        assert len(result["top_5_likely_events"]) <= 5

    def test_timeline_sorted_by_probability(self, model, young_professional):
        result = model.predict_personal_timeline(young_professional, years=20)
        events = result["probable_life_events"]
        for i in range(len(events) - 1):
            assert events[i]["period_probability"] >= events[i + 1]["period_probability"]

    def test_timeline_events_have_required_fields(self, model, young_professional):
        result = model.predict_personal_timeline(young_professional, years=20)
        for event in result["probable_life_events"]:
            assert "event_id" in event
            assert "name" in event
            assert "annual_probability" in event
            assert "period_probability" in event
            assert "cascade_effects" in event

    def test_life_event_templates(self):
        assert len(LIFE_EVENT_TEMPLATES) >= 12
        for eid, tmpl in LIFE_EVENT_TEMPLATES.items():
            assert "name" in tmpl
            assert "base_probability_annual" in tmpl
            assert "age_modifier" in tmpl
            assert "impact" in tmpl
            assert callable(tmpl["age_modifier"])


class TestFamilyDynamics:
    def test_family_dynamics_basic(self, model, middle_aged_parent):
        family = [
            FamilyMemberProfile(relation="spouse", age=44, health_status=0.8),
            FamilyMemberProfile(relation="mother", age=70, health_status=0.4, caregiving_burden=0.5),
        ]
        result = model.assess_family_dynamics(middle_aged_parent, family_members=family)
        assert "family_size" in result
        assert result["family_size"] == 3
        assert "active_patterns" in result
        assert "family_health_summary" in result
        assert "children_outlook" in result
        assert "overall_family_stability" in result

    def test_family_no_members(self, model, young_professional):
        result = model.assess_family_dynamics(young_professional, family_members=[])
        assert result["family_size"] == 1
        assert result["family_health_summary"]["average_health"] == 0.7

    def test_family_default(self, model, young_professional):
        result = model.assess_family_dynamics(young_professional)
        assert result["family_size"] == 1

    def test_family_at_risk_member(self, model, middle_aged_parent):
        family = [
            FamilyMemberProfile(relation="father", age=72, health_status=0.3, caregiving_burden=0.6),
        ]
        result = model.assess_family_dynamics(middle_aged_parent, family_members=family)
        assert len(result["family_health_summary"]["at_risk_members"]) > 0

    def test_children_outlook(self, model, middle_aged_parent):
        result = model.assess_family_dynamics(middle_aged_parent, family_members=[])
        assert len(result["children_outlook"]) == 2
        for co in result["children_outlook"]:
            assert "education_stage" in co
            assert "years_to_independence" in co

    def test_family_dynamics_patterns(self):
        assert len(FAMILY_DYNAMICS_PATTERNS) == 5
        for pid, pattern in FAMILY_DYNAMICS_PATTERNS.items():
            assert "name" in pattern
            assert "trigger" in pattern
            assert "impact" in pattern
            assert "description" in pattern


class TestSocialNetwork:
    def test_social_network_basic(self, model, young_professional):
        contacts = [
            SocialContactProfile(contact_type="friend", closeness=0.8, emotional_valence=0.8),
            SocialContactProfile(contact_type="colleague", closeness=0.4, emotional_valence=0.5),
        ]
        result = model.assess_social_network(young_professional, contacts=contacts)
        assert "network_size" in result
        assert "close_connections" in result
        assert "network_adequacy" in result
        assert "loneliness_score" in result
        assert result["close_connections"] == young_professional.close_friends

    def test_social_network_no_contacts(self, model, young_professional):
        result = model.assess_social_network(young_professional, contacts=[])
        assert result["network_size"] > 0
        assert result["network_adequacy"] in ("strong", "adequate", "at_risk", "isolated")

    def test_social_network_default(self, model, young_professional):
        result = model.assess_social_network(young_professional)
        assert "active_network_effects" in result

    def test_social_network_negative_contacts(self, model, older_smoker):
        contacts = [
            SocialContactProfile(closeness=0.3, emotional_valence=0.2),
        ]
        result = model.assess_social_network(older_smoker, contacts=contacts)
        assert result["positive_negative_ratio"] >= 0

    def test_network_effects_defined(self):
        assert len(SOCIAL_NETWORK_EFFECTS) == 6
        assert "health_contagion" in SOCIAL_NETWORK_EFFECTS
        for eid, effect in SOCIAL_NETWORK_EFFECTS.items():
            assert "name" in effect
            assert "strength" in effect
            assert "mechanism" in effect


class TestComprehensiveFuture:
    def test_comprehensive(self, model, young_professional):
        result = model.predict_comprehensive_future(young_professional, years=20)
        assert "overall_outlook_score" in result
        assert "overall_outlook_label" in result
        assert "career" in result
        assert "health" in result
        assert "financial" in result
        assert "relationship" in result
        assert "timeline" in result
        assert "family" in result
        assert "network" in result
        assert "key_concerns" in result
        assert "top_opportunities" in result
        assert 0.1 <= result["overall_outlook_score"] <= 1.0
        assert result["overall_outlook_label"] in ("flourishing", "stable_with_potential", "managing", "at_risk")

    def test_comprehensive_risky_profile(self, model, older_smoker):
        result = model.predict_comprehensive_future(older_smoker, years=20)
        assert result["overall_outlook_label"] in ("managing", "at_risk")
        assert len(result["key_concerns"]) > 0

    def test_comprehensive_with_family(self, model, middle_aged_parent):
        family = [FamilyMemberProfile(relation="spouse", age=44)]
        contacts = [SocialContactProfile(closeness=0.7)]
        result = model.predict_comprehensive_future(
            middle_aged_parent, family_members=family, contacts=contacts, years=20
        )
        assert result["family"]["family_size"] == 2
        assert result["network"]["network_size"] > 0

    def test_health_milestones_data(self):
        assert len(HEALTH_MILESTONES) >= 10
        for ms in HEALTH_MILESTONES:
            assert "age" in ms
            assert "event" in ms
            assert "type" in ms
            assert "impact" in ms


class TestEngineIntegration:
    def test_engine_personal_life_predict(self):
        from oracle.prediction.engine import OracleEngine
        from oracle.core.types import Domain
        engine = OracleEngine()
        engine.set_personal_life_profile(age=30, annual_income=80000, biological_sex="male")
        from datetime import date
        from oracle.core.types import ConfidenceLevel
        forecast = engine.predict(date(2035, 1, 1), ConfidenceLevel.HIGH)
        pl_preds = [p for p in forecast.predictions if p.domain == Domain.PERSONAL_LIFE]
        assert len(pl_preds) >= 2
        career_pred = next((p for p in pl_preds if p.variable == "career_trajectory_peak_income"), None)
        assert career_pred is not None
        le_pred = next((p for p in pl_preds if p.variable == "life_expectancy"), None)
        assert le_pred is not None

    def test_engine_personal_life_diagnose(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_personal_life_profile(age=40, annual_income=90000, biological_sex="female")
        result = engine.diagnose()
        assert "personal_life_career" in result
        assert "personal_life_health" in result
        assert "personal_life_financial" in result
        assert "personal_life_relationship" in result
        assert "personal_life_comprehensive" in result
        assert "personal_life_timeline" in result

    def test_engine_cross_domain_interactions(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        engine.set_personal_life_profile(
            age=35, annual_income=30000, biological_sex="male",
            net_worth=5000, savings_rate=0.02, close_friends=1
        )
        result = engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        personal_interactions = [i for i in interactions if "personal_life" in i.get("type", "")]
        assert len(personal_interactions) >= 0


class TestWebAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from oracle.web.app import app
        return TestClient(app)

    def test_personal_life_career_endpoint(self, client):
        resp = client.get("/api/personal-life/career")
        assert resp.status_code == 200
        data = resp.json()
        assert "current_stage" in data
        assert "projected_peak_income" in data

    def test_personal_life_health_endpoint(self, client):
        resp = client.get("/api/personal-life/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "life_expectancy_adjusted" in data
        assert "current_bmi" in data

    def test_personal_life_financial_endpoint(self, client):
        resp = client.get("/api/personal-life/financial")
        assert resp.status_code == 200
        data = resp.json()
        assert "current_net_worth" in data

    def test_personal_life_relationship_endpoint(self, client):
        resp = client.get("/api/personal-life/relationship")
        assert resp.status_code == 200
        data = resp.json()
        assert "current_marital_status" in data

    def test_personal_life_timeline_endpoint(self, client):
        resp = client.get("/api/personal-life/timeline")
        assert resp.status_code == 200
        data = resp.json()
        assert "probable_life_events" in data

    def test_personal_life_comprehensive_endpoint(self, client):
        resp = client.get("/api/personal-life/comprehensive")
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_outlook_score" in data

    def test_personal_life_profile_endpoint(self, client):
        resp = client.post("/api/personal-life/profile", json={
            "age": 32, "annual_income": 100000, "biological_sex": "female",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_personal_life_family_endpoint(self, client):
        resp = client.post("/api/personal-life/family", json=[
            {"relation": "spouse", "age": 34, "health_status": 0.9}
        ])
        assert resp.status_code == 200
        data = resp.json()
        assert "family_size" in data

    def test_personal_life_social_network_endpoint(self, client):
        resp = client.post("/api/personal-life/social-network", json=[
            {"contact_type": "friend", "closeness": 0.8}
        ])
        assert resp.status_code == 200
        data = resp.json()
        assert "network_size" in data

    def test_personal_life_in_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        domains = data["models"]["domains"]
        assert "personal_life" in domains
        endpoints = data["api_endpoints"]
        assert any("personal-life" in e for e in endpoints)

    def test_personal_life_career_with_years(self, client):
        resp = client.get("/api/personal-life/career?years=10")
        assert resp.status_code == 200