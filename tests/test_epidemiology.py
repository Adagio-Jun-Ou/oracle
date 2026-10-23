import pytest
import numpy as np
from oracle.domains.epidemiology import EpidemiologyModel, SEIRState, EpidemicPrediction


class TestSEIRState:
    def test_defaults(self):
        state = SEIRState()
        assert state.susceptible == 0.0
        assert state.exposed == 0.0
        assert state.infectious == 0.0
        assert state.recovered == 0.0
        assert state.deceased == 0.0
        assert state.vaccinated == 0.0
        assert state.hospitalized == 0.0

    def test_custom_values(self):
        state = SEIRState(susceptible=1000, infectious=50, recovered=200)
        assert state.susceptible == 1000
        assert state.infectious == 50
        assert state.recovered == 200


class TestEpidemicPrediction:
    def test_defaults(self):
        pred = EpidemicPrediction()
        assert pred.peak_infected == 0.0
        assert pred.peak_day == 0
        assert pred.daily_curve == []

    def test_custom(self):
        pred = EpidemicPrediction(peak_infected=5000, peak_day=30, r0=2.5)
        assert pred.peak_infected == 5000
        assert pred.peak_day == 30
        assert pred.r0 == 2.5


class TestEpidemiologyModelInit:
    def test_name(self):
        model = EpidemiologyModel()
        assert model.name == "epidemiology_model"


class TestSEIR:
    def test_basic_seir(self):
        epi = EpidemiologyModel()
        result = epi.seir(population=100000, initial_infected=10, days=100)
        assert "peak_infected" in result
        assert "peak_day" in result
        assert "total_deceased" in result
        assert "total_recovered" in result
        assert "r0" in result
        assert "herd_immunity_threshold" in result
        assert "daily" in result
        assert result["r0"] == pytest.approx(0.3 / (1 / 12), rel=0.01)

    def test_seir_parameters(self):
        epi = EpidemiologyModel()
        result = epi.seir(population=1e6, initial_infected=100, beta=0.3, sigma=1/5.2, gamma=1/12, mu=0.005, days=365)
        assert result["peak_infected"] > 0
        assert result["total_deceased"] >= 0
        assert result["total_recovered"] > 0
        assert result["population"] == 1e6
        assert result["parameters"]["beta"] == 0.3
        assert result["parameters"]["sigma"] == pytest.approx(1/5.2, rel=0.01)
        assert result["parameters"]["gamma"] == pytest.approx(1/12, rel=0.01)

    def test_seir_daily_curves(self):
        epi = EpidemiologyModel()
        result = epi.seir(population=1e6, days=365)
        daily = result["daily"]
        assert "susceptible" in daily
        assert "exposed" in daily
        assert "infectious" in daily
        assert "recovered" in daily
        assert "deceased" in daily
        assert "hospitalized" in daily
        assert len(daily["susceptible"]) > 0
        assert daily["susceptible"][0] > daily["susceptible"][-1]

    def test_seir_with_vaccination(self):
        epi = EpidemiologyModel()
        result_no_vax = epi.seir(population=1e6, initial_infected=100, days=200, vaccination_rate=0.0)
        result_vax = epi.seir(population=1e6, initial_infected=100, days=200, vaccination_rate=0.5)
        assert result_vax["peak_infected"] < result_no_vax["peak_infected"]

    def test_seir_with_intervention(self):
        epi = EpidemiologyModel()
        result_no_int = epi.seir(population=1e6, initial_infected=100, days=365)
        result_int = epi.seir(population=1e6, initial_infected=100, days=365, intervention_day=30, intervention_beta=0.1)
        assert result_int["peak_infected"] < result_no_int["peak_infected"]

    def test_seir_high_mortality(self):
        epi = EpidemiologyModel()
        result = epi.seir(population=1e6, initial_infected=100, mu=0.05, days=365)
        assert result["total_deceased"] > 0
        assert result["total_deceased"] > 0

    def test_seir_hospitalization(self):
        epi = EpidemiologyModel()
        result = epi.seir(population=1e6, initial_infected=100, hospitalization_rate=0.3, days=200)
        assert "hospitalized" in result["daily"]
        assert any(h > 0 for h in result["daily"]["hospitalized"][1:])

    def test_seir_short_simulation(self):
        epi = EpidemiologyModel()
        result = epi.seir(population=10000, initial_infected=10, days=30)
        assert result["duration_days"] == 30
        assert result["days_to_peak"] <= 30

    def test_seir_days_to_peak(self):
        epi = EpidemiologyModel()
        result = epi.seir(population=1e6, initial_infected=100, days=365)
        assert 0 <= result["days_to_peak"] <= 365


class TestPandemicScenarios:
    def test_default_scenarios(self):
        epi = EpidemiologyModel()
        result = epi.pandemic_scenarios()
        assert "scenarios" in result
        assert "no_intervention" in result["scenarios"]
        assert "moderate_lockdown" in result["scenarios"]
        assert "strict_lockdown" in result["scenarios"]
        assert "vaccination_50pct" in result["scenarios"]
        assert "vaccination_80pct" in result["scenarios"]

    def test_scenario_structure(self):
        epi = EpidemiologyModel()
        result = epi.pandemic_scenarios()
        for name, scenario in result["scenarios"].items():
            assert "peak_infected" in scenario
            assert "peak_day" in scenario
            assert "total_deceased" in scenario
            assert "r0" in scenario

    def test_intervention_reduces_deaths(self):
        epi = EpidemiologyModel()
        result = epi.pandemic_scenarios()
        no_int = result["scenarios"]["no_intervention"]["total_deceased"]
        strict = result["scenarios"]["strict_lockdown"]["total_deceased"]
        assert strict < no_int

    def test_vaccination_reduces_peak(self):
        epi = EpidemiologyModel()
        result = epi.pandemic_scenarios()
        no_int = result["scenarios"]["no_intervention"]["peak_infected"]
        vax80 = result["scenarios"]["vaccination_80pct"]["peak_infected"]
        assert vax80 < no_int

    def test_custom_r0(self):
        epi = EpidemiologyModel()
        result = epi.pandemic_scenarios(base_r0=5.0)
        assert result["base_r0"] == 5.0

    def test_custom_population(self):
        epi = EpidemiologyModel()
        result = epi.pandemic_scenarios(population=5e6)
        assert result["population"] == 5e6


class TestHealthRiskScore:
    def test_default_risk(self):
        epi = EpidemiologyModel()
        result = epi.health_risk_score()
        assert "total_risk_score" in result
        assert "severity" in result
        assert "specific_risks" in result
        assert "recommendations" in result
        assert 0 <= result["total_risk_score"] <= 1

    def test_low_risk_profile(self):
        epi = EpidemiologyModel()
        result = epi.health_risk_score(age=25, bmi=22, region_risk=0.1, healthcare_access=0.95)
        assert result["total_risk_score"] < 0.05
        assert result["severity"] in ["low", "minimal"]

    def test_high_risk_profile(self):
        epi = EpidemiologyModel()
        result = epi.health_risk_score(age=75, bmi=35, smoking=True, diabetes=True, hypertension=True, cardiovascular=True, region_risk=0.9, healthcare_access=0.2)
        assert result["total_risk_score"] > 0.05
        assert result["severity"] in ["high", "very_high", "critical"]

    def test_age_factors(self):
        epi = EpidemiologyModel()
        young = epi.health_risk_score(age=25)
        old = epi.health_risk_score(age=75)
        assert old["age_factor"] > young["age_factor"]

    def test_comorbidities(self):
        epi = EpidemiologyModel()
        healthy = epi.health_risk_score(smoking=False, diabetes=False, hypertension=False)
        sick = epi.health_risk_score(smoking=True, diabetes=True, hypertension=True, cardiovascular=True)
        assert sick["comorbidity_factor"] > healthy["comorbidity_factor"]
        assert sick["comorbidity_count"] > healthy["comorbidity_count"]

    def test_specific_risks(self):
        epi = EpidemiologyModel()
        result = epi.health_risk_score()
        assert "cardiovascular" in result["specific_risks"]
        assert "respiratory" in result["specific_risks"]
        assert "metabolic" in result["specific_risks"]
        assert "infection" in result["specific_risks"]
        assert all(0 <= v <= 1 for v in result["specific_risks"].values())

    def test_smoking_recommendation(self):
        epi = EpidemiologyModel()
        result = epi.health_risk_score(smoking=True)
        assert any("Smoking" in r for r in result["recommendations"])

    def test_bmi_recommendation(self):
        epi = EpidemiologyModel()
        result = epi.health_risk_score(bmi=35)
        assert any("Weight" in r or "BMI" in r for r in result["recommendations"])


class TestDiseaseSpreadNetwork:
    def test_default_network(self):
        epi = EpidemiologyModel()
        result = epi.disease_spread_network()
        assert "n_nodes" in result
        assert "n_edges" in result
        assert "peak_infected" in result
        assert "peak_day" in result
        assert "final_infected_pct" in result
        assert "daily_counts" in result
        assert result["n_nodes"] == 100
        assert result["n_edges"] == 200

    def test_network_spread_progression(self):
        epi = EpidemiologyModel()
        result = epi.disease_spread_network(n_nodes=50, n_edges=100, initial_infected=5, days=40, seed=42)
        counts = result["daily_counts"]
        assert len(counts) > 0
        assert counts[0]["susceptible"] > 0
        assert counts[0]["infected"] > 0

    def test_network_parameters(self):
        epi = EpidemiologyModel()
        result = epi.disease_spread_network(n_nodes=200, n_edges=500, initial_infected=5, transmission_prob=0.2, recovery_prob=0.03, days=90, seed=123)
        assert result["n_nodes"] == 200
        assert result["average_degree"] > 0
        assert result["parameters"]["transmission_prob"] == 0.2
        assert result["parameters"]["recovery_prob"] == 0.03

    def test_high_transmission_spreads_faster(self):
        epi = EpidemiologyModel()
        low_trans = epi.disease_spread_network(n_nodes=100, n_edges=200, transmission_prob=0.05, recovery_prob=0.1, days=60, seed=42)
        high_trans = epi.disease_spread_network(n_nodes=100, n_edges=200, transmission_prob=0.5, recovery_prob=0.1, days=60, seed=42)
        assert high_trans["peak_infected"] > low_trans["peak_infected"]

    def test_reproducibility_with_seed(self):
        epi = EpidemiologyModel()
        r1 = epi.disease_spread_network(seed=42)
        r2 = epi.disease_spread_network(seed=42)
        assert r1["peak_infected"] == r2["peak_infected"]

    def test_custom_parameters(self):
        epi = EpidemiologyModel()
        result = epi.disease_spread_network(n_nodes=500, n_edges=1000, initial_infected=10, transmission_prob=0.2, recovery_prob=0.08, days=100, seed=99)
        assert result["n_nodes"] == 500
        assert result["n_edges"] == 1000
        assert result["peak_infected"] > 0

    def test_seir_zero_beta(self):
        epi = EpidemiologyModel()
        result = epi.seir(population=100000, initial_infected=10, beta=0.01, days=50)
        assert result["peak_infected"] < 1000

    def test_seir_very_high_beta(self):
        epi = EpidemiologyModel()
        result = epi.seir(population=100000, initial_infected=10, beta=0.9, days=50)
        assert result["peak_infected"] > 1000

    def test_health_risk_young_healthy(self):
        epi = EpidemiologyModel()
        result = epi.health_risk_score(age=25, bmi=22.0, smoking=False, diabetes=False, hypertension=False)
        assert result["total_risk_score"] < 0.2
        assert result["severity"] in ["low", "minimal"]

    def test_health_risk_elderly_smoker(self):
        epi = EpidemiologyModel()
        result = epi.health_risk_score(age=75, bmi=30.0, smoking=True, diabetes=True, hypertension=True)
        young_healthy = epi.health_risk_score(age=25, bmi=22.0, smoking=False, diabetes=False, hypertension=False)
        assert result["total_risk_score"] > young_healthy["total_risk_score"]

    def test_health_risk_bmi_extremes(self):
        epi = EpidemiologyModel()
        low_bmi = epi.health_risk_score(age=40, bmi=20.0)
        high_bmi = epi.health_risk_score(age=40, bmi=40.0)
        assert high_bmi["total_risk_score"] > low_bmi["total_risk_score"]

    def test_pandemic_high_ifr(self):
        epi = EpidemiologyModel()
        result = epi.pandemic_scenarios(population=1e6, base_r0=3.0, ifr=0.02)
        assert "no_intervention" in result["scenarios"]
        assert result["scenarios"]["no_intervention"]["total_deceased"] > result["scenarios"]["moderate_lockdown"]["total_deceased"]

    def test_seir_with_vaccination_effect(self):
        epi = EpidemiologyModel()
        no_vax = epi.seir(population=1e6, initial_infected=100, beta=0.3, days=200, vaccination_rate=0.0)
        with_vax = epi.seir(population=1e6, initial_infected=100, beta=0.3, days=200, vaccination_rate=0.5, vaccine_efficacy=0.9)
        assert with_vax["peak_infected"] < no_vax["peak_infected"]

    def test_seir_intervention_reduces_peak(self):
        epi = EpidemiologyModel()
        no_int = epi.seir(population=1e6, initial_infected=100, beta=0.5, days=200)
        with_int = epi.seir(population=1e6, initial_infected=100, beta=0.5, days=200, intervention_day=20, intervention_beta=0.15)
        assert with_int["peak_infected"] < no_int["peak_infected"]

    def test_network_sparse_vs_dense(self):
        epi = EpidemiologyModel()
        sparse = epi.disease_spread_network(n_nodes=100, n_edges=100, transmission_prob=0.2, days=60, seed=42)
        dense = epi.disease_spread_network(n_nodes=100, n_edges=1000, transmission_prob=0.2, days=60, seed=42)
        assert dense["peak_infected"] > sparse["peak_infected"]


class TestInterventionOptimizer:
    def test_default_optimizer(self):
        epi = EpidemiologyModel()
        result = epi.intervention_optimizer(days=180)
        assert "baseline" in result
        assert "optimal_intervention" in result
        assert "ranking" in result
        assert "tradeoffs" in result
        assert "recommendation" in result
        assert "deaths" in result["baseline"]
        assert "peak_infected" in result["baseline"]

    def test_optimal_intervention(self):
        epi = EpidemiologyModel()
        result = epi.intervention_optimizer(days=180)
        opt = result["optimal_intervention"]
        assert opt is not None
        assert "intervention_day" in opt
        assert "intervention_beta" in opt
        assert "intervention_strength" in opt
        assert "deaths_prevented" in opt
        assert "peak_reduction_pct" in opt

    def test_ranking_sorted(self):
        epi = EpidemiologyModel()
        result = epi.intervention_optimizer(days=180)
        scores = [r["score"] for r in result["ranking"]]
        assert scores == sorted(scores)

    def test_early_intervention_better(self):
        epi = EpidemiologyModel()
        result = epi.intervention_optimizer(candidate_intervention_days=[7, 60, 90], candidate_intervention_betas=[0.03, 0.15], days=180)
        early = [r for r in result["ranking"] if r.get("intervention_day") == 7]
        late = [r for r in result["ranking"] if r.get("intervention_day") == 90]
        if early and late:
            assert early[0]["deaths_prevented"] >= late[0]["deaths_prevented"]

    def test_tradeoffs(self):
        epi = EpidemiologyModel()
        result = epi.intervention_optimizer(days=180)
        tradeoffs = result["tradeoffs"]
        assert "early_aggressive" in tradeoffs
        assert "delayed_moderate" in tradeoffs
        assert "vaccination_focused" in tradeoffs
        for key in tradeoffs:
            assert "description" in tradeoffs[key]
            assert "deaths_prevented" in tradeoffs[key]
            assert "constraint" in tradeoffs[key]

    def test_custom_candidates(self):
        epi = EpidemiologyModel()
        result = epi.intervention_optimizer(
            candidate_intervention_days=[14, 30],
            candidate_intervention_betas=[0.05, 0.15],
            vaccination_rates=[0.5],
            days=180,
        )
        assert result["baseline"]["deaths"] > 0
        assert len(result["ranking"]) > 0

    def test_recommendation_string(self):
        epi = EpidemiologyModel()
        result = epi.intervention_optimizer(days=180)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0