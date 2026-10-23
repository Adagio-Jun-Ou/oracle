"""Tests for Stress Testing, Robust Optimizer, and Red Team Forecaster."""
import pytest
import numpy as np
from oracle.prediction.stress_testing import (
    StressTestingEngine, StressScenario, StressScenarioType,
    StressTestResult, StressTestReport, HISTORICAL_SCENARIOS,
)
from oracle.prediction.robust_optimizer import (
    RobustOptimizer, RobustMethod, RobustSolution,
    UncertaintySet, RobustOptimizationResult,
)
from oracle.prediction.red_team import (
    RedTeamForecaster, RedTeamResult, ChallengeType,
    AdversarialChallenge, BiasFinding, AssumptionRupture,
    COGNITIVE_BIASES, BASE_RATE_DATA,
)


SAMPLE_PREDICTIONS = {
    "gdp_growth": 0.5,
    "stock_market": 0.7,
    "inflation": 0.3,
    "unemployment": 0.2,
    "oil_price": 0.4,
}


class TestStressTestingEngine:
    def test_initialization_default(self):
        engine = StressTestingEngine()
        assert len(engine.scenarios) == len(HISTORICAL_SCENARIOS)

    def test_initialization_custom_scenarios(self):
        custom = [{"name": "custom1", "description": "Custom test",
                    "shocks": {"gdp_growth": -0.1}, "probability": 0.1, "severity": 0.5}]
        engine = StressTestingEngine(scenarios=custom)
        assert len(engine.scenarios) == 1
        assert engine.scenarios[0].name == "custom1"

    def test_initialization_with_stress_scenario_objects(self):
        sc = StressScenario(
            name="obj_test", scenario_type=StressScenarioType.COMPOUND,
            description="Object test", parameter_shocks={"x": 0.5},
            probability=0.1, severity=0.5,
        )
        engine = StressTestingEngine(scenarios=[sc])
        assert len(engine.scenarios) == 1
        assert engine.scenarios[0].scenario_type == StressScenarioType.COMPOUND

    def test_stress_test_basic(self):
        engine = StressTestingEngine()
        report = engine.stress_test(SAMPLE_PREDICTIONS)
        assert isinstance(report, StressTestReport)
        assert len(report.results) > 0
        assert 0 <= report.overall_vulnerability <= 1
        assert 0 <= report.resilience_score <= 1
        assert report.worst_case_scenario != "none"

    def test_stress_test_with_sensitivities(self):
        engine = StressTestingEngine()
        sensitivities = {"gdp_growth": 0.3, "stock_market": 0.5}
        report = engine.stress_test(SAMPLE_PREDICTIONS, sensitivities=sensitivities)
        assert len(report.results) > len(HISTORICAL_SCENARIOS)

    def test_stress_test_threshold(self):
        engine = StressTestingEngine()
        report_low = engine.stress_test(SAMPLE_PREDICTIONS, threshold=0.1)
        report_high = engine.stress_test(SAMPLE_PREDICTIONS, threshold=0.9)
        failed_low = sum(1 for r in report_low.results if not r.passed)
        failed_high = sum(1 for r in report_high.results if not r.passed)
        assert failed_low >= failed_high

    def test_apply_stress_negative_shock(self):
        engine = StressTestingEngine()
        scenario = StressScenario(
            name="neg_shock", scenario_type=StressScenarioType.HISTORICAL,
            description="Negative shock", parameter_shocks={"gdp_growth": -0.5},
            probability=0.1, severity=0.8,
        )
        result = engine._apply_stress(SAMPLE_PREDICTIONS, scenario, 0.5)
        assert result.stressed_prediction["gdp_growth"] < SAMPLE_PREDICTIONS["gdp_growth"]
        assert result.impact["gdp_growth"] > 0

    def test_apply_stress_positive_shock(self):
        engine = StressTestingEngine()
        scenario = StressScenario(
            name="pos_shock", scenario_type=StressScenarioType.COMPOUND,
            description="Positive shock", parameter_shocks={"inflation": 0.5},
            probability=0.1, severity=0.3,
        )
        result = engine._apply_stress(SAMPLE_PREDICTIONS, scenario, 0.5)
        assert result.stressed_prediction["inflation"] > SAMPLE_PREDICTIONS["inflation"]

    def test_apply_stress_no_shock_variable(self):
        engine = StressTestingEngine()
        scenario = StressScenario(
            name="partial", scenario_type=StressScenarioType.SENSITIVITY,
            description="Partial shock", parameter_shocks={"gdp_growth": -0.3},
            probability=0.1, severity=0.5,
        )
        result = engine._apply_stress(SAMPLE_PREDICTIONS, scenario, 0.5)
        assert result.stressed_prediction["stock_market"] == SAMPLE_PREDICTIONS["stock_market"]
        assert result.impact["stock_market"] == 0.0

    def test_sensitivity_stress_no_sensitivities(self):
        engine = StressTestingEngine()
        results = engine._sensitivity_stress(SAMPLE_PREDICTIONS, None, 0.5)
        assert len(results) == 5
        assert all(r.scenario.scenario_type == StressScenarioType.SENSITIVITY for r in results)

    def test_sensitivity_stress_with_sensitivities(self):
        engine = StressTestingEngine()
        sens = {"gdp_growth": 0.4, "inflation": 0.2}
        results = engine._sensitivity_stress(SAMPLE_PREDICTIONS, sens, 0.5)
        assert len(results) == 2

    def test_reverse_stress(self):
        engine = StressTestingEngine()
        results = engine._reverse_stress(SAMPLE_PREDICTIONS, 0.5)
        assert len(results) == 5
        assert all(r.scenario.scenario_type == StressScenarioType.REVERSE for r in results)

    def test_tail_risk_stress(self):
        engine = StressTestingEngine()
        results = engine._tail_risk_stress(SAMPLE_PREDICTIONS, 0.5)
        assert len(results) == 15
        assert all(r.scenario.scenario_type == StressScenarioType.TAIL_RISK for r in results)

    def test_vulnerability_computation(self):
        engine = StressTestingEngine()
        report = engine.stress_test(SAMPLE_PREDICTIONS)
        assert isinstance(report.most_vulnerable_variables, list)

    def test_recommendations_generated(self):
        engine = StressTestingEngine()
        report = engine.stress_test(SAMPLE_PREDICTIONS)
        assert len(report.recommendations) > 0

    def test_stress_clamps_values(self):
        engine = StressTestingEngine()
        extreme_preds = {"x": 100.0}
        scenario = StressScenario(
            name="extreme", scenario_type=StressScenarioType.COMPOUND,
            description="Extreme", parameter_shocks={"x": -2.0},
            probability=0.01, severity=0.9,
        )
        result = engine._apply_stress(extreme_preds, scenario, 0.5)
        assert result.stressed_prediction["x"] >= -1.0
        assert result.stressed_prediction["x"] <= 2.0

    def test_empty_predictions(self):
        engine = StressTestingEngine()
        report = engine.stress_test({})
        assert isinstance(report, StressTestReport)

    def test_scenario_type_enum(self):
        assert StressScenarioType.HISTORICAL.value == "historical"
        assert StressScenarioType.REVERSE.value == "reverse"
        assert StressScenarioType.SENSITIVITY.value == "sensitivity"
        assert StressScenarioType.TAIL_RISK.value == "tail_risk"
        assert StressScenarioType.COMPOUND.value == "compound"

    def test_all_historical_scenarios_loaded(self):
        engine = StressTestingEngine()
        names = [s.name for s in engine.scenarios]
        assert "2008_financial_crisis" in names
        assert "covid_19_shock" in names
        assert "black_monday_1987" in names
        assert "climate_disaster" in names

    def test_result_passed_field(self):
        engine = StressTestingEngine()
        scenario = StressScenario(
            name="mild", scenario_type=StressScenarioType.SENSITIVITY,
            description="Mild shock", parameter_shocks={"gdp_growth": -0.01},
            probability=0.5, severity=0.1,
        )
        result = engine._apply_stress(SAMPLE_PREDICTIONS, scenario, 0.5)
        assert result.passed is True

    def test_historical_precedent(self):
        engine = StressTestingEngine()
        hist = [s for s in engine.scenarios if s.historical_precedent]
        assert len(hist) > 0


class TestRobustOptimizer:
    def test_initialization(self):
        opt = RobustOptimizer()
        assert opt.n_scenarios == 50
        assert opt.n_iterations == 200
        assert opt.uncertainty_radius == 0.1

    def test_minimax_optimize(self):
        opt = RobustOptimizer(n_scenarios=10, n_iterations=30, uncertainty_radius=0.2)
        result = opt.optimize(SAMPLE_PREDICTIONS, method=RobustMethod.MINIMAX)
        assert isinstance(result, RobustOptimizationResult)
        assert result.best_solution.method == RobustMethod.MINIMAX
        assert result.best_solution.worst_case_loss >= 0
        assert result.robustness_score >= 0
        assert result.price_of_robustness >= 0

    def test_dro_optimize(self):
        opt = RobustOptimizer(n_scenarios=10, n_iterations=30, uncertainty_radius=0.2)
        result = opt.optimize(SAMPLE_PREDICTIONS, method=RobustMethod.DISTRIBUTIONALLY_ROBUST)
        assert result.best_solution.method == RobustMethod.DISTRIBUTIONALLY_ROBUST

    def test_robust_bayesian_optimize(self):
        opt = RobustOptimizer(n_scenarios=10, n_iterations=30, uncertainty_radius=0.2)
        result = opt.optimize(SAMPLE_PREDICTIONS, method=RobustMethod.ROBUST_BAYESIAN)
        assert result.best_solution.method == RobustMethod.ROBUST_BAYESIAN

    def test_scenario_robust_optimize(self):
        opt = RobustOptimizer(n_scenarios=10, n_iterations=30, uncertainty_radius=0.2)
        result = opt.optimize(SAMPLE_PREDICTIONS, method=RobustMethod.SCENARIO_ROBUST)
        assert result.best_solution.method == RobustMethod.SCENARIO_ROBUST

    def test_pareto_robust_optimize(self):
        opt = RobustOptimizer(n_scenarios=10, n_iterations=30, uncertainty_radius=0.2)
        result = opt.optimize(SAMPLE_PREDICTIONS, method=RobustMethod.PARETO_ROBUST)
        assert result.best_solution.method == RobustMethod.PARETO_ROBUST

    def test_build_uncertainty_set(self):
        opt = RobustOptimizer(uncertainty_radius=0.2)
        u_set = opt._build_uncertainty_set(SAMPLE_PREDICTIONS, None)
        assert set(u_set.variables) == set(SAMPLE_PREDICTIONS.keys())
        for var in SAMPLE_PREDICTIONS:
            lo, hi = u_set.bounds[var]
            assert lo <= SAMPLE_PREDICTIONS[var] <= hi

    def test_constraints(self):
        opt = RobustOptimizer(n_scenarios=5, n_iterations=20)
        constraints = {"gdp_growth": (0.45, 0.55)}
        u_set = opt._build_uncertainty_set(SAMPLE_PREDICTIONS, constraints)
        lo, hi = u_set.bounds["gdp_growth"]
        assert lo >= 0.45
        assert hi <= 0.55

    def test_default_loss(self):
        opt = RobustOptimizer()
        decision = {"a": 1.0, "b": 2.0}
        realization = {"a": 1.5, "b": 1.5}
        loss = opt._default_loss(decision, realization)
        assert loss > 0
        assert abs(loss - (0.25 + 0.25)) < 1e-10

    def test_custom_loss_function(self):
        opt = RobustOptimizer(n_scenarios=5, n_iterations=20)
        def custom_loss(d, r):
            return sum(abs(d.get(k, 0) - r.get(k, 0)) for k in d)
        result = opt.optimize({"x": 0.5}, loss_function=custom_loss, method=RobustMethod.MINIMAX)
        assert result.best_solution.worst_case_loss >= 0

    def test_multi_method_solutions(self):
        opt = RobustOptimizer(n_scenarios=5, n_iterations=20)
        result = opt.optimize(SAMPLE_PREDICTIONS, method=RobustMethod.MINIMAX)
        assert len(result.all_solutions) == 4

    def test_recommendations(self):
        opt = RobustOptimizer(n_scenarios=5, n_iterations=20)
        result = opt.optimize(SAMPLE_PREDICTIONS)
        assert len(result.recommendations) > 0

    def test_robustness_score_range(self):
        opt = RobustOptimizer(n_scenarios=5, n_iterations=20)
        result = opt.optimize(SAMPLE_PREDICTIONS)
        assert 0 <= result.robustness_score <= 1

    def test_worst_case_gte_nominal(self):
        opt = RobustOptimizer(n_scenarios=10, n_iterations=30)
        result = opt.optimize(SAMPLE_PREDICTIONS, method=RobustMethod.MINIMAX)
        assert result.worst_case_performance >= result.nominal_performance or abs(result.worst_case_performance - result.nominal_performance) < 0.01

    def test_scenario_losses(self):
        opt = RobustOptimizer(n_scenarios=5, n_iterations=20, uncertainty_radius=0.3)
        result = opt.optimize(SAMPLE_PREDICTIONS, method=RobustMethod.SCENARIO_ROBUST)
        assert len(result.best_solution.scenario_losses) > 0

    def test_empty_predictions(self):
        opt = RobustOptimizer(n_scenarios=5, n_iterations=20)
        result = opt.optimize({})
        assert isinstance(result, RobustOptimizationResult)

    def test_robust_method_enum(self):
        assert RobustMethod.MINIMAX.value == "minimax"
        assert RobustMethod.DISTRIBUTIONALLY_ROBUST.value == "distributionally_robust"
        assert RobustMethod.ROBUST_BAYESIAN.value == "robust_bayesian"
        assert RobustMethod.PARETO_ROBUST.value == "pareto_robust"
        assert RobustMethod.SCENARIO_ROBUST.value == "scenario_robust"


class TestRedTeamForecaster:
    def test_initialization(self):
        rt = RedTeamForecaster()
        assert rt.n_adversarial_scenarios == 10

    def test_red_team_basic(self):
        rt = RedTeamForecaster(n_adversarial_scenarios=3)
        result = rt.red_team(SAMPLE_PREDICTIONS)
        assert isinstance(result, RedTeamResult)
        assert len(result.challenges) > 0
        assert 0 <= result.overall_vulnerability <= 1
        assert 0 <= result.confidence_adjustment <= 1

    def test_bias_decomposition(self):
        rt = RedTeamForecaster()
        biases = rt._decompose_biases(SAMPLE_PREDICTIONS, 0.9)
        assert isinstance(biases, list)
        assert len(biases) > 0
        for b in biases:
            assert isinstance(b, BiasFinding)
            assert b.estimated_bias_magnitude > 0

    def test_base_rate_neglect_with_rates(self):
        rt = RedTeamForecaster()
        base_rates = {"gdp_recession": 0.5, "market_crash_20pct": 0.01}
        challenges = rt._check_base_rate_neglect(SAMPLE_PREDICTIONS, base_rates)
        assert isinstance(challenges, list)
        for c in challenges:
            assert c.challenge_type == ChallengeType.BASE_RATE_NEGLECT

    def test_base_rate_neglect_no_rates(self):
        rt = RedTeamForecaster()
        challenges = rt._check_base_rate_neglect(SAMPLE_PREDICTIONS, None)
        assert isinstance(challenges, list)
        assert len(challenges) == 0

    def test_adversarial_scenarios(self):
        rt = RedTeamForecaster(n_adversarial_scenarios=5)
        scenarios = rt._generate_adversarial_scenarios(SAMPLE_PREDICTIONS)
        assert len(scenarios) <= 5
        for s in scenarios:
            assert s.challenge_type == ChallengeType.ADVERSARIAL_SCENARIO

    def test_assumption_rupture_default(self):
        rt = RedTeamForecaster()
        ruptures = rt._rupture_assumptions(SAMPLE_PREDICTIONS, None)
        assert len(ruptures) == 5
        for r in ruptures:
            assert isinstance(r, AssumptionRupture)
            assert r.probability > 0
            assert r.impact > 0

    def test_assumption_rupture_custom(self):
        rt = RedTeamForecaster()
        assumptions = ["AI surpasses human intelligence", "Fusion energy breakthrough"]
        ruptures = rt._rupture_assumptions(SAMPLE_PREDICTIONS, assumptions)
        assert len(ruptures) == 2

    def test_contrarian_arguments(self):
        rt = RedTeamForecaster()
        contrarian = rt._generate_contrarian_arguments(SAMPLE_PREDICTIONS)
        assert len(contrarian) <= 5
        for c in contrarian:
            assert c.challenge_type == ChallengeType.CONTRARIAN_ARGUMENT

    def test_overconfidence_detection_high(self):
        rt = RedTeamForecaster()
        challenges = rt._detect_overconfidence(SAMPLE_PREDICTIONS, 0.95)
        assert len(challenges) > 0
        assert any(c.challenge_type == ChallengeType.OVERCONFIDENCE for c in challenges)

    def test_overconfidence_detection_low(self):
        rt = RedTeamForecaster()
        challenges = rt._detect_overconfidence({"x": 0.5}, 0.5)
        assert len(challenges) == 0

    def test_adjusted_prediction(self):
        rt = RedTeamForecaster(n_adversarial_scenarios=3)
        result = rt.red_team(SAMPLE_PREDICTIONS)
        for var in SAMPLE_PREDICTIONS:
            assert var in result.adjusted_prediction

    def test_confidence_adjustment(self):
        rt = RedTeamForecaster(n_adversarial_scenarios=3)
        result = rt.red_team(SAMPLE_PREDICTIONS, confidence=0.9)
        assert result.confidence_adjustment <= 0.9

    def test_key_weaknesses(self):
        rt = RedTeamForecaster(n_adversarial_scenarios=3)
        result = rt.red_team(SAMPLE_PREDICTIONS)
        assert isinstance(result.key_weaknesses, list)

    def test_recommendations(self):
        rt = RedTeamForecaster(n_adversarial_scenarios=3)
        result = rt.red_team(SAMPLE_PREDICTIONS)
        assert len(result.recommendations) > 0

    def test_cognitive_biases_constant(self):
        assert len(COGNITIVE_BIASES) == 10
        names = [b["name"] for b in COGNITIVE_BIASES]
        assert "anchoring" in names
        assert "overconfidence" in names
        assert "confirmation_bias" in names

    def test_base_rate_data_constant(self):
        assert len(BASE_RATE_DATA) == 12
        assert "gdp_recession" in BASE_RATE_DATA
        assert "pandemic" in BASE_RATE_DATA

    def test_vulnerability_scores_bounded(self):
        rt = RedTeamForecaster(n_adversarial_scenarios=3)
        for _ in range(5):
            result = rt.red_team(SAMPLE_PREDICTIONS)
            assert 0 <= result.overall_vulnerability <= 1

    def test_challenge_type_enum(self):
        assert ChallengeType.BIAS_DECOMPOSITION.value == "bias_decomposition"
        assert ChallengeType.BASE_RATE_NEGLECT.value == "base_rate_neglect"
        assert ChallengeType.ADVERSARIAL_SCENARIO.value == "adversarial_scenario"
        assert ChallengeType.ASSUMPTION_RUPTURE.value == "assumption_rupture"
        assert ChallengeType.CONTRARIAN_ARGUMENT.value == "contrarian_argument"
        assert ChallengeType.OVERCONFIDENCE.value == "overconfidence"

    def test_empty_predictions(self):
        rt = RedTeamForecaster(n_adversarial_scenarios=3)
        result = rt.red_team({})
        assert isinstance(result, RedTeamResult)

    def test_high_confidence_penalty(self):
        rt = RedTeamForecaster(n_adversarial_scenarios=3)
        result_high = rt.red_team(SAMPLE_PREDICTIONS, confidence=0.95)
        result_low = rt.red_team(SAMPLE_PREDICTIONS, confidence=0.6)
        assert result_high.confidence_adjustment < 0.95
        assert result_low.confidence_adjustment <= 0.6
        high_penalty = 0.95 - result_high.confidence_adjustment
        low_penalty = 0.6 - result_low.confidence_adjustment
        assert high_penalty >= 0
        assert low_penalty >= 0

    def test_custom_assumptions_in_result(self):
        rt = RedTeamForecaster(n_adversarial_scenarios=3)
        assumptions = ["Custom assumption A", "Custom assumption B"]
        result = rt.red_team(SAMPLE_PREDICTIONS, assumptions=assumptions)
        assert len(result.assumption_ruptures) == 2