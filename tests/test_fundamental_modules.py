"""Tests for fundamental improvement modules: scenario engine, calibration, adversarial stress, narrative, paradox detection."""
import pytest
import numpy as np
from oracle.prediction.scenario_engine import (
    CausalLoopDiagram, SystemDynamicsModel, CrossDomainFeedback,
    ScenarioSimulator, SimulationResult, ScenarioResult
)
from oracle.prediction.calibration import (
    CalibrationTracker, CalibrationCurveResult, DomainCalibrationResult,
    ForecastAudit, PredictionAuditor
)
from oracle.prediction.adversarial_stress import (
    AdversarialStressTester, StressScenario, SystemicRiskResult
)
from oracle.prediction.narrative import (
    NarrativeEngine, EmergentNarrative, NarrativeSegment, StoryArc, Severity
)
from oracle.prediction.paradox_detector import (
    ParadoxDetector, Paradox, ParadoxSeverity, ParadoxReport, PhaseTransitionDetector,
    PhaseTransition, PhaseTransitionReport
)


class TestCausalLoopDiagram:
    def test_add_link(self):
        cld = CausalLoopDiagram()
        cld.add_link("population", "births", 1.0)
        cld.add_link("population", "deaths", -1.0)
        assert len(cld.links) == 2
        assert len(cld.variables) == 3

    def test_detect_loops(self):
        cld = CausalLoopDiagram()
        cld.add_link("population", "births", 0.3)
        cld.add_link("births", "population", 0.5)
        loops = cld.detect_loops()
        assert isinstance(loops, list)

    def test_feedback_matrix(self):
        cld = CausalLoopDiagram()
        cld.add_link("a", "b", 0.5)
        cld.add_link("b", "c", 0.3)
        cld.add_link("c", "a", 0.2)
        matrix = cld.compute_feedback_matrix()
        assert matrix.shape[0] == 3
        assert matrix.shape[1] == 3

    def test_analyze_leverage(self):
        cld = CausalLoopDiagram()
        cld.add_link("a", "b", 0.8)
        cld.add_link("b", "c", 0.5)
        cld.add_link("c", "d", 0.3)
        result = cld.analyze_lease_and_effect()
        assert "centrality" in result
        assert "leverage_points" in result


class TestSystemDynamicsModel:
    def test_simple_simulation(self):
        sdm = SystemDynamicsModel(dt=0.1)
        sdm.add_stock("population", 100.0)
        sdm.add_flow("growth", None, "population", "0.02 * population", {"population": 100.0})
        result = sdm.simulate(n_steps=50)
        assert isinstance(result, SimulationResult)
        assert "population" in result.time_series
        assert len(result.time_series["population"]) == 51
        assert result.stocks_final["population"] > 100.0

    def test_equilibrium_detection(self):
        sdm = SystemDynamicsModel(dt=0.05)
        sdm.add_stock("water_level", 50.0)
        sdm.add_flow("drain", None, "water_level", "-0.1 * population", {})
        result = sdm.simulate(n_steps=50)
        assert result.method == "system_dynamics"
        assert "converged" in result.equilibrium_analysis

    def test_with_bounds(self):
        sdm = SystemDynamicsModel(dt=0.1)
        sdm.add_stock("population", 10.0, max_value=200.0)
        sdm.add_flow("growth", None, "population", "0.3 * population", {"population": 10.0})
        result = sdm.simulate(n_steps=100)
        assert result.stocks_final["population"] <= 200.0

    def test_sensitivity_scores(self):
        sdm = SystemDynamicsModel(dt=0.1)
        sdm.add_stock("x", 1.0)
        sdm.add_stock("y", 0.5)
        sdm.add_flow("x_flow", None, "x", "0.1 * x", {"x": 1.0})
        sdm.add_flow("y_flow", None, "y", "0.05 * y", {"y": 0.5})
        result = sdm.simulate(n_steps=30)
        assert "x" in result.sensitivity_scores
        assert "y" in result.sensitivity_scores


class TestCrossDomainFeedback:
    def test_propagate(self):
        cdf = CrossDomainFeedback()
        domain_values = {"climate": 0.7, "energy": 0.5, "geopolitical": 0.6}
        result = cdf.propagate(domain_values)
        assert "adjusted_domains" in result
        assert "cross_effects" in result

    def test_add_rule(self):
        cdf = CrossDomainFeedback()
        cdf.add_rule("test_source", "test_target", "test_var", 0.5, "test rule")
        assert len(cdf.feedback_rules) > 15

    def test_default_rules_exist(self):
        cdf = CrossDomainFeedback()
        assert len(cdf.feedback_rules) >= 14


class TestScenarioSimulator:
    def test_simulate_scenario(self):
        sim = ScenarioSimulator()
        result = sim.simulate_scenario(
            name="test_scenario",
            assumptions={"growth_rate": 0.02},
            domain_values={"gdp_growth": 0.03, "unemployment_rate": 0.05},
            n_steps=20,
        )
        assert isinstance(result, ScenarioResult)
        assert result.scenario_name == "test_scenario"
        assert result.stability in ("stable", "unstable", "oscillating")
        assert result.method == "scenario_simulation"

    def test_compare_scenarios(self):
        sim = ScenarioSimulator()
        sim.simulate_scenario("baseline", {"rate": 0.01}, {"gdp_growth": 0.02})
        sim.simulate_scenario("stress", {"rate": 0.05}, {"gdp_growth": -0.01})
        comparison = sim.compare_scenarios(["baseline", "stress"])
        assert "scenarios" in comparison
        assert "divergence" in comparison

    def test_scenario_hash(self):
        sim = ScenarioSimulator()
        r1 = sim.simulate_scenario("s1", {"a": 1.0}, {"x": 0.5})
        assert len(r1.scenario_hash) == 16


class TestCalibrationTracker:
    def test_record(self):
        tracker = CalibrationTracker()
        rec = tracker.record("economics", "gdp_growth", 0.025, 0.01, 0.04, 0.022)
        assert rec.hit is True
        assert abs(rec.error - (-0.003)) < 0.001

    def test_record_miss(self):
        tracker = CalibrationTracker()
        rec = tracker.record("climate", "temp_anomaly", 1.5, 1.0, 1.4, 1.8)
        assert rec.hit is False

    def test_calibration_curve(self):
        tracker = CalibrationTracker()
        rng = np.random.RandomState(42)
        for i in range(50):
            predicted = rng.normal(0, 1)
            ci_width = 3.92
            actual = predicted + rng.normal(0, 0.5)
            tracker.record("test", "var", predicted, predicted - ci_width/2, predicted + ci_width/2, actual)
        curve = tracker.calibration_curve("test")
        assert isinstance(curve, CalibrationCurveResult)
        assert curve.n_observations == 50

    def test_domain_calibration(self):
        tracker = CalibrationTracker()
        rng = np.random.RandomState(42)
        for i in range(30):
            predicted = rng.normal(0, 1)
            actual = predicted + rng.normal(0, 0.3)
            tracker.record("econ", "gdp", predicted, predicted - 1.96, predicted + 1.96, actual)
        result = tracker.domain_calibration("econ")
        assert isinstance(result, DomainCalibrationResult)
        assert result.n_forecasts == 30
        assert 0.0 <= result.coverage_95 <= 1.0

    def test_calibrate_ci(self):
        tracker = CalibrationTracker()
        rng = np.random.RandomState(42)
        for i in range(30):
            predicted = rng.normal(0, 1)
            actual = predicted + rng.normal(0, 2.0)
            tracker.record("econ", "gdp", predicted, predicted - 1.96, predicted + 1.96, actual)
        lower, upper = tracker.calibrate_ci("econ", -1.96, 1.96, "HIGH")
        assert upper - lower >= 3.92

    def test_summary(self):
        tracker = CalibrationTracker()
        tracker.record("econ", "gdp", 0.02, -0.02, 0.06, 0.015)
        summary = tracker.summary()
        assert "total_forecasts" in summary
        assert summary["total_forecasts"] == 1


class TestPredictionAuditor:
    def test_create_audit(self):
        auditor = PredictionAuditor()
        audit = auditor.create_audit("economics", "gdp_growth", 0.025, 0.01, 0.04,
                                      "HIGH", "dalio_cycle", methodology="deterministic",
                                      state_hash="abc123")
        assert audit.prediction_id.startswith("P-")
        assert len(audit.prediction_hash) == 16

    def test_detect_drift(self):
        auditor = PredictionAuditor()
        for i in range(20):
            auditor.create_audit("econ", "gdp", 0.02 + i * 0.001, 0.0, 0.04, "HIGH", "model_a")
        drift = auditor.detect_drift("econ", "gdp", window=5)
        assert "drift_detected" in drift
        assert "drift_magnitude" in drift

    def test_lineage_trace(self):
        auditor = PredictionAuditor()
        audit = auditor.create_audit("econ", "gdp", 0.02, -0.01, 0.05, "HIGH", "model", lineage=["model_a", "model_b"])
        result = auditor.lineage_trace(audit.prediction_id)
        assert result["lineage"] == ["model_a", "model_b"]

    def test_integrity(self):
        auditor = PredictionAuditor()
        auditor.create_audit("econ", "gdp", 0.02, -0.01, 0.05, "HIGH", "model")
        report = auditor.verify_integrity()
        assert report["total_audits"] == 1


class TestAdversarialStressTester:
    def test_worst_case(self):
        tester = AdversarialStressTester(n_simulations=100, n_generations=5, population_size=10, seed=42)
        worst = tester.find_worst_case(domains=["geopolitical_tension", "climate_anomaly_c", "pandemic_risk"])
        assert isinstance(worst, StressScenario)
        assert worst.composite_risk > 0
        assert len(worst.domain_shocks) >= 3
        assert worst.method == "genetic_adversarial"

    def test_monte_carlo_stress(self):
        tester = AdversarialStressTester(n_simulations=200, seed=42)
        result = tester.monte_carlo_stress(domains=["geopolitical_tension", "climate_anomaly_c", "gdp_growth"])
        assert isinstance(result, SystemicRiskResult)
        assert result.worst_case is not None
        assert len(result.risk_surface) >= 3
        assert result.systemic_risk_score > 0

    def test_black_swan(self):
        tester = AdversarialStressTester(n_simulations=500, seed=42)
        swans = tester.black_swan_scenarios(n_scenarios=3)
        assert len(swans) >= 1
        assert all(s.composite_risk > 0 for s in swans)

    def test_cascade_amplifiers(self):
        tester = AdversarialStressTester(seed=42)
        assert len(tester.CASCADE_AMPLIFIERS) >= 8


class TestNarrativeEngine:
    def test_crisis_narrative(self):
        engine = NarrativeEngine()
        domain_states = {
            "geopolitical_tension": {"us_china": 0.8},
            "climate_anomaly_c": {"temp": 2.5},
            "inflation_rate": {"rate": 0.08},
            "unemployment_rate": {"rate": 0.08},
        }
        narrative = engine.generate_narrative(domain_states)
        assert isinstance(narrative, EmergentNarrative)
        assert narrative.story_arc in [StoryArc.CRISIS, StoryArc.CASCADE_FAILURE, StoryArc.TIPPING_POINT, StoryArc.STAGNATION]
        assert len(narrative.segments) > 0
        assert narrative.narrative_confidence > 0

    def test_recovery_narrative(self):
        engine = NarrativeEngine()
        domain_states = {
            "gdp_growth": {"rate": 0.04},
            "unemployment_rate": {"rate": 0.03},
            "inflation_rate": {"rate": 0.02},
            "geopolitical_tension": {"tension": 0.2},
        }
        narrative = engine.generate_narrative(domain_states)
        assert narrative.story_arc in [StoryArc.RECOVERY, StoryArc.BOOM, StoryArc.RESILIENT_GROWTH, StoryArc.STAGNATION]

    def test_segment_generation(self):
        engine = NarrativeEngine()
        segment = engine._generate_segment("climate", "worsening", 0.8, {"temp_anomaly": 0.8})
        assert isinstance(segment, NarrativeSegment)
        assert segment.domain == "climate"
        assert segment.severity in (Severity.HIGH, Severity.CRITICAL, Severity.EXTREME)

    def test_classify_domain(self):
        engine = NarrativeEngine()
        direction, mag = engine._classify_domain("geopolitics", {"tension": 0.8})
        assert direction in ("worsening", "elevated", "improving", "stable", "moderate", "low")

    def test_severity_mapping(self):
        engine = NarrativeEngine()
        assert engine._severity(0.95) == Severity.EXTREME
        assert engine._severity(0.75) == Severity.CRITICAL
        assert engine._severity(0.55) == Severity.HIGH
        assert engine._severity(0.35) == Severity.MODERATE
        assert engine._severity(0.15) == Severity.LOW


class TestParadoxDetector:
    def test_stagflation_paradox(self):
        detector = ParadoxDetector()
        report = detector.detect({
            "inflation_rate": 0.08, "gdp_growth": -0.01, "unemployment_rate": 0.08,
            "vix_level": 25, "geopolitical_tension": 0.3,
        })
        assert isinstance(report, ParadoxReport)
        paradox_names = [p.name for p in report.paradoxes]
        assert "Stagflation paradox" in paradox_names

    def test_no_paradox(self):
        detector = ParadoxDetector()
        report = detector.detect({
            "inflation_rate": 0.02, "gdp_growth": 0.03, "unemployment_rate": 0.04,
            "vix_level": 15,
        })
        assert report.consistency_score >= 0.5

    def test_add_custom_rule(self):
        detector = ParadoxDetector()
        detector.add_rule("Custom paradox", lambda v: v.get("x", 0) > 10, "Custom rule", ParadoxSeverity.MODERATE, ["Just because"])
        report = detector.detect({"x": 15})
        custom_found = any(p.name == "Custom paradox" for p in report.paradoxes)
        assert custom_found

    def test_consistency_score(self):
        detector = ParadoxDetector()
        report = detector.detect({"inflation_rate": 0.02, "gdp_growth": 0.03})
        assert 0.0 <= report.consistency_score <= 1.0


class TestPhaseTransitionDetector:
    def test_no_transition(self):
        detector = PhaseTransitionDetector()
        series = np.random.RandomState(42).normal(0, 1, 100)
        result = detector.detect(series, "test_var")
        assert isinstance(result, PhaseTransitionReport)
        assert result.n_regimes >= 1

    def test_single_transition(self):
        detector = PhaseTransitionDetector()
        series = np.concatenate([np.zeros(50) + 1.0, np.zeros(50) + 5.0])
        result = detector.detect(series, "shifted_var")
        assert len(result.transitions) >= 1
        assert result.transitions[0].before_mean < result.transitions[0].after_mean

    def test_order_parameter(self):
        detector = PhaseTransitionDetector()
        stable = np.random.RandomState(42).normal(0, 0.1, 100)
        result = detector.detect(stable, "stable")
        assert result.order_parameter >= 0