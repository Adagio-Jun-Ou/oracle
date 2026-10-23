"""Tests for Predictive Fate, Existential Risk, and Oracle Prophecy modules."""
import pytest
import numpy as np
from oracle.prediction.predictive_fate import (
    PredictiveFateEngine, FatePrediction, TrajectoryPoint, TrajectoryPhase,
    DivergencePoint, DivergenceType, AttractorState, BifurcationPoint,
)
from oracle.prediction.existential_risk import (
    ExistentialRiskAssessor, ExistentialRiskResult, RiskAssessment,
    RiskCategory, RiskSeverity,
)
from oracle.prediction.prophecy import (
    OracleProphecySynthesizer, OracleProphecy, ProphecyOutcome,
    ProphecySignal, ModuleSignal, ProphecyConfidence, ProphecyHorizon,
)


STATE = {"technology": 0.7, "energy": 0.5, "population": 0.6,
          "economy": 0.55, "cooperation": 0.4, "knowledge": 0.65,
          "environment": 0.35, "governance": 0.45}


class TestPredictiveFateEngine:
    def test_initialization(self):
        fe = PredictiveFateEngine()
        assert fe.n_simulations == 500

    def test_project_fate(self):
        fe = PredictiveFateEngine(n_simulations=100)
        result = fe.project_fate(STATE, horizon_years=50)
        assert isinstance(result, FatePrediction)
        assert len(result.trajectory) > 0

    def test_trajectory_points(self):
        fe = PredictiveFateEngine(n_simulations=50)
        result = fe.project_fate(STATE, horizon_years=50)
        for tp in result.trajectory:
            assert isinstance(tp, TrajectoryPoint)
            assert 0 <= tp.value <= 1
            assert 0 <= tp.confidence <= 1

    def test_collapse_probability(self):
        fe = PredictiveFateEngine(n_simulations=50)
        result = fe.project_fate(STATE, horizon_years=50)
        assert 0 <= result.collapse_probability <= 1

    def test_golden_age_probability(self):
        fe = PredictiveFateEngine(n_simulations=50)
        result = fe.project_fate(STATE, horizon_years=50)
        assert 0 <= result.golden_age_probability <= 1

    def test_dominant_phase(self):
        fe = PredictiveFateEngine(n_simulations=50)
        result = fe.project_fate(STATE, horizon_years=50)
        assert isinstance(result.dominant_phase, TrajectoryPhase)

    def test_divergences(self):
        fe = PredictiveFateEngine(n_simulations=50)
        result = fe.project_fate(STATE, horizon_years=100)
        assert isinstance(result.divergences, list)

    def test_attractors(self):
        fe = PredictiveFateEngine(n_simulations=50)
        result = fe.project_fate(STATE, horizon_years=50)
        assert isinstance(result.attractors, list)
        for a in result.attractors:
            assert 0 <= a.probability <= 1

    def test_bifurcations(self):
        fe = PredictiveFateEngine(n_simulations=50)
        result = fe.project_fate(STATE, horizon_years=50)
        assert isinstance(result.bifurcations, list)

    def test_key_drivers(self):
        fe = PredictiveFateEngine(n_simulations=50)
        result = fe.project_fate(STATE, horizon_years=50)
        assert len(result.key_drivers) > 0

    def test_narrative(self):
        fe = PredictiveFateEngine(n_simulations=50)
        result = fe.project_fate(STATE, horizon_years=50)
        assert len(result.narrative) > 10

    def test_recommendations(self):
        fe = PredictiveFateEngine(n_simulations=50)
        result = fe.project_fate(STATE, horizon_years=50)
        assert len(result.recommendations) > 0

    def test_low_state_collapse(self):
        fe = PredictiveFateEngine(n_simulations=50)
        low_state = {k: 0.1 for k in STATE}
        result = fe.project_fate(low_state, horizon_years=50)
        assert result.collapse_probability > 0.1

    def test_high_state_golden_age(self):
        fe = PredictiveFateEngine(n_simulations=50)
        high_state = {k: 0.9 for k in STATE}
        result = fe.project_fate(high_state, horizon_years=50)
        assert result.golden_age_probability > 0.1

    def test_empty_state(self):
        fe = PredictiveFateEngine(n_simulations=50)
        result = fe.project_fate({}, horizon_years=10)
        assert isinstance(result, FatePrediction)

    def test_trajectory_phase_enum(self):
        assert TrajectoryPhase.GROWTH.value == "growth"
        assert TrajectoryPhase.COLLAPSE.value == "collapse"
        assert TrajectoryPhase.RENAISSANCE.value == "renaissance"

    def test_divergence_type_enum(self):
        assert DivergenceType.ACCELERATING.value == "accelerating"
        assert DivergenceType.BIFURCATION.value == "bifurcation"


class TestExistentialRiskAssessor:
    def test_initialization(self):
        er = ExistentialRiskAssessor()
        assert er.discount_rate == 0.02

    def test_assess(self):
        er = ExistentialRiskAssessor()
        result = er.assess()
        assert isinstance(result, ExistentialRiskResult)
        assert len(result.risks) > 0
        assert 0 <= result.total_existential_probability <= 1

    def test_with_domain_states(self):
        er = ExistentialRiskAssessor()
        result = er.assess({"geopolitical_risk": 0.8, "technology_risk": 0.6, "climate_risk": 0.7})
        assert result.total_existential_probability > 0

    def test_high_geo_increases_nuclear(self):
        er = ExistentialRiskAssessor()
        baseline = er.assess({"geopolitical_risk": 0.3})
        elevated = er.assess({"geopolitical_risk": 0.9})
        nuc_baseline = [r for r in baseline.risks if r.category == RiskCategory.NUCLEAR]
        nuc_elevated = [r for r in elevated.risks if r.category == RiskCategory.NUCLEAR]
        if nuc_baseline and nuc_elevated:
            assert nuc_elevated[0].annual_probability >= nuc_baseline[0].annual_probability * 0.8

    def test_risk_timeline(self):
        er = ExistentialRiskAssessor()
        result = er.assess()
        assert "10yr_existential" in result.risk_timeline
        assert "100yr_existential" in result.risk_timeline
        assert result.risk_timeline["100yr_existential"] >= result.risk_timeline["10yr_existential"]

    def test_mitigation_priorities(self):
        er = ExistentialRiskAssessor()
        result = er.assess()
        assert len(result.mitigation_priorities) > 0

    def test_narrative(self):
        er = ExistentialRiskAssessor()
        result = er.assess()
        assert len(result.narrative) > 20

    def test_recommendations(self):
        er = ExistentialRiskAssessor()
        result = er.assess()
        assert len(result.recommendations) > 0

    def test_risk_categories(self):
        er = ExistentialRiskAssessor()
        result = er.assess()
        categories = set(r.category for r in result.risks)
        assert RiskCategory.NUCLEAR in categories
        assert RiskCategory.AI_ALIGNMENT in categories
        assert RiskCategory.CLIMATE in categories

    def test_risk_category_enum(self):
        assert RiskCategory.NUCLEAR.value == "nuclear"
        assert RiskCategory.AI_ALIGNMENT.value == "ai_alignment"
        assert RiskCategory.UNKNOWN.value == "unknown"

    def test_risk_severity_enum(self):
        assert RiskSeverity.EXISTENTIAL.value == "existential"
        assert RiskSeverity.GLOBAL_CATASTROPHIC.value == "global_catastrophic"

    def test_custom_risks(self):
        er = ExistentialRiskAssessor()
        custom = [{"category": "unknown", "name": "Test Risk", "annual_prob": 0.01,
                    "severity": "global_catastrophic", "impact": 0.5, "detectability": 0.5,
                    "mitigation": 0.5, "time_to": 1.0, "trajectory": "rising", "indicators": ["test"]}]
        result = er.assess(custom_risks=custom)
        assert any(r.name == "Test Risk" for r in result.risks)


class TestOracleProphecySynthesizer:
    def test_initialization(self):
        os = OracleProphecySynthesizer()
        assert len(os.module_weights) > 0

    def test_prophesize(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"gdp_growth": 0.55, "inflation": 0.3, "stock_market": 0.7})
        assert isinstance(result, OracleProphecy)
        assert len(result.outcomes) == 3

    def test_signals_collected(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"gdp": 0.5})
        assert len(result.signals) == 1
        assert len(result.signals[0].signals) > 5

    def test_consensus_prediction(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"gdp": 0.5})
        for sig in result.signals:
            assert 0 <= sig.consensus_prediction <= 1

    def test_confidence_levels(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"x": 0.5})
        for o in result.outcomes:
            assert isinstance(o.confidence, ProphecyConfidence)

    def test_prediction_intervals(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"gdp": 0.5})
        for o in result.outcomes:
            assert o.lower_bound <= o.prediction <= o.upper_bound

    def test_overall_confidence(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"x": 0.5})
        assert isinstance(result.overall_confidence, ProphecyConfidence)

    def test_overall_trajectory(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"x": 0.7})
        assert result.overall_trajectory in ("ascending", "ascending_cautiously", "balanced", "descending_cautiously", "descending")

    def test_pronouncement(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"gdp": 0.5, "inflation": 0.3})
        assert len(result.pronouncement) > 50

    def test_narrative(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"gdp": 0.5})
        assert len(result.narrative) > 20

    def test_recommendations(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"gdp": 0.5})
        assert len(result.recommendations) > 0

    def test_critical_junctures(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"x": 0.5})
        assert isinstance(result.critical_junctures, list)

    def test_divine_intervention_points(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"x": 0.5})
        assert isinstance(result.divine_intervention_points, list)

    def test_horizon_short(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"x": 0.5}, horizon=ProphecyHorizon.SHORT)
        for o in result.outcomes:
            assert o.horizon == ProphecyHorizon.SHORT

    def test_horizon_long(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"x": 0.5}, horizon=ProphecyHorizon.LONG)
        for o in result.outcomes:
            assert o.horizon == ProphecyHorizon.LONG

    def test_empty_predictions(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({})
        assert len(result.outcomes) == 0

    def test_prophecy_confidence_enum(self):
        assert ProphecyConfidence.ORACLE.value == "oracle"
        assert ProphecyConfidence.HIGH.value == "high"
        assert ProphecyConfidence.MODERATE.value == "moderate"
        assert ProphecyConfidence.LOW.value == "low"
        assert ProphecyConfidence.DIVINED.value == "divined"

    def test_prophecy_horizon_enum(self):
        assert ProphecyHorizon.IMMEDIATE.value == "immediate"
        assert ProphecyHorizon.LONG.value == "long_term"

    def test_with_sensitivities(self):
        os = OracleProphecySynthesizer()
        result = os.prophesize({"gdp": 0.5}, sensitivities={"gdp": {"superforecast": 0.2, "black_swan": 0.4}})
        assert isinstance(result, OracleProphecy)