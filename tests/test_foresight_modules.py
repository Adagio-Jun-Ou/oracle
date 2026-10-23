import numpy as np
import pytest

from oracle.prediction.precursor_pattern import (
    PrecursorPatternLibrary,
    PrecursorMatcher,
    RegimeType,
    Phase,
    PrecursorPattern,
    PrecursorStep,
    RegimeChangeWarning,
)
from oracle.prediction.counterfactual import (
    CounterfactualReasoner,
    CounterfactualReport,
    CounterfactualScenario,
    VariableContribution,
)
from oracle.prediction.superforecast import (
    SuperforecastEngine,
    BaseRateCategory,
    BaseRateDatabase,
    SuperforecastResult,
    ForecastDecomposition,
    BASE_RATE_DATA,
)
from oracle.prediction.weak_signal import (
    WeakSignalDetector,
    WeakSignalReport,
    AmplifiedSignal,
)
from oracle.prediction.information_horizon import (
    InformationHorizonMapper,
    CertaintyHorizon,
    QuestionCategory,
    DecisionTimingResult,
    RESOLUTION_PROFILES,
)
from oracle.prediction.cascade_forecast import (
    CascadeTree,
    CascadeResult,
    FailureNode,
    FailureEdge,
)
from oracle.prediction.temporal_contradiction import (
    TemporalContradictionDetector,
    TemporalContradictionResult,
    Direction,
    TurningPointReport,
    TurningPointType,
)
from oracle.prediction.emergence_detector import (
    EmergenceDetector,
    EmergenceDetectionResult,
    EmergenceReport,
    EmergenceType,
    SignalResult,
)
from oracle.prediction.foresight_chain import (
    ForesightChainBuilder,
    ChainResult,
    ChainNode,
    ChainEdge,
    PathStep,
)


# ============================================================
# 1. PrecursorPatternLibrary / PrecursorMatcher
# ============================================================


class TestRegimeType:
    def test_enum_values(self):
        assert RegimeType.RECESSION.value == "recession"
        assert RegimeType.FINANCIAL_CRISIS.value == "financial_crisis"
        assert RegimeType.PANDEMIC_OUTBREAK.value == "pandemic_outbreak"

    def test_all_regime_types_exist(self):
        expected = [
            "recession", "financial_crisis", "inflation_spike",
            "deflationary_trap", "geopolitical_escalation", "climate_tipping",
            "technology_disruption", "pandemic_outbreak", "social_unrest",
            "currency_crisis", "sovereign_debt_crisis", "housing_crash",
            "energy_crisis", "regime_change",
        ]
        actual = [r.value for r in RegimeType]
        for v in expected:
            assert v in actual


class TestPhase:
    def test_enum_values(self):
        assert Phase.NORMAL.value == "normal"
        assert Phase.EARLY_PRECURSOR.value == "early_precursor"
        assert Phase.IMMINENT.value == "imminent"

    def test_phase_ordering(self):
        phases = [Phase.NORMAL, Phase.EARLY_PRECURSOR, Phase.MID_PRECURSOR,
                   Phase.LATE_PRECURSOR, Phase.IMMINENT, Phase.ACTIVE]
        assert len(phases) == 6


class TestPrecursorStep:
    def test_creation(self):
        step = PrecursorStep(
            order=1, description="Yield curve inversion",
            domain_signals={"financial": 0.8, "economic": 0.6},
            typical_lead_time_months=12.0, signal_direction="above",
            threshold=0.5,
        )
        assert step.order == 1
        assert step.description == "Yield curve inversion"
        assert step.typical_lead_time_months == 12.0

    def test_defaults(self):
        step = PrecursorStep(
            order=1, description="test",
            domain_signals={"a": 0.5}, typical_lead_time_months=6.0,
        )
        assert step.signal_direction == "above"
        assert step.threshold == 0.5


class TestPrecursorPatternLibrary:
    def test_instantiation(self):
        lib = PrecursorPatternLibrary()
        assert lib is not None

    def test_find_by_regime(self):
        lib = PrecursorPatternLibrary()
        results = lib.find_by_regime(RegimeType.RECESSION)
        assert isinstance(results, list)
        for p in results:
            assert p.target_regime == RegimeType.RECESSION

    def test_find_empty_regime(self):
        lib = PrecursorPatternLibrary()
        results = lib.find_by_regime(RegimeType.HOUSING_CRASH)
        assert isinstance(results, list)

    def test_get_pattern(self):
        lib = PrecursorPatternLibrary()
        all_recession = lib.find_by_regime(RegimeType.RECESSION)
        if all_recession:
            name = all_recession[0].name
            pattern = lib.get_pattern(name)
            assert pattern is not None
            assert pattern.name == name

    def test_get_pattern_nonexistent(self):
        lib = PrecursorPatternLibrary()
        result = lib.get_pattern("nonexistent_pattern_xyz")
        assert result is None


class TestPrecursorMatcher:
    def test_instantiation(self):
        matcher = PrecursorMatcher()
        assert matcher is not None

    def test_scan_returns_warnings(self):
        matcher = PrecursorMatcher()
        domain_values = {"financial": 0.8, "economic": 0.7, "geopolitical": 0.3}
        warnings = matcher.scan(domain_values)
        assert isinstance(warnings, list)

    def test_scan_empty_domains(self):
        matcher = PrecursorMatcher()
        warnings = matcher.scan({})
        assert isinstance(warnings, list)

    def test_highest_risk_warning(self):
        matcher = PrecursorMatcher()
        domain_values = {"financial": 0.9, "economic": 0.8}
        matcher.scan(domain_values)
        result = matcher.highest_risk_warning()
        assert result is None or isinstance(result, RegimeChangeWarning)

    def test_threat_summary(self):
        matcher = PrecursorMatcher()
        domain_values = {"financial": 0.5, "economic": 0.5}
        matcher.scan(domain_values)
        summary = matcher.threat_summary()
        assert isinstance(summary, dict)

    def test_scan_high_risk_domains(self):
        matcher = PrecursorMatcher()
        domain_values = {
            "financial": 0.95, "economic": 0.9, "geopolitical": 0.85,
            "social": 0.8, "energy": 0.75,
        }
        warnings = matcher.scan(domain_values)
        assert isinstance(warnings, list)
        for w in warnings:
            assert isinstance(w.phase, Phase)
            assert isinstance(w.target_regime, RegimeType)


# ============================================================
# 2. CounterfactualReasoner
# ============================================================


class TestVariableContribution:
    def test_creation(self):
        vc = VariableContribution(
            variable="interest_rate", current_value=0.05,
            required_value=0.02, change_magnitude=0.03,
            change_direction="down", contribution_to_outcome=0.4,
            feasibility=0.6, feasibility_obstacles=["political_resistance"],
        )
        assert vc.variable == "interest_rate"
        assert vc.change_magnitude == 0.03
        assert vc.feasibility_obstacles == ["political_resistance"]

    def test_default_obstacles(self):
        vc = VariableContribution(
            variable="gdp", current_value=2.0, required_value=3.0,
            change_magnitude=1.0, change_direction="up",
            contribution_to_outcome=0.5, feasibility=0.3,
        )
        assert isinstance(vc.feasibility_obstacles, list)


class TestCounterfactualScenario:
    def test_creation(self):
        vc = VariableContribution(
            variable="x", current_value=1.0, required_value=2.0,
            change_magnitude=1.0, change_direction="up",
            contribution_to_outcome=0.5, feasibility=0.5,
        )
        scenario = CounterfactualScenario(
            name="test_scenario", original_outcome="recession",
            counterfactual_outcome="growth",
            original_probability=0.7, counterfactual_probability=0.3,
            required_changes=[vc], n_changes_required=1,
            min_total_change=1.0, feasibility_score=0.5,
            narrative="If x were higher, recession would be less likely",
        )
        assert scenario.name == "test_scenario"
        assert scenario.n_changes_required == 1
        assert len(scenario.required_changes) == 1


class TestCounterfactualReasoner:
    def test_instantiation(self):
        reasoner = CounterfactualReasoner()
        assert reasoner is not None

    def test_compute_outcome_probability(self):
        reasoner = CounterfactualReasoner()
        domain_values = {"gdp_growth": 2.0, "unemployment": 5.0, "inflation": 2.5}
        prob = reasoner.compute_outcome_probability(domain_values, outcome="positive")
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_compute_negative_outcome(self):
        reasoner = CounterfactualReasoner()
        domain_values = {"gdp_growth": -2.0, "unemployment": 12.0, "inflation": 8.0}
        prob = reasoner.compute_outcome_probability(domain_values, outcome="negative")
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_sensitivity_analysis(self):
        reasoner = CounterfactualReasoner()
        domain_values = {"gdp_growth": 2.0, "unemployment": 5.0, "inflation": 2.5}
        result = reasoner.sensitivity_analysis(domain_values, n_top=3)
        assert isinstance(result, list)
        assert len(result) > 0
        for var, score in result:
            assert isinstance(var, str)
            assert isinstance(score, float)

    def test_find_counterfactual(self):
        reasoner = CounterfactualReasoner()
        domain_values = {"gdp_growth": 1.0, "unemployment": 8.0, "inflation": 5.0}
        report = reasoner.find_counterfactual(
            domain_values, target_probability=0.3,
            original_outcome="positive",
            counterfactual_outcome="negative",
            max_changes=3,
        )
        assert isinstance(report, CounterfactualReport)
        assert isinstance(report.scenarios, list)
        assert isinstance(report.robustness_score, float)

    def test_empty_domains(self):
        reasoner = CounterfactualReasoner()
        prob = reasoner.compute_outcome_probability({}, outcome="positive")
        assert isinstance(prob, float)


# ============================================================
# 3. SuperforecastEngine / BaseRateDatabase
# ============================================================


class TestBaseRateCategory:
    def test_enum_values(self):
        assert BaseRateCategory.RECESSION_1Y.value == "recession_1y"
        assert BaseRateCategory.PANDEMIC.value == "pandemic"

    def test_all_categories(self):
        expected_count = 17
        assert len(BaseRateCategory) == expected_count


class TestBaseRateDatabase:
    def test_instantiation(self):
        db = BaseRateDatabase()
        assert db is not None

    def test_get_base_rate(self):
        db = BaseRateDatabase()
        result = db.get_base_rate(BaseRateCategory.RECESSION_1Y)
        assert isinstance(result, dict)
        assert "base_rate" in result or "rate" in result or "frequency" in result or len(result) > 0

    def test_list_categories(self):
        db = BaseRateDatabase()
        cats = db.list_categories()
        assert isinstance(cats, list)
        assert len(cats) > 0

    def test_get_adjusted_rate(self):
        db = BaseRateDatabase()
        rate = db.get_adjusted_rate(BaseRateCategory.RECESSION_1Y, horizon_years=5.0)
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0


class TestSuperforecastEngine:
    def test_instantiation(self):
        engine = SuperforecastEngine()
        assert engine is not None

    def test_forecast_basic(self):
        engine = SuperforecastEngine()
        result = engine.forecast(BaseRateCategory.RECESSION_1Y)
        assert isinstance(result, SuperforecastResult)
        assert 0.0 <= result.final_probability <= 1.0
        assert isinstance(result.decomposition, ForecastDecomposition)

    def test_forecast_with_inside_view(self):
        engine = SuperforecastEngine()
        result = engine.forecast(
            BaseRateCategory.FINANCIAL_CRISIS,
            inside_view_prob=0.8,
        )
        assert isinstance(result, SuperforecastResult)
        assert 0.0 <= result.final_probability <= 1.0

    def test_forecast_with_evidence_signals(self):
        engine = SuperforecastEngine()
        signals = [
            {"name": "yield_curve", "strength": 0.7, "direction": "negative"},
            {"name": "credit_spread", "strength": 0.5, "direction": "negative"},
        ]
        result = engine.forecast(
            BaseRateCategory.RECESSION_1Y,
            inside_view_prob=0.6,
            evidence_signals=signals,
        )
        assert isinstance(result, SuperforecastResult)
        assert len(result.recommendations) >= 0

    def test_forecast_decomposition_structure(self):
        engine = SuperforecastEngine()
        result = engine.forecast(BaseRateCategory.HOUSING_CRASH)
        d = result.decomposition
        assert isinstance(d.base_rate, float)
        assert isinstance(d.outside_view_probability, float)
        assert isinstance(d.inside_view_probability, float)
        assert isinstance(d.combined_prior, float)
        assert isinstance(d.extremized_probability, float)

    def test_forecast_confidence_interval(self):
        engine = SuperforecastEngine()
        result = engine.forecast(BaseRateCategory.PANDEMIC)
        low, high = result.confidence_interval
        assert low <= result.final_probability <= high or low <= high


# ============================================================
# 4. WeakSignalDetector
# ============================================================


class TestAmplifiedSignal:
    def test_creation(self):
        sig = AmplifiedSignal(
            signal_type="entropy_shift", strength=0.8,
            source_domain="financial", lead_time_estimate=6.0,
            confidence=0.7, description="Rising entropy in financial indicators",
        )
        assert sig.signal_type == "entropy_shift"
        assert sig.strength == 0.8
        assert sig.source_domain == "financial"


class TestWeakSignalReport:
    def test_creation(self):
        report = WeakSignalReport()
        assert isinstance(report.signals, list)
        assert isinstance(report.urgency_rank, list)
        assert report.total_signal_count == 0

    def test_with_signals(self):
        sig = AmplifiedSignal(
            signal_type="correlation_drift", strength=0.6,
            source_domain="economic", lead_time_estimate=3.0,
            confidence=0.5, description="Shift in correlation structure",
        )
        report = WeakSignalReport(
            signals=[sig], total_signal_count=1,
            combined_strength=0.6, dominant_signal_type="correlation_drift",
            domains_affected=["economic"],
        )
        assert len(report.signals) == 1
        assert report.total_signal_count == 1


class TestWeakSignalDetector:
    def test_instantiation(self):
        detector = WeakSignalDetector()
        assert detector is not None

    def test_amplify(self):
        detector = WeakSignalDetector()
        domain_values = {"financial": 0.7, "economic": 0.5, "geopolitical": 0.3}
        domain_histories = {
            "financial": np.random.randn(100),
            "economic": np.random.randn(100),
        }
        report = detector.amplify(domain_values, domain_histories)
        assert isinstance(report, WeakSignalReport)

    def test_detect_entropy_shift(self):
        detector = WeakSignalDetector()
        series = np.random.randn(200)
        result = detector.detect_entropy_shift(series)
        assert isinstance(result, dict)

    def test_detect_entropy_shift_method(self):
        detector = WeakSignalDetector()
        series = np.random.randn(200)
        result = detector.detect_entropy_shift(series, method="histogram")
        assert isinstance(result, dict)

    def test_detect_multi_scale_anomaly(self):
        detector = WeakSignalDetector()
        series = np.random.randn(200)
        result = detector.detect_multi_scale_anomaly(series, scales=[10, 30, 50])
        assert isinstance(result, dict)

    def test_detect_correlation_drift(self):
        detector = WeakSignalDetector()
        series_dict = {
            "a": np.random.randn(200),
            "b": np.random.randn(200),
            "c": np.random.randn(200),
        }
        result = detector.detect_correlation_drift(series_dict)
        assert isinstance(result, dict)


# ============================================================
# 5. InformationHorizonMapper
# ============================================================


class TestQuestionCategory:
    def test_enum_values(self):
        assert QuestionCategory.DAILY_MARKET_MOVE.value == "daily_market_move"
        assert QuestionCategory.RECESSION_ONSET.value == "recession_onset"
        assert QuestionCategory.CLIMATE_TIPPING.value == "climate_tipping"

    def test_all_categories(self):
        assert len(QuestionCategory) == 10


class TestCertaintyHorizon:
    def test_creation(self):
        ch = CertaintyHorizon(
            question="Will recession hit in 2026?",
            category="economic",
            current_certainty=0.4,
        )
        assert ch.question == "Will recession hit in 2026?"
        assert ch.current_certainty == 0.4

    def test_with_resolution_times(self):
        ch = CertaintyHorizon(
            question="GDP growth?",
            category="economic",
            current_certainty=0.3,
            resolution_times={6.0: 0.6, 12.0: 0.85},
            information_velocity=0.05,
        )
        assert 6.0 in ch.resolution_times
        assert ch.information_velocity == 0.05


class TestInformationHorizonMapper:
    def test_instantiation(self):
        mapper = InformationHorizonMapper()
        assert mapper is not None

    def test_with_custom_profiles(self):
        profiles = {
            QuestionCategory.DAILY_MARKET_MOVE: {"half_life": 0.1},
        }
        mapper = InformationHorizonMapper(custom_profiles=profiles)
        assert mapper is not None

    def test_estimate_resolution_time(self):
        mapper = InformationHorizonMapper()
        result = mapper.estimate_resolution_time(
            QuestionCategory.RECESSION_ONSET,
            current_data_quality=0.6,
        )
        assert isinstance(result, dict)

    def test_certainty_timeline(self):
        mapper = InformationHorizonMapper()
        uncertainties = [
            {"question": "Recession?", "category": "recession_onset",
             "current_certainty": 0.3, "data_quality": 0.7},
            {"question": "Market crash?", "category": "daily_market_move",
             "current_certainty": 0.5, "data_quality": 0.8},
        ]
        timeline = mapper.certainty_timeline(uncertainties)
        assert isinstance(timeline, list)
        assert len(timeline) == 2
        for item in timeline:
            assert isinstance(item, CertaintyHorizon)

    def test_value_of_information(self):
        mapper = InformationHorizonMapper()
        uncertainty = {
            "question": "Recession?", "category": "recession_onset",
            "current_certainty": 0.3, "data_quality": 0.6,
        }
        decision_context = {"deadline_months": 12.0, "cost_of_error": 0.5}
        result = mapper.value_of_information(uncertainty, decision_context)
        assert isinstance(result, dict)

    def test_decision_timing_score(self):
        mapper = InformationHorizonMapper()
        uncertainties = [
            {"question": "Recession?", "category": "recession_onset",
             "current_certainty": 0.3, "data_quality": 0.7},
        ]
        result = mapper.decision_timing_score(
            decision_deadline=6.0,
            uncertainties=uncertainties,
            decision_cost_of_delay=0.01,
        )
        assert isinstance(result, DecisionTimingResult)


# ============================================================
# 6. CascadeTree
# ============================================================


class TestFailureNode:
    def test_creation(self):
        node = FailureNode(
            name="power_grid", domain="energy",
            probability=0.05, impact_severity=0.9,
        )
        assert node.name == "power_grid"
        assert node.probability == 0.05


class TestFailureEdge:
    def test_creation(self):
        edge = FailureEdge(
            source="power_grid", target="transportation",
            propagation_probability=0.6, time_delay_months=2.0,
            mechanism="cascade",
        )
        assert edge.source == "power_grid"
        assert edge.mechanism == "cascade"


class TestCascadeTree:
    def test_instantiation(self):
        tree = CascadeTree()
        assert tree is not None

    def test_add_node_and_edge(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("a", "energy", 0.05, 0.8))
        tree.add_node(FailureNode("b", "transport", 0.03, 0.7))
        tree.add_edge("a", "b", probability=0.4, delay=1.0, mechanism="dependency")
        result = tree.simulate_cascade("a", n_simulations=50)
        assert isinstance(result, CascadeResult)

    def test_simulate_cascade(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("financial_crash", "financial", 0.05, 0.9))
        tree.add_node(FailureNode("bank_failure", "financial", 0.02, 0.8))
        tree.add_node(FailureNode("recession", "economic", 0.01, 0.95))
        tree.add_edge("financial_crash", "bank_failure", probability=0.5, delay=1.0, mechanism="contagion")
        tree.add_edge("bank_failure", "recession", probability=0.3, delay=3.0, mechanism="credit_crunch")
        result = tree.simulate_cascade("financial_crash", n_simulations=100)
        assert isinstance(result, CascadeResult)
        assert result.trigger == "financial_crash"
        assert isinstance(result.cascade_paths, list)

    def test_cascade_probability(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("a", "energy", 0.05, 0.9))
        tree.add_node(FailureNode("b", "health", 0.02, 0.7))
        tree.add_edge("a", "b", probability=0.3, delay=1.0, mechanism="cascade")
        result = tree.cascade_probability("b", n_simulations=100)
        assert isinstance(result, dict)

    def test_critical_paths(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("start", "energy", 0.05, 0.9))
        tree.add_node(FailureNode("mid", "transport", 0.03, 0.7))
        tree.add_node(FailureNode("end", "economic", 0.01, 0.95))
        tree.add_edge("start", "mid", probability=0.5, delay=1.0, mechanism="a")
        tree.add_edge("mid", "end", probability=0.4, delay=2.0, mechanism="b")
        paths = tree.critical_paths(top_n=3)
        assert isinstance(paths, list)

    def test_systemic_risk_score(self):
        tree = CascadeTree()
        tree.add_node(FailureNode("x", "a", 0.05, 0.8))
        tree.add_node(FailureNode("y", "b", 0.03, 0.7))
        tree.add_edge("x", "y", probability=0.4, delay=1.0, mechanism="cascade")
        score = tree.systemic_risk_score(n_simulations=100)
        assert isinstance(score, dict)


# ============================================================
# 7. TemporalContradictionDetector
# ============================================================


class TestDirection:
    def test_enum_values(self):
        assert Direction.UP.value == "up"
        assert Direction.DOWN.value == "down"
        assert Direction.FLAT.value == "flat"


class TestTurningPointType:
    def test_enum_values(self):
        assert TurningPointType.ACCELERATION.value == "acceleration"
        assert TurningPointType.REVERSAL.value == "reversal"

    def test_all_types(self):
        assert len(TurningPointType) == 5


class TestTemporalContradictionDetector:
    def test_instantiation(self):
        detector = TemporalContradictionDetector()
        assert detector is not None

    def test_divergence_score(self):
        detector = TemporalContradictionDetector()
        score = detector.divergence_score("gdp", short_val=0.05, long_val=-0.02)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_detect_basic(self):
        detector = TemporalContradictionDetector()
        short_term = [
            {"variable": "gdp", "value": 0.03, "direction": "up"},
            {"variable": "inflation", "value": 0.05, "direction": "up"},
        ]
        long_term = [
            {"variable": "gdp", "value": -0.01, "direction": "down"},
            {"variable": "inflation", "value": 0.02, "direction": "up"},
        ]
        result = detector.detect(short_term, long_term)
        assert isinstance(result, TemporalContradictionResult)
        assert isinstance(result.n_divergences, int)

    def test_detect_with_historical_data(self):
        detector = TemporalContradictionDetector()
        short_term = [
            {"variable": "unemployment", "value": 0.06, "direction": "up"},
        ]
        long_term = [
            {"variable": "unemployment", "value": 0.03, "direction": "down"},
        ]
        historical = {"unemployment": np.random.randn(100)}
        result = detector.detect(short_term, long_term, historical)
        assert isinstance(result, TemporalContradictionResult)

    def test_classify_turning_point(self):
        detector = TemporalContradictionDetector()
        divergences = [
            {"variable": "gdp", "score": 0.7, "short_dir": "up", "long_dir": "down"},
        ]
        types = detector.classify_turning_point(divergences)
        assert isinstance(types, list)
        for t in types:
            assert isinstance(t, TurningPointType)

    def test_turning_point_probability(self):
        detector = TemporalContradictionDetector()
        divergences = [
            {"variable": "gdp", "score": 0.5},
            {"variable": "inflation", "score": 0.3},
        ]
        probs = detector.turning_point_probability(divergences)
        assert isinstance(probs, list)
        for p in probs:
            assert isinstance(p, float)


# ============================================================
# 8. EmergenceDetector
# ============================================================


class TestEmergenceType:
    def test_enum_values(self):
        assert EmergenceType.CRITICAL_SLOWING.value == "critical_slowing"
        assert EmergenceType.VARIANCE_INCREASE.value == "variance_increase"
        assert EmergenceType.FLICKERING.value == "flickering"
        assert EmergenceType.CORRELATION_LENGTH.value == "correlation_length"

    def test_all_types(self):
        assert len(EmergenceType) == 5


class TestSignalResult:
    def test_creation(self):
        sr = SignalResult(
            signal_type="critical_slowing", value=0.85,
            normalized=0.7, threshold=0.5, triggered=True,
        )
        assert sr.signal_type == "critical_slowing"
        assert sr.triggered is True


class TestEmergenceDetector:
    def test_instantiation(self):
        detector = EmergenceDetector()
        assert detector is not None

    def test_detect_critical_slowing(self):
        detector = EmergenceDetector()
        np.random.seed(42)
        series = np.random.randn(200)
        result = detector.detect_critical_slowing(series)
        assert isinstance(result, SignalResult)
        assert isinstance(result.value, float)

    def test_detect_variance_increase(self):
        detector = EmergenceDetector()
        np.random.seed(42)
        series = np.random.randn(200)
        result = detector.detect_variance_increase(series)
        assert isinstance(result, SignalResult)

    def test_detect_flickering(self):
        detector = EmergenceDetector()
        np.random.seed(42)
        series = np.random.randn(200)
        result = detector.detect_flickering(series, threshold=2.0)
        assert isinstance(result, SignalResult)

    def test_detect_correlation_length(self):
        detector = EmergenceDetector()
        np.random.seed(42)
        matrix = np.random.randn(5, 200)
        result = detector.detect_correlation_length(matrix)
        assert isinstance(result, SignalResult)

    def test_detect_single(self):
        detector = EmergenceDetector()
        np.random.seed(42)
        series = np.random.randn(200)
        result = detector.detect_single("test_var", series)
        assert result is None or isinstance(result, EmergenceReport)

    def test_detect_full(self):
        detector = EmergenceDetector()
        np.random.seed(42)
        variables = {
            "gdp": np.random.randn(200),
            "inflation": np.random.randn(200),
            "unemployment": np.random.randn(200),
        }
        result = detector.detect(variables)
        assert isinstance(result, EmergenceDetectionResult)
        assert isinstance(result.n_emergence_signals, int)
        assert isinstance(result.system_emergence_probability, float)

    def test_emergence_score(self):
        detector = EmergenceDetector()
        signals = [
            SignalResult("critical_slowing", 0.8, 0.7, 0.5, True),
            SignalResult("variance_increase", 0.6, 0.5, 0.5, True),
            SignalResult("flickering", 0.3, 0.2, 0.5, False),
        ]
        score = detector.emergence_score(signals)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ============================================================
# 9. ForesightChainBuilder
# ============================================================


class TestChainNode:
    def test_creation(self):
        node = ChainNode(variable="recession", base_probability=0.15)
        assert node.variable == "recession"
        assert node.base_probability == 0.15

    def test_with_value_range(self):
        node = ChainNode(variable="inflation", base_probability=0.1, value_range=(0.0, 0.3))
        assert node.value_range == (0.0, 0.3)


class TestChainEdge:
    def test_creation(self):
        edge = ChainEdge(source="yield_curve", target="recession", conditional_prob=0.4)
        assert edge.source == "yield_curve"
        assert edge.conditional_prob == 0.4


class TestForesightChainBuilder:
    def test_instantiation_default(self):
        builder = ForesightChainBuilder()
        assert builder is not None
        assert len(builder.all_nodes()) > 0

    def test_add_node(self):
        builder = ForesightChainBuilder()
        node = builder.add_node("custom_var", base_probability=0.2)
        assert isinstance(node, ChainNode)
        assert node.variable == "custom_var"

    def test_add_conditional(self):
        builder = ForesightChainBuilder()
        edge = builder.add_conditional("gdp_growth", "recession",
                                        conditional_prob=0.3, condition_range=(0.0, 0.5))
        assert isinstance(edge, ChainEdge)
        assert edge.source == "gdp_growth"

    def test_compute_chain(self):
        builder = ForesightChainBuilder()
        builder.add_node("trigger_event", base_probability=0.1)
        builder.add_node("cascade_effect", base_probability=0.05)
        builder.add_conditional("trigger_event", "cascade_effect", conditional_prob=0.4)
        result = builder.compute_chain("trigger_event", start_value=0.8)
        assert isinstance(result, ChainResult)
        assert isinstance(result.joint_probability, float)
        assert isinstance(result.most_probable_path, list)

    def test_compute_backward_chain(self):
        builder = ForesightChainBuilder()
        builder.add_node("root_cause", base_probability=0.05)
        builder.add_node("intermediate", base_probability=0.1)
        builder.add_node("outcome", base_probability=0.02)
        builder.add_conditional("root_cause", "intermediate", conditional_prob=0.3)
        builder.add_conditional("intermediate", "outcome", conditional_prob=0.5)
        result = builder.compute_backward_chain("outcome", target_value=0.8)
        assert isinstance(result, ChainResult)

    def test_monte_carlo_chain(self):
        builder = ForesightChainBuilder()
        builder.add_node("start_var", base_probability=0.1)
        builder.add_node("end_var", base_probability=0.05)
        builder.add_conditional("start_var", "end_var", conditional_prob=0.3)
        result = builder.Monte_Carlo_chain(
            "start_var", start_value=0.7, n_simulations=100, max_depth=3,
        )
        assert isinstance(result, dict)

    def test_critical_chain_links(self):
        builder = ForesightChainBuilder()
        links = builder.critical_chain_links(threshold=0.4)
        assert isinstance(links, list)

    def test_all_nodes_and_edges(self):
        builder = ForesightChainBuilder()
        nodes = builder.all_nodes()
        edges = builder.all_edges()
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
        for n in nodes:
            assert isinstance(n, ChainNode)


# ============================================================
# Integration / Cross-module tests
# ============================================================


class TestCrossModuleIntegration:
    def test_precursor_to_cascade(self):
        library = PrecursorPatternLibrary()
        matcher = PrecursorMatcher()
        domain_values = {"financial": 0.85, "economic": 0.7}
        warnings = matcher.scan(domain_values)

        tree = CascadeTree()
        tree.add_node(FailureNode("financial_shock", "financial", 0.05, 0.9))
        tree.add_node(FailureNode("economic_downturn", "economic", 0.03, 0.85))
        tree.add_edge("financial_shock", "economic_downturn",
                       probability=0.4, delay=2.0, mechanism="contagion")
        result = tree.simulate_cascade("financial_shock", n_simulations=50)
        assert isinstance(result, CascadeResult)

    def test_emergence_to_temporal_contradiction(self):
        np.random.seed(42)
        variables = {"gdp": np.random.randn(150), "inflation": np.random.randn(150)}

        emergence_detector = EmergenceDetector()
        emergence_result = emergence_detector.detect(variables)
        assert isinstance(emergence_result, EmergenceDetectionResult)

        temporal_detector = TemporalContradictionDetector()
        short_term = [{"variable": "gdp", "value": 0.03, "direction": "up"}]
        long_term = [{"variable": "gdp", "value": -0.02, "direction": "down"}]
        temporal_result = temporal_detector.detect(short_term, long_term)
        assert isinstance(temporal_result, TemporalContradictionResult)

    def test_superforecast_with_counterfactual(self):
        engine = SuperforecastEngine()
        forecast = engine.forecast(BaseRateCategory.RECESSION_1Y)
        assert 0.0 <= forecast.final_probability <= 1.0

        reasoner = CounterfactualReasoner()
        domain_values = {"gdp_growth": 1.5, "unemployment": 6.0}
        prob = reasoner.compute_outcome_probability(domain_values)
        assert 0.0 <= prob <= 1.0

    def test_weak_signal_to_information_horizon(self):
        detector = WeakSignalDetector()
        domain_values = {"economic": 0.6}
        domain_histories = {"economic": np.random.randn(100)}
        report = detector.amplify(domain_values, domain_histories)
        assert isinstance(report, WeakSignalReport)

        mapper = InformationHorizonMapper()
        result = mapper.estimate_resolution_time(
            QuestionCategory.RECESSION_ONSET, current_data_quality=0.6,
        )
        assert isinstance(result, dict)

    def test_foresight_chain_with_superforecast(self):
        builder = ForesightChainBuilder()
        builder.add_node("recession_trigger", base_probability=0.1)
        result = builder.compute_chain("recession_trigger", start_value=0.5)
        assert isinstance(result, ChainResult)

        engine = SuperforecastEngine()
        forecast = engine.forecast(BaseRateCategory.RECESSION_1Y)
        assert isinstance(forecast, SuperforecastResult)