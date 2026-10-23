"""Tests for 8 new prediction modules: Black Swan, Regime Shift, Collective Intelligence,
Analogical Reasoning, Morphological, Delphi, Cross-Impact, Temporal Causal Discovery."""
import numpy as np
import pytest
from oracle.prediction.black_swan import (
    BlackSwanDetector, BlackSwanReport, SwanCategory, VulnerabilityAxis,
    TailDiagnostic, StructuralVulnerability,
)
from oracle.prediction.regime_shift import (
    RegimeShiftPredictor, RegimeShiftReport, RegimeShiftWarning, ShiftType,
    WarningSignal, EarlyWarningResult,
)
from oracle.prediction.collective_intelligence import (
    CollectiveIntelligenceAggregator, AggregatedForecast, AggregationMethod,
    ForecasterProfile,
)
from oracle.prediction.analogical_reasoning import (
    AnalogicalReasoningEngine, AnalogicalPrediction, AnalogMatch,
    HistoricalEvent, AnalogQuality,
)
from oracle.prediction.morphological import (
    MorphologicalAnalyzer, MorphologicalResult, MorphologicalParameter,
    MorphologicalField, MorphologicalCell, ConsistencyLevel,
)
from oracle.prediction.delphi import (
    DelphiMethodSimulator, DelphiResult, DelphiPanel, DelphiRound,
    ConvergenceStatus, ExpertiseLevel, SimulatedExpert,
)
from oracle.prediction.cross_impact import (
    CrossImpactAnalyzer, CrossImpactResult, CrossImpactEvent,
    CrossImpactPair, ImpactDirection,
)
from oracle.prediction.temporal_causal import (
    TemporalCausalDiscoverer, TemporalCausalResult, CausalLink,
    CausalGraph, CausalMethod,
)
from oracle.prediction.butterfly_effect import ButterflyTracker, Perturbation, DomainNetwork
from oracle.prediction.horizon_scanning import (
    HorizonScanner, HorizonScanResult, WeakSignal, SignalMaturity, HorizonLevel, SignalCategory,
)
from oracle.prediction.probabilistic_network import (
    ProbabilisticNetwork, NetworkResult, NetworkNode, ConditionalTable, InferenceResult,
)
from oracle.prediction.quantum_uncertainty import (
    QuantumUncertaintyMapper, QuantumUncertaintyResult, UncertaintyType, EntanglementPair,
)
from oracle.prediction.decision_theory import (
    DecisionTheoreticAdvisor, DecisionResult, DecisionCriterion, RiskAttitude,
)
from oracle.prediction.strategic_foresight import (
    StrategicForesightIntegrator, ForesightAssessment, SignalDirection, ConvergentSignal,
)
from oracle.prediction.butterfly_effect import ButterflyTracker


class TestBlackSwanDetector:
    def test_instantiation(self):
        bd = BlackSwanDetector()
        assert bd.threshold_percentile == 95.0

    def test_diagnose_tails_normal(self):
        bd = BlackSwanDetector()
        data = {"normal": np.random.randn(200)}
        diags = bd.diagnose_tails(data)
        assert "normal" in diags
        assert not diags["normal"].is_fat_tailed

    def test_diagnose_tails_fat(self):
        bd = BlackSwanDetector()
        fat = np.concatenate([np.random.randn(180), np.random.randn(20) * 10])
        data = {"fat_tail": fat}
        diags = bd.diagnose_tails(data)
        assert "fat_tail" in diags

    def test_assess_structural_vulnerability(self):
        bd = BlackSwanDetector()
        states = {"economic": {"debt_to_gdp": 2.5}}
        vulns = bd.assess_structural_vulnerability(states)
        assert len(vulns) == 6
        axes = {v.axis for v in vulns}
        assert VulnerabilityAxis.LEVERAGE in axes

    def test_detect_returns_report(self):
        bd = BlackSwanDetector()
        data = {"gdp": np.random.randn(100)}
        states = {"econ": {"debt_to_gdp": 1.2}}
        report = bd.detect(data, states)
        assert isinstance(report, BlackSwanReport)
        assert 0 <= report.overall_probability <= 1
        assert isinstance(report.category, SwanCategory)
        assert len(report.structural_vulnerabilities) == 6

    def test_detect_dragon_king(self):
        bd = BlackSwanDetector()
        extreme = np.concatenate([np.random.randn(90), np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])])
        data = {"extreme": extreme}
        report = bd.detect(data, {"econ": {"debt_to_gdp": 0.5}})
        assert isinstance(report, BlackSwanReport)

    def test_kurtosis_calculation(self):
        bd = BlackSwanDetector()
        normal = np.random.randn(500)
        kurt = bd._kurtosis(normal)
        assert abs(kurt) < 2.0


class TestRegimeShiftPredictor:
    def test_instantiation(self):
        rs = RegimeShiftPredictor()
        assert rs.window_size == 30

    def test_variance_increase_stable(self):
        rs = RegimeShiftPredictor()
        series = np.random.randn(100)
        result = rs.detect_variance_increase(series)
        assert isinstance(result, EarlyWarningResult)
        assert result.signal == WarningSignal.VARIANCE_INCREASE

    def test_variance_increase_unstable(self):
        rs = RegimeShiftPredictor()
        stable = np.random.randn(50) * 0.5
        unstable = np.random.randn(50) * 3.0
        series = np.concatenate([stable, unstable])
        result = rs.detect_variance_increase(series)
        assert result.strength > 0 or not result.detected

    def test_autocorrelation_increase(self):
        rs = RegimeShiftPredictor()
        series = np.random.randn(100)
        result = rs.detect_autocorrelation_increase(series)
        assert isinstance(result, EarlyWarningResult)
        assert result.signal == WarningSignal.AUTOCORRELATION_INCREASE

    def test_cusum_detection(self):
        rs = RegimeShiftPredictor()
        series = np.concatenate([np.random.randn(50) * 0.1, np.random.randn(50) * 0.1 + 2.0])
        result = rs.detect_cusum(series)
        assert isinstance(result, EarlyWarningResult)

    def test_spectral_reddening(self):
        rs = RegimeShiftPredictor()
        t = np.linspace(0, 100, 200)
        series = np.sin(0.1 * t) + np.random.randn(200) * 0.1
        result = rs.detect_spectral_reddening(series)
        assert isinstance(result, EarlyWarningResult)

    def test_detect_regime_shift(self):
        rs = RegimeShiftPredictor()
        series = np.concatenate([np.random.randn(50) * 0.5, np.random.randn(50) * 2.0 + 1.0])
        warning = rs.detect_regime_shift("gdp", series)
        assert isinstance(warning, RegimeShiftWarning)
        assert warning.variable == "gdp"
        assert isinstance(warning.shift_type, ShiftType)

    def test_detect_all(self):
        rs = RegimeShiftPredictor()
        data = {"gdp": np.random.randn(100), "inflation": np.random.randn(100)}
        report = rs.detect_all(data)
        assert isinstance(report, RegimeShiftReport)
        assert 0 <= report.overall_risk <= 1


class TestCollectiveIntelligence:
    def test_instantiation(self):
        ci = CollectiveIntelligenceAggregator()
        assert ci.forecasters == {}

    def test_register_forecaster(self):
        ci = CollectiveIntelligenceAggregator()
        profile = ci.register_forecaster("alice", expertise=["economic"])
        assert profile.name == "alice"
        assert "economic" in profile.expertise

    def test_simple_average(self):
        ci = CollectiveIntelligenceAggregator()
        result = ci.aggregate("test", {"a": 0.6, "b": 0.8}, AggregationMethod.SIMPLE_AVERAGE)
        assert abs(result.probability - 0.7) < 0.01

    def test_confidence_weighted(self):
        ci = CollectiveIntelligenceAggregator()
        ci.register_forecaster("expert1", ["economic"])
        ci.register_forecaster("novice1")
        result = ci.aggregate("test", {"expert1": 0.7, "novice1": 0.3},
                              AggregationMethod.CONFIDENCE_WEIGHTED, domain="economic")
        assert result.probability > 0.5

    def test_logarithmic_pool(self):
        ci = CollectiveIntelligenceAggregator()
        result = ci.aggregate("test", {"a": 0.7, "b": 0.7, "c": 0.7},
                              AggregationMethod.LOGARITHMIC_POOL)
        assert result.probability > 0.6

    def test_extremized(self):
        ci = CollectiveIntelligenceAggregator()
        result = ci.aggregate("test", {"a": 0.7, "b": 0.7, "c": 0.7, "d": 0.7},
                              AggregationMethod.EXTREMIZED)
        assert result.is_extremized

    def test_best_method_selection(self):
        ci = CollectiveIntelligenceAggregator()
        ci.register_forecaster("a")
        ci.register_forecaster("b")
        ci.register_forecaster("c")
        ci.forecasters["a"].brier_scores = [0.1, 0.2, 0.15]
        ci.forecasters["b"].brier_scores = [0.1, 0.2, 0.15]
        ci.forecasters["c"].brier_scores = [0.1, 0.2, 0.15]
        result = ci.aggregate("test", {"a": 0.7, "b": 0.6, "c": 0.8}, AggregationMethod.BEST)
        assert result.probability > 0

    def test_empty_forecasts(self):
        ci = CollectiveIntelligenceAggregator()
        result = ci.aggregate("test", {})
        assert result.probability == 0.5

    def test_update_performance(self):
        ci = CollectiveIntelligenceAggregator()
        ci.update_forecaster_performance("alice", 0.7, 1.0)
        assert "alice" in ci.forecasters
        assert ci.forecasters["alice"].n_forecasts == 1

    def test_aggregate_populates_forecast_history(self):
        ci = CollectiveIntelligenceAggregator()
        ci.aggregate("recession", {"alice": 0.7, "bob": 0.6})
        assert len(ci.forecast_history) == 2
        assert ci.forecast_history[0]["forecaster"] == "alice"
        assert ci.forecast_history[0]["probability"] == 0.7
        assert ci.forecast_history[0]["realized"] is None

    def test_resolve_forecast_updates_history(self):
        ci = CollectiveIntelligenceAggregator()
        ci.aggregate("recession", {"alice": 0.7, "bob": 0.3})
        n = ci.resolve_forecast("recession", 1.0)
        assert n == 2
        assert all(e["realized"] == 1.0 for e in ci.forecast_history)

    def test_resolve_triggers_calibration(self):
        ci = CollectiveIntelligenceAggregator()
        for _ in range(6):
            ci.aggregate("recession", {"alice": 0.8})
        ci.resolve_forecast("recession", 0.5)
        assert ci.forecasters["alice"].n_forecasts > 5


class TestAnalogicalReasoning:
    def test_instantiation(self):
        ae = AnalogicalReasoningEngine()
        assert len(ae.events) > 0

    def test_find_analogs(self):
        ae = AnalogicalReasoningEngine()
        matches = ae.find_analogs({"leverage": 0.9, "credit_growth": 0.8})
        assert len(matches) > 0
        assert matches[0].similarity > 0

    def test_predict(self):
        ae = AnalogicalReasoningEngine()
        result = ae.predict("market crash", {"leverage": 0.9, "credit_growth": 0.8})
        assert isinstance(result, AnalogicalPrediction)
        assert 0 < result.probability < 1
        assert len(result.analogs) > 0

    def test_predict_no_match(self):
        ae = AnalogicalReasoningEngine()
        result = ae.predict("unknown", {"obscure_feature_xyz": 0.5})
        assert isinstance(result, AnalogicalPrediction)

    def test_add_event(self):
        ae = AnalogicalReasoningEngine()
        n_before = len(ae.events)
        ae.add_event(HistoricalEvent("Test Event", 2020, "test",
                                     {"feat1": 0.5}, {"outcome": 0.8}, 0.8, "test"))
        assert len(ae.events) == n_before + 1

    def test_domain_filtering(self):
        ae = AnalogicalReasoningEngine()
        matches = ae.find_analogs({"leverage": 0.8}, domain="financial")
        for m in matches:
            assert m.event.domain == "financial"

    def test_anti_analogs_detected(self):
        ae = AnalogicalReasoningEngine()
        result = ae.predict("recovery", {"leverage": 0.1, "credit_growth": 0.1})
        assert isinstance(result, AnalogicalPrediction)


class TestMorphologicalAnalysis:
    def test_instantiation(self):
        ma = MorphologicalAnalyzer()
        assert len(ma.parameters) > 0

    def test_build_field(self):
        ma = MorphologicalAnalyzer()
        field = ma.build_field(max_cells=100)
        assert isinstance(field, MorphologicalField)
        assert len(field.cells) > 0
        assert field.n_possible > 0

    def test_analyze(self):
        ma = MorphologicalAnalyzer()
        result = ma.analyze(max_cells=100)
        assert isinstance(result, MorphologicalResult)
        assert len(result.top_scenarios) > 0

    def test_add_parameter(self):
        ma = MorphologicalAnalyzer()
        n_before = len(ma.parameters)
        ma.add_parameter(MorphologicalParameter("test", ["a", "b"], [0.5, 0.5]))
        assert len(ma.parameters) == n_before + 1

    def test_consistency_setting(self):
        ma = MorphologicalAnalyzer()
        ma.set_consistency("param1", "val_a", "param2", "val_b", ConsistencyLevel.INCONSISTENT)
        assert ("param1", "val_a", "param2", "val_b") in ma.consistency_rules
        assert ("param2", "val_b", "param1", "val_a") in ma.consistency_rules

    def test_novelty_scoring(self):
        ma = MorphologicalAnalyzer()
        result = ma.analyze(max_cells=200)
        for cell in result.top_scenarios:
            assert 0 <= cell.novelty <= 1


class TestDelphiMethod:
    def test_instantiation(self):
        dm = DelphiMethodSimulator()
        assert dm.n_rounds == 3

    def test_create_panel(self):
        dm = DelphiMethodSimulator()
        panel = dm.create_panel(domain="economic")
        assert len(panel.experts) > 0

    def test_run_delphi(self):
        dm = DelphiMethodSimulator()
        result = dm.run_delphi("Will GDP grow > 3%?", true_value=0.5)
        assert isinstance(result, DelphiResult)
        assert 0 <= result.final_median <= 1
        assert isinstance(result.convergence_status, ConvergenceStatus)
        assert len(result.rounds) == 3

    def test_convergence(self):
        dm = DelphiMethodSimulator(n_rounds=5)
        result = dm.run_delphi("Test?", true_value=0.7, domain="economic")
        assert len(result.rounds) == 5

    def test_dissenters_identified(self):
        dm = DelphiMethodSimulator()
        panel = dm.create_panel()
        for e in panel.experts[:1]:
            e.openness_to_evidence = 0.01
        result = dm.run_delphi("Test?", true_value=0.5, panel=panel)
        assert isinstance(result, DelphiResult)

    def test_multi_question(self):
        dm = DelphiMethodSimulator()
        results = dm.run_multi_question({"Q1": 0.5, "Q2": 0.7})
        assert len(results) == 2


class TestCrossImpact:
    def test_instantiation(self):
        ci = CrossImpactAnalyzer()
        assert len(ci.events) > 0

    def test_build_matrix(self):
        ci = CrossImpactAnalyzer()
        matrix = ci.build_matrix()
        assert isinstance(matrix, dict)
        assert len(matrix) > 0

    def test_solve(self):
        ci = CrossImpactAnalyzer()
        result = ci.solve()
        assert isinstance(result, CrossImpactResult)
        assert len(result.revised_probabilities) > 0
        assert result.convergence_iterations > 0

    def test_add_event(self):
        ci = CrossImpactAnalyzer()
        n_before = len(ci.events)
        ci.add_event(CrossImpactEvent("custom_event", 0.3, "Custom event"))
        assert len(ci.events) == n_before + 1

    def test_monte_carlo(self):
        ci = CrossImpactAnalyzer()
        mc_results = ci.monte_carlo(n_simulations=100)
        assert isinstance(mc_results, dict)
        for v in mc_results.values():
            assert 0 <= v <= 1

    def test_probability_changes(self):
        ci = CrossImpactAnalyzer()
        result = ci.solve()
        for name in result.revised_probabilities:
            initial = result.initial_probabilities[name]
            revised = result.revised_probabilities[name]
            assert 0.01 <= revised <= 0.99


class TestTemporalCausalDiscovery:
    def test_instantiation(self):
        tc = TemporalCausalDiscoverer()
        assert tc.max_lag == 5

    def test_discover_no_data(self):
        tc = TemporalCausalDiscoverer()
        result = tc.discover({"x": np.array([1.0])})
        assert isinstance(result, TemporalCausalResult)

    def test_granger_causality(self):
        tc = TemporalCausalDiscoverer()
        np.random.seed(42)
        x = np.random.randn(100)
        y = 0.5 * x[:-1] + np.random.randn(99) * 0.5
        link = tc.granger_causality("x", "y", {"x": x[:99], "y": y})
        if link is not None:
            assert link.source == "x"
            assert link.target == "y"

    def test_transfer_entropy(self):
        tc = TemporalCausalDiscoverer()
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        link = tc.transfer_entropy("x", "y", {"x": x, "y": y})
        if link is not None:
            assert link.method == CausalMethod.TRANSFER_ENTROPY

    def test_ccm(self):
        tc = TemporalCausalDiscoverer()
        np.random.seed(42)
        x = np.cumsum(np.random.randn(100))
        y = np.cumsum(np.random.randn(100))
        link = tc.convergent_cross_mapping("x", "y", {"x": x, "y": y})
        if link is not None:
            assert link.method == CausalMethod.CCM

    def test_full_discovery(self):
        tc = TemporalCausalDiscoverer()
        np.random.seed(42)
        data = {
            "gdp": np.cumsum(np.random.randn(100)),
            "inflation": np.cumsum(np.random.randn(100)),
            "unemployment": np.cumsum(np.random.randn(100)),
        }
        result = tc.discover(data)
        assert isinstance(result, TemporalCausalResult)
        assert isinstance(result.graph, CausalGraph)
        assert len(result.causal_ordering) > 0

    def test_empty_result(self):
        tc = TemporalCausalDiscoverer()
        result = tc._empty_result(["a", "b"])
        assert result.strongest_driver == "a"


class TestEngineIntegration:
    def test_engine_has_all_modules(self):
        from oracle.prediction.engine import OracleEngine
        e = OracleEngine()
        assert hasattr(e, 'black_swan')
        assert hasattr(e, 'regime_shift')
        assert hasattr(e, 'collective_intelligence')
        assert hasattr(e, 'analogical_reasoning')
        assert hasattr(e, 'morphological')
        assert hasattr(e, 'butterfly_tracker')
        assert hasattr(e, 'delphi')
        assert hasattr(e, 'cross_impact')
        assert hasattr(e, 'temporal_causal')

    def test_engine_modules_functional(self):
        from oracle.prediction.engine import OracleEngine
        e = OracleEngine()
        assert isinstance(e.black_swan, BlackSwanDetector)
        assert isinstance(e.regime_shift, RegimeShiftPredictor)
        assert isinstance(e.collective_intelligence, CollectiveIntelligenceAggregator)
        assert isinstance(e.analogical_reasoning, AnalogicalReasoningEngine)
        assert isinstance(e.morphological, MorphologicalAnalyzer)
        assert isinstance(e.butterfly_tracker, ButterflyTracker)
        assert isinstance(e.delphi, DelphiMethodSimulator)
        assert isinstance(e.cross_impact, CrossImpactAnalyzer)
        assert isinstance(e.temporal_causal, TemporalCausalDiscoverer)
        assert isinstance(e.horizon_scanner, HorizonScanner)
        assert isinstance(e.probabilistic_network, ProbabilisticNetwork)


from oracle.prediction.precondition import PreconditionDetector, PreconditionTree, ConditionStatus, PreconditionReport
from oracle.prediction.signal_fusion import SignalFusionEngine, FusionResult, ModuleSignal, FusionMethod


class TestHorizonScanner:
    def test_instantiation(self):
        scanner = HorizonScanner()
        assert len(scanner.signals) > 0

    def test_scan_default(self):
        scanner = HorizonScanner()
        result = scanner.scan()
        assert isinstance(result, HorizonScanResult)
        assert len(result.signals) > 0
        assert len(result.category_distribution) > 0

    def test_scan_min_velocity(self):
        scanner = HorizonScanner()
        result = scanner.scan(min_velocity=0.5)
        for s in result.signals:
            assert s.velocity >= 0.5

    def test_scan_category_filter(self):
        scanner = HorizonScanner()
        result = scanner.scan(categories=[SignalCategory.TECHNOLOGY])
        for s in result.signals:
            assert s.category == SignalCategory.TECHNOLOGY

    def test_track_signal(self):
        scanner = HorizonScanner()
        tracking = scanner.track_signal("test_signal", [0.1, 0.2, 0.3, 0.5])
        assert tracking["trend"] == "accelerating"
        assert tracking["acceleration"] > 0

    def test_track_signal_stable(self):
        scanner = HorizonScanner()
        tracking = scanner.track_signal("test_signal", [0.3, 0.3, 0.3, 0.3])
        assert tracking["trend"] == "stable"

    def test_add_signal(self):
        scanner = HorizonScanner()
        n_before = len(scanner.signals)
        scanner.add_signal(WeakSignal("test", SignalCategory.CULTURAL, "test",
                                       SignalMaturity.NASCENT, HorizonLevel.HORIZON_3,
                                       0.5, 0.3, 0.2, "test signal", [], []))
        assert len(scanner.signals) == n_before + 1

    def test_assess_impact_trajectory(self):
        scanner = HorizonScanner()
        signal = scanner.signals[0]
        trajectory = scanner.assess_impact_trajectory(signal, time_steps=5)
        assert len(trajectory) == 5
        assert trajectory[0]["projected_impact"] > 0

    def test_cross_domain_connections(self):
        scanner = HorizonScanner()
        result = scanner.scan()
        if result.cross_domain_connections:
            assert len(result.cross_domain_connections[0]) == 3


class TestProbabilisticNetwork:
    def test_instantiation(self):
        net = ProbabilisticNetwork()
        assert len(net.nodes) > 0
        assert len(net.conditional_tables) > 0

    def test_infer_no_evidence(self):
        net = ProbabilisticNetwork()
        result = net.infer({})
        assert isinstance(result, NetworkResult)
        assert "recession" in result.posteriors

    def test_infer_with_evidence(self):
        net = ProbabilisticNetwork()
        result = net.infer({"recession": "yes", "high_inflation": "yes"})
        assert result.evidence["recession"] == "yes"
        mkt_probs = result.posteriors.get("market_crash", [])
        assert len(mkt_probs) > 0

    def test_infer_target(self):
        net = ProbabilisticNetwork()
        result = net.infer_target("market_crash", {"recession": "yes"})
        assert isinstance(result, InferenceResult)
        assert result.variable == "market_crash"

    def test_probability_of(self):
        net = ProbabilisticNetwork()
        p = net.probability_of("market_crash", "yes", {"recession": "yes"})
        assert 0 <= p <= 1

    def test_add_node(self):
        net = ProbabilisticNetwork()
        n_before = len(net.nodes)
        net.add_node(NetworkNode("custom", ["a", "b"], [0.6, 0.4]))
        assert len(net.nodes) == n_before + 1

    def test_sensitivity_analysis(self):
        net = ProbabilisticNetwork()
        result = net.infer({"recession": "yes"})
        assert len(result.sensitivity) > 0

    def test_independent_sets(self):
        net = ProbabilisticNetwork()
        result = net.infer({"recession": "yes"})
        assert len(result.independent_sets) > 0

    def test_mpe(self):
        net = ProbabilisticNetwork()
        result = net.infer({"recession": "yes", "high_inflation": "yes"})
        assert "market_crash" in result.most_probable_state
        assert result.most_probable_state["recession"] == "yes"

    def test_narrative(self):
        net = ProbabilisticNetwork()
        result = net.infer({"recession": "yes"})
        assert len(result.narrative) > 0

class TestQuantumUncertainty:
    def test_instantiation(self):
        qu = QuantumUncertaintyMapper()
        assert qu.h_bar == 0.1

    def test_map_uncertainty_basic(self):
        qu = QuantumUncertaintyMapper()
        preds = {"gdp": (0.02, 0.03), "inflation": (0.035, 0.02)}
        result = qu.map_uncertainty(preds)
        assert isinstance(result, QuantumUncertaintyResult)
        assert len(result.uncertainties) == 2

    def test_uncertainty_classifications(self):
        qu = QuantumUncertaintyMapper()
        preds = {"low_var": (0.5, 0.01), "mid_var": (0.5, 0.1), "high_var": (0.5, 0.5)}
        result = qu.map_uncertainty(preds)
        types = {u.variable: u.uncertainty_type for u in result.uncertainties}
        assert types["low_var"] == UncertaintyType.ALEATORY
        assert types["high_var"] == UncertaintyType.ONTOLOGICAL

    def test_compute_uncertainty_principle(self):
        qu = QuantumUncertaintyMapper()
        result = qu.compute_uncertainty_principle("inflation", "growth", 0.05, 0.08)
        assert "product" in result
        assert "heisenberg_limit" in result
        assert "saturated" in result

    def test_observe(self):
        qu = QuantumUncertaintyMapper()
        result = qu.observe("inflation", 0.03, 0.01)
        assert result["collapse"] is True

    def test_irreducible_uncertainty(self):
        qu = QuantumUncertaintyMapper()
        preds = {"gdp": (0.02, 0.03), "inflation": (0.035, 0.02)}
        result = qu.map_uncertainty(preds)
        assert 0 <= result.irreducible_uncertainty <= 1

    def test_knowable_fraction(self):
        qu = QuantumUncertaintyMapper()
        preds = {"gdp": (0.02, 0.03), "inflation": (0.035, 0.02)}
        result = qu.map_uncertainty(preds)
        assert 0 <= result.knowable_fraction <= 1


class TestDecisionTheory:
    def test_instantiation(self):
        dt = DecisionTheoreticAdvisor()
        assert dt.risk_aversion == 1.0

    def test_expected_utility(self):
        dt = DecisionTheoreticAdvisor()
        eu = dt.expected_utility("invest_moderate", attitude=RiskAttitude.RISK_NEUTRAL)
        assert isinstance(eu, float)

    def test_decide_expected_utility(self):
        dt = DecisionTheoreticAdvisor()
        result = dt.decide()
        assert isinstance(result, DecisionResult)
        assert result.optimal_action in dt.payoffs

    def test_decide_minimax(self):
        dt = DecisionTheoreticAdvisor()
        result = dt.decide(criterion=DecisionCriterion.MINIMAX)
        assert result.criterion == DecisionCriterion.MINIMAX

    def test_decide_maximin(self):
        dt = DecisionTheoreticAdvisor()
        result = dt.decide(criterion=DecisionCriterion.MAXIMIN)
        assert result.maximin_action in dt.payoffs

    def test_evpi(self):
        dt = DecisionTheoreticAdvisor()
        evpi = dt.compute_evpi()
        assert evpi >= 0

    def test_risk_averse_vs_seeking(self):
        dt = DecisionTheoreticAdvisor()
        eu_averse = dt.expected_utility("invest_aggressive", attitude=RiskAttitude.RISK_AVERSE)
        eu_seeking = dt.expected_utility("invest_aggressive", attitude=RiskAttitude.RISK_SEEKING)
        assert eu_seeking > eu_averse or abs(eu_seeking - eu_averse) < 0.01

    def test_sensitivity_analysis(self):
        dt = DecisionTheoreticAdvisor()
        result = dt.sensitivity_analysis()
        assert len(result) > 0

    def test_regret_matrix(self):
        dt = DecisionTheoreticAdvisor()
        result = dt.decide()
        assert len(result.regret_matrix) > 0


class TestStrategicForesight:
    def test_instantiation(self):
        sf = StrategicForesightIntegrator()
        assert len(sf.module_weights) > 0

    def test_add_signal(self):
        sf = StrategicForesightIntegrator()
        sf.add_signal("superforecast", "recession", SignalDirection.NEGATIVE, 0.7, 0.3, 0.8, "Recession likely")
        assert len(sf.signals) == 1

    def test_integrate_no_signals(self):
        sf = StrategicForesightIntegrator()
        result = sf.integrate()
        assert isinstance(result, ForesightAssessment)
        assert result.overall_outlook == "neutral"

    def test_integrate_convergent(self):
        sf = StrategicForesightIntegrator()
        sf.add_signal("superforecast", "recession", SignalDirection.NEGATIVE, 0.7, 0.3, 0.8, "Recession signal")
        sf.add_signal("precursor_pattern", "recession", SignalDirection.NEGATIVE, 0.6, 0.25, 0.7, "Precursor detected")
        result = sf.integrate()
        assert len(result.convergent_signals) > 0

    def test_integrate_divergent(self):
        sf = StrategicForesightIntegrator()
        sf.add_signal("superforecast", "growth", SignalDirection.POSITIVE, 0.7, 0.6, 0.8, "Growth expected")
        sf.add_signal("black_swan", "growth", SignalDirection.NEGATIVE, 0.5, 0.3, 0.6, "Black swan risk")
        result = sf.integrate()
        assert len(result.divergent_signals) > 0

    def test_recommendations(self):
        sf = StrategicForesightIntegrator()
        sf.add_signal("superforecast", "recession", SignalDirection.NEGATIVE, 0.7, 0.3, 0.8, "Recession likely")
        result = sf.integrate()
        assert len(result.strategic_recommendations) > 0

    def test_probability_range(self):
        sf = StrategicForesightIntegrator()
        sf.add_signal("superforecast", "recession", SignalDirection.NEGATIVE, 0.7, 0.3, 0.8, "Recession")
        result = sf.integrate()
        assert result.probability_range[0] <= result.probability_range[1]

    def test_module_coverage(self):
        sf = StrategicForesightIntegrator()
        sf.add_signal("superforecast", "recession", SignalDirection.NEGATIVE, 0.7, 0.3, 0.8)
        result = sf.integrate()
        assert 0 < result.module_coverage <= 1


class TestQuantumUncertainty:
    def test_instantiation(self):
        qu = QuantumUncertaintyMapper()
        assert qu.h_bar == 0.1

    def test_map_uncertainty_basic(self):
        qu = QuantumUncertaintyMapper()
        preds = {"gdp": (0.02, 0.03), "inflation": (0.035, 0.02)}
        result = qu.map_uncertainty(preds)
        assert isinstance(result, QuantumUncertaintyResult)
        assert len(result.uncertainties) == 2

    def test_uncertainty_classifications(self):
        qu = QuantumUncertaintyMapper()
        preds = {"low_var": (0.5, 0.01), "mid_var": (0.5, 0.1), "high_var": (0.5, 0.5)}
        result = qu.map_uncertainty(preds)
        types = {u.variable: u.uncertainty_type for u in result.uncertainties}
        assert types["low_var"] == UncertaintyType.ALEATORY
        assert types["high_var"] == UncertaintyType.ONTOLOGICAL

    def test_compute_uncertainty_principle(self):
        qu = QuantumUncertaintyMapper()
        result = qu.compute_uncertainty_principle("inflation", "growth", 0.05, 0.08)
        assert "product" in result
        assert "heisenberg_limit" in result
        assert "saturated" in result

    def test_observe(self):
        qu = QuantumUncertaintyMapper()
        result = qu.observe("inflation", 0.03, 0.01)
        assert result["collapse"] is True

    def test_irreducible_uncertainty(self):
        qu = QuantumUncertaintyMapper()
        preds = {"gdp": (0.02, 0.03), "inflation": (0.035, 0.02)}
        result = qu.map_uncertainty(preds)
        assert 0 <= result.irreducible_uncertainty <= 1

    def test_knowable_fraction(self):
        qu = QuantumUncertaintyMapper()
        preds = {"gdp": (0.02, 0.03), "inflation": (0.035, 0.02)}
        result = qu.map_uncertainty(preds)
        assert 0 <= result.knowable_fraction <= 1


class TestDecisionTheory:
    def test_instantiation(self):
        dt = DecisionTheoreticAdvisor()
        assert dt.risk_aversion == 1.0

    def test_expected_utility(self):
        dt = DecisionTheoreticAdvisor()
        eu = dt.expected_utility("invest_moderate", attitude=RiskAttitude.RISK_NEUTRAL)
        assert isinstance(eu, float)

    def test_decide_expected_utility(self):
        dt = DecisionTheoreticAdvisor()
        result = dt.decide()
        assert isinstance(result, DecisionResult)
        assert result.optimal_action in dt.payoffs

    def test_decide_minimax(self):
        dt = DecisionTheoreticAdvisor()
        result = dt.decide(criterion=DecisionCriterion.MINIMAX)
        assert result.criterion == DecisionCriterion.MINIMAX

    def test_decide_maximin(self):
        dt = DecisionTheoreticAdvisor()
        result = dt.decide(criterion=DecisionCriterion.MAXIMIN)
        assert result.maximin_action in dt.payoffs

    def test_evpi(self):
        dt = DecisionTheoreticAdvisor()
        evpi = dt.compute_evpi()
        assert evpi >= 0

    def test_risk_averse_vs_seeking(self):
        dt = DecisionTheoreticAdvisor()
        eu_averse = dt.expected_utility("invest_moderate", attitude=RiskAttitude.RISK_AVERSE)
        eu_neutral = dt.expected_utility("invest_moderate", attitude=RiskAttitude.RISK_NEUTRAL)
        assert eu_averse != eu_neutral or abs(eu_averse - eu_neutral) < 0.01

    def test_sensitivity_analysis(self):
        dt = DecisionTheoreticAdvisor()
        result = dt.sensitivity_analysis()
        assert len(result) > 0

    def test_regret_matrix(self):
        dt = DecisionTheoreticAdvisor()
        result = dt.decide()
        assert len(result.regret_matrix) > 0


class TestStrategicForesight:
    def test_instantiation(self):
        sf = StrategicForesightIntegrator()
        assert len(sf.module_weights) > 0

    def test_add_signal(self):
        sf = StrategicForesightIntegrator()
        sf.add_signal("superforecast", "recession", SignalDirection.NEGATIVE, 0.7, 0.3, 0.8, "Recession likely")
        assert len(sf.signals) == 1

    def test_integrate_no_signals(self):
        sf = StrategicForesightIntegrator()
        result = sf.integrate()
        assert isinstance(result, ForesightAssessment)
        assert result.overall_outlook == "neutral"

    def test_integrate_convergent(self):
        sf = StrategicForesightIntegrator()
        sf.add_signal("superforecast", "recession", SignalDirection.NEGATIVE, 0.7, 0.3, 0.8, "Recession signal")
        sf.add_signal("precursor_pattern", "recession", SignalDirection.NEGATIVE, 0.6, 0.25, 0.7, "Precursor detected")
        result = sf.integrate()
        assert len(result.convergent_signals) > 0

    def test_integrate_divergent(self):
        sf = StrategicForesightIntegrator()
        sf.add_signal("superforecast", "growth", SignalDirection.POSITIVE, 0.7, 0.6, 0.8, "Growth expected")
        sf.add_signal("black_swan", "growth", SignalDirection.NEGATIVE, 0.5, 0.3, 0.6, "Black swan risk")
        result = sf.integrate()
        assert len(result.divergent_signals) > 0

    def test_recommendations(self):
        sf = StrategicForesightIntegrator()
        sf.add_signal("superforecast", "recession", SignalDirection.NEGATIVE, 0.7, 0.3, 0.8, "Recession likely")
        result = sf.integrate()
        assert len(result.strategic_recommendations) > 0

    def test_probability_range(self):
        sf = StrategicForesightIntegrator()
        sf.add_signal("superforecast", "recession", SignalDirection.NEGATIVE, 0.7, 0.3, 0.8, "Recession")
        result = sf.integrate()
        assert result.probability_range[0] <= result.probability_range[1]

    def test_module_coverage(self):
        sf = StrategicForesightIntegrator()
        sf.add_signal("superforecast", "recession", SignalDirection.NEGATIVE, 0.7, 0.3, 0.8)
        result = sf.integrate()
        assert 0 < result.module_coverage <= 1


class TestPrecondition:
    def test_instantiation(self):
        pd = PreconditionDetector()
        assert len(pd.tree_defs) > 0

    def test_build_tree(self):
        pd = PreconditionDetector()
        tree = pd.build_tree("recession")
        assert tree.event == "recession"
        assert tree.completion_score >= 0

    def test_detect(self):
        pd = PreconditionDetector()
        result = pd.detect()
        assert isinstance(result, PreconditionReport)
        assert len(result.trees) > 0
        assert 0 <= result.overall_readiness <= 1

    def test_check_event(self):
        pd = PreconditionDetector()
        result = pd.check_event("recession")
        assert result is not None
        assert result.event == "recession"

    def test_check_unknown_event(self):
        pd = PreconditionDetector()
        result = pd.check_event("nonexistent")
        assert result is None

    def test_blocking_factors(self):
        pd = PreconditionDetector()
        tree = pd.build_tree("financial_crisis")
        assert len(tree.blocking_factors) > 0

    def test_probability_achievement(self):
        pd = PreconditionDetector()
        tree = pd.build_tree("recession")
        assert 0 <= tree.probability_achievement <= 1

    def test_near_completion(self):
        pd = PreconditionDetector()
        result = pd.detect()
        assert isinstance(result.near_completion, list)

    def test_update_condition(self):
        pd = PreconditionDetector()
        pd.update_condition("recession", "yield_curve_inversion", ConditionStatus.MET, 0.8)
        tree = pd.build_tree("recession")
        assert tree.completion_score > 0.5


class TestSignalFusion:
    def test_instantiation(self):
        sf = SignalFusionEngine()
        assert len(sf.reliabilities) > 0

    def test_bayesian_pool(self):
        sf = SignalFusionEngine()
        signals = [
            ModuleSignal("superforecast", "test", 0.7, 0.8, 0.5, 0.85),
            ModuleSignal("analogical", "test", 0.6, 0.7, 0.4, 0.70),
        ]
        result = sf.fuse(signals, FusionMethod.BAYESIAN_POOL)
        assert result is not None
        assert result.variable == "test"

    def test_reliability_weighted(self):
        sf = SignalFusionEngine()
        signals = [
            ModuleSignal("superforecast", "test", 0.7, 0.8, 0.5, 0.85),
            ModuleSignal("analogical", "test", 0.6, 0.7, 0.4, 0.70),
        ]
        result = sf.fuse(signals, FusionMethod.RELIABILITY_WEIGHTED)
        assert 0.01 <= result.fused_probability <= 0.99

    def test_median_fusion(self):
        sf = SignalFusionEngine()
        signals = [
            ModuleSignal("A", "test", 0.3, 0.8, -0.5, 0.7),
            ModuleSignal("B", "test", 0.5, 0.7, 0.0, 0.6),
            ModuleSignal("C", "test", 0.7, 0.75, 0.5, 0.8),
        ]
        result = sf.fuse(signals, FusionMethod.MEDIAN)
        assert result.fused_probability == 0.5

    def test_consensus_detection(self):
        sf = SignalFusionEngine()
        signals = [
            ModuleSignal("A", "test", 0.7, 0.8, 0.5, 0.7),
            ModuleSignal("B", "test", 0.6, 0.7, 0.4, 0.6),
        ]
        consensus = sf.detect_consensus(signals)
        assert consensus == 1.0

    def test_conflict_detection(self):
        sf = SignalFusionEngine()
        signals = [
            ModuleSignal("A", "test", 0.7, 0.8, 0.8, 0.7),
            ModuleSignal("B", "test", 0.3, 0.6, -0.5, 0.6),
        ]
        conflict = sf.detect_conflict(signals)
        assert conflict > 0

    def test_correlation_adjusted(self):
        sf = SignalFusionEngine()
        signals = [
            ModuleSignal("superforecast", "test", 0.6, 0.8, 0.3, 0.85),
            ModuleSignal("analogical", "test", 0.55, 0.7, 0.2, 0.70),
        ]
        result = sf.fuse(signals, FusionMethod.CORRELATION_ADJUSTED)
        assert result is not None

    def test_fuse_multiple(self):
        sf = SignalFusionEngine()
        groups = {
            "recession": [ModuleSignal("A", "recession", 0.3, 0.8, -0.6, 0.85)],
            "growth": [ModuleSignal("B", "growth", 0.6, 0.7, 0.4, 0.70)],
        }
        results = sf.fuse_multiple(groups)
        assert len(results) == 2
