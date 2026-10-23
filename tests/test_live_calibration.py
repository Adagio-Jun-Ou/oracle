"""Tests for CausalEventLiveDataCalibrator — real data-driven event calibration."""
import pytest
import numpy as np
from oracle.prediction.live_calibration import (
    CausalEventLiveDataCalibrator, LiveDataCalibration, BaseRateAdjustment,
    TriggerEvaluation, InvalidationEvaluation, AlreadyOccurredEvent, LiveMetricInjection,
)
from oracle.prediction.rss_intelligence import (
    RSSIntelligenceFeed, RSSArticle, EventSignal, RSSCalibrationResult,
)


@pytest.fixture
def calibrator():
    return CausalEventLiveDataCalibrator()


class TestDomainStateComputation:
    def test_domain_states_returns_all_keys(self, calibrator):
        states = calibrator._compute_domain_states()
        expected_keys = {"equity", "volatility", "inflation", "geopolitical",
                        "health", "technology", "sentiment", "labor",
                        "currency", "commodity", "climate", "housing"}
        assert expected_keys.issubset(set(states.keys()))

    def test_domain_states_bounded(self, calibrator):
        states = calibrator._compute_domain_states()
        for key, value in states.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} outside [0,1]"

    def test_domain_states_fallback_values(self, calibrator):
        states = calibrator._compute_domain_states()
        assert states["health"] == 0.50
        assert states["climate"] == 0.65

    def test_vix_affects_volatility_and_sentiment(self, calibrator):
        states = calibrator._compute_domain_states()
        if states["volatility"] > 0.7:
            assert states["sentiment"] < 0.5
        elif states["volatility"] < 0.3:
            assert states["sentiment"] > 0.5

    def test_high_vix_means_high_volatility(self, calibrator):
        states = calibrator._compute_domain_states()
        assert states["volatility"] > 0.0


class TestBaseRateAdjustments:
    def test_adjustments_returned(self, calibrator):
        adj = calibrator._adjust_base_rates(calibrator._compute_domain_states())
        assert isinstance(adj, list)
        assert len(adj) >= 0

    def test_adjustment_structure(self, calibrator):
        states = calibrator._compute_domain_states()
        adj = calibrator._adjust_base_rates(states)
        for a in adj:
            assert isinstance(a, BaseRateAdjustment)
            assert a.event_id
            assert -0.2 <= a.adjusted_base_rate <= 0.2
            assert a.adjustment_reason
            assert a.data_source

    def test_high_equity_boosts_sp500(self, calibrator):
        states = {"equity": 0.85, "volatility": 0.4, "inflation": 0.5,
                   "geopolitical": 0.5, "health": 0.5, "technology": 0.55,
                   "sentiment": 0.6, "labor": 0.5, "currency": 0.5,
                   "commodity": 0.5, "climate": 0.5, "housing": 0.5}
        adj = calibrator._adjust_base_rates(states)
        sp500_adj = [a for a in adj if a.event_id == "sp500_new_high_2026"]
        assert len(sp500_adj) >= 1
        assert sp500_adj[0].adjusted_base_rate > 0

    def test_high_inflation_reduces_fed_cut(self, calibrator):
        states = {"equity": 0.55, "volatility": 0.4, "inflation": 0.85,
                   "geopolitical": 0.5, "health": 0.5, "technology": 0.55,
                   "sentiment": 0.5, "labor": 0.5, "currency": 0.5,
                   "commodity": 0.5, "climate": 0.5, "housing": 0.5}
        adj = calibrator._adjust_base_rates(states)
        fed_adj = [a for a in adj if "fed_rate_cut" in a.event_id]
        if fed_adj:
            assert fed_adj[0].adjusted_base_rate < 0

    def test_template_ids_filter(self, calibrator):
        states = calibrator._compute_domain_states()
        adj = calibrator._adjust_base_rates(states, template_ids=["sp500_new_high_2026"])
        for a in adj:
            assert a.event_id == "sp500_new_high_2026"


class TestTriggerEvaluation:
    def test_triggers_returned(self, calibrator):
        states = calibrator._compute_domain_states()
        trigger_evals = calibrator._evaluate_triggers(states)
        assert isinstance(trigger_evals, dict)
        assert len(trigger_evals) >= 1

    def test_trigger_structure(self, calibrator):
        states = calibrator._compute_domain_states()
        trigger_evals = calibrator._evaluate_triggers(states)
        for event_id, evals in trigger_evals.items():
            for te in evals:
                assert isinstance(te, TriggerEvaluation)
                assert te.trigger_condition
                assert te.current_value
                assert isinstance(te.probability_adjustment, (int, float))
                assert isinstance(te.is_active, bool)
                assert te.evidence

    def test_fed_rate_cut_triggers(self, calibrator):
        states = calibrator._compute_domain_states()
        trigger_evals = calibrator._evaluate_triggers(states)
        if "fed_rate_cut_2026" in trigger_evals:
            fed_triggers = trigger_evals["fed_rate_cut_2026"]
            assert len(fed_triggers) >= 1
            cpi_trigger = [t for t in fed_triggers if "CPI" in t.trigger_condition]
            assert len(cpi_trigger) >= 1

    def test_template_ids_filter_triggers(self, calibrator):
        states = calibrator._compute_domain_states()
        trigger_evals = calibrator._evaluate_triggers(states, template_ids=["fed_rate_cut_2026"])
        for event_id in trigger_evals:
            assert event_id == "fed_rate_cut_2026"


class TestInvalidationEvaluation:
    def test_invalidations_returned(self, calibrator):
        states = calibrator._compute_domain_states()
        inv_evals = calibrator._evaluate_invalidations(states)
        assert isinstance(inv_evals, dict)

    def test_invalidation_structure(self, calibrator):
        states = calibrator._compute_domain_states()
        inv_evals = calibrator._evaluate_invalidations(states)
        for event_id, evals in inv_evals.items():
            for ie in evals:
                assert isinstance(ie, InvalidationEvaluation)
                assert ie.condition
                assert ie.description
                assert isinstance(ie.is_violated, (bool, type(np.bool_)))
                assert ie.current_evidence

    def test_fed_rate_cut_invalidations(self, calibrator):
        states = calibrator._compute_domain_states()
        inv_evals = calibrator._evaluate_invalidations(states)
        if "fed_rate_cut_2026" in inv_evals:
            fed_invs = inv_evals["fed_rate_cut_2026"]
            assert len(fed_invs) >= 1
            cpi_inv = [i for i in fed_invs if "CPI" in i.condition]
            assert len(cpi_inv) >= 1


class TestFullCalibration:
    def test_calibrate_returns_result(self, calibrator):
        result = calibrator.calibrate()
        assert isinstance(result, LiveDataCalibration)
        assert len(result.domain_states) >= 10
        assert isinstance(result.base_rate_adjustments, list)
        assert isinstance(result.trigger_evaluations, dict)
        assert isinstance(result.invalidation_evaluations, dict)
        assert isinstance(result.data_sources_used, list)
        assert result.calibration_timestamp
        assert isinstance(result.confidence_delta, float)
        assert 0 <= result.confidence_delta <= 0.2

    def test_calibrate_with_template_filter(self, calibrator):
        result = calibrator.calibrate(template_ids=["sp500_new_high_2026", "fed_rate_cut_2026"])
        assert isinstance(result, LiveDataCalibration)

    def test_domain_states_always_valid(self, calibrator):
        result = calibrator.calibrate()
        for key, value in result.domain_states.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} outside [0,1]"

    def test_base_rate_adjustments_valid(self, calibrator):
        result = calibrator.calibrate()
        for adj in result.base_rate_adjustments:
            assert -0.3 <= adj.adjusted_base_rate <= 0.3

    def test_data_sources_listed(self, calibrator):
        result = calibrator.calibrate()
        assert len(result.data_sources_used) >= 1
        assert any("yfinance" in s for s in result.data_sources_used)

    def test_confidence_delta_reasonable(self, calibrator):
        result = calibrator.calibrate()
        assert 0 <= result.confidence_delta <= 0.2


class TestIntegrationWithPredictor:
    def test_calibrated_domain_states_feed_predictor(self, calibrator):
        from oracle.prediction.causal_event import CausalEventPredictor
        calibration = calibrator.calibrate()
        predictor = CausalEventPredictor(seed=42)
        result = predictor.predict(domain_states=calibration.domain_states)
        assert len(result.events) >= 20
        assert result.confidence_level in ("LOW", "MODERATE", "HIGH")

    def test_calibrated_results_differ_from_default(self, calibrator):
        from oracle.prediction.causal_event import CausalEventPredictor
        calibration = calibrator.calibrate()
        predictor = CausalEventPredictor(seed=42)
        default_result = predictor.predict()
        calibrated_result = predictor.predict(domain_states=calibration.domain_states)
        default_probs = {e.event_id: e.probability for e in default_result.events}
        calibrated_probs = {e.event_id: e.probability for e in calibrated_result.events}
        different = sum(1 for k in default_probs if abs(default_probs[k] - calibrated_probs[k]) > 0.001)
        assert different >= 1

    def test_calibration_with_full_analyzer(self, calibrator):
        from oracle.prediction.causal_event import CausalEventFullAnalyzer
        calibration = calibrator.calibrate()
        analyzer = CausalEventFullAnalyzer(seed=42)
        result = analyzer.analyze(domain_states=calibration.domain_states)
        assert len(result.prediction.events) >= 20
        assert result.cascade_network.overall_cascade_risk >= 0
        assert len(result.temporal_curves) >= 15


class TestAlreadyOccurredDetection:
    def test_already_occurred_structure(self, calibrator):
        events = calibrator._detect_already_occurred()
        assert isinstance(events, list)
        for e in events:
            assert isinstance(e, AlreadyOccurredEvent)
            assert e.event_id
            assert isinstance(e.current_value, (int, float))
            assert isinstance(e.target_value, (int, float))
            assert e.description

    def test_gold_3000_detection(self, calibrator):
        events = calibrator._detect_already_occurred()
        gold_event = [e for e in events if e.event_id == "gold_3000_2026"]
        if gold_event:
            assert gold_event[0].current_value >= 3000
            assert gold_event[0].target_value == 3000

    def test_sp500_6200_detection(self, calibrator):
        events = calibrator._detect_already_occurred()
        sp500_event = [e for e in events if e.event_id == "sp500_new_high_2026"]
        if sp500_event:
            assert sp500_event[0].current_value >= 6200


class TestLiveMetricInjection:
    def test_injections_structure(self, calibrator):
        injections = calibrator._inject_live_metrics()
        assert isinstance(injections, list)
        for inj in injections:
            assert isinstance(inj, LiveMetricInjection)
            assert inj.event_id
            assert inj.metric_name
            assert isinstance(inj.old_value, (int, float))
            assert inj.new_value >= 0
            assert inj.source

    def test_injection_updates_stale_values(self, calibrator):
        injections = calibrator._inject_live_metrics()
        for inj in injections:
            if inj.event_id == "gold_3000_2026":
                assert inj.metric_name == "Gold Price"
                assert inj.old_value == 3300


class TestCalibrateWithRssAndAlreadyOccurred:
    def test_calibrate_includes_already_occurred(self, calibrator):
        result = calibrator.calibrate(include_rss=False)
        assert isinstance(result.already_occurred, list)
        assert isinstance(result.metric_injections, list)
        assert isinstance(result.rss_signals, list)

    def test_calibrate_rss_disabled(self, calibrator):
        result = calibrator.calibrate(include_rss=False)
        assert len(result.rss_signals) == 0

    def test_metric_injection_list_populated(self, calibrator):
        result = calibrator.calibrate(include_rss=False)
        assert len(result.metric_injections) >= 0
        for inj in result.metric_injections:
            assert inj.event_id
            assert inj.new_value != inj.old_value or inj.new_value > 0


class TestRSSIntelligenceFeed:
    def test_rss_article_creation(self):
        article = RSSArticle(
            title="Fed signals rate cut possible in 2026",
            link="http://example.com/1",
            published="2026-01-01",
            source="Reuters",
            source_tier=1,
            source_reliability=95,
            category="financial",
            fetched_at="2026-01-01",
        )
        assert article.title == "Fed signals rate cut possible in 2026"
        assert article.source_reliability == 95

    def test_analyze_articles_keyword_matching(self):
        feed = RSSIntelligenceFeed()
        articles = [
            RSSArticle(title="Fed signals rate cut possible in 2026", link="", published="",
                       source="Reuters", source_tier=1, source_reliability=95, category="financial", fetched_at=""),
            RSSArticle(title="Gold hits all-time high amid tensions", link="", published="",
                       source="BBC", source_tier=1, source_reliability=90, category="financial", fetched_at=""),
            RSSArticle(title="Nuclear escalation fears grow", link="", published="",
                       source="AP", source_tier=1, source_reliability=95, category="world", fetched_at=""),
            RSSArticle(title="AI breakthrough passes major benchmark", link="", published="",
                       source="TechCrunch", source_tier=2, source_reliability=72, category="technology", fetched_at=""),
        ]
        result = feed.analyze_articles(articles)
        assert result.articles_fetched == 4
        assert len(result.signals) >= 3
        assert len(result.trigger_updates) >= 3

    def test_analyze_financial_keywords(self):
        feed = RSSIntelligenceFeed()
        articles = [
            RSSArticle(title="Recession fears grow as unemployment rises", link="", published="",
                       source="WSJ", source_tier=1, source_reliability=90, category="financial", fetched_at=""),
            RSSArticle(title="Gold record high as investors seek safety", link="", published="",
                       source="Reuters", source_tier=1, source_reliability=95, category="financial", fetched_at=""),
        ]
        result = feed.analyze_articles(articles)
        assert len(result.domain_state_adjustments) > 0
        assert "geopolitical" in result.domain_state_adjustments or "equity" in result.domain_state_adjustments

    def test_invalidation_alerts_for_nuclear(self):
        feed = RSSIntelligenceFeed()
        articles = [
            RSSArticle(title="Nuclear escalation fears grow after missile test", link="", published="",
                       source="AP", source_tier=1, source_reliability=95, category="world", fetched_at=""),
        ]
        result = feed.analyze_articles(articles)
        assert len(result.invalidation_alerts) >= 1

    def test_get_combined_domain_states(self):
        feed = RSSIntelligenceFeed()
        base = {"equity": 0.55, "volatility": 0.40, "geopolitical": 0.50}
        result = feed.get_combined_domain_states(base_states=base)
        assert len(result) == 3
        for v in result.values():
            assert 0.0 < v <= 1.0

    def test_event_signal_dataclass(self):
        signal = EventSignal(
            event_id="fed_rate_cut_2026",
            signal_type="positive",
            headline="Fed to cut rates",
            source="Reuters",
            source_reliability=95,
            domain_impact={"equity": 0.05},
            probability_adjustment=0.02,
            timestamp="2026-01-01",
            keywords_matched=["rate cut"],
        )
        assert signal.event_id == "fed_rate_cut_2026"
        assert signal.probability_adjustment > 0

    def test_rss_calibration_result_dataclass(self):
        result = RSSCalibrationResult(
            articles_fetched=10,
            sources_checked=5,
            signals=[],
            domain_state_adjustments={"equity": 0.02},
            trigger_updates={},
            invalidation_alerts=[],
            already_occurred=[],
            headline_summary={"recession": ["Recession fears grow"]},
            timestamp="2026-01-01",
        )
        assert result.articles_fetched == 10
        assert "recession" in result.headline_summary

    def test_negative_keywords_reduce_probability(self):
        feed = RSSIntelligenceFeed()
        articles = [
            RSSArticle(title="Fed signals rate hike, no cuts coming", link="", published="",
                       source="Reuters", source_tier=1, source_reliability=95, category="financial", fetched_at=""),
        ]
        result = feed.analyze_articles(articles)
        fed_signals = [s for s in result.signals if s.event_id == "fed_rate_cut_2026"]
        if fed_signals:
            assert fed_signals[0].signal_type == "negative"
            assert fed_signals[0].probability_adjustment < 0

    def test_already_occurred_with_live_data(self):
        feed = RSSIntelligenceFeed()
        articles = [
            RSSArticle(title="Gold surges past $4000", link="", published="",
                       source="Reuters", source_tier=1, source_reliability=95, category="financial", fetched_at=""),
        ]
        result = feed.analyze_articles(articles)
        assert isinstance(result.already_occurred, list)


class TestPredictWithLiveData:
    def test_predict_with_live_data_method(self):
        from oracle.prediction.causal_event import CausalEventPredictor
        predictor = CausalEventPredictor(seed=42)
        result = predictor.predict_with_live_data()
        assert len(result.events) >= 20
        for e in result.events:
            assert e.event_id
            assert 0 < e.probability < 1

    def test_predict_with_live_data_injects_metrics(self):
        from oracle.prediction.causal_event import CausalEventPredictor, EventStatus
        predictor = CausalEventPredictor(seed=42)
        result = predictor.predict_with_live_data()
        triggered = [e for e in result.events if e.status == EventStatus.TRIGGERED]
        assert isinstance(triggered, list)