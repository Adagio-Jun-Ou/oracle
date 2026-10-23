"""Tests for CausalEventPredictor — specific future event predictions."""
import pytest
from oracle.prediction.causal_event import (
    CausalEventPredictor, CausalPredictionResult, PredictedEvent,
    CausalStep, TriggerCondition, InvalidationCondition,
    EventCategory, ProbabilityBand, EventStatus, EVENT_TEMPLATES,
    CascadeNetwork, CascadeChain, CascadePath, CascadeNetworkResult,
    TemporalTimeline, TemporalCurve, TemporalPoint, InteractionEdge,
    CausalEventFullResult, CausalEventFullAnalyzer,
    CASCADE_EDGES, EVENT_INTERACTION_MATRIX,
)


@pytest.fixture
def predictor():
    return CausalEventPredictor(seed=42)


class TestEventTemplates:
    def test_templates_exist(self):
        assert len(EVENT_TEMPLATES) >= 8

    def test_templates_have_required_fields(self):
        for t in EVENT_TEMPLATES:
            assert "id" in t
            assert "category" in t
            assert "description" in t
            assert "base_rate" in t
            assert 0 <= t["base_rate"] <= 1
            assert "causal_chain" in t
            assert len(t["causal_chain"]) >= 2
            assert "timing_window" in t
            assert "evidence" in t
            assert len(t["evidence"]) >= 2
            assert "alternative" in t

    def test_template_categories_span_life(self):
        categories = {t["category"] for t in EVENT_TEMPLATES}
        life_cats = {EventCategory.HEALTH, EventCategory.MEDICINE,
                   EventCategory.RELATIONSHIP, EventCategory.CAREER,
                   EventCategory.CLIMATE, EventCategory.SOCIAL, EventCategory.HOUSING,
                   EventCategory.AI_BREAKTHROUGH}
        overlap = categories & life_cats
        assert len(overlap) >= 5, f"Need 5+ life categories, got: {overlap}"

    def test_causal_chain_structure(self):
        for t in EVENT_TEMPLATES:
            for step in t["causal_chain"]:
                assert len(step) == 5
                step_num, event, mechanism, prob, evidence = step
                assert isinstance(step_num, int)
                assert isinstance(event, str) and len(event) > 10
                assert isinstance(mechanism, str) and len(mechanism) > 10
                assert 0 < prob <= 1
                assert isinstance(evidence, str) and len(evidence) > 5


class TestBasicPrediction:
    def test_predict_returns_result(self, predictor):
        result = predictor.predict()
        assert isinstance(result, CausalPredictionResult)
        assert len(result.events) >= 8

    def test_events_have_all_fields(self, predictor):
        result = predictor.predict()
        for event in result.events:
            assert isinstance(event, PredictedEvent)
            assert event.event_id
            assert isinstance(event.category, EventCategory)
            assert len(event.event_description) > 20
            assert 0 < event.probability < 1
            assert isinstance(event.probability_band, ProbabilityBand)
            assert event.timing_earliest
            assert event.timing_latest
            assert len(event.causal_chain) >= 2
            assert len(event.evidence_citations) >= 2
            assert 0 < event.confidence <= 1.5
            assert 0 < event.base_rate < 1
            assert event.key_metric
            assert event.alternative_outcome
            assert 0 < event.alternative_probability < 1

    def test_events_sorted_by_probability(self, predictor):
        result = predictor.predict()
        probs = [e.probability for e in result.events]
        assert probs == sorted(probs, reverse=True)

    def test_high_probability_events(self, predictor):
        result = predictor.predict()
        assert len(result.high_probability_events) >= 1
        for e in result.high_probability_events:
            assert e.probability >= 0.6

    def test_contrarian_picks(self, predictor):
        result = predictor.predict()
        assert len(result.contrarian_picks) >= 1
        for e in result.contrarian_picks:
            assert e.probability <= 0.4


class TestCausalChains:
    def test_causal_chain_steps(self, predictor):
        result = predictor.predict()
        for event in result.events[:3]:
            assert len(event.causal_chain) >= 2
            for step in event.causal_chain:
                assert isinstance(step, CausalStep)
                assert step.step_number >= 1
                assert len(step.event) > 10
                assert len(step.mechanism) > 10
                assert 0 < step.probability_given_previous <= 1
                assert len(step.evidence) > 5

    def test_evidence_citations(self, predictor):
        result = predictor.predict()
        for event in result.events:
            assert len(event.evidence_citations) >= 2
            for citation in event.evidence_citations:
                assert len(citation) > 20


class TestProbabilityAdjustments:
    def test_high_equity_boosts_sp500(self, predictor):
        result_normal = predictor.predict({"equity": 0.55})
        result_high = predictor.predict({"equity": 0.80})
        sp500_normal = [e for e in result_normal.events if e.event_id == "sp500_new_high_2026"]
        sp500_high = [e for e in result_high.events if e.event_id == "sp500_new_high_2026"]
        if sp500_normal and sp500_high:
            assert sp500_high[0].probability >= sp500_normal[0].probability

    def test_high_inflation_reduces_rate_cut(self, predictor):
        result_low = predictor.predict({"inflation": 0.25})
        result_high = predictor.predict({"inflation": 0.75})
        fed_low = [e for e in result_low.events if e.event_id == "fed_rate_cut_2026"]
        fed_high = [e for e in result_high.events if e.event_id == "fed_rate_cut_2026"]
        if fed_low and fed_high:
            assert fed_low[0].probability >= fed_high[0].probability

    def test_technology_boosts_agi(self, predictor):
        result_low = predictor.predict({"technology": 0.40})
        result_high = predictor.predict({"technology": 0.80})
        agi_low = [e for e in result_low.events if e.event_id == "first_agi_benchmark_pass_2026"]
        agi_high = [e for e in result_high.events if e.event_id == "first_agi_benchmark_pass_2026"]
        if agi_low and agi_high:
            assert agi_high[0].probability >= agi_low[0].probability


class TestTriggerConditions:
    def test_triggers_present(self, predictor):
        result = predictor.predict()
        for event in result.events:
            assert len(event.triggers) >= 1
            for trigger in event.triggers:
                assert isinstance(trigger, TriggerCondition)
                assert trigger.condition
                assert isinstance(trigger.probability_boost, float)

    def test_invalidation_conditions(self, predictor):
        result = predictor.predict()
        for event in result.events:
            assert len(event.invalidation_conditions) >= 1
            for inv in event.invalidation_conditions:
                assert isinstance(inv, InvalidationCondition)
                assert inv.condition
                assert inv.description


class TestLifeEvents:
    def test_health_events_present(self, predictor):
        result = predictor.predict()
        health_events = [e for e in result.events if e.category == EventCategory.HEALTH]
        assert len(health_events) >= 1

    def test_career_events_present(self, predictor):
        result = predictor.predict()
        career_events = [e for e in result.events if e.category == EventCategory.CAREER]
        assert len(career_events) >= 1

    def test_medicine_events_present(self, predictor):
        result = predictor.predict()
        med_events = [e for e in result.events if e.category == EventCategory.MEDICINE]
        assert len(med_events) >= 1

    def test_social_events_present(self, predictor):
        result = predictor.predict()
        social_events = [e for e in result.events if e.category == EventCategory.SOCIAL]
        assert len(social_events) >= 1

    def test_climate_events_present(self, predictor):
        result = predictor.predict()
        climate_events = [e for e in result.events if e.category == EventCategory.CLIMATE]
        assert len(climate_events) >= 1

    def test_housing_events_present(self, predictor):
        result = predictor.predict()
        housing_events = [e for e in result.events if e.category == EventCategory.HOUSING]
        assert len(housing_events) >= 1

    def test_relationship_events_present(self, predictor):
        result = predictor.predict()
        rel_events = [e for e in result.events if e.category == EventCategory.RELATIONSHIP]
        assert len(rel_events) >= 1

    def test_ai_breakthrough_events(self, predictor):
        result = predictor.predict()
        ai_events = [e for e in result.events if e.category == EventCategory.AI_BREAKTHROUGH]
        assert len(ai_events) >= 1


class TestAssessment:
    def test_assessment_not_empty(self, predictor):
        result = predictor.predict()
        assert result.overall_assessment
        assert len(result.overall_assessment) > 50

    def test_risks_and_opportunities(self, predictor):
        result = predictor.predict()
        assert len(result.key_risks) >= 1
        assert len(result.key_opportunities) >= 1

    def test_confidence_level(self, predictor):
        result = predictor.predict()
        assert result.confidence_level in ("LOW", "MODERATE", "HIGH")

    def test_base_rates_are_realistic(self, predictor):
        result = predictor.predict()
        for event in result.events:
            assert 0.1 <= event.base_rate <= 0.9, \
                f"{event.event_id}: base_rate {event.base_rate} outside realistic range"

    def test_alternative_probability(self, predictor):
        result = predictor.predict()
        for event in result.events:
            assert 0.01 <= event.alternative_probability <= 0.99


class TestNewEventTemplates:
    def test_longevity_breakthrough(self, predictor):
        result = predictor.predict()
        longevity = [e for e in result.events if e.event_id == "longevity_breakthrough_2026"]
        assert len(longevity) == 1
        e = longevity[0]
        assert e.category == EventCategory.HEALTH
        assert 0.1 <= e.base_rate <= 0.5
        assert len(e.causal_chain) >= 3
        assert len(e.evidence_citations) >= 3

    def test_ai_job_displacement(self, predictor):
        result = predictor.predict()
        jobs = [e for e in result.events if e.event_id == "ai_job_displacement_5pct_2026"]
        assert len(jobs) == 1
        assert jobs[0].category == EventCategory.LABOR

    def test_mental_health_crisis(self, predictor):
        result = predictor.predict()
        mh = [e for e in result.events if e.event_id == "mental_health_crisis_2026"]
        assert len(mh) == 1
        assert mh[0].category == EventCategory.HEALTH

    def test_democracy_stress(self, predictor):
        result = predictor.predict()
        dem = [e for e in result.events if e.event_id == "democracy_stress_index_2026"]
        assert len(dem) == 1
        assert dem[0].category == EventCategory.POLITICAL

    def test_quantum_advantage(self, predictor):
        result = predictor.predict()
        q = [e for e in result.events if e.event_id == "quantum_advantage_useful_2026"]
        assert len(q) == 1
        assert q[0].category == EventCategory.SCIENCE

    def test_student_debt_crisis(self, predictor):
        result = predictor.predict()
        sd = [e for e in result.events if e.event_id == "student_debt_crisis_2026"]
        assert len(sd) == 1
        assert sd[0].category == EventCategory.EDUCATION

    def test_nuclear_near_miss(self, predictor):
        result = predictor.predict()
        n = [e for e in result.events if e.event_id == "nuclear_near_miss_2026"]
        assert len(n) == 1
        assert n[0].category == EventCategory.GEOPOLITICAL

    def test_pandemic_treaty(self, predictor):
        result = predictor.predict()
        pt = [e for e in result.events if e.event_id == "pandemic_preparedness_treaty_2026"]
        assert len(pt) == 1

    def test_longevity_escape_velocity(self, predictor):
        result = predictor.predict()
        lev = [e for e in result.events if e.event_id == "longevity_escape_velocity_2030"]
        assert len(lev) == 1
        assert lev[0].category == EventCategory.MEDICINE

    def test_total_event_count(self, predictor):
        result = predictor.predict()
        assert len(result.events) >= 24


class TestCascadeNetwork:
    @pytest.fixture
    def cascade(self):
        return CascadeNetwork(seed=42)

    def test_cascade_analyze(self, cascade):
        predictor = CausalEventPredictor(seed=42)
        events = predictor.predict().events
        result = cascade.analyze(events)
        assert isinstance(result, CascadeNetworkResult)
        assert len(result.chains) >= 1
        assert result.overall_cascade_risk >= 0
        assert len(result.systemic_risks) >= 1
        assert result.most_likely_cascade is not None
        assert result.highest_impact_cascade is not None

    def test_cascade_edges(self, cascade):
        predictor = CausalEventPredictor(seed=42)
        events = predictor.predict().events
        result = cascade.analyze(events)
        assert len(result.interaction_edges) >= 10
        for edge in result.interaction_edges:
            assert isinstance(edge, InteractionEdge)
            assert edge.source
            assert edge.target
            assert -1 <= edge.strength <= 1

    def test_cascade_chains_have_paths(self, cascade):
        predictor = CausalEventPredictor(seed=42)
        events = predictor.predict().events
        result = cascade.analyze(events)
        for chain in result.chains:
            assert len(chain.chain) >= 1
            assert chain.total_probability_impact > 0
            assert chain.risk_level in ("LOW", "MODERATE", "HIGH", "CRITICAL")
            assert chain.narrative

    def test_amplifying_vs_dampening(self, cascade):
        predictor = CausalEventPredictor(seed=42)
        events = predictor.predict().events
        result = cascade.analyze(events)
        amplifying = [e for e in result.interaction_edges if e.direction == "amplifying"]
        dampening = [e for e in result.interaction_edges if e.direction == "dampening"]
        assert len(amplifying) >= 1
        assert len(dampening) >= 1

    def test_cascade_with_domain_states(self, cascade):
        predictor = CausalEventPredictor(seed=42)
        events = predictor.predict({"equity": 0.8}).events
        result = cascade.analyze(events)
        assert result.overall_cascade_risk >= 0

    def test_cascade_risk_nonzero(self, cascade):
        predictor = CausalEventPredictor(seed=42)
        events = predictor.predict().events
        result = cascade.analyze(events)
        assert result.overall_cascade_risk > 0

    def test_protective_cascades(self, cascade):
        predictor = CausalEventPredictor(seed=42)
        events = predictor.predict().events
        result = cascade.analyze(events)
        assert isinstance(result.protective_cascades, list)


class TestTemporalTimeline:
    @pytest.fixture
    def timeline(self):
        return TemporalTimeline(seed=42)

    def test_generate_curves(self, timeline):
        predictor = CausalEventPredictor(seed=42)
        events = predictor.predict().events
        curves = timeline.generate(events)
        assert len(curves) == len(events)

    def test_curve_structure(self, timeline):
        predictor = CausalEventPredictor(seed=42)
        events = predictor.predict().events
        curves = timeline.generate(events)
        for curve in curves:
            assert isinstance(curve, TemporalCurve)
            assert curve.event_id
            assert curve.event_description
            assert len(curve.points) >= 4
            assert curve.peak_probability > 0
            assert curve.peak_date
            assert curve.shape in ("early_peak", "mid_peak", "late_peak")

    def test_temporal_points(self, timeline):
        predictor = CausalEventPredictor(seed=42)
        events = predictor.predict().events
        curves = timeline.generate(events)
        for curve in curves:
            for point in curve.points:
                assert isinstance(point, TemporalPoint)
                assert 0 < point.probability < 1
                assert 0 < point.confidence <= 1.5
                assert len(point.date) == 10

    def test_peak_probability_matches(self, timeline):
        predictor = CausalEventPredictor(seed=42)
        events = predictor.predict().events
        curves = timeline.generate(events)
        for curve in curves:
            max_point_prob = max(p.probability for p in curve.points)
            assert abs(curve.peak_probability - max_point_prob) < 0.05


class TestCausalEventFullAnalyzer:
    def test_full_analysis(self):
        analyzer = CausalEventFullAnalyzer(seed=42)
        result = analyzer.analyze()
        assert isinstance(result, CausalEventFullResult)
        assert len(result.prediction.events) >= 24
        assert len(result.cascade_network.chains) >= 1
        assert len(result.temporal_curves) >= 20
        assert len(result.event_interactions) >= 5

    def test_full_analysis_with_domain_states(self):
        analyzer = CausalEventFullAnalyzer(seed=42)
        result = analyzer.analyze(domain_states={"equity": 0.8, "technology": 0.8})
        assert len(result.prediction.events) >= 24
        assert result.cascade_network.overall_cascade_risk >= 0

    def test_interaction_edges_structure(self):
        analyzer = CausalEventFullAnalyzer(seed=42)
        result = analyzer.analyze()
        for edge in result.event_interactions:
            assert edge.source
            assert edge.target
            assert -1 <= edge.strength <= 1
            assert edge.mechanism
            assert edge.direction in ("amplifying", "dampening")

    def test_cascade_network_systemic_risks(self):
        analyzer = CausalEventFullAnalyzer(seed=42)
        result = analyzer.analyze()
        assert len(result.cascade_network.systemic_risks) >= 1
        for risk in result.cascade_network.systemic_risks:
            assert "cascade exposure" in risk


class TestEventInteractionMatrix:
    def test_matrix_exists(self):
        assert len(EVENT_INTERACTION_MATRIX) >= 10

    def test_matrix_entries_valid(self):
        for src, targets in EVENT_INTERACTION_MATRIX.items():
            for tgt, strength in targets.items():
                assert -1 <= strength <= 1, f"{src}->{tgt}: {strength}"

    def test_cascade_edges_valid(self):
        assert len(CASCADE_EDGES) >= 20
        for src, tgt, strength, mech in CASCADE_EDGES:
            assert src
            assert tgt
            assert -1 <= strength <= 1
            assert len(mech) > 10