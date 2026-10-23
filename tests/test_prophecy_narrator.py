"""Tests for ProphecyNarrator — AI-powered oracular narrative synthesis."""
import pytest
from oracle.prediction.prophecy_narrator import (
    ProphecyNarrator, ProphecyHorizon, ProphecyConfidence,
    MultiHorizonForecast, CascadeResult, CascadeStep,
    CounterfactualResult, PredictionScore, ProphecyNarration,
    CASCADE_CHAINS, EVIDENCE_DB, DOMAIN_NAMES
)


BULLISH_STATES = {"equity": 0.75, "volatility": 0.25, "commodity": 0.55, "inflation": 0.45,
                  "labor": 0.70, "monetary": 0.60, "geopolitical": 0.35, "sentiment": 0.65,
                  "technology": 0.72, "economic": 0.68}
BEARISH_STATES = {"equity": 0.25, "volatility": 0.75, "commodity": 0.40, "inflation": 0.70,
                  "labor": 0.30, "monetary": 0.30, "geopolitical": 0.65, "sentiment": 0.30,
                  "technology": 0.45, "economic": 0.30}
NEUTRAL_STATES = {"equity": 0.52, "volatility": 0.48, "commodity": 0.50, "inflation": 0.50,
                  "labor": 0.55, "monetary": 0.52, "geopolitical": 0.48, "sentiment": 0.51,
                  "technology": 0.53, "economic": 0.50}


@pytest.fixture
def narrator():
    return ProphecyNarrator(use_llm=False, seed=42)


class TestMultiHorizonForecasts:
    def test_all_five_horizons(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES)
        assert len(result.multi_horizon_forecasts) == 5
        horizons = [f.horizon for f in result.multi_horizon_forecasts]
        assert ProphecyHorizon.IMMEDIATE in horizons
        assert ProphecyHorizon.DISTANT in horizons

    def test_bullish_states_ascending(self, narrator):
        result = narrator.prophesize(BULLISH_STATES)
        assert result.overall_trajectory.startswith("ascending")

    def test_bearish_states_descending(self, narrator):
        result = narrator.prophesize(BEARISH_STATES)
        assert result.overall_trajectory.startswith("descending")

    def test_neutral_states_balanced(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES)
        assert result.overall_trajectory == "balanced"

    def test_forecast_domains(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES)
        imm = next(f for f in result.multi_horizon_forecasts if f.horizon == ProphecyHorizon.IMMEDIATE)
        for domain in NEUTRAL_STATES:
            assert domain in imm.domains
            assert 0 <= imm.domains[domain] <= 1

    def test_immediate_more_confident_than_distant(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES)
        imm = next(f for f in result.multi_horizon_forecasts if f.horizon == ProphecyHorizon.IMMEDIATE)
        dist = next(f for f in result.multi_horizon_forecasts if f.horizon == ProphecyHorizon.DISTANT)
        imm_variance = sum((v - 0.5) ** 2 for v in imm.domains.values())
        dist_variance = sum((v - 0.5) ** 2 for v in dist.domains.values())


class TestCascadeAnalysis:
    def test_auto_detect_yield_curve(self, narrator):
        result = narrator.prophesize({"monetary": 0.30})
        assert len(result.cascade_analyses) >= 1
        assert any("yield" in c.chain_name for c in result.cascade_analyses)

    def test_auto_detect_inflation(self, narrator):
        result = narrator.prophesize({"inflation": 0.75})
        assert any("inflation" in c.chain_name for c in result.cascade_analyses)

    def test_explicit_trigger(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES, cascade_triggers=["tech_disruption"])
        assert len(result.cascade_analyses) >= 1
        assert result.cascade_analyses[0].chain_name == "tech_disruption"

    def test_cascade_has_steps(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES, cascade_triggers=["yield_curve_recession"])
        for c in result.cascade_analyses:
            assert len(c.steps) >= 3
            for step in c.steps:
                assert 0 < step.probability <= 1
                assert step.evidence

    def test_cascade_cumulative_probability(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES, cascade_triggers=["yield_curve_recession"])
        for c in result.cascade_analyses:
            assert 0 < c.cumulative_probability < 1


class TestCounterfactuals:
    def test_five_counterfactuals(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES)
        assert len(result.counterfactuals) == 5

    def test_counterfactual_has_delta(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES)
        for cf in result.counterfactuals:
            assert cf.probability_delta != 0
            assert cf.probability_original >= 0
            assert len(cf.cascade_effects) >= 0
            assert cf.evidence

    def test_fed_cut_increases_probability(self, narrator):
        result = narrator.prophesize(BULLISH_STATES)
        fed_cut = next(cf for cf in result.counterfactuals if "Fed cuts" in cf.scenario)
        assert fed_cut.probability_delta > 0

    def test_geopolitical_decreases_probability(self, narrator):
        result = narrator.prophesize(BULLISH_STATES)
        geo = next(cf for cf in result.counterfactuals if "geopolitical" in cf.scenario)
        assert geo.probability_delta < 0


class TestJunctures:
    def test_inflection_detected(self, narrator):
        result = narrator.prophesize({"equity": 0.52})
        assert any("inflection" in j.lower() for j in result.key_junctures)

    def test_distressed_detected(self, narrator):
        result = narrator.prophesize({"equity": 0.20})
        assert any("distressed" in j.lower() for j in result.key_junctures)

    def test_extended_highs_detected(self, narrator):
        result = narrator.prophesize({"equity": 0.85})
        assert any("extended" in j.lower() or "mean reversion" in j.lower() for j in result.key_junctures)


class TestPronouncement:
    def test_balanced_pronouncement(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES)
        assert result.pronouncement
        assert len(result.pronouncement) > 20

    def test_bullish_pronouncement(self, narrator):
        result = narrator.prophesize(BULLISH_STATES)
        assert any(word in result.pronouncement.lower() for word in ["favorable", "bright", "growth", "ascending"])

    def test_bearish_pronouncement(self, narrator):
        result = narrator.prophesize(BEARISH_STATES)
        assert result.pronouncement
        assert len(result.pronouncement) > 20


class TestRecommendations:
    def test_bullish_recommendations(self, narrator):
        result = narrator.prophesize(BULLISH_STATES)
        assert len(result.recommendations) >= 2
        assert any("risk-on" in r.lower() or "growth" in r.lower() for r in result.recommendations)

    def test_bearish_recommendations(self, narrator):
        result = narrator.prophesize(BEARISH_STATES)
        assert any("risk-off" in r.lower() or "caution" in r.lower() for r in result.recommendations)

    def test_strongest_signal_in_recs(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES)
        assert any("Strongest signal" in r for r in result.recommendations)


class TestPredictionTracker:
    def test_record_and_resolve(self, narrator):
        pid = narrator.record_prediction("equity", 0.65, 0.55, 0.75, "short_term", "high")
        assert pid.startswith("equity_")
        score = narrator.resolve_prediction(pid, 0.68)
        assert score.resolved_predictions == 1
        assert score.within_ci_pct > 0
        assert score.accuracy_grade in ("A", "B", "C", "D", "F")

    def test_score_before_resolve(self, narrator):
        narrator.record_prediction("equity", 0.60, 0.50, 0.70, "medium_term", "moderate")
        score = narrator._compute_scores()
        assert score.resolved_predictions == 0
        assert score.accuracy_grade == "N/A"

    def test_multiple_predictions(self, narrator):
        p1 = narrator.record_prediction("equity", 0.65, 0.55, 0.75, "short_term", "high")
        p2 = narrator.record_prediction("bonds", 0.50, 0.40, 0.60, "medium_term", "moderate")
        narrator.resolve_prediction(p1, 0.68)
        narrator.resolve_prediction(p2, 0.55)
        score = narrator._compute_scores()
        assert score.total_predictions == 2
        assert score.resolved_predictions == 2


class TestNarrative:
    def test_narrative_nonempty(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES)
        assert result.narrative
        assert len(result.narrative) > 20

    def test_evidence_citations(self, narrator):
        result = narrator.prophesize({"monetary": 0.30, "volatility": 0.70})
        assert len(result.evidence_citations) >= 1

    def test_llm_synthesis_empty_when_disabled(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES)
        assert result.llm_synthesis == ""


class TestConfidenceClassification:
    def test_oracle_confidence(self, narrator):
        result = narrator.prophesize(BULLISH_STATES, domain_confidences={d: 0.9 for d in BULLISH_STATES})
        assert result.overall_confidence in (ProphecyConfidence.ORACLE, ProphecyConfidence.HIGH)

    def test_low_confidence(self, narrator):
        result = narrator.prophesize(NEUTRAL_STATES, domain_confidences={d: 0.2 for d in NEUTRAL_STATES})
        assert result.overall_confidence in (ProphecyConfidence.LOW, ProphecyConfidence.DIVINED, ProphecyConfidence.MODERATE)


class TestReproducibility:
    def test_same_seed_same_result(self):
        n1 = ProphecyNarrator(use_llm=False, seed=42)
        n2 = ProphecyNarrator(use_llm=False, seed=42)
        r1 = n1.prophesize(NEUTRAL_STATES)
        r2 = n2.prophesize(NEUTRAL_STATES)
        assert abs(r1.multi_horizon_forecasts[0].overall_prediction - r2.multi_horizon_forecasts[0].overall_prediction) < 0.01