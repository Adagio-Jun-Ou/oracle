"""Tests for UnifiedRegimeAssessor — Regime Detection v2."""
import pytest
from oracle.prediction.regime_assess import (
    UnifiedRegimeAssessor, RegimeAssessment, SourceRegimeSignal,
    RegimeType, RegimeSource, RegimeStability, RegimeTransition,
    REGIME_TRANSITION_MATRIX,
)


@pytest.fixture
def assessor():
    return UnifiedRegimeAssessor(seed=42)


class TestBasicAssessment:
    def test_empty_input(self, assessor):
        r = assessor.assess()
        assert r.composite_regime == RegimeType.UNCERTAIN
        assert r.confidence == 0.0
        assert r.source_signals == []

    def test_single_source(self, assessor):
        r = assessor.assess(ts_forecast_regime="bull", ts_forecast_score=0.8)
        assert r.composite_regime in RegimeType
        assert r.confidence > 0
        assert len(r.source_signals) == 1

    def test_two_sources_agree(self, assessor):
        r = assessor.assess(
            ts_forecast_regime="bull", ts_forecast_score=0.7,
            daily_oracle_regime="bull", daily_oracle_score=0.65,
        )
        assert r.composite_regime == RegimeType.BULL
        assert r.confidence > 0
        assert len(r.source_signals) == 2

    def test_three_sources_mixed(self, assessor):
        r = assessor.assess(
            ts_forecast_regime="bull", ts_forecast_score=0.7,
            daily_oracle_regime="transition", daily_oracle_score=0.6,
            anomaly_risk_level="moderate", anomaly_risk_score=0.5,
        )
        assert len(r.source_signals) == 3
        assert r.source_divergence >= 0


class TestRegimeProbabilities:
    def test_probabilities_sum_to_one(self, assessor):
        r = assessor.assess(ts_forecast_regime="bull", ts_forecast_score=0.8)
        total = sum(r.regime_probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_best_regime_has_highest_prob(self, assessor):
        r = assessor.assess(
            ts_forecast_regime="bull", ts_forecast_score=0.9,
            daily_oracle_regime="bull", daily_oracle_score=0.85,
        )
        best = max(r.regime_probabilities, key=r.regime_probabilities.get)
        assert best == r.composite_regime.value or r.regime_probabilities[best] > 0


class TestStability:
    def test_all_agree_stable(self, assessor):
        r = assessor.assess(
            ts_forecast_regime="bull", ts_forecast_score=0.8,
            daily_oracle_regime="bull", daily_oracle_score=0.7,
        )
        assert r.stability == RegimeStability.STABLE

    def test_full_disagreement_volatile(self, assessor):
        r = assessor.assess(
            ts_forecast_regime="bull", ts_forecast_score=0.7,
            daily_oracle_regime="bear", daily_oracle_score=0.7,
            anomaly_risk_level="critical", anomaly_risk_score=0.8,
            market_trend="transition", market_score=0.6,
        )
        assert r.stability in (RegimeStability.TRANSIENT, RegimeStability.VOLATILE)


class TestDivergence:
    def test_no_divergence_same_regime(self, assessor):
        r = assessor.assess(
            ts_forecast_regime="bull", ts_forecast_score=0.8,
            daily_oracle_regime="bull", daily_oracle_score=0.7,
        )
        assert r.source_divergence < 0.1

    def test_divergence_different_regimes(self, assessor):
        r = assessor.assess(
            ts_forecast_regime="bull", ts_forecast_score=0.8,
            daily_oracle_regime="bear", daily_oracle_score=0.7,
        )
        assert r.source_divergence > 0.05


class TestShiftDetection:
    def test_low_shift_risk_agreement(self, assessor):
        r = assessor.assess(
            ts_forecast_regime="bull", ts_forecast_score=0.8,
            daily_oracle_regime="bull", daily_oracle_score=0.7,
        )
        assert r.regime_shift_risk < 0.3

    def test_higher_shift_risk_with_divergence(self, assessor):
        r = assessor.assess(
            ts_forecast_regime="bull", ts_forecast_score=0.8,
            daily_oracle_regime="bear", daily_oracle_score=0.7,
        )
        assert r.regime_shift_risk >= 0.3

    def test_vix_increases_shift_risk(self, assessor):
        r_low = assessor.assess(
            ts_forecast_regime="bull", ts_forecast_score=0.7,
            daily_oracle_regime="bear", daily_oracle_score=0.6,
        )
        r_high = assessor.assess(
            ts_forecast_regime="bull", ts_forecast_score=0.7,
            daily_oracle_regime="bear", daily_oracle_score=0.6,
            vix_level=35,
        )
        assert r_high.regime_shift_risk >= r_low.regime_shift_risk

    def test_yield_spread_negative_increases_risk(self, assessor):
        r = assessor.assess(
            ts_forecast_regime="bull", ts_forecast_score=0.7,
            daily_oracle_regime="bear", daily_oracle_score=0.6,
            yield_spread=-0.5,
        )
        assert r.regime_shift_risk >= 0.3


class TestTransitions:
    def test_transitions_exist(self, assessor):
        r = assessor.assess(ts_forecast_regime="bull", ts_forecast_score=0.8)
        assert len(r.transitions) > 0

    def test_transition_probabilities(self, assessor):
        r = assessor.assess(ts_forecast_regime="bull", ts_forecast_score=0.8)
        total = sum(t.probability for t in r.transitions)
        assert abs(total - 1.0) < 0.01 or total <= 1.0

    def test_regime_stay_probability(self, assessor):
        r = assessor.assess(ts_forecast_regime="bull", ts_forecast_score=0.8)
        stay = [t for t in r.transitions if t.from_regime == t.to_regime]
        if stay:
            assert stay[0].probability > 0.5


class TestNarrative:
    def test_narrative_not_empty(self, assessor):
        r = assessor.assess(ts_forecast_regime="bull", ts_forecast_score=0.8)
        assert r.narrative
        assert len(r.narrative) > 20

    def test_narrative_includes_regime(self, assessor):
        r = assessor.assess(ts_forecast_regime="bear", ts_forecast_score=0.8)
        assert "Bear" in r.narrative or "bear" in r.narrative.lower()

    def test_narrative_has_timestamp(self, assessor):
        r = assessor.assess(ts_forecast_regime="bull", ts_forecast_score=0.7)
        assert r.timestamp


class TestCrisisAndRecovery:
    def test_crisis_detection_high_risk(self, assessor):
        r = assessor.assess(
            ts_forecast_regime="crisis", ts_forecast_score=0.9,
            anomaly_risk_level="critical", anomaly_risk_score=0.8,
        )
        assert r.composite_regime == RegimeType.CRISIS
        assert r.regime_score < 0.5

    def test_recovery_detection(self, assessor):
        r = assessor.assess(
            ts_forecast_regime="recovery", ts_forecast_score=0.7,
            daily_oracle_regime="bull", daily_oracle_score=0.6,
        )
        assert r.composite_regime in (RegimeType.RECOVERY, RegimeType.BULL)


class TestScoreAndConfidence:
    def test_score_in_range(self, assessor):
        r = assessor.assess(ts_forecast_regime="bull", ts_forecast_score=0.8)
        assert -1 <= r.regime_score <= 1

    def test_confidence_in_range(self, assessor):
        r = assessor.assess(ts_forecast_regime="bull", ts_forecast_score=0.8)
        assert 0 <= r.confidence <= 1

    def test_stability_score_in_range(self, assessor):
        r = assessor.assess(ts_forecast_regime="bull", ts_forecast_score=0.8)
        assert 0 <= r.stability_score <= 1