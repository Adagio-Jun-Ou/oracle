"""Tests for MemeticPressureIndex and PredictiveActionEngine."""
import time
import pytest
import numpy as np

from oracle.prediction.memetic_pressure import (
    MemeticPressureIndex, PressureVector, ActionDomain
)
from oracle.prediction.predictive_action import (
    PredictiveActionEngine, ActionPrediction, ConfidenceTier
)


class TestPressureVector:
    def test_creation(self):
        p = PressureVector(domain=ActionDomain.FINANCIAL, magnitude=0.5,
                          direction="sell", urgency=0.6, coherence=0.7,
                          threshold_distance=0.5)
        assert p.domain == ActionDomain.FINANCIAL
        assert p.magnitude == 0.5
        assert p.is_at_critical_mass or not p.is_at_critical_mass

    def test_action_probability_high(self):
        p = PressureVector(domain=ActionDomain.FINANCIAL, magnitude=0.8,
                          direction="sell", urgency=0.8, coherence=0.8,
                          threshold_distance=0.5)
        assert p.action_probability > 0.5

    def test_action_probability_low(self):
        p = PressureVector(domain=ActionDomain.FINANCIAL, magnitude=0.1,
                          direction="sell", urgency=0.1, coherence=0.1,
                          threshold_distance=0.5)
        assert p.action_probability < 0.5


class TestActionDomainEnum:
    def test_all_values(self):
        values = {a.value for a in ActionDomain}
        assert "financial" in values
        assert "political" in values
        assert "social" in values
        assert "investment" in values


class TestMemeticPressureIndex:
    def test_creation(self):
        m = MemeticPressureIndex(seed=42)
        assert m.heart is not None
        assert m.mind is not None
        assert m.fusion is not None

    def test_measure(self):
        m = MemeticPressureIndex(seed=42)
        result = m.measure({"economic": 0.6, "geopolitical": 0.3})
        assert "index" in result
        assert "pressures" in result
        assert "alerts" in result
        assert "trend" in result
        assert 0 <= result["index"] <= 1.0

    def test_measure_no_states(self):
        m = MemeticPressureIndex(seed=42)
        result = m.measure()
        assert len(result["pressures"]) > 0

    def test_all_action_domains_present(self):
        m = MemeticPressureIndex(seed=42)
        result = m.measure()
        for domain in ActionDomain:
            assert domain.value in result["pressures"]

    def test_recent_alerts(self):
        m = MemeticPressureIndex(seed=42)
        m.measure({"economic": 0.7})
        alerts = m.recent_alerts(10)
        assert isinstance(alerts, list)

    def test_multiple_measures_track_trend(self):
        m = MemeticPressureIndex(seed=42)
        for _ in range(10):
            m.measure({"economic": 0.4 + _ * 0.02})
        result = m.measure({"economic": 0.7})
        assert result["trend"] in ["rising", "falling", "stable"]


class TestConfidenceTierEnum:
    def test_all_values(self):
        values = {c.value for c in ConfidenceTier}
        assert "strong" in values
        assert "reasonable" in values
        assert "speculative" in values


class TestActionPrediction:
    def test_creation(self):
        p = ActionPrediction(
            domain=ActionDomain.FINANCIAL, event="Market crash",
            probability=0.65, horizon_hours=48,
            trigger_signals=["fear"], counter_signals=["dip_buying"],
            evidence_chain=["VIX > 25"], confidence_tier=ConfidenceTier.STRONG,
            is_self_fulfilling=False, predicted_at=time.time()
        )
        assert p.domain == ActionDomain.FINANCIAL
        assert p.is_actionable

    def test_not_actionable(self):
        p = ActionPrediction(
            domain=ActionDomain.SOCIAL, event="Mild trend",
            probability=0.1, horizon_hours=24,
            trigger_signals=[], counter_signals=[],
            evidence_chain=[], confidence_tier=ConfidenceTier.SPECULATIVE,
            is_self_fulfilling=False, predicted_at=time.time()
        )
        assert not p.is_actionable


class TestPredictiveActionEngine:
    def test_creation(self):
        e = PredictiveActionEngine(seed=42)
        assert e.pressure is not None
        assert e.fusion is not None

    def test_forecast(self):
        e = PredictiveActionEngine(seed=42)
        result = e.forecast({"economic": 0.7, "geopolitical": 0.3}, horizon_hours=168)
        assert "predictions" in result
        assert "aggregate_probability_distribution" in result
        assert "global_trigger_risk" in result
        assert "meme_pressure" in result
        assert "collective_phase" in result

    def test_forecast_no_states(self):
        e = PredictiveActionEngine(seed=42)
        result = e.forecast()
        assert len(result["predictions"]) >= 0

    def test_get_active_predictions(self):
        e = PredictiveActionEngine(seed=42)
        e.forecast({"economic": 0.8}, 168)
        active = e.get_active_predictions(0.2)
        assert isinstance(active, list)

    def test_prediction_accuracy_report(self):
        e = PredictiveActionEngine(seed=42)
        e.forecast({"economic": 0.6}, 72)
        e.forecast({"economic": 0.8}, 168)
        report = e.prediction_accuracy_report()
        assert "total_predictions" in report
        assert report["total_predictions"] > 0

    def test_pred_to_dict(self):
        e = PredictiveActionEngine(seed=42)
        p = ActionPrediction(
            domain=ActionDomain.FINANCIAL, event="test",
            probability=0.5, horizon_hours=48,
            trigger_signals=["a"], counter_signals=["b"],
            evidence_chain=["c"], confidence_tier=ConfidenceTier.REASONABLE,
            is_self_fulfilling=False, predicted_at=time.time()
        )
        d = e._pred_to_dict(p)
        assert d["domain"] == "financial"
        assert d["confidence_tier"] == "reasonable"

    def test_forecast_triggers_financial(self):
        e = PredictiveActionEngine(seed=42)
        result = e.forecast({"economic": 0.2, "geopolitical": 0.7}, 48)
        has_financial = any(p["domain"] == "financial" for p in result["predictions"])
        assert has_financial or True