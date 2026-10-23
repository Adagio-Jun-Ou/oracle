"""Comprehensive tests for the collective consciousness modules.

Tests cover:
- EmotionalPulseEngine: pulse measurement, contagion, sentiment maps, emotional shifts
- CognitiveLandscape: belief topology, narratives, attention, expectations, cascades
- ConsciousnessFusion: heart-mind fusion, alignment, tipping points, phases, dissonance
- HiveMindOracle: predictions, wisdom assessment, latent consensus, prediction markets
- CollectiveConsciousnessEngine: full multi-layer read, dashboard, scanning
"""
import math
import time
import pytest
import numpy as np

from oracle.prediction.heart_pulse import (
    EmotionalPulseEngine, EmotionalPulse, EmotionalState, EmotionalAxis,
    EmotionalContagion, CONTAGION_NETWORK, DOMAIN_EMOTION_PROFILES, DOMAIN_EMOTION_BASELINES
)
from oracle.prediction.mind_landscape import (
    CognitiveLandscape, BeliefNode, Narrative, AttentionVector, ExpectationCurve
)
from oracle.prediction.consciousness_fusion import (
    ConsciousnessFusion, FusionSignal, ResonanceMode, ConsciousCluster
)
from oracle.prediction.hive_oracle import (
    HiveMindOracle, HivePrediction, CrowdWisdom, LatentConsensus
)
from oracle.prediction.collective_engine import (
    CollectiveConsciousnessEngine, create_consciousness_engine, quick_pulse
)


class TestEmotionalPulse:
    def test_creation(self):
        p = EmotionalPulse(axis="fear_greed", intensity=0.5, polarity=-0.3,
                           coherence=0.6, velocity=0.01, acceleration=0.0, entropy=0.3)
        assert p.axis == "fear_greed"
        assert p.intensity == 0.5
        assert p.polarity == -0.3
        assert p.net_valence < 0

    def test_is_escalating(self):
        p = EmotionalPulse(axis="anger_calm", intensity=0.5, polarity=0.6,
                           coherence=0.5, velocity=0.05, acceleration=0.0, entropy=0.2)
        assert p.is_escalating

    def test_not_escalating_low_intensity(self):
        p = EmotionalPulse(axis="anger_calm", intensity=0.3, polarity=0.6,
                           coherence=0.5, velocity=0.05, acceleration=0.0, entropy=0.2)
        assert not p.is_escalating

    def test_deescalating(self):
        p = EmotionalPulse(axis="hope_despair", intensity=0.6, polarity=0.3,
                           coherence=0.5, velocity=-0.02, acceleration=0.0, entropy=0.2)
        assert p.is_deescalating


class TestEmotionalPulseEngine:
    def test_creation(self):
        e = EmotionalPulseEngine(seed=42)
        assert len(e._states) > 0
        assert "economic" in e._states

    def test_pulse_returns_state(self):
        e = EmotionalPulseEngine(seed=42)
        state = e.pulse("economic")
        assert state.domain == "economic"
        assert len(state.pulses) > 0
        assert 0 <= state.composite_mood <= 1.0
        assert 0 <= state.arousal <= 1.0

    def test_pulse_updates_over_time(self):
        e = EmotionalPulseEngine(seed=42)
        e._states["economic"].timestamp = time.time() - 3600
        s1 = e.pulse("economic", {"economic": 0.3})
        s2 = e.pulse("economic", {"economic": 0.7})
        assert s1.domain == "economic"
        assert s2.domain == "economic"
        assert len(s1.pulses) == len(s2.pulses)

    def test_sentiment_map(self):
        e = EmotionalPulseEngine(seed=42)
        e.pulse("economic")
        e.pulse("geopolitical")
        result = e.sentiment_map()
        assert "domains" in result
        assert "global" in result
        assert "economic" in result["domains"]
        assert "geopolitical" in result["domains"]
        assert "mood" in result["global"]

    def test_global_mood(self):
        e = EmotionalPulseEngine(seed=42)
        mood = e.get_global_mood()
        assert "mood" in mood
        assert "arousal" in mood
        assert 0 <= mood["mood"] <= 1.0

    def test_predict_emotional_shift(self):
        e = EmotionalPulseEngine(seed=42)
        e.pulse("economic", {"economic": 0.4})
        result = e.predict_emotional_shift("economic", horizon_hours=24)
        assert "predictions" in result
        assert "largest_shift_axis" in result
        assert "mood_trajectory" in result

    def test_heart_of_crowd(self):
        e = EmotionalPulseEngine(seed=42)
        result = e.heart_of_crowd()
        assert "narrative" in result
        assert "dominant_emotions" in result

    def test_contagion_network_has_entries(self):
        assert len(CONTAGION_NETWORK) > 0
        for c in CONTAGION_NETWORK:
            assert 0 <= c.coupling_strength <= 1.0
            assert c.source_domain
            assert c.target_domain

    def test_domain_profiles_have_all_axes(self):
        axes = {e.value for e in EmotionalAxis}
        for domain, profile in DOMAIN_EMOTION_PROFILES.items():
            for axis in axes:
                assert axis in profile, f"{axis} missing from {domain}"

    def test_pulse_with_custom_states(self):
        e = EmotionalPulseEngine(seed=42)
        s = e.pulse("economic", {"economic": 0.8, "confidence": 0.7})
        assert s.domain == "economic"


class TestBeliefNode:
    def test_creation(self):
        b = BeliefNode("test", "Test belief", 0.5, 0.6, 0.7, 0.4, 0.5, domain="economic")
        assert b.belief_id == "test"
        assert b.prevalence == 0.5
        assert b.centrality == 0.7


class TestAttentionVector:
    def test_creation(self):
        a = AttentionVector("economic", 0.3, 2.0, 0.5, 0.4, 0.3, 0.2)
        assert a.domain == "economic"
        assert a.effective_attention > 0

    def test_effective_attention(self):
        a = AttentionVector("economic", 0.5, 3.0, 0.8, 0.6, 0.4, 0.1)
        assert a.effective_attention == 0.5 * 0.8


class TestExpectationCurve:
    def test_creation(self):
        e = ExpectationCurve("economic", "SP500", 5000, 5100, 5300, 5600, 0.15, 0.08)
        assert e.domain == "economic"
        assert e.current_value == 5000
        assert len(e.term_structure) == 3

    def test_is_inverted(self):
        e = ExpectationCurve("economic", "SP500", 5600, 5500, 5400, 5300, 0.15, 0.08)
        assert e.is_inverted

    def test_not_inverted(self):
        e = ExpectationCurve("economic", "SP500", 5000, 5100, 5300, 5600, 0.15, 0.08)
        assert not e.is_inverted


class TestCognitiveLandscape:
    def test_creation(self):
        c = CognitiveLandscape(seed=42)
        assert len(c._beliefs) > 0
        assert len(c._narratives) > 0
        assert len(c._attention) > 0
        assert len(c._expectations) > 0

    def test_update_belief(self):
        c = CognitiveLandscape(seed=42)
        orig = c._beliefs["b1"].prevalence
        c.update_belief("b1", {"monetary": 0.7})
        assert c._beliefs["b1"].prevalence != orig

    def test_update_all_beliefs(self):
        c = CognitiveLandscape(seed=42)
        c.update_all_beliefs({"monetary": 0.8, "economic": 0.6})

    def test_belief_topology(self):
        c = CognitiveLandscape(seed=42)
        topo = c.belief_topology()
        assert "nodes" in topo
        assert "consensus" in topo
        assert "most_central" in topo
        assert len(topo["nodes"]) == len(c._beliefs)

    def test_narrative_landscape(self):
        c = CognitiveLandscape(seed=42)
        nl = c.narrative_landscape()
        assert "narratives" in nl
        assert "dominant_by_domain" in nl
        assert "most_dominant" in nl

    def test_attention_map(self):
        c = CognitiveLandscape(seed=42)
        am = c.attention_map()
        assert "vectors" in am
        assert "normalized" in am
        assert "attention_concentration" in am

    def test_expectation_map(self):
        c = CognitiveLandscape(seed=42)
        em = c.expectation_map()
        assert "curves" in em
        assert "inversion_count" in em
        assert "term_premium" in em

    def test_cascade_signals(self):
        c = CognitiveLandscape(seed=42)
        cs = c.information_cascade_signals()
        assert "signals" in cs
        assert "count" in cs

    def test_cognitive_dissonance(self):
        c = CognitiveLandscape(seed=42)
        cd = c.cognitive_dissonance_map()
        assert "dissonances" in cd
        assert "count" in cd

    def test_collective_belief_state(self):
        c = CognitiveLandscape(seed=42)
        s = c.collective_belief_state()
        assert isinstance(s, dict)
        for k, v in s.items():
            assert 0 <= v <= 1.0

    def test_herfindahl(self):
        c = CognitiveLandscape(seed=42)
        h = c._herfindahl([0.25, 0.25, 0.25, 0.25])
        assert abs(h - 0.25) < 0.01
        h = c._herfindahl([1.0])
        assert abs(h - 1.0) < 0.01


class TestFusionSignal:
    def test_creation(self):
        f = FusionSignal(domain="economic", heart_mood=0.6, mind_belief=0.5,
                        alignment=0.9, resonance_mode=ResonanceMode.ALIGNED,
                        collective_will=0.8, tipping_risk=0.3)
        assert f.domain == "economic"
        assert not f.is_at_tipping_point

    def test_at_tipping_point(self):
        f = FusionSignal(domain="economic", heart_mood=0.6, mind_belief=0.5,
                        alignment=0.9, resonance_mode=ResonanceMode.CRYSTALLIZING,
                        collective_will=0.8, tipping_risk=0.8)
        assert f.is_at_tipping_point
        assert f.will_strength == "dominant"

    def test_will_strength_weak(self):
        f = FusionSignal(domain="economic", heart_mood=0.5, mind_belief=0.5,
                        alignment=0.5, resonance_mode=ResonanceMode.CHAOTIC,
                        collective_will=0.2, tipping_risk=0.1)
        assert f.will_strength == "weak"


class TestConsciousnessFusion:
    def test_creation(self):
        f = ConsciousnessFusion(seed=42)
        assert f.heart is not None
        assert f.mind is not None

    def test_fuse(self):
        f = ConsciousnessFusion(seed=42)
        result = f.fuse({"economic": 0.6, "geopolitical": 0.4})
        assert "signals" in result
        assert "consciousness_map" in result
        assert "global_will" in result
        assert "phase" in result
        assert "collective_insight" in result
        assert 0 <= result["global_will"] <= 1.0

    def test_fuse_no_states(self):
        f = ConsciousnessFusion(seed=42)
        result = f.fuse()
        assert len(result["signals"]) > 0

    def test_detect_dissonance_events(self):
        f = ConsciousnessFusion(seed=42)
        for _ in range(5):
            f.fuse({"economic": 0.6})
        events = f.detect_dissonance_events()
        assert isinstance(events, list)

    def test_predict_consciousness_shift(self):
        f = ConsciousnessFusion(seed=42)
        for _ in range(5):
            f.fuse({"economic": 0.6, "geopolitical": 0.4})
        pred = f.predict_consciousness_shift(48)
        assert "predictions" in pred
        assert "horizon_hours" in pred
        assert "global_will" in pred

    def test_resonance_modes(self):
        f = ConsciousnessFusion(seed=42)
        mode = f._classify_resonance(0.8, 0.5, 0.5, 0.3)
        assert mode == ResonanceMode.ALIGNED
        mode = f._classify_resonance(0.1, 0.5, 0.5, 0.2)
        assert mode == ResonanceMode.DISSONANT
        mode = f._classify_resonance(0.5, 0.5, 0.5, 0.1)
        assert mode in [ResonanceMode.DISSOLVING, ResonanceMode.CHAOTIC]


class TestHiveMindOracle:
    def test_creation(self):
        h = HiveMindOracle(seed=42)
        assert h.fusion is not None
        assert h.heart is not None
        assert h.mind is not None

    def test_predict(self):
        h = HiveMindOracle(seed=42)
        result = h.predict({"economic": 0.6}, horizon_hours=168)
        assert "predictions" in result
        assert "global_wisdom" in result
        assert "latent_consensus" in result
        assert "prediction_market" in result
        assert "global_synthesis" in result

    def test_predict_no_states(self):
        h = HiveMindOracle(seed=42)
        result = h.predict()
        assert len(result["predictions"]) > 0

    def test_latent_prediction_scan(self):
        h = HiveMindOracle(seed=42)
        result = h.latent_prediction_scan()
        assert len(result) > 0
        for item in result:
            assert "domain" in item
            assert "topic" in item
            assert "implied_prob" in item

    def test_prediction_to_dict(self):
        h = HiveMindOracle(seed=42)
        p = HivePrediction(
            domain="economic", event="Market crash", probability=0.3,
            crowd_confidence=0.6, crowd_dispersion=0.4,
            wisdom_quality=CrowdWisdom.WISE,
            latent_knowledge_signal=0.2, self_fulfilling_risk=0.1,
            horizon_hours=72
        )
        d = h._pred_to_dict(p)
        assert d["domain"] == "economic"
        assert d["is_wise"]

    def test_wisdom_assessment(self):
        h = HiveMindOracle(seed=42)
        preds = {
            "eco": HivePrediction("eco", "A", 0.5, 0.6, 0.4, CrowdWisdom.WISE, 0.1, 0.1),
            "geo": HivePrediction("geo", "B", 0.3, 0.4, 0.6, CrowdWisdom.WISE, 0.1, 0.1),
        }
        signals = {"eco": {"alignment": 0.8}, "geo": {"alignment": 0.7}}
        wisdom = h._assess_crowd_wisdom(preds, signals)
        assert wisdom["dominant_quality"] == "wise"


class TestHivePrediction:
    def test_is_wise(self):
        p = HivePrediction("eco", "event", 0.5, 0.6, 0.3, CrowdWisdom.WISE, 0.1, 0.1)
        assert p.is_wise_crowd

    def test_not_wise(self):
        p = HivePrediction("eco", "event", 0.5, 0.6, 0.3, CrowdWisdom.MADNESS, 0.1, 0.1)
        assert not p.is_wise_crowd

    def test_is_self_fulfilling(self):
        p = HivePrediction("eco", "event", 0.9, 0.8, 0.1, CrowdWisdom.WISE, 0.1, 0.7)
        assert p.is_self_fulfilling

    def test_not_self_fulfilling(self):
        p = HivePrediction("eco", "event", 0.3, 0.3, 0.7, CrowdWisdom.DISPERSED, 0.1, 0.1)
        assert not p.is_self_fulfilling


class TestCollectiveConsciousnessEngine:
    def test_creation(self):
        e = CollectiveConsciousnessEngine(seed=42)
        assert e.heart is not None
        assert e.mind is not None
        assert e.fusion is not None
        assert e.hive is not None

    def test_read(self):
        e = CollectiveConsciousnessEngine(seed=42)
        result = e.read()
        assert "heart" in result
        assert "mind" in result
        assert "fusion" in result
        assert "hive" in result
        assert "read_at" in result

    def test_read_with_states(self):
        e = CollectiveConsciousnessEngine(seed=42)
        result = e.read({"economic": 0.7, "geopolitical": 0.3, "monetary": 0.5})
        assert result["heart"]["heart_narrative"]

    def test_predict_hive(self):
        e = CollectiveConsciousnessEngine(seed=42)
        pred = e.predict_hive("economic")
        assert "domain" in pred
        assert "probability" in pred

    def test_emotional_forecast(self):
        e = CollectiveConsciousnessEngine(seed=42)
        fc = e.emotional_forecast("economic", 24)
        assert "predictions" in fc
        assert "largest_shift_axis" in fc

    def test_collective_belief_state(self):
        e = CollectiveConsciousnessEngine(seed=42)
        state = e.collective_belief_state()
        assert isinstance(state, dict)
        assert len(state) > 0

    def test_consciousness_phase(self):
        e = CollectiveConsciousnessEngine(seed=42)
        phase = e.consciousness_phase()
        assert "phase" in phase
        assert "will" in phase

    def test_surface_latent(self):
        e = CollectiveConsciousnessEngine(seed=42)
        latent = e.surface_latent()
        assert len(latent) > 0

    def test_scan_for_tipping_points(self):
        e = CollectiveConsciousnessEngine(seed=42)
        tipping = e.scan_for_tipping_points()
        assert "tipping_domains" in tipping
        assert "phase" in tipping

    def test_collective_dashboard(self):
        e = CollectiveConsciousnessEngine(seed=42)
        dash = e.collective_dashboard()
        assert "consciousness_phase" in dash
        assert "collective_will_index" in dash
        assert "crowd_wisdom_quality" in dash
        assert "oracle_synthesis" in dash

    def test_full_read_all_keys(self):
        e = CollectiveConsciousnessEngine(seed=42)
        result = e.read({"economic": 0.6})
        assert "heart_narrative" in result["heart"]
        assert "belief_topology" in result["mind"]
        assert "phase" in result["fusion"]
        assert "global_wisdom" in result["hive"]

    def test_multiple_reads_consistent(self):
        e = CollectiveConsciousnessEngine(seed=42)
        r1 = e.read()
        r2 = e.read()
        assert abs(r1["fusion"]["global_will"] - r2["fusion"]["global_will"]) < 0.3

    def test_create_consciousness_engine(self):
        e = create_consciousness_engine(seed=42)
        assert e is not None
        assert isinstance(e, CollectiveConsciousnessEngine)

    def test_quick_pulse(self):
        result = quick_pulse()
        assert "heart" in result
        assert "mind" in result

    def test_quick_pulse_with_existing(self):
        e = CollectiveConsciousnessEngine(seed=42)
        result = quick_pulse(engine=e)
        assert result["heart"]["global_mood"]["mood"] > 0

    def test_read_none_states(self):
        e = CollectiveConsciousnessEngine(seed=42)
        result = e.read(None)
        assert result is not None

    def test_fusion_in_read(self):
        e = CollectiveConsciousnessEngine(seed=42)
        result = e.read()
        fusion = result["fusion"]
        assert "resonant_domains" in fusion
        assert "dissonant_domains" in fusion
        assert "has_tipping_point" in fusion
        assert "dissonance_events" in fusion

    def test_hive_in_read(self):
        e = CollectiveConsciousnessEngine(seed=42)
        result = e.read()
        hive = result["hive"]
        assert "predictions" in hive
        assert "latent_consensus" in hive
        assert "prediction_market" in hive
        assert "oracle_synthesis" in hive

    def test_mind_in_read(self):
        e = CollectiveConsciousnessEngine(seed=42)
        result = e.read()
        mind = result["mind"]
        assert "narrative_landscape" in mind
        assert "attention" in mind
        assert "expectations" in mind
        assert "cascade_signals" in mind
        assert "cognitive_dissonance" in mind

    def test_heart_axes_exist(self):
        e = CollectiveConsciousnessEngine(seed=42)
        domain_state = e.heart.pulse("economic")
        for axis in [ea.value for ea in EmotionalAxis]:
            assert axis in domain_state.pulses, f"Missing axis {axis} in economic state"


class TestEmotionalAxisEnum:
    def test_all_values_valid(self):
        axes = [ea.value for ea in EmotionalAxis]
        assert "fear_greed" in axes
        assert "hope_despair" in axes
        assert "trust_distrust" in axes
        assert "anger_calm" in axes
        assert "love_hate" in axes
        assert "surprise_anticipation" in axes


class TestResonanceModeEnum:
    def test_all_values(self):
        modes = {m.value for m in ResonanceMode}
        assert "aligned" in modes
        assert "dissonant" in modes
        assert "emerging" in modes
        assert "chaotic" in modes


class TestCrowdWisdomEnum:
    def test_all_values(self):
        values = {w.value for w in CrowdWisdom}
        assert "wise" in values
        assert "madness" in values
        assert "herding" in values
        assert "dispersed" in values
        assert "converging" in values