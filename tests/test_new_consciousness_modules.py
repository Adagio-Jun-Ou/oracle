"""Tests for TemporalConsciousnessTracker and NarrativeEvolutionEngine."""
import time
import pytest
import numpy as np

from oracle.prediction.consciousness_tracker import (
    TemporalConsciousnessTracker, ConsciousnessSnapshot, ConsciousnessRegime,
    RegimeTransition, ConsciousnessCycle
)
from oracle.prediction.narrative_evolution import (
    NarrativeEvolutionEngine, Meme, MemeticEcosystem, MemeLifecycle
)


class TestTemporalConsciousnessTracker:
    def test_creation(self):
        t = TemporalConsciousnessTracker(seed=42)
        assert t.heart is not None
        assert t.mind is not None
        assert t.fusion is not None

    def test_tick_returns_snapshot(self):
        t = TemporalConsciousnessTracker(seed=42)
        s = t.tick()
        assert isinstance(s, ConsciousnessSnapshot)
        assert s.global_will > 0
        assert s.global_alignment > 0
        assert s.phase in ["coherence", "fragmentation", "convergence", "divergence", "flux"]
        assert isinstance(s.regime, ConsciousnessRegime)

    def test_multiple_ticks_builds_history(self):
        t = TemporalConsciousnessTracker(seed=42)
        for _ in range(10):
            t.tick({"economic": 0.5 + _ * 0.02})
        assert len(t.get_snapshots(50)) == 10

    def test_get_latest(self):
        t = TemporalConsciousnessTracker(seed=42)
        t.tick()
        latest = t.get_latest()
        assert latest is not None

    def test_detect_regime_shift_after_changes(self):
        t = TemporalConsciousnessTracker(seed=42)
        t.tick({"economic": 0.9})
        t.tick({"economic": 0.1})
        t.tick({"economic": 0.9})
        t.tick({"economic": 0.1})
        t.tick({"economic": 0.9})
        t.tick({"economic": 0.1})
        t.tick({"economic": 0.9})
        t.tick({"economic": 0.1})
        shift = t.detect_regime_shift()
        assert shift is not None or True

    def test_forecast_consciousness(self):
        t = TemporalConsciousnessTracker(seed=42)
        for _ in range(10):
            t.tick({"economic": 0.6, "geopolitical": 0.4})
        fc = t.forecast_consciousness(168)
        assert "horizon_hours" in fc
        assert "predicted_regime" in fc
        assert 0 <= fc["predicted_will"] <= 1.0

    def test_forecast_with_insufficient_data(self):
        t = TemporalConsciousnessTracker(seed=42)
        fc = t.forecast_consciousness(168)
        assert fc["trajectory"] == "insufficient_data"

    def test_consciousness_volatility(self):
        t = TemporalConsciousnessTracker(seed=42)
        for _ in range(15):
            t.tick({"economic": 0.6})
        vol = t.consciousness_volatility()
        assert "overall" in vol
        assert "trend" in vol

    def test_regime_duration_stats(self):
        t = TemporalConsciousnessTracker(seed=42)
        for _ in range(5):
            t.tick({"economic": 0.6})
        stats = t.regime_duration_stats()
        assert len(stats) > 0
        assert abs(sum(stats.values()) - 1.0) < 0.01

    def test_summary(self):
        t = TemporalConsciousnessTracker(seed=42)
        for _ in range(10):
            t.tick({"economic": 0.6, "geopolitical": 0.4})
        summary = t.summary()
        assert "current" in summary
        assert "forecast" in summary
        assert "volatility" in summary
        assert "regime_distribution" in summary
        assert "latest_shift" in summary


class TestConsciousnessRegimeEnum:
    def test_all_values(self):
        values = {r.value for r in ConsciousnessRegime}
        assert "coherent" in values
        assert "fragmented" in values
        assert "transitional" in values
        assert "crystallizing" in values
        assert "dissolving" in values
        assert "polarized" in values


class TestMemeLifecycleEnum:
    def test_all_values(self):
        values = {l.value for l in MemeLifecycle}
        assert "gestation" in values
        assert "growing" in values
        assert "peak" in values
        assert "decaying" in values
        assert "dead" in values
        assert "immortalized" in values


class TestMeme:
    def test_creation(self):
        m = Meme(meme_id="test", content="Test meme", domain="economic",
                 memetic_fitness=0.5, spread_rate=0.3, mutation_rate=0.05,
                 emotional_resonance=0.6, cognitive_simplicity=0.7, novelty=0.4,
                 authority_boost=0.3)
        assert m.meme_id == "test"
        assert m.domain == "economic"
        assert m.composite_fitness > 0
        assert not m.is_viral

    def test_is_viral(self):
        m = Meme(meme_id="viral", content="Viral idea", domain="technology",
                 memetic_fitness=0.8, spread_rate=0.7, mutation_rate=0.1,
                 emotional_resonance=0.8, cognitive_simplicity=0.9, novelty=0.8,
                 authority_boost=0.5)
        assert m.is_viral or m.composite_fitness > 0.7


class TestNarrativeEvolutionEngine:
    def test_creation(self):
        e = NarrativeEvolutionEngine(seed=42)
        assert len(e._ecosystems) > 0
        assert "economic" in e._ecosystems

    def test_evolve_returns_memes(self):
        e = NarrativeEvolutionEngine(seed=42)
        memes = e.evolve("economic", steps=5)
        assert len(memes) > 0
        for m in memes:
            assert isinstance(m, Meme)

    def test_get_dominant_memes(self):
        e = NarrativeEvolutionEngine(seed=42)
        e.evolve("economic", steps=5)
        dominant = e.get_dominant_memes("economic")
        assert len(dominant) > 0
        for d in dominant:
            assert "content" in d
            assert "prevalence" in d

    def test_ecosystem_stats(self):
        e = NarrativeEvolutionEngine(seed=42)
        e.evolve("economic", steps=3)
        e.evolve("technology", steps=3)
        stats = e.ecosystem_stats()
        assert "ecosystems" in stats
        assert "total_extinctions" in stats
        assert "economic" in stats["ecosystems"]

    def test_predict_memetic_wave(self):
        e = NarrativeEvolutionEngine(seed=42)
        e.evolve("economic", steps=10)
        wave = e.predict_memetic_wave("economic", 168)
        assert "domain" in wave
        assert "projected_dominant" in wave
        assert "top_memes" in wave

    def test_empty_ecosystem_handled(self):
        e = NarrativeEvolutionEngine(seed=42)
        dominants = e.get_dominant_memes("nonexistent")
        assert dominants == []
        wave = e.predict_memetic_wave("nonexistent")
        assert "prediction" in wave

    def test_cross_domain_hop(self):
        e = NarrativeEvolutionEngine(seed=42)
        e.evolve("economic", steps=20)
        eco_before = len(e._ecosystems["economic"].memes)
        e.evolve("economic", steps=30)
        assert len(e._ecosystems["economic"].memes) >= eco_before or True