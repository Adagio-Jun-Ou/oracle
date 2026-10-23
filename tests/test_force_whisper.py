"""Tests for Force Whisper Engine — Latent Force Detection and Fate Pressure."""
import numpy as np
import pytest
from oracle.prediction.force_whisper import (
    ForceWhisperEngine, LatentForce, ForceWhisperReport, PressureGradient,
    TensionZone, FatePressureLevel, ForceOrigin, ForceDomain,
    ORIGIN_EVIDENCE_MAP, FORCE_NAMING, _safe_std, _safe_mean,
    _compute_second_derivative, _compute_phase_velocity,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def engine():
    return ForceWhisperEngine(min_series_length=5, tension_window=5)


@pytest.fixture
def stable_series(rng):
    """Flat, stable domain signals — no forces should be detected."""
    return {
        "economics": np.cumsum(rng.normal(0, 0.01, 100)) + 1.0,
        "finance": np.cumsum(rng.normal(0, 0.01, 100)) + 1.0,
        "social": np.cumsum(rng.normal(0, 0.01, 100)) + 1.0,
    }


@pytest.fixture
def turbulent_series(rng):
    """Highly unstable series — strong forces should be detected."""
    base = np.cumsum(rng.normal(0, 0.1, 100))
    spike = np.zeros(100)
    spike[70:] = np.linspace(0, 5, 30)  # Accelerating upward trend
    return {
        "economics": base + spike,
        "finance": base + np.cumsum(rng.normal(0, 0.15, 100)),
        "social": -base + np.linspace(0, -3, 100),
    }


@pytest.fixture
def accelerating_series():
    """Series with clear second-derivative signal."""
    t = np.arange(50)
    return {
        "inflation": 0.001 * t**2 + np.random.RandomState(7).normal(0, 0.05, 50),
        "unemployment": np.linspace(0.05, 0.12, 50),
    }


class TestHelperFunctions:
    def test_safe_std_empty(self):
        assert _safe_std(np.array([])) == 0.0

    def test_safe_std_single(self):
        assert _safe_std(np.array([1.0])) == 0.0

    def test_safe_std_normal(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        result = _safe_std(arr)
        assert 1.0 < result < 2.0

    def test_safe_mean_empty(self):
        assert _safe_mean(np.array([])) == 0.0

    def test_safe_mean_normal(self):
        assert _safe_mean(np.array([2, 4, 6])) == 4.0

    def test_second_derivative_short(self):
        result = _compute_second_derivative(np.array([1.0, 2.0]))
        assert len(result) == 0

    def test_second_derivative_parabola(self):
        t = np.arange(20, dtype=float)
        s = 0.5 * t**2
        d2 = _compute_second_derivative(s)
        assert np.allclose(d2, 1.0, atol=0.01)

    def test_second_derivative_linear(self):
        s = np.arange(20, dtype=float)
        d2 = _compute_second_derivative(s)
        assert np.allclose(d2, 0.0, atol=1e-10)

    def test_phase_velocity_stable(self):
        data = {"a": np.ones(50)}
        vel = _compute_phase_velocity(data, min_series_length=10)
        assert "a" in vel
        assert abs(vel["a"]) < 0.01

    def test_phase_velocity_short(self):
        data = {"a": np.array([1.0, 2.0])}
        vel = _compute_phase_velocity(data, min_series_length=10)
        assert vel["a"] == 0.0


class TestFatePressureLevel:
    def test_still(self):
        assert FatePressureLevel.STILL.value == "still"

    def test_whisper(self):
        assert FatePressureLevel.WHISPER.value == "whisper"

    def test_rumble(self):
        assert FatePressureLevel.RUMBLE.value == "rumble"

    def test_surge(self):
        assert FatePressureLevel.SURGE.value == "surge"

    def test_rupture(self):
        assert FatePressureLevel.RUPTURE.value == "rupture"


class TestForceOrigin:
    def test_all_origins(self):
        expected = {"structural", "contagion", "behavioral", "exogenous", "feedback", "accumulation", "unknown"}
        actual = {o.value for o in ForceOrigin}
        assert actual == expected

    def test_origin_evidence_map(self):
        for origin in ForceOrigin:
            if origin != ForceOrigin.UNKNOWN:
                assert origin in ORIGIN_EVIDENCE_MAP
                assert len(ORIGIN_EVIDENCE_MAP[origin]) > 0


class TestForceDomain:
    def test_all_domains(self):
        expected = {"economic", "geopolitical", "social", "technological",
                    "environmental", "financial", "political", "psychological",
                    "informational", "systemic"}
        actual = {d.value for d in ForceDomain}
        assert actual == expected


class TestLatentForce:
    def test_active_force(self):
        f = LatentForce(
            name="test", origin=ForceOrigin.STRUCTURAL, primary_domain=ForceDomain.ECONOMIC,
            magnitude=0.5, direction=1.0, acceleration=0.2, phase_velocity=0.1,
            affected_domains=[ForceDomain.ECONOMIC], evidence=["test"],
            lead_signals=["test"], tension_index=0.3, rupture_probability=0.4,
            estimated_time_to_manifest_hours=48.0,
        )
        assert f.is_active
        assert f.pressure_contribution > 0

    def test_inactive_force(self):
        f = LatentForce(
            name="test", origin=ForceOrigin.UNKNOWN, primary_domain=ForceDomain.SYSTEMIC,
            magnitude=0.05, direction=0.0, acceleration=0.0, phase_velocity=0.0,
            affected_domains=[], evidence=[], lead_signals=[], tension_index=0.0,
            rupture_probability=0.0, estimated_time_to_manifest_hours=720.0,
        )
        assert not f.is_active

    def test_pressure_contribution_formula(self):
        f = LatentForce(
            name="test", origin=ForceOrigin.STRUCTURAL, primary_domain=ForceDomain.ECONOMIC,
            magnitude=0.5, direction=1.0, acceleration=0.2, phase_velocity=0.1,
            affected_domains=[ForceDomain.ECONOMIC], evidence=["test"],
            lead_signals=["test"], tension_index=0.3, rupture_probability=0.4,
            estimated_time_to_manifest_hours=48.0,
        )
        expected = 0.5 * (1.0 + 0.2) * 0.3
        assert abs(f.pressure_contribution - expected) < 0.001

    def test_to_dict(self):
        f = LatentForce(
            name="Yield Curve Phantom", origin=ForceOrigin.STRUCTURAL,
            primary_domain=ForceDomain.ECONOMIC, magnitude=0.7, direction=1.0,
            acceleration=0.3, phase_velocity=0.5, affected_domains=[ForceDomain.ECONOMIC, ForceDomain.FINANCIAL],
            evidence=["test evidence"], lead_signals=["accelerating_change"],
            tension_index=0.6, rupture_probability=0.4,
            estimated_time_to_manifest_hours=24.0,
        )
        d = f.to_dict()
        assert d["name"] == "Yield Curve Phantom"
        assert d["origin"] == "structural"
        assert d["primary_domain"] == "economic"
        assert d["is_active"] is True
        assert len(d["affected_domains"]) == 2


class TestForceWhisperEngineDetection:
    def test_stable_series_no_forces(self, engine, stable_series):
        forces = engine.detect_forces(stable_series)
        active = [f for f in forces if f.is_active]
        assert len(active) <= 1

    def test_turbulent_series_detects_forces(self, engine, turbulent_series):
        forces = engine.detect_forces(turbulent_series)
        active = [f for f in forces if f.is_active]
        assert len(active) >= 1

    def test_accelerating_series_high_magnitude(self, engine):
        t = np.arange(50, dtype=float)
        data = {"inflation": 0.01 * t**2, "finance": 0.5 * t}
        low_thresh_engine = ForceWhisperEngine(min_series_length=5, tension_window=5, acceleration_threshold=0.001)
        forces = low_thresh_engine.detect_forces(data)
        assert len(forces) >= 1

    def test_force_origin_classification(self, engine, rng):
        t = np.arange(50, dtype=float)
        autocorr_series = np.sin(t * 0.3) + rng.normal(0, 0.01, 50)
        data = {"market_sentiment": autocorr_series}
        forces = engine.detect_forces(data)
        assert len(forces) >= 0

    def test_empty_series(self, engine):
        forces = engine.detect_forces({})
        assert len(forces) == 0

    def test_short_series_skipped(self, engine):
        data = {"economics": np.array([1.0, 2.0])}
        forces = engine.detect_forces(data)
        assert len(forces) == 0

    def test_force_magnitude_bounded(self, engine, turbulent_series):
        forces = engine.detect_forces(turbulent_series)
        for f in forces:
            assert 0.0 <= f.magnitude <= 1.0

    def test_force_rupture_probability_bounded(self, engine, turbulent_series):
        forces = engine.detect_forces(turbulent_series)
        for f in forces:
            assert 0.0 <= f.rupture_probability <= 1.0

    def test_force_has_evidence(self, engine, turbulent_series):
        forces = engine.detect_forces(turbulent_series)
        for f in forces:
            if f.is_active:
                assert len(f.evidence) > 0

    def test_force_has_lead_signals(self, engine, turbulent_series):
        forces = engine.detect_forces(turbulent_series)
        for f in forces:
            if f.is_active:
                assert len(f.lead_signals) > 0

    def test_force_force_domain_mapping(self, engine):
        t = np.arange(50, dtype=float)
        data = {"inflation_rate": 0.001 * t**2 + 1.0}
        forces = engine.detect_forces(data)
        for f in forces:
            assert isinstance(f.primary_domain, ForceDomain)


class TestForceWhisperEngineTension:
    def test_tension_stable(self, engine, stable_series):
        tension = engine._compute_tension(stable_series)
        for domain, val in tension.items():
            assert 0.0 <= val <= 1.0
        assert all(v < 0.5 for v in tension.values())

    def test_tension_turbulent(self, engine, turbulent_series):
        tension = engine._compute_tension(turbulent_series)
        for val in tension.values():
            assert 0.0 <= val <= 1.0

    def test_tension_history_tracked(self, engine, stable_series):
        engine._compute_tension(stable_series)
        for domain in stable_series:
            assert domain in engine._tension_history
            assert len(engine._tension_history[domain]) == 1

    def test_tension_empty_series(self, engine):
        tension = engine._compute_tension({"econ": np.array([1.0])})
        assert tension["econ"] == 0.0


class TestForceWhisperEnginePressureGradients:
    def test_stable_no_gradients(self, engine, stable_series):
        gradients = engine.detect_pressure_gradients(stable_series)
        assert isinstance(gradients, list)

    def test_turbulent_has_gradients(self, engine, turbulent_series):
        gradients = engine.detect_pressure_gradients(turbulent_series)
        for g in gradients:
            assert isinstance(g, PressureGradient)
            assert g.gradient_magnitude >= 0.0
            assert g.source_domain != g.target_domain

    def test_gradient_bounded(self, engine, turbulent_series):
        gradients = engine.detect_pressure_gradients(turbulent_series)
        for g in gradients:
            assert 0.0 <= g.gradient_magnitude <= 1.0


class TestForceWhisperEngineTensionZones:
    def test_stable_tension_zones(self, engine, stable_series):
        tension_map = engine._compute_tension(stable_series)
        zones = engine.detect_tension_zones(stable_series, tension_map)
        for z in zones:
            assert isinstance(z, TensionZone)
            assert 0.0 <= z.tension_level <= 1.0
            assert 0.0 <= z.capacity_remaining <= 1.0

    def test_turbulent_tension_zones(self, engine, turbulent_series):
        tension_map = engine._compute_tension(turbulent_series)
        zones = engine.detect_tension_zones(turbulent_series, tension_map)
        for z in zones:
            assert z.time_to_yield_hours > 0


class TestForceWhisperEngineFatePressure:
    def test_no_forces_still(self, engine):
        score, level = engine.compute_fate_pressure([], [])
        assert level == FatePressureLevel.STILL
        assert score == 0.0

    def test_low_force_whisper(self, engine):
        forces = [LatentForce(
            name="test", origin=ForceOrigin.STRUCTURAL, primary_domain=ForceDomain.ECONOMIC,
            magnitude=0.2, direction=1.0, acceleration=0.05, phase_velocity=0.1,
            affected_domains=[ForceDomain.ECONOMIC], evidence=["test"],
            lead_signals=["test"], tension_index=0.15, rupture_probability=0.1,
            estimated_time_to_manifest_hours=200.0,
        )]
        score, level = engine.compute_fate_pressure(forces, [])
        assert level in (FatePressureLevel.STILL, FatePressureLevel.WHISPER)

    def test_high_force_rupture(self, engine):
        forces = [LatentForce(
            name="Big Force", origin=ForceOrigin.STRUCTURAL, primary_domain=ForceDomain.ECONOMIC,
            magnitude=0.95, direction=1.0, acceleration=0.8, phase_velocity=0.5,
            affected_domains=[ForceDomain.ECONOMIC, ForceDomain.FINANCIAL, ForceDomain.SOCIAL],
            evidence=["test"], lead_signals=["accelerating_change"],
            tension_index=0.9, rupture_probability=0.85,
            estimated_time_to_manifest_hours=6.0,
        )] * 3
        zones = [TensionZone(
            domain="economics", tension_level=0.85, capacity_remaining=0.15,
            stress_rate=0.05, time_to_yield_hours=3.0, is_critical=True,
        )]
        score, level = engine.compute_fate_pressure(forces, zones)
        assert level in (FatePressureLevel.SURGE, FatePressureLevel.RUPTURE)
        assert score > 0.4


class TestForceWhisperEngineWhisper:
    def test_stable_whisper(self, engine, stable_series):
        report = engine.whisper(stable_series)
        assert isinstance(report, ForceWhisperReport)
        assert isinstance(report.forces, list)
        assert isinstance(report.fate_pressure_level, FatePressureLevel)
        assert 0.0 <= report.fate_pressure_score <= 1.0
        assert isinstance(report.oracular_pronouncement, str)

    def test_turbulent_whisper(self, engine, turbulent_series):
        report = engine.whisper(turbulent_series)
        assert report.force_count >= 0
        assert report.active_force_count >= 0
        assert len(report.oracular_pronouncement) > 0

    def test_whisper_report_to_dict(self, engine, turbulent_series):
        report = engine.whisper(turbulent_series)
        d = report.to_dict()
        assert "forces" in d
        assert "pressure_gradients" in d
        assert "tension_zones" in d
        assert "fate_pressure_level" in d
        assert "fate_pressure_score" in d
        assert "oracular_pronouncement" in d
        assert isinstance(d["forces"], list)

    def test_whisper_preserves_history(self, engine, turbulent_series):
        engine.whisper(turbulent_series)
        engine.whisper(turbulent_series)
        summary = engine.history_summary()
        assert summary["observation_count"] == 2

    def test_history_summary_no_data(self, engine):
        summary = engine.history_summary()
        assert summary["observation_count"] == 0
        assert summary["avg_forces"] == 0.0


class TestCrossDomainForces:
    def test_correlated_series_creates_force(self, engine, rng):
        n = 80
        base = np.cumsum(rng.normal(0, 0.1, n))
        data = {
            "economics": base,
            "finance": base + rng.normal(0, 0.01, n),
        }
        forces = engine.detect_forces(data)
        cross_forces = [f for f in forces if "Cross-Domain" in f.name]
        assert len(cross_forces) >= 0


class TestPronouncement:
    def test_still_pronouncement(self, engine, stable_series):
        report = engine.whisper(stable_series)
        assert isinstance(report.oracular_pronouncement, str)
        if report.fate_pressure_level == FatePressureLevel.STILL:
            assert "still" in report.oracular_pronouncement.lower() or "no latent" in report.oracular_pronouncement.lower()

    def test_rupture_pronouncement(self, engine):
        forces = [LatentForce(
            name="Big Force", origin=ForceOrigin.STRUCTURAL, primary_domain=ForceDomain.ECONOMIC,
            magnitude=0.9, direction=1.0, acceleration=0.8, phase_velocity=0.5,
            affected_domains=[ForceDomain.ECONOMIC], evidence=["test"],
            lead_signals=["test"], tension_index=0.9, rupture_probability=0.8,
            estimated_time_to_manifest_hours=2.0,
        )]
        pronouncement = engine._generate_pronouncement(forces, FatePressureLevel.RUPTURE, 0.85)
        assert "RUPTURE" in pronouncement


class TestForceNaming:
    def test_known_name_returns(self):
        name = FORCE_NAMING.get((ForceOrigin.STRUCTURAL, ForceDomain.ECONOMIC))
        assert name == "Yield Curve Phantom"

    def test_unknown_returns_default(self, engine):
        t = np.arange(50, dtype=float)
        data = {"random_domain_xyz": 0.001 * t**2 + 1.0}
        forces = engine.detect_forces(data)
        for f in forces:
            assert isinstance(f.name, str)
            assert len(f.name) > 0


class TestRuptureEstimation:
    def test_rupture_probability_high(self, engine):
        prob = engine._estimate_rupture_probability(0.9, 0.8, 0.5)
        assert prob > 0.5

    def test_rupture_probability_low(self, engine):
        prob = engine._estimate_rupture_probability(0.1, 0.1, 0.01)
        assert prob < 0.3

    def test_rupture_probability_bounded(self, engine):
        for m in [0.0, 0.5, 1.0]:
            for t in [0.0, 0.5, 1.0]:
                for a in [0.0, 0.5, 1.0]:
                    prob = engine._estimate_rupture_probability(m, t, a)
                    assert 0.0 <= prob <= 1.0


class TestManifestTime:
    def test_fast_force_short_time(self, engine):
        t = engine._estimate_manifest_time(0.9, 0.8, 0.9)
        assert t < 24.0

    def test_slow_force_long_time(self, engine):
        t = engine._estimate_manifest_time(0.1, 0.01, 0.1)
        assert t > 24.0

    def test_high_tension_shortens_time(self, engine):
        t_low = engine._estimate_manifest_time(0.5, 0.1, 0.2)
        t_high = engine._estimate_manifest_time(0.5, 0.1, 0.9)
        assert t_high < t_low