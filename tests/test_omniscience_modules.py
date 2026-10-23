"""Tests for prediction omniscience modules: convergence, blind_spot, phase_space, forecastability, auto_calibration."""
import pytest
import numpy as np
from oracle.prediction.convergence import (
    PredictiveConvergenceEngine, ModelPrediction, ConvergenceStrength,
    GlobalConvergenceReport, ConvergenceCluster,
)
from oracle.prediction.blind_spot import (
    BlindSpotDetector, BlindSpotCategory, BlindSpotSeverity, BlindSpotReport,
)
from oracle.prediction.phase_space import (
    PhaseSpaceMapper, AttractorType, BifurcationType, PhaseSpaceResult,
)
from oracle.prediction.forecastability import (
    EntropicForecastabilityIndex, ForecastabilityClass, DomainForecastability,
    ForecastabilityReport,
)
from oracle.prediction.auto_calibration import (
    AutoCalibrationEngine, CalibrationBias, DriftDirection,
    AutoCalibrationReport,
)


class TestPredictiveConvergenceEngine:
    def setup_method(self):
        self.engine = PredictiveConvergenceEngine()

    def _make_models(self, predictions, families=None, confidences=None, reliabilities=None):
        families = families or ["structural"] * len(predictions)
        confidences = confidences or [0.7] * len(predictions)
        reliabilities = reliabilities or [0.6] * len(predictions)
        return [
            ModelPrediction(
                model_name=f"model_{i}",
                model_family=families[i] if i < len(families) else "unknown",
                prediction=predictions[i],
                confidence=confidences[i],
                domain_reliability=reliabilities[i],
                assumptions=["efficient_market" if i % 2 == 0 else "behavioral"],
            )
            for i in range(len(predictions))
        ]

    def test_condorcet_basic(self):
        p = self.engine.compute_condorcet(5, 7, 0.6)
        assert 0.3 < p < 0.6

    def test_condorcet_unanimous(self):
        p = self.engine.compute_condorcet(7, 7, 0.8)
        assert 0.1 < p < 0.5

    def test_condorcet_minority(self):
        p = self.engine.compute_condorcet(2, 7, 0.6)
        assert p > 0.9

    def test_simple_convergence(self):
        models = self._make_models([0.65, 0.67, 0.66, 0.64])
        preds = {"equity": models}
        report = self.engine.converge(preds)
        assert len(report.results) == 1
        result = report.results["equity"]
        assert 0.63 < result.final_prediction < 0.68
        assert result.convergence_cluster is not None
        assert result.convergence_cluster.n_models_agreeing >= 3

    def test_divergence_detection(self):
        models = self._make_models(
            [0.8, 0.2, 0.75, 0.15, 0.7],
            families=["structural", "behavioral", "structural", "extreme", "behavioral"],
        )
        preds = {"geopolitical": models}
        report = self.engine.converge(preds, convergence_threshold=0.05)
        result = report.results["geopolitical"]
        assert result.divergence_zone is not None
        assert result.divergence_zone.spread > 0.1

    def test_false_convergence_detection(self):
        models = self._make_models(
            [0.65, 0.66, 0.65],
            families=["efficient_market", "efficient_market", "efficient_market"],
        )
        preds = {"equity": models}
        report = self.engine.converge(preds)
        result = report.results["equity"]
        assert result.convergence_cluster is not None
        assert result.convergence_cluster.false_convergence_risk > 0

    def test_empty_predictions(self):
        report = self.engine.converge({})
        assert len(report.results) == 0
        assert report.n_convergent == 0

    def test_model_weighting(self):
        models = self._make_models(
            [0.5, 0.7],
            confidences=[0.9, 0.3],
            reliabilities=[0.8, 0.4],
        )
        preds = {"test": models}
        report = self.engine.converge(preds)
        weights = report.results["test"].model_weight_map
        assert weights["model_0"] > weights["model_1"]

    def test_effective_n_models(self):
        models = self._make_models([0.5, 0.52, 0.51])
        preds = {"test": models}
        report = self.engine.converge(preds)
        assert report.results["test"].effective_n_models >= 1.0

    def test_reliability_update(self):
        self.engine.update_reliability("model_0", "equity", 0.7, 0.55)
        assert "model_0" in self.engine.model_reliability
        assert "equity" in self.engine.model_reliability["model_0"]


class TestBlindSpotDetector:
    def setup_method(self):
        self.detector = BlindSpotDetector()

    def test_basic_detection(self):
        domain_states = {"equity": 0.7, "volatility": 0.3}
        report = self.detector.detect(domain_states=domain_states)
        assert report.total_blind_spots > 0
        assert isinstance(report.omniscience_gap, float)
        assert 0.0 <= report.omniscience_gap <= 1.0

    def test_edge_case_detection(self):
        domain_states = {"equity": 0.96, "volatility": 0.03}
        report = self.detector.detect(domain_states=domain_states)
        edge_spots = [b for b in report.blind_spots if b.category == BlindSpotCategory.EDGE_CASE]
        assert len(edge_spots) >= 1

    def test_unmodeled_domains(self):
        report = self.detector.detect(domain_states={"equity": 0.5})
        assert len(report.unmodeled_domains) > 0

    def test_with_predictions(self):
        predictions = {
            "equity": {"signal_fusion": 0.65, "black_swan": 0.15, "daily_oracle": 0.60},
        }
        report = self.detector.detect(predictions=predictions)
        assert report.total_blind_spots > 0

    def test_narrative_traps(self):
        report = self.detector.detect()
        narrative_spots = [b for b in report.blind_spots if b.category == BlindSpotCategory.NARRATIVE_TRAP]
        assert len(narrative_spots) > 0

    def test_coverage_score(self):
        report = self.detector.detect()
        assert 0.0 <= report.coverage_score <= 1.0


class TestPhaseSpaceMapper:
    def setup_method(self):
        self.mapper = PhaseSpaceMapper()

    def test_goldilocks_state(self):
        states = {"equity": 0.6, "volatility": 0.25, "inflation": 0.4,
                  "geopolitical": 0.3, "technology": 0.6, "sentiment": 0.6,
                  "economic": 0.6, "monetary": 0.5}
        result = self.mapper.map_phase_space(states)
        assert result.current_attractor in ("goldilocks", "late_cycle_euphoria")
        assert result.resilience >= 0.0
        assert len(result.attractors) >= 3

    def test_crisis_state(self):
        states = {"equity": 0.15, "volatility": 0.85, "inflation": 0.5,
                  "geopolitical": 0.7, "technology": 0.3, "sentiment": 0.1,
                  "economic": 0.1, "monetary": 0.8}
        result = self.mapper.map_phase_space(states)
        assert result.current_attractor in ("crisis", "stagflation", "late_cycle_euphoria")
        assert result.lyapunov_exponent != 0

    def test_bifurcation_boundaries(self):
        states = {"equity": 0.45, "volatility": 0.35, "inflation": 0.35}
        result = self.mapper.map_phase_space(states)
        assert result.phase_space_dimension >= 3

    def test_escape_routes(self):
        states = {"equity": 0.5, "volatility": 0.4, "economic": 0.5}
        result = self.mapper.map_phase_space(states)
        assert isinstance(result.escape_routes, list)

    def test_saddle_points(self):
        states = {"equity": 0.5, "volatility": 0.35}
        result = self.mapper.map_phase_space(states)
        assert isinstance(result.saddle_points, list)

    def test_resilience_range(self):
        states = {"equity": 0.6, "volatility": 0.2}
        result = self.mapper.map_phase_space(states)
        assert 0.0 <= result.resilience <= 1.0

    def test_narrative_generation(self):
        states = {"equity": 0.5, "volatility": 0.3}
        result = self.mapper.map_phase_space(states)
        assert len(result.narrative) > 50


class TestEntropicForecastabilityIndex:
    def setup_method(self):
        self.engine = EntropicForecastabilityIndex()

    def test_compute_domain(self):
        result = self.engine.compute_domain("equity")
        assert result.domain == "equity"
        assert 0.0 <= result.forecastability_index <= 1.0
        assert result.entropy > 0.0
        assert result.predictability_horizon_days > 0

    def test_forecastability_classes(self):
        result_high = self.engine.compute_domain("monetary")  # low entropy
        result_low = self.engine.compute_domain("crypto")  # high entropy
        assert result_high.forecastability_index > result_low.forecastability_index
        assert result_high.forecastability_class in (
            ForecastabilityClass.HIGHLY_PREDICTABLE,
            ForecastabilityClass.PREDICTABLE,
            ForecastabilityClass.MODERATELY_PREDICTABLE,
        )
        assert result_low.forecastability_class in (
            ForecastabilityClass.WEAKLY_PREDICTABLE,
            ForecastabilityClass.UNPREDICTABLE,
            ForecastabilityClass.MODERATELY_PREDICTABLE,
        )
        assert result_low.forecastability_class in (
            ForecastabilityClass.WEAKLY_PREDICTABLE,
            ForecastabilityClass.UNPREDICTABLE,
        )

    def test_compute_all(self):
        states = {"equity": 0.6, "volatility": 0.3, "monetary": 0.5}
        report = self.engine.compute_all(domain_states=states)
        assert len(report.domains) >= 3
        assert 0.0 <= report.overall_forecastability <= 1.0
        assert len(report.most_predictable) > 0
        assert len(report.least_predictable) > 0

    def test_extreme_states_increase_entropy(self):
        result_normal = self.engine.compute_domain("equity", domain_states={"equity": 0.5})
        result_extreme = self.engine.compute_domain("equity", domain_states={"equity": 0.95})
        assert result_extreme.entropy >= result_normal.entropy

    def test_time_series_entropy(self):
        rng = np.random.RandomState(42)
        ts = rng.normal(0, 1, 100)
        result = self.engine.compute_domain("equity", time_series=ts)
        assert result.entropy > 0

    def test_omniscience_gap(self):
        report = self.engine.compute_all()
        assert report.omniscience_gap > 0
        assert report.omniscience_gap < 1.0
        assert report.practical_prediction_ceiling < report.overall_forecastability

    def test_recommendations(self):
        result = self.engine.compute_domain("crypto")
        assert len(result.recommendation) > 20
        result2 = self.engine.compute_domain("monetary")
        assert "predictable" in result2.recommendation.lower()


class TestAutoCalibrationEngine:
    def setup_method(self):
        self.engine = AutoCalibrationEngine()

    def _populate(self, domain="equity", n=30, bias=0.0, seed=42):
        rng = np.random.RandomState(seed)
        preds = []
        for i in range(n):
            predicted = rng.uniform(0.2, 0.8)
            actual = predicted + bias + rng.normal(0, 0.1)
            actual = max(0, min(1, actual))
            pid = self.engine.record(domain, predicted, actual=actual, confidence=0.7)
            preds.append(pid)
        return preds

    def test_basic_calibration(self):
        self._populate("test_domain", n=30)
        cal = self.engine.calibrate("test_domain")
        assert cal.n_resolved >= 25
        assert 0.0 <= cal.brier_score <= 1.0
        assert 0.0 <= cal.skill_score <= 1.0

    def test_well_calibrated(self):
        self._populate("well", n=50, bias=0.0)
        cal = self.engine.calibrate("well")
        assert cal.bias in (CalibrationBias.WELL_CALIBRATED, CalibrationBias.UNDERCONFIDENT, CalibrationBias.OVERCONFIDENT)

    def test_overconfident_bias(self):
        rng = np.random.RandomState(42)
        for i in range(30):
            predicted = rng.choice([0.9, 0.05])
            actual = predicted * 0.6 + 0.2
            self.engine.record("overconf", predicted, actual=actual)
        cal = self.engine.calibrate("overconf")
        assert cal.bias in (CalibrationBias.OVERCONFIDENT, CalibrationBias.EXTREME_BIAS, CalibrationBias.WELL_CALIBRATED)

    def test_calibrate_all(self):
        self._populate("domain_a", n=20)
        self._populate("domain_b", n=15)
        report = self.engine.calibrate_all()
        assert len(report.domain_calibrations) == 2
        assert 0.0 <= report.overall_brier <= 1.0

    def test_correction_applied(self):
        self._populate("correctable", n=30, bias=0.1)
        cal = self.engine.calibrate("correctable")
        corrected = self.engine.apply_correction("correctable", 0.6)
        assert 0.01 <= corrected <= 0.99

    def test_drift_detection(self):
        rng = np.random.RandomState(42)
        for i in range(20):
            self.engine.record("drift", 0.5 + rng.normal(0, 0.05), actual=0.55 + rng.normal(0, 0.03))
        for i in range(20):
            self.engine.record("drift", 0.5 + rng.normal(0, 0.05), actual=0.5 + rng.normal(0, 0.03))
        cal = self.engine.calibrate("drift")
        assert cal.drift in (DriftDirection.IMPROVING, DriftDirection.STABLE, DriftDirection.DEGRADING)

    def test_calibration_bins(self):
        self._populate("binned", n=50)
        cal = self.engine.calibrate("binned")
        assert len(cal.calibration_curve) >= 2
        for b in cal.calibration_curve:
            assert 0.0 <= b.observed_frequency <= 1.0

    def test_insufficient_data(self):
        self.engine.record("sparse", 0.5, actual=0.6)
        cal = self.engine.calibrate("sparse")
        assert cal.n_resolved < 3
        assert cal.correction_factor == 1.0

    def test_empty_report(self):
        report = self.engine.calibrate_all()
        assert report.overall_brier == 0.25
        assert "No resolved" in report.narrative

    def test_recommendations_generated(self):
        self._populate("recs", n=30)
        report = self.engine.calibrate_all()
        assert len(report.recommendations) >= 1

    def test_narrative_generated(self):
        self._populate("narr", n=20)
        report = self.engine.calibrate_all()
        assert len(report.narrative) > 20