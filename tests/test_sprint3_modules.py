"""Tests for Ensemble Belief Calibrator and Temporal Consistency Checker."""
import pytest
import numpy as np
from oracle.prediction.ensemble_calibrator import (
    EnsembleBeliefCalibrator, EnsembleCalibrationResult, CalibratorType,
    CalibrationBin, CalibrationCurve, CalibrationDrift,
)
from oracle.prediction.temporal_consistency import (
    TemporalConsistencyChecker, TemporalConsistencyResult, ConsistencyIssue,
    ConsistencyViolation,
)


class TestEnsembleBeliefCalibrator:
    def test_initialization(self):
        cal = EnsembleBeliefCalibrator()
        assert cal.n_bins == 10

    def test_calibrate_basic(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes = [False, False, True, True, True]
        result = cal.calibrate(preds, outcomes)
        assert isinstance(result, EnsembleCalibrationResult)
        assert len(result.calibrated_predictions) == 5
        assert result.best_calibrator == CalibratorType.ENSEMBLE_CAL

    def test_binned_calibration(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.2, 0.4, 0.6, 0.8]
        outcomes = [False, False, True, True]
        result = cal.calibrate(preds, outcomes, method=CalibratorType.BINNED)
        assert result.best_calibrator == CalibratorType.BINNED

    def test_isotonic_calibration(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes = [False, False, True, True, True]
        result = cal.calibrate(preds, outcomes, method=CalibratorType.ISOTONIC)
        assert result.best_calibrator == CalibratorType.ISOTONIC

    def test_platt_calibration(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes = [False, False, True, True, True]
        result = cal.calibrate(preds, outcomes, method=CalibratorType.PLATT)
        assert result.best_calibrator == CalibratorType.PLATT

    def test_temperature_calibration(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes = [False, False, True, True, True]
        result = cal.calibrate(preds, outcomes, method=CalibratorType.TEMPERATURE)
        assert result.best_calibrator == CalibratorType.TEMPERATURE

    def test_ensemble_calibration(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes = [False, False, True, True, True]
        result = cal.calibrate(preds, outcomes, method=CalibratorType.ENSEMBLE_CAL)
        assert CalibratorType.ENSEMBLE_CAL.value in result.calibrator_outputs

    def test_calibration_curve(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes = [False, False, True, True, True]
        result = cal.calibrate(preds, outcomes)
        curve = result.calibration_curve
        assert isinstance(curve, CalibrationCurve)
        assert curve.ece >= 0
        assert curve.mce >= 0
        assert curve.brier_score >= 0
        assert 0 <= curve.reliability <= 1

    def test_calibration_bins(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes = [False, False, True, True, True]
        result = cal.calibrate(preds, outcomes)
        assert len(result.calibration_curve.bins) == 5

    def test_well_calibrated_predictions(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes = [False, False, True, True, True]
        result = cal.calibrate(preds, outcomes)
        assert result.calibration_curve.ece < 1.0

    def test_poorly_calibrated_predictions(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.9, 0.9, 0.9, 0.9, 0.9]
        outcomes = [False, False, False, True, True]
        result = cal.calibrate(preds, outcomes)
        assert result.calibration_curve.ece > 0

    def test_confidence_adjustment(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes = [False, False, True, True, True]
        result = cal.calibrate(preds, outcomes)
        assert 0 < result.confidence_adjustment <= 1

    def test_recommendations(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes = [False, False, True, True, True]
        result = cal.calibrate(preds, outcomes)
        assert len(result.recommendations) > 0

    def test_no_outcomes_generates_synthetic(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = cal.calibrate(preds, outcomes=None)
        assert len(result.calibrated_predictions) == 5

    def test_calibrated_predictions_bounded(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        preds = [0.01, 0.5, 0.99]
        outcomes = [False, True, True]
        result = cal.calibrate(preds, outcomes)
        for p in result.calibrated_predictions:
            assert 0.01 <= p <= 0.99

    def test_calibrator_type_enum(self):
        assert CalibratorType.BINNED.value == "binned"
        assert CalibratorType.ISOTONIC.value == "isotonic"
        assert CalibratorType.PLATT.value == "platt"
        assert CalibratorType.TEMPERATURE.value == "temperature"
        assert CalibratorType.ENSEMBLE_CAL.value == "ensemble_cal"

    def test_multiple_calibrations_build_history(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        for _ in range(10):
            cal.calibrate([0.3, 0.7], [False, True])
        result = cal.calibrate([0.5, 0.5], [True, False])
        assert len(cal._history) >= 20
        assert len(result.drift_results) > 0

    def test_empty_predictions(self):
        cal = EnsembleBeliefCalibrator(n_bins=5)
        result = cal.calibrate([], [])
        assert len(result.calibrated_predictions) == 0


class TestTemporalConsistencyChecker:
    def test_initialization(self):
        tc = TemporalConsistencyChecker()
        assert tc.discount_rate == 0.05
        assert tc.smoothness_threshold == 0.3

    def test_consistent_predictions(self):
        tc = TemporalConsistencyChecker()
        preds = {
            "1day": {"gdp": 0.5, "inflation": 0.3},
            "1month": {"gdp": 0.52, "inflation": 0.31},
            "1year": {"gdp": 0.55, "inflation": 0.35},
            "5year": {"gdp": 0.60, "inflation": 0.40},
        }
        result = tc.check(preds)
        assert isinstance(result, TemporalConsistencyResult)
        assert 0 <= result.consistency_score <= 1

    def test_inconsistent_predictions(self):
        tc = TemporalConsistencyChecker()
        preds = {
            "1day": {"gdp": 0.5},
            "1month": {"gdp": 0.9},
            "1year": {"gdp": 0.2},
        }
        result = tc.check(preds)
        assert len(result.violations) > 0

    def test_cross_horizon_mismatch(self):
        tc = TemporalConsistencyChecker()
        preds = {
            "1day": {"gdp": 0.5},
            "1year": {"gdp": 0.05},
        }
        result = tc.check(preds)
        mismatches = [v for v in result.violations if v.issue_type == ConsistencyIssue.CROSS_HORIZON_MISMATCH]
        assert len(mismatches) > 0

    def test_monotonicity_violation(self):
        tc = TemporalConsistencyChecker()
        preds = {
            "1day": {"gdp": 0.5},
            "1month": {"gdp": 0.8},
            "1year": {"gdp": 0.4},
            "5year": {"gdp": 0.7},
        }
        result = tc.check(preds)
        mono_violations = [v for v in result.violations if v.issue_type == ConsistencyIssue.MONOTONICITY_VIOLATION]
        assert len(mono_violations) > 0

    def test_discount_consistency(self):
        tc = TemporalConsistencyChecker()
        preds = {
            "1day": {"gdp": 0.5},
            "1month": {"gdp": 0.52},
            "1year": {"gdp": 0.55},
        }
        uncertainties = {
            "1day": {"gdp": 0.1},
            "1month": {"gdp": 0.05},
            "1year": {"gdp": 0.02},
        }
        result = tc.check(preds, uncertainties)
        discount_violations = [v for v in result.violations if v.issue_type == ConsistencyIssue.DISCOUNT_INCONSISTENCY]
        assert len(discount_violations) > 0

    def test_smoothness_score(self):
        tc = TemporalConsistencyChecker()
        smooth_preds = {
            "1day": {"gdp": 0.50},
            "1month": {"gdp": 0.52},
            "1year": {"gdp": 0.55},
            "5year": {"gdp": 0.60},
        }
        result = tc.check(smooth_preds)
        assert result.smoothness_score >= 0

    def test_uncertainty_expansion(self):
        tc = TemporalConsistencyChecker()
        preds = {
            "1day": {"gdp": 0.5},
            "1month": {"gdp": 0.52},
            "1year": {"gdp": 0.55},
        }
        uncertainties = {
            "1day": {"gdp": 0.05},
            "1month": {"gdp": 0.10},
            "1year": {"gdp": 0.20},
        }
        result = tc.check(preds, uncertainties)
        assert result.uncertainty_expansion_score >= 0

    def test_change_point_detection(self):
        tc = TemporalConsistencyChecker()
        preds = {
            "1day": {"gdp": 0.50},
            "1week": {"gdp": 0.51},
            "1month": {"gdp": 0.52},
            "3month": {"gdp": 0.53},
            "6month": {"gdp": 0.54},
            "1year": {"gdp": 0.90},
        }
        result = tc.check(preds)
        cps = [v for v in result.violations if v.issue_type == ConsistencyIssue.CHANGE_POINT]
        assert len(cps) > 0

    def test_recommendations(self):
        tc = TemporalConsistencyChecker()
        preds = {
            "1day": {"gdp": 0.5},
            "1month": {"gdp": 0.52},
            "1year": {"gdp": 0.55},
        }
        result = tc.check(preds)
        assert len(result.recommendations) > 0

    def test_single_horizon(self):
        tc = TemporalConsistencyChecker()
        result = tc.check({"1day": {"gdp": 0.5}})
        assert isinstance(result, TemporalConsistencyResult)

    def test_empty_predictions(self):
        tc = TemporalConsistencyChecker()
        result = tc.check({})
        assert result.is_consistent is True

    def test_consistency_issue_enum(self):
        assert ConsistencyIssue.CROSS_HORIZON_MISMATCH.value == "cross_horizon_mismatch"
        assert ConsistencyIssue.MONOTONICITY_VIOLATION.value == "monotonicity_violation"
        assert ConsistencyIssue.DISCOUNT_INCONSISTENCY.value == "discount_inconsistency"
        assert ConsistencyIssue.HORIZON_INCOHERENCE.value == "horizon_incoherence"
        assert ConsistencyIssue.CHANGE_POINT.value == "change_point"
        assert ConsistencyIssue.BACKWARD_FORWARD_MISMATCH.value == "backward_forward_mismatch"

    def test_backward_forward_mismatch(self):
        tc = TemporalConsistencyChecker()
        preds = {
            "1day": {"gdp": 0.5},
            "1year": {"gdp": 0.01},
        }
        result = tc.check(preds)
        bf_or_cross = [v for v in result.violations if v.issue_type in (ConsistencyIssue.BACKWARD_FORWARD_MISMATCH, ConsistencyIssue.CROSS_HORIZON_MISMATCH)]
        assert len(bf_or_cross) > 0

    def test_horizon_coherence(self):
        tc = TemporalConsistencyChecker()
        preds = {
            "1day": {"gdp": 0.5},
            "1month": {"gdp": 0.1},
            "1year": {"gdp": 0.9},
            "5year": {"gdp": 0.2},
            "10year": {"gdp": 0.8},
        }
        result = tc.check(preds)
        coherence = [v for v in result.violations if v.issue_type == ConsistencyIssue.HORIZON_INCOHERENCE]
        assert len(coherence) > 0

    def test_violation_severity_bounded(self):
        tc = TemporalConsistencyChecker()
        preds = {
            "1day": {"x": 0.1}, "1month": {"x": 0.9},
            "1year": {"x": 0.1}, "5year": {"x": 0.9},
        }
        result = tc.check(preds)
        for v in result.violations:
            assert 0 <= v.severity <= 1

    def test_consistent_smooth_predictions(self):
        tc = TemporalConsistencyChecker()
        preds = {
            "1day": {"gdp": 0.500}, "1week": {"gdp": 0.505},
            "1month": {"gdp": 0.520}, "3month": {"gdp": 0.535},
            "6month": {"gdp": 0.550}, "1year": {"gdp": 0.570},
        }
        result = tc.check(preds)
        assert result.smoothness_score > 0.5