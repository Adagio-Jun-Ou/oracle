"""Tests for PredictionTracker — self-calibrating accuracy engine."""
import pytest
from datetime import date, timedelta
from oracle.prediction.prediction_tracker import (
    PredictionTracker, PredictionEntry, PredictionStatus,
    AccuracyGrade, DomainAccuracy, CalibrationBin, TrackerReport
)


@pytest.fixture
def tracker():
    return PredictionTracker(seed=42)


def _add_predictions(tracker, n=20, domain="equity"):
    pids = []
    rng = __import__("numpy").random.RandomState(42)
    for i in range(n):
        pred = 0.5 + rng.normal(0, 0.1)
        lb = pred - 0.1
        ub = pred + 0.1
        pid = tracker.record(domain, round(pred, 4), round(lb, 4), round(ub, 4),
                            horizon_days=7, confidence=0.6, method="test",
                            evidence=[f"test evidence {i}"])
        actual = pred + rng.normal(0, 0.05)
        tracker.resolve(pid, round(actual, 4))
        pids.append(pid)
    return pids


class TestRecording:
    def test_record_prediction(self, tracker):
        pid = tracker.record("equity", 0.65, 0.55, 0.75, 7, 0.7, "arima")
        assert pid.startswith("equity_")
        entry = tracker.get_prediction(pid)
        assert entry.prediction == 0.65
        assert entry.lower_bound == 0.55
        assert entry.upper_bound == 0.75
        assert entry.status == PredictionStatus.PENDING

    def test_record_with_evidence(self, tracker):
        pid = tracker.record("inflation", 0.55, 0.45, 0.65, 14, 0.6,
                            "ets", ["CPI data", "Fed minutes"])
        entry = tracker.get_prediction(pid)
        assert len(entry.evidence) == 2
        assert "CPI data" in entry.evidence

    def test_unique_ids(self, tracker):
        p1 = tracker.record("equity", 0.6, 0.5, 0.7, 7, 0.7)
        p2 = tracker.record("equity", 0.55, 0.45, 0.65, 7, 0.6)
        assert p1 != p2


class TestResolution:
    def test_resolve_prediction(self, tracker):
        pid = tracker.record("equity", 0.65, 0.55, 0.75, 7, 0.7)
        entry = tracker.resolve(pid, 0.68)
        assert entry.actual_outcome == 0.68
        assert entry.status == PredictionStatus.RESOLVED
        assert entry.date_resolved is not None

    def test_resolve_nonexistent_raises(self, tracker):
        with pytest.raises(ValueError):
            tracker.resolve("nonexistent_id", 0.5)

    def test_within_ci(self, tracker):
        pid = tracker.record("equity", 0.65, 0.55, 0.75, 7, 0.7)
        tracker.resolve(pid, 0.68)
        r = tracker.report()
        assert r.overall_within_ci_pct == 100.0

    def test_outside_ci(self, tracker):
        pid = tracker.record("equity", 0.65, 0.55, 0.75, 7, 0.7)
        tracker.resolve(pid, 0.40)
        r = tracker.report()
        assert r.overall_within_ci_pct == 0.0


class TestReport:
    def test_empty_report(self, tracker):
        r = tracker.report()
        assert r.total_predictions == 0
        assert r.resolved_predictions == 0
        assert r.overall_grade == AccuracyGrade.N_A

    def test_report_with_resolved(self, tracker):
        _add_predictions(tracker, n=10)
        r = tracker.report()
        assert r.total_predictions == 10
        assert r.resolved_predictions == 10
        assert r.overall_mae >= 0
        assert r.overall_rmse >= 0
        assert r.overall_brier >= 0

    def test_report_grades(self, tracker):
        for i in range(20):
            pid = tracker.record("equity", 0.60, 0.50, 0.70, 7, 0.7)
            tracker.resolve(pid, 0.60)
        r = tracker.report()
        assert r.overall_grade in (AccuracyGrade.A_PLUS, AccuracyGrade.A, AccuracyGrade.B)

    def test_domain_accuracies(self, tracker):
        _add_predictions(tracker, n=10, domain="equity")
        _add_predictions(tracker, n=10, domain="bonds")
        r = tracker.report()
        assert len(r.domain_accuracies) == 2
        domains = [d.domain for d in r.domain_accuracies]
        assert "equity" in domains
        assert "bonds" in domains

    def test_best_and_worst_domains(self, tracker):
        for i in range(5):
            pid = tracker.record("good_domain", 0.6, 0.5, 0.7, 7, 0.7)
            tracker.resolve(pid, 0.6)
        for i in range(5):
            pid = tracker.record("bad_domain", 0.5, 0.3, 0.7, 7, 0.5)
            tracker.resolve(pid, 0.1)
        r = tracker.report()
        assert len(r.best_domains) >= 1
        assert len(r.worst_domains) >= 1

    def test_narrative(self, tracker):
        _add_predictions(tracker, n=5)
        r = tracker.report()
        assert r.narrative
        assert len(r.narrative) > 20


class TestCalibrationCurve:
    def test_calibration_bins(self, tracker):
        rng = __import__("numpy").random.RandomState(42)
        for p in [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pid = tracker.record("equity", p, p - 0.1, p + 0.1, 7, 0.6)
            tracker.resolve(pid, p + rng.normal(0, 0.02))
        r = tracker.report()
        assert len(r.calibration_curve) >= 3

    def test_no_calibration_with_few_predictions(self, tracker):
        pid = tracker.record("equity", 0.5, 0.4, 0.6, 7, 0.5)
        tracker.resolve(pid, 0.52)
        r = tracker.report()
        assert len(r.calibration_curve) == 0


class TestAutoResolve:
    def test_auto_resolve_with_current_states(self, tracker):
        pid = tracker.record("equity", 0.65, 0.55, 0.75, 0, 0.7)
        entry = tracker.get_prediction(pid)
        entry.date_made = (date.today() - timedelta(days=10)).isoformat()
        count = tracker.auto_resolve({"equity": 0.60})
        assert count == 1
        assert entry.status == PredictionStatus.RESOLVED

    def test_auto_resolve_skips_not_yet_due(self, tracker):
        pid = tracker.record("equity", 0.65, 0.55, 0.75, 30, 0.7)
        entry = tracker.get_prediction(pid)
        entry.date_made = date.today().isoformat()
        count = tracker.auto_resolve({"equity": 0.60})
        assert count == 0
        assert entry.status == PredictionStatus.PENDING


class TestExpireOld:
    def test_expire_old_predictions(self, tracker):
        pid = tracker.record("equity", 0.65, 0.55, 0.75, 7, 0.7)
        entry = tracker.get_prediction(pid)
        entry.date_made = (date.today() - timedelta(days=400)).isoformat()
        count = tracker.expire_old(max_age_days=365)
        assert count == 1
        assert entry.status == PredictionStatus.EXPIRED


class TestListPredictions:
    def test_list_all(self, tracker):
        tracker.record("equity", 0.6, 0.5, 0.7, 7, 0.7)
        tracker.record("bonds", 0.5, 0.4, 0.6, 7, 0.5)
        preds = tracker.list_predictions()
        assert len(preds) == 2

    def test_list_by_domain(self, tracker):
        tracker.record("equity", 0.6, 0.5, 0.7, 7, 0.7)
        tracker.record("bonds", 0.5, 0.4, 0.6, 7, 0.5)
        preds = tracker.list_predictions(domain="equity")
        assert len(preds) == 1

    def test_list_by_status(self, tracker):
        pid = tracker.record("equity", 0.6, 0.5, 0.7, 7, 0.7)
        tracker.resolve(pid, 0.62)
        tracker.record("bonds", 0.5, 0.4, 0.6, 7, 0.5)
        resolved = tracker.list_predictions(status=PredictionStatus.RESOLVED)
        assert len(resolved) == 1


class TestImprovementTrend:
    def test_improvement_detected(self, tracker):
        for i in range(10):
            pid = tracker.record("equity", 0.60 + i * 0.01, 0.50 + i * 0.01, 0.70 + i * 0.01, 7, 0.7)
            tracker.resolve(pid, 0.60 + i * 0.01)
        r = tracker.report()
        assert r.improvement_trend is not None

    def test_domain_improvement_trend(self, tracker):
        _add_predictions(tracker, n=20)
        r = tracker.report()
        for d in r.domain_accuracies:
            assert d.improvement_trend is not None