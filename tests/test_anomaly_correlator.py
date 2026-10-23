"""Tests for AnomalyCorrelator — cross-domain anomaly correlation engine."""
import pytest
import numpy as np
from oracle.prediction.anomaly_correlator import (
    AnomalyCorrelator, AnomalyEvent, DomainCorrelation, AnomalyCluster,
    EmergingCorrelation, CascadePathway, AnomalyCorrelationResult,
    AnomalySeverity, CorrelationType, CascadeRiskLevel,
    KNOWN_CORRELATION_PATTERNS, DOMAIN_THRESHOLDS,
)


@pytest.fixture
def correlator():
    return AnomalyCorrelator(seed=42)


def _make_series(n=60, seed=42):
    rng = np.random.RandomState(seed)
    return {
        "equity": np.cumsum(rng.normal(0, 0.02, n)) + 0.5,
        "volatility": np.cumsum(rng.normal(0, 0.02, n)) + 0.4,
        "bonds": np.cumsum(rng.normal(0, 0.01, n)) + 0.5,
        "inflation": np.cumsum(rng.normal(0, 0.01, n)) + 0.5,
    }


class TestAnomalyDetection:
    def test_no_anomalies(self, correlator):
        states = {"equity": 0.5, "volatility": 0.5}
        anomalies = correlator.detect_anomalies(states)
        assert len(anomalies) == 0

    def test_single_anomaly(self, correlator):
        states = {"equity": 0.85, "volatility": 0.5}
        anomalies = correlator.detect_anomalies(states)
        equity_anomalies = [a for a in anomalies if a.domain == "equity"]
        assert len(equity_anomalies) >= 1
        assert equity_anomalies[0].z_score > 0

    def test_multiple_anomalies(self, correlator):
        states = {"equity": 0.85, "volatility": 0.75, "bonds": 0.5}
        anomalies = correlator.detect_anomalies(states)
        assert len(anomalies) >= 2

    def test_anomaly_severity(self, correlator):
        states = {"equity": 0.95, "volatility": 0.75, "bonds": 0.55}
        anomalies = correlator.detect_anomalies(states)
        severities = {a.domain: a.severity for a in anomalies}
        if "equity" in severities:
            assert severities["equity"] in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL)

    def test_direction_above_below(self, correlator):
        states = {"equity": 0.85, "bonds": 0.2}
        anomalies = correlator.detect_anomalies(states)
        for a in anomalies:
            if a.domain == "equity":
                assert a.direction == "above"
            elif a.domain == "bonds":
                assert a.direction == "below"

    def test_custom_baselines(self, correlator):
        states = {"equity": 0.6}
        baselines = {"equity": 0.55}
        std_devs = {"equity": 0.02}
        anomalies = correlator.detect_anomalies(states, baselines, std_devs)
        assert len(anomalies) >= 1

    def test_sorted_by_z_score(self, correlator):
        states = {"equity": 0.9, "volatility": 0.7}
        anomalies = correlator.detect_anomalies(states)
        for i in range(len(anomalies) - 1):
            assert abs(anomalies[i].z_score) >= abs(anomalies[i + 1].z_score)


class TestDomainCorrelation:
    def test_correlate_domains(self, correlator):
        series = _make_series()
        corrs = correlator.correlate_domains(series)
        assert len(corrs) >= 1
        for c in corrs:
            assert -1 <= c.correlation <= 1
            assert c.domain_a != c.domain_b

    def test_known_pattern_recognized(self, correlator):
        rng = np.random.RandomState(42)
        n = 100
        equity = np.cumsum(rng.normal(0, 0.02, n)) + 0.5
        volatility = -0.7 * equity + rng.normal(0, 0.1, n) + 0.6
        series = {"equity": equity, "volatility": volatility}
        corrs = correlator.correlate_domains(series)
        eq_vol = [c for c in corrs if {c.domain_a, c.domain_b} == {"equity", "volatility"}]
        assert len(eq_vol) >= 1

    def test_lagged_correlation(self, correlator):
        rng = np.random.RandomState(42)
        n = 100
        equity = np.cumsum(rng.normal(0, 0.02, n)) + 0.5
        bonds = np.zeros(n)
        for i in range(5, n):
            bonds[i] = -0.3 * equity[i - 5] + rng.normal(0, 0.1)
        series = {"equity": equity, "bonds": bonds}
        corrs = correlator.correlate_domains(series)
        assert len(corrs) >= 1

    def test_empty_series(self, correlator):
        corrs = correlator.correlate_domains({})
        assert len(corrs) == 0


class TestAnomalyClusters:
    def test_find_clusters(self, correlator):
        states = {"equity": 0.85, "volatility": 0.75, "bonds": 0.85}
        anomalies = correlator.detect_anomalies(states)
        series = _make_series(n=60)
        corrs = correlator.correlate_domains(series)
        clusters = correlator.find_clusters(anomalies, corrs)
        for cluster in clusters:
            assert len(cluster.domains_involved) >= 2
            assert cluster.cluster_score >= 0
            assert isinstance(cluster.cascade_risk, CascadeRiskLevel)

    def test_no_anomalies_no_clusters(self, correlator):
        states = {"equity": 0.5}
        anomalies = correlator.detect_anomalies(states)
        series = _make_series(n=60)
        corrs = correlator.correlate_domains(series)
        clusters = correlator.find_clusters(anomalies, corrs)
        assert len(clusters) == 0

    def test_cluster_score_high_for_critical(self, correlator):
        states = {"equity": 0.95, "volatility": 0.85, "bonds": 0.9}
        baselines = {"equity": 0.5, "volatility": 0.4, "bonds": 0.5}
        std_devs = {"equity": 0.1, "volatility": 0.1, "bonds": 0.1}
        anomalies = correlator.detect_anomalies(states, baselines, std_devs)
        series = _make_series(n=60)
        corrs = correlator.correlate_domains(series)
        clusters = correlator.find_clusters(anomalies, corrs)
        if clusters:
            assert clusters[0].cluster_score > 0


class TestEmergingCorrelations:
    def test_emerging_detection(self, correlator):
        rng = np.random.RandomState(42)
        n = 80
        equity = np.cumsum(rng.normal(0, 0.02, n)) + 0.5
        technology = np.zeros(n)
        technology[:40] = rng.normal(0, 0.02, 40) + 0.5
        technology[40:] = 0.6 * equity[40:] + rng.normal(0, 0.05, 40)
        series = {"equity": equity, "technology": technology}
        emerging = correlator.detect_emerging(series, recent_window=20)
        assert len(emerging) >= 1
        emerging_only = [e for e in emerging if e.is_emerging]
        if emerging_only:
            assert any(abs(e.divergence) > 0.1 for e in emerging_only)

    def test_no_emerging_short_series(self, correlator):
        series = {"equity": np.array([0.5, 0.51, 0.52]), "bonds": np.array([0.5, 0.49, 0.48])}
        emerging = correlator.detect_emerging(series, recent_window=20)
        assert len(emerging) == 0


class TestCascadePathways:
    def test_generate_cascade(self, correlator):
        states = {"equity": 0.85, "volatility": 0.8, "bonds": 0.75}
        baselines = {"equity": 0.5, "volatility": 0.4, "bonds": 0.5}
        std_devs = {"equity": 0.1, "volatility": 0.1, "bonds": 0.1}
        anomalies = correlator.detect_anomalies(states, baselines, std_devs)
        series = _make_series(n=60)
        corrs = correlator.correlate_domains(series)
        cascades = correlator.generate_cascade_pathways(anomalies, corrs)
        for cascade in cascades:
            assert len(cascade.steps) >= 2
            assert 0 <= cascade.cumulative_probability <= 1
            assert isinstance(cascade.risk_level, CascadeRiskLevel)
            assert cascade.narrative

    def test_empty_anomalies_no_cascade(self, correlator):
        cascades = correlator.generate_cascade_pathways([], [])
        assert len(cascades) == 0


class TestFullAnalysis:
    def test_analyze_normal(self, correlator):
        states = {"equity": 0.5, "volatility": 0.4, "bonds": 0.5}
        result = correlator.analyze(states)
        assert result.overall_risk_level in CascadeRiskLevel
        assert result.overall_risk_score >= 0
        assert isinstance(result.anomalies, list)

    def test_analyze_with_anomalies(self, correlator):
        states = {"equity": 0.85, "volatility": 0.75, "bonds": 0.5}
        result = correlator.analyze(states)
        assert len(result.anomalies) >= 2
        assert result.overall_risk_score > 0
        assert result.narrative

    def test_analyze_with_series(self, correlator):
        states = {"equity": 0.65, "volatility": 0.5, "bonds": 0.5}
        series = _make_series()
        result = correlator.analyze(states, domain_series=series)
        assert len(result.correlations) >= 1

    def test_analyze_known_patterns(self, correlator):
        states = {"equity": 0.85, "volatility": 0.8}
        result = correlator.analyze(states)
        eq_vol_corrs = [c for c in result.correlations
                        if {c.domain_a, c.domain_b} == {"equity", "volatility"}]
        assert len(eq_vol_corrs) >= 1

    def test_result_types(self, correlator):
        states = {"equity": 0.7, "inflation": 0.65}
        series = _make_series(n=80)
        result = correlator.analyze(states, domain_series=series)
        assert isinstance(result, AnomalyCorrelationResult)
        assert isinstance(result.anomalies, list)
        assert isinstance(result.correlations, list)
        assert isinstance(result.clusters, list)
        assert isinstance(result.emerging, list)
        assert isinstance(result.cascade_pathways, list)
        assert isinstance(result.evidence, list)

    def test_evidence_from_known_patterns(self, correlator):
        states = {"equity": 0.85, "volatility": 0.8}
        result = correlator.analyze(states)
        assert len(result.evidence) >= 1


class TestKnownPatterns:
    def test_patterns_exist(self):
        assert len(KNOWN_CORRELATION_PATTERNS) >= 10
        for p in KNOWN_CORRELATION_PATTERNS:
            assert "domains" in p
            assert "correlation" in p
            assert "description" in p
            assert "evidence" in p

    def test_thresholds_exist(self):
        assert len(DOMAIN_THRESHOLDS) >= 10
        for domain, thresholds in DOMAIN_THRESHOLDS.items():
            assert "low" in thresholds
            assert "critical" in thresholds