"""Tests for data quality scoring system."""
import pytest
from datetime import datetime, timedelta
from oracle.prediction.data_quality import (
    DataQualityScorer, DomainQualityReport, DataSourceProvenance,
    QualityAdjustedForecaster, DOMAIN_FIELD_REGISTRY,
)


class TestDataQualityScorer:
    def setup_method(self):
        self.scorer = DataQualityScorer()

    def test_completeness_full(self):
        state = {
            "us_china_tension": 0.5, "russia_west_tension": 0.4,
            "middle_east_instability": 0.6, "trade_protectionism": 0.3,
            "energy_geopolitics": 0.5, "cyber_warfare_likelihood": 0.4,
        }
        score, missing = self.scorer.score_completeness("geopolitical", state)
        assert score == 1.0
        assert len(missing) == 0

    def test_completeness_partial(self):
        state = {"us_china_tension": 0.5, "russia_west_tension": 0.4}
        score, missing = self.scorer.score_completeness("geopolitical", state)
        assert score == 2 / 6
        assert len(missing) == 4
        assert "middle_east_instability" in missing

    def test_completeness_empty(self):
        state = {}
        score, missing = self.scorer.score_completeness("geopolitical", state)
        assert score == 0.0
        assert len(missing) == 6

    def test_completeness_unknown_domain(self):
        score, missing = self.scorer.score_completeness("unknown_domain", {})
        assert score == 1.0
        assert len(missing) == 0

    def test_accuracy_within_range(self):
        state = {
            "us_china_tension": 0.5, "russia_west_tension": 0.4,
            "middle_east_instability": 0.6, "trade_protectionism": 0.3,
            "energy_geopolitics": 0.5, "cyber_warfare_likelihood": 0.4,
        }
        score, suspicious = self.scorer.score_accuracy("geopolitical", state)
        assert score == 1.0
        assert len(suspicious) == 0

    def test_accuracy_out_of_range(self):
        state = {
            "us_china_tension": 5.0, "russia_west_tension": -1,
            "middle_east_instability": 0.6, "trade_protectionism": 0.3,
            "energy_geopolitics": 0.5, "cyber_warfare_likelihood": 0.4,
        }
        score, suspicious = self.scorer.score_accuracy("geopolitical", state)
        assert score == 4 / 6
        assert len(suspicious) == 2

    def test_accuracy_unknown_domain(self):
        score, suspicious = self.scorer.score_accuracy("unknown_domain", {})
        assert score == 1.0

    def test_score_domain_complete(self):
        state = {
            "us_china_tension": 0.5, "russia_west_tension": 0.4,
            "middle_east_instability": 0.6, "trade_protectionism": 0.3,
            "energy_geopolitics": 0.5, "cyber_warfare_likelihood": 0.4,
        }
        report = self.scorer.score_domain("geopolitical", state)
        assert report.overall_score > 0.8
        assert report.completeness_score == 1.0
        assert report.accuracy_score == 1.0
        assert report.confidence_multiplier > 0.9
        assert isinstance(report.recommendations, list)

    def test_score_domain_sparse(self):
        state = {"us_china_tension": 5.0}
        report = self.scorer.score_domain("geopolitical", state)
        assert report.completeness_score < 0.5
        assert report.accuracy_score < 1.0
        assert report.overall_score < 0.5
        assert len(report.recommendations) > 0

    def test_score_all_domains(self):
        domains = {
            "geopolitical": {
                "us_china_tension": 0.5, "russia_west_tension": 0.4,
                "middle_east_instability": 0.6, "trade_protectionism": 0.3,
                "energy_geopolitics": 0.5, "cyber_warfare_likelihood": 0.4,
            },
            "climate": {
                "global_temp_anomaly_c": 1.2, "co2_concentration_ppm": 420,
                "sea_level_rise_mm": 100, "extreme_weather_index": 0.6,
                "ice_sheet_loss_gt": 200, "renewable_share": 30,
            },
        }
        results = self.scorer.score_all_domains(domains)
        assert "geopolitical" in results
        assert "climate" in results
        assert results["geopolitical"].overall_score > 0.8

    def test_global_quality_score(self):
        state = {
            "us_china_tension": 0.5, "russia_west_tension": 0.4,
            "middle_east_instability": 0.6, "trade_protectionism": 0.3,
            "energy_geopolitics": 0.5, "cyber_warfare_likelihood": 0.4,
        }
        self.scorer.score_domain("geopolitical", state)
        gq = self.scorer.global_quality_score()
        assert 0 <= gq <= 1.0

    def test_adjust_confidence_interval(self):
        state = {
            "us_china_tension": 0.5, "russia_west_tension": 0.4,
            "middle_east_instability": 0.6, "trade_protectionism": 0.3,
            "energy_geopolitics": 0.5, "cyber_warfare_likelihood": 0.4,
        }
        self.scorer.score_domain("geopolitical", state)
        adj_low, adj_high = self.scorer.adjust_confidence_interval("geopolitical", 0.3, 0.7)
        assert adj_low <= 0.3
        assert adj_high >= 0.7

    def test_adjust_confidence_interval_sparse(self):
        state = {"us_china_tension": 5.0}
        self.scorer.score_domain("geopolitical", state)
        adj_low, adj_high = self.scorer.adjust_confidence_interval("geopolitical", 0.3, 0.7)
        original_width = 0.4
        adjusted_width = adj_high - adj_low
        assert adjusted_width >= original_width

    def test_quality_summary(self):
        state = {
            "us_china_tension": 0.5, "russia_west_tension": 0.4,
            "middle_east_instability": 0.6, "trade_protectionism": 0.3,
            "energy_geopolitics": 0.5, "cyber_warfare_likelihood": 0.4,
        }
        self.scorer.score_domain("geopolitical", state)
        summary = self.scorer.quality_summary()
        assert "global_quality" in summary
        assert "domains" in summary
        assert "recommendations" in summary
        assert summary["n_domains"] >= 1

    def test_economics_scoring(self):
        state = {"gdp_growth": 0.02, "inflation": 0.04, "unemployment": 0.05,
                 "interest_rate": 0.05, "debt_to_gdp": 1.2}
        report = self.scorer.score_domain("economics", state)
        assert report.completeness_score == 1.0
        assert report.accuracy_score == 1.0

    def test_economics_out_of_range(self):
        state = {"gdp_growth": -0.5, "inflation": 0.5, "unemployment": 0.5,
                 "interest_rate": 0.5, "debt_to_gdp": 10.0}
        report = self.scorer.score_domain("economics", state)
        assert report.accuracy_score < 1.0
        assert len(report.suspicious_values) > 0


class TestQualityAdjustedForecaster:
    def setup_method(self):
        self.scorer = DataQualityScorer()

    def test_adjust_prediction_full_quality(self):
        state = {
            "us_china_tension": 0.5, "russia_west_tension": 0.4,
            "middle_east_instability": 0.6, "trade_protectionism": 0.3,
            "energy_geopolitics": 0.5, "cyber_warfare_likelihood": 0.4,
        }
        self.scorer.score_domain("geopolitical", state)
        forecaster = QualityAdjustedForecaster(self.scorer)
        result = forecaster.adjust_prediction("geopolitical", 0.5, 0.3, 0.7)
        assert result["quality_score"] > 0.8
        assert result["confidence_multiplier"] > 0.9

    def test_adjust_prediction_sparse(self):
        state = {"us_china_tension": 5.0}
        self.scorer.score_domain("geopolitical", state)
        forecaster = QualityAdjustedForecaster(self.scorer)
        result = forecaster.adjust_prediction("geopolitical", 0.5, 0.3, 0.7)
        assert result["quality_score"] < 0.5

    def test_adjust_forecast_set(self):
        state = {
            "us_china_tension": 0.5, "russia_west_tension": 0.4,
            "middle_east_instability": 0.6, "trade_protectionism": 0.3,
            "energy_geopolitics": 0.5, "cyber_warfare_likelihood": 0.4,
        }
        self.scorer.score_domain("geopolitical", state)
        forecaster = QualityAdjustedForecaster(self.scorer)
        predictions = [
            {"domain": "geopolitical", "value": 0.5, "confidence_interval": (0.3, 0.7)},
            {"domain": "unknown", "value": 0.4, "confidence_interval": (0.2, 0.6)},
        ]
        adjusted = forecaster.adjust_forecast_set(predictions)
        assert len(adjusted) == 2
        assert "quality_adjusted_ci" in adjusted[0]


class TestDataSourceProvenance:
    def test_provenance_not_stale(self):
        prov = DataSourceProvenance(
            source="yfinance", domain="economics",
            last_updated=datetime.now(), update_frequency_hours=24.0,
            reliability=0.9,
        )
        assert not prov.is_stale
        assert prov.staleness_hours < 1

    def test_provenance_stale(self):
        prov = DataSourceProvenance(
            source="manual", domain="geopolitical",
            last_updated=datetime.now() - timedelta(hours=100),
            update_frequency_hours=24.0, reliability=0.7,
        )
        assert prov.is_stale
        assert prov.staleness_hours > 48

    def test_domain_field_registry_coverage(self):
        assert len(DOMAIN_FIELD_REGISTRY) >= 10
        assert "geopolitical" in DOMAIN_FIELD_REGISTRY
        assert "climate" in DOMAIN_FIELD_REGISTRY
        assert "economics" in DOMAIN_FIELD_REGISTRY

    def test_domain_ranges_valid(self):
        for domain, config in DOMAIN_FIELD_REGISTRY.items():
            ranges = config.get("ranges", {})
            for field, (low, high) in ranges.items():
                assert low < high, f"Domain {domain}, field {field}: range [{low}, {high}] is invalid"