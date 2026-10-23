import pytest
import numpy as np
from oracle.prediction.information_horizon import (
    InformationHorizonMapper,
    CertaintyHorizon,
    DecisionTimingResult,
    QuestionCategory,
    RESOLUTION_PROFILES,
    _certainty_curve,
    _find_time_for_certainty,
)


class TestQuestionCategory:
    def test_all_categories_exist(self):
        expected = [
            "daily_market_move", "quarterly_earnings", "recession_onset",
            "geopolitical_crisis", "climate_tipping", "technology_disruption",
            "pandemic_outcome", "election_outcome", "regulatory_change",
            "housing_market_shift",
        ]
        for name in expected:
            assert QuestionCategory(name) is not None

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError):
            QuestionCategory("nonexistent_category")

    def test_resolution_profiles_match_categories(self):
        for cat in QuestionCategory:
            assert cat in RESOLUTION_PROFILES
            profile = RESOLUTION_PROFILES[cat]
            assert "min_months" in profile
            assert "max_months" in profile
            assert "typical_months" in profile
            assert "base_certainty" in profile
            assert "information_velocity" in profile
            assert "resolution_shape" in profile
            assert profile["min_months"] <= profile["typical_months"] <= profile["max_months"]
            assert 0.0 <= profile["base_certainty"] <= 1.0
            assert 0.0 <= profile["information_velocity"] <= 1.0


class TestCertaintyCurve:
    def test_exponential_shape(self):
        profile = RESOLUTION_PROFILES[QuestionCategory.DAILY_MARKET_MOVE]
        c0 = _certainty_curve(0.0, profile, 1.0)
        c1 = _certainty_curve(0.01, profile, 1.0)
        c100 = _certainty_curve(100.0, profile, 1.0)
        assert 0.0 <= c0 <= 1.0
        assert 0.0 <= c1 <= 1.0
        assert c1 > c0
        assert c100 > c1

    def test_step_shape(self):
        profile = RESOLUTION_PROFILES[QuestionCategory.QUARTERLY_EARNINGS]
        c_before = _certainty_curve(2.5, profile, 1.0)
        c_after = _certainty_curve(3.5, profile, 1.0)
        assert c_after > c_before

    def test_sigmoid_shape(self):
        profile = RESOLUTION_PROFILES[QuestionCategory.RECESSION_ONSET]
        c0 = _certainty_curve(0.0, profile, 1.0)
        c_mid = _certainty_curve(6.0, profile, 1.0)
        c_late = _certainty_curve(24.0, profile, 1.0)
        assert c_mid > c0
        assert c_late > c_mid

    def test_linear_shape(self):
        profile = RESOLUTION_PROFILES[QuestionCategory.CLIMATE_TIPPING]
        c0 = _certainty_curve(0.0, profile, 1.0)
        c100 = _certainty_curve(100.0, profile, 1.0)
        c500 = _certainty_curve(500.0, profile, 1.0)
        assert c100 > c0
        assert c500 > c100

    def test_monotonicity(self):
        for cat in QuestionCategory:
            profile = RESOLUTION_PROFILES[cat]
            prev = _certainty_curve(0.0, profile, 0.7)
            for m in [0.1, 0.5, 1, 3, 6, 12, 24, 60, 120, 600]:
                cur = _certainty_curve(m, profile, 0.7)
                assert cur >= prev - 1e-9, f"Non-monotonic for {cat.value} at month {m}"
                prev = cur

    def test_bounded_0_1(self):
        for cat in QuestionCategory:
            profile = RESOLUTION_PROFILES[cat]
            for m in [0, 0.01, 0.1, 1, 6, 12, 60, 600]:
                for dq in [0.1, 0.5, 1.0]:
                    c = _certainty_curve(m, profile, dq)
                    assert 0.0 <= c <= 1.0, f"Out of bounds: {cat.value} m={m} dq={dq} c={c}"

    def test_data_quality_effect(self):
        profile = RESOLUTION_PROFILES[QuestionCategory.RECESSION_ONSET]
        c_low = _certainty_curve(6.0, profile, 0.2)
        c_high = _certainty_curve(6.0, profile, 0.9)
        assert c_high > c_low


class TestFindTimeForCertainty:
    def test_returns_time_for_base_certainty(self):
        profile = RESOLUTION_PROFILES[QuestionCategory.DAILY_MARKET_MOVE]
        t = _find_time_for_certainty(profile["base_certainty"], profile, 0.7)
        assert t == 0.0

    def test_returns_none_for_impossible_target(self):
        profile = RESOLUTION_PROFILES[QuestionCategory.CLIMATE_TIPPING]
        t = _find_time_for_certainty(0.99, profile, 0.1, max_months=1.0)
        assert t is None

    def test_monotonic_thresholds(self):
        profile = RESOLUTION_PROFILES[QuestionCategory.ELECTION_OUTCOME]
        t50 = _find_time_for_certainty(0.50, profile, 0.7)
        t80 = _find_time_for_certainty(0.80, profile, 0.7)
        t95 = _find_time_for_certainty(0.95, profile, 0.7)
        assert t50 is not None
        assert t80 is not None
        assert t95 is not None
        assert t50 <= t80 <= t95


class TestEstimateResolutionTime:
    def setup_method(self):
        self.mapper = InformationHorizonMapper()

    def test_daily_market_fast(self):
        result = self.mapper.estimate_resolution_time(QuestionCategory.DAILY_MARKET_MOVE)
        assert result["p50_time"] < 1.0
        assert result["p95_time"] < 1.0

    def test_climate_slow(self):
        result = self.mapper.estimate_resolution_time(QuestionCategory.CLIMATE_TIPPING)
        assert result["p50_time"] > 50

    def test_data_quality_speeds_resolution(self):
        r_low = self.mapper.estimate_resolution_time(QuestionCategory.RECESSION_ONSET, 0.2)
        r_high = self.mapper.estimate_resolution_time(QuestionCategory.RECESSION_ONSET, 0.9)
        assert r_high["p50_time"] <= r_low["p50_time"]

    def test_all_keys_present(self):
        result = self.mapper.estimate_resolution_time(QuestionCategory.GEOPOLITICAL_CRISIS)
        for key in ["p50_time", "p80_time", "p95_time", "typical_resolution",
                     "min_resolution", "max_resolution", "information_velocity", "current_certainty"]:
            assert key in result

    def test_data_quality_clipped(self):
        r_neg = self.mapper.estimate_resolution_time(QuestionCategory.RECESSION_ONSET, -1.0)
        r_over = self.mapper.estimate_resolution_time(QuestionCategory.RECESSION_ONSET, 5.0)
        profile = RESOLUTION_PROFILES[QuestionCategory.RECESSION_ONSET]
        expected_clipped = profile["information_velocity"] * 0.01
        assert r_neg["information_velocity"] == pytest.approx(expected_clipped, rel=1e-6)
        expected_capped = profile["information_velocity"] * 1.0
        assert r_over["information_velocity"] == pytest.approx(expected_capped, rel=1e-6)


class TestCertaintyTimeline:
    def setup_method(self):
        self.mapper = InformationHorizonMapper()

    def test_basic_timeline(self):
        uncertainties = [
            {"question": "Will S&P drop tomorrow?", "category": "daily_market_move"},
            {"question": "Will there be a recession?", "category": "recession_onset"},
            {"question": "Will climate tip?", "category": "climate_tipping"},
        ]
        horizons = self.mapper.certainty_timeline(uncertainties)
        assert len(horizons) == 3
        for h in horizons:
            assert isinstance(h, CertaintyHorizon)
            assert h.current_certainty >= 0.0
            assert len(h.resolution_times) > 0
            assert h.information_velocity > 0.0
            assert h.recommendation != ""

    def test_unknown_category_falls_back(self):
        uncertainties = [{"question": "Test", "category": "nonexistent"}]
        horizons = self.mapper.certainty_timeline(uncertainties)
        assert len(horizons) == 1
        assert horizons[0].category == "nonexistent"

    def test_ordered_resolution_times(self):
        uncertainties = [{"question": "Q", "category": "election_outcome"}]
        horizons = self.mapper.certainty_timeline(uncertainties)
        h = horizons[0]
        thresholds = sorted(h.resolution_times.keys())
        times = [h.resolution_times[t] for t in thresholds]
        for i in range(len(times) - 1):
            assert times[i] <= times[i + 1]

    def test_recommendation_varies_by_speed(self):
        uncertainties_fast = [{"question": "Fast", "category": "daily_market_move"}]
        uncertainties_slow = [{"question": "Slow", "category": "climate_tipping"}]
        fast = self.mapper.certainty_timeline(uncertainties_fast)[0]
        slow = self.mapper.certainty_timeline(uncertainties_slow)[0]
        assert fast.recommendation != slow.recommendation

    def test_custom_data_quality(self):
        uncertainties = [
            {"question": "Q", "category": "recession_onset", "data_quality": 1.0},
            {"question": "Q2", "category": "recession_onset", "data_quality": 0.1},
        ]
        horizons = self.mapper.certainty_timeline(uncertainties)
        assert horizons[0].information_velocity > horizons[1].information_velocity

    def test_certainty_at_time(self):
        uncertainties = [{"question": "Q", "category": "election_outcome"}]
        horizons = self.mapper.certainty_timeline(uncertainties)
        h = horizons[0]
        c0 = h.certainty_at_time(0.0)
        c100 = h.certainty_at_time(100.0)
        assert c0 <= c100

    def test_is_resolvable_before(self):
        uncertainties = [{"question": "Q", "category": "daily_market_move"}]
        horizons = self.mapper.certainty_timeline(uncertainties)
        h = horizons[0]
        assert h.is_resolvable_before(1.0)
        assert h.is_resolvable_before(0.001) is False or h.resolution_times.get(0.50, 999) < 0.001

    def test_time_to_threshold(self):
        uncertainties = [{"question": "Q", "category": "election_outcome"}]
        horizons = self.mapper.certainty_timeline(uncertainties)
        h = horizons[0]
        if 0.50 in h.resolution_times:
            assert h.time_to_threshold(0.50) is not None
        assert h.time_to_threshold(0.999) is None or h.time_to_threshold(0.999) is not None
        assert h.time_to_threshold(0.50) is not None


class TestDecisionTimingScore:
    def setup_method(self):
        self.mapper = InformationHorizonMapper()

    def test_basic_decision_timing(self):
        uncertainties = [
            {"question": "Recession?", "category": "recession_onset"},
            {"question": "Election?", "category": "election_outcome"},
        ]
        result = self.mapper.decision_timing_score(12.0, uncertainties)
        assert isinstance(result, DecisionTimingResult)
        assert result.optimal_decision_time >= 0
        assert result.recommendation != ""
        assert len(result.uncertainty_breakdown) == 2

    def test_empty_uncertainties(self):
        result = self.mapper.decision_timing_score(12.0, [])
        assert result.optimal_decision_time == 0.0
        assert "No uncertainties" in result.recommendation

    def test_short_deadline(self):
        uncertainties = [
            {"question": "Climate?", "category": "climate_tipping"},
        ]
        result = self.mapper.decision_timing_score(1.0, uncertainties)
        assert result.optimal_decision_time <= 1.0

    def test_should_wait_method(self):
        uncertainties = [
            {"question": "Election?", "category": "election_outcome"},
        ]
        result = self.mapper.decision_timing_score(6.0, uncertainties)
        assert isinstance(result.should_wait(), bool)

    def test_high_delay_cost_reduces_wait(self):
        uncertainties = [
            {"question": "Recession?", "category": "recession_onset"},
        ]
        result_low = self.mapper.decision_timing_score(12.0, uncertainties, decision_cost_of_delay=0.001)
        result_high = self.mapper.decision_timing_score(12.0, uncertainties, decision_cost_of_delay=0.1)
        assert result_low.optimal_decision_time >= result_high.optimal_decision_time

    def test_resolvable_by_deadline(self):
        uncertainties = [
            {"question": "Market move", "category": "daily_market_move"},
            {"question": "Climate", "category": "climate_tipping"},
        ]
        result = self.mapper.decision_timing_score(6.0, uncertainties)
        for bd in result.uncertainty_breakdown:
            if bd["category"] == "daily_market_move":
                assert bd["resolvable_by_deadline"] is True
            if bd["category"] == "climate_tipping":
                assert bd["resolvable_by_deadline"] is False


class TestValueOfInformation:
    def setup_method(self):
        self.mapper = InformationHorizonMapper()

    def test_basic_voi(self):
        uncertainty = {"question": "Recession?", "category": "recession_onset"}
        context = {"stake": 1000.0, "deadline_months": 12.0}
        result = self.mapper.value_of_information(uncertainty, context)
        assert "value_of_information_full" in result
        assert "recommendation" in result
        assert "current_certainty" in result
        assert "certainty_at_deadline" in result
        assert "cost_ratio" in result

    def test_high_stake_high_voi(self):
        uncertainty = {"question": "Election?", "category": "election_outcome"}
        context_low = {"stake": 100.0, "deadline_months": 6.0}
        context_high = {"stake": 10000.0, "deadline_months": 6.0}
        voi_low = self.mapper.value_of_information(uncertainty, context_low)["value_of_information_full"]
        voi_high = self.mapper.value_of_information(uncertainty, context_high)["value_of_information_full"]
        assert abs(voi_high) >= abs(voi_low) * 0.5

    def test_irreversible_reduces_voi(self):
        uncertainty = {"question": "Reg change?", "category": "regulatory_change"}
        context_rev = {"stake": 1000.0, "reversible": True, "deadline_months": 18.0}
        context_irr = {"stake": 1000.0, "reversible": False, "deadline_months": 18.0}
        voi_rev = self.mapper.value_of_information(uncertainty, context_rev)["value_of_information_full"]
        voi_irr = self.mapper.value_of_information(uncertainty, context_irr)["value_of_information_full"]
        assert abs(voi_irr) < abs(voi_rev) or voi_irr < voi_rev

    def test_more_alternatives_increases_voi(self):
        uncertainty = {"question": "Recession?", "category": "recession_onset"}
        ctx1 = {"stake": 1000.0, "alternatives": 1, "deadline_months": 12.0}
        ctx5 = {"stake": 1000.0, "alternatives": 5, "deadline_months": 12.0}
        voi1 = self.mapper.value_of_information(uncertainty, ctx1)["value_of_information_full"]
        voi5 = self.mapper.value_of_information(uncertainty, ctx5)["value_of_information_full"]
        assert abs(voi5) >= abs(voi1) * 0.8

    def test_all_keys_present(self):
        uncertainty = {"question": "Q", "category": "pandemic_outcome"}
        context = {"stake": 500.0}
        result = self.mapper.value_of_information(uncertainty, context)
        expected_keys = [
            "question", "category", "value_of_information_full", "value_of_information_p50",
            "current_expected_value", "wait_expected_value", "p50_expected_value",
            "current_certainty", "certainty_at_deadline", "cost_of_wrong_decision",
            "delay_cost_total", "cost_ratio", "reversibility_factor", "recommendation",
        ]
        for k in expected_keys:
            assert k in result, f"Missing key: {k}"

    def test_unknown_category_falls_back(self):
        uncertainty = {"question": "Q", "category": "unknown_thing"}
        context = {"stake": 100.0}
        result = self.mapper.value_of_information(uncertainty, context)
        assert result["category"] == "unknown_thing"


class TestBatchValueOfInformation:
    def setup_method(self):
        self.mapper = InformationHorizonMapper()

    def test_batch(self):
        uncertainties = [
            {"question": "Market?", "category": "daily_market_move"},
            {"question": "Recession?", "category": "recession_onset"},
            {"question": "Climate?", "category": "climate_tipping"},
        ]
        context = {"stake": 1000.0, "deadline_months": 12.0}
        result = self.mapper.batch_value_of_information(uncertainties, context)
        assert result["n_uncertainties"] == 3
        assert result["total_voi"] != 0
        assert result["average_voi"] != 0
        assert result["best_voi_question"] != ""
        assert len(result["prioritized"]) == 3
        assert len(result["individual_results"]) == 3

    def test_resolution_fraction(self):
        uncertainties = [
            {"question": "Market?", "category": "daily_market_move"},
            {"question": "Climate?", "category": "climate_tipping"},
        ]
        context = {"stake": 500.0, "deadline_months": 6.0}
        result = self.mapper.batch_value_of_information(uncertainties, context)
        assert 0.0 <= result["resolution_fraction"] <= 1.0
        assert result["n_resolved_by_deadline"] >= 0


class TestCustomProfiles:
    def test_custom_profile_override(self):
        custom = {
            QuestionCategory.RECESSION_ONSET: {
                "typical_months": 24.0,
                "information_velocity": 0.60,
            }
        }
        mapper = InformationHorizonMapper(custom_profiles=custom)
        result = mapper.estimate_resolution_time(QuestionCategory.RECESSION_ONSET, 0.7)
        assert result["typical_resolution"] == 24.0

    def test_default_data_quality(self):
        mapper = InformationHorizonMapper(default_data_quality=0.9)
        result = mapper.estimate_resolution_time(QuestionCategory.ELECTION_OUTCOME)
        profile = RESOLUTION_PROFILES[QuestionCategory.ELECTION_OUTCOME]
        assert result["information_velocity"] == pytest.approx(profile["information_velocity"] * 0.9, rel=1e-6)


class TestEdgeCases:
    def setup_method(self):
        self.mapper = InformationHorizonMapper()

    def test_zero_data_quality(self):
        result = self.mapper.estimate_resolution_time(QuestionCategory.RECESSION_ONSET, 0.0)
        profile = RESOLUTION_PROFILES[QuestionCategory.RECESSION_ONSET]
        expected = profile["information_velocity"] * 0.01
        assert result["information_velocity"] == pytest.approx(expected, rel=1e-6)

    def test_very_short_deadline(self):
        result = self.mapper.decision_timing_score(0.01, [
            {"question": "Q", "category": "recession_onset"}
        ])
        assert result.optimal_decision_time <= 0.01 + 1e-6

    def test_single_uncertainty(self):
        result = self.mapper.certainty_timeline([
            {"question": "Single", "category": "pandemic_outcome"}
        ])
        assert len(result) == 1

    def test_many_uncertainties(self):
        uncertainties = [
            {"question": f"Q{i}", "category": cat.value}
            for i, cat in enumerate(QuestionCategory)
        ]
        result = self.mapper.certainty_timeline(uncertainties)
        assert len(result) == len(QuestionCategory)

    def test_cache_populated(self):
        self.mapper.certainty_timeline([{"question": "cached_q", "category": "election_outcome"}])
        assert "cached_q" in self.mapper._cache