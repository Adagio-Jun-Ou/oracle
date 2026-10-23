import pytest
from oracle.domains.prescience import PrescienceEngine, EVENT_TEMPLATES, TIPPING_POINTS, EARLY_WARNING_SIGNALS


@pytest.fixture
def engine():
    return PrescienceEngine(seed=42)


def test_event_templates_structure():
    assert len(EVENT_TEMPLATES) >= 10
    for eid, tmpl in EVENT_TEMPLATES.items():
        assert "name" in tmpl
        assert "base_rate_per_year" in tmpl
        assert "domains" in tmpl
        assert "severity_range" in tmpl
        assert "cascade_prob" in tmpl
        assert "recovery_years" in tmpl


def test_tipping_points_structure():
    assert len(TIPPING_POINTS) >= 10
    for tp in TIPPING_POINTS:
        assert "name" in tp
        assert "impact" in tp
        assert "irreversibility" in tp
        assert "domains" in tp


def test_early_warning_signals_structure():
    assert len(EARLY_WARNING_SIGNALS) >= 5
    for category, signals in EARLY_WARNING_SIGNALS.items():
        for sig in signals:
            assert "indicator" in sig
            assert "description" in sig
            assert "lead_time_months" in sig
            assert "event" in sig


def test_forecast_events_default(engine):
    result = engine.forecast_events({}, years=10, n_simulations=100)
    assert "n_events_forecast" in result
    assert "events" in result
    assert "any_catastrophic_probability" in result
    assert result["n_events_forecast"] > 0


def test_forecast_events_with_domain_scores(engine):
    scores = {"seismic": 0.3, "infrastructure": 0.4, "economic": 0.5}
    result = engine.forecast_events(scores, years=10, n_simulations=100)
    assert result["n_events_forecast"] > 0
    for event in result["events"]:
        assert event["annual_probability"] > 0
        assert event["severity"] > 0


def test_forecast_events_period_probability(engine):
    result = engine.forecast_events({}, years=10, n_simulations=100)
    for event in result["events"]:
        assert event["period_probability"] >= event["annual_probability"]
        assert event["period_probability"] <= 1.0


def test_forecast_events_top5_risks(engine):
    result = engine.forecast_events({}, years=10, n_simulations=100)
    assert "top_5_risks" in result
    assert len(result["top_5_risks"]) <= 5


def test_assess_tipping_points_default(engine):
    result = engine.assess_tipping_points({}, current_temp_anomaly=1.2)
    assert "n_tipping_points" in result
    assert "tipping_points" in result
    assert result["n_tipping_points"] >= 10
    for tp in result["tipping_points"]:
        assert "proximity" in tp
        assert "status" in tp
        assert tp["proximity"] >= 0
        assert tp["proximity"] <= 1


def test_assess_tipping_points_high_temp(engine):
    result = engine.assess_tipping_points({}, current_temp_anomaly=2.5)
    near_count = sum(1 for tp in result["tipping_points"] if tp["proximity"] > 0.7)
    assert near_count >= 3


def test_assess_tipping_points_irreversibility(engine):
    result = engine.assess_tipping_points({}, current_temp_anomaly=1.2)
    assert result["irreversibility_avg"] > 0
    assert result["irreversibility_avg"] < 1


def test_generate_early_warnings_default(engine):
    result = engine.generate_early_warnings({})
    assert "n_active_warnings" in result
    assert "warnings" in result
    assert "overall_alert_level" in result


def test_generate_early_warnings_triggered(engine):
    scores = {"us_china_tension": 0.1, "trade_protectionism": 0.1}
    result = engine.generate_early_warnings(scores)
    assert result["n_active_warnings"] > 0


def test_generate_early_warnings_severity_levels(engine):
    result = engine.generate_early_warnings({})
    for w in result["warnings"]:
        assert w["severity"] in ("critical", "high", "elevated")


def test_generate_scenario_great_filter(engine):
    result = engine.generate_scenario("great_filter", years=50, n_simulations=100)
    assert result["scenario"] == "great_filter"
    assert "timeline" in result
    assert 0 <= result["mean_final_stability"] <= 1
    assert 0 <= result["collapse_probability"] <= 1


def test_generate_scenario_slow_collapse(engine):
    result = engine.generate_scenario("slow_collapse", years=50, n_simulations=100)
    assert result["scenario_name"] == "Slow Collapse"


def test_generate_scenario_technological_disruption(engine):
    result = engine.generate_scenario("technological_disruption", years=50, n_simulations=100)
    assert "trigger_events" in result
    assert "ai_disruption" in result["trigger_events"]


def test_generate_scenario_climate_cascade(engine):
    result = engine.generate_scenario("climate_cascade", years=50, n_simulations=100)
    assert result["adjusted_severity"] > 0


def test_generate_scenario_geopolitical_spiral(engine):
    result = engine.generate_scenario("geopolitical_spiral", years=50, n_simulations=100)
    assert "cascade_probability" in result


def test_generate_scenario_domain_modifier(engine):
    scores = {"geopolitical": 0.2}
    result_high_risk = engine.generate_scenario("geopolitical_spiral", domain_scores=scores, years=50, n_simulations=100)
    scores_safe = {"geopolitical": 0.9}
    result_safe = engine.generate_scenario("geopolitical_spiral", domain_scores=scores_safe, years=50, n_simulations=100)
    assert result_high_risk["adjusted_severity"] >= result_safe["adjusted_severity"]


def test_predict_prescience_default(engine):
    result = engine.predict_prescience()
    assert "prescience_score" in result
    assert "outlook" in result
    assert "any_catastrophic_probability" in result
    assert result["outlook"] in ("stable", "cautious", "elevated_risk", "high_alert", "critical")


def test_predict_prescience_with_scores(engine):
    scores = {"seismic": 0.2, "infrastructure": 0.3, "economic": 0.4}
    result = engine.predict_prescience(scores, current_temp_anomaly=1.5)
    assert result["prescience_score"] > 0
    assert "recommendations" in result


def test_predict_prescience_all_good_scores(engine):
    scores = {d: 0.9 for d in ["seismic", "infrastructure", "economic", "social", "political", "pandemic", "climate", "biodiversity"]}
    result = engine.predict_prescience(scores, current_temp_anomaly=0.5)
    assert result["prescience_score"] < 0.95


def test_predict_prescience_all_bad_scores(engine):
    scores = {d: 0.1 for d in ["seismic", "infrastructure", "economic", "social", "political", "pandemic", "climate", "biodiversity"]}
    result = engine.predict_prescience(scores, current_temp_anomaly=2.5)
    assert result["prescience_score"] > 0.4


def test_tipping_point_recommendations(engine):
    result = engine.assess_tipping_points({}, current_temp_anomaly=2.0)
    assert "recommendations" in result
    assert len(result["recommendations"]) > 0


def test_prescience_recommendations(engine):
    result = engine.predict_prescience({}, current_temp_anomaly=1.2)
    assert "recommendations" in result
    assert len(result["recommendations"]) > 0


def test_compounding_events(engine):
    result = engine.forecast_events({}, years=10, n_simulations=100)
    events_with_compound = [e for e in result["events"] if e.get("compounding_events")]
    assert len(events_with_compound) > 0


def test_cascade_probability(engine):
    result = engine.generate_scenario("great_filter", years=50, n_simulations=100)
    assert 0 <= result["cascade_probability"] <= 1


def test_scenario_timeline(engine):
    result = engine.generate_scenario("great_filter", years=50, n_simulations=100)
    assert len(result["timeline"]) > 0
    for point in result["timeline"]:
        assert "year" in point
        assert "mean_stability" in point
        assert "p10" in point
        assert "p90" in point


def test_forecast_min_probability_filter(engine):
    result = engine.forecast_events({}, years=10, n_simulations=100, min_probability=0.01)
    for event in result["events"]:
        assert event["annual_probability"] >= 0.01


def test_early_warning_alert_levels(engine):
    scores = {"us_china_tension": 0.05}
    result = engine.generate_early_warnings(scores)
    assert result["overall_alert_level"] in ("green", "yellow", "orange", "red")