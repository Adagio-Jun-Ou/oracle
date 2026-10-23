"""Tests for Forecast Decomposer and Scenario Probability Mapper."""
import pytest
import numpy as np
from oracle.prediction.forecast_decomposition import (
    ForecastDecomposer, ForecastDecompositionResult, DecompositionMethod,
    DecompositionComponent,
)
from oracle.prediction.scenario_probability import (
    ScenarioProbabilityMapper, ScenarioProbabilityResult, ScenarioNode,
    ScenarioRelation, ScenarioPath, DEFAULT_SCENARIOS,
)


SAMPLE = {"gdp_growth": 0.5, "inflation": 0.3, "stock_market": 0.7}


class TestForecastDecomposer:
    def test_initialization(self):
        fd = ForecastDecomposer()
        assert fd.n_seasonal_periods == 4
        assert fd.n_shapley_samples == 100

    def test_decompose_additive(self):
        fd = ForecastDecomposer()
        result = fd.decompose(SAMPLE, method=DecompositionMethod.ADDITIVE)
        assert isinstance(result, ForecastDecompositionResult)
        assert result.method == DecompositionMethod.ADDITIVE
        assert set(result.trend.keys()) == set(SAMPLE.keys())

    def test_decompose_multiplicative(self):
        fd = ForecastDecomposer()
        result = fd.decompose(SAMPLE, method=DecompositionMethod.MULTIPLICATIVE)
        assert result.method == DecompositionMethod.MULTIPLICATIVE

    def test_decompose_shapley(self):
        fd = ForecastDecomposer()
        result = fd.decompose(SAMPLE, method=DecompositionMethod.SHAPLEY)
        assert result.method == DecompositionMethod.SHAPLEY

    def test_trend_extraction(self):
        fd = ForecastDecomposer()
        history = {"gdp": [0.1, 0.2, 0.3, 0.4, 0.5]}
        trend = fd._extract_trend({"gdp": 0.5}, history)
        assert "gdp" in trend

    def test_seasonal_extraction(self):
        fd = ForecastDecomposer(n_seasonal_periods=4)
        history = {"x": [1.0, 1.5, 1.0, 0.5] * 3}
        seasonal = fd._extract_seasonal({"x": 1.0}, history)
        assert "x" in seasonal

    def test_cyclical_extraction(self):
        fd = ForecastDecomposer()
        history = {"x": [0.1, 0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3]}
        cyclical = fd._extract_cyclical({"x": 0.5}, history)
        assert "x" in cyclical

    def test_residual_additive(self):
        fd = ForecastDecomposer()
        preds = {"x": 1.0}
        trend = {"x": 0.7}
        seasonal = {"x": 0.1}
        cyclical = {"x": 0.1}
        residual = fd._compute_residual_additive(preds, trend, seasonal, cyclical)
        assert abs(residual["x"] - 0.1) < 1e-10

    def test_shapley_attribution(self):
        fd = ForecastDecomposer()
        features = {"a": 0.5, "b": 0.3, "c": 0.2}
        result = fd._shapley_attribution({"a": 0.5, "b": 0.3, "c": 0.2}, features)
        assert abs(sum(result.values()) - 1.0) < 0.1

    def test_regime_decomposition(self):
        fd = ForecastDecomposer()
        regime = fd._regime_decomposition({"x": 0.8, "y": 0.9})
        assert sum(regime.values()) == 1.0

    def test_explained_variance(self):
        fd = ForecastDecomposer()
        result = fd.decompose(SAMPLE)
        assert 0 <= result.total_explained_variance <= 1

    def test_feature_importance(self):
        fd = ForecastDecomposer()
        result = fd.decompose(SAMPLE)
        assert len(result.feature_importance) > 0

    def test_narrative(self):
        fd = ForecastDecomposer()
        result = fd.decompose(SAMPLE)
        assert len(result.narrative) > 0

    def test_components_structure(self):
        fd = ForecastDecomposer()
        result = fd.decompose(SAMPLE)
        for var in SAMPLE:
            assert var in result.components
            assert len(result.components[var]) == 4
            names = [c.name for c in result.components[var]]
            assert "trend" in names
            assert "seasonal" in names
            assert "cyclical" in names
            assert "residual" in names

    def test_empty_predictions(self):
        fd = ForecastDecomposer()
        result = fd.decompose({})
        assert isinstance(result, ForecastDecompositionResult)

    def test_custom_history(self):
        fd = ForecastDecomposer()
        history = {"gdp_growth": list(np.linspace(0.3, 0.6, 12)),
                   "inflation": list(np.linspace(0.2, 0.4, 12)),
                   "stock_market": list(np.linspace(0.5, 0.8, 12))}
        result = fd.decompose(SAMPLE, history=history)
        assert result.total_explained_variance >= 0

    def test_decomposition_method_enum(self):
        assert DecompositionMethod.ADDITIVE.value == "additive"
        assert DecompositionMethod.MULTIPLICATIVE.value == "multiplicative"
        assert DecompositionMethod.SHAPLEY.value == "shapley"


class TestScenarioProbabilityMapper:
    def test_initialization(self):
        mapper = ScenarioProbabilityMapper()
        assert mapper.n_sensitivity_samples == 100

    def test_map_scenarios_default(self):
        mapper = ScenarioProbabilityMapper(n_sensitivity_samples=10)
        result = mapper.map_scenarios(SAMPLE)
        assert isinstance(result, ScenarioProbabilityResult)
        assert len(result.scenarios) == 5
        assert result.top_scenario != ""

    def test_map_scenarios_custom(self):
        mapper = ScenarioProbabilityMapper(n_sensitivity_samples=10)
        custom = [
            {"id": "s1", "name": "Good", "description": "Good times", "probability": 0.6, "impact": 0.8},
            {"id": "s2", "name": "Bad", "description": "Bad times", "probability": 0.4, "impact": 0.2},
        ]
        result = mapper.map_scenarios(SAMPLE, custom_scenarios=custom)
        assert len(result.scenarios) == 2

    def test_scenario_paths(self):
        mapper = ScenarioProbabilityMapper(n_sensitivity_samples=10)
        result = mapper.map_scenarios(SAMPLE)
        assert len(result.paths) >= 5

    def test_sensitivity_analysis(self):
        mapper = ScenarioProbabilityMapper(n_sensitivity_samples=10)
        result = mapper.map_scenarios(SAMPLE)
        assert len(result.sensitivity) == len(SAMPLE)

    def test_probability_distribution(self):
        mapper = ScenarioProbabilityMapper(n_sensitivity_samples=10)
        result = mapper.map_scenarios(SAMPLE)
        assert len(result.probability_distribution) > 0

    def test_recommendations(self):
        mapper = ScenarioProbabilityMapper(n_sensitivity_samples=10)
        result = mapper.map_scenarios(SAMPLE)
        assert len(result.recommendations) > 0

    def test_mutually_exclusive(self):
        mapper = ScenarioProbabilityMapper(n_sensitivity_samples=10)
        relations = {("s1", "s2"): ScenarioRelation.MUTUALLY_EXCLUSIVE}
        custom = [
            {"id": "s1", "name": "A", "description": "A", "probability": 0.5, "impact": 0.8},
            {"id": "s2", "name": "B", "description": "B", "probability": 0.4, "impact": 0.3},
        ]
        result = mapper.map_scenarios(SAMPLE, custom_scenarios=custom, relations=relations)
        assert isinstance(result, ScenarioProbabilityResult)

    def test_reinforcing(self):
        mapper = ScenarioProbabilityMapper(n_sensitivity_samples=10)
        relations = {("s1", "s2"): ScenarioRelation.REINFORCING}
        custom = [
            {"id": "s1", "name": "A", "description": "A", "probability": 0.3, "impact": 0.7},
            {"id": "s2", "name": "B", "description": "B", "probability": 0.2, "impact": 0.5},
        ]
        result = mapper.map_scenarios(SAMPLE, custom_scenarios=custom, relations=relations)
        assert isinstance(result, ScenarioProbabilityResult)

    def test_opposing(self):
        mapper = ScenarioProbabilityMapper(n_sensitivity_samples=10)
        relations = {("s1", "s2"): ScenarioRelation.OPPOSING}
        custom = [
            {"id": "s1", "name": "A", "description": "A", "probability": 0.4, "impact": 0.7},
            {"id": "s2", "name": "B", "description": "B", "probability": 0.3, "impact": 0.5},
        ]
        result = mapper.map_scenarios(SAMPLE, custom_scenarios=custom, relations=relations)
        assert isinstance(result, ScenarioProbabilityResult)

    def test_conditional(self):
        mapper = ScenarioProbabilityMapper(n_sensitivity_samples=10)
        relations = {("base_case", "upside"): ScenarioRelation.CONDITIONAL}
        result = mapper.map_scenarios(SAMPLE, relations=relations)
        conditional_paths = [p for p in result.paths if len(p.nodes) == 2]
        assert len(conditional_paths) > 0

    def test_expected_impact(self):
        mapper = ScenarioProbabilityMapper(n_sensitivity_samples=10)
        result = mapper.map_scenarios(SAMPLE)
        assert result.expected_impact >= 0

    def test_empty_predictions(self):
        mapper = ScenarioProbabilityMapper(n_sensitivity_samples=10)
        result = mapper.map_scenarios({})
        assert isinstance(result, ScenarioProbabilityResult)

    def test_default_scenarios_constant(self):
        assert len(DEFAULT_SCENARIOS) == 5
        ids = [s["id"] for s in DEFAULT_SCENARIOS]
        assert "base_case" in ids
        assert "crisis" in ids
        assert "breakout" in ids

    def test_scenario_relation_enum(self):
        assert ScenarioRelation.INDEPENDENT.value == "independent"
        assert ScenarioRelation.MUTUALLY_EXCLUSIVE.value == "mutually_exclusive"
        assert ScenarioRelation.REINFORCING.value == "reinforcing"
        assert ScenarioRelation.OPPOSING.value == "opposing"
        assert ScenarioRelation.CONDITIONAL.value == "conditional"

    def test_scenario_node_dataclass(self):
        node = ScenarioNode(id="test", name="Test", description="Test scenario",
                             probability=0.5, impact=0.7)
        assert node.probability == 0.5
        assert node.impact == 0.7
        assert node.children == []
        assert node.parent is None