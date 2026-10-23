"""Tests for AI integration module."""
import pytest
from oracle.domains.ai_integration import OllamaBridge, AIInsightEngine, ModelResponse, InsightResult, ForecastCommentary


class TestModelResponse:
    def test_creation(self):
        resp = ModelResponse(text="hello", model="test")
        assert resp.text == "hello"
        assert resp.model == "test"
        assert resp.prompt_tokens == 0
        assert resp.method == "ollama_api"

    def test_defaults(self):
        resp = ModelResponse(text="", model="m")
        assert resp.completion_tokens == 0
        assert resp.total_duration_ms == 0.0


class TestInsightResult:
    def test_creation(self):
        result = InsightResult(
            insights=["test insight"],
            cross_domain_connections=["test connection"],
            confidence_assessment="medium",
            recommendations=["test rec"],
            model_used="test"
        )
        assert len(result.insights) == 1
        assert result.confidence_assessment == "medium"
        assert result.method == "ai_insight"


class TestForecastCommentary:
    def test_creation(self):
        fc = ForecastCommentary(
            summary="test summary",
            key_risks=["risk1"],
            key_opportunities=["opp1"],
            scenario_narrative="test narrative",
            model_used="test"
        )
        assert fc.summary == "test summary"
        assert len(fc.key_risks) == 1
        assert fc.method == "forecast_commentary"


class TestOllamaBridge:
    def test_init_defaults(self):
        bridge = OllamaBridge()
        assert bridge.base_url is None  # No default localhost
        assert bridge.default_model == "glm-5.1:cloud"
        assert bridge.timeout == 120.0

    def test_init_custom(self):
        bridge = OllamaBridge(base_url="http://custom:1234", default_model="llama3")
        assert bridge.base_url == "http://custom:1234"
        assert bridge.default_model == "llama3"

    def test_returns_fallback_when_no_base_url(self):
        bridge = OllamaBridge()
        resp = bridge.generate("test")
        assert resp.method == "ollama_fallback"
        assert "unavailable" in resp.text.lower()


class TestAIInsightEngine:
    def test_init(self):
        engine = AIInsightEngine(model="glm-5.1:cloud")
        assert engine.model == "glm-5.1:cloud"
        assert engine.bridge is not None
        assert engine.bridge.default_model == "glm-5.1:cloud"

    def test_health_check_structure(self):
        engine = AIInsightEngine()
        result = engine.health_check()
        assert "ollama_available" in result
        assert "default_model" in result
        assert "method" in result
        assert result["default_model"] == "glm-5.1:cloud"
        assert result["method"] == "ai_health_check"

    def test_analyze_prediction_returns_model_response(self):
        engine = AIInsightEngine()
        result = engine.analyze_prediction("general", {"test": 1})
        assert isinstance(result, ModelResponse)
        assert result.model is not None

    def test_cross_domain_insights_returns_insight_result(self):
        engine = AIInsightEngine()
        result = engine.cross_domain_insights(["economics", "climate"], {"test": 1})
        assert isinstance(result, InsightResult)
        assert len(result.insights) > 0
        assert len(result.cross_domain_connections) > 0
        assert result.model_used is not None

    def test_forecast_commentary_returns_commentary(self):
        engine = AIInsightEngine()
        result = engine.forecast_commentary("markets", {"trend": "up"})
        assert isinstance(result, ForecastCommentary)
        assert result.summary is not None
        assert len(result.key_risks) > 0
        assert len(result.key_opportunities) > 0