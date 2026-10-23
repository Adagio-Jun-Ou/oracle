"""AI Model Integration module for Oracle.

Connects to large language models (GLM-5.1:cloud, Kimi-K2, etc.) via Ollama
for intelligent reasoning, explanation, and cross-domain synthesis.

Provides:
- Ollama API bridge for local/cloud models
- Structured prompt engineering for prediction analysis
- Cross-domain insight synthesis
- Model comparison and ensemble reasoning

Environment variables:
- ORACLE_LLM_BASE_URL: Ollama API base URL (falls back to local if omitted)
- ORACLE_LLM_API_KEY: API key for cloud endpoints
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ms: float = 0.0
    method: str = "ollama_api"


@dataclass
class InsightResult:
    insights: List[str]
    cross_domain_connections: List[str]
    confidence_assessment: str
    recommendations: List[str]
    model_used: str
    method: str = "ai_insight"


@dataclass
class ForecastCommentary:
    summary: str
    key_risks: List[str]
    key_opportunities: List[str]
    scenario_narrative: str
    model_used: str
    method: str = "forecast_commentary"


class OllamaBridge:
    """Bridge to Ollama API for LLM inference.

    Supports models like glm-5.1:cloud, and any Ollama-compatible model.
    Handles connection errors gracefully with fallback responses.
    """

    def __init__(self, base_url: Optional[str] = None,
                 default_model: str = "glm-5.1:cloud",
                 api_key: Optional[str] = None,
                 timeout: float = 120.0,
                 temperature: float = 0.3,
                 max_tokens: int = 2048):
        env_url = os.getenv("ORACLE_LLM_BASE_URL", "")
        self.base_url = (base_url or env_url).rstrip("/") if (base_url or env_url) else None
        self.default_model = default_model
        self.api_key = api_key or os.getenv("ORACLE_LLM_API_KEY", "")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._available_models: Optional[List[str]] = None

    def _make_request(self, endpoint: str, payload: dict) -> Optional[dict]:
        if not self.base_url:
            logger.debug("No LLM base_url configured — skipping request")
            return None
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            logger.debug(f"LLM request failed: {e}")
            return None

    def list_models(self) -> List[str]:
        try:
            result = self._make_request("/api/tags", {})
            if result and "models" in result:
                self._available_models = [m["name"] for m in result["models"]]
                return self._available_models
        except Exception:
            pass
        return []

    def generate(self, prompt: str, model: Optional[str] = None,
                  system: Optional[str] = None, temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None) -> ModelResponse:
        model = model or self.default_model
        temp = temperature or self.temperature
        max_tok = max_tokens or self.max_tokens
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tok,
            }
        }
        if system:
            payload["system"] = system
        start_time = time.time()
        result = self._make_request("/api/generate", payload)
        duration_ms = (time.time() - start_time) * 1000
        if result is None:
            return ModelResponse(
                text=f"[Model {model} unavailable - Ollama not responding]",
                model=model,
                total_duration_ms=duration_ms,
                method="ollama_fallback",
            )
        return ModelResponse(
            text=result.get("response", ""),
            model=result.get("model", model),
            prompt_tokens=result.get("prompt_eval_count", 0),
            completion_tokens=result.get("eval_count", 0),
            total_duration_ms=duration_ms,
        )

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
             temperature: Optional[float] = None) -> ModelResponse:
        model = model or self.default_model
        temp = temperature or self.temperature
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temp},
        }
        start_time = time.time()
        result = self._make_request("/api/chat", payload)
        duration_ms = (time.time() - start_time) * 1000
        if result is None:
            return ModelResponse(
                text=f"[Model {model} unavailable]",
                model=model,
                total_duration_ms=duration_ms,
                method="ollama_fallback",
            )
        content = ""
        if "message" in result:
            content = result["message"].get("content", "")
        return ModelResponse(
            text=content,
            model=result.get("model", model),
            prompt_tokens=result.get("prompt_eval_count", 0),
            completion_tokens=result.get("eval_count", 0),
            total_duration_ms=duration_ms,
        )


PREDICTION_SYSTEM_PROMPT = """You are an oracle prediction engine analyst. You provide concise, data-driven insights about forecasts, risks, and cross-domain connections. 

Rules:
- Be specific and quantitative when possible
- Identify 2-3 key risks and 2-3 opportunities
- Explain cross-domain connections clearly
- Keep responses under 300 words
- Use structured format with headers"""

INSIGHT_SYSTEM_PROMPT = """You are a cross-domain insight synthesizer for a prediction engine. You identify non-obvious connections between different domains (economics, geopolitics, climate, health, technology, social dynamics).

Rules:
- Identify 3-5 specific cross-domain interactions
- Rate confidence as high/medium/low
- Provide actionable recommendations
- Keep responses under 400 words"""

COMMENTARY_SYSTEM_PROMPT = """You are an expert forecast commentator. Given prediction data, provide narrative commentary explaining what the numbers mean in plain language.

Rules:
- Start with a one-sentence summary
- List key risks and opportunities
- Write a brief scenario narrative
- Keep under 250 words"""


class AIInsightEngine:
    """Uses LLM to generate insights, cross-domain analysis, and commentary for Oracle predictions."""

    def __init__(self, model: str = "glm-5.1:cloud"):
        self.bridge = OllamaBridge(default_model=model)
        self.model = model

    def analyze_prediction(self, domain: str, prediction_data: dict) -> ModelResponse:
        prompt = f"""Analyze this {domain} prediction data and provide insights:

{json.dumps(prediction_data, indent=2, default=str)[:2000]}

Provide:
1. Key insight (1-2 sentences)
2. Risks (2-3 items)
3. Opportunities (2-3 items) 
4. Cross-domain effects (1-2)
5. Confidence assessment (high/medium/low with reasoning)"""
        return self.bridge.generate(prompt, system=PREDICTION_SYSTEM_PROMPT)

    def cross_domain_insights(self, domains: List[str], states: dict) -> InsightResult:
        prompt = f"""Given the current states of these domains, identify non-obvious cross-domain interactions:

Domains: {', '.join(domains)}

Current states:
{json.dumps(states, indent=2, default=str)[:2000]}

Identify:
1. The 5 most important cross-domain connections
2. How they could amplify or dampen each other
3. Which interactions are most likely to produce unexpected outcomes"""
        response = self.bridge.generate(prompt, system=INSIGHT_SYSTEM_PROMPT)
        lines = response.text.split("\n") if response.text else []
        insights = [l.strip("- •* ") for l in lines if l.strip() and len(l.strip()) > 10][:5]
        connections = [l.strip("- •* ") for l in lines if "cross" in l.lower() or "domain" in l.lower() or "interact" in l.lower()][:5]
        if not insights:
            insights = ["Cross-domain analysis generated by AI model"]
        if not connections:
            connections = ["Domain interactions require further analysis"]
        return InsightResult(
            insights=insights,
            cross_domain_connections=connections,
            confidence_assessment="medium" if response.method != "ollama_fallback" else "low",
            recommendations=["Monitor identified cross-domain interactions", "Update models with new data"],
            model_used=response.model,
        )

    def forecast_commentary(self, domain: str, forecast: dict) -> ForecastCommentary:
        prompt = f"""Provide plain-language commentary for this {domain} forecast:

{json.dumps(forecast, indent=2, default=str)[:1500]}

Provide:
1. Summary (one sentence)
2. Key risks (2-3)
3. Key opportunities (2-3)
4. Scenario narrative (2-3 sentences)"""
        response = self.bridge.generate(prompt, system=COMMENTARY_SYSTEM_PROMPT, max_tokens=512)
        text = response.text if response.text else f"[Commentary for {domain} forecast unavailable]"
        lines = text.split("\n")
        risks = [l.strip("- •* ") for l in lines if "risk" in l.lower() or "danger" in l.lower() or "threat" in l.lower()][:3]
        opportunities = [l.strip("- •* ") for l in lines if "opportun" in l.lower() or "benefit" in l.lower() or "upside" in l.lower()][:3]
        if not risks:
            risks = [f"Key risk in {domain} forecast"]
        if not opportunities:
            opportunities = [f"Key opportunity in {domain} forecast"]
        summary = lines[0].strip() if lines else f"{domain} forecast commentary"
        return ForecastCommentary(
            summary=summary[:200],
            key_risks=risks,
            key_opportunities=opportunities,
            scenario_narrative=text[:300],
            model_used=response.model,
        )

    def compare_models(self, prompt: str, models: List[str] = None) -> Dict[str, ModelResponse]:
        models = models or ["glm-5.1:cloud"]
        results = {}
        for model in models:
            response = self.bridge.generate(prompt, model=model)
            results[model] = response
        return results

    def health_check(self) -> dict:
        models = self.bridge.list_models()
        return {
            "ollama_available": len(models) > 0,
            "models": models,
            "default_model": self.model,
            "method": "ai_health_check",
        }