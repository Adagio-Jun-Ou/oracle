import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from oracle.core.types import Domain, Prediction, ConfidenceLevel


@dataclass
class SentimentData:
    fear_greed_index: float = 0.5
    consumer_confidence: float = 0.5
    social_media_sentiment: float = 0.5
    news_sentiment: float = 0.5
    put_call_ratio: float = 0.7
    vix_level: float = 0.3
    market_breadth: float = 0.5
    institutional_flow: float = 0.0
    retail_flow: float = 0.0
    short_interest: float = 0.3
    safe_haven_demand: float = 0.4
    credit_spread: float = 0.3
    timestamp: date = field(default_factory=date.today)


SENTIMENT_INDICATORS = {
    "fear_greed": {"weight": 0.20, "contrarian": True},
    "consumer_confidence": {"weight": 0.15, "contrarian": False},
    "social_media": {"weight": 0.10, "contrarian": True},
    "news_flow": {"weight": 0.12, "contrarian": True},
    "put_call": {"weight": 0.10, "contrarian": True},
    "vix": {"weight": 0.10, "contrarian": True},
    "market_breadth": {"weight": 0.08, "contrarian": False},
    "institutional": {"weight": 0.08, "contrarian": False},
    "short_interest": {"weight": 0.04, "contrarian": True},
    "safe_haven": {"weight": 0.03, "contrarian": True},
}


class SentimentEngine:
    def __init__(self, lookback: int = 60, decay_rate: float = 0.05, seed: int = 42):
        self.lookback = lookback
        self.decay_rate = decay_rate
        self.rng = np.random.default_rng(seed)
        self.indicators = SENTIMENT_INDICATORS
        self._history: list[SentimentData] = []

    def update(self, data: SentimentData) -> "SentimentEngine":
        self._history.append(data)
        if len(self._history) > self.lookback * 2:
            self._history = self._history[-self.lookback:]
        return self

    def compute_composite_index(self, data: Optional[SentimentData] = None) -> dict:
        d = data or (self._history[-1] if self._history else SentimentData())

        raw_scores = {
            "fear_greed": d.fear_greed_index,
            "consumer_confidence": d.consumer_confidence,
            "social_media": d.social_media_sentiment,
            "news_flow": d.news_sentiment,
            "put_call": 1 - d.put_call_ratio,
            "vix": 1 - d.vix_level,
            "market_breadth": d.market_breadth,
            "institutional": (d.institutional_flow + 1) / 2,
            "short_interest": 1 - d.short_interest,
            "safe_haven": 1 - d.safe_haven_demand,
        }

        composite = 0.0
        total_weight = 0.0
        for name, config in self.indicators.items():
            score = raw_scores.get(name, 0.5)
            if config["contrarian"]:
                score = 1 - score
            weight = config["weight"]
            composite += score * weight
            total_weight += weight

        composite = composite / total_weight if total_weight > 0 else 0.5

        momentum = self._compute_momentum()
        divergence = self._compute_divergence(d)

        return {
            "composite_index": float(composite),
            "raw_scores": {k: round(v, 4) for k, v in raw_scores.items()},
            "interpretation": "extreme_fear" if composite < 0.2 else
                             "fear" if composite < 0.35 else
                             "neutral" if composite < 0.65 else
                             "greed" if composite < 0.8 else "extreme_greed",
            "momentum": float(momentum),
            "divergence": float(divergence),
            "signal": "buy" if composite < 0.3 and momentum > 0 else
                      "strong_buy" if composite < 0.2 and momentum > 0.05 else
                      "sell" if composite > 0.7 and momentum < 0 else
                      "strong_sell" if composite > 0.8 and momentum < -0.05 else
                      "hold",
        }

    def predict_sentiment(
        self, data: SentimentData, days_ahead: int = 30, n_simulations: int = 1000,
    ) -> dict:
        composite = self.compute_composite_index(data)
        current_value = composite["composite_index"]

        mean_reversion_strength = 0.02
        target = 0.5
        volatility = 0.015

        paths = np.zeros((n_simulations, days_ahead + 1))
        paths[:, 0] = current_value

        for t in range(1, days_ahead + 1):
            reversion = mean_reversion_strength * (target - paths[:, t - 1])
            noise = self.rng.normal(0, volatility, n_simulations)
            sentiment_shock = self.rng.normal(0, 0.005, n_simulations) * (1 + 2 * abs(current_value - 0.5))
            paths[:, t] = np.clip(paths[:, t - 1] + reversion + noise + sentiment_shock, 0, 1)

        return {
            "current": current_value,
            "interpretation": composite["interpretation"],
            "signal": composite["signal"],
            "forecasts": {
                "p05": float(np.percentile(paths[:, -1], 5)),
                "p25": float(np.percentile(paths[:, -1], 25)),
                "p50": float(np.percentile(paths[:, -1], 50)),
                "p75": float(np.percentile(paths[:, -1], 75)),
                "p95": float(np.percentile(paths[:, -1], 95)),
            },
            "prob_bullish": float(np.mean(paths[:, -1] > 0.6)),
            "prob_bearish": float(np.mean(paths[:, -1] < 0.4)),
            "mean_reversion_expected": float(abs(current_value - 0.5) > 0.15),
        }

    def detect_contrarian_signals(self, data: SentimentData) -> dict:
        composite = self.compute_composite_index(data)
        idx = composite["composite_index"]
        signals = []

        if idx < 0.25 and data.vix_level > 0.5:
            signals.append({"signal": "strong_buy_contrarian", "reason": "Extreme fear + high VIX", "strength": 0.8})
        if idx < 0.35 and data.put_call_ratio > 1.0:
            signals.append({"signal": "buy_contrarian", "reason": "Fear + high put/call", "strength": 0.6})
        if idx > 0.75 and data.short_interest < 0.2:
            signals.append({"signal": "strong_sell_contrarian", "reason": "Extreme greed + low short interest", "strength": 0.7})
        if idx > 0.65 and data.safe_haven_demand < 0.2:
            signals.append({"signal": "sell_contrarian", "reason": "Greed + low safe haven demand", "strength": 0.5})
        if data.institutional_flow > 0.3 and idx < 0.5:
            signals.append({"signal": "buy_follow_smart", "reason": "Smart money buying in neutral zone", "strength": 0.6})
        if data.institutional_flow < -0.3 and idx > 0.5:
            signals.append({"signal": "sell_follow_smart", "reason": "Smart money selling in neutral zone", "strength": 0.6})
        if data.credit_spread > 0.5 and data.vix_level > 0.5:
            signals.append({"signal": "caution_credit_stress", "reason": "High credit spreads + high VIX", "strength": 0.7})
        if data.market_breadth < 0.3 and idx > 0.5:
            signals.append({"signal": "bearish_divergence", "reason": "Narrow leadership despite optimism", "strength": 0.65})

        return {"signals": signals, "composite": composite}

    @staticmethod
    def contrarian_signal_detector(
        fear_greed_index: float = 0.5,
        vix_level: float = 0.3,
        put_call_ratio: float = 0.7,
        short_interest: float = 0.3,
        institutional_flow: float = 0.0,
        retail_flow: float = 0.0,
        market_breadth: float = 0.5,
        safe_haven_demand: float = 0.4,
        credit_spread: float = 0.3,
        social_media_sentiment: float = 0.5,
        news_sentiment: float = 0.5,
        consumer_confidence: float = 0.5,
        lookback_period: int = 20,
    ) -> dict:
        indicators = {
            "fear_greed": fear_greed_index,
            "vix": vix_level,
            "put_call": put_call_ratio,
            "short_interest": short_interest,
            "institutional_flow": institutional_flow,
            "retail_flow": retail_flow,
            "market_breadth": market_breadth,
            "safe_haven": safe_haven_demand,
            "credit_spread": credit_spread,
            "social_media": social_media_sentiment,
            "news": news_sentiment,
            "consumer_confidence": consumer_confidence,
        }
        extremes = []
        if fear_greed_index < 0.2:
            extremes.append({"indicator": "fear_greed", "direction": "extreme_fear", "z_score": round((0.5 - fear_greed_index) / 0.15, 2)})
        elif fear_greed_index > 0.8:
            extremes.append({"indicator": "fear_greed", "direction": "extreme_greed", "z_score": round((fear_greed_index - 0.5) / 0.15, 2)})
        if vix_level > 0.6:
            extremes.append({"indicator": "vix", "direction": "extreme_fear", "z_score": round((vix_level - 0.3) / 0.2, 2)})
        elif vix_level < 0.15:
            extremes.append({"indicator": "vix", "direction": "extreme_complacency", "z_score": round((0.3 - vix_level) / 0.15, 2)})
        if put_call_ratio > 1.2:
            extremes.append({"indicator": "put_call", "direction": "excessive_bearishness", "z_score": round((put_call_ratio - 0.7) / 0.3, 2)})
        elif put_call_ratio < 0.5:
            extremes.append({"indicator": "put_call", "direction": "excessive_bullishness", "z_score": round((0.7 - put_call_ratio) / 0.2, 2)})
        if short_interest > 0.5:
            extremes.append({"indicator": "short_interest", "direction": "heavy_shorting", "z_score": round((short_interest - 0.3) / 0.15, 2)})
        divergences = []
        if institutional_flow > 0.3 and retail_flow < -0.2:
            divergences.append({"type": "smart_money_retail_divergence", "direction": "bullish", "description": "Smart money accumulating while retail selling — classic bottom signal"})
        elif institutional_flow < -0.3 and retail_flow > 0.2:
            divergences.append({"type": "smart_money_retail_divergence", "direction": "bearish", "description": "Smart money distributing while retail buying — distribution top signal"})
        if social_media_sentiment > 0.8 and news_sentiment < 0.3:
            divergences.append({"type": "social_news_divergence", "direction": "bearish", "description": "Social media euphoria not supported by fundamentals"})
        elif social_media_sentiment < 0.2 and news_sentiment > 0.6:
            divergences.append({"type": "social_news_divergence", "direction": "bullish", "description": "Panic selling while fundamentals improving — contrarian buy"})
        if market_breadth < 0.3 and fear_greed_index > 0.5:
            divergences.append({"type": "breadth_sentiment_divergence", "direction": "bearish", "description": "Narrow leadership despite positive sentiment — topping pattern"})
        credit_risk = 0
        if credit_spread > 0.5:
            credit_risk = 1
        if credit_spread > 0.5 and vix_level > 0.5:
            credit_risk = 2
        signals = []
        if len(extremes) >= 3 and fear_greed_index < 0.3:
            signals.append({"signal": "strong_buy", "confidence": 0.8, "horizon": f"{lookback_period}d",
                           "reason": "Multi-indicator extreme fear — contrarian buy zone"})
        elif len(extremes) >= 2 and fear_greed_index < 0.3:
            signals.append({"signal": "buy", "confidence": 0.6, "horizon": f"{lookback_period}d",
                           "reason": "Elevated fear across indicators — buying opportunity"})
        elif len(extremes) >= 3 and fear_greed_index > 0.7:
            signals.append({"signal": "strong_sell", "confidence": 0.75, "horizon": f"{lookback_period}d",
                           "reason": "Multi-indicator euphoria — contrarian sell zone"})
        elif len(extremes) >= 2 and fear_greed_index > 0.7:
            signals.append({"signal": "sell", "confidence": 0.55, "horizon": f"{lookback_period}d",
                           "reason": "Elevated greed — reduce exposure"})
        for div in divergences:
            signals.append({"signal": div["direction"], "confidence": 0.5, "horizon": f"{lookback_period}d",
                           "reason": div["description"], "type": div["type"]})
        if credit_risk == 2:
            signals.append({"signal": "caution", "confidence": 0.7, "horizon": f"{lookback_period}d",
                           "reason": "Credit stress + VIX spike — systemic risk elevated"})
        composite_contrarian = 0.0
        total_weight = 0.0
        contrarian_map = {
            "fear_greed": (0.5 - abs(fear_greed_index - 0.5)) / 0.5,
            "vix": (0.5 - abs(vix_level - 0.3)) / 0.3 if vix_level < 0.3 else -(vix_level - 0.3) / 0.3,
            "put_call": -(put_call_ratio - 0.7) / 0.5 if put_call_ratio > 0.7 else (0.7 - put_call_ratio) / 0.5,
        }
        weights = {"fear_greed": 0.4, "vix": 0.3, "put_call": 0.3}
        for name, val in contrarian_map.items():
            composite_contrarian += val * weights[name]
            total_weight += weights[name]
        composite_contrarian = composite_contrarian / total_weight if total_weight > 0 else 0
        if composite_contrarian > 0.3:
            overall = "bullish_contrarian"
        elif composite_contrarian < -0.3:
            overall = "bearish_contrarian"
        else:
            overall = "neutral"
        return {
            "extreme_indicators": extremes,
            "divergences": divergences,
            "signals": signals,
            "composite_contrarian_score": round(float(composite_contrarian), 4),
            "overall_bias": overall,
            "credit_risk_level": ["low", "moderate", "elevated", "high"][min(credit_risk, 3)],
            "indicators_summary": {k: round(v, 3) for k, v in indicators.items()},
        }

    def predict(
        self, data: SentimentData, target_date: date,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    ) -> list[Prediction]:
        composite = self.compute_composite_index(data)
        years_ahead = max(0.08, (target_date - data.timestamp).days / 365.25)
        z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645,
             ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence]

        predictions = []
        metrics = [
            ("sentiment_composite", composite["composite_index"], 0.1 * years_ahead, Domain.FINANCE),
            ("fear_greed", data.fear_greed_index, 0.12 * years_ahead, Domain.FINANCE),
            ("consumer_confidence", data.consumer_confidence, 0.08 * years_ahead, Domain.ECONOMICS),
            ("market_breadth", data.market_breadth, 0.10 * years_ahead, Domain.FINANCE),
        ]
        for name, value, std, domain in metrics:
            ci = (max(0, value - z * std), min(1, value + z * std))
            predictions.append(Prediction(
                domain=domain, variable=f"sentiment_{name}", value=round(float(value), 4),
                confidence_interval=(round(float(ci[0]), 4), round(float(ci[1]), 4)),
                confidence_level=confidence, timestamp=datetime.now(),
                target_date=target_date, model_name="sentiment_engine",
                weights={"composite": 0.5, "contrarian": 0.3, "momentum": 0.2},
            ))
        return predictions

    def _compute_momentum(self) -> float:
        if len(self._history) < 2:
            return 0.0
        recent = [d.fear_greed_index for d in self._history[-10:]]
        if len(recent) < 2:
            return 0.0
        return float((recent[-1] - recent[0]) / len(recent))

    def _compute_divergence(self, data: SentimentData) -> float:
        sentiment_composite = (data.fear_greed_index + data.news_sentiment + data.social_media_sentiment) / 3
        flow_composite = (data.institutional_flow + data.market_breadth) / 2 + 0.5
        return float(sentiment_composite - flow_composite)