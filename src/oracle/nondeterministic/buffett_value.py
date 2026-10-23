import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from oracle.core.types import Domain, Prediction, ConfidenceLevel


@dataclass
class CompanyFundamentals:
    name: str = ""
    ticker: str = ""
    earnings_per_share: float = 5.0
    book_value_per_share: float = 40.0
    free_cash_flow_per_share: float = 4.0
    dividends_per_share: float = 1.5
    revenue_growth_5yr: float = 0.08
    earnings_growth_5yr: float = 0.10
    roe: float = 0.15
    roic: float = 0.12
    debt_to_equity: float = 0.5
    current_ratio: float = 2.0
    gross_margin: float = 0.45
    operating_margin: float = 0.20
    net_margin: float = 0.12
    payout_ratio: float = 0.35
    share_count_mm: float = 1000.0
    market_price: float = 100.0
    sector: str = "general"
    years_data: int = 10


@dataclass
class MoatAssessment:
    brand_strength: float = 0.5
    switching_costs: float = 0.5
    network_effects: float = 0.5
    cost_advantages: float = 0.5
    intangible_assets: float = 0.5
    efficient_scale: float = 0.5
    overall_moat_score: float = 0.5
    moat_durability_years: float = 10.0
    moat_type: str = "none"


@dataclass
class ValuationResult:
    intrinsic_value: float
    margin_of_safety: float
    fair_value_range: tuple[float, float]
    dcf_value: float
    graham_value: float
    earnings_power_value: float
    quality_score: float
    moat_assessment: MoatAssessment
    risk_level: str = "moderate"
    recommendation: str = "hold"


class BuffettValueEngine:
    def __init__(self, risk_free_rate: float = 0.04, market_return: float = 0.10,
                 margin_of_safety: float = 0.25, decay_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.market_return = market_return
        self.margin_of_safety_requirement = margin_of_safety
        self.moat_decay_rate = decay_rate

    def assess_moat(self, fundamentals: CompanyFundamentals) -> MoatAssessment:
        margins = {
            "brand_strength": self._score_brand(fundamentals),
            "switching_costs": self._score_switching(fundamentals),
            "network_effects": self._score_network(fundamentals),
            "cost_advantages": self._score_cost(fundamentals),
            "intangible_assets": self._score_intangibles(fundamentals),
            "efficient_scale": self._score_scale(fundamentals),
        }
        overall = float(np.mean(list(margins.values())))
        durability = self._estimate_moat_durability(margins, fundamentals)
        moat_type = self._classify_moat(margins)
        return MoatAssessment(
            brand_strength=margins["brand_strength"],
            switching_costs=margins["switching_costs"],
            network_effects=margins["network_effects"],
            cost_advantages=margins["cost_advantages"],
            intangible_assets=margins["intangible_assets"],
            efficient_scale=margins["efficient_scale"],
            overall_moat_score=overall,
            moat_durability_years=durability,
            moat_type=moat_type,
        )

    def dcf_valuation(
        self, fundamentals: CompanyFundamentals, years: int = 10,
        terminal_growth: float = 0.03, discount_rate: Optional[float] = None,
    ) -> float:
        r = discount_rate or self._wacc(fundamentals)
        fcf = fundamentals.free_cash_flow_per_share * fundamentals.share_count_mm
        growth = fundamentals.earnings_growth_5yr
        moat_adjusted_growth = growth * (1 + 0.2 * self.assess_moat(fundamentals).overall_moat_score)
        decay_growth = min(growth, 0.20 + moat_adjusted_growth * 0.3)

        present_value = 0.0
        for year in range(1, years + 1):
            year_growth = min(decay_growth, growth * max(0.5, 1 - 0.02 * year))
            fcf_projected = fcf * (1 + year_growth) ** year
            present_value += fcf_projected / (1 + r) ** year

        terminal_fcf = fcf * (1 + decay_growth) ** years * (1 + terminal_growth)
        terminal_value = terminal_fcf / (r - terminal_growth)
        present_terminal = terminal_value / (1 + r) ** years

        total_value = present_value + present_terminal
        per_share = total_value / fundamentals.share_count_mm

        return round(per_share, 2)

    def graham_number(self, fundamentals: CompanyFundamentals) -> float:
        earnings = fundamentals.arnings_per_share if hasattr(fundamentals, 'arnings_per_share') else fundamentals.earnings_per_share
        graham = np.sqrt(22.5 * earnings * fundamentals.book_value_per_share)
        return round(graham, 2)

    def earnings_power_value(self, fundamentals: CompanyFundamentals) -> float:
        epv = fundamentals.earnings_per_share / self.risk_free_rate
        adjustment = 0.3 * fundamentals.operating_margin + 0.2 * fundamentals.roe + 0.2 * fundamentals.roic
        epv_adjusted = epv * (0.7 + adjustment)
        return round(epv_adjusted, 2)

    def full_valuation(self, fundamentals: CompanyFundamentals) -> ValuationResult:
        moat = self.assess_moat(fundamentals)

        dcf = self.dcf_valuation(fundamentals)
        graham = self.graham_number(fundamentals)
        epv = self.earnings_power_value(fundamentals)

        moat_multiplier = 0.5 + 0.5 * moat.overall_moat_score
        weights = {
            "dcf": 0.45 * moat_multiplier,
            "graham": 0.20 * (2 - moat_multiplier),
            "epv": 0.35 * (1.5 - 0.5 * moat_multiplier),
        }
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}

        intrinsic_value = weights["dcf"] * dcf + weights["graham"] * graham + weights["epv"] * epv

        uncertainty = 1.0 - moat.overall_moat_score * 0.3
        value_range = (
            intrinsic_value * (1 - 0.15 * uncertainty),
            intrinsic_value * (1 + 0.15 * uncertainty),
        )

        margin = (intrinsic_value - fundamentals.market_price) / intrinsic_value if intrinsic_value > 0 else 0

        quality = self._quality_score(fundamentals, moat)

        risk = "low" if moat.overall_moat_score > 0.7 and margin > 0.2 else \
               "moderate" if moat.overall_moat_score > 0.4 else "high"

        rec = "strong_buy" if margin > 0.4 and moat.overall_moat_score > 0.6 else \
              "buy" if margin > self.margin_of_safety_requirement and moat.overall_moat_score > 0.4 else \
              "hold" if margin > -0.1 else "sell"

        return ValuationResult(
            intrinsic_value=round(intrinsic_value, 2),
            margin_of_safety=round(margin, 4),
            fair_value_range=(round(value_range[0], 2), round(value_range[1], 2)),
            dcf_value=round(dcf, 2),
            graham_value=round(graham, 2),
            earnings_power_value=round(epv, 2),
            quality_score=round(quality, 4),
            moat_assessment=moat,
            risk_level=risk,
            recommendation=rec,
        )

    def predict(
        self, fundamentals: CompanyFundamentals, target_date: date,
        confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH,
    ) -> Prediction:
        valuation = self.full_valuation(fundamentals)
        years_ahead = max(1, (target_date - date.today()).days / 365.25)
        expected_return = fundamentals.earnings_growth_5yr * valuation.moat_assessment.overall_moat_score
        predicted_value = fundamentals.market_price * (1 + expected_return) ** years_ahead

        volatility = self._estimate_volatility(fundamentals, valuation)
        z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645,
             ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence_level]
        ci_width = predicted_value * volatility * np.sqrt(years_ahead) * z

        return Prediction(
            domain=Domain.FINANCE,
            variable=f"{fundamentals.ticker}_price",
            value=round(predicted_value, 2),
            confidence_interval=(round(predicted_value - ci_width, 2), round(predicted_value + ci_width, 2)),
            confidence_level=confidence_level,
            timestamp=datetime.now(),
            target_date=target_date,
            model_name="buffett_value",
            weights={
                "dcf": valuation.dcf_value / valuation.intrinsic_value if valuation.intrinsic_value > 0 else 0.45,
                "graham": valuation.graham_value / valuation.intrinsic_value if valuation.intrinsic_value > 0 else 0.2,
                "epv": valuation.earnings_power_value / valuation.intrinsic_value if valuation.intrinsic_value > 0 else 0.35,
                "moat": valuation.moat_assessment.overall_moat_score,
            },
            signals_used=["earnings", "book_value", "free_cash_flow", "roe", "roic", "moat"],
        )

    def _wacc(self, f: CompanyFundamentals) -> float:
        cost_of_equity = self.risk_free_rate + 1.0 * (self.market_return - self.risk_free_rate)
        cost_of_debt = self.risk_free_rate * 1.5
        equity_weight = 1.0 / (1.0 + f.debt_to_equity)
        debt_weight = 1.0 - equity_weight
        return equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - 0.21)

    def _score_brand(self, f: CompanyFundamentals) -> float:
        score = 0.3
        if f.gross_margin > 0.4:
            score += 0.2
        if f.net_margin > 0.10:
            score += 0.15
        if f.roe > 0.15:
            score += 0.15
        if f.earnings_growth_5yr > 0.08:
            score += 0.1
        return min(1.0, score)

    def _score_switching(self, f: CompanyFundamentals) -> float:
        score = 0.2
        if f.gross_margin > 0.5:
            score += 0.2
        if f.current_ratio > 1.5:
            score += 0.1
        if f.revenue_growth_5yr > 0.05:
            score += 0.15
        return min(1.0, score)

    def _score_network(self, f: CompanyFundamentals) -> float:
        score = 0.2
        if f.revenue_growth_5yr > 0.10:
            score += 0.2
        if f.roic > 0.15:
            score += 0.15
        return min(1.0, score)

    def _score_cost(self, f: CompanyFundamentals) -> float:
        score = 0.2
        if f.operating_margin > 0.20:
            score += 0.25
        if f.gross_margin > 0.45:
            score += 0.2
        return min(1.0, score)

    def _score_intangibles(self, f: CompanyFundamentals) -> float:
        score = 0.2
        if f.roe > 0.20:
            score += 0.2
        if f.net_margin > 0.15:
            score += 0.2
        return min(1.0, score)

    def _score_scale(self, f: CompanyFundamentals) -> float:
        score = 0.2
        if f.operating_margin > 0.15:
            score += 0.15
        if f.debt_to_equity < 0.7:
            score += 0.15
        return min(1.0, score)

    def _estimate_moat_durability(self, margins: dict, f: CompanyFundamentals) -> float:
        base = 10.0
        if np.mean(list(margins.values())) > 0.7:
            base += 10
        if f.roe > 0.20:
            base += 5
        if f.earnings_growth_5yr > 0.10:
            base += 5
        return min(30.0, base)

    def _classify_moat(self, margins: dict) -> str:
        best = max(margins, key=margins.get)
        if margins[best] > 0.6:
            return best
        if margins.get("network_effects", 0) > 0.5 and margins.get("switching_costs", 0) > 0.5:
            return "platform"
        if margins.get("brand_strength", 0) > 0.5 and margins.get("cost_advantages", 0) > 0.5:
            return "brand_and_cost"
        return "weak"

    def _quality_score(self, f: CompanyFundamentals, moat: MoatAssessment) -> float:
        scores = {
            "roe": min(f.roe / 0.20, 1.0),
            "roic": min(f.roic / 0.15, 1.0),
            "margin": min(f.net_margin / 0.15, 1.0),
            "growth": min(f.earnings_growth_5yr / 0.10, 1.0),
            "leverage": min(0.5 / max(f.debt_to_equity, 0.01), 1.0),
            "moat": moat.overall_moat_score,
            "consistency": min(f.years_data / 20, 1.0),
        }
        return float(np.mean(list(scores.values())))

    def _estimate_volatility(self, f: CompanyFundamentals, v: ValuationResult) -> float:
        base = 0.25
        if f.debt_to_equity > 1.0:
            base += 0.10
        if f.earnings_growth_5yr > 0.15:
            base += 0.05
        if v.moat_assessment.overall_moat_score < 0.4:
            base += 0.10
        if f.net_margin < 0.08:
            base += 0.05
        return min(0.60, base)