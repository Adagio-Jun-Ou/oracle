import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from oracle.core.types import Domain, Prediction, ConfidenceLevel


@dataclass
class ESGState:
    environmental_score: float = 0.55
    social_score: float = 0.50
    governance_score: float = 0.60
    carbon_intensity: float = 0.45
    water_stress: float = 0.40
    waste_management_score: float = 0.50
    labor_practices_score: float = 0.55
    community_impact: float = 0.50
    board_independence: float = 0.65
    executive_compensation_ratio: float = 0.35
    anti_corruption_score: float = 0.60
    data_privacy_score: float = 0.45
    supply_chain_transparency: float = 0.40


@dataclass
class ESGRiskAssessment:
    overall_esg_score: float
    environmental_risk: float
    social_risk: float
    governance_risk: float
    risk_category: str
    key_concerns: list
    recommendations: list
    sector_comparison: dict
    trend: str


class ESGRiskModel:
    def __init__(self):
        self.sector_benchmarks = {
            "technology": {"e": 0.65, "s": 0.55, "g": 0.70},
            "healthcare": {"e": 0.55, "s": 0.65, "g": 0.60},
            "financials": {"e": 0.45, "s": 0.50, "g": 0.65},
            "energy": {"e": 0.25, "s": 0.40, "g": 0.55},
            "consumer_staples": {"e": 0.50, "s": 0.60, "g": 0.55},
            "industrials": {"e": 0.40, "s": 0.45, "g": 0.55},
            "materials": {"e": 0.30, "s": 0.40, "g": 0.50},
            "utilities": {"e": 0.35, "s": 0.50, "g": 0.60},
        }

        self.esg_weights = {
            "environmental": 0.35,
            "social": 0.30,
            "governance": 0.35,
        }

    def assess(self, state: ESGState) -> ESGRiskAssessment:
        e_score = self._compute_environmental(state)
        s_score = self._compute_social(state)
        g_score = self._compute_governance(state)

        overall = (
            e_score * self.esg_weights["environmental"]
            + s_score * self.esg_weights["social"]
            + g_score * self.esg_weights["governance"]
        )

        concerns = []
        recs = []

        if state.carbon_intensity > 0.6:
            concerns.append("High carbon intensity")
            recs.append("Implement carbon reduction targets and renewable energy transition plan")
        if state.water_stress > 0.6:
            concerns.append("Water stress risk")
            recs.append("Develop water conservation and recycling programs")
        if state.labor_practices_score < 0.4:
            concerns.append("Labor practices concerns")
            recs.append("Strengthen labor rights policies and supply chain audits")
        if state.board_independence < 0.4:
            concerns.append("Low board independence")
            recs.append("Increase independent board member representation")
        if state.anti_corruption_score < 0.4:
            concerns.append("Anti-corruption weaknesses")
            recs.append("Enhance anti-corruption policies and whistleblower protections")
        if state.data_privacy_score < 0.4:
            concerns.append("Data privacy gaps")
            recs.append("Implement comprehensive data privacy framework compliant with GDPR/CCPA")
        if state.supply_chain_transparency < 0.4:
            concerns.append("Low supply chain transparency")
            recs.append("Publish supply chain transparency reports and conduct regular audits")

        if overall >= 0.7:
            category = "low_risk"
        elif overall >= 0.5:
            category = "medium_risk"
        elif overall >= 0.3:
            category = "high_risk"
        else:
            category = "critical_risk"

        e_risk = 1.0 - e_score
        s_risk = 1.0 - s_score
        g_risk = 1.0 - g_score

        sector_comp = {}
        for sector, scores in self.sector_benchmarks.items():
            sector_comp[sector] = {
                "environmental": round(e_score - scores["e"], 3),
                "social": round(s_score - scores["s"], 3),
                "governance": round(g_score - scores["g"], 3),
                "overall": round(overall - (scores["e"] * 0.35 + scores["s"] * 0.3 + scores["g"] * 0.35), 3),
            }

        if e_score > 0.6 and s_score > 0.5 and g_score > 0.6:
            trend = "improving"
        elif e_score < 0.4 or s_score < 0.4 or g_score < 0.4:
            trend = "deteriorating"
        else:
            trend = "stable"

        return ESGRiskAssessment(
            overall_esg_score=round(overall, 3),
            environmental_risk=round(e_risk, 3),
            social_risk=round(s_risk, 3),
            governance_risk=round(g_risk, 3),
            risk_category=category,
            key_concerns=concerns,
            recommendations=recs,
            sector_comparison=sector_comp,
            trend=trend,
        )

    def _compute_environmental(self, state: ESGState) -> float:
        scores = [
            state.environmental_score,
            1.0 - state.carbon_intensity,
            1.0 - state.water_stress,
            state.waste_management_score,
        ]
        return float(np.mean(scores))

    def _compute_social(self, state: ESGState) -> float:
        scores = [
            state.social_score,
            state.labor_practices_score,
            state.community_impact,
            state.data_privacy_score,
        ]
        return float(np.mean(scores))

    def _compute_governance(self, state: ESGState) -> float:
        scores = [
            state.governance_score,
            state.board_independence,
            1.0 - state.executive_compensation_ratio,
            state.anti_corruption_score,
            state.supply_chain_transparency,
        ]
        return float(np.mean(scores))

    def simulate_esg_scenario(
        self, state: ESGState, improvements: dict[str, float]
    ) -> ESGRiskAssessment:
        fields = {
            "environmental_score", "social_score", "governance_score",
            "carbon_intensity", "water_stress", "waste_management_score",
            "labor_practices_score", "community_impact", "board_independence",
            "executive_compensation_ratio", "anti_corruption_score",
            "data_privacy_score", "supply_chain_transparency",
        }
        modified = ESGState(**{k: improvements.get(k, getattr(state, k)) for k in fields})
        return self.assess(modified)

    def sector_esg_benchmark(self, sector: str) -> dict:
        if sector not in self.sector_benchmarks:
            sector = "technology"
        scores = self.sector_benchmarks[sector]
        overall = scores["e"] * 0.35 + scores["s"] * 0.3 + scores["g"] * 0.35
        return {
            "sector": sector,
            "environmental_score": scores["e"],
            "social_score": scores["s"],
            "governance_score": scores["g"],
            "overall_esg_score": round(overall, 3),
        }

    @staticmethod
    def get_market_derived_esg() -> "ESGState":
        try:
            from oracle.ingestion.real_data import RealDataFetcher
            fetcher = RealDataFetcher(cache_ttl=3600)
            vix_arr = fetcher.get_close_array("vix", "1y")
            oil_arr = fetcher.get_close_array("oil", "1y")
            gold_arr = fetcher.get_close_array("gold", "1y")
            esg_state = ESGState()
            if vix_arr is not None and len(vix_arr) > 20:
                vix_level = float(vix_arr[-1])
                volatility_factor = min(vix_level / 40.0, 1.0)
                esg_state.governance_score = max(0.3, 0.75 - volatility_factor * 0.3)
                esg_state.social_score = max(0.3, 0.65 - volatility_factor * 0.2)
                esg_state.anti_corruption_score = max(0.3, 0.7 - volatility_factor * 0.2)
            if oil_arr is not None and len(oil_arr) > 20:
                oil_price = float(oil_arr[-1])
                oil_change = float((oil_arr[-1] / oil_arr[-22] - 1)) if len(oil_arr) > 22 else 0
                esg_state.carbon_intensity = min(1.0, max(0.1, 0.3 + oil_price / 200.0))
                esg_state.environmental_score = max(0.2, min(0.8, 0.6 - oil_price / 300.0 + 0.2))
            if gold_arr is not None and len(gold_arr) > 20:
                gold_price = float(gold_arr[-1])
                esg_state.data_privacy_score = max(0.3, min(0.8, 0.5 + (gold_price - 1800) / 5000.0))
            return esg_state
        except Exception:
            return ESGState()