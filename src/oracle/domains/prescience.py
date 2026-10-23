from dataclasses import dataclass, field
from typing import Optional
import numpy as np


EVENT_TEMPLATES = {
    "m7_earthquake": {"name": "M7+ Earthquake", "base_rate_per_year": 0.15, "domains": ["seismic", "infrastructure", "housing"], "severity_range": (0.3, 0.8), "cascade_prob": 0.4, "recovery_years": 5},
    "carrington_event": {"name": "Carrington-Class Solar Storm", "base_rate_per_year": 0.012, "domains": ["solar", "infrastructure", "cyber", "technology"], "severity_range": (0.4, 0.9), "cascade_prob": 0.6, "recovery_years": 8},
    "pandemic_sars2": {"name": "SARS-2-Level Pandemic", "base_rate_per_year": 0.03, "domains": ["pandemic", "epidemiology", "mental_health", "economic"], "severity_range": (0.3, 0.9), "cascade_prob": 0.7, "recovery_years": 3},
    "m9_megathrust": {"name": "M9 Megathrust Earthquake + Tsunami", "base_rate_per_year": 0.005, "domains": ["seismic", "oceanic", "infrastructure", "food", "water"], "severity_range": (0.5, 1.0), "cascade_prob": 0.8, "recovery_years": 15},
    "vei6_eruption": {"name": "VEI 6+ Volcanic Eruption", "base_rate_per_year": 0.01, "domains": ["seismic", "atmospheric", "climate", "food", "aviation"], "severity_range": (0.3, 0.8), "cascade_prob": 0.5, "recovery_years": 10},
    "amoc_collapse": {"name": "AMOC Collapse", "base_rate_per_year": 0.005, "domains": ["oceanic", "climate", "food", "energy", "biodiversity"], "severity_range": (0.5, 1.0), "cascade_prob": 0.7, "recovery_years": 50},
    "global_cyber_attack": {"name": "Systemic Cyber Attack on Critical Infrastructure", "base_rate_per_year": 0.08, "domains": ["cyber", "infrastructure", "economic", "supply_chain"], "severity_range": (0.2, 0.7), "cascade_prob": 0.5, "recovery_years": 2},
    "food_crisis": {"name": "Global Food Crisis (Multi-Breadbasket Failure)", "base_rate_per_year": 0.04, "domains": ["food", "climate", "water", "political", "social"], "severity_range": (0.2, 0.7), "cascade_prob": 0.6, "recovery_years": 3},
    "financial_crisis": {"name": "Systemic Financial Crisis (2008-Level)", "base_rate_per_year": 0.05, "domains": ["economic", "sentiment", "contagion", "housing", "political"], "severity_range": (0.3, 0.8), "cascade_prob": 0.6, "recovery_years": 5},
    "ai_disruption": {"name": "Major AI-Driven Economic Disruption", "base_rate_per_year": 0.06, "domains": ["technology", "social", "economic", "mental_health", "political"], "severity_range": (0.2, 0.6), "cascade_prob": 0.4, "recovery_years": 4},
    "water_crisis": {"name": "Multi-Region Water Crisis", "base_rate_per_year": 0.03, "domains": ["water", "food", "political", "social", "biodiversity"], "severity_range": (0.3, 0.7), "cascade_prob": 0.5, "recovery_years": 5},
    "biodiversity_tipping": {"name": "Ecosystem Tipping Point Cascade", "base_rate_per_year": 0.02, "domains": ["biodiversity", "ecological", "climate", "food", "water"], "severity_range": (0.4, 0.9), "cascade_prob": 0.7, "recovery_years": 50},
    "nuclear_confrontation": {"name": "Nuclear-Armed Confrontation", "base_rate_per_year": 0.01, "domains": ["geopolitical", "infrastructure", "economic", "mental_health", "political"], "severity_range": (0.5, 1.0), "cascade_prob": 0.8, "recovery_years": 20},
    "supply_chain_collapse": {"name": "Global Supply Chain Collapse", "base_rate_per_year": 0.04, "domains": ["supply_chain", "economic", "food", "cyber", "technology"], "severity_range": (0.2, 0.7), "cascade_prob": 0.5, "recovery_years": 3},
    "climate_tipping": {"name": "Climate Tipping Point Activation", "base_rate_per_year": 0.015, "domains": ["climate", "oceanic", "atmospheric", "paleoclimate", "biodiversity"], "severity_range": (0.5, 1.0), "cascade_prob": 0.7, "recovery_years": 100},
    "pandemic_geopolitical_cascade": {"name": "Pandemic-Geopolitical Cascade", "base_rate_per_year": 0.02, "domains": ["pandemic", "geopolitical", "economic", "supply_chain", "political", "social"], "severity_range": (0.4, 0.9), "cascade_prob": 0.8, "recovery_years": 10},
    "volcanic_winter_agricultural": {"name": "Volcanic Winter + Agricultural Collapse", "base_rate_per_year": 0.008, "domains": ["seismic", "atmospheric", "climate", "food", "water", "energy"], "severity_range": (0.5, 0.95), "cascade_prob": 0.7, "recovery_years": 15},
    "solar_cyber_infrastructure": {"name": "Solar Storm + Cyber Attack + Infrastructure Triple", "base_rate_per_year": 0.005, "domains": ["solar", "cyber", "infrastructure", "technology", "economic", "supply_chain"], "severity_range": (0.6, 1.0), "cascade_prob": 0.9, "recovery_years": 8},
    "biotech_pandemic_spillover": {"name": "Biotech/AI-Enhanced Pandemic Spillover", "base_rate_per_year": 0.01, "domains": ["pandemic", "technology", "biodiversity", "health", "infrastructure"], "severity_range": (0.5, 1.0), "cascade_prob": 0.6, "recovery_years": 5},
    "climate_migration_conflict": {"name": "Climate Refugee + Interstate Conflict", "base_rate_per_year": 0.025, "domains": ["climate", "social", "political", "geopolitical", "food", "water"], "severity_range": (0.3, 0.8), "cascade_prob": 0.6, "recovery_years": 20},
}

TIPPING_POINTS = [
    {"name": "Greenland Ice Sheet Collapse", "threshold_c": 1.5, "current_c": 1.2, "impact": "7m sea level rise, 1+ century recovery", "irreversibility": 0.9, "domains": ["climate", "oceanic"]},
    {"name": "West Antarctic Ice Sheet Disintegration", "threshold_c": 1.5, "current_c": 1.2, "impact": "3.3m sea level rise", "irreversibility": 0.85, "domains": ["climate", "oceanic"]},
    {"name": "Amazon Rainforest Dieback", "threshold_c": 2.5, "current_c": 1.2, "impact": "Carbon bomb releasing 90Gt CO2", "irreversibility": 0.7, "domains": ["biodiversity", "climate", "ecological"]},
    {"name": "AMOC Shutdown", "threshold_c": 2.0, "current_c": 1.2, "impact": "European cooling 5-10C, monsoon shift", "irreversibility": 0.6, "domains": ["oceanic", "climate", "atmospheric"]},
    {"name": "Permafrost Carbon Bomb", "threshold_c": 2.0, "current_c": 1.2, "impact": "50-250 Gt CO2 release", "irreversibility": 0.8, "domains": ["climate", "paleoclimate"]},
    {"name": "Coral Reef Global Die-off", "threshold_c": 1.5, "current_c": 1.2, "impact": "Marine ecosystem collapse, 500M people affected", "irreversibility": 0.75, "domains": ["biodiversity", "oceanic", "food"]},
    {"name": "Boreal Forest Shift", "threshold_c": 3.0, "current_c": 1.2, "impact": "Albedo change accelerating warming", "irreversibility": 0.5, "domains": ["biodiversity", "climate", "ecological"]},
    {"name": "Sahel Desertification", "threshold_c": 2.0, "current_c": 1.2, "impact": "100M+ climate refugees", "irreversibility": 0.6, "domains": ["climate", "social", "political", "food"]},
    {"name": "Global Financial System Collapse", "threshold_debt_gdp": 2.5, "current_debt_gdp": 1.0, "impact": "Decade-long depression, 30% GDP loss", "irreversibility": 0.3, "domains": ["economic", "contagion", "political"]},
    {"name": "AI Alignment Failure", "threshold_ai_cap": 0.5, "current_ai": 0.15, "impact": "Uncontrolled autonomous systems, economic disruption", "irreversibility": 0.9, "domains": ["technology", "cyber", "infrastructure"]},
    {"name": "Nuclear Exchange", "threshold_tension": 0.8, "current_tension": 0.5, "impact": "Nuclear winter, billions affected", "irreversibility": 1.0, "domains": ["geopolitical", "infrastructure"]},
    {"name": "Pandemic Spillover (H5N1-type)", "threshold_biodiversity": 0.3, "current_bio": 0.5, "impact": "Global pandemic, 10M+ deaths possible", "irreversibility": 0.2, "domains": ["pandemic", "biodiversity", "health"]},
    {"name": "Kessler Syndrome", "threshold_debris": 0.5, "current_debris": 0.25, "impact": "Loss of LEO, cascading satellite destruction", "irreversibility": 0.8, "domains": ["space", "technology", "infrastructure"]},
]

EARLY_WARNING_SIGNALS = {
    "geopolitical": [
        {"indicator": "us_china_tension_gt_0.7", "description": "US-China tensions above critical threshold", "lead_time_months": 6, "event": "nuclear_confrontation"},
        {"indicator": "trade_protectionism_gt_0.6", "description": "Trade protectionism surge", "lead_time_months": 12, "event": "supply_chain_collapse"},
    ],
    "economic": [
        {"indicator": "debt_to_gdp_gt_2.0", "description": "Debt-to-GDP above 200%", "lead_time_months": 18, "event": "financial_crisis"},
        {"indicator": "bubble_risk_gt_0.7", "description": "Asset bubble risk elevated", "lead_time_months": 6, "event": "financial_crisis"},
    ],
    "climate": [
        {"indicator": "temp_anomaly_gt_1.5", "description": "Global temperature anomaly above 1.5C", "lead_time_months": 60, "event": "climate_tipping"},
        {"indicator": "sea_level_rise_gt_5mm_yr", "description": "Sea level rise acceleration above 5mm/yr", "lead_time_months": 120, "event": "amoc_collapse"},
    ],
    "health": [
        {"indicator": "pandemic_preparedness_lt_0.3", "description": "Pandemic preparedness critically low", "lead_time_months": 3, "event": "pandemic_sars2"},
        {"indicator": "biodiversity_index_lt_0.3", "description": "Biodiversity index critically low", "lead_time_months": 6, "event": "pandemic_spillover (H5N1-type)"},
    ],
    "technology": [
        {"indicator": "ai_adoption_gt_0.5", "description": "AI adoption above 50%", "lead_time_months": 24, "event": "ai_disruption"},
        {"indicator": "kessler_risk_gt_0.5", "description": "Kessler syndrome risk elevated", "lead_time_months": 36, "event": "Kessler Syndrome"},
    ],
    "infrastructure": [
        {"indicator": "resilience_lt_0.3", "description": "Infrastructure resilience critically low", "lead_time_months": 12, "event": "global_cyber_attack"},
    ],
    "compound": [
        {"indicator": "pandemic_geopolitical_gt_0.6", "description": "Pandemic vulnerability + geopolitical tension compound risk", "lead_time_months": 3, "event": "pandemic_geopolitical_cascade"},
        {"indicator": "volcanic_food_gt_0.5", "description": "Volcanic unrest + food system vulnerability", "lead_time_months": 6, "event": "volcanic_winter_agricultural"},
        {"indicator": "solar_cyber_infra_gt_0.5", "description": "Solar storm probability + cyber threat + infrastructure weakness", "lead_time_months": 1, "event": "solar_cyber_infrastructure"},
        {"indicator": "climate_migration_gt_0.6", "description": "Climate stress + migration pressure compound risk", "lead_time_months": 18, "event": "climate_migration_conflict"},
    ],
}


@dataclass
class PrescienceEvent:
    event_id: str
    event_name: str
    annual_probability: float
    severity: float
    affected_domains: list
    cascade_probability: float
    recovery_years: int
    lead_time_months: int
    triggered_signals: list
    compounding_events: list
    impact_description: str


@dataclass
class TippingPointStatus:
    name: str
    proximity: float
    current_value: float
    threshold: float
    gap: float
    impact: str
    irreversibility: float
    affected_domains: list
    time_horizon_years: int
    early_warning_active: bool


class PrescienceEngine:

    def __init__(self, seed: int = 42):
        self.name = "prescience_engine"
        self.seed = seed

    def forecast_events(self, domain_scores: dict[str, float],
                         years: int = 10,
                         n_simulations: int = 1000,
                         min_probability: float = 0.001) -> dict:
        rng = np.random.RandomState(self.seed)
        events = []

        for event_id, template in EVENT_TEMPLATES.items():
            base_rate = template["base_rate_per_year"]
            severity_lo, severity_hi = template["severity_range"]

            domain_modifier = 1.0
            for domain in template["domains"]:
                score = domain_scores.get(domain, 0.5)
                risk = 1.0 - score
                domain_modifier *= (1.0 + risk * 0.3)

            adjusted_rate = min(0.5, base_rate * domain_modifier)

            period_prob = 1.0 - (1.0 - adjusted_rate) ** years

            if period_prob < min_probability:
                continue

            n_triggered = 0
            severities = []
            impacts = []
            for _ in range(n_simulations):
                if rng.random() < adjusted_rate * years:
                    n_triggered += 1
                    sev = rng.uniform(severity_lo, severity_hi)
                    severities.append(sev)
                    impact = sev * rng.uniform(0.3, 1.0) * template["cascade_prob"]
                    impacts.append(impact)

            compound_events = []
            for other_id, other_tmpl in EVENT_TEMPLATES.items():
                if other_id != event_id:
                    shared = len(set(template["domains"]) & set(other_tmpl["domains"]))
                    if shared >= 2:
                        compound_rate = other_tmpl["base_rate_per_year"] * template["cascade_prob"] * (shared / 5.0)
                        compound_prob = 1.0 - (1.0 - compound_rate) ** years
                        if compound_prob > 0.01:
                            compound_events.append({
                                "compound_with": other_tmpl["name"],
                                "compound_probability": round(compound_prob, 4),
                                "shared_domains": shared,
                            })

            lead_time = max(1, int(12 / (adjusted_rate * 10 + 0.1)))

            events.append(PrescienceEvent(
                event_id=event_id,
                event_name=template["name"],
                annual_probability=round(adjusted_rate, 4),
                severity=round(float(np.mean(severities)) if severities else (severity_lo + severity_hi) / 2, 3),
                affected_domains=template["domains"],
                cascade_probability=template["cascade_prob"],
                recovery_years=template["recovery_years"],
                lead_time_months=lead_time,
                triggered_signals=[],
                compounding_events=compound_events[:3],
                impact_description="",
            ))

        events.sort(key=lambda e: e.annual_probability, reverse=True)

        total_cascade_prob = 1.0 - np.prod([1.0 - e.annual_probability for e in events])

        return sanitize_nan({
            "n_events_forecast": len(events),
            "years": years,
            "n_simulations": n_simulations,
            "any_catastrophic_probability": round(float(total_cascade_prob), 4),
            "events": [
                {
                    "event_id": e.event_id,
                    "event_name": e.event_name,
                    "annual_probability": e.annual_probability,
                    "period_probability": round(1.0 - (1.0 - e.annual_probability) ** years, 4),
                    "severity": e.severity,
                    "affected_domains": e.affected_domains,
                    "cascade_probability": e.cascade_probability,
                    "recovery_years": e.recovery_years,
                    "lead_time_months": e.lead_time_months,
                    "compounding_events": e.compounding_events,
                }
                for e in events
            ],
            "top_5_risks": [
                {"event": e.event_name, "annual_prob": e.annual_probability, "severity": e.severity}
                for e in events[:5]
            ],
        })

    def assess_tipping_points(self, domain_scores: dict[str, float],
                                current_temp_anomaly: float = 1.2) -> dict:
        statuses = []
        for tp in TIPPING_POINTS:
            if "threshold_c" in tp:
                threshold = tp["threshold_c"]
                current = current_temp_anomaly
                domain_avg = np.mean([domain_scores.get(d, 0.5) for d in tp["domains"]])
                proximity = min(1.0, current / threshold)
                if domain_avg < 0.4:
                    current += 0.2
                    proximity = min(1.0, current / threshold)
            elif "threshold_debt_gdp" in tp:
                threshold = tp["threshold_debt_gdp"]
                current = domain_scores.get("economic", 0.5)
                risk = 1.0 - current
                proximity = min(1.0, risk * 2)
            elif "threshold_tension" in tp:
                threshold = tp["threshold_tension"]
                current = 1.0 - domain_scores.get("geopolitical", 0.5)
                proximity = min(1.0, current / threshold)
            elif "threshold_biodiversity" in tp:
                threshold = tp["threshold_biodiversity"]
                current = 1.0 - domain_scores.get("biodiversity", 0.5)
                proximity = min(1.0, current / threshold)
            elif "threshold_ai_cap" in tp:
                threshold = tp["threshold_ai_cap"]
                current = domain_scores.get("technology", 0.15)
                proximity = min(1.0, current / threshold)
            elif "threshold_debris" in tp:
                threshold = tp["threshold_debris"]
                current = 1.0 - domain_scores.get("space", 0.5)
                proximity = min(1.0, current / threshold)
            else:
                proximity = 0.5

            gap = max(0, 1.0 - proximity)
            early_warning = proximity > 0.7

            time_horizon = max(1, int(50 * gap)) if gap > 0 else 0

            if proximity < 0.5:
                status_label = "distant"
            elif proximity < 0.7:
                status_label = "approaching"
            elif proximity < 0.85:
                status_label = "near"
            elif proximity < 0.95:
                status_label = "imminent"
            else:
                status_label = "active"

            statuses.append(TippingPointStatus(
                name=tp["name"],
                proximity=round(proximity, 3),
                current_value=round(current_temp_anomaly if "threshold_c" in tp else proximity, 3),
                threshold=tp.get("threshold_c", tp.get("threshold_tension", 0.5)),
                gap=round(gap, 3),
                impact=tp["impact"],
                irreversibility=tp["irreversibility"],
                affected_domains=tp["domains"],
                time_horizon_years=time_horizon,
                early_warning_active=early_warning,
            ))

        irreversibility_avg = float(np.mean([s.irreversibility for s in statuses]))
        near_count = sum(1 for s in statuses if s.proximity > 0.7)
        active_count = sum(1 for s in statuses if s.proximity > 0.85)

        return sanitize_nan({
            "n_tipping_points": len(statuses),
            "irreversibility_avg": round(irreversibility_avg, 3),
            "near_or_closer": near_count,
            "active_or_breached": active_count,
            "tipping_points": [
                {
                    "name": s.name,
                    "proximity": s.proximity,
                    "status": "distant" if s.proximity < 0.5 else "approaching" if s.proximity < 0.7 else "near" if s.proximity < 0.85 else "imminent" if s.proximity < 0.95 else "active",
                    "threshold": s.threshold,
                    "gap": s.gap,
                    "impact": s.impact,
                    "irreversibility": s.irreversibility,
                    "affected_domains": s.affected_domains,
                    "time_horizon_years": s.time_horizon_years,
                    "early_warning_active": s.early_warning_active,
                }
                for s in sorted(statuses, key=lambda x: x.proximity, reverse=True)
            ],
            "highest_risk": statuses[0].name if statuses else "none",
            "recommendations": self._tipping_point_recommendations(statuses),
        })

    def generate_early_warnings(self, domain_scores: dict[str, float]) -> dict:
        warnings = []
        for category, signals in EARLY_WARNING_SIGNALS.items():
            for signal in signals:
                triggered = False
                sig_id = signal["indicator"]
                if "gt_0.7" in sig_id:
                    domain_key = sig_id.split("_gt_0.7")[0]
                    score = domain_scores.get(domain_key, 0.5)
                    risk = 1.0 - score
                    if risk > 0.7:
                        triggered = True
                elif "gt_0.6" in sig_id:
                    domain_key = sig_id.split("_gt_0.6")[0]
                    score = domain_scores.get(domain_key, 0.5)
                    risk = 1.0 - score
                    if risk > 0.6:
                        triggered = True
                elif "lt_0.3" in sig_id:
                    domain_key = sig_id.split("_lt_0.3")[0]
                    score = domain_scores.get(domain_key, 0.5)
                    if score < 0.3:
                        triggered = True
                elif "gt_2.0" in sig_id:
                    domain_key = sig_id.split("_gt_2.0")[0]
                    risk = 1.0 - domain_scores.get("economic", 0.5)
                    if risk > 0.6:
                        triggered = True
                elif "gt_0.5" in sig_id:
                    domain_key = sig_id.split("_gt_0.5")[0]
                    score = domain_scores.get(domain_key, 0.5)
                    if domain_key == "kessler_risk":
                        comp_risk = 1.0 - domain_scores.get("space", 0.5)
                        if comp_risk > 0.5:
                            triggered = True
                    elif score < 0.5:
                        triggered = True
                elif "gt_1.5" in sig_id or "gt_5mm" in sig_id:
                    temp_risk = 1.0 - domain_scores.get("climate", 0.5)
                    if temp_risk > 0.5:
                        triggered = True
                elif "pandemic_geopolitical_gt_0.6" in sig_id:
                    pandemic_risk = 1.0 - domain_scores.get("pandemic", 0.5)
                    geo_risk = 1.0 - domain_scores.get("geopolitical", 0.5)
                    if (pandemic_risk + geo_risk) / 2 > 0.6:
                        triggered = True
                elif "volcanic_food_gt_0.5" in sig_id:
                    seismic_risk = 1.0 - domain_scores.get("seismic", 0.5)
                    food_risk = 1.0 - domain_scores.get("food", 0.5)
                    if (seismic_risk + food_risk) / 2 > 0.5:
                        triggered = True
                elif "solar_cyber_infra_gt_0.5" in sig_id:
                    solar_risk = 1.0 - domain_scores.get("solar", 0.5)
                    cyber_risk = 1.0 - domain_scores.get("cyber", 0.5)
                    infra_risk = 1.0 - domain_scores.get("infrastructure", 0.5)
                    if (solar_risk + cyber_risk + infra_risk) / 3 > 0.5:
                        triggered = True
                elif "climate_migration_gt_0.6" in sig_id:
                    climate_risk = 1.0 - domain_scores.get("climate", 0.5)
                    social_risk = 1.0 - domain_scores.get("social", 0.5)
                    political_risk = 1.0 - domain_scores.get("political", 0.5)
                    if (climate_risk + social_risk + political_risk) / 3 > 0.6:
                        triggered = True

                if triggered:
                    severity = "critical" if signal.get("lead_time_months", 12) <= 3 else "high" if signal.get("lead_time_months", 12) <= 12 else "elevated"
                    warnings.append({
                        "signal_id": sig_id,
                        "category": category,
                        "description": signal["description"],
                        "lead_time_months": signal["lead_time_months"],
                        "associated_event": signal["event"],
                        "severity": severity,
                        "confidence": round(min(1.0, 0.6 + (1.0 - domain_scores.get(category, 0.5)) * 0.4), 2),
                    })

        active_count = len(warnings)
        critical_count = sum(1 for w in warnings if w["severity"] == "critical")

        return sanitize_nan({
            "n_active_warnings": active_count,
            "n_critical": critical_count,
            "n_high": sum(1 for w in warnings if w["severity"] == "high"),
            "n_elevated": sum(1 for w in warnings if w["severity"] == "elevated"),
            "overall_alert_level": "red" if critical_count > 0 else "orange" if active_count > 3 else "yellow" if active_count > 0 else "green",
            "warnings": sorted(warnings, key=lambda w: (0 if w["severity"] == "critical" else 1 if w["severity"] == "high" else 2)),
        })

    def generate_scenario(self, scenario: str = "great_filter",
                           domain_scores: dict[str, float] = None,
                           years: int = 50,
                           n_simulations: int = 500) -> dict:
        if domain_scores is None:
            domain_scores = {}

        rng = np.random.RandomState(self.seed)

        scenarios = {
            "great_filter": {
                "name": "The Great Filter",
                "description": "Multiple existential risks converging simultaneously — compound catastrophe that tests civilizational resilience",
                "trigger_events": ["carrington_event", "m9_megathrust", "pandemic_sars2"],
                "default_severity": 0.85,
                "domains": ["infrastructure", "economic", "pandemic", "climate", "social", "political"],
            },
            "slow_collapse": {
                "name": "Slow Collapse",
                "description": "Gradual multi-domain degradation over decades — insidious decline in institutional, ecological, and economic capacity",
                "trigger_events": ["biodiversity_tipping", "amoc_collapse", "food_crisis", "water_crisis"],
                "default_severity": 0.6,
                "domains": ["biodiversity", "climate", "food", "water", "political", "economic"],
            },
            "technological_disruption": {
                "name": "Technological Disruption",
                "description": "AI-driven economic transformation creating cascading social and institutional disruption faster than adaptation capacity",
                "trigger_events": ["ai_disruption", "global_cyber_attack", "supply_chain_collapse"],
                "default_severity": 0.5,
                "domains": ["technology", "economic", "social", "mental_health", "political", "cyber"],
            },
            "climate_cascade": {
                "name": "Climate Cascade",
                "description": "Climate tipping points triggering one another — permafrost → AMOC → ice sheet cascade with agricultural collapse",
                "trigger_events": ["climate_tipping", "amoc_collapse", "biodiversity_tipping", "food_crisis"],
                "default_severity": 0.8,
                "domains": ["climate", "oceanic", "atmospheric", "biodiversity", "food", "water", "paleoclimate"],
            },
            "geopolitical_spiral": {
                "name": "Geopolitical Spiral",
                "description": "Rising tensions triggering financial crisis, supply chain collapse, and potential military confrontation",
                "trigger_events": ["financial_crisis", "supply_chain_collapse", "nuclear_confrontation"],
                "default_severity": 0.7,
                "domains": ["geopolitical", "economic", "contagion", "supply_chain", "political", "mental_health"],
            },
            "pandemic_geopolitical_collapse": {
                "name": "Pandemic-Geopolitical Collapse",
                "description": "Pandemic reducing governance capacity while geopolitical tensions rise — compound crisis exceeding institutional response capability",
                "trigger_events": ["pandemic_sars2", "pandemic_geopolitical_cascade", "financial_crisis", "supply_chain_collapse"],
                "default_severity": 0.8,
                "domains": ["pandemic", "geopolitical", "economic", "supply_chain", "political", "social", "mental_health"],
            },
            "volcanic_winter": {
                "name": "Volcanic Winter",
                "description": "Major eruption causing multi-year global cooling, crop failures, and cascading food-water-energy crisis",
                "trigger_events": ["volcanic_winter_agricultural", "food_crisis", "water_crisis", "financial_crisis"],
                "default_severity": 0.85,
                "domains": ["seismic", "atmospheric", "climate", "food", "water", "energy", "economic"],
            },
            "solar_cyber_collapse": {
                "name": "Solar-Cyber-Infrastructure Collapse",
                "description": "Carrington-class solar event disabling grid infrastructure while cyber attack exploits recovery vulnerability — triple compound failure",
                "trigger_events": ["solar_cyber_infrastructure", "carrington_event", "global_cyber_attack", "supply_chain_collapse"],
                "default_severity": 0.9,
                "domains": ["solar", "cyber", "infrastructure", "technology", "economic", "supply_chain", "energy"],
            },
            "climate_refugee_conflict": {
                "name": "Climate Refugee Spiral",
                "description": "Climate-driven mass migration overwhelming borders, triggering xenophobic populism, interstate conflict, and democratic backsliding",
                "trigger_events": ["climate_migration_conflict", "climate_tipping", "food_crisis", "water_crisis", "nuclear_confrontation"],
                "default_severity": 0.7,
                "domains": ["climate", "social", "political", "geopolitical", "food", "water", "demographics"],
            },
        }

        config = scenarios.get(scenario, scenarios["great_filter"])
        base_severity = config["default_severity"]

        domain_modifier = 1.0
        for domain in config["domains"]:
            risk = 1.0 - domain_scores.get(domain, 0.5)
            domain_modifier *= (1.0 + risk * 0.15)
        adjusted_severity = min(1.0, base_severity * domain_modifier)

        cascade_prob = min(0.9, 0.3 + adjusted_severity * 0.4)
        trigger_probs = {}
        for event_id in config["trigger_events"]:
            tmpl = EVENT_TEMPLATES.get(event_id, {"base_rate_per_year": 0.05})
            base = tmpl["base_rate_per_year"]
            modifier = 1.0
            for d in tmpl["domains"]:
                modifier *= (1.0 + (1.0 - domain_scores.get(d, 0.5)) * 0.2)
            adjusted = min(0.15, base * modifier)
            trigger_probs[event_id] = round(adjusted, 4)

        results = []
        for _ in range(n_simulations):
            trajectory = []
            stability = 1.0 - adjusted_severity * 0.3
            for yr in range(years):
                annual_shock = rng.normal(0, 0.02)
                for event_id, prob in trigger_probs.items():
                    if rng.random() < prob:
                        impact = rng.uniform(0.05, 0.2) * adjusted_severity
                        annual_shock -= impact
                        stability -= impact * 0.5
                stability = min(1.0, max(0.0, stability + annual_shock + 0.001))
                trajectory.append(round(stability, 3))
            results.append(trajectory)

        final_stabilities = [t[-1] for t in results]
        min_stabilities = [min(t) for t in results]
        collapse_count = sum(1 for s in min_stabilities if s < 0.1)

        timeline = []
        step = max(1, years // 20)
        for yr in range(0, years, step):
            vals = [t[yr] for t in results if yr < len(t)]
            if vals:
                timeline.append({
                    "year": yr,
                    "mean_stability": round(float(np.mean(vals)), 3),
                    "p10": round(float(np.percentile(vals, 10)), 3),
                    "p50": round(float(np.percentile(vals, 50)), 3),
                    "p90": round(float(np.percentile(vals, 90)), 3),
                })

        return sanitize_nan({
            "scenario": scenario,
            "scenario_name": config["name"],
            "description": config["description"],
            "trigger_events": config["trigger_events"],
            "adjusted_severity": round(adjusted_severity, 3),
            "cascade_probability": round(cascade_prob, 3),
            "trigger_probabilities": trigger_probs,
            "mean_final_stability": round(float(np.mean(final_stabilities)), 3),
            "worst_case_stability": round(float(np.min(min_stabilities)), 3),
            "collapse_probability": round(collapse_count / n_simulations, 4),
            "timeline": timeline,
        })

    def predict_prescience(self, domain_scores: dict[str, float] = None,
                            current_temp_anomaly: float = 1.2) -> dict:
        if domain_scores is None:
            domain_scores = {}

        events = self.forecast_events(domain_scores, years=10, n_simulations=500)
        tipping = self.assess_tipping_points(domain_scores, current_temp_anomaly)
        warnings = self.generate_early_warnings(domain_scores)

        top_risks = events.get("top_5_risks", [])
        near_tipping = sum(1 for tp in tipping.get("tipping_points", []) if tp.get("proximity", 0) > 0.7)
        active_tipping = sum(1 for tp in tipping.get("tipping_points", []) if tp.get("proximity", 0) > 0.85)
        alert_level = warnings.get("overall_alert_level", "green")

        prescience_score = 0.5
        if events.get("any_catastrophic_probability", 0) > 0.5:
            prescience_score += 0.2
        if near_tipping > 0:
            prescience_score += near_tipping * 0.05
        if active_tipping > 0:
            prescience_score += 0.1
        if alert_level == "red":
            prescience_score += 0.15
        elif alert_level == "orange":
            prescience_score += 0.1
        elif alert_level == "yellow":
            prescience_score += 0.05
        prescience_score = min(1.0, max(0.0, prescience_score))

        if prescience_score < 0.35:
            outlook = "stable"
        elif prescience_score < 0.55:
            outlook = "cautious"
        elif prescience_score < 0.7:
            outlook = "elevated_risk"
        elif prescience_score < 0.85:
            outlook = "high_alert"
        else:
            outlook = "critical"

        return sanitize_nan({
            "prescience_score": round(prescience_score, 3),
            "outlook": outlook,
            "any_catastrophic_probability": events.get("any_catastrophic_probability", 0),
            "n_events_forecast": events.get("n_events_forecast", 0),
            "top_5_risks": top_risks,
            "n_tipping_points_near": near_tipping,
            "n_tipping_points_active": active_tipping,
            "highest_tipping_risk": tipping.get("highest_risk", "none"),
            "irreversibility_avg": tipping.get("irreversibility_avg", 0),
            "n_active_warnings": warnings.get("n_active_warnings", 0),
            "n_critical_warnings": warnings.get("n_critical", 0),
            "alert_level": alert_level,
            "key_concerns": [tp["name"] for tp in tipping.get("tipping_points", [])[:3] if tp.get("early_warning_active", False)],
            "recommendations": self._prescience_recommendations(prescience_score, near_tipping, alert_level),
        })

    def _tipping_point_recommendations(self, statuses: list) -> list:
        recs = []
        near = [s for s in statuses if s.proximity > 0.7]
        if near:
            names = [s.name for s in near[:3]]
            recs.append(f"Near-tipping points require immediate action: {', '.join(names)}")
        high_irrev = [s for s in statuses if s.irreversibility > 0.8]
        if high_irrev:
            recs.append(f"High-irreversibility points at risk: {', '.join(s.name for s in high_irrev[:3])} — prevention is the only option")
        active = [s for s in statuses if s.proximity > 0.85]
        if active:
            recs.append(f"CRITICAL: {', '.join(s.name for s in active)} tipping point(s) near or past threshold — transformative action needed")
        if not recs:
            recs.append("No immediate tipping point concerns — continue monitoring and building resilience")
        return recs[:5]

    def _prescience_recommendations(self, score: float, near_tipping: int, alert_level: str) -> list:
        recs = []
        if score > 0.7:
            recs.append("CRITICAL: Multiple convergence risks detected — establish cross-domain early warning system immediately")
        if near_tipping > 2:
            recs.append(f"{near_tipping} tipping points approaching threshold — prioritize decarbonization and ecosystem protection")
        if alert_level == "red":
            recs.append("RED ALERT: Imminent cascading failure risk — activate contingency protocols across all critical systems")
        if score > 0.5:
            recs.append("Elevated prescience score indicates compounding risks — increase system resilience margins")
        if score < 0.4:
            recs.append("Prescience score moderate — maintain monitoring, invest in redundancy and adaptive capacity")
        if not recs:
            recs.append("Prescience score low — stable conditions, continue monitoring and resilience building")
        return recs[:5]


def sanitize_nan(obj):
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or abs(obj) == float('inf'):
            return None
        return obj
    if isinstance(obj, (np.floating, np.float64)):
        v = float(obj)
        if v != v or abs(v) == float('inf'):
            return None
        return v
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj