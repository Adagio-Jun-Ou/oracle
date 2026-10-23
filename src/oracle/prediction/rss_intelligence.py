"""RSS Intelligence Feed — Real-time event awareness for CausalEventPredictor.

Fetches headlines from 163+ OSINT RSS sources, extracts relevant events
using keyword matching, and updates domain states, trigger conditions, and
invalidation evaluations based on current world events.

Sources: Conflict Intel Map OSINT_SOURCES (597 feeds across 9 tiers),
plus financial/tech RSS feeds for market-moving events.
"""
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from oracle.prediction.live_calibration import CausalEventLiveDataCalibrator
    LIVE_DATA_AVAILABLE = True
except ImportError:
    LIVE_DATA_AVAILABLE = False


@dataclass
class RSSArticle:
    title: str
    link: str
    published: str
    source: str
    source_tier: int
    source_reliability: int
    category: str
    fetched_at: str


@dataclass
class EventSignal:
    event_id: str
    signal_type: str
    headline: str
    source: str
    source_reliability: int
    domain_impact: Dict[str, float]
    probability_adjustment: float
    timestamp: str
    keywords_matched: List[str]


@dataclass
class RSSCalibrationResult:
    articles_fetched: int
    sources_checked: int
    signals: List[EventSignal]
    domain_state_adjustments: Dict[str, float]
    trigger_updates: Dict[str, List[EventSignal]]
    invalidation_alerts: List[EventSignal]
    already_occurred: List[str]
    headline_summary: Dict[str, List[str]]
    timestamp: str


EVENT_KEYWORD_MAP = {
    "fed_rate_cut_2026": {
        "positive": ["fed cut", "rate cut", "fomc dovish", "lower interest rates", "rate reduction", "easing cycle", "fed pivot", "powell dovish", "fed funds rate cut"],
        "negative": ["fed hike", "rate increase", "hawkish fed", "higher for longer", "inflation resilient", "no rate cut", "fed on hold", "inflation sticky"],
    },
    "sp500_new_high_2026": {
        "positive": ["s&p 500 record", "s&p high", "stock market rally", "bull market", "all-time high", "equity surge", "market gains", "tech rally", "ai boom"],
        "negative": ["market crash", "s&p decline", "bear market", "correction", "market sell-off", "recession fears", "trade war", "geopolitical shock"],
    },
    "bitcoin_100k_2026": {
        "positive": ["bitcoin rally", "crypto surge", "bitcoin etf inflow", "btc all-time high", "bitcoin adoption", "institutional crypto", "halving cycle"],
        "negative": ["crypto crash", "bitcoin ban", "sec crypto", "crypto regulation", "bitcoin hack", "stablecoin collapse"],
    },
    "gold_3000_2026": {
        "positive": ["gold record", "gold rally", "gold all-time high", "safe haven demand", "central bank gold", "gold price surge"],
        "negative": ["gold decline", "gold selloff", "risk-on sentiment", "dollar strength"],
    },
    "us_recession_avoided_2026": {
        "positive": ["soft landing", "economic growth", "gdp growth", "labor market strong", "employment rise", "consumer spending"],
        "negative": ["recession", "gdp contraction", "layoffs surge", "consumer confidence drop", "yield curve inversion", "jobless claims rise"],
    },
    "geopolitical_taiwan_crisis_no_war_2026": {
        "positive": ["taiwan status quo", "us-china dialogue", "diplomatic talks", "taiwan strait calm"],
        "negative": ["taiwan tensions", "military exercises", "pla drills", "taiwan strait crisis", "china taiwan threat", "military confrontation"],
    },
    "climate_first_15c_year_2026": {
        "positive": ["record heat", "hottest year", "temperature record", "climate warming", "el nino", "arctic ice low", "wildfire record"],
        "negative": ["la nina cooling", "volcanic eruption", "global cooling", "arctic ice recovery"],
    },
    "ozempic_obesity_decline_2026": {
        "positive": ["obesity decline", "weight loss drug", "glp-1", "ozempic weight", "wegovy results", "obesity rate drop", "medication weight loss"],
        "negative": ["ozempic side effects", "glp-1 risk", "weight loss plateau", "obesity rate stagnant"],
    },
    "first_agi_benchmark_pass_2026": {
        "positive": ["agi breakthrough", "ai passes benchmark", "arc-agi", "mmlu pro", "gpt-5", "claude 4", "frontier model", "major ai advance", "superintelligence"],
        "negative": ["ai safety incident", "regulation halts ai", "scaling law plateau", "ai winter"],
    },
    "ai_regulation_us_eu_2026": {
        "positive": ["ai regulation", "ai act", "ai safety bill", "legislation ai", "regulate artificial intelligence", "ai governance"],
        "negative": ["deregulation", "tech lobby", "no ai law", "regulation blocked"],
    },
    "nuclear_near_miss_2026": {
        "positive": [],
        "negative": ["nuclear threat", "defcon", "nuclear escalation", "tactical nuclear", "nuclear alert", "missile test north korea", "nuclear standoff", "atomic energy warning", "deterrence failure"],
    },
    "mental_health_crisis_2026": {
        "positive": ["mental health crisis", "depression rate", "anxiety epidemic", "loneliness epidemic", "therapy shortage", "suicide rate"],
        "negative": ["mental health parity", "therapy access expansion", "destigmatization", "mental health funding"],
    },
    "democracy_stress_index_2026": {
        "positive": ["democratic backsliding", "authoritarian", "democracy decline", "press freedom decline", "judicial independence", "election dispute"],
        "negative": ["democratic renewal", "voter turnout record", "transparency legislation"],
    },
    "home_prices_decline_top20_2026": {
        "positive": ["home price decline", "housing crash", "affordability crisis", "mortgage rate high", "housing inventory surge", "home sales drop"],
        "negative": ["housing recovery", "home price gain", "mortgage rate cut", "homebuilder confidence"],
    },
    "ai_job_displacement_5pct_2026": {
        "positive": ["ai replace jobs", "layoffs ai", "automation displacement", "job cut ai", "white-collar automation", "ai workforce reduction"],
        "negative": ["ai create jobs", "new ai roles", "reskilling", "job creation offset"],
    },
    "student_debt_crisis_2026": {
        "positive": ["student loan default", "delinquency rate rise", "student debt crisis", "borrower distress", "save plan blocked"],
        "negative": ["student debt cancellation", "idr expansion", "borrower relief", "loan forgiveness"],
    },
    "longevity_breakthrough_2026": {
        "positive": ["longevity breakthrough", "life extension study", "senolytics trial", "rapamycin lifespan", "anti-aging", "biological age reversal", "healthspan extension"],
        "negative": ["clinical trial failure", "safety concerns longevity", "aging drug halted"],
    },
    "pandemic_preparedness_treaty_2026": {
        "positive": ["pandemic treaty", "who accord", "pandemic agreement", "global health security", "pathogen access"],
        "negative": ["treaty blocked", "who treaty rejected", "pharma oppose treaty"],
    },
    "quantum_advantage_useful_2026": {
        "positive": ["quantum advantage", "quantum breakthrough", "quantum error correction", "logical qubit", "quantum milestone"],
        "negative": ["quantum plateau", "classical algorithm beats quantum"],
    },
    "longevity_escape_velocity_2030": {
        "positive": ["longevity escape velocity", "lev research", "mortality reversal", "aging clock reversal", "human lifespan extension"],
        "negative": ["diminishing returns longevity", "aging intervention failure"],
    },
}

FINANCIAL_KEYWORDS = {
    "recession": {"domain_impact": {"equity": -0.05, "sentiment": -0.10, "labor": 0.10}, "events": ["us_recession_avoided_2026"]},
    "inflation": {"domain_impact": {"inflation": 0.05}, "events": ["fed_rate_cut_2026"]},
    "rate cut": {"domain_impact": {"equity": 0.05, "inflation": -0.03}, "events": ["fed_rate_cut_2026"]},
    "rate hike": {"domain_impact": {"equity": -0.08, "inflation": 0.05}, "events": ["fed_rate_cut_2026"]},
    "bull market": {"domain_impact": {"equity": 0.05, "sentiment": 0.08}, "events": ["sp500_new_high_2026"]},
    "bear market": {"domain_impact": {"equity": -0.10, "sentiment": -0.15, "volatility": 0.15}, "events": ["sp500_new_high_2026"]},
    "volatility": {"domain_impact": {"volatility": 0.05}, "events": []},
    "geopolitical": {"domain_impact": {"geopolitical": 0.05}, "events": ["geopolitical_taiwan_crisis_no_war_2026"]},
    "nuclear": {"domain_impact": {"geopolitical": 0.15}, "events": ["nuclear_near_miss_2026"]},
    "ai breakthrough": {"domain_impact": {"technology": 0.10}, "events": ["first_agi_benchmark_pass_2026"]},
    "ai regulation": {"domain_impact": {"technology": -0.03}, "events": ["ai_regulation_us_eu_2026"]},
    "climate": {"domain_impact": {"climate": 0.05}, "events": ["climate_first_15c_year_2026"]},
    "housing": {"domain_impact": {"housing": 0.05}, "events": ["home_prices_decline_top20_2026"]},
    "gold record": {"domain_impact": {"commodity": 0.05, "geopolitical": 0.03}, "events": ["gold_3000_2026"]},
    "bitcoin": {"domain_impact": {"technology": 0.02}, "events": ["bitcoin_100k_2026"]},
    "mental health": {"domain_impact": {"health": 0.03}, "events": ["mental_health_crisis_2026"]},
}

ALREADY_OCCURRED_UPDATER = {
    "gold_3000_2026": lambda price, _: price > 3000,
    "sp500_new_high_2026": lambda price, _: price > 6200,
    "bitcoin_100k_2026": lambda price, _: price > 100000,
    "fed_rate_cut_2026": lambda _, data: data.get("federal_funds_rate", 5.3) < 4.5,
    "yield_curve_uninvert_2026": lambda _, data: data.get("treasury_10y", 4.3) > data.get("treasury_2y", 4.1),
}


class RSSIntelligenceFeed:
    def __init__(self, max_articles_per_source: int = 10, timeout: int = 15):
        self.max_articles = max_articles_per_source
        self.timeout = timeout
        self._cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 1800

    def fetch_headlines(self, categories: Optional[List[str]] = None) -> List[RSSArticle]:
        articles = []
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not available; cannot fetch RSS feeds")
            return articles
        sources = self._get_sources(categories)
        for source in sources:
            try:
                feed = feedparser.parse(source["url"], etag=source.get("etag"), modified=source.get("modified"))
                if not feed.entries:
                    continue
                for entry in feed.entries[:self.max_articles]:
                    title = entry.get("title", "")
                    link = entry.get("link", "")
                    published = entry.get("published", entry.get("updated", ""))
                    if title:
                        articles.append(RSSArticle(
                            title=title,
                            link=link,
                            published=published,
                            source=source["name"],
                            source_tier=source.get("tier", 3),
                            source_reliability=source.get("reliability", 50),
                            category=source.get("category", "general"),
                            fetched_at=datetime.now().isoformat(),
                        ))
            except Exception as e:
                logger.debug(f"Failed to fetch {source['name']}: {e}")
                continue
        return articles

    def analyze_articles(self, articles: List[RSSArticle],
                        event_ids: Optional[List[str]] = None) -> RSSCalibrationResult:
        target_ids = event_ids or list(EVENT_KEYWORD_MAP.keys())
        signals = []
        domain_adjustments = {}
        trigger_updates = {}
        invalidation_alerts = []
        already_occurred = []
        headline_summary = {}

        for article in articles:
            title_lower = article.title.lower()
            for event_id in target_ids:
                kw_map = EVENT_KEYWORD_MAP.get(event_id, {})
                for kw_type, keywords in kw_map.items():
                    for kw in keywords:
                        if kw in title_lower:
                            reliability_weight = article.source_reliability / 100.0
                            adj = 0.02 * reliability_weight
                            if kw_type == "negative":
                                adj = -adj
                            signals.append(EventSignal(
                                event_id=event_id,
                                signal_type=kw_type,
                                headline=article.title,
                                source=article.source,
                                source_reliability=article.source_reliability,
                                domain_impact={},
                                probability_adjustment=adj,
                                timestamp=article.published or article.fetched_at,
                                keywords_matched=[kw],
                            ))
                            break

            for fin_kw, impact in FINANCIAL_KEYWORDS.items():
                if fin_kw in title_lower:
                    for domain, delta in impact["domain_impact"].items():
                        domain_adjustments[domain] = domain_adjustments.get(domain, 0.0) + delta * (article.source_reliability / 100.0)
                    category = fin_kw.replace(" ", "_")
                    headline_summary.setdefault(category, []).append(article.title)

            for kw_type in ["nuclear", "war", "military conflict", "invasion", "attack"]:
                if kw_type in title_lower:
                    if article.source_reliability >= 70:
                        invalidation_alerts.append(EventSignal(
                            event_id="nuclear_near_miss_2026" if "nuclear" in title_lower else "geopolitical_taiwan_crisis_no_war_2026",
                            signal_type="negative",
                            headline=article.title,
                            source=article.source,
                            source_reliability=article.source_reliability,
                            domain_impact={"geopolitical": 0.05},
                            probability_adjustment=0.02,
                            timestamp=article.published or article.fetched_at,
                            keywords_matched=[kw_type],
                        ))

        for signal in signals:
            trigger_updates.setdefault(signal.event_id, []).append(signal)

        live_data = {}
        market = None
        if LIVE_DATA_AVAILABLE:
            try:
                calibrator = CausalEventLiveDataCalibrator()
                market = calibrator.market
                if market:
                    live_data["sp500"] = market.get_latest_value("sp500") or 0
                    live_data["gold"] = market.get_latest_value("gold") or 0
                    live_data["bitcoin"] = market.get_latest_value("bitcoin") or 0
                    live_data["treasury_10y"] = market.get_latest_value("treasury_10y") or 0
                    live_data["treasury_2y"] = market.get_latest_value("treasury_2y") or 0
                    live_data["vix"] = market.get_latest_value("vix") or 0
                econ = calibrator.econ
                if econ:
                    fred_data = econ.get_latest_values(["federal_funds_rate"])
                    live_data.update(fred_data)
            except Exception as e:
                logger.warning(f"Live data fetch failed: {e}")

        for event_id, check_fn in ALREADY_OCCURRED_UPDATER.items():
            try:
                if check_fn(live_data.get("sp500", 0), live_data):
                    pass
                sp = live_data.get("sp500", 0)
                gold = live_data.get("gold", 0)
                btc = live_data.get("bitcoin", 0)
                events_met = {
                    "gold_3000_2026": gold > 3000 if gold else False,
                    "sp500_new_high_2026": sp > 6200 if sp else False,
                    "bitcoin_100k_2026": btc > 100000 if btc else False,
                }
                if events_met.get(event_id, False):
                    already_occurred.append(event_id)
            except Exception:
                pass

        gold = live_data.get("gold", 0)
        btc = live_data.get("bitcoin", 0)
        sp = live_data.get("sp500", 0)
        if gold and gold > 3000:
            if "gold_3000_2026" not in already_occurred:
                already_occurred.append("gold_3000_2026")
        if sp and sp > 6200:
            if "sp500_new_high_2026" not in already_occurred:
                already_occurred.append("sp500_new_high_2026")
        if btc and btc > 100000:
            if "bitcoin_100k_2026" not in already_occurred:
                already_occurred.append("bitcoin_100k_2026")

        return RSSCalibrationResult(
            articles_fetched=len(articles),
            sources_checked=len(set(a.source for a in articles)) if articles else 0,
            signals=signals,
            domain_state_adjustments=domain_adjustments,
            trigger_updates=trigger_updates,
            invalidation_alerts=invalidation_alerts,
            already_occurred=already_occurred,
            headline_summary=headline_summary,
            timestamp=datetime.now().isoformat(),
        )

    def _get_sources(self, categories: Optional[List[str]] = None) -> list:
        sources = []
        try:
            from conflict_intel_map.osint_sources import OSINT_SOURCES, get_all_osint_feeds
            all_feeds = get_all_osint_feeds()
            if categories:
                sources = [f for f in all_feeds if f.get("category") in categories or f.get("tier", 3) <= 2]
            else:
                sources = [f for f in all_feeds if f.get("tier", 3) <= 2]
        except ImportError:
            pass

        financial_sources = [
            {"name": "Reuters Business", "url": "https://feeds.reuters.com/reuters/businessNews", "reliability": 95, "tier": 1, "category": "financial"},
            {"name": "Reuters World", "url": "https://feeds.reuters.com/reuters/worldNews", "reliability": 95, "tier": 1, "category": "world"},
            {"name": "Reuters Top News", "url": "https://feeds.reuters.com/reuters/topNews", "reliability": 95, "tier": 1, "category": "world"},
            {"name": "BBC World", "url": "http://feeds.bbci.co.uk/news/world/rss.xml", "reliability": 90, "tier": 1, "category": "world"},
            {"name": "BBC Business", "url": "http://feeds.bbci.co.uk/news/business/rss.xml", "reliability": 90, "tier": 1, "category": "financial"},
            {"name": "BBC Technology", "url": "http://feeds.bbci.co.uk/news/technology/rss.xml", "reliability": 90, "tier": 1, "category": "technology"},
            {"name": "BBC Science", "url": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml", "reliability": 90, "tier": 1, "category": "science"},
            {"name": "BBC Health", "url": "http://feeds.bbci.co.uk/news/health/rss.xml", "reliability": 90, "tier": 1, "category": "health"},
            {"name": "NYT World", "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "reliability": 88, "tier": 1, "category": "world"},
            {"name": "NYT Business", "url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "reliability": 88, "tier": 1, "category": "financial"},
            {"name": "NYT Tech", "url": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml", "reliability": 88, "tier": 1, "category": "technology"},
            {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml", "reliability": 82, "tier": 1, "category": "world"},
            {"name": "The Guardian World", "url": "https://www.theguardian.com/world/rss", "reliability": 85, "tier": 1, "category": "world"},
            {"name": "The Guardian Business", "url": "https://www.theguardian.com/business/rss", "reliability": 85, "tier": 1, "category": "financial"},
            {"name": "The Guardian Tech", "url": "https://www.theguardian.com/technology/rss", "reliability": 85, "tier": 1, "category": "technology"},
            {"name": "AP News", "url": "https://rsshub.app/apnews/topics/all", "reliability": 95, "tier": 1, "category": "world"},
            {"name": "Economist", "url": "https://www.economist.com/rss", "reliability": 88, "tier": 1, "category": "economic"},
            {"name": "CNBC", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147", "reliability": 82, "tier": 1, "category": "financial"},
            {"name": "MarketWatch", "url": "http://feeds.marketwatch.com/marketwatch/topstories/", "reliability": 80, "tier": 1, "category": "financial"},
            {"name": "FT", "url": "https://www.ft.com/rss/home", "reliability": 90, "tier": 1, "category": "financial"},
            {"name": "WSJ", "url": "https://feeds.content.dowjones.io/rss/RSSWorldNews.xml", "reliability": 90, "tier": 1, "category": "financial"},
            {"name": "NPR News", "url": "https://feeds.npr.org/1001/rss-feed.html", "reliability": 85, "tier": 1, "category": "world"},
            {"name": "Nature", "url": "https://www.nature.com/nature.rss", "reliability": 95, "tier": 1, "category": "science"},
            {"name": "Science Daily", "url": "https://www.sciencedaily.com/rss/all.xml", "reliability": 80, "tier": 2, "category": "science"},
            {"name": "TechCrunch", "url": "https://techcrunch.com/feed/", "reliability": 72, "tier": 2, "category": "technology"},
            {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/index", "reliability": 78, "tier": 2, "category": "technology"},
            {"name": "Wired", "url": "https://www.wired.com/feed/rss", "reliability": 75, "tier": 2, "category": "technology"},
            {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml", "reliability": 72, "tier": 2, "category": "technology"},
            {"name": "WHO News", "url": "https://www.who.int/rss-feeds/news-english.xml", "reliability": 95, "tier": 1, "category": "health"},
            {"name": "CDC News", "url": "https://www.cdc.gov/rss/cdcwh.xml", "reliability": 95, "tier": 1, "category": "health"},
        ]
        sources.extend(financial_sources)
        return sources

    def get_combined_domain_states(self, base_states: Optional[Dict[str, float]] = None,
                                  categories: Optional[List[str]] = None) -> Dict[str, float]:
        if base_states is None:
            if LIVE_DATA_AVAILABLE:
                try:
                    calibrator = CausalEventLiveDataCalibrator()
                    base_states = calibrator._compute_domain_states()
                except Exception:
                    base_states = {"equity": 0.55, "volatility": 0.40, "inflation": 0.50,
                                   "geopolitical": 0.50, "health": 0.50, "technology": 0.55,
                                   "sentiment": 0.50, "labor": 0.50, "currency": 0.50,
                                   "commodity": 0.50, "climate": 0.50, "housing": 0.50}
        result = dict(base_states)
        articles = self.fetch_headlines(categories)
        analysis = self.analyze_articles(articles)
        for domain, delta in analysis.domain_state_adjustments.items():
            result[domain] = max(0.01, min(0.99, result.get(domain, 0.5) + delta))
        return result