"""Tests for Daily Oracle — real-time data-driven daily future prediction."""
import pytest
from oracle.prediction.daily_oracle import (
    DailyOracle, AlertLevel, TrendDirection, RegimeState,
    DataPoint, DomainForecast, AnomalyAlert, DayForecast, WeeklyOutlook,
)


def _make_dp(val=100, chg=0.5, chg1w=2.0, chg1m=5.0, vol=15.0, src="test"):
    return DataPoint(symbol="^GSPC", value=val, change_pct=chg, change_1w=chg1w,
                     change_1m=chg1m, volatility_20d=vol, source=src,
                     timestamp="2026-01-01T00:00:00", evidence="test")


def _make_market(vix=18, sp500=5500, sp_chg=0.5, gold=2000, oil=75, btc=65000, treasury=4.5, usd=103, nasdaq=18000):
    return {
        "^GSPC": DataPoint("^GSPC", sp500, sp_chg, sp_chg*5, sp_chg*20, 15, "yfinance", "2026-01-01", "US equity benchmark"),
        "^IXIC": DataPoint("^IXIC", nasdaq, sp_chg*1.2, sp_chg*5, sp_chg*20, 18, "yfinance", "2026-01-01", "US tech equity"),
        "^VIX": DataPoint("^VIX", vix, 0, 0, 0, 5, "yfinance", "2026-01-01", "Market fear index"),
        "GC=F": DataPoint("GC=F", gold, 0.1, 0.5, 2, 12, "yfinance", "2026-01-01", "Safe-haven commodity"),
        "CL=F": DataPoint("CL=F", oil, -0.2, -1, -3, 25, "yfinance", "2026-01-01", "Energy commodity"),
        "BTC-USD": DataPoint("BTC-USD", btc, 1.5, 5, 10, 60, "yfinance", "2026-01-01", "Digital asset"),
        "^TNX": DataPoint("^TNX", treasury, 0.01, 0.05, 0.1, 5, "yfinance", "2026-01-01", "Risk-free rate"),
        "DX-Y.NYB": DataPoint("DX-Y.NYB", usd, 0.1, 0.3, 1, 6, "yfinance", "2026-01-01", "Dollar strength"),
    }


class TestDailyOracleCore:
    def test_initialization(self):
        oracle = DailyOracle(use_llm=False, n_simulations=100)
        assert oracle.use_llm is False
        assert oracle.n_simulations == 100

    def test_divine_today_returns_result(self):
        oracle = DailyOracle(use_llm=False, n_simulations=50)
        result = oracle.divine_today()
        assert isinstance(result, DayForecast)

    def test_divine_today_has_date(self):
        oracle = DailyOracle(use_llm=False)
        result = oracle.divine_today()
        assert result.date is not None
        assert len(result.date) > 0

    def test_domain_forecasts(self):
        oracle = DailyOracle(use_llm=False)
        result = oracle.divine_today()
        assert len(result.domain_forecasts) >= 6
        for domain, fc in result.domain_forecasts.items():
            assert isinstance(fc, DomainForecast)
            assert 0 <= fc.current_state <= 1
            assert 0 <= fc.confidence <= 1

    def test_overall_score_in_range(self):
        oracle = DailyOracle(use_llm=False)
        result = oracle.divine_today()
        assert 0 <= result.overall_score <= 1

    def test_overall_trend(self):
        oracle = DailyOracle(use_llm=False)
        result = oracle.divine_today()
        assert result.overall_trend in [t for t in TrendDirection]

    def test_regime(self):
        oracle = DailyOracle(use_llm=False)
        result = oracle.divine_today()
        assert result.regime in [r for r in RegimeState]

    def test_confidence(self):
        oracle = DailyOracle(use_llm=False)
        result = oracle.divine_today()
        assert 0 <= result.confidence <= 1

    def test_evidence_count(self):
        oracle = DailyOracle(use_llm=False)
        result = oracle.divine_today()
        assert result.evidence_count >= 0


class TestAnomalyDetection:
    def test_no_anomalies_calm(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market(vix=14, sp_chg=0.3)
        forecasts = oracle._forecast_all_domains(market)
        alerts = oracle._detect_anomalies(market, forecasts)
        assert all(a.level != AlertLevel.CRITICAL for a in alerts)

    def test_high_vix_critical(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market(vix=35)
        forecasts = oracle._forecast_all_domains(market)
        alerts = oracle._detect_anomalies(market, forecasts)
        critical = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical) >= 1
        assert any("VIX" in a.symbol for a in critical)

    def test_vix_warning(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market(vix=27)
        forecasts = oracle._forecast_all_domains(market)
        alerts = oracle._detect_anomalies(market, forecasts)
        vix_alerts = [a for a in alerts if a.symbol == "^VIX"]
        assert any(a.level in (AlertLevel.WARNING, AlertLevel.CRITICAL) for a in vix_alerts)

    def test_sp500_crash_alert(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market(sp_chg=-4.0)
        forecasts = oracle._forecast_all_domains(market)
        alerts = oracle._detect_anomalies(market, forecasts)
        sp_alerts = [a for a in alerts if a.symbol == "^GSPC"]
        assert any(a.level == AlertLevel.CRITICAL for a in sp_alerts)

    def test_gold_rally_alert(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market(gold=2100)
        market["GC=F"] = DataPoint("GC=F", 2100, 0.5, 6.0, 8.0, 12, "yfinance", "2026-01-01", "Gold")
        forecasts = oracle._forecast_all_domains(market)
        alerts = oracle._detect_anomalies(market, forecasts)
        gold_alerts = [a for a in alerts if a.symbol == "GC=F"]
        assert len(gold_alerts) >= 1

    def test_oil_crash_alert(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market(oil=60)
        market["CL=F"] = DataPoint("CL=F", 60, -6.0, -8.0, -12.0, 30, "yfinance", "2026-01-01", "Oil")
        forecasts = oracle._forecast_all_domains(market)
        alerts = oracle._detect_anomalies(market, forecasts)
        oil_alerts = [a for a in alerts if a.symbol == "CL=F"]
        assert any(a.level == AlertLevel.WARNING for a in oil_alerts)


class TestRegimeClassification:
    def test_crisis_regime(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market(vix=40)
        forecasts = oracle._forecast_all_domains(market)
        regime = oracle._classify_regime(market, forecasts)
        assert regime == RegimeState.CRISIS

    def test_expansion_regime(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market(vix=14, sp_chg=1.0)
        forecasts = oracle._forecast_all_domains(market)
        regime = oracle._classify_regime(market, forecasts)
        assert regime in (RegimeState.EXPANSION, RegimeState.RECOVERY)

    def test_recession_regime(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market(vix=27)
        forecasts = oracle._forecast_all_domains(market)
        regime = oracle._classify_regime(market, forecasts)
        assert regime in (RegimeState.RECESSION, RegimeState.LATE_CYCLE, RegimeState.CRISIS)


class TestWeeklyOutlook:
    def test_weekly_outlook(self):
        oracle = DailyOracle(use_llm=False)
        daily = oracle.divine_today()
        outlook = oracle.weekly_outlook(daily)
        assert isinstance(outlook, WeeklyOutlook)
        assert len(outlook.recommended_actions) > 0
        assert len(outlook.attention_domains) > 0

    def test_regime_probabilities(self):
        oracle = DailyOracle(use_llm=False)
        daily = oracle.divine_today()
        outlook = oracle.weekly_outlook(daily)
        total_prob = sum(outlook.regime_probability.values())
        assert 0.9 <= total_prob <= 1.1

    def test_portfolio_hint(self):
        oracle = DailyOracle(use_llm=False)
        daily = oracle.divine_today()
        outlook = oracle.weekly_outlook(daily)
        assert len(outlook.portfolio_hint) > 20

    def test_health_hint(self):
        oracle = DailyOracle(use_llm=False)
        daily = oracle.divine_today()
        outlook = oracle.weekly_outlook(daily)
        assert len(outlook.health_hint) > 10


class TestNarrativeGeneration:
    def test_narrative_not_empty(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market()
        forecasts = oracle._forecast_all_domains(market)
        alerts = oracle._detect_anomalies(market, forecasts)
        regime = oracle._classify_regime(market, forecasts)
        narrative = oracle._generate_narrative(forecasts, alerts, regime, market)
        assert len(narrative) > 30

    def test_narrative_includes_vix(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market(vix=22)
        forecasts = oracle._forecast_all_domains(market)
        alerts = oracle._detect_anomalies(market, forecasts)
        regime = oracle._classify_regime(market, forecasts)
        narrative = oracle._generate_narrative(forecasts, alerts, regime, market)
        assert "VIX" in narrative or "vix" in narrative.lower()


class TestDomainForecasts:
    def test_equity_forecast(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market()
        fc = oracle._forecast_equity(market["^GSPC"], market["^IXIC"], market["^VIX"])
        assert 0 <= fc.current_state <= 1
        assert fc.confidence > 0

    def test_volatility_forecast(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market()
        fc = oracle._forecast_volatility(market["^VIX"])
        assert fc.current_state > 0

    def test_all_domains_present(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market()
        forecasts = oracle._forecast_all_domains(market)
        expected = ["equity", "volatility", "commodity", "crypto", "bonds", "currency",
                     "geopolitical", "sentiment", "climate", "technology", "economic", "health",
                     "inflation", "labor", "monetary"]
        for d in expected:
            assert d in forecasts, f"Missing domain: {d}"


class TestEnums:
    def test_alert_levels(self):
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.CRITICAL.value == "critical"

    def test_trend_directions(self):
        assert TrendDirection.UP.value == "up"
        assert TrendDirection.ACCELERATING.value == "accelerating"

    def test_regime_states(self):
        assert RegimeState.EXPANSION.value == "expansion"
        assert RegimeState.CRISIS.value == "crisis"


class TestFREDIntegration:
    def test_fred_fetch_returns_data(self):
        oracle = DailyOracle(use_llm=False)
        fred = oracle._fetch_fred_data()
        assert isinstance(fred, dict)
        assert len(fred) == 5
        for sid in ["DFF", "CPIAUCSL", "UNRATE", "T10Y2Y", "UMCSENT"]:
            assert sid in fred, f"Missing FRED series: {sid}"
            dp = fred[sid]
            assert isinstance(dp, DataPoint)
            assert dp.symbol == sid

    def test_inflation_forecast(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market()
        forecasts = oracle._forecast_all_domains(market)
        assert "inflation" in forecasts
        inf = forecasts["inflation"]
        assert 0 <= inf.current_state <= 1
        assert inf.confidence > 0
        assert len(inf.key_drivers) > 0

    def test_labor_forecast(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market()
        forecasts = oracle._forecast_all_domains(market)
        assert "labor" in forecasts
        lab = forecasts["labor"]
        assert 0 <= lab.current_state <= 1
        assert lab.confidence > 0

    def test_monetary_forecast(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market()
        forecasts = oracle._forecast_all_domains(market)
        assert "monetary" in forecasts
        mon = forecasts["monetary"]
        assert 0 <= mon.current_state <= 1
        assert mon.confidence > 0
        assert any("yield" in d.lower() or "ffr" in d.lower() or "curve" in d.lower() for d in mon.key_drivers)

    def test_fred_anomaly_yield_curve_inversion(self):
        oracle = DailyOracle(use_llm=False)
        market = _make_market()
        oracle._fred_cache = {
            "T10Y2Y": DataPoint(symbol="T10Y2Y", value=-0.5, change_pct=-2.0, change_1w=-1.5,
                                 change_1m=-3.0, volatility_20d=0, source="fred",
                                 timestamp="2025-01-01", evidence="Yield curve spread"),
            "DFF": DataPoint(symbol="DFF", value=5.33, change_pct=0, change_1w=0, change_1m=0,
                             volatility_20d=0, source="fred", timestamp="2025-01-01",
                             evidence="Federal funds rate"),
            "UNRATE": DataPoint(symbol="UNRATE", value=3.7, change_pct=0, change_1w=0, change_1m=0,
                                volatility_20d=0, source="fred", timestamp="2025-01-01",
                                evidence="Unemployment rate"),
            "CPIAUCSL": DataPoint(symbol="CPIAUCSL", value=314, change_pct=0.3, change_1w=0.1,
                                  change_1m=0.4, volatility_20d=0, source="fred",
                                  timestamp="2025-01-01", evidence="Consumer price index"),
            "UMCSENT": DataPoint(symbol="UMCSENT", value=68, change_pct=0, change_1w=0,
                                 change_1m=0, volatility_20d=0, source="fred",
                                 timestamp="2025-01-01", evidence="Consumer sentiment"),
        }
        forecasts = oracle._forecast_all_domains(market)
        alerts = oracle._detect_anomalies(market, forecasts)
        yc_alerts = [a for a in alerts if a.symbol == "T10Y2Y"]
        assert len(yc_alerts) >= 1
        assert yc_alerts[0].level == AlertLevel.CRITICAL

    def test_fallback_fred_data(self):
        oracle = DailyOracle(use_llm=False)
        oracle._fred_cache = {}
        market = _make_market()
        forecasts = oracle._forecast_all_domains(market)
        assert "inflation" in forecasts
        assert "labor" in forecasts
        assert "monetary" in forecasts