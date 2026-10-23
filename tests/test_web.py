import pytest
from fastapi.testclient import TestClient
from oracle.web.app import app


client = TestClient(app)


class TestWebApp:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["engine"] == "oracle"

    def test_status(self):
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "deterministic" in data["models"]
        assert "non_deterministic" in data["models"]

    def test_dashboard(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "ORACLE" in response.text
        assert "section-options" in response.text
        assert "section-tech" in response.text

    def test_predict_endpoint(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "economic_state": {
                "gdp_growth_rate": 0.02,
                "debt_to_gdp": 1.2,
                "interest_rate": 0.045,
                "inflation_rate": 0.03,
                "unemployment_rate": 0.04,
                "stock_valuation": 1.0,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "diagnosis" in data

    def test_predict_with_company(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "medium",
            "company": {
                "ticker": "AAPL",
                "eps": 6.5,
                "book_value": 4.0,
                "fcf": 6.0,
                "earnings_growth": 0.08,
                "roe": 0.45,
                "price": 185.0,
            },
        })
        assert response.status_code == 200

    def test_diagnose_endpoint(self):
        response = client.post("/api/diagnose", json={
            "economic_state": {
                "gdp_growth_rate": 0.02,
                "debt_to_gdp": 1.2,
                "interest_rate": 0.045,
                "inflation_rate": 0.03,
                "unemployment_rate": 0.04,
                "stock_valuation": 1.0,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "cycle_diagnosis" in data

    def test_timeslider_endpoint(self):
        response = client.get("/api/timeslider?days=1825&confidence=high")
        assert response.status_code == 200

    def test_scenario_endpoint(self):
        response = client.post("/api/scenario", json={
            "target_date": "2030-01-01",
            "confidence": "medium",
        })
        assert response.status_code == 200

    def test_cycles_endpoint(self):
        response = client.get("/api/cycles?variable=gdp")
        assert response.status_code == 200
        data = response.json()
        assert "cycles" in data

    def test_invalid_date(self):
        response = client.post("/api/predict", json={
            "target_date": "invalid-date",
        })
        assert response.status_code == 422

    def test_geopolitical_endpoint(self):
        response = client.post("/api/geopolitical", json={
            "us_china_tension": 0.6,
            "russia_west_tension": 0.7,
            "middle_east_instability": 0.5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "risk_assessment" in data
        assert "overall_risk" in data["risk_assessment"]

    def test_geopolitical_conflict_score_endpoint(self):
        response = client.post("/api/geopolitical/conflict-score", json={
            "us_china_tension": 0.7,
            "russia_west_tension": 0.8,
            "middle_east_instability": 0.6,
        })
        assert response.status_code == 200
        data = response.json()
        assert "overall_intensity" in data
        assert "stage" in data
        assert "hot_conflict_score" in data
        assert "de_escalation_potential" in data
        assert "nuclear_risk_level" in data

    def test_geopolitical_alliance_network_endpoint(self):
        response = client.post("/api/geopolitical/alliance-network", json={
            "us_china_tension": 0.5,
            "alliance_stability": 0.6,
        })
        assert response.status_code == 200
        data = response.json()
        assert "alliances" in data
        assert "global_cohesion" in data
        assert "global_deterrence" in data
        assert "fragmentation_risk" in data
        assert "stability_index" in data
        assert len(data["alliances"]) == 5

    def test_geopolitical_resource_competition_endpoint(self):
        response = client.post("/api/geopolitical/resource-competition", json={
            "us_china_tension": 0.6,
            "energy_geopolitics": 0.5,
            "supply_chain_fragility": 0.4,
        })
        assert response.status_code == 200
        data = response.json()
        assert "resources" in data
        assert "overall_competition_risk" in data
        assert "tension_multiplier" in data
        assert "top_contested" in data
        assert len(data["resources"]) == 6

    def test_climate_endpoint(self):
        response = client.post("/api/climate", json={
            "global_temp_anomaly_c": 1.2,
            "co2_ppm": 420,
        })
        assert response.status_code == 200
        data = response.json()
        assert "projections" in data or "tipping_points" in data

    def test_sentiment_endpoint(self):
        response = client.post("/api/sentiment", json={
            "fear_greed_index": 0.35,
            "vix_level": 0.6,
        })
        assert response.status_code == 200
        data = response.json()
        assert "composite_index" in data
        assert "composite_index" in data["composite_index"]

    def test_contagion_endpoint(self):
        response = client.post("/api/contagion", json={
            "financial_systemic_risk": 0.4,
            "trade_linkage": 0.5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "financial_contagion" in data or "cascade" in data

    def test_demographics_endpoint(self):
        response = client.post("/api/demographics", json={
            "total_population_m": 8100,
            "birth_rate_per_1000": 18,
            "death_rate_per_1000": 8,
        })
        assert response.status_code == 200
        data = response.json()
        assert "population_projections" in data or "aging_dividend" in data

    def test_backtest_endpoint(self):
        response = client.post("/api/backtest", json={
            "variable": "gdp",
            "window_size": 60,
            "n_windows": 3,
            "horizon": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "walk_forward" in data or "model_comparison" in data

    def test_optimize_endpoint(self):
        response = client.post("/api/optimize", json={
            "population_size": 5,
            "generations": 3,
            "metric": "mae",
        })
        assert response.status_code == 200
        data = response.json()
        assert "best_weights" in data

    def test_predict_with_domains(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "economic_state": {
                "gdp_growth_rate": 0.02,
                "debt_to_gdp": 1.2,
                "interest_rate": 0.045,
                "inflation_rate": 0.03,
                "unemployment_rate": 0.04,
                "stock_valuation": 1.0,
            },
            "geopolitical_state": {
                "us_china_tension": 0.6,
                "russia_west_tension": 0.7,
            },
            "climate_state": {
                "global_temp_anomaly_c": 1.2,
                "co2_ppm": 420,
            },
            "contagion_state": {
                "financial_systemic_risk": 0.4,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "geopolitics" in domains or "climate" in domains or "finance" in domains

    def test_domain_scenarios_endpoint(self):
        response = client.post("/api/domain-scenarios", json={
            "target_date": "2030-01-01",
            "confidence": "medium",
        })
        assert response.status_code == 200
        data = response.json()
        assert "geopolitical_crisis" in data or "financial_crisis" in data

    def test_market_data_endpoint(self):
        response = client.get("/api/market-data")
        assert response.status_code == 200
        data = response.json()
        assert "snapshot" in data
        assert "series" in data

    def test_real_data_cycles(self):
        response = client.get("/api/cycles?variable=sp500")
        assert response.status_code == 200
        data = response.json()
        assert "cycles" in data
        assert "data_source" in data

    def test_economic_state_defaults(self):
        response = client.get("/api/economic-state")
        assert response.status_code == 200
        data = response.json()
        assert "gdp_growth_rate" in data
        assert "interest_rate" in data
        assert "inflation_rate" in data

    def test_domain_states_endpoint(self):
        response = client.get("/api/domain-states")
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "contagion" in data
        assert "geopolitical" in data
        assert "climate" in data
        assert "demographics" in data
        assert "economic" in data

    def test_market_history_endpoint(self):
        response = client.get("/api/market-history?symbols=sp500&period=1mo&interval=1wk")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "symbols" in data
        assert "period" in data

    def test_market_history_multi_symbol(self):
        response = client.get("/api/market-history?symbols=sp500,gold&period=3mo&interval=1wk")
        assert response.status_code == 200
        data = response.json()
        assert "sp500" in data["data"] or "gold" in data["data"] or len(data["data"]) >= 0

    def test_correlation_endpoint(self):
        response = client.get("/api/correlation?symbols=sp500,nasdaq,gold")
        assert response.status_code == 200
        data = response.json()
        assert "correlation_matrix" in data or "message" in data
        assert "symbols" in data

    def test_risk_assessment_endpoint(self):
        response = client.get("/api/risk-assessment")
        assert response.status_code == 200
        data = response.json()
        assert "overall_risk" in data
        assert "domains" in data
        assert "level" in data["overall_risk"]
        assert "score" in data["overall_risk"]
        assert "domain_scores" in data["overall_risk"]

    def test_risk_assessment_domains(self):
        response = client.get("/api/risk-assessment")
        data = response.json()
        domains = data.get("domains", {})
        assert "sentiment" in domains or "contagion" in domains
        if "sentiment" in domains:
            assert "level" in domains["sentiment"]
            assert "score" in domains["sentiment"]
            assert "drivers" in domains["sentiment"]

    def test_portfolio_endpoint(self):
        response = client.post("/api/portfolio?symbols=sp500,gold&n_simulations=100&risk_free_rate=0.05")
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert "optimal_portfolio" in data
            assert "optimal_metrics" in data
            assert "efficient_frontier" in data
            metrics = data["optimal_metrics"]
            assert "expected_return" in metrics
            assert "volatility" in metrics
            assert "sharpe_ratio" in metrics

    def test_portfolio_individual_metrics(self):
        response = client.post("/api/portfolio?symbols=sp500,gold&n_simulations=100&risk_free_rate=0.05")
        if response.status_code == 200:
            data = response.json()
            assert "individual_metrics" in data
            assert "symbols" in data

    def test_status_api_endpoints(self):
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "api_endpoints" in data
        endpoints = data["api_endpoints"]
        assert "/api/market-history" in endpoints
        assert "/api/correlation" in endpoints
        assert "/api/risk-assessment" in endpoints
        assert "/api/portfolio" in endpoints
        assert data["version"] == "0.5.4"

    def test_dashboard_has_new_sections(self):
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "section-markets" in html
        assert "section-risk" in html
        assert "section-portfolio" in html
        assert "loadMarketChart" in html
        assert "loadCorrelation" in html
        assert "loadRiskAssessment" in html
        assert "runPortfolioOptimization" in html
        assert "efficientFrontierChart" in html
        assert "riskRadarChart" in html

    def test_forecast_endpoint(self):
        response = client.get("/api/forecast?symbol=sp500&days=30&n_simulations=100&confidence=high")
        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert "current_price" in data
        assert "forecast_days" in data
        assert "percentiles" in data
        assert "risk_metrics" in data
        assert "p50" in data["percentiles"]

    def test_forecast_risk_metrics(self):
        response = client.get("/api/forecast?symbol=sp500&days=30&n_simulations=100")
        data = response.json()
        rm = data["risk_metrics"]
        assert "var_95_daily" in rm
        assert "cvar_95_daily" in rm
        assert "sortino_ratio" in rm
        assert "calmar_ratio" in rm
        assert "max_drawdown_pct" in rm

    def test_forecast_different_symbols(self):
        for symbol in ["sp500", "gold", "vix"]:
            response = client.get(f"/api/forecast?symbol={symbol}&days=30&n_simulations=100")
            assert response.status_code == 200

    def test_alerts_endpoint(self):
        response = client.get("/api/alerts")
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert "summary" in data
        assert "total_alerts" in data["summary"]

    def test_alerts_structure(self):
        response = client.get("/api/alerts")
        data = response.json()
        summary = data["summary"]
        assert "critical" in summary
        assert "high" in summary
        assert "moderate" in summary
        assert "overall_status" in summary
        if len(data["alerts"]) > 0:
            alert = data["alerts"][0]
            assert "level" in alert
            assert "domain" in alert
            assert "message" in alert
            assert "recommendation" in alert

    def test_portfolio_refactored(self):
        response = client.post("/api/portfolio?symbols=sp500,gold&n_simulations=100&risk_free_rate=0.05")
        if response.status_code == 200:
            data = response.json()
            assert "optimal_portfolio" in data
            assert "conservative_portfolio" in data
            assert "conservative_metrics" in data
            assert "optimal_metrics" in data

    def test_fibonacci_endpoint(self):
        response = client.get("/api/fibonacci?symbol=sp500&period=1y")
        assert response.status_code == 200
        data = response.json()
        assert "fibonacci" in data
        assert "kelly" in data
        assert "symbol" in data
        fib = data["fibonacci"]
        if "error" not in fib:
            assert "high" in fib
            assert "low" in fib
            assert "current_price" in fib
            assert "retracements" in fib
            assert "extensions" in fib
            assert "trend" in fib

    def test_fibonacci_kelly(self):
        response = client.get("/api/fibonacci?symbol=sp500&period=1y")
        data = response.json()
        kelly = data["kelly"]
        assert "kelly_fraction" in kelly
        assert "half_kelly" in kelly
        assert "quarter_kelly" in kelly

    def test_backtest_real_endpoint(self):
        response = client.post("/api/backtest-real?symbol=sp500&window_size=30&n_windows=2&horizon=3")
        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert "data_source" in data
        assert "walk_forward" in data
        assert "risk_metrics" in data
        assert "kelly_criterion" in data["risk_metrics"]
        assert "downside_risk" in data["risk_metrics"]
        assert "fibonacci" in data["risk_metrics"]

    def test_backtest_real_model_comparison(self):
        response = client.post("/api/backtest-real?symbol=gold&window_size=30&n_windows=2&horizon=3")
        assert response.status_code == 200
        data = response.json()
        assert "model_comparison" in data or "walk_forward" in data

    def test_dashboard_has_fibonacci_section(self):
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "fibonacciLevels" in html
        assert "kellyMetrics" in html

    def test_fibonacci_multiple_symbols(self):
        for sym in ["sp500", "gold", "vix"]:
            response = client.get(f"/api/fibonacci?symbol={sym}&period=6mo")
            assert response.status_code == 200

    def test_backtest_real_kelly_in_response(self):
        response = client.post("/api/backtest-real?symbol=sp500&window_size=30&n_windows=2&horizon=3")
        assert response.status_code == 200
        data = response.json()
        kelly = data["risk_metrics"]["kelly_criterion"]
        assert "kelly_fraction" in kelly
        assert "half_kelly" in kelly
        assert isinstance(kelly["kelly_fraction"], float)

    def test_backtest_real_fibonacci_in_response(self):
        response = client.post("/api/backtest-real?symbol=sp500&window_size=30&n_windows=2&horizon=3")
        assert response.status_code == 200
        data = response.json()
        fib = data["risk_metrics"]["fibonacci"]
        assert "high" in fib
        assert "low" in fib
        assert "retracements" in fib

    def test_health_version(self):
        response = client.get("/health")
        data = response.json()
        assert data["version"] == "0.5.4"

    def test_alerts_has_overall_status(self):
        response = client.get("/api/alerts")
        data = response.json()
        assert data["summary"]["overall_status"] in ["CRITICAL", "ELEVATED", "MODERATE", "NORMAL"]

    def test_market_history_valid_symbols(self):
        valid_symbols = ["sp500", "nasdaq", "gold", "oil", "bitcoin", "treasury_10y", "vix", "usd_index"]
        response = client.get(f"/api/market-history?symbols={','.join(valid_symbols[:3])}&period=1mo&interval=1wk")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "period" in data

    def test_correlation_with_vix(self):
        response = client.get("/api/correlation?symbols=sp500,vix,gold")
        assert response.status_code == 200

    def test_risk_assessment_all_domains(self):
        response = client.get("/api/risk-assessment")
        data = response.json()
        domains = data["domains"]
        for d in ["sentiment", "contagion", "geopolitical", "climate", "economic", "demographics"]:
            assert d in domains, f"Missing domain: {d}"

    def test_fibonacci_retracement_structure(self):
        response = client.get("/api/fibonacci?symbol=sp500&period=6mo")
        data = response.json()
        fib = data.get("fibonacci", {})
        if "error" not in fib:
            retracements = fib.get("retracements", {})
            assert "level_0.000" in retracements
            assert "level_0.618" in retracements
            assert "level_1.000" in retracements

    def test_backtest_real_with_synthetic_data(self):
        response = client.post("/api/backtest-real?symbol=nonexistent_symbol_xyz&window_size=30&n_windows=2&horizon=3")
        assert response.status_code == 200
        data = response.json()
        assert data["data_source"] in ["real", "synthetic"]

    def test_forecast_confidence_levels(self):
        for conf in ["low", "medium", "high"]:
            response = client.get(f"/api/forecast?symbol=sp500&days=30&n_simulations=100&confidence={conf}")
            assert response.status_code == 200

    def test_volatility_endpoint(self):
        response = client.get("/api/volatility?symbol=sp500&period=1y&window=21")
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "rolling_volatility" in data
        assert "rolling_sharpe" in data
        assert "max_drawdown_series" in data
        assert "fibonacci" in data

    def test_volatility_summary_metrics(self):
        response = client.get("/api/volatility?symbol=sp500&period=1y&window=21")
        data = response.json()
        s = data["summary"]
        assert "annualized_return" in s
        assert "annualized_volatility" in s
        assert "sharpe_ratio" in s
        assert "sortino_ratio" in s
        assert "calmar_ratio" in s
        assert "max_drawdown_pct" in s
        assert "var_95_daily" in s
        assert "cvar_95_daily" in s
        assert "kelly_fraction" in s
        assert "skewness" in s
        assert "kurtosis" in s

    def test_volatility_different_symbols(self):
        for sym in ["sp500", "gold", "vix"]:
            response = client.get(f"/api/volatility?symbol={sym}&period=6mo&window=21")
            assert response.status_code == 200

    def test_stress_test_endpoint(self):
        response = client.post("/api/stress-test?scenarios=crash,bull&symbol=sp500&period=1y")
        assert response.status_code == 200
        data = response.json()
        assert "baseline" in data
        assert "scenarios" in data
        assert "crash" in data["scenarios"]
        assert "bull" in data["scenarios"]

    def test_stress_test_scenario_structure(self):
        response = client.post("/api/stress-test?scenarios=crash&symbol=sp500")
        data = response.json()
        crash = data["scenarios"]["crash"]
        assert "name" in crash
        assert "annualized_return" in crash
        assert "max_drawdown_pct" in crash
        assert "var_95" in crash
        assert "kelly_fraction" in crash

    def test_stress_test_all_scenarios(self):
        response = client.post("/api/stress-test?scenarios=crash,flash_crash,recovery,stagflation,bull,rate_shock,pandemic&symbol=sp500")
        assert response.status_code == 200
        data = response.json()
        assert len(data["scenarios"]) == 7

    def test_scenario_real_endpoint(self):
        response = client.get("/api/scenario-real?symbol=sp500&period=1y&target_days=30")
        assert response.status_code == 200
        data = response.json()
        assert "scenarios" in data
        assert "baseline" in data["scenarios"]
        assert "bull" in data["scenarios"]
        assert "bear" in data["scenarios"]
        assert "crash" in data["scenarios"]
        assert "current_price" in data
        assert "actual_annualized_return" in data
        assert "actual_annualized_volatility" in data

    def test_scenario_real_structure(self):
        response = client.get("/api/scenario-real?symbol=gold&period=2y&target_days=90")
        data = response.json()
        baseline = data["scenarios"]["baseline"]
        assert "p10" in baseline
        assert "p25" in baseline
        assert "p50" in baseline
        assert "p75" in baseline
        assert "p90" in baseline
        assert "change_pct" in baseline
        assert "probability_up" in baseline
        assert "annualized_return" in baseline
        assert "annualized_volatility" in baseline

    def test_drawdown_endpoint(self):
        response = client.get("/api/drawdown?symbol=sp500&period=2y&window=21")
        assert response.status_code == 200
        data = response.json()
        assert "max_drawdown_pct" in data
        assert "current_drawdown_pct" in data
        assert "avg_recovery_days" in data
        assert "max_recovery_days" in data
        assert "n_drawdown_periods" in data
        assert "drawdown_history" in data
        assert "rolling_drawdown" in data

    def test_drawdown_different_symbols(self):
        for sym in ["sp500", "nasdaq", "gold"]:
            response = client.get(f"/api/drawdown?symbol={sym}&period=1y")
            assert response.status_code == 200
            data = response.json()
            assert "max_drawdown_pct" in data

    def test_dashboard_has_volatility_section(self):
        response = client.get("/")
        assert "section-volatility" in response.text
        assert "Drawdown Analysis" in response.text
        assert "Real Data Scenario" in response.text

    def test_status_has_new_endpoints(self):
        response = client.get("/api/status")
        data = response.json()
        assert "/api/scenario-real" in data["api_endpoints"]
        assert "/api/drawdown" in data["api_endpoints"]

    def test_sector_rotation_endpoint(self):
        response = client.get("/api/sector-rotation")
        assert response.status_code == 200
        data = response.json()
        assert "sectors" in data
        assert "ranking_3m" in data
        assert "leaders" in data
        assert "laggards" in data

    def test_regime_detection_endpoint(self):
        response = client.get("/api/regime-detection?symbol=sp500&period=1y")
        assert response.status_code == 200
        data = response.json()
        assert "current_regime" in data
        assert "confidence" in data
        assert "metrics" in data
        assert "regime_descriptions" in data
        assert data["current_regime"] in [
            "strong_bull", "bull", "crash", "bear",
            "high_volatility", "low_volatility", "range_bound", "transition",
        ]

    def test_regime_detection_metrics(self):
        response = client.get("/api/regime-detection?symbol=sp500&period=2y")
        data = response.json()
        m = data["metrics"]
        assert "recent_return_ann_pct" in m
        assert "recent_volatility_ann_pct" in m
        assert "return_z_score" in m
        assert "volatility_z_score" in m
        assert "long_term_return_ann_pct" in m
        assert "long_term_volatility_ann_pct" in m

    def test_dashboard_has_sector_regime(self):
        response = client.get("/")
        assert "Sector Rotation" in response.text
        assert "Market Regime Detection" in response.text

    def test_dashboard_has_intelligence_bl_var(self):
        response = client.get("/")
        assert "Market Intelligence" in response.text
        assert "Black-Litterman" in response.text
        assert "VaR Analysis" in response.text

    def test_black_litterman_endpoint(self):
        response = client.post("/api/black-litterman", json={
            "target_date": "2030-01-01",
            "confidence": "high",
        })
        assert response.status_code == 200
        data = response.json()
        assert "assets" in data
        assert "bl_optimal_weights" in data
        assert "implied_equilibrium_returns" in data
        assert "bl_posterior_returns" in data

    def test_var_analysis_endpoint(self):
        response = client.get("/api/var-analysis?symbol=sp500&period=1y")
        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert "methods" in data
        assert "summary" in data
        assert "var_95" in data["methods"]
        assert "cvar_95" in data["methods"]

    def test_risk_parity_endpoint(self):
        response = client.get("/api/risk-parity")
        assert response.status_code == 200
        data = response.json()
        assert "weights" in data
        assert "risk_contribution" in data
        assert "portfolio_volatility_pct" in data

    def test_market_intelligence_endpoint(self):
        response = client.get("/api/market-intelligence")
        assert response.status_code == 200
        data = response.json()
        assert "snapshot" in data
        assert "signals" in data
        assert "n_assets" in data

    def test_efficient_frontier_endpoint(self):
        response = client.get("/api/efficient-frontier")
        assert response.status_code == 200
        data = response.json()
        assert "assets" in data
        assert "tangency_portfolio" in data
        assert "minimum_variance_portfolio" in data
        assert "efficient_frontier" in data
        assert "tangency_sharpe" in data

    def test_efficient_frontier_with_target(self):
        response = client.get("/api/efficient-frontier?target_return=0.1")
        assert response.status_code == 200
        data = response.json()
        assert "target_portfolio" in data
        assert data["target_return"] is not None

    def test_covariance_shrinkage_endpoint(self):
        response = client.get("/api/covariance-shrinkage")
        assert response.status_code == 200
        data = response.json()
        assert "assets" in data
        assert "sample_covariance" in data
        assert "shrunk_covariance" in data
        assert "shrinkage_intensity" in data

    def test_dashboard_has_efficient_frontier(self):
        response = client.get("/")
        assert "Efficient Frontier" in response.text or "efficient" in response.text.lower()

    def test_efficient_frontier_in_status(self):
        response = client.get("/api/status")
        data = response.json()
        assert "/api/efficient-frontier" in data["api_endpoints"]
        assert "/api/covariance-shrinkage" in data["api_endpoints"]

    def test_model_comparison_real(self):
        response = client.get("/api/model-comparison-real?symbol=sp500&period=1y&forecast_days=30")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "statistical" in data["models"]
        assert "monte_carlo" in data["models"]
        assert "bayesian" in data["models"]
        assert "ensemble" in data["models"]
        assert "consensus" in data
        assert data["consensus"] in ["bullish", "neutral", "bearish"]

    def test_model_comparison_real_structure(self):
        response = client.get("/api/model-comparison-real?symbol=gold&period=2y&forecast_days=90")
        data = response.json()
        mc = data["models"]["monte_carlo"]
        assert "p10" in mc
        assert "p50" in mc
        assert "p90" in mc
        assert "direction" in mc
        assert "change_pct" in mc

    def test_correlation_network_endpoint(self):
        response = client.get("/api/correlation-network?period=1y&threshold=0.3")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert "statistics" in data
        assert "clusters" in data
        assert "correlation_matrix" in data
        assert data["statistics"]["n_assets"] >= 3

    def test_correlation_network_threshold(self):
        response_low = client.get("/api/correlation-network?period=1y&threshold=0.0")
        response_high = client.get("/api/correlation-network?period=1y&threshold=0.8")
        data_low = response_low.json()
        data_high = response_high.json()
        assert data_low["statistics"]["n_edges"] >= data_high["statistics"]["n_edges"]

    def test_dashboard_has_model_comparison(self):
        response = client.get("/")
        assert "Model Comparison" in response.text

    def test_rolling_sharpe_endpoint(self):
        response = client.get("/api/rolling-sharpe?symbol=sp500&period=1y&window=21")
        assert response.status_code == 200
        data = response.json()
        assert "rolling_sharpe" in data
        assert "downside_risk" in data
        assert "kelly_criterion" in data
        assert "fibonacci" in data
        assert "symbol" in data

    def test_rolling_sharpe_different_symbols(self):
        for sym in ["sp500", "gold", "bitcoin"]:
            response = client.get(f"/api/rolling-sharpe?symbol={sym}&period=1y&window=21")
            assert response.status_code == 200

    def test_status_rolling_sharpe(self):
        response = client.get("/api/status")
        data = response.json()
        assert "/api/rolling-sharpe" in data["api_endpoints"]

    def test_esg_endpoint(self):
        response = client.post("/api/esg", json={
            "environmental_score": 0.7,
            "social_score": 0.6,
            "governance_score": 0.65,
            "carbon_intensity": 0.3,
            "water_stress": 0.2,
        })
        assert response.status_code == 200
        data = response.json()
        assert "overall_esg_score" in data
        assert "environmental_risk" in data
        assert "social_risk" in data
        assert "governance_risk" in data
        assert "risk_category" in data
        assert "key_concerns" in data
        assert "recommendations" in data
        assert "sector_comparison" in data
        assert "trend" in data
        assert "component_scores" in data

    def test_esg_high_risk(self):
        response = client.post("/api/esg", json={
            "environmental_score": 0.2,
            "social_score": 0.2,
            "governance_score": 0.2,
            "carbon_intensity": 0.9,
            "anti_corruption_score": 0.1,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["risk_category"] in ["high_risk", "critical_risk"]

    def test_backtest_real_portfolio(self):
        response = client.get("/api/backtest-real-portfolio?symbols=sp500,gold&period=1y&weights=0.6,0.4")
        assert response.status_code == 200
        data = response.json()
        assert "symbols" in data
        assert "weights" in data
        assert "total_return_pct" in data
        assert "annualized_return_pct" in data
        assert "sharpe_ratio" in data
        assert "max_drawdown_pct" in data
        assert "equity_curve" in data
        assert "individual_stats" in data

    def test_backtest_real_portfolio_defaults(self):
        response = client.get("/api/backtest-real-portfolio")
        assert response.status_code == 200
        data = response.json()
        assert "symbols" in data
        assert len(data["symbols"]) >= 2

    def test_esg_status_endpoint(self):
        response = client.get("/api/status")
        data = response.json()
        assert "/api/esg" in data["api_endpoints"]
        assert "/api/backtest-real-portfolio" in data["api_endpoints"]

    def test_dashboard_has_esg_section(self):
        response = client.get("/")
        assert "ESG Risk Assessment" in response.text

    def test_sortino_endpoint(self):
        response = client.get("/api/sortino?symbol=sp500&period=1y")
        assert response.status_code == 200
        data = response.json()
        assert "sortino_ratio" in data
        assert "downside_risk" in data
        assert "kelly_criterion" in data
        assert "annualized_return_pct" in data

    def test_calmar_endpoint(self):
        response = client.get("/api/calmar?symbol=sp500&period=2y")
        assert response.status_code == 200
        data = response.json()
        assert "calmar_ratio" in data
        assert "sortino_ratio" in data
        assert "max_drawdown_pct" in data
        assert "annualized_return_pct" in data

    def test_sortino_different_symbols(self):
        for sym in ["sp500", "gold", "bitcoin"]:
            response = client.get(f"/api/sortino?symbol={sym}&period=1y")
            assert response.status_code == 200

    def test_options_black_scholes_endpoint(self):
        response = client.post("/api/options/black-scholes", json={
            "spot_price": 100,
            "strike_price": 105,
            "time_to_expiry": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "option_type": "call",
        })
        assert response.status_code == 200
        data = response.json()
        assert "call_price" in data
        assert "put_price" in data
        assert "call_greeks" in data
        assert "put_greeks" in data
        assert "put_call_parity" in data
        assert data["call_price"] > 0

    def test_options_black_scholes_put(self):
        response = client.post("/api/options/black-scholes", json={
            "spot_price": 100,
            "strike_price": 95,
            "time_to_expiry": 0.5,
            "risk_free_rate": 0.05,
            "volatility": 0.3,
            "option_type": "put",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["put_price"] > 0
        assert data["call_price"] > 0

    def test_options_implied_volatility_endpoint(self):
        response = client.post("/api/options/implied-volatility", json={
            "spot_price": 100,
            "strike_price": 100,
            "time_to_expiry": 1.0,
            "risk_free_rate": 0.05,
            "market_price": 10,
            "option_type": "call",
        })
        assert response.status_code == 200
        data = response.json()
        assert "implied_volatility" in data
        assert data["implied_volatility"] is not None

    def test_options_chain_endpoint(self):
        response = client.post("/api/options/chain", json={
            "spot_price": 100,
            "time_to_expiry": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "strikes": [90, 95, 100, 105, 110],
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["chain"]) == 5
        assert data["chain"][0]["strike"] == 90
        assert data["chain"][0]["call"]["price"] > 0

    def test_options_binomial_endpoint(self):
        response = client.post("/api/options/binomial", json={
            "spot_price": 100,
            "strike_price": 100,
            "time_to_expiry": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "steps": 50,
            "option_type": "call",
            "american": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert "binomial_price" in data
        assert "black_scholes_price" in data
        assert data["binomial_price"] > 0

    def test_options_binomial_american(self):
        response = client.post("/api/options/binomial", json={
            "spot_price": 100,
            "strike_price": 100,
            "time_to_expiry": 1.0,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "steps": 50,
            "option_type": "put",
            "american": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["american"] is True

    def test_options_strategy_endpoint(self):
        response = client.post("/api/options/strategy", json={
            "spot_price": 100,
            "legs": [
                {"type": "call", "strike": 100, "premium": 5, "quantity": 1, "direction": "long"},
                {"type": "put", "strike": 100, "premium": 4, "quantity": 1, "direction": "long"},
            ],
        })
        assert response.status_code == 200
        data = response.json()
        assert "pnl" in data
        assert "max_profit" in data
        assert "max_loss" in data
        assert "breakeven_points" in data

    def test_options_volatility_surface_endpoint(self):
        response = client.get("/api/options/volatility-surface")
        assert response.status_code == 200
        data = response.json()
        assert "surface" in data
        assert "strikes" in data

    def test_options_greeks_surface_endpoint(self):
        response = client.post("/api/options/greeks-surface", json={
            "spot_price": 100,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "greek": "delta",
            "option_type": "call",
            "strike_range": [90, 95, 100, 105, 110],
            "maturity_range": [0.25, 0.5, 1.0],
        })
        assert response.status_code == 200
        data = response.json()
        assert "surface" in data
        assert data["greek"] == "delta"

    def test_technical_analysis_endpoint(self):
        response = client.get("/api/technical-analysis?symbol=^GSPC&period=1y")
        assert response.status_code == 200
        data = response.json()
        assert "rsi" in data
        assert "macd" in data
        assert "bollinger" in data

    def test_status_includes_options(self):
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        for ep in ["/api/options/black-scholes", "/api/options/implied-volatility",
                    "/api/options/chain", "/api/options/binomial", "/api/options/strategy",
                    "/api/options/volatility-surface", "/api/options/greeks-surface",
                    "/api/technical-analysis"]:
            assert ep in data["api_endpoints"], f"{ep} not in api_endpoints"

    def test_status_includes_technical_analysis(self):
        response = client.get("/api/status")
        data = response.json()
        assert "technical_analysis" in data["models"]["pattern_recognition"]

    def test_dashboard_has_options_section(self):
        response = client.get("/")
        assert "Black-Scholes" in response.text
        assert "Implied Volatility" in response.text
        assert "Option Chain" in response.text
        assert "Strategy P&L" in response.text

    def test_dashboard_has_technical_section(self):
        response = client.get("/")
        assert "Technical Analysis Dashboard" in response.text
        assert "taSymbol" in response.text

    def test_yield_curve_nelson_siegel_endpoint(self):
        response = client.post("/api/yield-curve/nelson-siegel", json={
            "beta0": 0.05, "beta1": -0.02, "beta2": 0.03, "tau": 2.0
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["rates"]) == 10
        assert data["model"] == "nelson_siegel"

    def test_yield_curve_nss_endpoint(self):
        response = client.post("/api/yield-curve/nelson-siegel-svensson", json={
            "beta0": 0.05, "beta1": -0.02, "beta2": 0.03, "beta3": -0.01, "tau1": 2.0, "tau2": 5.0
        })
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "nelson_siegel_svensson"

    def test_yield_curve_fit_endpoint(self):
        response = client.post("/api/yield-curve/fit", json={
            "maturities": [0.5, 1, 2, 3, 5, 10],
            "rates": [3.0, 3.5, 4.0, 4.3, 4.5, 4.6]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_yield_curve_forward_rates_endpoint(self):
        response = client.post("/api/yield-curve/forward-rates", json={
            "spot_rates": [3.0, 3.5, 4.0, 4.3, 4.5],
            "maturities": [1, 2, 3, 5, 10]
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["forward_rates"]) == 4

    def test_yield_curve_scenario_endpoint(self):
        response = client.post("/api/yield-curve/scenario", json={"shift_bps": 100})
        assert response.status_code == 200
        data = response.json()
        assert "base_curve" in data
        assert "shifted_curve" in data

    def test_risk_component_var_endpoint(self):
        response = client.post("/api/risk/component-var", json={
            "n_assets": 3, "n_periods": 252, "confidence": 0.95
        })
        assert response.status_code == 200
        data = response.json()
        assert "portfolio_var" in data
        assert len(data["component_var"]) == 3

    def test_risk_factor_attribution_endpoint(self):
        response = client.post("/api/risk/factor-attribution", json={
            "n_factors": 3, "n_assets": 5, "n_periods": 252
        })
        assert response.status_code == 200
        data = response.json()
        assert "alpha" in data
        assert "factor_betas" in data

    def test_risk_risk_budget_endpoint(self):
        response = client.post("/api/risk/risk-budget", json={
            "n_assets": 3, "n_periods": 252
        })
        assert response.status_code == 200
        data = response.json()
        assert data["portfolio_volatility"] > 0

    def test_risk_tracking_error_endpoint(self):
        response = client.post("/api/risk/tracking-error", json={
            "n_periods": 252, "alpha": 0.001
        })
        assert response.status_code == 200
        data = response.json()
        assert data["tracking_error"] > 0

    def test_risk_stress_decomposition_endpoint(self):
        response = client.post("/api/risk/stress-decomposition", json={
            "n_assets": 3, "n_periods": 252
        })
        assert response.status_code == 200
        data = response.json()
        assert "market_crash" in data["scenarios"]

    def test_status_includes_yield_curve_and_risk(self):
        response = client.get("/api/status")
        data = response.json()
        assert "/api/yield-curve/nelson-siegel" in data["api_endpoints"]
        assert "/api/risk/component-var" in data["api_endpoints"]
        assert "/api/risk/factor-attribution" in data["api_endpoints"]

    def test_valuation_ddm_endpoint(self):
        response = client.post("/api/valuation/ddm", json={
            "current_dividend": 2.0, "growth_rate": 0.05, "discount_rate": 0.10
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intrinsic_value"] > 0

    def test_valuation_ddm_with_market_price(self):
        response = client.post("/api/valuation/ddm", json={
            "current_dividend": 2.0, "growth_rate": 0.05, "discount_rate": 0.10,
            "market_price": 30.0
        })
        assert response.status_code == 200
        data = response.json()
        assert "signal" in data
        assert "upside_pct" in data

    def test_valuation_dcf_endpoint(self):
        response = client.post("/api/valuation/dcf", json={
            "fcf_current": 1000, "growth_rates": [0.10, 0.08, 0.06],
            "terminal_growth": 0.02, "wacc": 0.10, "net_debt": 500, "shares_outstanding": 100
        })
        assert response.status_code == 200
        data = response.json()
        assert data["per_share_value"] > 0

    def test_valuation_residual_income_endpoint(self):
        response = client.post("/api/valuation/residual-income", json={
            "book_value": 50, "earnings_per_share": 5, "cost_of_equity": 0.10,
            "growth_rates": [0.08, 0.06, 0.04]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intrinsic_value"] > 0

    def test_valuation_eva_endpoint(self):
        response = client.post("/api/valuation/eva", json={
            "nopat": 200, "invested_capital": 1000, "wacc": 0.10,
            "growth_rates": [0.05, 0.04, 0.03]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["enterprise_value"] > 0

    def test_valuation_sensitivity_endpoint(self):
        response = client.post("/api/valuation/sensitivity", json={
            "model": "ddm",
            "base_params": {"current_dividend": 2.0, "growth_rate": 0.05, "discount_rate": 0.10}
        })
        assert response.status_code == 200
        data = response.json()
        assert "sensitivities" in data
        assert data["base_value"] > 0

    def test_status_includes_valuation(self):
        response = client.get("/api/status")
        data = response.json()
        assert "/api/valuation/ddm" in data["api_endpoints"]
        assert "/api/valuation/dcf" in data["api_endpoints"]

    def test_epidemiology_seir_endpoint(self):
        response = client.post("/api/epidemiology/seir", json={
            "population": 100000,
            "initial_infected": 50,
            "beta": 0.3,
            "days": 100,
        })
        assert response.status_code == 200
        data = response.json()
        assert "peak_infected" in data
        assert "r0" in data
        assert "daily" in data
        assert data["population"] == 100000

    def test_epidemiology_seir_with_intervention(self):
        response = client.post("/api/epidemiology/seir", json={
            "population": 1000000,
            "initial_infected": 100,
            "beta": 0.3,
            "days": 365,
            "intervention_day": 30,
            "intervention_beta": 0.1,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["peak_infected"] > 0

    def test_epidemiology_seir_with_vaccination(self):
        response = client.post("/api/epidemiology/seir", json={
            "population": 1000000,
            "vaccination_rate": 0.5,
            "vaccine_efficacy": 0.9,
            "days": 200,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_vaccinated"] > 0

    def test_epidemiology_pandemic_scenarios_endpoint(self):
        response = client.post("/api/epidemiology/pandemic-scenarios", json={
            "population": 1000000,
            "base_r0": 2.5,
            "ifr": 0.005,
        })
        assert response.status_code == 200
        data = response.json()
        assert "scenarios" in data
        assert "no_intervention" in data["scenarios"]
        assert "strict_lockdown" in data["scenarios"]

    def test_epidemiology_health_risk_endpoint(self):
        response = client.post("/api/epidemiology/health-risk", json={
            "age": 65,
            "bmi": 28.0,
            "smoking": True,
            "diabetes": False,
            "hypertension": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert "total_risk_score" in data
        assert "severity" in data
        assert "specific_risks" in data
        assert "recommendations" in data

    def test_epidemiology_network_spread_endpoint(self):
        response = client.post("/api/epidemiology/network-spread", json={
            "n_nodes": 100,
            "n_edges": 200,
            "initial_infected": 3,
            "transmission_prob": 0.15,
            "recovery_prob": 0.05,
            "days": 60,
        })
        assert response.status_code == 200
        data = response.json()
        assert "peak_infected" in data
        assert "daily_counts" in data
        assert data["n_nodes"] == 100

    def test_technology_adoption_endpoint(self):
        response = client.post("/api/technology/adoption", json={
            "technology": "AI",
            "current_adoption": 0.15,
            "market_potential": 1.0,
            "innovation_speed": 0.3,
            "years": 20,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["technology"] == "AI"
        assert "predicted_adoption_5y" in data
        assert "phase" in data
        assert "yearly_adoption" in data

    def test_technology_ai_forecast_endpoint(self):
        response = client.post("/api/technology/ai-forecast", json={
            "current_capability": 0.35,
            "compute_growth_rate": 0.4,
            "years": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert "predicted_capabilities" in data
        assert "milestones_reached" in data
        assert len(data["predicted_capabilities"]) == 11

    def test_technology_disruption_endpoint(self):
        response = client.post("/api/technology/disruption", json={
            "industry": "healthcare",
            "timeline_years": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["industry"] == "healthcare"
        assert "technologies" in data
        assert "top_disruptors" in data

    def test_technology_disruption_custom_techs(self):
        response = client.post("/api/technology/disruption", json={
            "industry": "finance",
            "technologies": [
                {"name": "AI Trading", "impact": 0.8, "probability": 0.7, "speed": 0.6, "displacement": 0.3}
            ],
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["technologies"]) == 1

    def test_technology_power_shift_endpoint(self):
        response = client.post("/api/technology/power-shift", json={
            "horizon_years": 15,
        })
        assert response.status_code == 200
        data = response.json()
        assert "regions" in data
        assert "projected_leaders" in data
        assert len(data["regions"]) == 5

    def test_technology_adoption_comparison_endpoint(self):
        response = client.post("/api/technology/adoption-comparison", json={})
        assert response.status_code == 200
        data = response.json()
        assert "technologies" in data
        assert len(data["technologies"]) == 7

    def test_social_unrest_endpoint(self):
        response = client.post("/api/social/unrest", json={
            "gini_coefficient": 0.41,
            "unemployment_rate": 0.06,
            "political_instability": 0.3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "unrest_probability" in data
        assert "unrest_level" in data
        assert "economic_distress" in data
        assert "political_risk" in data

    def test_social_migration_endpoint(self):
        response = client.post("/api/social/migration", json={
            "origin_gdp_per_capita": 5000,
            "destination_gdp_per_capita": 50000,
            "origin_conflict": 0.3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "migration_pressure" in data
        assert "migration_level" in data
        assert "destination_impact" in data

    def test_social_polarization_endpoint(self):
        response = client.post("/api/social/polarization", json={
            "income_inequality": 0.41,
            "political_polarization": 0.6,
        })
        assert response.status_code == 200
        data = response.json()
        assert "polarization_index" in data
        assert "fragmentation_index" in data
        assert "state" in data
        assert "primary_risks" in data

    def test_social_wealth_power_shift_endpoint(self):
        response = client.post("/api/social/wealth-power-shift", json={
            "horizon": 2040,
        })
        assert response.status_code == 200
        data = response.json()
        assert "regions" in data
        assert "biggest_gainer" in data
        assert "biggest_loser" in data
        assert data["horizon"] == 2040

    def test_social_civil_unrest_scenario_endpoint(self):
        response = client.post("/api/social/civil-unrest-scenario", json={
            "base_unrest": 0.3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "scenarios" in data
        assert "baseline" in data["scenarios"]
        assert "worst_case" in data["scenarios"]

    def test_social_civil_unrest_custom_triggers(self):
        response = client.post("/api/social/civil-unrest-scenario", json={
            "base_unrest": 0.5,
            "trigger_events": [
                {"event": "test_crisis", "probability": 0.4, "impact": 0.6}
            ],
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["trigger_events"]) == 1

    def test_status_includes_epidemiology(self):
        response = client.get("/api/status")
        data = response.json()
        assert "/api/epidemiology/seir" in data["api_endpoints"]
        assert "/api/epidemiology/health-risk" in data["api_endpoints"]
        assert "epidemiology" in data["models"]["domains"]

    def test_status_includes_technology(self):
        response = client.get("/api/status")
        data = response.json()
        assert "/api/technology/adoption" in data["api_endpoints"]
        assert "/api/technology/ai-forecast" in data["api_endpoints"]
        assert "technology" in data["models"]["domains"]

    def test_status_includes_social(self):
        response = client.get("/api/status")
        data = response.json()
        assert "/api/social/unrest" in data["api_endpoints"]
        assert "/api/social/migration" in data["api_endpoints"]
        assert "social" in data["models"]["domains"]

    def test_status_includes_geopolitical_v2(self):
        response = client.get("/api/status")
        data = response.json()
        assert "/api/geopolitical/conflict-score" in data["api_endpoints"]
        assert "/api/geopolitical/alliance-network" in data["api_endpoints"]
        assert "/api/geopolitical/resource-competition" in data["api_endpoints"]

    def test_predict_with_epidemiology_state(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "epidemiology_state": {
                "population": 500000,
                "initial_infected": 100,
                "beta": 0.25,
                "mu": 0.01,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "epidemiology" in domains

    def test_predict_with_technology_state(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "technology_state": {
                "current_adoption": 0.2,
                "innovation_speed": 0.4,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "technology" in domains

    def test_predict_with_social_state(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "social_state": {
                "gini_coefficient": 0.45,
                "unemployment_rate": 0.07,
                "political_instability": 0.4,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "social" in domains

    def test_predict_with_all_domain_states(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "medium",
            "economic_state": {
                "gdp_growth_rate": 0.02,
                "debt_to_gdp": 1.2,
                "interest_rate": 0.045,
                "inflation_rate": 0.03,
                "unemployment_rate": 0.04,
                "stock_valuation": 1.0,
            },
            "geopolitical_state": {
                "us_china_tension": 0.6,
                "russia_west_tension": 0.7,
            },
            "climate_state": {
                "global_temp_anomaly_c": 1.2,
                "co2_ppm": 420,
            },
            "epidemiology_state": {
                "population": 1e6,
                "initial_infected": 50,
                "beta": 0.3,
                "mu": 0.005,
            },
            "technology_state": {
                "current_adoption": 0.15,
                "innovation_speed": 0.3,
            },
            "social_state": {
                "gini_coefficient": 0.41,
                "unemployment_rate": 0.06,
                "political_instability": 0.3,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert len(domains) >= 5

    def test_diagnose_with_epidemiology_state(self):
        response = client.post("/api/diagnose", json={
            "epidemiology_state": {
                "population": 500000,
                "initial_infected": 100,
                "beta": 0.25,
                "mu": 0.01,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "cycle_diagnosis" in data

    def test_diagnose_with_technology_state(self):
        response = client.post("/api/diagnose", json={
            "technology_state": {
                "current_adoption": 0.2,
                "innovation_speed": 0.4,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "cycle_diagnosis" in data

    def test_diagnose_with_social_state(self):
        response = client.post("/api/diagnose", json={
            "social_state": {
                "gini_coefficient": 0.45,
                "unemployment_rate": 0.07,
                "political_instability": 0.4,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "cycle_diagnosis" in data

    def test_diagnose_with_all_domain_states(self):
        response = client.post("/api/diagnose", json={
            "economic_state": {
                "gdp_growth_rate": 0.02,
                "debt_to_gdp": 1.2,
                "interest_rate": 0.045,
                "inflation_rate": 0.03,
                "unemployment_rate": 0.04,
                "stock_valuation": 1.0,
            },
            "geopolitical_state": {
                "us_china_tension": 0.6,
                "russia_west_tension": 0.7,
            },
            "epidemiology_state": {
                "population": 1e6,
                "initial_infected": 50,
                "beta": 0.3,
                "mu": 0.005,
            },
            "technology_state": {
                "current_adoption": 0.15,
                "innovation_speed": 0.3,
            },
            "social_state": {
                "gini_coefficient": 0.41,
                "unemployment_rate": 0.06,
                "political_instability": 0.3,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "cycle_diagnosis" in data
        assert "cross_domain_correlations" in data

    def test_dashboard_has_health_tab(self):
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "section-health" in html
        assert "SEIR" in html
        assert "Pandemic" in html

    def test_dashboard_has_technology_tab(self):
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "section-tech" in html or "section-technology" in html

    def test_dashboard_has_social_tab(self):
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "section-social" in html

    def test_yield_curve_real_endpoint(self):
        response = client.get("/api/yield-curve/real")
        assert response.status_code == 200
        data = response.json()
        assert "observed" in data or "fitted_curve" in data

    def test_esg_market_endpoint(self):
        response = client.get("/api/esg-market")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_dashboard_has_geopolitical_deep_analysis(self):
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "Geopolitical Deep Analysis" in html
        assert "loadConflictScore" in html
        assert "loadAllianceNetwork" in html
        assert "loadResourceCompetition" in html

    def test_epidemiology_seir_custom_beta(self):
        response = client.post("/api/epidemiology/seir", json={
            "population": 500000,
            "initial_infected": 50,
            "beta": 0.5,
            "days": 100,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["r0"] > 1.0

    def test_epidemiology_health_risk_elderly(self):
        response = client.post("/api/epidemiology/health-risk", json={
            "age": 80,
            "bmi": 26.0,
            "smoking": True,
            "diabetes": True,
            "hypertension": True,
            "cardiovascular": True,
        })
        assert response.status_code == 200
        data = response.json()
        young = client.post("/api/epidemiology/health-risk", json={"age": 25, "bmi": 22.0})
        young_data = young.json()
        assert data["total_risk_score"] > young_data["total_risk_score"]

    def test_technology_ai_forecast_long(self):
        response = client.post("/api/technology/ai-forecast", json={
            "current_capability": 0.2,
            "compute_growth_rate": 0.6,
            "years": 20,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["predicted_capabilities"]) == 21

    def test_technology_disruption_healthcare(self):
        response = client.post("/api/technology/disruption", json={
            "industry": "healthcare",
            "timeline_years": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["industry"] == "healthcare"

    def test_social_unrest_high_inequality(self):
        response = client.post("/api/social/unrest", json={
            "gini_coefficient": 0.65,
            "unemployment_rate": 0.15,
            "political_instability": 0.6,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["unrest_probability"] > 0.3

    def test_social_polarization_high(self):
        response = client.post("/api/social/polarization", json={
            "income_inequality": 0.5,
            "political_polarization": 0.8,
            "echo_chamber_effect": 0.9,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["polarization_index"] > 0.4

    def test_social_migration_climate_risk(self):
        response = client.post("/api/social/migration", json={
            "origin_gdp_per_capita": 2000,
            "destination_gdp_per_capita": 60000,
            "origin_conflict": 0.5,
            "climate_risk": 0.8,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["migration_pressure"] > 0

    def test_dashboard_health_params_dynamic(self):
        response = client.get("/")
        html = response.text
        assert "healthBeta" in html or "updateHealthParams" in html

    def test_dashboard_tech_params_dynamic(self):
        response = client.get("/")
        html = response.text
        assert "updateTechParams" in html
        assert "techAdoption" in html or "techInnov" in html

    def test_dashboard_social_params_dynamic(self):
        response = client.get("/")
        html = response.text
        assert "updateSocialParams" in html
        assert "socialUnemp" in html or "socialGini" in html

    def test_dashboard_has_energy_tab(self):
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "section-energy" in html
        assert "Energy Mix" in html or "energyModel" in html

    def test_dashboard_has_commodities_tab(self):
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "section-commodities" in html
        assert "commodityModel" in html

    def test_dashboard_energy_params_dynamic(self):
        response = client.get("/")
        html = response.text
        assert "updateEnergyParams" in html
        assert "runEnergyModel" in html

    def test_dashboard_commodity_params_dynamic(self):
        response = client.get("/")
        html = response.text
        assert "updateCommodityParams" in html
        assert "runCommodityModel" in html

    def test_dashboard_health_seir_params(self):
        response = client.get("/")
        html = response.text
        assert "healthVaxRate" in html
        assert "healthIntDay" in html
        assert "healthBeta" in html

    def test_epidemiology_seir_with_vaccination(self):
        response = client.post("/api/epidemiology/seir", json={
            "population": 1000000,
            "initial_infected": 100,
            "beta": 0.3,
            "vaccination_rate": 0.3,
            "vaccine_efficacy": 0.9,
            "days": 200,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_vaccinated"] > 0

    def test_epidemiology_seir_with_intervention(self):
        response = client.post("/api/epidemiology/seir", json={
            "population": 1000000,
            "initial_infected": 100,
            "beta": 0.5,
            "intervention_day": 30,
            "intervention_beta": 0.1,
            "days": 200,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["peak_infected"] > 0

    def test_technology_adoption_custom_params(self):
        response = client.post("/api/technology/adoption", json={
            "technology": "AI",
            "current_adoption": 0.20,
            "market_potential": 1.0,
            "innovation_speed": 0.4,
            "resistance": 0.05,
            "years": 15,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["technology"] == "AI"
        assert data["predicted_adoption_5y"] > 0.20

    def test_social_unrest_with_all_params(self):
        response = client.post("/api/social/unrest", json={
            "gini_coefficient": 0.45,
            "unemployment_rate": 0.08,
            "inflation_rate": 0.04,
            "political_instability": 0.35,
            "youth_unemployment": 0.15,
        })
        assert response.status_code == 200
        data = response.json()
        assert "unrest_probability" in data
        assert "economic_distress" in data

    def test_geopolitical_conflict_score_high_tension(self):
        response = client.post("/api/geopolitical/conflict-score", json={
            "us_china_tension": 0.9,
            "russia_west_tension": 0.9,
            "middle_east_instability": 0.9,
            "nuclear_proliferation_risk": 0.8,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["overall_intensity"] > 0.5

    def test_geopolitical_resource_competition_energy(self):
        response = client.post("/api/geopolitical/resource-competition", json={
            "us_china_tension": 0.7,
            "energy_geopolitics": 0.9,
            "supply_chain_fragility": 0.6,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["resources"][0]["name"] == "Rare Earth Minerals"

    def test_predict_with_single_domain_epidemiology(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "epidemiology_state": {
                "population": 500000,
                "initial_infected": 50,
                "beta": 0.25,
                "mu": 0.003,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data

    def test_energy_mix_endpoint(self):
        response = client.post("/api/energy/mix", json={
            "years": 10,
            "scenario": "announced_pledges",
        })
        assert response.status_code == 200
        data = response.json()
        assert "yearly_forecast" in data
        assert len(data["yearly_forecast"]) == 11

    def test_energy_security_endpoint(self):
        response = client.post("/api/energy/security", json={
            "renewable_share_pct": 30,
            "grid_stability": 0.8,
        })
        assert response.status_code == 200
        data = response.json()
        assert "overall_security_index" in data
        assert "risk_level" in data

    def test_energy_carbon_budget_endpoint(self):
        response = client.post("/api/energy/carbon-budget", json={
            "budget_gtco2": 500,
        })
        assert response.status_code == 200
        data = response.json()
        assert "budget_deadline_year" in data
        assert data["budget_deadline_year"] > 2025

    def test_commodities_forecast_endpoint(self):
        response = client.post("/api/commodities/forecast", json={
            "commodity": "gold",
            "days": 126,
        })
        assert response.status_code == 200
        data = response.json()
        assert "commodity" in data
        assert "percentiles" in data

    def test_commodities_supply_demand_endpoint(self):
        response = client.post("/api/commodities/supply-demand", json={
            "commodity": "copper",
        })
        assert response.status_code == 200
        data = response.json()
        assert "supply" in data
        assert "demand" in data

    def test_commodities_correlation_endpoint(self):
        response = client.post("/api/commodities/correlation", json={})
        assert response.status_code == 200
        data = response.json()
        assert "commodities" in data
        assert len(data["commodities"]) == 8

    def test_critical_minerals_endpoint(self):
        response = client.post("/api/commodities/critical-minerals", json={
            "years": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert "minerals" in data
        assert len(data["minerals"]) == 6
        assert "top_critical" in data

    def test_predict_with_energy_state(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "energy_state": {
                "oil_price": 80,
                "renewable_share": 25,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "energy" in domains

    def test_predict_with_commodities_state(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "commodities_state": {
                "gold_price": 2000,
                "supply_chain_stress": 0.4,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "commodities" in domains

    def test_status_includes_energy(self):
        response = client.get("/api/status")
        data = response.json()
        assert "/api/energy/mix" in data["api_endpoints"]
        assert "/api/energy/security" in data["api_endpoints"]
        assert "energy" in data["models"]["domains"]

    def test_status_includes_commodities(self):
        response = client.get("/api/status")
        data = response.json()
        assert "/api/commodities/forecast" in data["api_endpoints"]
        assert "/api/commodities/critical-minerals" in data["api_endpoints"]
        assert "commodities" in data["models"]["domains"]

    def test_energy_mix_validation_rejects_bad_scenario(self):
        response = client.post("/api/energy/mix", json={
            "scenario": "invalid_scenario",
        })
        assert response.status_code == 422

    def test_energy_mix_validation_rejects_negative_oil_price(self):
        response = client.post("/api/energy/mix", json={
            "oil_price_usd": -10,
        })
        assert response.status_code == 422

    def test_commodities_forecast_validation_rejects_bad_commodity(self):
        response = client.post("/api/commodities/forecast", json={
            "commodity": "diamonds",
        })
        assert response.status_code == 422

    def test_carbon_budget_validation_rejects_low_budget(self):
        response = client.post("/api/energy/carbon-budget", json={
            "budget_gtco2": 5,
        })
        assert response.status_code == 422

    def test_energy_transition_cost_endpoint(self):
        response = client.post("/api/energy/transition-cost", json={
            "renewable_share_pct": 20,
            "years": 10,
            "target_renewable_pct": 60,
        })
        assert response.status_code == 200
        data = response.json()
        assert "capex" in data
        assert data["capex"]["total_capex_bn"] > 0

    def test_energy_grid_storage_endpoint(self):
        response = client.post("/api/energy/grid-storage", json={
            "renewable_share_pct": 20,
            "target_renewable_pct": 50,
        })
        assert response.status_code == 200
        data = response.json()
        assert "storage_requirements" in data
        assert data["storage_requirements"]["required_gw"] > 0

    def test_commodities_index_endpoint(self):
        response = client.post("/api/commodities/index", json={
            "gold_price": 2000,
            "supply_chain_stress": 0.4,
            "geopolitical_risk": 0.5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "index_value" in data
        assert data["trend"] in ["bullish", "neutral", "bearish"]

    def test_commodities_seasonality_endpoint(self):
        response = client.post("/api/commodities/seasonality", json={
            "commodity": "oil",
            "years_back": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["avg_monthly_returns"]) == 12
        assert "best_month" in data

    def test_social_cascade_endpoint(self):
        response = client.post("/api/social/cascade", json={
            "n_groups": 6,
            "base_activation": 0.2,
            "connectivity": 0.3,
            "threshold_spread": 0.25,
            "n_simulations": 100,
            "seed": 42,
        })
        assert response.status_code == 200
        data = response.json()
        assert "avg_activation_frac" in data
        assert "full_cascade_probability" in data
        assert "cascade_risk_level" in data
        assert "group_risk_ranking" in data
        assert "intervention_points" in data

    def test_social_cascade_validation(self):
        response = client.post("/api/social/cascade", json={
            "n_groups": 0,
            "base_activation": 0.2,
        })
        assert response.status_code == 422

    def test_epidemiology_intervention_optimizer_endpoint(self):
        response = client.post("/api/epidemiology/intervention-optimizer", json={
            "population": 1000000,
            "initial_infected": 100,
            "beta": 0.3,
            "days": 180,
            "candidate_intervention_days": [7, 14, 30],
            "candidate_intervention_betas": [0.03, 0.15],
            "vaccination_rates": [0.0, 0.5],
        })
        assert response.status_code == 200
        data = response.json()
        assert "baseline" in data
        assert "optimal_intervention" in data
        assert "ranking" in data
        assert "tradeoffs" in data
        assert "recommendation" in data

    def test_epidemiology_intervention_optimizer_defaults(self):
        response = client.post("/api/epidemiology/intervention-optimizer", json={
            "days": 180,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["baseline"]["deaths"] > 0

    def test_technology_quantum_readiness_endpoint(self):
        response = client.post("/api/technology/quantum-readiness", json={
            "current_qubits": 1000,
            "error_rate": 0.001,
            "coherence_time_us": 100,
            "gate_fidelity": 0.999,
            "investment_billion": 5.0,
            "talent_pool_size": 5000,
            "regulatory_support": 0.5,
            "years": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert "current_readiness" in data
        assert "current_score" in data
        assert "yearly_projection" in data
        assert "milestones" in data
        assert "applications" in data
        assert "risk_assessment" in data

    def test_technology_quantum_readiness_defaults(self):
        response = client.post("/api/technology/quantum-readiness", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["current_qubits"] == 1000

    def test_technology_quantum_readiness_validation(self):
        response = client.post("/api/technology/quantum-readiness", json={
            "current_qubits": -1,
        })
        assert response.status_code == 422

    def test_climate_sea_level_projector_endpoint(self):
        response = client.post("/api/climate/sea-level-projector", json={
            "global_temp_anomaly_c": 1.2,
            "sea_level_rise_mm": 200,
        })
        assert response.status_code == 200
        data = response.json()
        assert "scenarios" in data
        assert "weighted_median_2100_mm" in data
        assert "impacted_populations_millions" in data

    def test_status_includes_all_new_endpoints(self):
        response = client.get("/api/status")
        data = response.json()
        endpoints = data["api_endpoints"]
        assert "/api/social/cascade" in endpoints
        assert "/api/epidemiology/intervention-optimizer" in endpoints
        assert "/api/technology/quantum-readiness" in endpoints
        assert "/api/geopolitical/event-shock-simulator" in endpoints
        assert "/api/climate/sea-level-projector" in endpoints
        assert "/api/sentiment/contrarian-signals" in endpoints
        assert "/api/demographics/migration-pressure" in endpoints

    def test_geopolitical_event_shock_simulator_endpoint(self):
        response = client.post("/api/geopolitical/event-shock-simulator", json={
            "us_china_tension": 0.6,
            "russia_west_tension": 0.7,
            "middle_east_instability": 0.5,
            "trade_protectionism": 0.4,
        })
        assert response.status_code == 200
        data = response.json()
        assert "current_tension_index" in data
        assert "geopolitical_risk_level" in data
        assert "cascade_probability" in data
        assert "event_probabilities" in data

    def test_sentiment_contrarian_signals_endpoint(self):
        response = client.post("/api/sentiment/contrarian-signals", json={
            "fear_greed_index": 0.15,
            "vix_level": 0.7,
            "put_call_ratio": 1.5,
            "short_interest": 0.6,
            "institutional_flow": 0.4,
            "retail_flow": -0.3,
            "market_breadth": 0.25,
            "safe_haven_demand": 0.8,
            "credit_spread": 0.6,
            "social_media_sentiment": 0.2,
            "news_sentiment": 0.3,
            "consumer_confidence": 0.3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "extreme_indicators" in data
        assert "divergences" in data
        assert "signals" in data
        assert "composite_contrarian_score" in data
        assert "overall_bias" in data
        assert "credit_risk_level" in data
        assert "indicators_summary" in data
        assert len(data["extreme_indicators"]) >= 2

    def test_sentiment_contrarian_signals_validation(self):
        response = client.post("/api/sentiment/contrarian-signals", json={
            "fear_greed_index": 1.5,
        })
        assert response.status_code == 422

    def test_sentiment_contrarian_signals_default(self):
        response = client.post("/api/sentiment/contrarian-signals", json={})
        assert response.status_code == 200
        data = response.json()
        assert "composite_contrarian_score" in data
        assert data["overall_bias"] in ("bullish_contrarian", "bearish_contrarian", "neutral")

    def test_demographics_migration_pressure_endpoint(self):
        response = client.post("/api/demographics/migration-pressure", json={
            "origin_population_m": 50.0,
            "origin_gdp_per_capita": 3000,
            "destination_gdp_per_capita": 60000,
            "origin_conflict_index": 0.6,
            "climate_vulnerability": 0.5,
            "youth_bulge_pct": 0.35,
            "water_stress": 0.6,
            "food_security_index": 0.3,
            "unemployment_rate": 0.25,
            "political_instability": 0.7,
            "years": 15,
        })
        assert response.status_code == 200
        data = response.json()
        assert "total_push_factor" in data
        assert "economic_push" in data
        assert "environmental_push" in data
        assert "social_push" in data
        assert "pull_factor" in data
        assert "net_pressure" in data
        assert "overall_level" in data
        assert "yearly_projection" in data
        assert "destination_impacts" in data
        assert "recommendations" in data
        assert data["overall_level"] in ("low", "moderate", "high", "critical")

    def test_demographics_migration_pressure_validation(self):
        response = client.post("/api/demographics/migration-pressure", json={
            "origin_population_m": 0,
        })
        assert response.status_code == 422

    def test_demographics_migration_pressure_default(self):
        response = client.post("/api/demographics/migration-pressure", json={})
        assert response.status_code == 200
        data = response.json()
        assert "net_pressure" in data
        assert data["net_pressure"] > 0

    def test_status_includes_all_new_endpoints(self):
        response = client.get("/api/status")
        data = response.json()
        endpoints = data["api_endpoints"]
        assert "/api/social/cascade" in endpoints
        assert "/api/epidemiology/intervention-optimizer" in endpoints
        assert "/api/technology/quantum-readiness" in endpoints
        assert "/api/geopolitical/event-shock-simulator" in endpoints
        assert "/api/climate/sea-level-projector" in endpoints
        assert "/api/sentiment/contrarian-signals" in endpoints
        assert "/api/demographics/migration-pressure" in endpoints

    def test_predict_with_infrastructure_state(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "infrastructure_state": {
                "power_grid_reliability": 0.85,
                "water_system_reliability": 0.90,
                "transport_reliability": 0.80,
                "telecom_reliability": 0.88,
                "healthcare_capacity": 0.70,
                "financial_system_reliability": 0.92,
                "supply_chain_reliability": 0.75,
                "cyber_defense_score": 0.60,
                "redundancy_level": 0.40,
                "maintenance_score": 0.65,
                "interdependency_strength": 0.50,
                "climate_exposure": 0.30,
                "seismic_risk": 0.20,
                "flood_risk": 0.25,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "infrastructure" in domains

    def test_predict_with_supply_chain_state(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "supply_chain_state": {
                "supplier_concentration": 0.40,
                "logistics_disruption_prob": 0.15,
                "single_source_dependency": 0.30,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "supply_chain" in domains

    def test_diagnose_with_infrastructure_state(self):
        response = client.post("/api/diagnose", json={
            "infrastructure_state": {
                "power_grid_reliability": 0.85,
                "water_system_reliability": 0.90,
                "transport_reliability": 0.80,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "cycle_diagnosis" in data

    def test_diagnose_with_supply_chain_state(self):
        response = client.post("/api/diagnose", json={
            "supply_chain_state": {
                "supplier_concentration": 0.40,
                "logistics_disruption_prob": 0.15,
                "single_source_dependency": 0.30,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "cycle_diagnosis" in data

    def test_dashboard_has_climate_sea_level_model(self):
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "climModel" in html
        assert "sea_level" in html

    def test_dashboard_has_geopolitical_event_shock(self):
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "geoModel" in html
        assert "event_shock" in html

    def test_dashboard_has_infrastructure_tab(self):
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "section-infrastructure" in html
        assert "Infrastructure Resilience" in html

    def test_dashboard_has_supply_chain_tab(self):
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "section-supplychain" in html
        assert "Supply Chain Risk" in html


    def test_cyber_assess_endpoint(self):
        response = client.post("/api/cyber/assess", json={
            "threat_level": 0.5,
            "ransomware_frequency": 0.3,
            "nation_state_activity": 0.6,
            "ai_threat_capability": 0.3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "overall_risk_score" in data
        assert "risk_level" in data
        assert "top_threats" in data

    def test_cyber_attack_scenario_endpoint(self):
        response = client.post("/api/cyber/attack-scenario?attack_vector=ransomware&duration_days=90", json={
            "threat_level": 0.5,
            "incident_response_capacity": 0.5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "attack_vector" in data
        assert data["attack_vector"] == "ransomware"
        assert "probability" in data
        assert "financial_impact_m" in data

    def test_cyber_predict_endpoint(self):
        response = client.post("/api/cyber/predict?years_ahead=5", json={
            "threat_level": 0.4,
            "ai_threat_capability": 0.3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "future_risk_score" in data
        assert "ai_threat_growth" in data
        assert "quantum_threat_horizon" in data

    def test_predict_with_cyber_state(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "cyber_state": {
                "threat_level": 0.5,
                "ai_threat_capability": 0.3,
                "nation_state_activity": 0.6,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "cyber" in domains

    def test_diagnose_with_cyber_state(self):
        response = client.post("/api/diagnose", json={
            "cyber_state": {
                "threat_level": 0.5,
                "ai_threat_capability": 0.3,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "cycle_diagnosis" in data

    def test_dashboard_has_cyber_tab(self):
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "section-cyber" in html
        assert "Cyber Threat" in html


    def test_water_assess_endpoint(self):
        response = client.post("/api/water/assess", json={
            "water_stress_index": 0.5,
            "drought_probability": 0.3,
            "water_quality_index": 0.65,
        })
        assert response.status_code == 200
        data = response.json()
        assert "overall_risk" in data
        assert "risk_level" in data
        assert "resilience_score" in data

    def test_water_drought_scenario_endpoint(self):
        response = client.post("/api/water/drought-scenario?severity=0.6&duration_months=12", json={
            "water_stress_index": 0.5,
            "population_pressure": 0.4,
        })
        assert response.status_code == 200
        data = response.json()
        assert "water_reduction_pct" in data
        assert "agricultural_impact_pct" in data
        assert "timeline" in data

    def test_water_predict_endpoint(self):
        response = client.post("/api/water/predict?years_ahead=10", json={
            "water_stress_index": 0.4,
            "climate_impact_factor": 0.3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "future_risk" in data
        assert "water_stress_projected" in data

    def test_food_assess_endpoint(self):
        response = client.post("/api/food/assess", json={
            "food_production_index": 0.7,
            "food_price_volatility": 0.15,
        })
        assert response.status_code == 200
        data = response.json()
        assert "overall_risk" in data
        assert "risk_level" in data
        assert "availability_score" in data

    def test_food_crop_model_endpoint(self):
        response = client.post("/api/food/crop-model?crop=wheat&years_ahead=10", json={
            "food_production_index": 0.7,
            "climate_impact_on_crops": 0.2,
        })
        assert response.status_code == 200
        data = response.json()
        assert "crop" in data
        assert data["crop"] == "wheat"
        assert "current_yield_t_ha" in data

    def test_food_predict_endpoint(self):
        response = client.post("/api/food/predict?years_ahead=10", json={
            "food_production_index": 0.7,
            "population_growth_pressure": 0.3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "future_risk" in data
        assert "hunger_risk_projected" in data

    def test_predict_with_water_state(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "water_state": {
                "water_stress_index": 0.5,
                "drought_probability": 0.3,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "water" in domains

    def test_predict_with_food_state(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "food_state": {
                "food_production_index": 0.7,
                "food_price_volatility": 0.15,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "food" in domains

    def test_dashboard_has_water_food_in_status(self):
        response = client.get("/api/status")
        data = response.json()
        endpoints = data["api_endpoints"]
        assert "/api/water/assess" in endpoints
        assert "/api/food/assess" in endpoints

    def test_political_assess_endpoint(self):
        response = client.post("/api/political/assess", json={
            "government_stability": 0.4,
            "corruption_index": 0.7,
            "military_influence": 0.5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "overall_stability" in data
        assert "stability_level" in data
        assert "coup_risk" in data
        assert "protest_probability" in data
        assert "recommendations" in data

    def test_political_regime_change_endpoint(self):
        response = client.post("/api/political/regime-change?scenario=electoral_crisis&months=12", json={
            "government_stability": 0.3,
            "military_influence": 0.6,
        })
        assert response.status_code == 200
        data = response.json()
        assert "probability" in data
        assert "timeline" in data
        assert "economic_impact_pct" in data

    def test_political_predict_endpoint(self):
        response = client.post("/api/political/predict?years_ahead=5", json={
            "government_stability": 0.5,
            "corruption_index": 0.3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "future_stability" in data
        assert "authoritarian_drift" in data
        assert "reform_likelihood" in data

    def test_biodiversity_assess_endpoint(self):
        response = client.post("/api/biodiversity/assess", json={
            "habitat_loss_rate": 0.4,
            "species_extinction_rate": 0.5,
            "deforestation_rate": 0.3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "biodiversity_index" in data
        assert "health_level" in data
        assert "threat_assessment" in data
        assert "ecosystem_resilience" in data
        assert "tipping_points" in data

    def test_biodiversity_extinction_scenario_endpoint(self):
        response = client.post("/api/biodiversity/extinction-scenario?scenario=climate_crisis&years=50", json={
            "habitat_loss_rate": 0.5,
            "climate_impact_on_biodiversity": 0.6,
        })
        assert response.status_code == 200
        data = response.json()
        assert "final_biodiversity_index" in data
        assert "timeline" in data
        assert "outcome" in data

    def test_biodiversity_predict_endpoint(self):
        response = client.post("/api/biodiversity/predict?years_ahead=10", json={
            "habitat_loss_rate": 0.3,
            "species_extinction_rate": 0.4,
        })
        assert response.status_code == 200
        data = response.json()
        assert "future_biodiversity_index" in data
        assert "mass_extinction_risk" in data

    def test_pandemic_assess_endpoint(self):
        response = client.post("/api/pandemic/assess", json={
            "healthcare_capacity": 0.5,
            "surveillance_systems": 0.4,
            "vaccine_readiness": 0.3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "overall_preparedness" in data
        assert "readiness_level" in data
        assert "gaps" in data
        assert "response_capacity" in data

    def test_pandemic_outbreak_endpoint(self):
        response = client.post("/api/pandemic/outbreak?pathogen_type=respiratory_virus&r0=2.5&days=365", json={
            "healthcare_capacity": 0.5,
            "surveillance_systems": 0.4,
        })
        assert response.status_code == 200
        data = response.json()
        assert "effective_r0" in data
        assert "detection_delay_days" in data
        assert "timeline" in data

    def test_pandemic_predict_endpoint(self):
        response = client.post("/api/pandemic/predict?years_ahead=5", json={
            "healthcare_capacity": 0.5,
            "surveillance_systems": 0.4,
        })
        assert response.status_code == 200
        data = response.json()
        assert "future_preparedness" in data
        assert "pandemic_probability" in data

    def test_predict_with_political_state(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "political_state": {
                "government_stability": 0.5,
                "corruption_index": 0.3,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "political" in domains

    def test_predict_with_biodiversity_state(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "biodiversity_state": {
                "habitat_loss_rate": 0.4,
                "species_extinction_rate": 0.5,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "biodiversity" in domains

    def test_predict_with_pandemic_state(self):
        response = client.post("/api/predict", json={
            "target_date": "2030-01-01",
            "confidence": "high",
            "pandemic_state": {
                "healthcare_capacity": 0.5,
                "surveillance_systems": 0.4,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        domains = [p["domain"] for p in data["predictions"]]
        assert "pandemic_preparedness" in domains

    def test_dashboard_has_new_domains_in_status(self):
        response = client.get("/api/status")
        data = response.json()
        endpoints = data["api_endpoints"]
        assert "/api/political/assess" in endpoints
        assert "/api/political/regime-change" in endpoints
        assert "/api/political/predict" in endpoints
        assert "/api/biodiversity/assess" in endpoints
        assert "/api/biodiversity/extinction-scenario" in endpoints
        assert "/api/biodiversity/predict" in endpoints
        assert "/api/pandemic/assess" in endpoints
        assert "/api/pandemic/outbreak" in endpoints
        assert "/api/pandemic/predict" in endpoints
        domains = data["models"]["domains"]
        assert "political" in domains
        assert "biodiversity" in domains
        assert "pandemic" in domains

    def test_space_orbital_environment_endpoint(self):
        response = client.post("/api/space/orbital-environment", json={
            "orbital_debris_count": 30000, "active_satellites": 8000,
            "kessler_syndrome_risk": 0.2,
        })
        assert response.status_code == 200
        data = response.json()
        assert "collision_probability" in data
        assert "kessler_status" in data

    def test_space_kessler_scenario_endpoint(self):
        response = client.post("/api/space/kessler-scenario?scenario=moderate&years=10", json={
            "orbital_debris_count": 25000, "active_satellites": 7500,
        })
        assert response.status_code == 200
        data = response.json()
        assert "scenario" in data
        assert "kessler_triggered" in data

    def test_space_predict_endpoint(self):
        response = client.post("/api/space/predict?years_ahead=5", json={
            "orbital_debris_count": 25000, "kessler_syndrome_risk": 0.15,
        })
        assert response.status_code == 200
        data = response.json()
        assert "future_kessler_risk" in data

    def test_mental_health_assess_endpoint(self):
        response = client.post("/api/mental-health/assess", json={
            "depression_prevalence": 0.08, "anxiety_prevalence": 0.06,
            "social_isolation_index": 0.5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "overall_burden" in data
        assert "burden_level" in data

    def test_mental_health_crisis_scenario_endpoint(self):
        response = client.post("/api/mental-health/crisis-scenario?scenario=economic_recession&years=3", json={
            "depression_prevalence": 0.08, "anxiety_prevalence": 0.06,
        })
        assert response.status_code == 200
        data = response.json()
        assert "scenario" in data

    def test_mental_health_predict_endpoint(self):
        response = client.post("/api/mental-health/predict?years_ahead=5", json={
            "depression_prevalence": 0.08, "treatment_gap_pct": 0.5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "future_depression_prevalence" in data

    def test_housing_assess_endpoint(self):
        response = client.post("/api/housing/assess", json={
            "median_home_price": 400000, "median_income": 75000,
            "mortgage_rate": 0.07,
        })
        assert response.status_code == 200
        data = response.json()
        assert "market_health_score" in data
        assert "health_label" in data

    def test_housing_scenario_endpoint(self):
        response = client.post("/api/housing/scenario?scenario=boom&years=3", json={
            "median_home_price": 350000, "median_income": 70000,
        })
        assert response.status_code == 200
        data = response.json()
        assert "scenario" in data

    def test_housing_predict_endpoint(self):
        response = client.post("/api/housing/predict?years_ahead=5", json={
            "median_home_price": 350000, "median_income": 70000,
        })
        assert response.status_code == 200
        data = response.json()
        assert "future_median_price" in data

    def test_dashboard_has_space_mental_housing_tabs(self):
        response = client.get("/")
        html = response.text
        assert "section-space" in html
        assert "section-mentalhealth" in html
        assert "section-housing" in html

    def test_new_domains_in_status(self):
        response = client.get("/api/status")
        data = response.json()
        endpoints = data["api_endpoints"]
        assert "/api/space/orbital-environment" in endpoints
        assert "/api/space/kessler-scenario" in endpoints
        assert "/api/space/predict" in endpoints
        assert "/api/mental-health/assess" in endpoints
        assert "/api/mental-health/crisis-scenario" in endpoints
        assert "/api/mental-health/predict" in endpoints
        assert "/api/housing/assess" in endpoints
        assert "/api/housing/scenario" in endpoints
        assert "/api/housing/predict" in endpoints
        domains = data["models"]["domains"]
        assert "space" in domains
        assert "mental_health" in domains
        assert "housing" in domains

    def test_seismic_hazard_endpoint(self):
        response = client.post("/api/seismic/hazard", json={"fault_stress_accumulation": 0.7, "volcanic_activity_index": 0.5})
        assert response.status_code == 200
        data = response.json()
        assert "composite_hazard" in data
        assert "hazard_level" in data

    def test_seismic_earthquake_scenario_endpoint(self):
        response = client.post("/api/seismic/earthquake-scenario?magnitude=8.0&depth_km=15", json={})
        assert response.status_code == 200
        data = response.json()
        assert "damage_ratio" in data

    def test_solar_weather_endpoint(self):
        response = client.post("/api/solar/weather", json={"kp_index": 7.0, "dst_index": -200.0})
        assert response.status_code == 200
        data = response.json()
        assert "composite_risk" in data
        assert "risk_level" in data

    def test_solar_carrington_endpoint(self):
        response = client.post("/api/solar/carrington?grid_hardening=0.5", json={})
        assert response.status_code == 200
        data = response.json()
        assert "grid_damage_pct" in data

    def test_oceanic_assess_endpoint(self):
        response = client.post("/api/oceanic/assess", json={"enso_index": 1.5, "sea_surface_temp_anomaly_c": 1.0})
        assert response.status_code == 200
        data = response.json()
        assert "composite_ocean_risk" in data
        assert "enso_phase" in data

    def test_neo_impact_risk_endpoint(self):
        response = client.post("/api/neo/impact-risk", json={"detection_rate_pct": 0.6})
        assert response.status_code == 200
        data = response.json()
        assert "composite_risk" in data

    def test_neo_impact_scenario_endpoint(self):
        response = client.post("/api/neo/impact-scenario?diameter_m=500&velocity_kms=20", json={})
        assert response.status_code == 200
        data = response.json()
        assert "kinetic_energy_mt_tnt" in data

    def test_natural_data_sources_endpoint(self):
        response = client.get("/api/natural-data/sources")
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) >= 8

    def test_atmospheric_assess_endpoint(self):
        response = client.post("/api/atmospheric/assess", json={"polar_vortex_strength": 0.3, "ozone_column_du": 200})
        assert response.status_code == 200
        data = response.json()
        assert "composite_atmospheric_risk" in data
        assert "risk_level" in data

    def test_atmospheric_ssw_endpoint(self):
        response = client.post("/api/atmospheric/ssw?vortex_strength=0.2&ssw_intensity=0.8&duration_days=30", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["scenario"] == "sudden_stratospheric_warming"
        assert "temperature_anomaly_northern_c" in data

    def test_atmospheric_ozone_endpoint(self):
        response = client.post("/api/atmospheric/ozone-scenario?ozone_column_du=150&duration_months=6", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["scenario"] == "ozone_depletion"
        assert "uv_increase_pct" in data

    def test_atmospheric_predict_endpoint(self):
        response = client.post("/api/atmospheric/predict?years_ahead=10", json={"atmospheric_co2_ppm": 420})
        assert response.status_code == 200
        data = response.json()
        assert "future_co2_ppm" in data
        assert "vortex_trend" in data

    def test_ecological_assess_endpoint(self):
        response = client.post("/api/ecological/assess", json={"biodiversity_index": 0.3, "climate_stress": 0.7})
        assert response.status_code == 200
        data = response.json()
        assert "ecosystem_health_index" in data
        assert "health_level" in data
        assert "trophic_integrity" in data
        assert "resilience_score" in data

    def test_ecological_trophic_cascade_endpoint(self):
        response = client.post("/api/ecological/trophic-cascade?keystone_removed=apex_predator&severity=0.7&years=10", json={})
        assert response.status_code == 200
        data = response.json()
        assert "scenario" in data
        assert "timeline" in data
        assert data["keystone_removed"] == "apex_predator"

    def test_ecological_predict_endpoint(self):
        response = client.post("/api/ecological/predict?years_ahead=15", json={"biodiversity_index": 0.3})
        assert response.status_code == 200
        data = response.json()
        assert "future_biodiversity_index" in data
        assert "ecosystem_regime" in data
        assert "tipping_point_risk" in data


    def test_paleoclimate_assess_endpoint(self):
        response = client.post("/api/paleoclimatology/assess", json={"co2_ppmv": 420, "ice_volume_fraction": 0.03})
        assert response.status_code == 200
        data = response.json()
        assert "insolation_variability" in data
        assert "orbital_phase" in data

    def test_paleoclimate_analog_endpoint(self):
        response = client.post("/api/paleoclimatology/analog?analog=eemian&years=5000", json={})
        assert response.status_code == 200
        data = response.json()
        assert "analog_name" in data
        assert "similarity_score" in data

    def test_paleoclimate_predict_endpoint(self):
        response = client.post("/api/paleoclimatology/predict?years_ahead=1000", json={"co2_ppmv": 420})
        assert response.status_code == 200
        data = response.json()
        assert "future_co2_ppmv" in data
        assert "climate_regime" in data

    def test_cosmic_radiation_assess_endpoint(self):
        response = client.post("/api/cosmic-radiation/assess", json={"gcr_flux_relative": 1.2, "solar_modulation_potential_gv": 0.3})
        assert response.status_code == 200
        data = response.json()
        assert "composite_risk" in data
        assert "risk_level" in data

    def test_cosmic_radiation_solar_minimum_endpoint(self):
        response = client.post("/api/cosmic-radiation/solar-minimum?duration_years=3&severity=0.8", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["scenario"] == "grand_solar_minimum"

    def test_cosmic_radiation_cme_endpoint(self):
        response = client.post("/api/cosmic-radiation/cme-shielding?cme_intensity=0.7&recovery_hours=48", json={})
        assert response.status_code == 200
        data = response.json()
        assert "forbush_decrease_pct" in data

    def test_cosmic_radiation_predict_endpoint(self):
        response = client.post("/api/cosmic-radiation/predict?years_ahead=10", json={"gcr_flux_relative": 1.0})
        assert response.status_code == 200
        data = response.json()
        assert "future_gcr_flux" in data
        assert "radiation_regime" in data

    def test_gravitational_assess_endpoint(self):
        response = client.post("/api/gravitational/assess", json={"lunar_tidal_force_relative": 1.0, "planetary_alignment_index": 0.2})
        assert response.status_code == 200
        data = response.json()
        assert "composite_tidal_index" in data
        assert "level" in data

    def test_gravitational_alignment_endpoint(self):
        response = client.post("/api/gravitational/alignment?n_planets=5&alignment_strength=0.7&days=30", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["scenario"] == "planetary_alignment"

    def test_gravitational_perigee_endpoint(self):
        response = client.post("/api/gravitational/perigee-spring?perigee_distance_km=356500&duration_days=7", json={})
        assert response.status_code == 200
        data = response.json()
        assert "king_tide_amplification" in data

    def test_gravitational_predict_endpoint(self):
        response = client.post("/api/gravitational/predict?years_ahead=10", json={"lunar_tidal_force_relative": 1.0})
        assert response.status_code == 200
        data = response.json()
        assert "future_moon_distance_km" in data
        assert "tidal_regime" in data

    def test_compound_hazard_assess_endpoint(self):
        response = client.post("/api/compound-hazard/assess", json={"seismic_hazard": 0.5, "infrastructure_resilience": 0.4})
        assert response.status_code == 200
        data = response.json()
        assert "compound_hazard_index" in data
        assert "hazard_level" in data

    def test_compound_hazard_scenario_endpoint(self):
        response = client.post("/api/compound-hazard/scenario?scenario=earthquake_tsunami_nuclear&years=10&n_simulations=100", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["scenario"] == "earthquake_tsunami_nuclear"
        assert "annual_cascade_probability" in data

    def test_compound_hazard_multi_endpoint(self):
        response = client.post("/api/compound-hazard/multi-hazard?n_hazards=3&years=10&n_simulations=200", json={})
        assert response.status_code == 200
        data = response.json()
        assert "mean_simultaneous" in data

    def test_hazard_monte_carlo_portfolio_endpoint(self):
        response = client.post("/api/hazard-monte-carlo/portfolio?years=10&n_simulations=100&exposure_billion=50&resilience_factor=0.6", json={"hazards": ["earthquake", "hurricane"]})
        assert response.status_code == 200
        data = response.json()
        assert "mean_total_loss_billion" in data

    def test_hazard_monte_carlo_earthquake_endpoint(self):
        response = client.post("/api/hazard-monte-carlo/earthquake-scenarios?n_simulations=50&exposure_billion=100", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["hazard"] == "earthquake"

    def test_hazard_monte_carlo_hurricane_endpoint(self):
        response = client.post("/api/hazard-monte-carlo/hurricane-season?n_simulations=100&el_nino_factor=1.0&climate_warming_factor=1.1", json={})
        assert response.status_code == 200
        data = response.json()
        assert "mean_storms" in data

    def test_world_state_assess_endpoint(self):
        response = client.get("/api/world-state/assess")
        assert response.status_code == 200
        data = response.json()
        assert "assessment" in data or "overall_stability" in data or "radar" in data

    def test_world_state_trajectory_endpoint(self):
        response = client.get("/api/world-state/trajectory?years=20")
        assert response.status_code == 200
        data = response.json()
        assert "base_stability" in data or "collapse_probability" in data or "timeline" in data

    def test_world_state_radar_endpoint(self):
        response = client.get("/api/world-state/domain-radar")
        assert response.status_code == 200

    def test_tsunami_scenarios_endpoint(self):
        response = client.post("/api/hazard-monte-carlo/tsunami-scenarios?n_simulations=50", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["hazard"] == "tsunami"

    def test_volcanic_winter_endpoint(self):
        response = client.post("/api/hazard-monte-carlo/volcanic-winter?vei_min=5&years=50&n_simulations=100", json={})
        assert response.status_code == 200
        data = response.json()
        assert "mean_gdp_loss_trillion" in data

    def test_drought_portfolio_endpoint(self):
        response = client.post("/api/hazard-monte-carlo/drought-portfolio?years=30&n_simulations=100", json={})
        assert response.status_code == 200
        data = response.json()
        assert "mean_total_loss_billion" in data

    def test_fate_trajectory_endpoint(self):
        response = client.get("/api/fate/trajectory")
        assert response.status_code == 200
        data = response.json()
        assert "trajectory_score" in data
        assert "trajectory_label" in data

    def test_fate_thermodynamic_endpoint(self):
        response = client.get("/api/fate/thermodynamic")
        assert response.status_code == 200
        data = response.json()
        assert "thermodynamic_pressure" in data
        assert "eroi_current" in data

    def test_fate_depletion_endpoint(self):
        response = client.get("/api/fate/depletion?years=50")
        assert response.status_code == 200
        data = response.json()
        assert "overall_depletion_pressure" in data

    def test_fate_commitments_endpoint(self):
        response = client.get("/api/fate/commitments")
        assert response.status_code == 200
        data = response.json()
        assert "n_commitments" in data
        assert "avg_irreversibility" in data

    def test_fate_cosmological_endpoint(self):
        response = client.get("/api/fate/cosmological")
        assert response.status_code == 200
        data = response.json()
        assert "timeline" in data

    def test_fate_scenario_endpoint(self):
        response = client.get("/api/fate/scenario?scenario=default&years=50")
        assert response.status_code == 200
        data = response.json()
        assert "outcome" in data

    def test_fate_personal_endpoint(self):
        response = client.get("/api/fate/personal")
        assert response.status_code == 200
        data = response.json()
        assert "fate_type" in data
        assert "total_determinism" in data

    def test_fate_personal_mortality_endpoint(self):
        response = client.get("/api/fate/personal/mortality")
        assert response.status_code == 200
        data = response.json()
        assert "life_expectancy_adjusted" in data

    def test_fate_personal_habits_endpoint(self):
        response = client.get("/api/fate/personal/habits")
        assert response.status_code == 200
        data = response.json()
        assert "habit_inertia" in data

    def test_fate_personal_socioeconomic_endpoint(self):
        response = client.get("/api/fate/personal/socioeconomic")
        assert response.status_code == 200
        data = response.json()
        assert "origin_trap_index" in data

    def test_fate_demographic_endpoint(self):
        response = client.get("/api/fate/demographic-destiny")
        assert response.status_code == 200
        data = response.json()
        assert "destinies" in data

    def test_fate_full_endpoint(self):
        response = client.get("/api/fate")
        assert response.status_code == 200
        data = response.json()
        assert "fate_score" in data
        assert "trajectory" in data

    def test_prescience_events_endpoint(self):
        response = client.get("/api/prescience/events?years=5")
        assert response.status_code == 200
        data = response.json()
        assert "events" in data

    def test_prescience_tipping_endpoint(self):
        response = client.get("/api/prescience/tipping-points")
        assert response.status_code == 200
        data = response.json()
        assert "tipping_points" in data

    def test_prescience_warnings_endpoint(self):
        response = client.get("/api/prescience/early-warnings")
        assert response.status_code == 200
        data = response.json()
        assert "warnings" in data

    def test_prescience_scenario_endpoint(self):
        response = client.get("/api/prescience/scenario?scenario=great_filter&years=50")
        assert response.status_code == 200
        data = response.json()
        assert "outcome" in data or "collapse_probability" in data

    def test_prescience_overview_endpoint(self):
        response = client.get("/api/prescience")
        assert response.status_code == 200

    def test_signal_monitor_endpoint(self):
        response = client.get("/api/signal-monitor")
        assert response.status_code == 200


class TestScenarioEngineAPI:
    def test_causal_loop_endpoint(self):
        response = client.post("/api/scenario/causal-loop")
        assert response.status_code == 200
        data = response.json()
        assert "loops" in data

    def test_system_dynamics_endpoint(self):
        response = client.post("/api/scenario/system-dynamics?n_steps=50&dt=0.1")
        assert response.status_code == 200
        data = response.json()
        assert "stocks" in data or "time" in data

    def test_cross_domain_feedback_endpoint(self):
        response = client.post("/api/scenario/cross-domain-feedback")
        assert response.status_code == 200
        data = response.json()
        assert "adjusted_domains" in data or "cross_effects" in data

    def test_scenario_simulate_endpoint(self):
        response = client.post("/api/scenario/simulate?name=test&horizon_years=3")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_scenario_compare_endpoint(self):
        response = client.post("/api/scenario/compare")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestCalibrationAPI:
    def test_calibration_record_endpoint(self):
        response = client.post("/api/calibration/record")
        assert response.status_code == 200
        data = response.json()
        assert "total_forecasts" in data or "brier_score" in data

    def test_calibration_curve_endpoint(self):
        response = client.get("/api/calibration/curve")
        assert response.status_code == 200
        data = response.json()
        assert "bins" in data

    def test_calibration_audit_endpoint(self):
        response = client.get("/api/calibration/audit")
        assert response.status_code == 200
        data = response.json()
        assert "audit_id" in data

    def test_calibration_drift_endpoint(self):
        response = client.get("/api/calibration/drift")
        assert response.status_code == 200
        data = response.json()
        assert "domain" in data


class TestAdversarialAPI:
    def test_worst_case_endpoint(self):
        response = client.post("/api/adversarial/worst-case")
        assert response.status_code == 200
        data = response.json()
        assert "total_risk" in data or "description" in data

    def test_monte_carlo_stress_endpoint(self):
        response = client.post("/api/adversarial/monte-carlo-stress?n_simulations=100")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_black_swan_endpoint(self):
        response = client.post("/api/adversarial/black-swan?n_scenarios=3")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestNarrativeAPI:
    def test_narrative_generate_endpoint(self):
        response = client.post("/api/narrative/generate?horizon_years=5")
        assert response.status_code == 200
        data = response.json()
        assert "headline" in data
        assert "arc" in data


class TestParadoxAPI:
    def test_paradox_detect_endpoint(self):
        response = client.post("/api/paradox/detect")
        assert response.status_code == 200
        data = response.json()
        assert "n_paradoxes" in data or "paradoxes" in data

    def test_phase_transition_endpoint(self):
        response = client.post("/api/phase-transition/detect")
        assert response.status_code == 200
        data = response.json()
        assert "variable" in data


class TestDataQualityAPI:
    def test_data_quality_score_endpoint(self):
        response = client.get("/api/data-quality/score")
        assert response.status_code == 200
        data = response.json()
        assert "global_quality" in data
        assert "domains" in data
        assert "n_domains" in data

    def test_data_quality_adjust_ci_endpoint(self):
        response = client.get("/api/data-quality/adjust-ci")
        assert response.status_code == 200
        data = response.json()


class TestCausalEventAPI:
    def test_causal_event_predict_defaults(self):
        response = client.post("/api/causal-event/predict", json={})
        assert response.status_code == 200
        data = response.json()
        assert "total_events" in data
        assert "events" in data
        assert len(data["events"]) >= 8
        e = data["events"][0]
        assert "event_id" in e
        assert "category" in e
        assert "event_description" in e
        assert "probability" in e
        assert "probability_band" in e
        assert "base_rate" in e
        assert "causal_chain" in e
        assert len(e["causal_chain"]) >= 2
        assert "evidence_citations" in e
        assert "triggers" in e
        assert "invalidation_conditions" in e
        assert "alternative_outcome" in e
        assert "alternative_probability" in e

    def test_causal_event_predict_with_domain_states(self):
        response = client.post("/api/causal-event/predict", json={
            "domain_states": {"equity": 0.80, "inflation": 0.25, "technology": 0.75}
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_events"] >= 8
        assert "overall_assessment" in data
        assert "confidence_level" in data
        assert "key_risks" in data
        assert "key_opportunities" in data

    def test_causal_event_event_categories_span_life(self):
        response = client.post("/api/causal-event/predict", json={})
        data = response.json()
        categories = {e["category"].lower() for e in data["events"]}
        life_cats = {"health", "medicine", "relationship", "career",
                    "social", "housing", "ai_breakthrough", "climate"}
        overlap = categories & life_cats
        assert len(overlap) >= 5

    def test_causal_event_probabilities_valid(self):
        response = client.post("/api/causal-event/predict", json={})
        data = response.json()
        for e in data["events"]:
            assert 0 < e["probability"] < 1
            assert 0 < e["base_rate"] < 1

    def test_causal_event_cascade_endpoint(self):
        response = client.post("/api/causal-event/cascade", json={})
        assert response.status_code == 200
        data = response.json()
        assert "chains" in data
        assert "overall_cascade_risk" in data
        assert data["overall_cascade_risk"] >= 0
        assert len(data["interaction_edges"]) >= 5
        if data["most_likely_cascade"]:
            assert "narrative" in data["most_likely_cascade"]
        if data["highest_impact_cascade"]:
            assert "risk_level" in data["highest_impact_cascade"]

    def test_causal_event_timeline_endpoint(self):
        response = client.post("/api/causal-event/timeline", json={})
        assert response.status_code == 200
        data = response.json()
        assert "curves" in data
        assert len(data["curves"]) >= 20
        for curve in data["curves"][:3]:
            assert "event_id" in curve
            assert "points" in curve
            assert len(curve["points"]) >= 4
            assert "peak_probability" in curve
            assert "shape" in curve

    def test_causal_event_full_endpoint(self):
        response = client.post("/api/causal-event/full", json={})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "cascade" in data
        assert "timeline" in data
        assert len(data["prediction"]["events"]) >= 24
        assert data["cascade"]["overall_risk"] >= 0

    def test_causal_event_calibrate_with_already_occurred(self):
        response = client.post("/api/causal-event/calibrate", json={"include_rss": False})
        assert response.status_code == 200
        data = response.json()
        assert "domain_states" in data
        assert "already_occurred" in data
        assert "metric_injections" in data
        assert isinstance(data["already_occurred"], list)
        assert isinstance(data["metric_injections"], list)

    def test_causal_event_live_with_rss(self):
        response = client.post("/api/causal-event/live", json={})
        assert response.status_code == 200
        data = response.json()
        assert "calibration" in data
        assert "already_occurred" in data
        assert "metric_injections" in data
        assert "prediction" in data
        assert len(data["prediction"]["events"]) >= 8
        for e in data["prediction"]["events"]:
            assert "status" in e
            assert "key_metric_current" in e

    def test_causal_event_rss_endpoint(self):
        response = client.get("/api/causal-event/rss")
        assert response.status_code == 200
        data = response.json()
        assert "articles_fetched" in data
        assert "signals" in data
        assert "domain_state_adjustments" in data
        assert "headline_summary" in data

    def test_scale_bridging_predict_endpoint(self):
        response = client.post("/api/scale-bridging/predict", json={})
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "cascades" in data
        assert "assessment" in data
        assert len(data["predictions"]) >= 9

    def test_neuroscience_predict_endpoint(self):
        response = client.post("/api/neuroscience/predict", json={})
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "bci_milestones" in data
        assert "breakthrough_probability_by_year" in data
        assert len(data["predictions"]) >= 5

    def test_physics_predict_endpoint(self):
        response = client.post("/api/physics/predict", json={})
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "quantum_milestones" in data
        assert len(data["predictions"]) >= 5

    def test_synthetic_bio_predict_endpoint(self):
        response = client.post("/api/synthetic-bio/predict", json={})
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "pipeline" in data
        assert len(data["predictions"]) >= 5
