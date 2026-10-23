#!/usr/bin/env python3
"""Oracle - Prediction Engine Demo"""
import sys
sys.path.insert(0, ".")

import numpy as np
from datetime import date, timedelta
from oracle.core.config import OracleConfig
from oracle.core.types import Domain, ConfidenceLevel
from oracle.prediction.engine import OracleEngine
from oracle.prediction.timeslider import TimeSlider
from oracle.deterministic.dalio_cycles import DalioCycleEngine, DebtCycleState
from oracle.deterministic.economic_model import EconomicDeterministicModel, MacroState
from oracle.deterministic.physics_model import PhysicsModel
from oracle.deterministic.statistical_model import StatisticalDeterministicModel
from oracle.nondeterministic.buffett_value import BuffettValueEngine, CompanyFundamentals
from oracle.nondeterministic.monte_carlo import MonteCarloEngine
from oracle.nondeterministic.bayesian import BayesianEngine
from oracle.patterns.correlation_engine import CorrelationEngine
from oracle.patterns.cycle_detector import CycleDetector
from oracle.patterns.anomaly_detector import AnomalyDetector
from oracle.patterns.regime_detector import RegimeDetector
from oracle.domains.geopolitical import GeopoliticalRiskModel, GeopoliticalState
from oracle.domains.climate import ClimateModel, ClimateState
from oracle.domains.sentiment import SentimentEngine, SentimentData
from oracle.domains.contagion import ContagionModel, ContagionState
from oracle.domains.demographics import DemographicsModel, DemographicsState
from oracle.ingestion.real_data import RealDataFetcher, EconomicDataFetcher, DomainDataFetcher
from oracle.optimization.portfolio import PortfolioOptimizer
from oracle.optimization.options import OptionsPricer
from oracle.patterns.technical_analysis import TechnicalAnalysis
from oracle.domains.yield_curve import YieldCurveModel
from oracle.domains.risk_decomposition import RiskDecomposition
from oracle.domains.epidemiology import EpidemiologyModel
from oracle.domains.technology import TechnologyForecaster
from oracle.domains.social import SocialDynamicsModel
from oracle.prediction.stress_testing import StressTestingEngine
from oracle.prediction.robust_optimizer import RobustOptimizer, RobustMethod
from oracle.prediction.red_team import RedTeamForecaster
from oracle.prediction.ensemble_calibrator import EnsembleBeliefCalibrator, CalibratorType
from oracle.prediction.temporal_consistency import TemporalConsistencyChecker
from oracle.prediction.forecast_decomposition import ForecastDecomposer, DecompositionMethod
from oracle.prediction.scenario_probability import ScenarioProbabilityMapper, ScenarioRelation
from oracle.prediction.predictive_fate import PredictiveFateEngine
from oracle.prediction.existential_risk import ExistentialRiskAssessor
from oracle.prediction.prophecy import OracleProphecySynthesizer, ProphecyHorizon
from oracle.prediction.personal_oracle import PersonalOracle, PersonInput


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_dalio_cycles():
    separator("DALIO-STYLE ECONOMIC CYCLE ENGINE")
    
    engine = DalioCycleEngine()
    
    us_2024 = DebtCycleState(
        gdp_growth_rate=0.025, debt_to_gdp=1.23, interest_rate=0.0525,
        money_supply_growth=0.04, credit_growth=0.035, productivity_growth=0.018,
        inflation_rate=0.035, unemployment_rate=0.038, stock_valuation=1.5,
        capacity_utilization=0.79, timestamp=date(2024, 1, 1),
    )
    
    diagnosis = engine.diagnose_cycle(us_2024)
    print(f"  Phase: {diagnosis.phase.value.upper()}")
    print(f"  Short-term cycle position: {diagnosis.short_term_cycle_position:.2f}")
    print(f"  Long-term cycle position: {diagnosis.long_term_cycle_position:.2f}")
    print(f"  Debt sustainability score: {diagnosis.debt_sustainability_score:.2f}")
    print(f"  Bubble risk: {diagnosis.bubble_risk_score:.2f}")
    print(f"  Deleveraging risk: {diagnosis.deleveraging_risk_score:.2f}")
    print(f"  Policy effectiveness: {diagnosis.policy_effectiveness:.2f}")
    print(f"  Confidence: {diagnosis.confidence:.2f}")
    print(f"  Historical analogues: {', '.join(diagnosis.historical_analogues[:3])}")
    
    print(f"\n  Simulating forward 10 years:")
    states = engine.simulate_cycle_forward(us_2024, years=10, steps=120)
    print(f"  Year  | GDP Growth | Debt/GDP | Rate   | Inflation | Unemploy | Stock Val")
    print(f"  ------|-----------|----------|--------|-----------|----------|----------")
    for i in [0, 24, 48, 72, 96, 119]:
        if i < len(states):
            s = states[i]
            year = us_2024.timestamp.year + i // 12
            print(f"  {year}  | {s.gdp_growth_rate:+.4f}   | {s.debt_to_gdp:.3f}   | {s.interest_rate:.4f} | {s.inflation_rate:+.4f}    | {s.unemployment_rate:.4f}   | {s.stock_valuation:.3f}")
    
    predictions = engine.predict_from_cycle(us_2024, date(2030, 1, 1), ConfidenceLevel.HIGH)
    print(f"\n  Predictions for 2030:")
    for p in predictions[:3]:
        print(f"    {p.variable}: {p.value:.4f} [{p.confidence_interval[0]:.4f}, {p.confidence_interval[1]:.4f}]")


def demo_buffett_valuation():
    separator("BUFFETT-STYLE FUNDAMENTAL VALUATION")
    
    engine = BuffettValueEngine()
    
    companies = [
        ("Quality Compounder", CompanyFundamentals(
            name="QualityCo", ticker="QCO", earnings_per_share=8.0, book_value_per_share=55.0,
            free_cash_flow_per_share=6.5, dividends_per_share=2.0, revenue_growth_5yr=0.12,
            earnings_growth_5yr=0.15, roe=0.25, roic=0.20, debt_to_equity=0.4,
            current_ratio=2.5, gross_margin=0.55, operating_margin=0.30, net_margin=0.18,
            payout_ratio=0.30, market_price=200.0, sector="tech",
        )),
        ("Mature Value", CompanyFundamentals(
            name="ValueCorp", ticker="VCRP", earnings_per_share=6.5, book_value_per_share=50.0,
            free_cash_flow_per_share=5.0, dividends_per_share=3.0, revenue_growth_5yr=0.04,
            earnings_growth_5yr=0.05, roe=0.12, roic=0.10, debt_to_equity=0.6,
            current_ratio=1.8, gross_margin=0.38, operating_margin=0.18, net_margin=0.10,
            payout_ratio=0.45, market_price=85.0, sector="industrial",
        )),
    ]
    
    for name, company in companies:
        print(f"\n  === {name}: {company.ticker} @ ${company.market_price} ===")
        moat = engine.assess_moat(company)
        print(f"  Moat Score: {moat.overall_moat_score:.2f} ({moat.moat_type} moat, {moat.moat_durability_years:.0f}yr durability)")
        print(f"    Brand: {moat.brand_strength:.2f} | Switching: {moat.switching_costs:.2f}")
        print(f"    Network: {moat.network_effects:.2f} | Cost: {moat.cost_advantages:.2f}")
        
        valuation = engine.full_valuation(company)
        print(f"  Intrinsic Value: ${valuation.intrinsic_value:.2f}")
        print(f"  DCF Value:       ${valuation.dcf_value:.2f}")
        print(f"  Graham Number:   ${valuation.graham_value:.2f}")
        print(f"  EPV:             ${valuation.earnings_power_value:.2f}")
        print(f"  Fair Value Range: ${valuation.fair_value_range[0]:.2f} - ${valuation.fair_value_range[1]:.2f}")
        print(f"  Margin of Safety: {valuation.margin_of_safety:.1%}")
        print(f"  Quality Score: {valuation.quality_score:.2f}")
        print(f"  Risk Level: {valuation.risk_level}")
        print(f"  Recommendation: {valuation.recommendation.upper()}")
        
        pred = engine.predict(company, date(2030, 1, 1))
        print(f"  2030 Price Prediction: ${pred.value:.2f} [{pred.confidence_interval[0]:.2f}, {pred.confidence_interval[1]:.2f}]")


def demo_monte_carlo():
    separator("MONTE CARLO STOCHASTIC SIMULATION")
    
    engine = MonteCarloEngine(n_simulations=10000)
    
    print("  Geometric Brownian Motion (S&P 500-like):")
    gbm = engine.geometric_brownian_motion(s0=5000, mu=0.08, sigma=0.18, n_steps=252)
    print(f"    1-Year Expected: ${gbm.mean[-1]:.0f}")
    print(f"    5th Percentile:  ${gbm.percentiles['p05'][-1]:.0f}")
    print(f"    95th Percentile:  ${gbm.percentiles['p95'][-1]:.0f}")
    print(f"    Prob > 6000: {gbm.probability_above(6000):.1%}")
    print(f"    Prob < 4000: {gbm.probability_below(4000):.1%}")
    print(f"    Value at Risk (95%): ${gbm.value_at_risk(0.95):.0f}")
    
    print("\n  Regime-Switching Model (Bull/Bear/Neutral):")
    rs = engine.regime_switching(s0=5000, n_steps=252)
    print(f"    1-Year Expected: ${rs.mean[-1]:.0f}")
    print(f"    5th Percentile:  ${rs.percentiles['p05'][-1]:.0f}")
    print(f"    95th Percentile:  ${rs.percentiles['p95'][-1]:.0f}")
    print(f"    Max Drawdown: {rs.max_drawdown():.1%}")
    
    print("\n  Mean-Reverting (Interest Rate Simulation):")
    mr = engine.mean_reverting(x0=0.05, theta=0.3, mu=0.04, sigma=0.02, n_steps=252)
    print(f"    1-Year Expected Rate: {mr.mean[-1]:.4f}")
    print(f"    5th Percentile:  {mr.percentiles['p05'][-1]:.4f}")
    print(f"    95th Percentile:  {mr.percentiles['p95'][-1]:.4f}")


def demo_patterns():
    separator("PATTERN RECOGNITION & CROSS-DOMAIN CORRELATION")
    
    np.random.seed(42)
    n = 500
    
    gdp = np.cumsum(np.random.normal(0.0002, 0.01, n)) + 100
    inflation = 0.3 * gdp + np.random.normal(0, 0.5, n) - 15
    unemployment = -0.2 * gdp + np.random.normal(0, 0.3, n) + 25
    stock = 1.5 * gdp + np.random.normal(0, 2, n) - 50
    
    print("  Cycle Detection:")
    detector = CycleDetector()
    cycles = detector.detect_cycles(gdp)
    for c in cycles[:3]:
        print(f"    Period: {c.dominant_period:.1f} steps | Strength: {c.strength:.4f} | Frequency: {c.dominant_frequency:.6f}")
    
    print("\n  Cross-Domain Correlations:")
    corr_engine = CorrelationEngine()
    series_dict = {"gdp": gdp, "inflation": inflation, "unemployment": unemployment, "stock_market": stock}
    domains = {"gdp": Domain.ECONOMICS, "inflation": Domain.ECONOMICS, "unemployment": Domain.ECONOMICS, "stock_market": Domain.FINANCE}
    correlations = corr_engine.discover_cross_domain_patterns(series_dict, domains)
    for c in correlations[:5]:
        print(f"    {c.source_a} <-> {c.source_b}: r={c.correlation:.3f} ({c.strength} {c.direction}), lag={c.lag_days:.0f}")
    
    print("\n  Anomaly Detection:")
    anomalous = gdp.copy()
    anomalous[100] += 5 * np.std(gdp)
    anomalous[300] -= 4 * np.std(gdp)
    detector_a = AnomalyDetector()
    anomalies = detector_a.detect_all(anomalous)
    print(f"    Found {len(anomalies)} anomalies")
    for a in anomalies[:3]:
        print(f"      Index {a.index}: value={a.value:.2f}, type={a.anomaly_type}, score={a.score:.2f}")
    
    print("\n  Regime Detection:")
    regime_det = RegimeDetector()
    result = regime_det.detect_hmm_regimes(stock)
    print(f"    Current regime: {result.current_regime}")
    print(f"    Regime probabilities: {result.regime_probabilities}")
    print(f"    Number of regime changes: {len(result.regimes)}")


def demo_oracle_prediction():
    separator("ORACLE ENSEMBLE PREDICTION ENGINE")
    
    config = OracleConfig()
    engine = OracleEngine(config)
    
    engine.set_economic_state(
        gdp_growth_rate=0.025, debt_to_gdp=1.23, interest_rate=0.045,
        inflation_rate=0.03, unemployment_rate=0.04, stock_valuation=1.3,
    )
    
    engine.set_company(
        ticker="AAPL", eps=6.50, book_value=3.95, fcf=6.0,
        earnings_growth=0.08, roe=0.45, price=185.0,
    )
    
    np.random.seed(42)
    engine.add_time_series("gdp_index", np.cumsum(np.random.normal(0.0002, 0.01, 120)) + 100, Domain.ECONOMICS)
    engine.add_time_series("stock_market", np.cumsum(np.random.normal(0.0005, 0.015, 120)) + 100, Domain.FINANCE)
    
    target = date(2030, 1, 1)
    forecast = engine.predict(target, ConfidenceLevel.HIGH)
    
    print(f"  Forecast for {target.isoformat()}:")
    print(f"  Total predictions: {len(forecast.predictions)}")
    print(f"  Domains: {[d.value for d in forecast.domains]}")
    
    ensemble = [p for p in forecast.predictions if p.model_name == "oracle_ensemble"]
    print(f"\n  Ensemble Predictions:")
    for p in ensemble[:5]:
        ci = p.confidence_interval
        print(f"    {p.domain.value}:{p.variable} = {p.value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    diagnosis = engine.diagnose()
    cycle = diagnosis.get("cycle_diagnosis", {})
    if cycle:
        print(f"\n  Economic Cycle Diagnosis:")
        print(f"    Phase: {cycle.get('phase', 'N/A')}")
        print(f"    Debt sustainability: {cycle.get('debt_sustainability', 'N/A')}")
        print(f"    Bubble risk: {cycle.get('bubble_risk', 'N/A')}")
        print(f"    Analogues: {cycle.get('historical_analogues', [])}")


def demo_timeslider():
    separator("TIME SLIDER - SCENARIO ANALYSIS")
    
    engine = OracleEngine()
    engine.set_economic_state(
        gdp_growth_rate=0.02, debt_to_gdp=1.2, interest_rate=0.045,
        inflation_rate=0.03, unemployment_rate=0.04,
    )
    
    slider = TimeSlider(engine)
    
    print("  Quick Scan at multiple horizons:")
    scans = slider.quick_scan([90, 365, 1825, 3650])
    for ts in scans:
        years = ts.years_ahead
        n_preds = ts.summary.get("n_predictions", 0)
        ensemble_vals = ts.summary.get("ensemble_values", {})
        print(f"\n  {ts.target_date.isoformat()} ({years:.1f} years ahead):")
        print(f"    {n_preds} total predictions")
        for key, val in list(ensemble_vals.items())[:3]:
            print(f"    {key}: {val['value']:.4f} [{val['ci_lower']:.4f}, {val['ci_upper']:.4f}]")
    
    print("\n  Scenario Analysis (Baseline vs Bull vs Bear vs Stagflation):")
    scenarios = {
        "baseline": {},
        "bull": {"gdp_growth_rate": 0.04, "inflation_rate": 0.02, "unemployment_rate": 0.03},
        "bear": {"gdp_growth_rate": -0.01, "inflation_rate": 0.06, "unemployment_rate": 0.08},
        "stagflation": {"gdp_growth_rate": 0.005, "inflation_rate": 0.08, "unemployment_rate": 0.07},
    }
    
    target = date(2030, 1, 1)
    # Run scenarios individually to show results
    for name, params in scenarios.items():
        engine.set_economic_state(
            gdp_growth_rate=params.get("gdp_growth_rate", 0.02),
            debt_to_gdp=params.get("debt_to_gdp", 1.2),
            interest_rate=params.get("interest_rate", 0.045),
            inflation_rate=params.get("inflation_rate", 0.03),
            unemployment_rate=params.get("unemployment_rate", 0.04),
        )
        forecast = engine.predict(target, ConfidenceLevel.MEDIUM)
        ensemble = [p for p in forecast.predictions if p.model_name == "oracle_ensemble"]
        gdp_pred = [p for p in ensemble if "gdp" in p.variable.lower()]
        if gdp_pred:
            p = gdp_pred[0]
            print(f"    {name:12s}: GDP growth = {p.value:+.4f} [{p.confidence_interval[0]:+.4f}, {p.confidence_interval[1]:+.4f}]")
        else:
            print(f"    {name:12s}: {len(ensemble)} ensemble predictions")


def demo_physics():
    separator("PHYSICS-BASED MODELS")
    
    model = PhysicsModel()
    np.random.seed(42)
    series = np.cumsum(np.random.normal(0.01, 0.05, 500)) + 100
    
    print("  Coupled Oscillator (Economic Variables Coupling):")
    states = [(1.0, 0.0), (0.5, 0.1), (-0.3, -0.05)]
    trajectory = model.coupled_oscillator(states, n_steps=500)
    print(f"    Trajectory shape: {trajectory.shape}")
    print(f"    Final positions: {trajectory[-1].round(3)}")
    
    print("\n  Hurst Exponent (Long-Term Memory):")
    result = model.multi_scale_prediction(series, n_ahead=100, n_sims=100)
    print(f"    Hurst Exponent: {result['hurst_exponent']:.3f}")
    print(f"    {'Persistent (trend-following)' if result['hurst_exponent'] > 0.5 else 'Mean-reverting' if result['hurst_exponent'] < 0.5 else 'Random walk'}")
    print(f"    50th percentile: {result['predictions']['p50'][-1]:.2f}")
    print(f"    95th percentile: {result['predictions']['p95'][-1]:.2f}")
    
    print("\n  Entropy Flow (Information Dynamics):")
    entropy = model.entropy_flow(series)
    print(f"    Mean entropy: {entropy['mean_entropy']:.3f}")
    print(f"    Entropy trend: {entropy['entropy_trend']:.4f}")
    print(f"    Phase transitions detected: {len(entropy['phase_transitions'])}")
    
    print("\n  Critical Point Detection:")
    critical = model.critical_point_detection(series)
    print(f"    Mean variance: {critical['mean_variance']:.4f}")
    print(f"    Variance trend: {critical['variance_trend']:.4f}")
    print(f"    Critical points found: {len(critical['critical_points'])}")


def demo_bayesian():
    separator("BAYESIAN INFERENCE ENGINE")
    
    engine = BayesianEngine()
    np.random.seed(42)
    
    print("  Normal-Normal Conjugate Update:")
    data = np.random.normal(0.5, 0.3, 100)
    result = engine.conjugate_normal_normal(data, prior_mu=0.0, prior_sigma=1.0, known_sigma=0.3)
    print(f"    Prior: N(0.0, 1.0)")
    print(f"    Posterior: N({result.mean:.4f}, {result.std:.4f})")
    print(f"    95% Credible Interval: [{result.credible_interval[0]:.4f}, {result.credible_interval[1]:.4f}]")
    print(f"    Bayes Factor: {result.bayes_factor:.2f}")
    
    print("\n  Beta-Binomial (Probability Estimation):")
    result = engine.conjugate_beta_binomial(successes=65, trials=100)
    print(f"    Data: 65 successes out of 100 trials")
    print(f"    Posterior probability: {result.mean:.4f} +/- {result.std:.4f}")
    print(f"    95% Credible Interval: [{result.credible_interval[0]:.4f}, {result.credible_interval[1]:.4f}]")
    
    print("\n  Model Comparison:")
    data = np.random.normal(0.3, 0.5, 50)
    comparison = engine.model_comparison(data, null_mu=0.0)
    print(f"    H0: mu=0.0 vs H1: mu != 0.0")
    print(f"    Bayes Factor: {comparison['bayes_factor']:.2f}")
    print(f"    Favor null: {comparison['favor_null']}")
    print(f"    Favor alternative: {comparison['favor_alternative']}")
    
    print("\n  Bayesian Change Point Detection:")
    data = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)])
    cp = engine.bayesian_change_point(data)
    print(f"    Change points detected: {cp['change_points']}")
    print(f"    Number of segments: {cp['n_segments']}")


def demo_geopolitical():
    separator("GEOPOLITICAL RISK MODEL")
    
    model = GeopoliticalRiskModel()
    state = GeopoliticalState(
        us_china_tension=0.6, russia_west_tension=0.7, middle_east_instability=0.5,
        trade_protectionism=0.3, cyber_warfare_likelihood=0.4, nuclear_proliferation_risk=0.1,
    )
    
    risk = model.assess_risk(state)
    print(f"  Overall Risk: {risk['overall_risk']:.3f}")
    print(f"  Stability Score: {risk['stability_score']:.3f}")
    print(f"  Conflict Risk: {risk['conflict_risk']:.3f}")
    print(f"  Economic Impact: {risk['economic_impact_score']:.3f}")
    
    events = model.simulate_events(state, months=12, n_simulations=500)
    print(f"\n  Simulated Events (12 months, 500 sims):")
    for name, data in list(events.items())[:5]:
        if isinstance(data, dict) and 'mean_impact' in data:
            print(f"    {name}: prob={data.get('probability', 0):.2f}, mean_impact={data['mean_impact']:.3f}")
    
    preds = model.predict(state, date(2030, 1, 1))
    print(f"\n  Predictions for 2030:")
    for p in preds[:3]:
        print(f"    {p.domain.value}:{p.variable} = {p.value:.4f} [{p.confidence_interval[0]:.4f}, {p.confidence_interval[1]:.4f}]")


def demo_climate():
    separator("CLIMATE MODEL")
    
    model = ClimateModel()
    state = ClimateState(global_temp_anomaly_c=1.2, co2_ppm=420, sea_level_rise_mm=3.7)
    
    projections = model.project_temperature(state, years_ahead=30, n_simulations=100)
    print("  Temperature Projections (30 years):")
    for scenario in ["aggressive_mitigation", "current_policies", "high_emissions"]:
        if scenario in projections:
            p = projections[scenario]
            print(f"    {scenario}: +{p['temp_2080_median']:.2f}C by 2080, range [{p['temp_2080_p05']:.2f}, {p['temp_2080_p95']:.2f}]")
    
    tipping = model.tipping_points(state)
    print(f"\n  Tipping Points:")
    for tp in tipping.get("tipping_points", [])[:5]:
        status = "TRIGGERED" if tp["triggered"] else f"proximity={tp['proximity']:.2f}"
        print(f"    {tp['name']}: {status}")
    
    carbon = model.carbon_pricing_impact(50.0)
    print(f"\n  Carbon Pricing ($50/ton):")
    print(f"    Emission reduction: {carbon['emission_reduction_pct']:.1f}%")
    print(f"    GDP impact: {carbon['gdp_impact_pct']:.2f}%")


def demo_sentiment():
    separator("SENTIMENT ENGINE")
    
    engine = SentimentEngine()
    data = SentimentData(
        fear_greed_index=0.35, vix_level=0.6, put_call_ratio=1.1,
        consumer_confidence=0.55, social_media_sentiment=0.45, short_interest=0.35,
    )
    
    composite = engine.compute_composite_index(data)
    print(f"  Composite Index: {composite['composite_index']:.3f}")
    print(f"  Interpretation: {composite['interpretation']}")
    print(f"  Signal: {composite['signal']}")
    print(f"  Momentum: {composite['momentum']:.4f}")
    
    forecast = engine.predict_sentiment(data, days_ahead=30, n_simulations=500)
    print(f"\n  30-Day Sentiment Forecast:")
    print(f"    Current: {forecast['current']:.3f}")
    print(f"    P50: {forecast['forecasts']['p50']:.3f}")
    print(f"    P05-P95: [{forecast['forecasts']['p05']:.3f}, {forecast['forecasts']['p95']:.3f}]")
    print(f"    P(bullish): {forecast['prob_bullish']:.1%}")
    print(f"    P(bearish): {forecast['prob_bearish']:.1%}")


def demo_contagion():
    separator("CONTAGION MODEL")
    
    model = ContagionModel(n_nodes=15)
    state = ContagionState(financial_systemic_risk=0.4, trade_linkage=0.5)
    
    sir = model.sir_model(population=10000, beta=0.3, gamma=0.1, days=365)
    print("  SIR Epidemiological Model:")
    print(f"    R0: {sir['r0']:.2f}")
    print(f"    Peak day: {sir['peak_day']}")
    print(f"    Peak infected: {sir['peak_infected_pct']:.1%}")
    print(f"    Total infected: {sir['total_infected_pct']:.1%}")
    
    financial = model.financial_contagion(state, n_simulations=100, time_steps=20)
    print(f"\n  Financial Contagion:")
    print(f"    Mean systemic stress: {financial['mean_systemic_stress']:.3f}")
    print(f"    P95 systemic stress: {financial['p95_systemic_stress']:.3f}")
    print(f"    Cascade probability: {financial['cascade_probability']:.3f}")
    
    cascade = model.cascade_threshold_model(state, n_agents=100)
    print(f"\n  Cascade Threshold Model:")
    print(f"    Cascade size: {cascade['cascade_size']}/{cascade.get('seeds', 'N/A')} seeds")
    print(f"    Cascade pct: {cascade['cascade_pct']:.1%}")
    print(f"    Full cascade: {cascade['full_cascade']}")


def demo_demographics():
    separator("DEMOGRAPHICS MODEL")
    
    model = DemographicsModel()
    state = DemographicsState(
        total_population_m=8100, birth_rate_per_1000=18, death_rate_per_1000=8,
        fertility_rate=2.3, elderly_pct=0.10, median_age=30.5,
    )
    
    stage = model.classify_stage(state)
    print(f"  Demographic Stage: {stage}")
    
    proj = model.project_population(state, years=30, n_simulations=100)
    print(f"\n  Population Projections (30 years):")
    for scenario in ["baseline", "high_growth", "low_growth", "decline"]:
        if scenario in proj:
            p = proj[scenario]
            print(f"    {scenario}: {p['pop_final_median']:.0f}M (P05: {p['pop_final_p05']:.0f}M, P95: {p['pop_final_p95']:.0f}M)")
    
    dividend = model.aging_dividend(state)
    print(f"\n  Aging Dividend Analysis:")
    print(f"    Stage: {dividend['stage']}")
    print(f"    Working age pct: {dividend['working_age_pct']:.1%}")
    print(f"    GDP growth bonus: {dividend['gdp_growth_bonus_pct']:.2f}%")
    print(f"    Pension sustainability: {dividend['pension_sustainability']:.2f}")
    print(f"    Healthcare pressure: {dividend['healthcare_pressure']:.2f}")


def demo_real_data():
    separator("REAL MARKET DATA (YAHOO FINANCE)")
    
    fetcher = RealDataFetcher(cache_ttl=300)
    
    print("  Fetching live market data...")
    snapshot = fetcher.get_economic_snapshot()
    print(f"\n  Market Snapshot:")
    for name, value in sorted(snapshot.items()):
        print(f"    {name:20s}: {value:>12,.2f}")
    
    print(f"\n  S&P 500 Returns (1Y):")
    sp500_returns = fetcher.compute_returns("sp500", period="1y")
    if sp500_returns:
        print(f"    Current Price:      ${sp500_returns['current']:>10,.2f}")
        print(f"    1Y Return:          {sp500_returns['cumulative_return']:>10.2%}")
        print(f"    Annualized Vol:     {sp500_returns['annualized_volatility']:>10.2%}")
        print(f"    Max Drawdown:       {sp500_returns['max_drawdown']:>10.2%}")
        print(f"    Sharpe Ratio:       {sp500_returns['sharpe_ratio']:>10.2f}")
    
    print(f"\n  S&P 500 Cycle Detection:")
    sp500_data = fetcher.get_close_array("sp500", period="2y")
    if sp500_data is not None and len(sp500_data) > 30:
        detector = CycleDetector()
        cycles = detector.detect_cycles(sp500_data)
        for c in cycles[:3]:
            print(f"    Period: {c.dominant_period:.1f} days | Strength: {c.strength:.4f}")
        
        detector_a = AnomalyDetector()
        anomalies = detector_a.detect_all(sp500_data)
        print(f"    Anomalies detected: {len(anomalies)}")
        for a in anomalies[:3]:
            print(f"      Index {a.index}: value={a.value:.1f}, type={a.anomaly_type}")
        
        print(f"\n  Real Data Oracle Prediction:")
        engine = OracleEngine()
        engine.set_economic_state(gdp_growth_rate=0.025, inflation_rate=0.03, unemployment_rate=0.04)
        engine.add_time_series("sp500_real", sp500_data, Domain.FINANCE)
        forecast = engine.predict(date(2027, 1, 1), ConfidenceLevel.HIGH)
        ensemble = [p for p in forecast.predictions if p.model_name == "oracle_ensemble"]
        for p in ensemble[:3]:
            print(f"    {p.domain.value}:{p.variable} = {p.value:.2f} [{p.confidence_interval[0]:.2f}, {p.confidence_interval[1]:.2f}]")
    else:
        print("    (Could not fetch S&P 500 data - using synthetic)")
    
    multi_data = fetcher.fetch_multi(["sp500", "vix", "gold"], period="6mo")
    if multi_data:
        print(f"\n  Multi-Asset Correlations (6-month):")
        corr_engine = CorrelationEngine()
        series_dict = {}
        domains = {}
        for name, arr in multi_data.items():
            min_len = min(len(arr) for arr in multi_data.values())
            series_dict[name] = arr[-min_len:]
            domains[name] = Domain.FINANCE
        if len(series_dict) >= 2:
            correlations = corr_engine.discover_cross_domain_patterns(series_dict, domains)
            for c in correlations[:5]:
                print(f"    {c.source_a} <-> {c.source_b}: r={c.correlation:.3f} ({c.strength} {c.direction})")

    econ_state = EconomicDataFetcher().get_economic_state_approx()
    print(f"\n  Economic State (Real Data Informed):")
    for key, val in sorted(econ_state.items()):
        print(f"    {key:25s}: {val:>10.4f}" if isinstance(val, float) and abs(val) < 10 else f"    {key:25s}: {val:>10.1f}")


def demo_portfolio_optimizer():
    separator("PORTFOLIO OPTIMIZER (v0.3)")

    fetcher = RealDataFetcher(cache_ttl=3600)
    opt = PortfolioOptimizer(n_simulations=2000, risk_free_rate=0.05)

    print("  Fetching real market data...")
    multi = fetcher.fetch_multi(["sp500", "gold", "treasury_10y", "bitcoin"], period="2y")

    if len(multi) >= 2:
        print(f"  Got data for: {list(multi.keys())}")
        result = opt.optimize_from_prices(multi, n_simulations=2000)

        print(f"\n  Optimal Portfolio:")
        print(f"    Expected Return:  {result.optimal_return * 100:.2f}%")
        print(f"    Volatility:       {result.optimal_volatility * 100:.2f}%")
        print(f"    Sharpe Ratio:      {result.optimal_sharpe:.4f}")
        print(f"\n  Optimal Weights:")
        for name, weight in result.optimal_weights.items():
            print(f"    {name:15s}: {weight*100:.1f}%")

        print(f"\n  Conservative Portfolio:")
        print(f"    Expected Return:  {result.conservative_return * 100:.2f}%")
        print(f"    Volatility:       {result.conservative_volatility * 100:.2f}%")

        print(f"\n  Individual Asset Metrics:")
        for name in result.individual_returns:
            print(f"    {name:15s}: Return={result.individual_returns[name]*100:.2f}%, Vol={result.individual_volatilities[name]*100:.2f}%")
    else:
        print("  Insufficient real data, running with synthetic data...")
        np.random.seed(42)
        prices = {
            "A": np.cumprod(1 + np.random.normal(0.0005, 0.015, 500)) * 100,
            "B": np.cumprod(1 + np.random.normal(0.0003, 0.01, 500)) * 100,
            "C": np.cumprod(1 + np.random.normal(0.0001, 0.008, 500)) * 100,
        }
        result = opt.optimize_from_prices(prices)
        print(f"  Optimal Sharpe: {result.optimal_sharpe:.4f}")
        print(f"  Optimal Return: {result.optimal_return*100:.2f}%")
        print(f"  Optimal Vol: {result.optimal_volatility*100:.2f}%")


def demo_kelly_and_fibonacci():
    separator("KELLY CRITERION & FIBONACCI (v0.3)")

    fetcher = RealDataFetcher(cache_ttl=3600)
    opt = PortfolioOptimizer(seed=42)

    arr = fetcher.get_close_array("sp500", period="2y")
    if arr is None or len(arr) < 50:
        np.random.seed(42)
        arr = np.cumprod(1 + np.random.normal(0.0005, 0.015, 500)) * 100

    returns = np.diff(arr) / arr[:-1]

    kelly = opt.kelly_criterion(returns)
    print(f"  Kelly Criterion (S&P 500):")
    print(f"    Full Kelly:    {kelly['kelly_fraction']:.4f}")
    print(f"    Half Kelly:    {kelly['half_kelly']:.4f}")
    print(f"    Quarter Kelly: {kelly['quarter_kelly']:.4f}")
    print(f"    Growth Rate:   {kelly['expected_growth_rate']*100:.2f}%")
    print(f"    Ann. Return:   {kelly['annualized_return']*100:.2f}%")
    print(f"    Ann. Variance: {kelly['annualized_variance']*100:.4f}%")

    fib = opt.fibonacci_retracement(arr)
    if "error" not in fib:
        print(f"\n  Fibonacci Retracement (S&P 500):")
        print(f"    High:   ${fib['high']:.2f}")
        print(f"    Low:    ${fib['low']:.2f}")
        print(f"    Current: ${fib['current_price']:.2f}")
        print(f"    Trend:  {fib['trend']}")
        print(f"    Nearest: {fib['nearest_level']} = ${fib['nearest_price']:.2f}")
        print(f"\n  Key Levels:")
        for level, price in fib["retracements"].items():
            print(f"    {level}: ${price:.2f}")
        print(f"\n  Extensions:")
        for ext, price in fib["extensions"].items():
            print(f"    {ext}: ${price:.2f}")

    downside = opt.downside_risk(returns)
    print(f"\n  Downside Risk:")
    print(f"    Deviation:    {downside['downside_deviation']*100:.2f}%")
    print(f"    Frequency:    {downside['downside_frequency']*100:.1f}%")
    print(f"    Avg Loss:     {downside['avg_downside_loss']*100:.4f}%")
    print(f"    Max Loss:     {downside['max_downside_loss']*100:.4f}%")


def demo_risk_assessment():
    separator("MULTI-DOMAIN RISK ASSESSMENT (v0.3)")

    dd = DomainDataFetcher(cache_ttl=3600)
    states = dd.get_all_domain_states()

    domains = ["sentiment", "contagion", "geopolitical", "climate", "demographics", "economic"]
    for domain in domains:
        if domain in states:
            data = states[domain]
            keys = list(data.keys())[:4]
            vals = []
            for k in keys:
                v = data[k]
                if isinstance(v, float):
                    vals.append(f"{k}={v:.3f}")
                else:
                    vals.append(f"{k}={v}")
            print(f"  {domain.capitalize()}: {', '.join(vals)}")

    if "data_sources" in states:
        print(f"\n  Data Sources:")
        for src, desc in states["data_sources"].items():
            print(f"    {src}: {desc}")


def demo_domain_fetcher():
    separator("DOMAIN DATA FETCHER (v0.3)")

    dd = DomainDataFetcher(cache_ttl=3600)

    print("  Sentiment State (VIX-derived):")
    sent = dd.get_sentiment_state()
    print(f"    Fear/Greed Index: {sent['fear_greed_index']:.3f}")
    print(f"    VIX Level:        {sent['vix_level']:.3f}")
    print(f"    Put/Call Ratio:    {sent['put_call_ratio']:.3f}")
    print(f"    Safe Haven Demand: {sent['safe_haven_demand']:.3f}")

    print("\n  Contagion State (Volatility-derived):")
    cont = dd.get_contagion_state()
    print(f"    Systemic Risk:    {cont['financial_systemic_risk']:.3f}")
    print(f"    Banking Contagion: {cont['banking_contagion']:.3f}")
    print(f"    Currency Stress:  {cont['currency_stress']:.3f}")

    print("\n  Geopolitical State (Oil/Gold/USD-derived):")
    geo = dd.get_geopolitical_state()
    print(f"    US-China:         {geo['us_china_tension']:.3f}")
    print(f"    Russia-West:      {geo['russia_west_tension']:.3f}")
    print(f"    Energy Geopolitics: {geo['energy_geopolitics']:.3f}")


    print("\n" + "=" * 70)
    print("  ORACLE - Prediction Engine v0.3.2")
    print("  AI + ML + Deterministic + Non-Deterministic + Statistics + Physics")
    print("  Real Data + Portfolio + Options + Technical Analysis + Risk")
    print("=" * 70)
    
    demo_dalio_cycles()
    demo_buffett_valuation()
    demo_monte_carlo()
    demo_physics()
    demo_bayesian()
    demo_patterns()
    demo_geopolitical()
    demo_climate()
    demo_sentiment()
    demo_contagion()
    demo_demographics()
    demo_real_data()
    demo_oracle_prediction()
    demo_timeslider()
    demo_portfolio_optimizer()
    demo_kelly_and_fibonacci()
    demo_risk_assessment()
    demo_domain_fetcher()
    demo_sector_rotation()
    demo_regime_detection()
    demo_drawdown()
    demo_scenario_real()
    demo_mean_variance()
    demo_var_cvar()
    demo_shrinkage_covariance()
    demo_esg()
    demo_backtest_real_portfolio()
    demo_options_pricing()
    demo_technical_analysis()
    demo_yield_curve()
    demo_risk_decomposition()
    demo_epidemiology()
    demo_technology()
    demo_social_dynamics()
    
    print("\n" + "=" * 70)
    print("  Oracle engine ready. Start web dashboard:")
    print("  cd /home/user/oracle && PYTHONPATH=src uvicorn oracle.web.app:app --host 0.0.0.0 --port 8005")
    print("=" * 70 + "\n")


def demo_mean_variance():
    separator("MEAN-VARIANCE OPTIMIZATION (v0.3.1)")

    from oracle.optimization.portfolio import PortfolioOptimizer
    opt = PortfolioOptimizer(seed=42)
    np.random.seed(42)
    returns = np.random.randn(252, 4) * 0.01 + 0.0003
    names = ["SPY", "GLD", "TLT", "QQQ"]

    result = opt.mean_variance_optimize(returns, names)
    print(f"  Tangency Sharpe: {result['tangency_sharpe']:.4f}")
    print(f"  Tangency Return: {result['tangency_return']*100:.2f}%")
    print(f"  Tangency Vol: {result['tangency_volatility']*100:.2f}%")
    print(f"  Min Var Return: {result['min_var_return']*100:.2f}%")
    print(f"  Min Var Vol: {result['min_var_volatility']*100:.2f}%")
    print(f"  Frontier points: {len(result['efficient_frontier'])}")
    print(f"  Tangency weights: {result['tangency_portfolio']}")


def demo_var_cvar():
    separator("VaR / CVaR ANALYSIS (v0.3.1)")

    from oracle.optimization.portfolio import PortfolioOptimizer
    opt = PortfolioOptimizer(seed=42)
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01

    result = opt.historical_var_cvar(returns)
    summary = result["summary"]
    print(f"  Daily VaR 95%: {summary['daily_var_95']*100:.3f}%")
    print(f"  Daily CVaR 95%: {summary['daily_cvar_95']*100:.3f}%")
    print(f"  Ann. VaR 95%: {summary['annualized_var_95_pct']:.2f}%")
    print(f"  Ann. CVaR 95%: {summary['annualized_cvar_95_pct']:.2f}%")
    print(f"  Max daily loss: {summary['max_daily_loss']*100:.3f}%")
    print(f"  Tail events (>5%): {summary['tail_events_5pct']}")


def demo_shrinkage_covariance():
    separator("SHRINKAGE COVARIANCE (v0.3.1)")

    from oracle.optimization.portfolio import PortfolioOptimizer
    opt = PortfolioOptimizer(seed=42)
    np.random.seed(42)
    returns = np.random.randn(100, 3) * 0.01

    result = opt.shrinkage_covariance(returns, asset_names=["A", "B", "C"])
    print(f"  Shrinkage intensity: {result['shrinkage_intensity']:.4f}")
    print(f"  Observations: {result['n_observations']}")
    print(f"  Assets: {result['n_assets']}")


def demo_sector_rotation():
    separator("SECTOR ROTATION (v0.3)")

    from oracle.ingestion.real_data import RealDataFetcher
    fetcher = RealDataFetcher(cache_ttl=3600)

    sectors = {
        "technology": "XLK", "healthcare": "XLV", "financials": "XLF",
        "energy": "XLE", "consumer_discretionary": "XLY", "industrials": "XLI",
    }

    print("  Sector Momentum (3-month):")
    for name, ticker in sectors.items():
        try:
            arr = fetcher.get_close_array(ticker, "2y")
            if arr is not None and len(arr) > 63:
                returns = np.diff(arr[-63:]) / arr[-64:-1]
                mom = float((arr[-1] / arr[-63] - 1) * 100)
                ann_ret = float(np.mean(returns) * 252 * 100)
                ann_vol = float(np.std(returns, ddof=1) * np.sqrt(252) * 100)
                print(f"    {name:25s}  MoM={mom:+7.2f}%  Vol={ann_vol:6.2f}%  Ret={ann_ret:+7.2f}%")
        except Exception:
            print(f"    {name:25s}  (data unavailable)")


def demo_regime_detection():
    separator("REGIME DETECTION (v0.3)")

    from oracle.ingestion.real_data import RealDataFetcher
    from oracle.patterns.regime_detector import RegimeDetector

    fetcher = RealDataFetcher(cache_ttl=3600)
    for symbol in ["sp500", "nasdaq", "gold"]:
        arr = fetcher.get_close_array(symbol, "2y")
        if arr is not None and len(arr) > 60:
            returns = np.diff(arr) / arr[:-1]
            recent_ret = float(np.mean(returns[-63:]) * 252)
            recent_vol = float(np.std(returns[-63:], ddof=1) * np.sqrt(252))
            print(f"  {symbol}: recent_ann_ret={recent_ret*100:.2f}%, recent_vol={recent_vol*100:.2f}%")

            try:
                detector = RegimeDetector()
                result = detector.detect_hmm_regimes(arr)
                print(f"    Current regime: {result.current_regime}")
            except Exception as e:
                print(f"    HMM regime detection: {e}")


def demo_drawdown():
    separator("DRAWDOWN ANALYSIS (v0.3)")

    from oracle.ingestion.real_data import RealDataFetcher
    from oracle.optimization.portfolio import PortfolioOptimizer

    fetcher = RealDataFetcher(cache_ttl=3600)
    opt = PortfolioOptimizer()

    for symbol in ["sp500", "nasdaq", "gold"]:
        arr = fetcher.get_close_array(symbol, "2y")
        if arr is not None and len(arr) > 30:
            mdd = opt.max_drawdown(arr)
            ann_ret = float(np.mean(np.diff(arr) / arr[:-1]) * 252 * 100)
            print(f"  {symbol}: MaxDD={mdd*100:.2f}%, AnnReturn={ann_ret:.2f}%, Price={arr[-1]:.2f}")


def demo_scenario_real():
    separator("SCENARIO ANALYSIS (REAL DATA) (v0.3)")

    from oracle.ingestion.real_data import RealDataFetcher
    import numpy as np

    fetcher = RealDataFetcher(cache_ttl=3600)

    scenarios = ["baseline", "bull", "bear", "crash", "recovery", "sideways", "volatile_bull"]
    shifts = {"baseline": 0, "bull": 0.001, "bear": -0.001, "crash": -0.003, "recovery": 0.0005, "sideways": 0, "volatile_bull": 0.002}

    for symbol in ["sp500", "gold"]:
        arr = fetcher.get_close_array(symbol, "2y")
        if arr is not None and len(arr) > 30:
            returns = np.diff(arr) / arr[:-1]
            current = float(arr[-1])
            print(f"  {symbol} (current: {current:.2f}):")
            for name in scenarios:
                shift = shifts[name]
                sim = returns + shift
                final_prices = []
                for _ in range(200):
                    sampled = np.random.choice(sim, size=90, replace=True)
                    path = current * np.cumprod(1 + sampled)
                    final_prices.append(float(path[-1]))
                final_arr = np.array(final_prices)
                p50 = float(np.percentile(final_arr, 50))
                print(f"    {name:15s}: P50=${p50:.2f} ({(p50/current-1)*100:+.2f}%)")


def demo_esg():
    separator("ESG RISK SCORING (v0.3.1)")

    from oracle.domains.esg import ESGRiskModel, ESGState

    model = ESGRiskModel()

    print("  Default ESG State:")
    assessment = model.assess(ESGState())
    print(f"    Overall ESG Score: {assessment.overall_esg_score:.3f}")
    print(f"    Environmental Risk: {assessment.environmental_risk:.3f}")
    print(f"    Social Risk:       {assessment.social_risk:.3f}")
    print(f"    Governance Risk:   {assessment.governance_risk:.3f}")
    print(f"    Risk Category:     {assessment.risk_category}")
    print(f"    Trend:             {assessment.trend}")

    print("\n  High-ESG Company (Tech):")
    tech_state = ESGState(
        environmental_score=0.8, social_score=0.75, governance_score=0.85,
        carbon_intensity=0.2, water_stress=0.15,
        labor_practices_score=0.8, board_independence=0.8,
        anti_corruption_score=0.85, data_privacy_score=0.9, supply_chain_transparency=0.7,
    )
    tech_assessment = model.assess(tech_state)
    print(f"    ESG Score: {tech_assessment.overall_esg_score:.3f}")
    print(f"    Category:   {tech_assessment.risk_category}")

    print("\n  Low-ESG Company (Energy):")
    energy_state = ESGState(
        environmental_score=0.2, social_score=0.3, governance_score=0.4,
        carbon_intensity=0.85, water_stress=0.7,
        labor_practices_score=0.35, board_independence=0.3,
        anti_corruption_score=0.35, data_privacy_score=0.3, supply_chain_transparency=0.25,
    )
    energy_assessment = model.assess(energy_state)
    print(f"    ESG Score: {energy_assessment.overall_esg_score:.3f}")
    print(f"    Category:  {energy_assessment.risk_category}")
    print(f"    Concerns:  {', '.join(energy_assessment.key_concerns[:3])}")

    print("\n  Scenario: Improving carbon intensity from 0.85 to 0.2:")
    improved = model.simulate_esg_scenario(energy_state, {"carbon_intensity": 0.2})
    print(f"    Improved ESG Score: {improved.overall_esg_score:.3f} (from {energy_assessment.overall_esg_score:.3f})")


def demo_backtest_real_portfolio():
    separator("REAL PORTFOLIO BACKTEST (v0.3.1)")

    from oracle.ingestion.real_data import RealDataFetcher
    from oracle.optimization.portfolio import PortfolioOptimizer
    import numpy as np

    fetcher = RealDataFetcher(cache_ttl=3600)
    opt = PortfolioOptimizer(seed=42)

    symbols = ["sp500", "gold", "treasury_10y"]
    prices = {}
    for sym in symbols:
        arr = fetcher.get_close_array(sym, "2y")
        if arr is not None and len(arr) > 30:
            prices[sym] = arr

    if len(prices) >= 2:
        names = list(prices.keys())
        min_len = min(len(p) for p in prices.values())
        n = len(names)
        returns_matrix = np.zeros((min_len - 1, n))
        for i, name in enumerate(names):
            arr = prices[name][-min_len:]
            returns_matrix[:, i] = np.diff(arr) / arr[:-1]

        weights = np.array([0.5, 0.3, 0.2][:n])
        weights = weights / weights.sum()
        port_returns = returns_matrix @ weights

        ann_ret = float(np.mean(port_returns) * 252 * 100)
        ann_vol = float(np.std(port_returns, ddof=1) * np.sqrt(252) * 100)
        sharpe = (ann_ret - 5.0) / ann_vol if ann_vol > 0 else 0

        values = np.cumprod(1 + port_returns) * 100000
        mdd = opt.max_drawdown(values)

        print(f"  Portfolio: {', '.join([f'{n}:{w:.0%}' for n, w in zip(names, weights)])}")
        print(f"  Annualized Return:  {ann_ret:.2f}%")
        print(f"  Annualized Vol:    {ann_vol:.2f}%")
        print(f"  Sharpe Ratio:      {sharpe:.3f}")
        print(f"  Max Drawdown:      {mdd*100:.2f}%")
        print(f"  Final Value:       ${values[-1]:,.0f}")


def demo_options_pricing():
    separator("OPTIONS PRICING (Black-Scholes + Greeks)")
    pricer = OptionsPricer()

    S = 100
    K = 105
    T = 1.0
    r = 0.05
    sigma = 0.2

    call = pricer.black_scholes(S, K, T, r, sigma, "call")
    put = pricer.black_scholes(S, K, T, r, sigma, "put")
    call_greeks = pricer.greeks(S, K, T, r, sigma, "call")
    put_greeks = pricer.greeks(S, K, T, r, sigma, "put")
    parity = pricer.put_call_parity(S, K, T, r, call_price=call, put_price=put)

    print(f"  Spot={S}, Strike={K}, T={T}y, r={r}, σ={sigma}")
    print(f"  Call Price:  ${call:.4f}")
    print(f"  Put Price:   ${put:.4f}")
    print(f"  Call Greeks: Δ={call_greeks['delta']:.4f} Γ={call_greeks['gamma']:.4f} Θ={call_greeks['theta']:.4f} V={call_greeks['vega']:.4f} ρ={call_greeks['rho']:.4f}")
    print(f"  Put Greeks:  Δ={put_greeks['delta']:.4f} Γ={put_greeks['gamma']:.4f} Θ={put_greeks['theta']:.4f} V={put_greeks['vega']:.4f} ρ={put_greeks['rho']:.4f}")
    print(f"  Put-Call Parity: {'HOLDS' if parity['parity_holds'] else 'VIOLATED'} (diff={parity['parity_difference']:.6f})")

    iv = pricer.implied_volatility(S, K, T, r, call, "call")
    print(f"  Implied Vol from Call: {iv*100:.2f}%" if iv else "  Implied Vol: N/A")

    binomial = pricer.binomial_tree(S, K, T, r, sigma, 100, "call", american=False)
    print(f"  Binomial (100 steps): ${binomial['binomial_price']:.4f} vs BS ${binomial['black_scholes_price']:.4f}")

    strategy = [
        {"type": "call", "strike": 100, "premium": 5, "quantity": 1, "direction": "long"},
        {"type": "put", "strike": 100, "premium": 4, "quantity": 1, "direction": "long"},
    ]
    pnl = pricer.strategy_pnl(S, strategy)
    max_p = pnl["max_profit"] if isinstance(pnl["max_profit"], str) else f"${pnl['max_profit']:.2f}"
    max_l = pnl["max_loss"] if isinstance(pnl["max_loss"], str) else f"${pnl['max_loss']:.2f}"
    print(f"  Long Straddle: Max Profit={max_p}, Max Loss={max_l}, Breakevens={pnl['breakeven_points']}")


def demo_technical_analysis():
    separator("TECHNICAL ANALYSIS (RSI, MACD, Bollinger, etc.)")
    ta = TechnicalAnalysis()

    fetcher = RealDataFetcher(cache_ttl=3600)
    data = fetcher.fetch_series("^GSPC", period="1y", interval="1d")

    if data is not None and len(data) > 0:
        close = data["Close"].values
        high = data["High"].values if "High" in data else close
        low = data["Low"].values if "Low" in data else close
        volume = data["Volume"].values if "Volume" in data else np.ones(len(close))

        rsi = ta.rsi(close, 14)
        macd_data = ta.macd(close)
        bb = ta.bollinger_bands(close, 20, 2.0)
        sma20 = ta.sma(close, 20)
        sma50 = ta.sma(close, 50)
        ema12 = ta.ema(close, 12)

        valid_rsi = rsi[~np.isnan(rsi)]
        rsi_val = valid_rsi[-1] if len(valid_rsi) > 0 else None
        rsi_signal = "Overbought" if rsi_val and rsi_val > 70 else "Oversold" if rsi_val and rsi_val < 30 else "Neutral"

        valid_sma20 = sma20[~np.isnan(sma20)]
        valid_sma50 = sma50[~np.isnan(sma50)]
        valid_ema12 = ema12[~np.isnan(ema12)]

        macd_last = macd_data["macd_line"][-1] if macd_data["macd_line"] else None
        sig_last = macd_data["signal_line"][-1] if macd_data["signal_line"] else None
        macd_signal = "Bullish" if macd_last is not None and sig_last is not None and macd_last > sig_last else "Bearish"

        upper = bb["upper"][-1] if bb["upper"] else None
        lower = bb["lower"][-1] if bb["lower"] else None
        bb_signal = "Above Upper" if upper and not np.isnan(upper) and close[-1] > upper else "Below Lower" if lower and not np.isnan(lower) and close[-1] < lower else "Within Bands"

        print(f"  S&P 500 Last: ${close[-1]:.2f}")
        print(f"  RSI(14):       {rsi_val:.1f} ({rsi_signal})" if rsi_val else "  RSI(14): N/A")
        print(f"  MACD Signal:   {macd_signal} (MACD={macd_last:.4f}, Signal={sig_last:.4f})" if macd_last is not None else "  MACD: N/A")
        print(f"  Bollinger:     {bb_signal}")
        if len(valid_sma20) > 0:
            print(f"  SMA(20):       ${valid_sma20[-1]:.2f}")
        if len(valid_sma50) > 0:
            print(f"  SMA(50):       ${valid_sma50[-1]:.2f}")
        if len(valid_ema12) > 0:
            print(f"  EMA(12):       ${valid_ema12[-1]:.2f}")
        print(f"  Data Points:   {len(close)}")
    else:
        print("  Market data unavailable (network issue)")


def demo_yield_curve():
    separator("YIELD CURVE (Nelson-Siegel + Forward Rates)")
    ycm = YieldCurveModel()

    ns = ycm.nelson_siegel(beta0=0.05, beta1=-0.02, beta2=0.03, tau=2.0)
    print(f"  Nelson-Siegel: β0={ns['beta0']}, β1={ns['beta1']}, β2={ns['beta2']}, τ={ns['tau']}")
    print(f"  Long Rate:   {ns['long_rate']:.2f}%")
    print(f"  Short Rate:  {ns['short_rate']:.2f}%")
    print(f"  Curvature:   {ns['curvature']:.2f}%")
    print(f"  Yield curve: {', '.join([f'{m}y={r:.2f}%' for m, r in zip(ns['maturities'][:5], ns['rates'][:5])])}")

    fwd = ycm.forward_rates([0.03, 0.035, 0.04, 0.043, 0.045], [1, 2, 3, 5, 10])
    print(f"  Forward rates: {', '.join([f'{m}={r:.2f}%' for m, r in zip(fwd['forward_maturities'], fwd['forward_rates'])])}")

    nss = ycm.nelson_siegel_svensson(beta0=0.05, beta1=-0.02, beta2=0.03, beta3=-0.01, tau1=2.0, tau2=5.0)
    print(f"  NSS (extra β3): {', '.join([f'{m}y={r:.2f}%' for m, r in zip(nss['maturities'][:5], nss['rates'][:5])])}")

    scenario = ycm.yield_curve_scenario(shift_bps=100, steepen_bps=50)
    print(f"  Scenario (+100bp parallel, +50bp steepening): changes = {scenario['changes_bps'][:4]} bps")


def demo_risk_decomposition():
    separator("RISK DECOMPOSITION (Component VaR + Factor Attribution)")
    rd = RiskDecomposition()
    np.random.seed(42)

    returns = np.random.randn(252, 3) * 0.01 + 0.0003
    weights = np.array([0.5, 0.3, 0.2])

    cvar = rd.component_var(returns, weights, 0.95, "historical")
    print(f"  Portfolio VaR (95%): {cvar['portfolio_var']*100:.3f}%")
    for i in range(3):
        print(f"  Asset {i+1}: Weight={cvar['weights'][i]*100:.1f}%, Comp VaR={cvar['component_var'][i]:.6f}, % Contrib={cvar['pct_contribution'][i]:.1f}%")

    rb = rd.risk_budget(returns, weights)
    print(f"  Risk Budget: Vol={rb['portfolio_volatility']*100:.2f}%, Sharpe={rb['sharpe_ratio']:.3f}")
    for i in range(3):
        print(f"  Asset {i+1}: Risk Contrib={rb['pct_risk_contribution'][i]:.1f}%")

    factor_returns = np.random.randn(252, 3) * 0.01
    returns_from_factors = factor_returns @ np.random.randn(3, 3) * 0.5 + np.random.randn(252, 3) * 0.005
    fa = rd.factor_attribution(returns_from_factors[:, :1], factor_returns, np.array([1.0]), ["Market", "Size", "Value"])
    print(f"  Factor Attribution: Alpha={fa['alpha']*100:.3f}%, R²={fa['r_squared']:.3f}")
    for name, beta in fa['factor_betas'].items():
        print(f"  {name}: β={beta:.4f}, contrib={fa['factor_contributions'][name]*100:.3f}%")

    stress = rd.stress_decomposition(returns, weights)
    print(f"  Stress Decomposition: Baseline return={stress['baseline_return']*100:.3f}%")
    for name, s in stress['scenarios'].items():
        print(f"  {name}: Stressed return={s['stressed_return']*100:.3f}%, Loss={s['loss_vs_baseline']*100:.3f}%")


def demo_epidemiology():
    separator("EPIDEMIOLOGY MODEL (SEIR + Health Risk + Network Spread)")
    epi = EpidemiologyModel()

    seir = epi.seir(population=1e6, initial_infected=100, beta=0.3, days=365)
    print(f"  SEIR (Pop=1M, R0={seir['r0']:.2f}):")
    print(f"    Peak Infected:    {seir['peak_infected']:,.0f}")
    print(f"    Peak Day:          {seir['peak_day']}")
    print(f"    Total Deceased:    {seir['total_deceased']:,.0f}")
    print(f"    Total Recovered:    {seir['total_recovered']:,.0f}")
    print(f"    Herd Immunity:     {seir['herd_immunity_threshold']*100:.1f}%")

    pandemic = epi.pandemic_scenarios(population=1e6, base_r0=2.5)
    print(f"\n  Pandemic Scenarios (R0={pandemic['base_r0']}):")
    for name, s in pandemic["scenarios"].items():
        print(f"    {name:20s}: Peak={s['peak_infected']:>10,.0f}, Deaths={s['total_deceased']:>10,.0f}")

    hr = epi.health_risk_score(age=65, bmi=30, smoking=True, diabetes=True, hypertension=True)
    print(f"\n  Health Risk (65yo, BMI=30, smoker, diabetic, hypertensive):")
    print(f"    Total Risk:        {hr['total_risk_score']*100:.1f}%")
    print(f"    Severity:          {hr['severity']}")
    print(f"    Age Factor:        {hr['age_factor']:.2f}")
    print(f"    Comorbidity:       {hr['comorbidity_factor']:.2f}x ({hr['comorbidity_count']} conditions)")
    for risk_name, risk_val in hr["specific_risks"].items():
        print(f"    {risk_name:15s}:   {risk_val*100:.1f}%")

    network = epi.disease_spread_network(n_nodes=200, n_edges=400, initial_infected=5, days=90, seed=42)
    print(f"\n  Network Spread (200 nodes, 400 edges):")
    print(f"    Peak Infected:    {network['peak_infected']}")
    print(f"    Peak Day:         {network['peak_day']}")
    print(f"    Final Infected%:  {network['final_infected_pct']:.1f}%")
    print(f"    Avg Degree:       {network['average_degree']:.1f}")


def demo_technology():
    separator("TECHNOLOGY FORECASTER (S-Curve + AI + Disruption + Power Shift)")
    tf = TechnologyForecaster()

    ai_curve = tf.s_curve_adoption(technology="AI", current_adoption=0.15, innovation_speed=0.35, years=20)
    print(f"  AI S-Curve Adoption:")
    print(f"    Current:          {ai_curve['current_adoption']*100:.1f}%")
    print(f"    5-Year Forecast:  {ai_curve['predicted_adoption_5y']*100:.1f}%")
    print(f"    Phase:            {ai_curve['phase']}")
    print(f"    Years to 50%:    {ai_curve['years_to_50pct']}")
    print(f"    Saturation:       {ai_curve['saturation']*100:.1f}%")

    ai_forecast = tf.ai_capability_forecast(current_capability=0.35, years=10)
    print(f"\n  AI Capability Forecast:")
    print(f"    Current:          {ai_forecast['current_capability']*100:.1f}%")
    print(f"    10-Year:          {ai_forecast['predicted_capabilities'][-1]*100:.1f}%")
    print(f"    Near-AGI (0.8):   {ai_forecast['years_to_near_agi']} years" if ai_forecast['years_to_near_agi'] is not None else "    Near-AGI: Unknown")
    print(f"    Milestones Reached:")
    for m in ai_forecast["milestones_reached"][:5]:
        print(f"      Year {m['year']}: {m['milestone']} ({m['capability']*100:.1f}%)")

    disruption = tf.disruption_analysis(industry="technology", timeline_years=10)
    print(f"\n  Disruption Analysis ({disruption['industry']}):")
    print(f"    Total Disruption Score: {disruption['total_disruption_score']*100:.1f}%")
    print(f"    Top Disruptors: {', '.join(disruption['top_disruptors'])}")
    for t in disruption["technologies"][:3]:
        print(f"      {t['name']:25s}: Score={t['disruption_score']*100:.1f}%, Impact={t['impact']*100:.0f}%, Displacement={t['job_displacement_millions']}M jobs")

    power = tf.power_shift_forecast(horizon_years=10)
    print(f"\n  Power Shift (10-year horizon):")
    print(f"    Projected Leaders: {', '.join(power['projected_leaders'])}")
    for r in power["regions"]:
        arrow = "↑" if r["direction"] == "rising" else "↓" if r["direction"] == "declining" else "→"
        print(f"      {r['name']:15s}: {r['current_composite']*100:.2f}% → {r['projected_composite']*100:.2f}% {arrow} (shift={r['power_shift']*100:+.2f}%)")

    comparison = tf.technology_adoption_comparison()
    print(f"\n  Technology Comparison:")
    for t in comparison["technologies"]:
        yrs = f"{t['years_to_50pct']}yr" if t['years_to_50pct'] != float('inf') else "Never"
        print(f"    {t['name']:25s}: Now={t['current']*100:.1f}%, 5yr={t['5yr_adoption']*100:.1f}%, Phase={t['phase']}, 50%={yrs}")


def demo_social_dynamics():
    separator("SOCIAL DYNAMICS (Unrest + Migration + Polarization + Power Shift)")
    sdm = SocialDynamicsModel()

    unrest = sdm.unrest_prediction(gini_coefficient=0.41, unemployment_rate=0.06, political_instability=0.3)
    print(f"  Unrest Prediction (Gini=0.41, Unemployment=6%):")
    print(f"    Unrest Probability: {unrest['unrest_probability']*100:.1f}%")
    print(f"    Level:              {unrest['unrest_level']}")
    print(f"    Social Cohesion:    {unrest['social_cohesion']*100:.1f}%")
    print(f"    Economic Distress:  {unrest['economic_distress']*100:.1f}%")
    print(f"    Political Risk:     {unrest['political_risk']*100:.1f}%")
    print(f"    Triggers:           {', '.join(unrest['likely_triggers'][:3])}")

    migration = sdm.migration_prediction(origin_gdp_per_capita=5000, destination_gdp_per_capita=50000, origin_conflict=0.3)
    print(f"\n  Migration Analysis (Origin=$5K → Destination=$50K):")
    print(f"    Migration Pressure: {migration['migration_pressure']*100:.1f}%")
    print(f"    Level:              {migration['migration_level']}")
    print(f"    Push Factors:       {migration['push_factors']*100:.1f}%")
    print(f"    Pull Factors:       {migration['pull_factors']*100:.1f}%")
    print(f"    5-Year Flow:        {migration['projected_flow_5yr']:,.0f}")

    polarization = sdm.polarization_index(income_inequality=0.41, political_polarization=0.6, media_fragmentation=0.7)
    print(f"\n  Polarization Index:")
    print(f"    Polarization:       {polarization['polarization_index']*100:.1f}%")
    print(f"    Fragmentation:      {polarization['fragmentation_index']*100:.1f}%")
    print(f"    State:              {polarization['state'].replace('_', ' ')}")
    print(f"    Risks:              {', '.join(polarization['primary_risks'])}")

    wealth = sdm.wealth_power_shift(horizon=2035)
    print(f"\n  Wealth/Power Shift (→2035):")
    print(f"    Biggest Gainer:    {wealth['biggest_gainer']['name']} (+{wealth['biggest_gainer']['shift']*100:.2f}%)")
    print(f"    Biggest Loser:     {wealth['biggest_loser']['name']} ({wealth['biggest_loser']['shift']*100:.2f}%)")
    for r in wealth["regions"]:
        arrow = "↑" if r["direction"] == "rising" else "↓" if r["direction"] == "declining" else "→"
        print(f"      {r['name']:15s}: {r['current_influence']*100:.2f}% → {r['projected_influence']*100:.2f}% {arrow}")

    civil = sdm.civil_unrest_scenario(base_unrest=0.3)
    print(f"\n  Civil Unrest Scenarios (Base={civil['base_unrest']*100:.1f}%):")
    for name, prob in civil["scenarios"].items():
        print(f"    {name:20s}: {prob*100:.1f}%")

    # ============================================================
    # Political Stability Demo
    # ============================================================
    print("\n" + "=" * 70)
    print("POLITICAL STABILITY ANALYSIS")
    print("=" * 70)

    from oracle.domains.political import PoliticalStabilityModel, PoliticalState

    pm = PoliticalStabilityModel()

    pol_stable = PoliticalState(
        government_stability=0.75, rule_of_law=0.80, corruption_index=0.15,
        press_freedom=0.85, opposition_strength=0.35, military_influence=0.10,
        economic_inequality=0.25, ethnic_tension=0.15, institutional_trust=0.70,
        external_pressure=0.15, electoral_integrity=0.85, civil_liberties=0.80,
    )
    pol_unstable = PoliticalState(
        government_stability=0.20, rule_of_law=0.15, corruption_index=0.85,
        press_freedom=0.10, opposition_strength=0.80, military_influence=0.75,
        economic_inequality=0.80, ethnic_tension=0.65, institutional_trust=0.10,
        external_pressure=0.50, electoral_integrity=0.10, civil_liberties=0.15,
    )

    stable = pm.assess_stability(pol_stable)
    unstable = pm.assess_stability(pol_unstable)

    print(f"\n  Stable Democracy:")
    print(f"    Stability Score:  {stable['overall_stability']*100:.1f}%")
    print(f"    Level:           {stable['stability_level']}")
    print(f"    Coup Risk:       {stable['coup_risk']*100:.1f}%")
    print(f"    Protest Prob:    {stable['protest_probability']*100:.1f}%")
    print(f"    Democratic Idx:  {stable['democratic_health']*100:.1f}%")
    print(f"    Regime:          {stable['regime_health']['regime_classification']}")

    print(f"\n  Unstable Regime:")
    print(f"    Stability Score:  {unstable['overall_stability']*100:.1f}%")
    print(f"    Level:           {unstable['stability_level']}")
    print(f"    Coup Risk:        {unstable['coup_risk']*100:.1f}%")
    print(f"    Protest Prob:    {unstable['protest_probability']*100:.1f}%")
    print(f"    Instability Factors: {len(unstable['instability_factors'])}")
    print(f"    Regime:          {unstable['regime_health']['regime_classification']}")

    for scenario in ["gradual_reform", "military_coup", "electoral_crisis"]:
        sim = pm.simulate_regime_change(scenario, state=pol_unstable, months=24)
        print(f"\n  Regime Change: {scenario.replace('_', ' ').title()}")
        print(f"    Probability:      {sim['probability']*100:.1f}%")
        print(f"    Economic Impact:  {sim['economic_impact_pct']*100:.1f}%")
        print(f"    Democratic Setback: {sim['democratic_setback']*100:.1f}%")

    # ============================================================
    # Biodiversity Demo
    # ============================================================
    print("\n" + "=" * 70)
    print("BIODIVERSITY ANALYSIS")
    print("=" * 70)

    from oracle.domains.biodiversity import BiodiversityModel, BiodiversityState

    bm = BiodiversityModel()

    bio_healthy = BiodiversityState(
        habitat_loss_rate=0.10, species_extinction_rate=0.10, ecosystem_services_index=0.85,
        protected_area_coverage=0.30, deforestation_rate=0.05, ocean_acidification=0.10,
        invasive_species_impact=0.10, pollution_index=0.10, overexploitation=0.10,
        climate_impact_on_biodiversity=0.10, genetic_diversity_index=0.85,
        coral_reef_health=0.80, pollinator_decline=0.10,
    )
    bio_degraded = BiodiversityState(
        habitat_loss_rate=0.75, species_extinction_rate=0.80, ecosystem_services_index=0.20,
        deforestation_rate=0.70, ocean_acidification=0.65, pollution_index=0.70,
        climate_impact_on_biodiversity=0.75, coral_reef_health=0.10, pollinator_decline=0.75,
    )

    healthy = bm.assess_health(bio_healthy)
    degraded = bm.assess_health(bio_degraded)

    print(f"\n  Healthy Ecosystem:")
    print(f"    Biodiversity Index: {healthy['biodiversity_index']*100:.1f}%")
    print(f"    Health Level:       {healthy['health_level']}")
    print(f"    Resilience:         {healthy['ecosystem_resilience']['overall_resilience']*100:.1f}%")
    print(f"    Species Outlook:     {healthy['species_outlook']}")
    print(f"    Tipping Points:     {healthy['n_tipping_points']}")

    print(f"\n  Degraded Ecosystem:")
    print(f"    Biodiversity Index: {degraded['biodiversity_index']*100:.1f}%")
    print(f"    Health Level:       {degraded['health_level']}")
    print(f"    Resilience:         {degraded['ecosystem_resilience']['overall_resilience']*100:.1f}%")
    print(f"    Species Outlook:     {degraded['species_outlook']}")
    print(f"    Tipping Points:     {degraded['n_tipping_points']}")

    for scenario in ["business_as_usual", "aggressive_protection", "mass_extinction"]:
        ext = bm.simulate_extinction_scenario(scenario, state=bio_degraded, years=50)
        print(f"\n  Extinction Scenario: {scenario.replace('_', ' ').title()}")
        print(f"    Final Biodiversity: {ext['final_biodiversity_index']*100:.1f}%")
        print(f"    Species Lost:       {ext['species_at_risk_pct']:.1f}%")
        print(f"    Outcome:            {ext['outcome']}")

    # ============================================================
    # Pandemic Preparedness Demo
    # ============================================================
    print("\n" + "=" * 70)
    print("PANDEMIC PREPAREDNESS ANALYSIS")
    print("=" * 70)

    from oracle.domains.pandemic import PandemicPreparednessModel, PandemicState

    print("\n" + "="*60)
    print("  SPACE & ORBITAL ENVIRONMENT")
    print("="*60)
    from oracle.domains.space import SpaceModel, SpaceState
    space = SpaceModel()
    current_orbit = SpaceState(orbital_debris_count=30000, active_satellites=10000, kessler_syndrome_risk=0.25)
    orbital = space.assess_orbital_environment(current_orbit)
    print(f"  Collision Probability: {orbital['collision_probability']*100:.2f}%")
    print(f"  Kessler Status:        {orbital['kessler_status']}")
    print(f"  LEO Congestion:        {orbital['leo_congestion']*100:.1f}%")
    print(f"  Orbital Environment:   {orbital['orbital_environment']}")
    print(f"  Sustainable Orbit:     {orbital['sustainable_orbit']}")
    kessler_sim = space.simulate_kessler_scenario(scenario="mega_constellation", state=current_orbit, years=20)
    print(f"\n  Kessler Scenario (mega_constellation, 20yr):")
    print(f"    Kessler Triggered:   {kessler_sim['kessler_triggered']}")
    print(f"    Final Debris:        {kessler_sim['final_debris_count']:.0f}")
    sp_pred = space.predict(years_ahead=10, state=current_orbit)
    print(f"\n  10-Year Space Forecast:")
    print(f"    Future Kessler Risk:  {sp_pred['future_kessler_risk']*100:.1f}%")
    print(f"    Future Satellites:   {sp_pred['future_active_satellites']:.0f}")
    print(f"    Space Revenue:       ${sp_pred['future_commercial_revenue_billion']:.0f}B")
    print(f"    Sustainability:     {sp_pred['orbital_sustainability_trend']}")

    print("\n" + "="*60)
    print("  MENTAL HEALTH")
    print("="*60)
    from oracle.domains.mental_health import MentalHealthModel, MentalHealthState
    mh = MentalHealthModel()
    mh_state = MentalHealthState(depression_prevalence=0.08, anxiety_prevalence=0.06, social_isolation_index=0.5)
    mh_assess = mh.assess_population_health(mh_state)
    print(f"  Overall Burden:        {mh_assess['overall_burden']*100:.1f}% ({mh_assess['burden_level']})")
    print(f"  System Capacity:       {mh_assess['system_capacity']*100:.1f}% ({mh_assess['capacity_level']})")
    print(f"  Treatment Access:      {mh_assess['treatment_access_rate']*100:.1f}%")
    print(f"  Youth Risk:            {mh_assess['youth_mental_health_risk']*100:.1f}%")
    print(f"  Pop Affected:          {mh_assess['population_affected_millions']:.0f}M")
    crisis = mh.simulate_crisis_scenario(scenario="economic_recession", state=mh_state, years=5)
    print(f"\n  Crisis Scenario (economic_recession, 5yr):")
    print(f"    Final Depression:    {crisis['final_depression_prevalence']*100:.1f}%")
    print(f"    Final Suicide Rate:  {crisis['final_suicide_rate']:.1f}/100k")
    mh_pred = mh.predict(years_ahead=10, state=mh_state)
    print(f"\n  10-Year Mental Health Forecast:")
    print(f"    Future Depression:    {mh_pred['future_depression_prevalence']*100:.1f}%")
    print(f"    Trajectory:          {mh_pred['trajectory']}")
    print(f"    Affected:            {mh_pred['affected_population_millions']:.0f}M")

    print("\n" + "="*60)
    print("  HOUSING & REAL ESTATE")
    print("="*60)
    from oracle.domains.housing import HousingModel, HousingState
    housing = HousingModel()
    hs = HousingState(median_home_price=450000, median_income=75000, mortgage_rate=0.07)
    ha = housing.assess_market(hs)
    print(f"  Market Health:         {ha['market_health_score']*100:.1f}% ({ha['health_label']})")
    print(f"  Price/Income:          {ha['price_to_income_ratio']:.1f}x")
    print(f"  Affordability:         {ha['affordability_index']*100:.1f}%")
    print(f"  Bubble Risk:           {ha['bubble_risk']*100:.1f}%")
    print(f"  Rental Burden:         {ha['rental_burden']}")
    crash = housing.simulate_housing_scenario(scenario="crash", state=hs, years=5)
    print(f"\n  Crash Scenario (5yr):")
    print(f"    Final Price:         ${crash['final_median_price']:,.0f}")
    print(f"    Final P/I:           {crash['final_price_to_income']:.1f}x")
    h_pred = housing.predict(years_ahead=5, state=hs)
    print(f"\n  5-Year Housing Forecast:")
    print(f"    Future Price:        ${h_pred['future_median_price']:,.0f}")
    print(f"    Future P/I:          {h_pred['future_price_to_income']:.1f}x")
    print(f"    Outlook:             {h_pred['market_outlook'].replace('_',' ')}")

    # ─── NATURAL FORCE DOMAINS ───────────────────────────────

    from oracle.domains.seismic import SeismicModel, SeismicState
    seismic = SeismicModel()
    stressed = SeismicState(fault_stress_accumulation=0.8, volcanic_activity_index=0.6, swarm_activity=0.5, subduction_zone_stress=0.7, radon_anomaly=0.5, electromagnetic_anomaly=0.4)
    haz = seismic.assess_seismic_hazard(stressed)
    print(f"\n  SEISMIC HAZARD (High Stress):")
    print(f"    Composite Hazard:    {haz['composite_hazard']*100:.1f}%")
    print(f"    Hazard Level:        {haz['hazard_level']}")
    print(f"    M7 Prob 30d:         {haz['m7_probability_30d']*100:.2f}%")
    print(f"    M8 Prob 1yr:         {haz['m8_probability_365d']*100:.3f}%")
    print(f"    Volcanic Risk:       {haz['volcanic_risk']}")
    eq = seismic.simulate_earthquake_scenario(magnitude=8.0, depth_km=10, population_exposed_m=10)
    print(f"\n  M8.0 Shallow Quake (10M pop):")
    print(f"    Intensity MMI:       {eq['estimated_intensity_mmi']:.1f}")
    print(f"    Damage Ratio:        {eq['damage_ratio']*100:.1f}%")
    print(f"    Fatalities:          {eq['estimated_fatalities']:,.0f}")
    print(f"    Economic Loss:       ${eq['economic_loss_billion_usd']:.1f}B")
    print(f"    Tsunami Prob:        {eq['tsunami_probability']*100:.1f}%")
    sp = seismic.predict(years_ahead=5, state=stressed)
    print(f"\n  5-Year Seismic Forecast:")
    print(f"    M6 Expected:         {sp['expected_m6_count']:.1f}")
    print(f"    M7 Expected:         {sp['expected_m7_count']:.2f}")
    print(f"    M8 Prob 5yr:         {sp['m8_probability_5yr']*100:.3f}%")
    print(f"    Regime:              {sp['seismic_regime']}")

    from oracle.domains.solar import SolarModel, SolarState
    solar = SolarModel()
    active = SolarState(sunspot_number=180, kp_index=7, dst_index=-150, flare_probability=0.6, cme_probability=0.5, proton_flux=0.5)
    sw = solar.assess_solar_weather(active)
    print(f"\n  SOLAR WEATHER (Active Sun):")
    print(f"    Composite Risk:      {sw['composite_risk']*100:.1f}%")
    print(f"    Risk Level:          {sw['risk_level']}")
    print(f"    X-Flare Prob:        {sw['x_flare_probability']*100:.1f}%")
    print(f"    Grid Risk:           {sw['grid_infrastructure_risk']*100:.1f}%")
    print(f"    Satellite Risk:      {sw['satellite_risk']*100:.1f}%")
    print(f"    Aurora Latitude:     {sw['aurora_latitude']:.0f}°")
    car = solar.simulate_carrington_event(grid_hardening=0.3, satellite_shielding=0.2, backup_power_days=3)
    print(f"\n  Carrington Event Sim:")
    print(f"    Grid Damage:         {car['grid_damage_pct']*100:.1f}%")
    print(f"    Transformers At Risk: {car['transformers_at_risk']}")
    print(f"    Restoration Days:    {car['full_restoration_days']}")
    print(f"    Economic Impact:     ${car['economic_impact_trillion_usd']:.1f}T")
    print(f"    Pop Affected:        {car['population_affected_billions']:.1f}B")
    slp = solar.predict(years_ahead=10, state=active)
    print(f"\n  10-Year Solar Forecast:")
    print(f"    Activity Regime:     {slp['activity_regime'].replace('_',' ')}")
    print(f"    Carrington Prob:     {slp['carrington_probability']*100:.2f}%")
    print(f"    Years to Max:        {slp['years_to_next_maximum']:.1f}")

    from oracle.domains.oceanic import OceanicModel, OceanicState
    oceanic = OceanicModel()
    nino = OceanicState(enso_index=1.5, sea_surface_temp_anomaly_c=1.2, arctic_sea_ice_extent_mkm2=3.5, coral_bleaching_alert=0.6, thermohaline_circulation_strength=0.5)
    oa = oceanic.assess_ocean_state(nino)
    print(f"\n  OCEANIC STATE (Strong El Niño):")
    print(f"    Ocean Risk:          {oa['composite_ocean_risk']*100:.1f}%")
    print(f"    Risk Level:          {oa['risk_level']}")
    print(f"    ENSO Phase:          {oa['enso_phase']}")
    print(f"    ENSO Strength:       {oa['enso_strength']:.2f}")
    print(f"    Hurricane Season:   {oa['expected_hurricane_season']:.1f}")
    print(f"    AMOC Risk:           {oa['thermohaline_circulation_risk']*100:.1f}%")
    print(f"    Coral Bleaching:     {oa['coral_bleaching_risk']*100:.1f}%")
    enso_t = oceanic.simulate_enso_transition(current_phase="elnino", months=24)
    print(f"\n  ENSO Transition (24mo from El Niño):")
    print(f"    Final Phase:         {enso_t['final_phase']}")
    op = oceanic.predict(years_ahead=5, state=nino)
    print(f"\n  5-Year Ocean Forecast:")
    print(f"    Future SST:          {op['future_sst_anomaly_c']:.2f}°C")
    print(f"    Future Arctic Ice:   {op['future_arctic_ice_mkm2']:.1f}M km²")
    print(f"    AMOC Collapse Risk:  {op['amoc_collapse_risk']*100:.2f}%")
    print(f"    Sea Level Rise:      {op['cumulative_sea_level_rise_mm']:.1f}mm")
    print(f"    Ice-Free Summer:     {op['ice_free_summer_probability']*100:.2f}%")

    from oracle.domains.neo import NearEarthObjectModel, NearEarthObjectState
    neo_model = NearEarthObjectModel()
    neo_st = NearEarthObjectState(detection_rate_pct=0.6, deflection_readiness=0.05, torino_scale_max=0, kinetic_energy_typical_mt=10)
    ir = neo_model.assess_impact_risk(neo_st)
    print(f"\n  NEAR-EARTH OBJECT RISK:")
    print(f"    Composite Risk:      {ir['composite_risk']*100:.1f}%")
    print(f"    Risk Level:          {ir['risk_level']}")
    print(f"    Detection Gap:       {ir['detection_gap_pct']*100:.1f}%")
    print(f"    City Killer 50yr:    {ir['city_killer_probability_50yr']*100:.2f}%")
    print(f"    Global Catast 100yr: {ir['global_catastrophe_probability_100yr']*100:.4f}%")
    print(f"    Undetected Hazardous: {ir['undetected_hazardous_estimate']:.0f}")
    imp = neo_model.simulate_impact_scenario(diameter_m=200, velocity_kms=20, density_kgm3=3000, land_impact=True)
    print(f"\n  200m Urban Impact:")
    print(f"    Event Class:         {imp['event_class'].replace('_',' ')}")
    print(f"    Kinetic Energy:      {imp['kinetic_energy_mt_tnt']:.1f} MT TNT")
    print(f"    Crater Diameter:     {imp['crater_diameter_km']:.2f}km")
    print(f"    Nuclear Equiv:       {imp['nuclear_bomb_equivalent']:.0f} bombs")
    np_red = neo_model.predict(years_ahead=50, state=neo_st)
    print(f"\n  50-Year NEO Forecast:")
    print(f"    Future Detection:    {np_red['future_detection_rate']*100:.1f}%")
    print(f"    Future Deflection:  {np_red['future_deflection_readiness']*100:.1f}%")
    print(f"    Impact Prob Period:  {np_red['impact_probability_period']*100:.3f}%")

    from oracle.domains.atmospheric import AtmosphericModel, AtmosphericState
    atm = AtmosphericModel()
    atm_stressed = AtmosphericState(
        polar_vortex_strength=0.3, stratospheric_warming_risk=0.5,
        ozone_column_du=220, jet_stream_latitude_offset=-5.0,
        rossby_wave_amplitude=0.7, hadley_cell_extent_deg=33.0,
        atmospheric_co2_ppm=425, methane_ppb=1950,
    )
    aa = atm.assess_atmospheric_state(atm_stressed)
    print(f"\n  ATMOSPHERIC STATE (Weakened Vortex + Low Ozone):")
    print(f"    Composite Risk:      {aa['composite_atmospheric_risk']*100:.1f}%")
    print(f"    Risk Level:          {aa['risk_level']}")
    print(f"    Vortex Status:       {aa['polar_vortex_status']}")
    print(f"    Jet Stream:          {aa['jet_stream_status']}")
    print(f"    SSW Risk:            {aa['ssw_risk_score']*100:.1f}%")
    print(f"    Ozone Deficit:       {aa['ozone_deficit_pct']:.1f}%")
    print(f"    Hadley Expansion:    {aa['hadley_cell_expansion']*100:.1f}%")
    ssw = atm.simulate_ssw_scenario(vortex_strength=0.2, ssw_intensity=0.8, duration_days=30)
    print(f"\n  SSW Scenario (Strong Warming):")
    print(f"    Temp Anomaly:        {ssw['temperature_anomaly_northern_c']:.1f}°C")
    print(f"    Cold Outbreak Prob:  {ssw['cold_outbreak_probability']*100:.1f}%")
    print(f"    Storm Amplification: {ssw['winter_storm_amplification']:.1f}x")
    print(f"    Energy Demand +:     {ssw['energy_demand_increase_pct']:.0f}%")
    print(f"    Affected Pop:        {ssw['affected_population_millions']:.0f}M")
    oz = atm.simulate_ozone_scenario(ozone_column_du=150, duration_months=6)
    print(f"\n  Ozone Depletion (150 DU):")
    print(f"    UV Increase:         +{oz['uv_increase_pct']:.1f}%")
    print(f"    Cancer Risk +:       {oz['skin_cancer_risk_increase_pct']:.1f}%")
    print(f"    Crop Damage:         {oz['crop_damage_pct']:.1f}%")
    print(f"    Recovery:            {oz['recovery_years']}yr")
    ap = atm.predict(years_ahead=10, state=atm_stressed)
    print(f"\n  10-Year Atmospheric Forecast:")
    print(f"    Future CO2:          {ap['future_co2_ppm']:.0f}ppm")
    print(f"    Future Ozone:        {ap['future_ozone_du']:.0f}DU")
    print(f"    Vortex Trend:        {ap['vortex_trend']}")
    print(f"    Geomagnetic Trend:   {ap['geomagnetic_trend']}")
    print(f"    SSW Freq Mult:       {ap['ssw_frequency_multiplier']:.2f}x")

    from oracle.domains.ecological import EcologicalModel, EcologicalState
    eco = EcologicalModel()
    eco_stressed = EcologicalState(
        biodiversity_index=0.25, trophic_integrity=0.2, keystone_species_ratio=0.1,
        ecosystem_connectivity=0.15, habitat_fragmentation=0.8, invasive_species_impact=0.6,
        pollution_load=0.7, climate_stress=0.8, overexploitation_rate=0.6,
        adaptive_capacity=0.15, genetic_diversity_index=0.2, extinction_debt=0.7,
        evolutionary_pressure=0.6,
    )
    ea = eco.assess_ecosystem(eco_stressed)
    print(f"\n  ECOLOGICAL STATE (Stressed Ecosystem):")
    print(f"    Health Index:         {ea['ecosystem_health_index']*100:.1f}% ({ea['health_level']})")
    print(f"    Trophic Integrity:    {ea['trophic_integrity']*100:.1f}%")
    print(f"    Resilience:          {ea['resilience_score']*100:.1f}%")
    print(f"    Stress Score:        {ea['stress_score']*100:.1f}%")
    print(f"    Cascade Potential:    {ea['cascade_potential']*100:.1f}%")
    print(f"    Keystone Risk:       {ea['keystone_risk']*100:.1f}%")
    print(f"    Threshold Extinctions: {len(ea['threshold_extinctions'])}")
    for te in ea['threshold_extinctions']:
        print(f"      {te['ecosystem']}: prob={te['probability']*100:.1f}%, impact={te['impact']}")
    cascade = eco.simulate_trophic_cascade(keystone_removed="apex_predator", severity=0.8, years=15)
    print(f"\n  Apex Predator Cascade (severity=0.8, 15yr):")
    print(f"    Biodiversity Loss:    {cascade['projected_biodiversity_loss_pct']:.1f}%")
    print(f"    Recovery Years:       {cascade['recovery_years']}")
    print(f"    Recovery Possible:    {cascade['recovery_possible']}")
    for t in cascade['timeline'][::3]:
        print(f"      Year {t['year']}: BDI={t['biodiversity_index']*100:.1f}%, Lost={t['species_lost_pct']:.1f}%, Stability={t['ecosystem_stability']*100:.1f}%")
    ep = eco.predict(years_ahead=20, state=eco_stressed)
    print(f"\n  20-Year Ecological Forecast:")
    print(f"    Future BDI:          {ep['future_biodiversity_index']*100:.1f}%")
    print(f"    Future Genetic Div:  {ep['future_genetic_diversity']*100:.1f}%")
    print(f"    Cumulative Extinct:   {ep['cumulative_extinctions_pct']:.1f}%")
    print(f"    Ecosystem Regime:    {ep['ecosystem_regime']}")
    print(f"    Tipping Point Risk:  {ep['tipping_point_risk']*100:.1f}%")

    from oracle.domains.paleoclimatology import PaleoclimatologyModel, PaleoclimateState
    paleo = PaleoclimatologyModel()
    current_orbital = PaleoclimateState(co2_ppmv=420, global_temp_anomaly_c=1.2, ice_volume_fraction=0.03, thermohaline_circulation_strength=0.7)
    ml = paleo.assess_milankovitch(current_orbital)
    print(f"\n  PALEOCLIMATE / MILANKOVITCH:")
    print(f"    Orbital Phase:       {ml['orbital_phase'].replace('_',' ')}")
    print(f"    Insolation Var:      {ml['insolation_variability']*100:.1f}%")
    print(f"    Glaciation Risk:     {ml['glaciation_risk']*100:.1f}%")
    print(f"    Deglaciation Risk:   {ml['deglaciation_risk']*100:.1f}%")
    print(f"    Radiative Forcing:   {ml['radiative_forcing_wm2']:.2f} W/m2")
    print(f"    Next Glaciation:     {ml['next_glaciation_estimate_kyr']:.0f} kyr")
    petm = paleo.simulate_analog(analog="petm", years=10000)
    print(f"\n  PETM Analog:")
    print(f"    Similarity:          {petm['similarity_score']*100:.1f}%")
    print(f"    Relevance:           {petm['current_relevance']}")
    for l in petm['key_lessons'][:3]:
        print(f"    Lesson: {l}")
    pp = paleo.predict(years_ahead=5000, state=current_orbital)
    print(f"\n  5000yr Paleoclimate Forecast:")
    print(f"    Climate Regime:     {pp['climate_regime'].replace('_',' ')}")
    print(f"    Future CO2:         {pp['future_co2_ppmv']:.0f} ppmv")
    print(f"    Future Temp:        {pp['future_temp_anomaly_c']:.2f}°C")
    print(f"    Tipping Risk:       {pp['tipping_point_risk']*100:.1f}%")

    from oracle.domains.cosmic_radiation import CosmicRadiationModel, CosmicRadiationState
    cosmic = CosmicRadiationModel()
    cr_state = CosmicRadiationState(gcr_flux_relative=1.0, solar_modulation_potential_gv=0.5, svensmark_cloud_nucleation_factor=0.3)
    ca = cosmic.assess_radiation(cr_state)
    print(f"\n  COSMIC RADIATION:")
    print(f"    Composite Risk:      {ca['composite_risk']*100:.1f}% ({ca['risk_level']})")
    print(f"    GCR Flux:            {ca['effective_gcr_flux']:.3f}")
    print(f"    Svensmark Effect:    {ca['svensmark_cloud_effect']*100:.2f}%")
    print(f"    Aviation Risk:       {ca['aviation_risk']*100:.1f}%")
    print(f"    SEU Risk:            {ca['electronics_seu_risk']*100:.1f}%")
    sm = cosmic.simulate_solar_minimum(duration_years=3, severity=0.8)
    print(f"\n  Grand Solar Minimum (3yr, severity=0.8):")
    print(f"    Peak GCR Flux:      {sm['peak_gcr_flux']:.3f}")
    print(f"    Dose Increase:      {sm['average_dose_increase_pct']:.1f}%")
    print(f"    Cooling Potential:   {sm['cooling_potential_c']:.2f}°C")
    cp = cosmic.predict(years_ahead=10, state=cr_state)
    print(f"\n  10yr Cosmic Ray Forecast:")
    print(f"    Future GCR:         {cp['future_gcr_flux']:.3f}")
    print(f"    Regime:             {cp['radiation_regime'].replace('_',' ')}")

    from oracle.domains.gravitational import GravitationalModel, GravitationalState
    grav = GravitationalModel()
    gs = GravitationalState(lunar_tidal_force_relative=1.0, planetary_alignment_index=0.2, earth_moon_distance_km=384400)
    gt = grav.assess_tidal_forces(gs)
    print(f"\n  GRAVITATIONAL / TIDAL FORCES:")
    print(f"    Composite Index:     {gt['composite_tidal_index']*100:.1f}% ({gt['level']})")
    print(f"    Effective Force:     {gt['effective_tidal_force']:.3f}")
    print(f"    Perigee Ampl.:      {gt['perigee_amplification']:.3f}")
    print(f"    Seismic Triggering:  {gt['seismic_triggering_probability']*100:.1f}%")
    print(f"    Coastal Flood:       {gt['coastal_flood_risk']*100:.1f}%")
    al = grav.simulate_alignment(n_planets=5, alignment_strength=0.7, days=30)
    print(f"\n  Planetary Alignment (5 planets):")
    print(f"    Peak Seismic Mult:  {al['peak_seismic_probability_multiplier']:.3f}x")
    print(f"    Coastal Flood +:    {al['coastal_flood_risk_increase_pct']:.1f}%")
    ps = grav.simulate_perigee_spring(perigee_distance_km=356500, duration_days=7)
    print(f"\n  Perigee Spring Tide (356,500 km):")
    print(f"    King Tide Ampl:     {ps['king_tide_amplification']:.3f}x")
    print(f"    Coastal Flood Mult: {ps['coastal_flood_multiplier']:.3f}x")
    print(f"    Crustal Stress:    {ps['seismic_stress_mpa']:.3f} MPa")

    well_prepared = PandemicState(
        healthcare_capacity=0.85, surveillance_systems=0.80, vaccine_readiness=0.75,
        supply_chain_resilience=0.80, international_cooperation=0.75, public_trust=0.70,
        pandemic_preparedness_plan=0.80, laboratory_capacity=0.75, risk_communication=0.75,
        essential_worker_protection=0.70, genomic_surveillance=0.70, ppe_stockpile=0.65,
        antiviral_stockpile=0.50, contact_tracing_capacity=0.65, quarantine_capacity=0.70,
    )
    poorly_prepared = PandemicState(
        healthcare_capacity=0.15, surveillance_systems=0.10, vaccine_readiness=0.10,
        supply_chain_resilience=0.15, international_cooperation=0.10, public_trust=0.10,
        pandemic_preparedness_plan=0.10, laboratory_capacity=0.10, risk_communication=0.10,
        essential_worker_protection=0.10, genomic_surveillance=0.05, ppe_stockpile=0.10,
        antiviral_stockpile=0.05, contact_tracing_capacity=0.10, quarantine_capacity=0.10,
    )

    well = pan.assess_preparedness(well_prepared)
    poor = pan.assess_preparedness(poorly_prepared)

    print(f"\n  Well-Prepared System:")
    print(f"    Overall Preparedness:  {well['overall_preparedness']*100:.1f}%")
    print(f"    Readiness Level:       {well['readiness_level']}")
    print(f"    Detection Capacity:    {well['detection_capacity']*100:.1f}%")
    print(f"    Treatment Capacity:    {well['treatment_capacity']*100:.1f}%")
    print(f"    Prevention Capacity:   {well['prevention_capacity']*100:.1f}%")
    print(f"    Gaps Identified:       {well['n_gaps']}")

    print(f"\n  Poorly-Prepared System:")
    print(f"    Overall Preparedness:  {poor['overall_preparedness']*100:.1f}%")
    print(f"    Readiness Level:       {poor['readiness_level']}")
    print(f"    Gaps Identified:       {poor['n_gaps']}")
    print(f"    Top Gaps:")
    for g in poor['gaps'][:3]:
        print(f"      {g['area']}: {g['severity']} ({g['gap_pct']}% gap)")

    for pathogen in ["respiratory_virus", "coronavirus", "antibiotic_resistant_bacteria"]:
        outbreak = pan.simulate_outbreak(pathogen, state=poorly_prepared, r0=2.5, days=365)
        print(f"\n  Outbreak Simulation: {pathogen.replace('_', ' ').title()}")
        print(f"    Effective R0:          {outbreak['effective_r0']:.2f}")
        print(f"    Detection Delay:       {outbreak['detection_delay_days']} days")
        print(f"    IFR:                   {outbreak['infection_fatality_rate']*100:.2f}%")
        print(f"    Pop Infected:          {outbreak['total_populations_infected_pct']:.1f}%")
        print(f"    Healthcare Overwhelmed: {outbreak['healthcare_overwhelm']}")

    pred = pan.predict(years_ahead=10, state=poorly_prepared)
    print(f"\n  10-Year Pandemic Forecast (Poorly Prepared):")
    print(f"    Future Preparedness:   {pred['future_preparedness']*100:.1f}%")
    print(f"    Pandemic Probability:  {pred['pandemic_probability']*100:.1f}%")
    print(f"    Resilience Score:      {pred['resilience_score']*100:.1f}%")
    print(f"    Top Emergence Risks:")
    for path, risk in sorted(pred['emergence_risk'].items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"      {path.replace('_', ' ').title()}: {risk*100:.1f}%")


def demo_prescience():
    separator("PRESCIENCE (Event Forecast + Tipping Points + Early Warnings)")
    from oracle.domains.prescience import PrescienceEngine
    pe = PrescienceEngine()

    domain_scores = {"seismic": 0.5, "infrastructure": 0.5, "economic": 0.5, "social": 0.5, "political": 0.5}

    result = pe.predict_prescience(domain_scores, current_temp_anomaly=1.2)
    print(f"  Prescience Score:    {result['prescience_score']*100:.1f}%")
    print(f"  Outlook:             {result['outlook']}")
    print(f"  Catastrophic Prob:   {result['any_catastrophic_probability']*100:.1f}%")
    print(f"  Events Forecast:     {result['n_events_forecast']}")
    print(f"  Tipping Near/Active: {result['n_tipping_points_near']}/{result['n_tipping_points_active']}")
    print(f"  Alerts:              {result['alert_level']}")
    print(f"  Key Concerns:")
    for c in result.get('key_concerns', []):
        print(f"    {c}")

    events = pe.forecast_events(domain_scores, years=10, n_simulations=100)
    print(f"\n  Top 3 Foreseen Events:")
    for e in events['events'][:3]:
        print(f"    {e['event_name']}: {e['annual_probability']*100:.1f}%/yr, severity {e['severity']:.2f}, recovery {e['recovery_years']}yr")

    tipping = pe.assess_tipping_points(domain_scores, current_temp_anomaly=1.2)
    print(f"\n  Tipping Points (near or closer: {tipping['near_or_closer']}):")
    for tp in tipping['tipping_points'][:3]:
        print(f"    {tp['name']}: proximity {tp['proximity']*100:.1f}% ({tp['status']}), irreversibility {tp['irreversibility']*100:.0f}%")

    scenario = pe.generate_scenario("great_filter", domain_scores=domain_scores, years=50, n_simulations=200)
    print(f"\n  Great Filter Scenario:")
    print(f"    Severity: {scenario['adjusted_severity']:.3f}")
    print(f"    Cascade Prob: {scenario['cascade_probability']*100:.1f}%")
    print(f"    Mean Final Stability: {scenario['mean_final_stability']:.3f}")
    print(f"    Collapse Prob: {scenario['collapse_probability']*100:.1f}%")


def demo_fate():
    separator("FATE ENGINE (Civilizational Destiny + Personal Fate)")
    from oracle.domains.fate import FateEngine, FateState, PersonalFateState
    fe = FateEngine(seed=42)

    print("  === Civilizational Fate ===")
    civ = fe.assess_civilizational_trajectory()
    print(f"  Trajectory Score:   {civ['trajectory_score']*100:.1f}% ({civ['trajectory_label']})")
    print(f"  Kardashev Scale:    {civ['kardashev_scale_current']:.3f}")
    print(f"  Energy to Type I:   {civ['energy_progress_to_type_i_pct']:.1f}%")
    print(f"  Existential Risk:   {civ['existential_risk_composite']*100:.1f}%")
    print(f"  Fate Commitment:    {civ['fate_commitment']*100:.1f}%")

    thermo = fe.assess_thermodynamic_fate()
    print(f"\n  Thermodynamic Pressure: {thermo['thermodynamic_pressure']*100:.1f}% ({thermo['pressure_label']})")
    print(f"  EROI: {thermo['eroi_current']:.1f} (threshold: {thermo['eroi_threshold']})")
    print(f"  Carrying Capacity: {thermo['carrying_capacity_usage']*100:.1f}% {'OVERSHOOT' if thermo['overshoot'] else ''}")
    print(f"  Active Constraints: {thermo['n_active_constraints']}/{thermo['n_hard_constraints']} hard")

    commit = fe.assess_deterministic_commitments()
    print(f"\n  Deterministic Commitments:")
    print(f"  Total: {commit['n_commitments']} | Avg Irreversibility: {commit['avg_irreversibility']*100:.1f}% | High: {commit['n_high_irreversibility']}")
    for c in commit['commitments'][:4]:
        print(f"    {c['name']}: irreversibility {c['irreversibility']*100:.0f}%, probability {c.get('probability',0)*100:.0f}%")

    depletion = fe.assess_resource_depletion(years_ahead=50)
    print(f"\n  Resource Depletion: {depletion['overall_depletion_pressure']*100:.1f}% pressure")
    print(f"  Critical: {depletion['n_critical_resources']} | Substitute Gap: {depletion['n_substitute_gap']}")
    for d in depletion['depletions'][:4]:
        print(f"    {d['resource']}: {d['remaining_pct']:.1f}% remaining ({d['status']}), substitute: {d['substitute_readiness']*100:.0f}%")

    print(f"\n  === Personal Fate ===")
    personal = fe.predict_personal_fate()
    print(f"  Fate Type:          {personal['fate_type'].replace('_',' ').title()}")
    print(f"  Total Determinism:  {personal['total_determinism']*100:.1f}%")
    print(f"  Agency Score:       {personal['agency_score']*100:.1f}%")
    print(f"  Immutable/Modifiable: {personal['immutable_pct']}%/{personal['modifiable_pct']}%")

    pd = personal.get('personal_destiny', {})
    print(f"\n  Determinism Breakdown:")
    print(f"    Genetic:      {pd.get('genetic_determinism',0)*100:.1f}%")
    print(f"    Structural:   {pd.get('structural_determinism',0)*100:.1f}%")
    print(f"    Behavioral:   {pd.get('behavioral_determinism',0)*100:.1f}%")
    print(f"    Epigenetic:   {pd.get('epigenetic_determinism',0)*100:.1f}%")

    m = personal.get('mortality', {})
    print(f"\n  Mortality:")
    print(f"    Life Expectancy:  {m.get('life_expectancy_adjusted','?')} yr")
    print(f"    Biological Age:   {m.get('biological_age','?')}")
    print(f"    Healthspan:       {m.get('healthspan_estimate','?')} yr")
    print(f"    Modifiable Gain:   +{m.get('modifiable_years_gained',0)} yr")
    top_cause = sorted(m.get('cause_probabilities',{}).items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"    Top Causes: {', '.join(f'{k.replace('_',' ')} ({v*100:.1f}%)' for k,v in top_cause)}")

    h = personal.get('habit_destiny', {})
    print(f"\n  Habit Destiny:")
    print(f"    Inertia:           {h.get('habit_inertia',0)*100:.1f}% ({h.get('inertia_label','')})")
    print(f"    Change Probability: {h.get('change_probability',0)*100:.1f}%")
    print(f"    Annual Reversion:  {h.get('annual_habit_reversion',0)*100:.1f}%")

    s = personal.get('socioeconomic_path', {})
    print(f"\n  Socioeconomic Path:")
    print(f"    Origin Trap:       {s.get('origin_trap_index',0)*100:.1f}% ({s.get('path_label','')})")
    print(f"    Mobility Potential: {s.get('mobility_potential',0)*100:.1f}%")
    print(f"    Class Escape Prob: {s.get('probability_of_class_escape',0)*100:.1f}%")

    if personal.get('highest_leverage_actions'):
        print(f"\n  Highest Leverage Actions:")
        for a in personal['highest_leverage_actions'][:5]:
            print(f"    {a}")

    full_fate = fe.predict_fate()
    print(f"\n  Full Civilizational Fate:")
    print(f"    Fate Score: {full_fate['fate_score']*100:.1f}% ({full_fate['fate_label'].replace('_',' ')})")
    print(f"    Escape Velocities:")
    for ev in full_fate.get('escape_velocities', [])[:4]:
        print(f"      {ev['name']}: impact {ev['impact']*100:.0f}%, feasibility {ev['feasibility']*100:.0f}%")

    for sname in ["collapse", "resilience", "extinction"]:
        sc = fe.generate_fate_scenario(sname, years=100)
        print(f"    Scenario '{sname}': final stability {sc['final_stability']:.3f}, outcome: {sc['outcome']}")


def demo_stress_testing():
    separator("Stress Testing Engine")
    engine = StressTestingEngine()
    predictions = {"gdp_growth": 0.5, "stock_market": 0.7, "inflation": 0.3, "unemployment": 0.2}
    report = engine.stress_test(predictions, threshold=0.5)
    print(f"  Overall Vulnerability:  {report.overall_vulnerability:.2%}")
    print(f"  Resilience Score:       {report.resilience_score:.2%}")
    print(f"  Worst Case:            {report.worst_case_scenario} (impact={report.worst_case_impact:.3f})")
    print(f"  Most Vulnerable:       {', '.join(report.most_vulnerable_variables[:3])}")
    print(f"  Recommendations:")
    for r in report.recommendations[:3]:
        print(f"    - {r}")


def demo_robust_optimizer():
    separator("Robust Optimizer")
    opt = RobustOptimizer(n_scenarios=10, n_iterations=30, uncertainty_radius=0.2)
    predictions = {"gdp_growth": 0.5, "inflation": 0.3, "stock_market": 0.7}
    for method in [RobustMethod.MINIMAX, RobustMethod.DISTRIBUTIONALLY_ROBUST, RobustMethod.SCENARIO_ROBUST]:
        result = opt.optimize(predictions, method=method)
        print(f"  {method.value:30s}  worst={result.worst_case_performance:.4f}  "
              f"nominal={result.nominal_performance:.4f}  por={result.price_of_robustness:.2%}")


def demo_red_team():
    separator("Red Team Forecaster")
    rt = RedTeamForecaster(n_adversarial_scenarios=3)
    predictions = {"gdp_growth": 0.5, "inflation": 0.3, "stock_market": 0.7}
    result = rt.red_team(predictions, confidence=0.9)
    print(f"  Overall Vulnerability:  {result.overall_vulnerability:.2%}")
    print(f"  Confidence Adjusted:    {result.confidence_adjustment:.0%} (from 90%)")
    print(f"  Biases Found:          {len(result.biases_found)}")
    print(f"  Assumption Ruptures:   {len(result.assumption_ruptures)}")
    print(f"  Key Weaknesses:")
    for w in result.key_weaknesses[:3]:
        print(f"    - {w}")
    print(f"  Recommendations:")
    for r in result.recommendations[:3]:
        print(f"    - {r}")


def demo_ensemble_calibrator():
    separator("Ensemble Belief Calibrator")
    cal = EnsembleBeliefCalibrator(n_bins=10)
    preds = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95]
    outcomes = [False, False, True, True, True, False, False, True, True, True]
    result = cal.calibrate(preds, outcomes)
    print(f"  ECE:                   {result.calibration_curve.ece:.4f}")
    print(f"  MCE:                   {result.calibration_curve.mce:.4f}")
    print(f"  Brier Score:          {result.calibration_curve.brier_score:.4f}")
    print(f"  Reliability:          {result.calibration_curve.reliability:.2%}")
    print(f"  Best Calibrator:       {result.best_calibrator.value}")
    print(f"  Confidence Adj:        {result.confidence_adjustment:.2%}")


def demo_temporal_consistency():
    separator("Temporal Consistency Checker")
    tc = TemporalConsistencyChecker()
    preds = {
        "1day": {"gdp_growth": 0.50, "inflation": 0.30},
        "1month": {"gdp_growth": 0.52, "inflation": 0.31},
        "1year": {"gdp_growth": 0.55, "inflation": 0.35},
        "5year": {"gdp_growth": 0.60, "inflation": 0.40},
    }
    result = tc.check(preds)
    print(f"  Consistent:           {result.is_consistent}")
    print(f"  Consistency Score:    {result.consistency_score:.2%}")
    print(f"  Smoothness Score:     {result.smoothness_score:.2%}")
    print(f"  Uncertainty Expansion:{result.uncertainty_expansion_score:.2%}")
    print(f"  Violations:          {len(result.violations)}")
    for v in result.violations[:3]:
        print(f"    - {v.issue_type.value}: {v.description[:60]}... (severity={v.severity:.2f})")


def demo_forecast_decomposition():
    separator("Forecast Decomposition")
    fd = ForecastDecomposer()
    preds = {"gdp_growth": 0.5, "inflation": 0.3, "stock_market": 0.7}
    result = fd.decompose(preds, method=DecompositionMethod.ADDITIVE)
    print(f"  Method:                {result.method.value}")
    print(f"  Explained Variance:   {result.total_explained_variance:.2%}")
    for var in preds:
        comps = result.components.get(var, [])
        print(f"  {var}:")
        for c in comps[:3]:
            print(f"    {c.name:12s}: {c.value:+.4f} ({c.proportion:.0%})")


def demo_scenario_probability():
    separator("Scenario Probability Mapper")
    sp = ScenarioProbabilityMapper(n_sensitivity_samples=20)
    preds = {"gdp_growth": 0.5, "inflation": 0.3}
    result = sp.map_scenarios(preds)
    print(f"  Top Scenario:          {result.top_scenario} ({result.top_probability:.0%})")
    print(f"  Expected Impact:      {result.expected_impact:.3f}")
    print(f"  Total Probability:     {result.total_probability:.2%}")
    print(f"  Scenarios:")
    for sid, s in result.scenarios.items():
        print(f"    {s.name:25s}: prob={s.probability:.2%}  impact={s.impact:.2f}")
    print(f"  Recommendations:")
    for r in result.recommendations[:3]:
        print(f"    - {r}")


def demo_predictive_fate():
    separator("Predictive Fate Engine")
    fe = PredictiveFateEngine(n_simulations=100)
    state = {"technology": 0.7, "energy": 0.5, "population": 0.6,
             "economy": 0.55, "cooperation": 0.4, "knowledge": 0.65,
             "environment": 0.35, "governance": 0.45}
    result = fe.project_fate(state, horizon_years=50)
    print(f"  Collapse Probability:  {result.collapse_probability:.2%}")
    print(f"  Golden Age Probability: {result.golden_age_probability:.2%}")
    print(f"  Stagnation Probability: {result.stagnation_probability:.2%}")
    print(f"  Dominant Phase:        {result.dominant_phase.value}")
    print(f"  Divergences:           {len(result.divergences)}")
    print(f"  Attractors:            {len(result.attractors)}")
    print(f"  Bifurcations:          {len(result.bifurcations)}")
    print(f"  Key Drivers:")
    for d in result.key_drivers[:5]:
        print(f"    - {d}")
    print(f"  Narrative: {result.narrative[:120]}...")


def demo_existential_risk():
    separator("Existential Risk Assessor")
    er = ExistentialRiskAssessor()
    result = er.assess({"geopolitical_risk": 0.4, "technology_risk": 0.5, "climate_risk": 0.5})
    print(f"  Total Existential P:   {result.total_existential_probability:.4%}")
    print(f"  10yr Risk:             {result.risk_timeline['10yr_existential']:.4%}")
    print(f"  100yr Risk:            {result.risk_timeline['100yr_existential']:.4%}")
    print(f"  Risk Categories:")
    for r in sorted(result.risks, key=lambda x: x.annual_probability, reverse=True)[:5]:
        print(f"    {r.category.value:20s}: {r.annual_probability:.6f}/yr  [{r.severity.value}]")
    print(f"  Narrative: {result.narrative[:120]}...")


def demo_prophecy():
    separator("Oracle Prophecy Synthesizer")
    os_synth = OracleProphecySynthesizer()
    result = os_synth.prophesize(
        {"gdp_growth": 0.55, "inflation": 0.3, "stock_market": 0.7},
        horizon=ProphecyHorizon.MEDIUM)
    print(f"  Overall Trajectory:    {result.overall_trajectory}")
    print(f"  Overall Confidence:    {result.overall_confidence.value}")
    print(f"  Outcomes:")
    for o in result.outcomes:
        print(f"    {o.variable:15s}: {o.prediction:.0%} [{o.lower_bound:.0%}-{o.upper_bound:.0%}] {o.confidence.value}")
    print(f"  Critical Junctures:   {len(result.critical_junctures)}")
    print(f"  Intervention Points:   {len(result.divine_intervention_points)}")
    print(f"  Pronouncement:")
    print(f"    {result.pronouncement[:200]}")
    print(f"  Recommendations:")
    for r in result.recommendations[:3]:
        print(f"    - {r}")


def demo_personal_oracle():
    separator("Personal Oracle — Your Future, Told True")
    oracle = PersonalOracle(n_simulations=200)
    person = PersonInput(
        age=30, sex="male", height_cm=175, weight_kg=75,
        education_years=16, annual_income=65000, net_worth=50000,
        debt=30000, savings_rate=0.10, exercise_minutes_weekly=90,
        sleep_hours=7.0, diet_quality=0.5, smoking="never",
        alcohol_drinks_weekly=3, close_friends=4, loneliness=0.3,
        purpose_score=0.5, stress_level=0.4,
    )
    result = oracle.divine(person)
    print(f"  Life Expectancy:        {result.life_expectancy} ({result.life_expectancy_ci[0]}-{result.life_expectancy_ci[1]})")
    print(f"  Healthspan:             {result.healthspan}")
    print(f"  Biological Age:         {result.biological_age} (chronological: {person.age})")
    print(f"  Determinism:            {result.determinism_score:.0%}")
    print(f"  Agency:                 {result.agency_score:.0%}")
    print(f"  Monte Carlo median:     {result.monte_carlo_summary['median_life_expectancy']}")
    print(f"  Die-before-65 prob:     {result.monte_carlo_summary['probability_die_before_65']:.1%}")
    print(f"  Top Mortality:          {', '.join(f'{m.cause}={m.lifetime_probability:.0%}' for m in result.mortality_breakdown[:3])}")
    print(f"  Inflection Points:      {len(result.inflection_points)}")
    print(f"  Oracle Reading:")
    print(f"    {result.oracle_reading[:200]}")
    print(f"  Recommendations:")
    for r in result.recommendations[:5]:
        print(f"    - {r}")


def demo_ts_forecast():
    separator("TS Forecast Ensemble — ARIMA + ETS + Theta + VAR")
    from oracle.prediction.ts_forecast import TSForecastEnsemble
    import numpy as np
    ensemble = TSForecastEnsemble(horizon=12, seed=42)
    np.random.seed(42)
    trend_data = np.cumsum(np.random.randn(120)) + 100
    result = ensemble.forecast(trend_data)
    print(f"  Regime:               {result.dominant_regime}")
    print(f"  Trend:                {result.trend_direction}")
    print(f"  Recommendation:       {result.recommendation[:100]}")
    print(f"  Anomaly Flags:        {', '.join(result.anomaly_flags[:3]) if result.anomaly_flags else 'None'}")
    print(f"  Ensemble Weights:")
    for k, v in result.ensemble_weights.items():
        print(f"    {k}: {v:.1%}")
    print(f"  Forecast (next 6 steps):")
    for fp in result.forecast[:6]:
        print(f"    Step {fp.step}: {fp.value:.2f} [{fp.lower_80:.2f}, {fp.upper_80:.2f}]")
    print(f"  Best Method:          {min(result.method_results, key=lambda k: result.method_results[k].mae)}")
    print(f"  Backtest Error:       {result.backtest.avg_error:.3f}" if result.backtest else "  Backtest: N/A")


def demo_scenario_generator():
    separator("Scenario Generator — Financial Monte Carlo")
    from oracle.prediction.scenario_generator import ScenarioGenerator, ScenarioInput
    gen = ScenarioGenerator(seed=42, n_simulations=500)
    inp = ScenarioInput()
    ret = gen.retirement(inp)
    print(f"  === RETIREMENT ===")
    print(f"  Success Probability:  {ret.success_probability:.1%}")
    print(f"  Median Portfolio:     ${ret.median_outcome:,.0f}")
    print(f"  P10/P90:             ${ret.p10:,.0f} / ${ret.p90:,.0f}")
    print(f"  FIRE Target:         ${ret.key_metrics['fire_target']:,.0f}")
    print(f"  FIRE Age:            {ret.key_metrics.get('fire_age', 'N/A')}")
    debt = gen.debt_payoff(inp)
    print(f"\n  === DEBT PAYOFF ===")
    print(f"  Months to Payoff:    {debt.key_metrics['months_to_payoff']}")
    print(f"  Total Interest:      ${debt.key_metrics['total_interest_paid']:,.0f}")
    print(f"  Interest % of Debt:  {debt.key_metrics['interest_pct_of_debt']:.1f}%")
    house = gen.house_purchase(inp)
    print(f"\n  === HOUSE PURCHASE ===")
    print(f"  Down Payment:        ${house.key_metrics['down_payment']:,.0f}")
    print(f"  Monthly PITI:        ${house.key_metrics['monthly_payment_piti']:,.0f}")
    print(f"  DTI Ratio:           {house.key_metrics['dti_ratio']:.1%}")
    print(f"  Median Yrs to Save:  {house.key_metrics['median_years_to_save']:.1f}")
    stress = gen.stress_test(inp, "2008_financial_crisis")
    print(f"\n  === STRESS TEST (2008) ===")
    print(f"  Median Portfolio:    ${stress.median_outcome:,.0f}")
    print(f"  Success Probability:  {stress.success_probability:.1%}")


def demo_prophecy_narrator():
    separator("Prophecy Narrator — Mr. House Oracle Engine")
    from oracle.prediction.prophecy_narrator import ProphecyNarrator
    narrator = ProphecyNarrator(use_llm=False, seed=42)
    states = {"equity": 0.55, "volatility": 0.40, "inflation": 0.50, "labor": 0.55,
              "monetary": 0.48, "geopolitical": 0.42, "sentiment": 0.55, "technology": 0.62,
              "economic": 0.54, "health": 0.60}
    result = narrator.prophesize(states)
    print(f'  Pronouncement: "{result.pronouncement}"')
    print(f"  Trajectory:          {result.overall_trajectory}")
    print(f"  Confidence:          {result.overall_confidence.value}")
    print(f"  Horizons:")
    for f in result.multi_horizon_forecasts:
        print(f"    {f.horizon.value:12s}: {f.overall_prediction:.2f} ({f.confidence.value})")
    print(f"  Cascades:")
    for c in result.cascade_analyses:
        print(f"    {c.chain_name}: {c.cumulative_probability:.1%} → {c.dominant_risk}")
    print(f"  Counterfactuals:")
    for cf in result.counterfactuals[:3]:
        print(f"    {cf.scenario}: {cf.probability_original:.0%} → {cf.probability_counterfactual:.0%} (Δ{cf.probability_delta:+.0%})")
    print(f"  Junctures:")
    for j in result.key_junctures[:3]:
        print(f"    - {j}")
    print(f"  Narrative: {result.narrative[:200]}")
    pid = narrator.record_prediction("equity", 0.55, 0.45, 0.65, "short_term", "high")
    print(f"\n  Prediction recorded: {pid}")
    score = narrator.resolve_prediction(pid, 0.58)
    print(f"  Score after resolve: grade={score.accuracy_grade}, within_ci={score.within_ci_pct}%")


def main():
    demo_dalio_cycles()
    demo_economic_model()
    demo_physics_model()
    demo_statistical_model()
    demo_buffett_value()
    demo_monte_carlo()
    demo_bayesian()
    demo_correlation_engine()
    demo_cycle_detector()
    demo_anomaly_detector()
    demo_regime_detector()
    demo_stress_test()
    demo_robust_optimizer()
    demo_red_team()
    demo_ensemble_calibrator()
    demo_temporal_consistency()
    demo_forecast_decomposition()
    demo_scenario_probability()
    demo_predictive_fate()
    demo_existential_risk()
    demo_prophecy()
    demo_personal_oracle()
    demo_ts_forecast()
    demo_scenario_generator()
    demo_prophecy_narrator()


if __name__ == "__main__":
    main()