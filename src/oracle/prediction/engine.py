import numpy as np
import logging
import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict
from oracle.core.config import OracleConfig
from oracle.core.types import (
    Domain, Prediction, Forecast, ConfidenceLevel, DataSource,
    TimeSeries, Signal, Correlation,
)
from oracle.deterministic.dalio_cycles import DalioCycleEngine, DebtCycleState, CycleDiagnosis
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
from oracle.domains.epidemiology import EpidemiologyModel
from oracle.domains.technology import TechnologyForecaster
from oracle.domains.social import SocialDynamicsModel
from oracle.domains.energy import EnergyModel, EnergyState
from oracle.domains.commodities import CommoditiesModel, CommoditiesState
from oracle.domains.infrastructure import InfrastructureResilienceModel, InfrastructureState
from oracle.domains.supply_chain import SupplyChainRiskModel, SupplyChainState
from oracle.domains.cyber import CyberThreatModel, CyberState
from oracle.domains.water import WaterSecurityModel, WaterState
from oracle.domains.food import FoodSecurityModel, FoodState
from oracle.domains.political import PoliticalStabilityModel, PoliticalState
from oracle.domains.biodiversity import BiodiversityModel, BiodiversityState
from oracle.domains.pandemic import PandemicPreparednessModel, PandemicState
from oracle.domains.space import SpaceModel, SpaceState
from oracle.domains.mental_health import MentalHealthModel, MentalHealthState
from oracle.domains.housing import HousingModel, HousingState
from oracle.domains.seismic import SeismicModel, SeismicState
from oracle.domains.solar import SolarModel, SolarState
from oracle.domains.oceanic import OceanicModel, OceanicState
from oracle.domains.neo import NearEarthObjectModel, NearEarthObjectState
from oracle.domains.atmospheric import AtmosphericModel, AtmosphericState
from oracle.domains.ecological import EcologicalModel, EcologicalState
from oracle.domains.paleoclimatology import PaleoclimatologyModel, PaleoclimateState
from oracle.domains.cosmic_radiation import CosmicRadiationModel, CosmicRadiationState
from oracle.domains.gravitational import GravitationalModel, GravitationalState
from oracle.domains.compound_hazard import CompoundHazardModel, CompoundHazardState
from oracle.domains.hazard_monte_carlo import NaturalHazardMonteCarlo
from oracle.domains.world_state import WorldStateModel
from oracle.domains.prescience import PrescienceEngine
from oracle.domains.fate import FateEngine, FateState, PersonalFateState
from oracle.domains.personal_life import PersonalLifeModel, PersonProfile, FamilyMemberProfile, SocialContactProfile
from oracle.nondeterministic.conformal import ConformalPredictor
from oracle.nondeterministic.causal import CausalInference
from oracle.nondeterministic.probabilistic import ProbabilisticForecaster
from oracle.prediction.bsts import BayesianStructuralTimeSeries
from oracle.prediction.correlation_dynamics import CorrelationDynamicsModel
from oracle.prediction.shapley import ShapleyAttribution
from oracle.prediction.online_learning import OnlineLearner
from oracle.prediction.kan import KolmogorovArnoldNetwork
from oracle.nondeterministic.quantile_regression import QuantileRegression
from oracle.prediction.advanced_ensemble import AdvancedEnsemble
from oracle.prediction.conditional_var import ConditionalVaRDecomposition
from oracle.nondeterministic.causal_advanced import DoubleMachineLearning
from oracle.prediction.bayesian_optimization import BayesianOptimizer
from oracle.prediction.gaussian_process import GaussianProcessPredictor
from oracle.nondeterministic.extreme_value import ExtremeValueTheory
from oracle.prediction.wavelet import WaveletDecomposition
from oracle.prediction.vine_copula import VineCopulaModel
from oracle.prediction.neural_ode import NeuralODE, NormalizingFlow
from oracle.nondeterministic.survival import SurvivalAnalysis
from oracle.prediction.information_theory import InformationTheoreticPredictor
from oracle.prediction.topological import TopologicalDataAnalysis
from oracle.prediction.state_space import StateSpaceModel
from oracle.prediction.rnn import RNNPredictor, LSTMNetwork
from oracle.prediction.hmm import HiddenMarkovModel
from oracle.prediction.copula_garch import CopulaGARCH, GARCHModel
from oracle.nondeterministic.causal_discovery import CausalDiscoveryEngine
from oracle.prediction.transformer_ts import TFTPredictor
from oracle.prediction.reinforcement_learning import RLDecisionOptimizer
from oracle.prediction.koopman import KoopmanOperator
from oracle.prediction.agent_based import AgentBasedModel
from oracle.prediction.symbolic_regression import SymbolicRegressor
from oracle.prediction.evolutionary_simulation import GeneticAlgorithm, FitnessLandscape, SpeciationSimulator, EvolutionResult, FitnessLandscapeResult
from oracle.prediction.learning_simulation import PopulationLearning, CulturalEvolution, SkillCascade, LearningResult, CulturalEvolutionResult, SkillCascadeResult
from oracle.prediction.scenario_engine import CausalLoopDiagram, SystemDynamicsModel, CrossDomainFeedback, ScenarioSimulator
from oracle.prediction.calibration import CalibrationTracker, CalibrationCurveResult, DomainCalibrationResult, PredictionAuditor
from oracle.prediction.adversarial_stress import AdversarialStressTester
from oracle.prediction.narrative import NarrativeEngine, StoryArc
from oracle.prediction.paradox_detector import ParadoxDetector, PhaseTransitionDetector
from oracle.prediction.data_quality import DataQualityScorer, QualityAdjustedForecaster
from oracle.prediction.weak_signal import WeakSignalDetector
from oracle.prediction.information_horizon import InformationHorizonMapper
from oracle.prediction.cascade_forecast import CascadeTree
from oracle.prediction.temporal_contradiction import TemporalContradictionDetector
from oracle.prediction.emergence_detector import EmergenceDetector
from oracle.prediction.foresight_chain import ForesightChainBuilder
from oracle.prediction.precursor_pattern import PrecursorPatternLibrary, PrecursorMatcher
from oracle.prediction.counterfactual import CounterfactualReasoner
from oracle.prediction.superforecast import SuperforecastEngine
from oracle.prediction.prediction_market import MarketSimulator
from oracle.prediction.black_swan import BlackSwanDetector
from oracle.prediction.regime_shift import RegimeShiftPredictor
from oracle.prediction.collective_intelligence import CollectiveIntelligenceAggregator
from oracle.prediction.analogical_reasoning import AnalogicalReasoningEngine
from oracle.prediction.morphological import MorphologicalAnalyzer
from oracle.prediction.butterfly_effect import ButterflyTracker
from oracle.prediction.delphi import DelphiMethodSimulator
from oracle.prediction.cross_impact import CrossImpactAnalyzer
from oracle.prediction.temporal_causal import TemporalCausalDiscoverer
from oracle.prediction.horizon_scanning import HorizonScanner
from oracle.prediction.probabilistic_network import ProbabilisticNetwork
from oracle.prediction.quantum_uncertainty import QuantumUncertaintyMapper
from oracle.prediction.decision_theory import DecisionTheoreticAdvisor
from oracle.prediction.strategic_foresight import StrategicForesightIntegrator
from oracle.prediction.precondition import PreconditionDetector
from oracle.prediction.signal_fusion import SignalFusionEngine
from oracle.prediction.stress_testing import StressTestingEngine
from oracle.prediction.robust_optimizer import RobustOptimizer
from oracle.prediction.red_team import RedTeamForecaster
from oracle.prediction.ensemble_calibrator import EnsembleBeliefCalibrator
from oracle.prediction.temporal_consistency import TemporalConsistencyChecker
from oracle.prediction.forecast_decomposition import ForecastDecomposer
from oracle.prediction.scenario_probability import ScenarioProbabilityMapper
from oracle.prediction.causal_enhanced import CausalInferenceEnhanced
from oracle.prediction.predictive_fate import PredictiveFateEngine
from oracle.prediction.existential_risk import ExistentialRiskAssessor
from oracle.prediction.prophecy import OracleProphecySynthesizer
from oracle.prediction.personal_oracle import PersonalOracle, PersonInput
from oracle.prediction.daily_oracle import DailyOracle
from oracle.prediction.life_trajectories import (HealthTrajectoryEngine, CareerTrajectoryEngine, RelationshipTrajectoryEngine, FinancialTrajectoryEngine, CognitiveTrajectoryEngine)
from oracle.prediction.ts_forecast import TSForecastEnsemble
from oracle.prediction.prediction_tracker import PredictionTracker
from oracle.prediction.anomaly_correlator import AnomalyCorrelator
from oracle.prediction.regime_assess import UnifiedRegimeAssessor
from oracle.prediction.portfolio_stress import PortfolioStressTester
from oracle.prediction.causal_event import CausalEventPredictor, CausalEventFullAnalyzer
from oracle.prediction.scale_bridging import ScaleBridgingPredictionEngine, ScaleBridgingResult, PredictionScale, ScaleCouplingType
from oracle.prediction.neuroscience import NeurosciencePredictor, NeuroscienceResult
from oracle.prediction.physics_forecast import PhysicsPredictor, PhysicsForecastResult
from oracle.prediction.synthetic_bio import SyntheticBioPredictor, BioForecastResult
from oracle.prediction.human_outcome import HumanOutcomePredictor, HumanOutcomeResult, HumanOutcomeDomain
from oracle.prediction.convergence import PredictiveConvergenceEngine, ModelPrediction, ConvergenceResult
from oracle.prediction.blind_spot import BlindSpotDetector, BlindSpotReport
from oracle.prediction.phase_space import PhaseSpaceMapper, PhaseSpaceResult
from oracle.prediction.forecastability import EntropicForecastabilityIndex, ForecastabilityReport
from oracle.prediction.auto_calibration import AutoCalibrationEngine, AutoCalibrationReport
from oracle.prediction.collective_engine import CollectiveConsciousnessEngine
from oracle.prediction.consciousness_tracker import TemporalConsciousnessTracker
from oracle.prediction.narrative_evolution import NarrativeEvolutionEngine
from oracle.prediction.memetic_pressure import MemeticPressureIndex
from oracle.prediction.predictive_action import PredictiveActionEngine
from oracle.prediction.westworld_fidelity import FidelityEngine
from oracle.prediction.cradle import CradleSimulator
from oracle.prediction.outlier_host import OutlierHostDetector
from oracle.prediction.narrative_loops import NarrativeLoopDetector
from oracle.prediction.optimal_path import OptimalPathOptimizer, Objective, ObjectiveDirection, Constraint
from oracle.prediction.entropy_meter import EntropyMeter
from oracle.prediction.force_whisper import ForceWhisperEngine
from oracle.evolution.continuous_engine import ContinuousEvolutionEngine
from oracle.domains.physical_constants import PhysicalConstants
from oracle.domains.ai_integration import OllamaBridge, AIInsightEngine, ModelResponse, InsightResult, ForecastCommentary

logger = logging.getLogger(__name__)


@dataclass
class OracleState:
    current_date: date = field(default_factory=date.today)
    economic_state: Optional[DebtCycleState] = None
    macro_state: Optional[MacroState] = None
    company_fundamentals: Optional[CompanyFundamentals] = None
    geopolitical_state: Optional[GeopoliticalState] = None
    climate_state: Optional[ClimateState] = None
    sentiment_data: Optional[SentimentData] = None
    contagion_state: Optional[ContagionState] = None
    demographics_state: Optional[DemographicsState] = None
    epidemiology_state: Optional[dict] = None
    technology_state: Optional[dict] = None
    social_state: Optional[dict] = None
    energy_state: Optional[dict] = None
    commodities_state: Optional[dict] = None
    infrastructure_state: Optional[InfrastructureState] = None
    supply_chain_state: Optional[SupplyChainState] = None
    cyber_state: Optional[CyberState] = None
    water_state: Optional[WaterState] = None
    food_state: Optional[FoodState] = None
    political_state: Optional[PoliticalState] = None
    biodiversity_state: Optional[BiodiversityState] = None
    pandemic_state: Optional[PandemicState] = None
    space_state: Optional[SpaceState] = None
    mental_health_state: Optional[MentalHealthState] = None
    housing_state: Optional[HousingState] = None
    seismic_state: Optional[SeismicState] = None
    solar_state: Optional[SolarState] = None
    oceanic_state: Optional[OceanicState] = None
    neo_state: Optional[NearEarthObjectState] = None
    atmospheric_state: Optional[AtmosphericState] = None
    ecological_state: Optional[EcologicalState] = None
    paleoclimate_state: Optional[PaleoclimateState] = None
    cosmic_radiation_state: Optional[CosmicRadiationState] = None
    gravitational_state: Optional[GravitationalState] = None
    compound_hazard_state: Optional[CompoundHazardState] = None
    fate_state: Optional[FateState] = None
    personal_fate_state: Optional[PersonalFateState] = None
    personal_life_profile: Optional[PersonProfile] = None
    time_series_data: dict[str, TimeSeries] = field(default_factory=dict)
    raw_series: dict[str, np.ndarray] = field(default_factory=dict)
    correlations: list[Correlation] = field(default_factory=list)
    cycle_diagnosis: Optional[CycleDiagnosis] = None
    regime_result: Optional[dict] = None


class OracleEngine:
    def __init__(self, config: Optional[OracleConfig] = None):
        self.config = config or OracleConfig()
        self.dalio = DalioCycleEngine()
        self.economic = EconomicDeterministicModel()
        self.physics = PhysicsModel()
        self.statistical = StatisticalDeterministicModel()
        self.buffett = BuffettValueEngine()
        self.monte_carlo = MonteCarloEngine(n_simulations=config.monte_carlo_simulations if config else 10000)
        self.bayesian = BayesianEngine()
        self.correlation = CorrelationEngine()
        self.cycle_detector = CycleDetector()
        self.anomaly_detector = AnomalyDetector()
        self.regime_detector = RegimeDetector()
        self.geopolitical = GeopoliticalRiskModel()
        self.climate = ClimateModel()
        self.sentiment = SentimentEngine()
        self.contagion = ContagionModel()
        self.demographics = DemographicsModel()
        self.epidemiology = EpidemiologyModel()
        self.technology = TechnologyForecaster()
        self.social = SocialDynamicsModel()
        self.energy = EnergyModel()
        self.commodities = CommoditiesModel()
        self.infrastructure = InfrastructureResilienceModel()
        self.supply_chain = SupplyChainRiskModel()
        self.cyber = CyberThreatModel()
        self.water = WaterSecurityModel()
        self.food = FoodSecurityModel()
        self.political = PoliticalStabilityModel()
        self.biodiversity = BiodiversityModel()
        self.pandemic = PandemicPreparednessModel()
        self.space = SpaceModel()
        self.mental_health = MentalHealthModel()
        self.housing = HousingModel()
        self.seismic = SeismicModel()
        self.solar = SolarModel()
        self.oceanic = OceanicModel()
        self.neo = NearEarthObjectModel()
        self.atmospheric = AtmosphericModel()
        self.ecological = EcologicalModel()
        self.paleoclimatology = PaleoclimatologyModel()
        self.cosmic_radiation = CosmicRadiationModel()
        self.gravitational = GravitationalModel()
        self.compound_hazard = CompoundHazardModel()
        self.hazard_monte_carlo = NaturalHazardMonteCarlo()
        self.world_state = WorldStateModel()
        self.prescience = PrescienceEngine()
        self.fate = FateEngine()
        self.personal_life = PersonalLifeModel()
        self.conformal = ConformalPredictor()
        self.causal = CausalInference()
        self.probabilistic = ProbabilisticForecaster()
        self.bsts = BayesianStructuralTimeSeries()
        self.correlation_dynamics = CorrelationDynamicsModel()
        self.shapley = ShapleyAttribution()
        self.online_learner = OnlineLearner()
        self.quantile_regression = QuantileRegression()
        self.kan = KolmogorovArnoldNetwork()
        self.advanced_ensemble = AdvancedEnsemble()
        self.conditional_var_decomp = ConditionalVaRDecomposition()
        self.double_ml = DoubleMachineLearning()
        self.bayesian_optimizer = BayesianOptimizer()
        self.gaussian_process = GaussianProcessPredictor()
        self.evt = ExtremeValueTheory()
        self.wavelet = WaveletDecomposition()
        self.vine_copula = VineCopulaModel()
        self.neural_ode = NeuralODE()
        self.normalizing_flow = NormalizingFlow()
        self.survival = SurvivalAnalysis()
        self.information_theory = InformationTheoreticPredictor()
        self.topological = TopologicalDataAnalysis()
        self.state_space = StateSpaceModel()
        self.rnn = RNNPredictor()
        self.hmm = HiddenMarkovModel()
        self.copula_garch = CopulaGARCH()
        self.causal_discovery = CausalDiscoveryEngine()
        self.tft = TFTPredictor()
        self.rl = RLDecisionOptimizer()
        self.koopman = KoopmanOperator()
        self.abm = AgentBasedModel()
        self.symbolic = SymbolicRegressor()
        self.evolutionary = GeneticAlgorithm()
        self.fitness_landscape = FitnessLandscape()
        self.speciation = SpeciationSimulator()
        self.population_learning = PopulationLearning()
        self.cultural_evolution = CulturalEvolution()
        self.skill_cascade = SkillCascade()
        self.physics_constants = PhysicalConstants()
        self.ai_insight = AIInsightEngine()
        self.causal_loop = CausalLoopDiagram()
        self.system_dynamics = SystemDynamicsModel()
        self.cross_domain_feedback = CrossDomainFeedback()
        self.scenario_simulator = ScenarioSimulator()
        self.calibration_tracker = CalibrationTracker()
        self.prediction_auditor = PredictionAuditor()
        self.adversarial_stress = AdversarialStressTester()
        self.narrative_engine = NarrativeEngine()
        self.paradox_detector = ParadoxDetector()
        self.phase_transition_detector = PhaseTransitionDetector()
        self.data_quality_scorer = DataQualityScorer()
        self.weak_signal = WeakSignalDetector()
        self.information_horizon = InformationHorizonMapper()
        self.cascade_tree = CascadeTree()
        self.temporal_contradiction = TemporalContradictionDetector()
        self.emergence_detector = EmergenceDetector()
        self.foresight_chain = ForesightChainBuilder()
        self.precursor_matcher = PrecursorMatcher()
        self.counterfactual = CounterfactualReasoner()
        self.superforecast = SuperforecastEngine()
        self.prediction_market = MarketSimulator()
        self.black_swan = BlackSwanDetector()
        self.regime_shift = RegimeShiftPredictor()
        self.collective_intelligence = CollectiveIntelligenceAggregator()
        self.analogical_reasoning = AnalogicalReasoningEngine()
        self.morphological = MorphologicalAnalyzer()
        self.butterfly_tracker = ButterflyTracker()
        self.delphi = DelphiMethodSimulator()
        self.cross_impact = CrossImpactAnalyzer()
        self.temporal_causal = TemporalCausalDiscoverer()
        self.horizon_scanner = HorizonScanner()
        self.probabilistic_network = ProbabilisticNetwork()
        self.quantum_uncertainty = QuantumUncertaintyMapper()
        self.decision_advisor = DecisionTheoreticAdvisor()
        self.strategic_foresight = StrategicForesightIntegrator()
        self.precondition_detector = PreconditionDetector()
        self.signal_fusion = SignalFusionEngine()
        self.stress_testing = StressTestingEngine()
        self.robust_optimizer = RobustOptimizer()
        self.red_team = RedTeamForecaster()
        self.ensemble_calibrator = EnsembleBeliefCalibrator()
        self.temporal_consistency = TemporalConsistencyChecker()
        self.forecast_decomposer = ForecastDecomposer()
        self.scenario_probability = ScenarioProbabilityMapper()
        self.causal_enhanced = CausalInferenceEnhanced()
        self.predictive_fate = PredictiveFateEngine()
        self.existential_risk = ExistentialRiskAssessor()
        self.prophecy_synthesizer = OracleProphecySynthesizer()
        self.personal_oracle = PersonalOracle()
        self.daily_oracle = DailyOracle(use_llm=True)
        self.health_trajectory = HealthTrajectoryEngine()
        self.career_trajectory = CareerTrajectoryEngine()
        self.relationship_trajectory = RelationshipTrajectoryEngine()
        self.financial_trajectory = FinancialTrajectoryEngine()
        self.cognitive_trajectory = CognitiveTrajectoryEngine()
        self.ts_forecast = TSForecastEnsemble()
        self.prediction_tracker = PredictionTracker()
        self.anomaly_correlator = AnomalyCorrelator()
        self.regime_assessor = UnifiedRegimeAssessor()
        self.portfolio_stress = PortfolioStressTester()
        self.causal_event_predictor = CausalEventPredictor()
        self.causal_event_analyzer = CausalEventFullAnalyzer()
        self.scale_bridging = ScaleBridgingPredictionEngine()
        self.neuroscience_predictor = NeurosciencePredictor()
        self.physics_predictor = PhysicsPredictor()
        self.synthetic_bio_predictor = SyntheticBioPredictor()
        self.human_outcome_predictor = HumanOutcomePredictor()
        self.convergence_engine = PredictiveConvergenceEngine()
        self.blind_spot_detector = BlindSpotDetector()
        self.phase_space_mapper = PhaseSpaceMapper()
        self.forecastability_index = EntropicForecastabilityIndex()
        self.auto_calibration = AutoCalibrationEngine()
        self.evolution_engine = ContinuousEvolutionEngine()
        self.collective_consciousness = CollectiveConsciousnessEngine()
        self.consciousness_tracker = TemporalConsciousnessTracker(
            heart_engine=self.collective_consciousness.heart,
            mind_landscape=self.collective_consciousness.mind,
            seed=42
        )
        self.narrative_evolution = NarrativeEvolutionEngine(seed=42)
        self.memetic_pressure = MemeticPressureIndex(
            heart_engine=self.collective_consciousness.heart,
            mind_landscape=self.collective_consciousness.mind,
            fusion_engine=self.collective_consciousness.fusion,
            seed=42
        )
        self.predictive_action = PredictiveActionEngine(
            pressure_index=self.memetic_pressure,
            fusion_engine=self.collective_consciousness.fusion,
            seed=42
        )
        self.fidelity_engine = FidelityEngine()
        self.cradle_simulator = CradleSimulator(random_seed=42)
        self.outlier_host_detector = OutlierHostDetector()
        self.narrative_loop_detector = NarrativeLoopDetector()
        self.optimal_path_optimizer = OptimalPathOptimizer(
            objectives=[
                Objective(name="stability", direction=ObjectiveDirection.MAXIMIZE, weight=0.3),
                Objective(name="growth", direction=ObjectiveDirection.MAXIMIZE, weight=0.3),
                Objective(name="equity", direction=ObjectiveDirection.MAXIMIZE, weight=0.2),
                Objective(name="risk", direction=ObjectiveDirection.MINIMIZE, weight=0.2),
            ],
            constraints=[
                Constraint(name="risk_cap", lhs_var="risk", comparator="<=", rhs=0.7),
            ],
            n_paths=200,
            max_steps=10,
            seed=42,
        )
        self.entropy_meter = EntropyMeter()
        self.force_whisper = ForceWhisperEngine()
        self.state = OracleState()

    def set_economic_state(
        self, gdp_growth_rate: float = 0.02, debt_to_gdp: float = 1.2,
        interest_rate: float = 0.045, money_supply_growth: float = 0.05,
        credit_growth: float = 0.04, productivity_growth: float = 0.015,
        inflation_rate: float = 0.03, unemployment_rate: float = 0.04,
        stock_valuation: float = 1.0, capacity_utilization: float = 0.82,
    ) -> "OracleEngine":
        self.state.economic_state = DebtCycleState(
            gdp_growth_rate=gdp_growth_rate, debt_to_gdp=debt_to_gdp,
            interest_rate=interest_rate, money_supply_growth=money_supply_growth,
            credit_growth=credit_growth, productivity_growth=productivity_growth,
            inflation_rate=inflation_rate, unemployment_rate=unemployment_rate,
            stock_valuation=stock_valuation, capacity_utilization=capacity_utilization,
            timestamp=date.today(),
        )
        self.state.macro_state = MacroState(
            gdp=25.0 * (1 + gdp_growth_rate), debt=25.0 * debt_to_gdp,
            interest_rate=interest_rate, inflation=inflation_rate,
            unemployment=unemployment_rate, productivity_index=100.0 * (1 + productivity_growth),
            money_supply=20.0, population_growth=0.01, savings_rate=0.18,
        )
        self.state.cycle_diagnosis = self.dalio.diagnose_cycle(self.state.economic_state)
        return self

    def set_company(
        self, ticker: str = "STOCK", eps: float = 5.0, book_value: float = 40.0,
        fcf: float = 4.0, dividend: float = 1.5, revenue_growth: float = 0.08,
        earnings_growth: float = 0.10, roe: float = 0.15, roic: float = 0.12,
        debt_to_equity: float = 0.5, current_ratio: float = 2.0,
        gross_margin: float = 0.45, operating_margin: float = 0.20,
        net_margin: float = 0.12, payout_ratio: float = 0.35,
        price: float = 100.0, sector: str = "general",
    ) -> "OracleEngine":
        self.state.company_fundamentals = CompanyFundamentals(
            ticker=ticker, earnings_per_share=eps, book_value_per_share=book_value,
            free_cash_flow_per_share=fcf, dividends_per_share=dividend,
            revenue_growth_5yr=revenue_growth, earnings_growth_5yr=earnings_growth,
            roe=roe, roic=roic, debt_to_equity=debt_to_equity,
            current_ratio=current_ratio, gross_margin=gross_margin,
            operating_margin=operating_margin, net_margin=net_margin,
            payout_ratio=payout_ratio, market_price=price, sector=sector,
        )
        return self

    def add_time_series(self, name: str, data: np.ndarray, domain: Domain = Domain.ALL,
                        source: DataSource = DataSource.CUSTOM) -> "OracleEngine":
        self.state.raw_series[name] = data
        return self

    def set_geopolitical_state(self, **kwargs) -> "OracleEngine":
        self.state.geopolitical_state = GeopoliticalState(**kwargs)
        return self

    def set_climate_state(self, **kwargs) -> "OracleEngine":
        self.state.climate_state = ClimateState(**kwargs)
        return self

    def set_sentiment_data(self, **kwargs) -> "OracleEngine":
        self.state.sentiment_data = SentimentData(**kwargs)
        return self

    def set_contagion_state(self, **kwargs) -> "OracleEngine":
        self.state.contagion_state = ContagionState(**kwargs)
        return self

    def set_demographics_state(self, **kwargs) -> "OracleEngine":
        self.state.demographics_state = DemographicsState(**kwargs)
        return self

    def set_epidemiology_state(self, population: float = 1e6, initial_infected: float = 100,
                                beta: float = 0.3, mu: float = 0.005, **kwargs) -> "OracleEngine":
        self.state.epidemiology_state = {"population": population, "initial_infected": initial_infected, "beta": beta, "mu": mu, **kwargs}
        return self

    def set_technology_state(self, current_adoption: float = 0.15, innovation_speed: float = 0.3, **kwargs) -> "OracleEngine":
        self.state.technology_state = {"current_adoption": current_adoption, "innovation_speed": innovation_speed, **kwargs}
        return self

    def set_social_state(self, gini: float = 0.41, unemployment: float = 0.06,
                         political_instability: float = 0.3, **kwargs) -> "OracleEngine":
        self.state.social_state = {"gini": gini, "unemployment": unemployment, "political_instability": political_instability, **kwargs}
        return self

    def set_energy_state(self, oil_price: float = 75.0, renewable_share: float = 20.0, **kwargs) -> "OracleEngine":
        self.state.energy_state = {"oil_price": oil_price, "renewable_share": renewable_share, **kwargs}
        return self

    def set_commodities_state(self, gold_price: float = 1900.0, supply_chain_stress: float = 0.3, **kwargs) -> "OracleEngine":
        self.state.commodities_state = {"gold_price": gold_price, "supply_chain_stress": supply_chain_stress, **kwargs}
        return self

    def set_infrastructure_state(self, power_grid_reliability: float = 0.85, water_system_reliability: float = 0.90,
                                  transport_reliability: float = 0.80, telecom_reliability: float = 0.88,
                                  healthcare_capacity: float = 0.70, financial_system_reliability: float = 0.92,
                                  supply_chain_reliability: float = 0.75, cyber_defense_score: float = 0.60,
                                  redundancy_level: float = 0.40, maintenance_score: float = 0.65,
                                  interdependency_strength: float = 0.50, climate_exposure: float = 0.30,
                                  seismic_risk: float = 0.20, flood_risk: float = 0.25, **kwargs) -> "OracleEngine":
        self.state.infrastructure_state = InfrastructureState(
            power_grid_reliability=power_grid_reliability, water_system_reliability=water_system_reliability,
            transport_reliability=transport_reliability, telecom_reliability=telecom_reliability,
            healthcare_capacity=healthcare_capacity, financial_system_reliability=financial_system_reliability,
            supply_chain_reliability=supply_chain_reliability, cyber_defense_score=cyber_defense_score,
            redundancy_level=redundancy_level, maintenance_score=maintenance_score,
            interdependency_strength=interdependency_strength, climate_exposure=climate_exposure,
            seismic_risk=seismic_risk, flood_risk=flood_risk, **kwargs,
        )
        return self

    def set_supply_chain_state(self, supplier_concentration: float = 0.40, logistics_disruption_prob: float = 0.15,
                                single_source_dependency: float = 0.30, visibility_score: float = 0.50,
                                alternate_supplier_availability: float = 0.60, **kwargs) -> "OracleEngine":
        self.state.supply_chain_state = SupplyChainState(
            supplier_concentration=supplier_concentration,
            logistics_disruption_prob=logistics_disruption_prob,
            single_source_dependency=single_source_dependency,
            visibility_score=visibility_score,
            alternate_supplier_availability=alternate_supplier_availability,
            **kwargs,
        )
        return self

    def set_cyber_state(self, threat_level: float = 0.4, ransomware_frequency: float = 0.3,
                        nation_state_activity: float = 0.5, zero_day_likelihood: float = 0.2,
                        cloud_dependency: float = 0.7, iot_vulnerability: float = 0.6,
                        ai_threat_capability: float = 0.3, incident_response_capacity: float = 0.5,
                        regulatory_compliance: float = 0.6, workforce_skill_level: float = 0.5,
                        patch_velocity: float = 0.4, supply_chain_cyber_risk: float = 0.4,
                        social_engineering_risk: float = 0.5, data_sensitivity: float = 0.7, **kwargs) -> "OracleEngine":
        self.state.cyber_state = CyberState(
            threat_level=threat_level, ransomware_frequency=ransomware_frequency,
            nation_state_activity=nation_state_activity, zero_day_likelihood=zero_day_likelihood,
            cloud_dependency=cloud_dependency, iot_vulnerability=iot_vulnerability,
            ai_threat_capability=ai_threat_capability, incident_response_capacity=incident_response_capacity,
            regulatory_compliance=regulatory_compliance, workforce_skill_level=workforce_skill_level,
            patch_velocity=patch_velocity, supply_chain_cyber_risk=supply_chain_cyber_risk,
            social_engineering_risk=social_engineering_risk, data_sensitivity=data_sensitivity,
            **kwargs,
        )
        return self

    def set_water_state(self, freshwater_availability_m3_per_capita: float = 5000.0, water_stress_index: float = 0.4,
                        groundwater_depletion_rate: float = 0.02, drought_probability: float = 0.2,
                        flood_risk: float = 0.25, water_quality_index: float = 0.65,
                        irrigation_efficiency: float = 0.55, desalination_capacity: float = 0.05,
                        transboundary_dependency: float = 0.2, climate_impact_factor: float = 0.3,
                        population_pressure: float = 0.4, **kwargs) -> "OracleEngine":
        self.state.water_state = WaterState(
            freshwater_availability_m3_per_capita=freshwater_availability_m3_per_capita,
            water_stress_index=water_stress_index, groundwater_depletion_rate=groundwater_depletion_rate,
            drought_probability=drought_probability, flood_risk=flood_risk,
            water_quality_index=water_quality_index, irrigation_efficiency=irrigation_efficiency,
            desalination_capacity=desalination_capacity, transboundary_dependency=transboundary_dependency,
            climate_impact_factor=climate_impact_factor, population_pressure=population_pressure,
            **kwargs,
        )
        return self

    def set_food_state(self, food_production_index: float = 0.70, food_price_volatility: float = 0.15,
                       crop_yield_variability: float = 0.20, fertilizer_dependency: float = 0.60,
                       water_for_agriculture: float = 0.70, supply_chain_disruption: float = 0.25,
                       climate_impact_on_crops: float = 0.20, population_growth_pressure: float = 0.30,
                       dietary_diversity: float = 0.55, food_waste_pct: float = 0.30,
                       reserve_stock_months: float = 3.0, trade_dependency: float = 0.25, **kwargs) -> "OracleEngine":
        self.state.food_state = FoodState(
            food_production_index=food_production_index, food_price_volatility=food_price_volatility,
            crop_yield_variability=crop_yield_variability, fertilizer_dependency=fertilizer_dependency,
            water_for_agriculture=water_for_agriculture, supply_chain_disruption=supply_chain_disruption,
            climate_impact_on_crops=climate_impact_on_crops, population_growth_pressure=population_growth_pressure,
            dietary_diversity=dietary_diversity, food_waste_pct=food_waste_pct,
            reserve_stock_months=reserve_stock_months, trade_dependency=trade_dependency,
            **kwargs,
        )
        return self

    def set_political_state(self, government_stability: float = 0.5, rule_of_law: float = 0.5,
                            corruption_index: float = 0.4, press_freedom: float = 0.5,
                            opposition_strength: float = 0.4, military_influence: float = 0.3,
                            economic_inequality: float = 0.4, ethnic_tension: float = 0.3,
                            institutional_trust: float = 0.45, external_pressure: float = 0.2,
                            electoral_integrity: float = 0.5, civil_liberties: float = 0.5,
                            regime_type: str = "democracy", **kwargs) -> "OracleEngine":
        self.state.political_state = PoliticalState(
            government_stability=government_stability, rule_of_law=rule_of_law,
            corruption_index=corruption_index, press_freedom=press_freedom,
            opposition_strength=opposition_strength, military_influence=military_influence,
            economic_inequality=economic_inequality, ethnic_tension=ethnic_tension,
            institutional_trust=institutional_trust, external_pressure=external_pressure,
            electoral_integrity=electoral_integrity, civil_liberties=civil_liberties,
            regime_type=regime_type, **kwargs,
        )
        return self

    def set_biodiversity_state(self, habitat_loss_rate: float = 0.3, species_extinction_rate: float = 0.4,
                                ecosystem_services_index: float = 0.55, protected_area_coverage: float = 0.15,
                                deforestation_rate: float = 0.3, ocean_acidification: float = 0.3,
                                invasive_species_impact: float = 0.25, pollution_index: float = 0.35,
                                overexploitation: float = 0.3, climate_impact_on_biodiversity: float = 0.35,
                                genetic_diversity_index: float = 0.5, coral_reef_health: float = 0.5,
                                pollinator_decline: float = 0.4, **kwargs) -> "OracleEngine":
        self.state.biodiversity_state = BiodiversityState(
            habitat_loss_rate=habitat_loss_rate, species_extinction_rate=species_extinction_rate,
            ecosystem_services_index=ecosystem_services_index, protected_area_coverage=protected_area_coverage,
            deforestation_rate=deforestation_rate, ocean_acidification=ocean_acidification,
            invasive_species_impact=invasive_species_impact, pollution_index=pollution_index,
            overexploitation=overexploitation, climate_impact_on_biodiversity=climate_impact_on_biodiversity,
            genetic_diversity_index=genetic_diversity_index, coral_reef_health=coral_reef_health,
            pollinator_decline=pollinator_decline, **kwargs,
        )
        return self

    def set_pandemic_state(self, healthcare_capacity: float = 0.5, surveillance_systems: float = 0.4,
                           vaccine_readiness: float = 0.3, supply_chain_resilience: float = 0.4,
                           international_cooperation: float = 0.5, public_trust: float = 0.45,
                           pandemic_preparedness_plan: float = 0.3, laboratory_capacity: float = 0.4,
                           risk_communication: float = 0.35, essential_worker_protection: float = 0.3,
                           genomic_surveillance: float = 0.25, ppe_stockpile: float = 0.35,
                           antiviral_stockpile: float = 0.2, contact_tracing_capacity: float = 0.3,
                           quarantine_capacity: float = 0.35, **kwargs) -> "OracleEngine":
        self.state.pandemic_state = PandemicState(
            healthcare_capacity=healthcare_capacity, surveillance_systems=surveillance_systems,
            vaccine_readiness=vaccine_readiness, supply_chain_resilience=supply_chain_resilience,
            international_cooperation=international_cooperation, public_trust=public_trust,
            pandemic_preparedness_plan=pandemic_preparedness_plan, laboratory_capacity=laboratory_capacity,
            risk_communication=risk_communication, essential_worker_protection=essential_worker_protection,
            genomic_surveillance=genomic_surveillance, ppe_stockpile=ppe_stockpile,
            antiviral_stockpile=antiviral_stockpile, contact_tracing_capacity=contact_tracing_capacity,
            quarantine_capacity=quarantine_capacity, **kwargs,
        )
        return self

    def set_space_state(self, orbital_debris_count: float = 25000.0, active_satellites: float = 7500.0,
                        space_budget_billion: float = 80.0, launch_frequency_per_year: float = 200.0,
                        kessler_syndrome_risk: float = 0.15, space_governance_index: float = 0.35,
                        commercial_space_revenue_billion: float = 400.0, space_military_index: float = 0.4,
                        broadband_satellite_share: float = 0.08, asteroid_mining_readiness: float = 0.05,
                        space_station_occupancy: float = 0.7, solar_storm_risk: float = 0.2, **kwargs) -> "OracleEngine":
        self.state.space_state = SpaceState(
            orbital_debris_count=orbital_debris_count, active_satellites=active_satellites,
            space_budget_billion=space_budget_billion, launch_frequency_per_year=launch_frequency_per_year,
            kessler_syndrome_risk=kessler_syndrome_risk, space_governance_index=space_governance_index,
            commercial_space_revenue_billion=commercial_space_revenue_billion, space_military_index=space_military_index,
            broadband_satellite_share=broadband_satellite_share, asteroid_mining_readiness=asteroid_mining_readiness,
            space_station_occupancy=space_station_occupancy, solar_storm_risk=solar_storm_risk, **kwargs,
        )
        return self

    def set_mental_health_state(self, depression_prevalence: float = 0.05, anxiety_prevalence: float = 0.04,
                               suicide_rate_per_100k: float = 10.5, treatment_gap_pct: float = 0.55,
                               mental_health_budget_pct: float = 0.02, social_isolation_index: float = 0.45,
                               substance_abuse_rate: float = 0.08, youth_mental_health_index: float = 0.45,
                               workplace_stress_index: float = 0.55, digital_wellbeing_index: float = 0.40,
                               crisis_hotline_volume: float = 0.5, stigma_index: float = 0.50, **kwargs) -> "OracleEngine":
        self.state.mental_health_state = MentalHealthState(
            depression_prevalence=depression_prevalence, anxiety_prevalence=anxiety_prevalence,
            suicide_rate_per_100k=suicide_rate_per_100k, treatment_gap_pct=treatment_gap_pct,
            mental_health_budget_pct=mental_health_budget_pct, social_isolation_index=social_isolation_index,
            substance_abuse_rate=substance_abuse_rate, youth_mental_health_index=youth_mental_health_index,
            workplace_stress_index=workplace_stress_index, digital_wellbeing_index=digital_wellbeing_index,
            crisis_hotline_volume=crisis_hotline_volume, stigma_index=stigma_index, **kwargs,
        )
        return self

    def set_housing_state(self, median_home_price: float = 350000.0, median_income: float = 70000.0,
                          mortgage_rate: float = 0.065, vacancy_rate: float = 0.05,
                          construction_permits_index: float = 0.55, rent_to_income_ratio: float = 0.30,
                          homelessness_per_100k: float = 17.0, housing_affordability_index: float = 0.45,
                          urbanization_rate: float = 0.83, zoning_restrictiveness: float = 0.6,
                          housing_supply_elasticity: float = 0.4, **kwargs) -> "OracleEngine":
        self.state.housing_state = HousingState(
            median_home_price=median_home_price, median_income=median_income,
            mortgage_rate=mortgage_rate, vacancy_rate=vacancy_rate,
            construction_permits_index=construction_permits_index, rent_to_income_ratio=rent_to_income_ratio,
            homelessness_per_100k=homelessness_per_100k, housing_affordability_index=housing_affordability_index,
            urbanization_rate=urbanization_rate, zoning_restrictiveness=zoning_restrictiveness,
            housing_supply_elasticity=housing_supply_elasticity, **kwargs,
        )
        return self

    def set_seismic_state(self, earthquake_rate_monthly: float = 12.0, avg_magnitude: float = 5.2,
                          max_magnitude_30d: float = 6.0, seismic_energy_release_30d: float = 1e16,
                          depth_avg_km: float = 35.0, fault_stress_accumulation: float = 0.5,
                          volcanic_activity_index: float = 0.3, gps_strain_rate: float = 0.02,
                          radon_anomaly: float = 0.2, electromagnetic_anomaly: float = 0.15,
                          swarm_activity: float = 0.2, subduction_zone_stress: float = 0.5,
                          transform_fault_slip_rate: float = 0.02, **kwargs) -> "OracleEngine":
        self.state.seismic_state = SeismicState(
            earthquake_rate_monthly=earthquake_rate_monthly, avg_magnitude=avg_magnitude,
            max_magnitude_30d=max_magnitude_30d, seismic_energy_release_30d=seismic_energy_release_30d,
            depth_avg_km=depth_avg_km, fault_stress_accumulation=fault_stress_accumulation,
            volcanic_activity_index=volcanic_activity_index, gps_strain_rate=gps_strain_rate,
            radon_anomaly=radon_anomaly, electromagnetic_anomaly=electromagnetic_anomaly,
            swarm_activity=swarm_activity, subduction_zone_stress=subduction_zone_stress,
            transform_fault_slip_rate=transform_fault_slip_rate, **kwargs,
        )
        return self

    def set_solar_state(self, sunspot_number: float = 80.0, solar_flux_10_7cm: float = 130.0,
                        flare_probability: float = 0.3, cme_probability: float = 0.2,
                        solar_wind_speed_kms: float = 400.0, interplanetary_magnetic_field_bt: float = 5.0,
                        kp_index: float = 3.0, dst_index: float = -20.0, proton_flux: float = 0.1,
                        aurora_latitude: float = 65.0, solar_cycle_phase: float = 0.6,
                        f10_7_variance: float = 20.0, **kwargs) -> "OracleEngine":
        self.state.solar_state = SolarState(
            sunspot_number=sunspot_number, solar_flux_10_7cm=solar_flux_10_7cm,
            flare_probability=flare_probability, cme_probability=cme_probability,
            solar_wind_speed_kms=solar_wind_speed_kms, interplanetary_magnetic_field_bt=interplanetary_magnetic_field_bt,
            kp_index=kp_index, dst_index=dst_index, proton_flux=proton_flux,
            aurora_latitude=aurora_latitude, solar_cycle_phase=solar_cycle_phase,
            f10_7_variance=f10_7_variance, **kwargs,
        )
        return self

    def set_oceanic_state(self, enso_index: float = 0.5, pdo_index: float = 0.2,
                          amo_index: float = 0.1, sea_surface_temp_anomaly_c: float = 0.8,
                          ocean_heat_content_700m: float = 0.5, thermohaline_circulation_strength: float = 0.7,
                          arctic_sea_ice_extent_mkm2: float = 4.5, atlantic_hurricane_sst_threshold: float = 26.5,
                          coral_bleaching_alert: float = 0.3, fisheries_productivity_index: float = 0.6,
                          sea_level_trend_mm_yr: float = 3.4, ocean_acidification_ph: float = 8.07, **kwargs) -> "OracleEngine":
        self.state.oceanic_state = OceanicState(
            enso_index=enso_index, pdo_index=pdo_index, amo_index=amo_index,
            sea_surface_temp_anomaly_c=sea_surface_temp_anomaly_c, ocean_heat_content_700m=ocean_heat_content_700m,
            thermohaline_circulation_strength=thermohaline_circulation_strength,
            arctic_sea_ice_extent_mkm2=arctic_sea_ice_extent_mkm2, atlantic_hurricane_sst_threshold=atlantic_hurricane_sst_threshold,
            coral_bleaching_alert=coral_bleaching_alert, fisheries_productivity_index=fisheries_productivity_index,
            sea_level_trend_mm_yr=sea_level_trend_mm_yr, ocean_acidification_ph=ocean_acidification_ph, **kwargs,
        )
        return self

    def set_neo_state(self, neo_count_tracked: float = 34000.0, neo_count_hazardous: float = 2300.0,
                      avg_close_approach_distance_ld: float = 15.0, largest_approach_diameter_m: float = 50.0,
                      detection_rate_pct: float = 0.60, torino_scale_max: float = 0.0,
                      cumulative_impact_probability_100yr: float = 0.0002, kinetic_energy_typical_mt: float = 10.0,
                      deflection_readiness: float = 0.05, asteroid_mining_interest: float = 0.1, **kwargs) -> "OracleEngine":
        self.state.neo_state = NearEarthObjectState(
            neo_count_tracked=neo_count_tracked, neo_count_hazardous=neo_count_hazardous,
            avg_close_approach_distance_ld=avg_close_approach_distance_ld, largest_approach_diameter_m=largest_approach_diameter_m,
            detection_rate_pct=detection_rate_pct, torino_scale_max=torino_scale_max,
            cumulative_impact_probability_100yr=cumulative_impact_probability_100yr, kinetic_energy_typical_mt=kinetic_energy_typical_mt,
            deflection_readiness=deflection_readiness, asteroid_mining_interest=asteroid_mining_interest, **kwargs,
        )
        return self

    def set_atmospheric_state(self, jet_stream_latitude_offset: float = 0.0, jet_stream_speed_kts: float = 130.0,
                               polar_vortex_strength: float = 0.6, stratospheric_warming_risk: float = 0.15,
                               ozone_column_du: float = 300.0, uv_index_anomaly: float = 0.0,
                               geomagnetic_field_strength_nt: float = 45000.0, geomagnetic_delta_nt: float = 0.0,
                               atmospheric_circulation_index: float = 0.5, hadley_cell_extent_deg: float = 30.0,
                               rossby_wave_amplitude: float = 0.4, atmospheric_co2_ppm: float = 420.0,
                               methane_ppb: float = 1900.0, aerosol_optical_depth: float = 0.12) -> "OracleEngine":
        self.state.atmospheric_state = AtmosphericState(
            jet_stream_latitude_offset=jet_stream_latitude_offset, jet_stream_speed_kts=jet_stream_speed_kts,
            polar_vortex_strength=polar_vortex_strength, stratospheric_warming_risk=stratospheric_warming_risk,
            ozone_column_du=ozone_column_du, uv_index_anomaly=uv_index_anomaly,
            geomagnetic_field_strength_nt=geomagnetic_field_strength_nt, geomagnetic_delta_nt=geomagnetic_delta_nt,
            atmospheric_circulation_index=atmospheric_circulation_index, hadley_cell_extent_deg=hadley_cell_extent_deg,
            rossby_wave_amplitude=rossby_wave_amplitude, atmospheric_co2_ppm=atmospheric_co2_ppm,
            methane_ppb=methane_ppb, aerosol_optical_depth=aerosol_optical_depth,
        )
        return self

    def set_ecological_state(self, biodiversity_index: float = 0.6, species_richness_per_km2: float = 50.0,
                             trophic_integrity: float = 0.5, keystone_species_ratio: float = 0.3,
                             ecosystem_connectivity: float = 0.5, habitat_fragmentation: float = 0.4,
                             invasive_species_impact: float = 0.3, pollution_load: float = 0.3,
                             climate_stress: float = 0.4, overexploitation_rate: float = 0.3,
                             adaptive_capacity: float = 0.4, genetic_diversity_index: float = 0.5,
                             evolutionary_pressure: float = 0.3, extinction_debt: float = 0.4) -> "OracleEngine":
        self.state.ecological_state = EcologicalState(
            biodiversity_index=biodiversity_index, species_richness_per_km2=species_richness_per_km2,
            trophic_integrity=trophic_integrity, keystone_species_ratio=keystone_species_ratio,
            ecosystem_connectivity=ecosystem_connectivity, habitat_fragmentation=habitat_fragmentation,
            invasive_species_impact=invasive_species_impact, pollution_load=pollution_load,
            climate_stress=climate_stress, overexploitation_rate=overexploitation_rate,
            adaptive_capacity=adaptive_capacity, genetic_diversity_index=genetic_diversity_index,
            evolutionary_pressure=evolutionary_pressure, extinction_debt=extinction_debt,
        )
        return self

    def set_paleoclimate_state(self, milankovitch_obliquity_deg: float = 23.45,
                                milankovitch_eccentricity: float = 0.0167,
                                milankovitch_precession: float = 0.0167,
                                ice_volume_fraction: float = 0.03,
                                co2_ppmv: float = 420.0,
                                global_temp_anomaly_c: float = 1.2,
                                solar_luminosity_fraction: float = 1.0,
                                volcanic_aerosol_loading: float = 0.05,
                                thermohaline_circulation_strength: float = 0.7,
                                vegetation_cover_fraction: float = 0.30,
                                methane_ppbv: float = 1900.0,
                                albedo: float = 0.30,
                                sea_level_m_above_present: float = 0.0) -> "OracleEngine":
        self.state.paleoclimate_state = PaleoclimateState(
            milankovitch_obliquity_deg=milankovitch_obliquity_deg,
            milankovitch_eccentricity=milankovitch_eccentricity,
            milankovitch_precession=milankovitch_precession,
            ice_volume_fraction=ice_volume_fraction, co2_ppmv=co2_ppmv,
            global_temp_anomaly_c=global_temp_anomaly_c,
            solar_luminosity_fraction=solar_luminosity_fraction,
            volcanic_aerosol_loading=volcanic_aerosol_loading,
            thermohaline_circulation_strength=thermohaline_circulation_strength,
            vegetation_cover_fraction=vegetation_cover_fraction,
            methane_ppbv=methane_ppbv, albedo=albedo,
            sea_level_m_above_present=sea_level_m_above_present,
        )
        return self

    def set_cosmic_radiation_state(self, gcr_flux_relative: float = 1.0,
                                     solar_modulation_potential_gv: float = 0.5,
                                     geomagnetic_shielding_eff: float = 0.7,
                                     atmospheric_shielding_eff: float = 0.85,
                                     svensmark_cloud_nucleation_factor: float = 0.3,
                                     neutron_monitor_count_rate: float = 200.0,
                                     cosmic_ray_dose_rate_usv_hr: float = 0.35,
                                     space_weather_shielding: float = 0.5,
                                     magnetosphere_compression: float = 0.3,
                                     ionization_rate_cm3_s: float = 2.0) -> "OracleEngine":
        self.state.cosmic_radiation_state = CosmicRadiationState(
            gcr_flux_relative=gcr_flux_relative,
            solar_modulation_potential_gv=solar_modulation_potential_gv,
            geomagnetic_shielding_eff=geomagnetic_shielding_eff,
            atmospheric_shielding_eff=atmospheric_shielding_eff,
            svensmark_cloud_nucleation_factor=svensmark_cloud_nucleation_factor,
            neutron_monitor_count_rate=neutron_monitor_count_rate,
            cosmic_ray_dose_rate_usv_hr=cosmic_ray_dose_rate_usv_hr,
            space_weather_shielding=space_weather_shielding,
            magnetosphere_compression=magnetosphere_compression,
            ionization_rate_cm3_s=ionization_rate_cm3_s,
        )
        return self

    def set_gravitational_state(self, lunar_tidal_force_relative: float = 1.0,
                                 solar_tidal_force_relative: float = 0.46,
                                 planetary_alignment_index: float = 0.2,
                                 earth_moon_distance_km: float = 384400.0,
                                 tidal_period_hours: float = 12.42,
                                 tidal_range_variation_pct: float = 5.0,
                                 solid_earth_tide_amplitude_cm: float = 30.0,
                                 ocean_tide_amplitude_m: float = 1.0,
                                 resonance_frequency_hz: float = 0.0001,
                                 rotational_energy_dissipation_pw: float = 3.7,
                                 crustal_stress_tidal_mpa: float = 0.3) -> "OracleEngine":
        self.state.gravitational_state = GravitationalState(
            lunar_tidal_force_relative=lunar_tidal_force_relative,
            solar_tidal_force_relative=solar_tidal_force_relative,
            planetary_alignment_index=planetary_alignment_index,
            earth_moon_distance_km=earth_moon_distance_km,
            tidal_period_hours=tidal_period_hours,
            tidal_range_variation_pct=tidal_range_variation_pct,
            solid_earth_tide_amplitude_cm=solid_earth_tide_amplitude_cm,
            ocean_tide_amplitude_m=ocean_tide_amplitude_m,
            resonance_frequency_hz=resonance_frequency_hz,
            rotational_energy_dissipation_pw=rotational_energy_dissipation_pw,
            crustal_stress_tidal_mpa=crustal_stress_tidal_mpa,
        )
        return self

    def set_compound_hazard_state(self, seismic_hazard: float = 0.3,
                                   volcanic_risk: float = 0.1,
                                   solar_storm_risk: float = 0.2,
                                   hurricane_risk: float = 0.3,
                                   flood_risk: float = 0.3,
                                   drought_risk: float = 0.2,
                                   pandemic_risk: float = 0.15,
                                   cyber_attack_risk: float = 0.3,
                                   supply_chain_fragility: float = 0.3,
                                   infrastructure_resilience: float = 0.6,
                                   economic_vulnerability: float = 0.3,
                                   social_cohesion: float = 0.5,
                                   governance_capacity: float = 0.6,
                                   climate_stress: float = 0.4,
                                   biodiversity_collapse_risk: float = 0.3) -> "OracleEngine":
        self.state.compound_hazard_state = CompoundHazardState(
            seismic_hazard=seismic_hazard,
            volcanic_risk=volcanic_risk,
            solar_storm_risk=solar_storm_risk,
            hurricane_risk=hurricane_risk,
            flood_risk=flood_risk,
            drought_risk=drought_risk,
            pandemic_risk=pandemic_risk,
            cyber_attack_risk=cyber_attack_risk,
            supply_chain_fragility=supply_chain_fragility,
            infrastructure_resilience=infrastructure_resilience,
            economic_vulnerability=economic_vulnerability,
            social_cohesion=social_cohesion,
            governance_capacity=governance_capacity,
            climate_stress=climate_stress,
            biodiversity_collapse_risk=biodiversity_collapse_risk,
        )
        return self

    def set_fate_state(self, human_population_bn: float = 8.1, energy_consumption_tw: float = 18.0,
                       gdp_world_trillion: float = 105.0, co2_ppm: float = 420.0,
                       nuclear_warheads: float = 12500.0, ai_capability_index: float = 0.15,
                       biodiversity_integrity_index: float = 0.35, social_cohesion_index: float = 0.45,
                       technological_dependency: float = 0.75, resource_remaining_pct: float = 50.0,
                       climate_commitment_c: float = 1.5, space_capability: float = 0.1,
                       biotech_capability: float = 0.2, governance_effectiveness: float = 0.4,
                       information_integrity: float = 0.4, water_stress_global: float = 0.4,
                       food_system_resilience: float = 0.5, pandemic_preparedness: float = 0.35,
                       economic_inequality_gini: float = 0.65, nuclear_proliferation_risk: float = 0.3,
                       asteroid_impact_risk: float = 0.0001, solar_catastrophe_risk: float = 0.012,
                       supervolcano_probability: float = 0.0007, gamma_ray_burst_risk: float = 0.0000005,
                       **kwargs) -> "OracleEngine":
        self.state.fate_state = FateState(
            human_population_bn=human_population_bn, energy_consumption_tw=energy_consumption_tw,
            gdp_world_trillion=gdp_world_trillion, co2_ppm=co2_ppm,
            nuclear_warheads=nuclear_warheads, ai_capability_index=ai_capability_index,
            biodiversity_integrity_index=biodiversity_integrity_index, social_cohesion_index=social_cohesion_index,
            technological_dependency=technological_dependency, resource_remaining_pct=resource_remaining_pct,
            climate_commitment_c=climate_commitment_c, space_capability=space_capability,
            biotech_capability=biotech_capability, governance_effectiveness=governance_effectiveness,
            information_integrity=information_integrity, water_stress_global=water_stress_global,
            food_system_resilience=food_system_resilience, pandemic_preparedness=pandemic_preparedness,
            economic_inequality_gini=economic_inequality_gini, nuclear_proliferation_risk=nuclear_proliferation_risk,
            asteroid_impact_risk=asteroid_impact_risk, solar_catastrophe_risk=solar_catastrophe_risk,
            supervolcano_probability=supervolcano_probability, gamma_ray_burst_risk=gamma_ray_burst_risk,
            **kwargs,
        )
        return self

    def set_personal_fate_state(self, age: float = 35.0, genetic_risk_score: float = 0.3,
                                 socioeconomic_origin: float = 0.5, education_level: float = 0.5,
                                 health_baseline: float = 0.7, habit_strength: float = 0.5,
                                 social_capital: float = 0.5, neuroplasticity: float = 0.6,
                                 trauma_load: float = 0.2, addiction_vulnerability: float = 0.3,
                                 epigenetic_stress: float = 0.3, cognitive_reserve: float = 0.5,
                                 relationship_stability: float = 0.5, financial_safety_net: float = 0.4,
                                 geographic_mobility: float = 0.5, talent_alignment: float = 0.4,
                                 purpose_clarity: float = 0.3, intergenerational_mobility: float = 0.4,
                                 chronic_disease_risk: float = 0.3, mental_health_baseline: float = 0.6,
                                 sleep_quality: float = 0.5, exercise_habit: float = 0.4,
                                 diet_quality: float = 0.5, loneliness_index: float = 0.3,
                                 adverse_childhood_experiences: float = 0.2, zip_code_destiny: float = 0.5,
                                 **kwargs) -> "OracleEngine":
        self.state.personal_fate_state = PersonalFateState(
            age=age, genetic_risk_score=genetic_risk_score,
            socioeconomic_origin=socioeconomic_origin, education_level=education_level,
            health_baseline=health_baseline, habit_strength=habit_strength,
            social_capital=social_capital, neuroplasticity=neuroplasticity,
            trauma_load=trauma_load, addiction_vulnerability=addiction_vulnerability,
            epigenetic_stress=epigenetic_stress, cognitive_reserve=cognitive_reserve,
            relationship_stability=relationship_stability, financial_safety_net=financial_safety_net,
            geographic_mobility=geographic_mobility, talent_alignment=talent_alignment,
            purpose_clarity=purpose_clarity, intergenerational_mobility=intergenerational_mobility,
            chronic_disease_risk=chronic_disease_risk, mental_health_baseline=mental_health_baseline,
            sleep_quality=sleep_quality, exercise_habit=exercise_habit,
            diet_quality=diet_quality, loneliness_index=loneliness_index,
            adverse_childhood_experiences=adverse_childhood_experiences, zip_code_destiny=zip_code_destiny,
            **kwargs,
        )
        return self

    def set_personal_life_profile(self, age: int = 35, biological_sex: str = "male",
                                   height_cm: float = 175.0, weight_kg: float = 75.0,
                                   annual_income: float = 75000.0, net_worth: float = 150000.0,
                                   debt_total: float = 50000.0, exercise_minutes_weekly: int = 90,
                                   sleep_hours_avg: float = 7.0, diet_quality: float = 0.5,
                                   smoking_status: str = "never", alcohol_drinks_weekly: int = 3,
                                   stress_level: float = 0.4, relationship_quality: float = 0.7,
                                   job_satisfaction: float = 0.6, life_satisfaction: float = 0.6,
                                   purpose_score: float = 0.5, resilience_score: float = 0.6,
                                   savings_rate: float = 0.15, career_growth_rate: float = 0.03,
                                   social_network_size: int = 12, close_friends: int = 3,
                                   community_involvement: float = 0.3,
                                   marital_status: str = "married", n_children: int = 0,
                                   education_years: int = 16, occupation: str = "professional",
                                   risk_tolerance: float = 0.5, investment_risk_profile: str = "moderate",
                                   **kwargs) -> "OracleEngine":
        self.state.personal_life_profile = PersonProfile(
            age=age, biological_sex=biological_sex, height_cm=height_cm,
            weight_kg=weight_kg, annual_income=annual_income, net_worth=net_worth,
            debt_total=debt_total, exercise_minutes_weekly=exercise_minutes_weekly,
            sleep_hours_avg=sleep_hours_avg, diet_quality=diet_quality,
            smoking_status=smoking_status, alcohol_drinks_weekly=alcohol_drinks_weekly,
            stress_level=stress_level, relationship_quality=relationship_quality,
            job_satisfaction=job_satisfaction, life_satisfaction=life_satisfaction,
            purpose_score=purpose_score, resilience_score=resilience_score,
            savings_rate=savings_rate, career_growth_rate=career_growth_rate,
            social_network_size=social_network_size, close_friends=close_friends,
            community_involvement=community_involvement,
            marital_status=marital_status, n_children=n_children,
            education_years=education_years, occupation=occupation,
            risk_tolerance=risk_tolerance, investment_risk_profile=investment_risk_profile,
            **kwargs,
        )
        return self

    def converge(self, predictions: Dict[str, List[ModelPrediction]]) -> ConvergenceResult:
        return self.convergence_engine.converge(predictions)

    def detect_blind_spots(self, domain_states: Optional[Dict[str, Dict]] = None, predictions: Optional[Dict[str, Dict]] = None) -> BlindSpotReport:
        return self.blind_spot_detector.detect(predictions=predictions, domain_states=domain_states)

    def map_phase_space(self, domain_states: Dict[str, float]) -> PhaseSpaceResult:
        return self.phase_space_mapper.map_phase_space(domain_states)

    def compute_forecastability(self, domain_states: Optional[Dict[str, float]] = None, predictions: Optional[List] = None) -> ForecastabilityReport:
        return self.forecastability_index.compute_all(domain_states=domain_states)

    def calibrate_predictions(self) -> AutoCalibrationReport:
        return self.auto_calibration.calibrate_all()

    def predict(
        self, target_date: date, confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    ) -> Forecast:
        _start_time = _time.time()
        predictions = []
        _errors = []
        years_ahead = (target_date - date.today()).days / 365.25

        if self.state.economic_state:
            dalio_preds = self.dalio.predict_from_cycle(
                self.state.economic_state, target_date, confidence
            )
            predictions.extend(dalio_preds)

        if self.state.macro_state:
            econ_preds = self.economic.predict(self.state.macro_state, target_date, confidence)
            predictions.extend(econ_preds)

        if self.state.company_fundamentals:
            buffett_pred = self.buffett.predict(self.state.company_fundamentals, target_date, confidence)
            predictions.append(buffett_pred)

        for name, data in self.state.raw_series.items():
            domain = Domain.ALL
            if "gdp" in name.lower() or "inflation" in name.lower() or "unemploy" in name.lower():
                domain = Domain.ECONOMICS
            elif "stock" in name.lower() or "price" in name.lower() or "market" in name.lower():
                domain = Domain.FINANCE

            try:
                stat_pred = self.statistical.predict(data, name, domain, target_date, confidence)
                predictions.append(stat_pred)
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.contagion_state:
            try:
                contagion_preds = self.contagion.predict(self.state.contagion_state, target_date, confidence)
                predictions.extend(contagion_preds)
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.climate_state:
            try:
                climate_preds = self.climate.predict(self.state.climate_state, target_date, confidence)
                predictions.extend(climate_preds)
                sl = self.climate.sea_level_projector(
                    current_temp_anomaly_c=self.state.climate_state.global_temp_anomaly_c,
                    current_sea_level_mm=self.state.climate_state.sea_level_rise_mm,
                    years=80,
                )
                predictions.append(Prediction(
                    domain=Domain.CLIMATE, variable="sea_level_rise_2100_mm",
                    value=float(sl["weighted_median_2100_mm"]),
                    confidence_interval=(float(sl["likely_range_2100_mm"][0]), float(sl["likely_range_2100_mm"][1])),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="climate_sea_level",
                    metadata={"thermal_pct": sl["thermal_contribution_pct"], "impacted_pop_m": sl["impacted_populations_millions"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.sentiment_data:
            try:
                sentiment_preds = self.sentiment.predict(self.state.sentiment_data, target_date, confidence)
                predictions.extend(sentiment_preds)
                cs = SentimentEngine.contrarian_signal_detector(
                    fear_greed_index=self.state.sentiment_data.fear_greed_index,
                    vix_level=self.state.sentiment_data.vix_level,
                    put_call_ratio=self.state.sentiment_data.put_call_ratio,
                    short_interest=self.state.sentiment_data.short_interest,
                    institutional_flow=self.state.sentiment_data.institutional_flow,
                    retail_flow=self.state.sentiment_data.retail_flow,
                    market_breadth=self.state.sentiment_data.market_breadth,
                    safe_haven_demand=self.state.sentiment_data.safe_haven_demand,
                    credit_spread=self.state.sentiment_data.credit_spread,
                    social_media_sentiment=self.state.sentiment_data.social_media_sentiment,
                    news_sentiment=self.state.sentiment_data.news_sentiment,
                    consumer_confidence=self.state.sentiment_data.consumer_confidence,
                )
                predictions.append(Prediction(
                    domain=Domain.FINANCE, variable="contrarian_signal_score",
                    value=float(cs["composite_contrarian_score"]),
                    confidence_interval=(float(cs["composite_contrarian_score"] - 0.15), float(cs["composite_contrarian_score"] + 0.15)),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="contrarian_signals",
                    metadata={"bias": cs["overall_bias"], "credit_risk": cs["credit_risk_level"], "n_extreme": len(cs["extreme_indicators"]), "n_divergences": len(cs["divergences"])},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.demographics_state:
            try:
                demo_preds = self.demographics.predict(self.state.demographics_state, target_date, confidence)
                predictions.extend(demo_preds)
                mp = DemographicsModel.migration_pressure_model(
                    origin_gdp_per_capita=self.state.demographics_state.gdp_per_capita,
                    youth_bulge_pct=self.state.demographics_state.youth_pct,
                    urbanization_rate=self.state.demographics_state.urbanization_rate,
                )
                predictions.append(Prediction(
                    domain=Domain.DEMOGRAPHICS, variable="migration_pressure",
                    value=float(mp["net_pressure"]),
                    confidence_interval=(float(max(0, mp["net_pressure"] - 0.15)), float(min(1, mp["net_pressure"] + 0.15))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="migration_pressure",
                    metadata={"level": mp["overall_level"], "push": mp["total_push_factor"], "pull": mp["pull_factor"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.epidemiology_state:
            try:
                epi_params = self.state.epidemiology_state
                seir_result = self.epidemiology.seir(
                    population=epi_params.get("population", 1e6),
                    initial_infected=epi_params.get("initial_infected", 100),
                    beta=epi_params.get("beta", 0.3),
                    mu=epi_params.get("mu", 0.005),
                )
                predictions.append(Prediction(
                    domain=Domain.EPIDEMIOLOGY, variable="peak_infected",
                    value=float(seir_result["peak_infected"]),
                    confidence_interval=(float(seir_result["peak_infected"] * 0.7), float(seir_result["peak_infected"] * 1.3)),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="epidemiology_seir",
                    metadata={"r0": seir_result["r0"], "herd_immunity": seir_result["herd_immunity_threshold"]},
                ))
                predictions.append(Prediction(
                    domain=Domain.EPIDEMIOLOGY, variable="total_deceased",
                    value=float(seir_result["total_deceased"]),
                    confidence_interval=(float(seir_result["total_deceased"] * 0.6), float(seir_result["total_deceased"] * 1.4)),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="epidemiology_seir",
                ))
                if epi_params.get("beta", 0.3) > 0.01:
                    try:
                        opt = self.epidemiology.intervention_optimizer(
                            population=epi_params.get("population", 1e6),
                            initial_infected=epi_params.get("initial_infected", 100),
                            beta=epi_params.get("beta", 0.3),
                            days=365,
                        )
                        best = opt.get("optimal_intervention", {})
                        predictions.append(Prediction(
                            domain=Domain.HEALTH, variable="intervention_deaths_prevented",
                            value=float(best.get("deaths_prevented", 0)),
                            confidence_interval=(float(best.get("deaths_prevented", 0) * 0.6), float(best.get("deaths_prevented", 0) * 1.4)),
                            confidence_level=confidence, timestamp=datetime.now(),
                            target_date=target_date, model_name="intervention_optimizer",
                            metadata={"peak_reduction_pct": best.get("peak_reduction_pct", 0), "recommendation": opt.get("recommendation", "")},
                        ))
                    except Exception as e:
                        logger.warning(f"Prediction error: {e}")
                        pass
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.technology_state:
            try:
                tech_params = self.state.technology_state
                curve = self.technology.s_curve_adoption(
                    current_adoption=tech_params.get("current_adoption", 0.15),
                    innovation_speed=tech_params.get("innovation_speed", 0.3),
                    years=int(years_ahead) if years_ahead > 0 else 10,
                )
                predictions.append(Prediction(
                    domain=Domain.TECHNOLOGY, variable="technology_adoption",
                    value=float(curve.get("predicted_adoption_5y", curve.get("current_adoption", 0.15))),
                    confidence_interval=(float(max(0, curve.get("predicted_adoption_5y", 0.15) * 0.7)), float(min(1, curve.get("predicted_adoption_5y", 0.15) * 1.3))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="technology_s_curve",
                    metadata={"phase": curve.get("phase", ""), "years_to_50pct": curve.get("years_to_50pct", 0)},
                ))
                qr = self.technology.quantum_readiness(years=int(years_ahead) if years_ahead > 0 else 10)
                predictions.append(Prediction(
                    domain=Domain.TECHNOLOGY, variable="quantum_readiness_score",
                    value=float(qr["current_score"]),
                    confidence_interval=(float(qr["current_score"] * 0.7), float(min(1.0, qr["projected_score_final"] * 1.1))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="quantum_readiness",
                    metadata={"readiness": qr["current_readiness"], "rsa_threat_years": qr["risk_assessment"].get("years_to_rsa_threat"), "mitigation_urgency": qr["risk_assessment"].get("mitigation_urgency")},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.social_state:
            try:
                social_params = self.state.social_state
                unrest = self.social.unrest_prediction(
                    gini_coefficient=social_params.get("gini", 0.41),
                    unemployment_rate=social_params.get("unemployment", 0.06),
                    political_instability=social_params.get("political_instability", 0.3),
                )
                predictions.append(Prediction(
                    domain=Domain.SOCIAL, variable="unrest_probability",
                    value=float(unrest["unrest_probability"]),
                    confidence_interval=(float(unrest["unrest_probability"] * 0.6), float(min(1.0, unrest["unrest_probability"] * 1.4))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="social_unrest",
                    metadata={"level": unrest["unrest_level"], "economic_distress": unrest["economic_distress"]},
                ))
                cascade = self.social.civil_unrest_cascade(n_simulations=200)
                predictions.append(Prediction(
                    domain=Domain.SOCIAL, variable="cascade_probability",
                    value=float(cascade["full_cascade_probability"]),
                    confidence_interval=(0.0, float(min(1.0, cascade["full_cascade_probability"] * 1.5))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="social_cascade",
                    metadata={"risk_level": cascade["cascade_risk_level"], "avg_activation": cascade["avg_activation_frac"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.energy_state:
            try:
                es = self.state.energy_state
                energy_state = EnergyState(
                    oil_price_usd=es.get("oil_price", 75.0),
                    renewable_share_pct=es.get("renewable_share", 20.0),
                    nuclear_share_pct=es.get("nuclear_share", 10.0),
                    coal_share_pct=es.get("coal_share", 27.0),
                    energy_demand_growth=es.get("demand_growth", 0.02),
                    carbon_price_usd=es.get("carbon_price", 25.0),
                    grid_stability=es.get("grid_stability", 0.7),
                )
                security = self.energy.energy_security_index(energy_state)
                predictions.append(Prediction(
                    domain=Domain.ENERGY, variable="energy_security_index",
                    value=security["overall_security_index"],
                    confidence_interval=(max(0, security["overall_security_index"] - 0.15), min(1, security["overall_security_index"] + 0.15)),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="energy_security",
                    metadata={"risk_level": security["risk_level"], "vulnerabilities": len(security["vulnerabilities"])},
                ))
                mix = self.energy.energy_mix_forecast(energy_state, years=5)
                predictions.append(Prediction(
                    domain=Domain.ENERGY, variable="renewable_share_5yr",
                    value=mix["renewable_share_final"] / 100.0,
                    confidence_interval=(max(0, mix["renewable_share_final"] / 100.0 - 0.05), min(1, mix["renewable_share_final"] / 100.0 + 0.05)),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="energy_mix",
                    metadata={"scenario": mix["scenario"], "co2_reduction_pct": mix["co2_reduction_pct"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.commodities_state:
            try:
                cs = self.state.commodities_state
                comm_state = CommoditiesState(
                    gold_price=cs.get("gold_price", 1900.0),
                    supply_chain_stress=cs.get("supply_chain_stress", 0.3),
                    geopolitical_risk=cs.get("geopolitical_risk", 0.4),
                )
                outlook = self.commodities.critical_minerals_outlook(comm_state, years=10)
                avg_scarcity = outlook.get("avg_scarcity", 0)
                predictions.append(Prediction(
                    domain=Domain.COMMODITIES, variable="critical_minerals_scarcity",
                    value=avg_scarcity,
                    confidence_interval=(max(0, avg_scarcity - 0.2), avg_scarcity + 0.2),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="commodities_scarcity",
                    metadata={"top_critical": outlook["top_critical"]},
                ))
                gold_forecast = self.commodities.price_forecast("gold", comm_state, days=252)
                predictions.append(Prediction(
                    domain=Domain.COMMODITIES, variable="gold_price_forecast",
                    value=gold_forecast["percentiles"]["p50"],
                    confidence_interval=(gold_forecast["percentiles"]["p10"], gold_forecast["percentiles"]["p90"]),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="commodities_gold",
                    metadata={"current_price": gold_forecast["current_price"], "expected_return_pct": gold_forecast.get("expected_return_pct", 0)},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.infrastructure_state:
            try:
                resilience = self.infrastructure.assess_resilience(self.state.infrastructure_state)
                predictions.append(Prediction(
                    domain=Domain.INFRASTRUCTURE, variable="infrastructure_resilience_index",
                    value=float(resilience["resilience_index"]),
                    confidence_interval=(float(max(0, resilience["resilience_index"] - 0.15)), float(min(1, resilience["resilience_index"] + 0.15))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="infrastructure_resilience",
                    metadata={"threat_level": resilience["threat_level"], "weakest_sector": resilience["weakest_sector"], "n_vulnerabilities": len(resilience["vulnerabilities"])},
                ))
                cascade = self.infrastructure.cascade_simulation(self.state.infrastructure_state, initial_failure="power_grid", n_simulations=200)
                predictions.append(Prediction(
                    domain=Domain.INFRASTRUCTURE, variable="cascade_collapse_probability",
                    value=float(cascade["total_collapse_probability"]),
                    confidence_interval=(0.0, float(min(1.0, cascade["total_collapse_probability"] * 2))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="infrastructure_cascade",
                    metadata={"cascade_risk": cascade["cascade_risk"], "avg_failed": cascade["avg_sectors_failed"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.supply_chain_state:
            try:
                sc_risk = self.supply_chain.assess_risk(self.state.supply_chain_state)
                predictions.append(Prediction(
                    domain=Domain.SUPPLY_CHAIN, variable="supply_chain_risk",
                    value=float(sc_risk["overall_risk"]),
                    confidence_interval=(float(max(0, sc_risk["overall_risk"] - 0.15)), float(min(1, sc_risk["overall_risk"] + 0.15))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="supply_chain_risk",
                    metadata={"risk_level": sc_risk["risk_level"], "resilience_score": sc_risk["resilience_score"], "n_vulnerabilities": sc_risk["n_vulnerabilities"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.cyber_state:
            try:
                cyber_assess = self.cyber.assess_threats(self.state.cyber_state)
                predictions.append(Prediction(
                    domain=Domain.CYBER, variable="cyber_threat_risk",
                    value=float(cyber_assess["overall_risk_score"]),
                    confidence_interval=(float(max(0, cyber_assess["overall_risk_score"] - 0.15)), float(min(1, cyber_assess["overall_risk_score"] + 0.15))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="cyber_threat",
                    metadata={"risk_level": cyber_assess["risk_level"], "vulnerability_index": cyber_assess["vulnerability_index"], "resilience_score": cyber_assess["resilience_score"]},
                ))
                cyber_pred = self.cyber.predict(years_ahead=years_ahead, state=self.state.cyber_state)
                predictions.append(Prediction(
                    domain=Domain.CYBER, variable="future_cyber_risk",
                    value=float(cyber_pred["future_risk_score"]),
                    confidence_interval=(float(max(0, cyber_pred["future_risk_score"] - 0.12)), float(min(1, cyber_pred["future_risk_score"] + 0.12))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="cyber_future",
                    metadata={"ai_threat_growth": cyber_pred["ai_threat_growth"], "quantum_threat_horizon": cyber_pred["quantum_threat_horizon"], "attack_surface_trend": cyber_pred["attack_surface_trend"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.water_state:
            try:
                water_assess = self.water.assess_security(self.state.water_state)
                predictions.append(Prediction(
                    domain=Domain.WATER, variable="water_security_risk",
                    value=float(water_assess["overall_risk"]),
                    confidence_interval=(float(max(0, water_assess["overall_risk"] - 0.15)), float(min(1, water_assess["overall_risk"] + 0.15))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="water_security",
                    metadata={"risk_level": water_assess["risk_level"], "resilience_score": water_assess["resilience_score"], "per_capita_status": water_assess["per_capita_status"]},
                ))
                water_pred = self.water.predict(years_ahead=years_ahead, state=self.state.water_state)
                predictions.append(Prediction(
                    domain=Domain.WATER, variable="water_future_risk",
                    value=float(water_pred["future_risk"]),
                    confidence_interval=(float(max(0, water_pred["future_risk"] - 0.12)), float(min(1, water_pred["future_risk"] + 0.12))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="water_future",
                    metadata={"water_stress_projected": water_pred["water_stress_projected"], "freshwater_per_capita_projected": water_pred["freshwater_per_capita_projected"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.food_state:
            try:
                food_assess = self.food.assess_security(self.state.food_state)
                predictions.append(Prediction(
                    domain=Domain.FOOD, variable="food_security_risk",
                    value=float(food_assess["overall_risk"]),
                    confidence_interval=(float(max(0, food_assess["overall_risk"] - 0.15)), float(min(1, food_assess["overall_risk"] + 0.15))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="food_security",
                    metadata={"risk_level": food_assess["risk_level"], "availability_score": food_assess["availability_score"], "stability_score": food_assess["stability_score"]},
                ))
                food_pred = self.food.predict(years_ahead=years_ahead, state=self.state.food_state)
                predictions.append(Prediction(
                    domain=Domain.FOOD, variable="food_future_risk",
                    value=float(food_pred["future_risk"]),
                    confidence_interval=(float(max(0, food_pred["future_risk"] - 0.12)), float(min(1, food_pred["future_risk"] + 0.12))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="food_future",
                    metadata={"hunger_risk_projected": food_pred["hunger_risk_projected"], "price_volatility_projected": food_pred["price_volatility_projected"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.political_state:
            try:
                pol_assess = self.political.assess_stability(self.state.political_state)
                predictions.append(Prediction(
                    domain=Domain.POLITICAL, variable="political_stability",
                    value=float(pol_assess["overall_stability"]),
                    confidence_interval=(float(max(0, pol_assess["overall_stability"] - 0.15)), float(min(1, pol_assess["overall_stability"] + 0.15))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="political_stability",
                    metadata={"stability_level": pol_assess["stability_level"], "coup_risk": pol_assess["coup_risk"], "protest_probability": pol_assess["protest_probability"]},
                ))
                pol_pred = self.political.predict(years_ahead=years_ahead, state=self.state.political_state)
                predictions.append(Prediction(
                    domain=Domain.POLITICAL, variable="political_future_stability",
                    value=float(pol_pred["future_stability"]),
                    confidence_interval=(float(max(0, pol_pred["future_stability"] - 0.12)), float(min(1, pol_pred["future_stability"] + 0.12))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="political_future",
                    metadata={"authoritarian_drift": pol_pred["authoritarian_drift"], "reform_likelihood": pol_pred["reform_likelihood"], "protest_forecast": pol_pred["protest_forecast"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.biodiversity_state:
            try:
                bio_assess = self.biodiversity.assess_health(self.state.biodiversity_state)
                predictions.append(Prediction(
                    domain=Domain.BIODIVERSITY, variable="biodiversity_index",
                    value=float(bio_assess["biodiversity_index"]),
                    confidence_interval=(float(max(0, bio_assess["biodiversity_index"] - 0.15)), float(min(1, bio_assess["biodiversity_index"] + 0.15))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="biodiversity_health",
                    metadata={"health_level": bio_assess["health_level"], "ecosystem_resilience": bio_assess["ecosystem_resilience"]["overall_resilience"], "n_tipping_points": bio_assess["n_tipping_points"]},
                ))
                bio_pred = self.biodiversity.predict(years_ahead=years_ahead, state=self.state.biodiversity_state)
                predictions.append(Prediction(
                    domain=Domain.BIODIVERSITY, variable="biodiversity_future",
                    value=float(bio_pred["future_biodiversity_index"]),
                    confidence_interval=(float(max(0, bio_pred["future_biodiversity_index"] - 0.12)), float(min(1, bio_pred["future_biodiversity_index"] + 0.12))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="biodiversity_future",
                    metadata={"mass_extinction_risk": bio_pred["mass_extinction_risk"], "ecosystem_collapse_probability": bio_pred["ecosystem_collapse_probability"], "conservation_trajectory": bio_pred["conservation_trajectory"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.pandemic_state:
            try:
                pan_assess = self.pandemic.assess_preparedness(self.state.pandemic_state)
                predictions.append(Prediction(
                    domain=Domain.PANDEMIC_PREPAREDNESS, variable="pandemic_preparedness",
                    value=float(pan_assess["overall_preparedness"]),
                    confidence_interval=(float(max(0, pan_assess["overall_preparedness"] - 0.15)), float(min(1, pan_assess["overall_preparedness"] + 0.15))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="pandemic_preparedness",
                    metadata={"readiness_level": pan_assess["readiness_level"], "detection_capacity": pan_assess["detection_capacity"], "treatment_capacity": pan_assess["treatment_capacity"]},
                ))
                pan_pred = self.pandemic.predict(years_ahead=years_ahead, state=self.state.pandemic_state)
                predictions.append(Prediction(
                    domain=Domain.PANDEMIC_PREPAREDNESS, variable="pandemic_future_preparedness",
                    value=float(pan_pred["future_preparedness"]),
                    confidence_interval=(float(max(0, pan_pred["future_preparedness"] - 0.12)), float(min(1, pan_pred["future_preparedness"] + 0.12))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="pandemic_future",
                    metadata={"pandemic_probability": pan_pred["pandemic_probability"], "resilience_score": pan_pred["resilience_score"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.space_state:
            try:
                space_assess = self.space.assess_orbital_environment(self.state.space_state)
                predictions.append(Prediction(
                    domain=Domain.SPACE, variable="orbital_environment",
                    value=float(space_assess["collision_probability"]),
                    confidence_interval=(float(max(0, space_assess["collision_probability"] - 0.1)), float(min(1, space_assess["collision_probability"] + 0.1))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="space_orbital",
                    metadata={"kessler_status": space_assess["kessler_status"], "leo_congestion": space_assess["leo_congestion"]},
                ))
                space_pred = self.space.predict(years_ahead=years_ahead, state=self.state.space_state)
                predictions.append(Prediction(
                    domain=Domain.SPACE, variable="space_future",
                    value=float(space_pred["future_kessler_risk"]),
                    confidence_interval=(float(max(0, space_pred["future_kessler_risk"] - 0.1)), float(min(1, space_pred["future_kessler_risk"] + 0.1))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="space_future",
                    metadata={"future_active_satellites": space_pred["future_active_satellites"], "orbital_sustainability_trend": space_pred["orbital_sustainability_trend"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.mental_health_state:
            try:
                mh_assess = self.mental_health.assess_population_health(self.state.mental_health_state)
                predictions.append(Prediction(
                    domain=Domain.MENTAL_HEALTH, variable="mental_health_burden",
                    value=float(mh_assess["overall_burden"]),
                    confidence_interval=(float(max(0, mh_assess["overall_burden"] - 0.12)), float(min(1, mh_assess["overall_burden"] + 0.12))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="mental_health_burden",
                    metadata={"burden_level": mh_assess["burden_level"], "capacity_level": mh_assess["capacity_level"], "youth_risk": mh_assess["youth_mental_health_risk"]},
                ))
                mh_pred = self.mental_health.predict(years_ahead=years_ahead, state=self.state.mental_health_state)
                predictions.append(Prediction(
                    domain=Domain.MENTAL_HEALTH, variable="mental_health_future",
                    value=float(mh_pred["future_treatment_gap"]),
                    confidence_interval=(float(max(0, mh_pred["future_treatment_gap"] - 0.1)), float(min(1, mh_pred["future_treatment_gap"] + 0.1))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="mental_health_future",
                    metadata={"trajectory": mh_pred["trajectory"], "affected_population_millions": mh_pred["affected_population_millions"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.housing_state:
            try:
                housing_assess = self.housing.assess_market(self.state.housing_state)
                predictions.append(Prediction(
                    domain=Domain.HOUSING, variable="housing_market_health",
                    value=float(housing_assess["market_health_score"]),
                    confidence_interval=(float(max(0, housing_assess["market_health_score"] - 0.1)), float(min(1, housing_assess["market_health_score"] + 0.1))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="housing_market_health",
                    metadata={"health_label": housing_assess["health_label"], "bubble_risk": housing_assess["bubble_risk"], "affordability_index": housing_assess["affordability_index"]},
                ))
                housing_pred = self.housing.predict(years_ahead=int(years_ahead), state=self.state.housing_state)
                predictions.append(Prediction(
                    domain=Domain.HOUSING, variable="housing_future_affordability",
                    value=float(housing_pred["future_affordability"]),
                    confidence_interval=(float(max(0, housing_pred["future_affordability"] - 0.1)), float(min(1, housing_pred["future_affordability"] + 0.1))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="housing_future",
                    metadata={"future_price_to_income": housing_pred["future_price_to_income"], "market_outlook": housing_pred["market_outlook"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.seismic_state:
            try:
                seismic_assess = self.seismic.assess_seismic_hazard(self.state.seismic_state)
                predictions.append(Prediction(
                    domain=Domain.SEISMIC, variable="seismic_hazard",
                    value=float(seismic_assess["composite_hazard"]),
                    confidence_interval=(float(max(0, seismic_assess["composite_hazard"] - 0.1)), float(min(1, seismic_assess["composite_hazard"] + 0.1))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="seismic_hazard",
                    metadata={"hazard_level": seismic_assess["hazard_level"], "m7_probability_30d": seismic_assess["m7_probability_30d"], "volcanic_risk": seismic_assess["volcanic_risk"]},
                ))
                seismic_pred = self.seismic.predict(years_ahead=int(years_ahead), state=self.state.seismic_state)
                predictions.append(Prediction(
                    domain=Domain.SEISMIC, variable="seismic_future",
                    value=float(seismic_pred["future_fault_stress"]),
                    confidence_interval=(float(max(0, seismic_pred["future_fault_stress"] - 0.1)), float(min(1, seismic_pred["future_fault_stress"] + 0.1))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="seismic_future",
                    metadata={"seismic_regime": seismic_pred["seismic_regime"], "m8_probability_5yr": seismic_pred["m8_probability_5yr"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.solar_state:
            try:
                solar_assess = self.solar.assess_solar_weather(self.state.solar_state)
                predictions.append(Prediction(
                    domain=Domain.SOLAR, variable="solar_weather_risk",
                    value=float(solar_assess["composite_risk"]),
                    confidence_interval=(float(max(0, solar_assess["composite_risk"] - 0.1)), float(min(1, solar_assess["composite_risk"] + 0.1))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="solar_weather",
                    metadata={"risk_level": solar_assess["risk_level"], "grid_risk": solar_assess["grid_infrastructure_risk"], "satellite_risk": solar_assess["satellite_risk"]},
                ))
                solar_pred = self.solar.predict(years_ahead=int(years_ahead), state=self.state.solar_state)
                predictions.append(Prediction(
                    domain=Domain.SOLAR, variable="solar_future",
                    value=float(solar_pred["carrington_probability"]),
                    confidence_interval=(0.0, float(min(1, solar_pred["carrington_probability"] + 0.05))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="solar_future",
                    metadata={"activity_regime": solar_pred["activity_regime"], "future_sunspot": solar_pred["future_sunspot_number"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.oceanic_state:
            try:
                ocean_assess = self.oceanic.assess_ocean_state(self.state.oceanic_state)
                predictions.append(Prediction(
                    domain=Domain.OCEANIC, variable="oceanic_risk",
                    value=float(ocean_assess["composite_ocean_risk"]),
                    confidence_interval=(float(max(0, ocean_assess["composite_ocean_risk"] - 0.1)), float(min(1, ocean_assess["composite_ocean_risk"] + 0.1))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="oceanic",
                    metadata={"risk_level": ocean_assess["risk_level"], "enso_phase": ocean_assess["enso_phase"], "expected_hurricanes": ocean_assess["expected_hurricane_season"]},
                ))
                ocean_pred = self.oceanic.predict(years_ahead=int(years_ahead), state=self.state.oceanic_state)
                predictions.append(Prediction(
                    domain=Domain.OCEANIC, variable="oceanic_future",
                    value=float(ocean_pred["future_sst_anomaly_c"]),
                    confidence_interval=(float(ocean_pred["future_sst_anomaly_c"] - 0.3), float(ocean_pred["future_sst_anomaly_c"] + 0.3)),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="oceanic_future",
                    metadata={"amoc_collapse_risk": ocean_pred["amoc_collapse_risk"], "ice_free_probability": ocean_pred["ice_free_summer_probability"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.neo_state:
            try:
                neo_assess = self.neo.assess_impact_risk(self.state.neo_state)
                predictions.append(Prediction(
                    domain=Domain.NEAR_EARTH_OBJECT, variable="neo_impact_risk",
                    value=float(neo_assess["composite_risk"]),
                    confidence_interval=(float(max(0, neo_assess["composite_risk"] - 0.05)), float(min(1, neo_assess["composite_risk"] + 0.05))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="neo_impact_risk",
                    metadata={"risk_level": neo_assess["risk_level"], "detection_gap": neo_assess["detection_gap_pct"], "deflection_readiness": neo_assess["deflection_readiness"]},
                ))
                neo_pred = self.neo.predict(years_ahead=50, state=self.state.neo_state)
                predictions.append(Prediction(
                    domain=Domain.NEAR_EARTH_OBJECT, variable="neo_future",
                    value=float(neo_pred["city_killer_probability_period"]),
                    confidence_interval=(0.0, float(min(1, neo_pred["city_killer_probability_period"] + 0.05))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="neo_future",
                    metadata={"future_detection_rate": neo_pred["future_detection_rate"], "future_deflection_readiness": neo_pred["future_deflection_readiness"]},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.atmospheric_state:
            try:
                atm_pred = self.atmospheric.predict(years_ahead=int(years_ahead), state=self.state.atmospheric_state)
                predictions.append(Prediction(
                    domain=Domain.ATMOSPHERIC, variable="atmospheric_risk",
                    value=float(atm_pred.get("future_co2_ppm", 420) / 1000.0),
                    confidence_interval=(0.0, 1.0),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="atmospheric_future",
                    metadata={"future_ozone_du": atm_pred.get("future_ozone_du", 300), "vortex_trend": atm_pred.get("vortex_trend", "variable"), "geomagnetic_trend": atm_pred.get("geomagnetic_trend", "stable")},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.ecological_state:
            try:
                eco_pred = self.ecological.predict(years_ahead=int(years_ahead), state=self.state.ecological_state)
                predictions.append(Prediction(
                    domain=Domain.ECOLOGICAL, variable="ecosystem_health",
                    value=float(eco_pred.get("future_biodiversity_index", 0.5)),
                    confidence_interval=(0.0, 1.0),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="ecological_future",
                    metadata={"ecosystem_regime": eco_pred.get("ecosystem_regime", "stressed"), "tipping_point_risk": eco_pred.get("tipping_point_risk", 0.3)},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.paleoclimate_state:
            try:
                paleo_pred = self.paleoclimatology.predict(years_ahead=max(100, int(years_ahead)), state=self.state.paleoclimate_state)
                predictions.append(Prediction(
                    domain=Domain.PALEOCLIMATOLOGY, variable="orbital_forcing",
                    value=float(paleo_pred.get("cycle_forcing" if "cycle_forcing" in paleo_pred else "insolation_variability", 0.5)),
                    confidence_interval=(0.0, 1.0),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="paleoclimatology_future",
                    metadata={"climate_regime": paleo_pred.get("climate_regime", "interglacial"), "tipping_point_risk": paleo_pred.get("tipping_point_risk", 0)},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.cosmic_radiation_state:
            try:
                cr_pred = self.cosmic_radiation.predict(years_ahead=int(years_ahead), state=self.state.cosmic_radiation_state)
                predictions.append(Prediction(
                    domain=Domain.COSMIC_RADIATION, variable="gcr_flux",
                    value=float(cr_pred.get("future_gcr_flux", 1.0)),
                    confidence_interval=(0.0, 3.0),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="cosmic_radiation_future",
                    metadata={"radiation_regime": cr_pred.get("radiation_regime", "near_average"), "svensmark_effect": cr_pred.get("future_svensmark_effect", 0)},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.gravitational_state:
            try:
                grav_pred = self.gravitational.predict(years_ahead=int(years_ahead), state=self.state.gravitational_state)
                predictions.append(Prediction(
                    domain=Domain.GRAVITATIONAL, variable="tidal_force",
                    value=float(grav_pred.get("future_lunar_tidal_force", 1.0)),
                    confidence_interval=(0.0, 2.0),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="gravitational_future",
                    metadata={"tidal_regime": grav_pred.get("tidal_regime", "current_cycle"), "seismic_contribution": grav_pred.get("seismic_contribution_index", 0)},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.compound_hazard_state:
            try:
                compound_assess = self.compound_hazard.assess_compound_risk(self.state.compound_hazard_state)
                predictions.append(Prediction(
                    domain=Domain.COMPOUND_HAZARD, variable="compound_hazard_index",
                    value=float(compound_assess["compound_hazard_index"]),
                    confidence_interval=(float(max(0, compound_assess["compound_hazard_index"] - 0.12)), float(min(1, compound_assess["compound_hazard_index"] + 0.12))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="compound_hazard",
                    metadata={"hazard_level": compound_assess["hazard_level"], "resilience_score": compound_assess["resilience_score"], "systemic_collapse_probability": compound_assess["systemic_collapse_probability"]},
                ))
                compound_sim = self.compound_hazard.simulate_scenario(scenario="multi_hazard_synchronous", years=int(years_ahead) if years_ahead >= 1 else 10, n_simulations=200, state=self.state.compound_hazard_state)
                predictions.append(Prediction(
                    domain=Domain.COMPOUND_HAZARD, variable="synchronous_hazard_probability",
                    value=float(compound_sim.get("annual_cascade_probability", 0)),
                    confidence_interval=(0.0, float(min(1, compound_sim.get("annual_cascade_probability", 0) * 2))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="compound_hazard_scenario",
                    metadata={"scenario": compound_sim.get("scenario_name", ""), "mean_impact": compound_sim.get("mean_impact_given_trigger", 0), "mean_recovery_years": compound_sim.get("mean_recovery_years", 0)},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.fate_state:
            try:
                fate_traj = self.fate.assess_civilizational_trajectory(self.state.fate_state)
                predictions.append(Prediction(
                    domain=Domain.FATE, variable="civilizational_trajectory",
                    value=float(fate_traj.get("trajectory_score", 0.5)),
                    confidence_interval=(float(max(0, fate_traj.get("trajectory_score", 0.5) - 0.15)), float(min(1, fate_traj.get("trajectory_score", 0.5) + 0.15))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="fate_trajectory",
                    metadata={"trajectory_label": fate_traj.get("trajectory_label", ""), "existential_risk": fate_traj.get("existential_risk_composite", 0), "kardashev_scale": fate_traj.get("kardashev_scale_current", 0.73)},
                ))
                fate_commit = self.fate.assess_deterministic_commitments(self.state.fate_state)
                predictions.append(Prediction(
                    domain=Domain.FATE, variable="fate_commitment",
                    value=float(fate_commit.get("fate_coupling", 0.5)),
                    confidence_interval=(0.0, 1.0),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="fate_commitment",
                    metadata={"n_commitments": fate_commit.get("n_commitments", 0), "avg_irreversibility": fate_commit.get("avg_irreversibility", 0), "high_irreversibility": fate_commit.get("n_high_irreversibility", 0)},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.personal_fate_state:
            try:
                personal_fate = self.fate.predict_personal_fate(self.state.personal_fate_state)
                predictions.append(Prediction(
                    domain=Domain.FATE, variable="personal_destiny",
                    value=float(personal_fate.get("total_determinism", 0.5)),
                    confidence_interval=(0.0, 1.0),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="personal_fate",
                    metadata={"fate_type": personal_fate.get("fate_type", ""), "agency_score": personal_fate.get("agency_score", 0.5)},
                ))
                mortality = personal_fate.get("mortality", {})
                predictions.append(Prediction(
                    domain=Domain.FATE, variable="life_expectancy",
                    value=float(mortality.get("life_expectancy_adjusted", 73)),
                    confidence_interval=(float(mortality.get("life_expectancy_adjusted", 73) - 5), float(mortality.get("life_expectancy_adjusted", 73) + 5)),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="personal_mortality",
                    metadata={"biological_age": mortality.get("biological_age", 35), "healthspan": mortality.get("healthspan_estimate", 58), "modifiable_years_gained": mortality.get("modifiable_years_gained", 0)},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.personal_life_profile:
            try:
                career = self.personal_life.predict_career_trajectory(self.state.personal_life_profile, years=20)
                predictions.append(Prediction(
                    domain=Domain.PERSONAL_LIFE, variable="career_trajectory_peak_income",
                    value=float(career.get("projected_peak_income", 0)),
                    confidence_interval=(float(career.get("projected_peak_income", 0) * 0.8), float(career.get("projected_peak_income", 0) * 1.2)),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="personal_life_career",
                    metadata={"current_stage": career.get("current_stage", ""), "growth_rate": career.get("base_growth_rate", 0), "years_to_peak": career.get("years_to_peak", 0)},
                ))
                health = self.personal_life.predict_health_trajectory(self.state.personal_life_profile, years=30)
                predictions.append(Prediction(
                    domain=Domain.PERSONAL_LIFE, variable="life_expectancy",
                    value=float(health.get("life_expectancy_adjusted", 79)),
                    confidence_interval=(float(health.get("life_expectancy_adjusted", 79) - 5), float(health.get("life_expectancy_adjusted", 79) + 5)),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="personal_life_health",
                    metadata={"healthspan": health.get("healthspan_estimate", 63), "bmi": health.get("current_bmi", 24.4), "bmi_category": health.get("bmi_category", "normal")},
                ))
                comprehensive = self.personal_life.predict_comprehensive_future(self.state.personal_life_profile, years=20)
                predictions.append(Prediction(
                    domain=Domain.PERSONAL_LIFE, variable="overall_outlook",
                    value=float(comprehensive.get("overall_outlook_score", 0.5)),
                    confidence_interval=(0.0, 1.0),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="personal_life_comprehensive",
                    metadata={"outlook_label": comprehensive.get("overall_outlook_label", ""), "n_concerns": len(comprehensive.get("key_concerns", [])), "n_opportunities": len(comprehensive.get("top_opportunities", []))},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.fate_state or self.state.climate_state or self.state.geopolitical_state:
            try:
                domain_scores = {}
                if self.state.economic_state:
                    domain_scores["economic"] = max(0, 1.0 - self.state.economic_state.debt_to_gdp / 3.0)
                if self.state.geopolitical_state:
                    domain_scores["geopolitical"] = max(0, 1.0 - (self.state.geopolitical_state.us_china_tension + self.state.geopolitical_state.russia_west_tension) / 2)
                if self.state.climate_state:
                    domain_scores["climate"] = max(0, 1.0 - self.state.climate_state.global_temp_anomaly_c / 4.0)
                if self.state.seismic_state:
                    domain_scores["seismic"] = max(0, 1.0 - self.state.seismic_state.avg_magnitude / 10.0)
                if self.state.solar_state:
                    domain_scores["solar"] = max(0, 1.0 - self.state.solar_state.flare_probability)
                if self.state.pandemic_state:
                    domain_scores["pandemic"] = max(0, 1.0 - self.state.pandemic_state.healthcare_capacity / 2.0)
                if self.state.cyber_state:
                    domain_scores["cyber"] = max(0, 1.0 - self.state.cyber_state.threat_level)
                if self.state.infrastructure_state:
                    domain_scores["infrastructure"] = max(0, 1.0 - self.state.infrastructure_state.power_grid_reliability)
                if self.state.supply_chain_state:
                    domain_scores["supply_chain"] = max(0, 1.0 - self.state.supply_chain_state.concentration_risk)
                if len(domain_scores) >= 3:
                    prescience = self.prescience.predict_prescience(domain_scores, current_temp_anomaly=self.state.climate_state.global_temp_anomaly_c if self.state.climate_state else 1.2)
                    predictions.append(Prediction(
                        domain=Domain.PRESCIENCE, variable="prescience_score",
                        value=float(prescience.get("prescience_score", 0)),
                        confidence_interval=(0.0, 1.0),
                        confidence_level=confidence, timestamp=datetime.now(),
                        target_date=target_date, model_name="prescience",
                        metadata={"risk_direction": prescience.get("risk_direction", ""), "alert_level": prescience.get("alert_level", ""), "n_active_warnings": len(prescience.get("active_warnings", [])), "n_imminent_tipping": len(prescience.get("imminent_tipping_points", []))},
                    ))
                    tipping = self.prescience.assess_tipping_points(domain_scores, current_temp_anomaly=self.state.climate_state.global_temp_anomaly_c if self.state.climate_state else 1.2)
                    imminent = [tp for tp in tipping.get("tipping_points", []) if tp.get("proximity", 0) > 0.8]
                    if imminent:
                        predictions.append(Prediction(
                            domain=Domain.PRESCIENCE, variable="imminent_tipping_points",
                            value=float(len(imminent)),
                            confidence_interval=(float(len(imminent)), float(len(imminent) + 2)),
                            confidence_level=confidence, timestamp=datetime.now(),
                            target_date=target_date, model_name="prescience_tipping",
                            metadata={"tipping_names": [tp.get("name", "") for tp in imminent[:5]], "max_irreversibility": max(tp.get("irreversibility", 0) for tp in imminent)},
                        ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.geopolitical_state:
            try:
                geo_risk = self.geopolitical.assess_risk(self.state.geopolitical_state)
                predictions.append(Prediction(
                    domain=Domain.GEOPOLITICS, variable="geopolitical_risk_score",
                    value=float(geo_risk.get("overall_risk", 0.5)),
                    confidence_interval=(float(max(0, geo_risk.get("overall_risk", 0.5) - 0.15)), float(min(1, geo_risk.get("overall_risk", 0.5) + 0.15))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="geopolitical_risk",
                    metadata={"stability_score": geo_risk.get("stability_score", 0), "conflict_risk": geo_risk.get("conflict_risk", 0)},
                ))
                shock = self.geopolitical.event_shock_simulator(
                    us_china_tension=self.state.geopolitical_state.us_china_tension,
                    russia_west_tension=self.state.geopolitical_state.russia_west_tension,
                    middle_east_instability=self.state.geopolitical_state.middle_east_instability,
                    trade_protectionism=self.state.geopolitical_state.trade_protectionism,
                )
                predictions.append(Prediction(
                    domain=Domain.GEOPOLITICS, variable="event_shock_impact",
                    value=float(shock.get("aggregate_impact", {}).get("p50", 0)),
                    confidence_interval=(float(shock.get("aggregate_impact", {}).get("p05", 0)), float(shock.get("aggregate_impact", {}).get("p95", 0))),
                    confidence_level=confidence, timestamp=datetime.now(),
                    target_date=target_date, model_name="event_shock",
                    metadata={"top_trigger": shock.get("top_trigger", ""), "cascade_prob": shock.get("cascade_probability", 0)},
                ))
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        ensemble_preds = self._ensemble_predictions(predictions, confidence)
        predictions.extend(ensemble_preds)

        domains = list(set(p.domain for p in predictions))

        return Forecast(
            predictions=predictions,
            time_range=(date.today(), target_date),
            domains=domains,
            model_versions={
                "dalio_cycle": "1.0",
                "economic_deterministic": "1.0",
                "physics_multi_scale": "1.0",
                "statistical_deterministic": "1.0",
                "buffett_value": "1.0",
                "monte_carlo": "1.0",
                "bayesian": "1.0",
                "epidemiology_model": "1.1",
                "technology_forecaster": "1.1",
                "social_dynamics_model": "1.1",
                "energy_model": "1.0",
                "commodities_model": "1.0",
                "geopolitical_model": "1.1",
                "climate_sea_level": "1.0",
                "contrarian_signals": "1.0",
                "migration_pressure_model": "1.0",
                "event_shock": "1.0",
                "intervention_optimizer": "1.0",
                "infrastructure_model": "1.0",
                "supply_chain_model": "1.0",
                "cyber_threat": "1.0",
                "cyber_future": "1.0",
                "water_security": "1.0",
                "water_future": "1.0",
                "food_security": "1.0",
                "food_future": "1.0",
                "political_stability": "1.0",
                "political_future": "1.0",
                "biodiversity_health": "1.0",
                "biodiversity_future": "1.0",
                "pandemic_preparedness": "1.0",
                "pandemic_future": "1.0",
                "space_orbital": "1.0",
                "space_future": "1.0",
                "mental_health_burden": "1.0",
                "mental_health_future": "1.0",
                "housing_market_health": "1.0",
                "housing_future": "1.0",
                "seismic_hazard": "1.0",
                "seismic_future": "1.0",
                "solar_weather": "1.0",
                "solar_future": "1.0",
                "oceanic": "1.0",
                "oceanic_future": "1.0",
                "neo_impact_risk": "1.0",
                "compound_hazard": "1.0",
                "compound_hazard_scenario": "1.0",
                "fate_trajectory": "1.0",
                "fate_commitment": "1.0",
                "personal_fate": "1.0",
                "personal_mortality": "1.0",
                "personal_life_career": "1.0",
                "personal_life_health": "1.0",
                "personal_life_comprehensive": "1.0",
                "prescience": "1.0",
            },
            ensemble_weights_used=self.config.ensemble_weights,
            error_log=_errors,
            failed_domains=list(set(
                p.domain.value for p in predictions if hasattr(p, 'metadata') and p.metadata.get('error')
            )),
            computation_time_ms=(_time.time() - _start_time) * 1000,
        )

    def diagnose(self) -> dict:
        result = {
            "timestamp": datetime.now().isoformat(),
            "cycle_diagnosis": None,
            "regime_analysis": None,
            "cycle_analysis": None,
            "anomalies": {},
            "cross_domain_correlations": [],
            "summary": {},
        }

        if self.state.economic_state:
            diagnosis = self.dalio.diagnose_cycle(self.state.economic_state)
            self.state.cycle_diagnosis = diagnosis
            result["cycle_diagnosis"] = {
                "phase": diagnosis.phase.value,
                "short_term_position": diagnosis.short_term_cycle_position,
                "long_term_position": diagnosis.long_term_cycle_position,
                "debt_sustainability": diagnosis.debt_sustainability_score,
                "bubble_risk": diagnosis.bubble_risk_score,
                "deleveraging_risk": diagnosis.deleveraging_risk_score,
                "policy_effectiveness": diagnosis.policy_effectiveness,
                "historical_analogues": diagnosis.historical_analogues,
                "confidence": diagnosis.confidence,
                "key_signals": diagnosis.key_signals,
            }

        for name, data in self.state.raw_series.items():
            try:
                regime = self.regime_detector.detect_hmm_regimes(data)
                result["regime_analysis"] = {
                    "current_regime": regime.current_regime,
                    "regime_probabilities": regime.regime_probabilities,
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

            try:
                cycles = self.cycle_detector.detect_cycles(data)
                result["cycle_analysis"] = [
                    {"period": c.dominant_period, "strength": c.strength, "frequency": c.dominant_frequency}
                    for c in cycles[:3]
                ]
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

            try:
                anomalies = self.anomaly_detector.detect_all(data)
                result["anomalies"][name] = [
                    {"index": a.index, "value": float(a.value), "type": a.anomaly_type, "score": float(a.score)}
                    for a in anomalies[:5]
                ]
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if len(self.state.raw_series) >= 2:
            try:
                correlations = self.correlation.discover_cross_domain_patterns(self.state.raw_series)
                result["cross_domain_correlations"] = [
                    {
                        "a": c.source_a, "b": c.source_b,
                        "correlation": float(c.correlation), "strength": c.strength,
                        "direction": c.direction, "lag": float(c.lag_days),
                    }
                    for c in correlations[:10]
                ]
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.cycle_diagnosis:
            d = self.state.cycle_diagnosis
            result["summary"] = {
                "economic_phase": d.phase.value,
                "risk_assessment": {
                    "bubble_risk": "HIGH" if d.bubble_risk_score > 0.6 else "MEDIUM" if d.bubble_risk_score > 0.3 else "LOW",
                    "deleveraging_risk": "HIGH" if d.deleveraging_risk_score > 0.6 else "MEDIUM" if d.deleveraging_risk_score > 0.3 else "LOW",
                    "debt_sustainability": "GOOD" if d.debt_sustainability_score > 0.6 else "CONCERNING" if d.debt_sustainability_score > 0.3 else "CRITICAL",
                },
                "historical_analogues": d.historical_analogues,
            }

        if self.state.geopolitical_state:
            try:
                geo_risk = self.geopolitical.assess_risk(self.state.geopolitical_state)
                result["geopolitical_risk"] = geo_risk
                geo_events = self.geopolitical.simulate_events(self.state.geopolitical_state, months=12, n_simulations=500)
                result["geopolitical_events"] = {k: v for k, v in geo_events.items() if k != "aggregate_impact"}
                result["geopolitical_events"]["aggregate"] = geo_events.get("aggregate_impact", {})
                shock = self.geopolitical.event_shock_simulator(
                    us_china_tension=self.state.geopolitical_state.us_china_tension,
                    russia_west_tension=self.state.geopolitical_state.russia_west_tension,
                    middle_east_instability=self.state.geopolitical_state.middle_east_instability,
                    trade_protectionism=self.state.geopolitical_state.trade_protectionism,
                )
                result["event_shock"] = {
                    "n_events_simulated": shock["n_simulations"],
                    "top_trigger": shock.get("top_trigger", ""),
                    "aggregate_impact_p50": shock.get("aggregate_impact", {}).get("p50", 0),
                    "aggregate_impact_p95": shock.get("aggregate_impact", {}).get("p95", 0),
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.climate_state:
            try:
                tipping = self.climate.tipping_points(self.state.climate_state)
                result["climate_tipping_points"] = {k: v for k, v in tipping.items() if k != "tipping_points"}
                result["climate_tipping_points"]["triggered"] = [tp["name"] for tp in tipping.get("tipping_points", []) if tp.get("triggered")]
                result["climate_tipping_points"]["close"] = [tp["name"] for tp in tipping.get("tipping_points", []) if 0.7 < tp.get("proximity", 0) < 1.0 and not tp.get("triggered")]
                sea_level = self.climate.sea_level_projector(
                    current_temp_anomaly_c=self.state.climate_state.global_temp_anomaly_c,
                    current_sea_level_mm=self.state.climate_state.sea_level_rise_mm,
                )
                result["sea_level_projection"] = {
                    "weighted_median_2100_mm": sea_level.get("weighted_median_2100_mm", 0),
                    "thermal_contribution_pct": sea_level.get("thermal_contribution_pct", 0),
                    "ice_contribution_pct": sea_level.get("ice_contribution_pct", 0),
                    "impacted_populations_millions": sea_level.get("impacted_populations_millions", 0),
                    "coastal_cities_at_risk": sea_level.get("coastal_cities_at_risk", 0),
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.contagion_state:
            try:
                cascade = self.contagion.cascade_threshold_model(self.state.contagion_state)
                result["contagion_cascade"] = {k: v for k, v in cascade.items() if k != "final_activation"}
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.sentiment_data:
            try:
                sentiment_analysis = self.sentiment.compute_composite_index(self.state.sentiment_data)
                result["sentiment"] = {k: v for k, v in sentiment_analysis.items() if k != "raw_scores"}
                contrarian = self.sentiment.detect_contrarian_signals(self.state.sentiment_data)
                result["contrarian_signals"] = contrarian.get("signals", [])
                advanced_contrarian = SentimentEngine.contrarian_signal_detector(
                    fear_greed_index=self.state.sentiment_data.fear_greed_index,
                    vix_level=self.state.sentiment_data.vix_level,
                    put_call_ratio=self.state.sentiment_data.put_call_ratio,
                    short_interest=self.state.sentiment_data.short_interest,
                    institutional_flow=self.state.sentiment_data.institutional_flow,
                    retail_flow=self.state.sentiment_data.retail_flow,
                    market_breadth=self.state.sentiment_data.market_breadth,
                    safe_haven_demand=self.state.sentiment_data.safe_haven_demand,
                    credit_spread=self.state.sentiment_data.credit_spread,
                    social_media_sentiment=self.state.sentiment_data.social_media_sentiment,
                    news_sentiment=self.state.sentiment_data.news_sentiment,
                    consumer_confidence=self.state.sentiment_data.consumer_confidence,
                )
                result["advanced_contrarian"] = {
                    "composite_contrarian_score": advanced_contrarian["composite_contrarian_score"],
                    "overall_bias": advanced_contrarian["overall_bias"],
                    "credit_risk_level": advanced_contrarian["credit_risk_level"],
                    "n_extreme_indicators": len(advanced_contrarian["extreme_indicators"]),
                    "n_divergences": len(advanced_contrarian["divergences"]),
                    "signals": advanced_contrarian["signals"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.demographics_state:
            try:
                dividend = self.demographics.aging_dividend(self.state.demographics_state)
                result["demographics_dividend"] = dividend
                migration = DemographicsModel.migration_pressure_model(
                    origin_gdp_per_capita=self.state.demographics_state.gdp_per_capita,
                    youth_bulge_pct=self.state.demographics_state.youth_pct,
                    urbanization_rate=self.state.demographics_state.urbanization_rate,
                )
                result["migration_pressure"] = {
                    "net_pressure": migration["net_pressure"],
                    "overall_level": migration["overall_level"],
                    "total_push_factor": migration["total_push_factor"],
                    "pull_factor": migration["pull_factor"],
                    "20yr_pct_of_origin": migration["20yr_pct_of_origin"],
                    "recommendations": migration["recommendations"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.epidemiology_state:
            try:
                epi_params = self.state.epidemiology_state
                seir = self.epidemiology.seir(
                    population=epi_params.get("population", 1e6),
                    initial_infected=epi_params.get("initial_infected", 100),
                    beta=epi_params.get("beta", 0.3),
                )
                result["epidemiology"] = {
                    "peak_infected": seir["peak_infected"],
                    "peak_day": seir["peak_day"],
                    "total_deceased": seir["total_deceased"],
                    "r0": seir["r0"],
                    "herd_immunity_threshold": seir["herd_immunity_threshold"],
                }
                if epi_params.get("beta", 0.3) > 0.01:
                    optimizer = self.epidemiology.intervention_optimizer(
                        population=epi_params.get("population", 1e6),
                        initial_infected=epi_params.get("initial_infected", 100),
                        beta=epi_params.get("beta", 0.3),
                        days=365,
                    )
                    result["intervention_optimization"] = {
                        "optimal_intervention": optimizer.get("optimal_intervention", {}),
                        "recommendation": optimizer.get("recommendation", ""),
                    }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.technology_state:
            try:
                tech_params = self.state.technology_state
                adoption = self.technology.s_curve_adoption(
                    current_adoption=tech_params.get("current_adoption", 0.15),
                    innovation_speed=tech_params.get("innovation_speed", 0.3),
                )
                disruption = self.technology.disruption_analysis()
                result["technology"] = {
                    "current_adoption": adoption["current_adoption"],
                    "predicted_adoption_5y": adoption["predicted_adoption_5y"],
                    "phase": adoption["phase"],
                    "top_disruptors": disruption["top_disruptors"],
                    "total_disruption_score": disruption["total_disruption_score"],
                }
                qr = self.technology.quantum_readiness(
                    years=tech_params.get("quantum_years", 10),
                    current_qubits=tech_params.get("current_qubits", 1000),
                    error_rate=tech_params.get("error_rate", 0.001),
                )
                result["quantum_readiness"] = {
                    "current_score": qr["current_score"],
                    "projected_score_final": qr["projected_score_final"],
                    "current_readiness": qr["current_readiness"],
                    "rsa_threat_years": qr["risk_assessment"].get("years_to_rsa_threat"),
                    "mitigation_urgency": qr["risk_assessment"].get("mitigation_urgency"),
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.social_state:
            try:
                social_params = self.state.social_state
                unrest = self.social.unrest_prediction(
                    gini_coefficient=social_params.get("gini", 0.41),
                    unemployment_rate=social_params.get("unemployment", 0.06),
                    political_instability=social_params.get("political_instability", 0.3),
                )
                polarization = self.social.polarization_index()
                result["social"] = {
                    "unrest_probability": unrest["unrest_probability"],
                    "unrest_level": unrest["unrest_level"],
                    "economic_distress": unrest["economic_distress"],
                    "polarization_index": polarization["polarization_index"],
                    "polarization_state": polarization["state"],
                }
                cascade = self.social.civil_unrest_cascade(
                    n_simulations=200,
                    base_activation=social_params.get("cascade_base_activation", 0.2),
                    connectivity=social_params.get("cascade_connectivity", 0.3),
                )
                result["social_cascade"] = {
                    "full_cascade_probability": cascade["full_cascade_probability"],
                    "cascade_risk_level": cascade["cascade_risk_level"],
                    "avg_activation_frac": cascade["avg_activation_frac"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.energy_state:
            try:
                es = self.state.energy_state
                energy_state = EnergyState(
                    oil_price_usd=es.get("oil_price", 75.0),
                    renewable_share_pct=es.get("renewable_share", 20.0),
                    nuclear_share_pct=es.get("nuclear_share", 10.0),
                    coal_share_pct=es.get("coal_share", 27.0),
                    energy_demand_growth=es.get("demand_growth", 0.02),
                    carbon_price_usd=es.get("carbon_price", 25.0),
                    grid_stability=es.get("grid_stability", 0.7),
                )
                security = self.energy.energy_security_index(energy_state)
                result["energy"] = {
                    "security_index": security["overall_security_index"],
                    "risk_level": security["risk_level"],
                    "vulnerabilities": len(security["vulnerabilities"]),
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.commodities_state:
            try:
                cs = self.state.commodities_state
                comm_state = CommoditiesState(
                    gold_price=cs.get("gold_price", 1900.0),
                    supply_chain_stress=cs.get("supply_chain_stress", 0.3),
                    geopolitical_risk=cs.get("geopolitical_risk", 0.4),
                )
                outlook = self.commodities.critical_minerals_outlook(comm_state, years=10)
                result["commodities"] = {
                    "top_critical_minerals": outlook["top_critical"],
                    "avg_scarcity_score": outlook.get("avg_scarcity", 0),
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.infrastructure_state:
            try:
                resilience = self.infrastructure.assess_resilience(self.state.infrastructure_state)
                result["infrastructure_resilience"] = {
                    "resilience_index": resilience["resilience_index"],
                    "threat_level": resilience["threat_level"],
                    "weakest_sector": resilience["weakest_sector"],
                    "redundancy_score": resilience["redundancy_score"],
                    "hazard_exposure": resilience["hazard_exposure"],
                    "n_vulnerabilities": len(resilience["vulnerabilities"]),
                }
                cascade = self.infrastructure.cascade_simulation(self.state.infrastructure_state, initial_failure=resilience["weakest_sector"], n_simulations=200)
                result["infrastructure_cascade"] = {
                    "avg_sectors_failed": cascade["avg_sectors_failed"],
                    "total_collapse_probability": cascade["total_collapse_probability"],
                    "cascade_risk": cascade["cascade_risk"],
                    "most_common_path": cascade["most_common_path"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.supply_chain_state:
            try:
                sc_risk = self.supply_chain.assess_risk(self.state.supply_chain_state)
                result["supply_chain_risk"] = {
                    "overall_risk": sc_risk["overall_risk"],
                    "risk_level": sc_risk["risk_level"],
                    "concentration_risk": sc_risk["concentration_risk"],
                    "disruption_risk": sc_risk["disruption_risk"],
                    "resilience_score": sc_risk["resilience_score"],
                    "n_vulnerabilities": sc_risk["n_vulnerabilities"],
                }
                disruption = self.supply_chain.disruption_simulation(self.state.supply_chain_state, n_simulations=200)
                result["supply_chain_disruption"] = {
                    "revenue_impact_p50": disruption["revenue_impact"]["p50"],
                    "avg_recovery_days": disruption["avg_recovery_days"],
                    "cascade_risk": disruption["cascade_risk"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.cyber_state:
            try:
                cyber_assess = self.cyber.assess_threats(self.state.cyber_state)
                result["cyber_threat"] = {
                    "overall_risk_score": cyber_assess["overall_risk_score"],
                    "risk_level": cyber_assess["risk_level"],
                    "vulnerability_index": cyber_assess["vulnerability_index"],
                    "resilience_score": cyber_assess["resilience_score"],
                    "attack_surface": cyber_assess["attack_surface"],
                    "top_threats": cyber_assess["top_threats"][:5],
                }
                cyber_pred = self.cyber.predict(years_ahead=5, state=self.state.cyber_state)
                result["cyber_future"] = {
                    "future_risk_score": cyber_pred["future_risk_score"],
                    "ai_threat_growth": cyber_pred["ai_threat_growth"],
                    "quantum_threat_horizon": cyber_pred["quantum_threat_horizon"],
                    "attack_surface_trend": cyber_pred["attack_surface_trend"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.water_state:
            try:
                water_assess = self.water.assess_security(self.state.water_state)
                result["water_security"] = {
                    "overall_risk": water_assess["overall_risk"],
                    "risk_level": water_assess["risk_level"],
                    "resilience_score": water_assess["resilience_score"],
                    "scarcity_types": water_assess["scarcity_types"],
                    "components": water_assess["components"],
                }
                water_pred = self.water.predict(years_ahead=10, state=self.state.water_state)
                result["water_future"] = {
                    "future_risk": water_pred["future_risk"],
                    "water_stress_projected": water_pred["water_stress_projected"],
                    "freshwater_per_capita_projected": water_pred["freshwater_per_capita_projected"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.food_state:
            try:
                food_assess = self.food.assess_security(self.state.food_state)
                result["food_security"] = {
                    "overall_risk": food_assess["overall_risk"],
                    "risk_level": food_assess["risk_level"],
                    "availability_score": food_assess["availability_score"],
                    "access_score": food_assess["access_score"],
                    "stability_score": food_assess["stability_score"],
                    "sustainability_score": food_assess["sustainability_score"],
                }
                food_pred = self.food.predict(years_ahead=10, state=self.state.food_state)
                result["food_future"] = {
                    "future_risk": food_pred["future_risk"],
                    "hunger_risk_projected": food_pred["hunger_risk_projected"],
                    "price_volatility_projected": food_pred["price_volatility_projected"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.political_state:
            try:
                pol_assess = self.political.assess_stability(self.state.political_state)
                result["political_stability"] = {
                    "overall_stability": pol_assess["overall_stability"],
                    "stability_level": pol_assess["stability_level"],
                    "coup_risk": pol_assess["coup_risk"],
                    "protest_probability": pol_assess["protest_probability"],
                    "reform_prospects": pol_assess["reform_prospects"],
                    "democratic_health": pol_assess["democratic_health"],
                    "governance_effectiveness": pol_assess["governance_effectiveness"],
                    "n_instability_factors": pol_assess["n_instability_factors"],
                }
                pol_pred = self.political.predict(years_ahead=5, state=self.state.political_state)
                result["political_future"] = {
                    "future_stability": pol_pred["future_stability"],
                    "authoritarian_drift": pol_pred["authoritarian_drift"],
                    "reform_likelihood": pol_pred["reform_likelihood"],
                    "protest_forecast": pol_pred["protest_forecast"],
                    "political_violence_risk": pol_pred["political_violence_risk"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.biodiversity_state:
            try:
                bio_assess = self.biodiversity.assess_health(self.state.biodiversity_state)
                result["biodiversity_health"] = {
                    "biodiversity_index": bio_assess["biodiversity_index"],
                    "health_level": bio_assess["health_level"],
                    "ecosystem_resilience": bio_assess["ecosystem_resilience"]["overall_resilience"],
                    "n_tipping_points": bio_assess["n_tipping_points"],
                    "species_outlook": bio_assess["species_outlook"],
                }
                bio_pred = self.biodiversity.predict(years_ahead=10, state=self.state.biodiversity_state)
                result["biodiversity_future"] = {
                    "future_biodiversity_index": bio_pred["future_biodiversity_index"],
                    "mass_extinction_risk": bio_pred["mass_extinction_risk"],
                    "ecosystem_collapse_probability": bio_pred["ecosystem_collapse_probability"],
                    "conservation_trajectory": bio_pred["conservation_trajectory"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.pandemic_state:
            try:
                pan_assess = self.pandemic.assess_preparedness(self.state.pandemic_state)
                result["pandemic_preparedness"] = {
                    "overall_preparedness": pan_assess["overall_preparedness"],
                    "readiness_level": pan_assess["readiness_level"],
                    "detection_capacity": pan_assess["detection_capacity"],
                    "treatment_capacity": pan_assess["treatment_capacity"],
                    "prevention_capacity": pan_assess["prevention_capacity"],
                    "n_gaps": pan_assess["n_gaps"],
                }
                pan_pred = self.pandemic.predict(years_ahead=5, state=self.state.pandemic_state)
                result["pandemic_future"] = {
                    "future_preparedness": pan_pred["future_preparedness"],
                    "pandemic_probability": pan_pred["pandemic_probability"],
                    "resilience_score": pan_pred["resilience_score"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.space_state:
            try:
                space_assess = self.space.assess_orbital_environment(self.state.space_state)
                result["orbital_environment"] = {
                    "collision_probability": space_assess["collision_probability"],
                    "kessler_syndrome_risk": space_assess["kessler_syndrome_risk"],
                    "kessler_status": space_assess["kessler_status"],
                    "leo_congestion": space_assess["leo_congestion"],
                    "orbital_environment": space_assess["orbital_environment"],
                    "sustainable_orbit": space_assess["sustainable_orbit"],
                }
                space_pred = self.space.predict(years_ahead=5, state=self.state.space_state)
                result["space_future"] = {
                    "future_kessler_risk": space_pred["future_kessler_risk"],
                    "future_active_satellites": space_pred["future_active_satellites"],
                    "future_commercial_revenue_billion": space_pred["future_commercial_revenue_billion"],
                    "orbital_sustainability_trend": space_pred["orbital_sustainability_trend"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.mental_health_state:
            try:
                mh_assess = self.mental_health.assess_population_health(self.state.mental_health_state)
                result["mental_health"] = {
                    "overall_burden": mh_assess["overall_burden"],
                    "burden_level": mh_assess["burden_level"],
                    "system_capacity": mh_assess["system_capacity"],
                    "capacity_level": mh_assess["capacity_level"],
                    "treatment_access_rate": mh_assess["treatment_access_rate"],
                    "youth_mental_health_risk": mh_assess["youth_mental_health_risk"],
                    "workplace_impact_score": mh_assess["workplace_impact_score"],
                    "digital_wellbeing_risk": mh_assess["digital_wellbeing_risk"],
                    "population_affected_millions": mh_assess["population_affected_millions"],
                }
                mh_pred = self.mental_health.predict(years_ahead=5, state=self.state.mental_health_state)
                result["mental_health_future"] = {
                    "future_depression_prevalence": mh_pred["future_depression_prevalence"],
                    "future_anxiety_prevalence": mh_pred["future_anxiety_prevalence"],
                    "future_suicide_rate": mh_pred["future_suicide_rate"],
                    "future_treatment_gap": mh_pred["future_treatment_gap"],
                    "trajectory": mh_pred["trajectory"],
                    "affected_population_millions": mh_pred["affected_population_millions"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.housing_state:
            try:
                housing_assess = self.housing.assess_market(self.state.housing_state)
                result["housing_market"] = {
                    "market_health_score": housing_assess["market_health_score"],
                    "health_label": housing_assess["health_label"],
                    "affordability_index": housing_assess["affordability_index"],
                    "price_to_income_ratio": housing_assess["price_to_income_ratio"],
                    "monthly_payment_burden": housing_assess["monthly_payment_burden"],
                    "rental_burden": housing_assess["rental_burden"],
                    "bubble_risk": housing_assess["bubble_risk"],
                    "housing_supply_pressure": housing_assess["housing_supply_pressure"],
                    "homelessness_impact_score": housing_assess["homelessness_impact_score"],
                }
                housing_pred = self.housing.predict(years_ahead=5, state=self.state.housing_state)
                result["housing_future"] = {
                    "future_price_to_income": housing_pred["future_price_to_income"],
                    "future_affordability": housing_pred["future_affordability"],
                    "future_bubble_risk": housing_pred["future_bubble_risk"],
                    "future_homelessness_per_100k": housing_pred["future_homelessness_per_100k"],
                    "market_outlook": housing_pred["market_outlook"],
                    "affordability_trend": housing_pred["affordability_trend"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.seismic_state:
            try:
                seismic_assess = self.seismic.assess_seismic_hazard(self.state.seismic_state)
                result["seismic_hazard"] = {
                    "composite_hazard": seismic_assess["composite_hazard"],
                    "hazard_level": seismic_assess["hazard_level"],
                    "m7_probability_30d": seismic_assess["m7_probability_30d"],
                    "m8_probability_365d": seismic_assess["m8_probability_365d"],
                    "precursor_score": seismic_assess["precursor_score"],
                    "volcanic_risk": seismic_assess["volcanic_risk"],
                }
                seismic_pred = self.seismic.predict(years_ahead=5, state=self.state.seismic_state)
                result["seismic_future"] = {
                    "future_fault_stress": seismic_pred["future_fault_stress"],
                    "seismic_regime": seismic_pred["seismic_regime"],
                    "m8_probability_5yr": seismic_pred["m8_probability_5yr"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.solar_state:
            try:
                solar_assess = self.solar.assess_solar_weather(self.state.solar_state)
                result["solar_weather"] = {
                    "composite_risk": solar_assess["composite_risk"],
                    "risk_level": solar_assess["risk_level"],
                    "x_flare_probability": solar_assess["x_flare_probability"],
                    "grid_infrastructure_risk": solar_assess["grid_infrastructure_risk"],
                    "satellite_risk": solar_assess["satellite_risk"],
                    "gps_degradation": solar_assess["gps_degradation"],
                }
                solar_pred = self.solar.predict(years_ahead=5, state=self.state.solar_state)
                result["solar_future"] = {
                    "activity_regime": solar_pred["activity_regime"],
                    "carrington_probability": solar_pred["carrington_probability"],
                    "future_sunspot_number": solar_pred["future_sunspot_number"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.oceanic_state:
            try:
                ocean_assess = self.oceanic.assess_ocean_state(self.state.oceanic_state)
                result["oceanic_state"] = {
                    "composite_ocean_risk": ocean_assess["composite_ocean_risk"],
                    "risk_level": ocean_assess["risk_level"],
                    "enso_phase": ocean_assess["enso_phase"],
                    "enso_strength": ocean_assess["enso_strength"],
                    "expected_hurricane_season": ocean_assess["expected_hurricane_season"],
                    "thermohaline_circulation_risk": ocean_assess["thermohaline_circulation_risk"],
                    "arctic_ice_risk": ocean_assess["arctic_ice_risk"],
                    "ocean_acidification_ph": ocean_assess["ocean_acidification_ph"],
                    "coral_bleaching_risk": ocean_assess["coral_bleaching_risk"],
                    "fisheries_stress": ocean_assess["fisheries_stress"],
                }
                ocean_pred = self.oceanic.predict(years_ahead=5, state=self.state.oceanic_state)
                result["oceanic_future"] = {
                    "future_sst_anomaly_c": ocean_pred["future_sst_anomaly_c"],
                    "amoc_collapse_risk": ocean_pred["amoc_collapse_risk"],
                    "ice_free_summer_probability": ocean_pred["ice_free_summer_probability"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.neo_state:
            try:
                neo_assess = self.neo.assess_impact_risk(self.state.neo_state)
                result["neo_impact_risk"] = {
                    "composite_risk": neo_assess["composite_risk"],
                    "risk_level": neo_assess["risk_level"],
                    "detection_gap_pct": neo_assess["detection_gap_pct"],
                    "city_killer_probability_50yr": neo_assess["city_killer_probability_50yr"],
                    "deflection_readiness": neo_assess["deflection_readiness"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.atmospheric_state:
            try:
                atm_assess = self.atmospheric.assess_atmospheric_state(self.state.atmospheric_state)
                result["atmospheric_risk"] = {
                    "composite_atmospheric_risk": atm_assess["composite_atmospheric_risk"],
                    "risk_level": atm_assess["risk_level"],
                    "circulation_disruption": atm_assess["circulation_disruption"],
                    "ssw_risk_score": atm_assess["ssw_risk_score"],
                    "ozone_deficit_pct": atm_assess["ozone_deficit_pct"],
                    "hadley_cell_expansion": atm_assess["hadley_cell_expansion"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.ecological_state:
            try:
                eco_assess = self.ecological.assess_ecosystem(self.state.ecological_state)
                result["ecosystem_health"] = {
                    "ecosystem_health_index": eco_assess["ecosystem_health_index"],
                    "health_level": eco_assess["health_level"],
                    "trophic_integrity": eco_assess["trophic_integrity"],
                    "resilience_score": eco_assess["resilience_score"],
                    "cascade_potential": eco_assess["cascade_potential"],
                    "stress_score": eco_assess["stress_score"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.paleoclimate_state:
            try:
                paleo_assess = self.paleoclimatology.assess_milankovitch(self.state.paleoclimate_state)
                result["paleoclimate_forcing"] = {
                    "insolation_variability": paleo_assess["insolation_variability"],
                    "orbital_phase": paleo_assess["orbital_phase"],
                    "glaciation_risk": paleo_assess["glaciation_risk"],
                    "deglaciation_risk": paleo_assess["deglaciation_risk"],
                    "cycle_forcing": paleo_assess["cycle_forcing"],
                    "radiative_forcing_wm2": paleo_assess["radiative_forcing_wm2"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.cosmic_radiation_state:
            try:
                cr_assess = self.cosmic_radiation.assess_radiation(self.state.cosmic_radiation_state)
                result["cosmic_radiation_risk"] = {
                    "composite_risk": cr_assess["composite_risk"],
                    "risk_level": cr_assess["risk_level"],
                    "effective_gcr_flux": cr_assess["effective_gcr_flux"],
                    "svensmark_cloud_effect": cr_assess["svensmark_cloud_effect"],
                    "health_risk": cr_assess["health_risk"],
                    "aviation_risk": cr_assess["aviation_risk"],
                    "electronics_seu_risk": cr_assess["electronics_seu_risk"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.gravitational_state:
            try:
                grav_assess = self.gravitational.assess_tidal_forces(self.state.gravitational_state)
                result["gravitational_tidal"] = {
                    "composite_tidal_index": grav_assess["composite_tidal_index"],
                    "level": grav_assess["level"],
                    "effective_tidal_force": grav_assess["effective_tidal_force"],
                    "seismic_triggering_probability": grav_assess["seismic_triggering_probability"],
                    "volcanic_triggering_probability": grav_assess["volcanic_triggering_probability"],
                    "coastal_flood_risk": grav_assess["coastal_flood_risk"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.compound_hazard_state:
            try:
                compound_assess = self.compound_hazard.assess_compound_risk(self.state.compound_hazard_state)
                result["compound_hazard"] = {
                    "compound_hazard_index": compound_assess["compound_hazard_index"],
                    "hazard_level": compound_assess["hazard_level"],
                    "resilience_score": compound_assess["resilience_score"],
                    "hazard_burden": compound_assess["hazard_burden"],
                    "top_hazards": compound_assess["top_hazards"],
                    "worst_pair_probability": compound_assess["worst_pair_probability"],
                    "worst_triple_probability": compound_assess["worst_triple_probability"],
                    "systemic_collapse_probability": compound_assess["systemic_collapse_probability"],
                    "recommendations": compound_assess["recommendations"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        domain_scores_for_world = {}
        if result.get("cycle_diagnosis"):
            domain_scores_for_world["economic"] = max(0, 1.0 - result["cycle_diagnosis"].get("bubble_risk_score", 0.5))
        if result.get("geopolitical_risk"):
            domain_scores_for_world["geopolitical"] = max(0, 1.0 - result["geopolitical_risk"].get("overall_risk", 0.5))
        if result.get("climate_tipping_points") or result.get("sea_level_projection"):
            domain_scores_for_world["climate"] = max(0, 1.0 - self.state.climate_state.global_temp_anomaly_c / 4.0) if self.state.climate_state else 0.5
        if result.get("sentiment"):
            domain_scores_for_world["sentiment"] = result["sentiment"].get("composite_index", 0.5)
        if result.get("epidemiology"):
            domain_scores_for_world["epidemiology"] = max(0, 1.0 - result["epidemiology"].get("r0", 0.5) / 5.0)
        if result.get("technology"):
            domain_scores_for_world["technology"] = result["technology"].get("current_adoption", 0.15)
        if result.get("social"):
            domain_scores_for_world["social"] = max(0, 1.0 - result["social"].get("unrest_probability", 0.3))
        if result.get("energy"):
            domain_scores_for_world["energy"] = result["energy"].get("security_index", 0.5)
        if result.get("commodities"):
            domain_scores_for_world["commodities"] = max(0, 1.0 - result.get("commodities", {}).get("avg_scarcity_score", 0.3))
        if result.get("infrastructure_resilience"):
            domain_scores_for_world["infrastructure"] = result["infrastructure_resilience"].get("resilience_index", 0.5)
        if result.get("supply_chain_risk"):
            domain_scores_for_world["supply_chain"] = max(0, 1.0 - result["supply_chain_risk"].get("overall_risk", 0.3))
        if result.get("cyber_threat"):
            domain_scores_for_world["cyber"] = max(0, 1.0 - result["cyber_threat"].get("overall_risk_score", 0.4))
        if result.get("water_security"):
            domain_scores_for_world["water"] = max(0, 1.0 - result["water_security"].get("overall_risk", 0.3))
        if result.get("food_security"):
            domain_scores_for_world["food"] = max(0, 1.0 - result["food_security"].get("overall_risk", 0.3))
        if result.get("political_stability"):
            domain_scores_for_world["political"] = result["political_stability"].get("overall_stability", 0.5)
        if result.get("biodiversity_health"):
            domain_scores_for_world["biodiversity"] = result["biodiversity_health"].get("biodiversity_index", 0.5)
        if result.get("pandemic_preparedness"):
            domain_scores_for_world["pandemic"] = result["pandemic_preparedness"].get("overall_preparedness", 0.5)
        if result.get("orbital_environment"):
            domain_scores_for_world["space"] = max(0, 1.0 - result["orbital_environment"].get("collision_probability", 0.3))
        if result.get("mental_health"):
            domain_scores_for_world["mental_health"] = max(0, 1.0 - result["mental_health"].get("overall_burden", 0.3))
        if result.get("housing_market"):
            domain_scores_for_world["housing"] = result["housing_market"].get("market_health_score", 0.5)
        if result.get("seismic_hazard"):
            domain_scores_for_world["seismic"] = max(0, 1.0 - result["seismic_hazard"].get("composite_hazard", 0.3))
        if result.get("solar_weather"):
            domain_scores_for_world["solar"] = max(0, 1.0 - result["solar_weather"].get("composite_risk", 0.2))
        if result.get("oceanic_state"):
            domain_scores_for_world["oceanic"] = max(0, 1.0 - result["oceanic_state"].get("composite_ocean_risk", 0.3))
        if result.get("neo_impact_risk"):
            domain_scores_for_world["neo"] = max(0, 1.0 - result["neo_impact_risk"].get("composite_risk", 0.1))
        if result.get("atmospheric_risk"):
            domain_scores_for_world["atmospheric"] = max(0, 1.0 - result["atmospheric_risk"].get("composite_atmospheric_risk", 0.3))
        if result.get("ecosystem_health"):
            domain_scores_for_world["ecological"] = result["ecosystem_health"].get("ecosystem_health_index", 0.5)
        if result.get("paleoclimate_forcing"):
            domain_scores_for_world["paleoclimate"] = max(0, 1.0 - result["paleoclimate_forcing"].get("radiative_forcing_wm2", 2.0) / 6.0)
        if result.get("cosmic_radiation_risk"):
            domain_scores_for_world["cosmic_radiation"] = max(0, 1.0 - result["cosmic_radiation_risk"].get("composite_risk", 0.3))
        if result.get("gravitational_tidal"):
            domain_scores_for_world["gravitational"] = max(0, 1.0 - result["gravitational_tidal"].get("composite_tidal_index", 0.3))

        if self.state.fate_state:
            domain_scores_for_world["fate"] = self.state.fate_state.social_cohesion_index

        if len(domain_scores_for_world) >= 3:
            try:
                world_assess = self.world_state.assess_world_state(domain_scores_for_world)
                result["world_state"] = world_assess
                world_radar = self.world_state.domain_radar(domain_scores_for_world)
                result["world_state_radar"] = world_radar
                world_traj = self.world_state.stability_trajectory(domain_scores_for_world, years=50, n_simulations=200)
                result["world_state_trajectory"] = world_traj
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if len(domain_scores_for_world) >= 3:
            try:
                prescience = self.prescience.predict_prescience(domain_scores_for_world, current_temp_anomaly=self.state.climate_state.global_temp_anomaly_c if self.state.climate_state else 1.2)
                result["prescience"] = prescience
                prescience_events = self.prescience.forecast_events(domain_scores_for_world, years=10, n_simulations=500)
                result["prescience_events"] = prescience_events
                prescience_tipping = self.prescience.assess_tipping_points(domain_scores_for_world, current_temp_anomaly=self.state.climate_state.global_temp_anomaly_c if self.state.climate_state else 1.2)
                result["prescience_tipping_points"] = prescience_tipping
                prescience_warnings = self.prescience.generate_early_warnings(domain_scores_for_world)
                result["prescience_warnings"] = prescience_warnings
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.fate_state:
            try:
                fate_traj = self.fate.assess_civilizational_trajectory(self.state.fate_state)
                result["fate_trajectory"] = {
                    "trajectory_score": fate_traj["trajectory_score"],
                    "trajectory_label": fate_traj["trajectory_label"],
                    "kardashev_scale": fate_traj["kardashev_scale_current"],
                    "energy_progress_to_type_i_pct": fate_traj["energy_progress_to_type_i_pct"],
                    "technology_capacity": fate_traj["technology_capacity"],
                    "social_capacity": fate_traj["social_capacity"],
                    "existential_risk_composite": fate_traj["existential_risk_composite"],
                    "fate_commitment": fate_traj["fate_commitment"],
                }
                fate_thermo = self.fate.assess_thermodynamic_fate(self.state.fate_state)
                result["fate_thermodynamic"] = {
                    "thermodynamic_pressure": fate_thermo["thermodynamic_pressure"],
                    "pressure_label": fate_thermo["pressure_label"],
                    "eroi_current": fate_thermo["eroi_current"],
                    "carrying_capacity_usage": fate_thermo["carrying_capacity_usage"],
                    "overshoot": fate_thermo["overshoot"],
                    "n_active_constraints": fate_thermo["n_active_constraints"],
                }
                fate_depletion = self.fate.assess_resource_depletion(self.state.fate_state)
                result["fate_depletion"] = {
                    "overall_depletion_pressure": fate_depletion["overall_depletion_pressure"],
                    "n_critical_resources": fate_depletion["n_critical_resources"],
                    "critical_resources": fate_depletion["critical_resources"],
                    "n_substitute_gap": fate_depletion["n_substitute_gap"],
                }
                fate_commitment = self.fate.assess_deterministic_commitments(self.state.fate_state)
                result["fate_commitments"] = {
                    "n_commitments": fate_commitment["n_commitments"],
                    "avg_irreversibility": fate_commitment["avg_irreversibility"],
                    "n_high_irreversibility": fate_commitment["n_high_irreversibility"],
                    "high_irreversibility_commitments": fate_commitment["high_irreversibility_commitments"],
                    "fate_coupling": fate_commitment["fate_coupling"],
                }
                fate_cosmo = self.fate.assess_cosmological_timeline(self.state.fate_state)
                result["fate_cosmological"] = {
                    "near_term_fates": fate_cosmo["near_term_fates"],
                    "existential_risks": fate_cosmo["existential_risks"],
                    "long_term_trajectory": fate_cosmo["long_term_trajectory"],
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.personal_fate_state:
            try:
                personal_fate = self.fate.predict_personal_fate(self.state.personal_fate_state)
                result["personal_fate"] = {
                    "fate_type": personal_fate["fate_type"],
                    "total_determinism": personal_fate["total_determinism"],
                    "agency_score": personal_fate["agency_score"],
                    "immutable_pct": personal_fate["immutable_pct"],
                    "modifiable_pct": personal_fate["modifiable_pct"],
                    "highest_leverage_actions": personal_fate.get("highest_leverage_actions", []),
                }
                personal_destiny = personal_fate.get("personal_destiny", {})
                result["personal_destiny"] = {
                    "destiny_label": personal_destiny.get("destiny_label", ""),
                    "genetic_determinism": personal_destiny.get("genetic_determinism", 0),
                    "structural_determinism": personal_destiny.get("structural_determinism", 0),
                    "behavioral_determinism": personal_destiny.get("behavioral_determinism", 0),
                    "epigenetic_determinism": personal_destiny.get("epigenetic_determinism", 0),
                    "adjusted_life_expectancy": personal_destiny.get("adjusted_life_expectancy", 73),
                }
                mortality = personal_fate.get("mortality", {})
                result["personal_mortality"] = {
                    "life_expectancy_adjusted": mortality.get("life_expectancy_adjusted", 73),
                    "biological_age": mortality.get("biological_age", 35),
                    "healthspan_estimate": mortality.get("healthspan_estimate", 58),
                    "modifiable_years_gained": mortality.get("modifiable_years_gained", 0),
                    "cause_probabilities": mortality.get("cause_probabilities", {}),
                }
                habits = personal_fate.get("habit_destiny", {})
                result["habit_destiny"] = {
                    "habit_inertia": habits.get("habit_inertia", 0),
                    "inertia_label": habits.get("inertia_label", ""),
                    "change_probability": habits.get("change_probability", 0),
                    "annual_habit_reversion": habits.get("annual_habit_reversion", 0),
                    "compounding_loops": habits.get("compounding_loops", []),
                }
                socioeconomic = personal_fate.get("socioeconomic_path", {})
                result["socioeconomic_path"] = {
                    "origin_trap_index": socioeconomic.get("origin_trap_index", 0),
                    "path_label": socioeconomic.get("path_label", ""),
                    "mobility_potential": socioeconomic.get("mobility_potential", 0),
                    "probability_of_class_escape": socioeconomic.get("probability_of_class_escape", 0),
                    "probability_of_class_decline": socioeconomic.get("probability_of_class_decline", 0),
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        if self.state.personal_life_profile:
            try:
                career = self.personal_life.predict_career_trajectory(self.state.personal_life_profile, years=20)
                result["personal_life_career"] = {
                    "current_stage": career.get("current_stage", ""),
                    "base_growth_rate": career.get("base_growth_rate", 0),
                    "current_income": career.get("current_income", 0),
                    "projected_peak_income": career.get("projected_peak_income", 0),
                    "years_to_peak": career.get("years_to_peak", 0),
                    "retirement_income_estimate": career.get("retirement_income_estimate", 0),
                    "lifetime_earnings_estimate": career.get("lifetime_earnings_estimate", 0),
                }
                health = self.personal_life.predict_health_trajectory(self.state.personal_life_profile, years=30)
                result["personal_life_health"] = {
                    "life_expectancy_adjusted": health.get("life_expectancy_adjusted", 79),
                    "healthspan_estimate": health.get("healthspan_estimate", 63),
                    "current_bmi": health.get("current_bmi", 24.4),
                    "bmi_category": health.get("bmi_category", "normal"),
                    "top_3_risks": health.get("top_3_risks", []),
                    "modifiable_factors": health.get("modifiable_factors", {}),
                }
                financial = self.personal_life.predict_financial_trajectory(self.state.personal_life_profile, years=30)
                result["personal_life_financial"] = {
                    "current_net_worth": financial.get("current_net_worth", 0),
                    "current_debt": financial.get("current_debt", 0),
                    "savings_rate": financial.get("savings_rate", 0),
                    "projected_net_worth": financial.get("projected_net_worth", {}),
                    "retirement_target": financial.get("retirement_target", 0),
                    "financial_risks": financial.get("financial_risks", []),
                    "recommendations": financial.get("recommendations", []),
                }
                relationship = self.personal_life.predict_relationship_trajectory(self.state.personal_life_profile, years=20)
                result["personal_life_relationship"] = {
                    "current_marital_status": relationship.get("current_marital_status", ""),
                    "relationship_quality": relationship.get("relationship_quality", 0),
                    "close_friends": relationship.get("close_friends", 0),
                    "loneliness_risk": relationship.get("loneliness_risk", 0),
                    "social_support_adequacy": relationship.get("social_support_adequacy", ""),
                    "recommendations": relationship.get("recommendations", []),
                }
                comprehensive = self.personal_life.predict_comprehensive_future(self.state.personal_life_profile, years=20)
                result["personal_life_comprehensive"] = {
                    "overall_outlook_score": comprehensive.get("overall_outlook_score", 0.5),
                    "overall_outlook_label": comprehensive.get("overall_outlook_label", ""),
                    "key_concerns": comprehensive.get("key_concerns", []),
                    "top_opportunities": comprehensive.get("top_opportunities", []),
                }
                timeline = self.personal_life.predict_personal_timeline(self.state.personal_life_profile, years=20)
                result["personal_life_timeline"] = {
                    "top_5_likely_events": timeline.get("top_5_likely_events", []),
                    "total_event_burden": timeline.get("total_event_burden", 0),
                }
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                _errors.append(str(e))
                pass

        active_domains = []
        if self.state.economic_state:
            active_domains.append(("economic", self.state.economic_state))
        if self.state.geopolitical_state:
            active_domains.append(("geopolitical", self.state.geopolitical_state))
        if self.state.climate_state:
            active_domains.append(("climate", self.state.climate_state))
        if self.state.sentiment_data:
            active_domains.append(("sentiment", self.state.sentiment_data))
        if self.state.contagion_state:
            active_domains.append(("contagion", self.state.contagion_state))
        if self.state.demographics_state:
            active_domains.append(("demographics", self.state.demographics_state))
        if self.state.epidemiology_state:
            active_domains.append(("epidemiology", self.state.epidemiology_state))
        if self.state.technology_state:
            active_domains.append(("technology", self.state.technology_state))
        if self.state.social_state:
            active_domains.append(("social", self.state.social_state))
        if self.state.energy_state:
            active_domains.append(("energy", self.state.energy_state))
        if self.state.commodities_state:
            active_domains.append(("commodities", self.state.commodities_state))
        if self.state.infrastructure_state:
            active_domains.append(("infrastructure", self.state.infrastructure_state))
        if self.state.supply_chain_state:
            active_domains.append(("supply_chain", self.state.supply_chain_state))
        if self.state.cyber_state:
            active_domains.append(("cyber", self.state.cyber_state))
        if self.state.water_state:
            active_domains.append(("water", self.state.water_state))
        if self.state.food_state:
            active_domains.append(("food", self.state.food_state))
        if self.state.political_state:
            active_domains.append(("political", self.state.political_state))
        if self.state.biodiversity_state:
            active_domains.append(("biodiversity", self.state.biodiversity_state))
        if self.state.pandemic_state:
            active_domains.append(("pandemic", self.state.pandemic_state))
        if self.state.space_state:
            active_domains.append(("space", self.state.space_state))
        if self.state.mental_health_state:
            active_domains.append(("mental_health", self.state.mental_health_state))
        if self.state.housing_state:
            active_domains.append(("housing", self.state.housing_state))
        if self.state.seismic_state:
            active_domains.append(("seismic", self.state.seismic_state))
        if self.state.solar_state:
            active_domains.append(("solar", self.state.solar_state))
        if self.state.oceanic_state:
            active_domains.append(("oceanic", self.state.oceanic_state))
        if self.state.neo_state:
            active_domains.append(("neo", self.state.neo_state))
        if self.state.atmospheric_state:
            active_domains.append(("atmospheric", self.state.atmospheric_state))
        if self.state.ecological_state:
            active_domains.append(("ecological", self.state.ecological_state))
        if self.state.paleoclimate_state:
            active_domains.append(("paleoclimate", self.state.paleoclimate_state))
        if self.state.cosmic_radiation_state:
            active_domains.append(("cosmic_radiation", self.state.cosmic_radiation_state))
        if self.state.gravitational_state:
            active_domains.append(("gravitational", self.state.gravitational_state))
        if self.state.compound_hazard_state:
            active_domains.append(("compound_hazard", self.state.compound_hazard_state))

        if len(active_domains) >= 2:
            interactions = []
            if self.state.geopolitical_state and self.state.sentiment_data:
                geo_risk_raw = (self.state.geopolitical_state.us_china_tension + self.state.geopolitical_state.russia_west_tension) / 2
                sent_comp = self.sentiment.compute_composite_index(self.state.sentiment_data)
                if geo_risk_raw > 0.5 and sent_comp.get("composite_index", 0.5) < 0.4:
                    interactions.append({
                        "type": "geopolitical_sentiment_reinforcement",
                        "description": "High geopolitical risk compounds market fear (contrarian buy signal)",
                        "severity": "high",
                    })
                elif geo_risk_raw > 0.5 and sent_comp.get("composite_index", 0.5) > 0.6:
                    interactions.append({
                        "type": "geopolitical_sentiment_divergence",
                        "description": "High geopolitical risk but market complacency (risk of abrupt correction)",
                        "severity": "medium",
                    })

            if self.state.climate_state and self.state.geopolitical_state:
                if self.state.climate_state.global_temp_anomaly_c > 1.5 and self.state.geopolitical_state.energy_geopolitics > 0.5:
                    interactions.append({
                        "type": "climate_geopolitical_feedback",
                        "description": "Climate stress amplifying resource competition and geopolitical tension",
                        "severity": "high",
                    })

            if self.state.contagion_state and self.state.sentiment_data:
                if self.state.contagion_state.financial_systemic_risk > 0.5:
                    sent_comp = self.sentiment.compute_composite_index(self.state.sentiment_data)
                    if sent_comp.get("composite_index", 0.5) > 0.6:
                        interactions.append({
                            "type": "contagion_sentiment_amber",
                            "description": "Elevated systemic risk with market optimism creates vulnerability to contagion",
                            "severity": "medium",
                        })

            if self.state.demographics_state and self.state.economic_state:
                if self.state.demographics_state.elderly_pct > 0.2 and self.state.economic_state.debt_to_gdp > 1.5:
                    interactions.append({
                        "type": "demographics_debt_trap",
                        "description": "Aging population with high debt burden limits fiscal response capacity",
                        "severity": "medium",
                    })

            if self.state.climate_state and self.state.demographics_state:
                if self.state.demographics_state.total_population_m > 8000 and self.state.climate_state.global_temp_anomaly_c > 1.5:
                    interactions.append({
                        "type": "climate_demographic_pressure",
                        "description": "Large population under climate stress increases migration and resource conflict risk",
                        "severity": "medium",
                    })

            if self.state.contagion_state and self.state.geopolitical_state:
                if self.state.contagion_state.financial_systemic_risk > 0.4 and self.state.geopolitical_state.trade_protectionism > 0.5:
                    interactions.append({
                        "type": "trade_contagion_spiral",
                        "description": "Trade fragmentation amplifies financial contagion pathways (supply chain + capital flow)",
                        "severity": "high",
                    })

            if self.state.epidemiology_state and self.state.social_state:
                epi_params = self.state.epidemiology_state
                unrest_prob = result.get("social", {}).get("unrest_probability", 0.3)
                if unrest_prob > 0.4 and epi_params.get("mu", 0.005) > 0.01:
                    interactions.append({
                        "type": "health_social_instability",
                        "description": "High social unrest combined with elevated mortality risk creates compound crisis (health system collapse + civil disorder)",
                        "severity": "high",
                    })

            if self.state.technology_state and self.state.social_state:
                tech_params = self.state.technology_state
                polarization = result.get("social", {}).get("polarization_index", 0.4)
                if tech_params.get("current_adoption", 0.15) > 0.3 and polarization > 0.5:
                    interactions.append({
                        "type": "technology_polarization",
                        "description": "Rapid technology adoption exacerbating social polarization (displacement + information asymmetry)",
                        "severity": "medium",
                    })

            if self.state.epidemiology_state and self.state.geopolitical_state:
                if self.state.geopolitical_state.us_china_tension > 0.5:
                    interactions.append({
                        "type": "geopolitical_health_vulnerability",
                        "description": "Geopolitical tensions reducing pandemic cooperation and supply chain resilience for medical supplies",
                        "severity": "medium",
                    })

            if self.state.technology_state and self.state.geopolitical_state:
                if self.state.geopolitical_state.us_china_tension > 0.5 and self.state.geopolitical_state.trade_protectionism > 0.3:
                    interactions.append({
                        "type": "tech_decoupling",
                        "description": "Great power competition accelerating technology decoupling and reshaping global innovation networks",
                        "severity": "medium",
                    })

            if self.state.epidemiology_state and self.state.demographics_state:
                if self.state.demographics_state and self.state.demographics_state.elderly_pct > 0.2:
                    interactions.append({
                         "type": "aging_health_risk",
                         "description": "Aging population dramatically increasing pandemic vulnerability and healthcare system strain",
                         "severity": "high",
                     })

            if self.state.energy_state and self.state.geopolitical_state:
                geo_energy = (self.state.geopolitical_state.us_china_tension + self.state.geopolitical_state.energy_geopolitics) / 2
                security_idx = result.get("energy", {}).get("security_index", 0.5)
                if geo_energy > 0.5 and security_idx < 0.5:
                    interactions.append({
                        "type": "geopolitical_energy_insecurity",
                        "description": "High geopolitical tensions combining with energy security vulnerabilities creating supply disruption risk",
                        "severity": "high",
                    })

            if self.state.commodities_state and self.state.climate_state:
                comm_stress = self.state.commodities_state.get("supply_chain_stress", 0.3)
                climate_temp = self.state.climate_state.global_temp_anomaly_c if self.state.climate_state else 1.0
                if comm_stress > 0.5 and climate_temp > 1.5:
                    interactions.append({
                        "type": "climate_commodity_stress",
                        "description": "Climate disruptions amplifying critical mineral supply chain vulnerabilities",
                        "severity": "medium",
                    })

            if self.state.energy_state and self.state.technology_state:
                renewable_share = self.state.energy_state.get("renewable_share", 20.0)
                tech_adoption = self.state.technology_state.get("current_adoption", 0.15)
                if renewable_share < 25 and tech_adoption > 0.2:
                    interactions.append({
                        "type": "tech_energy_transition_acceleration",
                        "description": "Rapid technology adoption accelerating energy transition beyond current renewable capacity",
                        "severity": "medium",
                    })

            if self.state.energy_state and self.state.commodities_state:
                energy_stress = self.state.commodities_state.get("supply_chain_stress", 0.3)
                geo_risk_comm = self.state.commodities_state.get("geopolitical_risk", 0.4)
                security_idx = result.get("energy", {}).get("security_index", 0.5)
                if energy_stress > 0.4 and security_idx < 0.6:
                    interactions.append({
                        "type": "energy_commodity_nexus",
                        "description": "Supply chain stress in critical minerals amplifying energy security vulnerability (battery metals, rare earths for grid)",
                        "severity": "high",
                    })

            if self.state.energy_state and self.state.epidemiology_state:
                if self.state.epidemiology_state.get("mu", 0.005) > 0.008:
                    interactions.append({
                        "type": "pandemic_energy_demand_shock",
                        "description": "Severe pandemic reducing energy demand and disrupting energy supply chains",
                        "severity": "medium",
                    })

            if self.state.energy_state and self.state.demographics_state:
                if self.state.demographics_state.total_population_m > 8000 and self.state.energy_state.get("demand_growth", 0.02) > 0.015:
                    interactions.append({
                        "type": "demographic_energy_pressure",
                        "description": "Large and growing population creating sustained upward pressure on energy demand",
                        "severity": "medium",
                    })

            if self.state.technology_state and self.state.geopolitical_state:
                tech_params = self.state.technology_state
                if tech_params.get("quantum_readiness", 0) > 0.3 or tech_params.get("current_adoption", 0.15) > 0.3:
                    if self.state.geopolitical_state.cyber_warfare_likelihood > 0.4:
                        interactions.append({
                            "type": "quantum_geopolitical_threat",
                            "description": "Advancing quantum computing capability intersects with elevated cyber warfare risk, accelerating cryptographic arms race",
                            "severity": "high",
                        })

            if self.state.social_state and self.state.epidemiology_state:
                social_params = self.state.social_state
                cascade_risk = social_params.get("cascade_risk", 0)
                if cascade_risk > 0.2 and self.state.epidemiology_state.get("mu", 0.005) > 0.005:
                    interactions.append({
                        "type": "cascade_epidemiology_amplification",
                        "description": "Social cascade dynamics amplifying pandemic compliance failures and misinformation spread",
                        "severity": "medium",
                    })

            if self.state.geopolitical_state and self.state.energy_state:
                geo = self.state.geopolitical_state
                if geo.us_china_tension > 0.6 and geo.middle_east_instability > 0.5:
                    es = self.state.energy_state
                    if es.get("oil_price", 75) > 80:
                        interactions.append({
                            "type": "event_shock_energy_spillover",
                            "description": "Concurrent geopolitical shocks in multiple theaters creating energy supply disruption beyond modelled scenarios",
                            "severity": "high",
                        })

            if self.state.technology_state and self.state.social_state:
                tech_params = self.state.technology_state
                if tech_params.get("current_adoption", 0.15) > 0.25:
                    qr_score = tech_params.get("quantum_readiness", 0)
                    if qr_score > 0.2:
                        unrest = result.get("social", {}).get("unrest_probability", 0.3)
                        if unrest > 0.3:
                            interactions.append({
                                "type": "quantum_social_disruption",
                                "description": "Quantum computing advances coinciding with social instability creating workforce displacement anxiety and technology backlash risk",
                                "severity": "medium",
                            })

            if self.state.climate_state and self.state.demographics_state:
                migration_pressure = result.get("migration_pressure", {}).get("net_pressure", 0)
                if migration_pressure > 0.4 and self.state.climate_state.global_temp_anomaly_c > 1.5:
                    interactions.append({
                        "type": "climate_migration_escalation",
                        "description": "Climate stress amplifying migration pressure beyond economic drivers, creating climate refugee corridors and border crisis risk",
                        "severity": "high",
                    })

            if self.state.sentiment_data and self.state.commodities_state:
                contrarian_score = result.get("advanced_contrarian", {}).get("composite_contrarian_score", 0)
                comm_stress = self.state.commodities_state.get("supply_chain_stress", 0.3)
                if contrarian_score < -0.3 and comm_stress > 0.5:
                    interactions.append({
                        "type": "contrarian_commodity_crisis",
                        "description": "Bearish contrarian signals coinciding with commodity supply chain stress — potential stagflation setup with inflationary supply shock in risk-off environment",
                        "severity": "high",
                    })

            if self.state.geopolitical_state and self.state.demographics_state:
                migration_level = result.get("migration_pressure", {}).get("overall_level", "low")
                if migration_level in ("high", "critical") and self.state.geopolitical_state.trade_protectionism > 0.5:
                    interactions.append({
                        "type": "migration_geopolitical_nexus",
                        "description": "High migration pressure combining with trade protectionism fueling nationalist populism and border conflict escalation",
                        "severity": "high",
                    })

            if self.state.technology_state and self.state.energy_state:
                qr_data = result.get("quantum_readiness", {})
                qr_risk = qr_data.get("mitigation_urgency", "low")
                if qr_risk in ("medium", "high", "critical") and self.state.energy_state.get("renewable_share", 20) < 25:
                        interactions.append({
                            "type": "quantum_energy_infrastructure",
                            "description": "Quantum computing threats to grid cybersecurity while energy infrastructure remains fossil-dependent and less digitally resilient",
                            "severity": "medium",
                        })

            if self.state.infrastructure_state and self.state.climate_state:
                infra = result.get("infrastructure_resilience", {})
                if infra.get("threat_level") in ("high", "critical") and self.state.climate_state.global_temp_anomaly_c > 1.5:
                    interactions.append({
                        "type": "climate_infrastructure_fragility",
                        "description": "Critical infrastructure vulnerabilities amplifying under climate stress — compounding physical risk with low resilience capacity",
                        "severity": "high",
                    })

            if self.state.infrastructure_state and self.state.epidemiology_state:
                infra = result.get("infrastructure_resilience", {})
                if infra.get("weakest_sector") in ("healthcare", "supply_chain") and self.state.epidemiology_state.get("beta", 0.3) > 0.3:
                    interactions.append({
                        "type": "pandemic_infrastructure_cascade",
                        "description": "Pandemic pressure on already-weak healthcare/supply chain infrastructure creating cascade failure risk",
                        "severity": "high",
                    })

            if self.state.infrastructure_state and self.state.geopolitical_state:
                infra = result.get("infrastructure_resilience", {})
                if infra.get("hazard_exposure", 0) > 0.3 and self.state.geopolitical_state.cyber_warfare_likelihood > 0.4:
                    interactions.append({
                        "type": "cyber_infrastructure_geopolitical",
                        "description": "Elevated cyber threat coinciding with high infrastructure hazard exposure creating hybrid attack vulnerability",
                        "severity": "high",
                    })

            if self.state.cyber_state:
                cyber_result = result.get("cyber_threat", {})
                cyber_risk = cyber_result.get("overall_risk_score", 0)
                if self.state.cyber_state.supply_chain_cyber_risk > 0.4 and self.state.supply_chain_state:
                    interactions.append({
                        "type": "cyber_supply_chain_attack",
                        "description": "High cyber supply chain risk amplifying supply chain disruption vulnerability",
                        "severity": "high" if cyber_risk > 0.5 else "moderate",
                    })
                if self.state.cyber_state.ai_threat_capability > 0.4 and self.state.technology_state:
                    interactions.append({
                        "type": "ai_cyber_threat_escalation",
                        "description": "AI-augmented cyber threats accelerating alongside technology adoption creating compound digital risk",
                        "severity": "high" if self.state.cyber_state.ai_threat_capability > 0.6 else "moderate",
                    })
                if self.state.cyber_state.nation_state_activity > 0.5 and self.state.geopolitical_state:
                    interactions.append({
                        "type": "cyber_geopolitical_nexus",
                        "description": "Nation-state cyber activity escalating with geopolitical tensions increasing critical infrastructure targeting",
                        "severity": "high",
                    })

            if self.state.water_state and self.state.food_state:
                water_risk = result.get("water_security", {}).get("overall_risk", 0)
                food_risk = result.get("food_security", {}).get("overall_risk", 0)
                if water_risk > 0.3 and food_risk > 0.3:
                    interactions.append({
                        "type": "water_food_nexus_stress",
                        "description": "Concurrent water and food insecurity creating compound crisis with cascading humanitarian impact",
                        "severity": "high" if water_risk > 0.5 or food_risk > 0.5 else "moderate",
                    })

            if self.state.water_state and self.state.climate_state:
                water_risk = result.get("water_security", {}).get("overall_risk", 0)
                if water_risk > 0.3 and self.state.climate_state.global_temp_anomaly_c > 1.5:
                    interactions.append({
                        "type": "climate_water_compound",
                        "description": "Climate change amplifying water stress through increased drought probability and glacial melt dependency",
                        "severity": "high",
                    })

            if self.state.food_state and self.state.supply_chain_state:
                food_risk = result.get("food_security", {}).get("overall_risk", 0)
                if food_risk > 0.3 and self.state.food_state.supply_chain_disruption > 0.3:
                    interactions.append({
                        "type": "food_supply_chain_fragility",
                        "description": "Food system vulnerability amplified by supply chain disruption creating access crisis",
                        "severity": "moderate",
                    })

            if self.state.political_state and self.state.geopolitical_state:
                pol_stability = result.get("political_stability", {}).get("overall_stability", 0.5)
                if pol_stability < 0.4 and self.state.geopolitical_state.us_china_tension > 0.5:
                    interactions.append({
                        "type": "political_geopolitical_instability",
                        "description": "Domestic political instability amplifying geopolitical tensions and reducing diplomatic effectiveness",
                        "severity": "high",
                    })

            if self.state.political_state and self.state.economic_state:
                pol_stability = result.get("political_stability", {}).get("overall_stability", 0.5)
                if pol_stability < 0.4 and self.state.economic_state.debt_to_gdp > 1.2:
                    interactions.append({
                        "type": "political_economic_crisis",
                        "description": "Political instability compounding debt crisis — reduced fiscal response capacity with social unrest",
                        "severity": "high",
                    })

            if self.state.political_state and self.state.cyber_state:
                coup_risk = result.get("political_stability", {}).get("coup_risk", 0)
                if coup_risk > 0.3 and self.state.cyber_state.nation_state_activity > 0.4:
                    interactions.append({
                        "type": "political_cyber_destabilization",
                        "description": "Political instability creating opportunities for nation-state cyber operations targeting critical infrastructure",
                        "severity": "high",
                    })

            if self.state.biodiversity_state and self.state.climate_state:
                bio_idx = result.get("biodiversity_health", {}).get("biodiversity_index", 0.5)
                if bio_idx < 0.4 and self.state.climate_state.global_temp_anomaly_c > 1.5:
                    interactions.append({
                        "type": "climate_biodiversity_collapse",
                        "description": "Climate change accelerating biodiversity loss toward irreversible tipping points",
                        "severity": "high",
                    })

            if self.state.biodiversity_state and self.state.food_state:
                bio_idx = result.get("biodiversity_health", {}).get("biodiversity_index", 0.5)
                food_risk = result.get("food_security", {}).get("overall_risk", 0)
                if bio_idx < 0.5 and food_risk > 0.4:
                    interactions.append({
                        "type": "biodiversity_food_security_nexus",
                        "description": "Biodiversity loss (pollinators, soil biota) undermining food production resilience and nutrition security",
                        "severity": "high",
                    })

            if self.state.biodiversity_state and self.state.water_state:
                bio_idx = result.get("biodiversity_health", {}).get("biodiversity_index", 0.5)
                water_risk = result.get("water_security", {}).get("overall_risk", 0)
                if bio_idx < 0.5 and water_risk > 0.3:
                    interactions.append({
                        "type": "biodiversity_water_cycle_disruption",
                        "description": "Ecosystem degradation disrupting water purification and cycling services affecting freshwater availability",
                        "severity": "moderate",
                    })

            if self.state.pandemic_state and self.state.political_state:
                pan_prep = result.get("pandemic_preparedness", {}).get("overall_preparedness", 0.5)
                coup_risk = result.get("political_stability", {}).get("coup_risk", 0)
                if pan_prep < 0.4 and coup_risk > 0.3:
                    interactions.append({
                        "type": "pandemic_political_instability",
                        "description": "Low pandemic preparedness coinciding with political instability creating compound governance failure risk",
                        "severity": "high",
                    })

            if self.state.pandemic_state and self.state.biodiversity_state:
                pan_prep = result.get("pandemic_preparedness", {}).get("overall_preparedness", 0.5)
                bio_idx = result.get("biodiversity_health", {}).get("biodiversity_index", 0.5)
                if pan_prep < 0.4 and bio_idx < 0.4:
                    interactions.append({
                        "type": "biodiversity_pandemic_emergence",
                        "description": "Biodiversity loss increasing zoonotic spillover risk while pandemic preparedness remains inadequate",
                        "severity": "high",
                    })

            if self.state.pandemic_state and self.state.supply_chain_state:
                pan_prep = result.get("pandemic_preparedness", {}).get("overall_preparedness", 0.5)
                if pan_prep < 0.4 and self.state.pandemic_state.supply_chain_resilience < 0.4:
                    interactions.append({
                        "type": "pandemic_supply_chain_failure",
                        "description": "Pandemic risk converging with fragile supply chains creating medical supply and essential goods crisis",
                        "severity": "high",
                    })

            if self.state.political_state and self.state.biodiversity_state:
                pol_stability = result.get("political_stability", {}).get("overall_stability", 0.5)
                bio_idx = result.get("biodiversity_health", {}).get("biodiversity_index", 0.5)
                if pol_stability < 0.4 and bio_idx < 0.4:
                    interactions.append({
                        "type": "political_biodiversity_governance_failure",
                        "description": "Political instability undermining environmental governance and conservation enforcement during biodiversity crisis",
                        "severity": "high",
                    })

            if self.state.political_state and self.state.energy_state:
                pol_stability = result.get("political_stability", {}).get("overall_stability", 0.5)
                if pol_stability < 0.4:
                    es = self.state.energy_state
                    if es.get("renewable_share", 20) < 25:
                        interactions.append({
                            "type": "political_energy_policy_instability",
                            "description": "Political instability disrupting energy transition policy continuity and renewable investment certainty",
                            "severity": "moderate",
                        })

            if self.state.biodiversity_state and self.state.epidemiology_state:
                bio_idx = result.get("biodiversity_health", {}).get("biodiversity_index", 0.5)
                if bio_idx < 0.4:
                    epi_beta = self.state.epidemiology_state.get("beta", 0.3)
                    if epi_beta > 0.3:
                        interactions.append({
                            "type": "biodiversity_disease_emergence",
                            "description": "Biodiversity loss increasing zoonotic disease spillover risk while pandemic vulnerability remains elevated",
                            "severity": "high",
                        })

            if self.state.climate_state and self.state.energy_state:
                if self.state.climate_state.global_temp_anomaly_c > 1.5 and self.state.energy_state.get("renewable_share", 20) < 25:
                    interactions.append({
                        "type": "climate_energy_transition_pressure",
                        "description": "Accelerating climate impacts demanding faster energy transition than current renewable deployment allows",
                        "severity": "high",
                    })

            if self.state.climate_state and self.state.food_state:
                if self.state.climate_state.global_temp_anomaly_c > 1.5 and self.state.climate_state.extreme_weather_index > 0.5:
                    food_risk = result.get("food_security", {}).get("overall_risk", 0)
                    if food_risk > 0.3:
                        interactions.append({
                            "type": "climate_food_system_shock",
                            "description": "Climate-driven extreme events creating cascading food system disruptions through drought, flood, and heat stress",
                            "severity": "high",
                        })

            if self.state.demographics_state and self.state.food_state:
                if self.state.demographics_state.total_population_m > 9000 and self.state.demographics_state.urbanization_rate > 0.6:
                    food_risk = result.get("food_security", {}).get("overall_risk", 0)
                    if food_risk > 0.3:
                        interactions.append({
                            "type": "demographics_food_demand_pressure",
                            "description": "Large urbanizing population creating sustained food demand pressure exceeding sustainable agricultural capacity",
                            "severity": "medium",
                        })

            if self.state.energy_state and self.state.water_state:
                if self.state.energy_state.get("renewable_share", 20) < 25:
                    water_risk = result.get("water_security", {}).get("overall_risk", 0)
                    if water_risk > 0.3:
                        interactions.append({
                            "type": "energy_water_nexus",
                            "description": "Fossil-dependent energy production stressing water resources through thermal cooling and hydraulic fracturing water consumption",
                            "severity": "moderate",
                        })

            if self.state.cyber_state and self.state.water_state:
                if self.state.cyber_state.nation_state_activity > 0.4:
                    water_risk = result.get("water_security", {}).get("overall_risk", 0)
                    if water_risk > 0.3:
                        interactions.append({
                            "type": "cyber_water_infrastructure_threat",
                            "description": "Nation-state cyber threats targeting water infrastructure SCADA systems with existing water stress vulnerabilities",
                            "severity": "high",
                        })

            if self.state.cyber_state and self.state.food_state:
                if self.state.cyber_state.supply_chain_cyber_risk > 0.4:
                    food_risk = result.get("food_security", {}).get("overall_risk", 0)
                    if food_risk > 0.3:
                        interactions.append({
                            "type": "cyber_food_supply_attack",
                            "description": "Cyber threats to food supply chain logistics and agricultural technology creating food access disruption",
                            "severity": "high",
                        })

            if self.state.infrastructure_state and self.state.supply_chain_state:
                infra = result.get("infrastructure_resilience", {})
                if infra.get("overall_resilience", 1.0) < 0.5:
                    if self.state.supply_chain_state.concentration_risk > 0.4:
                        interactions.append({
                            "type": "infrastructure_supply_chain_cascade",
                            "description": "Weak infrastructure amplifying supply chain concentration risk — single points of failure across both physical and logistics networks",
                            "severity": "high",
                        })

            if self.state.infrastructure_state and self.state.water_state:
                infra = result.get("infrastructure_resilience", {})
                water_risk = result.get("water_security", {}).get("overall_risk", 0)
                if infra.get("overall_resilience", 1.0) < 0.5 and water_risk > 0.3:
                    interactions.append({
                        "type": "infrastructure_water_failure",
                        "description": "Aging infrastructure vulnerabilities combining with water stress creating compound failure risk in water treatment and distribution",
                        "severity": "high",
                    })

            if self.state.pandemic_state and self.state.epidemiology_state:
                pan_prep = result.get("pandemic_preparedness", {}).get("overall_preparedness", 0.5)
                epi_beta = self.state.epidemiology_state.get("beta", 0.3)
                if pan_prep < 0.4 and epi_beta > 0.3:
                    interactions.append({
                        "type": "pandemic_epidemiology_compound",
                        "description": "Low pandemic preparedness amplifying active epidemiological threat — compound vulnerability to disease emergence and spread",
                        "severity": "high",
                    })

            if self.state.sentiment_data and self.state.energy_state:
                sent_comp = self.sentiment.compute_composite_index(self.state.sentiment_data)
                oil_vol = result.get("energy", {}).get("oil_annualized_volatility", 0)
                if sent_comp.get("composite_index", 0.5) < 0.35 and oil_vol > 0.3:
                    interactions.append({
                        "type": "sentiment_energy_crash_signal",
                        "description": "Deeply negative sentiment coinciding with energy price volatility creating demand destruction and recession spiral risk",
                        "severity": "high",
                    })

            if self.state.sentiment_data and self.state.technology_state:
                sent_comp = self.sentiment.compute_composite_index(self.state.sentiment_data)
                tech_adoption = self.state.technology_state.get("current_adoption", 0.15)
                if sent_comp.get("composite_index", 0.5) > 0.65 and tech_adoption > 0.3:
                    interactions.append({
                        "type": "sentiment_technology_bubble",
                        "description": "Excessive market optimism combined with rapid technology adoption creating speculative bubble risk in tech sector",
                        "severity": "medium",
                    })

            if self.state.political_state and self.state.food_state:
                pol_stability = result.get("political_stability", {}).get("overall_stability", 0.5)
                food_risk = result.get("food_security", {}).get("overall_risk", 0)
                if pol_stability < 0.4 and food_risk > 0.4:
                    interactions.append({
                        "type": "political_food_crisis",
                        "description": "Political instability coinciding with food insecurity creating conditions for social upheaval and regime destabilization",
                        "severity": "high",
                    })

            if self.state.political_state and self.state.water_state:
                pol_stability = result.get("political_stability", {}).get("overall_stability", 0.5)
                water_risk = result.get("water_security", {}).get("overall_risk", 0)
                if pol_stability < 0.4 and water_risk > 0.4:
                    interactions.append({
                        "type": "political_water_conflict",
                        "description": "Political fragility with water scarcity increasing risk of interstate water conflict and internal resource disputes",
                        "severity": "high",
                    })

            if self.state.biodiversity_state and self.state.epidemiology_state:
                bio_idx = result.get("biodiversity_health", {}).get("biodiversity_index", 0.5)
                epi_beta = self.state.epidemiology_state.get("beta", 0.3)
                if bio_idx < 0.3 and epi_beta > 0.2:
                    interactions.append({
                        "type": "biodiversity_pathogen_spillover",
                        "description": "Severe biodiversity loss increasing pathogen spillover events from wildlife habitat disruption to human populations",
                        "severity": "high",
                    })

            if self.state.technology_state and self.state.biodiversity_state:
                tech_adoption = self.state.technology_state.get("current_adoption", 0.15)
                bio_idx = result.get("biodiversity_health", {}).get("biodiversity_index", 0.5)
                if tech_adoption > 0.3 and bio_idx < 0.4:
                    interactions.append({
                        "type": "technology_biodiversity_monitoring",
                        "description": "Advanced technology enabling biodiversity monitoring and conservation while habitat destruction continues — a race between sensing and loss",
                        "severity": "moderate",
                    })

            if self.state.pandemic_state and self.state.demographics_state:
                pan_prep = result.get("pandemic_preparedness", {}).get("overall_preparedness", 0.5)
                if pan_prep < 0.4 and self.state.demographics_state.elderly_pct > 0.15:
                    interactions.append({
                        "type": "pandemic_demographics_vulnerability",
                        "description": "Inadequate pandemic preparedness with aging population creating disproportionate health system burden and mortality risk",
                        "severity": "high",
                    })

            if self.state.climate_state and self.state.pandemic_state:
                if self.state.climate_state.global_temp_anomaly_c > 1.5:
                    pan_prep = result.get("pandemic_preparedness", {}).get("overall_preparedness", 0.5)
                    if pan_prep < 0.5:
                        interactions.append({
                            "type": "climate_pandemic_compound",
                            "description": "Climate change expanding pathogen ranges and vector habitats while pandemic preparedness remains insufficient — compound health-environment crisis",
                            "severity": "high",
                        })

            if self.state.energy_state and self.state.cyber_state:
                if self.state.cyber_state.nation_state_activity > 0.4 and self.state.energy_state.get("grid_stability", 0.7) < 0.6:
                    interactions.append({
                        "type": "energy_cyber_grid_attack",
                        "description": "Nation-state cyber activity targeting energy grid vulnerabilities creating potential for cascading power infrastructure failure",
                        "severity": "high",
                    })

            if self.state.space_state:
                space_assess = result.get("orbital_environment", {})
                if self.state.space_state.kessler_syndrome_risk > 0.3 and self.state.space_state.space_military_index > 0.4:
                    interactions.append({
                        "type": "space_military_debris_cascade",
                        "description": "Elevated Kessler syndrome risk coinciding with military space activity creating debris cascade and ASAT weapon threat",
                        "severity": "high",
                    })
                if self.state.space_state.broadband_satellite_share > 0.1:
                    cyber_result = result.get("cyber_threat", {})
                    if cyber_result.get("overall_risk_score", 0) > 0.3:
                        interactions.append({
                            "type": "space_cyber_satellite_vulnerability",
                            "description": "Growing satellite broadband dependency increasing vulnerability to cyber attacks on space infrastructure",
                            "severity": "moderate",
                        })

            if self.state.mental_health_state:
                mh_burden = result.get("mental_health", {}).get("overall_burden", 0)
                if mh_burden > 0.2 and self.state.economic_state:
                    if self.state.economic_state.debt_to_gdp > 1.2:
                        interactions.append({
                            "type": "mental_health_economic_burden",
                            "description": "High mental health burden compounding economic productivity loss and increasing welfare spending during debt stress",
                            "severity": "high",
                        })
                if mh_burden > 0.2 and self.state.pandemic_state:
                    pan_prep = result.get("pandemic_preparedness", {}).get("overall_preparedness", 0.5)
                    if pan_prep < 0.5:
                        interactions.append({
                            "type": "pandemic_mental_health_crisis",
                            "description": "Pandemic vulnerability compounding mental health crisis — reduced healthcare access during outbreak worsening psychological outcomes",
                            "severity": "high",
                        })
                if mh_burden > 0.2 and self.state.social_state:
                    social_unrest = result.get("social", {}).get("unrest_probability", 0.3)
                    if social_unrest > 0.3:
                        interactions.append({
                            "type": "mental_health_social_unrest",
                            "description": "Deteriorating mental health amplifying social unrest through despair-driven instability and reduced institutional trust",
                            "severity": "moderate",
                        })

            if self.state.housing_state:
                housing_assess = result.get("housing_market", {})
                bubble_risk = housing_assess.get("bubble_risk", 0)
                if bubble_risk > 0.3 and self.state.economic_state:
                    if self.state.economic_state.debt_to_gdp > 1.0:
                        interactions.append({
                            "type": "housing_bubble_debt_cycle",
                            "description": "Housing bubble risk compounding debt cycle vulnerability — property market correction could trigger broader deleveraging",
                            "severity": "high",
                        })
                affordability = housing_assess.get("affordability_index", 0.5)
                if affordability < 0.4 and self.state.demographics_state:
                    if self.state.demographics_state.urbanization_rate > 0.7:
                        interactions.append({
                            "type": "housing_demographics_affordability_crisis",
                            "description": "Rapid urbanization with declining housing affordability creating displacement pressure and homelessness risk",
                            "severity": "high",
                        })
                if affordability < 0.4:
                    mh_burden = result.get("mental_health", {}).get("overall_burden", 0)
                    if mh_burden > 0.2:
                        interactions.append({
                            "type": "housing_mental_health_displacement",
                            "description": "Housing affordability crisis driving mental health deterioration through housing insecurity and displacement stress",
                            "severity": "moderate",
                         })

            if self.state.seismic_state:
                seismic_assess = result.get("seismic_hazard", {})
                if seismic_assess.get("composite_hazard", 0) > 0.5 and self.state.infrastructure_state:
                    interactions.append({
                        "type": "seismic_infrastructure_cascade",
                        "description": "Elevated seismic hazard threatening infrastructure integrity — earthquake-damaged systems cascade risk",
                        "severity": "high",
                    })
                if seismic_assess.get("composite_hazard", 0) > 0.4 and self.state.housing_state:
                    interactions.append({
                        "type": "seismic_housing_vulnerability",
                        "description": "Seismic hazard compounding housing risk — building damage potential in vulnerable structures",
                        "severity": "high",
                    })

            if self.state.solar_state:
                solar_assess = result.get("solar_weather", {})
                if solar_assess.get("grid_infrastructure_risk", 0) > 0.2 and self.state.infrastructure_state:
                    interactions.append({
                        "type": "solar_storm_grid_failure",
                        "description": "Solar storm risk to power grid creating cascading infrastructure failure — geomagnetic induced currents in transformers",
                        "severity": "high",
                    })
                if solar_assess.get("satellite_risk", 0) > 0.3 and self.state.cyber_state:
                    interactions.append({
                        "type": "solar_cyber_satellite_disruption",
                        "description": "Solar radiation disrupting satellite communications that cyber infrastructure depends on — compound satellite+cyber failure",
                        "severity": "high",
                    })
                if solar_assess.get("x_flare_probability", 0) > 0.3:
                    if self.state.technology_state and self.state.technology_state.get("current_adoption", 0.15) > 0.2:
                        interactions.append({
                            "type": "solar_flare_technology_disruption",
                            "description": "High flare probability coinciding with technology-dependent civilization — GPS, satellite, and radio blackout cascade risk",
                            "severity": "high",
                        })

            if self.state.oceanic_state:
                ocean_assess = result.get("oceanic_state", {})
                if ocean_assess.get("expected_hurricane_season", 0) > 15 and self.state.infrastructure_state:
                    interactions.append({
                        "type": "hurricane_infrastructure_damage",
                        "description": "Active hurricane season threatening coastal infrastructure — compound storm surge + infrastructure vulnerability",
                        "severity": "high",
                    })
                if ocean_assess.get("thermohaline_circulation_risk", 0) > 0.3 and self.state.climate_state:
                    interactions.append({
                        "type": "amoc_climate_shift",
                        "description": "Thermohaline circulation weakening amplifying climate disruption — European cooling + tropical rain belt shift",
                        "severity": "high",
                    })
                if ocean_assess.get("enso_strength", 0) > 0.8 and self.state.food_state:
                    interactions.append({
                        "type": "enso_food_disruption",
                        "description": "Strong ENSO event disrupting global weather patterns and agricultural production — crop failure in teleconnection regions",
                        "severity": "high",
                    })
                if ocean_assess.get("composite_ocean_risk", 0) > 0.4:
                    if self.state.biodiversity_state:
                        bio_idx = result.get("biodiversity_health", {}).get("biodiversity_index", 0.5)
                        if bio_idx < 0.5:
                            interactions.append({
                                "type": "ocean_biodiversity_collapse",
                                "description": "Ocean warming and acidification accelerating marine biodiversity collapse — coral reef die-off and fisheries collapse",
                                "severity": "high",
                            })

            if self.state.neo_state:
                neo_assess = result.get("neo_impact_risk", {})
                if neo_assess.get("composite_risk", 0) > 0.2 and self.state.infrastructure_state:
                    interactions.append({
                        "type": "asteroid_infrastructure_catastrophe",
                        "description": "Asteroid impact risk creating potential infrastructure catastrophe — city-level or regional devastation depending on size",
                        "severity": "high",
                    })
                if neo_assess.get("city_killer_probability_50yr", 0) > 0.05:
                    if self.state.economic_state:
                        interactions.append({
                            "type": "asteroid_economic_reset",
                            "description": "Non-trivial city-killer asteroid probability over 50 years with potential for global economic reset event",
                            "severity": "medium",
                        })

            if self.state.atmospheric_state:
                atm_assess = result.get("atmospheric_risk", {})
                if atm_assess.get("ssw_risk_score", 0) > 0.3 and self.state.energy_state:
                    interactions.append({
                        "type": "ssw_energy_demand_spike",
                        "description": "Sudden stratospheric warming causing extreme cold events and energy demand spikes beyond grid capacity",
                        "severity": "high",
                    })
                if atm_assess.get("ozone_deficit_pct", 0) > 20 and self.state.food_state:
                    interactions.append({
                        "type": "ozone_uv_food_damage",
                        "description": "Ozone depletion increasing UV radiation damaging crops and reducing agricultural productivity",
                        "severity": "medium",
                    })
                if atm_assess.get("circulation_disruption", 0) > 0.4 and self.state.climate_state:
                    interactions.append({
                        "type": "atmospheric_climate_feedback",
                        "description": "Atmospheric circulation disruption amplifying climate extremes — jet stream meander causing persistent weather patterns",
                        "severity": "high",
                    })
                if atm_assess.get("hadley_cell_expansion", 0) > 0.3 and self.state.water_state:
                    interactions.append({
                        "type": "hadley_expansion_desertification",
                         "description": "Hadley cell expansion driving subtropical desertification and water stress in Mediterranean and semi-arid regions",
                         "severity": "medium",
                     })

            if self.state.seismic_state and self.state.solar_state:
                seismic_assess = result.get("seismic_hazard", {})
                solar_assess = result.get("solar_weather", {})
                if seismic_assess.get("composite_hazard", 0) > 0.3 and solar_assess.get("x_flare_probability", 0) > 0.3:
                    interactions.append({
                        "type": "seismic_solar_electromagnetic_coupling",
                        "description": "High seismic stress + solar electromagnetic anomaly — electromagnetic precursors may be solar-modulated, complicating earthquake prediction",
                        "severity": "low",
                    })
                if solar_assess.get("composite_risk", 0) > 0.5 and seismic_assess.get("volcanic_risk", False):
                    interactions.append({
                        "type": "solar_volcanic_aerosol_feedback",
                        "description": "Solar extremes during active volcanic period — volcanic aerosols modulate solar radiation reaching surface, compound climate cooling risk",
                        "severity": "medium",
                    })

            if self.state.oceanic_state and self.state.seismic_state:
                ocean_assess = result.get("oceanic_state", {})
                seismic_assess = result.get("seismic_hazard", {})
                if ocean_assess.get("composite_ocean_risk", 0) > 0.4 and seismic_assess.get("composite_hazard", 0) > 0.4:
                    interactions.append({
                        "type": "subduction_seismic_tsunami_compound",
                        "description": "Active seismic zone + warm ocean conditions — subduction zone earthquake with tsunami risk amplified by sea level rise",
                        "severity": "high",
                    })
                if ocean_assess.get("thermohaline_circulation_risk", 0) > 0.3 and seismic_assess.get("composite_hazard", 0) > 0.3:
                    interactions.append({
                        "type": "hydrothermal_circulation_disruption",
                        "description": "Seismic activity disrupting hydrothermal circulation patterns that influence deep ocean mixing and thermohaline dynamics",
                        "severity": "low",
                    })

            if self.state.solar_state and self.state.atmospheric_state:
                solar_assess = result.get("solar_weather", {})
                atm_assess = result.get("atmospheric_risk", {})
                if solar_assess.get("composite_risk", 0) > 0.4 and atm_assess.get("ssw_risk_score", 0) > 0.3:
                    interactions.append({
                        "type": "solar_atmospheric_coupling",
                        "description": "Solar particle precipitation heating upper atmosphere during SSW — compound stratospheric disruption amplifying polar vortex instability",
                        "severity": "high",
                    })
                if solar_assess.get("x_flare_probability", 0) > 0.4:
                    if atm_assess.get("ozone_deficit_pct", 0) > 15:
                        interactions.append({
                            "type": "solar_uv_ozone_compound",
                            "description": "High solar UV during ozone deficit — compound UV radiation increase with surface UV hazard and skin cancer risk",
                            "severity": "high",
                        })

            if self.state.oceanic_state and self.state.atmospheric_state:
                ocean_assess = result.get("oceanic_state", {})
                atm_assess = result.get("atmospheric_risk", {})
                if ocean_assess.get("enso_strength", 0) > 0.8 and atm_assess.get("circulation_disruption", 0) > 0.3:
                    interactions.append({
                        "type": "enso_atmospheric_teleconnection",
                        "description": "Strong ENSO + disrupted atmospheric circulation — ENSO teleconnections amplified by jet stream anomalies, extreme weather in both basins",
                        "severity": "high",
                    })
                if ocean_assess.get("thermohaline_circulation_risk", 0) > 0.3 and atm_assess.get("hadley_cell_expansion", 0) > 0.2:
                    interactions.append({
                        "type": "ocean_atmosphere_heat_transport",
                        "description": "AMOC weakening + Hadley cell expansion — reduced ocean heat transport to North Atlantic combined with expanded subtropical drying zone",
                        "severity": "medium",
                    })

            if self.state.seismic_state and self.state.atmospheric_state:
                seismic_assess = result.get("seismic_hazard", {})
                atm_assess = result.get("atmospheric_risk", {})
                if seismic_assess.get("volcanic_risk", False) and atm_assess.get("ozone_deficit_pct", 0) > 10:
                    interactions.append({
                        "type": "volcanic_ozone_destruction",
                        "description": "Active volcanism accelerating ozone destruction — volcanic sulfate aerosols providing surfaces for heterogeneous ozone depletion chemistry",
                        "severity": "medium",
                    })
                if seismic_assess.get("composite_hazard", 0) > 0.5 and atm_assess.get("circulation_disruption", 0) > 0.4:
                    interactions.append({
                        "type": "volcanic_atmospheric_circulation",
                        "description": "Major volcanic eruption disrupting atmospheric circulation — aerosol injection causing jet stream perturbation and regional climate shifts",
                        "severity": "high",
                    })

            if self.state.neo_state and self.state.atmospheric_state:
                neo_assess = result.get("neo_impact_risk", {})
                atm_assess = result.get("atmospheric_risk", {})
                if neo_assess.get("composite_risk", 0) > 0.2 and atm_assess.get("greenhouse_pressure", 0) > 0.4:
                    interactions.append({
                        "type": "impact_atmospheric_injection",
                        "description": "Asteroid impact injecting massive dust and aerosols into atmosphere — impact winter scenario compounding existing atmospheric stress",
                        "severity": "high",
                    })

            if self.state.oceanic_state and self.state.solar_state:
                ocean_assess = result.get("oceanic_state", {})
                solar_assess = result.get("solar_weather", {})
                if ocean_assess.get("coral_bleaching_risk", 0) > 0.4 and solar_assess.get("x_flare_probability", 0) > 0.3:
                    interactions.append({
                        "type": "solar_uv_ocean_warming",
                        "description": "Solar activity + warm ocean compound coral bleaching — increased UV penetration in warm clear water accelerating coral stress",
                         "severity": "medium",
                     })

            if self.state.ecological_state:
                eco_assess = result.get("ecosystem_health", {})
                if eco_assess:
                    bio_idx = eco_assess.get("ecosystem_health_index", 0.5)
                    cascade = eco_assess.get("cascade_potential", 0.3)
                    if bio_idx < 0.4 and self.state.climate_state:
                        interactions.append({
                            "type": "ecological_climate_collapse",
                            "description": "Low ecosystem health + climate stress — biodiversity collapse accelerating under warming, compound ecological-climate tipping point",
                            "severity": "high",
                        })
                    if cascade > 0.4 and self.state.food_state:
                        interactions.append({
                            "type": "trophic_cascade_food_insecurity",
                            "description": "Ecosystem trophic cascade threatening pollinator populations and fish stocks — compounding food security risk",
                            "severity": "high",
                        })
                    if eco_assess.get("keystone_risk", 0) > 0.4 and self.state.water_state:
                        interactions.append({
                            "type": "ecological_water_cycle_disruption",
                            "description": "Keystone species loss disrupting natural water filtration and regulation — ecosystems that purify water collapsing",
                            "severity": "medium",
                        })
                    if eco_assess.get("stress_score", 0) > 0.5 and self.state.biodiversity_state:
                        bio_state_idx = result.get("biodiversity_health", {}).get("biodiversity_index", 0.5)
                        if bio_state_idx < 0.4:
                            interactions.append({
                                "type": "ecological_biodiversity_compound",
                                "description": "Ecological stress + biodiversity collapse — compound ecosystem destabilization with cascading species loss",
                                "severity": "high",
                            })
                    if eco_assess.get("adaptation_rate", 0) < 0.02 and self.state.climate_state:
                        interactions.append({
                            "type": "evolutionary_lag_climate",
                            "description": "Slow evolutionary adaptation rate vs rapid climate change — species cannot adapt fast enough, extinction debt accelerating",
                            "severity": "medium",
                        })

            if self.state.paleoclimate_state:
                paleo_assess = result.get("paleoclimate_forcing", {})
                if paleo_assess:
                    glaciation_risk = paleo_assess.get("glaciation_risk", 0)
                    deglaciation_risk = paleo_assess.get("deglaciation_risk", 0)
                    if deglaciation_risk > 0.4 and self.state.climate_state:
                        interactions.append({
                            "type": "paleoclimate_current_warming_analog",
                            "description": "Paleoclimate analogs indicate current CO2 forcing exceeds any interglacial — commitment to multi-meter sea level rise confirmed by deep-time evidence",
                            "severity": "high",
                        })
                    if glaciation_risk > 0.4 and self.state.oceanic_state:
                        interactions.append({
                            "type": "orbital_forcing_thermohaline",
                            "description": "Milankovitch cooling forcing interacting with thermohaline circulation — potential for abrupt climate shift if AMOC weakens",
                            "severity": "medium",
                        })
                    if paleo_assess.get("radiative_forcing_wm2", 0) > 3.0:
                        interactions.append({
                            "type": "radiative_forcing_exceeds_petm",
                            "description": "Current radiative forcing rate exceeds PETM — unprecedented climate change velocity creating adaptation challenges across all domains",
                            "severity": "high",
                        })

            if self.state.cosmic_radiation_state:
                cr_assess = result.get("cosmic_radiation_risk", {})
                if cr_assess:
                    gcr_flux = cr_assess.get("effective_gcr_flux", 1.0)
                    svensmark = cr_assess.get("svensmark_cloud_effect", 0)
                    if svensmark > 0.1 and self.state.climate_state:
                        interactions.append({
                            "type": "gcr_cloud_nucleation_climate",
                            "description": "Cosmic ray cloud nucleation hypothesis — elevated GCR flux may increase low cloud cover, providing regional cooling feedback",
                            "severity": "low",
                        })
                    if gcr_flux > 1.3 and self.state.solar_state:
                        interactions.append({
                            "type": "solar_minimum_gcr_amplification",
                            "description": "Solar minimum amplifies GCR flux — combined effect increasing atmospheric ionization, potential cloud modulation, and aviation radiation exposure",
                            "severity": "medium",
                        })
                    if cr_assess.get("electronics_seu_risk", 0) > 0.4 and self.state.infrastructure_state:
                        interactions.append({
                            "type": "gcr_infrastructure_seu",
                            "description": "Elevated GCR causing single-event upsets in electronics — satellite and aviation systems at increased risk of computational errors",
                            "severity": "medium",
                        })

            if self.state.gravitational_state:
                grav_assess = result.get("gravitational_tidal", {})
                if grav_assess:
                    seismic_trigger = grav_assess.get("seismic_triggering_probability", 0)
                    volcanic_trigger = grav_assess.get("volcanic_triggering_probability", 0)
                    if seismic_trigger > 0.3 and self.state.seismic_state:
                        interactions.append({
                            "type": "tidal_seismic_coupling",
                            "description": "Enhanced tidal stress coupling with fault zones — gravitational loading slightly increasing probability of triggering earthquakes on critically stressed faults",
                            "severity": "low",
                        })
                    if volcanic_trigger > 0.2 and self.state.atmospheric_state:
                        interactions.append({
                            "type": "tidal_volcanic_atmospheric",
                            "description": "Tidal loading influencing volcanic eruption timing — potential for increased volcanic aerosol injection into stratosphere during perigee-spring tides",
                            "severity": "low",
                        })
                    coastal_risk = grav_assess.get("coastal_flood_risk", 0)
                    if coastal_risk > 0.4 and self.state.oceanic_state:
                        interactions.append({
                            "type": "king_tide_coastal_compound",
                            "description": "King tide + ENSO-driven sea level anomaly — compound coastal flooding from gravitational tidal forcing compounding oceanic conditions",
                            "severity": "medium",
                        })

            if self.state.seismic_state and self.state.climate_state:
                seismic_haz = result.get("seismic_hazard", {})
                if seismic_haz.get("volcanic_risk", False) or seismic_haz.get("composite_hazard", 0) > 0.5:
                    if self.state.climate_state.global_temp_anomaly_c > 1.0:
                        interactions.append({
                            "type": "volcanic_winter_climate_feedback",
                            "description": "Active volcanism + warming climate — volcanic aerosol injection can temporarily reverse warming trend, creating decade-scale climate whiplash with cascading agricultural and economic impacts",
                            "severity": "medium",
                        })

            if self.state.paleoclimate_state and self.state.seismic_state:
                paleo_assess = result.get("paleoclimate_forcing", {})
                if paleo_assess.get("radiative_forcing_wm2", 0) > 2.0 and result.get("seismic_hazard", {}).get("volcanic_risk", False):
                    interactions.append({
                        "type": "anthropogenic_volcanic_tug_of_war",
                        "description": "Anthropogenic radiative forcing exceeds volcanic cooling capacity — even major eruptions cannot offset current CO2 forcing, but create dangerous climate volatility",
                        "severity": "high",
                    })

            if self.state.paleoclimate_state and self.state.oceanic_state:
                paleo_assess = result.get("paleoclimate_forcing", {})
                ocean_assess = result.get("oceanic_state", {})
                if paleo_assess.get("deglaciation_risk", 0) > 0.3 and ocean_assess.get("thermohaline_circ_risk", ocean_assess.get("coral_bleaching_risk", 0)) > 0.3:
                    interactions.append({
                        "type": "paleoclimate_thermohaline_collapse",
                        "description": "Paleoclimate evidence suggests AMOC weakening under CO2 forcing — mid-Pliocene analog indicates committed sea level rise compounds oceanic circulation disruption",
                        "severity": "high",
                    })

            if self.state.cosmic_radiation_state and self.state.atmospheric_state:
                cr_assess = result.get("cosmic_radiation_risk", {})
                if cr_assess.get("svensmark_cloud_effect", 0) > 0.08 and result.get("atmospheric_risk", {}).get("hadley_cell_expansion", 0) > 5:
                    interactions.append({
                        "type": "gcr_cloud_hadley_coupling",
                        "description": "GCR-enhanced cloud nucleation interacting with Hadley cell expansion — potential regional climate modification through cloud-albedo feedback, especially during solar minimum",
                        "severity": "low",
                    })

            if self.state.ecological_state and self.state.paleoclimate_state:
                eco_assess = result.get("ecosystem_health", {})
                paleo_assess = result.get("paleoclimate_forcing", {})
                if eco_assess.get("ecosystem_health_index", 0.5) < 0.4 and paleo_assess.get("radiative_forcing_wm2", 0) > 2.0:
                    interactions.append({
                        "type": "paleoclimate_mass_extinction_analog",
                        "description": "Current forcing rate exceeds PETM — ecological collapse rate is unprecedented in paleoclimate record, species cannot migrate or adapt at this velocity",
                        "severity": "high",
                    })

            if self.state.gravitational_state and self.state.oceanic_state and self.state.atmospheric_state:
                grav_assess = result.get("gravitational_tidal", {})
                ocean_assess = result.get("oceanic_state", {})
                atm_assess = result.get("atmospheric_risk", {})
                if grav_assess.get("coastal_flood_risk", 0) > 0.3 and ocean_assess.get("hurricane_season", 0) and atm_assess.get("ssw_risk_score", 0) > 0.3:
                    interactions.append({
                        "type": "triple_compound_coastal",
                        "description": "King tides + hurricane season + sudden stratospheric warming — triple compound coastal hazard from gravitational, oceanic, and atmospheric convergence",
                        "severity": "high",
                    })

            if self.state.compound_hazard_state:
                compound_assess = result.get("compound_hazard", {})
                if compound_assess:
                    ch_idx = compound_assess.get("compound_hazard_index", 0)
                    sys_collapse = compound_assess.get("systemic_collapse_probability", 0)
                    if ch_idx > 0.5 and self.state.economic_state:
                        interactions.append({
                            "type": "compound_hazard_economic_cascade",
                            "description": "Elevated compound hazard index threatening economic stability — simultaneous physical and systemic risks exceeding resilience capacity",
                            "severity": "high",
                        })
                    if sys_collapse > 0.2:
                        interactions.append({
                            "type": "systemic_cascade_amplification",
                            "description": "Compound hazard systemic collapse probability elevated — cascading failures across multiple domains creating civilizational risk",
                            "severity": "high" if sys_collapse > 0.3 else "medium",
                        })
                    if compound_assess.get("resilience_score", 1.0) < 0.3:
                        interactions.append({
                            "type": "compound_hazard_resilience_crisis",
                            "description": "Low system resilience under compound hazards — infrastructure and governance failing to buffer multi-domain stress",
                            "severity": "high",
                        })

            if self.state.fate_state:
                fate_traj = result.get("fate_trajectory", {})
                thermo = result.get("fate_thermodynamic", {})
                depletion = result.get("fate_depletion", {})
                commitments = result.get("fate_commitments", {})
                if (self.state.fate_state.nuclear_warheads > 10000 and self.state.fate_state.nuclear_proliferation_risk > 0.3) and self.state.geopolitical_state:
                    if self.state.geopolitical_state.us_china_tension > 0.5:
                        interactions.append({
                            "type": "fate_geopolitical_existential",
                            "description": "Large nuclear arsenals with proliferation risk amplified by geopolitical tension — nuclear confrontation pushing civilizational trajectory toward collapse",
                            "severity": "high",
                        })
                if self.state.fate_state.climate_commitment_c > 1.5 and self.state.climate_state:
                    if self.state.climate_state.global_temp_anomaly_c > 1.5:
                        interactions.append({
                            "type": "fate_climate_commitment",
                            "description": "Civilizational fate coupling with climate commitment — warming locked into civilizational trajectory through infrastructure and energy commitments",
                            "severity": "high",
                        })
                if self.state.fate_state.energy_consumption_tw > 17 and self.state.energy_state:
                    if self.state.energy_state.get("renewable_share", 20) < 25:
                        interactions.append({
                            "type": "fate_thermodynamic_energy_trap",
                            "description": "High energy consumption with fossil dependence creating thermodynamic trap — declining EROI limiting energy transition capacity",
                            "severity": "high",
                        })
                if self.state.fate_state.resource_remaining_pct < 40 and self.state.commodities_state:
                    comm_stress = self.state.commodities_state.get("supply_chain_stress", 0.3)
                    if comm_stress > 0.4:
                        interactions.append({
                            "type": "fate_resource_depletion_commodity",
                            "description": "Accelerating resource depletion compounding commodity supply chain stress — critical mineral scarcity constraining technology and energy transition",
                            "severity": "high",
                        })
                if self.state.fate_state.technological_dependency > 0.7 and self.state.cyber_state:
                    if self.state.cyber_state.ai_threat_capability > 0.4:
                        interactions.append({
                            "type": "fate_technology_dependency_cyber",
                            "description": "High technological dependency creating fate-level cyber vulnerability — AI-augmented attacks could cascade into civilizational disruption",
                            "severity": "high",
                        })
                if self.state.fate_state.biodiversity_integrity_index < 0.4 and self.state.biodiversity_state:
                    bio_idx = result.get("biodiversity_health", {}).get("biodiversity_index", 0.5)
                    if bio_idx < 0.4:
                        interactions.append({
                            "type": "fate_biodiversity_determinism",
                            "description": "Low biodiversity integrity locked into civilizational fate — ecosystem collapse is a deterministic commitment reducing food and water security",
                            "severity": "high",
                        })
                if self.state.fate_state.governance_effectiveness < 0.4 and self.state.political_state:
                    pol_stability = result.get("political_stability", {}).get("overall_stability", 0.5)
                    if pol_stability < 0.4:
                        interactions.append({
                            "type": "fate_governance_failure",
                            "description": "Low governance effectiveness creating fate-level political instability — inability to coordinate global responses to existential risks",
                            "severity": "high",
                        })
                if self.state.fate_state.information_integrity < 0.4 and self.state.mental_health_state:
                    mh_burden = result.get("mental_health", {}).get("overall_burden", 0)
                    if mh_burden > 0.2:
                        interactions.append({
                            "type": "fate_information_mental_health",
                            "description": "Low information integrity driving mental health deterioration — misinformation as a deterministic force eroding societal epistemic capacity",
                            "severity": "medium",
                        })
                if self.state.fate_state.economic_inequality_gini > 0.6 and self.state.social_state:
                    unrest = result.get("social", {}).get("unrest_probability", 0.3)
                    if unrest > 0.4:
                        interactions.append({
                            "type": "fate_inequality_unrest",
                            "description": "High economic inequality locked into social unrest destiny — structural inequality as a deterministic driver of civil instability",
                            "severity": "high",
                        })
                if depletion and depletion.get("n_critical_resources", 0) > 3 and self.state.food_state:
                    food_risk = result.get("food_security", {}).get("overall_risk", 0)
                    if food_risk > 0.3:
                        interactions.append({
                            "type": "fate_resource_food_insecurity",
                            "description": "Critical resource depletion threatening food production capacity — soil, phosphorus, and water depletion as locked-in agricultural fate commitments",
                            "severity": "high",
                        })
                if commitments and commitments.get("fate_coupling", 0) > 0.6 and self.state.infrastructure_state:
                    infra = result.get("infrastructure_resilience", {})
                    if infra.get("threat_level", "low") in ("high", "critical"):
                        interactions.append({
                            "type": "fate_commitment_infrastructure",
                            "description": "High fate coupling with infrastructure vulnerability — locked-in commitments (sea level rise, climate) threatening coastal and critical infrastructure",
                            "severity": "high",
                        })
                if self.state.fate_state.ai_capability_index > 0.3 and self.state.technology_state:
                    tech_adoption = self.state.technology_state.get("current_adoption", 0.15)
                    if tech_adoption > 0.3:
                        interactions.append({
                            "type": "fate_ai_singularity_trajectory",
                            "description": "Rising AI capability approaching critical threshold — civilizational trajectory increasingly dependent on AI alignment, creating deterministic technology fate",
                            "severity": "high",
                        })
                if self.state.fate_state.nuclear_warheads > 10000 and self.state.fate_state.nuclear_proliferation_risk > 0.3:
                    interactions.append({
                        "type": "fate_nuclear_determinism",
                        "description": "Large nuclear arsenals with proliferation risk creating civilizational-level determinism — nuclear weapons as locked-in existential commitment",
                        "severity": "high",
                    })
                if self.state.fate_state.water_stress_global > 0.5 and self.state.water_state:
                    water_risk = result.get("water_security", {}).get("overall_risk", 0)
                    if water_risk > 0.3:
                        interactions.append({
                            "type": "fate_water_stress_determinism",
                            "description": "Global water stress written into civilizational fate — demographic and climate commitments guaranteeing sustained water insecurity",
                            "severity": "high",
                        })
                if self.state.fate_state.food_system_resilience < 0.5 and self.state.food_state:
                    food_risk = result.get("food_security", {}).get("overall_risk", 0)
                    if food_risk > 0.3:
                        interactions.append({
                            "type": "fate_food_resilience_determinism",
                            "description": "Low food system resilience as civilizational fate — population momentum and climate commitments locking in agricultural vulnerability",
                            "severity": "high",
                        })

            if self.state.personal_fate_state:
                personal_fate_result = result.get("personal_fate", {})
                if personal_fate_result:
                    agency = personal_fate_result.get("agency_score", 0.5)
                    determinism = personal_fate_result.get("total_determinism", 0.5)
                    if determinism > 0.5 and self.state.fate_state:
                        if self.state.fate_state.economic_inequality_gini > 0.6:
                            interactions.append({
                                "type": "personal_fate_structural_inequality",
                                "description": "High personal determinism coupling with structural inequality — individual fate constrained by civilizational-level economic inequality commitment",
                                "severity": "high",
                            })
                    if agency < 0.6 and self.state.mental_health_state:
                        mh_burden = result.get("mental_health", {}).get("overall_burden", 0)
                        if mh_burden > 0.2:
                            interactions.append({
                                "type": "personal_fate_agency_mental_health",
                                "description": "Low agency score compounding mental health burden — habit inertia and structural determinism reducing capacity for self-directed change",
                                "severity": "medium",
                            })
                    mortality = result.get("personal_mortality", {})
                    if mortality and mortality.get("modifiable_years_gained", 0) > 5:
                        if self.state.fate_state and self.state.fate_state.biotech_capability < 0.3:
                            interactions.append({
                                "type": "personal_fate_biotech_gap",
                                "description": "Significant modifiable life years available but biotech capability insufficient — gap between personal health potential and civilizational medical capacity",
                                "severity": "medium",
                            })

            if self.state.personal_life_profile:
                pl_comprehensive = result.get("personal_life_comprehensive", {})
                pl_health = result.get("personal_life_health", {})
                pl_financial = result.get("personal_life_financial", {})
                pl_relationship = result.get("personal_life_relationship", {})
                if pl_comprehensive:
                    outlook = pl_comprehensive.get("overall_outlook_score", 0.5)
                    if outlook < 0.4 and self.state.mental_health_state:
                        mh_burden = result.get("mental_health", {}).get("overall_burden", 0)
                        if mh_burden > 0.2:
                            interactions.append({
                                "type": "personal_life_outlook_mental_health",
                                "description": "Low personal life outlook compounding mental health burden — reinforcing cycle of reduced agency and deteriorating wellbeing",
                                "severity": "high",
                            })
                    if pl_health:
                        le = pl_health.get("life_expectancy_adjusted", 79)
                        if le < 70 and self.state.housing_state:
                            interactions.append({
                                "type": "personal_life_health_housing",
                                "description": "Below-average life expectancy interacting with housing stress — poor health reducing economic mobility and housing stability",
                                "severity": "high",
                            })
                    if pl_financial:
                        risks = pl_financial.get("financial_risks", [])
                        high_severity = [r for r in risks if r.get("severity") == "high"]
                        if len(high_severity) >= 2 and self.state.food_state:
                            interactions.append({
                                "type": "personal_life_financial_food_insecurity",
                                "description": "Multiple high-severity financial risks compounding food insecurity risk — financial fragility amplifying basic needs vulnerability",
                                "severity": "high",
                            })
                    if pl_relationship:
                        loneliness = pl_relationship.get("loneliness_risk", 0)
                        if loneliness > 0.4 and self.state.mental_health_state:
                            interactions.append({
                                "type": "personal_life_loneliness_mental_health",
                                "description": "Elevated loneliness risk compounding mental health vulnerability — social isolation as both predictor and amplifier of psychological distress",
                                "severity": "medium",
                            })
                    if pl_health:
                        top_risks = pl_health.get("top_3_risks", [])
                        cardiovascular_risk = next((r[1] for r in top_risks if "cardiovascular" in r[0]), 0)
                        if cardiovascular_risk > 0.3 and self.state.personal_life_profile and self.state.personal_life_profile.stress_level > 0.5:
                            interactions.append({
                                "type": "personal_life_cardiovascular_stress",
                                "description": "Elevated cardiovascular risk with personal stress compounding vascular health — behavioral and biological risk reinforcement",
                                "severity": "high",
                            })
                    if self.state.fate_state and self.state.fate_state.economic_inequality_gini > 0.6:
                        if pl_financial and pl_financial.get("current_net_worth", 0) < 50000:
                            interactions.append({
                                "type": "personal_life_inequality_trap",
                                "description": "Personal financial fragility within high structural inequality — inequality trap reducing individual economic mobility despite effort",
                                "severity": "high",
                            })
                    if pl_comprehensive.get("overall_outlook_label") == "at_risk":
                        interactions.append({
                            "type": "personal_life_comprehensive_at_risk",
                            "description": "Personal comprehensive outlook at risk — multiple life domains (health, financial, social, career) showing compounding negative trajectories",
                            "severity": "high",
                        })

            prescience_result = result.get("prescience", {})
            if prescience_result:
                prescience_score = prescience_result.get("prescience_score", 0)
                if prescience_score > 0.6:
                    if self.state.geopolitical_state and self.state.geopolitical_state.us_china_tension > 0.5:
                        interactions.append({
                            "type": "prescience_geopolitical_escalation",
                            "description": "Elevated prescience risk converging with geopolitical tensions — forecast events indicate higher probability of major conflict",
                            "severity": "high",
                        })
                    if self.state.climate_state and self.state.climate_state.global_temp_anomaly_c > 1.5:
                        interactions.append({
                            "type": "prescience_climate_tipping",
                            "description": "High prescience risk aligning with climate tipping point proximity — forecast events indicate climate cascade triggering",
                            "severity": "high",
                        })
                    if self.state.pandemic_state:
                        pan_prep = result.get("pandemic_preparedness", {}).get("overall_preparedness", 0.5)
                        if pan_prep < 0.4:
                            interactions.append({
                                "type": "prescience_pandemic_vulnerability",
                                "description": "Elevated prescience forecast converging with low pandemic preparedness — event probability higher than domain models alone suggest",
                                "severity": "high",
                            })
                    if self.state.infrastructure_state:
                        infra = result.get("infrastructure_resilience", {})
                        if infra.get("threat_level", "low") in ("high", "critical"):
                            interactions.append({
                                "type": "prescience_infrastructure_cascade",
                                "description": "Prescience forecast amplifying infrastructure vulnerability — compound event probability exceeds sum of individual domain risks",
                                "severity": "high",
                            })
                prescience_warnings = result.get("prescience_warnings", {})
                if prescience_warnings and len(prescience_warnings.get("active_warnings", [])) >= 3:
                    interactions.append({
                        "type": "prescience_multi_warning_convergence",
                        "description": "Multiple early warning signals active simultaneously — prescience engine detecting converging risk patterns across domains",
                        "severity": "high",
                    })
                prescience_events = result.get("prescience_events", {})
                if prescience_events:
                    high_prob_events = [e for e in prescience_events.get("events", []) if e.get("annual_probability", 0) > 0.05]
                    if len(high_prob_events) >= 2:
                        interactions.append({
                            "type": "prescience_concurrent_event_probability",
                            "description": "Multiple high-probability prescience events active simultaneously — compound event risk exceeding individual event forecasts",
                            "severity": "high",
                        })

            if result.get("conformal") and result.get("quantile_regression"):
                conformal_width = abs(result["conformal"].get("upper", 0) - result["conformal"].get("lower", 0)) if isinstance(result.get("conformal"), dict) else 0
                if conformal_width > 2.0:
                    interactions.append({
                        "type": "predictive_science_wide_uncertainty",
                        "description": "Conformal prediction intervals wide — high uncertainty across all domain models suggests regime shift or structural break, Bayesian model averaging recommended",
                        "severity": "medium",
                    })

            if result.get("probabilistic") and isinstance(result.get("probabilistic"), dict):
                crps = result["probabilistic"].get("crps", 0)
                if crps > 1.0:
                    interactions.append({
                        "type": "predictive_science_poor_calibration",
                        "description": "High CRPS score indicates poor probabilistic calibration — ensemble diversification and model reweighting recommended",
                        "severity": "medium",
                    })

            if self.state.climate_state and self.state.energy_state:
                climate_risk = self.state.climate_state.global_temp_anomaly_c if self.state.climate_state else 0
                energy_mix = result.get("energy", {})
                renewable_pct = energy_mix.get("renewable_pct", 0) if isinstance(energy_mix, dict) else 0
                if climate_risk > 1.2 and renewable_pct < 0.3:
                    gp_uncertainty = "high"
                    interactions.append({
                        "type": "predictive_science_gp_climate_energy",
                        "description": "Gaussian process uncertainty propagation shows climate-energy nexus risk amplification — temperature anomaly compounding with low renewable transition creating non-linear risk escalation",
                        "severity": "high",
                    })

            if self.state.geopolitical_state and self.state.supply_chain_state:
                geo_tension = self.state.geopolitical_state.us_china_tension if self.state.geopolitical_state else 0
                supply_risk = self.state.supply_chain_state.supplier_concentration if self.state.supply_chain_state else 0
                if geo_tension > 0.5 and supply_risk > 0.4:
                    interactions.append({
                        "type": "predictive_science_cvar_geopolitical_supply",
                        "description": "Conditional VaR decomposition reveals geopolitical-supply chain tail risk concentration — systematic risk exceeds diversification benefit during conflict escalation",
                        "severity": "high",
                    })

            if self.state.housing_state and self.state.mental_health_state:
                housing_afford = self.state.housing_state.housing_affordability_index if self.state.housing_state else 1.0
                mh_burden = self.state.mental_health_state.depression_prevalence if self.state.mental_health_state else 0
                if housing_afford < 0.5 and mh_burden > 0.3:
                    interactions.append({
                        "type": "predictive_science_dml_housing_mental",
                        "description": "Double ML causal inference identifies housing affordability as significant treatment effect on mental health burden — causal estimate suggests policy intervention leverage",
                        "severity": "medium",
                    })

            if self.hmm._fitted or True:
                if self.state.geopolitical_state and self.state.economic_state:
                    regime_result = self.hmm.detect_regimes(
                        np.array([self.state.geopolitical_state.conflict_intensity,
                                  self.state.economic_state.inflation_rate])
                    ) if hasattr(self, '_recent_regime_data') else None
                    regimes = (regime_result.regimes if regime_result else [])
                    if len(regimes) > 10:
                        regime_transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
                        if regime_transitions > 3:
                            interactions.append({
                                "type": "predictive_science_hmm_regime_volatility",
                                "description": "HMM regime detection finds frequent regime transitions in geopolitical-economic state — system in volatile multi-regime pattern suggesting elevated uncertainty",
                                "severity": "high",
                            })

            if self.copula_garch._fitted or True:
                if self.state.economic_state:
                    interactions.append({
                        "type": "predictive_science_copula_garch_tail_risk",
                        "description": "Copula-GARCH modeling reveals tail dependency structure in economic-climatic coupling — asymmetric lower tail dependence increases joint crash probability",
                        "severity": "medium",
                    })
                if self.state.financial_state if hasattr(self.state, 'financial_state') else None:
                    interactions.append({
                        "type": "predictive_science_rnn_sequence_forecast",
                        "description": "RNN/LSTM sequence model captures nonlinear temporal dependencies in financial volatility — recurrent memory gates identify regime-dependent forecast dynamics",
                        "severity": "low",
                    })

            if self.state.economic_state:
                interactions.append({
                    "type": "predictive_science_koopman_regime_shift",
                    "description": "Koopman operator analysis reveals dominant eigenvalue spectrum of economic dynamics — unstable modes indicate regime shift risk and bifurcation proximity",
                    "severity": "medium",
                })

            if self.state.geopolitical_state:
                interactions.append({
                    "type": "predictive_science_abm_emergent_crisis",
                    "description": "Agent-based simulation of geopolitical actors identifies emergent crisis patterns — network topology and local interaction rules produce systemic risk without centralized failure",
                    "severity": "medium",
                })

            if self.state.economic_state:
                interactions.append({
                    "type": "predictive_science_symbolic_law_discovery",
                    "description": "Symbolic regression discovers compact mathematical laws governing economic dynamics — interpretable expressions reveal hidden structure in GDP-inflation-unemployment relationships",
                    "severity": "low",
                })

            interactions.append({
                "type": "evolutionary_simulation_fitness_optimization",
                "description": "Evolutionary simulation models population adaptation under selection pressure — fitness landscape ruggedness and epistasis determine evolutionary trajectories and convergence rates",
                "severity": "low",
            })

            if self.state.technology_state:
                interactions.append({
                    "type": "learning_simulation_skill_cascade",
                    "description": "Population learning on networks models technology skill diffusion — success-biased learning accelerates adoption cascades while conformist strategies slow innovation spread",
                    "severity": "medium",
                })

            if self.state.social_state:
                interactions.append({
                    "type": "cultural_evolution_trait_selection",
                    "description": "Cultural evolution models trait frequency dynamics under selection pressure — social learning strategies determine which innovations persist and spread through populations",
                    "severity": "low",
                })

            if self.state.economic_state or self.state.geopolitical_state:
                interactions.append({
                    "type": "physical_constants_dimensional_consistency",
                    "description": "Physical constants module ensures dimensional consistency across all predictions — fundamental constant relationships govern energy-mass-time scaling laws that underpin economic and geopolitical models",
                    "severity": "low",
                })

            if self.state.economic_state:
                interactions.append({
                    "type": "ai_integration_cross_domain_synthesis",
                    "description": "AI insight engine synthesizes cross-domain analysis via large language model reasoning — identifies non-obvious connections between economic, geopolitical, climate, and social dynamics that quantitative models may miss",
                    "severity": "medium",
                })

            domain_values = {}
            for domain_name in ["geopolitical", "climate", "economics", "sentiment", "technology", "social", "energy", "health"]:
                domain_data = result.get(domain_name, {})
                if isinstance(domain_data, dict):
                    for k, v in domain_data.items():
                        if isinstance(v, (int, float)):
                            domain_values[f"{domain_name}_{k}"] = float(v)

            if domain_values:
                try:
                    paradox_report = self.paradox_detector.detect(domain_values)
                    if paradox_report.paradoxes:
                        for p in paradox_report.paradoxes:
                            interactions.append({
                                "type": f"paradox_{p.name}",
                                "description": p.description + (" — " + p.explanation if p.explanation else ""),
                                "severity": "high" if p.severity == "critical" else p.severity,
                            })
                except Exception:
                    pass

            if len(interactions) > 5:
                try:
                    narrative = self.narrative_engine.generate_narrative(
                        domain_states={k: {"value": v} for k, v in domain_values.items()} if domain_values else {},
                        horizon_years=5,
                    )
                    if narrative.arc != StoryArc.STAGNATION:
                        interactions.append({
                            "type": f"narrative_{narrative.arc.value}",
                            "description": narrative.headline,
                            "severity": "medium" if narrative.arc in (StoryArc.RECOVERY, StoryArc.BOOM) else "high",
                        })
                except Exception:
                    pass

            if len(interactions) > 3 and len(domain_values) > 2:
                try:
                    worst_case = self.adversarial_stress.find_worst_case(
                        domains=list(set(k.split("_")[0] for k in domain_values.keys()))[:5]
                    )
                    if worst_case.total_risk > 0.7:
                        interactions.append({
                            "type": "adversarial_worst_case",
                            "description": f"Adversarial stress testing reveals systemic vulnerability: worst-case total risk {worst_case.total_risk:.1%} across {len(worst_case.domain_shocks)} domains — {worst_case.description}",
                            "severity": "critical" if worst_case.total_risk > 0.8 else "high",
                        })
                except Exception:
                    pass

            if domain_values:
                try:
                    feedback = self.cross_domain_feedback.propagate(domain_values)
                    high_impact = {k: v for k, v in feedback.get("net_effects", {}).items() if abs(v) > 0.3}
                    if high_impact:
                        for var, effect in list(high_impact.items())[:3]:
                            interactions.append({
                                "type": "cross_domain_feedback_amplification",
                                "description": f"Cross-domain feedback amplification: {var} shows net effect {effect:+.2f} — cascading influence across interconnected domains",
                                "severity": "high" if abs(effect) > 0.5 else "medium",
                            })
                except Exception:
                    pass

            if len(interactions) > 10:
                try:
                    calibration_summary = self.calibration_tracker.summary()
                    if calibration_summary.get("total_forecasts", 0) > 5:
                        avg_coverage = calibration_summary.get("average_coverage_95", 1.0)
                        if avg_coverage < 0.8:
                            interactions.append({
                                "type": "calibration_overconfidence",
                                "description": f"Calibration tracking shows {(1-avg_coverage)*100:.0f}% overconfidence — {calibration_summary.get('total_forecasts', 0)} forecasts have narrower CIs than realized outcomes warrant",
                                "severity": "medium",
                            })
                except Exception:
                    pass

            if domain_values:
                try:
                    precursor_matches = self.precursor_matcher.match(domain_values)
                    if precursor_matches and any(m.overall_progress > 0.4 for m in precursor_matches):
                        for m in precursor_matches[:3]:
                            interactions.append({
                                "type": f"precursor_{m.pattern.regime_type.value}",
                                "description": f"Regime change precursor detected: {m.pattern.name} at {m.overall_progress:.0%} progress — {m.phase.value} phase, {len(m.matched_steps)}/{len(m.pattern.steps)} steps triggered",
                                "severity": "high" if m.overall_progress > 0.6 else "medium",
                            })
                except Exception:
                    pass

            if domain_values:
                try:
                    cascade_result = self.cascade_tree.simulate(list(domain_values.keys())[:5])
                    critical_paths = self.cascade_tree.critical_paths()
                    if critical_paths and cascade_result.probability > 0.3:
                        interactions.append({
                            "type": "cascade_failure_risk",
                            "description": f"Failure cascade risk: {cascade_result.probability:.0%} probability with {len(critical_paths)} critical paths — most likely path: {' → '.join(critical_paths[0][:3])}...",
                            "severity": "critical" if cascade_result.probability > 0.6 else "high",
                        })
                except Exception:
                    pass

            if domain_values and len(domain_values) > 4:
                try:
                    counterfactual = self.counterfactual.analyze(
                        domain_values, target_variable=list(domain_values.keys())[0]
                    )
                    if counterfactual.minimal_interventions:
                        top_intervention = counterfactual.minimal_interventions[0]
                        interactions.append({
                            "type": "counterfactual_intervention",
                            "description": f"Counterfactual analysis: changing {top_intervention.variable} by {top_intervention.required_change:+.2f} would flip outcome probability from {counterfactual.current_probability:.0%} to {top_intervention.resulting_probability:.0%}",
                            "severity": "medium",
                        })
                except Exception:
                    pass

            ts_data = self.state.time_series_data if hasattr(self.state, 'time_series_data') and self.state.time_series_data else {}
            ts_array_data = {}
            if ts_data:
                for k, v in ts_data.items():
                    try:
                        ts_array_data[k] = v.data if hasattr(v, 'data') else np.array(v, dtype=float)
                    except Exception:
                        pass
            if ts_array_data:
                try:
                    swan_report = self.black_swan.detect(ts_array_data, result)
                    if swan_report.category.value in ("black_swan", "dragon_king"):
                        interactions.append({
                            "type": f"black_swan_{swan_report.category.value}",
                            "description": f"Black Swan detection: {swan_report.category.value} probability {swan_report.overall_probability:.0%} — surprise potential {swan_report.surprise_potential:.0%}",
                            "severity": "critical" if swan_report.overall_probability > 0.5 else "high",
                        })
                    elif swan_report.category.value == "known_unknown":
                        interactions.append({
                            "type": "known_unknown_risk",
                            "description": f"Known unknown risk detected: {swan_report.overall_probability:.0%} probability, structural vulnerability {np.mean([v.score for v in swan_report.structural_vulnerabilities]):.0%}",
                            "severity": "medium",
                        })
                except Exception:
                    pass

            if ts_array_data and len(ts_array_data) > 1:
                try:
                    shift_report = self.regime_shift.detect_all(ts_array_data)
                    high_risk = [w for w in shift_report.warnings if w.probability > 0.4]
                    if high_risk:
                        for w in high_risk[:2]:
                            interactions.append({
                                "type": f"regime_shift_{w.shift_type.value}",
                                "description": f"Regime shift warning for {w.variable}: {w.probability:.0%} probability, {w.shift_type.value} type — {w.n_signals_triggered}/{len(w.warning_signals)} signals triggered",
                                "severity": "high" if w.probability > 0.6 else "medium",
                            })
                except Exception:
                    pass

            if domain_values and len(domain_values) > 3:
                try:
                    trace = self.butterfly_tracker.trace_perturbation(
                        list(domain_values.keys())[0], 0.3
                    )
                    if trace.butterfly_index > 5.0:
                        interactions.append({
                            "type": "butterfly_effect",
                            "description": f"Perturbation amplification detected: butterfly index {trace.butterfly_index:.1f}x — small changes in {list(domain_values.keys())[0]} can cascade through {len(trace.reached_domains)} domains",
                            "severity": "high" if trace.butterfly_index > 10 else "medium",
                        })
                except Exception:
                    pass

            if domain_values and len(domain_values) > 2:
                try:
                    cross_impact_result = self.cross_impact.solve()
                    big_changes = {k: v for k, v in cross_impact_result.probability_changes.items() if abs(v) > 0.05}
                    if big_changes:
                        top_change = max(big_changes.items(), key=lambda x: abs(x[1]))
                        interactions.append({
                            "type": "cross_impact_feedback",
                            "description": f"Cross-impact analysis reveals {top_change[0]} probability shifts {top_change[1]:+.0%} due to cascading event interactions — {len(big_changes)} events significantly affected",
                            "severity": "high" if abs(top_change[1]) > 0.1 else "medium",
                        })
                except Exception:
                    pass

            if domain_values and len(domain_values) > 2:
                try:
                    stress_report = self.stress_testing.stress_test(domain_values)
                    if stress_report.resilience_score < 0.4:
                        interactions.append({
                            "type": "stress_vulnerability",
                            "description": f"Systemic stress vulnerability detected: resilience={stress_report.resilience_score:.0%}, worst case={stress_report.worst_case_scenario} (impact={stress_report.worst_case_impact:.2f})",
                            "severity": "high" if stress_report.resilience_score < 0.2 else "medium",
                        })
                except Exception:
                    pass

            if domain_values and len(domain_values) > 2:
                try:
                    red_team_result = self.red_team.red_team(domain_values, confidence=0.9)
                    if red_team_result.overall_vulnerability > 0.3:
                        interactions.append({
                            "type": "red_team_vulnerability",
                            "description": f"Red team challenge reveals {red_team_result.overall_vulnerability:.0%} vulnerability — confidence adjusted from 90% to {red_team_result.confidence_adjustment:.0%}",
                            "severity": "high" if red_team_result.overall_vulnerability > 0.5 else "medium",
                        })
                except Exception:
                    pass

            if domain_values and len(domain_values) > 3:
                try:
                    robust_result = self.robust_optimizer.optimize(domain_values, method=__import__('oracle.prediction.robust_optimizer', fromlist=['RobustMethod']).RobustMethod.MINIMAX)
                    if robust_result.price_of_robustness > 0.3:
                        interactions.append({
                            "type": "robust_optimization_insight",
                            "description": f"Price of robustness is {robust_result.price_of_robustness:.0%} — nominal={robust_result.nominal_performance:.3f}, worst={robust_result.worst_case_performance:.3f}",
                            "severity": "medium",
                        })
                except Exception:
                    pass

            if domain_values and len(domain_values) > 2:
                try:
                    preds_list = list(domain_values.values())
                    labels_list = [f"p_{i}" for i in range(len(preds_list))]
                    outcomes = [v > 0.5 for v in preds_list]
                    cal_result = self.ensemble_calibrator.calibrate(preds_list, outcomes)
                    if cal_result.calibration_curve.ece > 0.1:
                        interactions.append({
                            "type": "calibration_drift",
                            "description": f"Prediction calibration ECE={cal_result.calibration_curve.ece:.3f} — {cal_result.best_calibrator.value} calibrator recommended, reliability={cal_result.calibration_curve.reliability:.0%}",
                            "severity": "medium",
                        })
                except Exception:
                    pass

            if len(domain_values) >= 3:
                try:
                    horizon_preds = {}
                    horizon_scale = {"1day": 1.0, "1month": 1.02, "1year": 1.05, "5year": 1.1}
                    for h, scale in horizon_scale.items():
                        horizon_preds[h] = {k: round(v * scale, 4) for k, v in domain_values.items()}
                    tc_result = self.temporal_consistency.check(horizon_preds)
                    if not tc_result.is_consistent:
                        interactions.append({
                            "type": "temporal_inconsistency",
                            "description": f"Temporal inconsistency detected: score={tc_result.consistency_score:.0%}, smoothness={tc_result.smoothness_score:.0%}, {len(tc_result.violations)} violations",
                            "severity": "high" if tc_result.consistency_score < 0.5 else "medium",
                        })
                except Exception:
                    pass

            if domain_values and len(domain_values) > 2:
                try:
                    fate_result = self.predictive_fate.project_fate(domain_values, horizon_years=50)
                    if fate_result.collapse_probability > 0.15:
                        interactions.append({
                            "type": "fate_collapse_risk",
                            "description": f"Predictive fate engine warns: {fate_result.collapse_probability:.0%} collapse probability, {fate_result.dominant_phase.value} trajectory — {len(fate_result.divergences)} divergence points detected",
                            "severity": "high" if fate_result.collapse_probability > 0.3 else "medium",
                        })
                except Exception:
                    pass

            if domain_values and len(domain_values) > 2:
                try:
                    risk_result = self.existential_risk.assess(domain_values)
                    if risk_result.total_existential_probability > 0.05:
                        interactions.append({
                            "type": "existential_risk_signal",
                            "description": f"Existential risk probability={risk_result.total_existential_probability:.2%} — top threats: {', '.join(r.name for r in sorted(risk_result.risks, key=lambda r: r.annual_probability, reverse=True)[:3])}",
                            "severity": "high" if risk_result.total_existential_probability > 0.1 else "medium",
                        })
                except Exception:
                    pass

            if domain_values and len(domain_values) > 1:
                try:
                    prophecy = self.prophecy_synthesizer.prophesize(domain_values)
                    if prophecy.overall_confidence.value in ("oracle", "high"):
                        interactions.append({
                            "type": "prophecy_convergence",
                            "description": f"Oracle prophecy: {prophecy.overall_trajectory} trajectory with {prophecy.overall_confidence.value} confidence — {len(prophecy.critical_junctures)} junctures, {len(prophecy.divine_intervention_points)} intervention points",
                            "severity": "medium",
                        })
                except Exception:
                    pass

            result["cross_domain_interactions"] = interactions

        domain_data = {}
        domain_map = {
            "geopolitical": self.state.geopolitical_state,
            "climate": self.state.climate_state,
            "contagion": self.state.contagion_state,
            "demographics": self.state.demographics_state,
            "sentiment": self.state.sentiment_data,
            "epidemiology": self.state.epidemiology_state,
            "technology": self.state.technology_state,
            "social": self.state.social_state,
            "energy": self.state.energy_state,
            "commodities": self.state.commodities_state,
            "cyber": self.state.cyber_state,
            "water": self.state.water_state,
            "food": self.state.food_state,
            "political": self.state.political_state,
            "biodiversity": self.state.biodiversity_state,
            "pandemic": self.state.pandemic_state,
        }
        for dname, dstate in domain_map.items():
            if dstate is not None:
                if isinstance(dstate, dict):
                    domain_data[dname] = dstate
                elif hasattr(dstate, '__dict__'):
                    domain_data[dname] = {k: v for k, v in dstate.__dict__.items() if not k.startswith('_') and isinstance(v, (int, float, bool))}

        if domain_data:
            try:
                quality_reports = self.data_quality_scorer.score_all_domains(domain_data)
                quality_summary = self.data_quality_scorer.quality_summary()
                result["data_quality"] = quality_summary
                forecaster = QualityAdjustedForecaster(self.data_quality_scorer)
                for key, pred_val in result.items():
                    if isinstance(pred_val, Prediction):
                        try:
                            domain_name = pred_val.domain.value if hasattr(pred_val.domain, 'value') else str(pred_val.domain)
                            adj_lower, adj_upper = self.data_quality_scorer.adjust_confidence_interval(
                                domain_name, pred_val.confidence_interval[0], pred_val.confidence_interval[1])
                            pred_val.confidence_interval = (adj_lower, adj_upper)
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Data quality scoring error: {e}")
                pass

        return result

    def _ensemble_predictions(self, predictions: list[Prediction], confidence: ConfidenceLevel) -> list[Prediction]:
        from collections import defaultdict
        variable_preds = defaultdict(list)
        for p in predictions:
            key = f"{p.domain.value}:{p.variable}"
            variable_preds[key].append(p)

        ensemble = []

        for key, preds in variable_preds.items():
            if len(preds) < 2:
                continue

            z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645,
                 ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence]

            model_scores = {}
            for p in preds:
                model_scores[p.model_name] = self.config.get_model_weight(p.model_name)

            total_weight = sum(model_scores.values()) + 1e-10
            weighted_value = sum(p.value * model_scores.get(p.model_name, self.config.default_model_weight) for p in preds) / total_weight
            weighted_ci_lower = sum(p.confidence_interval[0] * model_scores.get(p.model_name, self.config.default_model_weight) for p in preds) / total_weight
            weighted_ci_upper = sum(p.confidence_interval[1] * model_scores.get(p.model_name, self.config.default_model_weight) for p in preds) / total_weight

            ensemble_p = preds[0]
            ensemble.append(Prediction(
                domain=ensemble_p.domain,
                variable=ensemble_p.variable,
                value=round(weighted_value, 4),
                confidence_interval=(round(weighted_ci_lower, 4), round(weighted_ci_upper, 4)),
                confidence_level=confidence,
                timestamp=datetime.now(),
                target_date=ensemble_p.target_date,
                model_name="oracle_ensemble",
                weights=model_scores,
                signals_used=["ensemble_of"] + [p.model_name for p in preds],
                methodology="weighted_ensemble",
                uncertainty_type="ensemble_weighted",
            ))

        return ensemble

    def predict_cross_scale(
        self, domain_states: Optional[Dict[str, float]] = None,
    ) -> Dict:
        result = self.scale_bridging.predict(domain_states=domain_states)
        scale_summary = {}
        for scale_name, preds in result.scale_predictions.items():
            scale_summary[scale_name] = [
                {
                    "prediction": p.prediction[:120],
                    "probability": round(p.probability, 3),
                    "confidence": round(p.confidence, 3),
                    "key_metric": p.key_metric,
                    "key_metric_current": p.key_metric_current,
                    "key_metric_target": p.key_metric_target,
                    "time_horizon": p.time_horizon,
                    "uncertainty_band": list(p.uncertainty_band),
                }
                for p in preds
            ]
        cascade_narratives = [
            {
                "narrative": c.narrative,
                "total_impact": round(c.total_impact, 3),
                "direction": c.dominant_direction,
                "chain_length": len(c.chain),
            }
            for c in result.cascades
        ]
        macro_traces = [
            {
                "source_scale": t.get("source_scale", ""),
                "target_scale": t.get("target_scale", ""),
                "source_domain": t.get("source_domain", ""),
                "target_domain": t.get("target_domain", ""),
                "coupling_strength": round(t.get("coupling_strength", 0), 3),
                "propagated_impact": round(t.get("propagated_impact", 0), 4),
            }
            for t in result.macro_to_micro_traces[:10]
        ]
        micro_traces = [
            {
                "source_scale": t.get("source_scale", ""),
                "target_scale": t.get("target_scale", ""),
                "source_domain": t.get("source_domain", ""),
                "target_domain": t.get("target_domain", ""),
                "coupling_strength": round(t.get("coupling_strength", 0), 3),
                "propagated_impact": round(t.get("propagated_impact", 0), 4),
            }
            for t in result.micro_to_macro_traces[:10]
        ]
        return {
            "scale_predictions": scale_summary,
            "cascades": cascade_narratives,
            "macro_to_micro_traces": macro_traces,
            "micro_to_macro_traces": micro_traces,
            "overall_confidence": round(result.overall_confidence, 3),
            "assessment": result.assessment,
            "n_scales": len(result.scale_predictions),
            "n_cascades": len(result.cascades),
        }

    def predict_neuroscience(self, domain_states: Optional[Dict[str, float]] = None) -> Dict:
        result = self.neuroscience_predictor.predict(domain_states=domain_states)
        predictions = [
            {
                "domain": p.domain.value if hasattr(p.domain, "value") else str(p.domain),
                "prediction": p.prediction[:120],
                "probability": round(p.probability, 3),
                "confidence": round(p.confidence, 3),
                "key_metric": p.key_metric,
                "key_metric_current": p.key_metric_current,
                "key_metric_target": p.key_metric_target,
                "time_horizon": p.time_horizon,
            }
            for p in result.predictions
        ]
        milestones = [
            {
                "milestone": m.milestone,
                "probability_by_year": {str(y): round(p, 3) for y, p in m.probability_by_year.items()},
                "key_barrier": m.key_barrier,
            }
            for m in result.bci_milestones
        ]
        return {
            "predictions": predictions,
            "bci_milestones": milestones,
            "breakthrough_probability_by_year": {str(y): round(p, 3) for y, p in result.breakthrough_probability_by_year.items()},
            "overall_assessment": result.overall_assessment,
            "n_predictions": len(result.predictions),
            "n_milestones": len(result.bci_milestones),
        }

    def predict_physics(self, domain_states: Optional[Dict[str, float]] = None) -> Dict:
        result = self.physics_predictor.predict(domain_states=domain_states)
        predictions = [
            {
                "domain": p.domain.value if hasattr(p.domain, "value") else str(p.domain),
                "prediction": p.prediction[:120],
                "probability": round(p.probability, 3),
                "key_metric": p.key_metric,
                "key_metric_current": p.key_metric_current,
                "key_metric_target": p.key_metric_target,
            }
            for p in result.predictions
        ]
        milestones = [
            {
                "milestone": qm.milestone,
                "year": qm.year,
                "probability": round(qm.probability, 3),
                "qubits": qm.qubits,
                "logical_qubits": qm.logical_qubits,
                "key_barrier": qm.key_barrier,
            }
            for qm in result.quantum_milestones
        ]
        return {
            "predictions": predictions,
            "quantum_milestones": milestones,
            "breakthrough_probability_by_year": {str(y): round(p, 3) for y, p in result.breakthrough_probability_by_year.items()},
            "assessment": result.assessment,
            "n_predictions": len(result.predictions),
            "n_milestones": len(result.quantum_milestones),
        }

    def predict_synthetic_bio(self, domain_states: Optional[Dict[str, float]] = None) -> Dict:
        result = self.synthetic_bio_predictor.predict(domain_states=domain_states)
        predictions = [
            {
                "domain": p.domain.value if hasattr(p.domain, "value") else str(p.domain),
                "prediction": p.prediction[:120],
                "probability": round(p.probability, 3),
                "clinical_stage": p.clinical_stage,
                "key_metric": p.key_metric,
                "key_metric_current": p.key_metric_current,
                "key_metric_target": p.key_metric_target,
            }
            for p in result.predictions
        ]
        pipeline = [
            {
                "name": drug.name,
                "modality": drug.modality,
                "target_disease": drug.target_disease,
                "phase": drug.phase,
                "probability_of_approval": round(drug.probability_of_approval, 3),
                "estimated_approval_year": drug.estimated_approval_year,
            }
            for drug in result.pipeline
        ]
        return {
            "predictions": predictions,
            "pipeline": pipeline,
            "breakthrough_probability_by_year": {str(y): round(p, 3) for y, p in result.breakthrough_probability_by_year.items()},
            "assessment": result.assessment,
            "n_predictions": len(result.predictions),
            "n_pipeline": len(result.pipeline),
        }

    def predict_all_domains(
        self, domain_states: Optional[Dict[str, float]] = None,
        current_data: Optional[Dict[str, float]] = None,
    ) -> Dict:
        cross_scale = self.predict_cross_scale(domain_states=domain_states)
        neuroscience = self.predict_neuroscience(domain_states=domain_states)
        physics = self.predict_physics(domain_states=domain_states)
        synthetic_bio = self.predict_synthetic_bio(domain_states=domain_states)
        causal = self.causal_event_predictor.predict(
            domain_states=domain_states, current_data=current_data
        )
        causal_events = [
            {
                "id": e.event_id,
                "category": e.category.value,
                "description": e.event_description[:120],
                "probability": round(e.probability, 3),
                "timing": e.timing_most_likely,
                "status": e.status.value,
            }
            for e in causal.events
        ]
        return {
            "cross_scale": cross_scale,
            "neuroscience": neuroscience,
            "physics": physics,
            "synthetic_bio": synthetic_bio,
            "causal_events": causal_events,
            "assessment": cross_scale.get("assessment", ""),
            "total_predictions": (
                cross_scale.get("n_scales", 0)
                + neuroscience.get("n_predictions", 0)
                + physics.get("n_predictions", 0)
                + synthetic_bio.get("n_predictions", 0)
                + len(causal_events)
            ),
        }

    def predict_human_outcome(
        self, domain_states: Optional[Dict[str, float]] = None,
        domains: Optional[List[HumanOutcomeDomain]] = None,
    ) -> Dict:
        result = self.human_outcome_predictor.predict(
            domain_states=domain_states, domains=domains
        )
        predictions = [
            {
                "domain": p.domain.value if hasattr(p.domain, "value") else str(p.domain),
                "prediction": p.prediction[:120],
                "probability": round(p.probability, 3),
                "confidence": round(p.confidence, 3),
                "time_horizon": p.time_horizon,
                "key_metric": p.key_metric,
                "key_metric_current": p.key_metric_current,
                "key_metric_target": p.key_metric_target,
                "intervention_potential": round(p.intervention_potential, 3),
                "intervention_type": p.intervention_type.value if hasattr(p.intervention_type, "value") else str(p.intervention_type),
                "individual_factors": p.individual_factors,
                "uncertainty_band": list(p.uncertainty_band),
            }
            for p in result.predictions
        ]
        outcomes = [
            {
                "domain": o.decision_domain,
                "option_a": o.option_a,
                "option_a_probability": round(o.option_a_probability, 3),
                "option_a_expected_value": round(o.option_a_expected_value, 3),
                "option_b": o.option_b,
                "option_b_probability": round(o.option_b_probability, 3),
                "option_b_expected_value": round(o.option_b_expected_value, 3),
                "bias_flags": o.bias_flags,
                "recommended": o.recommended,
                "confidence": round(o.confidence, 3),
            }
            for o in result.decision_outcomes
        ]
        genetic_risks = [
            {
                "condition": r.condition,
                "baseline_risk": round(r.lifetime_risk_baseline, 4),
                "risk_with_allele": round(r.lifetime_risk_with_risk_allele, 4),
                "risk_allele_frequency": round(r.risk_allele_frequency, 4),
                "odds_ratio": round(r.odds_ratio, 2),
                "modifiable_fraction": round(r.modifiable_fraction, 3),
                "intervention": r.intervention,
            }
            for r in result.genetic_risks
        ]
        return {
            "predictions": predictions,
            "decision_outcomes": outcomes,
            "genetic_risks": genetic_risks,
            "breakthrough_probability_by_year": {str(y): round(p, 3) for y, p in result.breakthrough_probability_by_year.items()},
            "overall_assessment": result.overall_assessment,
            "domains_covered": result.domains_covered,
            "n_predictions": result.n_predictions,
        }