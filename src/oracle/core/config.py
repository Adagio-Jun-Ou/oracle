from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class OracleConfig:
    project_root: Path = field(default_factory=lambda: Path.home() / "oracle")
    data_cache_dir: Path = field(default_factory=lambda: Path.home() / "oracle" / "data_cache")
    max_concurrent_ingestion: int = 10
    default_confidence: float = 0.95
    monte_carlo_simulations: int = 10000
    ensemble_weights: dict = field(default_factory=lambda: {
        "deterministic": 0.3,
        "stochastic": 0.3,
        "ml_ensemble": 0.25,
        "bayesian": 0.15,
    })
    model_weights: dict = field(default_factory=lambda: {
        "dalio": 0.30,
        "economic": 0.25,
        "physics": 0.10,
        "statistical": 0.20,
        "buffett": 0.15,
        "monte_carlo": 0.10,
        "bayesian": 0.10,
        "conformal": 0.08,
        "causal": 0.08,
        "probabilistic": 0.10,
        "bsts": 0.10,
        "correlation_dynamics": 0.08,
        "shapley": 0.07,
        "online_learner": 0.06,
        "kan": 0.08,
        "advanced_ensemble": 0.10,
        "conditional_var": 0.07,
        "double_ml": 0.06,
        "bayesian_optimizer": 0.05,
        "gaussian_process": 0.08,
        "evt": 0.07,
        "wavelet": 0.06,
        "vine_copula": 0.07,
        "neural_ode": 0.08,
        "normalizing_flow": 0.06,
        "survival": 0.05,
        "information_theory": 0.07,
        "topological": 0.06,
        "state_space": 0.08,
        "rnn": 0.08,
        "hmm": 0.07,
        "copula_garch": 0.07,
        "causal_discovery": 0.06,
        "tft": 0.09,
        "rl": 0.07,
        "koopman": 0.06,
        "abm": 0.08,
        "symbolic": 0.05,
        "geopolitical": 0.15,
        "climate": 0.15,
        "sentiment": 0.10,
        "contagion": 0.10,
        "demographics": 0.08,
        "epidemiology": 0.12,
        "technology": 0.10,
        "social": 0.10,
        "energy": 0.10,
        "commodities": 0.10,
        "infrastructure": 0.08,
        "supply_chain": 0.08,
        "cyber": 0.08,
        "water": 0.08,
        "food": 0.08,
        "political": 0.08,
        "biodiversity": 0.08,
        "pandemic": 0.10,
        "space": 0.06,
        "mental_health": 0.08,
        "housing": 0.08,
        "seismic": 0.06,
        "solar": 0.06,
        "oceanic": 0.06,
        "neo": 0.05,
        "atmospheric": 0.06,
        "ecological": 0.06,
        "paleoclimatology": 0.05,
        "cosmic_radiation": 0.04,
        "gravitational": 0.04,
        "compound_hazard": 0.06,
        "fate": 0.10,
        "personal_life": 0.10,
        "prescience": 0.08,
        "evolutionary": 0.07,
        "learning": 0.07,
        "physical_constants": 0.03,
        "ai_integration": 0.05,
    })
    default_model_weight: float = 0.1
    dalio_cycle_periods: dict = field(default_factory=lambda: {
        "short_term_debt": (5, 8),
        "long_term_debt": (75, 100),
        "productivity": (50, 75),
    })
    buffett_margin_of_safety: float = 0.25
    correlation_threshold: float = 0.3
    signal_decay_rate: float = 0.05
    prediction_horizon_days: int = 365 * 10
    prediction_timeout_seconds: float = 300.0
    max_predictions_per_domain: int = 50
    log_level: str = "WARNING"
    fred_api_key: Optional[str] = None

    def __post_init__(self):
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=getattr(logging, self.log_level.upper(), logging.WARNING))

    def get_model_weight(self, model_name: str) -> float:
        key = model_name.lower().replace("_", "").replace("-", "").replace(" ", "")
        for k, v in self.model_weights.items():
            if k.lower().replace("_", "").replace("-", "").replace(" ", "") in key:
                return v
        return self.default_model_weight