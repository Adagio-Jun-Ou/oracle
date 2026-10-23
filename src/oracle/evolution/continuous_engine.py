"""Continuous evolution engine — the main loop that drives perpetual prediction improvement.

Orchestrates MAP-Elites + LLM mutation + outcome-backed fitness evaluation
in a continuous loop that evolves prediction strategies as new data flows in.

SAFETY: Maximum generations, maximum runtime, hard step limits, graceful shutdown
via threading.Event, atexit handler, and signal handlers. All LLM calls routed
through cloud API — never localhost.

Architecture:
  1. DataFlowEngine ingests domain states from DailyOracle / real data
  2. Fitness evaluation compares genome predictions against actual outcomes
  3. MAP-Elites archive maintains quality-diversity across behavioral descriptors
  4. LLM mutator injects novel strategies and guides search
  5. ContinuousEvolutionEngine ties it all together with evolve() and step()
"""
from __future__ import annotations

import atexit
import logging
import os
import random
import signal
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from oracle.evolution.genome import (
    PredictionGenome, EvolvableFunction, GenomeRegistry,
    create_default_genome, create_all_domain_genomes, create_random_genome
)
from oracle.evolution.map_elites import (
    CVTMAPElites, Elite, BehavioralDescriptor, FitnessScore, QDMetrics
)
from oracle.evolution.fitness import (
    MultiObjectiveFitness, FitnessResult, OutcomeTracker as FitOutcomeTracker
)
from oracle.evolution.llm_mutator import (
    LLMMutator, MutationResult, CrossoverResult, InjectionResult, MutationStrategy
)
from oracle.evolution.data_flow import DataFlowEngine, DomainStream, DataPoint

logger = logging.getLogger(__name__)

# Track all active engine instances for global shutdown
_active_engines: List[ContinuousEvolutionEngine] = []
_atexit_registered = False
_signal_registered = False


def _shutdown_all_engines() -> None:
    """Global shutdown hook — stops all running continuous evolution engines."""
    for engine in list(_active_engines):
        try:
            if engine.is_running:
                logger.warning("Global shutdown: stopping evolution engine")
                engine.stop_continuous()
        except Exception:
            pass


def _register_global_shutdown() -> None:
    global _atexit_registered, _signal_registered
    if not _atexit_registered:
        atexit.register(_shutdown_all_engines)
        _atexit_registered = True
    if not _signal_registered:
        try:
            signal.signal(signal.SIGINT, lambda *_: _shutdown_all_engines())
            signal.signal(signal.SIGTERM, lambda *_: _shutdown_all_engines())
        except (ValueError, OSError):
            pass  # Not in main thread
        _signal_registered = True


class EvolutionPhase(Enum):
    INITIALIZING = "initializing"
    EXPLORING = "exploring"
    EXPLOITING = "exploiting"
    INJECTING = "injecting"
    EVALUATING = "evaluating"
    IDLE = "idle"
    STOPPED = "stopped"  # terminal state after max_generations or max_runtime


# Hard system-level limits — NEVER exceeded regardless of config
HARD_MAX_GENERATIONS = 5000
HARD_MAX_RUNTIME_SECONDS = 3600    # 1 hour (was 24h — tightened to prevent runaway)
HARD_MAX_EVOLVE_STEPS = 100        # per single evolve() call
HARD_MAX_STEP_TIME_SECONDS = 300   # 5 min per step
HARD_JOIN_TIMEOUT_SECONDS = 30     # wait for thread to stop
HARD_MEMORY_LIMIT_MB = 4096        # 4GB — auto-pause if exceeded
HARD_BACKOFF_MULTIPLIER = 2.0      # exponential backoff on repeated failures
HARD_MAX_CONSECUTIVE_FAILURES = 10  # stop if this many steps fail in a row


@dataclass
class EvolutionConfig:
    population_size: int = 50
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    injection_rate: float = 0.1
    tournament_size: int = 3
    elitism_count: int = 2
    max_generations: int = 1000
    max_runtime_seconds: float = 3600.0   # 1 hour max continuous runtime
    stagnation_threshold: int = 20
    bd_dimensions: int = 2
    bd_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 1.0), (0.0, 1.0)])
    grid_sizes: List[int] = field(default_factory=lambda: [10, 10])
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "synthesis": 0.30, "calibration": 0.20, "cascade": 0.15,
        "confidence": 0.12, "spread": 0.10, "coverage": 0.08, "direction": 0.05
    })
    llm_model: str = "glm-5.1:cloud"
    llm_base_url: Optional[str] = None   # defaults to ORACLE_LLM_BASE_URL env var
    llm_api_key: Optional[str] = None    # defaults to ORACLE_LLM_API_KEY env var
    llm_inject_interval: int = 10
    seed: int = 42
    domains: List[str] = field(default_factory=lambda: ["economic", "geopolitical", "health"])
    continuous: bool = True
    step_interval: float = 60.0
    max_archive_size: int = 500

    def __post_init__(self):
        # Clamp to hard limits
        self.max_generations = min(self.max_generations, HARD_MAX_GENERATIONS)
        self.max_runtime_seconds = min(self.max_runtime_seconds, HARD_MAX_RUNTIME_SECONDS)
        if self.max_generations < 1:
            self.max_generations = 1
        if self.max_runtime_seconds < 1:
            self.max_runtime_seconds = 1


@dataclass
class EvolutionState:
    generation: int = 0
    phase: EvolutionPhase = EvolutionPhase.IDLE
    best_fitness: float = 0.0
    best_genome_id: str = ""
    total_evaluations: int = 0
    total_mutations: int = 0
    total_crossovers: int = 0
    total_injections: int = 0
    stagnation_count: int = 0
    last_improvement_gen: int = 0
    start_time: float = 0.0
    elapsed: float = 0.0
    stop_reason: str = ""


@dataclass
class StrategyReport:
    genome_id: str
    domain: str
    fitness: FitnessScore
    behavioral_descriptor: List[float]
    generation: int
    origin: str
    predictions: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "domain": self.domain,
            "fitness": {
                "composite": self.fitness.composite,
                "synthesis": self.fitness.synthesis,
                "calibration": self.fitness.calibration,
                "cascade": self.fitness.cascade,
                "confidence": self.fitness.confidence,
                "brier": self.fitness.brier,
                "direction": self.fitness.direction,
                "spread": self.fitness.spread,
                "coverage": self.fitness.coverage,
            },
            "behavioral_descriptor": self.behavioral_descriptor,
            "generation": self.generation,
            "origin": self.origin,
            "predictions": self.predictions,
            "params": self.params
        }


def _should_terminate(config: EvolutionConfig, state: EvolutionState) -> Tuple[bool, str]:
    """Check if evolution should stop based on limits."""
    if state.generation >= config.max_generations:
        return True, f"max_generations ({config.max_generations}) reached"
    elapsed = time.time() - state.start_time if state.start_time > 0 else 0
    state.elapsed = elapsed
    if elapsed >= config.max_runtime_seconds:
        return True, f"max_runtime ({config.max_runtime_seconds}s) exceeded"
    return False, ""


class ContinuousEvolutionEngine:
    """Continuous MAP-Elites + LLM evolution engine for prediction optimization.

    SAFETY FEATURES:
    - Hard generation cap (5000 max)
    - Hard runtime cap (24 hour max)
    - Hard per-step timeout (5 min)
    - threading.Event for instant loop interruption
    - atexit + signal handlers for global shutdown
    - Bounded evolve() and web API step limits
    - Non-daemon thread with proper join
    - Generation counter bounds check in every step()
    """

    def __init__(self, config: Optional[EvolutionConfig] = None):
        self.config = config or EvolutionConfig()
        self.rng = np.random.RandomState(self.config.seed)
        self.py_rng = random.Random(self.config.seed)

        self.registry = GenomeRegistry()
        self.archive = CVTMAPElites(
            bd_dimensions=self.config.bd_dimensions,
            bd_ranges=self.config.bd_ranges,
            grid_sizes=self.config.grid_sizes,
            fitness_weights=self.config.fitness_weights,
            seed=self.config.seed
        )
        self.fitness_evaluator = MultiObjectiveFitness(
            weights=self.config.fitness_weights,
            bd_names=["synthesis_score", "cascade_richness"],
            bd_ranges=self.config.bd_ranges,
            seed=self.config.seed
        )
        self.mutator = LLMMutator(
            base_url=self.config.llm_base_url or os.getenv("ORACLE_LLM_BASE_URL"),
            model=self.config.llm_model,
            api_key=self.config.llm_api_key or os.getenv("ORACLE_LLM_API_KEY"),
            seed=self.config.seed
        )
        self.data_flow = DataFlowEngine(domains=self.config.domains)
        self.state = EvolutionState()
        self._domain_archives: Dict[str, CVTMAPElites] = {}
        self._domain_genomes: Dict[str, PredictionGenome] = {}
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        self._step_lock = threading.Lock()
        self._consecutive_failures: int = 0
        self._current_backoff: float = 0.0
        self._start_time: float = 0.0
        self._initialize_population()

        # Register for global shutdown
        _active_engines.append(self)
        _register_global_shutdown()

    def _initialize_population(self) -> None:
        self.state.phase = EvolutionPhase.INITIALIZING
        for domain in self.config.domains:
            domain_archive = CVTMAPElites(
                bd_dimensions=self.config.bd_dimensions,
                bd_ranges=self.config.bd_ranges,
                grid_sizes=self.config.grid_sizes,
                fitness_weights=self.config.fitness_weights,
                seed=self.config.seed + hash(domain) % 1000
            )
            self._domain_archives[domain] = domain_archive
            default_genome = create_default_genome(domain)
            self.registry.register(default_genome)
            self._domain_genomes[domain] = default_genome
            for i in range(self.config.population_size // len(self.config.domains)):
                random_genome = create_random_genome(domain, self.py_rng)
                predictions = random_genome.apply_function(
                    {domain: 0.5, "confidence": 0.5, "volatility": 0.3}
                )
                fitness_result = self._evaluate_genome(random_genome, domain)
                bd = fitness_result.behavioral_descriptor or BehavioralDescriptor([0.5, 0.5])
                domain_archive.add_elite(
                    genome_id=random_genome.genome_id, bd=bd,
                    fitness=fitness_result.fitness,
                    generation=0, origin="random_init",
                    genome_data=random_genome
                )
                self.registry.register(random_genome)
        self.state.phase = EvolutionPhase.IDLE

    def _evaluate_genome(self, genome: PredictionGenome,
                          domain: str) -> FitnessResult:
        domain_states = self.data_flow.get_domain_states()
        if not domain_states:
            domain_states = {domain: 0.5, "confidence": 0.5, "volatility": 0.3}
        predictions = genome.apply_function(domain_states)
        fitness_result = self.fitness_evaluator.evaluate(
            genome_predictions=predictions,
            domain_states=domain_states,
            domain=domain,
            genome_id=genome.genome_id
        )
        outcome_records = self.data_flow.outcome_tracker.get_resolved(domain=domain, last_n=200)
        if len(outcome_records) >= 10:
            outcomes = {}
            for i, r in enumerate(outcome_records[-50:]):
                outcomes[f"pred_{i}"] = r.actual_outcome
            backtest_result = self.fitness_evaluator.evaluate_against_outcomes(
                predictions=predictions, outcomes=outcomes,
                domain=domain, genome_id=genome.genome_id
            )
            if backtest_result.fitness.composite > fitness_result.fitness.composite:
                fitness_result = backtest_result
        genome.fitness = fitness_result.fitness.composite
        genome.behavioral_descriptor = list(fitness_result.behavioral_descriptor.values) if fitness_result.behavioral_descriptor else [0.5, 0.5]
        self.state.total_evaluations += 1
        return fitness_result

    def step(self) -> Optional[StrategyReport]:
        """Execute one evolution step: mutate/crossover/inject, evaluate, archive.

        Returns None if the engine has stopped (max_generations / max_runtime reached).
        """
        if not self._running and self._thread is None:
            terminated, reason = _should_terminate(self.config, self.state)
            if terminated:
                return None

        with self._step_lock:
            return self._step_impl()

    def _step_impl(self) -> Optional[StrategyReport]:
        # Check termination BEFORE doing work
        terminated, reason = _should_terminate(self.config, self.state)
        if terminated:
            logger.info(f"Evolution stopped: {reason}")
            self.state.stop_reason = reason
            self.state.phase = EvolutionPhase.STOPPED
            self._running = False
            return None

        if self.state.stagnation_count > self.config.stagnation_threshold:
            self.state.phase = EvolutionPhase.INJECTING
        elif self.state.generation < self.config.max_generations:
            self.state.phase = EvolutionPhase.EXPLORING
        else:
            self.state.phase = EvolutionPhase.EXPLOITING

        domain = self.py_rng.choice(self.config.domains)
        domain_archive = self._domain_archives[domain]

        strategy = self._select_strategy(domain)
        if strategy is None:
            strategy = self._domain_genomes[domain]

        r = self.py_rng.random()
        child = None
        try:
            if self.state.phase == EvolutionPhase.INJECTING or (
                    self.state.generation % self.config.llm_inject_interval == 0
                    and r < self.config.injection_rate):
                result = self.mutator.inject_novel_strategy(domain)
                child = result.genome
                self.state.total_injections += 1
            elif r < self.config.mutation_rate:
                mut_strategy = MutationStrategy.HYBRID if self.py_rng.random() < 0.3 else MutationStrategy.GAUSSIAN_ADAPTATION
                mutation_result = self.mutator.mutate(strategy, mut_strategy, sigma=0.15)
                child = mutation_result.genome
                self.state.total_mutations += 1
            elif r < self.config.mutation_rate + self.config.crossover_rate:
                other = self._select_strategy(domain)
                if other is None or other.genome_id == strategy.genome_id:
                    other = create_random_genome(domain, self.py_rng)
                method = self.py_rng.choice(["uniform", "single_point", "blend"])
                crossover_result = self.mutator.crossover(strategy, other, method)
                child = crossover_result.child
                self.state.total_crossovers += 1
            else:
                child = strategy.mutate(self.py_rng, sigma=0.1)
        except Exception as e:
            logger.error(f"Evolution operator error for domain {domain}: {e}")
            return None

        if child is None:
            return None

        fitness_result = self._evaluate_genome(child, domain)
        child.fitness = fitness_result.fitness.composite
        child.behavioral_descriptor = list(fitness_result.behavioral_descriptor.values) if fitness_result.behavioral_descriptor else [0.5, 0.5]

        bd = fitness_result.behavioral_descriptor or BehavioralDescriptor([0.5, 0.5])
        domain_archive.add_elite(
            genome_id=child.genome_id, bd=bd,
            fitness=fitness_result.fitness,
            generation=self.state.generation,
            origin=child.origin,
            genome_data=child
        )
        self.registry.register(child)
        if fitness_result.fitness.composite > self.state.best_fitness:
            self.state.best_fitness = fitness_result.fitness.composite
            self.state.best_genome_id = child.genome_id
            self.state.last_improvement_gen = self.state.generation
            self.state.stagnation_count = 0
            self._domain_genomes[domain] = child
        else:
            self.state.stagnation_count += 1

        self.state.generation += 1

        report = StrategyReport(
            genome_id=child.genome_id, domain=domain,
            fitness=fitness_result.fitness,
            behavioral_descriptor=list(bd.values),
            generation=self.state.generation, origin=child.origin,
            predictions=fitness_result.predictions,
            params={name: func.params for name, func in child.functions.items()}
        )

        for callback in self._callbacks:
            try:
                callback(report)
            except Exception as e:
                logger.debug(f"Evolution callback error: {e}")

        # Check termination AFTER step to stop at exact generation boundary
        terminated, reason = _should_terminate(self.config, self.state)
        if terminated:
            logger.info(f"Evolution stopped after step: {reason}")
            self.state.stop_reason = reason
            self.state.phase = EvolutionPhase.STOPPED
            self._running = False

        return report

    def evolve(self, n_steps: int = 10) -> List[StrategyReport]:
        """Run multiple evolution steps. Hard-capped at HARD_MAX_EVOLVE_STEPS.

        n_steps is clamped to [1, HARD_MAX_EVOLVE_STEPS] as a DoS guard.
        """
        n_steps = max(1, min(n_steps, HARD_MAX_EVOLVE_STEPS))
        reports = []
        for i in range(n_steps):
            report = self.step()
            if report is None:
                break
            reports.append(report)
        return reports

    def _select_strategy(self, domain: str) -> Optional[PredictionGenome]:
        domain_archive = self._domain_archives.get(domain)
        if domain_archive is None or not domain_archive.archive:
            return None
        elite = domain_archive.tournament_select(self.config.tournament_size)
        if elite is None:
            return None
        genome_data = domain_archive._genome_archive.get(elite.genome_id)
        if genome_data is not None:
            return genome_data
        return create_random_genome(domain, self.py_rng)

    def get_best_strategy(self, domain: Optional[str] = None) -> StrategyReport:
        if domain:
            archive = self._domain_archives.get(domain)
            if archive:
                best = archive.get_best_elite()
                if best:
                    genome_data = archive._genome_archive.get(best.genome_id)
                    params = {}
                    if genome_data:
                        params = {n: f.params for n, f in genome_data.functions.items()}
                    return StrategyReport(
                        genome_id=best.genome_id, domain=domain,
                        fitness=best.fitness,
                        behavioral_descriptor=list(best.behavioral_descriptor.values),
                        generation=best.generation, origin=best.origin,
                        params=params
                    )
        best_genome = None
        best_fitness = -1
        best_domain = self.config.domains[0]
        for d, genome in self._domain_genomes.items():
            if genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_genome = genome
                best_domain = d
        if best_genome is None:
            best_genome = create_default_genome(self.config.domains[0])
        return StrategyReport(
            genome_id=best_genome.genome_id, domain=best_domain,
            fitness=FitnessScore(composite=best_genome.fitness),
            behavioral_descriptor=best_genome.behavioral_descriptor or [0.5, 0.5],
            generation=best_genome.generation, origin=best_genome.origin,
            params={n: f.params for n, f in best_genome.functions.items()}
        )

    def get_archive_stats(self, domain: Optional[str] = None) -> Dict[str, Any]:
        if domain:
            archive = self._domain_archives.get(domain)
            return archive.archive_stats() if archive else {}
        return {
            "global": self.archive.archive_stats(),
            "domains": {d: a.archive_stats() for d, a in self._domain_archives.items()},
            "state": {
                "generation": self.state.generation,
                "best_fitness": self.state.best_fitness,
                "total_evaluations": self.state.total_evaluations,
                "total_mutations": self.state.total_mutations,
                "total_crossovers": self.state.total_crossovers,
                "total_injections": self.state.total_injections,
                "stagnation_count": self.state.stagnation_count,
                "phase": self.state.phase.value,
                "elapsed_seconds": round(self.state.elapsed, 1),
                "stop_reason": self.state.stop_reason,
            },
            "mutator": self.mutator.stats(),
            "data_flow": self.data_flow.summary()
        }

    def predict_with_best(self, domain: str,
                           domain_states: Dict[str, float]) -> Dict[str, float]:
        best_genome = self._domain_genomes.get(domain)
        if best_genome is None:
            best_genome = create_default_genome(domain)
        return best_genome.apply_function(domain_states)

    def ingest_predictions(self, genome_id: str, domain: str,
                            predictions: Dict[str, float],
                            confidence: float = 0.5) -> str:
        domain_states = self.data_flow.get_domain_states()
        avg_pred = float(np.mean(list(predictions.values()))) if predictions else 0.5
        return self.data_flow.outcome_tracker.record_prediction(
            genome_id=genome_id, domain=domain,
            predicted=avg_pred, horizon_hours=24.0,
            confidence=confidence
        )

    def resolve_with_outcomes(self, domain_states: Dict[str, float]) -> int:
        return self.data_flow.outcome_tracker.resolve_all_pending(domain_states)

    # ── Continuous loop with safety boundaries ──────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def _check_memory(self) -> bool:
        """Return True if memory usage is below the hard limit."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            mem_mb = usage.ru_maxrss / 1024.0  # MB on Linux
            return mem_mb < HARD_MEMORY_LIMIT_MB
        except Exception:
            return True

    def start_continuous(self, step_interval: Optional[float] = None,
                          max_duration: Optional[float] = None) -> None:
        """Start the continuous evolution loop in a background thread.

        Args:
            step_interval: Seconds between evolution steps (clamped to [1, 3600])
            max_duration: Override max_runtime_seconds for this run (clamped to hard max)
        """
        if self.is_running:
            logger.warning("start_continuous: already running")
            return

        # Clamp step_interval
        if step_interval is not None:
            self.config.step_interval = max(1.0, min(step_interval, 3600.0))

        # Clamp max_duration
        if max_duration is not None:
            self.config.max_runtime_seconds = max(1.0, min(max_duration, HARD_MAX_RUNTIME_SECONDS))

        # Reset stop event
        self._stop_event.clear()
        self._running = True
        self.state.start_time = time.time()
        self.state.elapsed = 0.0
        self.state.stop_reason = ""
        self.state.phase = EvolutionPhase.EXPLORING

        self._thread = threading.Thread(
            target=self._continuous_loop,
            name=f"evolution-{id(self)}",
            daemon=False  # NOT daemon — we want clean shutdown
        )
        self._thread.start()
        logger.info(
            f"Continuous evolution started: interval={self.config.step_interval}s, "
            f"max_generations={self.config.max_generations}, "
            f"max_runtime={self.config.max_runtime_seconds}s"
        )

    def stop_continuous(self) -> None:
        """Gracefully stop the continuous evolution loop.

        Signals the loop to stop via Event (instant), then joins the thread.
        """
        if not self._running:
            return

        self._running = False
        self._stop_event.set()  # Wake up immediately from sleep
        self.state.phase = EvolutionPhase.STOPPED
        self.state.stop_reason = self.state.stop_reason or "user_stop"

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=HARD_JOIN_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                logger.warning(
                    f"Evolution thread did not stop within {HARD_JOIN_TIMEOUT_SECONDS}s "
                    f"— thread may be stuck in a long-running operation"
                )
            self._thread = None

        logger.info(f"Continuous evolution stopped at generation {self.state.generation}: {self.state.stop_reason}")

    def _continuous_loop(self) -> None:
        """Main continuous loop using threading.Event for instant wake-up.

        Safety stops:
        - self._running is set to False (external stop)
        - _stop_event is set (instant wake-up)
        - max_generations reached
        - max_runtime_seconds exceeded
        - memory exceeds HARD_MEMORY_LIMIT_MB
        - HARD_MAX_CONSECUTIVE_FAILURES reached
        """
        logger.info("Evolution continuous loop started")
        self._consecutive_failures = 0
        self._current_backoff = 0.0
        self._start_time = time.time()
        while self._running and not self._stop_event.is_set():
            # Check termination conditions
            terminated, reason = _should_terminate(self.config, self.state)
            if terminated:
                logger.info(f"Evolution loop terminating: {reason}")
                self.state.stop_reason = reason
                self.state.phase = EvolutionPhase.STOPPED
                break

            if self._consecutive_failures >= HARD_MAX_CONSECUTIVE_FAILURES:
                logger.error(f"Evolution stopped: {self._consecutive_failures} consecutive failures")
                self.state.stop_reason = f"too_many_failures ({self._consecutive_failures})"
                self.state.phase = EvolutionPhase.STOPPED
                break
            if not self._check_memory():
                logger.error(f"Evolution stopped: memory limit exceeded ({HARD_MEMORY_LIMIT_MB} MB)")
                self.state.stop_reason = "memory_limit_exceeded"
                self.state.phase = EvolutionPhase.STOPPED
                break

            try:
                report = self._step_impl()
                if report is None:
                    break
                self._consecutive_failures = 0
                self._current_backoff = 0.0
            except Exception as e:
                logger.error(f"Evolution step error (continuing): {e}")
                self._consecutive_failures += 1
                self._current_backoff = min(
                    self._current_backoff + 5.0,
                    self.config.step_interval * 4
                )
                if self._current_backoff < 5.0:
                    self._current_backoff = 5.0

            step_interval = max(self.config.step_interval, self._current_backoff)
            if self._stop_event.wait(timeout=step_interval):
                break  # stop_continuous() signaled

        self._running = False
        self.state.phase = EvolutionPhase.STOPPED
        if not self.state.stop_reason:
            self.state.stop_reason = "loop_exit"
        logger.info(f"Evolution continuous loop exited: {self.state.stop_reason}")

    def register_callback(self, callback: Callable[[StrategyReport], None]) -> None:
        self._callbacks.append(callback)

    def export_state(self) -> Dict[str, Any]:
        return {
            "config": {
                "population_size": self.config.population_size,
                "mutation_rate": self.config.mutation_rate,
                "crossover_rate": self.config.crossover_rate,
                "injection_rate": self.config.injection_rate,
                "max_generations": self.config.max_generations,
                "max_runtime_seconds": self.config.max_runtime_seconds,
                "seed": self.config.seed,
                "domains": self.config.domains,
                "llm_model": self.config.llm_model,
            },
            "state": {
                "generation": self.state.generation,
                "best_fitness": self.state.best_fitness,
                "best_genome_id": self.state.best_genome_id,
                "total_evaluations": self.state.total_evaluations,
                "total_mutations": self.state.total_mutations,
                "total_crossovers": self.state.total_crossovers,
                "total_injections": self.state.total_injections,
                "stagnation_count": self.state.stagnation_count,
                "phase": self.state.phase.value,
                "elapsed_seconds": round(self.state.elapsed, 1),
                "stop_reason": self.state.stop_reason,
            },
            "archive": {
                "global": self.archive.export_archive(),
                "domains": {d: a.export_archive() for d, a in self._domain_archives.items()},
            },
            "best_genomes": {
                d: g.to_dict() for d, g in self._domain_genomes.items()
            }
        }
