# -*- coding: utf-8 -*-
"""
Unified Oracle Dashboard — aggregates all prediction engines into a single coherent view.

Part of Oracle Prediction Engine v0.5.5.
"""

import datetime
from typing import Dict, Any, Optional

from oracle.prediction.daily_oracle import DailyOracle
from oracle.prediction.engine import OracleEngine


class OracleDashboard:
    """
    Aggregates state from all Oracle subsystems into a unified dashboard.
    Serves as the single source of truth for the web dashboard UI.
    """

    def __init__(self, engine: OracleEngine):
        self.engine = engine

    def collect(self, use_real_data: bool = True) -> Dict[str, Any]:
        """
        Collect current state from all Oracle subsystems.

        Returns a dict with keys:
        - evolution: MAP-Elites population + fitness
        - consciousness: emotional pulse + cognitive landscape + fusion
        - collective: hive mind + memetic pressure + narrative
        - predictions: core prediction engine outputs
        - daily: daily oracle readings
        - tracker: temporal consciousness tracker
        - action: predictive action engine
        - timestamp: ISO timestamp
        """

        result: Dict[str, Any] = {
            "version": "0.5.5",
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # ─── Evolution ───
        result["evolution"] = self._collect_evolution()

        # ─── Consciousness ───
        result["consciousness"] = self._collect_consciousness()

        # ─── Daily Oracle ───
        result["daily_oracle"] = self._collect_daily_oracle(use_real_data)

        # ─── Real-data sentiment injection ───
        self._ingest_daily_sentiment(result["daily_oracle"])

        # ─── Predictions Core ───
        result["predictions_status"] = self._collect_predictions()

        # ─── Tracker & Action ───
        result["tracker"] = self._collect_tracker()
        result["action"] = self._collect_action()

        return result

    def _collect_evolution(self) -> Dict[str, Any]:
        evo = getattr(self.engine, "evo_engine", None)
        if evo is None:
            return {"status": "not_initialized"}

        try:
            from oracle.engine import EvolutionConfig
            config = EvolutionConfig(
                population_size=30,
                domains=["economic", "geopolitical", "health"],
            )
            evo.configure(config)
            grid = evo.get_map_elites_grid()
            occupied = sum(1 for cell in grid.values() if cell["genome"] is not None)
            total = len(grid)
            best = evo.get_best_genome()

            # Serialize numpy types safely
            def _safe(val):
                if hasattr(val, "item"):
                    return val.item()
                return val

            return {
                "status": "active",
                "grid_size": total,
                "occupied_cells": occupied,
                "coverage": round(occupied / max(total, 1), 3),
                "best_fitness": _safe(best.get("fitness")) if best else None,
                "mutation_rate": evo.mutation_rate,
                "generation": evo.generation_count,
                "genes_evaluated": getattr(evo, "total_genes_evaluated", 0),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _collect_consciousness(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # Heart pulse
        heart = getattr(self.engine, "emotional_pulse", None)
        if heart:
            try:
                snapshot = heart.get_emotional_snapshot()
                out["emotional_pulse"] = {
                    "dominant": snapshot.get("dominant", "unknown"),
                    "intensity": round(snapshot.get("intensity", 0.0), 3),
                    "entropy": round(snapshot.get("entropy", 1.0), 3),
                    "contagion": {k: round(v, 3) for k, v in snapshot.get("contagion_vectors", {}).items()},
                }
            except Exception:
                pass

        # Cognitive landscape
        mind = getattr(self.engine, "cognitive_landscape", None)
        if mind:
            try:
                out["cognitive_landscape"] = {
                    "dominant_narrative": getattr(mind, "dominant_narrative", "unknown"),
                    "belief_coherence": round(getattr(mind, "belief_coherence", 0.0), 3),
                    "attention_focus": getattr(mind, "attention_focus", None),
                }
            except Exception:
                pass

        # Fusion
        fusion = getattr(self.engine, "consciousness_fusion", None)
        if fusion:
            try:
                out["fusion"] = {
                    "alignment": round(getattr(fusion, "last_alignment", 0.0), 3),
                    "will_strength": round(getattr(fusion, "collective_will_strength", 0.0), 3),
                    "tipping_point_probability": round(getattr(fusion, "tipping_point_probability", 0.0), 3),
                }
            except Exception:
                pass

        # Hive mind
        hive = getattr(self.engine, "hive_mind", None)
        if hive:
            try:
                consensus = hive.get_consensus()
                out["hive_mind"] = {
                    "wisdom_score": round(consensus.get("wisdom_score", 0.0), 3),
                    "latent_topics": len(consensus.get("latent_topics", [])),
                    "prediction_market_score": round(consensus.get("prediction_market_score", 0.0), 3),
                }
            except Exception:
                pass

        # Collective engine
        collective = getattr(self.engine, "collective_consciousness", None)
        if collective:
            try:
                dashboard = collective.get_dashboard()
                out["collective_dashboard"] = {
                    "layers": len(dashboard.get("layers", [])),
                    "overall_intensity": round(dashboard.get("overall_intensity", 0.0), 3),
                }
            except Exception:
                pass

        # Narrative evolution
        narrative = getattr(self.engine, "narrative_evolution", None)
        if narrative:
            try:
                summary = narrative.get_dominant_memes()
                out["narrative_evolution"] = {
                    "active_memes": len(summary),
                    "viral_memes": sum(1 for m in summary if m.get("viral", False)),
                }
            except Exception:
                pass

        return out or {"status": "not_initialized"}

    def _collect_daily_oracle(self, use_real_data: bool) -> Dict[str, Any]:
        daily = getattr(self.engine, "daily_oracle", None)
        if daily is None:
            return {"status": "not_initialized"}

        try:
            if use_real_data:
                readings = daily.get_daily_reading()
            else:
                readings = daily.get_current_readings()
            return {
                "status": "active",
                "timestamp": getattr(daily, "last_update", None),
                "readings": [
                    {
                        "domain": r.get("domain", "").strip().lower(),
                        "anomaly_score": round(float(r.get("anomaly_score", r.get("severity", 0.0))), 3),
                        "confidence": round(float(r.get("confidence", 0.0)), 3),
                    }
                    for r in readings
                ],
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _ingest_daily_sentiment(self, daily_oracle: Dict[str, Any]) -> None:
        """Pipe daily oracle readings into the emotional pulse engine."""
        if daily_oracle.get("status") != "active":
            return
        heart = getattr(self.engine, "emotional_pulse", None)
        if not hasattr(heart, "ingest_domain_signal"):
            return
        for reading in daily_oracle.get("readings", []):
            domain = reading.get("domain", "").strip().lower()
            if not domain:
                continue
            anomaly = float(reading.get("anomaly_score", 0.0))
            # Convert anomaly to sentiment (higher anomaly = more negative)
            sentiment = max(-1.0, min(1.0, -anomaly * 2.0))
            try:
                heart.ingest_domain_signal(domain, sentiment, anomaly)
            except Exception:
                pass

    def _collect_predictions(self) -> Dict[str, Any]:
        pred = getattr(self.engine, "prophecy_narrator", None)
        if pred is None:
            return {"status": "not_initialized"}

        try:
            return {
                "status": "active",
                "narrator_available": True,
            }
        except Exception:
            return {"status": "error"}

    def _collect_tracker(self) -> Dict[str, Any]:
        tracker = getattr(self.engine, "consciousness_tracker", None)
        if tracker is None:
            return {"status": "not_initialized"}

        try:
            forecast = tracker.get_consciousness_forecast()
            return {
                "status": "active",
                "current_regime": forecast.get("regime", "unknown"),
                "regime_stability": round(forecast.get("stability", 0.0), 3),
                "phase": forecast.get("phase", "unknown"),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _collect_action(self) -> Dict[str, Any]:
        action = getattr(self.engine, "predictive_action", None)
        if action is None:
            return {"status": "not_initialized"}

        try:
            recs = action.get_recommendations()
            return {
                "status": "active",
                "recommendations": len(recs.get("actions", [])),
                "confidence_tier": recs.get("confidence_tier", "none"),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
