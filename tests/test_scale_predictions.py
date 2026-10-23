"""Tests for scale-bridging prediction engine, neuroscience, physics, and synthetic biology modules."""
import pytest
import numpy as np


class TestScaleBridgingPredictionEngine:
    def test_predict_all_scales(self):
        from oracle.prediction.scale_bridging import ScaleBridgingPredictionEngine, PredictionScale
        engine = ScaleBridgingPredictionEngine(seed=42)
        result = engine.predict()
        assert len(result.scale_predictions) == 9
        assert sum(len(v) for v in result.scale_predictions.values()) >= 20

    def test_predict_specific_scales(self):
        from oracle.prediction.scale_bridging import ScaleBridgingPredictionEngine, PredictionScale
        engine = ScaleBridgingPredictionEngine(seed=42)
        result = engine.predict(scales=[PredictionScale.COSMOS, PredictionScale.MOLECULAR])
        assert len(result.scale_predictions) == 2
        total = sum(len(v) for v in result.scale_predictions.values())
        assert total >= 2

    def test_prediction_structure(self):
        from oracle.prediction.scale_bridging import ScaleBridgingPredictionEngine, ScalePrediction
        engine = ScaleBridgingPredictionEngine(seed=42)
        result = engine.predict()
        for scale_name, preds in result.scale_predictions.items():
            for pred in preds[:2]:
                assert isinstance(pred, ScalePrediction)
                assert 0 < pred.probability < 1
                assert 0 < pred.confidence <= 1
                assert pred.prediction
                assert len(pred.evidence_base) >= 2
                assert pred.falsification_criterion
                assert pred.method
                assert pred.base_rate_source
                assert pred.uncertainty_band[0] < pred.uncertainty_band[1]

    def test_cascades(self):
        from oracle.prediction.scale_bridging import ScaleBridgingPredictionEngine
        engine = ScaleBridgingPredictionEngine(seed=42)
        result = engine.predict()
        assert len(result.cascades) >= 1
        for cascade in result.cascades:
            assert cascade.total_impact >= 0
            assert cascade.narrative
            assert cascade.dominant_direction in ("amplifying", "dampening")

    def test_scale_couplings(self):
        from oracle.prediction.scale_bridging import ScaleBridgingPredictionEngine, ScaleCoupling, ScaleCouplingType
        engine = ScaleBridgingPredictionEngine(seed=42)
        result = engine.predict()
        assert len(result.couplings) >= 10
        for coupling in result.couplings:
            assert isinstance(coupling, ScaleCoupling)
            assert -1 <= coupling.strength <= 1
            assert isinstance(coupling.coupling_type, ScaleCouplingType)

    def test_macro_to_micro_traces(self):
        from oracle.prediction.scale_bridging import ScaleBridgingPredictionEngine
        engine = ScaleBridgingPredictionEngine(seed=42)
        result = engine.predict()
        assert len(result.macro_to_micro_traces) >= 1
        for trace in result.macro_to_micro_traces:
            assert "source_scale" in trace
            assert "target_scale" in trace
            assert "coupling_strength" in trace

    def test_confidence(self):
        from oracle.prediction.scale_bridging import ScaleBridgingPredictionEngine
        engine = ScaleBridgingPredictionEngine(seed=42)
        result = engine.predict()
        assert 0 < result.overall_confidence <= 1

    def test_domain_states_affect_predictions(self):
        from oracle.prediction.scale_bridging import ScaleBridgingPredictionEngine
        engine = ScaleBridgingPredictionEngine(seed=42)
        result_default = engine.predict()
        result_tech_boost = engine.predict(domain_states={"technology": 0.9, "health": 0.8})
        # Micro-scale predictions should be boosted with high tech
        micro_probs_default = [p.probability for scale, preds in result_default.scale_predictions.items()
                               for p in preds if scale in ("organism", "cellular", "molecular")]
        micro_probs_boost = [p.probability for scale, preds in result_tech_boost.scale_predictions.items()
                             for p in preds if scale in ("organism", "cellular", "molecular")]
        if micro_probs_default and micro_probs_boost:
            assert np.mean(micro_probs_boost) >= np.mean(micro_probs_default) * 0.95


class TestNeurosciencePredictor:
    def test_predict_returns_result(self):
        from oracle.prediction.neuroscience import NeurosciencePredictor
        ns = NeurosciencePredictor(seed=42)
        result = ns.predict()
        assert len(result.predictions) >= 5
        assert len(result.bci_milestones) >= 3

    def test_prediction_structure(self):
        from oracle.prediction.neuroscience import NeurosciencePredictor, NeurosciencePrediction, NeuroscienceDomain
        ns = NeurosciencePredictor(seed=42)
        result = ns.predict()
        for pred in result.predictions:
            assert isinstance(pred, NeurosciencePrediction)
            assert isinstance(pred.domain, NeuroscienceDomain)
            assert 0 < pred.probability < 1
            assert len(pred.evidence) >= 2
            assert pred.falsification
            assert pred.implication

    def test_bci_milestones(self):
        from oracle.prediction.neuroscience import NeurosciencePredictor, BCIMilestone, BCIMaturity
        ns = NeurosciencePredictor(seed=42)
        result = ns.predict()
        for milestone in result.bci_milestones:
            assert isinstance(milestone, BCIMilestone)
            assert isinstance(milestone.maturity, BCIMaturity)
            assert len(milestone.probability_by_year) >= 3
            assert milestone.key_barrier

    def test_breakthrough_probability(self):
        from oracle.prediction.neuroscience import NeurosciencePredictor
        ns = NeurosciencePredictor(seed=42)
        result = ns.predict()
        assert 2025 in result.breakthrough_probability_by_year
        assert 2030 in result.breakthrough_probability_by_year
        for year, prob in result.breakthrough_probability_by_year.items():
            assert 0 < prob <= 1

    def test_domain_summary(self):
        from oracle.prediction.neuroscience import NeurosciencePredictor
        ns = NeurosciencePredictor(seed=42)
        result = ns.predict()
        assert len(result.domain_summary) >= 3

    def test_bci_predictions_exist(self):
        from oracle.prediction.neuroscience import NeurosciencePredictor, NeuroscienceDomain
        ns = NeurosciencePredictor(seed=42)
        result = ns.predict()
        bci_preds = [p for p in result.predictions if p.domain == NeuroscienceDomain.BCI]
        assert len(bci_preds) >= 2


class TestPhysicsPredictor:
    def test_predict_returns_result(self):
        from oracle.prediction.physics_forecast import PhysicsPredictor
        phys = PhysicsPredictor(seed=42)
        result = phys.predict()
        assert len(result.predictions) >= 5
        assert len(result.quantum_milestones) >= 4

    def test_prediction_structure(self):
        from oracle.prediction.physics_forecast import PhysicsPredictor, PhysicsPrediction, PhysicsDomain
        phys = PhysicsPredictor(seed=42)
        result = phys.predict()
        for pred in result.predictions:
            assert isinstance(pred, PhysicsPrediction)
            assert isinstance(pred.domain, PhysicsDomain)
            assert 0 < pred.probability < 1
            assert len(pred.evidence) >= 2
            assert pred.paradigm_impact

    def test_quantum_milestones(self):
        from oracle.prediction.physics_forecast import PhysicsPredictor, QuantumMilestone
        phys = PhysicsPredictor(seed=42)
        result = phys.predict()
        for qm in result.quantum_milestones:
            assert isinstance(qm, QuantumMilestone)
            assert qm.qubits > 0
            assert qm.probability > 0

    def test_breakthrough_probability(self):
        from oracle.prediction.physics_forecast import PhysicsPredictor
        phys = PhysicsPredictor(seed=42)
        result = phys.predict()
        assert 2025 in result.breakthrough_probability_by_year
        assert result.breakthrough_probability_by_year[2030] > result.breakthrough_probability_by_year[2025]

    def test_materials_predictions(self):
        from oracle.prediction.physics_forecast import PhysicsPredictor, MaterialsPrediction
        phys = PhysicsPredictor(seed=42)
        result = phys.predict()
        assert len(result.materials_predictions) >= 2


class TestSyntheticBioPredictor:
    def test_predict_returns_result(self):
        from oracle.prediction.synthetic_bio import SyntheticBioPredictor
        bio = SyntheticBioPredictor(seed=42)
        result = bio.predict()
        assert len(result.predictions) >= 5
        assert len(result.pipeline) >= 3

    def test_prediction_structure(self):
        from oracle.prediction.synthetic_bio import SyntheticBioPredictor, BioPrediction, BioDomain
        bio = SyntheticBioPredictor(seed=42)
        result = bio.predict()
        for pred in result.predictions:
            assert isinstance(pred, BioPrediction)
            assert isinstance(pred.domain, BioDomain)
            assert 0 < pred.probability < 1
            assert len(pred.evidence) >= 2
            assert pred.clinical_stage
            assert pred.patient_impact

    def test_therapeutic_pipeline(self):
        from oracle.prediction.synthetic_bio import SyntheticBioPredictor, TherapeuticPipeline
        bio = SyntheticBioPredictor(seed=42)
        result = bio.predict()
        for drug in result.pipeline:
            assert isinstance(drug, TherapeuticPipeline)
            assert drug.probability_of_approval > 0
            assert drug.estimated_approval_year >= 2026
            assert len(drug.key_evidence) >= 2

    def test_crispr_predictions(self):
        from oracle.prediction.synthetic_bio import SyntheticBioPredictor, BioDomain
        bio = SyntheticBioPredictor(seed=42)
        result = bio.predict()
        crispr_preds = [p for p in result.predictions if p.domain == BioDomain.CRISPR]
        assert len(crispr_preds) >= 1

    def test_breakthrough_probability(self):
        from oracle.prediction.synthetic_bio import SyntheticBioPredictor
        bio = SyntheticBioPredictor(seed=42)
        result = bio.predict()
        assert 2025 in result.breakthrough_probability_by_year
        assert 2030 in result.breakthrough_probability_by_year

    def test_domain_summary(self):
        from oracle.prediction.synthetic_bio import SyntheticBioPredictor
        bio = SyntheticBioPredictor(seed=42)
        result = bio.predict()
        assert len(result.domain_summary) >= 3


class TestScaleCrossCoupling:
    def test_cascades_span_multiple_scales(self):
        from oracle.prediction.scale_bridging import ScaleBridgingPredictionEngine, PredictionScale
        engine = ScaleBridgingPredictionEngine(seed=42)
        result = engine.predict()
        for cascade in result.cascades[:3]:
            scales_in_chain = set()
            for cp in cascade.chain:
                scales_in_chain.add(cp.source_scale)
                scales_in_chain.add(cp.target_scale)
            assert len(scales_in_chain) >= 2, f"Cascade spans only {scales_in_chain}"

    def test_micro_to_macro_traces_exist(self):
        from oracle.prediction.scale_bridging import ScaleBridgingPredictionEngine
        engine = ScaleBridgingPredictionEngine(seed=42)
        result = engine.predict()
        total_couplings = len(result.couplings)
        assert total_couplings >= 10

    def test_all_scales_have_predictions(self):
        from oracle.prediction.scale_bridging import ScaleBridgingPredictionEngine, PredictionScale
        engine = ScaleBridgingPredictionEngine(seed=42)
        result = engine.predict()
        for scale in PredictionScale:
            scale_preds = result.scale_predictions.get(scale.value, [])
            assert len(scale_preds) >= 2, f"Scale {scale.value} has {len(scale_preds)} predictions"

    def test_cosmos_predictions_have_evidence(self):
        from oracle.prediction.scale_bridging import ScaleBridgingPredictionEngine
        engine = ScaleBridgingPredictionEngine(seed=42)
        result = engine.predict()
        cosmos_preds = result.scale_predictions.get("cosmos", [])
        for pred in cosmos_preds:
            assert len(pred.evidence_base) >= 2
            assert pred.base_rate_source


class TestEngineIntegration:
    def test_predict_cross_scale(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        result = engine.predict_cross_scale()
        assert "scale_predictions" in result
        assert "cascades" in result
        assert "macro_to_micro_traces" in result
        assert "micro_to_macro_traces" in result
        assert result["n_scales"] == 9
        assert len(result["cascades"]) >= 1

    def test_predict_cross_scale_with_domain_states(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        result = engine.predict_cross_scale(domain_states={"equity": 0.6, "technology": 0.7})
        assert result["n_scales"] == 9
        assert result["overall_confidence"] > 0

    def test_predict_neuroscience(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        result = engine.predict_neuroscience()
        assert "predictions" in result
        assert "bci_milestones" in result
        assert "breakthrough_probability_by_year" in result
        assert result["n_predictions"] >= 8
        assert result["n_milestones"] >= 3

    def test_predict_physics(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        result = engine.predict_physics()
        assert "predictions" in result
        assert "quantum_milestones" in result
        assert "breakthrough_probability_by_year" in result
        assert result["n_predictions"] >= 8
        assert result["n_milestones"] >= 3

    def test_predict_synthetic_bio(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        result = engine.predict_synthetic_bio()
        assert "predictions" in result
        assert "pipeline" in result
        assert "breakthrough_probability_by_year" in result
        assert result["n_predictions"] >= 8
        assert result["n_pipeline"] >= 3

    def test_predict_all_domains(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        result = engine.predict_all_domains()
        assert "cross_scale" in result
        assert "neuroscience" in result
        assert "physics" in result
        assert "synthetic_bio" in result
        assert "causal_events" in result
        assert result["total_predictions"] >= 40

    def test_predict_with_scale_bridging(self):
        from oracle.prediction.causal_event import CausalEventPredictor
        predictor = CausalEventPredictor()
        result = predictor.predict_with_scale_bridging()
        assert "causal_events" in result
        assert "scale_predictions" in result
        assert "cascades" in result
        assert result["n_events"] >= 20
        assert result["n_scales"] >= 9
        assert len(result["causal_events"]) >= 20
        first_event = result["causal_events"][0]
        assert "scale" in first_event
        assert "related_couplings" in first_event

    def test_predict_with_scale_bridging_domain_states(self):
        from oracle.prediction.causal_event import CausalEventPredictor
        predictor = CausalEventPredictor()
        result = predictor.predict_with_scale_bridging(
            domain_states={"equity": 0.6, "technology": 0.7}
        )
        assert result["n_events"] >= 20
        assert result["n_scales"] >= 9