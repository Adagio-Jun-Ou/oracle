"""Tests for human outcome prediction module."""
import pytest
import numpy as np


class TestHumanOutcomePredictor:
    def test_predict_all_domains(self):
        from oracle.prediction.human_outcome import HumanOutcomePredictor
        ho = HumanOutcomePredictor(seed=42)
        result = ho.predict()
        assert result.n_predictions >= 20
        assert result.domains_covered >= 5
        assert len(result.predictions) >= 20
        assert len(result.decision_outcomes) >= 3
        assert len(result.genetic_risks) >= 3

    def test_predict_specific_domains(self):
        from oracle.prediction.human_outcome import HumanOutcomePredictor, HumanOutcomeDomain
        ho = HumanOutcomePredictor(seed=42)
        result = ho.predict(domains=[HumanOutcomeDomain.BEHAVIORAL, HumanOutcomeDomain.COGNITIVE])
        domains_in_result = set(p.domain for p in result.predictions)
        assert HumanOutcomeDomain.BEHAVIORAL in domains_in_result
        assert HumanOutcomeDomain.COGNITIVE in domains_in_result
        assert HumanOutcomeDomain.PSYCHOLOGICAL not in domains_in_result

    def test_prediction_structure(self):
        from oracle.prediction.human_outcome import HumanOutcomePredictor, HumanPrediction, HumanOutcomeDomain, InterventionType
        ho = HumanOutcomePredictor(seed=42)
        result = ho.predict()
        for pred in result.predictions[:5]:
            assert isinstance(pred, HumanPrediction)
            assert isinstance(pred.domain, HumanOutcomeDomain)
            assert 0 < pred.probability < 1
            assert 0 < pred.confidence <= 1
            assert pred.prediction
            assert len(pred.evidence) >= 2
            assert pred.falsification
            assert pred.key_metric
            assert pred.uncertainty_band[0] < pred.uncertainty_band[1]
            assert isinstance(pred.intervention_type, InterventionType)
            assert 0 <= pred.intervention_potential <= 1
            assert len(pred.individual_factors) >= 2

    def test_decision_outcomes(self):
        from oracle.prediction.human_outcome import HumanOutcomePredictor
        ho = HumanOutcomePredictor(seed=42)
        result = ho.predict()
        for outcome in result.decision_outcomes:
            assert outcome.decision_domain
            assert outcome.option_a
            assert outcome.option_b
            assert 0 < outcome.option_a_probability < 1
            assert 0 < outcome.option_b_probability < 1
            assert 0 < outcome.option_a_expected_value <= 1
            assert 0 < outcome.option_b_expected_value <= 1
            assert len(outcome.bias_flags) >= 1
            assert outcome.recommended

    def test_genetic_risks(self):
        from oracle.prediction.human_outcome import HumanOutcomePredictor
        ho = HumanOutcomePredictor(seed=42)
        result = ho.predict()
        for risk in result.genetic_risks:
            assert risk.condition
            assert 0 < risk.lifetime_risk_baseline < 1
            assert risk.lifetime_risk_with_risk_allele >= risk.lifetime_risk_baseline
            assert risk.odds_ratio >= 1.0
            assert 0 <= risk.modifiable_fraction <= 1
            assert risk.intervention

    def test_breakthrough_probability(self):
        from oracle.prediction.human_outcome import HumanOutcomePredictor
        ho = HumanOutcomePredictor(seed=42)
        result = ho.predict()
        assert len(result.breakthrough_probability_by_year) >= 3
        for year, prob in result.breakthrough_probability_by_year.items():
            assert 2025 <= year <= 2035
            assert 0 < prob < 1

    def test_domain_states_adjustment(self):
        from oracle.prediction.human_outcome import HumanOutcomePredictor
        ho = HumanOutcomePredictor(seed=42)
        result_default = ho.predict(domain_states={"health": 0.5, "social": 0.5})
        result_healthy = ho.predict(domain_states={"health": 0.8, "social": 0.7})
        behavioral_probs_default = [p.probability for p in result_default.predictions
                                     if p.domain.value == "behavioral"]
        behavioral_probs_healthy = [p.probability for p in result_healthy.predictions
                                     if p.domain.value == "behavioral"]
        assert len(behavioral_probs_default) == len(behavioral_probs_healthy)

    def test_intervention_potential_range(self):
        from oracle.prediction.human_outcome import HumanOutcomePredictor
        ho = HumanOutcomePredictor(seed=42)
        result = ho.predict()
        for pred in result.predictions:
            assert 0 <= pred.intervention_potential <= 1

    def test_genetic_risk_factors_data(self):
        from oracle.prediction.human_outcome import GENETIC_RISK_FACTORS, GeneticRiskFactor
        assert len(GENETIC_RISK_FACTORS) >= 4
        for risk in GENETIC_RISK_FACTORS:
            assert isinstance(risk, GeneticRiskFactor)
            assert risk.lifetime_risk_with_risk_allele > risk.lifetime_risk_baseline
            assert risk.odds_ratio >= 1.0

    def test_decision_outcomes_data(self):
        from oracle.prediction.human_outcome import DECISION_OUTCOMES_DATA, DecisionOutcome
        assert len(DECISION_OUTCOMES_DATA) >= 3
        for outcome in DECISION_OUTCOMES_DATA:
            assert isinstance(outcome, DecisionOutcome)
            assert outcome.option_a_probability > 0
            assert outcome.option_b_probability > 0
            assert len(outcome.bias_flags) >= 1

    def test_predict_domain_coverage(self):
        from oracle.prediction.human_outcome import HumanOutcomePredictor, HumanOutcomeDomain
        ho = HumanOutcomePredictor(seed=42)
        result = ho.predict()
        domains_found = set(p.domain for p in result.predictions)
        for domain in [HumanOutcomeDomain.BEHAVIORAL, HumanOutcomeDomain.PSYCHOLOGICAL,
                       HumanOutcomeDomain.GENETIC_RISK, HumanOutcomeDomain.DECISION,
                       HumanOutcomeDomain.COGNITIVE, HumanOutcomeDomain.SOCIAL]:
            assert domain in domains_found, f"Missing domain: {domain}"

    def test_predictions_are_falsifiable(self):
        from oracle.prediction.human_outcome import HumanOutcomePredictor
        ho = HumanOutcomePredictor(seed=42)
        result = ho.predict()
        for pred in result.predictions:
            assert pred.falsification
            assert len(pred.falsification) >= 20
            assert pred.evidence
            assert len(pred.evidence) >= 2

    def test_predictions_have_evidence_citations(self):
        from oracle.prediction.human_outcome import HumanOutcomePredictor
        ho = HumanOutcomePredictor(seed=42)
        result = ho.predict()
        for pred in result.predictions:
            for ev in pred.evidence:
                assert any(char.isdigit() for char in ev), f"Evidence should contain year: {ev}"

    def test_deterministic_with_seed(self):
        from oracle.prediction.human_outcome import HumanOutcomePredictor
        ho1 = HumanOutcomePredictor(seed=42)
        ho2 = HumanOutcomePredictor(seed=42)
        r1 = ho1.predict()
        r2 = ho2.predict()
        for p1, p2 in zip(r1.predictions, r2.predictions):
            assert abs(p1.probability - p2.probability) < 0.001

    def test_assessment_not_empty(self):
        from oracle.prediction.human_outcome import HumanOutcomePredictor
        ho = HumanOutcomePredictor(seed=42)
        result = ho.predict()
        assert result.overall_assessment
        assert len(result.overall_assessment) > 50


class TestHumanOutcomeEngine:
    def test_engine_predict_human_outcome(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        result = engine.predict_human_outcome()
        assert "predictions" in result
        assert "decision_outcomes" in result
        assert "genetic_risks" in result
        assert "breakthrough_probability_by_year" in result
        assert result["n_predictions"] >= 20
        assert result["domains_covered"] >= 5
        assert len(result["predictions"]) >= 20
        assert len(result["decision_outcomes"]) >= 3
        assert len(result["genetic_risks"]) >= 3

    def test_engine_with_domain_states(self):
        from oracle.prediction.engine import OracleEngine
        engine = OracleEngine()
        result = engine.predict_human_outcome(
            domain_states={"health": 0.7, "social": 0.3, "technology": 0.6}
        )
        assert result["n_predictions"] >= 20

    def test_human_outcome_api(self):
        from oracle.web.app import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.post("/api/human-outcome/predict", json={"domain_states": {"health": 0.5, "social": 0.5}})
        data = resp.json()
        assert resp.status_code == 200
        assert "predictions" in data
        assert "decision_outcomes" in data
        assert "genetic_risks" in data
        assert data["n_predictions"] >= 20