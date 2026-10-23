"""Tests for Causal Inference Enhanced module."""
import pytest
import numpy as np
from oracle.prediction.causal_enhanced import (
    CausalInferenceEnhanced, CausalMethod,
    MediationResult, InstrumentalResult, DiffInDiffResult,
    PropensityResult, CausalInferenceEnhancedResult,
)


@pytest.fixture
def causal():
    return CausalInferenceEnhanced(n_bootstrap=100)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestMediation:
    def test_basic_mediation(self, causal, rng):
        n = 200
        treatment = rng.normal(0, 1, n)
        mediator = 0.5 * treatment + rng.normal(0, 0.5, n)
        outcome = 0.3 * treatment + 0.4 * mediator + rng.normal(0, 0.3, n)
        result = causal.mediate(treatment, outcome, mediator)
        assert isinstance(result, MediationResult)
        assert abs(result.total_effect) > 0
        assert result.proportion_mediated >= 0

    def test_no_mediator_effect(self, causal, rng):
        n = 200
        treatment = rng.normal(0, 1, n)
        mediator = rng.normal(0, 1, n)
        outcome = 0.5 * treatment + rng.normal(0, 0.3, n)
        result = causal.mediate(treatment, outcome, mediator)
        assert abs(result.indirect_effect) < abs(result.direct_effect) + 0.3

    def test_full_mediation(self, causal, rng):
        n = 200
        treatment = rng.normal(0, 1, n)
        mediator = 0.8 * treatment + rng.normal(0, 0.1, n)
        outcome = 0.7 * mediator + rng.normal(0, 0.1, n)
        result = causal.mediate(treatment, outcome, mediator)
        assert result.proportion_mediated > 0.3


class TestInstrumentalVariable:
    def test_basic_iv(self, causal, rng):
        n = 200
        instrument = rng.normal(0, 1, n)
        treatment = 0.7 * instrument + rng.normal(0, 0.3, n)
        outcome = 0.5 * treatment + rng.normal(0, 0.5, n)
        result = causal.instrumental_variable(treatment, outcome, instrument)
        assert isinstance(result, InstrumentalResult)
        assert result.first_stage_f > 0

    def test_weak_instrument(self, causal, rng):
        n = 200
        instrument = rng.normal(0, 1, n)
        treatment = 0.05 * instrument + rng.normal(0, 1, n)
        outcome = 0.5 * treatment + rng.normal(0, 0.5, n)
        result = causal.instrumental_variable(treatment, outcome, instrument)
        assert result.instrument_relevance < 0.5


class TestDiffInDiff:
    def test_basic_did(self, causal, rng):
        n = 100
        treated_pre = rng.normal(10, 2, n)
        treated_post = treated_pre + 3 + rng.normal(0, 1, n)
        control_pre = rng.normal(10, 2, n)
        control_post = control_pre + 1 + rng.normal(0, 1, n)
        result = causal.diff_in_diff(treated_pre, treated_post, control_pre, control_post)
        assert isinstance(result, DiffInDiffResult)
        assert result.did_estimate > 1.0

    def test_no_effect(self, causal, rng):
        n = 100
        treated_pre = rng.normal(10, 2, n)
        treated_post = treated_pre + 1 + rng.normal(0, 1, n)
        control_pre = rng.normal(10, 2, n)
        control_post = control_pre + 1 + rng.normal(0, 1, n)
        result = causal.diff_in_diff(treated_pre, treated_post, control_pre, control_post)
        assert abs(result.did_estimate) < 2.0


class TestPropensityScore:
    def test_basic_propensity(self, causal, rng):
        n = 200
        covariates = rng.normal(0, 1, (n, 3))
        prob = 1 / (1 + np.exp(-(covariates[:, 0] + 0.5 * covariates[:, 1])))
        treatment = (prob > 0.5).astype(float)
        outcome = 0.5 * treatment + 0.3 * covariates[:, 0] + rng.normal(0, 0.5, n)
        result = causal.propensity_score(treatment, outcome, covariates)
        assert isinstance(result, PropensityResult)
        assert 0 <= result.ps_overlap <= 1

    def test_ate_positive(self, causal, rng):
        n = 200
        covariates = rng.normal(0, 1, (n, 3))
        treatment = rng.binomial(1, 0.5, n).astype(float)
        outcome = 2.0 * treatment + rng.normal(0, 1, n)
        result = causal.propensity_score(treatment, outcome, covariates)
        assert result.ate > 0

    def test_balance_statistics(self, causal, rng):
        n = 200
        covariates = rng.normal(0, 1, (n, 3))
        treatment = rng.binomial(1, 0.5, n).astype(float)
        outcome = rng.normal(0, 1, n)
        result = causal.propensity_score(treatment, outcome, covariates)
        assert len(result.balance_statistics) > 0


class TestSensitivityAnalysis:
    def test_basic_sensitivity(self, causal):
        result = causal.sensitivity_analysis(estimate=0.5, se=0.1)
        assert isinstance(result, CausalInferenceEnhancedResult)
        assert result.method == CausalMethod.SENSITIVITY
        assert result.estimate == 0.5

    def test_e_value(self, causal):
        result = causal.sensitivity_analysis(estimate=0.5, se=0.1)
        assert result.diagnostics["e_value"] > 0

    def test_weak_effect(self, causal):
        result = causal.sensitivity_analysis(estimate=0.05, se=0.1)
        assert "not statistically significant" in " ".join(result.assumptions_violated) or result.p_value > 0.05

    def test_strong_effect(self, causal):
        result = causal.sensitivity_analysis(estimate=2.0, se=0.1)
        assert result.effect_size > 1.0

    def test_confidence_interval(self, causal):
        result = causal.sensitivity_analysis(estimate=0.5, se=0.1)
        assert result.confidence_interval[0] < 0.5 < result.confidence_interval[1]


class TestCausalMethodEnum:
    def test_values(self):
        assert CausalMethod.MEDIATION.value == "mediation"
        assert CausalMethod.INSTRUMENTAL.value == "instrumental"
        assert CausalMethod.DIFF_IN_DIFF.value == "diff_in_diff"
        assert CausalMethod.REG_DISC.value == "reg_disc"
        assert CausalMethod.PROPENSITY.value == "propensity"
        assert CausalMethod.SENSITIVITY.value == "sensitivity"