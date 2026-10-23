from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy import stats


@dataclass
class CausalEffect:
    treatment: str
    outcome: str
    ate: float
    att: float
    atc: float
    ite_std: float
    confidence_interval: tuple
    p_value: float
    effect_size: float
    direction: str
    assumptions: list = field(default_factory=list)
    sensitivity: dict = field(default_factory=dict)


@dataclass
class CounterfactualResult:
    observed: float
    counterfactual: float
    treatment_effect: float
    scenario: str
    confidence: float


class CausalInference:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._causal_graph = {}
        self._confounders = []

    def define_causal_graph(self, edges: list[tuple[str, str, float]],
                            confounders: list[tuple[str, list[str]]] = None):
        for src, dst, weight in edges:
            if src not in self._causal_graph:
                self._causal_graph[src] = {}
            self._causal_graph[src][dst] = weight
        if confounders:
            self._confounders = confounders

    def _get_confounders_of(self, treatment: str, outcome: str) -> list[str]:
        result = []
        for confounder, affects in self._confounders:
            if treatment in affects and outcome in affects:
                result.append(confounder)
        return result

    def average_treatment_effect(self, y_treated: np.ndarray, y_control: np.ndarray,
                                  x_treated: Optional[np.ndarray] = None,
                                  x_control: Optional[np.ndarray] = None,
                                  treatment: str = "treatment",
                                  outcome: str = "outcome") -> CausalEffect:
        n_t = len(y_treated)
        n_c = len(y_control)
        ate = float(np.mean(y_treated) - np.mean(y_control))
        se_ate = float(np.sqrt(np.var(y_treated, ddof=1) / n_t + np.var(y_control, ddof=1) / n_c))
        z = 1.96
        ci = (ate - z * se_ate, ate + z * se_ate)
        t_stat = ate / se_ate if se_ate > 0 else 0
        p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
        effect_size = ate / float(np.std(np.concatenate([y_treated, y_control]))) if len(y_treated) + len(y_control) > 0 else 0.0
        att = float(np.mean(y_treated) - np.mean(y_control))
        atc = float(np.mean(y_treated) - np.mean(y_control))
        if x_treated is not None and x_control is not None:
            att, atc, ite_std = self._regression_adjustment(
                y_treated, y_control, x_treated, x_control
            )
        else:
            ite_std = float(np.std(y_treated - np.mean(y_control)))
        direction = "positive" if ate > 0 else "negative" if ate < 0 else "null"
        assumptions = ["SUTVA", "ignorability", "consistency"]
        if x_treated is None:
            assumptions.append("unadjusted_estimation")
        else:
            assumptions.append("regression_adjusted")
        sensitivity = self._sensitivity_analysis(ate, se_ate, n_t, n_c)
        return CausalEffect(
            treatment=treatment, outcome=outcome, ate=ate, att=att, atc=atc,
            ite_std=ite_std, confidence_interval=ci, p_value=p_value,
            effect_size=effect_size, direction=direction,
            assumptions=assumptions, sensitivity=sensitivity,
        )

    def _regression_adjustment(self, y_treated: np.ndarray, y_control: np.ndarray,
                                x_treated: np.ndarray, x_control: np.ndarray) -> tuple:
        x_all = np.vstack([x_treated, x_control])
        y_all = np.concatenate([y_treated, y_control])
        n_t = len(y_treated)
        n_c = len(y_control)
        d_idx = np.concatenate([np.ones(n_t), np.zeros(n_c)])
        X = np.column_stack([np.ones(len(y_all)), d_idx, x_all])
        try:
            beta = np.linalg.lstsq(X, y_all, rcond=None)[0]
            ate = float(beta[1])
            x_control_mean = np.mean(x_control, axis=0)
            x_treated_mean = np.mean(x_treated, axis=0)
            x_treated_adj = np.column_stack([np.ones(n_t), np.zeros(n_t), x_treated])
            x_control_adj = np.column_stack([np.ones(n_c), np.ones(n_c), x_control])
            y_t_pred = X[:n_t] @ beta
            y_c_pred = X[n_t:] @ beta
            att = float(np.mean(y_treated) - np.mean(y_c_pred + beta[1]))
            atc = float(np.mean(y_t_pred - beta[1]) - np.mean(y_control))
            ite_std = float(np.std(y_treated - y_c_pred[:n_t]))
        except Exception:
            ate = float(np.mean(y_treated) - np.mean(y_control))
            att = ate
            atc = ate
            ite_std = float(np.std(np.concatenate([y_treated, y_control])))
        return att, atc, ite_std

    def _sensitivity_analysis(self, ate: float, se: float, n_t: int, n_c: int) -> dict:
        z = 1.96
        results = {}
        for gamma in [1.0, 1.5, 2.0, 2.5, 3.0]:
            gamma_se = se * gamma
            ci = (ate - z * gamma_se, ate + z * gamma_se)
            t = ate / gamma_se if gamma_se > 0 else 0
            p = float(2 * (1 - stats.norm.cdf(abs(t))))
            still_significant = p < 0.05
            results[f"gamma_{gamma}"] = {
                "ci": ci,
                "p_value": p,
                "still_significant": still_significant,
            }
        robustness = max([g for g in [1.0, 1.5, 2.0, 2.5, 3.0]
                         if results.get(f"gamma_{g}", {}).get("still_significant", False)] + [0], default=0)
        results["robustness_threshold"] = robustness
        return results

    def instrumental_variable(self, y: np.ndarray, d: np.ndarray, z: np.ndarray,
                               treatment: str = "treatment",
                               outcome: str = "outcome") -> CausalEffect:
        cov_dz = float(np.cov(d, z)[0, 1])
        cov_yz = float(np.cov(y, z)[0, 1])
        iv_estimate = cov_yz / cov_dz if abs(cov_dz) > 1e-10 else 0.0
        n = len(y)
        z2 = z * d
        X_iv = np.column_stack([np.ones(n), d])
        z_instr = np.column_stack([z, z2])
        try:
            beta_2sls = np.linalg.lstsq(z_instr @ X_iv, z_instr @ y, rcond=None)[0]
            ate = float(beta_2sls[1])
        except Exception:
            ate = iv_estimate
        se = float(np.std(y)) / np.sqrt(n) if n > 0 else 1.0
        ci = (ate - 1.96 * se, ate + 1.96 * se)
        direction = "positive" if ate > 0 else "negative" if ate < 0 else "null"
        t_stat = ate / se if se > 0 else 0
        p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
        return CausalEffect(
            treatment=treatment, outcome=outcome, ate=ate, att=ate, atc=ate,
            ite_std=se, confidence_interval=ci, p_value=p_value,
            effect_size=abs(ate) / se if se > 0 else 0,
            direction=direction,
            assumptions=["exclusion_restriction", "relevance", "independence"],
            sensitivity={"method": "instrumental_variable", "iv_estimate": iv_estimate},
        )

    def difference_in_differences(self, y_pre_treatment: np.ndarray, y_post_treatment: np.ndarray,
                                   y_pre_control: np.ndarray, y_post_control: np.ndarray,
                                   treatment: str = "treatment",
                                   outcome: str = "outcome") -> CausalEffect:
        diff_t = float(np.mean(y_post_treatment) - np.mean(y_pre_treatment))
        diff_c = float(np.mean(y_post_control) - np.mean(y_pre_control))
        did_estimate = diff_t - diff_c
        n_t = len(y_post_treatment)
        n_c = len(y_post_control)
        se = float(np.sqrt(
            np.var(y_post_treatment, ddof=1) / n_t +
            np.var(y_pre_treatment, ddof=1) / n_t +
            np.var(y_post_control, ddof=1) / n_c +
            np.var(y_pre_control, ddof=1) / n_c
        ))
        ci = (did_estimate - 1.96 * se, did_estimate + 1.96 * se)
        t_stat = did_estimate / se if se > 0 else 0
        p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
        direction = "positive" if did_estimate > 0 else "negative" if did_estimate < 0 else "null"
        return CausalEffect(
            treatment=treatment, outcome=outcome, ate=did_estimate, att=did_estimate, atc=did_estimate,
            ite_std=se, confidence_interval=ci, p_value=p_value,
            effect_size=abs(did_estimate) / se if se > 0 else 0,
            direction=direction,
            assumptions=["parallel_trends", "no_anticipation", "stable_unit_treatment_value"],
            sensitivity={"treatment_pre_mean": float(np.mean(y_pre_treatment)),
                         "treatment_post_mean": float(np.mean(y_post_treatment)),
                         "control_pre_mean": float(np.mean(y_pre_control)),
                         "control_post_mean": float(np.mean(y_post_control)),
                         "parallel_trend_assumption": "assumed"},
        )

    def counterfactual_analysis(self, factual: dict, interventions: list[dict],
                                 outcome_fn=None, n_simulations: int = 1000) -> list[CounterfactualResult]:
        rng = np.random.RandomState(self.seed)
        results = []
        for intervention in interventions:
            counterfactual = factual.copy()
            for key, value in intervention.items():
                if key in counterfactual:
                    counterfactual[key] = value
            if outcome_fn:
                factual_outcome = outcome_fn(**factual)
                cf_outcome = outcome_fn(**counterfactual)
            else:
                factual_outcome = sum(factual.values()) / len(factual)
                cf_outcome = sum(counterfactual.values()) / len(counterfactual)
            effect = cf_outcome - factual_outcome
            noise = rng.normal(0, abs(effect) * 0.1 + 0.01, n_simulations)
            effects = effect + noise
            ci_lower = float(np.percentile(effects, 5))
            ci_upper = float(np.percentile(effects, 95))
            confidence = float(np.mean((effects > 0) if effect > 0 else (effects < 0)))
            results.append(CounterfactualResult(
                observed=float(factual_outcome),
                counterfactual=float(cf_outcome),
                treatment_effect=float(effect),
                scenario=str(intervention),
                confidence=float(confidence),
            ))
        return results

    def mediation_analysis(self, y: np.ndarray, treatment: np.ndarray,
                            mediator: np.ndarray,
                            treatment_name: str = "X",
                            outcome_name: str = "Y",
                            mediator_name: str = "M") -> dict:
        n = len(y)
        X_total = np.column_stack([np.ones(n), treatment])
        try:
            beta_total = np.linalg.lstsq(X_total, y, rcond=None)[0]
            total_effect = float(beta_total[1])
        except Exception:
            total_effect = float(np.cov(y, treatment)[0, 1] / np.var(treatment)) if np.var(treatment) > 0 else 0.0
        X_med = np.column_stack([np.ones(n), treatment])
        try:
            alpha_path = np.linalg.lstsq(X_med, mediator, rcond=None)[0]
            a = float(alpha_path[1])
        except Exception:
            a = float(np.cov(mediator, treatment)[0, 1] / np.var(treatment)) if np.var(treatment) > 0 else 0.0
        X_direct = np.column_stack([np.ones(n), treatment, mediator])
        try:
            beta_direct = np.linalg.lstsq(X_direct, y, rcond=None)[0]
            direct_effect = float(beta_direct[1])
            b = float(beta_direct[2])
        except Exception:
            direct_effect = total_effect
            b = 0.0
        indirect_effect = a * b
        proportion_mediated = indirect_effect / total_effect if abs(total_effect) > 1e-10 else 0.0
        se_indirect = float(np.sqrt((a ** 2 * np.var(mediator) + b ** 2 * np.var(treatment)) / n)) if n > 0 else 0.0
        z_indirect = indirect_effect / se_indirect if se_indirect > 0 else 0.0
        p_indirect = float(2 * (1 - stats.norm.cdf(abs(z_indirect))))
        return {
            "total_effect": float(total_effect),
            "direct_effect": float(direct_effect),
            "indirect_effect": float(indirect_effect),
            "proportion_mediated": float(np.clip(proportion_mediated, -1, 1)),
            "path_a": float(a),
            "path_b": float(b),
            "path_c_prime": float(direct_effect),
            "indirect_p_value": float(p_indirect),
            "indirect_significant": p_indirect < 0.05,
            "treatment_name": treatment_name,
            "outcome_name": outcome_name,
            "mediator_name": mediator_name,
        }

    def propensity_score_matching(self, y: np.ndarray, treatment: np.ndarray,
                                   x: np.ndarray,
                                   treatment_name: str = "treatment",
                                   outcome_name: str = "outcome") -> CausalEffect:
        n = len(y)
        X_ps = np.column_stack([np.ones(n), x])
        try:
            beta_ps = np.linalg.lstsq(X_ps, treatment, rcond=None)[0]
            pscores = 1.0 / (1.0 + np.exp(-X_ps @ beta_ps))
            pscores = np.clip(pscores, 0.01, 0.99)
        except Exception:
            pscores = np.full(n, 0.5)
        treated_idx = np.where(treatment > 0.5)[0]
        control_idx = np.where(treatment <= 0.5)[0]
        matched_effects = []
        for t_idx in treated_idx:
            ps_t = pscores[t_idx]
            distances = np.abs(pscores[control_idx] - ps_t)
            best_match = control_idx[np.argmin(distances)]
            matched_effects.append(y[t_idx] - y[best_match])
        ate = float(np.mean(matched_effects)) if matched_effects else 0.0
        se = float(np.std(matched_effects) / np.sqrt(len(matched_effects))) if len(matched_effects) > 1 else 1.0
        ci = (ate - 1.96 * se, ate + 1.96 * se)
        t_stat = ate / se if se > 0 else 0
        p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
        direction = "positive" if ate > 0 else "negative" if ate < 0 else "null"
        return CausalEffect(
            treatment=treatment_name, outcome=outcome_name, ate=ate, att=ate, atc=ate,
            ite_std=se, confidence_interval=ci, p_value=p_value,
            effect_size=abs(ate) / se if se > 0 else 0,
            direction=direction,
            assumptions=["conditional_independence", "overlap", "SUTVA"],
            sensitivity={"n_matched": len(matched_effects), "method": "propensity_score_matching"},
        )

    def backdoor_adjustment(self, treatment: str, outcome: str,
                            confounders: list[str] = None) -> dict:
        if confounders is None:
            confounders = self._get_confounders_of(treatment, outcome)
        direct_paths = []
        if treatment in self._causal_graph:
            for target, weight in self._causal_graph[treatment].items():
                if target == outcome:
                    direct_paths.append({"path": f"{treatment} -> {outcome}", "weight": weight})
                elif target != outcome and target in self._causal_graph:
                    if outcome in self._causal_graph[target]:
                        w = weight * self._causal_graph[target][outcome]
                        direct_paths.append({"path": f"{treatment} -> {target} -> {outcome}", "weight": w})
        blocked_paths = []
        for conf in confounders:
            if conf in self._causal_graph:
                for target, w in self._causal_graph[conf].items():
                    if target == treatment or target == outcome:
                        blocked_paths.append({
                            "confounder": conf,
                            "path": f"{conf} -> {target}",
                            "weight": w,
                            "blocked": True,
                        })
        total_effect = sum(p["weight"] for p in direct_paths) if direct_paths else 0.0
        return {
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "direct_paths": direct_paths,
            "blocked_paths": blocked_paths,
            "total_effect": float(total_effect),
            "backdoor_criterion_satisfied": len(confounders) == 0 or len(blocked_paths) > 0,
            "adjustment_needed": len(confounders) > 0,
        }