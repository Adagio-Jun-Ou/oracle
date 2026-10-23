import pytest
from oracle.domains.world_state import WorldStateModel


class TestWorldStateModel:
    def setup_method(self):
        self.model = WorldStateModel()

    def test_assess_world_state_basic(self):
        scores = {"economic": 0.6, "seismic": 0.4, "climate": 0.3, "cyber": 0.5}
        r = self.model.assess_world_state(scores)
        assert "overall_stability" in r
        assert "threat_level" in r
        assert "n_domains_assessed" in r
        assert "clusters" in r
        assert "worst_domains" in r
        assert "best_domains" in r
        assert r["n_domains_assessed"] == 4
        assert 0.0 <= r["overall_stability"] <= 1.0

    def test_assess_world_state_levels(self):
        good = {k: 0.8 for k in ["economic", "climate", "political", "infrastructure", "water", "food"]}
        r_good = self.model.assess_world_state(good)
        assert r_good["threat_level"] in ("guarded_stability", "favorable_trajectory")

        bad = {k: 0.15 for k in ["economic", "climate", "political", "infrastructure", "water", "food", "cyber", "seismic", "social"]}
        r_bad = self.model.assess_world_state(bad)
        assert r_bad["overall_stability"] < 0.35
        assert r_bad["threat_level"] in ("existential_crisis", "severe_danger")

    def test_clusters_populated(self):
        scores = {"epidemiology": 0.5, "pandemic": 0.6, "mental_health": 0.4,
                   "food": 0.7, "water": 0.3, "seismic": 0.4, "solar": 0.8,
                   "infrastructure": 0.5, "supply_chain": 0.4, "cyber": 0.6,
                   "political": 0.3, "social": 0.5}
        r = self.model.assess_world_state(scores)
        assert "human_health" in r["clusters"]
        assert "natural_forces" in r["clusters"]
        assert "infrastructure_integrity" in r["clusters"]
        assert "governance_effectiveness" in r["clusters"]

    def test_worst_and_best_domains(self):
        scores = {"a": 0.1, "b": 0.2, "c": 0.5, "d": 0.8, "e": 0.9}
        r = self.model.assess_world_state(scores)
        assert r["worst_domains"][0]["domain"] == "a"
        assert r["best_domains"][0]["domain"] == "e"

    def test_recommendations(self):
        bad = {k: 0.15 for k in ["economic", "climate", "political", "infrastructure", "water", "food", "cyber"]}
        r = self.model.assess_world_state(bad)
        assert len(r["recommendations"]) > 0

    def test_stability_trajectory(self):
        scores = {"economic": 0.5, "climate": 0.4, "seismic": 0.5}
        r = self.model.stability_trajectory(scores, years=20, n_simulations=100)
        assert "base_stability" in r
        assert "mean_final_stability" in r
        assert "collapse_probability" in r
        assert "timeline" in r
        assert len(r["timeline"]) > 0
        assert 0.0 <= r["collapse_probability"] <= 1.0

    def test_domain_radar(self):
        scores = {"economic": 0.7, "seismic": 0.3, "climate": 0.4, "cyber": 0.5}
        r = self.model.domain_radar(scores)
        assert "domains" in r
        assert "total_weighted_danger" in r
        assert "most_dangerous_domain" in r
        assert "safest_domain" in r
        assert r["most_dangerous_domain"] == "seismic"
        assert r["safest_domain"] == "economic"

    def test_compound_hazard_burden(self):
        distressed = {"a": 0.1, "b": 0.15, "c": 0.2, "d": 0.3}
        r = self.model.assess_world_state(distressed)
        assert r["compound_hazard_burden"] > 0

    def test_imbalances_detected(self):
        imbalanced = {"economic": 0.1, "climate": 0.1, "political": 0.1,
                       "infrastructure": 0.8, "technology": 0.9, "space": 0.9}
        r = self.model.assess_world_state(imbalanced)
        assert len(r["imbalances"]) > 0

    def test_empty_scores(self):
        r = self.model.assess_world_state({})
        assert r["n_domains_assessed"] == 0