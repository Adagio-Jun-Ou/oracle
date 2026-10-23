import pytest
from oracle.domains.compound_hazard import CompoundHazardModel, CompoundHazardState
from oracle.domains.hazard_monte_carlo import NaturalHazardMonteCarlo


class TestCompoundHazardModel:
    def setup_method(self):
        self.model = CompoundHazardModel()

    def test_state_defaults(self):
        s = CompoundHazardState()
        assert s.seismic_hazard == 0.3
        assert s.infrastructure_resilience == 0.6
        assert s.social_cohesion == 0.5

    def test_assess_compound_risk(self):
        r = self.model.assess_compound_risk()
        assert "compound_hazard_index" in r
        assert "hazard_level" in r
        assert "resilience_score" in r
        assert "top_hazards" in r
        assert "worst_pair_probability" in r
        assert "systemic_collapse_probability" in r
        assert r["hazard_level"] in ("minimal", "low", "moderate", "elevated", "high", "critical")

    def test_high_risk_state(self):
        s = CompoundHazardState(
            seismic_hazard=0.8, solar_storm_risk=0.7, pandemic_risk=0.5,
            flood_risk=0.7, cyber_attack_risk=0.6, supply_chain_fragility=0.7,
            infrastructure_resilience=0.2, governance_capacity=0.2,
            economic_vulnerability=0.8, social_cohesion=0.2,
        )
        r = self.model.assess_compound_risk(s)
        assert r["compound_hazard_index"] > 0.4
        assert r["hazard_level"] in ("elevated", "high", "critical")

    def test_simulate_scenario(self):
        r = self.model.simulate_scenario(scenario="earthquake_tsunami_nuclear", years=10, n_simulations=100)
        assert r["scenario"] == "earthquake_tsunami_nuclear"
        assert r["scenario_name"] == "Earthquake-Tsunami-Nuclear Cascade"
        assert "annual_cascade_probability" in r
        assert "mean_impact_given_trigger" in r
        assert len(r["timeline"]) > 0

    def test_simulate_solar_storm_scenario(self):
        r = self.model.simulate_scenario(scenario="solar_storm_grid_cyber", years=10, n_simulations=100)
        assert r["scenario"] == "solar_storm_grid_cyber"
        assert "carrington" in r["description"].lower() or "solar" in r["description"].lower()

    def test_multi_hazard_simultaneous(self):
        r = self.model.multi_hazard_simultaneous(n_hazards=3, years=10, n_simulations=200)
        assert "mean_simultaneous" in r
        assert "simultaneous_1yr_probabilities" in r
        assert r["mean_simultaneous"] >= 0

    def test_set_state(self):
        self.model.set_state(seismic_hazard=0.8, infrastructure_resilience=0.2)
        r = self.model.assess_compound_risk()
        assert r["compound_hazard_index"] > 0.2


class TestNaturalHazardMonteCarlo:
    def setup_method(self):
        self.model = NaturalHazardMonteCarlo(seed=42)

    def test_simulate_portfolio(self):
        r = self.model.simulate_portfolio(
            hazards=["earthquake", "hurricane", "flood"],
            years=10, n_simulations=100, exposure_billion=50.0, resilience_factor=0.6,
        )
        assert "mean_total_loss_billion" in r
        assert "max_annual_loss_mean_billion" in r
        assert "total_events_by_type" in r
        assert r["mean_total_loss_billion"] >= 0

    def test_earthquake_scenarios(self):
        r = self.model.earthquake_scenarios(n_simulations=50, exposure_billion=100.0)
        assert r["hazard"] == "earthquake"
        assert len(r["scenarios"]) == 7
        assert r["scenarios"][0]["magnitude"] == 6.0
        assert r["scenarios"][-1]["mean_loss_billion"] > r["scenarios"][0]["mean_loss_billion"]

    def test_hurricane_season(self):
        r = self.model.hurricane_season(n_simulations=100, el_nino_factor=1.0, climate_warming_factor=1.1)
        assert "mean_storms" in r
        assert "mean_landfalls" in r
        assert "mean_major" in r
        assert r["mean_storms"] > 0

    def test_elnino_reduces_hurricanes(self):
        neutral = self.model.hurricane_season(n_simulations=200, el_nino_factor=1.0, climate_warming_factor=1.0)
        elnino = self.model.hurricane_season(n_simulations=200, el_nino_factor=0.5, climate_warming_factor=1.0)
        assert elnino["mean_storms"] < neutral["mean_storms"]

    def test_tsunami_scenarios(self):
        model = NaturalHazardMonteCarlo(seed=42)
        r = model.tsunami_scenarios(n_simulations=50)
        assert r["hazard"] == "tsunami"
        assert len(r["scenarios"]) == 5
        assert r["scenarios"][0]["trigger_magnitude"] == 7.5
        assert r["scenarios"][-1]["mean_loss_billion"] >= r["scenarios"][0]["mean_loss_billion"]

    def test_volcanic_winter_portfolio(self):
        model = NaturalHazardMonteCarlo(seed=42)
        r = model.volcanic_winter_portfolio(vei_min=5, years=50, n_simulations=100)
        assert r["hazard"] == "volcanic_winter"
        assert "triggered_count" in r
        assert "mean_gdp_loss_trillion" in r
        assert "period_50yr_probability" in r

    def test_drought_portfolio(self):
        model = NaturalHazardMonteCarlo(seed=42)
        r = model.drought_portfolio(years=30, n_simulations=100)
        assert r["hazard"] == "drought"
        assert "mean_total_loss_billion" in r
        assert "mean_drought_years" in r
        assert r["mean_drought_years"] > 0
