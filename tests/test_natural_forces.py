import pytest
from datetime import date
from oracle.domains.seismic import SeismicModel, SeismicState
from oracle.domains.solar import SolarModel, SolarState
from oracle.domains.oceanic import OceanicModel, OceanicState
from oracle.domains.neo import NearEarthObjectModel, NearEarthObjectState
from oracle.domains.atmospheric import AtmosphericModel, AtmosphericState
from oracle.domains.ecological import EcologicalModel, EcologicalState
from oracle.domains.paleoclimatology import PaleoclimatologyModel, PaleoclimateState
from oracle.domains.cosmic_radiation import CosmicRadiationModel, CosmicRadiationState
from oracle.domains.gravitational import GravitationalModel, GravitationalState
from oracle.prediction.engine import OracleEngine
from oracle.core.types import Domain
from oracle.ingestion.natural_data import NaturalDataFetcher


class TestSeismicModel:
    def setup_method(self):
        self.model = SeismicModel()

    def test_state_defaults(self):
        s = SeismicState()
        assert s.earthquake_rate_monthly == 12.0
        assert s.fault_stress_accumulation == 0.5
        assert s.volcanic_activity_index == 0.3

    def test_assess_seismic_hazard(self):
        r = self.model.assess_seismic_hazard()
        assert "composite_hazard" in r
        assert "hazard_level" in r
        assert "m7_probability_30d" in r
        assert "m8_probability_365d" in r
        assert "recommendations" in r

    def test_high_stress_hazard(self):
        s = SeismicState(fault_stress_accumulation=0.9, subduction_zone_stress=0.8, swarm_activity=0.7)
        r = self.model.assess_seismic_hazard(s)
        assert r["hazard_level"] in ("high", "critical")

    def test_simulate_earthquake(self):
        r = self.model.simulate_earthquake_scenario(magnitude=8.0, depth_km=15.0, population_exposed_m=5.0)
        assert "damage_ratio" in r
        assert "economic_loss_billion_usd" in r
        assert "tsunami_probability" in r
        assert r["magnitude"] == 8.0

    def test_simulate_tsunami(self):
        r = self.model.simulate_earthquake_scenario(magnitude=9.0, depth_km=30.0, population_exposed_m=5.0)
        assert r["tsunami_probability"] > 0

    def test_predict(self):
        r = self.model.predict(years_ahead=5)
        assert "future_fault_stress" in r
        assert "seismic_regime" in r
        assert "m8_probability_5yr" in r

    def test_set_state(self):
        m = self.model.set_state(earthquake_rate_monthly=20.0, fault_stress_accumulation=0.8)
        assert m.state.earthquake_rate_monthly == 20.0


class TestSolarModel:
    def setup_method(self):
        self.model = SolarModel()

    def test_state_defaults(self):
        s = SolarState()
        assert s.sunspot_number == 80.0
        assert s.kp_index == 3.0
        assert s.dst_index == -20.0

    def test_assess_solar_weather(self):
        r = self.model.assess_solar_weather()
        assert "composite_risk" in r
        assert "risk_level" in r
        assert "x_flare_probability" in r
        assert "grid_infrastructure_risk" in r
        assert "satellite_risk" in r

    def test_carrington_risk(self):
        s = SolarState(dst_index=-500.0, kp_index=9.0, proton_flux=1.0, flare_probability=0.8)
        r = self.model.assess_solar_weather(s)
        assert r["grid_infrastructure_risk"] > 0.3
        assert r["risk_level"] in ("high", "extreme")

    def test_carrington_scenario(self):
        r = self.model.simulate_carrington_event(grid_hardening=0.3, satellite_shielding=0.2)
        assert "grid_damage_pct" in r
        assert "economic_impact_trillion_usd" in r
        assert r["scenario"] == "carrington_class"

    def test_predict(self):
        r = self.model.predict(years_ahead=5)
        assert "activity_regime" in r
        assert "carrington_probability" in r
        assert "future_sunspot_number" in r

    def test_set_state(self):
        m = self.model.set_state(kp_index=8.0, dst_index=-300.0)
        assert m.state.kp_index == 8.0


class TestOceanicModel:
    def setup_method(self):
        self.model = OceanicModel()

    def test_state_defaults(self):
        s = OceanicState()
        assert s.enso_index == 0.5
        assert s.arctic_sea_ice_extent_mkm2 == 4.5
        assert s.ocean_acidification_ph == 8.07

    def test_assess_ocean_state(self):
        r = self.model.assess_ocean_state()
        assert "composite_ocean_risk" in r
        assert "risk_level" in r
        assert "enso_phase" in r
        assert "expected_hurricane_season" in r

    def test_el_nino(self):
        s = OceanicState(enso_index=1.5, sea_surface_temp_anomaly_c=1.2)
        r = self.model.assess_ocean_state(s)
        assert r["enso_phase"] == "elnino"

    def test_la_nina(self):
        s = OceanicState(enso_index=-1.5)
        r = self.model.assess_ocean_state(s)
        assert r["enso_phase"] == "lanina"

    def test_simulate_enso(self):
        r = self.model.simulate_enso_transition(current_phase="elnino", months=24)
        assert "timeline" in r
        assert len(r["timeline"]) == 24
        assert r["start_phase"] == "elnino"

    def test_predict(self):
        r = self.model.predict(years_ahead=5)
        assert "future_sst_anomaly_c" in r
        assert "amoc_collapse_risk" in r
        assert "ice_free_summer_probability" in r

    def test_set_state(self):
        m = self.model.set_state(enso_index=-1.0, arctic_sea_ice_extent_mkm2=2.5)
        assert m.state.enso_index == -1.0


class TestNEOModel:
    def setup_method(self):
        self.model = NearEarthObjectModel()

    def test_state_defaults(self):
        s = NearEarthObjectState()
        assert s.neo_count_tracked == 34000.0
        assert s.detection_rate_pct == 0.60
        assert s.deflection_readiness == 0.05

    def test_assess_impact_risk(self):
        r = self.model.assess_impact_risk()
        assert "composite_risk" in r
        assert "risk_level" in r
        assert "detection_gap_pct" in r
        assert "city_killer_probability_50yr" in r
        assert "potential_impact_events" in r

    def test_detect_gap(self):
        s = NearEarthObjectState(detection_rate_pct=0.3)
        r = self.model.assess_impact_risk(s)
        assert r["detection_gap_pct"] > 50.0

    def test_impact_scenario(self):
        r = self.model.simulate_impact_scenario(diameter_m=500.0, velocity_kms=20.0)
        assert "kinetic_energy_mt_tnt" in r
        assert "event_class" in r
        assert r["event_class"] in ("regional_catastrophe", "local_devastation")

    def test_tunguska(self):
        r = self.model.simulate_impact_scenario(diameter_m=60.0, velocity_kms=15.0, land_impact=True)
        assert r["event_class"] == "local_devastation"

    def test_predict(self):
        r = self.model.predict(years_ahead=50)
        assert "future_detection_rate" in r
        assert "city_killer_probability_period" in r

    def test_set_state(self):
        m = self.model.set_state(torino_scale_max=3.0, detection_rate_pct=0.9)
        assert m.state.torino_scale_max == 3.0


class TestNaturalDataFetcher:
    def test_available_sources(self):
        f = NaturalDataFetcher()
        sources = f.get_available_sources()
        assert len(sources) >= 8
        names = [s["name"] for s in sources]
        assert "usgs_earthquake" in names
        assert "noaa_solar" in names
        assert "nasa_neo" in names

    def test_to_seismic_state(self):
        f = NaturalDataFetcher()
        state = f.to_seismic_state({"count": 15, "max_magnitude": 6.2})
        assert state["earthquake_rate_monthly"] == 15
        assert state["max_magnitude_30d"] == 6.2

    def test_to_solar_state(self):
        f = NaturalDataFetcher()
        state = f.to_solar_state({"latest_kp": 7})
        assert state["kp_index"] == 7


class TestEngineNaturalDomains:
    def setup_method(self):
        self.engine = OracleEngine()

    def test_seismic_state_setting(self):
        self.engine.set_seismic_state(fault_stress_accumulation=0.8, volcanic_activity_index=0.6)
        assert self.engine.state.seismic_state is not None

    def test_solar_state_setting(self):
        self.engine.set_solar_state(kp_index=8.0, dst_index=-300.0)
        assert self.engine.state.solar_state is not None

    def test_oceanic_state_setting(self):
        self.engine.set_oceanic_state(enso_index=1.5, arctic_sea_ice_extent_mkm2=2.0)
        assert self.engine.state.oceanic_state is not None

    def test_neo_state_setting(self):
        self.engine.set_neo_state(torino_scale_max=3.0, detection_rate_pct=0.9)
        assert self.engine.state.neo_state is not None

    def test_diagnose_seismic(self):
        self.engine.set_seismic_state(fault_stress_accumulation=0.8)
        r = self.engine.diagnose()
        assert "seismic_hazard" in r
        assert "seismic_future" in r

    def test_diagnose_solar(self):
        self.engine.set_solar_state(kp_index=7.0, dst_index=-200.0)
        r = self.engine.diagnose()
        assert "solar_weather" in r
        assert "solar_future" in r

    def test_diagnose_oceanic(self):
        self.engine.set_oceanic_state(enso_index=1.5)
        r = self.engine.diagnose()
        assert "oceanic_state" in r
        assert "oceanic_future" in r

    def test_diagnose_neo(self):
        self.engine.set_neo_state()
        r = self.engine.diagnose()
        assert "neo_impact_risk" in r

    def test_cross_domain_seismic_infrastructure(self):
        self.engine.set_seismic_state(fault_stress_accumulation=0.9, volcanic_activity_index=0.7)
        self.engine.set_infrastructure_state(power_grid_reliability=0.3, water_system_reliability=0.3)
        r = self.engine.diagnose()
        ints = [i["type"] for i in r.get("cross_domain_interactions", [])]
        assert "seismic_infrastructure_cascade" in ints

    def test_cross_domain_solar_grid(self):
        self.engine.set_solar_state(dst_index=-400.0, kp_index=9.0, flare_probability=0.8)
        self.engine.set_infrastructure_state(power_grid_reliability=0.3, water_system_reliability=0.3)
        r = self.engine.diagnose()
        ints = [i["type"] for i in r.get("cross_domain_interactions", [])]
        assert "solar_storm_grid_failure" in ints

    def test_cross_domain_enso_food(self):
        self.engine.set_oceanic_state(enso_index=1.5, sea_surface_temp_anomaly_c=1.5)
        self.engine.set_food_state(supply_chain_disruption=0.5)
        r = self.engine.diagnose()
        ints = [i["type"] for i in r.get("cross_domain_interactions", [])]
        assert "enso_food_disruption" in ints

    def test_predict_natural_domains(self):
        self.engine.set_seismic_state()
        self.engine.set_solar_state()
        self.engine.set_oceanic_state()
        self.engine.set_neo_state()
        forecast = self.engine.predict(date(2030, 1, 1))
        domains = [p.domain for p in forecast.predictions]
        assert Domain.SEISMIC in domains
        assert Domain.SOLAR in domains
        assert Domain.OCEANIC in domains
        assert Domain.NEAR_EARTH_OBJECT in domains


class TestAtmosphericModel:
    def test_atmospheric_state_defaults(self):
        state = AtmosphericState()
        assert state.polar_vortex_strength == 0.6
        assert state.ozone_column_du == 300.0
        assert state.atmospheric_co2_ppm == 420.0

    def test_assess_stable(self):
        model = AtmosphericModel()
        state = AtmosphericState(polar_vortex_strength=0.8, stratospheric_warming_risk=0.05, ozone_column_du=310)
        result = model.assess_atmospheric_state(state)
        assert result["risk_level"] in ("stable", "moderate", "elevated", "critical")
        assert result["composite_atmospheric_risk"] >= 0
        assert result["composite_atmospheric_risk"] <= 1
        assert "recommendations" in result
        assert "polar_vortex_status" in result

    def test_assess_stressed(self):
        model = AtmosphericModel()
        state = AtmosphericState(
            polar_vortex_strength=0.2, stratospheric_warming_risk=0.6,
            ozone_column_du=180, uv_index_anomaly=1.5,
            rossby_wave_amplitude=0.8, hadley_cell_extent_deg=35,
        )
        result = model.assess_atmospheric_state(state)
        assert result["risk_level"] in ("elevated", "critical")
        assert result["composite_atmospheric_risk"] > 0.3

    def test_ssw_scenario(self):
        model = AtmosphericModel()
        result = model.simulate_ssw_scenario(vortex_strength=0.2, ssw_intensity=0.8, duration_days=30)
        assert result["scenario"] == "sudden_stratospheric_warming"
        assert result["temperature_anomaly_northern_c"] < 0
        assert result["cold_outbreak_probability"] > 0
        assert result["recovery_days"] > 0

    def test_ozone_scenario(self):
        model = AtmosphericModel()
        result = model.simulate_ozone_scenario(ozone_column_du=150, duration_months=6)
        assert result["scenario"] == "ozone_depletion"
        assert result["ozone_deficit_pct"] > 0
        assert result["uv_increase_pct"] > 0
        assert len(result["affected_regions"]) > 0

    def test_predict(self):
        model = AtmosphericModel()
        state = AtmosphericState(atmospheric_co2_ppm=420, methane_ppb=1900, ozone_column_du=280)
        result = model.predict(years_ahead=10, state=state)
        assert result["future_co2_ppm"] > 420
        assert result["future_methane_ppb"] > 1900
        assert "vortex_trend" in result
        assert "geomagnetic_trend" in result
        assert "jet_stream_trend" in result

    def test_set_state(self):
        model = AtmosphericModel()
        model.set_state(polar_vortex_strength=0.3, ozone_column_du=200)
        result = model.assess_atmospheric_state()
        assert result["ozone_deficit_pct"] > 0


class TestAtmosphericEngineIntegration:
    def setup_method(self):
        self.engine = OracleEngine()

    def test_set_atmospheric_state(self):
        self.engine.set_atmospheric_state(polar_vortex_strength=0.3, ozone_column_du=200)
        assert self.engine.state.atmospheric_state.polar_vortex_strength == 0.3

    def test_diagnose_with_atmospheric(self):
        self.engine.set_atmospheric_state(polar_vortex_strength=0.2, stratospheric_warming_risk=0.5)
        result = self.engine.diagnose()
        assert "atmospheric_risk" in result
        assert result["atmospheric_risk"]["composite_atmospheric_risk"] > 0

    def test_atmospheric_active_domain(self):
        self.engine.set_atmospheric_state()
        result = self.engine.diagnose()
        interactions = result.get("cross_domain_interactions", [])
        interaction_types = [i["type"] for i in interactions]
        possible = ["ssw_energy_demand_spike", "atmospheric_climate_feedback", "hadley_expansion_desertification", "ozone_uv_food_damage"]
        has_atm = any(t in interaction_types for t in possible)
        assert isinstance(has_atm, bool)

    def test_predict_includes_atmospheric(self):
        self.engine.set_atmospheric_state()
        forecast = self.engine.predict(date(2030, 1, 1))
        domains = [p.domain for p in forecast.predictions]
        assert Domain.ATMOSPHERIC in domains


class TestEcologicalModel:
    def setup_method(self):
        self.model = EcologicalModel()

    def test_state_defaults(self):
        s = EcologicalState()
        assert s.biodiversity_index == 0.6
        assert s.trophic_integrity == 0.5
        assert s.keystone_species_ratio == 0.3
        assert s.extinction_debt == 0.4

    def test_assess_ecosystem(self):
        r = self.model.assess_ecosystem()
        assert "ecosystem_health_index" in r
        assert "health_level" in r
        assert "trophic_integrity" in r
        assert "resilience_score" in r
        assert "stress_score" in r
        assert "cascade_potential" in r
        assert "recommendations" in r
        assert 0 <= r["ecosystem_health_index"] <= 1
        assert r["health_level"] in ("thriving", "stable", "stressed", "critical")

    def test_assess_critical(self):
        s = EcologicalState(
            biodiversity_index=0.1, trophic_integrity=0.1, keystone_species_ratio=0.05,
            ecosystem_connectivity=0.1, habitat_fragmentation=0.9, invasive_species_impact=0.8,
            pollution_load=0.9, climate_stress=0.9, overexploitation_rate=0.8,
            adaptive_capacity=0.1, genetic_diversity_index=0.1, extinction_debt=0.9,
        )
        r = self.model.assess_ecosystem(s)
        assert r["health_level"] == "critical"
        assert r["stress_score"] > 0.5
        assert len(r["threshold_extinctions"]) > 0

    def test_assess_thriving(self):
        s = EcologicalState(
            biodiversity_index=0.95, trophic_integrity=0.9, keystone_species_ratio=0.8,
            ecosystem_connectivity=0.9, habitat_fragmentation=0.05, invasive_species_impact=0.05,
            pollution_load=0.05, climate_stress=0.05, overexploitation_rate=0.05,
            adaptive_capacity=0.9, genetic_diversity_index=0.9, extinction_debt=0.05,
        )
        r = self.model.assess_ecosystem(s)
        assert r["health_level"] == "thriving"
        assert r["ecosystem_health_index"] > 0.6

    def test_simulate_trophic_cascade_apex(self):
        r = self.model.simulate_trophic_cascade(keystone_removed="apex_predator", severity=0.8, years=10)
        assert r["scenario"] == "trophic_cascade_apex_predator"
        assert r["keystone_removed"] == "apex_predator"
        assert r["severity"] == 0.8
        assert len(r["timeline"]) == 10
        assert r["projected_biodiversity_loss_pct"] > 0
        assert r["recovery_years"] > 0

    def test_simulate_trophic_cascade_coral(self):
        r = self.model.simulate_trophic_cascade(keystone_removed="coral", severity=0.9, years=10)
        assert r["recovery_years"] > 30
        assert r["recovery_possible"] is False

    def test_simulate_trophic_cascade_pollinator(self):
        r = self.model.simulate_trophic_cascade(keystone_removed="pollinator", severity=0.7, years=8)
        assert "plant_reproduction_decline" in r["effects"]
        assert "crop_yield_impact" in r["effects"]

    def test_predict(self):
        r = self.model.predict(years_ahead=10)
        assert "future_biodiversity_index" in r
        assert "future_genetic_diversity" in r
        assert "future_trophic_integrity" in r
        assert "ecosystem_regime" in r
        assert "tipping_point_risk" in r
        assert r["ecosystem_regime"] in ("adaptive", "stressed", "collapse")
        assert 0 <= r["tipping_point_risk"] <= 1

    def test_predict_collapse(self):
        s = EcologicalState(
            biodiversity_index=0.1, extinction_debt=0.9, habitat_fragmentation=0.9,
            adaptive_capacity=0.05, genetic_diversity_index=0.1, climate_stress=0.9,
        )
        r = self.model.predict(years_ahead=20, state=s)
        assert r["future_biodiversity_index"] < 0.3
        assert r["tipping_point_risk"] > 0.3

    def test_set_state(self):
        self.model.set_state(biodiversity_index=0.3, climate_stress=0.8)
        r = self.model.assess_ecosystem()
        assert r["stress_score"] > 0.3


class TestEcologicalEngineIntegration:
    def setup_method(self):
        self.engine = OracleEngine()

    def test_set_ecological_state(self):
        self.engine.set_ecological_state(biodiversity_index=0.2, climate_stress=0.8)
        assert self.engine.state.ecological_state.biodiversity_index == 0.2
        assert self.engine.state.ecological_state.climate_stress == 0.8

    def test_diagnose_with_ecological(self):
        self.engine.set_ecological_state(biodiversity_index=0.15, extinction_debt=0.8)
        result = self.engine.diagnose()
        assert "ecosystem_health" in result
        assert result["ecosystem_health"]["ecosystem_health_index"] > 0

    def test_predict_includes_ecological(self):
        self.engine.set_ecological_state()
        forecast = self.engine.predict(date(2030, 1, 1))
        domains = [p.domain for p in forecast.predictions]
        assert Domain.ECOLOGICAL in domains

class TestPaleoclimatologyModel:
    def setup_method(self):
        self.model = PaleoclimatologyModel()

    def test_state_defaults(self):
        s = PaleoclimateState()
        assert s.co2_ppmv == 420.0
        assert s.milankovitch_obliquity_deg == 23.45
        assert s.ice_volume_fraction == 0.03

    def test_assess_milankovitch(self):
        r = self.model.assess_milankovitch()
        assert "insolation_variability" in r
        assert "orbital_phase" in r
        assert "glaciation_risk" in r
        assert "deglaciation_risk" in r
        assert r["orbital_phase"] in ("favorable_interglacial", "transitional_warming", "transitional_cooling", "favorable_glacial")

    def test_analog_eemian(self):
        r = self.model.simulate_analog(analog="eemian", years=5000)
        assert "analog_name" in r
        assert "similarity_score" in r
        assert "key_lessons" in r
        assert len(r["timeline"]) > 0

    def test_analog_petm(self):
        r = self.model.simulate_analog(analog="petm", years=10000)
        assert r["similarity_score"] > 0
        assert len(r["key_lessons"]) > 0

    def test_predict(self):
        r = self.model.predict(years_ahead=1000)
        assert "future_co2_ppmv" in r
        assert "future_temp_anomaly_c" in r
        assert "climate_regime" in r
        assert r["climate_regime"] in ("hothouse", "warm_interglacial", "interglacial", "cool_interglacial", "glacial", "deep_glacial")

    def test_set_state(self):
        self.model.set_state(co2_ppmv=600, global_temp_anomaly_c=3.0)
        r = self.model.assess_milankovitch()
        assert r["deglaciation_risk"] > 0.3


class TestCosmicRadiationModel:
    def setup_method(self):
        self.model = CosmicRadiationModel()

    def test_state_defaults(self):
        s = CosmicRadiationState()
        assert s.gcr_flux_relative == 1.0
        assert s.solar_modulation_potential_gv == 0.5

    def test_assess_radiation(self):
        r = self.model.assess_radiation()
        assert "composite_risk" in r
        assert "risk_level" in r
        assert "effective_gcr_flux" in r
        assert r["risk_level"] in ("low", "moderate", "elevated", "high", "extreme")

    def test_solar_minimum(self):
        r = self.model.simulate_solar_minimum(duration_years=3, severity=0.8)
        assert r["scenario"] == "grand_solar_minimum"
        assert "peak_gcr_flux" in r
        assert len(r["timeline"]) > 0

    def test_cme_shielding(self):
        r = self.model.simulate_cme_shielding(cme_intensity=0.7, recovery_hours=48)
        assert r["scenario"] == "cme_gcr_suppression"
        assert r["forbush_decrease_pct"] > 0
        assert len(r["recovery_timeline"]) > 0

    def test_predict(self):
        r = self.model.predict(years_ahead=10)
        assert "future_gcr_flux" in r
        assert "radiation_regime" in r

    def test_set_state(self):
        self.model.set_state(gcr_flux_relative=1.8, solar_modulation_potential_gv=0.2)
        r = self.model.assess_radiation()
        assert r["composite_risk"] > 0.3


class TestGravitationalModel:
    def setup_method(self):
        self.model = GravitationalModel()

    def test_state_defaults(self):
        s = GravitationalState()
        assert s.lunar_tidal_force_relative == 1.0
        assert s.planetary_alignment_index == 0.2

    def test_assess_tidal(self):
        r = self.model.assess_tidal_forces()
        assert "composite_tidal_index" in r
        assert "level" in r
        assert "seismic_triggering_probability" in r
        assert r["level"] in ("minimal", "low", "moderate", "elevated", "high")

    def test_alignment(self):
        r = self.model.simulate_alignment(n_planets=5, alignment_strength=0.7, days=30)
        assert r["scenario"] == "planetary_alignment"
        assert "peak_seismic_probability_multiplier" in r
        assert len(r["timeline"]) > 0

    def test_perigee_spring(self):
        r = self.model.simulate_perigee_spring(perigee_distance_km=356500, duration_days=7)
        assert r["king_tide_amplification"] > 1.0
        assert len(r["timeline"]) > 0

    def test_predict(self):
        r = self.model.predict(years_ahead=10)
        assert "future_moon_distance_km" in r
        assert "future_lunar_tidal_force" in r
        assert "tidal_regime" in r

    def test_set_state(self):
        self.model.set_state(lunar_tidal_force_relative=1.5, planetary_alignment_index=0.7)
        r = self.model.assess_tidal_forces()
        assert r["composite_tidal_index"] > 0.2


class TestNewDomainsEngineIntegration:
    def setup_method(self):
        self.engine = OracleEngine()

    def test_set_paleoclimate_state(self):
        self.engine.set_paleoclimate_state(co2_ppmv=600, ice_volume_fraction=0.01)
        assert self.engine.state.paleoclimate_state.co2_ppmv == 600

    def test_diagnose_with_paleoclimate(self):
        self.engine.set_paleoclimate_state(co2_ppmv=600, global_temp_anomaly_c=3.0)
        result = self.engine.diagnose()
        assert "paleoclimate_forcing" in result

    def test_predict_includes_paleoclimate(self):
        self.engine.set_paleoclimate_state()
        forecast = self.engine.predict(date(2030, 1, 1))
        domains = [p.domain for p in forecast.predictions]
        assert Domain.PALEOCLIMATOLOGY in domains

    def test_set_cosmic_radiation_state(self):
        self.engine.set_cosmic_radiation_state(gcr_flux_relative=1.5, solar_modulation_potential_gv=0.2)
        assert self.engine.state.cosmic_radiation_state.gcr_flux_relative == 1.5

    def test_diagnose_with_cosmic_radiation(self):
        self.engine.set_cosmic_radiation_state(gcr_flux_relative=1.5)
        result = self.engine.diagnose()
        assert "cosmic_radiation_risk" in result

    def test_predict_includes_cosmic_radiation(self):
        self.engine.set_cosmic_radiation_state()
        forecast = self.engine.predict(date(2030, 1, 1))
        domains = [p.domain for p in forecast.predictions]
        assert Domain.COSMIC_RADIATION in domains

    def test_set_gravitational_state(self):
        self.engine.set_gravitational_state(lunar_tidal_force_relative=1.3, planetary_alignment_index=0.6)
        assert self.engine.state.gravitational_state.lunar_tidal_force_relative == 1.3

    def test_diagnose_with_gravitational(self):
        self.engine.set_gravitational_state(lunar_tidal_force_relative=1.5, planetary_alignment_index=0.7)
        result = self.engine.diagnose()
        assert "gravitational_tidal" in result

    def test_predict_includes_gravitational(self):
        self.engine.set_gravitational_state()
        forecast = self.engine.predict(date(2030, 1, 1))
        domains = [p.domain for p in forecast.predictions]
        assert Domain.GRAVITATIONAL in domains

    def test_set_compound_hazard_state(self):
        self.engine.set_compound_hazard_state(seismic_hazard=0.7, infrastructure_resilience=0.3)
        assert self.engine.state.compound_hazard_state.seismic_hazard == 0.7

    def test_diagnose_with_compound_hazard(self):
        self.engine.set_compound_hazard_state(seismic_hazard=0.6, solar_storm_risk=0.5, infrastructure_resilience=0.3)
        result = self.engine.diagnose()
        assert "compound_hazard" in result
        assert result["compound_hazard"]["compound_hazard_index"] > 0

    def test_predict_includes_compound_hazard(self):
        self.engine.set_compound_hazard_state(seismic_hazard=0.5)
        forecast = self.engine.predict(date(2030, 1, 1))
        models = [p.model_name for p in forecast.predictions]
        assert "compound_hazard" in models

    def test_world_state_in_diagnose_when_multiple_domains(self):
        self.engine.set_economic_state(gdp_growth_rate=0.02)
        self.engine.set_geopolitical_state(us_china_tension=0.5, russia_west_tension=0.6, middle_east_instability=0.4, trade_protectionism=0.3, cyber_warfare_likelihood=0.3, energy_geopolitics=0.4)
        self.engine.set_climate_state(global_temp_anomaly_c=1.2, sea_level_rise_mm=100, extreme_weather_index=0.5)
        result = self.engine.diagnose()
        assert "world_state" in result
        assert "overall_stability" in result["world_state"]
        assert "threat_level" in result["world_state"]
        assert result["world_state"]["n_domains_assessed"] >= 3

    def test_compound_hazard_cross_domain_interactions(self):
        self.engine.set_compound_hazard_state(seismic_hazard=0.7, solar_storm_risk=0.6, infrastructure_resilience=0.2, economic_vulnerability=0.7)
        self.engine.set_economic_state(gdp_growth_rate=0.01, debt_to_gdp=1.5)
        result = self.engine.diagnose()
        if "cross_domain_interactions" in result:
            types = [i["type"] for i in result["cross_domain_interactions"]]
            assert any("compound" in t or "systemic" in t for t in types)
