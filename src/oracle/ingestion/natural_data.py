import json
import time
from typing import Optional
from datetime import datetime, timezone


class NaturalDataFetcher:
    REAL_TIME_APIS = {
        "usgs_earthquake": {
            "url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson",
            "description": "USGS significant earthquakes in the past 30 days",
            "refresh_minutes": 15,
        },
        "usgs_all_hour": {
            "url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
            "description": "USGS all earthquakes in the past hour",
            "refresh_minutes": 1,
        },
        "noaa_solar": {
            "url": "https://services.swpc.noaa.gov/products/noaa-scales-planetary-k-index-forecast.json",
            "description": "NOAA SWPC planetary K-index forecast",
            "refresh_minutes": 15,
        },
        "noaa_solar_wind": {
            "url": "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json",
            "description": "NOAA solar wind magnetic field data",
            "refresh_minutes": 5,
        },
        "noaa_flare": {
            "url": "https://services.swpc.noaa.gov/json/fld_1d.json",
            "description": "Solar flare probability forecast",
            "refresh_minutes": 60,
        },
        "noaa_aurora": {
            "url": "https://services.swpc.noaa.gov/json/aurora_forecast.json",
            "description": "NOAA aurora forecast (3-hour intervals)",
            "refresh_minutes": 60,
        },
        "nasa_neo": {
            "url": "https://api.nasa.gov/neo/rest/v1/feed?api_key=DEMO_KEY",
            "description": "NASA Near-Earth Object feed (requires API key)",
            "refresh_minutes": 1440,
        },
        "nasa_cme": {
            "url": "https://api.nasa.gov/DONKI/CME?api_key=DEMO_KEY",
            "description": "NASA DONKI CME catalog",
            "refresh_minutes": 1440,
        },
        "noaa_enso": {
            "url": "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.81-10.ascii",
            "description": "NOAA ENSO index historical data",
            "refresh_minutes": 10080,
        },
        "noaa_co2": {
            "url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt",
            "description": "NOAA Mauna Loa CO2 monthly mean data",
            "refresh_minutes": 10080,
        },
        "noaa_gcr": {
            "url": "https://services.swpc.noaa.gov/json/goes/particle/m1.json",
            "description": "NOAA GOES proton flux (GCR/solar particle proxy)",
            "refresh_minutes": 15,
        },
        "noaa_geomagnetic": {
            "url": "https://services.swpc.noaa.gov/json/kyoto_dst_1h.json",
            "description": "NOAA geomagnetic Dst index (magnetosphere compression proxy)",
            "refresh_minutes": 60,
        },
    }

    def __init__(self, cache_ttl: int = 900):
        self.cache_ttl = cache_ttl
        self._cache = {}

    def _fetch_http(self, url: str, timeout: int = 15) -> Optional[dict]:
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "OraclePredictionEngine/0.3.4"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    raw = resp.read().decode("utf-8")
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        return {"raw_text": raw[:5000]}
        except Exception:
            return None

    def fetch_usgs_earthquakes(self) -> Optional[dict]:
        cached = self._cache.get("usgs_earthquake")
        if cached and time.time() - cached["ts"] < self.cache_ttl:
            return cached["data"]
        data = self._fetch_http(self.REAL_TIME_APIS["usgs_earthquake"]["url"])
        if data and "features" in data:
            features = data["features"]
            quakes = []
            for f in features[:20]:
                props = f.get("properties", {})
                quakes.append({
                    "magnitude": props.get("mag", 0),
                    "place": props.get("place", ""),
                    "time": props.get("time", 0),
                    "tsunami": props.get("tsunami", 0),
                    "type": props.get("type", ""),
                    "depth_km": f.get("geometry", {}).get("coordinates", [0, 0, 0])[2] if f.get("geometry") else 0,
                })
            result = {
                "count": len(features),
                "significant_quakes": quakes,
                "max_magnitude": max((q["magnitude"] for q in quakes if q["magnitude"]), default=0),
                "source": "USGS",
            }
            self._cache["usgs_earthquake"] = {"ts": time.time(), "data": result}
            return result
        return None

    def fetch_noaa_solar(self) -> Optional[dict]:
        cached = self._cache.get("noaa_solar")
        if cached and time.time() - cached["ts"] < self.cache_ttl:
            return cached["data"]
        data = self._fetch_http(self.REAL_TIME_APIS["noaa_solar"]["url"])
        if data and isinstance(data, list) and len(data) > 1:
            rows = data[1:] if data[0] else data
            latest = rows[-1] if rows else None
            result = {
                "raw_forecast": rows[-5:] if len(rows) > 5 else rows,
                "latest_kp": float(latest[1]) if latest and len(latest) > 1 else 0,
                "source": "NOAA SWPC",
            }
            self._cache["noaa_solar"] = {"ts": time.time(), "data": result}
            return result
        return None

    def fetch_noaa_flare(self) -> Optional[dict]:
        cached = self._cache.get("noaa_flare")
        if cached and time.time() - cached["ts"] < self.cache_ttl:
            return cached["data"]
        data = self._fetch_http(self.REAL_TIME_APIS["noaa_flare"]["url"])
        if data and isinstance(data, list) and len(data) > 1:
            rows = data[1:]
            result = {
                "flare_forecast": rows[-3:] if len(rows) >= 3 else rows,
                "source": "NOAA SWPC",
            }
            self._cache["noaa_flare"] = {"ts": time.time(), "data": result}
            return result
        return None

    def fetch_nasa_neo(self, api_key: str = "DEMO_KEY") -> Optional[dict]:
        cached = self._cache.get("nasa_neo")
        if cached and time.time() - cached["ts"] < 86400:
            return cached["data"]
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={today}&end_date={today}&api_key={api_key}"
        data = self._fetch_http(url)
        if data and "near_earth_objects" in data:
            neos_today = data["near_earth_objects"].get(today, [])
            objects = []
            for neo in neos_today[:10]:
                objects.append({
                    "name": neo.get("name", ""),
                    "diameter_m": float(neo.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_max", 0)),
                    "velocity_kms": float(neo.get("close_approach_data", [{}])[0].get("relative_velocity", {}).get("kilometers_per_second", 0)),
                    "miss_distance_ld": float(neo.get("close_approach_data", [{}])[0].get("miss_distance", {}).get("lunar", 0)),
                    "hazardous": neo.get("is_potentially_hazardous_asteroid", False),
                })
            result = {
                "count": len(neos_today),
                "objects": objects,
                "source": "NASA NEO",
            }
            self._cache["nasa_neo"] = {"ts": time.time(), "data": result}
            return result
        return None

    def to_seismic_state(self, usgs_data: Optional[dict] = None) -> dict:
        if usgs_data:
            return {
                "earthquake_rate_monthly": usgs_data.get("count", 0),
                "max_magnitude_30d": usgs_data.get("max_magnitude", 0),
            }
        return {"earthquake_rate_monthly": 12.0, "max_magnitude_30d": 5.5}

    def to_solar_state(self, noaa_data: Optional[dict] = None, flare_data: Optional[dict] = None) -> dict:
        state = {"sunspot_number": 80.0, "solar_flux_10_7cm": 130.0, "kp_index": 3.0, "dst_index": -20.0}
        if noaa_data:
            try:
                state["kp_index"] = float(noaa_data.get("latest_kp", 3))
            except (ValueError, TypeError):
                pass
        if flare_data:
            try:
                state["flare_probability"] = float(flare_data.get("m_flare_prob", 0.3))
            except (ValueError, TypeError):
                pass
        return state

    def to_oceanic_state(self, enso_data: Optional[dict] = None) -> dict:
        state = {"enso_index": 0.5, "sea_surface_temp_anomaly_c": 0.8, "arctic_sea_ice_extent_mkm2": 4.5}
        if enso_data:
            try:
                state["enso_index"] = float(enso_data.get("latest_nino34", 0.5))
            except (ValueError, TypeError):
                pass
        return state

    def to_neo_state(self, neo_data: Optional[dict] = None) -> dict:
        state = {"neo_count_tracked": 34000.0, "neo_count_hazardous": 2300.0, "detection_rate_pct": 0.6}
        if neo_data:
            try:
                state["neo_count_tracked"] = float(neo_data.get("element_count", 34000))
            except (ValueError, TypeError):
                pass
        return state

    def to_atmospheric_state(self, solar_state: Optional[dict] = None) -> dict:
        state = {"polar_vortex_strength": 0.6, "stratospheric_warming_risk": 0.15, "ozone_column_du": 300.0}
        if solar_state:
            kp = solar_state.get("kp_index", 3.0)
            if kp > 5:
                state["polar_vortex_strength"] = max(0.1, 0.6 - (kp - 5) * 0.1)
                state["stratospheric_warming_risk"] = min(1.0, 0.15 + (kp - 5) * 0.1)
        return state

    def fetch_enso_historical(self) -> Optional[dict]:
        cached = self._cache.get("noaa_enso")
        if cached and time.time() - cached["ts"] < 10080 * 60:
            return cached["data"]
        url = self.REAL_TIME_APIS["noaa_enso"]["url"]
        data = self._fetch_http(url)
        if data:
            result = {"source": "NOAA ENSO", "latest_nino34": 0.5}
            if isinstance(data, dict) and data.get("raw_text"):
                try:
                    lines = data["raw_text"].strip().split("\\n")
                    if len(lines) > 0:
                        last_line = lines[-1].strip()
                        if last_line:
                            parts = last_line.split()
                            for i, p in enumerate(parts):
                                try:
                                    result["latest_nino34"] = float(p)
                                    result["year"] = int(parts[0]) if len(parts) > 0 else None
                                    result["month"] = int(parts[1]) if len(parts) > 1 else None
                                    break
                                except (ValueError, IndexError):
                                    continue
                except Exception:
                    pass
            elif isinstance(data, list) and len(data) > 0:
                try:
                    last = data[-1] if isinstance(data[-1], (list, dict)) else data[-1]
                    if isinstance(last, dict):
                        result["latest_nino34"] = float(last.get("NINO3.4", last.get("nino34", 0.5)))
                except Exception:
                    pass
            self._cache["noaa_enso"] = {"ts": time.time(), "data": result}
            return result
        return None

    def fetch_co2_data(self) -> Optional[dict]:
        cached = self._cache.get("noaa_co2")
        if cached and time.time() - cached["ts"] < 10080 * 60:
            return cached["data"]
        url = self.REAL_TIME_APIS["noaa_co2"]["url"]
        data = self._fetch_http(url)
        if data:
            result = {"source": "NOAA Mauna Loa", "latest_co2_ppm": 420.0, "trend_ppm_per_year": 2.5}
            if isinstance(data, dict) and data.get("raw_text"):
                try:
                    lines = [l for l in data["raw_text"].strip().split("\\n") if not l.startswith("#")]
                    if len(lines) >= 2:
                        last = lines[-1].split()
                        if len(last) >= 4:
                            result["latest_co2_ppm"] = float(last[3])
                            if len(lines) >= 13:
                                year_ago = lines[-13].split()
                                if len(year_ago) >= 4:
                                    result["trend_ppm_per_year"] = round(float(last[3]) - float(year_ago[3]), 2)
                except Exception:
                    pass
            self._cache["noaa_co2"] = {"ts": time.time(), "data": result}
            return result
        return None

    def fetch_gcr_proxy(self) -> Optional[dict]:
        cached = self._cache.get("noaa_gcr")
        if cached and time.time() - cached["ts"] < self.cache_ttl:
            return cached["data"]
        url = self.REAL_TIME_APIS["noaa_gcr"]["url"]
        data = self._fetch_http(url)
        if data:
            result = {"source": "NOAA GOES Proton Flux", "latest_proton_flux": 1.0}
            if isinstance(data, list) and len(data) > 0:
                try:
                    last = data[-1] if isinstance(data[-1], dict) else data[-1]
                    if isinstance(last, dict):
                        for key in ["flux", "proton_flux", "p_flux"]:
                            if key in last:
                                result["latest_proton_flux"] = float(last[key])
                                break
                except Exception:
                    pass
            self._cache["noaa_gcr"] = {"ts": time.time(), "data": result}
            return result
        return None

    def fetch_geomagnetic(self) -> Optional[dict]:
        cached = self._cache.get("noaa_geomagnetic")
        if cached and time.time() - cached["ts"] < self.cache_ttl:
            return cached["data"]
        url = self.REAL_TIME_APIS["noaa_geomagnetic"]["url"]
        data = self._fetch_http(url)
        if data:
            result = {"source": "NOAA Geomagnetic", "latest_dst": -20.0}
            if isinstance(data, list) and len(data) > 0:
                try:
                    last = data[-1] if isinstance(data[-1], dict) else data[-1]
                    if isinstance(last, dict):
                        for key in ["dst", "Dst", "dst_index"]:
                            if key in last:
                                result["latest_dst"] = float(last[key])
                                break
                except Exception:
                    pass
            self._cache["noaa_geomagnetic"] = {"ts": time.time(), "data": result}
            return result
        return None

    def to_paleoclimate_state(self, co2_data: Optional[dict] = None) -> dict:
        state = {
            "milankovitch_obliquity_deg": 23.44,
            "milankovitch_eccentricity": 0.0167,
            "milankovitch_precession": 0.0,
            "ice_volume_fraction": 0.01,
            "co2_ppmv": 420.0,
            "global_temp_anomaly_c": 1.2,
            "solar_luminosity_fraction": 1.0,
            "volcanic_aerosol_loading": 0.0,
            "thermohaline_circulation_strength": 0.8,
            "vegetation_cover_fraction": 0.6,
            "methane_ppbv": 1900.0,
            "albedo": 0.30,
            "sea_level_m_above_present": 0.0,
        }
        if co2_data:
            try:
                state["co2_ppmv"] = float(co2_data.get("latest_co2_ppm", 420))
                state["global_temp_anomaly_c"] = max(0, (state["co2_ppmv"] - 280) * 0.008)
            except (ValueError, TypeError):
                pass
        return state

    def to_cosmic_radiation_state(self, gcr_data: Optional[dict] = None, solar_state: Optional[dict] = None) -> dict:
        state = {
            "gcr_flux_relative": 1.0,
            "solar_modulation_potential_gv": 0.5,
            "geomagnetic_shielding_eff": 0.8,
            "atmospheric_shielding_eff": 0.95,
            "svensmark_cloud_nucleation_factor": 0.05,
            "neutron_monitor_count_rate": 300.0,
            "cosmic_ray_dose_rate_usv_hr": 0.3,
            "space_weather_shielding": 0.7,
            "magnetosphere_compression": 0.2,
            "ionization_rate_cm3_s": 2.0,
        }
        if gcr_data:
            try:
                proton_flux = float(gcr_data.get("latest_proton_flux", 1.0))
                state["gcr_flux_relative"] = max(0.1, proton_flux)
            except (ValueError, TypeError):
                pass
        if solar_state:
            kp = solar_state.get("kp_index", 3.0)
            if kp > 5:
                state["magnetosphere_compression"] = min(1.0, 0.2 + (kp - 5) * 0.15)
                state["solar_modulation_potential_gv"] = min(1.5, 0.5 + (kp - 5) * 0.2)
                state["gcr_flux_relative"] = max(0.3, 1.0 - (kp - 5) * 0.1)
        return state

    def to_gravitational_state(self, geomagnetic_data: Optional[dict] = None) -> dict:
        state = {
            "lunar_tidal_force_relative": 1.0,
            "solar_tidal_force_relative": 0.46,
            "planetary_alignment_index": 0.2,
            "earth_moon_distance_km": 384400.0,
            "tidal_period_hours": 12.42,
            "tidal_range_variation_pct": 5.0,
            "solid_earth_tide_amplitude_cm": 30.0,
            "ocean_tide_amplitude_m": 1.0,
            "resonance_frequency_hz": 0.0001,
            "rotational_energy_dissipation_pw": 3.7,
            "crustal_stress_tidal_mpa": 0.3,
        }
        if geomagnetic_data:
            try:
                dst = float(geomagnetic_data.get("latest_dst", -20))
                if dst < -100:
                    state["planetary_alignment_index"] = min(1.0, 0.2 + (-dst - 100) * 0.01)
            except (ValueError, TypeError):
                pass
        return state

    def get_available_sources(self) -> list:
        return [
            {"name": k, "url": v["url"], "description": v["description"], "refresh_minutes": v["refresh_minutes"]}
            for k, v in self.REAL_TIME_APIS.items()
        ]