"""Physical Constants module for fundamental constants, dimensional analysis, and unit conversions.

Provides CODATA-recommended values, natural unit systems, dimensional analysis,
and physical constant lookups organized by category.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


FUNDAMENTAL_CONSTANTS = {
    "speed_of_light": {"value": 299792458.0, "unit": "m/s", "symbol": "c", "uncertainty": 0.0, "category": "universal"},
    "gravitational_constant": {"value": 6.67430e-11, "unit": "m^3/(kg*s^2)", "symbol": "G", "uncertainty": 1.5e-15, "category": "universal"},
    "planck_constant": {"value": 6.62607015e-34, "unit": "J*s", "symbol": "h", "uncertainty": 0.0, "category": "quantum"},
    "reduced_planck": {"value": 1.054571817e-34, "unit": "J*s", "symbol": "hbar", "uncertainty": 0.0, "category": "quantum"},
    "boltzmann_constant": {"value": 1.380649e-23, "unit": "J/K", "symbol": "k_B", "uncertainty": 0.0, "category": "thermodynamic"},
    "avogadro_number": {"value": 6.02214076e23, "unit": "mol^-1", "symbol": "N_A", "uncertainty": 0.0, "category": "thermodynamic"},
    "electron_mass": {"value": 9.1093837015e-31, "unit": "kg", "symbol": "m_e", "uncertainty": 2.8e-40, "category": "particle"},
    "proton_mass": {"value": 1.67262192369e-27, "unit": "kg", "symbol": "m_p", "uncertainty": 5.1e-37, "category": "particle"},
    "neutron_mass": {"value": 1.67492749804e-27, "unit": "kg", "symbol": "m_n", "uncertainty": 9.5e-35, "category": "particle"},
    "electron_charge": {"value": 1.602176634e-19, "unit": "C", "symbol": "e", "uncertainty": 0.0, "category": "electromagnetic"},
    "vacuum_permittivity": {"value": 8.8541878128e-12, "unit": "F/m", "symbol": "epsilon_0", "uncertainty": 1.3e-21, "category": "electromagnetic"},
    "vacuum_permeability": {"value": 1.25663706212e-6, "unit": "N/A^2", "symbol": "mu_0", "uncertainty": 1.9e-16, "category": "electromagnetic"},
    "fine_structure_constant": {"value": 7.2973525693e-3, "unit": "", "symbol": "alpha", "uncertainty": 1.1e-12, "category": "quantum"},
    "rydberg_constant": {"value": 1.0973731568160e7, "unit": "m^-1", "symbol": "R_inf", "uncertainty": 2.1e-4, "category": "atomic"},
    "bohr_radius": {"value": 5.29177210903e-11, "unit": "m", "symbol": "a_0", "uncertainty": 8.0e-21, "category": "atomic"},
    "stefan_boltzmann_constant": {"value": 5.670374419e-8, "unit": "W/(m^2*K^4)", "symbol": "sigma", "uncertainty": 0.0, "category": "thermodynamic"},
    "molar_gas_constant": {"value": 8.314462618, "unit": "J/(mol*K)", "symbol": "R", "uncertainty": 0.0, "category": "thermodynamic"},
    "atomic_mass_unit": {"value": 1.66053906660e-27, "unit": "kg", "symbol": "u", "uncertainty": 5.0e-37, "category": "atomic"},
    "planck_mass": {"value": 2.176434e-8, "unit": "kg", "symbol": "m_P", "uncertainty": 2.4e-13, "category": "planck"},
    "planck_length": {"value": 1.616255e-35, "unit": "m", "symbol": "l_P", "uncertainty": 1.8e-40, "category": "planck"},
    "planck_time": {"value": 5.391247e-44, "unit": "s", "symbol": "t_P", "uncertainty": 6.0e-49, "category": "planck"},
    "planck_temperature": {"value": 1.416785e32, "unit": "K", "symbol": "T_P", "uncertainty": 1.6e27, "category": "planck"},
    "planck_energy": {"value": 1.956082e9, "unit": "J", "symbol": "E_P", "uncertainty": 2.2e4, "category": "planck"},
    "wien_displacement_constant": {"value": 2.897771955e-3, "unit": "m*K", "symbol": "b", "uncertainty": 0.0, "category": "thermodynamic"},
    "faraday_constant": {"value": 96485.33212, "unit": "C/mol", "symbol": "F", "uncertainty": 0.0, "category": "electromagnetic"},
    "magnetic_flux_quantum": {"value": 2.067833848e-15, "unit": "Wb", "symbol": "Phi_0", "uncertainty": 1.6e-23, "category": "quantum"},
    "conductance_quantum": {"value": 7.748091729e-5, "unit": "S", "symbol": "G_0", "uncertainty": 6.4e-14, "category": "quantum"},
    "proton_electron_mass_ratio": {"value": 1836.15267343, "unit": "", "symbol": "m_p/m_e", "uncertainty": 1.1e-6, "category": "particle"},
}


UNIT_CONVERSIONS = {
    "length": {"m_to_km": 1e-3, "m_to_cm": 100, "m_to_mm": 1000, "m_to_um": 1e6, "m_to_nm": 1e9,
               "m_to_inch": 39.3701, "m_to_foot": 3.28084, "m_to_mile": 6.2137e-4,
               "m_to_au": 6.68459e-12, "m_to_ly": 1.057e-16, "m_to_pc": 3.24078e-17},
    "mass": {"kg_to_g": 1000, "kg_to_mg": 1e6, "kg_to_lb": 2.20462, "kg_to_oz": 35.274,
             "kg_to_amu": 6.02214e26, "kg_to_eV": 5.60959e35, "kg_to_solar_mass": 5.02785e-27},
    "time": {"s_to_ms": 1000, "s_to_us": 1e6, "s_to_ns": 1e9, "s_to_min": 1/60, "s_to_hr": 1/3600,
             "s_to_day": 1/86400, "s_to_yr": 3.17098e-8, "s_to_yr_julian": 3.16881e-8},
    "energy": {"J_to_eV": 6.24151e18, "J_to_keV": 6.24151e15, "J_to_MeV": 6.24151e12,
               "J_to_GeV": 6.24151e9, "J_to_TeV": 6.24151e6, "J_to_kWh": 2.77778e-7,
               "J_to_cal": 0.239006, "J_to_kcal": 2.39006e-4, "J_to_BTU": 9.47817e-4,
               "J_to_erg": 1e7, "J_to_Hartree": 2.29371e17},
    "temperature": {"K_to_C_offset": -273.15, "K_to_F_factor": 1.8, "K_to_F_offset": -459.67,
                     "C_to_K_offset": 273.15, "C_to_F_factor": 1.8, "C_to_F_offset": 32},
    "pressure": {"Pa_to_atm": 9.86923e-6, "Pa_to_bar": 1e-5, "Pa_to_PSI": 1.45038e-4,
                 "Pa_to_mmHg": 7.50062e-3, "Pa_to_Torr": 7.50062e-3},
    "angle": {"rad_to_deg": 57.2958, "deg_to_rad": 0.0174533, "rad_to_arcmin": 3437.75, "rad_to_arcsec": 206265},
}

DIMENSIONAL_ANALYSIS = {
    "force": {"dimensions": {"length": 1, "mass": 1, "time": -2}, "SI_unit": "N", "symbol": "F"},
    "energy": {"dimensions": {"length": 2, "mass": 1, "time": -2}, "SI_unit": "J", "symbol": "E"},
    "power": {"dimensions": {"length": 2, "mass": 1, "time": -3}, "SI_unit": "W", "symbol": "P"},
    "pressure": {"dimensions": {"length": -1, "mass": 1, "time": -2}, "SI_unit": "Pa", "symbol": "p"},
    "frequency": {"dimensions": {"time": -1}, "SI_unit": "Hz", "symbol": "f"},
    "charge": {"dimensions": {"time": 1, "current": 1}, "SI_unit": "C", "symbol": "Q"},
    "voltage": {"dimensions": {"length": 2, "mass": 1, "time": -3, "current": -1}, "SI_unit": "V", "symbol": "V"},
    "capacitance": {"dimensions": {"length": -2, "mass": -1, "time": 4, "current": 2}, "SI_unit": "F", "symbol": "C"},
    "magnetic_flux": {"dimensions": {"length": 2, "mass": 1, "time": -2, "current": -1}, "SI_unit": "Wb", "symbol": "Phi"},
    "resistance": {"dimensions": {"length": 2, "mass": 1, "time": -3, "current": -2}, "SI_unit": "Ohm", "symbol": "R"},
    "velocity": {"dimensions": {"length": 1, "time": -1}, "SI_unit": "m/s", "symbol": "v"},
    "acceleration": {"dimensions": {"length": 1, "time": -2}, "SI_unit": "m/s^2", "symbol": "a"},
    "momentum": {"dimensions": {"length": 1, "mass": 1, "time": -1}, "SI_unit": "kg*m/s", "symbol": "p"},
    "angular_momentum": {"dimensions": {"length": 2, "mass": 1, "time": -1}, "SI_unit": "J*s", "symbol": "L"},
    "luminosity": {"dimensions": {"length": 2, "mass": 1, "time": -3}, "SI_unit": "W", "symbol": "L"},
}


class PhysicalConstants:
    """Lookup and computation with fundamental physical constants."""

    def __init__(self):
        self.constants = FUNDAMENTAL_CONSTANTS
        self.conversions = UNIT_CONVERSIONS
        self.dimensions = DIMENSIONAL_ANALYSIS

    def get(self, name: str) -> dict:
        key = name.lower().replace(" ", "_")
        if key in self.constants:
            return self.constants[key]
        for k, v in self.constants.items():
            if v["symbol"].lower() == key or k == key:
                return v
        raise KeyError(f"Constant '{name}' not found. Available: {list(self.constants.keys())}")

    def value(self, name: str) -> float:
        return self.get(name)["value"]

    def search(self, query: str) -> List[dict]:
        query_lower = query.lower()
        results = []
        for key, val in self.constants.items():
            if query_lower in key or query_lower in val.get("symbol", "").lower() or query_lower in val.get("category", ""):
                results.append({"name": key, **val})
        return results

    def by_category(self, category: str) -> List[dict]:
        category_lower = category.lower()
        return [{"name": k, **v} for k, v in self.constants.items() if v.get("category", "") == category_lower]

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        from_unit = from_unit.lower().strip()
        to_unit = to_unit.lower().strip()
        if from_unit == to_unit:
            return value
        for category, conversions in self.conversions.items():
            lower_conversions = {k.lower(): v for k, v in conversions.items()}
            from_key = f"{from_unit}_to_{to_unit}"
            if from_key in lower_conversions:
                return value * lower_conversions[from_key]
            to_key = f"{to_unit}_to_{from_unit}"
            if to_key in lower_conversions:
                return value / lower_conversions[to_key]
        known_from = []
        for cat, convs in self.conversions.items():
            for k, v in convs.items():
                if k.startswith(from_unit + "_to_") or k.endswith("_to_" + from_unit):
                    known_from.append(k)
        if "temperature" in self.conversions:
            tc = self.conversions["temperature"]
            if from_unit == "k" and to_unit == "c":
                return value + tc["K_to_C_offset"]
            elif from_unit == "c" and to_unit == "k":
                return value + tc["C_to_K_offset"]
            elif from_unit == "k" and to_unit == "f":
                return value * tc["K_to_F_factor"] + tc["K_to_F_offset"]
            elif from_unit == "f" and to_unit == "k":
                return (value - tc["K_to_F_offset"]) / tc["K_to_F_factor"]
            elif from_unit == "c" and to_unit == "f":
                return value * tc["C_to_F_factor"] + tc["C_to_F_offset"]
            elif from_unit == "f" and to_unit == "c":
                return (value - tc["C_to_F_offset"]) / tc["C_to_F_factor"]
        raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")

    def dimensional_analysis(self, quantity: str) -> dict:
        key = quantity.lower().replace(" ", "_")
        if key in self.dimensions:
            return self.dimensions[key]
        for k, v in self.dimensions.items():
            if v.get("symbol", "").lower() == key:
                return v
        raise KeyError(f"Quantity '{quantity}' not found. Available: {list(self.dimensions.keys())}")

    def natural_units(self) -> dict:
        c = self.constants["speed_of_light"]["value"]
        hbar = self.constants["reduced_planck"]["value"]
        G = self.constants["gravitational_constant"]["value"]
        kB = self.constants["boltzmann_constant"]["value"]
        return {
            "planck_mass": {"value": float(math.sqrt(hbar * c / G)), "unit": "kg", "symbol": "m_P"},
            "planck_length": {"value": float(math.sqrt(hbar * G / c**3)), "unit": "m", "symbol": "l_P"},
            "planck_time": {"value": float(math.sqrt(hbar * G / c**5)), "unit": "s", "symbol": "t_P"},
            "planck_energy": {"value": float(math.sqrt(hbar * c**5 / G)), "unit": "J", "symbol": "E_P"},
            "planck_temperature": {"value": float(math.sqrt(hbar * c**5 / (G * kB**2))), "unit": "K", "symbol": "T_P"},
            "planck_charge": {"value": float(math.sqrt(4 * math.pi * 8.8541878128e-12 * hbar * c)), "unit": "C", "symbol": "q_P"},
            "natural_unit_energy_eV": {"value": 1.0, "unit": "eV"},
            "atomic_unit_energy_Hartree": {"value": 27.211386245988, "unit": "eV"},
        }

    def derived_constants(self) -> dict:
        c = self.constants["speed_of_light"]["value"]
        h = self.constants["planck_constant"]["value"]
        hbar = self.constants["reduced_planck"]["value"]
        e = self.constants["electron_charge"]["value"]
        kB = self.constants["boltzmann_constant"]["value"]
        me = self.constants["electron_mass"]["value"]
        NA = self.constants["avogadro_number"]["value"]
        eps0 = self.constants["vacuum_permittivity"]["value"]
        return {
            "hbar_times_c": {"value": float(hbar * c), "unit": "J*m", "description": "Reduced Planck constant times speed of light"},
            "electron_rest_energy": {"value": float(me * c**2), "unit": "J", "description": "Electron rest mass energy"},
            "electron_rest_energy_MeV": {"value": float(me * c**2 / 1.602176634e-13), "unit": "MeV"},
            "compton_wavelength": {"value": float(h / (me * c)), "unit": "m", "description": "Compton wavelength of electron"},
            "classical_electron_radius": {"value": float(e**2 / (4 * math.pi * eps0 * me * c**2)), "unit": "m"},
            "thermal_voltage_300K": {"value": float(kB * 300 / e), "unit": "V", "description": "Thermal voltage at 300K"},
            "bohr_magneton": {"value": float(e * hbar / (2 * me)), "unit": "J/T", "description": "Bohr magneton"},
            "photon_energy_1eV": {"value": 1.602176634e-19, "unit": "J", "description": "Energy of 1 eV photon"},
            "wavelength_1eV": {"value": float(h * c / (1.602176634e-19)), "unit": "m", "description": "Wavelength of 1 eV photon"},
            "ideal_gas_volume_STP": {"value": 22.41396954e-3, "unit": "m^3/mol", "description": "Molar volume at STP"},
        }

    def all_constants(self) -> dict:
        return {k: {"name": k, **v} for k, v in self.constants.items()}

    def constants_summary(self) -> dict:
        categories = {}
        for key, val in self.constants.items():
            cat = val.get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({"name": key, "symbol": val["symbol"], "value": val["value"], "unit": val["unit"]})
        return {"categories": categories, "total_constants": len(self.constants), "method": "physical_constants_summary"}