"""Tests for physical constants module."""
import pytest
from oracle.domains.physical_constants import PhysicalConstants, FUNDAMENTAL_CONSTANTS, UNIT_CONVERSIONS, DIMENSIONAL_ANALYSIS


class TestPhysicalConstants:
    def setup_method(self):
        self.pc = PhysicalConstants()

    def test_get_by_name(self):
        result = self.pc.get("speed_of_light")
        assert result["value"] == 299792458.0
        assert result["symbol"] == "c"
        assert result["unit"] == "m/s"

    def test_get_by_symbol(self):
        result = self.pc.get("G")
        assert result["value"] == 6.67430e-11

    def test_value_method(self):
        val = self.pc.value("planck_constant")
        assert val == 6.62607015e-34

    def test_search(self):
        results = self.pc.search("planck")
        assert len(results) >= 3

    def test_by_category(self):
        results = self.pc.by_category("quantum")
        assert len(results) >= 3
        names = [r["name"] for r in results]
        assert "planck_constant" in names
        assert "reduced_planck" in names

    def test_convert_length(self):
        km = self.pc.convert(1000, "m", "km")
        assert abs(km - 1.0) < 0.01

    def test_convert_mass(self):
        g = self.pc.convert(1, "kg", "g")
        assert abs(g - 1000.0) < 0.1

    def test_convert_time(self):
        hr = self.pc.convert(3600, "s", "hr")
        assert abs(hr - 1.0) < 0.01

    def test_convert_energy(self):
        eV = self.pc.convert(1.602176634e-19, "J", "eV")
        assert abs(eV - 1.0) < 0.01

    def test_convert_temperature(self):
        c = self.pc.convert(0, "k", "c")
        assert abs(c - (-273.15)) < 0.01

    def test_convert_same_unit(self):
        val = self.pc.convert(5.0, "m", "m")
        assert val == 5.0

    def test_convert_invalid(self):
        with pytest.raises(ValueError):
            self.pc.convert(1.0, "m", "bogus")

    def test_dimensional_analysis(self):
        result = self.pc.dimensional_analysis("force")
        assert result["SI_unit"] == "N"
        assert result["dimensions"]["length"] == 1
        assert result["dimensions"]["mass"] == 1
        assert result["dimensions"]["time"] == -2

    def test_dimensional_analysis_by_symbol(self):
        result = self.pc.dimensional_analysis("E")
        assert result["SI_unit"] == "J"

    def test_natural_units(self):
        result = self.pc.natural_units()
        assert "planck_mass" in result
        assert "planck_length" in result
        assert "planck_time" in result
        assert result["planck_length"]["value"] > 0
        assert result["planck_time"]["value"] > 0

    def test_derived_constants(self):
        result = self.pc.derived_constants()
        assert "compton_wavelength" in result
        assert "bohr_magneton" in result
        assert "thermal_voltage_300K" in result
        assert result["compton_wavelength"]["value"] > 0

    def test_all_constants(self):
        result = self.pc.all_constants()
        assert len(result) >= 25

    def test_constants_summary(self):
        result = self.pc.constants_summary()
        assert "categories" in result
        assert result["total_constants"] >= 25
        assert "quantum" in result["categories"] or "universal" in result["categories"]

    def test_get_not_found(self):
        with pytest.raises(KeyError):
            self.pc.get("nonexistent_constant")

    def test_fundamental_constants_direct(self):
        assert len(FUNDAMENTAL_CONSTANTS) >= 25
        assert "speed_of_light" in FUNDAMENTAL_CONSTANTS
        assert "gravitational_constant" in FUNDAMENTAL_CONSTANTS

    def test_unit_conversions_direct(self):
        assert "length" in UNIT_CONVERSIONS
        assert "mass" in UNIT_CONVERSIONS
        assert "energy" in UNIT_CONVERSIONS

    def test_dimensional_analysis_direct(self):
        assert "force" in DIMENSIONAL_ANALYSIS
        assert "energy" in DIMENSIONAL_ANALYSIS