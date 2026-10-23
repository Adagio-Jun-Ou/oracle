"""Typed dataclasses for Oracle domain states.

Replaces raw dict states with typed, validated dataclasses that are
backward-compatible with dict-like access patterns (bracket notation, .get()).
"""
from dataclasses import dataclass, field, fields
from typing import Optional, Any, Dict


class DictMixin:
    """Mixin to make dataclasses behave like dicts for backward compatibility."""

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return [f.name for f in fields(self)]

    def values(self):
        return [getattr(self, f.name) for f in fields(self)]

    def items(self):
        return [(f.name, getattr(self, f.name)) for f in fields(self)]

    def __iter__(self):
        return iter(f.name for f in fields(self))


@dataclass
class EpidemiologyState(DictMixin):
    population: float = 330_000_000
    initial_infected: float = 1000
    beta: float = 0.3
    gamma: float = 0.1
    mu: float = 0.005
    vaccination_rate: float = 0.5
    hospital_capacity: float = 0.8
    r0: float = 3.0
    total_infected: float = 0.0
    recovered: float = 0.0
    deceased: float = 0.0
    supply_chain_resilience: float = 0.7


@dataclass
class TechnologyState(DictMixin):
    current_adoption: float = 0.15
    innovation_speed: float = 0.1
    disruption_probability: float = 0.2
    quantum_readiness: float = 0.1
    ai_capability: float = 0.3
    s_curve_steepness: float = 0.5
    adoption_saturation: float = 0.9
    power_shift_index: float = 0.3


@dataclass
class SocialState(DictMixin):
    gini: float = 0.4
    unemployment: float = 0.05
    political_instability: float = 0.3
    unrest_probability: float = 0.3
    polarization_index: float = 0.4
    institutional_trust: float = 0.5
    migration_pressure: float = 0.3
    cascade_risk: float = 0.1
    wealth_concentration: float = 0.6
    social_mobility: float = 0.4
    polarization_velocity: float = 0.05
    unrest_intensity: float = 0.3
    protest_likelihood: float = 0.2


@dataclass
class EnergyState(DictMixin):
    oil_price: float = 75.0
    renewable_share: float = 20.0
    demand_growth: float = 0.02
    security_index: float = 0.7
    carbon_budget_used: float = 0.6
    price_volatility: float = 0.3
    nuclear_share: float = 10.0
    storage_capacity: float = 0.15
    grid_resilience: float = 0.7


@dataclass
class CommoditiesState(DictMixin):
    gold_price: float = 1800.0
    copper_price: float = 4.5
    lithium_price: float = 50000.0
    supply_chain_stress: float = 0.4
    geopolitical_risk: float = 0.5
    demand_pressure: float = 0.3
    inventory_level: float = 0.5
    concentration_risk: float = 0.4