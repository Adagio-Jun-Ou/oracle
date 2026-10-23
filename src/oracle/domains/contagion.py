import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from oracle.core.types import Domain, Prediction, ConfidenceLevel


@dataclass
class ContagionState:
    financial_systemic_risk: float = 0.3
    banking_contagion: float = 0.2
    sovereign_contagion: float = 0.25
    trade_linkage: float = 0.5
    capital_flow_volatility: float = 0.3
    currency_stress: float = 0.2
    supply_chain_coupling: float = 0.4
    info_propagation_rate: float = 0.5
    timestamp: date = field(default_factory=date.today)


@dataclass
class NetworkNode:
    id: str
    node_type: str
    vulnerability: float
    connectivity: float
    size: float
    initial_shock: float = 0.0


CONTAGION_THRESHOLDS = {
    "lehman_like": {"initial_shock": 0.8, "connectivity": 0.6, "propagation": 0.4},
    "asian_1997": {"initial_shock": 0.7, "connectivity": 0.5, "propagation": 0.35},
    "euro_2010": {"initial_shock": 0.6, "connectivity": 0.7, "propagation": 0.45},
    "covid_2020": {"initial_shock": 0.9, "connectivity": 0.8, "propagation": 0.5},
}


class ContagionModel:
    def __init__(self, n_nodes: int = 20, seed: int = 42):
        self.n_nodes = n_nodes
        self.rng = np.random.default_rng(seed)
        self.sir_params = {"beta": 0.3, "gamma": 0.1}
        self.threshold_params = {"threshold": 0.15, "recovery": 0.05}

    def sir_model(
        self, population: int = 10000, beta: float = 0.3, gamma: float = 0.1,
        days: int = 365, initial_infected: int = 10,
    ) -> dict:
        S = np.zeros(days)
        I = np.zeros(days)
        R = np.zeros(days)
        S[0] = population - initial_infected
        I[0] = initial_infected
        R[0] = 0

        for t in range(1, days):
            new_infected = beta * S[t - 1] * I[t - 1] / population
            new_recovered = gamma * I[t - 1]
            S[t] = S[t - 1] - new_infected
            I[t] = I[t - 1] + new_infected - new_recovered
            R[t] = R[t - 1] + new_recovered

        r0 = beta / gamma
        peak_day = int(np.argmax(I))
        total_infected_pct = R[-1] / population

        return {
            "susceptible": S, "infected": I, "recovered": R,
            "r0": r0, "peak_day": peak_day,
            "peak_infected_pct": float(I[peak_day] / population),
            "total_infected_pct": float(total_infected_pct),
            "days_above_1pct": int(np.sum(I > population * 0.01)),
        }

    def financial_contagion(
        self, state: ContagionState, nodes: Optional[list[NetworkNode]] = None,
        n_simulations: int = 500, time_steps: int = 30,
    ) -> dict:
        if nodes is None:
            nodes = self._generate_default_network()

        n = len(nodes)
        vulnerabilities = np.array([nd.vulnerability for nd in nodes])
        connectivities = np.array([nd.connectivity for nd in nodes])
        sizes = np.array([nd.size for nd in nodes])

        adjacency = self._generate_adjacency(n, connectivities)

        default_counts = np.zeros((n_simulations, time_steps))
        systemic_stress = np.zeros((n_simulations, time_steps))
        cascade_sizes = np.zeros(n_simulations)

        for sim in range(n_simulations):
            distressed = np.array([nd.initial_shock for nd in nodes], dtype=float)
            distressed += self.rng.uniform(0, state.financial_systemic_risk * 0.3, n)

            for t in range(time_steps):
                spillover = adjacency @ (distressed * sizes)
                spillover = spillover / (np.sum(sizes) + 1e-10)
                new_distress = vulnerabilities * spillover * (1 + state.banking_contagion)
                distressed = np.clip(distressed + new_distress + self.rng.normal(0, 0.02, n), 0, 1)

                defaults = distressed > 0.8
                default_counts[sim, t] = np.sum(defaults)
                systemic_stress[sim, t] = np.average(distressed, weights=sizes)
                distressed[defaults] *= 0.5

            cascade_sizes[sim] = np.sum(distressed > 0.8)

        return {
            "mean_systemic_stress": float(np.mean(systemic_stress[:, -1])),
            "p95_systemic_stress": float(np.percentile(systemic_stress[:, -1], 95)),
            "mean_defaults": float(np.mean(default_counts[:, -1])),
            "p95_defaults": float(np.percentile(default_counts[:, -1], 95)),
            "cascade_probability": float(np.mean(cascade_sizes > n * 0.3)),
            "mean_cascade_size": float(np.mean(cascade_sizes)),
            "worst_case_defaults": int(np.max(default_counts[:, -1])),
            "systemic_stress_ts": systemic_stress,
            "defaults_ts": default_counts,
        }

    def cascade_threshold_model(
        self, state: ContagionState, n_agents: int = 100, threshold_dist: str = "uniform",
    ) -> dict:
        if threshold_dist == "uniform":
            thresholds = self.rng.uniform(0.1, 0.9, n_agents)
        elif threshold_dist == "normal":
            thresholds = np.clip(self.rng.normal(0.5, 0.15, n_agents), 0.1, 0.9)
        else:
            thresholds = np.clip(self.rng.exponential(0.3, n_agents), 0.1, 0.9)

        adjacency = self.rng.random((n_agents, n_agents)) < state.trade_linkage
        np.fill_diagonal(adjacency, False)
        degrees = adjacency.sum(axis=1).astype(float)
        degrees[degrees == 0] = 1

        n_seeds = max(1, int(n_agents * state.financial_systemic_risk))
        seeds = self.rng.choice(n_agents, n_seeds, replace=False)
        active = np.zeros(n_agents, dtype=bool)
        active[seeds] = True

        steps = [active.copy()]
        max_steps = 50
        for _ in range(max_steps):
            neighbor_active = adjacency.astype(float) @ active.astype(float)
            fraction_active = neighbor_active / degrees
            newly_active = (fraction_active >= thresholds) & (~active)
            if not np.any(newly_active):
                break
            active |= newly_active
            steps.append(active.copy())

        cascade_size = np.sum(active)
        cascade_pct = cascade_size / n_agents

        return {
            "cascade_size": int(cascade_size),
            "cascade_pct": float(cascade_pct),
            "n_steps": len(steps),
            "seeds": int(n_seeds),
            "full_cascade": bool(cascade_pct > 0.9),
            "partial_cascade": bool(0.3 < cascade_pct <= 0.9),
            "contained": bool(cascade_pct <= 0.3),
            "final_activation": steps[-1].tolist() if len(steps[-1]) < 200 else None,
        }

    def predict(
        self, state: ContagionState, target_date: date,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    ) -> list[Prediction]:
        years_ahead = max(0.08, (target_date - state.timestamp).days / 365.25)
        z = {ConfidenceLevel.LOW: 1.0, ConfidenceLevel.MEDIUM: 1.645,
             ConfidenceLevel.HIGH: 1.96, ConfidenceLevel.VERY_HIGH: 2.576}[confidence]

        params = [
            ("systemic_risk", state.financial_systemic_risk, 0.1 * years_ahead, Domain.FINANCE),
            ("banking_contagion", state.banking_contagion, 0.08 * years_ahead, Domain.FINANCE),
            ("sovereign_contagion", state.sovereign_contagion, 0.08 * years_ahead, Domain.ECONOMICS),
            ("trade_disruption", state.supply_chain_coupling, 0.1 * years_ahead, Domain.ECONOMICS),
            ("currency_stress", state.currency_stress, 0.12 * years_ahead, Domain.FINANCE),
            ("info_cascade_probability", state.info_propagation_rate, 0.05 * years_ahead, Domain.SOCIAL),
        ]
        preds = []
        for name, val, std, domain in params:
            ci = (max(0, val - z * std), min(1, val + z * std))
            preds.append(Prediction(
                domain=domain, variable=f"contagion_{name}", value=round(float(val), 4),
                confidence_interval=(round(float(ci[0]), 4), round(float(ci[1]), 4)),
                confidence_level=confidence, timestamp=datetime.now(),
                target_date=target_date, model_name="contagion_model",
                weights={"sir": 0.3, "threshold": 0.35, "network": 0.35},
            ))
        return preds

    def _generate_default_network(self) -> list[NetworkNode]:
        node_specs = [
            ("US_Banking", "banking", 0.3, 0.7, 0.25),
            ("EU_Banking", "banking", 0.35, 0.6, 0.20),
            ("UK_Banking", "banking", 0.25, 0.5, 0.08),
            ("Japan_Banking", "banking", 0.30, 0.4, 0.12),
            ("China_Banking", "banking", 0.45, 0.5, 0.20),
            ("US_Sovereign", "sovereign", 0.15, 0.6, 0.25),
            ("EU_Sovereign", "sovereign", 0.30, 0.5, 0.15),
            ("EM_Sovereign", "sovereign", 0.50, 0.4, 0.10),
            ("Hedge_Funds", "shadow", 0.60, 0.5, 0.05),
            ("Insurers", "insurance", 0.25, 0.4, 0.10),
            ("Pension_Funds", "pension", 0.20, 0.3, 0.15),
            ("REITs", "realestate", 0.35, 0.3, 0.05),
            ("Tech_Corps", "corporate", 0.25, 0.5, 0.20),
            ("Energy_Corps", "corporate", 0.40, 0.4, 0.10),
            ("Crypto", "crypto", 0.70, 0.3, 0.02),
        ]
        nodes = []
        for nid, ntype, vuln, conn, sz in node_specs:
            nodes.append(NetworkNode(
                id=nid, node_type=ntype, vulnerability=vuln, connectivity=conn, size=sz,
                initial_shock=self.rng.uniform(0, 0.1),
            ))
        while len(nodes) < self.n_nodes:
            nodes.append(NetworkNode(
                id=f"Node_{len(nodes)}", node_type="generic",
                vulnerability=self.rng.uniform(0.2, 0.5),
                connectivity=self.rng.uniform(0.2, 0.5),
                size=self.rng.uniform(0.01, 0.05),
                initial_shock=0.0,
            ))
        return nodes

    def _generate_adjacency(self, n: int, connectivities: np.ndarray) -> np.ndarray:
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                connectivity = (connectivities[i] + connectivities[j]) / 2
                if self.rng.random() < connectivity * 0.5:
                    weight = self.rng.uniform(0.01, 0.1)
                    adj[i, j] = weight
                    adj[j, i] = weight
        return adj