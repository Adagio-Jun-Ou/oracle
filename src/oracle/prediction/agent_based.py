"""Agent-Based Model (ABM) for emergent behavior simulation.

Implements:
- Agent class with heterogeneous behaviors and interactions
- Network-based interaction topology
- Simulation engine with emergent pattern detection
- Market, epidemic, and opinion dynamics presets
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
import math


@dataclass
class Agent:
    id: int
    state: float
    wealth: float = 100.0
    opinion: float = 0.5
    risk_tolerance: float = 0.5
    connectivity: int = 0
    group: int = 0


@dataclass
class SimulationResult:
    time_series: Dict[str, List[float]]
    agent_states: List[float]
    agent_groups: List[int]
    statistics: Dict[str, float]
    n_steps: int
    n_agents: int
    emergent_patterns: List[str]
    method: str = "abm"


class AgentBasedModel:
    """Configurable ABM with network interactions and emergent pattern detection."""

    def __init__(self, n_agents: int = 100, network_type: str = "small_world",
                 interaction_radius: float = 0.3, seed: int = 42):
        self.n_agents = n_agents
        self.network_type = network_type
        self.interaction_radius = interaction_radius
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.agents: List[Agent] = []
        self.adjacency: Optional[np.ndarray] = None
        self._fitted = False

    def _build_network(self) -> np.ndarray:
        n = self.n_agents
        adj = np.zeros((n, n))
        if self.network_type == "random":
            p = self.interaction_radius
            for i in range(n):
                for j in range(i + 1, n):
                    if self.rng.random() < p:
                        adj[i, j] = 1
                        adj[j, i] = 1
        elif self.network_type == "scale_free":
            m0 = max(3, int(n * 0.05))
            degrees = np.zeros(n)
            for i in range(m0):
                for j in range(i + 1, m0):
                    adj[i, j] = 1
                    adj[j, i] = 1
                    degrees[i] += 1
                    degrees[j] += 1
            for new_node in range(m0, n):
                probs = degrees[:new_node] / max(degrees[:new_node].sum(), 1)
                m_connect = min(m0, new_node)
                targets = self.rng.choice(new_node, size=m_connect, replace=False, p=probs)
                for t in targets:
                    adj[new_node, t] = 1
                    adj[t, new_node] = 1
                    degrees[new_node] += 1
                    degrees[t] += 1
        elif self.network_type == "small_world":
            k = max(2, int(self.interaction_radius * n))
            k = min(k, n - 1)
            for i in range(n):
                for j in range(1, k // 2 + 1):
                    adj[i, (i + j) % n] = 1
                    adj[(i + j) % n, i] = 1
            rewire_prob = 0.1
            for i in range(n):
                for j in range(n):
                    if adj[i, j] == 1:
                        if self.rng.random() < rewire_prob:
                            adj[i, j] = 0
                            adj[j, i] = 0
                            new_j = self.rng.randint(0, n)
                            while new_j == i or adj[i, new_j] == 1:
                                new_j = self.rng.randint(0, n)
                            adj[i, new_j] = 1
                            adj[new_j, i] = 1
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    if self.rng.random() < self.interaction_radius:
                        adj[i, j] = 1
                        adj[j, i] = 1
        for i in range(n):
            if adj[i].sum() == 0:
                j = self.rng.randint(0, n)
                while j == i:
                    j = self.rng.randint(0, n)
                adj[i, j] = 1
                adj[j, i] = 1
        return adj

    def _init_agents(self, scenario: str = "market"):
        self.agents = []
        for i in range(self.n_agents):
            if scenario == "market":
                state = self.rng.normal(0, 1)
                wealth = max(10, self.rng.lognormal(4, 1.5))
                opinion = self.rng.uniform(0, 1)
                risk = self.rng.beta(2, 5)
            elif scenario == "epidemic":
                state = 0.0 if i > 5 else 1.0
                wealth = 100.0
                opinion = self.rng.uniform(0, 1)
                risk = self.rng.uniform(0.3, 0.7)
            elif scenario == "opinion":
                state = self.rng.beta(2, 2)
                wealth = 100.0
                opinion = state
                risk = self.rng.uniform(0.3, 0.7)
            else:
                state = self.rng.uniform(-1, 1)
                wealth = self.rng.lognormal(4, 1)
                opinion = self.rng.uniform(0, 1)
                risk = self.rng.uniform(0.1, 0.9)
            agent = Agent(id=i, state=state, wealth=wealth, opinion=opinion, risk_tolerance=risk)
            self.agents.append(agent)

    def simulate_market(self, n_steps: int = 100) -> SimulationResult:
        self._init_agents("market")
        self.adjacency = self._build_network()
        price = 100.0
        prices = [price]
        volumes = [0.0]
        avg_wealth = [float(np.mean([a.wealth for a in self.agents]))]
        gini_history = [self._gini([a.wealth for a in self.agents])]
        opinions_avg = [float(np.mean([a.opinion for a in self.agents]))]
        states_std = [float(np.std([a.state for a in self.agents]))]
        for step in range(n_steps):
            volume = 0.0
            new_states = []
            for i, agent in enumerate(self.agents):
                neighbors = np.where(self.adjacency[i] > 0)[0]
                if len(neighbors) > 0:
                    neighbor_states = [self.agents[j].state for j in neighbors]
                    avg_neighbor = float(np.mean(neighbor_states))
                    social_influence = 0.3 * (avg_neighbor - agent.state)
                    news_shock = self.rng.normal(0, 0.1)
                    agent.state = 0.8 * agent.state + social_influence + news_shock * agent.risk_tolerance
                else:
                    agent.state = 0.9 * agent.state + self.rng.normal(0, 0.05)
                new_states.append(agent.state)
            demand = sum(1 for s in new_states if s > 0)
            supply = sum(1 for s in new_states if s < 0)
            volume = demand + supply
            price_change = 0.01 * (demand - supply) / max(self.n_agents, 1) + self.rng.normal(0, 0.005)
            price *= (1 + price_change)
            for i, agent in enumerate(self.agents):
                agent.state = new_states[i]
                if new_states[i] > 0:
                    agent.wealth *= (1 + abs(price_change) * 0.5)
                elif new_states[i] < 0:
                    agent.wealth *= (1 - abs(price_change) * 0.3)
                agent.wealth = max(1.0, agent.wealth)
                if self.rng.random() < 0.02:
                    neighbors = np.where(self.adjacency[i] > 0)[0]
                    if len(neighbors) > 0:
                        j = self.rng.choice(neighbors)
                        agent.opinion = 0.7 * agent.opinion + 0.3 * self.agents[j].opinion
            prices.append(price)
            volumes.append(float(volume))
            avg_wealth.append(float(np.mean([a.wealth for a in self.agents])))
            gini_history.append(self._gini([a.wealth for a in self.agents]))
            opinions_avg.append(float(np.mean([a.opinion for a in self.agents])))
            states_std.append(float(np.std([a.state for a in self.agents])))
        patterns = self._detect_patterns(prices, gini_history, opinions_avg, states_std)
        return SimulationResult(
            time_series={"price": prices, "volume": volumes, "avg_wealth": avg_wealth,
                        "gini": gini_history, "opinion_avg": opinions_avg, "states_std": states_std},
            agent_states=[a.state for a in self.agents],
            agent_groups=[a.group for a in self.agents],
            statistics={"final_price": prices[-1], "price_volatility": float(np.std(np.diff(prices) / (np.array(prices[:-1]) + 1e-10))),
                        "final_gini": gini_history[-1], "final_avg_wealth": avg_wealth[-1],
                        "total_volume": sum(volumes), "max_drawdown": self._max_drawdown(prices)},
            n_steps=n_steps,
            n_agents=self.n_agents,
            emergent_patterns=patterns,
            method="abm_market",
        )

    def simulate_epidemic(self, n_steps: int = 100, beta: float = 0.3,
                          gamma: float = 0.1) -> SimulationResult:
        self._init_agents("epidemic")
        self.adjacency = self._build_network()
        states = ['S'] * self.n_agents
        for i in range(min(5, self.n_agents)):
            states[i] = 'I'
            self.agents[i].state = 1.0
        s_count = [states.count('S')]
        i_count = [states.count('I')]
        r_count = [states.count('R')]
        r0_estimates = [beta / max(gamma, 0.01)]
        for step in range(n_steps):
            new_states = states.copy()
            for i, agent in enumerate(self.agents):
                if states[i] == 'S':
                    neighbors = np.where(self.adjacency[i] > 0)[0]
                    n_infected_neighbors = sum(1 for j in neighbors if states[j] == 'I')
                    p_infection = 1 - (1 - beta) ** n_infected_neighbors * (1 - beta * 0.01)
                    if self.rng.random() < p_infection:
                        new_states[i] = 'I'
                        agent.state = 1.0
                elif states[i] == 'I':
                    if self.rng.random() < gamma:
                        new_states[i] = 'R'
                        agent.state = 2.0
            states = new_states
            s_count.append(states.count('S'))
            i_count.append(states.count('I'))
            r_count.append(states.count('R'))
            effective_contacts = float(np.mean([sum(1 for j in np.where(self.adjacency[i] > 0)[0] if states[j] == 'I')
                                               for i in range(self.n_agents) if states[i] == 'S'] or [0]))
            if effective_contacts > 0:
                r0_estimates.append(beta * effective_contacts / max(gamma, 0.01))
            else:
                r0_estimates.append(0.0)
        peak_day = int(np.argmax(i_count))
        total_infected = self.n_agents - s_count[-1]
        infection_rate = total_infected / max(self.n_agents, 1)
        patterns = []
        if peak_day < n_steps * 0.3:
            patterns.append("rapid_outbreak")
        if i_count[-1] > s_count[-1]:
            patterns.append("endemic_persistence")
        if total_infected > 0.7 * self.n_agents:
            patterns.append("mass_infection")
        if r0_estimates[0] < 1.0 and i_count[-1] < 5:
            patterns.append("contained_outbreak")
        return SimulationResult(
            time_series={"susceptible": s_count, "infected": i_count, "recovered": r_count, "r0": r0_estimates},
            agent_states=[1.0 if s == 'I' else (2.0 if s == 'R' else 0.0) for s in states],
            agent_groups=[0 if s == 'S' else (1 if s == 'I' else 2) for s in states],
            statistics={"peak_infected": max(i_count), "peak_day": peak_day,
                        "total_infected": total_infected, "infection_rate": infection_rate,
                        "final_susceptible": s_count[-1], "r0_initial": r0_estimates[0],
                        "r0_final": r0_estimates[-1]},
            n_steps=n_steps,
            n_agents=self.n_agents,
            emergent_patterns=patterns,
            method="abm_epidemic",
        )

    def simulate_opinion(self, n_steps: int = 100, conformity: float = 0.3,
                         polarization: float = 0.1) -> SimulationResult:
        self._init_agents("opinion")
        self.adjacency = self._build_network()
        opinions = [a.opinion for a in self.agents]
        opinion_history = [opinions[:]]
        polarization_history = [self._polarization(opinions)]
        clustering_history = [self._opinion_clustering(opinions)]
        consensus_history = [1.0 - float(np.std(opinions))]
        for step in range(n_steps):
            new_opinions = opinions.copy()
            for i in range(self.n_agents):
                neighbors = np.where(self.adjacency[i] > 0)[0]
                if len(neighbors) > 0:
                    neighbor_ops = [opinions[j] for j in neighbors]
                    distance_weights = [1.0 / (1.0 + abs(opinions[j] - opinions[i]) / max(polarization, 0.01))
                                       for j in neighbors]
                    total_weight = sum(distance_weights)
                    if total_weight > 0:
                        weighted_avg = sum(w * o for w, o in zip(distance_weights, neighbor_ops)) / total_weight
                        new_opinions[i] = opinions[i] + conformity * (weighted_avg - opinions[i])
                new_opinions[i] += self.rng.normal(0, 0.01)
                new_opinions[i] = max(0, min(1, new_opinions[i]))
            opinions = new_opinions
            opinion_history.append(opinions[:])
            polarization_history.append(self._polarization(opinions))
            clustering_history.append(self._opinion_clustering(opinions))
            consensus_history.append(1.0 - float(np.std(opinions)))
        patterns = []
        final_pol = polarization_history[-1]
        initial_pol = polarization_history[0]
        if final_pol > initial_pol * 1.5:
            patterns.append("polarization_escalation")
        elif final_pol < initial_pol * 0.5:
            patterns.append("consensus_emergence")
        if clustering_history[-1] > clustering_history[0] * 1.5:
            patterns.append("echo_chamber_formation")
        if consensus_history[-1] > 0.9:
            patterns.append("strong_consensus")
        return SimulationResult(
            time_series={"polarization": polarization_history, "clustering": clustering_history,
                        "consensus": consensus_history},
            agent_states=opinions,
            agent_groups=[0 if o < 0.3 else (1 if o > 0.7 else 2) for o in opinions],
            statistics={"final_polarization": polarization_history[-1], "final_consensus": consensus_history[-1],
                        "polarization_change": polarization_history[-1] - polarization_history[0],
                        "n_clusters": clustering_history[-1],
                        "opinion_mean": float(np.mean(opinions)), "opinion_std": float(np.std(opinions))},
            n_steps=n_steps,
            n_agents=self.n_agents,
            emergent_patterns=patterns,
            method="abm_opinion",
        )

    def _gini(self, values: List[float]) -> float:
        values = sorted(v for v in values if v > 0)
        n = len(values)
        if n == 0:
            return 0.0
        total = sum(values)
        if total == 0:
            return 0.0
        gini_sum = sum((2 * i - n + 1) * v for i, v in enumerate(values))
        return gini_sum / (n * total)

    def _max_drawdown(self, prices: List[float]) -> float:
        peak = prices[0]
        max_dd = 0.0
        for p in prices:
            if p > peak:
                peak = p
            dd = (peak - p) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _polarization(self, opinions: List[float]) -> float:
        ops = np.array(opinions)
        if len(ops) < 2:
            return 0.0
        left = ops[ops < 0.5]
        right = ops[ops >= 0.5]
        if len(left) == 0 or len(right) == 0:
            return float(np.var(ops))
        return float(np.var(left) + np.var(right) + (np.mean(left) - np.mean(right)) ** 2)

    def _opinion_clustering(self, opinions: List[float]) -> float:
        ops = np.array(opinions)
        n = len(ops)
        if n < 2:
            return 1.0
        sorted_ops = np.sort(ops)
        gaps = np.diff(sorted_ops)
        if len(gaps) == 0:
            return 1.0
        threshold = np.mean(gaps) + np.std(gaps)
        n_clusters = 1
        for g in gaps:
            if g > threshold:
                n_clusters += 1
        return n_clusters

    def _detect_patterns(self, prices, gini, opinions, states_std) -> List[str]:
        patterns = []
        if len(prices) > 10:
            returns = np.diff(prices) / (np.array(prices[:-1]) + 1e-10)
            if np.std(returns) > 0.02:
                patterns.append("high_volatility")
            if abs(returns[-1]) > 3 * np.std(returns):
                patterns.append("fat_tail_event")
        if len(gini) > 1 and gini[-1] > gini[0] * 1.3:
            patterns.append("wealth_inequality_increase")
        if len(opinions) > 1 and abs(opinions[-1] - opinions[0]) > 0.2:
            patterns.append("opinion_shift")
        if len(states_std) > 5:
            if max(states_std[-5:]) > 2 * np.mean(states_std[:5]):
                patterns.append("synchronization_burst")
        return patterns