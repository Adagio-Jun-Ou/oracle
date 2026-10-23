"""Reinforcement Learning decision optimizer module.

Implements Q-learning, policy gradient, and multi-armed bandit
for sequential decision optimization with state-aware exploration.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math


@dataclass
class QLearningResult:
    q_table: Dict[str, List[float]]
    optimal_policy: List[int]
    optimal_values: List[float]
    convergence_history: List[float]
    n_episodes: int
    total_reward: float
    exploration_rate: float
    method: str = "q_learning"


@dataclass
class BanditResult:
    arm_values: List[float]
    arm_counts: List[int]
    optimal_arm: int
    total_reward: float
    regret: float
    cumulative_rewards: List[float]
    n_pulls: int
    method: str = "multi_armed_bandit"


@dataclass
class PolicyGradientResult:
    policy_weights: List[List[float]]
    episode_rewards: List[float]
    optimal_actions: List[int]
    policy_entropy: List[float]
    n_episodes: int
    best_reward: float
    method: str = "policy_gradient"


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration.

    Supports discrete state-action spaces with configurable
    learning rate, discount factor, and exploration schedule.
    """

    def __init__(self, n_states: int = 10, n_actions: int = 4,
                 lr: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, seed: int = 42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        rng = np.random.RandomState(seed)
        self.q_table = rng.uniform(-0.1, 0.1, (n_states, n_actions))
        self._fitted = False

    def _epsilon(self, episode: int) -> float:
        eps = self.epsilon_start * (self.epsilon_decay ** episode)
        return max(eps, self.epsilon_end)

    def fit(self, transitions: Optional[List[Tuple[int, int, float, int]]] = None,
            n_episodes: int = 1000,
            env_func: Optional[callable] = None) -> 'QLearningAgent':
        rng = np.random.RandomState(self.seed)
        convergence = []
        if transitions is not None:
            total_reward = 0.0
            for _ in range(n_episodes):
                ep_reward = 0.0
                for s, a, r, s_next in transitions:
                    best_next = np.max(self.q_table[s_next])
                    td_target = r + self.gamma * best_next
                    self.q_table[s, a] += self.lr * (td_target - self.q_table[s, a])
                    ep_reward += r
                total_reward += ep_reward
                convergence.append(float(np.max(self.q_table)))
            self._fitted = True
            return self
        for episode in range(n_episodes):
            eps = self._epsilon(episode)
            state = rng.randint(0, self.n_states)
            ep_reward = 0.0
            for step in range(100):
                if rng.random() < eps:
                    action = rng.randint(0, self.n_actions)
                else:
                    action = int(np.argmax(self.q_table[state]))
                if env_func:
                    reward, next_state = env_func(state, action)
                else:
                    reward = rng.normal(0, 1) + self.q_table[state, action] * 0.1
                    next_state = rng.randint(0, self.n_states)
                best_next = np.max(self.q_table[next_state])
                td_target = reward + self.gamma * best_next
                self.q_table[state, action] += self.lr * (td_target - self.q_table[state, action])
                state = next_state
                ep_reward += reward
            convergence.append(float(ep_reward))
        self._fitted = True
        return self

    def get_policy(self) -> QLearningResult:
        optimal_policy = [int(np.argmax(self.q_table[s])) for s in range(self.n_states)]
        optimal_values = [float(np.max(self.q_table[s])) for s in range(self.n_states)]
        convergence = [0.0]  # simplified
        return QLearningResult(
            q_table={f"state_{s}": self.q_table[s].tolist() for s in range(self.n_states)},
            optimal_policy=optimal_policy,
            optimal_values=optimal_values,
            convergence_history=convergence,
            n_episodes=0,
            total_reward=float(np.sum(self.q_table)),
            exploration_rate=self.epsilon_end,
        )


class MultiArmedBandit:
    """Multi-armed bandit with Upper Confidence Bound (UCB1) and Thompson Sampling.

    Supports:
    - UCB1 (Auer et al.)
    - Thompson Sampling (Beta-Bernoulli)
    - Epsilon-greedy
    """

    def __init__(self, n_arms: int = 5, method: str = "ucb1",
                 epsilon: float = 0.1, seed: int = 42):
        self.n_arms = n_arms
        self.method = method
        self.epsilon = epsilon
        self.seed = seed
        rng = np.random.RandomState(seed)
        self.arm_values = rng.uniform(0, 1, n_arms)
        self.arm_counts = np.zeros(n_arms, dtype=int)
        self.arm_rewards = np.zeros(n_arms)
        self.alpha_params = np.ones(n_arms)
        self.beta_params = np.ones(n_arms)

    def fit(self, n_pulls: int = 500,
            true_values: Optional[List[float]] = None) -> 'MultiArmedBandit':
        rng = np.random.RandomState(self.seed)
        if true_values is not None:
            self.arm_values = np.array(true_values)
            self.n_arms = len(true_values)
        total_reward = 0.0
        cumulative_rewards = []
        optimal_arm = int(np.argmax(self.arm_values))
        regret = 0.0
        for t in range(1, n_pulls + 1):
            if self.method == "ucb1":
                ucb_values = np.zeros(self.n_arms)
                for a in range(self.n_arms):
                    if self.arm_counts[a] == 0:
                        ucb_values[a] = float('inf')
                    else:
                        avg_reward = self.arm_rewards[a] / max(self.arm_counts[a], 1)
                        ucb_values[a] = avg_reward + np.sqrt(2 * np.log(t) / max(self.arm_counts[a], 1))
                arm = int(np.argmax(ucb_values))
            elif self.method == "thompson":
                samples = [rng.beta(self.alpha_params[a], self.beta_params[a])
                          for a in range(self.n_arms)]
                arm = int(np.argmax(samples))
            elif self.method == "epsilon_greedy":
                if rng.random() < self.epsilon:
                    arm = rng.randint(0, self.n_arms)
                else:
                    avg_rewards = np.zeros(self.n_arms)
                    for a in range(self.n_arms):
                        if self.arm_counts[a] > 0:
                            avg_rewards[a] = self.arm_rewards[a] / self.arm_counts[a]
                    arm = int(np.argmax(avg_rewards))
            else:
                arm = rng.randint(0, self.n_arms)
            reward = float(rng.random() < self.arm_values[arm])
            self.arm_counts[arm] += 1
            self.arm_rewards[arm] += reward
            self.alpha_params[arm] += reward
            self.beta_params[arm] += (1 - reward)
            total_reward += reward
            regret += self.arm_values[optimal_arm] - self.arm_values[arm]
            cumulative_rewards.append(total_reward)
        return self

    def get_result(self) -> BanditResult:
        optimal_arm = int(np.argmax(self.arm_values))
        estimated_values = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            if self.arm_counts[a] > 0:
                estimated_values[a] = self.arm_rewards[a] / self.arm_counts[a]
        return BanditResult(
            arm_values=estimated_values.tolist(),
            arm_counts=self.arm_counts.tolist(),
            optimal_arm=optimal_arm,
            total_reward=float(np.sum(self.arm_rewards)),
            regret=float(np.sum(self.arm_values) * np.sum(self.arm_counts > 0) - np.sum(self.arm_rewards)),
            cumulative_rewards=[float(np.sum(self.arm_rewards[:i+1])) if i < len(self.arm_rewards) else 0
                              for i in range(min(100, self.n_arms))],
            n_pulls=int(np.sum(self.arm_counts)),
            method=f"bandit_{self.method}",
        )


class PolicyGradientAgent:
    """REINFORCE policy gradient with softmax action selection.

    Implements episodic policy gradient with baseline subtraction
    and entropy regularization for exploration control.
    """

    def __init__(self, n_states: int = 10, n_actions: int = 4,
                 lr: float = 0.01, gamma: float = 0.99,
                 entropy_coef: float = 0.01, seed: int = 42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.seed = seed
        rng = np.random.RandomState(seed)
        self.policy_weights = rng.randn(n_states, n_actions) * 0.01
        self._fitted = False

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        return exp_logits / (np.sum(exp_logits) + 1e-10)

    def fit(self, n_episodes: int = 200, max_steps: int = 50,
            env_func: Optional[callable] = None) -> 'PolicyGradientAgent':
        rng = np.random.RandomState(self.seed)
        all_rewards = []
        all_entropies = []
        for episode in range(n_episodes):
            states = []
            actions = []
            rewards = []
            state = rng.randint(0, self.n_states)
            for step in range(max_steps):
                probs = self._softmax(self.policy_weights[state])
                action = rng.choice(self.n_actions, p=probs)
                if env_func:
                    reward, next_state = env_func(state, action)
                else:
                    reward = rng.normal(self.policy_weights[state, action], 0.5)
                    next_state = rng.randint(0, self.n_states)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)
            for t in range(len(states)):
                s = states[t]
                a = actions[t]
                probs = self._softmax(self.policy_weights[s])
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                grad = np.zeros(self.n_actions)
                grad[:] = -probs * returns[t]
                grad[a] = (1 - probs[a]) * returns[t]
                entropy_grad = -probs * (1 - probs)
                self.policy_weights[s] += self.lr * (grad + self.entropy_coef * entropy_grad)
            all_rewards.append(sum(rewards))
            avg_entropy = np.mean([-np.sum(self._softmax(self.policy_weights[s]) * np.log(self._softmax(self.policy_weights[s]) + 1e-10))
                                   for s in range(min(len(states), self.n_states))])
            all_entropies.append(float(avg_entropy))
        self._fitted = True
        self._episode_rewards = all_rewards
        return self

    def get_result(self) -> PolicyGradientResult:
        optimal_actions = [int(np.argmax(self._softmax(self.policy_weights[s])))
                          for s in range(self.n_states)]
        return PolicyGradientResult(
            policy_weights=self.policy_weights.tolist(),
            episode_rewards=self._episode_rewards if hasattr(self, '_episode_rewards') else [],
            optimal_actions=optimal_actions,
            policy_entropy=[0.0],
            n_episodes=len(self._episode_rewards) if hasattr(self, '_episode_rewards') else 0,
            best_reward=max(self._episode_rewards) if hasattr(self, '_episode_rewards') and self._episode_rewards else 0.0,
        )


class RLDecisionOptimizer:
    """Unified RL decision optimizer combining Q-learning, bandits, and policy gradient."""

    def __init__(self, method: str = "q_learning", **kwargs):
        self.method = method
        self.kwargs = kwargs
        if method == "q_learning":
            self.agent = QLearningAgent(**kwargs)
        elif method == "bandit_ucb":
            self.agent = MultiArmedBandit(method="ucb1", **{k: v for k, v in kwargs.items() if k != 'method'})
        elif method == "bandit_thompson":
            self.agent = MultiArmedBandit(method="thompson", **{k: v for k, v in kwargs.items() if k != 'method'})
        elif method == "bandit_epsilon":
            self.agent = MultiArmedBandit(method="epsilon_greedy", **{k: v for k, v in kwargs.items() if k != 'method'})
        elif method == "policy_gradient":
            self.agent = PolicyGradientAgent(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def optimize(self, n_iterations: int = 500, **kwargs) -> dict:
        if isinstance(self.agent, QLearningAgent):
            self.agent.fit(n_episodes=n_iterations, **kwargs)
            result = self.agent.get_policy()
            return {
                "q_table": result.q_table,
                "optimal_policy": result.optimal_policy,
                "optimal_values": result.optimal_values,
                "total_reward": result.total_reward,
                "method": "q_learning_optimization",
            }
        elif isinstance(self.agent, MultiArmedBandit):
            self.agent.fit(n_pulls=n_iterations, **kwargs)
            result = self.agent.get_result()
            return {
                "arm_values": result.arm_values,
                "arm_counts": result.arm_counts,
                "optimal_arm": result.optimal_arm,
                "total_reward": result.total_reward,
                "regret": result.regret,
                "method": result.method,
            }
        elif isinstance(self.agent, PolicyGradientAgent):
            self.agent.fit(n_episodes=n_iterations, **kwargs)
            result = self.agent.get_result()
            return {
                "optimal_actions": result.optimal_actions,
                "episode_rewards": result.episode_rewards[-10:],
                "best_reward": result.best_reward,
                "n_episodes": result.n_episodes,
                "method": "policy_gradient_optimization",
            }
        return {"method": "unknown"}