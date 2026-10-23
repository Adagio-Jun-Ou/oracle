"""Tests for Reinforcement Learning decision optimizer."""
import pytest
import numpy as np


class TestQLearning:
    def test_q_learning_init(self):
        from oracle.prediction.reinforcement_learning import QLearningAgent
        agent = QLearningAgent(n_states=5, n_actions=3, seed=42)
        assert agent.n_states == 5
        assert agent.n_actions == 3
        assert agent.q_table.shape == (5, 3)

    def test_q_learning_fit(self):
        from oracle.prediction.reinforcement_learning import QLearningAgent
        agent = QLearningAgent(n_states=5, n_actions=3, seed=42)
        agent.fit(n_episodes=100)
        assert agent._fitted

    def test_q_learning_get_policy(self):
        from oracle.prediction.reinforcement_learning import QLearningAgent
        agent = QLearningAgent(n_states=5, n_actions=3, seed=42)
        agent.fit(n_episodes=100)
        result = agent.get_policy()
        assert len(result.optimal_policy) == 5
        assert len(result.optimal_values) == 5
        assert result.method == "q_learning"

    def test_q_learning_with_transitions(self):
        from oracle.prediction.reinforcement_learning import QLearningAgent
        rng = np.random.RandomState(42)
        transitions = [(rng.randint(0, 5), rng.randint(0, 3), rng.normal(0, 1), rng.randint(0, 5)) for _ in range(200)]
        agent = QLearningAgent(n_states=5, n_actions=3, seed=42)
        agent.fit(transitions=transitions)
        assert agent._fitted
        result = agent.get_policy()
        assert len(result.optimal_policy) == 5


class TestMultiArmedBandit:
    def test_ucb1(self):
        from oracle.prediction.reinforcement_learning import MultiArmedBandit
        bandit = MultiArmedBandit(n_arms=5, method="ucb1", seed=42)
        bandit.fit(n_pulls=200)
        result = bandit.get_result()
        assert len(result.arm_values) == 5
        assert len(result.arm_counts) == 5
        assert result.method == "bandit_ucb1"

    def test_thompson_sampling(self):
        from oracle.prediction.reinforcement_learning import MultiArmedBandit
        bandit = MultiArmedBandit(n_arms=5, method="thompson", seed=42)
        bandit.fit(n_pulls=200)
        result = bandit.get_result()
        assert result.method == "bandit_thompson"

    def test_epsilon_greedy(self):
        from oracle.prediction.reinforcement_learning import MultiArmedBandit
        bandit = MultiArmedBandit(n_arms=5, method="epsilon_greedy", seed=42)
        bandit.fit(n_pulls=200)
        result = bandit.get_result()
        assert result.method == "bandit_epsilon_greedy"

    def test_custom_values(self):
        from oracle.prediction.reinforcement_learning import MultiArmedBandit
        true_values = [0.1, 0.5, 0.8, 0.3, 0.6]
        bandit = MultiArmedBandit(n_arms=5, method="ucb1", seed=42)
        bandit.fit(n_pulls=300, true_values=true_values)
        result = bandit.get_result()
        assert result.optimal_arm == 2


class TestPolicyGradient:
    def test_policy_gradient_init(self):
        from oracle.prediction.reinforcement_learning import PolicyGradientAgent
        agent = PolicyGradientAgent(n_states=5, n_actions=3, seed=42)
        assert agent.policy_weights.shape == (5, 3)

    def test_policy_gradient_fit(self):
        from oracle.prediction.reinforcement_learning import PolicyGradientAgent
        agent = PolicyGradientAgent(n_states=5, n_actions=3, seed=42)
        agent.fit(n_episodes=50)
        assert agent._fitted

    def test_policy_gradient_result(self):
        from oracle.prediction.reinforcement_learning import PolicyGradientAgent
        agent = PolicyGradientAgent(n_states=5, n_actions=3, seed=42)
        agent.fit(n_episodes=50)
        result = agent.get_result()
        assert len(result.optimal_actions) == 5
        assert result.n_episodes == 50
        assert result.method == "policy_gradient"


class TestRLDecisionOptimizer:
    def test_q_learning_optimizer(self):
        from oracle.prediction.reinforcement_learning import RLDecisionOptimizer
        opt = RLDecisionOptimizer(method="q_learning", n_states=5, n_actions=3)
        result = opt.optimize(n_iterations=100)
        assert "optimal_policy" in result
        assert result["method"] == "q_learning_optimization"

    def test_bandit_ucb_optimizer(self):
        from oracle.prediction.reinforcement_learning import RLDecisionOptimizer
        opt = RLDecisionOptimizer(method="bandit_ucb", n_arms=5)
        result = opt.optimize(n_iterations=200)
        assert "optimal_arm" in result
        assert result["method"] == "bandit_ucb1"

    def test_policy_gradient_optimizer(self):
        from oracle.prediction.reinforcement_learning import RLDecisionOptimizer
        opt = RLDecisionOptimizer(method="policy_gradient", n_states=5, n_actions=3)
        result = opt.optimize(n_iterations=50)
        assert "optimal_actions" in result
        assert result["method"] == "policy_gradient_optimization"

    def test_unknown_method(self):
        from oracle.prediction.reinforcement_learning import RLDecisionOptimizer
        with pytest.raises(ValueError):
            RLDecisionOptimizer(method="unknown")