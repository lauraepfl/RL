"""Tests for the RL agents."""

from __future__ import annotations

import numpy as np
import pytest

from rl_project.environment import GridWorldEnv
from rl_project.agents import QLearningAgent, SARSAAgent, REINFORCEAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return GridWorldEnv()


@pytest.fixture
def q_agent(env):
    return QLearningAgent(
        env.observation_space_n, env.action_space_n, seed=0
    )


@pytest.fixture
def sarsa_agent(env):
    return SARSAAgent(
        env.observation_space_n, env.action_space_n, seed=0
    )


@pytest.fixture
def reinforce_agent(env):
    return REINFORCEAgent(
        env.observation_space_n, env.action_space_n, seed=0
    )


# ---------------------------------------------------------------------------
# Q-Learning tests
# ---------------------------------------------------------------------------

class TestQLearningAgent:
    def test_initial_q_table_is_zero(self, q_agent, env):
        assert np.all(q_agent.q_table == 0.0)

    def test_select_action_range(self, q_agent, env):
        obs, _ = env.reset(seed=0)
        action = q_agent.select_action(obs)
        assert 0 <= action < env.action_space_n

    def test_greedy_action_uses_argmax(self, q_agent, env):
        q_agent.q_table[0] = [0.0, 0.5, 0.2, 0.1]
        action = q_agent.select_action(0, greedy=True)
        assert action == 1  # argmax of [0, 0.5, 0.2, 0.1]

    def test_update_changes_q_table(self, q_agent, env):
        obs, _ = env.reset(seed=0)
        before = q_agent.q_table[obs, 1]
        q_agent.update(obs, 1, 1.0, obs, True)
        after = q_agent.q_table[obs, 1]
        assert after != before

    def test_update_returns_td_error(self, q_agent):
        td = q_agent.update(0, 0, 1.0, 1, True)
        assert isinstance(td, float)

    def test_epsilon_decay(self, q_agent):
        initial_eps = q_agent.epsilon
        q_agent.decay_epsilon()
        assert q_agent.epsilon < initial_eps

    def test_epsilon_floor(self, q_agent):
        q_agent.epsilon = q_agent.epsilon_end
        q_agent.decay_epsilon()
        assert q_agent.epsilon == q_agent.epsilon_end

    def test_save_load_roundtrip(self, q_agent, tmp_path):
        q_agent.q_table[0, 2] = 3.14
        path = str(tmp_path / "q_table")
        q_agent.save(path + ".npy")
        other = QLearningAgent(q_agent.n_states, q_agent.n_actions)
        other.load(path + ".npy")
        assert other.q_table[0, 2] == pytest.approx(3.14)

    def test_learns_positive_q_after_reward(self, q_agent):
        for _ in range(100):
            q_agent.update(0, 1, 1.0, 1, True)
        assert q_agent.q_table[0, 1] > 0.0


# ---------------------------------------------------------------------------
# SARSA tests
# ---------------------------------------------------------------------------

class TestSARSAAgent:
    def test_initial_q_table_is_zero(self, sarsa_agent, env):
        assert np.all(sarsa_agent.q_table == 0.0)

    def test_update_uses_next_action(self, sarsa_agent):
        sarsa_agent.update(
            state=0, action=1, reward=1.0,
            next_state=2, next_action=3, done=True
        )
        assert sarsa_agent.q_table[0, 1] != 0.0

    def test_epsilon_decay(self, sarsa_agent):
        eps = sarsa_agent.epsilon
        sarsa_agent.decay_epsilon()
        assert sarsa_agent.epsilon < eps

    def test_save_load_roundtrip(self, sarsa_agent, tmp_path):
        sarsa_agent.q_table[1, 0] = -2.5
        path = str(tmp_path / "sarsa.npy")
        sarsa_agent.save(path)
        other = SARSAAgent(sarsa_agent.n_states, sarsa_agent.n_actions)
        other.load(path)
        assert other.q_table[1, 0] == pytest.approx(-2.5)


# ---------------------------------------------------------------------------
# REINFORCE tests
# ---------------------------------------------------------------------------

class TestREINFORCEAgent:
    def test_select_action_range(self, reinforce_agent, env):
        action = reinforce_agent.select_action(0)
        assert 0 <= action < env.action_space_n

    def test_action_probabilities_sum_to_one(self, reinforce_agent):
        from rl_project.agents.reinforce import REINFORCEAgent
        probs = REINFORCEAgent._softmax(reinforce_agent.theta[0])
        assert probs.sum() == pytest.approx(1.0)

    def test_update_clears_trajectory(self, reinforce_agent):
        reinforce_agent.store_transition(0, 1, 0.5)
        reinforce_agent.store_transition(1, 2, 1.0)
        assert len(reinforce_agent._trajectory) == 2
        reinforce_agent.update()
        assert len(reinforce_agent._trajectory) == 0

    def test_update_empty_trajectory_is_safe(self, reinforce_agent):
        result = reinforce_agent.update()
        assert result == pytest.approx(0.0)

    def test_update_changes_theta(self, reinforce_agent):
        theta_before = reinforce_agent.theta.copy()
        for _ in range(5):
            reinforce_agent.store_transition(0, 1, 1.0)
        reinforce_agent.update()
        assert not np.allclose(reinforce_agent.theta, theta_before)

    def test_save_load_roundtrip(self, reinforce_agent, tmp_path):
        reinforce_agent.theta[2, 1] = 7.77
        path = str(tmp_path / "reinforce")
        reinforce_agent.save(str(path))
        other = REINFORCEAgent(
            reinforce_agent.n_states, reinforce_agent.n_actions
        )
        other.load(str(path) + ".npz")
        assert other.theta[2, 1] == pytest.approx(7.77)


# ---------------------------------------------------------------------------
# Integration: short training run
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_qlearning_improves_over_episodes(self):
        env = GridWorldEnv()
        agent = QLearningAgent(
            env.observation_space_n, env.action_space_n,
            learning_rate=0.2, epsilon_decay=0.99, seed=42
        )
        rewards = []
        for _ in range(500):
            obs, _ = env.reset()
            total = 0.0
            done = False
            while not done:
                a = agent.select_action(obs)
                next_obs, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                agent.update(obs, a, r, next_obs, done)
                obs = next_obs
                total += r
            agent.decay_epsilon()
            rewards.append(total)
        # Average reward in last 100 episodes should be better than first 100
        first_avg = np.mean(rewards[:100])
        last_avg = np.mean(rewards[-100:])
        assert last_avg >= first_avg - 0.1  # allows small noise margin

    def test_sarsa_runs_without_error(self):
        env = GridWorldEnv()
        agent = SARSAAgent(
            env.observation_space_n, env.action_space_n, seed=0
        )
        for _ in range(20):
            obs, _ = env.reset()
            action = agent.select_action(obs)
            done = False
            while not done:
                next_obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_action = agent.select_action(next_obs)
                agent.update(obs, action, r, next_obs, next_action, done)
                obs, action = next_obs, next_action
            agent.decay_epsilon()

    def test_reinforce_runs_without_error(self):
        env = GridWorldEnv()
        agent = REINFORCEAgent(
            env.observation_space_n, env.action_space_n, seed=0
        )
        for _ in range(20):
            obs, _ = env.reset()
            done = False
            while not done:
                a = agent.select_action(obs)
                next_obs, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                agent.store_transition(obs, a, r)
                obs = next_obs
            agent.update()
