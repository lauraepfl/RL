"""
Q-Learning Agent
================
An off-policy temporal-difference (TD) control algorithm that learns the
optimal action-value function Q*(s, a) directly, regardless of the policy
being followed.

Reference: Watkins & Dayan (1992).
"""

from __future__ import annotations

import numpy as np


class QLearningAgent:
    """Tabular Q-learning with ε-greedy exploration.

    Parameters
    ----------
    n_states:
        Size of the (discrete) observation space.
    n_actions:
        Size of the (discrete) action space.
    learning_rate:
        Step-size parameter α ∈ (0, 1].
    gamma:
        Discount factor γ ∈ [0, 1].
    epsilon_start:
        Initial exploration rate ε.
    epsilon_end:
        Minimum exploration rate.
    epsilon_decay:
        Multiplicative decay applied to ε after every episode.
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._rng = np.random.default_rng(seed)

        # Q-table initialised to zeros
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Return an action for *state* using ε-greedy policy.

        Parameters
        ----------
        state:
            Current environment observation.
        greedy:
            When *True* the greedy (argmax) action is always returned,
            ignoring ε (useful at evaluation time).
        """
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """Apply one Q-learning update step.

        Returns
        -------
        float
            The TD error (useful for debugging / logging).
        """
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        """Decay the exploration rate at the end of each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the Q-table to a `.npy` file."""
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        """Load a Q-table from a `.npy` file."""
        self.q_table = np.load(path)
