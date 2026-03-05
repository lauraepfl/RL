"""
REINFORCE Agent (Monte-Carlo Policy Gradient)
=============================================
A policy-gradient algorithm that optimises a parameterised *softmax* policy
directly from complete episode returns.

Reference: Williams (1992) – "Simple Statistical Gradient-Following Algorithms
for Connectionist Reinforcement Learning".
"""

from __future__ import annotations

import numpy as np


class REINFORCEAgent:
    """Tabular REINFORCE with a softmax policy and optional baseline.

    The policy is represented as a table of logits θ ∈ ℝ^{|S|×|A|}.
    At each state *s* the action probabilities are:

        π(a | s) = softmax(θ[s])

    Parameters
    ----------
    n_states:
        Size of the (discrete) observation space.
    n_actions:
        Size of the (discrete) action space.
    learning_rate:
        Step-size parameter α for the policy update.
    gamma:
        Discount factor γ ∈ [0, 1].
    use_baseline:
        When *True* subtract a state-value baseline (learned via a separate
        table) to reduce variance.
    baseline_lr:
        Learning rate for the value baseline.
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        use_baseline: bool = True,
        baseline_lr: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.baseline_lr = baseline_lr
        self._rng = np.random.default_rng(seed)

        # Policy logits
        self.theta = np.zeros((n_states, n_actions), dtype=np.float64)
        # Baseline (state-value estimates)
        self.value_table = np.zeros(n_states, dtype=np.float64)

        # Episode buffer: list of (state, action, reward)
        self._trajectory: list[tuple[int, int, float]] = []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """Sample an action from the current softmax policy at *state*."""
        probs = self._softmax(self.theta[state])
        return int(self._rng.choice(self.n_actions, p=probs))

    def store_transition(
        self, state: int, action: int, reward: float
    ) -> None:
        """Append a single ``(state, action, reward)`` to the episode buffer."""
        self._trajectory.append((state, action, reward))

    def update(self) -> float:
        """Compute Monte-Carlo returns and update the policy (and baseline).

        Should be called **once per episode** after the episode has finished.

        Returns
        -------
        float
            Mean absolute policy gradient magnitude (for monitoring).
        """
        if not self._trajectory:
            return 0.0

        states, actions, rewards = zip(*self._trajectory)
        T = len(rewards)

        # Compute discounted returns G_t
        returns = np.zeros(T, dtype=np.float64)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        # Normalise returns for stability
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        grad_magnitudes = []
        for t, (s, a) in enumerate(zip(states, actions)):
            G_t = returns[t]
            baseline = self.value_table[s] if self.use_baseline else 0.0
            advantage = G_t - baseline

            # Policy gradient update: θ[s] += α * advantage * ∇ log π(a|s)
            probs = self._softmax(self.theta[s])
            grad = -probs.copy()
            grad[a] += 1.0  # ∇ log π(a|s) = e_a - π(·|s)
            self.theta[s] += self.lr * advantage * grad
            grad_magnitudes.append(np.abs(advantage * grad).mean())

            # Baseline update (TD(0)-style)
            if self.use_baseline:
                self.value_table[s] += self.baseline_lr * (
                    G_t - self.value_table[s]
                )

        self._trajectory.clear()
        return float(np.mean(grad_magnitudes))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save policy logits and value table to a single `.npz` file."""
        np.savez(path, theta=self.theta, value_table=self.value_table)

    def load(self, path: str) -> None:
        """Load policy logits and value table from a `.npz` file."""
        data = np.load(path)
        self.theta = data["theta"]
        self.value_table = data["value_table"]
