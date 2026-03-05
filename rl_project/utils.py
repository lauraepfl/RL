"""Utility helpers shared across agents and training scripts."""

from __future__ import annotations

from collections import deque

import numpy as np
import matplotlib.pyplot as plt


def moving_average(values: list[float], window: int = 50) -> np.ndarray:
    """Compute a centered moving average with the given *window* size."""
    if len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training(
    rewards: list[float],
    title: str = "Training rewards",
    window: int = 50,
    save_path: str | None = None,
) -> None:
    """Plot episode rewards and a smoothed moving average."""
    fig, ax = plt.subplots(figsize=(8, 4))
    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha=0.3, label="Episode reward")
    if len(rewards) >= window:
        smoothed = moving_average(rewards, window)
        offset = len(rewards) - len(smoothed)
        ax.plot(
            np.arange(offset, len(rewards)),
            smoothed,
            label=f"Moving avg ({window})",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


class ReplayBuffer:
    """Fixed-size experience replay buffer for off-policy algorithms.

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored.
    """

    def __init__(self, capacity: int) -> None:
        self._buf: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> list[tuple]:
        rng = rng or np.random.default_rng()
        indices = rng.choice(len(self._buf), size=batch_size, replace=False)
        return [self._buf[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buf)
