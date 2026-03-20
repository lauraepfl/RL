"""
GridWorld Environment
=====================
A simple grid-based environment where an agent navigates from a start cell
to a goal cell while avoiding obstacles. Compatible with the Gymnasium API.

Grid legend:
  'S' – start position
  'G' – goal (reward +1.0)
  'H' – hole / obstacle (reward -1.0, episode ends)
  'F' – free cell (reward 0.0)
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Action constants
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

_ACTION_DELTA = {
    UP:    (-1,  0),
    RIGHT: ( 0,  1),
    DOWN:  ( 1,  0),
    LEFT:  ( 0, -1),
}

_DEFAULT_MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
]


class GridWorldEnv:
    """Tabular GridWorld environment.

    Parameters
    ----------
    grid_map:
        List of strings that define the grid layout.  Each character must be
        one of ``'S'``, ``'G'``, ``'H'``, or ``'F'``.
    is_slippery:
        When *True* the agent moves in the intended direction with probability
        ``1 - slip_prob`` and in one of the two perpendicular directions with
        probability ``slip_prob / 2`` each.
    slip_prob:
        Total probability of slipping sideways (only used when
        ``is_slippery=True``).
    max_steps:
        Maximum number of steps per episode before truncation.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_map: list[str] | None = None,
        is_slippery: bool = False,
        slip_prob: float = 0.2,
        max_steps: int = 200,
    ) -> None:
        self._map = list(grid_map or _DEFAULT_MAP)
        self.nrow = len(self._map)
        self.ncol = len(self._map[0])
        self.is_slippery = is_slippery
        self.slip_prob = slip_prob
        self.max_steps = max_steps

        # Parse map
        self._start: tuple[int, int] | None = None
        self._goals: list[tuple[int, int]] = []
        self._holes: set[tuple[int, int]] = set()
        for r, row in enumerate(self._map):
            for c, ch in enumerate(row):
                if ch == "S":
                    self._start = (r, c)
                elif ch == "G":
                    self._goals.append((r, c))
                elif ch == "H":
                    self._holes.add((r, c))
        if self._start is None:
            raise ValueError("Grid map must contain a start cell 'S'.")
        if not self._goals:
            raise ValueError("Grid map must contain at least one goal cell 'G'.")

        self.observation_space_n = self.nrow * self.ncol
        self.action_space_n = 4
        self._rng = np.random.default_rng()
        self._state: tuple[int, int] = self._start
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._start
        self._steps = 0
        return self._encode(self._state), {}

    def step(
        self, action: int
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Take *action* and return ``(obs, reward, terminated, truncated, info)``."""
        if action not in _ACTION_DELTA:
            raise ValueError(f"Invalid action {action}. Must be 0-3.")

        effective_action = self._apply_slip(action)
        dr, dc = _ACTION_DELTA[effective_action]
        nr = np.clip(self._state[0] + dr, 0, self.nrow - 1)
        nc = np.clip(self._state[1] + dc, 0, self.ncol - 1)
        self._state = (int(nr), int(nc))
        self._steps += 1

        terminated = False
        reward = 0.0
        if self._state in self._holes:
            reward = -1.0
            terminated = True
        elif self._state in self._goals:
            reward = 1.0
            terminated = True

        truncated = (not terminated) and (self._steps >= self.max_steps)
        return self._encode(self._state), reward, terminated, truncated, {}

    def render(self, mode: str = "ansi") -> str:
        """Return an ASCII string representation of the current grid."""
        rows = []
        for r, row in enumerate(self._map):
            line = ""
            for c, ch in enumerate(row):
                if (r, c) == self._state:
                    line += "A"
                else:
                    line += ch
            rows.append(line)
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, state: tuple[int, int]) -> int:
        return state[0] * self.ncol + state[1]

    def _decode(self, obs: int) -> tuple[int, int]:
        return obs // self.ncol, obs % self.ncol

    def _apply_slip(self, action: int) -> int:
        if not self.is_slippery:
            return action
        if self._rng.random() < self.slip_prob:
            perpendicular = [(action + 1) % 4, (action - 1) % 4]
            return int(self._rng.choice(perpendicular))
        return action
