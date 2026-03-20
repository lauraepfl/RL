"""Tests for the GridWorld environment."""

from __future__ import annotations

import pytest
from rl_project.environment import GridWorldEnv, UP, RIGHT, DOWN, LEFT


class TestGridWorldEnv:
    def setup_method(self):
        self.env = GridWorldEnv()

    def test_reset_returns_start_state(self):
        obs, info = self.env.reset(seed=0)
        # Default map: start 'S' is at (0,0) → encoded as 0
        assert obs == 0
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self):
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(RIGHT)
        assert isinstance(obs, int)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_within_bounds(self):
        self.env.reset(seed=1)
        for _ in range(50):
            action = 0
            obs, _, done, trunc, _ = self.env.step(action)
            assert 0 <= obs < self.env.observation_space_n
            if done or trunc:
                self.env.reset()

    def test_wall_clipping(self):
        """Moving UP from the top-left corner should stay at state 0."""
        self.env.reset(seed=0)
        obs, _, _, _, _ = self.env.step(UP)
        assert obs == 0  # Still top-left corner

    def test_goal_gives_positive_reward_and_terminates(self):
        """Walk a known path to the goal on the default 4×4 map."""
        # Default map goal 'G' is at (3,3), start at (0,0)
        # Path avoids holes: DOWN,DOWN,RIGHT,RIGHT,DOWN,RIGHT
        # (0,0)→(1,0)→(2,0)→(2,1)→(2,2)→(3,2)→(3,3)G
        path = [DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT]
        obs, _ = self.env.reset(seed=0)
        reward = 0.0
        terminated = False
        for action in path:
            obs, reward, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        assert terminated
        assert reward == pytest.approx(1.0)

    def test_hole_gives_negative_reward_and_terminates(self):
        """Step into a hole and check reward=-1 and termination."""
        # Default map: (1,1) is 'H' → encoded as 1*4+1=5
        # From (0,0): move RIGHT then DOWN
        obs, _ = self.env.reset(seed=0)
        obs, _, _, _, _ = self.env.step(RIGHT)   # (0,1)
        obs, reward, terminated, _, _ = self.env.step(DOWN)   # (1,1) – hole
        assert terminated
        assert reward == pytest.approx(-1.0)

    def test_truncation_on_max_steps(self):
        env = GridWorldEnv(max_steps=5)
        env.reset(seed=0)
        truncated = False
        for _ in range(6):
            _, _, terminated, truncated, _ = env.step(UP)
            if terminated or truncated:
                break
        assert truncated

    def test_render_contains_agent_marker(self):
        self.env.reset(seed=0)
        rendered = self.env.render()
        assert "A" in rendered

    def test_invalid_grid_no_start(self):
        with pytest.raises(ValueError, match="start"):
            GridWorldEnv(grid_map=["FFF", "FGF"])

    def test_invalid_grid_no_goal(self):
        with pytest.raises(ValueError, match="goal"):
            GridWorldEnv(grid_map=["SFF", "FFF"])

    def test_slippery_mode_does_not_crash(self):
        env = GridWorldEnv(is_slippery=True, slip_prob=0.5, max_steps=50)
        env.reset(seed=7)
        for _ in range(30):
            _, _, done, trunc, _ = env.step(RIGHT)
            if done or trunc:
                env.reset()
