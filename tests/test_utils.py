"""Tests for utility helpers."""

from __future__ import annotations

import numpy as np
import pytest

from rl_project.utils import moving_average, ReplayBuffer


class TestMovingAverage:
    def test_length_shorter_than_window(self):
        vals = [1.0, 2.0, 3.0]
        result = moving_average(vals, window=10)
        np.testing.assert_allclose(result, vals)

    def test_constant_sequence(self):
        vals = [5.0] * 100
        result = moving_average(vals, window=10)
        np.testing.assert_allclose(result, 5.0)

    def test_output_length(self):
        vals = list(range(100))
        result = moving_average(vals, window=10)
        assert len(result) == 91  # 100 - 10 + 1


class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=10)
        buf.push(0, 1, 0.5, 1, False)
        assert len(buf) == 1

    def test_capacity_is_respected(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(i, 0, 0.0, i + 1, False)
        assert len(buf) == 5

    def test_sample_size(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(i, 0, 1.0, i + 1, False)
        batch = buf.sample(10)
        assert len(batch) == 10

    def test_sample_requires_enough_transitions(self):
        buf = ReplayBuffer(capacity=100)
        buf.push(0, 0, 0.0, 1, False)
        with pytest.raises(Exception):
            buf.sample(10)
