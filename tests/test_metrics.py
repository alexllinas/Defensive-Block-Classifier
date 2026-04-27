"""Tests for metrics.py."""
from __future__ import annotations

import numpy as np
import pytest

from defensive_block.metrics import compute_median_x, compute_iqr_x


def _make_pos(x_values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Build (1, n, 2) pos_clean and (1,) valid_count from x_values."""
    n = len(x_values)
    pos = np.zeros((1, n, 2), dtype=float)
    pos[0, :, 0] = x_values
    counts = np.array([n])
    return pos, counts


class TestMedianX:
    def test_constant_positions(self):
        pos, counts = _make_pos([-20.0, -20.0, -20.0, -20.0])
        result = compute_median_x(pos, counts, min_outfield_players=4)
        np.testing.assert_allclose(result, [-20.0])

    def test_symmetric_spread(self):
        pos, counts = _make_pos([-30.0, -20.0, -10.0, 0.0])
        result = compute_median_x(pos, counts)
        np.testing.assert_allclose(result, [-15.0])

    def test_outlier_does_not_dominate(self):
        # One player pressing very high; median should remain close to the block
        pos, counts = _make_pos([-30.0, -25.0, -22.0, -20.0, +15.0])
        result = compute_median_x(pos, counts)
        assert result[0] < -10.0   # median should be near −22 m

    def test_nan_when_too_few_players(self):
        pos, counts = _make_pos([-20.0, -20.0, -20.0])
        counts_few = np.array([2])   # below min
        result = compute_median_x(pos, counts_few, min_outfield_players=4)
        assert np.isnan(result[0])

    def test_nan_frame_propagated(self):
        pos = np.full((3, 4, 2), np.nan)
        counts = np.array([0, 0, 0])
        result = compute_median_x(pos, counts)
        assert np.isnan(result).all()


class TestIQRX:
    def test_zero_iqr_same_x(self):
        pos, counts = _make_pos([-20.0, -20.0, -20.0, -20.0])
        result = compute_iqr_x(pos, counts)
        np.testing.assert_allclose(result, [0.0])

    def test_known_spread(self):
        # P25 = −20, P75 = −10 → IQR = 10
        pos, counts = _make_pos([-25.0, -20.0, -15.0, -10.0, -5.0])
        result = compute_iqr_x(pos, counts)
        assert 9.0 < result[0] < 16.0   # rough bound (exact depends on percentile method)

    def test_nan_when_too_few_players(self):
        pos, counts = _make_pos([-20.0, -15.0, -10.0])
        counts_few = np.array([2])
        result = compute_iqr_x(pos, counts_few, min_outfield_players=4)
        assert np.isnan(result[0])

    def test_iqr_robust_to_outliers(self):
        # IQR should be smaller than range (max − min)
        pos, counts = _make_pos([-40.0, -20.0, -18.0, -16.0, -14.0, -12.0, +30.0])
        median = compute_median_x(pos, counts)[0]
        iqr    = compute_iqr_x(pos, counts)[0]
        span   = 70.0   # max − min
        assert iqr < span / 2
