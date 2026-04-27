"""Per-frame robust metrics for the defensive block detection pipeline.

Both metrics operate on the X axis only (longitudinal depth).
Input positions must already be direction-normalized (defender toward x = −52.5)
and presence-masked (absent players = NaN).
"""
from __future__ import annotations

import warnings

import numpy as np


def compute_median_x(
    pos_clean: np.ndarray,
    valid_count: np.ndarray,
    min_outfield_players: int = 4,
) -> np.ndarray:
    """Median X position of outfield defenders per frame.

    Robust estimator of block height: unaffected by one or two players
    pressing high or dropping deep.

    Args:
        pos_clean: (N_frames, n_outfield, 2) — NaN for absent players.
        valid_count: (N_frames,) int — number of valid players per frame.
        min_outfield_players: Frames with fewer valid players are set to NaN.

    Returns:
        (N_frames,) float64 — NaN where insufficient data.
    """
    x_values = pos_clean[:, :, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_x = np.nanmedian(x_values, axis=1)
    median_x[valid_count < min_outfield_players] = np.nan
    return median_x


def compute_iqr_x(
    pos_clean: np.ndarray,
    valid_count: np.ndarray,
    min_outfield_players: int = 4,
) -> np.ndarray:
    """IQR (P75 − P25) of X positions of outfield defenders per frame.

    Robust estimator of block length: unaffected by the deepest defender or
    most advanced presser individually.

    Args:
        pos_clean: (N_frames, n_outfield, 2) — NaN for absent players.
        valid_count: (N_frames,) int — number of valid players per frame.
        min_outfield_players: Frames with fewer valid players are set to NaN.

    Returns:
        (N_frames,) float64 — NaN where insufficient data.
    """
    x_values = pos_clean[:, :, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p25 = np.nanpercentile(x_values, 25, axis=1)
        p75 = np.nanpercentile(x_values, 75, axis=1)
    iqr_x = p75 - p25
    iqr_x[valid_count < min_outfield_players] = np.nan
    return iqr_x
