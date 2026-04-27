"""Temporal smoothing for the defensive block detection pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd


def smooth_metric(
    values: np.ndarray,
    window_size: int = 5,
    center: bool = True,
) -> np.ndarray:
    """Apply a rolling nanmean to a 1-D metric array.

    NaN values are excluded from each window average (min_periods=1), so a
    window containing some NaNs still produces a valid output as long as at
    least one value is present.

    Args:
        values: (N_frames,) float array, may contain NaN.
        window_size: Number of frames in the rolling window.
        center: If True (default), centred window — suitable for post-hoc
            analysis. If False, causal window — suitable for real-time use.

    Returns:
        (N_frames,) float array — smoothed; NaN where all window values are NaN.
    """
    if window_size <= 1:
        return values.copy()

    return (
        pd.Series(values)
        .rolling(window=window_size, center=center, min_periods=1)
        .mean()
        .to_numpy()
    )
