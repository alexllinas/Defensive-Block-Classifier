"""Hysteresis state machines for defensive block classification.

Two independent classifiers:
  classify_height  — 3-state: "low" | "mid" | "high"
  classify_length  — 2-state: "compact" | "long"

Height transitions (no direct low ↔ high jump):
  low  → mid   when median_x > low_mid_boundary  + height_hysteresis
  mid  → low   when median_x < low_mid_boundary  − height_hysteresis
  mid  → high  when median_x > mid_high_boundary + height_hysteresis
  high → mid   when median_x < mid_high_boundary − height_hysteresis

Length transitions:
  compact → long    when iqr_x > compact_long_boundary + length_hysteresis
  long    → compact when iqr_x < compact_long_boundary − length_hysteresis
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import DefensiveBlockConfig


def classify_height(
    smoothed_median_x: np.ndarray,
    config: DefensiveBlockConfig,
) -> list[Optional[str]]:
    """Classify block height per frame using a hysteresis state machine.

    Args:
        smoothed_median_x: (N_frames,) float — NaN for invalid frames.
        config: Pipeline configuration.

    Returns:
        List of "low" | "mid" | "high" | None per frame.
    """
    low_mid  = config.low_mid_boundary
    mid_high = config.mid_high_boundary
    hyst     = config.height_hysteresis

    state  = config.height_initial_state
    states: list[Optional[str]] = []

    for x in smoothed_median_x:
        if np.isnan(x):
            states.append(None)
            continue

        if state == "low":
            if x > low_mid + hyst:
                state = "mid"
        elif state == "mid":
            if x < low_mid - hyst:
                state = "low"
            elif x > mid_high + hyst:
                state = "high"
        else:   # "high"
            if x < mid_high - hyst:
                state = "mid"

        states.append(state)

    return states


def classify_length(
    smoothed_iqr_x: np.ndarray,
    config: DefensiveBlockConfig,
) -> list[Optional[str]]:
    """Classify block length per frame using a hysteresis state machine.

    Args:
        smoothed_iqr_x: (N_frames,) float — NaN for invalid frames.
        config: Pipeline configuration.

    Returns:
        List of "compact" | "long" | None per frame.
    """
    boundary = config.compact_long_boundary
    hyst     = config.length_hysteresis

    state  = config.length_initial_state
    states: list[Optional[str]] = []

    for iqr in smoothed_iqr_x:
        if np.isnan(iqr):
            states.append(None)
            continue

        if state == "compact":
            if iqr > boundary + hyst:
                state = "long"
        else:   # "long"
            if iqr < boundary - hyst:
                state = "compact"

        states.append(state)

    return states


def combine_states(
    height_states: list[Optional[str]],
    length_states: list[Optional[str]],
) -> list[Optional[str]]:
    """Combine height and length states into a single label per frame.

    Returns:
        List of e.g. "mid_compact" | None if either component is None.
    """
    return [
        f"{h}_{l}" if (h is not None and l is not None) else None
        for h, l in zip(height_states, length_states)
    ]
