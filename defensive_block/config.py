"""Configuration dataclass for the defensive block detection pipeline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DefensiveBlockConfig:
    # --- Minimum players ---
    min_outfield_players: int   = 4

    # --- Smoothing ---
    smoothing_window:     int   = 5
    smoothing_center:     bool  = True

    # --- Height thresholds (metres, canonical orientation) ---
    low_mid_boundary:     float = -15.0
    mid_high_boundary:    float =   5.0
    height_hysteresis:    float =   2.5

    # --- Length threshold ---
    compact_long_boundary: float = 18.0
    length_hysteresis:     float =  1.5

    # --- State machine initial states ---
    height_initial_state:  str  = "mid"
    length_initial_state:  str  = "compact"

    # --- NaN handling ---
    nan_strategy:          str  = "none"   # "none" | "forward_fill"

    # --- Aggregation ---
    min_sequence_coverage: float = 0.5
    match_weight_strategy: str   = "duration"  # "duration" | "uniform"
