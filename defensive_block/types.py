"""Shared data structures for the defensive block detection pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Common input format
# ---------------------------------------------------------------------------

@dataclass
class TrackingInput:
    """Format-agnostic tracking data consumed by the pipeline.

    All loaders (MetricaLoader or any custom BaseLoader subclass) produce this struct.
    Positions are in metres with origin at the pitch centre:
        x ∈ [−52.5, 52.5]  (left goal line → right goal line)
        y ∈ [−34, 34]       (bottom touchline → top touchline)
    """
    # Core tracking arrays
    positions:         np.ndarray        # (N_frames, N_players, 2) — metres, raw
    in_field:          np.ndarray        # (N_frames, N_players) bool
    in_play:           np.ndarray        # (N_frames,) bool
    possession_team:   np.ndarray        # (N_frames,) int — 0 or 1; −1 = no possession

    # Player metadata (static, one entry per player slot)
    player_team:       np.ndarray        # (N_players,) int — 0 or 1
    player_is_gk:      np.ndarray        # (N_players,) bool

    # Direction: +1.0 → team defends toward x = +52.5 (flip X to canonicalise)
    #            −1.0 → team defends toward x = −52.5 (already canonical)
    team_direction:    np.ndarray        # (2, N_frames) float

    # Period index (0 = first half, 1 = second half)
    segment:           np.ndarray        # (N_frames,) int

    # Pitch geometry
    pitch_size:        tuple             # (length_m, width_m) — default (105.0, 68.0)

    # Identification
    match_id:          str

    # Optional — format-specific extras
    ball_positions:    Optional[np.ndarray] = None   # (N_frames, 2)
    player_jersey:     Optional[np.ndarray] = None   # (N_players,) int or str
    alpha_shape_order: Optional[np.ndarray] = None   # (N_frames, N_players) float


# ---------------------------------------------------------------------------
# Pipeline internal types
# ---------------------------------------------------------------------------

@dataclass
class DefensiveSequence:
    """A contiguous run of in-play frames where the analyzed team is defending."""
    match_id:    str
    segment:     int    # 0 = first half, 1 = second half
    start_frame: int    # inclusive
    end_frame:   int    # exclusive
    sequence_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.sequence_id = f"{self.match_id}_{self.start_frame}"

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class BlockState:
    """Detected block state for a single frame."""
    height:   str    # "low" | "mid" | "high"
    length:   str    # "compact" | "long"
    median_x: float
    iqr_x:    float

    @property
    def label(self) -> str:
        return f"{self.height}_{self.length}"


@dataclass
class SequenceFrameData:
    """Per-frame detection results sliced to a single defensive sequence."""
    height_states:   list[Optional[str]]
    length_states:   list[Optional[str]]
    combined_states: list[Optional[str]]
    median_x:        np.ndarray   # (n_frames,)
    iqr_x:           np.ndarray   # (n_frames,)


@dataclass
class SequenceSummary:
    """Aggregated detection results for a single defensive sequence."""
    sequence_id:          str
    match_id:             str
    segment:              int
    n_total_frames:       int
    n_valid_frames:       int
    coverage_ratio:       float

    dominant_combined:    Optional[str]
    dominant_height:      Optional[str]
    dominant_length:      Optional[str]

    height_dist:          dict[str, float]
    length_dist:          dict[str, float]
    combined_dist:        dict[str, float]

    height_transitions:   int
    length_transitions:   int
    combined_transitions: int

    mean_median_x:        float
    std_median_x:         float
    mean_iqr_x:           float
    std_iqr_x:            float


@dataclass
class FrameResults:
    """Per-frame outputs of the full detection pipeline (all frames in the match)."""
    smoothed_median_x: np.ndarray           # (N_frames,)
    smoothed_iqr_x:    np.ndarray           # (N_frames,)
    height_states:     list[Optional[str]]  # (N_frames,)
    length_states:     list[Optional[str]]  # (N_frames,)
    combined_states:   list[Optional[str]]  # (N_frames,)

    def block_state_at(self, frame: int) -> Optional[BlockState]:
        h = self.height_states[frame]
        l = self.length_states[frame]
        if h is None or l is None:
            return None
        return BlockState(
            height=h,
            length=l,
            median_x=float(self.smoothed_median_x[frame]),
            iqr_x=float(self.smoothed_iqr_x[frame]),
        )


@dataclass
class MatchSummary:
    """Aggregated detection results for a full match."""
    match_id:                 str
    n_sequences:              int
    total_valid_frames:       int

    dominant_combined:        Optional[str]
    dominant_height:          Optional[str]
    dominant_length:          Optional[str]

    match_combined_dist:      dict[str, float]
    match_height_dist:        dict[str, float]
    match_length_dist:        dict[str, float]

    match_mean_median_x:      float
    match_mean_iqr_x:         float

    height_transition_rate:   float
    length_transition_rate:   float
    combined_transition_rate: float

    sequences_by_type:        dict[str, int]
