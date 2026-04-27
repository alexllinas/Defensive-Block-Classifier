"""Data preprocessing for the defensive block detection pipeline.

All functions operate on a TrackingInput — a format-agnostic struct produced
by any loader (MetricaLoader or any custom BaseLoader subclass).

Key conventions:
  - analyzed_team_idx: 0 or 1 — the team whose defensive block we compute.
  - After normalize_direction(), the analyzed team always defends toward x = −52.5.
  - Defensive sequences are segmented from possession_team + in_play.
"""
from __future__ import annotations

import numpy as np

from .types import DefensiveSequence, TrackingInput


# ---------------------------------------------------------------------------
# Slot resolution
# ---------------------------------------------------------------------------

def get_team_slots(
    ti: TrackingInput,
    analyzed_team_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return outfield and GK slot indices for the analyzed team.

    Args:
        ti: TrackingInput produced by a loader.
        analyzed_team_idx: 0 or 1.

    Returns:
        (outfield_slots, gk_slots) — integer index arrays into the player axis.
    """
    is_team = ti.player_team == analyzed_team_idx
    is_gk   = ti.player_is_gk

    gk_slots       = np.where(is_team &  is_gk)[0]
    outfield_slots = np.where(is_team & ~is_gk)[0]
    return outfield_slots, gk_slots


# ---------------------------------------------------------------------------
# Direction normalization
# ---------------------------------------------------------------------------

def get_flip_mask(ti: TrackingInput, analyzed_team_idx: int) -> np.ndarray:
    """Return a per-frame boolean mask indicating where X must be flipped.

    team_direction[team, frame] == +1.0 means the team's goal is at x = +52.5
    (they defend toward positive X), so flip to bring into canonical orientation.

    Args:
        ti: TrackingInput.
        analyzed_team_idx: 0 or 1.

    Returns:
        (N_frames,) bool — True where X should be multiplied by −1.
    """
    return ti.team_direction[analyzed_team_idx] > 0


def normalize_direction(
    positions: np.ndarray,
    flip_mask: np.ndarray,
) -> np.ndarray:
    """Flip X so the analyzed team always defends toward x = −52.5.

    Args:
        positions: (N_frames, n_players, 2) float array.
        flip_mask: (N_frames,) bool.

    Returns:
        Copy with X flipped where indicated. Y is never modified.
    """
    out = positions.copy()
    out[flip_mask, :, 0] *= -1.0
    return out


# ---------------------------------------------------------------------------
# Presence mask
# ---------------------------------------------------------------------------

def get_outfield_presence(
    ti: TrackingInput,
    outfield_slots: np.ndarray,
) -> np.ndarray:
    """Return the per-frame presence mask for outfield players of the team.

    Args:
        ti: TrackingInput.
        outfield_slots: Slot indices from get_team_slots().

    Returns:
        (N_frames, n_outfield) bool.
    """
    return ti.in_field[:, outfield_slots]


def apply_presence_mask(
    positions: np.ndarray,
    presence_mask: np.ndarray,
) -> np.ndarray:
    """Set absent player positions to NaN.

    Args:
        positions: (N_frames, n_outfield, 2) float array.
        presence_mask: (N_frames, n_outfield) bool.

    Returns:
        Copy with NaN where presence_mask is False.
    """
    out = positions.copy()
    out[~presence_mask] = np.nan
    return out


def compute_valid_count(presence_mask: np.ndarray) -> np.ndarray:
    """Count valid (present) players per frame.

    Args:
        presence_mask: (N_frames, n_outfield) bool.

    Returns:
        (N_frames,) int.
    """
    return presence_mask.sum(axis=1)


# ---------------------------------------------------------------------------
# Position extraction
# ---------------------------------------------------------------------------

def extract_outfield_positions(
    ti: TrackingInput,
    outfield_slots: np.ndarray,
) -> np.ndarray:
    """Extract raw x/y positions for the outfield slots.

    Args:
        ti: TrackingInput.
        outfield_slots: Slot indices from get_team_slots().

    Returns:
        (N_frames, n_outfield, 2) float64 — raw, unflipped.
    """
    return ti.positions[:, outfield_slots, :]


# ---------------------------------------------------------------------------
# Sequence segmentation
# ---------------------------------------------------------------------------

def segment_defensive_sequences(
    ti: TrackingInput,
    analyzed_team_idx: int,
) -> list[DefensiveSequence]:
    """Identify contiguous defensive phases from possession and in_play.

    A defensive frame satisfies:
        in_play == True  AND  possession_team != analyzed_team_idx

    Out-of-play frames (in_play == False) break sequences.

    Args:
        ti: TrackingInput.
        analyzed_team_idx: 0 or 1.

    Returns:
        List of DefensiveSequence objects ordered by start_frame.
    """
    in_play    = ti.in_play
    possession = ti.possession_team
    segments   = ti.segment

    is_defending = in_play & (possession != analyzed_team_idx)

    sequences: list[DefensiveSequence] = []
    in_seq    = False
    seq_start = 0

    for t, defending in enumerate(is_defending):
        if defending and not in_seq:
            seq_start = t
            in_seq    = True
        elif not defending and in_seq:
            sequences.append(DefensiveSequence(
                match_id    = ti.match_id,
                segment     = int(segments[seq_start]),
                start_frame = seq_start,
                end_frame   = t,
            ))
            in_seq = False

    if in_seq:
        sequences.append(DefensiveSequence(
            match_id    = ti.match_id,
            segment     = int(segments[seq_start]),
            start_frame = seq_start,
            end_frame   = len(is_defending),
        ))

    return sequences


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    ti: TrackingInput,
    analyzed_team_idx: int,
) -> tuple[
    np.ndarray,   # pos_clean  (N_frames, n_outfield, 2)
    np.ndarray,   # presence_mask (N_frames, n_outfield)
    np.ndarray,   # valid_count (N_frames,)
    np.ndarray,   # flip_mask (N_frames,)
    list[DefensiveSequence],
]:
    """Normalize, mask, and segment a full match for the analyzed team.

    Args:
        ti: TrackingInput produced by a loader.
        analyzed_team_idx: 0 or 1.

    Returns:
        (pos_clean, presence_mask, valid_count, flip_mask, sequences)

        pos_clean     — direction-normalized outfield positions, NaN for absent players.
        presence_mask — bool mask used to build pos_clean.
        valid_count   — number of valid outfield players per frame.
        flip_mask     — per-frame X-flip flag (useful for alpha shape decoding).
        sequences     — defensive sequences sorted by start_frame.
    """
    outfield_slots, _ = get_team_slots(ti, analyzed_team_idx)
    flip_mask         = get_flip_mask(ti, analyzed_team_idx)
    positions         = extract_outfield_positions(ti, outfield_slots)
    positions         = normalize_direction(positions, flip_mask)
    presence_mask     = get_outfield_presence(ti, outfield_slots)
    pos_clean         = apply_presence_mask(positions, presence_mask)
    valid_count       = compute_valid_count(presence_mask)
    sequences         = segment_defensive_sequences(ti, analyzed_team_idx)

    return pos_clean, presence_mask, valid_count, flip_mask, sequences
