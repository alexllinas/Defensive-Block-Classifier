"""MetricaLoader — load Metrica Sports sample tracking data.

Reads the CSV-based tracking format from Games 1 and 2 of the Metrica Sports
open dataset (https://github.com/metrica-sports/sample-data).

Game 3 uses the FIFA EPTS XML format and is not supported by this loader.

CSV layout
----------
Row 0:  ``Period, Frame, Home_<jersey>,  , Home_<jersey>,  , …, ball,  ``
Row 1:  ``      ,      ,              x, y,              x, y, …,    x, y``
Row 2+: numeric data

Coordinate convention in raw CSV
---------------------------------
- x ∈ [0, 1], y ∈ [0, 1]
- Origin at top-left; (1, 1) = bottom-right.
- Field dimensions: 105 × 68 m.

After loading, positions are converted to model coordinates:
- x_m = x_norm × 105 − 52.5   (left goal line = −52.5)
- y_m = 34 − y_norm × 68      (Y flipped: origin at pitch centre, up = positive)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..types import TrackingInput
from .base import BaseLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PITCH_LENGTH = 105.0
_PITCH_WIDTH  =  68.0


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _parse_column_headers(filepath: Path) -> list[str]:
    """Build column names from the three-row Metrica header.

    Row 0: team identifiers (Home / Away / blank)
    Row 1: jersey numbers
    Row 2: Period, Frame, Time [s], Player<jersey>, (empty), …, Ball, (empty)

    Named columns in row 2 (Player* / Ball) are the x coordinate; the
    immediately following empty column is the y coordinate.
    """
    raw = pd.read_csv(filepath, nrows=3, header=None)
    row2 = [str(v).strip() if not pd.isna(v) else "" for v in raw.iloc[2]]

    cols: list[str] = []
    current: str = ""

    for name in row2:
        lower = name.lower()
        if name in ("Period", "Frame"):
            cols.append(name)
            current = ""
        elif lower in ("time [s]", "time"):
            cols.append("Time")
            current = ""
        elif lower.startswith("player") or lower == "ball":
            current = "ball" if lower == "ball" else name
            cols.append(f"{current}_x")
        elif not name and current:
            cols.append(f"{current}_y")
            current = ""        # reset — subsequent empties are trailing noise
        else:
            cols.append(f"_extra_{len(cols)}")

    return cols


def _read_tracking_csv(filepath: Path) -> pd.DataFrame:
    """Read a Metrica tracking CSV into a tidy DataFrame."""
    cols = _parse_column_headers(filepath)
    df   = pd.read_csv(filepath, skiprows=3, header=None, names=cols)
    df   = df.dropna(subset=["Frame"]).reset_index(drop=True)
    df["Frame"]  = df["Frame"].astype(int)
    df["Period"] = df["Period"].astype(int)
    return df


def _first_player_jersey(df: pd.DataFrame) -> int | None:
    """Return the jersey of the first Player*_x column (= GK in Metrica CSVs).

    Metrica always lists the goalkeeper first in each tracking file.
    This must be called on the raw DataFrame (before _prefix_player_cols renames
    the columns), when names still look like 'Player11_x'.
    """
    for col in df.columns:
        if col.startswith("Player") and col.endswith("_x"):
            try:
                return int(col[6:-2])
            except ValueError:
                pass
    return None


def _prefix_player_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Rename Player<n>_x/y columns to <prefix>_<n>_x/y.

    Converts the generic Player* names produced by _parse_column_headers into
    team-prefixed names (Home_11_x, Away_25_x, …) so columns from the home
    and away files don't clash after the merge.
    """
    rename: dict[str, str] = {}
    for col in df.columns:
        if col.startswith("Player") and (col.endswith("_x") or col.endswith("_y")):
            suffix = col[6:]           # "11_x" or "11_y"
            rename[col] = f"{prefix}_{suffix}"
    return df.rename(columns=rename)


def _norm_to_metres(x_norm: np.ndarray, y_norm: np.ndarray) -> np.ndarray:
    """Convert normalised [0,1] coords to centred metres.

    Returns (N, 2) array.
    """
    x_m = x_norm * _PITCH_LENGTH - (_PITCH_LENGTH / 2)
    y_m = (_PITCH_WIDTH / 2) - y_norm * _PITCH_WIDTH   # Y is flipped in Metrica
    return np.stack([x_m, y_m], axis=-1)


# ---------------------------------------------------------------------------
# Possession derivation from events
# ---------------------------------------------------------------------------

def _build_possession(
    events_path: Path,
    n_frames: int,
    team_name_to_idx: dict[str, int],
    frame_numbers: np.ndarray | None = None,
) -> np.ndarray:
    """Forward-fill team possession from the events CSV.

    Each event covers [Start Frame, End Frame] using Metrica's 1-based frame
    numbers. These are mapped to 0-based array indices via frame_numbers.
    Frames not covered by any event receive −1 (out of play for segmentation).

    Returns:
        (n_frames,) int — 0 or 1 for each tracked-possession frame, −1 otherwise.
    """
    possession = np.full(n_frames, -1, dtype=int)

    # Build frame-number → array-index lookup (Metrica frames are 1-indexed)
    if frame_numbers is not None:
        frame_to_idx: dict[int, int] = {int(fn): i for i, fn in enumerate(frame_numbers)}
    else:
        frame_to_idx = {i + 1: i for i in range(n_frames)}   # assume 1-indexed

    events = pd.read_csv(events_path)
    events.columns = [c.strip() for c in events.columns]

    required = {"Team", "Start Frame", "End Frame"}
    if not required.issubset(events.columns):
        return possession

    for _, row in events.iterrows():
        team_name = str(row["Team"]).strip()
        if team_name not in team_name_to_idx:
            continue
        team_idx = team_name_to_idx[team_name]
        s_frame  = int(row["Start Frame"])
        e_frame  = int(row["End Frame"])
        s_idx = frame_to_idx.get(s_frame, 0)
        e_idx = frame_to_idx.get(e_frame, n_frames - 1) + 1   # inclusive → exclusive
        possession[s_idx:e_idx] = team_idx

    return possession


# ---------------------------------------------------------------------------
# Attack direction inference
# ---------------------------------------------------------------------------

def _infer_team_direction(
    positions: np.ndarray,   # (N_frames, N_players, 2) — already in metres
    player_team: np.ndarray,  # (N_players,)
    in_field: np.ndarray,     # (N_frames, N_players)
    period: np.ndarray,       # (N_frames,)
) -> np.ndarray:
    """Infer per-frame team direction from mean pitch position per period.

    Convention:
        mean_x < 0  → team is in the left half → defends toward x = −52.5
                       → direction = −1.0  (no flip needed)
        mean_x > 0  → team is in the right half → defends toward x = +52.5
                       → direction = +1.0  (flip X to canonicalise)

    Returns:
        (2, N_frames) float — +1.0 or −1.0 per team per frame.
    """
    n_teams    = 2
    n_frames   = positions.shape[0]
    team_dir   = np.ones((n_teams, n_frames), dtype=float)

    for p in np.unique(period):
        period_mask = period == p
        for t in range(n_teams):
            team_mask = player_team == t
            x_vals = positions[np.ix_(np.where(period_mask)[0],
                                      np.where(team_mask)[0],
                                      [0])].squeeze(-1)   # (N_period, N_team)
            presence = in_field[np.ix_(np.where(period_mask)[0],
                                       np.where(team_mask)[0])]  # (N_period, N_team)
            valid_x = x_vals[presence]
            if len(valid_x) == 0:
                continue
            mean_x = float(np.mean(valid_x))
            dir_val = -1.0 if mean_x < 0 else 1.0
            team_dir[t, period_mask] = dir_val

    return team_dir


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class MetricaLoader(BaseLoader):
    """Load Metrica Sports open tracking data (Games 1 and 2).

    Example::

        from defensive_block import download_metrica_sample, MetricaLoader

        paths  = download_metrica_sample(game=1)
        loader = MetricaLoader()
        ti     = loader.load(**paths, analyzed_team="home")
    """

    def load(
        self,
        home_tracking_path: str | Path,
        away_tracking_path: str | Path,
        events_path:        str | Path,
        analyzed_team:      str   = "home",   # "home" or "away" — sets team index 0
        match_id:           str   = "",
        pitch_size:         tuple = (_PITCH_LENGTH, _PITCH_WIDTH),
    ) -> TrackingInput:
        """Load Metrica tracking and event data.

        Home team is always assigned team index 0; away team is index 1,
        regardless of which team is being analyzed. Pass ``analyzed_team_idx=0``
        or ``1`` to ``BlockDetectionPipeline.run()`` accordingly.

        Args:
            home_tracking_path: Path to Home tracking CSV.
            away_tracking_path: Path to Away tracking CSV.
            events_path: Path to events CSV.
            analyzed_team: Not used for slot assignment (always home=0, away=1);
                kept for API clarity and future use.
            match_id: Human-readable identifier stored in TrackingInput.
            pitch_size: (length_m, width_m). Default (105, 68).

        Returns:
            TrackingInput with all core fields populated. alpha_shape_order is
            always None (Metrica has no pre-computed concave hull).
        """
        _home_raw = _read_tracking_csv(Path(home_tracking_path))
        _away_raw = _read_tracking_csv(Path(away_tracking_path))

        # Capture GK jerseys before renaming (GK is always the first player in
        # each Metrica tracking file, regardless of jersey number).
        home_gk_jersey = _first_player_jersey(_home_raw)
        away_gk_jersey = _first_player_jersey(_away_raw)

        home_df = _prefix_player_cols(_home_raw, "Home")
        away_df = _prefix_player_cols(_away_raw, "Away")

        # Drop columns that exist in both files; home copy is kept.
        # Includes: ball_x/y (same ball data), Time, and any _extra_* noise cols.
        _away_drop = [c for c in away_df.columns
                      if c.lower().startswith("ball_")
                      or c in ("Time",)
                      or c.startswith("_extra_")]
        away_df = away_df.drop(columns=_away_drop, errors="ignore")

        _home_drop = [c for c in home_df.columns
                      if c in ("Time",) or c.startswith("_extra_")]
        home_df = home_df.drop(columns=_home_drop, errors="ignore")

        # Align on Frame (inner join — both files should have identical frames)
        df = pd.merge(home_df, away_df, on=["Period", "Frame"],
                      how="inner", suffixes=("_home", "_away"))
        df = df.sort_values("Frame").reset_index(drop=True)

        n_frames = len(df)
        period   = df["Period"].values.astype(int)

        # ---------------------------------------------------------------
        # Extract player slots
        # ---------------------------------------------------------------
        home_players, away_players, ball_cols = _identify_columns(df.columns.tolist())

        all_players = home_players + away_players
        n_players   = len(all_players)

        positions = np.full((n_frames, n_players, 2), np.nan, dtype=float)
        in_field  = np.zeros((n_frames, n_players), dtype=bool)

        player_team    = np.array([0] * len(home_players) + [1] * len(away_players), dtype=int)
        player_is_gk   = np.zeros(n_players, dtype=bool)
        player_jersey  = np.zeros(n_players, dtype=int)

        for slot_idx, (jersey, xcol, ycol) in enumerate(all_players):
            x_raw = df[xcol].values.astype(float)
            y_raw = df[ycol].values.astype(float)
            valid = ~(np.isnan(x_raw) | np.isnan(y_raw))

            metres = _norm_to_metres(x_raw, y_raw)
            positions[valid, slot_idx, :] = metres[valid]
            in_field[:, slot_idx] = valid

            player_jersey[slot_idx] = jersey
            team = player_team[slot_idx]
            if (team == 0 and jersey == home_gk_jersey) or \
               (team == 1 and jersey == away_gk_jersey):
                player_is_gk[slot_idx] = True

        # ---------------------------------------------------------------
        # Ball positions
        # ---------------------------------------------------------------
        ball_positions: Optional[np.ndarray] = None
        if ball_cols:
            bx = df[ball_cols[0]].values.astype(float)
            by = df[ball_cols[1]].values.astype(float)
            ball_metres = _norm_to_metres(bx, by)
            ball_valid  = ~np.isnan(bx)
            ball_arr    = np.full((n_frames, 2), np.nan, dtype=float)
            ball_arr[ball_valid] = ball_metres[ball_valid]
            ball_positions = ball_arr

        # ---------------------------------------------------------------
        # Period → segment (0-indexed)
        # ---------------------------------------------------------------
        period_vals = np.unique(period)
        period_to_seg = {p: i for i, p in enumerate(sorted(period_vals))}
        segment = np.array([period_to_seg[p] for p in period], dtype=int)

        # ---------------------------------------------------------------
        # Possession from events
        # ---------------------------------------------------------------
        team_name_to_idx = {"Home": 0, "Away": 1}
        possession = _build_possession(
            Path(events_path), n_frames, team_name_to_idx,
            frame_numbers=df["Frame"].values,
        )

        # ---------------------------------------------------------------
        # in_play: frames covered by at least one event
        # ---------------------------------------------------------------
        in_play = possession != -1

        # ---------------------------------------------------------------
        # Attack direction
        # ---------------------------------------------------------------
        team_direction = _infer_team_direction(positions, player_team, in_field, period)

        return TrackingInput(
            positions=positions,
            in_field=in_field,
            in_play=in_play,
            possession_team=possession,
            player_team=player_team,
            player_is_gk=player_is_gk,
            team_direction=team_direction,
            segment=segment,
            pitch_size=pitch_size,
            match_id=match_id,
            ball_positions=ball_positions,
            player_jersey=player_jersey,
            alpha_shape_order=None,
        )


# ---------------------------------------------------------------------------
# Column identification helper
# ---------------------------------------------------------------------------

def _identify_columns(
    columns: list[str],
) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]], list[str]]:
    """Separate home/away player columns and ball columns.

    Returns:
        home_players: list of (jersey_int, x_col, y_col)
        away_players: list of (jersey_int, x_col, y_col)
        ball_cols:    [x_col, y_col] or []
    """
    seen: dict[str, str] = {}   # base_name → x_col (waiting for y_col)

    home: list[tuple[int, str, str]] = []
    away: list[tuple[int, str, str]] = []
    ball: list[str] = []

    # Collect x columns first, then match with y columns
    x_cols = [c for c in columns if c.endswith("_x")]

    for xcol in x_cols:
        ycol = xcol[:-2] + "_y"
        if ycol not in columns:
            continue

        base = xcol[:-2]  # strip "_x"

        if base.lower().startswith("home_"):
            jersey_str = base.split("_")[-1]
            try:
                jersey = int(jersey_str)
            except ValueError:
                continue
            home.append((jersey, xcol, ycol))

        elif base.lower().startswith("away_"):
            jersey_str = base.split("_")[-1]
            try:
                jersey = int(jersey_str)
            except ValueError:
                continue
            away.append((jersey, xcol, ycol))

        elif base.lower() == "ball":
            ball = [xcol, ycol]

        # Handle suffixes added during merge (_home / _away on duplicate cols)
        elif base.lower().startswith("ball_"):
            if not ball:
                ball = [xcol, ycol]

    # Sort by jersey number for consistent slot ordering
    home.sort(key=lambda t: t[0])
    away.sort(key=lambda t: t[0])

    return home, away, ball
