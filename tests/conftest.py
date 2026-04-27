"""Shared fixtures for the defensive block test suite.

All fixtures use synthetic data — no external files are required.
"""
from __future__ import annotations

import numpy as np
import pytest

from defensive_block.types import TrackingInput


def make_tracking_input(
    n_frames: int = 200,
    n_home: int = 11,
    n_away: int = 11,
    seed: int = 42,
    match_id: str = "test_match",
) -> TrackingInput:
    """Create a synthetic TrackingInput with predictable properties.

    All home outfield players are placed at x ≈ −20 m (deep defensive block).
    Away players are placed at x ≈ +20 m (attacking half).
    Possession alternates every 50 frames (home → away → home → …).
    """
    rng = np.random.default_rng(seed)
    n_players = n_home + n_away

    # Positions: home in negative half, away in positive half
    positions = np.zeros((n_frames, n_players, 2), dtype=float)
    positions[:, :n_home, 0] = rng.normal(-20.0, 3.0, (n_frames, n_home))
    positions[:, :n_home, 1] = rng.normal(0.0, 10.0, (n_frames, n_home))
    positions[:, n_home:, 0] = rng.normal(+20.0, 3.0, (n_frames, n_away))
    positions[:, n_home:, 1] = rng.normal(0.0, 10.0, (n_frames, n_away))

    in_field = np.ones((n_frames, n_players), dtype=bool)

    # Possession: alternating blocks of 50 frames
    possession = np.zeros(n_frames, dtype=int)
    for i in range(0, n_frames, 50):
        block = (i // 50) % 2
        possession[i:i + 50] = block

    in_play = np.ones(n_frames, dtype=bool)

    player_team  = np.array([0] * n_home + [1] * n_away, dtype=int)
    player_is_gk = np.zeros(n_players, dtype=bool)
    player_is_gk[0]      = True   # slot 0: home GK
    player_is_gk[n_home] = True   # slot n_home: away GK

    # Direction: home attacks right (goal at +52.5) in first half
    # → team_direction[0] = +1.0 for first 100 frames, −1.0 for second 100
    team_direction = np.ones((2, n_frames), dtype=float)
    team_direction[0, :n_frames // 2] = +1.0
    team_direction[0, n_frames // 2:] = -1.0
    team_direction[1, :n_frames // 2] = -1.0
    team_direction[1, n_frames // 2:] = +1.0

    segment = np.zeros(n_frames, dtype=int)
    segment[n_frames // 2:] = 1

    return TrackingInput(
        positions=positions,
        in_field=in_field,
        in_play=in_play,
        possession_team=possession,
        player_team=player_team,
        player_is_gk=player_is_gk,
        team_direction=team_direction,
        segment=segment,
        pitch_size=(105.0, 68.0),
        match_id=match_id,
    )


@pytest.fixture
def ti() -> TrackingInput:
    return make_tracking_input()


@pytest.fixture
def ti_all_low() -> TrackingInput:
    """TrackingInput where home team always defends deep (median x ≈ −35 m)."""
    ti = make_tracking_input()
    ti.positions[:, :10, 0] = -35.0   # 10 home outfield players fixed at −35 m
    return ti


@pytest.fixture
def ti_all_high() -> TrackingInput:
    """TrackingInput where home team presses high (median x ≈ +10 m)."""
    ti = make_tracking_input()
    ti.positions[:, :10, 0] = +10.0
    return ti


@pytest.fixture
def metrica_csv_dir(tmp_path):
    """Write minimal valid Metrica-format CSVs to a temp directory."""
    import csv

    n_frames = 100

    def _write_tracking(filepath, team_label, jerseys, x_centre):
        """Write a minimal 3-row Metrica tracking CSV."""
        rng = np.random.default_rng(0)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            # Row 0: team identifier per player pair
            row0 = ["", "", ""]
            for _ in jerseys:
                row0 += [team_label, ""]
            row0 += ["", ""]
            writer.writerow(row0)
            # Row 1: jersey numbers
            row1 = ["", "", ""]
            for j in jerseys:
                row1 += [j, ""]
            row1 += ["", ""]
            writer.writerow(row1)
            # Row 2: column names (Period, Frame, Time [s], Player<jersey>…, Ball,)
            row2 = ["Period", "Frame", "Time [s]"]
            for j in jerseys:
                row2 += [f"Player{j}", ""]
            row2 += ["Ball", ""]
            writer.writerow(row2)
            # Data rows
            for frame in range(1, n_frames + 1):
                period = 1 if frame <= 50 else 2
                row = [period, frame, round(frame * 0.04, 3)]
                for _ in jerseys:
                    row += [round(x_centre + rng.normal(0, 0.02), 4),
                            round(0.5 + rng.normal(0, 0.05), 4)]
                # ball
                row += [0.5, 0.5]
                writer.writerow(row)

    home_jerseys = list(range(1, 12))    # jerseys 1–11
    away_jerseys = list(range(12, 23))   # jerseys 12–22 (distinct from home)

    home_path   = tmp_path / "home.csv"
    away_path   = tmp_path / "away.csv"
    events_path = tmp_path / "events.csv"

    _write_tracking(home_path, "Home", home_jerseys, x_centre=0.25)
    _write_tracking(away_path, "Away", away_jerseys, x_centre=0.75)


    # Minimal events CSV
    with open(events_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Team", "Type", "Subtype", "Period",
                         "Start Frame", "End Frame", "From", "To",
                         "Start X", "Start Y", "End X", "End Y"])
        # Home has possession for frames 1–40
        writer.writerow(["Home", "PASS", "", 1, 1, 40, "", "", 0.25, 0.5, 0.5, 0.5])
        # Away has possession for frames 41–100
        writer.writerow(["Away", "PASS", "", 1, 41, 100, "", "", 0.75, 0.5, 0.5, 0.5])

    return {
        "home_tracking_path": home_path,
        "away_tracking_path": away_path,
        "events_path": events_path,
    }
