"""Tests for io/metrica.py."""
from __future__ import annotations

import numpy as np
import pytest

from defensive_block.io.metrica import MetricaLoader, _norm_to_metres, _identify_columns


class TestCoordinateConversion:
    def test_centre_maps_to_zero(self):
        x = np.array([0.5])
        y = np.array([0.5])
        result = _norm_to_metres(x, y)
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-9)   # x
        np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-9)   # y

    def test_top_left_maps_to_negative_corner(self):
        x = np.array([0.0])
        y = np.array([0.0])
        result = _norm_to_metres(x, y)
        np.testing.assert_allclose(result[0, 0], -52.5)   # left goal line
        np.testing.assert_allclose(result[0, 1],  34.0)   # top touchline

    def test_bottom_right_maps_to_positive_corner(self):
        x = np.array([1.0])
        y = np.array([1.0])
        result = _norm_to_metres(x, y)
        np.testing.assert_allclose(result[0, 0], +52.5)
        np.testing.assert_allclose(result[0, 1], -34.0)

    def test_x_range(self):
        x = np.array([0.0, 0.5, 1.0])
        y = np.zeros(3)
        result = _norm_to_metres(x, y)
        assert result[0, 0] == pytest.approx(-52.5)
        assert result[1, 0] == pytest.approx(0.0)
        assert result[2, 0] == pytest.approx(52.5)


class TestIdentifyColumns:
    def test_home_away_ball(self):
        cols = [
            "Period", "Frame",
            "Home_1_x", "Home_1_y",
            "Home_7_x", "Home_7_y",
            "Away_11_x", "Away_11_y",
            "ball_x", "ball_y",
        ]
        home, away, ball = _identify_columns(cols)
        assert len(home) == 2
        assert len(away) == 1
        assert ball == ["ball_x", "ball_y"]
        jerseys_home = [t[0] for t in home]
        assert 1 in jerseys_home
        assert 7 in jerseys_home

    def test_gk_identified_by_jersey_1(self):
        cols = [
            "Period", "Frame",
            "Home_1_x", "Home_1_y",
            "Home_5_x", "Home_5_y",
        ]
        home, _, _ = _identify_columns(cols)
        jerseys = [t[0] for t in home]
        assert 1 in jerseys


class TestMetricaLoader:
    def test_loads_without_error(self, metrica_csv_dir):
        loader = MetricaLoader()
        ti = loader.load(**metrica_csv_dir, match_id="test_game")
        assert ti is not None

    def test_player_team_assignment(self, metrica_csv_dir):
        loader = MetricaLoader()
        ti = loader.load(**metrica_csv_dir)
        # 11 home (team 0) + 11 away (team 1)
        assert (ti.player_team == 0).sum() == 11
        assert (ti.player_team == 1).sum() == 11

    def test_gk_identification(self, metrica_csv_dir):
        loader = MetricaLoader()
        ti = loader.load(**metrica_csv_dir)
        # GK = first player listed in each CSV: home jersey 1, away jersey 12
        assert ti.player_is_gk.sum() == 2
        assert ti.player_is_gk[ti.player_jersey == 1].all()
        assert ti.player_is_gk[ti.player_jersey == 12].all()

    def test_positions_in_metres(self, metrica_csv_dir):
        loader = MetricaLoader()
        ti = loader.load(**metrica_csv_dir)
        # All positions should be within field bounds
        valid = ti.in_field
        x = ti.positions[valid, 0]
        y = ti.positions[valid, 1]
        assert x.min() >= -53.0 and x.max() <= 53.0
        assert y.min() >= -35.0 and y.max() <= 35.0

    def test_possession_derived_from_events(self, metrica_csv_dir):
        loader = MetricaLoader()
        ti = loader.load(**metrica_csv_dir)
        # Home has possession frames 1–40 (0-indexed: 0–39) → team 0
        # Away has possession frames 41–100 (0-indexed: 40–99) → team 1
        assert ti.possession_team[0]  == 0
        assert ti.possession_team[39] == 0
        assert ti.possession_team[40] == 1
        assert ti.possession_team[99] == 1

    def test_in_play_from_events(self, metrica_csv_dir):
        loader = MetricaLoader()
        ti = loader.load(**metrica_csv_dir)
        # All frames covered by events → all in_play
        assert ti.in_play.all()

    def test_alpha_shape_is_none(self, metrica_csv_dir):
        loader = MetricaLoader()
        ti = loader.load(**metrica_csv_dir)
        assert ti.alpha_shape_order is None

    def test_ball_positions_loaded(self, metrica_csv_dir):
        loader = MetricaLoader()
        ti = loader.load(**metrica_csv_dir)
        assert ti.ball_positions is not None
        assert ti.ball_positions.shape == (100, 2)

    def test_home_team_left_half(self, metrica_csv_dir):
        """Home players at x_norm ≈ 0.25 → x_metres ≈ −26.25 (negative half)."""
        loader = MetricaLoader()
        ti = loader.load(**metrica_csv_dir)
        home_mask = ti.player_team == 0
        home_x = ti.positions[:, home_mask, 0]
        assert np.nanmean(home_x) < 0

    def test_away_team_right_half(self, metrica_csv_dir):
        """Away players at x_norm ≈ 0.75 → x_metres ≈ +26.25 (positive half)."""
        loader = MetricaLoader()
        ti = loader.load(**metrica_csv_dir)
        away_mask = ti.player_team == 1
        away_x = ti.positions[:, away_mask, 0]
        assert np.nanmean(away_x) > 0
