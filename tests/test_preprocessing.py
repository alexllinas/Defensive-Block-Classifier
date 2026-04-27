"""Tests for preprocessing.py."""
from __future__ import annotations

import numpy as np
import pytest

from defensive_block.preprocessing import (
    get_team_slots,
    get_flip_mask,
    normalize_direction,
    apply_presence_mask,
    compute_valid_count,
    segment_defensive_sequences,
    preprocess,
)
from tests.conftest import make_tracking_input


class TestGetTeamSlots:
    def test_home_outfield_excludes_gk(self):
        ti = make_tracking_input(n_home=11, n_away=11)
        outfield, gk = get_team_slots(ti, analyzed_team_idx=0)
        assert 0 not in outfield      # slot 0 is home GK
        assert 0 in gk
        assert len(outfield) == 10    # 11 home - 1 GK

    def test_away_slots(self):
        ti = make_tracking_input(n_home=11, n_away=11)
        outfield, gk = get_team_slots(ti, analyzed_team_idx=1)
        assert 11 not in outfield     # slot 11 is away GK
        assert 11 in gk
        assert len(outfield) == 10


class TestFlipMask:
    def test_positive_direction_triggers_flip(self):
        ti = make_tracking_input(n_frames=10)
        ti.team_direction[0, :] = +1.0
        mask = get_flip_mask(ti, 0)
        assert mask.all()

    def test_negative_direction_no_flip(self):
        ti = make_tracking_input(n_frames=10)
        ti.team_direction[0, :] = -1.0
        mask = get_flip_mask(ti, 0)
        assert not mask.any()

    def test_per_frame_flip(self):
        ti = make_tracking_input(n_frames=10)
        ti.team_direction[0, :5]  = +1.0
        ti.team_direction[0, 5:]  = -1.0
        mask = get_flip_mask(ti, 0)
        assert mask[:5].all()
        assert not mask[5:].any()


class TestNormalizeDirection:
    def test_x_is_flipped(self):
        positions = np.ones((5, 3, 2)) * 10.0
        flip_mask = np.array([True, True, False, False, False])
        out = normalize_direction(positions, flip_mask)
        assert (out[:2, :, 0] == -10.0).all()
        assert (out[2:, :, 0] == +10.0).all()

    def test_y_is_never_flipped(self):
        positions = np.ones((5, 3, 2)) * 10.0
        flip_mask = np.ones(5, dtype=bool)
        out = normalize_direction(positions, flip_mask)
        assert (out[:, :, 1] == 10.0).all()

    def test_does_not_modify_input(self):
        positions = np.ones((4, 2, 2))
        original  = positions.copy()
        normalize_direction(positions, np.ones(4, dtype=bool))
        np.testing.assert_array_equal(positions, original)


class TestPresenceMask:
    def test_absent_slots_become_nan(self):
        positions     = np.ones((3, 4, 2))
        presence_mask = np.ones((3, 4), dtype=bool)
        presence_mask[1, 2] = False
        out = apply_presence_mask(positions, presence_mask)
        assert np.isnan(out[1, 2, 0])
        assert np.isnan(out[1, 2, 1])
        assert not np.isnan(out[0, 0, 0])

    def test_valid_count(self):
        presence = np.array([[True, True, False],
                              [True, False, False]])
        counts = compute_valid_count(presence)
        np.testing.assert_array_equal(counts, [2, 1])


class TestSequenceSegmentation:
    def test_basic_segmentation(self):
        ti = make_tracking_input(n_frames=100)
        # Frames 0–49: home possession (team 0), frames 50–99: away possession (team 1)
        ti.possession_team[:50]  = 0
        ti.possession_team[50:]  = 1
        ti.in_play[:] = True

        seqs = segment_defensive_sequences(ti, analyzed_team_idx=0)
        # Home defends when away has possession: frames 50–99
        assert len(seqs) == 1
        assert seqs[0].start_frame == 50
        assert seqs[0].end_frame   == 100

    def test_out_of_play_breaks_sequences(self):
        ti = make_tracking_input(n_frames=100)
        ti.possession_team[:] = 1   # away always in possession
        ti.in_play[:] = True
        ti.in_play[30:40] = False   # dead ball interrupts

        seqs = segment_defensive_sequences(ti, analyzed_team_idx=0)
        assert len(seqs) == 2
        assert seqs[0].end_frame == 30
        assert seqs[1].start_frame == 40

    def test_no_sequences_when_team_always_has_possession(self):
        ti = make_tracking_input(n_frames=50)
        ti.possession_team[:] = 0
        ti.in_play[:] = True
        seqs = segment_defensive_sequences(ti, analyzed_team_idx=0)
        assert len(seqs) == 0


class TestPreprocess:
    def test_returns_correct_shapes(self):
        ti = make_tracking_input(n_frames=100, n_home=11, n_away=11)
        pos_clean, presence, valid_count, flip_mask, seqs = preprocess(ti, 0)

        n_outfield = 10   # 11 home - 1 GK
        assert pos_clean.shape    == (100, n_outfield, 2)
        assert presence.shape     == (100, n_outfield)
        assert valid_count.shape  == (100,)
        assert flip_mask.shape    == (100,)

    def test_flip_mask_applied(self):
        ti = make_tracking_input(n_frames=10, n_home=11, n_away=11)
        ti.team_direction[0, :] = +1.0  # always flip
        ti.positions[:, :11, 0] = 20.0  # all home players at x = +20

        pos_clean, *_ = preprocess(ti, 0)
        # After flip, x should be −20
        assert (np.nanmean(pos_clean[:, :, 0]) < 0)
