"""Tests for classification.py (hysteresis state machines)."""
from __future__ import annotations

import numpy as np
import pytest

from defensive_block.classification import (
    classify_height,
    classify_length,
    combine_states,
)
from defensive_block.config import DefensiveBlockConfig


@pytest.fixture
def cfg() -> DefensiveBlockConfig:
    return DefensiveBlockConfig(
        low_mid_boundary=-15.0,
        mid_high_boundary=5.0,
        height_hysteresis=2.5,
        compact_long_boundary=18.0,
        length_hysteresis=1.5,
        height_initial_state="mid",
        length_initial_state="compact",
    )


class TestHeightClassifier:
    def test_stable_low(self, cfg):
        values = np.full(50, -30.0)
        states = classify_height(values, cfg)
        # Must pass through mid first, but since initial_state="mid" and
        # −30 < −17.5 (mid_to_low_exit), should transition to low quickly
        assert states[-1] == "low"
        assert all(s in ("low", "mid") for s in states if s is not None)

    def test_stable_mid(self, cfg):
        values = np.full(20, -10.0)   # between boundaries
        states = classify_height(values, cfg)
        assert all(s == "mid" for s in states)

    def test_stable_high(self, cfg):
        # Start high by pushing above +7.5 first
        ramp = np.array([0.0, 3.0, 6.0, 8.0])
        values = np.concatenate([ramp, np.full(20, 10.0)])
        states = classify_height(values, cfg)
        assert states[-1] == "high"

    def test_transition_low_to_mid(self, cfg):
        # Sit at low, then cross the entry threshold at −12.5
        values = np.concatenate([np.full(10, -30.0), np.full(10, -12.0)])
        states = classify_height(values, cfg)
        # First part should settle to "low", second to "mid"
        assert states[9]  == "low"
        assert states[-1] == "mid"

    def test_no_direct_low_to_high(self, cfg):
        # Jump from −30 to +15 in one step — must pass through mid
        values = np.array([-30.0, +15.0])
        states = classify_height(values, cfg)
        # Should never have a "low" at i and "high" at i+1 directly
        for a, b in zip(states, states[1:]):
            if a == "low" and b is not None:
                assert b != "high", "Direct low→high transition detected"

    def test_hysteresis_prevents_oscillation(self, cfg):
        # Oscillate just inside the hysteresis band: −13, −16, −13, −16, …
        # −13 > −12.5 (entry) → transitions to mid then stays
        # −16 is above −17.5 (exit) → no exit from mid
        values = np.tile([-13.0, -16.0], 10)
        states = classify_height(values, cfg)
        # Once we enter mid, we should stay mid (−16 > −17.5 so no exit)
        mid_entered = False
        for s in states:
            if s == "mid":
                mid_entered = True
            if mid_entered and s is not None:
                assert s == "mid", "Oscillation detected: exited mid unexpectedly"

    def test_nan_passthrough(self, cfg):
        values = np.array([-10.0, np.nan, -10.0])
        states = classify_height(values, cfg)
        assert states[1] is None
        assert states[0] == "mid"
        assert states[2] == "mid"


class TestLengthClassifier:
    def test_stable_compact(self, cfg):
        values = np.full(20, 10.0)
        states = classify_length(values, cfg)
        assert all(s == "compact" for s in states)

    def test_stable_long(self, cfg):
        values = np.concatenate([np.full(5, 20.0), np.full(20, 25.0)])
        states = classify_length(values, cfg)
        assert states[-1] == "long"

    def test_compact_to_long_transition(self, cfg):
        # Cross entry threshold (19.5) from compact
        values = np.concatenate([np.full(5, 10.0), np.full(5, 20.0)])
        states = classify_length(values, cfg)
        assert states[4] == "compact"
        assert states[-1] == "long"

    def test_long_to_compact_transition(self, cfg):
        # Start long, then drop below exit threshold (16.5)
        values = np.concatenate([np.full(5, 25.0), np.full(5, 15.0)])
        states = classify_length(values, cfg)
        assert states[4] == "long"
        assert states[-1] == "compact"

    def test_nan_passthrough(self, cfg):
        values = np.array([10.0, np.nan, 10.0])
        states = classify_length(values, cfg)
        assert states[1] is None


class TestCombineStates:
    def test_both_valid(self):
        h = ["mid", "low", "high"]
        l = ["compact", "long", "compact"]
        c = combine_states(h, l)
        assert c == ["mid_compact", "low_long", "high_compact"]

    def test_none_propagates(self):
        h = ["mid", None, "high"]
        l = ["compact", "compact", None]
        c = combine_states(h, l)
        assert c[0] == "mid_compact"
        assert c[1] is None
        assert c[2] is None
