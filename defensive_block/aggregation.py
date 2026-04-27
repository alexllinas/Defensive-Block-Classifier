"""Aggregation functions for the defensive block detection pipeline."""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from .config import DefensiveBlockConfig
from .types import (
    DefensiveSequence,
    SequenceFrameData,
    SequenceSummary,
    MatchSummary,
)

_HEIGHT_LABELS   = ("low", "mid", "high")
_LENGTH_LABELS   = ("compact", "long")
_COMBINED_LABELS = tuple(f"{h}_{l}" for h in _HEIGHT_LABELS for l in _LENGTH_LABELS)


def _transition_count(states: list[Optional[str]]) -> int:
    count = 0
    prev: Optional[str] = None
    for s in states:
        if s is None:
            continue
        if prev is not None and s != prev:
            count += 1
        prev = s
    return count


def _dominant(states: list[Optional[str]]) -> Optional[str]:
    valid = [s for s in states if s is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def _distribution(
    states: list[Optional[str]],
    labels: tuple[str, ...],
) -> dict[str, float]:
    valid = [s for s in states if s is not None]
    n = len(valid)
    if n == 0:
        return {lbl: 0.0 for lbl in labels}
    counts = Counter(valid)
    return {lbl: counts.get(lbl, 0) / n for lbl in labels}


def aggregate_sequence(
    seq: DefensiveSequence,
    frame_data: SequenceFrameData,
) -> SequenceSummary:
    """Compute per-sequence statistics from per-frame detection results."""
    h = frame_data.height_states
    l = frame_data.length_states
    c = frame_data.combined_states

    n_total  = seq.n_frames
    n_valid  = sum(1 for s in h if s is not None)
    coverage = n_valid / n_total if n_total > 0 else 0.0

    med_x = frame_data.median_x
    iqr_x = frame_data.iqr_x

    return SequenceSummary(
        sequence_id=seq.sequence_id,
        match_id=seq.match_id,
        segment=seq.segment,
        n_total_frames=n_total,
        n_valid_frames=n_valid,
        coverage_ratio=coverage,
        dominant_combined=_dominant(c),
        dominant_height=_dominant(h),
        dominant_length=_dominant(l),
        height_dist=_distribution(h, _HEIGHT_LABELS),
        length_dist=_distribution(l, _LENGTH_LABELS),
        combined_dist=_distribution(c, _COMBINED_LABELS),
        height_transitions=_transition_count(h),
        length_transitions=_transition_count(l),
        combined_transitions=_transition_count(c),
        mean_median_x=float(np.nanmean(med_x)) if len(med_x) > 0 else float("nan"),
        std_median_x=float(np.nanstd(med_x))   if len(med_x) > 0 else float("nan"),
        mean_iqr_x=float(np.nanmean(iqr_x))    if len(iqr_x) > 0 else float("nan"),
        std_iqr_x=float(np.nanstd(iqr_x))      if len(iqr_x) > 0 else float("nan"),
    )


def aggregate_match(
    match_id: str,
    summaries: list[SequenceSummary],
    config: DefensiveBlockConfig,
) -> MatchSummary:
    """Aggregate per-sequence summaries into a match-level summary.

    Sequences below config.min_sequence_coverage are excluded from distributions
    but still counted in n_sequences.
    """
    eligible = [s for s in summaries if s.coverage_ratio >= config.min_sequence_coverage]
    total_valid = sum(s.n_valid_frames for s in summaries)

    if not eligible:
        empty_h = {lbl: 0.0 for lbl in _HEIGHT_LABELS}
        empty_l = {lbl: 0.0 for lbl in _LENGTH_LABELS}
        empty_c = {lbl: 0.0 for lbl in _COMBINED_LABELS}
        return MatchSummary(
            match_id=match_id, n_sequences=len(summaries),
            total_valid_frames=total_valid,
            dominant_combined=None, dominant_height=None, dominant_length=None,
            match_combined_dist=empty_c, match_height_dist=empty_h,
            match_length_dist=empty_l,
            match_mean_median_x=float("nan"), match_mean_iqr_x=float("nan"),
            height_transition_rate=float("nan"),
            length_transition_rate=float("nan"),
            combined_transition_rate=float("nan"),
            sequences_by_type={},
        )

    if config.match_weight_strategy == "duration":
        weights = np.array([s.n_valid_frames for s in eligible], dtype=float)
    else:
        weights = np.ones(len(eligible), dtype=float)

    w_sum = weights.sum() or 1.0

    def _wdist(dists: list[dict[str, float]], labels: tuple[str, ...]) -> dict[str, float]:
        return {
            lbl: float(sum(w * d[lbl] for w, d in zip(weights, dists)) / w_sum)
            for lbl in labels
        }

    mhd = _wdist([s.height_dist   for s in eligible], _HEIGHT_LABELS)
    mld = _wdist([s.length_dist   for s in eligible], _LENGTH_LABELS)
    mcd = _wdist([s.combined_dist for s in eligible], _COMBINED_LABELS)

    mmx = float(sum(w * s.mean_median_x for w, s in zip(weights, eligible)
                    if not np.isnan(s.mean_median_x)) / w_sum)
    miq = float(sum(w * s.mean_iqr_x   for w, s in zip(weights, eligible)
                    if not np.isnan(s.mean_iqr_x)) / w_sum)

    total_wf = sum(w * s.n_valid_frames for w, s in zip(weights, eligible)) or 1.0
    hr = sum(w * s.height_transitions   for w, s in zip(weights, eligible)) / total_wf
    lr = sum(w * s.length_transitions   for w, s in zip(weights, eligible)) / total_wf
    cr = sum(w * s.combined_transitions for w, s in zip(weights, eligible)) / total_wf

    return MatchSummary(
        match_id=match_id,
        n_sequences=len(summaries),
        total_valid_frames=total_valid,
        dominant_combined=max(mcd, key=mcd.get),
        dominant_height=max(mhd, key=mhd.get),
        dominant_length=max(mld, key=mld.get),
        match_combined_dist=mcd,
        match_height_dist=mhd,
        match_length_dist=mld,
        match_mean_median_x=mmx,
        match_mean_iqr_x=miq,
        height_transition_rate=hr,
        length_transition_rate=lr,
        combined_transition_rate=cr,
        sequences_by_type=dict(Counter(s.dominant_combined for s in eligible
                                       if s.dominant_combined)),
    )
