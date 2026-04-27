"""End-to-end pipeline for defensive block detection.

Typical usage:

    from defensive_block import BlockDetectionPipeline, DefensiveBlockConfig, MetricaLoader
    from defensive_block import download_metrica_sample

    paths    = download_metrica_sample(game=1)
    loader   = MetricaLoader()
    ti       = loader.load(**paths, analyzed_team="home")

    pipeline = BlockDetectionPipeline()
    match_summary, frame_results, sequences = pipeline.run(ti, analyzed_team_idx=0)
"""
from __future__ import annotations

from .config import DefensiveBlockConfig
from .preprocessing import preprocess
from .metrics import compute_median_x, compute_iqr_x
from .smoothing import smooth_metric
from .classification import classify_height, classify_length, combine_states
from .aggregation import aggregate_sequence, aggregate_match
from .types import (
    DefensiveSequence,
    FrameResults,
    MatchSummary,
    SequenceFrameData,
    SequenceSummary,
    TrackingInput,
)


class BlockDetectionPipeline:
    """End-to-end defensive block detection for a single match.

    Args:
        config: Pipeline configuration. Defaults to DefensiveBlockConfig().
    """

    def __init__(self, config: DefensiveBlockConfig | None = None) -> None:
        self.config = config or DefensiveBlockConfig()

    def run(
        self,
        ti: TrackingInput,
        analyzed_team_idx: int,
    ) -> tuple[MatchSummary, FrameResults, list[DefensiveSequence]]:
        """Run the full detection pipeline for one match.

        Args:
            ti: TrackingInput produced by MetricaLoader or any custom loader
                that subclasses BaseLoader and returns this struct.
            analyzed_team_idx: 0 or 1 — the defending team to analyse.

        Returns:
            (match_summary, frame_results, sequences)

            match_summary  — aggregated match-level statistics.
            frame_results  — per-frame smoothed metrics and state lists.
            sequences      — defensive sequences ordered by start_frame.
        """
        cfg = self.config

        # 1. Preprocess
        pos_clean, _, valid_count, _, sequences = preprocess(ti, analyzed_team_idx)

        # 2. Metrics
        median_x = compute_median_x(pos_clean, valid_count, cfg.min_outfield_players)
        iqr_x    = compute_iqr_x(pos_clean, valid_count, cfg.min_outfield_players)

        # 3. Smoothing
        sm_median_x = smooth_metric(median_x, cfg.smoothing_window, cfg.smoothing_center)
        sm_iqr_x    = smooth_metric(iqr_x,    cfg.smoothing_window, cfg.smoothing_center)

        # 4. Classification
        height_states   = classify_height(sm_median_x, cfg)
        length_states   = classify_length(sm_iqr_x, cfg)
        combined_states = combine_states(height_states, length_states)

        # 5. Frame results
        frame_results = FrameResults(
            smoothed_median_x=sm_median_x,
            smoothed_iqr_x=sm_iqr_x,
            height_states=height_states,
            length_states=length_states,
            combined_states=combined_states,
        )

        # 6. Per-sequence aggregation
        seq_summaries: list[SequenceSummary] = []
        for seq in sequences:
            s, e = seq.start_frame, seq.end_frame
            frame_data = SequenceFrameData(
                height_states=height_states[s:e],
                length_states=length_states[s:e],
                combined_states=combined_states[s:e],
                median_x=sm_median_x[s:e],
                iqr_x=sm_iqr_x[s:e],
            )
            seq_summaries.append(aggregate_sequence(seq, frame_data))

        # 7. Match aggregation
        match_summary = aggregate_match(ti.match_id, seq_summaries, cfg)

        return match_summary, frame_results, sequences
