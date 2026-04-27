"""Defensive block visualizer.

Renders player positions on a mplsoccer pitch with:
  - Three colour-coded zone overlays (low / mid / high block).
  - Optional concave hull (alpha shape) of the defending team.
  - Player markers differentiated by team and role.
  - Frame title and metric annotations.

Supports static frame export and MP4/GIF animation.

Coordinate convention (matches TrackingInput):
    x ∈ [−52.5, 52.5]  (left goal → right goal)
    y ∈ [−34, 34]       (bottom touchline → top touchline)
    Origin at pitch centre.

Internally, positions are shifted to mplsoccer's custom pitch coordinate
system (x ∈ [0, 105], y ∈ [0, 68]) before rendering. The public API always
accepts and returns model coordinates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D

import mplsoccer

from .config import DefensiveBlockConfig
from .types import BlockState, DefensiveSequence, FrameResults


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

# Model coords (metres, centred) → mplsoccer custom pitch coords (0–105, 0–68)
_X_OFFSET = 52.5
_Y_OFFSET = 34.0


def _px(x: float) -> float:
    return x + _X_OFFSET


def _py(y: float) -> float:
    return y + _Y_OFFSET


def _plot_xy(model_xy: np.ndarray) -> np.ndarray:
    """Shift a (..., 2) array from model coords to plot coords."""
    out = model_xy.copy().astype(float)
    out[..., 0] += _X_OFFSET
    out[..., 1] += _Y_OFFSET
    return out


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

FIELD_X_MIN, FIELD_X_MAX = -52.5, 52.5
FIELD_Y_MIN, FIELD_Y_MAX = -34.0,  34.0

_ZONE_COLOUR         = "#F4B942"   # amber for all zones
_ZONE_ALPHA_ACTIVE   = 0.42
_ZONE_ALPHA_INACTIVE = 0.09

_DEFENDER_FC = "#FFFFFF"
_ATTACKER_FC = "#E63946"
_GK_FC       = "#FFD700"
_EDGE_C      = "#1A1A2E"
_BALL_FC     = "#FFD700"
_MARKER_LW   = 1.8
_MARKER_S    = 260

_FIG_FC      = "#0D1B2A"

_TITLE_FS      = 14
_SUBTITLE_FS   = 9
_ZONE_LABEL_FS = 11


# ---------------------------------------------------------------------------
# Pitch factory
# ---------------------------------------------------------------------------

def _create_pitch(pitch_length: float = 105.0, pitch_width: float = 68.0) -> mplsoccer.Pitch:
    return mplsoccer.Pitch(
        pitch_type="custom",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        pitch_color="grass",
        line_color="white",
        stripe=True,
        stripe_color="#3d7a3d",
        goal_type="box",
        linewidth=1.5,
    )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_block_zones(
    ax: plt.Axes,
    config: DefensiveBlockConfig,
    active_height: str,
) -> None:
    zones = [
        ("low",  FIELD_X_MIN,             config.low_mid_boundary),
        ("mid",  config.low_mid_boundary,  config.mid_high_boundary),
        ("high", config.mid_high_boundary, FIELD_X_MAX),
    ]
    for zone_id, x0, x1 in zones:
        alpha  = _ZONE_ALPHA_ACTIVE if zone_id == active_height else _ZONE_ALPHA_INACTIVE
        x0p, x1p = _px(x0), _px(x1)
        rect = mpatches.FancyBboxPatch(
            (x0p, 0), x1p - x0p, _py(FIELD_Y_MAX),
            boxstyle="square,pad=0",
            facecolor=_ZONE_COLOUR, edgecolor="none",
            alpha=alpha, zorder=2,
        )
        ax.add_patch(rect)

        label_alpha  = 0.90 if zone_id == active_height else 0.28
        label_weight = "bold" if zone_id == active_height else "normal"
        ax.text(
            (x0p + x1p) / 2, _py(FIELD_Y_MIN) + 2.0,
            zone_id.upper() + " BLOCK",
            color="white", alpha=label_alpha,
            fontsize=_ZONE_LABEL_FS, fontweight=label_weight,
            ha="center", va="bottom", zorder=5,
        )


def _draw_alpha_shape(
    ax: plt.Axes,
    polygon: np.ndarray,
    active_height: str,
) -> None:
    """Draw a concave hull polygon of the defending team.

    Args:
        polygon: (n_vertices + 1, 2) closed polygon in model coords.
        active_height: Determines fill colour.
    """
    if polygon is None or len(polygon) < 3:
        return
    polygon_plot = _plot_xy(polygon[:-1])   # shift; matplotlib closes automatically
    patch = mpatches.Polygon(
        polygon_plot,
        closed=True,
        facecolor=_ZONE_COLOUR,
        edgecolor=_ZONE_COLOUR,
        alpha=0.32,
        linewidth=4.0,
        linestyle="-",
        zorder=3,
    )
    ax.add_patch(patch)


def _draw_players(
    ax: plt.Axes,
    pitch: mplsoccer.Pitch,
    positions: np.ndarray,
    team_indices: dict[str, list[int]],
    goalkeeper_indices: list[int],
    jersey_numbers: Optional[dict[int, str]] = None,
) -> None:
    gk_set  = set(goalkeeper_indices)
    def_set = set(team_indices.get("defending", []))

    for role, indices in team_indices.items():
        fc = _DEFENDER_FC if role == "defending" else _ATTACKER_FC
        tc = _EDGE_C      if role == "defending" else "white"
        for idx in indices:
            if idx >= len(positions):
                continue
            x, y = float(positions[idx, 0]), float(positions[idx, 1])
            if np.isnan(x) or np.isnan(y):
                continue
            pitch.scatter(_px(x), _py(y), ax=ax, s=_MARKER_S, c=fc,
                          edgecolors=_EDGE_C, linewidths=_MARKER_LW,
                          marker="o", zorder=10)
            if jersey_numbers and idx in jersey_numbers:
                ax.text(_px(x), _py(y), jersey_numbers[idx],
                        color=tc, fontsize=7.5, fontweight="bold",
                        ha="center", va="center", zorder=11)

    for idx in goalkeeper_indices:
        if idx >= len(positions):
            continue
        x, y = float(positions[idx, 0]), float(positions[idx, 1])
        if np.isnan(x) or np.isnan(y):
            continue
        fc = _DEFENDER_FC if idx in def_set else _ATTACKER_FC
        tc = _EDGE_C      if idx in def_set else "white"
        pitch.scatter(_px(x), _py(y), ax=ax, s=_MARKER_S, c=fc,
                      edgecolors=_EDGE_C, linewidths=_MARKER_LW,
                      marker="o", zorder=10)
        if jersey_numbers and idx in jersey_numbers:
            ax.text(_px(x), _py(y), jersey_numbers[idx],
                    color=tc, fontsize=7.5, fontweight="bold",
                    ha="center", va="center", zorder=11)


def _draw_ball(
    ax: plt.Axes,
    pitch: mplsoccer.Pitch,
    ball_xy: Optional[np.ndarray],
) -> None:
    if ball_xy is None or np.isnan(ball_xy).any():
        return
    pitch.scatter(_px(float(ball_xy[0])), _py(float(ball_xy[1])), ax=ax,
                  s=120, c=_BALL_FC, edgecolors="#333333",
                  linewidths=1.5, marker="o", zorder=12)


def _draw_annotations(
    ax: plt.Axes,
    block_state: Optional[BlockState],
    frame_id: int,
    timestamp: float,
) -> None:
    if block_state is None:
        ax.set_title(f"Frame {frame_id}  |  t = {timestamp:.1f}s  |  no state",
                     fontsize=_TITLE_FS, color="white", pad=8)
        return

    ax.set_title(
        f"{block_state.height.upper()} BLOCK — {block_state.length.upper()}",
        fontsize=_TITLE_FS, fontweight="bold", color=_ZONE_COLOUR, pad=8,
    )
    ax.annotate(
        f"Frame {frame_id}  |  t = {timestamp:.1f}s  |  "
        f"Median X = {block_state.median_x:.1f}m  |  IQR X = {block_state.iqr_x:.1f}m",
        xy=(0.5, -0.02), xycoords="axes fraction",
        ha="center", va="top",
        fontsize=_SUBTITLE_FS, color="#AAAAAA",
    )


# ---------------------------------------------------------------------------
# Public API: static frame
# ---------------------------------------------------------------------------

def plot_frame(
    positions:          np.ndarray,
    team_indices:       dict[str, list[int]],
    goalkeeper_indices: list[int],
    config:             DefensiveBlockConfig,
    block_state:        Optional[BlockState] = None,
    alpha_shape:        Optional[np.ndarray] = None,
    ball_xy:            Optional[np.ndarray] = None,
    frame_id:           int   = 0,
    timestamp:          float = 0.0,
    jersey_numbers:     Optional[dict[int, str]] = None,
    pitch_length:       float = 105.0,
    pitch_width:        float =  68.0,
    figsize:            tuple = (13, 8),
    save_path:          Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Render a single frame.

    Args:
        positions: (N_players, 2) array in model coords; NaN for absent players.
        team_indices: {"defending": [slot_idx, …], "attacking": [slot_idx, …]}.
        goalkeeper_indices: GK slot indices for both teams.
        config: Pipeline configuration (used for zone boundaries).
        block_state: Detected state; if None, zones are not drawn.
        alpha_shape: (n+1, 2) closed polygon in model coords; None to skip.
        ball_xy: (2,) ball position in model coords; None to skip.
        frame_id: Frame index (display only).
        timestamp: Elapsed seconds (display only).
        jersey_numbers: {slot_idx: label} drawn inside markers.
        pitch_length / pitch_width: Pitch dimensions in metres.
        figsize: Matplotlib figure size.
        save_path: If given, save figure to this path at 150 dpi.

    Returns:
        (fig, ax)
    """
    pitch = _create_pitch(pitch_length, pitch_width)
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(_FIG_FC)
    ax.set_facecolor(_FIG_FC)

    if block_state is not None:
        _draw_block_zones(ax, config, block_state.height)
        if alpha_shape is not None:
            _draw_alpha_shape(ax, alpha_shape, block_state.height)

    _draw_ball(ax, pitch, ball_xy)
    _draw_players(ax, pitch, positions, team_indices, goalkeeper_indices, jersey_numbers)
    _draw_annotations(ax, block_state, frame_id, timestamp)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig, ax


# ---------------------------------------------------------------------------
# Public API: sequence animation
# ---------------------------------------------------------------------------

def animate_sequence(
    frames:             list[np.ndarray],
    team_indices:       dict[str, list[int]],
    goalkeeper_indices: list[int],
    config:             DefensiveBlockConfig,
    block_states:       Optional[list[Optional[BlockState]]] = None,
    alpha_shapes:       Optional[list[Optional[np.ndarray]]] = None,
    ball_positions:     Optional[list[Optional[np.ndarray]]] = None,
    fps:                int   = 25,
    output_path:        str   = "sequence.mp4",
    frame_step:         int   = 1,
    jersey_numbers:     Optional[dict[int, str]] = None,
    start_frame:        int   = 0,
    pitch_length:       float = 105.0,
    pitch_width:        float =  68.0,
    figsize:            tuple = (13, 8),
) -> None:
    """Render an animated MP4 or GIF of a defensive sequence.

    Args:
        frames: List of (N_players, 2) position arrays in model coords.
        team_indices / goalkeeper_indices / config: same as plot_frame.
        block_states: Per-frame BlockState (same length as frames); optional.
        alpha_shapes: Per-frame polygon (same length as frames); optional.
        ball_positions: Per-frame (2,) ball xy in model coords; optional.
        fps: Output frame rate.
        output_path: Destination (.mp4 uses ffmpeg, .gif uses pillow).
        frame_step: Sub-sample every N-th frame to reduce file size.
        start_frame: Absolute index of frames[0], used for display only.
        pitch_length / pitch_width: Pitch dimensions in metres.
        figsize: Figure size.
    """
    idx = list(range(0, len(frames), frame_step))

    s_frames = [frames[i] for i in idx]
    s_states = [block_states[i]   if block_states   else None for i in idx]
    s_shapes = [alpha_shapes[i]   if alpha_shapes   else None for i in idx]
    s_balls  = [ball_positions[i] if ball_positions else None for i in idx]

    pitch = _create_pitch(pitch_length, pitch_width)
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(_FIG_FC)

    def _update(i: int) -> None:
        ax.clear()
        pitch.draw(ax=ax)
        ax.set_facecolor(_FIG_FC)

        state = s_states[i]
        if state is not None:
            _draw_block_zones(ax, config, state.height)
            if s_shapes[i] is not None:
                _draw_alpha_shape(ax, s_shapes[i], state.height)

        _draw_ball(ax, pitch, s_balls[i])
        _draw_players(ax, pitch, s_frames[i], team_indices,
                      goalkeeper_indices, jersey_numbers)
        _draw_annotations(ax, state, start_frame + idx[i],
                          (start_frame + idx[i]) / fps)

    anim = FuncAnimation(fig, _update, frames=len(s_frames),
                         interval=1000 / fps, blit=False)

    output_path = Path(output_path)
    if output_path.suffix.lower() == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=2000,
                              extra_args=["-vcodec", "libx264",
                                          "-pix_fmt", "yuv420p"])

    anim.save(str(output_path), writer=writer, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Match timeline
# ---------------------------------------------------------------------------

_STATE_COLOURS = {
    "low_compact":  "#3B82F6",
    "low_long":     "#93C5FD",
    "mid_compact":  "#22C55E",
    "mid_long":     "#86EFAC",
    "high_compact": "#F4B942",
    "high_long":    "#FDE68A",
}
_SEQ_BAND_ALPHA   = 0.08
_STATE_SHADE_ALPHA = 0.30


def _shade_by_state(
    ax: plt.Axes,
    combined_states: list[Optional[str]],
    x_values: np.ndarray,
    frame_mask: Optional[np.ndarray] = None,
) -> None:
    n = len(combined_states)
    i = 0
    while i < n:
        if frame_mask is not None and not frame_mask[i]:
            i += 1
            continue
        state = combined_states[i]
        j = i + 1
        while j < n and combined_states[j] == state and (frame_mask is None or frame_mask[j]):
            j += 1
        if state is not None and state in _STATE_COLOURS:
            ax.axvspan(x_values[i], x_values[min(j, n - 1)],
                       color=_STATE_COLOURS[state], alpha=_STATE_SHADE_ALPHA,
                       linewidth=0, zorder=0)
        i = j


def plot_match_timeline(
    frame_results: FrameResults,
    sequences:     list[DefensiveSequence],
    fps:           float = 25.0,
    x_axis:        str   = "minutes",
    figsize:       tuple = (18, 6),
    title:         str   = "",
    save_path:     Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Time-series plot of block metrics and state classification for a match.

    Background is colour-coded by the combined block state during defensive
    sequences. Metric lines show smoothed median_x (depth) and IQR_x (length).

    Args:
        frame_results: Output of BlockDetectionPipeline.run().
        sequences: From the same pipeline run.
        fps: Frames per second — used to convert frame indices to minutes.
        x_axis: "minutes" (default) or "frames".
        figsize / title / save_path: Standard matplotlib args.

    Returns:
        (fig, ax_left, ax_right)
    """
    n_frames  = len(frame_results.smoothed_median_x)
    frame_idx = np.arange(n_frames)

    if x_axis == "minutes":
        x_values = frame_idx / (fps * 60.0)
        x_label  = "Time (minutes)"
    else:
        x_values = frame_idx.astype(float)
        x_label  = "Frame"

    def_mask = np.zeros(n_frames, dtype=bool)
    for seq in sequences:
        def_mask[seq.start_frame:seq.end_frame] = True

    med_x_def = frame_results.smoothed_median_x.copy()
    iqr_x_def = frame_results.smoothed_iqr_x.copy()
    med_x_def[~def_mask] = np.nan
    iqr_x_def[~def_mask] = np.nan

    fig, ax1 = plt.subplots(figsize=figsize, facecolor="#0D1B2A")
    ax1.set_facecolor("#0D1B2A")
    ax2 = ax1.twinx()

    _shade_by_state(ax1, frame_results.combined_states, x_values, frame_mask=def_mask)

    for seq in sequences:
        ax1.axvspan(x_values[seq.start_frame],
                    x_values[min(seq.end_frame, n_frames - 1)],
                    color="#FFFFFF", alpha=_SEQ_BAND_ALPHA, linewidth=0, zorder=1)

    ax1.plot(x_values, med_x_def, color="#FFFFFF", linewidth=1.0, alpha=0.9,
             label="Median X (depth)", zorder=4)
    ax2.plot(x_values, iqr_x_def, color="#F4B942", linewidth=1.0, alpha=0.85,
             label="IQR X (length)", zorder=4)

    cfg = DefensiveBlockConfig()
    for boundary, label in [(cfg.low_mid_boundary, "low | mid"),
                             (cfg.mid_high_boundary, "mid | high")]:
        ax1.axhline(boundary, color="#FFFFFF", linewidth=0.7,
                    linestyle="--", alpha=0.35, zorder=3)
        ax1.text(x_values[-1] * 0.01, boundary + 0.5, label,
                 color="#AAAAAA", fontsize=7.5, va="bottom", zorder=5)

    ax2.axhline(cfg.compact_long_boundary, color="#F4B942", linewidth=0.7,
                linestyle="--", alpha=0.35, zorder=3)
    ax2.text(x_values[-1] * 0.99, cfg.compact_long_boundary + 0.5,
             "compact | long", color="#F4B942", fontsize=7.5,
             ha="right", va="bottom", alpha=0.7, zorder=5)

    ax1.set_xlabel(x_label, color="#AAAAAA", fontsize=10)
    ax1.set_ylabel("Median X (m)", color="#FFFFFF", fontsize=10)
    ax2.set_ylabel("IQR X (m)", color="#F4B942", fontsize=10)
    ax1.tick_params(colors="#AAAAAA")
    ax2.tick_params(colors="#F4B942")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333333")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")
    ax1.set_xlim(x_values[0], x_values[-1])

    if title:
        ax1.set_title(title, color="#FFFFFF", fontsize=12, pad=10)

    state_patches = [
        mpatches.Patch(color=c, alpha=_STATE_SHADE_ALPHA + 0.3, label=lbl)
        for lbl, c in _STATE_COLOURS.items()
    ]
    metric_handles = [
        Line2D([0], [0], color="#FFFFFF", linewidth=1.5, label="Median X (depth)"),
        Line2D([0], [0], color="#F4B942", linewidth=1.5, label="IQR X (length)"),
        mpatches.Patch(color="#FFFFFF", alpha=_SEQ_BAND_ALPHA + 0.15,
                       label="Defensive sequence"),
    ]
    ax1.legend(handles=metric_handles + state_patches, loc="upper left",
               fontsize=7.5, framealpha=0.25, facecolor="#0D1B2A",
               edgecolor="#444444", labelcolor="#FFFFFF", ncol=3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig, ax1, ax2
