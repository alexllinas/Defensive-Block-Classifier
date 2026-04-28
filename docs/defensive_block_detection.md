# Defensive Block Detection System — Implementation Specification

**Version:** 1.1  
**Status:** Released  
**Scope:** Frame-level and sequence-level classification of the defensive block state from tracking data.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Coordinate System and Conventions](#2-coordinate-system-and-conventions)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Robust Metric Computation](#4-robust-metric-computation)
5. [Temporal Smoothing](#5-temporal-smoothing)
6. [Hysteresis Classification](#6-hysteresis-classification)
7. [Sequence-Level Analysis](#7-sequence-level-analysis)
8. [Match-Level Analysis](#8-match-level-analysis)
9. [Configurable Parameters](#9-configurable-parameters)
10. [Code Architecture](#10-code-architecture)
11. [Testing and Validation](#11-testing-and-validation)
12. [Visualization](#12-visualization)
13. [Future Extensions](#13-future-extensions)

---

## 1. System Overview

### 1.1 Purpose

This system classifies the **defensive block state** of a football team on a frame-by-frame basis, using raw player tracking data. The output characterises both *where* the team defends (block height) and *how stretched* the team is vertically (block length/compactness). This enables objective, repeatable quantification of defensive shape throughout a match.

### 1.2 Two Independent Dimensions

| Dimension | Question answered | States |
|---|---|---|
| **Block height** | How high up the pitch does the team defend? | `low`, `mid`, `high` |
| **Block length** | How vertically compact is the defensive shape? | `compact`, `long` |

Each frame receives a **combined label** from the Cartesian product of both dimensions:

```
{low, mid, high} × {compact, long}
→ low_compact, low_long, mid_compact, mid_long, high_compact, high_long
```

### 1.3 Conceptual Pipeline

```
Tracking data (any format)
    │
    ▼  io/ loader  (MetricaLoader or custom BaseLoader)
TrackingInput
    │   positions         (N_frames, N_players, 2)
    │   in_field          (N_frames, N_players)
    │   team_direction    (2, N_frames)
    │   possession_team   (N_frames,)
    │   in_play           (N_frames,)
    │   player_team       (N_players,)
    │   player_is_gk      (N_players,)
    │
    ▼
[1] Identify defending team slots (via player_team, player_is_gk)
    │
    ▼
[2] Normalize attack direction  (flip X so defender always faces x = -52.5)
    │   — derived frame-by-frame from team_direction
    │
    ▼
[3] Filter: outfield defenders only, mask via in_field
    │
    ▼
[5] Per-frame metrics
    │   ├── median_x   (block height proxy)
    │   └── iqr_x      (block length proxy)
    │
    ▼
[6] Temporal smoothing  (rolling mean over W frames)
    │
    ▼
[7] Hysteresis state machine  (frame-by-frame classification)
    │   ├── height_state  ∈ {low, mid, high}
    │   └── length_state  ∈ {compact, long}
    │
    ▼
[8] Sequence-level aggregation
    │   ├── dominant state, state distribution
    │   ├── transition count, mean metrics
    │   └── per-sequence summary record
    │
    ▼
[9] Match-level aggregation
        ├── distribution of block types across sequences
        ├── weighted mean metrics
        └── match summary record
```

---

## 2. Coordinate System and Conventions

### 2.1 Field Geometry

| Axis | Dimension | Range | Direction |
|---|---|---|---|
| **X** | Longitudinal (depth) | −52.5 m to +52.5 m | Left goal line → Right goal line |
| **Y** | Lateral (width) | −34 m to +34 m | Bottom touchline → Top touchline |
| **Origin** | Field center | (0, 0) | — |

All block-height and block-length metrics are computed **exclusively on the X axis**. The Y axis is not used in this version (see §13 for future horizontal compactness extension).

### 2.2 Attack Direction Normalization

Because teams switch ends at half-time, the raw X values of a defending team depend on which half of the field they occupy. To keep all thresholds frame-direction-agnostic, the system normalizes so that:

> **After normalization, the defending team always defends toward `x = −52.5`.**

This means:
- If the defending team attacks toward `x = +52.5` (i.e., their own goal is at `x = −52.5`), **no flip is needed** — X values are already in the canonical orientation.
- If the defending team attacks toward `x = −52.5` (i.e., their own goal is at `x = +52.5`), **flip X**: multiply all defending-team X values by −1.

After normalization, the semantic interpretation of X for the defending team is:

| Normalized X | Meaning |
|---|---|
| ≈ −52.5 | Own goal line (deepest defensive position) |
| ≈ −30 to −15 | Own defensive third |
| ≈ −15 to +5 | Middle third |
| ≈ +5 to +52.5 | Attacking half / pressing high |

**Important:** The attack direction changes between halves (and potentially in extra time). This is handled automatically frame-by-frame via `team_direction` — no per-period configuration is required from the caller.

### 2.3 Player Slot Convention

The tracking data holds positions for **N_players slots** (starters + substitutes for both teams). The slot count depends on the format and loader.

Slot-to-team and slot-to-role assignments are static arrays inside `TrackingInput`:

| Field | Shape | Description |
|---|---|---|
| `player_team` | (N_players,) int | Team index (0 or 1) per slot |
| `player_is_gk` | (N_players,) bool | True for goalkeeper slots |

**Goalkeeper identification:** use `player_is_gk` directly — there is no fixed GK slot index.

**Substitutes:** a slot that has not yet entered or has already left the pitch has `in_field[frame, slot] == False` and `NaN` in `positions`. The `in_field` mask handles activation timing implicitly — no additional frame-range metadata is needed.

### 2.4 Empty Slots and Absent Players

A player is absent from a given frame when `in_field[frame, slot] == False`. In that case, `positions[frame, slot, :]` contains `NaN`. There are **no `(0.0, 0.0)` sentinel values** — `in_field` is the authoritative presence mask and should always be used directly rather than inspecting position values.

---

## 3. Data Preprocessing

### 3.1 Input Data (TrackingInput)

All loaders produce a `TrackingInput` struct that the pipeline consumes. Format-specific I/O and schema mapping live entirely in the `io/` loaders and are opaque to the pipeline.

| Field | Shape | Description |
|---|---|---|
| `positions` | `(N_frames, N_players, 2)` | Player XY positions in metres |
| `in_field` | `(N_frames, N_players)` bool | True if the player is on the pitch in that frame |
| `team_direction` | `(2, N_frames)` float | Directional sign per team per frame: `−1.0` or `+1.0` |
| `possession_team` | `(N_frames,)` int | Team index (0 or 1) in possession; `−1` when not in play. **Never `−1` during live play.** Primary signal for sequence segmentation. |
| `in_play` | `(N_frames,)` bool | True during live play, False during dead ball |
| `segment` | `(N_frames,)` int | Half index: `0` = first half, `1` = second half |
| `player_team` | `(N_players,)` int | Team index (0 or 1) per slot |
| `player_is_gk` | `(N_players,)` bool | True for goalkeeper slots |
| `pitch_size` | tuple | `(length_m, width_m)` — default `(105.0, 68.0)` |
| `match_id` | str | Unique match identifier |
| `ball_positions` | `(N_frames, 2)` float | Ball XY in metres (optional) |
| `alpha_shape_order` | `(N_frames, N_players)` float | Pre-computed alpha shape vertex order (optional; see §3.10) |

### 3.2 Identifying the Analyzed Team and Its Slots

The system is designed to analyze **one specific team** per run. The caller supplies the `analyzed_team_idx` (0 or 1) identifying which team's defensive block to compute. This is static for the entire match.

**The team defends only when `possession_team != analyzed_team_idx`** and `in_play == True`. This is used for sequence segmentation (§3.8), not for slot selection — slot selection is always based on `analyzed_team_idx`.

```
FUNCTION get_team_slots(ti, analyzed_team_idx):
    # analyzed_team_idx: 0 or 1
    is_analyzed_team = (ti.player_team == analyzed_team_idx)  # (N_players,) bool
    is_gk            = ti.player_is_gk                        # (N_players,) bool

    gk_slots       = WHERE(is_analyzed_team AND is_gk)         # typically 1 slot
    outfield_slots = WHERE(is_analyzed_team AND NOT is_gk)

    RETURN outfield_slots, gk_slots
```

> **Note:** `player_team` and `player_is_gk` are static arrays — they cover all slots including substitutes. `in_field` handles activation timing implicitly.

### 3.3 Reading the Presence Mask

`in_field` is the authoritative per-frame presence flag. Use it directly — do not inspect position values for NaN or sentinel zeros.

```
FUNCTION get_outfield_presence(ti, outfield_slots):
    RETURN ti.in_field[:, outfield_slots]   # (N_frames, n_outfield) bool
```

### 3.4 Normalizing Attack Direction

`team_direction` has shape `(2, N_frames)` indexed by `[team_idx, frame]`. Its values are:

| Value | Meaning |
|---|---|
| `−1.0` | The team defends toward `x = −52.5` (their goal is at the negative end). **No flip needed** — already in canonical orientation. |
| `+1.0` | The team defends toward `x = +52.5` (their goal is at the positive end). **Flip X** (multiply by −1) to bring into canonical orientation. |

The value is **constant within each segment** (half) but may differ between segments. It is read frame-by-frame so that no half-specific configuration is required.

```
FUNCTION get_flip_mask(ti, analyzed_team_idx):
    # Returns (N_frames,) bool — True where X must be flipped
    direction = ti.team_direction[analyzed_team_idx]   # (N_frames,)
    RETURN direction > 0                               # True → flip X

FUNCTION normalize_direction(positions, flip_mask):
    # positions : (N_frames, n_outfield, 2)
    # flip_mask : (N_frames,) bool
    positions_out = positions.COPY()
    positions_out[flip_mask, :, 0] *= -1.0
    # Y axis is never flipped.
    RETURN positions_out
```

> **Verification:** After normalization, all players of the defending team should have X values concentrated in the negative half of the field during settled defensive phases. A quick sanity check: `median(positions_out[:, :, 0]) < 0`.

### 3.5 Extracting and Cleaning Defender Positions

```
FUNCTION extract_outfield_positions(ti, outfield_slots):
    RETURN ti.positions[:, outfield_slots, :]   # (N_frames, n_outfield, 2)

FUNCTION apply_presence_mask(positions, presence_mask):
    # positions     : (N_frames, n_outfield, 2)
    # presence_mask : (N_frames, n_outfield) bool
    pos_clean = positions.COPY()
    pos_clean[NOT presence_mask] = NaN     # absent players → NaN
    RETURN pos_clean
```

### 3.6 Valid Player Count and Frame Validation

```
FUNCTION compute_valid_count(presence_mask):
    # presence_mask: (N_frames, n_outfield) bool
    RETURN presence_mask.SUM(axis=1)   # (N_frames,) int

# Frames below threshold will have metrics set to NaN.
sufficient_frames = (valid_count >= MIN_OUTFIELD_PLAYERS)
```

Default: `MIN_OUTFIELD_PLAYERS = 4`.

### 3.7 Game State and Sequence Segmentation

**Defensive sequences** are identified directly from `ti.possession_team` and `ti.in_play` — no external possession feed is required. `possession_team` is **never `−1` when `in_play == True`**. A defensive phase for `analyzed_team_idx` is any contiguous run of frames where `in_play == True AND possession_team != analyzed_team_idx`. Dead-ball frames (`in_play == False`) break sequences. See §3.8 for the segmentation pseudocode and §7.1 for sequence structure. The `segment` field (0/1) marks the half.

### 3.8 Defensive Sequence Segmentation

```
FUNCTION segment_defensive_sequences(ti, analyzed_team_idx):
    in_play    = ti.in_play        # (N_frames,) bool
    possession = ti.possession_team  # (N_frames,) int; −1 when not in_play

    # possession == −1 when in_play is False, so the AND short-circuits correctly
    is_defending = in_play AND (possession != analyzed_team_idx)   # (N_frames,) bool

    sequences = []
    in_seq    = False
    seq_start = None

    FOR t IN 0..N_frames:
        IF is_defending[t] AND NOT in_seq:
            seq_start = t
            in_seq    = True
        ELSE IF (NOT is_defending[t]) AND in_seq:
            sequences.APPEND(DefensiveSequence(start_frame=seq_start, end_frame=t))
            in_seq = False

    IF in_seq:   # sequence running until end of file
        sequences.APPEND(DefensiveSequence(start_frame=seq_start, end_frame=N_frames))

    RETURN sequences
```

### 3.9 Full Preprocessing Pipeline (Pseudocode)

```
FUNCTION preprocess(ti, analyzed_team_idx):
    outfield_slots, gk_slots = get_team_slots(ti, analyzed_team_idx)
    positions      = extract_outfield_positions(ti, outfield_slots)
    presence_mask  = get_outfield_presence(ti, outfield_slots)
    flip_mask      = get_flip_mask(ti, analyzed_team_idx)
    positions      = normalize_direction(positions, flip_mask)
    pos_clean      = apply_presence_mask(positions, presence_mask)
    valid_count    = compute_valid_count(presence_mask)
    sequences      = segment_defensive_sequences(ti, analyzed_team_idx)
    RETURN pos_clean, presence_mask, valid_count, flip_mask, sequences
```

### 3.10 Decoding the Pre-computed Alpha Shape

`TrackingInput.alpha_shape_order` is an optional `(N_frames, N_players)` float array. When populated by a loader, it encodes a **pre-computed concave hull (alpha shape)** of each team's outfield players per frame. It requires no external library — the geometry is precomputed by the loader.

**Encoding rules:**
- `NaN` → the player is **not** a hull vertex in this frame (interior point, absent, or wrong team).
- Integer float (0.0, 1.0, 2.0, ...) → the player **is** a hull vertex; the value is its **index in the ordered polygon sequence** (CCW or CW, consistent within a dataset).
- Typically ~7 of ~11 outfield players are vertices (the rest are interior to the shape).

**Decoding pseudocode:**

```
FUNCTION decode_alpha_shape(ti, team_idx, frame):
    order      = ti.alpha_shape_order              # (N_frames, N_players) float
    team_slots = WHERE(ti.player_team == team_idx)

    order_vals = order[frame, team_slots]          # (n_team_slots,) float, NaN for non-vertices
    is_vertex  = NOT IS_NAN(order_vals)

    vertex_slots = team_slots[is_vertex]
    vertex_order = order_vals[is_vertex].ASTYPE(int)

    # Sort slots by their polygon order index
    sorted_idx     = ARGSORT(vertex_order)
    ordered_slots  = vertex_slots[sorted_idx]
    polygon_coords = ti.positions[frame, ordered_slots, :]   # (n_vertices, 2) — raw coords

    # Close the polygon by appending the first point
    polygon_closed = VSTACK(polygon_coords, polygon_coords[0])

    RETURN polygon_closed   # (n_vertices + 1, 2)
```

**Important:** The coordinates are in the **raw (pre-normalization) space**. For visualization in canonical orientation (defender toward x = −52.5), apply the same flip as `normalize_direction()`.

```
FUNCTION decode_alpha_shape_normalized(ti, team_idx, frame, flip_mask):
    polygon = decode_alpha_shape(ti, team_idx, frame)
    IF flip_mask[frame]:
        polygon[:, 0] *= -1.0   # flip X only
    RETURN polygon
```

---

## 4. Robust Metric Computation

### 4.1 Metric Definitions

| Metric | Formula | Dimension it informs |
|---|---|---|
| `median_x` | median of valid X values per frame | Block height |
| `iqr_x` | P75(X) − P25(X) of valid X values per frame | Block length |

Both metrics are computed using only the **valid outfield positions** (NaN-masked) for each frame independently.

### 4.2 Median X (Block Height)

```
FUNCTION compute_median_x(pos_clean, counts, min_players):
    # pos_clean: (N_frames, n_outfield, 2)  with NaN for invalid slots
    x_values = pos_clean[:, :, 0]   # (N_frames, n_outfield)

    median_x = NANMEDIAN(x_values, axis=1)   # (N_frames,)

    # Invalidate frames with too few players
    median_x[counts < min_players] = NaN

    RETURN median_x
```

**Why median over mean?**  
Individual defenders can temporarily push high (pressing) or drop very deep (recovering), creating outliers that would distort the mean. The median remains stable as long as fewer than half the outfield players are in an extreme position.

### 4.3 IQR X (Block Length)

```
FUNCTION compute_iqr_x(pos_clean, counts, min_players):
    x_values = pos_clean[:, :, 0]   # (N_frames, n_outfield)

    p25 = NANPERCENTILE(x_values, 25, axis=1)   # (N_frames,)
    p75 = NANPERCENTILE(x_values, 75, axis=1)   # (N_frames,)

    iqr_x = p75 - p25   # (N_frames,)

    iqr_x[counts < min_players] = NaN

    RETURN iqr_x
```

**Why IQR over range (max − min)?**  
Range is dominated by the most advanced presser and the deepest defender, which may be isolated outliers. IQR covers the central 50% of defenders, giving a robust measure of how compact the core of the defensive shape is.

### 4.4 Combined Metric Array

```
metrics = {
    "median_x": compute_median_x(pos_clean, counts, MIN_OUTFIELD_PLAYERS),
    "iqr_x":    compute_iqr_x(pos_clean, counts, MIN_OUTFIELD_PLAYERS),
}
# Each value: (N_frames,) float64, NaN where insufficient data.
```

---

## 5. Temporal Smoothing

### 5.1 Motivation

Frame-to-frame tracking data can oscillate due to:
- Interpolation artifacts in the tracking pipeline.
- Short-lived position spikes (e.g., a player jumping for a header).
- Rapid in-and-out of frame near boundaries.

A rolling mean over a short window absorbs these fluctuations without introducing significant latency.

### 5.2 Rolling Mean Application

```
FUNCTION smooth_metric(values, window_size):
    # values: (N_frames,) with possible NaN
    # NaN frames are excluded from the window average (nanmean semantics).
    smoothed = ROLLING_NANMEAN(values, window=window_size, center=True)
    # center=True: the smoothed value at frame t uses frames [t - W//2, t + W//2].
    # Frames at the start/end of a sequence where the window is incomplete
    # use the available data (min_periods=1).
    RETURN smoothed
```

**Centering vs. causal smoothing:**
- **Center=True** (default): introduces no lag; appropriate for post-hoc analysis.
- **Center=False** (causal): appropriate if the system must produce labels without future data. Use `center=False` for real-time or streaming scenarios.

### 5.3 Smoothed Metrics

```
smoothed_median_x = smooth_metric(metrics["median_x"], SMOOTHING_WINDOW)
smoothed_iqr_x    = smooth_metric(metrics["iqr_x"],    SMOOTHING_WINDOW)
```

Default `SMOOTHING_WINDOW = 5` (see §9).

---

## 6. Hysteresis Classification

### 6.1 Why Hysteresis?

Without hysteresis, a metric oscillating near a threshold produces rapid state toggling (e.g., "mid → low → mid → low" within a few seconds). This creates noise in the labels and makes downstream analysis unreliable. Hysteresis introduces **inertia**: a state is only exited if the metric moves sufficiently far from the boundary in the new direction.

### 6.2 Height Classification: Three-State Machine

#### 6.2.1 States and Thresholds

| State | Interpretation (after normalization) |
|---|---|
| `low` | Median X ≤ `LOW_MID_BOUNDARY` — deep defensive block |
| `mid` | `LOW_MID_BOUNDARY` < Median X ≤ `MID_HIGH_BOUNDARY` — mid block |
| `high` | Median X > `MID_HIGH_BOUNDARY` — high block / pressing line |

Default thresholds (see §9 for tuning guidance):

| Parameter | Default |
|---|---|
| `LOW_MID_BOUNDARY` | −15.0 m |
| `MID_HIGH_BOUNDARY` | 5.0 m |
| `HEIGHT_HYSTERESIS_BUFFER` | 2.5 m |

#### 6.2.2 Effective Entry and Exit Thresholds

For each boundary, define an **entry threshold** (to move INTO the higher state) and an **exit threshold** (to move OUT OF the current state back to the lower one):

```
# For the LOW ↔ MID boundary:
low_to_mid_entry  = LOW_MID_BOUNDARY  + HEIGHT_HYSTERESIS_BUFFER   # = -12.5m
mid_to_low_exit   = LOW_MID_BOUNDARY  - HEIGHT_HYSTERESIS_BUFFER   # = -17.5m

# For the MID ↔ HIGH boundary:
mid_to_high_entry = MID_HIGH_BOUNDARY + HEIGHT_HYSTERESIS_BUFFER   # = +7.5m
high_to_mid_exit  = MID_HIGH_BOUNDARY - HEIGHT_HYSTERESIS_BUFFER   # = +2.5m
```

#### 6.2.3 State Transition Diagram

```
                    median_x > -12.5m
     ┌─────────────────────────────────────►─────────────────────────────┐
     │                                                                    │
  [LOW]                              [MID]                            [HIGH]
     │                                                                    │
     └────────────────────◄──────────────────────────────────────────────┘
           median_x < -17.5m
                                        ▲              │
                                        │              │
                              median_x > +7.5m    median_x < +2.5m
                                        │              ▼
                                      [MID]        [HIGH]
```

Simplified table form:

| Current state | Condition to transition | New state |
|---|---|---|
| `low` | `median_x > low_to_mid_entry` | `mid` |
| `mid` | `median_x < mid_to_low_exit` | `low` |
| `mid` | `median_x > mid_to_high_entry` | `high` |
| `high` | `median_x < high_to_mid_exit` | `mid` |
| `low` | *(cannot jump directly to `high`)* | — |
| `high` | *(cannot jump directly to `low`)* | — |

> Direct `low ↔ high` transitions are **not permitted**. The state must pass through `mid`. This prevents single-frame spikes from causing unrealistic jumps.

#### 6.2.4 Height Classification Pseudocode

```
FUNCTION classify_height_with_hysteresis(smoothed_median_x, initial_state="mid"):
    height_states = ARRAY of length N_frames, dtype=str, initialized to None

    current_state = initial_state

    FOR t IN 0..N_frames:
        v = smoothed_median_x[t]

        IF v IS NaN:
            height_states[t] = None   # or carry forward previous state
            CONTINUE

        IF current_state == "low":
            IF v > low_to_mid_entry:
                current_state = "mid"

        ELSE IF current_state == "mid":
            IF v < mid_to_low_exit:
                current_state = "low"
            ELSE IF v > mid_to_high_entry:
                current_state = "high"

        ELSE IF current_state == "high":
            IF v < high_to_mid_exit:
                current_state = "mid"

        height_states[t] = current_state

    RETURN height_states
```

### 6.3 Length Classification: Two-State Machine

#### 6.3.1 States and Threshold

| State | Interpretation |
|---|---|
| `compact` | IQR X ≤ `COMPACT_LONG_BOUNDARY` — lines are close together |
| `long` | IQR X > `COMPACT_LONG_BOUNDARY` — team is vertically stretched |

Default threshold:

| Parameter | Default |
|---|---|
| `COMPACT_LONG_BOUNDARY` | 18.0 m |
| `LENGTH_HYSTERESIS_BUFFER` | 1.5 m |

Effective thresholds:

```
compact_to_long_entry = COMPACT_LONG_BOUNDARY + LENGTH_HYSTERESIS_BUFFER  # = 19.5m
long_to_compact_exit  = COMPACT_LONG_BOUNDARY - LENGTH_HYSTERESIS_BUFFER  # = 16.5m
```

#### 6.3.2 State Transition Diagram

```
                    iqr_x > 19.5m
          ┌──────────────────────────►──────────────────┐
          │                                              │
      [COMPACT]                                       [LONG]
          │                                              │
          └──────────────────────────◄──────────────────┘
                    iqr_x < 16.5m
```

| Current state | Condition | New state |
|---|---|---|
| `compact` | `iqr_x > compact_to_long_entry` | `long` |
| `long` | `iqr_x < long_to_compact_exit` | `compact` |

#### 6.3.3 Length Classification Pseudocode

```
FUNCTION classify_length_with_hysteresis(smoothed_iqr_x, initial_state="compact"):
    length_states = ARRAY of length N_frames, dtype=str, initialized to None

    current_state = initial_state

    FOR t IN 0..N_frames:
        v = smoothed_iqr_x[t]

        IF v IS NaN:
            length_states[t] = None
            CONTINUE

        IF current_state == "compact":
            IF v > compact_to_long_entry:
                current_state = "long"

        ELSE IF current_state == "long":
            IF v < long_to_compact_exit:
                current_state = "compact"

        length_states[t] = current_state

    RETURN length_states
```

### 6.4 Combined Label

```
FUNCTION combine_states(height_states, length_states):
    combined = ARRAY of length N_frames, dtype=str

    FOR t IN 0..N_frames:
        h = height_states[t]
        l = length_states[t]

        IF h IS None OR l IS None:
            combined[t] = None
        ELSE:
            combined[t] = h + "_" + l
            # e.g., "mid_compact", "low_long", "high_compact"

    RETURN combined
```

### 6.5 NaN Handling Strategy

Two strategies are supported (configurable):

| Strategy | Description |
|---|---|
| `"none"` | Frames with insufficient data produce `None` labels; no interpolation. |
| `"forward_fill"` | Carry the last valid state forward into NaN frames. Suitable for short gaps (< N frames) in the middle of a sequence. |

Default: `"none"`. Forward-fill should only be used if gaps are known to be data artifacts (e.g., a player momentarily out of frame) rather than genuine absences.

---

## 7. Sequence-Level Analysis

### 7.1 Definition of a Defensive Sequence

A **defensive sequence** is a contiguous run of `in_play` frames during which `last_touch_team_id != analyzed_team_idx` — i.e., the analyzed team does not have possession. Sequences are produced by `segment_defensive_sequences()` (§3.8) as part of the preprocessing pipeline; no external possession data is needed.

Out-of-play frames (`state.startswith("out_of_play")`) break sequences: a dead-ball event ends the current defensive sequence. Any immediately following in-play period where the opponent still has possession starts a new sequence.

**Sequence data structure:**

```
STRUCT DefensiveSequence:
    match_id        : str
    segment         : int       # 0 = first half, 1 = second half (from ds.coords["segment"])
    start_frame     : int       # inclusive, integer frame index
    end_frame       : int       # exclusive, integer frame index
    sequence_id     : str       # unique identifier (e.g. "{match_id}_{start_frame}")
```

### 7.2 Frame-Level Labels Within a Sequence

```
FUNCTION process_sequence(sequence, height_states, length_states, combined_states,
                           smoothed_median_x, smoothed_iqr_x):

    s = sequence.start_frame
    e = sequence.end_frame

    height_seq   = height_states[s:e]
    length_seq   = length_states[s:e]
    combined_seq = combined_states[s:e]
    median_x_seq = smoothed_median_x[s:e]
    iqr_x_seq    = smoothed_iqr_x[s:e]

    RETURN SequenceFrameData(
        height_states   = height_seq,
        length_states   = length_seq,
        combined_states = combined_seq,
        median_x        = median_x_seq,
        iqr_x           = iqr_x_seq,
    )
```

### 7.3 Sequence Aggregation

```
FUNCTION aggregate_sequence(frame_data, sequence):
    valid_height   = FILTER_NOT_NONE(frame_data.height_states)
    valid_length   = FILTER_NOT_NONE(frame_data.length_states)
    valid_combined = FILTER_NOT_NONE(frame_data.combined_states)
    valid_median_x = NANVALID(frame_data.median_x)
    valid_iqr_x    = NANVALID(frame_data.iqr_x)

    n_total_frames = sequence.end_frame - sequence.start_frame
    n_valid_frames = LENGTH(valid_combined)

    # --- Dominant state ---
    dominant_combined = MODE(valid_combined)    # most frequent combined label
    dominant_height   = MODE(valid_height)
    dominant_length   = MODE(valid_length)

    # --- State distributions ---
    height_dist   = COUNT_VALUES(valid_height)   / n_valid_frames
    length_dist   = COUNT_VALUES(valid_length)   / n_valid_frames
    combined_dist = COUNT_VALUES(valid_combined) / n_valid_frames
    # e.g., {"low": 0.3, "mid": 0.6, "high": 0.1}

    # --- Transition counts ---
    height_transitions   = COUNT_TRANSITIONS(valid_height)
    length_transitions   = COUNT_TRANSITIONS(valid_length)
    combined_transitions = COUNT_TRANSITIONS(valid_combined)
    # A transition = consecutive frames where state changes.

    # --- Metric summaries ---
    mean_median_x  = NANMEAN(valid_median_x)
    mean_iqr_x     = NANMEAN(valid_iqr_x)
    std_median_x   = NANSTD(valid_median_x)
    std_iqr_x      = NANSTD(valid_iqr_x)

    RETURN SequenceSummary(
        sequence_id          = sequence.sequence_id,
        match_id             = sequence.match_id,
        segment              = sequence.segment,
        n_total_frames       = n_total_frames,
        n_valid_frames       = n_valid_frames,
        coverage_ratio       = n_valid_frames / n_total_frames,

        dominant_combined    = dominant_combined,
        dominant_height      = dominant_height,
        dominant_length      = dominant_length,

        height_dist          = height_dist,
        length_dist          = length_dist,
        combined_dist        = combined_dist,

        height_transitions   = height_transitions,
        length_transitions   = length_transitions,
        combined_transitions = combined_transitions,

        mean_median_x        = mean_median_x,
        std_median_x         = std_median_x,
        mean_iqr_x           = mean_iqr_x,
        std_iqr_x            = std_iqr_x,
    )
```

### 7.4 Counting Transitions

```
FUNCTION count_transitions(states):
    count = 0
    FOR i IN 1..LENGTH(states):
        IF states[i] != states[i-1]:
            count += 1
    RETURN count
```

A high transition count within a sequence indicates instability of the defensive shape during that attacking phase. This can be a signal of disorganization or of a team actively shifting between defensive modes.

### 7.5 Coverage Ratio

The **coverage ratio** is the fraction of frames in the sequence that produced a valid label. If it falls below a configurable threshold (default `MIN_SEQUENCE_COVERAGE = 0.5`), the sequence summary should be flagged as unreliable and excluded from match-level aggregation.

---

## 8. Match-Level Analysis

### 8.1 Inputs

A collection of `SequenceSummary` objects, one per defensive sequence in the match. Sequences flagged as low-coverage are excluded.

### 8.2 Match Aggregation

```
FUNCTION aggregate_match(sequence_summaries, weights="duration"):
    # weights: how to weight each sequence's contribution.
    #   "duration"  → weight by n_valid_frames (default)
    #   "uniform"   → all sequences weighted equally

    IF weights == "duration":
        w = [s.n_valid_frames FOR s IN sequence_summaries]
    ELSE:
        w = [1 FOR s IN sequence_summaries]
    w_total = SUM(w)

    # --- Weighted distributions ---
    combined_labels = {low_compact, low_long, mid_compact, mid_long, high_compact, high_long}
    match_combined_dist = {}
    FOR label IN combined_labels:
        match_combined_dist[label] = SUM(
            w[i] * sequence_summaries[i].combined_dist.GET(label, 0.0)
            FOR i IN 0..N_sequences
        ) / w_total

    # Repeat for height_dist and length_dist independently.

    # --- Weighted mean metrics ---
    match_mean_median_x = WEIGHTED_MEAN([s.mean_median_x FOR s IN summaries], w)
    match_mean_iqr_x    = WEIGHTED_MEAN([s.mean_iqr_x    FOR s IN summaries], w)

    # --- Aggregate transition rates ---
    # Normalise by total valid frames to make it comparable across matches.
    total_valid = SUM([s.n_valid_frames FOR s IN summaries])
    height_transition_rate   = SUM([s.height_transitions   FOR s IN summaries]) / total_valid
    length_transition_rate   = SUM([s.length_transitions   FOR s IN summaries]) / total_valid
    combined_transition_rate = SUM([s.combined_transitions FOR s IN summaries]) / total_valid

    # --- Dominant block type ---
    dominant_combined = ARGMAX(match_combined_dist)
    dominant_height   = ARGMAX(match_height_dist)
    dominant_length   = ARGMAX(match_length_dist)

    # --- Sequence counts by dominant type ---
    sequences_by_type = GROUP_BY(summaries, key=lambda s: s.dominant_combined)
    # e.g., {"mid_compact": 12, "low_compact": 4, ...}

    RETURN MatchSummary(
        match_id                  = ...,
        n_sequences               = LENGTH(summaries),
        total_valid_frames        = total_valid,

        dominant_combined         = dominant_combined,
        dominant_height           = dominant_height,
        dominant_length           = dominant_length,

        match_combined_dist       = match_combined_dist,
        match_height_dist         = match_height_dist,
        match_length_dist         = match_length_dist,

        match_mean_median_x       = match_mean_median_x,
        match_mean_iqr_x          = match_mean_iqr_x,

        height_transition_rate    = height_transition_rate,
        length_transition_rate    = length_transition_rate,
        combined_transition_rate  = combined_transition_rate,

        sequences_by_type         = sequences_by_type,
    )
```

### 8.3 Period-Level Breakdown

The same aggregation should optionally be run **per period** (first half, second half, extra time), since teams often change their defensive line height at half-time. Period-level summaries are sub-records within the match summary.

---

## 9. Configurable Parameters

The following table lists all model parameters, their defaults, valid ranges, and guidance for adjustment.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `MIN_OUTFIELD_PLAYERS` | `4` | int | 1–10 | Minimum valid outfield defenders per frame to compute metrics. Below this threshold, frame is marked NaN. |
| `SMOOTHING_WINDOW` | `5` | int | 1–25 | Rolling mean window (frames). At 25 fps, 5 frames ≈ 0.2s. |
| `SMOOTHING_CENTER` | `True` | bool | — | If True, centered smoothing (post-hoc). If False, causal (real-time). |
| `LOW_MID_BOUNDARY` | `−15.0` | float | −40 to 0 | Median X boundary separating "low" from "mid" block. |
| `MID_HIGH_BOUNDARY` | `5.0` | float | −10 to +30 | Median X boundary separating "mid" from "high" block. |
| `HEIGHT_HYSTERESIS_BUFFER` | `2.5` | float | 0–10 | Half-width of the hysteresis band around each height boundary. |
| `COMPACT_LONG_BOUNDARY` | `18.0` | float | 5–40 | IQR X boundary separating "compact" from "long". |
| `LENGTH_HYSTERESIS_BUFFER` | `1.5` | float | 0–5 | Half-width of the hysteresis band around the length boundary. |
| `HEIGHT_INITIAL_STATE` | `"mid"` | str | `low/mid/high` | Initial state of the height state machine at the start of a period or sequence. |
| `LENGTH_INITIAL_STATE` | `"compact"` | str | `compact/long` | Initial state of the length state machine. |
| `NAN_STRATEGY` | `"none"` | str | `none/forward_fill` | How to handle frames with insufficient data. |
| `MIN_SEQUENCE_COVERAGE` | `0.5` | float | 0–1 | Minimum fraction of valid frames for a sequence to be included in match aggregation. |
| `MATCH_WEIGHT_STRATEGY` | `"duration"` | str | `duration/uniform` | How to weight sequences in match-level aggregation. |

### 9.1 Guidance for Threshold Tuning

**`LOW_MID_BOUNDARY` and `MID_HIGH_BOUNDARY`:**  
These depend on the tactical context and league. A starting point:
- In a typical match, a team defending in their own third has a median X around −20m to −30m (= low block).
- A mid-block team sits around −10m to 0m.
- A high-pressing team often has median X above 0m (in the opponent's half).
Calibrate against manually labelled examples from the same competition.

**`COMPACT_LONG_BOUNDARY`:**  
IQR of 18m means the central 50% of defenders span 18 metres longitudinally. A compact back-four with a compact midfield might show IQR ≈ 10–14m. A stretched team with deep defenders and high forwards might show IQR > 20m. Calibrate by inspecting histograms of IQR values across many matches.

**`HEIGHT_HYSTERESIS_BUFFER`:**  
Set to roughly 15–20% of the distance between thresholds. With thresholds at −15m and +5m (a range of 20m), a buffer of 2.5m (12.5%) is reasonable.

**`SMOOTHING_WINDOW`:**  
At 25 fps, 5 frames ≈ 0.2 seconds. This is short enough to track genuine tactical changes (which happen over seconds) but long enough to absorb single-frame noise. If tracking is at a different frame rate, adjust proportionally.

---

## 10. Code Architecture

### 10.1 Module Structure

```
defensive_block/
├── __init__.py
├── config.py              # DefensiveBlockConfig dataclass
├── data.py                # download_metrica_sample() helper
├── preprocessing.py       # Presence masking, direction normalisation, sequence segmentation
├── metrics.py             # median_x, iqr_x computation
├── smoothing.py           # Rolling mean application
├── classification.py      # Hysteresis state machines
├── aggregation.py         # Sequence and match aggregation
├── pipeline.py            # Orchestration: end-to-end run
├── visualize.py           # Pitch rendering, sequence animation, match timeline (see §12)
├── types.py               # Data classes: TrackingInput, DefensiveSequence,
│                          #   SequenceSummary, MatchSummary, SequenceFrameData, etc.
└── io/
    ├── base.py            # BaseLoader abstract class
    └── metrica.py         # MetricaLoader: reads Metrica Sports open CSV format
```

### 10.2 Class / Module Responsibilities

#### `config.py`

```
DATACLASS DefensiveBlockConfig:
    min_outfield_players    : int   = 4
    smoothing_window        : int   = 5
    smoothing_center        : bool  = True
    low_mid_boundary        : float = -15.0
    mid_high_boundary       : float = 5.0
    height_hysteresis       : float = 2.5
    compact_long_boundary   : float = 18.0
    length_hysteresis       : float = 1.5
    height_initial_state    : str   = "mid"
    length_initial_state    : str   = "compact"
    nan_strategy            : str   = "none"
    min_sequence_coverage   : float = 0.5
    match_weight_strategy   : str   = "duration"
```

#### `preprocessing.py`

All functions receive a `TrackingInput` struct (produced by any `BaseLoader` subclass) rather than a raw `xr.Dataset`. The format-specific loading and alpha-shape decoding live in the `io/` loaders.

- `get_team_slots(ti, analyzed_team_idx) → (outfield_slots, gk_slots)`
- `get_flip_mask(ti, analyzed_team_idx) → ndarray[bool]`  — from `ti.team_direction`
- `get_outfield_presence(ti, outfield_slots) → ndarray[bool]`  — from `ti.in_field`
- `extract_outfield_positions(ti, outfield_slots) → ndarray`
- `normalize_direction(positions, flip_mask) → ndarray`
- `apply_presence_mask(positions, presence_mask) → ndarray`
- `compute_valid_count(presence_mask) → ndarray[int]`
- `segment_defensive_sequences(ti, analyzed_team_idx) → List[DefensiveSequence]`  — from `ti.possession_team` + `ti.in_play`
- `preprocess(ti, analyzed_team_idx) → (pos_clean, presence_mask, valid_count, flip_mask, sequences)`

#### `metrics.py`

- `compute_median_x(pos_clean, counts, min_players) → ndarray`
- `compute_iqr_x(pos_clean, counts, min_players) → ndarray`

#### `smoothing.py`

- `smooth_metric(values, window_size, center) → ndarray`

#### `classification.py`

- `classify_height(smoothed_median_x, config) → List[str|None]`
- `classify_length(smoothed_iqr_x, config) → List[str|None]`
- `combine_states(height_states, length_states) → List[str|None]`

#### `aggregation.py`

- `aggregate_sequence(seq, frame_data) → SequenceSummary`
- `aggregate_match(match_id, summaries, config) → MatchSummary`

#### `pipeline.py`

```
CLASS BlockDetectionPipeline:

    __init__(self, config: DefensiveBlockConfig = None):
        self.config = config or DefensiveBlockConfig()

    run(self, ti: TrackingInput, analyzed_team_idx) → (MatchSummary, FrameResults, List[DefensiveSequence]):
        # ti: TrackingInput produced by any BaseLoader subclass (MetricaLoader or custom)
        # analyzed_team_idx: 0 or 1 — the team whose defensive block we compute.
        # Returns FrameResults alongside MatchSummary so callers (e.g. visualize.py) can
        # access per-frame states without re-running the pipeline.
        pos_clean, _, valid_count, _, sequences = preprocess(ti, analyzed_team_idx)

        median_x = compute_median_x(pos_clean, valid_count, self.config.min_outfield_players)
        iqr_x    = compute_iqr_x(pos_clean, valid_count, self.config.min_outfield_players)
        sm_median_x = smooth_metric(median_x, self.config.smoothing_window, self.config.smoothing_center)
        sm_iqr_x    = smooth_metric(iqr_x,    self.config.smoothing_window, self.config.smoothing_center)
        height_states   = classify_height(sm_median_x, self.config)
        length_states   = classify_length(sm_iqr_x,    self.config)
        combined_states = combine_states(height_states, length_states)
        frame_results   = FrameResults(sm_median_x, sm_iqr_x, height_states, length_states, combined_states)

        all_summaries = []
        FOR seq IN sequences:
            s, e = seq.start_frame, seq.end_frame
            frame_data = SequenceFrameData(
                height_states   = height_states[s:e],
                length_states   = length_states[s:e],
                combined_states = combined_states[s:e],
                median_x        = sm_median_x[s:e],
                iqr_x           = sm_iqr_x[s:e],
            )
            all_summaries.APPEND(aggregate_sequence(seq, frame_data))

        match_summary = aggregate_match(ti.match_id, all_summaries, self.config)
        RETURN match_summary, frame_results, sequences
```

#### `visualize.py`

- `_create_pitch(pitch_length, pitch_width) → mplsoccer.Pitch`  — custom-dims pitch, no coordinate transform needed.
- `_draw_block_zones(ax, config, active_height)` — coloured rectangles for low/mid/high zones.
- `_draw_alpha_shape(ax, polygon_coords, active_height)` — filled concave hull polygon, coloured by zone, drawn behind player dots.
- `_draw_players(ax, pitch, positions, team_indices, gk_indices, jersey_numbers)` — scatter with team/role differentiation.
- `_draw_ball(ax, pitch, ball_xy)` — renders the ball position when available.
- `_draw_annotations(ax, block_state, frame_id, timestamp)` — title + subtitle overlay.
- `plot_frame(...) → (fig, ax)` — single static frame render.
- `animate_sequence(...) → None` — full sequence animation via `FuncAnimation`.
- `plot_match_timeline(frame_results, sequences, ...) → (fig, ax_top, ax_bot)` — full-match time-series of block metrics and states.

See §12 for the full specification of this module.

### 10.3 Module Dependency Diagram

```
io/metrica.py (and custom BaseLoader subclasses)  →  produce TrackingInput from format-specific files
    └── io/base.py  →  BaseLoader abstract class

pipeline.py
    ├── preprocessing.py  →  (mask, normalize, segment_sequences)  — consumes TrackingInput
    ├── metrics.py        →  (median_x, iqr_x)
    ├── smoothing.py      →  (rolling mean)
    ├── classification.py →  (hysteresis state machines)
    └── aggregation.py    →  (sequence + match summaries)

visualize.py              →  (standalone; reads types.py + config.py)

data.py                   →  (download_metrica_sample utility; no pipeline dependency)

All modules import from:
    config.py             →  (DefensiveBlockConfig)
    types.py              →  (data structures: TrackingInput, DefensiveSequence, …)
```

---

## 11. Testing and Validation

### 11.1 Unit Tests

#### Preprocessing

| Test case | Input | Expected output |
|---|---|---|
| All players in field | `in_field` all True for 10 outfield slots | presence mask all True, counts = 10 |
| One player absent | `in_field[f, p] = False` | slot masked via NaN, counts − 1 |
| Substitute not yet on | slot inactive (`in_field == False` before entering) | NaN masked automatically |
| Direction flip needed | `team_direction[team, f] = +1.0` | X values multiplied by −1 |
| No flip needed | `team_direction[team, f] = −1.0` | X values unchanged |
| Half-time direction change | `team_direction` changes from −1 to +1 at segment boundary | flip_mask transitions correctly frame-by-frame |
| GK identification | `player_is_gk == True` | slot excluded from outfield_slots |

#### Metrics

| Test case | Input | Expected output |
|---|---|---|
| Perfect line (same X for all) | X = [−20, −20, −20, −20] | median = −20, IQR = 0 |
| Symmetric spread | X = [−30, −20, −10, 0] | median = −15, IQR = 15 |
| One outlier presser | X = [−30, −25, −22, −20, +15] | median ≈ −22, IQR robust to outlier |
| Too few players | n_valid < MIN_OUTFIELD_PLAYERS | output = NaN |

#### Classification

| Test case | Input | Expected output |
|---|---|---|
| Stable low block | median_x = −30 throughout | all frames = "low" |
| Clean transition | median_x rises from −30 to 0 | "low" → "mid" at crossing of −12.5m |
| Oscillation below hysteresis | median_x = −13, −16, −13, −16 | state stays "mid" (no flip) |
| Oscillation above hysteresis | median_x = −13, −18, −13, −18 | state flips each time |
| NaN frames | several NaN in middle | NaN handled per strategy |

#### Hysteresis correctness

For each boundary (two height, one length), verify:
- State does not change until the metric crosses `boundary ± buffer`.
- State changes correctly at the entry threshold.
- State does not jump from `low` to `high` directly.

### 11.2 Integration Tests

| Scenario | Description |
|---|---|
| Short sequence (< 10 frames) | Verify aggregation handles tiny n without crashing. |
| Sequence entirely NaN | coverage_ratio = 0; sequence flagged as unreliable. |
| State constant throughout sequence | transitions = 0; dominant = that state. |
| Many transitions | High transition_count; dominant state may have < 50% of frames. |
| Direction flip at half-time | `team_direction` flips at segment boundary; verify X values are correctly inverted without requiring any period-level config. |
| Sequence spanning segment boundary | Allowed — `flip_mask` is per-frame, so frames in each half are normalized independently. |

### 11.3 Edge Cases

| Edge case | Recommended handling |
|---|---|
| Fewer than `MIN_OUTFIELD_PLAYERS` valid for entire sequence | Flag sequence as invalid; exclude from aggregation. |
| Sequence of length 1 frame | Valid; returns single-frame state, transitions = 0. |
| All players in a single X position | IQR = 0 → always `compact`. Expected, not a bug. |
| X values outside [−52.5, +52.5] | Log a warning; clamp to field boundaries or discard frame. |
| `team_direction` contains unexpected value (not ±1) | Log a warning; treat as no-flip and flag frames. |
| Input data with unexpected extra fields | Ignore; only the fields listed in §3.1 are consumed by the pipeline. |
| Frame rate != 25 fps | `SMOOTHING_WINDOW` is in frames; document that tuning is needed for other frame rates. |

### 11.4 Validation Against Ground Truth

For calibration of thresholds:
1. Select a sample of sequences across several matches.
2. Have a tactical analyst manually label each sequence as low/mid/high and compact/long.
3. Compute the system's dominant state per sequence and compare with manual labels.
4. Report accuracy, confusion matrix, and common misclassifications.
5. Adjust thresholds to maximise agreement with analyst labels.

---

## 12. Visualization

### 12.1 Purpose and Scope

`visualize.py` renders the outputs of the detection pipeline onto a football pitch. It supports two modes: a **static frame** for inspection and reporting, and an **animated sequence** (MP4 or GIF) for reviewing a full defensive phase.

The module is **standalone** — it does not depend on any other module in `defensive_block/` at runtime beyond `config.py` and `types.py`. It can be driven directly from pipeline outputs or from pre-saved results.

### 12.2 Pitch Setup

A custom-dims `mplsoccer.Pitch` is constructed with the pitch's actual dimensions. Because the coordinate system is already centered at `(0, 0)` with extents `[−52.5, 52.5] × [−34, 34]`, **no coordinate transformation is needed** — player positions from `TrackingInput` are passed directly to `pitch.scatter()` and `pitch.annotate()`.

```
FUNCTION _create_pitch(pitch_length, pitch_width):
    RETURN mplsoccer.Pitch(
        pitch_type   = "custom",
        pitch_length = pitch_length,
        pitch_width  = pitch_width,
        pitch_color  = "grass",
        line_color   = "white",
        stripe       = True,
        goal_type    = "box",
    )
```

`pitch_length` and `pitch_width` come from `ti.pitch_size` (default `(105.0, 68.0)`).

### 12.3 Block Zone Overlay

Three coloured rectangles span the full pitch width (`y ∈ [−34, 34]`) and are bounded in X by the model thresholds from `DefensiveBlockConfig`:

| Zone | X range | Colour |
|---|---|---|
| **Low** | `[−52.5, low_mid_boundary]` | Red-orange `#E8553E` |
| **Mid** | `[low_mid_boundary, mid_high_boundary]` | Amber `#F4B942` |
| **High** | `[mid_high_boundary, +52.5]` | Green `#5BBF6B` |

The **active zone** (matching the frame's `height_state`) is drawn with `alpha ≈ 0.30`. Inactive zones use `alpha ≈ 0.07`. Zone labels (`LOW BLOCK`, `MID BLOCK`, `HIGH BLOCK`) are centred horizontally in each zone at a fixed Y position near the bottom touchline; the active label renders at full opacity and bold weight.

```
FUNCTION _draw_block_zones(ax, config, active_height):
    zones = [
        ("low",  -52.5,                    config.low_mid_boundary),
        ("mid",  config.low_mid_boundary,  config.mid_high_boundary),
        ("high", config.mid_high_boundary, +52.5),
    ]
    FOR zone_id, x_left, x_right IN zones:
        alpha = 0.30 IF zone_id == active_height ELSE 0.07
        draw Rectangle(x=x_left, y=-34, width=(x_right−x_left), height=68,
                        facecolor=ZONE_COLOURS[zone_id], alpha=alpha)
        draw zone label at (x_center, y=-31.5)
            opacity = 0.90 IF active ELSE 0.30
            fontweight = "bold" IF active ELSE "normal"
```

### 12.4 Alpha Shape Overlay

The pre-computed concave hull of the defending team (stored in `TrackingInput.alpha_shape_order`, see §3.10) is drawn as a filled polygon **behind the player markers**.

```
FUNCTION _draw_alpha_shape(ax, polygon_coords, active_height):
    # polygon_coords: (n_vertices + 1, 2) — last point == first point (closed)
    colour = ZONE_COLOURS[active_height]
    patch  = matplotlib.patches.Polygon(
        polygon_coords[:-1],   # matplotlib closes the polygon automatically
        closed    = True,
        facecolor = colour,
        edgecolor = colour,
        alpha     = 0.20,      # subtle fill
        linewidth = 2.0,
        linestyle = "--",
        zorder    = 3,         # above zone rectangles, below player dots
    )
    ax.add_patch(patch)
```

The alpha shape is drawn using the **same colour as the active block zone** but at lower opacity (`alpha = 0.20` fill, solid/dashed edge at `alpha ≈ 0.70`). This visually ties the shape to the current block state. If `polygon_coords` is `None` or has fewer than 3 vertices, the draw is skipped silently.

### 12.5 Player Rendering

Players are rendered with `pitch.scatter()`. Visual encoding by role:

| Role | Colour | Marker | Edge |
|---|---|---|---|
| Defending outfield | White `#FFFFFF` | Circle `o` | Dark `#1A1A2E` |
| Attacking outfield | Red `#E63946` | Circle `o` | Dark `#1A1A2E` |
| Goalkeeper (either team) | Gold `#FFD700` | Diamond `D` | Dark `#1A1A2E` |

Absent slots (`NaN` positions from `in_field == False`) are skipped.

If jersey numbers are provided (mapping `slot_idx → str`), they are drawn centred inside each marker using `ax.text()`.

```
FUNCTION _draw_players(ax, pitch, positions, team_indices, gk_indices, jersey_numbers):
    gk_set = SET(gk_indices)
    FOR role IN {"defending", "attacking"}:
        FOR idx IN team_indices[role]:
            x, y = positions[idx]
            IF x or y IS NaN: SKIP
            colour, marker = pick_style(idx IN gk_set, role)
            pitch.scatter(x, y, ax=ax, s=250, c=colour, marker=marker, ...)
            IF jersey_numbers has idx:
                ax.text(x, y, jersey_numbers[idx], ...)
```

### 12.6 Frame Annotations

Each frame displays:
- **Title** (above the pitch): `"LOW BLOCK — COMPACT"` in the zone's colour.
- **Subtitle** (below the pitch via `ax.annotate`):
  `"Frame {frame_id} | t = {timestamp:.1f}s | Median X = {median_x:.1f}m | IQR X = {iqr_x:.1f}m"`

### 12.7 Static Frame Output (`plot_frame`)

```
FUNCTION plot_frame(
    positions          : (N_players, 2) ndarray,   # raw positions for all slots; NaN for absent
    team_indices       : {"defending": [int], "attacking": [int]},
    goalkeeper_indices : [int],
    config             : DefensiveBlockConfig,
    block_state        : BlockState       = None,  # None → zones not drawn
    alpha_shape        : (n+1, 2) ndarray = None,  # closed polygon in model coords
    ball_xy            : (2,) ndarray     = None,  # ball position; None to skip
    frame_id           : int   = 0,
    timestamp          : float = 0.0,
    jersey_numbers     : dict  = None,
    pitch_length       : float = 105.0,
    pitch_width        : float = 68.0,
    figsize            : tuple = (13, 8),
    save_path          : str   = None,
) → (fig, ax):

    pitch = _create_pitch(pitch_length, pitch_width)
    fig, ax = pitch.draw(figsize=figsize)
    _draw_block_zones(ax, config, block_state.height)
    IF alpha_shape IS NOT None:
        _draw_alpha_shape(ax, alpha_shape, block_state.height)
    _draw_players(ax, pitch, positions, team_indices, goalkeeper_indices, jersey_numbers)
    _draw_annotations(ax, block_state, frame_id, timestamp)
    IF save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    RETURN fig, ax
```

### 12.7 Sequence Animation (`animate_sequence`)

```
FUNCTION animate_sequence(
    frames             : List[(N_players, 2) ndarray],  # raw positions per frame, all slots
    team_indices       : dict,
    goalkeeper_indices : [int],
    config             : DefensiveBlockConfig,
    block_states       : List[BlockState]               = None,  # one per frame, or None
    alpha_shapes       : List[(n+1, 2) ndarray]         = None,  # one per frame, or None
    ball_positions     : List[(2,) ndarray]             = None,  # one per frame, or None
    fps                : int   = 25,
    output_path        : str   = "sequence.mp4",  # .mp4 or .gif
    frame_step         : int   = 1,
    jersey_numbers     : dict  = None,
    start_frame        : int   = 0,
    pitch_length       : float = 105.0,
    pitch_width        : float = 68.0,
    figsize            : tuple = (13, 8),
) → None:

    sampled_frames  = frames[::frame_step]
    sampled_states  = block_states[::frame_step]
    sampled_shapes  = alpha_shapes[::frame_step] IF alpha_shapes ELSE [None] * N

    pitch = _create_pitch(pitch_length, pitch_width)
    fig, ax = pitch.draw(figsize=figsize)

    FUNCTION update(i):
        ax.clear()
        pitch.draw(ax=ax)
        _draw_block_zones(ax, config, sampled_states[i].height)
        IF sampled_shapes[i] IS NOT None:
            _draw_alpha_shape(ax, sampled_shapes[i], sampled_states[i].height)
        _draw_players(ax, pitch, sampled_frames[i], ...)
        _draw_annotations(ax, sampled_states[i], frame_id=i*frame_step, ...)

    anim = FuncAnimation(fig, update, frames=N, interval=1000/fps)

    IF output_path ends with ".gif": writer = PillowWriter(fps=fps)
    ELSE:                            writer = FFMpegWriter(fps=fps)

    anim.save(output_path, writer=writer, dpi=120)
```

### 12.8 Typical Usage Pattern

```
from defensive_block import BlockDetectionPipeline, DefensiveBlockConfig
from defensive_block import download_metrica_sample
from defensive_block.io.metrica import MetricaLoader
from defensive_block.visualize import plot_frame, animate_sequence

# 1. Load tracking data into a TrackingInput
paths    = download_metrica_sample(game=1)
loader   = MetricaLoader()
ti       = loader.load(**paths, analyzed_team="home")

# 2. Run the full pipeline
config   = DefensiveBlockConfig()
pipeline = BlockDetectionPipeline(config)
match_summary, frame_results, sequences = pipeline.run(ti, analyzed_team_idx=0)

# 3. Build team/GK index dicts for rendering
from defensive_block.preprocessing import get_team_slots
outfield_slots, gk_slots = get_team_slots(ti, analyzed_team_idx=0)
opponent_slots, _        = get_team_slots(ti, analyzed_team_idx=1)
team_indices = {"defending": list(outfield_slots), "attacking": list(opponent_slots)}

pitch_length, pitch_width = ti.pitch_size

# 4. Pick one sequence and extract its raw positions + states
seq           = sequences[0]
s, e          = seq.start_frame, seq.end_frame
raw_positions = ti.positions[s:e]                    # (T, N_players, 2)
ball_positions= ti.ball_positions[s:e] if ti.ball_positions is not None else None
block_states  = [frame_results.block_state_at(f) for f in range(s, e)]

# Decode pre-computed alpha shapes (available when ti.alpha_shape_order is not None)
alpha_shapes = None
if ti.alpha_shape_order is not None:
    alpha_shapes = [
        _decode_alpha_shape(ti, team_idx=0, frame=f)
        for f in range(s, e)
    ]

# 5. Static frame at the midpoint of the sequence
mid = len(raw_positions) // 2
plot_frame(
    positions          = raw_positions[mid],           # (N_players, 2)
    team_indices       = team_indices,
    goalkeeper_indices = list(gk_slots),
    config             = config,
    block_state        = block_states[mid],
    alpha_shape        = alpha_shapes[mid] if alpha_shapes else None,
    ball_xy            = ball_positions[mid] if ball_positions is not None else None,
    frame_id           = s + mid,
    timestamp          = (s + mid) / 25.0,
    pitch_length       = pitch_length,
    pitch_width        = pitch_width,
    save_path          = f"block_{seq.sequence_id}_mid.png",
)

# 6. Full animation
animate_sequence(
    frames             = list(raw_positions),
    team_indices       = team_indices,
    goalkeeper_indices = list(gk_slots),
    config             = config,
    block_states       = block_states,
    alpha_shapes       = alpha_shapes,
    ball_positions     = list(ball_positions) if ball_positions is not None else None,
    fps                = 25,
    output_path        = f"block_{seq.sequence_id}.mp4",
    start_frame        = s,
    pitch_length       = pitch_length,
    pitch_width        = pitch_width,
)
```

---

## 13. Future Extensions

### 13.1 Horizontal Compactness (Y Axis)

An analogous metric using the Y-axis IQR of the defending team would capture **width** of the defensive shape. A narrow team concentrates centrally; a wide team stretches across the pitch. This can be added as a third independent dimension:

- Metric: `iqr_y` = P75(Y) − P25(Y) of outfield defenders.
- States: `narrow` / `wide`, with its own threshold and hysteresis.

### 13.2 Defensive Line Detection

Rather than treating all outfield defenders as a single group, identify the **defensive line** (back four/five) separately from the **midfield line**, and compute block height and compactness per line. This enables detection of line separations (gap between midfield and defense) and of mismatches between the two lines.

### 13.3 Pressing Trigger Detection

Identify frames where the block height changes rapidly from `mid` or `low` to `high`, which may correspond to **pressing triggers** — moments when the team collectively decides to press high. Cross-reference with ball position and ball-carrier identity for richer context.

### 13.4 Synchronization Metric

Measure how **synchronized** individual defenders move along the X axis by computing the correlation or variance of individual X velocities. A highly synchronized team moves as a unit; a desynchronized team has individuals moving in different directions simultaneously.

### 13.5 Orientation to the Ball

Augment all metrics by conditioning on **ball position**. For example, a block that sits at median X = −15m when the ball is at X = 0m is different tactically from a block at median X = −15m when the ball is at X = −30m. Block depth **relative to the ball** may be more stable and interpretable than absolute block depth.

### 13.6 Real-Time Streaming Mode

Adapt the pipeline to operate frame-by-frame with causal smoothing (`center=False`) and maintain the state machine across frames without buffering the full period in memory. This enables live match analysis.

---

*End of specification.*
