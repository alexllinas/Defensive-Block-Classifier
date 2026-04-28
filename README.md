# Defensive Block Detection

Frame-level classification of a football team's **defensive block** from player tracking data.

For each frame of live play, the system labels the defending team's shape across two independent dimensions:

| Dimension | Metric | States |
|---|---|---|
| **Block height** | Median X of outfield defenders | `low` · `mid` · `high` |
| **Block length** | IQR of X positions | `compact` · `long` |

Combined labels: `low_compact`, `low_long`, `mid_compact`, `mid_long`, `high_compact`, `high_long`.

---

## Installation

```bash
pip install .
```

---

## Quickstart — Metrica open data

```python
from defensive_block import (
    download_metrica_sample,
    MetricaLoader,
    BlockDetectionPipeline,
)

# Download Metrica Game 1 CSVs (~3 MB total, cached after first run)
paths = download_metrica_sample(game=1, dest_dir="./data")

# Load tracking data
loader = MetricaLoader()
ti = loader.load(**paths, match_id="metrica_game_1")

# Run detection (analyze home team = index 0)
pipeline = BlockDetectionPipeline()
match_summary, frame_results, sequences = pipeline.run(ti, analyzed_team_idx=0)

# Results
print(f"Dominant block: {match_summary.dominant_combined}")
print(f"Defensive sequences: {match_summary.n_sequences}")
print(f"Mean depth (median X): {match_summary.match_mean_median_x:.1f} m")
```

---

## Visualization

```python
from defensive_block.visualize import plot_frame, plot_match_timeline
import matplotlib.pyplot as plt

# Match timeline
fig, ax1, ax2 = plot_match_timeline(
    frame_results, sequences, fps=25.0, title="Metrica Game 1"
)
plt.show()

# Single frame
seq = sequences[5]
mid_frame = seq.start_frame + seq.n_frames // 2
state = frame_results.block_state_at(mid_frame)

fig, ax = plot_frame(
    positions=ti.positions[mid_frame],
    team_indices={"defending": [...], "attacking": [...]},
    goalkeeper_indices=[...],
    config=pipeline.config,
    block_state=state,
    ball_xy=ti.ball_positions[mid_frame] if ti.ball_positions is not None else None,
)
plt.show()
```
<img width="1632" height="1187" alt="frame_high_compact" src="https://github.com/user-attachments/assets/977cfc9a-4fa3-4b48-ba6b-b41e7ee80b4f" />

See [examples/metrica_demo.ipynb](examples/metrica_demo.ipynb) for a full walkthrough.

<img width="1280" height="788" alt="sequence_04_optimized" src="https://github.com/user-attachments/assets/94b5cbb4-aacc-4d3d-96b7-e0de01ccad1d" />

---

## Algorithm overview

```
Tracking CSV / HDF5
      │
      ▼  loader (MetricaLoader / custom BaseLoader)
 TrackingInput   ← common internal format
      │
      ▼  preprocessing
 Outfield X positions, direction-normalized, presence-masked
      │
      ├─ median_x  (block height proxy)
      └─ iqr_x     (block length proxy)
            │
            ▼  temporal smoothing (rolling mean)
            │
            ▼  hysteresis state machines
            │    height: low / mid / high
            │    length: compact / long
            │
            ▼  sequence & match aggregation
       MatchSummary · SequenceSummary · FrameResults
```

Full algorithm specification: [docs/defensive_block_detection.md](docs/defensive_block_detection.md).

---

## Project structure

```
defensive_block/
├── config.py          — DefensiveBlockConfig
├── types.py           — TrackingInput and all output data classes
├── preprocessing.py   — direction normalization, masking, sequence segmentation
├── metrics.py         — median_x, iqr_x
├── smoothing.py       — rolling mean
├── classification.py  — hysteresis state machines
├── aggregation.py     — sequence and match aggregation
├── pipeline.py        — BlockDetectionPipeline (end-to-end orchestration)
├── visualize.py       — pitch rendering and timeline plots
├── data.py            — download_metrica_sample()
└── io/
    ├── base.py        — BaseLoader (abstract interface)
    └── metrica.py     — MetricaLoader (public open data)
```

---

## Adding a custom loader

Subclass `BaseLoader` and return a `TrackingInput`:

```python
from defensive_block.io.base import BaseLoader
from defensive_block.types import TrackingInput
import numpy as np

class MyFormatLoader(BaseLoader):
    def load(self, filepath, match_id="") -> TrackingInput:
        # Read your format here…
        return TrackingInput(
            positions     = ...,   # (N_frames, N_players, 2) metres, centred
            in_field      = ...,   # (N_frames, N_players) bool
            in_play       = ...,   # (N_frames,) bool
            possession_team = ..., # (N_frames,) int — 0 or 1; −1 = no possession
            player_team   = ...,   # (N_players,) int
            player_is_gk  = ...,   # (N_players,) bool
            team_direction= ...,   # (2, N_frames) — +1.0 or −1.0
            segment       = ...,   # (N_frames,) int
            pitch_size    = (105.0, 68.0),
            match_id      = match_id,
        )
```

---

## License

MIT
