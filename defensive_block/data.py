"""Utilities for downloading Metrica Sports open sample data.

Usage::

    from defensive_block import download_metrica_sample

    paths = download_metrica_sample(game=1, dest_dir="./data")
    # paths = {"home_tracking_path": Path(...),
    #          "away_tracking_path": Path(...),
    #          "events_path":        Path(...)}

The returned dict can be passed directly to MetricaLoader.load(**paths).
Files already present on disk are skipped unless force=True.
Only stdlib urllib is used — no extra dependencies.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

_BASE_URL = (
    "https://raw.githubusercontent.com/metrica-sports/"
    "sample-data/master/data"
)

_GAME_FILES = {
    "home": "Sample_Game_{n}/Sample_Game_{n}_RawTrackingData_Home_Team.csv",
    "away": "Sample_Game_{n}/Sample_Game_{n}_RawTrackingData_Away_Team.csv",
    "events": "Sample_Game_{n}/Sample_Game_{n}_RawEventsData.csv",
}


def download_metrica_sample(
    game: int = 1,
    dest_dir: str | Path = "./data",
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    """Download Metrica Sports sample tracking and event CSV files.

    Games 1 and 2 are supported (Game 3 uses the FIFA EPTS XML format).

    Args:
        game: Game number — 1 or 2.
        dest_dir: Directory where files are saved. Created if it does not exist.
        force: Re-download even if the file already exists on disk.
        verbose: Print download progress to stdout.

    Returns:
        Dict with keys matching MetricaLoader.load() parameters::

            {
                "home_tracking_path": Path,
                "away_tracking_path": Path,
                "events_path":        Path,
            }

    Raises:
        ValueError: If game is not 1 or 2.
        urllib.error.URLError: On network errors.
    """
    if game not in (1, 2):
        raise ValueError(f"game must be 1 or 2 (Game 3 uses EPTS format). Got: {game}")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}
    key_map = {
        "home":   "home_tracking_path",
        "away":   "away_tracking_path",
        "events": "events_path",
    }

    for file_key, template in _GAME_FILES.items():
        relative = template.format(n=game)
        url      = f"{_BASE_URL}/{relative}"
        local    = dest / Path(relative).name

        if local.exists() and not force:
            if verbose:
                print(f"  [skip] {local.name} already exists")
        else:
            if verbose:
                print(f"  [download] {local.name} …", end=" ", flush=True)
            _download_with_progress(url, local, verbose)

        result[key_map[file_key]] = local

    return result


def _download_with_progress(url: str, dest: Path, verbose: bool) -> None:
    """Download url to dest, printing a dot for each 10 % of progress."""
    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if not verbose or total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, int(downloaded * 100 / total_size))
        # Print a dot every ~10 %
        if block_num % max(1, int(total_size / block_size / 10)) == 0:
            print(".", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    if verbose:
        print(" done")
