"""Utilities for fetching nflverse play-by-play data with robust fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

PBP_MIRRORS: Iterable[str] = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/pbp_2025.csv.gz",
    "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/play_by_play/pbp_2025.csv.gz",
    "https://github.com/nflverse/nflfastR-data/raw/master/data/play_by_play/pbp_2025.csv.gz",
)

CACHE_PATH = Path("data/_cache/pbp_2025.csv.gz")
_MIN_BYTES = 1_000_000
_CHUNK_SIZE = 65536


def _ensure_cache_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _download_to_cache(url: str, path: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)


def get_pbp_2025() -> pd.DataFrame:
    """Fetch the 2025 nflverse play-by-play file with fallbacks.

    Downloads each mirror sequentially until one succeeds, streaming to disk to
    avoid loading large responses into memory. The cached file is validated to
    exceed 1 MB before being parsed.
    """

    _ensure_cache_dir(CACHE_PATH)
    last_error: Exception | None = None

    for url in PBP_MIRRORS:
        try:
            _download_to_cache(url, CACHE_PATH)
            if CACHE_PATH.stat().st_size <= _MIN_BYTES:
                last_error = RuntimeError(f"Downloaded file from {url} is too small")
                CACHE_PATH.unlink(missing_ok=True)
                continue
            df = pd.read_csv(CACHE_PATH, compression="gzip", low_memory=False)
            return df
        except Exception as err:  # noqa: PERF203 - we want the last error message
            last_error = err
            CACHE_PATH.unlink(missing_ok=True)
            continue

    raise RuntimeError(
        f"Unable to fetch 2025 pbp from mirrors. Last error: {last_error}"
    )
