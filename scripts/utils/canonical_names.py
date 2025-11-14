#!/usr/bin/env python3
"""
Backwards-compatible name canonicalization utilities.

Exports:
  - canonicalize_player_name(source_key: str) -> str
      (legacy API expected by make_player_form.py)
  - build_roles_map(), build_manual_map() helpers
  - norm_key(), strip_middle_initial()

The function canonicalize_player_name uses:
  1) roles_ourlads.csv (player_key -> full name, ignoring middle initials)
  2) data/manual_name_overrides.csv (player_source_name -> full name), overrides roles
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------- helpers ----------

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

_SPACES = re.compile(r"\s+")
_TRAIL_TAG = re.compile(r"\s+[A-Z]{1,3}\d{1,2}(?:/[A-Z]\d{1,2})?$")

def strip_middle_initial(full_name: str) -> str:
    if not isinstance(full_name, str):
        return ""
    parts = re.split(r"\s+", full_name.replace(",", " ").strip())
    parts = [p.replace(".", "") for p in parts if p.strip()]
    if not parts:
        return ""
    first = parts[0]
    last = parts[-1]
    if last.lower() in SUFFIXES and len(parts) >= 3:
        last = parts[-2] + " " + parts[-1]
    return f"{first} {last}".strip()


def _strip_ourlads_noise(name: str) -> str:
    """Normalize raw roster names into "First Last" style strings."""

    if not isinstance(name, str):
        return ""

    working = name.strip()
    if not working:
        return ""

    # remove trailing roster codes like "24/1", "CF25", "U/NE"
    working = _TRAIL_TAG.sub("", working)

    # flip "Last, First" ordering when present
    if "," in working:
        last, first = [p.strip() for p in working.split(",", 1)]
        working = f"{first} {last}".strip()

    # collapse whitespace + normalize apostrophes
    working = working.replace("’", "'")
    working = _SPACES.sub(" ", working)

    return working.strip()

def norm_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    for ch in (" ", "'", "-", "."):
        s = s.replace(ch, "")
    return s

# ---------- map builders (cached) ----------

_ROLES_CACHE: dict[str, str] | None = None
_ROLES_CACHE_PATH: Path | None = None


def build_roles_map(roles_path: str | Path | None = None) -> dict[str, str]:
    """
    Load the roles_ourlads CSV used for canonical name resolution.

    This function is hardened against missing/empty files and logs exactly
    which path is chosen (or why it failed).
    """

    global _ROLES_CACHE, _ROLES_CACHE_PATH

    if roles_path is not None:
        requested_path = Path(roles_path)
        if _ROLES_CACHE is not None and _ROLES_CACHE_PATH == requested_path:
            logger.debug("build_roles_map: returning cached map for %s", requested_path)
            return _ROLES_CACHE
        candidate_paths: list[Path] = [requested_path]
    else:
        if _ROLES_CACHE is not None and _ROLES_CACHE_PATH is not None:
            logger.debug(
                "build_roles_map: returning cached map for %s", _ROLES_CACHE_PATH
            )
            return _ROLES_CACHE
        candidate_paths = []

    candidate_paths.extend(
        p for p in [Path("outputs/roles_ourlads.csv"), Path("data/roles_ourlads.csv")]
        if p not in candidate_paths
    )

    tried: list[tuple[Path, str, int]] = []
    seen: set[Path] = set()

    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)

        if not path.exists():
            logger.info("build_roles_map: %s does not exist", path)
            tried.append((path, "missing", 0))
            continue

        try:
            size = path.stat().st_size
        except OSError:
            size = -1

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            logger.warning("build_roles_map: %s is empty (size=%s)", path, size)
            tried.append((path, "empty_csv", size))
            continue
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "build_roles_map: failed to read %s (size=%s) due to %s: %s",
                path,
                size,
                type(exc).__name__,
                exc,
            )
            tried.append((path, f"read_error:{type(exc).__name__}", size))
            continue

        if df.empty or len(df.columns) == 0:
            logger.warning(
                "build_roles_map: %s has no rows/cols (size=%s)", path, size
            )
            tried.append((path, "no_rows_or_cols", size))
            continue

        cols = {c.lower(): c for c in df.columns}
        if "player_key" not in cols or "player" not in cols:
            logger.warning(
                "build_roles_map: %s missing required columns (size=%s)",
                path,
                size,
            )
            tried.append((path, "missing_columns", size))
            continue

        key_col = cols["player_key"]
        name_col = cols["player"]
        df["_key"] = df[key_col].astype(str).map(norm_key)
        df["_full"] = df[name_col].astype(str).map(strip_middle_initial)
        roles_map = dict(zip(df["_key"], df["_full"]))

        logger.info(
            "build_roles_map: using %s (rows=%d, cols=%d, size=%s)",
            path,
            len(df),
            len(df.columns),
            size,
        )

        _ROLES_CACHE = roles_map
        _ROLES_CACHE_PATH = path
        return roles_map

    lines = ["build_roles_map: no usable roles_ourlads.csv candidates found:"]
    for path, status, size in tried:
        lines.append(f"  - {path}: {status}, size={size}")
    message = "\n".join(lines)
    logger.error(message)
    raise RuntimeError(message)

@lru_cache(maxsize=1)
def build_manual_map(overrides_path: str = "data/manual_name_overrides.csv") -> dict:
    p = Path(overrides_path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    need = {"player_source_name", "full_name"}
    if not need.issubset(df.columns):
        return {}
    df["_key"] = df["player_source_name"].astype(str).map(norm_key)
    return dict(zip(df["_key"], df["full_name"]))

# ---------- legacy API (kept for make_player_form.py) ----------

def canonicalize_player_name(source_key: str) -> str:
    """Return a cleaned "First Last" style name for varied upstream inputs."""

    if source_key is None:
        return ""

    cleaned = _strip_ourlads_noise(str(source_key))
    if not cleaned:
        return ""

    k = norm_key(cleaned)
    manual = build_manual_map()
    if k in manual:
        return manual[k]
    roles = build_roles_map()
    return roles.get(k, cleaned)

# Optional: explicit alias used in some places
canonicalize_name = canonicalize_player_name


_UNMAPPED_LOG = os.environ.get("UNMAPPED_NAME_LOG", "data/_debug/unmapped_names.jsonl")


def canonicalize_player_name_safe(raw_name: str) -> tuple[str, Optional[str]]:
    """
    Safe wrapper around canonicalize_player_name.

    - On success: returns (canonical_name, canonical_key)
    - On failure: logs and falls back to (raw_name, None) without raising.
    """

    try:
        out = canonicalize_player_name(raw_name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "canonicalize_player_name_safe: falling back to raw name %r due to %s: %s",
            raw_name,
            type(exc).__name__,
            exc,
        )
        fallback = "" if raw_name is None else str(raw_name).strip()
        return fallback, None

    if isinstance(out, tuple):
        if len(out) >= 2:
            name, key = out[0], out[1]
        elif len(out) == 1:
            name, key = out[0], out[0]
        else:
            name, key = "", ""
    else:
        name, key = out, out

    name_str = "" if name is None else str(name).strip()
    raw_fallback = "" if raw_name is None else str(raw_name).strip()

    if key is None:
        key_str: Optional[str] = None
    else:
        key_str = str(key).strip()
        if not key_str:
            key_str = None

    if not key_str:
        normalized = norm_key(name_str or raw_fallback)
        key_str = normalized or None

    return name_str, key_str


def log_unmapped_variant(raw: str, where: str = "unknown", *args, **kwargs) -> None:
    # Optional debug hook; keep silent unless you want to log variants externally.
    return
