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


def _resolve_roles_ourlads_path() -> Path:
    """
    Locate a non-empty roles_ourlads.csv file.

    Priority:
    1. ROLES_OURLADS_PATH env var, if set and non-empty.
    2. outputs/roles_ourlads.csv (authoritative).
    3. data/roles_ourlads.csv (mirror / legacy).

    Raises a clear ValueError if no suitable file is found.
    """

    candidates = []

    env_path = os.environ.get("ROLES_OURLADS_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(Path("outputs/roles_ourlads.csv"))
    candidates.append(Path("data/roles_ourlads.csv"))

    tried_messages: list[str] = []

    for path in candidates:
        if not path:
            continue
        tried_messages.append(str(path))
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                first = f.readline()
                second = f.readline()
                if not first:
                    continue
                if not second:
                    continue
        except OSError:
            continue

        print(f"[canonical_names] Using roles file: {path}")
        return path

    msg = (
        "[canonical_names] ERROR: Could not locate a non-empty roles_ourlads.csv.\n"
        f"Tried paths (in order): {', '.join(tried_messages) or '<none>'}.\n"
        "Ensure the 'Build depth / roles (Ourlads)' step completed successfully "
        "and that its artifact was restored into this job's workspace."
    )
    raise ValueError(msg)


def build_roles_map(roles_path: str | Path | None = None) -> dict[str, str]:
    """
    Load the roles_ourlads CSV used for canonical name resolution.

    This function is hardened against missing/empty files and logs exactly
    which path is chosen (or why it failed).
    """

    global _ROLES_CACHE, _ROLES_CACHE_PATH

    if roles_path is None:
        resolved_path = _resolve_roles_ourlads_path()
    else:
        resolved_path = Path(roles_path)

    if _ROLES_CACHE is not None and _ROLES_CACHE_PATH == resolved_path:
        logger.debug("build_roles_map: returning cached map for %s", resolved_path)
        return _ROLES_CACHE

    if _ROLES_CACHE is not None and roles_path is None and _ROLES_CACHE_PATH is not None:
        logger.debug(
            "build_roles_map: returning cached map for %s", _ROLES_CACHE_PATH
        )
        return _ROLES_CACHE

    print(f"[canonical_names] build_roles_map loading from: {resolved_path}")

    if not resolved_path.exists():
        message = f"build_roles_map: specified roles file does not exist: {resolved_path}"
        logger.error(message)
        raise ValueError(message)

    try:
        size = resolved_path.stat().st_size
    except OSError:
        size = -1

    try:
        df = pd.read_csv(resolved_path)
    except pd.errors.EmptyDataError as exc:
        logger.warning(
            "build_roles_map: %s is empty (size=%s)", resolved_path, size
        )
        raise ValueError(
            f"build_roles_map: {resolved_path} is empty (size={size})"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "build_roles_map: failed to read %s (size=%s) due to %s: %s",
            resolved_path,
            size,
            type(exc).__name__,
            exc,
        )
        raise

    if df.shape[0] <= 0:
        raise ValueError(
            f"[canonical_names] roles CSV {resolved_path} has no data rows."
        )
    if len(df.columns) == 0:
        logger.warning(
            "build_roles_map: %s has no columns (size=%s)", resolved_path, size
        )
        raise ValueError(
            f"build_roles_map: {resolved_path} has no columns (size={size})"
        )

    cols = {c.lower(): c for c in df.columns}
    if "player_key" not in cols or "player" not in cols:
        logger.warning(
            "build_roles_map: %s missing required columns (size=%s)",
            resolved_path,
            size,
        )
        raise ValueError(
            f"build_roles_map: {resolved_path} missing required columns (size={size})"
        )

    key_col = cols["player_key"]
    name_col = cols["player"]
    df["_key"] = df[key_col].astype(str).map(norm_key)
    df["_full"] = df[name_col].astype(str).map(strip_middle_initial)
    roles_map = dict(zip(df["_key"], df["_full"]))

    logger.info(
        "build_roles_map: using %s (rows=%d, cols=%d, size=%s)",
        resolved_path,
        len(df),
        len(df.columns),
        size,
    )

    _ROLES_CACHE = roles_map
    _ROLES_CACHE_PATH = resolved_path
    return roles_map

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
