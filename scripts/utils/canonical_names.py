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

log = logger

_ROLES_OURLADS_DF: pd.DataFrame | None = None
_ROLES_OURLADS_PATH: Path | None = None


def _load_roles_ourlads_df(override_path: str | Path | None = None) -> pd.DataFrame:
    """
    Locate and load roles_ourlads.csv from disk with robust path handling.
    Returns a DataFrame, or raises ValueError with detailed diagnostics.
    """

    global _ROLES_OURLADS_DF, _ROLES_OURLADS_PATH

    override_candidate = Path(override_path) if override_path else None
    if _ROLES_OURLADS_DF is not None:
        if override_candidate is None:
            return _ROLES_OURLADS_DF
        if _ROLES_OURLADS_PATH and Path(_ROLES_OURLADS_PATH) == override_candidate.resolve():
            return _ROLES_OURLADS_DF

    candidate_paths: list[Path] = []
    if override_candidate is not None:
        candidate_paths.append(override_candidate)

    env_path = os.environ.get("ROLES_OURLADS_PATH")
    if env_path:
        candidate_paths.append(Path(env_path))

    candidate_paths.extend(
        [
            Path("outputs") / "roles_ourlads.csv",
            Path("data") / "roles_ourlads.csv",
            Path("roles_ourlads.csv"),
            # common artifact nesting: outputs/roles_ourlads/roles_ourlads.csv
            Path("outputs") / "roles_ourlads" / "roles_ourlads.csv",
        ]
    )

    diagnostics: list[str] = []

    print("[canonical_names] DEBUG: attempting to locate roles_ourlads.csv")
    for p in candidate_paths:
        abs_path = p.resolve()
        exists = p.exists()
        size = os.path.getsize(p) if exists else 0
        diagnostics.append(f"- {abs_path} (exists={exists}, bytes={size})")
        print(f"[canonical_names] DEBUG: candidate {abs_path} exists={exists}, bytes={size}")

        if exists and size > 0:
            try:
                df = pd.read_csv(p)
            except Exception as e:
                print(f"[canonical_names] WARNING: failed to read {abs_path}: {e}")
                continue

            if df.empty:
                print(
                    f"[canonical_names] WARNING: {abs_path} read successfully but is empty (0 rows)"
                )
                continue

            print(f"[canonical_names] INFO: using roles_ourlads from {abs_path} shape={df.shape}")
            _ROLES_OURLADS_DF = df
            _ROLES_OURLADS_PATH = abs_path
            return df

    message = (
        "[canonical_names] ERROR: Could not locate a non-empty roles_ourlads.csv.\n"
        "Checked the following candidate paths:\n" + "\n".join(diagnostics)
    )
    print(message)
    raise ValueError(message)

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


def _build_roles_map_from_df(df: pd.DataFrame) -> dict[str, str]:
    if df.shape[0] <= 0:
        raise ValueError(
            "[canonical_names] roles CSV has no data rows."
        )
    if len(df.columns) == 0:
        raise ValueError(
            "build_roles_map: roles CSV has no columns"
        )

    cols = {c.lower(): c for c in df.columns}
    if "player_key" not in cols or "player" not in cols:
        raise ValueError(
            "build_roles_map: roles CSV missing required columns"
        )

    key_col = cols["player_key"]
    name_col = cols["player"]
    df["_key"] = df[key_col].astype(str).map(norm_key)
    df["_full"] = df[name_col].astype(str).map(strip_middle_initial)
    return dict(zip(df["_key"], df["_full"]))


def build_roles_map_from_csv(path: str | Path) -> dict[str, str]:
    """Load a roles_ourlads.csv and return the existing roles map structure.

    Raises a clear ValueError if the file is missing or empty.
    """

    global _ROLES_CACHE, _ROLES_CACHE_PATH, _ROLES_OURLADS_DF, _ROLES_OURLADS_PATH

    p = Path(path)
    if (
        _ROLES_CACHE is not None
        and _ROLES_CACHE_PATH is not None
        and _ROLES_CACHE_PATH.resolve() == p.resolve()
    ):
        return _ROLES_CACHE
    if not p.exists():
        raise ValueError(f"[canonical_names] roles CSV not found at {p}")
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        raise ValueError(f"[canonical_names] roles CSV at {p} is empty")

    if df.empty:
        raise ValueError(f"[canonical_names] roles CSV at {p} has no rows")

    log.info(
        "[canonical_names] Loaded roles CSV %s with shape=%s",
        p,
        df.shape,
    )

    roles_map = _build_roles_map_from_df(df)

    _ROLES_CACHE = roles_map
    _ROLES_CACHE_PATH = p
    _ROLES_OURLADS_DF = df
    _ROLES_OURLADS_PATH = p
    return roles_map


def build_roles_map(roles_path: str | Path | None = None) -> dict[str, str]:
    """
    Load the roles_ourlads CSV used for canonical name resolution.

    This function is hardened against missing/empty files and logs exactly
    which path is chosen (or why it failed).
    """

    global _ROLES_CACHE, _ROLES_CACHE_PATH

    resolved_override = Path(roles_path) if roles_path is not None else None
    if roles_path is None and _ROLES_CACHE is not None:
        logger.debug("build_roles_map: returning cached map for %s", _ROLES_CACHE_PATH)
        return _ROLES_CACHE
    if (
        resolved_override is not None
        and _ROLES_CACHE is not None
        and _ROLES_CACHE_PATH is not None
        and _ROLES_CACHE_PATH == resolved_override
    ):
        logger.debug(
            "build_roles_map: returning cached map for %s", _ROLES_CACHE_PATH
        )
        return _ROLES_CACHE

    _load_roles_ourlads_df(roles_path)
    resolved_path = _ROLES_OURLADS_PATH or resolved_override

    if resolved_path is None:
        raise ValueError("[canonical_names] unable to resolve roles_ourlads.csv path")

    roles_map = build_roles_map_from_csv(resolved_path)

    logger.info(
        "build_roles_map: using %s (rows=%d)",
        resolved_path,
        len(roles_map),
    )

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
    Canonicalize a player name using roles_ourlads.csv if available.
    If roles cannot be loaded, fall back to the raw name but log once.
    """

    fallback = "" if raw_name is None else str(raw_name).strip()

    try:
        roles_map = build_roles_map()
    except ValueError:
        logger.info(
            "[canonical_names] canonicalize_player_name_safe: falling back to raw "
            "name %r because roles_ourlads.csv could not be loaded.",
            fallback,
        )
        roles_map = None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "canonicalize_player_name_safe: falling back to raw name %r due to %s: %s",
            raw_name,
            type(exc).__name__,
            exc,
        )
        roles_map = None

    manual_map = build_manual_map()

    cleaned = _strip_ourlads_noise(str(raw_name))
    if not cleaned:
        return fallback, None

    k = norm_key(cleaned)
    if k in manual_map:
        name = manual_map[k]
    elif roles_map is not None and k in roles_map:
        name = roles_map[k]
    else:
        name = cleaned

    name_str = "" if name is None else str(name).strip()

    key_str = norm_key(name_str or fallback) or None
    return name_str, key_str


def log_unmapped_variant(raw: str, where: str = "unknown", *args, **kwargs) -> None:
    # Optional debug hook; keep silent unless you want to log variants externally.
    return
