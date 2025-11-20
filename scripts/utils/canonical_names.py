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

import os
import logging
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from scripts.utils.team_codes import canon_team as _canon_team_from_codes

logger = logging.getLogger(__name__)

_ROLES_CACHE: pd.DataFrame | None = None
_ROLES_LOOKUP_CACHE: dict[str, str] | None = None

# Optional override from env, if you ever want to force a path
_ROLES_CSV_OVERRIDE = os.environ.get("ROLES_CSV", "").strip() or None

_PUNCT_PATTERN = re.compile(r"[\.\u2019\u2018'`]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _clean_token(name: str | None) -> str:
    text = "" if name is None else str(name)
    text = text.strip()
    if not text:
        return ""
    text = _PUNCT_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text


def canon_team(name: str | None) -> str:
    """Canonicalize a team name or abbreviation into a standard code."""

    try:
        # Delay import to avoid circular dependency during module initialization.
        from scripts import _opponent_map as opponent_map
    except Exception:
        opponent_map = None

    cleaned = _clean_token(name)
    if not cleaned:
        return ""

    upper = cleaned.upper()
    mapping = getattr(opponent_map, "CANON_TEAM_ABBR", {}) if opponent_map else {}
    if upper in mapping:
        return mapping[upper]

    title = cleaned.title()
    city_map = getattr(opponent_map, "ESPN_CITY_TO_ABBR", {}) if opponent_map else {}
    if title in city_map:
        return city_map[title]

    name_map = getattr(opponent_map, "TEAM_NAME_TO_ABBR", {}) if opponent_map else {}
    if title in name_map:
        return name_map[title]

    fallback = _canon_team_from_codes(cleaned)
    if fallback:
        return str(fallback)

    return upper


def canon_team_series(series: pd.Series) -> pd.Series:
    """Vectorized wrapper for canon_team."""

    return series.fillna("").astype(str).apply(canon_team)


def build_roles_map() -> pd.DataFrame:
    """
    Load the roles_ourlads.csv file from one of several candidate locations.

    If nothing usable is found, we raise ValueError so callers can decide
    whether to fail hard or just fall back to raw names.
    """

    global _ROLES_CACHE, _ROLES_LOOKUP_CACHE

    if _ROLES_CACHE is not None:
        return _ROLES_CACHE

    candidates: list[Path] = []

    # Highest priority: explicit override from env / workflow
    if _ROLES_CSV_OVERRIDE:
        candidates.append(Path(_ROLES_CSV_OVERRIDE))

    # Then the usual suspects
    candidates.extend(
        [
            Path("data/roles_ourlads.csv"),
            Path("outputs/roles_ourlads.csv"),
            Path("roles_ourlads.csv"),
        ]
    )

    logger.info(
        "[canonical_names] build_roles_map: candidate paths=%s",
        [str(p) for p in candidates],
    )

    last_error: Exception | None = None

    for p in candidates:
        try:
            if not p.exists():
                logger.debug(
                    "[canonical_names] build_roles_map: %s does not exist; skipping",
                    p,
                )
                continue

            size = p.stat().st_size
            # Treat tiny files as effectively empty (e.g. touched stubs)
            if size < 10:
                logger.debug(
                    "[canonical_names] build_roles_map: %s is only %d bytes; treating as empty",
                    p,
                    size,
                )
                continue

            logger.info(
                "[canonical_names] build_roles_map: loading roles from %s (bytes=%d)",
                p,
                size,
            )
            df = pd.read_csv(p)

            if (
                df.empty
                or "player" not in df.columns
                or "player_key" not in df.columns
            ):
                logger.warning(
                    "[canonical_names] build_roles_map: %s loaded but empty or missing required columns; skipping",
                    p,
                )
                continue

            _ROLES_CACHE = df
            _ROLES_LOOKUP_CACHE = None
            logger.info(
                "[canonical_names] build_roles_map: loaded roles shape=%s", df.shape
            )
            return df

        except Exception as exc:  # includes EmptyDataError
            last_error = exc
            logger.warning(
                "[canonical_names] build_roles_map: failed to read %s; skipping. error=%s",
                p,
                exc,
            )
            continue

    # If we get here, nothing worked
    msg = (
        "Could not locate a non-empty roles_ourlads.csv. "
        f"Candidates={ [str(p) for p in candidates] }. "
        f"Last error={last_error!r}"
    )
    logger.error("[canonical_names] ERROR: %s", msg)
    raise ValueError(msg)

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


def _build_roles_map_from_df(df: pd.DataFrame) -> dict[str, str]:
    if df.shape[0] <= 0:
        raise ValueError("[canonical_names] roles CSV has no data rows.")
    if len(df.columns) == 0:
        raise ValueError("build_roles_map: roles CSV has no columns")

    cols = {c.lower(): c for c in df.columns}
    if "player_key" not in cols or "player" not in cols:
        raise ValueError("build_roles_map: roles CSV missing required columns")

    key_col = cols["player_key"]
    name_col = cols["player"]
    keys = df[key_col].astype(str).map(norm_key)
    names = df[name_col].astype(str).map(strip_middle_initial)
    return dict(zip(keys, names))


def build_roles_map_from_csv(path: str | Path) -> dict[str, str]:
    """Load a roles_ourlads.csv and return the existing roles map structure."""

    p = Path(path)
    if not p.exists():
        raise ValueError(f"[canonical_names] roles CSV not found at {p}")
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"[canonical_names] roles CSV at {p} is empty") from exc

    if df.empty:
        raise ValueError(f"[canonical_names] roles CSV at {p} has no rows")

    logger.info(
        "[canonical_names] Loaded roles CSV %s with shape=%s",
        p,
        df.shape,
    )

    return _build_roles_map_from_df(df)


def _get_roles_lookup() -> dict[str, str]:
    global _ROLES_LOOKUP_CACHE

    if _ROLES_LOOKUP_CACHE is None:
        df = build_roles_map()
        _ROLES_LOOKUP_CACHE = _build_roles_map_from_df(df)
    return _ROLES_LOOKUP_CACHE

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

    roles_lookup = _get_roles_lookup()
    return roles_lookup.get(k, cleaned)

# Optional: explicit alias used in some places
canonicalize_name = canonicalize_player_name


_UNMAPPED_LOG = os.environ.get("UNMAPPED_NAME_LOG", "data/_debug/unmapped_names.jsonl")


def canonicalize_player_name_safe(raw: str) -> tuple[str, str]:
    """
    Try to canonicalize a player name; on any problem, log a warning
    and fall back to (raw_name, raw_name).
    """

    try:
        canonical_name = canonicalize_player_name(raw)
        canonical_name = str(canonical_name or "").strip()
        canonical_key = norm_key(canonical_name) if canonical_name else ""
        return canonical_name, canonical_key
    except ValueError as exc:
        fallback = "" if raw is None else str(raw).strip()
        logger.warning(
            "[canonical_names] canonicalize_player_name_safe: falling back to raw name %r due to ValueError: %s",
            raw,
            exc,
        )
        return fallback, fallback
    except Exception as exc:
        fallback = "" if raw is None else str(raw).strip()
        logger.warning(
            "[canonical_names] canonicalize_player_name_safe: unexpected error for %r; falling back to raw. error=%s",
            raw,
            exc,
        )
        return fallback, fallback


def log_unmapped_variant(raw: str, where: str = "unknown", *args, **kwargs) -> None:
    # Optional debug hook; keep silent unless you want to log variants externally.
    return
