#!/usr/bin/env python3
"""Backwards-compatible name and team canonicalization utilities.

`roles_ourlads.csv` is a production enrichment artifact, not a universal
runtime dependency.  Historical/backtest jobs may legitimately run without
it, so name canonicalization must degrade to deterministic cleaned names/keys
instead of raising and emitting one warning per player.
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from scripts.utils.team_codes import canon_team as _canon_team_from_codes

logger = logging.getLogger(__name__)
_ROLES_CACHE: pd.DataFrame | None = None
_ROLES_LOOKUP_CACHE: dict[str, str] | None = None
_ROLES_MISSING_LOGGED = False
_ROLES_CSV_OVERRIDE = os.environ.get("ROLES_CSV", "").strip() or None
_PUNCT_PATTERN = re.compile(r"[\.\u2019\u2018'`]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_SPACES = re.compile(r"\s+")
_TRAIL_TAG = re.compile(r"\s+[A-Z]{1,3}\d{1,2}(?:/[A-Z]\d{1,2})?$")


def _clean_token(name: str | None) -> str:
    text = "" if name is None else str(name)
    text = _PUNCT_PATTERN.sub("", text.strip())
    return _WHITESPACE_PATTERN.sub(" ", text) if text else ""


def canon_team(name: str | None) -> str:
    try:
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
    return str(fallback) if fallback else upper


def canon_team_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).apply(canon_team)


def build_roles_map(*, required: bool = False) -> pd.DataFrame:
    """Load a roles artifact when present; otherwise return an empty schema.

    Set ``required=True`` only in production stages whose contract explicitly
    requires the Ourlads roles artifact.  Generic canonicalization and
    historical backtests intentionally treat it as optional.
    """
    global _ROLES_CACHE, _ROLES_LOOKUP_CACHE, _ROLES_MISSING_LOGGED
    if _ROLES_CACHE is not None:
        return _ROLES_CACHE
    candidates: list[Path] = []
    if _ROLES_CSV_OVERRIDE:
        candidates.append(Path(_ROLES_CSV_OVERRIDE))
    candidates.extend([Path("data/roles_ourlads.csv"), Path("outputs/roles_ourlads.csv"), Path("roles_ourlads.csv")])
    last_error: Exception | None = None
    for p in candidates:
        try:
            if not p.exists() or p.stat().st_size < 10:
                continue
            df = pd.read_csv(p)
            if df.empty or not {"player", "player_key"}.issubset(df.columns):
                continue
            _ROLES_CACHE = df
            _ROLES_LOOKUP_CACHE = None
            logger.info("[canonical_names] loaded optional roles artifact %s shape=%s", p, df.shape)
            return df
        except Exception as exc:
            last_error = exc
    if required:
        raise ValueError(
            "Could not locate a usable roles_ourlads.csv. "
            f"Candidates={[str(p) for p in candidates]}; last_error={last_error!r}"
        )
    if not _ROLES_MISSING_LOGGED:
        logger.info(
            "[canonical_names] roles_ourlads.csv unavailable; continuing with cleaned names/manual overrides"
        )
        _ROLES_MISSING_LOGGED = True
    _ROLES_CACHE = pd.DataFrame(columns=["player_key", "player"])
    _ROLES_LOOKUP_CACHE = {}
    return _ROLES_CACHE


def strip_middle_initial(full_name: str) -> str:
    if not isinstance(full_name, str):
        return ""
    parts = [p.replace(".", "") for p in re.split(r"\s+", full_name.replace(",", " ").strip()) if p.strip()]
    if not parts:
        return ""
    first, last = parts[0], parts[-1]
    if last.lower() in SUFFIXES and len(parts) >= 3:
        last = parts[-2] + " " + parts[-1]
    return f"{first} {last}".strip()


def _strip_ourlads_noise(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    working = _TRAIL_TAG.sub("", name.strip())
    if "," in working:
        last, first = [p.strip() for p in working.split(",", 1)]
        working = f"{first} {last}".strip()
    return _SPACES.sub(" ", working.replace("’", "'")).strip()


def norm_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    for ch in (" ", "'", "-", "."):
        s = s.replace(ch, "")
    return s


def _build_roles_map_from_df(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {}
    cols = {c.lower(): c for c in df.columns}
    if "player_key" not in cols or "player" not in cols:
        raise ValueError("build_roles_map: roles CSV missing required columns")
    keys = df[cols["player_key"]].astype(str).map(norm_key)
    names = df[cols["player"]].astype(str).map(strip_middle_initial)
    return dict(zip(keys, names))


def build_roles_map_from_csv(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"[canonical_names] roles CSV not found at {p}")
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"[canonical_names] roles CSV at {p} is empty") from exc
    if df.empty:
        raise ValueError(f"[canonical_names] roles CSV at {p} has no rows")
    return _build_roles_map_from_df(df)


def _get_roles_lookup() -> dict[str, str]:
    global _ROLES_LOOKUP_CACHE
    if _ROLES_LOOKUP_CACHE is None:
        _ROLES_LOOKUP_CACHE = _build_roles_map_from_df(build_roles_map())
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


def canonicalize_player_name(source_key: str) -> str:
    if source_key is None:
        return ""
    cleaned = _strip_ourlads_noise(str(source_key))
    if not cleaned:
        return ""
    k = norm_key(cleaned)
    manual = build_manual_map()
    if k in manual:
        return manual[k]
    return _get_roles_lookup().get(k, cleaned)


canonicalize_name = canonicalize_player_name
_UNMAPPED_LOG = os.environ.get("UNMAPPED_NAME_LOG", "data/_debug/unmapped_names.jsonl")


def canonicalize_player_name_safe(raw: str) -> tuple[str, str]:
    try:
        canonical_name = str(canonicalize_player_name(raw) or "").strip()
        return canonical_name, norm_key(canonical_name) if canonical_name else ""
    except Exception as exc:
        fallback = "" if raw is None else str(raw).strip()
        logger.warning("[canonical_names] unexpected canonicalization error for %r; using raw name: %s", raw, exc)
        return fallback, norm_key(fallback)


def log_unmapped_variant(raw: str, where: str = "unknown", *args, **kwargs) -> None:
    return
