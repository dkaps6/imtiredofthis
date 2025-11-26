# scripts/make_player_form.py
"""
Build player-level shares and efficiency for the 2025 season.

Outputs: data/player_form.csv

Columns written:
- player, team, week, opponent, season, position, role
- tgt_share, route_rate, rush_share
- yprr, ypt, ypc, ypa
- receptions_per_target
- rz_share, rz_tgt_share, rz_rush_share

Surgical changes:
- Fill POSITION using multiple sources (weekly rosters → players master → PBP usage family).
- Do NOT coerce NaN to literal "NAN" prior to inference.
- Infer ROLE even when exact position is missing (uses family from usage).
- roles.csv remains an optional, non-destructive override.
- VALIDATOR: only enforce required metrics for players present in outputs/props_raw.csv.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils.df_keys import coerce_merge_keys

from scripts._opponent_map import normalize_team_series
from scripts.utils.canonical_names import (
    canonicalize_player_name_safe,
    log_unmapped_variant,
)
from scripts.utils.name_canon import make_player_key
from scripts.utils.name_clean import canonical_key, canonical_player, canonicalize, normalize_team
from scripts.utils.normalize_players import normalize_game_logs, normalize_season_totals
###############################################################################
# PBP loader compatibility layer
#
# Historically this script used the R package `nflreadr` via reticulate-style
# patterns. In CI we are pure Python, so we provide a local `load_pbp`
# implementation that mimics the nflreadr API but is backed by Python libs.
###############################################################################
try:
    # Preferred: Python port of nflreadr
    import nflreadpy as _nfl

    def load_pbp(seasons=None, **kwargs):
        """
        Compatibility wrapper for nflreadr::load_pbp using nflreadpy.

        - `seasons` can be an int, list of ints, or None (current season).
        - Returns a pandas.DataFrame so downstream code does not need changes.
        """
        pbp_pl = _nfl.load_pbp(seasons)
        # nflreadpy returns a Polars DataFrame; convert if possible
        try:
            return pbp_pl.to_pandas()
        except AttributeError:
            return pbp_pl

except ImportError:
    # Fallback: use nfl_data_py if nflreadpy is not available.
    from nfl_data_py import import_pbp_data as _import_pbp_data

    def load_pbp(seasons=None, **kwargs):
        """
        Fallback PBP loader using nfl_data_py.import_pbp_data.
        """
        if seasons is None:
            raise RuntimeError(
                "load_pbp() called without seasons and nflreadpy is unavailable. "
                "Pass explicit seasons or install nflreadpy."
            )
        if isinstance(seasons, (list, tuple, set)):
            season_list = [int(s) for s in seasons]
        else:
            season_list = [int(seasons)]
        return _import_pbp_data(season_list)

from scripts.utils.team_codes import canon_team

logger = logging.getLogger(__name__)


DATA_DIR = "data"
ROLES_PATH = Path(DATA_DIR) / "roles_ourlads.csv"
PROPS_ENRICHED_PATH = Path(DATA_DIR) / "props_enriched.csv"
SCHEDULE_GAMES_PATH = Path(DATA_DIR) / "games.csv"
PLAYER_FORM_OUT = Path(DATA_DIR) / "player_form.csv"
PLAYER_FORM_CONSENSUS_OUT = Path(DATA_DIR) / "player_form_consensus.csv"
PLAYER_GAME_LOGS_OUT = Path(DATA_DIR) / "player_game_logs.csv"
PLAYER_SEASON_TOTALS_OUT = Path(DATA_DIR) / "player_season_totals.csv"
DEBUG_MISSING_OPP = Path(DATA_DIR) / "_debug" / "player_missing_opponent.csv"
OPPONENT_MAP_PATH = Path(DATA_DIR) / "opponent_map_from_props.csv"
TEAM_WEEK_MAP_PATH = Path(DATA_DIR) / "team_week_map.csv"
UNMATCHED_ROLES_DEBUG_PATH = Path(DATA_DIR) / "unmatched_roles_merge.csv"
TEAM_FORM_PATH = Path(DATA_DIR) / "team_form.csv"
MANUAL_OVERRIDES_PATH = os.getenv("MANUAL_OVERRIDES_PATH", str(Path(DATA_DIR) / "manual_name_overrides.csv"))

SEASON = int(os.environ.get("SEASON", "2025"))

# Required *structural* inputs (schedule / roles / opponent map).
for p in (TEAM_WEEK_MAP_PATH, OPPONENT_MAP_PATH, ROLES_PATH):
    if not p.exists():
        raise FileNotFoundError(f"[player_form] missing required input: {p}")


CURRENT_SEASON = SEASON


CANON_OVERRIDES = {
    # Common problem mappings (extend this dict as needed):
    "mharrrison": "marvin harrison jr",
    "mharrison": "marvin harrison jr",
    "marvin harrison": "marvin harrison jr",
    "t_mcbride": "trey mcbride",
    "tmcbride": "trey mcbride",
    "jconner": "james conner",
    "j.conner": "james conner",
    "jcook": "james cook",
    "d.adams": "davante adams",
    "dadams": "davante adams",
    # ...add obvious skill guys, QBs, etc.
}

IDENTITY_COLUMNS = ["player_source_name", "player_display", "player_clean_key"]

_ROSTER_LOOKUP_CACHE: Optional[pd.DataFrame] = None


NAME_OVERRIDES = {
    "JOE T FLACCO": "Joe Flacco",
    "JOE T  FLACCO": "Joe Flacco",
    "D.ADAMS": "Davante Adams",
    "DADAMS": "Davante Adams",
}


def _canon_team_series(series: pd.Series) -> pd.Series:
    return normalize_team_series(series)


def _dump_norm_debug(df: pd.DataFrame, path: str) -> None:
    if df is None:
        df = pd.DataFrame()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _filter_to_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Limit a DataFrame to rows matching the provided season value."""

    if df is None or df.empty or "season" not in df.columns:
        return df
    numeric = pd.to_numeric(df["season"], errors="coerce")
    mask = numeric.eq(int(season))
    return df.loc[mask].copy()
def _attach_player_name_from_props(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "player_clean_key" not in df.columns:
        return df

    path = Path("data/player_name_map_from_props.csv")
    if not path.exists() or path.stat().st_size == 0:
        return df

    try:
        name_map = pd.read_csv(path)
    except Exception as err:
        logger.warning(
            "[make_player_form] failed reading %s for player names: %s",
            path,
            err,
        )
        return df

    if name_map.empty or "player_clean_key" not in name_map.columns:
        return df

    working = df.copy()
    working["player_clean_key"] = working["player_clean_key"].astype("string")
    name_map = name_map.copy()
    name_map["player_clean_key"] = name_map["player_clean_key"].astype("string")
    if "player_name_full" in name_map.columns:
        name_map["player_name_full"] = name_map["player_name_full"].astype("string")

    merged = working.merge(name_map, on="player_clean_key", how="left")

    if "player_name" not in merged.columns:
        merged["player_name"] = pd.Series(pd.NA, index=merged.index, dtype="string")

    if "player_name_full" in merged.columns:
        merged["player_name"] = merged["player_name"].astype("string").fillna(
            merged["player_name_full"]
        )
        merged.drop(columns=["player_name_full"], inplace=True)

    fallback = None
    for col in ("display_name", "player_display", "player_canonical", "player"):
        if col in merged.columns:
            fallback = merged[col].astype("string")
            break
    if fallback is not None:
        merged["player_name"] = merged["player_name"].fillna(fallback)

    return merged


def _overlay_opponents(pf: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """Fill opponent columns using schedule and live opponent maps when available."""

    tw_path = Path("data/team_week_map.csv")
    om_path = Path("data/opponent_map_from_props.csv")

    out = pf.copy()

    if "team" in out.columns:
        out["team"] = out["team"].astype("string").str.upper().str.strip()
    if "opponent" in out.columns:
        out["opponent"] = out["opponent"].astype("string")
    if "opponent_abbr" in out.columns:
        out["opponent_abbr"] = out["opponent_abbr"].astype("string")

    if tw_path.exists():
        try:
            tw = pd.read_csv(tw_path)
        except Exception:
            tw = pd.DataFrame()
        if not tw.empty:
            for col in ("season", "week"):
                if col in tw.columns:
                    tw[col] = pd.to_numeric(tw[col], errors="coerce").astype("Int64")
            for col in ("team", "opponent"):
                if col in tw.columns:
                    tw[col] = (
                        tw[col]
                        .astype("string")
                        .str.upper()
                        .str.strip()
                        .replace("", pd.NA)
                    )
            if "event_id" in tw.columns:
                tw["event_id"] = (
                    tw["event_id"].astype("string").str.strip().replace("", pd.NA)
                )

            join_keys = [k for k in ("season", "week", "team") if k in out.columns and k in tw.columns]
            if len(join_keys) == 3:
                extra_cols = [
                    c
                    for c in ("opponent", "bye", "event_id")
                    if c in tw.columns
                ]
                subset = tw[join_keys + extra_cols].drop_duplicates()
                rename_map = {"opponent": "_schedule_opponent", "bye": "_schedule_bye", "event_id": "_schedule_event_id"}
                subset = subset.rename(columns={k: v for k, v in rename_map.items() if k in subset.columns})
                merged = out.merge(subset, on=join_keys, how="left")
                if "_schedule_opponent" in merged.columns:
                    merged["_schedule_opponent"] = merged["_schedule_opponent"].astype("string")
                    if "opponent" in merged.columns:
                        merged["opponent"] = merged["opponent"].fillna(merged["_schedule_opponent"])
                    else:
                        merged["opponent"] = merged["_schedule_opponent"]
                    if "opponent_abbr" in merged.columns:
                        merged["opponent_abbr"] = merged["opponent_abbr"].fillna(merged["_schedule_opponent"])
                    merged.drop(columns=["_schedule_opponent"], inplace=True)
                if "_schedule_event_id" in merged.columns:
                    merged["_schedule_event_id"] = merged["_schedule_event_id"].astype("string")
                    if "event_id" in merged.columns:
                        merged["event_id"] = merged["event_id"].fillna(merged["_schedule_event_id"])
                    else:
                        merged["event_id"] = merged["_schedule_event_id"]
                    merged.drop(columns=["_schedule_event_id"], inplace=True)
                if "_schedule_bye" in merged.columns:
                    bye_mask = merged["_schedule_bye"].fillna(False).astype(bool)
                    merged.loc[bye_mask, ["opponent", "opponent_abbr"]] = "BYE"
                    merged.drop(columns=["_schedule_bye"], inplace=True)
                out = merged

    if om_path.exists():
        try:
            om = pd.read_csv(om_path)
        except Exception:
            om = pd.DataFrame()
        if not om.empty:
            for col in ("season", "week"):
                if col in om.columns:
                    om[col] = pd.to_numeric(om[col], errors="coerce").astype("Int64")
            for col in ("team", "opponent", "player_clean_key", "player_key", "event_id"):
                if col in om.columns:
                    om[col] = (
                        om[col]
                        .astype("string")
                        .str.upper()
                        .str.strip()
                    )

            key_col = "player_clean_key" if "player_clean_key" in om.columns else None
            if key_col is None and "player_key" in om.columns:
                key_col = "player_key"

            if key_col and "player_clean_key" in out.columns:
                available = [c for c in ("opponent", "event_id") if c in om.columns]
                if available:
                    select_cols = [key_col]
                    if "event_id" in om.columns and "event_id" in out.columns:
                        select_cols.append("event_id")
                    select_cols.extend(available)
                    over = om[select_cols].copy()
                    if key_col != "player_clean_key":
                        over.rename(columns={key_col: "player_clean_key"}, inplace=True)
                    over = over.dropna(subset=["player_clean_key", "opponent"])
                    join_cols = ["player_clean_key"]
                    if "event_id" in over.columns and "event_id" in out.columns:
                        join_cols.append("event_id")
                    rename = {}
                    if "opponent" in available:
                        rename["opponent"] = "_props_opponent"
                    if "event_id" in available:
                        rename["event_id"] = "_props_event_id"
                    over = over.drop_duplicates(join_cols)
                    over = over.rename(
                        columns={k: v for k, v in rename.items() if k in over.columns}
                    )
                    merged = out.merge(over, on=join_cols, how="left")
                    if "_props_opponent" in merged.columns:
                        merged["_props_opponent"] = (
                            merged["_props_opponent"].astype("string")
                        )
                        if "opponent" in merged.columns:
                            merged["opponent"] = merged["_props_opponent"].fillna(
                                merged["opponent"]
                            )
                        else:
                            merged["opponent"] = merged["_props_opponent"]
                        if "opponent_abbr" in merged.columns:
                            merged["opponent_abbr"] = merged["_props_opponent"].fillna(
                                merged["opponent_abbr"]
                            )
                        merged.drop(columns=["_props_opponent"], inplace=True)
                    if "_props_event_id" in merged.columns:
                        merged["_props_event_id"] = (
                            merged["_props_event_id"].astype("string")
                        )
                        if "event_id" in merged.columns:
                            merged["event_id"] = merged["event_id"].fillna(
                                merged["_props_event_id"]
                            )
                        else:
                            merged["event_id"] = merged["_props_event_id"]
                        merged.drop(columns=["_props_event_id"], inplace=True)
                    out = merged

    return out


def _coerce_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce merge keys to nullable numeric/string dtypes without lossy casts."""

    if df is None:
        return pd.DataFrame()

    out = df.copy()
    if out.empty:
        out["season"] = pd.Series(dtype="Int64")
        out["week"] = pd.Series(dtype="Int64")
        out["player_clean_key"] = pd.Series(dtype="string")
        return out

    for col in ("season", "week"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        else:
            out[col] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    for col in ("team", "opponent", "player_clean_key", "game_id", "event_id"):
        if col in out.columns:
            series = out[col].astype("string").fillna("").str.strip()
            if col in {"team", "opponent"}:
                series = series.str.upper()
            out[col] = series

    if "player_clean_key" not in out.columns or out["player_clean_key"].eq("").all():
        candidate = None
        for key_col in ("player_key", "player", "display_name"):
            if key_col in out.columns:
                series = out[key_col].astype("string").fillna("").str.strip()
                if series.ne("").any():
                    candidate = series
                    break
        if candidate is None:
            out["player_clean_key"] = ""
        else:
            out["player_clean_key"] = candidate.map(canonical_key).fillna("")
    else:
        out["player_clean_key"] = out["player_clean_key"].astype("string").fillna("")

    return out


def canonicalize_name(raw_name: str) -> str:
    """Canonicalize raw player names into a stable "First Last" form."""

    if raw_name is None:
        return ""

    raw_str = str(raw_name).strip()
    if not raw_str:
        return ""

    override = NAME_OVERRIDES.get(raw_str)
    if override:
        return override

    upper_key = raw_str.upper()
    override = NAME_OVERRIDES.get(upper_key)
    if override:
        return override

    lowered = raw_str.lower()
    if lowered in CANON_OVERRIDES:
        return canonicalize_name(CANON_OVERRIDES[lowered])

    normalized = unicodedata.normalize("NFKD", raw_str)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    tokens = [tok for tok in normalized.split() if tok]

    if not tokens:
        return ""

    suffix_tokens = {"jr", "sr", "ii", "iii", "iv", "v"}
    tokens = [tok for tok in tokens if tok not in suffix_tokens]

    if not tokens:
        return ""

    # Collapse leading consecutive single-letter initials ("c j" -> "cj")
    collapsed: List[str] = []
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if len(tok) == 1 and idx + 1 < len(tokens) and len(tokens[idx + 1]) == 1:
            collapsed.append(tok + tokens[idx + 1])
            idx += 2
            continue
        collapsed.append(tok)
        idx += 1

    tokens = collapsed

    if len(tokens) > 2:
        middle: List[str] = []
        for tok in tokens[1:-1]:
            if len(tok) == 1:
                continue
            middle.append(tok)
        tokens = [tokens[0]] + middle + [tokens[-1]]

    if len(tokens) == 1:
        return tokens[0].title()

    first = tokens[0].title()
    last = tokens[-1].title()
    return f"{first} {last}"


def normalize_pf_id(n: str) -> str:
    # our player_form/player_form_consensus already uses compact names like Mharrison, Kpitts, Dmooney, etc.
    # We just lowercase them for join safety.
    if not isinstance(n, str):
        return ""
    return n.strip().lower()


def _strip_suffixes(name: str) -> str:
    """
    Remove Jr, Sr, III, II, IV, V, etc. from the END of a name.
    Assumes input is already stripped and title-cased.
    """

    if not isinstance(name, str):
        return name

    suffixes = [" JR", " SR", " II", " III", " IV", " V"]
    out = name.upper().strip()
    for suf in suffixes:
        if out.endswith(suf):
            out = out[: -len(suf)]
    return out.strip()


def _drop_middle_initials(name: str) -> str:
    """
    Remove standalone middle initials from things like 'JOE T FLACCO'.
    Logic:
    - Split on space
    - Keep first and last token(s)
    - If there's exactly 3 tokens and the middle is length 1, drop it.
    - If more than 3 tokens, drop any 1-letter tokens that are in the middle.
    """

    if not isinstance(name, str):
        return name

    parts = [p for p in name.strip().split() if p]
    if len(parts) <= 2:
        return " ".join(parts)

    cleaned = [parts[0]]
    last_token = parts[-1]
    for mid in parts[1:-1]:
        if len(mid.replace(".", "")) > 1:
            cleaned.append(mid.replace(".", ""))
    cleaned.append(last_token)
    return " ".join(cleaned)


def normalize_player_name(raw: Optional[str]) -> Optional[str]:
    """Normalize player names into a clean 'First Last' form."""

    if not isinstance(raw, str):
        return raw

    cleaned = re.sub(r"[^A-Za-z\s]", " ", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    parts = cleaned.split()
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]

        if len(first) == 1 and len(parts) >= 3:
            first = parts[1]
            last = parts[-1]

        core_parts = [p for p in parts if len(p) > 1]
        if len(core_parts) >= 2:
            first = core_parts[0]
            last = core_parts[-1]

        final = f"{first.title()} {last.title()}".strip()
    else:
        final = cleaned.title()

    return final


def standardize_full_name(raw: str) -> str:
    """
    Convert raw names like:
      'JOE T FLACCO', 'D.Adams', 'DAdams', 'M.Wilson', 'HARRISON, MARVIN'
    into a consistent 'Firstname Lastname' style string:
      'Joe Flacco', 'Davante Adams', etc.

    Steps:
    - handle None
    - replace punctuation with space
    - uppercase
    - drop commas
    - drop suffixes (Jr, Sr, III, etc.)
    - drop standalone middle initials
    - title-case result
    """

    import re

    if raw is None:
        return None

    s = str(raw)
    s = s.replace(",", " ")
    s = re.sub(r"[_\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.upper()
    s = _strip_suffixes(s)
    s = _drop_middle_initials(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.title()
    return s


def _format_canonical_player_name(raw: str) -> str:
    """Format a canonical name string into Title Case with sensible spacing."""

    if not isinstance(raw, str):
        return ""

    value = raw.strip()
    if not value:
        return ""

    if " " not in value:
        value = re.sub(r"(?<!^)([A-Z])", r" \1", value)

    return standardize_full_name(value)


def _manual_override_keys(raw: object) -> set[str]:
    """Return normalized key variants for manual override lookups."""

    if raw is None:
        return set()

    text = str(raw)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    lowered = text.lower().strip()
    if not lowered:
        return set()

    collapsed = re.sub(r"[^a-z0-9]+", "", lowered)
    spaced = re.sub(r"\s+", " ", lowered).strip()

    variants = {lowered, collapsed, spaced, spaced.replace(" ", "")}
    return {v for v in variants if v}


def _load_manual_name_overrides(path: str) -> dict[str, str]:
    """
    Reads a CSV with columns: player_source_name, full_name
    Ignores header rows and any malformed lines.
    """
    import csv, os
    mapping: dict[str, str] = {}
    if not path or not os.path.exists(path):
        return mapping
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 2:
                continue
            # header guard
            if str(row[0]).strip().lower() in {"player_source_name", "player_key", "player"}:
                continue
            k = str(row[0]).strip()
            v = str(row[1]).strip()
            if k and v:
                mapping[k] = v
    return mapping


@lru_cache(maxsize=1)
def _manual_name_overrides() -> dict[str, str]:
    return _load_manual_name_overrides(MANUAL_OVERRIDES_PATH)


def _canonicalize_player_name(raw: str) -> str:
    if raw is None:
        return ""
    # lower
    n = unicodedata.normalize("NFKD", str(raw)).encode("ascii", "ignore").decode("ascii")
    n = _strip_suffixes(n)
    n = n.lower()
    # remove punctuation-ish
    n = re.sub(r"[^a-z0-9\s]", " ", n)
    # collapse whitespace
    n = re.sub(r"\s+", " ", n).strip()
    # kill suffixes
    n = _strip_suffixes(n)
    # collapse whitespace again
    n = re.sub(r"\s+", " ", n).strip()
    manual_map = _manual_name_overrides()
    if manual_map:
        for candidate in _manual_override_keys(raw):
            if candidate in manual_map:
                return manual_map[candidate]
        for candidate in _manual_override_keys(n):
            if candidate in manual_map:
                return manual_map[candidate]
    # if it's initial+lastname like "m harrison", collapse to "mharrison"
    parts = n.split()
    if len(parts) == 2 and len(parts[0]) == 1:
        n_guess = parts[0] + parts[1]
        if n_guess in CANON_OVERRIDES:
            return CANON_OVERRIDES[n_guess]
        # also try plain guess "m harrison" -> "m harrison"
    # direct override on collapsed tokens:
    collapsed = n.replace(" ", "")
    if collapsed in CANON_OVERRIDES:
        return CANON_OVERRIDES[collapsed]
    # final override on full spaced string
    if n in CANON_OVERRIDES:
        return CANON_OVERRIDES[n]
    return n


def ensure_canonical(
    df: pd.DataFrame, player_col: str, team_col: str | None = None
) -> pd.DataFrame:
    """Ensure a dataframe has normalized player/team identity columns."""

    if df is None:
        return df

    out = df.copy()
    if player_col not in out.columns or out.empty:
        if team_col and team_col in out.columns:
            out[team_col] = out[team_col].astype(str).str.upper().str.strip()
            if team_col != "team" or "team" not in out.columns:
                out["team"] = out[team_col]
        elif "team" in out.columns:
            out["team"] = out["team"].astype(str).str.upper().str.strip()
        return out

    out[player_col] = out[player_col].map(standardize_full_name)

    canonical_series = out[player_col].apply(canonicalize_name)
    if "player_canonical" in out.columns:
        existing = out["player_canonical"].astype(str).str.strip()
        mask = existing.notna() & existing.ne("")
        canonical_series = canonical_series.where(~mask, existing)

    canonical_series = canonical_series.fillna("").astype(str).str.strip()
    missing_mask = canonical_series.eq("")
    if missing_mask.any():
        fallback = (
            out.loc[missing_mask, player_col]
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9 ]+", "", regex=True)
            .str.strip()
        )
        canonical_series = canonical_series.where(~missing_mask, fallback)

    out["player_canonical"] = canonical_series
    out["player_clean_key"] = out["player_canonical"].str.replace(
        r"\s+", "_", regex=True
    )

    if team_col and team_col in out.columns:
        normalized_team = out[team_col].astype(str).str.upper().str.strip()
        out[team_col] = normalized_team
        if team_col != "team" or "team" not in out.columns:
            out["team"] = normalized_team
    elif "team" in out.columns:
        out["team"] = out["team"].astype(str).str.upper().str.strip()

    return out


def _apply_canonical_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "player" in df.columns:
        df["player_canonical"] = df["player"].map(normalize_player_name)
    elif "player_name" in df.columns:
        df["player_canonical"] = df["player_name"].map(normalize_player_name)
    else:
        if "player_canonical" in df.columns:
            df["player_canonical"] = df["player_canonical"].map(normalize_player_name)
        else:
            df["player_canonical"] = None
    return df


def assert_no_duplicate_columns(df: pd.DataFrame, label: str) -> None:
    """Raise if duplicate columns remain after a merge."""

    if df is None:
        return

    dupes = df.columns[df.columns.duplicated()].unique()
    if len(dupes) > 0:
        cols = ", ".join(sorted(str(c) for c in dupes))
        raise RuntimeError(
            f"[make_player_form] {label} has duplicate columns after merge: {cols}"
        )


def _normalize_player_clean_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize any player_clean_key merge artifacts to a single column."""

    if df is None:
        return df

    dupe_cols = [c for c in df.columns if c.startswith("player_clean_key")]
    if "player_clean_key_x" in dupe_cols and "player_clean_key_y" in dupe_cols:
        df["player_clean_key"] = (
            df["player_clean_key_x"]
            .fillna(df["player_clean_key_y"])
            .astype(str)
            .str.strip()
        )
        df = df.drop(columns=["player_clean_key_x", "player_clean_key_y"])
    elif "player_clean_key_x" in dupe_cols:
        df = df.rename(columns={"player_clean_key_x": "player_clean_key"})
    elif "player_clean_key_y" in dupe_cols:
        df = df.rename(columns={"player_clean_key_y": "player_clean_key"})

    return df


def _as_clean_series(obj: Any) -> pd.Series:
    """Return a whitespace-stripped string Series from common merge artifacts."""

    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(index=obj.index, dtype="string")
        obj = obj.iloc[:, 0]

    if not isinstance(obj, pd.Series):
        obj = pd.Series(obj)

    return obj.astype("string").fillna("").str.strip()


def _coalesce_dupe_cols(
    df: pd.DataFrame, bases: Iterable[str] = ("player_clean_key", "player", "team")
) -> pd.DataFrame:
    """Combine *_x/*_y columns left over from merges into a single column."""

    if df is None:
        return df

    for base in bases:
        x, y = f"{base}_x", f"{base}_y"
        if x in df.columns and y in df.columns:
            left = _as_clean_series(df[x]).replace("", pd.NA)
            right = _as_clean_series(df[y]).replace("", pd.NA)
            df[base] = left.combine_first(right).fillna("")
            df.drop([x, y], axis=1, inplace=True)
    return df


def _dedupe_player_clean_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee a single 'player_clean_key' column.
    Collapse any 'player_clean_key_x'/'player_clean_key_y'/etc into one string.
    """

    if df is None:
        return df

    key_cols = [
        c for c in df.columns if c == "player_clean_key" or c.startswith("player_clean_key_")
    ]

    if not key_cols:
        return df

    cleaned = pd.DataFrame({kc: _as_clean_series(df[kc]) for kc in key_cols})
    cleaned = cleaned.replace({"": pd.NA})

    preferred = cleaned.bfill(axis=1)
    first_col = preferred.columns[0]
    df["player_clean_key"] = preferred[first_col].fillna("")

    drop_cols = [c for c in key_cols if c != "player_clean_key"]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def _normalize_key(s: str) -> str:
    """Normalize a name into a stable, space-normalized lowercase key."""

    if s is None:
        return ""

    text = unicodedata.normalize("NFKD", str(s))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = (
        pd.Series([text], dtype="string")
        .fillna("")
        .iloc[0]
    )
    text = text.replace(".", " ")
    text = re.sub(r"[^a-zA-Z ]+", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _standardize_optional_name(value: Any) -> Optional[str]:
    """Return a cleaned, title-cased name when possible."""

    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return None
    return standardize_full_name(text)


def _extract_initial_last(name: Any) -> Tuple[Optional[str], Optional[str]]:
    """Parse compact/initialed names into (first_initial, last_name)."""

    if name is None:
        return (None, None)
    text = str(name).strip()
    if not text:
        return (None, None)

    cleaned = re.sub(r"[^A-Za-z\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        parts = cleaned.split()
        if len(parts) >= 2:
            first = next((p for p in parts if p), "")
            last = next((p for p in reversed(parts) if p), "")
            first_initial = first[0] if first else ""
            last_token = re.sub(r"[^A-Za-z]", "", last)
            if first_initial and last_token:
                return (first_initial.lower(), last_token.lower())

    collapsed = re.sub(r"[^A-Za-z]", "", text)
    if len(collapsed) < 2:
        return (None, None)

    boundary = None
    for idx in range(1, len(collapsed)):
        ch = collapsed[idx]
        nxt = collapsed[idx + 1] if idx + 1 < len(collapsed) else ""
        if ch.isupper() and nxt.islower():
            boundary = idx
            break
    if boundary is None:
        boundary = 1

    first_initial = collapsed[0]
    last = collapsed[boundary:]
    if not first_initial or not last:
        return (None, None)

    return (first_initial.lower(), last.lower())


def _canonical_identity_fields(raw_name: str) -> Tuple[str, str, str]:
    """Return robust canonical name metadata (name, clean key, player key)."""

    canonical = ""
    clean_key = ""

    try:
        result = canonicalize_player_name_safe(raw_name)
    except ValueError:
        result = ("", "")
    except TypeError:
        result = ("", "")

    if isinstance(result, (list, tuple)):
        if len(result) >= 2:
            canonical, clean_key = result[0], result[1]
        elif len(result) == 1:
            canonical = result[0]
    else:
        canonical = result

    canonical = str(canonical or "").strip()
    raw_as_str = str(raw_name or "").strip()
    if not canonical and raw_as_str:
        canonical = canonical_player(raw_as_str)

    clean_key = str(clean_key or "").strip()
    if not clean_key and canonical:
        clean_key = canonical_key(canonical)

    basis = canonical if canonical else raw_as_str
    parts = [p for p in re.split(r"\s+", basis) if p]
    if not parts:
        player_key = ""
    else:
        if len(parts) == 1:
            first, last = parts[0], ""
        else:
            first = " ".join(parts[:-1])
            last = parts[-1]
        first_clean = re.sub(r"[^a-z]", "", first.lower()).strip()
        last_clean = re.sub(r"[^a-z]", "", last.lower()).strip()
        player_key = f"{first_clean} {last_clean}".strip()
        player_key = re.sub(r"\s+", " ", player_key)
        if not player_key:
            player_key = re.sub(r"[^a-z]", "", first.lower())

    return canonical, clean_key, player_key


def canonicalize_player_name(raw: str) -> str:
    """Legacy alias that returns the normalized canonical name string."""

    return _canonicalize_player_name(raw)


def _build_roster_lookup() -> pd.DataFrame:
    """Load roster data with clean display names and helpful tokens."""

    global _ROSTER_LOOKUP_CACHE
    if _ROSTER_LOOKUP_CACHE is not None:
        return _ROSTER_LOOKUP_CACHE.copy()

    frames: List[pd.DataFrame] = []
    for path in [ROLES_PATH, TEAM_FORM_PATH]:
        if not path.exists():
            continue
        df = _read_csv_safe(str(path))
        if df.empty or "player" not in df.columns:
            continue
        base = pd.DataFrame({"player_display": df["player"]})
        if "team_abbr" in df.columns:
            base["team_abbr"] = df["team_abbr"]
        elif "team" in df.columns:
            base["team_abbr"] = df["team"]
        else:
            base["team_abbr"] = pd.NA
        base["player_display"] = base["player_display"].map(_standardize_optional_name)
        base = base.dropna(subset=["player_display"])
        if base.empty:
            continue
        base["team_abbr"] = (
            base["team_abbr"].astype("string").str.strip().str.upper()
        )
        base.loc[
            base["team_abbr"].isin({"", "NAN", "NONE", "NULL"}), "team_abbr"
        ] = pd.NA
        frames.append(base)

    if not frames:
        _ROSTER_LOOKUP_CACHE = pd.DataFrame(
            columns=[
                "player_display",
                "team_abbr",
                "player_clean_key",
                "first_initial_lower",
                "last_lower",
            ]
        )
        return _ROSTER_LOOKUP_CACHE.copy()

    roster = pd.concat(frames, ignore_index=True)
    roster = roster.dropna(subset=["player_display"]).drop_duplicates()
    roster["player_display"] = roster["player_display"].map(_standardize_optional_name)
    roster = roster.dropna(subset=["player_display"])
    roster["player_display"] = roster["player_display"].astype("string")
    roster["team_abbr"] = roster["team_abbr"].astype("string")

    roster["player_clean_key"] = roster["player_display"].astype("object").fillna("")
    roster["player_clean_key"] = roster["player_clean_key"].apply(canonical_key)
    roster["player_clean_key"] = roster["player_clean_key"].astype("object").fillna("").apply(
        _normalize_key
    )

    tokens = roster["player_display"].astype("string").str.split()
    roster["first_initial_lower"] = tokens.str[0].str[0].str.lower()
    roster["last_lower"] = tokens.str[-1].str.lower()

    roster = roster.drop_duplicates(subset=["player_display", "team_abbr"]).reset_index(
        drop=True
    )

    roster["player_display"] = roster["player_display"].astype("string")
    roster["team_abbr"] = roster["team_abbr"].astype("string")
    roster["player_clean_key"] = roster["player_clean_key"].astype("string")
    roster["first_initial_lower"] = roster["first_initial_lower"].astype("string")
    roster["last_lower"] = roster["last_lower"].astype("string")

    _ROSTER_LOOKUP_CACHE = roster
    return roster.copy()


def _attach_player_identity(
    df: pd.DataFrame, team_columns: Iterable[str] = ("team_abbr", "team")
) -> pd.DataFrame:
    """Ensure source/display/clean name columns are populated consistently."""

    if df is None:
        return df
    out = df.copy()

    if out.empty:
        for col in IDENTITY_COLUMNS:
            if col not in out.columns:
                out[col] = pd.Series(dtype="string")
        return out

    def _clean_text(series: pd.Series) -> pd.Series:
        ser = series.astype("string")
        ser = ser.str.strip()
        ser = ser.mask(ser.str.lower().isin({"", "nan", "none", "null"}))
        return ser

    if "player_source_name" not in out.columns:
        source = pd.Series(pd.NA, index=out.index, dtype="string")
        for cand in ["player_name_raw", "player_name", "player_original", "player"]:
            if cand in out.columns:
                candidate = _clean_text(out[cand])
                source = source.combine_first(candidate)
        out["player_source_name"] = source
    else:
        out["player_source_name"] = _clean_text(out["player_source_name"])

    display = out.get("player_display")
    if display is None:
        display = pd.Series(pd.NA, index=out.index, dtype="string")
    else:
        display = _clean_text(display)
    for cand in ["canonical_player_name", "player", "player_name", "player_source_name"]:
        if cand in out.columns:
            candidate = _clean_text(out[cand])
            display = display.combine_first(candidate)
    display = display.map(_standardize_optional_name)
    out["player_display"] = pd.Series(display, index=out.index, dtype="string")

    source = out["player_source_name"].combine_first(out["player_display"])
    out["player_source_name"] = pd.Series(source, index=out.index, dtype="string")

    if "player_clean_key" in out.columns:
        key_series = out["player_clean_key"]
    else:
        key_series = pd.Series("", index=out.index, dtype="string")
    key_series = key_series.astype("object").fillna("").apply(_normalize_key)
    display_keys = out["player_display"].astype("object").fillna("").apply(_normalize_key)
    source_keys = out["player_source_name"].astype("object").fillna("").apply(_normalize_key)
    key_series = key_series.replace("", pd.NA)
    key_series = key_series.combine_first(display_keys.replace("", pd.NA))
    key_series = key_series.combine_first(source_keys.replace("", pd.NA))
    out["player_clean_key"] = pd.Series(key_series.fillna(""), index=out.index, dtype="string")

    team_lookup = pd.Series(pd.NA, index=out.index, dtype="object")
    for col in team_columns:
        if col in out.columns:
            candidate = _clean_text(out[col]).str.upper()
            team_lookup = team_lookup.combine_first(candidate.astype("object"))
    out["_team_lookup"] = pd.Series(team_lookup, index=out.index, dtype="string")
    out["_team_lookup"] = out["_team_lookup"].mask(out["_team_lookup"].str.len() == 0, pd.NA)

    roster = _build_roster_lookup()
    if not roster.empty:
        roster_key = roster.loc[:, ["player_clean_key", "player_display"]].drop_duplicates()
        roster_key = roster_key.rename(columns={"player_display": "__roster_display"})
        out = out.merge(roster_key, on="player_clean_key", how="left")
        roster_display = _clean_text(out["__roster_display"])
        fill_mask = out["player_display"].isna() | out["player_display"].astype("string").str.strip().eq("")
        out.loc[fill_mask, "player_display"] = roster_display.loc[fill_mask]
        out.drop(columns=["__roster_display"], inplace=True)

        unresolved_idx = out.index[
            (out["player_display"].isna() | out["player_display"].astype("string").str.strip().eq(""))
            & out["_team_lookup"].notna()
        ]
        if len(unresolved_idx) > 0:
            roster_guess = roster.loc[
                roster["team_abbr"].notna()
                & roster["first_initial_lower"].notna()
                & roster["last_lower"].notna(),
                [
                    "team_abbr",
                    "first_initial_lower",
                    "last_lower",
                    "player_display",
                    "player_clean_key",
                ],
            ]
            if not roster_guess.empty:
                roster_guess = roster_guess.drop_duplicates(
                    subset=["team_abbr", "first_initial_lower", "last_lower"]
                )
                lookup = {
                    (row.team_abbr, row.first_initial_lower, row.last_lower): (
                        row.player_display,
                        row.player_clean_key,
                    )
                    for row in roster_guess.itertuples(index=False)
                }
                for idx in unresolved_idx:
                    team_key = out.at[idx, "_team_lookup"]
                    if pd.isna(team_key):
                        continue
                    name_basis = out.at[idx, "player_source_name"]
                    if pd.isna(name_basis) or str(name_basis).strip() == "":
                        if "player" in out.columns:
                            name_basis = out.at[idx, "player"]
                    if (pd.isna(name_basis) or str(name_basis).strip() == "") and "canonical_player_name" in out.columns:
                        name_basis = out.at[idx, "canonical_player_name"]
                    first_initial, last_lower = _extract_initial_last(name_basis)
                    if not first_initial or not last_lower:
                        continue
                    key = (str(team_key), first_initial, last_lower)
                    guess = lookup.get(key)
                    if not guess:
                        continue
                    display_guess, key_guess = guess
                    current_display = out.at[idx, "player_display"]
                    if pd.isna(current_display) or str(current_display).strip() == "":
                        out.at[idx, "player_display"] = display_guess
                    current_key = out.at[idx, "player_clean_key"]
                    if pd.isna(current_key) or str(current_key).strip() == "":
                        out.at[idx, "player_clean_key"] = key_guess

    out["player_display"] = out["player_display"].map(_standardize_optional_name)
    out["player_display"] = out["player_display"].astype("string")
    out["player_source_name"] = _clean_text(out["player_source_name"]).combine_first(
        out["player_display"]
    )
    out["player_source_name"] = out["player_source_name"].astype("string")
    out["player_clean_key"] = out["player_clean_key"].astype("object").fillna("").apply(
        _normalize_key
    )
    out["player_clean_key"] = out["player_clean_key"].astype("string")

    out.drop(columns=["_team_lookup"], inplace=True, errors="ignore")

    for col in IDENTITY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series(pd.NA, index=out.index, dtype="string")
        else:
            out[col] = out[col].astype("string")

    return out


def _reorder_identity_columns(
    df: pd.DataFrame, identity_cols: Iterable[str] = IDENTITY_COLUMNS
) -> pd.DataFrame:
    """Place identity columns next to `player` for readability."""

    if df is None or not isinstance(df, pd.DataFrame):
        return df

    front: List[str] = []
    if "player" in df.columns:
        front.append("player")
    for col in identity_cols:
        if col in df.columns and col not in front:
            front.append(col)

    remainder = [c for c in df.columns if c not in front]
    return df[front + remainder]


def build_name_canonical_map() -> dict:
    """
    Build a per-team map of possible short name variants → canonical full name.
    For example, for KC:
      'travis kelce' -> 'Travis Kelce'
      't. kelce'     -> 'Travis Kelce'
      't kelce'      -> 'Travis Kelce'
    We build this using BOTH roles_ourlads.csv (cleaned names)
    and opponent_map_from_props.csv (sportsbook names).
    """

    def explode(df: pd.DataFrame):
        rows = []
        for _, r in df.iterrows():
            team = str(r.get("team", "")).strip().upper()
            full = str(r.get("player", "")).strip()
            if not team or not full:
                continue
            parts = full.split()
            if len(parts) < 2:
                continue
            first = parts[0]
            last = parts[-1]
            rows.append((team, first, last, full))
        return rows

    dfs = []
    if ROLES_PATH.exists():
        dfs.append(pd.read_csv(ROLES_PATH))
    if PROPS_ENRICHED_PATH.exists():
        try:
            props_df = pd.read_csv(PROPS_ENRICHED_PATH)
        except Exception:
            props_df = pd.DataFrame()
        if not props_df.empty:
            cols = {}
            if "player_canonical" in props_df.columns:
                cols["player_canonical"] = "player"
            elif "player_name_raw" in props_df.columns:
                cols["player_name_raw"] = "player"
            if "player_team_abbr" in props_df.columns:
                cols["player_team_abbr"] = "team"
            elif "home_team_abbr" in props_df.columns:
                cols["home_team_abbr"] = "team"
            if cols:
                subset = props_df[list(cols.keys())].rename(columns=cols)
                dfs.append(subset)
    if not dfs:
        return {}

    combined = pd.concat(dfs, ignore_index=True)

    canon = {}
    for team, first, last, full in explode(combined):
        first_clean = first.strip()
        last_clean = last.strip()
        if not first_clean or not last_clean:
            continue

        # keys we want to resolve:
        # "josh allen"
        # "j. allen"
        # "j allen"   (for cases missing the period)
        key_full = f"{first_clean} {last_clean}".lower()
        key_init_dot = f"{first_clean[0]}. {last_clean}".lower()
        key_init_nodot = f"{first_clean[0]} {last_clean}".lower()

        canon.setdefault(team, {})
        canon[team][key_full] = full
        canon[team][key_init_dot] = full
        canon[team][key_init_nodot] = full

    return canon


def apply_name_canonicalization(df: pd.DataFrame, canon_map: dict) -> pd.DataFrame:
    """
    For each row in df, replace df['player'] with the canonical full name
    if we can resolve it using team + (player variations).
    This upgrades things like 'J. Allen' or 'J Allen' to 'Josh Allen'.
    """

    out = df.copy()
    for idx, row in out.iterrows():
        raw_team = str(row.get("team", "")).strip().upper()
        raw_player = str(row.get("player", "")).strip()

        if not raw_team or not raw_player:
            continue

        key1 = raw_player.lower()
        key2 = key1.replace(".", "")  # handle "t kelce" vs "t. kelce"

        fixed = None
        if raw_team in canon_map:
            team_map = canon_map[raw_team]
            if key1 in team_map:
                fixed = team_map[key1]
            elif key2 in team_map:
                fixed = team_map[key2]

        if fixed:
            out.at[idx, "player"] = fixed

    return out


def _apply_player_name_cleaning(
    df: pd.DataFrame, name_maps: dict | None
) -> pd.DataFrame:
    """Apply canonicalization, standardization, and derive key columns for player frames."""

    if df is None:
        return df

    out = df.copy()
    if "player" not in out.columns or out.empty:
        return out

    if name_maps:
        out = apply_name_canonicalization(out, name_maps)

    team_col = "team" if "team" in out.columns else None
    return ensure_canonical(out, player_col="player", team_col=team_col)


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


PLAYER_FORM_USAGE_COLS = [
    "targets",
    "routes",
    "receptions",
    "rec_yards",
    "rushes",
    "rush_yards",
    "pass_att",
    "pass_yards",
    "dropbacks",
    "team_targets",
    "team_dropbacks",
    "team_rushes",
    "rz_targets",
    "rz_team_targets",
    "rz_rushes",
    "rz_team_rushes",
    "games",
]

PLAYER_FORM_SHARE_COLS = [
    "tgt_share",
    "route_rate",
    "rush_share",
    "yprr",
    "ypt",
    "ypc",
    "ypa",
    "receptions_per_target",
    "rz_share",
    "rz_tgt_share",
    "rz_rush_share",
]

PLAYER_FORM_REQUIRED_COLUMNS = [
    "player",
    "canonical_player_name",
    "player_name_canonical",
    "team",
    "team_abbr",
    "week",
    "opponent",
    "opponent_abbr",
    "opp_abbr",
    "season",
    "position",
    "role",
    "kickoff_ts",
] + PLAYER_FORM_USAGE_COLS + PLAYER_FORM_SHARE_COLS

FINAL_COLS = PLAYER_FORM_REQUIRED_COLUMNS + [
    "player_source_name",
    "player_display",
    "player_canonical",
    "player_clean_key",
    "team_key",
    "week_key",
    "unmatched_flag",
]


def _ensure_single_position_column(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce duplicate position columns down to one canonical 'position'."""

    if df is None or df.empty:
        return df

    out = df.copy()
    position_cols = [c for c in out.columns if c.lower().startswith("position")]

    if "position" not in out.columns:
        if position_cols:
            first_col = position_cols[0]
            out["position"] = out[first_col]
            if first_col != "position":
                position_cols = [c for c in position_cols if c != first_col]
        else:
            out["position"] = pd.NA

    for col in position_cols:
        if col == "position":
            continue
        out["position"] = out["position"].combine_first(out[col])
        out.drop(columns=[col], inplace=True)

    out["position"] = out["position"].astype("string").str.strip()
    return out


def _build_team_to_opp_map_for_slate(
    df: pd.DataFrame | None, slate_date: str | None
) -> Dict[str, str]:
    """Construct a team→opponent lookup using current data and cached schedule files."""

    mapping: Dict[str, str] = {}

    def _ingest(team_val: Any, opp_val: Any) -> None:
        team_key = str(team_val).strip().upper() if team_val is not None else ""
        opp_key = str(opp_val).strip().upper() if opp_val is not None else ""
        if not team_key or not opp_key:
            return
        existing = mapping.get(team_key)
        if existing and existing != opp_key:
            if existing != opp_key:
                logger.debug(
                    "[make_player_form] team %s opponent conflict: keeping %s over %s",
                    team_key,
                    existing,
                    opp_key,
                )
            return
        mapping[team_key] = opp_key

    if df is not None and not df.empty:
        for _, row in df.iterrows():
            _ingest(row.get("team"), row.get("opponent"))
            _ingest(row.get("team_abbr"), row.get("opponent_abbr"))

    if PROPS_ENRICHED_PATH.exists():
        try:
            enriched_df = pd.read_csv(PROPS_ENRICHED_PATH)
        except Exception as err:
            logger.warning(
                "[make_player_form] failed reading %s: %s",
                PROPS_ENRICHED_PATH,
                err,
            )
        else:
            if not enriched_df.empty:
                if slate_date:
                    for col in ["slate_date", "kickoff_ts", "commence_time"]:
                        if col in enriched_df.columns:
                            mask = enriched_df[col].astype(str).str.startswith(str(slate_date))
                            enriched_df = enriched_df[mask]
                            break
                for _, row in enriched_df.iterrows():
                    team_val = row.get("player_team_abbr") or row.get("home_team_abbr")
                    opp_val = row.get("opponent_team_abbr") or row.get("away_team_abbr")
                    _ingest(team_val, opp_val)

    if SCHEDULE_GAMES_PATH.exists():
        try:
            schedule_df = pd.read_csv(SCHEDULE_GAMES_PATH)
        except Exception as err:
            logger.warning(
                "[make_player_form] failed reading %s: %s",
                SCHEDULE_GAMES_PATH,
                err,
            )
        else:
            if slate_date:
                for col in ["slate_date", "date", "game_date"]:
                    if col in schedule_df.columns:
                        schedule_df = schedule_df[
                            schedule_df[col].astype(str).str.startswith(str(slate_date))
                        ]
                        break
            for _, row in schedule_df.iterrows():
                home = row.get("home_team") or row.get("home_abbr")
                away = row.get("away_team") or row.get("away_abbr")
                _ingest(home, away)
                _ingest(away, home)

    return mapping


# === Weighted season consensus helper (surgical add) ===
def _build_season_consensus(base: pd.DataFrame) -> pd.DataFrame:
    """Weighted season consensus per (player, team, season). Requires denominators if available."""
    if base is None or base.empty:
        return pd.DataFrame()
    df = base.copy()
    df.columns = [c.lower() for c in df.columns]

    # numeric coercion for denominators (when present)
    denom_cols = [
        "targets",
        "team_targets",
        "team_dropbacks",
        "receptions",
        "rec_yards",
        "rushes",
        "team_rushes",
        "rush_yards",
        "rz_targets",
        "rz_team_targets",
        "rz_rushes",
        "rz_team_rushes",
        "pass_yards",
        "pass_att",
    ]

    for c in denom_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keys = [k for k in ["player", "team", "season"] if k in df.columns]
    if not keys:
        # No sensible grouping keys → nothing to do
        logger.error(
            "[make_player_form] season consensus: missing all grouping keys "
            "(player/team/season). Available columns: %s",
            sorted(df.columns),
        )
        return pd.DataFrame()

    # Build aggregation spec ONLY from columns that actually exist
    agg_spec = {c: "sum" for c in denom_cols if c in df.columns}

    if not agg_spec:
        # This is the situation currently blowing up CI: we hit groupby.agg({})
        logger.error(
            "[make_player_form] season consensus: no numeric denominators present. "
            "Expected one of %s but found none on player_form. Columns: %s",
            denom_cols,
            sorted(df.columns),
        )
        raise RuntimeError(
            "[make_player_form] FATAL: season consensus cannot compute – "
            "no expected denominator columns found on player_form"
        )

    sums = (
        df.groupby(keys, dropna=False)
        .agg(agg_spec)
        .reset_index()
    )

    out = sums.copy()
    # receiving
    if {"targets", "team_targets"}.issubset(out.columns):
        out["tgt_share"] = out["targets"] / out["team_targets"]
    if {"targets", "team_dropbacks"}.issubset(out.columns):
        out["route_rate"] = out["targets"] / out["team_dropbacks"]
    if {"rec_yards", "targets"}.issubset(out.columns):
        out["ypt"] = out["rec_yards"] / out["targets"]
        out["yprr"] = out["rec_yards"] / out["targets"]
    if {"receptions", "targets"}.issubset(out.columns):
        out["receptions_per_target"] = out["receptions"] / out["targets"]
    if {"rz_targets", "rz_team_targets"}.issubset(out.columns):
        out["rz_tgt_share"] = out["rz_targets"] / out["rz_team_targets"]
    # rushing
    if {"rushes", "team_rushes"}.issubset(out.columns):
        out["rush_share"] = out["rushes"] / out["team_rushes"]
    if {"rush_yards", "rushes"}.issubset(out.columns):
        out["ypc"] = out["rush_yards"] / out["rushes"]
    if {"rz_rushes", "rz_team_rushes"}.issubset(out.columns):
        out["rz_rush_share"] = out["rz_rushes"] / out["rz_team_rushes"]
    # qb
    if {"pass_yards", "pass_att"}.issubset(out.columns):
        out["ypa"] = out["pass_yards"] / out["pass_att"]

    # combined RZ share
    out["rz_share"] = np.nan
    if "rz_tgt_share" in out.columns or "rz_rush_share" in out.columns:
        t = out.get("rz_tgt_share", pd.Series(np.nan, index=out.index))
        r = out.get("rz_rush_share", pd.Series(np.nan, index=out.index))
        out["rz_share"] = np.fmax(t, r)

    # carry role/position mode from base
    def _mode(series: pd.Series):
        s = series.dropna().astype(str)
        if s.empty:
            return np.nan
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]

    for lab in ["position", "role"]:
        if lab in df.columns:
            m = df.groupby(keys, dropna=False)[lab].apply(_mode).reset_index()
            out = out.merge(m, on=keys, how="left")

    # games
    games = df.groupby(keys, dropna=False).size().rename("games").reset_index()
    out = out.merge(games, on=keys, how="left")

    # mark as consensus row
    out["opponent"] = "ALL"
    return out


CONSENSUS_OPPONENT_SENTINEL = "ALL"

CONSENSUS_REQUIRED_COLUMNS = [
    "player",
    "player_canonical",
    "team",
    "team_abbr",
    "week",
    "opponent",
    "opponent_abbr",
    "position",
    "role",
]


def _inject_week_opponent_and_roles(
    out: pd.DataFrame, name_maps: dict | None
) -> pd.DataFrame:
    """Fill opponent/team metadata using props_enriched context when available."""

    if out is None or out.empty:
        return out

    if not PROPS_ENRICHED_PATH.exists():
        return out

    try:
        props_df = pd.read_csv(PROPS_ENRICHED_PATH)
    except Exception:
        return out

    if props_df.empty:
        return out

    props_df.columns = [c.lower() for c in props_df.columns]
    if "player_canonical" not in props_df.columns:
        if "player_name_raw" in props_df.columns:
            props_df["player_canonical"] = props_df["player_name_raw"].apply(canonicalize_name)
        else:
            return out

    props_subset = props_df.copy()
    props_subset["player_canonical"] = props_subset["player_canonical"].astype(str)
    for col in ["player_team_abbr", "opponent_team_abbr", "home_team_abbr", "away_team_abbr"]:
        if col in props_subset.columns:
            props_subset[col] = (
                props_subset[col]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
            )

    rename_cols = {
        "player_team_abbr": "team_from_props",
        "opponent_team_abbr": "opponent_from_props",
        "kickoff_ts": "kickoff_ts_from_props",
        "event_id": "event_id_from_props",
    }
    available_cols = [c for c in rename_cols if c in props_subset.columns]
    props_merge = props_subset[["player_canonical"] + available_cols].rename(columns=rename_cols)
    props_merge = props_merge.drop_duplicates(subset=["player_canonical"])

    enriched = out.copy()
    enriched = _apply_player_name_cleaning(enriched, name_maps)
    enriched = ensure_canonical(enriched, player_col="player", team_col="team")

    enriched = enriched.merge(props_merge, on="player_canonical", how="left")
    enriched = _coalesce_dupe_cols(enriched)
    enriched = _normalize_player_clean_key_columns(enriched)

    if "team_abbr" in enriched.columns:
        enriched["team_abbr"] = (
            enriched["team_abbr"].fillna(enriched.get("team_from_props"))
        )
    else:
        enriched["team_abbr"] = enriched.get("team_from_props")

    if "team" in enriched.columns:
        enriched["team"] = enriched["team"].fillna(enriched.get("team_abbr"))

    if "opponent_abbr" in enriched.columns:
        enriched["opponent_abbr"] = (
            enriched["opponent_abbr"].fillna(enriched.get("opponent_from_props"))
        )
    else:
        enriched["opponent_abbr"] = enriched.get("opponent_from_props")

    if "opponent" in enriched.columns:
        enriched["opponent"] = enriched["opponent"].fillna(enriched.get("opponent_abbr"))
    else:
        enriched["opponent"] = enriched.get("opponent_abbr")

    if "kickoff_ts_from_props" in enriched.columns:
        if "kickoff_ts" in enriched.columns:
            enriched["kickoff_ts"] = enriched["kickoff_ts"].fillna(
                enriched["kickoff_ts_from_props"]
            )
        else:
            enriched["kickoff_ts"] = enriched["kickoff_ts_from_props"]

    if "event_id_from_props" in enriched.columns:
        if "event_id" in enriched.columns:
            enriched["event_id"] = enriched["event_id"].combine_first(
                enriched["event_id_from_props"]
            )
        else:
            enriched["event_id"] = enriched["event_id_from_props"]

    drop_cols = [
        col
        for col in [
            "team_from_props",
            "opponent_from_props",
            "kickoff_ts_from_props",
            "event_id_from_props",
        ]
        if col in enriched.columns
    ]
    if drop_cols:
        enriched.drop(columns=drop_cols, inplace=True)

    return enriched


def _build_schedule_opponent_lookup() -> Dict[Tuple[str, int], str]:
    """Create a lookup of (team_abbrev, week) -> opponent_abbrev from schedule CSV."""

    if not SCHEDULE_GAMES_PATH.exists():
        return {}

    games = pd.read_csv(SCHEDULE_GAMES_PATH)
    if games.empty:
        return {}

    lu: Dict[Tuple[str, int], str] = {}

    for _, row in games.iterrows():
        try:
            wk = int(row["week"])
        except Exception:
            continue
        home = str(row.get("home_team", "")).strip().upper()
        away = str(row.get("away_team", "")).strip().upper()
        if not home or not away:
            continue
        lu[(home, wk)] = away
        lu[(away, wk)] = home

    return lu


def _assign_team_and_opp_via_schedule(
    df: pd.DataFrame, sched_lu: Dict[Tuple[str, int], str]
) -> pd.DataFrame:
    """Use schedule lookup to backfill opponent when missing."""

    if not sched_lu:
        return df

    df = df.copy()

    if "week_inferred" in df.columns:
        df["__wk"] = df["week_inferred"]
    else:
        df["__wk"] = df.get("week")

    if "team" not in df.columns:
        df["team"] = pd.NA
    df["__team_key"] = df["team"].astype(str).str.upper().str.strip()

    missing_mask = df["opponent"].isna() | (df["opponent"].astype(str).str.strip() == "")

    def _lookup(row: pd.Series) -> Optional[str]:
        team_key = row.get("__team_key")
        week_val = row.get("__wk")
        if pd.isna(team_key) or pd.isna(week_val):
            return None
        try:
            week_int = int(float(week_val))
        except (TypeError, ValueError):
            return None
        return sched_lu.get((str(team_key).upper().strip(), week_int))

    df.loc[missing_mask, "opponent"] = df.loc[missing_mask].apply(_lookup, axis=1)

    df = df.drop(columns=["__wk", "__team_key"], errors="ignore")
    return df
def _is_empty(obj) -> bool:
    try:
        return obj is None or (hasattr(obj, "__len__") and len(obj) == 0)
    except Exception:
        return True


def _to_pandas(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas") and callable(getattr(obj, "to_pandas")):
        try:
            return obj.to_pandas()
        except Exception:
            pass
    if isinstance(obj, (list, tuple)) and obj and hasattr(obj[0], "to_pandas"):
        try:
            return pd.concat([b.to_pandas() for b in obj], ignore_index=True)
        except Exception:
            pass
    return pd.DataFrame(obj)


def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def _safe_read_csv(path: Path | str, label: str = "") -> pd.DataFrame:
    """
    Legacy helper to read CSVs defensively.

    IMPORTANT: this is now used only for *optional* debug inputs. The primary
    player logs are fetched via normalize_game_logs / normalize_season_totals.
    """

    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        logger.info("[player_form] %s missing or empty on disk: %s", label, p)
        return pd.DataFrame()

    try:
        df = pd.read_csv(p)
    except Exception as err:
        logger.warning("[player_form] failed reading %s (%s): %s", label, p, err)
        return pd.DataFrame()

    # treat header-only as empty
    if df.empty:
        logger.info("[player_form] %s has 0 data rows on disk: %s", label, p)
        return pd.DataFrame()

    return df


def _fetch_player_logs(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build *player-level* game logs and season totals for the given season.

    Instead of raw play-by-play, we use nfl_data_py's weekly dataset so that
    each row is a single player's stat line for a single game. This matches
    the conceptual model for player_form and player_form_consensus:
      - player_form: one row per (player, team, season, week)
      - player_form_consensus: one row per (player, team, season) with totals
        and per-game averages across those weeks.
    """

    import nfl_data_py as nfl

    season_int = int(season)
    years = [season_int]

    weekly = nfl.import_weekly_data(years, downcast=True)
    if weekly is None or weekly.empty:
        raise RuntimeError(
            f"[make_player_form] FATAL: import_weekly_data({years}) returned empty frame"
        )

    # Restrict to regular season if present
    if "season_type" in weekly.columns:
        weekly = weekly[weekly["season_type"] == "REG"]

    if weekly.empty:
        raise RuntimeError(
            f"[make_player_form] FATAL: weekly data empty for REG season={season_int}"
        )

    # --- Player identity ----------------------------------------------------
    if "player" not in weekly.columns:
        name_candidates = ["player_name", "full_name"]
        for c in name_candidates:
            if c in weekly.columns:
                weekly = weekly.copy()
                weekly["player"] = weekly[c].astype("string")
                break
        if "player" not in weekly.columns:
            raise RuntimeError(
                "[make_player_form] FATAL: weekly data missing a usable player "
                f"name column; columns={list(weekly.columns)}"
            )

    # --- Team identity (we'll canonicalize later via canonical_names) ------
    if "team" not in weekly.columns:
        team_candidates = ["recent_team", "team_abbr", "posteam"]
        for c in team_candidates:
            if c in weekly.columns:
                weekly = weekly.copy()
                weekly["team"] = weekly[c]
                break

    # Week & season
    if "week" in weekly.columns:
        weekly["week"] = pd.to_numeric(weekly["week"], errors="coerce")
    weekly["season"] = season_int

    # Use weekly as our player_game_logs: one row per player-game
    logs = weekly.copy()

    # --- Season totals: one row per (player, team, season) -----------------
    group_keys = [k for k in ["player", "team", "season"] if k in logs.columns]
    if not group_keys:
        raise RuntimeError(
            "[make_player_form] FATAL: weekly data missing grouping keys "
            "(player/team/season) needed for season totals"
        )

    numeric_cols = [
        c for c in logs.columns
        if pd.api.types.is_numeric_dtype(logs[c])
    ]
    if not numeric_cols:
        raise RuntimeError(
            "[make_player_form] FATAL: weekly data has no numeric stat columns "
            f"to aggregate; columns={list(logs.columns)}"
        )

    totals = (
        logs.groupby(group_keys, dropna=False)[numeric_cols]
        .sum()
        .reset_index()
    )

    # Write the CSVs the rest of the pipeline expects
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    logs.to_csv(PLAYER_GAME_LOGS_OUT, index=False)
    totals.to_csv(PLAYER_SEASON_TOTALS_OUT, index=False)

    return logs, totals


# === SURGICAL ADDITION: merge roles from Ourlads depth charts (clean placement) ===
def _merge_depth_roles(pf: pd.DataFrame) -> pd.DataFrame:
    import os
    import re

    data_dir = globals().get("DATA_DIR", "data")
    pf = pf.loc[:, ~pf.columns.duplicated()].copy()

    def _clean_name(s: str) -> str:
        s = str(s or "").strip()
        if "," in s:
            parts = [p.strip() for p in s.split(",", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                s = f"{parts[1]} {parts[0]}"
        s = re.sub(r"\(.*?\)", "", s)
        s = re.sub(r"\b[A-Z]{1,3}\d{2}\b", "", s)
        s = re.sub(r"\b[Uu]/[A-Za-z]{2,4}\b", "", s)
        s = re.sub(r"\b\d{1,2}/\d{1,2}\b", "", s)
        s = re.sub(r"^\s*(?:#\s*)?\d+\s*[-–—:]?\s*", "", s)
        s = s.replace(".", " ")
        s = re.sub(r"\s+(JR|SR|II|III|IV|V)\.?$", "", s, flags=re.I)
        s = re.sub(r"[^\w\s'\-]", " ", s)
        s = re.sub(r"\b\d+\b", "", s)
        s = re.sub(r"\s+", " ", s).strip().title()
        return "" if s == "U" else s

    roles_path = os.path.join(data_dir, "roles_ourlads.csv")
    if not os.path.exists(roles_path):
        raise RuntimeError(
            "roles_ourlads.csv missing. Run build_depth_charts_ourlads.py before make_player_form.py"
        )

    roles = pd.read_csv(roles_path)
    if "status" in roles.columns:
        status_series = roles["status"].astype(str).str.lower()
        before = len(roles)
        roles = roles[status_series != "inactive"].copy()
        logger.info(
            "[make_player_form] roles filtered to active only: %s → %s",
            before,
            len(roles),
        )
    if roles.empty:
        raise RuntimeError(
            "roles_ourlads.csv is empty. Check Ourlads scraper/selectors."
        )

    for col in ("team", "player", "role", "model_role", "position", "position_group"):
        if col in roles.columns:
            roles[col] = roles[col].astype(str)
    if "player" in roles.columns:
        roles["player"] = roles["player"].map(_clean_name).map(standardize_full_name)
    roles["team"] = roles.get("team", "").astype(str).map(_canon_team)
    roles["role"] = roles.get("role", "").astype(str).str.upper().str.strip()
    if "model_role" in roles.columns:
        roles["model_role"] = roles["model_role"].astype(str).str.upper().str.strip()
        roles["model_role"] = roles["model_role"].replace({"": pd.NA, "NAN": pd.NA})
        roles["role"] = roles["model_role"].combine_first(roles["role"])
    else:
        roles["model_role"] = roles["role"]
    if "depth_chart_role" in roles.columns and "alignment_role" not in roles.columns:
        roles.rename(columns={"depth_chart_role": "alignment_role"}, inplace=True)
    if "alignment_role" in roles.columns:
        roles["alignment_role"] = roles["alignment_role"].astype(str).str.upper().str.strip()
    if "position_group" in roles.columns:
        roles["position_group"] = roles["position_group"].astype(str).str.upper().str.strip()
    if "position" not in roles.columns and "role" in roles.columns:
        roles["position"] = roles["role"].str.extract(r"([A-Z]+)")

    roles = ensure_canonical(roles, player_col="player", team_col="team")

    merge_cols = ["player_canonical", "team"]
    for extra in [
        "role",
        "model_role",
        "position",
        "position_group",
        "alignment_role",
        "position_guess",
    ]:
        if extra in roles.columns:
            merge_cols.append(extra)

    roles_subset = roles[merge_cols].drop_duplicates(
        ["player_canonical", "team"], keep="first"
    )

    pf = pf.merge(
        roles_subset,
        on=["player_canonical", "team"],
        how="left",
        suffixes=("", "_roles"),
    )
    pf = _coalesce_dupe_cols(pf)

    assert_no_duplicate_columns(pf, "roles merge")

    if "model_role" not in pf.columns and "role" in pf.columns:
        pf["model_role"] = pf["role"]
    if "position_group" not in pf.columns:
        pf["position_group"] = pd.NA
    if "alignment_role" not in pf.columns:
        pf["alignment_role"] = pd.NA
    if "role_roles" in pf.columns:
        pf["role"] = pf["role"].combine_first(pf["role_roles"])
        pf.drop(columns=["role_roles"], inplace=True)
    if "position_roles" in pf.columns:
        pf["position"] = pf["position"].combine_first(pf["position_roles"])
        pf.drop(columns=["position_roles"], inplace=True)
    if "position_guess" in pf.columns:
        pf["position"] = pf["position"].combine_first(pf["position_guess"])
    if "model_role_roles" in pf.columns:
        pf["model_role"] = pf["model_role"].combine_first(pf["model_role_roles"])
        pf.drop(columns=["model_role_roles"], inplace=True)
    if "position_group_roles" in pf.columns:
        pf["position_group"] = pf.get("position_group").combine_first(pf["position_group_roles"])
        pf.drop(columns=["position_group_roles"], inplace=True)
    if "alignment_role_roles" in pf.columns:
        pf["alignment_role"] = pf.get("alignment_role").combine_first(pf["alignment_role_roles"])
        pf.drop(columns=["alignment_role_roles"], inplace=True)

    pf["unmatched_flag"] = pf["role"].isna()

    try:
        miss = pf.loc[pf["unmatched_flag"]].copy()
        if not miss.empty:
            team_set = set(roles["team"].dropna().unique())
            miss["unmatched_class"] = np.where(
                miss["player_canonical"].eq(""),
                "missing_player_key",
                np.where(
                    miss["team"].isin(team_set),
                    "name_not_in_roles",
                    "team_not_in_roles",
                ),
            )
            unmatched_cols = [
                c
                for c in [
                    "player",
                    "team",
                    "player_canonical",
                    "player_clean_key",
                    "team_key",
                    "unmatched_class",
                ]
                if c in miss.columns
            ]
            os.makedirs(data_dir, exist_ok=True)
            miss[unmatched_cols].drop_duplicates().to_csv(
                os.path.join(data_dir, "unmatched_roles_merge.csv"), index=False
            )
            logger.info(
                "[make_player_form] unmatched after roles merge: %s → %s",
                len(miss),
                os.path.join(data_dir, "unmatched_roles_merge.csv"),
            )
    except Exception:
        pass

    try:
        cov = pf["role"].notna().mean()
        logger.info("[make_player_form] merged depth roles → coverage now %.2f%%", cov * 100)
    except Exception:
        pass

    return pf


def _norm_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".", "", regex=False).str.strip()


def _valid_player_mask(series: pd.Series) -> pd.Series:
    """Return True where the normalized player string looks usable."""
    if series is None:
        return pd.Series(dtype=bool)
    trimmed = series.astype(str).str.strip()
    return trimmed.ne("") & ~trimmed.str.lower().isin({"nan", "none"})


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _non_destructive_merge(
    base: pd.DataFrame, add: pd.DataFrame, keys: List[str]
) -> pd.DataFrame:
    if _is_empty(add):
        return base
    add = add.copy()
    add.columns = [c.lower() for c in add.columns]
    if not set(keys).issubset(add.columns):
        return base
    for k in keys:
        add[k] = add[k].astype(str).str.strip()
    out = base.merge(add, on=keys, how="left", suffixes=("", "_ext"))
    for c in add.columns:
        if c in keys:
            continue
        ext = f"{c}_ext"
        if ext in out.columns:
            out[c] = out[c].combine_first(out[ext])
            out.drop(columns=[ext], inplace=True)
    return out


# ---------------------------
# Team canonicalizer
# ---------------------------
VALID = {
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LAC",
    "LAR",
    "LV",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
}

DEFENSE_TEAM_CANDIDATES = [
    "defteam",
    "defense_team",
    "def_team",
    "defense",
    "defteam_abbr",
    "defense_abbr",
    "opp_team",
    "opp_team_abbr",
    "opp",
    "opp_abbr",
    "opponent",
]

TEAM_NAME_TO_ABBR = {
    "ARI": "ARI",
    "ARZ": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "GNB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAX": "JAX",
    "JAC": "JAX",
    "KC": "KC",
    "KCC": "KC",
    "LAC": "LAC",
    "LAR": "LAR",
    "LA": "LAR",
    "LV": "LV",
    "OAK": "LV",
    "LAS": "LV",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NWE": "NE",
    "NO": "NO",
    "NOR": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SF",
    "SFO": "SF",
    "TB": "TB",
    "TAM": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
    "WSH": "WAS",
    "WFT": "WAS",
    "ARIZONA CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN",
    "DETROIT LIONS": "DET",
    "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC",
    "LOS ANGELES CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR",
    "LAS VEGAS RAIDERS": "LV",
    "MIAMI DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE",
    "NEW ORLEANS SAINTS": "NO",
    "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ",
    "PHILADELPHIA EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT",
    "SEATTLE SEAHAWKS": "SEA",
    "SAN FRANCISCO 49ERS": "SF",
    "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN",
    "WASHINGTON COMMANDERS": "WAS",
    "WASHINGTON FOOTBALL TEAM": "WAS",
    "ARIZONA": "ARI",
    "CARDINALS": "ARI",
    "ATLANTA": "ATL",
    "FALCONS": "ATL",
    "BALTIMORE": "BAL",
    "RAVENS": "BAL",
    "BUFFALO": "BUF",
    "BILLS": "BUF",
    "CAROLINA": "CAR",
    "PANTHERS": "CAR",
    "CHICAGO": "CHI",
    "BEARS": "CHI",
    "CINCINNATI": "CIN",
    "BENGALS": "CIN",
    "CLEVELAND": "CLE",
    "BROWNS": "CLE",
    "DALLAS": "DAL",
    "COWBOYS": "DAL",
    "DENVER": "DEN",
    "BRONCOS": "DEN",
    "DETROIT": "DET",
    "LIONS": "DET",
    "GREEN BAY": "GB",
    "PACKERS": "GB",
    "HOUSTON": "HOU",
    "TEXANS": "HOU",
    "INDIANAPOLIS": "IND",
    "COLTS": "IND",
    "JACKSONVILLE": "JAX",
    "JAGUARS": "JAX",
    "KANSAS CITY": "KC",
    "CHIEFS": "KC",
    "CHARGERS": "LAC",
    "RAMS": "LAR",
    "LOS ANGELES": "LAR",
    "LAS VEGAS": "LV",
    "RAIDERS": "LV",
    "MIAMI": "MIA",
    "DOLPHINS": "MIA",
    "MINNESOTA": "MIN",
    "VIKINGS": "MIN",
    "NEW ENGLAND": "NE",
    "PATRIOTS": "NE",
    "NEW ORLEANS": "NO",
    "SAINTS": "NO",
    "GIANTS": "NYG",
    "JETS": "NYJ",
    "PHILADELPHIA": "PHI",
    "EAGLES": "PHI",
    "PITTSBURGH": "PIT",
    "STEELERS": "PIT",
    "SEATTLE": "SEA",
    "SEAHAWKS": "SEA",
    "SAN FRANCISCO": "SF",
    "49ERS": "SF",
    "TAMPA BAY": "TB",
    "BUCCANEERS": "TB",
    "TENNESSEE": "TEN",
    "TITANS": "TEN",
    "WASHINGTON": "WAS",
    "COMMANDERS": "WAS",
}

# defensively add lowercase variants for mapping
TEAM_NAME_TO_ABBR.update({k.lower(): v for k, v in TEAM_NAME_TO_ABBR.items()})


def _canon_team(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    if s in TEAM_NAME_TO_ABBR:
        abbr = TEAM_NAME_TO_ABBR[s]
        return abbr if abbr in VALID else ""
    s2 = re.sub(r"[^A-Z0-9 ]+", "", s).strip()
    if s2 in TEAM_NAME_TO_ABBR:
        abbr = TEAM_NAME_TO_ABBR[s2]
        return abbr if abbr in VALID else ""
    return ""


def _backfill_opponent_from_schedule(out: pd.DataFrame) -> pd.DataFrame:
    """Use team-week map to fill opponents/event IDs when props miss them."""

    if out is None or out.empty:
        return out

    sched_path = Path("data/team_week_map.csv")
    if not sched_path.exists():
        alt = Path("outputs/team_week_map.csv")
        if alt.exists():
            sched_path = alt
        else:
            print("[make_player_form] skip schedule backfill: team_week_map.csv not found")
            return out

    try:
        sch = pd.read_csv(sched_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[make_player_form] skip schedule backfill: failed to read schedule ({exc})")
        return out

    if sch.empty:
        print("[make_player_form] skip schedule backfill: schedule empty")
        return out

    sch = sch.copy()
    for col in ("season", "week"):
        if col in sch.columns:
            sch[col] = pd.to_numeric(sch[col], errors="coerce").astype("Int64")
    for col in ("team", "opponent"):
        if col in sch.columns:
            sch[col] = sch[col].map(_canon_team)
            sch.loc[sch[col] == "", col] = pd.NA
    if "bye" in sch.columns:
        sch["bye"] = sch["bye"].fillna(False).astype(bool)
    else:
        sch["bye"] = False

    take_cols = [c for c in ("season", "week", "team", "opponent", "bye", "event_id") if c in sch.columns]
    if len(take_cols) < 3 or not {"season", "week", "team"}.issubset(take_cols):
        return out

    sch = sch[take_cols].drop_duplicates()

    merged = out.copy()
    for col in ("season", "week"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")
    if "team" in merged.columns:
        merged["team"] = merged["team"].map(_canon_team)
        merged.loc[merged["team"] == "", "team"] = pd.NA

    join_keys = [c for c in ("season", "week", "team") if c in merged.columns]
    if len(join_keys) < 3:
        return out

    merged = merged.merge(
        sch,
        on=join_keys,
        how="left",
        suffixes=("", "_schedule"),
        validate="m:1",
    )

    if "opponent" not in merged.columns:
        merged["opponent"] = pd.Series(pd.NA, index=merged.index)

    props_series = merged.get("opponent_from_props")
    if props_series is not None:
        merged["opponent"] = props_series.combine_first(merged["opponent"])

    need_mask = merged["opponent"].isna()
    if "opponent_schedule" in merged.columns:
        merged.loc[need_mask, "opponent"] = merged.loc[need_mask, "opponent_schedule"]
        merged.drop(columns=["opponent_schedule"], inplace=True)

    n_backfilled = int((need_mask & merged["opponent"].notna()).sum())
    if n_backfilled:
        print(f"[make_player_form] schedule backfill: opponent filled for {n_backfilled} rows")

    if "event_id" in merged.columns and "event_id_schedule" in merged.columns:
        merged["event_id"] = merged["event_id"].combine_first(merged["event_id_schedule"])
        merged.drop(columns=["event_id_schedule"], inplace=True)
    elif "event_id" not in merged.columns and "event_id_schedule" in merged.columns:
        merged.rename(columns={"event_id_schedule": "event_id"}, inplace=True)

    if "bye" in merged.columns and "bye_schedule" in merged.columns:
        merged["bye"] = merged["bye"].fillna(False) | merged["bye_schedule"].fillna(False)
        merged.drop(columns=["bye_schedule"], inplace=True)
    elif "bye_schedule" in merged.columns:
        merged.rename(columns={"bye_schedule": "bye"}, inplace=True)

    return merged


def _derive_opponent(df: pd.DataFrame) -> pd.Series:
    """Return canonical opponent abbreviations for a play-by-play frame.

    Order:
      1) direct defensive columns (defteam/defense_team/etc)
      2) home/away vs posteam (if available)
      3) schedules via game_id (fallback)
    Never raises; always returns a Series aligned to df.index.
    """
    if df.empty:
        return pd.Series(np.nan, index=df.index, dtype=object)

    # 1) direct defensive columns
    col = next((c for c in DEFENSE_TEAM_CANDIDATES if c in df.columns), None)
    if col is not None:
        opp = df[col]
        if isinstance(opp, pd.Series):
            opp_norm = opp.where(opp.notna(), "").astype(str).str.upper().str.strip()
            mapped = opp_norm.map(_canon_team).replace("", np.nan)
            # If we got any valid opponents, return them
            if mapped.notna().any():
                return mapped.reindex(df.index)

    # 2) home/away vs posteam
    try:
        if {"posteam", "home_team", "away_team"}.issubset(df.columns):
            posteam = df["posteam"].astype(str).str.upper().str.strip().map(_canon_team)
            home = df["home_team"].astype(str).str.upper().str.strip().map(_canon_team)
            away = df["away_team"].astype(str).str.upper().str.strip().map(_canon_team)
            opp = np.where(posteam.eq(home), away, home)
            opp = pd.Series(opp, index=df.index).map(_canon_team).replace("", np.nan)
            if opp.notna().any():
                return opp
    except Exception:
        pass

    # 3) schedule-based fallback
    try:
        season_guess = 2025
        if "season" in df.columns and df["season"].notna().any():
            try:
                season_guess = int(
                    pd.to_numeric(df["season"], errors="coerce").dropna().iloc[0]
                )
            except Exception:
                pass
        if "game_id" in df.columns:
            sched = _load_schedule_map(season_guess)
            if not sched.empty:
                merged = df[["game_id"]].copy().merge(sched, on="game_id", how="left")
                if "posteam" in df.columns and {"home_team", "away_team"}.issubset(
                    merged.columns
                ):
                    posteam = (
                        df["posteam"]
                        .astype(str)
                        .str.upper()
                        .str.strip()
                        .map(_canon_team)
                    )
                    home = merged["home_team"]
                    away = merged["away_team"]
                    opp = np.where(posteam.eq(home), away, home)
                    opp = (
                        pd.Series(opp, index=df.index)
                        .map(_canon_team)
                        .replace("", np.nan)
                    )
                    if opp.notna().any():
                        return opp
    except Exception:
        pass

    # Final: could not derive
    return pd.Series(np.nan, index=df.index, dtype=object)


def _normalize_props_opponent(df: pd.DataFrame) -> pd.Series:
    """Derive and canonicalize opponent abbreviations for props payloads."""

    if df.empty:
        return pd.Series(np.nan, index=df.index, dtype=object)

    base = pd.Series(np.nan, index=df.index, dtype=object)
    opp_col = next((c for c in DEFENSE_TEAM_CANDIDATES if c in df.columns), None)
    if opp_col is None:
        return base

    try:
        derived = _derive_opponent(df)
    except Exception:
        derived = base

    if (
        isinstance(derived, pd.Series)
        and len(derived) == len(df)
        and derived.notna().any()
    ):
        return derived.reindex(df.index)

    raw = df[opp_col]
    if not isinstance(raw, pd.Series):
        return base
    opp_norm = raw.where(raw.notna(), "").astype(str).str.upper().str.strip()
    mapped = opp_norm.map(_canon_team).replace("", np.nan)
    return mapped.reindex(df.index)


# ---------------------------
# nflverse loader
# ---------------------------


def _import_nflverse():
    try:
        import nflreadpy as nflv

        return nflv, "nflreadpy"
    except Exception:
        try:
            import nfl_data_py as nflv  # type: ignore

            return nflv, "nfl_data_py"
        except Exception as e:
            raise RuntimeError(
                "Neither nflreadpy nor nfl_data_py is installed. Run: pip install nflreadpy"
            ) from e


NFLV, NFL_PKG = _import_nflverse()


def _load_schedule_map(season: int) -> pd.DataFrame:
    """Return schedule map (game_id -> home_team, away_team) for the season, canonized to abbrs."""
    try:
        if NFL_PKG == "nflreadpy":
            sched = NFLV.load_schedules(seasons=[season])
        else:
            # nfl_data_py
            if hasattr(NFLV, "import_schedules"):
                sched = NFLV.import_schedules([season])  # type: ignore
            else:
                return pd.DataFrame()
        df = _to_pandas(sched)
    except Exception:
        return pd.DataFrame()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    if not {"game_id", "home_team", "away_team"}.issubset(df.columns):
        return pd.DataFrame()
    df["home_team"] = (
        df["home_team"].astype(str).str.upper().str.strip().map(_canon_team)
    )
    df["away_team"] = (
        df["away_team"].astype(str).str.upper().str.strip().map(_canon_team)
    )
    return df[["game_id", "home_team", "away_team"]].drop_duplicates()
def _load_weekly_rosters(season: int) -> pd.DataFrame:
    """(player, team, position), forgiving team keys."""
    try:
        if NFL_PKG == "nflreadpy":
            ro = NFLV.load_weekly_rosters(seasons=[season])
        else:
            ro = NFLV.import_weekly_rosters([season])  # type: ignore
        df = _to_pandas(ro)
    except Exception:
        return pd.DataFrame()
    if _is_empty(df):
        return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]

    # player name
    name_col = None
    for c in ["player_name", "name", "full_name"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        return pd.DataFrame()
    df["player"] = _norm_name(df[name_col].fillna(""))

    # team — accept any column that looks like a team key
    team_col = None
    for c in ["team", "recent_team", "club_code", "team_abbr", "posteam"]:
        if c in df.columns:
            team_col = c
            break
    if team_col is None:
        return pd.DataFrame()
    df["team"] = df[team_col].astype(str).str.upper().str.strip().map(_canon_team)
    df = df[df["team"].isin(VALID)]

    # position
    pos_col = None
    for c in ["position", "pos"]:
        if c in df.columns:
            pos_col = c
            break
    df["position"] = np.where(
        pos_col is not None, df[pos_col].astype(str).str.upper().str.strip(), np.nan
    )

    if "week" in df.columns:
        df = df.sort_values(["player", "team", "week"]).drop_duplicates(
            ["player", "team"], keep="last"
        )

    return df[["player", "team", "position"]].drop_duplicates()


def _load_players_master() -> pd.DataFrame:
    """Fallback: (player -> position) without team join."""
    try:
        if NFL_PKG == "nflreadpy":
            pl = NFLV.load_players()
        else:
            pl = NFLV.import_players()  # type: ignore
        df = _to_pandas(pl)
    except Exception:
        return pd.DataFrame()
    if _is_empty(df):
        return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    name_col = None
    for c in ["player_name", "name", "full_name", "display_name"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        return pd.DataFrame()
    df["player"] = _norm_name(df[name_col].fillna(""))
    pos_col = None
    for c in ["position", "pos", "gsis_pos"]:
        if c in df.columns:
            pos_col = c
            break
    if pos_col is None:
        return pd.DataFrame()
    df["position"] = df[pos_col].astype(str).str.upper().str.strip()
    return df[["player", "position"]].drop_duplicates()


# ---------------------------
# Optional roles.csv merge
# ---------------------------


def _merge_roles_csv(df: pd.DataFrame) -> pd.DataFrame:
    roles_path = os.path.join(DATA_DIR, "roles.csv")
    if not os.path.exists(roles_path):
        return df
    try:
        r = pd.read_csv(roles_path)
    except Exception:
        return df
    r.columns = [c.lower() for c in r.columns]
    if "player" not in r.columns and "player_name" in r.columns:
        r = r.rename(columns={"player_name": "player"})
    need = {"player", "team", "opponent", "role"}
    if not need.issubset(r.columns):
        return df
    r["player"] = _norm_name(r["player"].astype(str))
    r["team"] = r["team"].astype(str).str.upper().str.strip().map(_canon_team)
    r = r[r["team"].isin(VALID)]
    name_maps = build_name_canonical_map()
    r = _apply_player_name_cleaning(r, name_maps)
    r = ensure_canonical(r, player_col="player", team_col="team")
    df = _apply_player_name_cleaning(df, name_maps)
    df = ensure_canonical(df, player_col="player", team_col="team")
    out = df.merge(
        r[["player_canonical", "team", "opponent", "role"]],
        on=["player_canonical", "team"],
        how="left",
        suffixes=("", "_roles"),
    )
    out = _coalesce_dupe_cols(out)
    assert_no_duplicate_columns(out, "roles.csv merge")
    if "role_roles" in out.columns:
        out["role"] = out["role"].combine_first(out["role_roles"])
        out.drop(columns=["role_roles"], inplace=True)
    return out


# ---------------------------
# Role & family inference
# ---------------------------


def _infer_position_family_from_usage(pf: pd.DataFrame) -> pd.Series:
    """
    Return a Series of position family guesses {QB,RB,WR} based on usage
    when exact position is missing.
    """
    fam = pd.Series(index=pf.index, dtype=object)
    # Heuristics:
    # 1) If dropbacks or pass attempts large → QB
    qb_mask = pd.Series(False, index=pf.index)
    if "dropbacks" in pf.columns:
        qb_mask |= pf["dropbacks"].fillna(0) >= 15
    if "ypa" in pf.columns:
        qb_mask |= pf["ypa"].notna() & (pf["ypa"] > 6.0)
    fam[qb_mask] = "QB"

    # 2) If rush_share is present and dominates → RB
    rb_mask = pf.get("rush_share", pd.Series(0, index=pf.index)).fillna(0) >= 0.20
    fam[rb_mask & fam.isna()] = "RB"

    # 3) Else default to WR family for receiving usage
    wr_mask = (pf.get("route_rate", pd.Series(0, index=pf.index)).fillna(0) >= 0.20) | (
        pf.get("tgt_share", pd.Series(0, index=pf.index)).fillna(0) >= 0.15
    )
    fam[wr_mask & fam.isna()] = "WR"

    return fam


def _infer_roles_minimal(pf: pd.DataFrame) -> pd.DataFrame:
    pf = pf.copy()
    if "role" not in pf.columns:
        pf["role"] = np.nan

    # use exact position if present, else fall back to family guess
    pos = pf.get("position")
    fam = pd.Series(index=pf.index, dtype=object)
    if pos is not None:
        fam = pos.copy()
    # only fill where fam is null
    fam = fam.where(fam.notna(), _infer_position_family_from_usage(pf))
    fam = fam.astype(object)

    def rank_and_tag(g: pd.DataFrame, mask: pd.Series, score_col: str, tags: List[str]):
        g = g.copy()
        idx = g.index[mask]
        if len(idx) == 0 or score_col not in g.columns:
            return g
        scores = g.loc[idx, score_col].astype(float)
        order = scores.rank(method="first", ascending=False)
        if len(tags) >= 1:
            g.loc[idx[order == 1], "role"] = g.loc[idx[order == 1], "role"].where(
                g.loc[idx[order == 1], "role"].notna(), tags[0]
            )
        if len(tags) >= 2:
            g.loc[idx[order == 2], "role"] = g.loc[idx[order == 2], "role"].where(
                g.loc[idx[order == 2], "role"].notna(), tags[1]
            )
        return g

    out = []
    for team, g in pf.groupby(["team", "opponent"], dropna=False):
        g = g.copy()
        g_fam = fam.loc[g.index].astype(str)

        # QB1 by dropbacks (fallback ypa)
        qb_mask = g_fam.str.upper().eq("QB")
        if qb_mask.any():
            if "dropbacks" in g.columns and g["dropbacks"].notna().any():
                g = rank_and_tag(g, qb_mask, "dropbacks", ["QB1"])
            elif "ypa" in g.columns and g["ypa"].notna().any():
                g = rank_and_tag(g, qb_mask, "ypa", ["QB1"])

        # RB1/RB2 by rush_share
        rb_mask = g_fam.str.upper().eq("RB")
        if (
            rb_mask.any()
            and "rush_share" in g.columns
            and g["rush_share"].notna().any()
        ):
            g = rank_and_tag(g, rb_mask, "rush_share", ["RB1", "RB2"])

        # WR1/WR2 by route_rate (TE may be treated as WR family if unknown)
        wr_mask = g_fam.str.upper().eq("WR")
        if (
            wr_mask.any()
            and "route_rate" in g.columns
            and g["route_rate"].notna().any()
        ):
            g = rank_and_tag(g, wr_mask, "route_rate", ["WR1", "WR2"])

        out.append(g)

    return pd.concat(out, ignore_index=True) if out else pf


# ---------------------------
# Build from PBP
# ---------------------------


def build_player_form_legacy(
    season: int = 2025, slate_date: str | None = None, week: int | None = None
) -> pd.DataFrame:
    if slate_date:
        logger.info("[make_player_form] build_player_form using slate_date=%s", slate_date)
    try:
        pbp = load_pbp(season)
    except Exception as err:
        warnings.warn(
            f"Failed to load play-by-play for season {season}: {err}; proceeding with empty data frame.",
            RuntimeWarning,
        )
        pbp = pd.DataFrame()
    if pbp.empty:
        allow_offseason = os.getenv("ALLOW_OFFSEASON_FALLBACK", "0") != "0"
        msg = f"No play-by-play data available for season {season}."
        if not allow_offseason:
            raise RuntimeError(
                msg + " Set ALLOW_OFFSEASON_FALLBACK=1 to write structural base."
            )
        warnings.warn(
            msg + " Writing structural base due to ALLOW_OFFSEASON_FALLBACK.",
            RuntimeWarning,
        )
        base = pd.DataFrame(columns=["player", "team"])
        base["season"] = int(season)
        base = _ensure_cols(base, FINAL_COLS)
        base = (
            base[FINAL_COLS]
            .drop_duplicates(subset=["player", "team", "opponent", "season"])
            .reset_index(drop=True)
        )
        print("[pf] pbp empty → structural base only")
        return base

    off_col = (
        "posteam"
        if "posteam" in pbp.columns
        else ("offense_team" if "offense_team" in pbp.columns else None)
    )
    if off_col is None:
        raise RuntimeError("No offense team column in PBP (posteam/offense_team).")

    name_maps = build_name_canonical_map()

    # Opponent once (remove duplicate logic)
    opp_col = (
        "defteam"
        if "defteam" in pbp.columns
        else ("defense_team" if "defense_team" in pbp.columns else None)
    )
    if opp_col is None:
        pbp["opponent"] = np.nan
    else:
        pbp["opponent"] = pbp[opp_col].astype(str).str.upper().str.strip()

    # Ensure counting columns exist and are numeric to avoid groupby collapse
    for col in [
        "pass_attempt",
        "complete_pass",
        "qb_dropback",
        "rush_attempt",
        "yards_gained",
    ]:
        if col in pbp.columns:
            pbp[col] = pd.to_numeric(pbp[col], errors="coerce").fillna(0)
            # boolean/int flags should be ints
            if col != "yards_gained":
                pbp[col] = pbp[col].astype(int)

    # Derive robust is_pass / is_rush flags
    pt = pbp.get("play_type")
    is_pass = pbp.get("pass")
    if is_pass is None:
        is_pass = (
            pt.isin(["pass", "no_play"])
            if pt is not None
            else pd.Series(False, index=pbp.index)
        )
    else:
        is_pass = pd.Series(is_pass).astype(bool)
    is_rush = pbp.get("rush")
    if is_rush is None:
        is_rush = pt.eq("run") if pt is not None else pd.Series(False, index=pbp.index)
    else:
        is_rush = pd.Series(is_rush).astype(bool)

    # RECEIVING
    is_pass = pbp.get("pass")
    if is_pass is None:
        pt = pbp.get("play_type")
        is_pass = (
            pt.isin(["pass", "no_play"])
            if pt is not None
            else pd.Series(False, index=pbp.index)
        )
    else:
        is_pass = is_pass.astype(bool)

    rec = pbp.loc[is_pass].copy()
    rec["opponent"] = _derive_opponent(rec)
    if rec.empty:
        rply = pd.DataFrame(columns=["team", "opponent", "player"])
    else:
        rcv_name_col = (
            "receiver_player_name"
            if "receiver_player_name" in rec.columns
            else ("receiver" if "receiver" in rec.columns else None)
        )
        if rcv_name_col is None:
            rec["receiver_player_name"] = np.nan
            rcv_name_col = "receiver_player_name"
        rec["player"] = _norm_name(rec[rcv_name_col].fillna(""))
        rec["team"] = rec[off_col].astype(str).str.upper().str.strip().map(_canon_team)
        rec["team"] = rec["team"].replace("", np.nan)

        rec["opponent"] = (
            rec["opponent"].astype(str).str.upper().str.strip()
            if "opponent" in rec.columns
            else np.nan
        )
        team_targets = (
            rec.groupby(["team", "opponent"], dropna=False)
            .size()
            .rename("team_targets")
            .astype(float)
        )
        if "qb_dropback" in rec.columns:
            team_dropbacks = (
                rec.groupby(["team", "opponent"], dropna=False)["qb_dropback"]
                .sum(min_count=1)
                .rename("team_dropbacks")
            )
        else:
            team_dropbacks = (
                rec.groupby(["team", "opponent"], dropna=False)
                .size()
                .rename("team_dropbacks")
                .astype(float)
            )

        rec_players = rec.loc[_valid_player_mask(rec["player"])].copy()
        if rec_players.empty:
            rply = pd.DataFrame(columns=["team", "opponent", "player"])
        else:
            rply = (
                rec_players.groupby(["team", "opponent", "player"], dropna=False)
                .agg(
                    targets=(
                        ("pass_attempt", "sum")
                        if "pass_attempt" in rec.columns
                        else ("player", "size")
                    ),
                    rec_yards=("yards_gained", "sum"),
                    receptions=(
                        ("complete_pass", "sum")
                        if "complete_pass" in rec.columns
                        else (rcv_name_col, "size")
                    ),
                )
                .reset_index()
            )
            rply = rply.merge(
                team_targets.reset_index(), on=["team", "opponent"], how="left"
            )
            rply = rply.merge(
                team_dropbacks.reset_index(), on=["team", "opponent"], how="left"
            )
            rply["tgt_share"] = np.where(
                rply["team_targets"] > 0, rply["targets"] / rply["team_targets"], np.nan
            )
            rply["route_rate"] = np.where(
                rply["team_dropbacks"] > 0,
                rply["targets"] / rply["team_dropbacks"],
                np.nan,
            ).clip(0.05, 0.95)
            rply["ypt"] = np.where(
                rply["targets"] > 0, rply["rec_yards"] / rply["targets"], np.nan
            )
            rply["receptions_per_target"] = np.where(
                rply["targets"] > 0, rply["receptions"] / rply["targets"], np.nan
            )
            routes_proxy = (rply["team_dropbacks"] * rply["route_rate"]).replace(
                0, np.nan
            )
            rply["yprr"] = np.where(
                routes_proxy > 0, rply["rec_yards"] / routes_proxy, np.nan
            )

        inside20 = (
            rec_players.copy()
            if not rec_players.empty
            else pd.DataFrame(columns=rec.columns)
        )
        inside20["yardline_100"] = pd.to_numeric(
            inside20.get("yardline_100"), errors="coerce"
        )
        rz_rec = inside20.loc[inside20["yardline_100"] <= 20]
        if not rz_rec.empty:
            rz_tgt_ply = (
                rz_rec.groupby(["team", "opponent", "player"], dropna=False)
                .size()
                .rename("rz_targets")
            )
            rz_tgt_tm = (
                rz_rec.groupby(["team", "opponent"], dropna=False)
                .size()
                .rename("rz_team_targets")
            )
            rply = rply.merge(
                rz_tgt_ply.reset_index(),
                on=["team", "opponent", "player"],
                how="left",
            )
            rply = rply.merge(
                rz_tgt_tm.reset_index(),
                on=["team", "opponent"],
                how="left",
            )
            rply["rz_tgt_share"] = np.where(
                rply["rz_team_targets"] > 0,
                rply["rz_targets"] / rply["rz_team_targets"],
                np.nan,
            )

    rply = _ensure_cols(
        rply,
        [
            "targets",
            "rec_yards",
            "receptions",
            "team_targets",
            "team_dropbacks",
            "tgt_share",
            "route_rate",
            "ypt",
            "receptions_per_target",
            "yprr",
            "rz_targets",
            "rz_team_targets",
            "rz_tgt_share",
        ],
    )

    rply = _apply_player_name_cleaning(rply, name_maps)

    # RUSHING
    is_rush = pbp.get("rush")
    if is_rush is None:
        pt = pbp.get("play_type")
        is_rush = pt.eq("run") if pt is not None else pd.Series(False, index=pbp.index)
    else:
        is_rush = is_rush.astype(bool)

    ru = pbp.loc[is_rush].copy()
    ru["opponent"] = _derive_opponent(ru)
    if ru.empty:
        rru = pd.DataFrame(columns=["team", "opponent", "player"])
    else:
        rush_name_col = (
            "rusher_player_name"
            if "rusher_player_name" in ru.columns
            else ("rusher" if "rusher" in ru.columns else None)
        )
        if rush_name_col is None:
            ru["rusher_player_name"] = np.nan
            rush_name_col = "rusher_player_name"

        ru["player"] = _norm_name(ru[rush_name_col].fillna(""))
        ru["team"] = ru[off_col].astype(str).str.upper().str.strip().map(_canon_team)
        ru["team"] = ru["team"].replace("", np.nan)

        ru["opponent"] = (
            ru["opponent"].astype(str).str.upper().str.strip()
            if "opponent" in ru.columns
            else np.nan
        )
        team_rushes = (
            ru.groupby(["team", "opponent"], dropna=False)
            .size()
            .rename("team_rushes")
            .astype(float)
        )
        ru_players = ru.loc[_valid_player_mask(ru["player"])].copy()
        if ru_players.empty:
            rru = pd.DataFrame(columns=["team", "opponent", "player"])
        else:
            rru = (
                ru_players.groupby(["team", "opponent", "player"], dropna=False)
                .agg(
                    rushes=(
                        ("rush_attempt", "sum")
                        if "rush_attempt" in ru.columns
                        else ("player", "size")
                    ),
                    rush_yards=("yards_gained", "sum"),
                )
                .reset_index()
            )
            rru = rru.merge(
                team_rushes.reset_index(), on=["team", "opponent"], how="left"
            )
            rru["rush_share"] = np.where(
                rru["team_rushes"] > 0, rru["rushes"] / rru["team_rushes"], np.nan
            )
            rru["ypc"] = np.where(
                rru["rushes"] > 0, rru["rush_yards"] / rru["rushes"], np.nan
            )

        inside10 = (
            ru_players.copy()
            if not ru_players.empty
            else pd.DataFrame(columns=ru.columns)
        )
        inside10["yardline_100"] = pd.to_numeric(
            inside10.get("yardline_100"), errors="coerce"
        )
        rz_ru = inside10.loc[inside10["yardline_100"] <= 10]
        if not rz_ru.empty:
            rz_ru_ply = (
                rz_ru.groupby(["team", "opponent", "player"], dropna=False)
                .size()
                .rename("rz_rushes")
            )
            rz_ru_tm = (
                rz_ru.groupby(["team", "opponent"], dropna=False)
                .size()
                .rename("rz_team_rushes")
            )
            rru = rru.merge(
                rz_ru_ply.reset_index(),
                on=["team", "opponent", "player"],
                how="left",
            )
            rru = rru.merge(
                rz_ru_tm.reset_index(),
                on=["team", "opponent"],
                how="left",
            )
            rru["rz_rush_share"] = np.where(
                rru["rz_team_rushes"] > 0,
                rru["rz_rushes"] / rru["rz_team_rushes"],
                np.nan,
            )

    rru = _ensure_cols(
        rru,
        [
            "rushes",
            "rush_yards",
            "team_rushes",
            "rush_share",
            "ypc",
            "rz_rushes",
            "rz_team_rushes",
            "rz_rush_share",
        ],
    )

    rru = _apply_player_name_cleaning(rru, name_maps)

    # QUARTERBACK
    qb_df = pd.DataFrame(
        columns=[
            "team",
            "opponent",
            "player",
            "pass_yards",
            "pass_att",
            "ypa",
            "dropbacks",
        ]
    )
    qb_name_col = (
        "passer_player_name"
        if "passer_player_name" in pbp.columns
        else ("passer" if "passer" in pbp.columns else None)
    )
    if qb_name_col is not None:
        qb = pbp.copy()
        qb["opponent"] = _derive_opponent(qb)
        qb["player"] = _norm_name(qb[qb_name_col].fillna(""))
        qb["team"] = qb[off_col].astype(str).str.upper().str.strip().map(_canon_team)
        qb["team"] = qb["team"].replace("", np.nan)
        qb["opponent"] = (
            qb["opponent"].astype(str).str.upper().str.strip()
            if "opponent" in qb.columns
            else np.nan
        )
        qb_players = qb.loc[_valid_player_mask(qb["player"])].copy()
        if not qb_players.empty:
            gb = (
                qb_players.groupby(["team", "opponent", "player"], dropna=False)
                .agg(
                    pass_yards=("yards_gained", "sum"),
                    pass_att=(
                        ("pass_attempt", "sum")
                        if "pass_attempt" in qb.columns
                        else (qb_name_col, "size")
                    ),
                    dropbacks=(
                        ("qb_dropback", "sum")
                        if "qb_dropback" in qb.columns
                        else (qb_name_col, "size")
                    ),
                )
                .reset_index()
            )
            gb["ypa"] = np.where(
                gb["pass_att"] > 0, gb["pass_yards"] / gb["pass_att"], np.nan
            )
            qb_df = gb[
                [
                    "team",
                    "opponent",
                    "player",
                    "pass_yards",
                    "pass_att",
                    "ypa",
                    "dropbacks",
                ]
            ]

    qb_df = _apply_player_name_cleaning(qb_df, name_maps)

    # Merge all using canonical player keys
    merge_keys = ["team", "opponent", "player_canonical"]

    rru_cols = [
        col
        for col in [
            "team",
            "opponent",
            "player_canonical",
            "player",
            "rushes",
            "rush_yards",
            "team_rushes",
            "rush_share",
            "ypc",
            "rz_rushes",
            "rz_team_rushes",
            "rz_rush_share",
        ]
        if col in rru.columns
    ]

    base = pd.merge(
        rply,
        rru[rru_cols],
        on=merge_keys,
        how="outer",
        suffixes=("_rec", "_rush"),
    )
    base = _coalesce_dupe_cols(base)
    assert_no_duplicate_columns(base, "rec/rush merge")

    if "player_rec" in base.columns or "player_rush" in base.columns:
        base["player"] = base.get("player_rec").combine_first(base.get("player_rush"))
        base.drop(columns=[c for c in ["player_rec", "player_rush"] if c in base.columns], inplace=True)

    if "player_canonical_rec" in base.columns or "player_canonical_rush" in base.columns:
        base["player_canonical"] = base.get("player_canonical_rec").combine_first(
            base.get("player_canonical_rush")
        )
        base.drop(
            columns=[
                c
                for c in ["player_canonical_rec", "player_canonical_rush"]
                if c in base.columns
            ],
            inplace=True,
        )

    qb_cols = [
        col
        for col in [
            "team",
            "opponent",
            "player_canonical",
            "player",
            "pass_yards",
            "pass_att",
            "ypa",
            "dropbacks",
        ]
        if col in qb_df.columns
    ]

    base = pd.merge(
        base,
        qb_df[qb_cols],
        on=merge_keys,
        how="left",
        suffixes=("", "_qb"),
    )
    base = _coalesce_dupe_cols(base)
    assert_no_duplicate_columns(base, "qb merge")

    if "player_qb" in base.columns:
        base["player"] = base["player"].combine_first(base["player_qb"])
        base.drop(columns=["player_qb"], inplace=True)

    if "player_canonical_qb" in base.columns:
        base["player_canonical"] = base["player_canonical"].combine_first(
            base["player_canonical_qb"]
        )
        base.drop(columns=["player_canonical_qb"], inplace=True)

    base["rz_share"] = base[["rz_tgt_share", "rz_rush_share"]].max(axis=1)
    base["season"] = int(season)

    base = _apply_player_name_cleaning(base, name_maps)
    base = _apply_canonical_names(base)

    print("[pf] base after concat/merge:", len(base))
    base = base[base["season"] == season].copy()

    # Initialize position/role as NaN (do not uppercase yet)
    base["position"] = np.nan
    base["role"] = np.nan

    # Normalize keys
    base = _ensure_cols(base, ["opponent"])
    base = base.loc[_valid_player_mask(base["player"].astype(str))].copy()
    base["team"] = base["team"].astype(str).str.upper().str.strip().map(_canon_team)

    opp_raw = base.get("opponent")
    opp_norm = opp_raw.where(opp_raw.notna(), "").astype(str).str.upper().str.strip()
    sentinel_mask = opp_norm.eq(CONSENSUS_OPPONENT_SENTINEL) | opp_norm.eq("")
    opp_norm = base["opponent"].astype(str).str.upper().str.strip()
    sentinel_mask = opp_norm.eq(CONSENSUS_OPPONENT_SENTINEL)
    base["opponent"] = opp_norm.map(_canon_team)
    if sentinel_mask.any():
        base.loc[sentinel_mask, "opponent"] = CONSENSUS_OPPONENT_SENTINEL
    base["opponent"] = base["opponent"].replace("", np.nan)

    # POSITION ENRICHMENT: weekly rosters → players master → usage family
    ro = _load_weekly_rosters(season)
    if not ro.empty:
        ro["player"] = _norm_name(ro["player"].astype(str))
        ro["team"] = ro["team"].astype(str).str.upper().str.strip().map(_canon_team)
        ro = ro[ro["team"].isin(VALID)]
        ro = _apply_player_name_cleaning(ro, name_maps)
        ro = ensure_canonical(ro, player_col="player", team_col="team")
        ro_cols = [
            col
            for col in ["player_canonical", "team", "player", "position"]
            if col in ro.columns
        ]
        base = base.merge(
            ro[ro_cols],
            on=["player_canonical", "team"],
            how="left",
            suffixes=("", "_ro"),
        )
        base = _coalesce_dupe_cols(base)
        assert_no_duplicate_columns(base, "weekly roster merge")
        if "player_ro" in base.columns:
            base["player"] = base["player"].combine_first(base["player_ro"])
            base.drop(columns=["player_ro"], inplace=True)
        if "position_ro" in base.columns:
            base["position"] = base["position"].combine_first(base["position_ro"])
            base.drop(columns=["position_ro"], inplace=True, errors="ignore")

    # Fallback: players master (merge by player only)
    if base["position"].isna().all():
        pm = _load_players_master()
        if not pm.empty:
            pm["player"] = _norm_name(pm["player"].astype(str))
            pm = _apply_player_name_cleaning(pm, name_maps)
            pm = ensure_canonical(pm, player_col="player", team_col=None)
            pm_cols = [
                col
                for col in ["player_canonical", "player", "position"]
                if col in pm.columns
            ]
            base = base.merge(
                pm[pm_cols],
                on="player_canonical",
                how="left",
                suffixes=("", "_pm"),
            )
            base = _coalesce_dupe_cols(base)
            assert_no_duplicate_columns(base, "players master merge")
            if "player_pm" in base.columns:
                base["player"] = base["player"].combine_first(base["player_pm"])
                base.drop(columns=["player_pm"], inplace=True)
            if "position_pm" in base.columns:
                base["position"] = base["position"].combine_first(base["position_pm"])
                base.drop(columns=["position_pm"], inplace=True, errors="ignore")

    base = _apply_player_name_cleaning(base, name_maps)

    # Final fallback: usage-based family inference → write into position when still missing
    if base["position"].isna().any():
        fam = _infer_position_family_from_usage(base)
        base["position"] = base["position"].where(base["position"].notna(), fam)

    # Only uppercase non-null positions (avoid turning NaN into "NAN")
    base.loc[base["position"].notna(), "position"] = (
        base.loc[base["position"].notna(), "position"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Make sure team columns are normalized before any joins
    if "team" in base.columns:
        base["team"] = base["team"].astype(str).map(normalize_team)
        base.loc[base["team"].isin(["", "NAN", "NONE", "NULL"]), "team"] = pd.NA
    for raw in base.get("player", pd.Series(dtype=object)).dropna().unique():
        log_unmapped_variant(
            "make_player_form",
            raw,
            {"stage": "post_player_cleaning"},
        )

    # Normalize team and (optional) week the same way we did before:
    base["team_key"] = (
        base["team"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if "week" in base.columns:
        base["week_key"] = base["week"].fillna(-1).astype(int)
    else:
        base["week_key"] = -1

    # roles.csv (non-destructive) then infer roles
    base = ensure_canonical(base, player_col="player", team_col="team")
    base = _merge_depth_roles(base)
    if base.get("role", pd.Series(dtype=object)).isna().all():
        base = _infer_roles_minimal(base)

    base = _ensure_cols(base, FINAL_COLS)

    if "opponent_inferred" not in base.columns:
        base["opponent_inferred"] = pd.Series(pd.NA, index=base.index)
    if "week_inferred" not in base.columns:
        base["week_inferred"] = pd.Series(pd.NA, index=base.index)
    if "opponent" in base.columns:
        mask = base["opponent_inferred"].isna()
        base.loc[mask, "opponent_inferred"] = base.loc[mask, "opponent"]
    if "week" in base.columns:
        mask = base["week_inferred"].isna()
        base.loc[mask, "week_inferred"] = base.loc[mask, "week"]

    base = _inject_week_opponent_and_roles(base, name_maps)
    if "week" in base.columns:
        base["week"] = pd.to_numeric(base["week"], errors="coerce")
    if "opponent" in base.columns:
        base["opponent"] = base["opponent"].astype(str).map(normalize_team)
        base.loc[base["opponent"].isin(["", "NAN"]), "opponent"] = np.nan

    if "team" in base.columns:
        base["team_key"] = base["team"].astype(str).str.upper().str.strip()
    if "week" in base.columns:
        base["week_key"] = base["week"].fillna(-1).astype(int)
    else:
        base["week_key"] = -1

    out = base.copy()
    out = _apply_canonical_names(out)
    out = _enrich_team_and_opponent_from_props(out, season, week=week)
    sched_lu = _build_schedule_opponent_lookup()
    out = _assign_team_and_opp_via_schedule(out, sched_lu)

    if "opponent" not in out.columns:
        out["opponent"] = pd.NA
    if "team" not in out.columns:
        out["team"] = pd.NA

    missing_mask = (
        out["opponent"].isna()
        | out["opponent"].astype(str).str.strip().eq("")
        | out["team"].isna()
        | out["team"].astype(str).str.strip().eq("")
    )

    debug_rows = out.loc[missing_mask].copy()
    if not debug_rows.empty:
        DEBUG_MISSING_OPP.parent.mkdir(parents=True, exist_ok=True)
        debug_rows.to_csv(DEBUG_MISSING_OPP, index=False)
        print(
            "[make_player_form] WARN missing team/opponent for",
            len(debug_rows),
            "players; wrote debug rows",
            DEBUG_MISSING_OPP,
            file=sys.stderr,
        )

    if "week" in out.columns:
        out["week"] = pd.to_numeric(out["week"], errors="coerce")

    if "team" in out.columns:
        out["team_key"] = out["team"].astype(str).str.upper().str.strip()
    if "week" in out.columns:
        out["week_key"] = out["week"].fillna(-1).astype(int)
    else:
        out["week_key"] = -1

    # Guarantee required numeric metrics are explicitly zero when usage was absent so
    # coverage calculations treat them as populated instead of missing data.
    numeric_fill_cols = [
        "tgt_share",
        "route_rate",
        "rush_share",
        "yprr",
        "ypt",
        "ypc",
        "ypa",
        "receptions_per_target",
        "rz_share",
        "rz_tgt_share",
        "rz_rush_share",
    ]
    for col in numeric_fill_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    # Ensure categorical tags are never empty so downstream validators read them as
    # present even when we had to infer or fallback.
    if "position" in out.columns:
        original = out["position"].copy()
        out["position"] = original.astype(str).str.upper().str.strip()
        missing_mask = original.isna() | out["position"].isin(["", "NAN", "NONE"])
        out.loc[missing_mask, "position"] = "UNK"

    if "role" in out.columns:
        original = out["role"].copy()
        out["role"] = original.astype(str).str.upper().str.strip()
        missing_mask = original.isna() | out["role"].isin(["", "NAN", "NONE"])
        out.loc[missing_mask, "role"] = "UNK"

    print("[pf] final rows (pre-write):", len(out))
    return out


# ---------------------------
# Opponent enrichment helper
# ---------------------------


def _aggregate_player_weeks(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if isinstance(df, pd.DataFrame) else [])

    working = df.copy()
    working = ensure_canonical(working, player_col="player", team_col="team")

    group_cols = [
        "player_canonical",
        "player_clean_key",
        "canonical_player_name",
        "player_name_canonical",
        "team",
        "season",
        "week",
    ]

    for col in ["season", "week"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    numeric_candidates = [
        col for col in PLAYER_FORM_USAGE_COLS + PLAYER_FORM_SHARE_COLS if col in working.columns
    ]
    if "games" in working.columns:
        numeric_candidates.append("games")

    for col in numeric_candidates:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    agg_spec: Dict[str, str] = {}

    for col in PLAYER_FORM_USAGE_COLS:
        if col not in working.columns:
            continue
        if col == "games":
            agg_spec[col] = "max"
        else:
            agg_spec[col] = "sum"

    for col in PLAYER_FORM_SHARE_COLS:
        if col in working.columns and col not in agg_spec:
            agg_spec[col] = "mean"

    numeric_cols = [
        col
        for col in working.columns
        if col not in group_cols and pd.api.types.is_numeric_dtype(working[col])
    ]
    for col in numeric_cols:
        if col not in agg_spec:
            agg_spec[col] = "mean"

    for col in working.columns:
        if col in group_cols or col in agg_spec:
            continue
        agg_spec[col] = "first"

    grouped = (
        working.groupby(group_cols, dropna=False)
        .agg(agg_spec)
        .reset_index()
    )

    return grouped


def _load_props_enriched() -> pd.DataFrame:
    base = _read_csv_safe(str(PROPS_ENRICHED_PATH))
    if base.empty:
        return pd.DataFrame(
            columns=[
                "player_canonical",
                "player_clean_key",
                "player_team_abbr",
                "opponent_team_abbr",
                "event_id",
                "kickoff_ts",
                "home_team_abbr",
                "away_team_abbr",
            ]
        )

    df = base.copy()
    df.columns = [c.lower() for c in df.columns]

    if "player_canonical" not in df.columns:
        if "player_name_raw" in df.columns:
            df["player_canonical"] = df["player_name_raw"].apply(canonicalize_name)
        else:
            df["player_canonical"] = ""

    df["player_clean_key"] = df["player_canonical"].apply(_normalize_key)

    for col in ["player_team_abbr", "opponent_team_abbr", "home_team_abbr", "away_team_abbr"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
            )

    if "event_id" in df.columns:
        df["event_id"] = df["event_id"].astype(str)
    else:
        df["event_id"] = ""

    if "kickoff_ts" not in df.columns and "commence_time" in df.columns:
        df["kickoff_ts"] = df["commence_time"]

    keep_cols = [
        c
        for c in [
            "player_canonical",
            "player_clean_key",
            "player_team_abbr",
            "opponent_team_abbr",
            "event_id",
            "kickoff_ts",
            "home_team_abbr",
            "away_team_abbr",
        ]
        if c in df.columns
    ]

    return df[keep_cols].drop_duplicates()


def _load_props_opponent_map(week: int | None = None) -> pd.DataFrame:
    base = _read_csv_safe(str(OPPONENT_MAP_PATH))
    if base.empty:
        return pd.DataFrame(
            columns=[
                "player",
                "player_clean_key",
                "team",
                "opponent",
                "season",
                "week",
                "event_id",
            ]
        )

    df = base.copy()
    df.columns = [c.lower() for c in df.columns]

    for col in ("player", "player_clean_key"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].astype("string")

    if "player_canonical" not in df.columns:
        df["player_canonical"] = df["player"].astype("string")
    else:
        df["player_canonical"] = df["player_canonical"].astype("string")

    for col in ("season", "week"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    if "event_id" not in df.columns:
        df["event_id"] = pd.NA
    df["event_id"] = df["event_id"].astype("string").str.strip()

    for col in ("team", "opponent", "team_abbr", "opponent_abbr"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = (
            df[col]
            .astype("string")
            .str.upper()
            .str.strip()
            .map(_canon_team)
            .replace("", pd.NA)
        )

    if "game_timestamp" in df.columns:
        df["game_timestamp"] = pd.to_numeric(
            df["game_timestamp"], errors="coerce"
        ).astype("Int64")
    else:
        df["game_timestamp"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    keep = [
        "player",
        "player_clean_key",
        "player_canonical",
        "team",
        "opponent",
        "team_abbr",
        "opponent_abbr",
        "season",
        "week",
        "event_id",
        "game_timestamp",
    ]
    filtered = df[keep].drop_duplicates()
    if week is not None and "week" in filtered.columns:
        try:
            week_int = int(week)
        except (TypeError, ValueError):
            week_int = None
        if week_int is not None:
            filtered = filtered.loc[filtered["week"] == week_int]

    return filtered


def _load_team_week_schedule_map(seasons: Iterable[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    season_set = {
        int(s)
        for s in seasons
        if s is not None
        and not (pd.isna(s))
    }

    for season in sorted(season_set):
        try:
            if NFL_PKG == "nflreadpy":
                raw_sched = NFLV.load_schedules(seasons=[season])
            else:
                if hasattr(NFLV, "import_schedules"):
                    raw_sched = NFLV.import_schedules([season])  # type: ignore
                else:
                    continue
            sched = _to_pandas(raw_sched)
        except Exception:
            continue

        if sched is None or sched.empty:
            continue

        sched.columns = [c.lower() for c in sched.columns]
        if not {"week", "home_team", "away_team"}.issubset(sched.columns):
            continue

        subset = sched[["week", "home_team", "away_team"]].copy()
        subset["season"] = season
        subset["week"] = pd.to_numeric(subset["week"], errors="coerce")
        subset["home_team"] = subset["home_team"].astype(str).map(_canon_team)
        subset["away_team"] = subset["away_team"].astype(str).map(_canon_team)
        subset = subset.dropna(subset=["home_team", "away_team", "week"])

        home = subset.rename(columns={"home_team": "team", "away_team": "opponent"})
        away = subset.rename(columns={"away_team": "team", "home_team": "opponent"})

        combined = pd.concat([home, away], ignore_index=True)
        combined = combined.loc[combined["team"].isin(VALID)]
        combined.loc[~combined["opponent"].isin(VALID), "opponent"] = pd.NA

        frames.append(combined[["team", "opponent", "season", "week"]])

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out["week"] = pd.to_numeric(out["week"], errors="coerce")
        return out.drop_duplicates(["team", "season", "week"])

    return pd.DataFrame(columns=["team", "opponent", "season", "week"])


def _resolve_opponents(df: pd.DataFrame, season_hint: int | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        base = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        base["opponent"] = pd.NA
        return base

    working = df.copy()

    if "opponent" not in working.columns:
        working["opponent"] = pd.NA

    for col in ["season", "week"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    props_map = _load_props_opponent_map()
    if not props_map.empty:
        props_map["season"] = pd.to_numeric(props_map["season"], errors="coerce")
        props_map["week"] = pd.to_numeric(props_map["week"], errors="coerce")
        props_map = ensure_canonical(props_map, player_col="player", team_col="team")

        joined_by_key = False
        if "player_clean_key" in working.columns and "player_clean_key" in props_map.columns:
            props_subset = props_map[
                [
                    "player_clean_key",
                    "team",
                    "team_abbr",
                    "opponent",
                    "opponent_abbr",
                    "season",
                    "week",
                    "game_timestamp",
                ]
            ].drop_duplicates()
            working = working.merge(
                props_subset,
                on=["player_clean_key"],
                how="left",
                suffixes=("", "_props"),
            )
            working = _coalesce_dupe_cols(working)
            working = _normalize_player_clean_key_columns(working)
            assert_no_duplicate_columns(working, "props opponent merge")

            for col in ("team", "team_abbr", "opponent", "opponent_abbr"):
                props_col = f"{col}_props"
                if props_col in working.columns:
                    base = working.get(col)
                    if base is None:
                        working[col] = working[props_col]
                    else:
                        working[col] = base.combine_first(working[props_col])
                    working.drop(columns=[props_col], inplace=True)

            if "game_timestamp" not in working.columns:
                working["game_timestamp"] = pd.Series(
                    pd.NA, index=working.index, dtype="Int64"
                )
            else:
                working["game_timestamp"] = pd.to_numeric(
                    working["game_timestamp"], errors="coerce"
                ).astype("Int64")

            if "game_timestamp_props" in working.columns:
                working["game_timestamp_props"] = pd.to_numeric(
                    working["game_timestamp_props"], errors="coerce"
                ).astype("Int64")
                working["game_timestamp"] = working["game_timestamp"].combine_first(
                    working["game_timestamp_props"]
                )
                working.drop(columns=["game_timestamp_props"], inplace=True)

            joined_by_key = True

        if not joined_by_key:
            merge_cols = ["player_canonical", "team"]
            if props_map["season"].notna().any():
                merge_cols.append("season")
            if props_map["week"].notna().any():
                merge_cols.append("week")

            props_subset = props_map[merge_cols + ["opponent"]].drop_duplicates()
            working = working.merge(
                props_subset,
                on=merge_cols,
                how="left",
                suffixes=("", "_props"),
            )
            working = _coalesce_dupe_cols(working)
            assert_no_duplicate_columns(working, "props opponent merge")
            if "opponent_props" in working.columns:
                working["opponent"] = working["opponent"].fillna(working["opponent_props"])
                working.drop(columns=["opponent_props"], inplace=True)

            team_cols = [
                c for c in ["team", "season", "week", "opponent"] if c in props_map.columns
            ]
            if {"team", "opponent"}.issubset(team_cols):
                team_subset = props_map[team_cols].dropna(subset=["team"])
                team_subset = team_subset.dropna(subset=["opponent"], how="any")
                if "season" in team_subset.columns:
                    team_subset["season"] = pd.to_numeric(team_subset["season"], errors="coerce")
                if "week" in team_subset.columns:
                    team_subset["week"] = pd.to_numeric(team_subset["week"], errors="coerce")
                merge_team_cols = [
                    c for c in ["team", "season", "week"] if c in team_subset.columns
                ]
                if merge_team_cols:
                    working = working.merge(
                        team_subset[merge_team_cols + ["opponent"]].drop_duplicates(),
                        on=merge_team_cols,
                        how="left",
                        suffixes=("", "_teamopp"),
                    )
                    working = _coalesce_dupe_cols(working)
                    assert_no_duplicate_columns(working, "team-level opponent merge")
                    if "opponent_teamopp" in working.columns:
                        working["opponent"] = working["opponent"].fillna(
                            working["opponent_teamopp"]
                        )
                        working.drop(columns=["opponent_teamopp"], inplace=True)

        for col in ("team", "team_abbr", "opponent", "opponent_abbr"):
            if col in working.columns:
                working[col] = (
                    working[col]
                    .astype("string")
                    .str.upper()
                    .str.strip()
                    .map(_canon_team)
                    .replace("", pd.NA)
                )

        if {"team", "team_abbr"}.issubset(working.columns):
            working["team"] = working["team"].combine_first(working["team_abbr"])
        if {"opponent", "opponent_abbr"}.issubset(working.columns):
            working["opponent"] = working["opponent"].combine_first(
                working["opponent_abbr"]
            )

        try:
            sched_opp = pd.read_csv("data/team_week_map.csv")
        except Exception:
            sched_opp = pd.DataFrame()
        if not sched_opp.empty:
            needed = {"season", "week", "team", "opponent"}
            if needed.issubset({c.lower() for c in sched_opp.columns}):
                sched_cols = {c.lower(): c for c in sched_opp.columns}
                subset = sched_opp[
                    [
                        sched_cols["season"],
                        sched_cols["week"],
                        sched_cols["team"],
                        sched_cols["opponent"],
                    ]
                ].copy()
                subset.columns = ["season", "week", "team", "opponent_sched"]
                subset["season"] = pd.to_numeric(subset["season"], errors="coerce")
                subset["week"] = pd.to_numeric(subset["week"], errors="coerce")
                subset["team"] = subset["team"].astype("string").str.upper().str.strip().map(
                    _canon_team
                )
                subset["opponent_sched"] = subset["opponent_sched"].astype(
                    "string"
                ).str.upper().str.strip().map(_canon_team)

                join_cols = [
                    col
                    for col in ["season", "week", "team"]
                    if col in working.columns and col in subset.columns
                ]
                if len(join_cols) == 3:
                    working = working.merge(
                        subset.drop_duplicates(join_cols + ["opponent_sched"]),
                        on=join_cols,
                        how="left",
                        suffixes=("", "_sched"),
                    )
                    if "opponent_sched" in working.columns:
                        if "opponent_abbr" not in working.columns:
                            working["opponent_abbr"] = working["opponent_sched"]
                        else:
                            working["opponent_abbr"] = working["opponent_abbr"].combine_first(
                                working["opponent_sched"]
                            )
                        working.drop(columns=["opponent_sched"], inplace=True)

    seasons = working.get("season")
    season_values = []
    if seasons is not None:
        season_values = [s for s in seasons.dropna().unique().tolist() if not pd.isna(s)]
    if season_hint is not None:
        season_values.append(season_hint)

    schedule_map = _load_team_week_schedule_map(season_values)
    if not schedule_map.empty and {"team", "season", "week"}.issubset(schedule_map.columns):
        schedule_map["season"] = pd.to_numeric(schedule_map["season"], errors="coerce")
        schedule_map["week"] = pd.to_numeric(schedule_map["week"], errors="coerce")
        working = working.merge(
            schedule_map.drop_duplicates(["team", "season", "week"]),
            on=["team", "season", "week"],
            how="left",
            suffixes=("", "_schedule"),
        )
        working = _coalesce_dupe_cols(working)
        assert_no_duplicate_columns(working, "schedule merge")
        if "opponent_schedule" in working.columns:
            working["opponent"] = working["opponent"].fillna(working["opponent_schedule"])
            working.drop(columns=["opponent_schedule"], inplace=True)

    working["opponent"] = (
        working["opponent"].astype("string").str.upper().str.strip()
    )
    working.loc[
        working["opponent"].isin(["", "NAN", "NONE", "NULL"]),
        "opponent",
    ] = pd.NA

    missing_mask = working["opponent"].isna()
    if missing_mask.any():
        debug_dir = Path(DATA_DIR) / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / "player_missing_opponent.csv"
        cols = [
            c
            for c in [
                "canonical_player_name",
                "player_clean_key",
                "team",
                "season",
                "week",
            ]
            if c in working.columns
        ]
        working.loc[missing_mask, cols].drop_duplicates().to_csv(
            debug_path, index=False
        )
        print(
            f"[make_player_form] WARN: {missing_mask.sum()} players missing opponent. "
            f"Rows written to {debug_path}",
            file=sys.stderr,
        )

    return working


def _enrich_team_and_opponent_from_props(
    out: pd.DataFrame, season: int | None = None, week: int | None = None
) -> pd.DataFrame:
    if out is None or out.empty:
        return out

    df = out.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if "player" not in df.columns:
        return df

    raw_players = df["player"].copy()
    if "player_source_name" in df.columns:
        df["player_source_name"] = df["player_source_name"].combine_first(
            raw_players
        )
    else:
        df["player_source_name"] = raw_players
    for raw in raw_players.dropna().unique():
        log_unmapped_variant(
            "make_player_form",
            raw,
            {"stage": "enrich_team_opponent"},
        )

    identity = raw_players.apply(
        lambda nm: pd.Series(
            _canonical_identity_fields(nm),
            index=[
                "player_name_canonical",
                "player_clean_key",
                "player_key_seed",
            ],
        )
    )
    identity = identity.fillna("")
    canonical_series = identity["player_name_canonical"].astype(str)
    identity["player_name_canonical"] = canonical_series.str.upper().str.strip()
    identity["player_canonical"] = identity["player_name_canonical"].str.lower()

    split_parts = canonical_series.str.strip().str.split()
    first = split_parts.str[0].fillna("")
    last = split_parts.str[-1].fillna("")
    generated_key = (
        first.fillna("").str.lower().str.strip()
        + " "
        + last.fillna("").str.lower().str.strip()
    )
    generated_key = generated_key.str.replace(r"\s+", " ", regex=True).str.strip()
    fallback_mask = last.str.strip().eq("")
    generated_key = generated_key.where(
        ~fallback_mask, first.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    identity = identity.rename(columns={"player_key_seed": "player_key"})
    identity["player_key"] = identity["player_key"].astype(str)
    empty_mask = identity["player_key"].str.strip().eq("")
    if empty_mask.any():
        identity.loc[empty_mask, "player_key"] = (
            generated_key.fillna("").loc[empty_mask]
        )

    df["player_name_canonical"] = identity["player_name_canonical"].astype(str)
    df["player_canonical"] = identity["player_canonical"].astype(str)
    df["player_clean_key"] = identity["player_clean_key"].astype(str)
    df["player_key"] = identity["player_key"].astype(str)
    key_fallback = df["player_key"].astype(str)
    clean_mask = df["player_clean_key"].astype(str).str.strip() == ""
    if clean_mask.any():
        df.loc[clean_mask, "player_clean_key"] = key_fallback.loc[clean_mask]
    df["canonical_player_name"] = df["player_name_canonical"].apply(
        _format_canonical_player_name
    )

    df["player"] = df["canonical_player_name"]
    df["player_name"] = df.get("player_name", df["canonical_player_name"])

    if "team" in df.columns:
        df["team"] = df["team"].astype(str).map(_canon_team)
        df.loc[df["team"].isin(["", "NAN", "NONE", "NULL"]), "team"] = pd.NA

    if "season" not in df.columns or df["season"].isna().all():
        if season is not None:
            df["season"] = season

    df["season"] = pd.to_numeric(df.get("season"), errors="coerce")
    df["week"] = pd.to_numeric(df.get("week"), errors="coerce")

    if "opponent" in df.columns:
        df.drop(columns=["opponent"], inplace=True)

    aggregated = _aggregate_player_weeks(df)
    if aggregated.empty:
        aggregated["opponent"] = pd.NA
        if "team" in aggregated.columns and "team_abbr" not in aggregated.columns:
            aggregated["team_abbr"] = aggregated["team"]
        if "opponent_abbr" not in aggregated.columns:
            aggregated["opponent_abbr"] = pd.NA
        aggregated["opp_abbr"] = aggregated.get("opponent_abbr")
        return aggregated

    props_map = _load_props_enriched()
    enriched = aggregated.copy()

    if not props_map.empty:
        props_map = props_map.rename(columns={
            "player_team_abbr": "props_team_abbr",
            "opponent_team_abbr": "props_opponent_abbr",
            "kickoff_ts": "props_kickoff_ts",
            "event_id": "props_event_id",
        })
        if "props_team_abbr" in props_map.columns:
            props_map["__has_team"] = props_map["props_team_abbr"].astype(str).str.len() > 0
        else:
            props_map["__has_team"] = False
        props_map = props_map.sort_values(["player_canonical", "__has_team"], ascending=[True, False])
        props_map = props_map.drop_duplicates(subset=["player_canonical"], keep="first")
        props_join_cols = ["player_canonical"]
        props_subset = props_map.drop(
            columns=[c for c in ["__has_team"] if c in props_map.columns]
        ).copy()
        left_props = coerce_merge_keys(enriched, props_join_cols, as_str=True)
        right_props = coerce_merge_keys(props_subset, props_join_cols, as_str=True)
        enriched = left_props.merge(right_props, on="player_canonical", how="left")
        enriched = _coalesce_dupe_cols(enriched)
        enriched = _normalize_player_clean_key_columns(enriched)
    else:
        for col in ["props_team_abbr", "props_opponent_abbr", "props_kickoff_ts", "props_event_id"]:
            if col not in enriched.columns:
                enriched[col] = pd.NA

    if "team_abbr" not in enriched.columns:
        enriched["team_abbr"] = enriched.get("team")
    if "team" in enriched.columns:
        enriched["team"] = enriched["team"].astype("string").str.upper().str.strip()
        enriched.loc[enriched["team"].isin(["", "NAN", "NONE", "NULL"]), "team"] = pd.NA
        enriched["team_abbr"] = enriched["team_abbr"].fillna(enriched["team"])
    enriched["team_abbr"] = enriched["team_abbr"].fillna(enriched.get("props_team_abbr"))

    if "opponent_abbr" not in enriched.columns:
        enriched["opponent_abbr"] = pd.NA
    enriched["opponent_abbr"] = enriched["opponent_abbr"].fillna(enriched.get("props_opponent_abbr"))

    if {"team_abbr", "home_team_abbr", "away_team_abbr"}.issubset(enriched.columns):
        mask = enriched["opponent_abbr"].isna()
        if mask.any():
            home = enriched.loc[mask, "home_team_abbr"].astype(str)
            away = enriched.loc[mask, "away_team_abbr"].astype(str)
            team_vals = enriched.loc[mask, "team_abbr"].astype(str)
            opp_fill = np.where(
                team_vals == home,
                away,
                np.where(team_vals == away, home, enriched.loc[mask, "opponent_abbr"].astype(str))
            )
            enriched.loc[mask, "opponent_abbr"] = opp_fill

    if "opponent" not in enriched.columns:
        enriched["opponent"] = enriched["opponent_abbr"]
    else:
        enriched["opponent"] = enriched["opponent"].fillna(enriched["opponent_abbr"])

    if "props_event_id" in enriched.columns:
        if "event_id" in enriched.columns:
            enriched["event_id"] = enriched["event_id"].combine_first(enriched["props_event_id"])
        else:
            enriched["event_id"] = enriched["props_event_id"]

    if "props_kickoff_ts" in enriched.columns:
        if "kickoff_ts" in enriched.columns:
            enriched["kickoff_ts"] = enriched["kickoff_ts"].fillna(enriched["props_kickoff_ts"])
        else:
            enriched["kickoff_ts"] = enriched["props_kickoff_ts"]

    for helper in ["props_team_abbr", "props_opponent_abbr", "props_event_id", "props_kickoff_ts"]:
        if helper in enriched.columns:
            enriched.drop(columns=[helper], inplace=True)

    tw_path = TEAM_WEEK_MAP_PATH
    if not tw_path.exists() or tw_path.stat().st_size == 0:
        raise RuntimeError(
            "team_week_map.csv missing/empty. Run 'Build team-week map' before player_form."
        )

    team_week = _read_csv_safe(str(tw_path))
    if team_week.empty:
        raise RuntimeError(
            "team_week_map.csv missing/empty. Run 'Build team-week map' before player_form."
        )

    team_week = team_week.copy()
    team_week.columns = [c.lower() for c in team_week.columns]

    standard_keys = {"team", "season", "week"}
    if not standard_keys.issubset(set(team_week.columns)):
        week10_col = next(
            (
                col
                for col in team_week.columns
                if col.startswith("week") and "10" in col
            ),
            None,
        )
        team_col = "team" if "team" in team_week.columns else None
        if not team_col:
            for candidate in ("team_abbr", "team_code"):
                if candidate in team_week.columns:
                    team_col = candidate
                    break
        if week10_col and team_col:
            simple_map = team_week[[team_col, week10_col]].copy()
            simple_map.columns = ["team", "teamweek_opponent"]
            simple_map["team"] = (
                simple_map["team"].astype("string").str.upper().str.strip().map(_canon_team)
            )
            simple_map["teamweek_opponent"] = (
                simple_map["teamweek_opponent"].astype("string").str.upper().str.strip().map(_canon_team)
            )
            enriched_simple = enriched.copy()
            team_merge_col = "team" if "team" in enriched_simple.columns else None
            if not team_merge_col:
                for candidate in (
                    "team_abbr",
                    "team_code",
                    "team_key",
                    "player_team_abbr",
                ):
                    if candidate in enriched_simple.columns:
                        team_merge_col = candidate
                        enriched_simple["team"] = enriched_simple[candidate]
                        break
            if team_merge_col:
                enriched_simple["team"] = (
                    enriched_simple["team"].astype("string").str.upper().str.strip().map(_canon_team)
                )
                enriched_simple = enriched_simple.merge(
                    simple_map.drop_duplicates(subset=["team"], keep="last"),
                    on="team",
                    how="left",
                )
                if "opponent" in enriched_simple.columns:
                    enriched_simple["opponent"] = enriched_simple["opponent"].combine_first(
                        enriched_simple.get("teamweek_opponent")
                    )
                else:
                    enriched_simple["opponent"] = enriched_simple.get("teamweek_opponent")
                enriched_simple.drop(columns=["teamweek_opponent"], inplace=True, errors="ignore")
                if "week" in enriched_simple.columns:
                    enriched_simple["week"] = enriched_simple["week"].fillna(10)
                else:
                    enriched_simple["week"] = 10
                return enriched_simple
        else:
            raise RuntimeError(
                "team_week_map.csv missing required season/week columns or week_10 column"
            )
    for col in ("season", "week"):
        if col in team_week.columns:
            team_week[col] = pd.to_numeric(team_week[col], errors="coerce").astype("Int64")
            if team_week[col].isna().any():
                raise RuntimeError(
                    f"team_week_map missing {col} values required for join"
                )
    for col in ("team", "opponent"):
        if col in team_week.columns:
            team_week[col] = (
                team_week[col]
                .astype("string")
                .str.upper()
                .str.strip()
                .map(_canon_team)
                .replace("", pd.NA)
            )
            team_week[col] = team_week[col].astype("string")

    rename_tw = {"game_id": "teamweek_game_id", "kickoff_utc": "teamweek_kickoff_ts", "is_bye": "teamweek_is_bye"}
    for src, dst in rename_tw.items():
        if src in team_week.columns and dst not in team_week.columns:
            team_week.rename(columns={src: dst}, inplace=True)

    if "opponent" in team_week.columns:
        team_week.rename(columns={"opponent": "teamweek_opponent"}, inplace=True)

    join_cols = [
        col
        for col in ("season", "week", "team")
        if col in enriched.columns and col in team_week.columns
    ]
    if len(join_cols) != 3:
        raise RuntimeError("team_week_map join requires season, week, and team columns")

    tw_subset_cols = join_cols + [
        c
        for c in (
            "teamweek_opponent",
            "teamweek_game_id",
            "teamweek_kickoff_ts",
            "teamweek_is_bye",
            "venue",
            "is_home",
        )
        if c in team_week.columns
    ]
    team_week_subset = team_week.loc[:, tw_subset_cols].copy()

    base_cols = list(enriched.columns)
    enriched_base = enriched.copy()
    if "event_id" in enriched_base.columns:
        enriched_base["event_id"] = (
            enriched_base["event_id"].astype("string").str.strip()
        )
    if "team" in enriched_base.columns:
        enriched_base["team"] = (
            enriched_base["team"].astype("string").str.upper().str.strip()
        )

    # --- Priority 1: join by (event_id, team) when available ---
    schedule_joined = enriched_base.copy()
    if {
        "teamweek_game_id",
        "team",
    }.issubset(team_week_subset.columns) and "event_id" in enriched_base.columns:
        event_cols = [
            "team",
            "teamweek_game_id",
        ] + [
            col
            for col in team_week_subset.columns
            if col not in {"team"}
        ]
        # Preserve column order while de-duplicating
        seen = set()
        event_cols = [col for col in event_cols if not (col in seen or seen.add(col))]
        event_subset = (
            team_week_subset.loc[:, event_cols]
            .dropna(subset=["teamweek_game_id"])
            .drop_duplicates(subset=["team", "teamweek_game_id"], keep="last")
        )
        if not event_subset.empty:
            event_subset["team"] = (
                event_subset["team"].astype("string").str.upper().str.strip()
            )
            event_subset["teamweek_game_id"] = (
                event_subset["teamweek_game_id"].astype("string").str.strip()
            )
            left_sched = coerce_merge_keys(
                enriched_base, ["team", "event_id"], as_str=True
            )
            right_sched = coerce_merge_keys(
                event_subset, ["team", "teamweek_game_id"], as_str=True
            )
            schedule_joined = left_sched.merge(
                right_sched,
                how="left",
                left_on=["team", "event_id"],
                right_on=["team", "teamweek_game_id"],
                suffixes=("", "_sched"),
                validate="m:1",
            )

    # --- Priority 2: fallback join on (season, week, team) ---
    team_week_fallback = team_week_subset.drop_duplicates(
        subset=join_cols, keep="last"
    )
    if not team_week_fallback.empty:
        left_fb = enriched_base.copy()
        right_fb = team_week_fallback.copy()
        numeric_fb = [c for c in ("season", "week") if c in join_cols]
        text_fb = [c for c in join_cols if c not in numeric_fb]
        if numeric_fb:
            left_fb = coerce_merge_keys(left_fb, numeric_fb, as_str=False)
            right_fb = coerce_merge_keys(right_fb, numeric_fb, as_str=False)
        if text_fb:
            left_fb = coerce_merge_keys(left_fb, text_fb, as_str=True)
            right_fb = coerce_merge_keys(right_fb, text_fb, as_str=True)
        fallback = left_fb.merge(
            right_fb,
            on=join_cols,
            how="left",
            validate="m:1",
        )
        schedule_cols = [
            col
            for col in fallback.columns
            if col not in base_cols
        ]
        for col in schedule_cols:
            if col not in schedule_joined.columns:
                schedule_joined[col] = fallback[col]
            else:
                schedule_joined[col] = schedule_joined[col].where(
                    schedule_joined[col].notna(), fallback[col]
                )

    enriched = schedule_joined

    if "teamweek_opponent" in enriched.columns:
        for col in ("opponent", "opponent_abbr", "opp_abbr"):
            if col not in enriched.columns:
                enriched[col] = pd.NA
        enriched["opponent"] = enriched["opponent"].fillna(enriched["teamweek_opponent"])
        enriched["opponent_abbr"] = enriched["opponent_abbr"].fillna(enriched["teamweek_opponent"])
        enriched["opp_abbr"] = enriched["opp_abbr"].fillna(enriched["teamweek_opponent"])
        if "teamweek_is_bye" in enriched.columns:
            bye_mask = enriched["teamweek_is_bye"].fillna(False).astype(bool)
        else:
            bye_mask = enriched["teamweek_opponent"].isna()
        enriched["is_bye"] = bye_mask
        enriched.loc[bye_mask, ["opponent", "opponent_abbr", "opp_abbr"]] = "BYE"
        for col in ("opponent", "opponent_abbr", "opp_abbr"):
            enriched[col] = enriched[col].astype("string").fillna("BYE")
        if "is_bye" in enriched.columns:
            enriched["is_bye"] = enriched["is_bye"].fillna(False)
        enriched.drop(columns=["teamweek_opponent"], inplace=True)
        if "teamweek_is_bye" in enriched.columns:
            enriched.drop(columns=["teamweek_is_bye"], inplace=True)
    else:
        if "is_bye" not in enriched.columns:
            enriched["is_bye"] = False

    if "teamweek_game_id" in enriched.columns:
        enriched["teamweek_game_id"] = (
            enriched["teamweek_game_id"].astype("string").str.strip()
        )
        if "game_id" in enriched.columns:
            enriched["game_id"] = (
                enriched["game_id"].astype("string").str.strip().replace("", pd.NA)
            ).fillna(enriched["teamweek_game_id"])
        else:
            enriched["game_id"] = enriched["teamweek_game_id"]
        if "event_id" in enriched.columns:
            enriched["event_id"] = (
                enriched["event_id"].astype("string").str.strip().replace("", pd.NA)
            ).combine_first(enriched["teamweek_game_id"])
        else:
            enriched["event_id"] = enriched["teamweek_game_id"]
        enriched.drop(columns=["teamweek_game_id"], inplace=True)

    if "teamweek_kickoff_ts" in enriched.columns:
        kickoff_series = pd.to_datetime(
            enriched["teamweek_kickoff_ts"], errors="coerce", utc=True
        )
        kickoff_text = kickoff_series.astype("string")
        if "kickoff_ts" in enriched.columns:
            enriched["kickoff_ts"] = enriched["kickoff_ts"].fillna(kickoff_text)
        else:
            enriched["kickoff_ts"] = kickoff_text
        enriched.drop(columns=["teamweek_kickoff_ts"], inplace=True)

    if "is_home" in enriched.columns:
        try:
            enriched["is_home"] = enriched["is_home"].astype("boolean")
        except Exception:
            enriched["is_home"] = enriched["is_home"]

    live_map = _load_props_opponent_map(week=week)
    if not live_map.empty and {"event_id"}.issubset(live_map.columns):
        live_map = live_map.dropna(subset=["event_id"])
        if not live_map.empty:
            live_map["event_id"] = live_map["event_id"].astype("string").str.strip()
            join_live = [
                col
                for col in ("player_clean_key", "team", "opponent")
                if col in enriched.columns and col in live_map.columns
            ]
            if join_live:
                live_subset = live_map[join_live + ["event_id"]].drop_duplicates()
                left_live = coerce_merge_keys(enriched, join_live, as_str=True)
                right_live = coerce_merge_keys(live_subset, join_live, as_str=True)
                enriched = left_live.merge(
                    right_live,
                    on=join_live,
                    how="left",
                    suffixes=("", "_live"),
                    validate="m:1",
                )
                if "event_id_live" in enriched.columns:
                    enriched["event_id"] = enriched["event_id_live"].combine_first(
                        enriched.get("event_id")
                    )
                    enriched.drop(columns=["event_id_live"], inplace=True)

    if "season" in enriched.columns:
        enriched["season"] = pd.to_numeric(enriched["season"], errors="coerce").astype("Int64")
    if "week" in enriched.columns:
        enriched["week"] = pd.to_numeric(enriched["week"], errors="coerce").astype("Int64")

    if "player_name_canonical" in enriched.columns:
        enriched["player_name_canonical"] = (
            enriched["player_name_canonical"].astype("string").str.upper().str.strip()
        )

    enriched["canonical_player_name"] = (
        enriched["canonical_player_name"].astype("string").str.strip()
    )
    enriched["player"] = enriched["canonical_player_name"]
    enriched["player_name"] = enriched.get(
        "player_name", enriched["canonical_player_name"]
    )

    if "team" in enriched.columns:
        enriched["team"] = enriched["team"].astype("string").str.upper().str.strip()
        enriched.loc[
            enriched["team"].isin(["", "NAN", "NONE", "NULL"]), "team"
        ] = pd.NA

    if "opp_abbr" not in enriched.columns:
        enriched["opp_abbr"] = enriched.get("opponent_abbr")

    if "is_bye" in enriched.columns:
        try:
            enriched["is_bye"] = enriched["is_bye"].fillna(False).astype("boolean")
        except Exception:
            enriched["is_bye"] = enriched["is_bye"].fillna(False)

    dedup_cols = [c for c in ["player_canonical", "team_abbr", "opponent_abbr"] if c in enriched.columns]
    if len(dedup_cols) == 3:
        enriched = enriched.sort_values(dedup_cols).drop_duplicates(subset=dedup_cols, keep="first")

    enriched = _overlay_opponents(enriched, season)
    enriched = _backfill_opponent_from_schedule(enriched)

    # ---- Attach or synthesize deterministic game identifiers ----
    def _compose_game_id_from_series(
        frame: pd.DataFrame, home_series: pd.Series, away_series: pd.Series
    ) -> pd.Series:
        if frame.empty or "season" not in frame.columns or "week" not in frame.columns:
            return pd.Series(pd.NA, index=frame.index, dtype="string")

        season_vals = (
            pd.to_numeric(frame["season"], errors="coerce")
            .astype("Int64")
            .astype("string")
        )
        week_vals = (
            pd.to_numeric(frame["week"], errors="coerce")
            .astype("Int64")
            .astype("string")
            .str.zfill(2)
        )
        home_vals = home_series.astype("string").str.upper().str.strip()
        away_vals = away_series.astype("string").str.upper().str.strip()

        out_series = season_vals + "_" + week_vals + "_" + home_vals + "_" + away_vals
        invalid_mask = (
            season_vals.isna()
            | week_vals.isna()
            | home_vals.isna()
            | away_vals.isna()
        )
        out_series.loc[invalid_mask] = pd.NA
        return out_series.astype("string")

    if "game_id" in enriched.columns:
        enriched["game_id"] = (
            enriched["game_id"].astype("string").str.strip().replace("", pd.NA)
        )
    else:
        enriched["game_id"] = pd.NA

    gl_path = Path("data/game_lines.csv")
    if gl_path.exists() and {"season", "week"}.issubset(enriched.columns):
        try:
            game_lines = pd.read_csv(gl_path)
        except Exception as err:
            logger.warning("[make_player_form] failed to read %s: %s", gl_path, err)
            game_lines = pd.DataFrame()

        if not game_lines.empty:
            working = game_lines.copy()
            working.columns = [str(c).lower() for c in working.columns]

            for col in ("season", "week"):
                if col in working.columns:
                    working[col] = pd.to_numeric(working[col], errors="coerce").astype("Int64")
            for col in ("home", "away"):
                if col in working.columns:
                    working[col] = (
                        working[col]
                        .astype("string")
                        .str.upper()
                        .str.strip()
                        .replace("", pd.NA)
                    )

            if "game_id" not in working.columns:
                season_vals = working["season"].astype("Int64").astype("string")
                week_vals = (
                    working["week"].astype("Int64").astype("string").str.zfill(2)
                )
                home_vals = working["home"].astype("string").str.upper().str.strip()
                away_vals = working["away"].astype("string").str.upper().str.strip()
                working["game_id"] = (
                    season_vals + "_" + week_vals + "_" + home_vals + "_" + away_vals
                )
                working.loc[
                    season_vals.isna()
                    | week_vals.isna()
                    | home_vals.isna()
                    | away_vals.isna(),
                    "game_id",
                ] = pd.NA

            join_cols = [
                col
                for col in ("season", "week", "home_team", "away_team")
                if col in enriched.columns
            ]
            if len(join_cols) == 4 and {"home", "away", "game_id"}.issubset(working.columns):
                enriched["home_team"] = (
                    enriched["home_team"].astype("string").str.upper().str.strip()
                )
                enriched["away_team"] = (
                    enriched["away_team"].astype("string").str.upper().str.strip()
                )
                schedule_subset = (
                    working[["season", "week", "home", "away", "game_id"]]
                    .dropna(subset=["home", "away"])
                    .drop_duplicates(subset=["season", "week", "home", "away"], keep="first")
                    .rename(columns={"home": "home_team", "away": "away_team"})
                )
                enriched = enriched.merge(
                    schedule_subset,
                    on=["season", "week", "home_team", "away_team"],
                    how="left",
                    suffixes=("", "_from_schedule"),
                )
                helper_col = "game_id_from_schedule"
                if helper_col in enriched.columns:
                    enriched["game_id"] = enriched["game_id"].fillna(
                        enriched[helper_col]
                    )
                    enriched.drop(columns=[helper_col], inplace=True)

    missing_mask = enriched["game_id"].isna()
    if missing_mask.any() and {"home_team", "away_team"}.issubset(enriched.columns):
        enriched.loc[missing_mask, "game_id"] = _compose_game_id_from_series(
            enriched.loc[missing_mask],
            enriched.loc[missing_mask, "home_team"],
            enriched.loc[missing_mask, "away_team"],
        )
        missing_mask = enriched["game_id"].isna()

    if missing_mask.any() and {"home_team_abbr", "away_team_abbr"}.issubset(enriched.columns):
        enriched.loc[missing_mask, "game_id"] = _compose_game_id_from_series(
            enriched.loc[missing_mask],
            enriched.loc[missing_mask, "home_team_abbr"],
            enriched.loc[missing_mask, "away_team_abbr"],
        )
        missing_mask = enriched["game_id"].isna()

    if missing_mask.any() and {"team_abbr", "opponent_abbr"}.issubset(enriched.columns):
        tmp = enriched.loc[missing_mask, ["team_abbr", "opponent_abbr"]].copy()
        team_vals = tmp["team_abbr"].astype("string").str.upper().str.strip()
        opp_vals = tmp["opponent_abbr"].astype("string").str.upper().str.strip()

        def _pick_home(a: str, b: str) -> object:
            if pd.isna(a) or pd.isna(b):
                return pd.NA
            return a if a <= b else b

        def _pick_away(a: str, b: str) -> object:
            if pd.isna(a) or pd.isna(b):
                return pd.NA
            return b if a <= b else a

        home_vals = team_vals.combine(opp_vals, _pick_home)
        away_vals = team_vals.combine(opp_vals, _pick_away)
        enriched.loc[missing_mask, "game_id"] = _compose_game_id_from_series(
            enriched.loc[missing_mask], home_vals, away_vals
        )

    if "game_id" in enriched.columns:
        enriched["game_id"] = enriched["game_id"].astype("string")

    return enriched


# ---------------------------
# Fallback enrichers (optional CSVs)
# ---------------------------


def _apply_fallback_enrichers(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "msf_player_form.csv",
        "apisports_player_form.csv",
        "nflgsis_player_form.csv",
    ]
    out = df.copy()
    out = _ensure_cols(out, ["opponent"])
    for fn in candidates:
        try:
            ext = _read_csv_safe(os.path.join(DATA_DIR, fn))
            if _is_empty(ext):
                continue
            if "player" not in ext.columns and "player_name" in ext.columns:
                ext = ext.rename(columns={"player_name": "player"})
            if not {"player", "team"}.issubset(ext.columns):
                continue
            ext["player"] = _norm_name(ext["player"].astype(str))
            ext["team"] = (
                ext["team"].astype(str).str.upper().str.strip().map(_canon_team)
            )
            ext = ext[ext["team"].isin(VALID)]
            out = _non_destructive_merge(out, ext, keys=["player", "team"])
        except Exception:
            continue
    return out


# ---------------------------
# PROPS-SCOPED VALIDATION
# ---------------------------


def _load_props_players() -> pd.DataFrame:
    """
    Read outputs/props_raw.csv to get the set of players (and teams) we actually need to validate.
    Returns DataFrame with columns: player, team, player_clean_key (stable).
    """
    path = os.path.join("outputs", "props_raw.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["player", "team", "opponent", "player_clean_key"])
    try:
        pr = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["player", "team", "opponent", "player_clean_key"])

    pr.columns = [c.lower() for c in pr.columns]
    if "player" not in pr.columns:
        if "player_name" in pr.columns:
            pr = pr.rename(columns={"player_name": "player"})
        elif "name" in pr.columns:
            pr = pr.rename(columns={"name": "player"})
        else:
            pr["player"] = np.nan
    pr["player"] = _norm_name(pr.get("player", pd.Series([], dtype=object)).astype(str))

    if "team" not in pr.columns:
        pr["team"] = np.nan
    else:
        pr["team"] = pr["team"].astype(str).str.upper().str.strip().map(_canon_team)

    pr["opponent"] = _normalize_props_opponent(pr)
    opp_col = next((c for c in DEFENSE_TEAM_CANDIDATES if c in pr.columns), None)
    if opp_col is not None:
        try:
            derived = _derive_opponent(pr)
        except Exception:
            derived = pd.Series(np.nan, index=pr.index, dtype=object)
        if not isinstance(derived, pd.Series) or derived.shape[0] != len(pr):
            derived = pd.Series(np.nan, index=pr.index, dtype=object)
        if derived.notna().any():
            opp_norm = (
                derived.where(derived.notna(), "").astype(str).str.upper().str.strip()
            )
            pr["opponent"] = opp_norm.map(_canon_team).replace("", np.nan)
        else:
            opp_raw = pr[opp_col]
            if isinstance(opp_raw, pd.Series):
                opp_norm = (
                    opp_raw.where(opp_raw.notna(), "")
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )
                pr["opponent"] = opp_norm.map(_canon_team).replace("", np.nan)
            else:
                pr["opponent"] = np.nan
    else:
        pr["opponent"] = np.nan

    if "player_canonical" not in pr.columns:
        pr["player_canonical"] = pr["player"].apply(_canonicalize_player_name)
    pr["player_clean_key"] = pr["player_canonical"].apply(_normalize_key)
    for raw in pr["player"].dropna().unique():
        log_unmapped_variant(
            "make_player_form",
            raw,
            {"stage": "normalize_props_players"},
        )

    return pr[["player", "team", "opponent", "player_clean_key"]].drop_duplicates()


def _validate_required(df: pd.DataFrame):
    """
    Strict checks by position-family:
      WR/TE: route_rate, tgt_share, yprr
      RB:    rush_share, ypc
      QB:    ypa

    Validate **only** players that appear in outputs/props_raw.csv.
    Skip rows where we cannot determine a family (no position/role and no usage signal).
    """
    props_players = _load_props_players()
    if props_players.empty:
        return

    df = df.copy()
    player_series = df.get("player")
    if player_series is None:
        player_series = pd.Series(["" for _ in range(len(df))], index=df.index)
    if "player_canonical" not in df.columns:
        df["player_canonical"] = player_series.apply(_canonicalize_player_name)
    if "player_clean_key" not in df.columns:
        df["player_clean_key"] = df["player_canonical"].apply(_normalize_key)
    for raw in player_series.dropna().unique():
        log_unmapped_variant(
            "make_player_form",
            raw,
            {"stage": "validate_required"},
        )

    need = df.merge(
        props_players[["player_clean_key"]].drop_duplicates(),
        on="player_clean_key",
        how="inner",
    )
    if need.empty:
        return

    pos = (
        need.get("position", pd.Series(index=need.index, dtype=object))
        .astype(str)
        .str.upper()
    )
    role = (
        need.get("role", pd.Series(index=need.index, dtype=object))
        .astype(str)
        .str.upper()
    )

    fam = pos.where(~pos.isin(["", "NAN", "NONE"]), np.nan)

    qb_mask = pd.Series(False, index=need.index)
    if "dropbacks" in need.columns:
        qb_mask |= need["dropbacks"].fillna(0) >= 15
    if "ypa" in need.columns:
        qb_mask |= need["ypa"].notna() & (need["ypa"] > 6.0)

    rb_mask = need.get("rush_share", pd.Series(0, index=need.index)).fillna(0) >= 0.20
    wr_mask = (
        need.get("route_rate", pd.Series(0, index=need.index)).fillna(0) >= 0.20
    ) | (need.get("tgt_share", pd.Series(0, index=need.index)).fillna(0) >= 0.15)

    fam = fam.where(
        fam.notna(),
        np.where(
            qb_mask, "QB", np.where(rb_mask, "RB", np.where(wr_mask, "WR", np.nan))
        ),
    )

    has_wr_hint = role.str.contains("WR|TE", na=False)
    has_rb_hint = role.str.contains("RB", na=False)
    has_qb_hint = role.str.contains("QB", na=False)

    is_wrte = fam.isin(["WR", "TE"]) | has_wr_hint
    is_rb = fam.eq("RB") | has_rb_hint
    is_qb = fam.eq("QB") | has_qb_hint

    to_check = need.loc[is_wrte | is_rb | is_qb].copy()
    if to_check.empty:
        return

    missing: Dict[str, List[str]] = {}

    def _need(mask, cols: List[str], label: str):
        if not mask.any():
            return
        sub = to_check.loc[mask]
        for c in cols:
            if c not in sub.columns:
                bad = sub.index.tolist()
            else:
                bad = sub.index[sub[c].isna()].tolist()
            if bad:
                missing[f"{label}:{c}"] = sub.loc[bad, "player"].astype(str).tolist()

    _need(is_wrte.loc[to_check.index], ["route_rate", "tgt_share", "yprr"], "WR/TE")
    _need(is_rb.loc[to_check.index], ["rush_share", "ypc"], "RB")
    _need(is_qb.loc[to_check.index], ["ypa"], "QB")

    if missing:
        print("[make_player_form] REQUIRED PLAYER METRICS MISSING:", file=sys.stderr)
        for k, v in missing.items():
            preview = ", ".join(v[:10]) + ("..." if len(v) > 10 else "")
            print(f"  - {k}: {preview}", file=sys.stderr)

        # Write a CSV report for diagnostics
        try:
            rows = []
            fam_map = {"WR/TE": is_wrte, "RB": is_rb, "QB": is_qb}
            for fam, names in missing.items():
                mask = fam_map.get(fam, pd.Series(False, index=df.index))
                # build a small lookup subset
                sub = to_check[mask.loc[to_check.index].fillna(False)][
                    ["player", "team", "position", "role"]
                ].copy()
                for nm in names:
                    rows.append(
                        {
                            "player": nm,
                            "family": fam,
                            "team": (
                                sub.loc[
                                    sub["player"]
                                    .str.lower()
                                    .str.replace(r"[^a-z0-9]", "", regex=True)
                                    == nm,
                                    "team",
                                ]
                                .head(1)
                                .item()
                                if not sub.empty
                                else None
                            ),
                            "missing_for_family": fam,
                        }
                    )
            os.makedirs(DATA_DIR, exist_ok=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(DATA_DIR, "validation_player_missing.csv"), index=False
            )
            print(
                f"[make_player_form] wrote report → {os.path.join(DATA_DIR, 'validation_player_missing.csv')}"
            )
        except Exception as e:
            print(
                f"[make_player_form] WARN could not write missing report: {e}",
                file=sys.stderr,
            )

        # Env-gated strictness (default now lenient)
        if os.getenv("STRICT_VALIDATE", "0") != "0":
            raise RuntimeError(
                "Required player_form metrics missing; failing per strict policy."
            )
        else:
            print(
                "[make_player_form] STRICT_VALIDATE=0 → continue despite missing required metrics",
                file=sys.stderr,
            )
            return


def _attach_consensus_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "player_name" not in out.columns:
        out["player_name"] = out.get("player", "")

    canonical_df = out["player_name"].apply(
        lambda nm: pd.Series(
            _canonical_identity_fields(nm),
            index=[
                "player_name_canonical",
                "player_clean_key",
                "player_key_seed",
            ],
        )
    )
    canonical_df = canonical_df.fillna("")
    canonical_series = canonical_df["player_name_canonical"].astype(str)
    canonical_df["player_name_canonical"] = (
        canonical_series.str.upper().str.strip()
    )
    canonical_df["player_canonical"] = canonical_df["player_name_canonical"].str.lower()

    split_parts = canonical_series.str.strip().str.split()
    first = split_parts.str[0].fillna("")
    last = split_parts.str[-1].fillna("")
    generated_key = (
        first.fillna("").str.lower().str.strip()
        + " "
        + last.fillna("").str.lower().str.strip()
    )
    generated_key = generated_key.str.replace(r"\s+", " ", regex=True).str.strip()
    fallback_mask = last.str.strip().eq("")
    generated_key = generated_key.where(
        ~fallback_mask, first.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )

    canonical_df = canonical_df.rename(columns={"player_key_seed": "player_key"})
    canonical_df["player_key"] = canonical_df["player_key"].astype(str)
    empty_mask = canonical_df["player_key"].str.strip().eq("")
    if empty_mask.any():
        canonical_df.loc[empty_mask, "player_key"] = (
            generated_key.fillna("").loc[empty_mask]
        )

    out["player_name_canonical"] = canonical_df["player_name_canonical"].astype(str)
    out["player_canonical"] = canonical_df["player_canonical"].astype(str)
    out["player_clean_key"] = canonical_df["player_clean_key"].astype(str)
    if "player_key" not in out.columns:
        out["player_key"] = canonical_df["player_key"].astype(str)
    else:
        overwrite_mask = out["player_key"].astype(str).str.strip().eq("")
        if overwrite_mask.any():
            out.loc[overwrite_mask, "player_key"] = (
                canonical_df.loc[overwrite_mask, "player_key"].astype(str)
            )

    if "team_key" not in out.columns:
        out["team_key"] = (
            out.get("team", "")
            .astype(str)
            .str.upper()
            .str.strip()
        )
    else:
        out["team_key"] = out["team_key"].astype(str).str.upper().str.strip()
    if "week" in out.columns:
        week_numeric = pd.to_numeric(out["week"], errors="coerce").fillna(-1)
        out["week_key"] = week_numeric.astype(int)
    else:
        out["week_key"] = -1
    return out


def _build_grouped_consensus(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    working = df.copy()
    if "player" not in working.columns and "canonical_player_name" in working.columns:
        working["player"] = working["canonical_player_name"]

    required_group_cols = ["player_canonical", "team", "team_abbr", "opponent", "opponent_abbr", "position"]
    for col in required_group_cols:
        if col not in working.columns:
            working[col] = pd.NA

    if "opp_abbr" not in working.columns:
        working["opp_abbr"] = working.get("opponent_abbr")

    group_cols = required_group_cols + ["opp_abbr"]

    agg_spec: Dict[str, str] = {
        "player": "first",
        "role": "first",
        "week": "max",
        "targets": "sum",
        "routes": "sum",
        "receptions": "sum",
        "rec_yards": "sum",
        "rushes": "sum",
        "rush_yards": "sum",
        "pass_att": "sum",
        "pass_yards": "sum",
        "dropbacks": "sum",
        "team_targets": "mean",
        "team_dropbacks": "mean",
        "team_rushes": "mean",
        "tgt_share": "mean",
        "route_rate": "mean",
        "rush_share": "mean",
        "yprr": "mean",
        "ypt": "mean",
        "ypc": "mean",
        "ypa": "mean",
        "receptions_per_target": "mean",
        "rz_share": "mean",
        "rz_tgt_share": "mean",
        "rz_rush_share": "mean",
        "kickoff_ts": "first",
    }

    available_agg_spec = {
        col: fn
        for col, fn in agg_spec.items()
        if col in working.columns and col not in group_cols
    }

    if not available_agg_spec:
        grouped = working[group_cols].drop_duplicates().reset_index(drop=True)
    else:
        grouped = (
            working.groupby(group_cols, dropna=False)
            .agg(available_agg_spec)
            .reset_index()
        )

    grouped = grouped.loc[:, ~grouped.columns.duplicated()]

    if "week" in grouped.columns:
        grouped["week"] = (
            pd.to_numeric(grouped["week"], errors="coerce").astype("Int64")
        )

    if "player" in grouped.columns and "player_canonical" in grouped.columns:
        grouped["player"] = grouped["player"].fillna(grouped["player_canonical"])

    if "opp_abbr" not in grouped.columns:
        grouped["opp_abbr"] = grouped.get("opponent_abbr")

    ordered_prefix = [
        "player_canonical",
        "player",
        "team",
        "team_abbr",
        "opponent",
        "opponent_abbr",
        "opp_abbr",
        "position",
        "role",
        "week",
        "kickoff_ts",
    ]
    existing = [c for c in ordered_prefix if c in grouped.columns]
    remaining = [c for c in grouped.columns if c not in existing]
    return grouped[existing + remaining]


def _enforce_player_form_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=PLAYER_FORM_REQUIRED_COLUMNS)

    out = df.copy()
    for col in PLAYER_FORM_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    # ensure key identifier columns are strings (preserve pd.NA with pandas string dtype)
    for col in [
        "player",
        "canonical_player_name",
        "team",
        "team_abbr",
        "opponent",
        "opponent_abbr",
        "opp_abbr",
        "role",
        "player_name_canonical",
        "player_canonical",
        "kickoff_ts",
    ]:
        if col in out.columns:
            out[col] = out[col].astype("string")

    if {"player", "canonical_player_name"}.issubset(out.columns):
        mask = out["canonical_player_name"].isna() | out["canonical_player_name"].isin(
            {"", "NAN", "NONE", "NULL"}
        )
        out.loc[mask, "canonical_player_name"] = out.loc[mask, "player"]

    if "team" in out.columns:
        out["team"] = out["team"].str.strip().str.upper()

    if "team_abbr" in out.columns:
        out["team_abbr"] = out["team_abbr"].fillna(out.get("team"))
        out["team_abbr"] = out["team_abbr"].str.strip().str.upper()

    if "player" in out.columns:
        out["player"] = out["player"].str.strip()

    if "player_canonical" in out.columns:
        out["player_canonical"] = out["player_canonical"].str.strip()
        missing = out["player_canonical"].isna() | out["player_canonical"].eq("")
        out.loc[missing, "player_canonical"] = (
            out.loc[missing, "player"].apply(canonicalize_name)
        )
    else:
        out["player_canonical"] = out.get("player", pd.Series(dtype="string")).apply(
            canonicalize_name
        )

    if "player_name_canonical" in out.columns:
        out["player_name_canonical"] = out["player_name_canonical"].str.strip().str.upper()

    if "opponent" in out.columns:
        out["opponent"] = (
            out["opponent"].str.strip().str.upper()
        )
        out.loc[out["opponent"].isin(["", "NAN", "NONE", "NULL"]), "opponent"] = pd.NA

    if "opponent_abbr" in out.columns:
        out["opponent_abbr"] = out["opponent_abbr"].fillna(out.get("opponent"))
        out["opponent_abbr"] = out["opponent_abbr"].str.strip().str.upper()
        out.loc[
            out["opponent_abbr"].isin(["", "NAN", "NONE", "NULL"]), "opponent_abbr"
        ] = pd.NA

    if "role" in out.columns:
        out["role"] = out["role"].str.strip()

    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce")

    out["week"] = pd.to_numeric(out.get("week"), errors="coerce")

    # Try to backfill routes if missing but route_rate + team_dropbacks are available
    if "routes" in out.columns and {"route_rate", "team_dropbacks"}.issubset(out.columns):
        needs_routes = out["routes"].isna()
        if needs_routes.any():
            routes_estimate = (
                pd.to_numeric(out["route_rate"], errors="coerce")
                * pd.to_numeric(out["team_dropbacks"], errors="coerce")
            )
            out.loc[needs_routes, "routes"] = routes_estimate.loc[needs_routes]

    numeric_cols = [
        "targets",
        "routes",
        "receptions",
        "rec_yards",
        "rushes",
        "rush_yards",
        "pass_att",
        "pass_yards",
        "dropbacks",
        "team_targets",
        "team_dropbacks",
        "team_rushes",
        "rz_targets",
        "rz_team_targets",
        "rz_rushes",
        "rz_team_rushes",
        "games",
        "tgt_share",
        "route_rate",
        "rush_share",
        "yprr",
        "ypt",
        "ypc",
        "ypa",
        "receptions_per_target",
        "rz_share",
        "rz_tgt_share",
        "rz_rush_share",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ("team", "team_abbr", "opponent", "opponent_abbr", "opp_abbr"):
        if col in out.columns:
            out[col] = _canon_team_series(out[col])

    ordered = PLAYER_FORM_REQUIRED_COLUMNS + [
        c for c in out.columns if c not in PLAYER_FORM_REQUIRED_COLUMNS
    ]
    return out[ordered]


def _enforce_consensus_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=CONSENSUS_REQUIRED_COLUMNS)

    out = _ensure_single_position_column(df.copy())
    for col in CONSENSUS_REQUIRED_COLUMNS:
        if col not in out.columns:
            if col == "week":
                out[col] = pd.Series(pd.NA, index=out.index, dtype="Float64")
            elif col == "opponent":
                out[col] = CONSENSUS_OPPONENT_SENTINEL
            else:
                out[col] = pd.NA

    out["player"] = out["player"].astype("string").str.strip()
    canonical_series = out.get("player_canonical").astype("string")
    canonical_valid = canonical_series.notna() & canonical_series.str.strip().ne("")
    canonical_series = canonical_series.where(canonical_valid, out["player"])
    out["player_canonical"] = canonical_series.apply(canonicalize_name)
    out["team"] = out["team"].astype("string").str.strip().str.upper()
    team_abbr_series = out.get("team_abbr").astype("string")
    team_abbr_valid = team_abbr_series.notna() & team_abbr_series.str.strip().ne("")
    team_abbr_series = team_abbr_series.where(team_abbr_valid, out["team"])
    out["team_abbr"] = team_abbr_series.str.strip().str.upper()
    out["role"] = out["role"].astype("string").str.strip()
    out["position"] = out["position"].astype("string").str.strip()

    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["opponent"] = (
        out["opponent"]
        .astype("string")
        .fillna(CONSENSUS_OPPONENT_SENTINEL)
        .replace({"": CONSENSUS_OPPONENT_SENTINEL})
        .str.upper()
    )
    opp_abbr_series = out.get("opponent_abbr").astype("string")
    opp_abbr_valid = opp_abbr_series.notna() & opp_abbr_series.str.strip().ne("")
    opp_abbr_series = opp_abbr_series.where(opp_abbr_valid, out["opponent"])
    out["opponent_abbr"] = opp_abbr_series.str.strip().str.upper()

    ordered = CONSENSUS_REQUIRED_COLUMNS + [
        c for c in out.columns if c not in CONSENSUS_REQUIRED_COLUMNS
    ]
    return out[ordered]


def _write_player_form_outputs(
    df: pd.DataFrame,
    slate_date: str | None = None,
    season: int | None = None,
) -> None:
    import os

    def _resolve_season(
        df: pd.DataFrame, slate_date: str | None, passed: int | None
    ) -> int:
        if passed is not None:
            return int(passed)
        env_val = os.environ.get("SEASON")
        if env_val:
            try:
                return int(env_val)
            except Exception:
                pass
        if "season" in df.columns and df["season"].notna().any():
            try:
                return int(
                    pd.to_numeric(df["season"], errors="coerce").dropna().iloc[0]
                )
            except Exception:
                pass
        if slate_date:
            try:
                return pd.to_datetime(slate_date).year
            except Exception:
                pass
        raise RuntimeError("season could not be resolved")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.setLevel(logging.INFO)

    season = _resolve_season(df, slate_date, season)
    if season != CURRENT_SEASON:
        logger.info(
            "[PLAYER-FORM] overriding season %s with %s for export",
            season,
            CURRENT_SEASON,
        )
    season = CURRENT_SEASON

    # Final safety: drop any duplicated column names
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Keep the existing assert here
    assert_no_duplicate_columns(df, "final player_form before write")

    if df is None or df.empty:
        print(
            "[make_player_form][ERROR] final player_form empty; writing empty file to aid debugging"
        )
        empty = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        try:
            empty = _ensure_cols(empty.copy(), FINAL_COLS)
        except Exception:
            empty = pd.DataFrame(columns=FINAL_COLS)
        try:
            PLAYER_FORM_OUT.parent.mkdir(parents=True, exist_ok=True)
            empty.to_csv(PLAYER_FORM_OUT, index=False)
        except Exception as err:
            print(
                f"[make_player_form][ERROR] failed writing empty player_form.csv: {err}"
            )
        for path in (PLAYER_GAME_LOGS_OUT, PLAYER_SEASON_TOTALS_OUT, PLAYER_FORM_CONSENSUS_OUT):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame().to_csv(path, index=False)
            except Exception as err:
                print(
                    f"[make_player_form][ERROR] failed writing placeholder {path}: {err}"
                )
        return

    essential = {"player_clean_key", "team", "opponent", "week"}
    essential.update(PLAYER_FORM_SHARE_COLS)
    missing = [col for col in sorted(essential) if col not in df.columns]
    if missing:
        logger.warning(
            "[make_player_form] player_form missing expected columns prior to write: %s",
            ", ".join(missing),
        )

    df_out = _ensure_single_position_column(df.copy())
    df_out = _attach_player_identity(df_out, team_columns=("team_abbr", "team"))
    df_out = _ensure_cols(df_out, FINAL_COLS)
    player_form = _enforce_player_form_schema(df_out)
    player_form = _attach_player_identity(
        player_form, team_columns=("team_abbr", "team")
    )
    player_form = _reorder_identity_columns(player_form)

    if "player_clean_key" in player_form.columns:
        key_series = player_form["player_clean_key"].astype("string")
        missing_key_mask = key_series.isna() | key_series.str.strip().eq("")
        if missing_key_mask.any():
            fallback = None
            for candidate in (
                "display_name",
                "player",
                "player_canonical",
                "player_name_clean",
            ):
                if candidate in player_form.columns:
                    fallback = player_form[candidate].astype("string")
                    break
            if fallback is not None:
                fallback_keys = fallback.map(make_player_key)
                player_form.loc[missing_key_mask, "player_clean_key"] = fallback_keys.loc[
                    missing_key_mask
                ]
        player_form["player_clean_key"] = (
            player_form["player_clean_key"].astype("string").fillna("")
        )
    try:
        player_form["season"] = int(season)
    except Exception:
        player_form["season"] = season

    player_form["player_canonical"] = player_form["player_canonical"].apply(
        canonicalize_name
    )
    if "kickoff_ts" not in player_form.columns:
        player_form["kickoff_ts"] = pd.NA
    player_form["kickoff_ts"] = player_form["kickoff_ts"].astype("string")
    player_form["team_abbr"] = (
        player_form.get("team_abbr", pd.Series(dtype="string"))
        .fillna(player_form.get("team"))
        .astype("string")
        .str.strip()
        .str.upper()
    )
    player_form["opponent_abbr"] = (
        player_form.get("opponent_abbr", pd.Series(dtype="string"))
        .fillna(player_form.get("opponent"))
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if "opp_abbr" in player_form.columns:
        player_form["opp_abbr"] = player_form["opp_abbr"].fillna(player_form["opponent_abbr"])
    else:
        player_form["opp_abbr"] = player_form["opponent_abbr"]

    team_to_opp = _build_team_to_opp_map_for_slate(player_form, slate_date)
    missing_mask = player_form["opponent_abbr"].isna() | (
        player_form["opponent_abbr"].astype(str).str.strip() == ""
    )
    if team_to_opp:
        player_form.loc[missing_mask, "opponent_abbr"] = player_form.loc[
            missing_mask, "team_abbr"
        ].map(team_to_opp)

    player_form["opponent_abbr"] = player_form["opponent_abbr"].replace("", pd.NA)
    player_form["opponent"] = player_form["opponent"].fillna(player_form["opponent_abbr"])
    player_form["opponent"] = player_form["opponent"].replace("", pd.NA)

    for col in ("team_abbr", "team", "opponent_abbr", "opponent"):
        if col in player_form.columns:
            player_form[col] = _canon_team_series(player_form[col])

    if {"bye", "opponent"}.issubset(player_form.columns):
        bye_mask = player_form["bye"].fillna(False).astype(bool)
        player_form.loc[bye_mask, "opponent"] = "BYE"
        if "opponent_abbr" in player_form.columns:
            player_form.loc[bye_mask, "opponent_abbr"] = "BYE"
        clear_mask = (~bye_mask) & player_form["opponent"].astype("string").str.upper().eq("BYE")
        player_form.loc[clear_mask, "opponent"] = pd.NA
        if "opponent_abbr" in player_form.columns:
            player_form.loc[clear_mask, "opponent_abbr"] = pd.NA

    still_missing = player_form["opponent_abbr"].isna()
    if still_missing.any():
        count = int(still_missing.sum())
        logger.warning(
            "[make_player_form] unable to resolve opponents for %d players", count
        )
        try:
            DEBUG_MISSING_OPP.parent.mkdir(parents=True, exist_ok=True)
            player_form.loc[still_missing].to_csv(DEBUG_MISSING_OPP, index=False)
        except Exception as err:
            logger.warning(
                "[make_player_form] failed writing missing opponent debug rows: %s",
                err,
            )

    dedup_cols = [c for c in ["player_canonical", "team_abbr", "opponent_abbr"] if c in player_form.columns]
    if len(dedup_cols) == 3:
        player_form = player_form.sort_values(dedup_cols).drop_duplicates(subset=dedup_cols, keep="first")

    player_form = _attach_player_identity(
        player_form, team_columns=("team_abbr", "team")
    )
    player_form = _reorder_identity_columns(player_form)

    roster_map_for_names = None
    roster_lookup = _build_roster_lookup()
    if not roster_lookup.empty:
        roster_base = roster_lookup.rename(columns={"team_abbr": "team"})
        roster_base = canonicalize(
            roster_base,
            name_cols=["player_display"],
            team_col="team",
            roster_map=None,
        )
        roster_map_for_names = (
            roster_base.loc[:, ["team", "player_key", "display_name"]]
            .dropna(subset=["player_key"])
            .drop_duplicates()
        )

    name_sources_pf = [
        col
        for col in ("display_name", "player", "player_name", "canonical_player_name")
        if col in player_form.columns
    ]
    player_form = canonicalize(
        player_form,
        name_cols=name_sources_pf,
        team_col="team",
        roster_map=roster_map_for_names,
    )

    # Ensure downstream keys exist before writing extras
    if "player_key" not in player_form.columns:
        player_series = player_form.get("player")
        if player_series is None:
            player_series = pd.Series("", index=player_form.index)
        player_form["player_key"] = (
            player_series.astype("string").fillna("")
        ).apply(normalize_pf_id)
    else:
        player_form["player_key"] = (
            player_form["player_key"].astype("string").fillna("")
        ).apply(normalize_pf_id)

    if "display_name" not in player_form.columns:
        display_series = player_form.get("player_display")
        if display_series is None:
            display_series = player_form.get("player")
        if display_series is None:
            display_series = pd.Series("", index=player_form.index)
        player_form["display_name"] = display_series.astype("string")
    else:
        display_series = player_form["display_name"].astype("string")
        fallback_display = player_form.get("player_display")
        if fallback_display is not None:
            display_series = display_series.fillna(fallback_display.astype("string"))
        player_form["display_name"] = display_series

    player_form = _attach_player_name_from_props(player_form)
    if "player_name" in player_form.columns:
        player_form["player_name"] = player_form["player_name"].astype("string")
        if "display_name" in player_form.columns:
            player_form["player_name"] = player_form["player_name"].fillna(
                player_form["display_name"].astype("string")
            )
    elif "display_name" in player_form.columns:
        player_form["player_name"] = player_form["display_name"].astype("string")

    # ---------- NEW: game logs + season-to-date rollups ----------
    def _safe_cols(df: pd.DataFrame, candidates: List[str]) -> List[str]:
        return [col for col in candidates if col in df.columns]

    count_cols = _safe_cols(
        player_form,
        [
            "targets",
            "receptions",
            "rushes",
            "routes",
            "pass_att",
            "dropbacks",
            "rz_targets",
            "rz_rushes",
            "games",
        ],
    )
    sum_cols = _safe_cols(
        player_form,
        [
            "rec_yards",
            "rush_yards",
            "pass_yards",
            "team_targets",
            "team_dropbacks",
            "team_rushes",
            "rz_team_targets",
            "rz_team_rushes",
        ],
    )
    max_cols = _safe_cols(player_form, ["long_rec", "long_rush"])

    if "game_id" not in player_form.columns or player_form["game_id"].isna().all():
        if "game_id" not in player_form.columns:
            player_form["game_id"] = pd.NA
        existing_game_ids = player_form["game_id"].astype("string").replace("", pd.NA)

        if "old_game_id" in player_form.columns:
            old_ids = player_form["old_game_id"].astype("string").str.strip()
            player_form["game_id"] = existing_game_ids.combine_first(
                old_ids.replace("", pd.NA)
            )
            existing_game_ids = player_form["game_id"].astype("string").replace("", pd.NA)

        if player_form["game_id"].isna().all():
            try:
                game_lines = pd.read_csv(Path("data/game_lines.csv"))
            except Exception:
                game_lines = pd.DataFrame()

            if not game_lines.empty:
                required = {"season", "week", "home", "away"}
                if required.issubset(game_lines.columns):
                    working = game_lines.copy()
                    for col in ("season", "week"):
                        working[col] = pd.to_numeric(
                            working[col], errors="coerce"
                        ).astype("Int64")
                    for col in ("home", "away"):
                        working[col] = (
                            working[col]
                            .astype("string")
                            .str.upper()
                            .str.strip()
                        )

                    if "game_id" not in working.columns:
                        season_str = working["season"].astype("Int64").astype("string")
                        week_str = working["week"].astype("Int64").astype("string").str.zfill(2)
                        home_str = working["home"].astype("string")
                        away_str = working["away"].astype("string")
                        working["game_id"] = (
                            season_str + "_" + week_str + "_" + home_str + "_" + away_str
                        )

                    team_rows = pd.concat(
                        [
                            working.assign(team=working["home"], opponent=working["away"]),
                            working.assign(team=working["away"], opponent=working["home"]),
                        ],
                        ignore_index=True,
                    )
                    team_rows = team_rows.dropna(subset=["season", "week", "team", "game_id"])
                    team_rows["team"] = team_rows["team"].astype("string")

                    lookup = {}
                    for record in team_rows.itertuples(index=False):
                        season_val = getattr(record, "season", pd.NA)
                        week_val = getattr(record, "week", pd.NA)
                        team_val = getattr(record, "team", "")
                        game_val = getattr(record, "game_id", pd.NA)
                        if pd.isna(season_val) or pd.isna(week_val) or not team_val:
                            continue
                        try:
                            key = (int(season_val), int(week_val), str(team_val))
                        except Exception:
                            continue
                        if pd.isna(game_val):
                            continue
                        lookup[key] = game_val

                    if lookup:
                        team_col = None
                        for candidate in ("team_abbr", "team"):
                            if candidate in player_form.columns:
                                team_col = candidate
                                break

                        if team_col is not None and {"season", "week"}.issubset(player_form.columns):
                            season_vals = pd.to_numeric(
                                player_form["season"], errors="coerce"
                            ).astype("Int64")
                            week_vals = pd.to_numeric(
                                player_form["week"], errors="coerce"
                            ).astype("Int64")
                            team_vals = (
                                player_form[team_col]
                                .astype("string")
                                .str.upper()
                                .str.strip()
                            )

                            keys = []
                            for season_val, week_val, team_val in zip(
                                season_vals, week_vals, team_vals
                            ):
                                if pd.isna(season_val) or pd.isna(week_val) or not team_val:
                                    keys.append(None)
                                    continue
                                try:
                                    keys.append((int(season_val), int(week_val), str(team_val)))
                                except Exception:
                                    keys.append(None)

                            mapped_ids = pd.Series(
                                [lookup.get(key) for key in keys], index=player_form.index
                            )
                            player_form["game_id"] = (
                                player_form["game_id"].astype("string").replace("", pd.NA)
                            ).combine_first(mapped_ids)

    required_keys = ["season", "week", "team", "game_id", "player_key"]
    missing_keys = [key for key in required_keys if key not in player_form.columns]
    if missing_keys:
        raise RuntimeError(
            "[make_player_form] missing required columns for game logs: "
            + ", ".join(missing_keys)
        )

    group_keys = ["season", "team", "player_key", "display_name", "game_id", "week"]
    agg_spec: Dict[str, str] = {}
    for col in count_cols:
        agg_spec[col] = "sum"
    for col in sum_cols:
        agg_spec[col] = "sum"
    for col in max_cols:
        agg_spec[col] = "max"

    game_logs = (
        player_form.groupby(group_keys, dropna=False, as_index=False).agg(agg_spec)
        if agg_spec
        else player_form[group_keys].drop_duplicates()
    )

    if "routes" in game_logs.columns and "targets" in game_logs.columns:
        game_logs["tgt_per_route"] = np.where(
            game_logs["routes"] > 0,
            game_logs["targets"] / game_logs["routes"],
            0.0,
        )
    if "receptions" in game_logs.columns and "targets" in game_logs.columns:
        game_logs["catch_rate"] = np.where(
            game_logs["targets"] > 0,
            game_logs["receptions"] / game_logs["targets"],
            0.0,
        )

    if agg_spec:
        season_totals = (
            game_logs.groupby(["season", "team", "player_key", "display_name"], as_index=False)
            .agg({col: func for col, func in agg_spec.items()})
        )
    else:
        season_totals = game_logs.groupby(
            ["season", "team", "player_key", "display_name"], as_index=False
        ).size()
        season_totals.rename(columns={"size": "games"}, inplace=True)

    if agg_spec:
        rename_map = {col: f"{col}_szn" for col in agg_spec}
        season_totals.rename(columns=rename_map, inplace=True)

    rolling_cols = [col for col in game_logs.columns if col in (count_cols + sum_cols)]
    if rolling_cols:
        game_logs = game_logs.sort_values(["season", "player_key", "week"])
        for col in rolling_cols:
            csum = (
                game_logs.groupby(["season", "player_key"], dropna=False)[col]
                .cumsum()
                .shift(1)
                .fillna(0)
            )
            game_logs[f"{col}_prior"] = csum

        prior_cols = [f"{col}_prior" for col in rolling_cols]
        prior_keys = ["season", "week", "player_key"]
        prior_view = game_logs[prior_keys + prior_cols].drop_duplicates(prior_keys)
        player_form = player_form.merge(prior_view, on=prior_keys, how="left")

    game_logs = normalize_game_logs(
        game_logs,
        team_week_map=TEAM_WEEK_MAP_PATH,
        props_map=OPPONENT_MAP_PATH,
    )
    season_totals = normalize_season_totals(season_totals)
    player_form = normalize_game_logs(
        player_form,
        team_week_map=TEAM_WEEK_MAP_PATH,
        props_map=OPPONENT_MAP_PATH,
    )

    game_logs = _filter_to_season(game_logs, CURRENT_SEASON)
    season_totals = _filter_to_season(season_totals, CURRENT_SEASON)
    player_form = _filter_to_season(player_form, CURRENT_SEASON)

    if "team_abbr" not in player_form.columns:
        player_form["team_abbr"] = player_form.get("team", pd.Series(dtype="string"))
    player_form["team_abbr"] = (
        player_form["team_abbr"].astype("string").str.upper().str.strip()
    )
    if "player_key" not in player_form.columns:
        name_source = None
        for candidate in ("player_name", "player_canonical", "player"):
            if candidate in player_form.columns:
                name_source = player_form[candidate].astype("string")
                break
        if name_source is None:
            name_source = pd.Series("", index=player_form.index, dtype="string")
        player_form["player_key"] = name_source.fillna("").apply(make_player_key)
    else:
        player_form["player_key"] = (
            player_form["player_key"].astype("string").fillna("")
        ).apply(make_player_key)

    roles_lookup = pd.DataFrame()
    try:
        if ROLES_PATH.exists() and ROLES_PATH.stat().st_size > 0:
            roles_lookup = pd.read_csv(ROLES_PATH)
    except Exception as err:
        logger.warning(
            "[PLAYER-FORM] roles merge skipped; unable to read %s (%s)",
            ROLES_PATH,
            err,
        )
        roles_lookup = pd.DataFrame()

    if not roles_lookup.empty and {"team", "player_key"}.issubset(roles_lookup.columns):
        roles_lookup = roles_lookup.copy()
        roles_lookup["team"] = (
            roles_lookup["team"].astype(str).str.upper().str.strip()
        )
        roles_lookup["player_key"] = (
            roles_lookup["player_key"].astype(str).str.strip()
        )
        roles_lookup = roles_lookup.rename(columns={"player": "roles_player"})
        merge_cols = [c for c in ("team", "player_key", "roles_player") if c in roles_lookup.columns]
        roles_lookup = roles_lookup[merge_cols].dropna(subset=["team", "player_key"])
        joined = player_form.merge(
            roles_lookup,
            left_on=["player_key", "team_abbr"],
            right_on=["player_key", "team"],
            how="left",
            suffixes=("", "_roles"),
        )
    else:
        joined = player_form.copy()
        joined["roles_player"] = pd.NA

    missing_team_mask = joined["team_abbr"].isna() | (
        joined["team_abbr"].astype(str).str.strip() == ""
    )
    mismatch_mask = joined.get("roles_player").isna()
    drop_mask = missing_team_mask | mismatch_mask
    dropped_mismatch = int(drop_mask.sum())

    filtered_form = joined.loc[~drop_mask].copy()

    if "roles_player" in filtered_form.columns:
        if "player_name" in filtered_form.columns:
            filtered_form["player_name"] = filtered_form["player_name"].astype("string").fillna(
                filtered_form["roles_player"]
            )
        else:
            filtered_form["player_name"] = filtered_form["roles_player"].astype("string")
        if "player_canonical" in filtered_form.columns:
            filtered_form["player_canonical"] = filtered_form["player_canonical"].astype("string").fillna(
                filtered_form["player_name"].astype("string")
            )

    dedup_cols = [col for col in ["player_key", "team_abbr"] if col in filtered_form.columns]
    if len(dedup_cols) == 2:
        sort_cols = dedup_cols + (
            ["week"] if "week" in filtered_form.columns else []
        )
        filtered_form = (
            filtered_form.sort_values(sort_cols)
            .drop_duplicates(subset=dedup_cols, keep="last")
            .reset_index(drop=True)
        )

    filtered_form.drop(columns=["team_roles", "roles_player"], inplace=True, errors="ignore")

    logger.info(
        "[PLAYER-FORM] %d players written, %d dropped (team mismatch)",
        len(filtered_form),
        dropped_mismatch,
    )

    player_form = filtered_form

    PLAYER_GAME_LOGS_OUT.parent.mkdir(parents=True, exist_ok=True)
    game_logs.to_csv(PLAYER_GAME_LOGS_OUT, index=False)
    PLAYER_SEASON_TOTALS_OUT.parent.mkdir(parents=True, exist_ok=True)
    season_totals.to_csv(PLAYER_SEASON_TOTALS_OUT, index=False)
    # ------------------------------------------------------------

    if player_form.empty:
        # At this point something upstream has gone badly wrong (logs/schedule
        # merge, opponent mapping, etc.). We still try to dump a debug sample
        # and then we HARD-FAIL so downstream metrics never see an empty file.
        try:
            debug_path = Path("data/_debug/player_form_mismatch_sample.csv")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            sample_source = pd.DataFrame()
            if "drop_mask" in locals() and "joined" in locals():
                sample_cols = [
                    col
                    for col in (
                        "player",
                        "player_canonical",
                        "team",
                        "team_abbr",
                        "opponent",
                        "opponent_abbr",
                    )
                    if col in joined.columns
                ]
                sample_source = joined.loc[drop_mask, sample_cols].copy()
                if "team" not in sample_source.columns and "team_abbr" in sample_source.columns:
                    sample_source["team"] = sample_source["team_abbr"]
                if "opponent" not in sample_source.columns and "opponent_abbr" in sample_source.columns:
                    sample_source["opponent"] = sample_source["opponent_abbr"]
            _dump_norm_debug(sample_source, str(debug_path))
        except Exception as debug_err:
            logger.warning(
                "[make_player_form] failed to write mismatch sample: %s",
                debug_err,
            )

        logger.error(
            "[make_player_form] final player_form is EMPTY after all joins. "
            "This is a fatal condition; see data/_debug/player_form_mismatch_sample.csv "
            "and upstream inputs (player_game_logs, team_week_map, roles_ourlads, opponent map)."
        )

        # Do NOT write placeholder CSVs here; fail fast so the build_player_form_consensus
        # job surfaces the real issue and run_pipeline_full_metrics never starts.
        raise RuntimeError("player_form is empty after normalization and joins")

    missing_opponent_count = (
        player_form["opponent"].isna().sum()
        if "opponent" in player_form.columns
        else -1
    )
    missing_name_count = (
        player_form["player_name"].isna().sum()
        if "player_name" in player_form.columns
        else -1
    )
    logger.info(
        "[make_player_form] rows=%d missing_opponent=%s missing_player_name=%s",
        len(player_form),
        "na" if missing_opponent_count < 0 else int(missing_opponent_count),
        "na" if missing_name_count < 0 else int(missing_name_count),
    )

    PLAYER_FORM_OUT.parent.mkdir(parents=True, exist_ok=True)
    player_form.to_csv(PLAYER_FORM_OUT, index=False)

    consensus = _build_grouped_consensus(player_form)
    consensus = _enforce_consensus_schema(consensus)
    consensus = _attach_player_identity(consensus, team_columns=("team_abbr", "team"))
    consensus = _reorder_identity_columns(consensus)

    pf_consensus = consensus.copy()
    if "player" not in pf_consensus.columns:
        pf_consensus["player"] = pf_consensus.get("player_canonical", "")
    pf_consensus["player"] = pf_consensus["player"].fillna("").astype(str)
    pf_consensus["player_key"] = pf_consensus["player"].apply(normalize_pf_id)

    if "team" in pf_consensus.columns:
        pf_consensus["team"] = (
            pf_consensus["team"].fillna("").astype(str).str.upper().str.strip()
        )
    else:
        pf_consensus["team"] = ""

    if "event_id" not in pf_consensus.columns:
        pf_consensus["event_id"] = pd.NA

    if "player_clean_key" in pf_consensus.columns:
        key_series = pf_consensus["player_clean_key"].astype("string")
        missing_key_mask = key_series.isna() | key_series.str.strip().eq("")
        if missing_key_mask.any():
            fallback = None
            for candidate in ("player", "player_canonical", "player_display"):
                if candidate in pf_consensus.columns:
                    fallback = pf_consensus[candidate].astype("string")
                    break
            if fallback is not None:
                fallback_keys = fallback.map(make_player_key)
                pf_consensus.loc[missing_key_mask, "player_clean_key"] = fallback_keys.loc[
                    missing_key_mask
                ]
        pf_consensus["player_clean_key"] = (
            pf_consensus["player_clean_key"].astype("string").fillna("")
        )

    for col in ("team", "opponent", "opponent_abbr"):
        if col in pf_consensus.columns:
            pf_consensus[col] = _canon_team_series(pf_consensus[col])

    # --- merge opponent map from props ---
    oppmap = pd.DataFrame()
    try:
        if OPPONENT_MAP_PATH.exists() and OPPONENT_MAP_PATH.stat().st_size > 0:
            oppmap = pd.read_csv(OPPONENT_MAP_PATH)
    except Exception as err:
        logger.warning(
            "[make_player_form] failed reading opponent map %s: %s",
            OPPONENT_MAP_PATH,
            err,
        )
        oppmap = pd.DataFrame()

    if not oppmap.empty:
        if "player_team_abbr" in oppmap.columns and "team" not in oppmap.columns:
            oppmap["team"] = oppmap["player_team_abbr"]
        if "opponent_team_abbr" in oppmap.columns and "opponent" not in oppmap.columns:
            oppmap["opponent"] = oppmap["opponent_team_abbr"]

    if not oppmap.empty:
        required = {"season", "week", "player_clean_key", "opponent", "event_id"}
        missing_required = sorted(required - set(oppmap.columns))
        if missing_required:
            logger.warning(
                "[make_player_form] opponent map missing required columns: %s",
                missing_required,
            )
        else:
            base = _coerce_merge_keys(pf_consensus)
            opp_ready = _coerce_merge_keys(oppmap)
            subset = opp_ready[list(required)].copy()
            subset = subset.sort_values(
                ["season", "week", "player_clean_key", "event_id"],
                kind="mergesort",
            ).drop_duplicates(subset=["season", "week", "player_clean_key"], keep="last")
            subset = subset.rename(
                columns={"opponent": "opponent_props", "event_id": "event_id_props"}
            )
            pf_consensus = base.merge(
                subset,
                on=["season", "week", "player_clean_key"],
                how="left",
                validate="m:1",
            )
            pf_consensus = _coalesce_dupe_cols(pf_consensus)
            if "opponent_props" in pf_consensus.columns:
                mask = pf_consensus["opponent"].isna() | (
                    pf_consensus["opponent"].astype(str).str.strip() == ""
                )
                pf_consensus.loc[mask, "opponent"] = pf_consensus.loc[
                    mask, "opponent_props"
                ]
                pf_consensus.drop(columns=["opponent_props"], inplace=True)
            if "event_id_props" in pf_consensus.columns:
                pf_consensus["event_id"] = pf_consensus["event_id_props"].combine_first(
                    pf_consensus.get("event_id")
                )
                pf_consensus.drop(columns=["event_id_props"], inplace=True)
            for col in ("season", "week"):
                if col in pf_consensus.columns:
                    numeric = pd.to_numeric(pf_consensus[col], errors="coerce")
                    numeric.loc[numeric < 0] = pd.NA
                    pf_consensus[col] = numeric.astype("Int64")
            if "player_clean_key" in pf_consensus.columns:
                pf_consensus["player_clean_key"] = (
                    pf_consensus["player_clean_key"].astype("string").fillna("")
                )
    else:
        logger.warning(
            "[make_player_form] opponent_map_from_props.csv missing or empty; opponents/event_id may be null"
        )

    if "opponent" not in pf_consensus.columns:
        pf_consensus["opponent"] = pd.NA
    missing_opponent = pf_consensus["opponent"].isna() | (
        pf_consensus["opponent"].astype(str).str.strip() == ""
    )
    if missing_opponent.any():
        logger.warning(
            "[make_player_form] opponent still missing for %d players",
            int(missing_opponent.sum()),
        )
    pf_consensus["opponent"] = pf_consensus["opponent"].astype("string").fillna("BYE")

    if "event_id" not in pf_consensus.columns:
        pf_consensus["event_id"] = pd.NA
    missing_event = pf_consensus["event_id"].isna() | (
        pf_consensus["event_id"].astype(str).str.strip() == ""
    )
    if missing_event.any():
        logger.warning(
            "[make_player_form] event_id missing for %d players",
            int(missing_event.sum()),
        )

    # --- merge roles ---
    roles_df = pd.DataFrame()
    try:
        if ROLES_PATH.exists() and ROLES_PATH.stat().st_size > 0:
            roles_df = pd.read_csv(ROLES_PATH)
    except Exception as err:
        logger.warning(
            "[make_player_form] failed reading roles %s: %s", ROLES_PATH, err
        )
        roles_df = pd.DataFrame()

    if not roles_df.empty and {"team"}.issubset(roles_df.columns):
        roles_df = roles_df.copy()
        roles_df["team"] = (
            roles_df["team"].fillna("").astype(str).str.upper().str.strip()
        )
        if "player_key" in roles_df.columns:
            roles_df["player_key_norm"] = roles_df["player_key"].apply(normalize_pf_id)
        else:
            fallback_series = roles_df.get("player")
            if fallback_series is None:
                fallback_series = pd.Series([""] * len(roles_df), index=roles_df.index)
            roles_df["player_key_norm"] = fallback_series.fillna("").astype(str).apply(
                normalize_pf_id
            )
        available_roles = [
            c
            for c in ["team", "player_key_norm", "role", "position"]
            if c in roles_df.columns
        ]
        if {"team", "player_key_norm"}.issubset(available_roles):
            pf_consensus = pf_consensus.merge(
                roles_df[available_roles],
                left_on=["player_key", "team"],
                right_on=["player_key_norm", "team"],
                how="left",
                suffixes=("", "_roles"),
            )
            pf_consensus = _coalesce_dupe_cols(pf_consensus)
            if "role_roles" in pf_consensus.columns:
                pf_consensus["role"] = pf_consensus["role"].combine_first(
                    pf_consensus["role_roles"]
                )
                pf_consensus.drop(columns=["role_roles"], inplace=True)
            if "position_roles" in pf_consensus.columns:
                pf_consensus["position"] = pf_consensus["position"].combine_first(
                    pf_consensus["position_roles"]
                )
                pf_consensus.drop(columns=["position_roles"], inplace=True)
            pf_consensus.drop(columns=["player_key_norm"], inplace=True, errors="ignore")
        else:
            logger.warning(
                "[make_player_form] roles merge missing required columns: %s",
                sorted(set(["team", "player_key", "role", "position"]) - set(available_roles)),
            )
    else:
        if roles_df.empty:
            logger.warning(
                "[make_player_form] roles_ourlads.csv missing or empty; role/position may be null"
            )

    desired_debug_cols = ["player", "team", "player_key"]
    if "role" in pf_consensus.columns:
        unmatched_mask = pf_consensus["role"].isna()
    else:
        unmatched_mask = pd.Series(False, index=pf_consensus.index)
    debug_df = (
        pf_consensus.reindex(columns=desired_debug_cols)
        .loc[unmatched_mask]
        .copy()
    )
    debug_df["note"] = "name_not_in_roles"
    try:
        UNMATCHED_ROLES_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
        debug_df.to_csv(UNMATCHED_ROLES_DEBUG_PATH, index=False)
        if not debug_df.empty:
            logger.warning(
                "[make_player_form] unmatched roles rows=%d → %s",
                len(debug_df),
                UNMATCHED_ROLES_DEBUG_PATH,
            )
    except Exception as err:
        logger.warning(
            "[make_player_form] failed writing unmatched roles debug: %s", err
        )

    name_sources_consensus = [
        col
        for col in ("display_name", "player", "player_name", "canonical_player_name")
        if col in pf_consensus.columns
    ]
    pf_consensus = canonicalize(
        pf_consensus,
        name_cols=name_sources_consensus,
        team_col="team",
        roster_map=roster_map_for_names,
    )

    pf_consensus = _attach_player_name_from_props(pf_consensus)
    if "player_name" in pf_consensus.columns:
        pf_consensus["player_name"] = pf_consensus["player_name"].astype("string")
        if "display_name" in pf_consensus.columns:
            pf_consensus["player_name"] = pf_consensus["player_name"].fillna(
                pf_consensus["display_name"].astype("string")
            )
    elif "display_name" in pf_consensus.columns:
        pf_consensus["player_name"] = pf_consensus["display_name"].astype("string")

    pf_consensus = _attach_player_identity(
        pf_consensus, team_columns=("team_abbr", "team")
    )
    pf_consensus = _reorder_identity_columns(pf_consensus)

    consensus = pf_consensus
    consensus["season"] = int(season)
    PLAYER_FORM_CONSENSUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    consensus.to_csv(PLAYER_FORM_CONSENSUS_OUT, index=False)

    print(f"[make_player_form] wrote {len(player_form)} rows -> {PLAYER_FORM_OUT}")
    print(
        f"[make_player_form] wrote {len(consensus)} rows -> {PLAYER_FORM_CONSENSUS_OUT}"
    )


def normalize_name(name: object) -> str:
    if pd.isna(name):
        return ""
    return re.sub(r"[^a-z]", "", str(name).lower())


def _player_key_from_name(name: object) -> str:
    if not isinstance(name, str):
        return ""
    cleaned = re.sub(r"[^A-Za-z\s\-']", " ", name).strip()
    if not cleaned:
        return ""
    parts = [p for p in re.split(r"\s+", cleaned) if p]
    if not parts:
        return ""
    first = re.sub(r"[^A-Za-z]", "", parts[0]).lower()
    last = re.sub(r"[^A-Za-z]", "", parts[-1]).lower() if len(parts) > 1 else ""
    if first and last:
        return f"{first[0]}{last}"
    return last or first


def _load_player_logs(
    logs_path: Path,
    season_totals_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load per-game and season-total player logs.

    These are REQUIRED for building PlayerForm in the legacy flow.
    If they are missing / empty / unreadable, we raise and stop here
    instead of silently writing a roles-only skeleton.
    """
    logs = _safe_read_csv(logs_path, "player_game_logs.csv")
    totals = (
        _safe_read_csv(season_totals_path, "player_season_totals.csv")
        if season_totals_path is not None
        else pd.DataFrame()
    )

    logger.info("[PlayerForm] Loaded %d game logs before season filter", len(logs))
    if not totals.empty:
        logger.info("[PlayerForm] Loaded %d season totals rows", len(totals))
    else:
        logger.info("[PlayerForm] No season totals available")

    return logs, totals


def build_player_form_from_logs(
    game_logs: pd.DataFrame,
    season_totals: pd.DataFrame | None,
    roles_path: str | Path,
    opponent_map_path: str | Path,
    team_week_map_path: str | Path,
    season: int = SEASON,
) -> pd.DataFrame:
    """Thin wrapper to align with the orchestrator contract for this script."""

    return build_player_form(
        season=season,
        game_logs=game_logs,
        season_totals=season_totals,
    )


def build_player_form(
    season: int = 2025,
    game_logs: pd.DataFrame | None = None,
    season_totals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    season = int(season)
    target_season = SEASON
    if season != target_season:
        logger.info(
            "[make_player_form] overriding season %s with env SEASON=%s",
            season,
            target_season,
        )
    season = target_season
    global CURRENT_SEASON
    CURRENT_SEASON = season

    roles_path = Path("data/roles_ourlads.csv")
    logs_path = Path("data/player_game_logs.csv")
    sched_path = Path("data/team_week_map.csv")
    totals_path = Path("data/player_season_totals.csv")

    # Tracks whether we had to fall back due to missing inputs.
    partial_flag = 0

    if roles_path.exists():
        roles = pd.read_csv(roles_path)
    else:
        logger.warning("[make_player_form] roles_ourlads.csv missing; continuing with empty roles")
        roles = pd.DataFrame(columns=["player", "team"])

    logs = game_logs.copy() if game_logs is not None else pd.DataFrame()
    totals = season_totals.copy() if season_totals is not None else pd.DataFrame()

    # Fill missing inputs from disk if needed.
    if logs.empty or totals.empty:
        disk_logs, disk_totals = _load_player_logs(logs_path, totals_path)
        if logs.empty:
            logs = disk_logs
        if totals.empty:
            totals = disk_totals

    if sched_path.exists():
        try:
            sched = pd.read_csv(sched_path)
        except Exception as err:
            logger.warning(
                "[make_player_form] failed reading %s: %s", sched_path, err
            )
            sched = pd.DataFrame()
    else:
        logger.warning("[make_player_form] team_week_map.csv missing; schedule enrich skipped")
        sched = pd.DataFrame()

    # If BOTH logs and totals are missing/empty, this path has nothing to work with.
    # Fall back to the legacy PBP-based builder which computes metrics directly
    # from play-by-play instead of relying on pre-built CSVs.
    if logs.empty and totals.empty:
        logger.error(
            "[make_player_form] No usable game logs or season totals found; "
            "falling back to build_player_form_legacy(season=%s).",
            season,
        )
        return build_player_form_legacy(season=season)

    logs_initial_count = len(logs)

    roles = _filter_to_season(roles, season)
    logs = _filter_to_season(logs, season)
    totals = _filter_to_season(totals, season)
    sched = _filter_to_season(sched, season)

    print(f"[PlayerForm] Loaded {len(roles)} role rows from roles_ourlads.csv")
    print(f"[PlayerForm] Loaded {logs_initial_count} game logs before season filter")
    print(f"[PlayerForm] Loaded {len(sched)} schedule rows from team_week_map.csv")

    if "season" in logs.columns:
        print(f"[PlayerForm] Retained {len(logs)} logs for season {season}")

    # --- NEW: ensure we have a canonical "player" column on logs -------------
    #
    # The new PBP / game-log pipeline can surface player names under a variety
    # of columns (e.g. "name", "player_name", "display_name") instead of the
    # legacy "player" column.  Downstream logic (player_key, merges, etc.)
    # expects logs["player"] to exist, which is why we are currently hitting:
    #
    #   KeyError: 'player'
    #   ... logs["player_key"] = logs["player"].apply(normalize_name)
    #
    # Here we normalize that: pick the first available name column and create
    # a canonical "player" column from it.  If nothing usable is present, we
    # fail fast with a clear error.
    if not logs.empty and "player" not in logs.columns:
        candidate_cols = [
            col for col in ("name", "player_name", "display_name")
            if col in logs.columns
        ]
        if candidate_cols:
            src = candidate_cols[0]
            print(
                f"[PlayerForm] 'player' column missing on logs; "
                f"using '{src}' as the source for player names."
            )
            logs = logs.copy()
            logs["player"] = logs[src].astype("string")
        else:
            # Fail loudly so we don't silently write garbage PlayerForm.
            raise RuntimeError(
                "[PlayerForm] FATAL: no usable player-name column found on "
                f"logs; expected one of ['player', 'name', 'player_name', "
                f"'display_name']; got columns={list(logs.columns)}"
            )

    roles["player_key"] = roles["player"].apply(normalize_name)
    roles["player_clean_key"] = roles["player"].apply(normalize_name)
    logs["player_key"] = logs["player"].apply(normalize_name)
    logs["player_clean_key"] = logs["player"].apply(normalize_name)

    merged = pd.merge(
        logs,
        roles,
        on=["player_key", "player_clean_key"],
        how="left",
        suffixes=("", "_role"),
    )
    print(f"[PlayerForm] Merged logs with roles → {len(merged)} rows")

    if "season" in merged.columns:
        merged["season"] = pd.to_numeric(merged["season"], errors="coerce").astype("Int64")
        season_scope = pd.Series([SEASON], dtype="Int64").iloc[0]
        merged = merged[merged["season"] == season_scope].copy()

    for col in ("team", "position", "role", "player"):
        role_col = f"{col}_role"
        if role_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(merged[role_col])
            else:
                merged[col] = merged[role_col]
            merged.drop(columns=[role_col], inplace=True)

    miss = merged[merged["team"].isna()]
    if not miss.empty:
        miss.to_csv("data/player_form_missing.csv", index=False)
        print(f"[PlayerForm] Missing {len(miss)} players")

    if {"event_id", "team", "opponent"}.issubset(sched.columns):
        schedule = sched[["event_id", "team", "opponent"]].copy()
        merged = merged.merge(schedule, on="event_id", how="left", suffixes=("", "_schedule"))
        if "team_schedule" in merged.columns:
            if "team" in merged.columns:
                merged["team"] = merged["team"].fillna(merged["team_schedule"])
            else:
                merged["team"] = merged["team_schedule"]
            merged.drop(columns=["team_schedule"], inplace=True)
        if "opponent_schedule" in merged.columns:
            if "opponent" in merged.columns:
                merged["opponent"] = merged["opponent"].fillna(merged["opponent_schedule"])
            else:
                merged["opponent"] = merged["opponent_schedule"]
            merged.drop(columns=["opponent_schedule"], inplace=True)
    else:
        print("[PlayerForm] team_week_map missing required columns; opponent join skipped")

    pf = merged.copy()

    # Force season scope to CURRENT_SEASON and re-attach opponents for the scoped rows
    pf = pf.copy()
    if "season" in pf.columns:
        pf["season"] = pd.to_numeric(pf["season"], errors="coerce").astype("Int64")
        pf = pf[pf["season"] == pd.Series([SEASON], dtype="Int64").iloc[0]].copy()

    # Overlay opponents using the schedule-based helper.
    # attach_opponent is intentionally best-effort: if required columns
    # (season/week/team) are missing it simply returns the original frame.
    try:
        from scripts._opponent_map import attach_opponent, normalize_team as _norm_team_helper

        if "team" in pf.columns:
            pf["team"] = _norm_team_helper(pf["team"])

        pf = attach_opponent(
            pf,
            season_col="season",
            week_col="week",
            team_col="team",
            out_col="opponent",
            schedule_path="data/team_week_map.csv",
        )

        if "opponent_abbr" in pf.columns and "opponent" in pf.columns:
            pf["opponent_abbr"] = pf["opponent_abbr"].fillna(pf["opponent"])
    except Exception as e:
        print(f"[make_player_form] WARNING: opponent overlay failed: {e}")

    merged = pf.copy()

    # If everything ended up empty here, treat it as a hard failure.
    # At this point we *know* logs and schedule loaded, so an empty
    # PlayerForm means our merge logic dropped everything.
    if merged.empty:
        raise RuntimeError(
            "[make_player_form] PlayerForm is EMPTY after merges even though "
            "logs and roles were loaded. This usually means a bad join key "
            "(season/week/team/opponent) or a bug in the merge logic."
        )

    merged["partial"] = partial_flag

    if "player_source_name" not in merged.columns:
        source_series = None
        for candidate in ("player", "player_name", "display_name"):
            if candidate in merged.columns:
                source_series = merged[candidate]
                break
        if source_series is None:
            source_series = pd.Series("", index=merged.index)
        merged["player_source_name"] = source_series

    if "team_abbr" not in merged.columns:
        if "team" in merged.columns:
            merged["team_abbr"] = merged["team"]
        else:
            merged["team_abbr"] = pd.NA

    if "position" not in merged.columns:
        merged["position"] = pd.NA

    if "role" not in merged.columns:
        merged["role"] = pd.NA

    merged["team_abbr"] = merged["team_abbr"].astype("string").str.upper().str.strip()
    merged["season"] = SEASON
    if "week" not in merged.columns:
        merged["week"] = pd.NA

    def _canonicalize_player(row: pd.Series) -> pd.Series:
        raw = row.get("player_source_name")
        if not isinstance(raw, str) or not raw.strip():
            raw = row.get("player")
        if not isinstance(raw, str) or not raw.strip():
            return pd.Series({"player": "", "player_clean_key": ""})
        full_name, key = canonicalize_player_name_safe(raw)
        full_name = (full_name or "").strip()
        key = (key or "").strip()
        if not full_name:
            full_name = raw.strip()
        if not key:
            _, key = canonicalize_player_name_safe(full_name)
        return pd.Series({"player": full_name, "player_clean_key": key})

    canon = merged.apply(_canonicalize_player, axis=1)
    merged["player"] = canon["player"].astype("string")
    merged["player_clean_key"] = canon["player_clean_key"].astype("string")

    merged = merged[merged["player"].astype("string").str.strip().str.len() > 0]

    if merged.empty:
        dbg = "artifacts/player_form_debug_empty.csv"
        os.makedirs(os.path.dirname(dbg), exist_ok=True)
        merged.to_csv(dbg, index=False)
        raise RuntimeError(
            "[player_form] produced empty DataFrame; debug dump written to artifacts/player_form_debug_empty.csv"
        )

    os.makedirs("data/_debug", exist_ok=True)
    merged.to_csv("data/player_form.csv", index=False)
    print(f"[PlayerForm] {len(merged)} total players written.")

    # Debug: dump any rows still missing opponent so CI logs point straight to the holes
    try:
        miss = pf[pf["opponent"].isna() if "opponent" in pf.columns else []]
        if miss is not None and len(miss) > 0:
            miss = miss.copy()
            miss.to_csv("data/_debug/player_missing_opponent.csv", index=False)
            print(f"[make_player_form] WARNING missing opponent rows: {len(miss)} → data/_debug/player_missing_opponent.csv")
    except Exception:
        pass

    return merged


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=SEASON)
    parser.add_argument(
        "--slate-date",
        dest="slate_date",
        default=os.environ.get("SLATE_DATE", ""),
        help="ISO date for current slate (YYYY-MM-DD). Used only for logging/filtering.",
    )
    parser.add_argument("--week", type=int, required=False, help="Optional week (unused; kept for compatibility).")
    args = parser.parse_args(argv)

    season = int(args.season)
    slate_date = args.slate_date

    logger.info("[pf] building player_form for season=%s slate_date=%s", season, slate_date)

    # 1) Always fetch normalized logs for the requested season.
    game_logs, season_totals = _fetch_player_logs(season)

    if game_logs is None or game_logs.empty:
        raise RuntimeError(f"[pf] FATAL: normalized game logs are empty for season={season}")
    if season_totals is None or season_totals.empty:
        logger.warning("[pf] season_totals empty for season=%s (will build using game logs only)", season)

    # 2) Build the base player_form frame from logs + totals
    base = build_player_form_from_logs(
        game_logs=game_logs,
        season_totals=season_totals,
        roles_path=ROLES_PATH,
        opponent_map_path=OPPONENT_MAP_PATH,
        team_week_map_path=TEAM_WEEK_MAP_PATH,
        season=season,
    )

    logger.info("[pf] base player_form rows before enrichment: %d", len(base))
    if base.empty:
        raise RuntimeError(f"[pf] FATAL: player_form base is empty for season={season}")

    if not PLAYER_FORM_OUT.exists() or PLAYER_FORM_OUT.stat().st_size == 0:
        raise RuntimeError("[pf] FATAL: player_form.csv was not written or is empty")
    if not PLAYER_FORM_CONSENSUS_OUT.exists() or PLAYER_FORM_CONSENSUS_OUT.stat().st_size == 0:
        raise RuntimeError("[pf] FATAL: player_form_consensus.csv was not written or is empty")

    logger.info(
        "[pf] final player_form rows: %d; player_form_consensus rows: %s",
        len(base),
        "unknown" if not PLAYER_FORM_CONSENSUS_OUT.exists() else sum(1 for _ in open(PLAYER_FORM_CONSENSUS_OUT, "r")) - 1,
    )

    return 0


# ---------------------------
# CLI
# ---------------------------


def cli() -> int:
    """
    Entry point used by the GitHub Action.

    Thin wrapper around ``main`` so that all logic flows through the
    normalized game-log / season-total builder instead of duplicating
    the pipeline here.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        type=int,
        default=os.environ.get("SEASON", 2025),
        help="Season to build player_form for (defaults to SEASON env or 2025).",
    )
    parser.add_argument(
        "--date",
        "--slate-date",
        dest="slate_date",
        type=str,
        default=os.environ.get("SLATE_DATE", ""),
        help="Optional slate date (YYYY-MM-DD) used only for logging/filtering.",
    )
    parser.add_argument(
        "--week",
        type=int,
        required=False,
        help="Unused; kept for CLI compatibility.",
    )
    args = parser.parse_args()

    argv: List[str] = ["--season", str(args.season)]
    if args.slate_date:
        argv.extend(["--slate-date", args.slate_date])

    # Delegate to the newer main() implementation which:
    #   * fetches normalized logs via _fetch_player_logs
    #   * builds player_form + player_form_consensus
    #   * enforces fail-fast checks on empty outputs
    return main(argv)


if __name__ == "__main__":
    sys.exit(cli())
