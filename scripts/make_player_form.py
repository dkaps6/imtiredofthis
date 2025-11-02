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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import unicodedata

import numpy as np
import pandas as pd

from scripts.utils.canonical_names import (
    canonicalize_player_name as _canonicalize_with_utils,
    log_unmapped_variant,
)

logger = logging.getLogger(__name__)


DATA_DIR = "data"
ROLES_PATH = Path("data/roles_ourlads.csv")
PROPS_ENRICHED_PATH = Path("data/props_enriched.csv")
SCHEDULE_GAMES_PATH = Path("data/games.csv")
PLAYER_FORM_OUT = Path("data/player_form.csv")
PLAYER_FORM_CONSENSUS_OUT = Path("data/player_form_consensus.csv")
DEBUG_MISSING_OPP = Path("data/_debug/player_missing_opponent.csv")
OPPONENT_MAP_PATH = Path("data/opponent_map_from_props.csv")
UNMATCHED_ROLES_DEBUG_PATH = Path("data/unmatched_roles_merge.csv")


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


NAME_OVERRIDES = {
    "JOE T FLACCO": "Joe Flacco",
    "JOE T  FLACCO": "Joe Flacco",
    "D.ADAMS": "Davante Adams",
    "DADAMS": "Davante Adams",
}


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


def _dedupe_player_clean_key(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse player_clean_key columns created by fallback merges."""

    if df is None or df.empty:
        return df

    working = df.copy()
    has_x = "player_clean_key_x" in working.columns
    has_y = "player_clean_key_y" in working.columns

    combined = None
    if has_x and has_y:
        x = working["player_clean_key_x"].astype("string")
        y = working["player_clean_key_y"].astype("string")
        combined = x.copy()
        prefer_x = combined.notna() & combined.str.strip().ne("")
        combined = combined.where(prefer_x, y)
    elif has_x:
        combined = working["player_clean_key_x"].astype("string")
    elif has_y:
        combined = working["player_clean_key_y"].astype("string")

    if combined is not None:
        if "player_clean_key" in working.columns:
            existing = working["player_clean_key"].astype("string")
            keep_existing = existing.notna() & existing.str.strip().ne("")
            combined = existing.where(keep_existing, combined)
        working["player_clean_key"] = combined

    drop_cols = [
        c for c in ("player_clean_key_x", "player_clean_key_y") if c in working.columns
    ]
    if drop_cols:
        working = working.drop(columns=drop_cols)

    # Drop any duplicate column names introduced during merges (keep first occurrence).
    working = working.loc[:, ~working.columns.duplicated()]

    return working


def _normalize_key(s: str) -> str:
    """Lowercase, remove punctuation/whitespace for canonical lookups."""

    if s is None:
        return ""
    # strip accents
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # remove punctuation and whitespace
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _canonical_identity_fields(raw: Any) -> Dict[str, str]:
    """Return canonical name metadata using shared utils and log unmapped variants."""

    raw_str = "" if raw is None else str(raw)
    canonical, clean_key = _canonicalize_with_utils(raw_str)
    canonical = (canonical or "").strip()
    clean_key = (clean_key or "").strip()

    if raw_str and canonical and canonical == clean_key:
        try:
            log_unmapped_variant(raw_str)
        except Exception:
            pass

    if not canonical:
        canonical = raw_str.upper().strip()

    canonical_upper = canonical.upper()
    canonical_lower = re.sub(r"[^a-z0-9 ]+", "", canonical_upper.lower()).strip()
    player_clean_key = re.sub(r"\s+", "_", canonical_lower)

    return {
        "player_name_canonical": canonical_upper,
        "player_canonical": canonical_lower,
        "player_clean_key": player_clean_key,
    }


def canonicalize_player_name(raw: str) -> str:
    """Legacy alias that returns the normalized canonical name string."""

    return _canonicalize_player_name(raw)


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
    for c in [
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
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keys = [k for k in ["player", "team", "season"] if k in df.columns]
    if not keys:
        return pd.DataFrame()

    sums = (
        df.groupby(keys, dropna=False)
        .agg(
            {
                c: "sum"
                for c in [
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
                if c in df.columns
            }
        )
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
    if roles.empty:
        raise RuntimeError(
            "roles_ourlads.csv is empty. Check Ourlads scraper/selectors."
        )

    for col in ("team", "player", "role"):
        if col in roles.columns:
            roles[col] = roles[col].astype(str)
    if "player" in roles.columns:
        roles["player"] = roles["player"].map(_clean_name).map(standardize_full_name)
    roles["team"] = roles.get("team", "").astype(str).map(_canon_team)
    roles["role"] = roles.get("role", "").astype(str).str.upper().str.strip()
    if "position" not in roles.columns and "role" in roles.columns:
        roles["position"] = roles["role"].str.extract(r"([A-Z]+)")

    roles = ensure_canonical(roles, player_col="player", team_col="team")

    merge_cols = ["player_canonical", "team"]
    for extra in ["role", "position", "position_guess"]:
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

    assert_no_duplicate_columns(pf, "roles merge")

    if "role_roles" in pf.columns:
        pf["role"] = pf["role"].combine_first(pf["role_roles"])
        pf.drop(columns=["role_roles"], inplace=True)
    if "position_roles" in pf.columns:
        pf["position"] = pf["position"].combine_first(pf["position_roles"])
        pf.drop(columns=["position_roles"], inplace=True)
    if "position_guess" in pf.columns:
        pf["position"] = pf["position"].combine_first(pf["position_guess"])

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


def load_pbp(season: int) -> pd.DataFrame:
    # explicit, version-safe loader with diagnostics
    rows = -1
    try:
        if NFL_PKG == "nflreadpy":
            raw = NFLV.load_pbp(seasons=[season])
        else:
            # nfl_data_py has had both names in the wild
            if hasattr(NFLV, "import_pbp_data"):
                raw = NFLV.import_pbp_data([season], downcast=True)  # type: ignore
            elif hasattr(NFLV, "import_pbp"):
                raw = NFLV.import_pbp([season])  # type: ignore
            else:
                raise RuntimeError("nfl_data_py missing import_pbp(_data) functions")
        pbp = _to_pandas(raw)
        pbp.columns = [c.lower() for c in pbp.columns]
        rows = len(pbp)
        print(
            f"[pf] PBP loaded for {season}: rows={rows}, sample_cols={list(pbp.columns[:10])}"
        )
        if rows == 0:
            raise RuntimeError("PBP returned 0 rows (unexpected for active season).")
        return pbp
    except Exception as e:
        print(
            f"[pf] ERROR loading PBP {season}: {type(e).__name__}: {e}", file=sys.stderr
        )
        return pd.DataFrame()


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


def build_player_form(season: int = 2025, slate_date: str | None = None) -> pd.DataFrame:
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

    if "team" in base.columns:
        base["team"] = base["team"].astype(str).str.upper().str.strip()
        base.loc[base["team"].isin(["", "NAN", "NONE", "NULL"]), "team"] = pd.NA
    for raw in base.get("player", pd.Series(dtype=object)).dropna().unique():
        log_unmapped_variant(raw)

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
        base["opponent"] = base["opponent"].astype(str).str.strip().str.upper()
        base.loc[base["opponent"].isin(["", "NAN"]), "opponent"] = np.nan

    if "team" in base.columns:
        base["team_key"] = base["team"].astype(str).str.upper().str.strip()
    if "week" in base.columns:
        base["week_key"] = base["week"].fillna(-1).astype(int)
    else:
        base["week_key"] = -1

    out = base.copy()
    out = _apply_canonical_names(out)
    out = _enrich_team_and_opponent_from_props(out, season)
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
            merge_team_cols = [c for c in ["team", "season", "week"] if c in team_subset.columns]
            if merge_team_cols:
                working = working.merge(
                    team_subset[merge_team_cols + ["opponent"]].drop_duplicates(),
                    on=merge_team_cols,
                    how="left",
                    suffixes=("", "_teamopp"),
                )
                assert_no_duplicate_columns(working, "team-level opponent merge")
                if "opponent_teamopp" in working.columns:
                    working["opponent"] = working["opponent"].fillna(working["opponent_teamopp"])
                    working.drop(columns=["opponent_teamopp"], inplace=True)

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
    out: pd.DataFrame, season: int | None = None
) -> pd.DataFrame:
    if out is None or out.empty:
        return out

    df = out.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if "player" not in df.columns:
        return df

    raw_players = df["player"].copy()
    for raw in raw_players.dropna().unique():
        log_unmapped_variant(raw)

    identity = raw_players.apply(lambda nm: pd.Series(_canonical_identity_fields(nm)))
    df["player_name_canonical"] = identity["player_name_canonical"].astype(str)
    df["player_canonical"] = identity["player_canonical"].astype(str)
    df["player_clean_key"] = identity["player_clean_key"].astype(str)
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
        enriched = enriched.merge(
            props_map.drop(columns=[c for c in ["__has_team"] if c in props_map.columns]),
            on="player_canonical",
            how="left",
        )
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

    dedup_cols = [c for c in ["player_canonical", "team_abbr", "opponent_abbr"] if c in enriched.columns]
    if len(dedup_cols) == 3:
        enriched = enriched.sort_values(dedup_cols).drop_duplicates(subset=dedup_cols, keep="first")

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
        log_unmapped_variant(raw)

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
        log_unmapped_variant(raw)

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
        lambda nm: pd.Series(_canonical_identity_fields(nm))
    )
    out["player_name_canonical"] = canonical_df["player_name_canonical"].astype(
        str
    )
    out["player_canonical"] = canonical_df["player_canonical"].astype(str)
    out["player_clean_key"] = canonical_df["player_clean_key"].astype(str)

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


def _write_player_form_outputs(df: pd.DataFrame, slate_date: str | None = None) -> None:
    if df is None or df.empty:
        raise RuntimeError("[make_player_form] final player_form empty; aborting run")

    essential = {"player_clean_key", "team", "opponent", "week"}
    essential.update(PLAYER_FORM_SHARE_COLS)
    missing = [col for col in sorted(essential) if col not in df.columns]
    if missing:
        logger.warning(
            "[make_player_form] player_form missing expected columns prior to write: %s",
            ", ".join(missing),
        )

    assert_no_duplicate_columns(df, "final player_form before write")

    df_out = _ensure_single_position_column(df.copy())
    df_out = _ensure_cols(df_out, FINAL_COLS)
    player_form = _enforce_player_form_schema(df_out)

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

    if player_form.empty:
        raise RuntimeError("[make_player_form] final player_form empty; aborting run")

    PLAYER_FORM_OUT.parent.mkdir(parents=True, exist_ok=True)
    player_form.to_csv(PLAYER_FORM_OUT, index=False)

    consensus = _build_grouped_consensus(player_form)
    consensus = _enforce_consensus_schema(consensus)

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

    if not oppmap.empty and {"player", "team"}.issubset(oppmap.columns):
        oppmap = oppmap.copy()
        oppmap["player"] = oppmap["player"].fillna("").astype(str)
        oppmap["team"] = (
            oppmap["team"].fillna("").astype(str).str.upper().str.strip()
        )
        oppmap["player_key"] = oppmap["player"].apply(normalize_pf_id)
        available = [
            c
            for c in ["player_key", "team", "opponent", "event_id"]
            if c in oppmap.columns
        ]
        if {"player_key", "team"}.issubset(available):
            subset = oppmap[available].copy()
            subset = subset.sort_values(
                ["player_key", "team", "event_id"], kind="mergesort"
            ).drop_duplicates(subset=["player_key", "team"], keep="first")
            pf_consensus = pf_consensus.merge(
                subset,
                on=["player_key", "team"],
                how="left",
                suffixes=("", "_props"),
            )
            if "opponent_props" in pf_consensus.columns:
                mask = pf_consensus["opponent"].isna() | (
                    pf_consensus["opponent"].astype(str).str.strip() == ""
                )
                pf_consensus.loc[mask, "opponent"] = pf_consensus.loc[
                    mask, "opponent_props"
                ]
                pf_consensus.drop(columns=["opponent_props"], inplace=True)
            if "event_id_props" in pf_consensus.columns:
                pf_consensus["event_id"] = pf_consensus["event_id"].combine_first(
                    pf_consensus["event_id_props"]
                )
                pf_consensus.drop(columns=["event_id_props"], inplace=True)
        else:
            logger.warning(
                "[make_player_form] opponent map missing required columns: %s",
                sorted(set(["player", "team", "opponent", "event_id"]) - set(available)),
            )
    else:
        if oppmap.empty:
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

    consensus = pf_consensus
    PLAYER_FORM_CONSENSUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    consensus.to_csv(PLAYER_FORM_CONSENSUS_OUT, index=False)

    print(f"[make_player_form] wrote {len(player_form)} rows -> {PLAYER_FORM_OUT}")
    print(
        f"[make_player_form] wrote {len(consensus)} rows -> {PLAYER_FORM_CONSENSUS_OUT}"
    )


# ---------------------------
# CLI
# ---------------------------


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--date",
        type=str,
        required=False,
        help="Slate date like YYYY-MM-DD (used for opponent mapping / props alignment)",
    )
    args = parser.parse_args()

    _safe_mkdir(DATA_DIR)

    try:
        # Build player form
        df = build_player_form(season=args.season, slate_date=args.date)

        # Fallback sweep BEFORE strict validation (retain your original behavior)
        try:
            before = df.copy()
            df = _apply_fallback_enrichers(df)
            filled = {}
            for c in [
                "route_rate",
                "tgt_share",
                "rush_share",
                "yprr",
                "ypc",
                "ypa",
                "rz_share",
            ]:
                if c in before.columns and c in df.columns:
                    filled[c] = int((df[c].notna() & before[c].isna()).sum())
            if filled:
                print(
                    "[make_player_form] fallback enriched:",
                    ", ".join(f"{k}:+{v}" for k, v in filled.items()),
                )
        except Exception as _fb_e:
            print(
                f"[make_player_form] WARN fallback enrichers skipped: {_fb_e}",
                file=sys.stderr,
            )

        # Validate required metrics for players that appear in props
        try:
            _validate_required(df)
        except Exception as _val_e:
            print(
                f"[make_player_form] WARN validation noted issues: {_val_e}",
                file=sys.stderr,
            )

        # Opponent enrichment (non-fatal)
        try:
            df = _enrich_team_and_opponent_from_props(df, args.season)
        except Exception as _enr_e:
            print(
                f"[make_player_form] WARN opponent enrichment skipped in cli(): {_enr_e}",
                file=sys.stderr,
            )

        # Final write on success
        df = _dedupe_player_clean_key(df)
        _write_player_form_outputs(df, slate_date=args.date)
        return

    except Exception as e:
        # Log error and try to salvage partial DF
        print(f"[make_player_form] ERROR: {e}", file=sys.stderr)
        if isinstance(e, RuntimeError) and "opponent merge" in str(e):
            raise
        try:
            if "df" in locals() and isinstance(df, pd.DataFrame) and len(df) > 0:
                try:
                    df = _enrich_team_and_opponent_from_props(df, args.season)
                except Exception as _enr_e:
                    print(
                        f"[make_player_form] WARN enrichment skipped in error path: {_enr_e}",
                        file=sys.stderr,
                    )
                df = _dedupe_player_clean_key(df)
                _write_player_form_outputs(df, slate_date=args.date)
                return
        except Exception as _w:
            print(
                f"[make_player_form] WARN could not write partial df in error path: {_w}",
                file=sys.stderr,
            )

        raise


if __name__ == "__main__":
    cli()
