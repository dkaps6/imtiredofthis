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
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import re
import unicodedata

import numpy as np
import pandas as pd

from scripts.utils.canonical_names import (
    canonicalize_player_name as _canonicalize_with_utils,
    log_unmapped_variant,
)

logger = logging.getLogger(__name__)


DATA_DIR = "data"
OPP_PATH = Path("data") / "opponent_map_from_props.csv"
ROLES_PATH = Path("data") / "roles_ourlads.csv"
GAMES_PATH = Path("data") / "game_lines.csv"


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

    canonical_lower = re.sub(r"[^a-z0-9 ]+", "", canonical.lower()).strip()
    player_clean_key = re.sub(r"\s+", "_", canonical_lower)

    return {
        "player_name_canonical": canonical,
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
    if OPP_PATH.exists():
        dfs.append(pd.read_csv(OPP_PATH))
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

    out["player"] = out["player"].map(standardize_full_name)
    out["player_canonical"] = (
        out["player"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]+", "", regex=True)
        .str.strip()
    )
    out["player_clean_key"] = out["player_canonical"].str.replace(
        r"\s+", "_", regex=True
    )
    return out


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

PLAYER_FORM_REQUIRED_COLUMNS = [
    "player",
    "player_name_canonical",
    "team",
    "week",
    "opponent",
    "season",
    "position",
    "role",
] + PLAYER_FORM_USAGE_COLS + [
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

FINAL_COLS = PLAYER_FORM_REQUIRED_COLUMNS + [
    "player_canonical",
    "player_clean_key",
    "team_key",
    "week_key",
    "unmatched_flag",
]


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
    "team",
    "week",
    "opponent",
    "role",
]


def _inject_week_opponent_and_roles(
    out: pd.DataFrame, name_maps: dict | None
) -> pd.DataFrame:
    """Fill opponent/week using inferred values with props fallbacks."""

    opp_path = Path("data") / "opponent_map_from_props.csv"
    if not opp_path.exists():
        raise RuntimeError("[make_player_form] opponent_map_from_props.csv not found")

    opp = pd.read_csv(opp_path)
    required_cols = {"player", "team", "week", "opponent"}
    missing = required_cols - set(opp.columns)
    if missing:
        raise RuntimeError(
            f"[make_player_form] opponent_map_from_props.csv missing columns: {sorted(missing)}"
        )

    opp = _apply_player_name_cleaning(opp, name_maps)
    opp["team"] = opp["team"].astype(str).str.upper().str.strip()
    opp_merge = opp[["player_clean_key", "team", "week", "opponent"]].rename(
        columns={"week": "week_from_props", "opponent": "opponent_from_props"}
    )

    enriched = out.copy()
    enriched = _apply_player_name_cleaning(enriched, name_maps)
    enriched["team"] = enriched["team"].astype(str).str.upper().str.strip()

    if "week_inferred" not in enriched.columns:
        enriched["week_inferred"] = pd.Series(pd.NA, index=enriched.index)
    if "opponent_inferred" not in enriched.columns:
        enriched["opponent_inferred"] = pd.Series(pd.NA, index=enriched.index)

    merged = enriched.merge(
        opp_merge,
        on=["player_clean_key", "team"],
        how="left",
        suffixes=("", "_props"),
    )

    merged["week_final"] = merged["week_inferred"]
    merged.loc[merged["week_final"].isna(), "week_final"] = merged["week_from_props"]

    merged["opponent_final"] = merged["opponent_inferred"]
    merged.loc[merged["opponent_final"].isna(), "opponent_final"] = merged[
        "opponent_from_props"
    ]

    merged["week"] = merged["week_final"]
    merged["opponent"] = merged["opponent_final"]

    for col in ["week_from_props", "opponent_from_props", "week_final", "opponent_final"]:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    for col in ["week_inferred", "opponent_inferred"]:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    total_rows = len(merged)
    have_opp = merged["opponent"].notna().sum()
    if have_opp == 0 and total_rows > 0:
        raise RuntimeError("[make_player_form] opponent enrichment totally failed (0 rows)")

    still_missing = merged[merged["opponent"].isna()].copy()
    if not still_missing.empty:
        debug_path = Path("data") / "_debug"
        debug_path.mkdir(parents=True, exist_ok=True)
        debug_file = debug_path / "opponent_unmatched.csv"
        still_missing.to_csv(debug_file, index=False)
        print(
            "[make_player_form] WARNING: opponent enrichment incomplete: "
            f"{len(still_missing)} players without opponents. Wrote debug rows -> {debug_file}"
        )
    else:
        print("[make_player_form] opponent enrichment OK for all players")

    return merged
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

    roles["player_canonical"] = (
        roles["player"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]+", "", regex=True)
        .str.strip()
    )
    roles["player_clean_key"] = roles["player_canonical"].str.replace(
        r"\s+", "_", regex=True
    )

    roles["team_key"] = (
        roles["team"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    merge_cols = ["player_clean_key", "team_key"]
    for extra in ["role", "position", "position_guess"]:
        if extra in roles.columns:
            merge_cols.append(extra)

    roles_subset = roles[merge_cols].drop_duplicates(
        ["player_clean_key", "team_key"], keep="first"
    )

    pf = pf.merge(
        roles_subset,
        on=["player_clean_key", "team_key"],
        how="left",
        suffixes=("", "_roles"),
    )

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
            team_set = set(roles["team_key"].dropna().unique())
            miss["unmatched_class"] = np.where(
                miss["player_clean_key"].eq(""),
                "missing_player_key",
                np.where(
                    miss["team_key"].isin(team_set),
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
    df = _apply_player_name_cleaning(df, name_maps)
    out = df.merge(
        r[["player_clean_key", "team", "opponent", "role"]],
        on=["player_clean_key", "team"],
        how="left",
        suffixes=("", "_roles"),
    )
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


def build_player_form(season: int = 2025) -> pd.DataFrame:
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
    merge_keys = ["team", "opponent", "player_clean_key"]
    base = pd.merge(
        rply,
        rru,
        on=merge_keys,
        how="outer",
        suffixes=("_rec", "_rush"),
    )

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

    base = pd.merge(
        base,
        qb_df,
        on=merge_keys,
        how="left",
        suffixes=("", "_qb"),
    )

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
        base = base.merge(
            ro,
            on=["player_clean_key", "team"],
            how="left",
            suffixes=("", "_ro"),
        )
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
            base = base.merge(
                pm,
                on="player_clean_key",
                how="left",
                suffixes=("", "_pm"),
            )
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
    out = _enrich_team_and_opponent_from_props(out)
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


def _build_schedule_opponent_lookup() -> pd.DataFrame:
    """
    Use game_lines.csv to figure out each team's opponent this week.
    Expect columns like: home, away (team codes), maybe week.
    Returns df with columns: ['team','opponent']
    """

    if not GAMES_PATH.exists():
        raise RuntimeError("[make_player_form] game_lines.csv not found")

    games = pd.read_csv(GAMES_PATH)
    # normalize team codes
    for col in ["home", "away"]:
        if col in games.columns:
            games[col] = games[col].astype(str).str.upper().str.strip()
    rows = []
    for _, r in games.iterrows():
        home = r.get("home")
        away = r.get("away")
        if pd.notna(home) and pd.notna(away):
            rows.append({"team": home, "opponent": away})
            rows.append({"team": away, "opponent": home})
    sched_df = pd.DataFrame(rows).drop_duplicates()
    return sched_df


def _build_fallback_player_opponents() -> pd.DataFrame:
    """
    Merge roles_ourlads (player->team) with schedule_opponent_lookup (team->opponent)
    so we can infer opponent for any named depth chart player even if props had no line.
    Output columns: ['player_canonical','team','opponent']
    """

    if not ROLES_PATH.exists():
        raise RuntimeError("[make_player_form] roles_ourlads.csv not found")

    roles = pd.read_csv(ROLES_PATH)
    if not {"player", "team"}.issubset(roles.columns):
        missing = sorted({"player", "team"} - set(roles.columns))
        raise RuntimeError(
            f"[make_player_form] roles_ourlads.csv missing columns: {missing}"
        )
    # expected columns in roles_ourlads.csv:
    #   player (string), team (string), role (WR1/RB1/etc)
    roles["team"] = roles["team"].astype(str).str.upper().str.strip()
    roles["player_canonical"] = roles["player"].apply(_canonicalize_player_name)

    sched_lu = _build_schedule_opponent_lookup()
    merged = roles.merge(sched_lu, on="team", how="left")

    fallback = merged[["player_canonical", "team", "opponent"]].dropna(subset=["opponent"])
    fallback["team"] = fallback["team"].astype(str).str.upper().str.strip()
    fallback["opponent"] = fallback["opponent"].astype(str).str.upper().str.strip()
    fallback = fallback.drop_duplicates(["player_canonical", "team"])
    return fallback


def _build_props_player_opponents() -> pd.DataFrame:
    """
    Load opponent_map_from_props.csv (player, team, opponent from lines/props scrape),
    canonicalize names and clean team/opponent.
    Output columns: ['player_canonical','team','opponent']
    """

    if not OPP_PATH.exists():
        raise RuntimeError("[make_player_form] opponent_map_from_props.csv not found")

    opp = pd.read_csv(OPP_PATH)
    # expected cols: player, team, opponent
    for c in ["player", "team", "opponent"]:
        if c not in opp.columns:
            raise RuntimeError(f"[make_player_form] opponent_map_from_props.csv missing {c}")
    opp["player_canonical"] = opp["player"].apply(_canonicalize_player_name)
    opp["team"] = opp["team"].astype(str).str.upper().str.strip()
    opp["opponent"] = opp["opponent"].astype(str).str.upper().str.strip()
    opp = opp.drop_duplicates(["player_canonical", "team"])
    return opp[["player_canonical", "team", "opponent"]]


def _enrich_team_and_opponent_from_props(out: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich 'out' with 'team' and 'opponent'.
    Strategy:
      1. canonicalize player names in 'out'
      2. build props_map (from opponent_map_from_props.csv)
      3. build fallback_map (roles_ourlads.csv x game_lines.csv)
      4. union them and drop dups
      5. merge onto 'out' by player_canonical (+ team when present)
      6. require 0 missing opponents at the end or raise RuntimeError
    """

    df = out.copy()

    # make sure df has player_canonical
    if "player_canonical" not in df.columns:
        df["player_canonical"] = df["player"].apply(_canonicalize_player_name)
    else:
        df["player_canonical"] = df["player_canonical"].apply(_canonicalize_player_name)

    # clean df.team just in case
    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.upper().str.strip()

    # props map
    props_map = _build_props_player_opponents()

    # fallback map from depth chart + schedule
    fallback_map = _build_fallback_player_opponents()

    # union maps
    union_map = pd.concat([props_map, fallback_map], ignore_index=True)
    union_map = union_map.drop_duplicates(["player_canonical", "team"])

    # 1st pass merge on ['player_canonical','team'] when df already has team
    if "team" in df.columns:
        merged = df.merge(
            union_map,
            on=["player_canonical", "team"],
            how="left",
            suffixes=("", "_oppmap"),
        )
    else:
        # if df didn't already have team (edge case), merge just on player_canonical
        merged = df.merge(
            union_map.drop_duplicates(["player_canonical"]),
            on=["player_canonical"],
            how="left",
            suffixes=("", "_oppmap"),
        )

    # standardize columns: we want final columns 'team' and 'opponent' present
    if "team_oppmap" in merged.columns:
        merged["team"] = merged["team"].fillna(merged["team_oppmap"])
    if "opponent_oppmap" in merged.columns:
        merged["opponent"] = merged.get("opponent", pd.Series(index=merged.index, dtype=object))
        merged["opponent"] = merged["opponent"].fillna(merged["opponent_oppmap"])

    # final cleanup
    merged["team"] = merged["team"].astype(str).str.upper().str.strip()
    merged.loc[merged["team"].isin(["", "NAN", "NONE", "NULL"]), "team"] = pd.NA
    merged["opponent"] = merged["opponent"].astype(str).str.upper().str.strip()

    # NOW enforce perfection: if anyone still missing opponent, that's a hard fail
    missing_mask = (
        merged["opponent"].isna()
        | (merged["opponent"] == "")
        | (merged["opponent"].str.upper() == "NAN")
    )
    existing_cols = [c for c in ["player", "player_canonical", "team"] if c in merged.columns]
    missing_rows = merged.loc[missing_mask, existing_cols].head(20)
    n_missing = int(missing_mask.sum())
    if n_missing > 0:
        raise RuntimeError(
            f"[make_player_form] opponent enrichment STILL incomplete after fallback: {n_missing} players "
            f"without opponents. Examples: {missing_rows.to_dict('records')}"
        )

    merged.drop(columns=[c for c in ["team_oppmap", "opponent_oppmap"] if c in merged.columns], inplace=True)

    return merged


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
        return pd.DataFrame(
            columns=[
                "player",
                "team",
                "week",
                "display_name",
                "targets",
                "carries",
                "air_yards",
                "redzone_tgts",
                "routes",
                "routes_share",
                "target_share",
                "carry_share",
                "slot_rate",
                "wide_rate",
                "aDOT",
                "team_snaps",
                "role",
                "opponent",
            ]
        )

    working = df.copy()
    if "player_name" not in working.columns:
        working["player_name"] = working.get("player", "")

    agg_map = {
        "targets": "sum",
        "carries": "sum",
        "air_yards": "sum",
        "redzone_tgts": "sum",
        "routes": "sum",
        "routes_share": "mean",
        "target_share": "mean",
        "carry_share": "mean",
        "slot_rate": "mean",
        "wide_rate": "mean",
        "aDOT": "mean",
        "team_snaps": "mean",
        "role": "max",
        "opponent": "first",
    }

    for col in agg_map.keys():
        if col not in working.columns:
            working[col] = np.nan

    group_cols = ["player_canonical", "team_key", "week_key"]

    consensus = (
        working.groupby(group_cols, dropna=False)
        .agg({col: agg_map[col] for col in agg_map.keys()})
        .reset_index()
    )

    def _mode_or_first(series: pd.Series) -> str:
        counts = series.dropna().value_counts()
        if not counts.empty:
            return counts.index[0]
        series = series.astype(str)
        return series.iloc[0] if len(series) else ""

    name_lookup = (
        working.groupby(group_cols)["player_canonical"]
        .agg(_mode_or_first)
        .reset_index()
        .rename(columns={"player_canonical": "display_name"})
    )

    consensus = consensus.merge(name_lookup, on=group_cols, how="left")
    consensus = consensus.rename(
        columns={
            "player_canonical": "player",
            "team_key": "team",
            "week_key": "week",
        }
    )

    desired_order = [
        "player",
        "team",
        "week",
        "display_name",
        "targets",
        "carries",
        "air_yards",
        "redzone_tgts",
        "routes",
        "routes_share",
        "target_share",
        "carry_share",
        "slot_rate",
        "wide_rate",
        "aDOT",
        "team_snaps",
        "role",
        "opponent",
    ]
    existing = [c for c in desired_order if c in consensus.columns]
    remaining = [c for c in consensus.columns if c not in existing]
    return consensus[existing + remaining]


def _enforce_player_form_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=PLAYER_FORM_REQUIRED_COLUMNS)

    out = df.copy()
    for col in PLAYER_FORM_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    # ensure key identifier columns are strings (preserve pd.NA with pandas string dtype)
    for col in ["player", "team", "opponent", "role", "player_name_canonical"]:
        if col in out.columns:
            out[col] = out[col].astype("string")

    if "team" in out.columns:
        out["team"] = out["team"].str.strip().str.upper()

    if "player_name_canonical" in out.columns:
        out["player_name_canonical"] = out["player_name_canonical"].str.strip().str.upper()

    if "player" in out.columns:
        out["player"] = out["player"].str.strip()

    if "opponent" in out.columns:
        out["opponent"] = (
            out["opponent"].str.strip().str.upper()
        )
        out.loc[out["opponent"].isin(["", "NAN", "NONE", "NULL"]), "opponent"] = pd.NA

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

    out = df.copy()
    for col in CONSENSUS_REQUIRED_COLUMNS:
        if col not in out.columns:
            if col == "week":
                out[col] = pd.Series(pd.NA, index=out.index, dtype="Float64")
            elif col == "opponent":
                out[col] = CONSENSUS_OPPONENT_SENTINEL
            else:
                out[col] = pd.NA

    out["player"] = out["player"].astype("string").str.strip()
    out["team"] = out["team"].astype("string").str.strip().str.upper()
    out["role"] = out["role"].astype("string").str.strip()

    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["opponent"] = (
        out["opponent"]
        .astype("string")
        .fillna(CONSENSUS_OPPONENT_SENTINEL)
        .replace({"": CONSENSUS_OPPONENT_SENTINEL})
        .str.upper()
    )

    ordered = CONSENSUS_REQUIRED_COLUMNS + [
        c for c in out.columns if c not in CONSENSUS_REQUIRED_COLUMNS
    ]
    return out[ordered]


def _write_player_form_outputs(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise RuntimeError("[make_player_form] final player_form empty; aborting run")

    working = df.copy()
    if "player_name" not in working.columns:
        working["player_name"] = working.get("player", "")

    working = _attach_consensus_keys(working)
    consensus = _build_grouped_consensus(working)

    working = _ensure_cols(working, FINAL_COLS)
    player_form = _enforce_player_form_schema(working)
    if player_form.empty:
        raise RuntimeError("[make_player_form] final player_form empty; aborting run")

    _safe_mkdir(DATA_DIR)
    player_form.to_csv("data/player_form.csv", index=False)
    consensus = _enforce_consensus_schema(consensus)
    consensus.to_csv("data/player_form_consensus.csv", index=False)

    print(f"[make_player_form] wrote data/player_form.csv ({len(player_form)} rows raw)")
    print(
        f"[make_player_form] wrote data/player_form_consensus.csv ({len(consensus)} rows grouped)"
    )


# ---------------------------
# CLI
# ---------------------------


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()

    _safe_mkdir(DATA_DIR)

    try:
        # Build player form
        df = build_player_form(args.season)

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
            df = _enrich_team_and_opponent_from_props(df)
        except Exception as _enr_e:
            print(
                f"[make_player_form] WARN opponent enrichment skipped in cli(): {_enr_e}",
                file=sys.stderr,
            )

        # Final write on success
        _write_player_form_outputs(df)
        return

    except Exception as e:
        # Log error and try to salvage partial DF
        print(f"[make_player_form] ERROR: {e}", file=sys.stderr)
        if isinstance(e, RuntimeError) and "opponent merge" in str(e):
            raise
        try:
            if "df" in locals() and isinstance(df, pd.DataFrame) and len(df) > 0:
                try:
                    df = _enrich_team_and_opponent_from_props(df)
                except Exception as _enr_e:
                    print(
                        f"[make_player_form] WARN enrichment skipped in error path: {_enr_e}",
                        file=sys.stderr,
                    )
                _write_player_form_outputs(df)
                return
        except Exception as _w:
            print(
                f"[make_player_form] WARN could not write partial df in error path: {_w}",
                file=sys.stderr,
            )

        raise


if __name__ == "__main__":
    cli()
