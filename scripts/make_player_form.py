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
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List
import re

import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPATH = Path(DATA_DIR) / "player_form.csv"
CONS_PATH = Path("data") / "player_form_consensus.csv"
OPP_PATH = Path("data") / "opponent_map_from_props.csv"
ROLES_PATH = Path("data") / "roles_ourlads.csv"


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
def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


FINAL_COLS = [
    "player",
    "team",
    "week",
    "opponent",
    "season",
    "position",
    "role",
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


def _inject_week_opponent_and_roles(out: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich 'out' with 'week' and 'opponent' from data/opponent_map_from_props.csv.

    This function:
    - Loads opponent_map_from_props.csv (already built earlier in the workflow).
    - Canonicalizes player names and team codes on BOTH sides.
    - Merges using (player, team, week) instead of just (player, team),
      because the same player/team appears in multiple different weeks.
    - Raises RuntimeError if we still can't map opponent/week for anyone,
      and drops debug previews in data/_debug for inspection.
    """

    OPP_PATH = Path("data") / "opponent_map_from_props.csv"

    if not OPP_PATH.exists():
        raise RuntimeError("[make_player_form] opponent_map_from_props.csv not found")

    opp = pd.read_csv(OPP_PATH)

    required_cols = {"player", "team", "week", "opponent"}
    missing = required_cols - set(opp.columns)
    if missing:
        raise RuntimeError(
            f"[make_player_form] opponent_map_from_props.csv missing columns: {sorted(missing)}"
        )

    import re
    import unicodedata

    def _deaccent_local(s: str) -> str:
        try:
            return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        except Exception:
            return s

    def _clean_player_name_key(raw: str) -> str:
        """
        Normalize names like:
        'DAK PRESCOTT', 'Dak Prescott', 'D. Prescott', 'Javonte U Williams'
        -> 'dakprescott' / 'javontewilliams'

        Strategy:
        - strip accents
        - drop punctuation / jersey numbers / obvious suffixes (JR, SR, II, etc.)
        - smash to [a-z0-9] lowercase using first+last fallback
        """
        s = str(raw or "").strip()
        s = _deaccent_local(s)
        # remove leading jersey/# fragments
        s = re.sub(r"^[0-9#\-\:\s]+", "", s)
        # normalize spacing/punctuation
        s = s.replace(",", " ")
        s = re.sub(r"\s+", " ", s).strip()
        # remove suffix tokens
        tokens = [t for t in s.split() if t.upper() not in ("JR", "SR", "II", "III", "IV", "V")]
        if not tokens:
            tokens = [s]
        all_joined = re.sub(r"[^a-z0-9]", "", "".join(tokens).lower())
        core_joined = re.sub(r"[^a-z0-9]", "", (tokens[0] + tokens[-1]).lower())
        return core_joined or all_joined

    def _clean_team_key(raw: str) -> str:
        """
        Normalize team strings like:
        'DAL', 'Dallas', 'Dallas Cowboys', 'Cowboys'
        -> 'DAL'
        """
        TEAM_MAP = {
            "ARI":"ARI","ARIZONA":"ARI","CARDINALS":"ARI","ARIZONA CARDINALS":"ARI",
            "ATL":"ATL","ATLANTA":"ATL","FALCONS":"ATL","ATLANTA FALCONS":"ATL",
            "BAL":"BAL","BALTIMORE":"BAL","RAVENS":"BAL","BALTIMORE RAVENS":"BAL",
            "BUF":"BUF","BUFFALO":"BUF","BILLS":"BUF","BUFFALO BILLS":"BUF",
            "CAR":"CAR","CAROLINA":"CAR","PANTHERS":"CAR","CAROLINA PANTHERS":"CAR",
            "CHI":"CHI","CHICAGO":"CHI","BEARS":"CHI","CHICAGO BEARS":"CHI",
            "CIN":"CIN","CINCINNATI":"CIN","BENGALS":"CIN","CINCINNATI BENGALS":"CIN",
            "CLE":"CLE","CLEVELAND":"CLE","BROWNS":"CLE","CLEVELAND BROWNS":"CLE",
            "DAL":"DAL","DALLAS":"DAL","COWBOYS":"DAL","DALLAS COWBOYS":"DAL",
            "DEN":"DEN","DENVER":"DEN","BRONCOS":"DEN","DENVER BRONCOS":"DEN",
            "DET":"DET","DETROIT":"DET","LIONS":"DET","DETROIT LIONS":"DET",
            "GB":"GB","GNB":"GB","GREEN BAY":"GB","PACKERS":"GB","GREEN BAY PACKERS":"GB",
            "HOU":"HOU","HOUSTON":"HOU","TEXANS":"HOU","HOUSTON TEXANS":"HOU",
            "IND":"IND","INDIANAPOLIS":"IND","COLTS":"IND","INDIANAPOLIS COLTS":"IND",
            "JAX":"JAX","JAC":"JAX","JACKSONVILLE":"JAX","JAGUARS":"JAX","JACKSONVILLE JAGUARS":"JAX",
            "KC":"KC","KCC":"KC","KANSAS CITY":"KC","CHIEFS":"KC","KANSAS CITY CHIEFS":"KC",
            "LAC":"LAC","CHARGERS":"LAC","LOS ANGELES CHARGERS":"LAC",
            "LAR":"LAR","LA":"LAR","RAMS":"LAR","LOS ANGELES RAMS":"LAR",
            "LV":"LV","LVR":"LV","RAIDERS":"LV","LAS VEGAS":"LV","LAS VEGAS RAIDERS":"LV",
            "MIA":"MIA","MIAMI":"MIA","DOLPHINS":"MIA","MIAMI DOLPHINS":"MIA",
            "MIN":"MIN","MINNESOTA":"MIN","VIKINGS":"MIN","MINNESOTA VIKINGS":"MIN",
            "NE":"NE","NWE":"NE","NEW ENGLAND":"NE","PATRIOTS":"NE","NEW ENGLAND PATRIOTS":"NE",
            "NO":"NO","NOR":"NO","NEW ORLEANS":"NO","SAINTS":"NO","NEW ORLEANS SAINTS":"NO",
            "NYG":"NYG","NEW YORK GIANTS":"NYG","GIANTS":"NYG",
            "NYJ":"NYJ","NEW YORK JETS":"NYJ","JETS":"NYJ",
            "PHI":"PHI","PHILADELPHIA":"PHI","EAGLES":"PHI","PHILADELPHIA EAGLES":"PHI",
            "PIT":"PIT","PITTSBURGH":"PIT","STEELERS":"PIT","PITTSBURGH STEELERS":"PIT",
            "SEA":"SEA","SEATTLE":"SEA","SEAHAWKS":"SEA","SEATTLE SEAHAWKS":"SEA",
            "SF":"SF","SFO":"SF","SAN FRANCISCO":"SF","49ERS":"SF","SAN FRANCISCO 49ERS":"SF",
            "TB":"TB","TAM":"TB","TAMPA BAY":"TB","BUCCANEERS":"TB","TAMPA BAY BUCCANEERS":"TB",
            "TEN":"TEN","TENNESSEE":"TEN","TITANS":"TEN","TENNESSEE TITANS":"TEN",
            "WAS":"WAS","WSH":"WAS","WASHINGTON":"WAS","COMMANDERS":"WAS","WASHINGTON COMMANDERS":"WAS"
        }
        raw_up = str(raw or "").strip().upper()
        if raw_up in TEAM_MAP:
            return TEAM_MAP[raw_up]
        raw_stripped = re.sub(r"[^A-Z0-9 ]+", "", raw_up).strip().upper()
        if raw_stripped in TEAM_MAP:
            return TEAM_MAP[raw_stripped]
        return raw_up

    # --- build keys on 'out' (the base frame we were passed in) ---
    enriched = out.copy()

    # we REQUIRE that 'week' already exists in base;
    # if it doesn't, we can't safely match week-specific opponents
    if "week" not in enriched.columns:
        debug_dir = Path("data") / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        enriched.head(50).to_csv(debug_dir / "merge_failure_no_week_in_base.csv", index=False)
        raise RuntimeError("[make_player_form] base/out frame has no 'week' column before matchup merge")

    enriched["__week_key"] = pd.to_numeric(enriched["week"], errors="coerce")
    enriched["__player_clean_key"] = (
        enriched["player"]
        .astype(str)
        .map(_clean_player_name_key)
    )
    enriched["__team_clean_key"] = (
        enriched["team"]
        .astype(str)
        .map(_clean_team_key)
    )

    # --- build keys on the opponent map side ---
    opp = opp.copy()
    opp["__player_clean_key"] = (
        opp["player"]
        .astype(str)
        .map(_clean_player_name_key)
    )
    opp["__team_clean_key"] = (
        opp["team"]
        .astype(str)
        .map(_clean_team_key)
    )
    opp["__week_key"] = pd.to_numeric(opp["week"], errors="coerce")

    # We need the merge to be one-week-specific: (player, team, week).
    # Dedupe opp on that triple key so pandas is OK with many_to_one.
    right_subset = (
        opp[
            [
                "__player_clean_key",
                "__team_clean_key",
                "__week_key",
                "week",
                "opponent",
            ]
        ]
        .drop_duplicates(
            subset=["__player_clean_key", "__team_clean_key", "__week_key"],
            keep="last",
        )
    )

    merged = enriched.merge(
        right_subset,
        how="left",
        left_on=["__player_clean_key", "__team_clean_key", "__week_key"],
        right_on=["__player_clean_key", "__team_clean_key", "__week_key"],
        validate="many_to_one",
    )

    # --- extract matchup columns safely ---
    #
    # After the merge, pandas may have created columns like:
    #   week_x (from enriched/base),   week_y (from opp)
    #   opponent_x / opponent_y
    #
    # We want to:
    #   1. derive 'week_final' and 'opponent_final' from the RIGHT side (opp),
    #   2. copy them back into canonical 'week' and 'opponent' columns,
    #   3. detect if we actually mapped ANY opponents. If not, dump debug and hard-stop.

    # Helper to pick a column if it exists
    def _pick_col(frame, primary, fallback=None):
        if primary in frame.columns:
            return frame[primary]
        if fallback and fallback in frame.columns:
            return frame[fallback]
        # otherwise return a Series of NaN
        return pd.Series([pd.NA] * len(frame), index=frame.index)

    # opponent from the RIGHT side (opp)
    merged["opponent_final"] = _pick_col(
        merged,
        "opponent_y",  # typical suffix when both sides have 'opponent'
        fallback="opponent"
    ).astype(str).str.strip()

    # numeric-ish week from the RIGHT side (opp)
    merged["week_final_raw"] = _pick_col(
        merged,
        "week_y",   # typical suffix when both sides have 'week'
        fallback="week"
    )

    # coerce week_final_raw to numeric where possible
    merged["week_final"] = pd.to_numeric(merged["week_final_raw"], errors="coerce")

    # if week_final is entirely NaN, we might still have a usable
    # left-side 'week_x' from enriched. Use that as a fallback.
    if merged["week_final"].isna().all():
        merged["week_final"] = pd.to_numeric(
            _pick_col(merged, "week_x", fallback="week"),
            errors="coerce"
        )

    # Copy stabilized values into canonical columns that downstream code expects.
    # 'week' stays numeric, 'opponent' is uppercase string or NA.
    merged["week"] = merged["week_final"]
    merged["opponent"] = merged["opponent_final"].where(
        merged["opponent_final"].str.len() > 0,
        pd.NA
    ).str.upper()

    # --- determine if we successfully mapped anything ---
    # If literally nobody got a non-null opponent, that's still a fatal data error.
    if merged["opponent"].isna().all():
        debug_dir = Path("data") / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # write a debug preview that will ALWAYS work
        debug_cols = []
        for c in [
            "player",
            "team",
            "week",
            "opponent",
            "__player_clean_key",
            "__team_clean_key",
            "__week_key",
            "week_x",
            "week_y",
            "opponent_x",
            "opponent_y",
        ]:
            if c in merged.columns and c not in debug_cols:
                debug_cols.append(c)

        merged[debug_cols].head(100).to_csv(
            debug_dir / "merge_failure_preview.csv",
            index=False
        )
        raise RuntimeError(
            "[make_player_form] cannot assign opponent/week after keyed merge (player, team, week)"
        )

    # If we DID map at least someone, great. We can drop helper cols and continue.

    cols_to_drop = [
        "__player_clean_key",
        "__team_clean_key",
        "__week_key",
        "week_x",
        "week_y",
        "opponent_x",
        "opponent_y",
        "week_final_raw",
        "week_final",
        "opponent_final",
    ]
    for col in cols_to_drop:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True, errors="ignore")

    return merged
# === BEGIN: SURGICAL NAME NORMALIZATION HELPERS (idempotent) ===
try:
    _NAME_HELPERS_DEFINED
except NameError:
    import re as _re_nh
    import unicodedata as _ud_nh

    _NAME_HELPERS_DEFINED = True

    _SUFFIX_RE_NH = _re_nh.compile(r"\b(JR|SR|II|III|IV|V)\b\.?", _re_nh.IGNORECASE)
    _LEADING_NUM_RE_NH = _re_nh.compile(
        r"^\s*(?:#\s*)?\d+\s*[-–—:]?\s*", _re_nh.UNICODE
    )

    def _deaccent_nh(s: str) -> str:
        try:
            return _ud_nh.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        except Exception:
            return s

    def _clean_person_name_nh(s: str) -> str:
        s = (s or "").replace("\xa0", " ").strip()
        s = _LEADING_NUM_RE_NH.sub("", s)
        s = s.replace(".", "")
        s = _SUFFIX_RE_NH.sub("", s)
        s = _re_nh.sub(r"\s+", " ", s)
        s = _deaccent_nh(s)
        return s.strip()

    def _player_key_from_name_nh(s: str) -> str:
        s = _clean_person_name_nh(s)
        return _re_nh.sub(r"[^a-z0-9]", "", s.lower())

    def _player_initial_last_key_nh(s: str) -> str:
        s = _clean_person_name_nh(s)
        if not s:
            return ""
        parts = s.split()
        if not parts:
            return ""
        first = parts[0]
        last = parts[-1] if len(parts) > 1 else parts[0]
        if len(parts) == 1:
            token = parts[0]
            match = _re_nh.search(r"(?<=[A-Z])([A-Z][a-z])", token)
            if match:
                start_idx = match.start(1)
                leading = token[:start_idx]
                surname = token[start_idx:]
                if leading and surname:
                    first = leading
                    last = surname
        if not first:
            return ""
        key = f"{first[0]}{last}".lower()
        return _re_nh.sub(r"[^a-z0-9]", "", key)


# === END: SURGICAL NAME NORMALIZATION HELPERS ===


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
    import os, re

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
        roles["player"] = roles["player"].map(_clean_name)
    roles["team"] = roles.get("team", "").astype(str).map(_canon_team)
    roles["role"] = roles.get("role", "").astype(str).str.upper().str.strip()
    if "position" not in roles.columns and "role" in roles.columns:
        roles["position"] = roles["role"].str.extract(r"([A-Z]+)")

    def _rank(r):
        m = re.search(r"(\d+)$", str(r))
        return int(m.group(1)) if m else 999

    if {"team", "player", "role"}.issubset(roles.columns):
        roles["_rk"] = roles["role"].map(_rank)
        roles = (
            roles.sort_values(["team", "player", "_rk"])
            .drop_duplicates(["team", "player"], keep="first")
            .drop(columns=["_rk"])
        )
    pf_team_raw = pf["team"].astype(str).str.upper().str.strip()
    pf_team_canon = pf_team_raw.map(_canon_team)
    pf["team"] = np.where(pf_team_canon == "", pf_team_raw, pf_team_canon)
    pf["player_key_full"] = pf["player"].astype(str).map(_player_key_from_name_nh)
    pf["player_key_initial_last"] = (
        pf["player"].astype(str).map(_player_initial_last_key_nh)
    )
    roles_join = roles.copy()
    if "player" in roles_join.columns:
        roles_join = roles_join.rename(columns={"player": "player_depth_name"})
    if "player_depth_name" not in roles_join.columns:
        roles_join["player_depth_name"] = roles_join.get("player_join", "")
    roles_team_raw = roles_join["team"].astype(str).str.upper().str.strip()
    roles_team_canon = roles_team_raw.map(_canon_team)
    roles_join["team"] = np.where(
        roles_team_canon == "", roles_team_raw, roles_team_canon
    )
    roles_join = roles_join[roles_join["team"] != ""].copy()
    roles_join["player_key_full"] = (
        roles_join.get("player_depth_name", "")
        .astype(str)
        .map(_player_key_from_name_nh)
    )
    roles_join["player_key_initial_last"] = (
        roles_join.get("player_depth_name", "")
        .astype(str)
        .map(_player_initial_last_key_nh)
    )
    roles_join = roles_join.loc[:, ~roles_join.columns.duplicated()].copy()
    roles_full = roles_join[
        [
            col
            for col in [
                "team",
                "player_key_full",
                "role",
                "position",
                "player_depth_name",
            ]
            if col in roles_join.columns
        ]
    ].copy()
    roles_full = roles_full.loc[
        roles_full["player_key_full"].notna() & (roles_full["player_key_full"] != "")
    ]
    roles_full = roles_full.sort_values(["team", "player_key_full"]).drop_duplicates(
        ["team", "player_key_full"], keep="first"
    )
    roles_full = roles_full.rename(
        columns={
            "role": "role_depth",
            "position": "position_depth",
            "player_depth_name": "player_depth_name_full",
        }
    )
    try:
        pf = pf.merge(
            roles_full,
            on=["team", "player_key_full"],
            how="left",
            validate="many_to_one",
        )
    except Exception as e:
        print(f"[make_player_form] WARN roles merge (full key) issue: {e}")
        pf = pf.merge(
            roles_full,
            on=["team", "player_key_full"],
            how="left",
        )

    roles_alias = roles_join[
        [
            col
            for col in [
                "team",
                "player_key_initial_last",
                "role",
                "position",
                "player_depth_name",
            ]
            if col in roles_join.columns
        ]
    ].copy()
    roles_alias = roles_alias.loc[
        roles_alias["player_key_initial_last"].notna()
        & (roles_alias["player_key_initial_last"] != "")
    ]
    roles_alias = roles_alias.sort_values(
        ["team", "player_key_initial_last"]
    ).drop_duplicates(["team", "player_key_initial_last"], keep="first")
    roles_alias = roles_alias.rename(
        columns={
            "role": "role_depth_alias",
            "position": "position_depth_alias",
            "player_depth_name": "player_depth_name_alias",
        }
    )
    try:
        pf = pf.merge(
            roles_alias,
            on=["team", "player_key_initial_last"],
            how="left",
            validate="many_to_one",
        )
    except Exception as e:
        print(f"[make_player_form] WARN roles merge (alias key) issue: {e}")
        pf = pf.merge(
            roles_alias,
            on=["team", "player_key_initial_last"],
            how="left",
        )

    if "position_depth" in pf.columns:
        pf["position"] = pf["position"].combine_first(pf["position_depth"])
    if "position_depth_alias" in pf.columns:
        pf["position"] = pf["position"].combine_first(pf["position_depth_alias"])
    if "role_depth" in pf.columns:
        pf["role"] = pf["role"].combine_first(pf["role_depth"])
    if "role_depth_alias" in pf.columns:
        pf["role"] = pf["role"].combine_first(pf["role_depth_alias"])

    # Key-only fallbacks when the depth charts have unique names
    unmatched_mask = pf["role"].isna()
    if unmatched_mask.any():
        unique_full = roles_join[["player_key_full", "role", "position"]].copy()
        unique_full = unique_full[
            unique_full["player_key_full"].notna()
            & (unique_full["player_key_full"] != "")
        ]
        dup_full = unique_full["player_key_full"].duplicated(keep=False)
        unique_full = unique_full[~dup_full].drop_duplicates("player_key_full")
        if not unique_full.empty:
            role_map_full = unique_full.set_index("player_key_full")["role"].to_dict()
            pos_map_full = unique_full.set_index("player_key_full")[
                "position"
            ].to_dict()
            pf.loc[unmatched_mask, "role"] = pf.loc[unmatched_mask, "role"].fillna(
                pf.loc[unmatched_mask, "player_key_full"].map(role_map_full)
            )
            pf.loc[unmatched_mask, "position"] = pf.loc[
                unmatched_mask, "position"
            ].fillna(pf.loc[unmatched_mask, "player_key_full"].map(pos_map_full))
            unmatched_mask = pf["role"].isna()

    if unmatched_mask.any():
        unique_alias = roles_join[
            ["player_key_initial_last", "role", "position"]
        ].copy()
        unique_alias = unique_alias[
            unique_alias["player_key_initial_last"].notna()
            & (unique_alias["player_key_initial_last"] != "")
        ]
        dup_alias = unique_alias["player_key_initial_last"].duplicated(keep=False)
        unique_alias = unique_alias[~dup_alias].drop_duplicates(
            "player_key_initial_last"
        )
        if not unique_alias.empty:
            role_map_alias = unique_alias.set_index("player_key_initial_last")[
                "role"
            ].to_dict()
            pos_map_alias = unique_alias.set_index("player_key_initial_last")[
                "position"
            ].to_dict()
            pf.loc[unmatched_mask, "role"] = pf.loc[unmatched_mask, "role"].fillna(
                pf.loc[unmatched_mask, "player_key_initial_last"].map(role_map_alias)
            )
            pf.loc[unmatched_mask, "position"] = pf.loc[
                unmatched_mask, "position"
            ].fillna(
                pf.loc[unmatched_mask, "player_key_initial_last"].map(pos_map_alias)
            )
    drop_cols = [
        c
        for c in pf.columns
        if c.endswith("_depth")
        or c.endswith("_depth_alias")
        or c in {"player_key_full", "player_key_initial_last"}
        or c.startswith("player_depth_name_")
    ]
    if drop_cols:
        pf.drop(columns=drop_cols, inplace=True, errors="ignore")
    pf = pf.loc[:, ~pf.columns.duplicated()]
    try:
        miss = pf[pf["role"].isna()][["player", "team"]].copy()
        if not miss.empty:
            os.makedirs(data_dir, exist_ok=True)
            miss.to_csv(
                os.path.join(data_dir, "unmatched_roles_merge.csv"), index=False
            )
            print(
                f"[make_player_form] unmatched after roles merge: {len(miss)} → {os.path.join(data_dir, 'unmatched_roles_merge.csv')}"
            )
    except Exception:
        pass
    try:
        cov = pf["position"].notna().mean()
        print(f"[make_player_form] merged depth roles → coverage now {cov:.2%}")
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
    out = df.merge(
        r[["player", "team", "opponent", "role"]],
        on=["player", "team"],
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

    # Merge all
    base = pd.merge(rply, rru, on=["team", "opponent", "player"], how="outer")
    base = pd.merge(base, qb_df, on=["team", "opponent", "player"], how="left")
    base["rz_share"] = base[["rz_tgt_share", "rz_rush_share"]].max(axis=1)
    base["season"] = int(season)

    print("[pf] base after concat/merge:", len(base))

    # Initialize position/role as NaN (do not uppercase yet)
    base["position"] = np.nan
    base["role"] = np.nan

    # Normalize keys
    base = _ensure_cols(base, ["opponent"])
    base["player"] = _norm_name(base["player"].astype(str))
    base = base.loc[_valid_player_mask(base["player"])].copy()
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
        base = base.merge(ro, on=["player", "team"], how="left", suffixes=("", "_ro"))
        if "position_ro" in base.columns:
            base["position"] = base["position"].combine_first(base["position_ro"])
            base.drop(columns=["position_ro"], inplace=True, errors="ignore")

    # Fallback: players master (merge by player only)
    if base["position"].isna().all():
        pm = _load_players_master()
        if not pm.empty:
            pm["player"] = _norm_name(pm["player"].astype(str))
            base = base.merge(pm, on="player", how="left", suffixes=("", "_pm"))
            if "position_pm" in base.columns:
                base["position"] = base["position"].combine_first(base["position_pm"])
                base.drop(columns=["position_pm"], inplace=True, errors="ignore")

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

    # roles.csv (non-destructive) then infer roles
    base = _merge_depth_roles(base)
    if base.get("role", pd.Series(dtype=object)).isna().all():
        base = _infer_roles_minimal(base)

    base = _ensure_cols(base, FINAL_COLS)
    base = _inject_week_opponent_and_roles(base)
    if "week" in base.columns:
        base["week"] = pd.to_numeric(base["week"], errors="coerce")
    if "opponent" in base.columns:
        base["opponent"] = base["opponent"].astype(str).str.strip().str.upper()
        base.loc[base["opponent"].isin(["", "NAN"]), "opponent"] = np.nan

    consensus = _build_season_consensus(base)
    consensus_to_write = _ensure_cols(pd.DataFrame(), FINAL_COLS)
    if consensus is not None and not _is_empty(consensus):
        consensus_to_write = _ensure_cols(consensus.copy(), FINAL_COLS)
        if not consensus_to_write.empty:
            consensus_to_write = _inject_week_opponent_and_roles(consensus_to_write)
            consensus_to_write["week"] = pd.NA
            consensus_to_write["opponent"] = CONSENSUS_OPPONENT_SENTINEL

    _safe_mkdir(DATA_DIR)
    consensus_to_write.to_csv(CONS_PATH, index=False)
    print(f"[pf] consensus rows: {len(consensus_to_write)} → {CONS_PATH}")

    frames: List[pd.DataFrame] = [base[FINAL_COLS]]
    if not consensus_to_write.empty:
        frames.append(consensus_to_write[FINAL_COLS])

    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["player", "team", "opponent", "season"])
        .reset_index(drop=True)
    )
    out = _enrich_team_and_opponent_from_props(out)
    if "week" in out.columns:
        out["week"] = pd.to_numeric(out["week"], errors="coerce")

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


def _enrich_team_and_opponent_from_props(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing team/opponent assignments using data/opponent_map_from_props.csv."""

    if df is None or _is_empty(df):
        return df

    out = df.copy()
    out = _ensure_cols(out, ["player", "team", "opponent"])

    path = os.path.join(DATA_DIR, "opponent_map_from_props.csv")
    try:
        mapping = pd.read_csv(path)
    except FileNotFoundError:
        mapping = pd.DataFrame(columns=["player", "team", "opponent"])
    except Exception as exc:
        raise RuntimeError(f"failed to read opponent map: {exc}") from exc

    if mapping is None or mapping.empty:
        mapping = pd.DataFrame(columns=["player", "team", "opponent"])

    mapping.columns = [c.lower() for c in mapping.columns]
    if "player" not in mapping.columns:
        if "player_name" in mapping.columns:
            mapping = mapping.rename(columns={"player_name": "player"})
        elif "name" in mapping.columns:
            mapping = mapping.rename(columns={"name": "player"})
        else:
            mapping["player"] = np.nan

    mapping["player"] = _norm_name(
        mapping.get("player", pd.Series([], dtype=object)).astype(str)
    )
    mapping = mapping[
        mapping["player"].notna() & mapping["player"].astype(str).str.strip().ne("")
    ]

    merge_keys: List[str] = ["player"]
    if "team" in mapping.columns:
        mapping["team"] = (
            mapping["team"]
            .where(mapping["team"].notna(), "")
            .astype(str)
            .str.upper()
            .str.strip()
        )
        mapping["team"] = mapping["team"].map(_canon_team)
        mapping.loc[mapping["team"].isin(["", None]), "team"] = np.nan
        if mapping["team"].notna().any():
            mapping = mapping.dropna(subset=["team"])
            merge_keys.append("team")
        else:
            mapping = mapping.drop(columns=["team"])

    if "opponent" not in mapping.columns:
        mapping["opponent"] = np.nan

    opp_norm = (
        mapping["opponent"]
        .where(mapping["opponent"].notna(), "")
        .astype(str)
        .str.upper()
        .str.strip()
    )
    sentinel_mask = opp_norm.eq(CONSENSUS_OPPONENT_SENTINEL)
    mapping["opponent"] = opp_norm.map(_canon_team)
    mapping.loc[sentinel_mask, "opponent"] = CONSENSUS_OPPONENT_SENTINEL
    mapping["opponent"] = mapping["opponent"].replace("", np.nan)

    mapping = mapping[merge_keys + ["opponent"]].drop_duplicates(merge_keys)

    if "team" in out.columns:
        out["team"] = out["team"].astype(str).str.upper().str.strip().map(_canon_team)

    if mapping.empty:
        opponents = out.get("opponent", pd.Series(index=out.index, dtype=object))
        opp_norm = (
            opponents.where(opponents.notna(), "").astype(str).str.upper().str.strip()
        )
        missing_mask = opp_norm.eq("")
        missing_mask &= ~opp_norm.eq(CONSENSUS_OPPONENT_SENTINEL)
        if missing_mask.any():
            preview_df = out.loc[missing_mask, ["player", "team"]].head(10)
            preview = ", ".join(
                preview_df.apply(
                    lambda row: (
                        f"{row['player']} ({row['team']})"
                        if row.get("team")
                        else str(row["player"])
                    ),
                    axis=1,
                )
            )
            count = int(missing_mask.sum())
            raise RuntimeError(
                "opponent enrichment incomplete: opponent_map_from_props.csv had no assignments "
                f"and {count} players remain without opponents: {preview}"
            )
        return out

    enriched = out.merge(mapping, on=merge_keys, how="left", suffixes=("", "_props"))

    current = enriched.get("opponent", pd.Series(index=enriched.index, dtype=object))
    current_norm = (
        current.where(current.notna(), "").astype(str).str.upper().str.strip()
    )
    current_sentinel = current_norm.eq(CONSENSUS_OPPONENT_SENTINEL)
    fill_mask = current_norm.eq("") & ~current_sentinel

    enriched.loc[fill_mask, "opponent"] = enriched.loc[fill_mask, "opponent_props"]
    enriched.drop(columns=["opponent_props"], inplace=True, errors="ignore")

    opp_final = (
        enriched["opponent"]
        .where(enriched["opponent"].notna(), "")
        .astype(str)
        .str.upper()
        .str.strip()
    )
    final_sentinel = opp_final.eq(CONSENSUS_OPPONENT_SENTINEL)
    enriched["opponent"] = opp_final.map(_canon_team)
    enriched.loc[final_sentinel, "opponent"] = CONSENSUS_OPPONENT_SENTINEL
    enriched["opponent"] = enriched["opponent"].replace("", np.nan)

    missing_mask = enriched["opponent"].isna() | enriched["opponent"].astype(
        str
    ).str.strip().eq("")
    missing_mask &= (
        ~enriched["opponent"].astype(str).str.upper().eq(CONSENSUS_OPPONENT_SENTINEL)
    )

    if missing_mask.any():
        preview_df = enriched.loc[missing_mask, ["player", "team"]].head(10)
        preview = ", ".join(
            preview_df.apply(
                lambda row: (
                    f"{row['player']} ({row['team']})"
                    if row.get("team")
                    else str(row["player"])
                ),
                axis=1,
            )
        )
        count = int(missing_mask.sum())
        raise RuntimeError(
            f"opponent enrichment incomplete: {count} players without opponents: {preview}"
        )

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
    Returns DataFrame with columns: player, team, player_key (stable).
    """
    path = os.path.join("outputs", "props_raw.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["player", "team", "opponent", "player_key"])
    try:
        pr = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["player", "team", "opponent", "player_key"])

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

    pr["player_key"] = (
        pr["player"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )
    return pr[["player", "team", "opponent", "player_key"]].drop_duplicates()


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
    df["player_key"] = (
        df["player"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )
    need = df.merge(
        props_players[["player_key"]].drop_duplicates(), on="player_key", how="inner"
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
        df = df[FINAL_COLS]
        if df.empty:
            raise RuntimeError("[make_player_form] final player_form empty; aborting run")
        df.to_csv(OUTPATH, index=False)
        df.to_csv(CONS_PATH, index=False)
        print(f"[make_player_form] wrote data/player_form.csv ({len(df)} rows)")
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
                df = _ensure_cols(df, FINAL_COLS)[FINAL_COLS]
                if df.empty:
                    raise RuntimeError("[make_player_form] final player_form empty; aborting run")
                df.to_csv(OUTPATH, index=False)
                df.to_csv(CONS_PATH, index=False)
                print(
                    f"[make_player_form] wrote data/player_form.csv ({len(df)} rows) (after handled error)"
                )
                return
        except Exception as _w:
            print(
                f"[make_player_form] WARN could not write partial df in error path: {_w}",
                file=sys.stderr,
            )

        raise


if __name__ == "__main__":
    cli()
