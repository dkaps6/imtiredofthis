#!/usr/bin/env python3
from __future__ import annotations

# scripts/make_metrics.py
__doc__ = """
Build a single, pricing-ready table that joins props with all context metrics.

Inputs (best-effort; missing inputs won’t crash):
- outputs/props_raw.csv
- data/team_form.csv
- data/player_form.csv
- data/coverage.csv
- data/cb_assignments.csv
- data/injuries.csv
- data/weather.csv
- outputs/odds_game.csv (preferred) or outputs/game_lines.csv (fallback)
- (optional) data/team_form_weekly.csv (for opponent week-specific env)

Output:
- data/metrics_ready.csv
"""

import argparse
import os
import re
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utils.df_keys import coerce_merge_keys

from scripts._opponent_map import attach_opponent
from scripts.make_player_form import canonicalize_name
from scripts.utils.name_clean import canonical_key


def _log_merge_counts(
    stage: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    join_cols: list[str] | tuple[str, ...],
    how: str,
    merged: pd.DataFrame,
    indicator_col: str = "_merge",
) -> None:
    """Emit join diagnostics and persist left-only keys for debugging."""

    join_cols_list = [col for col in join_cols if col in merged.columns]
    left_rows, right_rows, merged_rows = len(left), len(right), len(merged)
    print(f"[make_metrics] {stage}: left={left_rows} right={right_rows} → merged={merged_rows} (how={how})")

    if indicator_col in merged.columns:
        left_only_mask = merged[indicator_col].eq("left_only")
        left_only = int(left_only_mask.sum())
        if left_only > 0 and join_cols_list:
            debug_dir = Path("data") / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            unmatched = merged.loc[left_only_mask, join_cols_list].drop_duplicates()
            debug_path = debug_dir / f"left_join_unmatched__{stage}.csv"
            unmatched.to_csv(debug_path, index=False)
            print(
                f"[make_metrics] WARN: {left_only} rows missing matches at {stage}; keys saved → {debug_path}"
            )



def _inj_load_lines_preferring_odds():
    for p in ["outputs/odds_game.csv", "outputs/game_lines.csv"]:
        if os.path.exists(p):
            try:
                gl = pd.read_csv(p)
                gl.columns = [c.lower() for c in gl.columns]
                for c in ["home_team","away_team"]:
                    if c in gl.columns:
                        gl[c] = _inj_normalize_team(gl[c])
                return gl
            except Exception:
                pass
    return pd.DataFrame()
# --- injected helper utilities (surgical, idempotent) ---
import re as _re_inj

def _inj_normalize_player(s):
    suf = _re_inj.compile(r"\s+(JR|SR|II|III|IV|V)\.?$", flags=_re_inj.I)
    return (s.astype(str)
              .str.replace(r"\.", "", regex=True)
              .str.replace(suf, "", regex=True)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip())

def _inj_normalize_team(s):
    aliases = {
        "WSH":"WAS","WDC":"WAS","JAC":"JAX","ARZ":"ARI","AZ":"ARI","LA":"LAR",
        "LVR":"LV","OAK":"LV","SFO":"SF","TAM":"TB","GBP":"GB","KAN":"KC",
        "NOS":"NO","SD":"LAC","CLV":"CLE"
    }
    return s.astype(str).str.upper().str.strip().replace(aliases)

def _inj_player_key(s):
    return s.fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]","",regex=True)


def _canon_df(df: pd.DataFrame, player_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if player_col not in df.columns:
        df["player_clean_key"] = ""
        return df
    df["player_clean_key"] = df[player_col].astype(str).map(canonical_key)
    return df


def _ensure_player_clean_key(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    for col in candidates:
        if col in df.columns:
            df = _canon_df(df, col)
            if col != "player_clean_key":
                df.rename(columns={col: "player_clean_key"}, inplace=True)
            return df
    df["player_clean_key"] = ""
    return df



def merge_opponent_map(base_df: pd.DataFrame) -> pd.DataFrame:
    """Merge opponent abbreviations from opponent_map_from_props.csv."""
    path = DATA_PATH / "opponent_map_from_props.csv"
    base = base_df.copy()

    if not path.exists():
        print("[make_metrics] WARN: opponent_map_from_props.csv missing – proceeding without opponent join")
        if "opponent_abbr" not in base.columns:
            base["opponent_abbr"] = pd.NA
        return base

    try:
        opp = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print("[make_metrics] WARN: opponent_map_from_props.csv empty – proceeding without opponent join")
        if "opponent_abbr" not in base.columns:
            base["opponent_abbr"] = pd.NA
        return base

    if opp.empty:
        print("[make_metrics] WARN: opponent_map_from_props.csv empty – proceeding without opponent join")
        if "opponent_abbr" not in base.columns:
            base["opponent_abbr"] = pd.NA
        return base

    opp = opp.copy()
    for col in ("season", "week"):
        if col in opp.columns:
            opp[col] = pd.to_numeric(opp[col], errors="coerce").astype("Int64")
    if "event_id" in opp.columns:
        opp["event_id"] = opp["event_id"].astype("string")

    if "player_clean_key" not in opp.columns:
        source_col = next((c for c in ("player", "player_name_raw", "player_name") if c in opp.columns), None)
        if source_col:
            opp["player_clean_key"] = opp[source_col].astype("string").map(canonical_key)
        else:
            opp["player_clean_key"] = pd.Series(pd.NA, index=opp.index, dtype="string")
    else:
        opp["player_clean_key"] = opp["player_clean_key"].astype("string")

    if "opponent_abbr" in opp.columns:
        opp["opponent_abbr"] = opp["opponent_abbr"].astype("string")
    else:
        opp["opponent_abbr"] = pd.Series(pd.NA, index=opp.index, dtype="string")

    for col in ("season", "week"):
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce").astype("Int64")
    if "event_id" in base.columns:
        base["event_id"] = base["event_id"].astype("string")

    if "player_clean_key" not in base.columns:
        name_col = next(
            (
                c
                for c in base.columns
                if c.lower() in {"player", "player_name", "name", "display_name", "player_key"}
            ),
            None,
        )
        if name_col:
            base["player_clean_key"] = base[name_col].astype("string").map(canonical_key)
        else:
            base["player_clean_key"] = pd.Series(pd.NA, index=base.index, dtype="string")
    else:
        base["player_clean_key"] = base["player_clean_key"].astype("string")

    if "opponent_abbr" not in base.columns:
        base["opponent_abbr"] = pd.Series(pd.NA, index=base.index, dtype="string")
    else:
        base["opponent_abbr"] = base["opponent_abbr"].astype("string")

    join_cols = [
        col
        for col in ("season", "week", "event_id", "player_clean_key")
        if col in base.columns and col in opp.columns
    ]
    if not join_cols:
        print("[make_metrics] WARN: no shared columns to join opponent map; skipping")
        return base

    opp_subset_cols = join_cols + [c for c in ("team_abbr", "opponent_abbr", "team", "opponent") if c in opp.columns]
    opp_subset = opp.loc[:, opp_subset_cols].drop_duplicates(subset=join_cols, keep="last")

    left_merge = base.copy()
    right_merge = opp_subset.copy()
    numeric_cols = [c for c in ("season", "week") if c in join_cols]
    text_cols = [c for c in join_cols if c not in numeric_cols]
    if numeric_cols:
        left_merge = coerce_merge_keys(left_merge, numeric_cols, as_str=False)
        right_merge = coerce_merge_keys(right_merge, numeric_cols, as_str=False)
    if text_cols:
        left_merge = coerce_merge_keys(left_merge, text_cols, as_str=True)
        right_merge = coerce_merge_keys(right_merge, text_cols, as_str=True)

    indicator_name = "_merge_opponent"
    merged = left_merge.merge(
        right_merge,
        how="left",
        on=join_cols,
        suffixes=("", "_opp"),
        indicator=indicator_name,
    )
    _log_merge_counts(
        stage="join_opponent_map",
        left=left_merge,
        right=right_merge,
        join_cols=join_cols,
        how="left",
        merged=merged,
        indicator_col=indicator_name,
    )

    joined_rows = int((merged[indicator_name] == "both").sum())
    merged.drop(columns=[indicator_name], inplace=True, errors="ignore")

    for col in ("team", "team_abbr", "opponent", "opponent_abbr"):
        opp_col = f"{col}_opp"
        if opp_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].combine_first(merged[opp_col])
            else:
                merged[col] = merged[opp_col]
            merged.drop(columns=[opp_col], inplace=True)

    missing_mask = _missing_mask(merged.get("team_abbr"), merged.index) | _missing_mask(
        merged.get("opponent_abbr"), merged.index
    )
    missing_count = int(missing_mask.sum())

    if missing_count:
        audit = merged.loc[missing_mask].copy()
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        audit.to_csv(DATA_PATH / "metrics_missing_core.csv", index=False)
        print(
            f"[make_metrics] WARN: opponent unresolved for {missing_count} rows -> data/metrics_missing_core.csv"
        )

    print(f"[make_metrics] opponent map joined rows: {joined_rows}")
    print(f"[make_metrics] missing team/opponent after merge: {missing_count}")

    opp_counts = (
        merged.groupby("team", dropna=False)
        .size()
        .reset_index(name="rows")
    )
    opp_counts.to_csv("data/_debug_team_rows_after_opponent_join.csv", index=False)

    return merged
DATA_DIR = "data"
DATA_PATH = Path(DATA_DIR)
METRICS_OUT_PATH = DATA_PATH / "metrics_ready.csv"
TEAM_FORM_PATH = DATA_PATH / "team_form.csv"
PLAYER_FORM_PATH = DATA_PATH / "player_form.csv"
PLAYER_CONS_PATH = DATA_PATH / "player_form_consensus.csv"
QB_SCRAMBLE_PATH = DATA_PATH / "qb_scramble_rates.csv"
QB_DESIGNED_PATH = DATA_PATH / "qb_designed_runs.csv"
QB_MOBILITY_PATH = DATA_PATH / "qb_mobility.csv"
QB_RUN_METRICS_PATH = DATA_PATH / "qb_run_metrics.csv"
WEATHER_PATH = DATA_PATH / "weather.csv"
WEATHER_WEEK_PATH = DATA_PATH / "weather_week.csv"


PLAYER_FORM_CONSENSUS_OPPONENT = "ALL"


# --- helpers for initials+last fallback matching ---
import re as _re_nm2
def _nm_initial_last(name: str):
    if not isinstance(name, str):
        return ("","")
    n = name.replace(".", " ").strip()
    if n and " " not in n:
        caps = [i for i,ch in enumerate(n) if ch.isupper()]
        if len(caps) >= 2:
            return (n[caps[0]].upper(), n[caps[1]:].upper())
    toks = _re_nm2.split(r"\s+", n)
    if not toks or not toks[0]:
        return ("","")
    return (toks[0][0].upper(), toks[-1].upper())

def _nm_add_fallback_keys_df(df, player_col="player"):
    if player_col not in df.columns:
        df["first_initial"] = ""; df["last_name_u"] = ""
        return df
    fi, ln = [], []
    for nm in df[player_col].astype(str):
        f,l = _nm_initial_last(nm); fi.append(f); ln.append(l)
    df["first_initial"] = fi; df["last_name_u"] = ln
    return df
def _tm_norm(s):
    return _normalize_team_names(s.astype(str).str.upper().str.strip())

# ----------------------------
# Utilities
# ----------------------------

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_div(n, d):
    n = n.astype(float); d = d.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d == 0, np.nan, n / d)

def _drop_dupe_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    seen: set[str] = set()
    keep: list[str] = []
    for col in df.columns:
        if col not in seen:
            keep.append(col)
            seen.add(col)
    return df.loc[:, keep]


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    return _drop_dupe_cols(df)


def _missing_mask(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    mask = series.isna()
    if series.dtype == object:
        mask = mask | series.astype(str).str.strip().eq("")
    return mask

def _normalize_team_names(s: pd.Series) -> pd.Series:
    """Map common sportsbook aliases to nflverse team codes (best effort)."""
    if s is None:
        return s
    original_null_mask = s.isna()
    norm = s.astype(str).str.upper().str.strip()
    aliases = {
        # books ↔ nflverse
        "WSH": "WAS", "WDC": "WAS",
        "JAX": "JAX", "JAC": "JAX",
        "ARZ": "ARI", "AZ": "ARI",
        "LA":  "LAR", "STL": "LAR",
        "LVR": "LV",  "OAK": "LV",
        "SFO": "SF",
        "TAM": "TB",
        "GBP": "GB",
        "KAN": "KC",
        "NOS": "NO", "N.O.": "NO",
        "SD":  "LAC",
        # occasional typos seen in feeds
        "CLV": "CLE",
    }
    norm = norm.replace(aliases)

    # add full-name -> code mapping (fix odds_game full names)
    full_to_code = {
        "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
        "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
        "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
        "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
        "LAS VEGAS RAIDERS":"LV","LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LAR","MIAMI DOLPHINS":"MIA",
        "MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG",
        "NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT","SAN FRANCISCO 49ERS":"SF",
        "SEATTLE SEAHAWKS":"SEA","TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS"
    }
    norm = norm.replace(full_to_code)

    empty_sentinels = {"", "NAN", "NONE", "NULL", "NA", "N/A"}
    norm = norm.mask(norm.isin(empty_sentinels))
    norm = norm.mask(original_null_mask)
    return norm

_SUFFIX_RE = re.compile(r"\s+(JR|SR|II|III|IV|V)\.?$", flags=re.IGNORECASE)

def _normalize_player_name(x: pd.Series | str) -> pd.Series | str:
    """Remove punctuation/suffixes, collapse spaces."""
    def _one(name: str) -> str:
        if not isinstance(name, str):
            return name
        n = name.strip()
        n = n.replace(".", "")
        n = _SUFFIX_RE.sub("", n)
        n = re.sub(r"\s+", " ", n)
        return n
    if isinstance(x, pd.Series):
        return x.map(_one)
    return _one(x)

def _player_key_series(s: pd.Series) -> pd.Series:
    """Stable key for joins; lowercase, alnum only."""
    return s.fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def _player_form_consensus_mask(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    norm = series.fillna("").astype(str).str.upper().str.strip()
    return norm.eq("") | norm.eq(PLAYER_FORM_CONSENSUS_OPPONENT)

# ----------------------------
def load_roles() -> pd.DataFrame:
    """
    Load roles from roles_ourlads.csv or roles.csv to backfill team/role/position.
    """
    paths = [os.path.join(DATA_DIR, 'roles_ourlads.csv'), os.path.join(DATA_DIR, 'roles.csv')]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df.columns = [c.lower() for c in df.columns]
                if df.empty:
                    continue
                if 'team' in df.columns:
                    df['team'] = _normalize_team_names(df['team'])
                for pcol in ['player','player_name','name']:
                    if pcol in df.columns:
                        df = df.rename(columns={pcol:'player'})
                        break
                if 'player' not in df.columns:
                    continue
                df['player'] = _normalize_player_name(df['player'])
                df['player_key'] = _player_key_series(df['player'])
                if 'position' in df.columns:
                    df = df.rename(columns={'position': 'ourlads_position'})
                keep = [
                    c
                    for c in ['player_key', 'player', 'team', 'role', 'ourlads_position']
                    if c in df.columns
                ]
                return df[keep].drop_duplicates()
            except Exception:
                continue
    return pd.DataFrame()
# Core loaders
# ----------------------------

def load_props() -> pd.DataFrame:
    df = _read_csv(os.path.join("outputs", "props_raw.csv"))
    if df.empty:
        return df

    rename_map = {
        "eventid": "event_id",
        "player_name": "player",
        "name": "player",
        "market_key": "market",
        "key": "market",
        "odds_over": "over_odds",
        "odds_under": "under_odds",
        "participant": "team",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    # Ensure minimal schema
    for c in ["event_id","player","team","market","line","over_odds","under_odds"]:
        if c not in df.columns:
            df[c] = np.nan

    # Normalize strings
    if "team" in df.columns:
        df["team"] = _normalize_team_names(df["team"])
        df["team_abbr"] = df.get("team_abbr", df["team"])
    else:
        df["team_abbr"] = ""
    if "player" in df.columns:
        df["player"] = _normalize_player_name(df["player"])
        df["player_canonical"] = df["player"].apply(canonicalize_name)
    else:
        df["player_canonical"] = ""

    df = _canon_df(df, "player")

    if "commence_time" in df.columns:
        dt = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
        df["slate_date"] = dt.dt.date.astype(str)
        df.loc[dt.isna(), "slate_date"] = os.getenv("SLATE_DATE", "").strip()
    else:
        df["slate_date"] = os.getenv("SLATE_DATE", "").strip()

    # Deduplicate
    keep = [
        "event_id",
        "player",
        "team",
        "team_abbr",
        "player_canonical",
        "market",
        "line",
        "over_odds",
        "under_odds",
        "slate_date",
    ]
    df = df[keep + [c for c in df.columns if c not in keep]].drop_duplicates()

    # Pivot side rows into over/under odds if present
    if "side" in df.columns and "price_american" in df.columns:
        keycols = [c for c in ["event_id","player","team","market","line"] if c in df.columns]
        if keycols:
            tmp = df[keycols + ["side","price_american"]].copy()
            tmp["side"] = tmp["side"].astype(str).str.upper().str.strip()
            pvt = tmp.pivot_table(index=keycols, columns="side", values="price_american", aggfunc="first").reset_index()
            pvt.columns = [("over_odds" if c=="OVER" else "under_odds" if c=="UNDER" else c) for c in pvt.columns]
            for c in ["over_odds","under_odds"]:
                if c not in pvt.columns:
                    pvt[c] = np.nan
            df = df.merge(pvt, on=keycols, how="left", suffixes=("","_pvt"))
            for c in ["over_odds","under_odds"]:
                c_p = c + "_pvt"
                if c_p in df.columns:
                    df[c] = df[c].combine_first(df[c_p])
                    df.drop(columns=[c_p], inplace=True)

    enriched = _read_csv(os.path.join(DATA_DIR, "props_enriched.csv"))
    if not enriched.empty:
        if "player_canonical" not in df.columns:
            df["player_canonical"] = df["player"].apply(canonicalize_name)
        if "player_canonical" not in enriched.columns:
            if "player_name_raw" in enriched.columns:
                enriched["player_canonical"] = enriched["player_name_raw"].apply(canonicalize_name)
            else:
                enriched["player_canonical"] = enriched.get("player", "")
        merge_keys = [c for c in ["event_id", "player_canonical"] if c in df.columns and c in enriched.columns]
        if not merge_keys:
            merge_keys = ["player_canonical"]
        keep_cols = [c for c in ["player_canonical", "event_id", "player_team_abbr", "opponent_team_abbr", "kickoff_ts", "home_team_abbr", "away_team_abbr"] if c in enriched.columns]
        subset = enriched[keep_cols].drop_duplicates(subset=merge_keys, keep="first")
        df = df.merge(subset, on=merge_keys, how="left", suffixes=("", "_props_enriched"))
        if "player_team_abbr" in df.columns:
            df["team_abbr"] = df["team_abbr"].combine_first(df["player_team_abbr"])
        if "opponent_team_abbr" in df.columns:
            df["opponent_abbr"] = df.get("opponent_abbr", pd.Series(dtype=object)).combine_first(df["opponent_team_abbr"])
            if "opp_abbr" in df.columns:
                df["opp_abbr"] = df["opp_abbr"].combine_first(df["opponent_team_abbr"])
            else:
                df["opp_abbr"] = df["opponent_abbr"]
        if "kickoff_ts" in df.columns and "kickoff_ts_props_enriched" in df.columns:
            df["kickoff_ts"] = df["kickoff_ts"].combine_first(df["kickoff_ts_props_enriched"])
        elif "kickoff_ts_props_enriched" in df.columns:
            df["kickoff_ts"] = df["kickoff_ts_props_enriched"]
        if "event_id_props_enriched" in df.columns:
            df["event_id"] = df["event_id"].combine_first(df["event_id_props_enriched"])
        for helper in ["player_team_abbr", "opponent_team_abbr", "kickoff_ts_props_enriched", "event_id_props_enriched", "home_team_abbr", "away_team_abbr"]:
            if helper in df.columns:
                df.drop(columns=[helper], inplace=True)

    return df

def load_team_form() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "team_form.csv"))
    if df.empty:
        return df
    if "team" in df.columns:
        df["team"] = _normalize_team_names(df["team"])
    return df

def load_player_form() -> pd.DataFrame:
    # prefer consensus if present
    consensus_path = os.path.join(DATA_DIR, "player_form_consensus.csv")
    base_path = os.path.join(DATA_DIR, "player_form.csv")
    df = pd.DataFrame()
    if os.path.exists(consensus_path):
        try:
            df = pd.read_csv(consensus_path)
            df.columns = [c.lower() for c in df.columns]
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        # Fallback to the base player_form extract when consensus has no rows
        df = _read_csv(base_path)
    if df.empty:
        return df
    if "team" in df.columns:
        df["team"] = _normalize_team_names(df["team"])
    if "team_abbr" in df.columns:
        df["team_abbr"] = _normalize_team_names(df["team_abbr"])
    # normalize player column
    for pcol in ["player","player_name","name"]:
        if pcol in df.columns:
            df = df.rename(columns={pcol:"player"})
            break
    if "player" not in df.columns:
        df["player"] = np.nan
    df["player"] = _normalize_player_name(df["player"])
    if "player_canonical" in df.columns:
        df["player_canonical"] = df["player_canonical"].fillna(df["player"]).apply(
            canonicalize_name
        )
    else:
        df["player_canonical"] = df["player"].apply(canonicalize_name)

    df = _canon_df(df, "player")

    if "opponent_abbr" in df.columns:
        df["opponent_abbr"] = _normalize_team_names(df["opponent_abbr"])
    elif "opponent" in df.columns:
        df["opponent_abbr"] = _normalize_team_names(df["opponent"])
    else:
        df["opponent_abbr"] = ""
    if "opp_abbr" in df.columns:
        df["opp_abbr"] = _normalize_team_names(df["opp_abbr"])
        df["opponent_abbr"] = df["opponent_abbr"].combine_first(df["opp_abbr"])
    else:
        df["opp_abbr"] = df["opponent_abbr"]
    if "kickoff_ts" not in df.columns:
        df["kickoff_ts"] = pd.NA

    # --- alias column names for compatibility with make_player_form ---
    if "target_share" not in df.columns and "tgt_share" in df.columns:
        df["target_share"] = df["tgt_share"]
    if "yprr_proxy" not in df.columns and "yprr" in df.columns:
        df["yprr_proxy"] = df["yprr"]
    if "rz_carry_share" not in df.columns and "rz_rush_share" in df.columns:
        df["rz_carry_share"] = df["rz_rush_share"]

    return df

def load_coverage() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "coverage.csv"))
    if df.empty:
        return df
    if {"defense_team","tag"}.issubset(df.columns):
        df["defense_team"] = _normalize_team_names(df["defense_team"])
        pivot = pd.crosstab(df["defense_team"], df["tag"]).reset_index()
        pivot.columns = [str(c).lower() for c in pivot.columns]
        pivot = pivot.rename(columns={
            "top_shadow": "coverage_top_shadow",
            "heavy_man":  "coverage_heavy_man",
            "heavy_zone": "coverage_heavy_zone",
        })
        for c in pivot.columns:
            if c == "defense_team": continue
            pivot[c] = (pivot[c] > 0).astype(int)
        return pivot
    return pd.DataFrame()

def load_cb_assignments() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "cb_assignments.csv"))
    if df.empty:
        return df
    for c in ["defense_team","receiver"]:
        if c in df.columns and df[c].dtype == object:
            if c == "defense_team":
                df[c] = _normalize_team_names(df[c])
            else:
                df[c] = _normalize_player_name(df[c].astype(str))
    if "penalty" not in df.columns:
        if "quality" in df.columns:
            m = {"elite": 0.12, "good": 0.08, "avg": 0.04, "below_avg": 0.02}
            df["penalty"] = df["quality"].map(m).fillna(0.04)
        else:
            df["penalty"] = 0.06
    df = df[["defense_team","receiver","penalty"]].rename(
        columns={"receiver":"player","defense_team":"opp_def_team"}
    )
    df["player"] = _normalize_player_name(df["player"])
    df["player_canonical"] = df["player"].apply(canonicalize_name)
    df = _canon_df(df, "player")
    df["opp_def_team"] = _normalize_team_names(df["opp_def_team"])
    df["opponent_abbr"] = df["opp_def_team"]
    return df

def load_injuries() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "injuries.csv"))
    if df.empty:
        return df
    if "team" in df.columns:
        df["team"] = _normalize_team_names(df["team"])
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.title()
    df["player"] = _normalize_player_name(df.get("player", pd.Series([], dtype=object)))
    df["player_canonical"] = df["player"].apply(canonicalize_name)
    df = _canon_df(df, "player")
    df["team_abbr"] = _normalize_team_names(df.get("team", pd.Series([], dtype=object)))
    return df[["player","player_canonical","team","team_abbr","status"]].drop_duplicates()

def load_weather() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "weather.csv")
    if not os.path.exists(path):
        print("[make_metrics] WARN: weather.csv not found; continuing without weather")
        return pd.DataFrame()

    df = _read_csv(path)
    if df.empty:
        print("[make_metrics] WARN: weather.csv empty; continuing without weather")
    return df

def load_qb_run_metrics() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "qb_run_metrics.csv")
    df = _read_csv(path)
    if df.empty:
        # Fallback: derive from individual components when combined file missing
        scramble = _read_csv(os.path.join(DATA_DIR, "qb_scramble_rates.csv"))
        designed = _read_csv(os.path.join(DATA_DIR, "qb_designed_runs.csv"))
        if not scramble.empty or not designed.empty:
            df = scramble.merge(designed, on=["player", "week"], how="outer")
    if df.empty:
        return df
    # Normalize identifiers
    if "player" in df.columns:
        df["player"] = _normalize_player_name(df["player"])
        df["player_canonical"] = df["player"].apply(canonicalize_name)
    else:
        df["player"] = ""
        df["player_canonical"] = ""
    df = _canon_df(df, "player")
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    return df

# Preferred game-lines: Odds API file, then fallback to schedule
def _read_csv_any(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df.columns = [c.lower() for c in df.columns]
                if not df.empty:
                    return df.assign(_source=p)
            except Exception:
                pass
    return pd.DataFrame()

def load_game_lines_preferring_odds() -> pd.DataFrame:
    og = _read_csv_any(["outputs/odds_game.csv"])
    if not og.empty and {"home_team","away_team"}.issubset(og.columns):
        gl = og.copy()
    else:
        gl = _read_csv_any(["outputs/game_lines.csv"])

    if gl.empty:
        return gl

    # Normalize teams
    for col in ["home_team","away_team"]:
        if col in gl.columns:
            gl[col] = _normalize_team_names(gl[col])

    # Ensure event_id
    if "event_id" not in gl.columns:
        gl["event_id"] = (gl.get("season", np.nan).astype(str) + "_" +
                          gl.get("week", np.nan).astype(str) + "_" +
                          gl.get("home_team", "").astype(str) + "_" +
                          gl.get("away_team", "").astype(str))

    # Derive week from commence_time if missing
    if ("week" not in gl.columns or gl["week"].isna().all()) and "commence_time" in gl.columns:
        gl["week"] = pd.to_datetime(gl["commence_time"], errors="coerce", utc=True).dt.isocalendar().week.astype("Int64")

    keep = [c for c in ["event_id","home_team","away_team","week","season","commence_time","home_wp","away_wp"] if c in gl.columns]
    return gl[keep].drop_duplicates()

# ----------------------------
# NEW: schedule fallback (surgical)
# ----------------------------

def _load_schedule_long(season: int) -> pd.DataFrame:
    """
    Returns (team, opponent, week) long form for the given season.
    Only used as a fallback when event_id-based opponent inference is missing.
    """
    try:
        try:
            import nflreadpy as nflv  # preferred
            sch = nflv.load_schedules(seasons=[season])
        except Exception:
            import nfl_data_py as nflv  # fallback
            sch = nflv.import_schedules([season])  # type: ignore
    except Exception:
        return pd.DataFrame(columns=["team","opponent","week"])

    df = pd.DataFrame(sch)
    if df.empty:
        return pd.DataFrame(columns=["team","opponent","week"])

    df.columns = [c.lower() for c in df.columns]
    # normalize team codes
    for col in ["home_team","away_team"]:
        if col in df.columns:
            df[col] = _normalize_team_names(df[col])

    if {"home_team","away_team","week"}.issubset(df.columns):
        h = df[["home_team","away_team","week"]].rename(columns={"home_team":"team","away_team":"opponent"})
        a = df[["away_team","home_team","week"]].rename(columns={"away_team":"team","home_team":"opponent"})
        long = pd.concat([h, a], ignore_index=True).dropna()
        long["team"] = _normalize_team_names(long["team"])
        long["opponent"] = _normalize_team_names(long["opponent"])
        # dedupe in case feeds have duplicates
        return long.drop_duplicates()
    return pd.DataFrame(columns=["team","opponent","week"])

def _correct_wr_roles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-rank WRs within each team (and week if present) by route_rate (primary) then target_share.
    Assign WR1/WR2/WR3 accordingly. Only overwrite if role is missing or disagrees with rank.
    """
    if df.empty:
        return df
    keys = []
    if 'team' in df.columns: keys.append('team')
    if 'week' in df.columns: keys.append('week')
    if not keys:
        return df
    poscol = 'position' if 'position' in df.columns else None
    if poscol is None:
        return df
    wr_mask = df[poscol].fillna('').astype(str).str.upper().str.startswith('WR')
    if not wr_mask.any():
        return df
    rr = df.get('route_rate')
    ts = df.get('target_share')
    if rr is None:
        df['route_rate'] = 0.0
        rr = df['route_rate']
    if ts is None:
        df['target_share'] = 0.0
        ts = df['target_share']
    df['_wr_sort_rr'] = rr.fillna(0).astype(float)
    df['_wr_sort_ts'] = ts.fillna(0).astype(float)
    def assign_role(g):
        g = g.sort_values(by=['_wr_sort_rr','_wr_sort_ts'], ascending=[False, False]).copy()
        new_roles = []
        for i, idx in enumerate(g.index, start=1):
            tag = f'WR{i}' if i <= 5 else 'WR5+'
            new_roles.append(tag)
        g['_wr_new_role'] = new_roles
        return g
    df_wr = df[wr_mask].copy()
    df_wr = df_wr.groupby(keys, dropna=False, group_keys=False).apply(assign_role)
    df = df.merge(df_wr[['_wr_new_role']], left_index=True, right_index=True, how='left')
    if 'role' not in df.columns:
        df['role'] = np.nan
    need_fix = df['_wr_new_role'].notna() & (df['role'].isna() | (~df['role'].astype(str).str.upper().eq(df['_wr_new_role'])))
    df.loc[need_fix, 'role'] = df.loc[need_fix, '_wr_new_role']
    df.drop(columns=['_wr_new_role','_wr_sort_rr','_wr_sort_ts'], inplace=True, errors='ignore')
    return df
# ----------------------------
# Assembler
# ----------------------------

def build_metrics(season: int) -> pd.DataFrame:
    props = load_props()
    if props.empty:
        empty = pd.DataFrame(columns=[
            "event_id","player","team","opponent","market","line","over_odds","under_odds",
            "target_share","rush_share","route_rate","yprr_proxy","ypt","ypc","ypa_prior","rz_tgt_share","rz_carry_share","position","role",
            "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
            "def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp",
            "coverage_top_shadow_opp","coverage_heavy_man_opp","coverage_heavy_zone_opp",
            "cb_penalty","injury_status","wind_mph","temp_f","precip","team_wp","season",
            "week","opp_plays_wk","opp_pace_wk","opp_proe_wk"
        ])
        empty.attrs["row_counts"] = {"props_raw": 0, "player_form_consensus": 0, "team_form": 0, "joined": 0}
        empty.attrs["upstream_empty"] = {"props_raw": True}
        return empty

    pf  = load_player_form()
    pf = _ensure_player_clean_key(pf, ["player_clean_key", "player", "player_key"])
    tf  = load_team_form()
    cov = load_coverage()
    cba = load_cb_assignments()
    inj = load_injuries()
    qb_run = load_qb_run_metrics()
    wx  = load_weather()
    gl  = load_game_lines_preferring_odds()
    print(f"[make_metrics] player_form rows pre-merge: {len(pf)}")
    upstream_empty = {
        "props_raw": False,
        "player_form": pf.empty,
        "team_form": tf.empty,
        "coverage": cov.empty,
        "cb_assignments": cba.empty,
        "injuries": inj.empty,
        "qb_run_metrics": qb_run.empty,
        "weather": wx.empty,
        "odds_game": gl.empty,
    }
    pf = attach_opponent(pf, team_col="team", coverage_path="data/coverage_cb.csv")
    tf = attach_opponent(tf, team_col="team", coverage_path="data/coverage_cb.csv")

    # Week inference for props (if missing)
    if "week" not in props.columns or props["week"].isna().all():
        if "commence_time" in props.columns:
            props["week"] = pd.to_datetime(props["commence_time"], errors="coerce", utc=True)\
                                .dt.isocalendar().week.astype("Int64")
        elif not gl.empty and {"event_id","commence_time"}.issubset(gl.columns) and "event_id" in props.columns:
            wk_src = gl[["event_id","commence_time"]].dropna().copy()
            wk_src["week"] = pd.to_datetime(wk_src["commence_time"], errors="coerce", utc=True)\
                                .dt.isocalendar().week.astype("Int64")
            props = props.merge(wk_src[["event_id","week"]], on="event_id", how="left", suffixes=("", "_gl"))
            if "week_gl" in props.columns:
                props["week"] = props["week"].combine_first(props["week_gl"])
                props.drop(columns=["week_gl"], inplace=True, errors="ignore")
    if "week" not in props.columns:
        props["week"] = pd.Series(pd.array([], dtype="Int64"))

    # Stable player keys for joins
    props["player_key"] = _player_key_series(props.get("player", pd.Series(dtype=object)))
    props = _ensure_player_clean_key(props, ["player_clean_key", "player", "player_key"])
    pf_consensus = pd.DataFrame()
    pf_by_opponent = pd.DataFrame()
    if not pf.empty:
        pf["player_key"] = _player_key_series(pf.get("player", pd.Series(dtype=object)))
        if "opponent" in pf.columns:
            consensus_mask = _player_form_consensus_mask(pf["opponent"])
        else:
            consensus_mask = pd.Series(True, index=pf.index)
        pf_consensus = pf.loc[consensus_mask].copy()
        pf_by_opponent = pf.loc[~consensus_mask].copy()
        if pf_consensus.empty:
            pf_consensus = pf.copy()
        upstream_empty["player_form_consensus"] = pf_consensus.empty
    else:
        pf_consensus = pf.copy()
        pf_by_opponent = pd.DataFrame(columns=pf.columns)
        upstream_empty["player_form_consensus"] = True

    # Backfill team from player_form if all missing
    team_fill_source = pf_consensus if not pf_consensus.empty else pf
    if (
        "team" in props.columns
        and props["team"].isna().all()
        and not team_fill_source.empty
        and {"player", "team"}.issubset(team_fill_source.columns)
    ):
        props = props.merge(team_fill_source[["player","team"]].drop_duplicates(), on="player", how="left", suffixes=("","_pf"))
        props["team"] = props["team"].combine_first(props.get("team_pf"))
        if "team_pf" in props.columns:
            props = props.drop(columns=["team_pf"])

    # a few sources don't carry event_id; keep NaN and we still keep the rows
    if "event_id" not in props.columns:
        props["event_id"] = np.nan

    # Backfill team/role/position from roles files if still missing
    try:
        roles_df = load_roles()
        if not roles_df.empty:
            props = props.merge(roles_df, on='player_key', how='left', suffixes=('', '_roles'))
            if 'team' in props.columns and 'team_roles' in props.columns:
                props['team'] = props['team'].combine_first(props['team_roles'])
                props.drop(columns=['team_roles'], inplace=True)
            if 'role' in props.columns and 'role_roles' in props.columns:
                props['role'] = props['role'].combine_first(props['role_roles'])
                props.drop(columns=['role_roles'], inplace=True)
            elif 'role_roles' in props.columns:
                props.rename(columns={'role_roles':'role'}, inplace=True)
            if 'ourlads_position' in props.columns and 'ourlads_position_roles' in props.columns:
                props['ourlads_position'] = props['ourlads_position'].combine_first(
                    props['ourlads_position_roles']
                )
                props.drop(columns=['ourlads_position_roles'], inplace=True)
            if 'ourlads_position' in props.columns:
                if 'position' in props.columns:
                    props['position'] = props['position'].combine_first(props['ourlads_position'])
                else:
                    props['position'] = props['ourlads_position']
            props = _drop_duplicate_columns(props)
    except Exception as _e:
        print('[make_metrics] roles fill skipped:', _e)

    # Fallback fill: match roles by (first_initial, last_name_u) when player_key failed
    try:
        props = _nm_add_fallback_keys_df(props, "player")
        try:
            roles_df = roles_df.copy()
        except NameError:
            roles_df = load_roles()
        if not roles_df.empty:
            roles_df = _nm_add_fallback_keys_df(roles_df, "player")
            cols_key = ["first_initial","last_name_u"]
            fill_cols = [c for c in ["team", "role"] if c in roles_df.columns]
            if "ourlads_position" in roles_df.columns:
                fill_cols.append("ourlads_position")
            if "position" in roles_df.columns and "position" not in fill_cols:
                fill_cols.append("position")
            if fill_cols:
                fb = roles_df[cols_key + fill_cols].drop_duplicates()
                before_team_missing = props["team"].isna().sum() if "team" in props.columns else None
                props = props.merge(fb, on=cols_key, how="left", suffixes=("","_fbroles"))
                if "team" in props.columns and "team_fbroles" in props.columns:
                    props["team"] = props["team"].combine_first(props["team_fbroles"])
                if "position" in props.columns and "position_fbroles" in props.columns:
                    props["position"] = props["position"].combine_first(props["position_fbroles"])
                if "role" in props.columns and "role_fbroles" in props.columns:
                    props["role"] = props["role"].combine_first(props["role_fbroles"])
                if "ourlads_position" in props.columns and "ourlads_position_fbroles" in props.columns:
                    props["ourlads_position"] = props["ourlads_position"].combine_first(
                        props["ourlads_position_fbroles"]
                    )
                props.drop(columns=[c for c in props.columns if c.endswith("_fbroles")], inplace=True, errors="ignore")
                props = _drop_duplicate_columns(props)
    except Exception as _e:
        print("[make_metrics] roles initials/last fallback skipped:", _e)
    # ---- Opponent mapping
    if not gl.empty:
        if "event_id" in props.columns and "event_id" in gl.columns:
            cols_have = [c for c in ["event_id","home_team","away_team","week","home_wp","away_wp"] if c in gl.columns]
            tmp = props.merge(gl[cols_have], on="event_id", how="left")
            opp = np.where(tmp.get("team").eq(tmp.get("home_team")), tmp.get("away_team"),
                   np.where(tmp.get("team").eq(tmp.get("away_team")), tmp.get("home_team"), np.nan))
            props["opponent"] = _normalize_team_names(pd.Series(opp, index=props.index))
            # carry week from gl if needed
            if "week" not in props.columns or props["week"].isna().all():
                props["week"] = tmp.get("week")
            # team win prob
            if {"home_wp","away_wp"}.issubset(tmp.columns):
                props["team_wp"] = np.where(
                    tmp.get("team").eq(tmp.get("home_team")),
                    tmp.get("home_wp"),
                    np.where(tmp.get("team").eq(tmp.get("away_team")), tmp.get("away_wp"), np.nan),
                )
            else:
                if "team_wp" not in props.columns:
                    props["team_wp"] = np.nan
        elif {"team","week"}.issubset(props.columns) and {"home_team","away_team","week"}.issubset(gl.columns):
            cols_have_w = [c for c in ["home_team","away_team","week","home_wp","away_wp"] if c in gl.columns]
            left = props.merge(gl[cols_have_w], on="week", how="left")
            opp = np.where(left.get("team").eq(left.get("home_team")), left.get("away_team"),
                   np.where(left.get("team").eq(left.get("away_team")), left.get("home_team"), np.nan))
            props["opponent"] = _normalize_team_names(pd.Series(opp, index=props.index))
            if {"home_wp","away_wp"}.issubset(left.columns):
                props["team_wp"] = np.where(
                    left.get("team").eq(left.get("home_team")),
                    left.get("home_wp"),
                    np.where(left.get("team").eq(left.get("away_team")), left.get("away_wp"), np.nan),
                )

    if "opponent" in props.columns:
        props["opponent_abbr"] = _normalize_team_names(props["opponent"])
    else:
        props["opponent_abbr"] = ""

    # Base: props + opponent map + player_form (non-destructive)
    base = props.copy()
    if "season" not in base.columns:
        base["season"] = season
    else:
        base["season"] = base["season"].fillna(season)
    base = _ensure_player_clean_key(base, ["player_clean_key", "player", "player_key"])
    base["player_key"] = base["player_clean_key"].astype(str)

    base = merge_opponent_map(base)

    if not pf_consensus.empty:
        keep_pf = [
            "player",
            "player_canonical",
            "player_clean_key",
            "team",
            "team_abbr",
            "opponent",
            "opponent_abbr",
            "target_share",
            "rush_share",
            "route_rate",
            "yprr_proxy",
            "ypt",
            "ypc",
            "rz_tgt_share",
            "rz_carry_share",
            "position",
            "role",
            "season",
            "week",
            "ypa_prior",
        ]
        keep_pf = [c for c in keep_pf if c in pf_consensus.columns]
        if keep_pf:
            pf_subset = pf_consensus[keep_pf].drop_duplicates(
                subset=[c for c in ["season", "week", "player_clean_key"] if c in keep_pf],
                keep="last",
            )
            join_cols = [
                c
                for c in [
                    "season",
                    "week",
                    "game_id",
                    "player_clean_key",
                    "posteam",
                    "opponent_abbr",
                ]
                if c in base.columns and c in pf_subset.columns
            ]
            if join_cols:
                pf_subset = pf_subset[[c for c in keep_pf if c in pf_subset.columns]]
                numeric_join = [c for c in ("season", "week") if c in join_cols]
                text_join = [c for c in join_cols if c not in numeric_join]
                left_merge = base.copy()
                right_merge = pf_subset.copy()
                if numeric_join:
                    left_merge = coerce_merge_keys(left_merge, numeric_join, as_str=False)
                    right_merge = coerce_merge_keys(right_merge, numeric_join, as_str=False)
                if text_join:
                    left_merge = coerce_merge_keys(left_merge, text_join, as_str=True)
                    right_merge = coerce_merge_keys(right_merge, text_join, as_str=True)
                print(
                    "[make_metrics] dtypes(left):",
                    {c: str(left_merge[c].dtype) for c in join_cols if c in left_merge.columns},
                )
                print(
                    "[make_metrics] dtypes(right):",
                    {c: str(right_merge[c].dtype) for c in join_cols if c in right_merge.columns},
                )
                left_keys = left_merge[join_cols].drop_duplicates() if join_cols else pd.DataFrame()
                merged = left_merge.merge(
                    right_merge,
                    on=join_cols,
                    how="left",
                    suffixes=("", "_pf"),
                )
                if merged.empty:
                    right_keys = right_merge[join_cols].drop_duplicates() if join_cols else pd.DataFrame()
                    dbg = left_keys.merge(right_keys, on=join_cols, how="left", indicator=True)
                    dbg = dbg[dbg["_merge"] == "left_only"]
                    Path("data/_debug").mkdir(parents=True, exist_ok=True)
                    dbg.to_csv(
                        "data/_debug/metrics_left_keys_without_pf_match.csv",
                        index=False,
                    )
                    raise RuntimeError(
                        "metrics_ready is empty after merge (debug keys written)"
                    )
                base = merged
        for col in ["team", "team_abbr", "opponent_abbr"]:
            col_pf = f"{col}_pf"
            if col_pf in base.columns:
                if col in base.columns:
                    base[col] = base[col].combine_first(base[col_pf])
                else:
                    base[col] = base[col_pf]
                base.drop(columns=[col_pf], inplace=True, errors="ignore")

    if "position_role" not in base.columns:
        base["position_role"] = pd.NA
    if "position" in base.columns:
        base["position_role"] = base["position_role"].combine_first(base["position"])
    if "position_pf" in base.columns:
        base["position_role"] = base["position_role"].combine_first(base["position_pf"])
        base.drop(columns=["position_pf"], inplace=True, errors="ignore")

    player_metrics = _drop_duplicate_columns(base)
    base = player_metrics

    collision_cols = [c for c in base.columns if c.endswith("_x") or c.endswith("_y")]
    if collision_cols:
        for col in collision_cols:
            root = col[:-2]
            if root in base.columns:
                base[root] = base[root].combine_first(base[col])
            else:
                base[root] = base[col]
        base.drop(columns=[c for c in collision_cols if c in base.columns], inplace=True, errors="ignore")

    if "team_abbr" in base.columns:
        base["team_abbr"] = _normalize_team_names(base["team_abbr"])
    if "opponent_abbr" in base.columns:
        base["opponent_abbr"] = _normalize_team_names(base["opponent_abbr"])
    # --- fallback usage merge on (team, position, first_initial, last_name_u) ---
    try:
        # ensure initials/last exist
        base = _nm_add_fallback_keys_df(base, "player")
        pf_fb = pf_consensus.copy()
        pf_fb = _nm_add_fallback_keys_df(pf_fb, "player")
        cols_key = [
            c
            for c in ["team_abbr", "position", "first_initial", "last_name_u"]
            if c in base.columns and c in pf_fb.columns
        ]
        fb_cols = [c for c in ["target_share","route_rate","rush_share","yprr_proxy","ypt","ypc","ypa_prior","rz_tgt_share","rz_carry_share","role","position"] if c in pf_fb.columns]
        if fb_cols and len(cols_key) >= 3:
            # choose best candidate per key (highest route_rate then target_share)
            sort_cols = [c for c in ["route_rate","target_share"] if c in pf_fb.columns]
            if sort_cols:
                pf_fb = pf_fb.sort_values(cols_key + sort_cols, ascending=[True,True,True,True] + [False]*len(sort_cols))
            pf_fb = pf_fb.drop_duplicates(subset=cols_key, keep="first")
            miss_mask = base.get("target_share").isna() if "target_share" in base.columns else base.get("route_rate").isna()
            if miss_mask is None:
                miss_mask = base.index == -1  # no-op
            if miss_mask.any():
                fill = base.loc[miss_mask, cols_key].merge(pf_fb[cols_key + fb_cols], on=cols_key, how="left")
                for c in fb_cols:
                    if c in base.columns and c in fill.columns:
                        base.loc[miss_mask, c] = base.loc[miss_mask, c].combine_first(fill[c])
    except Exception as _e:
        print("[make_metrics] fallback usage merge skipped:", _e)


    # carry week explicitly
    if "week" not in base.columns and "week" in props.columns:
        base["week"] = props["week"]

    # Fix WR roles by usage (route_rate then target_share)
    base = _correct_wr_roles(base)

    team_map_series = base.get("team_abbr") if "team_abbr" in base.columns else base.get("team")
    base["team_abbr"] = _normalize_team_names(team_map_series if team_map_series is not None else pd.Series([], dtype=object))
    if "team" in base.columns:
        base["team"] = base["team_abbr"].combine_first(_normalize_team_names(base["team"]))
    else:
        base["team"] = base["team_abbr"]

    if "opponent_abbr" in base.columns:
        base["opponent_abbr"] = _normalize_team_names(base["opponent_abbr"])
    elif "opponent" in base.columns:
        base["opponent_abbr"] = _normalize_team_names(base["opponent"])
    else:
        base["opponent_abbr"] = pd.NA
    if "opp_abbr" in base.columns:
        base["opp_abbr"] = _normalize_team_names(base["opp_abbr"])
        base["opponent_abbr"] = base["opponent_abbr"].combine_first(base["opp_abbr"])
    else:
        base["opp_abbr"] = base["opponent_abbr"]

    team_to_opp: dict[str, str] = {}
    teams_series = base.get("team_abbr", pd.Series(dtype=object))
    opps_series = base.get("opponent_abbr", pd.Series(dtype=object))
    for team_val, opp_val in zip(teams_series, opps_series):
        if not isinstance(team_val, str) or not team_val or not isinstance(opp_val, str) or not opp_val:
            continue
        team_key = team_val.upper().strip()
        opp_key = opp_val.upper().strip()
        if team_key and opp_key:
            team_to_opp.setdefault(team_key, opp_key)
            team_to_opp.setdefault(opp_key, team_key)

    missing_opp_mask = base["opponent_abbr"].isna() & base["team_abbr"].notna()
    if missing_opp_mask.any() and team_to_opp:
        base.loc[missing_opp_mask, "opponent_abbr"] = base.loc[missing_opp_mask, "team_abbr"].map(team_to_opp)

    base["opponent_abbr"] = _normalize_team_names(base["opponent_abbr"])
    if "opponent" in base.columns:
        base["opponent"] = base["opponent"].fillna(base["opponent_abbr"])
        base["opponent"] = _normalize_team_names(base["opponent"])
    else:
        base["opponent"] = base["opponent_abbr"]

    qb_feature_cols = [
        "scramble_rate",
        "scrambles",
        "dropbacks",
        "designed_run_rate",
        "designed_runs",
        "snaps",
    ]
    if not qb_run.empty:
        qb = qb_run.copy()
        join_cols = [
            c
            for c in ["player_canonical", "week"]
            if c in qb.columns and c in base.columns
        ]
        if not join_cols:
            join_cols = [
                c
                for c in ["player", "week"]
                if c in qb.columns and c in base.columns
            ]
        keep_cols = [
            c
            for c in [
                "player",
                "player_canonical",
                "week",
                *qb_feature_cols,
            ]
            if c in qb.columns
        ]
        if join_cols and keep_cols:
            qb_subset = qb[keep_cols].drop_duplicates(subset=join_cols)
            base = base.merge(
                qb_subset,
                on=join_cols,
                how="left",
                suffixes=("", "_qb"),
            )
            for helper in ["player_qb", "player_canonical_qb"]:
                if helper in base.columns:
                    base.drop(columns=[helper], inplace=True)
    for col in qb_feature_cols:
        if col not in base.columns:
            base[col] = np.nan

    unresolved_opponents = int(base["opponent_abbr"].isna().sum())
    # Attach team env (pace/proe/rz/12p/slot/ay + boxes)
    if not tf.empty:
        tf = tf.copy()
        if "team_abbr" in tf.columns:
            tf["team_abbr"] = _normalize_team_names(tf["team_abbr"])
        else:
            tf["team_abbr"] = _normalize_team_names(tf.get("team", pd.Series([], dtype=object)))
        keep_tf = ["team_abbr","def_pass_epa","def_rush_epa","def_sack_rate",
                   "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
                   "light_box_rate","heavy_box_rate","season"]
        keep_tf = [c for c in keep_tf if c in tf.columns]
        base = base.merge(tf[keep_tf].drop_duplicates(subset=["team_abbr"]), on=["team_abbr"], how="left")

    # Attach opponent defenses (EPA/sacks/boxes)
    if not tf.empty:
        opp_tf = tf.rename(columns={
            "team_abbr": "opponent_abbr",
            "def_pass_epa": "def_pass_epa_opp",
            "def_rush_epa": "def_rush_epa_opp",
            "def_sack_rate": "def_sack_rate_opp",
            "light_box_rate": "light_box_rate_opp",
            "heavy_box_rate": "heavy_box_rate_opp",
        })
        keep_opp = ["opponent_abbr","def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp"]
        base = base.merge(opp_tf[keep_opp].drop_duplicates(subset=["opponent_abbr"]), on="opponent_abbr", how="left")
    else:
        for c in ["def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp"]:
            if c not in base.columns:
                base[c] = np.nan

    # Coverage tags (opponent defense)
    cov = load_coverage()
    if not cov.empty and "defense_team" in cov.columns:
        cov2 = cov.rename(columns={"defense_team":"opponent"})
        for c in ["coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]:
            if c not in cov2.columns:
                cov2[c] = 0
        base = base.merge(
            cov2[["opponent","coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]].drop_duplicates(),
            on="opponent", how="left"
        )
        for c in ["coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]:
            base[c + "_opp"] = base[c]
            if c in base.columns:
                base.drop(columns=[c], inplace=True)

    # CB assignment (player + opponent)
    if not cba.empty:
        cba2 = cba.rename(columns={"opp_def_team": "opponent"})
        join_cols = [
            c
            for c in ["player_canonical", "opponent_abbr"]
            if c in base.columns and c in cba2.columns
        ]
        if join_cols:
            base = base.merge(cba2, on=join_cols, how="left")
        else:
            base = base.merge(cba2, on=["player", "opponent"], how="left")
        base["cb_penalty"] = base.get("penalty")
        if "penalty" in base.columns:
            base.drop(columns=["penalty"], inplace=True)
    else:
        base["cb_penalty"] = np.nan

    # Injuries (player, team)
    if not inj.empty:
        inj2 = inj.rename(columns={"status": "injury_status"})
        join_cols = [
            c
            for c in ["player_canonical", "team_abbr"]
            if c in base.columns and c in inj2.columns
        ]
        if join_cols:
            base = base.merge(inj2, on=join_cols, how="left")
        else:
            base = base.merge(inj2, on=["player", "team"], how="left")
    else:
        base["injury_status"] = np.nan

    # Weather via event_id (keep rows without event_id)
    w = load_weather()
    if not w.empty and "event_id" in base.columns:
        wx_keep = [c for c in ["event_id","wind_mph","temp_f","precip"] if c in w.columns]
        right = w[wx_keep].drop_duplicates()
        indicator_name = "_merge_weather_event"
        merged = base.merge(right, on="event_id", how="left", indicator=indicator_name)
        _log_merge_counts(
            stage="join_weather_event",
            left=base,
            right=right,
            join_cols=["event_id"],
            how="left",
            merged=merged,
            indicator_col=indicator_name,
        )
        merged.drop(columns=[indicator_name], inplace=True, errors="ignore")
        base = merged
    else:
        for c in ["wind_mph","temp_f","precip"]:
            if c not in base.columns:
                base[c] = np.nan

    if not WEATHER_WEEK_PATH.exists():
        print("[make_metrics] WARN: weather_week.csv not found; skipping slate weather join")
        ww = pd.DataFrame()
    else:
        ww = _read_csv(WEATHER_WEEK_PATH)
        if ww.empty:
            print("[make_metrics] WARN: weather_week.csv empty; skipping slate weather join")

    if not ww.empty:
        ww.columns = [c.lower() for c in ww.columns]
        home_col = "home_team" if "home_team" in ww.columns else "home"
        away_col = "away_team" if "away_team" in ww.columns else "away"
        ww_home = ww.copy()
        ww_home["team_abbr"] = _normalize_team_names(ww_home.get(home_col, pd.Series([], dtype=object)))
        ww_home["opponent_abbr"] = _normalize_team_names(ww_home.get(away_col, pd.Series([], dtype=object)))
        ww_away = ww.copy()
        ww_away["team_abbr"] = _normalize_team_names(ww_away.get(away_col, pd.Series([], dtype=object)))
        ww_away["opponent_abbr"] = _normalize_team_names(ww_away.get(home_col, pd.Series([], dtype=object)))
        ww_long = pd.concat([ww_home, ww_away], ignore_index=True)
        if "slate_date" not in ww_long.columns:
            ww_long["slate_date"] = os.getenv("SLATE_DATE", "").strip()
        else:
            ww_long["slate_date"] = ww_long["slate_date"].astype(str)
        rename_map = {
            "temp_f_mean": "temp_f_game",
            "temp_f": "temp_f_game",
            "wind_mph_mean": "wind_mph_game",
            "wind_mph": "wind_mph_game",
            "precip_prob_max": "precip_probability",
            "rain_flag": "precip_flag_game",
        }
        for old, new in rename_map.items():
            if old in ww_long.columns and new not in ww_long.columns:
                ww_long[new] = ww_long[old]
        ww_keep = [
            c
            for c in [
                "team_abbr",
                "opponent_abbr",
                "slate_date",
                "wind_mph_game",
                "temp_f_game",
                "precip_flag_game",
                "precip_probability",
                "notes",
                "blurb",
            ]
            if c in ww_long.columns
        ]
        if {"team_abbr", "opponent_abbr", "slate_date"}.issubset(ww_keep):
            weather_cols = ww_long[ww_keep].drop_duplicates()
            indicator_name = "_merge_weather_slate"
            merged = base.merge(
                weather_cols,
                on=["team_abbr", "opponent_abbr", "slate_date"],
                how="left",
                indicator=indicator_name,
            )
            _log_merge_counts(
                stage="join_weather_slate",
                left=base,
                right=weather_cols,
                join_cols=["team_abbr", "opponent_abbr", "slate_date"],
                how="left",
                merged=merged,
                indicator_col=indicator_name,
            )
            merged.drop(columns=[indicator_name], inplace=True, errors="ignore")
            base = merged
    for target, source in [
        ("wind_mph", "wind_mph_game"),
        ("temp_f", "temp_f_game"),
        ("precip", "precip_probability"),
    ]:
        if source in base.columns:
            if target not in base.columns:
                base[target] = np.nan
            base[target] = base[target].combine_first(base[source])
    if "precip_flag" in base.columns and "precip_flag_game" in base.columns:
        base["precip_flag"] = base["precip_flag"].combine_first(base["precip_flag_game"])
    elif "precip_flag_game" in base.columns:
        base["precip_flag"] = base["precip_flag_game"]

    # stamp season explicitly
    base["season"] = season

    # Default QB YPA prior if missing
    if "ypa_prior" in base.columns:
        pos_col = base.get("position") if "position" in base.columns else None
        if pos_col is not None:
            qb_mask = pos_col.astype(str).str.upper().str.startswith("QB")
            base.loc[qb_mask & base["ypa_prior"].isna(), "ypa_prior"] = 6.8

    # Final tidy and order
    # --- injected: attach coverage (opponent-level flags) and CB assignments (player-level) ---
    try:
        _cov_df = load_coverage()
    except Exception:
        _cov_df = pd.DataFrame()
    if not _cov_df.empty:
        _cov2 = _cov_df.rename(columns={"defense_team":"opponent"})
        for _c in ["coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]:
            if _c not in _cov2.columns:
                _cov2[_c] = 0
        base = base.merge(
            _cov2[["opponent","coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]].drop_duplicates(),
            on="opponent", how="left"
        )
        for _c in ["coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]:
            if _c + "_opp" not in base.columns:
                base[_c + "_opp"] = base[_c]
        if "coverage_top_shadow_opp" in base.columns:
            base["has_cb_shadow"] = (base["coverage_top_shadow_opp"].fillna(0).astype(int) == 1).astype(int)
        else:
            base["has_cb_shadow"] = 0

    try:
        _cba_df = load_cb_assignments()
    except Exception:
        _cba_df = pd.DataFrame()
    if not _cba_df.empty:
        _cba2 = _cba_df.rename(columns={"opp_def_team":"opponent"})
        base = base.merge(_cba2, on=["player","opponent"], how="left", suffixes=("","_cba"))
        need_fb = base["penalty"].isna().mean() > 0.80 if "penalty" in base.columns else True
        if need_fb:
            if "first_initial" not in base.columns or "last_name_u" not in base.columns:
                fi, ln = _player_init_last(base["player"])
                base["first_initial"] = base.get("first_initial", fi)
                base["last_name_u"] = base.get("last_name_u", ln)
            _cba2["fi"], _cba2["ln"] = _player_init_last(_cba2["player"] if "player" in _cba2.columns else pd.Series([], dtype=object))
            join_left = ["first_initial","last_name_u","opponent"]
            join_right= ["fi","ln","opponent"]
            fb_cols = ["penalty"]
            _t = base[join_left].merge(_cba2[join_right+fb_cols].drop_duplicates(), left_on=join_left, right_on=join_right, how="left")
            if "penalty" in _t.columns:
                if "penalty" not in base.columns:
                    base["penalty"] = np.nan
                base["penalty"] = base["penalty"].combine_first(_t["penalty"])
        base["cb_penalty"] = base.get("penalty")
        if "penalty" in base.columns:
            base.drop(columns=["penalty"], inplace=True, errors="ignore")
    else:
        if "cb_penalty" not in base.columns:
            base["cb_penalty"] = np.nan


    want = [
        "event_id",
        "player",
        "player_canonical",
        "player_clean_key",
        "team",
        "team_abbr",
        "opponent",
        "opponent_abbr",
        "market",
        "line",
        "over_odds",
        "under_odds",
        "target_share","rush_share","route_rate","yprr_proxy","ypt","ypc","ypa_prior","rz_tgt_share","rz_carry_share","position","role",
        "position_role",
        "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
        "def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp",
        "coverage_top_shadow_opp","coverage_heavy_man_opp","coverage_heavy_zone_opp",
        "cb_penalty","injury_status","wind_mph","temp_f","precip","precip_flag","team_wp","season",
        "slate_date","week","opp_plays_wk","opp_pace_wk","opp_proe_wk"
    ]
    for c in want:
        if c not in base.columns:
            base[c] = np.nan

    for col in ["wind_mph_game", "temp_f_game", "precip_probability", "precip_flag_game"]:
        if col in base.columns:
            base.drop(columns=[col], inplace=True, errors="ignore")

    if "precip_flag" not in base.columns:
        base["precip_flag"] = np.nan

    if "slate_date" in base.columns:
        slate_default = os.getenv("SLATE_DATE", "").strip()
        base["slate_date"] = base["slate_date"].astype(str).str[:10]
        base.loc[base["slate_date"].str.upper().isin({"", "NAN", "NAT", "NONE"}), "slate_date"] = slate_default
    else:
        base["slate_date"] = os.getenv("SLATE_DATE", "").strip()

    pos_cols = [c for c in base.columns if c.lower().startswith("position")]
    if "position" not in base.columns:
        base["position"] = pd.NA
    for col in pos_cols:
        if col in {"position", "position_role"}:
            continue
        base["position"] = base["position"].combine_first(base[col])
        base.drop(columns=[col], inplace=True, errors="ignore")
    if "position_role" in base.columns:
        base["position"] = base["position"].combine_first(base["position_role"])

    drop_suffix_cols = [c for c in base.columns if c.endswith("_pf") or c.endswith("_pvt") or c.endswith("_fbroles")]
    if drop_suffix_cols:
        base.drop(columns=drop_suffix_cols, inplace=True, errors="ignore")

    missing_usage = int(base["target_share"].isna().sum()) if "target_share" in base.columns else 0
    missing_weather = int(base["wind_mph"].isna().sum()) if "wind_mph" in base.columns else 0

    if missing_usage:
        print(f"[make_metrics] WARN: {missing_usage} players missing usage metrics")
    if unresolved_opponents:
        print(f"[make_metrics] WARN: opponent unresolved for {unresolved_opponents} rows")
    if missing_weather:
        print(f"[make_metrics] WARN: weather missing for {missing_weather} rows")

    missing_core_mask = (
        _missing_mask(base.get("team_abbr"), base.index)
        | _missing_mask(base.get("opponent_abbr"), base.index)
        | _missing_mask(base.get("position_role"), base.index)
    )
    core_missing = base.loc[missing_core_mask].copy()
    Path("data").mkdir(parents=True, exist_ok=True)
    core_missing.to_csv("data/metrics_missing_core.csv", index=False)
    print(f"[make_metrics] missing core coverage rows: {len(core_missing)}")

    base = base[want].drop_duplicates().reset_index(drop=True)
    base.attrs["row_counts"] = {
        "props_raw": len(props),
        "player_form_consensus": len(pf_consensus),
        "team_form": len(tf),
        "joined": len(base),
    }
    base.attrs["upstream_empty"] = upstream_empty
    return base

def _generate_metrics_dataframe(season: int) -> pd.DataFrame:
    try:
        df = build_metrics(season)
        upstream_empty = getattr(df, "attrs", {}).get("upstream_empty", {})
        row_counts = getattr(df, "attrs", {}).get("row_counts", {})
        # ## -- injected: final odds rejoin (idempotent) --
        try:
            props_raw = pd.read_csv(os.path.join("outputs", "props_raw.csv"))
            props_raw.columns = [c.lower() for c in props_raw.columns]
            if {"side", "price_american"}.issubset(props_raw.columns):
                k = [c for c in ["event_id", "player", "market", "line"] if c in props_raw.columns]
                if k:
                    tmp = props_raw[k + ["side", "price_american"]].copy()
                    tmp["side"] = tmp["side"].astype(str).str.upper().str.strip()
                    pvt = tmp.pivot_table(index=k, columns="side", values="price_american", aggfunc="first").reset_index()
                    pvt.columns = [("over_odds" if c == "OVER" else "under_odds" if c == "UNDER" else c) for c in pvt.columns]
                    for c in ["over_odds", "under_odds"]:
                        if c not in pvt.columns:
                            pvt[c] = np.nan
                    df = df.merge(pvt, on=k, how="left", suffixes=("", "_pvt"))
                    for c in ["over_odds", "under_odds"]:
                        c_p = c + "_pvt"
                        if c_p in df.columns:
                            df[c] = df[c].combine_first(df[c_p])
                            df.drop(columns=[c_p], inplace=True)
        except Exception as _e:
            print("[make_metrics] final odds rejoin skipped:", _e)

        # ## -- injected: final player_form_consensus backfill (idempotent) --
        try:
            pfc = pd.read_csv(os.path.join("data", "player_form_consensus.csv"))
            pfc.columns = [c.lower() for c in pfc.columns]
            # normalize expected columns
            rename_map = {
                "tgt_share": "target_share",
                "yprr": "yprr_proxy",
                "rz_rush_share": "rz_carry_share",
            }
            for k, v in rename_map.items():
                if k in pfc.columns and v not in pfc.columns:
                    pfc[v] = pfc[k]

            # initials+last on both sides
            def _mk_init_last(series):
                import re as _re

                fi, ln = [], []
                for nm in series.astype(str):
                    n = nm.replace(".", " ").strip()
                    if n and " " not in n:
                        caps = [i for i, ch in enumerate(n) if ch.isupper()]
                        if len(caps) >= 2:
                            fi.append(n[caps[0]].upper())
                            ln.append(n[caps[1]:].upper())
                            continue
                    toks = _re.split(r"\s+", n)
                    if toks and toks[0]:
                        fi.append(toks[0][0].upper())
                        ln.append(toks[-1].upper())
                    else:
                        fi.append("")
                        ln.append("")
                return fi, ln

            if "first_initial" not in df.columns or "last_name_u" not in df.columns:
                fi, ln = _mk_init_last(df["player"] if "player" in df.columns else pd.Series([]))
                df["first_initial"] = df.get("first_initial", pd.Series(fi))
                df["last_name_u"] = df.get("last_name_u", pd.Series(ln))

            if "first_initial" not in pfc.columns or "last_name_u" not in pfc.columns:
                fi2, ln2 = _mk_init_last(pfc["player"] if "player" in pfc.columns else pd.Series([]))
                pfc["first_initial"] = fi2
                pfc["last_name_u"] = ln2

            # choose keys: prefer (team, position, initial, last); otherwise (team, initial, last)
            keys_full = [k for k in ["team", "position", "first_initial", "last_name_u"] if k in df.columns and k in pfc.columns]
            keys_lo = [k for k in ["team", "first_initial", "last_name_u"] if k in df.columns and k in pfc.columns]
            use_keys = keys_full if len(keys_full) == 4 else (keys_lo if len(keys_lo) == 3 else None)

            if use_keys:
                use_cols = [
                    c
                    for c in [
                        "target_share",
                        "route_rate",
                        "rush_share",
                        "yprr_proxy",
                        "ypt",
                        "ypc",
                        "ypa",
                        "ypa_prior",
                        "rz_share",
                        "rz_tgt_share",
                        "rz_carry_share",
                        "role",
                        "position",
                    ]
                    if c in pfc.columns
                ]

                if use_cols:
                    sort_cols = [c for c in ["route_rate", "target_share"] if c in pfc.columns]
                    pf_fb = (
                        pfc.sort_values(use_keys + sort_cols, ascending=[True] * len(use_keys) + [False] * len(sort_cols))
                        .drop_duplicates(subset=use_keys, keep="first")
                    )

                    # only fill rows that lack usage currently
                    if set(["target_share", "route_rate", "rush_share"]).issubset(df.columns):
                        miss_mask = ~df[["target_share", "route_rate", "rush_share"]].notna().any(axis=1)
                    else:
                        miss_mask = pd.Series([True] * len(df))

                    fill = df.loc[miss_mask, use_keys].merge(pf_fb[use_keys + use_cols], on=use_keys, how="left")
                    for c in use_cols:
                        if c not in df.columns:
                            df[c] = np.nan
                        if c in fill.columns:
                            df.loc[miss_mask, c] = df.loc[miss_mask, c].combine_first(fill[c])

        except Exception as _e:
            print("[make_metrics] final pf_consensus backfill skipped:", _e)

        # ## -- injected: final odds rejoin --
        try:
            props_raw = pd.read_csv(os.path.join("outputs", "props_raw.csv"))
            props_raw.columns = [c.lower() for c in props_raw.columns]
            if {"side", "price_american"}.issubset(props_raw.columns):
                key = [c for c in ["event_id", "player", "market", "line"] if c in props_raw.columns]
                if key:
                    tmp = props_raw[key + ["side", "price_american"]].copy()
                    tmp["side"] = tmp["side"].astype(str).str.upper().str.strip()
                    pvt = tmp.pivot_table(index=key, columns="side", values="price_american", aggfunc="first").reset_index()
                    pvt = pvt.rename(columns={"OVER": "over_odds", "UNDER": "under_odds"})
                    # attach where missing
                    join_key = [c for c in ["event_id", "player", "market", "line"] if c in df.columns and c in pvt.columns]
                    if join_key:
                        df = df.merge(
                            pvt[join_key + ["over_odds", "under_odds"]],
                            on=join_key,
                            how="left",
                            suffixes=("", "_fromprops"),
                        )
                        for c in ["over_odds", "under_odds"]:
                            alt = f"{c}_fromprops"
                            if alt in df.columns:
                                df[c] = df[c].combine_first(df[alt])
                                df.drop(columns=[alt], inplace=True)
        except Exception as _e:
            print("[make_metrics] final odds rejoin skipped:", _e)

        df = _drop_duplicate_columns(df)
        if row_counts:
            row_counts = dict(row_counts)
            row_counts["joined"] = len(df)
        df.attrs["row_counts"] = row_counts
        df.attrs["upstream_empty"] = upstream_empty

    except Exception as e:
        print(f"[make_metrics] ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        # emit empty but schema-correct file so pipeline continues
        df = pd.DataFrame(
            columns=[
                "event_id",
                "player",
                "team",
                "opponent",
                "market",
                "line",
                "over_odds",
                "under_odds",
                "target_share",
                "rush_share",
                "route_rate",
                "yprr_proxy",
                "ypt",
                "ypc",
                "ypa_prior",
                "rz_tgt_share",
                "rz_carry_share",
                "position",
                "role",
                "pace",
                "proe",
                "rz_rate",
                "12p_rate",
                "slot_rate",
                "ay_per_att",
                "def_pass_epa_opp",
                "def_rush_epa_opp",
                "def_sack_rate_opp",
                "light_box_rate_opp",
                "heavy_box_rate_opp",
                "coverage_top_shadow_opp",
                "coverage_heavy_man_opp",
                "coverage_heavy_zone_opp",
                "cb_penalty",
                "injury_status",
                "wind_mph",
                "temp_f",
                "precip",
                "team_wp",
                "season",
                "week",
                "opp_plays_wk",
                "opp_pace_wk",
                "opp_proe_wk",
            ]
        )
        df.attrs["row_counts"] = {}
        df.attrs["upstream_empty"] = {}

    if df.empty:
        counts = getattr(df, 'attrs', {}).get('row_counts', {})
        upstream_empty = getattr(df, 'attrs', {}).get('upstream_empty', {})
        if counts:
            print(
                "[make_metrics] ERROR: metrics_ready empty after join (props_raw={props}, player_form={pf}, team_form={tf}, joined={joined})"
                .format(
                    props=counts.get('props_raw', 0),
                    pf=counts.get('player_form_consensus', 0),
                    tf=counts.get('team_form', 0),
                    joined=counts.get('joined', 0),
                )
            )
        else:
            print("[make_metrics] ERROR: metrics_ready empty after join (no counts)")
        if upstream_empty:
            empty_sources = [name for name, flag in upstream_empty.items() if flag]
            if empty_sources:
                print(
                    "[make_metrics] ERROR: upstream inputs empty → {}".format(
                        ", ".join(sorted(empty_sources))
                    )
                )
            else:
                print("[make_metrics] INFO: upstream inputs all reported non-empty")
        raise RuntimeError("metrics_ready is empty after merge")

    # Optional: add opponent weekly env (pace/proe/plays_est) if present
    try:
        tfw = _read_csv(os.path.join(DATA_DIR, "team_form_weekly.csv"))
        if (
            not df.empty
            and not tfw.empty
            and {"opponent", "week"}.issubset(df.columns)
            and {"team", "week"}.issubset(tfw.columns)
        ):
            tfw = tfw.rename(columns={"team": "opponent"})
            cols = [c for c in ["opponent", "week", "plays_est", "pace", "proe"] if c in tfw.columns]
            tfw = tfw[cols].drop_duplicates().rename(
                columns={"plays_est": "opp_plays_wk", "pace": "opp_pace_wk", "proe": "opp_proe_wk"}
            )
            df = df.merge(tfw, on=["opponent", "week"], how="left")
            for c in ["opp_plays_wk", "opp_pace_wk", "opp_proe_wk"]:
                if c not in df.columns:
                    df[c] = np.nan
    except Exception:
        pass

    df.replace({"NAN": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan}, inplace=True)
    # FINAL_OPPONENT_SANITY
    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].replace({"NAN": np.nan, "nan": np.nan, "NaN": np.nan, "": np.nan})
    if 'row_counts' in locals() and isinstance(row_counts, dict):
        df.attrs["row_counts"] = dict(row_counts)
    if 'upstream_empty' in locals() and isinstance(upstream_empty, dict):
        df.attrs["upstream_empty"] = dict(upstream_empty)
    return df


REQUIRED_INPUTS: dict[str, Path] = {
    "team_form": TEAM_FORM_PATH,
    "player_form": PLAYER_FORM_PATH,
    "player_form_consensus": PLAYER_CONS_PATH,
}

OPTIONAL_INPUTS: dict[str, Path] = {
    "qb_scramble_rates": QB_SCRAMBLE_PATH,
    "qb_designed_runs": QB_DESIGNED_PATH,
    "qb_mobility": QB_MOBILITY_PATH,
    "qb_run_metrics": QB_RUN_METRICS_PATH,
    "weather": WEATHER_PATH,
    "weather_week": WEATHER_WEEK_PATH,
}


def _validate_core_inputs() -> bool:
    for label, path in REQUIRED_INPUTS.items():
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"[make_metrics] FATAL: missing required input CSV: {path}")
            return False
        except pd.errors.EmptyDataError:
            print(f"[make_metrics] FATAL: required input CSV is empty: {path}")
            return False
        except Exception as exc:
            print(f"[make_metrics] FATAL: unable to load required input '{label}': {exc}")
            return False

        if df.empty:
            print(f"[make_metrics] FATAL: required input '{label}' has no rows: {path}")
            return False
    return True


def _log_optional_inputs() -> None:
    for label, path in OPTIONAL_INPUTS.items():
        if not path.exists() or path.stat().st_size == 0:
            print(f"[make_metrics] WARN: optional input '{label}' missing or empty ({path})")


def main(args: argparse.Namespace) -> int:
    if not isinstance(args.season, int):
        print("[make_metrics] FATAL: season must be an integer")
        return 1

    _safe_mkdir(DATA_DIR)

    if not _validate_core_inputs():
        return 1

    _log_optional_inputs()

    try:
        df = _generate_metrics_dataframe(args.season)
        if "is_bye" in df.columns:
            before = len(df)
            df = df.loc[~df["is_bye"].fillna(False)].copy()
            removed = before - len(df)
            if removed:
                print(
                    f"[make_metrics] excluded BYE rows: {removed} (remaining {len(df)})"
                )
            df.drop(columns=["is_bye"], inplace=True, errors="ignore")
    except Exception as exc:
        print(f"[make_metrics] FATAL: unhandled error building metrics: {exc}")
        traceback.print_exc()
        return 1

    if df is None or df.empty:
        print("[make_metrics] FATAL: merged metrics is empty, aborting")
        return 1

    metrics = df.copy()

    try:
        from scripts.fantasypoints_wr_cb_scraper import (  # type: ignore
            get_current_week,
            main as update_wr_cb,
        )

        print("[make_metrics] Updating FantasyPoints WR–CB matchups…")
        update_wr_cb()
        current_week = get_current_week()
    except Exception as exc:  # pragma: no cover - network resiliency
        print(f"[make_metrics] WARN: WR–CB update failed ({exc})")
        current_week = None

    if current_week is not None:
        wr_cb_path = Path(f"data/wr_cb_matchups_week_{current_week}.csv")
        if wr_cb_path.exists():
            try:
                wr_cb_df = pd.read_csv(wr_cb_path)
                join_cols = ["player", "team", "opponent"]
                if all(col in wr_cb_df.columns for col in join_cols) and all(
                    col in metrics.columns for col in join_cols
                ):
                    metrics = metrics.merge(wr_cb_df, on=join_cols, how="left")
                    print(
                        f"[make_metrics] WR–CB matchup data merged for Week {current_week}"
                    )
                else:
                    print(
                        "[make_metrics] WARN: WR–CB matchup CSV missing expected columns; skipping merge"
                    )
            except Exception as exc:  # pragma: no cover - CSV read resiliency
                print(f"[make_metrics] WARN: Failed to merge WR–CB matchup data ({exc})")
        else:
            print(
                f"[make_metrics] WARN: WR–CB matchup file not found for Week {current_week} ({wr_cb_path})"
            )

    METRICS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_OUT_PATH, index=False)
    print(f"[make_metrics] Wrote {len(metrics)} rows → {METRICS_OUT_PATH}")

    # --- Auto validation hook ---
    try:
        from scripts.validate_build_integrity import run_core_validation

        print("\n[MAKE-METRICS] Running post-build validation...")
        ok = run_core_validation()
        if not ok:
            print("[MAKE-METRICS] ❌ Validation failed — stopping pipeline")
            raise SystemExit(1)
        else:
            print("[MAKE-METRICS] ✅ Validation passed successfully")
    except Exception as e:
        print(f"[MAKE-METRICS] Validation skipped or errored: {e}")
        raise
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year, e.g. 2025",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        help="Run mode. Leave default = full.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="",
        help="Slate date (YYYY-MM-DD). May be empty.",
    )
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return main(args)


if __name__ == "__main__":
    sys.exit(cli())
