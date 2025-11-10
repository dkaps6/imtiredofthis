# scripts/make_team_form.py
"""
Build team-level context features for pricing & modeling.

Outputs: data/team_form.csv

Patches (surgical, no nukes):
- Read Sharp FIRST in fallback sweep so light/heavy box and other tendencies seed the frame.
- Normalize external keys: accept team_abbr/team_name and map to canonical `team`.
- Normalize percent-like fields (e.g., "61%" -> 0.61, 61 -> 0.61).
- Prefer Sharp's light_box_rate/heavy_box_rate explicitly after all merges.
- Validate AFTER fallback, not before.
- Add debug prints to understand what's being merged during CI runs.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import List, Any
import re  # added for robust team-name normalization

import pandas as pd
import numpy as np

# --- Debug/version banner (helps confirm which file ran and what paths we read) ---
VERSION_TAG = "make_team_form PATCH v2025-10-19 22:10 ET"
print(f"[make_team_form] {VERSION_TAG}  __file__={__file__}")
print(f"[make_team_form] CWD={os.getcwd()}  DATA_DIR={os.path.abspath('data')}")
print(
    f"[make_team_form] sharp_path={os.path.abspath(os.path.join('data','sharp_team_form.csv'))} "
    f"exists={os.path.exists(os.path.join('data','sharp_team_form.csv'))} "
    f"size={os.path.getsize(os.path.join('data','sharp_team_form.csv')) if os.path.exists(os.path.join('data','sharp_team_form.csv')) else -1}"
)

# -----------------------------
# Imports: prefer nflreadpy
# -----------------------------
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
                "Neither nflreadpy nor nfl_data_py is available. Please `pip install nflreadpy`."
            ) from e


NFLV, NFL_PKG = _import_nflverse()

DATA_DIR = "data"
OUTPATH = Path(DATA_DIR) / "team_form.csv"
TEAM_FORM_OUTPUT_COLUMNS = [
    "team","team_abbr","season","games_played",
    "def_pass_epa","def_rush_epa","def_sack_rate",
    "pace","neutral_pace","neutral_pace_score",
    "pass_rate_over_expected","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
    "light_box_rate","heavy_box_rate",
    # NEW metrics:
    "success_rate_off","success_rate_def","success_rate_diff",
    "explosive_play_rate_allowed",
]

# -----------------------------
# Canonical team mapping (expanded + forgiving)
# -----------------------------
VALID = {"ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU",
         "IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ",
         "PHI","PIT","SEA","SF","TB","TEN","WAS"}

TEAM_NAME_TO_ABBR = {
    # Abbreviations / historical variants
    "ARI":"ARI","ARZ":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE",
    "DAL":"DAL","DEN":"DEN","DET":"DET","GB":"GB","GNB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX","JAC":"JAX",
    "KC":"KC","KCC":"KC","LAC":"LAC","LAR":"LAR","LA":"LAR","LV":"LV","OAK":"LV","LAS":"LV","MIA":"MIA",
    "MIN":"MIN","NE":"NE","NWE":"NE","NO":"NO","NOR":"NO","NYG":"NYG","NYJ":"NYJ","PHI":"PHI","PIT":"PIT",
    "SEA":"SEA","SF":"SF","SFO":"SF","TB":"TB","TAM":"TB","TEN":"TEN","WAS":"WAS","WSH":"WAS","WFT":"WAS",
    # Full official names
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
    "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
    "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
    "LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LAR","LAS VEGAS RAIDERS":"LV",
    "MIAMI DOLPHINS":"MIA","MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE",
    "NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG","NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI",
    "PITTSBURGH STEELERS":"PIT","SEATTLE SEAHAWKS":"SEA","SAN FRANCISCO 49ERS":"SF",
    "TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS",
    "WASHINGTON FOOTBALL TEAM":"WAS",
    # City-only & nicknames (common scraped variants)
    "ARIZONA":"ARI","CARDINALS":"ARI","ATLANTA":"ATL","FALCONS":"ATL","BALTIMORE":"BAL","RAVENS":"BAL",
    "BUFFALO":"BUF","BILLS":"BUF","CAROLINA":"CAR","PANTHERS":"CAR","CHICAGO":"CHI","BEARS":"CHI",
    "CINCINNATI":"CIN","BENGALS":"CIN","CLEVELAND":"CLE","BROWNS":"CLE","DALLAS":"DAL","COWBOYS":"DAL",
    "DENVER":"DEN","BRONCOS":"DEN","DETROIT":"DET","LIONS":"DET","GREEN BAY":"GB","PACKERS":"GB",
    "HOUSTON":"HOU","TEXANS":"HOU","INDIANAPOLIS":"IND","COLTS":"IND","JACKSONVILLE":"JAX","JAGUARS":"JAX",
    "KANSAS CITY":"KC","CHIEFS":"KC","CHARGERS":"LAC","RAMS":"LAR","LOS ANGELES":"LAR","LAS VEGAS":"LV","RAIDERS":"LV",
    "MIAMI":"MIA","DOLPHINS":"MIA","MINNESOTA":"MIN","VIKINGS":"MIN","NEW ENGLAND":"NE","PATRIOTS":"NE",
    "NEW ORLEANS":"NO","SAINTS":"NO","GIANTS":"NYG","JETS":"NYJ","PHILADELPHIA":"PHI","EAGLES":"PHI",
    "PITTSBURGH":"PIT","STEELERS":"PIT","SEATTLE":"SEA","SEAHAWKS":"SEA","SAN FRANCISCO":"SF","49ERS":"SF",
    "TAMPA BAY":"TB","BUCCANEERS":"TB","TENNESSEE":"TEN","TITANS":"TEN","WASHINGTON":"WAS","COMMANDERS":"WAS",
}

def canon_team(x: str) -> str:
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

# -----------------------------
# Helpers
# -----------------------------
def _safe_mkdir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _write_team_form_csv(df: pd.DataFrame | None, success: bool = False) -> pd.DataFrame:
    """Write ``df`` to ``OUTPATH`` ensuring headers exist."""

    out_path = OUTPATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df is None or df.empty:
        out_df = pd.DataFrame(columns=TEAM_FORM_OUTPUT_COLUMNS)
        out_df.to_csv(out_path, index=False)
        print(f"[make_team_form] WARNING wrote headers only (0 rows) → {out_path}")
        return out_df

    out_df = df.copy()
    missing = [col for col in TEAM_FORM_OUTPUT_COLUMNS if col not in out_df.columns]
    for col in missing:
        out_df[col] = pd.NA

    ordered = TEAM_FORM_OUTPUT_COLUMNS + [
        col for col in out_df.columns if col not in TEAM_FORM_OUTPUT_COLUMNS
    ]
    out_df = out_df[ordered]
    out_df.to_csv(out_path, index=False)
    if success:
        print(f"[make_team_form] wrote {out_path} rows={len(out_df)} with Sharp metrics merged OK")
    else:
        print(f"[make_team_form] Wrote {len(out_df)} rows → {out_path}")
    return out_df


def _is_empty(obj) -> bool:
    try:
        return (obj is None) or (not hasattr(obj, "__len__")) or (len(obj) == 0)
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
    if isinstance(obj, (list, tuple)) and len(obj) and hasattr(obj[0], "to_pandas"):
        try:
            return pd.concat([b.to_pandas() for b in obj], ignore_index=True)
        except Exception:
            pass
    return pd.DataFrame(obj)


def zscore(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df and df[c].notna().sum() >= 8:
            m = pd.to_numeric(df[c], errors="coerce").mean()
            s = pd.to_numeric(df[c], errors="coerce").std(ddof=0)
            if s and not np.isclose(s, 0):
                df[c + "_z"] = (pd.to_numeric(df[c], errors="coerce") - m) / s
            else:
                df[c + "_z"] = np.nan
    return df


def _neutral_mask(pbp: pd.DataFrame) -> pd.Series:
    m = pd.Series(True, index=pbp.index)
    if "score_differential" in pbp:
        m &= pbp["score_differential"].between(-7, 7)
    if "wp" in pbp:
        m &= pbp["wp"].between(0.2, 0.8)
    if "qtr" in pbp:
        m &= pbp["qtr"] <= 3
    return m


def safe_div(n, d):
    n = pd.to_numeric(n, errors="coerce").astype(float)
    d = pd.to_numeric(d, errors="coerce").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d == 0, np.nan, n / d)


def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


# Whitelist for non-destructive merge (includes Sharp extras)
MERGE_WHITELIST = set([
    "team", "pace", "proe", "rz_rate", "12p_rate", "slot_rate", "ay_per_att",
    "def_pass_epa", "def_rush_epa", "def_sack_rate",
    "light_box_rate", "heavy_box_rate",
    # Sharp extras:
    "seconds_per_play", "seconds_per_play_last5",
    "plays_per_game", "neutral_db_rate", "neutral_db_rate_last_5",
    "motion_rate", "play_action_rate", "shotgun_rate", "no_huddle_rate",
    "blitz_rate", "sub_package_rate",
    "dl_pressure_rate", "dl_no_blitz_pressure_rate",
    "dl_ybc_per_rush", "dl_stuff_rate",
    "ypt_allowed", "wr_ypt_allowed", "te_ypt_allowed", "rb_ypt_allowed",
    "outside_ypt_allowed", "slot_ypt_allowed",
    # alias we map:
    "air_yards_per_att",  # -> ay_per_att
    "airyardsatt",        # -> ay_per_att
])


def _ensure_plays_est_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee that pricing downstream can rely on a plays_est column."""
    if df is None or not isinstance(df, pd.DataFrame):
        return df

    out = df.copy()
    if "plays_est" in out.columns:
        out["plays_est"] = pd.to_numeric(out["plays_est"], errors="coerce")
    else:
        out["plays_est"] = np.nan

    if "plays_per_game" in out.columns:
        plays_pg = pd.to_numeric(out["plays_per_game"], errors="coerce")
        out["plays_est"] = out["plays_est"].combine_first(plays_pg)

    if "seconds_per_play" in out.columns:
        sec = pd.to_numeric(out["seconds_per_play"], errors="coerce")
        derived = pd.Series(np.where(sec > 0, 3600.0 / sec, np.nan), index=out.index)
        out["plays_est"] = out["plays_est"].combine_first(derived)

    return out


def _compute_team_proe_from_pbp(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate pass rate over expected (PROE) at the team level using only data
    we already loaded from nflverse/nflreadr.

    Steps:
    - Define 'neutral situations' = whatever logic we already consider for neutral pace
      (score differential small, early downs, etc.). Reuse that same mask if it exists.
      If no mask exists yet, approximate:
         offense_is_team = pbp_df['posteam'].notna()
         neutral_score   = pbp_df['score_differential'].between(-7,7, inclusive='both')
         early_downs     = pbp_df['down'].isin([1,2])
         neutral = offense_is_team & neutral_score & early_downs
    - For each team in those rows:
        team_pass_rate = mean(qb_dropback == 1)  (or pass_attempt == 1 if dropback not present)
    - Compute league_pass_rate = overall mean of team_pass_rate weighted by snaps.
    - PROE = team_pass_rate - league_pass_rate.
    Returns DataFrame with columns:
        team_abbr, pass_rate_over_expected
    """

    df = pbp_df.copy()

    # If we already build something like neutral mask elsewhere, try to reuse.
    # Otherwise construct a fallback neutral mask here:
    if "posteam" not in df.columns:
        return pd.DataFrame(columns=["team_abbr", "pass_rate_over_expected"])

    neutral = df["posteam"].notna()
    if "down" in df.columns:
        neutral &= df["down"].isin([1, 2])
    if "score_differential" in df.columns:
        neutral &= df["score_differential"].between(-7, 7, inclusive="both")

    neutral_df = df.loc[neutral].copy()

    # prefer qb_dropback if present, else pass_attempt
    pass_flag_col = "qb_dropback" if "qb_dropback" in neutral_df.columns else (
        "pass_attempt" if "pass_attempt" in neutral_df.columns else None
    )
    if pass_flag_col is None:
        # if we truly have neither, bail out with empty
        return pd.DataFrame(columns=["team_abbr", "pass_rate_over_expected"])

    # snaps per team
    grp = neutral_df.groupby("posteam", as_index=False).agg(
        snaps=("posteam", "size"),
        pass_snaps=(pass_flag_col, "sum"),
    )
    grp["team_pass_rate"] = grp["pass_snaps"] / grp["snaps"]

    # league avg baseline (weighted by snaps)
    total_snaps = grp["snaps"].sum()
    if total_snaps > 0:
        league_pass_rate = (grp["pass_snaps"].sum() / total_snaps)
    else:
        league_pass_rate = np.nan

    grp["pass_rate_over_expected"] = grp["team_pass_rate"] - league_pass_rate

    grp = grp.rename(columns={"posteam": "team"})
    grp["team"] = grp["team"].map(canon_team)
    grp = grp[grp["team"].isin(VALID)]
    return grp[["team", "pass_rate_over_expected"]]

PERCENTY_COLS = {
    "light_box_rate", "heavy_box_rate", "neutral_db_rate", "neutral_db_rate_last_5",
    "blitz_rate", "sub_package_rate", "motion_rate", "play_action_rate",
    "shotgun_rate", "no_huddle_rate",
}


def _prep_team_key_and_rates(ext: pd.DataFrame) -> pd.DataFrame:
    if _is_empty(ext):
        return ext
    df = ext.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # team key promotion
    if "team" not in df.columns:
        for cand in ("team_abbr", "team_name", "club_code", "offense_team", "posteam"):
            if cand in df.columns:
                df = df.rename(columns={cand: "team"})
                break

    # attempt strict → forgiving map
    drop_guard = False
    if "team" in df.columns:
        raw_team = df["team"].astype(str)
        df["team"] = raw_team.str.strip().str.upper().map(canon_team)
        before = len(df)
        df = df[df["team"] != ""]
        if len(df) == 0 and before > 0:
            # Rescue path: rebuild team codes from original strings using forgiving map
            drop_guard = True
            src = raw_team.str.strip().str.upper()
            codes = src.map(lambda t: TEAM_NAME_TO_ABBR.get(
                t,
                TEAM_NAME_TO_ABBR.get(re.sub(r"[^A-Z0-9 ]+", "", t).strip(), "")
            ))
            df = ext.copy()
            df.columns = [c.strip().lower() for c in df.columns]
            df["team"] = codes
            df = df[df["team"].isin(VALID)]

    # percent-like normalization ("61%" -> 0.61; "61" -> 0.61)
    for c in [col for col in PERCENTY_COLS if col in df.columns]:
        ser = df[c].astype(str).str.replace("%", "", regex=False).str.strip()
        ser = pd.to_numeric(ser, errors="coerce")
        if ser.notna().any() and ser.max(skipna=True) and ser.max(skipna=True) > 1.0:
            ser = ser / 100.0
        df[c] = ser

    # alias mapping for AY/Att
    if "air_yards_per_att" in df.columns and "ay_per_att" not in df.columns:
        df["ay_per_att"] = pd.to_numeric(df["air_yards_per_att"], errors="coerce")
    if "airyardsatt" in df.columns and "ay_per_att" not in df.columns:
        df["ay_per_att"] = pd.to_numeric(df["airyardsatt"], errors="coerce")

    # map seconds/play & plays per game if present
    if "secplay" in df.columns and "seconds_per_play" not in df.columns:
        df["seconds_per_play"] = pd.to_numeric(df["secplay"], errors="coerce")
    if "secplay_last_5" in df.columns and "seconds_per_play_last5" not in df.columns:
        df["seconds_per_play_last5"] = pd.to_numeric(df["secplay_last_5"], errors="coerce")
    for k in ("off_playsg", "total_playsg"):
        if k in df.columns and "plays_per_game" not in df.columns:
            df["plays_per_game"] = pd.to_numeric(df[k], errors="coerce")

    if drop_guard:
        print("[make_team_form] NOTE: Sharp team strings were non-standard; applied forgiving map and rescued codes.")

    return df


def _non_destructive_team_merge(base: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    """Merge 'add' into 'base' by team without overwriting non-null base values."""
    if _is_empty(add) or "team" not in add.columns:
        return base

    add = add.copy()
    add.columns = [c.lower() for c in add.columns]

    # alias to canonical
    if "air_yards_per_att" in add.columns and "ay_per_att" not in add.columns:
        add["ay_per_att"] = add["air_yards_per_att"]
    if "airyardsatt" in add.columns and "ay_per_att" not in add.columns:
        add["ay_per_att"] = add["airyardsatt"]

    # incoming may already be canonical now; still normalize defensively
    add["team"] = add["team"].astype(str).str.strip().str.upper()
    add["team"] = add["team"].map(lambda s: TEAM_NAME_TO_ABBR.get(
        s,
        TEAM_NAME_TO_ABBR.get(re.sub(r"[^A-Z0-9 ]+", "", s).strip(), s)
    ))
    add = add[add["team"].isin(VALID)]

    keep = [c for c in MERGE_WHITELIST if c in add.columns]
    if "team" not in keep:
        keep = ["team"] + keep
    add = add[keep].drop_duplicates()

    out = base.merge(add, on="team", how="left", suffixes=("", "_ext"))
    for c in keep:
        if c == "team":
            continue
        ext = f"{c}_ext"
        if ext in out.columns:
            out[c] = out[c].combine_first(out[ext])
            out.drop(columns=[ext], inplace=True)
    return out


def _force_team_col(df: pd.DataFrame, off_col: str | None) -> pd.DataFrame:
    if df is None or not hasattr(df, "columns"):
        return df
    if "team" in df.columns:
        return df
    if off_col and off_col in df.columns:
        return df.rename(columns={off_col: "team"})
    for cand in ["posteam", "offense_team", "team_abbr", "club_code"]:
        if cand in df.columns:
            return df.rename(columns={cand: "team"})
    return df


# -----------------------------
# Loaders
# -----------------------------
def load_pbp(season: int) -> pd.DataFrame:
    try:
        if NFL_PKG == "nflreadpy":
            pbp_raw = NFLV.load_pbp(seasons=[season])
        else:
            pbp_raw = NFLV.import_pbp_data([season], downcast=True)  # type: ignore
        pbp = _to_pandas(pbp_raw)
        pbp.columns = [c.lower() for c in pbp.columns]
        return pbp
    except Exception as e:
        print(f"[make_team_form] WARNING: failed to load play-by-play for {season}: {e}", file=sys.stderr)
        return pd.DataFrame()


def load_participation(season: int) -> pd.DataFrame:
    try:
        if NFL_PKG == "nflreadpy":
            part_raw = NFLV.load_participation(seasons=[season])
        else:
            return pd.DataFrame()
        part = _to_pandas(part_raw)
        part.columns = [c.lower() for c in part.columns]
        return part
    except Exception:
        return pd.DataFrame()


def load_schedules(season: int) -> pd.DataFrame:
    try:
        if NFL_PKG == "nflreadpy":
            sch_raw = NFLV.load_schedules(seasons=[season])
        else:
            sch_raw = NFLV.import_schedules([season])  # type: ignore
        sch = _to_pandas(sch_raw)
        sch.columns = [c.lower() for c in sch.columns]
        return sch
    except Exception:
        return pd.DataFrame()


# -----------------------------
# Feature builders
# -----------------------------
def compute_def_epa_and_sacks(pbp: pd.DataFrame) -> pd.DataFrame:
    if _is_empty(pbp):
        return pd.DataFrame(columns=["team", "def_pass_epa", "def_rush_epa", "def_sack_rate", "games_played"])

    df = pbp.copy()
    if "epa" not in df:
        df["epa"] = np.nan

    is_pass = df.get("pass", pd.Series(False, index=df.index)).astype(bool)
    is_rush = df.get("rush", pd.Series(False, index=df.index)).astype(bool)

    def_team_col = "defteam" if "defteam" in df else ("def_team" if "def_team" in df else None)
    off_team_col = "posteam" if "posteam" in df else ("offense_team" if "offense_team" in df else None)
    if def_team_col is None or off_team_col is None:
        return pd.DataFrame(columns=["team", "def_pass_epa", "def_rush_epa", "def_sack_rate", "games_played"])

    sack_flag = df.get("sack", pd.Series(0, index=df.index)).fillna(0).astype(int)
    dropbacks = df.get("qb_dropback", pd.Series(0, index=df.index)).fillna(0).astype(int)

    grp = df.groupby(def_team_col, dropna=False)
    def_pass = grp.apply(lambda x: x.loc[is_pass.reindex(x.index, fill_value=False), "epa"].mean())
    def_rush = grp.apply(lambda x: x.loc[is_rush.reindex(x.index, fill_value=False), "epa"].mean())
    sacks = grp[sack_flag.name].sum()
    opp_db = grp[dropbacks.name].sum()
    games = grp["game_id"].nunique() if "game_id" in df.columns else grp.size()

    agg = pd.DataFrame({
        "def_pass_epa": def_pass,
        "def_rush_epa": def_rush,
        "def_sacks": sacks,
        "opp_dropbacks": opp_db,
        "games_played": games
    }).reset_index().rename(columns={def_team_col: "team"})

    agg["team"] = agg["team"].astype(str).str.upper().str.strip().map(canon_team)
    agg = agg[agg["team"] != ""]
    agg["def_sack_rate"] = safe_div(agg["def_sacks"], agg["opp_dropbacks"])
    return agg


def compute_pace_and_proe(pbp: pd.DataFrame) -> pd.DataFrame:
    if _is_empty(pbp):
        return pd.DataFrame(columns=["team", "pace_neutral", "proe"])

    df = pbp.copy()
    df.columns = [c.lower() for c in df.columns]
    neutral = _neutral_mask(df)
    dfn = df.loc[neutral].copy()

    off_col = "posteam" if "posteam" in dfn else ("offense_team" if "offense_team" in dfn else None)
    if off_col is None:
        return pd.DataFrame(columns=["team", "pace_neutral", "proe"])

    dfn = dfn.sort_values([off_col, "game_id", "qtr", "play_id"], kind="mergesort")
    grp = dfn.groupby([off_col, "game_id"], dropna=False)

    if "game_seconds_remaining" in dfn.columns:
        dfn["__gsr_diff"] = grp["game_seconds_remaining"].diff(-1).abs()
        pace_per_game = dfn.groupby([off_col, "game_id"], dropna=False)["__gsr_diff"].mean()
        pace_team = pace_per_game.groupby(level=0).mean().rename("pace_neutral").reset_index()
        pace_team = pace_team.rename(columns={off_col: "team"})
    else:
        neutral_plays = grp.size().groupby(level=0).sum().rename("neutral_plays").reset_index()
        pace_team = neutral_plays.assign(pace_neutral=np.nan).rename(columns={off_col: "team"})

    prate = dfn.groupby(off_col, dropna=False)["pass"].mean() if "pass" in dfn.columns else pd.Series(dtype=float)
    if "xpass" in dfn:
        xpass = dfn.groupby(off_col, dropna=False)["xpass"].mean()
        proe = prate - xpass
    elif "pass_probability" in dfn:
        xp = dfn.groupby(off_col, dropna=False)["pass_probability"].mean()
        proe = prate - xp
    else:
        league_neutral_pass = prate.mean() if len(prate) else 0.55
        proe = prate - league_neutral_pass

    out = pace_team.merge(proe.rename("proe"), left_on="team", right_index=True, how="left")
    out = _force_team_col(out, off_col)
    out["team"] = out["team"].map(canon_team)
    out = out[out["team"] != ""]
    out = out[["team", "pace_neutral", "proe"]]
    return out


def compute_red_zone_and_airyards(pbp: pd.DataFrame) -> pd.DataFrame:
    if _is_empty(pbp):
        return pd.DataFrame(columns=["team", "rz_rate", "ay_per_att"])

    df = pbp.copy()
    df.columns = [c.lower() for c in df.columns]
    off_col = "posteam" if "posteam" in df.columns else ("offense_team" if "offense_team" in df.columns else None)
    if off_col is None:
        return pd.DataFrame(columns=["team", "rz_rate", "ay_per_att"])

    yardline = pd.to_numeric(df.get("yardline_100"), errors="coerce")
    rz = (df.assign(yardline_100=yardline, rz_flag=lambda x: (x["yardline_100"] <= 20).astype(int))
            .groupby(off_col, dropna=False)["rz_flag"].mean().rename("rz_rate"))

    is_pass = df.get("pass", pd.Series(False, index=df.index)).astype(bool)
    pass_df = df.loc[is_pass].copy()
    ay = pd.to_numeric(pass_df.get("air_yards"), errors="coerce")
    ay_per_att = (pass_df.assign(air_yards=ay)
                        .groupby(off_col, dropna=False)["air_yards"]
                        .mean().rename("ay_per_att"))

    out = pd.concat([rz, ay_per_att], axis=1).reset_index().rename(columns={off_col: "team"})
    out["team"] = out["team"].map(canon_team)
    out = out[out["team"] != ""]
    return out


def compute_personnel_rates(pbp: pd.DataFrame, participation: pd.DataFrame) -> pd.DataFrame:
    if _is_empty(pbp) and _is_empty(participation):
        return pd.DataFrame(columns=["team", "personnel_12_rate", "light_box_rate", "heavy_box_rate"])

    df = _to_pandas(pbp).copy()
    off_col = "posteam" if "posteam" in df else ("offense_team" if "offense_team" in df else None)
    if _is_empty(df) or off_col is None:
        base = pd.DataFrame(columns=["team", "personnel_12_rate"])
    else:
        per = df.get("personnel_offense", pd.Series(np.nan, index=df.index)).astype(str).str.extract(r"(\d\d)").rename(columns={0: "personnel"})
        df = df.assign(_per=per["personnel"])
        grp = df.groupby(off_col, dropna=False)
        p12 = grp.apply(lambda x: (x["_per"] == "12").mean() if len(x) else np.nan).rename("personnel_12_rate")
        base = p12.reset_index().rename(columns={off_col: "team"})
    if "team" in base.columns:
        base["team"] = base["team"].map(canon_team)
        base = base[base["team"] != ""]

    # box rates from participation (if available)
    light = heavy = None
    if not _is_empty(participation):
        p = participation.copy()
        box_col = None
        for cand in ["box", "men_in_box", "in_box", "defenders_in_box"]:
            if cand in p.columns:
                box_col = cand
                break
        team_col = None
        for cand in ["offense_team", "posteam", "team", "club_code"]:
            if cand in p.columns:
                team_col = cand
                break
        if box_col and team_col:
            p["_team"] = p[team_col].map(canon_team)
            p = p[p["_team"] != ""]
            if not p.empty:
                p["_light"] = (pd.to_numeric(p[box_col], errors="coerce") <= 6).astype(float)
                p["_heavy"] = (pd.to_numeric(p[box_col], errors="coerce") >= 8).astype(float)
                g = p.groupby("_team", dropna=False)
                light = g["_light"].mean().rename("light_box_rate")
                heavy = g["_heavy"].mean().rename("heavy_box_rate")

    out = base.copy()
    if light is not None:
        t = light.reset_index().rename(columns={light.index.name or "index": "team"})
        t["team"] = t["team"].map(canon_team)
        t = t[t["team"] != ""]
        out = out.merge(t, on="team", how="left") if not out.empty else t
    if heavy is not None:
        t = heavy.reset_index().rename(columns={heavy.index.name or "index": "team"})
        t["team"] = t["team"].map(canon_team)
        t = t[t["team"] != ""]
        out = out.merge(t, on="team", how="left") if not out.empty else t

    if "personnel_12_rate" not in out.columns:
        out["personnel_12_rate"] = np.nan

    return out.reset_index(drop=True)


def merge_slot_rate_from_roles(df: pd.DataFrame) -> pd.DataFrame:
    roles_path = os.path.join(DATA_DIR, "roles.csv")
    if not os.path.exists(roles_path):
        df["slot_rate"] = np.nan
        return df

    try:
        r = pd.read_csv(roles_path)
    except Exception:
        df["slot_rate"] = np.nan
        return df

    r.columns = [c.lower() for c in r.columns]
    if not {"team", "role"}.issubset(r.columns):
        df["slot_rate"] = np.nan
        return df

    r["team"] = r["team"].map(canon_team)
    wr = r[r["role"].astype(str).str.upper().isin(["WR1", "WR2", "WR3", "SLOT"])].copy()
    wr["is_slot"] = wr["role"].astype(str).str.upper().eq("SLOT").astype(int)
    rate = (wr.groupby("team", dropna=False)["is_slot"].sum() / wr.groupby("team", dropna=False).size()).rename("slot_rate").reset_index()
    rate = rate[rate["team"] != ""]
    out = df.merge(rate, on="team", how="left")
    return out


# -----------------------------
# Fallback sweep (Sharp FIRST) + prefer Sharp boxes
# -----------------------------
def _apply_fallback_enrichers(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "sharp_team_form.csv",        # Sharp first so box rates & tendencies seed the frame
        "espn_team_form.csv",
        "msf_team_form.csv",
        "apisports_team_form.csv",
        "nflgsis_team_form.csv",
    ]
    print("[make_team_form] Fallback order =", candidates)

    out = df.copy()
    for fn in candidates:
        print(f"[make_team_form] trying {fn} ...")
        try:
            ext_path = os.path.join(DATA_DIR, fn)
            ext = _read_csv_safe(ext_path)
            if _is_empty(ext):
                print(f"[make_team_form] {fn} empty or missing.")
                continue
            ext = _prep_team_key_and_rates(ext)
            cols_show = [c for c in ext.columns if c in ["team","team_abbr","light_box_rate","heavy_box_rate","airyardsatt","air_yards_per_att"]]
            sample = ext["team"].head(1).to_list() if "team" in ext.columns else "NA"
            print(f"[make_team_form] {fn} rows={len(ext)} cols={cols_show} sample_team={sample}")
            if "team" in ext.columns:
                out = _non_destructive_team_merge(out, ext)
            else:
                print(f"[make_team_form] skip {fn}: no team key", file=sys.stderr)
        except Exception as e:
            print(f"[make_team_form] enrich {fn} error: {e}", file=sys.stderr)
            continue

    # hard preference for Sharp box rates when present
    out = _prefer_boxes_from_sharp(out)
    return out


def _prefer_boxes_from_sharp(base: pd.DataFrame) -> pd.DataFrame:
    shp_path = os.path.join(DATA_DIR, "sharp_team_form.csv")
    shp = _read_csv_safe(shp_path)
    if _is_empty(shp):
        return base
    shp = _prep_team_key_and_rates(shp)
    cols = [c for c in ["team", "light_box_rate", "heavy_box_rate"] if c in shp.columns]
    if len(cols) < 2:
        return base
    shp = shp[cols].drop_duplicates()
    out = base.merge(shp, on="team", how="left", suffixes=("", "_sharp"))
    for c in ["light_box_rate", "heavy_box_rate"]:
        sc = f"{c}_sharp"
        if sc in out.columns:
            out[c] = np.where(out[sc].notna(), out[sc], out.get(c))
            out.drop(columns=[sc], inplace=True)
    return out


# -----------------------------
# Main builder
# -----------------------------
def build_team_form(season: int, box_backfill_prev: bool = False) -> pd.DataFrame:
    print(f"[make_team_form] Loading PBP for {season} via {NFL_PKG} ...")
    pbp = load_pbp(season)
    if _is_empty(pbp):
        print("[make_team_form] WARNING: PBP feed returned no rows; building skeleton then relying on fallbacks.")
        skeleton = pd.DataFrame({"team": sorted(VALID)})
        skeleton["season"] = int(season)
        skeleton["games_played"] = np.nan
        for col in [
            "def_pass_epa", "def_rush_epa", "def_sack_rate",
            "pace", "pass_rate_over_expected", "proe", "rz_rate", "12p_rate", "slot_rate", "ay_per_att",
            "light_box_rate", "heavy_box_rate",
        ]:
            skeleton[col] = np.nan
        skeleton["team_abbr"] = skeleton["team"]
        return skeleton

    proe_df = _compute_team_proe_from_pbp(pbp)

    print("[make_team_form] Computing defensive EPA & sack rate ...")
    def_tbl = compute_def_epa_and_sacks(pbp)

    print("[make_team_form] Computing pace & PROE ...")
    pace_tbl = compute_pace_and_proe(pbp)

    print("[make_team_form] Computing RZ rate & air yards per att ...")
    rz_ay_tbl = compute_red_zone_and_airyards(pbp)

    print("[make_team_form] Loading participation/personnel (optional) ...")
    part = load_participation(season)

    # (Optional) backfill the current season's box counts from prior season if participation is empty
    part_for_box = part
    if (part is None or part.empty) and box_backfill_prev:
        prev = season - 1
        try:
            prev_part = load_participation(prev)
            if prev_part is not None and not prev_part.empty:
                print(f"[make_team_form] participation {season} empty; backfilling LIGHT/HEAVY box from {prev}")
                part_for_box = prev_part
        except Exception:
            pass

    pers_tbl = compute_personnel_rates(pbp, part_for_box)

    print("[make_team_form] Merging components ...")
    # Canonicalize keys BEFORE merge
    for t in (def_tbl, pace_tbl, rz_ay_tbl, pers_tbl):
        if not _is_empty(t) and "team" in t.columns:
            t["team"] = t["team"].map(canon_team)

    def_tbl = def_tbl[def_tbl["team"] != ""]
    pace_tbl = pace_tbl[pace_tbl["team"] != ""]
    rz_ay_tbl = rz_ay_tbl[rz_ay_tbl["team"] != ""]
    pers_tbl = pers_tbl[pers_tbl["team"] != ""]

    out = def_tbl.merge(pace_tbl, on="team", how="left") \
                 .merge(rz_ay_tbl, on="team", how="left") \
                 .merge(pers_tbl, on="team", how="left")

    out["team_abbr"] = out["team"]
    if not proe_df.empty:
        out = out.merge(proe_df, on="team", how="left")

    out = merge_slot_rate_from_roles(out)
    out["season"] = int(season)

    z_cols = ["def_pass_epa","def_rush_epa","def_sack_rate","pace_neutral","proe",
              "rz_rate","personnel_12_rate","slot_rate","ay_per_att","light_box_rate","heavy_box_rate",
              "success_rate_off","success_rate_def","success_rate_diff","explosive_play_rate_allowed"]
    out = zscore(out, z_cols)

    out = out.rename(columns={"pace_neutral": "pace", "personnel_12_rate": "12p_rate"})
    if "pass_rate_over_expected" not in out.columns:
        out["pass_rate_over_expected"] = np.nan
    for need in ["rz_rate", "12p_rate", "slot_rate", "ay_per_att", "light_box_rate", "heavy_box_rate"]:
        if need not in out.columns:
            out[need] = np.nan

    cols_first = ["team","team_abbr","season","games_played",
                  "def_pass_epa","def_rush_epa","def_sack_rate",
                  "pace","pass_rate_over_expected","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
                  "light_box_rate","heavy_box_rate"]
    ordered = [c for c in cols_first if c in out.columns] + [c for c in out.columns if c not in cols_first]
    out = out[ordered].sort_values(["team"]).reset_index(drop=True)
    return out


def _validate_required(df: pd.DataFrame, allow_missing_box: bool = False):
    required = ["def_pass_epa","def_rush_epa","def_sack_rate","pace","proe","rz_rate","ay_per_att"]
    box_cols = ["light_box_rate","heavy_box_rate"]
    if not allow_missing_box:
        required += box_cols

    missing = {}
    for col in required:
        if col not in df.columns:
            missing[col] = "MISSING COLUMN"
        else:
            bad = df[df[col].isna()]["team"].tolist()
            if bad:
                missing[col] = bad

    if missing:
        print("[make_team_form] REQUIRED METRICS MISSING:", file=sys.stderr)
        for k, v in missing.items():
            print(f"  - {k}: {v}", file=sys.stderr)
        raise RuntimeError("Required team_form metrics missing; failing per strict policy.")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--allow-missing-box", action="store_true",
                        help="Do not fail if light/heavy box rates are missing (participation/merge unavailable).")
    parser.add_argument("--box-backfill-prev", action="store_true",
                        help="If current-season participation is empty, backfill ONLY box counts from prior season.")
    args = parser.parse_args()

    df_to_write: pd.DataFrame | None = None
    success = False
    team_form: pd.DataFrame | None = None

    try:
        df = build_team_form(args.season, box_backfill_prev=args.box_backfill_prev)

        # Fallback sweep BEFORE strict validation (includes Sharp-first merge)
        before = df.copy()
        df = _apply_fallback_enrichers(df)
        df = _ensure_plays_est_column(df)
        team_form = df.copy()

        # === Merge Sharp Football summary ===
        sharp_path = Path("data") / "sharp_team_form.csv"
        if not sharp_path.exists():
            raise RuntimeError("[make_team_form] sharp_team_form.csv missing. sharpfootball_pull.py must run BEFORE make_team_form.py in the workflow.")

        sharp_df = pd.read_csv(sharp_path)
        if sharp_df.empty:
            raise RuntimeError("[make_team_form] sharp_team_form.csv is empty. Sharp pull failed or returned no usable rows.")

        # Normalize join keys
        if "team" not in sharp_df.columns:
            if "team_abbr" in sharp_df.columns:
                sharp_df["team"] = sharp_df["team_abbr"].astype(str)
            else:
                raise RuntimeError("[make_team_form] sharp_team_form.csv has no team column")

        sharp_df["team"] = sharp_df["team"].astype(str).str.upper().str.strip().map(canon_team)
        sharp_df = sharp_df[sharp_df["team"].isin(VALID)]
        sharp_df = sharp_df.loc[:, ~sharp_df.columns.duplicated()]
        if "team_abbr" in sharp_df.columns:
            sharp_df.drop(columns=["team_abbr"], inplace=True)

        if team_form is None:
            team_form = df.copy()
        if "team" not in team_form.columns:
            if "team_abbr" in team_form.columns:
                team_form["team"] = team_form["team_abbr"]
            elif "defteam" in team_form.columns:
                team_form["team"] = team_form["defteam"]
            elif "off_team" in team_form.columns:
                team_form["team"] = team_form["off_team"]
            else:
                raise RuntimeError("[make_team_form] cannot infer team key for merge")

        team_form["team"] = team_form["team"].astype(str).str.upper().str.strip().map(canon_team)
        team_form = team_form[team_form["team"].isin(VALID)]
        team_form["team_abbr"] = team_form["team"]

        sharp_df = sharp_df.loc[:, ~sharp_df.columns.duplicated()]
        team_form = team_form.loc[:, ~team_form.columns.duplicated()]

        team_form = team_form.merge(
            sharp_df,
            on="team",
            how="left",
            suffixes=("", "_sharp")
        )

        if "team_abbr_sharp" in team_form.columns:
            team_form.drop(columns=["team_abbr_sharp"], inplace=True)

        if "pass_rate_over_expected_sharp" in team_form.columns:
            if "pass_rate_over_expected" in team_form.columns:
                team_form["pass_rate_over_expected"] = team_form["pass_rate_over_expected"].where(
                    team_form["pass_rate_over_expected"].notna(),
                    team_form["pass_rate_over_expected_sharp"]
                )
            else:
                team_form.rename(columns={"pass_rate_over_expected_sharp": "pass_rate_over_expected"}, inplace=True)
            if "pass_rate_over_expected_sharp" in team_form.columns:
                team_form.drop(columns=["pass_rate_over_expected_sharp"], inplace=True)

        # Basic sanity: pick a few must-have Sharp stats and make sure at least
        # one of them actually populated for at least one team.
        required_sharp_cols = [
            "pass_rate_over_expected",
            "neutral_pace",
            "coverage_man_rate",
            "coverage_zone_rate",
        ]
        team_key = "team" if "team" in team_form.columns else "team_abbr"
        for col in required_sharp_cols:
            if col not in team_form.columns:
                raise RuntimeError(f"[make_team_form] sharp merge missing {col} (column absent)")

            mask = team_form[col].isna()
            if mask.all():
                missing_teams = sorted(team_form.loc[mask, team_key].dropna().unique().tolist())
                raise RuntimeError(
                    f"[make_team_form] sharp merge missing {col} for teams {missing_teams}"
                )

        # Log what got filled by fallbacks (for quick triage)
        try:
            track_cols = [
                "def_pass_epa","def_rush_epa","def_sack_rate","pace","proe",
                "rz_rate","12p_rate","slot_rate","ay_per_att","light_box_rate","heavy_box_rate",
                "seconds_per_play","seconds_per_play_last5","plays_per_game",
                "neutral_db_rate","neutral_db_rate_last_5",
                "motion_rate","play_action_rate","shotgun_rate","no_huddle_rate",
                "blitz_rate","sub_package_rate",
                "dl_pressure_rate","dl_no_blitz_pressure_rate","dl_ybc_per_rush","dl_stuff_rate",
                "ypt_allowed","wr_ypt_allowed","te_ypt_allowed","rb_ypt_allowed","outside_ypt_allowed","slot_ypt_allowed",
            ]
            filled_counts = {}
            for col in track_cols:
                if team_form is not None and col in team_form.columns:
                    was_na = before[col].isna() if col in before.columns else pd.Series(True, index=team_form.index)
                    now_ok = was_na & team_form[col].notna()
                    if now_ok.any():
                        filled_counts[col] = int(now_ok.sum())
            if filled_counts:
                print("[make_team_form] Fallback fills (non-null adds):", filled_counts)
        except Exception:
            pass

        # Strict validation (AFTER fallback)
        if team_form is None:
            team_form = df.copy()
        _validate_required(team_form, allow_missing_box=args.allow_missing_box)
        df_to_write = team_form
        success = True

    except Exception as e:
        print(f"[make_team_form] ERROR: {e}", file=sys.stderr)
        if isinstance(team_form, pd.DataFrame):
            df_to_write = team_form
        elif "df" in locals() and isinstance(df, pd.DataFrame):
            df_to_write = df
        else:
            df_to_write = pd.DataFrame(columns=TEAM_FORM_OUTPUT_COLUMNS)

    if isinstance(df_to_write, pd.DataFrame):
        df = df_to_write.copy()

        # Ensure pace fields are present and consistent
        neutral = df.get("pace_neutral")
        if neutral is None:
            neutral = df.get("neutral_pace")
        if neutral is None:
            neutral = df.get("pace")
        df["neutral_pace"] = neutral if neutral is not None else np.nan
        df["pace"] = df["neutral_pace"]

        # normalize proe numeric
        if "proe" in df.columns:
            df["proe"] = pd.to_numeric(df["proe"], errors="coerce")

        # build neutral_pace_score using Sharp tempo data if available
        if "seconds_per_play" in df.columns and "plays_per_game" in df.columns:
            tempo = pd.to_numeric(df["seconds_per_play"], errors="coerce")
            volume = pd.to_numeric(df["plays_per_game"], errors="coerce")

            if tempo.notna().sum() and volume.notna().sum():
                tempo_range = tempo.max() - tempo.min()
                volume_range = volume.max() - volume.min()

                tempo_norm = (
                    (tempo.max() - tempo) / tempo_range
                    if tempo_range and not np.isclose(tempo_range, 0)
                    else pd.Series(np.nan, index=df.index)
                )
                volume_norm = (
                    (volume - volume.min()) / volume_range
                    if volume_range and not np.isclose(volume_range, 0)
                    else pd.Series(np.nan, index=df.index)
                )
                combined = 0.7 * tempo_norm + 0.3 * volume_norm
                df["neutral_pace_score"] = combined
            else:
                df["neutral_pace_score"] = np.nan
        else:
            df["neutral_pace_score"] = np.nan

        df_to_write = df

        # --- NEW: derive Success Rates + Explosive Play Rate Allowed from nflverse PBP ---
        try:
            season = 2025
            pbp = NFLV.import_pbp_data([season])  # nflreadpy or nfl_data_py via _import_nflverse()
            # Basic filters: no penalties-only rows
            pbp = pbp[pbp["play_type"].isin(["pass","run"])]
            # Success definition (common): EPA > 0
            pbp["is_success"] = (pbp["epa"] > 0).astype(int)
            # Explosive threshold: >=15 pass or >=10 rush (ceiling indicator)
            pbp["is_explosive"] = (
                ((pbp["play_type"] == "pass") & (pbp["yards_gained"] >= 15)) |
                ((pbp["play_type"] == "run") & (pbp["yards_gained"] >= 10))
            ).astype(int)

            # Map team abbreviations to your canonical codes
            if "posteam" in pbp:
                pbp["off_team"] = pbp["posteam"].astype(str).str.upper()
            else:
                pbp["off_team"] = pd.NA
            if "defteam" in pbp:
                pbp["def_team"] = pbp["defteam"].astype(str).str.upper()
            else:
                pbp["def_team"] = pd.NA

            # Aggregate per-team
            off_grp = pbp.groupby("off_team", dropna=True, as_index=False).agg(
                off_plays=("is_success","size"),
                off_success=("is_success","sum"),
            )
            off_grp["success_rate_off"] = off_grp["off_success"] / off_grp["off_plays"]

            def_grp = pbp.groupby("def_team", dropna=True, as_index=False).agg(
                def_plays=("is_success","size"),
                def_success_allowed=("is_success","sum"),
                def_explosive_allowed=("is_explosive","sum"),
            )
            def_grp["success_rate_def"] = def_grp["def_success_allowed"] / def_grp["def_plays"]
            def_grp["explosive_play_rate_allowed"] = def_grp["def_explosive_allowed"] / def_grp["def_plays"]

            # Join into existing team frame (we assume `df` holds the team rows by this point)
            # Ensure `df` has column 'team' canonicalized (e.g., "KC","PHI", etc.)
            df = df.copy()
            df["team"] = df["team"].astype(str).str.upper()

            df = df.merge(off_grp[["off_team","success_rate_off"]], left_on="team", right_on="off_team", how="left") \
                   .drop(columns=["off_team"], errors="ignore")
            df = df.merge(def_grp[["def_team","success_rate_def","explosive_play_rate_allowed"]],
                          left_on="team", right_on="def_team", how="left") \
                   .drop(columns=["def_team"], errors="ignore")

            df["success_rate_diff"] = df["success_rate_off"] - df["success_rate_def"]

            # Z-scores for the new metrics (only when enough teams present)
            df = zscore(df, ["success_rate_off","success_rate_def","success_rate_diff","explosive_play_rate_allowed"])
        except Exception as e:
            print(f"[make_team_form] WARNING: SRD/EPR derivation failed: {e}")

        df_to_write = df

    written_df = _write_team_form_csv(df_to_write, success=success)

    if success:
        # CI log: ensure we actually captured Sharp box rates
        try:
            for col in ["light_box_rate","heavy_box_rate"]:
                nn = int(written_df[col].notna().sum()) if col in written_df.columns else 0
                print(f"[make_team_form] {col}: non-null teams = {nn}/32")
        except Exception:
            pass
    else:
        sys.exit(1)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
