# scripts/make_team_form.py
"""
Build team-level context features for pricing & modeling.

Outputs: data/team_form.csv

Columns written (many with *_z standardized versions):
- team
- season
- games_played
- def_pass_epa
- def_rush_epa
- def_sack_rate
- pace_neutral              # seconds per snap in neutral situations
- proe                      # pass rate over expectation
- rz_rate                   # share of offensive plays run inside opp 20
- personnel_12_rate         # 12 personnel usage rate (offense)
- slot_rate                 # proxy from roles.csv (optional)
- ay_per_att                # air yards per pass attempt (offense)
- light_box_rate            # % snaps vs light boxes (if participation present)
- heavy_box_rate            # % snaps vs heavy boxes (if participation present)

Safe behavior:
- If a source is missing (e.g., participation or air yards), we write NaN and proceed.
- Z-scores only computed for columns that exist and have >= 8 non-null values.

Usage:
    python scripts/make_team_form.py --season 2025
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import List, Tuple

import pandas as pd
import numpy as np

# -----------------------------
# Imports: prefer nflreadpy
# -----------------------------
def _import_nflverse():
    try:
        import nflreadpy as nflv  # Python port maintained by nflverse
        return nflv, "nflreadpy"
    except ModuleNotFoundError:
        print(
            "[make_team_form] ⚠️ nflreadpy missing; falling back to nfl_data_py. "
            "Install nflreadpy to ensure 2025 data is available.",
            file=sys.stderr,
        )
    except ImportError as exc:
        if "original_mlq" in str(exc):
            print(
                "[make_team_form] ⚠️ nflreadpy requires nfl_data_py>=0.3.4 (missing "
                "`original_mlq`); upgrade the dependency to use nflreadpy. Falling "
                "back to nfl_data_py for now.",
                file=sys.stderr,
            )
        else:
            print(
                f"[make_team_form] ⚠️ nflreadpy import failed ({exc}); falling back to nfl_data_py.",
                file=sys.stderr,
            )
    except Exception as exc:
        print(
            f"[make_team_form] ⚠️ nflreadpy import failed ({exc}); falling back to nfl_data_py.",
            file=sys.stderr,
        )
    try:
        # Fallback for some environments; limited and deprecated upstream
        import nfl_data_py as nflv  # type: ignore
        return nflv, "nfl_data_py"
    except Exception as e:
        raise RuntimeError(
            "Neither nflreadpy nor nfl_data_py is available. "
            "Please `pip install nflreadpy`."
        ) from e


NFLV, NFL_PKG = _import_nflverse()

DATA_DIR = "data"
OUTPATH = os.path.join(DATA_DIR, "team_form.csv")
_CACHE_DIRS = [
    DATA_DIR,
    os.path.join("external", "nflverse_bundle"),
]

# -----------------------------
# Helpers
# -----------------------------

def _safe_mkdir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _load_cached_csv(kind: str, season: int) -> Tuple[pd.DataFrame, str]:
    """Return a cached nflverse export if one exists on disk."""
    for base in _CACHE_DIRS:
        path = os.path.join(base, f"{kind}_{season}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df.columns = [c.lower() for c in df.columns]
        return df, path
    return pd.DataFrame(), ""


def _validate_season(df: pd.DataFrame, season: int, label: str) -> None:
    """Ensure a dataframe only contains the requested ``season``."""
    if df.empty or "season" not in df.columns:
        return
    try:
        seasons = (
            pd.to_numeric(df["season"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
    except Exception:
        return
    if len(seasons) == 0:
        return
    if any(int(s) != int(season) for s in seasons):
        raise RuntimeError(
            f"{label} spans seasons {sorted(map(int, seasons))}; expected {season}"
        )


def zscore(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df and df[c].notna().sum() >= 8:
            m = df[c].mean()
            s = df[c].std(ddof=0)
            if s and not np.isclose(s, 0):
                df[c + "_z"] = (df[c] - m) / s
            else:
                df[c + "_z"] = np.nan
    return df


def _neutral_mask(pbp: pd.DataFrame) -> pd.Series:
    """
    Neutral situation mask:
    - score differential between -7 and +7
    - win probability between 0.2 and 0.8 (if available)
    - quarter <= 3 (avoid 4Q two-minute drill skew)
    """
    m = pd.Series(True, index=pbp.index)
    if "score_differential" in pbp:
        m &= pbp["score_differential"].between(-7, 7)
    if "wp" in pbp:
        m &= pbp["wp"].between(0.2, 0.8)
    if "qtr" in pbp:
        m &= pbp["qtr"] <= 3
    return m


def safe_div(n, d):
    n = n.astype(float)
    d = d.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(d == 0, np.nan, n / d)
    return out


def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path): 
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def _non_destructive_team_merge(base: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    """
    Merge team-level enrichers without overwriting existing non-null values.
    Fills only the following columns when they are missing in `base`.
    """
    if add.empty or "team" not in add.columns:
        return base
    add = add.copy()
    add.columns = [c.lower() for c in add.columns]
    keep = [c for c in [
        "team","pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
        "def_pass_epa","def_rush_epa","def_sack_rate","light_box_rate","heavy_box_rate"
    ] if c in add.columns]
    if not keep:
        return base
    add = add[keep].drop_duplicates()
    out = base.merge(add, on="team", how="left", suffixes=("","_ext"))
    for c in keep:
        if c == "team": 
            continue
        ext = f"{c}_ext"
        if ext in out.columns:
            out[c] = out[c].combine_first(out[ext])
            out.drop(columns=[ext], inplace=True)
    return out

# --- NEW: external enricher discovery + normalization -----------------------


def _team_enricher_paths() -> list[str]:
    """Return candidate CSV filenames that may contain team context enrichers."""
    return [
        "espn_team_form.csv",
        "espn_team.csv",
        "msf_team_form.csv",
        "msf_team.csv",
        "apisports_team_form.csv",
        "apisports_team.csv",
        "nflgsis_team_form.csv",
        "gsis_team.csv",
        "pfr_team_enrich.csv",
        "team.form.csv",
    ]


_TEAM_COLUMN_ALIASES = {
    "team_abbr": "team",
    "team_code": "team",
    "team_name": "team",
    "club": "team",
    "pace_seconds": "pace",
    "sec_per_play": "pace",
    "seconds_per_play": "pace",
    "pass_rate_over_expected": "proe",
    "proe_pct": "proe",
    "red_zone_rate": "rz_rate",
    "rz_pct": "rz_rate",
    "twelve_personnel_rate": "12p_rate",
    "personnel_12_pct": "12p_rate",
    "slot_usage_rate": "slot_rate",
    "slot_pct": "slot_rate",
    "air_yards_per_attempt": "ay_per_att",
    "ay_per_att_avg": "ay_per_att",
    "light_box_pct": "light_box_rate",
    "heavy_box_pct": "heavy_box_rate",
    "def_pass_epa_per_play": "def_pass_epa",
    "def_rush_epa_per_play": "def_rush_epa",
    "sack_rate": "def_sack_rate",
    "def_sack_pct": "def_sack_rate",
}


def _standardize_team_enricher(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    for src, dest in _TEAM_COLUMN_ALIASES.items():
        if src in df.columns and dest not in df.columns:
            df = df.rename(columns={src: dest})

    if "team" not in df.columns:
        for cand in ["team", "team_abbreviation", "abbr"]:
            if cand in df.columns:
                df["team"] = df[cand]
                break

    if "team" not in df.columns:
        return pd.DataFrame()

    df["team"] = df["team"].astype(str).str.upper().str.strip()

    keep = [
        c
        for c in [
            "team",
            "pace",
            "proe",
            "rz_rate",
            "12p_rate",
            "slot_rate",
            "ay_per_att",
            "def_pass_epa",
            "def_rush_epa",
            "def_sack_rate",
            "light_box_rate",
            "heavy_box_rate",
        ]
        if c in df.columns
    ]
    if not keep:
        return pd.DataFrame()
    return df[keep].drop_duplicates()

# --- ADD: helper to roll up weekly pace/proe (plays-weighted), non-destructive ---
def _rollup_weekly_pace_proe(df: pd.DataFrame) -> pd.DataFrame:
    try:
        tfw_path = os.path.join(DATA_DIR, "team_form_weekly.csv")
        if os.path.exists(tfw_path):
            tfw = pd.read_csv(tfw_path)
            tfw.columns = [c.lower() for c in tfw.columns]
            if {"team","plays_est","pace","proe"}.issubset(tfw.columns) and not tfw.empty:
                grp = tfw.groupby("team", dropna=False)
                pace_w = grp.apply(lambda g: np.average(g["pace"], weights=np.clip(g["plays_est"], 1, None))).rename("pace_season")
                proe_w = grp.apply(lambda g: np.average(g["proe"], weights=np.clip(g["plays_est"], 1, None))).rename("proe_season")
                roll = pd.concat([pace_w, proe_w], axis=1).reset_index()
                df = df.merge(roll, on="team", how="left")
                if "pace" in df.columns and "pace_season" in df.columns:
                    df["pace"] = df["pace"].where(df["pace"].notna(), df["pace_season"])
                if "proe" in df.columns and "proe_season" in df.columns:
                    df["proe"] = df["proe"].where(df["proe"].notna(), df["proe_season"])
                df.drop(columns=[c for c in ["pace_season","proe_season"] if c in df.columns], inplace=True)
    except Exception:
        pass
    return df
# --- END ADD ---

# -----------------------------
# Loaders (abstract across libs)
# -----------------------------

def load_pbp(season: int) -> pd.DataFrame:
    """
    Load regular-season PBP for given season.
    nflreadpy: NFLV.load_pbp(seasons=[season])
    nfl_data_py: nfl_data_py.import_pbp_data([season], downcast=True)
    """
    cached, source_path = _load_cached_csv("pbp", season)
    if not cached.empty:
        print(f"[make_team_form] ℹ️ Using cached pbp_{season}.csv from {source_path}")
        return cached

    if NFL_PKG == "nflreadpy":
        pbp = NFLV.load_pbp(seasons=[season], season_type="REG")
    else:
        try:
            pbp = NFLV.import_pbp_data([season], downcast=True, season_type="REG")  # type: ignore
        except TypeError:  # older nfl_data_py builds
            pbp = NFLV.import_pbp_data([season], downcast=True)  # type: ignore
    # Normalize column names we rely on
    pbp.columns = [c.lower() for c in pbp.columns]
    return pbp


def load_participation(season: int) -> pd.DataFrame:
    """
    Participation / personnel (optional). nflreadpy exposes load_participation()
    If not available, return empty DF.
    """
    cached, source_path = _load_cached_csv("participation", season)
    if not cached.empty:
        print(
            f"[make_team_form] ℹ️ Using cached participation_{season}.csv from {source_path}"
        )
        return cached
    try:
        if NFL_PKG == "nflreadpy":
            part = NFLV.load_participation(seasons=[season], season_type="REG")
        else:
            try:
                part = NFLV.import_participation([season])  # type: ignore[attr-defined]
            except Exception:
                return pd.DataFrame()
        part.columns = [c.lower() for c in part.columns]
        _validate_season(part, season, "participation cache")
        return part
    except Exception:
        return pd.DataFrame()


def _load_required_pbp(season: int) -> tuple[pd.DataFrame, int]:
    """Load play-by-play strictly for ``season`` or raise."""
    cached, cache_path = _load_cached_csv("pbp", season)
    if not cached.empty:
        _validate_season(cached, season, "cached pbp")
        print(
            f"[make_team_form] ℹ️ Loaded cached pbp_{season}.csv from {cache_path}"
        )
        return cached, season

    errors: list[str] = []
    try:
        pbp = load_pbp(season)
    except Exception as err:
        errors.append(f"{type(err).__name__}: {err}")
    else:
        if not pbp.empty:
            _validate_season(pbp, season, "pbp feed")
        errors.append(str(err))
    else:
        if not pbp.empty:
            return pbp, season
        errors.append("empty dataframe")

    raise RuntimeError(
        "PBP unavailable for requested season. "
def _load_pbp_with_fallback(
    season: int,
    *,
    allow_prior_seasons: bool,
    max_lookback: int = 5,
) -> tuple[pd.DataFrame, int]:
    """
    Attempt to load play-by-play data for ``season``. If unavailable (future season
    or network restriction), optionally fall back to the most recent prior season
    that returns data. Returns the dataframe and the season actually used.

    When ``allow_prior_seasons`` is False, the function will raise if it can only
    source data from seasons earlier than the requested one – ensuring we never
    silently populate 2025 projections with 2024 (or older) metrics.
def _load_pbp_with_fallback(season: int, max_lookback: int = 5) -> tuple[pd.DataFrame, int]:
    """
    Attempt to load play-by-play data for ``season``. If unavailable (future season
    or network restriction), fall back to the most recent prior season that returns
    data. Returns the dataframe and the season actually used.
    """
    errors: list[str] = []
    for offset in range(0, max_lookback + 1):
        candidate = season - offset
        if candidate < 2000:
            break
        try:
            pbp = load_pbp(candidate)
        except Exception as err:
            errors.append(f"season {candidate}: {err}")
            continue
        if not pbp.empty:
            if candidate == season:
                return pbp, candidate

            if allow_prior_seasons:
                print(
                    f"[make_team_form] ⚠️ No PBP for {season}; using {candidate} as fallback"
                )
                return pbp, candidate

            errors.append(
                f"season {candidate}: available but fallback disabled"
            )
            continue
            if candidate != season:
                print(
                    f"[make_team_form] ⚠️ No PBP for {season}; using {candidate} as fallback"
                )
            return pbp, candidate
        errors.append(f"season {candidate}: empty dataframe")
    raise RuntimeError(
        "PBP unavailable for requested season and fallbacks. "
        + "; ".join(errors) if errors else ""
    )


def load_schedules(season: int) -> pd.DataFrame:
    try:
        if NFL_PKG == "nflreadpy":
            sch = NFLV.load_schedules(seasons=[season])
        else:
            sch = NFLV.import_schedules([season])  # type: ignore
        sch.columns = [c.lower() for c in sch.columns]
        return sch
    except Exception:
        return pd.DataFrame()


# -----------------------------
# Feature builders
# -----------------------------

def compute_def_epa_and_sacks(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive EPA splits and sack rate allowed by offense (defense perspective).
    We'll aggregate by defense team (defteam) when available, else by posteam/opponent mapping.
    """
    df = pbp.copy()
    # EPA columns
    if "epa" not in df:
        df["epa"] = np.nan

    # Identify pass vs rush
    is_pass = df.get("pass", pd.Series(False, index=df.index)).astype(bool)
    is_rush = df.get("rush", pd.Series(False, index=df.index)).astype(bool)

    # Defense team column differs by source; try both
    def_team_col = "defteam" if "defteam" in df else ("def_team" if "def_team" in df else None)
    off_team_col = "posteam" if "posteam" in df else ("offense_team" if "offense_team" in df else None)

    if def_team_col is None or off_team_col is None:
        # Best-effort: try to construct def team from matchup logic if missing
        # If we can't, gracefully return empty.
        g = pd.DataFrame()
        return g

    # Sacks: many data sets mark with sack == 1 (or qb_hit/sack columns)
    sack_flag = df.get("sack", pd.Series(0, index=df.index)).fillna(0).astype(int)
    dropbacks = df.get("qb_dropback", pd.Series(0, index=df.index)).fillna(0).astype(int)

    # Aggregate defensive splits
    grp = df.groupby(def_team_col, dropna=False)
    agg = pd.DataFrame({
        "def_pass_epa": grp.apply(lambda x: x.loc[is_pass.reindex(x.index, fill_value=False), "epa"].mean()),
        "def_rush_epa": grp.apply(lambda x: x.loc[is_rush.reindex(x.index, fill_value=False), "epa"].mean()),
        "def_sacks": grp[sack_flag.name].sum(),
        "opp_dropbacks": grp[dropbacks.name].sum(),
        "games_played": grp["game_id"].nunique() if "game_id" in df else grp.size()
    }).reset_index().rename(columns={def_team_col: "team"})

    agg["def_sack_rate"] = safe_div(agg["def_sacks"], agg["opp_dropbacks"])
    return agg


def compute_pace_and_proe(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Pace: seconds per snap in neutral situations (lower is faster).
    PROE: pass rate over expectation (needs xpass if present; otherwise heuristic).
    """
    df = pbp.copy()
    df.columns = [c.lower() for c in df.columns]

    # Seconds per play proxy: if 'game_seconds_remaining' + sequential plays exist,
    # fall back to plays per minute in neutral script.
    neutral = _neutral_mask(df)
    dfn = df.loc[neutral].copy()

    # Group by offense team (posteam)
    off_col = "posteam" if "posteam" in dfn else ("offense_team" if "offense_team" in dfn else None)
    if off_col is None:
        return pd.DataFrame()

    # Pace proxy: plays per minute → seconds per play
    dfn["_one"] = 1
    pace_grp = (
        dfn.groupby(off_col, dropna=False)["_one"]
        .sum()
        .rename("neutral_plays")
        .reset_index()
    )

    if {"game_seconds_remaining", "game_id"}.issubset(dfn.columns):
        sort_cols = [
            col
            for col in [off_col, "game_id", "qtr", "game_seconds_remaining", "play_id"]
            if col in dfn.columns
        ]
        ordered = dfn.sort_values(sort_cols, ignore_index=True)
        grp = ordered.groupby([off_col, "game_id"], dropna=False)[
            "game_seconds_remaining"
        ]
        # diff(-1) gives next - current; multiply by -1 to get positive elapsed seconds
        deltas = (grp.diff(-1) * -1).clip(lower=0)
        ordered["_sec_delta"] = deltas
        pace_seconds = (
            ordered.groupby(off_col, dropna=False)["_sec_delta"]
            .mean()
            .rename("pace_neutral")
            .reset_index()
        )
        pace_grp = pace_grp.merge(pace_seconds, on=off_col, how="left")
    else:
        pace_grp["pace_neutral"] = np.nan

    if "pace_neutral" in pace_grp.columns:
        pace_grp.loc[pace_grp["pace_neutral"].isna() & (pace_grp["neutral_plays"] > 0), "pace_neutral"] = 24.0

    pass_col = "pass" if "pass" in dfn.columns else None
    if pass_col is None:
        return pace_grp.rename(columns={off_col: "team"})[["team", "pace_neutral"]].assign(proe=np.nan)


    pass_col = "pass" if "pass" in dfn.columns else None
    if pass_col is None:
        return pace_grp.rename(columns={off_col: "team"})[["team", "pace_neutral"]].assign(proe=np.nan)


    pass_col = "pass" if "pass" in dfn.columns else None
    if pass_col is None:
        return pace_grp.rename(columns={off_col: "team"})[["team", "pace_neutral"]].assign(proe=np.nan)

    dfn["_is_pass"] = dfn[pass_col].astype(float)
    prate = (
        dfn.groupby(off_col, dropna=False)["_is_pass"]
        .mean()
        .rename("pass_rate")
    )

    if "xpass" in dfn.columns:
        xpass = dfn.groupby(off_col, dropna=False)["xpass"].mean()
        proe = prate - xpass
    elif "pass_probability" in dfn.columns:
        xp = dfn.groupby(off_col, dropna=False)["pass_probability"].mean()
        proe = prate - xp
    else:
        league_neutral_pass = prate.mean() if len(prate) else 0.55
        proe = prate - league_neutral_pass

    out = (
        pace_grp.merge(proe.rename("proe"), left_on=off_col, right_index=True, how="left")
        .rename(columns={off_col: "team"})
    )
    return out[["team", "pace_neutral", "proe"]]


def compute_red_zone_and_airyards(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    RZ rate (offense): share of offensive plays inside opp 20.
    Air yards per attempt (offense).
    """
    df = pbp.copy()
    df.columns = [c.lower() for c in df.columns]

    # choose offense team column
    off_col = "posteam" if "posteam" in df.columns else ("offense_team" if "offense_team" in df.columns else None)
    if off_col is None:
        return pd.DataFrame()

    # --- Red-zone rate (snaps inside opp 20) ---
    rz = (
        df.assign(yardline_100=pd.to_numeric(df.get("yardline_100"), errors="coerce"))
          .assign(rz_flag=lambda x: (x["yardline_100"] <= 20).astype(int))
          .groupby(off_col, dropna=False)["rz_flag"].mean()
          .rename("rz_rate")
    )

    # --- Air yards per attempt (passing plays only) ---
    is_pass = df.get("pass", pd.Series(False, index=df.index)).astype(bool)
    pass_df = df.loc[is_pass].copy()
    ay_per_att = (
        pass_df.assign(air_yards=pd.to_numeric(pass_df.get("air_yards"), errors="coerce"))
               .groupby(off_col, dropna=False)["air_yards"]
               .mean()
               .rename("ay_per_att")
    )

    # --- Combine outputs ---
    out = (
        pd.concat([rz, ay_per_att], axis=1)
          .reset_index()
          .rename(columns={off_col: "team"})
    )
    return out

def compute_personnel_rates(pbp: pd.DataFrame, participation: pd.DataFrame) -> pd.DataFrame:
    """
    Personnel usage (12 personnel) and defensive box counts from participation if available.
    Also writes light/heavy box rates if 'box' or 'men_in_box' like fields exist.

    Note: nflverse participation schema can vary; we attempt a best-effort merge.
    """
    # 12 personnel from PBP (offense): personnel_offense like "11", "12", etc.
    df = pbp.copy()
    off_col = "posteam" if "posteam" in df else ("offense_team" if "offense_team" in df else None)
    if off_col is None:
        base = pd.DataFrame(columns=["team", "personnel_12_rate"])
    else:
        per = df.get("personnel_offense", pd.Series(np.nan, index=df.index)).astype(str).str.extract(r"(\d\d)").rename(columns={0: "personnel"})
        df = df.assign(_per=per["personnel"])
        grp = df.groupby(off_col, dropna=False)
        total = grp.size().rename("plays_total").astype(float)
        p12 = grp.apply(lambda x: (x["_per"] == "12").mean() if len(x) else np.nan).rename("personnel_12_rate")
        base = pd.concat([total, p12], axis=1).reset_index().rename(columns={off_col: "team"})
        base = base[["team", "personnel_12_rate"]]

    # Box counts from participation (optional)
    light = heavy = None
    if not participation.empty:
        p = participation.copy()
        # Try common box fields
        box_col = None
        for cand in ["box", "men_in_box", "in_box", "defenders_in_box"]:
            if cand in p.columns:
                box_col = cand
                break
        team_col = None
        for cand in ["offense_team", "posteam", "team"]:
            if cand in p.columns:
                team_col = cand
                break
        if box_col and team_col:
            p["_light"] = (p[box_col].astype(float) <= 6).astype(float)
            p["_heavy"] = (p[box_col].astype(float) >= 8).astype(float)
            g = p.groupby(team_col, dropna=False)
            light = g["_light"].mean().rename("light_box_rate")
            heavy = g["_heavy"].mean().rename("heavy_box_rate")

    out = base.copy()
    if light is not None:
        out = out.merge(light.reset_index().rename(columns={light.index.name or "index": "team"}), on="team", how="left")
    if heavy is not None:
        out = out.merge(heavy.reset_index().rename(columns={heavy.index.name or "index": "team"}), on="team", how="left")

    return out


def merge_slot_rate_from_roles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: derive a team-level slot rate proxy from roles.csv (player-level slot usage).
    roles.csv columns expected: player, team, role (SLOT, WR1, WR2, ...)
    We count % of WR routes tagged as SLOT to estimate team slot lean.
    """
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

    # crude proxy: fraction of WR roles labeled SLOT among WR roles
    wr = r[r["role"].isin(["WR1", "WR2", "WR3", "SLOT", "slot", "Wr1", "Wr2", "Wr3"])].copy()
    wr["is_slot"] = wr["role"].astype(str).str.upper().eq("SLOT").astype(int)
    grp = wr.groupby("team", dropna=False)
    rate = (grp["is_slot"].sum() / grp.size()).rename("slot_rate").reset_index()
    out = df.merge(rate, on="team", how="left")
    return out


WEEKLY_COLUMNS = ["team", "week", "plays_est", "pace", "proe"]


def _empty_weekly_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=WEEKLY_COLUMNS)


def _write_weekly_stub(path: str) -> None:
    _empty_weekly_frame().to_csv(path, index=False)


def _write_weekly_outputs(
    pbp: pd.DataFrame,
    source_season: int,
    requested_season: int,
    path: str,
) -> None:
    """Persist the team/week pace + PROE table or a schema-only placeholder."""

    if pbp.empty:
        _write_weekly_stub(path)
        return

    try:
        w = pbp.copy()

        # neutral-ish filter to avoid garbage-time skew
        if "down" in w.columns:
            w = w[w["down"].isin([1, 2])]
        if "wp" in w.columns:
            w = w[w["wp"].between(0.2, 0.8, inclusive="both")]

        team_col = "posteam" if "posteam" in w.columns else (
            "offense_team" if "offense_team" in w.columns else None
        )
        if team_col is None or "week" not in w.columns:
            _write_weekly_stub(path)
            return

        # plays per team-week
        plays = (
            w.groupby([team_col, "week"], dropna=False)["play_id"].size().rename("plays_est")
        )

        # pace proxy: seconds between snaps if available, else NaN
        if "game_seconds_remaining" in w.columns:
            w = w.sort_values([team_col, "game_id", "qtr", "play_id"])
            gsr_diff = (
                w.groupby([team_col, "game_id"])["game_seconds_remaining"].diff(-1).abs()
            )
            w["gsr_diff"] = gsr_diff
            pace = (
                w.groupby([team_col, "week"], dropna=False)["gsr_diff"].mean().rename("pace")
            )
        else:
            pace = plays * 0 + np.nan

        # PROE: pass rate minus league weekly pass rate
        if "play_type" in w.columns:
            is_pass = w["play_type"].isin(["pass", "no_play"])
        else:
            is_pass = pd.Series(False, index=w.index)

        if len(is_pass) > 0:
            pass_frame = pd.DataFrame({
                "is_pass": is_pass,
                "team": w[team_col].values,
                "week": w["week"].values,
            })
            grouped = (
                pass_frame.groupby(["team", "week"], dropna=False)["is_pass"]
                .mean()
                .rename("pass_rate")
            )
            league = (
                pass_frame.groupby("week", dropna=False)["is_pass"]
                .mean()
                .rename("lg_pass_rate_week")
            )
            wk = pd.concat([plays, pace, grouped], axis=1).reset_index().rename(
                columns={team_col: "team"}
            )
            wk = wk.merge(league.reset_index(), on="week", how="left")
            wk["proe"] = wk["pass_rate"] - wk["lg_pass_rate_week"]
        else:
            wk = pd.concat([plays, pace], axis=1).reset_index().rename(
                columns={team_col: "team"}
            )
            wk["proe"] = np.nan

        wk_out = wk[["team", "week", "plays_est", "pace", "proe"]].copy()
        if source_season != requested_season:
            wk_out["source_season"] = source_season

        wk_out.to_csv(path, index=False)
    except Exception:
        _write_weekly_stub(path)


# -----------------------------
# Main builder
# -----------------------------

def build_team_form(season: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return team-form dataframe, the PBP used, and the source season."""
    print(f"[make_team_form] Loading PBP for {season} via {NFL_PKG} ...")
    pbp, source_season = _load_required_pbp(season)
    print(f"[make_team_form] Loading PBP for {season} via {NFL_PKG} ...")
    pbp, source_season = _load_required_pbp(season)
    print(f"[make_team_form] Loading PBP for {season} via {NFL_PKG} ...")
    pbp, source_season = _load_required_pbp(season)
    print(f"[make_team_form] Loading PBP for {season} via {NFL_PKG} ...")
    pbp, source_season = _load_required_pbp(season)
    print(f"[make_team_form] Loading PBP for {season} via {NFL_PKG} ...")
    pbp, source_season = _load_required_pbp(season)
    print(f"[make_team_form] Loading PBP for {season} via {NFL_PKG} ...")
    pbp, source_season = _load_required_pbp(season)
    print(f"[make_team_form] Loading PBP for {season} via {NFL_PKG} ...")
    pbp, source_season = _load_required_pbp(season)
    print(f"[make_team_form] Loading PBP for {season} via {NFL_PKG} ...")
    pbp, source_season = _load_required_pbp(season)
    print(f"[make_team_form] Loading PBP for {season} via {NFL_PKG} ...")
    pbp, source_season = _load_required_pbp(season)
def build_team_form(season: int, *, allow_fallback: bool) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return team-form dataframe, the PBP used, and the source season."""
    print(f"[make_team_form] Loading PBP for {season} via {NFL_PKG} ...")
    pbp, source_season = _load_pbp_with_fallback(
        season, allow_prior_seasons=allow_fallback
    )
def build_team_form(season: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return team-form dataframe, the PBP used, and the source season."""
    print(f"[make_team_form] Loading PBP for {season} via {NFL_PKG} ...")
    pbp, source_season = _load_pbp_with_fallback(season)
    if pbp.empty:
        raise RuntimeError("PBP is empty; cannot compute team form.")

    print("[make_team_form] Computing defensive EPA & sack rate ...")
    def_tbl = compute_def_epa_and_sacks(pbp)

    print("[make_team_form] Computing pace & PROE ...")
    pace_tbl = compute_pace_and_proe(pbp)

    print("[make_team_form] Computing RZ rate & air yards per att ...")
    rz_ay_tbl = compute_red_zone_and_airyards(pbp)

    print("[make_team_form] Loading participation/personnel (optional) ...")
    part = load_participation(source_season)
    pers_tbl = compute_personnel_rates(pbp, part)

    print("[make_team_form] Merging components ...")
    out = (
        def_tbl
        .merge(pace_tbl, on="team", how="left")
        .merge(rz_ay_tbl, on="team", how="left")
        .merge(pers_tbl, on="team", how="left")
    )

    # add slot rate proxy from roles.csv if present
    out = merge_slot_rate_from_roles(out)

    # attach season and enforce dtypes
    out["season"] = int(season)
    out["source_season"] = int(source_season)

    # Z-score useful continuous features
    z_cols = [
        "def_pass_epa",
        "def_rush_epa",
        "def_sack_rate",
        "pace_neutral",
        "proe",
        "rz_rate",
        "personnel_12_rate",
        "slot_rate",
        "ay_per_att",
        "light_box_rate",
        "heavy_box_rate",
    ]
    out = zscore(out, z_cols)

    # rename to match your existing naming expectations where helpful
    out = out.rename(columns={
        "pace_neutral": "pace",
        "personnel_12_rate": "12p_rate"
    })

    # Sort and reset
    cols_first = ["team", "season", "source_season", "games_played",
                  "def_pass_epa", "def_rush_epa", "def_sack_rate",
                  "pace", "proe", "rz_rate", "12p_rate", "slot_rate", "ay_per_att",
                  "light_box_rate", "heavy_box_rate"]
    # preserve if missing
    ordered = [c for c in cols_first if c in out.columns] + [c for c in out.columns if c not in cols_first]
    out = out[ordered].sort_values(["team"]).reset_index(drop=True)

    return out, pbp, source_season


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Permit using prior seasons when the requested season is unavailable",
    )
    args = parser.parse_args()

    _safe_mkdir(DATA_DIR)

    success = True
    pbp_used = pd.DataFrame()
    source_season = args.season

    try:
        df, pbp_used, source_season = build_team_form(
            args.season
        )
            args.season, allow_fallback=args.allow_fallback
        )
        df, pbp_used, source_season = build_team_form(args.season)
        if source_season != args.season:
            print(
                f"[make_team_form] ℹ️ Using {source_season} metrics as proxy for {args.season}"
            )

        # --- ADD: plays-weighted season roll-up of weekly pace/proe (non-destructive) ---
        df = _rollup_weekly_pace_proe(df)
        # --- END ADD ---

        # --- Weekly writer (persist results or an empty schema) ---
        weekly_path = os.path.join(DATA_DIR, "team_form_weekly.csv")
        _write_weekly_outputs(pbp_used, source_season, args.season, weekly_path)
        # --- Weekly writer (your original logic, kept verbatim) ---
        try:
            pbp = pbp_used.copy()
            if not pbp.empty:
                w = pbp.copy()
                # neutral-ish filter to avoid garbage-time skew
                if 'down' in w.columns:
                    w = w[w['down'].isin([1, 2])]
                if 'wp' in w.columns:
                    w = w[w['wp'].between(0.2, 0.8, inclusive='both')]

                # offense team column available in pbp
                team_col = 'posteam' if 'posteam' in w.columns else ('offense_team' if 'offense_team' in w.columns else None)
                if team_col is not None and 'week' in w.columns:
                    # plays per team-week
                    plays = w.groupby([team_col, 'week'], dropna=False)['play_id'].size().rename('plays_est')

                    # pace proxy: seconds between snaps if available, else NaN
                    if 'game_seconds_remaining' in w.columns:
                        w = w.sort_values([team_col, 'game_id', 'qtr', 'play_id'])
                        w['gsr_diff'] = w.groupby([team_col, 'game_id'])['game_seconds_remaining'].diff(-1).abs()
                        pace = w.groupby([team_col, 'week'])['gsr_diff'].mean().rename('pace')
                    else:
                        pace = plays * 0 + np.nan

                    # PROE: pass rate minus league weekly pass rate
                    is_pass = w['play_type'].isin(['pass', 'no_play']) if 'play_type' in w.columns else pd.Series(False, index=w.index)
                    if len(is_pass) > 0:
                        pr = is_pass.groupby([w[team_col], w['week']]).mean().rename('pass_rate')
                        lg = is_pass.groupby(w['week']).mean().rename('lg_pass_rate_week')
                        wk = pd.concat([plays, pace, pr], axis=1).reset_index().rename(columns={team_col: 'team'})
                        wk = wk.merge(lg.reset_index(), on='week', how='left')
                        wk['proe'] = wk['pass_rate'] - wk['lg_pass_rate_week']
                    else:
                        wk = pd.concat([plays, pace], axis=1).reset_index().rename(columns={team_col: 'team'})
                        wk['proe'] = np.nan

                    wk_out = wk[['team', 'week', 'plays_est', 'pace', 'proe']].copy()
                    if source_season != args.season:
                        wk_out['source_season'] = source_season
                    wk_out.to_csv(os.path.join(DATA_DIR, 'team_form_weekly.csv'), index=False)
                else:
                    # schema-only so downstream won’t crash if weekly isn’t computable
                    pd.DataFrame(columns=['team', 'week', 'plays_est', 'pace', 'proe']).to_csv(os.path.join(DATA_DIR, 'team_form_weekly.csv'), index=False)
            else:
                pd.DataFrame(columns=['team', 'week', 'plays_est', 'pace', 'proe']).to_csv(os.path.join(DATA_DIR, 'team_form_weekly.csv'), index=False)
        except Exception:
            # never fail the run on weekly export
            try:
                pd.DataFrame(columns=['team', 'week', 'plays_est', 'pace', 'proe']).to_csv(os.path.join(DATA_DIR, 'team_form_weekly.csv'), index=False)
            except Exception:
                pass
        # --- END Weekly writer ---

    except Exception as e:
        print(f"[make_team_form] ERROR: {e}", file=sys.stderr)
        success = False
        df = pd.DataFrame(columns=[
            "team","season","source_season","games_played","def_pass_epa","def_rush_epa","def_sack_rate",
            "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
            "light_box_rate","heavy_box_rate"
        ])
        pbp_used = pd.DataFrame()
        source_season = args.season
        _write_weekly_outputs(pd.DataFrame(), args.season, args.season, os.path.join(DATA_DIR, "team_form_weekly.csv"))
        try:
            pd.DataFrame(columns=['team', 'week', 'plays_est', 'pace', 'proe']).to_csv(os.path.join(DATA_DIR, 'team_form_weekly.csv'), index=False)
        except Exception:
            pass

    # --- ADDED: optional external enrichers; fill only missing values ---
    try:
        for fn in _team_enricher_paths():
            ext = _read_csv_safe(os.path.join(DATA_DIR, fn))
            if ext.empty:
                continue
            norm = _standardize_team_enricher(ext)
            if norm.empty:
                continue
            df = _non_destructive_team_merge(df, norm)
    except Exception:
        # enrichment is optional; never crash
        pass
    # --- END ADDED ---

    if not success:
        print("[make_team_form] ⚠️ Wrote placeholder team_form.csv (no PBP available)")

    df.to_csv(OUTPATH, index=False)
    print(f"[make_team_form] Wrote {len(df)} rows → {OUTPATH}")


if __name__ == "__main__":
    # Silence noisy pandas warnings in CI
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
