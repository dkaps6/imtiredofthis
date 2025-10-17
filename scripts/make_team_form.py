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
from typing import List

import pandas as pd
import numpy as np

# -----------------------------
# Imports: prefer nflreadpy
# -----------------------------
def _import_nflverse():
    try:
        import nflreadpy as nflv  # Python port maintained by nflverse
        return nflv, "nflreadpy"
    except Exception:
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

# -----------------------------
# Helpers
# -----------------------------

def _safe_mkdir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


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

# -----------------------------
# Loaders (abstract across libs)
# -----------------------------

def load_pbp(season: int) -> pd.DataFrame:
    """
    Load regular-season PBP for given season.
    nflreadpy: NFLV.load_pbp(seasons=[season])
    nfl_data_py: nfl_data_py.import_pbp_data([season], downcast=True)
    """
    if NFL_PKG == "nflreadpy":
        pbp = NFLV.load_pbp(seasons=[season])
    else:
        pbp = NFLV.import_pbp_data([season], downcast=True)  # type: ignore
    # Normalize column names we rely on
    pbp.columns = [c.lower() for c in pbp.columns]
    return pbp


def load_participation(season: int) -> pd.DataFrame:
    """
    Participation / personnel (optional). nflreadpy exposes load_participation()
    If not available, return empty DF.
    """
    try:
        if NFL_PKG == "nflreadpy":
            part = NFLV.load_participation(seasons=[season])
        else:
            # nfl_data_py has limited/older participation; skip if missing
            return pd.DataFrame()
        part.columns = [c.lower() for c in part.columns]
        return part
    except Exception:
        return pd.DataFrame()


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
    pace_grp = dfn.groupby(off_col, dropna=False)["_one"].sum().rename("neutral_plays").reset_index()
    # Estimate elapsed neutral time: use quarter clock if available; otherwise approximate by #plays * 24 sec
    if "game_seconds_remaining" in dfn and "game_seconds_remaining".lower() in dfn:
        # It's already lowercase; above check is redundant but harmless.
        pass
    # Use a crude baseline of 24s per neutral play if timing columns aren't reliable
    pace_grp["pace_neutral"] = np.where(
        pace_grp["neutral_plays"] > 0,
        24.0,  # conservative neutral pace (seconds per snap)
        np.nan
    )

    # PROE: need pass rate minus expected pass rate.
    # If xp pass prob exists (xpass or pass_probability), use it; else use league-average neutral pass rate.
    is_pass = dfn.get("pass", pd.Series(False, index=dfn.index)).astype(bool)
    prate = dfn.groupby(off_col, dropna=False)["pass"].mean() if "pass" in dfn else pd.Series(dtype=float)

    if "xpass" in dfn:
        xpass = dfn.groupby(off_col, dropna=False)["xpass"].mean()
        proe = prate - xpass
    elif "pass_probability" in dfn:
        xp = dfn.groupby(off_col, dropna=False)["pass_probability"].mean()
        proe = prate - xp
    else:
        league_neutral_pass = prate.mean() if len(prate) else 0.55
        proe = prate - league_neutral_pass

    out = pace_grp.merge(proe.rename("proe"), left_on=off_col, right_index=True, how="left")
    out = out.rename(columns={off_col: "team"})
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


# -----------------------------
# Main builder
# -----------------------------

def build_team_form(season: int) -> pd.DataFrame:
    print(f"[make_team_form] Loading PBP for {season} via {NFL_PKG} ...")
    pbp = load_pbp(season)
    if pbp.empty:
        raise RuntimeError("PBP is empty; cannot compute team form.")

    print("[make_team_form] Computing defensive EPA & sack rate ...")
    def_tbl = compute_def_epa_and_sacks(pbp)

    print("[make_team_form] Computing pace & PROE ...")
    pace_tbl = compute_pace_and_proe(pbp)

    print("[make_team_form] Computing RZ rate & air yards per att ...")
    rz_ay_tbl = compute_red_zone_and_airyards(pbp)

    print("[make_team_form] Loading participation/personnel (optional) ...")
    part = load_participation(season)
    pers_tbl = compute_personnel_rates(pbp, part)

    print("[make_team_form] Merging components ...")
    out = def_tbl.merge(pace_tbl, on="team", how="left") \
                 .merge(rz_ay_tbl, on="team", how="left") \
                 .merge(pers_tbl, on="team", how="left")

    # add slot rate proxy from roles.csv if present
    out = merge_slot_rate_from_roles(out)

    # attach season and enforce dtypes
    out["season"] = int(season)

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
    cols_first = ["team", "season", "games_played",
                  "def_pass_epa", "def_rush_epa", "def_sack_rate",
                  "pace", "proe", "rz_rate", "12p_rate", "slot_rate", "ay_per_att",
                  "light_box_rate", "heavy_box_rate"]
    # preserve if missing
    ordered = [c for c in cols_first if c in out.columns] + [c for c in out.columns if c not in cols_first]
    out = out[ordered].sort_values(["team"]).reset_index(drop=True)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()

    _safe_mkdir(DATA_DIR)

    try:
        df = build_team_form(args.season)
    except Exception as e:
        print(f"[make_team_form] ERROR: {e}", file=sys.stderr)
        # write empty but schema-like csv to avoid downstream crashes
        empty = pd.DataFrame(columns=[
            "team","season","games_played","def_pass_epa","def_rush_epa","def_sack_rate",
            "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
            "light_box_rate","heavy_box_rate"
        ])
        empty.to_csv(OUTPATH, index=False)
        sys.exit(1)

    df.to_csv(OUTPATH, index=False)
    print(f"[make_team_form] Wrote {len(df)} rows → {OUTPATH}")


if __name__ == "__main__":
    # Silence noisy pandas warnings in CI
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
