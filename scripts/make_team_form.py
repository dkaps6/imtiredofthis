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
- rz_rate                   # share of offensive plays inside opp 20
- personnel_12_rate         # 12 personnel usage rate (offense)
- slot_rate                 # proxy from roles.csv (optional)
- ay_per_att                # air yards per pass attempt (offense)
- light_box_rate            # % snaps vs light boxes (if participation present)
- heavy_box_rate            # % snaps vs heavy boxes (if participation present)

Default behavior now FAILS the run if required metrics are missing.
Use --allow-missing-box to ignore missing box-count rates only.

Usage:
    python scripts/make_team_form.py --season 2025 [--allow-missing-box]
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import List, Any

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

def _is_empty(obj) -> bool:
    """Robust emptiness check that does not rely on .empty existing."""
    try:
        return (obj is None) or (not hasattr(obj, "__len__")) or (len(obj) == 0)
    except Exception:
        return True

def _to_pandas(obj: Any) -> pd.DataFrame:
    """
    Normalize any tabular object (Polars, PyArrow, Pandas, list of batches) to pandas.DataFrame.
    Fixes CI cases where nflreadpy returns Polars (no .copy()).
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    # Polars / PyArrow both usually expose .to_pandas()
    if hasattr(obj, "to_pandas") and callable(getattr(obj, "to_pandas")):
        try:
            return obj.to_pandas()
        except Exception:
            pass
    # List/tuple of arrow batches
    if isinstance(obj, (list, tuple)) and len(obj) and hasattr(obj[0], "to_pandas"):
        try:
            return pd.concat([b.to_pandas() for b in obj], ignore_index=True)
        except Exception:
            pass
    try:
        return pd.DataFrame(obj)
    except Exception:
        raise RuntimeError("Could not convert object to pandas.DataFrame")

def zscore(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df and df[c].notna().sum() >= 8:
            m = df[c].mean()
            s = df[c].std(ddof=0)
            if s and not np.isclose(s, 0):
                df[c] = pd.to_numeric(df[c], errors="coerce")
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
    n = pd.to_numeric(n, errors="coerce").astype(float)
    d = pd.to_numeric(d, errors="coerce").astype(float)
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
    if _is_empty(add) or "team" not in add.columns:
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

def _force_team_col(df: pd.DataFrame, off_col: str | None) -> pd.DataFrame:
    """Ensure there is a 'team' column; if off_col exists, rename it, else try common fallbacks."""
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
# Loaders (normalize to pandas)
# -----------------------------

def load_pbp(season: int) -> pd.DataFrame:
    """
    Load regular-season PBP for given season.
    nflreadpy: NFLV.load_pbp(seasons=[season])
    nfl_data_py: nfl_data_py.import_pbp_data([season], downcast=True)
    """
    if NFL_PKG == "nflreadpy":
        pbp_raw = NFLV.load_pbp(seasons=[season])
    else:
        pbp_raw = NFLV.import_pbp_data([season], downcast=True)  # type: ignore
    pbp = _to_pandas(pbp_raw)
    pbp.columns = [c.lower() for c in pbp.columns]
    return pbp

def load_participation(season: int) -> pd.DataFrame:
    """
    Participation / personnel (optional). nflreadpy exposes load_participation()
    If not available, return empty DF.
    """
    try:
        if NFL_PKG == "nflreadpy":
            part_raw = NFLV.load_participation(seasons=[season])
        else:
            # nfl_data_py has limited/older participation; skip if missing
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
    """
    Defensive EPA splits and sack rate allowed by offense (defense perspective).
    We'll aggregate by defense team (defteam) when available, else by posteam/opponent mapping.
    """
    df = pbp.copy()
    if "epa" not in df:
        df["epa"] = np.nan

    is_pass = df.get("pass", pd.Series(False, index=df.index)).astype(bool)
    is_rush = df.get("rush", pd.Series(False, index=df.index)).astype(bool)

    def_team_col = "defteam" if "defteam" in df else ("def_team" if "def_team" in df else None)
    off_team_col = "posteam" if "posteam" in df else ("offense_team" if "offense_team" in df else None)

    if def_team_col is None or off_team_col is None:
        return pd.DataFrame(columns=["team","def_pass_epa","def_rush_epa","def_sack_rate","games_played"])

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

    agg["def_sack_rate"] = safe_div(agg["def_sacks"], agg["opp_dropbacks"])
    return agg

def compute_pace_and_proe(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Pace: seconds per snap in neutral situations (lower is faster).
    PROE: pass rate over expectation (needs xpass if present; otherwise heuristic).
    Always returns columns: ['team', 'pace_neutral', 'proe'].
    """
    df = pbp.copy()
    df.columns = [c.lower() for c in df.columns]

    neutral = _neutral_mask(df)
    dfn = df.loc[neutral].copy()

    off_col = "posteam" if "posteam" in dfn else ("offense_team" if "offense_team" in dfn else None)
    if off_col is None:
        return pd.DataFrame(columns=["team","pace_neutral","proe"])

    dfn = dfn.sort_values([off_col, "game_id", "qtr", "play_id"], kind="mergesort")
    grp = dfn.groupby([off_col, "game_id"], dropna=False)

    # --- Pace ---
    if "game_seconds_remaining" in dfn.columns:
        dfn["__gsr_diff"] = grp["game_seconds_remaining"].diff(-1).abs()
        pace_per_game = dfn.groupby([off_col, "game_id"], dropna=False)["__gsr_diff"].mean()
        pace_team = pace_per_game.groupby(level=0).mean().rename("pace_neutral").reset_index()
        # ensure 'team' exists for downstream merges
        pace_team = pace_team.rename(columns={off_col: "team"})
    else:
        neutral_plays = grp.size().groupby(level=0).sum().rename("neutral_plays").reset_index()
        pace_team = neutral_plays.assign(pace_neutral=np.where(neutral_plays["neutral_plays"] > 0, 24.0, np.nan))
        pace_team = pace_team.rename(columns={off_col: "team"})

    # --- PROE ---
    is_pass = dfn.get("pass", pd.Series(False, index=dfn.index)).astype(bool)
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

    out = pace_team.merge(proe.rename("proe"), on="team", how="left")
    out = _force_team_col(out, off_col)
    out = out[["team", "pace_neutral", "proe"]]
    return out

def compute_red_zone_and_airyards(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    RZ rate (offense): share of offensive plays inside opp 20.
    Air yards per attempt (offense).
    """
    df = pbp.copy()
    df.columns = [c.lower() for c in df.columns]

    off_col = "posteam" if "posteam" in df.columns else ("offense_team" if "offense_team" in df.columns else None)
    if off_col is None:
        return pd.DataFrame(columns=["team","rz_rate","ay_per_att"])

    yardline = pd.to_numeric(df.get("yardline_100"), errors="coerce")
    rz = (
        df.assign(yardline_100=yardline,
                  rz_flag=lambda x: (x["yardline_100"] <= 20).astype(int))
          .groupby(off_col, dropna=False)["rz_flag"].mean()
          .rename("rz_rate")
    )

    is_pass = df.get("pass", pd.Series(False, index=df.index)).astype(bool)
    pass_df = df.loc[is_pass].copy()
    ay = pd.to_numeric(pass_df.get("air_yards"), errors="coerce")
    ay_per_att = (
        pass_df.assign(air_yards=ay)
               .groupby(off_col, dropna=False)["air_yards"]
               .mean()
               .rename("ay_per_att")
    )

    out = pd.concat([rz, ay_per_att], axis=1).reset_index().rename(columns={off_col: "team"})
    return out

def compute_personnel_rates(pbp: pd.DataFrame, participation: pd.DataFrame) -> pd.DataFrame:
    """
    Personnel usage (12 personnel) and defensive box counts from participation if available.
    Also writes light/heavy box rates if 'box' or 'men_in_box' like fields exist.
    """
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

    light = heavy = None
    if not _is_empty(participation):
        p = participation.copy()
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
            p["_light"] = (pd.to_numeric(p[box_col], errors="coerce") <= 6).astype(float)
            p["_heavy"] = (pd.to_numeric(p[box_col], errors="coerce") >= 8).astype(float)
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
    We count % of WR roles labeled SLOT among WR roles.
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

    wr = r[r["role"].astype(str).str.upper().isin(["WR1", "WR2", "WR3", "SLOT"])].copy()
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
    if _is_empty(pbp):
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
    # ensure each component has a 'team' column for merges (belt & suspenders)
    def_tbl   = _force_team_col(def_tbl,   None)
    pace_tbl  = _force_team_col(pace_tbl,  "posteam")
    rz_ay_tbl = _force_team_col(rz_ay_tbl, "posteam")
    pers_tbl  = _force_team_col(pers_tbl,  "posteam")

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

    # rename to match your pricing expectations
    out = out.rename(columns={
        "pace_neutral": "pace",
        "personnel_12_rate": "12p_rate"
    })

    # Ensure expected columns exist (they may still be NaN; we'll validate below)
    for need in ["rz_rate","12p_rate","slot_rate","ay_per_att","light_box_rate","heavy_box_rate"]:
        if need not in out.columns:
            out[need] = np.nan

    # Sort and reset
    cols_first = ["team", "season", "games_played",
                  "def_pass_epa", "def_rush_epa", "def_sack_rate",
                  "pace", "proe", "rz_rate", "12p_rate", "slot_rate", "ay_per_att",
                  "light_box_rate", "heavy_box_rate"]
    ordered = [c for c in cols_first if c in out.columns] + [c for c in out.columns if c not in cols_first]
    out = out[ordered].sort_values(["team"]).reset_index(drop=True)

    return out

def _validate_required(df: pd.DataFrame, allow_missing_box: bool = False):
    """Fail the run if any required metrics are missing (NaN) for any team."""
    required = [
        "def_pass_epa", "def_rush_epa", "def_sack_rate",
        "pace", "proe", "rz_rate", "ay_per_att",
    ]
    box_cols = ["light_box_rate", "heavy_box_rate"]
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--allow-missing-box", action="store_true",
                        help="Do not fail if light/heavy box rates are missing (participation feed unavailable).")
    args = parser.parse_args()

    _safe_mkdir(DATA_DIR)

    try:
        df = build_team_form(args.season)

        # --- STRICT VALIDATION (default on) ---
        _validate_required(df, allow_missing_box=args.allow_missing_box)

        # --- Also emit per-team, per-week environment safely (optional) ---
        try:
            pbp = load_pbp(args.season)
            if not _is_empty(pbp):
                w = pbp.copy()
                if 'down' in w.columns:
                    w = w[w['down'].isin([1, 2])]
                if 'wp' in w.columns:
                    w = w[w['wp'].between(0.2, 0.8, inclusive='both')]

                team_col = 'posteam' if 'posteam' in w.columns else ('offense_team' if 'offense_team' in w.columns else None)
                if team_col is not None and 'week' in w.columns:
                    plays = w.groupby([team_col, 'week'], dropna=False)['play_id'].size().rename('plays_est')
                    if 'game_seconds_remaining' in w.columns:
                        w = w.sort_values([team_col, 'game_id', 'qtr', 'play_id'])
                        w['gsr_diff'] = w.groupby([team_col, 'game_id'])['game_seconds_remaining'].diff(-1).abs()
                        pace = w.groupby([team_col, 'week'])['gsr_diff'].mean().rename('pace')
                    else:
                        pace = plays * 0 + np.nan

                    if 'play_type' in w.columns:
                        is_pass = w['play_type'].isin(['pass', 'no_play'])
                        pr = is_pass.groupby([w[team_col], w['week']]).mean().rename('pass_rate')
                        lg = is_pass.groupby(w['week']).mean().rename('lg_pass_rate_week')
                        wk = pd.concat([plays, pace, pr], axis=1).reset_index().rename(columns={team_col: 'team'})
                        wk = wk.merge(lg.reset_index(), on='week', how='left')
                        wk['proe'] = wk['pass_rate'] - wk['lg_pass_rate_week']
                    else:
                        wk = pd.concat([plays, pace], axis=1).reset_index().rename(columns={team_col: 'team'})
                        wk['proe'] = np.nan

                    wk[['team', 'week', 'plays_est', 'pace', 'proe']].to_csv(os.path.join(DATA_DIR, 'team_form_weekly.csv'), index=False)
                else:
                    pd.DataFrame(columns=['team', 'week', 'plays_est', 'pace', 'proe']).to_csv(os.path.join(DATA_DIR, 'team_form_weekly.csv'), index=False)
            else:
                pd.DataFrame(columns=['team', 'week', 'plays_est', 'pace', 'proe']).to_csv(os.path.join(DATA_DIR, 'team_form_weekly.csv'), index=False)
        except Exception:
            try:
                pd.DataFrame(columns=['team', 'week', 'plays_est', 'pace', 'proe']).to_csv(os.path.join(DATA_DIR, 'team_form_weekly.csv'), index=False)
            except Exception:
                pass
        # --- END weekly export ---

    except Exception as e:
        print(f"[make_team_form] ERROR: {e}", file=sys.stderr)
        empty = pd.DataFrame(columns=[
            "team","season","games_played","def_pass_epa","def_rush_epa","def_sack_rate",
            "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
            "light_box_rate","heavy_box_rate"
        ])
        empty.to_csv(OUTPATH, index=False)
        sys.exit(1)

    # --- Optional external enrichers; fill only missing values, never overwrite non-null ---
    try:
        for fn in ["espn_team_form.csv", "msf_team_form.csv", "apisports_team_form.csv", "nflgsis_team_form.csv"]:
            ext = _read_csv_safe(os.path.join(DATA_DIR, fn))
            if not _is_empty(ext) and "team" in ext.columns:
                ext["team"] = ext["team"].astype(str).str.upper().str.strip()
                df = _non_destructive_team_merge(df, ext)
    except Exception:
        pass

    df.to_csv(OUTPATH, index=False)
    print(f"[make_team_form] Wrote {len(df)} rows → {OUTPATH}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
