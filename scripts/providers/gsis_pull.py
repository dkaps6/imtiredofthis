#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GSIS (via nflverse) provider — 2025 aggregates.

Writes:
  - data/gsis_weekly_<season>.csv                (raw weekly dump for auditing)
  - data/nflgsis_team_form.csv                   (TEAM enricher for fallback sweep)
  - data/nflgsis_player_form.csv                 (PLAYER enricher; basic efficiency)
  - data/gsis_team_propensity_<season>.csv       (bonus: pass rates, PROE, pace)
  - data/gsis_team_rankings_<season>.csv         (bonus: +/- and simple ranks)

All metrics are computed from real nflverse data (GSIS-derived) — no scraping.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Any

import numpy as np
import pandas as pd

DATA_DIR = "data"

TEAM_COLS = [
    "team","def_pass_epa","def_rush_epa","def_sack_rate",
    "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
    "light_box_rate","heavy_box_rate"
]
PLAYER_COLS = [
    "player","team",
    "tgt_share","route_rate","rush_share",
    "yprr","ypt","ypc","ypa",
    "receptions_per_target",
    "rz_share","rz_tgt_share","rz_rush_share",
]

# ------------------------
# Canonical team mapping (fixes LA/JAC/WSH, etc.)
# ------------------------
CANON = {
    "OAK":"LV","SD":"LAC","STL":"LAR","JAC":"JAX","WSH":"WAS","LA":"LAR",
    "LAS":"LV","LOS ANGELES":"LAR","LOS ANGELES RAMS":"LAR","LOS ANGELES CHARGERS":"LAC",
    "WASHINGTON":"WAS","NEW YORK GIANTS":"NYG","NEW YORK JETS":"NYJ"
}
VALID = {"ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU",
         "IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ",
         "PHI","PIT","SEA","SF","TB","TEN","WAS"}

def canon_team(x: str) -> str:
    if x is None: return ""
    s = str(x).strip().upper()
    s = CANON.get(s, s)
    return s if s in VALID else ""

# ------------------------
# Helpers
# ------------------------

def _safe_mkdir(p: str): os.makedirs(p, exist_ok=True)

def _write_csv_with_schema(path: str, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out.columns = [c.lower() for c in out.columns]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols]
    out.to_csv(path, index=False)
    return out

def _to_pd(obj: Any) -> pd.DataFrame:
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

def _neutral_mask(df: pd.DataFrame) -> pd.Series:
    m = pd.Series(True, index=df.index)
    if "score_differential" in df.columns:
        m &= df["score_differential"].between(-7, 7)
    if "wp" in df.columns:
        m &= df["wp"].between(0.2, 0.8)
    if "qtr" in df.columns:
        m &= df["qtr"] <= 3
    return m

def _norm_team(s) -> str:
    return str(s).upper().strip() if s is not None else ""

# ------------------------
# Loads
# ------------------------

def _import_sources():
    try:
        import nflreadpy as nflv
        return {"pkg": "nflreadpy", "nflv": nflv, "nfl": None}
    except Exception:
        try:
            import nfl_data_py as nfl
            return {"pkg": "nfl_data_py", "nflv": None, "nfl": nfl}
        except Exception as e:
            raise RuntimeError("Install nflreadpy or nfl_data_py") from e

def load_pbp(season: int) -> pd.DataFrame:
    src = _import_sources()
    if src["pkg"] == "nflreadpy":
        raw = src["nflv"].load_pbp(seasons=[season])
    else:
        raw = src["nfl"].import_pbp_data([season], downcast=True)
    df = _to_pd(raw)
    df.columns = [c.lower() for c in df.columns]
    return df

def load_participation(season: int) -> pd.DataFrame:
    try:
        src = _import_sources()
        if src["pkg"] == "nflreadpy":
            raw = src["nflv"].load_participation(seasons=[season])
        else:
            return pd.DataFrame()
        df = _to_pd(raw)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def load_weekly(season: int) -> pd.DataFrame:
    src = _import_sources()
    if src["pkg"] == "nflreadpy":
        try:
            raw = src["nflv"].load_player_stats(seasons=[season], stat_type="weekly")
            df = _to_pd(raw)
        except Exception:
            return pd.DataFrame()
    else:
        df = src["nfl"].import_weekly_data([season], downcast=True)
    if "season" in df:
        df = df[df["season"] == season]
    df.columns = [c.lower() for c in df.columns]
    return df

# ------------------------
# Aggregations (TEAM)
# ------------------------

def team_def_epa_sacks(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    is_pass = df.get("pass", pd.Series(False, index=df.index)).astype(bool)
    is_rush = df.get("rush", pd.Series(False, index=df.index)).astype(bool)

    def_col = "defteam" if "defteam" in df.columns else ("def_team" if "def_team" in df.columns else None)
    if def_col is None:
        return pd.DataFrame(columns=["team","def_pass_epa","def_rush_epa","def_sack_rate"])

    sack = df.get("sack", pd.Series(0, index=df.index)).fillna(0).astype(int)
    dropbacks = df.get("qb_dropback", pd.Series(0, index=df.index)).fillna(0).astype(int)

    g = df.groupby(def_col, dropna=False)
    def_pass_epa = g.apply(lambda x: x.loc[is_pass.reindex(x.index, fill_value=False), "epa"].mean())
    def_rush_epa = g.apply(lambda x: x.loc[is_rush.reindex(x.index, fill_value=False), "epa"].mean())
    sacks = g[sack.name].sum()
    db = g[dropbacks.name].sum()

    out = pd.DataFrame({
        "team": def_pass_epa.index.astype(str),
        "def_pass_epa": def_pass_epa.values,
        "def_rush_epa": def_rush_epa.values,
        "def_sack_rate": np.where(db.values > 0, sacks.values / db.values, np.nan),
    })
    out["team"] = out["team"].map(_norm_team).map(canon_team)
    out = out[out["team"] != ""]
    return out

def team_pace_proe(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    off = "posteam" if "posteam" in df.columns else ("offense_team" if "offense_team" in df.columns else None)
    if off is None:
        return pd.DataFrame(columns=["team","pace","proe"])

    neutral = _neutral_mask(df)
    dn = df.loc[neutral].copy()
    dn = dn.sort_values([off, "game_id", "qtr", "play_id"], kind="mergesort")
    gg = dn.groupby([off, "game_id"], dropna=False)

    if "game_seconds_remaining" in dn.columns:
        dn["gsr_diff"] = gg["game_seconds_remaining"].diff(-1).abs()
        pace_game = dn.groupby([off, "game_id"], dropna=False)["gsr_diff"].mean()
        pace = pace_game.groupby(level=0).mean()
    else:
        pace = pd.Series(index=dn[off].unique(), data=np.nan)

    if "pass" in dn.columns:
        prate = dn.groupby(off, dropna=False)["pass"].mean()
    else:
        pt = dn.get("play_type")
        prate = pt.isin(["pass", "no_play"]).groupby(dn[off]).mean() if pt is not None else pd.Series(dtype=float)

    if "xpass" in dn.columns:
        xpass = dn.groupby(off, dropna=False)["xpass"].mean()
        proe = prate.sub(xpass, fill_value=np.nan)
    elif "pass_probability" in dn.columns:
        xp = dn.groupby(off, dropna=False)["pass_probability"].mean()
        proe = prate.sub(xp, fill_value=np.nan)
    else:
        league = prate.mean() if len(prate) else np.nan
        proe = prate - league

    out = pd.DataFrame({"team": proe.index.astype(str), "pace": pace.reindex(proe.index).values, "proe": proe.values})
    out["team"] = out["team"].map(_norm_team).map(canon_team)
    out = out[out["team"] != ""]
    return out

def team_rz_ay(personnel_pbp: pd.DataFrame) -> pd.DataFrame:
    df = personnel_pbp.copy()
    off = "posteam" if "posteam" in df.columns else ("offense_team" if "offense_team" in df.columns else None)
    if off is None:
        return pd.DataFrame(columns=["team","rz_rate","ay_per_att","12p_rate"])

    yd = pd.to_numeric(df.get("yardline_100"), errors="coerce")
    df["rz_flag"] = (yd <= 20).astype(float)

    is_pass = df.get("pass", pd.Series(False, index=df.index)).astype(bool)
    ay = pd.to_numeric(df.get("air_yards"), errors="coerce")
    ay_per_att = df.loc[is_pass].groupby(off)["air_yards"].mean()

    per = df.get("personnel_offense")
    if per is not None:
        per_codes = per.astype(str).str.extract(r"(\d\d)")[0]
        df["_per12"] = per_codes.eq("12").astype(float)
        p12 = df.groupby(off)["_per12"].mean()
    else:
        p12 = pd.Series(dtype=float)

    rz = df.groupby(off)["rz_flag"].mean()

    out = pd.DataFrame({
        "team": rz.index.astype(str),
        "rz_rate": rz.values,
        "ay_per_att": ay_per_att.reindex(rz.index).values,
        "12p_rate": p12.reindex(rz.index).values,
    })
    out["team"] = out["team"].map(_norm_team).map(canon_team)
    out = out[out["team"] != ""]
    return out

def team_box_counts(part: pd.DataFrame) -> pd.DataFrame:
    if part is None or not isinstance(part, pd.DataFrame) or part.empty:
        return pd.DataFrame(columns=["team","light_box_rate","heavy_box_rate"])

    team_col = None
    for cand in ["offense_team", "posteam", "team"]:
        if cand in part.columns:
            team_col = cand; break
    if team_col is None:
        return pd.DataFrame(columns=["team","light_box_rate","heavy_box_rate"])

    box_col = None
    for cand in ["box", "men_in_box", "in_box", "defenders_in_box"]:
        if cand in part.columns:
            box_col = cand; break
    if box_col is None:
        return pd.DataFrame(columns=["team","light_box_rate","heavy_box_rate"])

    x = part.copy()
    x["_box"] = pd.to_numeric(x[box_col], errors="coerce")
    g = x.groupby(team_col)
    light = g["_box"].apply(lambda s: (s <= 6).mean()).rename("light_box_rate")
    heavy = g["_box"].apply(lambda s: (s >= 8).mean()).rename("heavy_box_rate")
    out = pd.concat([light, heavy], axis=1).reset_index().rename(columns={team_col:"team"})
    out["team"] = out["team"].map(_norm_team).map(canon_team)
    out = out[out["team"] != ""]
    return out

def team_points_rankings(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    g = df.groupby("game_id")
    if {"total_home_score","total_away_score"}.issubset(df.columns):
        game_scores = g[["total_home_score","total_away_score"]].max().reset_index()
    elif {"home_score","away_score"}.issubset(df.columns):
        game_scores = g[["home_score","away_score"]].max().reset_index().rename(
            columns={"home_score":"total_home_score","away_score":"total_away_score"}
        )
    else:
        return pd.DataFrame(columns=["team","points_for","points_against","point_diff","rank_point_diff"])

    recs = []
    for _, row in game_scores.iterrows():
        gid = row["game_id"]
        try:
            sub = df[df["game_id"] == gid]
            home = _norm_team(sub["home_team"].iloc[0]) if "home_team" in sub.columns else ""
            away = _norm_team(sub["away_team"].iloc[0]) if "away_team" in sub.columns else ""
            th, ta = canon_team(home), canon_team(away)
            hs = row["total_home_score"]; as_ = row["total_away_score"]
            if th: recs.append({"team": th, "points_for": hs, "points_against": as_})
            if ta: recs.append({"team": ta, "points_for": as_, "points_against": hs})
        except Exception:
            continue
    pts = pd.DataFrame(recs)
    if pts.empty:
        return pd.DataFrame(columns=["team","points_for","points_against","point_diff","rank_point_diff"])

    agg = pts.groupby("team", dropna=False)[["points_for","points_against"]].sum().reset_index()
    agg["point_diff"] = agg["points_for"] - agg["points_against"]
    agg["rank_point_diff"] = (-agg["point_diff"]).rank(method="min")
    return agg

# ------------------------
# Aggregations (PLAYER basics)
# ------------------------

def player_efficiency_basics(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    off = "posteam" if "posteam" in df.columns else ("offense_team" if "offense_team" in df.columns else None)
    if off is None:
        return pd.DataFrame(columns=["player","team","ypa","ypc"])

    qb_name = "passer_player_name" if "passer_player_name" in df.columns else ("passer" if "passer" in df.columns else None)
    pass_df = df[df.get("pass", pd.Series(False, index=df.index)).astype(bool)].copy()
    if qb_name is not None and not pass_df.empty:
        pass_df["player"] = pass_df[qb_name].astype(str).str.replace(".","", regex=False).str.strip()
        pass_df["team"] = pass_df[off].astype(str).str.upper().str.strip().map(canon_team)
        qb = pass_df.groupby(["team","player"]).agg(
            pass_yards=("yards_gained","sum"),
            pass_att=("pass_attempt","sum") if "pass_attempt" in pass_df.columns else (qb_name,"size")
        ).reset_index()
        qb = qb[qb["team"] != ""]
        qb["ypa"] = np.where(qb["pass_att"]>0, qb["pass_yards"]/qb["pass_att"], np.nan)
        qb = qb[["team","player","ypa"]]
    else:
        qb = pd.DataFrame(columns=["team","player","ypa"])

    ru = df[df.get("rush", pd.Series(False, index=df.index)).astype(bool)].copy()
    rusher_name = "rusher_player_name" if "rusher_player_name" in ru.columns else ("rusher" if "rusher" in ru.columns else None)
    if rusher_name is not None and not ru.empty:
        ru["player"] = ru[rusher_name].astype(str).str.replace(".","", regex=False).str.strip()
        ru["team"] = ru[off].astype(str).str.upper().str.strip().map(canon_team)
        rb = ru.groupby(["team","player"]).agg(
            rush_yards=("yards_gained","sum"),
            rush_att=("rush_attempt","sum") if "rush_attempt" in ru.columns else (rusher_name,"size")
        ).reset_index()
        rb = rb[rb["team"] != ""]
        rb["ypc"] = np.where(rb["rush_att"]>0, rb["rush_yards"]/rb["rush_att"], np.nan)
        rb = rb[["team","player","ypc"]]
    else:
        rb = pd.DataFrame(columns=["team","player","ypc"])

    out = pd.merge(qb, rb, on=["team","player"], how="outer")
    return out

# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=int, help="Season year, e.g., 2025")
    args = parser.parse_args()

    season = int(args.season)
    _safe_mkdir(DATA_DIR)

    pbp = load_pbp(season)
    if pbp.empty:
        print("[gsis_pull] PBP empty; writing schema-only files", file=sys.stderr)
        _write_csv_with_schema(os.path.join(DATA_DIR,"nflgsis_team_form.csv"), pd.DataFrame(), TEAM_COLS)
        _write_csv_with_schema(os.path.join(DATA_DIR,"nflgsis_player_form.csv"), pd.DataFrame(), PLAYER_COLS)
        return 0

    # Weekly (audit)
    try:
        weekly = load_weekly(season)
        if not weekly.empty:
            weekly.to_csv(os.path.join(DATA_DIR, f"gsis_weekly_{season}.csv"), index=False)
    except Exception as e:
        print(f"[gsis_pull] weekly dump failed: {e}", file=sys.stderr)

    # TEAM metrics (canonize before merge)
    def_tbl   = team_def_epa_sacks(pbp)
    pace_tbl  = team_pace_proe(pbp)
    rz_tbl    = team_rz_ay(pbp)
    part      = load_participation(season)
    box_tbl   = team_box_counts(part)
    ranks_tbl = team_points_rankings(pbp)

    team = def_tbl.merge(pace_tbl, on="team", how="left") \
                  .merge(rz_tbl,   on="team", how="left") \
                  .merge(box_tbl,  on="team", how="left")

    # Emit TEAM enricher
    team_out = _write_csv_with_schema(os.path.join(DATA_DIR,"nflgsis_team_form.csv"), team, TEAM_COLS)
    print(f"[gsis_pull] wrote team enricher rows={len(team_out)} → data/nflgsis_team_form.csv")

    # PLAYER basics
    ply_eff = player_efficiency_basics(pbp)
    player_enricher = ply_eff.copy()
    player_enricher.columns = [c.lower() for c in player_enricher.columns]
    player_out = _write_csv_with_schema(os.path.join(DATA_DIR,"nflgsis_player_form.csv"), player_enricher, PLAYER_COLS)
    print(f"[gsis_pull] wrote player enricher rows={len(player_out)} → data/nflgsis_player_form.csv")

    # Bonus: propensity / rankings
    try:
        df = pbp.copy()
        off = "posteam" if "posteam" in df.columns else ("offense_team" if "offense_team" in df.columns else None)
        if off is not None:
            if "pass" in df.columns:
                pr_all = df.groupby(off)["pass"].mean().rename("pass_rate_all")
            else:
                pt = df.get("play_type")
                pr_all = pt.isin(["pass","no_play"]).groupby(df[off]).mean().rename("pass_rate_all") if pt is not None else pd.Series(dtype=float)

            neutral = _neutral_mask(df)
            dn = df.loc[neutral].copy()
            if "pass" in dn.columns:
                pr_neutral = dn.groupby(off)["pass"].mean().rename("pass_rate_neutral")
            else:
                pt = dn.get("play_type")
                pr_neutral = pt.isin(["pass","no_play"]).groupby(dn[off]).mean().rename("pass_rate_neutral") if pt is not None else pd.Series(dtype=float)

            early = df[df["down"].isin([1,2])] if "down" in df.columns else df.iloc[0:0]
            if not early.empty:
                if "pass" in early.columns:
                    pr_early = early.groupby(off)["pass"].mean().rename("pass_rate_early")
                else:
                    pt = early.get("play_type")
                    pr_early = pt.isin(["pass","no_play"]).groupby(early[off]).mean().rename("pass_rate_early")
            else:
                pr_early = pd.Series(dtype=float)

            proe = pace_tbl.set_index("team")["proe"]
            pace = pace_tbl.set_index("team")["pace"]

            prop = pd.concat([pr_all, pr_neutral, pr_early], axis=1).reset_index().rename(columns={off:"team"})
            prop["team"] = prop["team"].map(_norm_team).map(canon_team)
            prop = prop[prop["team"] != ""]
            prop = prop.merge(proe.rename("proe"), left_on="team", right_index=True, how="left")
            prop = prop.merge(pace.rename("pace"), left_on="team", right_index=True, how="left")
            prop.to_csv(os.path.join(DATA_DIR, f"gsis_team_propensity_{season}.csv"), index=False)
            print(f"[gsis_pull] wrote propensity → data/gsis_team_propensity_{season}.csv")
    except Exception as e:
        print(f"[gsis_pull] propensity failed: {e}", file=sys.stderr)

    try:
        if not ranks_tbl.empty:
            ranks_tbl.to_csv(os.path.join(DATA_DIR, f"gsis_team_rankings_{season}.csv"), index=False)
            print(f"[gsis_pull] wrote rankings → data/gsis_team_rankings_{season}.csv")
    except Exception as e:
        print(f"[gsis_pull] rankings failed: {e}", file=sys.stderr)

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except TypeError:
        sys.exit(main())
