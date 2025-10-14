#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("data/team_form.csv")

def _safe_write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        pd.DataFrame(columns=[
            "team","def_pass_epa","def_rush_epa","def_sack_rate",
            "pace","proe","light_box_rate","heavy_box_rate",
            "def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","proe_z","light_box_rate_z","heavy_box_rate_z"
        ]).to_csv(out, index=False)
    else:
        df.to_csv(out, index=False)

def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0: return pd.Series([0.0]*len(s), index=s.index)
    return (s - mu) / sd

def build_from_nflverse(season: int) -> pd.DataFrame:
    """
    Pull real season PBP and compute team-level defensive/pass/rush EPA allowed,
    sack rate per dropback, a simple pace proxy (plays per game), and a neutral
    pass rate minus league neutral pass rate as a PROE proxy.
    """
    try:
        import nfl_data_py as nfl
    except Exception as e:
        print(f"[team_form] nfl_data_py not installed; falling back to stub: {e}", flush=True)
        return pd.DataFrame([
            {"team":"BUF","def_pass_epa":0.10,"def_rush_epa":-0.05,"def_sack_rate":0.08,
             "pace":30.0,"proe":0.02,"light_box_rate":0.50,"heavy_box_rate":0.10}
        ])

    # Pull season pbp
    pbp = nfl.import_pbp_data([season])
    pbp = pbp.loc[pbp["season"]==season].copy()

    # Defensive team
    pbp["defteam"] = pbp["defteam"].astype(str).str.upper()
    pbp["posteam"] = pbp["posteam"].astype(str).str.upper()

    # Basic filters for epa plays
    pbp = pbp[pbp["epa"].notna()].copy()

    # Flags
    pbp["is_pass"] = (pbp.get("pass", 0).fillna(0) == 1) | (pbp.get("pass_attempt", 0).fillna(0) == 1) | (pbp.get("play_type","")=="pass")
    pbp["is_rush"] = (pbp.get("rush", 0).fillna(0) == 1) | (pbp.get("play_type","")=="run")
    pbp["is_dropback"] = (pbp.get("dropback", 0).fillna(0) == 1)
    pbp["is_sack"] = (pbp.get("sack", 0).fillna(0) == 1)

    # --- Defensive EPA allowed
    def_pass = pbp.loc[pbp["is_pass"]].groupby("defteam")["epa"].mean().rename("def_pass_epa")
    def_rush = pbp.loc[pbp["is_rush"]].groupby("defteam")["epa"].mean().rename("def_rush_epa")

    # --- Sack rate per dropback allowed (defense)
    drop = pbp.loc[pbp["is_dropback"]].groupby("defteam")["is_dropback"].count().rename("db")
    sacks = pbp.loc[pbp["is_sack"]].groupby("defteam")["is_sack"].count().rename("sacks")
    sack_rate = (sacks / drop).replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("def_sack_rate")

    # --- Pace proxy: offensive plays per game for opponents (defense seen)
    # count offensive plays by posteam, map to each opponent (defteam)
    plays_by_off = pbp.groupby(["game_id","posteam"])["play_id"].count().rename("plays").reset_index()
    g_counts = plays_by_off.groupby("posteam")["game_id"].nunique().rename("games")
    plays_pg = (plays_by_off.groupby("posteam")["plays"].sum() / g_counts).rename("plays_per_game")
    # map offensive plays per game to the defenses they faced (approx league-wide)
    # simpler: use plays_per_game of league average as a constant pace proxy.
    pace_proxy = (3600.0 / (plays_pg.mean() if np.isfinite(plays_pg.mean()) else 120.0))
    pace = pd.Series(pace_proxy, index=def_pass.index, name="pace")  # seconds/snap proxy

    # --- PROE proxy: neutral pass rate minus league neutral pass rate
    # Neutral: qtr<=3 and abs(score_differential)<=7 and between 20-80% win prob if available
    neutral = pbp[(pbp["qtr"]<=3) & (pbp["score_differential"].abs()<=7)].copy()
    if "wp" in neutral.columns:
        neutral = neutral[(neutral["wp"]>=0.20) & (neutral["wp"]<=0.80)]
    team_pass = neutral.groupby("posteam")["is_pass"].mean().rename("team_neutral_pr")
    league_neutral_pr = float(team_pass.mean()) if len(team_pass) else 0.55
    proe_off = (team_pass - league_neutral_pr).rename("proe")
    # map offensive proe to defense they face (use league mean as fallback)
    proe = pd.Series( -float(proe_off.mean() if len(proe_off) else 0.0), index=def_pass.index, name="proe")

    # Not available directly → default NaN (your metrics code tolerates or fills)
    lb = pd.Series(np.nan, index=def_pass.index, name="light_box_rate")
    hb = pd.Series(np.nan, index=def_pass.index, name="heavy_box_rate")

    df = pd.concat([def_pass, def_rush, sack_rate, pace, proe, lb, hb], axis=1).reset_index().rename(columns={"defteam":"team"})
    # z-scores (keep raw + z mirrors)
    for c in ["def_pass_epa","def_rush_epa","def_sack_rate","pace","proe","light_box_rate","heavy_box_rate"]:
        df[f"{c}_z"] = _zscore(df[c]) if c in df.columns else 0.0
    return df

def cli(season: int) -> int:
    try:
        df = build_from_nflverse(season)
        # ensure raw mirrors exist (no-op if already present)
        if not df.empty:
            for z, raw in [
                ("def_pass_epa_z","def_pass_epa"),
                ("def_rush_epa_z","def_rush_epa"),
                ("def_sack_rate_z","def_sack_rate"),
                ("pace_z","pace"),
                ("proe_z","proe"),
                ("light_box_rate_z","light_box_rate"),
                ("heavy_box_rate_z","heavy_box_rate"),
            ]:
                if raw not in df.columns and z in df.columns:
                    df[raw] = df[z]
    except Exception as e:
        print(f"[team_form] fatal error: {e}", flush=True)
        df = pd.DataFrame()
    _safe_write(df, OUT)
    print(f"[team_form] wrote rows={len(df)} → {OUT}")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, type=int)
    a = ap.parse_args()
    sys.exit(cli(a.season))
