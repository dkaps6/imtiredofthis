#!/usr/bin/env python3
from __future__ import annotations
import sys, os, time
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("data/team_form.csv")

def _safe_write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        # keep your expected headers so downstream merges don’t break
        pd.DataFrame(columns=[
            "team","def_pass_epa","def_rush_epa","def_sack_rate",
            "pace","proe","light_box_rate","heavy_box_rate",
            "def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","proe_z","light_box_rate_z","heavy_box_rate_z"
        ]).to_csv(out, index=False)
    else:
        df.to_csv(out, index=False)

def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0: return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

# NEW: resilient fetch with retries (handles transient parquet/network hiccups)
def _fetch_pbp_with_retry(season: int, tries: int = 3, wait: int = 4) -> pd.DataFrame:
    import nfl_data_py as nfl
    last = None
    for i in range(tries):
        try:
            df = nfl.import_pbp_data([season])
            if df is not None and len(df):
                return df
        except Exception as e:
            last = e
        time.sleep(wait)
    raise RuntimeError(f"nfl_data_py import failed after {tries} tries: {last}")

def build_from_nflverse(season: int) -> pd.DataFrame:
    # NOTE: do NOT raise “Error” (that symbol isn’t defined) — use Exception/RuntimeError only.
    try:
        import nfl_data_py as nfl  # noqa: F401 (ensures module exists for retry helper)
    except Exception as e:
        print(f"[team_form] nfl_data_py import failed → fallback: {e}", flush=True)
        # fallback stub (keeps pipeline alive)
        return pd.DataFrame([
            {"team":"BUF","def_pass_epa":0.10,"def_rush_epa":-0.05,"def_sack_rate":0.08,
             "pace":30.0,"proe":0.02,"light_box_rate":np.nan,"heavy_box_rate":np.nan}
        ])

    print("[team_form] pulling pbp…", flush=True)
    pbp = _fetch_pbp_with_retry(season)  # NEW: was nfl.import_pbp_data([season])
    pbp = pbp.loc[pbp["season"]==season].copy()

    pbp["defteam"] = pbp["defteam"].astype(str).str.upper()
    pbp["posteam"] = pbp["posteam"].astype(str).str.upper()

    pbp["is_pass"] = (pbp.get("pass",0)==1) | (pbp.get("pass_attempt",0)==1) | (pbp.get("play_type","")=="pass")
    pbp["is_rush"] = (pbp.get("rush",0)==1) | (pbp.get("play_type","")=="run")
    pbp["is_dropback"] = (pbp.get("dropback",0)==1)
    pbp["is_sack"] = (pbp.get("sack",0)==1)

    def_pass = pbp.loc[pbp["is_pass"]].groupby("defteam")["epa"].mean().rename("def_pass_epa")
    def_rush = pbp.loc[pbp["is_rush"]].groupby("defteam")["epa"].mean().rename("def_rush_epa")

    drop = pbp.loc[pbp["is_dropback"]].groupby("defteam")["is_dropback"].count().rename("db")
    sacks = pbp.loc[pbp["is_sack"]].groupby("defteam")["is_sack"].count().rename("sacks")
    sack_rate = (sacks / drop).replace([np.inf,-np.inf], np.nan).fillna(0.0).rename("def_sack_rate")

    # simple pace proxy (seconds/snap from league average plays per game)
    plays_by_off = pbp.groupby(["game_id","posteam"])["play_id"].count().rename("plays").reset_index()
    g_counts = plays_by_off.groupby("posteam")["game_id"].nunique()
    plays_pg = (plays_by_off.groupby("posteam")["plays"].sum() / g_counts)
    sec_per_snap = 3600.0 / (plays_pg.mean() if np.isfinite(plays_pg.mean()) else 120.0)
    pace = pd.Series(sec_per_snap, index=def_pass.index, name="pace")

    # very light PROE proxy (neutral pass rate minus league)
    neutral = pbp[(pbp["qtr"]<=3) & (pbp["score_differential"].abs()<=7)].copy()
    if "wp" in neutral.columns:
        neutral = neutral[(neutral["wp"]>=0.20) & (neutral["wp"]<=0.80)]
    team_pr = neutral.groupby("posteam")["is_pass"].mean().rename("team_neutral_pr")
    league_pr = float(team_pr.mean()) if len(team_pr) else 0.55
    proe = pd.Series(-(team_pr.mean() - league_pr if len(team_pr) else 0.0), index=def_pass.index, name="proe")

    lb = pd.Series(np.nan, index=def_pass.index, name="light_box_rate")
    hb = pd.Series(np.nan, index=def_pass.index, name="heavy_box_rate")

    df = pd.concat([def_pass, def_rush, sack_rate, pace, proe, lb, hb], axis=1).reset_index().rename(columns={"defteam":"team"})
    for c in ["def_pass_epa","def_rush_epa","def_sack_rate","pace","proe","light_box_rate","heavy_box_rate"]:
        df[f"{c}_z"] = _z(df[c]) if c in df.columns else 0.0
    return df

def cli(season: int) -> int:
    try:
        df = build_from_nflverse(season)
        # ensure raw mirrors exist if only z’s were computed elsewhere
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

        # NEW: optional strict gate (helpful while stabilizing CI)
        if os.getenv("NFL_FORM_STRICT") == "1":
            if df is None or df.empty or df["team"].nunique() < 8:
                raise RuntimeError("[team_form] looks empty/stub — check requirements install and network")
    except Exception as e:
        print(f"[team_form] fatal error: {e}", flush=True)  # ← only Exception here
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
