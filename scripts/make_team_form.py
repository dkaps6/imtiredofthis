#!/usr/bin/env python3
from __future__ import annotations
import sys, os, time, traceback
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("data/team_form.csv")
LOG_DIR = Path("logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
ERR_LOG = LOG_DIR / "nfl_pbp_error.txt"

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

def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0: return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

# ---- DIAGNOSTIC/RETRY WRAPPER (nflreadpy first, then nfl_data_py; 404 fallback seasons) ----
def _fetch_pbp_with_retry(season: int, tries: int = 3, wait: int = 4) -> pd.DataFrame:
    import traceback, time
    seasons_to_try = [season, season - 1, season - 2]
    last = None

    # prefer nflreadpy (mirrors nflreadr; 2025-ready)
    try:
        import nflreadpy as nfr
        use_readpy = True
    except Exception:
        use_readpy = False

    # fall back client
    if not use_readpy:
        import nfl_data_py as nfl

    with open(ERR_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n=== PBP fetch wanted season={season} ===\n")

    for s in seasons_to_try:
        for i in range(1, tries + 1):
            try:
                if use_readpy:
                    pf = nfr.load_pbp([s])     # Polars -> Pandas
                    df = pf.to_pandas()
                else:
                    df = nfl.import_pbp_data([s])

                if df is not None and len(df):
                    with open(ERR_LOG, "a", encoding="utf-8") as f:
                        f.write(f"season {s} try {i}: OK rows={len(df)} via "
                                f"{'nflreadpy' if use_readpy else 'nfl_data_py'}\n")
                    if s != season:
                        print(f"[team_form] NOTE: using season {s} as fallback for {season}", flush=True)
                    return df

                raise RuntimeError("PBP fetch returned empty dataframe")
            except Exception as e:
                last = e
                tb = traceback.format_exc()
                msg = f"{type(e).__name__}: {e}"
                print(f"[team_form] season {s} try {i}/{tries} failed: {msg}", flush=True)
                with open(ERR_LOG, "a", encoding="utf-8") as f:
                    f.write(f"season {s} try {i}: {msg}\n{tb}\n")
                # If nfl_data_py hit a 404 for current season, advance season quickly
                if (not use_readpy) and ("HTTP Error 404" in str(e)):
                    print(f"[team_form] season {s}: 404 detected; trying previous season…", flush=True)
                    break
                time.sleep(wait)

    raise RuntimeError(f"PBP fetch failed for {season}: {type(last).__name__}: {last}")

def build_from_nflverse(season: int) -> pd.DataFrame:
    try:
        import nfl_data_py as nfl  # noqa: F401
    except Exception as e:
        print(f"[team_form] nfl_data_py import failed → fallback: {e}", flush=True)
        return pd.DataFrame([
            {"team":"BUF","def_pass_epa":0.10,"def_rush_epa":-0.05,"def_sack_rate":0.08,
             "pace":30.0,"proe":0.02,"light_box_rate":np.nan,"heavy_box_rate":np.nan}
        ])

    # guard against shadowing; show exactly what got imported
    import importlib
    nfl_mod = importlib.import_module("nfl_data_py")
    nfl_path = getattr(nfl_mod, "__file__", "")
    print(f"[team_form] nfl_data_py path → {nfl_path}", flush=True)
    if "site-packages" not in (nfl_path or "") and "dist-packages" not in (nfl_path or ""):
        raise RuntimeError(f"Wrong nfl_data_py imported (shadowed). Path: {nfl_path}")

    print("[team_form] pulling pbp…", flush=True)
    pbp = _fetch_pbp_with_retry(season)
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

    plays_by_off = pbp.groupby(["game_id","posteam"])["play_id"].count().rename("plays").reset_index()
    g_counts = plays_by_off.groupby("posteam")["game_id"].nunique()
    plays_pg = (plays_by_off.groupby("posteam")["plays"].sum() / g_counts)
    sec_per_snap = 3600.0 / (plays_pg.mean() if np.isfinite(plays_pg.mean()) else 120.0)
    pace = pd.Series(sec_per_snap, index=def_pass.index, name="pace")

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
        if os.getenv("NFL_FORM_STRICT") == "1":
            if df is None or df.empty or df["team"].nunique() < 8:
                raise RuntimeError("[team_form] looks empty/stub — check requirements install and network; see logs/nfl_pbp_error.txt")
    except Exception as e:
        print(f"[team_form] fatal error: {type(e).__name__}: {e}", flush=True)
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
