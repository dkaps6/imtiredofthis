#!/usr/bin/env python3
# scripts/build_game_lines_from_schedule.py
# Build outputs/game_lines.csv using nfl_data_py schedules (no API keys needed).
# Produces: event_id (synthetic), home_team, away_team, home_wp, away_wp, commence_time, week, season
# home_wp/away_wp left NaN (you can fill them later from odds if desired).

import os, sys, pandas as pd, numpy as np

OUT = "outputs/game_lines.csv"

def _normalize_team_names(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    norm = s.astype(str).str.upper().str.strip()
    aliases = {"WSH":"WAS","WDC":"WAS","JAC":"JAX","ARZ":"ARI","AZ":"ARI","LA":"LAR","LVR":"LV","OAK":"LV","SFO":"SF","TAM":"TB","GBP":"GB","KAN":"KC","NOS":"NO","SD":"LAC"}
    return norm.replace(aliases)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, default=None, help="If provided, filter to this week only")
    args = parser.parse_args()

    try:
        import nfl_data_py as nfl
        sched = nfl.import_schedules([args.season])
    except Exception as e:
        print(f"[schedule] Unable to import schedules via nfl_data_py: {e}", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(sched)
    if df.empty:
        print("[schedule] empty schedules", file=sys.stderr); sys.exit(1)

    # Normalize columns
    df.columns = [c.lower() for c in df.columns]
    keep = [c for c in ["game_id","home_team","away_team","gameday","week","season"] if c in df.columns]
    df = df[keep].copy()

    if "gameday" in df.columns:
        df["commence_time"] = pd.to_datetime(df["gameday"], errors="coerce", utc=True)
    elif "game_date" in df.columns:
        df["commence_time"] = pd.to_datetime(df["game_date"], errors="coerce", utc=True)
    else:
        df["commence_time"] = pd.NaT

    df["home_team"] = _normalize_team_names(df["home_team"])
    df["away_team"] = _normalize_team_names(df["away_team"])

    if args.week is not None and "week" in df.columns:
        df = df[df["week"] == args.week]

    # Synthesize event_id if none
    if "game_id" in df.columns:
        df["event_id"] = df["game_id"]
    else:
        df["event_id"] = (df["season"].astype(str) + "_" + df["week"].astype(str) + "_" + df["home_team"] + "_" + df["away_team"])

    out = df.rename(columns={"gameday":"commence_time"})[["event_id","home_team","away_team","commence_time","week","season"]].copy()
    out["home_wp"] = np.nan
    out["away_wp"] = np.nan

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[schedule] wrote {OUT} rows={len(out)}")

if __name__ == "__main__":
    main()
