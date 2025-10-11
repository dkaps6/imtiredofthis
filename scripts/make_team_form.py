#!/usr/bin/env python3
from __future__ import annotations
import sys, os, time
from pathlib import Path
import pandas as pd

OUT = Path("data/team_form.csv")

def _safe_write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # Always write at least headers to avoid 0B files
    if df is None or df.empty:
        pd.DataFrame(columns=[
            "team","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","proe_z","light_box_rate_z","heavy_box_rate_z"
        ]).to_csv(out, index=False)
    else:
        df.to_csv(out, index=False)

def build_from_nflverse(season: int) -> pd.DataFrame:
    # Replace this with your real nflverse pull
    # (kept minimal so the pipeline runs even if nflverse is unreachable)
    try:
        # EXAMPLE stub (you will replace with your real join)
        df = pd.DataFrame([
            {"team":"BUF","def_pass_epa_z":0.1,"def_rush_epa_z":-0.2,"def_sack_rate_z":0.3,
             "pace_z":0.1,"proe_z":0.2,"light_box_rate_z":-0.1,"heavy_box_rate_z":0.0}
        ])
        return df
    except Exception as e:
        print(f"[team_form] nflverse error: {e}", flush=True)
        return pd.DataFrame()

def cli(season: int) -> int:
    try:
        df = build_from_nflverse(season)
    except Exception as e:
        # Catch everything explicitly (fixes: name 'Error' is not defined)
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
