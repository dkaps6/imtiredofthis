#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

OUT = Path("data/player_form.csv")

def _safe_write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        # Write headers so downstream never sees a 0B file
        pd.DataFrame(columns=[
            "player","team","position","role",
            "target_share","rush_share","route_rate",
            "rz_tgt_share","rz_carry_share",
            "yprr_proxy","ypc","qb_ypa"
        ]).to_csv(out, index=False)
    else:
        df.to_csv(out, index=False)

def build_from_nflverse(season: int) -> pd.DataFrame:
    """
    Replace this stub with your real nflverse/nflreadr join.
    Keeping a minimal row so the pipeline keeps running even if feeds are slow.
    """
    try:
        return pd.DataFrame([
            {
                "player":"Example WR1","team":"BUF","position":"WR","role":"WR1",
                "target_share":0.24,"rush_share":0.00,"route_rate":0.90,
                "rz_tgt_share":0.28,"rz_carry_share":0.00,
                "yprr_proxy":2.1,"ypc":0.0,"qb_ypa":7.6
            }
        ])
    except Exception as e:
        print(f"[player_form] nflverse error: {e}", flush=True)
        return pd.DataFrame()

def cli(season: int) -> int:
    try:
        df = build_from_nflverse(season)
    except Exception as e:
        # IMPORTANT: catch generic Exception; do NOT use 'except Error'
        print(f"[player_form] fatal error: {e}", flush=True)
        df = pd.DataFrame()
    _safe_write(df, OUT)
    print(f"[player_form] wrote rows={len(df)} → {OUT}")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, type=int)
    a = ap.parse_args()
    sys.exit(cli(a.season))
