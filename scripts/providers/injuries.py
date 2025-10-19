#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Injuries → data/injuries.csv

Surgical changes:
- Try nflreadpy first, then nfl_data_py.
- Better logging on which source worked/failed.
- If both fail, keep last cached data/data/injuries.csv (don’t overwrite with empty).
- Normalize columns: player, team, status
"""

import os, sys, warnings
import pandas as pd

DATA_DIR = "data"
OUT = os.path.join(DATA_DIR, "injuries.csv")

def _normalize_team(x: pd.Series) -> pd.Series:
    aliases = {
        "WSH":"WAS","WDC":"WAS","JAC":"JAX","ARZ":"ARI","LA":"LAR","LVR":"LV","OAK":"LV","SFO":"SF","TAM":"TB","GBP":"GB","KAN":"KC",
    }
    s = x.astype(str).str.upper().str.strip()
    return s.replace(aliases)

def _normalize_player(x: pd.Series) -> pd.Series:
    return (
        x.fillna("")
         .astype(str)
         .str.replace(".", "", regex=False)
         .str.replace(r"\s+(JR|SR|II|III|IV|V)\.?$", "", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )

def load_injuries_nflreadpy(season: int) -> pd.DataFrame:
    try:
        import nflreadpy as nflv
        df = nflv.load_injuries(seasons=[season])
        df = df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)
        df.columns = [c.lower() for c in df.columns]
        # common columns: player_name, team, report_status / status
        if "player_name" in df.columns and "team" in df.columns:
            df = df.rename(columns={"player_name":"player"})
            if "report_status" in df.columns and "status" not in df.columns:
                df["status"] = df["report_status"]
            return df[["player","team","status"]]
    except Exception as e:
        print(f"[injuries] nflreadpy failed: {e}", file=sys.stderr)
    return pd.DataFrame()

def load_injuries_nfl_data_py(season: int) -> pd.DataFrame:
    try:
        import nfl_data_py as nfld
        df = nfld.import_injuries([season])  # type: ignore
        df.columns = [c.lower() for c in df.columns]
        if "player_name" in df.columns and "team" in df.columns:
            df = df.rename(columns={"player_name":"player"})
            if "injury_status" in df.columns and "status" not in df.columns:
                df["status"] = df["injury_status"]
            return df[["player","team","status"]]
    except Exception as e:
        print(f"[injuries] nfl_data_py failed: {e}", file=sys.stderr)
    return pd.DataFrame()

def main():
    warnings.simplefilter("ignore")
    os.makedirs(DATA_DIR, exist_ok=True)
    season = int(os.environ.get("SEASON", "2025"))

    df = load_injuries_nflreadpy(season)
    src = "nflreadpy"
    if df.empty:
        df = load_injuries_nfl_data_py(season)
        src = "nfl_data_py"

    if df.empty:
        # keep prior file if exists
        if os.path.exists(OUT):
            print("[injuries] no injury data fetched; keeping existing injuries.csv", file=sys.stderr)
            return
        else:
            # write minimal stub once (downstream handles empty gracefully)
            pd.DataFrame(columns=["player","team","status"]).to_csv(OUT, index=False)
            print("[injuries] wrote empty stub → data/injuries.csv")
            return

    df["player"] = _normalize_player(df["player"])
    df["team"] = _normalize_team(df["team"])
    df["status"] = df["status"].astype(str).str.title()

    df = df.dropna(subset=["player"]).drop_duplicates()

    df.to_csv(OUT, index=False)
    print(f"[injuries] wrote {len(df)} rows from {src} → {OUT}")

if __name__ == "__main__":
    main()
