#!/usr/bin/env python3
# scripts/enrich_player_form.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def _safe_read_csv(p: str) -> pd.DataFrame:
    pth = Path(p)
    if not pth.exists() or pth.stat().st_size < 5:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def main():
    Path("data").mkdir(exist_ok=True)
    df = _safe_read_csv("data/player_form.csv")
    if df.empty:
        print("[enrich_player_form] player_form.csv empty or missing; nothing to enrich"); return 0

    # normalize join keys
    for c in ("player","team"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "team" in df.columns:
        df["team"] = df["team"].str.upper()

    # PFR enrich
    pfr = _safe_read_csv("data/pfr_player_enrich.csv")
    if not pfr.empty:
        pfr["player"] = pfr["player"].astype(str).str.strip()
        pfr["team"]   = pfr["team"].astype(str).str.upper()
        pfr = pfr.rename(columns={"routes_per_dropback":"route_rate_enrich",
                                  "yprr_proxy_est":"yprr_proxy_enrich"})
        df = df.merge(pfr[["player","team","route_rate_enrich","yprr_proxy_enrich"]],
                      on=["player","team"], how="left")
        if "route_rate" in df.columns:
            df["route_rate"] = df["route_rate"].fillna(df["route_rate_enrich"])
        else:
            df["route_rate"] = df["route_rate_enrich"]
        if "yprr_proxy" in df.columns:
            df["yprr_proxy"] = df["yprr_proxy"].fillna(df["yprr_proxy_enrich"])
        else:
            df["yprr_proxy"] = df["yprr_proxy_enrich"]
        df.drop(columns=["route_rate_enrich","yprr_proxy_enrich"], inplace=True, errors="ignore")
        print("[enrich_player_form] merged PFR routes/yprr_proxy")

    # Depth charts (prefer ESPN then OurLads)
    dc = _safe_read_csv("data/depth_chart_espn.csv")
    if dc.empty:
        dc = _safe_read_csv("data/depth_chart_ourlads.csv")
    if not dc.empty:
        dc["player"] = dc["player"].astype(str).str.strip()
        dc["team"]   = dc["team"].astype(str).str.upper()
        dc["position"] = dc["position"].astype(str).str.upper()
        dc["role"]   = dc["role"].astype(str).str.upper()
        df = df.merge(dc[["player","team","position","role"]],
                      on=["player","team"], how="left", suffixes=("","__dc"))
        for col in ("position","role"):
            src = f"{col}__dc"
            if src in df.columns:
                mask = df[col].isna() & df[src].notna()
                if mask.any():
                    df.loc[mask, col] = df.loc[mask, src]
                df.drop(columns=[src], inplace=True, errors="ignore")
        print("[enrich_player_form] merged depth chart position/role")

    df.to_csv("data/player_form.csv", index=False)
    print("[enrich_player_form] updated data/player_form.csv rows={}".format(len(df))); return 0

if __name__ == "__main__":
    raise SystemExit(main())
