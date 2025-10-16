#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

def _safe_read_csv(p: str) -> pd.DataFrame:
    pth = Path(p)
    if not pth.exists() or pth.stat().st_size < 5:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def _bootstrap_from_pfr() -> pd.DataFrame:
    """Create a minimal player_form from PFR when builders produced zero rows."""
    pfr = _safe_read_csv("data/pfr_player_enrich.csv")
    if pfr.empty:
        return pd.DataFrame()
    pfr["player"] = pfr["player"].astype(str).str.strip()
    pfr["team"]   = pfr["team"].astype(str).str.upper()
    df = pfr[["player","team"]].drop_duplicates().copy()
    # minimal schema your downstream expects
    for c, v in [
        ("position", np.nan), ("role", np.nan),
        ("target_share", np.nan), ("rush_share", np.nan), ("route_rate", np.nan),
        ("rz_tgt_share", np.nan), ("rz_carry_share", np.nan),
        ("yprr_proxy", np.nan), ("ypc", np.nan), ("qb_ypa", np.nan), ("ypt", np.nan),
    ]:
        df[c] = v
    return df

def main():
    Path("data").mkdir(exist_ok=True)

    df = _safe_read_csv("data/player_form.csv")
    if df.empty:
        print("[enrich_player_form] player_form.csv empty — bootstrapping from PFR…")
        df = _bootstrap_from_pfr()
        if df.empty:
            print("[enrich_player_form] PFR bootstrap unavailable; leaving player_form.csv empty")
            return 0

    # normalize join keys
    for c in ("player","team"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "team" in df.columns:
        df["team"] = df["team"].str.upper()

    # PFR routes-per-dropback + yprr proxy
    pfr = _safe_read_csv("data/pfr_player_enrich.csv")
    if not pfr.empty:
        pfr["player"] = pfr["player"].astype(str).str.strip()
        pfr["team"]   = pfr["team"].astype(str).str.upper()
        pfr = pfr.rename(columns={"routes_per_dropback":"route_rate_enrich",
                                  "yprr_proxy_est":"yprr_proxy_enrich"})
        df = df.merge(pfr[["player","team","route_rate_enrich","yprr_proxy_enrich"]],
                      on=["player","team"], how="left")
        df["route_rate"] = df.get("route_rate", pd.Series(index=df.index, dtype=float))
        df["yprr_proxy"] = df.get("yprr_proxy", pd.Series(index=df.index, dtype=float))
        df["route_rate"] = df["route_rate"].fillna(df["route_rate_enrich"])
        df["yprr_proxy"] = df["yprr_proxy"].fillna(df["yprr_proxy_enrich"])
        df.drop(columns=["route_rate_enrich","yprr_proxy_enrich"], inplace=True, errors="ignore")
        print("[enrich_player_form] merged PFR routes/yprr_proxy")

    # Depth charts (prefer ESPN then OurLads)
    dc = _safe_read_csv("data/depth_chart_espn.csv")
    if dc.empty:
        dc = _safe_read_csv("data/depth_chart_ourlads.csv")
    if not dc.empty:
        dc["player"]   = dc["player"].astype(str).str.strip()
        dc["team"]     = dc["team"].astype(str).str.upper()
        dc["position"] = dc["position"].astype(str).str.upper()
        dc["role"]     = dc["role"].astype(str).str.upper()
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

    # Fill super-basic defaults so nothing downstream explodes
    for c, v in [
        ("position", "WR"), ("role", "WR1"),
        ("target_share", 0.0), ("rush_share", 0.0),
        ("rz_tgt_share", 0.0), ("rz_carry_share", 0.0),
    ]:
        if c not in df.columns: df[c] = v
        df[c] = df[c].fillna(v)

    df.to_csv("data/player_form.csv", index=False)
    print(f"[enrich_player_form] updated data/player_form.csv rows={len(df)}"); return 0

if __name__ == "__main__":
    raise SystemExit(main())
