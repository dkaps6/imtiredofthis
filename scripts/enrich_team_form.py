#!/usr/bin/env python3
# scripts/enrich_team_form.py
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
    df = _safe_read_csv("data/team_form.csv")
    if df.empty:
        print("[enrich_team_form] team_form.csv empty; nothing to enrich"); return 0

    # (Optional) merge PFR team dropbacks for diagnostics
    pfrt = _safe_read_csv("data/pfr_team_enrich.csv")
    if not pfrt.empty:
        pfrt = pfrt.rename(columns={"team_abbr":"team"})
        pfrt["team"] = pfrt["team"].astype(str).str.upper()
        df = df.merge(pfrt[["team","team_dropbacks"]], on="team", how="left")
        print("[enrich_team_form] merged PFR team_dropbacks (diagnostic)")

    df.to_csv("data/team_form.csv", index=False)
    print("[enrich_team_form] updated data/team_form.csv rows={}".format(len(df))); return 0

if __name__ == "__main__":
    raise SystemExit(main())
