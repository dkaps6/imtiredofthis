from __future__ import annotations
import argparse, pandas as pd
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--season", type=int, required=True); a=ap.parse_args()
    Path("data").mkdir(exist_ok=True)

    tfp = Path("data/team_form.csv"); pfp = Path("data/player_form.csv")
    if not tfp.exists() or tfp.stat().st_size==0:
        print("[metrics] team_form missing/empty; wrote empty metrics_ready.csv")
        (Path("data")/"metrics_ready.csv").write_text(""); return
    if not pfp.exists() or pfp.stat().st_size==0:
        print("[metrics] player_form missing/empty; wrote empty metrics_ready.csv")
        (Path("data")/"metrics_ready.csv").write_text(""); return

    tf = pd.read_csv(tfp); pf = pd.read_csv(pfp)
    for df in (tf,pf):
        if "team" in df.columns: df["team"]=df["team"].astype(str).str.upper()

    mf = pf.merge(tf, on="team", how="left", suffixes=("","_team"))
    mf.to_csv("data/metrics_ready.csv", index=False)
    print(f"[metrics] rows={len(mf)} → data/metrics_ready.csv")

if __name__=="__main__":
    main()
