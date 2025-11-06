#!/usr/bin/env python3
"""
Build manual_name_overrides.csv from the FINAL player_form file
that already has the correct full names.

Input columns required in the final CSV:
  - player_source_name  (key like 'Zflowers')
  - player              (full name like 'Zay Flowers')

Output:
  data/manual_name_overrides.csv with columns:
  - player_source_name
  - full_name
"""

import argparse
from pathlib import Path
import pandas as pd

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-csv", type=Path, required=True,
                    help="Path to the approved final player_form CSV with full names.")
    ap.add_argument("--out", type=Path, default=Path("data/manual_name_overrides.csv"))
    ns = ap.parse_args(argv)

    df = pd.read_csv(ns.final_csv)
    need = {"player_source_name", "player"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{ns.final_csv} must contain columns: {sorted(need)}")

    out = (
        df[["player_source_name", "player"]]
        .rename(columns={"player": "full_name"})
        .dropna()
        .drop_duplicates(subset=["player_source_name"])
        .sort_values("player_source_name")
    )
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ns.out, index=False)
    print(f"Wrote {ns.out} ({len(out)} rows).")

if __name__ == "__main__":
    main()
