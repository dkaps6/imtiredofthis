#!/usr/bin/env python3
import os, pandas as pd

PF = "data/player_form_consensus.csv"
OM = "data/opponent_map_from_props.csv"
OUT = PF

def safe_read_csv(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        df = pd.read_csv(path)
        return df if len(df.columns) else None
    except Exception:
        return None

def norm_series(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(r"[^A-Za-z\-\.\' ]", "", regex=True)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip().str.lower())

pf = safe_read_csv(PF)
om = safe_read_csv(OM)

if pf is None or om is None:
    print(f"[merge_opponent] SKIP: missing/empty inputs (pf={pf is not None}, om={om is not None})")
    raise SystemExit(0)

for c in ("player","team","opponent"):
    if c in pf: pf[c] = pf[c].astype(str).str.strip()
    if c in om: om[c] = om[c].astype(str).str.strip()

pf["player_key"] = norm_series(pf["player"]) if "player" in pf else ""
om["player_key"] = norm_series(om["player"]) if "player" in om else ""

# Pass 1: (player, team, week)
keys1 = [k for k in ("player_key","team","week") if k in pf.columns and k in om.columns]
merged = pf.merge(om[[c for c in ("player_key","team","week","opponent") if c in om.columns]].drop_duplicates(),
                  on=keys1, how="left", suffixes=("","_m1"))
if "opponent_m1" in merged:
    merged["opponent"] = merged.get("opponent").combine_first(merged["opponent_m1"])
    merged.drop(columns=["opponent_m1"], inplace=True)

# Pass 2: (player, week) fallback
if ("opponent" not in merged) or merged["opponent"].isna().any():
    keys2 = [k for k in ("player_key","week") if k in merged.columns and k in om.columns]
    m2 = merged.merge(om[[c for c in ("player_key","week","opponent") if c in om.columns]].drop_duplicates(),
                      on=keys2, how="left", suffixes=("","_m2"))
    if "opponent_m2" in m2:
        m2["opponent"] = m2.get("opponent").combine_first(m2["opponent_m2"])
        m2.drop(columns=["opponent_m2"], inplace=True)
    merged = m2

merged.to_csv(OUT, index=False)
print(f"[merge_opponent] wrote {OUT} with {(merged['opponent'].notna()).sum()} mapped opponents.")
