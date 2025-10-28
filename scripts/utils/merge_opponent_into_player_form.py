#!/usr/bin/env python3
import os
import pandas as pd

PF = "data/player_form_consensus.csv"
OM = "data/opponent_map_from_props.csv"
OUT = PF

if not (os.path.exists(PF) and os.path.getsize(PF) > 0):
    print(f"[merge_opponent_into_player_form] SKIP: {PF} missing or empty")
    raise SystemExit(0)
if not (os.path.exists(OM) and os.path.getsize(OM) > 0):
    print(f"[merge_opponent_into_player_form] SKIP: {OM} missing or empty")
    raise SystemExit(0)

def norm_series(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(r"[^A-Za-z\-\.\' ]", "", regex=True)
def norm(s):
    return (s.astype(str)
              .str.replace(r"[^A-Za-z\\-\\.\\' ]", "", regex=True)
              .str.replace(r"\\s+", " ", regex=True)
              .str.strip()
              .str.lower())

pf = pd.read_csv(PF)
om = pd.read_csv(OM)

for c in ("player","team","opponent"):
    if c in pf.columns: pf[c] = pf[c].astype(str).str.strip()
    if c in om.columns: om[c] = om[c].astype(str).str.strip()

pf["player_key"] = norm_series(pf["player"]) if "player" in pf.columns else ""
om["player_key"] = norm_series(om["player"]) if "player" in om.columns else ""

left = pf
right1 = om[[c for c in ("player_key","team","week","opponent") if c in om.columns]].drop_duplicates()
keys1 = [k for k in ("player_key","team","week") if k in left.columns and k in right1.columns]
merged = left.merge(right1, on=keys1, how="left", suffixes=("", "_m1"))
if "opponent_m1" in merged.columns:
    merged["opponent"] = merged.get("opponent").combine_first(merged["opponent_m1"])
    merged = merged.drop(columns=["opponent_m1"])

if ("opponent" not in merged.columns) or merged["opponent"].isna().any():
    right2 = om[[c for c in ("player_key","week","opponent") if c in om.columns]].drop_duplicates()
    keys2 = [k for k in ("player_key","week") if k in merged.columns and k in right2.columns]
    merged2 = merged.merge(right2, on=keys2, how="left", suffixes=("", "_m2"))
    if "opponent_m2" in merged2.columns:
        merged2["opponent"] = merged2.get("opponent").combine_first(merged2["opponent_m2"])
        merged2 = merged2.drop(columns=["opponent_m2"])
    merged = merged2

merged.to_csv(OUT, index=False)
print(f"[merge_opponent_into_player_form] wrote {OUT} with {(merged['opponent'].notna()).sum()} mapped opponents.")
pf["player_key"] = norm(pf["player"] if "player" in pf.columns else pd.Series([]))
om["player_key"] = norm(om["player"] if "player" in om.columns else pd.Series([]))

keys1 = [k for k in ["player_key","team","week"] if k in pf.columns and (k in om.columns or k=="player_key")]
keys2 = [k for k in ["player_key","week"] if k in pf.columns and k in om.columns]

m = pf.merge(om[["player_key","team","week","opponent"]].drop_duplicates(), on=keys1, how="left", suffixes=("","_om1"))
if "opponent_om1" in m.columns:
    m["opponent"] = m["opponent"].combine_first(m["opponent_om1"])
    m = m.drop(columns=["opponent_om1"])

if ("opponent" not in m.columns) or m["opponent"].isna().any():
    m2 = m.merge(om[["player_key","week","opponent"]].drop_duplicates(), on=keys2, how="left", suffixes=("","_om2"))
    if "opponent_om2" in m2.columns:
        m2["opponent"] = m2["opponent"].combine_first(m2["opponent_om2"])
        m2 = m2.drop(columns=["opponent_om2"])
    m = m2

m.to_csv(OUT, index=False)
print(f"[merge_opponent_into_player_form] wrote {OUT} with {m['opponent'].notna().sum()} mapped opponents.")
