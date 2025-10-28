#!/usr/bin/env python3
import pandas as pd

PF = "data/player_form_consensus.csv"
OM = "data/opponent_map_from_props.csv"
OUT = PF

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
