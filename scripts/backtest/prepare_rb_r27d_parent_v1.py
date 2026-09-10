#!/usr/bin/env python3
"""Normalize the immutable R27B V2 parent into the exact R27D column contract.

This is value-neutral plumbing only. It does not fit a model or alter any parent
prediction. B0/B1 must reproduce the immutable R27 columns exactly.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def num(x): return pd.to_numeric(x, errors="coerce")


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True,type=Path); ap.add_argument("--output",required=True,type=Path)
    a=ap.parse_args(); d=pd.read_csv(a.input).copy()
    required=["season","week","player_clean_key","team","opponent","role","vacancy_active","vacancy_incumbent",
              "baseline_targets","candidate_targets","baseline_receptions","candidate_receptions","production_ypt",
              "production_catch_rate","r27_baseline_rec_yards","r27_candidate_rec_yards","b0_rec_yards","b1_rec_yards","actual_rec_yards"]
    miss=[c for c in required if c not in d.columns]
    if miss: raise RuntimeError(f"immutable parent missing required columns: {miss}")
    b0_gap=float((num(d.b0_rec_yards)-num(d.r27_baseline_rec_yards)).abs().max())
    b1_gap=float((num(d.b1_rec_yards)-num(d.r27_candidate_rec_yards)).abs().max())
    if b0_gap>1e-10 or b1_gap>1e-10: raise RuntimeError(f"parent B0/B1 alias drift: {b0_gap=} {b1_gap=}")
    cr=num(d.production_catch_rate); ypt=num(d.production_ypt)
    d["production_implied_ypr"]=np.where(cr.gt(0),ypt/cr,np.nan)
    bridge=(num(d.candidate_receptions)*num(d.production_implied_ypr)-num(d.b1_rec_yards)).abs()
    bridge_max=float(bridge.max())
    if bridge_max>1e-10: raise RuntimeError(f"parent reception/YPR bridge drift: {bridge_max}")
    a.output.parent.mkdir(parents=True,exist_ok=True); d.to_csv(a.output,index=False)
    print(json.dumps({"rows":int(len(d)),"b0_alias_max_gap":b0_gap,"b1_alias_max_gap":b1_gap,"reception_ypr_bridge_max_gap":bridge_max,"model_fit":False,"candidate_created":False},indent=2,sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())
