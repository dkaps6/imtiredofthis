#!/usr/bin/env python3
"""Materialize the frozen Football Context Signal Qualification V1 inventory.

Engineering/QA only. This script never reads target-game outcomes and never fits a model.
It consumes one or more candidate-profile CSVs and emits a normalized qualification table.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REQUIRED = [
    "feature_name","family","grain","seasons_available","eligible_rows",
    "pregame_coverage","stable_id_coverage","unknown_rate","duplicate_key_count",
    "fanout_count","prior_support_median","stability_stat","stability_value",
    "intended_component","redundancy_notes","source_class","mechanism_note"
]

DISPOSITIONS = {
    "READY_FOR_FROZEN_EXPERIMENT",
    "ENGINEERING_READY_SOURCE_THIN",
    "DESCRIPTIVE_ONLY",
    "SOURCE_BLOCKED",
    "REJECTED_INTEGRITY",
}

def qualify(r: pd.Series, min_coverage: float, min_rows: int) -> tuple[str,str]:
    if int(r.duplicate_key_count) != 0 or int(r.fanout_count) != 0:
        return "REJECTED_INTEGRITY", "duplicate published keys or canonical-base fanout"
    if float(r.stable_id_coverage) < 0.99:
        return "REJECTED_INTEGRITY", "stable-ID coverage below frozen integrity floor"
    if not str(r.mechanism_note).strip() or not str(r.intended_component).strip():
        return "DESCRIPTIVE_ONLY", "no specific projection-component mechanism"
    if str(r.source_class).upper() == "BLOCKED":
        return "SOURCE_BLOCKED", "authoritative source unavailable"
    if int(r.eligible_rows) < min_rows or float(r.pregame_coverage) < min_coverage:
        return "ENGINEERING_READY_SOURCE_THIN", "pregame coverage or sample support below qualification floor"
    if pd.isna(r.stability_value) or not str(r.stability_stat).strip():
        return "ENGINEERING_READY_SOURCE_THIN", "strict-prior stability evidence not yet attached"
    return "READY_FOR_FROZEN_EXPERIMENT", "integrity, support, stability and mechanism gates satisfied"

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pregame-coverage", type=float, default=0.80)
    ap.add_argument("--min-eligible-rows", type=int, default=500)
    args=ap.parse_args()
    frames=[pd.read_csv(p) for p in args.inputs]
    df=pd.concat(frames, ignore_index=True)
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise SystemExit(f"missing required columns: {missing}")
    if df.feature_name.duplicated().any(): raise SystemExit("duplicate feature_name rows in qualification inventory")
    results=[qualify(r,args.min_pregame_coverage,args.min_eligible_rows) for _,r in df.iterrows()]
    df["qualification_disposition"]=[x[0] for x in results]
    df["qualification_reason"]=[x[1] for x in results]
    if not set(df.qualification_disposition).issubset(DISPOSITIONS): raise SystemExit("invalid disposition")
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    df.sort_values(["qualification_disposition","family","feature_name"]).to_csv(out,index=False)
    print(df.qualification_disposition.value_counts().to_string())

if __name__ == "__main__": main()
