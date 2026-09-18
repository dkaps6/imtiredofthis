#!/usr/bin/env python3
"""Build outcome-free qualification profiles for football-context candidate signals.

Engineering/QA only. Profiles a pregame feature table and optional stability evidence;
never reads target-game outcomes and never fits a projection model.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--features", required=True, help="comma-separated feature columns")
    ap.add_argument("--family", required=True)
    ap.add_argument("--grain", required=True)
    ap.add_argument("--key-cols", required=True, help="comma-separated canonical key columns")
    ap.add_argument("--eligible-col", help="optional boolean/0-1 pregame eligibility column")
    ap.add_argument("--stable-id-col", help="optional boolean/0-1 stable identity indicator")
    ap.add_argument("--unknown-col", help="optional boolean/0-1 explicit unknown-state indicator")
    ap.add_argument("--prior-support-col", help="optional numeric strict-prior support count")
    ap.add_argument("--season-col", default="season")
    ap.add_argument("--stability", help="optional CSV from build_context_stability_evidence.py")
    ap.add_argument("--intended-component", required=True)
    ap.add_argument("--mechanism-note", required=True)
    ap.add_argument("--source-class", default="CANONICAL")
    ap.add_argument("--redundancy-notes", default="not yet audited")
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    df=pd.read_csv(args.input)
    features=[x.strip() for x in args.features.split(",") if x.strip()]
    keys=[x.strip() for x in args.key_cols.split(",") if x.strip()]
    required=[*keys,args.season_col,*features]
    for c in (args.eligible_col,args.stable_id_col,args.unknown_col,args.prior_support_col):
        if c: required.append(c)
    missing=[c for c in required if c not in df.columns]
    if missing: raise SystemExit(f"missing required columns: {missing}")
    if not keys: raise SystemExit("at least one canonical key column is required")

    duplicate_key_count=int(df.duplicated(keys, keep=False).sum())
    eligible=(pd.to_numeric(df[args.eligible_col],errors="coerce").fillna(0)>0) if args.eligible_col else pd.Series(True,index=df.index)
    eligible_rows=int(eligible.sum())
    stable=(pd.to_numeric(df[args.stable_id_col],errors="coerce").fillna(0)>0) if args.stable_id_col else pd.Series(True,index=df.index)
    unknown=(pd.to_numeric(df[args.unknown_col],errors="coerce").fillna(0)>0) if args.unknown_col else pd.Series(False,index=df.index)
    seasons=sorted(pd.to_numeric(df[args.season_col],errors="coerce").dropna().astype(int).unique().tolist())
    stability={}
    if args.stability:
        s=pd.read_csv(args.stability)
        if s.feature_name.duplicated().any(): raise SystemExit("duplicate feature_name in stability evidence")
        stability=s.set_index("feature_name").to_dict("index")

    rows=[]
    for feature in features:
        observed=df[feature].notna() & eligible
        support=pd.to_numeric(df.loc[observed,args.prior_support_col],errors="coerce") if args.prior_support_col else pd.Series(dtype=float)
        st=stability.get(feature,{})
        rows.append({
            "feature_name":feature,
            "family":args.family,
            "grain":args.grain,
            "seasons_available":",".join(map(str,seasons)),
            "eligible_rows":eligible_rows,
            "pregame_coverage":float(observed.sum()/eligible_rows) if eligible_rows else 0.0,
            "stable_id_coverage":float(stable[eligible].mean()) if eligible_rows else 0.0,
            "unknown_rate":float(unknown[eligible].mean()) if eligible_rows else 0.0,
            "duplicate_key_count":duplicate_key_count,
            "fanout_count":0,
            "prior_support_median":float(support.median()) if not support.empty else 0.0,
            "stability_stat":st.get("stability_stat",""),
            "stability_value":st.get("stability_value",float("nan")),
            "intended_component":args.intended_component,
            "redundancy_notes":args.redundancy_notes,
            "source_class":args.source_class,
            "mechanism_note":args.mechanism_note,
        })
    out=pd.DataFrame(rows).sort_values("feature_name")
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(path,index=False)
    print(out.to_string(index=False))

if __name__=="__main__": main()
