#!/usr/bin/env python3
"""Mechanical A/B validator for RB Rush+Receiving Conservation V2."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "RB_RUSH_REC_CONSERVATION_V2"
OFFER_KEY = ["event_id", "player", "team", "source_market", "vegas_line", "side", "book"]


def _base_key(v) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(v or "").lower())
    toks = [t for t in s.split() if t not in {"jr","sr","ii","iii","iv","v","vi","vii"}]
    return "".join(toks)


def _uniq_projection(df: pd.DataFrame, market: str) -> pd.DataFrame:
    q=df[df["market"].astype(str).eq(market)].copy()
    q["_key"]=q["player"].map(_base_key)
    spread=q.groupby(["event_id","team","_key"])["model_proj"].nunique(dropna=False)
    if (spread>1).any():
        raise RuntimeError(f"{market} model_proj varies by offer: {spread[spread>1].head(20).to_dict()}")
    return q.drop_duplicates(["event_id","team","_key"])[["event_id","team","_key","model_proj"]]


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline",type=Path,required=True)
    ap.add_argument("--candidate",type=Path,required=True)
    ap.add_argument("--graded",type=Path)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    b=pd.read_csv(a.baseline,low_memory=False)
    c=pd.read_csv(a.candidate,low_memory=False)
    for df in (b,c):
        for col in OFFER_KEY:
            if col not in df.columns:
                raise RuntimeError(f"priced output missing offer key {col}")
    if b.duplicated(OFFER_KEY).any() or c.duplicated(OFFER_KEY).any():
        raise RuntimeError("duplicate priced offer identity")
    bm=b.set_index(OFFER_KEY,drop=False)
    cm=c.set_index(OFFER_KEY,drop=False)
    if set(bm.index)!=set(cm.index):
        raise RuntimeError(f"candidate offer identity changed baseline={len(bm)} candidate={len(cm)}")

    joined=bm[["market","model_proj","fair_prob","model_sd"]].join(
        cm[["market","model_proj","fair_prob","model_sd",
            "rb_rush_rec_conservation_v2_applied",
            "rb_rush_rec_conservation_v2_version",
            "rb_rush_rec_conservation_v2_target_mean",
            "rb_rush_rec_conservation_v2_rush_mean",
            "rb_rush_rec_conservation_v2_rec_mean"]],
        lsuffix="_baseline",rsuffix="_candidate",how="inner",validate="one_to_one"
    ).reset_index()

    market_same=joined["market_baseline"].astype(str).eq(joined["market_candidate"].astype(str))
    if not market_same.all():
        raise RuntimeError("candidate changed canonical market identity")

    combo=joined["market_candidate"].astype(str).eq("rush_rec_yards")
    applied=pd.to_numeric(joined["rb_rush_rec_conservation_v2_applied"],errors="coerce").fillna(0).eq(1)
    changed=(pd.to_numeric(joined.model_proj_candidate,errors="coerce")-pd.to_numeric(joined.model_proj_baseline,errors="coerce")).abs()
    fair_changed=(pd.to_numeric(joined.fair_prob_candidate,errors="coerce")-pd.to_numeric(joined.fair_prob_baseline,errors="coerce")).abs()

    noncombo=~combo
    max_noncombo_proj=float(changed[noncombo].max()) if noncombo.any() else 0.0
    max_noncombo_prob=float(fair_changed[noncombo].max()) if noncombo.any() else 0.0

    cand=c.copy()
    cand["_key"]=cand["player"].map(_base_key)
    rush=_uniq_projection(cand,"rush_yards").rename(columns={"model_proj":"rush_model_proj"})
    rec=_uniq_projection(cand,"rec_yards").rename(columns={"model_proj":"rec_model_proj"})
    combo_rows=cand[cand["market"].astype(str).eq("rush_rec_yards")].copy()
    combo_rows=combo_rows[pd.to_numeric(combo_rows.get("rb_rush_rec_conservation_v2_applied",0),errors="coerce").fillna(0).eq(1)]
    if combo_rows.empty:
        raise RuntimeError("candidate applied to zero rush_rec_yards rows")
    combo_unique=combo_rows.drop_duplicates(["event_id","team","_key"])[[
        "event_id","team","_key","player","model_proj",
        "rb_rush_rec_conservation_v2_target_mean",
        "rb_rush_rec_conservation_v2_rush_mean",
        "rb_rush_rec_conservation_v2_rec_mean"
    ]]
    x=combo_unique.merge(rush,on=["event_id","team","_key"],how="left",validate="one_to_one")
    x=x.merge(rec,on=["event_id","team","_key"],how="left",validate="one_to_one")
    if x[["rush_model_proj","rec_model_proj"]].isna().any().any():
        raise RuntimeError("candidate combo missing priced standalone component")
    x["priced_component_sum"]=x.rush_model_proj+x.rec_model_proj
    x["final_gap"]=pd.to_numeric(x.model_proj,errors="coerce")-x.priced_component_sum
    x["adapter_gap"]=pd.to_numeric(x.rb_rush_rec_conservation_v2_target_mean,errors="coerce")-x.priced_component_sum
    max_final_gap=float(x.final_gap.abs().max())
    max_adapter_gap=float(x.adapter_gap.abs().max())

    versions=set(combo_rows["rb_rush_rec_conservation_v2_version"].fillna("").astype(str))
    gates={
        "same_offer_identity": True,
        "noncombo_model_proj_exact": max_noncombo_proj <= 1e-10,
        "noncombo_fair_prob_exact": max_noncombo_prob <= 1e-10,
        "candidate_applied_only_combo": bool((~applied | combo).all()),
        "candidate_version_exact": versions == {VERSION},
        "combo_final_equals_priced_component_sum": max_final_gap <= 1e-8,
        "combo_adapter_equals_priced_component_sum": max_adapter_gap <= 1e-8,
        "no_duplicate_offers": True,
    }

    observational={}
    if a.graded and a.graded.is_file():
        g=pd.read_csv(a.graded,low_memory=False)
        g=g[(pd.to_numeric(g.get("week"),errors="coerce").eq(2)) & g["market"].astype(str).eq("rush_rec_yards")].copy()
        g["_key"]=g["player"].map(_base_key)
        g["actual"]=pd.to_numeric(g["actual"],errors="coerce")
        actual=g.dropna(subset=["actual"]).drop_duplicates(["event_id","team","_key"])[["event_id","team","_key","actual"]]
        q=x.merge(actual,on=["event_id","team","_key"],how="inner",validate="one_to_one")
        if len(q):
            # Baseline unique combo mean from baseline A/B run.
            bb=b[b["market"].astype(str).eq("rush_rec_yards")].copy()
            bb["_key"]=bb["player"].map(_base_key)
            bb=bb.drop_duplicates(["event_id","team","_key"])[["event_id","team","_key","model_proj"]].rename(columns={"model_proj":"baseline_proj"})
            q=q.merge(bb,on=["event_id","team","_key"],how="left",validate="one_to_one")
            be=(q.actual-q.baseline_proj).abs(); ce=(q.actual-q.model_proj).abs()
            observational={
                "n":int(len(q)),
                "baseline_mae":float(be.mean()),
                "candidate_mae":float(ce.mean()),
                "mae_gain":float(be.mean()-ce.mean()),
                "baseline_bias":float((q.actual-q.baseline_proj).mean()),
                "candidate_bias":float((q.actual-q.model_proj).mean()),
                "candidate_closer":int((ce<be).sum()),
            }

    x.to_csv(a.out_dir/"combo_component_conservation.csv",index=False)
    payload={
        "study":"RB_RUSH_REC_CONSERVATION_V2_INTEGRATION",
        "disposition":"RB_RUSH_REC_CONSERVATION_V2_INTEGRATION_PASS" if all(gates.values()) else "RB_RUSH_REC_CONSERVATION_V2_INTEGRATION_FAIL",
        "baseline_rows":int(len(b)),
        "candidate_rows":int(len(c)),
        "applied_offer_rows":int(applied.sum()),
        "applied_player_games":int(len(x)),
        "max_noncombo_model_proj_gap":max_noncombo_proj,
        "max_noncombo_fair_prob_gap":max_noncombo_prob,
        "max_combo_final_component_gap":max_final_gap,
        "max_combo_adapter_component_gap":max_adapter_gap,
        "gates":gates,
        "week2_observational_only":observational,
        "sportsbook_inputs_added_to_football":0,
        "production_changed":False,
    }
    (a.out_dir/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
