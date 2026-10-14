#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

TOL=1e-8
WR_VARIANT="WR_R15_WR1_ANCHORED_PARTICIPATION"

def num(s): return pd.to_numeric(s,errors="coerce")
def read(p):
    x=pd.read_csv(p,low_memory=False); x.columns=[str(c).strip().lower() for c in x.columns]; return x

def consistency(x, value_col, keys):
    z=x[keys+[value_col]].dropna().copy()
    if z.empty: return {"groups":0,"max_within_group_range":None,"consistent":False}
    spans=z.groupby(keys)[value_col].agg(lambda s: float(s.max()-s.min()))
    m=float(spans.max()) if len(spans) else 0.0
    return {"groups":int(len(spans)),"max_within_group_range":m,"consistent":bool(m<=TOL)}

def audit_wr(path):
    x=read(path)
    req={"variant","season","week","team","player_clean_key","pred_targets","entitlement_tgt_share"}
    miss=req-set(x.columns)
    if miss: raise RuntimeError(f"WR authority missing {sorted(miss)}")
    x=x.loc[x["variant"].astype(str).eq(WR_VARIANT)].copy()
    x["pred_targets"]=num(x["pred_targets"]); x["entitlement_tgt_share"]=num(x["entitlement_tgt_share"])
    good=x["entitlement_tgt_share"].gt(0)&x["pred_targets"].notna()
    x.loc[good,"implied_team_targets"]=x.loc[good,"pred_targets"]/x.loc[good,"entitlement_tgt_share"]
    keys=["season","week","team"]
    if "event_id" in x.columns: keys=["season","week","event_id","team"]
    c=consistency(x.loc[good],"implied_team_targets",keys)
    return {
      "rows":int(len(x)),"players":int(x["player_clean_key"].astype(str).nunique()),
      "columns":list(x.columns),"recoverable_rows":int(good.sum()),
      "recoverable_rate":float(good.mean()) if len(x) else 0.0,
      "team_total_consistency":c,
      "pregame_team_total_ready":bool(good.any() and c["consistent"]),
      "method":"pred_targets / entitlement_tgt_share",
    }

def audit_te(path):
    x=read(path)
    req={"season","week","team","player_clean_key","candidate_targets_r5p"}
    miss=req-set(x.columns)
    if miss: raise RuntimeError(f"TE authority missing {sorted(miss)}")
    x["candidate_targets_r5p"]=num(x["candidate_targets_r5p"])
    interesting=[c for c in x.columns if any(k in c for k in ("target","share","pool","entitlement"))]
    share_candidates=[
      "candidate_entitlement_tgt_share","candidate_tgt_share_r5p","candidate_target_share_r5p",
      "candidate_share_r5p","candidate_entitlement_share_r5p","entitlement_tgt_share",
    ]
    explicit_total_candidates=["pred_team_targets","team_pred_targets","expected_team_targets","b0_team_targets"]
    method=None; recover=pd.Series(np.nan,index=x.index,dtype=float)
    for c in explicit_total_candidates:
        if c in x.columns:
            v=num(x[c])
            if v.notna().any():
                recover=v; method=c; break
    if method is None:
        for c in share_candidates:
            if c in x.columns:
                sh=num(x[c]); good=sh.gt(0)&x["candidate_targets_r5p"].notna()
                if good.any():
                    recover.loc[good]=x.loc[good,"candidate_targets_r5p"]/sh.loc[good]
                    method=f"candidate_targets_r5p / {c}"; break
    # Try exact baseline ratio if the casebook preserves baseline share/count.
    if method is None:
        pairs=[
          ("b0_expected_targets","baseline_entitlement_tgt_share"),
          ("b0_expected_targets","b0_entitlement_tgt_share"),
          ("b0_expected_targets","b0_tgt_share"),
          ("b0_targets","baseline_entitlement_tgt_share"),
        ]
        for count_col,share_col in pairs:
            if count_col in x.columns and share_col in x.columns:
                cnt=num(x[count_col]); sh=num(x[share_col]); good=cnt.notna()&sh.gt(0)
                if good.any():
                    recover.loc[good]=cnt.loc[good]/sh.loc[good]
                    method=f"{count_col} / {share_col}"; break
    x["implied_team_targets"]=recover
    keys=["season","week","team"]
    if "event_id" in x.columns: keys=["season","week","event_id","team"]
    c=consistency(x.loc[x["implied_team_targets"].notna()],"implied_team_targets",keys)
    ready=bool(x["implied_team_targets"].notna().any() and c["consistent"])
    return {
      "rows":int(len(x)),"players":int(x["player_clean_key"].astype(str).nunique()),
      "columns":list(x.columns),"interesting_columns":interesting,
      "recoverable_rows":int(x["implied_team_targets"].notna().sum()),
      "recoverable_rate":float(x["implied_team_targets"].notna().mean()) if len(x) else 0.0,
      "team_total_consistency":c,
      "pregame_team_total_ready":ready,
      "method":method,
      "replay_required":bool(not ready),
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--wr",type=Path,required=True); ap.add_argument("--te",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    wr=audit_wr(a.wr); te=audit_te(a.te)
    if wr["pregame_team_total_ready"] and te["pregame_team_total_ready"]:
        disp="PLAYER_OPPORTUNITY_DIFFICULTY_TARGET_ALLOCATION_SOURCE_READY"
    elif wr["pregame_team_total_ready"] and te["replay_required"]:
        disp="PLAYER_OPPORTUNITY_DIFFICULTY_TARGET_ALLOCATION_REPLAY_REQUIRED"
    else:
        disp="PLAYER_OPPORTUNITY_DIFFICULTY_TARGET_ALLOCATION_SOURCE_FAILURE"
    out={"version":"PLAYER_OPPORTUNITY_DIFFICULTY_TARGET_ALLOCATION_SOURCE_V1","disposition":disp,
         "wr":wr,"te":te,"sportsbook_inputs_used":0,"oddsapi_calls":0,"outcomes_2026_read":0,"production_changed":False}
    (a.out_dir/"result.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__": main()
