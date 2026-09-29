#!/usr/bin/env python3
"""Postgame grade of frozen Week-3 Receiving Rule Semantics V1 cells.

The four pregame entitlement cell CSVs are immutable inputs. This script does
not recompute rule semantics. It runs the canonical explicit-entitlement
simulator with production MC settings (25k, seed 42), attaches final Week-3
nflverse receiving outcomes, and reports the preregistered slices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_stats_loader_v2 import load_weekly_player_stats
from scripts.simulation_explicit_entitlement_v1 import simulate
from scripts.utils.canonical_names import canonicalize_player_name_safe

SEASON=2026
WEEK=3
CELLS=("A0B0","A1B0","A0B1","A1B1")
METRICS=("target_share","receptions","rec_yards")
IDENT=["event_id","team","player_clean_key"]

def _to_pandas(x):
    return x.to_pandas() if hasattr(x,"to_pandas") else pd.DataFrame(x)

def _pick(df,names):
    for c in names:
        if c in df.columns: return c
    raise RuntimeError(f"none of {names} present; have={sorted(df.columns)}")

def _verified_alias_keys() -> dict[tuple[str,str], str]:
    p=Path("data/player_identity_aliases.csv")
    out={}
    if not p.exists(): return out
    x=pd.read_csv(p,dtype="string").fillna("")
    for r in x.itertuples(index=False):
        team=canon_team(str(r.current_team))
        _,cur=canonicalize_player_name_safe(str(r.current_name))
        _,hist=canonicalize_player_name_safe(str(r.historical_name))
        if cur and hist:
            out[(team,cur)]=hist
            out[(team,hist)]=cur
    return out

def _load_actuals() -> tuple[pd.DataFrame,dict[str,float],pd.DataFrame]:
    stats=load_weekly_player_stats(SEASON).copy()
    stats.columns=[str(c).strip().lower() for c in stats.columns]
    stats=stats.loc[pd.to_numeric(stats["week"],errors="coerce").eq(WEEK)].copy()
    tc=_pick(stats,("recent_team","team","team_abbr","club"))
    nc=_pick(stats,("player_display_name","player_name","player"))
    targetc=_pick(stats,("targets",))
    recc=_pick(stats,("receptions",))
    yardc=_pick(stats,("receiving_yards","rec_yards"))
    stats["team"]=stats[tc].astype("string").fillna("").str.strip().map(canon_team)
    canon=stats[nc].astype("string").fillna("").str.strip().map(canonicalize_player_name_safe)
    stats["player_clean_key"]=canon.map(lambda t:t[1])
    stats["actual_targets"]=pd.to_numeric(stats[targetc],errors="coerce").fillna(0.0)
    stats["actual_receptions"]=pd.to_numeric(stats[recc],errors="coerce").fillna(0.0)
    stats["actual_rec_yards"]=pd.to_numeric(stats[yardc],errors="coerce").fillna(0.0)
    idc="player_id" if "player_id" in stats.columns else "gsis_id" if "gsis_id" in stats.columns else None
    stats["player_id_actual"]=stats[idc].astype("string").fillna("").str.strip() if idc else ""
    team_targets=stats.groupby("team")["actual_targets"].sum().to_dict()

    import nflreadpy as nfl
    roster=_to_pandas(nfl.load_rosters_weekly(SEASON)).copy()
    roster.columns=[str(c).strip().lower() for c in roster.columns]
    roster=roster.loc[pd.to_numeric(roster["week"],errors="coerce").eq(WEEK)].copy()
    rtc=_pick(roster,("team","team_abbr","club_code"))
    rnc=_pick(roster,("full_name","football_name","player_name","player"))
    roster["team"]=roster[rtc].astype("string").fillna("").str.strip().map(canon_team)
    rc=roster[rnc].astype("string").fillna("").str.strip().map(canonicalize_player_name_safe)
    roster["player_clean_key"]=rc.map(lambda t:t[1])
    ridc="gsis_id" if "gsis_id" in roster.columns else "player_id" if "player_id" in roster.columns else None
    roster["player_id_roster"]=roster[ridc].astype("string").fillna("").str.strip() if ridc else ""
    roster=roster[["team","player_clean_key","player_id_roster"]].drop_duplicates()
    return stats,team_targets,roster

def _attach_actuals(base:pd.DataFrame)->pd.DataFrame:
    stats,team_targets,roster=_load_actuals()
    aliases=_verified_alias_keys()
    rows=[]
    for r in base.itertuples(index=False):
        team=canon_team(str(r.team)); key=str(r.player_clean_key)
        pid=str(getattr(r,"player_id","") or "").strip()
        q=pd.DataFrame()
        if pid and pid.lower() not in {"nan","<na>"}:
            q=stats.loc[stats["team"].eq(team)&stats["player_id_actual"].eq(pid)]
        if q.empty:
            keys={key}
            alt=aliases.get((team,key))
            if alt: keys.add(alt)
            q=stats.loc[stats["team"].eq(team)&stats["player_clean_key"].isin(keys)]
        if len(q)>1:
            raise RuntimeError(f"ambiguous actual identity team={team} player={r.player} key={key} pid={pid} rows={len(q)}")
        if len(q)==1:
            z=q.iloc[0]
            at=float(z["actual_targets"]); ar=float(z["actual_receptions"]); ay=float(z["actual_rec_yards"])
            source="weekly_stats"
        else:
            keys={key}; alt=aliases.get((team,key))
            if alt: keys.add(alt)
            rq=roster.loc[roster["team"].eq(team)&roster["player_clean_key"].isin(keys)]
            if pid and pid.lower() not in {"nan","<na>"}:
                rq2=roster.loc[roster["team"].eq(team)&roster["player_id_roster"].eq(pid)]
                if not rq2.empty: rq=rq2
            if len(rq)!=1:
                raise RuntimeError(f"missing actual lacks exact roster verification team={team} player={r.player} key={key} pid={pid} roster_rows={len(rq)}")
            at=ar=ay=0.0; source="final_week_exact_roster_verified_zero"
        denom=float(team_targets.get(team,0.0))
        if denom<=0:
            raise RuntimeError(f"nonpositive team target total team={team}")
        rows.append({**{k:getattr(r,k) for k in IDENT},
                     "actual_targets":at,"actual_target_share":at/denom,
                     "actual_receptions":ar,"actual_rec_yards":ay,
                     "team_actual_targets":denom,"actual_source":source})
    actual=pd.DataFrame(rows)
    return base.merge(actual,on=IDENT,how="left",validate="one_to_one")

def _simulate_cell(df:pd.DataFrame)->pd.DataFrame:
    res=simulate(df,iterations=25000,seed=42)
    rows=[]
    for r in df.itertuples(index=False):
        key=(str(r.event_id),str(r.player_clean_key))
        rec=res.values.get((key[0],key[1],"receptions"))
        yd=res.values.get((key[0],key[1],"rec_yards"))
        if rec is None or yd is None:
            raise RuntimeError(f"simulation lookup missing {key}")
        rows.append({
            **{k:getattr(r,k) for k in IDENT},
            "pred_target_share":float(r.entitlement_tgt_share),
            "pred_receptions":float(np.mean(rec)),
            "pred_rec_yards":float(np.mean(yd)),
        })
    return pd.DataFrame(rows)

def _metric_cols(metric:str):
    return {
        "target_share":("pred_target_share","actual_target_share"),
        "receptions":("pred_receptions","actual_receptions"),
        "rec_yards":("pred_rec_yards","actual_rec_yards"),
    }[metric]

def _score(q:pd.DataFrame,metric:str)->dict:
    pc,ac=_metric_cols(metric)
    e=(pd.to_numeric(q[pc],errors="coerce")-pd.to_numeric(q[ac],errors="coerce")).abs()
    return {"rows":int(len(q)),"mae":float(e.mean()) if len(e) else np.nan,
            "p90_ae":float(e.quantile(.90)) if len(e) else np.nan}

def _cohort_mask(df:pd.DataFrame,name:str,changed:set[tuple[str,str,str]]|None=None)->pd.Series:
    pos=df["position"].astype(str).str.upper()
    if name=="WR_TE": return pos.isin(["WR","TE"])
    if name=="WR": return pos.eq("WR")
    if name=="TE": return pos.eq("TE")
    if name=="SWR": return pos.eq("WR") & df["alignment_position"].astype(str).str.upper().eq("SWR")
    if name=="CHANGED":
        changed=changed or set()
        return pd.Series([tuple(x) in changed for x in df[IDENT].itertuples(index=False,name=None)],index=df.index)
    raise KeyError(name)

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--artifact-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    cells={}
    identities=None
    for cell in CELLS:
        p=a.artifact_dir/f"entitlement_{cell}.csv"
        if not p.exists(): raise RuntimeError(f"missing frozen cell {p}")
        x=pd.read_csv(p,low_memory=False)
        if len(x)!=424: raise RuntimeError(f"{cell} row count drift {len(x)}")
        x["team"]=x["team"].map(canon_team)
        if identities is None:
            identities=x[IDENT].copy()
        elif not x[IDENT].equals(identities):
            raise RuntimeError(f"{cell} identity/order drift")
        cells[cell]=x

    base=cells["A0B0"]
    actual_base=_attach_actuals(base)
    actual_cols=IDENT+["actual_targets","actual_target_share","actual_receptions","actual_rec_yards","team_actual_targets","actual_source"]
    actual=actual_base[actual_cols].copy()

    predictions={}
    for cell,x in cells.items():
        p=_simulate_cell(x)
        p["cell"]=cell
        predictions[cell]=p

    detail=[]
    for cell,x in cells.items():
        d=x[IDENT+["player","position","alignment_position"]].copy()
        d=d.merge(predictions[cell],on=IDENT,how="left",validate="one_to_one")
        d=d.merge(actual,on=IDENT,how="left",validate="one_to_one")
        d["cell"]=cell
        detail.append(d)
    detail=pd.concat(detail,ignore_index=True,sort=False)

    base_ent=base.set_index(IDENT)["entitlement_tgt_share"].astype(float)
    changed={}
    for cell,x in cells.items():
        if cell=="A0B0": changed[cell]=set(); continue
        s=x.set_index(IDENT)["entitlement_tgt_share"].astype(float)
        diff=(s-base_ent).abs()
        changed[cell]=set(diff.loc[diff.gt(1e-12)].index.tolist())

    rows=[]
    cohorts=("WR_TE","WR","TE","SWR","CHANGED")
    for cell in CELLS:
        d=detail.loc[detail["cell"].eq(cell)].copy()
        for cohort in cohorts:
            mask=_cohort_mask(d,cohort,changed.get(cell))
            q=d.loc[mask].copy()
            for metric in METRICS:
                rows.append({"cell":cell,"cohort":cohort,"metric":metric,**_score(q,metric)})
    summary=pd.DataFrame(rows)

    # Same-row deltas vs A0B0.
    comparisons=[]
    for cell in ("A1B0","A0B1","A1B1"):
        for cohort in cohorts:
            for metric in METRICS:
                b=summary.loc[(summary.cell=="A0B0")&(summary.cohort==cohort)&(summary.metric==metric)]
                c=summary.loc[(summary.cell==cell)&(summary.cohort==cohort)&(summary.metric==metric)]
                if len(b)!=1 or len(c)!=1: continue
                comparisons.append({
                    "cell":cell,"cohort":cohort,"metric":metric,
                    "baseline_mae":float(b.iloc[0].mae),"candidate_mae":float(c.iloc[0].mae),
                    "mae_delta_candidate_minus_baseline":float(c.iloc[0].mae-b.iloc[0].mae),
                    "baseline_p90":float(b.iloc[0].p90_ae),"candidate_p90":float(c.iloc[0].p90_ae),
                    "p90_delta_candidate_minus_baseline":float(c.iloc[0].p90_ae-b.iloc[0].p90_ae),
                })
    comp=pd.DataFrame(comparisons)

    # Event concentration on each cell's targeted changed rows.
    event_rows=[]
    base_detail=detail.loc[detail.cell.eq("A0B0")].set_index(IDENT)
    for cell in ("A1B0","A0B1","A1B1"):
        cand=detail.loc[detail.cell.eq(cell)].set_index(IDENT)
        keys=changed[cell]
        for event in sorted(set(k[0] for k in keys)):
            ks=[k for k in keys if k[0]==event]
            for metric in METRICS:
                pc,ac=_metric_cols(metric)
                b=base_detail.loc[ks]; cc=cand.loc[ks]
                bmae=float((pd.to_numeric(b[pc])-pd.to_numeric(b[ac])).abs().mean())
                cmae=float((pd.to_numeric(cc[pc])-pd.to_numeric(cc[ac])).abs().mean())
                event_rows.append({"cell":cell,"event_id":event,"metric":metric,"rows":len(ks),
                                   "baseline_mae":bmae,"candidate_mae":cmae,
                                   "mae_delta_candidate_minus_baseline":cmae-bmae})
    event_df=pd.DataFrame(event_rows)

    # Strict directional checks, reported only; no Week-3 production qualification.
    directional={}
    targeted={"A1B0":"TE","A0B1":"SWR","A1B1":"CHANGED"}
    for cell,cohort in targeted.items():
        q=comp.loc[(comp.cell==cell)&(comp.cohort==cohort)]
        pooled=comp.loc[(comp.cell==cell)&(comp.cohort=="WR_TE")]
        directional[cell]={
            "targeted_cohort":cohort,
            "targeted_all_mae_improve":bool((q["mae_delta_candidate_minus_baseline"]<0).all()) if len(q)==3 else False,
            "targeted_all_p90_nonworse":bool((q["p90_delta_candidate_minus_baseline"]<=0).all()) if len(q)==3 else False,
            "pooled_all_mae_nonworse_strict":bool((pooled["mae_delta_candidate_minus_baseline"]<=0).all()) if len(pooled)==3 else False,
            "week3_production_qualification_authorized":False,
        }

    a.out_dir.mkdir(parents=True,exist_ok=True)
    detail.to_csv(a.out_dir/"receiving_semantics_week3_detail.csv",index=False)
    summary.to_csv(a.out_dir/"receiving_semantics_week3_summary.csv",index=False)
    comp.to_csv(a.out_dir/"receiving_semantics_week3_comparison.csv",index=False)
    event_df.to_csv(a.out_dir/"receiving_semantics_week3_event_concentration.csv",index=False)
    result={
        "status":"RECEIVING_RULE_SEMANTICS_WEEK3_PROSPECTIVE_GRADED",
        "season":SEASON,"week":WEEK,
        "cells":list(CELLS),
        "frozen_wr_te_rows":int(base["position"].astype(str).str.upper().isin(["WR","TE"]).sum()),
        "frozen_swr_rows":int((base["position"].astype(str).str.upper().eq("WR") & base["alignment_position"].astype(str).str.upper().eq("SWR")).sum()),
        "changed_rows":{k:len(v) for k,v in changed.items()},
        "directional_checks":directional,
        "disposition":"WEEK3_PROSPECTIVE_EVIDENCE_ONLY_CONTINUE_UNCHANGED_CELLS",
        "sportsbook_inputs_used":0,
        "parameters_fit":0,
        "production_changed":False,
    }
    (a.out_dir/"receiving_semantics_week3_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    print("\nTARGETED COMPARISONS")
    print(comp.loc[
        ((comp.cell=="A1B0")&(comp.cohort=="TE"))|
        ((comp.cell=="A0B1")&(comp.cohort=="SWR"))|
        ((comp.cell=="A1B1")&(comp.cohort=="CHANGED"))|
        (comp.cohort=="WR_TE")
    ].to_string(index=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
