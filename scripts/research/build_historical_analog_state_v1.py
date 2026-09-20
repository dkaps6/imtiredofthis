#!/usr/bin/env python3
"""Outcome-free strict-prior historical analog-state materializer V1."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

KEY=["season","week","team","player_identity_key"]
POSITIONS={"QB","RB","WR","TE"}
FEATURES=["prior_tgt_share_game","prior3_tgt_share_game_mean","prior5_tgt_share_game_mean","prior_rush_share_game","prior3_rush_share_game_mean","prior5_rush_share_game_mean","prior_tgt_share_game_top1","prior_tgt_share_game_top2","prior_rush_share_game_top1","prior_rush_share_game_top2","prior_tgt_share_game_returning_overlap","prior_rush_share_game_returning_overlap"]
K=10
MIN_PRIOR=25


def materialize(context:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    d=context.copy(); d.columns=[str(c).strip().lower() for c in d.columns]
    req=set(KEY+["position"]+FEATURES)
    miss=req-set(d.columns)
    if miss: raise RuntimeError(f"context missing columns: {sorted(miss)}")
    if d.duplicated(KEY).any(): raise RuntimeError("duplicate canonical player-game keys")
    d=d[d.position.isin(POSITIONS)].copy()
    d["season"]=pd.to_numeric(d.season,errors="coerce"); d["week"]=pd.to_numeric(d.week,errors="coerce")
    d=d.sort_values(["season","week","team","player_identity_key"]).reset_index(drop=True)
    out=[]; neighbors=[]
    for pos,p in d.groupby("position",sort=True):
        p=p.copy().reset_index().rename(columns={"index":"source_index"})
        # Frozen geometry: expanding strict-prior median imputation + z scaling, Euclidean; position-specific pool.
        for _,r in p.iterrows():
            prior=p[(p.season<r.season)|((p.season==r.season)&(p.week<r.week))]
            state="VALID_ANALOG" if len(prior)>=MIN_PRIOR else "NO_ANALOG_SUPPORT"
            rec={c:r[c] for c in KEY}; rec.update({"position":pos,"analog_state":state,"prior_pool_rows":len(prior),"neighbor_k":K})
            if state!="VALID_ANALOG":
                rec.update({"nearest_distance":np.nan,"mean_k_distance":np.nan,"effective_analog_count":0.0,"same_player_share":np.nan,"same_team_share":np.nan,"same_season_share":np.nan})
                out.append(rec); continue
            X=prior[FEATURES].apply(pd.to_numeric,errors="coerce"); q=pd.DataFrame([{f:r[f] for f in FEATURES}]).apply(pd.to_numeric,errors="coerce")
            imp=SimpleImputer(strategy="median"); Xi=imp.fit_transform(X); qi=imp.transform(q)
            sc=StandardScaler(); Xs=sc.fit_transform(Xi); qs=sc.transform(qi)[0]
            dist=np.sqrt(((Xs-qs)**2).sum(axis=1)); order=np.argsort(dist,kind="stable")[:min(K,len(prior))]
            nn=prior.iloc[order].copy(); dd=dist[order]; w=1/(1+dd); eff=float((w.sum()**2)/(w@w)) if (w@w)>0 else 0.0
            same_player=float((nn.player_identity_key.astype(str)==str(r.player_identity_key)).mean())
            same_team=float((nn.team.astype(str)==str(r.team)).mean()); same_season=float((nn.season==r.season).mean())
            rec.update({"nearest_distance":float(dd[0]),"mean_k_distance":float(dd.mean()),"effective_analog_count":eff,"same_player_share":same_player,"same_team_share":same_team,"same_season_share":same_season})
            out.append(rec)
            for rank,(ix,nr) in enumerate(nn.iterrows(),1):
                neighbors.append({**{f"target_{c}":r[c] for c in KEY},"position":pos,"neighbor_rank":rank,"analog_season":nr.season,"analog_week":nr.week,"analog_team":nr.team,"analog_player_identity_key":nr.player_identity_key,"distance":float(dd[rank-1]),"strict_prior":bool((nr.season<r.season) or (nr.season==r.season and nr.week<r.week))})
    return pd.DataFrame(out),pd.DataFrame(neighbors)


def audit(states:pd.DataFrame,neighbors:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for pos,p in states.groupby("position"):
        valid=p[p.analog_state=="VALID_ANALOG"]
        n=neighbors[neighbors.position==pos]
        season_counts=valid.groupby("season").size()
        rows.append({"position":pos,"target_rows":len(p),"valid_rows":len(valid),"explicit_state_coverage":float(p.analog_state.notna().mean()),"valid_rate":float(len(valid)/len(p)) if len(p) else np.nan,"seasons_with_targets":int(p.season.nunique()),"max_valid_season_share":float(season_counts.max()/season_counts.sum()) if season_counts.sum() else np.nan,"median_nearest_distance":float(valid.nearest_distance.median()) if len(valid) else np.nan,"median_effective_analog_count":float(valid.effective_analog_count.median()) if len(valid) else np.nan,"same_player_neighbor_share":float(valid.same_player_share.mean()) if len(valid) else np.nan,"same_team_neighbor_share":float(valid.same_team_share.mean()) if len(valid) else np.nan,"chronology_violations":int((~n.strict_prior).sum()) if len(n) else 0,"support_gate":bool(len(p)>=2000 and p.season.nunique()>=4),"coverage_gate":bool(p.analog_state.notna().mean()>=.80),"diversity_gate":bool(len(valid)>0 and valid.same_player_share.mean()<=.50),"chronology_gate":bool(len(n)==0 or n.strict_prior.all()),"outcomes_read":False,"sportsbook_read":False})
    return pd.DataFrame(rows)

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--context",type=Path,required=True); ap.add_argument("--state-out",type=Path,required=True); ap.add_argument("--neighbors-out",type=Path,required=True); ap.add_argument("--audit-out",type=Path,required=True); a=ap.parse_args()
    s,n=materialize(pd.read_csv(a.context)); q=audit(s,n)
    for path,df in [(a.state_out,s),(a.neighbors_out,n),(a.audit_out,q)]: path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False)
    print(q.to_string(index=False)); return 0
if __name__=="__main__": raise SystemExit(main())
