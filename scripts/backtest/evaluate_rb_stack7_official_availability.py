#!/usr/bin/env python3
"""RB STACK7: deterministic official game-day inactive reallocation.

No fitting/search. Official NFL.com inactive identity is pregame. Sportsbook is
joined only after the frozen football candidate is scored.
"""
from __future__ import annotations
import argparse,re
from pathlib import Path
import numpy as np,pandas as pd

TEAM={"OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA","JAX":"JAC","ARZ":"ARI","WSH":"WAS"}
ALIAS={"audricestime":"audricestim"}
RBPOS={"RB","FB","HB"}
MIN_ALL_YARD_GAIN=.10
MIN_ALL_CARRY_GAIN=.03
MIN_SEC_YARD_GAIN=.10
MAX_AFFECTED_ACTIVE_YARD_REG=.10
MAX_RISK_YARD_REG=.10
MAX_RMSE_REG=.10
MAX_ABS_BIAS_DET=.50

def lower(x): x=x.copy();x.columns=[str(c).strip().lower() for c in x.columns];return x
def num(s): return pd.to_numeric(s,errors="coerce")
def tm(v):
    s=str(v).strip().upper() if not pd.isna(v) else "";return TEAM.get(s,s) if s not in {"","NAN","NONE","<NA>"} else ""
def nk(v):
    s=re.sub(r"[^a-z0-9]","",str(v or "").lower());return ALIAS.get(s,s)
def one(root,name):
    h=list(root.rglob(name));
    if len(h)!=1:raise RuntimeError(f"expected exactly one {name}, found {len(h)}")
    return lower(pd.read_csv(h[0],low_memory=False))
def metric(y,p):
    q=pd.DataFrame({"a":num(y),"p":num(p)}).dropna()
    if q.empty:return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"corr":np.nan,"actual_mean":np.nan,"pred_mean":np.nan}
    e=q.p-q.a;corr=q.a.corr(q.p) if len(q)>=3 and q.a.nunique()>1 and q.p.nunique()>1 else np.nan
    return {"n":len(q),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.square(e).mean())),"bias":float(e.mean()),"corr":float(corr) if pd.notna(corr) else np.nan,"actual_mean":float(q.a.mean()),"pred_mean":float(q.p.mean())}

def parse_tokens(v):
    out={}
    if pd.isna(v):return out
    for tok in str(v).split("|"):
        parts=tok.split(":")
        if not parts:continue
        name=nk(parts[0]);pos=parts[1].upper().strip() if len(parts)>1 else ""
        if name:out[name]=pos
    return out

def prep_official(x):
    z=x.copy();z["season"]=num(z.season).astype(int);z["week"]=num(z.week).astype(int);z["team"]=z.team.map(tm);z["token_map"]=z.inactive_tokens.map(parse_tokens)
    if z.duplicated(["season","week","team"]).any():raise RuntimeError("duplicate official inactive team-week rows")
    return z

def prep_parent(x):
    z=x.copy();z["season"]=num(z.season).astype(int);z["week"]=num(z.week).astype(int);z["team"]=z.team.map(tm)
    z["join_key"]=z.get("join_key",z.get("player_clean_key",z.player)).astype(str).map(nk)
    z["parent_att"]=num(z.parent_att);z["parent_yards"]=num(z.parent_yards);z["actual_carries"]=num(z.actual_carries);z["actual_rush_yards"]=num(z.actual_rush_yards)
    rank=num(z.depth_rank);missing=num(z.get("depth_rank_missing",pd.Series(0,index=z.index))).fillna(0)
    z["secondary_defined"]=((rank.ge(2))&missing.eq(0)).astype(int);z["state_m95f_risk"]=num(z.state_m95f_risk).fillna(0).astype(int)
    return z

def attach_official(parent,official):
    o=official.loc[official.season.eq(2025),["season","week","team","token_map","inactive_count"]].copy()
    x=parent.merge(o,on=["season","week","team"],how="left",validate="many_to_one")
    coverage=float(x.token_map.notna().mean())
    if coverage<.999:raise RuntimeError(f"official inactive team-week coverage incomplete: {coverage:.6f}")
    x["official_inactive"]=x.apply(lambda r:int(r.join_key in r.token_map and r.token_map.get(r.join_key,"") in RBPOS),axis=1)
    # Position parsing is not required to find a projected player's identity, but audit any exact-name token with a non-RB position.
    x["official_name_match_anypos"]=x.apply(lambda r:int(r.join_key in r.token_map),axis=1)
    x["official_name_nonrb_position"]=((x.official_name_match_anypos.eq(1))&x.official_inactive.eq(0)).astype(int)
    return x,coverage

def previous_inactive_diagnostics(x,official):
    # Previous TEAM GAME, not previous numeric week, handles byes correctly.
    maps={(int(r.season),int(r.week),r.team):r.token_map for _,r in official.iterrows()}
    team_weeks={}
    for (s,t),g in official.groupby(["season","team"]):team_weeks[(int(s),t)]=sorted(num(g.week).astype(int).tolist())
    prev_flags=[]
    for _,r in x.iterrows():
        ws=[w for w in team_weeks.get((int(r.season),r.team),[]) if w<int(r.week)]
        if not ws:prev_flags.append(0);continue
        pm=maps.get((int(r.season),max(ws),r.team),{})
        prev_flags.append(int(r.join_key in pm and pm.get(r.join_key,"") in RBPOS))
    x["prev_game_official_inactive"]=prev_flags
    x["returned_from_official_inactive"]=((x.prev_game_official_inactive.eq(1))&x.official_inactive.eq(0)).astype(int)
    # Team teammate transition summaries at current projected-player grain.
    x["inactive_comp_count"]=x.groupby(["season","week","team"])["official_inactive"].transform("sum")-x.official_inactive
    x["returned_comp_count"]=x.groupby(["season","week","team"])["returned_from_official_inactive"].transform("sum")-x.returned_from_official_inactive
    return x

def add_candidate(x):
    z=x.copy();gkeys=["season","week","team"]
    pool=z.groupby(gkeys)["parent_att"].transform("sum");z["team_parent_pool"]=pool;z["parent_share"]=np.where(pool.gt(0),z.parent_att/pool,0.0)
    z["inactive_comp_parent_share"]=z.groupby(gkeys).apply(lambda g: pd.Series({i:float((g.loc[g.official_inactive.eq(1)&g.index.ne(i),"parent_share"]).sum()) for i in g.index})).reset_index(level=[0,1,2],drop=True).reindex(z.index) if len(z) else 0.0
    # Simpler returned teammate parent-share diagnostic.
    z["returned_share_component"]=z.parent_share*z.returned_from_official_inactive
    team_ret=z.groupby(gkeys)["returned_share_component"].transform("sum")
    z["returned_comp_parent_share"]=team_ret-z.returned_share_component
    score=z.parent_share.where(z.official_inactive.eq(0),0.0)
    final=pd.Series(np.nan,index=z.index,dtype=float)
    affected=pd.Series(False,index=z.index)
    for _,g in z.groupby(gkeys,sort=False):
        s=score.loc[g.index].clip(lower=0);tot=float(s.sum())
        if float(g.official_inactive.sum())>0:affected.loc[g.index]=True
        if float(g.team_parent_pool.iloc[0])>0 and tot<=0:raise RuntimeError(f"all projected RB/FB inactive for {tuple(g[gkeys].iloc[0])}")
        final.loc[g.index]=s/tot if tot>0 else g.parent_share
    z["affected_team_game"]=affected.astype(int);z["candidate_share"]=final;z["candidate_att"]=z.candidate_share*pool
    implied=np.where(z.parent_att.gt(.10),z.parent_yards/z.parent_att,np.nan);implied=pd.Series(implied,index=z.index).replace([np.inf,-np.inf],np.nan)
    fallback=num(z.get("stack_eff",pd.Series(np.nan,index=z.index))).replace([np.inf,-np.inf],np.nan)
    implied=implied.fillna(fallback).fillna(4.2);z["parent_implied_ypc"]=implied
    z["candidate_yards"]=z.candidate_att*z.parent_implied_ypc
    return z

def source_audit(x,coverage):
    ina=x.loc[x.official_inactive.eq(1)].copy()
    return pd.DataFrame([{"teamweek_coverage":coverage,"parent_rows":len(x),"official_inactive_projected_rows":len(ina),"inactive_actual_carry_nonzero":int(num(ina.actual_carries).fillna(0).ne(0).sum()),"inactive_actual_yard_nonzero":int(num(ina.actual_rush_yards).fillna(0).ne(0).sum()),"name_matches_with_nonrb_position":int(x.official_name_nonrb_position.sum()),"affected_team_games":int(x.loc[x.affected_team_game.eq(1),["season","week","team"]].drop_duplicates().shape[0]),"returned_player_rows":int(x.returned_from_official_inactive.sum()),"sportsbook_upstream":0}])
def pool_audit(x):
    q=x.groupby(["season","week","team"],as_index=False).agg(parent_pool=("team_parent_pool","first"),candidate_pool=("candidate_att","sum"));q["abs_diff"]=(q.parent_pool-q.candidate_pool).abs();return pd.DataFrame([{"team_games":len(q),"max_abs_pool_diff":float(q.abs_diff.max()),"mean_abs_pool_diff":float(q.abs_diff.mean())}])
def masks(x):
    sec=x.secondary_defined.eq(1)&x.state_m95f_risk.eq(0);risk=x.state_m95f_risk.eq(1);active=x.official_inactive.eq(0);affected=x.affected_team_game.eq(1)
    return {"all_rb":pd.Series(True,index=x.index),"secondary_nonrisk":sec,"m95f_risk":risk,"official_inactive":~active,"active_on_affected_team":active&affected,"returned_from_inactive":x.returned_from_official_inactive.eq(1),"has_returned_competitor":x.returned_comp_count.gt(0),"has_inactive_competitor":x.inactive_comp_count.gt(0)}
def score(x):
    rows=[]
    for sl,m in masks(x).items():
        q=x.loc[m]
        for target,a,p,c in [("rush_att",q.actual_carries,q.parent_att,q.candidate_att),("rush_yards",q.actual_rush_yards,q.parent_yards,q.candidate_yards)]:
            bm=metric(a,p);cm=metric(a,c);rows.append({"target":target,"slice":sl,"arm":"P3_PARENT",**bm});rows.append({"target":target,"slice":sl,"arm":"OFFICIAL_INACTIVE_REBALANCE",**cm})
    return pd.DataFrame(rows)
def get(sc,target,sl,arm):
    q=sc.loc[sc.target.eq(target)&sc.slice.eq(sl)&sc.arm.eq(arm)];
    if q.empty:raise RuntimeError(f"missing {target}/{sl}/{arm}")
    return q.iloc[0]
def gate(sc):
    def gain(target,sl):return float(get(sc,target,sl,"P3_PARENT").mae-get(sc,target,sl,"OFFICIAL_INACTIVE_REBALANCE").mae)
    b=get(sc,"rush_yards","all_rb","P3_PARENT");c=get(sc,"rush_yards","all_rb","OFFICIAL_INACTIVE_REBALANCE")
    vals=[("all_yard_mae_gain",gain("rush_yards","all_rb"),">=",MIN_ALL_YARD_GAIN),("all_carry_mae_gain",gain("rush_att","all_rb"),">=",MIN_ALL_CARRY_GAIN),("secondary_nonrisk_yard_gain",gain("rush_yards","secondary_nonrisk"),">=",MIN_SEC_YARD_GAIN),("active_affected_yard_regression",-gain("rush_yards","active_on_affected_team"),"<=",MAX_AFFECTED_ACTIVE_YARD_REG),("m95f_risk_yard_regression",-gain("rush_yards","m95f_risk"),"<=",MAX_RISK_YARD_REG),("all_rmse_regression",float(c.rmse-b.rmse),"<=",MAX_RMSE_REG),("abs_bias_deterioration",float(abs(c.bias)-abs(b.bias)),"<=",MAX_ABS_BIAS_DET)]
    rows=[]
    for name,v,op,t in vals:rows.append({"check":name,"value":v,"operator":op,"threshold":t,"pass":int(v>=t if op==">=" else v<=t)})
    g=pd.DataFrame(rows);return g,int(g["pass"].all())
def market_score(x,cb):
    c=cb.copy();c["week"]=num(c.week).astype(int);c["team"]=c.team.map(tm);c["join_key"]=c.player.astype(str).map(nk)
    q=x.merge(c[["week","team","join_key","consensus_line"]].drop_duplicates(["week","team","join_key"]),on=["week","team","join_key"],how="inner",validate="one_to_one")
    rows=[]
    for arm,col in {"P3_PARENT":"parent_yards","OFFICIAL_INACTIVE_REBALANCE":"candidate_yards","VEGAS_CONSENSUS":"consensus_line"}.items():rows.append({"arm":arm,**metric(q.actual_rush_yards,q[col])})
    diagnostics=pd.DataFrame([{"market_rows":len(q),"official_inactive_market_rows":int(q.official_inactive.sum()),"market_rows_with_inactive_competitor":int(q.inactive_comp_count.gt(0).sum()),"market_rows_returned_from_inactive":int(q.returned_from_official_inactive.sum()),"market_rows_with_returned_competitor":int(q.returned_comp_count.gt(0).sum())}])
    return q,pd.DataFrame(rows),diagnostics
def transition_atlas(x):
    rows=[]
    for label,m in {"returned_player":x.returned_from_official_inactive.eq(1),"active_with_returned_comp":x.official_inactive.eq(0)&x.returned_comp_count.gt(0),"active_with_inactive_comp":x.official_inactive.eq(0)&x.inactive_comp_count.gt(0),"secondary_with_returned_comp":x.secondary_defined.eq(1)&x.state_m95f_risk.eq(0)&x.returned_comp_count.gt(0)}.items():
        q=x.loc[m]
        if q.empty:continue
        rows.append({"stratum":label,"n":len(q),"parent_carry_mae":metric(q.actual_carries,q.parent_att)["mae"],"parent_yard_mae":metric(q.actual_rush_yards,q.parent_yards)["mae"],"mean_carry_bias":float((q.parent_att-q.actual_carries).mean()),"mean_yard_bias":float((q.parent_yards-q.actual_rush_yards).mean()),"mean_returned_comp_share":float(num(q.returned_comp_parent_share).mean()),"mean_inactive_comp_share":float(num(q.inactive_comp_parent_share).mean())})
    return pd.DataFrame(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stack6-root",type=Path,required=True);ap.add_argument("--official-csv",type=Path,required=True);ap.add_argument("--market-root",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    x=prep_parent(one(a.stack6_root,"stack6_2025_casebook.csv"));official=prep_official(lower(pd.read_csv(a.official_csv,low_memory=False)));x,cov=attach_official(x,official);x=previous_inactive_diagnostics(x,official);x=add_candidate(x)
    sa=source_audit(x,cov);pa=pool_audit(x)
    if int(sa.inactive_actual_carry_nonzero.iloc[0]) or int(sa.inactive_actual_yard_nonzero.iloc[0]):raise RuntimeError("official inactive identity matched nonzero actual usage; source reconciliation failed")
    if float(pa.max_abs_pool_diff.iloc[0])>1e-6:raise RuntimeError("team opportunity pool conservation failed")
    sc=score(x);g,passed=gate(sc);mq,mm,md=market_score(x,one(a.market_root,"rb_market_casebook.csv"));ta=transition_atlas(x)
    disp=pd.DataFrame([{"disposition":"STACK7_OFFICIAL_INACTIVE_REBALANCE_RETAIN_DEVELOPMENT_FREEZE_2026" if passed else "STACK7_OFFICIAL_INACTIVE_PRIMARY_FAILED_NO_RETUNE","gate_pass":passed,"model_fit":0,"threshold_search":0,"weight_search":0,"sportsbook_upstream":0,"market_used_for_selection":0,"validation_status":"2025_EXPOSED_DEVELOPMENT","next":"FREEZE_FOR_2026" if passed else "USE_RETURN_TRANSITION_ATLAS_TO_JUSTIFY_NEXT_NONREDUNDANT_MECHANISM"}])
    for name,df in [("stack7_source_audit.csv",sa),("stack7_pool_audit.csv",pa),("stack7_football_metrics.csv",sc),("stack7_primary_gate.csv",g),("stack7_transition_atlas.csv",ta),("stack7_market_metrics_DOWNSTREAM.csv",mm),("stack7_market_diagnostics_DOWNSTREAM.csv",md),("stack7_disposition.csv",disp),("stack7_2025_casebook.csv",x)]:df.to_csv(a.out_dir/name,index=False)
    print("=== SOURCE ===");print(sa.to_string(index=False));print("=== FOOTBALL ===");print(sc.to_string(index=False));print("=== GATE ===");print(g.to_string(index=False));print("=== TRANSITIONS ===");print(ta.to_string(index=False));print("=== MARKET DOWNSTREAM ===");print(mm.to_string(index=False));print(md.to_string(index=False));print("=== DISPOSITION ===");print(disp.to_string(index=False))
if __name__=="__main__":main()
