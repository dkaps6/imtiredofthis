#!/usr/bin/env python3
"""Evaluate named-player high-value opportunity persistence V1."""
from __future__ import annotations

import argparse, json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import audit_qb_official_inactive_availability as m78

VERSION="PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_V1"
SEASONS=(2022,2023,2024,2025)
BOOT=5000

CHANNELS={
 "RB_RUSH":{"family":"RB","channel":"rush","hv":"inside10","min_rows":500,"min_players":80,
            "qualified":"RB_RUSH_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_QUALIFIED",
            "closed":"NO_ACTIONABLE_RB_RUSH_HIGH_VALUE_OPPORTUNITY_PERSISTENCE","seed":2026101611},
 "RB_RECEIVING":{"family":"RB","channel":"target","hv":"inside20","min_rows":250,"min_players":70,
            "qualified":"RB_RECEIVING_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_QUALIFIED",
            "closed":"NO_ACTIONABLE_RB_RECEIVING_HIGH_VALUE_OPPORTUNITY_PERSISTENCE","seed":2026101621},
 "WR_RECEIVING":{"family":"WR","channel":"target","hv":"inside20","min_rows":800,"min_players":150,
            "qualified":"WR_RECEIVING_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_QUALIFIED",
            "closed":"NO_ACTIONABLE_WR_RECEIVING_HIGH_VALUE_OPPORTUNITY_PERSISTENCE","seed":2026101631},
 "TE_RECEIVING":{"family":"TE","channel":"target","hv":"inside20","min_rows":300,"min_players":70,
            "qualified":"TE_RECEIVING_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_QUALIFIED",
            "closed":"NO_ACTIONABLE_TE_RECEIVING_HIGH_VALUE_OPPORTUNITY_PERSISTENCE","seed":2026101641},
}
POS={"RB":{"RB","FB","HB"},"WR":{"WR","LWR","RWR","SWR"},"TE":{"TE"}}


def num(s): return pd.to_numeric(s,errors="coerce")


def fam(pos):
 p=str(pos or "").upper().strip().replace("-","").replace(" ","")
 for k,v in POS.items():
  if p in v:return k
 return ""


def first(df,names):
 for c in names:
  if c in df.columns:return c
 return None


def roster_frame(meta):
 parts=[]
 for s in SEASONS:
  q=m78.load_weekly_rosters(s,meta)
  if q.empty: raise RuntimeError(f"weekly roster empty {s}")
  q=q.copy();q.columns=[str(c).lower() for c in q.columns]
  wc=first(q,["week"]);tc=first(q,["team","club_code"]);ic=first(q,["gsis_id","player_id"])
  pc=first(q,["position","depth_chart_position","ngs_position","pos"])
  sc=first(q,["status","status_description_abbr"])
  if not all([wc,tc,ic,pc,sc]): raise RuntimeError(f"ACT roster schema missing {s}")
  z=pd.DataFrame({
   "season":s,"week":num(q[wc]),"team":q[tc].map(canon_team),
   "player_id":q[ic].astype("string").fillna("").str.strip(),
   "position":q[pc].astype("string").fillna("").str.upper().str.strip(),
   "status":q[sc].astype("string").fillna("").str.upper().str.strip(),
  })
  z["family"]=z.position.map(fam)
  z=z.loc[z.week.notna() & z.player_id.ne("") & z.team.astype(str).ne("") & z.family.isin(POS)].copy()
  z["week"]=z.week.astype(int)
  if not z.status.eq("ACT").any(): raise RuntimeError(f"no ACT roster status rows {s}")
  z=z.loc[z.status.eq("ACT")].copy()
  parts.append(z.drop_duplicates(["season","week","team","player_id"]))
 return pd.concat(parts,ignore_index=True)


def prepare_channel(source,roster,cfg):
 q=source.loc[source.channel.eq(cfg["channel"])].copy()
 q["season"]=num(q.season).astype(int);q["week"]=num(q.week).astype(int)
 for c in ["opportunities","inside20","inside10","inside5"]:
  q[c]=num(q[c]).fillna(0.0)
 # team denominator includes every player in the opportunity channel.
 den=q.groupby(["season","week","team"],as_index=False).agg(
  team_opp=("opportunities","sum"),
  team_inside20=("inside20","sum"),
  team_inside10=("inside10","sum"),
  team_inside5=("inside5","sum"),
 )
 player=q.groupby(["season","week","team","player_id"],as_index=False).agg(
  opportunities=("opportunities","sum"),
  inside20=("inside20","sum"),
  inside10=("inside10","sum"),
  inside5=("inside5","sum"),
 )
 u=roster.loc[roster.family.eq(cfg["family"]),["season","week","team","player_id","family"]].copy()
 u=u.merge(den,on=["season","week","team"],how="inner",validate="many_to_one")
 u=u.merge(player,on=["season","week","team","player_id"],how="left",validate="one_to_one")
 for c in ["opportunities","inside20","inside10","inside5"]:
  u[c]=num(u[c]).fillna(0.0)
 hv=cfg["hv"];team_hv="team_"+hv
 u["target_hv"]=u[hv]
 u["target_team_hv"]=u[team_hv]
 u["actual_hv_share"]=np.where(u["target_team_hv"]>0,u["target_hv"]/u["target_team_hv"],np.nan)
 u["actual_overall_share"]=np.where(u.team_opp>0,u.opportunities/u.team_opp,np.nan)
 u["ordinal"]=u.season*100+u.week
 return u.sort_values(["player_id","team","season","week"]).reset_index(drop=True)


def attach_predictors(u):
 out=[]
 for (pid,team),g in u.groupby(["player_id","team"],sort=False):
  g=g.sort_values(["season","week"]).copy()
  rows=list(g.to_dict("records"))
  for i,r in enumerate(rows):
   hist=rows[max(0,i-4):i]
   rec=dict(r)
   rec["prior_active_games"]=len(hist)
   if len(hist)>=3:
    po=sum(float(x["opportunities"]) for x in hist)
    pto=sum(float(x["team_opp"]) for x in hist)
    ph=sum(float(x["target_hv"]) for x in hist)
    pth=sum(float(x["target_team_hv"]) for x in hist)
    rec["prior_team_hv"]=pth
    rec["baseline_pred"]=po/pto if pto>0 else np.nan
    rec["candidate_pred"]=ph/pth if pth>=4 else np.nan
   else:
    rec["prior_team_hv"]=np.nan;rec["baseline_pred"]=np.nan;rec["candidate_pred"]=np.nan
   out.append(rec)
 z=pd.DataFrame(out)
 z=z.loc[
  z.prior_active_games.ge(3)
  & z.prior_team_hv.ge(4)
  & z.actual_hv_share.notna()
  & z.baseline_pred.notna()
  & z.candidate_pred.notna()
 ].copy()
 z["baseline_abs_error"]=(z.actual_hv_share-z.baseline_pred).abs()
 z["candidate_abs_error"]=(z.actual_hv_share-z.candidate_pred).abs()
 z["gain"]=z.baseline_abs_error-z.candidate_abs_error
 z["cluster_game"]=z.season.astype(str)+"-"+z.week.astype(str)+"-"+z.team.astype(str)
 return z


def boot_gain(df,cluster,seed):
 agg=df.groupby(cluster).gain.agg(["sum","count"]).reset_index(drop=True)
 sums=agg["sum"].to_numpy(float);counts=agg["count"].to_numpy(float)
 rng=np.random.default_rng(seed);vals=np.empty(BOOT,float);n=len(agg)
 for i in range(BOOT):
  ix=rng.integers(0,n,n)
  vals[i]=sums[ix].sum()/counts[ix].sum()
 return {"clusters":n,"reps":BOOT,"observed":float(df.gain.mean()),
         "ci_low":float(np.quantile(vals,.025)),"ci_high":float(np.quantile(vals,.975))}


def metrics(df):
 return {
  "rows":int(len(df)),"players":int(df.player_id.nunique()),
  "baseline_mae":float(df.baseline_abs_error.mean()),
  "candidate_mae":float(df.candidate_abs_error.mean()),
  "mean_gain":float(df.gain.mean()),
 }


def player_heterogeneity(df):
 g=df.groupby("player_id").agg(n=("gain","size"),baseline=("baseline_abs_error","mean"),candidate=("candidate_abs_error","mean"))
 g=g.loc[g.n.ge(4)].copy()
 g["gain"]=g.baseline-g.candidate
 return {
  "players_ge4":int(len(g)),
  "median_player_gain":float(g.gain.median()) if len(g) else np.nan,
  "positive_player_share":float(g.gain.gt(0).mean()) if len(g) else np.nan,
  "q25":float(g.gain.quantile(.25)) if len(g) else np.nan,
  "q75":float(g.gain.quantile(.75)) if len(g) else np.nan,
 }


def evaluate(source,roster,name,cfg):
 u=prepare_channel(source,roster,cfg);z=attach_predictors(u)
 by={}
 support=True
 for s in SEASONS:
  q=z.loc[z.season.eq(s)]
  m=metrics(q);by[str(s)]=m
  support=support and m["rows"]>=cfg["min_rows"] and m["players"]>=cfg["min_players"]
 pooled=metrics(z);conf=z.loc[z.season.eq(2025)].copy()
 bp=boot_gain(z,"player_id",cfg["seed"])
 bg=boot_gain(z,"cluster_game",cfg["seed"]+1)
 cp=boot_gain(conf,"player_id",cfg["seed"]+2)
 cg=boot_gain(conf,"cluster_game",cfg["seed"]+3)
 het=player_heterogeneity(conf)
 gates={
  "support":bool(support),
  "2022_candidate_better":by["2022"]["candidate_mae"]<by["2022"]["baseline_mae"],
  "2023_candidate_better":by["2023"]["candidate_mae"]<by["2023"]["baseline_mae"],
  "2024_candidate_better":by["2024"]["candidate_mae"]<by["2024"]["baseline_mae"],
  "2025_candidate_better":by["2025"]["candidate_mae"]<by["2025"]["baseline_mae"],
  "pooled_player_ci_low_positive":bp["ci_low"]>0,
  "pooled_game_ci_low_positive":bg["ci_low"]>0,
  "2025_player_ci_low_positive":cp["ci_low"]>0,
  "2025_game_ci_low_positive":cg["ci_low"]>0,
  "2025_median_player_gain_positive":bool(np.isfinite(het["median_player_gain"]) and het["median_player_gain"]>0),
  "sportsbook_zero":True,"oddsapi_zero":True,"outcomes_2026_zero":True,
  "models_fit_zero":True,"production_unchanged":True,
 }
 disposition=cfg["qualified"] if all(gates.values()) else cfg["closed"]
 return z,{
  "channel":name,"disposition":disposition,"by_season":by,"pooled":pooled,
  "pooled_player_bootstrap":bp,"pooled_game_bootstrap":bg,
  "confirmation_2025_player_bootstrap":cp,"confirmation_2025_game_bootstrap":cg,
  "confirmation_2025_player_heterogeneity":het,"gates":gates,
  "failed_gates":[k for k,v in gates.items() if not v],
 }


def main():
 ap=argparse.ArgumentParser();ap.add_argument("--playerweeks",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True)
 a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 source=pd.read_csv(a.playerweeks,low_memory=False)
 need={"season","week","team","player_id","opportunities","inside20","inside10","inside5","channel"}
 if need-set(source.columns):raise RuntimeError(f"parent missing {sorted(need-set(source.columns))}")
 if num(source.season).max()>2025:raise RuntimeError("2026 outcome leak in source parent")
 meta=[];roster=roster_frame(meta)
 results={};frames=[];passes=[]
 for name,cfg in CHANNELS.items():
  z,r=evaluate(source,roster,name,cfg);results[name]=r;z["evaluation_channel"]=name;frames.append(z)
  passes.append(r["disposition"]==cfg["qualified"])
 if all(passes):overall="PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_ALL_QUALIFIED"
 elif any(passes):overall="PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_PARTIAL"
 else:overall="PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED"
 result={"version":VERSION,"disposition":overall,"channels":results,
  "sportsbook_inputs_used":0,"oddsapi_calls":0,"outcomes_2026_read":0,
  "predictive_models_fit":0,"production_changed":False,"bootstrap_reps":BOOT}
 pd.concat(frames,ignore_index=True).to_csv(a.out_dir/"high_value_opportunity_persistence_rows.csv",index=False)
 pd.DataFrame(meta).to_csv(a.out_dir/"high_value_opportunity_persistence_source_manifest.csv",index=False)
 (a.out_dir/"high_value_opportunity_persistence_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
