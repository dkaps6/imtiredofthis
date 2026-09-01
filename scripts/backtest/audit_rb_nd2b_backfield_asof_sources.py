#!/usr/bin/env python3
"""RB-ND2B: audit timestamp-safe depth hierarchy and lagged snap state. No fit."""
from __future__ import annotations
import argparse,re
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np,pandas as pd
TEAM={"OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA","JAX":"JAC","ARZ":"ARI","WSH":"WAS"}
def pdx(v):
    if isinstance(v,pd.DataFrame): return v.copy()
    return v.to_pandas() if hasattr(v,"to_pandas") else pd.DataFrame(v.to_dicts()) if hasattr(v,"to_dicts") else pd.DataFrame(v)
def lower(x): x=x.copy();x.columns=[str(c).strip().lower() for c in x.columns];return x
def tm(v):
    s=str(v).strip().upper() if not pd.isna(v) else "";return TEAM.get(s,s) if s not in {"","NAN","NONE","<NA>"} else ""
def nk(v): return re.sub(r"[^a-z0-9]","",str(v or "").lower())
def one(root,name):
    h=list(root.rglob(name));
    if len(h)!=1: raise RuntimeError(f"expected one {name}, found {len(h)}")
    return lower(pd.read_csv(h[0],low_memory=False))
def games(s,season):
    x=s.copy();x["season"]=pd.to_numeric(x.season,errors="coerce");x["week"]=pd.to_numeric(x.week,errors="coerce");x=x.loc[x.season.eq(season)&x.week.between(1,18)]
    if "game_type" in x:x=x.loc[x.game_type.astype(str).str.upper().eq("REG")]
    gt=x.gametime.astype(str) if "gametime" in x else pd.Series("13:00",index=x.index)
    z=pd.to_datetime(x.gameday.astype(str)+" "+gt,errors="coerce"); east=ZoneInfo("America/New_York")
    x["kickoff_utc"]=[pd.Timestamp(d).tz_localize(east).tz_convert("UTC") if not pd.isna(d) else pd.NaT for d in z]
    out=[]
    for _,r in x.iterrows():
        for t,o in [(tm(r.home_team),tm(r.away_team)),(tm(r.away_team),tm(r.home_team))]: out.append({"season":season,"week":int(r.week),"team":t,"opponent":o,"kickoff_utc":r.kickoff_utc})
    return pd.DataFrame(out)
def rbmask(d):
    a=d.get("pos_abb",pd.Series("",index=d.index)).astype(str).str.upper(); n=d.get("pos_name",pd.Series("",index=d.index)).astype(str).str.lower(); g=d.get("pos_grp",pd.Series("",index=d.index)).astype(str).str.lower()
    return a.isin(["RB","HB","FB"])|n.str.contains("running back|halfback|fullback",regex=True)|g.str.contains("running back|backfield",regex=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--m94c-root",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    import nflreadpy as nfl
    rb=one(a.m94c_root,"m94c_2025_rb_trace.csv");rb=rb.loc[pd.to_numeric(rb.season,errors="coerce").eq(2025)&rb.position.astype(str).str.upper().isin(["RB","FB"])].copy();rb["week"]=pd.to_numeric(rb.week,errors="coerce").astype(int);rb["team"]=rb.team.map(tm);rb["name_key"]=rb.player.map(nk)
    sched=lower(pdx(nfl.load_schedules(seasons=[2024,2025])));g25=games(sched,2025)
    d24=lower(pdx(nfl.load_depth_charts(seasons=[2024])));d25=lower(pdx(nfl.load_depth_charts(seasons=[2025])));sn=lower(pdx(nfl.load_snap_counts(seasons=[2024,2025])))
    dt=pd.to_datetime(d25.dt,errors="coerce",utc=True);d25["dt_utc"]=dt;d25["team"]=d25.team.map(tm);dr=d25.loc[rbmask(d25)&d25.dt_utc.notna()].copy();dr["name_key"]=dr.player_name.map(nk)
    asof=[];dp=[]
    for _,g in g25.iterrows():
        q=dr.loc[dr.team.eq(g.team)&dr.dt_utc.lt(g.kickoff_utc)]
        if q.empty: asof.append({**g.to_dict(),"snapshot_utc":pd.NaT,"snapshot_age_hours":np.nan,"rb_depth_players":0});continue
        t=q.dt_utc.max();z=q.loc[q.dt_utc.eq(t)].copy();asof.append({**g.to_dict(),"snapshot_utc":t,"snapshot_age_hours":float((g.kickoff_utc-t).total_seconds()/3600),"rb_depth_players":len(z)})
        for _,r in z.iterrows(): dp.append({"week":int(g.week),"team":g.team,"player_name":r.player_name,"name_key":r.name_key,"gsis_id":r.get("gsis_id",""),"pos_abb":r.get("pos_abb",""),"pos_name":r.get("pos_name",""),"pos_slot":r.get("pos_slot",np.nan),"pos_rank":r.get("pos_rank",np.nan),"snapshot_utc":t})
    asof=pd.DataFrame(asof);dp=pd.DataFrame(dp)
    dm=rb[["week","team","name_key","player"]].copy()
    if not dp.empty: dm=dm.merge(dp[["week","team","name_key","pos_rank","pos_slot","snapshot_utc"]].drop_duplicates(["week","team","name_key"]),on=["week","team","name_key"],how="left",validate="many_to_one")
    else: dm["pos_rank"]=np.nan
    d24["week"]=pd.to_numeric(d24.week,errors="coerce");d24["club_code"]=d24.club_code.map(tm);m24=d24.position.astype(str).str.upper().isin(["RB","HB","FB"])|d24.depth_position.astype(str).str.upper().isin(["RB","HB","FB"]);old=d24.loc[m24&d24.week.between(1,18)]
    sn["season"]=pd.to_numeric(sn.season,errors="coerce");sn["week"]=pd.to_numeric(sn.week,errors="coerce");sn["team"]=sn.team.map(tm);sn["name_key"]=sn.player.map(nk);s25=sn.loc[sn.season.eq(2025)&sn.week.between(1,18)].copy();s25["offense_pct"]=pd.to_numeric(s25.offense_pct,errors="coerce")
    lag=s25[["week","team","name_key","offense_pct","offense_snaps"]].copy();lag["week"]+=1;lag=lag.rename(columns={"offense_pct":"prior_week_offense_pct","offense_snaps":"prior_week_offense_snaps"}).drop_duplicates(["week","team","name_key"],keep="last");sm=rb[["week","team","name_key","player"]].merge(lag,on=["week","team","name_key"],how="left",validate="many_to_one")
    src=pd.DataFrame([{"depth2025_rows":len(d25),"depth2025_min_dt":dt.min(),"depth2025_max_dt":dt.max(),"depth2025_unique_dates":int(dt.nunique()),"depth2025_rb_rows":len(dr),"depth2024_rows":len(d24),"depth2024_rb_rows":len(old),"snap_rows":len(sn),"snap_columns":";".join(sn.columns)}])
    cov=pd.DataFrame([{"team_games":len(asof),"team_game_depth_coverage":float(asof.snapshot_utc.notna().mean()),"median_snapshot_age_hours":float(asof.snapshot_age_hours.median()),"p90_snapshot_age_hours":float(asof.snapshot_age_hours.quantile(.9)),"m94c_rows":len(dm),"m94c_depth_rank_coverage":float(dm.pos_rank.notna().mean()),"m94c_week1_depth_rank_coverage":float(dm.loc[dm.week.eq(1),"pos_rank"].notna().mean())}])
    snap=pd.DataFrame([{"m94c_rows":len(sm),"prior_week_snap_coverage_all":float(sm.prior_week_offense_pct.notna().mean()),"prior_week_snap_coverage_weeks2_18":float(sm.loc[sm.week.ge(2),"prior_week_offense_pct"].notna().mean())}])
    olda=pd.DataFrame([{"rows":len(old),"team_weeks":old[["week","club_code"]].drop_duplicates().shape[0],"weeks":old.week.nunique(),"teams":old.club_code.nunique(),"depth_team_values":";".join(sorted(set(old.depth_team.astype(str)))[:20])}])
    for name,x in [("nd2b_source_summary.csv",src),("nd2b_2025_depth_game_asof.csv",asof),("nd2b_2025_depth_player_asof.csv",dp),("nd2b_m94c_depth_match.csv",dm),("nd2b_depth_coverage.csv",cov),("nd2b_2024_depth_audit.csv",olda),("nd2b_snap_lag_coverage.csv",snap),("nd2b_m94c_snap_lag_match.csv",sm)]: x.to_csv(a.out_dir/name,index=False)
    print("=== source ===");print(src.to_string(index=False));print("=== depth coverage ===");print(cov.to_string(index=False));print("=== old depth ===");print(olda.to_string(index=False));print("=== snap ===");print(snap.to_string(index=False));print("=== pos ranks ===");print(pd.to_numeric(dp.get("pos_rank"),errors="coerce").value_counts(dropna=False).sort_index().to_string())
if __name__=="__main__": main()
