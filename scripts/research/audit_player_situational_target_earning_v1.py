#!/usr/bin/env python3
"""Player Situational Target Earning V1 source/nonredundancy audit."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts._opponent_map import canon_team

VERSION="PLAYER_SITUATIONAL_TARGET_EARNING_V1"
SEASONS=(2022,2023,2024,2025,2026)
TARGET_SEASON=2026
TARGET_WEEK=5
MAX_WEEK=4
CONTEXTS=("ALL","EARLY_DOWN","THIRD_DOWN","RED_ZONE","TWO_MINUTE")

def to_pd(x):
    if isinstance(x,pd.DataFrame): return x.copy()
    if hasattr(x,"to_pandas"): return x.to_pandas()
    return pd.DataFrame(x)

def lower(x):
    y=to_pd(x); y.columns=[str(c).strip().lower() for c in y.columns]; return y

def num(x): return pd.to_numeric(x,errors="coerce")

def clean(v):
    if v is None or pd.isna(v): return ""
    s=str(v).strip()
    return "" if s.lower() in {"","nan","none","<na>"} else s

def team(v):
    try:
        t=canon_team(v)
        return "WAS" if t=="WSH" else t
    except Exception:
        return clean(v).upper()

def first(df,names,default=""):
    for n in names:
        if n in df.columns: return df[n]
    return pd.Series(default,index=df.index)

def regular_only(df):
    x=df.copy()
    c="season_type" if "season_type" in x.columns else "game_type" if "game_type" in x.columns else None
    if c:
        s=x[c].astype(str).str.upper()
        keep=s.isin(["REG","REGULAR","RS",""])
        if keep.any(): x=x.loc[keep].copy()
    return x

def load_pbp(season):
    import nflreadpy as nfl
    x=regular_only(lower(nfl.load_pbp(seasons=[int(season)])))
    for c in ["season","week","posteam","receiver_player_id","pass_attempt","sack","two_point_attempt",
              "down","yardline_100","half_seconds_remaining"]:
        if c not in x.columns: x[c]=np.nan
    x["season"]=num(x["season"]).fillna(season).astype(int)
    x["week"]=num(x["week"])
    x["team"]=x["posteam"].map(team)
    x["receiver_id"]=x["receiver_player_id"].map(clean)
    raw=num(x["pass_attempt"]).fillna(0).eq(1)
    sack=num(x["sack"]).fillna(0).eq(1)
    two=num(x["two_point_attempt"]).fillna(0).eq(1)
    x["official_pass_attempt"]=raw & ~sack & ~two
    x["target_event"]=x["official_pass_attempt"] & x["receiver_id"].ne("") & x["team"].ne("")
    d=num(x["down"])
    y=num(x["yardline_100"])
    hs=num(x["half_seconds_remaining"])
    x["ctx_all"]=x["target_event"]
    x["ctx_early_down"]=x["target_event"] & d.isin([1,2])
    x["ctx_third_down"]=x["target_event"] & d.eq(3)
    x["ctx_red_zone"]=x["target_event"] & y.le(20)
    x["ctx_two_minute"]=x["target_event"] & hs.le(120)
    if season==TARGET_SEASON:
        bad=sorted(x.loc[x["week"].ge(TARGET_WEEK),"week"].dropna().astype(int).unique().tolist())
        if bad: raise RuntimeError(f"2026 target/future PBP already present: weeks={bad}")
        x=x.loc[x["week"].le(MAX_WEEK)].copy()
    return x

def load_schedule():
    import nflreadpy as nfl
    try: x=lower(nfl.load_schedules(seasons=[TARGET_SEASON]))
    except TypeError: x=lower(nfl.load_schedules(TARGET_SEASON))
    x=regular_only(x)
    x=x.loc[num(x["season"]).eq(TARGET_SEASON)&num(x["week"]).eq(TARGET_WEEK)].copy()
    hc="home_team" if "home_team" in x.columns else "home"
    ac="away_team" if "away_team" in x.columns else "away"
    teams=[]
    for r in x.itertuples(index=False):
        h=team(getattr(r,hc)); a=team(getattr(r,ac))
        if h: teams.append(h)
        if a: teams.append(a)
    return sorted(set(teams))

def load_roster():
    import nflreadpy as nfl
    x=lower(nfl.load_rosters_weekly(TARGET_SEASON))
    out=pd.DataFrame(index=x.index)
    out["season"]=num(first(x,["season"],TARGET_SEASON)).fillna(TARGET_SEASON)
    out["week"]=num(first(x,["week"]))
    out["team"]=first(x,["team","team_abbr","club_code"]).map(team)
    out["player"]=first(x,["full_name","football_name","player_name","player","name"]).astype("string").fillna("").str.strip()
    out["receiver_id"]=first(x,["gsis_id","player_id"]).map(clean)
    out["position"]=first(x,["position","pos","depth_chart_position"]).astype("string").fillna("").str.upper().str.strip()
    out=out.loc[out["season"].eq(TARGET_SEASON)&out["position"].isin(["WR","TE"])&out["team"].ne("")].copy()
    exact=out.loc[out["week"].eq(TARGET_WEEK)].copy()
    if len(exact):
        use=exact; source_week=TARGET_WEEK
    else:
        prior=out.loc[out["week"].lt(TARGET_WEEK)].copy()
        if prior.empty: raise RuntimeError("no Week-5 or prior weekly roster")
        source_week=int(prior["week"].max())
        use=prior.loc[prior["week"].eq(source_week)].copy()
    use["roster_source_week"]=source_week
    use=use.sort_values(["team","receiver_id","player"]).drop_duplicates(["team","receiver_id","player"],keep="last")
    return use

def inventory(pbp):
    rows=[]
    for season,g in pbp.groupby("season",sort=True):
        t=g.loc[g["target_event"]].copy()
        rows.append({
          "season":int(season),
          "pbp_rows":int(len(g)),
          "target_events":int(len(t)),
          "receiver_id_coverage":float(t["receiver_id"].ne("").mean()) if len(t) else 0.0,
          "distinct_receivers":int(t["receiver_id"].nunique()),
          "distinct_team_receivers":int(t[["team","receiver_id"]].drop_duplicates().shape[0]),
          "all_targets":int(g["ctx_all"].sum()),
          "early_down_targets":int(g["ctx_early_down"].sum()),
          "third_down_targets":int(g["ctx_third_down"].sum()),
          "red_zone_targets":int(g["ctx_red_zone"].sum()),
          "two_minute_targets":int(g["ctx_two_minute"].sum()),
          "max_week":int(num(g["week"]).max()) if num(g["week"]).notna().any() else None,
        })
    return pd.DataFrame(rows)

def live_panel(pbp2026,roster,scheduled):
    roster=roster.loc[roster["team"].isin(scheduled)&roster["receiver_id"].ne("")].copy()
    t=pbp2026.loc[pbp2026["target_event"]&pbp2026["team"].isin(scheduled)].copy()
    mappings={
      "ALL":"ctx_all",
      "EARLY_DOWN":"ctx_early_down",
      "THIRD_DOWN":"ctx_third_down",
      "RED_ZONE":"ctx_red_zone",
      "TWO_MINUTE":"ctx_two_minute",
    }
    rows=[]
    for team_id,groom in roster.groupby("team",sort=True):
        tg=t.loc[t["team"].eq(team_id)].copy()
        team_counts={ctx:int(tg[col].sum()) for ctx,col in mappings.items()}
        for r in groom.itertuples(index=False):
            pg=tg.loc[tg["receiver_id"].eq(str(r.receiver_id))].copy()
            player_counts={ctx:int(pg[col].sum()) for ctx,col in mappings.items()}
            row={
              "team":team_id,"receiver_id":str(r.receiver_id),"player":str(r.player),
              "position":str(r.position),"roster_source_week":int(r.roster_source_week),
            }
            for ctx in CONTEXTS:
                den=team_counts[ctx]; n=player_counts[ctx]
                key=ctx.lower()
                row[f"{key}_player_targets"]=n
                row[f"{key}_team_targets"]=den
                row[f"{key}_share"]=float(n/den) if den>0 else np.nan
            rows.append(row)
    out=pd.DataFrame(rows)
    return out

def rho(a,b):
    z=pd.DataFrame({"a":num(a),"b":num(b)}).dropna()
    if len(z)<3 or z.a.nunique()<2 or z.b.nunique()<2: return np.nan
    return float(spearmanr(z.a.to_numpy(float),z.b.to_numpy(float)).statistic)

def nonredundancy(panel):
    live=panel.loc[num(panel["all_player_targets"]).gt(0)].copy()
    rows=[]
    for ctx in ["EARLY_DOWN","THIRD_DOWN","RED_ZONE","TWO_MINUTE"]:
        k=ctx.lower()
        sub=live.loc[live[f"{k}_share"].notna()&live["all_share"].notna()].copy()
        delta=num(sub[f"{k}_share"])-num(sub["all_share"])
        corr=rho(sub[f"{k}_share"],sub["all_share"])
        sd=float(delta.std(ddof=0)) if len(delta) else np.nan
        n5=int(delta.abs().ge(.05).sum()) if len(delta) else 0
        frac=float(delta.abs().ge(.05).mean()) if len(delta) else np.nan
        material=bool(np.isfinite(corr) and corr<.95 and np.isfinite(sd) and sd>=.02 and n5>=25)
        rows.append({
          "context":ctx,"live_players":int(len(sub)),
          "spearman_vs_all":corr,"delta_sd":sd,
          "players_abs_delta_ge_0p05":n5,"fraction_abs_delta_ge_0p05":frac,
          "materially_nonredundant":material,
        })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    frames=[load_pbp(s) for s in SEASONS]
    if any(x.empty for x in frames): raise RuntimeError("one or more PBP seasons empty")
    pbp=pd.concat(frames,ignore_index=True,sort=False)
    inv=inventory(pbp)
    sched=load_schedule()
    if len(sched)!=30: raise RuntimeError(f"expected 30 Week-5 teams, got {len(sched)}")
    roster=load_roster()
    p2026=pbp.loc[pbp["season"].eq(TARGET_SEASON)].copy()
    panel=live_panel(p2026,roster,sched)
    nr=nonredundancy(panel)
    live=panel.loc[num(panel["all_player_targets"]).gt(0)].copy()

    id_cov=float(live["receiver_id"].ne("").mean()) if len(live) else 0.0
    context_team_cov={}
    context_player_cov={}
    for ctx in CONTEXTS:
        k=ctx.lower()
        teams=panel.groupby("team")[f"{k}_team_targets"].max()
        context_team_cov[ctx]=float(teams.gt(0).mean()) if len(teams) else 0.0
        context_player_cov[ctx]=float(live[f"{k}_share"].notna().mean()) if len(live) else 0.0

    target2026=p2026.loc[p2026["target_event"]]
    receiver_cov=float(target2026["receiver_id"].ne("").mean()) if len(target2026) else 0.0
    material=int(nr["materially_nonredundant"].sum())
    ready=bool(
      len(inv)==5 and int(p2026["week"].max())<=4 and receiver_cov>=.99 and len(sched)==30
      and len(live)>=100 and context_player_cov["ALL"]==1.0
      and all(context_team_cov[c]>=.90 for c in ["EARLY_DOWN","THIRD_DOWN","RED_ZONE","TWO_MINUTE"])
      and material>=2
    )
    safe=bool(len(inv)==5 and int(p2026["week"].max())<=4 and receiver_cov>=.95)
    disp=("PLAYER_SITUATIONAL_TARGET_EARNING_SOURCE_READY" if ready else
          "PLAYER_SITUATIONAL_TARGET_EARNING_SOURCE_PARTIAL" if safe else
          "PLAYER_SITUATIONAL_TARGET_EARNING_SOURCE_NOT_READY")
    result={
      "version":VERSION,"captured_at_utc":datetime.now(timezone.utc).isoformat(),
      "disposition":disp,"target_season":TARGET_SEASON,"target_week":TARGET_WEEK,
      "scheduled_teams":len(sched),"roster_source_week":int(panel["roster_source_week"].max()) if len(panel) else None,
      "target_roster_wr_te":int(len(panel)),"live_wr_te_with_prior_target":int(len(live)),
      "live_wr":int(live["position"].eq("WR").sum()),"live_te":int(live["position"].eq("TE").sum()),
      "stable_id_coverage_live_panel":id_cov,
      "receiver_id_coverage_2026_target_events":receiver_cov,
      "max_2026_source_week":int(num(p2026["week"]).max()),
      "context_team_denominator_coverage":context_team_cov,
      "context_player_share_coverage":context_player_cov,
      "materially_nonredundant_contexts":material,
      "nonredundancy":nr.to_dict("records"),
      "sportsbook_inputs_used":0,"week5_outcomes_read":0,"candidate_models_fit":0,"production_changed":False,
    }
    inv.to_csv(a.out_dir/"player_situational_target_earning_source_inventory.csv",index=False)
    panel.to_csv(a.out_dir/"player_situational_target_earning_week5_panel.csv",index=False)
    nr.to_csv(a.out_dir/"player_situational_target_earning_nonredundancy.csv",index=False)
    (a.out_dir/"player_situational_target_earning_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    print("\nNONREDUNDANCY")
    print(nr.to_string(index=False))
    return 0

if __name__=="__main__": raise SystemExit(main())
