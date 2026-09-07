#!/usr/bin/env python3
"""Trace exact B0 vs canonical-simulation C2 integration across one season.

Frozen plan: docs/migrations/PASS_RECEIVING_CONSERVATION_INTEGRATION_V1_PLAN.md
Only B0_CURRENT and C2_INTEGRATED are produced.  Sportsbook inputs are absent.
Target-game outcomes are exported only after both projection paths are complete.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.backtest import trace_joint_pass_receiving_v1 as j
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts import simulation_v2


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--season",type=int,required=True); p.add_argument("--prior-season",type=int,required=True); p.add_argument("--weeks",required=True)
    p.add_argument("--iterations",type=int,default=2000); p.add_argument("--player-logs",type=Path,required=True); p.add_argument("--team-weekly",type=Path,required=True)
    p.add_argument("--schedule",type=Path,required=True); p.add_argument("--universe-dir",type=Path,required=True); p.add_argument("--injuries",type=Path,required=True)
    p.add_argument("--weather",type=Path,required=True); p.add_argument("--m89-root",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()

    logs=j.read(a.player_logs); team_weekly=j.read(a.team_weekly); schedule=j.read(a.schedule); injuries=j.opt(a.injuries); weather=j.opt(a.weather); m89=j.load_m89(a.m89_root); weeks=_parse_weeks(a.weeks)
    player_rows=[]; qb_rows=[]; cons_rows=[]; integrity_rows=[]

    for week in weeks:
        universe=j.read(a.universe_dir/f"{a.season}_week_{int(week):02d}.csv")
        bundle=build_historical_context_bundle(player_logs=logs,team_weekly=team_weekly,pregame_universe=universe,schedule=schedule,season=a.season,week=week,prior_season=a.prior_season,injuries=_exact_week(injuries,a.season,week),weather=_exact_week(weather,a.season,week))
        metrics=j.prepared(bundle); players=j.unique_players(metrics)
        b0=simulation_v2.simulate(metrics,iterations=a.iterations,seed=42+int(week))

        # Pregame mean anchors: exact M89/M90 on its available 2024-25 cohort;
        # otherwise the exact canonical B0 primary-QB MC mean for receiver continuity.
        anchor_map={}; b0_qb_raw={}; b0_qb_key={}; qb_eval={}
        for (game,team),tdf in players.groupby(["event_id","team"],dropna=False,sort=False):
            pkey,qarr=j.primary_qb_arrays(b0,tdf.reset_index(drop=True),game)
            if qarr is None: continue
            key=(str(game),str(team)); b0_qb_raw[key]=np.asarray(qarr,dtype=float); b0_qb_key[key]=str(pkey)
            hit=m89.loc[(m89.season.eq(int(a.season)))&(m89.week.eq(int(week)))&(m89.team.eq(j.canon_team(team)))] if int(a.season) in (2024,2025) else pd.DataFrame()
            if len(hit)>1: raise RuntimeError(f"duplicate M89 anchor season={a.season} week={week} team={team}")
            if len(hit)==1:
                anchor_map[key]=float(hit.iloc[0].football_synthesis); qb_eval[key]=True
            else:
                anchor_map[key]=float(np.mean(qarr)); qb_eval[key]=False

        week_cons=[]
        c2=simulation_v2.apply_pass_receiving_conservation(b0,metrics,anchor_map=anchor_map,seed=200000+42+int(week),conservation_trace=week_cons)
        for r in week_cons:
            cons_rows.append({"season":int(a.season),"week":int(week),**r})

        # Individual receiver means.  C2 intentionally preserves B0 target shares.
        for (game,team),tdf in players.groupby(["event_id","team"],dropna=False,sort=False):
            tdf=tdf.reset_index(drop=True); raw=np.array([simulation_v2._num(r,"rules_tgt_share","bayes_tgt_share","target_share","tgt_share",default=0.0) for _,r in tdf.iterrows()],dtype=float)
            probs=simulation_v2._sharpen_wr_target_shares(tdf,raw); pos=tdf.get("position",pd.Series("",index=tdf.index)).fillna("").astype(str).str.upper().str.strip().to_numpy(); pc=np.isin(pos,list(simulation_v2.PASS_CATCHER_POSITIONS)); probs=np.where(pc,probs,0.0)
            clean=np.clip(np.nan_to_num(probs,nan=0.0,posinf=0.0,neginf=0.0),0.0,0.95)
            if clean.sum()>0.95: clean*=0.95/clean.sum()
            pass_mean=float(np.mean(b0.team_states[(str(game),str(team),"pass_att")]))
            for idx,(_,r) in enumerate(tdf.iterrows()):
                if not pc[idx]: continue
                pkey=str(r.get("player_clean_key","") or ""); identity=str(r.get("player_identity_key","") or "").strip(); join_key=f"id:{identity}" if identity else f"name:{pkey}"; position=str(r.get("position","") or "").upper().strip()
                b0_rec=j.sim_arr(b0,game,pkey,"receptions"); c2_rec=j.sim_arr(c2,game,pkey,"receptions"); b0_y=j.sim_arr(b0,game,pkey,"rec_yards"); c2_y=j.sim_arr(c2,game,pkey,"rec_yards"); b0_r=j.sim_arr(b0,game,pkey,"rush_yards"); c2_r=j.sim_arr(c2,game,pkey,"rush_yards")
                br=float(np.mean(b0_r)) if b0_r is not None else np.nan; cr=float(np.mean(c2_r)) if c2_r is not None else np.nan; by=float(np.mean(b0_y)) if b0_y is not None else np.nan; cy=float(np.mean(c2_y)) if c2_y is not None else np.nan
                player_rows.append({
                    "season":int(a.season),"week":int(week),"event_id":str(game),"team":str(team),"player":r.get("player",""),"player_clean_key":pkey,"player_identity_key":identity,"join_key":join_key,
                    "position":position,"position_group":j.position_group(position),"mass_group":j.mass_group(position),"b0_target_probability":float(clean[idx]),"c2_target_probability":float(clean[idx]),
                    "b0_expected_targets":pass_mean*float(clean[idx]),"c2_expected_targets":pass_mean*float(clean[idx]),
                    "b0_receptions":float(np.mean(b0_rec)) if b0_rec is not None else np.nan,"c2_receptions":float(np.mean(c2_rec)) if c2_rec is not None else np.nan,
                    "b0_rec_yards":by,"c2_rec_yards":cy,"b0_rush_yards":br,"c2_rush_yards":cr,
                    "b0_rush_rec_yards":br+by if np.isfinite(br) and np.isfinite(by) else np.nan,"c2_rush_rec_yards":cr+cy if np.isfinite(cr) and np.isfinite(cy) else np.nan,
                })

        # Exact M89/M90 common-era QB scoreboard only.
        if int(a.season) in (2024,2025):
            for key,raw_b0 in b0_qb_raw.items():
                game,team=key
                if not qb_eval.get(key,False): continue
                hit=m89.loc[(m89.season.eq(int(a.season)))&(m89.week.eq(int(week)))&(m89.team.eq(j.canon_team(team)))]
                anchor=float(hit.iloc[0].football_synthesis); actual=float(hit.iloc[0].actual_pass_yards); raw_mean=float(np.mean(raw_b0))
                if not np.isfinite(raw_mean) or raw_mean<=0: raise RuntimeError(f"invalid B0 QB mean game={game} team={team}")
                b0_anchor=np.asarray(raw_b0,dtype=float)*(anchor/raw_mean); c2a=j.sim_arr(c2,game,b0_qb_key[key],"pass_yards")
                if c2a is None: raise RuntimeError(f"missing integrated C2 QB array game={game} team={team}")
                qb_rows.append({"season":int(a.season),"week":int(week),"event_id":game,"team":j.canon_team(team),"primary_qb_key":b0_qb_key[key],"football_synthesis":anchor,"actual_pass_yards":actual,**j.distribution_stats(b0_anchor,actual,"b0"),**j.distribution_stats(c2a,actual,"c2")})

        rush_keys=[k for k in b0.values if k[2] in {"rush_att","rush_yards"} and k in c2.values]
        max_rush_gap=max([float(np.max(np.abs(np.asarray(b0.values[k])-np.asarray(c2.values[k])))) for k in rush_keys],default=0.0)
        integrity_rows.append({"season":int(a.season),"week":int(week),"sportsbook_inputs":0,"max_rushing_array_gap":max_rush_gap,"zero_rec_positive_yards":int(sum(int(r.get("zero_reception_positive_yards",0)) for r in week_cons))})
        print(f"[integration-v1] season={a.season} week={int(week):02d} receiver_rows={sum(1 for r in player_rows if r['season']==int(a.season) and r['week']==int(week))}")

    out=a.out_dir; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(player_rows).to_csv(out/"integration_v1_player_projection_trace.csv",index=False); pd.DataFrame(qb_rows).to_csv(out/"integration_v1_qb_distribution_trace.csv",index=False); pd.DataFrame(cons_rows).to_csv(out/"integration_v1_conservation_trace.csv",index=False); pd.DataFrame(integrity_rows).to_csv(out/"integration_v1_integrity_counts.csv",index=False)
    j.actual_usage(logs,a.season,set(int(w) for w in weeks)).to_csv(out/"integration_v1_actual_usage.csv",index=False)
    print(f"[integration-v1] wrote player={len(player_rows)} qb={len(qb_rows)} -> {out}"); return 0

if __name__=="__main__": raise SystemExit(main())
