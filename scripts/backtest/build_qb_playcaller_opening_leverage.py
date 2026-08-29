#!/usr/bin/env python3
"""Migration 68 source builder: verified playcaller + opening script + playoff leverage.

M68 follows M67's negative new-information audit. It intentionally targets
week-specific intent-setting information rather than another aggregate feature
bundle.

Leakage rules:
- PBP opening-script features use games strictly before the target week.
- Playcaller mappings are frozen from public season-opening inventories plus
  documented midseason handoffs; no coach-name one-hot features are emitted.
- Playoff leverage uses only results from weeks strictly before the target week,
  a frozen entering-week Elo, and the known regular-season schedule. It never
  uses future actual scores or future betting lines.
- 2024/2025 are scored elsewhere; 2023 is history context only.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team


ESPN_SEASON_SOURCES = {
    2023: "https://www.espn.com/nfl/story/_/id/38108724/key-intel-all-32-nfl-playcallers-including-mike-mccarthy",
    2024: "https://www.espn.com/nfl/story/_/id/41018846/nfl-playcallers-32-teams-mike-mcdaniel-sean-mcvay-nathaniel-hackett",
    2025: "https://www.espn.com/nfl/story/_/id/46137832/nfl-playcallers-32-teams-mike-mcdaniel-sean-mcvay-brian-schottenheimer",
}

# Frozen, manually verified season-opening primary playcallers from the ESPN
# all-32-team inventories above. This table is deliberately embedded so M68 is
# reproducible even if an article later changes or blocks automated access.
SEASON_OPENING_CALLER = {
    2023: {
        "ARI":"Drew Petzing","ATL":"Arthur Smith","BAL":"Todd Monken","BUF":"Ken Dorsey",
        "CAR":"Frank Reich","CHI":"Luke Getsy","CIN":"Zac Taylor","CLE":"Kevin Stefanski",
        "DAL":"Mike McCarthy","DEN":"Sean Payton","DET":"Ben Johnson","GB":"Matt LaFleur",
        "HOU":"Bobby Slowik","IND":"Shane Steichen","JAX":"Doug Pederson","KC":"Andy Reid",
        "LAC":"Kellen Moore","LAR":"Sean McVay","LV":"Josh McDaniels","MIA":"Mike McDaniel",
        "MIN":"Kevin O'Connell","NE":"Bill O'Brien","NO":"Pete Carmichael","NYG":"Mike Kafka",
        "NYJ":"Nathaniel Hackett","PHI":"Brian Johnson","PIT":"Matt Canada","SF":"Kyle Shanahan",
        "SEA":"Shane Waldron","TB":"Dave Canales","TEN":"Tim Kelly","WAS":"Eric Bieniemy",
    },
    2024: {
        "ARI":"Drew Petzing","ATL":"Zac Robinson","BAL":"Todd Monken","BUF":"Joe Brady",
        "CAR":"Dave Canales","CHI":"Shane Waldron","CIN":"Zac Taylor","CLE":"Kevin Stefanski",
        "DAL":"Mike McCarthy","DEN":"Sean Payton","DET":"Ben Johnson","GB":"Matt LaFleur",
        "HOU":"Bobby Slowik","IND":"Shane Steichen","JAX":"Press Taylor","KC":"Andy Reid",
        "LAC":"Greg Roman","LAR":"Sean McVay","LV":"Luke Getsy","MIA":"Mike McDaniel",
        "MIN":"Kevin O'Connell","NE":"Alex Van Pelt","NO":"Klint Kubiak","NYG":"Brian Daboll",
        "NYJ":"Nathaniel Hackett","PHI":"Kellen Moore","PIT":"Arthur Smith","SF":"Kyle Shanahan",
        "SEA":"Ryan Grubb","TB":"Liam Coen","TEN":"Brian Callahan","WAS":"Kliff Kingsbury",
    },
    2025: {
        "ARI":"Drew Petzing","ATL":"Zac Robinson","BAL":"Todd Monken","BUF":"Joe Brady",
        "CAR":"Dave Canales","CHI":"Ben Johnson","CIN":"Zac Taylor","CLE":"Kevin Stefanski",
        "DAL":"Brian Schottenheimer","DEN":"Sean Payton","DET":"John Morton","GB":"Matt LaFleur",
        "HOU":"Nick Caley","IND":"Shane Steichen","JAX":"Liam Coen","KC":"Andy Reid",
        "LAC":"Greg Roman","LAR":"Sean McVay","LV":"Chip Kelly","MIA":"Mike McDaniel",
        "MIN":"Kevin O'Connell","NE":"Josh McDaniels","NO":"Kellen Moore","NYG":"Mike Kafka",
        "NYJ":"Tanner Engstrand","PHI":"Kevin Patullo","PIT":"Arthur Smith","SF":"Kyle Shanahan",
        "SEA":"Klint Kubiak","TB":"Josh Grizzard","TEN":"Brian Callahan","WAS":"Kliff Kingsbury",
    },
}

# Documented primary playcalling handoffs. effective_week means the first
# regular-season week for which the new caller handled game-day calls.
CALLER_OVERRIDES = [
    # 2023
    (2023,"BUF",11,"Joe Brady","https://www.buffalobills.com/news/joe-brady-to-take-over-as-bills-interim-offensive-coordinator"),
    (2023,"CAR",8,"Thomas Brown","https://www.nfl.com/news/panthers-head-coach-frank-reich-hands-over-play-calling-duties-to-offensive-coor"),
    (2023,"CAR",11,"Frank Reich","https://www.panthers.com/news/frank-reich-to-take-back-play-calling-duties"),
    (2023,"CAR",13,"Thomas Brown","https://www.panthers.com/news/thomas-brown-will-call-plays-again-after-frank-reich-firing"),
    (2023,"LV",9,"Bo Hardegree","https://www.raiders.com/news/bo-hardegree-named-interim-offensive-coordinator-2023-nfl-season"),
    (2023,"PIT",12,"Mike Sullivan","https://www.steelers.com/news/matt-canada-relieved-of-duties"),
    # 2024
    (2024,"CLE",8,"Ken Dorsey","https://www.clevelandbrowns.com/news/ken-dorsey-to-take-over-play-calling-duties"),
    (2024,"NYJ",6,"Todd Downing","https://www.nfl.com/news/jets-todd-downing-to-take-over-play-calling-duties-from-nathaniel-hackett"),
    (2024,"CHI",11,"Thomas Brown","https://www.chicagobears.com/news/bears-relieve-shane-waldron-of-duties-offensive-coordinator-thomas-brown"),
    (2024,"LV",11,"Scott Turner","https://www.raiders.com/news/scott-turner-named-interim-offensive-coordinator-2024"),
    # 2025
    (2025,"TEN",4,"Bo Hardegree","https://www.tennesseetitans.com/news/titans-hc-brian-callahan-hands-off-play-calling-duties-to-qbs-coach-bo-hardegree"),
    (2025,"CLE",10,"Tommy Rees","https://www.clevelandbrowns.com/news/tommy-rees-to-take-over-play-calling-duties"),
    (2025,"DET",10,"Dan Campbell","https://www.detroitlions.com/news/dan-campbell-takes-over-offensive-play-calling-duties"),
    (2025,"LV",13,"Greg Olson","https://www.raiders.com/news/greg-olson-to-call-offensive-plays-after-chip-kelly-departure"),
]

DIVISION = {
    "BUF":"AFC_E","MIA":"AFC_E","NE":"AFC_E","NYJ":"AFC_E",
    "BAL":"AFC_N","CIN":"AFC_N","CLE":"AFC_N","PIT":"AFC_N",
    "HOU":"AFC_S","IND":"AFC_S","JAX":"AFC_S","TEN":"AFC_S",
    "DEN":"AFC_W","KC":"AFC_W","LV":"AFC_W","LAC":"AFC_W",
    "DAL":"NFC_E","NYG":"NFC_E","PHI":"NFC_E","WAS":"NFC_E",
    "CHI":"NFC_N","DET":"NFC_N","GB":"NFC_N","MIN":"NFC_N",
    "ATL":"NFC_S","CAR":"NFC_S","NO":"NFC_S","TB":"NFC_S",
    "ARI":"NFC_W","LAR":"NFC_W","SF":"NFC_W","SEA":"NFC_W",
}
CONFERENCE = {t: d.split("_")[0] for t,d in DIVISION.items()}


def to_pd(o) -> pd.DataFrame:
    if isinstance(o, pd.DataFrame):
        return o.copy()
    if hasattr(o, "to_pandas"):
        return o.to_pandas()
    return pd.DataFrame(o)


def lower(x: pd.DataFrame) -> pd.DataFrame:
    y=x.copy()
    y.columns=[str(c).strip().lower() for c in y.columns]
    return y


def num(v):
    return pd.to_numeric(v, errors="coerce")


def text(v) -> pd.Series:
    return v.fillna("").astype(str).str.strip()


def canon(v) -> str:
    t=canon_team(v)
    return "WAS" if t == "WSH" else t


def caller_for(season: int, week: int, team: str) -> str:
    team=canon(team)
    caller=SEASON_OPENING_CALLER.get(int(season),{}).get(team,"")
    for s,t,w,c,_ in CALLER_OVERRIDES:
        if s==int(season) and t==team and int(week)>=w:
            caller=c
    return caller


def load_sources(seasons: list[int]) -> tuple[pd.DataFrame,pd.DataFrame]:
    import nflreadpy as nfl
    pbps=[]
    scheds=[]
    for s in sorted(set(map(int,seasons))):
        q=lower(to_pd(nfl.load_pbp(seasons=[s])))
        if len(q):
            pbps.append(q)
        r=lower(to_pd(nfl.load_schedules(s)))
        if len(r):
            scheds.append(r)
    pbp=pd.concat(pbps,ignore_index=True,sort=False) if pbps else pd.DataFrame()
    sched=pd.concat(scheds,ignore_index=True,sort=False) if scheds else pd.DataFrame()
    if pbp.empty or sched.empty:
        raise RuntimeError("M68 requires nflverse PBP and schedules")
    return pbp,sched


def regular_rows(x: pd.DataFrame) -> pd.DataFrame:
    q=x.copy()
    col="season_type" if "season_type" in q else "game_type" if "game_type" in q else None
    if col:
        s=text(q[col]).str.upper()
        reg=q[s.isin(["REG","REGULAR","RS",""])].copy()
        if len(reg):
            q=reg
    return q


def opening_game_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-game opening-script metrics from PBP only (no participation join)."""
    x=regular_rows(pbp)
    for c in ["season","week","play_id","qb_dropback","pass_attempt","rush_attempt","down",
              "score_differential","shotgun","drive"]:
        if c not in x:
            x[c]=np.nan
    team_col="posteam" if "posteam" in x else "possession_team"
    if team_col not in x:
        raise RuntimeError("M68 PBP missing possession team")
    x["team"]=x[team_col].map(canon)
    x=x[x.team.ne("")].copy()
    x["_dropback"]=num(x.qb_dropback).fillna(0).clip(0,1)
    x["_pass"]=num(x.pass_attempt).fillna(0).eq(1)
    x["_rush"]=num(x.rush_attempt).fillna(0).eq(1)
    x=x[(x._dropback.eq(1)|x._pass|x._rush)].copy()
    x["_play_id"]=num(x.play_id)
    x["_down"]=num(x.down)
    x["_score"]=num(x.score_differential)
    x["_shotgun"]=num(x.shotgun)
    x["_drive"]=num(x.drive)
    game_col="game_id" if "game_id" in x else None
    if not game_col:
        raise RuntimeError("M68 PBP missing game_id")
    x=x.sort_values(["season","week",game_col,"team","_play_id"],kind="mergesort")
    x["_team_play_no"]=x.groupby(["season","week",game_col,"team"]).cumcount()+1

    rows=[]
    for (season,week,game_id,team),g in x.groupby(["season","week",game_col,"team"],sort=True):
        g=g.sort_values("_play_id")
        if not np.isfinite(num(pd.Series([season])).iloc[0]) or not np.isfinite(num(pd.Series([week])).iloc[0]):
            continue
        def dbr(mask) -> float:
            z=g[mask]
            return float(z._dropback.mean()) if len(z) else np.nan
        first10=g._team_play_no.le(10)
        first15=g._team_play_no.le(15)
        rest=g._team_play_no.gt(15)
        dv=[v for v in g._drive.dropna().unique().tolist()]
        first_drive=g._drive.eq(dv[0]) if dv else first10
        first2=g._drive.isin(dv[:2]) if dv else g._team_play_no.le(20)
        early_neutral=first15 & g._down.le(2) & g._score.abs().le(7)
        f15=dbr(first15)
        rest_dbr=dbr(rest)
        rec={
            "season":int(season),"week":int(week),"game_id":str(game_id),"team":canon(team),
            "caller":caller_for(int(season),int(week),team),
            "opening_first10_dbr":dbr(first10),
            "opening_first15_dbr":f15,
            "opening_first_drive_dbr":dbr(first_drive),
            "opening_first2drives_dbr":dbr(first2),
            "opening_first15_early_neutral_dbr":dbr(early_neutral),
            "opening_first15_shotgun_rate":float(g.loc[first15,"_shotgun"].mean()) if g.loc[first15,"_shotgun"].notna().any() else np.nan,
            "opening_first15_vs_rest_dbr":f15-rest_dbr if np.isfinite(f15) and np.isfinite(rest_dbr) else np.nan,
            "opening_q1_dbr":np.nan,
        }
        if "qtr" in g:
            rec["opening_q1_dbr"]=dbr(num(g.qtr).eq(1))
        rows.append(rec)
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("M68 opening-script aggregation produced zero team-games")
    return out


OPEN_BASE=[
    "opening_first10_dbr","opening_first15_dbr","opening_first_drive_dbr",
    "opening_first2drives_dbr","opening_first15_early_neutral_dbr",
    "opening_first15_shotgun_rate","opening_first15_vs_rest_dbr","opening_q1_dbr",
]


def prior_roll_features(hist: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe team opening and caller tendency features."""
    h=hist.copy()
    t=target[["season","week","team"]].drop_duplicates().copy()
    h["team"]=h.team.map(canon)
    t["team"]=t.team.map(canon)
    rows=[]
    for r in t.itertuples(index=False):
        season,week,team=int(r.season),int(r.week),canon(r.team)
        curcaller=caller_for(season,week,team)
        prior_team=h[(h.team.eq(team)) & ((num(h.season)<season) | ((num(h.season)==season)&(num(h.week)<week)))].sort_values(["season","week"])
        rec={"season":season,"week":week,"team":team}
        for c in OPEN_BASE:
            vals=num(prior_team[c]).dropna()
            rec[f"{c}_last1"]=float(vals.tail(1).mean()) if len(vals) else np.nan
            rec[f"{c}_mean3"]=float(vals.tail(3).mean()) if len(vals) else np.nan
            rec[f"{c}_mean8"]=float(vals.tail(8).mean()) if len(vals) else np.nan
            m3=float(vals.tail(3).mean()) if len(vals) else np.nan
            m8=float(vals.tail(8).mean()) if len(vals) else np.nan
            rec[f"{c}_trend3v8"]=m3-m8 if np.isfinite(m3) and np.isfinite(m8) else np.nan

        prior_all=h[((num(h.season)<season) | ((num(h.season)==season)&(num(h.week)<week)))].copy()
        pc=prior_all[prior_all.caller.eq(curcaller)].sort_values(["season","week"]) if curcaller else prior_all.iloc[0:0]
        pct=pc[pc.team.eq(team)]
        prev_week_team=prior_team.tail(1)
        prev_caller=str(prev_week_team.caller.iloc[0]) if len(prev_week_team) else ""
        rec["playcaller_changed_since_last_game"]=float(bool(prev_caller) and bool(curcaller) and prev_caller!=curcaller)
        rec["playcaller_prior_games_allteams"]=float(len(pc))
        rec["playcaller_prior_games_team"]=float(len(pct))
        rec["playcaller_new_to_team"]=float(len(pct)==0)
        for c in ["opening_first15_dbr","opening_first_drive_dbr","opening_first2drives_dbr","opening_q1_dbr"]:
            vals=num(pc[c]).dropna()
            rec[f"playcaller_{c}_mean3"]=float(vals.tail(3).mean()) if len(vals) else np.nan
            rec[f"playcaller_{c}_mean8"]=float(vals.tail(8).mean()) if len(vals) else np.nan
            tv=num(pct[c]).dropna()
            rec[f"playcaller_team_{c}_mean3"]=float(tv.tail(3).mean()) if len(tv) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def schedule_regular(sched: pd.DataFrame) -> pd.DataFrame:
    x=regular_rows(sched)
    for c in ["season","week","home_team","away_team","home_score","away_score"]:
        if c not in x:
            x[c]=np.nan
    x["season"]=num(x.season)
    x["week"]=num(x.week)
    x["home"]=x.home_team.map(canon)
    x["away"]=x.away_team.map(canon)
    x["home_score"]=num(x.home_score)
    x["away_score"]=num(x.away_score)
    return x[x.season.notna()&x.week.notna()&x.home.ne("")&x.away.ne("")].copy()


def entering_elo(sched: pd.DataFrame, season: int, week: int) -> dict[str,float]:
    """Simple pregame Elo updated only through weeks strictly before target."""
    elo={t:1500.0 for t in DIVISION}
    s=sched[(sched.season==season)&(sched.week<week)].sort_values(["week"])
    for r in s.itertuples(index=False):
        if r.home not in elo or r.away not in elo or not np.isfinite(r.home_score) or not np.isfinite(r.away_score):
            continue
        eh=1/(1+10**(-(elo[r.home]+55-elo[r.away])/400))
        a=1.0 if r.home_score>r.away_score else 0.0 if r.home_score<r.away_score else 0.5
        margin=abs(float(r.home_score-r.away_score))
        mult=math.log(max(1.0,margin)+1.0)*(2.2/((abs(elo[r.home]-elo[r.away])*0.001)+2.2))
        delta=20.0*mult*(a-eh)
        elo[r.home]+=delta
        elo[r.away]-=delta
    return elo


def standings_before(sched: pd.DataFrame, season: int, week: int) -> tuple[dict[str,float],dict[str,float]]:
    wins={t:0.0 for t in DIVISION}
    games={t:0.0 for t in DIVISION}
    s=sched[(sched.season==season)&(sched.week<week)]
    for r in s.itertuples(index=False):
        if r.home not in wins or r.away not in wins or not np.isfinite(r.home_score) or not np.isfinite(r.away_score):
            continue
        games[r.home]+=1
        games[r.away]+=1
        if r.home_score>r.away_score:
            wins[r.home]+=1
        elif r.away_score>r.home_score:
            wins[r.away]+=1
        else:
            wins[r.home]+=.5
            wins[r.away]+=.5
    return wins,games


def playoff_qualifiers(win_totals: dict[str,float]) -> set[str]:
    """Approximate 7-team conference field using wins, deterministic tie break."""
    out=set()
    for conf in ("AFC","NFC"):
        div_winners=[]
        for div in sorted({d for d in DIVISION.values() if d.startswith(conf)}):
            ts=[t for t,d in DIVISION.items() if d==div]
            winner=sorted(ts,key=lambda t:(-win_totals.get(t,0.0),t))[0]
            div_winners.append(winner)
            out.add(winner)
        rest=[t for t,c in CONFERENCE.items() if c==conf and t not in div_winners]
        rest=sorted(rest,key=lambda t:(-win_totals.get(t,0.0),t))
        out.update(rest[:3])
    return out


def simulate_leverage_for_target(sched: pd.DataFrame, season: int, week: int, team: str, opponent: str, sims: int=400, seed: int=68) -> dict[str,float]:
    team,opponent=canon(team),canon(opponent)
    base_wins,games=standings_before(sched,season,week)
    elo=entering_elo(sched,season,week)
    future=sched[(sched.season==season)&(sched.week>=week)].copy().sort_values(["week"])
    mask=(future.week==week)&(((future.home==team)&(future.away==opponent))|((future.away==team)&(future.home==opponent)))
    if not mask.any():
        return {"leverage_playoff_if_win":np.nan,"leverage_playoff_if_loss":np.nan,
                "leverage_playoff_delta":np.nan,"leverage_entering_playoff_prob":np.nan,
                "leverage_games_played":games.get(team,0.0),"leverage_week":float(week)}
    target_idx=future[mask].index[0]
    rng=np.random.default_rng(seed + season*100 + week*3 + sum(ord(c) for c in team))
    hit={1:0,0:0}
    games_list=list(future.itertuples())
    draws=rng.random((sims,len(games_list)))
    for forced in (1,0):
        for i in range(sims):
            w=dict(base_wins)
            for j,r in enumerate(games_list):
                h,a=r.home,r.away
                if h not in w or a not in w:
                    continue
                if r.Index==target_idx:
                    team_wins=bool(forced)
                    home_win=team_wins if h==team else (not team_wins)
                else:
                    ph=1/(1+10**(-(elo.get(h,1500)+55-elo.get(a,1500))/400))
                    home_win=draws[i,j] < ph
                w[h]+=1.0 if home_win else 0.0
                w[a]+=0.0 if home_win else 1.0
            if team in playoff_qualifiers(w):
                hit[forced]+=1
    pwin=hit[1]/sims
    ploss=hit[0]/sims
    return {
        "leverage_playoff_if_win":pwin,
        "leverage_playoff_if_loss":ploss,
        "leverage_playoff_delta":pwin-ploss,
        "leverage_entering_playoff_prob":0.5*(pwin+ploss),
        "leverage_games_played":games.get(team,0.0),
        "leverage_week":float(week),
    }


def build_leverage(sched: pd.DataFrame,target: pd.DataFrame,sims:int) -> pd.DataFrame:
    s=schedule_regular(sched)
    rows=[]
    cache={}
    for r in target[["season","week","team","opponent"]].drop_duplicates().itertuples(index=False):
        key=(int(r.season),int(r.week),canon(r.team),canon(r.opponent))
        if key not in cache:
            cache[key]=simulate_leverage_for_target(s,*key,sims=sims)
        rec={"season":key[0],"week":key[1],"team":key[2],**cache[key]}
        wins,games=standings_before(s,key[0],key[1])
        gp=games.get(key[2],0.0)
        w=wins.get(key[2],0.0)
        rec["leverage_entering_win_pct"]=w/gp if gp else np.nan
        rec["leverage_games_remaining"]=17.0-gp
        rows.append(rec)
    return pd.DataFrame(rows)


def source_manifest() -> pd.DataFrame:
    rows=[]
    for s,u in ESPN_SEASON_SOURCES.items():
        rows.append({"family":"verified_playcaller","season":s,"effective_week":1,
                     "source":u,"status":"frozen_verified","notes":"season-opening all-32-team inventory"})
    for s,t,w,c,u in CALLER_OVERRIDES:
        rows.append({"family":"verified_playcaller_override","season":s,"team":t,
                     "effective_week":w,"caller":c,"source":u,"status":"frozen_verified",
                     "notes":"documented midseason primary playcalling handoff"})
    rows.append({"family":"opening_script","source":"nflverse_pbp","status":"recovered_live_capable",
                 "notes":"direct PBP only; independent of participation; strictly prior games"})
    rows.append({"family":"simulated_playoff_leverage","source":"nflverse_schedules","status":"derived_live_capable",
                 "notes":"pregame prior-results Elo simulation; approximate playoff seeding, not exact NFL tiebreakers"})
    return pd.DataFrame(rows)


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--master-game-level",type=Path,required=True)
    p.add_argument("--seasons",default="2023,2024,2025")
    p.add_argument("--leverage-sims",type=int,default=400)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)

    master=lower(pd.read_csv(args.master_game_level))
    master["team"]=master.team.map(canon)
    master["opponent"]=master.opponent.map(canon)
    seasons=[int(x) for x in args.seasons.split(",") if x.strip()]
    pbp,sched=load_sources(seasons)
    hist=opening_game_metrics(pbp)

    target=master[["season","week","team","opponent"]].copy()
    opens=prior_roll_features(hist,target)
    lev=build_leverage(sched,target,args.leverage_sims)
    out=target.merge(opens,on=["season","week","team"],how="left",validate="many_to_one")
    out=out.merge(lev,on=["season","week","team"],how="left",validate="many_to_one")
    if len(out)!=len(master):
        raise RuntimeError("M68 source merge changed canonical row count")

    out["playcaller_current_name"]=[caller_for(int(s),int(w),t) for s,w,t in zip(out.season,out.week,out.team)]
    model_cols=[c for c in out if c.startswith(("opening_","playcaller_","leverage_")) and c!="playcaller_current_name"]
    coverage=pd.DataFrame([
        {"feature":c,"non_null":int(num(out[c]).notna().sum()),"n":len(out),
         "coverage":float(num(out[c]).notna().mean()),
         "family":"opening_script" if c.startswith("opening_") else "verified_playcaller" if c.startswith("playcaller_") else "playoff_leverage"}
        for c in model_cols
    ])

    hist.to_csv(args.out_dir/"m68_historical_opening_script_team_games.csv",index=False)
    out.to_csv(args.out_dir/"m68_pregame_new_information_features.csv",index=False)
    coverage.to_csv(args.out_dir/"m68_feature_coverage.csv",index=False)
    source_manifest().to_csv(args.out_dir/"m68_source_manifest.csv",index=False)

    print("=== M68 SOURCE COVERAGE ===")
    print(coverage.groupby("family").agg(features=("feature","size"),median_coverage=("coverage","median"),min_coverage=("coverage","min")).reset_index().to_string(index=False))
    print(f"[M68] target_rows={len(out)} history_team_games={len(hist)} caller_missing={(out.playcaller_current_name=='').sum()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
