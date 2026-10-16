#!/usr/bin/env python3
"""Parameterized prospective player-state source audit for future opportunity locks.

This is a mechanical generalization of the frozen Week-5 Player State Live
Coverage V1 architecture. It reads no target-week outcomes, uses no sportsbook
inputs, fits no model, and changes no production code.

Execution must fail closed unless the immediately prior scheduled-week source
state is complete for target-week teams that played in that prior week.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas

VERSION="PLAYER_STATE_FUTURE_WEEK_COVERAGE_V1"
SKILL={"QB","RB","WR","TE"}
RB_RAW={"RB","HB","FB"}
# Preserve the pre-existing pregame-universe semantics: only game-roster players
# are legitimate current-role identities. Reserve/development/cut/retired rows
# may remain in nflverse's weekly roster snapshot but are not projected players.
GAME_ROSTER_STATUS={"ACT","INA"}
FORBIDDEN=(
    "sportsbook","bookmaker","prop_line","market_line","over_odds","under_odds",
    "spread_line","total_line","moneyline","closing_line","no_vig","implied_prob",
)

def _pd(x): return _to_pandas(x)

def _lower(x):
    y=_pd(x).copy()
    y.columns=[str(c).strip().lower() for c in y.columns]
    return y

def _first(df,names,default=""):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default,index=df.index)

def _num(x): return pd.to_numeric(x,errors="coerce")

def _clean_id(v):
    if v is None or pd.isna(v): return ""
    s=str(v).strip()
    return "" if s.lower() in {"","nan","none","<na>"} else s

def _name_key(v):
    s="" if v is None or pd.isna(v) else str(v).lower()
    s=re.sub(r"\b(jr|sr|ii|iii|iv|v)\b","",s)
    return "".join(ch for ch in s if ch.isalnum())

def _team(v):
    try: return canon_team(v)
    except Exception: return str(v or "").strip().upper()

def _pos(v):
    s=str(v or "").upper().strip()
    if s=="QB": return "QB"
    if s in RB_RAW: return "RB"
    if s in {"WR","LWR","RWR","SWR"} or "WR" in s: return "WR"
    if s=="TE" or s.startswith("TE"): return "TE"
    return ""

def _check_forbidden(label,df):
    bad=[c for c in df.columns if any(t in str(c).lower() for t in FORBIDDEN)]
    if bad: raise RuntimeError(f"forbidden fields in {label}: {bad}")

def _load_schedule_all(season:int)->pd.DataFrame:
    import nflreadpy as nfl
    try: raw=nfl.load_schedules(seasons=[season])
    except TypeError: raw=nfl.load_schedules(season)
    x=_lower(raw)
    if "game_type" in x.columns:
        x=x.loc[x["game_type"].astype(str).str.upper().eq("REG")].copy()
    elif "season_type" in x.columns:
        x=x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
    x=x.loc[_num(x["season"]).eq(season)].copy()
    hc="home_team" if "home_team" in x.columns else "home"
    ac="away_team" if "away_team" in x.columns else "away"
    rows=[]
    for r in x.itertuples(index=False):
        w=int(getattr(r,"week"))
        h=_team(getattr(r,hc)); a=_team(getattr(r,ac))
        if h and a:
            rows.append({"week":w,"home":h,"away":a})
    out=pd.DataFrame(rows)
    if out.empty: raise RuntimeError(f"no regular-season schedule rows season={season}")
    return out

def target_schedule(schedule:pd.DataFrame,target_week:int)->pd.DataFrame:
    x=schedule.loc[schedule["week"].eq(target_week)].copy()
    rows=[]
    for r in x.itertuples(index=False):
        rows.extend([
            {"team":r.home,"opponent":r.away},
            {"team":r.away,"opponent":r.home},
        ])
    out=pd.DataFrame(rows).drop_duplicates("team")
    if len(out)<20:
        raise RuntimeError(f"target schedule too small week={target_week}: {len(out)} teams")
    return out

def target_week_roster_source_ready(roster:pd.DataFrame,target_week:int)->bool:
    """Require exact target-week roster authority for an immutable future lock."""
    if roster.empty or "roster_source_week" not in roster.columns:
        return False
    weeks=_num(roster["roster_source_week"]).dropna().astype(int).unique().tolist()
    return len(weeks)==1 and weeks[0]==int(target_week)

def prior_week_scheduled_target_teams(
    schedule:pd.DataFrame,target_teams:set[str],target_week:int
)->set[str]:
    prior=target_week-1
    x=schedule.loc[schedule["week"].eq(prior)].copy()
    teams=set(x["home"])|set(x["away"]) if len(x) else set()
    return set(target_teams)&teams

def load_rosters(season:int,target_week:int):
    import nflreadpy as nfl
    raw=_lower(nfl.load_rosters_weekly(season))
    out=pd.DataFrame(index=raw.index)
    out["season"]=_num(_first(raw,["season"],season)).fillna(season)
    out["week"]=_num(_first(raw,["week"]))
    out["team"]=_first(raw,["team","team_abbr","club_code"]).map(_team)
    out["player"]=_first(raw,["full_name","football_name","player_name","player","name"]).astype("string").fillna("").str.strip()
    out["name_key"]=out["player"].map(_name_key)
    out["position_raw"]=_first(raw,["position","pos","depth_chart_position","ngs_position"]).astype("string").fillna("").str.upper().str.strip()
    out["position_group"]=out["position_raw"].map(_pos)
    out["gsis_id"]=_first(raw,["gsis_id","player_id"]).map(_clean_id)
    out["pfr_id"]=_first(raw,["pfr_id","pfr_player_id"]).map(_clean_id)
    out["status"]=_first(raw,["status","roster_status"]).astype("string").fillna("").str.upper().str.strip()
    out=out.loc[out["season"].eq(season)&out["position_group"].isin(SKILL)&out["team"].ne("")&out["name_key"].ne("")].copy()
    exact=out.loc[out["week"].eq(target_week)].copy()
    if len(exact):
        use=exact; source_week=target_week
    else:
        prior=out.loc[out["week"].lt(target_week)].copy()
        if prior.empty: raise RuntimeError("no target-week or prior weekly roster")
        source_week=int(prior["week"].max())
        use=prior.loc[prior["week"].eq(source_week)].copy()
    keep=use["status"].isin(GAME_ROSTER_STATUS)
    if keep.any():
        use=use.loc[keep].copy()
    use["roster_source_week"]=source_week
    return use.sort_values(["team","gsis_id","pfr_id","name_key"]).drop_duplicates(["team","name_key"],keep="last")

def load_player_stats(season:int,target_week:int)->pd.DataFrame:
    import nflreadpy as nfl
    x=_normalize_weekly(_pd(nfl.load_player_stats(seasons=[season],summary_level="week")),season)
    _check_forbidden(f"stats_{season}",x)
    seen=sorted(_num(x["week"]).dropna().astype(int).unique().tolist())
    bad=[w for w in seen if w>=target_week]
    if bad:
        raise RuntimeError(f"target/future stat rows already present; refuse audit: weeks={bad}")
    return x.loc[_num(x["week"]).lt(target_week)].copy()

def load_prior_stats(prior_season:int)->pd.DataFrame:
    import nflreadpy as nfl
    x=_normalize_weekly(_pd(nfl.load_player_stats(seasons=[prior_season],summary_level="week")),prior_season)
    _check_forbidden(f"stats_{prior_season}",x)
    return x

def load_snaps(season:int,target_week:int)->pd.DataFrame:
    import nflreadpy as nfl
    raw=_lower(nfl.load_snap_counts(seasons=[season]))
    out=pd.DataFrame(index=raw.index)
    out["season"]=_num(_first(raw,["season"],season)).fillna(season)
    out["week"]=_num(_first(raw,["week"]))
    out["team"]=_first(raw,["team","team_abbr"]).map(_team)
    out["player"]=_first(raw,["player","player_name","full_name"]).astype("string").fillna("").str.strip()
    out["name_key"]=out["player"].map(_name_key)
    out["pfr_id"]=_first(raw,["pfr_player_id","pfr_id"]).map(_clean_id)
    out["position_raw"]=_first(raw,["position","pos"]).astype("string").fillna("").str.upper().str.strip()
    out["position_group"]=out["position_raw"].map(_pos)
    out["offense_snaps"]=_num(_first(raw,["offense_snaps","off_snaps"],np.nan))
    out["offense_pct"]=_num(_first(raw,["offense_pct","off_pct","offense_percentage"],np.nan))
    bad=sorted(out.loc[out["week"].ge(target_week),"week"].dropna().astype(int).unique().tolist())
    if bad:
        raise RuntimeError(f"target/future snap rows already present; refuse audit: weeks={bad}")
    out=out.loc[out["season"].eq(season)&out["week"].lt(target_week)&out["position_group"].isin(SKILL)].copy()
    return out.drop_duplicates(["week","team","pfr_id","name_key"],keep="last")

def source_complete_for_prior_week(
    current:pd.DataFrame,
    snaps:pd.DataFrame,
    prior_required_teams:set[str],
    target_week:int,
)->dict:
    prior_week=target_week-1
    stat_teams=set(current.loc[_num(current["week"]).eq(prior_week),"team"].map(_team))
    snap_teams=set(snaps.loc[_num(snaps["week"]).eq(prior_week),"team"].map(_team))
    missing_stats=sorted(prior_required_teams-stat_teams)
    missing_snaps=sorted(prior_required_teams-snap_teams)
    return {
        "prior_week":prior_week,
        "required_teams":sorted(prior_required_teams),
        "stat_teams_present":sorted(prior_required_teams&stat_teams),
        "snap_teams_present":sorted(prior_required_teams&snap_teams),
        "missing_stat_teams":missing_stats,
        "missing_snap_teams":missing_snaps,
        "complete":not missing_stats and not missing_snaps,
    }

def load_injuries(season:int,target_week:int):
    import nflreadpy as nfl
    try: raw=_lower(nfl.load_injuries(seasons=[season]))
    except Exception:
        return pd.DataFrame(columns=["team","name_key","injury_status","practice_status"])
    if raw.empty:
        return pd.DataFrame(columns=["team","name_key","injury_status","practice_status"])
    out=pd.DataFrame(index=raw.index)
    out["week"]=_num(_first(raw,["week","report_week"]))
    out["team"]=_first(raw,["team","team_abbr","team_abbreviation","club"]).map(_team)
    out["name_key"]=_first(raw,["full_name","player_name","player","name"]).map(_name_key)
    out["injury_status"]=_first(raw,["report_status","game_status","status"]).astype("string").fillna("").str.upper().str.strip()
    out["practice_status"]=_first(raw,["practice_status","practice_participation"]).astype("string").fillna("").str.upper().str.strip()
    return out.loc[out["week"].eq(target_week)&out["team"].ne("")&out["name_key"].ne(""),
                   ["team","name_key","injury_status","practice_status"]].drop_duplicates(["team","name_key"],keep="last")

def stable_key(row):
    g=_clean_id(row.get("gsis_id",""))
    if g:return "gsis:"+g
    p=_clean_id(row.get("pfr_id",""))
    if p:return "pfr:"+p
    return "name:"+str(row.get("team",""))+":"+str(row.get("name_key",""))

def join_stats(roster,stats,prefix):
    s=stats.copy()
    s["gsis_id"]=s["player_id"].map(_clean_id)
    s["name_key"]=s["player"].map(_name_key)
    s["team"]=s["team"].map(_team)
    agg=(s.groupby("gsis_id",dropna=False).agg(**{
        f"{prefix}_games":("week","nunique"),
        f"{prefix}_targets":("targets","sum"),
        f"{prefix}_rushes":("rushes","sum"),
        f"{prefix}_rec_yards":("rec_yards","sum"),
        f"{prefix}_rush_yards":("rush_yards","sum"),
        f"{prefix}_last_team":("team","last"),
    }).reset_index())
    agg=agg.loc[agg["gsis_id"].ne("")].copy()
    out=roster.merge(agg,on="gsis_id",how="left",validate="many_to_one")
    missing=out[f"{prefix}_games"].isna()
    if missing.any():
        n=(s.groupby(["team","name_key"],dropna=False).agg(**{
            f"{prefix}_games_fb":("week","nunique"),
            f"{prefix}_targets_fb":("targets","sum"),
            f"{prefix}_rushes_fb":("rushes","sum"),
            f"{prefix}_rec_yards_fb":("rec_yards","sum"),
            f"{prefix}_rush_yards_fb":("rush_yards","sum"),
        }).reset_index())
        out=out.merge(n,on=["team","name_key"],how="left",validate="many_to_one")
        for c in ("games","targets","rushes","rec_yards","rush_yards"):
            base=f"{prefix}_{c}"; fb=f"{base}_fb"
            out.loc[out[base].isna(),base]=out.loc[out[base].isna(),fb]
        out.drop(columns=[c for c in out.columns if c.endswith("_fb")],inplace=True)
    return out

def player_recent_state(roster,current):
    x=current.copy()
    x["gsis_id"]=x["player_id"].map(_clean_id)
    x["name_key"]=x["player"].map(_name_key)
    x["team"]=x["team"].map(_team)
    rows=[]
    for r in roster.itertuples(index=False):
        g=str(getattr(r,"gsis_id","") or ""); team=str(r.team); nk=str(r.name_key)
        q=x.loc[x["gsis_id"].eq(g)].copy() if g else pd.DataFrame()
        if q.empty:q=x.loc[x["team"].eq(team)&x["name_key"].eq(nk)].copy()
        q=q.sort_values("week"); same=q.loc[q["team"].eq(team)].copy()
        last1=same.tail(1); last3=same.tail(3)
        rows.append({
            "state_key":stable_key(pd.Series(r._asdict())),
            "current_games":int(q["week"].nunique()) if len(q) else 0,
            "same_team_current_games":int(same["week"].nunique()) if len(same) else 0,
            "last1_rushes":float(last1["rushes"].sum()) if len(last1) else np.nan,
            "last3_rushes":float(last3["rushes"].sum()) if len(last3) else np.nan,
            "last1_targets":float(last1["targets"].sum()) if len(last1) else np.nan,
            "last3_targets":float(last3["targets"].sum()) if len(last3) else np.nan,
            "last3_source_max_week":int(last3["week"].max()) if len(last3) else np.nan,
        })
    return pd.DataFrame(rows)

def snap_state(roster,snaps):
    rows=[]
    for r in roster.itertuples(index=False):
        pfr=str(getattr(r,"pfr_id","") or "");team=str(r.team);nk=str(r.name_key)
        q=snaps.loc[snaps["pfr_id"].eq(pfr)].copy() if pfr else pd.DataFrame()
        if q.empty:q=snaps.loc[snaps["team"].eq(team)&snaps["name_key"].eq(nk)].copy()
        q=q.sort_values("week");same=q.loc[q["team"].eq(team)].copy();l3=same.tail(3)
        rows.append({
            "state_key":stable_key(pd.Series(r._asdict())),
            "snap_games":int(same["week"].nunique()) if len(same) else 0,
            "latest_offense_pct":float(same.iloc[-1]["offense_pct"]) if len(same) and pd.notna(same.iloc[-1]["offense_pct"]) else np.nan,
            "last3_offense_pct_mean":float(_num(l3["offense_pct"]).mean()) if len(l3) else np.nan,
            "snap_source_max_week":int(same["week"].max()) if len(same) else np.nan,
        })
    return pd.DataFrame(rows)

def add_room_state(df,current):
    out=df.copy();cur=current.copy()
    cur["gsis_id"]=cur["player_id"].map(_clean_id);cur["team"]=cur["team"].map(_team)
    room_rows=[]
    for (tm,pos),g in out.groupby(["team","position_group"],sort=True):
        if pos=="RB":
            q=cur.loc[cur["team"].eq(tm)&cur["position"].astype(str).str.upper().isin(RB_RAW)].copy();opp_col="rushes"
        elif pos in {"WR","TE"}:
            q=cur.loc[cur["team"].eq(tm)&cur["position"].astype(str).str.upper().map(_pos).eq(pos)].copy();opp_col="targets"
        else:
            q=pd.DataFrame();opp_col="targets"
        weeks=sorted(q["week"].dropna().astype(int).unique().tolist())[-3:] if len(q) else []
        q=q.loc[q["week"].isin(weeks)].copy() if weeks else q.iloc[0:0]
        by=q.groupby("player_id",dropna=False)[opp_col].sum() if len(q) else pd.Series(dtype=float)
        total=float(by.sum()) if len(by) else 0.0
        shares={_clean_id(k):(float(v)/total if total>0 else np.nan) for k,v in by.items()}
        finite=np.array([v for v in shares.values() if np.isfinite(v)],dtype=float)
        hhi=float(np.square(finite).sum()) if len(finite) else np.nan
        for idx,r in g.iterrows():
            room_rows.append({"_idx":idx,"room_size":int(len(g)),
                "last3_room_opportunity_share":shares.get(str(r["gsis_id"]),np.nan),
                "room_opportunity_hhi":hhi})
    rr=pd.DataFrame(room_rows).set_index("_idx") if room_rows else pd.DataFrame()
    for col in ["room_size","last3_room_opportunity_share","room_opportunity_hhi"]:
        out[col]=rr[col] if col in rr else np.nan
    out["last3_room_snap_fraction"]=np.nan;out["latest_snap_rank"]=np.nan
    for (_, _),idx in out.groupby(["team","position_group"]).groups.items():
        vals=_num(out.loc[idx,"last3_offense_pct_mean"]);den=float(vals.fillna(0).sum())
        if den>0:out.loc[idx,"last3_room_snap_fraction"]=vals/den
        out.loc[idx,"latest_snap_rank"]=_num(out.loc[idx,"latest_offense_pct"]).rank(method="min",ascending=False)
    return out

def consumption_label(pos,target_week):
    return {
        "QB":"QB_M89_M90_INDIVIDUAL_HISTORY_CONSUMED_ENV_ADDITIVE",
        "WR":"WR_M38_WR_R15_ENTITLEMENT_CONSUMED_EFFICIENCY_RESPONSE_SHARED",
        "TE":"TE_R5P_ENTITLEMENT_CONSUMED_EFFICIENCY_RESPONSE_SHARED",
        "RB":f"RB_PLAYERFORM_HISTORY_CONSUMED_NO_WEEK{target_week}_ROOM_SPECIALIST",
    }.get(pos,"")

def audit(season:int,target_week:int):
    prior_season=season-1
    schedule_all=_load_schedule_all(season)
    sched=target_schedule(schedule_all,target_week)
    teams=set(sched["team"])
    prior_required=prior_week_scheduled_target_teams(schedule_all,teams,target_week)

    roster=load_rosters(season,target_week)
    roster=roster.loc[roster["team"].isin(teams)].copy()
    exact_target_week_roster=target_week_roster_source_ready(roster,target_week)
    roster=roster.merge(sched,on="team",how="left",validate="many_to_one")
    roster["state_key"]=roster.apply(stable_key,axis=1)
    roster["stable_id"]=roster["gsis_id"].ne("")|roster["pfr_id"].ne("")

    prior=load_prior_stats(prior_season)
    current=load_player_stats(season,target_week)
    snaps=load_snaps(season,target_week)
    completion=source_complete_for_prior_week(current,snaps,prior_required,target_week)
    if not completion["complete"]:
        raise RuntimeError(
            "prior-week state incomplete for target-week teams: "
            + json.dumps(completion,sort_keys=True)
        )
    injuries=load_injuries(season,target_week)

    out=join_stats(roster,prior,"prior")
    out=join_stats(out,current,"current_total")
    out=out.merge(player_recent_state(roster,current),on="state_key",how="left",validate="one_to_one")
    out=out.merge(snap_state(roster,snaps),on="state_key",how="left",validate="one_to_one")
    out=out.merge(injuries,on=["team","name_key"],how="left",validate="many_to_one")
    out["injury_status"]=out["injury_status"].fillna("")
    out["practice_status"]=out["practice_status"].fillna("")
    out=add_room_state(out,current)

    out["out_doubtful"]=out["injury_status"].str.contains(r"\bOUT\b|DOUBT",case=False,regex=True)
    out["same_room_unavailable_teammates"]=0
    for (_, _),idx in out.groupby(["team","position_group"]).groups.items():
        unavailable=int(out.loc[idx,"out_doubtful"].sum())
        for i in idx:
            out.at[i,"same_room_unavailable_teammates"]=max(0,unavailable-int(bool(out.at[i,"out_doubtful"])))
    out["production_consumption"]=out["position_group"].map(lambda p:consumption_label(p,target_week))
    out["chronology_valid"]=(
        _num(out["last3_source_max_week"]).fillna(0).lt(target_week)
        & _num(out["snap_source_max_week"]).fillna(0).lt(target_week)
    )

    played=_num(out["current_games"]).fillna(0).gt(0)
    skill_snap=played&out["position_group"].isin({"RB","WR","TE"})
    stable_cov=float(out["stable_id"].mean()) if len(out) else 0.0
    snap_cov=float(out.loc[skill_snap,"latest_offense_pct"].notna().mean()) if skill_snap.any() else 0.0
    usage_cov=float(out.loc[played,["last1_rushes","last1_targets"]].notna().any(axis=1).mean()) if played.any() else 0.0
    chrono=int((~out["chronology_valid"]).sum())

    rb=out.loc[out["position_group"].eq("RB")].copy()
    rb_state=rb.loc[rb["last3_room_opportunity_share"].notna()&rb["last3_room_snap_fraction"].notna()].copy()
    multi=0;distinguish=0;room_rows=[]
    for tm,g in rb.groupby("team"):
        state=g.loc[g["last3_room_opportunity_share"].notna()|g["last3_room_snap_fraction"].notna()].copy()
        if len(state)>=2:
            multi+=1
            od=_num(state["last3_room_opportunity_share"]);sd=_num(state["last3_room_snap_fraction"])
            opp_gap=float(od.max()-od.min()) if od.notna().sum()>=2 else np.nan
            snap_gap=float(sd.max()-sd.min()) if sd.notna().sum()>=2 else np.nan
            d=bool((np.isfinite(opp_gap) and opp_gap>=.20) or (np.isfinite(snap_gap) and snap_gap>=.20))
            distinguish+=int(d)
            room_rows.append({"team":tm,"rb_count":int(len(g)),"state_count":int(len(state)),
                "opportunity_share_gap":opp_gap,"snap_fraction_gap":snap_gap,"distinguishable_state":d})
    rooms=pd.DataFrame(room_rows)

    pos=(out.groupby("position_group").agg(
        players=("state_key","size"),stable_id_coverage=("stable_id","mean"),
        current_game_players=("current_games",lambda x:int((_num(x)>0).sum())),
        latest_snap_coverage=("latest_offense_pct",lambda x:float(x.notna().mean())),
        last3_snap_coverage=("last3_offense_pct_mean",lambda x:float(x.notna().mean())),
        injury_rows=("injury_status",lambda x:int(x.astype(str).ne("").sum())),
    ).reset_index())

    ready=bool(
        len(sched)>=20 and exact_target_week_roster
        and stable_cov>=.90 and snap_cov>=.90 and usage_cov>=.90
        and chrono==0 and len(rb_state)>=20 and multi>=10 and distinguish>=10
        and completion["complete"]
    )
    source_safe=bool(
        len(sched)>=20 and exact_target_week_roster
        and stable_cov>=.80 and chrono==0 and completion["complete"]
    )
    disposition=("PLAYER_STATE_FUTURE_WEEK_READY" if ready else
                 "PLAYER_STATE_FUTURE_WEEK_PARTIAL" if source_safe else
                 "PLAYER_STATE_FUTURE_WEEK_NOT_READY")
    result={
        "version":VERSION,"captured_at_utc":datetime.now(timezone.utc).isoformat(),
        "season":season,"target_week":target_week,"prior_season":prior_season,
        "disposition":disposition,"scheduled_teams":int(len(sched)),
        "roster_source_week":int(out["roster_source_week"].max()) if len(out) else None,
        "target_week_roster_source_ready":bool(exact_target_week_roster),
        "skill_players":int(len(out)),"stable_id_coverage":stable_cov,
        "players_with_current_games":int(played.sum()),
        "rb_wr_te_latest_snap_coverage_among_played":snap_cov,
        "current_usage_coverage_among_played":usage_cov,
        "chronology_violations":chrono,"rb_players":int(len(rb)),
        "rb_players_with_room_opportunity_and_snap_state":int(len(rb_state)),
        "multi_back_rooms_with_state":int(multi),
        "multi_back_rooms_with_distinguishable_state":int(distinguish),
        "target_week_injury_rows_available":int(out["injury_status"].astype(str).ne("").sum()),
        "prior_week_source_completion":completion,
        "sportsbook_inputs_used":0,"target_week_outcomes_read":0,
        "candidate_models_fit":0,"production_changed":False,
    }
    return out,pos,rooms,result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--season",type=int,required=True)
    ap.add_argument("--target-week",type=int,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    rows,pos,rooms,result=audit(a.season,a.target_week)
    prefix=f"player_state_future_week{a.target_week}"
    rows.to_csv(a.out_dir/f"{prefix}_rows.csv",index=False)
    pos.to_csv(a.out_dir/f"{prefix}_position_coverage.csv",index=False)
    rooms.to_csv(a.out_dir/f"{prefix}_rb_rooms.csv",index=False)
    (a.out_dir/f"{prefix}_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
