#!/usr/bin/env python3
"""Prospective Week-5 player-state coverage audit.

Reads only free football sources. Fails closed if target-week stats/snaps already
exist. No sportsbook inputs, no model fitting, no production mutation.
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

VERSION="PLAYER_STATE_LIVE_COVERAGE_V1"
SEASON=2026
PRIOR_SEASON=2025
TARGET_WEEK=5
SKILL={"QB","RB","WR","TE"}
RB_RAW={"RB","HB","FB"}
FORBIDDEN=(
    "sportsbook","bookmaker","prop_line","market_line","over_odds","under_odds",
    "spread_line","total_line","moneyline","closing_line","no_vig","implied_prob",
)

def _pd(x):
    return _to_pandas(x)

def _lower(x:pd.DataFrame)->pd.DataFrame:
    y=_pd(x).copy()
    y.columns=[str(c).strip().lower() for c in y.columns]
    return y

def _first(df,names,default=""):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default,index=df.index)

def _num(x):
    return pd.to_numeric(x,errors="coerce")

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

def load_schedule():
    import nflreadpy as nfl
    try: raw=nfl.load_schedules(seasons=[SEASON])
    except TypeError: raw=nfl.load_schedules(SEASON)
    x=_lower(raw)
    if "game_type" in x.columns:
        x=x.loc[x["game_type"].astype(str).str.upper().eq("REG")].copy()
    elif "season_type" in x.columns:
        x=x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
    x=x.loc[_num(x["season"]).eq(SEASON)&_num(x["week"]).eq(TARGET_WEEK)].copy()
    hc="home_team" if "home_team" in x.columns else "home"
    ac="away_team" if "away_team" in x.columns else "away"
    rows=[]
    for r in x.itertuples(index=False):
        h=_team(getattr(r,hc)); a=_team(getattr(r,ac))
        if h and a:
            rows += [{"team":h,"opponent":a},{"team":a,"opponent":h}]
    out=pd.DataFrame(rows).drop_duplicates("team")
    if len(out)<20: raise RuntimeError(f"Week-5 schedule too small: {len(out)} teams")
    return out

def load_rosters():
    import nflreadpy as nfl
    raw=_lower(nfl.load_rosters_weekly(SEASON))
    out=pd.DataFrame(index=raw.index)
    out["season"]=_num(_first(raw,["season"],SEASON)).fillna(SEASON)
    out["week"]=_num(_first(raw,["week"]))
    out["team"]=_first(raw,["team","team_abbr","club_code"]).map(_team)
    out["player"]=_first(raw,["full_name","football_name","player_name","player","name"]).astype("string").fillna("").str.strip()
    out["name_key"]=out["player"].map(_name_key)
    out["position_raw"]=_first(raw,["position","pos","depth_chart_position","ngs_position"]).astype("string").fillna("").str.upper().str.strip()
    out["position_group"]=out["position_raw"].map(_pos)
    out["gsis_id"]=_first(raw,["gsis_id","player_id"]).map(_clean_id)
    out["pfr_id"]=_first(raw,["pfr_id","pfr_player_id"]).map(_clean_id)
    out["status"]=_first(raw,["status","roster_status"]).astype("string").fillna("").str.upper().str.strip()
    out=out.loc[out["season"].eq(SEASON)&out["position_group"].isin(SKILL)&out["team"].ne("")&out["name_key"].ne("")].copy()
    # Prefer exact target-week roster. If provider has not published W5 roster,
    # use the latest roster week strictly before W5 and stamp that fact.
    exact=out.loc[out["week"].eq(TARGET_WEEK)].copy()
    if len(exact):
        use=exact
        source_week=TARGET_WEEK
    else:
        prior=out.loc[out["week"].lt(TARGET_WEEK)].copy()
        if prior.empty: raise RuntimeError("no Week-5 or prior weekly roster")
        source_week=int(prior["week"].max())
        use=prior.loc[prior["week"].eq(source_week)].copy()
    use["roster_source_week"]=source_week
    use=use.sort_values(["team","gsis_id","pfr_id","name_key"]).drop_duplicates(["team","name_key"],keep="last")
    return use

def load_player_stats(season):
    import nflreadpy as nfl
    x=_normalize_weekly(_pd(nfl.load_player_stats(seasons=[int(season)],summary_level="week")),int(season))
    _check_forbidden(f"stats_{season}",x)
    if season==SEASON:
        seen=sorted(_num(x["week"]).dropna().astype(int).unique().tolist())
        bad=[w for w in seen if w>=TARGET_WEEK]
        if bad:
            raise RuntimeError(f"target/future 2026 stat rows already present; refuse audit: weeks={bad}")
        x=x.loc[_num(x["week"]).lt(TARGET_WEEK)].copy()
    return x

def load_snaps():
    import nflreadpy as nfl
    raw=_lower(nfl.load_snap_counts(seasons=[SEASON]))
    out=pd.DataFrame(index=raw.index)
    out["season"]=_num(_first(raw,["season"],SEASON)).fillna(SEASON)
    out["week"]=_num(_first(raw,["week"]))
    out["team"]=_first(raw,["team","team_abbr"]).map(_team)
    out["player"]=_first(raw,["player","player_name","full_name"]).astype("string").fillna("").str.strip()
    out["name_key"]=out["player"].map(_name_key)
    out["pfr_id"]=_first(raw,["pfr_player_id","pfr_id"]).map(_clean_id)
    out["position_raw"]=_first(raw,["position","pos"]).astype("string").fillna("").str.upper().str.strip()
    out["position_group"]=out["position_raw"].map(_pos)
    out["offense_snaps"]=_num(_first(raw,["offense_snaps","off_snaps"],np.nan))
    out["offense_pct"]=_num(_first(raw,["offense_pct","off_pct","offense_percentage"],np.nan))
    bad=sorted(out.loc[out["week"].ge(TARGET_WEEK),"week"].dropna().astype(int).unique().tolist())
    if bad:
        raise RuntimeError(f"target/future snap rows already present; refuse audit: weeks={bad}")
    out=out.loc[out["season"].eq(SEASON)&out["week"].lt(TARGET_WEEK)&out["position_group"].isin({"RB","WR","TE","QB"})].copy()
    return out.drop_duplicates(["week","team","pfr_id","name_key"],keep="last")

def load_injuries():
    import nflreadpy as nfl
    try: raw=_lower(nfl.load_injuries(seasons=[SEASON]))
    except Exception:
        return pd.DataFrame(columns=["team","name_key","injury_status","practice_status"])
    if raw.empty: return pd.DataFrame(columns=["team","name_key","injury_status","practice_status"])
    out=pd.DataFrame(index=raw.index)
    out["week"]=_num(_first(raw,["week","report_week"]))
    out["team"]=_first(raw,["team","team_abbr","team_abbreviation","club"]).map(_team)
    out["name_key"]=_first(raw,["full_name","player_name","player","name"]).map(_name_key)
    out["injury_status"]=_first(raw,["report_status","game_status","status"]).astype("string").fillna("").str.upper().str.strip()
    out["practice_status"]=_first(raw,["practice_status","practice_participation"]).astype("string").fillna("").str.upper().str.strip()
    out=out.loc[out["week"].eq(TARGET_WEEK)&out["team"].ne("")&out["name_key"].ne("")].copy()
    return out[["team","name_key","injury_status","practice_status"]].drop_duplicates(["team","name_key"],keep="last")

def stable_key(row):
    g=_clean_id(row.get("gsis_id",""))
    if g: return "gsis:"+g
    p=_clean_id(row.get("pfr_id",""))
    if p: return "pfr:"+p
    return "name:"+str(row.get("team",""))+":"+str(row.get("name_key",""))

def join_stats(roster,stats,season_prefix):
    s=stats.copy()
    s["gsis_id"]=s["player_id"].map(_clean_id)
    s["name_key"]=s["player"].map(_name_key)
    s["team"]=s["team"].map(_team)
    # Aggregate same player season while preserving target-team continuity separately.
    agg=(s.groupby("gsis_id",dropna=False)
         .agg(**{
           f"{season_prefix}_games":("week","nunique"),
           f"{season_prefix}_targets":("targets","sum"),
           f"{season_prefix}_rushes":("rushes","sum"),
           f"{season_prefix}_rec_yards":("rec_yards","sum"),
           f"{season_prefix}_rush_yards":("rush_yards","sum"),
           f"{season_prefix}_last_team":("team","last"),
         }).reset_index())
    # Blank GSIS rows cannot be many-to-one authority.
    agg=agg.loc[agg["gsis_id"].ne("")].copy()
    out=roster.merge(agg,on="gsis_id",how="left",validate="many_to_one")
    # Name/team fallback only for unmatched.
    missing=out[f"{season_prefix}_games"].isna()
    if missing.any():
        n=(s.groupby(["team","name_key"],dropna=False)
           .agg(**{
             f"{season_prefix}_games_fb":("week","nunique"),
             f"{season_prefix}_targets_fb":("targets","sum"),
             f"{season_prefix}_rushes_fb":("rushes","sum"),
             f"{season_prefix}_rec_yards_fb":("rec_yards","sum"),
             f"{season_prefix}_rush_yards_fb":("rush_yards","sum"),
           }).reset_index())
        out=out.merge(n,on=["team","name_key"],how="left",validate="many_to_one")
        for c in ("games","targets","rushes","rec_yards","rush_yards"):
            base=f"{season_prefix}_{c}"; fb=f"{base}_fb"
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
        g=str(getattr(r,"gsis_id","") or "")
        team=str(r.team); nk=str(r.name_key)
        q=x.loc[x["gsis_id"].eq(g)].copy() if g else pd.DataFrame()
        if q.empty:
            q=x.loc[x["team"].eq(team)&x["name_key"].eq(nk)].copy()
        q=q.sort_values("week")
        same=q.loc[q["team"].eq(team)].copy()
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
        pfr=str(getattr(r,"pfr_id","") or ""); team=str(r.team); nk=str(r.name_key)
        q=snaps.loc[snaps["pfr_id"].eq(pfr)].copy() if pfr else pd.DataFrame()
        if q.empty: q=snaps.loc[snaps["team"].eq(team)&snaps["name_key"].eq(nk)].copy()
        q=q.sort_values("week")
        same=q.loc[q["team"].eq(team)].copy()
        l3=same.tail(3)
        rows.append({
          "state_key":stable_key(pd.Series(r._asdict())),
          "snap_games":int(same["week"].nunique()) if len(same) else 0,
          "latest_offense_pct":float(same.iloc[-1]["offense_pct"]) if len(same) and pd.notna(same.iloc[-1]["offense_pct"]) else np.nan,
          "last3_offense_pct_mean":float(_num(l3["offense_pct"]).mean()) if len(l3) else np.nan,
          "snap_source_max_week":int(same["week"].max()) if len(same) else np.nan,
        })
    return pd.DataFrame(rows)

def add_room_state(df,current):
    out=df.copy()
    cur=current.copy()
    cur["gsis_id"]=cur["player_id"].map(_clean_id); cur["team"]=cur["team"].map(_team)
    # Most recent three observed team weeks.
    room_rows=[]
    for (team,pos),g in out.groupby(["team","position_group"],sort=True):
        members=g["gsis_id"].astype(str).tolist()
        if pos=="RB":
            q=cur.loc[cur["team"].eq(team)&cur["position"].astype(str).str.upper().isin(RB_RAW)].copy()
            opp_col="rushes"
        elif pos in {"WR","TE"}:
            q=cur.loc[cur["team"].eq(team)&cur["position"].astype(str).str.upper().map(_pos).eq(pos)].copy()
            opp_col="targets"
        else:
            q=pd.DataFrame(); opp_col="targets"
        weeks=sorted(q["week"].dropna().astype(int).unique().tolist())[-3:] if len(q) else []
        q=q.loc[q["week"].isin(weeks)].copy() if weeks else q.iloc[0:0]
        by=q.groupby("player_id",dropna=False)[opp_col].sum() if len(q) else pd.Series(dtype=float)
        total=float(by.sum()) if len(by) else 0.0
        shares={_clean_id(k):(float(v)/total if total>0 else np.nan) for k,v in by.items()}
        finite=np.array([v for v in shares.values() if np.isfinite(v)],dtype=float)
        hhi=float(np.square(finite).sum()) if len(finite) else np.nan
        for idx,r in g.iterrows():
            room_rows.append({
              "_idx":idx,
              "room_size":int(len(g)),
              "last3_room_opportunity_share":shares.get(str(r["gsis_id"]),np.nan),
              "room_opportunity_hhi":hhi,
            })
    rr=pd.DataFrame(room_rows).set_index("_idx") if room_rows else pd.DataFrame()
    for c in ["room_size","last3_room_opportunity_share","room_opportunity_hhi"]:
        out[c]=rr[c] if c in rr else np.nan

    # Relative snap fraction and latest snap rank are derived from already joined state.
    out["last3_room_snap_fraction"]=np.nan
    out["latest_snap_rank"]=np.nan
    for (team,pos),idx in out.groupby(["team","position_group"]).groups.items():
        vals=_num(out.loc[idx,"last3_offense_pct_mean"])
        den=float(vals.fillna(0).sum())
        if den>0: out.loc[idx,"last3_room_snap_fraction"]=vals/den
        ranks=_num(out.loc[idx,"latest_offense_pct"]).rank(method="min",ascending=False)
        out.loc[idx,"latest_snap_rank"]=ranks
    return out

def consumption_label(pos):
    return {
      "QB":"QB_M89_M90_INDIVIDUAL_HISTORY_CONSUMED_ENV_ADDITIVE",
      "WR":"WR_M38_WR_R15_ENTITLEMENT_CONSUMED_EFFICIENCY_RESPONSE_SHARED",
      "TE":"TE_R5P_ENTITLEMENT_CONSUMED_EFFICIENCY_RESPONSE_SHARED",
      "RB":"RB_PLAYERFORM_HISTORY_CONSUMED_NO_WEEK5_ROOM_SPECIALIST",
    }.get(pos,"")

def audit():
    sched=load_schedule(); teams=set(sched["team"])
    roster=load_rosters()
    roster=roster.loc[roster["team"].isin(teams)].copy()
    roster=roster.merge(sched,on="team",how="left",validate="many_to_one")
    roster["state_key"]=roster.apply(stable_key,axis=1)
    roster["stable_id"]=roster["gsis_id"].ne("")|roster["pfr_id"].ne("")

    prior=load_player_stats(PRIOR_SEASON)
    current=load_player_stats(SEASON)
    snaps=load_snaps()
    injuries=load_injuries()

    out=join_stats(roster,prior,"prior")
    out=join_stats(out,current,"current_total")
    out=out.merge(player_recent_state(roster,current),on="state_key",how="left",validate="one_to_one")
    out=out.merge(snap_state(roster,snaps),on="state_key",how="left",validate="one_to_one")
    out=out.merge(injuries,on=["team","name_key"],how="left",validate="many_to_one")
    out["injury_status"]=out["injury_status"].fillna("")
    out["practice_status"]=out["practice_status"].fillna("")
    out=add_room_state(out,current)

    # Availability context.
    out["out_doubtful"]=out["injury_status"].str.contains(r"\bOUT\b|DOUBT",case=False,regex=True)
    out["same_room_unavailable_teammates"]=0
    for (team,pos),idx in out.groupby(["team","position_group"]).groups.items():
        unavailable=int(out.loc[idx,"out_doubtful"].sum())
        for i in idx:
            out.at[i,"same_room_unavailable_teammates"]=max(0,unavailable-int(bool(out.at[i,"out_doubtful"])))

    out["production_consumption"]=out["position_group"].map(consumption_label)
    out["chronology_valid"]=(
        _num(out["last3_source_max_week"]).fillna(0).lt(TARGET_WEEK)
        & _num(out["snap_source_max_week"]).fillna(0).lt(TARGET_WEEK)
    )

    played=_num(out["current_games"]).fillna(0).gt(0)
    skill_snap=played & out["position_group"].isin({"RB","WR","TE"})
    stable_cov=float(out["stable_id"].mean()) if len(out) else 0.0
    snap_cov=float(out.loc[skill_snap,"latest_offense_pct"].notna().mean()) if skill_snap.any() else 0.0
    usage_cov=float(out.loc[played,["last1_rushes","last1_targets"]].notna().any(axis=1).mean()) if played.any() else 0.0
    chrono=int((~out["chronology_valid"]).sum())

    rb=out.loc[out["position_group"].eq("RB")].copy()
    rb_state=rb.loc[rb["last3_room_opportunity_share"].notna()&rb["last3_room_snap_fraction"].notna()].copy()
    multi=0; distinguish=0
    room_rows=[]
    for team,g in rb.groupby("team"):
        state=g.loc[g["last3_room_opportunity_share"].notna()|g["last3_room_snap_fraction"].notna()].copy()
        if len(state)>=2:
            multi += 1
            od=_num(state["last3_room_opportunity_share"])
            sd=_num(state["last3_room_snap_fraction"])
            opp_gap=float(od.max()-od.min()) if od.notna().sum()>=2 else np.nan
            snap_gap=float(sd.max()-sd.min()) if sd.notna().sum()>=2 else np.nan
            d=bool((np.isfinite(opp_gap) and opp_gap>=.20) or (np.isfinite(snap_gap) and snap_gap>=.20))
            distinguish += int(d)
            room_rows.append({"team":team,"rb_count":int(len(g)),"state_count":int(len(state)),
                              "opportunity_share_gap":opp_gap,"snap_fraction_gap":snap_gap,
                              "distinguishable_state":d})
    room_summary=pd.DataFrame(room_rows)

    pos_summary=(out.groupby("position_group")
      .agg(players=("state_key","size"),stable_id_coverage=("stable_id","mean"),
           current_game_players=("current_games",lambda x:int((_num(x)>0).sum())),
           latest_snap_coverage=("latest_offense_pct",lambda x:float(x.notna().mean())),
           last3_snap_coverage=("last3_offense_pct_mean",lambda x:float(x.notna().mean())),
           injury_rows=("injury_status",lambda x:int(x.astype(str).ne("").sum())))
      .reset_index())

    ready=bool(
      len(sched)>=20 and stable_cov>=.90 and snap_cov>=.90 and usage_cov>=.90
      and chrono==0 and len(rb_state)>=20 and multi>=10 and distinguish>=10
    )
    source_safe=bool(len(sched)>=20 and stable_cov>=.80 and chrono==0)
    disposition=("PLAYER_STATE_LIVE_READY_FOR_PROSPECTIVE_SHADOW" if ready else
                 "PLAYER_STATE_LIVE_PARTIAL" if source_safe else
                 "PLAYER_STATE_LIVE_NOT_READY")
    result={
      "version":VERSION,
      "captured_at_utc":datetime.now(timezone.utc).isoformat(),
      "season":SEASON,"target_week":TARGET_WEEK,
      "disposition":disposition,
      "scheduled_teams":int(len(sched)),
      "roster_source_week":int(out["roster_source_week"].max()) if len(out) else None,
      "skill_players":int(len(out)),
      "stable_id_coverage":stable_cov,
      "players_with_current_2026_games":int(played.sum()),
      "rb_wr_te_latest_snap_coverage_among_played":snap_cov,
      "current_usage_coverage_among_played":usage_cov,
      "chronology_violations":chrono,
      "rb_players":int(len(rb)),
      "rb_players_with_room_opportunity_and_snap_state":int(len(rb_state)),
      "multi_back_rooms_with_state":int(multi),
      "multi_back_rooms_with_distinguishable_state":int(distinguish),
      "week5_injury_rows_available":int(out["injury_status"].astype(str).ne("").sum()),
      "sportsbook_inputs_used":0,
      "week5_outcomes_read":0,
      "candidate_models_fit":0,
      "production_changed":False,
    }
    return out,pos_summary,room_summary,result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    rows,pos,rooms,result=audit()
    rows.to_csv(a.out_dir/"player_state_live_week5_rows.csv",index=False)
    pos.to_csv(a.out_dir/"player_state_live_position_coverage.csv",index=False)
    rooms.to_csv(a.out_dir/"player_state_live_rb_rooms.csv",index=False)
    (a.out_dir/"player_state_live_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    print("\nPOSITION COVERAGE")
    print(pos.to_string(index=False))
    print("\nRB ROOMS")
    print(rooms.to_string(index=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
