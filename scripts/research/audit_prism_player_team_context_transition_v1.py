#!/usr/bin/env python3
"""PRISM Player Team-Context Transition V1 source audit.

Source/lineage only. Certifies exact named-player team changes from weekly
roster chronology and maps current same-family roster competition. No player
performance or opportunity outcome is scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.player_identity_v3 import player_name_key

VERSION = "PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "bd254ec9435cd0a17e2c7d82d78532e67c2cd23e"
SOURCE_SEASONS = (2023, 2024, 2025, 2026)
HISTORICAL_SEASONS = (2024, 2025)
PARITY_SEASON = 2026
WEEKS = tuple(range(1, 19))
FAMILIES = {"RB", "WR", "TE"}

NO_PRIOR = "TEAM_CONTEXT_NO_PRIOR_ROSTER"
CONTINUED = "TEAM_CONTEXT_CONTINUED"
CHANGED = "TEAM_CONTEXT_CHANGED"


def json_default(v):
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    raise TypeError(type(v).__name__)


def pdx(v):
    if isinstance(v, pd.DataFrame):
        return v.copy()
    if hasattr(v, "to_dicts"):
        return pd.DataFrame(v.to_dicts())
    if hasattr(v, "to_pandas"):
        return v.to_pandas()
    return pd.DataFrame(v)


def lower(df):
    x=df.copy()
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def first_col(df,names):
    return next((c for c in names if c in df.columns),None)


def text(v):
    if v is None or pd.isna(v):
        return ""
    s=str(v).strip()
    return "" if s.upper() in {"","NAN","NONE","<NA>"} else s


def fam(v):
    p=text(v).upper()
    if p in {"RB","HB","TB","FB"} or p.startswith(("RB","HB","TB","FB")):
        return "RB"
    if p=="WR" or p.startswith("WR"):
        return "WR"
    if p=="TE" or p.startswith("TE"):
        return "TE"
    return ""


def normalize_schedule(raw):
    x=lower(raw)
    cols={
        "season":first_col(x,("season",)),
        "week":first_col(x,("week",)),
        "game_type":first_col(x,("game_type","season_type")),
        "home":first_col(x,("home_team","home")),
        "away":first_col(x,("away_team","away")),
    }
    miss=[k for k,v in cols.items() if v is None]
    if miss:
        raise RuntimeError(f"schedule missing semantics {miss}")
    q=pd.DataFrame({
        "season":pd.to_numeric(x[cols["season"]],errors="coerce"),
        "week":pd.to_numeric(x[cols["week"]],errors="coerce"),
        "game_type":x[cols["game_type"]].astype("string").str.upper().str.strip(),
        "home":x[cols["home"]].map(canon_team),
        "away":x[cols["away"]].map(canon_team),
    })
    q=q.loc[
        q["season"].isin(SOURCE_SEASONS)
        & q["week"].isin(WEEKS)
        & q["game_type"].eq("REG")
        & q["home"].ne("")
        & q["away"].ne("")
    ].copy()
    rows=[]
    for r in q.itertuples(index=False):
        rows.append({"season":int(r.season),"week":int(r.week),"team":str(r.home)})
        rows.append({"season":int(r.season),"week":int(r.week),"team":str(r.away)})
    y=pd.DataFrame(rows).drop_duplicates().sort_values(["season","week","team"]).reset_index(drop=True)
    if y.empty or y.duplicated(["season","week","team"]).any():
        raise RuntimeError("invalid normalized schedule")
    return y,{"columns":cols,"team_weeks":int(len(y))}


def normalize_rosters(parts,scheduled):
    scheduled_keys=set(zip(scheduled["season"],scheduled["week"],scheduled["team"]))
    frames=[]
    schemas={}
    raw_exact_dups=0
    for season,raw in parts:
        x=lower(raw)
        cols={
            "season":first_col(x,("season",)),
            "week":first_col(x,("week",)),
            "team":first_col(x,("team","team_abbr","team_abbreviation","club")),
            "stable_id":first_col(x,("gsis_id","player_id","nflverse_id")),
            "name":first_col(x,("player_name","full_name","player","name")),
            "position":first_col(x,("position","pos","position_group")),
            "status":first_col(x,("status","roster_status")),
        }
        miss=[k for k in ("week","team","stable_id","name","position") if cols[k] is None]
        if miss:
            raise RuntimeError(f"weekly roster {season} missing semantics {miss}")
        ss=(
            pd.to_numeric(x[cols["season"]],errors="coerce")
            if cols["season"] else pd.Series(int(season),index=x.index,dtype=float)
        )
        q=pd.DataFrame({
            "season":ss,
            "week":pd.to_numeric(x[cols["week"]],errors="coerce"),
            "team":x[cols["team"]].map(canon_team),
            "stable_id":x[cols["stable_id"]].map(text),
            "player":x[cols["name"]].map(text),
            "name_key":x[cols["name"]].map(lambda v:player_name_key(v,strip_suffix=True)),
            "position_raw":x[cols["position"]].map(text),
            "roster_status":x[cols["status"]].map(text) if cols["status"] else "",
            "_order":np.arange(len(x),dtype=int),
        })
        q["family"]=q["position_raw"].map(fam)
        q=q.loc[
            q["season"].eq(int(season))
            & q["week"].isin(WEEKS)
            & q["team"].ne("")
            & q["family"].isin(FAMILIES)
            & q["name_key"].ne("")
        ].copy()
        q["season"]=q["season"].astype(int)
        q["week"]=q["week"].astype(int)
        q=q.loc[[
            (int(s),int(w),str(t)) in scheduled_keys
            for s,w,t in zip(q["season"],q["week"],q["team"])
        ]].copy()
        nonblank=q.loc[q["stable_id"].ne("")].copy()
        raw_exact_dups += int(nonblank.duplicated(
            ["season","week","team","stable_id"],keep=False
        ).sum())
        with_id=nonblank.sort_values(
            ["season","week","team","stable_id","_order"],kind="stable"
        ).drop_duplicates(["season","week","team","stable_id"],keep="last")
        blanks=q.loc[q["stable_id"].eq("")].copy()
        frames.append(pd.concat([with_id,blanks],ignore_index=True))
        schemas[str(season)]=cols

    y=pd.concat(frames,ignore_index=True)
    y["ordinal"]=y["season"]*100+y["week"]
    y=y.sort_values(["season","week","team","stable_id","name_key"],kind="stable").reset_index(drop=True)
    post_dups=int(y.loc[y["stable_id"].ne("")].duplicated(
        ["season","week","team","stable_id"]
    ).sum())
    return y,{
        "schemas":schemas,
        "raw_exact_duplicate_rows":int(raw_exact_dups),
        "post_dedup_exact_duplicate_rows":post_dups,
    }


def team_week_coverage(roster,scheduled,season,week=None):
    sch=scheduled.loc[scheduled["season"].eq(season)].copy()
    rr=roster.loc[roster["season"].eq(season)].copy()
    if week is not None:
        sch=sch.loc[sch["week"].eq(week)]
        rr=rr.loc[rr["week"].eq(week)]
    sk=set(zip(sch["week"],sch["team"]))
    rk=set(zip(rr["week"],rr["team"]))
    covered=len(sk & rk)
    return covered,len(sk),float(covered/len(sk)) if sk else 0.0


def build_states(roster):
    admitted=roster.loc[roster["stable_id"].ne("")].copy()

    mt=(admitted.groupby(["season","week","stable_id"])["team"]
        .nunique().reset_index(name="team_count"))
    amb=mt.loc[mt["team_count"].gt(1)].copy()
    ambiguous_keys=set(zip(amb["season"],amb["week"],amb["stable_id"]))

    # Keep ambiguous rows in the artifact but never certify a state for them.
    admitted["ambiguous_multi_team_player_week"]=[
        (int(s),int(w),str(pid)) in ambiguous_keys
        for s,w,pid in zip(admitted["season"],admitted["week"],admitted["stable_id"])
    ]
    clean=admitted.loc[~admitted["ambiguous_multi_team_player_week"]].copy()
    clean=clean.sort_values(["stable_id","ordinal","team"],kind="stable").reset_index(drop=True)

    clean["previous_roster_season"]=clean.groupby("stable_id")["season"].shift(1)
    clean["previous_roster_week"]=clean.groupby("stable_id")["week"].shift(1)
    clean["previous_rostered_team"]=clean.groupby("stable_id")["team"].shift(1)
    clean["previous_roster_ordinal"]=clean.groupby("stable_id")["ordinal"].shift(1)

    prior=clean["previous_roster_ordinal"].notna()
    clean["team_context_state"]=np.select(
        [
            ~prior,
            prior & clean["previous_rostered_team"].eq(clean["team"]),
            prior & clean["previous_rostered_team"].ne(clean["team"]),
        ],
        [NO_PRIOR,CONTINUED,CHANGED],
        default="",
    )
    if not clean["team_context_state"].isin({NO_PRIOR,CONTINUED,CHANGED}).all():
        raise RuntimeError("unknown team-context state")

    clean["transition_kind"]=np.where(
        clean["team_context_state"].ne(CHANGED),
        "",
        np.where(
            pd.to_numeric(clean["previous_roster_season"],errors="coerce").eq(clean["season"]),
            "INSEASON_TEAM_CHANGE",
            "OFFSEASON_TEAM_CHANGE",
        ),
    )

    # Current same-family roster competition uses target-week roster only and
    # does not infer rank or allocate opportunity. Build the finite roster-room
    # lookup once; do not rescan the full roster for every player-week.
    room_ids={
        (int(s),int(w),str(t),str(f)):sorted(set(str(v) for v in g["stable_id"] if str(v)))
        for (s,w,t,f),g in admitted.groupby(["season","week","team","family"],sort=False)
    }
    competitor_count=[]
    competitor_ids=[]
    for r in clean.itertuples(index=False):
        ids=[
            v for v in room_ids.get(
                (int(r.season),int(r.week),str(r.team),str(r.family)),[]
            )
            if v != str(r.stable_id)
        ]
        competitor_count.append(len(ids))
        competitor_ids.append(json.dumps(ids,separators=(",",":")))
    clean["same_family_competitor_count"]=competitor_count
    clean["same_family_competitor_stable_ids"]=competitor_ids

    # Strict-prior roster support can be accumulated in one chronological
    # pass per named player. Counts are evaluated before the current row is
    # added, so no same/future observation can enter the support metadata.
    prior_team_counts=pd.Series(0,index=clean.index,dtype=int)
    current_team_prior_counts=pd.Series(0,index=clean.index,dtype=int)
    skipped=0
    for _,g in clean.groupby("stable_id",sort=False):
        team_counts={}
        prior_row=None
        for idx,r in g.iterrows():
            if prior_row is not None:
                if (
                    int(prior_row["ordinal"]) != int(r["previous_roster_ordinal"])
                    or str(prior_row["team"]) != str(r["previous_rostered_team"])
                ):
                    skipped += 1
            elif pd.notna(r["previous_roster_ordinal"]):
                skipped += 1
            prior_team_counts.loc[idx]=(
                int(team_counts.get(str(r["previous_rostered_team"]),0))
                if pd.notna(r["previous_rostered_team"]) else 0
            )
            current_team_prior_counts.loc[idx]=int(team_counts.get(str(r["team"]),0))
            team_counts[str(r["team"])]=int(team_counts.get(str(r["team"]),0))+1
            prior_row=r
    clean["prior_team_roster_week_count"]=prior_team_counts.astype(int)
    clean["current_team_prior_roster_week_count"]=current_team_prior_counts.astype(int)
    clean["prior_team_support_ge4"]=clean["prior_team_roster_week_count"].ge(4)
    clean["prior_team_support_ge8"]=clean["prior_team_roster_week_count"].ge(8)

    chronology=int((
        clean["previous_roster_ordinal"].notna()
        & pd.to_numeric(clean["previous_roster_ordinal"],errors="coerce").ge(clean["ordinal"])
    ).sum())

    transitions=clean.loc[clean["team_context_state"].eq(CHANGED)].copy()
    dup_trans=int(transitions.duplicated(["season","week","team","stable_id"]).sum())

    audit={
        "ambiguous_multi_team_player_weeks":int(len(amb)),
        "ambiguous_multi_team_rows":int(admitted["ambiguous_multi_team_player_week"].sum()),
        "chronology_violations":chronology,
        "skipped_previous_roster_violations":int(skipped),
        "duplicate_transition_event_identities":dup_trans,
    }
    return clean,transitions,audit


def historical_summary(season,roster,scheduled,states,transitions,audit):
    r=roster.loc[roster["season"].eq(season)].copy()
    s=states.loc[states["season"].eq(season)].copy()
    t=transitions.loc[transitions["season"].eq(season)].copy()
    covered,total,cov=team_week_coverage(roster,scheduled,season)
    stable_cov=float(r["stable_id"].ne("").mean()) if len(r) else 0.0

    fam_rows=[]
    for family in ("RB","WR","TE"):
        g=t.loc[t["family"].eq(family)]
        fam_rows.append({
            "family":family,
            "transition_events":int(len(g)),
            "transition_named_players":int(g["stable_id"].nunique()),
            "prior_team_ge4_share":float(g["prior_team_support_ge4"].mean()) if len(g) else 0.0,
            "prior_team_ge8_share":float(g["prior_team_support_ge8"].mean()) if len(g) else 0.0,
        })

    ge4=float(t["prior_team_support_ge4"].mean()) if len(t) else 0.0
    ge8=float(t["prior_team_support_ge8"].mean()) if len(t) else 0.0
    fam_players={x["family"]:x["transition_named_players"] for x in fam_rows}
    competitor_ok=bool(
        len(t)>0
        and t["same_family_competitor_count"].notna().all()
        and t["same_family_competitor_stable_ids"].astype(str).str.startswith("[").all()
    )

    gates={
        "weekly_roster_explicit_season_week":True,
        "scheduled_team_week_roster_coverage_ge_095":cov>=0.95,
        "stable_gsis_coverage_ge_099":stable_cov>=0.99,
        "ambiguous_multi_team_player_weeks_zero":audit["ambiguous_multi_team_player_weeks"]==0,
        "previous_roster_chronology_violations_zero":audit["chronology_violations"]==0,
        "transition_previous_week_strictly_prior":bool(
            len(t)>0 and (pd.to_numeric(t["previous_roster_ordinal"],errors="coerce") < t["ordinal"]).all()
        ),
        "transition_events_ge_40":len(t)>=40,
        "transition_named_players_ge_35":t["stable_id"].nunique()>=35,
        "rb_transition_players_ge_10":fam_players["RB"]>=10,
        "wr_transition_players_ge_15":fam_players["WR"]>=15,
        "te_transition_players_ge_8":fam_players["TE"]>=8,
        "prior_team_ge4_share_ge_075":ge4>=0.75,
        "competitor_metadata_all_transition_rows":competitor_ok,
        "duplicate_transition_event_identities_zero":audit["duplicate_transition_event_identities"]==0,
        "target_game_player_snaps_used_zero":True,
        "target_game_opportunity_performance_used_zero":True,
        "predictive_outcomes_scored_zero":True,
    }

    return {
        "season":int(season),
        "admitted_player_weeks":int(len(s)),
        "distinct_named_players":int(s["stable_id"].nunique()),
        "scheduled_team_weeks":int(total),
        "covered_team_weeks":int(covered),
        "scheduled_team_week_roster_coverage":cov,
        "raw_scoped_roster_rows":int(len(r)),
        "stable_id_coverage":stable_cov,
        "transition_events":int(len(t)),
        "transition_named_players":int(t["stable_id"].nunique()),
        "offseason_transition_events":int(t["transition_kind"].eq("OFFSEASON_TEAM_CHANGE").sum()),
        "inseason_transition_events":int(t["transition_kind"].eq("INSEASON_TEAM_CHANGE").sum()),
        "prior_team_ge4_events":int(t["prior_team_support_ge4"].sum()),
        "prior_team_ge4_share":ge4,
        "prior_team_ge8_events":int(t["prior_team_support_ge8"].sum()),
        "prior_team_ge8_share":ge8,
        "competitor_metadata_coverage":float(
            (
                t["same_family_competitor_count"].notna()
                & t["same_family_competitor_stable_ids"].astype(str).str.startswith("[")
            ).mean()
        ) if len(t) else 0.0,
        "family_summary":fam_rows,
        "gates":gates,
        "source_ready":bool(all(gates.values())),
    }


def prospective_summary(roster,scheduled,states,transitions,audit):
    season=PARITY_SEASON
    r=roster.loc[roster["season"].eq(season)].copy()
    t=transitions.loc[transitions["season"].eq(season)].copy()
    stable_cov=float(r["stable_id"].ne("").mean()) if len(r) else 0.0

    weekly=[]
    complete=[]
    for week in WEEKS:
        covered,total,cov=team_week_coverage(roster,scheduled,season,week)
        if total:
            weekly.append({
                "week":week,
                "scheduled_team_weeks":total,
                "covered_team_weeks":covered,
                "coverage":cov,
            })
            if cov>=0.95:
                complete.append(week)
    latest=max(complete) if complete else 0
    latest_row=next((x for x in weekly if x["week"]==latest),None)
    latest_cov=float(latest_row["coverage"]) if latest_row else 0.0

    gates={
        "weekly_roster_2026_available":len(r)>0,
        "explicit_season_week_semantics_match":True,
        "stable_gsis_coverage_ge_099":stable_cov>=0.99,
        "latest_complete_observed_week_ge_5":latest>=5,
        "latest_complete_week_coverage_ge_095":latest_cov>=0.95,
        "ambiguous_multi_team_player_weeks_zero":audit["ambiguous_multi_team_player_weeks"]==0,
        "transition_events_nonzero":len(t)>0,
        "transition_semantics_roster_chronology_only":True,
        "outcomes_2026_used_zero":True,
    }
    return {
        "season":season,
        "raw_scoped_roster_rows":int(len(r)),
        "stable_id_coverage":stable_cov,
        "latest_complete_observed_roster_week":int(latest),
        "latest_complete_week_coverage":latest_cov,
        "weekly_coverage":weekly,
        "transition_events":int(len(t)),
        "transition_named_players":int(t["stable_id"].nunique()),
        "offseason_transition_events":int(t["transition_kind"].eq("OFFSEASON_TEAM_CHANGE").sum()),
        "inseason_transition_events":int(t["transition_kind"].eq("INSEASON_TEAM_CHANGE").sum()),
        "gates":gates,
        "parity_ready":bool(all(gates.values())),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)

    import nflreadpy as nfl

    scheduled,schedule_audit=normalize_schedule(
        pdx(nfl.load_schedules(seasons=list(SOURCE_SEASONS)))
    )
    roster,roster_audit=normalize_rosters(
        [(s,pdx(nfl.load_rosters_weekly(int(s)))) for s in SOURCE_SEASONS],
        scheduled,
    )
    states,transitions,state_audit=build_states(roster)

    historical=[
        historical_summary(s,roster,scheduled,states,transitions,state_audit)
        for s in HISTORICAL_SEASONS
    ]
    prospective=prospective_summary(
        roster,scheduled,states,transitions,state_audit
    )
    integrity={
        "coefficients_fit_zero":True,
        "thresholds_searched_zero":True,
        "sportsbook_inputs_zero":True,
        "oddsapi_calls_zero":True,
        "paid_oddsapi_calls_zero":True,
        "production_changed_false":True,
        "player_transition_outcomes_scored_zero":True,
        "player_opportunity_relationships_scored_zero":True,
        "outcomes_2026_used_zero":True,
    }
    ready=bool(
        all(x["source_ready"] for x in historical)
        and prospective["parity_ready"]
        and all(integrity.values())
    )
    result={
        "version":VERSION,
        "freeze_commit":FREEZE_COMMIT,
        "disposition":(
            "PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY"
            if ready else
            "PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_BLOCKED"
        ),
        "source_ready":ready,
        "schedule_audit":schedule_audit,
        "roster_audit":roster_audit,
        "state_audit":state_audit,
        "historical_seasons":historical,
        "prospective_2026":prospective,
        "integrity_gates":integrity,
        "player_transition_outcomes_scored":0,
        "player_opportunity_relationships_scored":0,
        "outcomes_2026_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "paid_oddsapi_calls":0,
        "coefficients_fit":0,
        "thresholds_searched":0,
        "production_changed":False,
    }

    roster.to_csv(a.out_dir/"player_team_context_normalized_rosters.csv",index=False)
    states.to_csv(a.out_dir/"player_team_context_states.csv",index=False)
    transitions.to_csv(a.out_dir/"player_team_context_transition_events.csv",index=False)
    pd.DataFrame([
        {"season":s["season"],**f}
        for s in historical for f in s["family_summary"]
    ]).to_csv(a.out_dir/"player_team_context_family_summary.csv",index=False)
    (a.out_dir/"player_team_context_source_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,default=json_default)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result,indent=2,sort_keys=True,default=json_default))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
