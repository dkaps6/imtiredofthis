#!/usr/bin/env python3
"""Grade frozen Week-3 public-intent labels descriptively.

No coefficient fitting and no post-hoc concentration threshold. The frozen
contract asks for actual RB/FB carries/snaps, HHI, top-successor share, whether
the frozen lead identity led, and a structural comparison to the pregame label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_stats_loader_v2 import load_weekly_player_stats
from scripts.utils.canonical_names import canonicalize_player_name_safe

SEASON=2026
WEEK=3

def _to_pandas(x):
    return x.to_pandas() if hasattr(x,"to_pandas") else pd.DataFrame(x)

def _pick(df,names):
    for c in names:
        if c in df.columns:
            return c
    raise RuntimeError(f"missing any of {names}; columns={sorted(df.columns)}")

def _actual_room(team:str)->pd.DataFrame:
    import nflreadpy as nfl

    stats=load_weekly_player_stats(SEASON).copy()
    stats.columns=[str(c).strip().lower() for c in stats.columns]
    stats=stats.loc[pd.to_numeric(stats["week"],errors="coerce").eq(WEEK)].copy()
    tc=_pick(stats,("recent_team","team","team_abbr","club"))
    nc=_pick(stats,("player_display_name","player_name","player"))
    rac=_pick(stats,("carries","rushing_attempts","rush_attempts"))
    stats["team"]=stats[tc].astype("string").fillna("").str.strip().map(canon_team)
    cc=stats[nc].astype("string").fillna("").str.strip().map(canonicalize_player_name_safe)
    stats["player"]=cc.map(lambda t:t[0])
    stats["player_clean_key"]=cc.map(lambda t:t[1])
    stats["carries"]=pd.to_numeric(stats[rac],errors="coerce").fillna(0.0)

    roster=_to_pandas(nfl.load_rosters_weekly(SEASON)).copy()
    roster.columns=[str(c).strip().lower() for c in roster.columns]
    roster=roster.loc[pd.to_numeric(roster["week"],errors="coerce").eq(WEEK)].copy()
    rtc=_pick(roster,("team","team_abbr","club_code"))
    rnc=_pick(roster,("full_name","football_name","player_name","player"))
    rpc=_pick(roster,("position","depth_chart_position"))
    roster["team"]=roster[rtc].astype("string").fillna("").str.strip().map(canon_team)
    rc=roster[rnc].astype("string").fillna("").str.strip().map(canonicalize_player_name_safe)
    roster["player"]=rc.map(lambda t:t[0])
    roster["player_clean_key"]=rc.map(lambda t:t[1])
    roster["position"]=roster[rpc].astype("string").fillna("").str.upper().str.strip()
    roster["position"]=roster["position"].replace({"HB":"RB","TB":"RB"})
    roster=roster.loc[roster["team"].eq(canon_team(team)) & roster["position"].isin(["RB","FB"])].copy()

    snaps=_to_pandas(nfl.load_snap_counts(seasons=[SEASON])).copy()
    snaps.columns=[str(c).strip().lower() for c in snaps.columns]
    snaps=snaps.loc[pd.to_numeric(snaps["week"],errors="coerce").eq(WEEK)].copy()
    stc=_pick(snaps,("team","team_abbr","club"))
    snc=_pick(snaps,("player","player_name","full_name"))
    snaps["team"]=snaps[stc].astype("string").fillna("").str.strip().map(canon_team)
    sc=snaps[snc].astype("string").fillna("").str.strip().map(canonicalize_player_name_safe)
    snaps["player_clean_key"]=sc.map(lambda t:t[1])
    for c in ("offense_snaps","offense_pct"):
        if c not in snaps.columns:
            snaps[c]=np.nan
    snaps["offense_snaps"]=pd.to_numeric(snaps["offense_snaps"],errors="coerce").fillna(0.0)
    snaps["offense_pct"]=pd.to_numeric(snaps["offense_pct"],errors="coerce")
    snaps=snaps.loc[snaps["team"].eq(canon_team(team))].copy()
    snaps=snaps.groupby(["team","player_clean_key"],as_index=False).agg(
        offense_snaps=("offense_snaps","max"),
        offense_pct=("offense_pct","max"),
    )

    s=stats.loc[stats["team"].eq(canon_team(team)),["team","player_clean_key","carries"]].drop_duplicates(
        ["team","player_clean_key"],keep="last"
    )
    out=roster[["team","player","player_clean_key","position"]].drop_duplicates()
    out=out.merge(s,on=["team","player_clean_key"],how="left")
    out=out.merge(snaps,on=["team","player_clean_key"],how="left")
    out["carries"]=out["carries"].fillna(0.0)
    out["offense_snaps"]=out["offense_snaps"].fillna(0.0)
    # Actual participating room; retain positive carry OR positive offensive snap.
    out=out.loc[out["carries"].gt(0) | out["offense_snaps"].gt(0)].copy()
    return out.sort_values(["carries","offense_snaps"],ascending=False).reset_index(drop=True)

def _grade_team(frozen:dict)->tuple[dict,pd.DataFrame]:
    team=canon_team(frozen["team"])
    room=_actual_room(team)
    total=float(room["carries"].sum())
    room["carry_share"]=room["carries"]/total if total>0 else 0.0
    hhi=float(np.square(room["carry_share"]).sum()) if total>0 else np.nan
    top=room.iloc[0] if len(room) else None
    top_name=str(top["player"]) if top is not None else ""
    top_share=float(top["carry_share"]) if top is not None else np.nan

    lead=frozen.get("lead_identity")
    lead_key=""
    if lead:
        _,lead_key=canonicalize_player_name_safe(lead)
    lead_row=room.loc[room["player_clean_key"].astype(str).eq(lead_key)] if lead_key else room.iloc[0:0]
    lead_carries=float(lead_row["carries"].iloc[0]) if len(lead_row)==1 else np.nan
    lead_share=float(lead_row["carry_share"].iloc[0]) if len(lead_row)==1 else np.nan
    lead_led=bool(len(lead_row)==1 and lead_carries==room["carries"].max()) if len(room) else False

    label=str(frozen["intent_label"])
    # No invented numeric pass/fail threshold. Structural observations only.
    if label=="WARREN_LEAD_BACK_LEAN_WITH_DEPTH_SUPPORT":
        structural_observation=(
            "LEAD_IDENTITY_LED_WITH_DEPTH_PARTICIPATION"
            if lead_led and len(room)>=2
            else "LEAD_IDENTITY_DID_NOT_LEAD"
            if not lead_led
            else "LEAD_IDENTITY_LED_NO_DEPTH_PARTICIPATION"
        )
    elif label=="ROTATION_PRESERVED_NO_CLEAR_SUCCESSOR_CONCENTRATION":
        structural_observation="DESCRIPTIVE_ROTATION_OBSERVED" if len(room)>=2 else "DESCRIPTIVE_SINGLE_BACK_ROOM_OBSERVED"
    else:
        structural_observation="DESCRIPTIVE_ONLY"

    summary={
        "team":team,
        "frozen_intent_label":label,
        "frozen_lead_identity":lead,
        "frozen_confidence":frozen.get("concentration_confidence"),
        "actual_participating_rbfb_count":int(len(room)),
        "actual_rbfb_carries":total,
        "carry_hhi":hhi,
        "top_successor":top_name,
        "top_successor_carries":float(top["carries"]) if top is not None else np.nan,
        "top_successor_carry_share":top_share,
        "frozen_lead_carries":lead_carries,
        "frozen_lead_carry_share":lead_share,
        "frozen_lead_led_room":lead_led if lead else None,
        "structural_observation":structural_observation,
    }
    room["frozen_intent_label"]=label
    room["frozen_lead_identity"]=lead or ""
    return summary,room

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--capture-json",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    frozen=json.loads(a.capture_json.read_text(encoding="utf-8"))
    if frozen.get("study")!="PUBLIC_INTENT_WEEK3_PROSPECTIVE_CAPTURE_V1":
        raise RuntimeError("wrong frozen capture")
    if frozen.get("target_season")!=SEASON or frozen.get("target_week")!=WEEK:
        raise RuntimeError("target drift")
    if frozen.get("target_game_outcomes_read")!=0 or frozen.get("sportsbook_inputs_used")!=0:
        raise RuntimeError("pregame/no-sportsbook contract violated")

    summaries=[]
    details=[]
    for rec in frozen["teams"]:
        s,d=_grade_team(rec)
        summaries.append(s)
        details.append(d)
    summary=pd.DataFrame(summaries)
    detail=pd.concat(details,ignore_index=True,sort=False)

    # Primary frozen qualitative comparison: did PIT concentrate more than DEN?
    den=summary.loc[summary["team"].eq("DEN")].iloc[0]
    pit=summary.loc[summary["team"].eq("PIT")].iloc[0]
    comparison={
        "pit_hhi_minus_den_hhi":float(pit["carry_hhi"]-den["carry_hhi"]),
        "pit_top_share_minus_den_top_share":float(pit["top_successor_carry_share"]-den["top_successor_carry_share"]),
        "pit_more_concentrated_by_hhi":bool(pit["carry_hhi"]>den["carry_hhi"]),
        "pit_more_concentrated_by_top_share":bool(pit["top_successor_carry_share"]>den["top_successor_carry_share"]),
        "pit_frozen_lead_led_room":bool(pit["frozen_lead_led_room"]),
    }
    if comparison["pit_more_concentrated_by_hhi"] and comparison["pit_more_concentrated_by_top_share"] and comparison["pit_frozen_lead_led_room"]:
        observational_disposition="WEEK3_PUBLIC_INTENT_DIRECTIONALLY_INFORMATIVE"
    elif (not comparison["pit_more_concentrated_by_hhi"]) and (not comparison["pit_more_concentrated_by_top_share"]):
        observational_disposition="WEEK3_PUBLIC_INTENT_NOT_DIRECTIONALLY_INFORMATIVE"
    else:
        observational_disposition="WEEK3_PUBLIC_INTENT_MIXED_DESCRIPTIVE"

    a.out_dir.mkdir(parents=True,exist_ok=True)
    summary.to_csv(a.out_dir/"public_intent_week3_team_summary.csv",index=False)
    detail.to_csv(a.out_dir/"public_intent_week3_room_detail.csv",index=False)
    result={
        "status":"PUBLIC_INTENT_WEEK3_POSTGAME_GRADED",
        "observational_disposition":observational_disposition,
        "team_summaries":summary.to_dict(orient="records"),
        "frozen_primary_comparison":comparison,
        "parameters_fit":0,
        "sportsbook_inputs_used":0,
        "production_changed":False,
        "coefficient_promotion_authorized":False,
    }
    (a.out_dir/"public_intent_week3_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True,default=str))
    print(f"DISPOSITION={observational_disposition}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
