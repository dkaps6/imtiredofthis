#!/usr/bin/env python3
"""Source-only: require an identified corner to belong to its weekly opponent."""
import argparse
import json
from pathlib import Path
import pandas as pd
from scripts._opponent_map import canon_team
from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import _load_rosters, DEF_POSITIONS

def assess_cb_week_opponent(rows: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    needed = {"season","week","opponent","cb_gsis_id","cb_identity_method","source_quality_row_ready"}
    ro_needed = {"season","week","team","player_id","position"}
    if not needed.issubset(rows.columns) or not ro_needed.issubset(rosters.columns):
        raise ValueError("missing required source-quality or roster columns")
    r = rosters.loc[rosters.position.astype(str).isin(DEF_POSITIONS)].copy()
    r["team"] = r.team.map(canon_team)
    r["player_id"] = r.player_id.astype("string").fillna("").str.strip()
    r["season"] = pd.to_numeric(r.season,errors="raise").astype(int)
    r["week"] = pd.to_numeric(r.week,errors="raise").astype(int)
    r = r.loc[r.player_id.ne("") & r.team.notna()]
    membership = {
        (int(season),int(week),str(pid)): tuple(sorted(set(g.team.astype(str))))
        for (season,week,pid),g in r.groupby(["season","week","player_id"])
    }
    x=rows.copy()
    x["season"]=pd.to_numeric(x.season,errors="raise").astype(int)
    x["week"]=pd.to_numeric(x.week,errors="raise").astype(int)
    x["opponent"]=x.opponent.map(canon_team)
    labels,team_strings=[],[]
    for row in x.itertuples(index=False):
        pid="" if pd.isna(row.cb_gsis_id) else str(row.cb_gsis_id).strip()
        teams=membership.get((int(row.season),int(row.week),pid),()) if pid else ()
        team_strings.append("|".join(teams))
        if not pid: label="CB_GSIS_UNRESOLVED"
        elif not teams: label="WEEKLY_DEFENSIVE_ROSTER_UNOBSERVED"
        elif len(teams)>1: label="AMBIGUOUS_MULTITEAM_WEEK_ROSTER"
        elif teams[0]!=str(row.opponent): label="CONFIRMED_OTHER_WEEK_TEAM"
        else: label="WEEK_EXACT_DEFENDER_OPPONENT_CONFIRMED"
        labels.append(label)
    x["cb_week_roster_status"]=labels
    x["cb_week_roster_teams"]=team_strings
    prior=x.source_quality_row_ready.astype("string").str.lower().eq("true")
    x["previous_publication_eligible"]=prior
    x["source_quality_with_cb_week_guard"]=prior & x.cb_week_roster_status.eq(
        "WEEK_EXACT_DEFENDER_OPPONENT_CONFIRMED")
    x["previously_eligible_now_quarantined"]=prior & ~x.source_quality_with_cb_week_guard
    return x

def run(rows_file: Path, out_dir: Path) -> dict:
    x=pd.read_csv(rows_file,low_memory=False)
    rosters=_load_rosters(sorted(pd.to_numeric(x.season).astype(int).unique()))
    y=assess_cb_week_opponent(x,rosters)
    out_dir.mkdir(parents=True,exist_ok=True)
    columns=["season","week","wr_raw","wr_team","opponent","cb_raw","cb_source_player_id",
             "cb_gsis_id","cb_identity_method","source_quality_row_ready",
             "cb_week_roster_status","cb_week_roster_teams","source_quality_with_cb_week_guard",
             "previously_eligible_now_quarantined","content_version_timing_status"]
    y[[c for c in columns if c in y.columns]].to_csv(out_dir/"cb_week_opponent_guard_rows.csv",index=False)
    old=y.previous_publication_eligible
    lost=y.previously_eligible_now_quarantined
    result={
        "contract":"WR_CB_WEEK_OPPONENT_ROSTER_GUARD_V1",
        "input_rows":len(y),"prior_eligible":int(old.sum()),
        "eligible_after_week_roster_guard":int(y.source_quality_with_cb_week_guard.sum()),
        "newly_quarantined":int(lost.sum()),
        "prior_eligible_status":y.loc[old,"cb_week_roster_status"].value_counts().to_dict(),
        "newly_quarantined_by_identity_method":y.loc[lost,"cb_identity_method"].value_counts().to_dict(),
        "newly_quarantined_by_season":y.loc[lost,"season"].value_counts().sort_index().to_dict(),
        "no_archive_reacquisition":True,"outcomes_used":False,"sportsbook_used":False,
        "parameters_fit":0,"source_model_gate_cleared":False,
    }
    (out_dir/"cb_week_opponent_guard_summary.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True,indent=2))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--rows",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    run(a.rows,a.out_dir)
