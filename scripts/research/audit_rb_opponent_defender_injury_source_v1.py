#!/usr/bin/env python3
"""Audit nflverse injury source readiness for role-specific opponent defenders.

Source-contract audit only. No game outcomes, sportsbook data, model fitting or
production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team

SEASONS=(2023,2024,2025,2026)
HIST_FULL=(2024,2025)
CURRENT_COMPLETED_THROUGH_WEEK=4
FRONT_POS={"DT","NT","DL","DE","EDGE","LB","ILB","OLB"}


def _to_pandas(x):
    if isinstance(x,pd.DataFrame):
        return x.copy()
    if hasattr(x,"to_pandas"):
        return x.to_pandas()
    if hasattr(x,"to_dicts"):
        return pd.DataFrame(x.to_dicts())
    return pd.DataFrame(x)


def _first(df,names,default=pd.NA):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default,index=df.index)


def normalize(raw:pd.DataFrame)->pd.DataFrame:
    if raw is None or raw.empty:
        raise RuntimeError("nflverse injury source returned zero rows")
    x=raw.copy()
    x.columns=[str(c).strip().lower() for c in x.columns]
    if "season" not in x.columns:
        raise RuntimeError("injury source missing season")
    week_col=next((c for c in ("week","report_week") if c in x.columns),None)
    if week_col is None:
        raise RuntimeError("injury source missing week/report_week")

    out=pd.DataFrame(index=x.index)
    out["season"]=pd.to_numeric(x["season"],errors="coerce").astype("Int64")
    out["week"]=pd.to_numeric(x[week_col],errors="coerce").astype("Int64")
    out["team"]=_first(x,["team","team_abbr","team_abbreviation","club"]).map(canon_team)
    out["player"]=_first(x,["full_name","player_name","player","name"]).astype("string").fillna("").str.strip()
    out["gsis_id"]=_first(x,["gsis_id","player_id","nflverse_player_id"]).astype("string").fillna("").str.strip()
    out["position"]=_first(x,["position","position_group","pos"]).astype("string").fillna("").str.upper().str.strip()
    out["report_status"]=_first(x,["report_status","game_status","status"]).astype("string").fillna("").str.strip()
    out["practice_status"]=_first(x,["practice_status","practice_participation"]).astype("string").fillna("").str.strip()
    out["primary_injury"]=_first(x,["report_primary_injury","primary_injury","injury","injury_type"]).astype("string").fillna("").str.strip()

    out=out.loc[
        out["season"].notna() & out["week"].notna()
        & out["team"].astype(str).ne("")
        & out["player"].ne("")
    ].copy()
    out["season"]=out["season"].astype(int)
    out["week"]=out["week"].astype(int)
    out["front7_descriptive"]=out["position"].isin(FRONT_POS)
    return out.reset_index(drop=True)


def _rate(s:pd.Series)->float:
    return float(s.mean()) if len(s) else 0.0


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()

    import nflreadpy as nfl
    raw=_to_pandas(nfl.load_injuries(seasons=list(SEASONS)))
    x=normalize(raw)
    x=x.loc[x["season"].isin(SEASONS)].copy()

    weekly=[]
    for (season,week),q in x.groupby(["season","week"],sort=True):
        f=q.loc[q["front7_descriptive"]]
        weekly.append({
            "season":int(season),"week":int(week),
            "rows":int(len(q)),"teams":int(q["team"].nunique()),
            "position_complete_rate":_rate(q["position"].ne("")),
            "gsis_complete_rate":_rate(q["gsis_id"].ne("")),
            "report_status_complete_rate":_rate(q["report_status"].ne("")),
            "practice_status_complete_rate":_rate(q["practice_status"].ne("")),
            "front_rows":int(len(f)),
            "front_teams":int(f["team"].nunique()),
            "front_gsis_complete_rate":_rate(f["gsis_id"].ne("")),
            "front_report_status_complete_rate":_rate(f["report_status"].ne("")),
        })
    weekly_df=pd.DataFrame(weekly)

    season_rows=[]
    for season,q in x.groupby("season",sort=True):
        f=q.loc[q["front7_descriptive"]]
        season_rows.append({
            "season":int(season),
            "rows":int(len(q)),
            "weeks":int(q["week"].nunique()),
            "min_week":int(q["week"].min()),
            "max_week":int(q["week"].max()),
            "teams":int(q["team"].nunique()),
            "position_complete_rate":_rate(q["position"].ne("")),
            "gsis_complete_rate":_rate(q["gsis_id"].ne("")),
            "report_status_complete_rate":_rate(q["report_status"].ne("")),
            "practice_status_complete_rate":_rate(q["practice_status"].ne("")),
            "front_rows":int(len(f)),
            "front_teams":int(f["team"].nunique()),
            "front_gsis_complete_rate":_rate(f["gsis_id"].ne("")),
            "front_report_status_complete_rate":_rate(f["report_status"].ne("")),
        })
    season_df=pd.DataFrame(season_rows)

    hist_ok=True
    reasons=[]
    for season in HIST_FULL:
        q=season_df.loc[season_df["season"].eq(season)]
        if q.empty:
            hist_ok=False; reasons.append(f"{season}_missing")
            continue
        r=q.iloc[0]
        if int(r["weeks"])<16:
            hist_ok=False; reasons.append(f"{season}_weeks_lt16")
        if int(r["teams"])<32:
            hist_ok=False; reasons.append(f"{season}_teams_lt32")
        if float(r["front_gsis_complete_rate"])<.95:
            hist_ok=False; reasons.append(f"{season}_front_gsis_lt95")
        if float(r["front_report_status_complete_rate"])<.95:
            hist_ok=False; reasons.append(f"{season}_front_status_lt95")

    cur=x.loc[x["season"].eq(2026)]
    current_weeks=set(cur["week"].astype(int).tolist())
    required_current=set(range(1,CURRENT_COMPLETED_THROUGH_WEEK+1))
    current_ok=required_current.issubset(current_weeks)
    if not current_ok:
        reasons.append("2026_completed_weeks_missing")
    if len(cur):
        f=cur.loc[cur["front7_descriptive"]]
        if _rate(f["gsis_id"].ne(""))<.95:
            current_ok=False; reasons.append("2026_front_gsis_lt95")
        if _rate(f["report_status"].ne(""))<.95:
            current_ok=False; reasons.append("2026_front_status_lt95")
    else:
        current_ok=False; reasons.append("2026_missing")

    positions=(
        x.groupby(["season","position"]).size().rename("rows").reset_index()
        .sort_values(["season","rows"],ascending=[True,False])
    )
    statuses=(
        x.groupby(["season","report_status"]).size().rename("rows").reset_index()
        .sort_values(["season","rows"],ascending=[True,False])
    )

    disposition=(
        "SOURCE_READY_FOR_SEPARATE_PREDICTIVE_PLAN"
        if hist_ok and current_ok
        else "SOURCE_PARITY_NOT_CLEARED"
    )
    payload={
        "disposition":disposition,
        "seasons":list(SEASONS),
        "current_completed_through_week_required":CURRENT_COMPLETED_THROUGH_WEEK,
        "historical_gate_pass":bool(hist_ok),
        "current_gate_pass":bool(current_ok),
        "failure_reasons":reasons,
        "raw_rows":int(len(raw)),
        "normalized_rows":int(len(x)),
        "front_position_tokens":sorted(FRONT_POS),
        "game_outcomes_loaded":False,
        "sportsbook_inputs_used":False,
        "predictive_model_fit":False,
        "production_changed":False,
        "semantic_warning":"weekly injury report status is not T-75 official inactive certification",
    }
    args.out_dir.mkdir(parents=True,exist_ok=True)
    weekly_df.to_csv(args.out_dir/"opponent_defender_injury_weekly_coverage.csv",index=False)
    season_df.to_csv(args.out_dir/"opponent_defender_injury_season_coverage.csv",index=False)
    positions.to_csv(args.out_dir/"opponent_defender_injury_position_inventory.csv",index=False)
    statuses.to_csv(args.out_dir/"opponent_defender_injury_status_inventory.csv",index=False)
    (args.out_dir/"opponent_defender_injury_source_result.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(payload,indent=2,sort_keys=True))
    print("\nSEASON COVERAGE")
    print(season_df.to_string(index=False))
    print("\nPOSITION INVENTORY")
    print(positions.to_string(index=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
