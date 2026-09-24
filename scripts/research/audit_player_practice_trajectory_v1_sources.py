#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

SKILL={"RB","FB","TE","WR"}

def to_pd(x):
    if isinstance(x,pd.DataFrame): return x.copy()
    if hasattr(x,"to_pandas"): return x.to_pandas()
    return pd.DataFrame(x)

def lower(df):
    x=df.copy(); x.columns=[str(c).strip().lower() for c in x.columns]; return x

def first(df,names):
    for c in names:
        if c in df.columns: return c
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    import nflreadpy as nfl
    seasons=[2023,2024,2025,2026]
    raw=lower(to_pd(nfl.load_injuries(seasons=seasons)))
    players=lower(to_pd(nfl.load_players()))
    (a.out_dir/"raw_columns.txt").write_text("\n".join(raw.columns)+"\n")

    season_col=first(raw,["season"]); week_col=first(raw,["week","report_week"])
    date_col=first(raw,["report_date","date","practice_date","report_timestamp","timestamp"])
    id_col=first(raw,["gsis_id","player_gsis_id","player_id"])
    name_col=first(raw,["full_name","player_name","player","name"])
    pstat_col=first(raw,["practice_status","practice_participation"])
    gstat_col=first(raw,["report_status","game_status","status"])
    team_col=first(raw,["team","team_abbr","club"])

    pid=first(players,["gsis_id","player_gsis_id","player_id"])
    ppos=first(players,["position","position_group"])
    bridge=pd.DataFrame()
    if id_col and pid and ppos:
        bridge=players[[pid,ppos]].dropna(subset=[pid]).drop_duplicates(pid).copy()
        bridge["_id"]=bridge[pid].astype("string").fillna("").str.strip()
        bridge["_pos"]=bridge[ppos].astype("string").fillna("").str.upper().str.strip()
        bridge=bridge[["_id","_pos"]]

    x=pd.DataFrame(index=raw.index)
    x["season"]=pd.to_numeric(raw[season_col],errors="coerce") if season_col else pd.NA
    x["week"]=pd.to_numeric(raw[week_col],errors="coerce") if week_col else pd.NA
    x["player_id"]=raw[id_col].astype("string").fillna("").str.strip() if id_col else ""
    x["player"]=raw[name_col].astype("string").fillna("").str.strip() if name_col else ""
    x["team"]=raw[team_col].astype("string").fillna("").str.strip() if team_col else ""
    x["practice_status"]=raw[pstat_col].astype("string").fillna("").str.strip() if pstat_col else ""
    x["game_status"]=raw[gstat_col].astype("string").fillna("").str.strip() if gstat_col else ""
    x["report_date_raw"]=raw[date_col].astype("string").fillna("").str.strip() if date_col else ""
    x["report_date"]=pd.to_datetime(x["report_date_raw"],errors="coerce",utc=True) if date_col else pd.NaT
    if not bridge.empty:
        x=x.merge(bridge,left_on="player_id",right_on="_id",how="left",validate="many_to_one")
        x["position"]=x["_pos"].fillna("")
    else:
        x["position"]=""

    x=x[x.season.notna() & x.week.notna()].copy()
    x["season"]=x.season.astype(int); x["week"]=x.week.astype(int)
    skill=x[x.position.isin(SKILL)].copy()

    def summarize(g):
        return pd.Series({
            "raw_rows":len(g),
            "player_weeks":g[["season","week","player_id","player"]].drop_duplicates().shape[0],
            "dated_rows":int(g.report_date.notna().sum()),
            "unique_dates":int(g.report_date.dt.date.nunique()) if g.report_date.notna().any() else 0,
        })
    by_season=x.groupby("season",dropna=False).apply(summarize).reset_index()
    by_season.to_csv(a.out_dir/"season_summary.csv",index=False)

    keycols=["season","week","player_id","player"]
    pw=skill.groupby(keycols,dropna=False).agg(
        rows=("practice_status","size"),
        distinct_report_dates=("report_date",lambda s:int(pd.Series(s.dropna().dt.date).nunique())),
        nonblank_practice=("practice_status",lambda s:int(s.astype(str).str.strip().ne("").sum())),
        nonblank_game_status=("game_status",lambda s:int(s.astype(str).str.strip().ne("").sum())),
        position=("position","first"),
        team=("team","first"),
    ).reset_index()
    pw.to_csv(a.out_dir/"skill_player_week_density.csv",index=False)

    hist=pw[pw.season.isin([2023,2024,2025])]
    with_report=hist[hist.nonblank_practice.gt(0)]
    ge2=float(with_report.distinct_report_dates.ge(2).mean()) if len(with_report) else 0.0
    ge3=float(with_report.distinct_report_dates.ge(3).mean()) if len(with_report) else 0.0
    current=pw[(pw.season.eq(2026)) & pw.week.le(3)]
    vocab={
        "practice_status":sorted(set(v for v in x.practice_status.astype(str) if v.strip()))[:200],
        "game_status":sorted(set(v for v in x.game_status.astype(str) if v.strip()))[:200],
    }
    (a.out_dir/"status_vocab.json").write_text(json.dumps(vocab,indent=2,sort_keys=True)+"\n")

    date_exists=bool(date_col)
    current_same_schema=bool(len(current))
    position_rate=float(x.position.astype(str).str.strip().ne("").mean()) if len(x) else 0.0
    passed=bool(date_exists and ge2>=.70 and ge3>=.50 and current_same_schema)
    payload={
        "study":"PLAYER_PRACTICE_TRAJECTORY_V1_SOURCE_AUDIT",
        "raw_rows":int(len(x)),
        "skill_player_weeks_historical_with_practice":int(len(with_report)),
        "report_date_field":date_col or "",
        "practice_status_field":pstat_col or "",
        "game_status_field":gstat_col or "",
        "position_resolution_rate":position_rate,
        "historical_skill_player_weeks_ge2_dates_rate":ge2,
        "historical_skill_player_weeks_ge3_dates_rate":ge3,
        "current_2026_w1_w3_skill_player_weeks":int(len(current)),
        "source_gate_pass":passed,
        "disposition":"PRACTICE_TRAJECTORY_SOURCE_QUALIFIED" if passed else "PRACTICE_TRAJECTORY_SOURCE_NOT_DENSE",
        "football_outcomes_read":0,
        "sportsbook_inputs_used":0,
        "predictive_models_fit":0,
        "production_changes":0,
    }
    (a.out_dir/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=="__main__": main()
