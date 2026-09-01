#!/usr/bin/env python3
"""RB-ND2A: audit leakage-safe historical RB role-state sources.

No model is fit and no target outcome is used for feature selection.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TEAM_MAP = {"OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA","JAX":"JAC","ARZ":"ARI","WSH":"WAS"}


def to_pd(v):
    if isinstance(v, pd.DataFrame): return v.copy()
    if hasattr(v, "to_pandas"): return v.to_pandas()
    if hasattr(v, "to_dicts"): return pd.DataFrame(v.to_dicts())
    return pd.DataFrame(v)


def lower(x):
    x = x.copy(); x.columns = [str(c).strip().lower() for c in x.columns]; return x


def canon_team(v):
    if pd.isna(v): return ""
    s = str(v).strip().upper()
    return TEAM_MAP.get(s,s) if s not in {"","NAN","NONE","<NA>"} else ""


def name_key(v):
    s = re.sub(r"[^a-z0-9]", "", str(v or "").lower())
    return s


def first(df, names):
    for c in names:
        if c in df.columns: return c
    return None


def load_one(root: Path, name: str):
    h = list(root.rglob(name))
    if len(h)!=1: raise RuntimeError(f"expected one {name}, found {len(h)}")
    return lower(pd.read_csv(h[0], low_memory=False))


def schema_row(source, season, df, error=""):
    cols = list(df.columns)
    return {
        "source": source, "season": season, "rows": len(df), "columns": len(cols),
        "week_cols": ";".join(c for c in cols if "week" in c),
        "date_cols": ";".join(c for c in cols if any(k in c for k in ["date","time","timestamp","updated","report"])),
        "player_id_cols": ";".join(c for c in cols if any(k in c for k in ["gsis","player_id","nfl_id","pfr_id","espn_id"])),
        "name_cols": ";".join(c for c in cols if c in ["full_name","player_name","player","name","football_name","display_name"]),
        "team_cols": ";".join(c for c in cols if c in ["team","club_code","team_abbr","team_abbreviation","club","possession_team"]),
        "position_cols": ";".join(c for c in cols if "position" in c or c=="pos"),
        "role_cols": ";".join(c for c in cols if any(k in c for k in ["depth","snap","starter","status","practice","report"])),
        "all_columns": json.dumps(cols), "error": error,
    }


def load_sources():
    import nflreadpy as nfl
    outputs = {}
    rows=[]
    for season in [2024,2025]:
        calls = [
            ("weekly_rosters", lambda: nfl.load_rosters_weekly(seasons=[season])),
            ("participation", lambda: nfl.load_participation(seasons=[season])),
            ("injuries", lambda: nfl.load_injuries(seasons=[season])),
            ("depth_charts", lambda: nfl.load_depth_charts(seasons=[season])),
        ]
        for source, fn in calls:
            try:
                df = lower(to_pd(fn()))
                outputs[(source,season)] = df
                rows.append(schema_row(source,season,df))
            except Exception as e:
                outputs[(source,season)] = pd.DataFrame()
                rows.append(schema_row(source,season,pd.DataFrame(),str(e)))
    try:
        pl = lower(to_pd(nfl.load_players()))
        outputs[("players",0)] = pl
        rows.append(schema_row("players",0,pl))
    except Exception as e:
        outputs[("players",0)] = pd.DataFrame(); rows.append(schema_row("players",0,pd.DataFrame(),str(e)))
    try:
        sc = lower(to_pd(nfl.load_schedules(seasons=[2024,2025])))
        outputs[("schedules",0)] = sc
        rows.append(schema_row("schedules",0,sc))
    except Exception as e:
        outputs[("schedules",0)] = pd.DataFrame(); rows.append(schema_row("schedules",0,pd.DataFrame(),str(e)))
    return outputs, pd.DataFrame(rows)


def normalize_rosters(df, season):
    if df.empty: return pd.DataFrame()
    team = first(df,["team","club_code","team_abbr","team_abbreviation","club"])
    nm = first(df,["full_name","football_name","player_name","player","name"])
    wk = first(df,["week","report_week"])
    pos = first(df,["position","pos"])
    pid = first(df,["gsis_id","player_id","nfl_id"])
    if not all([team,nm,wk]): return pd.DataFrame()
    x=pd.DataFrame({"season":season,"week":pd.to_numeric(df[wk],errors="coerce"),"team":df[team].map(canon_team),"name_key":df[nm].map(name_key),"player":df[nm].astype(str)})
    x["position"] = df[pos].astype(str).str.upper() if pos else ""
    x["roster_status"] = df[first(df,["status","roster_status"])].astype(str) if first(df,["status","roster_status"]) else ""
    x["player_id"] = df[pid].astype(str) if pid else ""
    return x.dropna(subset=["week"]).drop_duplicates(["week","team","name_key"])


def normalize_injuries(df, season):
    if df.empty: return pd.DataFrame()
    team=first(df,["team","club_code","team_abbr","team_abbreviation","club"]); nm=first(df,["full_name","player_name","player","name"]); wk=first(df,["week","report_week"])
    if not all([team,nm,wk]): return pd.DataFrame()
    st=first(df,["report_status","game_status","status"]); pr=first(df,["practice_status","practice_participation"])
    x=pd.DataFrame({"season":season,"week":pd.to_numeric(df[wk],errors="coerce"),"team":df[team].map(canon_team),"name_key":df[nm].map(name_key),
                    "injury_status":df[st].astype(str) if st else "","practice_status":df[pr].astype(str) if pr else ""})
    return x.dropna(subset=["week"]).drop_duplicates(["week","team","name_key"],keep="last")


def schedule_games(df):
    if df.empty: return pd.DataFrame()
    datecol=first(df,["gameday","game_date","date","game_datetime","game_time"])
    if not all(c in df.columns for c in ["season","week","home_team","away_team"]): return pd.DataFrame()
    x=df.copy(); x["season"]=pd.to_numeric(x.season,errors="coerce"); x["week"]=pd.to_numeric(x.week,errors="coerce")
    if "game_type" in x.columns: x=x.loc[x.game_type.astype(str).str.upper().eq("REG")]
    dt=pd.to_datetime(x[datecol],errors="coerce",utc=True) if datecol else pd.Series(pd.NaT,index=x.index)
    rows=[]
    for i,r in x.iterrows():
        for t,o in [(canon_team(r.home_team),canon_team(r.away_team)),(canon_team(r.away_team),canon_team(r.home_team))]:
            rows.append({"season":int(r.season),"week":int(r.week),"team":t,"opponent":o,"kickoff":dt.loc[i]})
    return pd.DataFrame(rows)


def depth_audit(df, season, games):
    out={"season":season,"depth_rows":len(df),"date_field":"","week_field":"","rb_rows":0,"unique_dates":0,"pregame_asof_possible":0,"notes":""}
    if df.empty: return out
    d=df.copy(); team=first(d,["club_code","team","team_abbr"]); pos=first(d,["position","pos","depth_position"]); wk=first(d,["week"])
    date=first(d,["dt","date","snapshot_date","updated_at","timestamp"])
    out["date_field"]=date or ""; out["week_field"]=wk or ""
    if pos: out["rb_rows"]=int(d[pos].astype(str).str.upper().str.contains("RB|FB|HB",regex=True).sum())
    if date:
        dd=pd.to_datetime(d[date],errors="coerce",utc=True); out["unique_dates"]=int(dd.nunique()); out["pregame_asof_possible"]=int(dd.notna().any() and not games.empty)
        out["notes"]="date-bearing depth rows require as-of-before-kickoff validation before use"
    elif wk:
        out["pregame_asof_possible"]=1; out["notes"]="explicit week-tagged depth rows potentially usable after same-week semantics audit"
    else: out["notes"]="no week/date boundary; unsafe for historical target-week role"
    return out


def participation_audit(df, season):
    out={"season":season,"rows":len(df),"week_field":"","game_field":"","player_id_mode":"","snap_like_cols":"","player_list_cols":"","laggable_player_role_possible":0,"notes":""}
    if df.empty: return out
    wk=first(df,["week"]); game=first(df,["nflverse_game_id","game_id"]); out["week_field"]=wk or ""; out["game_field"]=game or ""
    direct=[c for c in df.columns if c in ["gsis_id","player_id","nfl_id","player_name","full_name"]]
    lists=[c for c in df.columns if "players" in c or "player_ids" in c or c in ["offense_players","defense_players"]]
    snaps=[c for c in df.columns if any(k in c for k in ["snap","route","participation","position"])]
    out["player_id_mode"]="direct" if direct else "list/play-level" if lists else "none"
    out["snap_like_cols"]=";".join(snaps); out["player_list_cols"]=";".join(lists)
    out["laggable_player_role_possible"]=int(bool(direct or lists) and bool(wk or game))
    out["notes"]="target-game participation is postgame and forbidden; only prior-game/rolling lagged aggregates may be used"
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--m94c-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    rb=load_one(a.m94c_root,"m94c_2025_rb_trace.csv"); rb=rb.loc[pd.to_numeric(rb.season,errors="coerce").eq(2025) & rb.position.astype(str).str.upper().isin(["RB","FB"])].copy(); rb["team"]=rb.team.map(canon_team); rb["name_key"]=rb.player.map(name_key); rb["week"]=pd.to_numeric(rb.week,errors="coerce")
    sources,schemas=load_sources(); games=schedule_games(sources[("schedules",0)])
    ros={s:normalize_rosters(sources[("weekly_rosters",s)],s) for s in [2024,2025]}; inj={s:normalize_injuries(sources[("injuries",s)],s) for s in [2024,2025]}

    r25=ros[2025]; i25=inj[2025]
    z=rb[["week","team","name_key","player"]].copy()
    if not r25.empty:
        z=z.merge(r25[["week","team","name_key","roster_status","player_id"]],on=["week","team","name_key"],how="left",validate="many_to_one")
    else: z["roster_status"]=np.nan; z["player_id"]=np.nan
    if not i25.empty: z=z.merge(i25,on=["week","team","name_key"],how="left",validate="many_to_one")
    else: z["injury_status"]=np.nan; z["practice_status"]=np.nan

    # prior-week roster continuity on same team; safe because it uses W-1 roster snapshot only.
    if not r25.empty:
        prev=r25[["week","team","name_key"]].copy(); prev["week"]+=1; prev["prior_week_same_team_roster"]=1
        z=z.merge(prev,on=["week","team","name_key"],how="left"); z["prior_week_same_team_roster"]=z.prior_week_same_team_roster.fillna(0)
    else: z["prior_week_same_team_roster"]=0

    # prior-season same team and any NFL roster history, for Week 1 initialization.
    r24=ros[2024]
    if not r24.empty:
        last24=r24.sort_values("week").drop_duplicates(["team","name_key"],keep="last")[["team","name_key"]]; last24["prior_season_same_team"]=1
        any24=r24[["name_key"]].drop_duplicates(); any24["prior_season_nfl_roster"]=1
        z=z.merge(last24,on=["team","name_key"],how="left").merge(any24,on="name_key",how="left")
    else: z["prior_season_same_team"]=np.nan; z["prior_season_nfl_roster"]=np.nan
    z["prior_season_same_team"]=z.prior_season_same_team.fillna(0); z["prior_season_nfl_roster"]=z.prior_season_nfl_roster.fillna(0)
    z["new_team_or_no_prior_same_team"]=(1-z.prior_season_same_team).astype(int); z["no_prior_nfl_roster"]=(1-z.prior_season_nfl_roster).astype(int)

    coverage=pd.DataFrame([
        {"signal":"current_week_roster_match","all_n":len(z),"all_coverage":float(z.roster_status.notna().mean()),"week1_n":int(z.week.eq(1).sum()),"week1_coverage":float(z.loc[z.week.eq(1),"roster_status"].notna().mean())},
        {"signal":"current_week_injury_report","all_n":len(z),"all_coverage":float(z.injury_status.notna().mean()),"week1_n":int(z.week.eq(1).sum()),"week1_coverage":float(z.loc[z.week.eq(1),"injury_status"].notna().mean())},
        {"signal":"prior_week_same_team_roster","all_n":len(z),"all_coverage":float(z.prior_week_same_team_roster.mean()),"week1_n":int(z.week.eq(1).sum()),"week1_coverage":0.0},
        {"signal":"prior_season_same_team","all_n":len(z),"all_coverage":float(z.prior_season_same_team.mean()),"week1_n":int(z.week.eq(1).sum()),"week1_coverage":float(z.loc[z.week.eq(1),"prior_season_same_team"].mean())},
        {"signal":"prior_season_any_nfl_roster","all_n":len(z),"all_coverage":float(z.prior_season_nfl_roster.mean()),"week1_n":int(z.week.eq(1).sum()),"week1_coverage":float(z.loc[z.week.eq(1),"prior_season_nfl_roster"].mean())},
    ])

    depth=pd.DataFrame([depth_audit(sources[("depth_charts",s)],s,games.loc[games.season.eq(s)]) for s in [2024,2025]])
    part=pd.DataFrame([participation_audit(sources[("participation",s)],s) for s in [2024,2025]])
    m94_role=pd.DataFrame([{"m94c_rows":len(rb),"role_nonnull":int(rb.get("role",pd.Series(index=rb.index,dtype=object)).notna().sum()),"rules_role_nonnull":int(rb.get("rules_role",pd.Series(index=rb.index,dtype=object)).notna().sum())}])

    # Save representative schema samples to make subsequent implementation deterministic.
    sample_rows=[]
    for (src,season),df in sources.items():
        if df.empty: continue
        for _,row in df.head(3).iterrows(): sample_rows.append({"source":src,"season":season,"sample_json":row.to_json(default_handler=str)})
    samples=pd.DataFrame(sample_rows)

    schemas.to_csv(a.out_dir/"nd2a_source_schema.csv",index=False); coverage.to_csv(a.out_dir/"nd2a_m94c_coverage.csv",index=False); depth.to_csv(a.out_dir/"nd2a_depth_audit.csv",index=False); part.to_csv(a.out_dir/"nd2a_participation_audit.csv",index=False); m94_role.to_csv(a.out_dir/"nd2a_m94c_role_hole.csv",index=False); samples.to_csv(a.out_dir/"nd2a_source_samples.csv",index=False); z.to_csv(a.out_dir/"nd2a_identity_coverage_trace.csv",index=False)
    print("=== M94C role hole ==="); print(m94_role.to_string(index=False)); print("=== source schema ==="); print(schemas[["source","season","rows","week_cols","date_cols","player_id_cols","role_cols","error"]].to_string(index=False)); print("=== coverage ==="); print(coverage.to_string(index=False)); print("=== depth ==="); print(depth.to_string(index=False)); print("=== participation ==="); print(part.to_string(index=False))
    return 0

if __name__=="__main__": raise SystemExit(main())
