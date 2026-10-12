#!/usr/bin/env python3
"""Source-readiness audit for individual RB run-location x defense location matchup.

No target-game RB outcome scoring, residuals, sportsbook inputs, fitted
predictor, or production changes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.pbp import get_pbp

VERSION="RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_SOURCE_V1"
TARGET_SEASONS=(2024,2025)
TARGET_WEEKS=tuple(range(2,19))
HISTORY_SEASONS=(2023,2024,2025)
PLAYER_GAME_WINDOW=8
DEF_GAME_WINDOW=8
MIN_PLAYER_GAMES=4
MIN_PLAYER_LOCATED_RUSHES=20
MIN_DEF_GAMES=4
MIN_DEF_RUSHES_PER_LOCATION=10
LOCATIONS=("LEFT","MIDDLE","RIGHT")


@dataclass
class AuditRow:
    season:int
    week:int
    team:str
    opponent:str
    player:str
    gsis_id:str
    stable_id:bool
    player_prior_games:int
    player_located_rushes:int
    player_state_ready:bool
    player_feature_max_time_key:int|None
    player_located_ypc:float|None
    player_share_left:float|None
    player_share_middle:float|None
    player_share_right:float|None
    player_location_mass_gap:float|None
    defense_prior_games:int
    defense_state_ready:bool
    defense_feature_max_time_key:int|None
    def_rushes_left:int
    def_rushes_middle:int
    def_rushes_right:int
    def_ypc_left:float|None
    def_ypc_middle:float|None
    def_ypc_right:float|None
    def_success_left:float|None
    def_success_middle:float|None
    def_success_right:float|None
    run_mix_def_ypc:float|None
    run_mix_def_success:float|None
    matchup_state_ready:bool


def _to_pandas(obj:Any)->pd.DataFrame:
    if isinstance(obj,pd.DataFrame):
        return obj.copy()
    if hasattr(obj,"to_dicts"):
        return pd.DataFrame(obj.to_dicts())
    if hasattr(obj,"to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _lower(df:pd.DataFrame)->pd.DataFrame:
    x=df.copy()
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def _num(s)->pd.Series:
    return pd.to_numeric(s,errors="coerce")


def _first(df:pd.DataFrame,names:list[str],default:Any="")->pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default,index=df.index)


def _clean_id(v:Any)->str:
    if v is None or pd.isna(v):
        return ""
    s=str(v).strip()
    return "" if s.lower() in {"","nan","none","<na>"} else s


def _team(v:Any)->str:
    try:
        return canon_team(v)
    except Exception:
        return str(v or "").strip().upper()


def _rb_position(v:Any)->bool:
    return str(v or "").strip().upper() in {"RB","HB","FB"}


def run_location(v:Any)->str:
    s=str(v or "").strip().upper()
    if s=="LEFT": return "LEFT"
    if s in {"MIDDLE","MID"}: return "MIDDLE"
    if s=="RIGHT": return "RIGHT"
    return ""


def _time_key(season:int,week:int)->int:
    return int(season)*100+int(week)


def _load_schedule()->pd.DataFrame:
    import nflreadpy as nfl
    rows=[]
    for season in TARGET_SEASONS:
        try:
            raw=nfl.load_schedules(seasons=[season])
        except TypeError:
            raw=nfl.load_schedules(season)
        x=_lower(_to_pandas(raw))
        if "game_type" in x.columns:
            x=x.loc[x["game_type"].astype(str).str.upper().eq("REG")].copy()
        elif "season_type" in x.columns:
            x=x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
        hc="home_team" if "home_team" in x.columns else "home"
        ac="away_team" if "away_team" in x.columns else "away"
        for r in x.itertuples(index=False):
            week=int(getattr(r,"week"))
            if week not in TARGET_WEEKS:
                continue
            h=_team(getattr(r,hc)); a=_team(getattr(r,ac))
            if h and a:
                rows.append({"season":season,"week":week,"team":h,"opponent":a})
                rows.append({"season":season,"week":week,"team":a,"opponent":h})
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero target schedule rows")
    if out.duplicated(["season","week","team"]).any():
        raise RuntimeError("duplicate schedule team-week")
    return out.sort_values(["season","week","team"]).reset_index(drop=True)


def _load_universe(schedule:pd.DataFrame)->pd.DataFrame:
    import nflreadpy as nfl
    rows=[]
    for season in TARGET_SEASONS:
        raw=_lower(_to_pandas(nfl.load_rosters_weekly(season)))
        team_col=next((c for c in ("team","team_abbr","club_code") if c in raw.columns),None)
        pos_col=next((c for c in ("position","pos","depth_chart_position","ngs_position") if c in raw.columns),None)
        name_col=next((c for c in ("full_name","football_name","player_name","player","name") if c in raw.columns),None)
        if not team_col or not pos_col or not name_col or "week" not in raw.columns:
            raise RuntimeError(f"weekly roster schema incomplete season={season}")
        raw=raw.loc[_num(raw["week"]).isin(TARGET_WEEKS)].copy()
        raw["team"]=raw[team_col].map(_team)
        raw["is_rb"]=raw[pos_col].map(_rb_position)
        raw["player"]=raw[name_col].astype("string").fillna("").str.strip()
        raw["gsis_id"]=_first(raw,["gsis_id","player_id"]).map(_clean_id)
        if "status" in raw.columns:
            st=raw["status"].astype(str).str.upper().str.strip()
            keep=st.isin({"ACT","INA"})
            if keep.any():
                raw=raw.loc[keep].copy()
        raw=raw.loc[raw["is_rb"]&raw["player"].ne("")].copy()
        sched=schedule.loc[schedule["season"].eq(season),["season","week","team","opponent"]]
        raw["season"]=season
        raw["week"]=_num(raw["week"]).astype(int)
        raw=raw.merge(sched,on=["season","week","team"],how="inner",validate="many_to_one")
        rows.extend(raw[["season","week","team","opponent","player","gsis_id"]].to_dict("records"))
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero target-week RB roster rows")
    out=out.drop_duplicates(["season","week","team","gsis_id","player"],keep="last")
    return out.sort_values(["season","week","team","player"]).reset_index(drop=True)


def _load_events()->pd.DataFrame:
    frames=[]
    for season in HISTORY_SEASONS:
        x=_lower(get_pbp(season,min_rows=1))
        if "season_type" in x.columns:
            reg=x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
            if not reg.empty:
                x=reg
        needed=["week","posteam","defteam","rusher_player_id","rush_attempt","rushing_yards","run_location"]
        miss=[c for c in needed if c not in x.columns]
        if miss:
            raise RuntimeError(f"PBP {season} missing {miss}")
        x["season"]=season
        x["week"]=_num(x["week"])
        x["team"]=x["posteam"].map(_team)
        x["defense"]=x["defteam"].map(_team)
        x["rusher_id"]=x["rusher_player_id"].map(_clean_id)
        x["rush_attempt"]=_num(x["rush_attempt"]).fillna(0)
        x["qb_kneel"]=_num(_first(x,["qb_kneel"],0)).fillna(0)
        x["qb_scramble"]=_num(_first(x,["qb_scramble"],0)).fillna(0)
        x["two_point_attempt"]=_num(_first(x,["two_point_attempt"],0)).fillna(0)
        x["rushing_yards"]=_num(x["rushing_yards"])
        x["success"]=_num(_first(x,["success"],np.nan))
        x["location"]=x["run_location"].map(run_location)
        x["game_id"]=_first(x,["game_id"],"").astype(str)
        eligible=(
            x["rush_attempt"].eq(1)
            & x["qb_kneel"].ne(1)
            & x["qb_scramble"].ne(1)
            & x["two_point_attempt"].ne(1)
            & x["rusher_id"].ne("")
            & x["team"].ne("")
            & x["defense"].ne("")
            & x["rushing_yards"].notna()
            & x["location"].isin(LOCATIONS)
        )
        y=x.loc[eligible,[
            "season","week","game_id","team","defense","rusher_id",
            "rushing_yards","success","location",
        ]].copy()
        y["time_key"]=season*100+y["week"].astype(int)
        frames.append(y)
    out=pd.concat(frames,ignore_index=True)
    if out.empty:
        raise RuntimeError("zero located RB rush events")
    return out


def _player_games(events:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for (season,week,gid,team,pid),g in events.groupby(
        ["season","week","game_id","team","rusher_id"],sort=True
    ):
        rec={
            "season":int(season),"week":int(week),"game_id":str(gid),
            "team":str(team),"rusher_id":str(pid),
            "time_key":_time_key(int(season),int(week)),
            "rushes":int(len(g)),"yards":float(g["rushing_yards"].sum()),
        }
        for loc in LOCATIONS:
            rec[f"rushes_{loc.lower()}"]=int(g["location"].eq(loc).sum())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["rusher_id","time_key","game_id"]).reset_index(drop=True)


def _defense_games(events:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for (season,week,gid,defense),g in events.groupby(
        ["season","week","game_id","defense"],sort=True
    ):
        rec={
            "season":int(season),"week":int(week),"game_id":str(gid),
            "defense":str(defense),"time_key":_time_key(int(season),int(week)),
        }
        for loc in LOCATIONS:
            q=g.loc[g["location"].eq(loc)]
            rec[f"rushes_{loc.lower()}"]=int(len(q))
            rec[f"yards_{loc.lower()}"]=float(q["rushing_yards"].sum())
            rec[f"success_n_{loc.lower()}"]=int(q["success"].notna().sum())
            rec[f"success_sum_{loc.lower()}"]=float(q["success"].dropna().sum())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["defense","time_key","game_id"]).reset_index(drop=True)


def player_state(prior_games:pd.DataFrame)->dict[str,Any]:
    if prior_games is None or prior_games.empty or not {"time_key","game_id","rushes"}.issubset(prior_games.columns):
        return {"ready":False,"prior_games":0,"located_rushes":0}
    g=prior_games.sort_values(["time_key","game_id"]).tail(PLAYER_GAME_WINDOW).copy()
    rushes=int(g["rushes"].sum())
    if len(g)<MIN_PLAYER_GAMES or rushes<MIN_PLAYER_LOCATED_RUSHES:
        return {"ready":False,"prior_games":int(len(g)),"located_rushes":rushes}
    shares={}
    for loc in LOCATIONS:
        shares[loc]=float(g[f"rushes_{loc.lower()}"].sum())/rushes
    mass=float(sum(shares.values()))
    out={
        "ready":True,"prior_games":int(len(g)),"located_rushes":rushes,
        "max_time_key":int(g["time_key"].max()),
        "located_ypc":float(g["yards"].sum()/rushes),
        "mass_gap":abs(mass-1.0),
    }
    out.update({f"share_{loc.lower()}":v for loc,v in shares.items()})
    return out


def defense_state(prior_games:pd.DataFrame)->dict[str,Any]:
    if prior_games is None or prior_games.empty or not {"time_key","game_id"}.issubset(prior_games.columns):
        return {"ready":False,"prior_games":0}
    g=prior_games.sort_values(["time_key","game_id"]).tail(DEF_GAME_WINDOW).copy()
    if len(g)<MIN_DEF_GAMES:
        return {"ready":False,"prior_games":int(len(g))}
    out={"ready":True,"prior_games":int(len(g)),"max_time_key":int(g["time_key"].max())}
    for loc in LOCATIONS:
        k=loc.lower()
        n=int(g[f"rushes_{k}"].sum())
        yards=float(g[f"yards_{k}"].sum())
        sn=int(g[f"success_n_{k}"].sum())
        ss=float(g[f"success_sum_{k}"].sum())
        out[f"rushes_{k}"]=n
        out[f"ypc_{k}"]=yards/n if n>0 else np.nan
        out[f"success_{k}"]=ss/sn if sn>0 else np.nan
        if n<MIN_DEF_RUSHES_PER_LOCATION or sn!=n:
            out["ready"]=False
    return out


def finite(v:Any)->float|None:
    try:
        x=float(v)
    except Exception:
        return None
    return x if np.isfinite(x) else None


def audit()->tuple[pd.DataFrame,dict[str,Any]]:
    schedule=_load_schedule()
    universe=_load_universe(schedule)
    events=_load_events()
    pg=_player_games(events)
    dg=_defense_games(events)
    pidx={pid:g for pid,g in pg.groupby("rusher_id",sort=False)}
    didx={team:g for team,g in dg.groupby("defense",sort=False)}

    rows=[];player_ready=0;def_ready_on_player=0
    pviol=0;dviol=0;max_gap=0.0

    for r in universe.itertuples(index=False):
        tk=_time_key(int(r.season),int(r.week))
        pid=str(r.gsis_id)
        pa=pidx.get(pid,pd.DataFrame())
        pp=pa.loc[pa["time_key"].lt(tk)].copy() if len(pa) else pd.DataFrame()
        ps=player_state(pp) if pid else {"ready":False,"prior_games":0,"located_rushes":0}
        da=didx.get(str(r.opponent),pd.DataFrame())
        dp=da.loc[da["time_key"].lt(tk)].copy() if len(da) else pd.DataFrame()
        ds=defense_state(dp)

        if ps.get("ready"):
            player_ready+=1
            max_gap=max(max_gap,float(ps.get("mass_gap",0.0)))
            pviol+=int(int(ps.get("max_time_key",-1))>=tk)
            if ds.get("ready"):
                def_ready_on_player+=1
        if ds.get("ready"):
            dviol+=int(int(ds.get("max_time_key",-1))>=tk)

        mix_ypc=np.nan;mix_success=np.nan
        if ps.get("ready") and ds.get("ready"):
            shares=np.array([float(ps[f"share_{l.lower()}"]) for l in LOCATIONS])
            ypc=np.array([float(ds[f"ypc_{l.lower()}"]) for l in LOCATIONS])
            suc=np.array([float(ds[f"success_{l.lower()}"]) for l in LOCATIONS])
            mix_ypc=float(np.dot(shares,ypc))
            mix_success=float(np.dot(shares,suc))

        rows.append(asdict(AuditRow(
            season=int(r.season),week=int(r.week),team=str(r.team),opponent=str(r.opponent),
            player=str(r.player),gsis_id=pid,stable_id=bool(pid),
            player_prior_games=int(ps.get("prior_games",0)),
            player_located_rushes=int(ps.get("located_rushes",0)),
            player_state_ready=bool(ps.get("ready",False)),
            player_feature_max_time_key=int(ps["max_time_key"]) if ps.get("ready") else None,
            player_located_ypc=finite(ps.get("located_ypc")),
            player_share_left=finite(ps.get("share_left")),
            player_share_middle=finite(ps.get("share_middle")),
            player_share_right=finite(ps.get("share_right")),
            player_location_mass_gap=finite(ps.get("mass_gap")),
            defense_prior_games=int(ds.get("prior_games",0)),
            defense_state_ready=bool(ds.get("ready",False)),
            defense_feature_max_time_key=int(ds["max_time_key"]) if ds.get("ready") else None,
            def_rushes_left=int(ds.get("rushes_left",0)),
            def_rushes_middle=int(ds.get("rushes_middle",0)),
            def_rushes_right=int(ds.get("rushes_right",0)),
            def_ypc_left=finite(ds.get("ypc_left")),
            def_ypc_middle=finite(ds.get("ypc_middle")),
            def_ypc_right=finite(ds.get("ypc_right")),
            def_success_left=finite(ds.get("success_left")),
            def_success_middle=finite(ds.get("success_middle")),
            def_success_right=finite(ds.get("success_right")),
            run_mix_def_ypc=finite(mix_ypc),
            run_mix_def_success=finite(mix_success),
            matchup_state_ready=bool(ps.get("ready") and ds.get("ready")),
        )))

    out=pd.DataFrame(rows)
    supported=out.loc[out["matchup_state_ready"]].copy()
    stable=float(out["stable_id"].mean()) if len(out) else 0.0
    def_rate=def_ready_on_player/player_ready if player_ready else 0.0
    by_season={str(int(k)):int(v) for k,v in supported.groupby("season").size().items()}
    cells=out[["season","week"]].drop_duplicates().shape[0]
    expected=len(TARGET_SEASONS)*len(TARGET_WEEKS)
    gates={
        "exact_target_seasons":sorted(out["season"].unique().tolist())==list(TARGET_SEASONS),
        "exact_target_week_cells":cells==expected,
        "nonempty_every_target_week":cells==expected,
        "stable_gsis_coverage_ge_095":stable>=0.95,
        "supported_player_weeks_ge_1200":len(supported)>=1200,
        "season_2024_supported_ge_400":by_season.get("2024",0)>=400,
        "season_2025_supported_ge_400":by_season.get("2025",0)>=400,
        "defense_support_on_player_profiles_ge_090":def_rate>=0.90,
        "player_location_mass_gap_le_1e_12":max_gap<=1e-12,
        "same_future_player_feature_violations_zero":pviol==0,
        "same_future_defense_feature_violations_zero":dviol==0,
        "sportsbook_inputs_zero":True,
        "outcome_residual_scoring_zero":True,
        "fitted_candidate_models_zero":True,
        "production_changed_false":True,
    }
    qualified=all(bool(v) for v in gates.values())
    result={
        "version":VERSION,
        "disposition":(
            "RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_SOURCE_QUALIFIED"
            if qualified else
            "RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_SOURCE_NOT_QUALIFIED"
        ),
        "target_seasons":list(TARGET_SEASONS),"target_weeks":[2,18],
        "universe_rows":int(len(out)),
        "stable_gsis_rows":int(out["stable_id"].sum()),
        "stable_gsis_coverage":stable,
        "player_profile_ready_rows":int(player_ready),
        "defense_ready_on_player_profile_rows":int(def_ready_on_player),
        "defense_support_on_player_profiles":def_rate,
        "matchup_state_ready_rows":int(len(supported)),
        "support_by_season":by_season,
        "max_player_location_mass_gap":max_gap,
        "same_future_player_feature_violations":int(pviol),
        "same_future_defense_feature_violations":int(dviol),
        "sportsbook_inputs_used":0,
        "football_outcomes_or_residuals_scored":0,
        "candidate_models_fit":0,
        "production_changed":False,
        "gates":gates,
    }
    return out,result


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    rows,result=audit()
    rows.to_csv(a.out_dir/"rb_individual_run_location_matchup_source_v1_rows.csv",index=False,float_format="%.12f")
    (a.out_dir/"rb_individual_run_location_matchup_source_v1_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
