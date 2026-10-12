#!/usr/bin/env python3
"""Source-readiness audit for individual receiver depth x defense depth matchup.

No target-game outcome scoring, residuals, sportsbook inputs, fitted predictor,
or production changes. The audit reconstructs only strictly-prior player and
opponent states for historical target-week roster rows.
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

VERSION="PLAYER_ROLE_MATCHUP_DEPTH_INTERACTION_SOURCE_V1"
TARGET_SEASONS=(2024,2025)
TARGET_WEEKS=tuple(range(2,19))
HISTORY_SEASONS=(2023,2024,2025)
PLAYER_GAME_WINDOW=8
DEF_GAME_WINDOW=8
MIN_PLAYER_GAMES=4
MIN_PLAYER_TARGETS=10
MIN_DEF_GAMES=4
MIN_DEF_TARGETS_PER_BAND=5
BANDS=("BEHIND","SHORT","INTERMEDIATE","DEEP")


@dataclass
class AuditRow:
    season:int
    week:int
    team:str
    opponent:str
    position:str
    player:str
    gsis_id:str
    stable_id:bool
    player_prior_games:int
    player_prior_targets:int
    player_state_ready:bool
    player_feature_max_time_key:int | None
    player_mean_air_yards:float | None
    player_air_yards_sd:float | None
    player_share_behind:float | None
    player_share_short:float | None
    player_share_intermediate:float | None
    player_share_deep:float | None
    player_band_mass_gap:float | None
    defense_prior_games:int
    defense_state_ready:bool
    defense_feature_max_time_key:int | None
    def_targets_behind:int
    def_targets_short:int
    def_targets_intermediate:int
    def_targets_deep:int
    def_comp_behind:float | None
    def_comp_short:float | None
    def_comp_intermediate:float | None
    def_comp_deep:float | None
    def_ypt_behind:float | None
    def_ypt_short:float | None
    def_ypt_intermediate:float | None
    def_ypt_deep:float | None
    depth_mix_def_ypt:float | None
    depth_mix_def_completion:float | None
    matchup_state_ready:bool


def _to_pandas(obj:Any)->pd.DataFrame:
    if isinstance(obj,pd.DataFrame):
        return obj.copy()
    # Avoid Polars -> pandas Arrow conversion here. Repository CI currently
    # carries NumPy 1.x while newer pyarrow wheels require NumPy 2.x.
    # to_dicts() preserves the source values without adding an Arrow runtime
    # dependency to this source-only audit.
    if hasattr(obj,"to_dicts"):
        return pd.DataFrame(obj.to_dicts())
    if hasattr(obj,"to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _lower(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy()
    out.columns=[str(c).strip().lower() for c in out.columns]
    return out


def _num(s:pd.Series)->pd.Series:
    return pd.to_numeric(s,errors="coerce")


def _clean_id(v:Any)->str:
    if v is None or pd.isna(v):
        return ""
    s=str(v).strip()
    return "" if s.lower() in {"","nan","none","<na>"} else s


def _first(df:pd.DataFrame,names:list[str],default:Any="")->pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default,index=df.index)


def _team(v:Any)->str:
    try:
        return canon_team(v)
    except Exception:
        return str(v or "").strip().upper()


def _position(v:Any)->str:
    p=str(v or "").strip().upper()
    if p in {"WR","LWR","RWR","SWR"} or "WR" in p:
        return "WR"
    if p=="TE" or p.startswith("TE"):
        return "TE"
    return ""


def depth_band(v:Any)->str:
    try:
        x=float(v)
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    if x<0:
        return "BEHIND"
    if x<10:
        return "SHORT"
    if x<20:
        return "INTERMEDIATE"
    return "DEEP"


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
            home=_team(getattr(r,hc))
            away=_team(getattr(r,ac))
            if home and away:
                rows.append({"season":season,"week":week,"team":home,"opponent":away})
                rows.append({"season":season,"week":week,"team":away,"opponent":home})
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero target schedule rows")
    if out.duplicated(["season","week","team"]).any():
        raise RuntimeError("duplicate target schedule team-week")
    return out.sort_values(["season","week","team"]).reset_index(drop=True)


def _load_universe(schedule:pd.DataFrame)->pd.DataFrame:
    import nflreadpy as nfl
    rows=[]
    for season in TARGET_SEASONS:
        raw=_lower(_to_pandas(nfl.load_rosters_weekly(season)))
        team_col="team" if "team" in raw.columns else "team_abbr" if "team_abbr" in raw.columns else "club_code"
        name_col="full_name" if "full_name" in raw.columns else "football_name" if "football_name" in raw.columns else "player_name"
        pos_col="position" if "position" in raw.columns else "pos" if "pos" in raw.columns else "depth_chart_position"
        if not {"season","week",team_col,pos_col,name_col}.issubset(raw.columns):
            raise RuntimeError(f"weekly roster missing required fields season={season}")
        raw=raw.loc[_num(raw["season"]).eq(season)&_num(raw["week"]).isin(TARGET_WEEKS)].copy()
        raw["team"]=raw[team_col].map(_team)
        raw["position_family"]=raw[pos_col].map(_position)
        raw["player"]=raw[name_col].astype("string").fillna("").str.strip()
        raw["gsis_id"]=_first(raw,["gsis_id","player_id"]).map(_clean_id)
        if "status" in raw.columns:
            status=raw["status"].astype(str).str.upper().str.strip()
            keep=status.isin({"ACT","INA"})
            if keep.any():
                raw=raw.loc[keep].copy()
        raw=raw.loc[raw["position_family"].isin({"WR","TE"})&raw["player"].ne("")].copy()
        sched=schedule.loc[schedule["season"].eq(season),["season","week","team","opponent"]]
        raw=raw.merge(sched,on=["season","week","team"],how="inner",validate="many_to_one")
        for r in raw.itertuples(index=False):
            rows.append({
                "season":season,
                "week":int(r.week),
                "team":str(r.team),
                "opponent":str(r.opponent),
                "position":str(r.position_family),
                "player":str(r.player),
                "gsis_id":str(r.gsis_id),
            })
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero WR/TE target-week roster rows")
    out=out.drop_duplicates(["season","week","team","gsis_id","player"],keep="last")
    return out.sort_values(["season","week","team","position","player"]).reset_index(drop=True)


def _load_target_events()->pd.DataFrame:
    frames=[]
    for season in HISTORY_SEASONS:
        x=_lower(get_pbp(season,min_rows=1))
        if "season_type" in x.columns:
            reg=x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
            if not reg.empty:
                x=reg
        required=["week","posteam","defteam","receiver_player_id","air_yards","pass_attempt","sack"]
        miss=[c for c in required if c not in x.columns]
        if miss:
            raise RuntimeError(f"PBP {season} missing {miss}")
        x["season"]=season
        x["week"]=_num(x["week"])
        x["team"]=x["posteam"].map(_team)
        x["defense"]=x["defteam"].map(_team)
        x["receiver_id"]=x["receiver_player_id"].map(_clean_id)
        x["air_yards"]=_num(x["air_yards"])
        x["pass_attempt"]=_num(x["pass_attempt"]).fillna(0)
        x["sack"]=_num(x["sack"]).fillna(0)
        x["two_point_attempt"]=_num(_first(x,["two_point_attempt"],0)).fillna(0)
        x["complete"]=_num(_first(x,["complete_pass"],0)).fillna(0).eq(1)
        yard_src=_first(x,["receiving_yards","passing_yards","yards_gained"],0)
        x["receiving_yards"]=_num(yard_src).fillna(0.0)
        x.loc[~x["complete"],"receiving_yards"]=0.0
        x["game_id"]=_first(x,["game_id"],"").astype(str)
        eligible=(
            x["pass_attempt"].eq(1)
            & ~x["sack"].eq(1)
            & ~x["two_point_attempt"].eq(1)
            & x["receiver_id"].ne("")
            & x["team"].ne("")
            & x["defense"].ne("")
            & x["air_yards"].notna()
        )
        y=x.loc[eligible,[
            "season","week","game_id","team","defense","receiver_id",
            "air_yards","complete","receiving_yards",
        ]].copy()
        y["band"]=y["air_yards"].map(depth_band)
        y["time_key"]=season*100+y["week"].astype(int)
        frames.append(y)
    out=pd.concat(frames,ignore_index=True)
    if out.empty:
        raise RuntimeError("zero finite-air target events")
    return out


def _player_games(events:pd.DataFrame)->pd.DataFrame:
    rows=[]
    keys=["season","week","game_id","team","receiver_id"]
    for key,g in events.groupby(keys,sort=True):
        season,week,game_id,team,pid=key
        rec={
            "season":int(season),"week":int(week),"game_id":str(game_id),
            "team":str(team),"receiver_id":str(pid),
            "time_key":_time_key(int(season),int(week)),
            "targets":int(len(g)),
            "air_sum":float(g["air_yards"].sum()),
            "air_sq_sum":float(np.square(g["air_yards"].to_numpy(float)).sum()),
        }
        for band in BANDS:
            rec[f"targets_{band.lower()}"]=int(g["band"].eq(band).sum())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["receiver_id","time_key","game_id"]).reset_index(drop=True)


def _defense_games(events:pd.DataFrame)->pd.DataFrame:
    rows=[]
    keys=["season","week","game_id","defense"]
    for key,g in events.groupby(keys,sort=True):
        season,week,game_id,defense=key
        rec={
            "season":int(season),"week":int(week),"game_id":str(game_id),
            "defense":str(defense),"time_key":_time_key(int(season),int(week)),
        }
        for band in BANDS:
            q=g.loc[g["band"].eq(band)]
            rec[f"targets_{band.lower()}"]=int(len(q))
            rec[f"completions_{band.lower()}"]=int(q["complete"].sum())
            rec[f"yards_{band.lower()}"]=float(q["receiving_yards"].sum())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["defense","time_key","game_id"]).reset_index(drop=True)


def player_state(prior_games:pd.DataFrame)->dict[str,Any]:
    g=prior_games.sort_values(["time_key","game_id"]).tail(PLAYER_GAME_WINDOW).copy()
    if len(g)<MIN_PLAYER_GAMES:
        return {"ready":False,"prior_games":int(len(g)),"prior_targets":int(g["targets"].sum()) if len(g) else 0}
    total=int(g["targets"].sum())
    if total<MIN_PLAYER_TARGETS:
        return {"ready":False,"prior_games":int(len(g)),"prior_targets":total}
    air_sum=float(g["air_sum"].sum())
    air_sq=float(g["air_sq_sum"].sum())
    mean=air_sum/total
    var=max(0.0,air_sq/total-mean*mean)
    shares={}
    for band in BANDS:
        shares[band]=float(g[f"targets_{band.lower()}"].sum())/total
    mass=float(sum(shares.values()))
    out={
        "ready":True,"prior_games":int(len(g)),"prior_targets":total,
        "max_time_key":int(g["time_key"].max()),
        "mean_air_yards":mean,"air_yards_sd":float(np.sqrt(var)),
        "mass_gap":abs(mass-1.0),
    }
    out.update({f"share_{k.lower()}":v for k,v in shares.items()})
    return out


def defense_state(prior_games:pd.DataFrame)->dict[str,Any]:
    g=prior_games.sort_values(["time_key","game_id"]).tail(DEF_GAME_WINDOW).copy()
    if len(g)<MIN_DEF_GAMES:
        return {"ready":False,"prior_games":int(len(g))}
    out={"ready":True,"prior_games":int(len(g)),"max_time_key":int(g["time_key"].max())}
    for band in BANDS:
        n=int(g[f"targets_{band.lower()}"].sum())
        comp=int(g[f"completions_{band.lower()}"].sum())
        yards=float(g[f"yards_{band.lower()}"].sum())
        out[f"targets_{band.lower()}"]=n
        out[f"comp_{band.lower()}"]=(comp/n if n>0 else np.nan)
        out[f"ypt_{band.lower()}"]=(yards/n if n>0 else np.nan)
        if n<MIN_DEF_TARGETS_PER_BAND:
            out["ready"]=False
    return out


def _finite_or_none(v:Any)->float | None:
    try:
        x=float(v)
    except Exception:
        return None
    return x if np.isfinite(x) else None


def audit()->tuple[pd.DataFrame,dict[str,Any]]:
    schedule=_load_schedule()
    universe=_load_universe(schedule)
    events=_load_target_events()
    pg=_player_games(events)
    dg=_defense_games(events)

    player_groups={pid:g for pid,g in pg.groupby("receiver_id",sort=False)}
    defense_groups={tm:g for tm,g in dg.groupby("defense",sort=False)}

    rows=[]
    player_qualified=0
    full_def_on_player=0
    player_future_violations=0
    defense_future_violations=0
    max_mass_gap=0.0

    for r in universe.itertuples(index=False):
        tk=_time_key(int(r.season),int(r.week))
        pid=str(r.gsis_id)
        p_all=player_groups.get(pid,pd.DataFrame())
        p_prior=p_all.loc[p_all["time_key"].lt(tk)].copy() if len(p_all) else pd.DataFrame()
        ps=player_state(p_prior) if pid else {"ready":False,"prior_games":0,"prior_targets":0}
        d_all=defense_groups.get(str(r.opponent),pd.DataFrame())
        d_prior=d_all.loc[d_all["time_key"].lt(tk)].copy() if len(d_all) else pd.DataFrame()
        ds=defense_state(d_prior) if len(d_prior) else {"ready":False,"prior_games":0}

        if ps.get("ready"):
            player_qualified+=1
            max_mass_gap=max(max_mass_gap,float(ps.get("mass_gap",0.0)))
            player_future_violations+=int(int(ps.get("max_time_key",-1))>=tk)
            if ds.get("ready"):
                full_def_on_player+=1
        if ds.get("ready"):
            defense_future_violations+=int(int(ds.get("max_time_key",-1))>=tk)

        mix_ypt=np.nan
        mix_comp=np.nan
        if ps.get("ready") and ds.get("ready"):
            mix_ypt=sum(
                float(ps[f"share_{band.lower()}"])*float(ds[f"ypt_{band.lower()}"])
                for band in BANDS
            )
            mix_comp=sum(
                float(ps[f"share_{band.lower()}"])*float(ds[f"comp_{band.lower()}"])
                for band in BANDS
            )

        row=AuditRow(
            season=int(r.season),week=int(r.week),team=str(r.team),opponent=str(r.opponent),
            position=str(r.position),player=str(r.player),gsis_id=pid,stable_id=bool(pid),
            player_prior_games=int(ps.get("prior_games",0)),
            player_prior_targets=int(ps.get("prior_targets",0)),
            player_state_ready=bool(ps.get("ready",False)),
            player_feature_max_time_key=(int(ps["max_time_key"]) if ps.get("ready") else None),
            player_mean_air_yards=_finite_or_none(ps.get("mean_air_yards")),
            player_air_yards_sd=_finite_or_none(ps.get("air_yards_sd")),
            player_share_behind=_finite_or_none(ps.get("share_behind")),
            player_share_short=_finite_or_none(ps.get("share_short")),
            player_share_intermediate=_finite_or_none(ps.get("share_intermediate")),
            player_share_deep=_finite_or_none(ps.get("share_deep")),
            player_band_mass_gap=_finite_or_none(ps.get("mass_gap")),
            defense_prior_games=int(ds.get("prior_games",0)),
            defense_state_ready=bool(ds.get("ready",False)),
            defense_feature_max_time_key=(int(ds["max_time_key"]) if ds.get("ready") else None),
            def_targets_behind=int(ds.get("targets_behind",0)),
            def_targets_short=int(ds.get("targets_short",0)),
            def_targets_intermediate=int(ds.get("targets_intermediate",0)),
            def_targets_deep=int(ds.get("targets_deep",0)),
            def_comp_behind=_finite_or_none(ds.get("comp_behind")),
            def_comp_short=_finite_or_none(ds.get("comp_short")),
            def_comp_intermediate=_finite_or_none(ds.get("comp_intermediate")),
            def_comp_deep=_finite_or_none(ds.get("comp_deep")),
            def_ypt_behind=_finite_or_none(ds.get("ypt_behind")),
            def_ypt_short=_finite_or_none(ds.get("ypt_short")),
            def_ypt_intermediate=_finite_or_none(ds.get("ypt_intermediate")),
            def_ypt_deep=_finite_or_none(ds.get("ypt_deep")),
            depth_mix_def_ypt=_finite_or_none(mix_ypt),
            depth_mix_def_completion=_finite_or_none(mix_comp),
            matchup_state_ready=bool(ps.get("ready") and ds.get("ready")),
        )
        rows.append(asdict(row))

    out=pd.DataFrame(rows)
    stable_id_cov=float(out["stable_id"].mean()) if len(out) else 0.0
    supported=out.loc[out["matchup_state_ready"]].copy()
    support_by_season={str(int(k)):int(v) for k,v in supported.groupby("season").size().items()}
    support_by_position={str(k):int(v) for k,v in supported.groupby("position").size().items()}
    def_support_rate=(full_def_on_player/player_qualified if player_qualified else 0.0)

    target_week_cells=(
        out[["season","week"]].drop_duplicates().shape[0]
    )
    expected_cells=len(TARGET_SEASONS)*len(TARGET_WEEKS)

    gates={
        "exact_target_seasons":sorted(out["season"].unique().tolist())==list(TARGET_SEASONS),
        "exact_target_week_cells":target_week_cells==expected_cells,
        "nonempty_every_target_week":target_week_cells==expected_cells,
        "stable_gsis_coverage_ge_095":stable_id_cov>=0.95,
        "supported_player_weeks_ge_1500":len(supported)>=1500,
        "season_2024_supported_ge_500":support_by_season.get("2024",0)>=500,
        "season_2025_supported_ge_500":support_by_season.get("2025",0)>=500,
        "wr_supported_ge_1000":support_by_position.get("WR",0)>=1000,
        "te_supported_ge_300":support_by_position.get("TE",0)>=300,
        "defense_support_on_player_profiles_ge_090":def_support_rate>=0.90,
        "player_band_mass_gap_le_1e_12":max_mass_gap<=1e-12,
        "same_future_player_feature_violations_zero":player_future_violations==0,
        "same_future_defense_feature_violations_zero":defense_future_violations==0,
        "sportsbook_inputs_zero":True,
        "outcome_residual_scoring_zero":True,
        "fitted_candidate_models_zero":True,
        "production_changed_false":True,
    }
    qualified=all(bool(v) for v in gates.values())
    summary={
        "version":VERSION,
        "disposition":"PLAYER_DEPTH_MATCHUP_SOURCE_QUALIFIED" if qualified else "PLAYER_DEPTH_MATCHUP_SOURCE_NOT_QUALIFIED",
        "target_seasons":list(TARGET_SEASONS),
        "target_weeks":[min(TARGET_WEEKS),max(TARGET_WEEKS)],
        "universe_rows":int(len(out)),
        "stable_gsis_rows":int(out["stable_id"].sum()),
        "stable_gsis_coverage":stable_id_cov,
        "player_profile_ready_rows":int(player_qualified),
        "defense_ready_on_player_profile_rows":int(full_def_on_player),
        "defense_support_on_player_profiles":def_support_rate,
        "matchup_state_ready_rows":int(len(supported)),
        "support_by_season":support_by_season,
        "support_by_position":support_by_position,
        "max_player_band_mass_gap":max_mass_gap,
        "same_future_player_feature_violations":int(player_future_violations),
        "same_future_defense_feature_violations":int(defense_future_violations),
        "sportsbook_inputs_used":0,
        "football_outcomes_or_residuals_scored":0,
        "candidate_models_fit":0,
        "production_changed":False,
        "gates":gates,
    }
    return out,summary


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    rows,summary=audit()
    rows.to_csv(a.out_dir/"player_depth_matchup_source_v1_rows.csv",index=False,float_format="%.12f")
    (a.out_dir/"player_depth_matchup_source_v1_result.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(summary,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
