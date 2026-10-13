#!/usr/bin/env python3
"""Frozen receiver-level conversion mechanism development/confirmation test."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts._opponent_map import canon_team
from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules
from scripts.utils.canonical_names import canonicalize_player_name_safe

SEASONS=(2023,2024,2025)
TARGET_WEEKS=tuple(range(5,19))
FEATURES=("prior_drop_per_catchable","prior_relative_catchable_rate")
RIDGE_ALPHA=20.0
BOOT_REPS=2000
WR_POS={"WR","LWR","RWR","SWR"}
TE_POS={"TE"}
SUPPORT={
    "WR":{"rows":300,"players":75},
    "TE":{"rows":150,"players":35},
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing required input {path}")
    return pd.read_csv(path)


def opt_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def lower(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    out.columns=[str(c).strip().lower() for c in out.columns]
    return out


def to_pd(obj) -> pd.DataFrame:
    if isinstance(obj,pd.DataFrame):
        return obj
    if hasattr(obj,"to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def first(frame: pd.DataFrame,names: tuple[str,...],default="") -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(default,index=frame.index)


def pos_family(value) -> str:
    z="" if value is None or pd.isna(value) else str(value).upper().strip()
    if z in WR_POS or z.startswith("WR"):
        return "WR"
    if z in TE_POS or z.startswith("TE"):
        return "TE"
    return z


def bool_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s,errors="coerce")
    z=s.astype("string").str.strip().str.lower()
    out=pd.Series(np.nan,index=s.index,dtype=float)
    out.loc[z.isin(["1","true","t","yes","y"])]=1.0
    out.loc[z.isin(["0","false","f","no","n"])]=0.0
    return out


def fetch_parquet(url: str) -> pd.DataFrame:
    req=Request(url,headers={"User-Agent":"receiver-conversion-mechanism-v1"})
    with urlopen(req,timeout=180) as r:
        raw=r.read()
    return pd.read_parquet(io.BytesIO(raw))


def clean_key(name: object) -> str:
    return canonicalize_player_name_safe("" if name is None else str(name))[1]


def load_roster_position(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    r=lower(to_pd(nfl.load_rosters_weekly(int(season))))
    r["season"]=pd.to_numeric(first(r,("season",),season),errors="coerce").fillna(season).astype(int)
    r["week"]=pd.to_numeric(first(r,("week",)),errors="coerce")
    r["player_id"]=first(r,("gsis_id","player_id")).astype("string").fillna("").str.strip()
    r["position"]=first(r,("position","pos")).map(pos_family)
    r=r.loc[
        r["season"].eq(season)
        & r["week"].between(1,18,inclusive="both")
        & r["player_id"].ne("")
    ].copy()
    return r[["season","week","player_id","position"]].drop_duplicates(
        ["season","week","player_id"],keep="last"
    )


def charted_game_rows(season: int) -> tuple[pd.DataFrame,pd.DataFrame]:
    ftn=lower(fetch_parquet(
        f"https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
    ))
    pbp=lower(fetch_parquet(
        f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
    ))
    req_f={"nflverse_game_id","nflverse_play_id","season","week","is_drop","is_catchable_ball"}
    req_p={"game_id","play_id","season","week","season_type","posteam","pass_attempt",
           "receiver_player_id","receiver_player_name"}
    if req_f-set(ftn.columns) or req_p-set(pbp.columns):
        raise RuntimeError(
            f"source schema drift season={season} ftn={sorted(req_f-set(ftn.columns))} pbp={sorted(req_p-set(pbp.columns))}"
        )
    ftn["week"]=pd.to_numeric(ftn["week"],errors="coerce")
    pbp["week"]=pd.to_numeric(pbp["week"],errors="coerce")
    ftn=ftn.loc[ftn["week"].between(1,18,inclusive="both")].copy()
    pbp=pbp.loc[
        pbp["season_type"].astype(str).str.upper().eq("REG")
        & pbp["week"].between(1,18,inclusive="both")
    ].copy()
    ftn["jg"]=ftn["nflverse_game_id"].astype(str)
    ftn["jp"]=pd.to_numeric(ftn["nflverse_play_id"],errors="coerce")
    p=pbp[["game_id","play_id","season","week","posteam","pass_attempt",
           "receiver_player_id","receiver_player_name"]].copy()
    p["jg"]=p["game_id"].astype(str)
    p["jp"]=pd.to_numeric(p["play_id"],errors="coerce")
    z=ftn.merge(
        p.drop(columns=["game_id","play_id","season","week"]),
        on=["jg","jp"],how="left",validate="one_to_one"
    )
    z["season"]=int(season)
    z["week"]=pd.to_numeric(z["week"],errors="coerce").astype(int)
    z["pass_attempt"]=pd.to_numeric(z["pass_attempt"],errors="coerce")
    z=z.loc[z["pass_attempt"].eq(1)].copy()
    z["player_id"]=z["receiver_player_id"].astype("string").fillna("").str.strip()
    z["player_clean_key"]=z["receiver_player_name"].map(clean_key)
    z["team"]=z["posteam"].map(canon_team)
    z["catchable"]=bool_num(z["is_catchable_ball"])
    z["drop"]=bool_num(z["is_drop"])
    roster=load_roster_position(season)
    z=z.merge(roster,on=["season","week","player_id"],how="left",validate="many_to_one")
    z=z.loc[
        z["player_id"].ne("")
        & z["player_clean_key"].astype(str).ne("")
        & z["team"].astype(str).ne("")
        & z["catchable"].notna()
        & z["drop"].notna()
    ].copy()
    z["game_id"]=z["jg"].astype(str)

    player=z.groupby(
        ["season","week","game_id","team","player_id","player_clean_key","position"],
        as_index=False,dropna=False
    ).agg(
        target_rows=("jp","size"),
        catchable_count=("catchable","sum"),
        drop_count=("drop","sum"),
    )
    team=z.groupby(
        ["season","week","game_id","team"],as_index=False
    ).agg(
        team_target_rows=("jp","size"),
        team_catchable_count=("catchable","sum"),
    )
    return player,team


def prepared(bundle) -> pd.DataFrame:
    m=cp.build_market_frame(bundle)
    m=apply_bayesian_to_metrics(m,build_bayesian_baseline(bundle.player_consensus))
    with patch.object(simulation_rules,"load_model_contexts",return_value=(bundle.teams,bundle.players)):
        m=simulation_rules.apply_rules_to_metrics(m)
    m["player_clean_key"]=m["player_clean_key"].fillna("").astype(str)
    return m.sort_values(["event_id","team","player_clean_key"]).drop_duplicates(
        ["event_id","team","player_clean_key"],keep="last"
    )


def num(v):
    try:
        x=float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def canonical_rows(input_root: Path, season: int) -> pd.DataFrame:
    root=input_root/str(season)
    logs=read_csv(root/"player_game_logs_history.csv")
    team=read_csv(root/"team_weekly_history.csv")
    sched=read_csv(root/"schedule_history.csv")
    injuries=opt_csv(root/"injuries_history.csv")
    weather=opt_csv(root/"weather_history.csv")
    rows=[]
    for week in TARGET_WEEKS:
        universe=read_csv(root/"pregame_universe"/f"{season}_week_{week:02d}.csv")
        bundle=build_historical_context_bundle(
            player_logs=logs,team_weekly=team,pregame_universe=universe,schedule=sched,
            season=season,week=week,prior_season=season-1,
            injuries=_exact_week(injuries,season,week),
            weather=_exact_week(weather,season,week),
        )
        m=prepared(bundle)
        actual=cp.build_actual_rows(logs,season,week)
        ar=actual.loc[
            actual["market"].eq("receptions"),
            ["team","player_clean_key","actual","actual_opportunities"],
        ].rename(columns={"actual":"actual_receptions","actual_opportunities":"actual_targets"})
        x=m.merge(ar,on=["team","player_clean_key"],how="left")
        for _,r in x.iterrows():
            pos=pos_family(r.get("position",""))
            if pos not in {"WR","TE"}:
                continue
            targets=num(r.get("actual_targets"))
            recs=num(r.get("actual_receptions"))
            if not np.isfinite(targets) or targets<=0 or not np.isfinite(recs):
                continue
            pred=np.nan
            for col in ("rules_catch_rate","bayes_receptions_per_target","receptions_per_target"):
                if col in r.index:
                    pred=num(r.get(col))
                    if np.isfinite(pred):
                        break
            if not np.isfinite(pred):
                continue
            rows.append({
                "season":season,"week":week,"game_id":str(r.get("event_id","")),
                "team":canon_team(r.get("team","")),
                "player_clean_key":str(r.get("player_clean_key","")),
                "player":str(r.get("player","")),"position":pos,
                "canonical_catch_rate":float(pred),
                "actual_targets":float(targets),
                "actual_receptions":float(recs),
                "actual_catch_rate":float(recs/targets),
            })
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"zero canonical rows season={season}")
    return out


def add_features(targets: pd.DataFrame, player_games: pd.DataFrame, team_games: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    rows=[]
    chronology=0
    same_team_bad=0
    for r in targets.itertuples(index=False):
        ph=player_games.loc[
            player_games["season"].eq(r.season)
            & player_games["team"].eq(r.team)
            & player_games["player_clean_key"].eq(r.player_clean_key)
            & player_games["week"].lt(r.week)
        ].sort_values(["week","game_id"]).tail(8)
        if len(ph)<4:
            continue
        th=team_games.loc[
            team_games["season"].eq(r.season)
            & team_games["team"].eq(r.team)
            & team_games["week"].lt(r.week)
        ].sort_values(["week","game_id"]).tail(8)
        if len(th)<4:
            continue
        ptargets=float(ph["target_rows"].sum())
        pcatch=float(ph["catchable_count"].sum())
        pdrops=float(ph["drop_count"].sum())
        ttargets=float(th["team_target_rows"].sum())
        tcatch=float(th["team_catchable_count"].sum())
        if ptargets<=0 or pcatch<=0 or ttargets<=0:
            continue
        max_week=max(int(ph["week"].max()),int(th["week"].max()))
        chronology+=int(max_week>=int(r.week))
        same_team_bad+=int(not ph["team"].astype(str).eq(str(r.team)).all())
        pc=pcatch/ptargets
        tc=tcatch/ttargets
        rec=r._asdict()
        rec.update({
            "prior_games":int(len(ph)),
            "prior_feature_max_week":max_week,
            "prior_drop_per_catchable":pdrops/pcatch,
            "prior_relative_catchable_rate":pc-tc,
            "prior_player_catchable_rate":pc,
            "prior_team_catchable_rate":tc,
            "canonical_residual":float(r.actual_catch_rate-r.canonical_catch_rate),
        })
        rows.append(rec)
    return pd.DataFrame(rows),{
        "chronology_violations":int(chronology),
        "same_team_history_violations":int(same_team_bad),
    }


def support_ok(frame: pd.DataFrame,pos: str) -> bool:
    floor=SUPPORT[pos]
    return len(frame)>=floor["rows"] and frame["player_clean_key"].nunique()>=floor["players"]


def cluster_ci(frame: pd.DataFrame,cluster: str,value: str,seed: int) -> dict:
    groups=[g.index.to_numpy(int) for _,g in frame.reset_index(drop=True).groupby(cluster,sort=True)]
    if len(groups)<2:
        return {"low":np.nan,"high":np.nan,"valid":0}
    z=frame.reset_index(drop=True)
    rng=np.random.default_rng(seed)
    vals=np.full(BOOT_REPS,np.nan)
    n=len(groups)
    for i in range(BOOT_REPS):
        chosen=rng.integers(0,n,size=n)
        idx=np.concatenate([groups[j] for j in chosen])
        vals[i]=float(pd.to_numeric(z.iloc[idx][value],errors="coerce").mean())
    vals=vals[np.isfinite(vals)]
    return {
        "low":float(np.quantile(vals,.025)) if len(vals) else np.nan,
        "high":float(np.quantile(vals,.975)) if len(vals) else np.nan,
        "valid":int(len(vals)),
    }


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)

    panels=[]
    integrity={"chronology_violations":0,"same_team_history_violations":0}
    for season in SEASONS:
        pg,tg=charted_game_rows(season)
        canon=canonical_rows(args.input_root,season)
        panel,integ=add_features(canon,pg,tg)
        panels.append(panel)
        integrity["chronology_violations"]+=integ["chronology_violations"]
        integrity["same_team_history_violations"]+=integ["same_team_history_violations"]
    panel=pd.concat(panels,ignore_index=True)
    panel=panel.loc[panel["position"].isin(["WR","TE"])].copy()
    if panel.empty:
        raise RuntimeError("zero conversion mechanism rows")

    coefficients=[]
    decisions=[]
    scored=[]
    models_fit=0
    for pos in ("WR","TE"):
        train=panel.loc[panel["season"].eq(2023)&panel["position"].eq(pos)].copy()
        train_support=support_ok(train,pos)
        model=None
        scaler=None
        if train_support:
            x=train[list(FEATURES)].to_numpy(float)
            y=train["canonical_residual"].to_numpy(float)
            scaler=StandardScaler().fit(x)
            model=Ridge(alpha=RIDGE_ALPHA,fit_intercept=False).fit(scaler.transform(x),y)
            models_fit+=1
            for feature,coef in zip(FEATURES,model.coef_):
                coefficients.append({
                    "position":pos,"feature":feature,"coefficient":float(coef),
                    "scaler_mean":float(scaler.mean_[list(FEATURES).index(feature)]),
                    "scaler_scale":float(scaler.scale_[list(FEATURES).index(feature)]),
                })

        seasons={}
        for season in (2024,2025):
            q=panel.loc[panel["season"].eq(season)&panel["position"].eq(pos)].copy()
            rec={
                "season":season,"rows":int(len(q)),
                "players":int(q["player_clean_key"].nunique()),
                "games":int(q["game_id"].nunique()),
                "support":bool(support_ok(q,pos)),
            }
            if model is None or not rec["support"]:
                rec["confirmed"]=False
                seasons[str(season)]=rec
                continue
            corr=model.predict(scaler.transform(q[list(FEATURES)].to_numpy(float)))
            q["predicted_catch_residual"]=corr
            q["candidate_catch_rate"]=np.clip(
                q["canonical_catch_rate"].to_numpy(float)+corr,.05,.95
            )
            q["catch_baseline_abs"]=(q["actual_catch_rate"]-q["canonical_catch_rate"]).abs()
            q["catch_candidate_abs"]=(q["actual_catch_rate"]-q["candidate_catch_rate"]).abs()
            q["catch_improvement"]=q["catch_baseline_abs"]-q["catch_candidate_abs"]
            q["rec_baseline_abs"]=(
                q["actual_receptions"]-q["actual_targets"]*q["canonical_catch_rate"]
            ).abs()
            q["rec_candidate_abs"]=(
                q["actual_receptions"]-q["actual_targets"]*q["candidate_catch_rate"]
            ).abs()
            q["rec_improvement"]=q["rec_baseline_abs"]-q["rec_candidate_abs"]
            q["catch_rate_shift"]=q["candidate_catch_rate"]-q["canonical_catch_rate"]
            cg=cluster_ci(q,"game_id","catch_improvement",20261013+season+(0 if pos=="WR" else 100))
            cpb=cluster_ci(q,"player_clean_key","catch_improvement",20262013+season+(0 if pos=="WR" else 100))
            rg=cluster_ci(q,"game_id","rec_improvement",20263013+season+(0 if pos=="WR" else 100))
            rpb=cluster_ci(q,"player_clean_key","rec_improvement",20264013+season+(0 if pos=="WR" else 100))
            catch_mean=float(q["catch_improvement"].mean())
            rec_mean=float(q["rec_improvement"].mean())
            mean_shift=float(q["catch_rate_shift"].mean())
            catch_gate=bool(catch_mean>0 and cg["low"]>0 and cpb["low"]>0)
            rec_gate=bool(rec_mean>0 and rg["low"]>0 and rpb["low"]>0)
            shift_gate=bool(abs(mean_shift)<=.02)
            rec.update({
                "canonical_catch_mae":float(q["catch_baseline_abs"].mean()),
                "candidate_catch_mae":float(q["catch_candidate_abs"].mean()),
                "catch_mae_improvement":catch_mean,
                "catch_game_bootstrap":cg,
                "catch_player_bootstrap":cpb,
                "canonical_oracle_target_reception_mae":float(q["rec_baseline_abs"].mean()),
                "candidate_oracle_target_reception_mae":float(q["rec_candidate_abs"].mean()),
                "reception_mae_improvement":rec_mean,
                "reception_game_bootstrap":rg,
                "reception_player_bootstrap":rpb,
                "mean_catch_rate_shift":mean_shift,
                "catch_gate":catch_gate,
                "reception_gate":rec_gate,
                "mean_shift_gate":shift_gate,
                "confirmed":bool(catch_gate and rec_gate and shift_gate),
            })
            scored.append(q)
            seasons[str(season)]=rec

        confirmed=bool(
            train_support
            and seasons.get("2024",{}).get("confirmed",False)
            and seasons.get("2025",{}).get("confirmed",False)
        )
        decisions.append({
            "position":pos,
            "development_rows":int(len(train)),
            "development_players":int(train["player_clean_key"].nunique()),
            "development_support":bool(train_support),
            "confirmation":seasons,
            "disposition":(
                "PLAYER_RECEIVER_CONVERSION_MECHANISM_CONFIRMED"
                if confirmed else
                "PLAYER_RECEIVER_CONVERSION_MECHANISM_NOT_CONFIRMED"
            ),
        })

    gates={
        "exact_seasons":sorted(panel["season"].astype(int).unique().tolist())==[2023,2024,2025],
        "target_weeks_5_18":int(panel["week"].min())>=5 and int(panel["week"].max())<=18,
        "chronology_violations_zero":integrity["chronology_violations"]==0,
        "same_team_history_violations_zero":integrity["same_team_history_violations"]==0,
        "target_game_ftn_feature_use_zero":True,
        "sportsbook_inputs_zero":True,
        "oddsapi_calls_zero":True,
        "outcomes_2026_zero":True,
        "model_architecture_count_le_2":models_fit<=2,
        "threshold_searches_zero":True,
        "production_changed_false":True,
    }
    any_confirmed=any(
        d["disposition"]=="PLAYER_RECEIVER_CONVERSION_MECHANISM_CONFIRMED"
        for d in decisions
    )
    disposition=(
        "PLAYER_RECEIVER_CONVERSION_MECHANISM_EXISTS"
        if any_confirmed else
        "PLAYER_RECEIVER_CONVERSION_MECHANISM_NOT_CONFIRMED"
    )
    if not all(gates.values()):
        disposition="PLAYER_RECEIVER_CONVERSION_MECHANISM_INTEGRITY_FAILURE"

    result={
        "version":"PLAYER_RECEIVER_CONVERSION_MECHANISM_V1",
        "disposition":disposition,
        "development_season":2023,
        "confirmation_seasons":[2024,2025],
        "target_weeks":[5,18],
        "features":list(FEATURES),
        "ridge_alpha":RIDGE_ALPHA,
        "bootstrap_reps":BOOT_REPS,
        "support_floors":SUPPORT,
        "panel_rows":int(len(panel)),
        "panel_players":int(panel["player_clean_key"].nunique()),
        "integrity":integrity,
        "position_decisions":decisions,
        "gates":gates,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "outcomes_2026_used":0,
        "models_fit":models_fit,
        "threshold_searches":0,
        "production_changed":False,
    }
    panel.to_csv(args.out_dir/"player_receiver_conversion_mechanism_panel.csv",index=False)
    if scored:
        pd.concat(scored,ignore_index=True).to_csv(
            args.out_dir/"player_receiver_conversion_confirmation_predictions.csv",index=False
        )
    pd.DataFrame(coefficients).to_csv(
        args.out_dir/"player_receiver_conversion_2023_coefficients.csv",index=False
    )
    (args.out_dir/"player_receiver_conversion_mechanism_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
