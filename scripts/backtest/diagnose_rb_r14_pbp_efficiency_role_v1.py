#!/usr/bin/env python3
"""RB-R14 diagnostic: strict-prior PBP receiving-role / explosive-efficiency signals.

R13 showed that box-score YPT history is not a useful standalone game-level
receiver-efficiency predictor. R14 asks a more football-specific question: does HOW
an RB has been used in the pass game (YAC, depth, behind-LOS usage, explosives,
target EPA) contain stable pregame information about future high-YPT games?

Diagnostic only: no target pools, R9/R12 values, sportsbook inputs, or production
parameters are changed. Current-game outcomes are labels only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.audit_rb_receiving_identity_v1 import _load_logs
from scripts.utils.pbp import get_pbp
from scripts.utils.canonical_names import canonicalize_player_name_safe

PRIMARY = [
    "prior_yac_per_target", "last8_yac_per_target",
    "prior_explosive_target_rate", "last8_explosive_target_rate",
    "prior_adot", "last8_adot",
]
FEATURES = PRIMARY + [
    "prior_yac_per_reception", "last8_yac_per_reception",
    "prior_behind_los_rate", "last8_behind_los_rate",
    "prior_target_epa", "last8_target_epa",
    "prior_target_success_rate", "last8_target_success_rate",
    "prev_season_yac_per_target", "prev_season_explosive_target_rate", "prev_season_adot",
    "same_team_yac_per_target", "same_team_explosive_target_rate", "same_team_adot",
    "frozen_ypt", "state_probability",
]

# Frozen diagnostic gates before execution.
MIN_PRIMARY_SPEARMAN = 0.08
MIN_PRIMARY_HIGH8_AUC = 0.55
MIN_POSITIVE_SEASONS = 2
MIN_PBP_MAPPING_RATE = 0.90


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _ratio(a, b):
    a = _num(a); b = _num(b)
    return np.where(b.gt(0), a / b, np.nan)


def _canon_key(v) -> str:
    try:
        _, key = canonicalize_player_name_safe(v)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def _pbp_rb_games(logs: pd.DataFrame, seasons: list[int]) -> tuple[pd.DataFrame, dict]:
    id_map = (
        logs.loc[logs.position_family.isin({"RB", "FB"}), ["season","player_id","player_clean_key"]]
        .dropna(subset=["player_clean_key"]).drop_duplicates()
    )
    id_map["player_id"] = id_map.player_id.astype(str).str.strip()
    id_map = id_map.loc[id_map.player_id.ne("")].drop_duplicates(["season","player_id"])
    rb_names = (
        logs.loc[logs.position_family.isin({"RB", "FB"}), ["season","player_clean_key"]]
        .drop_duplicates()
    )

    parts = []
    total_target_rows = mapped_rb_target_rows = 0
    yac_available = air_available = 0
    for season in seasons:
        p = get_pbp(int(season), min_rows=1)
        if "season_type" in p.columns:
            reg = p.loc[p.season_type.astype(str).str.upper().eq("REG")].copy()
            if not reg.empty:
                p = reg
        for c in ("pass_attempt","complete_pass","yards_gained","yards_after_catch","air_yards","epa"):
            if c not in p.columns:
                p[c] = np.nan
        if "receiver_player_id" not in p.columns:
            p["receiver_player_id"] = ""
        if "receiver_player_name" not in p.columns:
            p["receiver_player_name"] = ""
        if "posteam" not in p.columns or "week" not in p.columns:
            raise RuntimeError(f"PBP {season} missing posteam/week")
        p["pass_attempt"] = _num(p.pass_attempt).fillna(0).astype(int)
        t = p.loc[p.pass_attempt.eq(1) & (p.receiver_player_id.notna() | p.receiver_player_name.notna())].copy()
        t["receiver_player_id"] = t.receiver_player_id.astype(str).str.strip()
        t["player_clean_key_name"] = t.receiver_player_name.map(_canon_key)
        total_target_rows += len(t)
        yac_available += int(_num(t.yards_after_catch).notna().sum())
        air_available += int(_num(t.air_yards).notna().sum())

        m = id_map.loc[id_map.season.eq(season), ["player_id","player_clean_key"]].rename(columns={"player_id":"receiver_player_id"})
        t = t.merge(m, on="receiver_player_id", how="left")
        valid_names = set(rb_names.loc[rb_names.season.eq(season), "player_clean_key"].astype(str))
        fallback = t.player_clean_key.isna() & t.player_clean_key_name.isin(valid_names)
        t.loc[fallback, "player_clean_key"] = t.loc[fallback, "player_clean_key_name"]
        t = t.loc[t.player_clean_key.notna()].copy()
        mapped_rb_target_rows += len(t)

        t["season"] = int(season)
        t["week"] = _num(t.week).astype("Int64")
        max_week = 17 if season <= 2020 else 18
        t = t.loc[t.week.between(1, max_week)].copy()
        t["team"] = t.posteam.astype(str)
        t["complete_pass"] = _num(t.complete_pass).fillna(0).astype(int)
        t["yards_gained"] = _num(t.yards_gained).fillna(0.0)
        t["yards_after_catch"] = _num(t.yards_after_catch)
        t["air_yards"] = _num(t.air_yards)
        t["epa"] = _num(t.epa)
        t["yac_for_sum"] = t.yards_after_catch.fillna(0.0)
        t["air_for_sum"] = t.air_yards.fillna(0.0)
        t["behind_los"] = t.air_yards.le(0).astype(float)
        t["explosive"] = (t.complete_pass.eq(1) & t.yards_gained.ge(20)).astype(float)
        t["success"] = t.epa.gt(0).astype(float)
        t["epa_for_sum"] = t.epa.fillna(0.0)
        g = t.groupby(["season","week","team","player_clean_key"], as_index=False).agg(
            targets=("complete_pass","size"), receptions=("complete_pass","sum"),
            pbp_rec_yards=("yards_gained", lambda s: float(_num(s).sum())),
            yac_sum=("yac_for_sum","sum"), air_sum=("air_for_sum","sum"),
            behind_los_targets=("behind_los","sum"), explosive_targets=("explosive","sum"),
            target_epa_sum=("epa_for_sum","sum"), target_successes=("success","sum"),
            yac_nonmissing=("yards_after_catch", lambda s: int(_num(s).notna().sum())),
            air_nonmissing=("air_yards", lambda s: int(_num(s).notna().sum())),
        )
        parts.append(g)
        print(f"[rb-r14] season={season} pbp_rows={len(p)} targeted={len(t)} rb_player_games={len(g)}")

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    audit = {
        "seasons": seasons, "all_targeted_pass_rows": int(total_target_rows),
        "mapped_rb_target_rows": int(mapped_rb_target_rows),
        "rb_mapping_rate_of_all_target_rows": float(mapped_rb_target_rows / total_target_rows) if total_target_rows else np.nan,
        "yac_nonmissing_rate_all_targets": float(yac_available / total_target_rows) if total_target_rows else np.nan,
        "air_nonmissing_rate_all_targets": float(air_available / total_target_rows) if total_target_rows else np.nan,
    }
    return out, audit


def _state(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("time_key").copy()
    t = _num(g.targets).fillna(0); r = _num(g.receptions).fillna(0)
    yac = _num(g.yac_sum).fillna(0); air = _num(g.air_sum).fillna(0)
    expl = _num(g.explosive_targets).fillna(0); bl = _num(g.behind_los_targets).fillna(0)
    epa = _num(g.target_epa_sum).fillna(0); suc = _num(g.target_successes).fillna(0)
    ct, cr = t.cumsum(), r.cumsum()
    g["after_yac_per_target"] = _ratio(yac.cumsum(), ct)
    g["after_yac_per_reception"] = _ratio(yac.cumsum(), cr)
    g["after_adot"] = _ratio(air.cumsum(), ct)
    g["after_behind_los_rate"] = _ratio(bl.cumsum(), ct)
    g["after_explosive_target_rate"] = _ratio(expl.cumsum(), ct)
    g["after_target_epa"] = _ratio(epa.cumsum(), ct)
    g["after_target_success_rate"] = _ratio(suc.cumsum(), ct)
    rt=t.rolling(8,min_periods=1).sum(); rr=r.rolling(8,min_periods=1).sum()
    g["after_last8_yac_per_target"] = _ratio(yac.rolling(8,min_periods=1).sum(), rt)
    g["after_last8_yac_per_reception"] = _ratio(yac.rolling(8,min_periods=1).sum(), rr)
    g["after_last8_adot"] = _ratio(air.rolling(8,min_periods=1).sum(), rt)
    g["after_last8_behind_los_rate"] = _ratio(bl.rolling(8,min_periods=1).sum(), rt)
    g["after_last8_explosive_target_rate"] = _ratio(expl.rolling(8,min_periods=1).sum(), rt)
    g["after_last8_target_epa"] = _ratio(epa.rolling(8,min_periods=1).sum(), rt)
    g["after_last8_target_success_rate"] = _ratio(suc.rolling(8,min_periods=1).sum(), rt)
    g["pbp_source_time_key"] = g.time_key
    return g


def _team_state(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("time_key").copy()
    t=_num(g.targets).fillna(0); yac=_num(g.yac_sum).fillna(0); air=_num(g.air_sum).fillna(0); expl=_num(g.explosive_targets).fillna(0)
    ct=t.cumsum()
    g["same_team_after_yac_per_target"]=_ratio(yac.cumsum(),ct)
    g["same_team_after_explosive_target_rate"]=_ratio(expl.cumsum(),ct)
    g["same_team_after_adot"]=_ratio(air.cumsum(),ct)
    g["same_team_pbp_source_time_key"]=g.time_key
    return g


def _prev(games: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (k,s),g in games.groupby(["player_clean_key","season"]):
        t=float(_num(g.targets).sum()); yac=float(_num(g.yac_sum).sum()); air=float(_num(g.air_sum).sum()); expl=float(_num(g.explosive_targets).sum())
        rows.append({"player_clean_key":k,"season":int(s)+1,
                     "prev_season_yac_per_target":yac/t if t>0 else np.nan,
                     "prev_season_explosive_target_rate":expl/t if t>0 else np.nan,
                     "prev_season_adot":air/t if t>0 else np.nan})
    return pd.DataFrame(rows)


def _attach(q: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    games=games.copy(); games["time_key"]=_num(games.season).astype(int)*100+_num(games.week).astype(int)
    states=pd.concat([_state(g) for _,g in games.groupby("player_clean_key",sort=False)],ignore_index=True)
    teams=pd.concat([_team_state(g) for _,g in games.groupby(["player_clean_key","team"],sort=False)],ignore_index=True)
    prev=_prev(games)
    x=q.copy().reset_index(drop=True); x["time_key"]=_num(x.season).astype(int)*100+_num(x.week).astype(int); x["_qrow"]=np.arange(len(x))
    keep=["player_clean_key","time_key","pbp_source_time_key","after_yac_per_target","after_yac_per_reception","after_adot","after_behind_los_rate","after_explosive_target_rate","after_target_epa","after_target_success_rate","after_last8_yac_per_target","after_last8_yac_per_reception","after_last8_adot","after_last8_behind_los_rate","after_last8_explosive_target_rate","after_last8_target_epa","after_last8_target_success_rate"]
    s=states[keep].sort_values(["time_key","player_clean_key"])
    x=pd.merge_asof(x.sort_values(["time_key","player_clean_key"]),s,on="time_key",by="player_clean_key",direction="backward",allow_exact_matches=False)
    x=x.rename(columns={c:c.replace("after_","prior_").replace("prior_last8_","last8_") for c in keep if c.startswith("after_")})
    tk=teams[["player_clean_key","team","time_key","same_team_pbp_source_time_key","same_team_after_yac_per_target","same_team_after_explosive_target_rate","same_team_after_adot"]].sort_values(["time_key","player_clean_key","team"])
    x=pd.merge_asof(x.sort_values(["time_key","player_clean_key","team"]),tk,on="time_key",by=["player_clean_key","team"],direction="backward",allow_exact_matches=False)
    x=x.rename(columns={"same_team_after_yac_per_target":"same_team_yac_per_target","same_team_after_explosive_target_rate":"same_team_explosive_target_rate","same_team_after_adot":"same_team_adot"})
    x=x.merge(prev,on=["player_clean_key","season"],how="left").sort_values("_qrow").drop(columns="_qrow").reset_index(drop=True)
    audit={"query_rows":int(len(x)),"strict_prior_player_time_violations":int((x.pbp_source_time_key.notna()&(x.pbp_source_time_key>=x.time_key)).sum()),"strict_prior_same_team_time_violations":int((x.same_team_pbp_source_time_key.notna()&(x.same_team_pbp_source_time_key>=x.time_key)).sum())}
    return x,audit


def _auc(y,score):
    y=_num(y); score=_num(score); ok=y.notna()&score.notna(); y=y.loc[ok].astype(int); score=score.loc[ok]
    n1=int(y.sum()); n0=int(len(y)-n1)
    if n1==0 or n0==0: return np.nan
    ranks=score.rank(method="average")
    return float((ranks.loc[y.eq(1)].sum()-n1*(n1+1)/2)/(n1*n0))


def _row(g,feature,season_bucket,pop):
    a=_num(g.actual_ypt); f=_num(g[feature]); ok=a.notna()&f.notna(); z=g.loc[ok].copy()
    if len(z)<50: return None
    a=_num(z.actual_ypt); f=_num(z[feature]); y8=a.ge(8).astype(int); y10=a.ge(10).astype(int)
    sp=float(a.corr(f,method="spearman")) if a.nunique()>1 and f.nunique()>1 else np.nan
    pe=float(a.corr(f,method="pearson")) if a.std()>0 and f.std()>0 else np.nan
    pct=z.assign(_f=f).groupby(["season","week"])["_f"].rank(pct=True,method="average"); top=pct.gt(.80)
    e8=float(y8.mean()); e10=float(y10.mean()); t8=float(y8.loc[top].mean()) if top.any() else np.nan; t10=float(y10.loc[top].mean()) if top.any() else np.nan
    return {"season_bucket":season_bucket,"population":pop,"feature":feature,"n":int(len(z)),"pearson_actual_ypt":pe,"spearman_actual_ypt":sp,"high8_auc":_auc(y8,f),"high10_auc":_auc(y10,f),"high8_event_rate":e8,"high10_event_rate":e10,"top20_high8_lift":float(t8/e8) if pd.notna(t8) and e8>0 else np.nan,"top20_high10_lift":float(t10/e10) if pd.notna(t10) and e10>0 else np.nan}


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--predictions",type=Path,required=True); ap.add_argument("--pbp-start",type=int,default=2018); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args()
    q=pd.read_csv(a.predictions,low_memory=False)
    req={"season","week","team","player_clean_key","identity_bucket","actual_targets","actual_rec_yards","frozen_ypt","state_probability"}; missing=sorted(req-set(q.columns))
    if missing: raise RuntimeError(f"R14 predictions missing columns: {missing}")
    through=int(_num(q.season).max()); logs=_load_logs(list(range(a.pbp_start,through+1)))
    games,map_audit=_pbp_rb_games(logs,list(range(a.pbp_start,through+1)))
    if games.empty: raise RuntimeError("R14 produced zero RB PBP receiving games")
    x,time_audit=_attach(q,games)
    x["actual_targets"]=_num(x.actual_targets); x["actual_rec_yards"]=_num(x.actual_rec_yards); x["actual_ypt"]=np.where(x.actual_targets.gt(0),x.actual_rec_yards/x.actual_targets,np.nan)
    rows=[]
    for sb,g0 in [("COMBINED",x)]+[(str(int(s)),g) for s,g in x.groupby("season")]:
        for pop,g1 in [("ALL_RB",g0),("TOP20_IDENTITY",g0.loc[g0.identity_bucket.eq("TOP20")]),("REST80_IDENTITY",g0.loc[g0.identity_bucket.eq("REST80")])]:
            g=g1.loc[g1.actual_targets.ge(3)].copy()
            for f in FEATURES:
                if f in g.columns:
                    r=_row(g,f,sb,pop)
                    if r is not None: rows.append(r)
    summary=pd.DataFrame(rows)
    combined=summary.loc[(summary.season_bucket.eq("COMBINED"))&summary.population.eq("ALL_RB")].copy()
    primary=combined.loc[combined.feature.isin(PRIMARY)].copy(); primary["positive_seasons"]=0
    for i,r in primary.iterrows():
        n=0
        for s in (2023,2024,2025):
            z=summary.loc[(summary.season_bucket.eq(str(s)))&summary.population.eq("ALL_RB")&summary.feature.eq(r.feature)]
            n+=int(len(z)==1 and pd.notna(z.iloc[0].spearman_actual_ypt) and float(z.iloc[0].spearman_actual_ypt)>0)
        primary.loc[i,"positive_seasons"]=n
    mapping_ok=bool(map_audit["rb_mapping_rate_of_all_target_rows"]>=MIN_PBP_MAPPING_RATE)
    signal=( _num(primary.spearman_actual_ypt).ge(MIN_PRIMARY_SPEARMAN) & _num(primary.high8_auc).ge(MIN_PRIMARY_HIGH8_AUC) & _num(primary.positive_seasons).ge(MIN_POSITIVE_SEASONS) )
    supported=bool(mapping_ok and signal.any() and time_audit["strict_prior_player_time_violations"]==0 and time_audit["strict_prior_same_team_time_violations"]==0)
    best=None
    if len(primary):
        b=primary.sort_values(["spearman_actual_ypt","high8_auc"],ascending=False).iloc[0]
        best={"feature":str(b.feature),"spearman_actual_ypt":float(b.spearman_actual_ypt),"high8_auc":float(b.high8_auc),"high10_auc":float(b.high10_auc),"positive_seasons":int(b.positive_seasons)}
    result={"diagnostic":"RB_R14_PBP_EFFICIENCY_ROLE_V1","disposition":"RB_R14_PBP_EFFICIENCY_SIGNAL_SUPPORTED_DIAGNOSTIC_ONLY" if supported else "RB_R14_PBP_EFFICIENCY_SIGNAL_NOT_SUPPORTED_DIAGNOSTIC_ONLY","pbp_efficiency_signal_supported":supported,"r13_status":"RB_R13_EFFICIENCY_HISTORY_SIGNAL_NOT_SUPPORTED_DIAGNOSTIC_ONLY_UNCHANGED","best_primary_signal":best,"thresholds":{"min_primary_spearman":MIN_PRIMARY_SPEARMAN,"min_primary_high8_auc":MIN_PRIMARY_HIGH8_AUC,"min_positive_seasons":MIN_POSITIVE_SEASONS,"min_pbp_mapping_rate":MIN_PBP_MAPPING_RATE},"pbp_mapping_audit":map_audit,"strict_prior_audit":time_audit,"sportsbook_inputs_added":0,"production_parameters_changed":0,"governance_note":"2023-2025 are research-visible diagnostic seasons; support only authorizes a separately frozen candidate/OOS design."}
    a.out_dir.mkdir(parents=True,exist_ok=True); x.to_csv(a.out_dir/"rb_r14_pbp_efficiency_casebook.csv",index=False); summary.to_csv(a.out_dir/"rb_r14_pbp_efficiency_feature_summary.csv",index=False); primary.to_csv(a.out_dir/"rb_r14_primary_summary.csv",index=False); (a.out_dir/"rb_r14_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); print("\n=== primary PBP signals ===\n",primary.to_string(index=False)); return 0

if __name__=="__main__": raise SystemExit(main())
