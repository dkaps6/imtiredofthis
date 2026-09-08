#!/usr/bin/env python3
"""RB-R15 diagnostic: strict-prior team/opponent receiving-efficiency context.

R13 rejected raw player YPT history and R14B rejected persistent player PBP role
as strong game-level YPT predictors. R15 tests the remaining football-context
hypothesis: team RB pass-game environment and opponent RB-receiving defense may
explain some efficiency state variation.

Diagnostic only. Current-game outcomes are labels. Team/opponent context uses only
completed prior games. No production parameters, target pools, or sportsbook data
are touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.audit_rb_receiving_identity_v1 import _load_logs
from scripts.backtest.diagnose_rb_r14_pbp_efficiency_role_v1 import _pbp_rb_games, _row
from scripts.backtest.historical_inputs import build_schedule_history

PRIMARY = [
    "prior_team_rb_yac_per_target", "last8_team_rb_yac_per_target",
    "prior_opp_rb_yac_per_target_allowed", "last8_opp_rb_yac_per_target_allowed",
    "context_yac_per_target", "context_explosive_target_rate",
]
FEATURES = PRIMARY + [
    "prior_team_rb_explosive_target_rate", "last8_team_rb_explosive_target_rate",
    "prior_opp_rb_explosive_target_rate_allowed", "last8_opp_rb_explosive_target_rate_allowed",
    "prior_team_rb_adot", "last8_team_rb_adot",
    "prior_opp_rb_adot_allowed", "last8_opp_rb_adot_allowed",
    "prior_team_rb_target_epa", "last8_team_rb_target_epa",
    "prior_opp_rb_target_epa_allowed", "last8_opp_rb_target_epa_allowed",
    "context_adot", "context_target_epa", "state_probability", "frozen_ypt",
]

MIN_PRIMARY_SPEARMAN = 0.08
MIN_PRIMARY_HIGH8_AUC = 0.55
MIN_POSITIVE_SEASONS = 2


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _ratio(a, b):
    a = _num(a); b = _num(b)
    return np.where(b.gt(0), a / b, np.nan)


def _aggregate_team(games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    x = games.groupby(["season","week","team"], as_index=False).agg(
        targets=("targets","sum"), yac_sum=("yac_sum","sum"), air_sum=("air_sum","sum"),
        explosive_targets=("explosive_targets","sum"), target_epa_sum=("target_epa_sum","sum"),
        target_successes=("target_successes","sum"),
    )
    x = x.merge(schedule[["season","week","team","opponent"]], on=["season","week","team"], how="left", validate="one_to_one")
    if x.opponent.isna().any():
        raise RuntimeError(f"R15 schedule opponent missing for {int(x.opponent.isna().sum())} team-weeks")
    x["time_key"] = _num(x.season).astype(int) * 100 + _num(x.week).astype(int)
    return x


def _state(g: pd.DataFrame, prefix: str) -> pd.DataFrame:
    g = g.sort_values("time_key").copy()
    t = _num(g.targets).fillna(0); yac = _num(g.yac_sum).fillna(0); air = _num(g.air_sum).fillna(0)
    expl = _num(g.explosive_targets).fillna(0); epa = _num(g.target_epa_sum).fillna(0)
    ct=t.cumsum()
    g[f"after_{prefix}_yac_per_target"] = _ratio(yac.cumsum(), ct)
    g[f"after_{prefix}_explosive_target_rate"] = _ratio(expl.cumsum(), ct)
    g[f"after_{prefix}_adot"] = _ratio(air.cumsum(), ct)
    g[f"after_{prefix}_target_epa"] = _ratio(epa.cumsum(), ct)
    rt=t.rolling(8,min_periods=1).sum()
    g[f"after_last8_{prefix}_yac_per_target"] = _ratio(yac.rolling(8,min_periods=1).sum(), rt)
    g[f"after_last8_{prefix}_explosive_target_rate"] = _ratio(expl.rolling(8,min_periods=1).sum(), rt)
    g[f"after_last8_{prefix}_adot"] = _ratio(air.rolling(8,min_periods=1).sum(), rt)
    g[f"after_last8_{prefix}_target_epa"] = _ratio(epa.rolling(8,min_periods=1).sum(), rt)
    g[f"{prefix}_source_time_key"] = g.time_key
    return g


def _build_states(team_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    off = pd.concat([_state(g, "team_rb") for _,g in team_games.groupby("team", sort=False)], ignore_index=True)
    d = team_games.rename(columns={"opponent":"def_team"}).copy()
    deff = pd.concat([_state(g, "opp_rb") for _,g in d.groupby("def_team", sort=False)], ignore_index=True)
    return off, deff


def _attach(q: pd.DataFrame, schedule: pd.DataFrame, off: pd.DataFrame, deff: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x=q.copy().reset_index(drop=True); x["time_key"]=_num(x.season).astype(int)*100+_num(x.week).astype(int); x["_qrow"]=np.arange(len(x))
    x=x.merge(schedule[["season","week","team","opponent"]],on=["season","week","team"],how="left",validate="many_to_one")
    if x.opponent.isna().any(): raise RuntimeError(f"R15 query opponent missing: {int(x.opponent.isna().sum())}")

    off_cols=["team","time_key","team_rb_source_time_key","after_team_rb_yac_per_target","after_team_rb_explosive_target_rate","after_team_rb_adot","after_team_rb_target_epa","after_last8_team_rb_yac_per_target","after_last8_team_rb_explosive_target_rate","after_last8_team_rb_adot","after_last8_team_rb_target_epa"]
    o=off[off_cols].sort_values(["time_key","team"])
    x=pd.merge_asof(x.sort_values(["time_key","team"]),o,on="time_key",by="team",direction="backward",allow_exact_matches=False)
    x=x.rename(columns={
        "after_team_rb_yac_per_target":"prior_team_rb_yac_per_target",
        "after_team_rb_explosive_target_rate":"prior_team_rb_explosive_target_rate",
        "after_team_rb_adot":"prior_team_rb_adot","after_team_rb_target_epa":"prior_team_rb_target_epa",
        "after_last8_team_rb_yac_per_target":"last8_team_rb_yac_per_target",
        "after_last8_team_rb_explosive_target_rate":"last8_team_rb_explosive_target_rate",
        "after_last8_team_rb_adot":"last8_team_rb_adot","after_last8_team_rb_target_epa":"last8_team_rb_target_epa",
    })

    def_cols=["def_team","time_key","opp_rb_source_time_key","after_opp_rb_yac_per_target","after_opp_rb_explosive_target_rate","after_opp_rb_adot","after_opp_rb_target_epa","after_last8_opp_rb_yac_per_target","after_last8_opp_rb_explosive_target_rate","after_last8_opp_rb_adot","after_last8_opp_rb_target_epa"]
    d=deff[def_cols].sort_values(["time_key","def_team"])
    x=x.rename(columns={"opponent":"def_team"})
    x=pd.merge_asof(x.sort_values(["time_key","def_team"]),d,on="time_key",by="def_team",direction="backward",allow_exact_matches=False)
    x=x.rename(columns={
        "def_team":"opponent",
        "after_opp_rb_yac_per_target":"prior_opp_rb_yac_per_target_allowed",
        "after_opp_rb_explosive_target_rate":"prior_opp_rb_explosive_target_rate_allowed",
        "after_opp_rb_adot":"prior_opp_rb_adot_allowed","after_opp_rb_target_epa":"prior_opp_rb_target_epa_allowed",
        "after_last8_opp_rb_yac_per_target":"last8_opp_rb_yac_per_target_allowed",
        "after_last8_opp_rb_explosive_target_rate":"last8_opp_rb_explosive_target_rate_allowed",
        "after_last8_opp_rb_adot":"last8_opp_rb_adot_allowed","after_last8_opp_rb_target_epa":"last8_opp_rb_target_epa_allowed",
    })
    x=x.sort_values("_qrow").drop(columns="_qrow").reset_index(drop=True)
    x["context_yac_per_target"]=( _num(x.last8_team_rb_yac_per_target)+_num(x.last8_opp_rb_yac_per_target_allowed) )/2.0
    x["context_explosive_target_rate"]=( _num(x.last8_team_rb_explosive_target_rate)+_num(x.last8_opp_rb_explosive_target_rate_allowed) )/2.0
    x["context_adot"]=( _num(x.last8_team_rb_adot)+_num(x.last8_opp_rb_adot_allowed) )/2.0
    x["context_target_epa"]=( _num(x.last8_team_rb_target_epa)+_num(x.last8_opp_rb_target_epa_allowed) )/2.0
    audit={
        "query_rows":int(len(x)),
        "strict_prior_team_time_violations":int((x.team_rb_source_time_key.notna()&(x.team_rb_source_time_key>=x.time_key)).sum()),
        "strict_prior_opponent_time_violations":int((x.opp_rb_source_time_key.notna()&(x.opp_rb_source_time_key>=x.time_key)).sum()),
        "opponent_missing":int(x.opponent.isna().sum()),
    }
    return x,audit


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--predictions",type=Path,required=True); ap.add_argument("--pbp-start",type=int,default=2018); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args()
    q=pd.read_csv(a.predictions,low_memory=False)
    through=int(_num(q.season).max()); seasons=list(range(int(a.pbp_start),through+1))
    logs=_load_logs(seasons); games,map_audit=_pbp_rb_games(logs,seasons); schedule=build_schedule_history(seasons)
    team_games=_aggregate_team(games,schedule); off,deff=_build_states(team_games); x,audit=_attach(q,schedule,off,deff)
    x["actual_targets"]=_num(x.actual_targets); x["actual_rec_yards"]=_num(x.actual_rec_yards); x["actual_ypt"]=np.where(x.actual_targets.gt(0),x.actual_rec_yards/x.actual_targets,np.nan)
    rows=[]
    for sb,g0 in [("COMBINED",x)]+[(str(int(s)),g) for s,g in x.groupby("season")]:
        for pop,g1 in [("ALL_RB",g0),("TOP20_IDENTITY",g0.loc[g0.identity_bucket.eq("TOP20")]),("REST80_IDENTITY",g0.loc[g0.identity_bucket.eq("REST80")])]:
            g=g1.loc[g1.actual_targets.ge(3)].copy()
            for f in FEATURES:
                if f in g.columns:
                    r=_row(g,f,sb,pop)
                    if r is not None: rows.append(r)
    summary=pd.DataFrame(rows); combined=summary.loc[summary.season_bucket.eq("COMBINED")&summary.population.eq("ALL_RB")]; primary=combined.loc[combined.feature.isin(PRIMARY)].copy(); primary["positive_seasons"]=0
    for i,row in primary.iterrows():
        n=0
        for s in (2023,2024,2025):
            z=summary.loc[summary.season_bucket.eq(str(s))&summary.population.eq("ALL_RB")&summary.feature.eq(row.feature)]
            n+=int(len(z)==1 and pd.notna(z.iloc[0].spearman_actual_ypt) and float(z.iloc[0].spearman_actual_ypt)>0)
        primary.loc[i,"positive_seasons"]=n
    sig=_num(primary.spearman_actual_ypt).ge(MIN_PRIMARY_SPEARMAN)&_num(primary.high8_auc).ge(MIN_PRIMARY_HIGH8_AUC)&_num(primary.positive_seasons).ge(MIN_POSITIVE_SEASONS)
    strict=bool(audit["strict_prior_team_time_violations"]==0 and audit["strict_prior_opponent_time_violations"]==0 and audit["opponent_missing"]==0); supported=bool(strict and sig.any())
    best=None
    if len(primary):
        b=primary.sort_values(["spearman_actual_ypt","high8_auc"],ascending=False).iloc[0]; best={"feature":str(b.feature),"spearman_actual_ypt":float(b.spearman_actual_ypt),"high8_auc":float(b.high8_auc),"high10_auc":float(b.high10_auc),"positive_seasons":int(b.positive_seasons)}
    result={"diagnostic":"RB_R15_EFFICIENCY_CONTEXT_V1","disposition":"RB_R15_EFFICIENCY_CONTEXT_SIGNAL_SUPPORTED_DIAGNOSTIC_ONLY" if supported else "RB_R15_EFFICIENCY_CONTEXT_SIGNAL_NOT_SUPPORTED_DIAGNOSTIC_ONLY","context_signal_supported":supported,"parents":["RB_R13_EFFICIENCY_HISTORY_SIGNAL_NOT_SUPPORTED","RB_R14_PBP_EFFICIENCY_SIGNAL_NOT_SUPPORTED"],"best_primary_signal":best,"thresholds":{"min_primary_spearman":MIN_PRIMARY_SPEARMAN,"min_primary_high8_auc":MIN_PRIMARY_HIGH8_AUC,"min_positive_seasons":MIN_POSITIVE_SEASONS},"strict_prior_audit":audit,"pbp_mapping_lineage_audit":map_audit,"sportsbook_inputs_added":0,"production_parameters_changed":0,"governance_note":"Diagnostic-only on research-visible seasons; no efficiency mean or production model is promoted here."}
    a.out_dir.mkdir(parents=True,exist_ok=True); x.to_csv(a.out_dir/"rb_r15_efficiency_context_casebook.csv",index=False); summary.to_csv(a.out_dir/"rb_r15_efficiency_context_summary.csv",index=False); primary.to_csv(a.out_dir/"rb_r15_primary_summary.csv",index=False); (a.out_dir/"rb_r15_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2)); print("\n=== primary context signals ===\n",primary.to_string(index=False)); return 0

if __name__=="__main__": raise SystemExit(main())
