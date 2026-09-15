#!/usr/bin/env python3
"""WR Phase 4C Gate-0: market-only tail game-script bridge diagnostic.

Research only. Reuses canonical Phase 4B outputs and reviewed PR #558/#559
pregame market descriptors. No model fitting, no player-prop odds, no
production changes, and no realized-script predictor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts._opponent_map import canon_team
from scripts.research.diagnose_market_implied_game_script_v1 import load_market_schedule
from scripts.research.diagnose_vegas_line_gamescript_calibration_v1 import (
    SPREAD_BINS, SPREAD_BIN_LABELS, TOTAL_BIN_EDGES_DEFAULT, TOTAL_BIN_LABELS,
)

TG = ["season", "week", "team"]
BOOT_REPS = 1000
BOOT_SEED = 20260915
EXPECTED_L1 = 4193
EXPECTED_L23 = 1025
EXPECTED_COUNTS = {
    "ACTUAL_100_PLUS": 307,
    "ACTUAL_100_PLUS_OPP_DOM": 185,
    "ABS_RESIDUAL_30_PLUS": 1048,
    "ABS_RESIDUAL_30_PLUS_OPP_DOM": 586,
    "UNDERPROJECT_30_PLUS": 809,
    "UNDERPROJECT_30_PLUS_OPP_DOM": 517,
    "OVERPROJECT_30_PLUS": 239,
}
EVENT_COHORTS = list(EXPECTED_COUNTS)
PRIMARY = "UNDERPROJECT_30_PLUS_OPP_DOM"
SECONDARY = ["ACTUAL_100_PLUS_OPP_DOM", "UNDERPROJECT_30_PLUS"]
CONT = ["market_total", "market_team_implied", "market_team_spread", "market_abs_spread"]


def _masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    oppdom = df["opportunity_yards"].abs() > df["efficiency_yards"].abs()
    return {
        "ALL": pd.Series(True, index=df.index),
        "ACTUAL_100_PLUS": df["actual_rec_yards"].ge(100.0),
        "ACTUAL_100_PLUS_OPP_DOM": df["actual_rec_yards"].ge(100.0) & oppdom,
        "ABS_RESIDUAL_30_PLUS": df["yard_residual"].abs().ge(30.0),
        "ABS_RESIDUAL_30_PLUS_OPP_DOM": df["yard_residual"].abs().ge(30.0) & oppdom,
        "UNDERPROJECT_30_PLUS": df["yard_residual"].ge(30.0),
        "UNDERPROJECT_30_PLUS_OPP_DOM": df["yard_residual"].ge(30.0) & oppdom,
        "OVERPROJECT_30_PLUS": df["yard_residual"].le(-30.0),
    }


def _season_scopes(df: pd.DataFrame):
    yield "POOLED", df
    for s in (2023, 2024):
        yield str(s), df.loc[df.season.eq(s)].copy()


def _cluster_resamples(df: pd.DataFrame, reps: int, seed: int):
    """Yield row-position arrays after resampling (season,week,team) clusters.

    Pooled resampling is stratified by season. Each team-game is a distinct
    cluster; opposing teams in the same game remain distinct clusters.
    """
    x = df.reset_index(drop=True)
    key_to_rows = x.groupby(TG, sort=True).indices
    strata: dict[int, list[np.ndarray]] = {}
    for key, rows in key_to_rows.items():
        season = int(key[0])
        strata.setdefault(season, []).append(np.asarray(rows, dtype=int))
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        out = []
        for season in sorted(strata):
            clusters = strata[season]
            picks = rng.integers(0, len(clusters), size=len(clusters))
            out.extend(clusters[i] for i in picks)
        yield np.concatenate(out) if out else np.array([], dtype=int)


def _percentile_ci(vals) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 100:
        return np.nan, np.nan
    return tuple(float(v) for v in np.percentile(a, [2.5, 97.5]))


def _boot_continuous_diff(df, cohort_col, value_col, control_col=None, reps=BOOT_REPS, seed=BOOT_SEED):
    cohort = df[cohort_col].astype(bool)
    control = (~cohort) if control_col is None else df[control_col].astype(bool)
    if cohort.sum() == 0 or control.sum() == 0:
        return np.nan, np.nan, np.nan
    obs = float(df.loc[cohort, value_col].mean() - df.loc[control, value_col].mean())
    boots=[]
    for idx in _cluster_resamples(df, reps, seed):
        b=df.iloc[idx]
        cm=b[cohort_col].astype(bool)
        ctl=(~cm) if control_col is None else b[control_col].astype(bool)
        if cm.sum() and ctl.sum():
            boots.append(float(b.loc[cm,value_col].mean()-b.loc[ctl,value_col].mean()))
    lo,hi=_percentile_ci(boots)
    return obs,lo,hi


def _boot_prevalence_diff(df, cohort_col, stratum_mask, reps=BOOT_REPS, seed=BOOT_SEED):
    c=df[cohort_col].astype(bool)
    s=pd.Series(stratum_mask,index=df.index).astype(bool)
    if s.sum()==0:
        return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    base=float(c.mean()); prev=float(c.loc[s].mean()); diff=prev-base; ratio=prev/base if base>0 else np.nan
    boots=[]
    for idx in _cluster_resamples(df,reps,seed):
        b=df.iloc[idx]
        cb=b[cohort_col].astype(bool).reset_index(drop=True)
        sb=s.iloc[idx].reset_index(drop=True)
        if sb.sum()==0: continue
        boots.append(float(cb.loc[sb].mean()-cb.mean()))
    lo,hi=_percentile_ci(boots)
    return prev,base,diff,ratio,lo,hi


def _player_rows(l1: pd.DataFrame, reps: int) -> tuple[pd.DataFrame,pd.DataFrame]:
    masks=_masks(l1)
    for name,n in EXPECTED_COUNTS.items():
        got=int(masks[name].sum())
        if got!=n: raise RuntimeError(f"cohort drift {name}: {got} != {n}")
    x=l1.copy()
    for name,m in masks.items(): x[name]=m.astype(bool)
    rows=[]; ctrls=[]
    seed_offset=0
    for scope,g in _season_scopes(x):
        for cohort in EVENT_COHORTS:
            n=int(g[cohort].sum()); base=float(g[cohort].mean())
            for label in TOTAL_BIN_LABELS:
                sm=g["total_bucket"].eq(label)
                prev,basep,diff,ratio,lo,hi=_boot_prevalence_diff(g,cohort,sm,reps,BOOT_SEED+seed_offset); seed_offset+=1
                rows.append(dict(scope=scope,cohort=cohort,family="total_bucket",concept="market_total",view=str(label),n_cohort=n,n_stratum=int(sm.sum()),base_prevalence=basep,stratum_prevalence=prev,estimate=diff,enrichment_ratio=ratio,ci_low=lo,ci_high=hi))
            for label in SPREAD_BIN_LABELS:
                sm=g["spread_bucket"].eq(label)
                prev,basep,diff,ratio,lo,hi=_boot_prevalence_diff(g,cohort,sm,reps,BOOT_SEED+seed_offset); seed_offset+=1
                rows.append(dict(scope=scope,cohort=cohort,family="spread_bucket",concept="market_abs_spread",view=str(label),n_cohort=n,n_stratum=int(sm.sum()),base_prevalence=basep,stratum_prevalence=prev,estimate=diff,enrichment_ratio=ratio,ci_low=lo,ci_high=hi))
            for label,sm in [("FAVORITE",g.market_team_spread.gt(0)),("UNDERDOG",g.market_team_spread.lt(0))]:
                prev,basep,diff,ratio,lo,hi=_boot_prevalence_diff(g,cohort,sm,reps,BOOT_SEED+seed_offset); seed_offset+=1
                rows.append(dict(scope=scope,cohort=cohort,family="favorite_split",concept="market_team_spread",view=label,n_cohort=n,n_stratum=int(sm.sum()),base_prevalence=basep,stratum_prevalence=prev,estimate=diff,enrichment_ratio=ratio,ci_low=lo,ci_high=hi))
            for col in CONT:
                est,lo,hi=_boot_continuous_diff(g,cohort,col,None,reps,BOOT_SEED+seed_offset); seed_offset+=1
                rows.append(dict(scope=scope,cohort=cohort,family="continuous",concept=col,view=col,n_cohort=n,n_stratum=int(len(g)-n),base_prevalence=base,stratum_prevalence=np.nan,estimate=est,enrichment_ratio=np.nan,ci_low=lo,ci_high=hi))
            if cohort in {"UNDERPROJECT_30_PLUS","UNDERPROJECT_30_PLUS_OPP_DOM"}:
                for col in CONT:
                    est,lo,hi=_boot_continuous_diff(g,cohort,col,"OVERPROJECT_30_PLUS",reps,BOOT_SEED+seed_offset); seed_offset+=1
                    ctrls.append(dict(scope=scope,cohort=cohort,control="OVERPROJECT_30_PLUS",concept=col,estimate=est,ci_low=lo,ci_high=hi,n_cohort=n,n_control=int(g.OVERPROJECT_30_PLUS.sum())))
    return pd.DataFrame(rows),pd.DataFrame(ctrls)


def _bootstrap_spearman(df,xcol,ycol,reps,seed):
    obs=float(spearmanr(df[xcol],df[ycol],nan_policy="omit").statistic)
    rng=np.random.default_rng(seed); vals=[]; n=len(df)
    for _ in range(reps):
        idx=rng.integers(0,n,size=n); b=df.iloc[idx]
        r=spearmanr(b[xcol],b[ycol],nan_policy="omit").statistic
        if np.isfinite(r): vals.append(float(r))
    lo,hi=_percentile_ci(vals); return obs,lo,hi


def _layer2_rows(l23: pd.DataFrame,reps:int) -> tuple[pd.DataFrame,pd.DataFrame]:
    x=l23.copy(); x["abs_team_pool_component"]=x.team_pool_component.abs(); x["direct_team_target_residual"]=x.actual_team_targets-x.implied_team_target_pool
    outcomes=["team_pool_component","abs_team_pool_component","direct_team_target_residual"]
    corr=[]; buckets=[]; seed=BOOT_SEED+50000
    for scope,g in _season_scopes(x):
        for concept in CONT:
            for outcome in outcomes:
                rho,lo,hi=_bootstrap_spearman(g,concept,outcome,reps,seed); seed+=1
                corr.append(dict(scope=scope,concept=concept,outcome=outcome,estimate=rho,ci_low=lo,ci_high=hi,n=len(g)))
        for family,col,labels in [("total_bucket","total_bucket",TOTAL_BIN_LABELS),("spread_bucket","spread_bucket",SPREAD_BIN_LABELS)]:
            for label in labels:
                sm=g[col].eq(label)
                for outcome in outcomes:
                    est=float(g.loc[sm,outcome].mean()-g[outcome].mean()) if sm.sum() else np.nan
                    buckets.append(dict(scope=scope,family=family,concept="market_total" if family=="total_bucket" else "market_abs_spread",view=str(label),outcome=outcome,estimate=est,n_stratum=int(sm.sum()),n=len(g)))
        for label,sm in [("FAVORITE",g.market_team_spread.gt(0)),("UNDERDOG",g.market_team_spread.lt(0))]:
            for outcome in outcomes:
                est=float(g.loc[sm,outcome].mean()-g[outcome].mean()) if sm.sum() else np.nan
                buckets.append(dict(scope=scope,family="favorite_split",concept="market_team_spread",view=label,outcome=outcome,estimate=est,n_stratum=int(sm.sum()),n=len(g)))
    return pd.DataFrame(corr),pd.DataFrame(buckets)


def _sign(v): return 1 if v>0 else (-1 if v<0 else 0)

def _ci_excludes_zero(r): return bool((r.ci_low>0 and r.ci_high>0) or (r.ci_low<0 and r.ci_high<0))


def _evaluate_gate(player: pd.DataFrame, l2corr: pd.DataFrame, l2bucket: pd.DataFrame) -> dict:
    p=player[(player.scope=="POOLED")&(player.cohort==PRIMARY)].copy()
    candidates=p[p.apply(_ci_excludes_zero,axis=1)]
    tested=[]
    for _,r in candidates.iterrows():
        sign=_sign(float(r.estimate))
        seas=player[(player.cohort==PRIMARY)&(player.family==r.family)&(player.view==r.view)&(player.scope.isin(["2023","2024"]))]
        season_ok=(len(seas)==2 and all(_sign(float(v))==sign for v in seas.estimate))
        reps=player[(player.scope=="POOLED")&(player.cohort.isin(SECONDARY))&(player.family==r.family)&(player.view==r.view)]
        replicate_ok=any(_sign(float(v))==sign for v in reps.estimate if np.isfinite(v))
        if r.family=="continuous":
            z=l2corr[(l2corr.scope=="POOLED")&(l2corr.concept==r.concept)]
        else:
            z=l2bucket[(l2bucket.scope=="POOLED")&(l2bucket.family==r.family)&(l2bucket.view==r.view)]
        coherent=z[z.estimate.apply(lambda v: np.isfinite(v) and _sign(float(v))==sign)]
        layer2_ok=len(coherent)>0
        tested.append(dict(family=r.family,concept=r.concept,view=r.view,primary_estimate=float(r.estimate),primary_ci_low=float(r.ci_low),primary_ci_high=float(r.ci_high),direction=sign,season_coherence=season_ok,layer2_coherence=layer2_ok,secondary_replication=replicate_ok,advances=bool(season_ok and layer2_ok and replicate_ok),layer2_matching_rows=int(len(coherent))))
    adv=[x for x in tested if x["advances"]]
    return {"disposition":"ADVANCE_TO_SCRIPT_PREDICTOR_DESIGN" if adv else "NO_ACTIONABLE_TAIL_GAMESCRIPT_BRIDGE","primary_nonnull_candidates":len(tested),"advancing_candidates":adv,"candidate_audit":tested}


def _add_market(df, market):
    x=df.copy(); x.team=x.team.map(canon_team)
    out=x.merge(market[TG+["market_total","market_team_spread","market_abs_spread","market_team_implied"]],on=TG,how="left",validate="many_to_one")
    if out[CONT].isna().any().any():
        bad=out.loc[out[CONT].isna().any(axis=1),TG].drop_duplicates().head(20)
        raise RuntimeError(f"market join incomplete: {bad.to_dict('records')}")
    out["total_bucket"]=pd.cut(out.market_total,bins=TOTAL_BIN_EDGES_DEFAULT,labels=TOTAL_BIN_LABELS,right=False)
    out["spread_bucket"]=pd.cut(out.market_abs_spread,bins=SPREAD_BINS,labels=SPREAD_BIN_LABELS,right=False)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--layer1",type=Path,required=True); ap.add_argument("--layer23",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); ap.add_argument("--bootstrap-reps",type=int,default=BOOT_REPS); a=ap.parse_args()
    l1=pd.read_csv(a.layer1); l23=pd.read_csv(a.layer23)
    if len(l1)!=EXPECTED_L1: raise RuntimeError(f"Layer1 row drift {len(l1)}")
    if len(l23)!=EXPECTED_L23: raise RuntimeError(f"Layer23 row drift {len(l23)}")
    seasons=sorted(set(l1.season.astype(int))|set(l23.season.astype(int)))
    if seasons!=[2023,2024]: raise RuntimeError(f"season drift {seasons}")
    market=load_market_schedule(seasons)
    l1=_add_market(l1,market); l23=_add_market(l23,market)
    p,ctrl=_player_rows(l1,a.bootstrap_reps); c,b=_layer2_rows(l23,a.bootstrap_reps); gate=_evaluate_gate(p,c,b)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    p.to_csv(a.out_dir/"phase4c_player_tail_market_associations.csv",index=False); ctrl.to_csv(a.out_dir/"phase4c_directional_control_associations.csv",index=False); c.to_csv(a.out_dir/"phase4c_layer2_market_spearman.csv",index=False); b.to_csv(a.out_dir/"phase4c_layer2_market_buckets.csv",index=False)
    meta={"specification":"WR_PHASE4C_TAIL_GAMESCRIPT_GATE_V1","phase4b_layer1_rows":len(l1),"phase4b_layer23_team_games":len(l23),"cluster_unit":"season-week-team","bootstrap_reps":int(a.bootstrap_reps),"comparison_space":{"pooled_cells":112,"pooled_plus_season_cells":336,"directional_control_cells":24},"market_lineage":{"implied_team_target_pool_has_market_input":False,"r15_sportsbook_inputs_used":False},"gate":gate,"production_change":False,"script_predictor_built":False,"paid_full_slate":False,"rb_work":False}
    (a.out_dir/"phase4c_gate0_result.json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")
    print(json.dumps(meta,indent=2,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
