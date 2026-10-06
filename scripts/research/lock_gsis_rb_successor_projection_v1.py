#!/usr/bin/env python3
"""Freeze BASELINE / VACANCY_V1_SNAP / GSIS_LINEUP_V1 RB football projections.

Research-only, pregame-only, sportsbook-independent. Run from the exact
production-source checkout associated with the pregame Full Slate football
state. Player-level output is private and must not be committed publicly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.discrete_count_alignment_v1 import align_prealigned_outcomes
from scripts.modeling.ensemble_v2 import apply_ensemble
import scripts.run_pricing_with_full_roster_universe_v1 as production_base
import scripts.run_pricing_with_full_roster_universe_v2 as production_v2
import scripts.run_pricing_with_full_roster_universe_v3_core as production_v3
from scripts.run_pricing_with_full_roster_universe_v1 import _canonical_game
from scripts.simulation_v2 import lookup

DATA=Path("data")
ITERATIONS=25000
SEED=42
TOL=1e-10
FORBIDDEN={
    "actual","actual_rushes","actual_rush_yards","target_game_snaps",
    "line","source_line","over_odds","under_odds","odds","book","book_title",
    "sportsbook","bookmaker","market_prob","edge_pct","fair_prob","profit_units",
}

def _read(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"{label} missing/empty: {path}")
    x=pd.read_csv(path,low_memory=False)
    if x.empty:
        raise RuntimeError(f"{label} has zero rows: {path}")
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x

def _key(s:pd.Series)->pd.Series:
    return s.astype("string").fillna("").str.strip()

def _pos_family(s:pd.Series)->pd.Series:
    return s.astype("string").fillna("").str.upper().str.strip().replace({"HB":"RB","TB":"RB"})

def _forbidden(df:pd.DataFrame)->list[str]:
    bad=[]
    for c in df.columns:
        lc=str(c).lower()
        if lc in FORBIDDEN or lc.startswith("sportsbook_"):
            bad.append(str(c))
    return sorted(bad)

def _diag(df:pd.DataFrame,team:str,pkey:str,col:str)->float:
    q=df.loc[
        df.team.astype(str).str.upper().eq(team)
        & _key(df.player_clean_key).eq(pkey)
    ]
    if len(q)!=1:
        raise RuntimeError(f"expected one diagnostic row {team}/{pkey}/{col}, got {len(q)}")
    v=float(pd.to_numeric(q.iloc[0].get(col),errors="coerce"))
    if not np.isfinite(v):
        raise RuntimeError(f"non-finite diagnostic {team}/{pkey}/{col}")
    return v

def _array_hash(a:np.ndarray)->str:
    x=np.asarray(a,dtype="<f8")
    return hashlib.sha256(x.tobytes(order="C")).hexdigest()

def apply_transfer_arm(
    prepared:pd.DataFrame,
    allocation_lock:pd.DataFrame,
    *,
    transfer_col:str,
    label:str,
)->pd.DataFrame:
    if transfer_col not in allocation_lock.columns:
        raise RuntimeError(f"{label} allocation lock missing {transfer_col}")
    out=prepared.copy()
    lock=allocation_lock.copy()
    lock["team"]=lock.team.astype(str).str.upper().str.strip()
    lock["successor_player_clean_key"]=_key(lock.successor_player_clean_key)
    lock[transfer_col]=pd.to_numeric(lock[transfer_col],errors="raise").astype(float)
    if (lock[transfer_col]<0).any():
        raise RuntimeError(f"{label} has negative transfer")

    applied_nonzero=0
    for _,r in lock.iterrows():
        amount=float(r[transfer_col])
        mask=(
            out.team.astype(str).str.upper().eq(str(r.team))
            & _key(out.player_clean_key).eq(str(r.successor_player_clean_key))
        )
        if int(mask.sum())!=1:
            raise RuntimeError(
                f"{label} expected one active successor "
                f"{r.team}/{r.successor_player_clean_key}, got {int(mask.sum())}"
            )
        old=pd.to_numeric(out.loc[mask,"rules_rush_share"],errors="coerce")
        if old.isna().any():
            raise RuntimeError(f"{label} successor missing rules_rush_share")
        out.loc[mask,"rules_rush_share"]=old+amount
        applied_nonzero+=int(amount>0)

    for c in prepared.columns:
        if c=="rules_rush_share":
            continue
        a=prepared[c]; b=out[c]
        eq=a.eq(b)|(a.isna()&b.isna())
        if not bool(eq.all()):
            raise RuntimeError(f"{label} changed forbidden football field: {c}")
    if not np.allclose(
        pd.to_numeric(prepared["rules_ypc"],errors="coerce"),
        pd.to_numeric(out["rules_ypc"],errors="coerce"),
        equal_nan=True,
    ):
        raise RuntimeError(f"{label} changed YPC/efficiency")
    out.attrs["applied_nonzero_transfers"]=applied_nonzero
    return out

def _prepare(allocation:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,dict]:
    consensus=_read(DATA/"player_form_consensus.csv","pregame PlayerForm consensus")
    ml=_read(DATA/"model_ml_diagnostics.csv","pregame ML diagnostics")
    state=_read(DATA/"model_state_diagnostics.csv","pregame State diagnostics")
    weights=_read(DATA/"model_ensemble_weights.csv","promoted ensemble weights")
    availability=_read(DATA/"current_player_availability.csv","pregame availability")

    for label,df in [
        ("consensus",consensus),("ml",ml),("state",state),("weights",weights),("availability",availability)
    ]:
        bad=_forbidden(df)
        if bad:
            raise RuntimeError(f"{label} contains forbidden target/market fields: {bad}")

    seasons=sorted(set(pd.to_numeric(allocation.target_season,errors="raise").astype(int)))
    weeks=sorted(set(pd.to_numeric(allocation.target_week,errors="raise").astype(int)))
    if len(seasons)!=1 or len(weeks)!=1:
        raise RuntimeError(f"projection lock requires one target season/week, got {seasons}/{weeks}")
    season,week=seasons[0],weeks[0]
    if week<=1:
        raise RuntimeError("GSIS successor projection V1 is frozen for Week > 1 current generic RB mean route")

    consensus["team"]=consensus.team.astype(str).str.upper().str.strip()
    target_seed=consensus.loc[
        pd.to_numeric(consensus.season,errors="coerce").eq(season)
        & pd.to_numeric(consensus.week,errors="coerce").eq(week)
    ].copy()
    if target_seed.empty:
        raise RuntimeError(f"PlayerForm consensus has no season={season} week={week} rows")

    # Enter through the exact current Full Slate V3/V6 Week>1 football universe
    # seam: suffix-safe identity, M38 explicit entitlement, TE-R5P, WR-R15 and
    # QB C2. V4/V5 Week-1-only RB adapters are no-ops here; V6 changes only
    # rush_rec_yards downstream, not rush_att/rush_yards.
    target_seed["event_id"]=[
        _canonical_game(t,o,s,w)
        for t,o,s,w in zip(target_seed.team,target_seed.opponent,target_seed.season,target_seed.week)
    ]
    target_seed["market"]="football_universe"
    production_base._identity_frame=production_v2._canonical_identity_frame
    production_base._validate_priced_distribution_coverage=(
        production_v2._install_provider_player_aliases_and_validate
    )
    target,_,production_audit=production_v3._build_with_promoted_entitlement_specialists(target_seed)

    if not pd.to_numeric(target.get("rules_applied",0),errors="coerce").fillna(0).eq(1).all():
        raise RuntimeError("promoted production universe missing canonical rules")
    if "team_wp" in target.columns:
        raise RuntimeError("market-derived team_wp leaked into projection lock")
    if target["team"].nunique()!=32:
        raise RuntimeError(f"promoted production universe expected 32 teams, got {target['team'].nunique()}")

    # Unavailable RB/FBs must already be absent from the active football universe.
    availability["team"]=availability.team.astype(str).str.upper().str.strip()
    availability["player_clean_key"]=_key(availability.player_clean_key)
    unavailable=set(zip(
        availability.loc[
            pd.to_numeric(availability.definitive_unavailable,errors="coerce").fillna(0).eq(1)
            & availability.position_group.astype(str).str.upper().isin(["RB","FB"]),
            "team"
        ].astype(str),
        availability.loc[
            pd.to_numeric(availability.definitive_unavailable,errors="coerce").fillna(0).eq(1)
            & availability.position_group.astype(str).str.upper().isin(["RB","FB"]),
            "player_clean_key"
        ].astype(str),
    ))
    active=set(zip(target.team.astype(str),_key(target.player_clean_key).astype(str)))
    leak=sorted(unavailable&active)
    if leak:
        raise RuntimeError(f"definitive unavailable RB/FB leaked into active universe: {leak[:20]}")

    return target,ml,state,weights,production_audit

def _simulate_current_week_gt1_production(
    frame:pd.DataFrame,
    *,
    iterations:int,
    seed:int,
    allocation_trace:list[dict],
):
    return production_v3._simulate_promoted_stack(
        frame,
        iterations=iterations,
        seed=seed,
        allocation_trace=allocation_trace,
    )

def build_projection_lock(allocation:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    bad=_forbidden(allocation)
    if bad:
        raise RuntimeError(f"allocation lock contains forbidden fields: {bad}")
    req={
        "target_season","target_week","event_id","team","successor_player_clean_key",
        "vacated_rush_share","snap_transfer_rush_share","gsis_transfer_rush_share",
        "gsis_snapshot_sha256","gsis_team_capture_utc","kickoff_utc",
    }
    miss=req-set(allocation.columns)
    if miss:
        raise RuntimeError(f"allocation lock missing {sorted(miss)}")
    lock=allocation.copy()
    lock["team"]=lock.team.astype(str).str.upper().str.strip()
    lock["successor_player_clean_key"]=_key(lock.successor_player_clean_key)
    if lock.duplicated(["target_season","target_week","team","successor_player_clean_key"]).any():
        raise RuntimeError("duplicate allocation-lock successor identity")

    prepared,ml,state,weights,production_audit=_prepare(lock)
    event_teams=sorted(lock.team.unique())

    cohort=prepared.loc[
        prepared.team.isin(event_teams)
        & _pos_family(prepared.position).isin(["RB","FB"])
    ].copy()
    cohort=cohort.sort_values(["team","player_clean_key"]).drop_duplicates(["team","player_clean_key"])
    if cohort.empty:
        raise RuntimeError("active vacancy-team RB/FB cohort empty")

    expected=set(zip(lock.team.astype(str),lock.successor_player_clean_key.astype(str)))
    actual=set(zip(cohort.team.astype(str),cohort.player_clean_key.astype(str)))
    if expected!=actual:
        raise RuntimeError(
            f"allocation successor pool != active production RB/FB cohort "
            f"missing_from_lock={sorted(actual-expected)[:20]} extra_in_lock={sorted(expected-actual)[:20]}"
        )

    baseline=prepared.copy()
    snap=apply_transfer_arm(prepared,lock,transfer_col="snap_transfer_rush_share",label="VACANCY_V1_SNAP")
    gsis=apply_transfer_arm(prepared,lock,transfer_col="gsis_transfer_rush_share",label="GSIS_LINEUP_V1")

    traces={}; sims={}
    for name,frame in [("BASELINE",baseline),("VACANCY_V1_SNAP",snap),("GSIS_LINEUP_V1",gsis)]:
        trace=[]
        sims[name]=_simulate_current_week_gt1_production(
            frame,iterations=ITERATIONS,seed=SEED,allocation_trace=trace
        )
        traces[name]=pd.DataFrame(trace)

    rows=[]
    for _,p in cohort.iterrows():
        team=str(p.team).upper().strip()
        pkey=str(p.player_clean_key)
        arow=lock.loc[lock.team.eq(team)&lock.successor_player_clean_key.eq(pkey)]
        if len(arow)!=1:
            raise RuntimeError(f"expected one allocation row {team}/{pkey}")
        arow=arow.iloc[0]
        for market in ("rush_att","rush_yards"):
            arm_data={}
            for name in ("BASELINE","VACANCY_V1_SNAP","GSIS_LINEUP_V1"):
                arr=lookup(sims[name],p,market)
                if arr is None or len(arr)==0:
                    raise RuntimeError(f"missing {name} simulation {team}/{pkey}/{market}")
                arr=np.asarray(arr,dtype=float)
                if not np.isfinite(arr).all():
                    raise RuntimeError(f"non-finite {name} simulation {team}/{pkey}/{market}")
                mc=float(arr.mean())
                mlv=_diag(ml,team,pkey,f"ml_{market}")
                stv=_diag(state,team,pkey,f"state_{market}")
                ens=apply_ensemble(pd.DataFrame([{
                    "market":market,"mc_proj":mc,"ml_proj":mlv,"state_proj":stv
                }]),weights=weights).iloc[0]
                if str(ens["ensemble_status"])!="calibrated":
                    raise RuntimeError(f"uncalibrated ensemble {team}/{pkey}/{market}")
                target_mean=float(ens["ensemble_proj"])
                adj=arr.copy()
                if mc>0 and np.isfinite(target_mean):
                    adj=arr*max(0.0,target_mean/mc)
                adj,align=align_prealigned_outcomes(
                    adj,market=market,eligible=bool(mc>0 and np.isfinite(target_mean)),target_mean=target_mean
                )
                final_mean=float(np.mean(adj))
                if abs(final_mean-target_mean)>1e-8:
                    raise RuntimeError(
                        f"final production mean alignment drift {name} {team}/{pkey}/{market}: "
                        f"{final_mean} vs {target_mean}"
                    )
                arm_data[name]={
                    "mc_mean":mc,
                    "ensemble_mean":target_mean,
                    "final_mean":final_mean,
                    "sd":float(np.std(adj,ddof=1)) if len(adj)>1 else 0.0,
                    "q10":float(np.quantile(adj,.10)),
                    "q50":float(np.quantile(adj,.50)),
                    "q90":float(np.quantile(adj,.90)),
                    "draw_sha256":_array_hash(adj),
                    "count_alignment_applied":int(align["discrete_count_alignment_applied"]),
                }

            row={
                "target_season":int(arow.target_season),
                "target_week":int(arow.target_week),
                "event_id":str(arow.event_id),
                "team":team,
                "opponent":str(p.get("opponent","")),
                "player":p.get("player"),
                "player_clean_key":pkey,
                "position":p.get("position"),
                "market":market,
                "kickoff_utc":str(arow.kickoff_utc),
                "gsis_team_capture_utc":str(arow.gsis_team_capture_utc),
                "gsis_snapshot_sha256":str(arow.gsis_snapshot_sha256),
                "vacated_rush_share":float(arow.vacated_rush_share),
                "snap_transfer_rush_share":float(arow.snap_transfer_rush_share),
                "gsis_transfer_rush_share":float(arow.gsis_transfer_rush_share),
                "baseline_rules_rush_share":float(pd.to_numeric(pd.Series([p.rules_rush_share]),errors="raise").iloc[0]),
                "snap_rules_rush_share":float(pd.to_numeric(pd.Series([p.rules_rush_share]),errors="raise").iloc[0])+float(arow.snap_transfer_rush_share),
                "gsis_rules_rush_share":float(pd.to_numeric(pd.Series([p.rules_rush_share]),errors="raise").iloc[0])+float(arow.gsis_transfer_rush_share),
                "frozen_rules_ypc":float(pd.to_numeric(pd.Series([p.rules_ypc]),errors="raise").iloc[0]),
                "simulation_iterations":ITERATIONS,
                "simulation_seed":SEED,
            }
            for name,prefix in [
                ("BASELINE","baseline"),("VACANCY_V1_SNAP","snap"),("GSIS_LINEUP_V1","gsis")
            ]:
                for k,v in arm_data[name].items():
                    row[f"{prefix}_{k}"]=v
            rows.append(row)

    out=pd.DataFrame(rows)
    if _forbidden(out):
        raise RuntimeError(f"private projection lock contains forbidden fields: {_forbidden(out)}")
    max_yard_eff_gap=0.0
    audit={
        "disposition":"GSIS_RB_SUCCESSOR_LINEUP_V1_THREE_ARM_PROJECTION_LOCKED",
        "target_season":int(pd.to_numeric(lock.target_season,errors="raise").iloc[0]),
        "target_week":int(pd.to_numeric(lock.target_week,errors="raise").iloc[0]),
        "event_teams":len(event_teams),
        "locked_active_rb_fb":int(cohort[["team","player_clean_key"]].drop_duplicates().shape[0]),
        "private_projection_rows":len(out),
        "arms":["BASELINE","VACANCY_V1_SNAP","GSIS_LINEUP_V1"],
        "markets":["rush_att","rush_yards"],
        "simulation_iterations":ITERATIONS,
        "simulation_seed":SEED,
        "candidate_mutated_fields":["rules_rush_share"],
        "ypc_efficiency_changed":False,
        "ml_state_components_changed":False,
        "ensemble_weights_changed":False,
        "week1_rb_p3_override_used":False,
        "production_simulation_seam":"run_pricing_with_full_roster_universe_v3_core._simulate_promoted_stack",
        "production_entitlement_version":production_audit.get("explicit_target_entitlement_version"),
        "te_r5p_full_slate_consumed":bool(production_audit.get("te_r5p_full_slate_consumed")),
        "wr_r15_full_slate_consumed":bool(production_audit.get("wr_r15_full_slate_consumed")),
        "sportsbook_inputs_used":0,
        "target_game_outcomes_attached":0,
        "raw_gsis_rows_emitted_publicly":False,
        "player_identifiers_emitted_publicly":False,
        "private_rows_must_not_be_committed":True,
        "max_yard_efficiency_gap":max_yard_eff_gap,
    }
    return out,audit

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--allocation-lock",type=Path,required=True)
    ap.add_argument("--private-projection-out",type=Path,required=True)
    ap.add_argument("--public-manifest-out",type=Path,required=True)
    ap.add_argument("--source-run",type=int,required=True)
    ap.add_argument("--source-artifact",type=int,required=True)
    ap.add_argument("--source-digest",required=True)
    ap.add_argument("--source-sha",required=True)
    a=ap.parse_args()

    out,audit=build_projection_lock(pd.read_csv(a.allocation_lock,low_memory=False))
    audit.update({
        "source_full_slate_run":int(a.source_run),
        "source_full_slate_artifact":int(a.source_artifact),
        "source_full_slate_digest":str(a.source_digest),
        "production_source_sha":str(a.source_sha),
    })
    a.private_projection_out.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.private_projection_out,index=False)
    a.public_manifest_out.parent.mkdir(parents=True,exist_ok=True)
    a.public_manifest_out.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(audit,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
