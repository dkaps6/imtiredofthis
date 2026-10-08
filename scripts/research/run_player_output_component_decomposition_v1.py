#!/usr/bin/env python3
"""Player Output Component Decomposition V1.

Decomposes final individual point projections into effective workload and
per-opportunity output using frozen 2026 W1-W4 artifacts. Diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.operations.grade_market_track_record_gsis_v1 import load_roster_identity
from scripts.utils.canonical_names import canonicalize_player_name_safe

SEASON=2026
WEEKS={1,2,3,4}
TOL=1e-10
EPS=1e-12

REQ_MARKETS={
    ("QB","pass_attempts"):("pass_yards",),
    ("RB","carries"):("rush_yards",),
    ("RB","targets"):("rec_yards","receptions"),
    ("WR","targets"):("rec_yards","receptions"),
    ("TE","targets"):("rec_yards","receptions"),
}


def _read(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def _num(s)->pd.Series:
    return pd.to_numeric(s,errors="coerce")


def _truthy(s:pd.Series)->pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.astype(str).str.strip().str.lower().isin({"true","1","yes","y"})


def _pos(v)->str:
    p=str(v or "").upper().strip()
    if p in {"FB","HB","TB"} or p.startswith("RB"):
        return "RB"
    if p in {"LWR","RWR","SWR"} or p.startswith("WR"):
        return "WR"
    return p


def _workload_bin(pos:str,opp:str,value:float)->str:
    x=float(value)
    if x<=0: return "ZERO"
    if opp=="pass_attempts":
        if x<=20: return "01_20"
        if x<=30: return "21_30"
        if x<=40: return "31_40"
        return "41_PLUS"
    if opp=="carries":
        if x<=3: return "01_03"
        if x<=8: return "04_08"
        if x<=14: return "09_14"
        return "15_PLUS"
    if opp=="targets":
        if x<=2: return "01_02"
        if x<=5: return "03_05"
        if x<=8: return "06_08"
        return "09_PLUS"
    raise RuntimeError(f"unsupported opportunity type {opp}")


def _score(actual:pd.Series,pred:pd.Series,prefix:str)->dict:
    y=_num(actual)
    p=_num(pred)
    ok=y.notna() & p.notna()
    y=y.loc[ok]; p=p.loc[ok]
    if not len(y):
        return {
            f"{prefix}_rows":0,
            f"{prefix}_mae":np.nan,
            f"{prefix}_bias":np.nan,
            f"{prefix}_rmse":np.nan,
            f"{prefix}_median_ae":np.nan,
        }
    e=p-y
    return {
        f"{prefix}_rows":int(len(e)),
        f"{prefix}_mae":float(e.abs().mean()),
        f"{prefix}_bias":float(e.mean()),
        f"{prefix}_rmse":float(np.sqrt(np.mean(np.square(e)))),
        f"{prefix}_median_ae":float(e.abs().median()),
    }


def _closer(g:pd.DataFrame,cand_col:str)->dict:
    base=(g["baseline_projection"]-g["actual_output"]).abs()
    cand=(g[cand_col]-g["actual_output"]).abs()
    ok=base.notna() & cand.notna()
    base=base.loc[ok]; cand=cand.loc[ok]
    return {
        "candidate_closer":int((cand < base-TOL).sum()),
        "baseline_closer":int((base < cand-TOL).sum()),
        "ties":int((np.abs(cand-base)<=TOL).sum()),
    }


def _summary(g:pd.DataFrame)->dict:
    out={"rows":int(len(g))}
    out.update(_score(g["actual_output"],g["baseline_projection"],"baseline"))
    out.update(_score(g["actual_output"],g["opportunity_oracle"],"opportunity_oracle"))

    bmae=out["baseline_mae"]
    omae=out["opportunity_oracle_mae"]
    out["opportunity_oracle_mae_improvement"]=float(bmae-omae) if np.isfinite(bmae) and np.isfinite(omae) else np.nan
    out["opportunity_oracle_fraction_mae_removed"]=float((bmae-omae)/bmae) if np.isfinite(bmae) and bmae>0 and np.isfinite(omae) else np.nan
    out.update({f"opportunity_{k}":v for k,v in _closer(g,"opportunity_oracle").items()})

    eff=g.loc[g["efficiency_oracle_eligible"].fillna(False).astype(bool)].copy()
    out["efficiency_eligible_rows"]=int(len(eff))
    if len(eff):
        eb=_score(eff["actual_output"],eff["baseline_projection"],"efficiency_subset_baseline")
        eo=_score(eff["actual_output"],eff["efficiency_oracle"],"efficiency_oracle")
        out.update(eb); out.update(eo)
        bm=eb["efficiency_subset_baseline_mae"]; em=eo["efficiency_oracle_mae"]
        out["efficiency_oracle_mae_improvement"]=float(bm-em)
        out["efficiency_oracle_fraction_mae_removed"]=float((bm-em)/bm) if bm>0 else np.nan
        out.update({f"efficiency_{k}":v for k,v in _closer(eff,"efficiency_oracle").items()})
    else:
        out.update({
            "efficiency_subset_baseline_rows":0,
            "efficiency_subset_baseline_mae":np.nan,
            "efficiency_subset_baseline_bias":np.nan,
            "efficiency_subset_baseline_rmse":np.nan,
            "efficiency_subset_baseline_median_ae":np.nan,
            "efficiency_oracle_rows":0,
            "efficiency_oracle_mae":np.nan,
            "efficiency_oracle_bias":np.nan,
            "efficiency_oracle_rmse":np.nan,
            "efficiency_oracle_median_ae":np.nan,
            "efficiency_oracle_mae_improvement":np.nan,
            "efficiency_oracle_fraction_mae_removed":np.nan,
            "efficiency_candidate_closer":0,
            "efficiency_baseline_closer":0,
            "efficiency_ties":0,
        })

    out["zero_actual_opportunity_rate"]=float(g["actual_opportunities"].eq(0).mean())
    out["mean_predicted_opportunities"]=float(g["predicted_opportunities"].mean())
    out["mean_actual_opportunities"]=float(g["actual_opportunities"].mean())
    out["mean_model_effective_efficiency"]=float(g["model_effective_efficiency"].mean())

    opp_err=g["predicted_opportunities"]-g["actual_opportunities"]
    output_err=g["baseline_projection"]-g["actual_output"]
    out["opportunity_error_vs_output_error_pearson"]=(
        float(opp_err.corr(output_err))
        if opp_err.nunique()>1 and output_err.nunique()>1 else np.nan
    )
    e2=g.loc[g["efficiency_oracle_eligible"].fillna(False).astype(bool)]
    if len(e2):
        eff_err=e2["model_effective_efficiency"]-e2["actual_efficiency"]
        oe=e2["baseline_projection"]-e2["actual_output"]
        out["efficiency_error_vs_output_error_pearson"]=(
            float(eff_err.corr(oe)) if eff_err.nunique()>1 and oe.nunique()>1 else np.nan
        )
    else:
        out["efficiency_error_vs_output_error_pearson"]=np.nan
    return out


def _name_key(value)->str:
    try:
        _, key=canonicalize_player_name_safe(value)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _to_pandas(obj)->pd.DataFrame:
    if isinstance(obj,pd.DataFrame):
        return obj.copy()
    if hasattr(obj,"to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _resolve_pbp_target_identity_rows(
    target_events:pd.DataFrame,
    roster_rows:pd.DataFrame,
)->pd.DataFrame:
    """Resolve completed-game receiver IDs to canonical full-name aliases.

    Receiver GSIS ID is primary. Weekly roster aliases are identity-only and may
    expose multiple validated names for the same stable ID. PBP receiver-name
    fallback is allowed only when a targeted ID has no eligible roster alias.
    """
    t=target_events.copy()
    r=roster_rows.copy()

    required_t={"week","team","receiver_player_id"}
    missing_t=required_t-set(t.columns)
    if missing_t:
        raise RuntimeError(f"target events missing columns: {sorted(missing_t)}")
    if "receiver_player_name" not in t.columns:
        t["receiver_player_name"]=""

    required_r={"week","receiver_player_id","player_clean_key"}
    missing_r=required_r-set(r.columns)
    if missing_r:
        raise RuntimeError(f"roster identity rows missing columns: {sorted(missing_r)}")

    t["week"]=_num(t["week"])
    t["team"]=t["team"].map(canon_team)
    t["receiver_player_id"]=t["receiver_player_id"].fillna("").astype(str).str.strip()
    t["pbp_receiver_name_key"]=t["receiver_player_name"].map(_name_key)
    r["week"]=_num(r["week"])
    r["receiver_player_id"]=r["receiver_player_id"].fillna("").astype(str).str.strip()
    r["player_clean_key"]=r["player_clean_key"].fillna("").astype(str).str.strip()

    t=t.loc[
        t["week"].isin(WEEKS)
        & t["team"].astype(str).ne("")
        & t["receiver_player_id"].ne("")
    ].copy()
    r=r.loc[
        r["week"].notna()
        & r["receiver_player_id"].ne("")
        & r["player_clean_key"].ne("")
    ].copy()

    multi_team=(
        t.groupby(["week","receiver_player_id"])["team"]
        .nunique()
        .reset_index(name="offense_count")
    )
    if multi_team["offense_count"].gt(1).any():
        bad=multi_team.loc[multi_team["offense_count"].gt(1)].head(20)
        raise RuntimeError(
            f"receiver appeared for multiple PBP offenses in one week: {bad.to_dict('records')}"
        )

    grouped=[]
    for (week,pid),g in t.groupby(["week","receiver_player_id"],sort=True):
        names=sorted({x for x in g["pbp_receiver_name_key"].astype(str) if x})
        if len(names)>1:
            raise RuntimeError(
                f"PBP receiver ID has multiple canonical names week={int(week)} "
                f"player_id={pid} names={names}"
            )
        grouped.append({
            "week":int(week),
            "receiver_player_id":str(pid),
            "pbp_actual_targets":float(len(g)),
            "pbp_actual_team":str(g["team"].iloc[0]),
            "pbp_receiver_name_key":names[0] if names else "",
        })
    counts=pd.DataFrame(grouped)
    if counts.empty:
        return pd.DataFrame(columns=[
            "season","week","player_clean_key","receiver_player_id",
            "pbp_actual_team","pbp_actual_targets",
            "pbp_target_identity_resolved","pbp_identity_route",
        ])

    rows=[]
    for rec in counts.to_dict("records"):
        week=int(rec["week"]); pid=str(rec["receiver_player_id"])
        aliases=sorted(set(
            r.loc[
                r["receiver_player_id"].eq(pid) & r["week"].le(week),
                "player_clean_key",
            ].astype(str)
        ))
        aliases=[a for a in aliases if a]
        route="ROSTER_GSIS_ALIAS"
        if not aliases:
            fallback=str(rec.get("pbp_receiver_name_key","") or "")
            if not fallback:
                raise RuntimeError(
                    f"PBP target receiver has no roster alias or unique PBP name "
                    f"week={week} player_id={pid}"
                )
            aliases=[fallback]
            route="PBP_UNIQUE_NAME_FALLBACK"
        for key in aliases:
            rows.append({
                "season":SEASON,
                "week":week,
                "player_clean_key":key,
                "receiver_player_id":pid,
                "pbp_actual_team":rec["pbp_actual_team"],
                "pbp_actual_targets":rec["pbp_actual_targets"],
                "pbp_target_identity_resolved":True,
                "pbp_identity_route":route,
            })

    out=pd.DataFrame(rows).drop_duplicates()
    dup=out.duplicated(["season","week","player_clean_key"],keep=False)
    if dup.any():
        bad=out.loc[dup,[
            "week","player_clean_key","receiver_player_id","pbp_actual_team"
        ]].head(20)
        raise RuntimeError(
            f"ambiguous PBP target identity after GSIS alias resolution: {bad.to_dict('records')}"
        )
    return out.sort_values(
        ["season","week","player_clean_key","receiver_player_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_pbp_target_actuals()->pd.DataFrame:
    """Completed-game target counts for grading only, never prediction input."""
    import nflreadpy as nfl

    pbp=_to_pandas(nfl.load_pbp(seasons=[SEASON]))
    pbp.columns=[str(c).strip().lower() for c in pbp.columns]
    required={"week","posteam","receiver_player_id","pass_attempt","sack"}
    missing=required-set(pbp.columns)
    if missing:
        raise RuntimeError(f"PBP target authority missing columns: {sorted(missing)}")
    if "season_type" in pbp.columns:
        reg=pbp["season_type"].astype(str).str.upper().eq("REG")
        if reg.any():
            pbp=pbp.loc[reg].copy()
    if "two_point_attempt" not in pbp.columns:
        pbp["two_point_attempt"]=0.0
    if "receiver_player_name" not in pbp.columns:
        pbp["receiver_player_name"]=""

    pbp["week"]=_num(pbp["week"])
    pbp=pbp.loc[pbp["week"].isin(WEEKS)].copy()
    pbp["team"]=pbp["posteam"].map(canon_team)
    pbp["receiver_player_id"]=pbp["receiver_player_id"].fillna("").astype(str).str.strip()
    target=(
        _num(pbp["pass_attempt"]).fillna(0).eq(1)
        & ~_num(pbp["sack"]).fillna(0).eq(1)
        & ~_num(pbp["two_point_attempt"]).fillna(0).eq(1)
        & pbp["receiver_player_id"].ne("")
        & pbp["team"].astype(str).ne("")
    )
    target_events=pbp.loc[
        target,
        ["week","team","receiver_player_id","receiver_player_name"],
    ].copy()

    raw_roster=_to_pandas(nfl.load_rosters_weekly(SEASON))
    if raw_roster.empty:
        raise RuntimeError("weekly roster identity source returned zero rows")
    raw_roster.columns=[str(c).strip().lower() for c in raw_roster.columns]
    rid_col=next((c for c in ("gsis_id","player_id") if c in raw_roster.columns),None)
    name_col=next((
        c for c in ("full_name","football_name","player_name","player","name")
        if c in raw_roster.columns
    ),None)
    if rid_col is None or name_col is None or "week" not in raw_roster.columns:
        raise RuntimeError("weekly roster source cannot bridge PBP receiver GSIS identities")

    roster=pd.DataFrame({
        "week":_num(raw_roster["week"]),
        "receiver_player_id":raw_roster[rid_col].fillna("").astype(str).str.strip(),
        "player_clean_key":raw_roster[name_col].map(_name_key),
    })
    roster=roster.loc[
        roster["week"].isin(WEEKS)
        & roster["receiver_player_id"].ne("")
        & roster["player_clean_key"].astype(str).ne("")
    ].drop_duplicates()

    return _resolve_pbp_target_identity_rows(target_events,roster)


def build_rows(
    points:pd.DataFrame,
    opps:pd.DataFrame,
    *,
    pbp_targets:pd.DataFrame|None=None,
    require_full_weeks:bool=True,
)->pd.DataFrame:
    p=points.copy()
    o=opps.copy()

    required_p={
        "season","week","event_id","team","player","player_clean_key",
        "position_family","market","projection_mean","actual","actual_opportunities",
        "sportsbook_inputs_used_upstream",
    }
    required_o={
        "season","week","event_id","team","player","player_clean_key",
        "position_family","opportunity_type","predicted_opportunities",
        "actual_opportunities","sportsbook_inputs_used_upstream",
    }
    mp=required_p-set(p.columns); mo=required_o-set(o.columns)
    if mp: raise RuntimeError(f"point scoreboard missing columns: {sorted(mp)}")
    if mo: raise RuntimeError(f"opportunity rows missing columns: {sorted(mo)}")

    if _truthy(p["sportsbook_inputs_used_upstream"]).any():
        raise RuntimeError("sportsbook leakage in point parent")
    if _truthy(o["sportsbook_inputs_used_upstream"]).any():
        raise RuntimeError("sportsbook leakage in opportunity parent")

    p["season"]=_num(p["season"]); p["week"]=_num(p["week"])
    o["season"]=_num(o["season"]); o["week"]=_num(o["week"])
    p=p.loc[p["season"].eq(SEASON)&p["week"].isin(WEEKS)].copy()
    o=o.loc[o["season"].eq(SEASON)&o["week"].isin(WEEKS)].copy()
    if require_full_weeks:
        if set(p["week"].dropna().astype(int).unique())!=WEEKS:
            raise RuntimeError("point parent does not cover exact W1-W4")
        if set(o["week"].dropna().astype(int).unique())!=WEEKS:
            raise RuntimeError("opportunity parent does not cover exact W1-W4")

    p["position_family"]=p["position_family"].map(_pos)
    o["position_family"]=o["position_family"].map(_pos)
    for c in ("projection_mean","actual","actual_opportunities"):
        p[c]=_num(p[c])
    for c in ("predicted_opportunities","actual_opportunities"):
        o[c]=_num(o[c])

    # Expand each frozen opportunity row only into its predeclared output market(s).
    expanded=[]
    for _,r in o.iterrows():
        key=(str(r["position_family"]),str(r["opportunity_type"]))
        markets=REQ_MARKETS.get(key)
        if not markets:
            continue
        for market in markets:
            z=r.to_dict()
            z["market"]=market
            expanded.append(z)
    e=pd.DataFrame(expanded)
    if e.empty:
        raise RuntimeError("zero expanded opportunity rows")

    keys=["season","week","event_id","team","player_clean_key","position_family","market"]
    if e.duplicated(keys).any():
        bad=e.loc[e.duplicated(keys,keep=False),keys].head(20)
        raise RuntimeError(f"duplicate expanded opportunity identity: {bad.to_dict('records')}")

    point_keep=keys+["player","projection_mean","actual","actual_opportunities"]
    pp=p.loc[
        p.apply(
            lambda r: str(r["market"]) in REQ_MARKETS.get(
                (str(r["position_family"]),
                 "pass_attempts" if str(r["position_family"])=="QB"
                 else ("carries" if str(r["market"])=="rush_yards" else "targets")),
                ()
            ),
            axis=1,
        ),
        point_keep,
    ].copy()
    if pp.duplicated(keys).any():
        raise RuntimeError("duplicate point identity after market filter")

    e=e.rename(columns={
        "player":"opportunity_player",
        "actual_opportunities":"opportunity_actual_opportunities",
    })
    joined=e.merge(pp,on=keys,how="left",validate="one_to_one")
    missing=joined["projection_mean"].isna()
    if missing.any():
        bad=joined.loc[missing,keys+["opportunity_player"]].head(30)
        raise RuntimeError(f"replay-matched opportunity identities missing final point rows: {bad.to_dict('records')}")

    gap=(joined["opportunity_actual_opportunities"]-joined["actual_opportunities"]).abs()
    if float(gap.max())>TOL:
        bad=joined.loc[gap.gt(TOL),keys+["opportunity_actual_opportunities","actual_opportunities"]].head(20)
        raise RuntimeError(f"actual opportunity identity drift: {bad.to_dict('records')}")

    out=pd.DataFrame({
        "season":joined["season"].astype(int),
        "week":joined["week"].astype(int),
        "event_id":joined["event_id"],
        "team":joined["team"],
        "player":joined["player"],
        "player_clean_key":joined["player_clean_key"],
        "position_family":joined["position_family"],
        "market":joined["market"],
        "opportunity_type":joined["opportunity_type"],
        "predicted_opportunities":joined["predicted_opportunities"],
        "artifact_actual_opportunities":joined["actual_opportunities"],
        "baseline_projection":joined["projection_mean"],
        "actual_output":joined["actual"],
    })

    out["actual_opportunities"]=out["artifact_actual_opportunities"]
    out["actual_opportunity_source"]="FROZEN_REPLAY_MATCHED_ARTIFACT"
    out["pbp_actual_targets"]=np.nan
    out["pbp_target_identity_resolved"]=False
    out["actual_opportunity_discrepancy"]=False
    out["pbp_actual_team"]=""
    out["pbp_identity_route"]=""
    out["grading_identity_valid"]=True
    out["grading_exclusion_reason"]=""

    target_rows=out["opportunity_type"].eq("targets")
    if pbp_targets is not None:
        pt=pbp_targets.copy()
        required_pt={
            "season","week","player_clean_key","pbp_actual_targets",
            "pbp_actual_team","pbp_target_identity_resolved",
        }
        missing_pt=required_pt-set(pt.columns)
        if missing_pt:
            raise RuntimeError(f"PBP target authority missing columns: {sorted(missing_pt)}")
        if "pbp_identity_route" not in pt.columns:
            pt["pbp_identity_route"]="INJECTED_PBP_TARGET_AUTHORITY"
        if pt.duplicated(["season","week","player_clean_key"]).any():
            raise RuntimeError("duplicate PBP target identity")
        out=out.merge(
            pt[[
                "season","week","player_clean_key","pbp_actual_targets",
                "pbp_actual_team","pbp_identity_route","pbp_target_identity_resolved"
            ]].rename(columns={
                "pbp_actual_targets":"_pbp_actual_targets",
                "pbp_actual_team":"_pbp_actual_team",
                "pbp_identity_route":"_pbp_identity_route",
                "pbp_target_identity_resolved":"_pbp_target_identity_resolved",
            }),
            on=["season","week","player_clean_key"],
            how="left",
            validate="many_to_one",
        )
        resolved=target_rows & out["_pbp_target_identity_resolved"].fillna(False).astype(bool)
        unresolved=target_rows & ~resolved
        safe_zero=(
            unresolved
            & out["artifact_actual_opportunities"].abs().le(TOL)
            & out["actual_output"].abs().le(TOL)
        )
        bad_unresolved=unresolved & ~safe_zero
        conflict_keys=out.loc[
            bad_unresolved,["season","week","player_clean_key"]
        ].drop_duplicates()
        if not conflict_keys.empty:
            conflict_keys["_grading_source_conflict"]=True
            out=out.merge(
                conflict_keys,
                on=["season","week","player_clean_key"],
                how="left",
                validate="many_to_one",
            )
            conflict_all=out["_grading_source_conflict"].fillna(False).astype(bool)
            out.loc[conflict_all,"grading_identity_valid"]=False
            out.loc[
                conflict_all,"grading_exclusion_reason"
            ]="GRADING_SOURCE_CONFLICT_UNRESOLVED_PBP_TARGET"
            conflict_target=(
                out["opportunity_type"].eq("targets")
                & out["grading_exclusion_reason"].eq(
                    "GRADING_SOURCE_CONFLICT_UNRESOLVED_PBP_TARGET"
                )
            )
            out.loc[
                conflict_target,"actual_opportunity_source"
            ]="GRADING_SOURCE_CONFLICT_UNRESOLVED_PBP_TARGET"
            out.drop(columns=["_grading_source_conflict"],inplace=True)

        out.loc[resolved,"pbp_actual_targets"]=_num(
            out.loc[resolved,"_pbp_actual_targets"]
        ).to_numpy()
        out.loc[resolved,"pbp_actual_team"]=out.loc[resolved,"_pbp_actual_team"].fillna("").map(canon_team)
        out.loc[resolved,"pbp_identity_route"]=out.loc[resolved,"_pbp_identity_route"].fillna("").astype(str)
        out.loc[resolved,"pbp_target_identity_resolved"]=True
        out.loc[resolved,"actual_opportunities"]=out.loc[resolved,"pbp_actual_targets"]
        out.loc[resolved,"actual_opportunity_source"]="COMPLETED_GAME_PBP_TARGETS"
        out.loc[resolved,"actual_opportunity_discrepancy"]=(
            out.loc[resolved,"artifact_actual_opportunities"]
            - out.loc[resolved,"actual_opportunities"]
        ).abs().gt(TOL)

        team_mismatch=(
            resolved
            & out["pbp_actual_targets"].gt(0)
            & out["pbp_actual_team"].astype(str).ne("")
            & out["pbp_actual_team"].map(canon_team).ne(out["team"].map(canon_team))
        )
        mismatch_keys=out.loc[
            team_mismatch,["season","week","player_clean_key"]
        ].drop_duplicates()
        if not mismatch_keys.empty:
            mismatch_keys["_grading_identity_mismatch"]=True
            out=out.merge(
                mismatch_keys,
                on=["season","week","player_clean_key"],
                how="left",
                validate="many_to_one",
            )
            bad_all=out["_grading_identity_mismatch"].fillna(False).astype(bool)
            out.loc[bad_all,"grading_identity_valid"]=False
            out.loc[bad_all,"grading_exclusion_reason"]="HISTORICAL_TEAM_IDENTITY_MISMATCH"
            out.drop(columns=["_grading_identity_mismatch"],inplace=True)

        out.loc[safe_zero,"pbp_target_identity_resolved"]=False
        out.loc[safe_zero,"actual_opportunities"]=0.0
        out.loc[safe_zero,"actual_opportunity_source"]="FROZEN_ZERO_NO_RECEIVING_USAGE"
        out.loc[safe_zero,"actual_opportunity_discrepancy"]=False
        out.drop(
            columns=[
                "_pbp_actual_targets","_pbp_actual_team","_pbp_identity_route",
                "_pbp_target_identity_resolved"
            ],
            inplace=True,
        )
    if out[["predicted_opportunities","actual_opportunities","baseline_projection","actual_output"]].isna().any().any():
        raise RuntimeError("missing numeric value in paired decomposition rows")

    out["model_efficiency_eligible"]=out["predicted_opportunities"].gt(EPS)
    if not out["model_efficiency_eligible"].all():
        # Retain for audit but oracle decomposition is unavailable on these rows.
        pass
    out["model_effective_efficiency"]=np.where(
        out["model_efficiency_eligible"],
        out["baseline_projection"]/out["predicted_opportunities"],
        np.nan,
    )
    out["baseline_reconstructed"]=(
        out["predicted_opportunities"]*out["model_effective_efficiency"]
    )
    recon=(out["baseline_reconstructed"]-out["baseline_projection"]).abs()
    finite=recon.loc[out["model_efficiency_eligible"]]
    if len(finite) and float(finite.max())>TOL:
        raise RuntimeError(f"baseline algebraic reconstruction failed max_gap={float(finite.max())}")

    zero_actual=out["actual_opportunities"].le(EPS)
    bad_zero=zero_actual & out["actual_output"].abs().gt(TOL) & out["grading_identity_valid"]
    if bad_zero.any():
        bad=out.loc[bad_zero,[
            "week","team","player","position_family","market",
            "actual_opportunities","actual_output"
        ]].head(20)
        raise RuntimeError(f"nonzero output with zero actual opportunity: {bad.to_dict('records')}")

    valid_grade=out["grading_identity_valid"].fillna(False).astype(bool)
    out["actual_efficiency"]=np.where(
        valid_grade & (~zero_actual),
        out["actual_output"]/out["actual_opportunities"],
        np.where(valid_grade & zero_actual,0.0,np.nan),
    )
    out["component_decomposition_eligible"]=(
        out["model_efficiency_eligible"] & valid_grade
    )
    out["efficiency_oracle_eligible"]=(
        out["component_decomposition_eligible"] & (~zero_actual)
    )
    out["opportunity_oracle"]=np.where(
        out["component_decomposition_eligible"],
        out["actual_opportunities"]*out["model_effective_efficiency"],
        np.nan,
    )
    out["efficiency_oracle"]=np.where(
        out["efficiency_oracle_eligible"],
        out["predicted_opportunities"]*out["actual_efficiency"],
        np.nan,
    )
    out["full_oracle"]=np.where(
        valid_grade,
        out["actual_opportunities"]*out["actual_efficiency"],
        np.nan,
    )
    full_gap=(out.loc[valid_grade,"full_oracle"]-out.loc[valid_grade,"actual_output"]).abs()
    if len(full_gap) and float(full_gap.max())>TOL:
        raise RuntimeError(f"full actual identity failed max_gap={float(full_gap.max())}")

    out["baseline_error"]=out["baseline_projection"]-out["actual_output"]
    out["opportunity_error"]=out["predicted_opportunities"]-out["actual_opportunities"]
    out["effective_efficiency_error"]=np.where(
        out["efficiency_oracle_eligible"],
        out["model_effective_efficiency"]-out["actual_efficiency"],
        np.nan,
    )
    out["actual_workload_bin"]=[
        _workload_bin(str(pos),str(opp),float(v))
        for pos,opp,v in zip(
            out["position_family"],out["opportunity_type"],out["actual_opportunities"]
        )
    ]
    out["sportsbook_inputs_used_upstream"]=False
    out["parameters_fit"]=0
    out["automatic_promotion"]=False
    return out


def run(*,points_path:Path,opportunity_path:Path,out_dir:Path)->dict:
    out_dir.mkdir(parents=True,exist_ok=True)
    points=_read(points_path,"all-player point scoreboard")
    opps=_read(opportunity_path,"replay-matched opportunity rows")
    pbp_targets=build_pbp_target_actuals()
    rows=build_rows(points,opps,pbp_targets=pbp_targets)

    scoreable=rows.loc[rows["component_decomposition_eligible"]].copy()
    if scoreable.empty:
        raise RuntimeError("zero scoreable decomposition rows")

    market_rows=[]
    week_rows=[]
    for (pos,market),g in scoreable.groupby(["position_family","market"],sort=True):
        rec={"position_family":pos,"market":market}
        rec.update(_summary(g))
        market_rows.append(rec)
        for week,w in g.groupby("week",sort=True):
            rr={"position_family":pos,"market":market,"week":int(week)}
            rr.update(_summary(w))
            week_rows.append(rr)

    workload_rows=[]
    for (pos,market,bin_name),g in scoreable.groupby(
        ["position_family","market","actual_workload_bin"],sort=True
    ):
        rr={
            "position_family":pos,"market":market,
            "actual_workload_bin":bin_name,
        }
        rr.update(_summary(g))
        workload_rows.append(rr)

    market=pd.DataFrame(market_rows)
    week=pd.DataFrame(week_rows)
    workload=pd.DataFrame(workload_rows)

    rows.to_csv(out_dir/"player_output_component_rows.csv",index=False)
    market.to_csv(out_dir/"player_output_component_market_summary.csv",index=False)
    week.to_csv(out_dir/"player_output_component_week_summary.csv",index=False)
    workload.to_csv(out_dir/"player_output_component_workload_summary.csv",index=False)

    dispositions=[]
    for r in market.itertuples(index=False):
        oi=float(r.opportunity_oracle_fraction_mae_removed)
        ei=float(r.efficiency_oracle_fraction_mae_removed)
        if np.isfinite(oi) and np.isfinite(ei):
            label="OPPORTUNITY_DOMINANT" if oi>ei else ("EFFICIENCY_DOMINANT" if ei>oi else "MIXED_EQUAL")
        else:
            label="INCONCLUSIVE"
        dispositions.append({
            "position_family":r.position_family,
            "market":r.market,
            "descriptive_dominance":label,
            "opportunity_fraction_mae_removed":oi,
            "efficiency_fraction_mae_removed":ei,
        })

    payload={
        "version":"PLAYER_OUTPUT_COMPONENT_DECOMPOSITION_V1",
        "season":SEASON,
        "weeks":sorted(WEEKS),
        "paired_rows":int(len(rows)),
        "scoreable_rows":int(len(scoreable)),
        "model_efficiency_unavailable_rows":int((~rows["model_efficiency_eligible"]).sum()),
        "grading_exclusion_rows":int((~rows["grading_identity_valid"]).sum()),
        "grading_exclusion_player_weeks":int(
            rows.loc[~rows["grading_identity_valid"],["week","player_clean_key"]]
            .drop_duplicates().shape[0]
        ),
        "grading_exclusion_reason_counts":(
            rows.loc[~rows["grading_identity_valid"],"grading_exclusion_reason"]
            .value_counts(dropna=False).astype(int).to_dict()
        ),
        "grading_identity_mismatch_rows":int(
            rows["grading_exclusion_reason"].eq("HISTORICAL_TEAM_IDENTITY_MISMATCH").sum()
        ),
        "grading_identity_mismatch_player_weeks":int(
            rows.loc[
                rows["grading_exclusion_reason"].eq("HISTORICAL_TEAM_IDENTITY_MISMATCH"),
                ["week","player_clean_key"]
            ].drop_duplicates().shape[0]
        ),
        "grading_source_conflict_rows":int(
            rows["grading_exclusion_reason"].eq(
                "GRADING_SOURCE_CONFLICT_UNRESOLVED_PBP_TARGET"
            ).sum()
        ),
        "grading_source_conflict_player_weeks":int(
            rows.loc[
                rows["grading_exclusion_reason"].eq(
                    "GRADING_SOURCE_CONFLICT_UNRESOLVED_PBP_TARGET"
                ),
                ["week","player_clean_key"]
            ].drop_duplicates().shape[0]
        ),
        "actual_opportunity_discrepancy_rows":int(rows["actual_opportunity_discrepancy"].sum()),
        "target_rows_graded_by_pbp":int(rows["actual_opportunity_source"].eq("COMPLETED_GAME_PBP_TARGETS").sum()),
        "target_zero_usage_fallback_rows":int(rows["actual_opportunity_source"].eq("FROZEN_ZERO_NO_RECEIVING_USAGE").sum()),
        "max_baseline_reconstruction_gap":float(
            (scoreable["baseline_reconstructed"]-scoreable["baseline_projection"]).abs().max()
        ),
        "max_full_actual_identity_gap":float(
            (
                rows.loc[rows["grading_identity_valid"],"full_oracle"]
                - rows.loc[rows["grading_identity_valid"],"actual_output"]
            ).abs().max()
        ),
        "market_summary":market.to_dict("records"),
        "descriptive_dominance":dispositions,
        "parameters_fit":0,
        "threshold_searches":0,
        "sportsbook_inputs_used_upstream":False,
        "paid_odds_api_used":False,
        "automatic_promotion":False,
        "interpretation_boundary":"DIAGNOSTIC_ORACLE_DECOMPOSITION_NOT_CAUSAL_MODEL",
    }
    (out_dir/"player_output_component_summary.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(payload,indent=2,sort_keys=True,default=str))
    return payload


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--points",type=Path,required=True)
    p.add_argument("--opportunity-rows",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    run(points_path=a.points,opportunity_path=a.opportunity_rows,out_dir=a.out_dir)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
