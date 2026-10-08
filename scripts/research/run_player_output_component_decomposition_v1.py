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


def build_pbp_target_actuals()->pd.DataFrame:
    """Completed-game target counts for grading only, never prediction input."""
    import nflreadpy as nfl

    pbp=nfl.load_pbp(seasons=[SEASON])
    x=pbp.to_pandas() if hasattr(pbp,"to_pandas") else pd.DataFrame(pbp)
    x.columns=[str(c).strip().lower() for c in x.columns]
    required={"week","posteam","receiver_player_id","pass_attempt","sack"}
    missing=required-set(x.columns)
    if missing:
        raise RuntimeError(f"PBP target authority missing columns: {sorted(missing)}")
    if "season_type" in x.columns:
        reg=x["season_type"].astype(str).str.upper().eq("REG")
        if reg.any():
            x=x.loc[reg].copy()
    if "two_point_attempt" not in x.columns:
        x["two_point_attempt"]=0.0
    x["week"]=_num(x["week"])
    x=x.loc[x["week"].isin(WEEKS)].copy()
    x["team"]=x["posteam"].map(canon_team)
    x["receiver_player_id"]=x["receiver_player_id"].fillna("").astype(str).str.strip()
    target=(
        _num(x["pass_attempt"]).fillna(0).eq(1)
        & ~_num(x["sack"]).fillna(0).eq(1)
        & ~_num(x["two_point_attempt"]).fillna(0).eq(1)
        & x["receiver_player_id"].ne("")
        & x["team"].astype(str).ne("")
    )
    counts=(
        x.loc[target]
        .groupby(["week","team","receiver_player_id"],as_index=False)
        .size()
        .rename(columns={"size":"pbp_actual_targets"})
    )

    weekly=nfl.load_player_stats(seasons=[SEASON],summary_level="week")
    w=weekly.to_pandas() if hasattr(weekly,"to_pandas") else pd.DataFrame(weekly)
    w.columns=[str(c).strip().lower() for c in w.columns]
    id_col=next((c for c in ("player_id","gsis_id","player_gsis_id") if c in w.columns),None)
    name_col=next((c for c in ("player_display_name","player_name","display_name","name") if c in w.columns),None)
    team_col=next((c for c in ("recent_team","team","posteam") if c in w.columns),None)
    if id_col is None or name_col is None or team_col is None or "week" not in w.columns:
        raise RuntimeError("weekly identity source cannot bridge PBP receiver IDs")
    w["week"]=_num(w["week"])
    w=w.loc[w["week"].isin(WEEKS)].copy()
    w["team"]=w[team_col].map(canon_team)
    w["receiver_player_id"]=w[id_col].fillna("").astype(str).str.strip()
    w["player_clean_key"]=w[name_col].map(_name_key)
    ident=w.loc[
        w["receiver_player_id"].ne("")
        & w["player_clean_key"].ne("")
        & w["team"].astype(str).ne(""),
        ["week","team","receiver_player_id","player_clean_key"],
    ].drop_duplicates()

    if ident.duplicated(["week","team","receiver_player_id"]).any():
        bad=ident.loc[
            ident.duplicated(["week","team","receiver_player_id"],keep=False)
        ].head(20)
        raise RuntimeError(f"ambiguous weekly receiver identity bridge: {bad.to_dict('records')}")

    # Retain every identified weekly player so a genuine zero-target game is a
    # real zero rather than an unresolved merge.
    out=ident.merge(
        counts,
        on=["week","team","receiver_player_id"],
        how="left",
        validate="one_to_one",
    )
    out["pbp_actual_targets"]=_num(out["pbp_actual_targets"]).fillna(0.0)
    out["season"]=SEASON
    out["pbp_target_identity_resolved"]=True
    return out[
        ["season","week","team","player_clean_key","receiver_player_id",
         "pbp_actual_targets","pbp_target_identity_resolved"]
    ].drop_duplicates(["season","week","team","player_clean_key"])


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

    target_rows=out["opportunity_type"].eq("targets")
    if pbp_targets is not None:
        pt=pbp_targets.copy()
        required_pt={
            "season","week","team","player_clean_key","pbp_actual_targets",
            "pbp_target_identity_resolved",
        }
        missing_pt=required_pt-set(pt.columns)
        if missing_pt:
            raise RuntimeError(f"PBP target authority missing columns: {sorted(missing_pt)}")
        if pt.duplicated(["season","week","team","player_clean_key"]).any():
            raise RuntimeError("duplicate PBP target identity")
        out=out.merge(
            pt[[
                "season","week","team","player_clean_key","pbp_actual_targets",
                "pbp_target_identity_resolved"
            ]].rename(columns={
                "pbp_actual_targets":"_pbp_actual_targets",
                "pbp_target_identity_resolved":"_pbp_target_identity_resolved",
            }),
            on=["season","week","team","player_clean_key"],
            how="left",
            validate="many_to_one",
        )
        unresolved=target_rows & ~out["_pbp_target_identity_resolved"].fillna(False).astype(bool)
        if unresolved.any():
            bad=out.loc[
                unresolved,
                ["week","team","player","player_clean_key","market"]
            ].drop_duplicates().head(30)
            raise RuntimeError(f"unresolved PBP target identities: {bad.to_dict('records')}")
        out.loc[target_rows,"pbp_actual_targets"]=_num(
            out.loc[target_rows,"_pbp_actual_targets"]
        ).to_numpy()
        out.loc[target_rows,"pbp_target_identity_resolved"]=True
        out.loc[target_rows,"actual_opportunities"]=out.loc[target_rows,"pbp_actual_targets"]
        out.loc[target_rows,"actual_opportunity_source"]="COMPLETED_GAME_PBP_TARGETS"
        out.loc[target_rows,"actual_opportunity_discrepancy"]=(
            out.loc[target_rows,"artifact_actual_opportunities"]
            - out.loc[target_rows,"actual_opportunities"]
        ).abs().gt(TOL)
        out.drop(columns=["_pbp_actual_targets","_pbp_target_identity_resolved"],inplace=True)
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
    bad_zero=zero_actual & out["actual_output"].abs().gt(TOL)
    if bad_zero.any():
        bad=out.loc[bad_zero,[
            "week","team","player","position_family","market",
            "actual_opportunities","actual_output"
        ]].head(20)
        raise RuntimeError(f"nonzero output with zero actual opportunity: {bad.to_dict('records')}")

    out["actual_efficiency"]=np.where(
        ~zero_actual,
        out["actual_output"]/out["actual_opportunities"],
        0.0,
    )
    out["efficiency_oracle_eligible"]=out["model_efficiency_eligible"] & (~zero_actual)
    out["opportunity_oracle"]=np.where(
        out["model_efficiency_eligible"],
        out["actual_opportunities"]*out["model_effective_efficiency"],
        np.nan,
    )
    out["efficiency_oracle"]=np.where(
        out["efficiency_oracle_eligible"],
        out["predicted_opportunities"]*out["actual_efficiency"],
        np.nan,
    )
    out["full_oracle"]=out["actual_opportunities"]*out["actual_efficiency"]
    full_gap=(out["full_oracle"]-out["actual_output"]).abs()
    if float(full_gap.max())>TOL:
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

    scoreable=rows.loc[rows["model_efficiency_eligible"]].copy()
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
        "actual_opportunity_discrepancy_rows":int(rows["actual_opportunity_discrepancy"].sum()),
        "target_rows_graded_by_pbp":int(rows["opportunity_type"].eq("targets").sum()),
        "max_baseline_reconstruction_gap":float(
            (scoreable["baseline_reconstructed"]-scoreable["baseline_projection"]).abs().max()
        ),
        "max_full_actual_identity_gap":float(
            (rows["full_oracle"]-rows["actual_output"]).abs().max()
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
