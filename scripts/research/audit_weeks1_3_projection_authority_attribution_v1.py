#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FLAGS = [
    "ml_applied",
    "state_applied",
    "qb_synthesis_applied",
    "rb_synthesis_applied",
    "qb_distribution_specialist_applied",
    "rb_receiving_tail_applied",
    "rb_r26_receptions_applied",
    "rb_rush_rec_conservation_v2_applied",
    "discrete_count_alignment_applied",
]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def flag_state(s: pd.Series) -> pd.Series:
    """Classify stored application flags without conflating missing-era fields."""
    out = pd.Series("UNAVAILABLE", index=s.index, dtype="string")
    present = s.notna()
    numeric = pd.to_numeric(s, errors="coerce")
    applied_numeric = present & numeric.notna() & numeric.ne(0)
    not_applied_numeric = present & numeric.notna() & numeric.eq(0)

    text = s.astype("string").fillna("").str.strip().str.lower()
    applied_text = present & numeric.isna() & text.isin({"true", "yes", "y", "on"})
    not_applied_text = present & numeric.isna() & text.isin({"false", "no", "n", "off", ""})

    out.loc[applied_numeric | applied_text] = "APPLIED"
    out.loc[not_applied_numeric | not_applied_text] = "NOT_APPLIED"

    unknown = present & out.eq("UNAVAILABLE")
    if unknown.any():
        vals = sorted(set(text.loc[unknown].tolist()))
        raise RuntimeError(f"unrecognized application-flag values: {vals}")
    return out


def metrics(x: pd.DataFrame, col: str) -> dict:
    q=x.loc[num(x[col]).notna() & num(x["actual"]).notna()].copy()
    if q.empty:
        return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan}
    e=num(q[col])-num(q["actual"])
    return {
        "n":int(len(q)),
        "mae":float(e.abs().mean()),
        "rmse":float(np.sqrt(np.mean(np.square(e)))),
        "bias":float(e.mean()),
    }


def paired(x: pd.DataFrame, a: str, b: str) -> dict:
    q=x.loc[num(x[a]).notna() & num(x[b]).notna() & num(x["actual"]).notna()].copy()
    if q.empty:
        return {"n":0,"a_mae":np.nan,"b_mae":np.nan,"paired_abs_improvement":np.nan,
                "a_bias":np.nan,"b_bias":np.nan,"toward_rate":np.nan}
    actual=num(q["actual"])
    ea=(num(q[a])-actual)
    eb=(num(q[b])-actual)
    imp=ea.abs()-eb.abs()
    move=(num(q[b])-num(q[a]))
    toward=(move*(actual-num(q[a]))>0)
    away=(move*(actual-num(q[a]))<0)
    moved=move.abs()>1e-12
    toward_rate=float(toward[moved].mean()) if moved.any() else np.nan
    return {
        "n":int(len(q)),
        "a_mae":float(ea.abs().mean()),
        "b_mae":float(eb.abs().mean()),
        "paired_abs_improvement":float(imp.mean()),
        "a_bias":float(ea.mean()),
        "b_bias":float(eb.mean()),
        "toward_rate":toward_rate,
        "moved_rows":int(moved.sum()),
        "away_rows":int(away.sum()),
    }


def cluster_bootstrap_mean_improvement(
    x: pd.DataFrame, a: str, b: str, *, reps: int=10000, seed: int=42027
) -> dict:
    q=x.loc[num(x[a]).notna() & num(x[b]).notna() & num(x["actual"]).notna()].copy()
    if q.empty:
        return {"clusters":0,"mean":np.nan,"lo":np.nan,"hi":np.nan}
    cluster_col="event_id" if "event_id" in q.columns else None
    if cluster_col is None:
        q["_cluster"]=np.arange(len(q)).astype(str)
        cluster_col="_cluster"
    q["_imp"]=(num(q[a])-num(q["actual"])).abs()-(num(q[b])-num(q["actual"])).abs()
    groups={str(k):v["_imp"].to_numpy(float) for k,v in q.groupby(cluster_col, dropna=False)}
    keys=list(groups)
    rng=np.random.default_rng(seed)
    vals=np.empty(reps,float)
    for i in range(reps):
        sampled=rng.choice(keys,size=len(keys),replace=True)
        arr=np.concatenate([groups[k] for k in sampled])
        vals[i]=float(np.mean(arr))
    return {
        "clusters":int(len(keys)),
        "mean":float(q["_imp"].mean()),
        "lo":float(np.quantile(vals,.025)),
        "hi":float(np.quantile(vals,.975)),
    }


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--graded",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    df=pd.read_csv(a.graded)
    if "settlement_status" not in df.columns:
        raise RuntimeError("graded input missing settlement_status")
    x=df.loc[df["settlement_status"].astype(str).eq("SETTLED") & num(df["actual"]).notna()].copy()
    if x.empty:
        raise RuntimeError("no settled finite-actual rows")

    required={"mc_proj","ensemble_proj","model_proj","actual","week","position","market"}
    missing=sorted(required-set(x.columns))
    if missing:
        raise RuntimeError(f"graded input missing required columns: {missing}")

    rows=[]
    for group_name,group_cols in [
        ("ALL",[]),
        ("WEEK",["week"]),
        ("POSITION",["position"]),
        ("MARKET",["market"]),
        ("POSITION_MARKET",["position","market"]),
    ]:
        groups=[(("ALL",),x)] if not group_cols else x.groupby(group_cols,dropna=False)
        for key,g in groups:
            if not isinstance(key,tuple):
                key=(key,)
            label=" | ".join(map(str,key))
            for left,right in [("mc_proj","model_proj"),("ensemble_proj","model_proj")]:
                m=paired(g,left,right)
                rows.append({
                    "group_type":group_name,
                    "group_value":label,
                    "comparison":f"{left}_to_{right}",
                    **m,
                })
    paired_df=pd.DataFrame(rows)

    flag_rows=[]
    for flag in FLAGS:
        if flag not in x.columns:
            continue
        states = flag_state(x[flag])
        for state in ("APPLIED", "NOT_APPLIED", "UNAVAILABLE"):
            q = x.loc[states.eq(state)]
            if q.empty:
                continue
            m=metrics(q,"model_proj")
            flag_rows.append({"flag":flag,"state":state,**m})
    flag_df=pd.DataFrame(flag_rows)

    movement=x.loc[num(x["mc_proj"]).notna() & num(x["model_proj"]).notna()].copy()
    movement["mc_error"]=num(movement["mc_proj"])-num(movement["actual"])
    movement["final_error"]=num(movement["model_proj"])-num(movement["actual"])
    movement["final_minus_mc"]=num(movement["model_proj"])-num(movement["mc_proj"])
    movement["paired_abs_improvement"]=movement["mc_error"].abs()-movement["final_error"].abs()
    movement["movement_toward_actual"]=(
        movement["final_minus_mc"]*(num(movement["actual"])-num(movement["mc_proj"]))>0
    )
    movement["movement_away_from_actual"]=(
        movement["final_minus_mc"]*(num(movement["actual"])-num(movement["mc_proj"]))<0
    )

    overall=paired(x,"mc_proj","model_proj")
    boot=cluster_bootstrap_mean_improvement(x,"mc_proj","model_proj")

    a.out_dir.mkdir(parents=True,exist_ok=True)
    paired_df.to_csv(a.out_dir/"projection_authority_paired_summary.csv",index=False)
    flag_df.to_csv(a.out_dir/"projection_authority_flag_summary.csv",index=False)
    movement.to_csv(a.out_dir/"projection_authority_row_detail.csv",index=False)

    lines=[]
    lines.append("=== WEEKS 1-3 PROJECTION AUTHORITY ATTRIBUTION V1 ===")
    lines.append(f"settled_rows={len(x)}")
    lines.append("")
    lines.append("OVERALL MC -> FINAL")
    lines.append(str(overall))
    lines.append(
        f"cluster_bootstrap paired_abs_improvement mean={boot['mean']:.6f} "
        f"95%CI=[{boot['lo']:.6f},{boot['hi']:.6f}] clusters={boot['clusters']}"
    )
    lines.append("")
    lines.append("BY MARKET — MC -> FINAL")
    z=paired_df.loc[
        paired_df["group_type"].eq("MARKET")
        & paired_df["comparison"].eq("mc_proj_to_model_proj")
    ]
    lines.append(z.to_string(index=False))
    lines.append("")
    lines.append("BY POSITION — MC -> FINAL")
    z=paired_df.loc[
        paired_df["group_type"].eq("POSITION")
        & paired_df["comparison"].eq("mc_proj_to_model_proj")
    ]
    lines.append(z.to_string(index=False))
    lines.append("")
    lines.append("APPLICATION FLAG COHORTS — FINAL MODEL ERROR")
    lines.append(flag_df.to_string(index=False))
    report="\n".join(lines)+"\n"
    (a.out_dir/"projection_authority_attribution_report.txt").write_text(report,encoding="utf-8")
    print(report)

    mean=float(overall["paired_abs_improvement"])
    lo=float(boot["lo"])
    hi=float(boot["hi"])
    if np.isfinite(lo) and lo>0:
        disposition="FINAL_IMPROVES_BASE_MC"
    elif np.isfinite(hi) and hi<0:
        disposition="BASE_MC_DOMINATES_FINAL"
    elif np.isfinite(mean) and abs(mean)<0.1:
        disposition="NO_MATERIAL_PROJECTION_AUTHORITY_DIFFERENCE"
    else:
        disposition="MIXED_BY_MARKET_OR_POSITION"
    print(f"DISPOSITION={disposition}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
