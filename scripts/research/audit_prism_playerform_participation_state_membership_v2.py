#!/usr/bin/env python3
"""PRISM PlayerForm Participation-State Membership V2 source audit.

Deterministic membership audit between the certified stable-ID anomaly ledger
and the exact PlayerForm-admitted weekly history ledger. No future outcome
scoring, projection generation, or production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
import scripts.player_form_v2 as pf

VERSION = "PRISM_PLAYERFORM_PARTICIPATION_STATE_MEMBERSHIP_V2_SOURCE_AUDIT"
FREEZE_COMMIT = "a76f9508089d0b158752adf635dd7b4764150221"
SEASONS = (2024, 2025)
STATES = (
    "PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",
    "PARTICIPATION_COLLAPSE_UNRESOLVED",
)

REQ = {
    "season", "week", "team", "stable_id", "family",
    "participation_anomaly_state", "corroboration_source",
    "corroboration_week", "player_prior_snap_pct", "offense_pct",
    "snap_retention_ratio",
}


def pdx(v):
    if isinstance(v,pd.DataFrame):
        return v.copy()
    if hasattr(v,"to_dicts"):
        return pd.DataFrame(v.to_dicts())
    if hasattr(v,"to_pandas"):
        return v.to_pandas()
    return pd.DataFrame(v)


def text(v)->str:
    if v is None or pd.isna(v):
        return ""
    s=str(v).strip()
    return "" if s.upper() in {"","NAN","NONE","<NA>"} else s


def load_anomaly(path:Path)->pd.DataFrame:
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    miss=REQ-set(x.columns)
    if miss:
        raise RuntimeError(f"anomaly parent missing {sorted(miss)}")
    x["season"]=pd.to_numeric(x["season"],errors="coerce")
    x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x["team"]=x["team"].map(canon_team)
    x["stable_id"]=x["stable_id"].map(text)
    x["family"]=x["family"].astype("string").str.upper().str.strip()
    x["participation_anomaly_state"]=x["participation_anomaly_state"].astype("string").str.strip()
    x=x.loc[
        x["season"].isin(SEASONS)
        & x["participation_anomaly_state"].isin(STATES)
    ].copy()
    x["season"]=x["season"].astype(int)
    x["week"]=x["week"].astype(int)
    nonblank=x["stable_id"].ne("")
    keys=["season","week","team","stable_id"]
    if x.loc[nonblank].duplicated(keys).any():
        bad=x.loc[nonblank & x.duplicated(keys,keep=False),keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate anomaly membership keys: {bad}")
    return x.sort_values(keys,kind="mergesort").reset_index(drop=True)


def admitted_ledger()->pd.DataFrame:
    import nflreadpy as nfl

    parts=[]
    for season in SEASONS:
        raw=pdx(nfl.load_player_stats(seasons=[int(season)],summary_level="week"))
        if raw.empty:
            raise RuntimeError(f"weekly player stats empty for {season}")
        q=pf._normalize_weekly(raw,int(season)).copy()
        q["season"]=int(season)
        q["week"]=pd.to_numeric(q["week"],errors="coerce").astype(int)
        q["team"]=q["team"].map(canon_team)
        q["stable_id"]=q["player_id"].map(text)
        q["playerform_position"]=q["position"].astype("string").fillna("").str.upper().str.strip()
        q=q[["season","week","team","stable_id","playerform_position"]]
        q=q.loc[q["stable_id"].ne("")].copy()
        parts.append(q)
    out=pd.concat(parts,ignore_index=True)
    keys=["season","week","team","stable_id"]
    if out.duplicated(keys).any():
        bad=out.loc[out.duplicated(keys,keep=False),keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate PlayerForm admitted membership keys: {bad}")
    return out


def code_audit(path:Path)->dict:
    txt=path.read_text(encoding="utf-8")
    forbidden=[
        "participation_anomaly_state",
        "corroboration_week",
        "PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",
        "PARTICIPATION_COLLAPSE_UNRESOLVED",
    ]
    hits=[t for t in forbidden if t in txt]
    return {
        "state_aware_tokens_present":hits,
        "state_aware_history_branch_absent":len(hits)==0,
        "normalize_weekly_present":"def _normalize_weekly" in txt,
        "season_totals_present":"def _season_totals" in txt,
        "positive_usage_gate_present":"(usage > 0)" in txt,
    }


def build(anom:pd.DataFrame,admitted:pd.DataFrame)->pd.DataFrame:
    keys=["season","week","team","stable_id"]
    aset=set(map(tuple,admitted[keys].itertuples(index=False,name=None)))
    pos={
        tuple(r[k] for k in keys):str(r["playerform_position"])
        for r in admitted.to_dict("records")
    }
    out=anom.copy()
    out["stable_id_supported"]=out["stable_id"].ne("")
    out["playerform_row_present"]=[
        tuple(v) in aset for v in out[keys].itertuples(index=False,name=None)
    ]
    out["playerform_position"]=[
        pos.get(tuple(v),"") for v in out[keys].itertuples(index=False,name=None)
    ]
    return out


def summarize(out:pd.DataFrame,season:int,code:dict)->dict:
    s=out.loc[out["season"].eq(season)].copy()
    corr=s.loc[s["participation_anomaly_state"].eq(STATES[0])]
    unr=s.loc[s["participation_anomaly_state"].eq(STATES[1])]
    stable_cov=float(s["stable_id_supported"].mean()) if len(s) else 0.0

    fam=[]
    for family,g in s.groupby("family",sort=True):
        fam.append({
            "family":str(family),
            "anomaly_rows":int(len(g)),
            "injury_corroborated_rows":int(g["participation_anomaly_state"].eq(STATES[0]).sum()),
            "injury_corroborated_admitted_rows":int((
                g["participation_anomaly_state"].eq(STATES[0])
                & g["playerform_row_present"]
            ).sum()),
            "unresolved_rows":int(g["participation_anomaly_state"].eq(STATES[1]).sum()),
            "unresolved_admitted_rows":int((
                g["participation_anomaly_state"].eq(STATES[1])
                & g["playerform_row_present"]
            ).sum()),
        })

    gates={
        "stable_id_coverage_ge_099":bool(stable_cov>=0.99),
        "duplicate_anomaly_membership_keys_zero":bool(
            not s.loc[s["stable_id_supported"]].duplicated(["season","week","team","stable_id"]).any()
        ),
        "injury_corroborated_rows_nonzero":bool(len(corr)>0),
        "injury_corroborated_admitted_rows_nonzero":bool(corr["playerform_row_present"].sum()>0),
        "unresolved_rows_nonzero":bool(len(unr)>0),
        "unresolved_admitted_rows_nonzero":bool(unr["playerform_row_present"].sum()>0),
        "playerform_state_aware_history_branch_absent":bool(code["state_aware_history_branch_absent"]),
        "playerform_positive_usage_gate_present":bool(code["positive_usage_gate_present"]),
    }
    return {
        "season":int(season),
        "anomaly_rows":int(len(s)),
        "stable_id_coverage":stable_cov,
        "playerform_admitted_rows":int(s["playerform_row_present"].sum()),
        "injury_corroborated_rows":int(len(corr)),
        "injury_corroborated_admitted_rows":int(corr["playerform_row_present"].sum()),
        "unresolved_rows":int(len(unr)),
        "unresolved_admitted_rows":int(unr["playerform_row_present"].sum()),
        "family_summary":fam,
        "gates":gates,
        "source_ready":bool(all(gates.values())),
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--anomaly-parent",type=Path,required=True)
    ap.add_argument("--player-form-code",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)

    anom=load_anomaly(a.anomaly_parent)
    admitted=admitted_ledger()
    code=code_audit(a.player_form_code)
    if not code["normalize_weekly_present"] or not code["season_totals_present"]:
        raise RuntimeError("PlayerForm structure drift")

    out=build(anom,admitted)
    summaries=[summarize(out,s,code) for s in SEASONS]
    ready=bool(
        all(s["source_ready"] for s in summaries)
        and not admitted.duplicated(["season","week","team","stable_id"]).any()
    )

    result={
        "version":VERSION,
        "freeze_commit":FREEZE_COMMIT,
        "anomaly_parent_run":39160554692,
        "anomaly_parent_artifact":12147549966,
        "anomaly_parent_digest":"sha256:e7753e981f0ecaaf23289280830fd32b9d733402c02ab9aaa0116fc929450449",
        "player_form_blob_sha":"9246e4daf19f3f01cf61de3ad036b07873fd9561",
        "disposition":(
            "PRISM_PLAYERFORM_PARTICIPATION_STATE_MEMBERSHIP_V2_SOURCE_READY"
            if ready else
            "PRISM_PLAYERFORM_PARTICIPATION_STATE_MEMBERSHIP_V2_SOURCE_BLOCKED"
        ),
        "admitted_ledger_rows":int(len(admitted)),
        "admitted_ledger_duplicate_keys":int(admitted.duplicated(["season","week","team","stable_id"]).sum()),
        "code_audit":code,
        "season_summaries":summaries,
        "future_player_outcomes_scored":0,
        "projections_generated":0,
        "coefficients_fit":0,
        "thresholds_searched":0,
        "outcomes_2026_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "production_changed":False,
    }

    out.to_csv(a.out_dir/"playerform_participation_state_membership_v2_rows.csv",index=False,float_format="%.12f")
    pd.DataFrame([
        dict(season=s["season"],**f)
        for s in summaries for f in s["family_summary"]
    ]).to_csv(a.out_dir/"playerform_participation_state_membership_v2_family_summary.csv",index=False)
    (a.out_dir/"playerform_participation_state_membership_v2_source_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
