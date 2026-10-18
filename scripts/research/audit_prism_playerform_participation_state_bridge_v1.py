#!/usr/bin/env python3
"""PRISM PlayerForm Participation-State Bridge V1 source audit.

Audits whether player-specific participation-anomaly rows are admitted into the
current PlayerForm weekly-history ledger. Source/implementation bridge only:
no future outcome scoring, projection generation, or production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
import scripts.player_form_v2 as pf

VERSION = "PRISM_PLAYERFORM_PARTICIPATION_STATE_BRIDGE_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "3bf001af49ec103d3233b5299d743482f0505daf"
SEASONS = (2024, 2025)
STATES = (
    "PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",
    "PARTICIPATION_COLLAPSE_UNRESOLVED",
)

REQ_ANOM = {
    "season", "week", "team", "stable_id", "family",
    "participation_anomaly_state", "corroboration_source",
    "corroboration_week", "player_prior_snap_pct", "offense_pct",
    "snap_retention_ratio",
}


def pdx(v) -> pd.DataFrame:
    if isinstance(v, pd.DataFrame):
        return v.copy()
    if hasattr(v, "to_dicts"):
        return pd.DataFrame(v.to_dicts())
    if hasattr(v, "to_pandas"):
        return v.to_pandas()
    return pd.DataFrame(v)


def first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def text(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.upper() in {"", "NAN", "NONE", "<NA>"} else s


def load_anomaly(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty anomaly parent: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = REQ_ANOM - set(x.columns)
    if miss:
        raise RuntimeError(f"anomaly parent missing {sorted(miss)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["stable_id"] = x["stable_id"].map(text)
    x["family"] = x["family"].astype("string").str.upper().str.strip()
    x["participation_anomaly_state"] = (
        x["participation_anomaly_state"].astype("string").str.strip()
    )
    x = x.loc[
        x["season"].isin(SEASONS)
        & x["participation_anomaly_state"].isin(STATES)
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    keys = ["season","week","team","stable_id"]
    nonblank = x["stable_id"].ne("")
    if x.loc[nonblank].duplicated(keys).any():
        bad=x.loc[nonblank & x.duplicated(keys,keep=False),keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate anomaly identities: {bad}")
    return x.sort_values(["season","week","team","stable_id"],kind="mergesort").reset_index(drop=True)


def raw_weekly_identity(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    c_week = first_col(x, ("week",))
    c_team = first_col(x, ("recent_team","team","posteam"))
    c_id = first_col(x, ("player_id","gsis_id","player_gsis_id"))
    c_pos = first_col(x, ("position","position_group","pos"))
    if not c_week or not c_team or not c_id:
        raise RuntimeError(
            f"raw weekly provider missing identity semantics week={c_week} team={c_team} id={c_id}"
        )
    q = pd.DataFrame({
        "season": int(season),
        "week": pd.to_numeric(x[c_week], errors="coerce"),
        "team": x[c_team].map(canon_team),
        "stable_id": x[c_id].map(text),
        "raw_position": (
            x[c_pos].astype("string").str.upper().str.strip()
            if c_pos else pd.Series("", index=x.index)
        ),
    })
    q = q.loc[
        q["week"].notna() & q["team"].ne("") & q["stable_id"].ne("")
    ].copy()
    q["week"] = q["week"].astype(int)
    if q.duplicated(["season","week","team","stable_id"]).any():
        bad=q.loc[q.duplicated(["season","week","team","stable_id"],keep=False)].head(20).to_dict("records")
        raise RuntimeError(f"raw weekly provider duplicate identity rows: {bad}")
    return q


def admitted_weekly(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    # Exact current PlayerForm admission semantics.
    q = pf._normalize_weekly(raw, int(season)).copy()
    q["stable_id"] = q["player_id"].map(text)
    q["team"] = q["team"].map(canon_team)
    q["week"] = pd.to_numeric(q["week"], errors="coerce").astype(int)
    q["season"] = int(season)
    keep = ["season","week","team","stable_id","position"]
    q = q[keep].loc[q["stable_id"].ne("")].copy()
    if q.duplicated(["season","week","team","stable_id"]).any():
        bad=q.loc[q.duplicated(["season","week","team","stable_id"],keep=False)].head(20).to_dict("records")
        raise RuntimeError(f"PlayerForm admitted ledger duplicate identity rows: {bad}")
    return q


def code_audit(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")
    forbidden = [
        "participation_anomaly_state",
        "corroboration_week",
        "PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",
        "PARTICIPATION_COLLAPSE_UNRESOLVED",
    ]
    hits = [t for t in forbidden if t in txt]
    return {
        "path": str(path),
        "state_aware_tokens_present": hits,
        "state_aware_history_branch_absent": len(hits) == 0,
        "normalize_weekly_present": "def _normalize_weekly" in txt,
        "season_totals_present": "def _season_totals" in txt,
        "positive_usage_gate_present": "(usage > 0)" in txt,
    }


def build_bridge(anom: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import nflreadpy as nfl

    raw_parts=[]
    admitted_parts=[]
    provider_parts=[]
    provider_schema={}

    for season in SEASONS:
        raw = pdx(nfl.load_player_stats(seasons=[int(season)], summary_level="week"))
        if raw.empty:
            raise RuntimeError(f"weekly provider returned zero rows for {season}")
        raw_parts.append(raw.assign(_requested_season=season))
        provider_parts.append(raw_weekly_identity(raw, season))
        admitted_parts.append(admitted_weekly(raw, season))
        provider_schema[str(season)] = [str(c) for c in raw.columns]

    provider = pd.concat(provider_parts, ignore_index=True)
    admitted = pd.concat(admitted_parts, ignore_index=True)
    if provider.duplicated(["season","week","team","stable_id"]).any():
        raise RuntimeError("provider identity duplicates after concat")
    if admitted.duplicated(["season","week","team","stable_id"]).any():
        raise RuntimeError("admitted identity duplicates after concat")

    keys=["season","week","team","stable_id"]
    pset=set(map(tuple,provider[keys].itertuples(index=False,name=None)))
    aset=set(map(tuple,admitted[keys].itertuples(index=False,name=None)))
    pos_map={
        tuple(r[k] for k in keys): str(r["position"])
        for r in admitted.to_dict("records")
    }

    out=anom.copy()
    out["stable_id_supported"]=out["stable_id"].ne("")
    out["provider_identity_present"]=[
        tuple(v) in pset for v in out[keys].itertuples(index=False,name=None)
    ]
    out["playerform_row_present"]=[
        tuple(v) in aset for v in out[keys].itertuples(index=False,name=None)
    ]
    out["playerform_position"]=[
        pos_map.get(tuple(v),"")
        for v in out[keys].itertuples(index=False,name=None)
    ]

    mismatch = out["provider_identity_present"] & out["stable_id"].eq("")
    if mismatch.any():
        raise RuntimeError("provider join occurred without stable ID")

    return out, {
        "provider_rows": int(len(provider)),
        "admitted_rows": int(len(admitted)),
        "provider_duplicate_identities": int(
            provider.duplicated(keys).sum()
        ),
        "admitted_duplicate_identities": int(
            admitted.duplicated(keys).sum()
        ),
        "provider_schema_columns": provider_schema,
    }


def summarize(out: pd.DataFrame, season: int, code: dict) -> dict:
    s=out.loc[out["season"].eq(season)].copy()
    stable=s["stable_id_supported"]
    stable_cov=float(stable.mean()) if len(s) else 0.0
    supported=s.loc[stable]
    provider_cov=float(supported["provider_identity_present"].mean()) if len(supported) else 0.0

    corr=s.loc[s["participation_anomaly_state"].eq(STATES[0])]
    unr=s.loc[s["participation_anomaly_state"].eq(STATES[1])]

    by_family=[]
    for fam,g in s.groupby("family",sort=True):
        by_family.append({
            "family":str(fam),
            "anomaly_rows":int(len(g)),
            "provider_identity_rows":int(g["provider_identity_present"].sum()),
            "playerform_admitted_rows":int(g["playerform_row_present"].sum()),
            "injury_corroborated_rows":int(g["participation_anomaly_state"].eq(STATES[0]).sum()),
            "injury_corroborated_admitted_rows":int(
                (
                    g["participation_anomaly_state"].eq(STATES[0])
                    & g["playerform_row_present"]
                ).sum()
            ),
            "unresolved_rows":int(g["participation_anomaly_state"].eq(STATES[1]).sum()),
            "unresolved_admitted_rows":int(
                (
                    g["participation_anomaly_state"].eq(STATES[1])
                    & g["playerform_row_present"]
                ).sum()
            ),
        })

    gates={
        "stable_id_coverage_ge_099":bool(stable_cov>=0.99),
        "provider_identity_reconciliation_ge_098":bool(provider_cov>=0.98),
        "injury_corroborated_admitted_rows_nonzero":bool(corr["playerform_row_present"].sum()>0),
        "unresolved_admitted_rows_nonzero":bool(unr["playerform_row_present"].sum()>0),
        "playerform_state_aware_history_branch_absent":bool(code["state_aware_history_branch_absent"]),
        "playerform_positive_usage_gate_present":bool(code["positive_usage_gate_present"]),
    }

    return {
        "season":int(season),
        "anomaly_rows":int(len(s)),
        "stable_id_coverage":stable_cov,
        "provider_identity_reconciliation_coverage":provider_cov,
        "provider_identity_rows":int(s["provider_identity_present"].sum()),
        "playerform_admitted_rows":int(s["playerform_row_present"].sum()),
        "injury_corroborated_rows":int(len(corr)),
        "injury_corroborated_admitted_rows":int(corr["playerform_row_present"].sum()),
        "unresolved_rows":int(len(unr)),
        "unresolved_admitted_rows":int(unr["playerform_row_present"].sum()),
        "family_summary":by_family,
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
    code=code_audit(a.player_form_code)
    if not code["normalize_weekly_present"] or not code["season_totals_present"]:
        raise RuntimeError("PlayerForm code structure drift")

    out,provider_audit=build_bridge(anom)
    summaries=[summarize(out,s,code) for s in SEASONS]
    ready=bool(
        all(s["source_ready"] for s in summaries)
        and provider_audit["provider_duplicate_identities"]==0
        and provider_audit["admitted_duplicate_identities"]==0
    )

    result={
        "version":VERSION,
        "freeze_commit":FREEZE_COMMIT,
        "anomaly_parent_run":39160554692,
        "anomaly_parent_artifact":12147549966,
        "anomaly_parent_digest":"sha256:e7753e981f0ecaaf23289280830fd32b9d733402c02ab9aaa0116fc929450449",
        "player_form_blob_sha":"9246e4daf19f3f01cf61de3ad036b07873fd9561",
        "disposition":(
            "PRISM_PLAYERFORM_PARTICIPATION_STATE_BRIDGE_V1_SOURCE_READY"
            if ready else
            "PRISM_PLAYERFORM_PARTICIPATION_STATE_BRIDGE_V1_SOURCE_BLOCKED"
        ),
        "code_audit":code,
        "provider_audit":provider_audit,
        "season_summaries":summaries,
        "performance_outcomes_scored":0,
        "projections_generated":0,
        "coefficients_fit":0,
        "thresholds_searched":0,
        "outcomes_2026_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "production_changed":False,
    }

    out.to_csv(
        a.out_dir/"playerform_participation_state_bridge_rows.csv",
        index=False,
        float_format="%.12f",
    )
    pd.DataFrame([
        dict(season=s["season"],**f)
        for s in summaries for f in s["family_summary"]
    ]).to_csv(
        a.out_dir/"playerform_participation_state_bridge_family_summary.csv",
        index=False,
    )
    (a.out_dir/"playerform_participation_state_bridge_source_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
