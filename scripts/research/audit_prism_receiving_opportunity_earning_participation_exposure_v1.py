#!/usr/bin/env python3
"""PRISM receiving opportunity-earning / participation exposure V1 source audit.

Pairs exact named-player weekly target opportunity with source-certified
offensive participation exposure. Source/descriptive state only; no predictive
scoring, no route inference, no sportsbook, no production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
import scripts.player_form_v2 as pf

VERSION = "PRISM_RECEIVING_OPPORTUNITY_EARNING_PARTICIPATION_EXPOSURE_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "42ff90db6202b783f7e97b73d0b3d629e9566bb3"
SEASONS = (2024, 2025)
FAMILIES = {"RB", "WR", "TE"}

HEALTH_REQ = {
    "season","week","team","stable_id","family","positive_participation",
    "offense_snaps","offense_pct","injury_reported","observation_state",
}
ANOM_REQ = {
    "season","week","team","stable_id","participation_anomaly_state",
    "player_prior_snap_pct","snap_retention_ratio","corroboration_source",
    "corroboration_week",
}


def pdx(v):
    if isinstance(v, pd.DataFrame):
        return v.copy()
    if hasattr(v, "to_dicts"):
        return pd.DataFrame(v.to_dicts())
    if hasattr(v, "to_pandas"):
        return v.to_pandas()
    return pd.DataFrame(v)


def text(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.upper() in {"", "NAN", "NONE", "<NA>"} else s


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype("string").str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False).astype(bool)
    )


def load_health(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = HEALTH_REQ - set(x.columns)
    if miss:
        raise RuntimeError(f"health parent missing {sorted(miss)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["stable_id"] = x["stable_id"].map(text)
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["positive_participation"] = as_bool(x["positive_participation"])
    x["offense_snaps"] = pd.to_numeric(x["offense_snaps"], errors="coerce")
    x["offense_pct"] = pd.to_numeric(x["offense_pct"], errors="coerce")
    x["injury_reported"] = as_bool(x["injury_reported"])
    x = x.loc[
        x["season"].isin(SEASONS)
        & x["family"].isin(FAMILIES)
        & x["positive_participation"]
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    keys = ["season","week","team","stable_id"]
    nonblank = x["stable_id"].ne("")
    if x.loc[nonblank].duplicated(keys).any():
        raise RuntimeError("duplicate stable-ID health rows")
    return x.sort_values(["season","week","team","stable_id"],kind="mergesort").reset_index(drop=True)


def load_anomaly(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = ANOM_REQ - set(x.columns)
    if miss:
        raise RuntimeError(f"anomaly parent missing {sorted(miss)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["stable_id"] = x["stable_id"].map(text)
    x["corroboration_week"] = pd.to_numeric(x["corroboration_week"], errors="coerce")
    x = x.loc[x["season"].isin(SEASONS)].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    keys = ["season","week","team","stable_id"]
    nonblank=x["stable_id"].ne("")
    if x.loc[nonblank].duplicated(keys).any():
        raise RuntimeError("duplicate stable-ID anomaly rows")
    keep = [
        *keys, "participation_anomaly_state", "player_prior_snap_pct",
        "snap_retention_ratio", "corroboration_source", "corroboration_week",
    ]
    return x[keep]


def weekly_opportunity_ledgers() -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    import nflreadpy as nfl

    player_parts=[]
    team_parts=[]
    audit={}
    for season in SEASONS:
        raw=pdx(nfl.load_player_stats(seasons=[int(season)],summary_level="week"))
        if raw.empty:
            raise RuntimeError(f"weekly player stats empty for {season}")
        q=pf._normalize_weekly(raw,int(season)).copy()
        q["season"]=int(season)
        q["week"]=pd.to_numeric(q["week"],errors="coerce").astype(int)
        q["team"]=q["team"].map(canon_team)
        q["stable_id"]=q["player_id"].map(text)
        q["family"]=q["position"].astype("string").fillna("").str.upper().str.strip()
        q["targets"]=pd.to_numeric(q["targets"],errors="coerce")
        q["team_targets"]=pd.to_numeric(q["team_targets"],errors="coerce")

        # Exact admitted PlayerForm rows with stable identity.
        p=q.loc[q["stable_id"].ne(""),[
            "season","week","team","stable_id","family","targets","team_targets"
        ]].copy()
        if p.duplicated(["season","week","team","stable_id"]).any():
            raise RuntimeError(f"duplicate PlayerForm weekly identity {season}")
        player_parts.append(p)

        # Repeated PlayerForm team_targets must resolve to one exact value.
        t=[]
        for key,g in q.groupby(["season","week","team"],sort=False):
            vals=pd.to_numeric(g["team_targets"],errors="coerce").dropna().unique()
            if len(vals)!=1:
                raise RuntimeError(f"team target total ambiguity {key}: {vals[:10]}")
            t.append({
                "season":int(key[0]),"week":int(key[1]),"team":str(key[2]),
                "team_targets":float(vals[0]),
            })
        team=pd.DataFrame(t)
        if team.duplicated(["season","week","team"]).any():
            raise RuntimeError(f"duplicate team target totals {season}")
        team_parts.append(team)
        audit[str(season)]={
            "raw_weekly_rows":int(len(raw)),
            "playerform_admitted_rows":int(len(q)),
            "stable_id_admitted_rows":int(len(p)),
            "team_week_rows":int(len(team)),
        }

    return (
        pd.concat(player_parts,ignore_index=True),
        pd.concat(team_parts,ignore_index=True),
        audit,
    )


def build_panel(
    health:pd.DataFrame,
    anomaly:pd.DataFrame,
    player_ledger:pd.DataFrame,
    team_ledger:pd.DataFrame,
)->pd.DataFrame:
    keys=["season","week","team","stable_id"]
    stable=health.loc[health["stable_id"].ne("")].copy()

    player_map={
        tuple(r[k] for k in keys): {
            "targets":float(r["targets"]),
            "playerform_family":str(r["family"]),
        }
        for r in player_ledger.to_dict("records")
    }
    team_map={
        (int(r.season),int(r.week),str(r.team)):float(r.team_targets)
        for r in team_ledger.itertuples(index=False)
    }

    rows=[]
    for r in stable.to_dict("records"):
        key=(int(r["season"]),int(r["week"]),str(r["team"]),str(r["stable_id"]))
        p=player_map.get(key)
        tt=team_map.get((key[0],key[1],key[2]))
        targets=0.0 if p is None else float(p["targets"])
        snaps=float(r["offense_snaps"]) if pd.notna(r["offense_snaps"]) else np.nan
        pct=float(r["offense_pct"]) if pd.notna(r["offense_pct"]) else np.nan
        team_targets=float(tt) if tt is not None else np.nan

        rec=dict(r)
        rec.update({
            "playerform_weekly_row_present":bool(p is not None),
            "playerform_family":"" if p is None else str(p["playerform_family"]),
            "targets":targets,
            "team_targets":team_targets,
            "target_share_game":(
                targets/team_targets
                if np.isfinite(team_targets) and team_targets>0 else np.nan
            ),
            "targets_per_offensive_snap":(
                targets/snaps if np.isfinite(snaps) and snaps>0 else np.nan
            ),
            "team_targets_times_snap_share":(
                team_targets*pct
                if np.isfinite(team_targets) and np.isfinite(pct) else np.nan
            ),
            "targets_per_snap_scaled_team_target":(
                targets/(team_targets*pct)
                if np.isfinite(team_targets) and team_targets>0
                and np.isfinite(pct) and pct>0 else np.nan
            ),
        })
        rows.append(rec)

    out=pd.DataFrame(rows)
    a=anomaly.copy()
    out=out.merge(
        a,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("","_anomaly"),
    )
    out["participation_anomaly_state"]=out["participation_anomaly_state"].fillna("PARTICIPATION_STATE_UNAVAILABLE")
    return out


def summarize(panel:pd.DataFrame,season:int)->dict:
    s=panel.loc[panel["season"].eq(season)].copy()
    finite_team=s["team_targets"].notna() & s["team_targets"].gt(0)
    finite_snap=s["offense_snaps"].notna() & s["offense_snaps"].gt(0)
    finite_pct=s["offense_pct"].notna() & s["offense_pct"].gt(0)
    target_finite=s["targets"].notna()
    target_share_ok=s.loc[finite_team,"target_share_game"].notna().all()
    per_snap_ok=s.loc[finite_snap,"targets_per_offensive_snap"].notna().all()

    corr=s["participation_anomaly_state"].eq("PARTICIPATION_COLLAPSE_INJURY_CORROBORATED")
    unr=s["participation_anomaly_state"].eq("PARTICIPATION_COLLAPSE_UNRESOLVED")

    families=[]
    for fam,g in s.groupby("family",sort=True):
        families.append({
            "family":str(fam),
            "player_games":int(len(g)),
            "players":int(g["stable_id"].nunique()),
            "team_target_coverage":float((g["team_targets"].notna() & g["team_targets"].gt(0)).mean()) if len(g) else 0.0,
            "snap_count_coverage":float((g["offense_snaps"].notna() & g["offense_snaps"].gt(0)).mean()) if len(g) else 0.0,
            "snap_pct_coverage":float((g["offense_pct"].notna() & g["offense_pct"].gt(0)).mean()) if len(g) else 0.0,
            "zero_target_rows":int(g["targets"].eq(0).sum()),
            "injury_corroborated_collapse_rows":int(g["participation_anomaly_state"].eq("PARTICIPATION_COLLAPSE_INJURY_CORROBORATED").sum()),
            "unresolved_collapse_rows":int(g["participation_anomaly_state"].eq("PARTICIPATION_COLLAPSE_UNRESOLVED").sum()),
            "median_offense_pct":float(g["offense_pct"].median()),
            "median_targets_per_offensive_snap":float(g["targets_per_offensive_snap"].median()),
            "median_target_share":float(g["target_share_game"].median()),
        })

    gates={
        "stable_id_coverage_ge_099":True,
        "duplicate_positive_participation_keys_zero":bool(
            not s.duplicated(["season","week","team","stable_id"]).any()
        ),
        "team_target_coverage_ge_099":bool(finite_team.mean()>=0.99),
        "snap_count_coverage_ge_099":bool(finite_snap.mean()>=0.99),
        "snap_pct_coverage_ge_099":bool(finite_pct.mean()>=0.99),
        "target_count_finite_all":bool(target_finite.all()),
        "target_share_finite_when_team_targets_positive":bool(target_share_ok),
        "targets_per_snap_finite_when_snaps_positive":bool(per_snap_ok),
        "injury_corroborated_collapse_rows_nonzero":bool(corr.sum()>0),
        "unresolved_collapse_rows_nonzero":bool(unr.sum()>0),
    }
    return {
        "season":int(season),
        "player_games":int(len(s)),
        "players":int(s["stable_id"].nunique()),
        "team_target_coverage":float(finite_team.mean()) if len(s) else 0.0,
        "snap_count_coverage":float(finite_snap.mean()) if len(s) else 0.0,
        "snap_pct_coverage":float(finite_pct.mean()) if len(s) else 0.0,
        "zero_target_positive_participation_rows":int(s["targets"].eq(0).sum()),
        "injury_corroborated_collapse_rows":int(corr.sum()),
        "unresolved_collapse_rows":int(unr.sum()),
        "family_summary":families,
        "gates":gates,
        "source_ready":bool(all(gates.values())),
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--health-parent",type=Path,required=True)
    ap.add_argument("--anomaly-parent",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)

    health=load_health(a.health_parent)
    stable_cov=float(health["stable_id"].ne("").mean()) if len(health) else 0.0
    anomaly=load_anomaly(a.anomaly_parent)
    players,teams,provider_audit=weekly_opportunity_ledgers()
    panel=build_panel(health,anomaly,players,teams)

    summaries=[summarize(panel,s) for s in SEASONS]
    for s in summaries:
        s["gates"]["stable_id_coverage_ge_099"]=bool(stable_cov>=0.99)
        s["source_ready"]=bool(all(s["gates"].values()))

    ready=bool(all(s["source_ready"] for s in summaries))

    result={
        "version":VERSION,
        "freeze_commit":FREEZE_COMMIT,
        "health_parent_run":39154145666,
        "health_parent_artifact":12146625382,
        "health_parent_digest":"sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb",
        "anomaly_parent_run":39160554692,
        "anomaly_parent_artifact":12147549966,
        "anomaly_parent_digest":"sha256:e7753e981f0ecaaf23289280830fd32b9d733402c02ab9aaa0116fc929450449",
        "player_form_blob_sha":"9246e4daf19f3f01cf61de3ad036b07873fd9561",
        "disposition":(
            "PRISM_RECEIVING_OPPORTUNITY_EARNING_PARTICIPATION_EXPOSURE_V1_SOURCE_READY"
            if ready else
            "PRISM_RECEIVING_OPPORTUNITY_EARNING_PARTICIPATION_EXPOSURE_V1_SOURCE_BLOCKED"
        ),
        "stable_id_coverage":stable_cov,
        "provider_audit":provider_audit,
        "season_summaries":summaries,
        "routes_inferred_from_snaps":False,
        "predictive_outcomes_scored":0,
        "projections_generated":0,
        "coefficients_fit":0,
        "thresholds_searched":0,
        "outcomes_2026_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "production_changed":False,
    }

    panel.to_csv(a.out_dir/"receiving_opportunity_earning_participation_exposure_rows.csv",index=False,float_format="%.12f")
    pd.DataFrame([
        dict(season=s["season"],**f)
        for s in summaries for f in s["family_summary"]
    ]).to_csv(a.out_dir/"receiving_opportunity_earning_participation_exposure_family_summary.csv",index=False)
    (a.out_dir/"receiving_opportunity_earning_participation_exposure_source_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
