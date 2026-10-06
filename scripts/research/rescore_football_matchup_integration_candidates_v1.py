#!/usr/bin/env python3
"""Artifact-anchored repair score for Football Matchup Transmission candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.research.score_football_matchup_integration_candidates_v1 as core

VERSION = "FOOTBALL_MATCHUP_TRANSMISSION_INTEGRATION_CANDIDATES_PARITY_REPAIR_V1"
KEYS = ["season","week","game_id","team","opponent","player_clean_key","market"]


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _secondary_candidate_mask(x: pd.DataFrame, cand: core.Candidate) -> pd.Series:
    if cand.position == "RB":
        pos = x["position"].isin(core.RB_POS)
    else:
        pos = x["position"].eq(cand.position)
    return (
        x["season"].isin(core.SECONDARY_SEASONS)
        & x["market"].eq(cand.market)
        & pos
    )


def independent_rebuild_drift(current: pd.DataFrame, parent: pd.DataFrame) -> dict:
    per = {}
    any_drift = False
    overall = 0.0
    for cand in core.CANDIDATES:
        a = current.loc[_secondary_candidate_mask(current, cand), KEYS + ["baseline_projection"]].copy()
        b = parent.loc[_secondary_candidate_mask(parent, cand), KEYS + ["baseline_projection"]].copy()
        a = a.sort_values(KEYS).reset_index(drop=True)
        b = b.sort_values(KEYS).reset_index(drop=True)
        identity_match = len(a) == len(b) and a[KEYS].astype(str).equals(b[KEYS].astype(str))
        if not identity_match:
            raise RuntimeError(
                f"secondary frozen authority identity mismatch {cand.candidate_id}: "
                f"{len(a)} rebuilt vs {len(b)} parent"
            )
        av = pd.to_numeric(a["baseline_projection"], errors="coerce").to_numpy(float)
        bv = pd.to_numeric(b["baseline_projection"], errors="coerce").to_numpy(float)
        if not np.array_equal(np.isnan(av), np.isnan(bv)):
            raise RuntimeError(f"secondary baseline missingness mismatch {cand.candidate_id}")
        m = np.isfinite(av) & np.isfinite(bv)
        gap = float(np.max(np.abs(av[m] - bv[m]))) if m.any() else 0.0
        any_drift = any_drift or gap > core.TOL
        overall = max(overall, gap)
        per[cand.candidate_id] = {
            "rows": int(len(a)),
            "identity_match": True,
            "max_abs_gap": gap,
        }
    return {
        "independent_rebuild_drift_detected": bool(any_drift),
        "max_abs_gap": overall,
        "per_candidate": per,
    }


def build_hybrid_projection(
    current: pd.DataFrame,
    parent: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    drift = independent_rebuild_drift(current, parent)
    train_primary = current.loc[current["season"].isin([core.TRAIN_SEASON, core.PRIMARY_SEASON])].copy()
    secondary = parent.loc[parent["season"].isin(core.SECONDARY_SEASONS)].copy()

    keep = sorted(set(train_primary.columns).intersection(secondary.columns))
    required = set(KEYS + [
        "actual","baseline_projection","position","player_identity_key"
    ])
    if not required.issubset(set(keep)):
        raise RuntimeError(f"hybrid projection missing common columns: {sorted(required-set(keep))}")

    train_primary = train_primary[keep].copy()
    secondary = secondary[keep].copy()
    train_primary["baseline_source"] = "REBUILT_2022_2023_CANDIDATE_PREP"
    secondary["baseline_source"] = "FROZEN_RIGHT_TAIL_PARENT"

    out = pd.concat([train_primary, secondary], ignore_index=True)
    if out.duplicated(KEYS).any():
        raise RuntimeError("hybrid projection contains duplicate identities")
    return out, drift


def build_hybrid_team_features(
    rebuilt_team: pd.DataFrame,
    schedule: pd.DataFrame,
    frozen_phase_bc: pd.DataFrame,
) -> pd.DataFrame:
    rebuilt = core.build_team_features(rebuilt_team, schedule)
    rebuilt = rebuilt.loc[rebuilt["season"].isin([2022, 2023])].copy()

    frozen = frozen_phase_bc.copy()
    for c in ("season","week"):
        frozen[c] = pd.to_numeric(frozen[c], errors="coerce").astype("Int64")
    frozen["team"] = frozen["team"].map(core.canon_team)
    frozen["opponent"] = frozen["opponent"].map(core.canon_team)
    frozen = frozen.loc[
        frozen["season"].isin(core.SECONDARY_SEASONS)
        & frozen["week"].isin(core.TARGET_WEEKS)
    ].copy()

    candidate_features = sorted({c.feature for c in core.CANDIDATES})
    cols = ["season","week","team","opponent"]
    for f in candidate_features:
        cols.extend([f, f"{f}__z"])
    missing = sorted(set(cols)-set(frozen.columns))
    if missing:
        raise RuntimeError(f"frozen Phase B/C feature authority missing: {missing}")
    frozen = frozen[cols].copy()

    rebuilt = rebuilt[cols].copy()
    rebuilt["feature_source"] = "REBUILT_2022_2023_M89_M90_SEMANTICS"
    frozen["feature_source"] = "FROZEN_PHASE_BC_AUTHORITY"
    out = pd.concat([rebuilt, frozen], ignore_index=True)
    if out.duplicated(["season","week","team"]).any():
        raise RuntimeError("hybrid team features contain duplicate team-week")
    return out


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-trace", type=Path, required=True)
    ap.add_argument("--parent-right-tail-detail", type=Path, required=True)
    ap.add_argument("--frozen-phase-bc-team-features", type=Path, required=True)
    ap.add_argument("--team-weekly-matchup", type=Path, required=True)
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a=ap.parse_args()

    projection=_read(a.projection_trace,"preserved candidate projection trace")
    parent_raw=_read(a.parent_right_tail_detail,"frozen right-tail detail")
    phase_bc=_read(a.frozen_phase_bc_team_features,"frozen Phase B/C team features")
    team=_read(a.team_weekly_matchup,"preserved matchup team history")
    logs=_read(a.player_logs,"preserved player logs")
    schedule=_read(a.schedule,"preserved schedule")

    for label, frame in (
        ("projection",projection),("right_tail",parent_raw),("phase_bc",phase_bc),
        ("team",team),("logs",logs),("schedule",schedule)
    ):
        core._validate_no_forbidden(label,frame)

    current=core.attach_position(projection,logs,projection_col="ensemble_proj")
    parent=core.load_parent_detail(a.parent_right_tail_detail,logs)

    hybrid, drift=build_hybrid_projection(current,parent)
    team_features=build_hybrid_team_features(team,schedule,phase_bc)

    candidate_results=[]
    score_rows=[]
    trace_rows=[]
    for i,cand in enumerate(core.CANDIDATES):
        q=core.candidate_cohort(hybrid,team_features,cand)
        result,rows=core.score_candidate(q,cand,i*1000)
        candidate_results.append(result)
        score_rows.extend(rows)
        keep=[
            "season","week","game_id","team","opponent","player_clean_key",
            "player_identity_key","position","market","actual","baseline_projection",
            "baseline_source",cand.feature,"weakness_z"
        ]
        qt=q[keep].copy()
        beta=result["train"]["beta_train"]
        qt["candidate_id"]=cand.candidate_id
        qt["beta_train"]=beta
        qt["candidate_projection"]=(
            qt["baseline_projection"]+beta*qt["weakness_z"]
            if np.isfinite(beta) else np.nan
        )
        trace_rows.append(qt)

    confirmed=[r["candidate_id"] for r in candidate_results
               if r["disposition"]=="INTEGRATION_CANDIDATE_CONFIRMED"]

    payload={
        "version":VERSION,
        "repair_contract":"docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_CANDIDATE_PARITY_REPAIR.md",
        "train_season":2022,
        "primary_confirmation_season":2023,
        "secondary_consistency_seasons":[2024,2025],
        "evaluation_weeks":[2,18],
        "sportsbook_inputs_used":0,
        "production_changed":False,
        "combined_candidate_scored":False,
        "secondary_baseline_method":"DIRECT_FROZEN_RIGHT_TAIL_AUTHORITY",
        "secondary_feature_method":"DIRECT_FROZEN_PHASE_BC_FEATURE_AUTHORITY",
        "secondary_direct_authority_parity":{
            "status":"PASS",
            "method":"DIRECT_FROZEN_AUTHORITY",
            "baseline_max_gap_by_construction":0.0,
        },
        "independent_rebuild_drift_audit":drift,
        "candidates":candidate_results,
        "confirmed_candidates":confirmed,
        "confirmed_count":len(confirmed),
        "next_step":(
            "FREEZE_SEPARATE_PRODUCTION_ORDER_SHADOW_CONTRACT"
            if confirmed else "CLOSE_ALL_INTEGRATION_CANDIDATES"
        ),
    }

    a.out_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(score_rows).to_csv(
        a.out_dir/"football_matchup_integration_candidate_repair_scorecard.csv",
        index=False
    )
    pd.concat(trace_rows,ignore_index=True).to_csv(
        a.out_dir/"football_matchup_integration_candidate_repair_trace.csv",
        index=False
    )
    team_features.to_csv(
        a.out_dir/"football_matchup_integration_candidate_repair_team_features.csv",
        index=False
    )
    (a.out_dir/"football_matchup_integration_candidate_repair_result.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(payload,indent=2,sort_keys=True))
    print(pd.DataFrame(score_rows).to_string(index=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
