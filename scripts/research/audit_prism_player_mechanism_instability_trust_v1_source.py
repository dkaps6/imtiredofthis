#!/usr/bin/env python3
"""PRISM player mechanism-instability trust V1 historical source audit.

Source/chronology only. No reliability scoring, no predictive fit, no threshold
search, no 2026 outcomes, no sportsbook/OddsAPI, no production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_SOURCE_AUDIT"
TOL = 1e-10
MIN_PRIOR = 4
MAX_PRIOR = 8

REQUIRED = {
    "season", "week", "team", "player_clean_key", "prior_games",
    "prior8_abs_opportunity_error_mean", "prior8_abs_efficiency_error_mean",
    "current_opportunity_error", "current_efficiency_error", "current_total_error",
}


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = REQUIRED - set(x.columns)
    if miss:
        raise RuntimeError(f"{label} missing {sorted(miss)}")
    return x


def audit_panel(x: pd.DataFrame, position: str, expected_seasons: tuple[int, int]) -> tuple[pd.DataFrame, dict]:
    y = x.copy()
    y["season"] = pd.to_numeric(y["season"], errors="coerce")
    y["week"] = pd.to_numeric(y["week"], errors="coerce")
    for c in (
        "current_opportunity_error", "current_efficiency_error", "current_total_error",
        "prior8_abs_opportunity_error_mean", "prior8_abs_efficiency_error_mean",
    ):
        y[c] = pd.to_numeric(y[c], errors="coerce")

    if y[list(REQUIRED)].isna().any().any():
        cols=[c for c in REQUIRED if y[c].isna().any()]
        raise RuntimeError(f"{position} nonfinite required source fields: {cols}")

    seasons=tuple(sorted(int(v) for v in y["season"].unique()))
    if seasons != expected_seasons:
        raise RuntimeError(f"{position} season drift {seasons} != {expected_seasons}")

    keys=["season","week","team","player_clean_key"]
    if y.duplicated(keys).any():
        bad=y.loc[y.duplicated(keys,keep=False),keys].head(20).to_dict("records")
        raise RuntimeError(f"{position} duplicate player-week identity: {bad}")

    y["identity_gap"] = (
        y["current_opportunity_error"]
        + y["current_efficiency_error"]
        - y["current_total_error"]
    ).abs()
    max_gap=float(y["identity_gap"].max())
    if max_gap > TOL:
        raise RuntimeError(f"{position} decomposition identity gap {max_gap}")

    y=y.sort_values(["season","player_clean_key","week","team"],kind="mergesort").reset_index(drop=True)
    y["prior_component_rows_available"]=0
    y["prior_component_min_week"]=np.nan
    y["prior_component_max_week"]=np.nan
    eligible=[]

    for (_, _), idx in y.groupby(["season","player_clean_key"],sort=False).groups.items():
        inds=list(idx)
        for j, row_idx in enumerate(inds):
            prior=inds[max(0,j-MAX_PRIOR):j]
            y.loc[row_idx,"prior_component_rows_available"]=len(prior)
            if prior:
                y.loc[row_idx,"prior_component_min_week"]=int(y.loc[prior,"week"].min())
                y.loc[row_idx,"prior_component_max_week"]=int(y.loc[prior,"week"].max())
                if not (y.loc[prior,"week"] < y.loc[row_idx,"week"]).all():
                    raise RuntimeError(f"{position} chronology violation at row {row_idx}")
            eligible.append((row_idx,len(prior) >= MIN_PRIOR))

    eligible_map=dict(eligible)
    y["instability_state_source_eligible"]=[
        bool(eligible_map.get(i,False)) for i in y.index
    ]

    support=[]
    ready_all=True
    for season in expected_seasons:
        s=y.loc[y["season"].eq(season) & y["instability_state_source_eligible"]].copy()
        rows=int(len(s))
        players=int(s["player_clean_key"].nunique())
        ready=bool(rows >= 250 and players >= 50)
        ready_all &= ready
        support.append({
            "position":position,
            "season":season,
            "eligible_target_rows":rows,
            "distinct_players":players,
            "rows_gate":rows >= 250,
            "players_gate":players >= 50,
            "source_ready":ready,
        })

    summary={
        "position":position,
        "source_rows":int(len(y)),
        "source_players":int(y["player_clean_key"].nunique()),
        "seasons":list(expected_seasons),
        "max_decomposition_identity_gap":max_gap,
        "chronology_violations":0,
        "support":support,
        "source_ready":bool(ready_all),
    }
    return y, summary


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--wr",type=Path,required=True)
    ap.add_argument("--te",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)

    wr, wr_s = audit_panel(read_csv(args.wr,"WR mechanism panel"),"WR",(2023,2024))
    te, te_s = audit_panel(read_csv(args.te,"TE mechanism panel"),"TE",(2024,2025))

    ready=bool(wr_s["source_ready"] and te_s["source_ready"])
    result={
        "version":VERSION,
        "status":(
            "PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_SOURCE_READY_WR_TE"
            if ready else
            "PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_SOURCE_BLOCKED"
        ),
        "wr":wr_s,
        "te":te_s,
        "rb_status":"SOURCE_NOT_PROVEN_FOR_THIS_V1",
        "qb_status":"EXCLUDED_BY_ANTI_RETEST",
        "candidate_scored":False,
        "week4_5_candidate_rows_used":0,
        "outcomes_2026_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "models_fit":0,
        "threshold_searches":0,
        "projection_changed":False,
        "production_changed":False,
    }

    wr.to_csv(args.out_dir/"wr_mechanism_instability_source_rows.csv",index=False,float_format="%.12f")
    te.to_csv(args.out_dir/"te_mechanism_instability_source_rows.csv",index=False,float_format="%.12f")
    pd.DataFrame(wr_s["support"]+te_s["support"]).to_csv(
        args.out_dir/"mechanism_instability_source_support.csv",index=False
    )
    (args.out_dir/"mechanism_instability_source_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
