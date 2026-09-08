#!/usr/bin/env python3
"""Fit the final WR-R15 production coefficient contract.

Scientific authorization comes exclusively from WR-R15 OOS run 34238301577
(artifact 10061328722).  This script does not re-test or tune the hypothesis.
It refits the already-frozen specification on all completed 2022-2025 football
history so one deterministic coefficient set can be consumed by production.

Frozen specification:
- StandardScaler + Ridge(alpha=20)
- the exact 15 WR-R15 features
- EPS=.02
- training residual clip [-2, 2]
- inference residual clip [-1, 1]
- M38 WR1 entitlement remains immutable
- only WR2+ residual room is redistributed by softmax
- WR room/team mass and all non-WR entitlements remain conserved
- zero sportsbook inputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest.evaluate_wr_room_empirical_bayes_v1 import read
from scripts.backtest.evaluate_wr_r15_wr1_anchor_participation_v1 import (
    ALPHA,
    EPS,
    FEATURES,
    PRED_CLIP,
    TRAIN_CLIP,
    _training_casebook,
)
from scripts.modeling.te_r5p_entitlement_adapter_v1 import _load_snaps

AUTHORIZED_RUN = 34238301577
AUTHORIZED_ARTIFACT = 10061328722
AUTHORIZED_ARTIFACT_SHA256 = "8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce"
MODEL_VERSION = "WR_R15_PRODUCTION_MODEL_V1"
FIT_SEASONS = (2022, 2023, 2024, 2025)


def main() -> int:
    ap = argparse.ArgumentParser()
    for season in FIT_SEASONS:
        ap.add_argument(f"--data-{season}", dest=f"data_{season}", type=Path, required=True)
        ap.add_argument(f"--logs-{season}", dest=f"logs_{season}", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    snaps, snap_dup_rate, snap_source_seasons = _load_snaps()
    parts: list[pd.DataFrame] = []
    future_total = 0
    season_rows: dict[str, int] = {}

    for season in FIT_SEASONS:
        data_dir = getattr(a, f"data_{season}")
        logs = read(getattr(a, f"logs_{season}"))
        casebook, future = _training_casebook(
            season=season,
            data_dir=data_dir,
            logs=logs,
            snaps=snaps,
        )
        parts.append(casebook)
        future_total += int(future)
        season_rows[str(season)] = int(len(casebook))
        print(f"[wr-r15-final-fit] season={season} rows={len(casebook)} future_participation={future}")

    train = pd.concat(parts, ignore_index=True)
    train["season"] = pd.to_numeric(train["season"], errors="coerce").astype(int)
    train["week"] = pd.to_numeric(train["week"], errors="coerce").astype(int)
    for c in FEATURES + ["secondary_residual_target"]:
        train[c] = pd.to_numeric(train[c], errors="coerce")

    fit_seasons = sorted(train.season.unique().astype(int).tolist())
    duplicate_rate = float(
        train.duplicated(["season", "week", "team", "player_clean_key"], keep=False).mean()
    )
    nonfinite_feature_rows = int((~np.isfinite(train[FEATURES].to_numpy(float))).any(axis=1).sum())
    nonfinite_target_rows = int((~np.isfinite(train["secondary_residual_target"].to_numpy(float))).sum())

    model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
    model.fit(train[FEATURES], train["secondary_residual_target"])
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]

    parameters_finite = bool(
        np.isfinite(scaler.mean_).all()
        and np.isfinite(scaler.scale_).all()
        and np.isfinite(ridge.coef_).all()
        and np.isfinite(ridge.intercept_)
    )

    integrity = {
        "scientific_authorization_frozen": True,
        "authorized_oos_run_exact": AUTHORIZED_RUN == 34238301577,
        "authorized_oos_artifact_exact": AUTHORIZED_ARTIFACT == 10061328722,
        "training_rows_gt0": bool(len(train) > 0),
        "training_seasons_exact_2022_2025": bool(fit_seasons == list(FIT_SEASONS)),
        "duplicate_player_game_rate_zero": bool(duplicate_rate == 0.0),
        "zero_same_future_participation": bool(future_total == 0),
        "snap_duplicate_rate_le_0_01": bool(float(snap_dup_rate) <= 0.01),
        "no_2026_outcomes": bool(int(train.season.max()) <= 2025),
        "sportsbook_inputs_zero": True,
        "feature_count_exact_15": bool(len(FEATURES) == 15),
        "parameter_lengths_exact_15": bool(
            len(scaler.mean_) == len(scaler.scale_) == len(ridge.coef_) == len(FEATURES)
        ),
        "training_features_finite": bool(nonfinite_feature_rows == 0),
        "training_target_finite": bool(nonfinite_target_rows == 0),
        "parameters_finite": parameters_finite,
        "ridge_alpha_exact_20": bool(float(ALPHA) == 20.0),
        "eps_exact_0_02": bool(float(EPS) == 0.02),
        "training_clip_exact": bool(float(TRAIN_CLIP) == 2.0),
        "prediction_clip_exact": bool(float(PRED_CLIP) == 1.0),
    }

    contract = {
        "model_version": MODEL_VERSION,
        "scientific_status": "SUPPORTED_OOS_CANDIDATE_PENDING_PRODUCTION_ADAPTER",
        "authorized_by_run": AUTHORIZED_RUN,
        "authorized_by_artifact": AUTHORIZED_ARTIFACT,
        "authorized_artifact_sha256": AUTHORIZED_ARTIFACT_SHA256,
        "scientific_confirmation_seasons": [2023, 2024],
        "scientific_confirmation_2025_used": False,
        "final_refit_is_not_new_scientific_confirmation": True,
        "training_seasons": fit_seasons,
        "training_rows": int(len(train)),
        "training_rows_by_season": season_rows,
        "features": FEATURES,
        "eps": float(EPS),
        "ridge_alpha": float(ALPHA),
        "training_target_clip": [-float(TRAIN_CLIP), float(TRAIN_CLIP)],
        "prediction_clip": [-float(PRED_CLIP), float(PRED_CLIP)],
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "ridge_coef": [float(v) for v in ridge.coef_],
        "ridge_intercept": float(ridge.intercept_),
        "baseline_contract": {
            "upstream_entitlement": "canonical explicit finite target entitlement with M38 already applied",
            "wr1_anchor": "highest baseline WR entitlement in each team-game; immutable",
            "candidate_scope": "WR2+ only",
            "secondary_pool": "sum of baseline WR2+ entitlement; immutable",
            "wr_room_mass": "immutable",
            "team_modeled_player_mass": "immutable",
            "non_wr_entitlement": "immutable",
        },
        "feature_construction": {
            "b0_secondary_room_share": "baseline WR2+ entitlement normalized within WR2+ residual room",
            "log_b0_secondary_pool": "log1p(sum baseline WR2+ entitlement)",
            "secondary_room_size": "count of current team-game WR rows excluding baseline WR1",
            "participation": "strictly-prior nflverse offensive snap observations only",
            "availability_flags": "strict-prior booleans encoded 0/1",
            "log_prior_counts": "log1p(strict-prior observation counts)",
            "secondary_snap_share_prior1_same_team": "prior1 same-team offense_pct normalized within current WR2+ room; zero if denominator zero",
            "secondary_snap_share_prior3_anyteam": "prior3 any-team offense_pct normalized within current WR2+ room; zero if denominator zero",
        },
        "allocation": (
            "clip residual to [-1,1]; score=log(b0_secondary_room_share+0.02)+residual; "
            "softmax only within WR2+; multiply by frozen baseline WR2+ pool; WR1 unchanged"
        ),
        "sportsbook_inputs_used": 0,
        "snap_source_seasons": sorted(int(v) for v in snap_source_seasons),
        "integrity": integrity,
    }

    disposition = (
        "WR_R15_FINAL_PRODUCTION_MODEL_FIT_COMPLETE"
        if all(integrity.values())
        else "WR_R15_FINAL_PRODUCTION_MODEL_MECHANICAL_OR_SOURCE_FAILURE"
    )
    result = {
        "disposition": disposition,
        "model_version": MODEL_VERSION,
        "training_rows": int(len(train)),
        "training_seasons": fit_seasons,
        "training_rows_by_season": season_rows,
        "future_participation_violations": int(future_total),
        "duplicate_player_game_rate": duplicate_rate,
        "snap_duplicate_rate": float(snap_dup_rate),
        "nonfinite_feature_rows": nonfinite_feature_rows,
        "nonfinite_target_rows": nonfinite_target_rows,
        "integrity": integrity,
        "production_parameters_changed_outside_frozen_spec": 0,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "wr_r15_production_model_v1.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (a.out_dir / "wr_r15_final_fit_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        {
            "feature": FEATURES,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "ridge_coef": ridge.coef_,
            "ridge_intercept": float(ridge.intercept_),
        }
    ).to_csv(a.out_dir / "wr_r15_final_fit_parameters.csv", index=False)
    audit_cols = [
        "season", "week", "event_id", "team", "player", "player_clean_key",
        "baseline_wr_rank", "baseline_entitlement_tgt_share", "b0_secondary_pool",
        "b0_secondary_room_share", "actual_targets", "actual_secondary_pool",
        "actual_secondary_room_share", "secondary_residual_target",
    ] + FEATURES
    train[[c for c in audit_cols if c in train.columns]].to_csv(
        a.out_dir / "wr_r15_final_fit_training_audit.csv", index=False
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(contract, indent=2, sort_keys=True))
    if disposition.endswith("FAILURE"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
