#!/usr/bin/env python3
"""GBM-vs-Ridge mean-correction trial on the frozen WR-R3 strict-prior signal.

Frozen plan: docs/migrations/WR_R3_GBM_VS_RIDGE_MEAN_CORRECTION_PLAN.md

Genuine holdout discipline: both model arms are fit ONCE on 2020-2024 rows
(prior_games >= MIN_PRIOR), frozen, then applied blind to 2025 only. No
refitting on 2025. Reuses the exact frozen WR-R3 strict-prior feature
artifact (wr_r3_walkforward_casebook.csv) and the same M38 rebuild/actual/
target-map/scoring helpers already validated in
evaluate_wr_r3_combined_calibration.py -- this script adds only the
model-fitting and model-based correction-application logic on top.

Research only. No production/model/weight/threshold change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from scripts._opponent_map import canon_team
from scripts.backtest.evaluate_wr_r3_combined_calibration import (
    EXPECTED_2025_ALL_REC,
    EXPECTED_2025_WR_ROWS,
    MIN_PRIOR,
    WR_POS,
    ITERATIONS,
    actual_map,
    load_r3_features,
    m38_map,
    miss_rate,
    num,
    one,
    parent_2025,
    prepared_metrics,
    read,
    score,
    target_map,
)
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week
from scripts import simulation_v2

TRAIN_SEASONS = [2020, 2021, 2022, 2023, 2024]
TEST_SEASON = 2025
FEATURE_COLS = ["prior8_m38_bias", "prior8_m38_mae", "prior8_m38_miss30_rate", "prior_games"]
CORRECTION_CLIP = 8.0
RIDGE_ALPHA = 20.0  # matches M89/M90's own frozen alpha


def key(v) -> str:
    from scripts.backtest.evaluate_wr_r3_combined_calibration import key as _key
    return _key(v)


def build_training_rows(features: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Direct residual target from already-produced per-season m38/actual maps.

    No historical-context-bundle rebuild or MC re-simulation needed for
    training rows: the fit target (actual - m38_baseline_proj) only depends on
    already-certified per-season m38.csv and player_game_logs, not on any
    re-simulated distribution.
    """
    rows = []
    for season in TRAIN_SEASONS:
        d = root / str(season)
        logs = read(d / "inputs" / "player_game_logs_history.csv", f"{season} player logs")
        pred = read(d / "m38.csv", f"{season} exact M38 predictions")
        baseline = m38_map(pred)
        # Re-index both maps by player_key only (dropping team) for O(1)
        # lookup -- a player_key is unique per week in practice, and both
        # source maps already dedupe within (week, team, player)/(team,
        # player), so this cannot silently merge two distinct players.
        baseline_by_week_player: dict[tuple[int, str], float] = {
            (w, pk): val for (w, _team, pk), val in baseline.items()
        }
        last_week = 17 if season == 2020 else 18
        for week in range(1, last_week + 1):
            actuals = actual_map(logs, season, week)
            actuals_by_player = {pk: val for (_team, pk), val in actuals.items()}
            f = features.loc[features.season.eq(season) & features.week.eq(week)]
            for _, r in f.iterrows():
                if int(r["prior_games"]) < MIN_PRIOR:
                    continue
                pkey = str(r["player_key"])
                base = baseline_by_week_player.get((week, pkey))
                if base is None:
                    continue
                actual = actuals_by_player.get(pkey)
                if actual is None:
                    continue
                feat = [num(r.get(c)) for c in FEATURE_COLS]
                if not all(np.isfinite(v) for v in feat):
                    continue
                rows.append({
                    "season": season, "week": week, "player_key": pkey,
                    **{c: feat[i] for i, c in enumerate(FEATURE_COLS)},
                    "baseline_proj": float(base), "actual": float(actual),
                    "residual": float(actual) - float(base),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no training rows built -- check root/features alignment")
    return out


def fit_frozen_models(train: pd.DataFrame) -> tuple[Ridge, HistGradientBoostingRegressor, dict]:
    X = train[FEATURE_COLS].to_numpy(dtype=float)
    y = train["residual"].to_numpy(dtype=float)

    ridge = Ridge(alpha=RIDGE_ALPHA, random_state=42)
    ridge.fit(X, y)

    gbm = HistGradientBoostingRegressor(
        max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=30, random_state=42,
    )
    gbm.fit(X, y)

    train_diag = {
        "train_rows": int(len(train)),
        "ridge_train_mae": float(np.mean(np.abs(ridge.predict(X) - y))),
        "gbm_train_mae": float(np.mean(np.abs(gbm.predict(X) - y))),
        "naive_zero_correction_train_mae": float(np.mean(np.abs(y))),
    }
    return ridge, gbm, train_diag


def apply_model_correction(metrics: pd.DataFrame, targets: pd.DataFrame, fmap: dict,
                            season: int, week: int, model) -> pd.DataFrame:
    out = metrics.copy()
    tmap = {(str(r.event_id), canon_team(r.team), str(r.player_key)): r for r in targets.itertuples(index=False)}
    for i, r in out.iterrows():
        if str(r.get("position", "") or "").upper().strip() not in WR_POS:
            continue
        pkey = key(r.get("player_clean_key", ""))
        f = fmap.get((int(season), int(week), pkey))
        if not f or int(f.get("prior_games", 0)) < MIN_PRIOR:
            continue
        feat = [num(f.get(c)) for c in FEATURE_COLS]
        if not all(np.isfinite(v) for v in feat):
            continue
        tm = tmap.get((str(r.get("event_id")), canon_team(r.get("team")), pkey))
        if tm is None:
            continue
        yard_delta = float(np.clip(float(model.predict([feat])[0]), -CORRECTION_CLIP, CORRECTION_CLIP))
        base_ypt = num(r.get("rules_ypt"), num(r.get("bayes_ypt"), num(r.get("ypt"), 7.5)))
        out.at[i, "rules_ypt"] = float(np.clip(base_ypt + yard_delta / max(float(tm.pred_targets), 1.0), 2.0, 20.0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--r3-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    features = load_r3_features(a.r3_root)
    fmap = {(int(r.season), int(r.week), str(r.player_key)): r._asdict() for r in features.itertuples(index=False)}

    train = build_training_rows(features, a.root)
    train.to_csv(a.out_dir / "gbm_vs_ridge_training_rows.csv", index=False)
    ridge, gbm, train_diag = fit_frozen_models(train)
    (a.out_dir / "gbm_vs_ridge_train_diagnostics.json").write_text(json.dumps(train_diag, indent=2, sort_keys=True))

    d = a.root / str(TEST_SEASON); inp = d / "inputs"
    logs = read(inp / "player_game_logs_history.csv", f"{TEST_SEASON} player logs")
    team_weekly = read(inp / "team_weekly_history.csv", f"{TEST_SEASON} team weekly")
    schedule = read(inp / "schedule_history.csv", f"{TEST_SEASON} schedule")
    injuries = read(inp / "injuries_history.csv", f"{TEST_SEASON} injuries")
    weather = read(inp / "weather_history.csv", f"{TEST_SEASON} weather")
    pred = read(d / "m38.csv", f"{TEST_SEASON} exact M38 predictions")
    p2025 = parent_2025(pred)
    baseline = m38_map(pred)

    rows = []
    base_repro_max = 0.0
    for week in range(1, 19):
        universe = read(inp / "pregame_universe" / f"{TEST_SEASON}_week_{week:02d}.csv", f"{TEST_SEASON} W{week} universe")
        bundle = build_historical_context_bundle(
            player_logs=logs, team_weekly=team_weekly, pregame_universe=universe, schedule=schedule,
            season=TEST_SEASON, week=week, prior_season=TEST_SEASON - 1,
            injuries=_exact_week(injuries, TEST_SEASON, week), weather=_exact_week(weather, TEST_SEASON, week),
        )
        metrics = prepared_metrics(bundle)
        targets = target_map(metrics, TEST_SEASON, week)

        base_sim = simulation_v2.simulate(metrics, iterations=ITERATIONS, seed=42 + week)
        ridge_metrics = apply_model_correction(metrics, targets, fmap, TEST_SEASON, week, ridge)
        ridge_sim = simulation_v2.simulate(ridge_metrics, iterations=ITERATIONS, seed=42 + week)
        gbm_metrics = apply_model_correction(metrics, targets, fmap, TEST_SEASON, week, gbm)
        gbm_sim = simulation_v2.simulate(gbm_metrics, iterations=ITERATIONS, seed=42 + week)

        actuals = actual_map(logs, TEST_SEASON, week)
        roles = {(str(r.event_id), canon_team(r.team), str(r.player_key)): r for r in targets.itertuples(index=False)}
        pcols = ["event_id", "team", "player_clean_key"]
        players = metrics.sort_values(pcols).drop_duplicates(pcols, keep="last")
        for _, r in players.iterrows():
            if str(r.get("position", "") or "").upper().strip() not in WR_POS:
                continue
            pkey = key(r.get("player_clean_key", "")); team = canon_team(r.get("team"))
            actual = actuals.get((team, pkey)); base = baseline.get((week, team, pkey))
            if actual is None or base is None:
                continue
            bo = simulation_v2.lookup(base_sim, r, "rec_yards")
            ro = simulation_v2.lookup(ridge_sim, r, "rec_yards")
            go = simulation_v2.lookup(gbm_sim, r, "rec_yards")
            if bo is None or ro is None or go is None or not len(bo) or not len(ro) or not len(go):
                continue
            base_mean = float(np.mean(bo))
            base_repro_max = max(base_repro_max, abs(base_mean - float(base)))
            f = fmap.get((TEST_SEASON, week, pkey), {})
            tm = roles.get((str(r.get("event_id")), team, pkey))
            rows.append({
                "season": TEST_SEASON, "week": week, "team": team, "player_key": pkey,
                "player": r.get("player", ""), "wr_role": getattr(tm, "wr_role", "NON_WR") if tm is not None else "NON_WR",
                "actual": float(actual), "baseline_proj": float(base),
                "ridge_proj": float(np.mean(ro)), "gbm_proj": float(np.mean(go)),
                "prior_games": int(f.get("prior_games", 0) or 0),
                "eligible": int(int(f.get("prior_games", 0) or 0) >= MIN_PRIOR),
            })

    r25 = pd.DataFrame(rows)
    r25.to_csv(a.out_dir / "gbm_vs_ridge_2025_rows.csv", index=False)
    if r25.empty:
        raise RuntimeError("no 2025 test rows produced")

    parent_gates = {
        "parent_n_exact": p2025["n"] == EXPECTED_2025_ALL_REC["n"],
        "parent_mae_exact": abs(p2025["mae"] - EXPECTED_2025_ALL_REC["mae"]) <= 1e-6,
        "wr_2025_rows_exact": len(r25) == EXPECTED_2025_WR_ROWS,
        "baseline_sim_reproduction": base_repro_max <= 1e-6,
    }

    b25 = score(r25["actual"], r25["baseline_proj"])
    arm_results = {}
    for arm in ["ridge", "gbm"]:
        c25 = score(r25["actual"], r25[f"{arm}_proj"])
        gates = {
            "mae_improve_ge_1pct": c25["mae"] <= 0.99 * b25["mae"],
            "rmse_nonworse": c25["rmse"] <= b25["rmse"],
            "abs_bias_nonworse": abs(c25["bias"]) <= abs(b25["bias"]),
            "correlation_nonworse_tol_005": c25["correlation"] >= b25["correlation"] - 0.005,
        }
        for thr in [20.0, 30.0, 40.0]:
            b_miss = miss_rate(r25, "baseline_proj", thr)
            c_miss = miss_rate(r25, f"{arm}_proj", thr)
            gates[f"miss{int(thr)}_nonworse"] = c_miss <= b_miss
        role_rows = []
        for role in ["WR1", "WR2", "WR3"]:
            q = r25.loc[r25.wr_role.eq(role)]
            if q.empty:
                continue
            b = score(q["actual"], q["baseline_proj"]); c = score(q["actual"], q[f"{arm}_proj"])
            role_rows.append({"wr_role": role, "n": len(q), "baseline_mae": b["mae"], f"{arm}_mae": c["mae"]})
        arm_results[arm] = {
            "baseline_2025": b25, "candidate_2025": c25, "gates": {k: bool(v) for k, v in gates.items()},
            "all_gates_pass": all(gates.values()), "role_breakdown": role_rows,
        }

    result = {
        "trial": "WR_R3_GBM_VS_RIDGE_MEAN_CORRECTION",
        "m38_parent": "b98518d97b3038f471aee9ae3201009b2c70bb29",
        "r3_source_run": 34064572328,
        "train_seasons": TRAIN_SEASONS, "test_season": TEST_SEASON,
        "iterations": ITERATIONS, "sportsbook_inputs_used": False, "production_changed": False,
        "train_diagnostics": train_diag,
        "parent_2025_all_rec": p2025,
        "baseline_sim_reproduction_max_abs_delta": base_repro_max,
        "parent_gates": {k: bool(v) for k, v in parent_gates.items()},
        "arms": arm_results,
    }
    (a.out_dir / "gbm_vs_ridge_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
