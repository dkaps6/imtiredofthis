#!/usr/bin/env python3
"""No-fit historical test of RB rush+receiving mean conservation.

Frozen candidate:
  rush_rec_yards = standalone rush_yards ensemble mean + standalone rec_yards ensemble mean

No sportsbook inputs, no coefficient search, no router.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["season", "week", "team", "opponent", "player_clean_key", "game_id"]
MARKETS = ["rush_yards", "rec_yards", "rush_rec_yards"]


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = actual - pred
    ae = np.abs(err)
    return {
        "n": int(len(actual)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(err.mean()),
        "abs_bias": float(abs(err.mean())),
        "p90_ae": float(np.quantile(ae, 0.90)),
        "miss30": int(np.sum(ae >= 30.0)),
    }


def score(q: pd.DataFrame) -> dict:
    actual = q["actual_rush_rec_yards"].to_numpy(float)
    base = q["ensemble_proj_rush_rec_yards"].to_numpy(float)
    cand = q["candidate_component_sum"].to_numpy(float)
    bm = metrics(actual, base)
    cm = metrics(actual, cand)
    bae = np.abs(actual - base)
    cae = np.abs(actual - cand)
    return {
        "n": int(len(q)),
        "baseline_mae": bm["mae"],
        "candidate_mae": cm["mae"],
        "mae_gain": bm["mae"] - cm["mae"],
        "baseline_rmse": bm["rmse"],
        "candidate_rmse": cm["rmse"],
        "baseline_bias": bm["bias"],
        "candidate_bias": cm["bias"],
        "baseline_abs_bias": bm["abs_bias"],
        "candidate_abs_bias": cm["abs_bias"],
        "baseline_p90_ae": bm["p90_ae"],
        "candidate_p90_ae": cm["p90_ae"],
        "baseline_miss30": bm["miss30"],
        "candidate_miss30": cm["miss30"],
        "paired_candidate_minus_baseline_ae_mean": float((cae - bae).mean()),
        "candidate_closer_rate": float(np.mean(cae < bae)),
        "ties_rate": float(np.mean(np.isclose(cae, bae, atol=1e-12, rtol=0))),
        "mean_projection_gap_combo_minus_components": float(q["projection_gap"].mean()),
        "mean_abs_projection_gap": float(q["projection_gap"].abs().mean()),
        "max_abs_projection_gap": float(q["projection_gap"].abs().max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x = pd.read_csv(args.trace, low_memory=False)
    required = set(KEY + ["position", "market", "ensemble_proj", "actual"])
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"trace missing columns: {sorted(missing)}")

    q = x[
        x["position"].astype(str).str.upper().eq("RB")
        & x["market"].astype(str).isin(MARKETS)
        & pd.to_numeric(x["season"], errors="coerce").between(2020, 2025)
    ].copy()
    q["season"] = pd.to_numeric(q["season"], errors="coerce").astype(int)
    q["week"] = pd.to_numeric(q["week"], errors="coerce").astype(int)

    if q.duplicated(KEY + ["market"]).any():
        sample = q.loc[q.duplicated(KEY + ["market"], keep=False), KEY + ["market", "player"]].head(20)
        raise RuntimeError("nonunique player-game-market identity: " + str(sample.to_dict("records")))

    wide = q.pivot(index=KEY, columns="market", values=["ensemble_proj", "actual"]).reset_index()
    wide.columns = [
        "_".join([str(v) for v in c if str(v)])
        if isinstance(c, tuple) else str(c)
        for c in wide.columns
    ]

    need = []
    for prefix in ["ensemble_proj", "actual"]:
        for m in MARKETS:
            need.append(f"{prefix}_{m}")
    for c in need:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")
    wide = wide.dropna(subset=need).copy()

    if wide.empty:
        raise RuntimeError("zero complete RB component triples")

    wide["actual_component_sum"] = wide["actual_rush_yards"] + wide["actual_rec_yards"]
    wide["actual_identity_gap"] = wide["actual_rush_rec_yards"] - wide["actual_component_sum"]
    max_actual_gap = float(wide["actual_identity_gap"].abs().max())
    if not np.isfinite(max_actual_gap) or max_actual_gap > 1e-9:
        sample = wide.loc[wide.actual_identity_gap.abs().gt(1e-9), KEY + [
            "actual_rush_yards","actual_rec_yards","actual_rush_rec_yards","actual_identity_gap"
        ]].head(20)
        raise RuntimeError(
            f"actual rush+receiving identity failed max_gap={max_actual_gap}: "
            + str(sample.to_dict("records"))
        )

    wide["candidate_component_sum"] = (
        wide["ensemble_proj_rush_yards"] + wide["ensemble_proj_rec_yards"]
    )
    wide["projection_gap"] = (
        wide["ensemble_proj_rush_rec_yards"] - wide["candidate_component_sum"]
    )

    pooled = score(wide)
    rows = [{"season": "POOLED", **pooled}]
    for season, g in wide.groupby("season", sort=True):
        rows.append({"season": int(season), **score(g)})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "season_summary.csv", index=False)
    wide.to_csv(args.out_dir / "paired_rows.csv", index=False)

    by_season = {
        int(r["season"]): r
        for r in rows
        if r["season"] != "POOLED"
    }
    gates = {
        "actual_identity_exact": max_actual_gap <= 1e-9,
        "pooled_mae_improves": pooled["candidate_mae"] < pooled["baseline_mae"],
        "season_2024_mae_improves": by_season[2024]["candidate_mae"] < by_season[2024]["baseline_mae"],
        "season_2025_mae_improves": by_season[2025]["candidate_mae"] < by_season[2025]["baseline_mae"],
        "pooled_rmse_nonworse": pooled["candidate_rmse"] <= pooled["baseline_rmse"] + 1e-12,
        "pooled_abs_bias_nonworse": pooled["candidate_abs_bias"] <= pooled["baseline_abs_bias"] + 1e-12,
        "pooled_p90_nonworse": pooled["candidate_p90_ae"] <= pooled["baseline_p90_ae"] + 1e-12,
        "pooled_miss30_nonincrease": pooled["candidate_miss30"] <= pooled["baseline_miss30"],
        "candidate_closer_gt_half": pooled["candidate_closer_rate"] > 0.50,
        "no_season_mae_degradation_gt_0_50": all(
            (r["candidate_mae"] - r["baseline_mae"]) <= 0.50 + 1e-12
            for r in rows if r["season"] != "POOLED"
        ),
    }
    passed = all(gates.values())
    payload = {
        "study": "RB_RUSH_REC_CONSERVATION_V2_MEAN",
        "disposition": (
            "RB_RUSH_REC_CONSERVATION_V2_MEAN_QUALIFIED"
            if passed else "RB_RUSH_REC_CONSERVATION_V2_MEAN_FAIL"
        ),
        "candidate_formula": "ensemble_proj_rush_yards + ensemble_proj_rec_yards",
        "baseline": "ensemble_proj_rush_rec_yards",
        "complete_player_games": int(len(wide)),
        "max_actual_identity_gap": max_actual_gap,
        "pooled": pooled,
        "season_rows": rows[1:],
        "gates": gates,
        "sportsbook_inputs_used": 0,
        "outcomes_2026_used": 0,
        "fitted_parameters": 0,
        "production_changed": False,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
