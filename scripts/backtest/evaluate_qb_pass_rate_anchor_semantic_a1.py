#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

KEYS = ["season", "week", "team", "player_clean_key"]
ANCHORS = [0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63]
BASE_ANCHOR = 0.57
BOOT_N = 5000
SEED = 5701
TOL = 1e-9


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def num(v):
    return pd.to_numeric(v, errors="coerce")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].fillna("").astype(str).map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str).str.strip()
    return x


def load_play(root: Path) -> pd.DataFrame:
    p = one(root, "play_rate_decomposition_casebook.csv")
    x = normalize(pd.read_csv(p, low_memory=False))
    need = KEYS + ["pred_d", "actual_d", "pred_plays", "pred_rate", "actual_rate"]
    missing = [c for c in need if c not in x.columns]
    if missing:
        raise RuntimeError(f"play artifact missing columns {missing}")
    x = x.loc[x["season"].eq(2024), need].copy()
    if len(x) != 444 or x.duplicated(KEYS).any():
        raise RuntimeError(f"play development cohort drift rows={len(x)} dup={int(x.duplicated(KEYS).sum())}")
    for c in ["pred_d", "actual_d", "pred_plays", "pred_rate", "actual_rate"]:
        x[c] = num(x[c])
    return x


def load_chain(root: Path) -> pd.DataFrame:
    p = one(root, "qb_opportunity_chain_casebook.csv")
    x = normalize(pd.read_csv(p, low_memory=False))
    need = KEYS + [
        "actual_pass_yards", "actual_attempts", "pred_attempts", "pred_ypa",
        "pred_c", "pred_s", "football_synthesis",
    ]
    missing = [c for c in need if c not in x.columns]
    if missing:
        raise RuntimeError(f"chain artifact missing columns {missing}")
    x = x.loc[x["season"].eq(2024), need].copy()
    if len(x) != 444 or x.duplicated(KEYS).any():
        raise RuntimeError(f"chain development cohort drift rows={len(x)} dup={int(x.duplicated(KEYS).sum())}")
    for c in need:
        if c not in KEYS:
            x[c] = num(x[c])
    return x


def metric(actual, pred) -> dict:
    z = pd.DataFrame({"a": num(actual), "p": num(pred)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan, "p90_abs_error": np.nan}
    e = z["p"] - z["a"]
    return {
        "n": int(len(z)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "bias": float(e.mean()),
        "corr": float(z["a"].corr(z["p"])) if z["p"].nunique(dropna=True) > 1 and len(z) >= 3 else np.nan,
        "p90_abs_error": float(e.abs().quantile(0.90)),
    }


def boot_gain(actual, base, cand, seed: int) -> dict:
    a = num(actual).to_numpy(float)
    b = num(base).to_numpy(float)
    c = num(cand).to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    a, b, c = a[ok], b[ok], c[ok]
    n = len(a)
    rng = np.random.default_rng(seed)
    gains = np.empty(BOOT_N, dtype=float)
    for i in range(BOOT_N):
        idx = rng.integers(0, n, size=n)
        gains[i] = np.mean(np.abs(b[idx] - a[idx])) - np.mean(np.abs(c[idx] - a[idx]))
    return {
        "n": int(n), "draws": BOOT_N, "seed": int(seed),
        "mean_gain": float(gains.mean()),
        "p_gain_gt_0": float((gains > 0).mean()),
        "p05": float(np.quantile(gains, 0.05)),
        "p50": float(np.quantile(gains, 0.50)),
        "p95": float(np.quantile(gains, 0.95)),
    }


def bias_improvement(base_bias: float, cand_bias: float) -> float:
    b = abs(float(base_bias))
    c = abs(float(cand_bias))
    if b <= 1e-12:
        return 0.0 if c > 1e-12 else 1.0
    return float((b - c) / b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--play-root", type=Path, required=True)
    ap.add_argument("--chain-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    play = load_play(a.play_root)
    chain = load_chain(a.chain_root)
    z = play.merge(chain, on=KEYS, how="inner", validate="one_to_one")
    if len(z) != 444:
        raise RuntimeError(f"merged development cohort drift rows={len(z)}")
    if z[["pred_d", "actual_d", "pred_plays", "pred_rate", "actual_rate", "actual_attempts", "pred_attempts", "pred_ypa", "pred_c", "pred_s", "actual_pass_yards"]].isna().any().any():
        bad = z.isna().sum().loc[lambda s: s.gt(0)].to_dict()
        raise RuntimeError(f"development cohort has missing required values {bad}")

    baseline_d_identity = float((z["pred_d"] - z["pred_plays"] * BASE_ANCHOR).abs().max())
    baseline_attempt_identity = float((z["pred_attempts"] - z["pred_d"] * z["pred_c"] * z["pred_s"]).abs().max())
    baseline_rate_exact = bool(np.isclose(z["pred_rate"], BASE_ANCHOR, atol=1e-12, rtol=0).all())

    candidate_rows = []
    traces = []
    by_anchor: dict[float, dict] = {}
    for anchor in ANCHORS:
        rate = pd.Series(float(anchor), index=z.index, dtype=float)
        d = z["pred_plays"] * float(anchor)
        attempts = d * z["pred_c"] * z["pred_s"]
        mech_yards = attempts * z["pred_ypa"]
        rate_m = metric(z["actual_rate"], rate)
        d_m = metric(z["actual_d"], d)
        a_m = metric(z["actual_attempts"], attempts)
        y_m = metric(z["actual_pass_yards"], mech_yards)
        rec = {
            "anchor": float(anchor),
            "pass_rate": rate_m,
            "team_pass_opportunity": d_m,
            "qb_attempts": {
                **a_m,
                "miss_8_plus_rate": float((z["actual_attempts"].sub(attempts).abs() >= 8).mean()),
                "miss_10_plus_rate": float((z["actual_attempts"].sub(attempts).abs() >= 10).mean()),
            },
            "mechanics_pass_yards": y_m,
        }
        by_anchor[float(anchor)] = rec
        candidate_rows.append({
            "anchor": float(anchor),
            "pass_rate_mae": rate_m["mae"], "pass_rate_rmse": rate_m["rmse"], "pass_rate_bias": rate_m["bias"], "pass_rate_p90": rate_m["p90_abs_error"],
            "team_d_mae": d_m["mae"], "team_d_rmse": d_m["rmse"], "team_d_bias": d_m["bias"], "team_d_corr": d_m["corr"], "team_d_p90": d_m["p90_abs_error"],
            "qb_attempt_mae": a_m["mae"], "qb_attempt_rmse": a_m["rmse"], "qb_attempt_bias": a_m["bias"], "qb_attempt_corr": a_m["corr"], "qb_attempt_p90": a_m["p90_abs_error"],
            "qb_attempt_8_plus_miss_rate": rec["qb_attempts"]["miss_8_plus_rate"], "qb_attempt_10_plus_miss_rate": rec["qb_attempts"]["miss_10_plus_rate"],
            "mechanics_pass_yards_mae": y_m["mae"], "mechanics_pass_yards_rmse": y_m["rmse"], "mechanics_pass_yards_bias": y_m["bias"], "mechanics_pass_yards_corr": y_m["corr"], "mechanics_pass_yards_p90": y_m["p90_abs_error"],
        })
        t = z[KEYS + ["pred_plays", "actual_rate", "actual_d", "actual_attempts", "actual_pass_yards", "pred_c", "pred_s", "pred_ypa"]].copy()
        t["anchor"] = float(anchor)
        t["candidate_rate"] = rate
        t["candidate_d"] = d
        t["candidate_qb_attempts"] = attempts
        t["candidate_mechanics_pass_yards"] = mech_yards
        traces.append(t)

    summary = pd.DataFrame(candidate_rows).sort_values(
        ["pass_rate_mae", "team_d_mae", "qb_attempt_mae", "anchor"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    winner_anchor = float(summary.iloc[0]["anchor"])
    base = by_anchor[BASE_ANCHOR]
    win = by_anchor[winner_anchor]

    gains = {
        "pass_rate_mae": float(base["pass_rate"]["mae"] - win["pass_rate"]["mae"]),
        "team_d_mae": float(base["team_pass_opportunity"]["mae"] - win["team_pass_opportunity"]["mae"]),
        "qb_attempt_mae": float(base["qb_attempts"]["mae"] - win["qb_attempts"]["mae"]),
        "mechanics_pass_yards_mae": float(base["mechanics_pass_yards"]["mae"] - win["mechanics_pass_yards"]["mae"]),
    }
    bias_gain = {
        "pass_rate": bias_improvement(base["pass_rate"]["bias"], win["pass_rate"]["bias"]),
        "team_d": bias_improvement(base["team_pass_opportunity"]["bias"], win["team_pass_opportunity"]["bias"]),
        "qb_attempts": bias_improvement(base["qb_attempts"]["bias"], win["qb_attempts"]["bias"]),
    }

    win_rate = pd.Series(winner_anchor, index=z.index, dtype=float)
    win_d = z["pred_plays"] * winner_anchor
    base_d = z["pred_plays"] * BASE_ANCHOR
    win_att = win_d * z["pred_c"] * z["pred_s"]
    base_att = base_d * z["pred_c"] * z["pred_s"]
    boot = {
        "pass_rate": boot_gain(z["actual_rate"], pd.Series(BASE_ANCHOR, index=z.index), win_rate, SEED),
        "team_d": boot_gain(z["actual_d"], base_d, win_d, SEED + 1),
        "qb_attempts": boot_gain(z["actual_attempts"], base_att, win_att, SEED + 2),
    }

    adj_ok = False
    for neighbor in (round(winner_anchor - 0.01, 2), round(winner_anchor + 0.01, 2)):
        if neighbor in by_anchor:
            nm = by_anchor[neighbor]
            if nm["pass_rate"]["mae"] < base["pass_rate"]["mae"] and nm["team_pass_opportunity"]["mae"] < base["team_pass_opportunity"]["mae"]:
                adj_ok = True

    integrity = {
        "exact_444_2024_rows": len(z) == 444,
        "unique_keys": not z.duplicated(KEYS).any(),
        "2025_candidate_scored_or_summarized": False,
        "zero_sportsbook_inputs": True,
        "zero_model_fitting": True,
        "zero_production_changes": True,
        "candidate_grid_exact": [float(x) for x in ANCHORS] == [0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63],
        "baseline_rate_exact_0_57": baseline_rate_exact,
        "baseline_d_identity_max_abs": baseline_d_identity,
        "baseline_attempt_identity_max_abs": baseline_attempt_identity,
        "m89_synthesis_used_for_selection": False,
    }
    integrity_pass = bool(
        integrity["exact_444_2024_rows"]
        and integrity["unique_keys"]
        and not integrity["2025_candidate_scored_or_summarized"]
        and integrity["zero_sportsbook_inputs"]
        and integrity["zero_model_fitting"]
        and integrity["zero_production_changes"]
        and integrity["candidate_grid_exact"]
        and integrity["baseline_rate_exact_0_57"]
        and baseline_d_identity <= TOL
        and baseline_attempt_identity <= TOL
        and not integrity["m89_synthesis_used_for_selection"]
    )
    if not integrity_pass:
        raise RuntimeError(f"A1 integrity failure: {integrity}")

    gates = {
        "all_integrity_gates_pass": integrity_pass,
        "winner_gt_0_57": winner_anchor > 0.57,
        "winner_lt_0_63": winner_anchor < 0.63,
        "pass_rate_mae_gain_ge_0_0040": gains["pass_rate_mae"] >= 0.0040,
        "team_d_mae_gain_ge_0_20": gains["team_d_mae"] >= 0.20,
        "qb_attempt_mae_gain_ge_0_15": gains["qb_attempt_mae"] >= 0.15,
        "pass_rate_abs_bias_improve_ge_25pct": bias_gain["pass_rate"] >= 0.25,
        "team_d_abs_bias_improve_ge_25pct": bias_gain["team_d"] >= 0.25,
        "qb_attempt_abs_bias_improve_ge_20pct": bias_gain["qb_attempts"] >= 0.20,
        "pass_rate_p90_nonworse": win["pass_rate"]["p90_abs_error"] <= base["pass_rate"]["p90_abs_error"] + 1e-12,
        "team_d_p90_nonworse": win["team_pass_opportunity"]["p90_abs_error"] <= base["team_pass_opportunity"]["p90_abs_error"] + 1e-12,
        "qb_attempt_p90_nonworse": win["qb_attempts"]["p90_abs_error"] <= base["qb_attempts"]["p90_abs_error"] + 1e-12,
        "qb_10_plus_miss_rate_nonworse": win["qb_attempts"]["miss_10_plus_rate"] <= base["qb_attempts"]["miss_10_plus_rate"] + 1e-12,
        "mechanics_pass_yards_mae_guardrail": win["mechanics_pass_yards"]["mae"] <= base["mechanics_pass_yards"]["mae"] + 0.25,
        "mechanics_pass_yards_p90_guardrail": win["mechanics_pass_yards"]["p90_abs_error"] <= base["mechanics_pass_yards"]["p90_abs_error"] + 1.0,
        "bootstrap_pass_rate_p_ge_0_95": boot["pass_rate"]["p_gain_gt_0"] >= 0.95,
        "bootstrap_team_d_p_ge_0_95": boot["team_d"]["p_gain_gt_0"] >= 0.95,
        "bootstrap_qb_attempt_p_ge_0_90": boot["qb_attempts"]["p_gain_gt_0"] >= 0.90,
        "adjacent_anchor_robustness": adj_ok,
    }
    all_pass = bool(all(gates.values()))
    disposition = "QB_PASS_RATE_ANCHOR_SEMANTIC_A1_PASS_READY_FOR_2025_CONFIRMATION" if all_pass else "QB_PASS_RATE_ANCHOR_SEMANTIC_A1_FAIL_NO_CONFIRMATION"

    result = {
        "migration": "QB_PASS_RATE_ANCHOR_SEMANTIC_RECALIBRATION_A1",
        "development_season": 2024,
        "confirmation_season_scored": False,
        "candidate_grid": ANCHORS,
        "baseline_anchor": BASE_ANCHOR,
        "winner_anchor": winner_anchor,
        "selection_rule": "min pass-rate MAE, then team-D MAE, then QB-attempt MAE, then lower anchor",
        "gains_vs_0_57": gains,
        "absolute_bias_improvement_fraction": bias_gain,
        "baseline_metrics": base,
        "winner_metrics": win,
        "bootstrap": boot,
        "integrity": integrity,
        "advance_gates": gates,
        "all_advance_gates_pass": all_pass,
        "disposition": disposition,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "m89_m90_synthesis_changed": False,
    }

    summary.to_csv(a.out_dir / "qb_pass_rate_anchor_a1_candidate_summary.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(a.out_dir / "qb_pass_rate_anchor_a1_casebook.csv", index=False)
    with open(a.out_dir / "qb_pass_rate_anchor_a1_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
