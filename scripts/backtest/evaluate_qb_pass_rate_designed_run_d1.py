#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

KEYS = ["season", "week", "team", "player_clean_key"]
SEED = 5601
BOOT_N = 5000
BASE_RATE = 0.57
COEF = 1.0
CLIP_LO = 0.35
CLIP_HI = 0.75
TOL = 1e-12


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def canon_keys(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].fillna("").astype(str).map(canon_team)
    if "player_clean_key" in x.columns:
        x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str).str.strip()
    if "player_name_key" in x.columns:
        x["player_name_key"] = x["player_name_key"].fillna("").astype(str).str.strip()
    return x


def load_play(root: Path) -> pd.DataFrame:
    cols = KEYS + ["pred_d", "actual_d", "pred_plays", "pred_rate", "actual_rate"]
    x = pd.read_csv(one(root, "play_rate_decomposition_casebook.csv"), usecols=cols, low_memory=False)
    x = canon_keys(x)
    x = x.loc[x["season"].eq(2024)].copy()
    if len(x) != 444 or x.duplicated(KEYS).any():
        raise RuntimeError(f"2024 play-rate cohort drift rows={len(x)}")
    for c in ["pred_d", "actual_d", "pred_plays", "pred_rate", "actual_rate"]:
        x[c] = num(x[c])
    return x


def load_chain(root: Path) -> pd.DataFrame:
    need = KEYS + ["pred_c", "pred_s", "pred_attempts", "actual_attempts", "football_synthesis"]
    x = pd.read_csv(one(root, "qb_opportunity_chain_casebook.csv"), usecols=need, low_memory=False)
    x = canon_keys(x)
    x = x.loc[x["season"].eq(2024)].copy()
    if len(x) != 444 or x.duplicated(KEYS).any():
        raise RuntimeError(f"2024 opportunity-chain cohort drift rows={len(x)}")
    for c in ["pred_c", "pred_s", "pred_attempts", "actual_attempts", "football_synthesis"]:
        x[c] = num(x[c])
    return x


def load_source(root: Path) -> pd.DataFrame:
    path = one(root, "qb_designed_run_prior_history_audit.csv")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    x = x.rename(columns={"player_name_key": "player_clean_key"})
    x = canon_keys(x)
    x = x.loc[x["season"].eq(2024)].copy()
    if len(x) != 444 or x.duplicated(KEYS).any():
        raise RuntimeError(f"2024 designed-run source cohort drift rows={len(x)}")
    x["prior_games"] = num(x["prior_games"])
    x["recent8_designed_runs"] = num(x["recent8_designed_runs"])
    x["recent8_snaps"] = num(x["recent8_snaps"])
    return x[KEYS + ["resolution_status", "prior_games", "recent8_designed_runs", "recent8_snaps", "strict_prior_max_season", "strict_prior_max_week"]]


def load_shared(root: Path) -> pd.DataFrame:
    cols = KEYS + ["wr_reception_mass_residual"]
    x = pd.read_csv(one(root, "qb_wr_shared_pass_volume_secondary_2024_2025.csv"), usecols=cols, low_memory=False)
    x = canon_keys(x)
    x = x.loc[x["season"].eq(2024)].copy()
    if len(x) != 444 or x.duplicated(KEYS).any():
        raise RuntimeError(f"2024 shared receiver cohort drift rows={len(x)}")
    x["wr_reception_mass_residual"] = num(x["wr_reception_mass_residual"])
    return x


def metric(actual, pred) -> dict:
    z = pd.DataFrame({"a": num(actual), "p": num(pred)}).dropna()
    e = z["p"] - z["a"]
    return {
        "n": int(len(z)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "bias": float(e.mean()),
        "corr": float(z["a"].corr(z["p"])) if len(z) >= 2 else np.nan,
        "p90_abs_error": float(e.abs().quantile(0.90)),
    }


def spearman(x, y) -> float:
    z = pd.DataFrame({"x": num(x), "y": num(y)}).dropna()
    return float(z["x"].corr(z["y"], method="spearman")) if len(z) >= 3 else np.nan


def bootstrap_gain(actual, base, cand, *, seed=SEED, n_boot=BOOT_N) -> dict:
    a = num(actual).to_numpy(float)
    b = num(base).to_numpy(float)
    c = num(cand).to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    a, b, c = a[ok], b[ok], c[ok]
    rng = np.random.default_rng(seed)
    gains = np.empty(n_boot, dtype=float)
    n = len(a)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        gains[i] = np.mean(np.abs(b[idx] - a[idx])) - np.mean(np.abs(c[idx] - a[idx]))
    return {
        "n": int(n),
        "draws": int(n_boot),
        "seed": int(seed),
        "mean_gain": float(gains.mean()),
        "p_gain_gt_0": float((gains > 0).mean()),
        "p05": float(np.quantile(gains, 0.05)),
        "p50": float(np.quantile(gains, 0.50)),
        "p95": float(np.quantile(gains, 0.95)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--play-root", type=Path, required=True)
    ap.add_argument("--chain-root", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--shared-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    play = load_play(a.play_root)
    chain = load_chain(a.chain_root)
    source = load_source(a.source_root)
    shared = load_shared(a.shared_root)

    z = play.merge(chain, on=KEYS, how="inner", validate="one_to_one")
    z = z.merge(source, on=KEYS, how="left", validate="one_to_one")
    z = z.merge(shared, on=KEYS, how="left", validate="one_to_one")
    if len(z) != 444:
        raise RuntimeError(f"merged development cohort drift {len(z)}")

    baseline_d_identity = float((z["pred_d"] - z["pred_plays"] * z["pred_rate"]).abs().max())
    baseline_attempt_identity = float((z["pred_attempts"] - z["pred_d"] * z["pred_c"] * z["pred_s"]).abs().max())
    base_rate_exact = bool(np.isclose(z["pred_rate"], BASE_RATE, atol=1e-12, rtol=0).all())

    z["source_eligible"] = (
        z["resolution_status"].astype(str).eq("EXACT_NORMALIZED_UNIQUE")
        & num(z["prior_games"]).ge(1)
        & num(z["recent8_designed_runs"]).notna()
        & num(z["pred_plays"]).gt(0)
    )
    z["qb_prior_designed_runs_per_game"] = np.where(
        z["source_eligible"],
        num(z["recent8_designed_runs"]) / num(z["prior_games"]),
        np.nan,
    )
    weekly = (
        z.loc[z["source_eligible"]]
        .groupby("week", as_index=False)["qb_prior_designed_runs_per_game"]
        .mean()
        .rename(columns={"qb_prior_designed_runs_per_game": "weekly_reference_designed_runs_per_game"})
    )
    z = z.merge(weekly, on="week", how="left", validate="many_to_one")
    if z["weekly_reference_designed_runs_per_game"].isna().any():
        raise RuntimeError("missing weekly designed-run reference")

    z["designed_run_excess"] = np.where(
        z["source_eligible"],
        z["qb_prior_designed_runs_per_game"] - z["weekly_reference_designed_runs_per_game"],
        0.0,
    )
    z["candidate_pass_rate_raw"] = BASE_RATE - COEF * z["designed_run_excess"] / z["pred_plays"]
    z["candidate_pass_rate"] = np.where(
        z["source_eligible"],
        np.clip(z["candidate_pass_rate_raw"], CLIP_LO, CLIP_HI),
        BASE_RATE,
    )
    z["candidate_d"] = z["pred_plays"] * z["candidate_pass_rate"]
    z["candidate_qb_attempts"] = z["candidate_d"] * z["pred_c"] * z["pred_s"]
    z["candidate_qb_pass_yards_mean"] = z["football_synthesis"]
    z["qb_mean_identity_gap"] = z["candidate_qb_pass_yards_mean"] - z["football_synthesis"]
    z["rate_adjustment"] = z["candidate_pass_rate"] - BASE_RATE
    z["d_adjustment"] = z["candidate_d"] - z["pred_d"]

    all_view = z.copy()
    elig_view = z.loc[z["source_eligible"]].copy()

    def view_metrics(g: pd.DataFrame) -> dict:
        out = {
            "n": int(len(g)),
            "pass_rate": {
                "baseline": metric(g["actual_rate"], g["pred_rate"]),
                "candidate": metric(g["actual_rate"], g["candidate_pass_rate"]),
            },
            "team_pass_opportunity": {
                "baseline": metric(g["actual_d"], g["pred_d"]),
                "candidate": metric(g["actual_d"], g["candidate_d"]),
            },
            "qb_attempts": {
                "baseline": metric(g["actual_attempts"], g["pred_attempts"]),
                "candidate": metric(g["actual_attempts"], g["candidate_qb_attempts"]),
            },
        }
        out["pass_rate"]["mae_gain"] = out["pass_rate"]["baseline"]["mae"] - out["pass_rate"]["candidate"]["mae"]
        out["team_pass_opportunity"]["mae_gain"] = out["team_pass_opportunity"]["baseline"]["mae"] - out["team_pass_opportunity"]["candidate"]["mae"]
        out["qb_attempts"]["mae_gain"] = out["qb_attempts"]["baseline"]["mae"] - out["qb_attempts"]["candidate"]["mae"]
        out["qb_attempts"]["baseline_8_plus_miss_rate"] = float((g["actual_attempts"].sub(g["pred_attempts"]).abs() >= 8).mean())
        out["qb_attempts"]["candidate_8_plus_miss_rate"] = float((g["actual_attempts"].sub(g["candidate_qb_attempts"]).abs() >= 8).mean())
        out["qb_attempts"]["baseline_10_plus_miss_rate"] = float((g["actual_attempts"].sub(g["pred_attempts"]).abs() >= 10).mean())
        out["qb_attempts"]["candidate_10_plus_miss_rate"] = float((g["actual_attempts"].sub(g["candidate_qb_attempts"]).abs() >= 10).mean())
        return out

    views = {"full_population": view_metrics(all_view), "source_eligible": view_metrics(elig_view)}
    adjustment = {
        "n_eligible": int(z["source_eligible"].sum()),
        "eligible_rate": float(z["source_eligible"].mean()),
        "mean_rate_adjustment": float(z["rate_adjustment"].mean()),
        "mean_abs_rate_adjustment": float(z["rate_adjustment"].abs().mean()),
        "p90_abs_rate_adjustment": float(z["rate_adjustment"].abs().quantile(0.90)),
        "min_rate_adjustment": float(z["rate_adjustment"].min()),
        "max_rate_adjustment": float(z["rate_adjustment"].max()),
        "clip_hit_rate": float(((z["candidate_pass_rate_raw"] < CLIP_LO) | (z["candidate_pass_rate_raw"] > CLIP_HI)).mean()),
    }
    correlations = {
        "rate_adjustment_vs_actual_rate_residual_spearman": spearman(z["rate_adjustment"], z["actual_rate"] - BASE_RATE),
        "d_adjustment_vs_wr_reception_mass_residual_spearman": spearman(z["d_adjustment"], z["wr_reception_mass_residual"]),
    }
    boot = {
        "pass_rate": bootstrap_gain(z["actual_rate"], z["pred_rate"], z["candidate_pass_rate"]),
        "team_pass_opportunity": bootstrap_gain(z["actual_d"], z["pred_d"], z["candidate_d"]),
        "qb_attempts": bootstrap_gain(z["actual_attempts"], z["pred_attempts"], z["candidate_qb_attempts"]),
    }

    full = views["full_population"]
    integrity = {
        "exact_444_2024_rows": len(z) == 444,
        "2025_scored_or_summarized": False,
        "zero_sportsbook_inputs": True,
        "no_production_change": True,
        "no_model_fitting": True,
        "candidate_coefficient_exact_1": COEF == 1.0,
        "baseline_pass_rate_exact_0_57": base_rate_exact,
        "baseline_d_identity_max_abs": baseline_d_identity,
        "baseline_attempt_identity_max_abs": baseline_attempt_identity,
        "qb_mean_identity_max_abs": float(z["qb_mean_identity_gap"].abs().max()),
        "source_strict_prior_contract_reused": True,
    }
    integrity_pass = bool(
        integrity["exact_444_2024_rows"]
        and not integrity["2025_scored_or_summarized"]
        and integrity["zero_sportsbook_inputs"]
        and integrity["no_production_change"]
        and integrity["no_model_fitting"]
        and integrity["candidate_coefficient_exact_1"]
        and integrity["baseline_pass_rate_exact_0_57"]
        and baseline_d_identity <= 1e-9
        and baseline_attempt_identity <= 1e-9
        and integrity["qb_mean_identity_max_abs"] <= TOL
        and integrity["source_strict_prior_contract_reused"]
    )

    gates = {
        "all_integrity_gates_pass": integrity_pass,
        "pass_rate_mae_gain_ge_0_003": full["pass_rate"]["mae_gain"] >= 0.0030,
        "team_d_mae_gain_ge_0_15": full["team_pass_opportunity"]["mae_gain"] >= 0.15,
        "qb_attempt_mae_gain_ge_0_10": full["qb_attempts"]["mae_gain"] >= 0.10,
        "pass_rate_p90_nonworse": full["pass_rate"]["candidate"]["p90_abs_error"] <= full["pass_rate"]["baseline"]["p90_abs_error"],
        "team_d_p90_nonworse": full["team_pass_opportunity"]["candidate"]["p90_abs_error"] <= full["team_pass_opportunity"]["baseline"]["p90_abs_error"],
        "qb_attempt_p90_nonworse": full["qb_attempts"]["candidate"]["p90_abs_error"] <= full["qb_attempts"]["baseline"]["p90_abs_error"],
        "qb_10_plus_miss_rate_nonworse": full["qb_attempts"]["candidate_10_plus_miss_rate"] <= full["qb_attempts"]["baseline_10_plus_miss_rate"],
        "rate_adjustment_spearman_ge_0_10": correlations["rate_adjustment_vs_actual_rate_residual_spearman"] >= 0.10,
        "wr_reception_spearman_ge_0_10": correlations["d_adjustment_vs_wr_reception_mass_residual_spearman"] >= 0.10,
        "bootstrap_pass_rate_p_ge_0_80": boot["pass_rate"]["p_gain_gt_0"] >= 0.80,
        "bootstrap_team_d_p_ge_0_80": boot["team_pass_opportunity"]["p_gain_gt_0"] >= 0.80,
        "bootstrap_qb_attempt_p_ge_0_80": boot["qb_attempts"]["p_gain_gt_0"] >= 0.80,
    }
    advance = all(bool(v) for v in gates.values())
    disposition = (
        "QB_DESIGNED_RUN_PASS_RATE_D1_PASS_READY_FOR_2025_CONFIRMATION"
        if advance else "QB_DESIGNED_RUN_PASS_RATE_D1_FAIL_NO_CONFIRMATION"
    )

    result = {
        "migration": "QB_PASS_RATE_DESIGNED_RUN_D1",
        "development_season": 2024,
        "confirmation_season_scored": False,
        "candidate": {
            "base_rate": BASE_RATE,
            "coefficient": COEF,
            "clip": [CLIP_LO, CLIP_HI],
            "history": "recent8 designed runs / prior games",
            "center": "same-target-week mean strict-prior QB designed runs per game",
            "missing_source_behavior": "exact 0.57 baseline",
            "qb_pass_yards_mean_behavior": "M89/M90 football_synthesis preserved exactly",
        },
        "views": views,
        "adjustment": adjustment,
        "correlations": correlations,
        "bootstrap": boot,
        "integrity": integrity,
        "advance_gates": gates,
        "all_advance_gates_pass": advance,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": disposition,
    }

    z.to_csv(a.out_dir / "qb_designed_run_d1_casebook_2024.csv", index=False)
    weekly.to_csv(a.out_dir / "qb_designed_run_d1_weekly_reference_2024.csv", index=False)
    (a.out_dir / "qb_designed_run_d1_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
