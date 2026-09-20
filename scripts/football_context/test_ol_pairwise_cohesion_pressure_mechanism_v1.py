#!/usr/bin/env python3
"""Frozen OL pairwise-cohesion -> pressure mechanism experiment V1.

Plan:
docs/research/OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1.md

2019-2023 fits one nested OLS pair. 2024 is the primary holdout. The separate
2025 team-week outcome file is not opened unless every frozen 2024 gate passes.
No player projection or production artifact is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.football_context import qualify_ol_roster_continuity_v1 as cont
from scripts.football_context import qualify_ol_roster_pairwise_cohesion_v1 as coh

TRAIN_SEASONS = set(range(2019, 2024))
PRIMARY_SEASON = 2024
REPLICATION_SEASON = 2025
MIN_EVAL_ROWS = 400
MIN_EVAL_COVERAGE = 0.80
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 92026

PRIOR_METRICS = list(cont.PRIOR_TEAM_METRICS)
BASE_NUMERIC = [
    "week",
    cont.CANDIDATE,
    *[f"prior_{m}" for m in PRIOR_METRICS],
]
CANDIDATE_NUMERIC = BASE_NUMERIC + [coh.CANDIDATE]


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _num(v):
    return pd.to_numeric(v, errors="coerce")


def normalize_team_weekly(df: pd.DataFrame) -> pd.DataFrame:
    x = _lower(df)
    required = {"season", "week", "team", "pressure_rate_allowed", *PRIOR_METRICS}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"team-weekly missing columns: {sorted(missing)}")
    x["season"] = _num(x["season"]).astype("Int64")
    x["week"] = _num(x["week"]).astype("Int64")
    x["team"] = x["team"].map(canon_team)
    x = x.dropna(subset=["season", "week"])
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    if x.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("team-weekly duplicate team-week rows")
    for m in {"pressure_rate_allowed", *PRIOR_METRICS}:
        x[m] = _num(x[m])
    return x.sort_values(["team", "season", "week"]).reset_index(drop=True)


def attach_crossseason_prior_team_state(
    schedule: pd.DataFrame,
    team_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach target outcome and strictly previous scheduled-game team state."""
    sched = schedule.sort_values(["team", "season", "week"]).copy()
    tw = normalize_team_weekly(team_weekly)

    target = tw[["season", "week", "team", "pressure_rate_allowed"]].rename(
        columns={"pressure_rate_allowed": "target_pressure_rate_allowed"}
    )
    out = sched.merge(
        target,
        on=["season", "week", "team"],
        how="left",
        validate="one_to_one",
    )

    prior_lookup = {
        (int(r.season), int(r.week), str(r.team)): r
        for r in tw.itertuples(index=False)
    }
    rows = []
    chronology_violations = 0
    for team, g in sched.groupby("team", sort=True):
        ordered = g.sort_values(["season", "week"])
        prev_key = None
        for r in ordered.itertuples(index=False):
            rec = {"season": int(r.season), "week": int(r.week), "team": str(team)}
            if prev_key is None:
                for m in PRIOR_METRICS:
                    rec[f"prior_{m}"] = np.nan
            else:
                ps, pw, pt = prev_key
                if (ps > int(r.season)) or (ps == int(r.season) and pw >= int(r.week)):
                    chronology_violations += 1
                prev = prior_lookup.get(prev_key)
                for m in PRIOR_METRICS:
                    rec[f"prior_{m}"] = getattr(prev, m) if prev is not None else np.nan
            rows.append(rec)
            prev_key = (int(r.season), int(r.week), str(team))

    prior_df = pd.DataFrame(rows)
    before = len(out)
    out = out.merge(
        prior_df,
        on=["season", "week", "team"],
        how="left",
        validate="one_to_one",
    )
    return out, {
        "prior_team_state_chronology_violations": int(chronology_violations),
        "target_prior_state_join_fanout": int(len(out) - before),
        "target_game_pbp_used_as_predictor": False,
    }


def build_feature_frame(
    schedule: pd.DataFrame,
    roster_sets: dict[tuple[int, int, str], set[str]],
    team_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    pair, pair_integrity = coh.materialize_pairwise_cohesion(schedule, roster_sets)
    immediate, immediate_integrity = cont.materialize_continuity(schedule, roster_sets)
    team_frame, team_integrity = attach_crossseason_prior_team_state(schedule, team_weekly)

    base = pair.merge(
        immediate[["season", "week", "team", cont.CANDIDATE]],
        on=["season", "week", "team"],
        how="left",
        validate="one_to_one",
    )
    before = len(base)
    out = base.merge(
        team_frame,
        on=["season", "week", "team"],
        how="left",
        validate="one_to_one",
    )
    integrity = {
        **pair_integrity,
        "immediate_continuity_chronology_violations": int(immediate_integrity["chronology_violations"]),
        "feature_join_fanout": int(len(out) - before),
        **team_integrity,
    }
    return out, integrity


def _fit_encoder(train: pd.DataFrame, numeric: list[str]) -> dict:
    medians = {}
    for c in numeric:
        v = _num(train[c])
        medians[c] = float(v.median()) if v.notna().any() else 0.0
    teams = sorted(train["team"].fillna("").astype(str).unique().tolist())
    # Intercept is explicit; omit one reference team to keep coefficients identified.
    team_dummies = teams[1:] if teams else []
    return {"numeric": numeric, "medians": medians, "team_dummies": team_dummies}


def _matrix(df: pd.DataFrame, enc: dict) -> np.ndarray:
    pieces = [np.ones((len(df), 1), dtype=float)]
    nums = []
    for c in enc["numeric"]:
        vals = _num(df[c]).fillna(enc["medians"][c]).to_numpy(dtype=float)
        nums.append(vals[:, None])
    if nums:
        pieces.append(np.hstack(nums))
    team = df["team"].fillna("").astype(str)
    for t in enc["team_dummies"]:
        pieces.append(team.eq(t).astype(float).to_numpy()[:, None])
    return np.hstack(pieces)


def fit_nested_models(train: pd.DataFrame) -> dict:
    train = train.copy()
    train["target_pressure_rate_allowed"] = _num(train["target_pressure_rate_allowed"])
    train = train.dropna(subset=["target_pressure_rate_allowed", coh.CANDIDATE, "team"])
    if len(train) < 1000:
        raise RuntimeError(f"insufficient 2019-2023 train rows: {len(train)}")

    base_enc = _fit_encoder(train, BASE_NUMERIC)
    cand_enc = _fit_encoder(train, CANDIDATE_NUMERIC)
    y = train["target_pressure_rate_allowed"].to_numpy(dtype=float)
    Xb = _matrix(train, base_enc)
    Xc = _matrix(train, cand_enc)
    beta_b, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    beta_c, *_ = np.linalg.lstsq(Xc, y, rcond=None)

    # Candidate numeric columns begin immediately after the intercept.
    cohesion_index = 1 + CANDIDATE_NUMERIC.index(coh.CANDIDATE)
    cohesion_coef = float(beta_c[cohesion_index])
    return {
        "train_rows": int(len(train)),
        "base_enc": base_enc,
        "cand_enc": cand_enc,
        "beta_base": beta_b,
        "beta_candidate": beta_c,
        "cohesion_coefficient": cohesion_coef,
    }


def predict_models(fit: dict, df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z["baseline_pred_pressure"] = _matrix(z, fit["base_enc"]) @ fit["beta_base"]
    z["candidate_pred_pressure"] = _matrix(z, fit["cand_enc"]) @ fit["beta_candidate"]
    return z


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    abs_err = np.abs(err)
    corr = float(np.corrcoef(actual, pred)[0, 1]) if len(actual) > 2 and np.std(actual) > 0 and np.std(pred) > 0 else np.nan
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "p90_abs_error": float(np.quantile(abs_err, 0.90)),
        "corr": corr,
        "bias": float(err.mean()),
    }


def cluster_bootstrap_gain(scored: pd.DataFrame, seed: int) -> dict:
    teams = sorted(scored["team"].astype(str).unique().tolist())
    by_team = {
        t: scored.loc[scored["team"].astype(str).eq(t), "abs_error_gain"].to_numpy(dtype=float)
        for t in teams
    }
    if len(teams) < 2:
        return {"ci_low": np.nan, "ci_high": np.nan, "replicates": 0, "clusters": len(teams)}
    rng = np.random.default_rng(seed)
    sims = np.empty(BOOTSTRAP_N, dtype=float)
    for i in range(BOOTSTRAP_N):
        sampled = rng.choice(teams, size=len(teams), replace=True)
        vals = np.concatenate([by_team[t] for t in sampled])
        sims[i] = float(vals.mean())
    return {
        "ci_low": float(np.quantile(sims, 0.025)),
        "ci_high": float(np.quantile(sims, 0.975)),
        "replicates": BOOTSTRAP_N,
        "clusters": int(len(teams)),
    }


def score_season(
    frame: pd.DataFrame,
    fit: dict,
    season: int,
    scheduled_rows: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    q = frame.loc[frame["season"].eq(season)].copy()
    q["target_pressure_rate_allowed"] = _num(q["target_pressure_rate_allowed"])
    q[coh.CANDIDATE] = _num(q[coh.CANDIDATE])
    score = q.dropna(subset=["target_pressure_rate_allowed", coh.CANDIDATE, "team"]).copy()
    coverage = float(len(score) / scheduled_rows) if scheduled_rows else 0.0
    if len(score):
        score = predict_models(fit, score)
        actual = score["target_pressure_rate_allowed"].to_numpy(dtype=float)
        base = _metrics(actual, score["baseline_pred_pressure"].to_numpy(dtype=float))
        cand = _metrics(actual, score["candidate_pred_pressure"].to_numpy(dtype=float))
        score["baseline_abs_error"] = np.abs(score["baseline_pred_pressure"] - score["target_pressure_rate_allowed"])
        score["candidate_abs_error"] = np.abs(score["candidate_pred_pressure"] - score["target_pressure_rate_allowed"])
        score["abs_error_gain"] = score["baseline_abs_error"] - score["candidate_abs_error"]
        boot = cluster_bootstrap_gain(score, seed)
    else:
        base = cand = {k: np.nan for k in ["mae", "rmse", "p90_abs_error", "corr", "bias"]}
        boot = {"ci_low": np.nan, "ci_high": np.nan, "replicates": 0, "clusters": 0}

    result = {
        "season": int(season),
        "scheduled_team_games": int(scheduled_rows),
        "scored_rows": int(len(score)),
        "scoring_coverage": coverage,
        "base_mae": base["mae"],
        "candidate_mae": cand["mae"],
        "mae_gain": base["mae"] - cand["mae"],
        "base_rmse": base["rmse"],
        "candidate_rmse": cand["rmse"],
        "rmse_gain": base["rmse"] - cand["rmse"],
        "base_p90_abs_error": base["p90_abs_error"],
        "candidate_p90_abs_error": cand["p90_abs_error"],
        "p90_abs_error_gain": base["p90_abs_error"] - cand["p90_abs_error"],
        "base_corr": base["corr"],
        "candidate_corr": cand["corr"],
        "corr_gain": cand["corr"] - base["corr"],
        "base_bias": base["bias"],
        "candidate_bias": cand["bias"],
        "bootstrap_ci_low": boot["ci_low"],
        "bootstrap_ci_high": boot["ci_high"],
        "bootstrap_replicates": int(boot["replicates"]),
        "bootstrap_team_clusters": int(boot["clusters"]),
    }
    return result, score


def primary_gate(result: dict, cohesion_coefficient: float) -> tuple[pd.DataFrame, bool]:
    checks = [
        ("support_rows", result["scored_rows"], f">={MIN_EVAL_ROWS}", result["scored_rows"] >= MIN_EVAL_ROWS),
        ("scoring_coverage", result["scoring_coverage"], f">={MIN_EVAL_COVERAGE}", result["scoring_coverage"] >= MIN_EVAL_COVERAGE),
        ("candidate_mae_lower", result["mae_gain"], ">0", result["mae_gain"] > 0),
        ("cluster_bootstrap_ci_low", result["bootstrap_ci_low"], ">0", np.isfinite(result["bootstrap_ci_low"]) and result["bootstrap_ci_low"] > 0),
        ("rmse_nonincrease", result["rmse_gain"], ">=0", result["rmse_gain"] >= 0),
        ("p90_nonincrease", result["p90_abs_error_gain"], ">=0", result["p90_abs_error_gain"] >= 0),
        ("cohesion_coefficient_direction", cohesion_coefficient, "<0", cohesion_coefficient < 0),
    ]
    gate = pd.DataFrame(checks, columns=["gate", "value", "threshold", "passed"])
    return gate, bool(gate["passed"].all())


def replication_gate(result: dict) -> tuple[pd.DataFrame, bool]:
    checks = [
        ("support_rows", result["scored_rows"], f">={MIN_EVAL_ROWS}", result["scored_rows"] >= MIN_EVAL_ROWS),
        ("scoring_coverage", result["scoring_coverage"], f">={MIN_EVAL_COVERAGE}", result["scoring_coverage"] >= MIN_EVAL_COVERAGE),
        ("candidate_mae_lower", result["mae_gain"], ">0", result["mae_gain"] > 0),
        ("cluster_bootstrap_ci_low", result["bootstrap_ci_low"], ">0", np.isfinite(result["bootstrap_ci_low"]) and result["bootstrap_ci_low"] > 0),
        ("rmse_nonincrease", result["rmse_gain"], ">=0", result["rmse_gain"] >= 0),
        ("p90_nonincrease", result["p90_abs_error_gain"], ">=0", result["p90_abs_error_gain"] >= 0),
    ]
    gate = pd.DataFrame(checks, columns=["gate", "value", "threshold", "passed"])
    return gate, bool(gate["passed"].all())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--team-weekly-primary", type=Path, required=True)
    ap.add_argument("--team-weekly-replication", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--git-sha", required=True)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    schedule = cont.normalize_schedule(pd.read_csv(args.schedule, low_memory=False))
    roster = pd.read_csv(args.roster, low_memory=False)
    roster_sets, identity = cont.build_roster_sets(roster, schedule)

    # The primary input intentionally contains only 2019-2024 team-week outcomes.
    primary_tw = pd.read_csv(args.team_weekly_primary, low_memory=False)
    primary_schedule = schedule.loc[schedule["season"].between(2019, PRIMARY_SEASON)].copy()
    primary_roster_sets = {
        k: v for k, v in roster_sets.items() if k[0] <= PRIMARY_SEASON
    }
    frame, integrity = build_feature_frame(primary_schedule, primary_roster_sets, primary_tw)

    train = frame.loc[frame["season"].isin(TRAIN_SEASONS)].copy()
    fit = fit_nested_models(train)

    n_sched_2024 = int(primary_schedule["season"].eq(PRIMARY_SEASON).sum())
    primary_result, primary_rows = score_season(
        frame, fit, PRIMARY_SEASON, n_sched_2024, BOOTSTRAP_SEED
    )
    p_gate, p_pass = primary_gate(primary_result, fit["cohesion_coefficient"])

    pd.DataFrame([primary_result]).to_csv(out / "ol_cohesion_pressure_primary_2024_v1.csv", index=False)
    p_gate.to_csv(out / "ol_cohesion_pressure_primary_gate_v1.csv", index=False)
    pd.DataFrame([{
        "train_rows": fit["train_rows"],
        "cohesion_coefficient_raw": fit["cohesion_coefficient"],
        "base_feature_count_including_week": len(BASE_NUMERIC),
        "candidate_feature_count_including_week": len(CANDIDATE_NUMERIC),
        "team_dummy_count": len(fit["cand_enc"]["team_dummies"]),
    }]).to_csv(out / "ol_cohesion_pressure_fit_summary_v1.csv", index=False)

    replication_exposed = False
    replication_result = None
    r_gate = pd.DataFrame()
    final = "OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_PRIMARY"

    if p_pass:
        # This is the first open/read of the separate 2025 team-week outcome file.
        replication_exposed = True
        rep_tw = pd.read_csv(args.team_weekly_replication, low_memory=False)
        full_tw = pd.concat([primary_tw, rep_tw], ignore_index=True, sort=False)
        full_frame, rep_integrity = build_feature_frame(schedule, roster_sets, full_tw)
        for k, v in rep_integrity.items():
            integrity[f"replication_{k}"] = v
        n_sched_2025 = int(schedule["season"].eq(REPLICATION_SEASON).sum())
        replication_result, replication_rows = score_season(
            full_frame, fit, REPLICATION_SEASON, n_sched_2025, BOOTSTRAP_SEED + 1
        )
        r_gate, r_pass = replication_gate(replication_result)
        pd.DataFrame([replication_result]).to_csv(out / "ol_cohesion_pressure_replication_2025_v1.csv", index=False)
        r_gate.to_csv(out / "ol_cohesion_pressure_replication_gate_v1.csv", index=False)
        if r_pass:
            final = "OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_VALIDATED"
        else:
            final = "OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_REPLICATION"

    manifest = {
        "experiment": "OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1",
        "frozen_plan": "docs/research/OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1.md",
        "git_sha": args.git_sha,
        "train_seasons": "2019-2023",
        "primary_season": PRIMARY_SEASON,
        "replication_season": REPLICATION_SEASON,
        "replication_exposed": replication_exposed,
        "cohesion_candidate": coh.CANDIDATE,
        "cohesion_lookback_games": coh.LOOKBACK_GAMES,
        "target": "pressure_rate_allowed",
        "target_semantics": "mean(sack_or_qb_hit) over offensive dropbacks",
        "identity_stable_id_coverage": float(identity["stable_id_coverage"]),
        "identity_ambiguous_same_week_gsis_team_conflicts": int(identity["ambiguous_same_week_gsis_team_conflicts"]),
        "integrity": integrity,
        "cohesion_coefficient_raw": float(fit["cohesion_coefficient"]),
        "primary_passed": p_pass,
        "final_disposition": final,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed_primary": BOOTSTRAP_SEED,
        "bootstrap_cluster": "team",
        "sportsbook_read": False,
        "production_changed": False,
        "issue_535_touched": False,
        "player_projection_changed": False,
        "target_game_pbp_used_as_predictor": False,
        "target_game_snap_or_participation_used": False,
        "schedule_sha256": _sha256(args.schedule),
        "weekly_roster_sha256": _sha256(args.roster),
        "team_weekly_primary_sha256": _sha256(args.team_weekly_primary),
        "team_weekly_replication_sha256": _sha256(args.team_weekly_replication) if replication_exposed else "NOT_READ_NOT_HASHED",
    }
    (out / "ol_cohesion_pressure_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("FIT")
    print(pd.DataFrame([{
        "train_rows": fit["train_rows"],
        "cohesion_coefficient_raw": fit["cohesion_coefficient"],
    }]).to_string(index=False))
    print("\n2024 PRIMARY")
    print(pd.DataFrame([primary_result]).to_string(index=False))
    print("\n2024 GATE")
    print(p_gate.to_string(index=False))
    print(f"\nPRIMARY_PASS={p_pass}")
    print(f"REPLICATION_EXPOSED={replication_exposed}")
    if replication_exposed:
        print("\n2025 REPLICATION")
        print(pd.DataFrame([replication_result]).to_string(index=False))
        print("\n2025 GATE")
        print(r_gate.to_string(index=False))
    print(f"\nFINAL_DISPOSITION={final}")
    print("\nMANIFEST")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
