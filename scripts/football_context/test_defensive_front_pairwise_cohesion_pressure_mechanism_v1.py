#!/usr/bin/env python3
"""Frozen Defensive Front Pairwise Cohesion -> Pressure Generation V1.

Plan:
docs/research/DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1.md

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
from scripts.football_context import qualify_ol_roster_continuity_v1 as ol
from scripts.football_context import qualify_defensive_front_pairwise_cohesion_v1 as coh

TRAIN_SEASONS = set(range(2019, 2024))
PRIMARY_SEASON = 2024
REPLICATION_SEASON = 2025
MIN_EVAL_ROWS = 400
MIN_EVAL_COVERAGE = 0.80
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 92027

EXPECTED_SCHEDULE_SHA256 = "60db4d57a7132b4f00d7f51996dab19b4d171e8e90393f3f95c8fa8b19b14f04"
EXPECTED_WEEKLY_ROSTER_SHA256 = "f2b791d47b146fe703a73d3111d609504779c7e9d2dc0ad47b3bd1996776f18a"

DEF_PRIOR_METRICS = [
    "pressure_rate_generated",
    "success_rate_def",
    "def_pass_epa",
    "explosive_play_rate_allowed",
]
OPP_PRIOR_METRICS = [
    "pressure_rate_allowed",
    "success_rate_off",
    "dropback_rate",
    "plays_est",
    "proe",
]
BASE_NUMERIC = [
    "week",
    coh.IMMEDIATE,
    *[f"prior_def_{m}" for m in DEF_PRIOR_METRICS],
    *[f"prior_opp_{m}" for m in OPP_PRIOR_METRICS],
]
CANDIDATE_NUMERIC = BASE_NUMERIC + [coh.CANDIDATE]


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _num(v):
    return pd.to_numeric(v, errors="coerce")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_schedule_with_opponent(df: pd.DataFrame) -> pd.DataFrame:
    x = _lower(df)
    required = {"season", "week", "team", "opponent"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"schedule missing columns: {sorted(missing)}")
    x["season"] = _num(x["season"])
    x["week"] = _num(x["week"])
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    x = x.loc[x["season"].isin(coh.SEASONS) & x["week"].gt(0)].copy()
    x = x[["season", "week", "team", "opponent"]].drop_duplicates()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    if x.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("schedule duplicate team-week rows")
    return x.sort_values(["season", "week", "team"]).reset_index(drop=True)


def normalize_team_weekly(df: pd.DataFrame) -> pd.DataFrame:
    x = _lower(df)
    required = {
        "season", "week", "team", "pressure_rate_generated",
        *DEF_PRIOR_METRICS, *OPP_PRIOR_METRICS,
    }
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
    for c in {"pressure_rate_generated", *DEF_PRIOR_METRICS, *OPP_PRIOR_METRICS}:
        x[c] = _num(x[c])
    return x.sort_values(["team", "season", "week"]).reset_index(drop=True)


def _is_strictly_prior(prev_key, current_key) -> bool:
    if prev_key is None:
        return True
    ps, pw, _ = prev_key
    cs, cw, _ = current_key
    return (ps < cs) or (ps == cs and pw < cw)


def attach_target_and_prior_states(
    schedule: pd.DataFrame,
    team_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    """Attach outcome plus strictly prior defense and opponent offense state."""
    sched = schedule.sort_values(["team", "season", "week"]).copy()
    tw = normalize_team_weekly(team_weekly)
    lookup = {
        (int(r.season), int(r.week), str(r.team)): r
        for r in tw.itertuples(index=False)
    }

    prev_key: dict[tuple[int, int, str], tuple[int, int, str] | None] = {}
    for team, g in sched.groupby("team", sort=True):
        previous = None
        for r in g.sort_values(["season", "week"]).itertuples(index=False):
            key = (int(r.season), int(r.week), str(team))
            prev_key[key] = previous
            previous = key

    rows = []
    def_chron = 0
    opp_chron = 0
    missing_opp_schedule_key = 0
    for r in sched.itertuples(index=False):
        key = (int(r.season), int(r.week), str(r.team))
        opp_current = (int(r.season), int(r.week), str(r.opponent))
        dprev = prev_key.get(key)
        if opp_current not in prev_key:
            missing_opp_schedule_key += 1
        oprev = prev_key.get(opp_current)

        if dprev is not None and not _is_strictly_prior(dprev, key):
            def_chron += 1
        if oprev is not None and not _is_strictly_prior(oprev, opp_current):
            opp_chron += 1

        target = lookup.get(key)
        drow = lookup.get(dprev) if dprev is not None else None
        orow = lookup.get(oprev) if oprev is not None else None
        rec = {
            "season": key[0],
            "week": key[1],
            "team": key[2],
            "opponent": str(r.opponent),
            "target_pressure_rate_generated": (
                getattr(target, "pressure_rate_generated") if target is not None else np.nan
            ),
        }
        for m in DEF_PRIOR_METRICS:
            rec[f"prior_def_{m}"] = getattr(drow, m) if drow is not None else np.nan
        for m in OPP_PRIOR_METRICS:
            rec[f"prior_opp_{m}"] = getattr(orow, m) if orow is not None else np.nan
        rows.append(rec)

    out = pd.DataFrame(rows)
    return out, {
        "prior_def_state_chronology_violations": int(def_chron),
        "prior_opponent_state_chronology_violations": int(opp_chron),
        "missing_opponent_schedule_keys": int(missing_opp_schedule_key),
        "target_game_pbp_used_as_predictor": False,
    }


def build_feature_frame(
    schedule: pd.DataFrame,
    front_sets: dict[tuple[int, int, str], set[str]],
    team_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    team_schedule = schedule[["season", "week", "team"]].drop_duplicates()
    cohesion, c_integrity = coh.materialize(team_schedule, front_sets)
    states, s_integrity = attach_target_and_prior_states(schedule, team_weekly)

    base = cohesion.merge(
        schedule[["season", "week", "team", "opponent"]],
        on=["season", "week", "team"],
        how="left",
        validate="one_to_one",
    )
    before = len(base)
    out = base.merge(
        states,
        on=["season", "week", "team", "opponent"],
        how="left",
        validate="one_to_one",
    )
    return out, {
        **c_integrity,
        **s_integrity,
        "feature_target_join_fanout": int(len(out) - before),
        "published_feature_duplicate_team_week_rows": int(
            out.duplicated(["season", "week", "team"], keep=False).sum()
        ),
    }


def _fit_encoder(train: pd.DataFrame, numeric: list[str]) -> dict:
    medians = {}
    for c in numeric:
        v = _num(train[c])
        medians[c] = float(v.median()) if v.notna().any() else 0.0
    teams = sorted(train["team"].fillna("").astype(str).unique().tolist())
    opponents = sorted(train["opponent"].fillna("").astype(str).unique().tolist())
    return {
        "numeric": numeric,
        "medians": medians,
        "team_dummies": teams[1:] if teams else [],
        "opponent_dummies": opponents[1:] if opponents else [],
    }


def _matrix(df: pd.DataFrame, enc: dict) -> np.ndarray:
    pieces = [np.ones((len(df), 1), dtype=float)]
    nums = []
    for c in enc["numeric"]:
        vals = _num(df[c]).fillna(enc["medians"][c]).to_numpy(dtype=float)
        nums.append(vals[:, None])
    if nums:
        pieces.append(np.hstack(nums))
    team = df["team"].fillna("").astype(str)
    opponent = df["opponent"].fillna("").astype(str)
    for t in enc["team_dummies"]:
        pieces.append(team.eq(t).astype(float).to_numpy()[:, None])
    for t in enc["opponent_dummies"]:
        pieces.append(opponent.eq(t).astype(float).to_numpy()[:, None])
    return np.hstack(pieces)


def fit_nested_models(train: pd.DataFrame) -> dict:
    z = train.copy()
    z["target_pressure_rate_generated"] = _num(z["target_pressure_rate_generated"])
    z[coh.CANDIDATE] = _num(z[coh.CANDIDATE])
    z = z.dropna(subset=["target_pressure_rate_generated", coh.CANDIDATE, "team", "opponent"])
    if len(z) < 1000:
        raise RuntimeError(f"insufficient 2019-2023 train rows: {len(z)}")

    base_enc = _fit_encoder(z, BASE_NUMERIC)
    cand_enc = _fit_encoder(z, CANDIDATE_NUMERIC)
    y = z["target_pressure_rate_generated"].to_numpy(dtype=float)
    Xb = _matrix(z, base_enc)
    Xc = _matrix(z, cand_enc)
    beta_b, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    beta_c, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    cohesion_index = 1 + CANDIDATE_NUMERIC.index(coh.CANDIDATE)
    return {
        "train_rows": int(len(z)),
        "base_enc": base_enc,
        "cand_enc": cand_enc,
        "beta_base": beta_b,
        "beta_candidate": beta_c,
        "cohesion_coefficient": float(beta_c[cohesion_index]),
    }


def predict_models(fit: dict, df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z["baseline_pred_pressure_generated"] = _matrix(z, fit["base_enc"]) @ fit["beta_base"]
    z["candidate_pred_pressure_generated"] = _matrix(z, fit["cand_enc"]) @ fit["beta_candidate"]
    return z


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    ae = np.abs(err)
    corr = (
        float(np.corrcoef(actual, pred)[0, 1])
        if len(actual) > 2 and np.std(actual) > 0 and np.std(pred) > 0
        else np.nan
    )
    return {
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "p90_abs_error": float(np.quantile(ae, 0.90)),
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
    q["target_pressure_rate_generated"] = _num(q["target_pressure_rate_generated"])
    q[coh.CANDIDATE] = _num(q[coh.CANDIDATE])
    score = q.dropna(
        subset=["target_pressure_rate_generated", coh.CANDIDATE, "team", "opponent"]
    ).copy()
    coverage = float(len(score) / scheduled_rows) if scheduled_rows else 0.0

    if len(score):
        score = predict_models(fit, score)
        actual = score["target_pressure_rate_generated"].to_numpy(dtype=float)
        base = _metrics(actual, score["baseline_pred_pressure_generated"].to_numpy(dtype=float))
        cand = _metrics(actual, score["candidate_pred_pressure_generated"].to_numpy(dtype=float))
        score["baseline_abs_error"] = np.abs(
            score["baseline_pred_pressure_generated"] - score["target_pressure_rate_generated"]
        )
        score["candidate_abs_error"] = np.abs(
            score["candidate_pred_pressure_generated"] - score["target_pressure_rate_generated"]
        )
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
        (
            "cluster_bootstrap_ci_low",
            result["bootstrap_ci_low"],
            ">0",
            np.isfinite(result["bootstrap_ci_low"]) and result["bootstrap_ci_low"] > 0,
        ),
        ("rmse_nonincrease", result["rmse_gain"], ">=0", result["rmse_gain"] >= 0),
        ("p90_nonincrease", result["p90_abs_error_gain"], ">=0", result["p90_abs_error_gain"] >= 0),
        ("cohesion_coefficient_direction", cohesion_coefficient, ">0", cohesion_coefficient > 0),
    ]
    gate = pd.DataFrame(checks, columns=["gate", "value", "threshold", "passed"])
    return gate, bool(gate["passed"].all())


def replication_gate(result: dict) -> tuple[pd.DataFrame, bool]:
    checks = [
        ("support_rows", result["scored_rows"], f">={MIN_EVAL_ROWS}", result["scored_rows"] >= MIN_EVAL_ROWS),
        ("scoring_coverage", result["scoring_coverage"], f">={MIN_EVAL_COVERAGE}", result["scoring_coverage"] >= MIN_EVAL_COVERAGE),
        ("candidate_mae_lower", result["mae_gain"], ">0", result["mae_gain"] > 0),
        (
            "cluster_bootstrap_ci_low",
            result["bootstrap_ci_low"],
            ">0",
            np.isfinite(result["bootstrap_ci_low"]) and result["bootstrap_ci_low"] > 0,
        ),
        ("rmse_nonincrease", result["rmse_gain"], ">=0", result["rmse_gain"] >= 0),
        ("p90_nonincrease", result["p90_abs_error_gain"], ">=0", result["p90_abs_error_gain"] >= 0),
    ]
    gate = pd.DataFrame(checks, columns=["gate", "value", "threshold", "passed"])
    return gate, bool(gate["passed"].all())


def _integrity_clean(identity: dict, integrity: dict) -> bool:
    return bool(
        identity["stable_id_coverage"] >= coh.MIN_STABLE_ID
        and identity["ambiguous_same_week_gsis_team_conflicts"] == 0
        and integrity["published_duplicate_team_week_rows"] == 0
        and integrity["chronology_violations"] == 0
        and integrity["schedule_join_fanout"] == 0
        and integrity["prior_def_state_chronology_violations"] == 0
        and integrity["prior_opponent_state_chronology_violations"] == 0
        and integrity["missing_opponent_schedule_keys"] == 0
        and integrity["feature_target_join_fanout"] == 0
        and integrity["published_feature_duplicate_team_week_rows"] == 0
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--team-weekly-primary", type=Path, required=True)
    ap.add_argument("--team-weekly-replication", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--git-sha", required=True)
    args = ap.parse_args()

    schedule_sha = _sha256(args.schedule)
    roster_sha = _sha256(args.roster)
    if schedule_sha != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError(f"frozen schedule SHA drift: {schedule_sha}")
    if roster_sha != EXPECTED_WEEKLY_ROSTER_SHA256:
        raise RuntimeError(f"frozen weekly-roster SHA drift: {roster_sha}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    schedule = normalize_schedule_with_opponent(pd.read_csv(args.schedule, low_memory=False))
    roster = pd.read_csv(args.roster, low_memory=False)
    team_schedule = ol.normalize_schedule(schedule[["season", "week", "team"]])
    front_sets, identity = coh.build_front_sets(roster, team_schedule)
    if identity["stable_id_coverage"] < coh.MIN_STABLE_ID:
        raise RuntimeError("stable-ID coverage failed before outcome exposure")
    if identity["ambiguous_same_week_gsis_team_conflicts"] != 0:
        raise RuntimeError("same-week GSIS/team ambiguity survived semantic quarantine")

    # Primary input contains only 2019-2024 target-game outcomes.
    primary_tw = pd.read_csv(args.team_weekly_primary, low_memory=False)
    primary_schedule = schedule.loc[schedule["season"].between(2019, PRIMARY_SEASON)].copy()
    primary_front_sets = {k: v for k, v in front_sets.items() if k[0] <= PRIMARY_SEASON}
    frame, integrity = build_feature_frame(primary_schedule, primary_front_sets, primary_tw)
    if not _integrity_clean(identity, integrity):
        raise RuntimeError(f"primary integrity gate failed: {integrity}")

    train = frame.loc[frame["season"].isin(TRAIN_SEASONS)].copy()
    fit = fit_nested_models(train)

    n_sched_2024 = int(primary_schedule["season"].eq(PRIMARY_SEASON).sum())
    primary_result, _ = score_season(
        frame, fit, PRIMARY_SEASON, n_sched_2024, BOOTSTRAP_SEED
    )
    p_gate, p_pass = primary_gate(primary_result, fit["cohesion_coefficient"])

    pd.DataFrame([primary_result]).to_csv(
        out / "def_front_cohesion_pressure_primary_2024_v1.csv", index=False
    )
    p_gate.to_csv(out / "def_front_cohesion_pressure_primary_gate_v1.csv", index=False)
    pd.DataFrame([{
        "train_rows": fit["train_rows"],
        "cohesion_coefficient_raw": fit["cohesion_coefficient"],
        "base_numeric_feature_count": len(BASE_NUMERIC),
        "candidate_numeric_feature_count": len(CANDIDATE_NUMERIC),
        "team_dummy_count": len(fit["cand_enc"]["team_dummies"]),
        "opponent_dummy_count": len(fit["cand_enc"]["opponent_dummies"]),
    }]).to_csv(out / "def_front_cohesion_pressure_fit_summary_v1.csv", index=False)

    replication_exposed = False
    replication_result = None
    r_gate = pd.DataFrame()
    final = "DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_PRIMARY"

    if p_pass:
        # First open/read of the physically separate 2025 outcome file.
        replication_exposed = True
        rep_tw = pd.read_csv(args.team_weekly_replication, low_memory=False)
        full_tw = pd.concat([primary_tw, rep_tw], ignore_index=True, sort=False)
        full_frame, rep_integrity = build_feature_frame(schedule, front_sets, full_tw)
        if not _integrity_clean(identity, rep_integrity):
            raise RuntimeError(f"replication integrity gate failed: {rep_integrity}")
        for k, v in rep_integrity.items():
            integrity[f"replication_{k}"] = v
        n_sched_2025 = int(schedule["season"].eq(REPLICATION_SEASON).sum())
        replication_result, _ = score_season(
            full_frame, fit, REPLICATION_SEASON, n_sched_2025, BOOTSTRAP_SEED + 1
        )
        r_gate, r_pass = replication_gate(replication_result)
        pd.DataFrame([replication_result]).to_csv(
            out / "def_front_cohesion_pressure_replication_2025_v1.csv", index=False
        )
        r_gate.to_csv(out / "def_front_cohesion_pressure_replication_gate_v1.csv", index=False)
        final = (
            "DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_VALIDATED"
            if r_pass
            else "DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_REPLICATION"
        )

    manifest = {
        "experiment": "DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1",
        "frozen_plan": "docs/research/DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1.md",
        "git_sha": args.git_sha,
        "train_seasons": "2019-2023",
        "primary_season": PRIMARY_SEASON,
        "replication_season": REPLICATION_SEASON,
        "replication_exposed": replication_exposed,
        "cohesion_candidate": coh.CANDIDATE,
        "cohesion_lookback_games": coh.LOOKBACK_GAMES,
        "target": "pressure_rate_generated",
        "target_semantics": "mean(sack_or_qb_hit) over opponent offensive dropbacks",
        "identity_stable_id_coverage": float(identity["stable_id_coverage"]),
        "semantic_gsis_collision_id_count": int(identity["semantic_gsis_collision_id_count"]),
        "semantic_gsis_collision_rows_quarantined": int(
            identity["semantic_gsis_collision_rows_quarantined"]
        ),
        "identity_ambiguous_same_week_gsis_team_conflicts": int(
            identity["ambiguous_same_week_gsis_team_conflicts"]
        ),
        "integrity": integrity,
        "cohesion_coefficient_raw": float(fit["cohesion_coefficient"]),
        "primary_passed": p_pass,
        "final_disposition": final,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed_primary": BOOTSTRAP_SEED,
        "bootstrap_cluster": "defense_team",
        "sportsbook_read": False,
        "production_changed": False,
        "issue_535_touched": False,
        "player_projection_changed": False,
        "target_game_pbp_used_as_predictor": False,
        "target_game_snap_or_participation_used": False,
        "schedule_sha256": schedule_sha,
        "weekly_roster_sha256": roster_sha,
        "team_weekly_primary_sha256": _sha256(args.team_weekly_primary),
        "team_weekly_replication_sha256": (
            _sha256(args.team_weekly_replication)
            if replication_exposed
            else "NOT_READ_NOT_HASHED"
        ),
    }
    (out / "def_front_cohesion_pressure_manifest_v1.json").write_text(
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
