#!/usr/bin/env python3
"""Frozen BDB2023 exact blocker-rusher assignment value-of-information evaluator.

Plan:
docs/research/BDB2023_EXACT_BLOCKER_RUSHER_ASSIGNMENT_VOI_EXPERIMENT_V1.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SOURCE_HASH = "1c3e1eb6fcd0cf85807c649804fa0c72421c76e973e40a742523f1dd3a929182"
EXPECTED_FILES = ["games.csv", "players.csv", "plays.csv", "pffScoutingData.csv"] + [
    f"week{i}.csv" for i in range(1, 9)
]
PRIOR_MIN = 10
GAME_MIN_EDGES = 20
GAME_MIN_FEATURE_COVERAGE = 0.80
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 92028
TRAIN_WEEKS = {5, 6}
HOLDOUT_WEEKS = {7, 8}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def corpus_hash(root: Path) -> str:
    present = {p.name: p for p in root.rglob("*") if p.is_file()}
    missing = [name for name in EXPECTED_FILES if name not in present]
    if missing:
        raise RuntimeError(f"BDB2023 source files missing: {missing}")
    rows = []
    for name in EXPECTED_FILES:
        rows.append({"name": name, "sha256": sha256(present[name])})
    rows = sorted(rows, key=lambda x: x["name"])
    material = "\n".join(f"{x['name']}:{x['sha256']}" for x in rows)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def norm_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    try:
        v = float(text)
        if np.isfinite(v) and v.is_integer():
            return str(int(v))
    except (TypeError, ValueError):
        pass
    return text


def boolish(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (known, positive) for PFF bool/0-1-like values."""
    s = series.copy()
    known = s.notna()
    text = s.astype(str).str.strip().str.lower()
    pos = text.isin({"1", "1.0", "true", "t", "yes", "y"})
    num = pd.to_numeric(s, errors="coerce")
    pos = pos | num.eq(1)
    return known, pos


def build_edge_panel(corpus_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    pff_path = next(corpus_dir.rglob("pffScoutingData.csv"))
    plays_path = next(corpus_dir.rglob("plays.csv"))
    games_path = next(corpus_dir.rglob("games.csv"))

    pff = pd.read_csv(pff_path, low_memory=False)
    required = {
        "gameId", "playId", "nflId", "pff_nflIdBlockedPlayer",
        "pff_hitAllowed", "pff_hurryAllowed", "pff_sackAllowed",
    }
    missing = required - set(pff.columns)
    if missing:
        raise RuntimeError(f"pffScoutingData missing frozen fields: {sorted(missing)}")

    edge = pff.loc[pff["pff_nflIdBlockedPlayer"].notna(), list(required)].copy()
    edge["blocker_nfl_id"] = edge["nflId"].map(norm_id)
    edge["defender_nfl_id"] = edge["pff_nflIdBlockedPlayer"].map(norm_id)
    edge = edge.loc[edge["blocker_nfl_id"].ne("") & edge["defender_nfl_id"].ne("")].copy()

    known_cols = []
    pos_cols = []
    for col in ["pff_hitAllowed", "pff_hurryAllowed", "pff_sackAllowed"]:
        known, pos = boolish(edge[col])
        known_cols.append(known)
        pos_cols.append(pos)
    known_any = known_cols[0] | known_cols[1] | known_cols[2]
    pressure = pos_cols[0] | pos_cols[1] | pos_cols[2]
    edge["pressure_outcome_known"] = known_any
    edge["pressure_allowed_edge"] = pressure.astype(float)
    edge = edge.loc[edge["pressure_outcome_known"]].copy()

    plays = pd.read_csv(plays_path, low_memory=False)
    if "possessionTeam" not in plays.columns:
        raise RuntimeError("plays.csv missing possessionTeam")
    play_team = plays[["gameId", "playId", "possessionTeam"]].drop_duplicates()
    if play_team.duplicated(["gameId", "playId"]).any():
        raise RuntimeError("plays.csv has ambiguous possessionTeam by play")

    games = pd.read_csv(games_path, low_memory=False)
    if "week" not in games.columns:
        raise RuntimeError("games.csv missing week")
    game_week = games[["gameId", "week"]].drop_duplicates()
    if game_week["gameId"].duplicated().any():
        raise RuntimeError("games.csv duplicate gameId")

    edge = edge.merge(play_team, on=["gameId", "playId"], how="left", validate="many_to_one")
    edge = edge.merge(game_week, on="gameId", how="left", validate="many_to_one")
    missing_team = int(edge["possessionTeam"].isna().sum())
    missing_week = int(edge["week"].isna().sum())
    if missing_team or missing_week:
        raise RuntimeError(
            f"edge context missing possessionTeam={missing_team} week={missing_week}"
        )
    edge["week"] = pd.to_numeric(edge["week"], errors="raise").astype(int)

    duplicate_edges = int(
        edge.duplicated(
            ["gameId", "playId", "blocker_nfl_id", "defender_nfl_id"], keep=False
        ).sum()
    )
    if duplicate_edges:
        # Source rows should represent one exact assignment relationship per blocker/play.
        # Do not average duplicate truth.
        raise RuntimeError(f"duplicate exact assignment outcome edges: {duplicate_edges}")

    audit = {
        "pff_rows": int(len(pff)),
        "scoreable_exact_assignment_edges": int(len(edge)),
        "missing_possession_team": missing_team,
        "missing_week": missing_week,
        "duplicate_exact_assignment_edges": duplicate_edges,
        "pressure_outcome_fields": [
            "pff_hitAllowed", "pff_hurryAllowed", "pff_sackAllowed"
        ],
        "pff_beatenByDefender_used": False,
    }
    return edge, audit


def add_strict_prior_rates(edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    parts = []
    prior = edges.iloc[0:0].copy()
    same_week_violations = 0
    for week in sorted(edges["week"].unique()):
        cur = edges.loc[edges["week"].eq(week)].copy()
        hist = prior.loc[prior["week"].lt(week)].copy()
        same_week_violations += int(hist["week"].ge(week).sum())

        b = hist.groupby("blocker_nfl_id")["pressure_allowed_edge"].agg(["sum", "count"])
        d = hist.groupby("defender_nfl_id")["pressure_allowed_edge"].agg(["sum", "count"])

        cur["prior_blocker_n"] = cur["blocker_nfl_id"].map(b["count"] if len(b) else pd.Series(dtype=float)).fillna(0).astype(int)
        cur["prior_defender_n"] = cur["defender_nfl_id"].map(d["count"] if len(d) else pd.Series(dtype=float)).fillna(0).astype(int)
        cur["blocker_prior_allow_rate"] = cur["blocker_nfl_id"].map(
            (b["sum"] / b["count"]) if len(b) else pd.Series(dtype=float)
        )
        cur["defender_prior_pressure_rate"] = cur["defender_nfl_id"].map(
            (d["sum"] / d["count"]) if len(d) else pd.Series(dtype=float)
        )
        cur["feature_scoreable"] = (
            cur["prior_blocker_n"].ge(PRIOR_MIN)
            & cur["prior_defender_n"].ge(PRIOR_MIN)
            & cur["blocker_prior_allow_rate"].notna()
            & cur["defender_prior_pressure_rate"].notna()
        )
        parts.append(cur)
        prior = pd.concat([prior, cur], ignore_index=True)
    return pd.concat(parts, ignore_index=True), {
        "same_or_future_week_history_violations": same_week_violations
    }


def aggregate_team_games(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diag = []
    key = ["week", "gameId", "possessionTeam"]
    for values, g in edges.groupby(key, sort=True):
        week, game_id, offense = values
        known_n = len(g)
        s = g.loc[g["feature_scoreable"]].copy()
        coverage = len(s) / known_n if known_n else 0.0
        eligible = (
            len(s) >= GAME_MIN_EDGES
            and coverage >= GAME_MIN_FEATURE_COVERAGE
        )
        diag.append({
            "week": int(week),
            "game_id": int(game_id),
            "possession_team": str(offense),
            "known_outcome_edges": int(known_n),
            "scoreable_feature_edges": int(len(s)),
            "feature_edge_coverage": float(coverage),
            "eligible": bool(eligible),
        })
        if not eligible:
            continue

        bmean = float(s["blocker_prior_allow_rate"].mean())
        dmean = float(s["defender_prior_pressure_rate"].mean())
        product_mean = float(
            (s["blocker_prior_allow_rate"] * s["defender_prior_pressure_rate"]).mean()
        )
        rows.append({
            "week": int(week),
            "game_id": int(game_id),
            "possession_team": str(offense),
            "target_pressure_allowed_edge_rate": float(s["pressure_allowed_edge"].mean()),
            "blocker_prior_allow_mean": bmean,
            "defender_prior_pressure_mean": dmean,
            "scoreable_assignment_edge_count": int(len(s)),
            "assignment_pairing_covariance": float(product_mean - bmean * dmean),
            "feature_edge_coverage": float(coverage),
        })
    return pd.DataFrame(rows), pd.DataFrame(diag)


def fit_ols(train: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    x = np.column_stack(
        [np.ones(len(train)), train[feature_cols].to_numpy(dtype=float)]
    )
    y = train["target_pressure_allowed_edge_rate"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def predict(frame: pd.DataFrame, feature_cols: list[str], beta: np.ndarray) -> np.ndarray:
    x = np.column_stack(
        [np.ones(len(frame)), frame[feature_cols].to_numpy(dtype=float)]
    )
    return x @ beta


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    ae = np.abs(err)
    if len(y) >= 2 and np.std(y) > 0 and np.std(pred) > 0:
        corr = float(np.corrcoef(y, pred)[0, 1])
    else:
        corr = float("nan")
    return {
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(err.mean()),
        "p90_ae": float(np.quantile(ae, 0.90)),
        "corr": corr,
    }


def cluster_bootstrap_gain(
    holdout: pd.DataFrame,
    base_pred: np.ndarray,
    cand_pred: np.ndarray,
) -> dict[str, float]:
    y = holdout["target_pressure_allowed_edge_rate"].to_numpy(float)
    b = np.abs(base_pred - y)
    c = np.abs(cand_pred - y)
    games = np.array(sorted(holdout["game_id"].unique()))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gains = np.empty(BOOTSTRAP_REPS, dtype=float)
    indices = {
        g: np.flatnonzero(holdout["game_id"].to_numpy() == g) for g in games
    }
    for i in range(BOOTSTRAP_REPS):
        sampled = rng.choice(games, size=len(games), replace=True)
        idx = np.concatenate([indices[g] for g in sampled])
        gains[i] = float(b[idx].mean() - c[idx].mean())
    lo, hi = np.quantile(gains, [0.025, 0.975])
    return {
        "reps": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "clusters": int(len(games)),
        "mean_gain": float(gains.mean()),
        "ci95_lower": float(lo),
        "ci95_upper": float(hi),
    }


def run(corpus_dir: Path, out_dir: Path, *, active_sha: str, frozen_plan_sha: str) -> dict[str, object]:
    observed_hash = corpus_hash(corpus_dir)
    if observed_hash != EXPECTED_SOURCE_HASH:
        raise RuntimeError(
            f"BDB2023 source hash mismatch {observed_hash} != {EXPECTED_SOURCE_HASH}"
        )

    edges, source_audit = build_edge_panel(corpus_dir)
    edges, chronology = add_strict_prior_rates(edges)
    team_games, coverage_diag = aggregate_team_games(edges)

    denominator_holdout = coverage_diag.loc[coverage_diag["week"].isin(HOLDOUT_WEEKS)]
    holdout_eligible = team_games.loc[team_games["week"].isin(HOLDOUT_WEEKS)].copy()
    train = team_games.loc[team_games["week"].isin(TRAIN_WEEKS)].copy()

    integrity_pass = (
        chronology["same_or_future_week_history_violations"] == 0
        and source_audit["duplicate_exact_assignment_edges"] == 0
        and observed_hash == EXPECTED_SOURCE_HASH
    )

    if len(train) < 30:
        raise RuntimeError(f"insufficient frozen training offense-team rows: {len(train)}")

    base_features = [
        "blocker_prior_allow_mean",
        "defender_prior_pressure_mean",
        "scoreable_assignment_edge_count",
        "week",
    ]
    cand_features = base_features + ["assignment_pairing_covariance"]
    b_beta = fit_ols(train, base_features)
    c_beta = fit_ols(train, cand_features)
    bp = predict(holdout_eligible, base_features, b_beta)
    cp = predict(holdout_eligible, cand_features, c_beta)
    y = holdout_eligible["target_pressure_allowed_edge_rate"].to_numpy(float)

    bm = metrics(y, bp)
    cm = metrics(y, cp)
    boot = cluster_bootstrap_gain(holdout_eligible, bp, cp)
    coverage = (
        len(holdout_eligible) / len(denominator_holdout)
        if len(denominator_holdout) else 0.0
    )
    covariance_coef = float(c_beta[-1])

    gates = {
        "integrity": bool(integrity_pass),
        "holdout_rows_ge_50": bool(len(holdout_eligible) >= 50),
        "holdout_coverage_ge_0_80": bool(coverage >= 0.80),
        "candidate_mae_lower": bool(cm["mae"] < bm["mae"]),
        "bootstrap_ci_lower_gt_0": bool(boot["ci95_lower"] > 0),
        "candidate_rmse_nonincrease": bool(cm["rmse"] <= bm["rmse"]),
        "candidate_p90_nonincrease": bool(cm["p90_ae"] <= bm["p90_ae"]),
        "covariance_coefficient_positive": bool(covariance_coef > 0),
    }
    if not integrity_pass:
        disposition = "BDB2023_EXACT_ASSIGNMENT_VOI_INTEGRITY_FAILURE"
    elif all(gates.values()):
        disposition = "BDB2023_EXACT_ASSIGNMENT_VOI_SIGNAL"
    else:
        disposition = "BDB2023_EXACT_ASSIGNMENT_VOI_NO_ACTIONABLE_SIGNAL_V1"

    report = {
        "experiment": "BDB2023_EXACT_BLOCKER_RUSHER_ASSIGNMENT_VOI_EXPERIMENT_V1",
        "frozen_plan_sha": frozen_plan_sha,
        "active_sha": active_sha,
        "source_hash": observed_hash,
        "source_audit": source_audit,
        "chronology_audit": chronology,
        "train_weeks": sorted(TRAIN_WEEKS),
        "holdout_weeks": sorted(HOLDOUT_WEEKS),
        "train_offense_team_rows": int(len(train)),
        "holdout_denominator_offense_team_rows": int(len(denominator_holdout)),
        "holdout_scored_offense_team_rows": int(len(holdout_eligible)),
        "holdout_coverage": float(coverage),
        "baseline_features": base_features,
        "candidate_additional_feature": "assignment_pairing_covariance",
        "baseline_coefficients": [float(x) for x in b_beta],
        "candidate_coefficients": [float(x) for x in c_beta],
        "assignment_pairing_covariance_coefficient": covariance_coef,
        "baseline_metrics": bm,
        "candidate_metrics": cm,
        "mae_gain_baseline_minus_candidate": float(bm["mae"] - cm["mae"]),
        "rmse_gain_baseline_minus_candidate": float(bm["rmse"] - cm["rmse"]),
        "p90_gain_baseline_minus_candidate": float(bm["p90_ae"] - cm["p90_ae"]),
        "bootstrap": boot,
        "gates": gates,
        "target_game_realized_assignment_used": True,
        "deployable_pregame_claim": False,
        "sportsbook_read": False,
        "production_changed": False,
        "issue_535_touched": False,
        "disposition": disposition,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_diag.to_csv(out_dir / "bdb2023_exact_assignment_voi_coverage_v1.csv", index=False)
    pd.DataFrame([
        {"model": "baseline", **bm},
        {"model": "candidate", **cm},
    ]).to_csv(out_dir / "bdb2023_exact_assignment_voi_metrics_v1.csv", index=False)
    pd.DataFrame([{"gate": k, "pass": v} for k, v in gates.items()]).to_csv(
        out_dir / "bdb2023_exact_assignment_voi_gates_v1.csv", index=False
    )
    (out_dir / "bdb2023_exact_assignment_voi_result_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--active-sha", required=True)
    ap.add_argument("--frozen-plan-sha", required=True)
    args = ap.parse_args()
    run(
        args.corpus_dir,
        args.out_dir,
        active_sha=args.active_sha,
        frozen_plan_sha=args.frozen_plan_sha,
    )


if __name__ == "__main__":
    main()
