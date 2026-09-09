#!/usr/bin/env python3
"""RB-R21 Phase B: grade sealed Week-1 CONTROL vs SHADOW distributions.

The scoring contract was frozen before 2026 outcomes. This grader consumes only the
sealed R21 draw matrices for forecasts; it never regenerates Week-1 predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nflv

CANDIDATE = "RB_R21_WEEK1_PROSPECTIVE_TAIL_GRADE_V1"
SEASON = 2026
WEEK = 1
EXPECTED_ROWS = 94
EXPECTED_DRAWS = 10000
EXPECTED_LOCK = {
    "run_id": 34294227588,
    "artifact_id": 10082525892,
    "name": "rb-r21-week1-prospective-forecast-lock-v1",
    "digest": "sha256:302df98f83c461d8abd74da9985dacc80c9477596b8bba85cc4d4a27c4fbc6f3",
    "head_sha": "eefa0a4fc0b272de03db804acc4e77d7cbe39b92",
}
EXPECTED_CONTROL_HASH = "0149d2bc2fcd4cd9f401b1a46c1c27ecc9489c3ce00c40ec8f33880bb8b5181e"
EXPECTED_SHADOW_HASH = "6be1c67952f465ffb6d01bf3bbdbbdb0ea3e5758684e10c2f075715f58b582d9"
EXPECTED_LEDGER_HASH = "b6f528217f1f14057730f52a64174f59497354256a5037ccbf4a43c8bd4a980a"

TEAM_ALIASES = {
    "LA": "LAR", "STL": "LAR",
    "JAX": "JAC",
    "WSH": "WAS",
    "OAK": "LV",
    "SD": "LAC",
}


def to_pandas(frame) -> pd.DataFrame:
    return frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame)


def norm_team(value: object) -> str:
    s = str(value or "").upper().strip()
    return TEAM_ALIASES.get(s, s)


def norm_player(value: object) -> str:
    s = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", s.lower())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_f64(arr: np.ndarray) -> str:
    x = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    return hashlib.sha256(x.tobytes(order="C")).hexdigest()


def artifact_record(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arts = raw.get("artifacts", [])
    matches = [a for a in arts if a.get("name") == EXPECTED_LOCK["name"]]
    if len(matches) != 1:
        return {"pass": False, "reason": f"expected one lock artifact, got {len(matches)}"}
    a = matches[0]
    wr = a.get("workflow_run") or {}
    checks = {
        "id": int(a.get("id", -1)) == EXPECTED_LOCK["artifact_id"],
        "name": a.get("name") == EXPECTED_LOCK["name"],
        "digest": a.get("digest") == EXPECTED_LOCK["digest"],
        "run_id": int(wr.get("id", -1)) == EXPECTED_LOCK["run_id"],
        "head_sha": wr.get("head_sha") == EXPECTED_LOCK["head_sha"],
        "not_expired": a.get("expired") is False,
    }
    return {"pass": bool(all(checks.values())), "checks": checks, "observed": a}


def crps_empirical(draws: np.ndarray, y: float) -> float:
    x = np.sort(np.asarray(draws, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    weights = 2.0 * np.arange(1, n + 1, dtype=float) - n - 1.0
    half_pairwise = float(np.sum(weights * x) / (n * n))
    return float(np.mean(np.abs(x - float(y))) - half_pairwise)


def pinball(y: float, qhat: float, tau: float) -> float:
    err = float(y) - float(qhat)
    return float(tau * err if err >= 0 else (tau - 1.0) * err)


def score_distribution(draws: np.ndarray, actual: np.ndarray, means: np.ndarray) -> tuple[dict, pd.DataFrame]:
    n = draws.shape[0]
    rows = []
    crps_vals = []
    q90_loss = []
    q95_loss = []
    b30 = []
    b50 = []
    cov80 = []
    cov90 = []
    for i in range(n):
        x = np.asarray(draws[i], dtype=float)
        y = float(actual[i])
        mu = float(means[i])
        q05, q10, q90, q95 = np.quantile(x, [0.05, 0.10, 0.90, 0.95])
        p30 = float(np.mean(x >= mu + 30.0))
        p50 = float(np.mean(x >= mu + 50.0))
        e30 = float(y >= mu + 30.0)
        e50 = float(y >= mu + 50.0)
        c = crps_empirical(x, y)
        l90 = pinball(y, q90, 0.90)
        l95 = pinball(y, q95, 0.95)
        s30 = (p30 - e30) ** 2
        s50 = (p50 - e50) ** 2
        in80 = float(q10 <= y <= q90)
        in90 = float(q05 <= y <= q95)
        crps_vals.append(c); q90_loss.append(l90); q95_loss.append(l95)
        b30.append(s30); b50.append(s50); cov80.append(in80); cov90.append(in90)
        rows.append({
            "crps": c, "brier30": s30, "brier50": s50,
            "q90_pinball": l90, "q95_pinball": l95,
            "cover80": in80, "cover90": in90,
            "prob_mu_plus_30": p30, "prob_mu_plus_50": p50,
            "event_mu_plus_30": e30, "event_mu_plus_50": e50,
            "q05": q05, "q10": q10, "q90": q90, "q95": q95,
        })
    metrics = {
        "n": int(n),
        "crps": float(np.mean(crps_vals)),
        "brier30": float(np.mean(b30)),
        "brier50": float(np.mean(b50)),
        "q90_pinball": float(np.mean(q90_loss)),
        "q95_pinball": float(np.mean(q95_loss)),
        "coverage80": float(np.mean(cov80)),
        "coverage90": float(np.mean(cov90)),
        "coverage80_abs_error": abs(float(np.mean(cov80)) - 0.80),
        "coverage90_abs_error": abs(float(np.mean(cov90)) - 0.90),
        "event30_count": int(np.sum(np.asarray(actual) >= np.asarray(means) + 30.0)),
        "event50_count": int(np.sum(np.asarray(actual) >= np.asarray(means) + 50.0)),
    }
    return metrics, pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-dir", type=Path, required=True)
    ap.add_argument("--lock-artifact-metadata", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    lock_meta = artifact_record(a.lock_artifact_metadata)
    result_path = a.lock_dir / "rb_r21_forecast_lock_result.json"
    ledger_path = a.lock_dir / "rb_r21_forecast_ledger.csv"
    draws_path = a.lock_dir / "rb_r21_locked_rec_yards_draws_v1.npz"
    for p in [result_path, ledger_path, draws_path]:
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f"missing sealed lock file: {p}")

    lock_result = json.loads(result_path.read_text(encoding="utf-8"))
    ledger = pd.read_csv(ledger_path, low_memory=False)
    with np.load(draws_path) as z:
        control = np.asarray(z["control"], dtype="<f8")
        shadow = np.asarray(z["shadow"], dtype="<f8")

    lock_integrity = bool(
        lock_result.get("pass") is True
        and lock_result.get("disposition") == "RB_R21_WEEK1_PROSPECTIVE_FORECAST_LOCK_PASS_SHADOW_ONLY"
        and control.shape == (EXPECTED_ROWS, EXPECTED_DRAWS)
        and shadow.shape == control.shape
        and len(ledger) == EXPECTED_ROWS
        and sha256_f64(control) == EXPECTED_CONTROL_HASH
        and sha256_f64(shadow) == EXPECTED_SHADOW_HASH
        and sha256_file(ledger_path) == EXPECTED_LEDGER_HASH
    )

    schedules = to_pandas(nflv.load_schedules(seasons=[SEASON]))
    wk_sched = schedules.loc[
        pd.to_numeric(schedules.get("season"), errors="coerce").eq(SEASON)
        & pd.to_numeric(schedules.get("week"), errors="coerce").eq(WEEK)
        & schedules.get("game_type", "").astype(str).str.upper().eq("REG")
    ].copy()
    schedule_cols = [c for c in ["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "away_score", "home_team", "home_score"] if c in wk_sched.columns]
    wk_sched = wk_sched[schedule_cols].copy()
    games_final = bool(
        len(wk_sched) == 16
        and "away_score" in wk_sched.columns and "home_score" in wk_sched.columns
        and wk_sched["away_score"].notna().all() and wk_sched["home_score"].notna().all()
    )

    player_stats = to_pandas(nflv.load_player_stats(seasons=[SEASON], summary_level="week"))
    stats = player_stats.loc[
        pd.to_numeric(player_stats.get("season"), errors="coerce").eq(SEASON)
        & pd.to_numeric(player_stats.get("week"), errors="coerce").eq(WEEK)
        & player_stats.get("season_type", "").astype(str).str.upper().eq("REG")
    ].copy()
    required_stats = {"team", "receiving_yards"}
    if not required_stats.issubset(stats.columns):
        raise RuntimeError(f"player stats missing required columns: {sorted(required_stats - set(stats.columns))}")
    name_col = "player_display_name" if "player_display_name" in stats.columns else "player_name"
    if name_col not in stats.columns:
        raise RuntimeError("player stats missing player name column")
    stats["team_norm"] = stats["team"].map(norm_team)
    stats["player_clean_key"] = stats[name_col].map(norm_player)
    stats["receiving_yards"] = pd.to_numeric(stats["receiving_yards"], errors="coerce").fillna(0.0)

    snaps_raw = to_pandas(nflv.load_snap_counts(seasons=[SEASON]))
    snaps = snaps_raw.loc[
        pd.to_numeric(snaps_raw.get("season"), errors="coerce").eq(SEASON)
        & pd.to_numeric(snaps_raw.get("week"), errors="coerce").eq(WEEK)
        & snaps_raw.get("game_type", "").astype(str).str.upper().eq("REG")
    ].copy()
    for c in ["team", "player", "offense_snaps"]:
        if c not in snaps.columns:
            raise RuntimeError(f"snap counts missing required column: {c}")
    snaps["team_norm"] = snaps["team"].map(norm_team)
    snaps["player_clean_key"] = snaps["player"].map(norm_player)
    snaps["offense_snaps"] = pd.to_numeric(snaps["offense_snaps"], errors="coerce").fillna(0.0)

    stats_keep = [c for c in ["player_id", name_col, "position", "team", "opponent_team", "receiving_yards", "team_norm", "player_clean_key"] if c in stats.columns]
    snaps_keep = [c for c in ["game_id", "player", "pfr_player_id", "position", "team", "opponent", "offense_snaps", "team_norm", "player_clean_key"] if c in snaps.columns]
    stats_out = stats[stats_keep].copy()
    snaps_out = snaps[snaps_keep].copy()
    stats_out.to_csv(a.out_dir / "rb_r21_week1_player_stats_source.csv", index=False)
    snaps_out.to_csv(a.out_dir / "rb_r21_week1_snap_counts_source.csv", index=False)
    wk_sched.to_csv(a.out_dir / "rb_r21_week1_schedule_source.csv", index=False)

    duplicate_stats = stats_out.duplicated(["team_norm", "player_clean_key"], keep=False)
    duplicate_snaps = snaps_out.duplicated(["team_norm", "player_clean_key"], keep=False)

    audits = []
    actual = np.full(len(ledger), np.nan, dtype=float)
    for i, row in ledger.iterrows():
        team = norm_team(row["team"])
        pkey = str(row["player_clean_key"])
        sm = stats_out.loc[(stats_out.team_norm == team) & (stats_out.player_clean_key == pkey)]
        nm = snaps_out.loc[(snaps_out.team_norm == team) & (snaps_out.player_clean_key == pkey)]
        status = "unresolved"
        source = ""
        value = np.nan
        if len(sm) == 1:
            value = float(sm.iloc[0]["receiving_yards"])
            status = "resolved"
            source = "player_stats"
        elif len(sm) == 0 and len(nm) == 1:
            value = 0.0
            status = "resolved"
            source = "snap_count_zero_receiving"
        elif len(sm) > 1:
            status = "ambiguous_player_stats"
        elif len(nm) > 1:
            status = "ambiguous_snap_counts"
        actual[i] = value
        audits.append({
            "row_index": int(i), "event_id": str(row["event_id"]), "team": team,
            "player": str(row["player"]), "player_clean_key": pkey,
            "status": status, "outcome_source": source,
            "actual_rec_yards": value,
            "stats_matches": int(len(sm)), "snap_matches": int(len(nm)),
            "offense_snaps": float(nm.iloc[0]["offense_snaps"]) if len(nm) == 1 else np.nan,
        })
    match_audit = pd.DataFrame(audits)
    resolved = match_audit.status.eq("resolved").to_numpy()
    resolved_count = int(resolved.sum())
    coverage = resolved_count / len(ledger) if len(ledger) else 0.0
    outcome_coverage_valid = bool(coverage >= 0.90 and resolved_count > 0)

    eval_ledger = ledger.loc[resolved].reset_index(drop=True).copy()
    y = actual[resolved]
    control_eval = control[resolved]
    shadow_eval = shadow[resolved]
    means = eval_ledger["frozen_mean"].to_numpy(float)

    control_metrics, control_rows = score_distribution(control_eval, y, means)
    shadow_metrics, shadow_rows = score_distribution(shadow_eval, y, means)
    casebook = eval_ledger.copy()
    casebook["actual_rec_yards"] = y
    for c in control_rows.columns:
        casebook[f"control_{c}"] = control_rows[c].to_numpy()
        casebook[f"shadow_{c}"] = shadow_rows[c].to_numpy()
    casebook["actual_minus_frozen_mean"] = y - means
    casebook["control_crps_advantage"] = casebook["control_crps"] - casebook["shadow_crps"]

    gates = {
        "lock_artifact_exact": bool(lock_meta.get("pass")),
        "lock_integrity_exact": lock_integrity,
        "week1_games_final": games_final,
        "outcome_matching_coverage_valid": outcome_coverage_valid,
        "source_keys_unambiguous": bool(not duplicate_stats.any() and not duplicate_snaps.any()),
        "shadow_crps_nonworse": shadow_metrics["crps"] <= control_metrics["crps"],
        "shadow_brier30_nonworse": shadow_metrics["brier30"] <= control_metrics["brier30"],
        "shadow_brier50_nonworse": shadow_metrics["brier50"] <= control_metrics["brier50"],
        "shadow_q90_better": shadow_metrics["q90_pinball"] < control_metrics["q90_pinball"],
        "shadow_q95_better": shadow_metrics["q95_pinball"] < control_metrics["q95_pinball"],
        "coverage80_guard": shadow_metrics["coverage80_abs_error"] <= control_metrics["coverage80_abs_error"] + 0.03,
        "coverage90_guard": shadow_metrics["coverage90_abs_error"] <= control_metrics["coverage90_abs_error"] + 0.03,
        "sportsbook_zero_upstream": True,
        "production_parameters_zero": True,
    }
    passed = bool(all(gates.values()))

    match_audit.to_csv(a.out_dir / "rb_r21_week1_outcome_match_audit.csv", index=False)
    casebook.to_csv(a.out_dir / "rb_r21_week1_prospective_casebook.csv", index=False)
    metrics_df = pd.DataFrame([
        {"distribution": "CONTROL", **control_metrics},
        {"distribution": "SHADOW", **shadow_metrics},
    ])
    metrics_df.to_csv(a.out_dir / "rb_r21_week1_metrics.csv", index=False)

    result = {
        "candidate": CANDIDATE,
        "disposition": "RB_R21_WEEK1_PROSPECTIVE_GRADE_PASS_CONTINUE_SHADOW" if passed else "RB_R21_WEEK1_PROSPECTIVE_GRADE_FAIL_DIAGNOSE",
        "pass": passed,
        "season": SEASON,
        "week": WEEK,
        "lock": EXPECTED_LOCK,
        "resolved_rows": resolved_count,
        "locked_rows": int(len(ledger)),
        "outcome_coverage": coverage,
        "control": control_metrics,
        "shadow": shadow_metrics,
        "delta_shadow_minus_control": {k: float(shadow_metrics[k] - control_metrics[k]) for k in ["crps", "brier30", "brier50", "q90_pinball", "q95_pinball", "coverage80", "coverage90"]},
        "gates": gates,
        "source_hashes": {
            "player_stats": sha256_file(a.out_dir / "rb_r21_week1_player_stats_source.csv"),
            "snap_counts": sha256_file(a.out_dir / "rb_r21_week1_snap_counts_source.csv"),
            "schedule": sha256_file(a.out_dir / "rb_r21_week1_schedule_source.csv"),
        },
        "sportsbook_inputs_added": 0,
        "production_parameters_changed": 0,
        "governance_note": "Week-1 PASS supports continuing prospective SHADOW evidence only. It does not authorize production; the frozen cumulative evidence floor and a separate governed promotion decision still apply.",
    }
    (a.out_dir / "rb_r21_week1_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
