#!/usr/bin/env python3
"""WR-QB Shared Tail Signal V1 — frozen evaluator.

Implements docs/research/WR_QB_SHARED_TAIL_SIGNAL_V1_PLAN.md and
WR_QB_SHARED_TAIL_SIGNAL_V1_COHORT_CLARIFICATION.md exactly, against the
preserved Joint Pass/Receiving Conservation V1 artifact (run 34081764151,
artifact 10004223287, digest sha256:753aa191...). No new modeling, no
sportsbook inputs, no target-game outcome as a feature -- outcome fields are
used only for scoring.

Question: does the pregame QB C2 distribution's upper-tail width
(qb_c2_upper90 = c2_p90 - c2_mean) carry incremental, leakage-safe signal
about a true model-WR1 receiving-yard right-tail miss, on top of M38
opportunity that is already fixed?

Inputs (from the preserved artifact, concatenated across 2024 and 2025):
- joint_v1_qb_distribution_trace.csv (season, week, event_id, team,
  c2_mean, c2_p10, c2_p90, c2_p95, ...)
- joint_v1_player_projection_trace.csv (season, week, event_id, team,
  player, player_clean_key, position, b0_target_probability, b0_rec_yards,
  c2_rec_yards, ...)
- joint_v1_actual_usage.csv (season, week, team, player_clean_key,
  rec_yards, ...) -- actual outcomes, scoring only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = (2024, 2025)
DEV_SEASON = 2024
HOLDOUT_SEASON = 2025
EXPECTED_QB_ROWS = {2024: 444, 2025: 440}
BOOT_N = 10_000
BOOT_SEED = 5601

GATES = {
    "primary_spearman_min": 0.08,
    "top_quartile_residual_gap_min": 5.0,
    "bootstrap_p_min": 0.90,
    "tail100_rate_ratio_min": 1.30,
    "cat_under40_rate_ratio_min": 1.20,
    "opportunity_conditional_min_quartiles": 3,
}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} missing/empty: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    if out.empty:
        raise RuntimeError(f"{label} has 0 rows: {path}")
    return out


def load_concat(root: Path, filename: str, label: str) -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        p = root / str(season) / "trace" / filename
        frames.append(_read(p, f"{label} season={season}"))
    return pd.concat(frames, ignore_index=True)


def build_cohort(qb: pd.DataFrame, players: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """One true model WR1 per (season, week, event_id, team), restricted to
    the exact QB C2 authority identities. Fails closed on drift."""
    wr = players.loc[players.position.astype(str).str.upper().eq("WR")].copy()
    if wr.empty:
        raise RuntimeError("no WR rows in player projection trace")
    wr["b0_target_probability"] = pd.to_numeric(wr["b0_target_probability"], errors="coerce")
    wr = wr.sort_values(
        ["season", "week", "event_id", "team", "b0_target_probability", "player_clean_key"],
        ascending=[True, True, True, True, False, True],
    )
    wr1 = wr.groupby(["season", "week", "event_id", "team"], as_index=False).first()

    dup = qb.duplicated(["season", "week", "event_id", "team"], keep=False)
    if dup.any():
        raise RuntimeError(f"duplicate QB C2 authority identity:\n{qb.loc[dup, ['season', 'week', 'event_id', 'team']].to_string(index=False)}")

    cohort = wr1.merge(
        qb[["season", "week", "event_id", "team", "c2_mean", "c2_p10", "c2_p90", "c2_p95",
            "b0_mean", "b0_p90", "b0_p10"]],
        on=["season", "week", "event_id", "team"], how="inner", validate="one_to_one",
    )
    if cohort.duplicated(["season", "week", "event_id", "team"]).any():
        raise RuntimeError("cohort has duplicate team-game identity after QB join")

    for season in SEASONS:
        n = int((cohort.season == season).sum())
        if n != EXPECTED_QB_ROWS[season]:
            raise RuntimeError(f"cohort row-count drift season={season} expected={EXPECTED_QB_ROWS[season]} got={n}")

    au = actual[["season", "week", "team", "player_clean_key", "rec_yards"]].rename(
        columns={"rec_yards": "actual_rec_yards"}
    )
    cohort = cohort.merge(au, on=["season", "week", "team", "player_clean_key"], how="left", validate="one_to_one")
    if cohort["actual_rec_yards"].isna().any():
        missing = cohort.loc[cohort.actual_rec_yards.isna(), ["season", "week", "team", "player_clean_key"]]
        raise RuntimeError(f"missing actual outcome for cohort rows:\n{missing.to_string(index=False)}")
    return cohort.reset_index(drop=True)


def add_features_and_outcomes(cohort: pd.DataFrame) -> pd.DataFrame:
    c = cohort.copy()
    c["qb_c2_upper90"] = c.c2_p90 - c.c2_mean
    c["qb_c2_upper95"] = c.c2_p95 - c.c2_mean
    c["qb_c2_i80"] = c.c2_p90 - c.c2_p10
    c["qb_c2_right_skew"] = (c.c2_p90 - c.c2_mean) - (c.c2_mean - c.c2_p10)
    c["qb_c2_tail_expansion80"] = (c.c2_p90 - c.c2_p10) - (c.b0_p90 - c.b0_p10)

    c["wr_residual"] = c.actual_rec_yards - c.b0_rec_yards
    c["tail100"] = (c.actual_rec_yards >= 100).astype(int)
    c["cat_under40"] = (c.wr_residual >= 40).astype(int)
    c["cat_under50"] = (c.wr_residual >= 50).astype(int)
    return c


def anti_retest_scoreboard(cohort: pd.DataFrame) -> pd.DataFrame:
    """Broad-C2 vs B0 on true WR1 -- must be reported separately, cannot be
    called a new candidate even if it looks good."""
    rows = []
    for season in list(SEASONS) + ["ALL"]:
        g = cohort if season == "ALL" else cohort.loc[cohort.season.eq(season)]
        b0_err = (g.b0_rec_yards - g.actual_rec_yards).to_numpy(float)
        c2_err = (g.c2_rec_yards - g.actual_rec_yards).to_numpy(float)
        rows.append({
            "season": season, "n": len(g),
            "b0_mae": float(np.mean(np.abs(b0_err))), "c2_mae": float(np.mean(np.abs(c2_err))),
            "b0_rmse": float(np.sqrt(np.mean(b0_err ** 2))), "c2_rmse": float(np.sqrt(np.mean(c2_err ** 2))),
            "b0_bias": float(np.mean(b0_err)), "c2_bias": float(np.mean(c2_err)),
            "b0_p90_ae": float(np.quantile(np.abs(b0_err), 0.90)), "c2_p90_ae": float(np.quantile(np.abs(c2_err), 0.90)),
            "tail100_n": int(g.tail100.sum()),
            "b0_underproj_on_tail100_rate": float((g.loc[g.tail100.eq(1), "b0_rec_yards"] < g.loc[g.tail100.eq(1), "actual_rec_yards"]).mean()) if int(g.tail100.sum()) else np.nan,
            "c2_underproj_on_tail100_rate": float((g.loc[g.tail100.eq(1), "c2_rec_yards"] < g.loc[g.tail100.eq(1), "actual_rec_yards"]).mean()) if int(g.tail100.sum()) else np.nan,
        })
    return pd.DataFrame(rows)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ar = pd.Series(a).rank()
    br = pd.Series(b).rank()
    return float(np.corrcoef(ar, br)[0, 1])


def bootstrap_p_top_exceeds_rest(top: np.ndarray, rest: np.ndarray, *, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    top = np.asarray(top, dtype=float)
    rest = np.asarray(rest, dtype=float)
    count = 0
    for _ in range(n):
        t = rng.choice(top, size=len(top), replace=True)
        r = rng.choice(rest, size=len(rest), replace=True)
        if t.mean() > r.mean():
            count += 1
    return count / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="joint_pass_receiving_v1 artifact root (has 2024/2025 subdirs)")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    qb = load_concat(a.root, "joint_v1_qb_distribution_trace.csv", "QB distribution trace")
    players = load_concat(a.root, "joint_v1_player_projection_trace.csv", "player projection trace")
    actual = load_concat(a.root, "joint_v1_actual_usage.csv", "actual usage")

    cohort = build_cohort(qb, players, actual)
    cohort = add_features_and_outcomes(cohort)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(a.out_dir / "wr_qb_shared_tail_signal_v1_cohort.csv", index=False)

    scoreboard = anti_retest_scoreboard(cohort)
    scoreboard.to_csv(a.out_dir / "wr_qb_shared_tail_signal_v1_anti_retest_scoreboard.csv", index=False)

    dev = cohort.loc[cohort.season.eq(DEV_SEASON)].copy()
    hold = cohort.loc[cohort.season.eq(HOLDOUT_SEASON)].copy()

    threshold = float(dev.qb_c2_upper90.quantile(0.75))
    opp_quartile_edges = [float(x) for x in dev.b0_rec_yards.quantile([0.25, 0.50, 0.75]).tolist()]

    def bucket(row_val: float) -> int:
        if row_val <= opp_quartile_edges[0]:
            return 1
        if row_val <= opp_quartile_edges[1]:
            return 2
        if row_val <= opp_quartile_edges[2]:
            return 3
        return 4

    results = {}
    for label, frame in (("dev_2024", dev), ("holdout_2025", hold)):
        rho = spearman(frame.qb_c2_upper90.to_numpy(float), frame.wr_residual.to_numpy(float))
        top = frame.loc[frame.qb_c2_upper90 >= threshold]
        rest = frame.loc[frame.qb_c2_upper90 < threshold]
        gap = float(top.wr_residual.mean() - rest.wr_residual.mean())
        tail100_ratio = float(top.tail100.mean() / rest.tail100.mean()) if rest.tail100.mean() > 0 else np.nan
        cat40_ratio = float(top.cat_under40.mean() / rest.cat_under40.mean()) if rest.cat_under40.mean() > 0 else np.nan
        results[label] = {
            "n": len(frame), "n_top": len(top), "n_rest": len(rest),
            "spearman_qb_c2_upper90_vs_wr_residual": rho,
            "top_quartile_threshold_source": "2024_frozen" if label == "holdout_2025" else "2024_own",
            "threshold_value": threshold,
            "top_quartile_mean_residual": float(top.wr_residual.mean()),
            "rest_mean_residual": float(rest.wr_residual.mean()),
            "top_vs_rest_residual_gap": gap,
            "top_tail100_rate": float(top.tail100.mean()),
            "rest_tail100_rate": float(rest.tail100.mean()),
            "tail100_rate_ratio": tail100_ratio,
            "top_cat_under40_rate": float(top.cat_under40.mean()),
            "rest_cat_under40_rate": float(rest.cat_under40.mean()),
            "cat_under40_rate_ratio": cat40_ratio,
            "top_cat_under50_rate": float(top.cat_under50.mean()),
            "rest_cat_under50_rate": float(rest.cat_under50.mean()),
        }

    boot_p = bootstrap_p_top_exceeds_rest(
        hold.loc[hold.qb_c2_upper90 >= threshold, "wr_residual"].to_numpy(float),
        hold.loc[hold.qb_c2_upper90 < threshold, "wr_residual"].to_numpy(float),
        n=BOOT_N, seed=BOOT_SEED,
    )
    results["holdout_2025"]["bootstrap_p_top_exceeds_rest"] = boot_p

    hold = hold.copy()
    hold["opp_quartile"] = hold.b0_rec_yards.map(bucket)
    opp_rows = []
    nonneg_count = 0
    for q in (1, 2, 3, 4):
        gq = hold.loc[hold.opp_quartile.eq(q)]
        top = gq.loc[gq.qb_c2_upper90 >= threshold]
        rest = gq.loc[gq.qb_c2_upper90 < threshold]
        gap_q = float(top.wr_residual.mean() - rest.wr_residual.mean()) if len(top) and len(rest) else np.nan
        if np.isfinite(gap_q) and gap_q >= 0:
            nonneg_count += 1
        opp_rows.append({"baseline_opportunity_quartile": q, "n_top": len(top), "n_rest": len(rest), "residual_gap": gap_q})
    opp_df = pd.DataFrame(opp_rows)
    opp_df.to_csv(a.out_dir / "wr_qb_shared_tail_signal_v1_opportunity_conditional.csv", index=False)

    with pd.option_context("display.width", 200):
        pd.DataFrame(results).to_csv(a.out_dir / "wr_qb_shared_tail_signal_v1_summary.csv")

    dev_r = results["dev_2024"]
    hold_r = results["holdout_2025"]
    dev_coherent = (dev_r["spearman_qb_c2_upper90_vs_wr_residual"] > 0) and (dev_r["top_vs_rest_residual_gap"] > 0)

    gate_checks = {
        "identity_parity_pass": True,
        "sportsbook_inputs_used": 0,
        "primary_spearman_pass": hold_r["spearman_qb_c2_upper90_vs_wr_residual"] >= GATES["primary_spearman_min"],
        "top_quartile_residual_gap_pass": hold_r["top_vs_rest_residual_gap"] >= GATES["top_quartile_residual_gap_min"],
        "bootstrap_pass": hold_r["bootstrap_p_top_exceeds_rest"] >= GATES["bootstrap_p_min"],
        "tail100_ratio_pass": (hold_r["tail100_rate_ratio"] >= GATES["tail100_rate_ratio_min"]) if np.isfinite(hold_r["tail100_rate_ratio"]) else False,
        "cat_under40_ratio_pass": (hold_r["cat_under40_rate_ratio"] >= GATES["cat_under40_rate_ratio_min"]) if np.isfinite(hold_r["cat_under40_rate_ratio"]) else False,
        "opportunity_conditional_pass": nonneg_count >= GATES["opportunity_conditional_min_quartiles"],
        "dev_2024_coherence_pass": dev_coherent,
    }
    all_pass = all(bool(v) for k, v in gate_checks.items() if k != "sportsbook_inputs_used")
    disposition = "WR_QB_SHARED_TAIL_SIGNAL_SUPPORTED" if all_pass else "NO_ACTIONABLE_WR_QB_SHARED_TAIL_SIGNAL"

    gate_json = {
        "disposition": disposition,
        "gate_checks": gate_checks,
        "gate_thresholds": GATES,
        "dev_2024": dev_r,
        "holdout_2025": hold_r,
        "opportunity_conditional_nonneg_quartiles": nonneg_count,
    }
    (a.out_dir / "wr_qb_shared_tail_signal_v1_result.json").write_text(json.dumps(gate_json, indent=2, sort_keys=True))

    print("=== WR-QB SHARED TAIL SIGNAL V1 RESULT ===")
    print(json.dumps(gate_json, indent=2, sort_keys=True))
    print("\n=== ANTI-RETEST SCOREBOARD (broad C2 vs B0 on true WR1) ===")
    print(scoreboard.to_string(index=False))
    print("\n=== OPPORTUNITY-CONDITIONAL (2025, 2024-frozen quartiles) ===")
    print(opp_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
