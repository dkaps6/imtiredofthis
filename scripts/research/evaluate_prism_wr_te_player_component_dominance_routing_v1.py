#!/usr/bin/env python3
"""PRISM WR/TE player component-dominance routing V1.

Frozen player-specific mechanism-routing test. Predicts the next receiving-yard
error component class from strictly-prior same-player component-state history,
shrunk toward a target-player-excluded position prior. No projection, width,
sportsbook, or production changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PRISM_WR_TE_PLAYER_COMPONENT_DOMINANCE_ROUTING_V1"
FREEZE_COMMIT = "f72e8c644c8cac0280fddb39ba4e02cc96c30931"
DOMINANCE_RATIO = 1.5
MAX_PRIOR = 8
MIN_PRIOR = 4
PRIOR_PSEUDO = 4.0
N_BOOT = 5000
PLAYER_SEED = 20261018
TEAM_WEEK_SEED = 20261019
TAIL_FLOOR = -0.010000
TOL = 1e-10
LOG_EPS = 1e-15

CLASSES = ("OPPORTUNITY_DOMINANT", "EFFICIENCY_DOMINANT", "MIXED")
REQ = {
    "season", "week", "team", "player_clean_key",
    "current_opportunity_error", "current_efficiency_error", "current_total_error",
}


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = REQ - set(x.columns)
    if miss:
        raise RuntimeError(f"{label} missing required fields: {sorted(miss)}")
    return x


def component_class(opp, eff) -> str:
    o = abs(float(opp))
    e = abs(float(eff))
    if o >= DOMINANCE_RATIO * e:
        return "OPPORTUNITY_DOMINANT"
    if e >= DOMINANCE_RATIO * o:
        return "EFFICIENCY_DOMINANT"
    return "MIXED"


def prepare_panel(x: pd.DataFrame, position: str, seasons: tuple[int, int]) -> pd.DataFrame:
    y = x.copy()
    y["season"] = pd.to_numeric(y["season"], errors="coerce")
    y["week"] = pd.to_numeric(y["week"], errors="coerce")
    for c in ("current_opportunity_error", "current_efficiency_error", "current_total_error"):
        y[c] = pd.to_numeric(y[c], errors="coerce")
    required = ["season","week","team","player_clean_key","current_opportunity_error","current_efficiency_error","current_total_error"]
    if y[required].isna().any().any():
        raise RuntimeError(f"{position} has nonfinite required fields")
    observed = tuple(sorted(int(v) for v in y["season"].unique()))
    if observed != seasons:
        raise RuntimeError(f"{position} season drift {observed} != {seasons}")
    keys = ["season","week","team","player_clean_key"]
    if y.duplicated(keys).any():
        bad = y.loc[y.duplicated(keys, keep=False), keys].head(20).to_dict("records")
        raise RuntimeError(f"{position} duplicate player-week identity: {bad}")

    y["decomposition_gap"] = (
        y["current_opportunity_error"] + y["current_efficiency_error"] - y["current_total_error"]
    ).abs()
    max_gap = float(y["decomposition_gap"].max())
    if max_gap > TOL:
        raise RuntimeError(f"{position} decomposition identity failure {max_gap}")

    y["target_class"] = [
        component_class(o, e)
        for o, e in zip(y["current_opportunity_error"], y["current_efficiency_error"])
    ]
    y["position"] = position
    y = y.sort_values(["season","week","team","player_clean_key"], kind="mergesort").reset_index(drop=True)
    return y


def _distribution(labels: pd.Series) -> dict[str, float]:
    n = int(len(labels))
    if n <= 0:
        raise RuntimeError("zero-row class distribution")
    counts = labels.value_counts()
    return {c: float(counts.get(c, 0) / n) for c in CLASSES}


def build_predictions(panel: pd.DataFrame, position: str) -> pd.DataFrame:
    rows = []
    player_groups = {
        (int(s), str(p)): g.sort_values(["week","team"], kind="mergesort")
        for (s,p), g in panel.groupby(["season","player_clean_key"], sort=False)
    }

    for r in panel.itertuples(index=False):
        season = int(r.season)
        week = int(r.week)
        player = str(r.player_clean_key)

        g = player_groups[(season, player)]
        prior_player = g.loc[g["week"].lt(week)].tail(MAX_PRIOR)
        if len(prior_player) < MIN_PRIOR:
            continue
        if not (prior_player["week"] < week).all():
            raise RuntimeError(f"{position} player chronology violation {season} W{week} {player}")

        # Frozen position prior: same position/season, strictly earlier weeks,
        # excluding every row from the target player.
        prior_position = panel.loc[
            panel["season"].eq(season)
            & panel["week"].lt(week)
            & ~panel["player_clean_key"].astype(str).eq(player)
        ]
        if prior_position.empty:
            raise RuntimeError(f"{position} zero position-prior rows {season} W{week} {player}")
        if not (prior_position["week"] < week).all():
            raise RuntimeError(f"{position} position-prior chronology violation")

        p_player = _distribution(prior_player["target_class"])
        p_position = _distribution(prior_position["target_class"])
        n = int(len(prior_player))
        p_candidate = {
            c: float((n * p_player[c] + PRIOR_PSEUDO * p_position[c]) / (n + PRIOR_PSEUDO))
            for c in CLASSES
        }
        if abs(sum(p_candidate.values()) - 1.0) > 1e-12:
            raise RuntimeError("candidate probabilities do not sum to one")
        if abs(sum(p_position.values()) - 1.0) > 1e-12:
            raise RuntimeError("baseline probabilities do not sum to one")

        realized = str(r.target_class)
        yvec = {c: 1.0 if c == realized else 0.0 for c in CLASSES}
        baseline_brier = float(sum((p_position[c] - yvec[c]) ** 2 for c in CLASSES))
        candidate_brier = float(sum((p_candidate[c] - yvec[c]) ** 2 for c in CLASSES))
        baseline_logloss = float(-np.log(max(p_position[realized], LOG_EPS)))
        candidate_logloss = float(-np.log(max(p_candidate[realized], LOG_EPS)))

        top_base = max(CLASSES, key=lambda c: (p_position[c], -CLASSES.index(c)))
        top_cand = max(CLASSES, key=lambda c: (p_candidate[c], -CLASSES.index(c)))
        rows.append({
            "position": position,
            "season": season,
            "week": week,
            "team": str(r.team),
            "player_clean_key": player,
            "target_class": realized,
            "prior_player_rows": n,
            "prior_player_min_week": int(prior_player["week"].min()),
            "prior_player_max_week": int(prior_player["week"].max()),
            "prior_position_rows": int(len(prior_position)),
            "player_opp_share": p_player["OPPORTUNITY_DOMINANT"],
            "player_eff_share": p_player["EFFICIENCY_DOMINANT"],
            "player_mixed_share": p_player["MIXED"],
            "baseline_opp_prob": p_position["OPPORTUNITY_DOMINANT"],
            "baseline_eff_prob": p_position["EFFICIENCY_DOMINANT"],
            "baseline_mixed_prob": p_position["MIXED"],
            "candidate_opp_prob": p_candidate["OPPORTUNITY_DOMINANT"],
            "candidate_eff_prob": p_candidate["EFFICIENCY_DOMINANT"],
            "candidate_mixed_prob": p_candidate["MIXED"],
            "baseline_brier": baseline_brier,
            "candidate_brier": candidate_brier,
            "brier_improvement": baseline_brier - candidate_brier,
            "baseline_logloss": baseline_logloss,
            "candidate_logloss": candidate_logloss,
            "logloss_improvement": baseline_logloss - candidate_logloss,
            "baseline_realized_prob": p_position[realized],
            "candidate_realized_prob": p_candidate[realized],
            "baseline_top_class": top_base,
            "candidate_top_class": top_cand,
            "baseline_correct": int(top_base == realized),
            "candidate_correct": int(top_cand == realized),
            "decomposition_gap": float(r.decomposition_gap),
            "team_week_cluster": f"{season}-{week}-{r.team}",
            "chronology_valid": True,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"{position} zero scoreable rows")
    return out.sort_values(["season","week","team","player_clean_key"], kind="mergesort").reset_index(drop=True)


def _cluster_bootstrap(rows: pd.DataFrame, cluster_col: str, seed: int) -> dict:
    z = rows[["brier_improvement", cluster_col]].dropna().copy()
    clusters = sorted(z[cluster_col].astype(str).unique())
    if len(clusters) < 2:
        return {"reps": 0, "ci_low": np.nan, "ci_high": np.nan}
    groups = {c: z.loc[z[cluster_col].astype(str).eq(c), "brier_improvement"].to_numpy(float) for c in clusters}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        sample = rng.choice(clusters, size=len(clusters), replace=True)
        arr = np.concatenate([groups[c] for c in sample])
        vals.append(float(arr.mean()))
    a = np.asarray(vals, float)
    return {
        "reps": int(len(a)),
        "ci_low": float(np.quantile(a, 0.025)),
        "ci_high": float(np.quantile(a, 0.975)),
    }


def _equal_player_bootstrap(rows: pd.DataFrame, seed: int) -> dict:
    by = rows.groupby("player_clean_key")["brier_improvement"].mean().astype(float)
    vals = by.to_numpy(float)
    if len(vals) < 2:
        return {"reps": 0, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        boots.append(float(rng.choice(vals, size=len(vals), replace=True).mean()))
    a = np.asarray(boots, float)
    return {
        "reps": int(len(a)),
        "ci_low": float(np.quantile(a, 0.025)),
        "ci_high": float(np.quantile(a, 0.975)),
    }


def summarize(rows: pd.DataFrame, season: int, *, confirm: bool) -> dict:
    s = rows.loc[rows["season"].eq(season)].copy()
    if s.empty:
        raise RuntimeError(f"zero score rows for season {season}")
    player = s.groupby("player_clean_key")["brier_improvement"].mean().astype(float)
    out = {
        "season": int(season),
        "rows": int(len(s)),
        "players": int(s["player_clean_key"].nunique()),
        "teams": int(s["team"].nunique()),
        "baseline_brier": float(s["baseline_brier"].mean()),
        "candidate_brier": float(s["candidate_brier"].mean()),
        "brier_improvement": float(s["brier_improvement"].mean()),
        "baseline_logloss": float(s["baseline_logloss"].mean()),
        "candidate_logloss": float(s["candidate_logloss"].mean()),
        "logloss_improvement": float(s["logloss_improvement"].mean()),
        "mean_equal_player_brier_improvement": float(player.mean()),
        "median_player_brier_improvement": float(player.median()),
        "p10_player_brier_improvement": float(player.quantile(0.10)),
        "player_improved_share": float((player > 0).mean()),
        "player_tied_share": float(np.isclose(player, 0.0, atol=1e-15).mean()),
        "player_worsened_share": float((player < 0).mean()),
        "baseline_top_class_accuracy": float(s["baseline_correct"].mean()),
        "candidate_top_class_accuracy": float(s["candidate_correct"].mean()),
        "mean_baseline_realized_prob": float(s["baseline_realized_prob"].mean()),
        "mean_candidate_realized_prob": float(s["candidate_realized_prob"].mean()),
        "class_counts": {c: int((s["target_class"] == c).sum()) for c in CLASSES},
    }
    if confirm:
        out["player_cluster"] = _cluster_bootstrap(s, "player_clean_key", PLAYER_SEED)
        out["team_week_cluster"] = _cluster_bootstrap(s, "team_week_cluster", TEAM_WEEK_SEED)
        out["equal_player"] = _equal_player_bootstrap(s, PLAYER_SEED + 1)
    return out


def evaluate_position(rows: pd.DataFrame, position: str, dev: int, conf: int) -> dict:
    d = summarize(rows, dev, confirm=False)
    c = summarize(rows, conf, confirm=True)
    gates = {
        "development_brier_improves": bool(d["brier_improvement"] > 0),
        "development_logloss_nonworse": bool(d["logloss_improvement"] >= 0),
        "development_equal_player_positive": bool(d["mean_equal_player_brier_improvement"] > 0),
        "confirmation_brier_improves": bool(c["brier_improvement"] > 0),
        "confirmation_logloss_nonworse": bool(c["logloss_improvement"] >= 0),
        "confirmation_player_cluster_ci_low_positive": bool(c["player_cluster"]["ci_low"] > 0),
        "confirmation_team_week_cluster_ci_low_positive": bool(c["team_week_cluster"]["ci_low"] > 0),
        "confirmation_equal_player_ci_low_positive": bool(c["equal_player"]["ci_low"] > 0),
        "confirmation_median_player_nonnegative": bool(c["median_player_brier_improvement"] >= 0),
        "confirmation_improved_share_gt_worsened": bool(c["player_improved_share"] > c["player_worsened_share"]),
        "confirmation_p10_tail_floor": bool(c["p10_player_brier_improvement"] >= TAIL_FLOOR),
        "chronology_violations_zero": bool(rows["chronology_valid"].all()),
        "decomposition_identity_pass": bool(float(rows["decomposition_gap"].max()) <= TOL),
        "outcomes_2026_zero": True,
        "sportsbook_oddsapi_zero": True,
        "coefficients_fit_zero": True,
        "thresholds_searched_zero": True,
    }
    return {
        "position": position,
        "development": d,
        "confirmation": c,
        "gates": gates,
        "qualified": bool(all(gates.values())),
        "max_decomposition_identity_gap": float(rows["decomposition_gap"].max()),
        "chronology_violations": int((~rows["chronology_valid"]).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr", type=Path, required=True)
    ap.add_argument("--te", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wr_panel = prepare_panel(read_csv(args.wr, "WR source panel"), "WR", (2023, 2024))
    te_panel = prepare_panel(read_csv(args.te, "TE source panel"), "TE", (2024, 2025))
    wr_rows = build_predictions(wr_panel, "WR")
    te_rows = build_predictions(te_panel, "TE")

    wr = evaluate_position(wr_rows, "WR", 2023, 2024)
    te = evaluate_position(te_rows, "TE", 2024, 2025)

    if wr["qualified"] and te["qualified"]:
        disposition = "PRISM_WR_TE_PLAYER_COMPONENT_DOMINANCE_ROUTING_V1_QUALIFIED_ALL"
    elif wr["qualified"] or te["qualified"]:
        disposition = "PRISM_WR_TE_PLAYER_COMPONENT_DOMINANCE_ROUTING_V1_QUALIFIED_PARTIAL"
    else:
        disposition = "PRISM_WR_TE_PLAYER_COMPONENT_DOMINANCE_ROUTING_V1_CLOSED_ALL"

    result = {
        "version": VERSION,
        "freeze_commit": FREEZE_COMMIT,
        "disposition": disposition,
        "wr": wr,
        "te": te,
        "dominance_ratio": DOMINANCE_RATIO,
        "prior_window_max": MAX_PRIOR,
        "prior_window_min": MIN_PRIOR,
        "position_prior_pseudo_observations": PRIOR_PSEUDO,
        "tail_harm_floor": TAIL_FLOOR,
        "bootstrap_replicates": N_BOOT,
        "rb_status": "SOURCE_NOT_CERTIFIED_FOR_THIS_V1",
        "qb_status": "EXCLUDED_BY_ANTI_RETEST",
        "projection_means_changed": 0,
        "distributions_changed": 0,
        "coefficients_fit": 0,
        "thresholds_searched": 0,
        "outcomes_2026_used": 0,
        "week6_week7_outcomes_used": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "production_changed": False,
    }

    wr_rows.to_csv(args.out_dir / "wr_component_dominance_routing_rows.csv", index=False, float_format="%.12f")
    te_rows.to_csv(args.out_dir / "te_component_dominance_routing_rows.csv", index=False, float_format="%.12f")
    (args.out_dir / "component_dominance_routing_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
