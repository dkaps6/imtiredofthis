#!/usr/bin/env python3
"""docs/research/GAME_SCRIPT_CONFIRMED_PLAYER_USAGE_V1_PLAN.md.

Diagnostic only. Distinct from:
- scripts/research/diagnose_market_implied_game_script_v1.py (PR #558): does
  the market add value on top of our own historical team-plays baseline?
- scripts/research/diagnose_vegas_line_gamescript_calibration_v1.py (PR #559):
  is the market's own posted line accurate against actual outcomes?

This asks: in games where Vegas's implied script was actually realized (the
"confirmed" subset, defined from #559's actual-vs-predicted error), do
specific player-position volumes (RB rush volume for the leading team,
WR/TE volume in high-total games) show a cleaner pattern than in the
unconditional sample -- and does that pattern approach the ground-truth
(actual-outcome) ceiling once restricted to confirmed games?

This measures a ceiling (script-confirmed is defined from the actual
outcome), not a pregame-actionable signal by itself -- see the plan doc's
"honest framing" section. No production/model/threshold change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.historical_player_logs import build_historical_player_logs
from scripts.research.diagnose_vegas_line_gamescript_calibration_v1 import load_game_outcomes

CONFIRM_THRESHOLDS = [3.0, 7.0]
MIN_ROWS_PER_SIDE = 30


def to_team_game_market_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Expand one row per game into two team-perspective rows (home, away)."""
    rows = []
    for r in games.itertuples(index=False):
        for side, team, opponent in [("home", r.home_team, r.away_team), ("away", r.away_team, r.home_team)]:
            sign = 1.0 if side == "home" else -1.0
            rows.append({
                "season": int(r.season), "week": int(r.week), "game_id": r.game_id,
                "team": team, "opponent": opponent,
                "predicted_team_margin": sign * r.predicted_margin_home,
                "actual_team_margin": sign * r.actual_margin_home,
                "predicted_total": r.predicted_total,
                "actual_total": r.actual_total,
            })
    out = pd.DataFrame(rows)
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate team-game market rows")
    return out


def games_to_schedule_history(games: pd.DataFrame) -> pd.DataFrame:
    """Team/opponent schedule shape required by build_historical_player_logs,
    derived from the same game rows already fetched -- no second schedule call.
    """
    rows = []
    for r in games.itertuples(index=False):
        rows.append({"season": int(r.season), "week": int(r.week), "team": r.home_team, "opponent": r.away_team, "game_id": r.game_id})
        rows.append({"season": int(r.season), "week": int(r.week), "team": r.away_team, "opponent": r.home_team, "game_id": r.game_id})
    return pd.DataFrame(rows)


def team_game_position_aggregates(player_logs: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with RB rush volume and WR/TE
    receiving volume, plus the team-level dropback/rush/target denominators
    already computed per player row in historical_player_logs.
    """
    keys = ["season", "week", "team"]
    base = player_logs[keys + ["team_rushes", "team_targets", "team_dropbacks"]].drop_duplicates(keys)

    rb = (
        player_logs.loc[player_logs["position"].eq("RB")]
        .groupby(keys, dropna=False)
        .agg(rb_rush_att=("rushes", "sum"), rb_rush_yards=("rush_yards", "sum"))
        .reset_index()
    )
    wrte = (
        player_logs.loc[player_logs["position"].isin(["WR", "TE"])]
        .groupby(keys, dropna=False)
        .agg(wrte_targets=("targets", "sum"), wrte_rec_yards=("rec_yards", "sum"))
        .reset_index()
    )

    out = base.merge(rb, on=keys, how="left").merge(wrte, on=keys, how="left")
    for col in ["rb_rush_att", "rb_rush_yards", "wrte_targets", "wrte_rec_yards"]:
        out[col] = out[col].fillna(0.0)
    return out


def build_analysis_frame(games: pd.DataFrame, player_logs: pd.DataFrame) -> pd.DataFrame:
    market = to_team_game_market_frame(games)
    positions = team_game_position_aggregates(player_logs)
    out = market.merge(positions, on=["season", "week", "team"], how="inner", validate="one_to_one")
    out["margin_abs_error"] = (out["actual_team_margin"] - out["predicted_team_margin"]).abs()
    out["total_abs_error"] = (out["actual_total"] - out["predicted_total"]).abs()
    return out


def _cohens_d(high: np.ndarray, low: np.ndarray) -> float:
    n1, n2 = len(high), len(low)
    if n1 < 2 or n2 < 2:
        return float("nan")
    v1, v2 = np.var(high, ddof=1), np.var(low, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return float("nan")
    return float((np.mean(high) - np.mean(low)) / pooled_sd)


def compare_by_side(frame: pd.DataFrame, *, side_col: str, metric_col: str) -> dict:
    high = frame.loc[frame[side_col].eq(True), metric_col].dropna().to_numpy(dtype=float)
    low = frame.loc[frame[side_col].eq(False), metric_col].dropna().to_numpy(dtype=float)
    if len(high) < MIN_ROWS_PER_SIDE or len(low) < MIN_ROWS_PER_SIDE:
        return {"status": "INSUFFICIENT_ROWS", "n_high": int(len(high)), "n_low": int(len(low))}
    return {
        "status": "OK",
        "n_high": int(len(high)), "n_low": int(len(low)),
        "mean_high": float(np.mean(high)), "mean_low": float(np.mean(low)),
        "diff": float(np.mean(high) - np.mean(low)),
        "cohens_d": _cohens_d(high, low),
    }


def run_margin_hypothesis(frame: pd.DataFrame, *, season_label: str, threshold: float) -> list[dict]:
    rows = []
    ground_truth = frame.loc[~frame["actual_team_margin"].eq(0.0)].copy()
    ground_truth["side"] = ground_truth["actual_team_margin"] > 0

    predicted_all = frame.loc[~frame["predicted_team_margin"].eq(0.0)].copy()
    predicted_all["side"] = predicted_all["predicted_team_margin"] > 0

    confirmed = predicted_all.loc[predicted_all["margin_abs_error"] <= threshold].copy()

    for arm_label, arm_frame in [
        ("A_ground_truth_actual_split", ground_truth),
        ("B_vegas_unconditional", predicted_all),
        ("C_vegas_confirmed_only", confirmed),
    ]:
        for metric in ["rb_rush_att", "rb_rush_yards"]:
            result = compare_by_side(arm_frame, side_col="side", metric_col=metric)
            result.update({"season": season_label, "hypothesis": "margin_to_rb_volume", "arm": arm_label, "metric": metric, "confirm_threshold": threshold})
            rows.append(result)
    return rows


def run_total_hypothesis(frame: pd.DataFrame, *, season_label: str, threshold: float, total_cutoff: float) -> list[dict]:
    rows = []
    ground_truth = frame.copy()
    ground_truth["side"] = ground_truth["actual_total"] >= total_cutoff

    predicted_all = frame.copy()
    predicted_all["side"] = predicted_all["predicted_total"] >= total_cutoff

    confirmed = predicted_all.loc[predicted_all["total_abs_error"] <= threshold].copy()

    for arm_label, arm_frame in [
        ("A_ground_truth_actual_split", ground_truth),
        ("B_vegas_unconditional", predicted_all),
        ("C_vegas_confirmed_only", confirmed),
    ]:
        for metric in ["wrte_targets", "wrte_rec_yards"]:
            result = compare_by_side(arm_frame, side_col="side", metric_col=metric)
            result.update({"season": season_label, "hypothesis": "total_to_wrte_volume", "arm": arm_label, "metric": metric, "confirm_threshold": threshold})
            rows.append(result)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=str, default="2023,2024,2025")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    games = load_game_outcomes(seasons)
    schedule_history = games_to_schedule_history(games)
    player_logs = build_historical_player_logs(seasons=seasons, schedule_history=schedule_history)

    frame = build_analysis_frame(games, player_logs)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out_dir / "game_script_confirmed_player_usage_frame.csv", index=False)

    # Frozen once from the full pooled sample, used identically across arms.
    total_cutoff = float(frame["predicted_total"].median())

    results = []
    groups = [("ALL_SEASONS", frame)] + [(str(s), frame.loc[frame.season.eq(s)]) for s in sorted(frame.season.unique())]
    for season_label, group in groups:
        for threshold in CONFIRM_THRESHOLDS:
            results.extend(run_margin_hypothesis(group, season_label=season_label, threshold=threshold))
            results.extend(run_total_hypothesis(group, season_label=season_label, threshold=threshold, total_cutoff=total_cutoff))

    out = pd.DataFrame(results)
    out.to_csv(a.out_dir / "game_script_confirmed_player_usage_summary.csv", index=False)

    print("=== GAME-SCRIPT-CONFIRMED PLAYER USAGE DIAGNOSIS ===")
    print(f"total_split_cutoff={total_cutoff}")
    print(out.loc[out.season.eq("ALL_SEASONS")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
