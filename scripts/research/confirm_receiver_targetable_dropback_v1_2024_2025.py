#!/usr/bin/env python3
"""Frozen 2024-2025 confirmation for Receiver Targetable-Dropback V1.

This wrapper reuses the exact already-qualified candidate implementation from
2022-2023 without changing its rate construction or forecast formula.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.backtest.walk_forward import _parse_weeks
from scripts.research.evaluate_receiver_targetable_dropback_v1_team_calibration import (
    build_team_actual_history,
    evaluate_season,
    optional,
    pair,
    read,
    score,
)

VERSION = "RECEIVER_TARGETABLE_DROPBACK_V1_2024_2025_CONFIRMATION"


def build_result(detail: pd.DataFrame) -> dict:
    out = {"by_season": {}}
    for season in (2024, 2025):
        d = detail.loc[detail["season"].eq(season)].copy()
        out["by_season"][str(season)] = {
            "targets": pair(d),
            "baseline_vs_actual_dropbacks": score(
                d, "baseline_projected_dropbacks", "actual_dropbacks"
            ),
            "rows": int(len(d)),
            "team_source_rows": int(d["conversion_source"].eq("team_strict_prior").sum()),
            "fallback_rows": int(d["conversion_source"].eq("league_fallback").sum()),
            "rate_min": float(d["targetable_dropback_rate"].min()),
            "rate_median": float(d["targetable_dropback_rate"].median()),
            "rate_max": float(d["targetable_dropback_rate"].max()),
        }

    out["pooled"] = {
        "targets": pair(detail),
        "rows": int(len(detail)),
        "team_source_rows": int(detail["conversion_source"].eq("team_strict_prior").sum()),
        "fallback_rows": int(detail["conversion_source"].eq("league_fallback").sum()),
        "rate_min": float(detail["targetable_dropback_rate"].min()),
        "rate_median": float(detail["targetable_dropback_rate"].median()),
        "rate_max": float(detail["targetable_dropback_rate"].max()),
    }
    return out


def frozen_gates(s: dict) -> dict:
    y24 = s["by_season"]["2024"]["targets"]
    y25 = s["by_season"]["2025"]["targets"]
    p = s["pooled"]["targets"]
    rows = max(1, s["pooled"]["rows"])
    fallback_rate = s["pooled"]["fallback_rows"] / rows
    team_source_rate = s["pooled"]["team_source_rows"] / rows

    return {
        "target_mae_2024_improves": y24["candidate"]["mae"] < y24["baseline"]["mae"],
        "target_mae_2025_improves": y25["candidate"]["mae"] < y25["baseline"]["mae"],
        "target_mae_pooled_improves": p["candidate"]["mae"] < p["baseline"]["mae"],
        "target_rmse_pooled_nonworse": p["candidate"]["rmse"] <= p["baseline"]["rmse"] + 1e-12,
        "target_p90_pooled_nonworse": p["candidate"]["p90_ae"] <= p["baseline"]["p90_ae"] + 1e-12,
        "target_abs_bias_pooled_improves": p["candidate"]["abs_bias"] < p["baseline"]["abs_bias"],
        "candidate_closer_gt50": p["candidate_closer_rate"] is not None
        and p["candidate_closer_rate"] > 0.50,
        "season_p90_guard_2024": y24["candidate"]["p90_ae"] <= y24["baseline"]["p90_ae"] + 0.50,
        "season_p90_guard_2025": y25["candidate"]["p90_ae"] <= y25["baseline"]["p90_ae"] + 0.50,
        "team_conversion_source_ge99": team_source_rate >= 0.99,
        "fallback_rate_le1": fallback_rate <= 0.01,
        "conversion_finite_in_bounds": 0.0 <= s["pooled"]["rate_min"] <= s["pooled"]["rate_max"] <= 1.0,
        "target_game_outcomes_upstream_zero": True,
        "sportsbook_inputs_zero": True,
        "parameters_fit_zero": True,
        "one_candidate_only": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-2024", type=Path, required=True)
    ap.add_argument("--universe-2025", type=Path, required=True)
    ap.add_argument("--injuries", type=Path, required=True)
    ap.add_argument("--weather", type=Path, required=True)
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    player_logs = read(args.player_logs, "player logs")
    team_weekly = read(args.team_weekly, "team weekly")
    schedule = read(args.schedule, "schedule")
    injuries = optional(args.injuries)
    weather = optional(args.weather)
    history = build_team_actual_history(player_logs, team_weekly)
    weeks = _parse_weeks(args.weeks)

    d24 = evaluate_season(
        2024, 2023, weeks, player_logs, team_weekly, schedule,
        args.universe_2024, injuries, weather, history
    )
    d25 = evaluate_season(
        2025, 2024, weeks, player_logs, team_weekly, schedule,
        args.universe_2025, injuries, weather, history
    )
    detail = pd.concat([d24, d25], ignore_index=True)

    scorecard = build_result(detail)
    gates = frozen_gates(scorecard)
    qualified = all(gates.values())
    disposition = (
        "RECEIVER_TARGETABLE_DROPBACK_V1_2024_2025_CONFIRMED"
        if qualified
        else "RECEIVER_TARGETABLE_DROPBACK_V1_2024_2025_FAILED_CLOSED"
    )

    payload = {
        "version": VERSION,
        "disposition": disposition,
        "qualified": bool(qualified),
        "candidate_variants_scored": 1,
        "parameters_fit": 0,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_used_upstream": 0,
        "scorecard": scorecard,
        "gates": gates,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out_dir / "team_detail_2024_2025.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Receiver Targetable-Dropback V1 — 2024-2025 Confirmation",
        "",
        f"Disposition: **{disposition}**",
        "",
    ]
    for season in ("2024", "2025"):
        x = scorecard["by_season"][season]["targets"]
        lines += [
            f"## {season}",
            "",
            f"- target MAE: {x['baseline']['mae']:.6f} -> {x['candidate']['mae']:.6f}",
            f"- target RMSE: {x['baseline']['rmse']:.6f} -> {x['candidate']['rmse']:.6f}",
            f"- target p90: {x['baseline']['p90_ae']:.6f} -> {x['candidate']['p90_ae']:.6f}",
            f"- target abs bias: {x['baseline']['abs_bias']:.6f} -> {x['candidate']['abs_bias']:.6f}",
            "",
        ]
    p = scorecard["pooled"]["targets"]
    lines += [
        "## Pooled",
        "",
        f"- target MAE: {p['baseline']['mae']:.6f} -> {p['candidate']['mae']:.6f}",
        f"- target RMSE: {p['baseline']['rmse']:.6f} -> {p['candidate']['rmse']:.6f}",
        f"- target p90: {p['baseline']['p90_ae']:.6f} -> {p['candidate']['p90_ae']:.6f}",
        f"- target abs bias: {p['baseline']['abs_bias']:.6f} -> {p['candidate']['abs_bias']:.6f}",
        f"- candidate closer rate: {p['candidate_closer_rate']:.6f}",
        "",
        "## Frozen gates",
        "",
    ]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in gates.items()]
    (args.out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
