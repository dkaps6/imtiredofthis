#!/usr/bin/env python3
"""Diagnostic-only decomposition of current RB receiving error.

This does not define or test a production candidate.  It uses the current
football-only finite target-entitlement architecture (M38 applied before the team
cap; no sportsbook inputs) and oracle substitutions from completed games only to
measure where RB receiving error is recoverable:

1. current target expectation;
2. actual modeled-player team target volume, current player shares;
3. actual RB-room target mass, current within-RB allocation;
4. current RB-room mass, actual within-RB allocation;
5. actual player target share;
6. actual player targets with current yards-per-target.

Stages 2-6 are forensic oracles, never deployable inputs.  The purpose is to tell
future research whether to model team volume, RB-room mass, within-room entitlement,
or receiving efficiency.  Production parameters are never modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest.evaluate_wr_room_empirical_bayes_v1 import finite, optional, prepared, read
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement

RB_POS = {"RB", "FB", "HB", "TB"}
MODELED_MASS = 0.95


def _metric(actual: pd.Series, pred: pd.Series) -> dict:
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(pred, errors="coerce")
    ok = a.notna() & p.notna() & np.isfinite(a) & np.isfinite(p)
    a, p = a.loc[ok].astype(float), p.loc[ok].astype(float)
    if not len(a):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan, "p90_abs_error": np.nan}
    e = p - a
    return {
        "n": int(len(a)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(e.to_numpy() ** 2))),
        "bias": float(e.mean()),
        "corr": float(p.corr(a)) if p.nunique() > 1 and a.nunique() > 1 else np.nan,
        "p90_abs_error": float(e.abs().quantile(0.90)),
    }


def _build_baseline(
    *, logs: pd.DataFrame, team: pd.DataFrame, schedule: pd.DataFrame,
    injuries: pd.DataFrame, weather: pd.DataFrame, universe: pd.DataFrame,
    season: int, week: int, prior_season: int,
) -> pd.DataFrame:
    bundle = build_historical_context_bundle(
        player_logs=logs, team_weekly=team, pregame_universe=universe,
        schedule=schedule, season=season, week=week, prior_season=prior_season,
        injuries=_exact_week(injuries, season, week),
        weather=_exact_week(weather, season, week),
    )
    raw = prepared(bundle)
    baseline, _ = materialize_target_entitlement(raw)
    return baseline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--prior-season", type=int, default=2024)
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--player-logs", type=Path, default=Path("data/backtests/player_game_logs_history.csv"))
    ap.add_argument("--team-weekly", type=Path, default=Path("data/backtests/team_weekly_history.csv"))
    ap.add_argument("--schedule", type=Path, default=Path("data/backtests/schedule_history.csv"))
    ap.add_argument("--universe-dir", type=Path, default=Path("data/backtests/pregame_universe"))
    ap.add_argument("--injuries", type=Path, default=Path("data/backtests/injuries_history.csv"))
    ap.add_argument("--weather", type=Path, default=Path("data/backtests/weather_history.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_receiving_decomposition_current_v1"))
    a = ap.parse_args()

    logs = read(a.player_logs)
    team = read(a.team_weekly)
    schedule = read(a.schedule)
    injuries = optional(a.injuries)
    weather = optional(a.weather)
    rows: list[dict] = []
    team_rows: list[dict] = []

    for week in _parse_weeks(a.weeks):
        universe = read(a.universe_dir / f"{a.season}_week_{week:02d}.csv")
        baseline = _build_baseline(
            logs=logs, team=team, schedule=schedule, injuries=injuries, weather=weather,
            universe=universe, season=a.season, week=week, prior_season=a.prior_season,
        )
        actual = cp.build_actual_rows(logs, a.season, week)
        at = actual.loc[actual.market.eq("receptions"), ["team", "player_clean_key", "actual_opportunities"]].rename(columns={"actual_opportunities": "actual_targets"})
        ay = actual.loc[actual.market.eq("rec_yards"), ["team", "player_clean_key", "actual"]].rename(columns={"actual": "actual_rec_yards"})
        at["actual_targets"] = pd.to_numeric(at["actual_targets"], errors="coerce").fillna(0.0)
        ay["actual_rec_yards"] = pd.to_numeric(ay["actual_rec_yards"], errors="coerce")

        x = baseline.merge(at, on=["team", "player_clean_key"], how="left").merge(ay, on=["team", "player_clean_key"], how="left")
        x["actual_targets"] = pd.to_numeric(x["actual_targets"], errors="coerce").fillna(0.0)
        x["position_family"] = x.get("position", "").astype(str).str.upper().str.strip().replace({"HB": "RB", "TB": "RB"})
        x["entitlement_tgt_share"] = pd.to_numeric(x["entitlement_tgt_share"], errors="coerce").fillna(0.0)
        x["rules_ypt_used"] = pd.to_numeric(x.get("rules_ypt", np.nan), errors="coerce")

        for (event_id, tm), g in x.groupby(["event_id", "team"], dropna=False, sort=False):
            rb = g.loc[g.position_family.isin({"RB", "FB"})].copy()
            if rb.empty:
                continue
            plays = float(np.mean([finite(v, 64.0) for v in g.get("rules_plays_est", pd.Series([64.0] * len(g)))]))
            pass_rate = float(np.mean([finite(v, 0.57) for v in g.get("rules_pass_rate", pd.Series([0.57] * len(g)))]))
            pred_team_pass = plays * pass_rate
            pred_modeled_targets = pred_team_pass * MODELED_MASS

            actual_modeled_targets = float(pd.to_numeric(g.actual_targets, errors="coerce").fillna(0.0).sum())
            actual_rb_targets = float(pd.to_numeric(rb.actual_targets, errors="coerce").fillna(0.0).sum())
            baseline_rb_mass = float(rb.entitlement_tgt_share.sum())
            baseline_rb_room_fraction = baseline_rb_mass / MODELED_MASS if MODELED_MASS > 0 else 0.0
            actual_rb_room_fraction = actual_rb_targets / actual_modeled_targets if actual_modeled_targets > 0 else 0.0

            team_rows.append({
                "season": a.season, "week": week, "event_id": event_id, "team": tm,
                "pred_team_pass_attempts": pred_team_pass,
                "pred_modeled_player_targets": pred_modeled_targets,
                "actual_modeled_player_targets": actual_modeled_targets,
                "baseline_rb_target_mass": baseline_rb_mass,
                "baseline_rb_room_fraction_of_modeled": baseline_rb_room_fraction,
                "actual_rb_targets": actual_rb_targets,
                "actual_rb_room_fraction_of_modeled": actual_rb_room_fraction,
                "sportsbook_inputs_used": 0,
            })

            base_within = (
                rb.entitlement_tgt_share.to_numpy(float) / baseline_rb_mass
                if baseline_rb_mass > 0 else np.full(len(rb), 1.0 / len(rb))
            )
            actual_t = rb.actual_targets.to_numpy(float)
            actual_within = actual_t / actual_rb_targets if actual_rb_targets > 0 else base_within.copy()
            actual_player_share = actual_t / actual_modeled_targets if actual_modeled_targets > 0 else np.zeros(len(rb), dtype=float)

            for j, (_, r) in enumerate(rb.iterrows()):
                ypt = finite(r.get("rules_ypt_used"), 7.0)
                base_targets = pred_team_pass * float(r.entitlement_tgt_share)
                team_volume_oracle_targets = actual_modeled_targets * (float(r.entitlement_tgt_share) / MODELED_MASS)
                rb_room_mass_oracle_targets = pred_modeled_targets * actual_rb_room_fraction * float(base_within[j])
                within_rb_oracle_targets = pred_modeled_targets * baseline_rb_room_fraction * float(actual_within[j])
                full_share_oracle_targets = pred_modeled_targets * float(actual_player_share[j])
                actual_targets = float(actual_t[j])
                rows.append({
                    "season": a.season, "week": week, "event_id": event_id, "team": tm,
                    "player": r.get("player", ""), "player_clean_key": r.player_clean_key,
                    "position": r.get("position", ""),
                    "actual_targets": actual_targets,
                    "actual_rec_yards": r.get("actual_rec_yards", np.nan),
                    "rules_ypt": ypt,
                    "baseline_entitlement_tgt_share": float(r.entitlement_tgt_share),
                    "baseline_within_rb_room_share": float(base_within[j]),
                    "actual_within_rb_room_share": float(actual_within[j]),
                    "baseline_targets": base_targets,
                    "team_volume_oracle_targets": team_volume_oracle_targets,
                    "rb_room_mass_oracle_targets": rb_room_mass_oracle_targets,
                    "within_rb_oracle_targets": within_rb_oracle_targets,
                    "full_player_share_oracle_targets": full_share_oracle_targets,
                    "actual_targets_oracle_targets": actual_targets,
                    "baseline_rec_yards": base_targets * ypt,
                    "team_volume_oracle_rec_yards": team_volume_oracle_targets * ypt,
                    "rb_room_mass_oracle_rec_yards": rb_room_mass_oracle_targets * ypt,
                    "within_rb_oracle_rec_yards": within_rb_oracle_targets * ypt,
                    "full_player_share_oracle_rec_yards": full_share_oracle_targets * ypt,
                    "actual_targets_current_ypt_rec_yards": actual_targets * ypt,
                    "sportsbook_inputs_used": 0,
                    "oracle_current_game_outcomes_used": 1,
                })
        print(f"[rb-recv-decomp] week={week:02d} complete")

    player = pd.DataFrame(rows)
    teams = pd.DataFrame(team_rows)
    if player.empty:
        raise RuntimeError("RB receiving decomposition produced zero player rows")

    stages = [
        ("baseline", "baseline_targets", "baseline_rec_yards"),
        ("team_volume_oracle", "team_volume_oracle_targets", "team_volume_oracle_rec_yards"),
        ("rb_room_mass_oracle", "rb_room_mass_oracle_targets", "rb_room_mass_oracle_rec_yards"),
        ("within_rb_oracle", "within_rb_oracle_targets", "within_rb_oracle_rec_yards"),
        ("full_player_share_oracle", "full_player_share_oracle_targets", "full_player_share_oracle_rec_yards"),
        ("actual_targets_current_ypt", "actual_targets_oracle_targets", "actual_targets_current_ypt_rec_yards"),
    ]
    summary_rows = []
    for label, tgt_col, yd_col in stages:
        summary_rows.append({"stage": label, "market": "targets", **_metric(player.actual_targets, player[tgt_col])})
        summary_rows.append({"stage": label, "market": "rec_yards", **_metric(player.actual_rec_yards, player[yd_col])})
    summary = pd.DataFrame(summary_rows)

    # Phase and baseline-volume buckets are diagnostics, not decision gates.
    player["phase"] = pd.cut(player.week, [0, 4, 9, 13, 18], labels=["W1-4", "W5-9", "W10-13", "W14-18"])
    bucket_rows = []
    for phase, g in player.groupby("phase", observed=False):
        if g.empty:
            continue
        for label, tgt_col, yd_col in stages:
            bucket_rows.append({"bucket": str(phase), "stage": label, "market": "rec_yards", **_metric(g.actual_rec_yards, g[yd_col])})
    buckets = pd.DataFrame(bucket_rows)

    base_t = summary.loc[(summary.stage == "baseline") & (summary.market == "targets")].iloc[0]
    base_y = summary.loc[(summary.stage == "baseline") & (summary.market == "rec_yards")].iloc[0]
    attribution = []
    for label, _, _ in stages[1:]:
        t = summary.loc[(summary.stage == label) & (summary.market == "targets")].iloc[0]
        y = summary.loc[(summary.stage == label) & (summary.market == "rec_yards")].iloc[0]
        attribution.append({
            "oracle_stage": label,
            "target_mae_gain_vs_baseline": float(base_t.mae - t.mae),
            "rec_yards_mae_gain_vs_baseline": float(base_y.mae - y.mae),
            "rec_yards_p90_gain_vs_baseline": float(base_y.p90_abs_error - y.p90_abs_error),
        })
    attribution = pd.DataFrame(attribution)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    player.to_csv(a.out_dir / "rb_receiving_player_decomposition.csv", index=False)
    teams.to_csv(a.out_dir / "rb_receiving_team_room_diagnostics.csv", index=False)
    summary.to_csv(a.out_dir / "rb_receiving_stage_summary.csv", index=False)
    buckets.to_csv(a.out_dir / "rb_receiving_phase_summary.csv", index=False)
    attribution.to_csv(a.out_dir / "rb_receiving_oracle_headroom.csv", index=False)
    print("\n[rb-recv-decomp] stage summary\n", summary.to_string(index=False))
    print("\n[rb-recv-decomp] oracle headroom\n", attribution.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
