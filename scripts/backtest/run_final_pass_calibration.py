#!/usr/bin/env python3
"""Final leakage-safe pass-tendency calibration gate before production promotion.

Tests fixed league-centered pass shares and very small historical team-identity
adjustments. No PROE or game-state correction is used in candidate variants.
Production project_game_script() remains unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_actual_rows, build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.run_pass_tendency_recalibration import rolling_dropback_baselines
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling import simulation_rules
from scripts.modeling.rules_v2 import GameScriptProjection, estimate_plays, offensive_pressure_mismatch, script_distribution, success_diff

FIXED_SHARES = {
    "fixed_53": 0.53,
    "fixed_54": 0.54,
    "fixed_55": 0.55,
    "fixed_56": 0.56,
    "fixed_57": 0.57,
}
TEAM_IDENTITY_WEIGHTS = {
    "team_identity_05": 0.05,
    "team_identity_10": 0.10,
    "team_identity_15": 0.15,
}
VARIANTS = ["current_full", *FIXED_SHARES, *TEAM_IDENTITY_WEIGHTS]


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def calibrated_share(variant: str, *, team_rate: float, league_rate: float) -> float:
    if variant in FIXED_SHARES:
        return float(FIXED_SHARES[variant])
    if variant in TEAM_IDENTITY_WEIGHTS:
        weight = float(TEAM_IDENTITY_WEIGHTS[variant])
        team_rate = float(team_rate) if np.isfinite(team_rate) else float(league_rate)
        league_rate = float(league_rate) if np.isfinite(league_rate) else 0.55
        # Keep the entire league centered on the empirically strong 55% baseline;
        # historical data contributes only the team's deviation from its league.
        return float(np.clip(0.55 + weight * (team_rate - league_rate), 0.42, 0.70))
    raise ValueError(f"unknown final calibration variant: {variant}")


def _project_week(variant: str, *, player_logs, team_weekly, schedule, universe, injuries, weather, season, week, prior_season, iterations, seed):
    team_rates, league = rolling_dropback_baselines(team_weekly, season, week, prior_season)
    bundle = build_historical_context_bundle(
        player_logs=player_logs, team_weekly=team_weekly, pregame_universe=universe,
        schedule=schedule, season=season, week=week, prior_season=prior_season,
        injuries=injuries, weather=weather,
    )

    if variant == "current_full":
        mc = build_mc_predictions(bundle, iterations=iterations, seed=seed)
    else:
        def patched(offense, defense):
            share = calibrated_share(
                variant,
                team_rate=team_rates.get(str(offense.team), league),
                league_rate=league,
            )
            diff = success_diff(offense, defense)
            lead, neutral, trail = script_distribution(diff)
            plays = estimate_plays(offense)
            pressure = offensive_pressure_mismatch(offense, defense)
            return GameScriptProjection(
                projected_plays=plays,
                projected_pass_attempts=plays * share,
                projected_rush_attempts=plays * (1.0 - share),
                lead_prob=lead,
                neutral_prob=neutral,
                trail_prob=trail,
                pressure_mismatch=abs(pressure) >= 0.05,
                blowout_risk=abs(diff) >= 0.06,
                shootout_risk=abs(diff) < 0.03 and plays >= 68.0,
            )
        with patch.object(simulation_rules, "project_game_script", side_effect=patched):
            mc = build_mc_predictions(bundle, iterations=iterations, seed=seed)

    actual = build_actual_rows(player_logs, season, week)
    keep = [c for c in [
        "player", "player_clean_key", "team", "opponent", "position", "market",
        "mc_proj", "mc_expected_pass_attempts",
    ] if c in mc.columns]
    out = mc[keep].merge(actual, on=["team", "player_clean_key", "market"], how="inner", validate="one_to_one")
    out.insert(0, "variant", variant)
    out.insert(1, "season", int(season))
    out.insert(2, "week", int(week))
    return out


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, market), g in predictions.groupby(["variant", "market"], sort=True):
        a = pd.to_numeric(g["actual"], errors="coerce")
        p = pd.to_numeric(g["mc_proj"], errors="coerce")
        ok = a.notna() & p.notna(); a, p = a[ok], p[ok]
        if not len(a):
            continue
        err = p - a
        rows.append({
            "variant": variant, "market": market, "n": int(len(a)),
            "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(err.mean()),
            "correlation": float(a.corr(p)) if len(a) > 1 else np.nan,
        })
    s = pd.DataFrame(rows)
    if s.empty:
        return s
    base = s[s.variant.eq("current_full")][["market", "mae", "rmse"]].rename(columns={"mae":"current_mae", "rmse":"current_rmse"})
    s = s.merge(base, on="market", how="left", validate="many_to_one")
    s["delta_mae_vs_current"] = s.mae - s.current_mae
    s["delta_rmse_vs_current"] = s.rmse - s.current_rmse
    return s.sort_values(["market", "mae"]).reset_index(drop=True)


def recommendation(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates on pass-yards MAE with simple skill-market guardrails."""
    if summary.empty:
        return pd.DataFrame()
    piv = summary.pivot(index="variant", columns="market", values="mae")
    current = piv.loc["current_full"] if "current_full" in piv.index else pd.Series(dtype=float)
    rows = []
    for variant, r in piv.iterrows():
        if variant == "current_full" or "pass_yards" not in r.index or pd.isna(r.get("pass_yards")):
            continue
        rec_delta = float(r.get("rec_yards", np.nan) - current.get("rec_yards", np.nan))
        receptions_delta = float(r.get("receptions", np.nan) - current.get("receptions", np.nan))
        rush_delta = float(r.get("rush_yards", np.nan) - current.get("rush_yards", np.nan))
        guardrail = (
            (not np.isfinite(rec_delta) or rec_delta <= 0.15)
            and (not np.isfinite(receptions_delta) or receptions_delta <= 0.02)
            and (not np.isfinite(rush_delta) or rush_delta <= 0.10)
        )
        rows.append({
            "variant": variant,
            "pass_yards_mae": float(r["pass_yards"]),
            "pass_yards_delta_vs_current": float(r["pass_yards"] - current.get("pass_yards", np.nan)),
            "rec_yards_delta_vs_current": rec_delta,
            "receptions_delta_vs_current": receptions_delta,
            "rush_yards_delta_vs_current": rush_delta,
            "passes_guardrails": int(guardrail),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["passes_guardrails", "pass_yards_mae"], ascending=[False, True]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--prior-season", type=int, default=2024)
    p.add_argument("--weeks", default="1-18")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--player-logs", type=Path, default=Path("data/backtests/player_game_logs_history.csv"))
    p.add_argument("--team-weekly", type=Path, default=Path("data/backtests/team_weekly_history.csv"))
    p.add_argument("--schedule", type=Path, default=Path("data/backtests/schedule_history.csv"))
    p.add_argument("--universe-dir", type=Path, default=Path("data/backtests/pregame_universe"))
    p.add_argument("--injuries", type=Path, default=Path("data/backtests/injuries_history.csv"))
    p.add_argument("--weather", type=Path, default=Path("data/backtests/weather_history.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/final_pass_calibration"))
    a = p.parse_args()

    logs = _read(a.player_logs, "player logs")
    team = _read(a.team_weekly, "team weekly")
    sched = _read(a.schedule, "schedule")
    inj = pd.read_csv(a.injuries) if a.injuries.exists() and a.injuries.stat().st_size else pd.DataFrame()
    wx = pd.read_csv(a.weather) if a.weather.exists() and a.weather.stat().st_size else pd.DataFrame()
    rows = []
    for week in _parse_weeks(a.weeks):
        universe = _read(a.universe_dir / f"{a.season}_week_{week:02d}.csv", f"universe W{week}")
        injuries = _exact_week(inj, a.season, week)
        weather = _exact_week(wx, a.season, week)
        for variant in VARIANTS:
            rows.append(_project_week(
                variant, player_logs=logs, team_weekly=team, schedule=sched,
                universe=universe, injuries=injuries, weather=weather,
                season=a.season, week=week, prior_season=a.prior_season,
                iterations=a.iterations, seed=18000 + week,
            ))
            print(f"[final-pass-cal] W{week:02d} {variant}")
    predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary = summarize(predictions)
    rec = recommendation(summary)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(a.out_dir / "final_pass_calibration_predictions.csv", index=False)
    summary.to_csv(a.out_dir / "final_pass_calibration_summary.csv", index=False)
    rec.to_csv(a.out_dir / "final_pass_calibration_recommendation.csv", index=False)
    print("\n[final-pass-cal] summary")
    print(summary.to_string(index=False))
    print("\n[final-pass-cal] recommendation ranking")
    print(rec.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
