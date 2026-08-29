#!/usr/bin/env python3
"""Migration 66 sub-audit: Monte Carlo mean versus median for QB passing.

Replays the canonical current and M59 Raw-Attempts passing simulations on the
stable-QB target rows and records the simulation mean and median. Diagnostic
only; the production point statistic is untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest.diagnose_qb_raw_tail_attribution_mc import clean_factor, wrapper
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling import simulation_rules

KEYS = ["week", "team", "player_clean_key"]


def read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def opt(path: Path) -> pd.DataFrame:
    return read(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def num(v):
    return pd.to_numeric(v, errors="coerce")


def simulate_point_stats(bundle, rules_fn, *, iterations: int, seed: int) -> pd.DataFrame:
    metrics = cp.build_market_frame(bundle)
    bayes = cp.build_bayesian_baseline(bundle.player_consensus)
    metrics = cp.apply_bayesian_to_metrics(metrics, bayes)
    with patch.object(simulation_rules, "load_model_contexts", return_value=(bundle.teams, bundle.players)):
        metrics = rules_fn(metrics)
    if int(num(metrics["rules_applied"]).fillna(0).sum()) == 0:
        raise RuntimeError("M66 mean/median replay matched zero rules rows")

    metrics = cp._attach_historical_passing_volume(metrics, bundle)
    trace = cp._context_trace_frame(bundle)
    metrics = metrics.merge(trace, on=["team", "player_clean_key"], how="left", validate="many_to_one")
    metrics["mc_qb_pass_att_share"] = num(metrics.get("qb_pass_att_share"))

    sims = cp.simulate(metrics, iterations=int(iterations), seed=int(seed))
    rows = []
    for _, row in metrics[metrics.market.astype(str).eq("pass_yards")].iterrows():
        outcomes = cp.lookup(sims, row, "pass_yards")
        if outcomes is None or not len(outcomes):
            continue
        outcomes = np.asarray(outcomes, dtype=float)
        attempt_rate = num(pd.Series([row.get("mc_pass_attempts_per_dropback")])).iloc[0]
        share = num(pd.Series([row.get("qb_pass_att_share")])).iloc[0]
        if pd.notna(attempt_rate):
            outcomes = outcomes * float(np.clip(attempt_rate, 0.50, 1.00))
        if pd.notna(share):
            outcomes = outcomes * float(np.clip(share, 0.0, 1.0))
        rows.append({
            "team": str(row.team),
            "player_clean_key": str(row.player_clean_key),
            "mean": float(np.mean(outcomes)),
            "median": float(np.median(outcomes)),
            "p25": float(np.percentile(outcomes, 25)),
            "p75": float(np.percentile(outcomes, 75)),
            "sim_sd": float(np.std(outcomes, ddof=0)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--prior-season", type=int, required=True)
    p.add_argument("--weeks", default="1-18")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--master-game-level", type=Path, required=True)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--universe-dir", type=Path, required=True)
    p.add_argument("--injuries", type=Path, required=True)
    p.add_argument("--weather", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    master = read(a.master_game_level)
    master = master[num(master.season).eq(a.season)].copy()
    required = {
        "attempts_raw", "pred_attempts", "ypa_contextual", "ypa_current",
        "actual", "team", "player_clean_key", "week",
    }
    missing = required - set(master.columns)
    if missing:
        raise RuntimeError(f"M66 mean/median master missing columns: {sorted(missing)}")

    logs, tw, sched = read(a.player_logs), read(a.team_weekly), read(a.schedule)
    inj, weather = opt(a.injuries), opt(a.weather)
    original = simulation_rules.apply_rules_to_metrics
    rows = []

    weeks = [w for w in _parse_weeks(a.weeks) if w in set(num(master.week).dropna().astype(int))]
    for week in weeks:
        block = master[num(master.week).eq(week)].copy()
        if block.empty:
            continue
        rat = dict(zip(
            block.team.astype(str).str.upper().str.strip(),
            clean_factor(block.attempts_raw, block.pred_attempts),
        ))
        jyp = {
            (str(r.team).upper().strip(), str(r.player_clean_key)): float(f)
            for (_, r), f in zip(block.iterrows(), clean_factor(block.ypa_contextual, block.ypa_current, (.75, 1.25)))
        }
        raw_rules = wrapper(original, rat, jyp)

        universe = read(a.universe_dir / f"{a.season}_week_{week:02d}.csv")
        bundle = build_historical_context_bundle(
            player_logs=logs,
            team_weekly=tw,
            pregame_universe=universe,
            schedule=sched,
            season=a.season,
            week=week,
            prior_season=a.prior_season,
            injuries=_exact_week(inj, a.season, week),
            weather=_exact_week(weather, a.season, week),
        )

        current = simulate_point_stats(bundle, original, iterations=a.iterations, seed=53 + week)
        raw = simulate_point_stats(bundle, raw_rules, iterations=a.iterations, seed=53 + week)
        current = current.rename(columns={c: f"current_{c}" for c in ["mean", "median", "p25", "p75", "sim_sd"]})
        raw = raw.rename(columns={c: f"raw_{c}" for c in ["mean", "median", "p25", "p75", "sim_sd"]})
        z = block[["season", "week", "team", "player_clean_key", "actual", "mc_proj_attempts_raw_only"]].merge(
            current, on=["team", "player_clean_key"], how="left", validate="one_to_one"
        ).merge(raw, on=["team", "player_clean_key"], how="left", validate="one_to_one")
        if z[["current_mean", "current_median", "raw_mean", "raw_median"]].isna().any().any():
            raise RuntimeError(f"M66 mean/median missing simulation rows for {a.season} W{week}")
        # Exact replay check: same seed/context as M59 Raw-Attempts should agree.
        diff = (num(z.raw_mean) - num(z.mc_proj_attempts_raw_only)).abs().max()
        if np.isfinite(diff) and diff > 1e-8:
            raise RuntimeError(f"M66 raw mean replay mismatch {a.season} W{week}: max_abs_diff={diff}")
        rows.append(z)
        print(f"[m66 mean/median] {a.season} W{week:02d} stable_qbs={len(z)}")

    if not rows:
        raise RuntimeError("M66 mean/median produced no rows")
    out = pd.concat(rows, ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    for mode in ("current", "raw"):
        for stat in ("mean", "median"):
            pred = num(out[f"{mode}_{stat}"])
            actual = num(out.actual)
            err = pred - actual
            print(
                f"[m66 mean/median] season={a.season} mode={mode} stat={stat} "
                f"mae={err.abs().mean():.6f} rmse={np.sqrt(np.mean(np.square(err))):.6f} "
                f"corr={pred.corr(actual):.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
