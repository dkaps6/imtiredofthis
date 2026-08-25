#!/usr/bin/env python3
"""Migration 52B: run the capped game-script attempt candidate inside canonical MC."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_actual_rows, build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling import simulation_rules


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size: raise RuntimeError(f"missing {path}")
    return pd.read_csv(path)


def opt(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def num(value) -> pd.Series: return pd.to_numeric(value, errors="coerce")


def met(actual, pred) -> dict:
    z = pd.DataFrame({"a": num(actual), "p": num(pred)}).dropna(); e = z.p - z.a
    return {"n": len(z), "mae": float(e.abs().mean()), "rmse": float(np.sqrt(np.mean(e * e))), "bias": float(e.mean()), "correlation": float(z.p.corr(z.a)) if len(z) > 2 else np.nan, "catastrophic_100plus": int(e.abs().ge(100).sum()), "under_100plus": int(e.le(-100).sum()), "over_100plus": int(e.ge(100).sum())}


def wrapper(original, factors: dict[str, float]):
    def apply(metrics: pd.DataFrame) -> pd.DataFrame:
        out = original(metrics)
        team = out.team.astype(str).str.upper().str.strip()
        factor = team.map(factors).fillna(1.0)
        out["rules_pass_rate"] = (num(out.rules_pass_rate) * factor).clip(0.25, 0.85)
        return out
    return apply


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025); p.add_argument("--prior-season", type=int, default=2024); p.add_argument("--weeks", default="1-18"); p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--candidate-trace", type=Path, default=Path("data/backtests/qb_gamescript_attribution/qb_gamescript_attribution_trace.csv")); p.add_argument("--player-logs", type=Path, default=Path("data/backtests/player_game_logs_history.csv")); p.add_argument("--team-weekly", type=Path, default=Path("data/backtests/team_weekly_history.csv")); p.add_argument("--schedule", type=Path, default=Path("data/backtests/schedule_history.csv")); p.add_argument("--universe-dir", type=Path, default=Path("data/backtests/pregame_universe")); p.add_argument("--injuries", type=Path, default=Path("data/backtests/injuries_history.csv")); p.add_argument("--weather", type=Path, default=Path("data/backtests/weather_history.csv")); p.add_argument("--out-dir", type=Path, default=Path("data/backtests/qb_gamescript_canonical_mc"))
    a = p.parse_args(); cand = read(a.candidate_trace); logs = read(a.player_logs); team = read(a.team_weekly); sched = read(a.schedule); inj = opt(a.injuries); weather = opt(a.weather)
    original = simulation_rules.apply_rules_to_metrics; traces = []
    candidate_weeks = set(num(cand.week).dropna().astype(int))
    for week in [w for w in _parse_weeks(a.weeks) if w in candidate_weeks]:
        cw = cand[num(cand.week).eq(week)].copy()
        factors = dict(zip(cw.team.astype(str).str.upper().str.strip(), (num(cw.attempts_full_capped) / num(cw.attempts_current).replace(0, np.nan)).clip(.75, 1.25)))
        universe = read(a.universe_dir / f"{a.season}_week_{week:02d}.csv")
        bundle = build_historical_context_bundle(player_logs=logs, team_weekly=team, pregame_universe=universe, schedule=sched, season=a.season, week=week, prior_season=a.prior_season, injuries=_exact_week(inj, a.season, week), weather=_exact_week(weather, a.season, week))
        actual = build_actual_rows(logs, a.season, week)
        for mode in ["current", "full_capped"]:
            fn = original if mode == "current" else wrapper(original, factors)
            with patch.object(simulation_rules, "apply_rules_to_metrics", side_effect=fn): mc = build_mc_predictions(bundle, iterations=a.iterations, seed=52 + week)
            z = mc.merge(actual, on=["team", "player_clean_key", "market"], how="inner")
            z["candidate"] = mode; z["week"] = week; traces.append(z)
        print(f"[m52] W{week:02d} stable_qbs={len(cw)} adjusted_teams={len(factors)}")
    t = pd.concat(traces, ignore_index=True)
    keys = cand[["week", "team", "player_clean_key"]].drop_duplicates(); stable = t[t.market.eq("pass_yards")].merge(keys, on=["week", "team", "player_clean_key"], how="inner")
    summary = []
    for mode, g in stable.groupby("candidate"): summary.append({"candidate": mode, "slice": "stable_qb", "market": "pass_yards", **met(g.actual, g.mc_proj)})
    for (mode, market), g in t.groupby(["candidate", "market"]): summary.append({"candidate": mode, "slice": "all_available", "market": market, **met(g.actual, g.mc_proj)})
    s = pd.DataFrame(summary)
    a.out_dir.mkdir(parents=True, exist_ok=True); t.to_csv(a.out_dir / "qb_gamescript_canonical_mc_trace.csv", index=False); stable.to_csv(a.out_dir / "qb_gamescript_canonical_mc_stable_qbs.csv", index=False); s.to_csv(a.out_dir / "qb_gamescript_canonical_mc_summary.csv", index=False)
    print("=== CANONICAL MC SUMMARY ==="); print(s.to_string(index=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
