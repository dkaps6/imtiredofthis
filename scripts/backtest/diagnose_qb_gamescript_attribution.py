#!/usr/bin/env python3
"""Migration 52A: attribute the OOS game-script gain and inspect its tail behavior."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest.fit_qb_gamescript_attempts_walkforward import (
    add_lagged_context, moneyline_win_probability, metrics, num, read,
)


MARKET = ["market_spread", "market_abs_spread", "market_total", "market_team_implied", "market_opp_implied", "market_is_underdog", "pregame_win_probability", "expected_trailing_probability", "competitive_game", "total_x_trailing", "spread_x_total"]
PACE = ["team_neutral_pace", "team_plays_est", "opp_neutral_pace", "opp_plays_est"]
TEAM_TENDENCY = ["team_dropback_rate", "team_proe", "team_success_rate_off"]
OPP_OFFENSE = ["opp_dropback_rate", "opp_success_rate_off"]
FORCE_PASS = ["opp_pressure_rate_generated", "opp_def_pass_epa", "opp_success_rate_def", "opponent_force_pass"]


def prepare(market_trace: Path, team_weekly: Path, season: int) -> pd.DataFrame:
    x = read(market_trace)
    x = x[num(x.get("market_total")).notna()].copy().reset_index(drop=True)
    x = add_lagged_context(x, read(team_weekly), season)
    x["market_win_probability"] = moneyline_win_probability(x.get("market_moneyline"))
    spread_prob = 1.0 / (1.0 + np.exp(num(x.market_spread) / 6.5))
    x["pregame_win_probability"] = x.market_win_probability.fillna(spread_prob)
    x["expected_trailing_probability"] = 1.0 - x.pregame_win_probability
    x["competitive_game"] = (num(x.market_abs_spread) <= 3.0).astype(float)
    x["total_x_trailing"] = num(x.market_total) * x.expected_trailing_probability
    x["spread_x_total"] = num(x.market_spread) * num(x.market_total)
    x["opponent_force_pass"] = num(x.opp_pressure_rate_generated).fillna(0) + num(x.opp_def_pass_epa).fillna(0) + num(x.opp_dropback_rate).fillna(0)
    return x


def fit_residual(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> pd.Series:
    if not features:
        raw = pd.Series(float((num(train.actual_pass_att) - num(train.pred_attempts)).mean()), index=test.index)
    else:
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=30.0))
        model.fit(train[features], num(train.actual_pass_att) - num(train.pred_attempts))
        raw = pd.Series(model.predict(test[features]), index=test.index)
    return (num(test.pred_attempts) + raw.clip(-5, 5) * 0.60).clip(18, 48)


def tail_row(name: str, x: pd.DataFrame, pred_col: str) -> dict:
    pred = num(x[pred_col]); actual = num(x.actual_pass_yards_raw)
    err = pred - actual; current_err = num(x.pass_yards_current) - actual
    high = num(x.actual_pass_att) >= 40
    try: auc = float(roc_auc_score(high.astype(int), num(x[pred_col.replace("pass_yards", "attempts")]))) if high.nunique() > 1 else np.nan
    except Exception: auc = np.nan
    return {
        "candidate": name, "n": len(x), "mae": float(err.abs().mean()), "rmse": float(np.sqrt(np.mean(err * err))),
        "bias": float(err.mean()), "correlation": float(pred.corr(actual)),
        "catastrophic_100plus": int(err.abs().ge(100).sum()), "under_100plus": int(err.le(-100).sum()), "over_100plus": int(err.ge(100).sum()),
        "catastrophic_mae": float(err.loc[current_err.abs().ge(100)].abs().mean()),
        "rows_improved_vs_current": int(err.abs().lt(current_err.abs()).sum()),
        "rows_worsened_vs_current": int(err.abs().gt(current_err.abs()).sum()),
        "actual_40plus_games": int(high.sum()), "attempt_40plus_auc": auc,
        "attempt_mae_actual_40plus": float((num(x[pred_col.replace("pass_yards", "attempts")]) - num(x.actual_pass_att)).loc[high].abs().mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--market-trace", type=Path, default=Path("data/backtests/qb_gamescript_market/qb_gamescript_market_trace.csv"))
    p.add_argument("--team-weekly", type=Path, default=Path("data/backtests/team_weekly_history.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/qb_gamescript_attribution"))
    p.add_argument("--min-train", type=int, default=80)
    a = p.parse_args()
    x = prepare(a.market_trace, a.team_weekly, a.season)
    groups = {
        "bias_only": [], "market": MARKET, "pace": PACE, "team_tendency": TEAM_TENDENCY,
        "opponent_offense": OPP_OFFENSE, "force_pass": FORCE_PASS,
        "full_capped": MARKET + ["pred_attempts"] + PACE + TEAM_TENDENCY + OPP_OFFENSE + FORCE_PASS,
    }
    x["attempts_current"] = np.nan
    for name in groups: x[f"attempts_{name}"] = np.nan
    scored = []
    for week in sorted(num(x.week).dropna().astype(int).unique()):
        train, test = x[num(x.week) < week], x[num(x.week) == week]
        if len(train) < a.min_train or test.empty: continue
        x.loc[test.index, "attempts_current"] = num(test.pred_attempts)
        for name, features in groups.items():
            x.loc[test.index, f"attempts_{name}"] = fit_residual(train, test, features)
        scored.extend(test.index.tolist())
    x = x.loc[sorted(set(scored))].copy()
    if x.empty: raise RuntimeError("no out-of-sample attribution rows")
    candidates = ["current", *groups]
    summary = []
    for name in candidates:
        ap = num(x[f"attempts_{name}"])
        x[f"pass_yards_{name}"] = num(x.mc_proj) * ap / num(x.pred_attempts).replace(0, np.nan)
        summary.append({"candidate": name, "metric": "attempts", **metrics(x.actual_pass_att, ap)})
        summary.append({"candidate": name, "metric": "pass_yards", **metrics(x.actual_pass_yards_raw, x[f"pass_yards_{name}"])})
    tails = pd.DataFrame([tail_row(name, x, f"pass_yards_{name}") for name in candidates])
    weeks = []
    for week, g in x.groupby("week"):
        cur = metrics(g.actual_pass_yards_raw, g.pass_yards_current)
        for name in candidates:
            m = metrics(g.actual_pass_yards_raw, g[f"pass_yards_{name}"])
            weeks.append({"week": int(week), "candidate": name, **m, "mae_improvement_vs_current": cur["mae"] - m["mae"]})
    weekly = pd.DataFrame(weeks)
    stability = weekly.groupby("candidate").agg(weeks=("week", "count"), weeks_won=("mae_improvement_vs_current", lambda s: int((s > 0).sum())), mean_weekly_improvement=("mae_improvement_vs_current", "mean"), worst_week=("mae_improvement_vs_current", "min"), best_week=("mae_improvement_vs_current", "max")).reset_index()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "qb_gamescript_attribution_trace.csv", index=False)
    pd.DataFrame(summary).to_csv(a.out_dir / "qb_gamescript_ablation_summary.csv", index=False)
    tails.to_csv(a.out_dir / "qb_gamescript_tail_summary.csv", index=False)
    weekly.to_csv(a.out_dir / "qb_gamescript_weekly_summary.csv", index=False)
    stability.to_csv(a.out_dir / "qb_gamescript_weekly_stability.csv", index=False)
    print("=== ABLATION SUMMARY ==="); print(pd.DataFrame(summary).to_string(index=False))
    print("\n=== TAIL SUMMARY ==="); print(tails.to_string(index=False))
    print("\n=== WEEKLY STABILITY ==="); print(stability.to_string(index=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
