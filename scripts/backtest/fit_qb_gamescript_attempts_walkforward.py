#!/usr/bin/env python3
"""Migration 51: leakage-safe walk-forward QB game-script attempts candidates.

Stable-primary QB rows are scored only with models fit on earlier target weeks.
Pregame features combine betting markets with lagged team/opponent PBP context.
No target-game outcome is used to construct a feature.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {path}")
    return pd.read_csv(path)


def num(value) -> pd.Series:
    return pd.to_numeric(value, errors="coerce")


def metrics(actual, predicted) -> dict:
    z = pd.DataFrame({"actual": num(actual), "pred": num(predicted)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan}
    error = z.pred - z.actual
    return {
        "n": len(z), "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(error * error))), "bias": float(error.mean()),
        "correlation": float(z.pred.corr(z.actual)) if len(z) > 2 and z.pred.nunique() > 1 else np.nan,
    }


def moneyline_win_probability(ml: pd.Series) -> pd.Series:
    x = num(ml)
    return pd.Series(np.where(x < 0, -x / (-x + 100.0), 100.0 / (x + 100.0)), index=x.index).where(x.notna())


def add_lagged_context(trace: pd.DataFrame, history: pd.DataFrame, season: int) -> pd.DataFrame:
    h = history.copy()
    h.columns = [str(c).strip().lower() for c in h.columns]
    h["season"] = num(h.season); h["week"] = num(h.week)
    wanted = [
        "neutral_pace", "plays_est", "dropback_rate", "proe", "success_rate_off",
        "pressure_rate_generated", "def_pass_epa", "success_rate_def",
    ]
    rows = []
    for _, r in trace.iterrows():
        prior = h[(h.season < season) | ((h.season == season) & (h.week < int(r.week)))]
        rec = {"_row": r.name}
        for prefix, club in (("team", r.team), ("opp", r.opponent)):
            g = prior[prior.team.astype(str).str.upper().eq(str(club).upper())].sort_values(["season", "week"]).tail(8)
            weights = np.arange(1, len(g) + 1, dtype=float)
            for col in wanted:
                v = num(g[col]) if col in g else pd.Series(dtype=float)
                ok = v.notna()
                rec[f"{prefix}_{col}"] = float(np.average(v[ok], weights=weights[ok])) if ok.any() else np.nan
        rows.append(rec)
    lag = pd.DataFrame(rows).set_index("_row")
    return trace.join(lag)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--market-trace", type=Path, default=Path("data/backtests/qb_gamescript_market/qb_gamescript_market_trace.csv"))
    p.add_argument("--team-weekly", type=Path, default=Path("data/backtests/team_weekly_history.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/qb_gamescript_attempts_walkforward"))
    p.add_argument("--min-train", type=int, default=80)
    a = p.parse_args()

    x = read(a.market_trace)
    x = x[num(x.get("market_total")).notna()].copy().reset_index(drop=True)
    x = add_lagged_context(x, read(a.team_weekly), a.season)
    x["market_win_probability"] = moneyline_win_probability(x.get("market_moneyline"))
    spread_prob = 1.0 / (1.0 + np.exp(num(x.market_spread) / 6.5))
    x["pregame_win_probability"] = x.market_win_probability.fillna(spread_prob)
    x["expected_trailing_probability"] = 1.0 - x.pregame_win_probability
    x["competitive_game"] = (num(x.market_abs_spread) <= 3.0).astype(float)
    x["total_x_trailing"] = num(x.market_total) * x.expected_trailing_probability
    x["spread_x_total"] = num(x.market_spread) * num(x.market_total)
    x["opponent_force_pass"] = (
        num(x.opp_pressure_rate_generated).fillna(0)
        + num(x.opp_def_pass_epa).fillna(0)
        + num(x.opp_dropback_rate).fillna(0)
    )

    market = ["market_spread", "market_abs_spread", "market_total", "market_team_implied", "market_opp_implied", "market_is_underdog", "pregame_win_probability", "expected_trailing_probability", "competitive_game", "total_x_trailing", "spread_x_total"]
    history = ["pred_attempts", "team_neutral_pace", "team_plays_est", "team_dropback_rate", "team_proe", "team_success_rate_off", "opp_neutral_pace", "opp_plays_est", "opp_dropback_rate", "opp_success_rate_off", "opp_pressure_rate_generated", "opp_def_pass_epa", "opp_success_rate_def", "opponent_force_pass"]
    candidates = {"market_only": market, "history_plus_market": market + history}
    for c in ["current", "market_only", "history_plus_market", "capped_residual"]:
        x[f"attempts_{c}"] = np.nan

    scored = []
    for week in sorted(num(x.week).dropna().astype(int).unique()):
        train = x[num(x.week) < week].copy(); test = x[num(x.week) == week].copy()
        if len(train) < a.min_train or test.empty:
            continue
        x.loc[test.index, "attempts_current"] = num(test.pred_attempts)
        for name, features in candidates.items():
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=20.0))
            model.fit(train[features], num(train.actual_pass_att))
            pred = pd.Series(model.predict(test[features]), index=test.index).clip(18, 48)
            x.loc[test.index, f"attempts_{name}"] = pred
        residual_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=30.0))
        residual_model.fit(train[market + history], num(train.actual_pass_att) - num(train.pred_attempts))
        delta = pd.Series(residual_model.predict(test[market + history]), index=test.index).clip(-5, 5) * 0.60
        x.loc[test.index, "attempts_capped_residual"] = (num(test.pred_attempts) + delta).clip(18, 48)
        scored.extend(test.index.tolist())

    x = x.loc[sorted(set(scored))].copy()
    if x.empty:
        raise RuntimeError("no out-of-sample rows; lower --min-train or provide more weeks")
    summary = []
    for candidate in ["current", "market_only", "history_plus_market", "capped_residual"]:
        ap = x[f"attempts_{candidate}"]
        # The existing mc_proj is canonical. Rescaling its expectation by the OOS
        # attempt ratio isolates the downstream volume change while holding YPA fixed.
        x[f"pass_yards_{candidate}"] = num(x.mc_proj) * ap / num(x.pred_attempts).replace(0, np.nan)
        summary.append({"candidate": candidate, "metric": "attempts", **metrics(x.actual_pass_att, ap)})
        summary.append({"candidate": candidate, "metric": "pass_yards", **metrics(x.actual_pass_yards_raw, x[f"pass_yards_{candidate}"])})
    s = pd.DataFrame(summary)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "qb_gamescript_attempts_walkforward_trace.csv", index=False)
    s.to_csv(a.out_dir / "qb_gamescript_attempts_walkforward_summary.csv", index=False)
    coverage = pd.DataFrame({"feature": market + history, "non_null": [int(x[c].notna().sum()) for c in market + history], "n": len(x)})
    coverage["coverage"] = coverage.non_null / coverage.n
    coverage.to_csv(a.out_dir / "qb_gamescript_feature_coverage.csv", index=False)
    print("=== WALK-FORWARD SUMMARY ==="); print(s.to_string(index=False))
    print("\n=== FEATURE COVERAGE ==="); print(coverage.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
