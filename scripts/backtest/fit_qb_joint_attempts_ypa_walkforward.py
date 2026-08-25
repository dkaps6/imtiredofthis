#!/usr/bin/env python3
"""Migration 53A: leakage-safe joint QB attempts and contextual-YPA candidates.

Evaluation remains restricted to actual primary QBs with at least 80 percent of
their team's attempts, inherited from the Migration 50 stable-primary trace.
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

from scripts.backtest.diagnose_qb_gamescript_attribution import (
    FORCE_PASS, MARKET, OPP_OFFENSE, PACE, TEAM_TENDENCY, fit_residual, prepare,
)
from scripts.backtest.fit_qb_gamescript_attempts_walkforward import metrics, num, read


def pick(frame: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def weighted_ratio(g: pd.DataFrame, numerator: str | None, denominator: str | None, weights: np.ndarray) -> float:
    if not numerator or not denominator: return np.nan
    n, d = num(g[numerator]), num(g[denominator]); ok = n.notna() & d.notna() & d.gt(0)
    return float(np.sum(n[ok] * weights[ok]) / np.sum(d[ok] * weights[ok])) if ok.any() else np.nan


def add_qb_and_matchup_context(x: pd.DataFrame, logs: pd.DataFrame, team_weekly: pd.DataFrame, weather: pd.DataFrame, season: int) -> pd.DataFrame:
    p = logs.copy(); p.columns = [str(c).strip().lower() for c in p.columns]
    h = team_weekly.copy(); h.columns = [str(c).strip().lower() for c in h.columns]
    p["season"], p["week"] = num(p.season), num(p.week); h["season"], h["week"] = num(h.season), num(h.week)
    att = pick(p, "pass_att", "attempts"); yards = pick(p, "pass_yards", "passing_yards")
    completions = pick(p, "completions", "passing_completions"); tds = pick(p, "pass_tds", "passing_tds"); ints = pick(p, "interceptions", "passing_interceptions"); epa = pick(p, "passing_epa", "pass_epa")
    if not att or not yards: raise RuntimeError("player logs require pass_att and pass_yards")
    extras = ["pressure_rate_allowed", "explosive_play_rate_allowed", "coverage_man_rate", "coverage_zone_rate"]
    rows = []
    for _, r in x.iterrows():
        prior_p = p[(p.season < season) | ((p.season == season) & (p.week < int(r.week)))]
        g = prior_p[prior_p.player_clean_key.astype(str).eq(str(r.player_clean_key))].sort_values(["season", "week"]).tail(8)
        w = np.arange(1, len(g) + 1, dtype=float)
        rec = {"_row": r.name}
        rec["qb_recent_ypa"] = weighted_ratio(g, yards, att, w)
        rec["qb_recent_completion_pct"] = weighted_ratio(g, completions, att, w)
        rec["qb_recent_td_rate"] = weighted_ratio(g, tds, att, w)
        rec["qb_recent_int_rate"] = weighted_ratio(g, ints, att, w)
        rec["qb_recent_epa_per_att"] = weighted_ratio(g, epa, att, w)
        av = num(g[att]) if len(g) else pd.Series(dtype=float); ok = av.notna()
        rec["qb_recent_pass_att"] = float(np.average(av[ok], weights=w[ok])) if ok.any() else np.nan
        prior_h = h[(h.season < season) | ((h.season == season) & (h.week < int(r.week)))]
        for prefix, club in (("team", r.team), ("opp", r.opponent)):
            tg = prior_h[prior_h.team.astype(str).str.upper().eq(str(club).upper())].sort_values(["season", "week"]).tail(8); tw = np.arange(1, len(tg) + 1, dtype=float)
            for col in extras:
                v = num(tg[col]) if col in tg else pd.Series(dtype=float); good = v.notna()
                rec[f"{prefix}_{col}"] = float(np.average(v[good], weights=tw[good])) if good.any() else np.nan
        rows.append(rec)
    out = x.join(pd.DataFrame(rows).set_index("_row"))
    if weather is not None and not weather.empty:
        z = weather.copy(); z.columns = [str(c).strip().lower() for c in z.columns]
        wr = []
        for _, r in z.iterrows():
            for club in [r.get("home"), r.get("away")]: wr.append({"season": r.get("season"), "week": r.get("week"), "team": club, "controlled_environment": r.get("controlled_environment")})
        out = out.merge(pd.DataFrame(wr).drop_duplicates(["season", "week", "team"]), on=["season", "week", "team"], how="left", validate="many_to_one")
    else: out["controlled_environment"] = np.nan
    return out


def fit_ypa(train: pd.DataFrame, test: pd.DataFrame, features: list[str], *, bias_only: bool = False) -> pd.Series:
    target = num(train.actual_ypa) - num(train.pred_ypa)
    usable = [f for f in features if f in train and num(train[f]).notna().sum() >= 10 and num(train[f]).nunique() > 1]
    if bias_only or not usable:
        delta = pd.Series(float(target.mean()), index=test.index)
    else:
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=30.0))
        model.fit(train[usable], target)
        delta = pd.Series(model.predict(test[usable]), index=test.index)
    return (num(test.pred_ypa) + delta.clip(-1.5, 1.5) * 0.60).clip(4.5, 10.5)


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--season", type=int, default=2025)
    p.add_argument("--market-trace", type=Path, default=Path("data/backtests/qb_gamescript_market/qb_gamescript_market_trace.csv")); p.add_argument("--team-weekly", type=Path, default=Path("data/backtests/team_weekly_history.csv")); p.add_argument("--player-logs", type=Path, default=Path("data/backtests/player_game_logs_history.csv")); p.add_argument("--weather", type=Path, default=Path("data/backtests/weather_history.csv")); p.add_argument("--out-dir", type=Path, default=Path("data/backtests/qb_joint_attempts_ypa")); p.add_argument("--min-train", type=int, default=80); a = p.parse_args()
    x = prepare(a.market_trace, a.team_weekly, a.season)
    x = add_qb_and_matchup_context(x, read(a.player_logs), read(a.team_weekly), read(a.weather), a.season)
    attempt_features = MARKET + ["pred_attempts"] + PACE + TEAM_TENDENCY + OPP_OFFENSE + FORCE_PASS
    qb_features = ["pred_ypa", "qb_recent_ypa", "qb_recent_pass_att", "qb_recent_completion_pct", "qb_recent_td_rate", "qb_recent_int_rate", "qb_recent_epa_per_att"]
    matchup_features = ["team_pressure_rate_allowed", "team_success_rate_off", "opp_pressure_rate_generated", "opp_def_pass_epa", "opp_success_rate_def", "opp_explosive_play_rate_allowed", "opp_coverage_man_rate", "opp_coverage_zone_rate", "market_total", "market_team_implied", "market_opp_implied", "market_spread", "controlled_environment"]
    for c in ["current", "gamescript"]: x[f"attempts_{c}"] = np.nan
    for c in ["current", "bias", "qb_history", "matchup", "contextual"]: x[f"ypa_{c}"] = np.nan
    scored = []
    for week in sorted(num(x.week).dropna().astype(int).unique()):
        train, test = x[num(x.week) < week], x[num(x.week) == week]
        if len(train) < a.min_train or test.empty: continue
        x.loc[test.index, "attempts_current"] = num(test.pred_attempts)
        x.loc[test.index, "attempts_gamescript"] = fit_residual(train, test, attempt_features)
        x.loc[test.index, "ypa_current"] = num(test.pred_ypa)
        x.loc[test.index, "ypa_bias"] = fit_ypa(train, test, [], bias_only=True)
        x.loc[test.index, "ypa_qb_history"] = fit_ypa(train, test, qb_features)
        x.loc[test.index, "ypa_matchup"] = fit_ypa(train, test, matchup_features)
        x.loc[test.index, "ypa_contextual"] = fit_ypa(train, test, qb_features + matchup_features)
        scored.extend(test.index.tolist())
    x = x.loc[sorted(set(scored))].copy()
    candidates = {
        "current": ("attempts_current", "ypa_current"),
        "attempts_only": ("attempts_gamescript", "ypa_current"),
        "ypa_bias_only": ("attempts_current", "ypa_bias"),
        "ypa_qb_history": ("attempts_current", "ypa_qb_history"),
        "ypa_matchup": ("attempts_current", "ypa_matchup"),
        "ypa_contextual": ("attempts_current", "ypa_contextual"),
        "joint": ("attempts_gamescript", "ypa_contextual"),
    }
    summary, tail, weekly = [], [], []
    for name, (ac, yc) in candidates.items():
        x[f"pass_yards_{name}"] = num(x.mc_proj) * num(x[ac]) / num(x.pred_attempts).replace(0, np.nan) * num(x[yc]) / num(x.pred_ypa).replace(0, np.nan)
        summary.append({"candidate": name, "metric": "pass_yards", **metrics(x.actual_pass_yards_raw, x[f"pass_yards_{name}"])})
        err = num(x[f"pass_yards_{name}"]) - num(x.actual_pass_yards_raw)
        tail.append({"candidate": name, "catastrophic_100plus": int(err.abs().ge(100).sum()), "under_100plus": int(err.le(-100).sum()), "over_100plus": int(err.ge(100).sum()), "catastrophic_mae": float(err.loc[(num(x.pass_yards_current)-num(x.actual_pass_yards_raw)).abs().ge(100)].abs().mean())})
        for week, g in x.groupby("week"):
            m = metrics(g.actual_pass_yards_raw, g[f"pass_yards_{name}"]); cur = metrics(g.actual_pass_yards_raw, g.pass_yards_current)
            weekly.append({"week": int(week), "candidate": name, **m, "mae_improvement_vs_current": cur["mae"] - m["mae"]})
    weekly = pd.DataFrame(weekly); stability = weekly.groupby("candidate").agg(weeks=("week", "count"), weeks_won=("mae_improvement_vs_current", lambda s: int((s>0).sum())), mean_weekly_improvement=("mae_improvement_vs_current", "mean"), worst_week=("mae_improvement_vs_current", "min"), best_week=("mae_improvement_vs_current", "max")).reset_index()
    features = list(dict.fromkeys(attempt_features + qb_features + matchup_features)); coverage = pd.DataFrame({"feature": features, "non_null": [int(x[f].notna().sum()) for f in features], "n": len(x)}); coverage["coverage"] = coverage.non_null / coverage.n
    a.out_dir.mkdir(parents=True, exist_ok=True); x.to_csv(a.out_dir/"qb_joint_attempts_ypa_trace.csv", index=False); pd.DataFrame(summary).to_csv(a.out_dir/"qb_joint_attempts_ypa_summary.csv", index=False); pd.DataFrame(tail).to_csv(a.out_dir/"qb_joint_attempts_ypa_tail.csv", index=False); weekly.to_csv(a.out_dir/"qb_joint_attempts_ypa_weekly.csv", index=False); stability.to_csv(a.out_dir/"qb_joint_attempts_ypa_stability.csv", index=False); coverage.to_csv(a.out_dir/"qb_joint_attempts_ypa_feature_coverage.csv", index=False)
    print("=== JOINT SUMMARY ==="); print(pd.DataFrame(summary).to_string(index=False)); print("\n=== TAIL ==="); print(pd.DataFrame(tail).to_string(index=False)); print("\n=== STABILITY ==="); print(stability.to_string(index=False)); print("\n=== COVERAGE ==="); print(coverage.to_string(index=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
