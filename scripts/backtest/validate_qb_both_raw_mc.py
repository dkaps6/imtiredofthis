#!/usr/bin/env python3
"""Migration 58B: canonical MC validation of Migration 57's BOTH RAW candidate.

Compares the official Migration 53 cap+shrink joint candidate against the
Migration 57 BOTH RAW candidate inside the same canonical simulation engine.
No production defaults are changed.
"""
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
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {path}")
    return pd.read_csv(path)


def opt(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def num(value):
    return pd.to_numeric(value, errors="coerce")


def met(actual, predicted) -> dict:
    z = pd.DataFrame({"actual": num(actual), "pred": num(predicted)}).dropna()
    e = z.pred - z.actual
    return {
        "n": len(z),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()),
        "correlation": float(z.pred.corr(z.actual)) if len(z) > 2 and z.pred.nunique() > 1 else np.nan,
        "catastrophic_100plus": int(e.abs().ge(100).sum()),
        "under_100plus": int(e.le(-100).sum()),
        "over_100plus": int(e.ge(100).sum()),
    }


def wrapper(original, attempt_factors: dict[str, float], ypa_factors: dict[tuple[str, str], float]):
    def apply(metrics: pd.DataFrame) -> pd.DataFrame:
        out = original(metrics)
        team = out.team.astype(str).str.upper().str.strip()
        key = out.player_clean_key.astype(str)
        out["rules_pass_rate"] = (
            num(out.rules_pass_rate) * team.map(attempt_factors).fillna(1.0)
        ).clip(0.25, 0.85)
        factor = pd.Series(
            [ypa_factors.get((t, k), 1.0) for t, k in zip(team, key)],
            index=out.index,
        )
        out["rules_ypa"] = (num(out.rules_ypa) * factor).clip(4.5, 10.5)
        return out

    return apply


def clean_factor(numerator, denominator, *, clip: tuple[float, float] | None = None) -> pd.Series:
    f = num(numerator) / num(denominator).replace(0, np.nan)
    f = f.where(np.isfinite(f) & f.gt(0), 1.0).fillna(1.0)
    return f.clip(*clip) if clip else f


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--prior-season", type=int, required=True)
    p.add_argument("--weeks", default="1-18")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--joint-trace", type=Path, required=True)
    p.add_argument("--raw-trace", type=Path, required=True)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--universe-dir", type=Path, required=True)
    p.add_argument("--injuries", type=Path, required=True)
    p.add_argument("--weather", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    joint = read(a.joint_trace)
    raw = read(a.raw_trace)
    logs = read(a.player_logs)
    team_weekly = read(a.team_weekly)
    schedule = read(a.schedule)
    injuries = opt(a.injuries)
    weather = opt(a.weather)
    original = simulation_rules.apply_rules_to_metrics

    key_cols = ["week", "team", "player_clean_key"]
    common = joint[key_cols].drop_duplicates().merge(
        raw[key_cols].drop_duplicates(), on=key_cols, how="inner", validate="one_to_one"
    )
    if common.empty:
        raise RuntimeError("no common stable-QB rows between joint and BOTH RAW traces")

    candidate_weeks = set(num(common.week).dropna().astype(int))
    traces: list[pd.DataFrame] = []

    for week in [w for w in _parse_weeks(a.weeks) if w in candidate_weeks]:
        keys = common[num(common.week).eq(week)].copy()
        j = joint[num(joint.week).eq(week)].merge(keys, on=key_cols, how="inner")
        r = raw[num(raw.week).eq(week)].merge(keys, on=key_cols, how="inner")
        if j.empty or r.empty:
            continue

        # Official Migration 53 canonical factors retain the validator's existing
        # +/-25% factor guardrail so the reference is identical to prior runs.
        joint_att_factor = clean_factor(j.attempts_gamescript, j.attempts_current, clip=(0.75, 1.25))
        joint_ypa_factor = clean_factor(j.ypa_contextual, j.ypa_current, clip=(0.75, 1.25))
        joint_att = dict(zip(j.team.astype(str).str.upper().str.strip(), joint_att_factor))
        joint_ypa = {
            (str(row.team).upper().strip(), str(row.player_clean_key)): float(factor)
            for (_, row), factor in zip(j.iterrows(), joint_ypa_factor)
        }

        # BOTH RAW already has the Migration 57 outer plausibility bounds applied
        # (18-48 attempts; 4.5-10.5 YPA). Do not reintroduce the removed residual
        # caps/shrinkage through an extra +/-25% ratio clamp here.
        raw_att_factor = clean_factor(r.attempts_raw, r.pred_attempts)
        raw_ypa_factor = clean_factor(r.ypa_raw, r.pred_ypa)
        raw_att = dict(zip(r.team.astype(str).str.upper().str.strip(), raw_att_factor))
        raw_ypa = {
            (str(row.team).upper().strip(), str(row.player_clean_key)): float(factor)
            for (_, row), factor in zip(r.iterrows(), raw_ypa_factor)
        }

        universe = read(a.universe_dir / f"{a.season}_week_{week:02d}.csv")
        bundle = build_historical_context_bundle(
            player_logs=logs,
            team_weekly=team_weekly,
            pregame_universe=universe,
            schedule=schedule,
            season=a.season,
            week=week,
            prior_season=a.prior_season,
            injuries=_exact_week(injuries, a.season, week),
            weather=_exact_week(weather, a.season, week),
        )
        actual = build_actual_rows(logs, a.season, week)

        modes = {
            "current": original,
            "joint_cap_shrink": wrapper(original, joint_att, joint_ypa),
            "both_raw": wrapper(original, raw_att, raw_ypa),
        }
        for mode, fn in modes.items():
            if mode == "current":
                mc = build_mc_predictions(bundle, iterations=a.iterations, seed=53 + week)
            else:
                with patch.object(simulation_rules, "apply_rules_to_metrics", side_effect=fn):
                    mc = build_mc_predictions(bundle, iterations=a.iterations, seed=53 + week)
            z = mc.merge(actual, on=["team", "player_clean_key", "market"], how="inner")
            z["candidate"] = mode
            z["week"] = week
            traces.append(z)
        print(f"[m58] W{week:02d} stable_qbs={len(keys)}")

    if not traces:
        raise RuntimeError("no canonical MC rows")
    t = pd.concat(traces, ignore_index=True)
    stable = t[t.market.eq("pass_yards")].merge(common, on=key_cols, how="inner")

    rows = []
    for mode, g in stable.groupby("candidate"):
        rows.append({
            "season": a.season,
            "candidate": mode,
            "slice": "stable_qb",
            "market": "pass_yards",
            **met(g.actual, g.mc_proj),
        })
    for (mode, market), g in t.groupby(["candidate", "market"]):
        rows.append({
            "season": a.season,
            "candidate": mode,
            "slice": "all_available",
            "market": market,
            **met(g.actual, g.mc_proj),
        })

    a.out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out_dir / "qb_both_raw_mc_trace.csv", index=False)
    stable.to_csv(a.out_dir / "qb_both_raw_mc_stable.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(a.out_dir / "qb_both_raw_mc_summary.csv", index=False)
    print("=== MIGRATION 58 CANONICAL MC ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
