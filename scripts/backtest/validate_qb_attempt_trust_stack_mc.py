#!/usr/bin/env python3
"""Migration 61B: canonical MC validation of selective-trust QB attempt models.

All M61 candidates use Migration 53's capped/shrunk contextual YPA so this
migration isolates the pass-attempt mechanism. Vegas player-prop lines are not
loaded here and cannot influence the football projection.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_actual_rows, build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.validate_qb_both_raw_mc import clean_factor, met, num, opt, read, wrapper
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling import simulation_rules

KEYS = ["week", "team", "player_clean_key"]
CANDIDATES = [
    "current",
    "joint_cap_shrink",
    "attempts_raw_only",
    "stack_ridge",
    "stack_gbr",
    "stack_consensus",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--prior-season", type=int, required=True)
    p.add_argument("--weeks", default="1-18")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--joint-trace", type=Path, required=True)
    p.add_argument("--raw-trace", type=Path, required=True)
    p.add_argument("--trust-trace", type=Path, required=True)
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
    trust = read(a.trust_trace)
    logs = read(a.player_logs)
    tw = read(a.team_weekly)
    sched = read(a.schedule)
    injuries = opt(a.injuries)
    weather = opt(a.weather)
    original = simulation_rules.apply_rules_to_metrics

    common = joint[KEYS].drop_duplicates()
    common = common.merge(raw[KEYS].drop_duplicates(), on=KEYS, how="inner", validate="one_to_one")
    common = common.merge(trust[KEYS].drop_duplicates(), on=KEYS, how="inner", validate="one_to_one")
    if common.empty:
        raise RuntimeError("no common stable-QB rows for M61")

    candidate_weeks = set(num(common.week).dropna().astype(int))
    traces: list[pd.DataFrame] = []

    for week in [w for w in _parse_weeks(a.weeks) if w in candidate_weeks]:
        keys = common[num(common.week).eq(week)].copy()
        j = joint[num(joint.week).eq(week)].merge(keys, on=KEYS, how="inner")
        r = raw[num(raw.week).eq(week)].merge(keys, on=KEYS, how="inner")
        q = trust[num(trust.week).eq(week)].merge(keys, on=KEYS, how="inner")
        if j.empty or r.empty or q.empty:
            continue

        joint_att = dict(zip(
            j.team.astype(str).str.upper().str.strip(),
            clean_factor(j.attempts_gamescript, j.attempts_current, clip=(0.75, 1.25)),
        ))
        joint_ypa = {
            (str(row.team).upper().strip(), str(row.player_clean_key)): float(f)
            for (_, row), f in zip(
                j.iterrows(), clean_factor(j.ypa_contextual, j.ypa_current, clip=(0.75, 1.25))
            )
        }
        raw_att = dict(zip(
            r.team.astype(str).str.upper().str.strip(),
            clean_factor(r.attempts_raw, r.pred_attempts),
        ))

        attempt_maps = {
            "stack_ridge": dict(zip(
                q.team.astype(str).str.upper().str.strip(),
                clean_factor(q.attempts_stack_ridge, q.pred_attempts),
            )),
            "stack_gbr": dict(zip(
                q.team.astype(str).str.upper().str.strip(),
                clean_factor(q.attempts_stack_gbr, q.pred_attempts),
            )),
            "stack_consensus": dict(zip(
                q.team.astype(str).str.upper().str.strip(),
                clean_factor(q.attempts_stack_consensus, q.pred_attempts),
            )),
        }

        universe = read(a.universe_dir / f"{a.season}_week_{week:02d}.csv")
        bundle = build_historical_context_bundle(
            player_logs=logs,
            team_weekly=tw,
            pregame_universe=universe,
            schedule=sched,
            season=a.season,
            week=week,
            prior_season=a.prior_season,
            injuries=_exact_week(injuries, a.season, week),
            weather=_exact_week(weather, a.season, week),
        )
        actual = build_actual_rows(logs, a.season, week)

        modes = {
            "current": None,
            "joint_cap_shrink": wrapper(original, joint_att, joint_ypa),
            "attempts_raw_only": wrapper(original, raw_att, joint_ypa),
            "stack_ridge": wrapper(original, attempt_maps["stack_ridge"], joint_ypa),
            "stack_gbr": wrapper(original, attempt_maps["stack_gbr"], joint_ypa),
            "stack_consensus": wrapper(original, attempt_maps["stack_consensus"], joint_ypa),
        }
        for mode, fn in modes.items():
            if fn is None:
                mc = build_mc_predictions(bundle, iterations=a.iterations, seed=53 + week)
            else:
                with patch.object(simulation_rules, "apply_rules_to_metrics", side_effect=fn):
                    mc = build_mc_predictions(bundle, iterations=a.iterations, seed=53 + week)
            z = mc.merge(actual, on=["team", "player_clean_key", "market"], how="inner")
            z["candidate"] = mode
            z["week"] = week
            z["season"] = a.season
            traces.append(z)
        print(f"[m61 mc] {a.season} W{week:02d} stable_qbs={len(keys)}")

    if not traces:
        raise RuntimeError("no canonical M61 MC rows")
    t = pd.concat(traces, ignore_index=True)
    stable = t[t.market.eq("pass_yards")].merge(common, on=KEYS, how="inner")

    rows = []
    for mode, g in stable.groupby("candidate"):
        rows.append({"season": a.season, "candidate": mode, "slice": "stable_qb", "market": "pass_yards", **met(g.actual, g.mc_proj)})
    for (mode, market), g in t.groupby(["candidate", "market"]):
        rows.append({"season": a.season, "candidate": mode, "slice": "all_available", "market": market, **met(g.actual, g.mc_proj)})
    summary = pd.DataFrame(rows)

    # Market-benchmark-ready wide file. Sportsbook lines are joined only later.
    pw = stable.pivot_table(
        index=KEYS + ["season"], columns="candidate", values=["actual", "mc_proj"], aggfunc="first"
    )
    pw.columns = [f"{a}_{b}" for a, b in pw.columns]
    pw = pw.reset_index()
    actual_cols = [c for c in pw.columns if c.startswith("actual_")]
    pw["actual"] = num(pw[actual_cols[0]])
    sk = sched.copy(); sk.columns = [str(c).strip().lower() for c in sk.columns]
    sk = sk[num(sk.get("season", a.season)).eq(a.season)].copy()
    keep = [c for c in ["week", "team", "game_id"] if c in sk.columns]
    sk = sk[keep].drop_duplicates(["week", "team"])
    pw = pw.merge(sk, on=["week", "team"], how="left", validate="many_to_one")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out_dir / "qb_attempt_trust_stack_mc_trace.csv", index=False)
    stable.to_csv(a.out_dir / "qb_attempt_trust_stack_mc_stable.csv", index=False)
    summary.to_csv(a.out_dir / "qb_attempt_trust_stack_mc_summary.csv", index=False)
    pw.to_csv(a.out_dir / f"m61_qb_projection_wide_{a.season}.csv", index=False)
    print("=== MIGRATION 61 CANONICAL MC ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
