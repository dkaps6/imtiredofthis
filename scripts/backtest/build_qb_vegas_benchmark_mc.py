#!/usr/bin/env python3
"""Migration 60A: build paired canonical QB projections for market benchmarking.

Measurement only. Produces independent football projections first, before any
sportsbook line is loaded. The output is then safe to join to historical prop
lines in Migration 60B without allowing the market line to influence the model.

Scored modes:
- current
- joint_cap_shrink
- attempts_raw_only
- ypa_raw_only (diagnostic reference)
- both_raw
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_actual_rows, build_mc_predictions
from scripts.backtest.diagnose_qb_raw_tail_attribution_mc import (
    KEYS,
    clean_factor,
    met,
    num,
    opt,
    read,
    wrapper,
)
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling import simulation_rules


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

    joint, raw = read(a.joint_trace), read(a.raw_trace)
    logs, tw, sched = read(a.player_logs), read(a.team_weekly), read(a.schedule)
    inj, weather = opt(a.injuries), opt(a.weather)
    original = simulation_rules.apply_rules_to_metrics

    common = joint[KEYS].drop_duplicates().merge(
        raw[KEYS].drop_duplicates(), on=KEYS, how="inner", validate="one_to_one"
    )
    if common.empty:
        raise RuntimeError("no common stable-QB rows")

    candidate_weeks = set(num(common.week).dropna().astype(int))
    traces: list[pd.DataFrame] = []

    for week in [w for w in _parse_weeks(a.weeks) if w in candidate_weeks]:
        keys = common[num(common.week).eq(week)].copy()
        j = joint[num(joint.week).eq(week)].merge(keys, on=KEYS, how="inner")
        r = raw[num(raw.week).eq(week)].merge(keys, on=KEYS, how="inner")
        if j.empty or r.empty:
            continue

        jat = dict(
            zip(
                j.team.astype(str).str.upper().str.strip(),
                clean_factor(j.attempts_gamescript, j.attempts_current, (.75, 1.25)),
            )
        )
        jyp = {
            (str(row.team).upper().strip(), str(row.player_clean_key)): float(f)
            for (_, row), f in zip(
                j.iterrows(), clean_factor(j.ypa_contextual, j.ypa_current, (.75, 1.25))
            )
        }
        rat = dict(
            zip(
                r.team.astype(str).str.upper().str.strip(),
                clean_factor(r.attempts_raw, r.pred_attempts),
            )
        )
        ryp = {
            (str(row.team).upper().strip(), str(row.player_clean_key)): float(f)
            for (_, row), f in zip(r.iterrows(), clean_factor(r.ypa_raw, r.pred_ypa))
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
            injuries=_exact_week(inj, a.season, week),
            weather=_exact_week(weather, a.season, week),
        )
        actual = build_actual_rows(logs, a.season, week)

        modes = {
            "current": None,
            "joint_cap_shrink": wrapper(original, jat, jyp),
            "attempts_raw_only": wrapper(original, rat, jyp),
            "ypa_raw_only": wrapper(original, jat, ryp),
            "both_raw": wrapper(original, rat, ryp),
        }
        for mode, fn in modes.items():
            if fn is None:
                mc = build_mc_predictions(bundle, iterations=a.iterations, seed=53 + week)
            else:
                with patch.object(simulation_rules, "apply_rules_to_metrics", side_effect=fn):
                    mc = build_mc_predictions(bundle, iterations=a.iterations, seed=53 + week)
            z = mc.loc[mc.market.eq("pass_yards")].merge(
                actual.loc[actual.market.eq("pass_yards")],
                on=["team", "player_clean_key", "market"],
                how="inner",
            )
            z = z.merge(keys, on=KEYS, how="inner")
            z["candidate"] = mode
            z["week"] = week
            traces.append(z)
        print(f"[m60 projection] {a.season} W{week:02d} stable_qbs={len(keys)}")

    if not traces:
        raise RuntimeError("no Migration 60 canonical QB rows")

    long = pd.concat(traces, ignore_index=True)
    wide = long.pivot_table(
        index=KEYS,
        columns="candidate",
        values=["actual", "mc_proj"],
        aggfunc="first",
    )
    wide.columns = [f"{x}_{y}" for x, y in wide.columns]
    wide = wide.reset_index()
    actual_cols = [c for c in wide if c.startswith("actual_")]
    if not actual_cols:
        raise RuntimeError("Migration 60 projection trace has no actual passing yards")
    wide["actual"] = num(wide[actual_cols[0]])

    sk = sched.copy()
    sk.columns = [str(c).strip().lower() for c in sk.columns]
    sk = sk.loc[
        num(sk.get("season", a.season)).eq(a.season)
        & num(sk.week).isin(candidate_weeks)
    ].copy()
    keep = [c for c in ["week", "team", "opponent", "home_away", "game_id"] if c in sk]
    sk = sk[keep].drop_duplicates(["week", "team"])
    wide = wide.merge(sk, on=["week", "team"], how="left", validate="many_to_one")
    wide["season"] = a.season

    modes = ["current", "joint_cap_shrink", "attempts_raw_only", "ypa_raw_only", "both_raw"]
    summary = []
    for mode in modes:
        col = f"mc_proj_{mode}"
        if col not in wide:
            continue
        summary.append({"season": a.season, "candidate": mode, **met(wide.actual, wide[col])})
    summary = pd.DataFrame(summary)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(a.out_dir / f"m60_qb_projection_long_{a.season}.csv", index=False)
    wide.to_csv(a.out_dir / f"m60_qb_projection_wide_{a.season}.csv", index=False)
    summary.to_csv(a.out_dir / f"m60_qb_football_metrics_{a.season}.csv", index=False)
    print("=== M60 INDEPENDENT FOOTBALL PROJECTIONS ===")
    print(summary.to_string(index=False))
    print(f"[m60 projection] unique_game_ids={wide.game_id.nunique(dropna=True) if 'game_id' in wide else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
