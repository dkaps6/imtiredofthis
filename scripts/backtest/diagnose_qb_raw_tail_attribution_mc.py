#!/usr/bin/env python3
"""Migration 59: canonical attribution of BOTH RAW QB tail failures.

Diagnostic only. Replays the Migration 58 canonical comparison with two crossed
controls so new 100+ yard overprojections can be attributed to raw attempts,
raw YPA, or the interaction/compounding of both. No production defaults or
candidate coefficients are changed.
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

KEYS = ["week", "team", "player_clean_key"]


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {path}")
    return pd.read_csv(path)


def opt(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def num(v):
    return pd.to_numeric(v, errors="coerce")


def clean_factor(numerator, denominator, clip=None):
    f = num(numerator) / num(denominator).replace(0, np.nan)
    f = f.where(np.isfinite(f) & f.gt(0), 1.0).fillna(1.0)
    return f.clip(*clip) if clip else f


def wrapper(original, attempt_factors, ypa_factors):
    def apply(metrics: pd.DataFrame) -> pd.DataFrame:
        out = original(metrics)
        team = out.team.astype(str).str.upper().str.strip()
        key = out.player_clean_key.astype(str)
        out["rules_pass_rate"] = (num(out.rules_pass_rate) * team.map(attempt_factors).fillna(1.0)).clip(.25, .85)
        yf = pd.Series([ypa_factors.get((t, k), 1.0) for t, k in zip(team, key)], index=out.index)
        out["rules_ypa"] = (num(out.rules_ypa) * yf).clip(4.5, 10.5)
        return out
    return apply


def tail_state(error):
    e = num(error)
    return np.select([e.le(-100), e.ge(100)], ["under100", "over100"], default="normal")


def met(actual, pred):
    z = pd.DataFrame({"a": num(actual), "p": num(pred)}).dropna(); e = z.p - z.a
    return {
        "n": len(z), "mae": float(e.abs().mean()), "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()), "correlation": float(z.p.corr(z.a)) if len(z) > 2 else np.nan,
        "catastrophic_100plus": int(e.abs().ge(100).sum()),
        "under_100plus": int(e.le(-100).sum()), "over_100plus": int(e.ge(100).sum()),
    }


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

    common = joint[KEYS].drop_duplicates().merge(raw[KEYS].drop_duplicates(), on=KEYS, validate="one_to_one")
    if common.empty:
        raise RuntimeError("no common stable-QB rows")

    # Persist component metadata for interpretation after the canonical MC run.
    jc = [c for c in ["attempts_current", "attempts_gamescript", "ypa_current", "ypa_contextual"] if c in joint]
    rc = [c for c in ["pred_attempts", "pred_ypa", "attempts_raw", "ypa_raw", "att_raw_delta", "ypa_raw_delta", "qb_recent_ypa", "actual_pass_att", "actual_ypa", "actual_pass_yards_raw"] if c in raw]
    meta = common.merge(joint[KEYS + jc], on=KEYS, how="left").merge(raw[KEYS + rc], on=KEYS, how="left", suffixes=("_joint", "_raw"))

    traces = []
    for week in [w for w in _parse_weeks(a.weeks) if w in set(num(common.week).dropna().astype(int))]:
        keys = common[num(common.week).eq(week)]
        j = joint[num(joint.week).eq(week)].merge(keys, on=KEYS, how="inner")
        r = raw[num(raw.week).eq(week)].merge(keys, on=KEYS, how="inner")
        if j.empty or r.empty:
            continue

        jat = dict(zip(j.team.astype(str).str.upper().str.strip(), clean_factor(j.attempts_gamescript, j.attempts_current, (.75, 1.25))))
        jyp = {(str(row.team).upper().strip(), str(row.player_clean_key)): float(f) for (_, row), f in zip(j.iterrows(), clean_factor(j.ypa_contextual, j.ypa_current, (.75, 1.25)))}
        rat = dict(zip(r.team.astype(str).str.upper().str.strip(), clean_factor(r.attempts_raw, r.pred_attempts)))
        ryp = {(str(row.team).upper().strip(), str(row.player_clean_key)): float(f) for (_, row), f in zip(r.iterrows(), clean_factor(r.ypa_raw, r.pred_ypa))}

        universe = read(a.universe_dir / f"{a.season}_week_{week:02d}.csv")
        bundle = build_historical_context_bundle(
            player_logs=logs, team_weekly=tw, pregame_universe=universe, schedule=sched,
            season=a.season, week=week, prior_season=a.prior_season,
            injuries=_exact_week(inj, a.season, week), weather=_exact_week(weather, a.season, week),
        )
        actual = build_actual_rows(logs, a.season, week)
        modes = {
            "joint_cap_shrink": wrapper(original, jat, jyp),
            "attempts_raw_only": wrapper(original, rat, jyp),
            "ypa_raw_only": wrapper(original, jat, ryp),
            "both_raw": wrapper(original, rat, ryp),
        }
        for mode, fn in modes.items():
            with patch.object(simulation_rules, "apply_rules_to_metrics", side_effect=fn):
                mc = build_mc_predictions(bundle, iterations=a.iterations, seed=53 + week)
            z = mc.merge(actual, on=["team", "player_clean_key", "market"], how="inner")
            z["candidate"], z["week"] = mode, week
            traces.append(z)
        print(f"[m59] W{week:02d} stable_qbs={len(keys)}")

    if not traces:
        raise RuntimeError("no M59 canonical rows")
    t = pd.concat(traces, ignore_index=True)
    stable_long = t[t.market.eq("pass_yards")].merge(common, on=KEYS, how="inner")

    # Wide, paired per-game canonical projections.
    wide = stable_long.pivot_table(index=KEYS, columns="candidate", values=["actual", "mc_proj"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index().merge(meta, on=KEYS, how="left")
    if "actual_joint_cap_shrink" in wide:
        wide["actual"] = num(wide["actual_joint_cap_shrink"])
    else:
        actual_cols = [c for c in wide if c.startswith("actual_")]
        wide["actual"] = num(wide[actual_cols[0]])

    modes = ["joint_cap_shrink", "attempts_raw_only", "ypa_raw_only", "both_raw"]
    for m in modes:
        wide[f"error_{m}"] = num(wide[f"mc_proj_{m}"]) - num(wide.actual)
        wide[f"tail_{m}"] = tail_state(wide[f"error_{m}"])
    wide["delta_attempts_mode"] = num(wide.mc_proj_attempts_raw_only) - num(wide.mc_proj_joint_cap_shrink)
    wide["delta_ypa_mode"] = num(wide.mc_proj_ypa_raw_only) - num(wide.mc_proj_joint_cap_shrink)
    wide["delta_both_mode"] = num(wide.mc_proj_both_raw) - num(wide.mc_proj_joint_cap_shrink)
    wide["interaction_delta"] = wide.delta_both_mode - wide.delta_attempts_mode - wide.delta_ypa_mode

    wide["new_over"] = wide.tail_joint_cap_shrink.ne("over100") & wide.tail_both_raw.eq("over100")
    wide["new_under"] = wide.tail_joint_cap_shrink.ne("under100") & wide.tail_both_raw.eq("under100")
    wide["rescued_under"] = wide.tail_joint_cap_shrink.eq("under100") & wide.tail_both_raw.ne("under100")
    wide["rescued_over"] = wide.tail_joint_cap_shrink.eq("over100") & wide.tail_both_raw.ne("over100")

    def mechanism(row):
        if not row.new_over:
            return "not_new_over"
        a_over = row.tail_attempts_raw_only == "over100"
        y_over = row.tail_ypa_raw_only == "over100"
        if a_over and not y_over: return "attempts_alone_crosses_100"
        if y_over and not a_over: return "ypa_alone_crosses_100"
        if a_over and y_over: return "both_individually_cross_100"
        return "compounding_only_crosses_100"
    wide["new_over_mechanism"] = wide.apply(mechanism, axis=1)

    def dominant(row):
        vals = {"attempts": abs(row.delta_attempts_mode), "ypa": abs(row.delta_ypa_mode), "interaction": abs(row.interaction_delta)}
        return max(vals, key=vals.get)
    wide["dominant_projection_change"] = wide.apply(dominant, axis=1)

    ar = num(wide.get("att_raw_delta", np.nan)); yr = num(wide.get("ypa_raw_delta", np.nan))
    wide["raw_delta_direction"] = np.select(
        [(ar.gt(0) & yr.gt(0)), (ar.lt(0) & yr.lt(0)), (ar.gt(0) & yr.le(0)), (ar.le(0) & yr.gt(0))],
        ["both_up", "both_down", "attempts_up_ypa_not", "ypa_up_attempts_not"], default="other")

    metrics = []
    for m in modes:
        metrics.append({"season": a.season, "candidate": m, **met(wide.actual, wide[f"mc_proj_{m}"])})
    metrics = pd.DataFrame(metrics)

    transitions = pd.crosstab(wide.tail_joint_cap_shrink, wide.tail_both_raw, margins=True).reset_index()
    mech = wide[wide.new_over].groupby("new_over_mechanism", dropna=False).agg(
        n=("new_over", "size"), mean_actual=("actual", "mean"),
        mean_joint_proj=("mc_proj_joint_cap_shrink", "mean"), mean_both_raw_proj=("mc_proj_both_raw", "mean"),
        mean_attempt_delta=("delta_attempts_mode", "mean"), mean_ypa_delta=("delta_ypa_mode", "mean"),
        mean_interaction=("interaction_delta", "mean"),
    ).reset_index()
    dom = wide[wide.new_over].groupby("dominant_projection_change").size().rename("n").reset_index()

    direction_rows = []
    for label, g in wide.groupby("raw_delta_direction"):
        direction_rows.append({
            "season": a.season, "raw_delta_direction": label, "n": len(g),
            "joint_mae": float(num(g.error_joint_cap_shrink).abs().mean()),
            "both_raw_mae": float(num(g.error_both_raw).abs().mean()),
            "joint_over100_rate": float(g.tail_joint_cap_shrink.eq("over100").mean()),
            "both_raw_over100_rate": float(g.tail_both_raw.eq("over100").mean()),
            "new_over_rate": float(g.new_over.mean()), "rescued_under_rate": float(g.rescued_under.mean()),
        })
    directions = pd.DataFrame(direction_rows)

    # Precommitted interpretation: identify a single dominant tail mechanism only
    # if it explains >50% of new overprojections; otherwise call the result mixed.
    n_new = int(wide.new_over.sum())
    if n_new:
        counts = wide.loc[wide.new_over, "new_over_mechanism"].value_counts()
        top = str(counts.index[0]); share = float(counts.iloc[0] / n_new)
        if share > .50:
            interpretation = top
        else:
            interpretation = "mixed_no_single_mechanism_over_50pct"
    else:
        top, share, interpretation = "none", 0.0, "no_new_overprojections"
    decision = pd.DataFrame([{
        "season": a.season, "new_over_n": n_new, "rescued_under_n": int(wide.rescued_under.sum()),
        "new_under_n": int(wide.new_under.sum()), "rescued_over_n": int(wide.rescued_over.sum()),
        "top_new_over_mechanism": top, "top_mechanism_share": share,
        "precommitted_interpretation": interpretation,
    }])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out_dir / f"m59_mc_trace_{a.season}.csv", index=False)
    wide.to_csv(a.out_dir / f"m59_game_attribution_{a.season}.csv", index=False)
    metrics.to_csv(a.out_dir / f"m59_metrics_{a.season}.csv", index=False)
    transitions.to_csv(a.out_dir / f"m59_tail_transitions_{a.season}.csv", index=False)
    mech.to_csv(a.out_dir / f"m59_new_over_mechanisms_{a.season}.csv", index=False)
    dom.to_csv(a.out_dir / f"m59_new_over_dominant_component_{a.season}.csv", index=False)
    directions.to_csv(a.out_dir / f"m59_raw_delta_directions_{a.season}.csv", index=False)
    decision.to_csv(a.out_dir / f"m59_interpretation_{a.season}.csv", index=False)

    print("=== M59 METRICS ==="); print(metrics.to_string(index=False))
    print("\n=== M59 TAIL TRANSITIONS ==="); print(transitions.to_string(index=False))
    print("\n=== M59 NEW OVER MECHANISMS ==="); print(mech.to_string(index=False))
    print("\n=== M59 RAW DELTA DIRECTIONS ==="); print(directions.to_string(index=False))
    print("\n=== M59 INTERPRETATION ==="); print(decision.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
