#!/usr/bin/env python3
"""WR-ND2: exact post-M38 catch-rate vs yards-per-reception decomposition."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import evaluate_wr_nd1_post_m38_decomposition as nd1
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks

COMPONENTS = (
    "TEAM_TARGET_VOLUME",
    "WR_TARGET_MASS",
    "WITHIN_WR_ALLOCATION",
    "CATCH_RATE",
    "YARDS_PER_RECEPTION",
)
PARENT_BASE_MAE = 22.022544815892125
_ORIGINAL_ACTUAL_WEEK = nd1.actual_week
_ANOMALIES: list[pd.DataFrame] = []


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing required input: {path}")
    return pd.read_csv(path)


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def filtered_actual_week(logs: pd.DataFrame, season: int, week: int):
    x = logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    s = pd.to_numeric(x.get("season"), errors="coerce")
    w = pd.to_numeric(x.get("week"), errors="coerce")
    targets = pd.to_numeric(x.get("targets"), errors="coerce").fillna(0.0)
    yards = pd.to_numeric(x.get("rec_yards"), errors="coerce").fillna(0.0)
    mask = s.eq(int(season)) & w.eq(int(week)) & targets.le(0) & yards.abs().gt(1e-9)
    if mask.any():
        bad = x.loc[mask].copy()
        bad["wr_nd2_exclusion_reason"] = "NONZERO_REC_YARDS_WITH_ZERO_RECORDED_TARGETS"
        _ANOMALIES.append(bad)
        filtered = logs.loc[~mask.to_numpy()].copy()
    else:
        filtered = logs
    return _ORIGINAL_ACTUAL_WEEK(filtered, season, week)


def pred(frame: pd.DataFrame, corrected: frozenset[str]) -> pd.Series:
    team = frame.actual_team_targets if "TEAM_TARGET_VOLUME" in corrected else frame.pred_team_targets
    mass = frame.actual_wr_mass if "WR_TARGET_MASS" in corrected else frame.pred_wr_mass
    within = frame.actual_within_wr if "WITHIN_WR_ALLOCATION" in corrected else frame.pred_within_wr
    catch = frame.actual_catch_rate if "CATCH_RATE" in corrected else frame.pred_catch_rate
    ypr = frame.actual_ypr if "YARDS_PER_RECEPTION" in corrected else frame.pred_ypr
    return team * mass * within * catch * ypr


def score(actual: pd.Series, prediction: pd.Series) -> dict:
    z = pd.DataFrame({"a": pd.to_numeric(actual, errors="coerce"), "p": pd.to_numeric(prediction, errors="coerce")}).dropna()
    e = z.p - z.a
    corr = float(z.p.corr(z.a)) if len(z) > 1 and z.p.nunique() > 1 and z.a.nunique() > 1 else np.nan
    return {
        "n": int(len(z)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()),
        "correlation": corr,
    }


def shapley(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n = len(COMPONENTS)
    maes: dict[frozenset[str], float] = {}
    for mask in range(1 << n):
        subset = frozenset(COMPONENTS[i] for i in range(n) if mask & (1 << i))
        maes[subset] = score(frame.actual_rec_yards, pred(frame, subset))["mae"]
    rows = []
    for comp in COMPONENTS:
        phi = 0.0
        for subset, base in maes.items():
            if comp in subset:
                continue
            k = len(subset)
            weight = math.factorial(k) * math.factorial(n-k-1) / math.factorial(n)
            phi += weight * (base - maes[frozenset(set(subset) | {comp})])
        rows.append({"component": comp, "shapley_mae_recovery": float(phi)})
    base = float(maes[frozenset()])
    full = float(maes[frozenset(COMPONENTS)])
    recovery = base - full
    phi_sum = float(sum(r["shapley_mae_recovery"] for r in rows))
    if abs(phi_sum - recovery) > 1e-8:
        raise RuntimeError(f"WR-ND2 Shapley identity failed: {phi_sum} vs {recovery}")
    max_full = float((pred(frame, frozenset(COMPONENTS)) - frame.actual_rec_yards).abs().max())
    if max_full > 1e-8:
        raise RuntimeError(f"WR-ND2 full-truth identity failed: {max_full}")
    meta = {"n": int(len(frame)), "base_mae": base, "full_mae": full, "total_mae_recovery": recovery, "shapley_sum": phi_sum, "max_full_identity_abs_diff": max_full}
    for r in rows:
        r.update(meta)
    return pd.DataFrame(rows), meta


def efficiency_gate(overall: pd.DataFrame, slices: pd.DataFrame) -> tuple[str, dict]:
    eff = overall.loc[overall.component.isin(["CATCH_RATE", "YARDS_PER_RECEPTION"])].copy()
    eff = eff.sort_values("shapley_mae_recovery", ascending=False)
    top = str(eff.iloc[0].component)
    positive = float(eff.shapley_mae_recovery.clip(lower=0).sum())
    top_val = float(eff.iloc[0].shapley_mae_recovery)
    share = top_val / positive if positive > 0 and top_val > 0 else 0.0
    role_wins = 0
    role_tops = {}
    for role in ("WR1", "WR2", "WR3"):
        q = slices.loc[(slices.slice_type.eq("wr_role")) & (slices.slice_value.eq(role)) & slices.component.isin(["CATCH_RATE", "YARDS_PER_RECEPTION"])].sort_values("shapley_mae_recovery", ascending=False)
        role_top = str(q.iloc[0].component) if not q.empty else ""
        role_tops[role] = role_top
        role_wins += int(role_top == top)
    passed = top_val > 0 and share >= .60 and role_wins >= 2
    if passed and top == "YARDS_PER_RECEPTION":
        disposition = "YARDS_PER_RECEPTION_DOMINANT"
    elif passed and top == "CATCH_RATE":
        disposition = "CATCH_RATE_DOMINANT"
    else:
        disposition = "MIXED_WR_EFFICIENCY_MECHANICS"
    return disposition, {
        "overall_top_efficiency_component": top,
        "overall_top_efficiency_shapley": top_val,
        "overall_positive_efficiency_share": share,
        "wr1_wr2_wr3_same_top_count": role_wins,
        "role_top_efficiency_components": role_tops,
        "dominance_gate_passed": bool(passed),
    }


def descriptives(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("ALL_WR", "ALL_WR", frame)] + [("wr_role", r, frame.loc[frame.wr_role.eq(r)]) for r in ("WR1","WR2","WR3","WR4_PLUS")]
    for st, sv, g in groups:
        if g.empty:
            continue
        for metric, pcol, acol in (("catch_rate","pred_catch_rate","actual_catch_rate"),("yards_per_reception","pred_ypr","actual_ypr"),("yards_per_target","pred_ypt","actual_ypt")):
            p = pd.to_numeric(g[pcol], errors="coerce")
            a = pd.to_numeric(g[acol], errors="coerce")
            rows.append({"slice_type": st, "slice_value": sv, "metric": metric, "n": int(len(g)), "pred_mean": float(p.mean()), "actual_mean": float(a.mean()), "mean_error": float((p-a).mean()), "mae": float((p-a).abs().mean()), "correlation": float(p.corr(a)) if p.nunique()>1 and a.nunique()>1 else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--prior-season", type=int, default=2024)
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--player-logs", type=Path, default=Path("data/backtests/player_game_logs_history.csv"))
    ap.add_argument("--team-weekly", type=Path, default=Path("data/backtests/team_weekly_history.csv"))
    ap.add_argument("--schedule", type=Path, default=Path("data/backtests/schedule_history.csv"))
    ap.add_argument("--universe-dir", type=Path, default=Path("data/backtests/pregame_universe"))
    ap.add_argument("--injuries", type=Path, default=Path("data/backtests/injuries_history.csv"))
    ap.add_argument("--weather", type=Path, default=Path("data/backtests/weather_history.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/wr_nd2_ypt_mechanics"))
    args = ap.parse_args()

    logs = read_required(args.player_logs)
    team = read_required(args.team_weekly)
    sched = read_required(args.schedule)
    inj = read_optional(args.injuries)
    weather = read_optional(args.weather)
    weeks = _parse_weeks(args.weeks)
    nd1.actual_week = filtered_actual_week

    frames = []
    for week in weeks:
        universe = read_required(args.universe_dir / f"{args.season}_week_{week:02d}.csv")
        bundle = build_historical_context_bundle(player_logs=logs, team_weekly=team, pregame_universe=universe, schedule=sched, season=args.season, week=week, prior_season=args.prior_season, injuries=_exact_week(inj,args.season,week), weather=_exact_week(weather,args.season,week))
        frames.append(nd1.build_week_factors(bundle, logs, args.season, week, args.iterations))
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty or frame.duplicated(["season","week","team","player_clean_key"]).any():
        raise RuntimeError("WR-ND2 invalid evaluation frame")
    if (pd.to_numeric(frame.pred_catch_rate, errors="coerce") <= 0).any():
        raise RuntimeError("WR-ND2 non-positive base catch rate")

    frame["pred_ypr"] = frame.pred_ypt / frame.pred_catch_rate
    frame["actual_ypr"] = np.where(frame.actual_receptions > 0, frame.actual_rec_yards / frame.actual_receptions, frame.pred_ypr)
    frame["actual_ypt"] = np.where(frame.actual_targets > 0, frame.actual_rec_yards / frame.actual_targets, frame.pred_ypt)
    ypt_rebuild = frame.pred_catch_rate * frame.pred_ypr
    max_ypt_drift = float((ypt_rebuild - frame.pred_ypt).abs().max())
    if max_ypt_drift > 1e-10:
        raise RuntimeError(f"WR-ND2 base YPT factorization drift: {max_ypt_drift}")

    overall, identity = shapley(frame)
    if abs(identity["base_mae"] - PARENT_BASE_MAE) > 1e-9:
        raise RuntimeError(f"WR-ND2 parent base MAE drift: {identity['base_mae']} vs {PARENT_BASE_MAE}")

    sliced = []
    for st, sv, sf in nd1.slice_specs(frame):
        if sf.empty:
            continue
        q, _ = shapley(sf)
        q.insert(0, "slice_value", str(sv)); q.insert(0, "slice_type", str(st))
        sliced.append(q)
    slices = pd.concat(sliced, ignore_index=True)
    desc = descriptives(frame)
    disposition, gate = efficiency_gate(overall, slices)

    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    anomaly = pd.concat(_ANOMALIES, ignore_index=True) if _ANOMALIES else pd.DataFrame(columns=["season","week","team","player","position","targets","rec_yards","wr_nd2_exclusion_reason"])
    frame.to_csv(out/"wr_nd2_player_factors.csv", index=False)
    overall.to_csv(out/"wr_nd2_shapley_summary.csv", index=False)
    slices.to_csv(out/"wr_nd2_slice_shapley.csv", index=False)
    desc.to_csv(out/"wr_nd2_efficiency_descriptives.csv", index=False)
    anomaly.to_csv(out/"wr_nd2_factorization_anomalies.csv", index=False)
    summary = {
        "migration":"WR-ND2", "season":int(args.season), "prior_season":int(args.prior_season), "weeks":[int(w) for w in weeks],
        "evaluation_rows":int(len(frame)), "sportsbook_inputs_used":False, "model_fitting_used":False, "production_changed":False,
        "parent_wr_nd1_base_mae_expected":PARENT_BASE_MAE, "base_ypt_factorization_max_abs_drift":max_ypt_drift,
        "factorization_anomalies_excluded_from_target_result_view":int(len(anomaly)), "identity":identity,
        "disposition":disposition, "dominance_gate":gate,
    }
    (out/"wr_nd2_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print("[wr-nd2] overall Shapley")
    print(overall[["component","shapley_mae_recovery","base_mae","full_mae"]].to_string(index=False))
    print("[wr-nd2] routing")
    print(json.dumps({"disposition":disposition,**gate},indent=2))
    print("[wr-nd2] efficiency descriptives")
    print(desc.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
