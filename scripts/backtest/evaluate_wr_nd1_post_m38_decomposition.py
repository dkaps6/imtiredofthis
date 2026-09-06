#!/usr/bin/env python3
"""WR-ND1: exact post-M38 WR residual Shapley decomposition.

Diagnostic only. No model fitting, sportsbook inputs, or production mutation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules
from scripts.simulation_v2 import (
    WR_POSITIONS,
    WR_TARGET_HIERARCHY_MULTIPLIERS,
    _clip_prob,
    _num,
    _sharpen_wr_target_shares,
    _team_inputs,
)

WR_POS = set(WR_POSITIONS)
EXPECTED_M38 = (1.40, 1.14, 0.91, 0.78)
TARGET_COMPONENTS = ("TEAM_TARGET_VOLUME", "WR_TARGET_MASS", "WITHIN_WR_ALLOCATION")
YARD_COMPONENTS = TARGET_COMPONENTS + ("YARDS_PER_TARGET",)
REC_COMPONENTS = TARGET_COMPONENTS + ("CATCH_RATE",)


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing required input: {path}")
    return pd.read_csv(path)


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def score(actual: pd.Series, pred: pd.Series) -> dict:
    z = pd.DataFrame({"actual": pd.to_numeric(actual, errors="coerce"), "pred": pd.to_numeric(pred, errors="coerce")}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan}
    err = z.pred - z.actual
    corr = float(z.pred.corr(z.actual)) if len(z) > 1 and z.pred.nunique() > 1 and z.actual.nunique() > 1 else np.nan
    return {
        "n": int(len(z)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "bias": float(err.mean()),
        "correlation": corr,
    }


def allocator_probabilities(shares: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, 0.0, 0.95)
    total = float(clean.sum())
    if total > 0.95:
        clean *= 0.95 / total
    residual = max(0.0, 1.0 - float(clean.sum()))
    probs = np.append(clean, residual)
    probs /= probs.sum()
    return probs[:-1]


def prepared(bundle) -> pd.DataFrame:
    frame = cp.build_market_frame(bundle)
    frame = apply_bayesian_to_metrics(frame, build_bayesian_baseline(bundle.player_consensus))
    with patch.object(simulation_rules, "load_model_contexts", return_value=(bundle.teams, bundle.players)):
        frame = simulation_rules.apply_rules_to_metrics(frame)
    frame["player_clean_key"] = frame.player_clean_key.fillna("").astype(str)
    keys = ["event_id", "team", "player_clean_key"]
    return frame.sort_values(keys).drop_duplicates(keys, keep="last").copy()


def actual_week(logs: pd.DataFrame, season: int, week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "team", "player", "position", "targets", "receptions", "rec_yards"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"WR-ND1 actual logs missing columns: {sorted(missing)}")
    x["season"] = pd.to_numeric(x.season, errors="coerce")
    x["week"] = pd.to_numeric(x.week, errors="coerce")
    x = x.loc[x.season.eq(int(season)) & x.week.eq(int(week))].copy()
    if x.empty:
        raise RuntimeError(f"WR-ND1 no target-game logs for {season} week {week}")
    x["team"] = x.team.fillna("").astype(str).str.upper().str.strip()
    x["position"] = x.position.fillna("").astype(str).str.upper().str.strip()
    x["player_clean_key"] = x.get("player_clean_key", x.player).map(cp._key)
    for col in ("targets", "receptions", "rec_yards"):
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    bad = x.loc[(x.targets <= 0) & (x.rec_yards.abs() > 1e-9)]
    if not bad.empty:
        raise RuntimeError(f"WR-ND1 found receiving yards with zero targets: n={len(bad)}")
    player = (
        x.groupby(["team", "player_clean_key"], as_index=False)
        .agg(
            player=("player", "first"),
            actual_position=("position", "first"),
            actual_targets=("targets", "sum"),
            actual_receptions=("receptions", "sum"),
            actual_rec_yards=("rec_yards", "sum"),
        )
    )
    team = x.groupby("team", as_index=False).agg(actual_team_targets=("targets", "sum"))
    wr = x.loc[x.position.isin(WR_POS)].groupby("team", as_index=False).agg(actual_wr_targets=("targets", "sum"))
    team = team.merge(wr, on="team", how="left")
    team["actual_wr_targets"] = team.actual_wr_targets.fillna(0.0)
    if (team.actual_team_targets <= 0).any():
        bad_teams = team.loc[team.actual_team_targets <= 0, "team"].tolist()
        raise RuntimeError(f"WR-ND1 actual team targets non-positive: {bad_teams}")
    team["actual_wr_mass"] = team.actual_wr_targets / team.actual_team_targets
    return player, team


def build_week_factors(bundle, logs: pd.DataFrame, season: int, week: int, iterations: int) -> pd.DataFrame:
    if tuple(float(x) for x in WR_TARGET_HIERARCHY_MULTIPLIERS) != EXPECTED_M38:
        raise RuntimeError(
            f"WR-ND1 M38 multiplier drift: got={WR_TARGET_HIERARCHY_MULTIPLIERS} expected={EXPECTED_M38}"
        )
    m = prepared(bundle)
    mc = cp.build_mc_predictions(bundle, iterations=int(iterations), seed=42 + int(week))
    mr = mc.loc[mc.market.eq("receptions"), ["event_id", "team", "player_clean_key", "mc_proj"]].rename(columns={"mc_proj": "mc_receptions"})
    my = mc.loc[mc.market.eq("rec_yards"), ["event_id", "team", "player_clean_key", "mc_proj"]].rename(columns={"mc_proj": "mc_rec_yards"})
    actual_player, actual_team = actual_week(logs, season, week)

    rows: list[dict] = []
    for (event_id, team), g0 in m.groupby(["event_id", "team"], dropna=False):
        g = g0.copy().reset_index(drop=True)
        raw = np.array([
            _num(r, "rules_tgt_share", "bayes_tgt_share", "target_share", "tgt_share", default=0.0)
            for _, r in g.iterrows()
        ], dtype=float)
        positions = g.position.fillna("").astype(str).str.upper().str.strip().to_numpy()
        wr_idx = np.flatnonzero(np.isin(positions, list(WR_POS)))
        if not len(wr_idx):
            continue
        raw_clean = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 0.95)
        post = _sharpen_wr_target_shares(g, raw)
        raw_wr_mass = float(raw_clean[wr_idx].sum())
        post_wr_share_mass = float(np.asarray(post, dtype=float)[wr_idx].sum())
        if abs(raw_wr_mass - post_wr_share_mass) > 1e-12:
            raise RuntimeError(
                f"WR-ND1 M38 WR-share mass drift event={event_id} team={team} raw={raw_wr_mass} post={post_wr_share_mass}"
            )
        probs = allocator_probabilities(post)
        pred_wr_mass = float(probs[wr_idx].sum())
        plays_mean, pass_rate_mean = _team_inputs(g)
        pred_team_targets = float(plays_mean * pass_rate_mean)
        if pred_team_targets <= 0:
            raise RuntimeError(f"WR-ND1 non-positive predicted team target volume event={event_id} team={team}")

        wr_order = wr_idx[np.argsort(-raw_clean[wr_idx], kind="stable")]
        rank_by_index = {int(idx): rank + 1 for rank, idx in enumerate(wr_order)}
        for j in wr_idx:
            r = g.iloc[int(j)]
            pred_within = float(probs[j] / pred_wr_mass) if pred_wr_mass > 0 else 0.0
            catch = _clip_prob(_num(r, "rules_catch_rate", "bayes_receptions_per_target", "receptions_per_target", "catch_rate", default=0.64), 0.64)
            ypt = _num(r, "rules_ypt", "bayes_ypt", "ypt", default=7.5)
            ypt = 7.5 if not np.isfinite(ypt) or ypt <= 0 else float(ypt)
            rank = int(rank_by_index[int(j)])
            role = "WR1" if rank == 1 else "WR2" if rank == 2 else "WR3" if rank == 3 else "WR4_PLUS"
            rows.append({
                "season": int(season),
                "week": int(week),
                "event_id": str(event_id),
                "team": str(team).upper().strip(),
                "player_clean_key": str(r.player_clean_key),
                "player": r.get("player", ""),
                "position": str(r.get("position", "")).upper().strip(),
                "wr_role": role,
                "raw_target_share": float(raw_clean[j]),
                "post_m38_target_share": float(post[j]),
                "pred_team_targets": pred_team_targets,
                "pred_wr_mass": pred_wr_mass,
                "pred_within_wr": pred_within,
                "pred_catch_rate": float(catch),
                "pred_ypt": float(ypt),
                "pred_targets": pred_team_targets * pred_wr_mass * pred_within,
                "pred_receptions": pred_team_targets * pred_wr_mass * pred_within * float(catch),
                "pred_rec_yards": pred_team_targets * pred_wr_mass * pred_within * float(ypt),
                "m38_wr_share_mass_before": raw_wr_mass,
                "m38_wr_share_mass_after": post_wr_share_mass,
            })

    pred = pd.DataFrame(rows)
    if pred.empty:
        raise RuntimeError(f"WR-ND1 produced no WR factors for week {week}")
    x = pred.merge(actual_player, on=["team", "player_clean_key"], how="inner", validate="one_to_one")
    x = x.merge(actual_team, on="team", how="left", validate="many_to_one")
    x = x.merge(mr, on=["event_id", "team", "player_clean_key"], how="left", validate="one_to_one")
    x = x.merge(my, on=["event_id", "team", "player_clean_key"], how="left", validate="one_to_one")
    if x.empty:
        raise RuntimeError(f"WR-ND1 no matched projected/actual WR rows for week {week}")
    if x[["actual_team_targets", "actual_wr_targets", "actual_wr_mass"]].isna().any().any():
        raise RuntimeError(f"WR-ND1 team truth join incomplete for week {week}")

    x["actual_within_wr"] = np.where(x.actual_wr_targets > 0, x.actual_targets / x.actual_wr_targets, 0.0)
    x["actual_ypt"] = np.where(x.actual_targets > 0, x.actual_rec_yards / x.actual_targets, x.pred_ypt)
    x["actual_catch_rate"] = np.where(x.actual_targets > 0, x.actual_receptions / x.actual_targets, x.pred_catch_rate)
    x["base_yard_error"] = x.pred_rec_yards - x.actual_rec_yards
    x["yard_error_direction"] = np.where(x.base_yard_error >= 0, "OVER", "UNDER")
    x["week_phase"] = pd.cut(x.week, [0, 4, 9, 13, 18], labels=["W1-4", "W5-9", "W10-13", "W14-18"]).astype(str)
    x["actual_target_tier"] = pd.cut(
        x.actual_targets,
        [-np.inf, 3, 6, 9, np.inf],
        labels=["0-3", "4-6", "7-9", "10+"],
    ).astype(str)
    return x


def prediction(frame: pd.DataFrame, outcome: str, corrected: frozenset[str]) -> pd.Series:
    team = frame.actual_team_targets if "TEAM_TARGET_VOLUME" in corrected else frame.pred_team_targets
    wr_mass = frame.actual_wr_mass if "WR_TARGET_MASS" in corrected else frame.pred_wr_mass
    within = frame.actual_within_wr if "WITHIN_WR_ALLOCATION" in corrected else frame.pred_within_wr
    targets = team * wr_mass * within
    if outcome == "targets":
        return targets
    if outcome == "rec_yards":
        eff = frame.actual_ypt if "YARDS_PER_TARGET" in corrected else frame.pred_ypt
        return targets * eff
    if outcome == "receptions":
        eff = frame.actual_catch_rate if "CATCH_RATE" in corrected else frame.pred_catch_rate
        return targets * eff
    raise ValueError(outcome)


def actual_series(frame: pd.DataFrame, outcome: str) -> pd.Series:
    return {
        "targets": frame.actual_targets,
        "rec_yards": frame.actual_rec_yards,
        "receptions": frame.actual_receptions,
    }[outcome]


def exact_shapley(frame: pd.DataFrame, outcome: str, components: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    ncomp = len(components)
    subset_mae: dict[frozenset[str], float] = {}
    for mask in range(1 << ncomp):
        subset = frozenset(components[i] for i in range(ncomp) if mask & (1 << i))
        s = score(actual_series(frame, outcome), prediction(frame, outcome, subset))
        subset_mae[subset] = float(s["mae"])
    empty = frozenset()
    full = frozenset(components)
    rows = []
    for comp in components:
        phi = 0.0
        for subset, base_mae in subset_mae.items():
            if comp in subset:
                continue
            k = len(subset)
            weight = math.factorial(k) * math.factorial(ncomp - k - 1) / math.factorial(ncomp)
            phi += weight * (base_mae - subset_mae[frozenset(set(subset) | {comp})])
        rows.append({"outcome": outcome, "component": comp, "shapley_mae_recovery": float(phi)})
    base_mae = subset_mae[empty]
    full_mae = subset_mae[full]
    recovery = base_mae - full_mae
    shapley_sum = float(sum(r["shapley_mae_recovery"] for r in rows))
    if abs(shapley_sum - recovery) > 1e-8:
        raise RuntimeError(
            f"WR-ND1 Shapley identity failed outcome={outcome}: sum={shapley_sum} recovery={recovery}"
        )
    pred_full = prediction(frame, outcome, full)
    truth = actual_series(frame, outcome)
    tol = 1e-9 if outcome == "targets" else 1e-8
    max_identity = float((pd.to_numeric(pred_full) - pd.to_numeric(truth)).abs().max())
    if max_identity > tol:
        raise RuntimeError(
            f"WR-ND1 full-oracle identity failed outcome={outcome}: max_abs_diff={max_identity} tol={tol}"
        )
    meta = {
        "outcome": outcome,
        "n": int(len(frame)),
        "base_mae": float(base_mae),
        "full_mae": float(full_mae),
        "total_mae_recovery": float(recovery),
        "shapley_sum": shapley_sum,
        "max_full_identity_abs_diff": max_identity,
    }
    for r in rows:
        r.update(meta)
    return pd.DataFrame(rows), meta


def stage_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, actual_col, det_col, mc_col in (
        ("targets", "actual_targets", "pred_targets", None),
        ("receptions", "actual_receptions", "pred_receptions", "mc_receptions"),
        ("rec_yards", "actual_rec_yards", "pred_rec_yards", "mc_rec_yards"),
    ):
        for stage, col in (("POST_M38_DETERMINISTIC", det_col), ("CANONICAL_MC_REFERENCE", mc_col)):
            if col is None:
                continue
            rows.append({"outcome": outcome, "stage": stage, **score(frame[actual_col], frame[col])})
        if outcome == "targets":
            rows.append({"outcome": outcome, "stage": "POST_M38_DETERMINISTIC", **score(frame[actual_col], frame[det_col])})
    return pd.DataFrame(rows)


def slice_specs(frame: pd.DataFrame):
    yield "ALL_WR", "ALL_WR", frame
    for role in ("WR1", "WR2", "WR3", "WR4_PLUS"):
        yield "wr_role", role, frame.loc[frame.wr_role.eq(role)]
    for val in ("OVER", "UNDER"):
        yield "yard_error_direction", val, frame.loc[frame.yard_error_direction.eq(val)]
    for val in ("W1-4", "W5-9", "W10-13", "W14-18"):
        yield "week_phase", val, frame.loc[frame.week_phase.eq(val)]
    for val in ("0-3", "4-6", "7-9", "10+"):
        yield "actual_target_tier", val, frame.loc[frame.actual_target_tier.eq(val)]


def disposition(shapley: pd.DataFrame, slices: pd.DataFrame) -> tuple[str, dict]:
    overall = shapley.loc[shapley.outcome.eq("rec_yards")].copy()
    if overall.empty:
        raise RuntimeError("WR-ND1 missing overall receiving-yard Shapley rows")
    overall = overall.sort_values("shapley_mae_recovery", ascending=False)
    top = str(overall.iloc[0].component)
    positive_total = float(overall.shapley_mae_recovery.clip(lower=0).sum())
    top_value = float(overall.iloc[0].shapley_mae_recovery)
    share = top_value / positive_total if positive_total > 0 and top_value > 0 else 0.0
    role_wins = 0
    role_tops = {}
    for role in ("WR1", "WR2", "WR3"):
        r = slices.loc[
            slices.outcome.eq("rec_yards")
            & slices.slice_type.eq("wr_role")
            & slices.slice_value.eq(role)
        ].sort_values("shapley_mae_recovery", ascending=False)
        role_top = str(r.iloc[0].component) if not r.empty else ""
        role_tops[role] = role_top
        role_wins += int(role_top == top)
    mapping = {
        "WITHIN_WR_ALLOCATION": "WITHIN_WR_ALLOCATION_DOMINANT",
        "WR_TARGET_MASS": "WR_TARGET_MASS_DOMINANT",
        "TEAM_TARGET_VOLUME": "TEAM_TARGET_VOLUME_DOMINANT",
        "YARDS_PER_TARGET": "YARDS_PER_TARGET_DOMINANT",
    }
    passes = top_value > 0 and share >= 0.40 and role_wins >= 2 and top in mapping
    disp = mapping[top] if passes else "MIXED_POST_M38_WR_ERROR"
    return disp, {
        "overall_top_component": top,
        "overall_top_shapley_mae_recovery": top_value,
        "overall_positive_shapley_share": float(share),
        "wr1_wr2_wr3_same_top_count": int(role_wins),
        "role_top_components": role_tops,
        "dominance_gate_passed": bool(passes),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--prior-season", type=int, default=2024)
    p.add_argument("--weeks", default="1-18")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--player-logs", type=Path, default=Path("data/backtests/player_game_logs_history.csv"))
    p.add_argument("--team-weekly", type=Path, default=Path("data/backtests/team_weekly_history.csv"))
    p.add_argument("--schedule", type=Path, default=Path("data/backtests/schedule_history.csv"))
    p.add_argument("--universe-dir", type=Path, default=Path("data/backtests/pregame_universe"))
    p.add_argument("--injuries", type=Path, default=Path("data/backtests/injuries_history.csv"))
    p.add_argument("--weather", type=Path, default=Path("data/backtests/weather_history.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/wr_nd1_post_m38"))
    args = p.parse_args()

    logs = read_required(args.player_logs)
    team = read_required(args.team_weekly)
    sched = read_required(args.schedule)
    inj = read_optional(args.injuries)
    weather = read_optional(args.weather)

    weeks = _parse_weeks(args.weeks)
    all_rows = []
    for week in weeks:
        universe = read_required(args.universe_dir / f"{args.season}_week_{week:02d}.csv")
        bundle = build_historical_context_bundle(
            player_logs=logs,
            team_weekly=team,
            pregame_universe=universe,
            schedule=sched,
            season=args.season,
            week=week,
            prior_season=args.prior_season,
            injuries=_exact_week(inj, args.season, week),
            weather=_exact_week(weather, args.season, week),
        )
        wf = build_week_factors(bundle, logs, args.season, week, args.iterations)
        all_rows.append(wf)
        print(
            f"[wr-nd1] W{week:02d} matched_wr={len(wf)} "
            f"det_yard_mae={(wf.pred_rec_yards-wf.actual_rec_yards).abs().mean():.4f}"
        )

    frame = pd.concat(all_rows, ignore_index=True)
    if frame.empty:
        raise RuntimeError("WR-ND1 produced no evaluation rows")
    if frame.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("WR-ND1 duplicate player-week identities")
    if float((frame.m38_wr_share_mass_after - frame.m38_wr_share_mass_before).abs().max()) > 1e-12:
        raise RuntimeError("WR-ND1 M38 mass-preservation audit failed after concatenation")

    stages = stage_rows(frame)
    overall_parts = []
    overall_meta = []
    for outcome, comps in (("targets", TARGET_COMPONENTS), ("receptions", REC_COMPONENTS), ("rec_yards", YARD_COMPONENTS)):
        parts, meta = exact_shapley(frame, outcome, comps)
        overall_parts.append(parts)
        overall_meta.append(meta)
    overall = pd.concat(overall_parts, ignore_index=True)

    sliced = []
    for slice_type, slice_value, sf in slice_specs(frame):
        if sf.empty:
            continue
        for outcome, comps in (("targets", TARGET_COMPONENTS), ("receptions", REC_COMPONENTS), ("rec_yards", YARD_COMPONENTS)):
            parts, _ = exact_shapley(sf, outcome, comps)
            parts.insert(0, "slice_value", str(slice_value))
            parts.insert(0, "slice_type", str(slice_type))
            sliced.append(parts)
    slices = pd.concat(sliced, ignore_index=True)

    disp, gate = disposition(overall, slices)
    summary = {
        "migration": "WR-ND1",
        "branch": "research-wr-nd1-post-m38-decomposition",
        "season": int(args.season),
        "prior_season": int(args.prior_season),
        "weeks": [int(w) for w in weeks],
        "evaluation_rows": int(len(frame)),
        "teams": int(frame.team.nunique()),
        "m38_multipliers": list(EXPECTED_M38),
        "m38_max_wr_share_mass_drift": float((frame.m38_wr_share_mass_after - frame.m38_wr_share_mass_before).abs().max()),
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": disp,
        "dominance_gate": gate,
        "overall_identity": overall_meta,
        "stage_metrics": stages.to_dict(orient="records"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "wr_nd1_player_factors.csv", index=False)
    stages.to_csv(args.out_dir / "wr_nd1_stage_metrics.csv", index=False)
    overall.to_csv(args.out_dir / "wr_nd1_shapley_summary.csv", index=False)
    slices.to_csv(args.out_dir / "wr_nd1_slice_shapley.csv", index=False)
    (args.out_dir / "wr_nd1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n[wr-nd1] stage metrics")
    print(stages.to_string(index=False))
    print("\n[wr-nd1] overall Shapley")
    print(overall[["outcome", "component", "shapley_mae_recovery", "base_mae", "full_mae"]].to_string(index=False))
    print("\n[wr-nd1] routing")
    print(json.dumps({"disposition": disp, **gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
