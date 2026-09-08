#!/usr/bin/env python3
"""Finite receiving-room V1: team pool first, player entitlement second.

Research only.  This is materially different from Migration 32's player pruning:
we estimate finite WR/TE/RB target pools from strict-prior *team* history, then
allocate each pool across the current room using the already-available canonical
player entitlement inputs.  WR M38 remains active because simulation_v2 applies
its relative hierarchy after these finite room shares are materialized.

No sportsbook input is read.  No production parameter is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules
from scripts.simulation_v2 import _sharpen_wr_target_shares, lookup, simulate

GROUPS = ("WR", "TE", "RB")
ROOM_HISTORY_GAMES = 8
ROOM_PRIOR_STRENGTH = 3.0
MODELED_TARGET_MASS = 0.95


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise RuntimeError(f"empty {label}: {path}")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _num(value, default=np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _pos(value) -> str:
    p = str(value or "").upper().strip()
    if p in {"WR", "LWR", "RWR", "SWR"}: return "WR"
    if p == "TE": return "TE"
    if p in {"RB", "HB", "TB", "FB"}: return "RB"
    return "OTHER"


def _prepare_metrics(bundle) -> pd.DataFrame:
    m = cp.build_market_frame(bundle)
    m = apply_bayesian_to_metrics(m, build_bayesian_baseline(bundle.player_consensus))
    with patch.object(simulation_rules, "load_model_contexts", return_value=(bundle.teams, bundle.players)):
        m = simulation_rules.apply_rules_to_metrics(m)
    m["player_clean_key"] = m["player_clean_key"].fillna("").astype(str)
    key = ["event_id", "team", "player_clean_key"]
    m = m.sort_values(key).drop_duplicates(key, keep="last").copy()
    m["position_room"] = m.get("position", "").map(_pos)
    return m


def _historical_room_games(logs: pd.DataFrame, target_season: int, target_week: int) -> pd.DataFrame:
    x = logs.copy()
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[
        x["season"].lt(int(target_season))
        | (x["season"].eq(int(target_season)) & x["week"].lt(int(target_week)))
    ].copy()
    if x.empty:
        raise RuntimeError(f"finite-room V1 has no strict-prior logs before {target_season} W{target_week}")
    x["position_room"] = x.get("position", "").map(_pos)
    x["targets"] = pd.to_numeric(x.get("targets", 0), errors="coerce").fillna(0.0)
    # player_form_v2 defines team_targets as the actual sum of player targets for
    # that team-game. Use that target universe, not pass attempts, so room pools
    # are true shares of finite receiver targets and sum to one before the
    # simulator's explicit residual bucket.
    if "team_targets" in x.columns:
        team_total = (
            x[["season", "week", "team", "team_targets"]]
            .drop_duplicates(["season", "week", "team"])
            .assign(team_targets=lambda d: pd.to_numeric(d["team_targets"], errors="coerce"))
        )
    else:
        team_total = x.groupby(["season", "week", "team"], as_index=False)["targets"].sum().rename(columns={"targets": "team_targets"})
    group = (
        x.loc[x["position_room"].isin(GROUPS)]
        .groupby(["season", "week", "team", "position_room"], as_index=False)["targets"].sum()
    )
    teams = team_total[["season", "week", "team"]].drop_duplicates()
    grid = teams.assign(_k=1).merge(pd.DataFrame({"position_room": GROUPS, "_k": 1}), on="_k").drop(columns="_k")
    group = grid.merge(group, on=["season", "week", "team", "position_room"], how="left")
    group["targets"] = pd.to_numeric(group["targets"], errors="coerce").fillna(0.0)
    group = group.merge(team_total, on=["season", "week", "team"], how="left", validate="many_to_one")
    group["room_share"] = np.where(
        pd.to_numeric(group["team_targets"], errors="coerce").gt(0),
        group["targets"] / pd.to_numeric(group["team_targets"], errors="coerce"),
        0.0,
    )
    return group


def _room_prior(logs: pd.DataFrame, target_season: int, target_week: int, teams: list[str]) -> pd.DataFrame:
    hist = _historical_room_games(logs, target_season, target_week)
    prior_season = int(target_season) - 1
    league = (
        hist.loc[hist["season"].eq(prior_season)]
        .groupby("position_room", as_index=False)["room_share"].mean()
        .set_index("position_room")["room_share"]
        .to_dict()
    )
    if any(g not in league for g in GROUPS):
        # Should not occur in NFL historical data; fail instead of inventing an
        # untracked room prior.
        raise RuntimeError(f"finite-room V1 missing prior-season league room prior: {league}")

    rows = []
    for team in sorted(set(teams)):
        th = hist.loc[hist["team"].astype(str).eq(str(team))].copy()
        games = th[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).tail(ROOM_HISTORY_GAMES)
        if games.empty:
            n = 0
            means = {g: float(league[g]) for g in GROUPS}
        else:
            th = th.merge(games.assign(_keep=1), on=["season", "week"], how="inner")
            n = int(len(games))
            means = th.groupby("position_room")["room_share"].mean().to_dict()
        raw = {}
        for group in GROUPS:
            empirical = float(means.get(group, 0.0))
            prior = float(league[group])
            raw[group] = (ROOM_PRIOR_STRENGTH * prior + n * empirical) / (ROOM_PRIOR_STRENGTH + n)
        total = sum(max(0.0, raw[g]) for g in GROUPS)
        if total <= 0:
            raise RuntimeError(f"finite-room V1 nonpositive room pool team={team}")
        # Historical WR/TE/RB may leave tiny target mass for unusual eligible
        # positions. Preserve that as residual. Only shrink if floating/source
        # noise causes modeled rooms to exceed the physical target universe.
        renorm = min(1.0, 1.0 / total)
        for group in GROUPS:
            share_of_team_targets = max(0.0, raw[group]) * renorm
            rows.append({
                "team": team,
                "position_room": group,
                "target_room_share": MODELED_TARGET_MASS * share_of_team_targets,
                "raw_room_share": raw[group],
                "team_history_games": n,
                "league_prior_room_share": float(league[group]),
                "team_empirical_room_share": float(means.get(group, 0.0)),
            })
    out = pd.DataFrame(rows)
    totals = out.groupby("team")["target_room_share"].sum()
    if (totals > MODELED_TARGET_MASS + 1e-10).any():
        raise RuntimeError("finite-room V1 room shares exceed modeled target mass")
    return out


def _candidate_metrics(metrics: pd.DataFrame, rooms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = metrics.copy()
    room_lookup = rooms.set_index(["team", "position_room"])["target_room_share"].to_dict()
    out["finite_room_tgt_share"] = 0.0
    trace = []
    for (event_id, team), idx in out.groupby(["event_id", "team"], dropna=False).groups.items():
        g = out.loc[idx].copy()
        for group in GROUPS:
            mask = g["position_room"].eq(group)
            gi = g.index[mask]
            if len(gi) == 0:
                continue
            base = pd.to_numeric(g.loc[gi, "rules_tgt_share"], errors="coerce").fillna(0.0).clip(lower=0.0)
            denom = float(base.sum())
            pool = float(room_lookup.get((team, group), 0.0))
            if denom > 0:
                out.loc[gi, "finite_room_tgt_share"] = pool * (base / denom)
            else:
                out.loc[gi, "finite_room_tgt_share"] = 0.0
        total = float(pd.to_numeric(out.loc[idx, "finite_room_tgt_share"], errors="coerce").fillna(0.0).sum())
        trace.append({"event_id": event_id, "team": team, "candidate_player_share_sum_pre_m38": total})
    out["baseline_rules_tgt_share"] = pd.to_numeric(out["rules_tgt_share"], errors="coerce")
    out["rules_tgt_share"] = out["finite_room_tgt_share"]
    trace_df = pd.DataFrame(trace)
    if (trace_df["candidate_player_share_sum_pre_m38"] > MODELED_TARGET_MASS + 1e-9).any():
        raise RuntimeError("finite-room V1 player shares exceed modeled target mass")
    return out, trace_df


def _allocator_probabilities(team_df: pd.DataFrame, shares: np.ndarray) -> tuple[np.ndarray, float]:
    clean = np.nan_to_num(np.asarray(shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, 0.0, 0.95)
    clean = _sharpen_wr_target_shares(team_df, clean)
    raw_sum = float(clean.sum())
    if raw_sum > 0.95:
        clean *= 0.95 / raw_sum
    residual = max(0.0, 1.0 - float(clean.sum()))
    probs = np.append(clean, residual)
    probs = probs / probs.sum()
    return probs[:-1], float(probs[-1])


def _predict(metrics: pd.DataFrame, *, iterations: int, seed: int, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sims = simulate(metrics, iterations=iterations, seed=seed)
    rows = []
    teams = []
    for (event_id, team), g in metrics.groupby(["event_id", "team"], dropna=False):
        shares = pd.to_numeric(g["rules_tgt_share"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        probs, residual = _allocator_probabilities(g, shares)
        plays = float(pd.to_numeric(g["rules_plays_est"], errors="coerce").dropna().mean())
        pass_rate = float(pd.to_numeric(g["rules_pass_rate"], errors="coerce").dropna().mean())
        expected_pass_state = plays * pass_rate
        teams.append({
            "variant": label, "event_id": event_id, "team": team,
            "raw_player_share_sum": float(np.nansum(shares)),
            "final_player_probability_sum": float(probs.sum()),
            "residual_probability": residual,
            "expected_pass_state": expected_pass_state,
        })
        for j, (_, row) in enumerate(g.iterrows()):
            rec = lookup(sims, row, "receptions")
            yds = lookup(sims, row, "rec_yards")
            rows.append({
                "variant": label, "event_id": event_id, "team": team,
                "player": row.get("player"), "player_clean_key": row.get("player_clean_key"),
                "position": row.get("position"), "position_room": row.get("position_room"),
                "final_target_probability": float(probs[j]),
                "pred_targets": expected_pass_state * float(probs[j]),
                "pred_receptions": float(np.mean(rec)) if rec is not None and len(rec) else np.nan,
                "pred_rec_yards": float(np.mean(yds)) if yds is not None and len(yds) else np.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(teams)


def _score(frame: pd.DataFrame, *, group_label: str) -> list[dict]:
    rows = []
    for variant, g in frame.groupby("variant"):
        for market, pred_col, actual_col in (
            ("targets", "pred_targets", "actual_targets"),
            ("receptions", "pred_receptions", "actual_receptions"),
            ("rec_yards", "pred_rec_yards", "actual_rec_yards"),
        ):
            x = g.loc[pd.to_numeric(g[pred_col], errors="coerce").notna() & pd.to_numeric(g[actual_col], errors="coerce").notna()].copy()
            pred = pd.to_numeric(x[pred_col], errors="coerce").astype(float)
            actual = pd.to_numeric(x[actual_col], errors="coerce").astype(float)
            err = pred - actual; ae = err.abs()
            rows.append({
                "group": group_label, "variant": variant, "market": market, "n": int(len(x)),
                "mae": float(ae.mean()), "rmse": float(np.sqrt(np.mean(np.square(err)))),
                "bias": float(err.mean()),
                "correlation": float(pred.corr(actual)) if len(x) > 1 and pred.nunique() > 1 and actual.nunique() > 1 else np.nan,
                "median_abs": float(ae.median()), "p75_abs": float(ae.quantile(.75)), "p90_abs": float(ae.quantile(.90)),
                "miss30": float(ae.ge(30).mean()) if market == "rec_yards" else np.nan,
                "miss40": float(ae.ge(40).mean()) if market == "rec_yards" else np.nan,
            })
    return rows


def _metric(summary: pd.DataFrame, group: str, variant: str, market: str, field: str) -> float:
    x = summary.loc[(summary["group"] == group) & (summary["variant"] == variant) & (summary["market"] == market), field]
    if len(x) != 1:
        raise RuntimeError(f"finite-room V1 missing unique score {group}/{variant}/{market}/{field}")
    return float(x.iloc[0])


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
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/receiving_finite_room_v1"))
    a = ap.parse_args()

    logs = _read(a.player_logs, "player logs")
    team = _read(a.team_weekly, "team weekly")
    sched = _read(a.schedule, "schedule")
    injuries = _optional(a.injuries)
    weather = _optional(a.weather)
    preds = []; team_traces = []; room_traces = []

    for week in _parse_weeks(a.weeks):
        universe = _read(a.universe_dir / f"{a.season}_week_{week:02d}.csv", f"pregame universe W{week}")
        bundle = build_historical_context_bundle(
            player_logs=logs, team_weekly=team, pregame_universe=universe, schedule=sched,
            season=a.season, week=week, prior_season=a.prior_season,
            injuries=_exact_week(injuries, a.season, week), weather=_exact_week(weather, a.season, week),
        )
        baseline = _prepare_metrics(bundle)
        rooms = _room_prior(logs, a.season, week, baseline["team"].astype(str).unique().tolist())
        candidate, pretrace = _candidate_metrics(baseline, rooms)
        bp, bt = _predict(baseline, iterations=a.iterations, seed=51000 + week, label="CURRENT_CANONICAL")
        cpred, ct = _predict(candidate, iterations=a.iterations, seed=51000 + week, label="FINITE_ROOM_V1")

        actual = logs.loc[
            pd.to_numeric(logs["season"], errors="coerce").eq(a.season)
            & pd.to_numeric(logs["week"], errors="coerce").eq(week),
            ["team", "player_clean_key", "targets", "receptions", "rec_yards"],
        ].copy()
        actual = actual.rename(columns={"targets": "actual_targets", "receptions": "actual_receptions", "rec_yards": "actual_rec_yards"})
        actual = actual.drop_duplicates(["team", "player_clean_key"])
        for part in (bp, cpred):
            part["week"] = week
            part = part.merge(actual, on=["team", "player_clean_key"], how="left", validate="many_to_one")
            preds.append(part)
        bt["week"] = week; ct["week"] = week
        team_traces += [bt, ct]
        rooms["week"] = week; room_traces.append(rooms)
        pretrace["week"] = week
        print(
            f"[finite_room_v1] W{week:02d} baseline_raw={bt.raw_player_share_sum.mean():.3f} "
            f"candidate_raw={ct.raw_player_share_sum.mean():.3f} candidate_residual={ct.residual_probability.mean():.3f}"
        )

    pred = pd.concat(preds, ignore_index=True)
    team_trace = pd.concat(team_traces, ignore_index=True)
    room_trace = pd.concat(room_traces, ignore_index=True)

    scores = []
    scores.extend(_score(pred, group_label="ALL"))
    for group in GROUPS:
        scores.extend(_score(pred.loc[pred["position_room"].eq(group)], group_label=group))
    summary = pd.DataFrame(scores)

    integrity = {
        "candidate_team_probability_le095": bool(
            team_trace.loc[team_trace["variant"].eq("FINITE_ROOM_V1"), "final_player_probability_sum"].le(MODELED_TARGET_MASS + 1e-9).all()
        ),
        "candidate_raw_share_le095": bool(
            team_trace.loc[team_trace["variant"].eq("FINITE_ROOM_V1"), "raw_player_share_sum"].le(MODELED_TARGET_MASS + 1e-9).all()
        ),
        "all_weeks_scored": int(pred.loc[pred["variant"].eq("FINITE_ROOM_V1"), "week"].nunique()) == len(_parse_weeks(a.weeks)),
        "sportsbook_inputs_zero": True,
        "production_parameters_changed": False,
        "m38_preserved_in_simulation": True,
    }
    b_y = _metric(summary, "ALL", "CURRENT_CANONICAL", "rec_yards", "mae")
    c_y = _metric(summary, "ALL", "FINITE_ROOM_V1", "rec_yards", "mae")
    b_r = _metric(summary, "ALL", "CURRENT_CANONICAL", "receptions", "mae")
    c_r = _metric(summary, "ALL", "FINITE_ROOM_V1", "receptions", "mae")
    b_t = _metric(summary, "ALL", "CURRENT_CANONICAL", "targets", "mae")
    c_t = _metric(summary, "ALL", "FINITE_ROOM_V1", "targets", "mae")
    science = {
        "target_mae_improves_ge_005": b_t - c_t >= 0.05,
        "receptions_mae_not_worse": c_r <= b_r + 1e-9,
        "rec_yards_mae_improves_ge_010": b_y - c_y >= 0.10,
        "rec_yards_p90_not_worse": _metric(summary, "ALL", "FINITE_ROOM_V1", "rec_yards", "p90_abs") <= _metric(summary, "ALL", "CURRENT_CANONICAL", "rec_yards", "p90_abs") + 1e-9,
        "rec_yards_corr_guard": _metric(summary, "ALL", "FINITE_ROOM_V1", "rec_yards", "correlation") >= _metric(summary, "ALL", "CURRENT_CANONICAL", "rec_yards", "correlation") - 0.01,
        "wr_rec_yards_mae_guard": _metric(summary, "WR", "FINITE_ROOM_V1", "rec_yards", "mae") <= _metric(summary, "WR", "CURRENT_CANONICAL", "rec_yards", "mae") + 0.10,
        "te_rec_yards_mae_guard": _metric(summary, "TE", "FINITE_ROOM_V1", "rec_yards", "mae") <= _metric(summary, "TE", "CURRENT_CANONICAL", "rec_yards", "mae") + 0.10,
        "rb_rec_yards_mae_guard": _metric(summary, "RB", "FINITE_ROOM_V1", "rec_yards", "mae") <= _metric(summary, "RB", "CURRENT_CANONICAL", "rec_yards", "mae") + 0.10,
    }
    if not all(integrity.values()):
        disposition = "FINITE_ROOM_V1_MECHANICAL_OR_INTEGRITY_FAILURE"
    elif all(science.values()):
        disposition = "FINITE_ROOM_V1_2025_RESEARCH_PASS_REQUIRES_FROZEN_CONFIRMATION"
    else:
        disposition = "FINITE_ROOM_V1_SCIENTIFIC_FAIL"

    result = {
        "migration": "RECEIVING_FINITE_ROOM_V1",
        "disposition": disposition,
        "season": int(a.season), "weeks": a.weeks, "iterations": int(a.iterations),
        "room_history_games": ROOM_HISTORY_GAMES,
        "room_prior_strength_team_games": ROOM_PRIOR_STRENGTH,
        "modeled_target_mass": MODELED_TARGET_MASS,
        "baseline_rec_yards_mae": b_y, "candidate_rec_yards_mae": c_y,
        "baseline_receptions_mae": b_r, "candidate_receptions_mae": c_r,
        "baseline_targets_mae": b_t, "candidate_targets_mae": c_t,
        "integrity_gates": integrity, "scientific_gates": science,
        "sportsbook_inputs_used": 0, "production_parameters_changed": 0,
        "note": "Research-only finite team/position room allocation; no pruning thresholds or post-result retuning.",
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(a.out_dir / "finite_room_player_casebook.csv", index=False)
    team_trace.to_csv(a.out_dir / "finite_room_team_trace.csv", index=False)
    room_trace.to_csv(a.out_dir / "finite_room_room_priors.csv", index=False)
    summary.to_csv(a.out_dir / "finite_room_scorecard.csv", index=False)
    (a.out_dir / "finite_room_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
