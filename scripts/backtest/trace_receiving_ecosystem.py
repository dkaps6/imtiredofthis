#!/usr/bin/env python3
"""Frozen receiving-ecosystem trace for WR/TE/RB/FB and QB conservation.

This is diagnostic only.  It replays the exact pregame Bayesian/rules/M38 joint
simulation, records the player-level receiving/rushing means that simulation
produces, and separately exports target-week actual usage.  Target-week outcomes
are never used while building the projection side.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules
from scripts import simulation_v2

PASS_CATCHER_POSITIONS = {"WR", "LWR", "RWR", "SWR", "TE", "RB", "FB"}


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing required input: {path}")
    return pd.read_csv(path, low_memory=False)


def opt(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() and path.stat().st_size else pd.DataFrame()


def numeric(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def finite(value, default=0.0) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def group(position) -> str:
    pos = str(position or "").upper().strip()
    if pos in {"WR", "LWR", "RWR", "SWR"}:
        return "WR"
    if pos == "TE":
        return "TE"
    if pos == "RB":
        return "RB"
    if pos == "FB":
        return "FB"
    return "OTHER"


def prepared(bundle) -> pd.DataFrame:
    metrics = cp.build_market_frame(bundle)
    metrics = apply_bayesian_to_metrics(
        metrics,
        build_bayesian_baseline(bundle.player_consensus),
    )
    with patch.object(
        simulation_rules,
        "load_model_contexts",
        return_value=(bundle.teams, bundle.players),
    ):
        metrics = simulation_rules.apply_rules_to_metrics(metrics)
    metrics["player_clean_key"] = metrics["player_clean_key"].fillna("").astype(str)
    return metrics.copy()


def target_probabilities(team_df: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    raw = np.array(
        [
            simulation_v2._num(
                row,
                "rules_tgt_share",
                "bayes_tgt_share",
                "target_share",
                "tgt_share",
                default=0.0,
            )
            for _, row in team_df.iterrows()
        ],
        dtype=float,
    )
    sharpened = simulation_v2._sharpen_wr_target_shares(team_df, raw)
    clean = np.clip(
        np.nan_to_num(sharpened.astype(float), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        0.95,
    )
    raw_sum = float(clean.sum())
    if raw_sum > 0.95:
        clean *= 0.95 / raw_sum
    residual = max(0.0, 1.0 - float(clean.sum()))
    probs = np.append(clean, residual)
    probs /= probs.sum()
    modeled_pass_catcher = np.array(
        [str(v or "").upper().strip() in PASS_CATCHER_POSITIONS for v in team_df["position"]],
        dtype=bool,
    )
    modeled_receiver_mass = float(probs[:-1][modeled_pass_catcher].sum())
    return probs[:-1], float(probs[-1]), modeled_receiver_mass


def arr(sim, game, key, market):
    value = sim.values.get((str(game), str(key), str(market)))
    return value if value is not None and len(value) else None


def qstats(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {
            "mean_signed_gap": np.nan,
            "mean_abs_gap": np.nan,
            "median_abs_gap": np.nan,
            "p90_abs_gap": np.nan,
            "p95_abs_gap": np.nan,
        }
    a = np.abs(x)
    return {
        "mean_signed_gap": float(np.mean(x)),
        "mean_abs_gap": float(np.mean(a)),
        "median_abs_gap": float(np.median(a)),
        "p90_abs_gap": float(np.quantile(a, 0.90)),
        "p95_abs_gap": float(np.quantile(a, 0.95)),
    }


def actual_usage(logs: pd.DataFrame, season: int, weeks: set[int]) -> pd.DataFrame:
    x = logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["season"] = numeric(x, "season")
    x["week"] = numeric(x, "week")
    x = x.loc[x["season"].eq(int(season)) & x["week"].isin(sorted(weeks))].copy()
    for c in ["targets", "receptions", "rec_yards", "rushes", "rush_yards", "pass_att", "pass_yards"]:
        x[c] = numeric(x, c, 0.0).fillna(0.0)
    for c in ["player", "player_clean_key", "player_identity_key", "team", "position"]:
        if c not in x.columns:
            x[c] = ""
        x[c] = x[c].fillna("").astype(str)
    x["position_group"] = x["position"].map(group)
    x["rush_rec_yards"] = x["rush_yards"] + x["rec_yards"]
    x["join_key"] = np.where(
        x["player_identity_key"].str.strip().ne(""),
        "id:" + x["player_identity_key"].str.strip(),
        "name:" + x["player_clean_key"].str.strip(),
    )
    keep = [
        "season", "week", "team", "player", "player_clean_key", "player_identity_key",
        "join_key", "position", "position_group", "targets", "receptions", "rec_yards",
        "rushes", "rush_yards", "rush_rec_yards", "pass_att", "pass_yards",
    ]
    return x[keep].sort_values(["season", "week", "team", "join_key"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--prior-season", type=int, required=True)
    p.add_argument("--weeks", required=True)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--universe-dir", type=Path, required=True)
    p.add_argument("--injuries", type=Path, required=True)
    p.add_argument("--weather", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    logs = read(args.player_logs)
    team_weekly = read(args.team_weekly)
    schedule = read(args.schedule)
    injuries = opt(args.injuries)
    weather = opt(args.weather)
    weeks = _parse_weeks(args.weeks)

    player_rows: list[dict] = []
    conservation_rows: list[dict] = []

    for week in weeks:
        universe = read(args.universe_dir / f"{args.season}_week_{week:02d}.csv")
        bundle = build_historical_context_bundle(
            player_logs=logs,
            team_weekly=team_weekly,
            pregame_universe=universe,
            schedule=schedule,
            season=args.season,
            week=week,
            prior_season=args.prior_season,
            injuries=_exact_week(injuries, args.season, week),
            weather=_exact_week(weather, args.season, week),
        )
        metrics = prepared(bundle)
        sim = simulation_v2.simulate(metrics, iterations=args.iterations, seed=42 + week)

        key_cols = ["event_id", "team", "player_clean_key"]
        players = (
            metrics.sort_values(key_cols)
            .drop_duplicates(key_cols, keep="last")
            .reset_index(drop=True)
        )

        for (game, team), team_df in players.groupby(["event_id", "team"], dropna=False, sort=False):
            team_df = team_df.reset_index(drop=True)
            probs, residual_prob, modeled_receiver_mass = target_probabilities(team_df)
            plays_mean, pass_rate_mean = simulation_v2._team_inputs(team_df)
            expected_team_pass_count = float(plays_mean * pass_rate_mean)

            receiver_arrays = []
            qb_candidates = []
            for j, row in team_df.iterrows():
                pkey = str(row.get("player_clean_key", "") or "")
                position = str(row.get("position", "") or "").upper().strip()
                pos_group = group(position)
                rec = arr(sim, game, pkey, "receptions")
                rec_yards = arr(sim, game, pkey, "rec_yards")
                rush_yards = arr(sim, game, pkey, "rush_yards")
                rush_rec = arr(sim, game, pkey, "rush_rec_yards")
                pass_yards = arr(sim, game, pkey, "pass_yards")
                if position in PASS_CATCHER_POSITIONS and rec_yards is not None:
                    receiver_arrays.append(rec_yards)
                if position == "QB" and pass_yards is not None:
                    eligible = finite(row.get("qb_projection_eligible"), 0.0)
                    role_score = finite(row.get("qb_role_score"), 0.0)
                    qb_candidates.append((eligible, role_score, pkey, pass_yards))

                identity = str(row.get("player_identity_key", "") or "").strip()
                join_key = f"id:{identity}" if identity else f"name:{pkey}"
                within_modeled = (
                    float(probs[j] / modeled_receiver_mass)
                    if position in PASS_CATCHER_POSITIONS and modeled_receiver_mass > 0
                    else 0.0
                )
                player_rows.append(
                    {
                        "season": int(args.season),
                        "week": int(week),
                        "event_id": str(game),
                        "team": str(team),
                        "player": row.get("player", ""),
                        "player_clean_key": pkey,
                        "player_identity_key": identity,
                        "join_key": join_key,
                        "position": position,
                        "position_group": pos_group,
                        "target_probability_all_pass_counts": float(probs[j]),
                        "target_share_within_modeled_receivers": within_modeled,
                        "residual_pass_probability": float(residual_prob),
                        "modeled_receiver_probability_mass": float(modeled_receiver_mass),
                        "det_expected_team_pass_count": expected_team_pass_count,
                        "det_expected_targets": expected_team_pass_count * float(probs[j]),
                        "proj_receptions": float(np.mean(rec)) if rec is not None else np.nan,
                        "proj_rec_yards": float(np.mean(rec_yards)) if rec_yards is not None else np.nan,
                        "proj_rush_yards": float(np.mean(rush_yards)) if rush_yards is not None else np.nan,
                        "proj_rush_rec_yards": float(np.mean(rush_rec)) if rush_rec is not None else np.nan,
                        "raw_sim_pass_yards": float(np.mean(pass_yards)) if pass_yards is not None else np.nan,
                        "qb_projection_eligible": finite(row.get("qb_projection_eligible"), 0.0),
                        "qb_role_score": finite(row.get("qb_role_score"), 0.0),
                    }
                )

            if receiver_arrays and qb_candidates:
                receiver_sum = np.sum(np.vstack(receiver_arrays), axis=0)
                qb_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                _, _, primary_key, primary_pass = qb_candidates[0]
                gap = np.asarray(primary_pass, dtype=float) - np.asarray(receiver_sum, dtype=float)
                stats = qstats(gap)
                conservation_rows.append(
                    {
                        "season": int(args.season),
                        "week": int(week),
                        "event_id": str(game),
                        "team": str(team),
                        "primary_qb_key": primary_key,
                        "raw_sim_primary_qb_pass_yards_mean": float(np.mean(primary_pass)),
                        "raw_sim_modeled_receiver_yards_sum_mean": float(np.mean(receiver_sum)),
                        **stats,
                    }
                )

        print(
            f"[ecosystem-trace] season={args.season} week={week:02d} "
            f"players={len(players)} cumulative={len(player_rows)}"
        )

    projection = pd.DataFrame(player_rows)
    conservation = pd.DataFrame(conservation_rows)
    actual = actual_usage(logs, args.season, set(int(w) for w in weeks))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    projection.to_csv(args.out_dir / "receiving_ecosystem_projection_trace.csv", index=False)
    conservation.to_csv(args.out_dir / "raw_sim_pass_receiving_conservation.csv", index=False)
    actual.to_csv(args.out_dir / "receiving_ecosystem_actual_usage.csv", index=False)
    print(
        f"[ecosystem-trace] wrote projection={len(projection)} actual={len(actual)} "
        f"raw_conservation={len(conservation)} -> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
