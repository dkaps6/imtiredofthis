#!/usr/bin/env python3
"""WR-R12: conserved within-room empirical-Bayes entitlement around M38.

Scientific question
-------------------
Can strictly-prior same-team WR opportunity continuity improve *individual* WR
allocation without changing the finite team/WR opportunity pool or discarding
M38's durable hierarchy prior?

This is deliberately NOT an additive target correction.  For each current WR,
we take the target counts from the team's four most recent completed games and
combine them with exactly one average prior WR-room game's worth of pseudo-target
mass distributed according to the canonical M38 room shares.  The resulting
scores are normalized inside the current WR room, so WR-room mass is conserved
exactly.  New/current players with no same-team history retain M38 prior mass;
players no longer on the current roster cannot consume target mass.

Frozen before results:
- evaluation season: 2025 regular season Weeks 1-18;
- history: 2024 plus strictly-prior 2025 games only;
- history window: four most recent team games;
- prior strength: exactly one average historical WR-room game (not tuned);
- baseline: canonical prepared rules_tgt_share (M38 already active);
- no sportsbook inputs;
- same Monte-Carlo seed for baseline/candidate per week;
- promotion-like research gates (all required): receiving-yard MAE improves by
  >=0.20 yards, target MAE worsens by <=0.01 targets, receiving-yard p90 absolute
  error does not worsen, >=3/4 season phases do not worsen receiving-yard MAE,
  and combined baseline WR1/WR2 receiving-yard MAE does not worsen.

Passing these gates authorizes a separately frozen integration confirmation only;
it does not directly change production.
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

WR_POS = {"WR", "LWR", "RWR", "SWR"}
HISTORY_GAMES = 4
MIN_REC_YARDS_MAE_GAIN = 0.20
MAX_TARGET_MAE_WORSEN = 0.01
REQUIRED_NONWORSE_PHASES = 3


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing required artifact: {path}")
    return pd.read_csv(path)


def optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def finite(value, default=np.nan) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def prepared(bundle) -> pd.DataFrame:
    metrics = cp.build_market_frame(bundle)
    metrics = apply_bayesian_to_metrics(metrics, build_bayesian_baseline(bundle.player_consensus))
    with patch.object(simulation_rules, "load_model_contexts", return_value=(bundle.teams, bundle.players)):
        metrics = simulation_rules.apply_rules_to_metrics(metrics)
    metrics["player_clean_key"] = metrics["player_clean_key"].fillna("").astype(str)
    keys = ["event_id", "team", "player_clean_key"]
    return metrics.sort_values(keys).drop_duplicates(keys, keep="last").copy()


def allocator_probs(shares) -> np.ndarray:
    clipped = np.clip(np.nan_to_num(np.asarray(shares, float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 0.95)
    if float(clipped.sum()) > 0.95:
        clipped *= 0.95 / float(clipped.sum())
    residual = max(0.0, 1.0 - float(clipped.sum()))
    probs = np.append(clipped, residual)
    probs /= probs.sum()
    return probs[:-1]


def prior_team_wr_games(logs: pd.DataFrame, season: int, week: int, team: str) -> pd.DataFrame:
    x = logs.copy()
    season_col = pd.to_numeric(x["season"], errors="coerce")
    week_col = pd.to_numeric(x["week"], errors="coerce")
    prior = (season_col < season) | ((season_col == season) & (week_col < week))
    pos = x.get("position", pd.Series("", index=x.index)).fillna("").astype(str).str.upper().str.strip()
    x = x.loc[prior & x["team"].astype(str).eq(str(team)) & pos.isin(WR_POS)].copy()
    if x.empty:
        return x
    games = (
        x[["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"], ascending=[False, False])
        .head(HISTORY_GAMES)
    )
    return x.merge(games, on=["season", "week"], how="inner")


def apply_candidate(base: pd.DataFrame, logs: pd.DataFrame, season: int, week: int) -> tuple[pd.DataFrame, list[dict]]:
    pieces = []
    audits: list[dict] = []
    for (event_id, team), group in base.groupby(["event_id", "team"], dropna=False, sort=False):
        g = group.copy()
        pos = g.get("position", pd.Series("", index=g.index)).fillna("").astype(str).str.upper().str.strip()
        wr_mask = pos.isin(WR_POS)
        raw = pd.to_numeric(g.get("rules_tgt_share"), errors="coerce").fillna(0.0).clip(0.0, 0.95)
        wr_total = float(raw.loc[wr_mask].sum())
        if wr_mask.sum() <= 1 or wr_total <= 0:
            pieces.append(g)
            continue

        wr_idx = list(g.index[wr_mask])
        baseline_room = raw.loc[wr_idx].to_numpy(float) / wr_total
        hist = prior_team_wr_games(logs, season, week, str(team))
        hist_targets = pd.to_numeric(hist.get("targets", 0.0), errors="coerce").fillna(0.0) if not hist.empty else pd.Series(dtype=float)
        n_games = int(hist[["season", "week"]].drop_duplicates().shape[0]) if not hist.empty else 0
        total_hist_targets = float(hist_targets.sum()) if not hist.empty else 0.0
        avg_room_targets = total_hist_targets / n_games if n_games > 0 and total_hist_targets > 0 else 20.0
        by_player = (
            hist.assign(_targets=hist_targets)
            .groupby("player_clean_key", dropna=False)["_targets"].sum()
            if not hist.empty
            else pd.Series(dtype=float)
        )
        current_keys = g.loc[wr_idx, "player_clean_key"].astype(str).tolist()
        observed = np.asarray([float(by_player.get(k, 0.0)) for k in current_keys], dtype=float)
        score = observed + avg_room_targets * baseline_room
        candidate_room = score / float(score.sum()) if float(score.sum()) > 0 else baseline_room.copy()
        candidate = candidate_room * wr_total
        g.loc[wr_idx, "rules_tgt_share"] = candidate

        audits.append({
            "week": int(week),
            "event_id": str(event_id),
            "team": str(team),
            "history_games": n_games,
            "history_wr_targets": total_hist_targets,
            "pseudo_wr_targets": float(avg_room_targets),
            "baseline_wr_room_mass": wr_total,
            "candidate_wr_room_mass": float(candidate.sum()),
            "mass_gap": float(candidate.sum() - wr_total),
            "max_player_room_share_move": float(np.max(np.abs(candidate_room - baseline_room))),
        })
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True), audits


def metric(actual, pred) -> dict:
    z = pd.DataFrame({"actual": pd.to_numeric(actual, errors="coerce"), "pred": pd.to_numeric(pred, errors="coerce")}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan, "median_abs_error": np.nan, "p75_abs_error": np.nan, "p90_abs_error": np.nan}
    err = z["pred"] - z["actual"]
    ae = err.abs()
    return {
        "n": int(len(z)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()),
        "correlation": float(z["pred"].corr(z["actual"])) if len(z) > 1 and z["actual"].nunique() > 1 and z["pred"].nunique() > 1 else np.nan,
        "median_abs_error": float(ae.median()),
        "p75_abs_error": float(ae.quantile(0.75)),
        "p90_abs_error": float(ae.quantile(0.90)),
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
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/wr_room_empirical_bayes_v1"))
    args = p.parse_args()

    logs = read(args.player_logs)
    team = read(args.team_weekly)
    schedule = read(args.schedule)
    injuries = optional(args.injuries)
    weather = optional(args.weather)
    predictions: list[pd.DataFrame] = []
    allocation_audit: list[dict] = []

    for week in _parse_weeks(args.weeks):
        universe = read(args.universe_dir / f"{args.season}_week_{week:02d}.csv")
        bundle = build_historical_context_bundle(
            player_logs=logs, team_weekly=team, pregame_universe=universe, schedule=schedule,
            season=args.season, week=week, prior_season=args.prior_season,
            injuries=_exact_week(injuries, args.season, week), weather=_exact_week(weather, args.season, week),
        )
        baseline = prepared(bundle)
        candidate, audit = apply_candidate(baseline, logs, args.season, week)
        allocation_audit.extend(audit)
        actual = cp.build_actual_rows(logs, args.season, week)
        actual_targets = actual[actual.market.eq("receptions")][["team", "player_clean_key", "actual_opportunities"]].rename(columns={"actual_opportunities": "actual_targets"})
        actual_yards = actual[actual.market.eq("rec_yards")][["team", "player_clean_key", "actual"]].rename(columns={"actual": "actual_rec_yards"})

        for variant, frame in (("M38_BASELINE", baseline), ("WR_R12_EB4", candidate)):
            sim = simulation_v2.simulate(frame, iterations=args.iterations, seed=73100 + week)
            rows = []
            for (event_id, team_name), group in frame.groupby(["event_id", "team"], dropna=False, sort=False):
                pos = group.get("position", pd.Series("", index=group.index)).fillna("").astype(str).str.upper().str.strip()
                raw = pd.to_numeric(group.get("rules_tgt_share"), errors="coerce").fillna(0.0).to_numpy(float)
                probs = allocator_probs(raw)
                plays = float(np.mean([finite(v, 64.0) for v in group.get("rules_plays_est", pd.Series([64.0] * len(group)))]))
                pass_rate = float(np.mean([finite(v, 0.57) for v in group.get("rules_pass_rate", pd.Series([0.57] * len(group)))]))
                team_targets = plays * pass_rate
                wr_positions = np.flatnonzero(pos.isin(WR_POS).to_numpy())
                wr_raw = raw[wr_positions]
                rank_order = wr_positions[np.argsort(-wr_raw, kind="stable")] if len(wr_positions) else np.asarray([], dtype=int)
                rank_lookup = {int(idx): rank + 1 for rank, idx in enumerate(rank_order)}
                for j, (_, row) in enumerate(group.iterrows()):
                    if str(pos.iloc[j]) not in WR_POS:
                        continue
                    key = str(row.get("player_clean_key", ""))
                    rec = sim.values.get((str(event_id), key, "receptions"))
                    yards = sim.values.get((str(event_id), key, "rec_yards"))
                    rows.append({
                        "variant": variant, "week": int(week), "event_id": str(event_id), "team": str(team_name),
                        "player_clean_key": key, "player": row.get("player", ""), "wr_rank": int(rank_lookup.get(j, 99)),
                        "pred_targets": float(team_targets * probs[j]),
                        "mc_receptions": float(np.mean(rec)) if rec is not None else np.nan,
                        "mc_rec_yards": float(np.mean(yards)) if yards is not None else np.nan,
                    })
            x = pd.DataFrame(rows).merge(actual_targets, on=["team", "player_clean_key"], how="inner").merge(actual_yards, on=["team", "player_clean_key"], how="inner")
            predictions.append(x)
        print(f"[wr-r12] week={week:02d} complete")

    pred = pd.concat(predictions, ignore_index=True)
    pred["phase"] = pd.cut(pred["week"], [0, 4, 9, 13, 18], labels=["W1-4", "W5-9", "W10-13", "W14-18"])
    pred["role"] = np.select([pred.wr_rank.eq(1), pred.wr_rank.eq(2), pred.wr_rank.eq(3)], ["WR1", "WR2", "WR3"], default="WR4+")

    summary_rows = []
    for variant, g in pred.groupby("variant"):
        for market, actual_col, pred_col in (("targets", "actual_targets", "pred_targets"), ("rec_yards", "actual_rec_yards", "mc_rec_yards")):
            summary_rows.append({"variant": variant, "market": market, **metric(g[actual_col], g[pred_col])})
    summary = pd.DataFrame(summary_rows)

    bucket_rows = []
    for bucket_col in ("phase", "role"):
        for (variant, bucket), g in pred.groupby(["variant", bucket_col], observed=False):
            if g.empty:
                continue
            bucket_rows.append({"bucket_type": bucket_col, "bucket": str(bucket), "variant": variant, **metric(g.actual_rec_yards, g.mc_rec_yards)})
    buckets = pd.DataFrame(bucket_rows)

    def row(variant: str, market: str) -> pd.Series:
        return summary.loc[summary.variant.eq(variant) & summary.market.eq(market)].iloc[0]

    b_y = row("M38_BASELINE", "rec_yards"); c_y = row("WR_R12_EB4", "rec_yards")
    b_t = row("M38_BASELINE", "targets"); c_t = row("WR_R12_EB4", "targets")
    phase_pivot = buckets[buckets.bucket_type.eq("phase")].pivot(index="bucket", columns="variant", values="mae")
    comparable = phase_pivot.dropna()
    nonworse_phases = int((comparable["WR_R12_EB4"] <= comparable["M38_BASELINE"]).sum()) if not comparable.empty else 0
    top = pred[pred.role.isin(["WR1", "WR2"])]
    b_top = metric(top.loc[top.variant.eq("M38_BASELINE"), "actual_rec_yards"], top.loc[top.variant.eq("M38_BASELINE"), "mc_rec_yards"])
    c_top = metric(top.loc[top.variant.eq("WR_R12_EB4"), "actual_rec_yards"], top.loc[top.variant.eq("WR_R12_EB4"), "mc_rec_yards"])
    audit_df = pd.DataFrame(allocation_audit)
    max_mass_gap = float(audit_df.mass_gap.abs().max()) if not audit_df.empty else 0.0

    gates = {
        "rec_yards_mae_gain_ge_0_20": bool(float(b_y.mae - c_y.mae) >= MIN_REC_YARDS_MAE_GAIN),
        "target_mae_worsen_le_0_01": bool(float(c_t.mae - b_t.mae) <= MAX_TARGET_MAE_WORSEN),
        "rec_yards_p90_nonworse": bool(float(c_y.p90_abs_error) <= float(b_y.p90_abs_error)),
        "phase_nonworse_at_least_3_of_4": bool(nonworse_phases >= REQUIRED_NONWORSE_PHASES),
        "wr1_wr2_rec_yards_mae_nonworse": bool(float(c_top["mae"]) <= float(b_top["mae"])),
        "wr_room_mass_exact": bool(max_mass_gap <= 1e-12),
    }
    disposition = "WR_R12_CONSERVED_ENTITLEMENT_PASS" if all(gates.values()) else "WR_R12_CONSERVED_ENTITLEMENT_FAIL"
    decision = pd.DataFrame([{
        "disposition": disposition,
        "baseline_rec_yards_mae": float(b_y.mae), "candidate_rec_yards_mae": float(c_y.mae), "rec_yards_mae_gain": float(b_y.mae - c_y.mae),
        "baseline_target_mae": float(b_t.mae), "candidate_target_mae": float(c_t.mae), "target_mae_delta": float(c_t.mae - b_t.mae),
        "baseline_rec_yards_p90": float(b_y.p90_abs_error), "candidate_rec_yards_p90": float(c_y.p90_abs_error),
        "nonworse_phases": nonworse_phases, "baseline_wr1_wr2_mae": float(b_top["mae"]), "candidate_wr1_wr2_mae": float(c_top["mae"]),
        "max_wr_room_mass_gap": max_mass_gap, **gates,
    }])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_dir / "wr_r12_player_predictions.csv", index=False)
    summary.to_csv(args.out_dir / "wr_r12_market_summary.csv", index=False)
    buckets.to_csv(args.out_dir / "wr_r12_bucket_summary.csv", index=False)
    audit_df.to_csv(args.out_dir / "wr_r12_conservation_audit.csv", index=False)
    decision.to_csv(args.out_dir / "wr_r12_decision.csv", index=False)
    print("\n[wr-r12] summary\n", summary.to_string(index=False))
    print("\n[wr-r12] decision\n", decision.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
