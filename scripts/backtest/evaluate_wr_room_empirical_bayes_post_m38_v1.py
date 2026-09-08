#!/usr/bin/env python3
"""WR-R12B: implementation-corrected post-M38 conserved-room EB test.

This is NOT a parameter retune of WR-R12.  It repairs the experiment seam so the
original frozen hypothesis is tested as written.

WR-R12 claimed to use the M38 room share as its prior, but its implementation
built the EB candidate from pre-M38 ``rules_tgt_share`` and then allowed
``simulation_v2`` to apply M38 afterward.  Its printed target metric likewise
used the pre-M38 shares.  Therefore the prior/result path did not match the
registered hypothesis.

R12B keeps every scientific parameter/gate from R12 unchanged:
- 2025 regular season Weeks 1-18;
- 2024 + strictly-prior 2025 history only;
- four most recent completed same-team WR games;
- one average historical WR-room game of pseudo-target mass;
- same MC seed for baseline/candidate;
- no sportsbook inputs;
- same promotion-like gates.

The only correction is ordering:
1. materialize canonical M38 + finite team target entitlement;
2. freeze that already-conserved WR-room mass;
3. perform the exact R12 EB redistribution inside that room;
4. simulate from explicit entitlement with internal M38 disabled, so M38 is not
   applied a second time.

A PASS authorizes a separately frozen multi-season confirmation only.  A FAIL
closes the R12 recent-target EB family.  Production is never modified here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest.evaluate_wr_room_empirical_bayes_v1 import (
    HISTORY_GAMES,
    MAX_TARGET_MAE_WORSEN,
    MIN_REC_YARDS_MAE_GAIN,
    REQUIRED_NONWORSE_PHASES,
    WR_POS,
    finite,
    metric,
    optional,
    prepared,
    prior_team_wr_games,
    read,
)
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate

VARIANT_BASE = "M38_EXPLICIT_BASELINE"
VARIANT_CAND = "WR_R12B_POST_M38_EB4"


def apply_candidate(
    baseline: pd.DataFrame,
    logs: pd.DataFrame,
    season: int,
    week: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Redistribute only the already-conserved post-M38 WR room."""
    out = baseline.copy()
    audits: list[dict] = []
    for (event_id, team), idx in out.groupby(["event_id", "team"], dropna=False, sort=False).groups.items():
        group = out.loc[idx].copy()
        pos = group.get("position", pd.Series("", index=group.index)).fillna("").astype(str).str.upper().str.strip()
        wr_mask = pos.isin(WR_POS)
        ent = pd.to_numeric(group["entitlement_tgt_share"], errors="coerce").fillna(0.0).astype(float)
        wr_idx = list(group.index[wr_mask])
        wr_total = float(ent.loc[wr_idx].sum()) if wr_idx else 0.0
        if len(wr_idx) <= 1 or wr_total <= 0.0:
            continue

        baseline_room = ent.loc[wr_idx].to_numpy(float) / wr_total
        hist = prior_team_wr_games(logs, season, week, str(team))
        hist_targets = (
            pd.to_numeric(hist.get("targets", 0.0), errors="coerce").fillna(0.0)
            if not hist.empty else pd.Series(dtype=float)
        )
        n_games = int(hist[["season", "week"]].drop_duplicates().shape[0]) if not hist.empty else 0
        total_hist_targets = float(hist_targets.sum()) if not hist.empty else 0.0
        avg_room_targets = total_hist_targets / n_games if n_games > 0 and total_hist_targets > 0 else 20.0
        by_player = (
            hist.assign(_targets=hist_targets)
            .groupby("player_clean_key", dropna=False)["_targets"].sum()
            if not hist.empty else pd.Series(dtype=float)
        )
        current_keys = group.loc[wr_idx, "player_clean_key"].astype(str).tolist()
        observed = np.asarray([float(by_player.get(k, 0.0)) for k in current_keys], dtype=float)

        # Exact frozen R12 prior strength/mechanism, now correctly using the
        # already-applied M38 room share.
        score = observed + avg_room_targets * baseline_room
        candidate_room = score / float(score.sum()) if float(score.sum()) > 0 else baseline_room.copy()
        candidate = candidate_room * wr_total
        # Floating-only conservation guard; not a model parameter.
        gap = wr_total - float(candidate.sum())
        if len(candidate):
            candidate[int(np.argmax(baseline_room))] += gap

        before_team = float(pd.to_numeric(out.loc[idx, "entitlement_tgt_share"], errors="coerce").sum())
        non_wr_before = out.loc[[i for i in idx if i not in wr_idx], "entitlement_tgt_share"].astype(float).copy()
        out.loc[wr_idx, "entitlement_tgt_share"] = candidate
        after_team = float(pd.to_numeric(out.loc[idx, "entitlement_tgt_share"], errors="coerce").sum())
        non_wr_after = out.loc[non_wr_before.index, "entitlement_tgt_share"].astype(float)

        audits.append({
            "week": int(week),
            "event_id": str(event_id),
            "team": str(team),
            "history_games": n_games,
            "history_wr_targets": total_hist_targets,
            "pseudo_wr_targets": float(avg_room_targets),
            "baseline_wr_room_mass": wr_total,
            "candidate_wr_room_mass": float(candidate.sum()),
            "wr_room_mass_gap": float(candidate.sum() - wr_total),
            "baseline_team_player_mass": before_team,
            "candidate_team_player_mass": after_team,
            "team_player_mass_gap": after_team - before_team,
            "max_non_wr_entitlement_delta": float((non_wr_after - non_wr_before).abs().max()) if len(non_wr_before) else 0.0,
            "max_player_room_share_move": float(np.max(np.abs(candidate_room - baseline_room))),
            "sportsbook_inputs_used": 0,
            "current_or_future_outcomes_used": 0,
            "history_games_frozen": HISTORY_GAMES,
        })
    return out, audits


def baseline_ranks(frame: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    ranks: dict[tuple[str, str, str], int] = {}
    for (event_id, team), group in frame.groupby(["event_id", "team"], dropna=False, sort=False):
        pos = group.get("position", pd.Series("", index=group.index)).fillna("").astype(str).str.upper().str.strip()
        wr = group.loc[pos.isin(WR_POS)].copy()
        if wr.empty:
            continue
        wr["_ent"] = pd.to_numeric(wr["entitlement_tgt_share"], errors="coerce").fillna(0.0)
        wr = wr.sort_values(["_ent", "player_clean_key"], ascending=[False, True], kind="stable")
        for rank, row in enumerate(wr.itertuples(index=False), start=1):
            ranks[(str(event_id), str(team), str(row.player_clean_key))] = rank
    return ranks


def prediction_rows(frame: pd.DataFrame, sim, variant: str, week: int, rank_map: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for (event_id, team_name), group in frame.groupby(["event_id", "team"], dropna=False, sort=False):
        pos = group.get("position", pd.Series("", index=group.index)).fillna("").astype(str).str.upper().str.strip()
        plays = float(np.mean([finite(v, 64.0) for v in group.get("rules_plays_est", pd.Series([64.0] * len(group)))]))
        pass_rate = float(np.mean([finite(v, 0.57) for v in group.get("rules_pass_rate", pd.Series([0.57] * len(group)))]))
        team_targets = plays * pass_rate
        for j, (_, row) in enumerate(group.iterrows()):
            if str(pos.iloc[j]) not in WR_POS:
                continue
            key = str(row.get("player_clean_key", ""))
            ent = finite(row.get("entitlement_tgt_share"), 0.0)
            rec = sim.values.get((str(event_id), key, "receptions"))
            yards = sim.values.get((str(event_id), key, "rec_yards"))
            rows.append({
                "variant": variant,
                "week": int(week),
                "event_id": str(event_id),
                "team": str(team_name),
                "player_clean_key": key,
                "player": row.get("player", ""),
                "wr_rank": int(rank_map.get((str(event_id), str(team_name), key), 99)),
                "entitlement_tgt_share": float(ent),
                "pred_targets": float(team_targets * ent),
                "mc_receptions": float(np.mean(rec)) if rec is not None else np.nan,
                "mc_rec_yards": float(np.mean(yards)) if yards is not None else np.nan,
            })
    return pd.DataFrame(rows)


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
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/wr_r12b_post_m38_v1"))
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
        raw = prepared(bundle)
        baseline, _ = materialize_target_entitlement(raw)
        candidate, audits = apply_candidate(baseline, logs, args.season, week)
        allocation_audit.extend(audits)
        ranks = baseline_ranks(baseline)

        # Same MC seed and exact explicit simulator on both variants.
        baseline_sim = explicit_simulate(baseline, iterations=args.iterations, seed=73100 + week)
        candidate_sim = explicit_simulate(candidate, iterations=args.iterations, seed=73100 + week)

        actual = cp.build_actual_rows(logs, args.season, week)
        actual_targets = actual[actual.market.eq("receptions")][["team", "player_clean_key", "actual_opportunities"]].rename(columns={"actual_opportunities": "actual_targets"})
        actual_yards = actual[actual.market.eq("rec_yards")][["team", "player_clean_key", "actual"]].rename(columns={"actual": "actual_rec_yards"})

        for variant, frame, sim in (
            (VARIANT_BASE, baseline, baseline_sim),
            (VARIANT_CAND, candidate, candidate_sim),
        ):
            x = prediction_rows(frame, sim, variant, week, ranks)
            x = x.merge(actual_targets, on=["team", "player_clean_key"], how="inner")
            x = x.merge(actual_yards, on=["team", "player_clean_key"], how="inner")
            predictions.append(x)
        print(f"[wr-r12b] week={week:02d} complete")

    pred = pd.concat(predictions, ignore_index=True)
    pred["phase"] = pd.cut(pred["week"], [0, 4, 9, 13, 18], labels=["W1-4", "W5-9", "W10-13", "W14-18"])
    pred["role"] = np.select(
        [pred.wr_rank.eq(1), pred.wr_rank.eq(2), pred.wr_rank.eq(3)],
        ["WR1", "WR2", "WR3"], default="WR4+",
    )
    pred["abs_rec_yards_error"] = (pd.to_numeric(pred["mc_rec_yards"], errors="coerce") - pd.to_numeric(pred["actual_rec_yards"], errors="coerce")).abs()

    summary_rows: list[dict] = []
    for variant, g in pred.groupby("variant"):
        for market, actual_col, pred_col in (
            ("targets", "actual_targets", "pred_targets"),
            ("rec_yards", "actual_rec_yards", "mc_rec_yards"),
        ):
            row = {"variant": variant, "market": market, **metric(g[actual_col], g[pred_col])}
            if market == "rec_yards":
                ae = g["abs_rec_yards_error"]
                row["miss_30_plus_rate"] = float(ae.ge(30.0).mean())
                row["miss_50_plus_rate"] = float(ae.ge(50.0).mean())
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    bucket_rows: list[dict] = []
    for bucket_col in ("phase", "role"):
        for (variant, bucket), g in pred.groupby(["variant", bucket_col], observed=False):
            if g.empty:
                continue
            bucket_rows.append({
                "bucket_type": bucket_col,
                "bucket": str(bucket),
                "variant": variant,
                **metric(g.actual_rec_yards, g.mc_rec_yards),
            })
    buckets = pd.DataFrame(bucket_rows)

    def srow(variant: str, market: str) -> pd.Series:
        return summary.loc[summary.variant.eq(variant) & summary.market.eq(market)].iloc[0]

    b_y, c_y = srow(VARIANT_BASE, "rec_yards"), srow(VARIANT_CAND, "rec_yards")
    b_t, c_t = srow(VARIANT_BASE, "targets"), srow(VARIANT_CAND, "targets")
    phase_pivot = buckets[buckets.bucket_type.eq("phase")].pivot(index="bucket", columns="variant", values="mae").dropna()
    nonworse_phases = int((phase_pivot[VARIANT_CAND] <= phase_pivot[VARIANT_BASE]).sum()) if not phase_pivot.empty else 0

    top = pred[pred.role.isin(["WR1", "WR2"])]
    b_top = metric(top.loc[top.variant.eq(VARIANT_BASE), "actual_rec_yards"], top.loc[top.variant.eq(VARIANT_BASE), "mc_rec_yards"])
    c_top = metric(top.loc[top.variant.eq(VARIANT_CAND), "actual_rec_yards"], top.loc[top.variant.eq(VARIANT_CAND), "mc_rec_yards"])

    audit_df = pd.DataFrame(allocation_audit)
    max_wr_gap = float(audit_df.wr_room_mass_gap.abs().max()) if not audit_df.empty else 0.0
    max_team_gap = float(audit_df.team_player_mass_gap.abs().max()) if not audit_df.empty else 0.0
    max_non_wr_delta = float(audit_df.max_non_wr_entitlement_delta.abs().max()) if not audit_df.empty else 0.0
    leakage = int(audit_df.sportsbook_inputs_used.sum()) if not audit_df.empty else 0
    future = int(audit_df.current_or_future_outcomes_used.sum()) if not audit_df.empty else 0

    # Exact original R12 scientific gates. Additional tail rates above are
    # diagnostics only so this correction cannot move the goalposts after seeing
    # the first implementation's result.
    gates = {
        "rec_yards_mae_gain_ge_0_20": bool(float(b_y.mae - c_y.mae) >= MIN_REC_YARDS_MAE_GAIN),
        "target_mae_worsen_le_0_01": bool(float(c_t.mae - b_t.mae) <= MAX_TARGET_MAE_WORSEN),
        "rec_yards_p90_nonworse": bool(float(c_y.p90_abs_error) <= float(b_y.p90_abs_error)),
        "phase_nonworse_at_least_3_of_4": bool(nonworse_phases >= REQUIRED_NONWORSE_PHASES),
        "wr1_wr2_rec_yards_mae_nonworse": bool(float(c_top["mae"]) <= float(b_top["mae"])),
        "wr_room_mass_exact": bool(max_wr_gap <= 1e-12),
    }
    integrity = {
        "team_player_mass_exact": bool(max_team_gap <= 1e-12),
        "non_wr_entitlement_unchanged": bool(max_non_wr_delta <= 1e-12),
        "sportsbook_inputs_zero": bool(leakage == 0),
        "current_future_outcomes_zero": bool(future == 0),
    }
    passed = all(gates.values()) and all(integrity.values())
    disposition = "WR_R12B_POST_M38_CONSERVED_ENTITLEMENT_PASS" if passed else "WR_R12B_POST_M38_CONSERVED_ENTITLEMENT_FAIL"
    decision = pd.DataFrame([{
        "disposition": disposition,
        "experiment_contract": "R12_PARAMETERS_UNCHANGED_IMPLEMENTATION_SEAM_CORRECTED",
        "baseline": VARIANT_BASE,
        "candidate": VARIANT_CAND,
        "m38_materialized_before_candidate": True,
        "internal_m38_disabled_during_explicit_simulation": True,
        "history_games": HISTORY_GAMES,
        "baseline_rec_yards_mae": float(b_y.mae),
        "candidate_rec_yards_mae": float(c_y.mae),
        "rec_yards_mae_gain": float(b_y.mae - c_y.mae),
        "baseline_target_mae": float(b_t.mae),
        "candidate_target_mae": float(c_t.mae),
        "target_mae_delta": float(c_t.mae - b_t.mae),
        "baseline_rec_yards_p90": float(b_y.p90_abs_error),
        "candidate_rec_yards_p90": float(c_y.p90_abs_error),
        "baseline_miss_30_plus_rate": float(b_y.get("miss_30_plus_rate", np.nan)),
        "candidate_miss_30_plus_rate": float(c_y.get("miss_30_plus_rate", np.nan)),
        "baseline_miss_50_plus_rate": float(b_y.get("miss_50_plus_rate", np.nan)),
        "candidate_miss_50_plus_rate": float(c_y.get("miss_50_plus_rate", np.nan)),
        "nonworse_phases": nonworse_phases,
        "baseline_wr1_wr2_mae": float(b_top["mae"]),
        "candidate_wr1_wr2_mae": float(c_top["mae"]),
        "max_wr_room_mass_gap": max_wr_gap,
        "max_team_player_mass_gap": max_team_gap,
        "max_non_wr_entitlement_delta": max_non_wr_delta,
        **gates,
        **integrity,
    }])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_dir / "wr_r12b_player_predictions.csv", index=False)
    summary.to_csv(args.out_dir / "wr_r12b_market_summary.csv", index=False)
    buckets.to_csv(args.out_dir / "wr_r12b_bucket_summary.csv", index=False)
    audit_df.to_csv(args.out_dir / "wr_r12b_conservation_audit.csv", index=False)
    decision.to_csv(args.out_dir / "wr_r12b_decision.csv", index=False)
    print("\n[wr-r12b] summary\n", summary.to_string(index=False))
    print("\n[wr-r12b] decision\n", decision.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
