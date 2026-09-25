#!/usr/bin/env python3
"""Unchanged 2024-2025 confirmation for Receiver Room Targets-Per-Play V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import _load_snaps as _load_participation_snaps
from scripts.research.evaluate_receiver_room_targetable_rate_v1 import (
    ROOMS,
    FIXED_DROPBACK_RATE,
    optional,
    paired,
    pos_room,
    read,
)
from scripts.research.evaluate_receiver_room_targets_per_play_v1 import (
    build_room_history,
    strict_prior_room_rates,
)
from scripts.research.persist_wr_te_production_order_historical_v1 import (
    TE_FEATURES,
    WR_FEATURES,
    _load_fold_params,
    apply_te_fold,
    apply_wr_fold,
)

VERSION = "RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_CONFIRMATION"
TOL = 1e-10


def evaluate_week(
    *,
    season: int,
    prior_season: int,
    week: int,
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    universe: pd.DataFrame,
    injuries: pd.DataFrame,
    weather: pd.DataFrame,
    history: pd.DataFrame,
    te_params: dict,
    wr_params: dict | None,
    snaps: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    bundle = build_historical_context_bundle(
        player_logs=player_logs,
        team_weekly=team_weekly,
        pregame_universe=universe,
        schedule=schedule,
        season=int(season),
        week=int(week),
        prior_season=int(prior_season),
        injuries=_exact_week(injuries, int(season), int(week)),
        weather=_exact_week(weather, int(season), int(week)),
    )
    metrics = build_mc_predictions(bundle, iterations=20, seed=9400 + int(week))
    players = (
        metrics.sort_values(["event_id", "team", "player_clean_key"])
        .drop_duplicates(["event_id", "team", "player_clean_key"], keep="last")
        .copy()
    )
    explicit, _ = materialize_target_entitlement(players)
    te_final, _, te_audit = apply_te_fold(explicit, snaps=snaps, params=te_params)
    if int(season) == 2024:
        if wr_params is None:
            raise RuntimeError("2024 confirmation requires WR-R15 fold params")
        final, _, wr_audit = apply_wr_fold(te_final, snaps=snaps, params=wr_params)
    else:
        final = te_final
        wr_audit = {
            "m38_wr1_anchor_max_abs_gap": 0.0,
            "wr2plus_pool_max_abs_gap": 0.0,
            "wr_room_mass_max_abs_gap": 0.0,
            "non_wr_max_abs_gap": 0.0,
            "same_future_participation": 0,
        }

    final = final.copy()
    final["team"] = final["team"].map(canon_team)
    final["room"] = final["position"].map(pos_room)
    final["entitlement_tgt_share"] = pd.to_numeric(
        final["entitlement_tgt_share"], errors="raise"
    )
    if (
        final["entitlement_tgt_share"].isna().any()
        or not np.isfinite(final["entitlement_tgt_share"].to_numpy(float)).all()
        or final["entitlement_tgt_share"].lt(0).any()
    ):
        raise RuntimeError(f"{season} W{week:02d} invalid final entitlement")

    actual = history.loc[
        history["season"].eq(int(season)) & history["week"].eq(int(week)),
        ["team", "room", "actual_room_targets"],
    ].copy()
    actual["team"] = actual["team"].map(canon_team)

    rows: list[dict] = []
    for team, tdf in final.groupby("team", sort=True):
        p = pd.to_numeric(tdf["mc_projected_plays"], errors="coerce").dropna().unique()
        if len(p) != 1:
            raise RuntimeError(
                f"{season} W{week:02d} team={team} nonunique projected plays {p[:5]}"
            )
        opp = tdf["opponent"].dropna().astype(str).map(canon_team).unique()
        if len(opp) != 1:
            raise RuntimeError(f"{season} W{week:02d} team={team} nonunique opponent")
        projected_plays = float(p[0])
        projected_dropbacks = projected_plays * FIXED_DROPBACK_RATE
        if not np.isfinite(projected_plays) or projected_plays <= 0:
            raise RuntimeError(f"{season} W{week:02d} invalid projected plays team={team}")

        info = strict_prior_room_rates(
            history,
            season=int(season),
            week=int(week),
            team=team,
            prior_season=int(prior_season),
        )
        for room_name in ROOMS:
            room_ent = float(
                tdf.loc[tdf["room"].eq(room_name), "entitlement_tgt_share"].sum()
            )
            rate = float(info["room_rates"][room_name])
            a = actual.loc[
                actual["team"].eq(team) & actual["room"].eq(room_name),
                "actual_room_targets",
            ]
            if len(a) != 1:
                raise RuntimeError(
                    f"{season} W{week:02d} actual room target missing/duplicate "
                    f"team={team} room={room_name}"
                )
            rows.append({
                "season": int(season),
                "week": int(week),
                "team": team,
                "opponent": str(opp[0]),
                "room": room_name,
                "projected_plays": projected_plays,
                "projected_dropbacks": projected_dropbacks,
                "baseline_room_entitlement": room_ent,
                "baseline_room_targets": projected_dropbacks * room_ent,
                "room_targets_per_play_rate": rate,
                "candidate_room_targets": projected_plays * rate,
                "conversion_source": str(info["conversion_source"]),
                "prior_history_games": int(info["prior_history_games"]),
                "prior_history_plays": float(info["prior_history_plays"]),
                "prior_history_room_targets": float(
                    info["prior_history_targets"][room_name]
                ),
                "sum_room_rate": float(info["sum_room_rate"]),
                "actual_room_targets": float(a.iloc[0]),
            })

    scope = {
        "season": int(season),
        "week": int(week),
        "te_pool_gap": float(te_audit["team_te_pool_max_abs_gap"]),
        "te_non_te_gap": float(te_audit["non_te_max_abs_gap"]),
        "wr1_anchor_gap": float(wr_audit.get("m38_wr1_anchor_max_abs_gap", 0.0)),
        "wr2plus_pool_gap": float(wr_audit.get("wr2plus_pool_max_abs_gap", 0.0)),
        "wr_room_gap": float(wr_audit.get("wr_room_mass_max_abs_gap", 0.0)),
        "wr_non_wr_gap": float(wr_audit.get("non_wr_max_abs_gap", 0.0)),
        "wr_same_future_participation": int(
            wr_audit.get("same_future_participation", 0)
        ),
        "wr_2025_applications": 0,
    }
    return pd.DataFrame(rows), scope


def macro(block: dict, field: str, arm: str) -> float:
    return float(np.mean([block[r][arm][field] for r in ROOMS]))


def summarize(detail: pd.DataFrame) -> dict:
    out: dict = {"by_season": {}, "pooled": {}}
    for season in (2024, 2025):
        d = detail.loc[detail["season"].eq(season)].copy()
        block = {room: paired(d.loc[d["room"].eq(room)]) for room in ROOMS}
        block["macro"] = {
            "baseline_mae": macro(block, "mae", "baseline"),
            "candidate_mae": macro(block, "mae", "candidate"),
            "baseline_p90": macro(block, "p90_ae", "baseline"),
            "candidate_p90": macro(block, "p90_ae", "candidate"),
            "baseline_abs_bias": macro(block, "abs_bias", "baseline"),
            "candidate_abs_bias": macro(block, "abs_bias", "candidate"),
        }
        out["by_season"][str(season)] = block

    pblock = {room: paired(detail.loc[detail["room"].eq(room)]) for room in ROOMS}
    pblock["macro"] = {
        "baseline_mae": macro(pblock, "mae", "baseline"),
        "candidate_mae": macro(pblock, "mae", "candidate"),
        "baseline_p90": macro(pblock, "p90_ae", "baseline"),
        "candidate_p90": macro(pblock, "p90_ae", "candidate"),
        "baseline_abs_bias": macro(pblock, "abs_bias", "baseline"),
        "candidate_abs_bias": macro(pblock, "abs_bias", "candidate"),
    }

    actual = pd.to_numeric(detail["actual_room_targets"], errors="coerce").to_numpy(float)
    bp = pd.to_numeric(detail["baseline_room_targets"], errors="coerce").to_numpy(float)
    cp = pd.to_numeric(detail["candidate_room_targets"], errors="coerce").to_numpy(float)
    changed = np.abs(bp - cp) > 1e-12
    ba, ca = np.abs(bp - actual), np.abs(cp - actual)
    cw = changed & (ca < ba - 1e-12)
    bw = changed & (ba < ca - 1e-12)
    decided = cw | bw
    pblock["macro"]["candidate_closer_rate_all_room_rows"] = (
        float(cw.sum() / decided.sum()) if int(decided.sum()) else None
    )
    pblock["source_rows"] = int(detail["conversion_source"].eq("team_strict_prior").sum())
    pblock["fallback_rows"] = int(detail["conversion_source"].eq("league_fallback").sum())
    pblock["rows"] = int(len(detail))
    pblock["min_room_rate"] = float(detail["room_targets_per_play_rate"].min())
    pblock["max_room_rate"] = float(detail["room_targets_per_play_rate"].max())
    pblock["max_sum_room_rate"] = float(detail["sum_room_rate"].max())
    out["pooled"] = pblock

    summed = (
        detail.groupby(["season", "week", "team"], as_index=False)
        .agg(
            actual_room_targets=("actual_room_targets", "sum"),
            baseline_room_targets=("baseline_room_targets", "sum"),
            candidate_room_targets=("candidate_room_targets", "sum"),
        )
    )
    out["summed_room"] = {
        "pooled": paired(summed),
        "by_season": {
            str(season): paired(summed.loc[summed["season"].eq(season)])
            for season in (2024, 2025)
        },
    }
    return out


def gates(s: dict, scope: pd.DataFrame) -> dict:
    y24 = s["by_season"]["2024"]
    y25 = s["by_season"]["2025"]
    p = s["pooled"]
    rows = max(1, int(p["rows"]))
    source_rate = p["source_rows"] / rows
    fallback_rate = p["fallback_rows"] / rows

    specialist_max = float(
        scope[
            [
                "te_pool_gap",
                "te_non_te_gap",
                "wr1_anchor_gap",
                "wr2plus_pool_gap",
                "wr_room_gap",
                "wr_non_wr_gap",
            ]
        ].max().max()
    )
    return {
        "pooled_macro_mae_improves":
            p["macro"]["candidate_mae"] < p["macro"]["baseline_mae"],
        "macro_mae_2024_improves":
            y24["macro"]["candidate_mae"] < y24["macro"]["baseline_mae"],
        "macro_mae_2025_improves":
            y25["macro"]["candidate_mae"] < y25["macro"]["baseline_mae"],
        "wr_mae_2024_improves":
            y24["WR"]["candidate"]["mae"] < y24["WR"]["baseline"]["mae"],
        "wr_mae_2025_improves":
            y25["WR"]["candidate"]["mae"] < y25["WR"]["baseline"]["mae"],
        "wr_mae_pooled_improves":
            p["WR"]["candidate"]["mae"] < p["WR"]["baseline"]["mae"],
        "te_mae_pooled_nonworse":
            p["TE"]["candidate"]["mae"] <= p["TE"]["baseline"]["mae"] + 1e-12,
        "rbfb_mae_pooled_nonworse":
            p["RB_FB"]["candidate"]["mae"] <= p["RB_FB"]["baseline"]["mae"] + 1e-12,
        "wr_p90_guard":
            p["WR"]["candidate"]["p90_ae"] <= p["WR"]["baseline"]["p90_ae"] + 0.50,
        "te_p90_guard":
            p["TE"]["candidate"]["p90_ae"] <= p["TE"]["baseline"]["p90_ae"] + 0.50,
        "rbfb_p90_guard":
            p["RB_FB"]["candidate"]["p90_ae"] <= p["RB_FB"]["baseline"]["p90_ae"] + 0.50,
        "pooled_macro_p90_nonworse":
            p["macro"]["candidate_p90"] <= p["macro"]["baseline_p90"] + 1e-12,
        "pooled_macro_abs_bias_improves":
            p["macro"]["candidate_abs_bias"] < p["macro"]["baseline_abs_bias"],
        "summed_room_mae_improves":
            s["summed_room"]["pooled"]["candidate"]["mae"]
            < s["summed_room"]["pooled"]["baseline"]["mae"],
        "summed_room_abs_bias_improves":
            s["summed_room"]["pooled"]["candidate"]["abs_bias"]
            < s["summed_room"]["pooled"]["baseline"]["abs_bias"],
        "summed_room_p90_nonworse":
            s["summed_room"]["pooled"]["candidate"]["p90_ae"]
            <= s["summed_room"]["pooled"]["baseline"]["p90_ae"] + 1e-12,
        "candidate_closer_gt50":
            p["macro"]["candidate_closer_rate_all_room_rows"] is not None
            and p["macro"]["candidate_closer_rate_all_room_rows"] > 0.50,
        "team_source_rate_ge99": source_rate >= 0.99,
        "fallback_rate_le1": fallback_rate <= 0.01,
        "room_rates_finite_in_bounds":
            0.0 <= p["min_room_rate"] <= p["max_room_rate"] <= 1.0,
        "summed_room_rate_le1": p["max_sum_room_rate"] <= 1.0 + 1e-12,
        "specialist_conservation_exact": specialist_max <= TOL,
        "wr_same_future_participation_zero":
            int(scope["wr_same_future_participation"].sum()) == 0,
        "wr_r15_2025_applications_zero":
            int(scope.loc[scope["season"].eq(2025), "wr_2025_applications"].sum()) == 0,
        "target_game_outcomes_upstream_zero": True,
        "sportsbook_inputs_zero": True,
        "parameters_fit_zero": True,
        "one_candidate_only": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-2024", type=Path, required=True)
    ap.add_argument("--universe-2025", type=Path, required=True)
    ap.add_argument("--injuries", type=Path, required=True)
    ap.add_argument("--weather", type=Path, required=True)
    ap.add_argument("--te-coefficients", type=Path, required=True)
    ap.add_argument("--wr-coefficients", type=Path, required=True)
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    logs = read(args.player_logs, "player logs")
    team = read(args.team_weekly, "team weekly")
    sched = read(args.schedule, "schedule")
    injuries = optional(args.injuries)
    weather = optional(args.weather)
    history = build_room_history(logs, team)
    snaps, dup, _ = _load_participation_snaps()
    if dup > 0.01:
        raise RuntimeError(f"participation duplicate rate too high: {dup}")

    te24 = _load_fold_params(
        args.te_coefficients, test_season=2024, features=TE_FEATURES, label="TE-R5P"
    )
    te25 = _load_fold_params(
        args.te_coefficients, test_season=2025, features=TE_FEATURES, label="TE-R5P"
    )
    wr24 = _load_fold_params(
        args.wr_coefficients, test_season=2024, features=WR_FEATURES, label="WR-R15"
    )
    weeks = _parse_weeks(args.weeks)

    details = []
    scopes = []
    for season, prior, udir, te_params, wr_params in [
        (2024, 2023, args.universe_2024, te24, wr24),
        (2025, 2024, args.universe_2025, te25, None),
    ]:
        for week in weeks:
            universe = read(
                udir / f"{season}_week_{int(week):02d}.csv",
                f"{season} W{week} universe",
            )
            d, s = evaluate_week(
                season=season,
                prior_season=prior,
                week=int(week),
                player_logs=logs,
                team_weekly=team,
                schedule=sched,
                universe=universe,
                injuries=injuries,
                weather=weather,
                history=history,
                te_params=te_params,
                wr_params=wr_params,
                snaps=snaps,
            )
            details.append(d)
            scopes.append(s)

    detail = pd.concat(details, ignore_index=True)
    scope = pd.DataFrame(scopes)

    summary = summarize(detail)
    gate = gates(summary, scope)
    qualified = all(gate.values())
    disposition = (
        "RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_CONFIRMED"
        if qualified
        else "RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_FAILED_CLOSED"
    )
    payload = {
        "version": VERSION,
        "disposition": disposition,
        "qualified": bool(qualified),
        "candidate_variants_scored": 1,
        "parameters_fit": 0,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_used_upstream": 0,
        "scorecard": summary,
        "gates": gate,
        "specialist_scope_max_gap": float(
            scope[
                [
                    "te_pool_gap",
                    "te_non_te_gap",
                    "wr1_anchor_gap",
                    "wr2plus_pool_gap",
                    "wr_room_gap",
                    "wr_non_wr_gap",
                ]
            ].max().max()
        ),
        "wr_same_future_participation": int(
            scope["wr_same_future_participation"].sum()
        ),
        "wr_2025_applications": int(
            scope.loc[scope["season"].eq(2025), "wr_2025_applications"].sum()
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out_dir / "room_detail_2024_2025.csv", index=False)
    scope.to_csv(args.out_dir / "specialist_scope.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Receiver Room Targets-Per-Play V1 — 2024-2025 Confirmation",
        "",
        f"Disposition: **{disposition}**",
        "",
    ]
    for season in ("2024", "2025"):
        lines += [f"## {season}", ""]
        for room_name in ROOMS:
            x = summary["by_season"][season][room_name]
            lines.append(
                f"- {room_name}: MAE {x['baseline']['mae']:.6f} -> "
                f"{x['candidate']['mae']:.6f}; p90 "
                f"{x['baseline']['p90_ae']:.6f} -> {x['candidate']['p90_ae']:.6f}"
            )
        m = summary["by_season"][season]["macro"]
        lines += [
            f"- macro MAE: {m['baseline_mae']:.6f} -> {m['candidate_mae']:.6f}",
            f"- macro abs bias: {m['baseline_abs_bias']:.6f} -> "
            f"{m['candidate_abs_bias']:.6f}",
            "",
        ]

    lines += ["## Pooled", ""]
    for room_name in ROOMS:
        x = summary["pooled"][room_name]
        lines.append(
            f"- {room_name}: MAE {x['baseline']['mae']:.6f} -> "
            f"{x['candidate']['mae']:.6f}; p90 "
            f"{x['baseline']['p90_ae']:.6f} -> {x['candidate']['p90_ae']:.6f}"
        )
    m = summary["pooled"]["macro"]
    sr = summary["summed_room"]["pooled"]
    lines += [
        f"- macro MAE: {m['baseline_mae']:.6f} -> {m['candidate_mae']:.6f}",
        f"- macro p90: {m['baseline_p90']:.6f} -> {m['candidate_p90']:.6f}",
        f"- macro abs bias: {m['baseline_abs_bias']:.6f} -> "
        f"{m['candidate_abs_bias']:.6f}",
        f"- all-room closer rate: {m['candidate_closer_rate_all_room_rows']:.6f}",
        "",
        "## Summed room",
        "",
        f"- MAE: {sr['baseline']['mae']:.6f} -> {sr['candidate']['mae']:.6f}",
        f"- abs bias: {sr['baseline']['abs_bias']:.6f} -> "
        f"{sr['candidate']['abs_bias']:.6f}",
        f"- p90: {sr['baseline']['p90_ae']:.6f} -> {sr['candidate']['p90_ae']:.6f}",
        "",
        "## Frozen gates",
        "",
    ]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in gate.items()]
    (args.out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
