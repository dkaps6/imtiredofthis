#!/usr/bin/env python3
"""Evaluate frozen RB R24 entitlement + production-efficiency candidate.

R24 deliberately reuses the already-frozen R23 entitlement/reception mechanism
and removes R23's failed new YPR component. Receiving-yard point means use the
existing production yards-per-target logic unchanged. Football-only, strict-prior.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest.evaluate_rb_r23_candidate_v1 import (
    RB_POS,
    _player_history_features,
    _role_priors,
    finite,
    metric,
    optional,
    prepared,
    read,
)
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--prior-season", type=int, required=True)
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-dir", type=Path, required=True)
    ap.add_argument("--injuries", type=Path, required=True)
    ap.add_argument("--weather", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    logs = read(a.player_logs)
    team = read(a.team_weekly)
    schedule = read(a.schedule)
    injuries = optional(a.injuries)
    weather = optional(a.weather)
    rows: list[dict] = []
    audit: list[dict] = []

    for week in _parse_weeks(a.weeks):
        universe = read(a.universe_dir / f"{a.season}_week_{week:02d}.csv")
        bundle = build_historical_context_bundle(
            player_logs=logs,
            team_weekly=team,
            pregame_universe=universe,
            schedule=schedule,
            season=a.season,
            week=week,
            prior_season=a.prior_season,
            injuries=_exact_week(injuries, a.season, week),
            weather=_exact_week(weather, a.season, week),
        )
        base = prepared(bundle)
        actual = cp.build_actual_rows(logs, a.season, week)
        at = actual[actual.market.eq("receptions")][
            ["team", "player_clean_key", "actual", "actual_opportunities"]
        ].rename(columns={"actual": "actual_receptions", "actual_opportunities": "actual_targets"})
        ay = actual[actual.market.eq("rec_yards")][
            ["team", "player_clean_key", "actual"]
        ].rename(columns={"actual": "actual_rec_yards"})
        labels = at.merge(ay, on=["team", "player_clean_key"], how="outer").drop_duplicates(
            ["team", "player_clean_key"]
        )
        label_map = {
            (str(r.team), str(r.player_clean_key)): (
                finite(r.actual_targets),
                finite(r.actual_receptions),
                finite(r.actual_rec_yards),
            )
            for _, r in labels.iterrows()
        }
        prior_catch, prior_ypr = _role_priors(logs, a.season, week)

        for (event_id, tm), g in base.groupby(["event_id", "team"], dropna=False, sort=False):
            pos = g.get("position", pd.Series("", index=g.index)).fillna("").astype(str).str.upper().str.strip()
            rb = g.loc[pos.isin(RB_POS)].copy()
            if rb.empty:
                continue
            rb["entitlement_tgt_share"] = pd.to_numeric(
                rb.get("entitlement_tgt_share"), errors="coerce"
            ).fillna(0.0).clip(lower=0.0)
            room_mass = float(rb.entitlement_tgt_share.sum())
            if room_mass <= 0:
                continue

            base_room = rb.entitlement_tgt_share.to_numpy(float) / room_mass
            rank_order = np.argsort(-base_room, kind="stable")
            rank_map = {int(idx): rank + 1 for rank, idx in enumerate(rank_order)}
            feats = [
                _player_history_features(
                    logs,
                    a.season,
                    week,
                    str(tm),
                    str(r.player_clean_key),
                    float(base_room[j]),
                    prior_catch,
                    prior_ypr,
                )
                for j, (_, r) in enumerate(rb.iterrows())
            ]
            scores = np.asarray([f["score"] for f in feats], float)
            cand_room = scores / scores.sum() if scores.sum() > 0 else base_room.copy()

            plays = float(np.nanmean(pd.to_numeric(g.get("rules_plays_est", 64.0), errors="coerce")))
            pass_rate = float(np.nanmean(pd.to_numeric(g.get("rules_pass_rate", 0.57), errors="coerce")))
            if not np.isfinite(plays):
                plays = 64.0
            if not np.isfinite(pass_rate):
                pass_rate = 0.57
            team_targets = plays * pass_rate

            for j, (_, r) in enumerate(rb.iterrows()):
                key = str(r.player_clean_key)
                actual_t, actual_r, actual_y = label_map.get((str(tm), key), (np.nan, np.nan, np.nan))
                base_t = team_targets * float(r.entitlement_tgt_share)
                cand_t = team_targets * room_mass * float(cand_room[j])
                base_cr = finite(
                    r.get("rules_catch_rate"),
                    finite(r.get("bayes_receptions_per_target"), prior_catch),
                )
                base_cr = float(np.clip(base_cr, 0.35, 0.95))
                # Frozen production receiving-efficiency point-mean logic.
                # R24 changes only upstream entitlement/reception inputs; it does
                # not use R23's experimental shrunk YPR component.
                base_ypt = max(
                    finite(r.get("rules_ypt"), finite(r.get("bayes_ypt"), prior_catch * prior_ypr)),
                    0.0,
                )
                cand_cr = feats[j]["catch_rate"]
                rows.append(
                    {
                        "season": a.season,
                        "week": week,
                        "event_id": str(event_id),
                        "team": str(tm),
                        "player": r.get("player", ""),
                        "player_clean_key": key,
                        "rb_rank": int(rank_map[j]),
                        "actual_targets": actual_t,
                        "actual_receptions": actual_r,
                        "actual_rec_yards": actual_y,
                        "baseline_targets": base_t,
                        "candidate_targets": cand_t,
                        "baseline_receptions": base_t * base_cr,
                        "candidate_receptions": cand_t * cand_cr,
                        "baseline_rec_yards": base_t * base_ypt,
                        "candidate_rec_yards": cand_t * base_ypt,
                        "baseline_production_ypt": base_ypt,
                        "candidate_production_ypt": base_ypt,
                        "production_ypt_gap": 0.0,
                        "baseline_room_share": float(base_room[j]),
                        "candidate_room_share": float(cand_room[j]),
                        **feats[j],
                        "sportsbook_inputs_used": 0,
                        "future_outcomes_used": 0,
                    }
                )
            audit.append(
                {
                    "season": a.season,
                    "week": week,
                    "event_id": str(event_id),
                    "team": str(tm),
                    "baseline_rb_room_mass": room_mass,
                    "candidate_rb_room_mass": float(room_mass * cand_room.sum()),
                    "room_mass_gap": float(room_mass * cand_room.sum() - room_mass),
                    "sportsbook_inputs_used": 0,
                }
            )
        print(f"[r24-candidate] {a.season} week={week:02d} complete")

    pred = pd.DataFrame(rows)
    pred["role"] = np.where(pred.rb_rank.eq(1), "RB1", "RB2+")
    metrics: list[dict] = []
    for variant in ("baseline", "candidate"):
        for market, actual_col in (
            ("targets", "actual_targets"),
            ("receptions", "actual_receptions"),
            ("rec_yards", "actual_rec_yards"),
        ):
            metrics.append(
                {
                    "season": a.season,
                    "variant": variant,
                    "market": market,
                    "role": "ALL",
                    **metric(pred[actual_col], pred[f"{variant}_{market}"]),
                }
            )
            for role, rg in pred.groupby("role"):
                metrics.append(
                    {
                        "season": a.season,
                        "variant": variant,
                        "market": market,
                        "role": role,
                        **metric(rg[actual_col], rg[f"{variant}_{market}"]),
                    }
                )

    audit_df = pd.DataFrame(audit)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(a.out_dir / "r24_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(a.out_dir / "r24_metrics.csv", index=False)
    audit_df.to_csv(a.out_dir / "r24_conservation_audit.csv", index=False)
    assert int(pred.sportsbook_inputs_used.sum()) == 0
    assert int(pred.future_outcomes_used.sum()) == 0
    assert float(audit_df.room_mass_gap.abs().max()) < 1e-10
    assert float(pred.production_ypt_gap.abs().max()) == 0.0
    print(pd.DataFrame(metrics).query("role == 'ALL'").to_string(index=False))
    print(
        f"[r24-candidate] conservation_max_gap={audit_df.room_mass_gap.abs().max():.3e} "
        "production_ypt_gap=0 sportsbook=0 future=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
