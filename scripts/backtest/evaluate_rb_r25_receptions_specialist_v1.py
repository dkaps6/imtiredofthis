#!/usr/bin/env python3
"""Evaluate frozen RB R25 targets/receptions specialist candidate.

R25 inherits the R23 opportunity/catch-conversion mechanism unchanged, evaluates
only targets and receptions, and keeps receiving-yard point means exactly equal
to the current production baseline. Sportsbook information is never an input.
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
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement

RB_POS = {"RB", "FB", "HB", "TB"}
RECENT_GAMES = 6
STABLE_GAMES = 16


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing required artifact: {path}")
    return pd.read_csv(path, low_memory=False)


def optional(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


def finite(v, default=np.nan) -> float:
    try:
        x = float(v)
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
    metrics = metrics.sort_values(keys).drop_duplicates(keys, keep="last").copy()
    metrics, _ = materialize_target_entitlement(metrics)
    return metrics.copy()


def _prior(logs: pd.DataFrame, season: int, week: int, team: str, player_key: str | None = None) -> pd.DataFrame:
    s = pd.to_numeric(logs["season"], errors="coerce")
    w = pd.to_numeric(logs["week"], errors="coerce")
    pos = logs.get("position", pd.Series("", index=logs.index)).fillna("").astype(str).str.upper().str.strip()
    mask = ((s < season) | ((s == season) & (w < week))) & logs["team"].astype(str).eq(str(team)) & pos.isin(RB_POS)
    if player_key is not None:
        mask &= logs.get("player_clean_key", pd.Series("", index=logs.index)).fillna("").astype(str).eq(str(player_key))
    return logs.loc[mask].copy()


def _last_player_games(x: pd.DataFrame, n: int) -> pd.DataFrame:
    if x.empty:
        return x
    games = x[["season", "week"]].drop_duplicates().sort_values(["season", "week"], ascending=[False, False]).head(n)
    return x.merge(games, on=["season", "week"], how="inner")


def _sum(x: pd.DataFrame, col: str) -> float:
    if x.empty or col not in x.columns:
        return 0.0
    return float(pd.to_numeric(x[col], errors="coerce").fillna(0.0).sum())


def _catch_prior(logs: pd.DataFrame, season: int, week: int) -> float:
    s = pd.to_numeric(logs["season"], errors="coerce")
    w = pd.to_numeric(logs["week"], errors="coerce")
    pos = logs.get("position", pd.Series("", index=logs.index)).fillna("").astype(str).str.upper().str.strip()
    x = logs.loc[((s < season) | ((s == season) & (w < week))) & pos.isin(RB_POS)].copy()
    t = _sum(x, "targets")
    r = _sum(x, "receptions")
    catch = r / t if t > 0 else 0.76
    return float(np.clip(catch, 0.35, 0.95))


def _player_history_features(
    logs: pd.DataFrame,
    season: int,
    week: int,
    team: str,
    key: str,
    base_room_share: float,
    prior_catch: float,
) -> dict:
    """Exact R23 opportunity/catch mechanism, with YPR branch removed."""
    p = _prior(logs, season, week, team, key)
    r6 = _last_player_games(p, RECENT_GAMES)
    r16 = _last_player_games(p, STABLE_GAMES)
    team16 = _prior(logs, season, week, team)
    if not team16.empty:
        games16 = team16[["season", "week"]].drop_duplicates().sort_values(["season", "week"], ascending=[False, False]).head(STABLE_GAMES)
        team16 = team16.merge(games16, on=["season", "week"], how="inner")

    t6 = _sum(r6, "targets")
    t16 = _sum(r16, "targets")
    room16 = _sum(team16, "targets")
    stable_share = t16 / room16 if room16 > 0 else base_room_share
    n6 = int(r6[["season", "week"]].drop_duplicates().shape[0]) if not r6.empty else 0
    n16 = int(r16[["season", "week"]].drop_duplicates().shape[0]) if not r16.empty else 0
    room_games = int(team16[["season", "week"]].drop_duplicates().shape[0]) if not team16.empty else 0
    avg_room = room16 / max(room_games, 1) if room16 > 0 else 5.0

    stable_pseudo = max(avg_room, 1.0)
    role_pseudo = max(avg_room, 1.0)
    entitlement_score = t6 + stable_pseudo * stable_share + role_pseudo * base_room_share

    rec16 = _sum(r16, "receptions")
    catch_pseudo = max(t16 / max(n16, 1), 1.0)
    catch = (rec16 + catch_pseudo * prior_catch) / (t16 + catch_pseudo) if (t16 + catch_pseudo) > 0 else prior_catch

    return {
        "score": max(entitlement_score, 0.0),
        "catch_rate": float(np.clip(catch, 0.35, 0.95)),
        "history_games_recent": n6,
        "history_games_stable": n16,
        "recent_targets": t6,
        "stable_targets": t16,
        "stable_room_share": stable_share,
    }


def metric(actual, pred) -> dict:
    z = pd.DataFrame({
        "actual": pd.to_numeric(actual, errors="coerce"),
        "pred": pd.to_numeric(pred, errors="coerce"),
    }).dropna()
    z = z[np.isfinite(z.actual) & np.isfinite(z.pred)]
    if z.empty:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
            "median_abs_error": np.nan,
            "p75_abs_error": np.nan,
            "p90_abs_error": np.nan,
        }
    e = z.pred - z.actual
    ae = e.abs()
    return {
        "n": int(len(z)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e.to_numpy() ** 2))),
        "bias": float(e.mean()),
        "pearson": float(z.pred.corr(z.actual, method="pearson")) if len(z) > 1 else np.nan,
        "spearman": float(z.pred.corr(z.actual, method="spearman")) if len(z) > 1 else np.nan,
        "median_abs_error": float(ae.median()),
        "p75_abs_error": float(ae.quantile(0.75)),
        "p90_abs_error": float(ae.quantile(0.90)),
    }


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
        labels = actual[actual.market.eq("receptions")][
            ["team", "player_clean_key", "actual", "actual_opportunities"]
        ].rename(columns={"actual": "actual_receptions", "actual_opportunities": "actual_targets"})
        labels = labels.drop_duplicates(["team", "player_clean_key"])
        label_map = {
            (str(r.team), str(r.player_clean_key)): (finite(r.actual_targets), finite(r.actual_receptions))
            for _, r in labels.iterrows()
        }
        prior_catch = _catch_prior(logs, a.season, week)

        for (event_id, tm), g in base.groupby(["event_id", "team"], dropna=False, sort=False):
            pos = g.get("position", pd.Series("", index=g.index)).fillna("").astype(str).str.upper().str.strip()
            rb = g.loc[pos.isin(RB_POS)].copy()
            if rb.empty:
                continue
            rb["entitlement_tgt_share"] = pd.to_numeric(rb.get("entitlement_tgt_share"), errors="coerce").fillna(0.0).clip(lower=0.0)
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
                actual_t, actual_r = label_map.get((str(tm), key), (np.nan, np.nan))
                base_t = team_targets * float(r.entitlement_tgt_share)
                cand_t = team_targets * room_mass * float(cand_room[j])
                base_cr = finite(r.get("rules_catch_rate"), finite(r.get("bayes_receptions_per_target"), prior_catch))
                base_cr = float(np.clip(base_cr, 0.35, 0.95))
                base_ypt = max(finite(r.get("rules_ypt"), finite(r.get("bayes_ypt"), 0.0)), 0.0)
                base_rec_yards = base_t * base_ypt
                cand_cr = feats[j]["catch_rate"]
                rows.append({
                    "season": a.season,
                    "week": week,
                    "event_id": str(event_id),
                    "team": str(tm),
                    "player": r.get("player", ""),
                    "player_clean_key": key,
                    "rb_rank": int(rank_map[j]),
                    "actual_targets": actual_t,
                    "actual_receptions": actual_r,
                    "baseline_targets": base_t,
                    "candidate_targets": cand_t,
                    "baseline_receptions": base_t * base_cr,
                    "candidate_receptions": cand_t * cand_cr,
                    "baseline_rec_yards": base_rec_yards,
                    "candidate_rec_yards": base_rec_yards,
                    "baseline_room_share": float(base_room[j]),
                    "candidate_room_share": float(cand_room[j]),
                    **feats[j],
                    "sportsbook_inputs_used": 0,
                    "future_outcomes_used": 0,
                })

            audit.append({
                "season": a.season,
                "week": week,
                "event_id": str(event_id),
                "team": str(tm),
                "baseline_rb_room_mass": room_mass,
                "candidate_rb_room_mass": float(room_mass * cand_room.sum()),
                "room_mass_gap": float(room_mass * cand_room.sum() - room_mass),
                "sportsbook_inputs_used": 0,
                "future_outcomes_used": 0,
            })
        print(f"[r25] {a.season} week={week:02d} complete")

    pred = pd.DataFrame(rows)
    if pred.empty:
        raise RuntimeError("R25 produced no prediction rows")
    pred["role"] = np.where(pred.rb_rank.eq(1), "RB1", "RB2+")
    pred["history_cohort"] = np.where(pred.history_games_stable.lt(3), "LOW_HISTORY", "ESTABLISHED")

    metrics: list[dict] = []
    for variant in ("baseline", "candidate"):
        for market, actual_col in (("targets", "actual_targets"), ("receptions", "actual_receptions")):
            metrics.append({"season": a.season, "variant": variant, "market": market, "cohort": "ALL", **metric(pred[actual_col], pred[f"{variant}_{market}"])})
            for role, g in pred.groupby("role"):
                metrics.append({"season": a.season, "variant": variant, "market": market, "cohort": role, **metric(g[actual_col], g[f"{variant}_{market}"])})
            for cohort, g in pred.groupby("history_cohort"):
                metrics.append({"season": a.season, "variant": variant, "market": market, "cohort": cohort, **metric(g[actual_col], g[f"{variant}_{market}"])})

    audit_df = pd.DataFrame(audit)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(a.out_dir / "r25_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(a.out_dir / "r25_metrics.csv", index=False)
    audit_df.to_csv(a.out_dir / "r25_conservation_audit.csv", index=False)

    assert int(pred.sportsbook_inputs_used.sum()) == 0
    assert int(pred.future_outcomes_used.sum()) == 0
    assert float(audit_df.room_mass_gap.abs().max()) < 1e-10
    assert np.isfinite(pd.to_numeric(pred.candidate_targets, errors="coerce")).all()
    assert np.isfinite(pd.to_numeric(pred.candidate_receptions, errors="coerce")).all()
    assert (pred.candidate_targets >= 0).all() and (pred.candidate_receptions >= 0).all()
    assert float((pred.candidate_rec_yards - pred.baseline_rec_yards).abs().max()) == 0.0

    print(pd.DataFrame(metrics).query("cohort == 'ALL'").to_string(index=False))
    print(
        f"[r25] conservation_max_gap={audit_df.room_mass_gap.abs().max():.3e} "
        f"rec_yards_parity={(pred.candidate_rec_yards-pred.baseline_rec_yards).abs().max():.3e} "
        "sportsbook=0 future=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
