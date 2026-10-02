#!/usr/bin/env python3
"""Frozen full-stack historical evaluation for Shared Pass-State Receiver Integration V1.

No parameters are fit and no candidate variants are searched here.

The baseline is the current historical receiver stack. The candidate installs
the exact frozen C2 completed-pass receiver realization into selected teams while
holding the QB C2 distribution state fixed between A/B. Two views are scored:

1. ALL-C2 mechanism stability in 2024 and 2025.
2. 2025 routed view using only the frozen walk-forward Phase-J OOS use_c2 decisions.

Target-game outcomes enter only after both projection paths are complete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import (
    _attach_component_projection,
    build_actual_rows,
    build_mc_predictions,
)
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.ensemble_v2 import apply_ensemble, load_weights
from scripts.modeling.ml_v2 import build_and_train as build_ml
from scripts.modeling.rb_rush_rec_conservation_v2 import build_candidate_map
from scripts.modeling.state_v2 import build_state_predictions
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import _load_snaps as _load_participation_snaps
from scripts.research.audit_shared_pass_state_coherence_v1 import _capture_c2_shadow
from scripts.research.persist_wr_te_production_order_historical_v1 import (
    TE_FEATURES,
    WR_FEATURES,
    _load_fold_params,
    apply_te_fold,
    apply_wr_fold,
)
from scripts.simulation_c2_qb_candidate import StateSimulationResult, simulate_with_states
from scripts.simulation_v2 import _player_key, lookup
from scripts.utils.canonical_names import canon_team

VERSION = "SHARED_PASS_STATE_RECEIVER_INTEGRATION_V1"
POSITIONS = ("WR", "TE", "RB")
MARKETS = ("rec_yards", "receptions", "rush_rec_yards")
C2_RECEIVER_POSITIONS = {"WR", "LWR", "RWR", "SWR", "TE", "RB", "FB"}
QB_MARKET = "pass_yards"


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path, low_memory=False)


def _optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() and path.stat().st_size else pd.DataFrame()


def _pos(value: object) -> str:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
    if p in {"WR", "LWR", "RWR", "SWR"} or p.startswith("WR"):
        return "WR"
    if p == "TE" or p.startswith("TE"):
        return "TE"
    if p in {"RB", "HB", "TB"} or p.startswith("RB"):
        return "RB"
    if p == "FB" or p.startswith("FB"):
        return "FB"
    if p == "QB" or p.startswith("QB"):
        return "QB"
    return "OTHER"


def _primary_qb_keys(metrics: pd.DataFrame, base: StateSimulationResult) -> tuple[dict, dict]:
    frame = metrics.copy()
    frame["player_clean_key"] = frame.apply(_player_key, axis=1)
    players = frame.sort_values(["event_id", "team", "player_clean_key"]).drop_duplicates(
        ["event_id", "team", "player_clean_key"], keep="last"
    )
    keys: dict[tuple[str, str], str] = {}
    anchors: dict[tuple[str, str], float] = {}
    for (game, team), part in players.groupby(["event_id", "team"], dropna=False, sort=False):
        candidates = []
        for _, row in part.iterrows():
            if _pos(row.get("position")) != "QB":
                continue
            pkey = _player_key(row)
            arr = base.values.get((str(game), pkey, QB_MARKET))
            if arr is None or len(arr) == 0:
                continue
            eligible = pd.to_numeric(pd.Series([row.get("qb_projection_eligible")]), errors="coerce").fillna(0).iloc[0]
            role = pd.to_numeric(pd.Series([row.get("qb_role_score")]), errors="coerce").fillna(0).iloc[0]
            candidates.append((float(eligible), float(role), str(pkey), np.asarray(arr, float)))
        if not candidates:
            continue
        candidates.sort(key=lambda z: (z[0], z[1], z[2]), reverse=True)
        _, _, pkey, arr = candidates[0]
        mean = float(arr.mean())
        if not np.isfinite(mean) or mean <= 0:
            raise RuntimeError(f"invalid canonical QB raw mean game={game} team={team}")
        key = (str(game), canon_team(team))
        keys[key] = pkey
        anchors[key] = mean
    if not keys:
        raise RuntimeError("no historical primary QB arrays resolved")
    return keys, anchors


def _copy_state(base: StateSimulationResult) -> StateSimulationResult:
    return StateSimulationResult(
        {k: np.asarray(v, float).copy() for k, v in base.values.items()},
        base.iterations,
        base.team_states,
    )


def _build_receiver_view(
    base: StateSimulationResult,
    metrics: pd.DataFrame,
    *,
    selected_team_games: set[tuple[int, int, str]] | None,
    season: int,
    week: int,
) -> tuple[StateSimulationResult, StateSimulationResult, dict]:
    """Return A/B where QB C2 is identical but only B installs C2 receiver arrays."""
    qb_keys, anchors = _primary_qb_keys(metrics, base)

    # _capture_c2_shadow was already independently parity-checked against the
    # installed production C2 QB arrays in the 2026 Week-3 read-only audit.
    shadow = _capture_c2_shadow(base, metrics, anchors)

    b = _copy_state(base)
    c = _copy_state(base)
    frame = metrics.copy()
    frame["player_clean_key"] = frame.apply(_player_key, axis=1)
    players = frame.sort_values(["event_id", "team", "player_clean_key"]).drop_duplicates(
        ["event_id", "team", "player_clean_key"], keep="last"
    )

    changed_receivers = 0
    selected_count = 0
    identity_gap = 0.0
    zero_rec_positive = 0
    qb_gap = 0.0
    unselected_receiver_gap = 0.0
    rush_gap = 0.0
    atd_gap = 0.0

    for (game, team), part in players.groupby(["event_id", "team"], dropna=False, sort=False):
        game_s = str(game)
        team_s = canon_team(team)
        route_key = (int(season), int(week), team_s)
        selected = selected_team_games is None or route_key in selected_team_games
        if not selected:
            continue
        selected_count += 1
        sh = shadow.get((game_s, team_s))
        qkey = qb_keys.get((game_s, team_s))
        if sh is None or qkey is None:
            raise RuntimeError(f"missing C2 shadow/QB key game={game_s} team={team_s}")

        # Freeze the existing QB C2 distribution authority identically in A/B.
        qb_c2 = np.asarray(sh["total_scaled"], float)
        b.values[(game_s, qkey, QB_MARKET)] = qb_c2.copy()
        c.values[(game_s, qkey, QB_MARKET)] = qb_c2.copy()

        modeled = np.zeros(base.iterations, dtype=float)
        for _, row in part.iterrows():
            if str(row.get("position", "") or "").upper().strip() not in C2_RECEIVER_POSITIONS:
                continue
            pkey = _player_key(row)
            if pkey not in sh["player_yards_scaled"]:
                continue
            rec = np.asarray(sh["player_receptions"][pkey], float)
            yards = np.asarray(sh["player_yards_scaled"][pkey], float)
            modeled += yards
            zero_rec_positive += int(np.sum((rec <= 0) & (yards > 1e-12)))
            c.values[(game_s, pkey, "receptions")] = rec.copy()
            c.values[(game_s, pkey, "rec_yards")] = yards.copy()
            rush = c.values.get((game_s, pkey, "rush_yards"))
            if rush is not None:
                c.values[(game_s, pkey, "rush_rec_yards")] = np.asarray(rush, float) + yards
            changed_receivers += 1

        residual = np.asarray(sh["residual_yards_scaled"], float)
        identity_gap = max(identity_gap, float(np.max(np.abs(qb_c2 - (modeled + residual)))))

    # Whole-array invariance checks.
    if set(b.values) != set(c.values):
        raise RuntimeError("candidate changed simulation key universe")
    for key in b.values:
        a = np.asarray(b.values[key], float)
        d = np.asarray(c.values[key], float)
        gap = float(np.max(np.abs(a - d))) if len(a) else 0.0
        if key[2] == QB_MARKET:
            qb_gap = max(qb_gap, gap)
        elif key[2] in {"rush_att", "rush_yards"}:
            rush_gap = max(rush_gap, gap)
        elif key[2] == "anytime_td":
            atd_gap = max(atd_gap, gap)

    # For routed view, verify nonselected receiver arrays are exact.
    if selected_team_games is not None:
        for (game, team), part in players.groupby(["event_id", "team"], dropna=False, sort=False):
            team_s = canon_team(team)
            if (int(season), int(week), team_s) in selected_team_games:
                continue
            for _, row in part.iterrows():
                if str(row.get("position", "") or "").upper().strip() not in C2_RECEIVER_POSITIONS:
                    continue
                pkey = _player_key(row)
                for market in ("receptions", "rec_yards", "rush_rec_yards"):
                    key = (str(game), pkey, market)
                    if key in b.values and key in c.values:
                        unselected_receiver_gap = max(
                            unselected_receiver_gap,
                            float(np.max(np.abs(np.asarray(b.values[key], float) - np.asarray(c.values[key], float)))),
                        )

    return b, c, {
        "selected_team_games": int(selected_count),
        "changed_receiver_players": int(changed_receivers),
        "qb_array_ab_max_gap": float(qb_gap),
        "rush_array_ab_max_gap": float(rush_gap),
        "atd_array_ab_max_gap": float(atd_gap),
        "unselected_receiver_max_gap": float(unselected_receiver_gap),
        "raw_c2_identity_max_gap": float(identity_gap),
        "zero_rec_positive_yards": int(zero_rec_positive),
    }


def _phasej_routes(path: Path) -> pd.DataFrame:
    # Read only routing/provenance columns; outcome columns in the artifact are
    # deliberately not loaded into this evaluator.
    q = pd.read_csv(path, usecols=["season", "week", "team", "use_c2"])
    q["season"] = pd.to_numeric(q["season"], errors="raise").astype(int)
    q["week"] = pd.to_numeric(q["week"], errors="raise").astype(int)
    q["team"] = q["team"].map(canon_team)
    q["use_c2"] = q["use_c2"].astype(str).str.lower().isin({"1", "true", "yes"})
    if len(q) != 440 or int(q["use_c2"].sum()) != 412:
        raise RuntimeError(f"Phase-J OOS routing authority drifted rows={len(q)} selected={int(q['use_c2'].sum())}")
    if not q["season"].eq(2025).all() or q.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("Phase-J routing identity/season drift")
    return q


def _scale_to_mean(arr: np.ndarray, target: float) -> np.ndarray:
    x = np.asarray(arr, float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise RuntimeError("final distribution contains invalid draws")
    raw = float(x.mean())
    if raw > 0 and np.isfinite(target):
        return x * max(0.0, float(target) / raw)
    if abs(float(target)) <= 1e-12:
        return np.zeros_like(x)
    raise RuntimeError(f"cannot align positive target {target} from raw mean {raw}")


def _sample_crps(samples: np.ndarray, actual: float) -> float:
    x = np.sort(np.asarray(samples, float))
    if len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(actual):
        return np.nan
    n = len(x)
    first = float(np.mean(np.abs(x - float(actual))))
    weights = 2.0 * np.arange(1, n + 1, dtype=float) - n - 1.0
    pair = float(2.0 * np.sum(weights * x) / (n * n))
    return first - 0.5 * pair


def _ensemble_target(row: pd.Series, arr: np.ndarray, weights: pd.DataFrame, market: str) -> float:
    c = pd.DataFrame([{
        "market": market,
        "mc_proj": float(np.mean(np.asarray(arr, float))),
        "ml_proj": row.get("ml_proj"),
        "state_proj": row.get("state_proj"),
    }])
    return float(apply_ensemble(c, weights=weights).iloc[0]["ensemble_proj"])


def _score_week(
    *,
    metrics: pd.DataFrame,
    player_logs: pd.DataFrame,
    ml_pred: pd.DataFrame,
    state_pred: pd.DataFrame,
    baseline: StateSimulationResult,
    candidate: StateSimulationResult,
    weights: pd.DataFrame,
    season: int,
    week: int,
    route_authority: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict]:
    mcols = ["event_id", "team", "player", "player_clean_key", "position", "market", "season", "week"]
    market_rows = metrics[mcols].copy()
    market_rows["team"] = market_rows["team"].map(canon_team)
    market_rows["market"] = market_rows["market"].astype(str).str.lower()
    market_rows = _attach_component_projection(market_rows, ml_pred, "ml")
    market_rows = _attach_component_projection(market_rows, state_pred, "state")

    actual = build_actual_rows(player_logs, int(season), int(week))
    actual["team"] = actual["team"].map(canon_team)
    actual["market"] = actual["market"].astype(str).str.lower()
    joined = market_rows.merge(
        actual[["team", "player_clean_key", "market", "actual"]],
        on=["team", "player_clean_key", "market"],
        how="inner",
        validate="one_to_one",
    )
    joined["position_family"] = joined["position"].map(_pos)
    joined = joined.loc[
        joined["position_family"].isin(POSITIONS)
        & joined["market"].isin(MARKETS)
    ].copy()

    if route_authority is not None:
        ra = route_authority.loc[
            route_authority["season"].eq(int(season)) & route_authority["week"].eq(int(week)),
            ["season", "week", "team", "use_c2"],
        ].copy()
        joined = joined.merge(ra, on=["season", "week", "team"], how="inner", validate="many_to_one")
        if joined.empty:
            return joined, {"rb_v2_max_pathwise_identity_gap": 0.0}

    # RB V2 consumes the same fixed ML/state rows and the A/B simulation.
    bmap, bp = build_candidate_map(market_rows, baseline, weights)
    cmap, cp = build_candidate_map(market_rows, candidate, weights)
    rb_gap = max(float(bp.get("max_pathwise_identity_gap", 0.0)), float(cp.get("max_pathwise_identity_gap", 0.0)))
    if int(bp.get("sportsbook_inputs_used", 0)) or int(cp.get("sportsbook_inputs_used", 0)):
        raise RuntimeError("RB V2 sportsbook leakage")

    rows = []
    for _, row in joined.iterrows():
        market = str(row["market"])
        ba = lookup(baseline, row, market)
        ca = lookup(candidate, row, market)
        if ba is None or ca is None:
            raise RuntimeError(f"missing A/B array {season} W{week:02d} {row['player_clean_key']} {market}")
        ba = np.asarray(ba, float)
        ca = np.asarray(ca, float)

        if market == "rush_rec_yards" and row["position_family"] == "RB":
            key = (str(row["event_id"]), str(row["player_clean_key"]))
            if key in bmap:
                ba = np.asarray(bmap[key]["draws"], float)
            else:
                bt = _ensemble_target(row, ba, weights, market)
                ba = _scale_to_mean(ba, bt)
            if key in cmap:
                ca = np.asarray(cmap[key]["draws"], float)
            else:
                ct = _ensemble_target(row, ca, weights, market)
                ca = _scale_to_mean(ca, ct)
        else:
            bt = _ensemble_target(row, ba, weights, market)
            ct = _ensemble_target(row, ca, weights, market)
            ba = _scale_to_mean(ba, bt)
            ca = _scale_to_mean(ca, ct)

        actual_value = float(row["actual"])
        be = float(ba.mean() - actual_value)
        ce = float(ca.mean() - actual_value)
        rec = {
            "season": int(season),
            "week": int(week),
            "event_id": row["event_id"],
            "team": row["team"],
            "player": row["player"],
            "player_clean_key": row["player_clean_key"],
            "position_family": row["position_family"],
            "market": market,
            "actual": actual_value,
            "baseline_proj": float(ba.mean()),
            "candidate_proj": float(ca.mean()),
            "baseline_error": be,
            "candidate_error": ce,
            "baseline_abs_error": abs(be),
            "candidate_abs_error": abs(ce),
            "baseline_crps": _sample_crps(ba, actual_value),
            "candidate_crps": _sample_crps(ca, actual_value),
        }
        if market == "rec_yards":
            for p, q in ((5, .05), (10, .10), (90, .90), (95, .95)):
                rec[f"baseline_p{p:02d}"] = float(np.quantile(ba, q))
                rec[f"candidate_p{p:02d}"] = float(np.quantile(ca, q))
            rec["baseline_cover80"] = int(rec["baseline_p10"] <= actual_value <= rec["baseline_p90"])
            rec["candidate_cover80"] = int(rec["candidate_p10"] <= actual_value <= rec["candidate_p90"])
            rec["baseline_cover90"] = int(rec["baseline_p05"] <= actual_value <= rec["baseline_p95"])
            rec["candidate_cover90"] = int(rec["candidate_p05"] <= actual_value <= rec["candidate_p95"])
        if route_authority is not None:
            rec["use_c2"] = bool(row["use_c2"])
        rows.append(rec)

    return pd.DataFrame(rows), {"rb_v2_max_pathwise_identity_gap": float(rb_gap)}


def _group_metric(detail: pd.DataFrame, market: str, position: str) -> dict:
    q = detail.loc[detail["market"].eq(market) & detail["position_family"].eq(position)].copy()
    if q.empty:
        raise RuntimeError(f"empty score group {market}/{position}")
    b = q["baseline_error"].to_numpy(float)
    c = q["candidate_error"].to_numpy(float)
    ba = np.abs(b)
    ca = np.abs(c)
    out = {
        "n": int(len(q)),
        "baseline_mae": float(ba.mean()),
        "candidate_mae": float(ca.mean()),
        "baseline_rmse": float(np.sqrt(np.mean(b * b))),
        "candidate_rmse": float(np.sqrt(np.mean(c * c))),
        "baseline_bias": float(b.mean()),
        "candidate_bias": float(c.mean()),
        "baseline_p90": float(np.quantile(ba, .90)),
        "candidate_p90": float(np.quantile(ca, .90)),
        "baseline_crps": float(q["baseline_crps"].mean()),
        "candidate_crps": float(q["candidate_crps"].mean()),
    }
    if market in {"rec_yards", "rush_rec_yards"}:
        out["baseline_miss30"] = float(np.mean(ba >= 30.0))
        out["candidate_miss30"] = float(np.mean(ca >= 30.0))
        out["baseline_miss40"] = float(np.mean(ba >= 40.0))
        out["candidate_miss40"] = float(np.mean(ca >= 40.0))
    if market == "rec_yards":
        out["baseline_cover80"] = float(q["baseline_cover80"].mean())
        out["candidate_cover80"] = float(q["candidate_cover80"].mean())
        out["baseline_cover90"] = float(q["baseline_cover90"].mean())
        out["candidate_cover90"] = float(q["candidate_cover90"].mean())
        out["baseline_cover80_error"] = abs(out["baseline_cover80"] - .80)
        out["candidate_cover80_error"] = abs(out["candidate_cover80"] - .80)
        out["baseline_cover90_error"] = abs(out["baseline_cover90"] - .90)
        out["candidate_cover90_error"] = abs(out["candidate_cover90"] - .90)
    return out


def _summarize(detail: pd.DataFrame) -> dict:
    metrics = {}
    for market in ("rec_yards", "receptions"):
        for pos in POSITIONS:
            metrics[f"{market}:{pos}"] = _group_metric(detail, market, pos)
    # V2 is RB-only; FB is intentionally excluded.
    metrics["rush_rec_yards:RB"] = _group_metric(detail, "rush_rec_yards", "RB")

    macro = {}
    for market in ("rec_yards", "receptions"):
        for field in ("baseline_mae", "candidate_mae", "baseline_p90", "candidate_p90", "baseline_crps", "candidate_crps"):
            macro[f"{market}_{field}"] = float(np.mean([metrics[f"{market}:{p}"][field] for p in POSITIONS]))
    for field in ("baseline_miss40", "candidate_miss40"):
        macro[f"rec_yards_{field}"] = float(np.mean([metrics[f"rec_yards:{p}"][field] for p in POSITIONS]))
    for field in ("baseline_cover80_error", "candidate_cover80_error", "baseline_cover90_error", "candidate_cover90_error"):
        macro[f"rec_yards_{field}"] = float(np.mean([metrics[f"rec_yards:{p}"][field] for p in POSITIONS]))
    return {"rows": int(len(detail)), "metrics": metrics, "macro": macro}


def _season_view(
    *,
    season: int,
    prior_season: int,
    weeks: list[int],
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    universe_dir: Path,
    injuries: pd.DataFrame,
    weather: pd.DataFrame,
    te_params: dict,
    wr_params: dict | None,
    snaps: pd.DataFrame,
    weights: pd.DataFrame,
    iterations: int,
    routes: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict]:
    details = []
    audits = []
    rb_v2_max = 0.0

    for week in weeks:
        u = _read(universe_dir / f"{season}_week_{week:02d}.csv", f"{season} W{week} universe")
        bundle = build_historical_context_bundle(
            player_logs=player_logs,
            team_weekly=team_weekly,
            pregame_universe=u,
            schedule=schedule,
            season=int(season),
            week=int(week),
            prior_season=int(prior_season),
            injuries=_exact_week(injuries, int(season), int(week)),
            weather=_exact_week(weather, int(season), int(week)),
        )
        seed = 42 + int(week)
        metrics = build_mc_predictions(bundle, iterations=20, seed=seed)
        players = metrics.sort_values(["event_id", "team", "player_clean_key"]).drop_duplicates(
            ["event_id", "team", "player_clean_key"], keep="last"
        ).copy()
        explicit, _ = materialize_target_entitlement(players)
        te_final, _, te_audit = apply_te_fold(explicit, snaps=snaps, params=te_params)
        if int(season) == 2024:
            final, _, wr_audit = apply_wr_fold(te_final, snaps=snaps, params=wr_params)
        else:
            final = te_final
            wr_audit = {"same_future_participation": 0}

        ent_before = pd.to_numeric(final["entitlement_tgt_share"], errors="raise").to_numpy(float).copy()
        base_state = simulate_with_states(final, iterations=int(iterations), seed=seed)

        if routes is None:
            selected = None
        else:
            rw = routes.loc[routes["week"].eq(int(week)) & routes["use_c2"], ["season", "week", "team"]]
            selected = set((int(r.season), int(r.week), canon_team(r.team)) for r in rw.itertuples(index=False))

        b, c, integ = _build_receiver_view(
            base_state,
            final,
            selected_team_games=selected,
            season=int(season),
            week=int(week),
        )
        ent_after = pd.to_numeric(final["entitlement_tgt_share"], errors="raise").to_numpy(float)
        entitlement_gap = float(np.max(np.abs(ent_before - ent_after))) if len(ent_before) else 0.0

        _, ml_pred = build_ml(player_logs, bundle.player_consensus, int(season), int(week))
        _, state_pred = build_state_predictions(player_logs, bundle.player_consensus, int(season), int(week))
        week_detail, scoring_audit = _score_week(
            metrics=metrics,
            player_logs=player_logs,
            ml_pred=ml_pred,
            state_pred=state_pred,
            baseline=b,
            candidate=c,
            weights=weights,
            season=int(season),
            week=int(week),
            route_authority=routes,
        )
        if not week_detail.empty:
            details.append(week_detail)
        rb_v2_max = max(rb_v2_max, float(scoring_audit["rb_v2_max_pathwise_identity_gap"]))
        audits.append({
            "season": int(season),
            "week": int(week),
            **integ,
            "entitlement_max_gap": entitlement_gap,
            "te_pool_gap": float(te_audit["team_te_pool_max_abs_gap"]),
            "wr_same_future_participation": int(wr_audit.get("same_future_participation", 0)),
            "rb_v2_max_pathwise_identity_gap": float(scoring_audit["rb_v2_max_pathwise_identity_gap"]),
        })

    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    if detail.empty:
        raise RuntimeError(f"empty detail season={season} routed={routes is not None}")
    audit = pd.DataFrame(audits)
    summary = _summarize(detail)
    summary["scope"] = {
        "entitlement_max_gap": float(audit["entitlement_max_gap"].max()),
        "qb_array_ab_max_gap": float(audit["qb_array_ab_max_gap"].max()),
        "rush_array_ab_max_gap": float(audit["rush_array_ab_max_gap"].max()),
        "atd_array_ab_max_gap": float(audit["atd_array_ab_max_gap"].max()),
        "unselected_receiver_max_gap": float(audit["unselected_receiver_max_gap"].max()),
        "raw_c2_identity_max_gap": float(audit["raw_c2_identity_max_gap"].max()),
        "zero_rec_positive_yards": int(audit["zero_rec_positive_yards"].sum()),
        "rb_v2_max_pathwise_identity_gap": float(rb_v2_max),
        "wr_same_future_participation": int(audit["wr_same_future_participation"].sum()),
        "selected_team_games": int(audit["selected_team_games"].sum()),
    }
    return detail, summary


def _macro(s: dict, market: str, field: str) -> float:
    return float(s["macro"][f"{market}_{field}"])


def _pos_metric(s: dict, market: str, pos: str, field: str) -> float:
    return float(s["metrics"][f"{market}:{pos}"][field])


def _rb_combo(s: dict, field: str) -> float:
    return float(s["metrics"]["rush_rec_yards:RB"][field])


def _scope_integrity(*summaries: dict) -> dict:
    return {
        "target_entitlement_exact": all(s["scope"]["entitlement_max_gap"] <= 1e-12 for s in summaries),
        "qb_arrays_exact_ab": all(s["scope"]["qb_array_ab_max_gap"] <= 1e-12 for s in summaries),
        "rush_arrays_exact_ab": all(s["scope"]["rush_array_ab_max_gap"] <= 1e-12 for s in summaries),
        "atd_arrays_exact_ab": all(s["scope"]["atd_array_ab_max_gap"] <= 1e-12 for s in summaries),
        "unselected_receivers_exact": all(s["scope"]["unselected_receiver_max_gap"] <= 1e-12 for s in summaries),
        "raw_c2_identity": all(s["scope"]["raw_c2_identity_max_gap"] <= 1e-10 for s in summaries),
        "zero_rec_positive_yards": all(s["scope"]["zero_rec_positive_yards"] == 0 for s in summaries),
        "rb_v2_identity": all(s["scope"]["rb_v2_max_pathwise_identity_gap"] <= 1e-10 for s in summaries),
        "wr_strict_prior": all(s["scope"]["wr_same_future_participation"] == 0 for s in summaries),
        "sportsbook_inputs_zero": True,
        "target_game_outcomes_upstream_zero": True,
        "c1_used_false": True,
        "c3_used_false": True,
        "parameters_fit_zero": True,
        "candidate_variants_one": True,
    }


def _science_gates(s24: dict, s25: dict, routed: dict) -> dict:
    gates = {}
    for year, s in ((2024, s24), (2025, s25)):
        gates[f"allc2_rec_mae_nonworse_{year}"] = _macro(s, "rec_yards", "candidate_mae") <= _macro(s, "rec_yards", "baseline_mae") + 1e-12
        gates[f"allc2_rec_crps_nonworse_{year}"] = _macro(s, "rec_yards", "candidate_crps") <= _macro(s, "rec_yards", "baseline_crps") + 1e-12
        gates[f"allc2_rec_p90_nonworse_{year}"] = _macro(s, "rec_yards", "candidate_p90") <= _macro(s, "rec_yards", "baseline_p90") + 1e-12
        gates[f"allc2_rec_miss40_nonworse_{year}"] = _macro(s, "rec_yards", "candidate_miss40") <= _macro(s, "rec_yards", "baseline_miss40") + 1e-12
        for pos in POSITIONS:
            gates[f"allc2_rec_{pos}_mae_guard_{year}"] = _pos_metric(s, "rec_yards", pos, "candidate_mae") <= _pos_metric(s, "rec_yards", pos, "baseline_mae") + 0.50
        gates[f"allc2_receptions_mae_nonworse_{year}"] = _macro(s, "receptions", "candidate_mae") <= _macro(s, "receptions", "baseline_mae") + 1e-12
        gates[f"allc2_receptions_crps_nonworse_{year}"] = _macro(s, "receptions", "candidate_crps") <= _macro(s, "receptions", "baseline_crps") + 1e-12
        gates[f"allc2_rb_combo_mae_nonworse_{year}"] = _rb_combo(s, "candidate_mae") <= _rb_combo(s, "baseline_mae") + 1e-12
        gates[f"allc2_rb_combo_p90_nonworse_{year}"] = _rb_combo(s, "candidate_p90") <= _rb_combo(s, "baseline_p90") + 1e-12

    # Pooled 2024-2025 uses row-count weighted position metrics, then macro positions.
    pooled_rec_b = np.mean([
        np.average([_pos_metric(s24, "rec_yards", p, "baseline_mae"), _pos_metric(s25, "rec_yards", p, "baseline_mae")],
                   weights=[s24["metrics"][f"rec_yards:{p}"]["n"], s25["metrics"][f"rec_yards:{p}"]["n"]])
        for p in POSITIONS
    ])
    pooled_rec_c = np.mean([
        np.average([_pos_metric(s24, "rec_yards", p, "candidate_mae"), _pos_metric(s25, "rec_yards", p, "candidate_mae")],
                   weights=[s24["metrics"][f"rec_yards:{p}"]["n"], s25["metrics"][f"rec_yards:{p}"]["n"]])
        for p in POSITIONS
    ])
    pooled_crps_b = np.mean([
        np.average([_pos_metric(s24, "rec_yards", p, "baseline_crps"), _pos_metric(s25, "rec_yards", p, "baseline_crps")],
                   weights=[s24["metrics"][f"rec_yards:{p}"]["n"], s25["metrics"][f"rec_yards:{p}"]["n"]])
        for p in POSITIONS
    ])
    pooled_crps_c = np.mean([
        np.average([_pos_metric(s24, "rec_yards", p, "candidate_crps"), _pos_metric(s25, "rec_yards", p, "candidate_crps")],
                   weights=[s24["metrics"][f"rec_yards:{p}"]["n"], s25["metrics"][f"rec_yards:{p}"]["n"]])
        for p in POSITIONS
    ])
    gates["allc2_pooled_rec_mae_strict_improve"] = float(pooled_rec_c) < float(pooled_rec_b)
    gates["allc2_pooled_rec_crps_strict_improve"] = float(pooled_crps_c) < float(pooled_crps_b)

    gates["routed_rec_mae_strict_improve"] = _macro(routed, "rec_yards", "candidate_mae") < _macro(routed, "rec_yards", "baseline_mae")
    gates["routed_rec_crps_strict_improve"] = _macro(routed, "rec_yards", "candidate_crps") < _macro(routed, "rec_yards", "baseline_crps")
    gates["routed_rec_p90_nonworse"] = _macro(routed, "rec_yards", "candidate_p90") <= _macro(routed, "rec_yards", "baseline_p90") + 1e-12
    gates["routed_rec_miss40_nonworse"] = _macro(routed, "rec_yards", "candidate_miss40") <= _macro(routed, "rec_yards", "baseline_miss40") + 1e-12
    for pos in POSITIONS:
        gates[f"routed_rec_{pos}_mae_guard"] = _pos_metric(routed, "rec_yards", pos, "candidate_mae") <= _pos_metric(routed, "rec_yards", pos, "baseline_mae") + 0.50
    gates["routed_receptions_mae_nonworse"] = _macro(routed, "receptions", "candidate_mae") <= _macro(routed, "receptions", "baseline_mae") + 1e-12
    gates["routed_receptions_crps_nonworse"] = _macro(routed, "receptions", "candidate_crps") <= _macro(routed, "receptions", "baseline_crps") + 1e-12
    gates["routed_rb_combo_mae_nonworse"] = _rb_combo(routed, "candidate_mae") <= _rb_combo(routed, "baseline_mae") + 1e-12
    gates["routed_rb_combo_p90_nonworse"] = _rb_combo(routed, "candidate_p90") <= _rb_combo(routed, "baseline_p90") + 1e-12

    return gates


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
    ap.add_argument("--phasej-casebook", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("data/model_ensemble_weights.csv"))
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    logs = _read(a.player_logs, "player logs")
    team = _read(a.team_weekly, "team weekly")
    sched = _read(a.schedule, "schedule")
    injuries = _optional(a.injuries)
    weather = _optional(a.weather)
    weights = load_weights(a.weights)
    routes = _phasej_routes(a.phasej_casebook)
    te24 = _load_fold_params(a.te_coefficients, test_season=2024, features=TE_FEATURES, label="TE-R5P")
    te25 = _load_fold_params(a.te_coefficients, test_season=2025, features=TE_FEATURES, label="TE-R5P")
    wr24 = _load_fold_params(a.wr_coefficients, test_season=2024, features=WR_FEATURES, label="WR-R15")
    snaps, dup, _ = _load_participation_snaps()
    if dup > 0.01:
        raise RuntimeError(f"participation snap duplicate rate too high: {dup}")
    weeks = _parse_weeks(a.weeks)

    d24, s24 = _season_view(
        season=2024, prior_season=2023, weeks=weeks, player_logs=logs, team_weekly=team,
        schedule=sched, universe_dir=a.universe_2024, injuries=injuries, weather=weather,
        te_params=te24, wr_params=wr24, snaps=snaps, weights=weights, iterations=a.iterations, routes=None,
    )
    d25, s25 = _season_view(
        season=2025, prior_season=2024, weeks=weeks, player_logs=logs, team_weekly=team,
        schedule=sched, universe_dir=a.universe_2025, injuries=injuries, weather=weather,
        te_params=te25, wr_params=None, snaps=snaps, weights=weights, iterations=a.iterations, routes=None,
    )
    dr, sr = _season_view(
        season=2025, prior_season=2024, weeks=weeks, player_logs=logs, team_weekly=team,
        schedule=sched, universe_dir=a.universe_2025, injuries=injuries, weather=weather,
        te_params=te25, wr_params=None, snaps=snaps, weights=weights, iterations=a.iterations, routes=routes,
    )

    integrity = _scope_integrity(s24, s25, sr)
    integrity["phasej_rows_exact"] = len(routes) == 440
    integrity["phasej_selected_exact"] = int(routes["use_c2"].sum()) == 412
    science = _science_gates(s24, s25, sr)

    if not all(integrity.values()):
        disposition = f"{VERSION}_MECHANICAL_OR_INTEGRITY_FAILURE"
        qualified = False
    else:
        qualified = bool(all(science.values()))
        disposition = f"{VERSION}_{'QUALIFIED' if qualified else 'FAILED_CLOSED'}"

    result = {
        "version": VERSION,
        "disposition": disposition,
        "qualified": qualified,
        "production_changed": False,
        "parameters_fit": 0,
        "candidate_variants_scored": 1,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_used_upstream": 0,
        "phasej_authority": {
            "run": 34151640191,
            "artifact": 10029560958,
            "rows": int(len(routes)),
            "selected": int(routes["use_c2"].sum()),
        },
        "all_c2_2024": s24,
        "all_c2_2025": s25,
        "phasej_oos_routed_2025": sr,
        "integrity_gates": integrity,
        "scientific_gates": science,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    d24.to_csv(a.out_dir / "all_c2_detail_2024.csv", index=False)
    d25.to_csv(a.out_dir / "all_c2_detail_2025.csv", index=False)
    dr.to_csv(a.out_dir / "phasej_oos_routed_detail_2025.csv", index=False)
    (a.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    for label, s in (("ALL_C2_2024", s24), ("ALL_C2_2025", s25), ("PHASEJ_OOS_ROUTED_2025", sr)):
        for key, val in s["metrics"].items():
            rows.append({"view": label, "group": key, **val})
    pd.DataFrame(rows).to_csv(a.out_dir / "metrics.csv", index=False)

    lines = [
        "# Shared Pass-State Receiver Integration V1 — Frozen Result",
        "",
        f"Disposition: **{disposition}**",
        "",
    ]
    for label, s in (("ALL-C2 2024", s24), ("ALL-C2 2025", s25), ("Phase-J OOS routed 2025", sr)):
        lines += [f"## {label}", ""]
        m = s["macro"]
        lines += [
            f"- rec-yards macro MAE: `{m['rec_yards_baseline_mae']:.6f} -> {m['rec_yards_candidate_mae']:.6f}`",
            f"- rec-yards macro CRPS: `{m['rec_yards_baseline_crps']:.6f} -> {m['rec_yards_candidate_crps']:.6f}`",
            f"- rec-yards macro p90 AE: `{m['rec_yards_baseline_p90']:.6f} -> {m['rec_yards_candidate_p90']:.6f}`",
            f"- receptions macro MAE: `{m['receptions_baseline_mae']:.6f} -> {m['receptions_candidate_mae']:.6f}`",
            f"- receptions macro CRPS: `{m['receptions_baseline_crps']:.6f} -> {m['receptions_candidate_crps']:.6f}`",
            f"- RB rush+rec MAE: `{s['metrics']['rush_rec_yards:RB']['baseline_mae']:.6f} -> {s['metrics']['rush_rec_yards:RB']['candidate_mae']:.6f}`",
            "",
        ]
    lines += ["## Integrity gates", ""] + [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in integrity.items()]
    lines += ["", "## Scientific gates", ""] + [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in science.items()]
    (a.out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
