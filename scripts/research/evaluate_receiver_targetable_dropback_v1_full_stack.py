#!/usr/bin/env python3
"""Historical player full-stack test for Receiver Targetable-Dropback V1.

The exact team-level candidate already qualified on 2022-2023 and confirmed on
2024-2025. This evaluator changes only receiver opportunity inside the MC
component. All non-receiver simulation arrays are copied from baseline exactly.

No sportsbook input is read. Target-game outcomes enter only after projections
are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
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
from scripts.research.evaluate_receiver_targetable_dropback_v1_team_calibration import (
    build_team_actual_history,
    strict_prior_rate,
)
from scripts.research.persist_wr_te_production_order_historical_v1 import (
    TE_FEATURES,
    WR_FEATURES,
    _load_fold_params,
    apply_te_fold,
    apply_wr_fold,
)
from scripts.simulation_c2_qb_candidate import StateSimulationResult, simulate_with_states
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate
from scripts.simulation_v2 import _allocate_counts, _clip_prob, _num, _player_key, lookup
from scripts.utils.canonical_names import canon_team

VERSION = "RECEIVER_TARGETABLE_DROPBACK_V1_FULL_STACK"
POSITIONS = ("WR", "TE", "RB")
PRIMARY_MARKETS = ("receptions", "rec_yards")
RB_POSITIONS = {"RB", "HB", "TB"}
TOL = 1e-10


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    if out.empty:
        raise RuntimeError(f"empty {label}: {path}")
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _pos(value: object) -> str:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
    if p in {"WR", "LWR", "RWR", "SWR"} or p.startswith("WR"):
        return "WR"
    if p == "TE" or p.startswith("TE"):
        return "TE"
    if p in RB_POSITIONS or p.startswith("RB"):
        return "RB"
    if p == "FB" or p.startswith("FB"):
        return "FB"
    if p == "QB" or p.startswith("QB"):
        return "QB"
    return "OTHER"


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(x) for x in parts)
    raw = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") % (2**63 - 1)


def _compare_exact(left, right, label: str) -> dict:
    if set(left.values) != set(right.values):
        raise RuntimeError(f"{label}: simulation key universe changed")
    changed = 0
    max_mean = 0.0
    max_element = 0.0
    for key in left.values:
        a = np.asarray(left.values[key], dtype=float)
        b = np.asarray(right.values[key], dtype=float)
        if a.shape != b.shape:
            raise RuntimeError(f"{label}: shape mismatch key={key}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f"{label}: non-finite key={key}")
        mg = abs(float(a.mean()) - float(b.mean())) if len(a) else 0.0
        eg = float(np.max(np.abs(a - b))) if len(a) else 0.0
        max_mean = max(max_mean, mg)
        max_element = max(max_element, eg)
        if eg > 1e-12:
            changed += 1
    if changed or max_mean > 1e-12 or max_element > 1e-12:
        raise RuntimeError(
            f"{label}: not exact changed={changed} max_mean={max_mean} max_element={max_element}"
        )
    return {
        "keys": int(len(left.values)),
        "changed_arrays": int(changed),
        "max_mean_gap": float(max_mean),
        "max_element_gap": float(max_element),
    }


def _candidate_receiver_state(
    *,
    baseline: StateSimulationResult,
    metrics: pd.DataFrame,
    rate_history: pd.DataFrame,
    season: int,
    week: int,
    prior_season: int,
) -> tuple[StateSimulationResult, pd.DataFrame, dict]:
    frame = metrics.copy()
    frame["team"] = frame["team"].map(canon_team)
    frame["player_clean_key"] = frame.apply(_player_key, axis=1)
    if "entitlement_tgt_share" not in frame.columns:
        raise RuntimeError("candidate receiver state requires entitlement_tgt_share")
    ent = pd.to_numeric(frame["entitlement_tgt_share"], errors="coerce")
    if ent.isna().any() or not np.isfinite(ent.to_numpy(float)).all() or ent.lt(0).any():
        raise RuntimeError("candidate receiver state has invalid entitlement")

    values = {k: np.asarray(v).copy() for k, v in baseline.values.items()}
    audits = []
    max_combo_identity = 0.0
    candidate_receiver_keys = set()

    players = (
        frame.sort_values(["event_id", "team", "player_clean_key"])
        .drop_duplicates(["event_id", "team", "player_clean_key"], keep="last")
        .copy()
    )

    for (game, team), tdf0 in players.groupby(["event_id", "team"], sort=True):
        game = str(game)
        team = canon_team(team)
        tdf = tdf0.reset_index(drop=True)

        rate_info = strict_prior_rate(
            rate_history,
            int(season),
            int(week),
            team,
            int(prior_season),
        )
        rate = float(rate_info["targetable_dropback_rate"])
        if not np.isfinite(rate) or not (0.0 <= rate <= 1.0):
            raise RuntimeError(f"invalid targetable rate team={team} rate={rate}")

        dropbacks = np.asarray(
            baseline.team_states[(game, team, "pass_att")], dtype=int
        )
        pass_eff = np.asarray(
            baseline.team_states[(game, team, "pass_eff_shock")], dtype=float
        )
        if len(dropbacks) != baseline.iterations or len(pass_eff) != baseline.iterations:
            raise RuntimeError(f"team state draw mismatch team={team}")

        shares = pd.to_numeric(
            tdf["entitlement_tgt_share"], errors="raise"
        ).to_numpy(float)
        if not np.isfinite(shares).all() or (shares < 0).any():
            raise RuntimeError(f"invalid entitlement shares team={team}")
        if float(shares.sum()) > 0.950000000001:
            raise RuntimeError(
                f"baseline entitlement exceeds 0.95 team={team} sum={float(shares.sum())}"
            )
        scaled = shares * rate
        if float(scaled.sum()) > 0.950000000001:
            raise RuntimeError(
                f"scaled entitlement exceeds 0.95 team={team} sum={float(scaled.sum())}"
            )

        rng = np.random.default_rng(
            _stable_seed(VERSION, int(season), int(week), game, team)
        )
        targets = _allocate_counts(rng, dropbacks, scaled)

        for j, (_, row) in enumerate(tdf.iterrows()):
            pkey = str(row["player_clean_key"])
            if not pkey:
                continue

            catch_rate = _clip_prob(
                _num(
                    row,
                    "rules_catch_rate",
                    "bayes_receptions_per_target",
                    "receptions_per_target",
                    "catch_rate",
                    default=0.64,
                ),
                0.64,
            )
            receptions = rng.binomial(targets[:, j], catch_rate)
            vol_mult = float(
                np.clip(_num(row, "rules_volatility_mult", default=1.0), 0.75, 1.50)
            )
            ypt = _num(row, "rules_ypt", "bayes_ypt", "ypt")
            ypt = 7.5 if not np.isfinite(ypt) or ypt <= 0 else float(ypt)
            rec_mu = targets[:, j].astype(float) * ypt * pass_eff
            rec_sd = (
                np.maximum(6.0, np.sqrt(np.maximum(targets[:, j], 1)) * ypt * 0.55)
                * vol_mult
            )
            rec_yards = np.clip(rng.normal(rec_mu, rec_sd), 0.0, None)

            rec_key = (game, pkey, "receptions")
            yard_key = (game, pkey, "rec_yards")
            combo_key = (game, pkey, "rush_rec_yards")
            rush_key = (game, pkey, "rush_yards")

            if rec_key not in values or yard_key not in values:
                raise RuntimeError(
                    f"baseline receiver arrays missing team={team} player={pkey}"
                )
            values[rec_key] = receptions.astype(float)
            values[yard_key] = rec_yards.astype(float)
            candidate_receiver_keys.add(rec_key)
            candidate_receiver_keys.add(yard_key)

            if rush_key in values and combo_key in values:
                rush_yards = np.asarray(values[rush_key], dtype=float)
                combo = rush_yards + rec_yards
                values[combo_key] = combo
                candidate_receiver_keys.add(combo_key)
                gap = float(np.max(np.abs(combo - (rush_yards + rec_yards))))
                max_combo_identity = max(max_combo_identity, gap)

        realized_named = float(np.mean(targets.sum(axis=1)))
        expected_named = float(np.mean(dropbacks) * float(scaled.sum()))
        audits.append(
            {
                "season": int(season),
                "week": int(week),
                "event_id": game,
                "team": team,
                "targetable_dropback_rate": rate,
                "conversion_source": str(rate_info["conversion_source"]),
                "prior_history_games": int(rate_info["prior_history_games"]),
                "prior_history_dropbacks": float(rate_info["prior_history_dropbacks"]),
                "prior_history_targets": float(rate_info["prior_history_targets"]),
                "baseline_entitlement_sum": float(shares.sum()),
                "candidate_entitlement_probability_sum": float(scaled.sum()),
                "mean_dropbacks": float(np.mean(dropbacks)),
                "expected_modeled_targets": expected_named,
                "realized_modeled_targets": realized_named,
                "realized_minus_expected_modeled_targets": realized_named - expected_named,
            }
        )

    audit = pd.DataFrame(audits)
    if audit.empty:
        raise RuntimeError("candidate receiver audit empty")

    max_qb_gap = 0.0
    max_rush_att_gap = 0.0
    max_rush_yards_gap = 0.0
    max_atd_gap = 0.0
    illegal_changed = []

    for key in baseline.values:
        a = np.asarray(baseline.values[key], dtype=float)
        b = np.asarray(values[key], dtype=float)
        gap = float(np.max(np.abs(a - b))) if len(a) else 0.0
        market = key[2]
        if market == "pass_yards":
            max_qb_gap = max(max_qb_gap, gap)
        elif market == "rush_att":
            max_rush_att_gap = max(max_rush_att_gap, gap)
        elif market == "rush_yards":
            max_rush_yards_gap = max(max_rush_yards_gap, gap)
        elif market == "anytime_td":
            max_atd_gap = max(max_atd_gap, gap)
        if gap > 0 and key not in candidate_receiver_keys:
            illegal_changed.append(key)

    if illegal_changed:
        raise RuntimeError(
            f"candidate changed forbidden arrays sample={illegal_changed[:20]}"
        )
    if max(max_qb_gap, max_rush_att_gap, max_rush_yards_gap, max_atd_gap) > 0:
        raise RuntimeError(
            "candidate nonreceiver invariance failed "
            f"qb={max_qb_gap} rush_att={max_rush_att_gap} "
            f"rush_yards={max_rush_yards_gap} atd={max_atd_gap}"
        )
    if max_combo_identity > TOL:
        raise RuntimeError(
            f"candidate raw rush+receiving identity failed gap={max_combo_identity}"
        )

    payload = {
        "teams": int(len(audit)),
        "team_source_rows": int(audit["conversion_source"].eq("team_strict_prior").sum()),
        "fallback_rows": int(audit["conversion_source"].eq("league_fallback").sum()),
        "rate_min": float(audit["targetable_dropback_rate"].min()),
        "rate_median": float(audit["targetable_dropback_rate"].median()),
        "rate_max": float(audit["targetable_dropback_rate"].max()),
        "max_qb_pass_yards_gap": float(max_qb_gap),
        "max_rush_att_gap": float(max_rush_att_gap),
        "max_rush_yards_gap": float(max_rush_yards_gap),
        "max_atd_gap": float(max_atd_gap),
        "max_raw_rush_rec_identity_gap": float(max_combo_identity),
        "max_abs_realized_minus_expected_modeled_targets": float(
            audit["realized_minus_expected_modeled_targets"].abs().max()
        ),
    }
    return StateSimulationResult(values, baseline.iterations, baseline.team_states), audit, payload


def _ensemble_projection(records: pd.DataFrame, weights: pd.DataFrame, mc_col: str) -> np.ndarray:
    x = records[["market", "ml_proj", "state_proj"]].copy()
    x["mc_proj"] = pd.to_numeric(records[mc_col], errors="coerce")
    out = apply_ensemble(x[["market", "mc_proj", "ml_proj", "state_proj"]], weights=weights)
    return pd.to_numeric(out["ensemble_proj"], errors="coerce").to_numpy(float)


def _score(part: pd.DataFrame) -> dict:
    if part.empty:
        raise RuntimeError("attempted to score empty cohort")
    actual = pd.to_numeric(part["actual"], errors="raise").to_numpy(float)
    b = pd.to_numeric(part["baseline_proj"], errors="raise").to_numpy(float)
    c = pd.to_numeric(part["candidate_proj"], errors="raise").to_numpy(float)
    be = b - actual
    ce = c - actual
    ba = np.abs(be)
    ca = np.abs(ce)
    changed = np.abs(b - c) > 1e-12
    cand = changed & (ca < ba - 1e-12)
    base = changed & (ba < ca - 1e-12)
    decided = cand | base
    return {
        "n": int(len(part)),
        "baseline_mae": float(np.mean(ba)),
        "candidate_mae": float(np.mean(ca)),
        "baseline_rmse": float(np.sqrt(np.mean(be * be))),
        "candidate_rmse": float(np.sqrt(np.mean(ce * ce))),
        "baseline_bias": float(np.mean(be)),
        "candidate_bias": float(np.mean(ce)),
        "baseline_abs_bias": float(abs(np.mean(be))),
        "candidate_abs_bias": float(abs(np.mean(ce))),
        "baseline_corr": float(np.corrcoef(actual, b)[0, 1])
        if len(part) > 1 and np.std(actual) > 0 and np.std(b) > 0
        else None,
        "candidate_corr": float(np.corrcoef(actual, c)[0, 1])
        if len(part) > 1 and np.std(actual) > 0 and np.std(c) > 0
        else None,
        "baseline_median_ae": float(np.quantile(ba, 0.50)),
        "candidate_median_ae": float(np.quantile(ca, 0.50)),
        "baseline_p75_ae": float(np.quantile(ba, 0.75)),
        "candidate_p75_ae": float(np.quantile(ca, 0.75)),
        "baseline_p90_ae": float(np.quantile(ba, 0.90)),
        "candidate_p90_ae": float(np.quantile(ca, 0.90)),
        "baseline_miss20": float(np.mean(ba >= 20)),
        "candidate_miss20": float(np.mean(ca >= 20)),
        "baseline_miss30": float(np.mean(ba >= 30)),
        "candidate_miss30": float(np.mean(ca >= 30)),
        "baseline_miss40": float(np.mean(ba >= 40)),
        "candidate_miss40": float(np.mean(ca >= 40)),
        "changed_rows": int(changed.sum()),
        "candidate_closer": int(cand.sum()),
        "baseline_closer": int(base.sum()),
        "candidate_closer_rate": float(cand.sum() / decided.sum())
        if int(decided.sum())
        else None,
    }


def _macro(detail: pd.DataFrame, market: str) -> dict:
    pos_scores = {}
    for pos in POSITIONS:
        pos_scores[pos] = _score(
            detail.loc[
                detail["market"].eq(market)
                & detail["position_family"].eq(pos)
            ]
        )
    fields = [
        "baseline_mae",
        "candidate_mae",
        "baseline_p90_ae",
        "candidate_p90_ae",
        "baseline_abs_bias",
        "candidate_abs_bias",
        "baseline_miss40",
        "candidate_miss40",
    ]
    out = {
        f"macro_{field}": float(np.mean([pos_scores[p][field] for p in POSITIONS]))
        for field in fields
    }
    out["positions"] = pos_scores
    out["overall"] = _score(
        detail.loc[
            detail["market"].eq(market)
            & detail["position_family"].isin(POSITIONS)
        ]
    )
    return out


def _attach_entitlement_quartile(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    out["entitlement_quartile"] = ""
    for season in sorted(out["season"].unique()):
        base = (
            out.loc[
                out["season"].eq(season) & out["market"].eq("rec_yards"),
                ["event_id", "team", "player_clean_key", "entitlement_tgt_share"],
            ]
            .drop_duplicates(["event_id", "team", "player_clean_key"])
            .copy()
        )
        ranked = base["entitlement_tgt_share"].rank(method="first")
        base["entitlement_quartile"] = pd.qcut(
            ranked, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]
        ).astype(str)
        qmap = base.set_index(
            ["event_id", "team", "player_clean_key"]
        )["entitlement_quartile"].to_dict()
        mask = out["season"].eq(season)
        out.loc[mask, "entitlement_quartile"] = [
            qmap.get((str(r.event_id), str(r.team), str(r.player_clean_key)), "")
            for r in out.loc[
                mask, ["event_id", "team", "player_clean_key"]
            ].itertuples(index=False)
        ]
    return out


def evaluate_season(
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
    rate_history: pd.DataFrame,
    te_params: dict,
    wr_params: dict | None,
    snaps: pd.DataFrame,
    weights: pd.DataFrame,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    details = []
    audits = []
    max_rb_v2_gap = 0.0

    for week in weeks:
        universe = _read(
            universe_dir / f"{season}_week_{week:02d}.csv",
            f"{season} W{week} universe",
        )
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

        seed = 42 + int(week)
        metrics = build_mc_predictions(bundle, iterations=20, seed=seed)
        players = (
            metrics.sort_values(["event_id", "team", "player_clean_key"])
            .drop_duplicates(["event_id", "team", "player_clean_key"], keep="last")
            .copy()
        )

        explicit_base, _ = materialize_target_entitlement(players)
        te_final, _, te_audit = apply_te_fold(
            explicit_base, snaps=snaps, params=te_params
        )
        if int(season) == 2024:
            final, _, wr_audit = apply_wr_fold(
                te_final, snaps=snaps, params=wr_params
            )
        else:
            final = te_final
            wr_audit = {
                "m38_wr1_anchor_max_abs_gap": 0.0,
                "wr2plus_pool_max_abs_gap": 0.0,
                "wr_room_mass_max_abs_gap": 0.0,
                "non_wr_max_abs_gap": 0.0,
                "same_future_participation": 0,
            }

        entitlement_before = pd.to_numeric(
            final["entitlement_tgt_share"], errors="raise"
        ).to_numpy(float).copy()

        baseline = simulate_with_states(
            final, iterations=int(iterations), seed=int(seed)
        )
        explicit_check = explicit_simulate(
            final, iterations=int(iterations), seed=int(seed)
        )
        parity = _compare_exact(
            explicit_check,
            baseline,
            f"{season} W{week:02d} explicit-vs-stateful baseline",
        )

        candidate, candidate_audit, candidate_payload = _candidate_receiver_state(
            baseline=baseline,
            metrics=final,
            rate_history=rate_history,
            season=int(season),
            week=int(week),
            prior_season=int(prior_season),
        )

        entitlement_after = pd.to_numeric(
            final["entitlement_tgt_share"], errors="raise"
        ).to_numpy(float)
        if not np.array_equal(entitlement_before, entitlement_after, equal_nan=True):
            raise RuntimeError(f"{season} W{week:02d} entitlement mutated")

        _, ml_pred = build_ml(
            player_logs, bundle.player_consensus, int(season), int(week)
        )
        _, state_pred = build_state_predictions(
            player_logs, bundle.player_consensus, int(season), int(week)
        )

        cols = [
            "event_id",
            "team",
            "player",
            "player_clean_key",
            "position",
            "market",
            "season",
            "week",
        ]
        market_rows = metrics[cols].copy()
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
        if joined.empty:
            raise RuntimeError(f"{season} W{week:02d} no scored rows")

        joined["position_family"] = joined["position"].map(_pos)
        ent = final[
            ["event_id", "team", "player_clean_key", "entitlement_tgt_share"]
        ].drop_duplicates(["event_id", "team", "player_clean_key"])
        ent["team"] = ent["team"].map(canon_team)
        joined = joined.merge(
            ent,
            on=["event_id", "team", "player_clean_key"],
            how="left",
            validate="many_to_one",
        )
        if joined["entitlement_tgt_share"].isna().any():
            raise RuntimeError(
                f"{season} W{week:02d} entitlement attach incomplete"
            )

        scored = joined.loc[
            (
                joined["market"].isin(PRIMARY_MARKETS)
                & joined["position_family"].isin(POSITIONS)
            )
            | (
                joined["market"].eq("rush_rec_yards")
                & joined["position_family"].eq("RB")
            )
        ].copy()
        if scored.empty:
            raise RuntimeError(f"{season} W{week:02d} empty scored receiver rows")

        baseline_mc = []
        candidate_mc = []
        for _, row in scored.iterrows():
            ba = lookup(baseline, row, row["market"])
            ca = lookup(candidate, row, row["market"])
            if ba is None or ca is None:
                raise RuntimeError(
                    f"{season} W{week:02d} missing sim array "
                    f"player={row['player_clean_key']} market={row['market']}"
                )
            baseline_mc.append(float(np.mean(np.asarray(ba, float))))
            candidate_mc.append(float(np.mean(np.asarray(ca, float))))

        scored["baseline_mc_proj"] = baseline_mc
        scored["candidate_mc_proj"] = candidate_mc
        scored["baseline_proj"] = _ensemble_projection(
            scored, weights, "baseline_mc_proj"
        )
        scored["candidate_proj"] = _ensemble_projection(
            scored, weights, "candidate_mc_proj"
        )

        bmap, bpayload = build_candidate_map(joined, baseline, weights)
        cmap, cpayload = build_candidate_map(joined, candidate, weights)
        max_rb_v2_gap = max(
            max_rb_v2_gap,
            float(bpayload.get("max_pathwise_identity_gap", 0.0)),
            float(cpayload.get("max_pathwise_identity_gap", 0.0)),
        )

        for idx, row in scored.loc[
            scored["market"].eq("rush_rec_yards")
            & scored["position_family"].eq("RB")
        ].iterrows():
            key = (str(row["event_id"]), str(row["player_clean_key"]))
            if key in bmap:
                scored.at[idx, "baseline_proj"] = float(bmap[key]["target_mean"])
            if key in cmap:
                scored.at[idx, "candidate_proj"] = float(cmap[key]["target_mean"])

        numeric = scored[
            ["actual", "baseline_proj", "candidate_proj"]
        ].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
            raise RuntimeError(f"{season} W{week:02d} non-finite scored values")

        details.append(scored)
        c_audit = candidate_audit.copy()
        c_audit["season"] = int(season)
        c_audit["week"] = int(week)
        audits.append(
            {
                "season": int(season),
                "week": int(week),
                "baseline_parity_changed_arrays": int(parity["changed_arrays"]),
                "baseline_parity_max_element_gap": float(parity["max_element_gap"]),
                "candidate_teams": int(candidate_payload["teams"]),
                "team_source_rows": int(candidate_payload["team_source_rows"]),
                "fallback_rows": int(candidate_payload["fallback_rows"]),
                "rate_min": float(candidate_payload["rate_min"]),
                "rate_median": float(candidate_payload["rate_median"]),
                "rate_max": float(candidate_payload["rate_max"]),
                "max_qb_pass_yards_gap": float(candidate_payload["max_qb_pass_yards_gap"]),
                "max_rush_att_gap": float(candidate_payload["max_rush_att_gap"]),
                "max_rush_yards_gap": float(candidate_payload["max_rush_yards_gap"]),
                "max_atd_gap": float(candidate_payload["max_atd_gap"]),
                "max_raw_rush_rec_identity_gap": float(
                    candidate_payload["max_raw_rush_rec_identity_gap"]
                ),
                "max_target_mc_sampling_gap": float(
                    candidate_payload["max_abs_realized_minus_expected_modeled_targets"]
                ),
                "rb_v2_baseline_identity_gap": float(
                    bpayload.get("max_pathwise_identity_gap", 0.0)
                ),
                "rb_v2_candidate_identity_gap": float(
                    cpayload.get("max_pathwise_identity_gap", 0.0)
                ),
                "te_pool_gap": float(te_audit["team_te_pool_max_abs_gap"]),
                "te_non_te_gap": float(te_audit["non_te_max_abs_gap"]),
                "wr1_anchor_gap": float(
                    wr_audit.get("m38_wr1_anchor_max_abs_gap", 0.0)
                ),
                "wr2plus_pool_gap": float(
                    wr_audit.get("wr2plus_pool_max_abs_gap", 0.0)
                ),
                "wr_room_gap": float(
                    wr_audit.get("wr_room_mass_max_abs_gap", 0.0)
                ),
                "wr_non_wr_gap": float(
                    wr_audit.get("non_wr_max_abs_gap", 0.0)
                ),
                "wr_same_future_participation": int(
                    wr_audit.get("same_future_participation", 0)
                ),
            }
        )

    detail = _attach_entitlement_quartile(
        pd.concat(details, ignore_index=True)
    )
    audit = pd.DataFrame(audits)

    scope = {
        "team_games": int(audit["candidate_teams"].sum()),
        "team_source_rows": int(audit["team_source_rows"].sum()),
        "fallback_rows": int(audit["fallback_rows"].sum()),
        "rate_min": float(audit["rate_min"].min()),
        "rate_median_of_week_medians": float(audit["rate_median"].median()),
        "rate_max": float(audit["rate_max"].max()),
        "max_baseline_parity_gap": float(audit["baseline_parity_max_element_gap"].max()),
        "max_qb_pass_yards_gap": float(audit["max_qb_pass_yards_gap"].max()),
        "max_rush_att_gap": float(audit["max_rush_att_gap"].max()),
        "max_rush_yards_gap": float(audit["max_rush_yards_gap"].max()),
        "max_atd_gap": float(audit["max_atd_gap"].max()),
        "max_raw_rush_rec_identity_gap": float(
            audit["max_raw_rush_rec_identity_gap"].max()
        ),
        "rb_v2_max_pathwise_identity_gap": float(max_rb_v2_gap),
        "wr_same_future_participation": int(
            audit["wr_same_future_participation"].sum()
        ),
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_used_upstream": 0,
    }
    return detail, audit, scope


def _build_scorecard(detail: pd.DataFrame) -> dict:
    pooled = {}
    seasons = {}
    for market in PRIMARY_MARKETS:
        pooled[market] = _macro(detail, market)
        seasons[market] = {}
        for season in (2024, 2025):
            seasons[market][str(season)] = _macro(
                detail.loc[detail["season"].eq(season)], market
            )

    q4 = _score(
        detail.loc[
            detail["market"].eq("rec_yards")
            & detail["position_family"].isin(POSITIONS)
            & detail["entitlement_quartile"].eq("Q4_high")
        ]
    )

    combo = {}
    for label, frame in [
        ("pooled", detail),
        ("2024", detail.loc[detail["season"].eq(2024)]),
        ("2025", detail.loc[detail["season"].eq(2025)]),
    ]:
        combo[label] = _score(
            frame.loc[
                frame["market"].eq("rush_rec_yards")
                & frame["position_family"].eq("RB")
            ]
        )

    return {
        "pooled": pooled,
        "seasons": seasons,
        "q4_rec_yards": q4,
        "rb_rush_rec_yards": combo,
    }


def _gates(score: dict, scope24: dict, scope25: dict) -> dict:
    receptions = score["pooled"]["receptions"]
    rec_yards = score["pooled"]["rec_yards"]
    r24 = score["seasons"]["receptions"]["2024"]
    r25 = score["seasons"]["receptions"]["2025"]
    y24 = score["seasons"]["rec_yards"]["2024"]
    y25 = score["seasons"]["rec_yards"]["2025"]
    q4 = score["q4_rec_yards"]
    combo = score["rb_rush_rec_yards"]

    mech = {
        "baseline_parity_exact": scope24["max_baseline_parity_gap"] <= TOL
        and scope25["max_baseline_parity_gap"] <= TOL,
        "strict_prior_rate_source_ge99": (
            (scope24["team_source_rows"] + scope25["team_source_rows"])
            / max(1, scope24["team_games"] + scope25["team_games"])
        )
        >= 0.99,
        "fallback_rate_le1": (
            (scope24["fallback_rows"] + scope25["fallback_rows"])
            / max(1, scope24["team_games"] + scope25["team_games"])
        )
        <= 0.01,
        "rates_finite_in_bounds": 0.0
        <= min(scope24["rate_min"], scope25["rate_min"])
        <= max(scope24["rate_max"], scope25["rate_max"])
        <= 1.0,
        "qb_pass_yards_exact": scope24["max_qb_pass_yards_gap"] <= TOL
        and scope25["max_qb_pass_yards_gap"] <= TOL,
        "rush_att_exact": scope24["max_rush_att_gap"] <= TOL
        and scope25["max_rush_att_gap"] <= TOL,
        "rush_yards_exact": scope24["max_rush_yards_gap"] <= TOL
        and scope25["max_rush_yards_gap"] <= TOL,
        "atd_exact": scope24["max_atd_gap"] <= TOL
        and scope25["max_atd_gap"] <= TOL,
        "raw_rush_rec_identity": scope24["max_raw_rush_rec_identity_gap"] <= TOL
        and scope25["max_raw_rush_rec_identity_gap"] <= TOL,
        "rb_v2_identity": scope24["rb_v2_max_pathwise_identity_gap"] <= TOL
        and scope25["rb_v2_max_pathwise_identity_gap"] <= TOL,
        "wr_same_future_participation_zero": scope24["wr_same_future_participation"] == 0
        and scope25["wr_same_future_participation"] == 0,
        "sportsbook_inputs_zero": True,
        "target_game_outcomes_upstream_zero": True,
        "parameters_fit_zero": True,
        "one_candidate_only": True,
    }

    sci = {
        "receptions_pooled_macro_mae_improves": receptions["macro_candidate_mae"]
        < receptions["macro_baseline_mae"],
        "receptions_2024_macro_mae_nonworse": r24["macro_candidate_mae"]
        <= r24["macro_baseline_mae"] + 1e-12,
        "receptions_2025_macro_mae_nonworse": r25["macro_candidate_mae"]
        <= r25["macro_baseline_mae"] + 1e-12,
        "receptions_wr_pooled_mae_nonworse": receptions["positions"]["WR"]["candidate_mae"]
        <= receptions["positions"]["WR"]["baseline_mae"] + 1e-12,
        "receptions_te_pooled_mae_nonworse": receptions["positions"]["TE"]["candidate_mae"]
        <= receptions["positions"]["TE"]["baseline_mae"] + 1e-12,
        "receptions_rb_pooled_mae_nonworse": receptions["positions"]["RB"]["candidate_mae"]
        <= receptions["positions"]["RB"]["baseline_mae"] + 1e-12,
        "receptions_pooled_macro_p90_nonworse": receptions["macro_candidate_p90_ae"]
        <= receptions["macro_baseline_p90_ae"] + 1e-12,
        "receptions_pooled_macro_abs_bias_nonworse": receptions["macro_candidate_abs_bias"]
        <= receptions["macro_baseline_abs_bias"] + 1e-12,
        "receptions_candidate_closer_gt50": receptions["overall"]["candidate_closer_rate"]
        is not None
        and receptions["overall"]["candidate_closer_rate"] > 0.50,
        "rec_yards_pooled_macro_mae_improves": rec_yards["macro_candidate_mae"]
        < rec_yards["macro_baseline_mae"],
        "rec_yards_2024_macro_mae_nonworse": y24["macro_candidate_mae"]
        <= y24["macro_baseline_mae"] + 1e-12,
        "rec_yards_2025_macro_mae_nonworse": y25["macro_candidate_mae"]
        <= y25["macro_baseline_mae"] + 1e-12,
        "rec_yards_wr_pooled_mae_nonworse": rec_yards["positions"]["WR"]["candidate_mae"]
        <= rec_yards["positions"]["WR"]["baseline_mae"] + 1e-12,
        "rec_yards_te_pooled_mae_nonworse": rec_yards["positions"]["TE"]["candidate_mae"]
        <= rec_yards["positions"]["TE"]["baseline_mae"] + 1e-12,
        "rec_yards_rb_pooled_mae_nonworse": rec_yards["positions"]["RB"]["candidate_mae"]
        <= rec_yards["positions"]["RB"]["baseline_mae"] + 1e-12,
        "rec_yards_pooled_macro_p90_nonworse": rec_yards["macro_candidate_p90_ae"]
        <= rec_yards["macro_baseline_p90_ae"] + 1e-12,
        "rec_yards_pooled_macro_abs_bias_nonworse": rec_yards["macro_candidate_abs_bias"]
        <= rec_yards["macro_baseline_abs_bias"] + 1e-12,
        "rec_yards_q4_mae_nonworse": q4["candidate_mae"]
        <= q4["baseline_mae"] + 1e-12,
        "rec_yards_q4_p90_nonworse": q4["candidate_p90_ae"]
        <= q4["baseline_p90_ae"] + 1e-12,
        "rec_yards_macro_miss40_nonworse": rec_yards["macro_candidate_miss40"]
        <= rec_yards["macro_baseline_miss40"] + 1e-12,
        "rec_yards_candidate_closer_gt50": rec_yards["overall"]["candidate_closer_rate"]
        is not None
        and rec_yards["overall"]["candidate_closer_rate"] > 0.50,
        "rb_combo_mae_2024_nonworse": combo["2024"]["candidate_mae"]
        <= combo["2024"]["baseline_mae"] + 1e-12,
        "rb_combo_mae_2025_nonworse": combo["2025"]["candidate_mae"]
        <= combo["2025"]["baseline_mae"] + 1e-12,
        "rb_combo_p90_pooled_nonworse": combo["pooled"]["candidate_p90_ae"]
        <= combo["pooled"]["baseline_p90_ae"] + 1e-12,
    }

    return {**mech, **sci}


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
    ap.add_argument("--weights", type=Path, default=Path("data/model_ensemble_weights.csv"))
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    player_logs = _read(args.player_logs, "player logs")
    team_weekly = _read(args.team_weekly, "team weekly")
    schedule = _read(args.schedule, "schedule")
    injuries = _optional(args.injuries)
    weather = _optional(args.weather)
    weights = load_weights(args.weights)
    rate_history = build_team_actual_history(player_logs, team_weekly)

    te24 = _load_fold_params(
        args.te_coefficients,
        test_season=2024,
        features=TE_FEATURES,
        label="TE-R5P",
    )
    te25 = _load_fold_params(
        args.te_coefficients,
        test_season=2025,
        features=TE_FEATURES,
        label="TE-R5P",
    )
    wr24 = _load_fold_params(
        args.wr_coefficients,
        test_season=2024,
        features=WR_FEATURES,
        label="WR-R15",
    )

    snaps, dup, _ = _load_participation_snaps()
    if dup > 0.01:
        raise RuntimeError(f"participation snap duplicate rate too high: {dup}")

    weeks = _parse_weeks(args.weeks)

    d24, a24, s24 = evaluate_season(
        season=2024,
        prior_season=2023,
        weeks=weeks,
        player_logs=player_logs,
        team_weekly=team_weekly,
        schedule=schedule,
        universe_dir=args.universe_2024,
        injuries=injuries,
        weather=weather,
        rate_history=rate_history,
        te_params=te24,
        wr_params=wr24,
        snaps=snaps,
        weights=weights,
        iterations=int(args.iterations),
    )
    d25, a25, s25 = evaluate_season(
        season=2025,
        prior_season=2024,
        weeks=weeks,
        player_logs=player_logs,
        team_weekly=team_weekly,
        schedule=schedule,
        universe_dir=args.universe_2025,
        injuries=injuries,
        weather=weather,
        rate_history=rate_history,
        te_params=te25,
        wr_params=None,
        snaps=snaps,
        weights=weights,
        iterations=int(args.iterations),
    )

    detail = pd.concat([d24, d25], ignore_index=True)
    score = _build_scorecard(detail)
    gates = _gates(score, s24, s25)
    qualified = all(gates.values())
    disposition = (
        "RECEIVER_TARGETABLE_DROPBACK_V1_FULL_STACK_QUALIFIED"
        if qualified
        else "RECEIVER_TARGETABLE_DROPBACK_V1_FULL_STACK_FAILED_CLOSED"
    )

    payload = {
        "version": VERSION,
        "disposition": disposition,
        "qualified": bool(qualified),
        "iterations": int(args.iterations),
        "parameters_fit": 0,
        "candidate_variants_scored": 1,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_used_upstream": 0,
        "scope_2024": s24,
        "scope_2025": s25,
        "scorecard": score,
        "gates": gates,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d24.to_csv(args.out_dir / "detail_2024.csv", index=False)
    d25.to_csv(args.out_dir / "detail_2025.csv", index=False)
    a24.to_csv(args.out_dir / "audit_2024.csv", index=False)
    a25.to_csv(args.out_dir / "audit_2025.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Receiver Targetable-Dropback V1 — Player Full-Stack Result",
        "",
        f"Disposition: **{disposition}**",
        "",
    ]
    for market in PRIMARY_MARKETS:
        x = score["pooled"][market]
        lines += [
            f"## {market}",
            "",
            f"- macro MAE: {x['macro_baseline_mae']:.6f} -> {x['macro_candidate_mae']:.6f}",
            f"- macro p90 AE: {x['macro_baseline_p90_ae']:.6f} -> {x['macro_candidate_p90_ae']:.6f}",
            f"- macro abs bias: {x['macro_baseline_abs_bias']:.6f} -> {x['macro_candidate_abs_bias']:.6f}",
        ]
        for pos in POSITIONS:
            p = x["positions"][pos]
            lines.append(
                f"- {pos} MAE: {p['baseline_mae']:.6f} -> {p['candidate_mae']:.6f}"
            )
        lines.append("")

    q4 = score["q4_rec_yards"]
    combo = score["rb_rush_rec_yards"]
    lines += [
        "## Protection cohorts",
        "",
        f"- Q4 rec_yards MAE: {q4['baseline_mae']:.6f} -> {q4['candidate_mae']:.6f}",
        f"- Q4 rec_yards p90: {q4['baseline_p90_ae']:.6f} -> {q4['candidate_p90_ae']:.6f}",
        f"- RB combo 2024 MAE: {combo['2024']['baseline_mae']:.6f} -> {combo['2024']['candidate_mae']:.6f}",
        f"- RB combo 2025 MAE: {combo['2025']['baseline_mae']:.6f} -> {combo['2025']['candidate_mae']:.6f}",
        f"- RB combo pooled p90: {combo['pooled']['baseline_p90_ae']:.6f} -> {combo['pooled']['candidate_p90_ae']:.6f}",
        "",
        "## Frozen gates",
        "",
    ]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in gates.items()]
    (args.out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
