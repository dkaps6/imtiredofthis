#!/usr/bin/env python3
"""Historical full-stack evaluation for hierarchical receiver mean reconciliation V1.

Frozen candidate:
- exact OOS M89/M90 football_synthesis is the QB/team aggregate authority;
- current fold-safe receiver stack supplies named receiver base means;
- residual absorbs positive aggregate gap first;
- only when named receiver means exceed QB mean are named rec_yards means
  projected downward with the frozen Bayesian uncertainty geometry;
- receptions, QB, rushing and all football inputs remain unchanged;
- current RB Rush+Receiving Conservation V2 semantics protect the dependent
  combo mean for Weeks 2-18.
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
from scripts.research.persist_wr_te_production_order_historical_v1 import (
    TE_FEATURES,
    WR_FEATURES,
    _load_fold_params,
    apply_te_fold,
    apply_wr_fold,
)
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate
from scripts.simulation_v2 import _team_inputs, lookup
from scripts.utils.canonical_names import canon_team

VERSION = "HIERARCHICAL_RECEIVER_MEAN_RECONCILIATION_V1"
TOL = 1e-10
PRIMARY_POSITIONS = ("WR", "TE", "RB")
PASS_CATCHER_POSITIONS = {"WR", "LWR", "RWR", "SWR", "TE", "RB", "HB", "TB", "FB"}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    if out.empty:
        raise RuntimeError(f"empty {label}: {path}")
    return out


def _optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() and path.stat().st_size else pd.DataFrame()


def _pos(value: object) -> str:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
    if p in {"HB", "TB"} or p.startswith("RB"):
        return "RB"
    if p.startswith("FB"):
        return "FB"
    if p.startswith("WR") or p in {"LWR", "RWR", "SWR"}:
        return "WR"
    if p.startswith("TE"):
        return "TE"
    if p.startswith("QB"):
        return "QB"
    return "OTHER"


def _canonicalize_m89(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "team", "opponent", "player_clean_key", "football_synthesis"
    }
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"M89 authority missing columns: {missing}")
    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    x["football_synthesis"] = pd.to_numeric(x["football_synthesis"], errors="raise")
    x = x.loc[x["season"].isin([2024, 2025])].copy()
    counts = x.groupby("season").size().to_dict()
    if counts != {2024: 444, 2025: 440}:
        raise RuntimeError(f"M89 authority row counts drifted: {counts}")
    if x.duplicated(["season", "week", "team"]).any():
        bad = x.loc[x.duplicated(["season", "week", "team"], keep=False),
                    ["season", "week", "team", "player_clean_key"]]
        raise RuntimeError(f"M89 duplicate team-game authority:\n{bad.head(20).to_string(index=False)}")
    if not np.isfinite(x["football_synthesis"].to_numpy(float)).all() or x["football_synthesis"].le(0).any():
        raise RuntimeError("M89 football_synthesis contains invalid values")
    return x


def _weighted_nonnegative_projection(
    base_means: np.ndarray,
    variances: np.ndarray,
    target_sum: float,
) -> np.ndarray:
    b = np.asarray(base_means, dtype=float)
    v = np.asarray(variances, dtype=float)
    target = float(target_sum)
    if not np.isfinite(b).all() or (b < -TOL).any():
        raise RuntimeError("invalid base receiver means")
    if not np.isfinite(v).all() or (v <= 0).any():
        raise RuntimeError("invalid propagated variances")
    if target < -TOL or target > float(b.sum()) + TOL:
        raise RuntimeError(f"invalid downward target={target} base_sum={float(b.sum())}")
    if abs(target - float(b.sum())) <= TOL:
        return b.copy()
    if target <= TOL:
        return np.zeros_like(b)

    x = np.zeros_like(b)
    active = np.ones(len(b), dtype=bool)
    remaining_target = target
    while active.any():
        idx = np.flatnonzero(active)
        ba = b[idx]
        va = v[idx]
        reduction = float(ba.sum() - remaining_target)
        if reduction < -TOL:
            raise RuntimeError("active-set target exceeds active base sum")
        trial = ba - reduction * va / float(va.sum())
        negative = trial < 0
        if not negative.any():
            x[idx] = np.maximum(trial, 0.0)
            break
        bind = idx[negative]
        x[bind] = 0.0
        active[bind] = False
        if not active.any() and remaining_target > TOL:
            raise RuntimeError("active-set exhausted before satisfying target")

    gap = float(x.sum() - target)
    if abs(gap) > TOL:
        raise RuntimeError(f"weighted projection identity failed gap={gap}")
    return np.maximum(x, 0.0)


def _ensemble_rows(rows: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    out = rows.copy()
    ens = apply_ensemble(
        out[["market", "mc_proj", "ml_proj", "state_proj"]].copy(),
        weights=weights,
    )
    for c in (
        "ensemble_proj", "ensemble_status", "ensemble_method",
        "ensemble_weight_mc", "ensemble_weight_ml", "ensemble_weight_state",
        "ensemble_calibration_rows",
    ):
        out[c] = ens[c].to_numpy()
    if pd.to_numeric(out["ensemble_proj"], errors="coerce").isna().any():
        raise RuntimeError("non-finite ensemble receiver/component projection")
    return out


def _sim_mean(sims, row: pd.Series, market: str) -> tuple[float, np.ndarray]:
    arr = lookup(sims, row, market)
    if arr is None:
        raise RuntimeError(
            f"missing simulation array event={row.get('event_id')} player={row.get('player_clean_key')} market={market}"
        )
    a = np.asarray(arr, dtype=float)
    if not np.isfinite(a).all() or (a < 0).any():
        raise RuntimeError("invalid simulation array")
    return float(a.mean()), a


def _score(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise RuntimeError("empty scoring cohort")
    actual = pd.to_numeric(frame["actual"], errors="raise").to_numpy(float)
    b = pd.to_numeric(frame["baseline_proj"], errors="raise").to_numpy(float)
    c = pd.to_numeric(frame["candidate_proj"], errors="raise").to_numpy(float)
    be = b - actual
    ce = c - actual
    ba = np.abs(be)
    ca = np.abs(ce)
    changed = np.abs(b - c) > 1e-12
    cand_closer = changed & (ca < ba - 1e-12)
    base_closer = changed & (ba < ca - 1e-12)
    tie = changed & ~(cand_closer | base_closer)
    decided = cand_closer | base_closer
    return {
        "n": int(len(frame)),
        "baseline_mae": float(ba.mean()),
        "candidate_mae": float(ca.mean()),
        "baseline_rmse": float(np.sqrt(np.mean(be * be))),
        "candidate_rmse": float(np.sqrt(np.mean(ce * ce))),
        "baseline_bias": float(be.mean()),
        "candidate_bias": float(ce.mean()),
        "baseline_abs_bias": float(abs(be.mean())),
        "candidate_abs_bias": float(abs(ce.mean())),
        "baseline_corr": float(np.corrcoef(actual, b)[0, 1]) if len(frame) > 1 and np.std(actual) > 0 and np.std(b) > 0 else None,
        "candidate_corr": float(np.corrcoef(actual, c)[0, 1]) if len(frame) > 1 and np.std(actual) > 0 and np.std(c) > 0 else None,
        "baseline_median_ae": float(np.quantile(ba, .50)),
        "candidate_median_ae": float(np.quantile(ca, .50)),
        "baseline_p75_ae": float(np.quantile(ba, .75)),
        "candidate_p75_ae": float(np.quantile(ca, .75)),
        "baseline_p90_ae": float(np.quantile(ba, .90)),
        "candidate_p90_ae": float(np.quantile(ca, .90)),
        "baseline_miss20_rate": float(np.mean(ba >= 20)),
        "candidate_miss20_rate": float(np.mean(ca >= 20)),
        "baseline_miss30_rate": float(np.mean(ba >= 30)),
        "candidate_miss30_rate": float(np.mean(ca >= 30)),
        "baseline_miss40_rate": float(np.mean(ba >= 40)),
        "candidate_miss40_rate": float(np.mean(ca >= 40)),
        "changed_rows": int(changed.sum()),
        "candidate_closer": int(cand_closer.sum()),
        "baseline_closer": int(base_closer.sum()),
        "changed_ties": int(tie.sum()),
        "candidate_closer_rate": float(cand_closer.sum() / decided.sum()) if int(decided.sum()) else None,
    }


def _macro(detail: pd.DataFrame) -> dict:
    by_pos = {
        p: _score(detail.loc[detail["position_family"].eq(p)])
        for p in PRIMARY_POSITIONS
    }
    fields = [
        "baseline_mae", "candidate_mae",
        "baseline_p90_ae", "candidate_p90_ae",
        "baseline_miss40_rate", "candidate_miss40_rate",
        "baseline_abs_bias", "candidate_abs_bias",
    ]
    out = {
        f"macro_{field}": float(np.mean([by_pos[p][field] for p in PRIMARY_POSITIONS]))
        for field in fields
    }
    out["positions"] = by_pos
    return out


def _prepare_week(
    *,
    season: int,
    week: int,
    prior_season: int,
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
):
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
    players = (
        metrics.sort_values(["event_id", "team", "player_clean_key"])
        .drop_duplicates(["event_id", "team", "player_clean_key"], keep="last")
        .copy()
    )
    explicit_base, _ = materialize_target_entitlement(players)
    te_final, _, te_audit = apply_te_fold(explicit_base, snaps=snaps, params=te_params)
    if int(season) == 2024:
        final, _, wr_audit = apply_wr_fold(te_final, snaps=snaps, params=wr_params)
    else:
        final = te_final
        wr_audit = {"same_future_participation": 0}

    required_uncertainty = {
        "entitlement_tgt_share",
        "bayes_tgt_share_sd",
        "bayes_tgt_share_effective_n",
        "bayes_ypt",
        "bayes_ypt_sd",
        "bayes_ypt_effective_n",
        "bayes_evidence_state",
    }
    missing = sorted(required_uncertainty - set(final.columns))
    if missing:
        raise RuntimeError(f"{season} W{week:02d} missing uncertainty fields: {missing}")

    sims = explicit_simulate(final, iterations=int(iterations), seed=seed)

    _, ml_pred = build_ml(player_logs, bundle.player_consensus, int(season), int(week))
    _, state_pred = build_state_predictions(player_logs, bundle.player_consensus, int(season), int(week))

    mcols = [
        "event_id", "team", "opponent", "player", "player_clean_key",
        "position", "market", "season", "week",
    ]
    market_rows = metrics[mcols].copy()
    market_rows["team"] = market_rows["team"].map(canon_team)
    market_rows["opponent"] = market_rows["opponent"].map(canon_team)
    market_rows["market"] = market_rows["market"].astype(str).str.lower()
    market_rows["position_family"] = market_rows["position"].map(_pos)
    market_rows = _attach_component_projection(market_rows, ml_pred, "ml")
    market_rows = _attach_component_projection(market_rows, state_pred, "state")

    # Attach MC means for the markets used by this candidate.
    use = market_rows["market"].isin(["rec_yards", "rush_yards", "rush_rec_yards"])
    relevant = market_rows.loc[use].copy()
    mc = []
    for _, row in relevant.iterrows():
        mean, _ = _sim_mean(sims, row, str(row["market"]))
        mc.append(mean)
    relevant["mc_proj"] = mc
    relevant = _ensemble_rows(relevant, weights)

    # Carry current Bayesian uncertainty and explicit entitlement to receiver rows.
    context_cols = [
        "event_id", "team", "player_clean_key", "position",
        "entitlement_tgt_share",
        "bayes_tgt_share_sd", "bayes_tgt_share_effective_n",
        "bayes_ypt", "bayes_ypt_sd", "bayes_ypt_effective_n",
        "bayes_evidence_state",
        "rules_plays_est", "rules_pass_rate",
    ]
    context = final[context_cols].copy()
    context["team"] = context["team"].map(canon_team)
    context = context.drop_duplicates(["event_id", "team", "player_clean_key"])
    if context.duplicated(["event_id", "team", "player_clean_key"]).any():
        raise RuntimeError(f"{season} W{week:02d} duplicate receiver context")

    relevant = relevant.merge(
        context,
        on=["event_id", "team", "player_clean_key", "position"],
        how="left",
        validate="many_to_one",
    )

    rec = relevant.loc[
        relevant["market"].eq("rec_yards")
        & relevant["position"].astype(str).str.upper().isin(PASS_CATCHER_POSITIONS)
    ].copy()
    if rec.empty:
        raise RuntimeError(f"{season} W{week:02d} no receiver rows")
    needed = [
        "ensemble_proj", "entitlement_tgt_share",
        "bayes_tgt_share_sd", "bayes_tgt_share_effective_n",
        "bayes_ypt", "bayes_ypt_sd", "bayes_ypt_effective_n",
    ]
    if rec[needed].apply(pd.to_numeric, errors="coerce").isna().any().any():
        raise RuntimeError(f"{season} W{week:02d} receiver rows missing uncertainty/base means")

    rec["baseline_proj"] = pd.to_numeric(rec["ensemble_proj"], errors="raise")
    rec["candidate_proj"] = rec["baseline_proj"].copy()
    rec["reconciled"] = False
    rec["implied_residual_mean"] = np.nan
    rec["team_qb_authority"] = np.nan
    rec["propagated_variance"] = np.nan
    rec["adjustment_yards"] = 0.0
    rec["adjustment_pct"] = 0.0

    return bundle, final, sims, relevant, rec, te_audit, wr_audit


def evaluate_season(
    *,
    season: int,
    prior_season: int,
    weeks: list[int],
    m89: pd.DataFrame,
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rec_score_rows = []
    combo_score_rows = []
    team_audit_rows = []
    max_receiver_identity_gap = 0.0
    max_unadjusted_array_gap = 0.0
    max_rb_combo_identity_gap = 0.0
    wr_future_violations = 0
    m89_rows_consumed = 0

    for week in weeks:
        bundle, final, sims, relevant, rec, te_audit, wr_audit = _prepare_week(
            season=season,
            week=week,
            prior_season=prior_season,
            player_logs=player_logs,
            team_weekly=team_weekly,
            schedule=schedule,
            universe_dir=universe_dir,
            injuries=injuries,
            weather=weather,
            te_params=te_params,
            wr_params=wr_params,
            snaps=snaps,
            weights=weights,
            iterations=iterations,
        )
        wr_future_violations += int(wr_audit.get("same_future_participation", 0))

        mweek = m89.loc[m89["season"].eq(int(season)) & m89["week"].eq(int(week))].copy()
        # M89 authority is intentionally incomplete in some weeks; candidate
        # only acts where exact OOS authority exists.
        for _, q in mweek.iterrows():
            team = canon_team(q["team"])
            opponent = canon_team(q["opponent"])
            qmean = float(q["football_synthesis"])
            team_rec = rec.loc[
                rec["team"].eq(team) & rec["opponent"].eq(opponent)
            ].copy()
            if team_rec.empty:
                raise RuntimeError(
                    f"{season} W{week:02d} M89 authority missing receiver universe team={team} opp={opponent}"
                )
            events = team_rec["event_id"].astype(str).unique().tolist()
            if len(events) != 1:
                raise RuntimeError(
                    f"{season} W{week:02d} M89 authority maps to {len(events)} events team={team}"
                )
            event = str(events[0])
            team_player = final.loc[
                final["event_id"].astype(str).eq(event)
                & final["team"].map(canon_team).eq(team)
            ].copy()
            if team_player.empty:
                raise RuntimeError("M89 team has no historical football rows")
            plays_mean, pass_rate_mean = _team_inputs(team_player)
            A = float(plays_mean * pass_rate_mean)
            if not np.isfinite(A) or A <= 0:
                raise RuntimeError(f"invalid deterministic pass opportunity team={team} A={A}")

            idx = team_rec.index.to_numpy()
            b = pd.to_numeric(team_rec["baseline_proj"], errors="raise").to_numpy(float)
            share = pd.to_numeric(team_rec["entitlement_tgt_share"], errors="raise").to_numpy(float)
            share_sd = pd.to_numeric(team_rec["bayes_tgt_share_sd"], errors="raise").to_numpy(float)
            ypt = pd.to_numeric(team_rec["bayes_ypt"], errors="raise").to_numpy(float)
            ypt_sd = pd.to_numeric(team_rec["bayes_ypt_sd"], errors="raise").to_numpy(float)
            variance = (A * ypt * share_sd) ** 2 + (A * share * ypt_sd) ** 2
            if not np.isfinite(variance).all():
                raise RuntimeError(f"non-finite propagated variance team={team}")
            zero = variance <= 0
            if zero.any():
                if (b[zero] > TOL).any():
                    bad = team_rec.iloc[np.flatnonzero(zero & (b > TOL))][
                        ["player", "player_clean_key", "baseline_proj", "entitlement_tgt_share"]
                    ]
                    raise RuntimeError(
                        f"positive receiver mean has zero uncertainty team={team}:\n{bad.to_string(index=False)}"
                    )
                variance[zero] = 1e-18

            named_sum = float(b.sum())
            if named_sum <= qmean + TOL:
                x = b.copy()
                residual = max(0.0, qmean - named_sum)
                reduction = 0.0
            else:
                x = _weighted_nonnegative_projection(b, variance, qmean)
                residual = 0.0
                reduction = named_sum - qmean

            identity_gap = float(x.sum() + residual - qmean)
            max_receiver_identity_gap = max(max_receiver_identity_gap, abs(identity_gap))
            if abs(identity_gap) > TOL:
                raise RuntimeError(f"receiver mean identity failed team={team} gap={identity_gap}")
            if (x < -TOL).any() or residual < -TOL:
                raise RuntimeError("negative reconciled mean/residual")

            rec.loc[idx, "candidate_proj"] = x
            rec.loc[idx, "reconciled"] = np.abs(x - b) > 1e-12
            rec.loc[idx, "implied_residual_mean"] = residual
            rec.loc[idx, "team_qb_authority"] = qmean
            rec.loc[idx, "propagated_variance"] = variance
            rec.loc[idx, "adjustment_yards"] = x - b
            rec.loc[idx, "adjustment_pct"] = np.where(b > TOL, (x - b) / b, 0.0)

            # Distribution-shape contract: final baseline/candidate rec_yards
            # differ only by mean rescaling. Unchanged rows remain bit-identical.
            for local_i, (_, rr) in enumerate(team_rec.iterrows()):
                raw_mean, raw = _sim_mean(sims, rr, "rec_yards")
                base_mean = float(b[local_i])
                cand_mean = float(x[local_i])
                if raw_mean > 0:
                    base_arr = raw * (base_mean / raw_mean)
                    cand_arr = raw * (cand_mean / raw_mean)
                elif base_mean <= TOL and cand_mean <= TOL:
                    base_arr = np.zeros_like(raw)
                    cand_arr = np.zeros_like(raw)
                else:
                    raise RuntimeError("cannot rescale positive receiver target from zero MC mean")
                if abs(cand_mean - base_mean) <= 1e-12:
                    max_unadjusted_array_gap = max(
                        max_unadjusted_array_gap,
                        float(np.max(np.abs(base_arr - cand_arr))),
                    )

            team_audit_rows.append({
                "season": int(season),
                "week": int(week),
                "event_id": event,
                "team": team,
                "opponent": opponent,
                "qb_player_clean_key": str(q["player_clean_key"]),
                "qb_football_synthesis": qmean,
                "named_receiver_baseline_sum": named_sum,
                "named_receiver_candidate_sum": float(x.sum()),
                "implied_residual_mean": residual,
                "residual_only_feasible": bool(named_sum <= qmean + TOL),
                "required_named_reduction_yards": float(reduction),
                "required_named_reduction_pct": float(reduction / named_sum) if named_sum > 0 else 0.0,
                "candidate_identity_gap": identity_gap,
                "deterministic_pass_attempt_input": A,
                "receiver_rows": int(len(team_rec)),
            })
            m89_rows_consumed += 1

        # Score receiver rows only after all candidate means are frozen for week.
        actual = build_actual_rows(player_logs, int(season), int(week))
        actual["team"] = actual["team"].map(canon_team)
        actual["market"] = actual["market"].astype(str).str.lower()
        actual_rec = actual.loc[actual["market"].eq("rec_yards")].copy()
        score = rec.merge(
            actual_rec[["team", "player_clean_key", "market", "actual"]],
            on=["team", "player_clean_key", "market"],
            how="inner",
            validate="one_to_one",
        )
        score = score.loc[score["position_family"].isin(PRIMARY_POSITIONS)].copy()
        if score.empty:
            raise RuntimeError(f"{season} W{week:02d} no receiver scoring rows")
        rec_score_rows.append(score)

        # Current RB V2 baseline + candidate dependent mean.
        rb_map, rb_payload = build_candidate_map(relevant, sims, weights)
        if int(rb_payload.get("sportsbook_inputs_used", 1)) != 0:
            raise RuntimeError("RB V2 reported sportsbook input")
        max_rb_combo_identity_gap = max(
            max_rb_combo_identity_gap,
            float(rb_payload.get("max_pathwise_identity_gap", 0.0)),
        )

        combo_rows = relevant.loc[
            relevant["market"].eq("rush_rec_yards")
            & relevant["position_family"].eq("RB")
        ].copy()
        if not combo_rows.empty:
            candidate_rec_map = rec.set_index(
                ["event_id", "player_clean_key"]
            )["candidate_proj"].to_dict()
            baseline_rec_map = rec.set_index(
                ["event_id", "player_clean_key"]
            )["baseline_proj"].to_dict()
            combo_rows["baseline_proj"] = pd.to_numeric(
                combo_rows["ensemble_proj"], errors="raise"
            )
            combo_rows["candidate_proj"] = combo_rows["baseline_proj"].copy()

            for idx2, rr in combo_rows.iterrows():
                key = (str(rr["event_id"]), str(rr["player_clean_key"]))
                if int(week) == 1:
                    # V2 is non-Week-1: dependent market remains exact current baseline.
                    continue
                meta = rb_map.get(key)
                if meta is None:
                    raise RuntimeError(
                        f"{season} W{week:02d} RB V2 missing eligible combo key={key}"
                    )
                base_rec = float(baseline_rec_map[key])
                cand_rec = float(candidate_rec_map[key])
                if abs(float(meta["rec_target_mean"]) - base_rec) > 1e-8:
                    raise RuntimeError(
                        f"RB V2 baseline rec target drift key={key} "
                        f"map={meta['rec_target_mean']} base={base_rec}"
                    )
                combo_rows.at[idx2, "baseline_proj"] = float(meta["target_mean"])
                combo_rows.at[idx2, "candidate_proj"] = float(meta["rush_target_mean"]) + cand_rec

                # Pathwise candidate identity with unchanged rush component and
                # rescaled candidate rec component.
                rush_row = relevant.loc[
                    relevant["event_id"].astype(str).eq(key[0])
                    & relevant["player_clean_key"].astype(str).eq(key[1])
                    & relevant["market"].eq("rush_yards")
                ]
                rec_row = relevant.loc[
                    relevant["event_id"].astype(str).eq(key[0])
                    & relevant["player_clean_key"].astype(str).eq(key[1])
                    & relevant["market"].eq("rec_yards")
                ]
                if len(rush_row) != 1 or len(rec_row) != 1:
                    raise RuntimeError(f"RB V2 component row multiplicity key={key}")
                rush_raw_mean, rush_raw = _sim_mean(sims, rush_row.iloc[0], "rush_yards")
                rec_raw_mean, rec_raw = _sim_mean(sims, rec_row.iloc[0], "rec_yards")
                rush_target = float(meta["rush_target_mean"])
                rush_adj = rush_raw * (rush_target / rush_raw_mean) if rush_raw_mean > 0 else np.zeros_like(rush_raw)
                rec_adj = rec_raw * (cand_rec / rec_raw_mean) if rec_raw_mean > 0 else np.zeros_like(rec_raw)
                conserved = rush_adj + rec_adj
                target = rush_target + cand_rec
                mean_gap = abs(float(conserved.mean() - target))
                identity = float(np.max(np.abs(conserved - (rush_adj + rec_adj))))
                max_rb_combo_identity_gap = max(max_rb_combo_identity_gap, mean_gap, identity)

            actual_combo = actual.loc[actual["market"].eq("rush_rec_yards")].copy()
            combo_score = combo_rows.merge(
                actual_combo[["team", "player_clean_key", "market", "actual"]],
                on=["team", "player_clean_key", "market"],
                how="inner",
                validate="one_to_one",
            )
            if not combo_score.empty:
                combo_score_rows.append(combo_score)

    rec_detail = pd.concat(rec_score_rows, ignore_index=True)
    combo_detail = pd.concat(combo_score_rows, ignore_index=True) if combo_score_rows else pd.DataFrame()
    team_audit = pd.DataFrame(team_audit_rows)

    expected = 444 if int(season) == 2024 else 440
    if m89_rows_consumed != expected:
        raise RuntimeError(
            f"season={season} M89 authority consumption mismatch expected={expected} got={m89_rows_consumed}"
        )
    if max_receiver_identity_gap > TOL:
        raise RuntimeError("receiver identity gate failed")
    if max_unadjusted_array_gap > TOL:
        raise RuntimeError("unadjusted receiver arrays changed")
    if max_rb_combo_identity_gap > TOL:
        raise RuntimeError(f"RB combo identity gap={max_rb_combo_identity_gap}")

    scope = {
        "season": int(season),
        "m89_rows_consumed": int(m89_rows_consumed),
        "receiver_identity_max_gap": float(max_receiver_identity_gap),
        "unadjusted_receiver_array_max_gap": float(max_unadjusted_array_gap),
        "rb_v2_combo_identity_max_gap": float(max_rb_combo_identity_gap),
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_used_upstream": 0,
        "qb_means_changed": 0,
        "qb_arrays_changed": 0,
        "receptions_arrays_changed": 0,
        "rush_att_arrays_changed": 0,
        "rush_yards_arrays_changed": 0,
        "target_entitlement_mutations": 0,
        "wr_same_future_participation": int(wr_future_violations),
        "residual_only_feasible_rate": float(team_audit["residual_only_feasible"].mean()),
        "named_reduction_required_rate": float((~team_audit["residual_only_feasible"]).mean()),
        "median_required_named_reduction_pct": float(
            team_audit.loc[
                ~team_audit["residual_only_feasible"], "required_named_reduction_pct"
            ].median()
        ) if (~team_audit["residual_only_feasible"]).any() else 0.0,
        "median_implied_residual_share": float(
            (
                team_audit["implied_residual_mean"]
                / team_audit["qb_football_synthesis"]
            ).median()
        ),
    }
    return rec_detail, combo_detail, team_audit, scope


def _add_quartiles(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    out["entitlement_quartile"] = ""
    for season in sorted(out["season"].unique()):
        mask = out["season"].eq(season)
        ranked = out.loc[mask, "entitlement_tgt_share"].rank(method="first")
        out.loc[mask, "entitlement_quartile"] = pd.qcut(
            ranked, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]
        ).astype(str).to_numpy()
    return out


def build_scorecard(rec_detail: pd.DataFrame, combo_detail: pd.DataFrame) -> dict:
    rec_detail = _add_quartiles(rec_detail)

    pooled = _macro(rec_detail)
    by_season = {
        str(season): _macro(rec_detail.loc[rec_detail["season"].eq(season)])
        for season in (2024, 2025)
    }
    q4 = _score(rec_detail.loc[rec_detail["entitlement_quartile"].eq("Q4_high")])
    quartiles = {
        q: _score(rec_detail.loc[rec_detail["entitlement_quartile"].eq(q)])
        for q in ("Q1_low", "Q2", "Q3", "Q4_high")
    }

    combo = {}
    for label, frame in (
        ("pooled", combo_detail),
        ("2024", combo_detail.loc[combo_detail["season"].eq(2024)]),
        ("2025", combo_detail.loc[combo_detail["season"].eq(2025)]),
    ):
        if frame.empty:
            raise RuntimeError(f"empty RB combo cohort {label}")
        combo[label] = _score(frame)

    return {
        "receiving_yards_pooled": pooled,
        "receiving_yards_by_season": by_season,
        "receiving_yards_q4": q4,
        "receiving_yards_by_entitlement_quartile": quartiles,
        "rb_rush_rec_yards": combo,
    }


def apply_gates(score: dict, scope24: dict, scope25: dict) -> dict:
    p = score["receiving_yards_pooled"]
    s24 = score["receiving_yards_by_season"]["2024"]
    s25 = score["receiving_yards_by_season"]["2025"]
    q4 = score["receiving_yards_q4"]
    combo = score["rb_rush_rec_yards"]

    mech = (
        scope24["m89_rows_consumed"] == 444
        and scope25["m89_rows_consumed"] == 440
        and scope24["receiver_identity_max_gap"] <= TOL
        and scope25["receiver_identity_max_gap"] <= TOL
        and scope24["unadjusted_receiver_array_max_gap"] <= TOL
        and scope25["unadjusted_receiver_array_max_gap"] <= TOL
        and scope24["rb_v2_combo_identity_max_gap"] <= TOL
        and scope25["rb_v2_combo_identity_max_gap"] <= TOL
        and scope24["sportsbook_inputs_used"] == 0
        and scope25["sportsbook_inputs_used"] == 0
        and scope24["target_game_outcomes_used_upstream"] == 0
        and scope25["target_game_outcomes_used_upstream"] == 0
        and scope24["qb_means_changed"] == 0
        and scope25["qb_means_changed"] == 0
        and scope24["qb_arrays_changed"] == 0
        and scope25["qb_arrays_changed"] == 0
        and scope24["receptions_arrays_changed"] == 0
        and scope25["receptions_arrays_changed"] == 0
        and scope24["rush_att_arrays_changed"] == 0
        and scope25["rush_att_arrays_changed"] == 0
        and scope24["rush_yards_arrays_changed"] == 0
        and scope25["rush_yards_arrays_changed"] == 0
    )

    gates = {
        "pooled_macro_mae_improves":
            p["macro_candidate_mae"] < p["macro_baseline_mae"],
        "macro_mae_2024_nonworse":
            s24["macro_candidate_mae"] <= s24["macro_baseline_mae"] + 1e-12,
        "macro_mae_2025_nonworse":
            s25["macro_candidate_mae"] <= s25["macro_baseline_mae"] + 1e-12,
        "wr_pooled_mae_nonworse":
            p["positions"]["WR"]["candidate_mae"] <= p["positions"]["WR"]["baseline_mae"] + 1e-12,
        "te_pooled_mae_nonworse":
            p["positions"]["TE"]["candidate_mae"] <= p["positions"]["TE"]["baseline_mae"] + 1e-12,
        "rb_pooled_mae_nonworse":
            p["positions"]["RB"]["candidate_mae"] <= p["positions"]["RB"]["baseline_mae"] + 1e-12,
        "pooled_macro_p90_nonworse":
            p["macro_candidate_p90_ae"] <= p["macro_baseline_p90_ae"] + 1e-12,
        "macro_p90_2024_nonworse":
            s24["macro_candidate_p90_ae"] <= s24["macro_baseline_p90_ae"] + 1e-12,
        "macro_p90_2025_nonworse":
            s25["macro_candidate_p90_ae"] <= s25["macro_baseline_p90_ae"] + 1e-12,
        "pooled_macro_miss40_nonworse":
            p["macro_candidate_miss40_rate"] <= p["macro_baseline_miss40_rate"] + 1e-12,
        "q4_mae_nonworse":
            q4["candidate_mae"] <= q4["baseline_mae"] + 1e-12,
        "q4_p90_nonworse":
            q4["candidate_p90_ae"] <= q4["baseline_p90_ae"] + 1e-12,
        "q4_miss40_nonworse":
            q4["candidate_miss40_rate"] <= q4["baseline_miss40_rate"] + 1e-12,
        "pooled_macro_abs_bias_nonworse":
            p["macro_candidate_abs_bias"] <= p["macro_baseline_abs_bias"] + 1e-12,
        "rb_combo_mae_2024_nonworse":
            combo["2024"]["candidate_mae"] <= combo["2024"]["baseline_mae"] + 1e-12,
        "rb_combo_mae_2025_nonworse":
            combo["2025"]["candidate_mae"] <= combo["2025"]["baseline_mae"] + 1e-12,
        "rb_combo_p90_pooled_nonworse":
            combo["pooled"]["candidate_p90_ae"] <= combo["pooled"]["baseline_p90_ae"] + 1e-12,
        "mechanical_provenance_all_pass": bool(mech),
    }
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
    ap.add_argument("--m89-casebook", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("data/model_ensemble_weights.csv"))
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    logs = _read(a.player_logs, "player logs")
    team = _read(a.team_weekly, "team weekly")
    schedule = _read(a.schedule, "schedule")
    injuries = _optional(a.injuries)
    weather = _optional(a.weather)
    weights = load_weights(a.weights)
    m89 = _canonicalize_m89(_read(a.m89_casebook, "M89 authority"))

    te24 = _load_fold_params(a.te_coefficients, test_season=2024, features=TE_FEATURES, label="TE-R5P")
    te25 = _load_fold_params(a.te_coefficients, test_season=2025, features=TE_FEATURES, label="TE-R5P")
    wr24 = _load_fold_params(a.wr_coefficients, test_season=2024, features=WR_FEATURES, label="WR-R15")
    snaps, dup, _ = _load_participation_snaps()
    if dup > 0.01:
        raise RuntimeError(f"participation snap duplicate rate too high: {dup}")
    weeks = _parse_weeks(a.weeks)

    d24, c24, t24, s24 = evaluate_season(
        season=2024, prior_season=2023, weeks=weeks, m89=m89,
        player_logs=logs, team_weekly=team, schedule=schedule,
        universe_dir=a.universe_2024, injuries=injuries, weather=weather,
        te_params=te24, wr_params=wr24, snaps=snaps,
        weights=weights, iterations=a.iterations,
    )
    d25, c25, t25, s25 = evaluate_season(
        season=2025, prior_season=2024, weeks=weeks, m89=m89,
        player_logs=logs, team_weekly=team, schedule=schedule,
        universe_dir=a.universe_2025, injuries=injuries, weather=weather,
        te_params=te25, wr_params=None, snaps=snaps,
        weights=weights, iterations=a.iterations,
    )

    rec_detail = pd.concat([d24, d25], ignore_index=True)
    combo_detail = pd.concat([c24, c25], ignore_index=True)
    team_audit = pd.concat([t24, t25], ignore_index=True)
    score = build_scorecard(rec_detail, combo_detail)
    gates = apply_gates(score, s24, s25)
    qualified = all(gates.values())
    disposition = (
        f"{VERSION}_QUALIFIED"
        if qualified else f"{VERSION}_FAILED_CLOSED"
    )

    result = {
        "version": VERSION,
        "disposition": disposition,
        "qualified": bool(qualified),
        "production_changed": False,
        "parameters_fit": 0,
        "candidate_variants_scored": 1,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_used_upstream": 0,
        "m89_authority": {
            "run": 34122984048,
            "artifact": 10018942911,
            "rows_2024": 444,
            "rows_2025": 440,
        },
        "scope_2024": s24,
        "scope_2025": s25,
        "scorecard": score,
        "gates": gates,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    d24.to_csv(a.out_dir / "receiver_detail_2024.csv", index=False)
    d25.to_csv(a.out_dir / "receiver_detail_2025.csv", index=False)
    c24.to_csv(a.out_dir / "rb_combo_detail_2024.csv", index=False)
    c25.to_csv(a.out_dir / "rb_combo_detail_2025.csv", index=False)
    team_audit.to_csv(a.out_dir / "team_reconciliation_audit.csv", index=False)
    (a.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    p = score["receiving_yards_pooled"]
    q4 = score["receiving_yards_q4"]
    combo = score["rb_rush_rec_yards"]
    lines = [
        "# Hierarchical Receiver Mean Reconciliation V1 — Historical Result",
        "",
        f"Disposition: **{disposition}**",
        "",
        "## Receiving yards pooled",
        "",
        f"- macro MAE: `{p['macro_baseline_mae']:.6f} -> {p['macro_candidate_mae']:.6f}`",
        f"- macro p90 AE: `{p['macro_baseline_p90_ae']:.6f} -> {p['macro_candidate_p90_ae']:.6f}`",
        f"- macro 40+ miss rate: `{p['macro_baseline_miss40_rate']:.6f} -> {p['macro_candidate_miss40_rate']:.6f}`",
        f"- macro absolute bias: `{p['macro_baseline_abs_bias']:.6f} -> {p['macro_candidate_abs_bias']:.6f}`",
        "",
    ]
    for pos in PRIMARY_POSITIONS:
        s = p["positions"][pos]
        lines.append(
            f"- {pos}: MAE `{s['baseline_mae']:.6f} -> {s['candidate_mae']:.6f}`; "
            f"p90 `{s['baseline_p90_ae']:.6f} -> {s['candidate_p90_ae']:.6f}`"
        )
    lines += [
        "",
        "## Q4 high entitlement",
        "",
        f"- MAE: `{q4['baseline_mae']:.6f} -> {q4['candidate_mae']:.6f}`",
        f"- p90 AE: `{q4['baseline_p90_ae']:.6f} -> {q4['candidate_p90_ae']:.6f}`",
        f"- 40+ miss rate: `{q4['baseline_miss40_rate']:.6f} -> {q4['candidate_miss40_rate']:.6f}`",
        "",
        "## RB rush+receiving",
        "",
        f"- 2024 MAE: `{combo['2024']['baseline_mae']:.6f} -> {combo['2024']['candidate_mae']:.6f}`",
        f"- 2025 MAE: `{combo['2025']['baseline_mae']:.6f} -> {combo['2025']['candidate_mae']:.6f}`",
        f"- pooled p90 AE: `{combo['pooled']['baseline_p90_ae']:.6f} -> {combo['pooled']['candidate_p90_ae']:.6f}`",
        "",
        "## Frozen gates",
        "",
    ]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in gates.items()]
    (a.out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
