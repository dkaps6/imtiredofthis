#!/usr/bin/env python3
"""Current-stack historical A/B for ONE_PASS_STATE_INTEGRATION_V1.

Frozen contracts:
- exact M38 -> TE-R5P -> WR-R15 fold-safe historical ordering;
- leakage-safe Phase-J walk-forward C2 selector;
- current C2 completed-pass process;
- current ensemble weights;
- current RB Rush+Receiving Conservation V2 downstream mean authority;
- zero sportsbook input;
- no target-game outcome enters target-game projection/selection.

Baseline represents current production simulation behavior:
selected QB pass_yards use C2 while receiver arrays remain canonical.

Candidate keeps every QB pass_yards array bit-identical and, only on C2-selected
team-games, installs the exact receiver receptions/rec_yards generated inside
that same C2 completed-pass realization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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
from scripts.simulation_c2_qb_candidate import PASS_CATCHER_POSITIONS, StateSimulationResult, simulate_with_states
from scripts.simulation_v2 import lookup
from scripts.utils.canonical_names import canon_team

VERSION = "ONE_PASS_STATE_INTEGRATION_V1"
SELECTOR_FEATURES = [
    "pass_opportunity_spot",
    "pass_efficiency_spot",
    "rush_opportunity_spot",
    "rush_efficiency_spot",
    "pred_qb_attempts",
    "week",
]
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
    return out


def _optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() and path.stat().st_size else pd.DataFrame()


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


def _selector_source(phase_c: pd.DataFrame) -> pd.DataFrame:
    d = phase_c.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    d["season"] = pd.to_numeric(d["season"], errors="raise").astype(int)
    d["week"] = pd.to_numeric(d["week"], errors="raise").astype(int)
    d["team"] = d["team"].map(canon_team)
    qb = d.loc[d["position"].astype(str).str.upper().eq("QB")].copy()
    keys = ["season", "week", "team"]
    qb["pred_opportunity"] = pd.to_numeric(qb["pred_opportunity"], errors="coerce")
    qb = (
        qb.sort_values(keys + ["pred_opportunity"], ascending=[True, True, True, False], kind="mergesort")
        .drop_duplicates(keys, keep="first")
        .copy()
    )
    keep = keys + [
        "player",
        "player_clean_key",
        "pred_opportunity",
        "actual_opportunity",
        "pass_opportunity_spot",
        "pass_efficiency_spot",
        "rush_opportunity_spot",
        "rush_efficiency_spot",
    ]
    q = qb[keep].rename(
        columns={
            "pred_opportunity": "pred_qb_attempts",
            "actual_opportunity": "actual_qb_attempts",
        }
    )
    q = q.sort_values(keys).reset_index(drop=True)
    if q.duplicated(keys).any():
        raise RuntimeError("Phase-C QB selector source has duplicate team-games")
    return q


def _walk_forward_selector(source: pd.DataFrame, seasons=(2024, 2025)) -> pd.DataFrame:
    rows = []
    src = source.copy()
    for season in seasons:
        target_season = src.loc[src["season"].eq(int(season))].copy()
        if target_season.empty:
            raise RuntimeError(f"selector source has no season={season}")
        for week in sorted(target_season["week"].unique()):
            test = target_season.loc[target_season["week"].eq(int(week))].copy()
            train = src.loc[
                src["season"].lt(int(season))
                | (src["season"].eq(int(season)) & src["week"].lt(int(week)))
            ].dropna(subset=SELECTOR_FEATURES + ["actual_qb_attempts"]).copy()
            train["residual_attempts"] = train["actual_qb_attempts"] - train["pred_qb_attempts"]
            if len(train) < 128:
                delta = np.zeros(len(test), dtype=float)
            else:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(train[SELECTOR_FEATURES])
                model = Ridge(alpha=20.0).fit(x_train, train["residual_attempts"])
                delta = model.predict(scaler.transform(test[SELECTOR_FEATURES]))
            test["selector_training_rows"] = int(len(train))
            test["delta_pass_attempts"] = delta
            test["selector_c2_selected"] = test["delta_pass_attempts"].gt(0)
            rows.append(test)
    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out.sort_values(["season", "week", "team"]).reset_index(drop=True)
    return out


def _verify_phase_j(selector: pd.DataFrame, phase_j: pd.DataFrame) -> dict:
    ref = phase_j.copy()
    ref.columns = [str(c).strip().lower() for c in ref.columns]
    ref["season"] = pd.to_numeric(ref["season"], errors="raise").astype(int)
    ref["week"] = pd.to_numeric(ref["week"], errors="raise").astype(int)
    ref["team"] = ref["team"].map(canon_team)
    ref = ref.loc[ref["season"].eq(2025)].copy()
    got = selector.loc[selector["season"].eq(2025)].copy()
    keys = ["season", "week", "team"]
    joined = got.merge(
        ref[keys + ["delta_pass_attempts"]],
        on=keys,
        how="outer",
        suffixes=("_rebuilt", "_phase_j"),
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("rebuilt 2025 selector universe does not exactly match preserved Phase-J")
    a = pd.to_numeric(joined["delta_pass_attempts_rebuilt"], errors="raise").to_numpy(float)
    b = pd.to_numeric(joined["delta_pass_attempts_phase_j"], errors="raise").to_numpy(float)
    max_gap = float(np.max(np.abs(a - b))) if len(a) else np.inf
    decision_mismatch = int(((a > 0) != (b > 0)).sum())
    if max_gap > 1e-9 or decision_mismatch:
        raise RuntimeError(
            f"Phase-J reproduction failed max_delta_gap={max_gap} decision_mismatch={decision_mismatch}"
        )
    return {
        "rows": int(len(joined)),
        "max_abs_delta_gap": max_gap,
        "decision_mismatches": decision_mismatch,
        "selected_rows": int((a > 0).sum()),
    }


def _qb_key_for_team(
    state: StateSimulationResult,
    final: pd.DataFrame,
    selector_row: pd.Series,
) -> tuple[str, str]:
    team = canon_team(selector_row["team"])
    part = final.loc[final["team"].map(canon_team).eq(team)].copy()
    events = part["event_id"].dropna().astype(str).unique().tolist()
    if len(events) != 1:
        raise RuntimeError(f"expected one event for team={team}, got {events}")
    game = str(events[0])
    wanted = str(selector_row.get("player_clean_key", "") or "").strip()
    if wanted and (game, wanted, "pass_yards") in state.values:
        return game, wanted
    q = part.loc[part["position"].map(_pos).eq("QB")].copy()
    exact = q.loc[q["player_clean_key"].astype(str).eq(wanted)]
    if len(exact) == 1:
        pk = str(exact.iloc[0]["player_clean_key"])
        if (game, pk, "pass_yards") in state.values:
            return game, pk
    available = [
        str(r.player_clean_key)
        for r in q[["player_clean_key"]].drop_duplicates().itertuples(index=False)
        if (game, str(r.player_clean_key), "pass_yards") in state.values
    ]
    if len(available) == 1:
        return game, available[0]
    raise RuntimeError(
        f"could not uniquely resolve historical primary QB team={team} wanted={wanted} available={available}"
    )


def _build_ab_states(
    state: StateSimulationResult,
    final: pd.DataFrame,
    selector_week: pd.DataFrame,
) -> tuple[StateSimulationResult, StateSimulationResult, dict]:
    selector_week = selector_week.copy()
    selector_week["team"] = selector_week["team"].map(canon_team)
    frame = final.copy()
    frame["team"] = frame["team"].map(canon_team)
    frame["position_family"] = frame["position"].map(_pos)

    selector_teams = set(selector_week["team"].astype(str))
    football_teams = set(frame["team"].dropna().astype(str))
    extra_selector = sorted(selector_teams - football_teams)
    if extra_selector:
        raise RuntimeError(f"historical selector contains teams absent from football universe: {extra_selector}")

    # C2 consumes one deterministic RNG stream across the entire football slate.
    # Preserve that exact draw order by building shadow state for every football
    # team. Teams outside the preserved Phase-C selector universe receive only a
    # deterministic filler anchor; their shadow outputs are never installed.
    anchors: dict[tuple[str, str], float] = {}
    qb_keys: dict[str, tuple[str, str]] = {}
    selector_by_team = {canon_team(r.team): r for r in selector_week.itertuples(index=False)}
    for team in sorted(football_teams):
        part = frame.loc[frame["team"].eq(team)].copy()
        events = part["event_id"].dropna().astype(str).unique().tolist()
        if len(events) != 1:
            raise RuntimeError(f"expected one historical event for team={team}, got {events}")
        game = str(events[0])
        if team in selector_by_team:
            srow = pd.Series(selector_by_team[team]._asdict())
            game, qpk = _qb_key_for_team(state, frame, srow)
            qb_keys[team] = (game, qpk)
        else:
            q = part.loc[part["position_family"].eq("QB")].copy()
            available = sorted(
                {
                    str(r.player_clean_key)
                    for r in q[["player_clean_key"]].drop_duplicates().itertuples(index=False)
                    if (game, str(r.player_clean_key), "pass_yards") in state.values
                }
            )
            if not available:
                raise RuntimeError(f"historical selector-ineligible team has no canonical QB array team={team}")
            qpk = available[0]
        arr = np.asarray(state.values[(game, qpk, "pass_yards")], dtype=float)
        mean = float(arr.mean())
        if not np.isfinite(mean) or mean <= 0:
            raise RuntimeError(f"invalid canonical QB anchor team={team} mean={mean}")
        anchors[(game, team)] = mean

    shadow = _capture_c2_shadow(state, frame, anchors)
    baseline_values = {k: np.asarray(v).copy() for k, v in state.values.items()}
    candidate_values = {k: np.asarray(v).copy() for k, v in state.values.items()}

    selected_teams = set(
        selector_week.loc[selector_week["selector_c2_selected"].astype(bool), "team"].astype(str)
    )
    selected_semantic_violations = 0
    max_identity_gap = 0.0
    max_qb_baseline_candidate_gap = 0.0
    max_unselected_receiver_gap = 0.0
    max_rush_att_gap = 0.0
    max_rush_yards_gap = 0.0
    changed_receiver_keys = 0

    for team, (game, qpk) in qb_keys.items():
        selected = team in selected_teams
        if selected:
            sh = shadow[(game, team)]
            qb_c2 = np.asarray(sh["total_scaled"], dtype=float)
            baseline_values[(game, qpk, "pass_yards")] = qb_c2.copy()
            candidate_values[(game, qpk, "pass_yards")] = qb_c2.copy()

            part = frame.loc[frame["team"].eq(team) & frame["event_id"].astype(str).eq(game)].copy()
            modeled = np.zeros(state.iterations, dtype=float)
            for _, row in part.iterrows():
                pos = str(row["position"]).upper().strip()
                if pos not in PASS_CATCHER_POSITIONS:
                    continue
                pk = str(row["player_clean_key"])
                if pk not in sh["player_yards_scaled"]:
                    continue
                recs = np.asarray(sh["player_receptions"][pk], dtype=float)
                yards = np.asarray(sh["player_yards_scaled"][pk], dtype=float)
                selected_semantic_violations += int(np.sum((recs <= 0) & (yards > 1e-12)))
                modeled += yards
                old_rec = np.asarray(candidate_values[(game, pk, "receptions")], dtype=float)
                old_yards = np.asarray(candidate_values[(game, pk, "rec_yards")], dtype=float)
                changed_receiver_keys += int(float(np.max(np.abs(old_rec - recs))) > 0)
                changed_receiver_keys += int(float(np.max(np.abs(old_yards - yards))) > 0)
                candidate_values[(game, pk, "receptions")] = recs.copy()
                candidate_values[(game, pk, "rec_yards")] = yards.copy()
                rush = np.asarray(candidate_values[(game, pk, "rush_yards")], dtype=float)
                candidate_values[(game, pk, "rush_rec_yards")] = rush + yards
            ident = float(np.max(np.abs(qb_c2 - (modeled + np.asarray(sh["residual_yards_scaled"], float)))))
            max_identity_gap = max(max_identity_gap, ident)

    if selected_semantic_violations:
        raise RuntimeError(
            f"selected-team candidate has zero-reception positive-yard draws: {selected_semantic_violations}"
        )

    # Mechanical invariance checks.
    for key in baseline_values:
        b = np.asarray(baseline_values[key], dtype=float)
        c = np.asarray(candidate_values[key], dtype=float)
        if b.shape != c.shape:
            raise RuntimeError(f"shape drift key={key}")
        gap = float(np.max(np.abs(b - c))) if len(b) else 0.0
        market = key[2]
        if market == "pass_yards":
            max_qb_baseline_candidate_gap = max(max_qb_baseline_candidate_gap, gap)
        if market == "rush_att":
            max_rush_att_gap = max(max_rush_att_gap, gap)
        if market == "rush_yards":
            max_rush_yards_gap = max(max_rush_yards_gap, gap)

    for _, srow in selector_week.loc[~selector_week["selector_c2_selected"].astype(bool)].iterrows():
        team = canon_team(srow["team"])
        game, _ = qb_keys[team]
        part = frame.loc[frame["team"].eq(team) & frame["event_id"].astype(str).eq(game)]
        for _, row in part.iterrows():
            pk = str(row["player_clean_key"])
            for market in ("receptions", "rec_yards", "rush_rec_yards"):
                key = (game, pk, market)
                if key not in baseline_values:
                    continue
                gap = float(
                    np.max(
                        np.abs(
                            np.asarray(baseline_values[key], float)
                            - np.asarray(candidate_values[key], float)
                        )
                    )
                )
                max_unselected_receiver_gap = max(max_unselected_receiver_gap, gap)

    if max_qb_baseline_candidate_gap > TOL:
        raise RuntimeError(f"QB arrays changed baseline vs candidate max_gap={max_qb_baseline_candidate_gap}")
    if max_unselected_receiver_gap > TOL:
        raise RuntimeError(f"unselected receiver arrays changed max_gap={max_unselected_receiver_gap}")
    if max_rush_att_gap > TOL or max_rush_yards_gap > TOL:
        raise RuntimeError(
            f"rushing arrays changed rush_att={max_rush_att_gap} rush_yards={max_rush_yards_gap}"
        )
    if max_identity_gap > TOL:
        raise RuntimeError(f"selected-team pass identity failed max_gap={max_identity_gap}")

    baseline = StateSimulationResult(baseline_values, state.iterations, state.team_states)
    candidate = StateSimulationResult(candidate_values, state.iterations, state.team_states)
    audit = {
        "selected_teams": int(len(selected_teams)),
        "selector_eligible_teams": int(len(selector_teams)),
        "selector_ineligible_teams": int(len(football_teams - selector_teams)),
        "total_teams": int(len(football_teams)),
        "selected_semantic_violations": int(selected_semantic_violations),
        "max_selected_pass_identity_gap": float(max_identity_gap),
        "max_qb_baseline_candidate_gap": float(max_qb_baseline_candidate_gap),
        "max_unselected_receiver_gap": float(max_unselected_receiver_gap),
        "max_rush_att_gap": float(max_rush_att_gap),
        "max_rush_yards_gap": float(max_rush_yards_gap),
        "changed_receiver_array_keys": int(changed_receiver_keys),
    }
    return baseline, candidate, audit


def _ensemble_projection(records: pd.DataFrame, *, weights: pd.DataFrame, mc_col: str) -> np.ndarray:
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
    cand_closer = changed & (ca < ba - 1e-12)
    base_closer = changed & (ba < ca - 1e-12)
    decided = cand_closer | base_closer
    return {
        "n": int(len(part)),
        "baseline_mae": float(np.mean(ba)),
        "candidate_mae": float(np.mean(ca)),
        "baseline_rmse": float(np.sqrt(np.mean(be * be))),
        "candidate_rmse": float(np.sqrt(np.mean(ce * ce))),
        "baseline_bias": float(np.mean(be)),
        "candidate_bias": float(np.mean(ce)),
        "baseline_corr": float(np.corrcoef(actual, b)[0, 1]) if len(part) > 1 and np.std(actual) > 0 and np.std(b) > 0 else None,
        "candidate_corr": float(np.corrcoef(actual, c)[0, 1]) if len(part) > 1 and np.std(actual) > 0 and np.std(c) > 0 else None,
        "baseline_median_ae": float(np.quantile(ba, .50)),
        "candidate_median_ae": float(np.quantile(ca, .50)),
        "baseline_p75_ae": float(np.quantile(ba, .75)),
        "candidate_p75_ae": float(np.quantile(ca, .75)),
        "baseline_p90_ae": float(np.quantile(ba, .90)),
        "candidate_p90_ae": float(np.quantile(ca, .90)),
        "baseline_miss20": float(np.mean(ba >= 20)),
        "candidate_miss20": float(np.mean(ca >= 20)),
        "baseline_miss30": float(np.mean(ba >= 30)),
        "candidate_miss30": float(np.mean(ca >= 30)),
        "baseline_miss40": float(np.mean(ba >= 40)),
        "candidate_miss40": float(np.mean(ca >= 40)),
        "changed_rows": int(changed.sum()),
        "candidate_closer": int(cand_closer.sum()),
        "baseline_closer": int(base_closer.sum()),
        "candidate_closer_rate": float(cand_closer.sum() / decided.sum()) if int(decided.sum()) else None,
    }


def _macro(detail: pd.DataFrame, market: str) -> dict:
    rows = {}
    for pos in POSITIONS:
        rows[pos] = _score(detail.loc[detail["market"].eq(market) & detail["position_family"].eq(pos)])
    fields = [
        "baseline_mae",
        "candidate_mae",
        "baseline_p90_ae",
        "candidate_p90_ae",
        "baseline_miss40",
        "candidate_miss40",
    ]
    out = {f"macro_{f}": float(np.mean([rows[p][f] for p in POSITIONS])) for f in fields}
    out["positions"] = rows
    return out


def _add_entitlement_quartile(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    out["entitlement_quartile"] = ""
    for season in sorted(out["season"].unique()):
        base = (
            out.loc[out["season"].eq(season) & out["market"].eq("rec_yards"), ["event_id", "team", "player_clean_key", "entitlement_tgt_share"]]
            .drop_duplicates(["event_id", "team", "player_clean_key"])
            .copy()
        )
        if base.empty:
            continue
        ranked = base["entitlement_tgt_share"].rank(method="first")
        base["entitlement_quartile"] = pd.qcut(
            ranked, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]
        ).astype(str)
        qmap = base.set_index(["event_id", "team", "player_clean_key"])["entitlement_quartile"].to_dict()
        mask = out["season"].eq(season)
        out.loc[mask, "entitlement_quartile"] = [
            qmap.get((str(r.event_id), str(r.team), str(r.player_clean_key)), "")
            for r in out.loc[mask, ["event_id", "team", "player_clean_key"]].itertuples(index=False)
        ]
    return out


def evaluate_season(
    *,
    season: int,
    prior_season: int,
    weeks: list[int],
    selector: pd.DataFrame,
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
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    details = []
    audits = []
    max_v2_gap = 0.0
    sportsbook_inputs = 0

    for week in weeks:
        universe = _read(universe_dir / f"{season}_week_{week:02d}.csv", f"{season} W{week} universe")
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
        te_final, _, te_audit = apply_te_fold(explicit_base, snaps=snaps, params=te_params)
        if int(season) == 2024:
            final, _, wr_audit = apply_wr_fold(te_final, snaps=snaps, params=wr_params)
        else:
            final = te_final
            wr_audit = {"same_future_participation": 0}

        entitlement_before = pd.to_numeric(final["entitlement_tgt_share"], errors="raise").to_numpy(float).copy()
        state = simulate_with_states(final, iterations=int(iterations), seed=int(seed))
        selector_week = selector.loc[
            selector["season"].eq(int(season)) & selector["week"].eq(int(week))
        ].copy()
        football_teams = set(final["team"].map(canon_team).dropna().astype(str))
        selector_teams = set(selector_week["team"].map(canon_team).dropna().astype(str))
        extra_selector = sorted(selector_teams - football_teams)
        if extra_selector:
            raise RuntimeError(
                f"{season} W{week:02d} selector contains teams absent from football universe: {extra_selector}"
            )

        baseline, candidate, ab_audit = _build_ab_states(state, final, selector_week)
        entitlement_after = pd.to_numeric(final["entitlement_tgt_share"], errors="raise").to_numpy(float)
        if not np.array_equal(entitlement_before, entitlement_after, equal_nan=True):
            raise RuntimeError(f"{season} W{week:02d} entitlement mutated")

        _, ml_pred = build_ml(player_logs, bundle.player_consensus, int(season), int(week))
        _, state_pred = build_state_predictions(player_logs, bundle.player_consensus, int(season), int(week))
        mcols = ["event_id", "team", "player", "player_clean_key", "position", "market", "season", "week"]
        market_rows = metrics[mcols].copy()
        market_rows["team"] = market_rows["team"].map(canon_team)
        market_rows["market"] = market_rows["market"].astype(str).str.lower()
        market_rows = _attach_component_projection(market_rows, ml_pred, "ml")
        market_rows = _attach_component_projection(market_rows, state_pred, "state")
        actual = build_actual_rows(player_logs, int(season), int(week))
        actual["team"] = actual["team"].map(canon_team)
        actual["market"] = actual["market"].astype(str).str.lower()
        all_joined = market_rows.merge(
            actual[["team", "player_clean_key", "market", "actual"]],
            on=["team", "player_clean_key", "market"],
            how="inner",
            validate="one_to_one",
        )
        if all_joined.empty:
            raise RuntimeError(f"{season} W{week:02d} no scored rows")
        all_joined["position_family"] = all_joined["position"].map(_pos)

        ent = final[["event_id", "team", "player_clean_key", "entitlement_tgt_share"]].drop_duplicates(
            ["event_id", "team", "player_clean_key"]
        )
        ent["team"] = ent["team"].map(canon_team)
        all_joined = all_joined.merge(
            ent,
            on=["event_id", "team", "player_clean_key"],
            how="left",
            validate="many_to_one",
        )
        if all_joined["entitlement_tgt_share"].isna().any():
            raise RuntimeError(f"{season} W{week:02d} could not attach entitlement to scored rows")

        sel_map = selector_week.set_index("team")["selector_c2_selected"].astype(bool).to_dict()
        all_joined["c2_selected_team"] = all_joined["team"].map(sel_map).fillna(False).astype(bool)

        scored = all_joined.loc[
            (
                all_joined["market"].isin(PRIMARY_MARKETS)
                & all_joined["position_family"].isin(POSITIONS)
            )
            | (
                all_joined["market"].eq("rush_rec_yards")
                & all_joined["position_family"].eq("RB")
            )
        ].copy()
        if scored.empty:
            raise RuntimeError(f"{season} W{week:02d} empty receiver score rows")

        bmc = []
        cmc = []
        for _, row in scored.iterrows():
            ba = lookup(baseline, row, row["market"])
            ca = lookup(candidate, row, row["market"])
            if ba is None or ca is None:
                raise RuntimeError(
                    f"{season} W{week:02d} missing simulation array player={row['player_clean_key']} market={row['market']}"
                )
            bmc.append(float(np.mean(np.asarray(ba, float))))
            cmc.append(float(np.mean(np.asarray(ca, float))))
        scored["baseline_mc_proj"] = bmc
        scored["candidate_mc_proj"] = cmc
        scored["baseline_proj"] = _ensemble_projection(scored, weights=weights, mc_col="baseline_mc_proj")
        scored["candidate_proj"] = _ensemble_projection(scored, weights=weights, mc_col="candidate_mc_proj")

        # Current RB Rush+Receiving Conservation V2 remains the final combo mean authority.
        bmap, bpayload = build_candidate_map(all_joined, baseline, weights)
        cmap, cpayload = build_candidate_map(all_joined, candidate, weights)
        sportsbook_inputs = max(
            sportsbook_inputs,
            int(bpayload.get("sportsbook_inputs_used", 0)),
            int(cpayload.get("sportsbook_inputs_used", 0)),
        )
        max_v2_gap = max(
            max_v2_gap,
            float(bpayload.get("max_pathwise_identity_gap", 0.0)),
            float(cpayload.get("max_pathwise_identity_gap", 0.0)),
        )
        for idx, row in scored.loc[
            scored["market"].eq("rush_rec_yards") & scored["position_family"].eq("RB")
        ].iterrows():
            key = (str(row["event_id"]), str(row["player_clean_key"]))
            if key in bmap:
                scored.at[idx, "baseline_proj"] = float(bmap[key]["target_mean"])
            if key in cmap:
                scored.at[idx, "candidate_proj"] = float(cmap[key]["target_mean"])

        numeric = scored[["actual", "baseline_proj", "candidate_proj"]].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
            raise RuntimeError(f"{season} W{week:02d} non-finite scored projection")

        details.append(scored)
        audits.append(
            {
                "season": int(season),
                "week": int(week),
                **ab_audit,
                "entitlement_mutations": 0,
                "sportsbook_inputs_used": int(
                    max(
                        int(bpayload.get("sportsbook_inputs_used", 0)),
                        int(cpayload.get("sportsbook_inputs_used", 0)),
                    )
                ),
                "rb_v2_max_pathwise_identity_gap": float(
                    max(
                        float(bpayload.get("max_pathwise_identity_gap", 0.0)),
                        float(cpayload.get("max_pathwise_identity_gap", 0.0)),
                    )
                ),
                "te_pool_gap": float(te_audit["team_te_pool_max_abs_gap"]),
                "wr_same_future_participation": int(wr_audit.get("same_future_participation", 0)),
            }
        )

    detail = _add_entitlement_quartile(pd.concat(details, ignore_index=True))
    audit = pd.DataFrame(audits)
    scope = {
        "selected_teams": int(audit["selected_teams"].sum()),
        "selector_eligible_team_games": int(audit["selector_eligible_teams"].sum()),
        "selector_ineligible_team_games": int(audit["selector_ineligible_teams"].sum()),
        "total_team_games": int(audit["total_teams"].sum()),
        "selected_semantic_violations": int(audit["selected_semantic_violations"].sum()),
        "max_selected_pass_identity_gap": float(audit["max_selected_pass_identity_gap"].max()),
        "max_qb_baseline_candidate_gap": float(audit["max_qb_baseline_candidate_gap"].max()),
        "max_unselected_receiver_gap": float(audit["max_unselected_receiver_gap"].max()),
        "max_rush_att_gap": float(audit["max_rush_att_gap"].max()),
        "max_rush_yards_gap": float(audit["max_rush_yards_gap"].max()),
        "entitlement_mutations": int(audit["entitlement_mutations"].sum()),
        "sportsbook_inputs_used": int(max(sportsbook_inputs, int(audit["sportsbook_inputs_used"].max()))),
        "rb_v2_max_pathwise_identity_gap": float(max(max_v2_gap, float(audit["rb_v2_max_pathwise_identity_gap"].max()))),
        "wr_same_future_participation": int(audit["wr_same_future_participation"].sum()),
    }
    return detail, audit, scope


def _build_scorecard(detail: pd.DataFrame) -> dict:
    pooled = {}
    seasons = {}
    for market in PRIMARY_MARKETS:
        pooled[market] = _macro(detail, market)
        seasons[market] = {}
        for season in (2024, 2025):
            seasons[market][str(season)] = _macro(detail.loc[detail["season"].eq(season)], market)

    q4 = detail.loc[
        detail["market"].eq("rec_yards")
        & detail["position_family"].isin(POSITIONS)
        & detail["entitlement_quartile"].eq("Q4_high")
    ]
    q4_score = _score(q4)

    rb_combo = {}
    for scope, frame in [
        ("pooled", detail),
        ("2024", detail.loc[detail["season"].eq(2024)]),
        ("2025", detail.loc[detail["season"].eq(2025)]),
    ]:
        rb_combo[scope] = _score(
            frame.loc[frame["market"].eq("rush_rec_yards") & frame["position_family"].eq("RB")]
        )
    return {
        "pooled": pooled,
        "seasons": seasons,
        "q4_rec_yards": q4_score,
        "rb_rush_rec_yards": rb_combo,
    }


def _gates(score: dict, scope24: dict, scope25: dict, selector_repro: dict) -> dict:
    rec = score["pooled"]["rec_yards"]
    receptions = score["pooled"]["receptions"]
    rec24 = score["seasons"]["rec_yards"]["2024"]
    rec25 = score["seasons"]["rec_yards"]["2025"]
    rcp24 = score["seasons"]["receptions"]["2024"]
    rcp25 = score["seasons"]["receptions"]["2025"]
    q4 = score["q4_rec_yards"]
    combo = score["rb_rush_rec_yards"]

    gates = {
        "phase_j_2025_selector_exact": selector_repro["decision_mismatches"] == 0
        and selector_repro["max_abs_delta_gap"] <= 1e-9,
        "qb_arrays_bit_identical": scope24["max_qb_baseline_candidate_gap"] <= TOL
        and scope25["max_qb_baseline_candidate_gap"] <= TOL,
        "unselected_receiver_arrays_bit_identical": scope24["max_unselected_receiver_gap"] <= TOL
        and scope25["max_unselected_receiver_gap"] <= TOL,
        "rush_att_bit_identical": scope24["max_rush_att_gap"] <= TOL and scope25["max_rush_att_gap"] <= TOL,
        "rush_yards_bit_identical": scope24["max_rush_yards_gap"] <= TOL
        and scope25["max_rush_yards_gap"] <= TOL,
        "target_entitlement_identical": scope24["entitlement_mutations"] == 0
        and scope25["entitlement_mutations"] == 0,
        "selected_semantics_exact": scope24["selected_semantic_violations"] == 0
        and scope25["selected_semantic_violations"] == 0,
        "selected_pass_identity_exact": scope24["max_selected_pass_identity_gap"] <= TOL
        and scope25["max_selected_pass_identity_gap"] <= TOL,
        "rb_v2_identity": scope24["rb_v2_max_pathwise_identity_gap"] <= TOL
        and scope25["rb_v2_max_pathwise_identity_gap"] <= TOL,
        "sportsbook_inputs_zero": scope24["sportsbook_inputs_used"] == 0
        and scope25["sportsbook_inputs_used"] == 0,
        "rec_yards_pooled_macro_mae_improves": rec["macro_candidate_mae"] < rec["macro_baseline_mae"],
        "rec_yards_2024_macro_mae_nonworse": rec24["macro_candidate_mae"] <= rec24["macro_baseline_mae"] + 1e-12,
        "rec_yards_2025_macro_mae_nonworse": rec25["macro_candidate_mae"] <= rec25["macro_baseline_mae"] + 1e-12,
        "rec_yards_no_position_pooled_regression_gt035": all(
            rec["positions"][p]["candidate_mae"] <= rec["positions"][p]["baseline_mae"] + 0.35
            for p in POSITIONS
        ),
        "rec_yards_pooled_macro_p90_nonworse": rec["macro_candidate_p90_ae"] <= rec["macro_baseline_p90_ae"] + 1e-12,
        "rec_yards_q4_mae_nonworse": q4["candidate_mae"] <= q4["baseline_mae"] + 1e-12,
        "rec_yards_q4_p90_nonworse": q4["candidate_p90_ae"] <= q4["baseline_p90_ae"] + 1e-12,
        "rec_yards_macro_miss40_guard": rec["macro_candidate_miss40"] <= rec["macro_baseline_miss40"] + 0.0025,
        "receptions_pooled_macro_mae_nonworse": receptions["macro_candidate_mae"] <= receptions["macro_baseline_mae"] + 1e-12,
        "receptions_2024_macro_mae_nonworse": rcp24["macro_candidate_mae"] <= rcp24["macro_baseline_mae"] + 1e-12,
        "receptions_2025_macro_mae_nonworse": rcp25["macro_candidate_mae"] <= rcp25["macro_baseline_mae"] + 1e-12,
        "receptions_no_position_pooled_regression_gt005": all(
            receptions["positions"][p]["candidate_mae"]
            <= receptions["positions"][p]["baseline_mae"] + 0.05
            for p in POSITIONS
        ),
        "receptions_pooled_macro_p90_nonworse": receptions["macro_candidate_p90_ae"]
        <= receptions["macro_baseline_p90_ae"] + 1e-12,
        "rb_combo_mae_nonworse_2024": combo["2024"]["candidate_mae"] <= combo["2024"]["baseline_mae"] + 1e-12,
        "rb_combo_mae_nonworse_2025": combo["2025"]["candidate_mae"] <= combo["2025"]["baseline_mae"] + 1e-12,
        "rb_combo_p90_nonworse_pooled": combo["pooled"]["candidate_p90_ae"] <= combo["pooled"]["baseline_p90_ae"] + 1e-12,
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
    ap.add_argument("--phase-c-all-rows", type=Path, required=True)
    ap.add_argument("--phase-j-state-casebook", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("data/model_ensemble_weights.csv"))
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    logs = _read(args.player_logs, "player logs")
    team = _read(args.team_weekly, "team weekly")
    schedule = _read(args.schedule, "schedule")
    injuries = _optional(args.injuries)
    weather = _optional(args.weather)
    weights = load_weights(args.weights)
    phase_c = _read(args.phase_c_all_rows, "Phase-C all rows")
    phase_j = _read(args.phase_j_state_casebook, "Phase-J state casebook")

    selector_source = _selector_source(phase_c)
    selector = _walk_forward_selector(selector_source)
    selector_repro = _verify_phase_j(selector, phase_j)

    te24 = _load_fold_params(args.te_coefficients, test_season=2024, features=TE_FEATURES, label="TE-R5P")
    te25 = _load_fold_params(args.te_coefficients, test_season=2025, features=TE_FEATURES, label="TE-R5P")
    wr24 = _load_fold_params(args.wr_coefficients, test_season=2024, features=WR_FEATURES, label="WR-R15")
    snaps, dup, _ = _load_participation_snaps()
    if dup > 0.01:
        raise RuntimeError(f"participation snap duplicate rate too high: {dup}")
    weeks = _parse_weeks(args.weeks)

    d24, a24, s24 = evaluate_season(
        season=2024,
        prior_season=2023,
        weeks=weeks,
        selector=selector,
        player_logs=logs,
        team_weekly=team,
        schedule=schedule,
        universe_dir=args.universe_2024,
        injuries=injuries,
        weather=weather,
        te_params=te24,
        wr_params=wr24,
        snaps=snaps,
        weights=weights,
        iterations=args.iterations,
    )
    d25, a25, s25 = evaluate_season(
        season=2025,
        prior_season=2024,
        weeks=weeks,
        selector=selector,
        player_logs=logs,
        team_weekly=team,
        schedule=schedule,
        universe_dir=args.universe_2025,
        injuries=injuries,
        weather=weather,
        te_params=te25,
        wr_params=None,
        snaps=snaps,
        weights=weights,
        iterations=args.iterations,
    )

    detail = pd.concat([d24, d25], ignore_index=True)
    score = _build_scorecard(detail)
    gates = _gates(score, s24, s25, selector_repro)
    qualified = all(gates.values())
    disposition = (
        "ONE_PASS_STATE_INTEGRATION_V1_QUALIFIED"
        if qualified
        else "ONE_PASS_STATE_INTEGRATION_V1_FAILED_CLOSED"
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
        "selector_reproduction": selector_repro,
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
    selector.to_csv(args.out_dir / "selector_casebook_2024_2025.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# One-Pass-State Integration V1 — Frozen Historical Result",
        "",
        f"Disposition: **{disposition}**",
        "",
        f"Phase-J reproduction max delta gap: `{selector_repro['max_abs_delta_gap']:.12g}`",
        f"Phase-J decision mismatches: `{selector_repro['decision_mismatches']}`",
        "",
    ]
    for market in PRIMARY_MARKETS:
        m = score["pooled"][market]
        lines += [
            f"## {market}",
            "",
            f"- macro MAE: `{m['macro_baseline_mae']:.6f} -> {m['macro_candidate_mae']:.6f}`",
            f"- macro p90 AE: `{m['macro_baseline_p90_ae']:.6f} -> {m['macro_candidate_p90_ae']:.6f}`",
            "",
        ]
        for pos in POSITIONS:
            p = m["positions"][pos]
            lines.append(
                f"- {pos}: MAE `{p['baseline_mae']:.6f} -> {p['candidate_mae']:.6f}`; "
                f"p90 `{p['baseline_p90_ae']:.6f} -> {p['candidate_p90_ae']:.6f}`"
            )
        lines.append("")
    combo = score["rb_rush_rec_yards"]
    lines += [
        "## RB rush+receiving",
        "",
        f"- 2024 MAE: `{combo['2024']['baseline_mae']:.6f} -> {combo['2024']['candidate_mae']:.6f}`",
        f"- 2025 MAE: `{combo['2025']['baseline_mae']:.6f} -> {combo['2025']['candidate_mae']:.6f}`",
        f"- pooled p90: `{combo['pooled']['baseline_p90_ae']:.6f} -> {combo['pooled']['candidate_p90_ae']:.6f}`",
        "",
        "## Frozen gates",
        "",
    ]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in gates.items()]
    (args.out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
