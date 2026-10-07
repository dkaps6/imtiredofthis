#!/usr/bin/env python3
"""Leakage-safe 2026 Weeks 1-4 all-player/all-position replay.

Frozen by docs/research/ALL_PLAYER_ALL_POSITION_REPLAY_V1_PLAN.md.

Primary unit: one individual player-game-market. Sportsbook data is optional
downstream overlap evidence only and is never read before all football
projections/distributions have been frozen.

Production authorities represented here:
- canonical historical MC + ML + State + frozen ensemble weights;
- explicit M38 target entitlement;
- TE-R5P then WR-R15 entitlement specialists before joint simulation;
- M89/M90 promoted QB football-only passing-yards mean;
- Week-1-only RB P3 rushing authority when the frozen P3 context contains the
  player/team; otherwise the generic calibrated mean;
- RB rush+receiving conservation V2 outside Week 1;
- frozen WR/TE target-depth distribution shadow, mean-neutral by construction;
- target-share trajectory remains exactly ineligible in Weeks 1-4.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import (
    _attach_component_projection,
    build_actual_rows,
    build_mc_predictions,
)
from scripts.backtest.historical_context import (
    assert_no_future_rows,
    build_historical_context_bundle,
)
from scripts.modeling.ensemble_v2 import apply_ensemble, load_weights
from scripts.modeling.ml_v2 import build_and_train as build_ml
from scripts.modeling.qb_pass_synthesis_v1 import (
    load_artifact as load_qb_artifact,
    predict_correction as predict_qb_correction,
)
from scripts.modeling.rb_pricing_adapter_v1 import (
    load_rb_context,
    lookup_rb_projection,
    rb_context_teams,
)
from scripts.modeling.rb_rush_rec_conservation_v2 import build_candidate_map
from scripts.modeling.state_v2 import build_state_predictions
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import apply_te_r5p_entitlement
from scripts.modeling.wr_r15_entitlement_adapter_v1 import apply_wr_r15_entitlement
from scripts.research.lock_player_target_depth_distribution_shadow_v1 import (
    DEPTH_ANCHOR,
    depth_scale,
    mean_neutral_distribution_shadow,
)
from scripts.backtest.run_m89_pregame_synthesis import add_history_features
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate
from scripts.simulation_v2 import MARKET_MAP, lookup

SEASON = 2026
PRIOR_SEASON = 2025
WEEKS = (1, 2, 3, 4)
POSITION_FAMILIES = {"QB", "RB", "FB", "WR", "TE"}
REQUIRED_MARKETS = {
    "QB": {"pass_yards"},
    "RB": {"rush_yards", "rec_yards", "receptions", "rush_rec_yards"},
    "FB": {"rush_yards", "rec_yards", "receptions", "rush_rec_yards"},
    "WR": {"rec_yards", "receptions"},
    "TE": {"rec_yards", "receptions"},
}
WR_POS = {"WR", "LWR", "RWR", "SWR"}
TE_POS = {"TE"}
POINT_KEYS = ["season", "week", "team", "player_clean_key", "market"]
SCORE_TOL = 1e-10


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _clean(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def _pkey(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def _pos(v) -> str:
    p = str(v or "").upper().strip()
    if p in {"HB", "TB"} or p.startswith("RB"):
        return "RB"
    if p.startswith("FB"):
        return "FB"
    if p.startswith("QB"):
        return "QB"
    if p.startswith("WR") or p in {"LWR", "RWR", "SWR"}:
        return "WR"
    if p.startswith("TE"):
        return "TE"
    return p


def _num(v, default=np.nan) -> float:
    try:
        z = float(v)
        return z if np.isfinite(z) else float(default)
    except Exception:
        return float(default)


def _canonical_market(v) -> str:
    s = str(v or "").strip().lower()
    return MARKET_MAP.get(s, s)


def _required_market(position_family: str, market: str) -> bool:
    return market in REQUIRED_MARKETS.get(position_family, set())


def _historical_outcomes(sims, row: pd.Series) -> np.ndarray | None:
    market = _canonical_market(row.get("market"))
    out = lookup(sims, row, market)
    if out is None or len(out) == 0:
        return None
    arr = np.asarray(out, dtype=float)
    if market == "pass_yards":
        attempt_rate = _num(row.get("mc_pass_attempts_per_dropback"))
        share = _num(row.get("qb_pass_att_share"))
        if np.isfinite(attempt_rate):
            arr = arr * float(np.clip(attempt_rate, 0.50, 1.00))
        if np.isfinite(share):
            arr = arr * float(np.clip(share, 0.0, 1.0))
    return arr


def _build_specialist_state(metrics: pd.DataFrame, *, iterations: int, seed: int):
    player_cols = ["event_id", "team", "player_clean_key"]
    players = metrics.sort_values(player_cols).drop_duplicates(player_cols, keep="last").copy()
    if players.duplicated(player_cols).any():
        raise RuntimeError("specialist reconstruction player universe is not unique")

    base, _ = materialize_target_entitlement(players)
    te_final, _, te_audit = apply_te_r5p_entitlement(base)
    final, _, wr_audit = apply_wr_r15_entitlement(te_final)
    sims = explicit_simulate(final, iterations=int(iterations), seed=int(seed))

    team_before = base.groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
    team_after = final.groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
    max_gap = float((team_after - team_before).abs().max()) if len(team_before) else 0.0
    if max_gap > SCORE_TOL:
        raise RuntimeError(f"WR/TE specialists changed team player target mass: {max_gap}")
    return final, sims, {
        "te": te_audit,
        "wr": wr_audit,
        "max_team_entitlement_gap": max_gap,
    }


def _attach_ml_state(
    metrics: pd.DataFrame,
    bundle,
    player_logs: pd.DataFrame,
    *,
    week: int,
) -> pd.DataFrame:
    _, ml_pred = build_ml(player_logs, bundle.player_consensus, SEASON, int(week))
    _, state_pred = build_state_predictions(player_logs, bundle.player_consensus, SEASON, int(week))
    out = metrics.copy()
    out = _attach_component_projection(out, ml_pred, "ml")
    out = _attach_component_projection(out, state_pred, "state")
    return out


def _recompute_mc_from_specialist(metrics: pd.DataFrame, sims) -> pd.DataFrame:
    out = metrics.copy()
    vals = []
    missing = []
    for idx, row in out.iterrows():
        arr = _historical_outcomes(sims, row)
        if arr is None:
            vals.append(np.nan)
            missing.append((idx, row.get("player"), row.get("market")))
        else:
            vals.append(float(np.mean(arr)))
    out["mc_proj"] = vals
    if missing:
        sample = missing[:20]
        raise RuntimeError(f"specialist simulation missing player-market arrays: {sample}")
    return out


def _controlled_environment_map() -> dict[tuple[int, int, str], float]:
    try:
        import nflreadpy as nfl
        raw = nfl.load_schedules(seasons=[SEASON])
        s = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
    except Exception:
        return {}
    s.columns = [str(c).strip().lower() for c in s.columns]
    if "game_type" in s.columns:
        reg = s.loc[s["game_type"].astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            s = reg
    out: dict[tuple[int, int, str], float] = {}
    for _, g in s.iterrows():
        wk = int(_num(g.get("week"), -1))
        if wk not in WEEKS:
            continue
        roof = str(g.get("roof", "") or "").lower()
        if roof:
            controlled = float(int(any(x in roof for x in ("dome", "closed", "indoor"))))
        else:
            controlled = np.nan
        for c in ("home_team", "away_team"):
            team = canon_team(g.get(c))
            if team:
                out[(SEASON, wk, team)] = controlled
    return out


def _apply_qb_synthesis(
    frame: pd.DataFrame,
    *,
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    controlled_map: dict[tuple[int, int, str], float],
) -> pd.DataFrame:
    out = frame.copy()
    qb_mask = out["market"].eq("pass_yards") & out["position_family"].eq("QB")
    out["qb_synthesis_applied"] = False
    out["qb_synthesis_version"] = ""
    out["qb_synthesis_correction"] = np.nan
    out["point_projection_pre_position_authority"] = out["ensemble_proj"].astype(float)
    out["projection_mean"] = out["ensemble_proj"].astype(float)
    if not qb_mask.any():
        raise RuntimeError("all-player replay produced zero QB pass_yards rows")

    q = out.loc[qb_mask].copy()
    # add_history_features resets its row index. Carry the source row explicitly
    # so every synthesized QB result is written back to the exact player-game.
    q["_source_index"] = q.index.astype(int)
    q["base_proj"] = pd.to_numeric(q["ensemble_proj"], errors="coerce")
    comps = q[["mc_proj", "ml_proj", "state_proj"]].apply(pd.to_numeric, errors="coerce")
    q["component_sd"] = comps.std(axis=1, skipna=True)
    q["component_range"] = comps.max(axis=1, skipna=True) - comps.min(axis=1, skipna=True)
    q["pred_attempts"] = pd.to_numeric(q.get("mc_expected_pass_attempts"), errors="coerce")
    q["pred_ypa"] = q["mc_proj"] / q["pred_attempts"].replace(0, np.nan)

    # add_history_features is strictly-prior by construction.
    enriched = add_history_features(q, team_weekly, player_logs)
    enriched["controlled_environment"] = [
        controlled_map.get((SEASON, int(w), canon_team(t)), np.nan)
        for w, t in zip(enriched["week"], enriched["team"])
    ]

    artifact = load_qb_artifact()
    feature_names = list(artifact["feature_contract"])
    if "base_proj" not in feature_names:
        raise RuntimeError("QB synthesis artifact lost base_proj feature")

    for idx, r in enriched.iterrows():
        features = {name: r.get(name, np.nan) for name in feature_names}
        pred, corr, version = predict_qb_correction(features, artifact=artifact)
        src_idx = int(r["_source_index"])
        out.loc[src_idx, "projection_mean"] = float(pred)
        out.loc[src_idx, "qb_synthesis_applied"] = True
        out.loc[src_idx, "qb_synthesis_version"] = str(version)
        out.loc[src_idx, "qb_synthesis_correction"] = float(corr)

    if not out.loc[qb_mask, "qb_synthesis_applied"].all():
        raise RuntimeError("not every QB pass_yards row consumed promoted QB synthesis")
    return out


def _apply_rb_authorities(
    frame: pd.DataFrame,
    *,
    sims,
    weights: pd.DataFrame,
    rb_context: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()
    out["rb_p3_applied"] = False
    out["rb_p3_route"] = ""
    out["rb_rush_rec_v2_applied"] = False
    p3_teams = rb_context_teams(rb_context) if rb_context is not None else set()

    # Week-1 P3 rush_yards: only the frozen context's own team scope is eligible.
    mask = (
        out["week"].eq(1)
        & out["position_family"].isin({"RB", "FB"})
        & out["market"].eq("rush_yards")
    )
    for idx, row in out.loc[mask].iterrows():
        if str(row.get("team")) not in p3_teams:
            continue
        meta = lookup_rb_projection(row, rb_context)
        out.loc[idx, "projection_mean"] = float(meta["rb_synthesis_proj"])
        out.loc[idx, "rb_p3_applied"] = True
        out.loc[idx, "rb_p3_route"] = str(meta["rb_synthesis_route"])

    # Outside Week 1 production uses V2 pathwise rush+rec conservation. Build
    # the candidate from the same specialist simulation and component columns.
    candidate_map, payload = build_candidate_map(out, sims, weights)
    if int(payload.get("week1_rows_changed", 0)) != 0:
        raise RuntimeError("RB rush+rec V2 illegally changed Week 1")

    for idx, row in out.iterrows():
        if (
            int(row["week"]) == 1
            or row["position_family"] != "RB"
            or row["market"] != "rush_rec_yards"
        ):
            continue
        key = (str(row["event_id"]), str(row["player_clean_key"]))
        meta = candidate_map.get(key)
        if meta is not None:
            out.loc[idx, "projection_mean"] = float(meta["target_mean"])
            out.loc[idx, "rb_rush_rec_v2_applied"] = True

    # Week-1 rush+rec exact production ordering:
    #   1) scale the raw rush MC path to the frozen P3 rush mean;
    #   2) add the *raw* receiving MC path;
    #   3) treat that conserved path as the combo MC component;
    #   4) apply the frozen rush_rec_yards ensemble weights.
    # run_pricing_v2 then mean-aligns that conserved path to this ensemble mean.
    week1_combo = out.loc[
        out["week"].eq(1)
        & out["position_family"].isin({"RB", "FB"})
        & out["market"].eq("rush_rec_yards")
    ].copy()
    for idx, row in week1_combo.iterrows():
        if str(row.get("team")) not in p3_teams:
            continue
        rush = lookup(sims, row, "rush_yards")
        rec = lookup(sims, row, "rec_yards")
        if rush is None or rec is None:
            raise RuntimeError(
                f"Week-1 P3 rush+rec missing component draws player={row.get('player')} team={row.get('team')}"
            )
        rush = np.asarray(rush, dtype=float)
        rec = np.asarray(rec, dtype=float)
        if len(rush) != len(rec) or not np.isfinite(rush).all() or not np.isfinite(rec).all():
            raise RuntimeError("Week-1 P3 rush+rec component arrays are invalid")
        meta = lookup_rb_projection(row, rb_context)
        p3_mean = float(meta["rb_synthesis_proj"])
        raw_rush_mean = float(np.mean(rush))
        if raw_rush_mean > 0:
            scaled_rush = rush * (p3_mean / raw_rush_mean)
        elif abs(p3_mean) <= SCORE_TOL:
            scaled_rush = np.zeros_like(rush)
        else:
            raise RuntimeError("cannot align positive Week-1 P3 rush mean from zero raw rush distribution")
        conserved = scaled_rush + rec
        combo_mc = float(np.mean(conserved))
        component = pd.DataFrame([{
            "market": "rush_rec_yards",
            "mc_proj": combo_mc,
            "ml_proj": row.get("ml_proj"),
            "state_proj": row.get("state_proj"),
        }])
        ens = apply_ensemble(component, weights=weights).iloc[0]
        out.loc[idx, "mc_proj"] = combo_mc
        out.loc[idx, "ensemble_proj"] = float(ens["ensemble_proj"])
        out.loc[idx, "ensemble_status"] = str(ens["ensemble_status"])
        out.loc[idx, "projection_mean"] = float(ens["ensemble_proj"])
        out.loc[idx, "rb_p3_applied"] = True
        out.loc[idx, "rb_p3_route"] = "WEEK1_P3_PATHWISE_CONSERVATION_THEN_COMBO_ENSEMBLE"

    return out


def _actual_lookup(player_logs: pd.DataFrame, week: int) -> dict[tuple[str, str, str], tuple[float, float, str]]:
    actual = build_actual_rows(player_logs, SEASON, int(week))
    out: dict[tuple[str, str, str], tuple[float, float, str]] = {}
    for r in actual.itertuples(index=False):
        out[(canon_team(r.team), str(r.player_clean_key), str(r.market))] = (
            float(r.actual),
            _num(r.actual_opportunities),
            "NFLVERSE_WEEKLY_STATS",
        )
    return out


def _attach_actuals(
    frame: pd.DataFrame,
    *,
    player_logs: pd.DataFrame,
    pregame_universe: pd.DataFrame,
    week: int,
) -> pd.DataFrame:
    out = frame.copy()
    amap = _actual_lookup(player_logs, week)
    roster_keys = {
        (canon_team(t), _pkey(p))
        for t, p in zip(pregame_universe["team"], pregame_universe["player"])
    }
    actuals, opps, sources = [], [], []
    for _, r in out.iterrows():
        key = (canon_team(r["team"]), str(r["player_clean_key"]), str(r["market"]))
        if key in amap:
            a, o, src = amap[key]
        elif (canon_team(r["team"]), str(r["player_clean_key"])) in roster_keys:
            # The weekly stats feed omits many rostered players who recorded no
            # box-score activity. For required yardage/count markets, that is a
            # verified zero outcome, not an unknown row.
            a, o, src = 0.0, 0.0, "PREGAME_ROSTER_NO_WEEKLY_STAT_ROW_ZERO"
        else:
            a, o, src = np.nan, np.nan, "UNRESOLVED"
        actuals.append(a); opps.append(o); sources.append(src)
    out["actual"] = actuals
    out["actual_opportunities"] = opps
    out["actual_source"] = sources
    return out


def _target_events() -> pd.DataFrame:
    import nflreadpy as nfl
    frames = []
    for season in (2022, 2023, 2024, 2025, 2026):
        raw = nfl.load_pbp(seasons=[season])
        x = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
        x.columns = [str(c).strip().lower() for c in x.columns]
        if "season_type" in x.columns:
            reg = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
            if not reg.empty:
                x = reg
        for c in ("season", "week", "game_id", "receiver_player_id", "pass_attempt", "sack", "two_point_attempt", "air_yards"):
            if c not in x.columns:
                x[c] = np.nan
        x["season"] = pd.to_numeric(x["season"], errors="coerce").fillna(season).astype(int)
        x["week"] = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
        x["receiver_id"] = x["receiver_player_id"].map(_clean)
        target = (
            pd.to_numeric(x["pass_attempt"], errors="coerce").fillna(0).eq(1)
            & ~pd.to_numeric(x["sack"], errors="coerce").fillna(0).eq(1)
            & ~pd.to_numeric(x["two_point_attempt"], errors="coerce").fillna(0).eq(1)
            & x["receiver_id"].ne("")
        )
        x["air_yards"] = pd.to_numeric(x["air_yards"], errors="coerce")
        frames.append(x.loc[target & x["air_yards"].notna(), ["season", "week", "game_id", "receiver_id", "air_yards"]])
    return pd.concat(frames, ignore_index=True, sort=False)


def _identity_history(player_logs: pd.DataFrame) -> pd.DataFrame:
    x = player_logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    id_col = next((c for c in ("player_id", "gsis_id", "nflverse_id") if c in x.columns), None)
    if id_col is None:
        return pd.DataFrame(columns=["season", "week", "player_clean_key", "receiver_id"])
    x["receiver_id"] = x[id_col].map(_clean)
    x["player_clean_key"] = x.get("player_clean_key", x["player"]).astype(str)
    return x[["season", "week", "player_clean_key", "receiver_id"]].copy()


def _resolve_receiver_id(identity: pd.DataFrame, player_key: str, week: int) -> tuple[str, str]:
    if identity.empty:
        return "", "UNRESOLVED"
    s = pd.to_numeric(identity["season"], errors="coerce")
    w = pd.to_numeric(identity["week"], errors="coerce")
    q = identity.loc[
        identity["player_clean_key"].astype(str).eq(str(player_key))
        & (s.lt(SEASON) | (s.eq(SEASON) & w.lt(int(week))))
        & identity["receiver_id"].astype(str).ne("")
    ].copy()
    if q.empty:
        return "", "UNRESOLVED"
    q["season"] = pd.to_numeric(q["season"], errors="coerce")
    q["week"] = pd.to_numeric(q["week"], errors="coerce")
    q = q.sort_values(["season", "week"])
    ids = q["receiver_id"].astype(str).drop_duplicates().tolist()
    return str(q.iloc[-1]["receiver_id"]), "STRICT_PRIOR_PLAYER_LOG_ID" if ids else "UNRESOLVED"


def _depth_feature(events: pd.DataFrame, receiver_id: str, week: int) -> dict | None:
    if not receiver_id:
        return None
    s = pd.to_numeric(events["season"], errors="coerce")
    w = pd.to_numeric(events["week"], errors="coerce")
    h = events.loc[
        events["receiver_id"].astype(str).eq(str(receiver_id))
        & (s.lt(SEASON) | (s.eq(SEASON) & w.lt(int(week))))
    ].copy()
    if h.empty:
        return None
    games = (
        h[["season", "week", "game_id"]]
        .drop_duplicates()
        .sort_values(["season", "week", "game_id"], kind="mergesort")
        .tail(8)
    )
    if len(games) < 4:
        return None
    keys = set(zip(games["season"].astype(int), games["week"].astype(int), games["game_id"].astype(str)))
    mask = [
        (int(a), int(b), str(c)) in keys
        for a, b, c in zip(h["season"], h["week"], h["game_id"])
    ]
    q = h.loc[mask].copy()
    air = pd.to_numeric(q["air_yards"], errors="coerce").dropna().to_numpy(float)
    if len(air) < 10:
        return None
    max_season = int(pd.to_numeric(q["season"], errors="coerce").max())
    max_week = int(pd.to_numeric(q.loc[pd.to_numeric(q["season"], errors="coerce").eq(max_season), "week"], errors="coerce").max())
    if max_season > SEASON or (max_season == SEASON and max_week >= int(week)):
        raise RuntimeError("target-depth feature crossed target-week boundary")
    return {
        "prior_receiver_games": int(len(games)),
        "prior_finite_air_targets": int(len(air)),
        "prior8_target_depth_sd": float(np.std(air, ddof=0)),
        "feature_max_season": max_season,
        "feature_max_week": max_week,
    }


def _empirical_crps(draws: np.ndarray, actual: float) -> float:
    x = np.sort(np.asarray(draws, dtype=float))
    n = len(x)
    if n == 0:
        return np.nan
    first = float(np.mean(np.abs(x - float(actual))))
    coeff = (2.0 * np.arange(1, n + 1) - n - 1.0)
    half_pair = float(np.sum(coeff * x) / (n * n))
    return first - half_pair


def _distribution_metrics(draws: np.ndarray, actual: float, prefix: str) -> dict:
    x = np.asarray(draws, dtype=float)
    row = {
        f"{prefix}_crps": _empirical_crps(x, actual),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        f"{prefix}_pit": float((np.sum(x < actual) + 0.5 * np.sum(x == actual)) / len(x)),
    }
    for level in (50, 80, 90):
        alpha = (100 - level) / 200.0
        lo, hi = np.quantile(x, [alpha, 1.0 - alpha])
        row[f"{prefix}_ci{level}_lo"] = float(lo)
        row[f"{prefix}_ci{level}_hi"] = float(hi)
        row[f"{prefix}_ci{level}_width"] = float(hi - lo)
        row[f"{prefix}_ci{level}_covered"] = bool(lo <= actual <= hi)
    return row


def _build_distribution_rows(
    point: pd.DataFrame,
    sims,
    *,
    events: pd.DataFrame,
    identity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dist_rows = []
    feat_rows = []
    wrte = point.loc[
        point["position_family"].isin({"WR", "TE"})
        & point["market"].eq("rec_yards")
    ].copy()
    if wrte.empty:
        raise RuntimeError("replay produced zero WR/TE rec_yards rows")

    for _, row in wrte.iterrows():
        week = int(row["week"])
        rid, identity_route = _resolve_receiver_id(identity, str(row["player_clean_key"]), week)
        feat = _depth_feature(events, rid, week)
        available = feat is not None
        sd = float(feat["prior8_target_depth_sd"]) if available else np.nan
        scale = depth_scale(str(row["position_family"]), sd)
        raw = _historical_outcomes(sims[(week, str(row["event_id"]))], row)
        if raw is None:
            raise RuntimeError(f"missing empirical rec_yards draws for {row['player']} W{week}")
        exact_mean = float(row["projection_mean"])
        baseline, shadow = mean_neutral_distribution_shadow(raw, exact_mean=exact_mean, scale=scale)
        mean_gap = float(np.mean(shadow) - np.mean(baseline))
        if abs(float(np.mean(baseline)) - exact_mean) > SCORE_TOL or abs(float(np.mean(shadow)) - exact_mean) > SCORE_TOL:
            raise RuntimeError("target-depth distribution shadow violated exact football mean")
        actual = float(row["actual"])
        rec = {
            "season": SEASON, "week": week, "event_id": row["event_id"],
            "team": row["team"], "opponent": row["opponent"], "player": row["player"],
            "player_clean_key": row["player_clean_key"], "position_family": row["position_family"],
            "actual": actual, "exact_football_mean": exact_mean,
            "receiver_id": rid, "identity_route": identity_route,
            "feature_available": bool(available), "prior8_target_depth_sd": sd,
            "position_depth_sd_anchor": float(DEPTH_ANCHOR[str(row["position_family"])]),
            "depth_distribution_scale": float(scale), "mean_gap_shadow_minus_baseline": mean_gap,
            "absolute_point_error": abs(exact_mean - actual),
        }
        rec.update(_distribution_metrics(baseline, actual, "baseline"))
        rec.update(_distribution_metrics(shadow, actual, "shadow"))
        rec["crps_improvement"] = rec["baseline_crps"] - rec["shadow_crps"]
        dist_rows.append(rec)

        feat_rows.append({
            "season": SEASON, "week": week, "event_id": row["event_id"], "team": row["team"],
            "player": row["player"], "player_clean_key": row["player_clean_key"],
            "position_family": row["position_family"],
            "target_share_trajectory_status": "FROZEN_FEATURE_NOT_YET_ELIGIBLE",
            "target_share_trajectory_eligible": False,
            "target_share_trajectory_reason": "REQUIRES_FOUR_PRIOR_SAME_SEASON_TEAM_GAMES",
            "receiver_id": rid, "identity_route": identity_route,
            "target_depth_feature_available": bool(available),
            "prior_receiver_games": int(feat["prior_receiver_games"]) if available else np.nan,
            "prior_finite_air_targets": int(feat["prior_finite_air_targets"]) if available else np.nan,
            "prior8_target_depth_sd": sd,
            "feature_max_season": int(feat["feature_max_season"]) if available else np.nan,
            "feature_max_week": int(feat["feature_max_week"]) if available else np.nan,
            "depth_distribution_scale": float(scale),
            "same_or_future_feature_violation": False,
        })
    return pd.DataFrame(dist_rows), pd.DataFrame(feat_rows)


def _add_depth_scale_quantiles(dist: pd.DataFrame) -> pd.DataFrame:
    """Attach pooled diagnostic quartiles to feature-available depth scales.

    This is reporting only: quartile membership is never used to change a
    projection, distribution, threshold, or promotion decision.
    """
    out = dist.copy()
    out["depth_scale_quantile"] = "UNAVAILABLE"
    available = out["feature_available"].fillna(False).astype(bool)
    if available.any():
        ranks = out.loc[available, "depth_distribution_scale"].rank(method="first", pct=True)
        labels = pd.cut(
            ranks,
            bins=[0.0, 0.25, 0.50, 0.75, 1.0],
            labels=["Q1_LOW", "Q2", "Q3", "Q4_HIGH"],
            include_lowest=True,
        )
        out.loc[available, "depth_scale_quantile"] = labels.astype(str).to_numpy()
    return out


def _depth_quantile_summary(dist: pd.DataFrame) -> list[dict]:
    usable = dist.loc[dist["feature_available"].fillna(False).astype(bool)].copy()
    rows = []
    for label, g in usable.groupby("depth_scale_quantile", sort=False):
        rows.append({
            "depth_scale_quantile": str(label),
            "rows": int(len(g)),
            "scale_min": float(g["depth_distribution_scale"].min()),
            "scale_max": float(g["depth_distribution_scale"].max()),
            "mean_absolute_point_error": float(g["absolute_point_error"].mean()),
            "baseline_crps_mean": float(g["baseline_crps"].mean()),
            "shadow_crps_mean": float(g["shadow_crps"].mean()),
            "crps_mean_improvement": float(g["crps_improvement"].mean()),
        })
    return rows


def _score_point_rows(point: pd.DataFrame) -> pd.DataFrame:
    x = point.copy()
    x["error"] = pd.to_numeric(x["projection_mean"], errors="coerce") - pd.to_numeric(x["actual"], errors="coerce")
    x["absolute_error"] = x["error"].abs()
    x["squared_error"] = x["error"] ** 2
    return x


def _group_point_summary(point: pd.DataFrame) -> list[dict]:
    rows = []
    for keys, g in point.groupby(["position_family", "market"], dropna=False):
        pos, market = keys
        e = pd.to_numeric(g["error"], errors="coerce").dropna()
        rows.append({
            "position_family": str(pos), "market": str(market), "rows": int(len(e)),
            "mae": float(e.abs().mean()) if len(e) else np.nan,
            "median_absolute_error": float(e.abs().median()) if len(e) else np.nan,
            "signed_bias": float(e.mean()) if len(e) else np.nan,
            "rmse": float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan,
        })
    return rows


def _read_live_board_rows(root: Path | None) -> pd.DataFrame:
    if root is None or not root.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(root.rglob("*.csv")):
        try:
            x = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        x.columns = [str(c).strip().lower() for c in x.columns]
        if "model_proj" not in x.columns or "player" not in x.columns:
            continue
        week_col = next((c for c in ("week", "nfl_week") if c in x.columns), None)
        market_col = next((c for c in ("market", "source_market") if c in x.columns), None)
        if week_col is None or market_col is None:
            continue
        x["week"] = pd.to_numeric(x[week_col], errors="coerce")
        x = x.loc[x["week"].isin(WEEKS)].copy()
        if x.empty:
            continue
        x["market"] = x[market_col].map(_canonical_market)
        x["player_clean_key"] = x.get("player_clean_key", x["player"]).map(_pkey)
        x["team"] = x.get("team", "").map(canon_team) if "team" in x.columns else ""
        line_col = next((c for c in ("line", "source_line", "vegas_line") if c in x.columns), None)
        actual_col = next((c for c in ("actual", "actual_value", "actual_yards") if c in x.columns), None)
        keep = ["week", "team", "player", "player_clean_key", "market", "model_proj"]
        if line_col:
            x["live_selected_line"] = pd.to_numeric(x[line_col], errors="coerce"); keep.append("live_selected_line")
        if actual_col:
            x["live_actual"] = pd.to_numeric(x[actual_col], errors="coerce"); keep.append("live_actual")
        x["live_source_file"] = str(path); keep.append("live_source_file")
        frames.append(x[keep])
    if not frames:
        return pd.DataFrame()
    z = pd.concat(frames, ignore_index=True, sort=False)
    z["model_proj"] = pd.to_numeric(z["model_proj"], errors="coerce")
    return z.drop_duplicates(["week", "team", "player_clean_key", "market", "live_source_file"])


def _live_overlap(point: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "season", "week", "team", "player", "player_clean_key", "position_family",
        "market", "projection_mean", "actual",
    ]
    base = point[cols].copy()
    if live.empty:
        base["live_overlap"] = False
        base["sportsbook_fields_used_upstream"] = False
        base["live_source_file"] = ""
        return base.iloc[0:0].copy()
    merged = base.merge(
        live,
        on=["week", "team", "player_clean_key", "market"],
        how="inner",
        suffixes=("_replay", "_live"),
    )
    if merged.empty:
        return pd.DataFrame(columns=[*cols, "live_overlap", "sportsbook_fields_used_upstream"])
    merged["live_overlap"] = True
    merged["sportsbook_fields_used_upstream"] = False
    if "live_selected_line" in merged.columns:
        merged["replay_closer_than_live_selected_line"] = (
            (merged["projection_mean"] - merged["actual"]).abs()
            < (merged["live_selected_line"] - merged["actual"]).abs()
        )
    return merged


def run_replay(
    *,
    player_logs_path: Path,
    team_weekly_path: Path,
    schedule_path: Path,
    universe_dir: Path,
    rb_week1_context_path: Path,
    out_dir: Path,
    iterations: int,
    live_board_root: Path | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    player_logs = _read(player_logs_path, "player logs")
    team_weekly = _read(team_weekly_path, "team weekly history")
    schedule = _read(schedule_path, "schedule history")
    assert_no_future_rows(
        player_logs.loc[
            (pd.to_numeric(player_logs["season"], errors="coerce").lt(SEASON))
            | (
                pd.to_numeric(player_logs["season"], errors="coerce").eq(SEASON)
                & pd.to_numeric(player_logs["week"], errors="coerce").lt(max(WEEKS) + 1)
            )
        ],
        SEASON,
        max(WEEKS) + 1,
        "replay_player_history_max_boundary",
    )
    weights = load_weights()
    if weights.empty:
        raise RuntimeError("frozen ensemble weights are missing")
    rb_context = load_rb_context(rb_week1_context_path)
    controlled_map = _controlled_environment_map()

    point_parts = []
    sims_by_week_game: dict[tuple[int, str], object] = {}
    specialist_audits = {}
    for week in WEEKS:
        universe_path = universe_dir / f"{SEASON}_week_{week:02d}.csv"
        universe = _read(universe_path, f"pregame universe {SEASON} W{week}")
        bundle = build_historical_context_bundle(
            player_logs=player_logs,
            team_weekly=team_weekly,
            pregame_universe=universe,
            schedule=schedule,
            season=SEASON,
            week=week,
            prior_season=PRIOR_SEASON,
        )
        assert_no_future_rows(bundle.player_history, SEASON, week, f"W{week} player_history")
        assert_no_future_rows(bundle.team_history, SEASON, week, f"W{week} team_history")

        seed = 42 + week
        metrics = build_mc_predictions(bundle, iterations=int(iterations), seed=seed)
        metrics = _attach_ml_state(metrics, bundle, player_logs, week=week)
        _, sims, spec_audit = _build_specialist_state(metrics, iterations=int(iterations), seed=seed)
        metrics = _recompute_mc_from_specialist(metrics, sims)
        metrics["market"] = metrics["market"].map(_canonical_market)
        metrics["position_family"] = metrics["position"].map(_pos)
        metrics = metrics.loc[
            metrics.apply(lambda r: _required_market(str(r["position_family"]), str(r["market"])), axis=1)
        ].copy()
        if metrics.empty:
            raise RuntimeError(f"W{week}: zero required player-market rows")

        metrics = apply_ensemble(metrics, weights=weights)
        metrics["season"] = SEASON
        metrics["week"] = week
        metrics = _apply_qb_synthesis(
            metrics,
            player_logs=player_logs,
            team_weekly=team_weekly,
            controlled_map=controlled_map,
        )
        metrics = _apply_rb_authorities(
            metrics,
            sims=sims,
            weights=weights,
            rb_context=rb_context,
        )
        metrics = _attach_actuals(metrics, player_logs=player_logs, pregame_universe=universe, week=week)
        metrics["prediction_cutoff"] = f"{SEASON}-W{week:02d} pregame"
        metrics["sportsbook_inputs_used_upstream"] = False
        metrics["target_share_trajectory_status"] = "FROZEN_FEATURE_NOT_YET_ELIGIBLE"
        metrics["rb_week5_room_allocation_shadow_applied"] = False

        if metrics["projection_mean"].isna().any() or metrics["actual"].isna().any():
            bad = metrics.loc[
                metrics["projection_mean"].isna() | metrics["actual"].isna(),
                ["player", "team", "market", "projection_mean", "actual", "actual_source"],
            ].head(20)
            raise RuntimeError(f"W{week}: unresolved point rows:\n{bad.to_string(index=False)}")
        if metrics["rb_week5_room_allocation_shadow_applied"].any():
            raise RuntimeError("Week-5 RB room allocation shadow leaked into W1-4 replay")

        # Retain simulation object by week/game for WR/TE empirical scoring.
        for event_id in metrics["event_id"].astype(str).unique():
            sims_by_week_game[(week, event_id)] = sims
        point_parts.append(metrics)
        specialist_audits[str(week)] = spec_audit
        print(
            f"[all-player-replay] W{week} rows={len(metrics)} players={metrics['player_clean_key'].nunique()} "
            f"QB={int(metrics.position_family.eq('QB').sum())} RB/FB={int(metrics.position_family.isin(['RB','FB']).sum())} "
            f"WR={int(metrics.position_family.eq('WR').sum())} TE={int(metrics.position_family.eq('TE').sum())}"
        )

    point = pd.concat(point_parts, ignore_index=True, sort=False)
    point = _score_point_rows(point)
    if set(point["week"].unique()) != set(WEEKS):
        raise RuntimeError("replay did not cover all Weeks 1-4")
    if point.duplicated(POINT_KEYS).any():
        bad = point.loc[point.duplicated(POINT_KEYS, keep=False), POINT_KEYS].head(20)
        raise RuntimeError(f"duplicate point scoreboard identities:\n{bad.to_string(index=False)}")

    # Build target-depth evidence only after all point means are frozen.
    events = _target_events()
    identity = _identity_history(player_logs)
    dist, feature = _build_distribution_rows(point, sims_by_week_game, events=events, identity=identity)
    dist = _add_depth_scale_quantiles(dist)
    if not feature["target_share_trajectory_eligible"].eq(False).all():
        raise RuntimeError("trajectory illegally became eligible in Weeks 1-4")
    if feature["same_or_future_feature_violation"].any():
        raise RuntimeError("target-depth feature leakage flag was raised")
    if float(dist["mean_gap_shadow_minus_baseline"].abs().max()) > SCORE_TOL:
        raise RuntimeError("target-depth shadow changed point mean")

    live = _read_live_board_rows(live_board_root)
    overlap = _live_overlap(point, live)
    if not overlap.empty and overlap["sportsbook_fields_used_upstream"].any():
        raise RuntimeError("sportsbook field leaked upstream")

    point_cols = [
        "season", "week", "event_id", "team", "opponent", "player", "player_clean_key",
        "position_family", "market", "projection_mean", "actual", "actual_source",
        "actual_opportunities", "error", "absolute_error", "squared_error",
        "mc_proj", "ml_proj", "state_proj", "ensemble_proj", "ensemble_status",
        "qb_synthesis_applied", "qb_synthesis_version", "qb_synthesis_correction",
        "rb_p3_applied", "rb_p3_route", "rb_rush_rec_v2_applied",
        "target_share_trajectory_status", "rb_week5_room_allocation_shadow_applied",
        "prediction_cutoff", "sportsbook_inputs_used_upstream",
    ]
    for c in point_cols:
        if c not in point.columns:
            point[c] = np.nan
    point[point_cols].sort_values(POINT_KEYS).to_csv(out_dir / "all_players_point_scoreboard.csv", index=False)
    dist.sort_values(["week", "position_family", "team", "player_clean_key"]).to_csv(
        out_dir / "wr_te_rec_yards_distribution_scoreboard.csv", index=False
    )
    feature.sort_values(["week", "position_family", "team", "player_clean_key"]).to_csv(
        out_dir / "feature_eligibility_audit.csv", index=False
    )
    overlap.to_csv(out_dir / "live_board_overlap_audit.csv", index=False)

    depth_available = int(feature["target_depth_feature_available"].sum())
    summary = {
        "version": "ALL_PLAYER_ALL_POSITION_REPLAY_V1",
        "season": SEASON,
        "weeks": list(WEEKS),
        "iterations": int(iterations),
        "point_rows": int(len(point)),
        "unique_player_weeks": int(point[["week", "team", "player_clean_key"]].drop_duplicates().shape[0]),
        "position_counts": point.groupby("position_family").size().astype(int).to_dict(),
        "unique_player_weeks_by_position": (
            point[["week", "team", "player_clean_key", "position_family"]]
            .drop_duplicates()
            .groupby("position_family")
            .size()
            .astype(int)
            .to_dict()
        ),
        "market_counts": point.groupby("market").size().astype(int).to_dict(),
        "point_summary": _group_point_summary(point),
        "wr_te_distribution_rows": int(len(dist)),
        "target_depth_feature_available_rows": depth_available,
        "target_depth_feature_available_rate": float(depth_available / len(feature)) if len(feature) else 0.0,
        "target_depth_mean_invariance_max_abs_gap": float(dist["mean_gap_shadow_minus_baseline"].abs().max()),
        "target_depth_crps_baseline_mean": float(dist["baseline_crps"].mean()),
        "target_depth_crps_shadow_mean": float(dist["shadow_crps"].mean()),
        "target_depth_crps_mean_improvement": float(dist["crps_improvement"].mean()),
        "target_depth_scale_quantile_summary": _depth_quantile_summary(dist),
        "trajectory_eligible_rows": int(feature["target_share_trajectory_eligible"].sum()),
        "trajectory_status": "FROZEN_FEATURE_NOT_YET_ELIGIBLE",
        "rb_week5_room_shadow_rows": int(point["rb_week5_room_allocation_shadow_applied"].sum()),
        "rb_p3_rows": int(point["rb_p3_applied"].sum()),
        "rb_rush_rec_v2_rows": int(point["rb_rush_rec_v2_applied"].sum()),
        "qb_synthesis_rows": int(point["qb_synthesis_applied"].sum()),
        "sportsbook_inputs_used_upstream": False,
        "live_overlap_rows": int(len(overlap)),
        "specialist_audits": specialist_audits,
        "automatic_promotion": False,
        "paid_odds_api_used": False,
    }
    (out_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--universe-dir", type=Path, required=True)
    p.add_argument("--rb-week1-context", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--live-board-root", type=Path)
    a = p.parse_args()
    run_replay(
        player_logs_path=a.player_logs,
        team_weekly_path=a.team_weekly,
        schedule_path=a.schedule,
        universe_dir=a.universe_dir,
        rb_week1_context_path=a.rb_week1_context,
        out_dir=a.out_dir,
        iterations=a.iterations,
        live_board_root=a.live_board_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
