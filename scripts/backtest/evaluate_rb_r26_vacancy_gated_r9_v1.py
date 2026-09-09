#!/usr/bin/env python3
"""Evaluate frozen RB R26 vacancy-gated R9 retrospective mechanism.

Scientific label: retrospective mechanism evidence only.
No sportsbook input. No target/future outcome features. No production writes.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week
from scripts.backtest.evaluate_rb_r25_receptions_specialist_v1 import (
    RB_POS,
    finite,
    metric,
    optional,
    prepared,
    read,
    _catch_prior,
)
from scripts.modeling.rb_receiving_identity_runtime_v1 import (
    EPS,
    FEATURES,
    attach_identity,
    identity_atlas,
)

ALPHA = 20.0
TRAIN_CLIP = 2.0
PRED_CLIP = 1.0
HISTORY_START = 2013
BASE = "baseline"
CAND = "candidate"


def weeks(season: int) -> range:
    return range(1, 18 if int(season) <= 2020 else 19)


def name_key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def load_bundle_inputs(data_dir: Path):
    return {
        "logs": read(data_dir / "player_game_logs_history.csv"),
        "team": read(data_dir / "team_weekly_history.csv"),
        "schedule": read(data_dir / "schedule_history.csv"),
        "injuries": optional(data_dir / "injuries_history.csv"),
        "weather": optional(data_dir / "weather_history.csv"),
    }


def build_base(data_dir: Path, inputs: dict, season: int, week: int) -> pd.DataFrame:
    universe = read(data_dir / "pregame_universe" / f"{season}_week_{week:02d}.csv")
    bundle = build_historical_context_bundle(
        player_logs=inputs["logs"],
        team_weekly=inputs["team"],
        pregame_universe=universe,
        schedule=inputs["schedule"],
        season=int(season),
        week=int(week),
        prior_season=int(season) - 1,
        injuries=_exact_week(inputs["injuries"], int(season), int(week)),
        weather=_exact_week(inputs["weather"], int(season), int(week)),
    )
    return prepared(bundle)


def target_labels(logs: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    actual = cp.build_actual_rows(logs, int(season), int(week))
    x = actual.loc[actual.market.eq("receptions"), [
        "team", "player_clean_key", "actual", "actual_opportunities"
    ]].rename(columns={"actual": "actual_receptions", "actual_opportunities": "actual_targets"})
    return x.drop_duplicates(["team", "player_clean_key"])


def rb_frame(base: pd.DataFrame, season: int, week: int, states: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    x = base.copy().reset_index(drop=False).rename(columns={"index": "_row_index"})
    x["position_family"] = x.get("position", "").fillna("").astype(str).str.upper().str.strip().replace({"HB": "RB", "TB": "RB"})
    x["baseline_entitlement_tgt_share"] = pd.to_numeric(x.get("entitlement_tgt_share"), errors="coerce").fillna(0.0).clip(lower=0.0)
    rb = x.loc[x.position_family.isin({"RB", "FB"})].copy()
    if rb.empty:
        raise RuntimeError(f"R26 zero RB rows for {season} W{week}")
    rb["b0_rb_pool"] = rb.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].transform("sum")
    rb["b0_rb_within_share"] = np.where(
        rb.b0_rb_pool.gt(0),
        rb.baseline_entitlement_tgt_share / rb.b0_rb_pool,
        0.0,
    )
    rb = attach_identity(rb, int(season), int(week), states, prev)
    return rb


def training_cases(season: int, data_dir: Path, states: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    inp = load_bundle_inputs(data_dir)
    parts: list[pd.DataFrame] = []
    for week in weeks(season):
        base = build_base(data_dir, inp, season, week)
        rb = rb_frame(base, season, week, states, prev)
        labels = target_labels(inp["logs"], season, week)
        rb = rb.merge(labels[["team", "player_clean_key", "actual_targets"]], on=["team", "player_clean_key"], how="left", validate="one_to_one")
        rb["actual_targets"] = pd.to_numeric(rb.actual_targets, errors="coerce").fillna(0.0)
        rb["actual_rb_targets"] = rb.groupby(["event_id", "team"])["actual_targets"].transform("sum")
        rb["actual_rb_within_share"] = np.where(
            rb.actual_rb_targets.gt(0), rb.actual_targets / rb.actual_rb_targets, 0.0
        )
        rb["within_residual_target"] = (
            np.log(rb.actual_rb_within_share.clip(lower=0.0) + EPS)
            - np.log(rb.b0_rb_within_share.clip(lower=0.0) + EPS)
        ).clip(-TRAIN_CLIP, TRAIN_CLIP)
        rb["season"] = int(season)
        rb["week"] = int(week)
        use = rb.loc[rb.actual_rb_targets.gt(0) & rb.b0_rb_pool.gt(0)].copy()
        parts.append(use)
        print(f"[r26-train] season={season} week={week:02d} rows={len(use)}")
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if out.empty:
        raise RuntimeError(f"R26 empty training casebook season={season}")
    return out


def fit_r9(train: pd.DataFrame):
    full_model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
    full_model.fit(train[FEATURES], train["within_residual_target"])

    oof_parts = []
    for start, end in ((5, 8), (9, 12), (13, 17)):
        tr = train.loc[pd.to_numeric(train.week, errors="coerce").lt(start)].copy()
        va = train.loc[pd.to_numeric(train.week, errors="coerce").between(start, end)].copy()
        if tr.empty or va.empty:
            continue
        m = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
        m.fit(tr[FEATURES], tr["within_residual_target"])
        raw = np.clip(m.predict(va[FEATURES]), -PRED_CLIP, PRED_CLIP)
        oof_parts.append(pd.DataFrame({
            "week": va.week.to_numpy(),
            "raw_pred": raw,
            "actual_residual": pd.to_numeric(va.within_residual_target, errors="coerce").to_numpy(float),
        }))
    if not oof_parts:
        raise RuntimeError("R26 R9 reliability produced zero OOF rows")
    oof = pd.concat(oof_parts, ignore_index=True)
    p = oof.raw_pred.to_numpy(float)
    y = oof.actual_residual.to_numpy(float)
    den = float(np.dot(p, p))
    raw_slope = float(np.dot(p, y) / den) if den > 1e-12 else 0.0
    reliability = float(np.clip(raw_slope, 0.0, 1.0))
    oof["shrunk_pred"] = reliability * oof.raw_pred
    meta = {
        "alpha": ALPHA,
        "train_clip": TRAIN_CLIP,
        "pred_clip": PRED_CLIP,
        "training_rows": int(len(train)),
        "oof_rows": int(len(oof)),
        "raw_reliability_slope": raw_slope,
        "reliability": reliability,
        "strict_prior_fit": True,
    }
    return full_model, reliability, oof, meta


def transition_maps(path: Path, season: int):
    st = read(path)
    st = st.loc[pd.to_numeric(st.season, errors="coerce").eq(int(season))].copy()
    if st.empty:
        raise RuntimeError(f"R26 transition state has no rows for {season}")
    st["week"] = pd.to_numeric(st.week, errors="coerce").astype(int)
    st["team"] = st.team.astype(str)
    room = st[["season", "week", "team", "room_exits_n", "room_entrants_n", "room_turnover_flag"]].drop_duplicates(["season", "week", "team"])
    room_map = {
        (int(r.season), int(r.week), str(r.team)): {
            "room_exits_n": int(r.room_exits_n),
            "room_entrants_n": int(r.room_entrants_n),
            "room_turnover_flag": int(r.room_turnover_flag),
        }
        for _, r in room.iterrows()
    }
    player_map = {
        (int(r.season), int(r.week), str(r.team), str(r.player_key)): {
            "continuing_same_team": int(r.continuing_same_team),
            "new_to_team_veteran": int(r.new_to_team_veteran),
            "no_prior_nfl_roster": int(r.no_prior_nfl_roster),
            "prior_depth_available": int(r.prior_depth_available),
        }
        for _, r in st.iterrows()
    }
    return room_map, player_map


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-season", type=int, required=True)
    ap.add_argument("--test-season", type=int, required=True)
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--transition-state", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.train_season != a.test_season - 1:
        raise RuntimeError("R26 fold must use immediately-prior training season")

    # Identity atlas may contain later rows, but attach_identity is an as-of join with
    # allow_exact_matches=False; target/future rows cannot enter a feature snapshot.
    states, prev = identity_atlas(HISTORY_START, a.test_season)
    train = training_cases(a.train_season, a.train_dir, states, prev)
    model, reliability, oof, fit_meta = fit_r9(train)

    inp = load_bundle_inputs(a.test_dir)
    room_map, player_map = transition_maps(a.transition_state, a.test_season)
    prior_catch_by_week = {w: _catch_prior(inp["logs"], a.test_season, w) for w in weeks(a.test_season)}

    rows: list[dict] = []
    audit_rows: list[dict] = []
    room_state_hits = 0
    room_state_total = 0
    player_state_hits = 0
    player_state_total = 0

    for week in weeks(a.test_season):
        base = build_base(a.test_dir, inp, a.test_season, week)
        rb = rb_frame(base, a.test_season, week, states, prev)
        labels = target_labels(inp["logs"], a.test_season, week)
        label_map = {
            (str(r.team), str(r.player_clean_key)): (finite(r.actual_targets), finite(r.actual_receptions))
            for _, r in labels.iterrows()
        }
        raw = np.clip(model.predict(rb[FEATURES]), -PRED_CLIP, PRED_CLIP)
        rb["r8_raw_residual"] = raw
        rb["r9_reliability"] = reliability
        rb["r9_calibrated_residual"] = reliability * raw
        rb["r9_score"] = np.log(rb.b0_rb_within_share.clip(lower=0.0) + EPS) + rb.r9_calibrated_residual

        for (event_id, tm), g in rb.groupby(["event_id", "team"], sort=False, dropna=False):
            g = g.copy()
            room_state_total += 1
            state = room_map.get((a.test_season, week, str(tm)))
            if state is None:
                raise RuntimeError(f"R26 missing canonical transition room state for {a.test_season} W{week} {tm}")
            room_state_hits += 1
            vacancy = int(state["room_exits_n"] >= 1)
            room_mass = float(pd.to_numeric(g.baseline_entitlement_tgt_share, errors="coerce").fillna(0.0).sum())
            if room_mass <= 0:
                continue
            base_room = pd.to_numeric(g.b0_rb_within_share, errors="coerce").fillna(0.0).to_numpy(float)
            if vacancy:
                score = pd.to_numeric(g.r9_score, errors="coerce").fillna(0.0).to_numpy(float)
                ww = np.exp(score - np.max(score))
                cand_room = ww / ww.sum() if ww.sum() > 0 else base_room.copy()
            else:
                cand_room = base_room.copy()
            if len(cand_room):
                cand_room[int(np.argmax(cand_room))] += 1.0 - float(cand_room.sum())

            rank_order = np.argsort(-base_room, kind="stable")
            rank_map = {int(idx): rank + 1 for rank, idx in enumerate(rank_order)}
            plays = float(np.nanmean(pd.to_numeric(g.get("rules_plays_est", 64.0), errors="coerce")))
            pass_rate = float(np.nanmean(pd.to_numeric(g.get("rules_pass_rate", 0.57), errors="coerce")))
            if not np.isfinite(plays):
                plays = 64.0
            if not np.isfinite(pass_rate):
                pass_rate = 0.57
            team_targets = plays * pass_rate
            prior_catch = prior_catch_by_week[week]

            for j, (_, r) in enumerate(g.iterrows()):
                player_state_total += 1
                pk = name_key(r.get("player", "")) or name_key(r.get("player_clean_key", ""))
                ps = player_map.get((a.test_season, week, str(tm), pk))
                if ps is not None:
                    player_state_hits += 1
                else:
                    ps = {
                        "continuing_same_team": 0,
                        "new_to_team_veteran": 0,
                        "no_prior_nfl_roster": 0,
                        "prior_depth_available": 0,
                    }
                key = str(r.player_clean_key)
                actual_t, actual_r = label_map.get((str(tm), key), (np.nan, np.nan))
                base_t = team_targets * float(r.baseline_entitlement_tgt_share)
                cand_t = team_targets * room_mass * float(cand_room[j])
                base_cr = finite(r.get("rules_catch_rate"), finite(r.get("bayes_receptions_per_target"), prior_catch))
                base_cr = float(np.clip(base_cr, 0.35, 0.95))
                base_ypt = max(finite(r.get("rules_ypt"), finite(r.get("bayes_ypt"), 0.0)), 0.0)
                base_rec_yards = base_t * base_ypt
                rows.append({
                    "train_season": a.train_season,
                    "season": a.test_season,
                    "week": week,
                    "event_id": str(event_id),
                    "team": str(tm),
                    "player": r.get("player", ""),
                    "player_clean_key": key,
                    "rb_rank": int(rank_map[j]),
                    "vacancy_active": vacancy,
                    "room_exits_n": int(state["room_exits_n"]),
                    "room_entrants_n": int(state["room_entrants_n"]),
                    **ps,
                    "actual_targets": actual_t,
                    "actual_receptions": actual_r,
                    "baseline_targets": base_t,
                    "candidate_targets": cand_t,
                    "baseline_receptions": base_t * base_cr,
                    "candidate_receptions": cand_t * base_cr,
                    "baseline_rec_yards": base_rec_yards,
                    "candidate_rec_yards": base_rec_yards,
                    "baseline_room_share": float(base_room[j]),
                    "candidate_room_share": float(cand_room[j]),
                    "r8_raw_residual": float(r.r8_raw_residual),
                    "r9_reliability": reliability,
                    "r9_calibrated_residual": float(r.r9_calibrated_residual),
                    "sportsbook_inputs_used": 0,
                    "future_outcomes_used": 0,
                })

            audit_rows.append({
                "train_season": a.train_season,
                "season": a.test_season,
                "week": week,
                "event_id": str(event_id),
                "team": str(tm),
                "vacancy_active": vacancy,
                "room_exits_n": int(state["room_exits_n"]),
                "baseline_rb_room_mass": room_mass,
                "candidate_rb_room_mass": float(room_mass * cand_room.sum()),
                "room_mass_gap": float(room_mass * cand_room.sum() - room_mass),
                "team_entitlement_gap": 0.0,
                "max_non_rb_entitlement_delta": 0.0,
                "max_receiving_yard_mean_delta": 0.0,
                "r22_authority_delta": 0.0,
                "sportsbook_inputs_used": 0,
                "future_outcomes_used": 0,
            })
        print(f"[r26-test] train={a.train_season} test={a.test_season} week={week:02d} reliability={reliability:.6f}")

    pred = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    if pred.empty or audit.empty:
        raise RuntimeError("R26 produced empty prediction/audit output")
    pred["role"] = np.where(pred.rb_rank.eq(1), "RB1", "RB2+")
    pred["vacancy_incumbent"] = ((pred.vacancy_active.eq(1)) & (pred.continuing_same_team.eq(1))).astype(int)
    pred["vacancy_new_veteran"] = ((pred.vacancy_active.eq(1)) & (pred.new_to_team_veteran.eq(1))).astype(int)
    pred["vacancy_no_prior_nfl"] = ((pred.vacancy_active.eq(1)) & (pred.no_prior_nfl_roster.eq(1))).astype(int)

    cohorts = {
        "ALL": pred.index == pred.index,
        "VACANCY_ACTIVE": pred.vacancy_active.eq(1),
        "VACANCY_INCUMBENT": pred.vacancy_incumbent.eq(1),
        "VACANCY_RB1_INCUMBENT": pred.vacancy_incumbent.eq(1) & pred.role.eq("RB1"),
        "VACANCY_RB2PLUS_INCUMBENT": pred.vacancy_incumbent.eq(1) & pred.role.eq("RB2+"),
        "VACANCY_NEW_VETERAN": pred.vacancy_new_veteran.eq(1),
        "VACANCY_NO_PRIOR_NFL": pred.vacancy_no_prior_nfl.eq(1),
        "WEEK1": pred.week.eq(1),
        "WEEKS2PLUS": pred.week.ge(2),
    }
    mrows = []
    for cohort, mask in cohorts.items():
        g = pred.loc[mask].copy()
        for variant in (BASE, CAND):
            for market, actual_col in (("targets", "actual_targets"), ("receptions", "actual_receptions")):
                mrows.append({
                    "season": a.test_season,
                    "cohort": cohort,
                    "variant": variant,
                    "market": market,
                    **metric(g[actual_col], g[f"{variant}_{market}"]),
                })
    metrics = pd.DataFrame(mrows)

    fit_meta.update({
        "train_season": a.train_season,
        "test_season": a.test_season,
        "history_start": HISTORY_START,
        "room_state_coverage": room_state_hits / max(room_state_total, 1),
        "player_state_coverage": player_state_hits / max(player_state_total, 1),
        "candidate": "RB_R26_VACANCY_GATED_R9_RETROSPECTIVE_V1",
        "scientific_label": "PREDECLARED_RETROSPECTIVE_MECHANISM_TEST",
        "sportsbook_inputs_used": 0,
        "future_outcomes_used_in_features": 0,
        "receiving_yard_mean_changed": False,
        "r22_changed": False,
    })

    a.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(a.out_dir / "r26_predictions.csv", index=False)
    audit.to_csv(a.out_dir / "r26_structural_audit.csv", index=False)
    metrics.to_csv(a.out_dir / "r26_metrics.csv", index=False)
    oof.to_csv(a.out_dir / "r26_r9_oof_reliability.csv", index=False)
    (a.out_dir / "r26_fit_metadata.json").write_text(json.dumps(fit_meta, indent=2, sort_keys=True) + "\n")
    print(metrics.to_csv(index=False))
    print(json.dumps(fit_meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
