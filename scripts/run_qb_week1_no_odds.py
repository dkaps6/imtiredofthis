#!/usr/bin/env python3
"""Generate 2026 Week-1 QB passing-yard projections without sportsbook inputs.

This is a production-validation runner, not a new model. It constructs one
internal pass_yards row per current Ourlads QB1, runs the same canonical
ML/State/Bayesian/rules/Monte-Carlo/ensemble stack as production, applies the
M89 official-attempt conversion, then applies the promoted M89/M90 football-only
QB_PASS_SYNTHESIS_V1 mean. No prop line, game line, odds, or sportsbook file is
read or required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.metrics_v2 import _join_optional, _join_player_form, _join_team_context
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics
from scripts.modeling.ensemble_v2 import apply_ensemble, load_weights
from scripts.modeling.ml_v2 import apply_ml_to_metrics
from scripts.modeling.qb_pass_synthesis_v1 import (
    attempt_conversion,
    build_feature_dict,
    load_artifact as load_qb_synthesis_artifact,
    load_player_logs as load_qb_player_logs,
    load_team_context as load_qb_team_context,
    predict_correction as predict_qb_synthesis,
)
from scripts.modeling.state_v2 import apply_state_to_metrics
from scripts.modeling.simulation_rules import apply_rules_to_metrics
from scripts.runtime_context import resolve_season, resolve_week
from scripts.simulation_v2 import lookup, simulate

DATA = Path("data")
ML_DIAGNOSTICS = DATA / "model_ml_diagnostics.csv"
STATE_DIAGNOSTICS = DATA / "model_state_diagnostics.csv"
WEATHER_PATH = DATA / "weather_week.csv"
OUT = DATA / "qb_week1_no_odds_projections.csv"


def _key(value) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} missing/empty: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if x.empty:
        raise RuntimeError(f"{label} has zero rows: {path}")
    return x


def build_internal_qb_metrics(season: int, week: int) -> pd.DataFrame:
    roles = _read(DATA / "roles_ourlads.csv", "Ourlads roles")
    schedule = _read(DATA / "team_week_map.csv", "team-week map")

    schedule["season"] = pd.to_numeric(schedule["season"], errors="coerce")
    schedule["week"] = pd.to_numeric(schedule["week"], errors="coerce")
    schedule["team"] = schedule["team"].map(canon_team)
    schedule["opponent"] = schedule["opponent"].map(canon_team)
    cur = schedule.loc[
        schedule["season"].eq(int(season)) & schedule["week"].eq(int(week)),
        ["team", "opponent"],
    ].drop_duplicates("team")
    if len(cur) != 32 or cur["team"].nunique() != 32:
        raise RuntimeError(
            f"Week-1 schedule must contain 32 teams; rows={len(cur)} teams={cur['team'].nunique()}"
        )

    roles["team"] = roles["team"].map(canon_team)
    roles["player_clean_key"] = roles["player"].map(_key)
    pos = roles.get("position", pd.Series("", index=roles.index)).fillna("").astype(str).str.upper().str.strip()
    group = roles.get("position_group", pd.Series("", index=roles.index)).fillna("").astype(str).str.upper().str.strip()
    role = roles.get("role", pd.Series("", index=roles.index)).fillna("").astype(str).str.upper().str.strip()
    model_role = roles.get("model_role", pd.Series("", index=roles.index)).fillna("").astype(str).str.upper().str.strip()
    depth_idx = pd.to_numeric(roles.get("depth_index", np.nan), errors="coerce")
    status = roles.get("status", pd.Series("active", index=roles.index)).fillna("active").astype(str).str.lower()

    qbs = roles.loc[(pos.eq("QB") | group.eq("QB")) & ~status.eq("inactive")].copy()
    qbs["_starter_priority"] = np.where(role.eq("QB1") | model_role.eq("QB1"), 0, np.where(depth_idx.eq(1), 1, 2))
    qbs["_depth_idx"] = depth_idx
    qbs = qbs.sort_values(["team", "_starter_priority", "_depth_idx"], na_position="last", kind="stable")
    qbs = qbs.drop_duplicates("team", keep="first")
    qbs = qbs.merge(cur, on="team", how="inner", validate="one_to_one")
    if len(qbs) != 32 or qbs["team"].nunique() != 32:
        missing = sorted(set(cur["team"]) - set(qbs["team"]))
        raise RuntimeError(f"Ourlads QB1 universe incomplete rows={len(qbs)} missing={missing}")

    base = qbs[["player", "player_clean_key", "team", "opponent"]].copy()
    if "player_identity_key" in qbs.columns:
        base["player_identity_key"] = qbs["player_identity_key"]
    base["event_id"] = base.apply(lambda r: "|".join(sorted([str(r["team"]), str(r["opponent"])])), axis=1)
    base["season"] = int(season)
    base["week"] = int(week)
    base["market"] = "pass_yards"

    out = _join_player_form(base, int(season), int(week))
    out = _join_team_context(out, int(season))
    out = _join_optional(out, int(week))
    out["season"] = int(season)
    out["week"] = int(week)
    out["team_abbr"] = out["team"]
    out["opponent_abbr"] = out["opponent"]
    out["player_canonical"] = out["player"]
    if "tgt_share" in out.columns and "target_share" not in out.columns:
        out["target_share"] = out["tgt_share"]
    if "yprr" in out.columns and "yprr_proxy" not in out.columns:
        out["yprr_proxy"] = out["yprr"]
    return out.loc[:, ~out.columns.duplicated()].copy()


def project_week1(season: int, week: int, *, iterations: int | None = None) -> pd.DataFrame:
    if int(week) != 1:
        raise RuntimeError("run_qb_week1_no_odds.py is intentionally Week-1 only")

    metrics = build_internal_qb_metrics(season, week)
    ml = _read(ML_DIAGNOSTICS, "ML diagnostics")
    state = _read(STATE_DIAGNOSTICS, "State diagnostics")
    metrics = apply_ml_to_metrics(metrics, ml)
    metrics = apply_state_to_metrics(metrics, state)
    metrics = apply_bayesian_to_metrics(metrics)
    metrics = apply_rules_to_metrics(metrics)

    for flag, label in [
        ("ml_applied", "ML"), ("state_applied", "State"),
        ("bayes_applied", "Bayesian"), ("rules_applied", "rules"),
    ]:
        n = int(pd.to_numeric(metrics.get(flag, 0), errors="coerce").fillna(0).sum())
        if n == 0:
            raise RuntimeError(f"QB Week-1 dry run matched zero {label} rows")

    qb_artifact = load_qb_synthesis_artifact()
    qb_context = load_qb_team_context()
    qb_logs = load_qb_player_logs()
    weather = pd.read_csv(WEATHER_PATH, low_memory=False) if WEATHER_PATH.exists() and WEATHER_PATH.stat().st_size else pd.DataFrame()
    weights = load_weights()
    sims = simulate(metrics, iterations=iterations)

    rows = []
    for _, row in metrics.iterrows():
        outcomes = lookup(sims, row, "pass_yards")
        if outcomes is None or len(outcomes) == 0:
            raise RuntimeError(f"QB Week-1 simulation missing {row.get('player')} pass_yards")

        raw = np.asarray(outcomes, dtype=float)
        conv = attempt_conversion(row, qb_context)
        share = pd.to_numeric(pd.Series([row.get("qb_pass_att_share", 1.0)]), errors="coerce").iloc[0]
        share = float(np.clip(share, 0.0, 1.0)) if np.isfinite(share) else 1.0
        converted = raw * float(conv) * share
        mc_proj = float(np.mean(converted))

        ens = apply_ensemble(pd.DataFrame([{
            "market": "pass_yards",
            "mc_proj": mc_proj,
            "ml_proj": row.get("ml_proj"),
            "state_proj": row.get("state_proj"),
        }]), weights=weights).iloc[0]
        ensemble_proj = float(ens["ensemble_proj"])

        features = build_feature_dict(
            row,
            base_proj=ensemble_proj,
            mc_proj=mc_proj,
            team_context=qb_context,
            player_logs=qb_logs,
            weather=weather,
            season=int(season),
            week=int(week),
        )
        synthesis_proj, correction, version = predict_qb_synthesis(features, artifact=qb_artifact)
        if not np.isfinite(synthesis_proj):
            raise RuntimeError(f"non-finite QB synthesis projection for {row.get('player')}")

        if mc_proj > 0:
            adjusted = converted * max(0.0, float(synthesis_proj) / mc_proj)
        else:
            adjusted = converted

        rows.append({
            "season": int(season),
            "week": int(week),
            "event_id": row.get("event_id"),
            "player": row.get("player"),
            "player_clean_key": row.get("player_clean_key"),
            "team": row.get("team"),
            "opponent": row.get("opponent"),
            "market": "pass_yards",
            "mc_proj_after_attempt_conversion": mc_proj,
            "ml_proj": row.get("ml_proj"),
            "state_proj": row.get("state_proj"),
            "ensemble_proj": ensemble_proj,
            "ensemble_status": ens["ensemble_status"],
            "ensemble_method": ens["ensemble_method"],
            "ensemble_weight_mc": ens["ensemble_weight_mc"],
            "ensemble_weight_ml": ens["ensemble_weight_ml"],
            "ensemble_weight_state": ens["ensemble_weight_state"],
            "ensemble_calibration_rows": ens["ensemble_calibration_rows"],
            "qb_synthesis_proj": float(synthesis_proj),
            "qb_synthesis_correction": float(correction),
            "qb_synthesis_version": str(version),
            "qb_attempt_conversion": float(conv),
            "qb_pass_att_share": share,
            "qb_pred_attempts": features.get("pred_attempts"),
            "qb_pred_ypa": features.get("pred_ypa"),
            "model_mean": float(np.mean(adjusted)),
            "model_sd": float(np.std(adjusted, ddof=1)) if len(adjusted) > 1 else 0.0,
            "p10": float(np.quantile(adjusted, 0.10)),
            "p25": float(np.quantile(adjusted, 0.25)),
            "p50": float(np.quantile(adjusted, 0.50)),
            "p75": float(np.quantile(adjusted, 0.75)),
            "p90": float(np.quantile(adjusted, 0.90)),
            "simulation_iterations": int(sims.iterations),
            "football_only_no_odds": 1,
            "sportsbook_inputs_used": 0,
        })

    out = pd.DataFrame(rows).sort_values(["team", "player_clean_key"]).reset_index(drop=True)
    if len(out) != 32 or out["team"].nunique() != 32:
        raise RuntimeError(f"QB Week-1 output expected 32 starters; rows={len(out)} teams={out['team'].nunique()}")
    if not pd.to_numeric(out["sportsbook_inputs_used"], errors="coerce").eq(0).all():
        raise RuntimeError("QB Week-1 no-odds leakage flag failure")
    if not np.allclose(out["model_mean"], out["qb_synthesis_proj"], rtol=0, atol=1e-8):
        raise RuntimeError("QB Week-1 final distribution mean does not equal promoted synthesis")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=25000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    season = int(args.season if args.season is not None else resolve_season())
    week = int(args.week if args.week is not None else resolve_week(season=season))
    out = project_week1(season, week, iterations=args.iterations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(
        f"QB WEEK1 NO-ODDS: PASS rows={len(out)} teams={out['team'].nunique()} "
        f"min={out['qb_synthesis_proj'].min():.3f} max={out['qb_synthesis_proj'].max():.3f} out={args.out}"
    )
    print(out[[
        "team", "player", "opponent", "mc_proj_after_attempt_conversion", "ensemble_proj",
        "qb_synthesis_proj", "qb_synthesis_correction", "qb_pred_attempts", "qb_pred_ypa", "p25", "p50", "p75"
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
