#!/usr/bin/env python3
"""Production pricing from canonical predictive components + joint simulation.

Production order:
1. independent leakage-safe ML v2 and state v2 projections,
2. leakage-safe empirical-Bayesian player baseline,
3. empirical football/context rules,
4. joint Monte Carlo distribution,
5. evidence-weighted ensemble mean (only when OOS-calibrated weights exist),
6. M89/M90-promoted football-only QB passing-yards residual synthesis,
7. sportsbook comparison.

For QB pass_yards, Monte Carlo's pass-opportunity count is first converted to
official pass attempts using the M89 semantic contract (official attempts /
[official attempts + sacks + QB scrambles]) before the ensemble is formed.
The promoted synthesis then corrects the ensemble mean using football-only
pregame features. Sportsbook lines/odds never construct a player projection,
ensemble weight, or synthesis correction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
from scripts.pricing_v2 import _fair_market_prob, _fair_odds
from scripts.runtime_context import resolve_season, resolve_week
from scripts.simulation_v2 import MARKET_MAP, lookup, simulate

DATA = Path("data")
OUTPUTS = Path("outputs")
OUT = OUTPUTS / "props_priced_clean.csv"
RULE_INPUTS = DATA / "model_rule_simulation_inputs.csv"
ML_DIAGNOSTICS = DATA / "model_ml_diagnostics.csv"
STATE_DIAGNOSTICS = DATA / "model_state_diagnostics.csv"
WEATHER_PATH = DATA / "weather_week.csv"


def _finite(value, default=np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _runtime_week(row: pd.Series) -> int:
    week = _finite(row.get("week"))
    return int(week) if np.isfinite(week) else int(resolve_week())


def price(season: int) -> pd.DataFrame:
    metrics_path = DATA / "metrics_ready.csv"
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        raise RuntimeError("data/metrics_ready.csv missing or empty")
    df = pd.read_csv(metrics_path)
    df.columns = [str(c).lower() for c in df.columns]
    if "season" in df.columns:
        df = df.loc[pd.to_numeric(df["season"], errors="coerce").eq(int(season))].copy()
    if df.empty:
        raise RuntimeError(f"metrics_ready contains no rows for season={season}")

    if not ML_DIAGNOSTICS.exists() or ML_DIAGNOSTICS.stat().st_size == 0:
        raise RuntimeError("data/model_ml_diagnostics.csv missing; ML v2 must train before production pricing")
    df = apply_ml_to_metrics(df, pd.read_csv(ML_DIAGNOSTICS))
    ml_rows = int(pd.to_numeric(df.get("ml_applied", 0), errors="coerce").fillna(0).sum())
    if ml_rows == 0:
        raise RuntimeError("ML v2 matched 0 supported pricing rows; refusing silent placeholder behavior")

    if not STATE_DIAGNOSTICS.exists() or STATE_DIAGNOSTICS.stat().st_size == 0:
        raise RuntimeError("data/model_state_diagnostics.csv missing; state v2 must train before production pricing")
    df = apply_state_to_metrics(df, pd.read_csv(STATE_DIAGNOSTICS))
    state_rows = int(pd.to_numeric(df.get("state_applied", 0), errors="coerce").fillna(0).sum())
    if state_rows == 0:
        raise RuntimeError("State v2 matched 0 supported pricing rows; refusing legacy 0.5 fallback behavior")

    df = apply_bayesian_to_metrics(df)
    bayes_rows = int(pd.to_numeric(df.get("bayes_applied", 0), errors="coerce").fillna(0).sum())
    if bayes_rows == 0:
        raise RuntimeError("Bayesian adapter matched 0 metrics rows; refusing baseline-only production pricing")

    df = apply_rules_to_metrics(df)
    rule_rows = int(pd.to_numeric(df.get("rules_applied", 0), errors="coerce").fillna(0).sum())
    if rule_rows == 0:
        raise RuntimeError("Canonical rule adapter matched 0 rows; refusing untracked production pricing")
    RULE_INPUTS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RULE_INPUTS, index=False)
    print(f"[pricing] ML v2 supported rows={ml_rows}/{len(df)}")
    print(f"[pricing] state v2 supported rows={state_rows}/{len(df)}")
    print(f"[pricing] bayesian baseline rows={bayes_rows}/{len(df)}")
    print(f"[pricing] canonical rules applied rows={rule_rows}/{len(df)} -> {RULE_INPUTS}")

    weights = load_weights()
    if weights.empty:
        print("[pricing] ensemble weights unavailable: explicit MC-only fallback until walk-forward calibration")
    else:
        print(f"[pricing] ensemble calibrated markets={sorted(weights['market'].astype(str).str.lower().unique().tolist())}")

    canonical_markets = df.get("market", pd.Series("", index=df.index)).fillna("").astype(str).str.lower().map(
        lambda value: MARKET_MAP.get(value, value)
    )
    has_qb_pass = bool(canonical_markets.eq("pass_yards").any())
    qb_artifact = qb_team_context = qb_player_logs = None
    weather = pd.DataFrame()
    if has_qb_pass:
        # Promotion is fail-closed: once pass-yards synthesis is part of production,
        # missing artifact/context is a fatal data-contract error, not a silent
        # fallback to the superseded pre-M89 QB projection.
        qb_artifact = load_qb_synthesis_artifact()
        qb_team_context = load_qb_team_context()
        qb_player_logs = load_qb_player_logs()
        if WEATHER_PATH.exists() and WEATHER_PATH.stat().st_size > 0:
            weather = pd.read_csv(WEATHER_PATH, low_memory=False)
        print(
            f"[pricing] promoted QB synthesis enabled version={qb_artifact['version']} "
            f"teams={len(qb_team_context)} player_logs={len(qb_player_logs)}"
        )

    sims = simulate(df)
    rows, missed = [], []
    qb_synthesis_rows = 0
    for _, row in df.iterrows():
        raw_market = str(row.get("market", "") or "").lower()
        market = MARKET_MAP.get(raw_market, raw_market)
        outcomes = lookup(sims, row, raw_market)
        if outcomes is None or len(outcomes) == 0:
            missed.append((row.get("player"), raw_market)); continue

        base_outcomes = np.asarray(outcomes, dtype=float)
        qb_attempt_rate = np.nan
        qb_share = np.nan

        # Historical M89/M90 evaluation converted the simulator's pass-opportunity
        # count to official attempts before computing QB passing yards. Production
        # must use the identical semantic or the promoted model would be fed a
        # different MC component than the one that passed confirmation.
        if market == "pass_yards":
            try:
                qb_attempt_rate = attempt_conversion(row, qb_team_context)
                qb_share = _finite(row.get("qb_pass_att_share"), 1.0)
                qb_share = float(np.clip(qb_share, 0.0, 1.0)) if np.isfinite(qb_share) else 1.0
                base_outcomes = base_outcomes * qb_attempt_rate * qb_share
            except Exception as exc:
                raise RuntimeError(
                    f"promoted QB attempt conversion failed player={row.get('player')} "
                    f"team={row.get('team')} opponent={row.get('opponent')}: {exc}"
                ) from exc

        mc_proj = float(np.mean(base_outcomes))
        component_row = pd.DataFrame([{
            "market": market,
            "mc_proj": mc_proj,
            "ml_proj": row.get("ml_proj"),
            "state_proj": row.get("state_proj"),
        }])
        ens = apply_ensemble(component_row, weights=weights).iloc[0]
        ensemble_proj = float(ens["ensemble_proj"])

        target_mean = ensemble_proj
        qb_synthesis_proj = np.nan
        qb_synthesis_correction = np.nan
        qb_synthesis_applied = 0
        qb_synthesis_version = ""
        qb_pred_attempts = np.nan
        qb_pred_ypa = np.nan

        if market == "pass_yards":
            try:
                features = build_feature_dict(
                    row,
                    base_proj=ensemble_proj,
                    mc_proj=mc_proj,
                    team_context=qb_team_context,
                    player_logs=qb_player_logs,
                    weather=weather,
                    season=int(season),
                    week=_runtime_week(row),
                )
                qb_synthesis_proj, qb_synthesis_correction, qb_synthesis_version = predict_qb_synthesis(
                    features, artifact=qb_artifact
                )
                if not np.isfinite(qb_synthesis_proj):
                    raise RuntimeError("non-finite promoted QB synthesis projection")
                target_mean = float(qb_synthesis_proj)
                qb_pred_attempts = _finite(features.get("pred_attempts"))
                qb_pred_ypa = _finite(features.get("pred_ypa"))
                qb_synthesis_applied = 1
                qb_synthesis_rows += 1
            except Exception as exc:
                raise RuntimeError(
                    f"promoted QB synthesis failed player={row.get('player')} "
                    f"team={row.get('team')} opponent={row.get('opponent')}: {exc}"
                ) from exc

        # Preserve Monte Carlo's non-negative distribution shape while aligning
        # its mean to the final football projection. For QB pass_yards this is the
        # promoted synthesis mean; for every other market it remains the canonical
        # ensemble mean. Sportsbook information still enters only after this step.
        if np.isfinite(mc_proj) and mc_proj > 0 and np.isfinite(target_mean):
            adjusted_outcomes = base_outcomes * max(0.0, target_mean / mc_proj)
        else:
            adjusted_outcomes = base_outcomes

        if market == "anytime_td":
            line = 0.5
            p_over = float(np.mean(adjusted_outcomes >= 1.0))
        else:
            try:
                line = float(row.get("line"))
            except Exception:
                missed.append((row.get("player"), raw_market)); continue
            p_over = float(np.mean(adjusted_outcomes > line))
        p_under = 1.0 - p_over
        model_proj = float(np.mean(adjusted_outcomes))
        model_sd = float(np.std(adjusted_outcomes, ddof=1)) if len(adjusted_outcomes) > 1 else 0.0
        mkt_over, mkt_under = _fair_market_prob(row.get("over_odds"), row.get("under_odds"))

        common = {
            "event_id": row.get("event_id"), "player": row.get("player"), "player_clean_key": row.get("player_clean_key"),
            "team": row.get("team"), "opponent": row.get("opponent"), "market": market, "source_market": raw_market,
            "vegas_line": line, "model_proj": model_proj, "mc_proj": mc_proj, "model_sd": model_sd,
            "simulation_iterations": sims.iterations,
            "ensemble_proj": ensemble_proj, "ensemble_status": ens["ensemble_status"], "ensemble_method": ens["ensemble_method"],
            "ensemble_weight_mc": ens["ensemble_weight_mc"], "ensemble_weight_ml": ens["ensemble_weight_ml"],
            "ensemble_weight_state": ens["ensemble_weight_state"], "ensemble_calibration_rows": ens["ensemble_calibration_rows"],
            "ml_proj": row.get("ml_proj"), "ml_applied": int(row.get("ml_applied", 0) or 0), "ml_method": row.get("ml_method"), "ml_training_cutoff": row.get("ml_training_cutoff"),
            "state_proj": row.get("state_proj"), "state_applied": int(row.get("state_applied", 0) or 0), "state_method": row.get("state_method"), "state_training_cutoff": row.get("state_training_cutoff"),
            "bayes_applied": int(row.get("bayes_applied", 0) or 0), "bayes_evidence_state": row.get("bayes_evidence_state"),
            "rules_applied": int(row.get("rules_applied", 0) or 0), "rules_role": row.get("rules_role"),
            "qb_synthesis_applied": qb_synthesis_applied,
            "qb_synthesis_proj": qb_synthesis_proj,
            "qb_synthesis_correction": qb_synthesis_correction,
            "qb_synthesis_version": qb_synthesis_version,
            "qb_attempt_conversion": qb_attempt_rate,
            "qb_pass_att_share": qb_share,
            "qb_pred_attempts": qb_pred_attempts,
            "qb_pred_ypa": qb_pred_ypa,
            "season": int(season), "week": row.get("week"), "book": row.get("book"), "book_title": row.get("book_title"),
            "vegas_over_odds": row.get("over_odds"), "vegas_under_odds": row.get("under_odds"),
        }
        for side, prob, market_prob, vegas_odds in (("OVER", p_over, mkt_over, row.get("over_odds")), ("UNDER", p_under, mkt_under, row.get("under_odds"))):
            edge = prob - market_prob if pd.notna(market_prob) else np.nan
            rec = dict(common)
            rec.update({"side": side, "fair_prob": prob, "market_prob": market_prob, "vegas_odds": vegas_odds, "fair_odds": _fair_odds(prob), "edge_pct": edge, "edge_abs": abs(edge) if pd.notna(edge) else np.nan})
            rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Monte Carlo pricing produced 0 rows")
    if has_qb_pass and qb_synthesis_rows == 0:
        raise RuntimeError("promoted QB synthesis applied to zero pass_yards pricing rows")
    if missed:
        debug = DATA / "_debug" / "pricing_unsimulated_props.csv"
        debug.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(missed, columns=["player", "market"]).drop_duplicates().to_csv(debug, index=False)
        print(f"[pricing] WARN unsimulated player/markets={len(set(missed))} -> {debug}")
    print("[pricing] ensemble status:", out["ensemble_status"].value_counts().to_dict())
    if has_qb_pass:
        print(f"[pricing] promoted QB synthesis applied input_rows={qb_synthesis_rows} output_side_rows={int(out['qb_synthesis_applied'].sum())}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--season", type=int, default=None); args = parser.parse_args()
    season = int(args.season if args.season is not None else resolve_season())
    out = price(season); OUTPUTS.mkdir(parents=True, exist_ok=True); out.to_csv(OUT, index=False)
    print(f"[pricing] wrote rows={len(out)} -> {OUT}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
