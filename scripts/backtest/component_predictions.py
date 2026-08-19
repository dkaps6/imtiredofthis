"""Walk-forward component predictions for historical NFL player-game backtests.

This module uses the canonical production components at an explicit historical
cutoff. Target-week outcomes are joined only after all component projections
have been created. Diagnostic trace columns record which historical context was
actually available and the inputs that produced Monte Carlo passing estimates.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest.historical_context import HistoricalContextBundle, build_historical_context_bundle
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling.ml_v2 import build_and_train as build_ml
from scripts.modeling.state_v2 import build_state_predictions
from scripts.modeling import simulation_rules
from scripts.simulation_v2 import lookup, simulate
from scripts.utils.canonical_names import canonicalize_player_name_safe

BACKTEST_DIR = Path("data") / "backtests"
COMPONENT_PATH = BACKTEST_DIR / "component_predictions.csv"

TARGET_COLUMNS = {
    "pass_yards": "pass_yards", "rush_yards": "rush_yards", "rec_yards": "rec_yards",
    "receptions": "receptions", "rush_att": "rushes", "rush_rec_yards": "rush_rec_yards",
}
OPPORTUNITY_COLUMNS = {
    "pass_yards": "pass_att", "rush_yards": "rushes", "rec_yards": "targets",
    "receptions": "targets", "rush_att": "rushes", "rush_rec_yards": "rush_rec_opportunities",
}


def _key(value) -> str:
    try:
        _, key = canonicalize_player_name_safe(value)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _game_key(team: str, opponent: str) -> str:
    return "|".join(sorted([str(team or "").upper().strip(), str(opponent or "").upper().strip()]))


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _numeric_series(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _context_trace_frame(bundle: HistoricalContextBundle) -> pd.DataFrame:
    rows = []
    for p in bundle.players:
        f = p.features or {}
        offense, defense = p.offense, p.defense
        matchup = int(f.get("matchup_available") or 0) == 1 and bool(str(f.get("primary_cb") or "").strip())
        coverage = defense is not None and _finite(defense.coverage_man_rate) and _finite(defense.coverage_zone_rate)
        rows.append({
            "team": p.team, "player_clean_key": _key(p.player),
            "ctx_tgt_share_available": int(_finite(f.get("tgt_share"))),
            "ctx_rush_share_available": int(_finite(f.get("rush_share"))),
            "ctx_ypa_available": int(_finite(f.get("ypa"))),
            "ctx_success_rate_available": int(offense is not None and defense is not None and _finite(offense.success_rate_off) and _finite(defense.success_rate_def)),
            "ctx_pace_available": int(offense is not None and (_finite(offense.neutral_pace) or _finite(offense.neutral_pace_last5) or _finite(offense.sec_per_play_last5))),
            "ctx_plays_available": int(offense is not None and _finite(offense.plays_est)),
            "ctx_proe_available": int(offense is not None and _finite(offense.proe)),
            "ctx_pressure_available": int(offense is not None and defense is not None and _finite(offense.pressure_rate_allowed) and _finite(defense.pressure_rate_generated)),
            "ctx_explosive_available": int(defense is not None and _finite(defense.explosive_play_rate_allowed)),
            "ctx_def_epa_available": int(defense is not None and (_finite(defense.def_pass_epa) or _finite(defense.def_rush_epa))),
            "ctx_coverage_scheme_available": int(coverage),
            "ctx_box_rates_available": int(defense is not None and _finite(defense.light_box_rate) and _finite(defense.heavy_box_rate)),
            "ctx_wr_cb_matchup_available": int(matchup),
            "ctx_injury_available": int(f.get("injury_report_available") or 0),
            "ctx_weather_available": int(f.get("weather_forecast_available") or 0),
            "mc_off_pressure_allowed": getattr(offense, "pressure_rate_allowed", None) if offense is not None else np.nan,
            "mc_def_pressure_generated": getattr(defense, "pressure_rate_generated", None) if defense is not None else np.nan,
        })
    return pd.DataFrame(rows).drop_duplicates(["team", "player_clean_key"])


def build_market_frame(bundle: HistoricalContextBundle) -> pd.DataFrame:
    """Expand the pregame universe; only the inferred primary QB gets pass_yards."""
    base = bundle.player_form.copy()
    base.columns = [str(c).strip().lower() for c in base.columns]
    if base.empty:
        raise RuntimeError("historical bundle contains no pregame players")
    base["player_clean_key"] = base.get("player_clean_key", base["player"]).map(_key)
    base["event_id"] = [_game_key(t, o) for t, o in zip(base["team"], base["opponent"])]
    rows = []
    for _, player in base.iterrows():
        pos = str(player.get("position", "")).upper().strip()
        markets = ["rush_yards", "rush_att", "rush_rec_yards"]
        if pos == "QB" and int(pd.to_numeric(pd.Series([player.get("qb_projection_eligible", 0)]), errors="coerce").fillna(0).iloc[0]) == 1:
            markets.append("pass_yards")
        if pos in {"RB", "WR", "LWR", "RWR", "SWR", "TE", "FB"}:
            markets += ["rec_yards", "receptions"]
        for market in dict.fromkeys(markets):
            rec = player.to_dict(); rec["market"] = market; rows.append(rec)
    return pd.DataFrame(rows)


def _attach_component_projection(frame: pd.DataFrame, predictions: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy(); pred = predictions.copy()
    pred.columns = [str(c).strip().lower() for c in pred.columns]
    pred["player_clean_key"] = pred.get("player_clean_key", pred["player"]).map(_key)
    pred = pred.drop_duplicates(["team", "player_clean_key"]); by_key = pred.set_index(["team", "player_clean_key"])
    vals = []
    for _, row in out.iterrows():
        key = (str(row["team"]), str(row["player_clean_key"])); col = f"{prefix}_{row['market']}"
        if key in by_key.index and col in by_key.columns:
            value = by_key.loc[key, col]
            if isinstance(value, pd.Series): value = value.iloc[0]
            vals.append(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])
        else: vals.append(np.nan)
    out[f"{prefix}_proj"] = vals
    return out


def _attach_historical_passing_volume(metrics: pd.DataFrame, bundle: HistoricalContextBundle) -> pd.DataFrame:
    """Attach leakage-safe attempt/dropback conversion from the pre-cutoff team form."""
    out = metrics.copy()
    tf = bundle.team_form.copy()
    tf.columns = [str(c).strip().lower() for c in tf.columns]
    if "team" not in tf.columns or "pass_attempts_per_dropback" not in tf.columns:
        out["mc_pass_attempts_per_dropback"] = 1.0
        out["mc_pass_attempt_rate_source"] = "fallback_1.0"
        return out
    conv = tf[["team", "pass_attempts_per_dropback"]].drop_duplicates("team").copy()
    conv["pass_attempts_per_dropback"] = pd.to_numeric(conv["pass_attempts_per_dropback"], errors="coerce")
    out = out.merge(conv, on="team", how="left", validate="many_to_one")
    raw = pd.to_numeric(out["pass_attempts_per_dropback"], errors="coerce")
    valid = raw.between(0.50, 1.00, inclusive="both")
    out["mc_pass_attempts_per_dropback"] = raw.where(valid, 1.0).clip(0.50, 1.00)
    out["mc_pass_attempt_rate_source"] = np.where(valid, "historical_pregame_pbp", "fallback_1.0")
    out.drop(columns=["pass_attempts_per_dropback"], inplace=True)
    return out


def build_mc_predictions(bundle: HistoricalContextBundle, *, iterations: int = 5000, seed: int = 42) -> pd.DataFrame:
    metrics = build_market_frame(bundle)
    bayes = build_bayesian_baseline(bundle.player_consensus)
    metrics = apply_bayesian_to_metrics(metrics, bayes)
    with patch.object(simulation_rules, "load_model_contexts", return_value=(bundle.teams, bundle.players)):
        metrics = simulation_rules.apply_rules_to_metrics(metrics)
    if int(pd.to_numeric(metrics["rules_applied"], errors="coerce").fillna(0).sum()) == 0:
        raise RuntimeError("historical rules matched zero rows")

    metrics = _attach_historical_passing_volume(metrics, bundle)
    trace = _context_trace_frame(bundle)
    metrics = metrics.merge(trace, on=["team", "player_clean_key"], how="left", validate="many_to_one")
    metrics["mc_projected_plays"] = pd.to_numeric(metrics.get("rules_plays_est"), errors="coerce")
    # rules_pass_rate is derived from qb_dropback share in the historical PBP.
    # It therefore represents dropbacks/plays, not official pass attempts/plays.
    metrics["mc_dropback_rate"] = pd.to_numeric(metrics.get("rules_pass_rate"), errors="coerce")
    metrics["mc_pass_rate"] = metrics["mc_dropback_rate"] * metrics["mc_pass_attempts_per_dropback"]
    metrics["mc_team_expected_dropbacks"] = metrics["mc_projected_plays"] * metrics["mc_dropback_rate"]
    metrics["mc_team_expected_pass_attempts"] = metrics["mc_team_expected_dropbacks"] * metrics["mc_pass_attempts_per_dropback"]
    metrics["mc_qb_pass_att_share"] = pd.to_numeric(metrics.get("qb_pass_att_share"), errors="coerce")
    metrics["mc_expected_pass_attempts"] = metrics["mc_team_expected_pass_attempts"] * metrics["mc_qb_pass_att_share"].fillna(1.0)
    metrics["mc_qb_projection_eligible"] = pd.to_numeric(metrics.get("qb_projection_eligible"), errors="coerce")
    metrics["mc_qb_role_score"] = pd.to_numeric(metrics.get("qb_role_score"), errors="coerce")
    metrics["mc_qb_role_source"] = metrics.get("qb_role_source", "")
    metrics["mc_base_ypa"] = pd.to_numeric(metrics.get("ypa"), errors="coerce")
    metrics["mc_bayes_ypa"] = pd.to_numeric(metrics.get("bayes_ypa"), errors="coerce")
    metrics["mc_rules_ypa"] = pd.to_numeric(metrics.get("rules_ypa"), errors="coerce")
    metrics["mc_pass_eff_mult"] = pd.to_numeric(metrics.get("rules_pass_eff_mult"), errors="coerce")
    metrics["mc_pressure_mismatch"] = pd.to_numeric(metrics.get("mc_def_pressure_generated"), errors="coerce") - pd.to_numeric(metrics.get("mc_off_pressure_allowed"), errors="coerce")

    sims = simulate(metrics, iterations=int(iterations), seed=int(seed)); rows = []
    for _, row in metrics.iterrows():
        outcomes = lookup(sims, row, str(row["market"]))
        if outcomes is not None and len(outcomes) and str(row["market"]) == "pass_yards":
            # simulation_v2 currently treats its team pass count as dropbacks.
            # Convert that to official attempts before applying the QB's share.
            attempt_rate = pd.to_numeric(pd.Series([row.get("mc_pass_attempts_per_dropback")]), errors="coerce").iloc[0]
            share = pd.to_numeric(pd.Series([row.get("qb_pass_att_share")]), errors="coerce").iloc[0]
            if pd.notna(attempt_rate): outcomes = outcomes * float(np.clip(attempt_rate, 0.50, 1.00))
            if pd.notna(share): outcomes = outcomes * float(np.clip(share, 0.0, 1.0))
        rows.append(float(np.mean(outcomes)) if outcomes is not None and len(outcomes) else np.nan)
    metrics["mc_proj"] = rows
    return metrics


def build_actual_rows(player_logs: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    x = player_logs.copy(); x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "player", "team"}; missing = required - set(x.columns)
    if missing: raise RuntimeError(f"player logs missing actual-result columns: {sorted(missing)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce"); x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].eq(int(season)) & x["week"].eq(int(week))].copy()
    if x.empty: return pd.DataFrame(columns=["team", "player_clean_key", "market", "actual", "actual_opportunities"])
    x["player_clean_key"] = x.get("player_clean_key", x["player"]).map(_key)
    x["rush_rec_yards"] = _numeric_series(x, "rush_yards").fillna(0.0) + _numeric_series(x, "rec_yards").fillna(0.0)
    x["rush_rec_opportunities"] = _numeric_series(x, "rushes").fillna(0.0) + _numeric_series(x, "targets").fillna(0.0)
    rows = []
    for _, r in x.iterrows():
        for market, col in TARGET_COLUMNS.items():
            if col not in x.columns: continue
            actual = pd.to_numeric(pd.Series([r.get(col)]), errors="coerce").iloc[0]
            opp_col = OPPORTUNITY_COLUMNS.get(market)
            opportunities = pd.to_numeric(pd.Series([r.get(opp_col)]), errors="coerce").iloc[0] if opp_col else np.nan
            if pd.notna(actual):
                rows.append({"team": r["team"], "player_clean_key": r["player_clean_key"], "market": market, "actual": float(actual), "actual_opportunities": float(opportunities) if pd.notna(opportunities) else np.nan})
    return pd.DataFrame(rows).drop_duplicates(["team", "player_clean_key", "market"])


def predict_week(*, player_logs: pd.DataFrame, team_weekly: pd.DataFrame, pregame_universe: pd.DataFrame, schedule: pd.DataFrame, season: int, week: int, prior_season: int, team_coverage: pd.DataFrame | None = None, exposure: pd.DataFrame | None = None, injuries: pd.DataFrame | None = None, weather: pd.DataFrame | None = None, iterations: int = 5000, seed: int = 42) -> pd.DataFrame:
    bundle = build_historical_context_bundle(player_logs=player_logs, team_weekly=team_weekly, pregame_universe=pregame_universe, schedule=schedule, season=int(season), week=int(week), prior_season=int(prior_season), team_coverage=team_coverage, exposure=exposure, injuries=injuries, weather=weather)
    mc = build_mc_predictions(bundle, iterations=iterations, seed=seed)
    _, ml_pred = build_ml(player_logs, bundle.player_consensus, int(season), int(week)); _, state_pred = build_state_predictions(player_logs, bundle.player_consensus, int(season), int(week))
    diagnostic_cols = [c for c in mc.columns if c.startswith("ctx_") or c.startswith("mc_") or c.startswith("rules_") or c.startswith("qb_")]
    base_cols = ["player", "player_clean_key", "team", "opponent", "season", "week", "position", "role", "event_id", "market", "mc_proj"]
    out = mc[list(dict.fromkeys([*base_cols, *diagnostic_cols]))].copy()
    out = _attach_component_projection(out, ml_pred, "ml"); out = _attach_component_projection(out, state_pred, "state")
    out["prediction_cutoff"] = f"{int(season)}-W{int(week):02d} pregame"; out["prior_season"] = int(prior_season)
    actual = build_actual_rows(player_logs, int(season), int(week))
    out = out.merge(actual, on=["team", "player_clean_key", "market"], how="left", validate="one_to_one")
    return out.loc[pd.to_numeric(out["actual"], errors="coerce").notna()].reset_index(drop=True)


def append_component_predictions(frame: pd.DataFrame, path: Path = COMPONENT_PATH) -> None:
    if frame is None or frame.empty: return
    path.parent.mkdir(parents=True, exist_ok=True); new = frame.copy()
    if path.exists() and path.stat().st_size > 0: both = pd.concat([pd.read_csv(path), new], ignore_index=True)
    else: both = new
    keys = ["season", "week", "team", "player_clean_key", "market"]
    both.sort_values(keys).drop_duplicates(keys, keep="last").to_csv(path, index=False)
