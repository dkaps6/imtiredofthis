"""Walk-forward component predictions for historical NFL player-game backtests.

This module uses the canonical production components at an explicit historical
cutoff. Target-week outcomes are joined only after all component projections
have been created.
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
    "pass_yards": "pass_yards",
    "rush_yards": "rush_yards",
    "rec_yards": "rec_yards",
    "receptions": "receptions",
    "rush_att": "rushes",
    "rush_rec_yards": "rush_rec_yards",
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


def build_market_frame(bundle: HistoricalContextBundle) -> pd.DataFrame:
    """Expand the explicit pregame universe into supported model markets."""
    base = bundle.player_form.copy()
    base.columns = [str(c).strip().lower() for c in base.columns]
    if base.empty:
        raise RuntimeError("historical bundle contains no pregame players")
    base["player_clean_key"] = base.get("player_clean_key", base["player"]).map(_key)
    base["event_id"] = [
        _game_key(t, o) for t, o in zip(base["team"], base["opponent"])
    ]
    rows = []
    for _, player in base.iterrows():
        pos = str(player.get("position", "")).upper().strip()
        markets = ["rush_yards", "rush_att", "rush_rec_yards"]
        if pos == "QB":
            markets.append("pass_yards")
        if pos in {"RB", "WR", "LWR", "RWR", "SWR", "TE", "FB"}:
            markets += ["rec_yards", "receptions"]
        for market in dict.fromkeys(markets):
            rec = player.to_dict()
            rec["market"] = market
            rows.append(rec)
    return pd.DataFrame(rows)


def _attach_component_projection(frame: pd.DataFrame, predictions: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    pred = predictions.copy()
    pred.columns = [str(c).strip().lower() for c in pred.columns]
    pred["player_clean_key"] = pred.get("player_clean_key", pred["player"]).map(_key)
    pred = pred.drop_duplicates(["team", "player_clean_key"])
    by_key = pred.set_index(["team", "player_clean_key"])
    vals = []
    for _, row in out.iterrows():
        key = (str(row["team"]), str(row["player_clean_key"]))
        col = f"{prefix}_{row['market']}"
        if key in by_key.index and col in by_key.columns:
            value = by_key.loc[key, col]
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            vals.append(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])
        else:
            vals.append(np.nan)
    out[f"{prefix}_proj"] = vals
    return out


def build_mc_predictions(
    bundle: HistoricalContextBundle,
    *,
    iterations: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the canonical Bayes -> rules -> joint-Monte-Carlo path at a cutoff."""
    metrics = build_market_frame(bundle)
    bayes = build_bayesian_baseline(bundle.player_consensus)
    metrics = apply_bayesian_to_metrics(metrics, bayes)

    # simulation_rules normally loads today's production context from disk.
    # During a historical run we inject the already-validated cutoff bundle so
    # the exact same production rule adapter is used without touching live files.
    with patch.object(simulation_rules, "load_model_contexts", return_value=(bundle.teams, bundle.players)):
        metrics = simulation_rules.apply_rules_to_metrics(metrics)

    if int(pd.to_numeric(metrics["rules_applied"], errors="coerce").fillna(0).sum()) == 0:
        raise RuntimeError("historical rules matched zero rows")

    sims = simulate(metrics, iterations=int(iterations), seed=int(seed))
    rows = []
    for _, row in metrics.iterrows():
        outcomes = lookup(sims, row, str(row["market"]))
        rows.append(float(np.mean(outcomes)) if outcomes is not None and len(outcomes) else np.nan)
    metrics["mc_proj"] = rows
    return metrics


def build_actual_rows(player_logs: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Create target-week outcomes. This function is called only after prediction."""
    x = player_logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "player", "team"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"player logs missing actual-result columns: {sorted(missing)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].eq(int(season)) & x["week"].eq(int(week))].copy()
    if x.empty:
        return pd.DataFrame(columns=["team", "player_clean_key", "market", "actual"])
    x["player_clean_key"] = x.get("player_clean_key", x["player"]).map(_key)
    x["rush_rec_yards"] = pd.to_numeric(x.get("rush_yards"), errors="coerce").fillna(0.0) + pd.to_numeric(x.get("rec_yards"), errors="coerce").fillna(0.0)
    rows = []
    for _, r in x.iterrows():
        for market, col in TARGET_COLUMNS.items():
            if col not in x.columns:
                continue
            actual = pd.to_numeric(pd.Series([r.get(col)]), errors="coerce").iloc[0]
            if pd.notna(actual):
                rows.append({
                    "team": r["team"], "player_clean_key": r["player_clean_key"],
                    "market": market, "actual": float(actual),
                })
    return pd.DataFrame(rows).drop_duplicates(["team", "player_clean_key", "market"])


def predict_week(
    *,
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    pregame_universe: pd.DataFrame,
    schedule: pd.DataFrame,
    season: int,
    week: int,
    prior_season: int,
    team_coverage: pd.DataFrame | None = None,
    exposure: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    iterations: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate one leakage-safe OOS component table for a historical week."""
    bundle = build_historical_context_bundle(
        player_logs=player_logs,
        team_weekly=team_weekly,
        pregame_universe=pregame_universe,
        schedule=schedule,
        season=int(season), week=int(week), prior_season=int(prior_season),
        team_coverage=team_coverage, exposure=exposure, injuries=injuries, weather=weather,
    )

    # All three components are created before target-week outcomes are touched.
    mc = build_mc_predictions(bundle, iterations=iterations, seed=seed)
    _, ml_pred = build_ml(player_logs, bundle.player_consensus, int(season), int(week))
    _, state_pred = build_state_predictions(player_logs, bundle.player_consensus, int(season), int(week))

    out = mc[[
        "player", "player_clean_key", "team", "opponent", "season", "week", "position", "role", "event_id", "market", "mc_proj"
    ]].copy()
    out = _attach_component_projection(out, ml_pred, "ml")
    out = _attach_component_projection(out, state_pred, "state")
    out["prediction_cutoff"] = f"{int(season)}-W{int(week):02d} pregame"
    out["prior_season"] = int(prior_season)

    actual = build_actual_rows(player_logs, int(season), int(week))
    out = out.merge(actual, on=["team", "player_clean_key", "market"], how="left", validate="one_to_one")
    # Ensemble calibration requires an observed result, but missing state is valid.
    out = out.loc[pd.to_numeric(out["actual"], errors="coerce").notna()].reset_index(drop=True)
    return out


def append_component_predictions(frame: pd.DataFrame, path: Path = COMPONENT_PATH) -> None:
    """Append a completed week while preventing duplicate season/week/player/market rows."""
    if frame is None or frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = frame.copy()
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        both = pd.concat([old, new], ignore_index=True)
    else:
        both = new
    keys = ["season", "week", "team", "player_clean_key", "market"]
    both = both.sort_values(keys).drop_duplicates(keys, keep="last")
    both.to_csv(path, index=False)
