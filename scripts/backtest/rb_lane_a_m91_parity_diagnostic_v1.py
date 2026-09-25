"""RB Lane A -- M91 parity diagnostic V1.

Implements GPT-5.6's four-step diagnostic protocol (Issue #535, comment
`5703860966`) for isolating the cause of the residual `mc_proj`-only parity
delta between a corrected-invocation fresh rebuild of 2024 and the canonical
M91 artifact, after `ml_proj`/`state_proj` were confirmed to match exactly.

Also implements GPT-5.6's follow-up correction (Issue #535, comment
`5704073674`): the Step-3 exclusion falsification is invalid on its own
because `component_predictions.predict_week()` only returns rows with a
non-null joined `actual` -- so "fresh-only" identifies a scored-OUTPUT
membership difference, not a proven pre-simulation universe difference.
`build_deterministic_trace()` / `compare_deterministic_trace_arm()`
implement the required A/B membership-inference test: build the
RNG-free, pre-`simulate()` metrics frame with the contested player
included vs excluded, and compare each arm's *other* players' deterministic
trace fields against canonical to infer which arm canonical's simulation
universe actually matched.

STRICT SCOPE: this is comparator-integrity diagnostic work only. It computes
no candidate rushing-yard output and makes no candidate-vs-outcome
comparison. Nothing here is a production change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
from unittest.mock import patch

import pandas as pd

from scripts.backtest.component_predictions import (
    _attach_historical_passing_volume,
    _context_trace_frame,
    build_market_frame,
    predict_week,
)
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week
from scripts.modeling import simulation_rules
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline

DETERMINISTIC_TRACE_COLUMNS = [
    "rules_tgt_share",
    "rules_catch_rate",
    "rules_ypt",
    "rules_rush_share",
    "rules_ypc",
    "rules_plays_est",
    "rules_pass_rate",
]

IDENTITY_COLS = ["season", "week", "team", "player_clean_key", "market"]

MC_TRACE_COLUMNS = [
    "event_id",
    "opponent",
    "rules_plays_est",
    "rules_pass_rate",
    "rules_rush_share",
    "rules_ypc",
    "bayes_rush_share",
    "bayes_ypc",
    "mc_projected_plays",
    "mc_dropback_rate",
    "mc_pass_rate",
]


def find_fresh_only_rows(fresh: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    """Step 1: full row detail for identities present in fresh but not canonical."""
    report_cols = IDENTITY_COLS + [c for c in ("player", "position") if c in fresh.columns]
    f = fresh[report_cols].drop_duplicates(IDENTITY_COLS)
    c = canonical[IDENTITY_COLS].drop_duplicates(IDENTITY_COLS)
    merged = f.merge(c.assign(_in_canonical=1), on=IDENTITY_COLS, how="left")
    return merged.loc[merged["_in_canonical"].isna()].drop(columns="_in_canonical").reset_index(drop=True)


def compare_trace_columns_by_week(
    fresh: pd.DataFrame, canonical: pd.DataFrame, columns: Iterable[str] = MC_TRACE_COLUMNS
) -> dict:
    """Step 2: max-abs-delta per trace column, per week, on matched rows only."""
    out = {}
    for week in sorted(set(fresh["week"].unique()) | set(canonical["week"].unique())):
        f = fresh.loc[fresh.week == week]
        c = canonical.loc[canonical.week == week]
        merged = f.merge(c, on=IDENTITY_COLS, how="inner", suffixes=("_fresh", "_canonical"))
        week_report = {"matched_rows": int(len(merged))}
        for col in columns:
            fc, cc = f"{col}_fresh", f"{col}_canonical"
            if fc not in merged.columns or cc not in merged.columns:
                week_report[col] = "column_missing"
                continue
            a = pd.to_numeric(merged[fc], errors="coerce")
            b = pd.to_numeric(merged[cc], errors="coerce")
            delta = (a - b).abs()
            week_report[col] = float(delta.max()) if delta.notna().any() else None
        out[str(int(week))] = week_report
    return out


def rerun_week_excluding_players(
    *,
    season: int,
    week: int,
    prior_season: int,
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    universe_dir: Path,
    injuries_path: Path | None,
    weather_path: Path | None,
    exclude_player_names: Iterable[str],
    iterations: int = 2000,
) -> pd.DataFrame:
    """Step 3 falsification: rebuild one week's MC predictions after removing
    only the fresh-only player identity/identities from the pregame universe,
    preserving the exact same seed policy (`42 + week`) as walk_forward.py.
    """
    universe_path = universe_dir / f"{season}_week_{week:02d}.csv"
    universe = pd.read_csv(universe_path)
    exclude = set(exclude_player_names)
    before = len(universe)
    universe = universe.loc[~universe["player"].isin(exclude)].copy()
    removed = before - len(universe)
    if removed != len(exclude):
        raise RuntimeError(
            f"expected to remove {len(exclude)} players from week {week} universe, removed {removed}"
        )

    injuries_history = pd.read_csv(injuries_path) if injuries_path and injuries_path.exists() else pd.DataFrame()
    weather_history = pd.read_csv(weather_path) if weather_path and weather_path.exists() else pd.DataFrame()
    injuries = _exact_week(injuries_history, season, week)
    weather = _exact_week(weather_history, season, week)

    return predict_week(
        player_logs=player_logs,
        team_weekly=team_weekly,
        pregame_universe=universe,
        schedule=schedule,
        season=season,
        week=week,
        prior_season=prior_season,
        injuries=injuries,
        weather=weather,
        iterations=iterations,
        seed=42 + week,
    )


def compare_common_universe_mc_proj(
    resimulated: pd.DataFrame, canonical: pd.DataFrame, week: int
) -> dict:
    """Compare mc_proj on rows common to both, after excluding the fresh-only
    player(s) from the resimulation. If this collapses to <=1e-6, the
    mc_proj drift is RNG/universe sensitivity from the extra player(s), not
    independent source/simulation-input drift.
    """
    r = resimulated.loc[resimulated.week == week]
    c = canonical.loc[canonical.week == week]
    merged = r.merge(c, on=IDENTITY_COLS, how="inner", suffixes=("_resim", "_canonical"))
    a = pd.to_numeric(merged["mc_proj_resim"], errors="coerce")
    b = pd.to_numeric(merged["mc_proj_canonical"], errors="coerce")
    delta = (a - b).abs()
    return {
        "week": week,
        "matched_rows": int(len(merged)),
        "max_abs_delta_mc_proj": float(delta.max()) if delta.notna().any() else None,
        "collapses_to_tolerance": bool(delta.max() <= 1e-6) if delta.notna().any() else False,
    }


def build_deterministic_trace(
    *,
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    pregame_universe: pd.DataFrame,
    schedule: pd.DataFrame,
    season: int,
    week: int,
    prior_season: int,
    injuries: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reproduce component_predictions.build_mc_predictions()'s deterministic,
    RNG-free metrics frame -- identical logic, calling the exact same
    production functions, stopping before the `simulate()` MC step. Used to
    test whether canonical's simulation universe included or excluded a
    contested player, without any RNG-sensitive re-simulation.
    """
    bundle = build_historical_context_bundle(
        player_logs=player_logs,
        team_weekly=team_weekly,
        pregame_universe=pregame_universe,
        schedule=schedule,
        season=int(season),
        week=int(week),
        prior_season=int(prior_season),
        injuries=injuries,
        weather=weather,
    )
    metrics = build_market_frame(bundle)
    bayes = build_bayesian_baseline(bundle.player_consensus)
    metrics = apply_bayesian_to_metrics(metrics, bayes)
    with patch.object(simulation_rules, "load_model_contexts", return_value=(bundle.teams, bundle.players)):
        metrics = simulation_rules.apply_rules_to_metrics(metrics)
    metrics = _attach_historical_passing_volume(metrics, bundle)
    trace = _context_trace_frame(bundle)
    metrics = metrics.merge(trace, on=["team", "player_clean_key"], how="left", validate="many_to_one")
    metrics["season"] = int(season)
    metrics["week"] = int(week)
    return metrics


def compare_deterministic_trace_arm(
    arm_metrics: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    week: int,
    team: str,
    exclude_player_clean_key: str,
    columns: Iterable[str] = DETERMINISTIC_TRACE_COLUMNS,
) -> dict:
    """Compare one arm's deterministic trace (excluding the contested player's
    own rows) against canonical's trace for the same team/week, on matched
    OTHER-player rows only. Whichever arm (contested player included vs
    excluded from the pregame universe) minimizes these deltas is the arm
    whose universe membership canonical's simulation actually matched.
    """
    a = arm_metrics.loc[
        (arm_metrics["team"] == team)
        & (pd.to_numeric(arm_metrics["week"], errors="coerce") == week)
        & (arm_metrics["player_clean_key"] != exclude_player_clean_key)
    ]
    c = canonical.loc[
        (canonical["team"] == team)
        & (pd.to_numeric(canonical["week"], errors="coerce") == week)
        & (canonical["player_clean_key"] != exclude_player_clean_key)
    ]
    merged = a.merge(
        c,
        on=["season", "week", "team", "player_clean_key", "market"],
        how="inner",
        suffixes=("_arm", "_canonical"),
    )
    report = {"week": week, "team": team, "matched_rows": int(len(merged))}
    for col in columns:
        ac, cc = f"{col}_arm", f"{col}_canonical"
        if ac not in merged.columns or cc not in merged.columns:
            report[col] = "column_missing"
            continue
        delta = (
            pd.to_numeric(merged[ac], errors="coerce") - pd.to_numeric(merged[cc], errors="coerce")
        ).abs()
        report[col] = float(delta.max()) if delta.notna().any() else None
    return report
