"""RB Lane A -- M91 parity diagnostic V1.

Implements GPT-5.6's four-step diagnostic protocol (Issue #535, comment
`5703860966`) for isolating the cause of the residual `mc_proj`-only parity
delta between a corrected-invocation fresh rebuild of 2024 and the canonical
M91 artifact, after `ml_proj`/`state_proj` were confirmed to match exactly.

STRICT SCOPE: this is comparator-integrity diagnostic work only. It computes
no candidate rushing-yard output and makes no candidate-vs-outcome
comparison. Nothing here is a production change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from scripts.backtest.component_predictions import predict_week
from scripts.backtest.walk_forward import _exact_week

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
