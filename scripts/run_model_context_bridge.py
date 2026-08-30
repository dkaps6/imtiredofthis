#!/usr/bin/env python3
"""Validate and materialize the canonical model-context bridge.

This runner does not create projections. It proves that the current Full Slate
artifacts can be translated into the canonical modeling contracts introduced in
Migration 1. Team-level context comes exclusively from Team Context v3, and the
runtime provider artifacts must pass the 2026 readiness gate first.
"""
from __future__ import annotations

import os
from pathlib import Path

from scripts.modeling.context_bridge_v3 import load_model_contexts, player_context_frame
from scripts.runtime_context import resolve_prior_season, resolve_season, resolve_week
from scripts.team_context_v3 import materialize as materialize_team_context
from scripts.validate_2026_provider_artifacts import run_provider_readiness

OUT = Path("data/model_context_bridge.csv")
PROVIDER_OUT = Path("data/provider_readiness_v3.csv")


def main() -> int:
    season = int(resolve_season())
    prior = int(resolve_prior_season())
    week = int(resolve_week())
    live_odds = os.getenv("FETCH_LIVE_ODDS", "false").strip().lower() in {"1", "true", "yes", "on"}

    provider_summary = run_provider_readiness(
        season,
        prior,
        week,
        live_odds_enabled=live_odds,
    )
    PROVIDER_OUT.parent.mkdir(parents=True, exist_ok=True)
    provider_summary.to_csv(PROVIDER_OUT, index=False)
    print(
        f"[model_context_bridge] provider gate passed rows={len(provider_summary)} "
        f"season={season} week={week} live_odds={live_odds}"
    )

    team_context, provenance = materialize_team_context(season=season, week=week)
    if len(team_context) != 32 or team_context["team"].nunique() != 32:
        raise RuntimeError("Team Context v3 failed 32-team materialization contract")
    if provenance.empty:
        raise RuntimeError("Team Context v3 provenance is empty")

    teams, players = load_model_contexts()
    if not teams:
        raise RuntimeError("Model context bridge produced zero TeamContext objects")
    if not players:
        raise RuntimeError("Model context bridge produced zero PlayerContext objects")

    unresolved = [p for p in players if not p.team or not p.opponent or p.offense is None or p.defense is None]
    if unresolved:
        sample = [(p.player, p.team, p.opponent) for p in unresolved[:10]]
        raise RuntimeError(f"Model context bridge contains unresolved active player contexts: {sample}")

    seasons = sorted({p.season for p in players})
    weeks = sorted({p.week for p in players})
    if len(seasons) != 1 or len(weeks) != 1:
        raise RuntimeError(f"Model context bridge mixed runtime contexts seasons={seasons} weeks={weeks}")

    frame = player_context_frame(players)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)

    cov = int(frame.get("team_coverage_available", 0).fillna(0).sum()) if "team_coverage_available" in frame else 0
    direct = int(frame.get("matchup_available", 0).fillna(0).sum()) if "matchup_available" in frame else 0
    injured = int(frame.get("injury_report_available", 0).fillna(0).sum()) if "injury_report_available" in frame else 0
    weather = int(frame.get("weather_forecast_available", 0).fillna(0).sum()) if "weather_forecast_available" in frame else 0

    print(
        f"[model_context_bridge] teams={len(teams)} players={len(players)} "
        f"season={seasons[0]} week={weeks[0]} team_source=TEAM_CONTEXT_V3"
    )
    print(
        f"[model_context_bridge] team_coverage_player_rows={cov} direct_wr_cb_rows={direct} "
        f"injury_report_player_rows={injured} weather_forecast_player_rows={weather}"
    )
    print(f"[model_context_bridge] team_provenance_rows={len(provenance)}")
    print(f"[model_context_bridge] wrote {len(frame)} rows -> {OUT}")
    print("[model_context_bridge] projection-neutral: simulation_v2/pricing behavior unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
