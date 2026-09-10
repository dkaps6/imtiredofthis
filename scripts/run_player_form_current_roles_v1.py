#!/usr/bin/env python3
"""Run protected PlayerForm against explicit current roles and strict-prior history publication.

Only operational seams change:
- current-role input path is explicit;
- published history artifacts are restricted to the same strict-prior rows already
  used by PlayerForm's model blend.

PlayerForm identity/history/blend formulas are unchanged.
"""
from __future__ import annotations

import pandas as pd

from scripts.runtime_context import resolve_prior_season, resolve_season, resolve_slate_date, resolve_week
from scripts.utils.current_roles_v1 import resolve_current_roles_path
import scripts.run_player_form_v2_loader as loader


def strict_prior_logs(logs: pd.DataFrame, *, season: int, prior_season: int, week: int) -> pd.DataFrame:
    """Return exactly the history legally available to the target slate week."""
    if logs.empty:
        return logs.copy()
    s = pd.to_numeric(logs["season"], errors="coerce")
    w = pd.to_numeric(logs["week"], errors="coerce")
    keep = s.eq(int(prior_season)) | (s.eq(int(season)) & w.lt(int(week)))
    return logs.loc[keep].copy()


def publish_strict_prior_history() -> None:
    season = int(resolve_season())
    prior = int(resolve_prior_season())
    slate_date = resolve_slate_date() or ""
    week = int(resolve_week(season=season, slate_date=slate_date))

    logs_path = loader.runner.pf.DATA / "player_game_logs.csv"
    totals_path = loader.runner.pf.DATA / "player_season_totals.csv"
    if not logs_path.exists() or logs_path.stat().st_size == 0:
        raise RuntimeError("PlayerForm strict-prior publication requires player_game_logs.csv")

    logs = pd.read_csv(logs_path)
    strict = strict_prior_logs(logs, season=season, prior_season=prior, week=week)
    dropped = int(len(logs) - len(strict))

    # The core PlayerForm model already used this exact subset for its prior/current
    # blend. Re-publishing it here changes diagnostics/provider artifacts only.
    totals = loader.runner.pf._season_totals(strict)
    strict.to_csv(logs_path, index=False)
    totals.to_csv(totals_path, index=False)

    s = pd.to_numeric(strict.get("season"), errors="coerce")
    w = pd.to_numeric(strict.get("week"), errors="coerce")
    illegal = s.eq(season) & w.ge(week)
    if bool(illegal.fillna(False).any()):
        sample = strict.loc[illegal.fillna(False), [c for c in ("season", "week", "player", "team") if c in strict.columns]].head(20).to_dict("records")
        raise RuntimeError(f"strict-prior PlayerForm publication retained illegal rows: {sample}")

    print(
        "[current_roles_v1] strict-prior history publication "
        f"season={season} week={week} prior={prior} rows={len(strict)} dropped_same_or_future={dropped}"
    )


def main() -> int:
    path = resolve_current_roles_path(require_active=True)
    loader.runner.pf.ROLES = path
    print(f"[current_roles_v1] PlayerForm roles={path}")
    rc = int(loader.main())
    if rc != 0:
        return rc
    publish_strict_prior_history()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
