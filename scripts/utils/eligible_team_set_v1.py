"""Availability-aware expected current team-set utility.

Default/no-availability behavior remains the legacy 32-team contract. When an
explicit ACTIVE_ROLES_CSV is configured, that certified current-role artifact is
the sole authority for the current production-eligible team set.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team

LEGACY_TEAM_COUNT = 32


def explicit_active_roles_path() -> Path | None:
    raw = str(os.environ.get("ACTIVE_ROLES_CSV", "")).strip()
    return Path(raw) if raw else None


def expected_current_teams(*, active_roles_path: Path | None = None) -> set[str] | None:
    """Return explicit eligible teams, or None to mean legacy 32-team mode."""
    path = active_roles_path if active_roles_path is not None else explicit_active_roles_path()
    if path is None:
        return None
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"explicit active-role artifact missing/empty: {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "team" not in frame.columns:
        raise RuntimeError("explicit active-role artifact missing team")
    teams = {canon_team(x) for x in frame["team"].dropna().astype(str)}
    teams.discard("")
    if not teams or len(teams) % 2:
        raise RuntimeError(f"explicit eligible-team set invalid count={len(teams)} teams={sorted(teams)}")
    return teams


def validate_current_team_set(observed, *, active_roles_path: Path | None = None, label: str = "current football universe") -> dict:
    """Require every certified-eligible team to be present in ``observed``.

    ``observed`` is allowed to be a superset of the certified-eligible teams.
    The football-only layers this guards (PlayerForm's full simulation
    universe, R26/QB-C2 context, ...) are contracted to cover every rostered
    team regardless of sportsbook/kickoff-timing availability; only pricing
    actually narrows to teams with live offers, later and separately. A
    stricter set-equality check here would require football-only artifacts to
    be pre-narrowed to the eligible subset, which nothing in this pipeline
    does -- and shouldn't, since that would make "which players the football
    model knows" depend on sportsbook availability again.
    """
    obs = {canon_team(x) for x in observed if str(x).strip()}
    obs.discard("")
    expected = expected_current_teams(active_roles_path=active_roles_path)
    if expected is None:
        if len(obs) != LEGACY_TEAM_COUNT:
            raise RuntimeError(f"{label} legacy coverage expected {LEGACY_TEAM_COUNT} teams, got {len(obs)}")
        return {"mode": "LEGACY_32_TEAM", "expected_teams": LEGACY_TEAM_COUNT, "observed_teams": len(obs)}
    missing = sorted(expected - obs)
    extra = sorted(obs - expected)
    if missing:
        raise RuntimeError(f"{label} missing certified eligible teams: {missing}")
    return {
        "mode": "EXPLICIT_CURRENT_AVAILABILITY",
        "expected_teams": len(expected),
        "observed_teams": len(obs),
        "extra_non_eligible_teams": extra,
        "canonical_games": len(expected) // 2,
        "teams": sorted(expected),
    }
