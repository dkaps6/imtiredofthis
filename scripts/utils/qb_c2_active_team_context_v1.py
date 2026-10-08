"""Schedule-grounded QB C2 active-team context V1.

The immutable model/mean/selector stays untouched. This validates an active
NFL slate against authoritative full-season schedule and the separately
certified eligible-team roster; bye teams never receive fabricated opponents.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scripts._opponent_map import canon_team
from scripts.utils.eligible_team_set_v1 import expected_current_teams

TEAM_MAP = Path("data/team_week_map.csv")
SPOTS = (
    "pass_opportunity_spot", "pass_efficiency_spot",
    "rush_opportunity_spot", "rush_efficiency_spot",
)

def active_qb_c2_schedule(
    schedule: pd.DataFrame, *, season: int, week: int,
    expected_teams: set[str] | None = None,
) -> pd.DataFrame:
    x = schedule.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "team", "opponent"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"QB C2 schedule missing {sorted(missing)}")
    x = x.loc[
        pd.to_numeric(x["season"], errors="coerce").eq(int(season))
        & pd.to_numeric(x["week"], errors="coerce").eq(int(week))
    ].copy()
    if "bye" in x.columns:
        bye = x["bye"].fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))
        x = x.loc[~bye].copy()
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    if x["team"].eq("").any() or x["opponent"].eq("").any() or x["team"].eq(x["opponent"]).any():
        raise RuntimeError("QB C2 active schedule has blank/self opponents")
    if x["team"].duplicated().any():
        raise RuntimeError("QB C2 active schedule has duplicate team")
    n = len(x)
    if n not in {26, 28, 30, 32}:
        raise RuntimeError(f"QB C2 active schedule has invalid team count {n}")
    mapping = dict(zip(x["team"], x["opponent"]))
    for team, opp in mapping.items():
        if mapping.get(opp) != team:
            raise RuntimeError(f"QB C2 schedule opponent reciprocity failed team={team} opp={opp}")
    reference = expected_teams if expected_teams is not None else expected_current_teams()
    if reference is None:
        # Legacy non-availability mode is specifically full league. Current
        # production always supplies an explicit certified eligible roster.
        if n != 32:
            raise RuntimeError("QB C2 bye-week schedule requires ACTIVE_ROLES_CSV authority")
    else:
        expected = {canon_team(t) for t in reference}
        if not expected or len(expected) % 2:
            raise RuntimeError("QB C2 certified eligible game set empty or odd-sized")
        extra = sorted(expected - set(mapping))
        if extra:
            raise RuntimeError(
                f"QB C2 certified active roles contain team absent from schedule: {extra}"
            )
        # T-75/started-game eligibility correctly withholds BOTH teams of a
        # game, but the authoritative entire-week schedule must stay intact.
        # Scope only to the certified games after validating the whole week.
        partial = sorted(t for t in expected if mapping[t] not in expected)
        if partial:
            raise RuntimeError(
                f"QB C2 certified eligible set includes half a game: {partial}"
            )
        x = x.loc[x["team"].isin(expected)].copy()
    return x


def validate_qb_c2_state_context(
    context: pd.DataFrame, *, season: int, week: int,
    schedule_path: Path = TEAM_MAP,
    expected_teams: set[str] | None = None,
) -> dict:
    if not schedule_path.is_file() or schedule_path.stat().st_size <= 0:
        raise RuntimeError(f"QB C2 authoritative schedule missing: {schedule_path}")
    pairs = active_qb_c2_schedule(
        pd.read_csv(schedule_path, low_memory=False),
        season=season, week=week, expected_teams=expected_teams,
    )
    x = context.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "team", "opponent", "sportsbook_inputs_used"} | set(SPOTS)
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"QB C2 state context missing {sorted(missing)}")
    x = x.loc[
        pd.to_numeric(x["season"], errors="coerce").eq(int(season))
        & pd.to_numeric(x["week"], errors="coerce").eq(int(week))
    ].copy()
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    if len(x) != len(pairs) or x["team"].duplicated().any():
        raise RuntimeError(f"QB C2 context coverage invalid rows={len(x)} expected={len(pairs)}")
    found = dict(zip(x["team"], x["opponent"]))
    canonical = dict(zip(pairs["team"], pairs["opponent"]))
    if found != canonical:
        raise RuntimeError(
            f"QB C2 context schedule mismatch missing={sorted(set(canonical)-set(found))} "
            f"extra={sorted(set(found)-set(canonical))}"
        )
    if not pd.to_numeric(x["sportsbook_inputs_used"], errors="coerce").eq(0).all():
        raise RuntimeError("QB C2 state context sportsbook leakage")
    values = x[list(SPOTS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("QB C2 state context non-finite spot values")
    return {
        "version": "QB_C2_ACTIVE_SCHEDULE_CONTEXT_V1",
        "teams": len(pairs),
        "games": len(pairs) // 2,
        "bye_teams": 32-len(pairs),
        "schedule_reciprocity": True,
        "certified_roster_parity": True,
        "sportsbook_inputs_used": 0,
    }
