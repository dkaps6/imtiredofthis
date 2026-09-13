#!/usr/bin/env python3
"""Football-only evidence helper for stale QB starter authority reconciliation.

This helper never reads sportsbook data. It answers one narrow question: is the
versioned official starter now explicitly marked definitively unavailable by the
current availability artifact? Malformed or duplicate evidence fails closed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.player_identity_v3 import player_name_key

DEFAULT_AVAILABILITY = Path("data/current_player_availability.csv")
_REQUIRED = {
    "team",
    "player",
    "player_clean_key",
    "definitive_unavailable",
    "final_availability_state",
    "availability_authority",
    "availability_reason",
}


def _key(value: object) -> str:
    try:
        return str(player_name_key(value, strip_suffix=True) or "").strip()
    except Exception:
        return ""


def definitive_unavailable_evidence(
    team: str,
    starter: str,
    *,
    path: Path = DEFAULT_AVAILABILITY,
) -> dict[str, str] | None:
    """Return explicit unavailability evidence for ``team/starter`` or ``None``.

    The caller may use a reconciled QB1 fallback only when this returns evidence.
    Missing/malformed/duplicate identity evidence is an error rather than a silent
    fallback. A uniquely matched player who is not definitively unavailable
    returns ``None`` so the existing official-starter authority remains binding.
    """
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"QB starter availability evidence missing: {path}")

    frame = pd.read_csv(path, dtype=str).fillna("")
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = sorted(_REQUIRED - set(frame.columns))
    if missing:
        raise RuntimeError(f"QB starter availability evidence missing columns: {missing}")

    frame["team"] = frame["team"].map(canon_team)
    # player_clean_key is the canonical availability identity, but normalize it
    # through the same QB-C2 identity function so suffix/name treatment matches.
    frame["_identity_key"] = [
        _key(k if str(k).strip() else p)
        for k, p in zip(frame["player_clean_key"], frame["player"])
    ]
    wanted_team = canon_team(team)
    wanted_key = _key(starter)
    if not wanted_team or not wanted_key:
        raise RuntimeError(f"invalid QB starter availability lookup team={team!r} starter={starter!r}")

    match = frame.loc[frame["team"].eq(wanted_team) & frame["_identity_key"].eq(wanted_key)].copy()
    if len(match) != 1:
        raise RuntimeError(
            "QB starter availability identity must match exactly once "
            f"team={wanted_team} starter={starter} matches={len(match)}"
        )

    row = match.iloc[0]
    raw_flag = str(row["definitive_unavailable"]).strip()
    try:
        flag = int(float(raw_flag))
    except Exception as exc:
        raise RuntimeError(
            f"invalid definitive_unavailable flag team={wanted_team} starter={starter}: {raw_flag!r}"
        ) from exc
    if flag not in {0, 1}:
        raise RuntimeError(
            f"invalid definitive_unavailable flag team={wanted_team} starter={starter}: {raw_flag!r}"
        )
    state = str(row["final_availability_state"]).strip()
    if flag == 0:
        return None
    if not state.startswith("UNAVAILABLE_"):
        raise RuntimeError(
            "definitive_unavailable flag/state disagreement "
            f"team={wanted_team} starter={starter} state={state!r}"
        )

    return {
        "team": wanted_team,
        "starter": str(row["player"]).strip(),
        "starter_key": wanted_key,
        "final_availability_state": state,
        "availability_authority": str(row["availability_authority"]).strip(),
        "availability_reason": str(row["availability_reason"]).strip(),
        "sportsbook_inputs_used": "0",
    }
