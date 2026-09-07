"""Build the active PlayerForm slate universe from football roster + schedule only."""
from __future__ import annotations

from typing import Callable

import pandas as pd

from scripts._opponent_map import canon_team


def build_slate_universe(
    pf,
    load_schedule: Callable[[], pd.DataFrame],
    season: int,
    week: int,
    *,
    live_odds_enabled: bool,
) -> pd.DataFrame:
    """Return one football-defined player row per team/player for the active week.

    Production contract:
    - sportsbook availability NEVER defines which players the football model knows;
    - the active player universe is current Ourlads offensive-skill roles;
    - the authoritative schedule supplies opponent identity;
    - live props, when enabled, are validated/quarantined separately upstream and
      are joined only later when deciding which already-modeled markets to price;
    - unresolved current team/opponent/player identity is fatal.

    ``live_odds_enabled`` is retained only for runtime logging/backward-compatible
    call signatures. It has no effect on the PlayerForm universe.
    """
    roles = pf._load_roles()
    if roles.empty:
        raise RuntimeError("Ourlads role universe is empty")
    required_roles = {"team", "player", "player_clean_key"}
    missing_roles = required_roles - set(roles.columns)
    if missing_roles:
        raise RuntimeError(f"Ourlads role universe missing columns: {sorted(missing_roles)}")

    roles = roles.copy()
    roles["team"] = roles["team"].map(canon_team)
    player = roles["player"].astype("string").fillna("").str.strip()
    key = roles["player_clean_key"].astype("string").fillna("").str.strip()
    if roles["team"].eq("").any() or player.eq("").any() or key.eq("").any():
        raise RuntimeError("Ourlads role universe contains unresolved team/player identity")
    if roles.duplicated(["team", "player_clean_key"]).any():
        sample = roles.loc[
            roles.duplicated(["team", "player_clean_key"], keep=False),
            ["team", "player", "player_clean_key"],
        ].head(20).to_dict("records")
        raise RuntimeError(f"Ourlads role universe contains duplicate current identities: {sample}")

    schedule = load_schedule()
    required_sched = {"season", "week", "team", "opponent"}
    missing_sched = required_sched - set(schedule.columns)
    if missing_sched:
        raise RuntimeError(f"active schedule missing columns: {sorted(missing_sched)}")
    cur = schedule.loc[
        (pd.to_numeric(schedule["season"], errors="coerce") == int(season))
        & (pd.to_numeric(schedule["week"], errors="coerce") == int(week)),
        ["team", "opponent"],
    ].copy()
    cur["team"] = cur["team"].map(canon_team)
    cur["opponent"] = cur["opponent"].map(canon_team)
    if cur.empty:
        raise RuntimeError(f"No schedule rows available for active slate season={season} week={week}")
    if cur["team"].eq("").any() or cur["opponent"].eq("").any():
        raise RuntimeError("active schedule contains unresolved team/opponent identity")
    if cur["team"].duplicated().any():
        dupes = cur.loc[cur["team"].duplicated(keep=False)].to_dict("records")
        raise RuntimeError(f"Active schedule is not unique by team: {dupes[:20]}")

    # Football model universe is independent of sportsbook posting coverage.
    base = roles[["player", "player_clean_key", "team"]].copy()
    base = base.merge(cur, on="team", how="inner", validate="many_to_one")
    if base.empty:
        raise RuntimeError("Ourlads + active schedule produced zero PlayerForm players")

    missing = (
        base["team"].isna()
        | base["team"].astype("string").str.strip().eq("")
        | base["opponent"].isna()
        | base["opponent"].astype("string").str.strip().eq("")
        | base["player_clean_key"].isna()
        | base["player_clean_key"].astype("string").str.strip().eq("")
    )
    if missing.any():
        path = pf.DATA / "_debug" / "player_form_unresolved_slate_identity.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        base.loc[missing].to_csv(path, index=False)
        raise RuntimeError(
            f"PlayerForm slate identity unresolved for {int(missing.sum())} rows; see {path}"
        )

    # Attach full current Ourlads metadata after identity/schedule resolution.
    base = base.merge(roles, on=["team", "player_clean_key"], how="left", suffixes=("", "_role"), validate="one_to_one")
    if "player_role" in base.columns:
        base["player"] = base["player"].replace("", pd.NA).combine_first(base["player_role"])
    base["season"] = int(season)
    base["week"] = int(week)
    base = base.drop_duplicates(["team", "player_clean_key"])

    if base["team"].nunique() < 24:
        raise RuntimeError(f"PlayerForm active slate universe has implausible team coverage: {base['team'].nunique()}")
    print(
        "[slate_universe_v2] sportsbook_independent=1 "
        f"live_odds_enabled={int(bool(live_odds_enabled))} season={season} week={week} "
        f"players={len(base)} teams={base['team'].nunique()} source=OURLADS_PLUS_SCHEDULE"
    )
    return base
