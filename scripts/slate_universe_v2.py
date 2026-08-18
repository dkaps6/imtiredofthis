"""Build the active PlayerForm slate universe in live-market or no-market mode."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.artifact_io import read_valid_csv


def _canon_player_frame(pf, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    canon = out["player"].map(pf._canon_name)
    out["player"] = canon.map(lambda t: t[0])
    out["player_clean_key"] = canon.map(lambda t: t[1])
    return out


def build_slate_universe(
    pf,
    load_schedule: Callable[[], pd.DataFrame],
    season: int,
    week: int,
    *,
    live_odds_enabled: bool,
) -> pd.DataFrame:
    """Return one roster/slate row per team/player for the requested NFL week.

    Contract:
    - no-odds mode never consumes props/odds/opponent-map placeholders;
    - no-odds mode derives the universe from Ourlads and opponent from schedule;
    - live-odds mode requires a valid, non-empty props artifact;
    - schedule remains authoritative for team opponent identity in both modes;
    - unresolved team/opponent identity is always fatal.
    """
    roles = pf._load_roles()
    if roles.empty:
        raise RuntimeError("Ourlads role universe is empty")

    schedule = load_schedule()
    cur = schedule.loc[
        (pd.to_numeric(schedule["season"], errors="coerce") == int(season))
        & (pd.to_numeric(schedule["week"], errors="coerce") == int(week)),
        ["team", "opponent"],
    ].copy()
    cur["team"] = cur["team"].map(canon_team)
    cur["opponent"] = cur["opponent"].map(canon_team)
    if cur.empty:
        raise RuntimeError(f"No schedule rows available for active slate season={season} week={week}")
    if cur["team"].duplicated().any():
        dupes = cur.loc[cur["team"].duplicated(keep=False)].to_dict("records")
        raise RuntimeError(f"Active schedule is not unique by team: {dupes[:20]}")

    if not live_odds_enabled:
        print("[slate_universe_v2] live odds disabled; building player universe from Ourlads + authoritative schedule")
        base = roles.rename(columns={"display_name": "player"})[["player", "player_clean_key", "team"]].copy()
        base["team"] = base["team"].map(canon_team)
        base = base.merge(cur, on="team", how="left", validate="many_to_one")
    else:
        props = read_valid_csv(
            pf.PROPS,
            required_columns=("player",),
            min_rows=1,
            required=True,
            label="live props_raw",
        )
        assert props is not None
        props = _canon_player_frame(pf, props)
        props["team"] = pf._first(props, ["team_abbr", "team", "player_team_abbr"], "").map(canon_team)
        keep = [c for c in ("event_id", "player", "player_clean_key", "team") if c in props.columns]
        base = props[keep].drop_duplicates().copy()

        # Use live enriched/team identity only when valid. These are helpers, not
        # authority: the active schedule still provides final opponent identity.
        enriched = read_valid_csv(
            pf.DATA / "props_enriched.csv",
            min_rows=1,
            required=False,
            label="props_enriched",
        )
        if enriched is not None:
            name_col = "player_canonical" if "player_canonical" in enriched.columns else "player_name_raw" if "player_name_raw" in enriched.columns else None
            if name_col:
                enriched["player_clean_key"] = enriched[name_col].map(pf._canon_name).map(lambda t: t[1])
                if "player_team_abbr" in enriched.columns:
                    enriched["team_enriched"] = enriched["player_team_abbr"].map(canon_team)
                    join = [c for c in ("event_id", "player_clean_key") if c in base.columns and c in enriched.columns] or ["player_clean_key"]
                    right = enriched[join + ["team_enriched"]].drop_duplicates(join, keep="last")
                    base = base.merge(right, on=join, how="left")
                    base["team"] = base["team"].replace("", pd.NA).combine_first(base["team_enriched"])

        # Ourlads can resolve team when player identity is unique across the roster.
        role_unique = roles.groupby("player_clean_key")["team"].nunique()
        unique_keys = set(role_unique.loc[role_unique.eq(1)].index)
        role_team = roles.loc[
            roles["player_clean_key"].isin(unique_keys),
            ["player_clean_key", "team"],
        ].drop_duplicates("player_clean_key")
        base = base.merge(role_team.rename(columns={"team": "team_roster"}), on="player_clean_key", how="left")
        base["team"] = base["team"].replace("", pd.NA).combine_first(base["team_roster"])
        base = base.merge(cur, on="team", how="left", validate="many_to_one")

    missing = (
        base["team"].isna()
        | base["team"].astype("string").str.strip().eq("")
        | base["opponent"].isna()
        | base["opponent"].astype("string").str.strip().eq("")
    )
    if missing.any():
        path = pf.DATA / "_debug" / "player_form_unresolved_slate_identity.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        base.loc[missing].to_csv(path, index=False)
        raise RuntimeError(
            f"PlayerForm slate identity unresolved for {int(missing.sum())} rows; see {path}"
        )

    # Attach current Ourlads metadata after slate identity is resolved.
    base = base.merge(roles, on=["team", "player_clean_key"], how="left", suffixes=("", "_role"))
    if "display_name" in base.columns:
        base["player"] = base["player"].replace("", pd.NA).combine_first(base["display_name"])
    base["season"] = int(season)
    base["week"] = int(week)

    drop = [c for c in ("team_enriched", "team_roster") if c in base.columns]
    base.drop(columns=drop, inplace=True, errors="ignore")
    base = base.drop_duplicates(["team", "player_clean_key"])
    print(
        f"[slate_universe_v2] mode={'live_odds' if live_odds_enabled else 'roster_schedule'} "
        f"season={season} week={week} players={len(base)} teams={base['team'].nunique()}"
    )
    return base
