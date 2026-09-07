"""Build the active PlayerForm slate universe in live-market or no-market mode."""
from __future__ import annotations

from typing import Callable

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.artifact_io import read_valid_csv


def _canon_player_frame(pf, frame: pd.DataFrame, *, source_col: str = "player") -> pd.DataFrame:
    out = frame.copy()
    if source_col not in out.columns:
        raise RuntimeError(f"PlayerForm live slate missing player source column: {source_col}")
    canon = out[source_col].map(pf._canon_name)
    out["player"] = canon.map(lambda t: t[0])
    out["player_clean_key"] = canon.map(lambda t: t[1])
    return out


def _live_prop_player_column(props: pd.DataFrame) -> str:
    """Resolve the maintained OddsAPI player-name contract without guessing silently."""
    for col in ("canonical_player_name", "player_canonical", "player", "book_player_name"):
        if col in props.columns:
            return col
    raise RuntimeError(
        "Live props contain no supported player-name column; "
        f"columns={sorted(str(c) for c in props.columns)}"
    )


def _active_live_props(pf, props: pd.DataFrame) -> pd.DataFrame:
    """Return only real posted player offers with canonical, current-slate identity.

    Bookmaker-missing rows are provider sentinels, not players. They are excluded
    explicitly. Any *real* posted row missing canonical player/team/opponent
    identity is fatal; PlayerForm must never turn an unmapped sportsbook row into
    a silent name-only fallback.
    """
    out = props.copy()
    if "bookmaker_missing" in out.columns:
        placeholder = pd.to_numeric(out["bookmaker_missing"], errors="coerce").fillna(0).eq(1)
        out = out.loc[~placeholder].copy()
    if out.empty:
        raise RuntimeError("Live props contain zero real posted player rows after excluding bookmaker sentinels")

    source_col = _live_prop_player_column(out)
    source = out[source_col].astype("string").fillna("").str.strip()
    if source.eq("").any():
        sample = out.loc[source.eq(""), [c for c in ("event_id", "market", source_col) if c in out.columns]].head(20)
        raise RuntimeError(
            "Real live prop rows contain blank player identity; "
            f"rows={int(source.eq('').sum())} sample={sample.to_dict('records')}"
        )

    out = _canon_player_frame(pf, out, source_col=source_col)
    key = out["player_clean_key"].astype("string").fillna("").str.strip()
    if key.eq("").any():
        sample = out.loc[key.eq(""), [c for c in ("event_id", "market", source_col) if c in out.columns]].head(20)
        raise RuntimeError(
            "Real live prop rows failed canonical PlayerForm identity; "
            f"rows={int(key.eq('').sum())} sample={sample.to_dict('records')}"
        )

    out["team"] = pf._first(out, ["team_abbr", "team", "player_team_abbr"], "").map(canon_team)
    out["opponent_prop"] = pf._first(out, ["opponent_abbr", "opponent", "opponent_team_abbr"], "").map(canon_team)
    team_bad = out["team"].isna() | out["team"].astype("string").str.strip().eq("")
    opp_bad = out["opponent_prop"].isna() | out["opponent_prop"].astype("string").str.strip().eq("")
    if team_bad.any() or opp_bad.any():
        bad = team_bad | opp_bad
        sample = out.loc[
            bad,
            [c for c in ("event_id", "market", source_col, "team_abbr", "opponent_abbr") if c in out.columns],
        ].head(30)
        raise RuntimeError(
            "Real live prop rows contain unresolved current team/opponent identity; "
            f"rows={int(bad.sum())} sample={sample.to_dict('records')}"
        )
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
    - live-odds mode consumes the maintained canonical OddsAPI schema;
    - bookmaker-missing sentinel rows are excluded rather than treated as players;
    - every real live offer must resolve canonical player/team/opponent identity;
    - schedule remains authoritative for final opponent identity in both modes;
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
            min_rows=1,
            required=True,
            label="live props_raw",
        )
        assert props is not None
        props = _active_live_props(pf, props)
        keep = [c for c in ("event_id", "player", "player_clean_key", "team") if c in props.columns]
        base = props[keep].drop_duplicates().copy()

        # Ourlads can resolve roster metadata when player identity is unique across
        # the current roster. Team identity itself has already been fail-closed at
        # the sportsbook boundary above.
        role_unique = roles.groupby("player_clean_key")["team"].nunique()
        unique_keys = set(role_unique.loc[role_unique.eq(1)].index)
        role_team = roles.loc[
            roles["player_clean_key"].isin(unique_keys),
            ["player_clean_key", "team"],
        ].drop_duplicates("player_clean_key")
        base = base.merge(role_team.rename(columns={"team": "team_roster"}), on="player_clean_key", how="left")

        # The authoritative schedule supplies final opponent identity. It must
        # agree with the already-repaired prop team identity by construction.
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

    drop = [c for c in ("team_roster",) if c in base.columns]
    base.drop(columns=drop, inplace=True, errors="ignore")
    base = base.drop_duplicates(["team", "player_clean_key"])
    if base.empty:
        raise RuntimeError("PlayerForm active slate universe produced zero players")
    if base["team"].nunique() < 2:
        raise RuntimeError(f"PlayerForm active slate universe has implausible team coverage: {base['team'].nunique()}")
    print(
        f"[slate_universe_v2] mode={'live_odds' if live_odds_enabled else 'roster_schedule'} "
        f"season={season} week={week} players={len(base)} teams={base['team'].nunique()}"
    )
    return base
