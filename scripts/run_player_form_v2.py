#!/usr/bin/env python3
"""Run PlayerForm v2 with robust current player/team identity resolution."""
from __future__ import annotations

import pandas as pd

import scripts.player_form_v2 as pf
from scripts._opponent_map import canon_team


def _enhanced_slate_universe(season: int, week: int) -> pd.DataFrame:
    roles = pf._load_roles()
    base = pd.DataFrame()

    if pf.PROPS.exists() and pf.PROPS.stat().st_size > 0:
        p = pd.read_csv(pf.PROPS)
        p.columns = [str(c).lower() for c in p.columns]
        if "player" in p.columns:
            canon = p["player"].map(pf._canon_name)
            p["player"] = canon.map(lambda t: t[0])
            p["player_clean_key"] = canon.map(lambda t: t[1])
            p["team"] = pf._first(p, ["team_abbr", "team", "player_team_abbr"], "").map(canon_team)
            p["opponent"] = pf._first(p, ["opponent_abbr", "opponent", "opponent_team_abbr"], "").map(canon_team)
            keep = [c for c in ("event_id", "player", "player_clean_key", "team", "opponent") if c in p.columns]
            base = p[keep].drop_duplicates()

    if base.empty:
        base = roles.rename(columns={"display_name": "player"})[["player", "player_clean_key", "team"]].copy()
        base["opponent"] = pd.NA

    # Props-enriched identity is the strongest sportsbook-event source.
    enriched_path = pf.DATA / "props_enriched.csv"
    if enriched_path.exists() and enriched_path.stat().st_size > 0:
        e = pd.read_csv(enriched_path)
        e.columns = [str(c).lower() for c in e.columns]
        name_col = "player_canonical" if "player_canonical" in e.columns else "player_name_raw" if "player_name_raw" in e.columns else None
        if name_col:
            e["player_clean_key"] = e[name_col].map(pf._canon_name).map(lambda t: t[1])
            if "player_team_abbr" in e.columns:
                e["team_enriched"] = e["player_team_abbr"].map(canon_team)
            if "opponent_team_abbr" in e.columns:
                e["opponent_enriched"] = e["opponent_team_abbr"].map(canon_team)
            join = [c for c in ("event_id", "player_clean_key") if c in base.columns and c in e.columns]
            if not join:
                join = ["player_clean_key"]
            cols = join + [c for c in ("team_enriched", "opponent_enriched") if c in e.columns]
            if len(cols) > len(join):
                base = base.merge(e[cols].drop_duplicates(join, keep="last"), on=join, how="left")
                if "team_enriched" in base.columns:
                    base["team"] = base["team"].replace("", pd.NA).combine_first(base["team_enriched"])
                if "opponent_enriched" in base.columns:
                    base["opponent"] = base["opponent"].replace("", pd.NA).combine_first(base["opponent_enriched"])

    # Live opponent map provides another canonical current-event identity source.
    if pf.OPPONENT_MAP.exists() and pf.OPPONENT_MAP.stat().st_size > 0:
        om = pd.read_csv(pf.OPPONENT_MAP)
        om.columns = [str(c).lower() for c in om.columns]
        if "player_clean_key" not in om.columns and "player" in om.columns:
            om["player_clean_key"] = om["player"].map(pf._canon_name).map(lambda t: t[1])
        if "season" in om.columns:
            om["season"] = pd.to_numeric(om["season"], errors="coerce").astype("Int64")
        if "week" in om.columns:
            om["week"] = pd.to_numeric(om["week"], errors="coerce").astype("Int64")
        if {"season", "week"}.issubset(om.columns):
            om = om.loc[(om["season"] == int(season)) & (om["week"] == int(week))].copy()
        for c in ("team", "opponent"):
            if c in om.columns:
                om[c] = om[c].map(canon_team)
        join = [c for c in ("event_id", "player_clean_key") if c in base.columns and c in om.columns]
        if not join and "player_clean_key" in om.columns:
            join = ["player_clean_key"]
        if join:
            cols = join + [c for c in ("team", "opponent") if c in om.columns]
            right = om[cols].drop_duplicates(join, keep="last").rename(columns={"team": "team_map", "opponent": "opponent_map"})
            base = base.merge(right, on=join, how="left")
            if "team_map" in base.columns:
                base["team"] = base["team"].replace("", pd.NA).combine_first(base["team_map"])
            if "opponent_map" in base.columns:
                base["opponent"] = base["opponent"].replace("", pd.NA).combine_first(base["opponent_map"])

    # If a sportsbook feed omitted team completely, use the current Ourlads
    # roster only when the player resolves to exactly one team.
    role_unique = roles.groupby("player_clean_key")["team"].nunique()
    unique_keys = set(role_unique.loc[role_unique.eq(1)].index)
    role_team = roles.loc[roles["player_clean_key"].isin(unique_keys), ["player_clean_key", "team"]].drop_duplicates("player_clean_key")
    base = base.merge(role_team.rename(columns={"team": "team_roster"}), on="player_clean_key", how="left")
    base["team"] = base["team"].replace("", pd.NA).combine_first(base["team_roster"])

    sched = pf._load_schedule()
    cur = sched.loc[(sched["season"] == int(season)) & (sched["week"] == int(week)), ["team", "opponent"]].drop_duplicates("team")
    base = base.merge(cur.rename(columns={"opponent": "schedule_opponent"}), on="team", how="left")
    base["opponent"] = base["opponent"].replace("", pd.NA).combine_first(base["schedule_opponent"])

    missing = base["team"].isna() | base["opponent"].isna()
    if missing.any():
        path = pf.DATA / "_debug" / "player_form_unresolved_slate_identity.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        base.loc[missing].to_csv(path, index=False)
        raise RuntimeError(f"PlayerForm slate identity unresolved for {int(missing.sum())} rows; see {path}")

    base = base.merge(roles, on=["team", "player_clean_key"], how="left", suffixes=("", "_role"))
    base["player"] = base["player"].replace("", pd.NA).combine_first(base.get("display_name"))
    base["season"] = int(season)
    base["week"] = int(week)
    drop = [c for c in ("team_enriched", "opponent_enriched", "team_map", "opponent_map", "team_roster", "schedule_opponent") if c in base.columns]
    base.drop(columns=drop, inplace=True, errors="ignore")
    return base.drop_duplicates(["team", "player_clean_key"])


def main() -> int:
    pf._load_slate_universe = _enhanced_slate_universe
    return pf.main()


if __name__ == "__main__":
    raise SystemExit(main())
