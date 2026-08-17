#!/usr/bin/env python3
"""Run PlayerForm v2 with robust identity, historical schedules, and model roles."""
from __future__ import annotations

import pandas as pd

import scripts.player_form_v2 as pf
from scripts._opponent_map import canon_team
from scripts.build._schedule_utils import get_nfl_schedule
from scripts.runtime_context import resolve_prior_season, resolve_season

_ORIGINAL_LOAD_SCHEDULE = pf._load_schedule
_ORIGINAL_BLEND = pf._blend


def _long_schedule_for_season(season: int) -> pd.DataFrame:
    """Return season/week/team/opponent/game_id for any requested season."""
    # Prefer the authoritative active-season map because it also carries any
    # workflow-specific corrections. Historical seasons are fetched from the
    # shared nflverse schedule helper.
    active = resolve_season()
    if int(season) == int(active):
        current = _ORIGINAL_LOAD_SCHEDULE().copy()
        scoped = current.loc[pd.to_numeric(current["season"], errors="coerce").eq(int(season))].copy()
        if not scoped.empty:
            return scoped

    schedule = get_nfl_schedule(int(season))
    if schedule.empty:
        raise RuntimeError(f"No schedule rows available for season={season}")
    home = schedule.assign(team=schedule["home"].map(canon_team), opponent=schedule["away"].map(canon_team))
    away = schedule.assign(team=schedule["away"].map(canon_team), opponent=schedule["home"].map(canon_team))
    out = pd.concat([home, away], ignore_index=True)
    if "game_id" not in out.columns:
        out["game_id"] = (
            out["season"].astype(int).astype(str)
            + "_" + out["week"].astype(int).astype(str).str.zfill(2)
            + "_" + out["home"].map(canon_team).astype(str)
            + "_" + out["away"].map(canon_team).astype(str)
        )
    return out[["season", "week", "team", "opponent", "game_id"]].drop_duplicates()


def _enhanced_load_schedule() -> pd.DataFrame:
    seasons = [resolve_prior_season(), resolve_season()]
    frames = [_long_schedule_for_season(year) for year in seasons]
    return pd.concat(frames, ignore_index=True).drop_duplicates(["season", "week", "team"], keep="last")


def _position_family(value) -> str:
    pos = "" if pd.isna(value) else str(value).upper().strip()
    if pos in {"LWR", "RWR", "SWR", "WR", "WIDE RECEIVER"} or pos.startswith("WR"):
        return "WR"
    if pos in {"HB", "TB", "RB"} or pos.startswith("RB"):
        return "RB"
    if pos in {"TE", "Y"} or pos.startswith("TE"):
        return "TE"
    if pos == "QB" or pos.startswith("QB"):
        return "QB"
    if pos == "FB" or pos.startswith("FB"):
        return "FB"
    return pos or "UNK"


def _assign_model_roles(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive model hierarchy from usage; preserve Ourlads depth alignment."""
    out = frame.copy()
    out["depth_role"] = out.get("role", pd.Series(pd.NA, index=out.index, dtype="string"))
    out["alignment_position"] = out.get("position", pd.Series(pd.NA, index=out.index, dtype="string"))
    out["position_group"] = out["alignment_position"].map(_position_family)
    out["model_role"] = pd.Series(pd.NA, index=out.index, dtype="string")

    score_by_family = {
        "WR": ("tgt_share", "WR"),
        "TE": ("tgt_share", "TE"),
        "RB": ("rush_share", "RB"),
        "QB": ("ypa", "QB"),
        "FB": ("rush_share", "FB"),
    }
    for team, team_df in out.groupby("team", dropna=False):
        for family, (score_col, prefix) in score_by_family.items():
            idx = team_df.index[team_df["position_group"].eq(family)]
            if not len(idx):
                continue
            scores = pd.to_numeric(out.loc[idx, score_col], errors="coerce") if score_col in out.columns else pd.Series(index=idx, dtype=float)
            # Unknown usage sorts below known usage, but never changes player identity.
            order = scores.fillna(-1).sort_values(ascending=False, kind="mergesort").index.tolist()
            for rank, row_idx in enumerate(order, start=1):
                out.at[row_idx, "model_role"] = f"{prefix}{rank}"

    # For a player whose position is missing, retain depth role as a last-resort
    # label rather than inventing a usage family.
    missing = out["model_role"].isna()
    out.loc[missing, "model_role"] = out.loc[missing, "depth_role"].astype("string")
    out["role"] = out["model_role"]
    # Consumers should reason about WR/RB/TE/QB family, not LWR/RWR alignment.
    out["position"] = out["position_group"]
    return out


def _enhanced_blend(prior: pd.DataFrame, current: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    return _assign_model_roles(_ORIGINAL_BLEND(prior, current, universe))


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
            join = [c for c in ("event_id", "player_clean_key") if c in base.columns and c in e.columns] or ["player_clean_key"]
            cols = join + [c for c in ("team_enriched", "opponent_enriched") if c in e.columns]
            if len(cols) > len(join):
                base = base.merge(e[cols].drop_duplicates(join, keep="last"), on=join, how="left")
                if "team_enriched" in base.columns:
                    base["team"] = base["team"].replace("", pd.NA).combine_first(base["team_enriched"])
                if "opponent_enriched" in base.columns:
                    base["opponent"] = base["opponent"].replace("", pd.NA).combine_first(base["opponent_enriched"])

    if pf.OPPONENT_MAP.exists() and pf.OPPONENT_MAP.stat().st_size > 0:
        om = pd.read_csv(pf.OPPONENT_MAP)
        om.columns = [str(c).lower() for c in om.columns]
        if "player_clean_key" not in om.columns and "player" in om.columns:
            om["player_clean_key"] = om["player"].map(pf._canon_name).map(lambda t: t[1])
        for c in ("season", "week"):
            if c in om.columns:
                om[c] = pd.to_numeric(om[c], errors="coerce").astype("Int64")
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

    role_unique = roles.groupby("player_clean_key")["team"].nunique()
    unique_keys = set(role_unique.loc[role_unique.eq(1)].index)
    role_team = roles.loc[roles["player_clean_key"].isin(unique_keys), ["player_clean_key", "team"]].drop_duplicates("player_clean_key")
    base = base.merge(role_team.rename(columns={"team": "team_roster"}), on="player_clean_key", how="left")
    base["team"] = base["team"].replace("", pd.NA).combine_first(base["team_roster"])

    sched = _enhanced_load_schedule()
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
    pf._load_schedule = _enhanced_load_schedule
    pf._load_slate_universe = _enhanced_slate_universe
    pf._blend = _enhanced_blend
    return pf.main()


if __name__ == "__main__":
    raise SystemExit(main())
