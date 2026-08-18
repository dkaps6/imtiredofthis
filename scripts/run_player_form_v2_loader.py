#!/usr/bin/env python3
"""Run PlayerForm v2 with maintained loaders and strict runtime contracts."""
from __future__ import annotations

import os

import pandas as pd

import scripts.run_player_form_v2 as runner
from scripts.player_stats_loader_v2 import load_weekly_player_stats
from scripts.slate_universe_v2 import build_slate_universe


def _clean_id(series: pd.Series) -> pd.Series:
    """Normalize identifier-like values while preserving missingness."""
    out = series.astype("string").str.strip()
    return out.mask(out.isin(["", "<NA>", "nan", "None"]))


def _live_odds_enabled() -> bool:
    return os.getenv("FETCH_LIVE_ODDS", "false").strip().lower() in {"1", "true", "yes", "on"}


def attach_schedule_with_game_identity(logs: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Attach opponent/game identity with an explicit many-to-one contract.

    The authoritative historical scope is the set of season/week pairs present
    in the regular-season schedule. Provider postseason rows are excluded
    deliberately, while unresolved rows inside schedule scope remain fatal.
    """
    keys = ["season", "week", "team"]
    missing_logs = [c for c in keys if c not in logs.columns]
    missing_sched = [c for c in [*keys, "opponent"] if c not in schedule.columns]
    if missing_logs:
        raise RuntimeError(f"Player game logs missing schedule join keys: {missing_logs}")
    if missing_sched:
        raise RuntimeError(f"Schedule missing PlayerForm identity columns: {missing_sched}")

    left = logs.copy()
    right = schedule.copy()

    for frame in (left, right):
        frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
        frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")
        frame["team"] = frame["team"].astype("string").str.upper().str.strip()

    duplicate_schedule = right.duplicated(keys, keep=False)
    if duplicate_schedule.any():
        sample = right.loc[duplicate_schedule, [*keys, "opponent"]].head(20).to_dict("records")
        raise RuntimeError(
            "Schedule is not unique at season/week/team grain; "
            f"duplicate_rows={int(duplicate_schedule.sum())} sample={sample}"
        )

    schedule_season_weeks = pd.MultiIndex.from_frame(
        right[["season", "week"]].dropna().drop_duplicates()
    )
    player_season_weeks = pd.MultiIndex.from_frame(left[["season", "week"]])
    in_schedule_scope = player_season_weeks.isin(schedule_season_weeks)
    if (~in_schedule_scope).any():
        excluded = left.loc[~in_schedule_scope]
        weeks = sorted(
            {
                (int(s), int(w))
                for s, w in excluded[["season", "week"]].dropna().itertuples(index=False, name=None)
            }
        )
        print(
            "[player_form_v2] excluding out-of-scope weekly stat rows not present "
            f"in regular-season schedule: rows={len(excluded)} season_weeks={weeks}"
        )
        left = left.loc[in_schedule_scope].copy()

    if left.empty:
        raise RuntimeError(
            "No player weekly-stat rows remain after applying the authoritative regular-season schedule scope"
        )
    left["season_type"] = "REG"

    if "game_id" in left.columns:
        left["source_game_id"] = _clean_id(left["game_id"])
        left = left.drop(columns=["game_id"])
    elif "source_game_id" in left.columns:
        left["source_game_id"] = _clean_id(left["source_game_id"])
    else:
        left["source_game_id"] = pd.Series(pd.NA, index=left.index, dtype="string")

    if "game_id" in right.columns:
        right["schedule_game_id"] = _clean_id(right["game_id"])
        right = right.drop(columns=["game_id"])
    else:
        right["schedule_game_id"] = pd.Series(pd.NA, index=right.index, dtype="string")

    keep = [*keys, "opponent", "schedule_game_id"]
    out = left.merge(right[keep], on=keys, how="left", validate="many_to_one")

    unresolved_opp = out["opponent"].isna() | out["opponent"].astype("string").str.strip().eq("")
    if unresolved_opp.any():
        sample = out.loc[
            unresolved_opp,
            [*keys, "player"] if "player" in out.columns else keys,
        ].head(20).to_dict("records")
        raise RuntimeError(
            "Historical player rows failed schedule/opponent attachment inside the "
            "authoritative regular-season scope; "
            f"rows={int(unresolved_opp.sum())} sample={sample}"
        )

    out["opponent"] = out["opponent"].astype("string").str.upper().str.strip()
    team_a = out["team"].astype("string")
    team_b = out["opponent"].astype("string")
    first = team_a.where(team_a <= team_b, team_b)
    second = team_b.where(team_a <= team_b, team_a)
    fallback = (
        out["season"].astype("Int64").astype("string")
        + "_" + out["week"].astype("Int64").astype("string").str.zfill(2)
        + "_" + first
        + "_" + second
    )

    out["game_id"] = _clean_id(out["schedule_game_id"]).combine_first(
        _clean_id(out["source_game_id"])
    )
    out["game_id"] = out["game_id"].combine_first(fallback)
    if out["game_id"].isna().any() or out["game_id"].astype("string").str.strip().eq("").any():
        raise RuntimeError("PlayerForm game identity resolution produced missing game_id values")
    return out


def main() -> int:
    live_odds = _live_odds_enabled()
    print(f"[player_form_v2] FETCH_LIVE_ODDS={'true' if live_odds else 'false'}")

    # Maintained weekly-stat provider.
    runner.pf._load_weekly = load_weekly_player_stats

    # Collision-safe historical schedule/game identity attachment.
    def _attach(logs: pd.DataFrame) -> pd.DataFrame:
        return attach_schedule_with_game_identity(logs, runner.pf._load_schedule())

    runner.pf._attach_schedule = _attach

    # Make offseason/no-credit mode a real production mode. The slate builder
    # ignores stale props/odds placeholders entirely when live odds are disabled,
    # and derives the player universe from Ourlads + the authoritative schedule.
    def _slate(season: int, week: int) -> pd.DataFrame:
        return build_slate_universe(
            runner.pf,
            runner._enhanced_load_schedule,
            season,
            week,
            live_odds_enabled=live_odds,
        )

    runner._enhanced_slate_universe = _slate
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
