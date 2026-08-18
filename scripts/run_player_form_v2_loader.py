#!/usr/bin/env python3
"""Run PlayerForm v2 with maintained loaders and strict game identity handling.

This production entry point owns provider compatibility that should not leak into
the PlayerForm feature logic. In particular, weekly player-stat feeds may already
contain a ``game_id`` while the schedule layer also supplies one. A normal pandas
merge would silently suffix those columns to ``game_id_x``/``game_id_y`` and break
the downstream game-log contract. We resolve that collision explicitly here.
"""
from __future__ import annotations

import pandas as pd

import scripts.run_player_form_v2 as runner
from scripts.player_stats_loader_v2 import load_weekly_player_stats


def _clean_id(series: pd.Series) -> pd.Series:
    """Normalize identifier-like values while preserving missingness."""
    out = series.astype("string").str.strip()
    return out.mask(out.isin(["", "<NA>", "nan", "None"]))


def attach_schedule_with_game_identity(logs: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Attach opponent/game identity with an explicit many-to-one contract.

    Invariants:
    - schedule must have exactly one row per season/week/team;
    - the historical PlayerForm scope is exactly the set of season/week pairs
      present in the authoritative regular-season schedule;
    - source rows outside that schedule scope (for example postseason weeks
      returned by nflreadpy) are excluded deliberately and reported;
    - any unresolved row inside a schedule-covered week is still a hard failure;
    - the output always contains one canonical ``game_id`` column;
    - schedule game identity wins when available because it is shared by both
      teams in a game;
    - a source-provided player-stat game id is preserved as ``source_game_id``;
    - if neither source supplies an id, a deterministic symmetric id is built so
      both teams receive the same game key.
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

    # The shared schedule utility intentionally returns the regular season.
    # nflreadpy weekly player stats may include postseason weeks as well. Rather
    # than hard-coding 'week <= 18', define PlayerForm's historical scope from
    # the authoritative schedule itself. This keeps the rule correct if the NFL
    # calendar changes and prevents playoff teams from receiving extra prior
    # weight merely because they played additional games.
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

    # Preserve provider game identity before the merge so pandas cannot create
    # game_id_x/game_id_y and silently destroy the canonical schema.
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

    # Deterministic fallback is symmetric in the two teams. This matters: a
    # team/opponent ordered fallback would create two IDs for the same game.
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
    # PlayerForm keeps the build/blend semantics; this adapter supplies the
    # maintained weekly-stat provider and collision-safe game identity contract.
    runner.pf._load_weekly = load_weekly_player_stats

    def _attach(logs: pd.DataFrame) -> pd.DataFrame:
        return attach_schedule_with_game_identity(logs, runner.pf._load_schedule())

    runner.pf._attach_schedule = _attach
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
