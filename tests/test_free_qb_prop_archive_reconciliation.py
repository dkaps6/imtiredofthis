"""Regression test for the M60B free Action Network archive reconciliation.

Root cause (found while auditing the parked historical market-certification
lane): the free `gcampb41/nfl_data-` archive's own `team` column is a
denormalized snapshot that can reflect a player's *later* team (e.g. after an
offseason signing) applied retroactively across an entire historical season,
rather than the team the player was actually on for that specific game. The
archive's GSIS-format `player_id` does not have this staleness. Rejecting a
row merely because the archive's self-reported team disagrees with the
production (season, week, player_id) identity join therefore discarded good,
correctly-identified rows. A stale team label must not be trusted over a
verified GSIS player_id match; it must still be trusted as the only integrity
check when no player_id anchor exists (the exact-name fallback path).
"""
from __future__ import annotations

import pandas as pd

from scripts.backtest.prepare_free_qb_prop_archive import attach_projection_games


def _projection_row(season, week, player_id, team, player_clean_key, game_id):
    return {
        "season": season,
        "week": week,
        "player_id": player_id,
        "team": team,
        "player_clean_key": player_clean_key,
        "game_id": game_id,
    }


def test_gsis_matched_row_survives_stale_archive_team_label():
    projections = pd.DataFrame(
        [_projection_row(2024, 5, "00-0030565", "SEA", "genosmith", "2024_05_LV_DEN")]
    )
    # Archive says LV (a real historical bug: stale/current-team snapshot),
    # but the player was actually on SEA that week per production identity.
    props = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 5,
                "identity_key": "id:00-0030565",
                "book": "draftkings",
                "line": 245.5,
                "over_odds": -110.0,
                "under_odds": -110.0,
                "player": "g.smith",
                "source_name_key": "genosmith",
                "source_player_id": "00-0030565",
                "source_team": "LV",
                "source_event_id": "999",
                "source_line_definition": "archived_latest_per_book",
                "source_dataset": "gcampb41/nfl_data- Action Network-derived archive",
            }
        ]
    )

    matched, stats = attach_projection_games(props, projections)

    assert stats["team_mismatch_rows_flagged"] == 1
    assert stats["team_mismatch_rows_dropped"] == 0
    assert stats["gsis_matched_stale_team_label_rows_kept"] == 1
    assert len(matched) == 1
    assert matched.iloc[0]["game_id"] == "2024_05_LV_DEN"
    assert matched.iloc[0]["team_mismatch"] == True  # noqa: E712 (explicit bool check)


def test_name_fallback_row_is_still_rejected_on_team_mismatch():
    projections = pd.DataFrame(
        [_projection_row(2024, 5, "", "SEA", "j.smith", "2024_05_SEA_ARI")]
    )
    # No GSIS anchor on this row, so an archive-reported team that disagrees
    # with the projection's team must still be rejected: a bare name match
    # with no identity corroboration and no team agreement is not trustworthy.
    props = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 5,
                "identity_key": "name:j.smith",
                "book": "draftkings",
                "line": 60.5,
                "over_odds": -110.0,
                "under_odds": -110.0,
                "player": "j.smith",
                "source_name_key": "j.smith",
                "source_player_id": "",
                "source_team": "LV",
                "source_event_id": "998",
                "source_line_definition": "archived_latest_per_book",
                "source_dataset": "gcampb41/nfl_data- Action Network-derived archive",
            }
        ]
    )

    matched, stats = attach_projection_games(props, projections)

    assert stats["team_mismatch_rows_dropped"] == 1
    assert matched.empty
