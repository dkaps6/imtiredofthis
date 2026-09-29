import pandas as pd

from scripts.operations.grade_market_track_record_gsis_v1 import (
    build_alias_index,
    resolve_gsis,
)
from scripts.utils.canonical_names import canonicalize_player_name_safe


def _actual_row(name: str, gid: str, team: str) -> pd.DataFrame:
    _, key = canonicalize_player_name_safe(name)
    return pd.DataFrame([
        {
            "season": 2026,
            "week": 2,
            "team": team,
            "gsis_id": gid,
            "player": name,
            "player_clean_key": key,
            "position": "WR",
            "receptions": 1.0,
            "rec_yards": 12.0,
            "rush_yards": 0.0,
            "pass_yards": 0.0,
        }
    ])


def _roster_row(name: str, gid: str, team: str) -> pd.DataFrame:
    _, key = canonicalize_player_name_safe(name)
    return pd.DataFrame([
        {
            "season": 2026,
            "week": 2,
            "team": team,
            "gsis_id": gid,
            "player_clean_key": key,
            "status": "ACT",
            "position": "WR",
        }
    ])


def test_verified_joshua_palmer_alias_resolves_to_anchored_gsis():
    actual = _actual_row("Josh Palmer", "00-0036988", "BUF")
    roster = _roster_row("Josh Palmer", "00-0036988", "BUF")
    _, current_key = canonicalize_player_name_safe("Joshua Palmer")

    idx = build_alias_index(actual, roster)
    gid, status = resolve_gsis(current_key, "BUF", idx)

    assert gid == "00-0036988"
    assert status == "RESOLVED_GSIS"


def test_verified_alias_is_not_added_without_source_anchor():
    actual = _actual_row("Other Receiver", "00-0099999", "BUF")
    roster = _roster_row("Other Receiver", "00-0099999", "BUF")
    _, current_key = canonicalize_player_name_safe("Joshua Palmer")

    idx = build_alias_index(actual, roster)
    gid, status = resolve_gsis(current_key, "BUF", idx)

    assert gid == ""
    assert status == "UNRESOLVED_IDENTITY"
