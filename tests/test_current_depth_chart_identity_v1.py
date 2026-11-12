"""Contract tests for the current-season depth-chart identity source.

Why it exists: weekly rosters are the historical identity spine, but they lag an
in-season signing and carry the player under his RELEASING club. Only the depth
chart carries him under the new one with the same GSIS id.

The outage it closed: Josh Johnson (QB, gsis 00-0026300) was cut by Cincinnati
and signed by Arizona on 2026-11-10. His 2021-2025 roster teams are BAL/SF/WAS,
so the team-scoped lookup missed, the resolver fell to the global name, found him
AND a 2021-2022 Seattle RB of the same name, and fail-closed. Every Full Slate
run from that date died at PlayerForm.

These tests use synthetic frames so they neither hit the network nor rot when
rosters change.
"""
from __future__ import annotations

import pandas as pd
import pytest

import scripts.player_identity_roster_history_v1 as mod

FOOTBALL_EVIDENCE = (
    "targets", "carries", "receiving_yards", "rushing_yards", "passing_yards",
    "routes", "touchdowns", "receptions", "attempts", "fantasy_points",
)
IDENTITY_COLUMNS = [
    "season", "week", "team", "position", "player", "player_clean_key",
    "player_id", "player_identity_key", "identity_full_name_key",
    "identity_base_name_key",
]


def _chart(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.fixture
def patch_provider(monkeypatch):
    def _apply(frame: pd.DataFrame | Exception):
        def _loader(seasons=None):
            if isinstance(frame, Exception):
                raise frame
            return frame
        monkeypatch.setattr(mod, "_to_pandas", lambda obj: obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj))
        import sys, types
        fake = types.ModuleType("nflreadpy")
        fake.load_depth_charts = _loader
        monkeypatch.setitem(sys.modules, "nflreadpy", fake)
    return _apply


# ------------------------------------------------- the real-world regression

def test_an_in_season_signing_resolves_to_one_identity(patch_provider):
    """The Josh Johnson case: same name, two people, only one on this team."""
    patch_provider(_chart([
        {"dt": "2026-11-12T15:57:12Z", "team": "ARI", "gsis_id": "00-0026300",
         "player_name": "Josh Johnson", "pos_abb": "QB"},
        {"dt": "2026-11-12T15:57:12Z", "team": "ARI", "gsis_id": "00-0041561",
         "player_name": "Carson Beck", "pos_abb": "QB"},
    ]))
    out, stats = mod._load_current_depth_charts(2026)
    assert stats["available"] is True
    hit = out.loc[out["identity_full_name_key"].eq("joshjohnson") & out["team"].eq("ARI")]
    assert len(hit) == 1
    assert hit.iloc[0]["player_id"] == "00-0026300"
    assert hit.iloc[0]["player_identity_key"] == "gsis:00-0026300"


def test_the_colliding_player_on_another_team_does_not_interfere(patch_provider):
    """Two people with one name on DIFFERENT teams is resolvable, not ambiguous."""
    patch_provider(_chart([
        {"dt": "2026-11-12T00:00:00Z", "team": "ARI", "gsis_id": "00-0026300",
         "player_name": "Josh Johnson", "pos_abb": "QB"},
        {"dt": "2026-11-12T00:00:00Z", "team": "SEA", "gsis_id": "00-0036799",
         "player_name": "Josh Johnson", "pos_abb": "RB"},
    ]))
    out, stats = mod._load_current_depth_charts(2026)
    assert stats["dropped_ambiguous"] == 0
    by_team = out.set_index("team")["player_id"].to_dict()
    assert by_team["ARI"] == "00-0026300"
    assert by_team["SEA"] == "00-0036799"


# --------------------------------------------- never manufacture an ambiguity

def test_two_people_of_one_name_on_the_SAME_team_are_dropped(patch_provider):
    """This source may resolve an ambiguity; it must never create one."""
    patch_provider(_chart([
        {"dt": "2026-11-12T00:00:00Z", "team": "ARI", "gsis_id": "00-0026300",
         "player_name": "Josh Johnson", "pos_abb": "QB"},
        {"dt": "2026-11-12T00:00:00Z", "team": "ARI", "gsis_id": "00-0036799",
         "player_name": "Josh Johnson", "pos_abb": "RB"},
        {"dt": "2026-11-12T00:00:00Z", "team": "ARI", "gsis_id": "00-0041561",
         "player_name": "Carson Beck", "pos_abb": "QB"},
    ]))
    out, stats = mod._load_current_depth_charts(2026)
    assert stats["dropped_ambiguous"] == 2
    assert out["identity_full_name_key"].tolist() == ["carsonbeck"]


# ------------------------------------------------- only the current placement

def test_only_the_latest_snapshot_per_team_is_kept(patch_provider):
    """The provider accumulates every chart published in the season. Stale
    placements would reintroduce the ambiguity this source exists to remove."""
    patch_provider(_chart([
        {"dt": "2026-03-22T06:38:42Z", "team": "CIN", "gsis_id": "00-0026300",
         "player_name": "Josh Johnson", "pos_abb": "QB"},
        {"dt": "2026-11-12T15:57:12Z", "team": "CIN", "gsis_id": "00-0041999",
         "player_name": "Someone Else", "pos_abb": "QB"},
    ]))
    out, _ = mod._load_current_depth_charts(2026)
    assert out["player_id"].tolist() == ["00-0041999"]


def test_latest_snapshot_is_computed_per_team_not_globally(patch_provider):
    """One team publishing later must not erase another team's current chart."""
    patch_provider(_chart([
        {"dt": "2026-11-12T15:00:00Z", "team": "ARI", "gsis_id": "00-0026300",
         "player_name": "Josh Johnson", "pos_abb": "QB"},
        {"dt": "2026-11-01T09:00:00Z", "team": "SEA", "gsis_id": "00-0036799",
         "player_name": "Zach Charbonnet", "pos_abb": "RB"},
    ]))
    out, _ = mod._load_current_depth_charts(2026)
    assert set(out["team"]) == {"ARI", "SEA"}


# ---------------------------------------------------- identity-only contract

def test_no_football_evidence_column_survives(patch_provider):
    """The module's whole premise: identity metadata, never model features."""
    patch_provider(_chart([
        {"dt": "2026-11-12T00:00:00Z", "team": "ARI", "gsis_id": "00-0026300",
         "player_name": "Josh Johnson", "pos_abb": "QB",
         "targets": 99, "rushing_yards": 1234, "fantasy_points": 50.0},
    ]))
    out, _ = mod._load_current_depth_charts(2026)
    assert list(out.columns) == IDENTITY_COLUMNS
    for col in FOOTBALL_EVIDENCE:
        assert col not in out.columns


def test_rows_are_stamped_week_zero_so_they_cannot_read_as_game_evidence(patch_provider):
    patch_provider(_chart([
        {"dt": "2026-11-12T00:00:00Z", "team": "ARI", "gsis_id": "00-0026300",
         "player_name": "Josh Johnson", "pos_abb": "QB"},
    ]))
    out, _ = mod._load_current_depth_charts(2026)
    assert set(out["week"]) == {0}
    assert set(out["season"]) == {2026}


def test_non_skill_positions_are_excluded(patch_provider):
    patch_provider(_chart([
        {"dt": "2026-11-12T00:00:00Z", "team": "ARI", "gsis_id": "00-0038540",
         "player_name": "Paris Johnson Jr.", "pos_abb": "OL"},
        {"dt": "2026-11-12T00:00:00Z", "team": "ARI", "gsis_id": "00-0040751",
         "player_name": "Will Johnson", "pos_abb": "DB"},
        {"dt": "2026-11-12T00:00:00Z", "team": "ARI", "gsis_id": "00-0026300",
         "player_name": "Josh Johnson", "pos_abb": "QB"},
    ]))
    out, _ = mod._load_current_depth_charts(2026)
    assert out["player_id"].tolist() == ["00-0026300"]


def test_rows_without_a_stable_gsis_id_are_excluded(patch_provider):
    """A blank id must never manufacture a person key."""
    patch_provider(_chart([
        {"dt": "2026-11-12T00:00:00Z", "team": "ARI", "gsis_id": "",
         "player_name": "Tryout Guy", "pos_abb": "WR"},
        {"dt": "2026-11-12T00:00:00Z", "team": "ARI", "gsis_id": "00-0026300",
         "player_name": "Josh Johnson", "pos_abb": "QB"},
    ]))
    out, _ = mod._load_current_depth_charts(2026)
    assert out["player_id"].tolist() == ["00-0026300"]


# ------------------------------------------------------ enrichment, not a gate

def test_a_provider_failure_is_visible_and_not_fatal(patch_provider):
    """Without this source the registry resolves exactly as it did before, so an
    outage degrades to the old behaviour rather than blocking the slate."""
    patch_provider(RuntimeError("provider down"))
    out, stats = mod._load_current_depth_charts(2026)
    assert out.empty
    assert stats["available"] is False
    assert "provider down" in stats["error"]


def test_zero_rows_is_visible_and_not_fatal(patch_provider):
    patch_provider(_chart([]))
    out, stats = mod._load_current_depth_charts(2026)
    assert out.empty and stats["available"] is False and "error" in stats


def test_unparseable_timestamps_are_reported_rather_than_guessed(patch_provider):
    patch_provider(_chart([
        {"dt": "not-a-date", "team": "ARI", "gsis_id": "00-0026300",
         "player_name": "Josh Johnson", "pos_abb": "QB"},
    ]))
    out, stats = mod._load_current_depth_charts(2026)
    assert out.empty and stats["available"] is False


def test_a_chart_without_a_dt_column_still_contributes(patch_provider):
    """Not every provider shape carries dt; identity is still usable."""
    patch_provider(_chart([
        {"team": "ARI", "gsis_id": "00-0026300", "player_name": "Josh Johnson",
         "pos_abb": "QB"},
    ]))
    out, stats = mod._load_current_depth_charts(2026)
    assert stats["available"] is True and len(out) == 1
