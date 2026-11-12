"""Contract tests for the depth-chart union in the PlayerForm slate universe.

Why it exists: in 2026 week 10 the Ourlads-derived universe held NYG RB1 =
Devin Singletary while nflverse had Cam Skattebo -- who practised in FULL,
carried 418 rushing yards and started that night -- and CLE QB1 = Shedeur
Sanders rather than Deshaun Watson. Both starters were absent from the universe
entirely, so materialize_live_props fail-closed on ten priced rows naming five
unknown players and no board could be produced.

The union must do three things at once, and the tests below pin all three
because the first draft of it got two of them wrong:
  * restore a mis-ranked or missing player,
  * never add the same person twice under a different spelling,
  * never re-admit a player the availability certification withheld.
"""
from __future__ import annotations

import pandas as pd
import pytest

import scripts.slate_universe_v2 as su


def _roles(rows):
    return pd.DataFrame(rows, columns=["team", "player", "player_clean_key", "position", "role"])


@pytest.fixture(autouse=True)
def _enable_union(monkeypatch):
    """The union is an explicit production opt-in, OFF by default, so that it can
    never silently mask the certified-team-scope gate. Every test here asks for
    it deliberately."""
    monkeypatch.setenv(su.DEPTH_CHART_UNION_ENV, "1")


def test_the_union_is_off_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv(su.DEPTH_CHART_UNION_ENV, raising=False)
    _patch_depth(monkeypatch, [("NYG", "Cam Skattebo", "camskattebo", "RB", 1)])
    base = _roles([("NYG", "Devin Singletary", "devinsingletary", "RB", "RB1")])
    out, stats = su.augment_roles_with_depth_chart(base, 2026)
    assert stats["enabled"] is False
    assert out.equals(base)


@pytest.fixture
def no_external(monkeypatch):
    """Neutralise the two providers so each test states its own world."""
    monkeypatch.setattr(su, "_season_usage_keys", lambda season: set())
    monkeypatch.setattr(su, "_unavailable_designation_keys", lambda: set())


def _patch_depth(monkeypatch, rows):
    monkeypatch.setattr(su, "_depth_chart_universe", lambda season: pd.DataFrame(
        rows, columns=["team", "player", "player_clean_key", "position", "rank"]))


# --------------------------------------------------------- the real regression

def test_a_missing_starter_is_added_as_rank_one(monkeypatch, no_external):
    """Skattebo: absent from Ourlads, RB1 on the depth chart."""
    _patch_depth(monkeypatch, [
        ("NYG", "Cam Skattebo", "camskattebo", "RB", 1),
        ("NYG", "Devin Singletary", "devinsingletary", "RB", 2),
    ])
    out, stats = su.augment_roles_with_depth_chart(
        _roles([("NYG", "Devin Singletary", "devinsingletary", "RB", "RB1")]), 2026)
    assert stats["added"] == 1
    roles = dict(zip(out.player, out.role))
    assert roles["Cam Skattebo"] == "RB1"
    assert roles["Devin Singletary"] == "RB2"


def test_a_demoted_starter_is_restored_not_merely_present(monkeypatch, no_external):
    """Presence alone is not enough: the football error is the RANK."""
    _patch_depth(monkeypatch, [
        ("CLE", "Deshaun Watson", "deshaunwatson", "QB", 1),
        ("CLE", "Shedeur Sanders", "shedeursanders", "QB", 2),
    ])
    out, _ = su.augment_roles_with_depth_chart(
        _roles([("CLE", "Shedeur Sanders", "shedeursanders", "QB", "QB1")]), 2026)
    assert dict(zip(out.player, out.role))["Deshaun Watson"] == "QB1"


def test_nobody_from_the_roles_artifact_is_ever_dropped(monkeypatch, no_external):
    _patch_depth(monkeypatch, [("GB", "MarShawn Lloyd", "marshawnlloyd", "RB", 2)])
    base = _roles([
        ("GB", "Josh Jacobs", "joshjacobs", "RB", "RB1"),
        ("GB", "Some Guy", "someguy", "RB", "RB2"),
    ])
    out, _ = su.augment_roles_with_depth_chart(base, 2026)
    assert set(base.player).issubset(set(out.player))


# ------------------------------------------------- never duplicate a person

@pytest.mark.parametrize("ourlads,depth", [
    ("Pierre Strong", "Pierre Strong Jr."),
    ("Pierre Strong Jr.", "Pierre Strong"),
    ("Michael Sturdivant", "J. Michael Sturdivant"),
])
def test_suffix_and_initial_variants_are_not_added_twice(monkeypatch, no_external, ourlads, depth):
    """The two sources disagree on suffixes and initials. A strict key compare
    would add the same person twice and split his opportunity."""
    _patch_depth(monkeypatch, [("GB", depth, su.player_name_key(depth), "RB", 2)])
    out, stats = su.augment_roles_with_depth_chart(
        _roles([("GB", ourlads, su.player_name_key(ourlads), "RB", "RB1")]), 2026)
    assert stats["added"] == 0
    assert len(out) == 1


def test_a_genuinely_different_player_of_similar_name_is_still_added(monkeypatch, no_external):
    _patch_depth(monkeypatch, [("GB", "Kaleb Johnson", "kalebjohnson", "RB", 3)])
    out, stats = su.augment_roles_with_depth_chart(
        _roles([("GB", "Josh Johnson", "joshjohnson", "QB", "QB2")]), 2026)
    assert stats["added"] == 1
    assert {"Josh Johnson", "Kaleb Johnson"} <= set(out.player)


# ------------------------------------- never re-admit an unavailable player

def test_a_player_designated_out_is_never_added(monkeypatch):
    """The roles artifact is already availability-certified; the union must not
    bypass that gate. Week 10 would otherwise have re-admitted TreVeyon
    Henderson (NE, OUT) and Colson Yankoff (WAS, OUT)."""
    monkeypatch.setattr(su, "_season_usage_keys", lambda season: set())
    monkeypatch.setattr(su, "_unavailable_designation_keys",
                        lambda: {su.player_name_key("TreVeyon Henderson", strip_suffix=True)})
    _patch_depth(monkeypatch, [
        ("NE", "TreVeyon Henderson", "treveyonhenderson", "RB", 2),
        ("NE", "Rhamondre Stevenson", "rhamondrestevenson", "RB", 1),
    ])
    out, stats = su.augment_roles_with_depth_chart(
        _roles([("NE", "Rhamondre Stevenson", "rhamondrestevenson", "RB", "RB1")]), 2026)
    assert stats["dropped_unavailable"] == 1
    assert "TreVeyon Henderson" not in set(out.player)


def test_merely_not_practising_does_not_exclude(monkeypatch, no_external):
    """No designation until Friday. Treating midweek practice as unavailability
    would have dropped Puka Nacua and Sam Darnold in week 10."""
    _patch_depth(monkeypatch, [("LAR", "Puka Nacua", "pukanacua", "WR", 1)])
    out, _ = su.augment_roles_with_depth_chart(
        _roles([("LAR", "Other Guy", "otherguy", "WR", "WR1")]), 2026)
    assert "Puka Nacua" in set(out.player)


# ------------------------------------------------ require football evidence

def test_a_deep_reserve_with_no_usage_is_not_added(monkeypatch):
    """A WR8 with no stat row still draws a shrunk baseline and would take a
    share of a finite target pool he never sees snaps for."""
    monkeypatch.setattr(su, "_season_usage_keys", lambda season: set())
    monkeypatch.setattr(su, "_unavailable_designation_keys", lambda: set())
    _patch_depth(monkeypatch, [("GB", "Practice Squad Guy", "practicesquadguy", "WR", 8)])
    out, stats = su.augment_roles_with_depth_chart(
        _roles([("GB", "Real WR", "realwr", "WR", "WR1")]), 2026)
    assert stats["dropped_no_evidence"] == 1
    assert "Practice Squad Guy" not in set(out.player)


def test_usage_overrides_a_deep_depth_rank(monkeypatch):
    monkeypatch.setattr(su, "_season_usage_keys",
                        lambda season: {su.player_name_key("Deep Contributor", strip_suffix=True)})
    monkeypatch.setattr(su, "_unavailable_designation_keys", lambda: set())
    _patch_depth(monkeypatch, [("GB", "Deep Contributor", "deepcontributor", "WR", 7)])
    out, stats = su.augment_roles_with_depth_chart(
        _roles([("GB", "Real WR", "realwr", "WR", "WR1")]), 2026)
    assert stats["added"] == 1 and "Deep Contributor" in set(out.player)


def test_top_four_is_added_without_usage(monkeypatch):
    monkeypatch.setattr(su, "_season_usage_keys", lambda season: set())
    monkeypatch.setattr(su, "_unavailable_designation_keys", lambda: set())
    _patch_depth(monkeypatch, [("GB", "Fresh Signing", "freshsigning", "RB", 3)])
    out, _ = su.augment_roles_with_depth_chart(
        _roles([("GB", "Starter", "starter", "RB", "RB1")]), 2026)
    assert "Fresh Signing" in set(out.player)


# --------------------------------------------------- enrichment, not a gate

def test_a_provider_failure_leaves_the_universe_untouched(monkeypatch, no_external):
    monkeypatch.setattr(su, "_depth_chart_universe", lambda season: pd.DataFrame())
    base = _roles([("GB", "Josh Jacobs", "joshjacobs", "RB", "RB1")])
    out, stats = su.augment_roles_with_depth_chart(base, 2026)
    assert stats["available"] is False
    assert out.equals(base)


def test_an_empty_roles_artifact_is_returned_unchanged(monkeypatch, no_external):
    _patch_depth(monkeypatch, [("GB", "X", "x", "RB", 1)])
    empty = _roles([])
    out, stats = su.augment_roles_with_depth_chart(empty, 2026)
    assert out.empty and stats["available"] is False


def test_a_position_the_depth_chart_does_not_cover_keeps_its_roles(monkeypatch, no_external):
    """Re-ranking must be scoped to covered team/positions only."""
    _patch_depth(monkeypatch, [("GB", "Josh Jacobs", "joshjacobs", "RB", 1)])
    out, _ = su.augment_roles_with_depth_chart(_roles([
        ("GB", "Josh Jacobs", "joshjacobs", "RB", "RB1"),
        ("GB", "Tucker Kraft", "tuckerkraft", "TE", "TE1"),
    ]), 2026)
    assert dict(zip(out.player, out.role))["Tucker Kraft"] == "TE1"
