import pandas as pd
import pytest

from scripts import player_form_v2 as pf
from scripts.utils.player_identity_v3 import (
    attach_historical_identity,
    build_identity_registry,
    player_name_key,
    resolve_slate_identities,
)


def _logs(rows):
    df = pd.DataFrame(rows)
    if "position" not in df.columns:
        df["position"] = "WR"
    return attach_historical_identity(df)


def test_suffix_stripped_ourlads_name_resolves_to_stable_gsis_identity():
    logs = _logs([
        {
            "season": 2025,
            "week": 18,
            "player_id": "00-0039999",
            "player": "Marvin Harrison Jr.",
            "team": "ARI",
            "position": "WR",
        }
    ])
    registry = build_identity_registry(logs)
    slate = pd.DataFrame([{"player": "Marvin Harrison", "team": "ARI"}])
    out = resolve_slate_identities(slate, registry)
    row = out.iloc[0]
    assert row.player_identity_key == "gsis:00-0039999"
    assert row.player_id == "00-0039999"
    assert row.identity_resolution == "team_suffix_alias"


def test_trade_carries_prior_identity_to_new_team_when_name_is_unique():
    logs = _logs([
        {
            "season": 2025,
            "week": 17,
            "player_id": "00-0031111",
            "player": "Example Receiver",
            "team": "BUF",
            "position": "WR",
        }
    ])
    registry = build_identity_registry(logs)
    slate = pd.DataFrame([{"player": "Example Receiver", "team": "KC"}])
    out = resolve_slate_identities(slate, registry)
    row = out.iloc[0]
    assert row.player_identity_key == "gsis:00-0031111"
    assert row.identity_resolution == "unique_name_trade"


def test_true_rookie_gets_temporary_identity_and_no_inherited_id():
    registry = build_identity_registry(pd.DataFrame())
    slate = pd.DataFrame([{"player": "Rookie Player Jr.", "team": "IND"}])
    out = resolve_slate_identities(slate, registry)
    row = out.iloc[0]
    assert row.player_identity_key == "temp:IND:rookieplayer"
    assert row.player_id == ""
    assert row.identity_resolution == "new_or_unmapped_roster"
    assert row.identity_confidence == 0.50


def test_ambiguous_same_name_different_people_fails_closed():
    logs = _logs([
        {
            "season": 2025,
            "week": 10,
            "player_id": "00-0010001",
            "player": "Chris Smith",
            "team": "BUF",
            "position": "WR",
        },
        {
            "season": 2025,
            "week": 11,
            "player_id": "00-0010002",
            "player": "Chris Smith",
            "team": "KC",
            "position": "WR",
        },
    ])
    registry = build_identity_registry(logs)
    # A third-team current row has no team-specific disambiguation.  Global name
    # matching sees two real player IDs and must not guess.
    slate = pd.DataFrame([{"player": "Chris Smith", "team": "IND"}])
    with pytest.raises(RuntimeError, match="Ambiguous player identity"):
        resolve_slate_identities(slate, registry)


def test_team_context_can_disambiguate_same_name_people():
    logs = _logs([
        {
            "season": 2025,
            "week": 10,
            "player_id": "00-0010001",
            "player": "Chris Smith",
            "team": "BUF",
            "position": "WR",
        },
        {
            "season": 2025,
            "week": 11,
            "player_id": "00-0010002",
            "player": "Chris Smith",
            "team": "KC",
            "position": "WR",
        },
    ])
    registry = build_identity_registry(logs)
    slate = pd.DataFrame([{"player": "Chris Smith", "team": "KC"}])
    out = resolve_slate_identities(slate, registry)
    assert out.iloc[0].player_identity_key == "gsis:00-0010002"
    assert out.iloc[0].identity_resolution == "team_exact_name"


def test_playerform_season_totals_group_name_variants_by_stable_id():
    base = pd.DataFrame([
        {
            "season": 2025,
            "week": 1,
            "player_id": "00-0099999",
            "player": "D.J. Example",
            "player_clean_key": "djexample",
            "team": "DAL",
            "position": "WR",
            "targets": 5,
            "receptions": 3,
            "rec_yards": 40,
            "rushes": 0,
            "rush_yards": 0,
            "pass_att": 0,
            "pass_yards": 0,
            "routes": 20,
            "team_targets": 30,
            "team_rushes": 25,
            "team_dropbacks": 35,
            "team_routes": 100,
        },
        {
            "season": 2025,
            "week": 2,
            "player_id": "00-0099999",
            "player": "DJ Example Jr.",
            "player_clean_key": "djexamplejr",
            "team": "DAL",
            "position": "WR",
            "targets": 7,
            "receptions": 5,
            "rec_yards": 70,
            "rushes": 0,
            "rush_yards": 0,
            "pass_att": 0,
            "pass_yards": 0,
            "routes": 25,
            "team_targets": 32,
            "team_rushes": 23,
            "team_dropbacks": 38,
            "team_routes": 105,
        },
    ])
    logs = attach_historical_identity(base)
    totals = pf._season_totals(logs)
    assert len(totals) == 1
    row = totals.iloc[0]
    assert row.player_identity_key == "gsis:00-0099999"
    assert row.games == 2
    assert row.targets == 12
    assert row.rec_yards == 110


def test_suffix_key_normalization_is_explicit_not_universal():
    assert player_name_key("Marvin Harrison Jr.") == "marvinharrisonjr"
    assert player_name_key("Marvin Harrison Jr.", strip_suffix=True) == "marvinharrison"
    assert player_name_key("Marvin Harrison") == "marvinharrison"
