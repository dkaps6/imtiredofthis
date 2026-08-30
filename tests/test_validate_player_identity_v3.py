import pandas as pd
import pytest

from scripts.validate_player_identity_v3 import validate


def _registry():
    return pd.DataFrame([{
        "player_identity_key": "gsis:00-0012345",
        "player_id": "00-0012345",
        "player": "Veteran Player Jr.",
        "team": "IND",
    }])


def test_identity_validator_allows_explicit_rookie_temporary_identity():
    form = pd.DataFrame([
        {
            "player": "Veteran Player", "team": "IND", "player_id": "00-0012345",
            "player_identity_key": "gsis:00-0012345", "identity_resolution": "team_suffix_alias",
            "identity_confidence": .985,
        },
        {
            "player": "Rookie Player", "team": "IND", "player_id": "",
            "player_identity_key": "temp:IND:rookieplayer", "identity_resolution": "new_or_unmapped_roster",
            "identity_confidence": .50,
        },
    ])
    slate = form.copy()
    out = validate(form, _registry(), slate)
    values = dict(zip(out.metric, out.value))
    assert values["slate_players"] == 2
    assert values["stable_gsis"] == 1
    assert values["temporary_new_or_unmapped"] == 1


def test_identity_validator_rejects_same_stable_person_on_two_current_teams():
    form = pd.DataFrame([
        {
            "player": "Same Player", "team": "IND", "player_id": "00-0012345",
            "player_identity_key": "gsis:00-0012345", "identity_resolution": "team_exact_name",
            "identity_confidence": .995,
        },
        {
            "player": "Same Player", "team": "CHI", "player_id": "00-0012345",
            "player_identity_key": "gsis:00-0012345", "identity_resolution": "unique_name_trade",
            "identity_confidence": .95,
        },
    ])
    with pytest.raises(RuntimeError, match="multiple current teams"):
        validate(form, _registry(), form.copy())


def test_identity_validator_rejects_stable_key_without_player_id():
    form = pd.DataFrame([{
        "player": "Veteran Player", "team": "IND", "player_id": "",
        "player_identity_key": "gsis:00-0012345", "identity_resolution": "team_exact_name",
        "identity_confidence": .995,
    }])
    with pytest.raises(RuntimeError, match="missing player_id"):
        validate(form, _registry(), form.copy())
