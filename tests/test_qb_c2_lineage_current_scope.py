from pathlib import Path

import pandas as pd
import pytest

from scripts.stamp_qb_c2_pricing_lineage_v1 import (
    _validate_current_c2_scope,
    _validate_priced_c2_subset,
)


ALL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]
PLAYED = {"LAR", "NE", "SEA", "SF"}
LIVE = [team for team in ALL_TEAMS if team not in PLAYED]


def _c2(teams):
    return pd.DataFrame(
        [{"team": team, "player": f"{team} QB", "player_clean_key": f"{team.lower()}qb"} for team in teams]
    )


def _active_roles(tmp_path: Path, teams) -> Path:
    path = tmp_path / "active_roles.csv"
    pd.DataFrame(
        [{"team": team, "player": f"{team} QB", "position": "QB"} for team in teams]
    ).to_csv(path, index=False)
    return path


def _lookup(teams, selected_teams=()):
    selected = set(selected_teams)
    return {
        (team, f"{team.lower()}qb"): pd.Series({"selector_c2_selected": int(team in selected)})
        for team in teams
    }


def test_c2_scope_accepts_explicit_28_team_current_output(tmp_path):
    path = _active_roles(tmp_path, LIVE)
    scope = _validate_current_c2_scope(
        {"football_qb_rows": len(LIVE)},
        _c2(LIVE),
        active_roles_path=path,
    )
    assert scope["mode"] == "EXPLICIT_CURRENT_AVAILABILITY"
    assert scope["observed_teams"] == 28


def test_c2_scope_rejects_missing_explicit_eligible_team(tmp_path):
    path = _active_roles(tmp_path, LIVE)
    with pytest.raises(RuntimeError, match="certified eligible teams"):
        _validate_current_c2_scope(
            {"football_qb_rows": len(LIVE) - 1},
            _c2(LIVE[:-1]),
            active_roles_path=path,
        )


def test_c2_scope_keeps_legacy_32_team_contract():
    scope = _validate_current_c2_scope(
        {"football_qb_rows": 32},
        _c2(ALL_TEAMS),
        active_roles_path=None,
    )
    assert scope["mode"] == "LEGACY_32_TEAM"
    assert scope["observed_teams"] == 32


def test_priced_c2_scope_allows_one_football_qb_without_pass_yard_offer():
    selected = set(LIVE[:26])
    lookup = _lookup(LIVE, selected)
    missing_offer_team = LIVE[-1]
    matched = set(lookup) - {(missing_offer_team, f"{missing_offer_team.lower()}qb")}
    selected_priced = len({identity for identity in matched if identity[0] in selected})

    scope = _validate_priced_c2_subset(lookup, matched, selected_priced)

    assert scope["football_qbs"] == 28
    assert scope["priced_pass_yard_qbs"] == 27
    assert scope["football_qbs_without_priced_pass_yard_offer"] == 1
    assert scope["c2_selected_priced_qbs"] == selected_priced


def test_priced_c2_scope_rejects_identity_outside_football_audit():
    lookup = _lookup(LIVE)
    matched = set(lookup) | {("ZZZ", "unknownqb")}
    with pytest.raises(RuntimeError, match="absent from C2 football audit"):
        _validate_priced_c2_subset(lookup, matched, 0)


def test_priced_c2_scope_rejects_selected_count_drift():
    selected = {LIVE[0]}
    lookup = _lookup(LIVE, selected)
    with pytest.raises(RuntimeError, match="selected-player count drift"):
        _validate_priced_c2_subset(lookup, set(lookup), 0)
