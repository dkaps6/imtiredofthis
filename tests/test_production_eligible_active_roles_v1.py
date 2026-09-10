from __future__ import annotations

import pandas as pd
import pytest

from scripts.build.build_production_eligible_active_roles_v1 import build


def roles():
    return pd.DataFrame([
        {"team":"IND","player":"A","role":"QB1","position":"QB","player_clean_key":"a"},
        {"team":"HOU","player":"B","role":"QB1","position":"QB","player_clean_key":"b"},
        {"team":"CHI","player":"C","role":"QB1","position":"QB","player_clean_key":"c"},
        {"team":"GB","player":"D","role":"QB1","position":"QB","player_clean_key":"d"},
    ])


def cert():
    return pd.DataFrame([
        {"away_team":"HOU","home_team":"IND","production_eligible":1,"certification_state":"NOT_YET_REQUIRED"},
        {"away_team":"GB","home_team":"CHI","production_eligible":0,"certification_state":"REQUIRED_MISSING_FAIL_CLOSED"},
    ])


def test_withheld_game_removed_before_opportunity():
    out, meta = build(roles(), cert())
    assert set(out.team) == {"IND", "HOU"}
    assert meta["withheld_games"] == 1
    assert set(meta["withheld_teams"]) == {"CHI", "GB"}


def test_not_yet_required_stays_eligible():
    out, _ = build(roles().iloc[:2].copy(), cert().iloc[:1].copy())
    assert len(out) == 2


def test_duplicate_team_in_certification_fails_closed():
    bad = pd.concat([cert(), pd.DataFrame([{"away_team":"IND","home_team":"DET","production_eligible":1,"certification_state":"NOT_YET_REQUIRED"}])], ignore_index=True)
    with pytest.raises(RuntimeError, match="not one game per team"):
        build(roles(), bad)
