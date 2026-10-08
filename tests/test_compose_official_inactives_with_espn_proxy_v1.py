import pandas as pd
import pytest

from scripts.operations.compose_official_inactives_with_espn_proxy_v1 import compose


def test_composite_preserves_true_official_completion_and_proxy_out():
    nfl=pd.DataFrame([
        {
            "team":"DAL","player":"","listed_position":"",
            "section_complete":1,"source_url":"nfl","source_asof_utc":"2026-10-08T23:00:00+00:00",
        },
        {
            "team":"DAL","player":"Example Inactive","listed_position":"WR",
            "section_complete":1,"source_url":"nfl","source_asof_utc":"2026-10-08T23:00:00+00:00",
        },
        {
            "team":"TB","player":"","listed_position":"",
            "section_complete":1,"source_url":"nfl","source_asof_utc":"2026-10-08T23:00:00+00:00",
        },
    ])
    espn=pd.DataFrame([
        {
            "team":"DAL","player":"","listed_position":"",
            "section_complete":0,"source_semantics":"INJURY_STATUS_OUT",
            "provider_status":"","source_fetch_complete":1,
            "source_url":"espn","source_asof_utc":"2026-10-08T23:01:00+00:00",
        },
        {
            "team":"DAL","player":"Reported Out","listed_position":"",
            "section_complete":0,"source_semantics":"INJURY_STATUS_OUT",
            "provider_status":"OUT","source_fetch_complete":1,
            "source_url":"espn","source_asof_utc":"2026-10-08T23:01:00+00:00",
        },
    ])
    out,status=compose(nfl,espn,nfl_status={"payload_valid":True},espn_status={"payload_valid":True})
    assert status["complete_teams"]==["DAL","TB"]
    assert status["complete_team_sections"]==2
    assert status["sportsbook_inputs_used"]==0
    official=out[out["source_semantics"].eq("OFFICIAL_GAME_DAY_INACTIVE")]
    proxy=out[out["source_semantics"].eq("INJURY_STATUS_OUT")]
    assert set(official["team"])=={"DAL","TB"}
    assert proxy["section_complete"].eq(0).all()
    assert "Reported Out" in set(proxy["player"])


def test_proxy_can_never_mark_section_complete():
    espn=pd.DataFrame([{
        "team":"DAL","player":"","section_complete":1,
        "source_semantics":"INJURY_STATUS_OUT",
    }])
    out,status=compose(pd.DataFrame(),espn,nfl_status={},espn_status={"payload_valid":True})
    assert out["section_complete"].eq(0).all()
    assert status["complete_teams"]==[]


def test_non_proxy_semantics_in_espn_ledger_fail():
    espn=pd.DataFrame([{
        "team":"DAL","player":"","section_complete":0,
        "source_semantics":"OFFICIAL_GAME_DAY_INACTIVE",
    }])
    with pytest.raises(RuntimeError,match="non-proxy semantics"):
        compose(pd.DataFrame(),espn,nfl_status={},espn_status={})
