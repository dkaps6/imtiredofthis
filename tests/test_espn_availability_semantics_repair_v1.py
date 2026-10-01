from datetime import datetime, timezone
import pandas as pd

from scripts.build.build_current_player_availability_v1 import build
from scripts.validate_current_player_availability_timing_v1 import certify


def _depth():
    return pd.DataFrame([{
        "team":"PIT","player":"Rico Dowdle","status":"active","role":"RB2",
        "position":"RB","position_group":"RB","depth_index":2,
    }])


def test_espn_injury_out_proxy_becomes_reported_unavailable_not_official():
    proxy=pd.DataFrame([
        {
            "team":"PIT","player":"","section_complete":0,
            "source_semantics":"INJURY_STATUS_OUT","provider_status":"",
            "source_fetch_complete":1,
        },
        {
            "team":"PIT","player":"Rico Dowdle","section_complete":0,
            "source_semantics":"INJURY_STATUS_OUT","provider_status":"OUT",
            "source_fetch_complete":1,
        },
    ])
    out,meta=build(_depth(),pd.DataFrame(),proxy)
    r=out.iloc[0]
    assert r["final_availability_state"]=="UNAVAILABLE_REPORTED"
    assert r["availability_authority"]=="weekly_injury_report"
    assert r["injury_status"]=="OUT"
    assert r["injury_source"]=="espn_core_injury_status_out"
    assert int(r["official_inactive_section_complete"])==0
    assert pd.isna(r["official_inactive"])
    assert int(r["definitive_unavailable"])==1
    assert meta["official_complete_teams"]==0


def test_legacy_true_official_inactive_semantics_still_supported():
    official=pd.DataFrame([
        {"team":"PIT","player":"","section_complete":1},
        {"team":"PIT","player":"Rico Dowdle","section_complete":1},
    ])
    out,_=build(_depth(),pd.DataFrame(),official)
    r=out.iloc[0]
    assert r["final_availability_state"]=="UNAVAILABLE_OFFICIAL_INACTIVE"
    assert r["availability_authority"]=="official_inactive"
    assert int(r["official_inactive_section_complete"])==1
    assert bool(r["official_inactive"])


def test_injury_proxy_cannot_satisfy_t75_official_certification():
    schedule=pd.DataFrame([{
        "season":2026,"week":4,"game_id":"g",
        "away_team":"PIT","home_team":"CLE",
        "kickoff_utc":"2026-10-02T00:15:00+00:00",
    }])
    proxy=pd.DataFrame([
        {
            "team":"PIT","player":"","section_complete":0,
            "source_semantics":"INJURY_STATUS_OUT","provider_status":"",
            "source_fetch_complete":1,
            "source_asof_utc":"2026-10-01T23:15:00+00:00",
        },
        {
            "team":"CLE","player":"","section_complete":0,
            "source_semantics":"INJURY_STATUS_OUT","provider_status":"",
            "source_fetch_complete":1,
            "source_asof_utc":"2026-10-01T23:15:00+00:00",
        },
    ])
    out,meta=certify(
        schedule,proxy,asof_utc=datetime(2026,10,1,23,15,tzinfo=timezone.utc)
    )
    r=out.iloc[0]
    assert bool(r["official_required"])
    assert r["certification_state"]=="REQUIRED_MISSING_FAIL_CLOSED"
    assert not bool(r["production_eligible"])
    assert meta["withheld_games"]==1


def test_espn_out_overrides_older_questionable_injury_status():
    injuries=pd.DataFrame([{
        "team":"PIT","player":"Rico Dowdle","status":"QUESTIONABLE",
        "designation":"QUESTIONABLE","practice_status":"DNP",
        "source":"nflverse","report_date":"2026-09-30",
    }])
    proxy=pd.DataFrame([
        {
            "team":"PIT","player":"","section_complete":0,
            "source_semantics":"INJURY_STATUS_OUT","provider_status":"",
            "source_fetch_complete":1,
        },
        {
            "team":"PIT","player":"Rico Dowdle","section_complete":0,
            "source_semantics":"INJURY_STATUS_OUT","provider_status":"OUT",
            "source_fetch_complete":1,
        },
    ])
    out,_=build(_depth(),injuries,proxy)
    r=out.iloc[0]
    assert r["injury_status"]=="OUT"
    assert r["designation"]=="OUT"
    assert r["injury_source"]=="espn_core_injury_status_out"
    assert r["final_availability_state"]=="UNAVAILABLE_REPORTED"
    assert int(r["definitive_unavailable"])==1
