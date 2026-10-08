import json
from pathlib import Path

import pandas as pd

import scripts.operations.restore_preserved_pregame_game_lock_v1 as mod


CERT_COLS=[
    "season","week","game_id","away_team","home_team","kickoff_utc","asof_utc",
    "minutes_to_kickoff","official_required","away_official_section_complete",
    "home_official_section_complete","official_snapshot_asof_utc",
    "certification_state","production_eligible","failure_reason",
]


def _cert_row(*, minutes, eligible, state, official_required, failure_reason=""):
    return {
        "season":2026,
        "week":5,
        "game_id":"2026_5_TB_DAL",
        "away_team":"TB",
        "home_team":"DAL",
        "kickoff_utc":"2026-10-09T00:15:00+00:00",
        "asof_utc":"2026-10-08T23:18:22+00:00" if minutes < 75 else "2026-10-08T22:22:28+00:00",
        "minutes_to_kickoff":minutes,
        "official_required":official_required,
        "away_official_section_complete":False,
        "home_official_section_complete":False,
        "official_snapshot_asof_utc":"",
        "certification_state":state,
        "production_eligible":eligible,
        "failure_reason":failure_reason,
    }


def _role(team, player):
    return {
        "player":player,
        "team":team,
        "role":"QB1",
        "position":"QB",
        "position_group":"QB",
        "player_key":player.lower().replace(" ","")[:8],
        "player_clean_key":player.lower().replace(" ",""),
        "depth_index":1,
        "raw_depth_role":"QB1",
        "availability_authority":"ourlads_depth",
        "final_availability_state":"AVAILABLE_DEPTH_SOURCE",
        "source_asof_utc":"2026-10-08T22:21:31+00:00",
        "availability_generated_at_utc":"2026-10-08T22:22:28+00:00",
    }


def _avail(team, player):
    return {
        "team":team,
        "player":player,
        "position":"QB",
        "position_group":"QB",
        "depth_slot":"Player 1",
        "depth_index":1,
        "depth_chart_role":"QB1",
        "raw_depth_role":"QB1",
        "model_role":"QB1",
        "ourlads_status":"active",
        "player_key":player.lower().replace(" ","")[:8],
        "player_clean_key":player.lower().replace(" ",""),
        "source":"ourlads_depth",
        "source_url":"https://example.test",
        "source_asof_utc":"2026-10-08T22:21:31+00:00",
        "injury_status":"",
        "designation":"",
        "practice_status":"",
        "injury_source":"",
        "report_date":"",
        "official_inactive_section_complete":0,
        "official_inactive":"",
        "final_availability_state":"AVAILABLE_DEPTH_SOURCE",
        "availability_authority":"ourlads_depth",
        "availability_reason":"present active on current depth source",
        "definitive_unavailable":0,
        "role_after_availability":"QB1",
        "role_rank_after_availability":1,
        "eligible_for_opportunity":1,
        "availability_generated_at_utc":"2026-10-08T22:22:28+00:00",
    }


def _wire(tmp_path, monkeypatch):
    data=tmp_path/"data"; outputs=tmp_path/"outputs"; source=tmp_path/"source"; source_data=source/"data"
    data.mkdir(); outputs.mkdir(); source_data.mkdir(parents=True)
    monkeypatch.setattr(mod,"DATA",data)
    monkeypatch.setattr(mod,"OUTPUTS",outputs)
    monkeypatch.setattr(mod,"CUR_CERT",data/"current_player_availability_game_certification.csv")
    monkeypatch.setattr(mod,"CUR_AVAIL",data/"current_player_availability.csv")
    monkeypatch.setattr(mod,"CUR_ROLES",data/"roles_current_production_eligible_v1.csv")
    monkeypatch.setattr(mod,"AUDIT",data/"preserved_pregame_game_lock_audit.json")
    return data,outputs,source,source_data


def test_restores_whole_upcoming_game_from_paid_acquisition_lock(tmp_path, monkeypatch):
    data,outputs,source,source_data=_wire(tmp_path,monkeypatch)

    pd.DataFrame([_cert_row(
        minutes=56.6,
        eligible=False,
        state="REQUIRED_MISSING_FAIL_CLOSED",
        official_required=True,
        failure_reason="missing_complete_section:DAL|missing_complete_section:TB",
    )],columns=CERT_COLS).to_csv(data/"current_player_availability_game_certification.csv",index=False)

    pd.DataFrame([_cert_row(
        minutes=112.5,
        eligible=True,
        state="NOT_YET_REQUIRED",
        official_required=False,
    )],columns=CERT_COLS).to_csv(source_data/"current_player_availability_game_certification.csv",index=False)

    # Current replay universe has already withheld both teams.
    pd.DataFrame([_avail("NYG","Malik Nabers"),_avail("PHI","Jalen Hurts")]).to_csv(
        data/"current_player_availability.csv",index=False
    )
    pd.DataFrame([_role("NYG","Malik Nabers"),_role("PHI","Jalen Hurts")]).to_csv(
        data/"roles_current_production_eligible_v1.csv",index=False
    )

    source_avail=pd.DataFrame([
        _avail("TB","Jalon Daniels"),
        _avail("DAL","Dak Prescott"),
        _avail("NYG","Malik Nabers"),
        _avail("PHI","Jalen Hurts"),
    ])
    source_roles=pd.DataFrame([
        _role("TB","Jalon Daniels"),
        _role("DAL","Dak Prescott"),
        _role("NYG","Malik Nabers"),
        _role("PHI","Jalen Hurts"),
    ])
    source_avail.to_csv(source_data/"current_player_availability.csv",index=False)
    source_roles.to_csv(source_data/"roles_current_production_eligible_v1.csv",index=False)

    result=mod.restore(source,37852811339,11582178322)

    cert=pd.read_csv(data/"current_player_availability_game_certification.csv")
    roles=pd.read_csv(data/"roles_current_production_eligible_v1.csv")
    avail=pd.read_csv(data/"current_player_availability.csv")
    audit=json.loads((data/"preserved_pregame_game_lock_audit.json").read_text())

    assert result["disposition"]=="PRESERVED_PREGAME_ACQUISITION_LOCK_RESTORED"
    assert result["locked_game_ids"]==["2026_5_TB_DAL"]
    assert result["locked_teams"]==["DAL","TB"]
    assert cert.iloc[0]["certification_state"]==mod.LOCK_STATE
    assert bool(cert.iloc[0]["production_eligible"]) is True
    assert set(roles["team"])=={"NYG","PHI","TB","DAL"}
    assert set(avail["team"])=={"NYG","PHI","TB","DAL"}
    assert audit["source_run_id"]==37852811339
    assert audit["sportsbook_inputs_changed"] is False
    assert (outputs/"roles_current_production_eligible_v1.csv").exists()


def test_never_restores_kicked_off_game(tmp_path, monkeypatch):
    data,outputs,source,source_data=_wire(tmp_path,monkeypatch)
    pd.DataFrame([_cert_row(
        minutes=-5,
        eligible=False,
        state="KICKED_OFF_LOCKED",
        official_required=False,
        failure_reason="kickoff_at_or_before_asof",
    )],columns=CERT_COLS).to_csv(data/"current_player_availability_game_certification.csv",index=False)
    pd.DataFrame([_cert_row(
        minutes=112.5,
        eligible=True,
        state="NOT_YET_REQUIRED",
        official_required=False,
    )],columns=CERT_COLS).to_csv(source_data/"current_player_availability_game_certification.csv",index=False)

    pd.DataFrame([_avail("NYG","Malik Nabers"),_avail("PHI","Jalen Hurts")]).to_csv(
        data/"current_player_availability.csv",index=False
    )
    pd.DataFrame([_role("NYG","Malik Nabers"),_role("PHI","Jalen Hurts")]).to_csv(
        data/"roles_current_production_eligible_v1.csv",index=False
    )
    pd.DataFrame([_avail("TB","Jalon Daniels"),_avail("DAL","Dak Prescott")]).to_csv(
        source_data/"current_player_availability.csv",index=False
    )
    pd.DataFrame([_role("TB","Jalon Daniels"),_role("DAL","Dak Prescott")]).to_csv(
        source_data/"roles_current_production_eligible_v1.csv",index=False
    )

    result=mod.restore(source,37852811339,11582178322)
    assert result["disposition"]=="NO_PREGAME_ACQUISITION_LOCK_NEEDED"
    assert set(pd.read_csv(data/"roles_current_production_eligible_v1.csv")["team"])=={"NYG","PHI"}


def test_does_not_override_non_timing_failure(tmp_path, monkeypatch):
    data,outputs,source,source_data=_wire(tmp_path,monkeypatch)
    pd.DataFrame([_cert_row(
        minutes=56.6,
        eligible=False,
        state="REQUIRED_MISSING_FAIL_CLOSED",
        official_required=True,
        failure_reason="some_other_failure",
    )],columns=CERT_COLS).to_csv(data/"current_player_availability_game_certification.csv",index=False)
    pd.DataFrame([_cert_row(
        minutes=112.5,
        eligible=True,
        state="NOT_YET_REQUIRED",
        official_required=False,
    )],columns=CERT_COLS).to_csv(source_data/"current_player_availability_game_certification.csv",index=False)

    pd.DataFrame([_avail("NYG","Malik Nabers"),_avail("PHI","Jalen Hurts")]).to_csv(
        data/"current_player_availability.csv",index=False
    )
    pd.DataFrame([_role("NYG","Malik Nabers"),_role("PHI","Jalen Hurts")]).to_csv(
        data/"roles_current_production_eligible_v1.csv",index=False
    )
    pd.DataFrame([_avail("TB","Jalon Daniels"),_avail("DAL","Dak Prescott")]).to_csv(
        source_data/"current_player_availability.csv",index=False
    )
    pd.DataFrame([_role("TB","Jalon Daniels"),_role("DAL","Dak Prescott")]).to_csv(
        source_data/"roles_current_production_eligible_v1.csv",index=False
    )

    result=mod.restore(source,37852811339,11582178322)
    assert result["disposition"]=="NO_PREGAME_ACQUISITION_LOCK_NEEDED"
