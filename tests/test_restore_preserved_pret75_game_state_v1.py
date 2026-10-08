import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.operations.restore_preserved_pret75_game_state_v1 as mod


def _write_fixture(root: Path, current_state: str, current_eligible: bool, source_state: str = "NOT_YET_REQUIRED", source_eligible: bool = True):
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    cert_cols = [
        "season","week","game_id","away_team","home_team","kickoff_utc","asof_utc",
        "minutes_to_kickoff","official_required","away_official_section_complete",
        "home_official_section_complete","official_snapshot_asof_utc",
        "certification_state","production_eligible","failure_reason",
    ]
    cur = pd.DataFrame([[
        2026,5,"2026_05_TB_DAL","TB","DAL","2099-10-08T00:20:00+00:00","2099-10-07T23:30:00+00:00",
        50.0,True,False,False,"",current_state,current_eligible,
        "missing_complete_section:TB|missing_complete_section:DAL" if not current_eligible else "",
    ]], columns=cert_cols)
    src = pd.DataFrame([[
        2026,5,"2026_05_TB_DAL","TB","DAL","2099-10-08T00:20:00+00:00","2099-10-07T22:55:00+00:00",
        85.0,False,False,False,"",source_state,source_eligible,"",
    ]], columns=cert_cols)
    cur.to_csv(data/"current_player_availability_game_certification.csv",index=False)
    (data/"current_player_availability_game_certification.json").write_text(json.dumps({
        "games":1,"eligible_games":int(current_eligible),"withheld_games":int(not current_eligible),
        "state_counts":{current_state:1},"withheld_teams":[] if current_eligible else ["DAL","TB"],
        "sportsbook_inputs_used":0,
    }))
    pd.DataFrame([
        {"team":"TB","player":"Current TB","player_clean_key":"currenttb","definitive_unavailable":0},
        {"team":"TB","player":"Current TB Out","player_clean_key":"currenttbout","definitive_unavailable":1},
        {"team":"DAL","player":"Current DAL","player_clean_key":"currentdal","definitive_unavailable":0},
    ]).to_csv(data/"current_player_availability.csv",index=False)
    pd.DataFrame([
        {"team":"TB","player":"Current TB","role":"QB1","position":"QB","player_clean_key":"currenttb"},
        {"team":"DAL","player":"Current DAL","role":"QB1","position":"QB","player_clean_key":"currentdal"},
    ]).to_csv(data/"roles_ourlads_active_v1.csv",index=False)
    pd.DataFrame(columns=["team","player","role","position","player_clean_key"]).to_csv(
        data/"roles_current_production_eligible_v1.csv",index=False
    )
    (data/"roles_current_production_eligible_v1_status.json").write_text("{}")

    src_root = root/"source"
    sdata = src_root/"data"
    sdata.mkdir(parents=True,exist_ok=True)
    src.to_csv(sdata/"current_player_availability_game_certification.csv",index=False)
    (sdata/"current_player_availability_game_certification.json").write_text(json.dumps({
        "games":1,"eligible_games":int(source_eligible),"withheld_games":int(not source_eligible),
        "state_counts":{source_state:1},"withheld_teams":[],"sportsbook_inputs_used":0,
    }))
    pd.DataFrame([
        {"team":"TB","player":"Preserved TB","player_clean_key":"preservedtb","definitive_unavailable":0},
        {"team":"DAL","player":"Preserved DAL","player_clean_key":"preserveddal","definitive_unavailable":0},
    ]).to_csv(sdata/"current_player_availability.csv",index=False)
    pd.DataFrame([
        {"team":"TB","player":"Preserved TB","role":"QB1","position":"QB","player_clean_key":"preservedtb"},
        {"team":"DAL","player":"Preserved DAL","role":"QB1","position":"QB","player_clean_key":"preserveddal"},
    ]).to_csv(sdata/"roles_current_production_eligible_v1.csv",index=False)
    return data, src_root


def test_restores_whole_game_from_pinned_pre_t75_source(tmp_path, monkeypatch):
    data, src_root = _write_fixture(tmp_path, "REQUIRED_MISSING_FAIL_CLOSED", False)
    monkeypatch.setattr(mod, "DATA", data)
    monkeypatch.setattr(mod, "AUDIT", data/"preserved_pret75_game_state_audit.json")

    result = mod.restore(src_root, 37852811339)

    cert = pd.read_csv(data/"current_player_availability_game_certification.csv")
    roles = pd.read_csv(data/"roles_current_production_eligible_v1.csv")
    avail = pd.read_csv(data/"current_player_availability.csv")
    meta = json.loads((data/"current_player_availability_game_certification.json").read_text())

    assert result["disposition"] == "PRESERVED_PRE_T75_GAME_ELIGIBILITY_RESTORED_CURRENT_PLAYER_STATE_PRESERVED"
    assert result["restored_teams"] == ["DAL","TB"]
    assert cert.loc[0,"certification_state"] == "PRESERVED_PRE_T75_REPLAY"
    assert bool(cert.loc[0,"production_eligible"])
    assert set(roles.team) == {"DAL","TB"}
    assert set(roles.player) == {"Current DAL","Current TB"}
    assert set(avail.player) == {"Current DAL","Current TB","Current TB Out"}
    assert "Current TB Out" not in set(roles.player)
    assert result["current_definitive_unavailable_resurrected"] == 0
    assert meta["withheld_games"] == 0
    assert meta["eligible_games"] == 1
    assert meta["sportsbook_inputs_used"] == 0


def test_does_not_restore_kicked_off_game(tmp_path, monkeypatch):
    data, src_root = _write_fixture(tmp_path, "KICKED_OFF_LOCKED", False)
    # Make current kickoff safely in the past.
    cur = pd.read_csv(data/"current_player_availability_game_certification.csv")
    cur.loc[0,"kickoff_utc"] = "2000-01-01T00:00:00+00:00"
    cur.to_csv(data/"current_player_availability_game_certification.csv",index=False)
    monkeypatch.setattr(mod, "DATA", data)
    monkeypatch.setattr(mod, "AUDIT", data/"preserved_pret75_game_state_audit.json")

    result = mod.restore(src_root, 37852811339)
    assert result["disposition"] == "NO_PRE_T75_GAME_RESTORE_REQUIRED"


def test_rejects_source_not_outside_t75(tmp_path, monkeypatch):
    data, src_root = _write_fixture(
        tmp_path, "REQUIRED_MISSING_FAIL_CLOSED", False,
        source_state="REQUIRED_AND_CERTIFIED", source_eligible=True
    )
    monkeypatch.setattr(mod, "DATA", data)
    monkeypatch.setattr(mod, "AUDIT", data/"preserved_pret75_game_state_audit.json")

    with pytest.raises(RuntimeError, match="did not certify withheld game pre-T75"):
        mod.restore(src_root, 37852811339)


def test_rejects_source_with_sportsbook_availability_provenance(tmp_path, monkeypatch):
    data, src_root = _write_fixture(tmp_path, "REQUIRED_MISSING_FAIL_CLOSED", False)
    p = src_root/"data/current_player_availability_game_certification.json"
    meta = json.loads(p.read_text())
    meta["sportsbook_inputs_used"] = 1
    p.write_text(json.dumps(meta))
    monkeypatch.setattr(mod, "DATA", data)
    monkeypatch.setattr(mod, "AUDIT", data/"preserved_pret75_game_state_audit.json")

    with pytest.raises(RuntimeError, match="not football-only"):
        mod.restore(src_root, 37852811339)


def test_current_definitive_unavailable_cannot_be_resurrected(tmp_path, monkeypatch):
    data, src_root = _write_fixture(tmp_path, "REQUIRED_MISSING_FAIL_CLOSED", False)
    # Deliberately poison the active-role input with a player that current
    # availability already marks definitively unavailable. The seam must fail.
    active = pd.read_csv(data/"roles_ourlads_active_v1.csv")
    active = pd.concat([active, pd.DataFrame([{
        "team":"TB","player":"Current TB Out","role":"WR3","position":"WR","player_clean_key":"currenttbout"
    }])], ignore_index=True)
    active.to_csv(data/"roles_ourlads_active_v1.csv",index=False)

    monkeypatch.setattr(mod, "DATA", data)
    monkeypatch.setattr(mod, "AUDIT", data/"preserved_pret75_game_state_audit.json")
    with pytest.raises(RuntimeError, match="resurrected current definitive-unavailable"):
        mod.restore(src_root, 37852811339)
