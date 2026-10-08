import json

import pandas as pd
import pytest

import scripts.operations.reconcile_preserved_live_props_current_roster_v1 as mod


def _wire(tmp_path, monkeypatch):
    data=tmp_path/"data"; outputs=tmp_path/"outputs"
    data.mkdir(); outputs.mkdir()
    monkeypatch.setattr(mod,"DATA",data)
    monkeypatch.setattr(mod,"OUTPUTS",outputs)
    monkeypatch.setattr(mod,"STATUS",data/"live_odds_status.json")
    monkeypatch.setattr(mod,"FORM",data/"player_form.csv")
    monkeypatch.setattr(mod,"COMPACT",outputs/"props_raw_compact.csv")
    monkeypatch.setattr(mod,"MODEL_PROPS",outputs/"props_raw.csv")
    monkeypatch.setattr(mod,"QUARANTINE",data/"live_odds_placeholder_rows.csv")
    monkeypatch.setattr(mod,"AUDIT",data/"preserved_live_props_current_roster_reconciliation.json")
    monkeypatch.setattr(mod,"GAME_CERT",data/"current_player_availability_game_certification.csv")
    return data,outputs


def _write_cert(data, *, away="DAL", home="NYG", eligible=True, state="NOT_YET_REQUIRED"):
    pd.DataFrame([{
        "away_team":away,
        "home_team":home,
        "production_eligible":eligible,
        "certification_state":state,
    }]).to_csv(data/"current_player_availability_game_certification.csv",index=False)


def test_quarantines_only_unrostered_noncore_row(tmp_path,monkeypatch):
    data,outputs=_wire(tmp_path,monkeypatch)
    pd.DataFrame([
        {"team":"DAL","player":"Dak Prescott"},
    ]).to_csv(data/"player_form.csv",index=False)
    _write_cert(data)

    compact=pd.DataFrame([
        {"event_id":"e1","market":"player_pass_yds","player":"Dak Prescott","team_abbr":"DAL","opponent_abbr":"NYG","offers_json":"[]"},
        {"event_id":"e1","market":"player_anytime_td","player":"Camden Brown","team_abbr":"DAL","opponent_abbr":"NYG","offers_json":"[]"},
    ])
    compact.to_csv(outputs/"props_raw_compact.csv",index=False)
    compact.to_csv(outputs/"props_raw.csv",index=False)
    pd.DataFrame([dict(compact.iloc[0],quarantine_reason="EXISTING")]).to_csv(
        data/"live_odds_placeholder_rows.csv",index=False
    )
    (data/"live_odds_status.json").write_text(json.dumps({
        "production_compact_rows":2,
        "production_quarantined_rows":1,
        "production_quarantine_reasons":{"EXISTING":1},
    }))

    result=mod.reconcile()
    out=pd.read_csv(outputs/"props_raw_compact.csv")
    q=pd.read_csv(data/"live_odds_placeholder_rows.csv")
    status=json.loads((data/"live_odds_status.json").read_text())

    assert result["rows_quarantined"]==1
    assert result["quarantined_players"]==["DAL:Camden Brown"]
    assert out["player"].tolist()==["Dak Prescott"]
    assert q["quarantine_reason"].value_counts().to_dict()=={
        "EXISTING":1,
        "UNROSTERED_NONCORE_PLAYER":1,
    }
    assert status["production_compact_rows"]==1
    assert status["production_quarantined_rows"]==2
    assert status["preserved_replay_noncore_rows_quarantined"]==1


def test_unrostered_strict_market_remains_fatal(tmp_path,monkeypatch):
    data,outputs=_wire(tmp_path,monkeypatch)
    pd.DataFrame([{"team":"DAL","player":"Dak Prescott"}]).to_csv(
        data/"player_form.csv",index=False
    )
    _write_cert(data)
    compact=pd.DataFrame([
        {"event_id":"e1","market":"player_pass_yds","player":"Other QB","team_abbr":"DAL","opponent_abbr":"NYG","offers_json":"[]"},
    ])
    compact.to_csv(outputs/"props_raw_compact.csv",index=False)
    compact.to_csv(outputs/"props_raw.csv",index=False)
    pd.DataFrame(columns=list(compact.columns)+["quarantine_reason"]).to_csv(
        data/"live_odds_placeholder_rows.csv",index=False
    )
    (data/"live_odds_status.json").write_text("{}")

    with pytest.raises(RuntimeError,match="strict-market players absent"):
        mod.reconcile()


def test_withheld_game_quarantines_strict_and_noncore_rows(tmp_path,monkeypatch):
    data,outputs=_wire(tmp_path,monkeypatch)
    pd.DataFrame([
        {"team":"NYG","player":"Malik Nabers"},
    ]).to_csv(data/"player_form.csv",index=False)
    _write_cert(
        data,
        away="DAL",
        home="TB",
        eligible=False,
        state="REQUIRED_MISSING_FAIL_CLOSED",
    )

    compact=pd.DataFrame([
        {"event_id":"e1","market":"player_pass_yds","player":"Dak Prescott","team_abbr":"DAL","opponent_abbr":"TB","offers_json":"[]"},
        {"event_id":"e1","market":"player_reception_yds","player":"CeeDee Lamb","team_abbr":"DAL","opponent_abbr":"TB","offers_json":"[]"},
        {"event_id":"e1","market":"player_rush_yds","player":"Bucky Irving","team_abbr":"TB","opponent_abbr":"DAL","offers_json":"[]"},
        {"event_id":"e1","market":"player_anytime_td","player":"Camden Brown","team_abbr":"DAL","opponent_abbr":"TB","offers_json":"[]"},
        {"event_id":"e2","market":"player_reception_yds","player":"Malik Nabers","team_abbr":"NYG","opponent_abbr":"PHI","offers_json":"[]"},
    ])
    compact.to_csv(outputs/"props_raw_compact.csv",index=False)
    compact.to_csv(outputs/"props_raw.csv",index=False)
    pd.DataFrame(columns=list(compact.columns)+["quarantine_reason"]).to_csv(
        data/"live_odds_placeholder_rows.csv",index=False
    )
    (data/"live_odds_status.json").write_text("{}")

    result=mod.reconcile()
    out=pd.read_csv(outputs/"props_raw_compact.csv")
    q=pd.read_csv(data/"live_odds_placeholder_rows.csv")
    status=json.loads((data/"live_odds_status.json").read_text())

    assert result["withheld_teams"]==["DAL","TB"]
    assert result["withheld_game_rows_removed"]==4
    assert result["strict_market_rows_removed_due_withheld_game"]==3
    assert result["strict_market_rows_removed_outside_withheld_games"]==0
    assert out["player"].tolist()==["Malik Nabers"]
    assert q["quarantine_reason"].value_counts().to_dict()=={
        "WITHHELD_GAME_CURRENT_AVAILABILITY":4,
    }
    assert status["preserved_replay_withheld_teams"]==["DAL","TB"]
    assert status["preserved_replay_withheld_game_rows_quarantined"]==4


def test_unrostered_strict_market_on_eligible_game_still_fails(tmp_path,monkeypatch):
    data,outputs=_wire(tmp_path,monkeypatch)
    pd.DataFrame([{"team":"NYG","player":"Malik Nabers"}]).to_csv(
        data/"player_form.csv",index=False
    )
    _write_cert(data,away="DAL",home="NYG",eligible=True)

    compact=pd.DataFrame([
        {"event_id":"e1","market":"player_pass_yds","player":"Other QB","team_abbr":"DAL","opponent_abbr":"NYG","offers_json":"[]"},
    ])
    compact.to_csv(outputs/"props_raw_compact.csv",index=False)
    compact.to_csv(outputs/"props_raw.csv",index=False)
    pd.DataFrame(columns=list(compact.columns)+["quarantine_reason"]).to_csv(
        data/"live_odds_placeholder_rows.csv",index=False
    )
    (data/"live_odds_status.json").write_text("{}")

    with pytest.raises(RuntimeError,match="outside explicitly withheld games"):
        mod.reconcile()
