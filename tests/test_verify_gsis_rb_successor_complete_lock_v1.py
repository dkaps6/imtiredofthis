import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.research.verify_gsis_rb_successor_complete_lock_v1 import main


def _write_fixture(tmp_path: Path):
    alloc=tmp_path/"alloc.csv"
    events=tmp_path/"events.csv"
    proj=tmp_path/"proj.csv"
    manifest=tmp_path/"manifest.json"

    pd.DataFrame([
        {"target_season":2026,"target_week":5,"event_id":"g1","team":"DEN","successor_player_clean_key":"a"},
        {"target_season":2026,"target_week":5,"event_id":"g1","team":"DEN","successor_player_clean_key":"b"},
    ]).to_csv(alloc,index=False)
    pd.DataFrame([
        {"event_id":"g1","team":"DEN"}
    ]).to_csv(events,index=False)
    pd.DataFrame([
        {"target_season":2026,"target_week":5,"event_id":"g1","team":"DEN","player_clean_key":"a","market":"rush_att"},
        {"target_season":2026,"target_week":5,"event_id":"g1","team":"DEN","player_clean_key":"a","market":"rush_yards"},
        {"target_season":2026,"target_week":5,"event_id":"g1","team":"DEN","player_clean_key":"b","market":"rush_att"},
        {"target_season":2026,"target_week":5,"event_id":"g1","team":"DEN","player_clean_key":"b","market":"rush_yards"},
    ]).to_csv(proj,index=False)
    manifest.write_text(json.dumps({
        "disposition":"GSIS_RB_SUCCESSOR_LINEUP_V1_THREE_ARM_PROJECTION_LOCKED",
        "sportsbook_inputs_used":0,
        "target_game_outcomes_attached":0,
        "private_rows_must_not_be_committed":True,
        "arms":["BASELINE","VACANCY_V1_SNAP","GSIS_LINEUP_V1"],
    }),encoding="utf-8")
    return alloc,events,proj,manifest


def test_complete_lock_verifier_requires_projection_and_emits_public_safe_receipt(tmp_path, monkeypatch):
    alloc,events,proj,manifest=_write_fixture(tmp_path)
    out=tmp_path/"receipt.json"
    monkeypatch.setattr("sys.argv",[
        "verify",
        "--allocation-lock",str(alloc),
        "--event-audit",str(events),
        "--projection-lock",str(proj),
        "--projection-manifest",str(manifest),
        "--out",str(out),
        "--expected-season","2026",
        "--expected-week","5",
    ])
    assert main()==0
    receipt=json.loads(out.read_text())
    assert receipt["disposition"]=="GSIS_RB_SUCCESSOR_LINEUP_V1_COMPLETE_PREGAME_LOCK_READY"
    assert receipt["locked_players"]==2
    assert "player" not in receipt
    assert receipt["sportsbook_inputs_used"]==0


def test_complete_lock_verifier_fails_if_projection_missing(tmp_path, monkeypatch):
    alloc,events,proj,manifest=_write_fixture(tmp_path)
    proj.unlink()
    out=tmp_path/"receipt.json"
    monkeypatch.setattr("sys.argv",[
        "verify",
        "--allocation-lock",str(alloc),
        "--event-audit",str(events),
        "--projection-lock",str(proj),
        "--projection-manifest",str(manifest),
        "--out",str(out),
        "--expected-season","2026",
        "--expected-week","5",
    ])
    with pytest.raises(RuntimeError, match="missing/empty required private file"):
        main()
