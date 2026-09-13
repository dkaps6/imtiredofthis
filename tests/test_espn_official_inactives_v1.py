from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts.providers import espn_official_inactives_v1 as espn


def _item(*, athlete_id: str, status: str, date: datetime) -> dict:
    return {
        "status": status,
        "date": date.strftime("%Y-%m-%dT%H:%MZ"),
        "athlete": {"$ref": f"http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2026/athletes/{athlete_id}?lang=en&region=us"},
    }


def test_athlete_id_from_ref_extracts_numeric_id():
    ref = "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2026/athletes/3126352?lang=en"
    assert espn._athlete_id_from_ref(ref) == "3126352"


def test_athlete_id_from_ref_empty_on_bad_input():
    assert espn._athlete_id_from_ref("") == ""
    assert espn._athlete_id_from_ref(None) == ""
    assert espn._athlete_id_from_ref("not a ref") == ""


def test_current_out_dedupes_to_latest_entry_per_athlete():
    now = datetime(2026, 9, 13, 20, 0, tzinfo=timezone.utc)
    items = [
        _item(athlete_id="1", status="Questionable", date=now - timedelta(hours=1)),
        _item(athlete_id="1", status="Out", date=now - timedelta(days=3)),  # stale duplicate, ignored
        _item(athlete_id="2", status="Out", date=now - timedelta(hours=2)),
    ]
    result = espn.current_out_athlete_ids(items, now=now)
    ids = {r["athlete_id"] for r in result}
    assert ids == {"2"}


def test_current_out_filters_to_status_out_only():
    now = datetime(2026, 9, 13, 20, 0, tzinfo=timezone.utc)
    items = [
        _item(athlete_id="1", status="Questionable", date=now - timedelta(hours=1)),
        _item(athlete_id="2", status="Doubtful", date=now - timedelta(hours=1)),
        _item(athlete_id="3", status="Active", date=now - timedelta(hours=1)),
        _item(athlete_id="4", status="Out", date=now - timedelta(hours=1)),
        _item(athlete_id="5", status="out", date=now - timedelta(hours=1)),  # case-insensitive
    ]
    result = espn.current_out_athlete_ids(items, now=now)
    ids = {r["athlete_id"] for r in result}
    assert ids == {"4", "5"}


def test_current_out_skips_stale_entries_beyond_recent_window():
    now = datetime(2026, 9, 13, 20, 0, tzinfo=timezone.utc)
    items = [
        _item(athlete_id="1", status="Out", date=now - timedelta(days=espn.RECENT_WINDOW_DAYS + 1)),
        _item(athlete_id="2", status="Out", date=now - timedelta(days=espn.RECENT_WINDOW_DAYS - 1)),
    ]
    result = espn.current_out_athlete_ids(items, now=now)
    ids = {r["athlete_id"] for r in result}
    assert ids == {"2"}


def test_current_out_skips_malformed_dates_without_raising():
    now = datetime(2026, 9, 13, 20, 0, tzinfo=timezone.utc)
    items = [
        {"status": "Out", "date": "not-a-date", "athlete": {"$ref": "http://.../athletes/1"}},
        {"status": "Out", "date": None, "athlete": {"$ref": "http://.../athletes/2"}},
        _item(athlete_id="3", status="Out", date=now),
    ]
    result = espn.current_out_athlete_ids(items, now=now)
    ids = {r["athlete_id"] for r in result}
    assert ids == {"3"}


def test_current_out_skips_items_with_no_athlete_ref():
    now = datetime(2026, 9, 13, 20, 0, tzinfo=timezone.utc)
    items = [{"status": "Out", "date": now.strftime("%Y-%m-%dT%H:%MZ"), "athlete": {}}]
    assert espn.current_out_athlete_ids(items, now=now) == []


def test_build_writes_ledger_row_plus_player_rows_for_complete_team():
    records = [{"team": "ATL", "section_complete": True, "players": ["Tua Tagovailoa", "Kyle Pitts"]}]
    frame, status = espn.build(records)
    assert list(frame.columns) == ["team", "player", "listed_position", "section_complete", "source_url", "source_asof_utc"]
    assert len(frame) == 3  # 1 ledger row + 2 player rows
    ledger = frame[frame["player"] == ""]
    assert len(ledger) == 1
    assert ledger.iloc[0]["team"] == "ATL"
    assert ledger.iloc[0]["section_complete"] == 1
    assert set(frame.loc[frame["player"] != "", "player"]) == {"Tua Tagovailoa", "Kyle Pitts"}
    assert status["complete_teams"] == ["ATL"]
    assert status["complete_team_sections"] == 1
    assert status["listed_players"] == 2
    assert status["payload_valid"] is True


def test_build_incomplete_team_has_ledger_row_but_no_players_and_not_marked_complete():
    records = [{"team": "PIT", "section_complete": False, "players": []}]
    frame, status = espn.build(records)
    assert len(frame) == 1
    assert frame.iloc[0]["section_complete"] == 0
    assert status["complete_teams"] == []
    assert status["complete_team_sections"] == 0
    assert status["listed_players"] == 0
    assert status["payload_valid"] is False


def test_build_empty_input_produces_empty_frame_and_invalid_payload():
    frame, status = espn.build([])
    assert frame.empty
    assert status["payload_valid"] is False
    assert status["complete_teams"] == []


def test_main_writes_csv_and_status_on_scoreboard_failure(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("network down")

    monkeypatch.setattr(espn, "fetch_scoreboard_teams", boom)
    out = tmp_path / "official_inactives_v1.csv"
    status_path = tmp_path / "official_inactives_v1_status.json"
    monkeypatch.setattr("sys.argv", ["prog", "--out", str(out), "--status", str(status_path)])

    rc = espn.main()
    assert rc == 0
    assert out.exists()
    frame = pd.read_csv(out)
    assert frame.empty
    status = json.loads(status_path.read_text())
    assert status["endpoint_reachable"] is False
    assert status["payload_valid"] is False


def test_main_end_to_end_with_mocked_fetchers(tmp_path, monkeypatch):
    monkeypatch.setattr(espn, "fetch_scoreboard_teams", lambda: [
        {"team_id": "1", "team": "ATL"},
        {"team_id": "23", "team": "PIT"},
    ])

    def fake_items(team_id, *, limit=espn.ITEMS_PER_TEAM):
        if team_id == "1":
            return ["atl-item"]
        raise RuntimeError("PIT endpoint unreachable")

    monkeypatch.setattr(espn, "fetch_team_injury_items", fake_items)
    monkeypatch.setattr(
        espn, "current_out_athlete_ids",
        lambda items, *, now: [{"athlete_id": "9", "athlete_ref": "ref-9", "status_date": now.isoformat()}] if items == ["atl-item"] else [],
    )
    monkeypatch.setattr(espn, "fetch_athlete_name", lambda ref: "Tua Tagovailoa" if ref == "ref-9" else "")

    out = tmp_path / "official_inactives_v1.csv"
    status_path = tmp_path / "official_inactives_v1_status.json"
    monkeypatch.setattr("sys.argv", ["prog", "--out", str(out), "--status", str(status_path)])

    rc = espn.main()
    assert rc == 0

    frame = pd.read_csv(out, keep_default_na=False)
    atl_rows = frame[frame["team"] == "ATL"]
    pit_rows = frame[frame["team"] == "PIT"]
    assert (atl_rows["section_complete"] == 1).all()
    assert "Tua Tagovailoa" in set(atl_rows["player"])
    assert (pit_rows["section_complete"] == 0).all()
    assert pit_rows["player"].eq("").all()

    status = json.loads(status_path.read_text())
    assert status["complete_teams"] == ["ATL"]
    assert status["listed_players"] == 1
