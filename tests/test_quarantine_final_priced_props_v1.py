import pandas as pd
import pytest

from scripts.operations.quarantine_final_priced_props_v1 import (
    _load_quarantine_keys,
    quarantine_final_priced_props,
)

VALID_ROW = {
    "player": "Tua Tagovailoa",
    "team": "ATL",
    "season": 2026,
    "week": 1,
    "reason": "Injury report and official starter announcement conflict for tonight.",
    "verified_source": "https://www.atlantafalcons.com/news/falcons-name-tua-tagovailoa-starting-quarterback-week-1",
    "verified_date": "2026-09-13",
}


def _write_quarantine(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_priced(path, players, teams=None):
    teams = teams or ["ATL"] * len(players)
    pd.DataFrame(
        {
            "player": players,
            "team": teams,
            "source_market": ["player_pass_yds"] * len(players),
            "model_proj": [1.0] * len(players),
        }
    ).to_csv(path, index=False)


def test_load_quarantine_keys_empty_when_file_missing(tmp_path):
    assert _load_quarantine_keys(tmp_path / "missing.csv", season=2026, week=1) == set()


def test_load_quarantine_keys_reads_verified_rows_for_current_week(tmp_path):
    path = tmp_path / "quarantine.csv"
    _write_quarantine(path, [VALID_ROW])
    assert _load_quarantine_keys(path, season=2026, week=1) == {("ATL", "tuatagovailoa")}


def test_load_quarantine_keys_does_not_apply_to_a_different_week(tmp_path):
    path = tmp_path / "quarantine.csv"
    _write_quarantine(path, [VALID_ROW])
    assert _load_quarantine_keys(path, season=2026, week=2) == set()
    assert _load_quarantine_keys(path, season=2027, week=1) == set()


def test_load_quarantine_keys_missing_columns_raises(tmp_path):
    path = tmp_path / "quarantine.csv"
    pd.DataFrame({"player": ["Tua Tagovailoa"]}).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="missing columns"):
        _load_quarantine_keys(path, season=2026, week=1)


def test_load_quarantine_keys_rejects_shifted_index(tmp_path):
    path = tmp_path / "quarantine.csv"
    path.write_text(
        "player,team,season,week,reason,verified_source,verified_date\n"
        "Tua Tagovailoa,ATL,2026,1,unquoted, comma, breaks, parsing,src,2026-09-13\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="non-default index"):
        _load_quarantine_keys(path, season=2026, week=1)


def test_load_quarantine_keys_rejects_non_numeric_season_week(tmp_path):
    path = tmp_path / "quarantine.csv"
    _write_quarantine(path, [{**VALID_ROW, "season": "not-a-season"}])
    with pytest.raises(RuntimeError, match="non-numeric season/week"):
        _load_quarantine_keys(path, season=2026, week=1)


def test_quarantine_removes_only_matching_team_player_pairs(tmp_path):
    out_path = tmp_path / "props_priced_clean.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    _write_priced(
        out_path,
        ["Tua Tagovailoa", "Cooper Rush", "Bijan Robinson"],
        teams=["ATL", "ATL", "ATL"],
    )
    _write_quarantine(quarantine_path, [VALID_ROW, {**VALID_ROW, "player": "Cooper Rush"}])

    status = quarantine_final_priced_props(
        out_path=out_path,
        quarantine_path=quarantine_path,
        status_path=tmp_path / "status.json",
        season=2026,
        week=1,
    )

    assert status["rows_removed"] == 2
    assert status["rows_remaining"] == 1
    remaining = pd.read_csv(out_path)
    assert list(remaining["player"]) == ["Bijan Robinson"]


def test_quarantine_does_not_suppress_a_same_named_player_on_another_team(tmp_path):
    out_path = tmp_path / "props_priced_clean.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    _write_priced(
        out_path,
        ["Tua Tagovailoa", "Tua Tagovailoa"],
        teams=["ATL", "MIA"],
    )
    _write_quarantine(quarantine_path, [VALID_ROW])

    status = quarantine_final_priced_props(
        out_path=out_path,
        quarantine_path=quarantine_path,
        status_path=tmp_path / "status.json",
        season=2026,
        week=1,
    )

    assert status["rows_removed"] == 1
    remaining = pd.read_csv(out_path)
    assert list(remaining["team"]) == ["MIA"]


def test_quarantine_does_not_apply_outside_its_scoped_week(tmp_path):
    out_path = tmp_path / "props_priced_clean.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    _write_priced(out_path, ["Tua Tagovailoa"], teams=["ATL"])
    _write_quarantine(quarantine_path, [VALID_ROW])

    status = quarantine_final_priced_props(
        out_path=out_path,
        quarantine_path=quarantine_path,
        status_path=tmp_path / "status.json",
        season=2026,
        week=2,
    )

    assert status["rows_removed"] == 0
    assert status["rows_remaining"] == 1


def test_quarantine_is_a_noop_without_keys(tmp_path):
    out_path = tmp_path / "props_priced_clean.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    _write_priced(out_path, ["Bijan Robinson"])

    status = quarantine_final_priced_props(
        out_path=out_path,
        quarantine_path=quarantine_path,
        status_path=tmp_path / "status.json",
        season=2026,
        week=1,
    )

    assert status["rows_removed"] == 0
    assert status["rows_remaining"] == 1


def test_quarantine_requires_existing_output(tmp_path):
    with pytest.raises(RuntimeError, match="missing/empty"):
        quarantine_final_priced_props(
            out_path=tmp_path / "missing.csv",
            quarantine_path=tmp_path / "quarantine.csv",
            status_path=tmp_path / "status.json",
            season=2026,
            week=1,
        )


def test_tracked_manual_final_board_quarantine_file_is_well_formed():
    from pathlib import Path

    path = Path("data/manual_final_board_quarantine.csv")
    keys = _load_quarantine_keys(path, season=2026, week=1)
    assert keys == {
        ("ATL", "michaelpenix"),
        ("ATL", "tuatagovailoa"),
        ("ATL", "cooperrush"),
    }


def test_tracked_manual_final_board_quarantine_file_does_not_apply_to_other_weeks():
    from pathlib import Path

    path = Path("data/manual_final_board_quarantine.csv")
    assert _load_quarantine_keys(path, season=2026, week=2) == set()
