import pandas as pd
import pytest

from scripts.operations.quarantine_final_priced_props_v1 import (
    _load_quarantine_keys,
    quarantine_final_priced_props,
)

VALID_ROW = {
    "player": "Tua Tagovailoa",
    "team": "ATL",
    "reason": "Injury report and official starter announcement conflict for tonight.",
    "verified_source": "https://www.atlantafalcons.com/news/falcons-name-tua-tagovailoa-starting-quarterback-week-1",
    "verified_date": "2026-09-13",
}


def _write_quarantine(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_priced(path, players):
    pd.DataFrame(
        {
            "player": players,
            "team": ["ATL"] * len(players),
            "source_market": ["player_pass_yds"] * len(players),
            "model_proj": [1.0] * len(players),
        }
    ).to_csv(path, index=False)


def test_load_quarantine_keys_empty_when_file_missing(tmp_path):
    assert _load_quarantine_keys(tmp_path / "missing.csv") == set()


def test_load_quarantine_keys_reads_verified_rows(tmp_path):
    path = tmp_path / "quarantine.csv"
    _write_quarantine(path, [VALID_ROW])
    assert _load_quarantine_keys(path) == {"tuatagovailoa"}


def test_load_quarantine_keys_missing_columns_raises(tmp_path):
    path = tmp_path / "quarantine.csv"
    pd.DataFrame({"player": ["Tua Tagovailoa"]}).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="missing columns"):
        _load_quarantine_keys(path)


def test_load_quarantine_keys_rejects_shifted_index(tmp_path):
    path = tmp_path / "quarantine.csv"
    path.write_text(
        "player,team,reason,verified_source,verified_date\n"
        "Tua Tagovailoa,ATL,unquoted, comma, breaks, parsing,src,2026-09-13\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="non-default index"):
        _load_quarantine_keys(path)


def test_quarantine_removes_only_matching_players(tmp_path):
    out_path = tmp_path / "props_priced_clean.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    _write_priced(out_path, ["Tua Tagovailoa", "Cooper Rush", "Bijan Robinson"])
    _write_quarantine(quarantine_path, [VALID_ROW, {**VALID_ROW, "player": "Cooper Rush"}])

    status = quarantine_final_priced_props(
        out_path=out_path, quarantine_path=quarantine_path, status_path=tmp_path / "status.json"
    )

    assert status["rows_removed"] == 2
    assert status["rows_remaining"] == 1
    remaining = pd.read_csv(out_path)
    assert list(remaining["player"]) == ["Bijan Robinson"]


def test_quarantine_is_a_noop_without_keys(tmp_path):
    out_path = tmp_path / "props_priced_clean.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    _write_priced(out_path, ["Bijan Robinson"])

    status = quarantine_final_priced_props(
        out_path=out_path, quarantine_path=quarantine_path, status_path=tmp_path / "status.json"
    )

    assert status["rows_removed"] == 0
    assert status["rows_remaining"] == 1


def test_quarantine_requires_existing_output(tmp_path):
    with pytest.raises(RuntimeError, match="missing/empty"):
        quarantine_final_priced_props(
            out_path=tmp_path / "missing.csv",
            quarantine_path=tmp_path / "quarantine.csv",
            status_path=tmp_path / "status.json",
        )


def test_tracked_manual_final_board_quarantine_file_is_well_formed():
    from pathlib import Path

    path = Path("data/manual_final_board_quarantine.csv")
    keys = _load_quarantine_keys(path)
    assert keys == {"michaelpenix", "tuatagovailoa", "cooperrush"}
