"""A union-by-key ledger cannot express a removal, so it must be told about one.

2026 Week 9: run 42102376747 archived 1,544 rows before the MIN starter
conflict was understood. The certified re-run archived 1,538 with Carson Wentz
withheld -- and the ledger came back to 1,544, republishing the row
certification had explicitly refused. A reader of the durable board could not
tell it from a certified row.
"""
from __future__ import annotations

import pandas as pd
import pytest

import scripts.operations.archive_priced_board_v1 as mod


def _board(players: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": f"E{i}", "team": team, "player": player,
                "market": "pass_yards", "side": "OVER", "book": "draftkings",
                "vegas_line": 200.5 + i, "model_proj": 210.0 + i, "fair_prob": 0.5,
                "mc_proj": 208.0 + i, "ensemble_proj": 209.0 + i,
                "ensemble_status": "calibrated", "edge_pct": 1.5,
            }
            for i, (team, player) in enumerate(players)
        ]
    )


@pytest.fixture()
def staged(tmp_path, monkeypatch):
    ledger_dir = tmp_path / "boards"
    monkeypatch.setattr(mod, "LEDGER_DIR", ledger_dir)
    monkeypatch.setattr(mod, "_ledger_path", lambda s, w: ledger_dir / f"{s}_wk{w:02d}.csv")
    qpath = tmp_path / "manual_final_board_quarantine.csv"
    pd.DataFrame([{
        "player": "Carson Wentz", "team": "MIN", "season": 2026, "week": 9,
        "reason": "starter unresolved", "verified_source": "test", "verified_date": "2026-11-05",
    }]).to_csv(qpath, index=False)
    monkeypatch.setattr(
        mod, "_load_quarantine_keys",
        lambda season=None, week=None: __import__(
            "scripts.operations.quarantine_final_priced_props_v1", fromlist=["x"]
        )._load_quarantine_keys(qpath, season=season, week=week),
    )
    return ledger_dir


def _write(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_an_earlier_captures_quarantined_row_does_not_survive_the_merge(staged, tmp_path):
    # An earlier, pre-quarantine capture is already in the ledger.
    _write(staged / "2026_wk09.csv", _board([("MIN", "Carson Wentz"), ("BAL", "Lamar Jackson")]))
    # The certified capture simply has no Wentz row to replace it with.
    priced = tmp_path / "props_priced_clean.csv"
    _board([("BAL", "Lamar Jackson")]).to_csv(priced, index=False)

    res = mod.archive(priced_path=priced, season=2026, week=9, run_id="t", git_sha="t")
    out = pd.read_csv(staged / "2026_wk09.csv")
    assert res["final_board_quarantined_rows_dropped"] == 1
    assert "Carson Wentz" not in set(out["player"]), "the withheld row was republished"
    assert "Lamar Jackson" in set(out["player"]), "a certified row must survive"


def test_a_week_without_a_quarantine_is_untouched(staged, tmp_path):
    _write(staged / "2026_wk08.csv", _board([("MIN", "Carson Wentz")]))
    priced = tmp_path / "props_priced_clean.csv"
    _board([("BAL", "Lamar Jackson")]).to_csv(priced, index=False)
    res = mod.archive(priced_path=priced, season=2026, week=8, run_id="t", git_sha="t")
    out = pd.read_csv(staged / "2026_wk08.csv")
    assert res["final_board_quarantined_rows_dropped"] == 0
    # The quarantine row is scoped to week 9, so week 8 keeps both.
    assert {"Carson Wentz", "Lamar Jackson"} <= set(out["player"])
