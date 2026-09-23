"""Tests for postgame settlement/provenance used by the full-board replay."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.operations import backtest_full_report_v1 as BF
from scripts.operations import grade_market_track_record_gsis_v1 as GG
from scripts.operations import grade_market_track_record_v1 as G
from scripts.operations import report_weekly_backtest_v1 as RW
from scripts.operations.grade_market_track_record_gsis_v1 import apply_postgame_settlement
from scripts.operations.backtest_full_report_v1 import _cell
from scripts.operations.report_weekly_backtest_v1 import _row_stats


def _rows():
    return pd.DataFrame(
        [
            {
                "identity_status": "RESOLVED_GSIS",
                "roster_confirmed_this_team_week": True,
                "roster_status": "ACT",
                "snap_participated": True,
                "book": "draftkings",
                "actual": 12.0,
            },
            {
                "identity_status": "RESOLVED_GSIS",
                "roster_confirmed_this_team_week": True,
                "roster_status": "ACT",
                "snap_participated": True,
                "book": "fanduel",
                "actual": np.nan,
            },
            {
                "identity_status": "RESOLVED_GSIS",
                "roster_confirmed_this_team_week": True,
                "roster_status": "INA",
                "snap_participated": False,
                "book": "draftkings",
                "actual": np.nan,
            },
            {
                "identity_status": "RESOLVED_GSIS",
                "roster_confirmed_this_team_week": True,
                "roster_status": "ACT",
                "snap_participated": False,
                "book": "draftkings",
                "actual": np.nan,
            },
            {
                "identity_status": "UNRESOLVED_IDENTITY",
                "roster_confirmed_this_team_week": False,
                "roster_status": "",
                "snap_participated": False,
                "book": "draftkings",
                "actual": np.nan,
            },
        ]
    )


def _all_pass_board() -> pd.DataFrame:
    common = {
        "season": 2026,
        "week": 1,
        "event_id": "e1",
        "player": "Example Player",
        "player_clean_key": "exampleplayer",
        "team": "KC",
        "opponent": "BUF",
        "market": "pass_yards",
        "source_market": "player_pass_yds",
        "book": "draftkings",
        "book_title": "DraftKings",
        "vegas_line": 250.5,
        "model_proj": 250.0,
        "source_run_id": "34650067599",
        "source_git_sha": "be061eaf23372f080db3911d3b4919120c744c53",
    }
    return pd.DataFrame(
        [
            {
                **common,
                "side": "OVER",
                "fair_prob": 0.50,
                "vegas_odds": -110.0,
                "edge_pct": 0.0,
            },
            {
                **common,
                "side": "UNDER",
                "fair_prob": 0.50,
                "vegas_odds": -110.0,
                "edge_pct": 0.0,
            },
        ]
    )


def test_postgame_settlement_distinguishes_stats_zero_dnp_and_unknown():
    out = apply_postgame_settlement(_rows())

    assert out.loc[0, "actual_source"] == "stats_table"
    assert out.loc[0, "settlement_status"] == "SETTLED"
    assert out.loc[0, "actual"] == 12.0

    assert out.loc[1, "actual_source"] == "snap_confirmed_verified_zero"
    assert out.loc[1, "settlement_status"] == "SETTLED"
    assert out.loc[1, "actual"] == 0.0

    assert out.loc[2, "actual_source"] == "sportsbook_void_dnp"
    assert out.loc[2, "settlement_status"] == "VOID"
    assert pd.isna(out.loc[2, "actual"])

    assert out.loc[3, "actual_source"] == "unresolved"
    assert out.loc[3, "settlement_status"] == "UNRESOLVED"
    assert pd.isna(out.loc[3, "actual"])

    assert out.loc[4, "actual_source"] == "unresolved"
    assert out.loc[4, "settlement_status"] == "UNRESOLVED"


def test_dnp_void_rule_is_fail_closed_for_unknown_book():
    row = _rows().iloc[[2]].copy()
    row["book"] = "unknownbook"
    out = apply_postgame_settlement(row)
    assert out.iloc[0]["settlement_status"] == "UNRESOLVED"


def test_accuracy_metrics_exclude_void_rows_from_denominator():
    rows = pd.DataFrame(
        [
            {
                "bet_result": "WIN",
                "unit_result": 1.0,
                "model_error": 1.0,
                "vegas_error": 2.0,
                "model_closer": True,
                "model_closer_than_vegas": True,
            },
            {
                "bet_result": "LOSS",
                "unit_result": -1.0,
                "model_error": 3.0,
                "vegas_error": 2.0,
                "model_closer": False,
                "model_closer_than_vegas": False,
            },
            {
                "bet_result": "VOID",
                "unit_result": 0.0,
                "model_error": np.nan,
                "vegas_error": np.nan,
                "model_closer": False,
                "model_closer_than_vegas": False,
            },
        ]
    )

    full = _cell(rows)
    weekly = _row_stats(rows)

    assert full["bets"] == weekly["bets"] == 2
    assert full["closer"] == weekly["closer"] == 0.5
    assert full["m_mae"] == weekly["model_mae"] == 2.0
    assert full["v_mae"] == weekly["vegas_mae"] == 2.0


def test_dnp_void_uses_row_level_book_title_fallback():
    row = _rows().iloc[[2]].copy()
    row["book"] = ""
    row["book_title"] = "DraftKings"
    out = apply_postgame_settlement(row)
    assert out.iloc[0]["actual_source"] == "sportsbook_void_dnp"
    assert out.iloc[0]["settlement_status"] == "VOID"


def test_gsis_grader_all_pass_returns_structured_zero_and_empty_detail(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(GG, "load_boards", lambda season, weeks: _all_pass_board())
    detail_path = tmp_path / "detail.csv"
    summary = GG.grade(2026, [1], detail_out=detail_path)

    assert summary["selection_status"] == "all_pass_zero_selected_bets"
    assert summary["selected_settlement_rows"] == 0
    assert summary["decided_bets"] == 0
    assert summary["units"] == 0.0
    assert detail_path.exists()
    detail = pd.read_csv(detail_path)
    assert detail.empty
    assert {"gsis_id", "settlement_status", "bet_result", "unit_result"} <= set(
        detail.columns
    )


def test_full_report_all_pass_returns_downstream_safe_empty_frame(monkeypatch):
    monkeypatch.setattr(G, "load_boards", lambda season, weeks: _all_pass_board())
    graded = BF.build_graded(2026, [1])

    assert graded.empty
    assert {"gsis_id", "settlement_status", "bet_result", "unit_result"} <= set(
        graded.columns
    )


def test_weekly_report_all_pass_header_only_detail_exits_cleanly(tmp_path: Path):
    detail_path = tmp_path / "detail.csv"
    empty = GG.empty_graded_frame(_all_pass_board().iloc[0:0])
    empty.to_csv(detail_path, index=False)

    assert RW.report(detail_path) == 0
