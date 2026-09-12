from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _subprocess_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _trace_row(**overrides):
    row = {
        "season": 2024, "week": 1, "team": "NYJ", "opponent": "SF",
        "player_clean_key": "aaronrodgers", "game_id": "2024_01_NYJ_SF", "market": "pass_yards",
        "actual": 210.0, "base_proj": 200.0, "football_synthesis": 205.0, "market_assisted": 208.0,
        "mc_proj": 198.0, "ml_proj": 202.0, "state_proj": 200.0,
    }
    row.update(overrides)
    return row


def _props_row(**overrides):
    # prepare_free_qb_prop_archive.py's real output columns -- deliberately
    # no "market" column, since that archive is single-market by
    # construction. This exact shape crashed grade()'s select_one_book_row()
    # with KeyError: 'market' before the fix.
    row = {
        "game_id": "2024_01_NYJ_SF", "player_clean_key": "aaronrodgers", "book": "draftkings",
        "line": 220.5, "over_odds": -110, "under_odds": -110, "player": "a.rodgers",
        "season": 2024, "week": 1,
    }
    row.update(overrides)
    return row


def test_grade_script_handles_marketless_qb_props(tmp_path):
    trace = pd.DataFrame([_trace_row()])
    props = pd.DataFrame([_props_row()])
    assert "market" not in props.columns

    trace_path = tmp_path / "trace.csv"
    props_path = tmp_path / "props.csv"
    out_dir = tmp_path / "grade"
    trace.to_csv(trace_path, index=False)
    props.to_csv(props_path, index=False)

    result = subprocess.run(
        [sys.executable, "scripts/backtest/grade_qb_synthesis_vegas_v1.py",
         "--trace", str(trace_path), "--props", str(props_path), "--out-dir", str(out_dir)],
        cwd=_REPO_ROOT, capture_output=True, text=True, env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "qb_base_summary.csv").exists()
    assert (out_dir / "qb_synthesis_summary.csv").exists()
    assert (out_dir / "qb_market_assisted_summary.csv").exists()

    synthesis = pd.read_csv(out_dir / "qb_synthesis_summary.csv")
    assert not synthesis.empty
