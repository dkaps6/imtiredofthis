from __future__ import annotations

import os

import pandas as pd
import pytest

from scripts.backtest.attach_qb_synthesis_game_identity_v1 import _read  # noqa: F401 -- smoke import

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _subprocess_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _trace_row(**overrides):
    row = {
        "season": 2024, "week": 1, "team": "NYJ", "opponent": "SF",
        "player_clean_key": "aaronrodgers", "actual_pass_yards": 210.0,
        "base_proj": 200.0, "football_synthesis": 205.0, "market_assisted": 208.0,
        "mc_proj": 198.0, "ml_proj": 202.0, "state_proj": 200.0,
    }
    row.update(overrides)
    return row


def _schedule_row(**overrides):
    row = {"season": 2024, "week": 1, "team": "NYJ", "opponent": "SF", "game_id": "2024_01_NYJ_SF"}
    row.update(overrides)
    return row


def test_attach_resolves_correct_game_id(tmp_path):
    import subprocess
    import sys

    trace = pd.DataFrame([_trace_row()])
    schedule = pd.DataFrame([_schedule_row()])
    trace_path = tmp_path / "trace.csv"
    schedule_path = tmp_path / "schedule.csv"
    out_path = tmp_path / "out.csv"
    trace.to_csv(trace_path, index=False)
    schedule.to_csv(schedule_path, index=False)

    subprocess.run(
        [sys.executable, "scripts/backtest/attach_qb_synthesis_game_identity_v1.py",
         "--trace", str(trace_path), "--schedule", str(schedule_path), "--out", str(out_path)],
        check=True, cwd=".", env=_subprocess_env(),
    )

    out = pd.read_csv(out_path)
    assert out.loc[0, "game_id"] == "2024_01_NYJ_SF"
    assert out.loc[0, "market"] == "pass_yards"
    assert out.loc[0, "actual"] == 210.0


def test_attach_fails_closed_on_opponent_mismatch(tmp_path):
    import subprocess
    import sys

    # Trace claims NYJ played SF in week 1, but the authoritative schedule
    # says NYJ actually played a different opponent that week -- exactly the
    # silent wrong-game join that corrupted the original benchmark.
    trace = pd.DataFrame([_trace_row(opponent="BUF")])
    schedule = pd.DataFrame([_schedule_row()])
    trace_path = tmp_path / "trace.csv"
    schedule_path = tmp_path / "schedule.csv"
    out_path = tmp_path / "out.csv"
    trace.to_csv(trace_path, index=False)
    schedule.to_csv(schedule_path, index=False)

    result = subprocess.run(
        [sys.executable, "scripts/backtest/attach_qb_synthesis_game_identity_v1.py",
         "--trace", str(trace_path), "--schedule", str(schedule_path), "--out", str(out_path)],
        cwd=".", capture_output=True, text=True, env=_subprocess_env(),
    )
    assert result.returncode != 0
    assert "opponent mismatch" in result.stderr.lower()
    assert not out_path.exists()
