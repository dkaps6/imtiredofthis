from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research import assemble_rb_pd2_forward_locks_v1 as assembler
from scripts.research import build_rb_pd2_forward_history_v1 as history_builder
from scripts.research import rb_pd2_shadow_capture_v1 as capture
from scripts.research import rb_pd2_forward_shadow_v1 as fwd
from scripts.research import run_rb_pd2_live_lock_v1 as live_lock


def test_2025_history_builder_uses_pregame_components_only():
    weights = pd.DataFrame([{
        "market": "rush_yards",
        "mc_weight": fwd.RUSH_YARDS_MC_WEIGHT,
        "ml_weight": fwd.RUSH_YARDS_ML_WEIGHT,
        "state_weight": 0.0,
        "fit_scope": fwd.RUSH_YARDS_FIT_SCOPE,
        "promotion_lineage": fwd.RUSH_YARDS_PROMOTION_LINEAGE,
    }])
    rows = []
    for week in range(1, 7):
        rows.append({
            "season": 2025, "week": week, "team": "CHI", "opponent": "GB",
            "event_id": f"g{week}", "player": "Alpha Back",
            "player_clean_key": "alphaback", "position": "RB",
            "market": "rush_yards", "mc_proj": 50.0, "ml_proj": 40.0,
            "state_proj": 999.0, "actual": 45.0,
            "prediction_cutoff": f"2025-W{week:02d} pregame", "prior_season": 2024,
        })
    out = history_builder.build_2025_seed(pd.DataFrame(rows), weights)
    expected = (
        fwd.RUSH_YARDS_MC_WEIGHT * 50.0
        + fwd.RUSH_YARDS_ML_WEIGHT * 40.0
    )
    assert out["projection_mean"].eq(expected).all()
    assert out["pregame_lineage_certified"].all()
    assert out["actual_rush_yards"].eq(45.0).all()


def test_2025_history_builder_rejects_nonpregame_lineage():
    weights = pd.DataFrame([{
        "market": "rush_yards",
        "mc_weight": fwd.RUSH_YARDS_MC_WEIGHT,
        "ml_weight": fwd.RUSH_YARDS_ML_WEIGHT,
        "state_weight": 0.0,
        "fit_scope": fwd.RUSH_YARDS_FIT_SCOPE,
        "promotion_lineage": fwd.RUSH_YARDS_PROMOTION_LINEAGE,
    }])
    row = pd.DataFrame([{
        "season": 2025, "week": 1, "team": "CHI", "player_clean_key": "x",
        "position": "RB", "market": "rush_yards", "mc_proj": 1.0,
        "ml_proj": 1.0, "state_proj": 1.0, "actual": 1.0,
        "prediction_cutoff": "postgame", "prior_season": 2024,
    }])
    with pytest.raises(RuntimeError, match="pregame"):
        history_builder.build_2025_seed(row, weights)


def test_week1_2026_additional_history_recomputes_p3_stack1_parity():
    row = pd.DataFrame([{
        "season": 2026, "week": 1, "team": "CHI", "player_clean_key": "x",
        "position": "RB", "projection_mean": 50.0, "actual_rush_yards": 55.0,
        "pregame_lineage_certified": True, "projection_lineage": "P3",
    }])
    with pytest.raises(RuntimeError, match="parity"):
        history_builder.validate_additional_history(row)

    row["week1_p3_stack1_parity_pass"] = True
    with pytest.raises(RuntimeError, match="mechanical"):
        history_builder.validate_additional_history(row)

    row["week1_p3_projection"] = 50.0
    row["week1_stack1_projection"] = 50.0
    out = history_builder.validate_additional_history(row)
    assert len(out) == 1

    drifted = row.copy()
    drifted["week1_stack1_projection"] = 49.0
    with pytest.raises(RuntimeError, match="recomputation failed"):
        history_builder.validate_additional_history(drifted)

    false_string = row.copy()
    false_string["week1_p3_stack1_parity_pass"] = "False"
    with pytest.raises(RuntimeError, match="audit flag"):
        history_builder.validate_additional_history(false_string)


def _make_capture_session(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(capture, "DEFAULT_ROOT", tmp_path / "capture")
    monkeypatch.setenv("GITHUB_SHA", "d" * 40)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_JOB", "full-slate")

    # Make fingerprint provenance deterministic without depending on repo files.
    monkeypatch.setattr(capture, "_provenance", lambda: {
        "code_sha": "d" * 40,
        "workflow": "Full Slate",
        "workflow_run_id": "123",
        "workflow_run_attempt": "1",
        "workflow_job": "full-slate",
        "runner_ref": "refs/heads/test",
        "manual_name_overrides_sha256": "a" * 64,
        "roles_ourlads_sha256": "b" * 64,
    })
    capture.reset()
    capture.begin_session(season=2026)
    row = pd.Series({
        "event_id": "event-car-atl", "team": "CAR", "opponent": "ATL",
        "player": "Alpha Back", "player_clean_key": "back0",
    })
    draws = np.linspace(0.0, 120.0, 128)
    capture.note_expected(
        row=row, market="rush_yards", position="RB", season=2026, week=2
    )
    assert capture.capture(
        row=row, adjusted_outcomes=draws, target_mean=float(draws.mean()),
        mc_proj=float(draws.mean()), market="rush_yards", position="RB",
        season=2026, week=2,
    )
    # Force a deterministic prospective capture time for the fixture.
    key = next(iter(capture._SESSION["records"]))
    capture._SESSION["records"][key]["captured_at_utc"] = "2026-09-20T15:00:00Z"
    receipt = capture.finalize(expected_football_keys=capture.noted_expected_keys())
    capture.reset()
    return Path(receipt["dir"]), draws


def _history_for_lock():
    rows = []
    for week in range(1, 11):
        for p in range(20):
            rows.append({
                "season": 2025, "week": week,
                "team": "CAR" if p % 2 == 0 else "ATL",
                "player_clean_key": f"back{p}", "position": "RB",
                "projection_mean": 50.0 + p,
                "actual_rush_yards": 45.0 + (week % 3) + p,
                "pregame_lineage_certified": True,
            })
    return fwd.build_history_state(pd.DataFrame(rows))


def test_lock_assembler_writes_self_contained_baseline_and_candidate_arrays(tmp_path, monkeypatch):
    session, _draws = _make_capture_session(tmp_path, monkeypatch)
    history = _history_for_lock()
    history_path = tmp_path / "history.csv"
    history.to_csv(history_path, index=False)

    manifest = {
        "history_state_sha256": fwd.history_state_digest(history),
        "manual_name_overrides_sha256": "a" * 64,
        "roles_ourlads_sha256": "b" * 64,
    }
    manifest_path = tmp_path / "history.json"
    manifest_path.write_text(json.dumps(manifest))

    schedule = pd.DataFrame([{
        "season": 2026, "week": 2, "home": "ATL", "away": "CAR",
        "kickoff_utc": "2026-09-20T17:00:00Z",
    }])
    schedule_path = tmp_path / "schedule.csv"
    schedule.to_csv(schedule_path, index=False)

    out_dir = tmp_path / "locks"
    result = assembler.assemble(
        session_dir=session,
        history_state_path=history_path,
        history_manifest_path=manifest_path,
        prospective_start_utc="2026-09-20T14:00:00Z",
        out_dir=out_dir,
        schedule_csv=schedule_path,
        lock_timestamp_utc="2026-09-20T15:05:00Z",
    )
    assert result["valid"] is True
    assert result["locked_rows"] == 1
    rows = [
        json.loads(line)
        for line in (out_dir / assembler.LOCK_MANIFEST).read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    with np.load(out_dir / assembler.LOCK_ARRAYS, allow_pickle=False) as npz:
        base = npz[rows[0]["baseline_array_key"]]
        cand = npz[rows[0]["candidate_array_key"]]
    assert len(base) == len(cand)
    assert float(np.mean(base)) == pytest.approx(float(np.mean(cand)), abs=1e-8)
    assert rows[0]["outcome_present_at_lock"] is False



def test_live_lock_selects_exact_current_capture_session(tmp_path):
    root = tmp_path / "capture"
    for sid, run_id, sha in [
        ("a", "111", "a" * 40),
        ("b", "222", "b" * 40),
    ]:
        d = root / sid
        d.mkdir(parents=True)
        (d / "session_receipt.json").write_text(json.dumps({
            "session_id": sid,
            "provenance": {"workflow_run_id": run_id, "code_sha": sha},
        }))

    got = live_lock.find_capture_session(
        root, workflow_run_id="222", code_sha="b" * 40
    )
    assert got == root / "b"

    with pytest.raises(RuntimeError, match="found=0"):
        live_lock.find_capture_session(
            root, workflow_run_id="333", code_sha="c" * 40
        )


def test_live_lock_rejects_ambiguous_same_run_sessions(tmp_path):
    root = tmp_path / "capture"
    for sid in ["a", "b"]:
        d = root / sid
        d.mkdir(parents=True)
        (d / "session_receipt.json").write_text(json.dumps({
            "session_id": sid,
            "provenance": {"workflow_run_id": "123", "code_sha": "d" * 40},
        }))
    with pytest.raises(RuntimeError, match="found=2"):
        live_lock.find_capture_session(
            root, workflow_run_id="123", code_sha="d" * 40
        )
