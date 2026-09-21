from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from scripts.research import rb_pd2_forward_shadow_v1 as fwd
from scripts.research.evaluate_rb_pd2_yard_difficulty_mc_width_v1 import (
    strict_prior_difficulty_scores as historical_strict_scores,
)


def _history(n_players=20, weeks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)):
    rows = []
    for week in weeks:
        for p in range(n_players):
            rows.append({
                "season": 2025,
                "week": week,
                "team": "CHI" if p % 2 == 0 else "CAR",
                "player_clean_key": f"back{p}",
                "position": "RB",
                "projection_mean": 40.0 + p + week,
                "actual_rush_yards": 35.0 + p + (week % 3),
                "pregame_lineage_certified": True,
            })
    return pd.DataFrame(rows)


def test_frozen_2025_weights_match_repository_contract():
    weights = pd.read_csv("data/model_ensemble_weights.csv")
    checks = fwd.verify_frozen_2025_weights(weights)
    assert all(checks.values())


def test_history_state_is_strict_prior_last8_and_cross_season():
    rows = []
    for week in range(11, 19):
        rows.append({
            "season": 2025, "week": week, "team": "CHI",
            "player_clean_key": "alpha", "position": "RB",
            "projection_mean": 50.0, "actual_rush_yards": float(week),
            "pregame_lineage_certified": True,
        })
    rows.append({
        "season": 2026, "week": 1, "team": "CHI",
        "player_clean_key": "alpha", "position": "RB",
        "projection_mean": 60.0, "actual_rush_yards": 55.0,
        "pregame_lineage_certified": True,
    })
    state = fwd.build_history_state(pd.DataFrame(rows))
    target = state.loc[state["season"].eq(2026) & state["week"].eq(1)].iloc[0]
    assert target["prior_games"] == 8
    expected = np.mean([abs(50.0 - float(w)) for w in range(11, 19)])
    assert target["prior8_yard_mae"] == pytest.approx(expected)
    assert int(target["last_prior_ord"]) == 202518
    assert int(target["last_prior_ord"]) < int(target["target_ord"])


def test_uncertified_history_fails_closed():
    h = _history(n_players=1, weeks=(1,))
    h.loc[0, "pregame_lineage_certified"] = False
    with pytest.raises(RuntimeError, match="uncertified"):
        fwd.build_history_state(h)


def test_same_week_reference_semantics_match_frozen_historical_helper():
    state = fwd.build_history_state(_history())
    legacy = state.rename(columns={"player_clean_key": "player_key"}).copy()
    legacy = legacy.drop(columns=["difficulty_score", "difficulty_reference_n", "difficulty_reference_max_ord"])
    expected = historical_strict_scores(legacy)
    got = state.sort_values(["season", "week", "player_clean_key"]).reset_index(drop=True)
    expected = expected.sort_values(["season", "week", "player_key"]).reset_index(drop=True)

    np.testing.assert_allclose(
        got["difficulty_reference_n"].to_numpy(float),
        expected["difficulty_reference_n"].to_numpy(float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        got["difficulty_reference_max_ord"].to_numpy(float),
        expected["difficulty_reference_max_ord"].to_numpy(float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        got["difficulty_score"].to_numpy(float),
        expected["difficulty_score"].to_numpy(float),
        equal_nan=True,
    )


def test_target_state_excludes_same_week_and_future_rows():
    state = fwd.build_history_state(_history())
    t = fwd.target_difficulty_state(
        state, player_clean_key="back0", season=2025, week=10
    )
    assert t["prior_games"] == 8
    assert t["difficulty_reference_max_ord"] < 202510


def test_width_transform_is_exactly_mean_neutral_and_nonnegative():
    x = np.array([0.0, 0.0, 10.0, 20.0, 90.0], dtype=np.float64)
    y, mult = fwd.build_candidate(x, 1.0)
    assert mult == pytest.approx(1.30)
    assert len(y) == len(x)
    assert np.isfinite(y).all()
    assert (y >= 0).all()
    assert float(np.mean(y)) == pytest.approx(float(np.mean(x)), abs=1e-8)


def test_unscored_difficulty_is_identity_transform():
    x = np.linspace(0.0, 100.0, 101)
    y, mult = fwd.build_candidate(x, np.nan)
    assert mult == 1.0
    np.testing.assert_array_equal(y, x)


def test_empty_or_mismatched_identity_fingerprints_fail_closed():
    good = "a" * 64
    hist = {
        "manual_name_overrides_sha256": good,
        "roles_ourlads_sha256": good,
    }
    receipt = {"provenance": {
        "manual_name_overrides_sha256": good,
        "roles_ourlads_sha256": good,
    }}
    fwd.assert_identity_fingerprints_match(hist, receipt)

    bad_empty = {"provenance": {
        "manual_name_overrides_sha256": "",
        "roles_ourlads_sha256": good,
    }}
    with pytest.raises(RuntimeError, match="unprovable"):
        fwd.assert_identity_fingerprints_match(hist, bad_empty)

    bad_mismatch = {"provenance": {
        "manual_name_overrides_sha256": "b" * 64,
        "roles_ourlads_sha256": good,
    }}
    with pytest.raises(RuntimeError, match="mismatch"):
        fwd.assert_identity_fingerprints_match(hist, bad_mismatch)


def _schedule():
    return pd.DataFrame([
        {
            "season": 2026, "week": 2, "home": "LA", "away": "NYG",
            "kickoff_utc": "2026-09-22T00:15:00Z",
        },
        {
            "season": 2026, "week": 2, "home": "KC", "away": "IND",
            "kickoff_utc": "2026-09-21T00:20:00Z",
        },
        {
            "season": 2026, "week": 2, "home": "ATL", "away": "CAR",
            "kickoff_utc": "2026-09-20T17:00:00Z",
        },
    ])


def test_schedule_resolver_canonicalizes_la_to_lar_and_handles_utc_day_rollover():
    kickoff = fwd.resolve_kickoff_utc(_schedule(), season=2026, week=2, team="LAR")
    assert kickoff == pd.Timestamp("2026-09-22T00:15:00Z")
    snf = fwd.resolve_kickoff_utc(_schedule(), season=2026, week=2, team="IND")
    assert snf == pd.Timestamp("2026-09-21T00:20:00Z")


def test_date_only_or_ambiguous_schedule_fails_closed():
    date_only = pd.DataFrame([{
        "season": 2026, "week": 2, "home": "CAR", "away": "ATL",
        "kickoff_utc": "2026-09-20",
    }])
    with pytest.raises(RuntimeError, match="date-only"):
        fwd.normalize_schedule(date_only)

    dup = pd.concat([_schedule(), _schedule().iloc[[2]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="exactly one"):
        fwd.resolve_kickoff_utc(dup, season=2026, week=2, team="CAR")


def _history_manifest():
    good1 = "a" * 64
    good2 = "b" * 64
    return {
        "history_state_sha256": "c" * 64,
        "manual_name_overrides_sha256": good1,
        "roles_ourlads_sha256": good2,
    }


def _capture_receipt():
    return {
        "valid": True,
        "provenance": {
            "code_sha": "d" * 40,
            "workflow_run_id": "123456",
            "workflow_run_attempt": "1",
            "workflow_job": "full-slate",
            "manual_name_overrides_sha256": "a" * 64,
            "roles_ourlads_sha256": "b" * 64,
        },
    }


def _capture_record(draws):
    return {
        "baseline_lock_eligible": True,
        "outcome_present_at_lock": False,
        "sportsbook_inputs_used_in_candidate": False,
        "production_output_mutated": False,
        "captured_at_utc": "2026-09-20T15:00:00Z",
        "season": 2026,
        "week": 2,
        "event_id": "event-car-atl",
        "team": "CAR",
        "opponent": "ATL",
        "player": "Alpha Back",
        "player_clean_key": "back0",
        "position": "RB",
        "session_id": "session-x",
        "target_mean": float(np.mean(draws)),
        "baseline": {"draw_digest_sha256": fwd.draw_digest(draws)},
    }


def test_lock_row_requires_capture_and_artifact_to_both_be_pregame():
    history = fwd.build_history_state(_history())
    draws = np.linspace(0.0, 120.0, 256, dtype=np.float64)
    rec, candidate = fwd.lock_row(
        capture_record=_capture_record(draws),
        capture_receipt=_capture_receipt(),
        baseline_draws=draws,
        history_state=history,
        history_manifest=_history_manifest(),
        schedule=_schedule(),
        prospective_start_utc="2026-09-20T14:00:00Z",
        lock_timestamp_utc="2026-09-20T15:05:00Z",
    )
    assert rec["kickoff_utc"] == "2026-09-20T17:00:00+00:00"
    assert rec["outcome_present_at_lock"] is False
    assert rec["sportsbook_inputs_used_in_candidate"] is False
    assert rec["production_output_mutated"] is False
    assert rec["baseline"]["draw_count"] == rec["candidate"]["draw_count"]
    assert rec["baseline"]["mean"] == pytest.approx(rec["candidate"]["mean"], abs=1e-8)
    assert fwd.draw_digest(candidate) == rec["candidate"]["draw_digest_sha256"]

    with pytest.raises(RuntimeError, match="persisted pregame"):
        fwd.lock_row(
            capture_record=_capture_record(draws),
            capture_receipt=_capture_receipt(),
            baseline_draws=draws,
            history_state=history,
            history_manifest=_history_manifest(),
            schedule=_schedule(),
            prospective_start_utc="2026-09-20T14:00:00Z",
            lock_timestamp_utc="2026-09-20T18:00:00Z",
        )


def test_lock_row_rejects_rows_before_frozen_prospective_start():
    history = fwd.build_history_state(_history())
    draws = np.linspace(0.0, 120.0, 64, dtype=np.float64)
    with pytest.raises(RuntimeError, match="predates"):
        fwd.lock_row(
            capture_record=_capture_record(draws),
            capture_receipt=_capture_receipt(),
            baseline_draws=draws,
            history_state=history,
            history_manifest=_history_manifest(),
            schedule=_schedule(),
            prospective_start_utc="2026-09-20T16:00:00Z",
            lock_timestamp_utc="2026-09-20T16:05:00Z",
        )
