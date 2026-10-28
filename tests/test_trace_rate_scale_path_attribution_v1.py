"""Guards for the rate-scale path-attribution trace.

The trace's value is that it retracts a merge proposal, so its own retraction
must stay falsifiable: if the structural facts it rests on change, it must refuse
to run rather than keep reporting a stale disposition.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research import trace_rate_scale_path_attribution_v1 as T


def test_contract_now_refuses_because_the_root_cause_IS_FIXED():
    """PERCENTY_COLS was repaired, so this trace must refuse to run.

    It documented a live defect. Once the defect is gone the trace's effect sizes
    describe a repaired build, and continuing to report them as current would be
    stale. The guard firing here is the fix landing, not a regression.
    """
    with pytest.raises(RuntimeError, match="has been fixed"):
        T.assert_source_contract()


def test_path_attribution_facts_still_hold_after_the_fix():
    """The path finding is independent of the scale fix and must survive it."""
    sim = Path("scripts/modeling/simulation_rules.py").read_text(encoding="utf-8")
    cb1 = Path("scripts/modeling/context_bridge.py").read_text(encoding="utf-8")
    assert "from scripts.modeling.context_bridge import load_model_contexts" in sim
    assert "_, players = load_model_contexts()" in sim
    assert "team_context_v3" not in cb1


def test_contract_fails_closed_if_rules_stage_switches_bridge(monkeypatch):
    real = Path.read_text

    def fake(self, *a, **k):
        text = real(self, *a, **k)
        if self.as_posix().endswith("scripts/modeling/simulation_rules.py"):
            return text.replace(
                "from scripts.modeling.context_bridge import load_model_contexts",
                "from scripts.modeling.context_bridge_v3 import load_model_contexts",
            )
        return text

    monkeypatch.setattr(Path, "read_text", fake)
    with pytest.raises(RuntimeError, match="no longer imports load_model_contexts"):
        T.assert_source_contract()


def test_contract_fails_closed_if_teams_dict_stops_being_discarded(monkeypatch):
    real = Path.read_text

    def fake(self, *a, **k):
        text = real(self, *a, **k)
        if self.as_posix().endswith("scripts/modeling/simulation_rules.py"):
            return text.replace("_, players = load_model_contexts()", "teams, players = load_model_contexts()")
        return text

    monkeypatch.setattr(Path, "read_text", fake)
    with pytest.raises(RuntimeError, match="no longer discards the teams dict"):
        T.assert_source_contract()


def test_contract_fails_closed_once_percenty_cols_is_repaired(monkeypatch):
    """After the real fix lands, these effect sizes describe a repaired build."""
    real = Path.read_text

    def fake(self, *a, **k):
        text = real(self, *a, **k)
        if self.as_posix().endswith("scripts/make_team_form.py"):
            return text.replace(
                'PERCENTY_COLS = {\n    "light_box_rate"',
                'PERCENTY_COLS = {\n    "coverage_man_rate", "coverage_zone_rate",\n'
                '    "middle_open_rate", "middle_closed_rate",\n    "light_box_rate"',
            )
        return text

    monkeypatch.setattr(Path, "read_text", fake)
    with pytest.raises(RuntimeError, match="has been fixed"):
        T.assert_source_contract()


def test_fix_scale_mirrors_the_repo_rule():
    """/100 iff max > 1.0, per make_team_form._prep_team_key_and_rates."""
    df = pd.DataFrame({
        "team": ["AAA", "BBB"],
        "middle_open_rate": [24.0, 64.6],      # percent -> rescaled
        "coverage_man_rate": [0.05, 0.41],     # already fraction -> untouched
        "light_box_rate": [25.3, 60.6],        # not in scope -> untouched
    })
    out = T._fix_scale(df)
    assert out["middle_open_rate"].tolist() == pytest.approx([0.24, 0.646])
    assert out["coverage_man_rate"].tolist() == pytest.approx([0.05, 0.41])
    assert out["light_box_rate"].tolist() == pytest.approx([25.3, 60.6])


def test_fix_scale_tolerates_missing_and_all_nan_columns():
    df = pd.DataFrame({"team": ["AAA"], "middle_open_rate": [np.nan]})
    out = T._fix_scale(df)
    assert out["middle_open_rate"].isna().all()
    assert "coverage_zone_rate" not in out.columns


def test_path_attribution_reports_guard_off_the_rules_path():
    out = T.path_attribution()
    assert out["guard_on_rules_path"] is False
    assert out["guard_installed_in"] == "scripts/team_context_v3.py"
    assert "team_form.csv" in out["rules_stage_reads"]
    assert "team_context_v3.csv" not in out["rules_stage_reads"]


def test_one_variable_proof_does_not_call_all_nan_columns_unchanged():
    """An all-NaN comparison is UNMEASURED, never UNCHANGED."""
    a = pd.DataFrame({
        "rules_plays_est": [60.0, 61.0], "rules_pass_rate": [0.57, 0.57],
        "rules_rush_share": [0.2, 0.3], "rules_ypt": [np.nan, np.nan], "rules_ypc": [np.nan, np.nan],
    })
    out = T.one_variable_proof(a, a.copy())
    assert out["rules_plays_est"]["verdict"] == "UNCHANGED"
    assert out["rules_ypt"]["verdict"] == "UNMEASURED_ALL_NAN"
    assert out["rules_ypc"]["verdict"] == "UNMEASURED_ALL_NAN"


def test_one_variable_proof_flags_a_moved_invariant():
    a = pd.DataFrame({
        "rules_plays_est": [60.0], "rules_pass_rate": [0.57], "rules_rush_share": [0.2],
        "rules_ypt": [np.nan], "rules_ypc": [np.nan],
    })
    b = a.copy()
    b.loc[0, "rules_plays_est"] = 64.5
    assert T.one_variable_proof(a, b)["rules_plays_est"]["verdict"] == "CHANGED"


def test_effect_size_negative_control_isolates_unaffected_teams():
    a = pd.DataFrame({
        "position": ["TE", "TE", "WR"], "team": ["AAA", "BBB", "BBB"],
        "rules_role": ["", "", "WR1"], "player_clean_key": ["p1", "p2", "p3"],
        "rules_tgt_share": [0.110, 0.200, 0.300],
    })
    b = a.copy()
    b.loc[0, "rules_tgt_share"] = 0.110 / 1.10     # only AAA's TE moves
    out = T.effect_size(a, b, slate={"AAA", "BBB"})
    assert out["rows_moved"] == 1
    assert out["positions_moved"] == ["TE"]
    assert out["distinct_relative_multipliers"] == [pytest.approx(1 / 1.10, abs=1e-6)]
    assert out["negative_control_unaffected_teams"] == 1
    assert out["negative_control_max_abs_delta"] == 0.0


def test_coverage_fallback_reach_reports_whether_false_fires_can_land():
    tf = pd.DataFrame({"team": ["AAA", "BBB", "CCC"]})
    cov = pd.DataFrame({"team": ["AAA"]})
    out = T.coverage_fallback_reach(tf, cov, slate={"AAA"})
    assert out["fallback_teams"] == ["BBB", "CCC"]
    assert out["fallback_teams_on_slate"] == []
    assert out["man_gate_false_fires_reach_slate"] is False

    out2 = T.coverage_fallback_reach(tf, cov, slate={"AAA", "BBB"})
    assert out2["fallback_teams_on_slate"] == ["BBB"]
    assert out2["man_gate_false_fires_reach_slate"] is True


def test_main_refuses_a_non_te_effect(monkeypatch):
    monkeypatch.setattr(T, "assert_source_contract", lambda: {})
    """The TE-only claim must be an assertion, not a description of one run."""
    monkeypatch.setattr(T, "effect_size", lambda *a, **k: {
        "rows_moved": 5, "rows_total": 100, "positions_moved": ["TE", "WR"],
        "distinct_relative_multipliers": [0.909091], "teams_moved": 1, "players_moved": 1,
        "mean_delta_among_moved": -0.01, "by_position": {},
        "negative_control_unaffected_teams": 0, "negative_control_max_abs_delta": 0.0,
        "slate_teams": 28,
    })
    with pytest.raises(RuntimeError, match="TE-only"):
        T.main()


def test_main_refuses_an_unexpected_multiplier(monkeypatch):
    monkeypatch.setattr(T, "assert_source_contract", lambda: {})
    monkeypatch.setattr(T, "effect_size", lambda *a, **k: {
        "rows_moved": 5, "rows_total": 100, "positions_moved": ["TE"],
        "distinct_relative_multipliers": [0.75], "teams_moved": 1, "players_moved": 1,
        "mean_delta_among_moved": -0.01, "by_position": {},
        "negative_control_unaffected_teams": 0, "negative_control_max_abs_delta": 0.0,
        "slate_teams": 28,
    })
    with pytest.raises(RuntimeError, match="exactly one relative multiplier"):
        T.main()
