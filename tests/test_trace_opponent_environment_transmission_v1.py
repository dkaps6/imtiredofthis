"""Guard the invariant that makes the transmission trace's conclusion valid.

The trace concludes that a column absent from the player-path source cannot
reach a model. That is only sound while the learned components declare their
features statically. These tests pin the guard, so a future change to dynamic
feature selection fails here rather than silently invalidating the published
classification.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.research.trace_opponent_environment_transmission_v1 import (
    BOOKKEEPING,
    FAMILIES,
    _family,
    assert_no_dynamic_selection,
    orphaned_mechanisms,
    teamcontext_transmission,
)


def test_method_validity_guard_passes_on_current_source():
    checks = assert_no_dynamic_selection()
    assert checks["ml_v2_static_feature_columns"] is True
    assert checks["state_v2_static_required_set"] is True
    assert checks["no_runtime_opp_name_construction"] is True


def test_guard_fails_closed_if_feature_selection_becomes_dynamic(tmp_path,
                                                                 monkeypatch):
    """The guard must raise, not warn, if the premise stops holding."""
    src = Path("scripts/modeling/ml_v2.py").read_text(encoding="utf-8")
    broken = src.replace("FEATURE_COLUMNS = (",
                         "FEATURE_COLUMNS = list(", 1)
    work = tmp_path / "scripts" / "modeling"
    work.mkdir(parents=True)
    (work / "ml_v2.py").write_text(broken, encoding="utf-8")
    (work / "state_v2.py").write_text(
        Path("scripts/modeling/state_v2.py").read_text(encoding="utf-8"),
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="FEATURE_COLUMNS"):
        assert_no_dynamic_selection()


def test_ml_and_state_declare_no_opponent_feature():
    """The structural-zero claim, asserted against source rather than prose."""
    ml = Path("scripts/modeling/ml_v2.py").read_text(encoding="utf-8")
    block = ml.split("FEATURE_COLUMNS = (", 1)[1].split(")", 1)[0]
    for token in ("def_", "_allowed", "box_rate", "coverage_", "opp",
                  "pressure", "explosive"):
        assert token not in block, (
            f"ml_v2 FEATURE_COLUMNS now contains {token!r}; the published "
            "structural-zero finding is stale and must be revised")
    st = Path("scripts/modeling/state_v2.py").read_text(encoding="utf-8")
    required = re.search(r"required = \{([^}]*)\}", st)
    assert required is not None
    assert "opp" not in required.group(1)


def test_teamcontext_separates_carried_from_consumed():
    t = teamcontext_transmission()
    populated = set(t["teamcontext_fields_populated"])
    read = set(t["teamcontext_fields_read_by_a_rule"])
    ignored = set(t["carried_onto_teamcontext_then_ignored_by_every_rule"])
    assert read <= populated
    assert ignored == populated - read
    assert ignored, "a carried-but-ignored set of zero would contradict the trace"
    # The specific finding the document reports.
    assert "def_rush_epa" in ignored
    assert {"light_box_rate", "heavy_box_rate", "coverage_zone_rate"} <= read


def test_rush_epa_mechanism_is_orphaned_not_absent():
    o = orphaned_mechanisms()
    assert o["projection_imported_by_runner"] is False
    assert o["on_canonical_path"] is False
    assert "_fair_odds" in o["names_the_production_runner_imports"]


def test_bookkeeping_is_not_counted_as_a_missing_football_mechanism():
    for col in ("pbp_feature_source", "team_form_active_season",
                "qb_promoted_context_version", "prior_season", "games_played"):
        assert BOOKKEEPING.match(col), f"{col} should be classed bookkeeping"
    for col in ("def_rush_epa", "rush_stuff_rate_x", "ypt_allowed_rb",
                "heavy_box_rate", "explosive_play_rate_allowed"):
        assert not BOOKKEEPING.match(col), f"{col} is football signal"


@pytest.mark.parametrize("col,family", [
    ("def_rush_epa", "B_RUN_STOP_AT_THE_LINE"),
    ("rush_stuff_rate_x", "B_RUN_STOP_AT_THE_LINE"),
    ("yards_before_contact_per_rb_rush_y", "B_RUN_STOP_AT_THE_LINE"),
    ("explosive_play_rate_allowed", "C_EXPLOSIVE_RUN_SUPPRESSION"),
    ("heavy_box_rate", "A_RUN_FIT_BOX_STRUCTURE"),
    ("ypt_allowed_rb", "D_RB_RECEIVING_COVERAGE"),
    ("blitz_rate", "E_PASS_RUSH_BLITZ"),
    ("pace", "F_EXPECTED_GAME_STATE_VOLUME"),
])
def test_families_are_assigned_by_football_meaning(col, family):
    assert _family(col) == family


def test_explosive_suppression_is_not_routed_to_the_mean():
    """The channel distinction is load-bearing, so pin it.

    A defense that surrenders few explosive runs compresses the right tail.
    Routing that into the mean would mis-model the distribution, so the family
    must stay declared as a sigma channel.
    """
    c = FAMILIES["C_EXPLOSIVE_RUN_SUPPRESSION"]["channel"]
    assert "sigma" in c and "NOT the mean" in c
    b = FAMILIES["B_RUN_STOP_AT_THE_LINE"]["channel"]
    assert "central tendency" in b
    assert FAMILIES["F_EXPECTED_GAME_STATE_VOLUME"]["channel"] == "opportunity only"
