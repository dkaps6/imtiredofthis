"""Pin the slot-label contract mismatch against production source.

GPT replay 6455253348 observed zero SLOT-assigned WRs in the saved slate. This
pins the stronger structural claim: upstream collapses LWR/RWR/SWR to the WR
family deliberately, while _wr_role_labels' SLOT branch requires the alignment
value, so no slate can produce a SLOT label. If either side changes, these fail
and the published finding must be revised.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

TRACE = Path("data/research/week7_opportunity_trace")
DIAG = TRACE / "week7_trace__data__model_rule_diagnostics.csv"
ROLES = TRACE / "week7_trace__data__roles_ourlads.csv"


def test_upstream_collapses_alignment_into_the_wr_family():
    src = Path("scripts/run_player_form_v2.py").read_text(encoding="utf-8")
    assert re.search(
        r'if pos in \{"LWR", "RWR", "SWR", "WR", "WIDE RECEIVER"\}', src), (
        "the position-family collapse changed; revise the finding")
    assert 'out["position"] = out["position_group"]' in src


def test_slot_branch_requires_the_erased_alignment_value():
    src = Path("scripts/modeling/simulation_rules.py").read_text(encoding="utf-8")
    body = src.split("def _wr_role_labels(", 1)[1].split("\ndef ", 1)[0]
    assert 'str(p.position).upper() == "SWR"' in body, (
        "the SLOT branch no longer keys on the SWR alignment value")
    assert '"SLOT" in str(p.role).upper()' in body


def test_ourlads_depth_roles_never_contain_the_slot_substring():
    src = Path("scripts/providers/ourlads_depth.py").read_text(encoding="utf-8")
    assert 'WR_ROLE_BY_COL = {"LWR": "WR1", "SWR": "WR2", "RWR": "WR3"}' in src, (
        "depth-role mapping changed; the 'SLOT' in role condition may now match")


@pytest.mark.skipif(not DIAG.exists(), reason="week7 trace not present")
def test_no_player_receives_the_slot_label_in_the_slate():
    d = pd.read_csv(DIAG, low_memory=False)
    roles = d["rules_role"].astype(str).str.upper()
    assert int(roles.eq("SLOT").sum()) == 0
    # and only the two perimeter labels are ever assigned
    assigned = set(roles[roles.ne("NAN") & roles.ne("")].unique())
    assert assigned <= {"WR1", "WR1_5"}, f"unexpected labels {assigned}"


@pytest.mark.skipif(not (DIAG.exists() and ROLES.exists()),
                    reason="week7 trace not present")
def test_alignment_exists_upstream_and_is_gone_downstream():
    r = pd.read_csv(ROLES, low_memory=False)
    pos = r["position"].astype(str).str.upper()
    assert int(pos.eq("SWR").sum()) > 0, "roster no longer distinguishes SWR"
    assert int(pos.astype(str).str.contains("SLOT").sum()) == 0
    d = pd.read_csv(DIAG, low_memory=False)
    seen = set(d["position"].astype(str).str.upper().unique())
    assert not (seen & {"SWR", "LWR", "RWR"}), (
        f"alignment now survives to the rule layer: {seen}")
    assert "WR" in seen
