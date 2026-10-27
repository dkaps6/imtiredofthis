"""Pin the game-script transmission finding against production source."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.research.trace_game_script_transmission_v1 import (
    CONSTANT_DERIVED,
    OPPONENT_AWARE,
    assert_source_contract,
    consumption,
)

STATUS = Path("data/research/game_script/game_script_transmission_v1_status.json")


def test_source_contract_holds():
    c = assert_source_contract()
    assert c["pass_share_literal"] == 0.57
    assert c["opponent_aware_diff_computed"] is True
    assert c["split_built_from_constant"] is True
    assert c["rules_pass_rate_is_constant_ratio"] is True
    assert c["sim_prefers_rules_pass_rate"] is True
    assert c["proe_fallback_base"] == 0.58
    assert c["proe_fallback_is_opponent_environment_aware"] is True


def test_contract_raises_if_the_split_becomes_opponent_aware(tmp_path,
                                                             monkeypatch):
    """If someone wires diff into the split, this finding must stop asserting."""
    src = Path("scripts/modeling/rules_v2.py").read_text(encoding="utf-8")
    broken = src.replace("    pass_share = 0.57",
                         "    pass_share = 0.57 + diff", 1)
    assert broken != src
    d = tmp_path / "scripts" / "modeling"
    d.mkdir(parents=True)
    (d / "rules_v2.py").write_text(broken, encoding="utf-8")
    (d / "simulation_rules.py").write_text(
        Path("scripts/modeling/simulation_rules.py").read_text(encoding="utf-8"),
        encoding="utf-8")
    (tmp_path / "scripts" / "simulation_v2.py").write_text(
        Path("scripts/simulation_v2.py").read_text(encoding="utf-8"),
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="bare literal"):
        assert_source_contract()


def test_every_opponent_aware_field_reaches_no_projection():
    cons = consumption()
    for field in OPPONENT_AWARE:
        assert cons[field]["reaches_a_projection"] is False, (
            f"{field} now reaches a projection; the published finding is stale")
    # and the two that do reach one are plays and the constant-derived split
    assert cons["projected_plays"]["reaches_a_projection"] is True
    assert cons["projected_pass_attempts"]["reaches_a_projection"] is True


def test_lead_and_trail_are_diagnostics_only_not_nowhere():
    """The distinction matters: referenced, but only by a diagnostics writer."""
    cons = consumption()
    for field in ("lead_prob", "trail_prob"):
        assert cons[field]["diagnostics_only_references"] >= 1
        assert cons[field]["referenced_in"], f"{field} should be referenced"
        assert cons[field]["reaches_a_projection"] is False
    for field in ("neutral_prob", "blowout_risk", "shootout_risk"):
        assert not cons[field]["referenced_in"], f"{field} should be unreferenced"


def test_rules_pass_rate_is_constant_by_construction():
    """(plays * 0.57) / plays == 0.57 for any finite positive plays."""
    sr = Path("scripts/modeling/simulation_rules.py").read_text(encoding="utf-8")
    assert re.search(
        r'rules_pass_rate"\]\s*=\s*script\.projected_pass_attempts\s*/\s*'
        r'script\.projected_plays', sr)
    rules = Path("scripts/modeling/rules_v2.py").read_text(encoding="utf-8")
    body = rules.split("def project_game_script(", 1)[1].split("\ndef ", 1)[0]
    assert re.search(r"pass_attempts\s*=\s*plays\s*\*\s*pass_share", body)
    for plays in (48.0, 64.0, 80.0):
        assert (plays * 0.57) / plays == pytest.approx(0.57)


@pytest.mark.skipif(not STATUS.exists(), reason="trace not run")
def test_status_makes_no_predictive_or_alternative_claim():
    st = json.loads(STATUS.read_text(encoding="utf-8"))
    assert st["artifacts_read"] is False
    assert st["outcomes_inspected"] is False
    assert st["sportsbook_inputs_used"] == 0
    assert any("would improve" in c for c in st["not_established"])
    doc = Path("docs/research/"
               "PRISM_GAME_SCRIPT_TRANSMISSION_TRACE_V1.md").read_text(
                   encoding="utf-8")
    assert "no alternative is proposed" in doc.lower()
