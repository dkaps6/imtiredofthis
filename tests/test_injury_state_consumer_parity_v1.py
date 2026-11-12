"""Every injury-state consumer must handle every state the builder can emit.

This test exists because the same defect shipped twice.

`scripts/build/build_injuries_weekly.py` defines VALID_STATES. Several
validators independently hard-code the states they accept and raise on anything
else. Adding a state to the builder without extending those allow-lists breaks
the pipeline at `validate_full_slate_pre_model_v1`, which is the first validator
PlayerForm calls -- so the symptom is "Build PlayerForm failed", a long way from
the cause.

  2026-11-04  `practice_report_only` added, broke PlayerForm, was reverted.
  2026-11-10  `practice_only_report` added and three validators updated;
              `validate_full_slate_pre_model_v1` was missed. Every Full Slate
              run from that date failed, and the breakage went unnoticed for two
              days because the readiness audit it also runs reported only a
              non-blocking P2.

So this test enumerates the builder's states and asserts each one is reachable
in each consumer, rather than trusting that a future change updates them all.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build" / "build_injuries_weekly.py"

# Consumers that dispatch on the injury state and FAIL CLOSED on an unknown
# value. These are the ones a new state can break, so each must name every
# state the builder can emit.
#
# Deliberately excluded, with reasons:
#   repair_injuries_nflcom_v1       -- holds an allow-list of the states that
#                                      NEED repair and no-ops on the rest. Not
#                                      naming `no_official_report` is correct
#                                      there: there is nothing to repair.
#   certify_preserved_injury_scope_v1 -- certifies a preserved replay of an
#                                      already-graded week, where an official
#                                      designation report is a real requirement
#                                      rather than an allow-list oversight.
STATE_CONSUMERS = (
    ROOT / "scripts" / "validate_full_slate_pre_model_v1.py",
    ROOT / "scripts" / "validate_2026_provider_artifacts.py",
    ROOT / "scripts" / "validate_full_slate_data_quality_v1.py",
)
# Consumers that select the states needing work instead of dispatching on all.
REPAIR_ALLOWLIST = ROOT / "scripts" / "repair_injuries_nflcom_v1.py"


def _builder_states() -> tuple[str, ...]:
    """Read VALID_STATES out of the builder without importing it."""
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    consts: dict[str, str] = {}
    valid: tuple[str, ...] | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            consts[target.id] = node.value.value
        elif target.id == "VALID_STATES" and isinstance(node.value, (ast.Tuple, ast.List)):
            out = []
            for el in node.value.elts:
                if isinstance(el, ast.Constant) and isinstance(el.value, str):
                    out.append(el.value)
                elif isinstance(el, ast.Name) and el.id in consts:
                    out.append(consts[el.id])
                else:  # pragma: no cover - defensive
                    pytest.fail(f"VALID_STATES holds an unresolvable element: {ast.dump(el)}")
            valid = tuple(out)
    if valid is None:
        pytest.fail("build_injuries_weekly.py no longer defines VALID_STATES")
    return valid


def test_builder_defines_the_expected_state_vocabulary():
    assert set(_builder_states()) == {
        "official_report", "practice_only_report",
        "no_official_report", "provider_outage",
    }


def test_every_builder_state_is_named_by_every_consumer():
    """The regression guard. A state the builder can emit but a consumer never
    names will fall through that consumer's else-branch and raise."""
    states = _builder_states()
    missing: list[str] = []
    for path in STATE_CONSUMERS:
        text = path.read_text(encoding="utf-8")
        for state in states:
            if state not in text:
                missing.append(f"{path.relative_to(ROOT)} does not handle {state!r}")
    assert not missing, (
        "injury-state consumers are out of parity with the builder:\n  "
        + "\n  ".join(missing)
        + "\n\nAdding a state to build_injuries_weekly.VALID_STATES requires "
          "extending every consumer in the same change."
    )


@pytest.mark.parametrize("path", STATE_CONSUMERS, ids=lambda p: p.name)
def test_consumer_still_rejects_an_unknown_state(path: Path):
    """Parity must not be achieved by deleting the fail-closed else-branch."""
    text = path.read_text(encoding="utf-8")
    assert "raise" in text, f"{path.name} no longer raises on any state"


def test_pre_model_validator_accepts_the_midweek_practice_state():
    """The specific state that broke production: practice participation
    published Wed/Thu with no game designation until Friday."""
    text = (ROOT / "scripts" / "validate_full_slate_pre_model_v1.py").read_text(encoding="utf-8")
    assert 'state == "practice_only_report"' in text
    assert "PASS_PRACTICE_ONLY_REPORT" in text
    # and it must still demand rows, or an empty artifact would pass as practice
    assert "practice_only_report but injuries.csv has zero rows" in text


def test_repair_allowlist_covers_exactly_the_states_with_rows_to_repair():
    """repair_injuries_nflcom_v1 is an allow-list, not a dispatch. It must cover
    the two states that carry rows, and it must not try to repair the two that
    do not."""
    text = REPAIR_ALLOWLIST.read_text(encoding="utf-8")
    assert '{"official_report", "practice_only_report"}' in text, (
        "the repair allow-list no longer covers both row-bearing states; a "
        "practice-only report would skip identity repair"
    )
