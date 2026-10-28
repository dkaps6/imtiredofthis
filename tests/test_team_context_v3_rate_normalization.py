"""Rate-scale guard at the canonical team-context boundary.

Confirmed defect: Sharp publishes several rate fields as percentage points and
`team_context_v3` had no normalization or range validation, so
`middle_open_rate` reached `rules_v2.matchup_multipliers` as e.g. 24.4 while
the rule evaluates `middle >= 0.50` expecting a fraction. The gate therefore
fired for every defense. The bye-team coverage fallback had the same defect for
man/zone.

Found by GPT artifact audit 6451272159 and its addendum 6451282717, and
independently confirmed in the recovered Week-7 artifacts: 32/32
middle_open_rate values > 1 (min 24.0, median 48.15, max 64.6), with
slot_target_mult = 1.10 on all 396 player rows.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.team_context_v3 import _RATE_FIELDS, _normalize_rate


@pytest.mark.parametrize("field,raw,expected", [
    # the confirmed middle-open defect
    ("middle_open_rate", 24.4, 0.244),
    ("middle_open_rate", 48.15, 0.4815),
    ("middle_open_rate", 62.8, 0.628),
    ("middle_closed_rate", 73.0, 0.73),
    # the confirmed bye-team coverage fallback defect
    ("coverage_man_rate", 15.7, 0.157),
    ("coverage_zone_rate", 80.3, 0.803),
])
def test_percentage_inputs_are_rescaled(field, raw, expected):
    value, state = _normalize_rate(field, raw)
    assert value == pytest.approx(expected)
    assert state == "_percent_rescaled"


@pytest.mark.parametrize("field,raw", [
    ("light_box_rate", 0.4525), ("heavy_box_rate", 0.250),
    ("coverage_zone_rate", 0.7560), ("coverage_man_rate", 0.1810),
    ("middle_open_rate", 0.244), ("rz_rate", 0.0),
])
def test_correct_fractions_are_untouched(field, raw):
    value, state = _normalize_rate(field, raw)
    assert value == pytest.approx(raw)
    assert state == ""


@pytest.mark.parametrize("raw", [143.0, 1000.0, -0.2, -15.0])
def test_out_of_range_fails_closed_rather_than_clamping(raw):
    """Clamping would hide a provider change behind a plausible number."""
    value, state = _normalize_rate("middle_open_rate", raw)
    assert not np.isfinite(value)
    assert state == "_out_of_range_failed_closed"


def test_exactly_one_is_left_alone():
    """1.0 is a valid fraction; the 100% ambiguity cannot be resolved here."""
    value, state = _normalize_rate("coverage_zone_rate", 1.0)
    assert value == pytest.approx(1.0)
    assert state == ""


@pytest.mark.parametrize("field,raw", [
    # A naive "rescale anything > 1" would corrupt these.
    ("neutral_pace_last5", 27.4),
    ("sec_per_play_last5", 29.1),
    ("def_rush_epa", -0.325),
    ("ay_per_att", 8.4),
])
def test_non_rate_fields_are_never_rescaled(field, raw):
    assert field not in _RATE_FIELDS
    value, state = _normalize_rate(field, raw)
    assert value == pytest.approx(raw)
    assert state == ""


def test_nan_and_non_numeric_pass_through():
    v, s = _normalize_rate("middle_open_rate", float("nan"))
    assert not np.isfinite(v) and s == ""
    v, s = _normalize_rate("middle_open_rate", None)
    assert v is None and s == ""


def test_the_production_thresholds_are_not_touched_by_this_fix():
    """The fix restores intended semantics; it must not retune a gate."""
    from pathlib import Path
    src = Path("scripts/modeling/rules_v2.py").read_text(encoding="utf-8")
    assert "if middle >= 0.50:" in src
    assert "if light >= 0.60:" in src
    assert "if heavy >= 0.60:" in src
    assert "if zone >= 0.60:" in src
    assert "if man >= 0.50:" in src


def test_guard_covers_every_rate_field_the_rules_layer_gates_on():
    for gated in ("coverage_man_rate", "coverage_zone_rate",
                  "middle_open_rate", "light_box_rate", "heavy_box_rate"):
        assert gated in _RATE_FIELDS, f"{gated} is gated but unguarded"
