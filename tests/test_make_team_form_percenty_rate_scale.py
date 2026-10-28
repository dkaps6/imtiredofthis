"""The coverage/middle rates must reach the gates as fractions, not percents.

rules_v2.matchup_multipliers compares against fraction thresholds (zone >= 0.60,
man >= 0.50, middle >= 0.50). Sharp supplies these fields as percents. Before the
fix they were absent from PERCENTY_COLS, so middle_open_rate arrived as 24.0-64.6
and its gate fired for every team in the league instead of discriminating.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.make_team_form import PERCENTY_COLS, _prep_team_key_and_rates

GATE_FIELDS = ("coverage_man_rate", "coverage_zone_rate", "middle_open_rate", "middle_closed_rate")


@pytest.mark.parametrize("field", GATE_FIELDS)
def test_gate_consumed_rate_is_in_the_percent_allowlist(field):
    assert field in PERCENTY_COLS


def test_box_rates_remain_in_the_allowlist():
    """The fix must not disturb the fields that already worked."""
    for field in ("light_box_rate", "heavy_box_rate", "blitz_rate"):
        assert field in PERCENTY_COLS


def test_percent_scaled_sharp_values_are_converted_to_fractions():
    df = pd.DataFrame({
        "team": ["KC", "SEA"],
        "middle_open_rate": [24.0, 64.6],
        "coverage_zone_rate": [54.9, 89.2],
        "coverage_man_rate": [5.0, 41.7],
        "middle_closed_rate": [27.6, 73.0],
    })
    out = _prep_team_key_and_rates(df)
    assert out["middle_open_rate"].tolist() == pytest.approx([0.240, 0.646])
    assert out["coverage_zone_rate"].tolist() == pytest.approx([0.549, 0.892])
    assert out["coverage_man_rate"].tolist() == pytest.approx([0.050, 0.417])
    assert out["middle_closed_rate"].tolist() == pytest.approx([0.276, 0.730])


def test_already_fractional_values_are_left_alone():
    """Idempotent: a second pass, or a provider that switches to fractions, must not re-divide."""
    df = pd.DataFrame({"team": ["KC"], "middle_open_rate": [0.24], "coverage_zone_rate": [0.549]})
    out = _prep_team_key_and_rates(df)
    assert out["middle_open_rate"].tolist() == pytest.approx([0.24])
    assert out["coverage_zone_rate"].tolist() == pytest.approx([0.549])


def test_percent_sign_strings_are_handled():
    df = pd.DataFrame({"team": ["KC"], "middle_open_rate": ["64.6%"]})
    assert _prep_team_key_and_rates(df)["middle_open_rate"].tolist() == pytest.approx([0.646])


def test_the_gate_now_discriminates_instead_of_firing_for_everyone():
    """The regression this fix exists to prevent, stated as the gate's behaviour."""
    observed_w7 = [24.0, 31.2, 38.5, 44.1, 49.9, 50.1, 55.0, 64.6]   # percent scale
    # real team codes: _prep_team_key_and_rates canonicalizes and drops unknown ones
    df = pd.DataFrame({"team": ["KC", "SEA", "SF", "DAL", "BUF", "PHI", "DET", "GB"],
                       "middle_open_rate": observed_w7})
    raw_fires = sum(v >= 0.50 for v in observed_w7)
    fixed = _prep_team_key_and_rates(df)["middle_open_rate"]
    fixed_fires = int((fixed >= 0.50).sum())
    assert raw_fires == len(observed_w7), "unconverted percents fire the gate for every team"
    assert fixed_fires == 3, "after conversion only genuinely middle-open defenses fire"


def test_conversion_is_per_column_not_per_frame():
    """A frame mixing a percent column and a fraction column must convert only the percent one."""
    df = pd.DataFrame({"team": ["KC", "SEA"],
                       "middle_open_rate": [24.0, 64.6],     # percent
                       "light_box_rate": [0.253, 0.606]})     # already fraction
    out = _prep_team_key_and_rates(df)
    assert out["middle_open_rate"].tolist() == pytest.approx([0.240, 0.646])
    assert out["light_box_rate"].tolist() == pytest.approx([0.253, 0.606])
