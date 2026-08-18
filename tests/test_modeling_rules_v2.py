import pytest

from scripts.modeling.contracts import TeamContext
from scripts.modeling.rules_v2 import (
    coverage_penalty,
    matchup_multipliers,
    offensive_pressure_mismatch,
    project_game_script,
    redistribute_alpha_usage,
)


def _team(team: str, **kwargs) -> TeamContext:
    base = dict(
        team=team,
        season=2025,
        success_rate_off=0.48,
        success_rate_def=0.46,
        pressure_rate_generated=0.30,
        pressure_rate_allowed=0.30,
        neutral_pace=28.0,
        neutral_pace_last5=27.0,
        sec_per_play_last5=27.0,
        plays_est=64.0,
        proe=0.0,
        explosive_play_rate_allowed=0.10,
        coverage_man_rate=0.30,
        coverage_zone_rate=0.50,
        middle_open_rate=0.30,
        light_box_rate=0.30,
        heavy_box_rate=0.30,
        def_pass_epa=0.0,
        def_rush_epa=0.0,
    )
    base.update(kwargs)
    return TeamContext(**base)


def test_pressure_mismatch_compares_defense_generation_to_offense_allowed():
    offense = _team("IND", pressure_rate_allowed=0.20, pressure_rate_generated=0.45)
    defense = _team("HOU", pressure_rate_generated=0.35, pressure_rate_allowed=0.10)
    assert offensive_pressure_mismatch(offense, defense) == pytest.approx(0.15)


def test_legacy_zone_man_and_box_rules_are_preserved():
    offense = _team("IND", pressure_rate_allowed=0.20)
    defense = _team(
        "HOU",
        pressure_rate_generated=0.30,
        coverage_zone_rate=0.65,
        coverage_man_rate=0.55,
        middle_open_rate=0.55,
        light_box_rate=0.65,
    )
    mods = matchup_multipliers(offense, defense)
    assert mods.wr1_target_mult == pytest.approx(0.95)
    assert mods.wr1_5_target_mult == pytest.approx(1.15)
    assert mods.slot_target_mult == pytest.approx(1.05 * 1.10)
    assert mods.te_target_mult == pytest.approx(1.15 * 1.10)
    assert mods.rb_rec_target_mult > 1.20
    assert mods.rb_rush_eff_mult == pytest.approx(1.07)
    assert mods.pass_eff_mult < 1.0


def test_script_projection_is_bounded_and_sums_volume():
    offense = _team("IND", success_rate_off=0.55, proe=0.08)
    defense = _team("HOU", success_rate_def=0.42)
    script = project_game_script(offense, defense)
    assert 48.0 <= script.projected_plays <= 82.0
    assert (script.projected_pass_attempts + script.projected_rush_attempts) == pytest.approx(script.projected_plays)
    assert (script.lead_prob + script.neutral_prob + script.trail_prob) == pytest.approx(1.0)


def test_empirical_coverage_and_injury_rules_are_preserved():
    ypt, share = coverage_penalty(10.0, 0.25, tough_shadow=True)
    assert ypt == pytest.approx(9.4)
    assert share == pytest.approx(0.23)

    alpha, wr2, slot_te, rb = redistribute_alpha_usage(0.30, 0.20, 0.20, 0.10, alpha_limited=True)
    assert alpha == pytest.approx(0.15)
    assert wr2 == pytest.approx(0.29)
    assert slot_te == pytest.approx(0.245)
    assert rb == pytest.approx(0.115)
