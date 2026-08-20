import pytest

from scripts.modeling.contracts import TeamContext
from scripts.modeling.rules_v2 import coverage_penalty, matchup_multipliers, offensive_pressure_mismatch, project_game_script, redistribute_alpha_usage


def _team(team: str, **kwargs) -> TeamContext:
    base = dict(team=team, season=2025, success_rate_off=0.48, success_rate_def=0.46, pressure_rate_generated=0.30, pressure_rate_allowed=0.30, neutral_pace=28.0, neutral_pace_last5=27.0, sec_per_play_last5=27.0, plays_est=64.0, proe=0.0, explosive_play_rate_allowed=0.10, coverage_man_rate=0.30, coverage_zone_rate=0.50, middle_open_rate=0.30, light_box_rate=0.30, heavy_box_rate=0.30, def_pass_epa=0.0, def_rush_epa=0.0)
    base.update(kwargs); return TeamContext(**base)


def test_pressure_mismatch_compares_defense_generation_to_offense_allowed():
    offense = _team("IND", pressure_rate_allowed=0.20, pressure_rate_generated=0.45)
    defense = _team("HOU", pressure_rate_generated=0.35, pressure_rate_allowed=0.10)
    assert offensive_pressure_mismatch(offense, defense) == pytest.approx(0.15)


def test_coverage_man_and_box_rules_are_preserved():
    offense = _team("IND", pressure_rate_allowed=0.20)
    defense = _team("HOU", pressure_rate_generated=0.30, coverage_zone_rate=0.65, coverage_man_rate=0.55, middle_open_rate=0.55, light_box_rate=0.65)
    mods = matchup_multipliers(offense, defense)
    assert mods.wr1_target_mult == pytest.approx(0.95)
    assert mods.wr1_5_target_mult == pytest.approx(1.15)
    assert mods.slot_target_mult == pytest.approx(1.05 * 1.10)
    assert mods.te_target_mult == pytest.approx(1.15 * 1.10)
    assert mods.rb_rec_target_mult == pytest.approx(1.20 * 1.10)
    assert mods.rb_rush_eff_mult == pytest.approx(1.07)
    assert mods.pass_eff_mult == pytest.approx(0.98)


def test_migration23_gentle_pressure_rule_and_threshold():
    offense = _team("IND", pressure_rate_allowed=0.20)
    strong = matchup_multipliers(offense, _team("HOU", pressure_rate_generated=0.30))
    assert strong.pass_eff_mult == pytest.approx(0.98)
    assert strong.rb_rec_target_mult == pytest.approx(1.10)
    assert strong.sack_mult == pytest.approx(1.05)
    assert strong.int_mult == pytest.approx(1.05)
    assert strong.volatility_mult == pytest.approx(1.05)

    neutral = matchup_multipliers(offense, _team("HOU", pressure_rate_generated=0.27))
    assert neutral.pass_eff_mult == pytest.approx(1.0)
    assert neutral.rb_rec_target_mult == pytest.approx(1.0)
    assert neutral.sack_mult == pytest.approx(1.0)

    clean = matchup_multipliers(_team("IND", pressure_rate_allowed=0.30), _team("HOU", pressure_rate_generated=0.20))
    assert clean.pass_eff_mult == pytest.approx(1.01)


def test_migration23_pressure_flag_uses_calibrated_threshold():
    below = project_game_script(_team("IND", pressure_rate_allowed=0.20), _team("HOU", pressure_rate_generated=0.27))
    above = project_game_script(_team("IND", pressure_rate_allowed=0.20), _team("HOU", pressure_rate_generated=0.30))
    assert below.pressure_mismatch is False
    assert above.pressure_mismatch is True


def test_migration21_uses_calibrated_fixed_57_percent_pass_share():
    offense = _team("IND", success_rate_off=0.60, proe=0.10); defense = _team("HOU", success_rate_def=0.35)
    script = project_game_script(offense, defense)
    assert script.projected_pass_attempts / script.projected_plays == pytest.approx(0.57)
    assert script.projected_rush_attempts / script.projected_plays == pytest.approx(0.43)


def test_migration21_proe_and_script_state_do_not_change_opportunity_split():
    defense = _team("HOU", success_rate_def=0.45)
    a = project_game_script(_team("IND", success_rate_off=0.60, proe=0.10), defense)
    b = project_game_script(_team("IND", success_rate_off=0.30, proe=-0.10), defense)
    assert a.projected_pass_attempts / a.projected_plays == pytest.approx(0.57)
    assert b.projected_pass_attempts / b.projected_plays == pytest.approx(0.57)
    assert a.lead_prob != pytest.approx(b.lead_prob)


def test_empirical_coverage_and_injury_rules_are_preserved():
    ypt, share = coverage_penalty(10.0, 0.25, tough_shadow=True)
    assert ypt == pytest.approx(9.4); assert share == pytest.approx(0.23)
    alpha, wr2, slot_te, rb = redistribute_alpha_usage(0.30, 0.20, 0.20, 0.10, alpha_limited=True)
    assert alpha == pytest.approx(0.15); assert wr2 == pytest.approx(0.29); assert slot_te == pytest.approx(0.245); assert rb == pytest.approx(0.115)
