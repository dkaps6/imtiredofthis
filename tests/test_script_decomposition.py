from types import SimpleNamespace

from scripts.backtest.run_script_decomposition import project_script_variant


def _team(**kwargs):
    base = dict(
        success_rate_off=0.52,
        success_rate_def=0.45,
        proe=0.06,
        plays_est=64.0,
        neutral_pace=28.0,
        neutral_pace_last5=28.0,
        sec_per_play_last5=28.0,
        pressure_rate_allowed=0.25,
        pressure_rate_generated=0.25,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_fixed_pass_share_is_exactly_55_percent_before_clamp():
    offense = _team(proe=0.10)
    defense = _team()
    out = project_script_variant(offense, defense, use_proe=False, use_success=False, game_state_coeff=0.0)
    assert abs(out.projected_pass_attempts / out.projected_plays - 0.55) < 1e-12


def test_no_proe_removes_only_proe_contribution():
    offense = _team(proe=0.06)
    defense = _team(success_rate_def=0.52)
    full = project_script_variant(offense, defense, use_proe=True, use_success=True, game_state_coeff=0.08)
    no_proe = project_script_variant(offense, defense, use_proe=False, use_success=True, game_state_coeff=0.08)
    assert abs((full.projected_pass_attempts - no_proe.projected_pass_attempts) / full.projected_plays - 0.06) < 1e-12


def test_game_state_coefficient_monotonically_changes_pass_share_when_trailing():
    offense = _team(success_rate_off=0.40, proe=0.0)
    defense = _team(success_rate_def=0.55)
    low = project_script_variant(offense, defense, use_proe=True, use_success=True, game_state_coeff=0.02)
    high = project_script_variant(offense, defense, use_proe=True, use_success=True, game_state_coeff=0.08)
    assert high.trail_prob > high.lead_prob
    assert high.projected_pass_attempts > low.projected_pass_attempts
