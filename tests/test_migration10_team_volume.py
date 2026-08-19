import pytest

from scripts.modeling.contracts import TeamContext
from scripts.modeling.rules_v2 import estimate_plays, project_game_script


def _team(team: str, **kwargs) -> TeamContext:
    base = dict(
        team=team,
        season=2025,
        success_rate_off=0.48,
        success_rate_def=0.46,
        pressure_rate_generated=0.30,
        pressure_rate_allowed=0.30,
        neutral_pace=28.0,
        plays_est=64.0,
        proe=0.0,
    )
    base.update(kwargs)
    return TeamContext(**base)


def test_team_pace_uses_half_game_clock_not_full_game_clock():
    offense = _team("BUF", neutral_pace=28.0, plays_est=64.0)
    # 1800 / 28 = 64.29, so blending with a 64-play prior should stay near 64.
    # The old 3600 / 28 conversion clipped at 80 and inflated this to 72 plays.
    assert estimate_plays(offense) == pytest.approx(64.1428571429)


def test_game_script_no_longer_inflates_neutral_team_to_72_plays():
    offense = _team("BUF", neutral_pace=28.0, plays_est=64.0, proe=0.0)
    defense = _team("MIA", success_rate_def=0.48)
    script = project_game_script(offense, defense)
    assert script.projected_plays < 66.0
    assert script.projected_pass_attempts < 37.0
    assert script.projected_pass_attempts + script.projected_rush_attempts == pytest.approx(script.projected_plays)
