import pandas as pd

from scripts.modeling.context_bridge import build_player_contexts, build_team_contexts


def test_team_context_prefers_coverage_v2_and_maps_legacy_team_form_aliases():
    team_form = pd.DataFrame([
        {"team":"LA","season":2026,"success_rate_off":0.51,"success_rate_def":0.47,"dl_pressure_rate":0.34,"pressure_rate_allowed":0.26,"neutral_pace":28.0,"seconds_per_play_last5":27.0,"plays_est":64.0,"proe":0.03,"light_box_rate":0.61,"heavy_box_rate":0.20},
        {"team":"SF","season":2026,"success_rate_off":0.50,"success_rate_def":0.45,"dl_pressure_rate":0.31,"pressure_rate_allowed":0.25,"neutral_pace":29.0,"seconds_per_play_last5":28.0,"plays_est":63.0,"proe":0.01,"light_box_rate":0.30,"heavy_box_rate":0.30},
    ])
    coverage = pd.DataFrame([
        {"team":"LAR","man_rate":0.41,"zone_rate":0.59},
        {"team":"SF","man_rate":0.35,"zone_rate":0.65},
    ])
    teams = build_team_contexts(team_form, coverage)
    assert set(teams) == {"LAR", "SF"}
    assert teams["LAR"].pressure_rate_generated == 0.34
    assert teams["LAR"].coverage_man_rate == 0.41
    assert teams["SF"].coverage_zone_rate == 0.65


def test_player_context_keeps_authoritative_identity_and_optional_enrichment_separate():
    team_form = pd.DataFrame([
        {"team":"IND","season":2026,"success_rate_off":0.50,"success_rate_def":0.45},
        {"team":"HOU","season":2026,"success_rate_off":0.49,"success_rate_def":0.46},
    ])
    teams = build_team_contexts(team_form)
    player_form = pd.DataFrame([
        {"player":"Michael Pittman Jr.","team":"IND","opponent":"HOU","season":2026,"week":1,"position":"LWR","role":"WR1","game_id":"2026_01_HOU_IND"},
    ])
    consensus = pd.DataFrame([
        {"player":"Michael Pittman Jr.","team":"IND","season":2026,"position":"LWR","role":"WR1","tgt_share":0.25,"rush_share":0.0,"ypt":8.4},
    ])
    exposure = pd.DataFrame([
        {"player":"Michael Pittman Jr.","team":"IND","opponent":"HOU","primary_cb":"Example CB","exp_vs_man":0.40,"exp_vs_zone":0.60,"matchup_available":1,"team_coverage_available":1},
    ])
    injuries = pd.DataFrame(columns=["player","team","status","practice_status","designation","report_available"])
    weather = pd.DataFrame([{"game_id":"2026_01_HOU_IND","forecast_ok":0,"temp_f":None,"wind_mph":None,"precip_flag":None}])

    players = build_player_contexts(player_form, consensus, teams, exposure, injuries, weather)
    assert len(players) == 1
    p = players[0]
    assert p.team == "IND"
    assert p.opponent == "HOU"
    assert p.offense is teams["IND"]
    assert p.defense is teams["HOU"]
    assert p.features["tgt_share"] == 0.25
    assert p.features["primary_cb"] == "Example CB"
    assert p.features["matchup_available"] == 1
    assert p.features["injury_report_available"] == 0
    assert p.features["weather_forecast_available"] == 0


def test_player_context_hard_fails_unresolved_opponent():
    teams = build_team_contexts(pd.DataFrame([
        {"team":"IND","season":2026}, {"team":"HOU","season":2026}
    ]))
    bad = pd.DataFrame([
        {"player":"Test Player","team":"IND","opponent":"","season":2026,"week":1,"position":"WR"}
    ])
    consensus = pd.DataFrame([{"player":"Test Player","team":"IND","tgt_share":0.2,"rush_share":0.0}])
    try:
        build_player_contexts(bad, consensus, teams)
    except RuntimeError as exc:
        assert "unresolved team/opponent" in str(exc)
    else:
        raise AssertionError("Expected unresolved opponent to fail")
