import pandas as pd

from scripts.backtest.enrich_historical_defense import (
    audit_frame,
    build_defensive_observations_from_joined,
    enrich_team_weekly,
)


def test_participation_reconstructs_box_and_man_zone_rates():
    joined = pd.DataFrame([
        {"season":2025,"week":1,"team":"BUF","rush_attempt":1,"defenders_in_box":6,"defense_man_zone_type":"MAN","defense_coverage_type":"COVER 1"},
        {"season":2025,"week":1,"team":"BUF","rush_attempt":1,"defenders_in_box":7,"defense_man_zone_type":"ZONE","defense_coverage_type":"COVER 3"},
        {"season":2025,"week":1,"team":"BUF","rush_attempt":1,"defenders_in_box":8,"defense_man_zone_type":"ZONE","defense_coverage_type":"COVER 4"},
        {"season":2025,"week":1,"team":"BUF","rush_attempt":1,"defenders_in_box":9,"defense_man_zone_type":"MAN","defense_coverage_type":"COVER 1"},
        # Passing play contributes to coverage but not the rushing box denominator.
        {"season":2025,"week":1,"team":"BUF","rush_attempt":0,"defenders_in_box":5,"defense_man_zone_type":"ZONE","defense_coverage_type":"COVER 3"},
    ])
    out = build_defensive_observations_from_joined(joined)
    buf = out.iloc[0]
    assert buf["box_snap_count"] == 4
    assert buf["avg_defenders_in_box"] == 7.5
    assert buf["light_box_rate"] == 0.25
    assert buf["heavy_box_rate"] == 0.50
    assert buf["coverage_snap_count"] == 5
    assert buf["coverage_man_rate"] == 0.4
    assert buf["coverage_zone_rate"] == 0.6
    assert buf["cover_1_rate"] == 0.4
    assert buf["cover_3_rate"] == 0.4
    assert buf["cover_4_rate"] == 0.2


def test_enrichment_preserves_team_week_identity_and_reports_both_layers():
    base = pd.DataFrame([
        {"season":2025,"week":1,"team":"BUF","def_rush_epa":-0.1},
        {"season":2025,"week":1,"team":"MIA","def_rush_epa":0.1},
    ])
    defense = pd.DataFrame([
        {"season":2025,"week":1,"team":"BUF","box_snap_count":20,"avg_defenders_in_box":7.2,"light_box_rate":0.2,"heavy_box_rate":0.4,"box_source":"nflverse_participation","coverage_snap_count":30,"coverage_man_rate":0.45,"coverage_zone_rate":0.55,"coverage_source":"nflverse_participation"},
    ])
    out = enrich_team_weekly(base, defense)
    assert len(out) == 2
    assert not out.duplicated(["season","week","team"]).any()
    audit = audit_frame(out).set_index("feature")
    assert audit.loc["box_rates","available_team_weeks"] == 1
    assert audit.loc["box_rates","coverage"] == 0.5
    assert audit.loc["coverage_scheme","available_team_weeks"] == 1
    assert audit.loc["coverage_scheme","coverage"] == 0.5
    assert audit.loc["coverage_scheme","status"] == "recovered"
