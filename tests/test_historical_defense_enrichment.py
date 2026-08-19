import pandas as pd

from scripts.backtest.enrich_historical_defense import (
    audit_frame,
    build_box_observations,
    enrich_team_weekly,
)


def test_box_rates_are_reconstructed_from_defenders_in_box(monkeypatch):
    pbp = pd.DataFrame([
        {"season_type": "REG", "week": 1, "posteam": "MIA", "defteam": "BUF", "rush_attempt": 1, "defenders_in_box": 6},
        {"season_type": "REG", "week": 1, "posteam": "MIA", "defteam": "BUF", "rush_attempt": 1, "defenders_in_box": 7},
        {"season_type": "REG", "week": 1, "posteam": "MIA", "defteam": "BUF", "rush_attempt": 1, "defenders_in_box": 8},
        {"season_type": "REG", "week": 1, "posteam": "MIA", "defteam": "BUF", "rush_attempt": 1, "defenders_in_box": 9},
        # Pass plays must not enter the rushing-box denominator.
        {"season_type": "REG", "week": 1, "posteam": "MIA", "defteam": "BUF", "rush_attempt": 0, "defenders_in_box": 5},
    ])
    monkeypatch.setattr("scripts.backtest.enrich_historical_defense.get_pbp", lambda season, min_rows=1: pbp.copy())
    out = build_box_observations([2025])
    buf = out.loc[out["team"].eq("BUF")].iloc[0]
    assert buf["box_snap_count"] == 4
    assert buf["avg_defenders_in_box"] == 7.5
    assert buf["light_box_rate"] == 0.25
    assert buf["heavy_box_rate"] == 0.50


def test_enrichment_preserves_team_week_identity_and_reports_coverage():
    base = pd.DataFrame([
        {"season": 2025, "week": 1, "team": "BUF", "def_rush_epa": -0.1},
        {"season": 2025, "week": 1, "team": "MIA", "def_rush_epa": 0.1},
    ])
    box = pd.DataFrame([
        {"season": 2025, "week": 1, "team": "BUF", "box_snap_count": 20, "avg_defenders_in_box": 7.2, "light_box_rate": 0.2, "heavy_box_rate": 0.4, "box_source": "nflverse_pbp_defenders_in_box"},
    ])
    out = enrich_team_weekly(base, box)
    assert len(out) == 2
    assert not out.duplicated(["season", "week", "team"]).any()
    audit = audit_frame(out).set_index("feature")
    assert audit.loc["box_rates", "available_team_weeks"] == 1
    assert audit.loc["box_rates", "coverage"] == 0.5
    assert audit.loc["coverage_scheme", "available_team_weeks"] == 0
    assert audit.loc["coverage_scheme", "status"] == "unsupported_no_trustworthy_man_zone_label"
