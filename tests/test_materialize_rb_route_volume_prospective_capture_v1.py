from __future__ import annotations

import pandas as pd

from scripts.research.materialize_rb_route_volume_prospective_capture_v1 import (
    derive_week,
    parity,
)


def _stat(rows):
    return pd.DataFrame(rows, columns=[
        "player", "player_clean_key", "team", "position", "cumulative_routes"
    ])


def _heat(rows):
    return pd.DataFrame(rows, columns=[
        "player", "player_clean_key", "team", "position", "routes", "route_pct"
    ])


def test_weekly_delta_never_coerces_missing_to_zero():
    current = _stat([
        ["Back A", "backa", "IND", "RB", 30],
        ["Back B", "backb", "CHI", "RB", 12],
    ])
    prior = _stat([
        ["Back A", "backa", "IND", "RB", 20],
        ["Back C", "backc", "GB", "RB", 8],
    ])
    out = derive_week(current, prior)

    a = out.loc[out.player_clean_key.eq("backa")].iloc[0]
    b = out.loc[out.player_clean_key.eq("backb")].iloc[0]
    c = out.loc[out.player_clean_key.eq("backc")].iloc[0]

    assert a.derivation_status == "READY"
    assert a.derived_week_routes == 10
    assert b.derivation_status == "NO_PRIOR_SNAPSHOT"
    assert pd.isna(b.derived_week_routes)
    assert c.derivation_status == "MISSING_CURRENT_SNAPSHOT"
    assert pd.isna(c.derived_week_routes)


def test_negative_provider_revision_fails_closed():
    current = _stat([["Back A", "backa", "IND", "RB", 18]])
    prior = _stat([["Back A", "backa", "IND", "RB", 20]])
    out = derive_week(current, prior)
    row = out.iloc[0]
    assert row.derivation_status == "NEGATIVE_PROVIDER_DELTA"
    assert pd.isna(row.derived_week_routes)


def test_team_change_does_not_create_fake_weekly_delta():
    current = _stat([["Back A", "backa", "IND", "RB", 30]])
    prior = _stat([["Back A", "backa", "CHI", "RB", 20]])
    out = derive_week(current, prior)
    row = out.iloc[0]
    assert row.derivation_status == "TEAM_CHANGED"
    assert pd.isna(row.derived_week_routes)


def test_cross_source_parity_reports_exact_and_provider_only_rows():
    heat = _heat([
        ["Back A", "backa", "IND", "RB", 10, 0.40],
        ["Back B", "backb", "CHI", "RB", 7, 0.30],
    ])
    derived = pd.DataFrame([
        {
            "player_clean_key": "backa",
            "team_current": "IND",
            "derived_week_routes": 10.0,
            "derivation_status": "READY",
        },
        {
            "player_clean_key": "backc",
            "team_current": "GB",
            "derived_week_routes": 5.0,
            "derivation_status": "READY",
        },
    ])
    out = parity(heat, derived)

    a = out.loc[out.player_clean_key.eq("backa")].iloc[0]
    b = out.loc[out.player_clean_key.eq("backb")].iloc[0]
    c = out.loc[out.player_clean_key.eq("backc")].iloc[0]

    assert a.parity_status == "MATCHABLE"
    assert bool(a.exact_route_match)
    assert a.abs_route_gap == 0
    assert b.parity_status == "HEATRADAR_ONLY"
    assert c.parity_status == "STATRANKINGS_ONLY"
