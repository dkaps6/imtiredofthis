import numpy as np
import pandas as pd

from scripts.research.audit_prism_opportunity_quality_team_high_value_volume_v1 import (
    FREEZE_COMMIT,
    HISTORY_MIN,
    HISTORY_WINDOW,
    TARGET_SEASONS,
    TARGET_WEEKS,
    build_team_topology,
    attach_strict_prior_state,
)


def _schedule():
    return pd.DataFrame([
        {"season": 2024, "week": 1, "team": "AAA", "opponent": "BBB"},
        {"season": 2024, "week": 1, "team": "BBB", "opponent": "AAA"},
        {"season": 2024, "week": 2, "team": "AAA", "opponent": "CCC"},
        {"season": 2024, "week": 2, "team": "CCC", "opponent": "AAA"},
    ])


def test_frozen_source_contract_constants():
    assert FREEZE_COMMIT == "48d9331f86aa052c90bc48816b2f031b84900c4b"
    assert TARGET_SEASONS == (2024, 2025)
    assert min(TARGET_WEEKS) == 5
    assert max(TARGET_WEEKS) == 18
    assert HISTORY_WINDOW == 4
    assert HISTORY_MIN == 3


def test_missing_parent_channel_stays_missing_not_fake_zero():
    # AAA has a real rush channel and zero inside-10 rushes. BBB has no rush
    # source row at all. Those states must remain distinguishable.
    pw = pd.DataFrame([
        {
            "season": 2024, "week": 1, "team": "AAA", "player_id": "p1",
            "family": "RB", "opportunities": 10, "inside20": 1,
            "inside10": 0, "inside5": 0, "channel": "rush",
        },
        {
            "season": 2024, "week": 1, "team": "AAA", "player_id": "p2",
            "family": "TE", "opportunities": 3, "inside20": 0,
            "inside10": 0, "inside5": 0, "channel": "target",
        },
    ])
    out = build_team_topology(pw, _schedule())

    aaa = out.loc[(out.team == "AAA") & (out.week == 1)].iloc[0]
    bbb = out.loc[(out.team == "BBB") & (out.week == 1)].iloc[0]

    assert bool(aaa.rush_source_present)
    assert float(aaa.team_inside10_rushes) == 0.0

    assert not bool(bbb.rush_source_present)
    assert np.isnan(float(bbb.team_inside10_rushes))


def test_strict_prior_state_never_reads_same_or_future_week():
    rows = []
    for week in range(1, 6):
        rows.extend([
            {
                "season": 2024, "week": week, "team": "AAA", "opponent": "BBB",
                "team_rushes": 20.0, "team_inside20_rushes": 3.0,
                "team_inside10_rushes": float(week), "team_inside5_rushes": 1.0,
                "team_targets": 30.0, "team_redzone_targets": 4.0,
                "team_inside10_targets": 2.0,
                "rush_source_present": True, "target_source_present": True,
            },
            {
                "season": 2024, "week": week, "team": "BBB", "opponent": "AAA",
                "team_rushes": 22.0, "team_inside20_rushes": 3.0,
                "team_inside10_rushes": 2.0, "team_inside5_rushes": 1.0,
                "team_targets": 28.0, "team_redzone_targets": 3.0,
                "team_inside10_targets": 1.0,
                "rush_source_present": True, "target_source_present": True,
            },
        ])
    x = attach_strict_prior_state(pd.DataFrame(rows))
    w5 = x.loc[(x.team == "AAA") & (x.week == 5)].iloc[0]

    assert int(w5.prior_off_games) == 4
    assert int(w5.off_max_prior_week) == 4
    assert int(w5.opp_max_prior_week) == 4
    assert int(w5.chronology_violation) == 0
    # Last-four weeks 1-4 only: mean inside-10 rushes is 2.5, not 3.0
    # (which would indicate target Week 5 leaked into the state).
    assert np.isclose(float(w5.prior_off_team_inside10_rushes_pg), 2.5)
