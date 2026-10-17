from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research.evaluate_prism_team_scoring_territory_drive_creation_v1 import (
    build_team_drive_rows,
)


def _row(
    season,
    week,
    team,
    opponent,
    *,
    rushes=20,
    i10=2,
    targets=30,
    rz=4,
    drives=10,
    reach20=5,
    reach10=2,
):
    return {
        "season": season,
        "week": week,
        "game_id": f"{season}_{week:02d}_{team}_{opponent}",
        "game_key": f"{season}_{week:02d}_{team}_{opponent}",
        "team": team,
        "opponent": opponent,
        "rush_source_present": True,
        "target_source_present": True,
        "drive_source_present": True,
        "team_rush_attempts": float(rushes),
        "inside10_rush": float(i10),
        "team_targets": float(targets),
        "red_zone_targets": float(rz),
        "offensive_drives": float(drives),
        "drives_reached20": float(reach20),
        "drives_reached10": float(reach10),
    }


def test_rush_drive_candidate_is_last4_strict_prior_territory_transmission():
    rows = []
    # AAA W1 is excluded from its W6 last-four drive state.
    rows += [
        _row(2024, 1, "AAA", "ZZZ", i10=10, reach20=6, reach10=1),
        _row(2024, 2, "AAA", "ZZZ", i10=2, reach20=6, reach10=4),
        _row(2024, 3, "AAA", "ZZZ", i10=2, reach20=6, reach10=4),
        _row(2024, 4, "AAA", "ZZZ", i10=2, reach20=6, reach10=4),
        _row(2024, 5, "AAA", "ZZZ", i10=2, reach20=6, reach10=4),
        _row(2024, 6, "AAA", "ZZZ", i10=1, reach20=6, reach10=1),
    ]
    # BBB W6 is deliberately extreme and must not leak into AAA W6 state.
    rows += [
        _row(2024, 1, "BBB", "YYY", i10=0, reach20=1, reach10=0),
        _row(2024, 2, "BBB", "YYY", i10=0, reach20=1, reach10=0),
        _row(2024, 3, "BBB", "YYY", i10=0, reach20=1, reach10=0),
        _row(2024, 4, "BBB", "YYY", i10=0, reach20=1, reach10=0),
        _row(2024, 5, "BBB", "YYY", i10=0, reach20=1, reach10=0),
        _row(2024, 6, "BBB", "YYY", i10=20, reach20=10, reach10=10),
    ]
    x = pd.DataFrame(rows)
    out = build_team_drive_rows(x, "RB_RUSH_INSIDE10")
    r = out.loc[(out["team"] == "AAA") & (out["week"] == 6)].iloc[0]

    # Strict-prior league high-value baseline: AAA 18 + BBB 0 / 200 rushes.
    assert np.isclose(r["league_high_value_rate"], 18 / 200)

    # Frozen team drive state is AAA W2-W5 only: 16 reached-10 / 40 drives.
    assert np.isclose(r["team_drive_entry_rate"], 16 / 40)
    assert r["team_drive_history_min_week"] == 2
    assert r["team_drive_history_max_week"] == 5

    # Strict-prior league drive state: AAA (1+4+4+4+4)=17, BBB=0 / 100.
    assert np.isclose(r["league_drive_entry_rate"], 17 / 100)
    expected = (18 / 200) * ((16 / 40) / (17 / 100))
    assert np.isclose(r["candidate_high_value_rate"], expected)

    # The candidate is NOT direct team recent high-value event-rate persistence:
    # AAA W2-W5 direct inside-10 rush rate is 8/80 = 0.10.
    assert not np.isclose(r["candidate_high_value_rate"], 8 / 80)

    assert r["league_hv_history_max_week"] == 5
    assert r["league_drive_history_max_week"] == 5
    assert r["chronology_violation"] == 0


def test_receiving_uses_reach20_not_reach10():
    rows = []
    for week in range(1, 7):
        rows.append(
            _row(
                2024,
                week,
                "AAA",
                "ZZZ",
                rz=3,
                reach20=8 if week < 6 else 2,
                reach10=0,
            )
        )
        rows.append(
            _row(
                2024,
                week,
                "BBB",
                "YYY",
                rz=3,
                reach20=2 if week < 6 else 10,
                reach10=0,
            )
        )
    x = pd.DataFrame(rows)
    out = build_team_drive_rows(x, "RB_RECEIVING_REDZONE")
    r = out.loc[(out["team"] == "AAA") & (out["week"] == 6)].iloc[0]

    # If reach10 had been used this row would be unscoreable because league g=0.
    assert np.isfinite(r["candidate_high_value_rate"])
    assert r["territory"] == "OPPONENT_20"
    assert np.isclose(r["team_drive_entry_rate"], 0.8)
    assert np.isclose(r["league_drive_entry_rate"], 0.5)


def test_candidate_rate_is_clipped_at_one_by_frozen_contract():
    rows = []
    for week in range(1, 7):
        # AAA has strong reach10 propensity.
        rows.append(
            _row(
                2024,
                week,
                "AAA",
                "ZZZ",
                rushes=10,
                i10=9 if week < 6 else 1,
                drives=10,
                reach20=10,
                reach10=10,
            )
        )
        # BBB creates enough poor drive state to make AAA relative multiplier >1.
        rows.append(
            _row(
                2024,
                week,
                "BBB",
                "YYY",
                rushes=10,
                i10=9 if week < 6 else 1,
                drives=10,
                reach20=1,
                reach10=1,
            )
        )
    x = pd.DataFrame(rows)
    out = build_team_drive_rows(x, "RB_RUSH_INSIDE10")
    r = out.loc[(out["team"] == "AAA") & (out["week"] == 6)].iloc[0]

    assert r["candidate_high_value_rate_raw"] > 1.0
    assert r["candidate_high_value_rate"] == 1.0
    assert bool(r["candidate_rate_clipped"])


def test_target_game_drive_outcome_does_not_change_pregame_candidate():
    base = []
    changed = []
    for week in range(1, 7):
        a = _row(
            2024, week, "AAA", "ZZZ",
            i10=2, drives=10, reach20=6, reach10=4,
        )
        b = _row(
            2024, week, "BBB", "YYY",
            i10=1, drives=10, reach20=3, reach10=1,
        )
        base.extend([a, b])
        changed.extend([dict(a), dict(b)])

    # Change only target-week W6 drive outcomes. Candidate must be identical.
    for r in changed:
        if r["week"] == 6:
            r["drives_reached20"] = 10.0
            r["drives_reached10"] = 10.0

    a = build_team_drive_rows(pd.DataFrame(base), "RB_RUSH_INSIDE10")
    b = build_team_drive_rows(pd.DataFrame(changed), "RB_RUSH_INSIDE10")
    ka = a.loc[(a["team"] == "AAA") & (a["week"] == 6)].iloc[0]
    kb = b.loc[(b["team"] == "AAA") & (b["week"] == 6)].iloc[0]

    assert np.isclose(
        ka["candidate_high_value_rate"], kb["candidate_high_value_rate"]
    )
