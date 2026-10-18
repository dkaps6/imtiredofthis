import pandas as pd
import pytest

from scripts.research.evaluate_prism_wr_te_player_component_dominance_routing_v1 import (
    build_predictions,
    component_class,
    prepare_panel,
)


def _row(season, week, team, player, state):
    if state == "O":
        opp, eff = 6.0, 1.0
    elif state == "E":
        opp, eff = 1.0, 6.0
    else:
        opp, eff = 3.0, 3.0
    return {
        "season": season,
        "week": week,
        "team": team,
        "player_clean_key": player,
        "current_opportunity_error": opp,
        "current_efficiency_error": eff,
        "current_total_error": opp + eff,
    }


def _panel():
    rows = []
    # Target player A: four strictly-prior opportunity-dominant rows, then target.
    for week in range(1, 6):
        rows.append(_row(2024, week, "AAA", "a", "O"))
    # Other players define the target-player-excluded position prior as efficiency-dominant.
    for player, team in [("b", "BBB"), ("c", "CCC")]:
        for week in range(1, 6):
            rows.append(_row(2024, week, team, player, "E"))
    # Second season required by panel contract; enough rows for chronology.
    for player, team, state in [("a", "AAA", "O"), ("b", "BBB", "E"), ("c", "CCC", "E")]:
        for week in range(1, 6):
            rows.append(_row(2025, week, team, player, state))
    return pd.DataFrame(rows)


def test_component_class_uses_inherited_1_5_ratio():
    assert component_class(6, 1) == "OPPORTUNITY_DOMINANT"
    assert component_class(1, 6) == "EFFICIENCY_DOMINANT"
    assert component_class(3, 3) == "MIXED"


def test_hierarchy_excludes_target_player_and_same_week():
    panel = prepare_panel(_panel(), "TE", (2024, 2025))
    out = build_predictions(panel, "TE")
    row = out.loc[
        out["season"].eq(2024)
        & out["week"].eq(5)
        & out["player_clean_key"].eq("a")
    ].iloc[0]

    # A's four prior rows are all opportunity-dominant.
    assert row["prior_player_rows"] == 4
    assert row["prior_player_max_week"] == 4
    assert row["player_opp_share"] == pytest.approx(1.0)

    # Position prior excludes A and excludes target week: B/C weeks 1-4 = 8 E rows.
    assert row["prior_position_rows"] == 8
    assert row["baseline_opp_prob"] == pytest.approx(0.0)
    assert row["baseline_eff_prob"] == pytest.approx(1.0)

    # Four player observations + four position pseudo-observations => 50/50 O/E.
    assert row["candidate_opp_prob"] == pytest.approx(0.5)
    assert row["candidate_eff_prob"] == pytest.approx(0.5)
    assert row["candidate_mixed_prob"] == pytest.approx(0.0)


def test_target_row_never_enters_its_own_history():
    panel = prepare_panel(_panel(), "TE", (2024, 2025))
    out = build_predictions(panel, "TE")
    assert (out["prior_player_max_week"] < out["week"]).all()


def test_decomposition_identity_fails_closed():
    bad = _panel()
    bad.loc[0, "current_total_error"] = 999.0
    with pytest.raises(RuntimeError, match="decomposition identity"):
        prepare_panel(bad, "TE", (2024, 2025))
