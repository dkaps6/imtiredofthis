import pandas as pd
import scripts.research.audit_prism_pass_catcher_qb_regime_target_cpoe_state_v1_source as m


def test_before_mask_excludes_target_week():
    x = pd.DataFrame({"season": [2023, 2024, 2024], "week": [18, 4, 5]})
    assert m.before_mask(x, 2024, 5).tolist() == [True, True, False]


def test_latest_target_bearing_games_does_not_backfill_for_null_cpoe():
    rows = []
    for week in range(1, 10):
        rows.append({
            "season": 2024,
            "week": week,
            "game_id": f"g{week}",
            "cpoe": None if week == 9 else float(week),
        })
    x = pd.DataFrame(rows)
    z = m.latest_target_bearing_games(x, 8)
    assert sorted(z["week"].tolist()) == list(range(2, 10))
    assert 1 not in z["week"].tolist()
    assert 9 in z["week"].tolist()


def test_cpoe_summary_counts_games_before_valid_events():
    x = pd.DataFrame({
        "season": [2024, 2024, 2024, 2024],
        "week": [1, 1, 2, 3],
        "game_id": ["g1", "g1", "g2", "g3"],
        "cpoe": [1.0, None, 3.0, None],
    })
    s = m.cpoe_summary(x)
    assert s["target_bearing_games"] == 3
    assert s["finite_cpoe_targets"] == 2
    assert abs(s["mean_cpoe"] - 2.0) < 1e-12
