import pandas as pd
import scripts.research.audit_prism_pass_catcher_qb_regime_target_depth_state_v1_source as m


def test_before_mask_excludes_target_week():
    x = pd.DataFrame({"season": [2023, 2024, 2024], "week": [18, 4, 5]})
    assert m.before_mask(x, 2024, 5).tolist() == [True, True, False]


def test_latest_games_keeps_only_latest_eight():
    x = pd.DataFrame({
        "season": [2024] * 10,
        "week": list(range(1, 11)),
        "game_id": [f"g{i}" for i in range(1, 11)],
        "air_yards": [1.0] * 10,
    })
    z = m.latest_games(x, 8)
    assert sorted(z["week"].tolist()) == list(range(3, 11))


def test_event_summary_counts_same_qb_history():
    x = pd.DataFrame({
        "season": [2024, 2024, 2024, 2024],
        "week": [1, 1, 2, 2],
        "game_id": ["g1", "g1", "g2", "g2"],
        "air_yards": [5.0, 15.0, 10.0, 20.0],
    })
    s = m.event_summary(x)
    assert s["games"] == 2
    assert s["finite_air_targets"] == 4
    assert abs(s["mean_air_yards"] - 12.5) < 1e-12
