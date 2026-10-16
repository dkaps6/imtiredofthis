import pandas as pd
import pytest

from scripts.research.evaluate_player_target_share_trajectory_oos_fold_integration_v1 import (
    TOL,
    apply_te_trajectory,
    apply_wr_trajectory,
)


def _base(position):
    rows = [
        dict(season=2024, week=5, event_id="A|B", team="A",
             player_clean_key="p1", player="P1", baseline_targets=6.0,
             baseline_rec_yards=60.0, trajectory_delta=0.10,
             trajectory_available=True, feature_max_week=4,
             position_group=position, is_wr1=(position == "WR")),
        dict(season=2024, week=5, event_id="A|B", team="A",
             player_clean_key="p2", player="P2", baseline_targets=3.0,
             baseline_rec_yards=30.0, trajectory_delta=0.20,
             trajectory_available=True, feature_max_week=4,
             position_group=position, is_wr1=False),
        dict(season=2024, week=5, event_id="A|B", team="A",
             player_clean_key="p3", player="P3", baseline_targets=1.0,
             baseline_rec_yards=10.0, trajectory_delta=-0.10,
             trajectory_available=True, feature_max_week=4,
             position_group=position, is_wr1=False),
    ]
    return pd.DataFrame(rows)


def test_wr_anchor_and_wr2_pool_are_exactly_conserved():
    x = _base("WR")
    y, audit = apply_wr_trajectory(x)
    assert y.loc[y.player_clean_key.eq("p1"), "shadow_targets"].iloc[0] == 6.0
    assert abs(y.loc[y.player_clean_key.isin(["p2", "p3"]), "shadow_targets"].sum() - 4.0) <= TOL
    assert abs(y.shadow_targets.sum() - 10.0) <= TOL
    assert audit.wr1_max_abs_gap.max() <= TOL
    assert audit.protected_pool_max_abs_gap.max() <= TOL


def test_te_room_pool_is_exactly_conserved():
    x = _base("TE")
    y, audit = apply_te_trajectory(x)
    assert abs(y.shadow_targets.sum() - 10.0) <= TOL
    assert audit.room_pool_max_abs_gap.max() <= TOL
    assert y.loc[y.player_clean_key.eq("p2"), "shadow_targets"].iloc[0] > 3.0


def test_zero_deltas_are_identity_transform():
    x = _base("TE")
    x["trajectory_delta"] = 0.0
    y, _ = apply_te_trajectory(x)
    assert (y["shadow_targets"] - y["baseline_targets"]).abs().max() <= TOL


def test_wr_requires_exactly_one_anchor_per_room():
    x = _base("WR")
    x["is_wr1"] = False
    with pytest.raises(RuntimeError, match="exactly one"):
        apply_wr_trajectory(x)
