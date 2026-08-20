import numpy as np
import pandas as pd

from scripts.backtest.trace_canonical_rushing_path import _selected_player_rows, _stage_metrics


def test_selected_player_rows_mirrors_simulator_normalization():
    frame = pd.DataFrame([
        {"event_id":"A|B","team":"A","player_clean_key":"p1","market":"rush_att","rules_rush_share":0.60,"rules_plays_est":60.0,"rules_pass_rate":0.50},
        {"event_id":"A|B","team":"A","player_clean_key":"p1","market":"rush_yards","rules_rush_share":0.60,"rules_plays_est":60.0,"rules_pass_rate":0.50},
        {"event_id":"A|B","team":"A","player_clean_key":"p2","market":"rush_att","rules_rush_share":0.50,"rules_plays_est":60.0,"rules_pass_rate":0.50},
    ])
    out = _selected_player_rows(frame)
    assert len(out) == 2
    assert np.isclose(out.sim_team_player_share_sum.iloc[0], 0.95)
    assert np.isclose(out.sim_team_expected_rushes_deterministic.iloc[0], 30.0)
    assert out.sim_used_rush_share.sum() == pytest.approx(0.95)


def test_stage_metrics_reports_correlation_and_error():
    x = pd.DataFrame({
        "actual":[1.0,2.0,3.0],
        "component_mc_proj":[1.0,2.0,4.0],
        "rebuilt_mc_proj":[1.0,2.0,4.0],
        "direct_lookup_mean":[1.0,2.0,4.0],
        "sim_deterministic_player_carries":[1.0,2.0,4.0],
    })
    s = _stage_metrics(x)
    row = s.loc[s.stage.eq("component_mc_proj")].iloc[0]
    assert row.n == 3
    assert row.mae == pytest.approx(1.0/3.0)
    assert row.correlation > 0.9


import pytest
