from __future__ import annotations

import pandas as pd

from scripts.research.audit_player_opportunity_share_stability_2023_holdout_v1 import (
    TARGET_SEASON,
)
from scripts.research.audit_player_opportunity_share_stability_v1 import (
    build_panel,
    score_cell,
)


def _fixture():
    rows=[]
    for i in range(1,9):
        rows.append({
            "season":TARGET_SEASON,
            "week":i,
            "game_id":f"g{i}",
            "team":"IND",
            "opponent":"HOU",
            "player_identity_key":"gsis:test-rb",
            "player":"Test Back",
            "position":"RB",
            "tgt_share_game":0.05 + 0.01*(i%2),
            "rush_share_game":0.20 + 0.05*(i%3),
        })
    return pd.DataFrame(rows)


def test_holdout_target_season_is_2023():
    assert TARGET_SEASON==2023


def test_holdout_reuses_strictly_prior_parent_panel_semantics():
    panel,integrity=build_panel(_fixture())
    row=panel.loc[
        panel["cell"].eq("RB_CARRY_SHARE")
        & panel["week"].eq(5)
    ].iloc[0]
    assert row["season"]==2023
    assert row["prior_games"]==4
    assert row["prior_feature_max_week"]==4
    assert integrity["same_future_feature_violations"]==0
    assert integrity["same_team_feature_violations"]==0


def test_parent_score_cell_accepts_2023_holdout():
    panel,_=build_panel(_fixture())
    rec=score_cell(panel,2023,"RB_CARRY_SHARE")
    assert rec["season"]==2023
    assert rec["cell"]=="RB_CARRY_SHARE"
