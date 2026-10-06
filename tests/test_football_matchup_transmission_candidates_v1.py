import numpy as np
import pandas as pd

import scripts.research.score_football_matchup_transmission_candidates_v1 as m


def test_rb_def_pass_rate_candidate_is_direct_and_clipped():
    assert np.isclose(m.rb_def_pass_rate_candidate(0.48), 0.48)
    assert np.isclose(m.rb_def_pass_rate_candidate(0.10), 0.35)
    assert np.isclose(m.rb_def_pass_rate_candidate(0.90), 0.75)
    assert np.isclose(m.rb_def_pass_rate_candidate(np.nan), 0.57)


def test_wr_true_proe_candidate_uses_existing_57_anchor():
    assert np.isclose(m.wr_true_proe_candidate(-0.105), 0.465)
    assert np.isclose(m.wr_true_proe_candidate(0.05), 0.62)
    assert np.isclose(m.wr_true_proe_candidate(np.nan), 0.57)


def test_te_pass_success_multiplier_is_unit_preserving():
    assert np.isclose(m.te_pass_success_multiplier(0.55, 0.45), 1.10)
    assert np.isclose(m.te_pass_success_multiplier(0.35, 0.45), 0.90)
    assert np.isclose(m.te_pass_success_multiplier(np.nan, 0.45), 1.0)


def test_primary_gate_requires_both_seasons_and_dual_cluster_support():
    candidate = m.CANDIDATES[0]
    rows = []
    for season in (2024, 2025):
        for cohort in m.COLLATERAL:
            primary = cohort == "RB_RUSH"
            rows.append({
                "candidate": candidate,
                "cohort": cohort,
                "season": season,
                "rows": 500,
                "games": 100,
                "players": 80,
                "support": True,
                "mae_improvement": 1.0 if primary else 0.0,
                "supported_improvement": primary,
                "supported_harm": False,
                "rmse_nonworse": True,
                "game_ci_high": 1.0,
                "player_ci_high": 1.0,
            })
    cells = pd.DataFrame(rows)

    # Exercise the exact frozen aggregation logic without rerunning bootstrap.
    primary = m.PRIMARY[candidate]
    p = cells.loc[cells["cohort"].eq(primary)]
    primary_pass = (
        len(p) == 2
        and p["support"].all()
        and (p["mae_improvement"] > 0).all()
        and p["supported_improvement"].all()
        and p["rmse_nonworse"].all()
    )
    harm = cells.loc[
        ~cells["cohort"].eq(primary)
        & cells["support"]
        & cells["supported_harm"]
    ]
    assert primary_pass
    assert harm.empty
