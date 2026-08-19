import numpy as np
import pandas as pd
import pytest

from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline


def _consensus():
    rows = [
        # Veteran WR with strong prior and two current games.
        dict(player="Alpha WR", team="IND", season=2026, position="LWR", prior_games=17, current_games=2,
             tgt_share_prior=.28, tgt_share_current=.34, rush_share_prior=.01, rush_share_current=.01,
             route_rate_prior=.88, route_rate_current=.91, receptions_per_target_prior=.66, receptions_per_target_current=.70,
             yprr_prior=2.10, yprr_current=2.40, ypt_prior=9.0, ypt_current=10.0,
             ypc_prior=5.0, ypc_current=6.0, ypa_prior=np.nan, ypa_current=np.nan),
        # Other WR establishes a reasonable population prior.
        dict(player="Beta WR", team="HOU", season=2026, position="RWR", prior_games=17, current_games=0,
             tgt_share_prior=.14, tgt_share_current=np.nan, rush_share_prior=.00, rush_share_current=np.nan,
             route_rate_prior=.72, route_rate_current=np.nan, receptions_per_target_prior=.60, receptions_per_target_current=np.nan,
             yprr_prior=1.30, yprr_current=np.nan, ypt_prior=7.0, ypt_current=np.nan,
             ypc_prior=4.0, ypc_current=np.nan, ypa_prior=np.nan, ypa_current=np.nan),
        # Rookie has no player history and must fall back to population prior.
        dict(player="Rookie WR", team="JAX", season=2026, position="SWR", prior_games=0, current_games=0,
             tgt_share_prior=np.nan, tgt_share_current=np.nan, rush_share_prior=np.nan, rush_share_current=np.nan,
             route_rate_prior=np.nan, route_rate_current=np.nan, receptions_per_target_prior=np.nan, receptions_per_target_current=np.nan,
             yprr_prior=np.nan, yprr_current=np.nan, ypt_prior=np.nan, ypt_current=np.nan,
             ypc_prior=np.nan, ypc_current=np.nan, ypa_prior=np.nan, ypa_current=np.nan),
        # QBs establish YPA pool.
        dict(player="QB One", team="BUF", season=2026, position="QB", prior_games=17, current_games=3,
             tgt_share_prior=0.0, tgt_share_current=0.0, rush_share_prior=.18, rush_share_current=.20,
             route_rate_prior=np.nan, route_rate_current=np.nan, receptions_per_target_prior=0.0, receptions_per_target_current=0.0,
             yprr_prior=np.nan, yprr_current=np.nan, ypt_prior=np.nan, ypt_current=np.nan,
             ypc_prior=5.2, ypc_current=5.5, ypa_prior=7.5, ypa_current=8.1),
        dict(player="QB Two", team="MIA", season=2026, position="QB", prior_games=17, current_games=0,
             tgt_share_prior=0.0, tgt_share_current=np.nan, rush_share_prior=.05, rush_share_current=np.nan,
             route_rate_prior=np.nan, route_rate_current=np.nan, receptions_per_target_prior=0.0, receptions_per_target_current=np.nan,
             yprr_prior=np.nan, yprr_current=np.nan, ypt_prior=np.nan, ypt_current=np.nan,
             ypc_prior=3.0, ypc_current=np.nan, ypa_prior=7.0, ypa_current=np.nan),
    ]
    return pd.DataFrame(rows)


def test_posterior_shrinks_current_evidence_instead_of_copying_it():
    out = build_bayesian_baseline(_consensus())
    alpha = out.loc[out.player.eq("Alpha WR")].iloc[0]
    assert alpha.bayes_evidence_state == "prior+current"

    # The posterior combines three sources of evidence: the WR population prior,
    # capped player-prior evidence, and current-season evidence. It is therefore
    # allowed to sit below the player's .28 prior when the WR population prior is
    # lower; the important contract is that the current .34 is not copied blindly.
    wr_population_tgt_share = (.28 * 17 + .14 * 17) / (17 + 17)
    expected_tgt_share = (3.0 * wr_population_tgt_share + 6.0 * .28 + 2.0 * .34) / 11.0
    assert alpha.bayes_tgt_share == pytest.approx(expected_tgt_share)
    assert wr_population_tgt_share < alpha.bayes_tgt_share < .34

    # Same principle for YPT: population + player prior + current evidence.
    wr_population_ypt = (9.0 * 17 + 7.0 * 17) / (17 + 17)
    expected_ypt = (5.0 * wr_population_ypt + 8.0 * 9.0 + 2.0 * 10.0) / 15.0
    assert alpha.bayes_ypt == pytest.approx(expected_ypt)
    assert wr_population_ypt < alpha.bayes_ypt < 10.0
    assert alpha.bayes_tgt_share_effective_n > alpha.current_games


def test_rookie_uses_position_population_prior_without_fake_current_data():
    out = build_bayesian_baseline(_consensus())
    rookie = out.loc[out.player.eq("Rookie WR")].iloc[0]
    assert rookie.bayes_evidence_state == "position_prior_only"
    assert 0.0 < rookie.bayes_tgt_share < 1.0
    assert np.isfinite(rookie.bayes_ypt)
    assert rookie.current_games == 0


def test_qb_ypa_uses_qb_pool_and_current_evidence():
    out = build_bayesian_baseline(_consensus())
    qb = out.loc[out.player.eq("QB One")].iloc[0]
    assert 7.0 < qb.bayes_ypa < 8.1
    assert qb.bayes_position_group == "QB"


def test_metrics_join_is_many_books_to_one_player_posterior():
    post = build_bayesian_baseline(_consensus())
    metrics = pd.DataFrame([
        {"player": "Alpha WR", "team": "IND", "market": "player_receiving_yards", "line": 80.5},
        {"player": "Alpha WR", "team": "IND", "market": "player_receptions", "line": 6.5},
    ])
    out = apply_bayesian_to_metrics(metrics, post)
    assert len(out) == 2
    assert out.bayes_applied.sum() == 2
    assert out.bayes_tgt_share.nunique() == 1
