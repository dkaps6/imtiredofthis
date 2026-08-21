import numpy as np
import pandas as pd

from scripts.simulation_v2 import lookup, simulate


def _metrics():
    return pd.DataFrame([
        {"event_id": "g1", "player": "QB One", "player_clean_key": "qbone", "team": "BUF", "opponent": "NYJ", "position": "QB", "role": "QB1", "market": "player_pass_yds", "pace": 28.0, "proe": 0.02, "ypa": 7.5, "target_share": 0.0, "rush_share": 0.10, "ypc": 4.5, "offensive_td_rate": 0.2},
        {"event_id": "g1", "player": "WR One", "player_clean_key": "wrone", "team": "BUF", "opponent": "NYJ", "position": "WR", "role": "WR1", "market": "player_reception_yds", "pace": 28.0, "proe": 0.02, "target_share": 0.28, "rush_share": 0.0, "ypt": 9.0, "receptions_per_target": 0.68, "offensive_td_rate": 0.55, "rz_share": 0.30},
        {"event_id": "g1", "player": "WR Two", "player_clean_key": "wrtwo", "team": "BUF", "opponent": "NYJ", "position": "WR", "role": "WR2", "market": "player_reception_yds", "pace": 28.0, "proe": 0.02, "target_share": 0.20, "rush_share": 0.0, "ypt": 8.0, "receptions_per_target": 0.64, "offensive_td_rate": 0.35, "rz_share": 0.20},
    ])


def test_real_simulation_produces_distribution_not_single_normal_formula():
    df = _metrics()
    result = simulate(df, iterations=3000, seed=7)
    wr = lookup(result, df.iloc[1], "player_reception_yds")
    assert wr is not None
    assert len(wr) == 3000
    assert np.std(wr) > 0
    assert len(np.unique(np.round(wr, 2))) > 100


def test_qb_and_receiver_share_game_environment():
    df = _metrics()
    result = simulate(df, iterations=5000, seed=11)
    qb = lookup(result, df.iloc[0], "player_pass_yds")
    wr = lookup(result, df.iloc[1], "player_reception_yds")
    assert qb is not None and wr is not None
    corr = np.corrcoef(qb, wr)[0, 1]
    assert corr > 0.05


def test_target_competition_creates_finite_volume():
    df = _metrics()
    result = simulate(df, iterations=3000, seed=19)
    wr1 = lookup(result, df.iloc[1], "player_receptions")
    wr2 = lookup(result, df.iloc[2], "player_receptions")
    assert wr1 is not None and wr2 is not None
    # Both players receive finite opportunity from the same team pass volume.
    assert np.percentile(wr1 + wr2, 99) < 45


def test_keyed_allocation_trace_matches_exact_rush_lookup_without_changing_results():
    df = _metrics()
    baseline = simulate(df, iterations=2500, seed=31)
    trace = []
    traced = simulate(df, iterations=2500, seed=31, allocation_trace=trace)
    assert len(trace) == 3
    by_key = {row["player_clean_key"]: row for row in trace}
    for _, row in df.iterrows():
        base = lookup(baseline, row, "rush_att")
        got = lookup(traced, row, "rush_att")
        assert base is not None and got is not None
        assert np.array_equal(base, got)
        assert np.isclose(got.mean(), by_key[row.player_clean_key]["realized_multinomial_mean_carries"])
        assert abs(
            by_key[row.player_clean_key]["realized_multinomial_mean_carries"]
            - by_key[row.player_clean_key]["expected_carries_from_final_probability"]
        ) < 0.15
