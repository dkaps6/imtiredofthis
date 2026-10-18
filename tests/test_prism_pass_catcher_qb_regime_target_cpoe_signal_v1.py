import pandas as pd
import scripts.research.evaluate_prism_pass_catcher_qb_regime_target_cpoe_signal_v1 as m


def test_bool_parser_handles_csv_strings():
    x = pd.Series(["True", "False", "1", "0"])
    assert m.bools(x).tolist() == [True, False, True, False]


def test_metrics_reports_structural_ties_separately():
    x = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "teamweek": ["t1", "t2", "t3"],
        "actual_target_cpoe": [0.0, 0.0, 0.0],
        "baseline_target_cpoe": [2.0, 2.0, 2.0],
        "candidate_target_cpoe": [2.0, 1.0, 3.0],
        "baseline_abs_error": [2.0, 2.0, 2.0],
        "candidate_abs_error": [2.0, 1.0, 3.0],
        "error_delta": [0.0, -1.0, 1.0],
        "actual_finite_cpoe_targets": [3, 4, 5],
    })
    out = m.metrics(x)
    assert abs(out["tie_rate"] - (1/3)) < 1e-12
    assert out["non_tied_rows"] == 2
    assert abs(out["candidate_closer_rate_non_tied"] - 0.5) < 1e-12


def test_cluster_bootstrap_negative_delta_stays_negative():
    x = pd.DataFrame({
        "cluster": ["a", "a", "b", "b", "c", "c"],
        "error_delta": [-1.0, -2.0, -1.5, -2.5, -0.5, -1.0],
    })
    lo, hi = m.bootstrap_cluster_ci(x, "cluster", reps=200, seed=1)
    assert lo < 0
    assert hi < 0
