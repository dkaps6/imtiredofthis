import numpy as np
import pandas as pd
import pytest

from scripts.backtest.rb_workhorse_gate_v1_classifier import (
    CONFIRMED,
    CUTOFF_GRID,
    INSUFFICIENT_EVIDENCE,
    NOT_QUALIFIED,
    PRECISION_UNDEFINED_DETAIL,
    confirmation_report,
    fit_rotation_model,
    predict_probabilities,
    run_rotation,
    select_cutoff,
)
from scripts.backtest.rb_workhorse_gate_v1_features import FEATURE_COLUMNS


def _separable_rows(n, positive_rate, seed):
    """Synthetic feature rows where active_top_prior3_rb_share is strongly
    separable by label and the remaining 12 columns are unrelated noise --
    enough signal for a real (not degenerate) logistic fit without
    fabricating anything football-real.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(round(n * positive_rate))
    labels = np.array([1] * n_pos + [0] * (n - n_pos))
    rng.shuffle(labels)
    data = {}
    top_share = np.where(labels == 1, rng.normal(0.75, 0.05, n), rng.normal(0.25, 0.05, n))
    data["active_top_prior3_rb_share"] = np.clip(top_share, 0.01, 0.99)
    for col in FEATURE_COLUMNS[1:]:
        data[col] = rng.normal(10.0, 1.0, n)
    rows = pd.DataFrame(data)
    return rows, pd.Series(labels)


def test_fit_rotation_model_happy_path():
    rows, labels = _separable_rows(60, 0.3, seed=1)
    fitted = fit_rotation_model(rows, labels)
    assert set(fitted["coefficients"]) == set(FEATURE_COLUMNS)
    assert fitted["n_fit_events"] == 60
    assert fitted["n_fit_positive"] == int(labels.sum())


def test_fit_rotation_model_fails_closed_on_single_class():
    rows, labels = _separable_rows(20, 0.3, seed=2)
    labels = pd.Series([0] * len(labels))
    with pytest.raises(RuntimeError, match="both classes"):
        fit_rotation_model(rows, labels)


def test_fit_rotation_model_fails_closed_on_nonfinite_feature():
    rows, labels = _separable_rows(20, 0.3, seed=3)
    rows.loc[0, "mc_projected_plays"] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        fit_rotation_model(rows, labels)


def test_predict_probabilities_returns_values_in_unit_interval():
    rows, labels = _separable_rows(60, 0.3, seed=4)
    fitted = fit_rotation_model(rows, labels)
    probs = predict_probabilities(fitted, rows)
    assert len(probs) == 60
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_select_cutoff_picks_higher_cutoff_on_exact_tie():
    # Two labels, all probabilities identical -> every cutoff in the grid
    # produces the exact same predictions and therefore the exact same
    # F0.5 -- the tie must resolve to the highest cutoff in the grid.
    probs = np.array([0.95, 0.95, 0.95, 0.95])
    labels = np.array([1, 1, 0, 0])
    result = select_cutoff(probs, labels)
    assert result["selected_cutoff"] == max(CUTOFF_GRID)
    assert len(result["cutoff_table"]) == len(CUTOFF_GRID)


def test_select_cutoff_zero_division_semantics_when_no_predicted_positives():
    # No probability clears even the lowest grid cutoff -> every row is
    # zero_division=0 (f0.5=0.0), never raises.
    probs = np.array([0.10, 0.20, 0.05, 0.15])
    labels = np.array([1, 0, 1, 0])
    result = select_cutoff(probs, labels)
    assert all(r["n_predicted_positive"] == 0 for r in result["cutoff_table"])
    assert all(r["f0.5"] == 0.0 for r in result["cutoff_table"])
    assert result["selected_cutoff"] == max(CUTOFF_GRID)  # all-zero tie -> higher cutoff


def test_select_cutoff_favors_high_precision_cutoff():
    # A cutoff that predicts only the true positives (high precision, lower
    # recall) should beat a looser cutoff that also catches false positives,
    # since F0.5 weights precision more heavily.
    probs = np.array([0.92, 0.72, 0.68, 0.30, 0.20, 0.10])
    labels = np.array([1, 1, 0, 1, 0, 0])
    result = select_cutoff(probs, labels)
    row_90 = next(r for r in result["cutoff_table"] if r["cutoff"] == 0.90)
    row_50 = next(r for r in result["cutoff_table"] if r["cutoff"] == 0.50)
    assert row_90["precision"] >= row_50["precision"]


def test_confirmation_report_fails_closed_on_adequacy():
    probs = np.array([0.9] * 5 + [0.1] * 5)
    labels = np.array([1] * 5 + [0] * 5)
    result = confirmation_report(probs, labels, cutoff=0.5)
    assert result["disposition"] == INSUFFICIENT_EVIDENCE
    assert result["adequacy_pass"] is False


def test_confirmation_report_precision_undefined_when_zero_predicted_positives():
    n = 40
    labels = np.array([1] * 12 + [0] * (n - 12))
    probs = np.array([0.3] * n)  # never clears any grid cutoff
    result = confirmation_report(probs, labels, cutoff=0.90)
    assert result["adequacy_pass"] is True
    assert result["disposition"] == NOT_QUALIFIED
    assert result["detail"] == PRECISION_UNDEFINED_DETAIL
    assert result["precision"] is None


def test_confirmation_report_confirms_on_strong_separable_signal():
    n = 40
    n_pos = 12
    labels = np.array([1] * n_pos + [0] * (n - n_pos))
    # Perfect separation: positives all above cutoff, negatives all below.
    probs = np.array([0.95] * n_pos + [0.05] * (n - n_pos))
    result = confirmation_report(probs, labels, cutoff=0.50, sportsbook_inputs_used=0)
    assert result["disposition"] == CONFIRMED
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_confirmation_report_fails_closed_on_nonzero_sportsbook_inputs():
    n = 40
    n_pos = 12
    labels = np.array([1] * n_pos + [0] * (n - n_pos))
    probs = np.array([0.95] * n_pos + [0.05] * (n - n_pos))
    result = confirmation_report(probs, labels, cutoff=0.50, sportsbook_inputs_used=1)
    assert result["disposition"] == NOT_QUALIFIED
    assert result["gates"]["sportsbook_inputs_zero"] is False


def test_run_rotation_end_to_end_confirms_on_strong_signal():
    fit_rows, fit_labels = _separable_rows(80, 0.3, seed=10)
    cutoff_rows, cutoff_labels = _separable_rows(40, 0.25, seed=11)
    confirm_rows, confirm_labels = _separable_rows(40, 0.3, seed=12)
    result = run_rotation(
        fit_rows=fit_rows, fit_labels=fit_labels,
        cutoff_rows=cutoff_rows, cutoff_labels=cutoff_labels,
        confirm_rows=confirm_rows, confirm_labels=confirm_labels,
    )
    assert result["cutoff_selection"]["selected_cutoff"] in CUTOFF_GRID
    assert result["confirmation"]["disposition"] == CONFIRMED
    assert result["fit"]["n_fit_events"] == 80


def test_run_rotation_never_refits_on_cutoff_or_confirm_data():
    fit_rows, fit_labels = _separable_rows(80, 0.3, seed=20)
    cutoff_rows, cutoff_labels = _separable_rows(40, 0.25, seed=21)
    confirm_rows, confirm_labels = _separable_rows(40, 0.3, seed=22)
    result = run_rotation(
        fit_rows=fit_rows, fit_labels=fit_labels,
        cutoff_rows=cutoff_rows, cutoff_labels=cutoff_labels,
        confirm_rows=confirm_rows, confirm_labels=confirm_labels,
    )
    fitted = result["fitted_objects"]
    # Coefficients must depend only on the fit rows -- refitting on the same
    # fit rows again must reproduce identical coefficients exactly.
    refit = fit_rotation_model(fit_rows, fit_labels)
    assert refit["coefficients"] == fitted["coefficients"]
