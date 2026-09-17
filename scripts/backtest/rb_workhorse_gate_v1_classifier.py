"""RB Workhorse-Transition-Gate V1 -- frozen two-rotation classifier pipeline.

Implements the frozen classifier family (plan Section 8), cutoff protocol
(Section 9), and confirmation gates (Section 10), as amended by the
two-rotation chronology (Section 16). One rotation's fit/cutoff/confirm
sequence is fully self-contained in ``run_rotation`` -- Rotation A and
Rotation B are two independent calls with different (fit_rows, cutoff_rows,
confirm_rows) inputs, never sharing a fitted model or cutoff.

Nothing here selects features, tunes hyperparameters, or changes the
classifier family -- those are frozen constants matching the plan exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.backtest.rb_workhorse_gate_v1_features import FEATURE_COLUMNS

CUTOFF_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
ADEQUACY_MIN_EVENTS = 30
ADEQUACY_MIN_POSITIVE = 10
PRECISION_PREVALENCE_MARGIN = 0.10
PRECISION_FLOOR = 0.60
RECALL_FLOOR = 0.25
ROC_AUC_FLOOR = 0.60
RANDOM_STATE = 42

CONFIRMED = "RB_WORKHORSE_TRANSITION_GATE_V1_CONFIRMED_EARLY_OOS"
NOT_QUALIFIED = "RB_WORKHORSE_TRANSITION_GATE_V1_NOT_QUALIFIED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
PRECISION_UNDEFINED_DETAIL = "PRECISION_UNDEFINED_NO_PREDICTED_POSITIVES"


def fit_rotation_model(fit_rows: pd.DataFrame, fit_labels: pd.Series) -> dict:
    """Fit the frozen StandardScaler + L2 logistic regression once on a
    rotation's own fit years only (plan Section 8). Returns the fitted
    objects plus their serializable parameters for the evidence artifact.
    """
    x = fit_rows[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = fit_labels.to_numpy(dtype=int)
    if not np.isfinite(x).all():
        raise RuntimeError("fit_rotation_model: non-finite feature value in fit rows")
    if len(set(y.tolist())) < 2:
        raise RuntimeError("fit_rotation_model: fit sample must contain both classes")

    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        fit_intercept=True,
        max_iter=1000,
        random_state=RANDOM_STATE,
    ).fit(scaler.transform(x), y)

    return {
        "scaler": scaler,
        "classifier": clf,
        "n_fit_events": int(len(fit_rows)),
        "n_fit_positive": int(y.sum()),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": dict(zip(FEATURE_COLUMNS, clf.coef_[0].tolist())),
        "intercept": float(clf.intercept_[0]),
    }


def predict_probabilities(fitted: dict, rows: pd.DataFrame) -> np.ndarray:
    """Apply a rotation's already-fitted, frozen scaler/classifier -- never
    refit -- to any other year's feature rows for that same rotation.
    """
    x = rows[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise RuntimeError("predict_probabilities: non-finite feature value")
    xs = fitted["scaler"].transform(x)
    return fitted["classifier"].predict_proba(xs)[:, 1]


def select_cutoff(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """F0.5-argmax over the frozen grid (plan Section 9). zero_division=0
    semantics for any cutoff with zero predicted positives. Ties -> the
    higher cutoff, resolved deterministically via the (f0.5, cutoff) sort
    key rather than relying on iteration order.
    """
    y = labels.astype(int)
    table = []
    for cutoff in CUTOFF_GRID:
        preds = (probabilities >= cutoff).astype(int)
        precision, recall, f_beta, _ = precision_recall_fscore_support(
            y, preds, average="binary", beta=0.5, zero_division=0, pos_label=1
        )
        table.append(
            {
                "cutoff": float(cutoff),
                "precision": float(precision),
                "recall": float(recall),
                "f0.5": float(f_beta),
                "n_predicted_positive": int(preds.sum()),
            }
        )
    best = max(table, key=lambda r: (r["f0.5"], r["cutoff"]))
    return {"cutoff_table": table, "selected_cutoff": best["cutoff"]}


def confirmation_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    cutoff: float,
    *,
    sportsbook_inputs_used: int = 0,
) -> dict:
    """Score the frozen confirmation gates (plan Section 10) on a rotation's
    own confirmation year, using its own unchanged fit-year model and its
    own cutoff-year-selected cutoff.
    """
    y = labels.astype(int)
    n = int(len(y))
    n_positive = int(y.sum())
    prevalence = (n_positive / n) if n else None
    adequacy_pass = n >= ADEQUACY_MIN_EVENTS and n_positive >= ADEQUACY_MIN_POSITIVE

    preds = (probabilities >= cutoff).astype(int)
    n_predicted_positive = int(preds.sum())

    base = {
        "n": n,
        "n_positive": n_positive,
        "prevalence": prevalence,
        "adequacy_pass": adequacy_pass,
        "cutoff": float(cutoff),
        "n_predicted_positive": n_predicted_positive,
        "sportsbook_inputs_used": int(sportsbook_inputs_used),
    }

    if not adequacy_pass:
        return {**base, "disposition": INSUFFICIENT_EVIDENCE, "detail": "adequacy_floor_not_met", "gates": {}}

    if n_predicted_positive == 0:
        return {
            **base,
            "disposition": NOT_QUALIFIED,
            "detail": PRECISION_UNDEFINED_DETAIL,
            "precision": None,
            "recall": None,
            "roc_auc": None,
            "pr_auc": None,
            "gates": {"precision_vs_prevalence_margin": False, "precision_floor": False},
        }

    precision, recall, _, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0, pos_label=1
    )
    roc_auc = float(roc_auc_score(y, probabilities)) if len(set(y.tolist())) > 1 else None
    pr_auc = float(average_precision_score(y, probabilities))

    gates = {
        "precision_vs_prevalence_margin": bool(
            prevalence is not None and precision >= prevalence + PRECISION_PREVALENCE_MARGIN
        ),
        "precision_floor": bool(precision >= PRECISION_FLOOR),
        "recall_floor": bool(recall >= RECALL_FLOOR),
        "roc_auc_floor": bool(roc_auc is not None and roc_auc > ROC_AUC_FLOOR),
        "pr_auc_vs_prevalence": bool(prevalence is not None and pr_auc > prevalence),
        "sportsbook_inputs_zero": int(sportsbook_inputs_used) == 0,
    }
    passed = all(gates.values())
    return {
        **base,
        "disposition": CONFIRMED if passed else NOT_QUALIFIED,
        "detail": None if passed else "gate_failure",
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "gates": gates,
    }


def run_rotation(
    *,
    fit_rows: pd.DataFrame,
    fit_labels: pd.Series,
    cutoff_rows: pd.DataFrame,
    cutoff_labels: pd.Series,
    confirm_rows: pd.DataFrame,
    confirm_labels: pd.Series,
    sportsbook_inputs_used: int = 0,
) -> dict:
    """Full fit -> cutoff-select -> confirm sequence for one rotation.
    Never touches the other rotation's data or model.
    """
    fitted = fit_rotation_model(fit_rows, fit_labels)

    cutoff_probs = predict_probabilities(fitted, cutoff_rows)
    cutoff_result = select_cutoff(cutoff_probs, cutoff_labels.to_numpy(dtype=int))
    selected_cutoff = cutoff_result["selected_cutoff"]

    confirm_probs = predict_probabilities(fitted, confirm_rows)
    confirm_result = confirmation_report(
        confirm_probs, confirm_labels.to_numpy(dtype=int), selected_cutoff,
        sportsbook_inputs_used=sportsbook_inputs_used,
    )

    return {
        "fit": {k: v for k, v in fitted.items() if k not in ("scaler", "classifier")},
        "cutoff_selection": cutoff_result,
        "confirmation": confirm_result,
        "fitted_objects": fitted,  # scaler/classifier kept for optional downstream transport use
    }
