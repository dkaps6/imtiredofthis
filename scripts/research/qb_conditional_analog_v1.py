#!/usr/bin/env python3
"""QB Conditional Historical-Analog Reliability V1 — leakage-safe mechanism.

Implements exactly the frozen design in
docs/research/QB_CONDITIONAL_ANALOG_V1_PLAN.md: sections 1-8 (join, feature
selection, missingness exclusion, 2024-only frozen scaler, single Euclidean
distance metric, k=15 season-scoped neighbor rule with a feature-space-only
density gate, row-level evidence classes, and the architecture-level 2025
blind pass/fail gate).

This module never reads or ships real 2024/2025 outcome data. The two
upstream artifacts it is designed to join
(`PROMOTED_STACK_AUTHORITY_EXACT_VEGAS_DETAIL.csv`,
`m89_2024_2025_synthesis_trace.csv`) are ephemeral CI artifacts that are not
present in this repository or environment; `main()` accepts explicit file
paths and does nothing unless a caller supplies real files. Every function
below is dependency-injected (takes DataFrames in, returns DataFrames/arrays
out) so it can be — and is, in tests/test_qb_conditional_analog_v1.py —
proven correct entirely on synthetic fixture data without opening any real
outcome.

Research only. No production/projection/probability/EV/signal/pricing path
is touched by this module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

JOIN_KEYS = ["season", "week", "team", "player_clean_key"]

# Columns that reveal a realized/graded outcome (actual result, model-vs-line
# grading, ROI, or a downstream betting decision). These must never reach the
# join/missingness/scaler/distance/density path (Sections 1, 3-6) — only
# `attach_outcomes_at_confirmation_boundary` (Section 8's single sanctioned
# entry point) may ever merge them back in. Named defensively/broadly on
# purpose: any column matching this list on the real Vegas artifact is held
# back, whether or not this exact V1 mechanism ends up reading it.
OUTCOME_COLUMNS = [
    "actual_pass_yards", "actual", "projection", "football_synthesis",
    "vegas_line", "side", "best_side", "odds", "best_odds", "roi",
    "realized_roi", "realized_direction", "ev_roi", "best_ev_roi",
    "probability_edge", "decision", "bettable_now", "snapshot_signal",
]

DISAGREEMENT_FEATURES = ["component_sd", "component_range", "pred_attempts", "pred_ypa"]
PLAYER_PRIOR_FEATURES = ["qb_prior_attempts", "qb_prior_ypa"]
OPPONENT_PRIOR_FEATURES = [
    "def_pass_epa_allowed", "def_success_allowed", "def_ypa_allowed", "def_pass_rate_faced",
]
MARKET_SCRIPT_FEATURES = [
    "market_total", "market_spread", "market_abs_spread", "market_team_implied",
    "market_opp_implied", "market_is_underdog", "market_moneyline",
]
FEATURE_COLUMNS = (
    DISAGREEMENT_FEATURES + PLAYER_PRIOR_FEATURES + OPPONENT_PRIOR_FEATURES + MARKET_SCRIPT_FEATURES
)

K_NEIGHBORS = 15
DENSITY_PERCENTILE = 90.0
MIN_SUPPORTED_N_2025 = 40

EVIDENCE_SUPPORTED = "SUPPORTED"
EVIDENCE_DESCRIPTIVE_ONLY = "DESCRIPTIVE_ONLY"
EVIDENCE_NO_ANALOG_SUPPORT = "NO_ANALOG_SUPPORT"
EVIDENCE_NO_HISTORICAL_TRUST_SCORE = "NO_HISTORICAL_TRUST_SCORE"

DISPOSITION_SUPPORTED = "QB_CONDITIONAL_ANALOG_RELIABILITY_SUPPORTED"
DISPOSITION_NOT_ACTIONABLE = "NO_ACTIONABLE_QB_CONDITIONAL_ANALOG_RELIABILITY"


def split_population_and_outcomes(vegas_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outcome-boundary hardening: separate pre-outcome population/identity columns
    from realized-outcome columns before anything touches the join/scaler/distance/
    density path.

    Returns (population_df, outcomes_df). `population_df` carries JOIN_KEYS plus any
    non-outcome identity/context columns (opponent, game_id, player, position, market,
    benchmark_arm, etc.) and is the only object Sections 1 and 3-6 may ever see.
    `outcomes_df` carries JOIN_KEYS plus whichever OUTCOME_COLUMNS are present, and may
    only ever be reattached by `attach_outcomes_at_confirmation_boundary` — the single
    sanctioned Section 8 entry point.
    """
    present_outcome_cols = [c for c in OUTCOME_COLUMNS if c in vegas_df.columns]
    population_cols = [c for c in vegas_df.columns if c not in present_outcome_cols]
    population_df = vegas_df[population_cols].copy()
    outcomes_df = vegas_df[JOIN_KEYS + present_outcome_cols].copy()
    return population_df, outcomes_df


def attach_outcomes_at_confirmation_boundary(
    evidence_df: pd.DataFrame, outcomes_df: pd.DataFrame
) -> pd.DataFrame:
    """Section 8's single sanctioned entry point for opening realized outcomes.

    `evidence_df` is the already-computed, outcome-blind row table (join + missingness
    + scaler + distances + neighbors + density gate + evidence class), keyed on
    JOIN_KEYS. This is the only function in this module permitted to merge
    `outcomes_df` (from `split_population_and_outcomes`) back onto it, and it must be
    called at most once, immediately before `score_architecture_gate`.
    """
    missing = sorted(set(JOIN_KEYS) - set(evidence_df.columns))
    if missing:
        raise RuntimeError(f"evidence_df missing join keys: {missing}")
    return evidence_df.merge(outcomes_df, on=JOIN_KEYS, how="left", validate="one_to_one")


def join_vegas_and_features(vegas_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Section 1: inner join the grading population to the feature trace on JOIN_KEYS.

    `vegas_df` here must already be outcome-stripped (i.e. the `population_df` output
    of `split_population_and_outcomes`) for any real, non-test data path — this
    function itself is a generic merge and does not enforce that by inspection, so
    callers on real data must pass the pre-outcome population, never the raw artifact.
    """
    missing_v = sorted(set(JOIN_KEYS) - set(vegas_df.columns))
    missing_f = sorted(set(JOIN_KEYS) - set(features_df.columns))
    if missing_v:
        raise RuntimeError(f"vegas_df missing join keys: {missing_v}")
    if missing_f:
        raise RuntimeError(f"features_df missing join keys: {missing_f}")
    missing_feat_cols = sorted(set(FEATURE_COLUMNS) - set(features_df.columns))
    if missing_feat_cols:
        raise RuntimeError(f"features_df missing frozen feature columns: {missing_feat_cols}")
    feat_cols = JOIN_KEYS + [c for c in FEATURE_COLUMNS if c not in JOIN_KEYS]
    merged = vegas_df.merge(features_df[feat_cols], on=JOIN_KEYS, how="inner", validate="one_to_one")
    return merged


def exclude_missing(df: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Section 3: exclude-only missingness policy. Never impute."""
    cols = feature_cols or FEATURE_COLUMNS
    mask = df[cols].notna().all(axis=1)
    kept = df.loc[mask].reset_index(drop=True)
    dropped = int((~mask).sum())
    return kept, dropped


def fit_frozen_scaler(reference_df: pd.DataFrame, feature_cols: list[str] | None = None) -> StandardScaler:
    """Section 4: fit once on the reference pool only (2024). Caller freezes and reuses."""
    cols = feature_cols or FEATURE_COLUMNS
    scaler = StandardScaler()
    scaler.fit(reference_df[cols].to_numpy(dtype=float))
    return scaler


def standardize(df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str] | None = None) -> np.ndarray:
    cols = feature_cols or FEATURE_COLUMNS
    return scaler.transform(df[cols].to_numpy(dtype=float))


def pairwise_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Section 5: single frozen Euclidean distance metric. Returns shape (len(a), len(b))."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def k_nearest(distances: np.ndarray, k: int = K_NEIGHBORS) -> tuple[np.ndarray, np.ndarray]:
    """Section 6: indices and distances of the k nearest columns for each row."""
    k = min(k, distances.shape[1])
    order = np.argsort(distances, axis=1)[:, :k]
    dists = np.take_along_axis(distances, order, axis=1)
    return order, dists


def leave_one_out_density_threshold(
    reference_std: np.ndarray, k: int = K_NEIGHBORS, percentile: float = DENSITY_PERCENTILE
) -> float:
    """Section 6 quality gate: threshold derived from 2024 feature-space coverage only.

    No outcome or ROI data is touched to compute this threshold.
    """
    d = pairwise_euclidean(reference_std, reference_std)
    np.fill_diagonal(d, np.inf)
    _, kth = k_nearest(d, k=k)
    kth_distance = kth[:, -1]
    return float(np.percentile(kth_distance, percentile))


def density_gate_pass(kth_distance: np.ndarray, threshold: float) -> np.ndarray:
    return kth_distance <= threshold


def classify_row_evidence(density_pass: np.ndarray, directionally_supported: np.ndarray) -> list[str]:
    """Section 7 row-level evidence classes (SUPPORTED/DESCRIPTIVE_ONLY/NO_ANALOG_SUPPORT only;
    NO_HISTORICAL_TRUST_SCORE is assigned upstream by callers for out-of-scope rows, not here).
    """
    out = []
    for dp, ds in zip(density_pass, directionally_supported):
        if not dp:
            out.append(EVIDENCE_NO_ANALOG_SUPPORT)
        elif ds:
            out.append(EVIDENCE_SUPPORTED)
        else:
            out.append(EVIDENCE_DESCRIPTIVE_ONLY)
    return out


def analog_direction_consistency(
    reference_direction: np.ndarray, neighbor_indices: np.ndarray
) -> np.ndarray:
    """For each evaluation row, the majority realized direction among its k analogs.

    `reference_direction` is a 1/0 (or True/False) array over the reference pool
    (e.g. "model beat the market on this side"). Generic: operates identically
    whether fed synthetic labels or real ones; it does not itself open any file.
    """
    ref = np.asarray(reference_direction, dtype=float)
    neighbor_rates = ref[neighbor_indices].mean(axis=1)
    return neighbor_rates >= 0.5


def score_architecture_gate(
    reference_bucket_supported: np.ndarray,
    reference_realized_positive: np.ndarray,
    evaluation_bucket_supported: np.ndarray,
    evaluation_realized_roi: np.ndarray,
    evaluation_baseline_roi: float,
    min_n: int = MIN_SUPPORTED_N_2025,
) -> dict:
    """Section 8: architecture-level hard pass/fail gate.

    All four arrays/values are already-realized, season-scoped inputs supplied by the
    caller (2024 = reference/in-sample, evaluation = 2025 blind). This function performs
    no data loading and applies no correction beyond the single preregistered rule frozen
    in the plan (one k, one metric, one threshold, one evidence rule -> no multiple-
    comparisons correction needed).
    """
    eval_supported_roi = evaluation_realized_roi[evaluation_bucket_supported]
    n_supported = int(evaluation_bucket_supported.sum())
    roi_2025_supported = float(np.mean(eval_supported_roi)) if n_supported else float("nan")
    ref_supported_positive_rate = (
        float(np.mean(reference_realized_positive[reference_bucket_supported]))
        if reference_bucket_supported.sum()
        else float("nan")
    )

    gate_n = n_supported >= min_n
    gate_positive = n_supported > 0 and roi_2025_supported > 0
    gate_beats_baseline = n_supported > 0 and roi_2025_supported > evaluation_baseline_roi
    gate_2024_consistent = reference_bucket_supported.sum() > 0 and ref_supported_positive_rate > 0.5

    passed = bool(gate_n and gate_positive and gate_beats_baseline and gate_2024_consistent)
    return {
        "disposition": DISPOSITION_SUPPORTED if passed else DISPOSITION_NOT_ACTIONABLE,
        "n_2025_supported": n_supported,
        "roi_2025_supported": roi_2025_supported,
        "evaluation_baseline_roi": float(evaluation_baseline_roi),
        "ref_2024_supported_positive_rate": ref_supported_positive_rate,
        "gate_n_ge_min": gate_n,
        "gate_roi_2025_positive": gate_positive,
        "gate_roi_2025_beats_baseline": gate_beats_baseline,
        "gate_2024_directionally_consistent": gate_2024_consistent,
        "rescue_authorized": False,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vegas-csv", type=Path, default=None, help="CURRENT_PRODUCTION_ORDER QB rows CSV")
    p.add_argument("--features-csv", type=Path, default=None, help="m89_2024_2025_synthesis_trace.csv")
    p.add_argument("--out-dir", type=Path, default=Path("data/research/qb_conditional_analog_v1"))
    args, _ = p.parse_known_args()
    return args


def main() -> int:
    a = parse_args()
    if a.vegas_csv is None or a.features_csv is None:
        print(json.dumps({
            "disposition": "NOT_RUN_NO_INPUT_FILES",
            "note": (
                "Real CURRENT_PRODUCTION_ORDER / m89_2024_2025_synthesis_trace artifacts are "
                "ephemeral CI outputs not present in this environment. Pass --vegas-csv and "
                "--features-csv explicitly once those artifacts are downloaded or regenerated. "
                "See docs/research/QB_CONDITIONAL_ANALOG_V1_PLAN.md Section 1."
            ),
        }, indent=2))
        return 0
    if not a.vegas_csv.exists() or not a.features_csv.exists():
        raise RuntimeError("qb conditional analog v1: input CSV path does not exist")
    vegas_df_raw = pd.read_csv(a.vegas_csv, low_memory=False)
    features_df = pd.read_csv(a.features_csv, low_memory=False)
    # Outcome-boundary hardening: strip realized-outcome columns before anything
    # downstream (join, missingness exclusion, and — when wired in a future
    # orchestration step — scaler/distance/density) can see them. `outcomes_df` is
    # computed but deliberately never read past this point in this CLI path; it
    # exists only so a future confirmation step can call
    # attach_outcomes_at_confirmation_boundary() exactly once, at the Section 8
    # gate, never earlier.
    population_df, outcomes_df = split_population_and_outcomes(vegas_df_raw)
    merged = join_vegas_and_features(population_df, features_df)
    clean, dropped = exclude_missing(merged)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "disposition": "JOINED_AND_CLEANED_ONLY",
        "rows_joined": int(len(merged)),
        "rows_excluded_missingness": dropped,
        "rows_clean": int(len(clean)),
        "outcome_columns_held_back": sorted(set(outcomes_df.columns) - set(JOIN_KEYS)),
        "note": (
            "This CLI path performs Sections 1 and 3 only (join + missingness exclusion), "
            "and it does so on an outcome-stripped population frame -- merged/clean above "
            "structurally cannot contain any OUTCOME_COLUMNS value, only the identity/context "
            "and pregame feature columns. Sections 4-8 (scaler fit, distances, neighbor rule, "
            "density gate, evidence classes, and the 2025 architecture gate) are implemented "
            "as library functions in this module and proven on synthetic fixtures in "
            "tests/test_qb_conditional_analog_v1.py, but are deliberately not wired to real "
            "outcome data by this CLI. The held-back outcome columns are reported by name "
            "only (never their values) so this summary stays auditable without opening any "
            "real 2025 result."
        ),
    }
    (a.out_dir / "qb_conditional_analog_v1_join_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
