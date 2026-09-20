#!/usr/bin/env python3
"""Evaluate frozen Historical Analog State Experiment V1.

Plan: docs/research/HISTORICAL_ANALOG_STATE_EXPERIMENT_V1.md

The qualified analog geometry is read-only here. This evaluator constructs the
preregistered analog_novelty_risk from outcome-free analog descriptors, then
tests whether high novelty identifies worse canonical PlayerForm-style
opportunity reliability. 2025 is inspected only after a complete 2024 pass.
No sportsbook fields are read and no production mean is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.build_role_room_redundancy_audit import (
    build_production_opportunity_state,
)

KEY = ["season", "week", "team", "player_identity_key"]

# QB rush opportunity is included because the repository's already-frozen
# Event/Regime Reliability V1 evaluator exposes this exact canonical
# opportunity forecast/outcome pair. No new QB model is introduced.
FAMILIES = {
    "RB_RUSH_OPPORTUNITY": {"position": "RB", "domain": "rush"},
    "RB_TARGET_OPPORTUNITY": {"position": "RB", "domain": "tgt"},
    "WR_TARGET_OPPORTUNITY": {"position": "WR", "domain": "tgt"},
    "TE_TARGET_OPPORTUNITY": {"position": "TE", "domain": "tgt"},
    "QB_RUSH_OPPORTUNITY": {"position": "QB", "domain": "rush"},
}

NOVELTY_THRESHOLD = 0.80
PRIMARY_HIGH_MIN = 100
PRIMARY_COMPARISON_MIN = 250
REPLICATION_HIGH_MIN = 75
REPLICATION_COMPARISON_MIN = 200
DEGRADATION_RATIO = 1.03

FORBIDDEN_STATE_TOKENS = (
    "outcome",
    "actual",
    "residual",
    "sportsbook",
    "odds",
    "bet_result",
    "final_score",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    if b == 0:
        return np.inf if a > 0 else 1.0
    return float(a / b)


def _empirical_strict_prior_percentile(prior: list[float], value: float) -> float:
    """Frozen deterministic percentile: fraction of strict-prior values <= target."""
    if not prior or not np.isfinite(value):
        return np.nan
    arr = np.asarray(prior, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return np.nan
    return float(np.mean(arr <= float(value)))


def _cohort_from_risk(risk: float) -> str:
    if not np.isfinite(risk):
        return "UNSCORED"
    return "HIGH_NOVELTY" if float(risk) >= NOVELTY_THRESHOLD else "COMPARISON"


def build_novelty_scores(states: pd.DataFrame) -> pd.DataFrame:
    """Construct the frozen outcome-free novelty composite.

    Chronology intentionally inherits the already-qualified analog V1 geometry:
    rows in the same season/week block are never prior to one another.
    """
    s = _normalize(states)
    required = {
        *KEY,
        "position",
        "analog_state",
        "mean_k_distance",
        "effective_analog_count",
    }
    missing = required - set(s.columns)
    if missing:
        raise RuntimeError(f"analog state missing columns: {sorted(missing)}")
    if s.duplicated(KEY).any():
        raise RuntimeError("duplicate analog-state canonical keys")

    forbidden = sorted(
        c for c in s.columns if any(token in c for token in FORBIDDEN_STATE_TOKENS)
    )
    if forbidden:
        raise RuntimeError(f"target/outcome leakage columns present in analog state: {forbidden}")

    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s["week"] = pd.to_numeric(s["week"], errors="coerce")
    s["position"] = s["position"].astype(str).str.upper()
    s["mean_k_distance"] = pd.to_numeric(s["mean_k_distance"], errors="coerce")
    # The qualified materializer names effective analog support
    # effective_analog_count. Alias it without changing the frozen geometry.
    s["effective_analog_support"] = pd.to_numeric(
        s["effective_analog_count"], errors="coerce"
    )
    s = s.sort_values(
        ["position", "season", "week", "team", "player_identity_key"]
    ).reset_index(drop=True)

    records: list[dict[str, object]] = []
    for position, pos in s.groupby("position", sort=True):
        prior_distance: list[float] = []
        prior_support: list[float] = []
        for (_, _), block in pos.groupby(["season", "week"], sort=False):
            # Score the entire week from only earlier season/week blocks.
            for _, row in block.iterrows():
                rec = {c: row[c] for c in KEY}
                rec.update(
                    {
                        "position": position,
                        "analog_state": row["analog_state"],
                        "mean_k_distance": row["mean_k_distance"],
                        "effective_analog_support": row["effective_analog_support"],
                        "percentile_reference_rows": int(len(prior_distance)),
                    }
                )
                valid = str(row["analog_state"]) == "VALID_ANALOG"
                components_ok = np.isfinite(row["mean_k_distance"]) and np.isfinite(
                    row["effective_analog_support"]
                )
                if not valid:
                    rec.update(
                        {
                            "mean_k_distance_pct": np.nan,
                            "effective_analog_support_pct": np.nan,
                            "analog_novelty_risk": np.nan,
                            "analog_scoring_state": "NO_ANALOG_SUPPORT",
                            "novelty_cohort": "UNSCORED",
                        }
                    )
                elif not components_ok:
                    rec.update(
                        {
                            "mean_k_distance_pct": np.nan,
                            "effective_analog_support_pct": np.nan,
                            "analog_novelty_risk": np.nan,
                            "analog_scoring_state": "UNKNOWN_ANALOG_COMPONENT",
                            "novelty_cohort": "UNSCORED",
                        }
                    )
                elif not prior_distance:
                    rec.update(
                        {
                            "mean_k_distance_pct": np.nan,
                            "effective_analog_support_pct": np.nan,
                            "analog_novelty_risk": np.nan,
                            "analog_scoring_state": "NO_PERCENTILE_SUPPORT",
                            "novelty_cohort": "UNSCORED",
                        }
                    )
                else:
                    distance_pct = _empirical_strict_prior_percentile(
                        prior_distance, float(row["mean_k_distance"])
                    )
                    support_pct = _empirical_strict_prior_percentile(
                        prior_support, float(row["effective_analog_support"])
                    )
                    risk = 0.5 * distance_pct + 0.5 * (1.0 - support_pct)
                    rec.update(
                        {
                            "mean_k_distance_pct": distance_pct,
                            "effective_analog_support_pct": support_pct,
                            "analog_novelty_risk": float(risk),
                            "analog_scoring_state": "SCORED",
                            "novelty_cohort": _cohort_from_risk(float(risk)),
                        }
                    )
                records.append(rec)

            # Only after the block is scored may its values enter future percentiles.
            eligible = block[
                block["analog_state"].astype(str).eq("VALID_ANALOG")
                & block["mean_k_distance"].notna()
                & block["effective_analog_support"].notna()
            ]
            prior_distance.extend(eligible["mean_k_distance"].astype(float).tolist())
            prior_support.extend(
                eligible["effective_analog_support"].astype(float).tolist()
            )

    return pd.DataFrame(records)


def _parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
        .fillna(False)
        .astype(bool)
    )


def audit_integrity(
    history: pd.DataFrame, states: pd.DataFrame, neighbors: pd.DataFrame
) -> dict[str, object]:
    h, s, n = _normalize(history), _normalize(states), _normalize(neighbors)
    for label, frame, cols in [
        ("history", h, [*KEY, "position"]),
        ("states", s, [*KEY, "position", "analog_state", "mean_k_distance", "effective_analog_count"]),
    ]:
        missing = set(cols) - set(frame.columns)
        if missing:
            raise RuntimeError(f"{label} missing integrity columns: {sorted(missing)}")

    neighbor_required = {
        "target_season",
        "target_week",
        "target_team",
        "target_player_identity_key",
        "analog_season",
        "analog_week",
        "analog_team",
        "analog_player_identity_key",
        "neighbor_rank",
        "strict_prior",
    }
    missing_neighbor = neighbor_required - set(n.columns)
    if missing_neighbor:
        raise RuntimeError(f"neighbors missing integrity columns: {sorted(missing_neighbor)}")

    history_duplicates = int(h.duplicated(KEY).sum())
    state_duplicates = int(s.duplicated(KEY).sum())
    identity_missing = int(
        s["player_identity_key"].isna().sum()
        + s["player_identity_key"].astype(str).str.strip().eq("").sum()
    )
    stable_identity_coverage = float(
        1.0 - identity_missing / len(s)
    ) if len(s) else 0.0

    ns = pd.to_numeric(n["target_season"], errors="coerce")
    nw = pd.to_numeric(n["target_week"], errors="coerce")
    aas = pd.to_numeric(n["analog_season"], errors="coerce")
    aaw = pd.to_numeric(n["analog_week"], errors="coerce")
    recomputed_strict = (aas < ns) | ((aas == ns) & (aaw < nw))
    chronology_violations = int((~recomputed_strict.fillna(False)).sum())
    declared = _parse_bool(n["strict_prior"])
    declared_strict_mismatches = int((declared != recomputed_strict.fillna(False)).sum())
    analog_identity_missing = int(
        n["analog_player_identity_key"].isna().sum()
        + n["analog_player_identity_key"].astype(str).str.strip().eq("").sum()
    )
    neighbor_rank_duplicates = int(
        n.duplicated(
            [
                "target_season",
                "target_week",
                "target_team",
                "target_player_identity_key",
                "neighbor_rank",
            ]
        ).sum()
    )

    state_keys = s[KEY + ["position"]].copy()
    hist_keys = h[KEY + ["position"]].copy()
    joined = state_keys.merge(
        hist_keys,
        on=KEY,
        how="left",
        validate="one_to_one",
        suffixes=("_state", "_history"),
        indicator=True,
    )
    state_history_unmatched = int((joined["_merge"] != "both").sum())
    position_mismatches = int(
        (
            joined["position_state"].astype(str).str.upper()
            != joined["position_history"].astype(str).str.upper()
        ).fillna(True).sum()
    )

    forbidden_state_columns = sorted(
        c for c in s.columns if any(token in c for token in FORBIDDEN_STATE_TOKENS)
    )
    leakage_violations = int(len(forbidden_state_columns))

    integrity_gate = bool(
        history_duplicates == 0
        and state_duplicates == 0
        and identity_missing == 0
        and analog_identity_missing == 0
        and stable_identity_coverage >= 0.99
        and chronology_violations == 0
        and declared_strict_mismatches == 0
        and neighbor_rank_duplicates == 0
        and state_history_unmatched == 0
        and position_mismatches == 0
        and leakage_violations == 0
    )
    return {
        "history_duplicate_keys": history_duplicates,
        "state_duplicate_keys": state_duplicates,
        "stable_identity_coverage": stable_identity_coverage,
        "identity_missing_rows": identity_missing,
        "analog_identity_missing_rows": analog_identity_missing,
        "chronology_violations": chronology_violations,
        "declared_strict_prior_mismatches": declared_strict_mismatches,
        "neighbor_rank_duplicates": neighbor_rank_duplicates,
        "state_history_unmatched_rows": state_history_unmatched,
        "position_mismatches": position_mismatches,
        "forbidden_state_columns": forbidden_state_columns,
        "leakage_violations": leakage_violations,
        "integrity_gate": integrity_gate,
    }


def _domain_spec(domain: str) -> tuple[str, str, list[str]]:
    if domain == "rush":
        target_num, target_den = "rushes", "team_rushes"
    elif domain == "tgt":
        target_num, target_den = "targets", "team_targets"
    else:
        raise ValueError(domain)
    features = [
        f"prod_{domain}_prior_share",
        f"prod_{domain}_prior_games",
        f"prod_{domain}_current_share",
        f"prod_{domain}_current_games",
        f"prod_{domain}_playerform_blend",
    ]
    return target_num, target_den, features


def _fit_beta(train: pd.DataFrame, features: list[str]) -> np.ndarray:
    t = train[["outcome_share", *features]].dropna()
    if t.empty:
        raise RuntimeError("empty 2019-2023 baseline training cohort")
    x = np.column_stack([np.ones(len(t)), t[features].to_numpy(float)])
    y = t["outcome_share"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def _predict(df: pd.DataFrame, features: list[str], beta: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(df)), df[features].to_numpy(float)])
    return x @ beta


def _cohort_metrics(residual: np.ndarray, catastrophic_threshold: float) -> dict[str, float]:
    residual = np.asarray(residual, dtype=float)
    abs_resid = np.abs(residual)
    return {
        "rows": int(len(residual)),
        "mae": float(abs_resid.mean()) if len(residual) else np.nan,
        "rmse": float(np.sqrt(np.mean(residual * residual))) if len(residual) else np.nan,
        "bias": float(residual.mean()) if len(residual) else np.nan,
        "p90_ae": float(np.quantile(abs_resid, 0.90)) if len(residual) else np.nan,
        "catastrophic_miss_rate": (
            float(np.mean(abs_resid >= catastrophic_threshold)) if len(residual) else np.nan
        ),
    }


def _ratios(high: dict[str, float], comparison: dict[str, float]) -> dict[str, float]:
    names = ["mae", "rmse", "p90_ae", "catastrophic_miss_rate"]
    return {f"{name}_ratio": _safe_ratio(high[name], comparison[name]) for name in names}


def _primary_gate(
    high: dict[str, float], comparison: dict[str, float], integrity_gate: bool = True
) -> dict[str, object]:
    ratios = _ratios(high, comparison)
    tail_count = sum(
        bool(ratios[name] >= DEGRADATION_RATIO)
        for name in ["rmse_ratio", "p90_ae_ratio", "catastrophic_miss_rate_ratio"]
        if np.isfinite(ratios[name]) or np.isinf(ratios[name])
    )
    gates = {
        "high_novelty_rows_gate": high["rows"] >= PRIMARY_HIGH_MIN,
        "comparison_rows_gate": comparison["rows"] >= PRIMARY_COMPARISON_MIN,
        "mae_gate": ratios["mae_ratio"] >= DEGRADATION_RATIO,
        "two_of_three_tail_gate": tail_count >= 2,
        "integrity_gate": bool(integrity_gate),
    }
    return {
        **ratios,
        "tail_three_pct_metric_count": int(tail_count),
        **gates,
        "season_gate_pass": bool(all(gates.values())),
    }


def _replication_gate(
    high: dict[str, float],
    comparison: dict[str, float],
    primary_mae_delta: float,
    integrity_gate: bool = True,
) -> dict[str, object]:
    ratios = _ratios(high, comparison)
    tail_count = sum(
        bool(ratios[name] >= DEGRADATION_RATIO)
        for name in ["rmse_ratio", "p90_ae_ratio", "catastrophic_miss_rate_ratio"]
        if np.isfinite(ratios[name]) or np.isinf(ratios[name])
    )
    replication_mae_delta = float(high["mae"] - comparison["mae"])
    same_direction = bool(
        np.isfinite(primary_mae_delta)
        and np.isfinite(replication_mae_delta)
        and primary_mae_delta != 0
        and replication_mae_delta != 0
        and np.sign(primary_mae_delta) == np.sign(replication_mae_delta)
    )
    gates = {
        "high_novelty_rows_gate": high["rows"] >= REPLICATION_HIGH_MIN,
        "comparison_rows_gate": comparison["rows"] >= REPLICATION_COMPARISON_MIN,
        "mae_gate": ratios["mae_ratio"] >= DEGRADATION_RATIO,
        "two_of_three_tail_gate": tail_count >= 2,
        "same_mae_direction_gate": same_direction,
        "integrity_gate": bool(integrity_gate),
    }
    return {
        **ratios,
        "tail_three_pct_metric_count": int(tail_count),
        "primary_mae_delta": float(primary_mae_delta),
        "replication_mae_delta": replication_mae_delta,
        **gates,
        "season_gate_pass": bool(all(gates.values())),
    }


def _score_holdout(
    holdout: pd.DataFrame,
    features: list[str],
    beta: np.ndarray,
    catastrophic_threshold: float,
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    scored = holdout[
        holdout["analog_scoring_state"].eq("SCORED")
        & holdout["novelty_cohort"].isin(["HIGH_NOVELTY", "COMPARISON"])
    ].dropna(subset=["outcome_share", *features, "analog_novelty_risk"])
    pred = _predict(scored, features, beta) if len(scored) else np.array([], dtype=float)
    scored = scored.assign(_residual=pred - scored["outcome_share"].to_numpy(float))
    metrics: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []
    for cohort in ["HIGH_NOVELTY", "COMPARISON"]:
        residual = scored.loc[
            scored["novelty_cohort"].eq(cohort), "_residual"
        ].to_numpy(float)
        m = _cohort_metrics(residual, catastrophic_threshold)
        metrics[cohort] = m
        rows.append({"cohort": cohort, **m})
    return metrics, rows


def _evaluate_family(
    frame: pd.DataFrame,
    family: str,
    spec: dict[str, str],
    integrity_gate: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    num, den, features = _domain_spec(spec["domain"])
    z = frame[
        frame["position"].astype(str).str.upper().eq(spec["position"])
    ].copy()
    numer = pd.to_numeric(z[num], errors="coerce")
    denom = pd.to_numeric(z[den], errors="coerce")
    z["outcome_share"] = np.where(denom > 0, numer / denom, np.nan)
    for feature in features:
        z[feature] = pd.to_numeric(z[feature], errors="coerce")

    train = z[z["season"].between(2019, 2023)].dropna(
        subset=["outcome_share", *features]
    )
    beta = _fit_beta(train, features)
    train_residual = _predict(train, features, beta) - train["outcome_share"].to_numpy(float)
    catastrophic_threshold = float(np.quantile(np.abs(train_residual), 0.90))

    metric_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []

    primary = z[z["season"].eq(2024)].copy()
    primary_metrics, primary_rows = _score_holdout(
        primary, features, beta, catastrophic_threshold
    )
    for row in primary_rows:
        metric_rows.append(
            {
                "family": family,
                "position": spec["position"],
                "domain": spec["domain"],
                "evaluation_season": 2024,
                "train_seasons": "2019-2023",
                "train_rows": int(len(train)),
                "novelty_threshold": NOVELTY_THRESHOLD,
                "catastrophic_threshold": catastrophic_threshold,
                **row,
            }
        )
    pg = _primary_gate(
        primary_metrics["HIGH_NOVELTY"],
        primary_metrics["COMPARISON"],
        integrity_gate,
    )
    primary_mae_delta = float(
        primary_metrics["HIGH_NOVELTY"]["mae"]
        - primary_metrics["COMPARISON"]["mae"]
    )
    primary_gate_row = {
        "family": family,
        "position": spec["position"],
        "domain": spec["domain"],
        "evaluation_season": 2024,
        "gate_type": "PRIMARY",
        **pg,
        "high_novelty_abs_bias": abs(primary_metrics["HIGH_NOVELTY"]["bias"]),
        "comparison_abs_bias": abs(primary_metrics["COMPARISON"]["bias"]),
        "replication_inspected": bool(pg["season_gate_pass"]),
    }

    if not pg["season_gate_pass"]:
        primary_gate_row["family_disposition"] = "FAILED_CLOSED_PRIMARY"
        gate_rows.append(primary_gate_row)
        return metric_rows, gate_rows

    # Only a complete 2024 pass exposes predictive 2025 results.
    replication = z[z["season"].eq(2025)].copy()
    rep_metrics, rep_rows = _score_holdout(
        replication, features, beta, catastrophic_threshold
    )
    for row in rep_rows:
        metric_rows.append(
            {
                "family": family,
                "position": spec["position"],
                "domain": spec["domain"],
                "evaluation_season": 2025,
                "train_seasons": "2019-2023",
                "train_rows": int(len(train)),
                "novelty_threshold": NOVELTY_THRESHOLD,
                "catastrophic_threshold": catastrophic_threshold,
                **row,
            }
        )
    rg = _replication_gate(
        rep_metrics["HIGH_NOVELTY"],
        rep_metrics["COMPARISON"],
        primary_mae_delta,
        integrity_gate,
    )
    disposition = (
        "ANALOG_RELIABILITY_SIGNAL_REPLICATED"
        if rg["season_gate_pass"]
        else "FAILED_CLOSED_REPLICATION"
    )
    primary_gate_row["family_disposition"] = disposition
    gate_rows.append(primary_gate_row)
    gate_rows.append(
        {
            "family": family,
            "position": spec["position"],
            "domain": spec["domain"],
            "evaluation_season": 2025,
            "gate_type": "REPLICATION",
            **rg,
            "high_novelty_abs_bias": abs(rep_metrics["HIGH_NOVELTY"]["bias"]),
            "comparison_abs_bias": abs(rep_metrics["COMPARISON"]["bias"]),
            "replication_inspected": True,
            "family_disposition": disposition,
        }
    )
    return metric_rows, gate_rows


def evaluate(
    history: pd.DataFrame,
    states: pd.DataFrame,
    neighbors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    h = _normalize(history)
    scores = build_novelty_scores(states)
    integrity = audit_integrity(h, states, neighbors)
    if not integrity["integrity_gate"]:
        raise RuntimeError(
            "historical analog reliability integrity gate failed before scientific evaluation: "
            + json.dumps(integrity, sort_keys=True)
        )

    required_history = {
        *KEY,
        "position",
        "rushes",
        "team_rushes",
        "targets",
        "team_targets",
    }
    missing = required_history - set(h.columns)
    if missing:
        raise RuntimeError(f"history missing evaluation columns: {sorted(missing)}")
    if h.duplicated(KEY).any():
        raise RuntimeError("history contains duplicate canonical player-game keys")

    production = build_production_opportunity_state(h)
    score_keep = KEY + [
        "position",
        "analog_state",
        "mean_k_distance",
        "effective_analog_support",
        "percentile_reference_rows",
        "mean_k_distance_pct",
        "effective_analog_support_pct",
        "analog_novelty_risk",
        "analog_scoring_state",
        "novelty_cohort",
    ]
    frame = (
        h[KEY + ["position", "rushes", "team_rushes", "targets", "team_targets"]]
        .merge(
            scores[score_keep].rename(columns={"position": "analog_position"}),
            on=KEY,
            how="inner",
            validate="one_to_one",
        )
        .merge(production, on=KEY, how="left", validate="one_to_one")
    )

    metric_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    for family, spec in FAMILIES.items():
        fm, fg = _evaluate_family(frame, family, spec, bool(integrity["integrity_gate"]))
        metric_rows.extend(fm)
        gate_rows.extend(fg)

    metrics = pd.DataFrame(metric_rows)
    gates = pd.DataFrame(gate_rows)
    return scores, metrics, gates, integrity


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--neighbors", type=Path, required=True)
    p.add_argument("--risk-out", type=Path, required=True)
    p.add_argument("--metrics-out", type=Path, required=True)
    p.add_argument("--gates-out", type=Path, required=True)
    p.add_argument("--manifest-out", type=Path, required=True)
    a = p.parse_args()

    history = pd.read_csv(a.history)
    states = pd.read_csv(a.states)
    neighbors = pd.read_csv(a.neighbors)
    scores, metrics, gates, integrity = evaluate(history, states, neighbors)

    for path, frame in [
        (a.risk_out, scores),
        (a.metrics_out, metrics),
        (a.gates_out, gates),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    dispositions = (
        gates.groupby("family", sort=True)["family_disposition"].first().to_dict()
        if len(gates)
        else {}
    )
    replication_families = sorted(
        gates.loc[gates["evaluation_season"].eq(2025), "family"].unique().tolist()
    ) if len(gates) else []
    manifest = {
        "experiment": "HISTORICAL_ANALOG_STATE_EXPERIMENT_V1",
        "frozen_disposition": "HISTORICAL_ANALOG_STATE_EXPERIMENT_V1_FROZEN_PRE_OUTCOME",
        "history_sha256": sha256(a.history),
        "states_sha256": sha256(a.states),
        "neighbors_sha256": sha256(a.neighbors),
        "sportsbook_read": False,
        "production_mean_changed": False,
        "analog_geometry_changed": False,
        "percentile_scope": "POSITION_SPECIFIC_EXPANDING_STRICT_PRIOR_SEASON_WEEK",
        "percentile_rule": "fraction_of_strict_prior_valid_values_less_than_or_equal_to_target",
        "effective_analog_support_source_column": "effective_analog_count",
        "novelty_formula": "0.5*pct(mean_k_distance)+0.5*(1-pct(effective_analog_support))",
        "novelty_threshold": NOVELTY_THRESHOLD,
        "primary_holdout": 2024,
        "replication_holdout": 2025,
        "replication_policy": "2025_PREDICTIVE_RESULTS_EXPOSED_ONLY_AFTER_COMPLETE_2024_PRIMARY_PASS",
        "replication_families": replication_families,
        "integrity": integrity,
        "no_analog_support_rows": int(
            scores["analog_scoring_state"].eq("NO_ANALOG_SUPPORT").sum()
        ),
        "no_percentile_support_rows": int(
            scores["analog_scoring_state"].eq("NO_PERCENTILE_SUPPORT").sum()
        ),
        "dispositions": dispositions,
    }
    a.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(gates.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
