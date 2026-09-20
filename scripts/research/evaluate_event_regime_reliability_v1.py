#!/usr/bin/env python3
"""Evaluate frozen Event/Regime Reliability Experiment V1.

Plan: docs/research/EVENT_REGIME_RELIABILITY_EXPERIMENT_V1.md

This evaluator does not alter the baseline mean prediction. It asks whether
strict-prior regime-change events identify player-games where the canonical
pregame opportunity baseline is less reliable. No sportsbook fields are read.

Important sequencing rule: 2025 is inspected only for families that pass the
frozen 2024 primary gate.
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

# Frozen candidate families from the outcome-free event qualification result.
# Continuous returning-overlap descriptors are intentionally excluded because
# V1 did not freeze a threshold that would define an event cohort.
FAMILIES = {
    "RB_JOINT_TRANSITION": {
        "position": "RB",
        "domain": "rush",
        "event": "joint_player_room_transition_flag",
        "tier": "PRIMARY",
    },
    "WR_JOINT_TRANSITION": {
        "position": "WR",
        "domain": "tgt",
        "event": "joint_player_room_transition_flag",
        "tier": "PRIMARY",
    },
    "RB_TARGET_ROOM_CHURN": {
        "position": "RB",
        "domain": "tgt",
        "event": "target_room_churn_flag",
        "tier": "SECONDARY",
    },
    "TE_TARGET_ROOM_CHURN": {
        "position": "TE",
        "domain": "tgt",
        "event": "target_room_churn_flag",
        "tier": "SECONDARY",
    },
    "WR_TARGET_ROOM_CHURN": {
        "position": "WR",
        "domain": "tgt",
        "event": "target_room_churn_flag",
        "tier": "SECONDARY",
    },
    "QB_RUSH_ROOM_CHURN": {
        "position": "QB",
        "domain": "rush",
        "event": "rush_room_churn_flag",
        "tier": "SECONDARY",
    },
    "RB_RUSH_ROOM_CHURN": {
        "position": "RB",
        "domain": "rush",
        "event": "rush_room_churn_flag",
        "tier": "SECONDARY",
    },
    "WR_RUSH_ROOM_CHURN": {
        "position": "WR",
        "domain": "rush",
        "event": "rush_room_churn_flag",
        "tier": "SECONDARY",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    if b == 0:
        return np.inf if a > 0 else 1.0
    return float(a / b)


def _fit_beta(train: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    t = train[[target, *features]].dropna()
    x = np.column_stack([np.ones(len(t)), t[features].to_numpy(float)])
    y = t[target].to_numpy(float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def _predict(df: pd.DataFrame, features: list[str], beta: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(df)), df[features].to_numpy(float)])
    return x @ beta


def _cohort_metrics(residual: np.ndarray, catastrophic_threshold: float) -> dict[str, float]:
    residual = np.asarray(residual, float)
    abs_resid = np.abs(residual)
    return {
        "rows": int(len(residual)),
        "mae": float(abs_resid.mean()) if len(residual) else np.nan,
        "rmse": float(np.sqrt(np.mean(residual * residual))) if len(residual) else np.nan,
        "bias": float(residual.mean()) if len(residual) else np.nan,
        "median_ae": float(np.median(abs_resid)) if len(residual) else np.nan,
        "p75_ae": float(np.quantile(abs_resid, 0.75)) if len(residual) else np.nan,
        "p90_ae": float(np.quantile(abs_resid, 0.90)) if len(residual) else np.nan,
        "p95_ae": float(np.quantile(abs_resid, 0.95)) if len(residual) else np.nan,
        "residual_sd": float(np.std(residual, ddof=0)) if len(residual) else np.nan,
        "catastrophic_miss_rate": (
            float(np.mean(abs_resid >= catastrophic_threshold)) if len(residual) else np.nan
        ),
    }


def _ratios(event: dict[str, float], non_event: dict[str, float]) -> dict[str, float]:
    names = ["mae", "rmse", "p90_ae", "p95_ae", "residual_sd", "catastrophic_miss_rate"]
    return {f"{n}_ratio": _safe_ratio(event[n], non_event[n]) for n in names}


def _primary_gate(event: dict[str, float], non_event: dict[str, float]) -> dict[str, object]:
    r = _ratios(event, non_event)
    gates = {
        "event_rows_gate": event["rows"] >= 50,
        "mae_gate": r["mae_ratio"] >= 1.05,
        "rmse_gate": r["rmse_ratio"] >= 1.05,
        "p90_gate": r["p90_ae_ratio"] >= 1.05,
        "catastrophic_gate": r["catastrophic_miss_rate_ratio"] >= 1.20,
    }
    return {**r, **gates, "season_gate_pass": bool(all(gates.values()))}


def _replication_gate(event: dict[str, float], non_event: dict[str, float]) -> dict[str, object]:
    r = _ratios(event, non_event)
    three_pct = sum(
        r[name] >= 1.03
        for name in ["mae_ratio", "rmse_ratio", "p90_ae_ratio", "p95_ae_ratio", "residual_sd_ratio"]
    )
    gates = {
        "event_rows_gate": event["rows"] >= 50,
        "mae_gate": r["mae_ratio"] > 1.0,
        "rmse_gate": r["rmse_ratio"] > 1.0,
        "catastrophic_gate": r["catastrophic_miss_rate_ratio"] > 1.0,
        "two_of_five_three_pct_gate": three_pct >= 2,
    }
    return {
        **r,
        **gates,
        "three_pct_metric_count": int(three_pct),
        "season_gate_pass": bool(all(gates.values())),
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


def evaluate(history: pd.DataFrame, detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = history.copy()
    h.columns = [str(c).strip().lower() for c in h.columns]
    d = detail.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]

    event_cols = sorted({spec["event"] for spec in FAMILIES.values()})
    required_detail = {*KEY, "known_context_flag", *event_cols}
    missing = required_detail - set(d.columns)
    if missing:
        raise RuntimeError(f"missing detail columns: {sorted(missing)}")
    if d.duplicated(KEY).any():
        raise RuntimeError("duplicate event-detail player-game keys")

    state = build_production_opportunity_state(h)
    base_keep = KEY + ["position", "rushes", "team_rushes", "targets", "team_targets"]
    detail_keep = KEY + ["known_context_flag", *event_cols]
    x = (
        h[base_keep]
        .merge(d[detail_keep], on=KEY, how="inner", validate="one_to_one")
        .merge(state, on=KEY, how="left", validate="one_to_one")
    )

    metric_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []

    for family, spec in FAMILIES.items():
        num, den, features = _domain_spec(spec["domain"])
        z = x[
            x["position"].astype(str).str.upper().eq(spec["position"])
            & x["known_context_flag"].astype(bool)
        ].copy()
        denom = pd.to_numeric(z[den], errors="coerce")
        numer = pd.to_numeric(z[num], errors="coerce")
        z["outcome_share"] = np.where(denom > 0, numer / denom, np.nan)
        z[spec["event"]] = z[spec["event"]].astype(bool)
        z = z.dropna(subset=["outcome_share", *features])

        train = z[z["season"].between(2019, 2023)].copy()
        if train.empty:
            raise RuntimeError(f"{family}: empty training cohort")
        beta = _fit_beta(train, features, "outcome_share")
        train_resid = _predict(train, features, beta) - train["outcome_share"].to_numpy(float)
        catastrophic_threshold = float(np.quantile(np.abs(train_resid), 0.90))

        # Primary holdout is always inspected first.
        primary = z[z["season"].eq(2024)].copy()
        primary_resid = _predict(primary, features, beta) - primary["outcome_share"].to_numpy(float)
        primary = primary.assign(_residual=primary_resid)

        primary_metrics: dict[str, dict[str, float]] = {}
        for cohort, mask in [
            ("EVENT", primary[spec["event"]]),
            ("NON_EVENT", ~primary[spec["event"]]),
        ]:
            m = _cohort_metrics(primary.loc[mask, "_residual"].to_numpy(float), catastrophic_threshold)
            primary_metrics[cohort] = m
            metric_rows.append(
                {
                    "family": family,
                    "family_tier": spec["tier"],
                    "event_name": spec["event"],
                    "position": spec["position"],
                    "domain": spec["domain"],
                    "evaluation_season": 2024,
                    "cohort": cohort,
                    "train_seasons": "2019-2023",
                    "train_rows": int(len(train)),
                    "catastrophic_threshold": catastrophic_threshold,
                    **m,
                }
            )

        pg = _primary_gate(primary_metrics["EVENT"], primary_metrics["NON_EVENT"])
        gate_rows.append(
            {
                "family": family,
                "family_tier": spec["tier"],
                "event_name": spec["event"],
                "position": spec["position"],
                "domain": spec["domain"],
                "evaluation_season": 2024,
                "gate_type": "PRIMARY",
                **pg,
                "event_abs_bias": abs(primary_metrics["EVENT"]["bias"]),
                "non_event_abs_bias": abs(primary_metrics["NON_EVENT"]["bias"]),
                "replication_inspected": bool(pg["season_gate_pass"]),
            }
        )

        # Preserve the untouched replication season for primary failures.
        if not pg["season_gate_pass"]:
            gate_rows[-1]["family_disposition"] = "RELIABILITY_SIGNAL_PRIMARY_FAIL_CLOSED_V1"
            continue

        replication = z[z["season"].eq(2025)].copy()
        rep_resid = _predict(replication, features, beta) - replication["outcome_share"].to_numpy(float)
        replication = replication.assign(_residual=rep_resid)

        rep_metrics: dict[str, dict[str, float]] = {}
        for cohort, mask in [
            ("EVENT", replication[spec["event"]]),
            ("NON_EVENT", ~replication[spec["event"]]),
        ]:
            m = _cohort_metrics(replication.loc[mask, "_residual"].to_numpy(float), catastrophic_threshold)
            rep_metrics[cohort] = m
            metric_rows.append(
                {
                    "family": family,
                    "family_tier": spec["tier"],
                    "event_name": spec["event"],
                    "position": spec["position"],
                    "domain": spec["domain"],
                    "evaluation_season": 2025,
                    "cohort": cohort,
                    "train_seasons": "2019-2023",
                    "train_rows": int(len(train)),
                    "catastrophic_threshold": catastrophic_threshold,
                    **m,
                }
            )

        rg = _replication_gate(rep_metrics["EVENT"], rep_metrics["NON_EVENT"])
        disposition = (
            "RELIABILITY_SIGNAL_REPLICATED"
            if rg["season_gate_pass"]
            else "RELIABILITY_SIGNAL_PRIMARY_PASS_REPLICATION_FAILED"
        )
        gate_rows[-1]["family_disposition"] = disposition
        gate_rows.append(
            {
                "family": family,
                "family_tier": spec["tier"],
                "event_name": spec["event"],
                "position": spec["position"],
                "domain": spec["domain"],
                "evaluation_season": 2025,
                "gate_type": "REPLICATION",
                **rg,
                "event_abs_bias": abs(rep_metrics["EVENT"]["bias"]),
                "non_event_abs_bias": abs(rep_metrics["NON_EVENT"]["bias"]),
                "replication_inspected": True,
                "family_disposition": disposition,
            }
        )

    return pd.DataFrame(metric_rows), pd.DataFrame(gate_rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--detail", type=Path, required=True)
    p.add_argument("--metrics-out", type=Path, required=True)
    p.add_argument("--gates-out", type=Path, required=True)
    p.add_argument("--manifest-out", type=Path, required=True)
    a = p.parse_args()

    metrics, gates = evaluate(pd.read_csv(a.history), pd.read_csv(a.detail))
    a.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(a.metrics_out, index=False)
    gates.to_csv(a.gates_out, index=False)

    dispositions = gates.groupby("family", sort=True)["family_disposition"].first().to_dict()
    manifest = {
        "experiment": "EVENT_REGIME_RELIABILITY_EXPERIMENT_V1",
        "history_sha256": sha256(a.history),
        "detail_sha256": sha256(a.detail),
        "sportsbook_read": False,
        "replication_policy": "2025_INSPECTED_ONLY_AFTER_2024_PRIMARY_PASS",
        "dispositions": dispositions,
    }
    a.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")

    print(gates.to_string(index=False))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
