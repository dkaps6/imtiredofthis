#!/usr/bin/env python3
"""Score frozen WR opportunity-bias topology deployment eligibility V1.

The named-player signed opportunity-bias signal was already qualified overall.
This script tests whether that exact signal independently retains predictive
utility in the immutable WR-R15 ANCHOR and SECONDARY topology classes.

It is blocked unless the separate conservation-bridge source-recovery audit is
READY. It changes no projection or entitlement.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

VERSION = "PRISM_WR_OPPORTUNITY_BIAS_TOPOLOGY_ELIGIBILITY_V1"
FREEZE_COMMIT = "9b90018f0906f64371daa7e350fc9375bb7996d4"

SIGNAL_RUN = 39109423266
SIGNAL_ARTIFACT = 12128983909
SIGNAL_DIGEST = "sha256:bb14d6474778d0afd53ca74bb60237b381aa4e973195bab3d09d54ed87ef2490"
TOPOLOGY_RUN = 36203881629
TOPOLOGY_ARTIFACT = 10893281689
TOPOLOGY_DIGEST = "sha256:33b12fd521196838054f4ee4240d8cea8f66c2fce5b5bfc9eb902bc67d730adb"

SOURCE_READY = "PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_SOURCE_RECOVERED_READY_ALL"

BOOT_REPS = 5000
BOOT_SEED = 20261018
MIN_ROWS = 200
MIN_PLAYERS = 40

CLASSES = {
    "ANCHOR": "FROZEN_ANCHOR_IDENTITY_COMPLEMENT_BRIDGE",
    "SECONDARY": "FROZEN_SECONDARY",
}


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty JSON: {path}")
    return json.loads(path.read_text())


def load_recovery(path: Path) -> dict:
    p = load_json(path)
    if p.get("version") != "PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_V1_SOURCE_RECOVERY_AUDIT":
        raise RuntimeError("wrong source-recovery result version")
    if p.get("disposition") != SOURCE_READY:
        raise RuntimeError(f"source recovery not READY: {p.get('disposition')}")
    if p.get("bridge_outcome_rows_scored") != 0:
        raise RuntimeError("source recovery unexpectedly scored bridge outcomes")
    if p.get("target_corrections_applied") != 0:
        raise RuntimeError("source recovery unexpectedly applied target correction")
    if p.get("production_changed") is not False:
        raise RuntimeError("source recovery unexpectedly changed production")
    return p


def load_signal(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "position", "component", "season", "week", "team", "player_clean_key",
        "source_eligible", "position_prior_mean", "hierarchical_signed_bias",
        "target_component_error", "team_week_cluster",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"signal rows missing columns: {sorted(missing)}")
    x["position"] = x["position"].astype("string").fillna("").str.strip().str.upper()
    x["component"] = x["component"].astype("string").fillna("").str.strip().str.upper()
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].astype("string").fillna("").str.strip()
    x["player_clean_key"] = x["player_clean_key"].astype("string").fillna("").str.strip()
    if x["source_eligible"].dtype != bool:
        x["source_eligible"] = (
            x["source_eligible"].astype("string").str.strip().str.lower()
            .map({"true": True, "false": False}).fillna(False).astype(bool)
        )
    for c in ("position_prior_mean", "hierarchical_signed_bias", "target_component_error"):
        x[c] = num(x[c])
    x = x.loc[
        x["position"].eq("WR")
        & x["component"].eq("OPPORTUNITY")
        & x["source_eligible"].eq(True)
        & x["season"].isin([2023, 2024])
    ].copy()
    keys = ["season", "week", "team", "player_clean_key"]
    if x.empty or x.duplicated(keys).any():
        raise RuntimeError("WR opportunity signal identity invalid")
    if not np.isfinite(x[["position_prior_mean", "hierarchical_signed_bias", "target_component_error"]].to_numpy(float)).all():
        raise RuntimeError("WR opportunity signal has nonfinite score fields")
    return x


def load_topology(path: Path, audit_path: Path) -> pd.DataFrame:
    audit = load_json(audit_path)
    if audit.get("version") != "WR_ANCHOR_ROLE_TRANSMISSION_AUDIT_V1":
        raise RuntimeError("wrong WR topology audit version")
    if int(audit.get("frozen_full_wr_rows", -1)) != 6409:
        raise RuntimeError("WR topology full-row count drift")
    if int(audit.get("rank1_anchor_identity_mismatches", -1)) != 0:
        raise RuntimeError("WR topology anchor identity mismatch")
    if float(audit.get("secondary_bridge_coverage", 0.0)) != 1.0:
        raise RuntimeError("WR topology secondary coverage drift")
    if int(audit.get("strict_prior_future_violations", -1)) != 0:
        raise RuntimeError("WR topology strict-prior violation")

    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "team", "player_clean_key",
        "authority_row_type",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"WR topology missing columns: {sorted(missing)}")
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].astype("string").fillna("").str.strip()
    x["player_clean_key"] = x["player_clean_key"].astype("string").fillna("").str.strip()
    x["authority_row_type"] = x["authority_row_type"].astype("string").fillna("").str.strip()
    x = x.loc[x["season"].isin([2023, 2024])].copy()
    keys = ["season", "week", "team", "player_clean_key"]
    if x.empty or x.duplicated(keys).any():
        raise RuntimeError("WR topology identity invalid")
    observed = set(x["authority_row_type"].unique())
    expected = set(CLASSES.values())
    if not expected.issubset(observed):
        raise RuntimeError(f"WR topology classes missing expected={expected} observed={observed}")
    return x


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    aa = num(a).to_numpy(float)
    bb = num(b).to_numpy(float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 3 or np.unique(aa[mask]).size < 2 or np.unique(bb[mask]).size < 2:
        return math.nan
    return float(spearmanr(aa[mask], bb[mask]).statistic)


def sign_accuracy(pred: pd.Series, target: pd.Series) -> float:
    p = num(pred).to_numpy(float)
    y = num(target).to_numpy(float)
    mask = np.isfinite(p) & np.isfinite(y) & (np.abs(y) > 1e-12)
    if not mask.any():
        return math.nan
    return float((np.sign(p[mask]) == np.sign(y[mask])).mean())


def cluster_bootstrap(df: pd.DataFrame, cluster_col: str, seed: int) -> dict:
    z = df[["delta_meta_ae", cluster_col]].dropna().copy()
    grouped = z.groupby(cluster_col, sort=True)["delta_meta_ae"].agg(["sum", "count"])
    if len(grouped) < 2:
        return {"clusters": int(len(grouped)), "valid_reps": 0, "ci95_low": math.nan, "ci95_high": math.nan}
    a = grouped[["sum", "count"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT_REPS, dtype=float)
    for j in range(BOOT_REPS):
        idx = rng.integers(0, len(a), size=len(a))
        total = a[idx].sum(axis=0)
        vals[j] = total[0] / total[1] if total[1] > 0 else np.nan
    vals = vals[np.isfinite(vals)]
    return {
        "clusters": int(len(a)),
        "valid_reps": int(len(vals)),
        "ci95_low": float(np.quantile(vals, .025)) if len(vals) else math.nan,
        "ci95_high": float(np.quantile(vals, .975)) if len(vals) else math.nan,
    }


def equal_player_frame(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, p in g.groupby("player_clean_key", sort=True):
        if len(p) < 5:
            continue
        rows.append({
            "player_clean_key": str(pid),
            "rows": int(len(p)),
            "mean_delta_meta_ae": float(p["delta_meta_ae"].mean()),
            "baseline_meta_ae": float(p["baseline_meta_ae"].mean()),
            "candidate_meta_ae": float(p["candidate_meta_ae"].mean()),
        })
    return pd.DataFrame(rows)


def equal_player_bootstrap(p: pd.DataFrame, seed: int) -> dict:
    if p.empty:
        return {"players": 0, "valid_reps": 0, "ci95_low": math.nan, "ci95_high": math.nan}
    vals = num(p["mean_delta_meta_ae"]).dropna().to_numpy(float)
    if len(vals) < 2:
        return {"players": int(len(vals)), "valid_reps": 0, "ci95_low": math.nan, "ci95_high": math.nan}
    rng = np.random.default_rng(seed)
    out = np.empty(BOOT_REPS, dtype=float)
    for j in range(BOOT_REPS):
        out[j] = float(rng.choice(vals, size=len(vals), replace=True).mean())
    return {
        "players": int(len(vals)),
        "valid_reps": BOOT_REPS,
        "ci95_low": float(np.quantile(out, .025)),
        "ci95_high": float(np.quantile(out, .975)),
    }


def score_slice(g: pd.DataFrame, topology_class: str, season: int, seed_offset: int) -> tuple[dict, pd.DataFrame]:
    z = g.loc[
        g["topology_class"].eq(topology_class) & g["season"].eq(season)
    ].copy()
    target = num(z["target_component_error"]).to_numpy(float)
    base = num(z["position_prior_mean"]).to_numpy(float)
    cand = num(z["hierarchical_signed_bias"]).to_numpy(float)
    z["baseline_meta_ae"] = np.abs(target - base)
    z["candidate_meta_ae"] = np.abs(target - cand)
    z["delta_meta_ae"] = z["baseline_meta_ae"] - z["candidate_meta_ae"]
    z["baseline_sqerr"] = np.square(target - base)
    z["candidate_sqerr"] = np.square(target - cand)
    z["baseline_signed_residual"] = target - base
    z["candidate_signed_residual"] = target - cand

    players = equal_player_frame(z)
    pv = num(players["mean_delta_meta_ae"]).dropna() if not players.empty else pd.Series(dtype=float)

    rec = {
        "topology_class": topology_class,
        "season": season,
        "rows": int(len(z)),
        "players": int(z["player_clean_key"].nunique()),
        "support_rows_ge200": int(len(z)) >= MIN_ROWS,
        "support_players_ge40": int(z["player_clean_key"].nunique()) >= MIN_PLAYERS,
        "mean_delta_meta_ae": float(z["delta_meta_ae"].mean()) if len(z) else math.nan,
        "baseline_meta_ae": float(z["baseline_meta_ae"].mean()) if len(z) else math.nan,
        "candidate_meta_ae": float(z["candidate_meta_ae"].mean()) if len(z) else math.nan,
        "baseline_meta_rmse": float(np.sqrt(z["baseline_sqerr"].mean())) if len(z) else math.nan,
        "candidate_meta_rmse": float(np.sqrt(z["candidate_sqerr"].mean())) if len(z) else math.nan,
        "baseline_spearman": safe_spearman(z["position_prior_mean"], z["target_component_error"]) if len(z) else math.nan,
        "candidate_spearman": safe_spearman(z["hierarchical_signed_bias"], z["target_component_error"]) if len(z) else math.nan,
        "baseline_sign_accuracy": sign_accuracy(z["position_prior_mean"], z["target_component_error"]) if len(z) else math.nan,
        "candidate_sign_accuracy": sign_accuracy(z["hierarchical_signed_bias"], z["target_component_error"]) if len(z) else math.nan,
        "baseline_mean_signed_residual": float(z["baseline_signed_residual"].mean()) if len(z) else math.nan,
        "candidate_mean_signed_residual": float(z["candidate_signed_residual"].mean()) if len(z) else math.nan,
        "evaluable_players_ge5": int(len(players)),
        "equal_player_mean_improvement": float(pv.mean()) if len(pv) else math.nan,
        "equal_player_median_improvement": float(pv.median()) if len(pv) else math.nan,
        "equal_player_improved_share": float((pv > 0).mean()) if len(pv) else math.nan,
        "equal_player_p10_improvement": float(pv.quantile(.10)) if len(pv) else math.nan,
        "equal_player_p25_improvement": float(pv.quantile(.25)) if len(pv) else math.nan,
        "equal_player_p50_improvement": float(pv.quantile(.50)) if len(pv) else math.nan,
        "equal_player_p75_improvement": float(pv.quantile(.75)) if len(pv) else math.nan,
        "equal_player_p90_improvement": float(pv.quantile(.90)) if len(pv) else math.nan,
        "worst_named_player_mean_improvement": float(pv.min()) if len(pv) else math.nan,
        "baseline_equal_player_meta_ae": float(players["baseline_meta_ae"].mean()) if len(players) else math.nan,
        "player_cluster_bootstrap": cluster_bootstrap(z, "player_clean_key", BOOT_SEED + seed_offset),
        "team_week_cluster_bootstrap": cluster_bootstrap(z, "team_week_cluster", BOOT_SEED + 10000 + seed_offset),
        "equal_player_bootstrap": equal_player_bootstrap(players, BOOT_SEED + 20000 + seed_offset),
    }
    return rec, players


def qualify(dev: dict, conf: dict) -> tuple[dict, bool]:
    tail_floor = (
        -0.25 * float(conf["baseline_equal_player_meta_ae"])
        if np.isfinite(conf["baseline_equal_player_meta_ae"]) else math.nan
    )
    gates = {
        "development_support_rows": bool(dev["support_rows_ge200"]),
        "development_support_players": bool(dev["support_players_ge40"]),
        "development_mean_improves": bool(dev["mean_delta_meta_ae"] > 0),
        "development_rmse_improves": bool(dev["candidate_meta_rmse"] < dev["baseline_meta_rmse"]),
        "development_spearman_improves": bool(dev["candidate_spearman"] > dev["baseline_spearman"]),
        "confirmation_support_rows": bool(conf["support_rows_ge200"]),
        "confirmation_support_players": bool(conf["support_players_ge40"]),
        "confirmation_mean_improves": bool(conf["mean_delta_meta_ae"] > 0),
        "confirmation_player_cluster_ci_low_gt_zero": bool(conf["player_cluster_bootstrap"].get("ci95_low", math.nan) > 0),
        "confirmation_team_week_cluster_ci_low_gt_zero": bool(conf["team_week_cluster_bootstrap"].get("ci95_low", math.nan) > 0),
        "confirmation_equal_player_mean_improves": bool(conf["equal_player_mean_improvement"] > 0),
        "confirmation_equal_player_ci_low_gt_zero": bool(conf["equal_player_bootstrap"].get("ci95_low", math.nan) > 0),
        "confirmation_median_player_nonworse": bool(conf["equal_player_median_improvement"] >= 0),
        "confirmation_player_improved_share_majority": bool(conf["equal_player_improved_share"] > .50),
        "confirmation_rmse_improves": bool(conf["candidate_meta_rmse"] < conf["baseline_meta_rmse"]),
        "confirmation_spearman_improves": bool(conf["candidate_spearman"] > conf["baseline_spearman"]),
        "confirmation_tail_harm_guard": bool(
            np.isfinite(tail_floor) and conf["equal_player_p10_improvement"] >= tail_floor
        ),
        "tail_floor": tail_floor,
    }
    ok = all(v for k, v in gates.items() if k != "tail_floor")
    return gates, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-rows", required=True, type=Path)
    ap.add_argument("--topology-detail", required=True, type=Path)
    ap.add_argument("--topology-audit", required=True, type=Path)
    ap.add_argument("--source-recovery-result", required=True, type=Path)
    ap.add_argument("--source-recovery-run-id", required=True, type=int)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        recovery = load_recovery(args.source_recovery_result)
        signal = load_signal(args.signal_rows)
        topology = load_topology(args.topology_detail, args.topology_audit)

        keys = ["season", "week", "team", "player_clean_key"]
        joined = signal.merge(
            topology[keys + ["authority_row_type"]],
            on=keys,
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if not joined["_merge"].eq("both").all():
            missing = joined.loc[~joined["_merge"].eq("both"), keys].head(20)
            raise RuntimeError(f"WR signal/topology join is not 100%: {missing.to_dict('records')}")
        reverse = {v: k for k, v in CLASSES.items()}
        joined["topology_class"] = joined["authority_row_type"].map(reverse)
        if joined["topology_class"].isna().any():
            bad = joined.loc[joined["topology_class"].isna(), "authority_row_type"].value_counts().to_dict()
            raise RuntimeError(f"unexpected WR topology class in signal rows: {bad}")

        metrics = {}
        gates = {}
        dispositions = {}
        player_parts = []

        for i, topology_class in enumerate(("ANCHOR", "SECONDARY")):
            dev, pdev = score_slice(joined, topology_class, 2023, i * 100)
            conf, pconf = score_slice(joined, topology_class, 2024, i * 100 + 1)
            metrics[topology_class] = {"2023": dev, "2024": conf}
            for season, p in ((2023, pdev), (2024, pconf)):
                if not p.empty:
                    p = p.copy()
                    p["topology_class"] = topology_class
                    p["season"] = season
                    player_parts.append(p)
            cg, ok = qualify(dev, conf)
            gates[topology_class] = cg
            dispositions[topology_class] = (
                f"WR_OPPORTUNITY_BIAS_{topology_class}_ELIGIBLE"
                if ok else f"WR_OPPORTUNITY_BIAS_{topology_class}_CLOSED"
            )

        eligible = [k for k, v in dispositions.items() if v.endswith("_ELIGIBLE")]
        if len(eligible) == 2:
            overall = "PRISM_WR_OPPORTUNITY_BIAS_TOPOLOGY_ELIGIBLE_BOTH"
        elif len(eligible) == 1:
            overall = f"PRISM_WR_OPPORTUNITY_BIAS_TOPOLOGY_ELIGIBLE_{eligible[0]}_ONLY"
        else:
            overall = "PRISM_WR_OPPORTUNITY_BIAS_TOPOLOGY_CLOSED_ALL"

        joined.to_csv(
            args.out_dir / "wr_opportunity_bias_topology_signal_rows.csv", index=False
        )
        if player_parts:
            pd.concat(player_parts, ignore_index=True, sort=False).to_csv(
                args.out_dir / "wr_opportunity_bias_topology_player_diagnostics.csv", index=False
            )

        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "signal_run": SIGNAL_RUN,
            "signal_artifact": SIGNAL_ARTIFACT,
            "signal_digest": SIGNAL_DIGEST,
            "topology_run": TOPOLOGY_RUN,
            "topology_artifact": TOPOLOGY_ARTIFACT,
            "topology_digest": TOPOLOGY_DIGEST,
            "source_recovery_run_id": int(args.source_recovery_run_id),
            "source_recovery_disposition": recovery["disposition"],
            "minimum_rows": MIN_ROWS,
            "minimum_players": MIN_PLAYERS,
            "metrics": metrics,
            "qualification_gates": gates,
            "class_dispositions": dispositions,
            "disposition": overall,
            "projection_means_changed": 0,
            "opportunity_entitlements_changed": 0,
            "conservation_rooms_changed": 0,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "source_recovery_run_id": int(args.source_recovery_run_id),
            "disposition": "PRISM_WR_OPPORTUNITY_BIAS_TOPOLOGY_ELIGIBILITY_INTEGRITY_FAILURE",
            "error": str(exc),
            "projection_means_changed": 0,
            "opportunity_entitlements_changed": 0,
            "conservation_rooms_changed": 0,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
        out = args.out_dir / "wr_opportunity_bias_topology_eligibility_result.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    out = args.out_dir / "wr_opportunity_bias_topology_eligibility_result.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
