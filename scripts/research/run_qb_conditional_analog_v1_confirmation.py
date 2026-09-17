#!/usr/bin/env python3
"""Two-phase real-data orchestration for QB Conditional Analog V1.

PREPARE is outcome-blind: it reads only identity/scope columns from the Vegas
artifact and the frozen 17 pregame features from the exact QB-PD3 authority
casebook. It freezes all geometry needed for the blind confirmation.

CONFIRM is the single outcome-opening step: it reads unit_result once, derives
the frozen 2024 neighbor-win rule, applies it unchanged to the already-frozen
2025 neighborhoods, and emits the preregistered PASS/FAIL disposition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

JOIN_KEYS = ["season", "week", "team", "player_clean_key"]
FEATURE_COLUMNS = [
    "component_sd", "component_range", "pred_attempts", "pred_ypa",
    "qb_prior_attempts", "qb_prior_ypa",
    "def_pass_epa_allowed", "def_success_allowed", "def_ypa_allowed", "def_pass_rate_faced",
    "market_total", "market_spread", "market_abs_spread", "market_team_implied",
    "market_opp_implied", "market_is_underdog", "market_moneyline",
]
K = 15
DENSITY_PCT = 90.0
MIN_N = 40
ARM = "CURRENT_PRODUCTION_ORDER"
POSITION = "QB"
MARKET = "pass_yards"
SUPPORTED = "QB_CONDITIONAL_ANALOG_RELIABILITY_SUPPORTED"
NOT_ACTIONABLE = "NO_ACTIONABLE_QB_CONDITIONAL_ANALOG_RELIABILITY"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(d * d, axis=2))


def knearest(d: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    if d.shape[1] < k:
        raise RuntimeError(f"reference pool too small for k={k}: {d.shape[1]}")
    idx = np.argsort(d, axis=1)[:, :k]
    vals = np.take_along_axis(d, idx, axis=1)
    return idx, vals


def _scope_identity(vegas: Path) -> pd.DataFrame:
    """Read only outcome-blind scope/identity columns from the real Vegas artifact."""
    cols = JOIN_KEYS + ["benchmark_arm", "position", "market"]
    v = pd.read_csv(vegas, usecols=cols, low_memory=False)
    v = v.loc[v.benchmark_arm.eq(ARM) & v.position.eq(POSITION) & v.market.eq(MARKET)].copy()
    if v.duplicated(JOIN_KEYS).any():
        raise RuntimeError("scoped Vegas rows duplicate join keys")
    return v


def prepare(vegas: Path, features: Path, out: Path) -> dict:
    """Freeze all 2024-fit / 2025-application geometry without reading an outcome column."""
    out.mkdir(parents=True, exist_ok=True)
    v = _scope_identity(vegas)
    f = pd.read_csv(features, usecols=JOIN_KEYS + FEATURE_COLUMNS, low_memory=False)
    if f.duplicated(JOIN_KEYS).any():
        raise RuntimeError("feature casebook duplicates join keys")

    joined = v.merge(f, on=JOIN_KEYS, how="inner", validate="one_to_one")
    unmatched = v[JOIN_KEYS].merge(
        f[JOIN_KEYS], on=JOIN_KEYS, how="left", indicator=True, validate="one_to_one"
    )
    unmatched = unmatched.loc[unmatched._merge.eq("left_only"), JOIN_KEYS]

    miss = joined[FEATURE_COLUMNS].isna().any(axis=1)
    excluded_missing = joined.loc[miss, JOIN_KEYS].copy()
    clean = joined.loc[~miss].reset_index(drop=True)
    ref = clean.loc[clean.season.eq(2024)].reset_index(drop=True)
    ev = clean.loc[clean.season.eq(2025)].reset_index(drop=True)
    if ref.empty or ev.empty:
        raise RuntimeError("expected both 2024 reference and 2025 evaluation rows")

    scaler = StandardScaler().fit(ref[FEATURE_COLUMNS].to_numpy(float))
    refz = scaler.transform(ref[FEATURE_COLUMNS].to_numpy(float))
    evz = scaler.transform(ev[FEATURE_COLUMNS].to_numpy(float))

    rr = pairwise(refz, refz)
    np.fill_diagonal(rr, np.inf)
    ref_idx, ref_d = knearest(rr)
    threshold = float(np.percentile(ref_d[:, -1], DENSITY_PCT))

    er = pairwise(evz, refz)
    ev_idx, ev_d = knearest(er)
    ref_density = ref_d[:, -1] <= threshold
    ev_density = ev_d[:, -1] <= threshold

    ref_meta = ref[JOIN_KEYS].copy()
    ref_meta["row_index"] = np.arange(len(ref))
    ref_meta["kth_distance"] = ref_d[:, -1]
    ref_meta["density_pass"] = ref_density
    ev_meta = ev[JOIN_KEYS].copy()
    ev_meta["row_index"] = np.arange(len(ev))
    ev_meta["kth_distance"] = ev_d[:, -1]
    ev_meta["density_pass"] = ev_density

    ref_meta.to_csv(out / "reference_2024_preoutcome.csv", index=False)
    ev_meta.to_csv(out / "evaluation_2025_preoutcome.csv", index=False)
    unmatched.to_csv(out / "unmatched_scope_rows.csv", index=False)
    excluded_missing.to_csv(out / "excluded_missing_feature_rows.csv", index=False)
    np.save(out / "reference_neighbor_indices.npy", ref_idx)
    np.save(out / "evaluation_neighbor_indices.npy", ev_idx)
    np.save(out / "scaler_mean.npy", scaler.mean_)
    np.save(out / "scaler_scale.npy", scaler.scale_)

    manifest = {
        "phase": "PREPARED_OUTCOME_BLIND",
        "vegas_sha256": sha256(vegas),
        "features_sha256": sha256(features),
        "feature_source_contract": "QB_PD3_CASEBOOK_EXACT_AUTHORITY_LINEAGE",
        "feature_columns": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "k": K,
        "density_percentile": DENSITY_PCT,
        "density_threshold": threshold,
        "scope_rows": int(len(v)),
        "joined_rows": int(len(joined)),
        "unmatched_rows": int(len(unmatched)),
        "missing_feature_rows": int(miss.sum()),
        "clean_rows": int(len(clean)),
        "reference_2024_rows": int(len(ref)),
        "evaluation_2025_rows": int(len(ev)),
        "reference_density_pass_rows": int(ref_density.sum()),
        "evaluation_density_pass_rows": int(ev_density.sum()),
        "outcome_values_read": False,
        "rescue_authorized": False,
    }
    (out / "preoutcome_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _outcomes(vegas: Path) -> pd.DataFrame:
    """Single confirmation-only reader for the realized flat-unit result."""
    cols = JOIN_KEYS + ["benchmark_arm", "position", "market", "unit_result"]
    x = pd.read_csv(vegas, usecols=cols, low_memory=False)
    x = x.loc[x.benchmark_arm.eq(ARM) & x.position.eq(POSITION) & x.market.eq(MARKET)].copy()
    if x.duplicated(JOIN_KEYS).any():
        raise RuntimeError("outcome rows duplicate join keys")
    x["unit_result"] = pd.to_numeric(x.unit_result, errors="coerce")
    if x.unit_result.isna().any():
        raise RuntimeError("unit_result missing in scoped outcome rows")
    return x


def confirm(vegas: Path, prepared: Path, out: Path) -> dict:
    """Open outcomes exactly once and evaluate the frozen 2025 confirmation gates."""
    manifest = json.loads((prepared / "preoutcome_manifest.json").read_text())
    if manifest.get("phase") != "PREPARED_OUTCOME_BLIND":
        raise RuntimeError("invalid prepared manifest")
    if sha256(vegas) != manifest["vegas_sha256"]:
        raise RuntimeError("Vegas artifact hash drifted since PREPARE")

    ref = pd.read_csv(prepared / "reference_2024_preoutcome.csv")
    ev = pd.read_csv(prepared / "evaluation_2025_preoutcome.csv")
    ref_idx = np.load(prepared / "reference_neighbor_indices.npy")
    ev_idx = np.load(prepared / "evaluation_neighbor_indices.npy")
    if len(ref) != ref_idx.shape[0] or len(ev) != ev_idx.shape[0]:
        raise RuntimeError("prepared neighbor shape mismatch")

    # SINGLE OUTCOME-OPENING BOUNDARY. No outcome column is read before this line.
    outcomes = _outcomes(vegas)
    o24 = outcomes.loc[outcomes.season.eq(2024), JOIN_KEYS + ["unit_result"]]
    o25 = outcomes.loc[outcomes.season.eq(2025), JOIN_KEYS + ["unit_result"]]
    ref = ref.merge(o24, on=JOIN_KEYS, how="left", validate="one_to_one")
    ev = ev.merge(o25, on=JOIN_KEYS, how="left", validate="one_to_one")
    if ref.unit_result.isna().any() or ev.unit_result.isna().any():
        raise RuntimeError("prepared rows missing outcomes at confirmation")

    ref_win = (ref.unit_result.to_numpy(float) > 0).astype(float)
    neighbor_win_rate_ref = ref_win[ref_idx].mean(axis=1)
    neighbor_win_rate_ev = ref_win[ev_idx].mean(axis=1)
    ref_dir = neighbor_win_rate_ref >= 0.5
    ev_dir = neighbor_win_rate_ev >= 0.5
    ref_density = ref.density_pass.astype(bool).to_numpy()
    ev_density = ev.density_pass.astype(bool).to_numpy()
    ref_supported = ref_density & ref_dir
    ev_supported = ev_density & ev_dir

    n25 = int(ev_supported.sum())
    roi25 = float(ev.loc[ev_supported, "unit_result"].mean()) if n25 else float("nan")
    baseline25 = float(o25.unit_result.mean())
    n24 = int(ref_supported.sum())
    roi24 = float(ref.loc[ref_supported, "unit_result"].mean()) if n24 else float("nan")

    gates = {
        "n_2025_ge_40": n25 >= MIN_N,
        "roi_2025_positive": bool(n25 and roi25 > 0),
        "roi_2025_beats_unconditional_baseline": bool(n25 and roi25 > baseline25),
        "roi_2024_supported_positive": bool(n24 and roi24 > 0),
    }
    passed = all(gates.values())
    disposition = SUPPORTED if passed else NOT_ACTIONABLE

    ref["neighbor_win_rate_2024"] = neighbor_win_rate_ref
    ref["candidate_supported_pre_gate"] = ref_supported
    ev["neighbor_win_rate_2024"] = neighbor_win_rate_ev
    ev["candidate_supported_pre_gate"] = ev_supported
    if passed:
        ev["final_evidence_class"] = np.where(
            ~ev_density, "NO_ANALOG_SUPPORT",
            np.where(ev_supported, "SUPPORTED", "DESCRIPTIVE_ONLY"),
        )
    else:
        ev["final_evidence_class"] = np.where(
            ~ev_density, "NO_ANALOG_SUPPORT", "DESCRIPTIVE_ONLY"
        )

    out.mkdir(parents=True, exist_ok=True)
    ref.to_csv(out / "reference_2024_confirmed.csv", index=False)
    ev.to_csv(out / "evaluation_2025_confirmed.csv", index=False)
    result = {
        "disposition": disposition,
        "gates": gates,
        "n_2024_supported": n24,
        "roi_2024_supported": roi24,
        "n_2025_supported": n25,
        "roi_2025_supported": roi25,
        "roi_2025_unconditional_baseline": baseline25,
        "reference_2024_rows": int(len(ref)),
        "evaluation_2025_rows": int(len(ev)),
        "k": K,
        "density_percentile": DENSITY_PCT,
        "density_threshold": manifest["density_threshold"],
        "rescue_authorized": False,
        "production_mutation_authorized": False,
    }
    (out / "QB_CONDITIONAL_ANALOG_V1_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--vegas-csv", type=Path, required=True)
    a.add_argument("--features-csv", type=Path, required=True)
    a.add_argument("--out-dir", type=Path, required=True)
    b = sub.add_parser("confirm")
    b.add_argument("--vegas-csv", type=Path, required=True)
    b.add_argument("--prepared-dir", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    z = p.parse_args()
    r = (
        prepare(z.vegas_csv, z.features_csv, z.out_dir)
        if z.cmd == "prepare"
        else confirm(z.vegas_csv, z.prepared_dir, z.out_dir)
    )
    print(json.dumps(r, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
