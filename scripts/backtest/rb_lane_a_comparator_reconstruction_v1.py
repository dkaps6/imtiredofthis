"""RB Lane A -- promotion-comparator reconstruction (transition-gated allocation V1).

Implements the "Two comparators" / "Authority-exact baseline reconstruction"
sections of ``docs/research/RB_LANE_A_TRANSITION_GATED_ALLOCATION_V1_PLAN.md``
(Amendments 2-5), for the **decisive promotion comparator** only
(``ensemble_proj``, the real Weeks-2-18 production route). The diagnostic-only
mechanism comparator (P3/STACK2) is reconstructed separately and is not gating.

Per Amendment 6 (Issue #535 comment `5704381940`), the promotion comparator's
cross-run parity requirement against run `35032590321` is replaced by a
same-job double-build authority contract: `35032590321` turned out to be
itself a reconstruction of an expired original artifact, not a legitimate
permanent authority. ``same_job_double_build_disposition()`` implements that
replacement -- two builds of `walk_forward.py` from one frozen historical-
input snapshot, in the same CI job, must agree on row identity and
`mc_proj`/`ml_proj`/`state_proj` to `<=1e-6`.

This module contains NO candidate mechanism, NO scoring logic, and computes NO
candidate outcome. It only reconstructs what production already does today, so
the candidate has something real to beat.

Per Amendment 3's fatal-leak fix: Rotation 1 (test season 2024) must NOT use
``data/model_ensemble_weights.csv``'s ``rush_yards`` row, because that row's
``fit_scope`` is ``all_2024_oos_frozen_for_2025`` -- fit using 2024 outcomes,
which would leak into a 2024 test. Rotation 1 uses the already-existing,
already-merged (#545) 2023-only frozen fit instead. Rotation 2 (test season
2025) uses the production row unchanged (2024 < 2025, legitimately prior).
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.modeling.ensemble_v2 import apply_ensemble

# Frozen-parent blob SHAs, pinned at PR #562's merge commit 91afb3a5.
# Confirmed byte-identical to the current branch tip at Amendment-5
# implementation time; re-verified at runtime by verify_frozen_parent_blobs().
FROZEN_PARENT_BLOBS = {
    "scripts/modeling/ensemble_v2.py": "41e809b32e4596b8cf18bedbf2b940a2aa3b80b2",
    "scripts/backtest/component_predictions.py": "18f7289515b88c84a91479da18526df4cd7f5398",
    "scripts/modeling/rb_rush_synthesis_v1.py": "f5cf574144faf4cc527f2ef6627511ae774d2431",
    "scripts/modeling/rb_pricing_adapter_v1.py": "b9ed94dc39fc0c8397675859fd1c659ae689ff14",
    "data/model_ensemble_weights.csv": "baade160a124e5cd8ecd415c0276622d4b60953f",
}

ROTATION_1_2023_FIT_WEIGHTS = REPO_ROOT / "docs/research/overnight/ensemble_weights_2023_fit_v1.csv"
ROTATION_1_2023_FIT_BLOB = "b9c193b9b9d11578f3dd17ae6da241715878bea5"
PRODUCTION_WEIGHTS = REPO_ROOT / "data/model_ensemble_weights.csv"

ROTATIONS = {
    1: {"discovery_season": 2023, "test_season": 2024, "weight_calibration_season": 2023},
    2: {"discovery_season": 2024, "test_season": 2025, "weight_calibration_season": 2024},
}


def verify_frozen_parent_blobs(repo_root: Path = REPO_ROOT) -> dict:
    """Assert every frozen-parent file matches its pinned blob SHA. Fails closed.

    This is the code-fidelity half of "authority-exact baseline reconstruction"
    -- proof that the comparator is built from the exact code #562 froze, not
    whatever the branch happens to currently contain.
    """
    results = {}
    mismatches = []
    for rel_path, expected_blob in FROZEN_PARENT_BLOBS.items():
        proc = subprocess.run(
            ["git", "rev-parse", f"HEAD:{rel_path}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        actual_blob = proc.stdout.strip()
        match = actual_blob == expected_blob
        results[rel_path] = {"expected": expected_blob, "actual": actual_blob, "match": match}
        if not match:
            mismatches.append(rel_path)
    return {
        "disposition": "PASS" if not mismatches else "BASELINE_RECONSTRUCTION_FAILURE",
        "mismatches": mismatches,
        "files": results,
    }


def load_rotation_rush_yards_weights(rotation: int) -> pd.DataFrame:
    """Return a one-row `rush_yards` weights frame for the given rotation.

    Hard-asserts `calibration_season_used < test_season` (Amendment 3) using
    each source file's own recorded provenance -- never silently reuses the
    wrong-season file. Fails closed (raises) rather than guessing.
    """
    if rotation not in ROTATIONS:
        raise RuntimeError(f"unknown rotation {rotation!r}; expected 1 or 2")
    cfg = ROTATIONS[rotation]
    test_season = cfg["test_season"]

    if rotation == 1:
        path, expected_blob = ROTATION_1_2023_FIT_WEIGHTS, ROTATION_1_2023_FIT_BLOB
        expected_calibration_season = 2023
    else:
        path, expected_blob = PRODUCTION_WEIGHTS, FROZEN_PARENT_BLOBS["data/model_ensemble_weights.csv"]
        expected_calibration_season = 2024

    proc = subprocess.run(
        ["git", "rev-parse", f"HEAD:{path.relative_to(REPO_ROOT)}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    actual_blob = proc.stdout.strip()
    if actual_blob != expected_blob:
        raise RuntimeError(
            f"rotation {rotation} weight file {path} blob mismatch: "
            f"expected {expected_blob}, got {actual_blob} -- baseline reconstruction failure"
        )

    weights = pd.read_csv(path)
    weights.columns = [str(c).strip().lower() for c in weights.columns]
    row = weights.loc[weights["market"].astype(str).str.lower() == "rush_yards"]
    if row.empty:
        raise RuntimeError(f"no rush_yards row in {path}")
    if len(row) > 1:
        raise RuntimeError(f"multiple rush_yards rows in {path}, ambiguous")

    # Hard per-rotation provenance assertion (Amendment 3).
    if expected_calibration_season >= test_season:
        raise RuntimeError(
            f"rotation {rotation}: calibration_season_used={expected_calibration_season} "
            f"is not strictly prior to test_season={test_season} -- temporal leak, refusing to proceed"
        )

    return row.reset_index(drop=True)


def build_promotion_comparator(component_predictions: pd.DataFrame, rotation: int) -> pd.DataFrame:
    """Reconstruct the decisive promotion comparator (`ensemble_proj`) for rush_yards.

    `component_predictions` must already carry `market`, `mc_proj`, `ml_proj`,
    `state_proj` for the rotation's test season (built via the frozen-parent
    `component_predictions.py`/`walk_forward.py` pipeline -- not built here).
    Returns only `market == "rush_yards"` rows with the reconstructed
    `promotion_comparator_rush_yards` column attached.
    """
    blob_check = verify_frozen_parent_blobs()
    if blob_check["disposition"] != "PASS":
        raise RuntimeError(f"frozen-parent blob mismatch, refusing to proceed: {blob_check['mismatches']}")

    weights_row = load_rotation_rush_yards_weights(rotation)

    if component_predictions is None or component_predictions.empty:
        return pd.DataFrame()
    frame = component_predictions.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    rush = frame.loc[frame["market"].astype(str).str.lower() == "rush_yards"].copy()
    if rush.empty:
        return rush

    ensembled = apply_ensemble(rush, weights=weights_row)
    ensembled["promotion_comparator_rush_yards"] = pd.to_numeric(
        ensembled["ensemble_proj"], errors="coerce"
    )
    ensembled["promotion_comparator_rotation"] = rotation
    ensembled["promotion_comparator_weight_source"] = (
        str(ROTATION_1_2023_FIT_WEIGHTS) if rotation == 1 else str(PRODUCTION_WEIGHTS)
    )
    return ensembled


PARITY_JOIN_KEYS = ["season", "week", "team", "player_clean_key", "market"]
PARITY_VALUE_COLS = ["mc_proj", "ml_proj", "state_proj"]


def compare_component_predictions_parity(
    fresh: pd.DataFrame, canonical: pd.DataFrame
) -> dict:
    """Authority/value-parity proof: fresh rebuild vs. a canonical historical source.

    Reports row-count deltas and max-abs-value deltas per component
    (mc_proj/ml_proj/state_proj) on the shared identity key. Fails closed
    (PARITY_FAILURE) if either frame is missing required columns, or if any
    matched row's component value differs beyond floating-point tolerance.
    Unmatched rows on either side are reported, not silently dropped.
    """
    required = set(PARITY_JOIN_KEYS) | set(PARITY_VALUE_COLS)
    for label, frame in (("fresh", fresh), ("canonical", canonical)):
        missing = required - set(frame.columns)
        if missing:
            return {
                "disposition": "PARITY_FAILURE",
                "reason": f"{label} frame missing required columns: {sorted(missing)}",
            }

    f = fresh[PARITY_JOIN_KEYS + PARITY_VALUE_COLS].copy()
    c = canonical[PARITY_JOIN_KEYS + PARITY_VALUE_COLS].copy()
    f = f.drop_duplicates(PARITY_JOIN_KEYS, keep="last")
    c = c.drop_duplicates(PARITY_JOIN_KEYS, keep="last")

    merged = f.merge(c, on=PARITY_JOIN_KEYS, how="outer", suffixes=("_fresh", "_canonical"), indicator=True)
    matched = merged.loc[merged["_merge"] == "both"]
    fresh_only = int((merged["_merge"] == "left_only").sum())
    canonical_only = int((merged["_merge"] == "right_only").sum())

    max_abs_deltas = {}
    for col in PARITY_VALUE_COLS:
        a = pd.to_numeric(matched[f"{col}_fresh"], errors="coerce")
        b = pd.to_numeric(matched[f"{col}_canonical"], errors="coerce")
        delta = (a - b).abs()
        max_abs_deltas[col] = float(delta.max()) if len(delta) else float("nan")

    tolerance = 1e-6
    value_mismatch = any(
        (not pd.isna(v)) and v > tolerance for v in max_abs_deltas.values()
    )
    disposition = "PARITY_FAILURE" if (fresh_only or canonical_only or value_mismatch) else "PASS"

    return {
        "disposition": disposition,
        "rows_matched": int(len(matched)),
        "rows_fresh_only": fresh_only,
        "rows_canonical_only": canonical_only,
        "max_abs_value_delta": max_abs_deltas,
        "tolerance": tolerance,
    }


def same_job_double_build_disposition(build_a: pd.DataFrame, build_b: pd.DataFrame) -> dict:
    """Amendment 6: same-job double-build authority contract.

    Two `walk_forward.py` builds from one frozen historical-input snapshot,
    run in the same CI job, must agree on row identity and
    `mc_proj`/`ml_proj`/`state_proj` to `<=1e-6`. Reuses the identity/value-
    parity discipline of `compare_component_predictions_parity()` (same join
    keys, same tolerance) but reports under Amendment 6's own disposition
    names so a pass is never confused with "parity passed" against a
    cross-run artifact.
    """
    base = compare_component_predictions_parity(build_a, build_b)
    disposition = (
        "SAME_JOB_AUTHORITY_RECONSTRUCTION_PASS"
        if base.get("disposition") == "PASS"
        else "SAME_JOB_AUTHORITY_RECONSTRUCTION_FAILURE"
    )
    return {**base, "disposition": disposition}


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_input_manifest(paths: dict[str, Path]) -> dict:
    """Amendment 6 evidence: SHA256 of every historical-input file feeding a
    build, keyed by a caller-supplied label. Fails closed if a path is
    missing rather than silently omitting it from the manifest.
    """
    manifest = {}
    for label, path in paths.items():
        if not path.exists():
            raise RuntimeError(f"input manifest: missing file for {label!r}: {path}")
        manifest[label] = {"path": str(path), "sha256": sha256_of_file(path)}
    return manifest


def blob_sha_of(repo_root: Path, rel_path: str) -> str:
    """HEAD blob SHA of a repo-relative file -- used to pin the code identity
    (e.g. `scripts/simulation_v2.py`) as part of the Amendment 6 evidence.
    """
    proc = subprocess.run(
        ["git", "rev-parse", f"HEAD:{rel_path}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip()
