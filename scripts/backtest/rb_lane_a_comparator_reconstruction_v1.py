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


def load_rotation_market_weights(rotation: int, market: str) -> pd.DataFrame:
    """Return a one-row weights frame for the given rotation and market.

    Hard-asserts `calibration_season_used < test_season` (Amendment 3) using
    each source file's own recorded provenance -- never silently reuses the
    wrong-season file. Fails closed (raises) rather than guessing. Amendment 8
    extends this same per-rotation weight-file provenance rule to `rush_att`
    (not just `rush_yards`), for the held-incumbent-efficiency translation.
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
    row = weights.loc[weights["market"].astype(str).str.lower() == market.lower()]
    if row.empty:
        raise RuntimeError(f"no {market} row in {path}")
    if len(row) > 1:
        raise RuntimeError(f"multiple {market} rows in {path}, ambiguous")

    # Hard per-rotation provenance assertion (Amendment 3, extended to
    # rush_att by Amendment 8).
    if expected_calibration_season >= test_season:
        raise RuntimeError(
            f"rotation {rotation}: calibration_season_used={expected_calibration_season} "
            f"is not strictly prior to test_season={test_season} -- temporal leak, refusing to proceed"
        )

    return row.reset_index(drop=True)


def load_rotation_rush_yards_weights(rotation: int) -> pd.DataFrame:
    """Return a one-row `rush_yards` weights frame for the given rotation."""
    return load_rotation_market_weights(rotation, "rush_yards")


def build_promotion_comparator(
    component_predictions: pd.DataFrame, rotation: int, market: str = "rush_yards"
) -> pd.DataFrame:
    """Reconstruct the decisive promotion comparator (`ensemble_proj`) for one market.

    `component_predictions` must already carry `market`, `mc_proj`, `ml_proj`,
    `state_proj` for the rotation's test season (built via the frozen-parent
    `component_predictions.py`/`walk_forward.py` pipeline -- not built here).
    Returns only rows matching `market` with the reconstructed
    `promotion_comparator_{market}` column attached. Amendment 8 reuses this
    for `rush_att` as well as `rush_yards`, to hold incumbent efficiency fixed.
    """
    blob_check = verify_frozen_parent_blobs()
    if blob_check["disposition"] != "PASS":
        raise RuntimeError(f"frozen-parent blob mismatch, refusing to proceed: {blob_check['mismatches']}")

    weights_row = load_rotation_market_weights(rotation, market)

    if component_predictions is None or component_predictions.empty:
        return pd.DataFrame()
    frame = component_predictions.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    rush = frame.loc[frame["market"].astype(str).str.lower() == market.lower()].copy()
    if rush.empty:
        return rush

    ensembled = apply_ensemble(rush, weights=weights_row)
    ensembled[f"promotion_comparator_{market}"] = pd.to_numeric(
        ensembled["ensemble_proj"], errors="coerce"
    )
    ensembled["promotion_comparator_rotation"] = rotation
    ensembled["promotion_comparator_weight_source"] = (
        str(ROTATION_1_2023_FIT_WEIGHTS) if rotation == 1 else str(PRODUCTION_WEIGHTS)
    )
    return ensembled


PROMOTION_IDENTITY_KEYS = ["season", "week", "team", "player_clean_key"]


def build_dual_market_promotion_comparator(component_predictions: pd.DataFrame, rotation: int) -> pd.DataFrame:
    """Amendment 8: reconstruct both `promotion_rush_att` and
    `promotion_rush_yards` from the same component source, joined on exact
    player identity. Asserts zero duplicates and zero ambiguous joins on
    either side before merging -- fails closed rather than silently
    many-to-one/one-to-many joining.
    """
    att = build_promotion_comparator(component_predictions, rotation, market="rush_att")
    yards = build_promotion_comparator(component_predictions, rotation, market="rush_yards")

    for label, frame in (("rush_att", att), ("rush_yards", yards)):
        if frame is None or frame.empty:
            raise RuntimeError(f"dual-market promotion comparator: {label} reconstruction is empty")
        if frame.duplicated(PROMOTION_IDENTITY_KEYS).any():
            raise RuntimeError(f"dual-market promotion comparator: {label} has duplicate identity rows")

    att_slim = att[PROMOTION_IDENTITY_KEYS + ["promotion_comparator_rush_att"]].rename(
        columns={"promotion_comparator_rush_att": "promotion_rush_att"}
    )
    yards_slim = yards[PROMOTION_IDENTITY_KEYS + ["promotion_comparator_rush_yards"]].rename(
        columns={"promotion_comparator_rush_yards": "promotion_rush_yards"}
    )
    merged = att_slim.merge(
        yards_slim, on=PROMOTION_IDENTITY_KEYS, how="outer", validate="one_to_one"
    )
    merged["rotation"] = rotation
    return merged


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


# Amendment 7: mechanism comparator (P3/STACK2) authority-exact reconstruction,
# Rotation 2 (2024-fit/2025-eval) only -- no canonical 2023-fit/2024-eval STACK2
# casebook exists, so Rotation 1 discloses NOT_CONSTRUCTIBLE_NO_CASEBOOK per the
# plan's Amendment-7 text, never gating BASELINE_RECONSTRUCTION_FAILURE alone.
#
# The mechanism comparator's frozen formula (`enriched_att * stack_implied_ypc`,
# `compose_p3_row()` in rb_rush_synthesis_v1.py) is already precomputed, byte for
# byte, as the STACK2 casebook's own `arch_enriched_opp_stack_eff_yards` column
# (evaluate_rb_stack2_enriched_allocation.py::add_projection_arms) -- no separate
# recomposition is performed here; this only proves the fresh rebuild reproduces
# that already-frozen column against the canonical casebook (run `33538770934`,
# cited directly in rb_rush_synthesis_v1.py's own docstring).
MECHANISM_COMPARATOR_SCRIPT_BLOB = "7baa23a2db32fac60cdeffec875e20b4f6176d25"
MECHANISM_COMPARATOR_COLUMN = "arch_enriched_opp_stack_eff_yards"
MECHANISM_PARITY_JOIN_KEYS = ["season", "week", "team", "name_key"]
MECHANISM_CANONICAL_RUN_ID = "33538770934"
MECHANISM_FROZEN_PARENT_RUN_IDS = {
    "m94c": "33353485070",
    "stack1": "33535308110",
    "rb_market": "33499129109",
}


def compare_mechanism_comparator_parity(fresh: pd.DataFrame, canonical: pd.DataFrame) -> dict:
    """Authority-exact reconstruction proof for the mechanism comparator
    (Rotation 2 only, per Amendment 7). Same identity/value-parity discipline
    as `compare_component_predictions_parity()`, applied to STACK2's own
    identity keys (`season, week, team, name_key`) and its single precomputed
    mechanism-comparator column.
    """
    required = set(MECHANISM_PARITY_JOIN_KEYS) | {MECHANISM_COMPARATOR_COLUMN}
    for label, frame in (("fresh", fresh), ("canonical", canonical)):
        missing = required - set(frame.columns)
        if missing:
            return {
                "disposition": "MECHANISM_PARITY_FAILURE",
                "reason": f"{label} frame missing required columns: {sorted(missing)}",
            }

    f = fresh[MECHANISM_PARITY_JOIN_KEYS + [MECHANISM_COMPARATOR_COLUMN]].copy()
    c = canonical[MECHANISM_PARITY_JOIN_KEYS + [MECHANISM_COMPARATOR_COLUMN]].copy()
    f = f.drop_duplicates(MECHANISM_PARITY_JOIN_KEYS, keep="last")
    c = c.drop_duplicates(MECHANISM_PARITY_JOIN_KEYS, keep="last")

    merged = f.merge(
        c, on=MECHANISM_PARITY_JOIN_KEYS, how="outer",
        suffixes=("_fresh", "_canonical"), indicator=True,
    )
    matched = merged.loc[merged["_merge"] == "both"]
    fresh_only = int((merged["_merge"] == "left_only").sum())
    canonical_only = int((merged["_merge"] == "right_only").sum())

    a = pd.to_numeric(matched[f"{MECHANISM_COMPARATOR_COLUMN}_fresh"], errors="coerce")
    b = pd.to_numeric(matched[f"{MECHANISM_COMPARATOR_COLUMN}_canonical"], errors="coerce")
    delta = (a - b).abs()
    max_abs_delta = float(delta.max()) if len(delta) else float("nan")

    tolerance = 1e-6
    value_mismatch = (not pd.isna(max_abs_delta)) and max_abs_delta > tolerance
    disposition = (
        "MECHANISM_PARITY_FAILURE"
        if (fresh_only or canonical_only or value_mismatch)
        else "MECHANISM_AUTHORITY_RECONSTRUCTION_PASS"
    )

    return {
        "disposition": disposition,
        "rows_matched": int(len(matched)),
        "rows_fresh_only": fresh_only,
        "rows_canonical_only": canonical_only,
        "max_abs_value_delta": {MECHANISM_COMPARATOR_COLUMN: max_abs_delta},
        "tolerance": tolerance,
    }
