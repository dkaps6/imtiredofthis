#!/usr/bin/env python3
"""Build the frozen RB-PD2 forward difficulty-history artifact.

The 2025 seed comes only from leakage-safe historical component predictions.
Optional completed 2026 rows must already carry explicit pregame-lineage
certification; Week 1 additionally requires the documented P3/STACK1 parity
reassertion before it may enter later predictor history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.ensemble_v2 import apply_ensemble
from scripts.research.rb_pd2_forward_shadow_v1 import (
    ELIGIBLE_POSITIONS,
    build_history_manifest,
    build_history_state,
    verify_frozen_2025_weights,
)

DEFAULT_COMPONENTS = Path("data/backtests/component_predictions.csv")
DEFAULT_WEIGHTS = Path("data/model_ensemble_weights.csv")
WEEK1_PARITY_TOLERANCE = 1e-8


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def build_2025_seed(components: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    verify_frozen_2025_weights(weights)
    x = components.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]

    required = {
        "season", "week", "team", "player_clean_key", "position", "market",
        "mc_proj", "ml_proj", "state_proj", "actual",
        "prediction_cutoff", "prior_season",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"2025 components missing lineage columns: {sorted(missing)}")

    season = pd.to_numeric(x["season"], errors="coerce")
    week = pd.to_numeric(x["week"], errors="coerce")
    pos = x["position"].fillna("").astype(str).str.upper().str.strip()
    market = x["market"].fillna("").astype(str).str.lower().str.strip()
    q = x.loc[
        season.eq(2025)
        & week.between(1, 18)
        & pos.isin(ELIGIBLE_POSITIONS)
        & market.eq("rush_yards")
    ].copy()
    if q.empty:
        raise RuntimeError("no eligible 2025 RB/HB/FB rush_yards component rows")

    prior = pd.to_numeric(q["prior_season"], errors="coerce")
    if not prior.eq(2024).all():
        raise RuntimeError("2025 component row does not prove prior_season=2024")
    cutoff = q["prediction_cutoff"].fillna("").astype(str).str.lower()
    if not cutoff.str.contains("pregame", regex=False).all():
        raise RuntimeError("2025 component row lacks explicit pregame prediction cutoff")

    # Construct the football mean from components only. Realized outcomes are
    # intentionally not present in this frame passed to apply_ensemble().
    pred_input = q[["market", "mc_proj", "ml_proj", "state_proj"]].copy()
    rush_weights = weights.loc[
        weights["market"].astype(str).str.lower().eq("rush_yards")
    ].copy()
    ensemble = apply_ensemble(pred_input, weights=rush_weights)
    q["projection_mean"] = pd.to_numeric(ensemble["ensemble_proj"], errors="coerce")
    q["actual_rush_yards"] = pd.to_numeric(q["actual"], errors="coerce")
    q["pregame_lineage_certified"] = True
    q["projection_lineage"] = (
        "2025_GENERIC_ENSEMBLE_2024_OOS_FROZEN|"
        + q["prediction_cutoff"].astype(str)
    )

    if not np.isfinite(q["projection_mean"]).all():
        raise RuntimeError("non-finite 2025 reconstructed projection mean")
    if not np.isfinite(q["actual_rush_yards"]).all():
        raise RuntimeError("non-finite 2025 completed rushing outcome")

    keep = [
        "season", "week", "team", "opponent", "event_id", "player",
        "player_clean_key", "position", "projection_mean", "actual_rush_yards",
        "pregame_lineage_certified", "projection_lineage",
    ]
    for col in keep:
        if col not in q.columns:
            q[col] = ""
    return q[keep].copy()


def validate_additional_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    x = frame.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "team", "player_clean_key", "position",
        "projection_mean", "actual_rush_yards", "pregame_lineage_certified",
        "projection_lineage",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"additional history missing columns: {sorted(missing)}")

    season = pd.to_numeric(x["season"], errors="coerce")
    week = pd.to_numeric(x["week"], errors="coerce")
    if not season.eq(2026).all():
        raise RuntimeError("additional history may contain completed 2026 rows only")
    cert = x["pregame_lineage_certified"].astype(str).str.strip().str.lower()
    if not cert.isin({"true", "1"}).all():
        raise RuntimeError("additional 2026 history contains uncertified or malformed pregame lineage")
    if x["projection_lineage"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError("additional 2026 history missing projection_lineage")

    w1 = week.eq(1)
    if w1.any():
        parity_required = {
            "week1_p3_stack1_parity_pass",
            "week1_p3_projection",
            "week1_stack1_projection",
        }
        missing_parity = parity_required - set(x.columns)
        if missing_parity:
            raise RuntimeError(
                f"2026 Week 1 history requires mechanical P3/STACK1 parity inputs: {sorted(missing_parity)}"
            )

        # The flag is only an audit field; never trust it as the parity proof.
        # Parse it strictly so a string such as "False" cannot become truthy.
        flags = (
            x.loc[w1, "week1_p3_stack1_parity_pass"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        if not flags.isin({"true", "1"}).all():
            raise RuntimeError("2026 Week 1 P3/STACK1 parity audit flag did not pass")

        p3 = pd.to_numeric(x.loc[w1, "week1_p3_projection"], errors="coerce")
        stack = pd.to_numeric(x.loc[w1, "week1_stack1_projection"], errors="coerce")
        if p3.isna().any() or stack.isna().any():
            raise RuntimeError("2026 Week 1 P3/STACK1 parity inputs are non-finite")
        max_abs_diff = float((p3 - stack).abs().max())
        if max_abs_diff > WEEK1_PARITY_TOLERANCE:
            raise RuntimeError(
                f"2026 Week 1 P3/STACK1 parity recomputation failed max_abs_diff={max_abs_diff}"
            )

    return x


def build_artifacts(
    *,
    components_path: Path,
    weights_path: Path,
    out_dir: Path,
    additional_history_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    components = _read(components_path, "2025 component predictions")
    weights = _read(weights_path, "ensemble weights")
    seed = build_2025_seed(components, weights)

    parts = [seed]
    if additional_history_path is not None:
        additional = _read(additional_history_path, "additional certified 2026 history")
        parts.append(validate_additional_history(additional))

    history = pd.concat(parts, ignore_index=True, sort=False)
    state = build_history_state(history)
    manifest = build_history_manifest(
        state,
        root=Path("."),
        source_label="2025 leakage-safe components + explicitly certified completed 2026 history",
    )
    manifest.update({
        "component_predictions_path": str(components_path),
        "component_predictions_sha256": _file_sha256(components_path),
        "ensemble_weights_path": str(weights_path),
        "ensemble_weights_sha256": _file_sha256(weights_path),
        "additional_history_path": str(additional_history_path) if additional_history_path else "",
        "additional_history_sha256": (
            _file_sha256(additional_history_path) if additional_history_path else ""
        ),
        "zero_prospective_outcomes_used": True,
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "rb_pd2_forward_history_state.csv"
    manifest_path = out_dir / "rb_pd2_forward_history_manifest.json"
    state.to_csv(state_path, index=False)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return state, manifest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--additional-certified-history", type=Path, default=None)
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("data/research/rb_pd2_forward/history"),
    )
    a = p.parse_args()
    state, manifest = build_artifacts(
        components_path=a.components,
        weights_path=a.weights,
        additional_history_path=a.additional_certified_history,
        out_dir=a.out_dir,
    )
    print(
        f"[rb-pd2-history] rows={len(state)} players={state['player_clean_key'].nunique()} "
        f"-> {a.out_dir}"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
