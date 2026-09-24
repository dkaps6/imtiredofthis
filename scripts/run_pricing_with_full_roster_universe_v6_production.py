#!/usr/bin/env python3
"""Full Slate V6 production candidate with RB rush+receiving conservation V2.

V6 preserves the certified V5 stack and activates the already-qualified V2
non-Week-1 RB/FB rush+receiving adapter inside run_pricing_v2. Week 1 remains
governed by P3/R22/R26 and V2 must be a no-op there.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v2 as v2
import scripts.run_pricing_with_full_roster_universe_v5_production as v5
from scripts.modeling.rb_rush_rec_conservation_v2 import (
    AUDIT_JSON as V2_SEAM_AUDIT,
    VERSION as V2_VERSION,
)
from scripts.runtime_context import resolve_week

OUT = Path("outputs/props_priced_clean.csv")
AUDIT = Path("data/rb_rush_rec_conservation_v2_production_audit.json")
DISPOSITION = "RB_RUSH_REC_CONSERVATION_V2_PRODUCTION_CERTIFIED"


def _stamp(*, week: int) -> dict:
    if not OUT.is_file():
        raise RuntimeError("V6 pricing output missing")
    if not V2_SEAM_AUDIT.is_file():
        raise RuntimeError("V6 V2 seam audit missing")

    priced = pd.read_csv(OUT, low_memory=False)
    seam = json.loads(V2_SEAM_AUDIT.read_text(encoding="utf-8"))
    required = {
        "event_id", "player", "team", "market", "model_proj",
        "rb_rush_rec_conservation_v2_applied",
        "rb_rush_rec_conservation_v2_version",
        "rb_rush_rec_conservation_v2_target_mean",
        "rb_rush_rec_conservation_v2_rush_mean",
        "rb_rush_rec_conservation_v2_rec_mean",
    }
    missing = required - set(priced.columns)
    if missing:
        raise RuntimeError(f"V6 priced output missing V2 lineage columns: {sorted(missing)}")

    applied = pd.to_numeric(
        priced["rb_rush_rec_conservation_v2_applied"], errors="coerce"
    ).fillna(0).eq(1)
    combo = priced["market"].astype(str).eq("rush_rec_yards")

    if int(week) == 1:
        if applied.any():
            sample = priced.loc[applied, ["player", "team", "market"]].head(20).to_dict("records")
            raise RuntimeError(f"V2 changed Week-1 rows: {sample}")
        payload = {
            "disposition": "RB_RUSH_REC_CONSERVATION_V2_NOT_APPLICABLE_WEEK1",
            "integration_valid": True,
            "version": V2_VERSION,
            "week": 1,
            "applied_rows": 0,
            "sportsbook_inputs_to_v2_football": int(seam.get("sportsbook_inputs_used", -1)),
            "week1_rows_changed": 0,
        }
        AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload

    if not (~applied | combo).all():
        sample = priced.loc[applied & ~combo, ["player", "team", "market"]].head(20).to_dict("records")
        raise RuntimeError(f"V2 applied outside rush_rec_yards: {sample}")

    applied_rows = priced.loc[applied].copy()
    if applied_rows.empty:
        combo_rows = priced.loc[combo]
        # A slate may legitimately have no RB/FB combo offers; the seam itself
        # remains valid as long as it reports no eligible rows.
        if not combo_rows.empty and seam.get("disposition") != "RB_RUSH_REC_CONSERVATION_V2_NO_ELIGIBLE_ROWS":
            raise RuntimeError(
                "V6 has combo offers but zero V2-applied rows without explicit no-eligible seam state"
            )
        payload = {
            "disposition": DISPOSITION,
            "integration_valid": True,
            "version": V2_VERSION,
            "week": int(week),
            "applied_rows": 0,
            "applied_player_games": 0,
            "max_final_vs_target_gap": 0.0,
            "max_target_vs_component_sum_gap": 0.0,
            "max_pathwise_identity_gap": float(seam.get("max_pathwise_identity_gap", 0.0)),
            "sportsbook_inputs_to_v2_football": int(seam.get("sportsbook_inputs_used", -1)),
            "week1_rows_changed": int(seam.get("week1_rows_changed", -1)),
        }
        AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload

    versions = set(applied_rows["rb_rush_rec_conservation_v2_version"].fillna("").astype(str))
    if versions != {V2_VERSION}:
        raise RuntimeError(f"V6 unexpected V2 version(s): {sorted(versions)}")

    target = pd.to_numeric(applied_rows["rb_rush_rec_conservation_v2_target_mean"], errors="coerce")
    rush = pd.to_numeric(applied_rows["rb_rush_rec_conservation_v2_rush_mean"], errors="coerce")
    rec = pd.to_numeric(applied_rows["rb_rush_rec_conservation_v2_rec_mean"], errors="coerce")
    final = pd.to_numeric(applied_rows["model_proj"], errors="coerce")
    if target.isna().any() or rush.isna().any() or rec.isna().any() or final.isna().any():
        raise RuntimeError("V6 V2 applied rows contain non-finite lineage means")

    final_gap = np.abs(final.to_numpy(float) - target.to_numpy(float))
    sum_gap = np.abs(target.to_numpy(float) - (rush + rec).to_numpy(float))
    max_final_gap = float(final_gap.max()) if len(final_gap) else 0.0
    max_sum_gap = float(sum_gap.max()) if len(sum_gap) else 0.0
    if max_final_gap > 1e-8:
        raise RuntimeError(f"V6 final combo mean bypassed V2 target: {max_final_gap}")
    if max_sum_gap > 1e-8:
        raise RuntimeError(f"V6 V2 target not equal component sum: {max_sum_gap}")

    if int(seam.get("sportsbook_inputs_used", -1)) != 0:
        raise RuntimeError(f"V6 V2 seam reports sportsbook input: {seam}")
    if int(seam.get("week1_rows_changed", -1)) != 0:
        raise RuntimeError(f"V6 V2 seam changed Week-1 rows: {seam}")
    path_gap = float(seam.get("max_pathwise_identity_gap", float("inf")))
    if not np.isfinite(path_gap) or path_gap > 1e-10:
        raise RuntimeError(f"V6 V2 pathwise identity failure: {path_gap}")

    canonical = applied_rows["player"].map(v2._suffix_safe_key)
    player_games = applied_rows.assign(_canonical=canonical).drop_duplicates(
        ["event_id", "team", "_canonical"]
    )

    payload = {
        "disposition": DISPOSITION,
        "integration_valid": True,
        "version": V2_VERSION,
        "week": int(week),
        "applied_rows": int(applied.sum()),
        "applied_player_games": int(len(player_games)),
        "max_final_vs_target_gap": max_final_gap,
        "max_target_vs_component_sum_gap": max_sum_gap,
        "max_pathwise_identity_gap": path_gap,
        "sportsbook_inputs_to_v2_football": 0,
        "week1_rows_changed": 0,
        "formula": "rush_rec_yards = final-mean-aligned rush_yards draws + final-mean-aligned rec_yards draws",
        "note": (
            "V2 changes only non-Week-1 RB/FB rush+receiving distributions and means. "
            "Standalone rush/receiving authorities and sportsbook inputs are unchanged."
        ),
    }
    AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    previous = os.environ.get("RB_RUSH_REC_CONSERVATION_V2")
    os.environ["RB_RUSH_REC_CONSERVATION_V2"] = "1"
    try:
        rc = int(v5.main())
    finally:
        if previous is None:
            os.environ.pop("RB_RUSH_REC_CONSERVATION_V2", None)
        else:
            os.environ["RB_RUSH_REC_CONSERVATION_V2"] = previous

    if rc != 0:
        return rc
    payload = _stamp(week=int(resolve_week()))
    print("[rb_rush_rec_conservation_v2_production] " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
