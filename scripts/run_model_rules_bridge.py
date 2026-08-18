#!/usr/bin/env python3
"""Materialize canonical rule diagnostics without requiring sportsbook props."""
from __future__ import annotations

from pathlib import Path

from scripts.modeling.simulation_rules import build_rule_diagnostics

OUT = Path("data/model_rule_diagnostics.csv")


def main() -> int:
    df = build_rule_diagnostics()
    if df.empty:
        raise RuntimeError("Canonical rule diagnostics produced 0 rows")
    required = {"player", "team", "opponent", "season", "week", "projected_plays", "pass_eff_mult", "rush_eff_mult"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Canonical rule diagnostics missing columns: {sorted(missing)}")
    if df[["team", "opponent"]].fillna("").eq("").any().any():
        raise RuntimeError("Canonical rule diagnostics contain unresolved team/opponent identity")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    changed = int(((df["pass_eff_mult"] != 1.0) | (df["rush_eff_mult"] != 1.0) |
                   (df["wr1_target_mult"] != 1.0) | (df["wr1_5_target_mult"] != 1.0) |
                   (df["slot_target_mult"] != 1.0) | (df["te_target_mult"] != 1.0) |
                   (df["rb_rec_target_mult"] != 1.0)).sum())
    print(f"[model_rules] players={len(df)} contexts_with_non_neutral_rule={changed}")
    print(f"[model_rules] wrote {len(df)} rows -> {OUT}")
    print("[model_rules] these are pre-simulation assumptions; no final projection multiplier is used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
