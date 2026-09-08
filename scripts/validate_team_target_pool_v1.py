#!/usr/bin/env python3
"""Audit current-team target entitlement before Monte Carlo allocation.

The joint simulator conserves team pass attempts, but conservation alone can hide
an invalid entitlement layer: when player target shares sum well above 1.0 the
simulator uniformly rescales the entire room to 0.95. That keeps arithmetic valid
while silently suppressing every player's current entitlement.

This audit is football-only. It records raw PlayerForm/Bayesian/rules target-pool
mass by team and fails closed when the final pre-simulation rules pool exceeds the
physical 1.0 target-share budget. It does not normalize shares and uses no
sportsbook input; repairing the entitlement model requires separate scientific
validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
RULE_INPUTS = DATA / "model_rule_simulation_inputs.csv"
BAYES = DATA / "model_bayesian_diagnostics.csv"
CONTEXT = DATA / "model_context_bridge.csv"
OUT_CSV = DATA / "team_target_pool_audit.csv"
OUT_JSON = DATA / "team_target_pool_audit.json"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"target-pool audit required artifact missing/empty: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if df.empty:
        raise RuntimeError(f"target-pool audit artifact has zero rows: {path}")
    return df


def _pool(frame: pd.DataFrame, share_col: str, label: str) -> pd.DataFrame:
    if "team" not in frame.columns or share_col not in frame.columns:
        raise RuntimeError(f"{label} missing team/{share_col}")
    # Metrics/pricing may repeat a player across markets/books. One current player
    # can contribute to the entitlement pool only once.
    player_key = "player_identity_key" if "player_identity_key" in frame.columns else "player_clean_key" if "player_clean_key" in frame.columns else "player"
    if player_key not in frame.columns:
        raise RuntimeError(f"{label} has no player identity key")
    x = frame[["team", player_key, share_col]].copy()
    x[share_col] = pd.to_numeric(x[share_col], errors="coerce")
    x = x.drop_duplicates(["team", player_key], keep="last")
    grouped = x.groupby("team", as_index=False).agg(
        **{
            f"{label}_sum": (share_col, lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())),
            f"{label}_players_with_share": (share_col, lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        }
    )
    return grouped


def main() -> int:
    rules = _read(RULE_INPUTS)
    bayes = _read(BAYES)
    context = _read(CONTEXT)

    # Bayesian diagnostics are one row/player; context is current PlayerForm one
    # row/player. Rules inputs repeat offers and therefore require dedup above.
    r = _pool(rules, "rules_tgt_share", "rules")
    bcol = "bayes_tgt_share" if "bayes_tgt_share" in bayes.columns else "tgt_share_posterior" if "tgt_share_posterior" in bayes.columns else None
    if bcol is None:
        raise RuntimeError("Bayesian diagnostics expose no target-share posterior")
    b = _pool(bayes, bcol, "bayes")
    ccol = "tgt_share" if "tgt_share" in context.columns else "target_share" if "target_share" in context.columns else None
    if ccol is None:
        raise RuntimeError("model context exposes no current target-share baseline")
    c = _pool(context, ccol, "playerform")

    out = r.merge(b, on="team", how="outer").merge(c, on="team", how="outer")
    if len(out) != 32 or out["team"].nunique() != 32:
        raise RuntimeError(f"target-pool audit expected 32 teams; rows={len(out)} teams={out['team'].nunique()}")
    for col in ("rules_sum", "bayes_sum", "playerform_sum"):
        vals = pd.to_numeric(out[col], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals).all() or (vals < 0).any():
            raise RuntimeError(f"target-pool audit invalid values in {col}")

    # 1.0 is the physical player target-share budget. The simulator reserves up
    # to a residual bucket internally, but an entitlement layer >1.0 is invalid
    # before that downstream allocation rule is applied.
    out["rules_pool_exceeds_one"] = pd.to_numeric(out["rules_sum"], errors="coerce").gt(1.0 + 1e-9).astype(int)
    out["bayes_pool_exceeds_one"] = pd.to_numeric(out["bayes_sum"], errors="coerce").gt(1.0 + 1e-9).astype(int)
    out["playerform_pool_exceeds_one"] = pd.to_numeric(out["playerform_sum"], errors="coerce").gt(1.0 + 1e-9).astype(int)
    out["simulator_uniform_scale_if_capped_095"] = np.where(
        pd.to_numeric(out["rules_sum"], errors="coerce") > 0.95,
        0.95 / pd.to_numeric(out["rules_sum"], errors="coerce"),
        1.0,
    )
    out = out.sort_values("rules_sum", ascending=False).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    bad_rules = out.loc[out["rules_pool_exceeds_one"].eq(1)]
    payload = {
        "disposition": "TARGET_ENTITLEMENT_POOL_INVALID_RESEARCH_REPAIR_REQUIRED" if not bad_rules.empty else "TARGET_ENTITLEMENT_POOL_VALID",
        "teams": int(len(out)),
        "teams_rules_over_1": int(len(bad_rules)),
        "teams_bayes_over_1": int(out["bayes_pool_exceeds_one"].sum()),
        "teams_playerform_over_1": int(out["playerform_pool_exceeds_one"].sum()),
        "rules_sum_min": float(out["rules_sum"].min()),
        "rules_sum_median": float(out["rules_sum"].median()),
        "rules_sum_mean": float(out["rules_sum"].mean()),
        "rules_sum_max": float(out["rules_sum"].max()),
        "minimum_uniform_scale_applied_by_simulator": float(out["simulator_uniform_scale_if_capped_095"].min()),
        "sportsbook_inputs_used": False,
        "normalization_applied_by_audit": False,
        "audit": str(OUT_CSV),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[team_target_pool] " + json.dumps(payload, sort_keys=True))
    if not bad_rules.empty:
        raise SystemExit(
            f"Current receiving entitlement pool is physically invalid for {len(bad_rules)}/32 teams; "
            f"see {OUT_CSV}. No automatic normalization was applied."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
