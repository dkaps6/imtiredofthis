#!/usr/bin/env python3
"""Audit receiving target entitlement on the sportsbook-independent full roster.

V1 read the offer-derived rule-input frame.  This version reads the certified
football_simulation_universe artifact, where Ourlads + schedule define players
before any market lookup.  It applies no normalization and remains a model-
quality blocker when raw entitlement exceeds the physical target budget.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
UNIVERSE = DATA / "football_simulation_universe.csv"
UNIVERSE_AUDIT = DATA / "football_simulation_universe_audit.json"
OUT_CSV = DATA / "team_target_pool_audit.csv"
OUT_JSON = DATA / "team_target_pool_audit.json"
FORBIDDEN = {
    "line", "source_line", "over_odds", "under_odds", "book", "book_title",
    "vegas_line", "vegas_odds", "market_prob", "edge_pct", "edge_abs",
    "home_wp", "away_wp", "team_wp",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"full-universe target audit missing/empty: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise RuntimeError(f"full-universe target audit zero rows: {path}")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def main() -> int:
    frame = _read(UNIVERSE)
    if not UNIVERSE_AUDIT.exists() or UNIVERSE_AUDIT.stat().st_size <= 0:
        raise RuntimeError("full-universe certification JSON missing")
    source = json.loads(UNIVERSE_AUDIT.read_text(encoding="utf-8"))
    if source.get("disposition") != "FOOTBALL_SIMULATION_UNIVERSE_CERTIFIED":
        raise RuntimeError(f"football simulation universe not certified: {source.get('disposition')}")
    if source.get("sportsbook_inputs_used_to_generate_football_distributions") is not False:
        raise RuntimeError("football simulation universe does not certify sportsbook-independent generation")

    leaked = sorted(FORBIDDEN & set(frame.columns))
    if leaked:
        raise RuntimeError(f"sportsbook/market fields present in target-pool source: {leaked}")
    required = {"team", "player_clean_key", "tgt_share", "bayes_tgt_share", "rules_tgt_share"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"full football universe missing target audit columns: {sorted(missing)}")
    if frame.duplicated(["team", "player_clean_key"]).any():
        raise RuntimeError("full football universe has duplicate player/team rows")
    if frame["team"].nunique() != 32:
        raise RuntimeError(f"target audit expected 32 teams, found {frame['team'].nunique()}")

    x = frame.copy()
    for c in ("tgt_share", "bayes_tgt_share", "rules_tgt_share"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    out = x.groupby("team", as_index=False).agg(
        playerform_sum=("tgt_share", lambda s: float(s.fillna(0.0).sum())),
        playerform_players_with_share=("tgt_share", lambda s: int(s.notna().sum())),
        bayes_sum=("bayes_tgt_share", lambda s: float(s.fillna(0.0).sum())),
        bayes_players_with_share=("bayes_tgt_share", lambda s: int(s.notna().sum())),
        rules_sum=("rules_tgt_share", lambda s: float(s.fillna(0.0).sum())),
        rules_players_with_share=("rules_tgt_share", lambda s: int(s.notna().sum())),
        roster_players=("player_clean_key", "nunique"),
    )
    for col in ("playerform_sum", "bayes_sum", "rules_sum"):
        vals = pd.to_numeric(out[col], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals).all() or vals.lt(0).any():
            raise RuntimeError(f"invalid full-universe target sums in {col}")

    out["playerform_pool_exceeds_one"] = out["playerform_sum"].gt(1.0 + 1e-9).astype(int)
    out["bayes_pool_exceeds_one"] = out["bayes_sum"].gt(1.0 + 1e-9).astype(int)
    out["rules_pool_exceeds_one"] = out["rules_sum"].gt(1.0 + 1e-9).astype(int)
    out["simulator_uniform_scale_if_capped_095"] = np.where(
        out["rules_sum"] > 0.95, 0.95 / out["rules_sum"], 1.0
    )
    out = out.sort_values("rules_sum", ascending=False).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    bad = out.loc[out["rules_pool_exceeds_one"].eq(1)]
    payload = {
        "disposition": "TARGET_ENTITLEMENT_POOL_INVALID_RESEARCH_REPAIR_REQUIRED" if not bad.empty else "TARGET_ENTITLEMENT_POOL_VALID",
        "source": "CERTIFIED_FULL_FOOTBALL_SIMULATION_UNIVERSE",
        "football_player_rows": int(len(frame)),
        "teams": int(len(out)),
        "teams_rules_over_1": int(len(bad)),
        "teams_bayes_over_1": int(out["bayes_pool_exceeds_one"].sum()),
        "teams_playerform_over_1": int(out["playerform_pool_exceeds_one"].sum()),
        "rules_sum_min": float(out["rules_sum"].min()),
        "rules_sum_median": float(out["rules_sum"].median()),
        "rules_sum_mean": float(out["rules_sum"].mean()),
        "rules_sum_max": float(out["rules_sum"].max()),
        "minimum_uniform_scale_applied_by_simulator": float(out["simulator_uniform_scale_if_capped_095"].min()),
        "sportsbook_inputs_used": False,
        "normalization_applied_by_audit": False,
        "provider_event_identity_used_for_pool": False,
        "audit": str(OUT_CSV),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[team_target_pool_full_universe] " + json.dumps(payload, sort_keys=True))
    if not bad.empty:
        raise SystemExit(
            f"Full-roster receiving entitlement pool is physically invalid for {len(bad)}/32 teams; "
            f"see {OUT_CSV}. No automatic normalization was applied."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
