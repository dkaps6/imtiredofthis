#!/usr/bin/env python3
"""Audit receiving target entitlement on the sportsbook-independent full roster.

The football_simulation_universe retains raw PlayerForm/Bayes/rules target-share
inputs for provenance.  Once TEAM_TARGET_ENTITLEMENT_V1 is materialized, those
raw pre-entitlement sums are diagnostics rather than the physical pool consumed
by simulation.  This validator therefore certifies the explicit entitlement
trace when present and projection-neutral, while preserving the raw over-
entitlement telemetry.  On legacy runs without an explicit trace, the original
raw-pool fail-closed behavior remains in force.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
UNIVERSE = DATA / "football_simulation_universe.csv"
UNIVERSE_AUDIT = DATA / "football_simulation_universe_audit.json"
ENTITLEMENT_TRACE = DATA / "target_entitlement_v1_trace.csv"
ENTITLEMENT_AUDIT = DATA / "target_entitlement_v1_audit.json"
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
    out["legacy_uniform_scale_if_capped_095"] = np.where(
        out["rules_sum"] > 0.95, 0.95 / out["rules_sum"], 1.0
    )

    explicit = bool(source.get("explicit_target_entitlement_materialized", False))
    explicit_payload: dict[str, object] = {}
    explicit_bad = pd.DataFrame()
    if explicit:
        trace = _read(ENTITLEMENT_TRACE)
        if not ENTITLEMENT_AUDIT.exists() or ENTITLEMENT_AUDIT.stat().st_size <= 0:
            raise RuntimeError("explicit entitlement is declared but its audit JSON is missing")
        ent = json.loads(ENTITLEMENT_AUDIT.read_text(encoding="utf-8"))
        if ent.get("disposition") != "EXPLICIT_TARGET_ENTITLEMENT_MATERIALIZED":
            raise RuntimeError(f"explicit entitlement audit not certified: {ent.get('disposition')}")
        if ent.get("sportsbook_inputs_used") is not False:
            raise RuntimeError("explicit entitlement does not certify sportsbook-independent construction")
        if ent.get("projection_neutral_gate") != "PASS":
            raise RuntimeError("explicit entitlement failed projection-neutral gate")
        if int(ent.get("projection_invariance_changed_arrays", -1)) != 0:
            raise RuntimeError("explicit entitlement changed legacy projection arrays")
        if float(ent.get("projection_invariance_max_mean_gap", np.inf)) > 1e-12:
            raise RuntimeError("explicit entitlement changed legacy projection means")

        trace_required = {
            "team", "player_clean_key", "entitlement_tgt_share", "modeled_player_sum",
            "residual_share", "team_scale", "entitlement_version",
        }
        trace_missing = trace_required - set(trace.columns)
        if trace_missing:
            raise RuntimeError(f"explicit entitlement trace missing columns: {sorted(trace_missing)}")
        if trace.duplicated(["team", "player_clean_key"]).any():
            raise RuntimeError("explicit entitlement trace has duplicate player/team rows")
        if len(trace) != len(frame):
            raise RuntimeError(f"explicit entitlement trace/player universe mismatch: {len(trace)} != {len(frame)}")
        if trace["team"].nunique() != 32:
            raise RuntimeError(f"explicit entitlement expected 32 teams, found {trace['team'].nunique()}")
        if set(zip(trace["team"], trace["player_clean_key"])) != set(zip(frame["team"], frame["player_clean_key"])):
            raise RuntimeError("explicit entitlement player/team keys do not exactly match football universe")

        for c in ("entitlement_tgt_share", "modeled_player_sum", "residual_share", "team_scale"):
            trace[c] = pd.to_numeric(trace[c], errors="coerce")
            if trace[c].isna().any() or not np.isfinite(trace[c]).all():
                raise RuntimeError(f"invalid explicit entitlement values in {c}")
        if trace["entitlement_tgt_share"].lt(-1e-12).any() or trace["team_scale"].le(0).any():
            raise RuntimeError("explicit entitlement contains negative share or non-positive scale")

        physical = trace.groupby("team", as_index=False).agg(
            entitlement_sum=("entitlement_tgt_share", "sum"),
            declared_modeled_sum=("modeled_player_sum", "first"),
            residual_share=("residual_share", "first"),
            team_scale=("team_scale", "first"),
            entitlement_players=("player_clean_key", "nunique"),
        )
        physical["physical_sum"] = physical["entitlement_sum"] + physical["residual_share"]
        physical["entitlement_gap"] = (physical["entitlement_sum"] - physical["declared_modeled_sum"]).abs()
        physical["physical_gap"] = (physical["physical_sum"] - 1.0).abs()
        explicit_bad = physical.loc[
            physical["entitlement_gap"].gt(1e-9)
            | physical["physical_gap"].gt(1e-9)
            | physical["entitlement_sum"].gt(1.0 + 1e-9)
            | physical["residual_share"].lt(-1e-12)
        ]
        if len(physical) != 32:
            raise RuntimeError(f"explicit entitlement physical audit expected 32 teams, found {len(physical)}")

        out = out.merge(physical, on="team", how="left", validate="one_to_one")
        explicit_payload = {
            "entitlement_version": str(trace["entitlement_version"].iloc[0]),
            "explicit_modeled_sum_min": float(physical["entitlement_sum"].min()),
            "explicit_modeled_sum_max": float(physical["entitlement_sum"].max()),
            "explicit_residual_min": float(physical["residual_share"].min()),
            "explicit_residual_max": float(physical["residual_share"].max()),
            "explicit_physical_sum_min": float(physical["physical_sum"].min()),
            "explicit_physical_sum_max": float(physical["physical_sum"].max()),
            "explicit_max_physical_gap": float(physical["physical_gap"].max()),
            "projection_invariance_changed_arrays": int(ent.get("projection_invariance_changed_arrays", -1)),
            "projection_invariance_max_mean_gap": float(ent.get("projection_invariance_max_mean_gap", np.inf)),
            "projection_invariance_max_element_gap": float(ent.get("projection_invariance_max_element_gap", np.inf)),
        }

    out = out.sort_values("rules_sum", ascending=False).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    raw_bad = out.loc[out["rules_pool_exceeds_one"].eq(1)]
    if explicit:
        disposition = "EXPLICIT_TARGET_ENTITLEMENT_POOL_VALID" if explicit_bad.empty else "EXPLICIT_TARGET_ENTITLEMENT_POOL_INVALID"
    else:
        disposition = "TARGET_ENTITLEMENT_POOL_INVALID_RESEARCH_REPAIR_REQUIRED" if not raw_bad.empty else "TARGET_ENTITLEMENT_POOL_VALID"

    payload = {
        "disposition": disposition,
        "source": "CERTIFIED_FULL_FOOTBALL_SIMULATION_UNIVERSE",
        "football_player_rows": int(len(frame)),
        "teams": int(len(out)),
        "raw_teams_rules_over_1": int(len(raw_bad)),
        "raw_teams_bayes_over_1": int(out["bayes_pool_exceeds_one"].sum()),
        "raw_teams_playerform_over_1": int(out["playerform_pool_exceeds_one"].sum()),
        "raw_rules_sum_min": float(out["rules_sum"].min()),
        "raw_rules_sum_median": float(out["rules_sum"].median()),
        "raw_rules_sum_mean": float(out["rules_sum"].mean()),
        "raw_rules_sum_max": float(out["rules_sum"].max()),
        "legacy_minimum_uniform_scale_if_capped_095": float(out["legacy_uniform_scale_if_capped_095"].min()),
        "explicit_target_entitlement_materialized": explicit,
        "raw_pre_entitlement_overage_is_provenance_only": explicit,
        "sportsbook_inputs_used": False,
        "normalization_applied_by_audit": False,
        "provider_event_identity_used_for_pool": False,
        "audit": str(OUT_CSV),
        **explicit_payload,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[team_target_pool_full_universe] " + json.dumps(payload, sort_keys=True))

    if explicit and not explicit_bad.empty:
        raise SystemExit(
            f"Explicit full-roster target entitlement is physically invalid for {len(explicit_bad)}/32 teams; see {OUT_CSV}."
        )
    if (not explicit) and not raw_bad.empty:
        raise SystemExit(
            f"Full-roster receiving entitlement pool is physically invalid for {len(raw_bad)}/32 teams; "
            f"see {OUT_CSV}. No automatic normalization was applied."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
