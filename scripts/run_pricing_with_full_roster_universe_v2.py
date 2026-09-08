#!/usr/bin/env python3
"""Full-roster football-first simulation with explicit conserved entitlement.

The wrapper enforces two distinct contracts:
1. the explicit M38 + team-cap entitlement refactor must remain exactly neutral
   to the legacy simulator under a matched seed;
2. after that neutral seam is certified, the frozen TE-R5P specialist may
   redistribute only the already-existing TE-room target mass.

Sportsbook offers remain downstream lookup/pricing inputs and are never used to
construct either the football universe or TE-R5P entitlement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import apply_te_r5p_entitlement
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate
from scripts.simulation_v2 import MARKET_MAP, lookup, simulate as legacy_simulate
from scripts.utils.player_identity_v3 import player_name_key

DATA = Path("data")
ENTITLEMENT_TRACE = DATA / "target_entitlement_v1_trace.csv"
ENTITLEMENT_AUDIT = DATA / "target_entitlement_v1_audit.json"
ENTITLEMENT_INVARIANCE = DATA / "target_entitlement_v1_projection_invariance.csv"
TE_R5P_TRACE = DATA / "te_r5p_full_slate_entitlement_trace.csv"
TE_R5P_AUDIT = DATA / "te_r5p_full_slate_entitlement_audit.json"
TE_R5P_SIM_DELTA = DATA / "te_r5p_full_slate_simulation_delta.csv"
_ORIGINAL_BUILD = base._build_full_universe


def _suffix_safe_key(value) -> str:
    try:
        return str(player_name_key(value, strip_suffix=True) or "").strip()
    except Exception:
        return ""


def _canonical_identity_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["team"] = out["team"].map(base.canon_team)
    if "player" not in out.columns:
        raise RuntimeError("full-roster identity frame requires player display name for suffix-safe identity")
    out["player_clean_key"] = out["player"].map(_suffix_safe_key)
    blank = out["player_clean_key"].astype("string").fillna("").str.strip().eq("")
    if blank.any():
        sample = out.loc[blank, [c for c in ("player", "team") if c in out.columns]].head(20).to_dict("records")
        raise RuntimeError(f"suffix-safe full-roster identity key unresolved: {sample}")

    names = (
        out.assign(_display=out["player"].astype("string").fillna("").str.strip())
        .groupby(["team", "player_clean_key"], dropna=False)["_display"]
        .nunique(dropna=False)
    )
    bad_keys = names.loc[names.gt(1)]
    if not bad_keys.empty:
        bad_index = set(bad_keys.index.tolist())
        mask = [(str(t), str(k)) in bad_index for t, k in zip(out["team"], out["player_clean_key"])]
        sample = out.loc[mask, ["player", "team", "player_clean_key"]].drop_duplicates().sort_values(
            ["team", "player_clean_key", "player"], kind="mergesort"
        ).head(20).to_dict("records")
        raise RuntimeError(f"suffix-safe full-roster identity is ambiguous within current team: {sample}")
    return out


def _install_provider_player_aliases_and_validate(result, metrics: pd.DataFrame) -> tuple[int, list[dict]]:
    frame = metrics.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    need = {"event_id", "player", "team", "opponent", "season", "week", "market"}
    missing_cols = need - set(frame.columns)
    if missing_cols:
        raise RuntimeError(f"provider distribution aliasing missing columns: {sorted(missing_cols)}")

    unique = frame.sort_values(["event_id", "team", "player", "market"]).drop_duplicates(
        ["event_id", "team", "player", "market"], keep="last"
    )
    missing: list[dict] = []
    aliases = 0
    for _, row in unique.iterrows():
        provider_game = str(row.get("event_id") or "").strip()
        canonical_game = base._canonical_game(
            row.get("team"), row.get("opponent"), row.get("season"), row.get("week")
        )
        canonical_player = _suffix_safe_key(row.get("player"))
        provider_player = base._player_key(row)
        market = MARKET_MAP.get(str(row.get("market", "")).lower(), str(row.get("market", "")).lower())
        if not canonical_player or not provider_player:
            missing.append({
                "event_id": provider_game, "player": row.get("player"), "team": row.get("team"),
                "market": row.get("market"), "reason": "blank_player_lookup_key",
            })
            continue
        values = result.values.get((canonical_game, canonical_player, market))
        if values is None:
            values = result.values.get((provider_game, canonical_player, market))
        if values is None or len(values) == 0:
            missing.append({
                "event_id": provider_game, "player": row.get("player"), "team": row.get("team"),
                "market": row.get("market"), "reason": "canonical_distribution_missing",
                "canonical_player_key": canonical_player,
            })
            continue
        for game in {canonical_game, provider_game}:
            key = (game, provider_player, market)
            if key not in result.values:
                result.values[key] = values
                aliases += 1

    if missing:
        return int(len(unique)), missing
    verify_missing: list[dict] = []
    for _, row in unique.iterrows():
        values = lookup(result, row, str(row.get("market", "")))
        if values is None or len(values) == 0:
            verify_missing.append({
                "event_id": row.get("event_id"), "player": row.get("player"),
                "team": row.get("team"), "market": row.get("market"),
                "reason": "provider_lookup_after_alias_failed",
            })
    if verify_missing:
        return int(len(unique)), verify_missing
    print(
        f"[football_simulation_universe] provider_player_lookup_aliases={aliases} "
        f"validated_provider_player_market_keys={len(unique)}"
    )
    return int(len(unique)), []


def _build_with_explicit_entitlement(pricing_metrics: pd.DataFrame):
    universe, aliases, audit = _ORIGINAL_BUILD(pricing_metrics)
    baseline, baseline_trace = materialize_target_entitlement(universe)
    explicit, te_trace, te_audit = apply_te_r5p_entitlement(baseline)

    ENTITLEMENT_TRACE.parent.mkdir(parents=True, exist_ok=True)
    final_by_key = explicit.set_index(["event_id", "team", "player_clean_key"])
    trace = baseline_trace.copy()
    keys = pd.MultiIndex.from_frame(trace[["event_id", "team", "player_clean_key"]])
    trace["baseline_entitlement_tgt_share"] = trace["entitlement_tgt_share"].astype(float)
    trace["entitlement_tgt_share"] = final_by_key.loc[keys, "entitlement_tgt_share"].to_numpy(float)
    trace["te_r5p_applied"] = final_by_key.loc[keys, "te_r5p_applied"].to_numpy(bool)
    trace["te_r5p_model_version"] = final_by_key.loc[keys, "te_r5p_model_version"].astype(str).to_numpy()
    trace["entitlement_version"] = "TEAM_TARGET_ENTITLEMENT_V1_PLUS_TE_R5P_V1"
    trace.to_csv(ENTITLEMENT_TRACE, index=False)
    te_trace.to_csv(TE_R5P_TRACE, index=False)
    TE_R5P_AUDIT.write_text(json.dumps(te_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    team = trace.drop_duplicates(["event_id", "team"])
    payload = {
        "disposition": "EXPLICIT_TARGET_ENTITLEMENT_MATERIALIZED",
        "version": "TEAM_TARGET_ENTITLEMENT_V1_PLUS_TE_R5P_V1",
        "baseline_refactor_version": "TEAM_TARGET_ENTITLEMENT_V1_PROJECTION_NEUTRAL",
        "football_players": int(len(explicit)),
        "teams": int(team["team"].nunique()),
        "games": int(team["event_id"].nunique()),
        "raw_team_sum_min": float(team["raw_team_sum"].min()),
        "raw_team_sum_median": float(team["raw_team_sum"].median()),
        "raw_team_sum_max": float(team["raw_team_sum"].max()),
        "explicit_modeled_sum_min": float(team["modeled_player_sum"].min()),
        "explicit_modeled_sum_max": float(team["modeled_player_sum"].max()),
        "residual_min": float(team["residual_share"].min()),
        "residual_max": float(team["residual_share"].max()),
        "m38_applied_before_entitlement": True,
        "production_specialists": ["TE_R5P_PRODUCTION_MODEL_V1"],
        "te_r5p_entitlement_audit": str(TE_R5P_AUDIT),
        "te_r5p_entitlement_trace": str(TE_R5P_TRACE),
        "te_r5p_team_pool_preserved": bool(te_audit["team_te_pool_preserved"]),
        "te_r5p_non_te_entitlement_preserved": bool(te_audit["non_te_entitlement_preserved"]),
        "sportsbook_inputs_used": False,
        "new_scientific_parameters_introduced": True,
        "trace": str(ENTITLEMENT_TRACE),
    }
    ENTITLEMENT_AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit.update({
        "explicit_target_entitlement_version": payload["version"],
        "explicit_target_entitlement_materialized": True,
        "te_r5p_full_slate_consumed": True,
        "te_r5p_model_version": te_audit["model_version"],
    })
    return explicit, aliases, audit


def _projection_neutral_simulate(metrics: pd.DataFrame, *, iterations=None, seed=None, allocation_trace=None):
    if "baseline_entitlement_tgt_share" not in metrics.columns:
        raise RuntimeError("TE-R5P Full Slate simulation missing projection-neutral baseline entitlement")

    # First certify the architecture refactor itself against the legacy simulator.
    legacy_input = metrics.drop(
        columns=[
            c for c in metrics.columns
            if c.startswith("entitlement_") or c.startswith("baseline_entitlement_") or c.startswith("te_r5p_")
        ],
        errors="ignore",
    )
    baseline_metrics = metrics.copy()
    baseline_metrics["entitlement_tgt_share"] = pd.to_numeric(
        baseline_metrics["baseline_entitlement_tgt_share"], errors="raise"
    ).astype(float)
    legacy = legacy_simulate(legacy_input, iterations=iterations, seed=seed)
    baseline_explicit = explicit_simulate(baseline_metrics, iterations=iterations, seed=seed)

    legacy_keys = set(legacy.values)
    baseline_keys = set(baseline_explicit.values)
    if legacy_keys != baseline_keys:
        raise RuntimeError(
            "projection-neutral entitlement baseline changed simulation key universe; "
            f"missing={list(legacy_keys-baseline_keys)[:20]} extra={list(baseline_keys-legacy_keys)[:20]}"
        )

    rows = []
    max_mean_gap = 0.0
    max_element_gap = 0.0
    changed_arrays = 0
    for key in sorted(legacy_keys):
        a = np.asarray(legacy.values[key], dtype=float)
        b = np.asarray(baseline_explicit.values[key], dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f"projection-neutral baseline produced invalid arrays key={key}")
        mean_gap = abs(float(a.mean()) - float(b.mean()))
        element_gap = float(np.max(np.abs(a-b))) if len(a) else 0.0
        changed_arrays += int(element_gap > 0)
        max_mean_gap = max(max_mean_gap, mean_gap)
        max_element_gap = max(max_element_gap, element_gap)
        rows.append({
            "event_id": key[0], "player_clean_key": key[1], "market": key[2],
            "legacy_mean": float(a.mean()) if len(a) else np.nan,
            "baseline_explicit_mean": float(b.mean()) if len(b) else np.nan,
            "mean_gap": mean_gap, "max_element_gap": element_gap,
        })
    pd.DataFrame(rows).to_csv(ENTITLEMENT_INVARIANCE, index=False)
    if changed_arrays != 0 or max_mean_gap > 1e-12 or max_element_gap > 1e-12:
        raise RuntimeError(
            "explicit target entitlement baseline is not exactly projection-neutral: "
            f"changed_arrays={changed_arrays} max_mean_gap={max_mean_gap} max_element_gap={max_element_gap}"
        )

    # Then consume TE-R5P as an intentional football-only specialist change.
    final = explicit_simulate(metrics, iterations=iterations, seed=seed, allocation_trace=allocation_trace)
    final_keys = set(final.values)
    if final_keys != baseline_keys:
        raise RuntimeError(
            "TE-R5P changed simulation key universe; "
            f"missing={list(baseline_keys-final_keys)[:20]} extra={list(final_keys-baseline_keys)[:20]}"
        )

    pos_map = {
        (str(r.event_id), str(r.player_clean_key)): str(r.position)
        for r in metrics[["event_id", "player_clean_key", "position"]].drop_duplicates().itertuples(index=False)
    }
    delta_rows = []
    for key in sorted(final_keys):
        a = np.asarray(baseline_explicit.values[key], dtype=float)
        b = np.asarray(final.values[key], dtype=float)
        if a.shape != b.shape or not np.isfinite(b).all():
            raise RuntimeError(f"TE-R5P produced invalid simulation arrays key={key}")
        delta_rows.append({
            "event_id": key[0],
            "player_clean_key": key[1],
            "position": pos_map.get((str(key[0]), str(key[1])), ""),
            "market": key[2],
            "baseline_mean": float(a.mean()) if len(a) else np.nan,
            "te_r5p_mean": float(b.mean()) if len(b) else np.nan,
            "mean_delta": float(b.mean() - a.mean()) if len(a) else np.nan,
            "abs_mean_delta": abs(float(b.mean() - a.mean())) if len(a) else np.nan,
        })
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(TE_R5P_SIM_DELTA, index=False)

    status = json.loads(ENTITLEMENT_AUDIT.read_text(encoding="utf-8"))
    status.update({
        "projection_invariance_scope": "legacy_vs_explicit_baseline_before_position_specialists",
        "projection_invariance_keys": int(len(rows)),
        "projection_invariance_changed_arrays": int(changed_arrays),
        "projection_invariance_max_mean_gap": max_mean_gap,
        "projection_invariance_max_element_gap": max_element_gap,
        "projection_neutral_gate": "PASS",
        "invariance_audit": str(ENTITLEMENT_INVARIANCE),
        "te_r5p_simulation_delta_audit": str(TE_R5P_SIM_DELTA),
        "te_r5p_changed_distribution_keys": int((delta["abs_mean_delta"].fillna(0).gt(1e-12)).sum()),
        "te_r5p_max_abs_mean_delta": float(delta["abs_mean_delta"].max()) if len(delta) else 0.0,
        "te_r5p_te_max_abs_mean_delta": float(delta.loc[delta["position"].eq("TE"), "abs_mean_delta"].max()) if delta["position"].eq("TE").any() else 0.0,
    })
    ENTITLEMENT_AUDIT.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[target_entitlement_v1_plus_te_r5p] " + json.dumps(status, sort_keys=True))
    return final


def main() -> int:
    base._identity_frame = _canonical_identity_frame
    base._validate_priced_distribution_coverage = _install_provider_player_aliases_and_validate
    base._build_full_universe = _build_with_explicit_entitlement
    base.canonical_simulate = _projection_neutral_simulate
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
