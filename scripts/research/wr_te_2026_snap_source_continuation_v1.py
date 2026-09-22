#!/usr/bin/env python3
"""WR/TE 2026 Snap Source Continuation V1.

Research-only replay. Keeps the frozen WR-R15 / TE-R5P learned models unchanged
and compares their current production snap source (2020-2025) with a source-only
continuation through 2026.
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.modeling.te_r5p_entitlement_adapter_v1 as te
import scripts.modeling.wr_r15_entitlement_adapter_v1 as wr
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate

BASELINE_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
CANDIDATE_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
W2_SOURCE_RUN = 35282021679
W2_SOURCE_ARTIFACT = 10523345092
W2_SOURCE_DIGEST = "sha256:6024ed21d7032e6e6440145277d9d4b74e98f319ef286962bac2a786a8d28e3c"


def _load_prod_snaps(seasons: list[int]) -> tuple[pd.DataFrame, float, list[int]]:
    original = list(te.SOURCE_SEASONS)
    try:
        te.SOURCE_SEASONS[:] = list(seasons)
        q, dup_rate, loaded = te._load_snaps()
        return q.copy(), float(dup_rate), [int(v) for v in loaded]
    finally:
        te.SOURCE_SEASONS[:] = original


@contextmanager
def _snap_override(payload: tuple[pd.DataFrame, float, list[int]]):
    q, dup_rate, seasons = payload
    original_te = te._load_snaps
    original_wr = wr._load_snaps

    def loader():
        return q.copy(), float(dup_rate), list(seasons)

    te._load_snaps = loader
    wr._load_snaps = loader
    try:
        yield
    finally:
        te._load_snaps = original_te
        wr._load_snaps = original_wr


def _prepare_preserved_week2(source_dir: Path) -> pd.DataFrame:
    path = source_dir / "data" / "football_simulation_universe.csv"
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"preserved Week-2 universe missing: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "event_id", "season", "week", "team", "player", "player_clean_key",
        "position", "entitlement_tgt_share", "entitlement_residual_share",
        "baseline_entitlement_tgt_share",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"preserved Week-2 universe missing required columns: {sorted(missing)}")
    season = pd.to_numeric(x["season"], errors="coerce")
    week = pd.to_numeric(x["week"], errors="coerce")
    if not season.eq(2026).all() or not week.eq(2).all():
        bad = x.loc[~(season.eq(2026) & week.eq(2)), ["season", "week"]].head(20).to_dict("records")
        raise RuntimeError(f"preserved universe is not exact 2026 Week 2: {bad}")
    if x.duplicated(["event_id", "team", "player_clean_key"]).any():
        raise RuntimeError("preserved Week-2 universe has duplicate player/game/team rows")

    baseline = pd.to_numeric(x["baseline_entitlement_tgt_share"], errors="coerce")
    if baseline.isna().any() or (baseline < 0).any() or not np.isfinite(baseline.to_numpy(float)).all():
        raise RuntimeError("preserved pre-specialist entitlement is invalid")
    x["entitlement_tgt_share"] = baseline.astype(float)
    return x


def _apply_stack(
    pre_specialist: pd.DataFrame,
    snaps: tuple[pd.DataFrame, float, list[int]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    with _snap_override(snaps):
        te_out, te_trace, te_audit = te.apply_te_r5p_entitlement(pre_specialist.copy())
        te_out["te_only_entitlement_tgt_share"] = pd.to_numeric(
            te_out["entitlement_tgt_share"], errors="raise"
        ).astype(float)
        final, wr_trace, wr_audit = wr.apply_wr_r15_entitlement(te_out)
    return final, te_trace, wr_trace, te_audit, wr_audit


def _week1_semantic_invariance(
    baseline_snaps: tuple[pd.DataFrame, float, list[int]],
    candidate_snaps: tuple[pd.DataFrame, float, list[int]],
) -> dict:
    # Frozen synthetic entitlement frame plus a controlled snap fixture tests the
    # exact production strict-prior boundary. Real-source readiness is reported
    # separately from candidate_snaps below.
    fixture = pd.DataFrame([
        {
            "event_id": "2026_01_TEST", "season": 2026, "week": 1, "team": "AAA",
            "player": "Tight End", "player_clean_key": "tightend", "position": "TE",
            "entitlement_tgt_share": 0.18, "entitlement_residual_share": 0.20,
        },
        {
            "event_id": "2026_01_TEST", "season": 2026, "week": 1, "team": "AAA",
            "player": "Wide One", "player_clean_key": "wideone", "position": "WR",
            "entitlement_tgt_share": 0.30, "entitlement_residual_share": 0.20,
        },
        {
            "event_id": "2026_01_TEST", "season": 2026, "week": 1, "team": "AAA",
            "player": "Wide Two", "player_clean_key": "widetwo", "position": "WR",
            "entitlement_tgt_share": 0.18, "entitlement_residual_share": 0.20,
        },
        {
            "event_id": "2026_01_TEST", "season": 2026, "week": 1, "team": "AAA",
            "player": "Runner", "player_clean_key": "runner", "position": "RB",
            "entitlement_tgt_share": 0.14, "entitlement_residual_share": 0.20,
        },
    ])
    hist = pd.DataFrame([
        {"season": 2025, "week": 18, "team": "AAA", "player_key": "tightend", "offense_pct": .75, "offense_snaps": 51, "ordinal": 202518},
        {"season": 2025, "week": 18, "team": "AAA", "player_key": "wideone", "offense_pct": .91, "offense_snaps": 62, "ordinal": 202518},
        {"season": 2025, "week": 18, "team": "AAA", "player_key": "widetwo", "offense_pct": .63, "offense_snaps": 43, "ordinal": 202518},
    ])
    future = pd.DataFrame([
        {"season": 2026, "week": 1, "team": "AAA", "player_key": "tightend", "offense_pct": .99, "offense_snaps": 70, "ordinal": 202601},
        {"season": 2026, "week": 1, "team": "AAA", "player_key": "wideone", "offense_pct": .99, "offense_snaps": 70, "ordinal": 202601},
        {"season": 2026, "week": 1, "team": "AAA", "player_key": "widetwo", "offense_pct": .99, "offense_snaps": 70, "ordinal": 202601},
    ])
    base_payload = (hist.copy(), 0.0, BASELINE_SEASONS)
    cand_payload = (pd.concat([hist, future], ignore_index=True), 0.0, CANDIDATE_SEASONS)
    b, bt, bw, _, _ = _apply_stack(fixture, base_payload)
    c, ct, cw, _, _ = _apply_stack(fixture, cand_payload)

    key = ["event_id", "team", "player_clean_key"]
    bj = b.set_index(key)["entitlement_tgt_share"].sort_index()
    cj = c.set_index(key)["entitlement_tgt_share"].sort_index()
    max_gap = float((bj - cj).abs().max())
    if max_gap > 1e-12:
        raise RuntimeError(f"Week-1 source continuation is not invariant: max_gap={max_gap}")

    for trace_a, trace_b, label in ((bt, ct, "TE"), (bw, cw, "WR")):
        shared = [col for col in trace_a.columns if col in trace_b.columns]
        # Compare numeric strict-prior participation columns; target-week poison
        # rows must not affect any of them.
        cols = [c for c in shared if "prior" in c and pd.api.types.is_numeric_dtype(trace_a[c])]
        for col in cols:
            aa = pd.to_numeric(trace_a[col], errors="coerce").fillna(-999999.0).to_numpy(float)
            bb = pd.to_numeric(trace_b[col], errors="coerce").fillna(-999999.0).to_numpy(float)
            if not np.array_equal(aa, bb):
                raise RuntimeError(f"Week-1 strict-prior feature changed label={label} column={col}")

    real_candidate = candidate_snaps[0]
    real_2026 = real_candidate.loc[pd.to_numeric(real_candidate["season"], errors="coerce").eq(2026)].copy()
    weeks = sorted(int(v) for v in pd.to_numeric(real_2026["week"], errors="coerce").dropna().unique())
    return {
        "week1_adapter_invariance_pass": True,
        "max_entitlement_gap": max_gap,
        "real_2026_snap_weeks_loaded": weeks,
        "real_2026_snap_rows_loaded": int(len(real_2026)),
        "note": "real source readiness + controlled target-week poison invariance",
    }


def _strict_prior_candidate_payload(
    candidate_snaps: tuple[pd.DataFrame, float, list[int]],
    target_week: int,
) -> tuple[pd.DataFrame, float, list[int]]:
    q, dup, seasons = candidate_snaps
    ordinal = pd.to_numeric(q["ordinal"], errors="coerce")
    safe = q.loc[ordinal.lt(202600 + int(target_week))].copy()
    if len(safe) and float(pd.to_numeric(safe["ordinal"], errors="coerce").max()) >= 202600 + int(target_week):
        raise RuntimeError("strict-prior candidate payload contains target/future snap row")
    return safe, dup, seasons


def _projection_delta(
    baseline_final: pd.DataFrame,
    candidate_final: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    base_sim = explicit_simulate(baseline_final, iterations=iterations, seed=seed)
    cand_sim = explicit_simulate(candidate_final, iterations=iterations, seed=seed)
    if set(base_sim.values) != set(cand_sim.values):
        raise RuntimeError("candidate changed simulation key universe")
    pos = (
        baseline_final[["event_id", "player_clean_key", "player", "team", "position"]]
        .drop_duplicates(["event_id", "player_clean_key"])
        .set_index(["event_id", "player_clean_key"])
    )
    rows = []
    for key in sorted(base_sim.values):
        event_id, player_key, market = key
        if str(market) not in {"rec_yards", "receptions"}:
            continue
        meta = pos.loc[(event_id, player_key)] if (event_id, player_key) in pos.index else None
        if meta is None or str(meta["position"]).upper() not in {"WR", "TE"}:
            continue
        a = np.asarray(base_sim.values[key], dtype=float)
        b = np.asarray(cand_sim.values[key], dtype=float)
        if a.shape != b.shape:
            raise RuntimeError(f"simulation shape changed for {key}")
        rows.append({
            "event_id": event_id,
            "player_clean_key": player_key,
            "player": str(meta["player"]),
            "team": str(meta["team"]),
            "position": str(meta["position"]),
            "market": str(market),
            "baseline_mean": float(a.mean()),
            "candidate_mean": float(b.mean()),
            "mean_delta": float(b.mean() - a.mean()),
            "abs_mean_delta": abs(float(b.mean() - a.mean())),
            "max_element_gap": float(np.max(np.abs(a - b))),
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--week2-source-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260922)
    a = p.parse_args()

    baseline_snaps = _load_prod_snaps(BASELINE_SEASONS)
    candidate_snaps = _load_prod_snaps(CANDIDATE_SEASONS)
    real_2026 = candidate_snaps[0].loc[
        pd.to_numeric(candidate_snaps[0]["season"], errors="coerce").eq(2026)
    ].copy()
    weeks = sorted(int(v) for v in pd.to_numeric(real_2026["week"], errors="coerce").dropna().unique())
    if 1 not in weeks or 2 not in weeks:
        raise RuntimeError(f"2026 snap continuation requires Weeks 1-2, got {weeks}")

    week1 = _week1_semantic_invariance(baseline_snaps, candidate_snaps)

    source = _prepare_preserved_week2(a.week2_source_dir)
    candidate_safe = _strict_prior_candidate_payload(candidate_snaps, 2)

    baseline_final, baseline_te, baseline_wr, base_te_audit, base_wr_audit = _apply_stack(
        source.copy(), baseline_snaps
    )
    candidate_final, candidate_te, candidate_wr, cand_te_audit, cand_wr_audit = _apply_stack(
        source.copy(), candidate_snaps
    )
    safe_final, safe_te, safe_wr, _, _ = _apply_stack(source.copy(), candidate_safe)

    key = ["event_id", "team", "player_clean_key"]
    full = candidate_final.set_index(key)["entitlement_tgt_share"].sort_index()
    safe = safe_final.set_index(key)["entitlement_tgt_share"].sort_index()
    strict_prior_gap = float((full - safe).abs().max())
    if strict_prior_gap > 1e-12:
        raise RuntimeError(
            f"candidate output changed when target/future 2026 snap rows removed: {strict_prior_gap}"
        )

    b = baseline_final.set_index(key)["entitlement_tgt_share"].sort_index()
    c = candidate_final.set_index(key)["entitlement_tgt_share"].sort_index()
    ent = baseline_final[key + ["player", "position"]].copy().set_index(key)
    ent["baseline_entitlement"] = b
    ent["candidate_entitlement"] = c
    ent["entitlement_delta"] = c - b
    ent["abs_entitlement_delta"] = ent["entitlement_delta"].abs()
    ent = ent.reset_index()

    projection = _projection_delta(
        baseline_final, candidate_final, iterations=int(a.iterations), seed=int(a.seed)
    )

    def trace_2026_effect(base: pd.DataFrame, cand: pd.DataFrame, label: str) -> dict:
        join = ["event_id", "team", "player_clean_key"]
        cols = [c for c in cand.columns if "prior" in c and c in base.columns]
        merged = base[join + cols].merge(
            cand[join + cols], on=join, suffixes=("_base", "_cand"), validate="one_to_one"
        )
        changed = pd.Series(False, index=merged.index)
        for col in cols:
            aa = pd.to_numeric(merged[f"{col}_base"], errors="coerce")
            bb = pd.to_numeric(merged[f"{col}_cand"], errors="coerce")
            changed |= ~(aa.fillna(-999999.0).eq(bb.fillna(-999999.0)))
        return {
            "label": label,
            "rows": int(len(merged)),
            "rows_with_any_prior_feature_change": int(changed.sum()),
        }

    te_change = trace_2026_effect(baseline_te, candidate_te, "TE")
    wr_change = trace_2026_effect(baseline_wr, candidate_wr, "WR")

    pos_summary = []
    for pos, q in ent.loc[ent["position"].astype(str).str.upper().isin(["WR", "TE"])].groupby(
        ent["position"].astype(str).str.upper()
    ):
        pos_summary.append({
            "position": str(pos),
            "rows": int(len(q)),
            "changed_rows": int(q["abs_entitlement_delta"].gt(1e-12).sum()),
            "mean_abs_entitlement_delta": float(q["abs_entitlement_delta"].mean()),
            "max_abs_entitlement_delta": float(q["abs_entitlement_delta"].max()),
        })

    market_summary = []
    for (pos, market), q in projection.groupby(["position", "market"], sort=True):
        market_summary.append({
            "position": str(pos),
            "market": str(market),
            "rows": int(len(q)),
            "changed_rows": int(q["abs_mean_delta"].gt(1e-9).sum()),
            "mean_abs_projection_delta": float(q["abs_mean_delta"].mean()),
            "max_abs_projection_delta": float(q["abs_mean_delta"].max()),
        })

    changed_rows = int(ent["abs_entitlement_delta"].gt(1e-12).sum())
    disposition = (
        "SNAP_CONTINUATION_MECHANICALLY_VALID_AND_MATERIAL"
        if changed_rows > 0 and float(projection["abs_mean_delta"].max()) >= 0.25
        else "SNAP_CONTINUATION_MECHANICALLY_VALID_SMALL_EFFECT"
        if changed_rows > 0
        else "SNAP_CONTINUATION_NO_EFFECT"
    )

    result = {
        "version": "WR_TE_2026_SNAP_SOURCE_CONTINUATION_V1",
        "plan": "WR_TE_2026_SNAP_SOURCE_CONTINUATION_V1_PLAN_FROZEN",
        "week2_source_run": W2_SOURCE_RUN,
        "week2_source_artifact": W2_SOURCE_ARTIFACT,
        "week2_source_digest": W2_SOURCE_DIGEST,
        "baseline_source_seasons": BASELINE_SEASONS,
        "candidate_source_seasons": CANDIDATE_SEASONS,
        "learned_model_refit": False,
        "sportsbook_inputs_to_football_model": False,
        "production_changed": False,
        "week1_invariance": week1,
        "week2_target_future_snap_exclusion_max_output_gap": strict_prior_gap,
        "week2_te_trace_change": te_change,
        "week2_wr_trace_change": wr_change,
        "entitlement_summary": pos_summary,
        "projection_summary": market_summary,
        "te_conservation": {
            "baseline_team_te_pool_preserved": bool(base_te_audit["team_te_pool_preserved"]),
            "candidate_team_te_pool_preserved": bool(cand_te_audit["team_te_pool_preserved"]),
            "baseline_non_te_preserved": bool(base_te_audit["non_te_entitlement_preserved"]),
            "candidate_non_te_preserved": bool(cand_te_audit["non_te_entitlement_preserved"]),
        },
        "wr_conservation": {
            "baseline_m38_wr1_anchor_preserved": bool(base_wr_audit["m38_wr1_anchor_preserved"]),
            "candidate_m38_wr1_anchor_preserved": bool(cand_wr_audit["m38_wr1_anchor_preserved"]),
            "baseline_wr2plus_pool_preserved": bool(base_wr_audit["wr2plus_pool_preserved"]),
            "candidate_wr2plus_pool_preserved": bool(cand_wr_audit["wr2plus_pool_preserved"]),
            "baseline_non_wr_preserved": bool(base_wr_audit["non_wr_entitlement_preserved"]),
            "candidate_non_wr_preserved": bool(cand_wr_audit["non_wr_entitlement_preserved"]),
        },
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    ent.to_csv(a.out_dir / "week2_entitlement_delta.csv", index=False)
    projection.to_csv(a.out_dir / "week2_projection_delta.csv", index=False)
    candidate_te.to_csv(a.out_dir / "week2_candidate_te_trace.csv", index=False)
    candidate_wr.to_csv(a.out_dir / "week2_candidate_wr_trace.csv", index=False)
    (a.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "disposition": disposition,
        "week1_invariance": week1,
        "strict_prior_gap": strict_prior_gap,
        "te_trace_change": te_change,
        "wr_trace_change": wr_change,
        "entitlement_summary": pos_summary,
        "projection_summary": market_summary,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
