#!/usr/bin/env python3
"""Freeze prospective Week-6 TE trust labels from exact player allocation difficulty.

Research only. No Week-6 outcomes, sportsbook inputs, projection changes, or
production mutation are permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import apply_te_r5p_entitlement
from scripts.utils.player_identity_v3 import (
    attach_historical_identity,
    historical_identity_key,
    player_name_key,
)

VERSION = "TE_PLAYER_DIFFICULTY_TRUST_WEEK6_PREGAME_LOCK_V1"
SEASON = 2026
WEEK = 6
PRIOR_WINDOW = 8
MIN_PRIOR = 4
PARITY_TOL = 1e-5


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(s):
    return pd.to_numeric(s, errors="coerce")


def pkey(v) -> str:
    return player_name_key(v, strip_suffix=True)


def semantic_parity(te_authority: pd.DataFrame, component_2025: pd.DataFrame) -> dict:
    need_te = {"season","week","team","player_clean_key","b0_expected_targets","b0_target_probability"}
    need_cp = {"season","week","team","player_clean_key","position","rules_plays_est","rules_pass_rate"}
    if need_te - set(te_authority.columns):
        raise RuntimeError(f"TE authority parity columns missing: {sorted(need_te-set(te_authority.columns))}")
    if need_cp - set(component_2025.columns):
        raise RuntimeError(f"component parity columns missing: {sorted(need_cp-set(component_2025.columns))}")
    t = te_authority.loc[num(te_authority["season"]).eq(2025)].copy()
    c = component_2025.loc[
        num(component_2025["season"]).eq(2025)
        & component_2025["position"].astype(str).str.upper().eq("TE")
    ].copy()
    keys = ["season","week","team","player_clean_key"]
    c = c[keys + ["rules_plays_est","rules_pass_rate"]].drop_duplicates(keys)
    if c.duplicated(keys).any():
        raise RuntimeError("duplicate 2025 component parity identities")
    z = t.merge(c, on=keys, how="inner", validate="one_to_one")
    if len(z) < 1100:
        raise RuntimeError(f"2025 TE semantic parity support too small: {len(z)}")
    den = num(z["b0_target_probability"])
    if den.le(0).any() or den.isna().any():
        raise RuntimeError("invalid b0 target probability in parity authority")
    authority = num(z["b0_expected_targets"]) / den
    candidate = (
        num(z["rules_plays_est"]).clip(50.0, 80.0)
        * num(z["rules_pass_rate"]).clip(0.35, 0.75)
    )
    gap = (authority - candidate).abs()
    if gap.isna().any() or float(gap.max()) > PARITY_TOL:
        raise RuntimeError(
            f"team-target semantic parity failed rows={len(z)} max_gap={float(gap.max())}"
        )
    return {
        "matched_rows": int(len(z)),
        "max_abs_gap": float(gap.max()),
        "mean_abs_gap": float(gap.mean()),
        "formula": "clip(rules_plays_est,50,80)*clip(rules_pass_rate,.35,.75)",
        "gate_tolerance": PARITY_TOL,
    }


def current_te_cohort(te_trace: pd.DataFrame, volatility: pd.DataFrame) -> pd.DataFrame:
    required = {"season","week","team","player","player_clean_key","te_r5p_entitlement_tgt_share"}
    miss = required - set(te_trace.columns)
    if miss:
        raise RuntimeError(f"Week6 TE trace missing {sorted(miss)}")
    t = te_trace.loc[num(te_trace["season"]).eq(SEASON) & num(te_trace["week"]).eq(WEEK)].copy()
    if t.empty:
        raise RuntimeError("Week6 TE-R5P cohort is empty")
    t["team"] = t["team"].map(canon_team)
    t["name_key"] = t["player"].map(pkey)
    if t.duplicated(["team","name_key"]).any():
        raise RuntimeError("ambiguous Week6 TE trace identity")

    v = volatility.loc[volatility["position"].astype(str).str.upper().eq("TE")].copy()
    v["team"] = v["team"].map(canon_team)
    v["name_key"] = v["player"].map(pkey)
    if v.duplicated(["team","name_key"]).any():
        raise RuntimeError("ambiguous Week6 volatility TE identity")
    t = t.merge(
        v[["team","name_key","player_identity_key","gsis_id"]],
        on=["team","name_key"], how="left", validate="one_to_one",
    )
    if t["player_identity_key"].fillna("").astype(str).str.strip().eq("").any():
        sample = t.loc[
            t["player_identity_key"].fillna("").astype(str).str.strip().eq(""),
            ["team","player","player_clean_key"],
        ].head(20).to_dict("records")
        raise RuntimeError(f"Week6 TE stable identity misses: {sample}")
    if t["player_identity_key"].duplicated().any():
        raise RuntimeError("Week6 TE stable identity duplicates")
    return t.sort_values(["team","player_clean_key"]).reset_index(drop=True)


def reconstruct_2026_errors(
    *,
    cohort: pd.DataFrame,
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    universe_dir: Path,
) -> pd.DataFrame:
    logs = player_logs.copy()
    logs.columns = [str(c).strip().lower() for c in logs.columns]
    if "player_identity_key" not in logs.columns:
        logs = attach_historical_identity(logs)
    if "player_clean_key" not in logs.columns:
        logs["player_clean_key"] = logs["player"].map(pkey)
    logs["season"] = num(logs["season"])
    logs["week"] = num(logs["week"])
    logs["team"] = logs["team"].map(canon_team)

    future = logs.loc[logs["season"].eq(SEASON) & logs["week"].ge(WEEK)]
    if not future.empty:
        bad = sorted(future["week"].dropna().astype(int).unique().tolist())
        raise RuntimeError(f"Week6-or-later outcomes already present; refuse lock: {bad}")

    current_ids = set(cohort["player_identity_key"].astype(str))
    current_name_to_id = {}
    for r in cohort.itertuples(index=False):
        nk = pkey(r.player)
        if nk in current_name_to_id and current_name_to_id[nk] != str(r.player_identity_key):
            raise RuntimeError(f"current TE name collision: {nk}")
        current_name_to_id[nk] = str(r.player_identity_key)

    rows = []
    for week in range(1, WEEK):
        upath = universe_dir / f"{SEASON}_week_{week:02d}.csv"
        universe = read_csv(upath, f"2026 W{week} pregame universe")
        bundle = build_historical_context_bundle(
            player_logs=player_logs,
            team_weekly=team_weekly,
            pregame_universe=universe,
            schedule=schedule,
            season=SEASON,
            week=week,
            prior_season=2025,
            injuries=None,
            weather=None,
        )
        metrics = build_mc_predictions(bundle, iterations=2000, seed=42 + week)
        players = (
            metrics.sort_values(["event_id","team","player_clean_key"])
            .drop_duplicates(["event_id","team","player_clean_key"], keep="last")
            .copy()
        )
        baseline, _ = materialize_target_entitlement(players)
        te_final, te_trace, te_audit = apply_te_r5p_entitlement(baseline)
        if str(te_audit.get("model_version")) != "TE_R5P_PRODUCTION_MODEL_V1":
            raise RuntimeError("TE-R5P model version drift in Week1-5 reconstruction")

        core = te_trace[[
            "event_id","season","week","team","player","player_clean_key",
            "te_r5p_entitlement_tgt_share",
        ]].copy()
        core["team"] = core["team"].map(canon_team)
        core["name_key"] = core["player"].map(pkey)
        core["player_identity_key"] = core["name_key"].map(current_name_to_id)
        core = core.loc[core["player_identity_key"].isin(current_ids)].copy()
        if core.empty:
            continue

        pm = players[[
            "event_id","team","player_clean_key","rules_plays_est","rules_pass_rate"
        ]].drop_duplicates(["event_id","team","player_clean_key"])
        core = core.merge(
            pm,
            on=["event_id","team","player_clean_key"],
            how="left", validate="one_to_one",
        )
        core["pred_team_targets"] = (
            num(core["rules_plays_est"]).clip(50.0, 80.0)
            * num(core["rules_pass_rate"]).clip(0.35, 0.75)
        )
        if core["pred_team_targets"].isna().any() or core["pred_team_targets"].le(0).any():
            raise RuntimeError(f"invalid pregame team target expectation W{week}")
        core["promoted_share"] = num(core["te_r5p_entitlement_tgt_share"])
        core["promoted_targets"] = core["pred_team_targets"] * core["promoted_share"]

        a = logs.loc[logs["season"].eq(SEASON) & logs["week"].eq(week)].copy()
        if a.empty:
            raise RuntimeError(f"missing completed 2026 Week{week} player logs")
        a["targets"] = num(a.get("targets")).fillna(0.0)
        team_den = (
            a.groupby("team", as_index=False)["targets"].sum()
            .rename(columns={"targets":"actual_team_targets"})
        )
        pactual = (
            a.groupby(["team","player_clean_key"], as_index=False)["targets"].sum()
            .rename(columns={"targets":"actual_player_targets"})
        )
        core = core.merge(team_den, on="team", how="left", validate="many_to_one")
        core = core.merge(
            pactual, on=["team","player_clean_key"], how="left", validate="one_to_one"
        )
        core["actual_player_targets"] = num(core["actual_player_targets"]).fillna(0.0)
        core["actual_team_targets"] = num(core["actual_team_targets"])
        if core["actual_team_targets"].isna().any() or core["actual_team_targets"].le(0).any():
            raise RuntimeError(f"invalid actual team target denominator W{week}")
        core["actual_share"] = core["actual_player_targets"] / core["actual_team_targets"]
        core["allocation_actual_targets"] = core["pred_team_targets"] * core["actual_share"]
        core["allocation_error"] = core["promoted_targets"] - core["allocation_actual_targets"]
        core["history_source"] = "RECONSTRUCTED_2026_STRICT_PRIOR_TE_R5P"
        core["source_max_week"] = week - 1
        rows.append(core[[
            "season","week","event_id","team","player","player_clean_key",
            "player_identity_key","pred_team_targets","promoted_targets",
            "promoted_share","actual_team_targets","actual_player_targets",
            "actual_share","allocation_actual_targets","allocation_error",
            "history_source","source_max_week",
        ]])

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        raise RuntimeError("zero reconstructed 2026 TE allocation-error rows")
    if out.duplicated(["season","week","player_identity_key"]).any():
        sample = out.loc[
            out.duplicated(["season","week","player_identity_key"], keep=False),
            ["season","week","team","player","player_identity_key"],
        ].head(20).to_dict("records")
        raise RuntimeError(f"duplicate 2026 TE allocation history: {sample}")
    if int(num(out["week"]).max()) >= WEEK:
        raise RuntimeError("Week6 history leakage")
    return out


def lock_trust(
    *,
    cohort: pd.DataFrame,
    parent_history: pd.DataFrame,
    current_history: pd.DataFrame,
    parity: dict,
    parent_trust: dict,
    opportunity_pack: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if parent_trust.get("disposition") != "TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED":
        raise RuntimeError("trust parent is not qualified")
    if opportunity_pack.get("status") != "WEEK6_INDIVIDUAL_OPPORTUNITY_LOCK_PACK_FROZEN":
        raise RuntimeError("Week6 opportunity pack not frozen")
    if int(opportunity_pack.get("target_week_outcomes_read", 1)) != 0:
        raise RuntimeError("Week6 opportunity pack read Week6 outcomes")
    state = (opportunity_pack.get("components") or {}).get("player_state") or {}
    if state.get("disposition") != "PLAYER_STATE_FUTURE_WEEK_READY":
        raise RuntimeError("Week6 player state is not ready")
    if int(state.get("roster_source_week", -1)) != WEEK:
        raise RuntimeError("Week6 roster source is not exact Week6")

    ph = parent_history.copy()
    ph = ph.loc[
        ph["position"].astype(str).str.upper().eq("TE")
        & num(ph["season"]).isin([2024, 2025])
    ].copy()
    need = {"season","week","player_identity_key","allocation_error"}
    if need - set(ph.columns):
        raise RuntimeError(f"parent allocation history missing {sorted(need-set(ph.columns))}")
    ph = ph[["season","week","player_identity_key","allocation_error"]].copy()
    ph["history_source"] = "EXACT_2024_2025_ALLOCATION_ONLY_PARENT"
    ph["source_max_week"] = num(ph["week"]) - 1

    ch = current_history[[
        "season","week","player_identity_key","allocation_error",
        "history_source","source_max_week"
    ]].copy()
    h = pd.concat([ph, ch], ignore_index=True)
    h["season"] = num(h["season"]).astype(int)
    h["week"] = num(h["week"]).astype(int)
    h["allocation_error"] = num(h["allocation_error"])
    h = h.loc[
        h["player_identity_key"].astype(str).isin(set(cohort["player_identity_key"].astype(str)))
        & h["allocation_error"].notna()
    ].copy()
    h = h.sort_values(["season","week"]).reset_index(drop=True)

    rows = []
    for r in cohort.itertuples(index=False):
        pid = str(r.player_identity_key)
        q = h.loc[h["player_identity_key"].astype(str).eq(pid)].sort_values(["season","week"]).tail(PRIOR_WINDOW)
        supported = len(q) >= MIN_PRIOR
        mae = float(np.mean(np.abs(q["allocation_error"].to_numpy(float)))) if supported else np.nan
        rows.append({
            "season": SEASON,
            "week": WEEK,
            "event_id": str(r.event_id),
            "team": str(r.team),
            "player": str(r.player),
            "player_clean_key": str(r.player_clean_key),
            "player_identity_key": pid,
            "te_r5p_entitlement_tgt_share": float(r.te_r5p_entitlement_tgt_share),
            "prior_games": int(len(q)),
            "history_min_season": int(q.iloc[0]["season"]) if len(q) else np.nan,
            "history_min_week": int(q.iloc[0]["week"]) if len(q) else np.nan,
            "history_max_season": int(q.iloc[-1]["season"]) if len(q) else np.nan,
            "history_max_week": int(q.iloc[-1]["week"]) if len(q) else np.nan,
            "player_mae": mae,
            "supported": bool(supported),
        })
    out = pd.DataFrame(rows)
    supported = out.loc[out["supported"]].copy()
    if supported.empty:
        raise RuntimeError("zero supported Week6 TE trust rows")
    pool = float(supported["player_mae"].median())
    if not np.isfinite(pool) or pool <= 0:
        raise RuntimeError("invalid Week6 TE difficulty pool median")
    supported["week_pool_mae"] = pool
    supported["difficulty_ratio_vs_pool"] = supported["player_mae"] / pool
    supported["trust_fragility_score"] = supported["player_mae"].rank(method="average", pct=True)
    supported["trust_bucket"] = "MIDDLE"
    supported.loc[supported["trust_fragility_score"].le(.25), "trust_bucket"] = "HIGH_TRUST"
    supported.loc[supported["trust_fragility_score"].ge(.75), "trust_bucket"] = "LOW_TRUST"
    out = out.merge(
        supported[[
            "player_identity_key","week_pool_mae","difficulty_ratio_vs_pool",
            "trust_fragility_score","trust_bucket"
        ]],
        on="player_identity_key", how="left", validate="one_to_one",
    )
    out["trust_bucket"] = out["trust_bucket"].fillna("UNSUPPORTED")

    canonical = out.sort_values(["team","player_clean_key"]).reset_index(drop=True)
    payload = canonical.to_csv(index=False, float_format="%.12f", lineterminator="\n").encode("utf-8")
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    history_counts = {
        str(k): int(v) for k, v in h.groupby("season").size().to_dict().items()
    }
    buckets = {
        str(k): int(v) for k, v in canonical["trust_bucket"].value_counts().to_dict().items()
    }
    result = {
        "version": VERSION,
        "status": "WEEK6_TE_TRUST_PREGAME_LOCK_FROZEN",
        "parent_disposition": parent_trust.get("disposition"),
        "season": SEASON,
        "week": WEEK,
        "current_te_r5p_cohort": int(len(canonical)),
        "supported_te": int(canonical["supported"].sum()),
        "unsupported_te": int((~canonical["supported"]).sum()),
        "trust_bucket_counts": buckets,
        "week_pool_mae": pool,
        "history_rows_by_season": history_counts,
        "reconstructed_2026_rows": int(len(current_history)),
        "history_source_max_week_2026": int(num(current_history["week"]).max()),
        "team_target_semantic_parity": parity,
        "row_csv_sha256": digest,
        "prior_window": PRIOR_WINDOW,
        "min_prior_games": MIN_PRIOR,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "week6_outcomes_read": 0,
        "models_fit": 0,
        "projection_distribution_values_changed": 0,
        "production_changed": False,
    }
    return canonical, h, result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week6-te-trace", type=Path, required=True)
    ap.add_argument("--week6-volatility", type=Path, required=True)
    ap.add_argument("--week6-pack-result", type=Path, required=True)
    ap.add_argument("--parent-allocation-rows", type=Path, required=True)
    ap.add_argument("--parent-trust-result", type=Path, required=True)
    ap.add_argument("--te-authority-2025", type=Path, required=True)
    ap.add_argument("--component-2025", type=Path, required=True)
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    parity = semantic_parity(
        read_csv(a.te_authority_2025, "TE 2025 authority"),
        read_csv(a.component_2025, "2025 component predictions"),
    )
    cohort = current_te_cohort(
        read_csv(a.week6_te_trace, "Week6 TE trace"),
        read_csv(a.week6_volatility, "Week6 volatility lock"),
    )
    player_logs = read_csv(a.player_logs, "historical player logs")
    if not player_logs.empty:
        current = player_logs.loc[num(player_logs["season"]).eq(SEASON)]
        bad = sorted(num(current.loc[num(current["week"]).ge(WEEK), "week"]).dropna().astype(int).unique().tolist())
        if bad:
            raise RuntimeError(f"Week6-or-later outcomes present in source: {bad}")

    current_history = reconstruct_2026_errors(
        cohort=cohort,
        player_logs=player_logs,
        team_weekly=read_csv(a.team_weekly, "team weekly"),
        schedule=read_csv(a.schedule, "schedule"),
        universe_dir=a.universe_dir,
    )
    rows, history, result = lock_trust(
        cohort=cohort,
        parent_history=read_csv(a.parent_allocation_rows, "allocation-only parent"),
        current_history=current_history,
        parity=parity,
        parent_trust=json.loads(a.parent_trust_result.read_text(encoding="utf-8")),
        opportunity_pack=json.loads(a.week6_pack_result.read_text(encoding="utf-8")),
    )
    rows.to_csv(a.out_dir / "te_week6_trust_lock.csv", index=False, float_format="%.12f")
    history.to_csv(a.out_dir / "te_week6_trust_history.csv", index=False, float_format="%.12f")
    current_history.to_csv(a.out_dir / "te_week6_reconstructed_2026_allocation_errors.csv", index=False, float_format="%.12f")
    (a.out_dir / "te_week6_trust_lock.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
