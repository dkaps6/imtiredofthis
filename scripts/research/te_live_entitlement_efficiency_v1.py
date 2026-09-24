#!/usr/bin/env python3
"""Live 2026 TE entitlement-vs-efficiency diagnostic.

Research-only. Reuses the exact paid Week-2 origin artifact and final nflverse
outcomes. It never fetches sportsbook data and never changes production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.modeling import te_r5p_entitlement_adapter_v1 as T
from scripts.utils.canonical_names import canonicalize_player_name_safe

SUPPORTED_SEASON = 2026
SUPPORTED_WEEK = 2
ORIGIN_RUN_ID = "35282021679"
ORIGIN_ARTIFACT_ID = "10523345092"
ORIGIN_ARTIFACT_DIGEST = "sha256:6024ed21d7032e6e6440145277d9d4b74e98f319ef286962bac2a786a8d28e3c"


def _pd(obj) -> pd.DataFrame:
    return obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _load_origin(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = root / "data"
    need = {
        "te_trace": data / "te_r5p_full_slate_entitlement_trace.csv",
        "target_trace": data / "target_entitlement_v1_trace.csv",
        "universe": data / "football_simulation_universe.csv",
    }
    for label, path in need.items():
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing Week-2 origin evidence {label}: {path}")
    te = pd.read_csv(need["te_trace"], low_memory=False)
    target = pd.read_csv(need["target_trace"], low_memory=False)
    universe = pd.read_csv(need["universe"], low_memory=False)
    return te, target, universe


def _load_week2_actuals() -> pd.DataFrame:
    import nflreadpy as nfl

    x = _pd(nfl.load_player_stats(seasons=[SUPPORTED_SEASON], summary_level="week"))
    x.columns = [str(c).strip().lower() for c in x.columns]
    x = x.loc[
        _num(x.get("season")).eq(SUPPORTED_SEASON)
        & _num(x.get("week")).eq(SUPPORTED_WEEK)
    ].copy()
    if x.empty:
        raise RuntimeError("nflreadpy returned zero Week-2 player-stat rows")
    x["gsis_id"] = x.get("player_id", x.get("gsis_id", "")).astype("string").fillna("").str.strip()
    team_col = next((c for c in ("recent_team", "team", "team_abbr") if c in x.columns), None)
    if team_col is None:
        raise RuntimeError("Week-2 stats missing team")
    x["team"] = x[team_col].astype("string").fillna("").map(canon_team)
    raw_name = x.get("player_display_name", x.get("player_name", "")).astype("string").fillna("")
    canon = raw_name.map(canonicalize_player_name_safe)
    x["player_clean_key"] = canon.map(lambda t: t[1])
    x["position"] = x.get("position", x.get("position_group", "")).astype("string").fillna("").str.upper().str.strip()
    for src, dst in [
        ("targets", "actual_targets"),
        ("receptions", "actual_receptions"),
        ("receiving_yards", "actual_rec_yards"),
    ]:
        if src not in x.columns:
            raise RuntimeError(f"Week-2 stats missing {src}")
        x[dst] = _num(x[src]).fillna(0.0)
    return x[[
        "gsis_id", "team", "player_clean_key", "position",
        "actual_targets", "actual_receptions", "actual_rec_yards",
    ]].drop_duplicates(["gsis_id"], keep="last")


def _load_current_snap_history() -> tuple[pd.DataFrame, dict]:
    """Load historical snaps plus Week-1 2026 for a research-only W2 counterfactual."""
    import nflreadpy as nfl

    seasons = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    q = T._lower(T._pdx(nfl.load_snap_counts(seasons=seasons)))
    if "season" not in q.columns or "week" not in q.columns:
        raise RuntimeError("snap source missing season/week")
    q["season"] = _num(q["season"])
    q["week"] = _num(q["week"])
    q = q.loc[q["season"].isin(seasons) & q["week"].between(1, 18)].copy()
    q["team"] = T._first(q, ["team", "team_abbr", "club"]).map(T._team)
    q["player_key"] = T._first(q, ["player", "player_name", "full_name"]).map(T._key)
    q["offense_pct"] = _num(T._first(q, ["offense_pct", "offense_percentage"]))
    q["offense_snaps"] = _num(T._first(q, ["offense_snaps"]))
    q["ordinal"] = q["season"] * 100 + q["week"]
    q = q.loc[q["team"].ne("") & q["player_key"].ne("")].copy()
    keys = ["season", "week", "team", "player_key"]
    dup_rate = float(q.duplicated(keys, keep=False).mean()) if len(q) else 1.0
    q = q.sort_values(keys, kind="stable").drop_duplicates(keys, keep="last").reset_index(drop=True)
    expected = T._expected_regular_season_teams(2026, 1)
    freshness = T._validate_snap_week_coverage(q, season=2026, week=1, expected_teams=expected)
    if dup_rate > 0.01:
        raise RuntimeError(f"research snap duplicate rate too high: {dup_rate}")
    return q, {"dup_rate": dup_rate, "week1_freshness": freshness}


def _candidate_current_snap_entitlement(
    te_trace: pd.DataFrame,
    target_trace: pd.DataFrame,
    snaps: pd.DataFrame,
) -> pd.DataFrame:
    """Re-run frozen TE-R5P using W1 2026 snaps; preserve the exact W2 TE pool."""
    key = ["event_id", "team", "player_clean_key"]
    t = te_trace.copy()
    base = t[[
        "event_id", "season", "week", "team", "player", "player_clean_key",
        "baseline_entitlement_tgt_share", "te_r5p_entitlement_tgt_share",
        "te_r5p_room_share", "b0_te_pool",
    ]].copy()
    extra = target_trace[key + ["position", "residual_share"]].drop_duplicates(key)
    base = base.merge(extra, on=key, how="left", validate="one_to_one")
    if base["position"].isna().any():
        raise RuntimeError("could not recover position for every TE trace row")
    model = T._load_model()

    inp = base[[
        "event_id", "season", "week", "team", "player", "player_clean_key",
        "position", "baseline_entitlement_tgt_share", "residual_share",
    ]].rename(columns={
        "baseline_entitlement_tgt_share": "entitlement_tgt_share",
        "residual_share": "entitlement_residual_share",
    })
    feat = T._strict_prior_features(inp, snaps)
    feat["b0_te_pool"] = feat.groupby(["event_id", "team"])["entitlement_tgt_share"].transform("sum")
    feat["b0_te_room_share"] = np.where(
        feat["b0_te_pool"].gt(0),
        feat["entitlement_tgt_share"] / feat["b0_te_pool"],
        0.0,
    )
    feat["log_b0_te_pool"] = np.log1p(feat["b0_te_pool"].clip(lower=0))
    feat["pool_ratio"] = 1.0
    feat["room_size"] = feat.groupby(["event_id", "team"])["player_clean_key"].transform("count").astype(float)
    feat["prior1_same_team_available"] = feat["prior1_same_team"].fillna(False).astype(float)
    feat["prior3_same_team_available"] = feat["prior3_same_team"].fillna(False).astype(float)
    feat["log1p_prior_count_same_team"] = np.log1p(_num(feat["prior_count_same_team"]).fillna(0).clip(lower=0))
    feat["log1p_prior_count_anyteam"] = np.log1p(_num(feat["prior_count_anyteam"]).fillna(0).clip(lower=0))
    for src, dst in [
        ("prior1_same_team_offense_pct", "snap_share_prior1_same_team"),
        ("prior3_anyteam_offense_pct", "snap_share_prior3_anyteam"),
    ]:
        z = _num(feat[src]).fillna(0).clip(lower=0)
        den = z.groupby([feat["event_id"], feat["team"]]).transform("sum")
        feat[dst] = np.where(den.gt(0), z / den, 0.0)

    features = list(model["features"])
    for col in features:
        feat[col] = _num(feat[col]).fillna(0.0)
    x = feat[features].to_numpy(float)
    z = (x - np.asarray(model["scaler_mean"], float)) / np.asarray(model["scaler_scale"], float)
    residual = z @ np.asarray(model["ridge_coef"], float) + float(model["ridge_intercept"])
    lo, hi = [float(v) for v in model["prediction_clip"]]
    residual = np.clip(residual, lo, hi)
    eps = float(model["eps"])
    feat["candidate_score"] = np.log(feat["b0_te_room_share"].clip(lower=0).to_numpy(float) + eps) + residual
    feat["candidate_room_share"] = 0.0
    feat["candidate_entitlement_tgt_share"] = 0.0

    for _, idx in feat.groupby(["event_id", "team"], sort=False).groups.items():
        pool = float(feat.loc[idx, "b0_te_pool"].iloc[0])
        score = feat.loc[idx, "candidate_score"].to_numpy(float)
        if pool <= 0:
            room = np.zeros(len(idx), dtype=float)
            candidate = np.zeros(len(idx), dtype=float)
        else:
            stable = score - float(np.max(score))
            w = np.exp(stable)
            room = w / float(w.sum())
            candidate = pool * room
            candidate[int(np.argmax(room))] += pool - float(candidate.sum())
        feat.loc[idx, "candidate_room_share"] = room
        feat.loc[idx, "candidate_entitlement_tgt_share"] = candidate

    out = base.merge(
        feat[["_row_index", "event_id", "team", "player_clean_key",
              "prior1_same_team_offense_pct", "prior1_same_team_offense_snaps",
              "candidate_room_share", "candidate_entitlement_tgt_share"]],
        on=["event_id", "team", "player_clean_key"],
        how="left",
        validate="one_to_one",
    )
    gap = (
        out.groupby(["event_id", "team"])["candidate_entitlement_tgt_share"].sum()
        - out.groupby(["event_id", "team"])["b0_te_pool"].first()
    ).abs()
    if float(gap.max()) > 1e-12:
        raise RuntimeError(f"candidate TE-pool conservation failure: {float(gap.max())}")
    out["candidate_pool_gap"] = out["candidate_entitlement_tgt_share"].groupby(
        [out["event_id"], out["team"]]
    ).transform("sum") - out["b0_te_pool"]
    return out


def _attach_actuals(frame: pd.DataFrame, universe: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    key = ["event_id", "team", "player_clean_key"]
    ids = universe.copy()
    ids.columns = [str(c).strip().lower() for c in ids.columns]
    ids = ids.loc[ids["position"].astype(str).str.upper().eq("TE")].copy()
    ids["team"] = ids["team"].map(canon_team)
    ids["player_clean_key"] = ids["player_clean_key"].astype(str)
    ids["gsis_id"] = ids.get("player_id", "").astype("string").fillna("").str.strip()
    ids = ids[key + ["gsis_id"]].drop_duplicates(key)
    out = frame.merge(ids, on=key, how="left", validate="one_to_one")
    out = out.merge(
        actual[["gsis_id", "actual_targets", "actual_receptions", "actual_rec_yards"]],
        on="gsis_id", how="left", validate="many_to_one",
    )

    # Deterministic same-team canonical-name fallback for source rows lacking GSIS.
    miss = out["actual_targets"].isna()
    if miss.any():
        fallback = actual[[
            "team", "player_clean_key", "actual_targets", "actual_receptions", "actual_rec_yards"
        ]].drop_duplicates(["team", "player_clean_key"])
        fb = out.loc[miss, key].merge(
            fallback, on=["team", "player_clean_key"], how="left", validate="many_to_one"
        )
        for col in ("actual_targets", "actual_receptions", "actual_rec_yards"):
            out.loc[miss, col] = fb[col].to_numpy()

    team_targets = actual.groupby("team", as_index=False)["actual_targets"].sum().rename(
        columns={"actual_targets": "actual_team_targets"}
    )
    te_targets = actual.loc[actual["position"].eq("TE")].groupby(
        "team", as_index=False
    )["actual_targets"].sum().rename(columns={"actual_targets": "actual_te_room_targets"})
    out = out.merge(team_targets, on="team", how="left", validate="many_to_one")
    out = out.merge(te_targets, on="team", how="left", validate="many_to_one")
    out["actual_target_share"] = np.where(
        out["actual_team_targets"].gt(0),
        out["actual_targets"] / out["actual_team_targets"],
        np.nan,
    )
    out["actual_te_room_share"] = np.where(
        out["actual_te_room_targets"].gt(0),
        out["actual_targets"] / out["actual_te_room_targets"],
        np.nan,
    )
    return out


def _metric(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    return float((a[mask] - b[mask]).abs().mean()) if mask.any() else float("nan")


def run(origin_dir: Path, ledger_path: Path, out_dir: Path) -> dict:
    te_trace, target_trace, universe = _load_origin(origin_dir)
    actual = _load_week2_actuals()
    snaps, snap_audit = _load_current_snap_history()
    live = _candidate_current_snap_entitlement(te_trace, target_trace, snaps)
    live = _attach_actuals(live, universe, actual)
    live["production_target_share_abs_error"] = (
        live["te_r5p_entitlement_tgt_share"] - live["actual_target_share"]
    ).abs()
    live["candidate_target_share_abs_error"] = (
        live["candidate_entitlement_tgt_share"] - live["actual_target_share"]
    ).abs()
    live["production_room_share_abs_error"] = (
        live["te_r5p_room_share"] - live["actual_te_room_share"]
    ).abs()
    live["candidate_room_share_abs_error"] = (
        live["candidate_room_share"] - live["actual_te_room_share"]
    ).abs()

    matched = live.loc[live["actual_targets"].notna()].copy()
    if len(matched) < 60:
        raise RuntimeError(f"too few exact Week-2 TE outcome matches: {len(matched)}")

    prod_ts_mae = _metric(matched["te_r5p_entitlement_tgt_share"], matched["actual_target_share"])
    cand_ts_mae = _metric(matched["candidate_entitlement_tgt_share"], matched["actual_target_share"])
    prod_room_mae = _metric(matched["te_r5p_room_share"], matched["actual_te_room_share"])
    cand_room_mae = _metric(matched["candidate_room_share"], matched["actual_te_room_share"])
    ts_gain = (prod_ts_mae - cand_ts_mae) / prod_ts_mae if prod_ts_mae > 0 else np.nan

    q75 = float(matched["production_target_share_abs_error"].quantile(0.75))
    top = matched.loc[matched["production_target_share_abs_error"].ge(q75)].copy()
    top_prod = float(top["production_target_share_abs_error"].mean())
    top_cand = float(top["candidate_target_share_abs_error"].mean())

    ledger = pd.read_csv(ledger_path, low_memory=False)
    selected = ledger.loc[
        _num(ledger["season"]).eq(2026)
        & _num(ledger["week"]).eq(2)
        & ledger["position"].astype(str).eq("TE")
        & ledger["market"].astype(str).eq("rec_yards")
        & ledger["settlement_status"].astype(str).eq("SETTLED")
    ].copy()
    selected = selected.merge(
        live[[
            "event_id", "team", "player_clean_key", "gsis_id",
            "actual_targets", "actual_receptions", "actual_rec_yards",
            "actual_team_targets", "actual_target_share", "actual_te_room_share",
            "te_r5p_entitlement_tgt_share", "candidate_entitlement_tgt_share",
            "te_r5p_room_share", "candidate_room_share",
            "prior1_same_team_offense_pct", "prior1_same_team_offense_snaps",
        ]],
        on=["event_id", "team", "player_clean_key"],
        how="left",
        validate="many_to_one",
    )
    # The canonical ledger proves zero outcome for snap-confirmed no-stat rows.
    zero_verified = selected["actual_source"].astype(str).eq("snap_confirmed_verified_zero")
    for col in ("actual_targets", "actual_receptions", "actual_rec_yards"):
        selected.loc[zero_verified & selected[col].isna(), col] = 0.0

    selected["production_target_count_proxy"] = (
        selected["te_r5p_entitlement_tgt_share"] * selected["actual_team_targets"]
    )
    selected["candidate_target_count_proxy"] = (
        selected["candidate_entitlement_tgt_share"] * selected["actual_team_targets"]
    )
    selected["implied_final_ypt"] = np.where(
        selected["production_target_count_proxy"].gt(0.05),
        _num(selected["model_proj"]) / selected["production_target_count_proxy"],
        np.nan,
    )
    selected["actual_ypt"] = np.where(
        selected["actual_targets"].gt(0),
        selected["actual_rec_yards"] / selected["actual_targets"],
        np.nan,
    )
    selected["perfect_entitlement_cf_rec_yards"] = (
        selected["actual_targets"] * selected["implied_final_ypt"]
    )
    selected["perfect_efficiency_cf_rec_yards"] = (
        selected["production_target_count_proxy"] * selected["actual_ypt"]
    )
    selected["current_snap_cf_rec_yards"] = (
        selected["candidate_target_count_proxy"] * selected["implied_final_ypt"]
    )
    selected["final_abs_error"] = (_num(selected["model_proj"]) - selected["actual_rec_yards"]).abs()
    selected["mc_abs_error"] = (_num(selected["mc_proj"]) - selected["actual_rec_yards"]).abs()
    selected["perfect_entitlement_abs_error"] = (
        selected["perfect_entitlement_cf_rec_yards"] - selected["actual_rec_yards"]
    ).abs()
    selected["perfect_efficiency_abs_error"] = (
        selected["perfect_efficiency_cf_rec_yards"] - selected["actual_rec_yards"]
    ).abs()
    selected["current_snap_cf_abs_error"] = (
        selected["current_snap_cf_rec_yards"] - selected["actual_rec_yards"]
    ).abs()
    selected["entitlement_error_recovery"] = (
        selected["final_abs_error"] - selected["perfect_entitlement_abs_error"]
    )
    selected["efficiency_error_recovery"] = (
        selected["final_abs_error"] - selected["perfect_efficiency_abs_error"]
    )
    selected["final_vs_mc_error_delta"] = selected["final_abs_error"] - selected["mc_abs_error"]

    selected_matched = selected.loc[
        selected["actual_targets"].notna() & selected["actual_team_targets"].gt(0)
    ].copy()
    if len(selected_matched) < 25:
        raise RuntimeError(f"too few selected TE rec-yard rows with target outcomes: {len(selected_matched)}")
    sel_prod_mae = _metric(
        selected_matched["te_r5p_entitlement_tgt_share"],
        selected_matched["actual_target_share"],
    )
    sel_cand_mae = _metric(
        selected_matched["candidate_entitlement_tgt_share"],
        selected_matched["actual_target_share"],
    )
    sel_change = (sel_cand_mae - sel_prod_mae) / sel_prod_mae if sel_prod_mae > 0 else np.nan

    gates = {
        "all_te_target_share_mae_gain_ge_2pct": bool(ts_gain >= 0.02),
        "all_te_room_share_mae_nonworse": bool(cand_room_mae <= prod_room_mae + 1e-12),
        "selected_te_target_share_mae_not_worse_gt_1pct": bool(sel_change <= 0.01),
        "te_pool_conservation_le_1e_12": bool(live["candidate_pool_gap"].abs().max() <= 1e-12),
        "sportsbook_inputs_in_candidate": 0,
    }
    gates["current_snap_entitlement_signal_supported"] = bool(
        gates["all_te_target_share_mae_gain_ge_2pct"]
        and gates["all_te_room_share_mae_nonworse"]
        and gates["selected_te_target_share_mae_not_worse_gt_1pct"]
        and gates["te_pool_conservation_le_1e_12"]
    )

    w12 = ledger.loc[
        ledger["position"].astype(str).eq("TE")
        & ledger["market"].astype(str).isin(["rec_yards", "receptions"])
        & ledger["bet_result"].astype(str).isin(["WIN", "LOSS"])
    ].copy()
    context = {}
    for (week, market), g in w12.groupby(["week", "market"]):
        context[f"W{int(week)}_{market}"] = {
            "n": int(len(g)),
            "wins": int(g["bet_result"].eq("WIN").sum()),
            "losses": int(g["bet_result"].eq("LOSS").sum()),
            "model_mae": float(_num(g["model_error"]).abs().mean()),
            "model_bias": float(_num(g["model_error"]).mean()),
        }

    summary = {
        "study": "TE_LIVE_ENTITLEMENT_EFFICIENCY_V1",
        "status": "research_only",
        "origin_run_id": ORIGIN_RUN_ID,
        "origin_artifact_id": ORIGIN_ARTIFACT_ID,
        "origin_artifact_digest": ORIGIN_ARTIFACT_DIGEST,
        "sportsbook_inputs_added_to_football_model": 0,
        "production_changed": False,
        "week2_all_te": {
            "matched_rows": int(len(matched)),
            "production_target_share_mae": prod_ts_mae,
            "current_snap_candidate_target_share_mae": cand_ts_mae,
            "relative_target_share_mae_gain": float(ts_gain),
            "production_room_share_mae": prod_room_mae,
            "current_snap_candidate_room_share_mae": cand_room_mae,
            "top_quartile_production_target_share_mae": top_prod,
            "top_quartile_candidate_target_share_mae": top_cand,
        },
        "week2_selected_te_rec_yards": {
            "rows": int(len(selected)),
            "target_outcome_matched_rows": int(len(selected_matched)),
            "record": {
                "wins": int(selected["bet_result"].eq("WIN").sum()),
                "losses": int(selected["bet_result"].eq("LOSS").sum()),
            },
            "production_target_share_mae": sel_prod_mae,
            "current_snap_candidate_target_share_mae": sel_cand_mae,
            "relative_candidate_mae_change": float(sel_change),
            "mean_final_abs_error": float(selected_matched["final_abs_error"].mean()),
            "mean_mc_abs_error": float(selected_matched["mc_abs_error"].mean()),
            "mean_entitlement_error_recovery": float(selected_matched["entitlement_error_recovery"].mean()),
            "mean_efficiency_error_recovery_nonzero_targets": float(
                selected_matched.loc[selected_matched["actual_targets"].gt(0), "efficiency_error_recovery"].mean()
            ),
            "zero_actual_target_rows": int(selected_matched["actual_targets"].eq(0).sum()),
            "mean_current_snap_cf_abs_error": float(selected_matched["current_snap_cf_abs_error"].mean()),
            "mean_final_vs_mc_error_delta": float(selected_matched["final_vs_mc_error_delta"].mean()),
        },
        "snap_audit": snap_audit,
        "gates": gates,
        "w1_w2_te_context": context,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    live.to_csv(out_dir / "week2_all_te_entitlement.csv", index=False)
    selected.to_csv(out_dir / "week2_selected_te_rec_yards.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    disposition = (
        "CURRENT_SNAP_ENTITLEMENT_SIGNAL_SUPPORTED"
        if gates["current_snap_entitlement_signal_supported"]
        else "CURRENT_SNAP_ENTITLEMENT_SIGNAL_NOT_CONFIRMED"
    )
    lines = [
        "# TE Live Entitlement vs Efficiency V1 — Result",
        "",
        f"**Disposition:** `{disposition}`",
        "",
        "Research only. No production change and no sportsbook input added to football projections.",
        "",
        "## Week-2 all-TE entitlement",
        "",
        f"- matched rows: **{len(matched)}**",
        f"- production target-share MAE: **{prod_ts_mae:.5f}**",
        f"- current-snap candidate target-share MAE: **{cand_ts_mae:.5f}**",
        f"- relative target-share MAE gain: **{ts_gain*100:.2f}%**",
        f"- production TE-room-share MAE: **{prod_room_mae:.5f}**",
        f"- current-snap candidate TE-room-share MAE: **{cand_room_mae:.5f}**",
        f"- worst production-error quartile target-share MAE: **{top_prod:.5f} -> {top_cand:.5f}**",
        "",
        "## Canonical Week-2 selected TE receiving yards",
        "",
        f"- selected rows: **{len(selected)}**",
        f"- target-outcome matched: **{len(selected_matched)}**",
        f"- record: **{int(selected['bet_result'].eq('WIN').sum())}-{int(selected['bet_result'].eq('LOSS').sum())}**",
        f"- target-share MAE production -> current-snap candidate: **{sel_prod_mae:.5f} -> {sel_cand_mae:.5f}**",
        f"- mean final projection absolute error: **{selected_matched['final_abs_error'].mean():.2f} yd**",
        f"- mean MC absolute error: **{selected_matched['mc_abs_error'].mean():.2f} yd**",
        f"- mean recoverable error with perfect target entitlement: **{selected_matched['entitlement_error_recovery'].mean():+.2f} yd**",
        f"- mean recoverable error with perfect realized efficiency (nonzero-target rows): **{selected_matched.loc[selected_matched['actual_targets'].gt(0), 'efficiency_error_recovery'].mean():+.2f} yd**",
        f"- zero-actual-target rows: **{int(selected_matched['actual_targets'].eq(0).sum())}**",
        f"- current-snap counterfactual mean absolute error: **{selected_matched['current_snap_cf_abs_error'].mean():.2f} yd**",
        f"- final-minus-MC absolute-error delta: **{selected_matched['final_vs_mc_error_delta'].mean():+.2f} yd** (positive = final ensemble worse)",
        "",
        "## Frozen gates",
        "",
    ]
    for k, v in gates.items():
        lines.append(f"- {k}: **{v}**")
    lines += [
        "",
        "## Interpretation rule",
        "",
        "This run diagnoses whether Week-2 TE misses came from opportunity/allocation, efficiency, or downstream ensemble movement. It does not authorize coefficient tuning from two live weeks.",
        "",
        "Week 1 is retained as canonical scoreboard context only because its exact full pregame entitlement artifact expired; this study does not invent a Week-1 mechanism trace.",
    ]
    (out_dir / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"disposition={disposition}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin-dir", type=Path, required=True)
    ap.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/market_track_record/graded/2026_wk01_wk02_graded.csv"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/te_live_v1"))
    args = ap.parse_args()
    run(args.origin_dir, args.ledger, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
