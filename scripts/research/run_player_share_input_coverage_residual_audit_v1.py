#!/usr/bin/env python3
"""Audit existing player-share inputs/stages against frozen ACT-only realized share.

No candidate is fit. Pregame stage traces are reconstructed first; realized
shares from the frozen parent are joined only after the trace is complete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import build_market_frame
from scripts.backtest.historical_context import (
    assert_no_future_rows,
    build_historical_context_bundle,
)
from scripts.modeling.bayesian_v2 import (
    apply_bayesian_to_metrics,
    build_bayesian_baseline,
)
from scripts.modeling import simulation_rules
from scripts.modeling.target_entitlement_v1 import (
    ALLOCATOR_SAFE_CAP,
    TARGET_MASS_CAP,
    materialize_target_entitlement,
)
from scripts.modeling.te_r5p_entitlement_adapter_v1 import apply_te_r5p_entitlement
from scripts.modeling.wr_r15_entitlement_adapter_v1 import apply_wr_r15_entitlement
from scripts.simulation_v2 import _top_n_shares

SEASON = 2026
PRIOR_SEASON = 2025
WEEKS = (1, 2, 3, 4)
TOL = 1e-10
TARGET_POSITIONS = {"RB", "FB", "WR", "LWR", "RWR", "SWR", "TE"}
WR_POSITIONS = {"WR", "LWR", "RWR", "SWR"}
HIGH_BIN = {
    ("RB", "carries"): "15_PLUS",
    ("RB", "targets"): "09_PLUS",
    ("WR", "targets"): "09_PLUS",
    ("TE", "targets"): "09_PLUS",
}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _pos(v) -> str:
    p = str(v or "").upper().strip()
    if p in {"HB", "TB"} or p.startswith("RB") or p.startswith("FB"):
        return "RB"
    if p.startswith("WR") or p in {"LWR", "RWR", "SWR"}:
        return "WR"
    if p.startswith("TE"):
        return "TE"
    return p


def _num_series(frame: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _target_allocator_equivalent(frame: pd.DataFrame, column: str) -> pd.Series:
    """Map a target-share input to the canonical allocator probability scale.

    No M38 or specialist redistribution is applied here. This only performs the
    canonical finite player-mass cap so all stages are compared on the same
    per-dropback target-allocation probability scale as the corrected realized
    parent share.
    """
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for (_, _), idx in frame.groupby(["event_id", "team"], sort=False).groups.items():
        raw = pd.to_numeric(frame.loc[idx, column], errors="coerce")
        clean = np.clip(np.nan_to_num(raw.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, TARGET_MASS_CAP)
        total = float(clean.sum())
        if total > TARGET_MASS_CAP:
            clean *= ALLOCATOR_SAFE_CAP / total
        # Preserve row-level evidence availability: missing raw evidence is not
        # re-labeled as a real zero even though zero is used in team arithmetic.
        vals = pd.Series(clean, index=idx, dtype=float)
        vals.loc[raw.index[raw.isna()]] = np.nan
        out.loc[idx] = vals
    return out


def _rush_allocator_equivalent(frame: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    """Map a rush-share input through canonical top-five + 0.95 allocator."""
    prob = pd.Series(np.nan, index=frame.index, dtype=float)
    member = pd.Series(False, index=frame.index, dtype=bool)
    for (_, _), idx in frame.groupby(["event_id", "team"], sort=False).groups.items():
        raw = pd.to_numeric(frame.loc[idx, column], errors="coerce")
        clean = np.clip(np.nan_to_num(raw.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, TARGET_MASS_CAP)
        top = _top_n_shares(clean, 5)
        selected = top > 0
        total = float(top.sum())
        if total > TARGET_MASS_CAP:
            top *= TARGET_MASS_CAP / total
        vals = pd.Series(top, index=idx, dtype=float)
        vals.loc[raw.index[raw.isna()]] = np.nan
        prob.loc[idx] = vals
        member.loc[idx] = selected
    return prob, member


def _consensus_fields(bundle) -> pd.DataFrame:
    c = bundle.player_consensus.copy()
    c.columns = [str(x).strip().lower() for x in c.columns]
    keep = [
        "team", "player_clean_key", "prior_games", "current_games",
        "tgt_share", "tgt_share_prior", "tgt_share_current",
        "rush_share", "rush_share_prior", "rush_share_current",
    ]
    for col in keep:
        if col not in c.columns:
            c[col] = np.nan
    return c[keep].drop_duplicates(["team", "player_clean_key"])


def _build_week_stage_trace(
    *,
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    universe: pd.DataFrame,
    week: int,
) -> tuple[pd.DataFrame, dict]:
    bundle = build_historical_context_bundle(
        player_logs=player_logs,
        team_weekly=team_weekly,
        pregame_universe=universe,
        schedule=schedule,
        season=SEASON,
        week=week,
        prior_season=PRIOR_SEASON,
    )
    assert_no_future_rows(bundle.player_history, SEASON, week, f"W{week} player_history")
    assert_no_future_rows(bundle.team_history, SEASON, week, f"W{week} team_history")

    metrics = build_market_frame(bundle)
    bayes = build_bayesian_baseline(bundle.player_consensus)
    metrics = apply_bayesian_to_metrics(metrics, bayes)
    with patch.object(simulation_rules, "load_model_contexts", return_value=(bundle.teams, bundle.players)):
        metrics = simulation_rules.apply_rules_to_metrics(metrics)
    if int(pd.to_numeric(metrics["rules_applied"], errors="coerce").fillna(0).sum()) == 0:
        raise RuntimeError(f"W{week}: rules matched zero rows")

    player_cols = ["event_id", "team", "player_clean_key"]
    players = metrics.sort_values(player_cols).drop_duplicates(player_cols, keep="last").copy()
    if players.duplicated(player_cols).any():
        raise RuntimeError(f"W{week}: deterministic player frame is not unique")

    consensus = _consensus_fields(bundle)
    players = players.merge(
        consensus,
        on=["team", "player_clean_key"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_consensus"),
    )
    # Bayesian evidence state and uncertainty belong to the exact existing
    # posterior, not a reconstructed proxy.
    bayes_keep = [
        "team", "player_clean_key", "bayes_evidence_state",
        "bayes_tgt_share", "bayes_tgt_share_sd", "bayes_tgt_share_effective_n",
        "bayes_rush_share", "bayes_rush_share_sd", "bayes_rush_share_effective_n",
    ]
    b = bayes.copy()
    for col in bayes_keep:
        if col not in b.columns:
            b[col] = np.nan
    b = b[bayes_keep].drop_duplicates(["team", "player_clean_key"])
    # Avoid duplicate posterior columns already carried by metrics.
    for col in [x for x in bayes_keep if x not in {"team", "player_clean_key"} and x in players.columns]:
        players.drop(columns=[col], inplace=True)
    players = players.merge(b, on=["team", "player_clean_key"], how="left", validate="one_to_one")

    # Allocator-equivalent input stages before explicit entitlement.
    players["raw_target_allocator_share"] = _target_allocator_equivalent(players, "tgt_share")
    players["bayes_target_allocator_share"] = _target_allocator_equivalent(players, "bayes_tgt_share")
    players["rules_target_allocator_share"] = _target_allocator_equivalent(players, "rules_tgt_share")
    players["raw_rush_allocator_share"], players["raw_rush_top5_member"] = _rush_allocator_equivalent(players, "rush_share")
    players["bayes_rush_allocator_share"], players["bayes_rush_top5_member"] = _rush_allocator_equivalent(players, "bayes_rush_share")
    players["rules_rush_allocator_share"], players["rules_rush_top5_member"] = _rush_allocator_equivalent(players, "rules_rush_share")

    base, _ = materialize_target_entitlement(players)
    base["post_m38_entitlement_tgt_share"] = pd.to_numeric(base["entitlement_tgt_share"], errors="coerce")
    te_final, te_trace, te_audit = apply_te_r5p_entitlement(base)
    te_final["post_te_r5p_entitlement_tgt_share"] = pd.to_numeric(te_final["entitlement_tgt_share"], errors="coerce")
    final, wr_trace, wr_audit = apply_wr_r15_entitlement(te_final)
    final["final_target_allocator_share"] = pd.to_numeric(final["entitlement_tgt_share"], errors="coerce")
    final["final_rush_allocator_share"] = pd.to_numeric(final["rules_rush_allocator_share"], errors="coerce")

    # Specialist trace coverage/features.
    te = te_trace.copy()
    if not te.empty:
        te["position_family"] = "TE"
        te["specialist_prior1_anyteam_available"] = pd.to_numeric(te["prior_count_anyteam"], errors="coerce").ge(1)
        te["specialist_prior3_anyteam_available"] = pd.to_numeric(te["prior_count_anyteam"], errors="coerce").ge(3)
        te["specialist_prior1_same_team_available"] = pd.to_numeric(te["prior_count_same_team"], errors="coerce").ge(1)
        te["specialist_prior3_same_team_available"] = pd.to_numeric(te["prior_count_same_team"], errors="coerce").ge(3)
        te_keep = [
            "event_id", "team", "player_clean_key", "prior_count_anyteam", "prior_count_same_team",
            "specialist_prior1_anyteam_available", "specialist_prior3_anyteam_available",
            "specialist_prior1_same_team_available", "specialist_prior3_same_team_available",
            "te_r5p_residual", "te_r5p_score", "entitlement_delta",
        ]
        te = te[te_keep].rename(columns={"entitlement_delta": "te_r5p_entitlement_delta"})
        final = final.merge(te, on=["event_id", "team", "player_clean_key"], how="left", validate="one_to_one")

    wr = wr_trace.copy()
    if not wr.empty:
        wr["specialist_prior1_anyteam_available"] = pd.to_numeric(wr["prior_count_anyteam"], errors="coerce").ge(1)
        wr["specialist_prior3_anyteam_available"] = pd.to_numeric(wr["prior_count_anyteam"], errors="coerce").ge(3)
        wr["specialist_prior1_same_team_available"] = pd.to_numeric(wr["prior_count_same_team"], errors="coerce").ge(1)
        wr["specialist_prior3_same_team_available"] = pd.to_numeric(wr["prior_count_same_team"], errors="coerce").ge(3)
        wr_keep = [
            "event_id", "team", "player_clean_key", "prior_count_anyteam", "prior_count_same_team",
            "specialist_prior1_anyteam_available", "specialist_prior3_anyteam_available",
            "specialist_prior1_same_team_available", "specialist_prior3_same_team_available",
            "wr_r15_residual", "wr_r15_score", "entitlement_delta",
        ]
        wr = wr[wr_keep].rename(columns={
            "prior_count_anyteam": "wr_prior_count_anyteam",
            "prior_count_same_team": "wr_prior_count_same_team",
            "specialist_prior1_anyteam_available": "wr_specialist_prior1_anyteam_available",
            "specialist_prior3_anyteam_available": "wr_specialist_prior3_anyteam_available",
            "specialist_prior1_same_team_available": "wr_specialist_prior1_same_team_available",
            "specialist_prior3_same_team_available": "wr_specialist_prior3_same_team_available",
            "entitlement_delta": "wr_r15_entitlement_delta",
        })
        final = final.merge(wr, on=["event_id", "team", "player_clean_key"], how="left", validate="one_to_one")

    # Standardize specialist coverage names for WR/TE. WR1 anchors deliberately
    # consume no WR-R15 participation model; mark that explicitly.
    pos = final["position"].astype(str).str.upper().str.strip()
    is_wr = pos.isin(WR_POSITIONS)
    is_te = pos.eq("TE")
    for col in [
        "specialist_prior1_anyteam_available", "specialist_prior3_anyteam_available",
        "specialist_prior1_same_team_available", "specialist_prior3_same_team_available",
    ]:
        if col not in final.columns:
            final[col] = np.nan
    for suffix in ["prior1_anyteam_available", "prior3_anyteam_available", "prior1_same_team_available", "prior3_same_team_available"]:
        wcol = f"wr_specialist_{suffix}"
        if wcol not in final.columns:
            final[wcol] = np.nan
        final.loc[is_wr, f"specialist_{suffix}"] = final.loc[is_wr, wcol]
    final["specialist_evidence_consumed"] = False
    final.loc[is_te & final.get("te_r5p_applied", False).astype(bool), "specialist_evidence_consumed"] = True
    final.loc[is_wr & final.get("wr_r15_applied", False).astype(bool), "specialist_evidence_consumed"] = True

    final["season"] = SEASON
    final["week"] = week
    final["position_family"] = final["position"].map(_pos)
    final["target_share_trajectory_status"] = "FROZEN_FEATURE_NOT_YET_ELIGIBLE"
    final["rb_week5_room_allocation_shadow_applied"] = False
    final["sportsbook_inputs_used_upstream"] = False

    audit = {
        "week": week,
        "te": te_audit,
        "wr": wr_audit,
        "players": int(len(final)),
    }
    return final, audit


def _attach_parent(stage: pd.DataFrame, parent_path: Path) -> pd.DataFrame:
    """Attach realized share only after all pregame stage traces are frozen."""
    parent = _read(parent_path, "frozen volume/share parent")
    required = {
        "season", "week", "event_id", "team", "player_clean_key", "position_family",
        "opportunity_type", "actual_opportunities", "actual_team_volume",
        "actual_player_share", "actual_opportunity_bin",
        "linked_yards_error", "linked_count_error", "final_player_probability",
        "sportsbook_inputs_used_upstream",
    }
    missing = required - set(parent.columns)
    if missing:
        raise RuntimeError(f"corrected frozen parent missing columns: {sorted(missing)}")
    if parent["sportsbook_inputs_used_upstream"].astype(bool).any():
        raise RuntimeError("frozen parent contains sportsbook input")
    parent["actual_player_opportunity"] = pd.to_numeric(
        parent["actual_opportunities"], errors="coerce"
    )
    parent["predicted_player_share"] = pd.to_numeric(
        parent["final_player_probability"], errors="coerce"
    )
    if parent["actual_player_opportunity"].isna().any() or parent["predicted_player_share"].isna().any():
        raise RuntimeError("corrected frozen parent has non-finite opportunity/share")

    rows = []
    # Build carry + target player rows from deterministic share trace.
    for _, r in stage.iterrows():
        pos = str(r["position_family"])
        if pos == "RB":
            for opp in ("carries", "targets"):
                rec = r.to_dict(); rec["opportunity_type"] = opp; rows.append(rec)
        elif pos in {"WR", "TE"}:
            rec = r.to_dict(); rec["opportunity_type"] = "targets"; rows.append(rec)
    x = pd.DataFrame(rows)
    if x.empty:
        raise RuntimeError("share-stage audit produced zero scoped player rows")

    key = ["season", "week", "event_id", "team", "player_clean_key", "opportunity_type"]
    if x.duplicated(key).any():
        raise RuntimeError("share-stage pregame trace has duplicate identity")
    p = parent.copy()
    p["position_family"] = p["position_family"].map(_pos)
    keep = key + [
        "actual_player_opportunity", "actual_team_volume", "actual_player_share",
        "actual_opportunity_bin", "linked_yards_error", "linked_count_error",
        "predicted_player_share",
    ]
    p = p[keep].drop_duplicates(key)
    out = x.merge(p, on=key, how="inner", validate="one_to_one")
    if out.empty:
        raise RuntimeError("zero parent identities joined to share-stage trace")
    return out


def _stage_columns(row: pd.Series) -> list[tuple[str, str]]:
    pos = str(row["position_family"])
    opp = str(row["opportunity_type"])
    if pos == "RB" and opp == "carries":
        return [
            ("raw_history", "raw_rush_allocator_share"),
            ("bayes", "bayes_rush_allocator_share"),
            ("rules", "rules_rush_allocator_share"),
            ("final_allocator", "final_rush_allocator_share"),
        ]
    if pos == "RB" and opp == "targets":
        return [
            ("raw_history", "raw_target_allocator_share"),
            ("bayes", "bayes_target_allocator_share"),
            ("rules", "rules_target_allocator_share"),
            ("post_m38", "post_m38_entitlement_tgt_share"),
            ("final_allocator", "final_target_allocator_share"),
        ]
    if pos == "WR":
        return [
            ("raw_history", "raw_target_allocator_share"),
            ("bayes", "bayes_target_allocator_share"),
            ("rules", "rules_target_allocator_share"),
            ("post_m38", "post_m38_entitlement_tgt_share"),
            ("final_specialist", "final_target_allocator_share"),
        ]
    if pos == "TE":
        return [
            ("raw_history", "raw_target_allocator_share"),
            ("bayes", "bayes_target_allocator_share"),
            ("rules", "rules_target_allocator_share"),
            ("pre_te_r5p", "post_m38_entitlement_tgt_share"),
            ("final_specialist", "final_target_allocator_share"),
        ]
    return []


def _long_stage_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in rows.iterrows():
        actual = float(r["actual_player_share"])
        for stage, col in _stage_columns(r):
            value = pd.to_numeric(pd.Series([r.get(col)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            rec = {
                "season": int(r["season"]), "week": int(r["week"]),
                "event_id": r["event_id"], "team": r["team"], "opponent": r["opponent"],
                "player": r["player"], "player_clean_key": r["player_clean_key"],
                "position_family": r["position_family"], "opportunity_type": r["opportunity_type"],
                "stage": stage, "predicted_share": float(value), "actual_share": actual,
                "share_error": float(value - actual), "absolute_share_error": abs(float(value - actual)),
                "actual_player_opportunity": float(r["actual_player_opportunity"]),
                "actual_opportunity_bin": r["actual_opportunity_bin"],
                "bayes_evidence_state": r.get("bayes_evidence_state", ""),
                "prior_games": r.get("prior_games", np.nan),
                "current_games": r.get("current_games", np.nan),
                "specialist_evidence_consumed": bool(r.get("specialist_evidence_consumed", False)),
                "specialist_prior1_anyteam_available": r.get("specialist_prior1_anyteam_available", np.nan),
                "specialist_prior3_anyteam_available": r.get("specialist_prior3_anyteam_available", np.nan),
                "specialist_prior1_same_team_available": r.get("specialist_prior1_same_team_available", np.nan),
                "specialist_prior3_same_team_available": r.get("specialist_prior3_same_team_available", np.nan),
            }
            out.append(rec)
    return pd.DataFrame(out)


def _score(g: pd.DataFrame) -> dict:
    e = pd.to_numeric(g["share_error"], errors="coerce")
    p = pd.to_numeric(g["predicted_share"], errors="coerce")
    a = pd.to_numeric(g["actual_share"], errors="coerce")
    return {
        "rows": int(len(g)),
        "share_mae": float(e.abs().mean()),
        "share_bias": float(e.mean()),
        "share_rmse": float(np.sqrt(np.mean(np.square(e)))),
        "pred_actual_pearson": float(p.corr(a)) if len(g) > 2 and p.nunique() > 1 and a.nunique() > 1 else np.nan,
        "mean_predicted_share": float(p.mean()),
        "mean_actual_share": float(a.mean()),
    }


def _stage_summary(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pos, opp, stage), g in long.groupby(["position_family", "opportunity_type", "stage"], sort=False):
        rows.append({"position_family": pos, "opportunity_type": opp, "stage": stage, "subset": "ALL", **_score(g)})
        active = g.loc[pd.to_numeric(g["actual_player_opportunity"], errors="coerce").gt(0)]
        rows.append({"position_family": pos, "opportunity_type": opp, "stage": stage, "subset": "ACTIVE", **_score(active)})
        label = HIGH_BIN.get((str(pos), str(opp)))
        if label:
            high = g.loc[g["actual_opportunity_bin"].astype(str).eq(label)]
            if not high.empty:
                rows.append({"position_family": pos, "opportunity_type": opp, "stage": stage, "subset": "HIGH_WORKLOAD", **_score(high)})
    return pd.DataFrame(rows)


def _coverage_summary(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    dimensions: list[tuple[str, str]] = [
        ("bayes_evidence_state", "bayes_evidence_state"),
        ("current_games", "current_games"),
    ]
    for col in [
        "specialist_prior1_anyteam_available", "specialist_prior3_anyteam_available",
        "specialist_prior1_same_team_available", "specialist_prior3_same_team_available",
    ]:
        dimensions.append((col, col))

    for (pos, opp), g in rows.groupby(["position_family", "opportunity_type"], sort=False):
        final_col = "final_rush_allocator_share" if opp == "carries" else "final_target_allocator_share"
        q = g.copy()
        q["share_error"] = pd.to_numeric(q[final_col], errors="coerce") - pd.to_numeric(q["actual_player_share"], errors="coerce")
        q["predicted_share"] = pd.to_numeric(q[final_col], errors="coerce")
        q["actual_share"] = pd.to_numeric(q["actual_player_share"], errors="coerce")
        for dim, col in dimensions:
            if col not in q.columns:
                continue
            vals = q[col]
            if vals.notna().sum() == 0:
                continue
            for value, sub in q.loc[vals.notna()].groupby(col, dropna=False):
                rec = {
                    "position_family": pos, "opportunity_type": opp,
                    "evidence_dimension": dim, "evidence_value": str(value),
                    "subset": "ALL", **_score(sub),
                    "actual_zero_rate": float(pd.to_numeric(sub["actual_player_opportunity"], errors="coerce").eq(0).mean()),
                }
                out.append(rec)
                active = sub.loc[pd.to_numeric(sub["actual_player_opportunity"], errors="coerce").gt(0)]
                if not active.empty:
                    out.append({
                        "position_family": pos, "opportunity_type": opp,
                        "evidence_dimension": dim, "evidence_value": str(value),
                        "subset": "ACTIVE", **_score(active), "actual_zero_rate": 0.0,
                    })
                label = HIGH_BIN.get((str(pos), str(opp)))
                if label:
                    high = sub.loc[sub["actual_opportunity_bin"].astype(str).eq(label)]
                    if not high.empty:
                        out.append({
                            "position_family": pos, "opportunity_type": opp,
                            "evidence_dimension": dim, "evidence_value": str(value),
                            "subset": "HIGH_WORKLOAD", **_score(high),
                            "actual_zero_rate": 0.0,
                        })
    return pd.DataFrame(out)


def _specialist_effect(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pos in ("WR", "TE"):
        g = rows.loc[rows["position_family"].eq(pos) & rows["opportunity_type"].eq("targets")].copy()
        if g.empty:
            continue
        actual = pd.to_numeric(g["actual_player_share"], errors="coerce")
        if pos == "WR":
            stages = [
                ("rules", "rules_target_allocator_share"),
                ("post_m38", "post_m38_entitlement_tgt_share"),
                ("final", "final_target_allocator_share"),
            ]
            groups = [("ALL", pd.Series(True, index=g.index))]
            if "wr_r15_anchor" in g.columns:
                groups += [
                    ("WR1_ANCHOR", g["wr_r15_anchor"].fillna(False).astype(bool)),
                    ("WR2PLUS", ~g["wr_r15_anchor"].fillna(False).astype(bool)),
                ]
        else:
            stages = [
                ("rules", "rules_target_allocator_share"),
                ("pre_te_r5p", "post_m38_entitlement_tgt_share"),
                ("final", "final_target_allocator_share"),
            ]
            groups = [("ALL", pd.Series(True, index=g.index))]
        for scope, mask in groups:
            h = g.loc[mask].copy()
            if h.empty:
                continue
            a = pd.to_numeric(h["actual_player_share"], errors="coerce")
            rec = {"position_family": pos, "scope": scope, "rows": int(len(h))}
            for name, col in stages:
                pred = pd.to_numeric(h[col], errors="coerce")
                rec[f"{name}_mae"] = float((pred-a).abs().mean())
                rec[f"{name}_bias"] = float((pred-a).mean())
            rec["rules_to_final_mae_improvement"] = rec["rules_mae"] - rec["final_mae"]
            middle = stages[1][0]
            rec[f"{middle}_to_final_mae_improvement"] = rec[f"{middle}_mae"] - rec["final_mae"]
            out.append(rec)
    return pd.DataFrame(out)


def run(
    *,
    player_logs_path: Path,
    team_weekly_path: Path,
    schedule_path: Path,
    universe_dir: Path,
    parent_path: Path,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    player_logs = _read(player_logs_path, "player logs")
    team_weekly = _read(team_weekly_path, "team weekly")
    schedule = _read(schedule_path, "schedule")

    # Freeze complete pregame traces for all four weeks before reading outcomes.
    traces = []
    audits = {}
    for week in WEEKS:
        universe = _read(universe_dir / f"{SEASON}_week_{week:02d}.csv", f"ACT-only W{week} universe")
        trace, audit = _build_week_stage_trace(
            player_logs=player_logs,
            team_weekly=team_weekly,
            schedule=schedule,
            universe=universe,
            week=week,
        )
        traces.append(trace)
        audits[str(week)] = audit
        print(f"[player-share-input-audit] froze W{week} pregame players={len(trace)}")

    stage = pd.concat(traces, ignore_index=True, sort=False)
    if stage["sportsbook_inputs_used_upstream"].astype(bool).any():
        raise RuntimeError("sportsbook input entered pregame share trace")
    if stage["rb_week5_room_allocation_shadow_applied"].astype(bool).any():
        raise RuntimeError("Week-5 RB room shadow entered W1-4 share trace")
    if not stage["target_share_trajectory_status"].eq("FROZEN_FEATURE_NOT_YET_ELIGIBLE").all():
        raise RuntimeError("target-share trajectory illegally entered W1-4 trace")

    rows = _attach_parent(stage, parent_path)

    # Exact parity against frozen parent final allocator probability.
    final = np.where(
        rows["opportunity_type"].eq("carries"),
        pd.to_numeric(rows["final_rush_allocator_share"], errors="coerce"),
        pd.to_numeric(rows["final_target_allocator_share"], errors="coerce"),
    )
    parent_final = pd.to_numeric(rows["predicted_player_share"], errors="coerce")
    parity = np.abs(final - parent_final)
    if float(np.nanmax(parity)) > TOL:
        raise RuntimeError(f"final allocator share does not reproduce frozen parent max_gap={float(np.nanmax(parity))}")

    long = _long_stage_rows(rows)
    stage_summary = _stage_summary(long)
    coverage = _coverage_summary(rows)
    specialist = _specialist_effect(rows)

    rows.to_csv(out_dir / "player_share_stage_rows.csv", index=False)
    stage_summary.to_csv(out_dir / "player_share_stage_error_summary.csv", index=False)
    coverage.to_csv(out_dir / "player_share_evidence_coverage_summary.csv", index=False)
    specialist.to_csv(out_dir / "player_share_specialist_effect_summary.csv", index=False)

    payload = {
        "version": "PLAYER_SHARE_INPUT_COVERAGE_AND_RESIDUAL_AUDIT_V1",
        "season": SEASON,
        "weeks": list(WEEKS),
        "rows": int(len(rows)),
        "unique_player_weeks": int(rows[["week","team","player_clean_key"]].drop_duplicates().shape[0]),
        "final_parent_share_max_abs_gap": float(np.nanmax(parity)),
        "stage_summary": stage_summary.to_dict("records"),
        "specialist_effect_summary": specialist.to_dict("records"),
        "parameters_fit": 0,
        "automatic_promotion": False,
        "sportsbook_inputs_used_upstream": False,
        "paid_odds_api_used": False,
        "target_share_trajectory_applied_rows": 0,
        "rb_week5_room_shadow_rows": 0,
        "outcomes_loaded_after_pregame_trace_freeze": True,
        "weekly_specialist_audits": audits,
        "disposition": "AUDIT_COMPLETE_RAW_RESULT_REQUIRES_INTERPRETATION",
    }
    (out_dir / "player_share_input_coverage_residual_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--universe-dir", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    run(
        player_logs_path=a.player_logs,
        team_weekly_path=a.team_weekly,
        schedule_path=a.schedule,
        universe_dir=a.universe_dir,
        parent_path=a.parent,
        out_dir=a.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
