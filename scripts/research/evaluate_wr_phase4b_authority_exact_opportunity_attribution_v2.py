#!/usr/bin/env python3
"""WR Phase 4B authority-exact opportunity attribution, static-identity/PBP source v2.

Amends only the retrospective actual-target identity/source mechanics from V1.
Static alias metadata may use all audited 2022-2024 roster evidence; actual
receiver targets remain nflverse PBP-by-GSIS only. Predictive temporal rules do
not change. Ambiguity fails closed. No sportsbook inputs, fitting, or production
changes.

The --preflight-only path never reads receiving-yard projection/outcome columns.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.research import evaluate_wr_phase4b_authority_exact_opportunity_attribution_v1 as base
from scripts.research import audit_wr_phase4b_prior_roster_gsis_alias_v2 as ids
from scripts.research import audit_wr_phase4b_identity_temporal_ablation_v3 as v3
from scripts.utils.player_identity_v3 import clean_player_id

TG = base.TG
IDENT = base.IDENT
TOL = base.TOL
WR_POS = base.WR_POS


@dataclass
class TargetAuthority:
    aliases: pd.DataFrame
    full_index: dict
    base_index: dict
    target_counts: pd.DataFrame
    team_games: pd.DataFrame
    roster_positions: pd.DataFrame

    def resolve(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "player" not in frame.columns:
            raise ValueError("static identity resolution requires player display alias")
        return v3.resolve_static_frame(
            frame.copy(), self.full_index, self.base_index,
            self.target_counts, self.team_games,
        )


def _first(frame: pd.DataFrame, names: Iterable[str], default="") -> pd.Series:
    for c in names:
        if c in frame.columns:
            return frame[c]
    return pd.Series(default, index=frame.index)


def _to_pd(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _load_roster_positions(seasons=(2023, 2024)) -> pd.DataFrame:
    """Target-week roster position metadata for retrospective off-model disclosure only."""
    import nflreadpy as nfl

    frames = []
    for season in seasons:
        raw = _to_pd(nfl.load_rosters_weekly(int(season)))
        x = raw.copy()
        x.columns = [str(c).strip().lower() for c in x.columns]
        x["season"] = pd.to_numeric(_first(x, ["season"], season), errors="coerce").fillna(season).astype(int)
        x["week"] = pd.to_numeric(_first(x, ["week"]), errors="coerce")
        x["team"] = _first(x, ["team", "team_abbr", "club_code"]).map(canon_team)
        x["receiver_id"] = _first(x, ["gsis_id", "player_id"]).map(clean_player_id)
        x["position"] = _first(x, ["position", "position_group", "depth_chart_position"]).astype("string").fillna("").str.upper().str.strip()
        x = x.loc[
            x["season"].eq(int(season))
            & x["week"].between(1, 18, inclusive="both")
            & x["team"].astype(str).str.len().gt(0)
            & x["receiver_id"].astype(str).str.len().gt(0)
        , ["season", "week", "team", "receiver_id", "position"]].copy()
        x["week"] = x["week"].astype(int)
        frames.append(x)
    out = pd.concat(frames, ignore_index=True)

    def choose_pos(s: pd.Series) -> str:
        vals = [str(v).strip().upper() for v in s if str(v).strip()]
        wr = [v for v in vals if v in WR_POS or v == "WR"]
        return wr[0] if wr else (vals[0] if vals else "")

    return (
        out.groupby(TG + ["receiver_id"], as_index=False)["position"]
        .agg(choose_pos)
    )


def load_target_authority() -> TargetAuthority:
    aliases = ids.load_roster_aliases([2022, 2023, 2024])
    full_index = ids._index_aliases(aliases, "full_key")
    base_index = ids._index_aliases(aliases, "base_key")
    target_counts, team_games = ids.load_pbp_targets([2023, 2024])
    roster_positions = _load_roster_positions((2023, 2024))
    return TargetAuthority(
        aliases=aliases,
        full_index=full_index,
        base_index=base_index,
        target_counts=target_counts,
        team_games=team_games,
        roster_positions=roster_positions,
    )


def _parity_summary(cand: pd.DataFrame) -> dict:
    resolved = cand.loc[cand["identity_status"].eq("RESOLVED_GSIS") & cand["pbp_targets"].notna()].copy()
    resolved["target_delta"] = (
        pd.to_numeric(resolved["pbp_targets"], errors="raise")
        - pd.to_numeric(resolved["actual_targets"], errors="raise")
    )
    parity = resolved["target_delta"].abs().le(1e-12)
    return {
        "rows": int(len(cand)),
        "resolved_rows": int(len(resolved)),
        "exact_target_parity_rows": int(parity.sum()),
        "target_parity_fail_rows": int((~parity).sum()),
        "max_abs_target_delta": float(resolved["target_delta"].abs().max()) if len(resolved) else np.nan,
    }


def _overlap_reassignment_audit(frame: pd.DataFrame, ctx: TargetAuthority) -> dict:
    strict = ids.resolve_frame(frame.copy(), ctx.full_index, ctx.base_index, ctx.target_counts, ctx.team_games)
    static = ctx.resolve(frame.copy())
    a = strict[IDENT + ["identity_status", "resolved_player_id", "pbp_targets"]].rename(columns={
        "identity_status": "strict_status", "resolved_player_id": "strict_id", "pbp_targets": "strict_targets"
    })
    b = static[IDENT + ["identity_status", "resolved_player_id", "pbp_targets"]].rename(columns={
        "identity_status": "static_status", "resolved_player_id": "static_id", "pbp_targets": "static_targets"
    })
    x = a.merge(b, on=IDENT, how="inner", validate="one_to_one")
    both = x.loc[x["strict_status"].eq("RESOLVED_GSIS") & x["static_status"].eq("RESOLVED_GSIS")].copy()
    id_flip = both["strict_id"].astype(str).ne(both["static_id"].astype(str))
    target_flip = ~np.isclose(
        pd.to_numeric(both["strict_targets"], errors="raise"),
        pd.to_numeric(both["static_targets"], errors="raise"),
        atol=0.0, rtol=0.0,
    )
    return {
        "both_resolved_rows": int(len(both)),
        "gsis_reassignment_rows": int(id_flip.sum()),
        "target_change_rows": int(target_flip.sum()),
    }


def _canonical_inputs(pred: pd.DataFrame, feat: pd.DataFrame):
    anchors = pred.loc[pred["wr_rank"].eq(1), IDENT + ["player", "pred_targets"]].copy()
    anchor_games = anchors[TG].drop_duplicates()
    sec = feat.merge(anchor_games, on=TG, how="inner", validate="many_to_one").copy()
    canonical = pd.concat(
        [anchors[IDENT + ["player"]], sec[IDENT + ["player"]]],
        ignore_index=True,
    ).drop_duplicates(IDENT)
    return anchors, sec, canonical


def mechanics_preflight(predictions: Path, features: Path, conservation: Path, out_dir: Path) -> dict:
    pred, feat, cons, pools = base._load_structures(predictions, features, conservation, include_yards=False)
    ctx = load_target_authority()
    anchors, sec, canonical = _canonical_inputs(pred, feat)

    l4 = ctx.resolve(feat[IDENT + ["player"]])
    l23 = ctx.resolve(canonical)
    cand = ctx.resolve(pred[IDENT + ["player", "actual_targets", "wr_rank"]])
    parity = _parity_summary(cand)
    if parity["resolved_rows"] != base.EXPECTED_CANDIDATE_ROWS or parity["target_parity_fail_rows"] != 0:
        raise RuntimeError(f"R15 authority PBP target parity failed: {parity}")

    l4_summary = ids._summary(l4)
    l23_summary = ids._summary(l23)
    if l4_summary["resolved_gsis"] != 5320 or l4_summary["ambiguous"] != 1 or l4_summary["unresolved"] != 0:
        raise RuntimeError(f"Layer4 V3 coverage drift: {l4_summary}")
    if l23_summary["resolved_gsis"] != 6011 or l23_summary["ambiguous"] != 1 or l23_summary["unresolved"] != 0:
        raise RuntimeError(f"Layer23 V3 coverage drift: {l23_summary}")

    reassignment = {
        "layer4": _overlap_reassignment_audit(feat[IDENT + ["player"]], ctx),
        "layer23": _overlap_reassignment_audit(canonical, ctx),
        "candidate": _overlap_reassignment_audit(pred[IDENT + ["player", "actual_targets"]], ctx),
    }
    if any(v["gsis_reassignment_rows"] or v["target_change_rows"] for v in reassignment.values()):
        raise RuntimeError(f"V3 changed a prior resolved identity/target: {reassignment}")

    ambiguous_l4 = l4.loc[~l4["identity_status"].eq("RESOLVED_GSIS")].copy()
    ambiguous_l23 = l23.loc[~l23["identity_status"].eq("RESOLVED_GSIS")].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    ambiguous_l4.to_csv(out_dir / "phase4b_preflight_layer4_ambiguous.csv", index=False)
    ambiguous_l23.to_csv(out_dir / "phase4b_preflight_layer23_ambiguous.csv", index=False)
    result = {
        "specification": "WR_PHASE4B_AUTHORITY_EXACT_OPPORTUNITY_ATTRIBUTION_V2_PREFLIGHT",
        "source_contract": "static_roster_alias_to_gsis__pbp_targets_only",
        "candidate_rows": int(len(pred)),
        "authority_team_games": int(len(pred[TG].drop_duplicates())),
        "anchor_observable_team_games": int(len(anchors)),
        "canonical_wr2plus_feature_rows": int(len(feat)),
        "layer4_identity": l4_summary,
        "layer23_identity": l23_summary,
        "candidate_target_parity": parity,
        "v2_v3_overlap_nonreassignment": reassignment,
        "implied_pool_team_games": int(len(pools)),
        "conservation_team_games": int(len(cons)),
        "receiving_yard_fields_loaded": False,
        "attribution_outcomes_run": False,
        "predictive_feature_temporal_rules_changed": False,
        "fuzzy_matching": False,
        "sportsbook_inputs": 0,
        "challenger_model_authorized": False,
        "production_change": False,
    }
    (out_dir / "phase4b_preflight_v2.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    return result


def _valid_layer23_source(pred: pd.DataFrame, feat: pd.DataFrame, ctx: TargetAuthority):
    anchors, sec, canonical = _canonical_inputs(pred, feat)
    anchor_res = ctx.resolve(anchors)
    sec_res = ctx.resolve(sec[IDENT + ["player"]])
    bad = pd.concat([
        anchor_res.loc[~anchor_res["identity_status"].eq("RESOLVED_GSIS"), TG],
        sec_res.loc[~sec_res["identity_status"].eq("RESOLVED_GSIS"), TG],
    ], ignore_index=True).drop_duplicates()
    all_anchor = anchors[TG].drop_duplicates()
    valid = all_anchor.merge(bad.assign(_bad=1), on=TG, how="left")
    valid = valid.loc[valid["_bad"].isna(), TG]
    anchor_res = anchor_res.merge(valid, on=TG, how="inner", validate="many_to_one")
    sec_res = sec_res.merge(valid, on=TG, how="inner", validate="many_to_one")
    return anchors, sec, canonical, anchor_res, sec_res, valid, bad


def _off_model_wr_disclosure(ctx: TargetAuthority, canonical_resolved: pd.DataFrame, valid_games: pd.DataFrame):
    canon = canonical_resolved.loc[canonical_resolved["identity_status"].eq("RESOLVED_GSIS"), TG + ["resolved_player_id"]].drop_duplicates()
    tc = ctx.target_counts.merge(valid_games, on=TG, how="inner", validate="many_to_one")
    tc = tc.merge(ctx.roster_positions, on=TG + ["receiver_id"], how="left", validate="many_to_one")
    tc = tc.merge(
        canon.rename(columns={"resolved_player_id": "receiver_id"}).assign(_canonical=1),
        on=TG + ["receiver_id"], how="left", validate="one_to_one",
    )
    is_wr = tc["position"].astype("string").fillna("").str.upper().isin(WR_POS | {"WR"})
    off = tc.loc[is_wr & tc["_canonical"].isna()].copy()
    off_tg = off.groupby(TG, as_index=False)["pbp_targets"].sum().rename(columns={"pbp_targets": "off_model_wr_targets"})
    sf = valid_games.merge(off_tg, on=TG, how="left", validate="one_to_one")
    sf["off_model_wr_targets"] = sf["off_model_wr_targets"].fillna(0.0)
    unclassified = tc.loc[tc["position"].astype("string").fillna("").eq("")].copy()
    summary = {
        "team_games": int(len(sf)),
        "mean": float(sf["off_model_wr_targets"].mean()),
        "median": float(sf["off_model_wr_targets"].median()),
        "p90": float(sf["off_model_wr_targets"].quantile(0.90)),
        "max": float(sf["off_model_wr_targets"].max()),
        "total": float(sf["off_model_wr_targets"].sum()),
        "team_games_with_any": int(sf["off_model_wr_targets"].gt(0).sum()),
        "share_team_games_with_any": float(sf["off_model_wr_targets"].gt(0).mean()),
        "unclassified_target_rows": int(len(unclassified)),
        "unclassified_target_mass": float(unclassified["pbp_targets"].sum()) if len(unclassified) else 0.0,
    }
    return off, summary


def _layer2_3(pred: pd.DataFrame, feat: pd.DataFrame, cons: pd.DataFrame, pools: pd.DataFrame, ctx: TargetAuthority):
    anchors, sec, canonical, anchor_res, sec_res, valid_games, bad_games = _valid_layer23_source(pred, feat, ctx)
    actual_secondary = sec_res.groupby(TG, as_index=False)["pbp_targets"].sum().rename(columns={"pbp_targets": "actual_secondary_targets"})
    actual_anchor = anchor_res[TG + ["pred_targets", "pbp_targets", "player_clean_key"]].rename(columns={
        "pred_targets": "pred_wr1_targets", "pbp_targets": "actual_wr1_targets", "player_clean_key": "wr1_player_clean_key"
    })
    team_actual = ctx.target_counts.groupby(TG, as_index=False)["pbp_targets"].sum().rename(columns={"pbp_targets": "actual_team_targets"})

    df = valid_games.merge(pools, on=TG, how="left", validate="one_to_one")
    df = df.merge(cons[TG + ["candidate_wr_room_mass"]], on=TG, how="left", validate="one_to_one")
    df = df.merge(actual_anchor, on=TG, how="left", validate="one_to_one")
    df = df.merge(actual_secondary, on=TG, how="left", validate="one_to_one")
    df = df.merge(team_actual, on=TG, how="left", validate="one_to_one")
    if df.isna().any().any():
        raise RuntimeError("Layer2/3 static-GSIS/PBP assembly incomplete")

    df["pred_wr_room_targets"] = df["implied_team_target_pool"] * df["candidate_wr_room_mass"]
    df["actual_wr_room_targets"] = df["actual_wr1_targets"] + df["actual_secondary_targets"]
    if (df["actual_team_targets"] <= 0).any():
        raise RuntimeError("non-positive PBP actual target pool")
    df["pred_wr_share"] = df["candidate_wr_room_mass"]
    df["actual_wr_share"] = df["actual_wr_room_targets"] / df["actual_team_targets"]
    comps = [
        base.symmetric_product_decomposition(pm, ps, am, ash)
        for pm, ps, am, ash in zip(df["implied_team_target_pool"], df["pred_wr_share"], df["actual_team_targets"], df["actual_wr_share"])
    ]
    df["team_pool_component"] = [x[0] for x in comps]
    df["wr_room_share_component"] = [x[1] for x in comps]
    resid = df["actual_wr_room_targets"] - df["pred_wr_room_targets"]
    if not np.allclose(df["team_pool_component"] + df["wr_room_share_component"], resid, atol=TOL, rtol=1e-10):
        raise RuntimeError("Layer2 target decomposition identity failed")
    df["pred_secondary_targets"] = df["pred_wr_room_targets"] - df["pred_wr1_targets"]

    canonical_res = pd.concat([anchor_res, sec_res], ignore_index=True, sort=False)
    off, off_summary = _off_model_wr_disclosure(ctx, canonical_res, valid_games)

    all_games = pred[TG].drop_duplicates()
    anchor_games = anchors[TG].drop_duplicates()
    cov = all_games.merge(anchor_games.assign(anchor_observable=True), on=TG, how="left")
    cov["anchor_observable"] = cov["anchor_observable"].fillna(False).astype(bool)
    cov = cov.merge(valid_games.assign(complete_static_identity_room=True), on=TG, how="left")
    cov["complete_static_identity_room"] = cov["complete_static_identity_room"].fillna(False).astype(bool)

    summaries = []
    for label, g in [("POOLED", df), ("2023", df.loc[df["season"].eq(2023)]), ("2024", df.loc[df["season"].eq(2024)])]:
        summaries.append({
            "slice": label, "n": int(len(g)),
            "room_mae": float((g["pred_wr_room_targets"] - g["actual_wr_room_targets"]).abs().mean()),
            "room_bias_pred_minus_actual": float((g["pred_wr_room_targets"] - g["actual_wr_room_targets"]).mean()),
            "mean_abs_team_pool_component": float(g["team_pool_component"].abs().mean()),
            "median_abs_team_pool_component": float(g["team_pool_component"].abs().median()),
            "mean_abs_wr_room_share_component": float(g["wr_room_share_component"].abs().mean()),
            "median_abs_wr_room_share_component": float(g["wr_room_share_component"].abs().median()),
            "share_abs_team_pool_gt_share_component": float((g["team_pool_component"].abs() > g["wr_room_share_component"].abs()).mean()),
        })
        summaries.append({"slice": label, "role": "WR1", **base._metric_summary(g["actual_wr1_targets"], g["pred_wr1_targets"])})
        summaries.append({"slice": label, "role": "WR2PLUS_POOL", **base._metric_summary(g["actual_secondary_targets"], g["pred_secondary_targets"])})

    coverage_summary = {
        "authority_team_games": int(len(all_games)),
        "anchor_observable_team_games": int(len(anchor_games)),
        "complete_static_identity_team_games": int(len(valid_games)),
        "excluded_for_identity_ambiguity_team_games": int(len(bad_games)),
        "excluded_identity_games": bad_games.to_dict("records"),
        "included_season_counts": {str(int(k)): int(v) for k, v in df.groupby("season").size().to_dict().items()},
        "excluded_anchor_absent_season_counts": {
            str(int(k)): int(v) for k, v in cov.loc[~cov["anchor_observable"]].groupby("season").size().to_dict().items()
        },
        "included_week_mean": float(df["week"].mean()),
        "included_week_median": float(df["week"].median()),
        "included_week_min": int(df["week"].min()),
        "included_week_max": int(df["week"].max()),
    }
    return df, off, cov, off_summary, summaries, coverage_summary


def _layer4(feat: pd.DataFrame, pools: pd.DataFrame, ctx: TargetAuthority):
    l4 = feat.merge(pools, on=TG, how="left", validate="many_to_one")
    if l4["implied_team_target_pool"].isna().any():
        raise RuntimeError("Layer4 missing implied team target pool")
    l4["baseline_pred_targets"] = l4["baseline_entitlement_tgt_share"] * l4["implied_team_target_pool"]
    l4["r15_pred_targets"] = l4["candidate_entitlement_tgt_share"] * l4["implied_team_target_pool"]
    l4 = ctx.resolve(l4)
    unresolved = l4.loc[~l4["identity_status"].eq("RESOLVED_GSIS")].copy()
    resolved = l4.loc[l4["identity_status"].eq("RESOLVED_GSIS") & l4["pbp_targets"].notna()].copy()
    if len(resolved) != 5320 or len(unresolved) != 1:
        raise RuntimeError(f"Layer4 static identity coverage drift resolved={len(resolved)} unresolved={len(unresolved)}")
    resolved["baseline_abs_error"] = (resolved["baseline_pred_targets"] - resolved["pbp_targets"]).abs()
    resolved["r15_abs_error"] = (resolved["r15_pred_targets"] - resolved["pbp_targets"]).abs()
    resolved["paired_mae_delta"] = resolved["r15_abs_error"] - resolved["baseline_abs_error"]

    metrics, deltas = [], {}
    for label, g in [("POOLED", resolved), ("2023", resolved.loc[resolved["season"].eq(2023)]), ("2024", resolved.loc[resolved["season"].eq(2024)])]:
        b = base._metric_summary(g["pbp_targets"], g["baseline_pred_targets"])
        r = base._metric_summary(g["pbp_targets"], g["r15_pred_targets"])
        delta = float(r["mae"] - b["mae"])
        deltas[label] = delta
        ci = base.cluster_bootstrap_mean_ci(g, "paired_mae_delta", reps=base.BOOT_REPS, seed=base.BOOT_SEED, stratify_season=(label == "POOLED"))
        toward = g["r15_abs_error"] < g["baseline_abs_error"]
        away = g["r15_abs_error"] > g["baseline_abs_error"]
        metrics.append({
            "slice": label, "resolved_n": int(len(g)),
            "baseline_target_mae": b["mae"], "r15_target_mae": r["mae"],
            "target_mae_delta_r15_minus_baseline": delta,
            "baseline_rmse": b["rmse"], "r15_rmse": r["rmse"],
            "baseline_bias_pred_minus_actual": b["bias_pred_minus_actual"],
            "r15_bias_pred_minus_actual": r["bias_pred_minus_actual"],
            "r15_toward_actual_n": int(toward.sum()), "r15_away_from_actual_n": int(away.sum()),
            "r15_equal_error_n": int(len(g) - toward.sum() - away.sum()),
            "r15_toward_actual_share": float(toward.mean()),
            "paired_mae_delta_ci_low": ci["ci_low"], "paired_mae_delta_ci_high": ci["ci_high"],
            "bootstrap_clusters": ci["clusters"], "bootstrap_reps": ci["reps"], "bootstrap_seed": ci["seed"],
        })

    expected = l4.groupby(TG)["player_clean_key"].nunique().rename("expected_n")
    got = resolved.groupby(TG)["player_clean_key"].nunique().rename("resolved_n")
    fg = pd.concat([expected, got], axis=1).fillna(0)
    full_keys = fg.loc[fg["expected_n"].eq(fg["resolved_n"])].reset_index()[TG]
    sh = resolved.merge(full_keys, on=TG, how="inner", validate="many_to_one")
    sh["actual_secondary_targets"] = sh.groupby(TG)["pbp_targets"].transform("sum")
    sh = sh.loc[sh["actual_secondary_targets"].gt(0)].copy()
    sh["actual_secondary_share"] = sh["pbp_targets"] / sh["actual_secondary_targets"]
    sh["baseline_share_den"] = sh.groupby(TG)["baseline_entitlement_tgt_share"].transform("sum")
    sh["r15_share_den"] = sh.groupby(TG)["candidate_entitlement_tgt_share"].transform("sum")
    sh["baseline_secondary_share"] = sh["baseline_entitlement_tgt_share"] / sh["baseline_share_den"]
    sh["r15_secondary_share"] = sh["candidate_entitlement_tgt_share"] / sh["r15_share_den"]
    share_diag = {
        "fully_resolved_positive_secondary_team_games": int(len(sh[TG].drop_duplicates())),
        "rows": int(len(sh)),
        "baseline_share_mae": float((sh["baseline_secondary_share"] - sh["actual_secondary_share"]).abs().mean()) if len(sh) else np.nan,
        "r15_share_mae": float((sh["r15_secondary_share"] - sh["actual_secondary_share"]).abs().mean()) if len(sh) else np.nan,
    }
    disp = base.r15_disposition(deltas["POOLED"], deltas["2023"], deltas["2024"])
    coverage = {
        "feature_rows": int(len(l4)), "resolved_gsis_rows": int(len(resolved)),
        "ambiguous_or_unresolved_rows": int(len(unresolved)),
        "unresolved_pct": float(len(unresolved) / len(l4)),
        "identity_status_counts": {str(k): int(v) for k, v in l4["identity_status"].value_counts().to_dict().items()},
        "by_season": {
            str(int(s)): {
                "feature_rows": int(len(g)),
                "resolved_gsis_rows": int(g["identity_status"].eq("RESOLVED_GSIS").sum()),
                "ambiguous_or_unresolved_rows": int((~g["identity_status"].eq("RESOLVED_GSIS")).sum()),
            } for s, g in l4.groupby("season")
        },
        "disposition": disp, "share_diagnostic": share_diag,
    }
    return resolved, unresolved, metrics, coverage


def run_attribution(predictions: Path, features: Path, conservation: Path, out_dir: Path) -> dict:
    pred, feat, cons, pools = base._load_structures(predictions, features, conservation, include_yards=True)
    ctx = load_target_authority()
    cand = ctx.resolve(pred[IDENT + ["player", "actual_targets", "wr_rank"]])
    parity = _parity_summary(cand)
    if parity["resolved_rows"] != base.EXPECTED_CANDIDATE_ROWS or parity["target_parity_fail_rows"] != 0:
        raise RuntimeError(f"authority target parity failed before attribution: {parity}")

    l1_detail, l1_summary = base._layer1(pred)
    l23_detail, off_detail, anchor_cov, off_summary, l23_summary, l23_cov_summary = _layer2_3(pred, feat, cons, pools, ctx)
    l4_resolved, l4_unresolved, l4_metrics, l4_coverage = _layer4(feat, pools, ctx)

    out_dir.mkdir(parents=True, exist_ok=True)
    l1_detail.to_csv(out_dir / "phase4b_layer1_yard_decomposition.csv", index=False)
    pd.DataFrame(l1_summary).to_csv(out_dir / "phase4b_layer1_summary.csv", index=False)
    l23_detail.to_csv(out_dir / "phase4b_layer2_3_team_game_detail.csv", index=False)
    off_detail.to_csv(out_dir / "phase4b_off_model_wr_targets.csv", index=False)
    anchor_cov.to_csv(out_dir / "phase4b_anchor_observability.csv", index=False)
    pd.DataFrame(l23_summary).to_csv(out_dir / "phase4b_layer2_3_summary.csv", index=False)
    l4_resolved.to_csv(out_dir / "phase4b_layer4_resolved.csv", index=False)
    l4_unresolved.to_csv(out_dir / "phase4b_layer4_unresolved.csv", index=False)
    pd.DataFrame(l4_metrics).to_csv(out_dir / "phase4b_layer4_metrics.csv", index=False)

    result = {
        "specification": "WR_PHASE4B_AUTHORITY_EXACT_OPPORTUNITY_ATTRIBUTION_V2",
        "source_contract": "static_roster_alias_to_gsis__pbp_targets_only",
        "candidate_target_parity": parity,
        "r15_layer4_disposition": l4_coverage["disposition"],
        "layer4_coverage": l4_coverage,
        "layer23_coverage": l23_cov_summary,
        "off_model_wr_target_mass": off_summary,
        "layer1_rows": int(len(l1_detail)),
        "layer23_team_games": int(len(l23_detail)),
        "predictive_feature_temporal_rules_changed": False,
        "fuzzy_matching": False,
        "sportsbook_inputs": 0,
        "challenger_model_authorized": False,
        "production_change": False,
        "attribution_outcomes_run": True,
    }
    (out_dir / "phase4b_result_v2.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--conservation", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    if args.preflight_only:
        mechanics_preflight(args.predictions, args.features, args.conservation, args.out_dir)
    else:
        run_attribution(args.predictions, args.features, args.conservation, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
