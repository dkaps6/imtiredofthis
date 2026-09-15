#!/usr/bin/env python3
"""WR Phase 4B authority-exact opportunity attribution.

The module contains the full frozen Phase-4B diagnostic implementation, but it
supports a --preflight-only mode that never reads receiving-yard projection or
outcome columns. The preflight mode is the only mode authorized before Claude's
code-level implementation review.

No sportsbook inputs, no model fitting, no production changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas

CANDIDATE = "WR_R15_WR1_ANCHORED_PARTICIPATION"
EXPECTED_CANDIDATE_ROWS = 4193
EXPECTED_CANDIDATE_BY_SEASON = {2023: 2076, 2024: 2117}
EXPECTED_TEAM_GAMES = 1088
EXPECTED_ANCHOR_GAMES = 1026
EXPECTED_FEATURE_ROWS = 5321
MATERIALITY = 0.05
BOOT_REPS = 10000
BOOT_SEED = 20260915
TOL = 1e-9
WR_POS = {"WR", "LWR", "RWR", "SWR"}
TG = ["season", "week", "team"]
IDENT = TG + ["player_clean_key"]


def _numeric(frame: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        frame[c] = pd.to_numeric(frame[c], errors="raise")


def symmetric_yard_decomposition(
    t_pred: float, t_actual: float, y_pred: float, y_actual: float, tol: float = TOL
) -> tuple[float, float]:
    """Exact symmetric Y=T*E decomposition with frozen zero-target handling."""
    tp, ta, yp, ya = map(float, (t_pred, t_actual, y_pred, y_actual))
    if tp < -tol or ta < -tol:
        raise ValueError("negative target count")
    tp = 0.0 if abs(tp) <= tol else tp
    ta = 0.0 if abs(ta) <= tol else ta

    if tp == 0.0 and abs(yp) > tol:
        raise ValueError("predicted yards nonzero with zero predicted targets")
    if ta == 0.0 and abs(ya) > tol:
        raise ValueError("actual yards nonzero with zero actual targets")

    if tp == 0.0 and ta == 0.0:
        return 0.0, 0.0
    if tp == 0.0:
        ea = ya / ta
        ep = ea
    elif ta == 0.0:
        ep = yp / tp
        ea = ep
    else:
        ep = yp / tp
        ea = ya / ta

    opportunity = (ta - tp) * (ea + ep) / 2.0
    efficiency = (ea - ep) * (ta + tp) / 2.0
    residual = ya - yp
    if not np.isclose(opportunity + efficiency, residual, atol=tol, rtol=1e-10):
        raise AssertionError(
            f"yard decomposition identity failed: opp={opportunity} eff={efficiency} "
            f"resid={residual}"
        )
    return float(opportunity), float(efficiency)


def symmetric_product_decomposition(
    pred_mass: float,
    pred_share: float,
    actual_mass: float,
    actual_share: float,
    tol: float = TOL,
) -> tuple[float, float]:
    """Exact symmetric decomposition of actual_mass*share - pred_mass*share."""
    pm, ps, am, a_s = map(float, (pred_mass, pred_share, actual_mass, actual_share))
    mass_component = (am - pm) * (a_s + ps) / 2.0
    share_component = (a_s - ps) * (am + pm) / 2.0
    lhs = mass_component + share_component
    rhs = am * a_s - pm * ps
    if not np.isclose(lhs, rhs, atol=tol, rtol=1e-10):
        raise AssertionError(
            f"product decomposition identity failed: mass={mass_component} "
            f"share={share_component} rhs={rhs}"
        )
    return float(mass_component), float(share_component)


def r15_disposition(delta_pooled: float, delta_2023: float, delta_2024: float) -> str:
    """Frozen R15 diagnostic disposition. Delta = R15 MAE - baseline MAE."""
    dp, d23, d24 = map(float, (delta_pooled, delta_2023, delta_2024))
    if dp <= -MATERIALITY and d23 < MATERIALITY and d24 < MATERIALITY:
        return "R15_WR2PLUS_ALLOCATION_HEALTHY_OR_IMPROVED"
    if dp >= MATERIALITY and d23 > 0.0 and d24 > 0.0:
        return "R15_WR2PLUS_ALLOCATION_STRUCTURED_ERROR"
    return "R15_WR2PLUS_ALLOCATION_MIXED_OR_SMALL"


def cluster_bootstrap_mean_ci(
    frame: pd.DataFrame,
    value_col: str,
    reps: int = BOOT_REPS,
    seed: int = BOOT_SEED,
    stratify_season: bool = True,
) -> dict:
    """Cluster bootstrap mean CI using team-games as frozen cluster unit."""
    if frame.empty:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "reps": int(reps)}
    needed = set(TG + [value_col])
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"bootstrap frame missing columns: {sorted(missing)}")
    x = frame[TG + [value_col]].copy()
    x[value_col] = pd.to_numeric(x[value_col], errors="raise").astype(float)
    grouped = x.groupby(TG, sort=True)[value_col].agg(["sum", "count"]).reset_index()
    observed = float(x[value_col].mean())
    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(reps), dtype=float)

    if stratify_season:
        strata = {}
        for season, g in grouped.groupby("season", sort=True):
            strata[int(season)] = (
                g["sum"].to_numpy(dtype=float),
                g["count"].to_numpy(dtype=float),
            )
        for b in range(int(reps)):
            total_sum = 0.0
            total_n = 0.0
            for _, (sums, counts) in strata.items():
                idx = rng.integers(0, len(sums), size=len(sums))
                total_sum += float(sums[idx].sum())
                total_n += float(counts[idx].sum())
            boots[b] = total_sum / total_n
    else:
        sums = grouped["sum"].to_numpy(dtype=float)
        counts = grouped["count"].to_numpy(dtype=float)
        for b in range(int(reps)):
            idx = rng.integers(0, len(sums), size=len(sums))
            boots[b] = float(sums[idx].sum() / counts[idx].sum())

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "reps": int(reps),
        "seed": int(seed),
        "cluster_unit": "season-week-team",
        "stratified_by_season": bool(stratify_season),
        "clusters": int(len(grouped)),
        "rows": int(len(x)),
    }


def _load_weekly_actuals(seasons=(2023, 2024)) -> pd.DataFrame:
    import nflreadpy as nfl

    frames = []
    for season in seasons:
        raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
        x = _normalize_weekly(_to_pandas(raw), int(season))
        x = x.loc[pd.to_numeric(x["week"], errors="coerce").between(1, 18)].copy()
        x["season"] = int(season)
        x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
        x["team"] = x["team"].map(canon_team)
        x["player_clean_key"] = x["player_clean_key"].astype(str)
        x["position"] = x["position"].astype("string").fillna("").str.upper().str.strip()
        x["targets"] = pd.to_numeric(x["targets"], errors="raise").astype(float)
        frames.append(
            x[["season", "week", "team", "player_clean_key", "player", "position", "targets"]]
        )
    out = pd.concat(frames, ignore_index=True, sort=False)
    if out.duplicated(IDENT).any():
        bad = out.loc[out.duplicated(IDENT, keep=False), IDENT].head(20)
        raise RuntimeError(f"weekly actual source duplicate identities: {bad.to_dict('records')}")
    return out


def _load_structures(
    predictions_path: Path,
    features_path: Path,
    conservation_path: Path,
    include_yards: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_cols = [
        "variant", "event_id", "team", "player_clean_key", "player", "wr_rank",
        "entitlement_tgt_share", "pred_targets", "season", "week", "actual_targets",
    ]
    if include_yards:
        pred_cols += ["mc_rec_yards", "actual_rec_yards"]
    pred = pd.read_csv(predictions_path, usecols=pred_cols)
    pred = pred.loc[pred["variant"].astype(str).eq(CANDIDATE)].copy()
    if len(pred) != EXPECTED_CANDIDATE_ROWS:
        raise RuntimeError(f"candidate row count drift {len(pred)} != {EXPECTED_CANDIDATE_ROWS}")
    _numeric(pred, ["wr_rank", "entitlement_tgt_share", "pred_targets", "season", "week", "actual_targets"])
    if include_yards:
        _numeric(pred, ["mc_rec_yards", "actual_rec_yards"])
    pred["season"] = pred["season"].astype(int)
    pred["week"] = pred["week"].astype(int)
    pred["wr_rank"] = pred["wr_rank"].astype(int)
    pred["team"] = pred["team"].map(canon_team)
    pred["player_clean_key"] = pred["player_clean_key"].astype(str)
    if pred.duplicated(IDENT).any():
        raise RuntimeError("duplicate candidate authority identity")
    by_season = pred.groupby("season").size().to_dict()
    if by_season != EXPECTED_CANDIDATE_BY_SEASON:
        raise RuntimeError(f"candidate season counts drift: {by_season}")
    team_games = pred[TG].drop_duplicates()
    if len(team_games) != EXPECTED_TEAM_GAMES:
        raise RuntimeError(f"team-game count drift: {len(team_games)}")
    anchors = pred.loc[pred["wr_rank"].eq(1)].copy()
    if anchors.duplicated(TG).any():
        raise RuntimeError("multiple WR1 anchors in candidate team-game")
    if len(anchors) != EXPECTED_ANCHOR_GAMES:
        raise RuntimeError(f"anchor count drift {len(anchors)}")

    feature_cols = [
        "event_id", "player", "player_clean_key", "team", "season", "week",
        "baseline_entitlement_tgt_share", "baseline_wr_rank",
        "candidate_entitlement_tgt_share",
    ]
    feat = pd.read_csv(features_path, usecols=feature_cols)
    _numeric(
        feat,
        ["season", "week", "baseline_entitlement_tgt_share", "baseline_wr_rank",
         "candidate_entitlement_tgt_share"],
    )
    feat["season"] = feat["season"].astype(int)
    feat["week"] = feat["week"].astype(int)
    feat["baseline_wr_rank"] = feat["baseline_wr_rank"].astype(int)
    feat["team"] = feat["team"].map(canon_team)
    feat["player_clean_key"] = feat["player_clean_key"].astype(str)
    feat = feat.loc[feat["baseline_wr_rank"].ge(2)].copy()
    if len(feat) != EXPECTED_FEATURE_ROWS:
        raise RuntimeError(f"WR2+ feature row count drift {len(feat)} != {EXPECTED_FEATURE_ROWS}")
    if feat.duplicated(IDENT).any():
        raise RuntimeError("duplicate WR2+ feature identity")

    cons_cols = [
        "event_id", "team", "candidate_wr_room_mass", "baseline_wr_room_mass",
        "wr_room_mass_gap", "baseline_secondary_pool", "candidate_secondary_pool",
        "secondary_pool_gap", "sportsbook_inputs_used", "current_or_future_outcomes_used",
        "test_season", "week",
    ]
    cons = pd.read_csv(conservation_path, usecols=cons_cols)
    _numeric(
        cons,
        [
            "candidate_wr_room_mass", "baseline_wr_room_mass", "wr_room_mass_gap",
            "baseline_secondary_pool", "candidate_secondary_pool", "secondary_pool_gap",
            "sportsbook_inputs_used", "current_or_future_outcomes_used", "test_season", "week",
        ],
    )
    cons["season"] = cons["test_season"].astype(int)
    cons["week"] = cons["week"].astype(int)
    cons["team"] = cons["team"].map(canon_team)
    if len(cons) != EXPECTED_TEAM_GAMES or cons.duplicated(TG).any():
        raise RuntimeError("conservation audit team-game grain drift")
    if cons["sportsbook_inputs_used"].abs().max() > TOL:
        raise RuntimeError("sportsbook input boundary violated")
    if cons["current_or_future_outcomes_used"].abs().max() > TOL:
        raise RuntimeError("future outcome boundary violated")
    for c in ["wr_room_mass_gap", "secondary_pool_gap"]:
        if cons[c].abs().max() > 1e-8:
            raise RuntimeError(f"conservation gap drift in {c}: {cons[c].abs().max()}")

    positive = pred.loc[pred["entitlement_tgt_share"].gt(TOL)].copy()
    positive["implied_team_target_pool"] = positive["pred_targets"] / positive["entitlement_tgt_share"]
    pool_audit = (
        positive.groupby(TG)["implied_team_target_pool"]
        .agg(["min", "max", "mean", "count"])
        .reset_index()
    )
    pool_audit["spread"] = pool_audit["max"] - pool_audit["min"]
    if pool_audit["spread"].max() > 1e-9:
        raise RuntimeError(f"implied team target pool not constant: {pool_audit['spread'].max()}")
    pools = pool_audit[TG + ["mean"]].rename(columns={"mean": "implied_team_target_pool"})
    if len(pools) != EXPECTED_TEAM_GAMES:
        raise RuntimeError("missing implied team target pool for authority team-game")

    return pred, feat, cons, pools


def _identity_status_frame(
    identities: pd.DataFrame,
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    cols = IDENT + ["targets"]
    raw = weekly[cols].rename(columns={"targets": "actual_targets_weekly"})
    out = identities.merge(raw, on=IDENT, how="left", validate="one_to_one", indicator=True)
    out["weekly_identity_status"] = np.where(
        out["_merge"].eq("both"), "EXACT_WEEKLY_MATCH", "UNRESOLVED_WEEKLY_IDENTITY"
    )
    out = out.drop(columns=["_merge"])
    return out


def mechanics_preflight(
    predictions_path: Path,
    features_path: Path,
    conservation_path: Path,
    out_dir: Path,
) -> dict:
    """Validate Phase-4B source/identity mechanics without loading yard outcomes."""
    pred, feat, cons, pools = _load_structures(
        predictions_path, features_path, conservation_path, include_yards=False
    )
    weekly = _load_weekly_actuals((2023, 2024))

    layer4_ids = feat[IDENT].copy()
    l4 = _identity_status_frame(layer4_ids, weekly)
    l4_counts = l4["weekly_identity_status"].value_counts().to_dict()

    anchors = pred.loc[pred["wr_rank"].eq(1), IDENT].copy()
    feature_on_anchor_games = feat.merge(anchors[TG].drop_duplicates(), on=TG, how="inner")
    canonical = pd.concat([anchors[IDENT], feature_on_anchor_games[IDENT]], ignore_index=True)
    canonical = canonical.drop_duplicates(IDENT)
    l23 = _identity_status_frame(canonical, weekly)
    l23_counts = l23["weekly_identity_status"].value_counts().to_dict()

    resolved_l4 = int(l4_counts.get("EXACT_WEEKLY_MATCH", 0))
    unresolved_l4 = int(l4_counts.get("UNRESOLVED_WEEKLY_IDENTITY", 0))
    resolved_l23 = int(l23_counts.get("EXACT_WEEKLY_MATCH", 0))
    unresolved_l23 = int(l23_counts.get("UNRESOLVED_WEEKLY_IDENTITY", 0))

    out_dir.mkdir(parents=True, exist_ok=True)
    l4.loc[l4["weekly_identity_status"].eq("UNRESOLVED_WEEKLY_IDENTITY")].to_csv(
        out_dir / "phase4b_preflight_layer4_unresolved_identities.csv", index=False
    )
    l23.loc[l23["weekly_identity_status"].eq("UNRESOLVED_WEEKLY_IDENTITY")].to_csv(
        out_dir / "phase4b_preflight_layer23_unresolved_identities.csv", index=False
    )
    result = {
        "specification": "WR_PHASE4B_AUTHORITY_EXACT_OPPORTUNITY_ATTRIBUTION_V1_PREFLIGHT",
        "candidate_rows": int(len(pred)),
        "authority_team_games": int(len(pred[TG].drop_duplicates())),
        "anchor_observable_team_games": int(len(anchors)),
        "canonical_wr2plus_feature_rows": int(len(feat)),
        "layer4_exact_weekly_matches": resolved_l4,
        "layer4_unresolved_weekly_identities": unresolved_l4,
        "layer4_unresolved_pct": float(unresolved_l4 / len(l4)) if len(l4) else np.nan,
        "layer23_canonical_identity_rows": int(len(l23)),
        "layer23_exact_weekly_matches": resolved_l23,
        "layer23_unresolved_weekly_identities": unresolved_l23,
        "implied_pool_team_games": int(len(pools)),
        "conservation_team_games": int(len(cons)),
        "receiving_yard_fields_loaded": False,
        "attribution_outcomes_run": False,
        "sportsbook_inputs": 0,
        "challenger_model_authorized": False,
        "production_change": False,
    }
    (out_dir / "phase4b_preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    return result


def _metric_summary(actual: pd.Series, pred: pd.Series) -> dict:
    a = pd.to_numeric(actual, errors="raise").to_numpy(dtype=float)
    p = pd.to_numeric(pred, errors="raise").to_numpy(dtype=float)
    e = p - a
    return {
        "n": int(len(a)),
        "mae": float(np.mean(np.abs(e))) if len(a) else np.nan,
        "rmse": float(np.sqrt(np.mean(e ** 2))) if len(a) else np.nan,
        "bias_pred_minus_actual": float(np.mean(e)) if len(a) else np.nan,
    }


def _layer1(pred: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    for r in pred.itertuples(index=False):
        opp, eff = symmetric_yard_decomposition(
            r.pred_targets, r.actual_targets, r.mc_rec_yards, r.actual_rec_yards
        )
        rows.append({
            "season": int(r.season),
            "week": int(r.week),
            "team": str(r.team),
            "player_clean_key": str(r.player_clean_key),
            "player": str(r.player),
            "wr_rank": int(r.wr_rank),
            "pred_targets": float(r.pred_targets),
            "actual_targets": float(r.actual_targets),
            "mc_rec_yards": float(r.mc_rec_yards),
            "actual_rec_yards": float(r.actual_rec_yards),
            "yard_residual": float(r.actual_rec_yards - r.mc_rec_yards),
            "opportunity_yards": opp,
            "efficiency_yards": eff,
        })
    detail = pd.DataFrame(rows)

    def summarize(g: pd.DataFrame, label: str) -> dict:
        return {
            "slice": label,
            "n": int(len(g)),
            "mean_abs_opportunity": float(g["opportunity_yards"].abs().mean()) if len(g) else np.nan,
            "median_abs_opportunity": float(g["opportunity_yards"].abs().median()) if len(g) else np.nan,
            "mean_abs_efficiency": float(g["efficiency_yards"].abs().mean()) if len(g) else np.nan,
            "median_abs_efficiency": float(g["efficiency_yards"].abs().median()) if len(g) else np.nan,
            "mean_signed_opportunity": float(g["opportunity_yards"].mean()) if len(g) else np.nan,
            "mean_signed_efficiency": float(g["efficiency_yards"].mean()) if len(g) else np.nan,
            "share_abs_opportunity_gt_efficiency": float(
                (g["opportunity_yards"].abs() > g["efficiency_yards"].abs()).mean()
            ) if len(g) else np.nan,
        }

    summaries = [summarize(detail, "POOLED")]
    for season in [2023, 2024]:
        summaries.append(summarize(detail.loc[detail["season"].eq(season)], str(season)))
    summaries.append(summarize(detail.loc[detail["wr_rank"].eq(1)], "WR1"))
    summaries.append(summarize(detail.loc[detail["wr_rank"].ge(2)], "WR2PLUS"))
    summaries.append(summarize(detail.loc[detail["actual_rec_yards"].ge(100.0)], "ACTUAL_100_PLUS"))
    summaries.append(summarize(detail.loc[detail["yard_residual"].abs().ge(30.0)], "ABS_RESIDUAL_30_PLUS"))
    return detail, summaries


def _anchor_feature_identity_sets(pred: pd.DataFrame, feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors = pred.loc[pred["wr_rank"].eq(1), IDENT + ["pred_targets", "player"]].copy()
    anchor_games = anchors[TG].drop_duplicates()
    sec = feat.merge(anchor_games, on=TG, how="inner", validate="many_to_one").copy()
    return anchors, sec


def _resolved_actuals_for_identities(
    identities: pd.DataFrame, weekly: pd.DataFrame, require_all: bool
) -> pd.DataFrame:
    x = _identity_status_frame(identities, weekly)
    if require_all and x["weekly_identity_status"].ne("EXACT_WEEKLY_MATCH").any():
        bad = x.loc[x["weekly_identity_status"].ne("EXACT_WEEKLY_MATCH"), IDENT].head(20)
        raise RuntimeError(f"unresolved canonical identity in required layer: {bad.to_dict('records')}")
    return x


def _layer2_3(
    pred: pd.DataFrame,
    feat: pd.DataFrame,
    cons: pd.DataFrame,
    pools: pd.DataFrame,
    weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    anchors, sec = _anchor_feature_identity_sets(pred, feat)
    anchor_games = anchors[TG].drop_duplicates()

    anchor_actual = _resolved_actuals_for_identities(
        anchors[IDENT + ["pred_targets", "player"]], weekly, require_all=True
    )
    sec_actual = _resolved_actuals_for_identities(sec[IDENT], weekly, require_all=True)

    actual_secondary = (
        sec_actual.groupby(TG, as_index=False)["actual_targets_weekly"]
        .sum()
        .rename(columns={"actual_targets_weekly": "actual_secondary_targets"})
    )
    actual_anchor = anchor_actual[TG + ["pred_targets", "actual_targets_weekly", "player_clean_key"]].rename(
        columns={
            "pred_targets": "pred_wr1_targets",
            "actual_targets_weekly": "actual_wr1_targets",
            "player_clean_key": "wr1_player_clean_key",
        }
    )

    team_actual = (
        weekly.groupby(TG, as_index=False)["targets"]
        .sum()
        .rename(columns={"targets": "actual_team_targets"})
    )
    base = anchor_games.merge(pools, on=TG, how="left", validate="one_to_one")
    base = base.merge(
        cons[TG + ["candidate_wr_room_mass"]],
        on=TG, how="left", validate="one_to_one"
    )
    base = base.merge(actual_anchor, on=TG, how="left", validate="one_to_one")
    base = base.merge(actual_secondary, on=TG, how="left", validate="one_to_one")
    base = base.merge(team_actual, on=TG, how="left", validate="one_to_one")
    if len(base) != EXPECTED_ANCHOR_GAMES or base.isna().any().any():
        raise RuntimeError("Layer2/3 anchor-observable assembly incomplete")

    base["pred_wr_room_targets"] = base["implied_team_target_pool"] * base["candidate_wr_room_mass"]
    base["actual_wr_room_targets"] = base["actual_wr1_targets"] + base["actual_secondary_targets"]
    if (base["actual_team_targets"] <= 0).any():
        raise RuntimeError("non-positive actual team target pool")
    base["pred_wr_share"] = base["candidate_wr_room_mass"]
    base["actual_wr_share"] = base["actual_wr_room_targets"] / base["actual_team_targets"]
    components = [
        symmetric_product_decomposition(pm, ps, am, a_s)
        for pm, ps, am, a_s in zip(
            base["implied_team_target_pool"], base["pred_wr_share"],
            base["actual_team_targets"], base["actual_wr_share"]
        )
    ]
    base["team_pool_component"] = [x[0] for x in components]
    base["wr_room_share_component"] = [x[1] for x in components]
    residual = base["actual_wr_room_targets"] - base["pred_wr_room_targets"]
    if not np.allclose(
        base["team_pool_component"] + base["wr_room_share_component"],
        residual, atol=TOL, rtol=1e-10
    ):
        raise RuntimeError("Layer2 product identity failed")

    base["pred_secondary_targets"] = base["pred_wr_room_targets"] - base["pred_wr1_targets"]

    canonical_ids = pd.concat([anchors[IDENT], sec[IDENT]], ignore_index=True).drop_duplicates(IDENT)
    raw_wr = weekly.loc[weekly["position"].isin(WR_POS)].merge(
        anchor_games, on=TG, how="inner", validate="many_to_one"
    )
    raw_wr = raw_wr.merge(
        canonical_ids.assign(_canonical=1),
        on=IDENT, how="left", validate="one_to_one"
    )
    off = raw_wr.loc[raw_wr["_canonical"].isna() & raw_wr["targets"].gt(TOL)].copy()
    off_tg = (
        off.groupby(TG, as_index=False)["targets"].sum()
        .rename(columns={"targets": "off_model_wr_targets"})
    )
    off_summary_frame = anchor_games.merge(off_tg, on=TG, how="left", validate="one_to_one")
    off_summary_frame["off_model_wr_targets"] = off_summary_frame["off_model_wr_targets"].fillna(0.0)
    off_summary = {
        "team_games": int(len(off_summary_frame)),
        "mean": float(off_summary_frame["off_model_wr_targets"].mean()),
        "median": float(off_summary_frame["off_model_wr_targets"].median()),
        "p90": float(off_summary_frame["off_model_wr_targets"].quantile(0.90)),
        "max": float(off_summary_frame["off_model_wr_targets"].max()),
        "total": float(off_summary_frame["off_model_wr_targets"].sum()),
        "team_games_with_any": int(off_summary_frame["off_model_wr_targets"].gt(0).sum()),
        "share_team_games_with_any": float(off_summary_frame["off_model_wr_targets"].gt(0).mean()),
    }

    all_games = pred[TG].drop_duplicates()
    obs = anchor_games.assign(anchor_observable=True)
    coverage = all_games.merge(obs, on=TG, how="left")
    coverage["anchor_observable"] = coverage["anchor_observable"].fillna(False).astype(bool)

    summaries = []
    for label, g in [("POOLED", base), ("2023", base.loc[base["season"].eq(2023)]), ("2024", base.loc[base["season"].eq(2024)])]:
        summaries.append({
            "slice": label,
            "n": int(len(g)),
            "room_mae": float((g["pred_wr_room_targets"] - g["actual_wr_room_targets"]).abs().mean()),
            "room_bias_pred_minus_actual": float((g["pred_wr_room_targets"] - g["actual_wr_room_targets"]).mean()),
            "mean_abs_team_pool_component": float(g["team_pool_component"].abs().mean()),
            "median_abs_team_pool_component": float(g["team_pool_component"].abs().median()),
            "mean_abs_wr_room_share_component": float(g["wr_room_share_component"].abs().mean()),
            "median_abs_wr_room_share_component": float(g["wr_room_share_component"].abs().median()),
            "share_abs_team_pool_gt_share_component": float(
                (g["team_pool_component"].abs() > g["wr_room_share_component"].abs()).mean()
            ),
        })

    for label, g in [("POOLED", base), ("2023", base.loc[base["season"].eq(2023)]), ("2024", base.loc[base["season"].eq(2024)])]:
        wr1 = _metric_summary(g["actual_wr1_targets"], g["pred_wr1_targets"])
        sec_m = _metric_summary(g["actual_secondary_targets"], g["pred_secondary_targets"])
        summaries.append({"slice": label, "role": "WR1", **wr1})
        summaries.append({"slice": label, "role": "WR2PLUS_POOL", **sec_m})

    return base, off, coverage, off_summary, summaries


def _layer4(
    feat: pd.DataFrame,
    pools: pd.DataFrame,
    weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict]:
    l4 = feat.merge(pools, on=TG, how="left", validate="many_to_one")
    if l4["implied_team_target_pool"].isna().any():
        raise RuntimeError("Layer4 missing implied team pool")
    l4["baseline_pred_targets"] = (
        l4["baseline_entitlement_tgt_share"] * l4["implied_team_target_pool"]
    )
    l4["r15_pred_targets"] = (
        l4["candidate_entitlement_tgt_share"] * l4["implied_team_target_pool"]
    )
    l4 = _identity_status_frame(l4, weekly)
    unresolved = l4.loc[l4["weekly_identity_status"].eq("UNRESOLVED_WEEKLY_IDENTITY")].copy()
    resolved = l4.loc[l4["weekly_identity_status"].eq("EXACT_WEEKLY_MATCH")].copy()
    if resolved.empty:
        raise RuntimeError("Layer4 resolved cohort empty")
    resolved["baseline_abs_error"] = (
        resolved["baseline_pred_targets"] - resolved["actual_targets_weekly"]
    ).abs()
    resolved["r15_abs_error"] = (
        resolved["r15_pred_targets"] - resolved["actual_targets_weekly"]
    ).abs()
    resolved["paired_mae_delta"] = resolved["r15_abs_error"] - resolved["baseline_abs_error"]

    metrics = []
    deltas = {}
    for label, g in [
        ("POOLED", resolved),
        ("2023", resolved.loc[resolved["season"].eq(2023)]),
        ("2024", resolved.loc[resolved["season"].eq(2024)]),
    ]:
        b = _metric_summary(g["actual_targets_weekly"], g["baseline_pred_targets"])
        r = _metric_summary(g["actual_targets_weekly"], g["r15_pred_targets"])
        delta = float(r["mae"] - b["mae"])
        deltas[label] = delta
        ci = cluster_bootstrap_mean_ci(
            g, "paired_mae_delta", reps=BOOT_REPS, seed=BOOT_SEED,
            stratify_season=(label == "POOLED")
        )
        toward = (g["r15_abs_error"] < g["baseline_abs_error"])
        away = (g["r15_abs_error"] > g["baseline_abs_error"])
        metrics.append({
            "slice": label,
            "resolved_n": int(len(g)),
            "baseline_target_mae": b["mae"],
            "r15_target_mae": r["mae"],
            "target_mae_delta_r15_minus_baseline": delta,
            "baseline_rmse": b["rmse"],
            "r15_rmse": r["rmse"],
            "baseline_bias_pred_minus_actual": b["bias_pred_minus_actual"],
            "r15_bias_pred_minus_actual": r["bias_pred_minus_actual"],
            "r15_toward_actual_n": int(toward.sum()),
            "r15_away_from_actual_n": int(away.sum()),
            "r15_equal_error_n": int(len(g) - toward.sum() - away.sum()),
            "r15_toward_actual_share": float(toward.mean()),
            "paired_mae_delta_ci_low": ci["ci_low"],
            "paired_mae_delta_ci_high": ci["ci_high"],
            "bootstrap_clusters": ci["clusters"],
            "bootstrap_reps": ci["reps"],
            "bootstrap_seed": ci["seed"],
        })

    expected = feat.groupby(TG)["player_clean_key"].nunique().rename("expected_n")
    resolved_n = resolved.groupby(TG)["player_clean_key"].nunique().rename("resolved_n")
    full_games = pd.concat([expected, resolved_n], axis=1).fillna(0)
    full_keys = full_games.loc[full_games["expected_n"].eq(full_games["resolved_n"])].reset_index()[TG]
    share_frame = resolved.merge(full_keys, on=TG, how="inner", validate="many_to_one")
    share_frame["actual_secondary_targets"] = share_frame.groupby(TG)["actual_targets_weekly"].transform("sum")
    share_frame = share_frame.loc[share_frame["actual_secondary_targets"].gt(0)].copy()
    share_frame["actual_secondary_share"] = (
        share_frame["actual_targets_weekly"] / share_frame["actual_secondary_targets"]
    )
    share_frame["baseline_share_den"] = share_frame.groupby(TG)["baseline_entitlement_tgt_share"].transform("sum")
    share_frame["r15_share_den"] = share_frame.groupby(TG)["candidate_entitlement_tgt_share"].transform("sum")
    share_frame["baseline_secondary_share"] = (
        share_frame["baseline_entitlement_tgt_share"] / share_frame["baseline_share_den"]
    )
    share_frame["r15_secondary_share"] = (
        share_frame["candidate_entitlement_tgt_share"] / share_frame["r15_share_den"]
    )
    share_diag = {
        "fully_resolved_positive_secondary_team_games": int(len(share_frame[TG].drop_duplicates())),
        "rows": int(len(share_frame)),
        "baseline_share_mae": float(
            (share_frame["baseline_secondary_share"] - share_frame["actual_secondary_share"]).abs().mean()
        ) if len(share_frame) else np.nan,
        "r15_share_mae": float(
            (share_frame["r15_secondary_share"] - share_frame["actual_secondary_share"]).abs().mean()
        ) if len(share_frame) else np.nan,
    }

    disp = r15_disposition(deltas["POOLED"], deltas["2023"], deltas["2024"])
    coverage = {
        "feature_rows": int(len(l4)),
        "exact_weekly_matches": int(len(resolved)),
        "unresolved_weekly_identities": int(len(unresolved)),
        "unresolved_pct": float(len(unresolved) / len(l4)),
        "by_season": {
            str(int(s)): {
                "feature_rows": int(len(g)),
                "exact_weekly_matches": int(g["weekly_identity_status"].eq("EXACT_WEEKLY_MATCH").sum()),
                "unresolved_weekly_identities": int(g["weekly_identity_status"].eq("UNRESOLVED_WEEKLY_IDENTITY").sum()),
            }
            for s, g in l4.groupby("season")
        },
        "disposition": disp,
        "share_diagnostic": share_diag,
    }
    return resolved, unresolved, metrics, coverage


def run_attribution(
    predictions_path: Path,
    features_path: Path,
    conservation_path: Path,
    out_dir: Path,
) -> dict:
    pred, feat, cons, pools = _load_structures(
        predictions_path, features_path, conservation_path, include_yards=True
    )
    weekly = _load_weekly_actuals((2023, 2024))

    l1_detail, l1_summary = _layer1(pred)
    l23_detail, off_detail, anchor_coverage, off_summary, l23_summary = _layer2_3(
        pred, feat, cons, pools, weekly
    )
    l4_resolved, l4_unresolved, l4_metrics, l4_coverage = _layer4(feat, pools, weekly)

    out_dir.mkdir(parents=True, exist_ok=True)
    l1_detail.to_csv(out_dir / "phase4b_layer1_yard_decomposition.csv", index=False)
    pd.DataFrame(l1_summary).to_csv(out_dir / "phase4b_layer1_summary.csv", index=False)
    l23_detail.to_csv(out_dir / "phase4b_layer2_3_team_game_detail.csv", index=False)
    off_detail.to_csv(out_dir / "phase4b_off_model_wr_targets.csv", index=False)
    anchor_coverage.to_csv(out_dir / "phase4b_anchor_observability.csv", index=False)
    pd.DataFrame(l23_summary).to_csv(out_dir / "phase4b_layer2_3_summary.csv", index=False)
    l4_resolved.to_csv(out_dir / "phase4b_layer4_resolved.csv", index=False)
    l4_unresolved.to_csv(out_dir / "phase4b_layer4_unresolved.csv", index=False)
    pd.DataFrame(l4_metrics).to_csv(out_dir / "phase4b_layer4_metrics.csv", index=False)

    result = {
        "specification": "WR_PHASE4B_AUTHORITY_EXACT_OPPORTUNITY_ATTRIBUTION_V1",
        "r15_layer4_disposition": l4_coverage["disposition"],
        "layer4_coverage": l4_coverage,
        "off_model_wr_target_mass": off_summary,
        "layer1_rows": int(len(l1_detail)),
        "layer23_team_games": int(len(l23_detail)),
        "sportsbook_inputs": 0,
        "challenger_model_authorized": False,
        "production_change": False,
        "attribution_outcomes_run": True,
    }
    (out_dir / "phase4b_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
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
