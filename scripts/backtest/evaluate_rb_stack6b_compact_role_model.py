#!/usr/bin/env python3
"""RB STACK6B: frozen compact secondary-back role concept model.

Research/development only.

STACK6B is an architecture-compression test around the frozen P3/STACK6 parent.
It consumes the frozen broad-STACK6 casebook so that parent projections,
eligibility, identity mapping, and strictly-prior situational history are exactly
the same rows/state that produced the STACK6 failure atlas.

Sportsbook information is downstream audit only and is not loaded until the
football-first retention disposition is frozen.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHA = 10.0
START_WEEK = 6
MIN_TRAIN = 40
DELTA_CAP = 4.0

TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "JAX": "JAC",
    "ARZ": "ARI",
    "WSH": "WAS",
}

# Frozen existing aggregate pregame role/snap block from STACK6.
AGG_FEATURES = [
    "depth_rank",
    "depth_slot",
    "prior1_snap_pct",
    "prior3_snap_pct",
    "prior3_rb_share",
    "credible_competitors",
    "prior_backfield_hhi",
    "injury_reported",
    "injury_out_doubtful",
    "injury_questionable",
    "rookie_flag",
    "prior1_rb_share",
    "prior1_carries",
    "prior3_carries",
]

# Exactly the eight frozen STACK6B compact concepts. No raw situational block.
COMPACT_FEATURES = [
    "concept_role_balance",
    "concept_passing_role",
    "concept_goal_line",
    "concept_rush_vs_presence",
    "concept_rush_momentum",
    "concept_early_momentum",
    "concept_role_stability",
    "concept_team_concentration",
]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def tm(v):
    s = "" if pd.isna(v) else str(v).strip().upper()
    return TEAM_ALIASES.get(s, s) if s not in {"", "NAN", "NONE", "<NA>"} else ""


def nk(v):
    s = "" if pd.isna(v) else str(v)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", s)


def lower(x):
    z = x.copy()
    z.columns = [str(c).strip().lower() for c in z.columns]
    return z


def one(root: Path, name: str):
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} under {root}; found {len(hits)}")
    return lower(pd.read_csv(hits[0], low_memory=False))


def bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    n = num(s)
    out = n.eq(1)
    unresolved = n.isna() & s.notna()
    if unresolved.any():
        text = s.astype(str).str.strip().str.lower()
        out = out.where(~unresolved, text.isin({"true", "t", "yes", "y"}))
    return out.fillna(False).astype(bool)


def prep_market(x):
    z = x.copy()
    z["season"] = num(z.get("season", 2025)).fillna(2025).astype(int)
    z["week"] = num(z["week"]).astype(int)
    z["team"] = z["team"].map(tm)
    if "player_key" in z:
        z["player_key"] = z["player_key"].map(nk)
    elif "join_key" in z:
        z["player_key"] = z["join_key"].map(nk)
    elif "player_clean_key" in z:
        z["player_key"] = z["player_clean_key"].map(nk)
    elif "name_key" in z:
        z["player_key"] = z["name_key"].map(nk)
    else:
        z["player_key"] = z["player"].map(nk)
    return z


def metric(y, p):
    q = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    if q.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
    e = q.p - q.y
    corr = q.y.corr(q.p) if len(q) >= 3 and q.y.nunique() > 1 and q.p.nunique() > 1 else np.nan
    return {
        "n": int(len(q)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.square(e).mean())),
        "bias": float(e.mean()),
        "corr": float(corr) if pd.notna(corr) else np.nan,
    }


def ridge():
    # No missingness indicators: COMPACT_ROLE must remain exactly eight concepts.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def fallback(z, base, n=3):
    same = f"sr_same_p{n}_{base}"
    anyc = f"sr_any_p{n}_{base}"
    if same not in z or anyc not in z:
        raise RuntimeError(f"STACK6B missing frozen as-of source columns for {base} p{n}")
    a = num(z[same])
    b = num(z[anyc])
    return a.where(a.notna(), b)


def safe_mean(series_list):
    return pd.concat(series_list, axis=1).mean(axis=1, skipna=True)


def add_compact_concepts(x):
    """Build exactly the eight precommitted football concepts.

    Player role state uses same-team prior history when available and any-team
    prior history as fallback, matching the frozen STACK6B protocol and failure
    atlas construction.
    """
    z = x.copy()

    role_presence = fallback(z, "hist_rb_presence_share", 3)
    role_rush = fallback(z, "hist_rush_share", 3)
    role_early = fallback(z, "hist_early_down_presence_share", 3)
    role_third = fallback(z, "hist_third_down_presence_share", 3)
    role_third_long = fallback(z, "hist_third_long_presence_share", 3)
    role_two_min = fallback(z, "hist_two_minute_presence_share", 3)
    role_short = fallback(z, "hist_short_yardage_presence_share", 3)
    role_rz = fallback(z, "hist_red_zone_presence_share", 3)
    role_i10 = fallback(z, "hist_inside10_presence_share", 3)
    role_i5 = fallback(z, "hist_inside5_presence_share", 3)

    role_rush_p1 = fallback(z, "hist_rush_share", 1)
    role_early_p1 = fallback(z, "hist_early_down_presence_share", 1)
    role_third_p1 = fallback(z, "hist_third_down_presence_share", 1)

    rushing_role = safe_mean([role_rush, role_early, role_short, role_rz])
    passing_role = safe_mean([role_third, role_third_long, role_two_min])

    z["concept_role_balance"] = rushing_role - passing_role
    z["concept_passing_role"] = passing_role
    z["concept_goal_line"] = safe_mean([role_i10, role_i5])
    z["concept_rush_vs_presence"] = role_rush - role_presence
    z["concept_rush_momentum"] = role_rush_p1 - role_rush
    z["concept_early_momentum"] = role_early_p1 - role_early
    z["concept_role_stability"] = -safe_mean(
        [
            (role_rush_p1 - role_rush).abs(),
            (role_early_p1 - role_early).abs(),
            (role_third_p1 - role_third).abs(),
        ]
    )

    hhi = "sr_team_p3_team_hhi_rb_presence"
    if hhi not in z:
        raise RuntimeError(f"STACK6B missing frozen team concentration source: {hhi}")
    z["concept_team_concentration"] = num(z[hhi])

    if COMPACT_FEATURES != [c for c in COMPACT_FEATURES if c in z.columns]:
        missing = [c for c in COMPACT_FEATURES if c not in z.columns]
        raise RuntimeError(f"STACK6B compact concept construction incomplete: {missing}")
    return z


def validate_and_prepare_parent(x):
    z = x.copy()
    required = [
        "season",
        "week",
        "team",
        "player_key",
        "parent_att",
        "parent_yards",
        "parent_ypc",
        "actual_rush_att",
        "actual_rush_yards",
        "depth_rank",
        "stack6_risk",
        "stack6_history_games",
        "stack6_asof_leakage_safe",
        "stack6_model_eligible",
    ]
    missing = [c for c in required if c not in z]
    if missing:
        raise RuntimeError(f"STACK6B frozen STACK6 casebook missing columns: {missing}")

    z["season"] = num(z.season).astype(int)
    z["week"] = num(z.week).astype(int)
    z["team"] = z.team.map(tm)
    z["player_key"] = z.player_key.map(nk)
    z["parent_att"] = num(z.parent_att)
    z["parent_yards"] = num(z.parent_yards)
    z["parent_ypc"] = num(z.parent_ypc)
    z["actual_rush_att"] = num(z.actual_rush_att)
    z["actual_rush_yards"] = num(z.actual_rush_yards)
    z["stack6_risk"] = bool_series(z.stack6_risk)
    z["stack6_asof_leakage_safe"] = bool_series(z.stack6_asof_leakage_safe)
    original_eligible = bool_series(z.stack6_model_eligible)

    recomputed = (
        num(z.week).ge(START_WEEK)
        & (~z.stack6_risk)
        & num(z.depth_rank).ge(2)
        & num(z.stack6_history_games).ge(1)
        & z.stack6_asof_leakage_safe
    )
    mismatch = int((original_eligible != recomputed).sum())
    if mismatch:
        raise RuntimeError(f"STACK6B eligibility mismatch vs frozen STACK6 casebook: {mismatch} rows")
    if not z.stack6_asof_leakage_safe.all():
        raise RuntimeError("STACK6B input contains non-strict-prior situational rows")

    z["stack6b_model_eligible"] = recomputed
    z["carry_residual"] = z.actual_rush_att - z.parent_att
    return z


def require_features(z):
    missing_agg = [c for c in AGG_FEATURES if c not in z]
    missing_compact = [c for c in COMPACT_FEATURES if c not in z]
    if missing_agg or missing_compact:
        raise RuntimeError(
            f"STACK6B frozen feature contract missing aggregate={missing_agg} compact={missing_compact}"
        )
    if len(AGG_FEATURES) != 14 or len(COMPACT_FEATURES) != 8:
        raise RuntimeError("STACK6B feature-count contract changed")
    return {
        "COMPACT_ROLE": list(COMPACT_FEATURES),
        "AGG_PLUS_COMPACT": list(AGG_FEATURES) + list(COMPACT_FEATURES),
    }


def training_clip_bounds(y):
    """Preserve broad STACK6 clipping semantics without test leakage.

    The 5th/95th percentile limits are calculated from training residuals only,
    then intersected with [-4,+4]. Training targets themselves are not
    winsorized; the model-predicted carry correction is clipped to those frozen
    training-derived bounds.
    """
    v = num(y).dropna()
    if len(v) < 2:
        raise RuntimeError("STACK6B insufficient residuals for training-only clip bounds")
    q05, q95 = np.nanquantile(v, [0.05, 0.95])
    lo = max(float(q05), -DELTA_CAP)
    hi = min(float(q95), DELTA_CAP)
    if lo > hi:
        raise RuntimeError(f"STACK6B invalid training-only clip interval: {lo} > {hi}")
    return float(q05), float(q95), lo, hi


def oof_predictions(z, feature_blocks):
    q = z.copy()
    for arm in feature_blocks:
        q[f"pred_att_{arm}"] = num(q.parent_att)
        q[f"pred_yards_{arm}"] = num(q.parent_yards)
        q[f"delta_{arm}"] = 0.0

    fits = []
    for week in range(START_WEEK, 19):
        base_train = q.loc[
            num(q.week).lt(week) & q.stack6b_model_eligible & q.carry_residual.notna()
        ].copy()
        test = q.loc[num(q.week).eq(week) & q.stack6b_model_eligible].copy()
        if test.empty:
            continue

        for arm, feats in feature_blocks.items():
            # Same broad-STACK6 rule: rows need at least one observed feature.
            train = base_train.loc[base_train[feats].notna().any(axis=1)].copy()
            if len(train) < MIN_TRAIN:
                fits.append(
                    {
                        "week": week,
                        "arm": arm,
                        "feature_count": len(feats),
                        "train_n": len(train),
                        "test_n": len(test),
                        "fit_performed": 0,
                        "resid_q05_train": np.nan,
                        "resid_q95_train": np.nan,
                        "clip_lo": np.nan,
                        "clip_hi": np.nan,
                        "training_target_winsorized": 0,
                        "prediction_clipped": 1,
                        "mean_delta": 0.0,
                        "mean_abs_delta": 0.0,
                    }
                )
                continue

            q05, q95, clip_lo, clip_hi = training_clip_bounds(train.carry_residual)
            model = ridge()
            model.fit(train[feats], num(train.carry_residual))
            raw_delta = np.asarray(model.predict(test[feats]), dtype=float)
            delta = np.clip(raw_delta, clip_lo, clip_hi)
            att = np.clip(num(test.parent_att).to_numpy(dtype=float) + delta, 0, None)
            ypc = num(test.parent_ypc).to_numpy(dtype=float)
            yards = np.where(
                np.isfinite(ypc),
                att * ypc,
                num(test.parent_yards).to_numpy(dtype=float),
            )

            q.loc[test.index, f"delta_{arm}"] = delta
            q.loc[test.index, f"pred_att_{arm}"] = att
            q.loc[test.index, f"pred_yards_{arm}"] = yards
            fits.append(
                {
                    "week": week,
                    "arm": arm,
                    "feature_count": len(feats),
                    "train_n": len(train),
                    "test_n": len(test),
                    "fit_performed": 1,
                    "resid_q05_train": q05,
                    "resid_q95_train": q95,
                    "clip_lo": clip_lo,
                    "clip_hi": clip_hi,
                    "training_target_winsorized": 0,
                    "prediction_clipped": 1,
                    "mean_raw_delta": float(np.mean(raw_delta)),
                    "mean_delta": float(np.mean(delta)),
                    "mean_abs_delta": float(np.mean(np.abs(delta))),
                }
            )
    return q, pd.DataFrame(fits)


def score_tables(z, arms):
    masks = {
        "all_rb_w6_18": num(z.week).ge(START_WEEK),
        "eligible_w6_18": z.stack6b_model_eligible,
        "eligible_w13_18": z.stack6b_model_eligible & num(z.week).ge(13),
        "m95f_risk_w6_18": z.stack6_risk & num(z.week).ge(START_WEEK),
        "depth1_w6_18": num(z.depth_rank).eq(1) & num(z.week).ge(START_WEEK),
        "depth2_w6_18": num(z.depth_rank).eq(2) & num(z.week).ge(START_WEEK),
        "depth3plus_w6_18": num(z.depth_rank).ge(3) & num(z.week).ge(START_WEEK),
    }
    rows = []
    cols = {
        "P3_PARENT": ("parent_att", "parent_yards"),
        **{a: (f"pred_att_{a}", f"pred_yards_{a}") for a in arms},
    }
    for scope, mask in masks.items():
        g = z.loc[mask]
        for arm, (att_col, yard_col) in cols.items():
            cm = metric(g.actual_rush_att, g[att_col])
            ym = metric(g.actual_rush_yards, g[yard_col])
            rows.append(
                {
                    "scope": scope,
                    "arm": arm,
                    "n": ym["n"],
                    "carry_mae": cm["mae"],
                    "carry_rmse": cm["rmse"],
                    "carry_bias": cm["bias"],
                    "carry_corr": cm["corr"],
                    "yard_mae": ym["mae"],
                    "yard_rmse": ym["rmse"],
                    "yard_bias": ym["bias"],
                    "yard_corr": ym["corr"],
                }
            )
    return pd.DataFrame(rows)


def retention_gates(z, scores, feature_blocks):
    def row(scope, arm):
        q = scores.loc[scores.scope.eq(scope) & scores.arm.eq(arm)]
        if q.empty:
            raise RuntimeError(f"missing STACK6B score {scope}/{arm}")
        return q.iloc[0]

    base = row("eligible_w6_18", "P3_PARENT")
    all_base = row("all_rb_w6_18", "P3_PARENT")
    late_base = row("eligible_w13_18", "P3_PARENT")
    out = []

    for arm, feats in feature_blocks.items():
        r = row("eligible_w6_18", arm)
        all_r = row("all_rb_w6_18", arm)
        late_r = row("eligible_w13_18", arm)
        carry_gain = float(base.carry_mae - r.carry_mae)
        yard_gain = float(base.yard_mae - r.yard_mae)
        late_gain = float(late_base.yard_mae - late_r.yard_mae)
        all_reg = float(all_r.yard_mae - all_base.yard_mae)
        bias_worsen = abs(float(r.carry_bias)) - abs(float(base.carry_bias))

        risk = z.stack6_risk
        risk_change = (
            float(
                np.nanmax(
                    np.abs(
                        num(z.loc[risk, f"pred_yards_{arm}"])
                        - num(z.loc[risk, "parent_yards"])
                    )
                )
            )
            if risk.any()
            else 0.0
        )
        depth1 = num(z.depth_rank).eq(1)
        depth1_change = (
            float(
                np.nanmax(
                    np.abs(
                        num(z.loc[depth1, f"pred_yards_{arm}"])
                        - num(z.loc[depth1, "parent_yards"])
                    )
                )
            )
            if depth1.any()
            else 0.0
        )

        passed = int(
            carry_gain >= 0.20
            and yard_gain >= 0.15
            and late_gain > 0.0
            and all_reg <= 0.05
            and bias_worsen <= 0.25
            and risk_change <= 1e-9
            and depth1_change <= 1e-9
        )
        out.append(
            {
                "arm": arm,
                "feature_count": len(feats),
                "carry_mae_gain": carry_gain,
                "yard_mae_gain": yard_gain,
                "late_yard_mae_gain": late_gain,
                "all_rb_yard_mae_regression": all_reg,
                "carry_abs_bias_worsening": bias_worsen,
                "max_risk_yard_change": risk_change,
                "max_depth1_yard_change": depth1_change,
                "gate_carry_gain_ge_020": int(carry_gain >= 0.20),
                "gate_yard_gain_ge_015": int(yard_gain >= 0.15),
                "gate_late_yard_gain_gt_0": int(late_gain > 0.0),
                "gate_all_rb_regression_le_005": int(all_reg <= 0.05),
                "gate_bias_worsening_le_025": int(bias_worsen <= 0.25),
                "gate_risk_unchanged": int(risk_change <= 1e-9),
                "gate_depth1_unchanged": int(depth1_change <= 1e-9),
                "gate_pass": passed,
            }
        )

    gates = pd.DataFrame(out)
    passing = gates.loc[gates.gate_pass.eq(1)].copy()
    selected = "NONE"
    if len(passing):
        best_gain = float(passing.yard_mae_gain.max())
        pool = passing.loc[passing.yard_mae_gain.ge(best_gain - 0.05)].sort_values(
            ["feature_count", "yard_mae_gain", "arm"],
            ascending=[True, False, True],
        )
        selected = str(pool.iloc[0].arm)
    gates["selected_arm"] = selected
    return gates, selected


def protocol_audit(z, feature_blocks):
    rows = []
    for arm, feats in feature_blocks.items():
        for feature in feats:
            rows.append(
                {
                    "arm": arm,
                    "feature": feature,
                    "feature_count_arm": len(feats),
                    "nonnull_rate_all": float(num(z[feature]).notna().mean()),
                    "nonnull_rate_eligible": float(
                        num(z.loc[z.stack6b_model_eligible, feature]).notna().mean()
                    )
                    if z.stack6b_model_eligible.any()
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def market_audit(z, market, arms):
    cb = prep_market(market)
    keys = ["season", "week", "team", "player_key"]
    needed = keys + ["consensus_line"]
    missing = [c for c in needed if c not in cb]
    if missing:
        raise RuntimeError(f"STACK6B market casebook missing columns: {missing}")

    q = z.merge(
        cb[needed].drop_duplicates(keys),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    rows = []
    details = []
    cols = {
        "P3_PARENT": "parent_yards",
        **{a: f"pred_yards_{a}" for a in arms},
        "VEGAS_CONSENSUS": "consensus_line",
    }
    strata = {
        "all": pd.Series(True, index=q.index),
        "m95f_risk": q.stack6_risk,
        "m95f_nonrisk": ~q.stack6_risk,
        "depth1": num(q.depth_rank).eq(1),
        "depth2": num(q.depth_rank).eq(2),
        "depth3plus": num(q.depth_rank).ge(3),
        "eligible": q.stack6b_model_eligible,
    }

    for arm, col in cols.items():
        for scope, mask in strata.items():
            met = metric(q.loc[mask, "actual_rush_yards"], q.loc[mask, col])
            rows.append({"scope": scope, "arm": arm, **met})

    for arm, col in {k: v for k, v in cols.items() if k != "VEGAS_CONSENSUS"}.items():
        edge = num(q[col]) - num(q.consensus_line)
        model_err = (num(q[col]) - num(q.actual_rush_yards)).abs()
        vegas_err = (num(q.consensus_line) - num(q.actual_rush_yards)).abs()
        buckets = {
            "within5": edge.abs().lt(5),
            "above5_10": edge.between(5, 10, inclusive="left"),
            "below5_10": edge.between(-10, -5, inclusive="right"),
            "above10": edge.ge(10),
            "below10": edge.le(-10),
        }
        for bucket, mask in buckets.items():
            if not mask.any():
                continue
            details.append(
                {
                    "arm": arm,
                    "bucket": bucket,
                    "n": int(mask.sum()),
                    "model_mae": float(model_err[mask].mean()),
                    "vegas_mae": float(vegas_err[mask].mean()),
                    "model_closer_rate": float((model_err[mask] < vegas_err[mask]).mean()),
                    "mean_edge": float(edge[mask].mean()),
                }
            )
    return q, pd.DataFrame(rows), pd.DataFrame(details)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack6-root", type=Path, required=True)
    ap.add_argument("--market-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    parent = one(a.stack6_root, "stack6_2025_casebook.csv")
    x = validate_and_prepare_parent(parent)
    x = add_compact_concepts(x)
    blocks = require_features(x)

    coverage = pd.DataFrame(
        [
            {
                "target_rows": int(len(x)),
                "w6_18_rows": int(num(x.week).ge(START_WEEK).sum()),
                "eligible_model_rows": int(x.stack6b_model_eligible.sum()),
                "m95f_risk_w6_18_rows": int((x.stack6_risk & num(x.week).ge(START_WEEK)).sum()),
                "depth1_w6_18_rows": int((num(x.depth_rank).eq(1) & num(x.week).ge(START_WEEK)).sum()),
                "asof_leakage_pass_rate": float(x.stack6_asof_leakage_safe.mean()),
                "compact_feature_count": len(COMPACT_FEATURES),
                "aggregate_feature_count": len(AGG_FEATURES),
                "compact_arm_feature_count": len(blocks["COMPACT_ROLE"]),
                "agg_plus_compact_feature_count": len(blocks["AGG_PLUS_COMPACT"]),
            }
        ]
    )

    pred, fits = oof_predictions(x, blocks)
    scores = score_tables(pred, blocks)
    gates, selected = retention_gates(pred, scores, blocks)

    # Freeze football-first disposition before loading or evaluating sportsbook data.
    if selected == "NONE":
        disposition = "STACK6B_NO_RETAINABLE_COMPACT_ROLE_INCREMENT"
        next_step = "NEW_INFORMATION_FAMILY_OR_COMPACT_ROLE_POSTMORTEM_NO_RETUNING"
    else:
        disposition = "STACK6B_COMPACT_ROLE_INCREMENT_RETAINED_RESEARCH_ONLY"
        next_step = "FREEZE_WINNER_BUILD_LIVE_EQUIVALENT_AND_REQUIRE_2026_CONFIRMATION"

    disp = pd.DataFrame(
        [
            {
                "selected_arm": selected,
                "passing_arm_count": int(gates.gate_pass.sum()),
                "disposition": disposition,
                "sportsbook_upstream": 0,
                "sportsbook_loaded_after_disposition": 1,
                "model_fit": 1,
                "ridge_alpha": RIDGE_ALPHA,
                "start_week": START_WEEK,
                "min_train": MIN_TRAIN,
                "residual_quantile_low": 0.05,
                "residual_quantile_high": 0.95,
                "delta_cap": DELTA_CAP,
                "training_target_winsorized": 0,
                "prediction_clip_from_training_residuals": 1,
                "hyperparameter_search": 0,
                "feature_search": 0,
                "threshold_search": 0,
                "weight_search": 0,
                "production_change": 0,
                "validation_status": "2025_EXPOSED_RETROSPECTIVE_DEVELOPMENT",
                "prospective_2026_confirmation_required": int(selected != "NONE"),
                "next": next_step,
            }
        ]
    )

    # Downstream-only benchmark after the research disposition is frozen.
    market = one(a.market_root, "rb_market_casebook.csv")
    market_casebook, market_metrics, market_edges = market_audit(pred, market, blocks)

    protocol = protocol_audit(pred, blocks)
    outputs = {
        "stack6b_coverage.csv": coverage,
        "stack6b_protocol_features.csv": protocol,
        "stack6b_weekly_fits.csv": fits,
        "stack6b_score_table.csv": scores,
        "stack6b_retention_gates.csv": gates,
        "stack6b_disposition.csv": disp,
        "stack6b_market_metrics.csv": market_metrics,
        "stack6b_market_disagreement.csv": market_edges,
        "stack6b_2025_casebook.csv": pred,
        "stack6b_market_casebook.csv": market_casebook,
    }
    for name, df in outputs.items():
        df.to_csv(a.out_dir / name, index=False)

    print("=== STACK6B coverage ===")
    print(coverage.to_string(index=False))
    print("=== STACK6B eligible/all/late football scores ===")
    print(
        scores.loc[
            scores.scope.isin(["eligible_w6_18", "eligible_w13_18", "all_rb_w6_18"])
        ].to_string(index=False)
    )
    print("=== STACK6B retention gates ===")
    print(gates.to_string(index=False))
    print("=== STACK6B frozen disposition ===")
    print(disp.to_string(index=False))
    print("=== STACK6B downstream market benchmark ===")
    print(
        market_metrics.loc[
            market_metrics.scope.isin(["all", "eligible", "m95f_nonrisk", "depth2", "depth3plus"])
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
