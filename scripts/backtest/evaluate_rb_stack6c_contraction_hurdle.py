#!/usr/bin/env python3
"""RB STACK6C: frozen rotation-gated one-sided contraction hurdle.

Research/development only.

Consumes the frozen P3/broad-STACK6 casebook, the STACK6C rotation-source
artifact, and the corrected historical identity bridge. Rotation features are
strictly prior, same-team PBP-derived usage/rotation state. Historical delayed
participation is used only through the source-audit identity bridge and is not
a fitted feature. Sportsbook data is loaded only after the football-first
retention disposition is frozen.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

START_WEEK = 6
MIN_TRAIN = 40
MIN_OVERALLOCATED = 20
CLASSIFIER_C = 1.0
CLASSIFIER_THRESHOLD = 0.50
RIDGE_ALPHA = 10.0
CONTRACTION_CAP = 4.0

TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "JAX": "JAC",
    "ARZ": "ARI",
    "WSH": "WAS",
}

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

ROTATION_CONCEPTS = [
    "touch_opp_share",
    "touch_lead_drive_share",
    "opening_drive_touch_share",
    "team_touch_leader_switch_rate",
    "team_touch_hhi",
]

ROTATION_FEATURES = [
    f"rotation_prior{n}_{concept}"
    for concept in ROTATION_CONCEPTS
    for n in (1, 3)
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


def classifier_pipeline():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=CLASSIFIER_C,
                    solver="lbfgs",
                    max_iter=1000,
                ),
            ),
        ]
    )


def magnitude_pipeline():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def prepare_parent(x):
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
    ] + AGG_FEATURES
    missing = [c for c in required if c not in z.columns]
    if missing:
        raise RuntimeError(f"STACK6C frozen parent missing fields: {missing}")

    z["season"] = num(z.season).astype(int)
    z["week"] = num(z.week).astype(int)
    z["team"] = z.team.map(tm)
    z["player_key"] = z.player_key.map(nk)
    z["target_order"] = z.season * 100 + z.week
    z["parent_att"] = num(z.parent_att)
    z["parent_yards"] = num(z.parent_yards)
    z["parent_ypc"] = num(z.parent_ypc)
    z["actual_rush_att"] = num(z.actual_rush_att)
    z["actual_rush_yards"] = num(z.actual_rush_yards)
    z["depth_rank"] = num(z.depth_rank)
    z["stack6_risk"] = bool_series(z.stack6_risk)
    for c in AGG_FEATURES:
        z[c] = num(z[c])
    z["carry_residual"] = z.actual_rush_att - z.parent_att
    z["overallocated"] = z.carry_residual.lt(0).astype(int)
    return z


def prepare_identity(x):
    z = x.copy()
    required = ["season", "team", "player_id", "player_clean_key"]
    missing = [c for c in required if c not in z.columns]
    if missing:
        raise RuntimeError(f"STACK6C identity source missing fields: {missing}")
    z["season"] = num(z.season).astype(int)
    z["team"] = z.team.map(tm)
    z["player_id"] = z.player_id.astype(str).str.strip()
    z["player_key"] = z.player_clean_key.map(nk)
    z = z.loc[z.player_id.ne("") & z.player_key.ne("")].copy()

    amb = (
        z[["season", "team", "player_id", "player_key"]]
        .drop_duplicates()
        .groupby(["season", "team", "player_id"])["player_key"]
        .nunique()
    )
    if int(amb.gt(1).sum()):
        raise RuntimeError(f"STACK6C ambiguous identity mappings: {int(amb.gt(1).sum())}")
    return z[["season", "team", "player_id", "player_key"]].drop_duplicates()


def prepare_rotation(rotation, identity):
    z = rotation.copy()
    required = ["season", "week", "team", "player_id"] + ROTATION_CONCEPTS
    missing = [c for c in required if c not in z.columns]
    if missing:
        raise RuntimeError(f"STACK6C rotation source missing fields: {missing}")

    z["season"] = num(z.season).astype(int)
    z["week"] = num(z.week).astype(int)
    z["team"] = z.team.map(tm)
    z["player_id"] = z.player_id.astype(str).str.strip()
    for c in ROTATION_CONCEPTS:
        z[c] = num(z[c])

    z = z.merge(identity, on=["season", "team", "player_id"], how="left", validate="many_to_one")
    miss = int(z.player_key.isna().sum())
    if miss:
        raise RuntimeError(f"STACK6C rotation identity bridge missed {miss} historical player-games")
    z["source_order"] = z.season * 100 + z.week

    dup = z.duplicated(["season", "week", "team", "player_key"], keep=False)
    if dup.any():
        raise RuntimeError(f"STACK6C duplicate mapped rotation player-games: {int(dup.sum())}")
    return z


def add_asof_rotation(target, rotation):
    z = target.copy()
    rows = []
    source_keys = set(zip(rotation.team.astype(str), rotation.player_key.astype(str)))

    for idx, r in z.iterrows():
        order = int(r.target_order)
        team = str(r.team)
        pk = str(r.player_key)
        hist = rotation.loc[
            rotation.team.eq(team)
            & rotation.player_key.eq(pk)
            & rotation.source_order.lt(order)
        ].sort_values("source_order")

        rec = {
            "_idx": idx,
            "rotation_identity_match_same_team_anywhere": int((team, pk) in source_keys),
            "rotation_history_games": int(len(hist)),
            "rotation_source_max_order": float(hist.source_order.max()) if len(hist) else np.nan,
        }
        for n in (1, 3):
            tail = hist.tail(n)
            for c in ROTATION_CONCEPTS:
                rec[f"rotation_prior{n}_{c}"] = (
                    float(num(tail[c]).mean()) if len(tail) and num(tail[c]).notna().any() else np.nan
                )
        rows.append(rec)

    f = pd.DataFrame(rows).set_index("_idx")
    z = z.join(f, how="left")
    max_order = num(z.rotation_source_max_order)
    z["rotation_asof_safe"] = (max_order.isna() | max_order.lt(num(z.target_order))).astype(int)

    z["stack6c_model_eligible"] = (
        num(z.week).ge(START_WEEK)
        & (~z.stack6_risk)
        & num(z.depth_rank).ge(2)
        & num(z.rotation_history_games).ge(1)
        & z.rotation_asof_safe.eq(1)
    )
    return z


def feature_blocks(z):
    missing_rotation = [c for c in ROTATION_FEATURES if c not in z.columns]
    missing_agg = [c for c in AGG_FEATURES if c not in z.columns]
    if missing_rotation or missing_agg:
        raise RuntimeError(
            f"STACK6C feature contract missing rotation={missing_rotation} aggregate={missing_agg}"
        )
    if len(ROTATION_FEATURES) != 10 or len(AGG_FEATURES) != 14:
        raise RuntimeError("STACK6C frozen feature counts changed")
    return {
        "ROTATION_HURDLE": list(ROTATION_FEATURES),
        "AGG_PLUS_ROTATION_HURDLE": list(AGG_FEATURES) + list(ROTATION_FEATURES),
    }


def oof_predictions(z, blocks):
    q = z.copy()
    for arm in blocks:
        q[f"overalloc_prob_{arm}"] = np.nan
        q[f"pred_contraction_{arm}"] = 0.0
        q[f"delta_{arm}"] = 0.0
        q[f"pred_att_{arm}"] = num(q.parent_att)
        q[f"pred_yards_{arm}"] = num(q.parent_yards)

    fits = []
    for week in range(START_WEEK, 19):
        base_train = q.loc[
            num(q.week).lt(week)
            & q.stack6c_model_eligible
            & q.carry_residual.notna()
        ].copy()
        test = q.loc[num(q.week).eq(week) & q.stack6c_model_eligible].copy()
        if test.empty:
            continue

        for arm, feats in blocks.items():
            train = base_train.loc[base_train[feats].notna().any(axis=1)].copy()
            magnitude_train = train.loc[train.overallocated.eq(1)].copy()
            class_count = int(train.overallocated.nunique())
            can_fit = (
                len(train) >= MIN_TRAIN
                and len(magnitude_train) >= MIN_OVERALLOCATED
                and class_count == 2
            )
            if not can_fit:
                fits.append(
                    {
                        "week": week,
                        "arm": arm,
                        "feature_count": len(feats),
                        "train_n": int(len(train)),
                        "overallocated_train_n": int(len(magnitude_train)),
                        "classifier_classes": class_count,
                        "test_n": int(len(test)),
                        "fit_performed": 0,
                        "mean_overalloc_prob": np.nan,
                        "classified_contraction_rate": 0.0,
                        "mean_contraction": 0.0,
                        "mean_delta": 0.0,
                    }
                )
                continue

            clf = classifier_pipeline()
            clf.fit(train[feats], train.overallocated.astype(int))
            prob = np.asarray(clf.predict_proba(test[feats])[:, 1], dtype=float)
            gate = prob >= CLASSIFIER_THRESHOLD

            mag_model = magnitude_pipeline()
            mag_target = -num(magnitude_train.carry_residual)
            mag_model.fit(magnitude_train[feats], mag_target)
            raw_mag = np.asarray(mag_model.predict(test[feats]), dtype=float)
            magnitude = np.clip(raw_mag, 0.0, CONTRACTION_CAP)
            used = np.where(gate, magnitude, 0.0)
            delta = -used
            if np.nanmax(delta) > 1e-12:
                raise RuntimeError(f"STACK6C positive delta generated in {arm} week {week}")

            att = np.clip(num(test.parent_att).to_numpy(dtype=float) + delta, 0.0, None)
            ypc = num(test.parent_ypc).to_numpy(dtype=float)
            yards = np.where(
                np.isfinite(ypc),
                att * ypc,
                num(test.parent_yards).to_numpy(dtype=float),
            )

            q.loc[test.index, f"overalloc_prob_{arm}"] = prob
            q.loc[test.index, f"pred_contraction_{arm}"] = used
            q.loc[test.index, f"delta_{arm}"] = delta
            q.loc[test.index, f"pred_att_{arm}"] = att
            q.loc[test.index, f"pred_yards_{arm}"] = yards

            fits.append(
                {
                    "week": week,
                    "arm": arm,
                    "feature_count": len(feats),
                    "train_n": int(len(train)),
                    "overallocated_train_n": int(len(magnitude_train)),
                    "classifier_classes": class_count,
                    "test_n": int(len(test)),
                    "fit_performed": 1,
                    "train_overallocated_rate": float(train.overallocated.mean()),
                    "mean_overalloc_prob": float(np.mean(prob)),
                    "classified_contraction_rate": float(np.mean(gate)),
                    "mean_raw_magnitude": float(np.mean(raw_mag)),
                    "mean_contraction": float(np.mean(used)),
                    "mean_delta": float(np.mean(delta)),
                }
            )

    for arm in blocks:
        if (num(q[f"delta_{arm}"]) > 1e-12).any():
            raise RuntimeError(f"STACK6C final casebook contains positive deltas for {arm}")
    return q, pd.DataFrame(fits)


def score_tables(z, arms):
    masks = {
        "all_rb_w6_18": num(z.week).ge(START_WEEK),
        "eligible_w6_18": z.stack6c_model_eligible,
        "eligible_w13_18": z.stack6c_model_eligible & num(z.week).ge(13),
        "m95f_risk_w6_18": z.stack6_risk & num(z.week).ge(START_WEEK),
        "depth1_w6_18": num(z.depth_rank).eq(1) & num(z.week).ge(START_WEEK),
        "depth2_w6_18": num(z.depth_rank).eq(2) & num(z.week).ge(START_WEEK),
        "depth3plus_w6_18": num(z.depth_rank).ge(3) & num(z.week).ge(START_WEEK),
    }
    cols = {
        "P3_PARENT": ("parent_att", "parent_yards"),
        **{a: (f"pred_att_{a}", f"pred_yards_{a}") for a in arms},
    }
    rows = []
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


def retention_gates(z, scores, blocks):
    def row(scope, arm):
        q = scores.loc[scores.scope.eq(scope) & scores.arm.eq(arm)]
        if q.empty:
            raise RuntimeError(f"STACK6C missing score {scope}/{arm}")
        return q.iloc[0]

    base = row("eligible_w6_18", "P3_PARENT")
    all_base = row("all_rb_w6_18", "P3_PARENT")
    late_base = row("eligible_w13_18", "P3_PARENT")
    rows = []

    for arm, feats in blocks.items():
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
        max_positive_delta = float(num(z[f"delta_{arm}"]).max())

        passed = int(
            carry_gain >= 0.20
            and yard_gain >= 0.15
            and late_gain > 0.0
            and all_reg <= 0.05
            and bias_worsen <= 0.25
            and risk_change <= 1e-9
            and depth1_change <= 1e-9
            and max_positive_delta <= 1e-12
        )
        rows.append(
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
                "max_positive_delta": max_positive_delta,
                "gate_carry_gain_ge_020": int(carry_gain >= 0.20),
                "gate_yard_gain_ge_015": int(yard_gain >= 0.15),
                "gate_late_yard_gain_gt_0": int(late_gain > 0.0),
                "gate_all_rb_regression_le_005": int(all_reg <= 0.05),
                "gate_bias_worsening_le_025": int(bias_worsen <= 0.25),
                "gate_risk_unchanged": int(risk_change <= 1e-9),
                "gate_depth1_unchanged": int(depth1_change <= 1e-9),
                "gate_no_expansion": int(max_positive_delta <= 1e-12),
                "gate_pass": passed,
            }
        )

    gates = pd.DataFrame(rows)
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


def protocol_features(z, blocks):
    rows = []
    for arm, feats in blocks.items():
        for feature in feats:
            rows.append(
                {
                    "arm": arm,
                    "feature": feature,
                    "feature_count_arm": len(feats),
                    "nonnull_rate_all": float(num(z[feature]).notna().mean()),
                    "nonnull_rate_eligible": float(
                        num(z.loc[z.stack6c_model_eligible, feature]).notna().mean()
                    )
                    if z.stack6c_model_eligible.any()
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def prep_market(x):
    z = x.copy()
    z["season"] = num(z.get("season", 2025)).fillna(2025).astype(int)
    z["week"] = num(z.week).astype(int)
    z["team"] = z.team.map(tm)
    if "player_key" in z.columns:
        z["player_key"] = z.player_key.map(nk)
    elif "join_key" in z.columns:
        z["player_key"] = z.join_key.map(nk)
    elif "player_clean_key" in z.columns:
        z["player_key"] = z.player_clean_key.map(nk)
    elif "name_key" in z.columns:
        z["player_key"] = z.name_key.map(nk)
    else:
        z["player_key"] = z.player.map(nk)
    return z


def market_audit(z, market, arms):
    cb = prep_market(market)
    keys = ["season", "week", "team", "player_key"]
    needed = keys + ["consensus_line"]
    missing = [c for c in needed if c not in cb.columns]
    if missing:
        raise RuntimeError(f"STACK6C market casebook missing fields: {missing}")

    q = z.merge(
        cb[needed].drop_duplicates(keys),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    cols = {
        "P3_PARENT": "parent_yards",
        **{a: f"pred_yards_{a}" for a in arms},
        "VEGAS_CONSENSUS": "consensus_line",
    }
    strata = {
        "all": pd.Series(True, index=q.index),
        "eligible": q.stack6c_model_eligible,
        "m95f_risk": q.stack6_risk,
        "m95f_nonrisk": ~q.stack6_risk,
        "depth1": num(q.depth_rank).eq(1),
        "depth2": num(q.depth_rank).eq(2),
        "depth3plus": num(q.depth_rank).ge(3),
    }
    rows = []
    for arm, col in cols.items():
        for scope, mask in strata.items():
            rows.append({"scope": scope, "arm": arm, **metric(q.loc[mask, "actual_rush_yards"], q.loc[mask, col])})

    details = []
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
    ap.add_argument("--rotation-root", type=Path, required=True)
    ap.add_argument("--identity-root", type=Path, required=True)
    ap.add_argument("--market-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    parent = prepare_parent(one(a.stack6_root, "stack6_2025_casebook.csv"))
    identity = prepare_identity(one(a.identity_root, "stack6_rb_player_game_roles.csv"))
    rotation = prepare_rotation(one(a.rotation_root, "stack6c_rotation_player_game_comparison.csv"), identity)
    x = add_asof_rotation(parent, rotation)
    blocks = feature_blocks(x)

    if float(x.rotation_asof_safe.mean()) != 1.0:
        raise RuntimeError("STACK6C strict-prior rotation leakage check failed")

    coverage = pd.DataFrame(
        [
            {
                "target_rows": int(len(x)),
                "w6_18_rows": int(num(x.week).ge(START_WEEK).sum()),
                "eligible_model_rows": int(x.stack6c_model_eligible.sum()),
                "m95f_risk_w6_18_rows": int((x.stack6_risk & num(x.week).ge(START_WEEK)).sum()),
                "depth1_w6_18_rows": int((num(x.depth_rank).eq(1) & num(x.week).ge(START_WEEK)).sum()),
                "rotation_identity_match_same_team_anywhere_rate": float(x.rotation_identity_match_same_team_anywhere.mean()),
                "rotation_history_coverage_rate": float(num(x.rotation_history_games).ge(1).mean()),
                "rotation_asof_leakage_pass_rate": float(x.rotation_asof_safe.mean()),
                "rotation_feature_count": len(ROTATION_FEATURES),
                "aggregate_feature_count": len(AGG_FEATURES),
                "rotation_arm_feature_count": len(blocks["ROTATION_HURDLE"]),
                "agg_plus_rotation_feature_count": len(blocks["AGG_PLUS_ROTATION_HURDLE"]),
            }
        ]
    )

    pred, fits = oof_predictions(x, blocks)
    scores = score_tables(pred, blocks)
    gates, selected = retention_gates(pred, scores, blocks)
    protocol = protocol_features(pred, blocks)

    # Freeze football-first disposition before sportsbook is loaded.
    if selected == "NONE":
        disposition = "STACK6C_NO_RETAINABLE_ROTATION_CONTRACTION_INCREMENT"
        next_step = "EXACT_AVAILABILITY_COMPETITOR_STATE_OR_NEW_INFORMATION_NO_HURDLE_RETUNING"
    else:
        disposition = "STACK6C_ROTATION_CONTRACTION_RETAINED_RESEARCH_ONLY"
        next_step = "FREEZE_WINNER_REQUIRE_2026_CONFIRMATION_AND_LIVE_IDENTITY_PBP_PIPELINE"

    disp = pd.DataFrame(
        [
            {
                "selected_arm": selected,
                "passing_arm_count": int(gates.gate_pass.sum()),
                "disposition": disposition,
                "sportsbook_upstream": 0,
                "sportsbook_loaded_after_disposition": 1,
                "classifier": "LOGISTIC_L2",
                "classifier_c": CLASSIFIER_C,
                "classifier_threshold": CLASSIFIER_THRESHOLD,
                "magnitude_model": "RIDGE",
                "ridge_alpha": RIDGE_ALPHA,
                "contraction_cap": CONTRACTION_CAP,
                "start_week": START_WEEK,
                "min_train": MIN_TRAIN,
                "min_overallocated_train": MIN_OVERALLOCATED,
                "positive_corrections_allowed": 0,
                "hyperparameter_search": 0,
                "feature_search": 0,
                "threshold_search": 0,
                "weight_search": 0,
                "delta_cap_search": 0,
                "population_search": 0,
                "production_change": 0,
                "validation_status": "2025_EXPOSED_RETROSPECTIVE_DEVELOPMENT",
                "prospective_2026_confirmation_required": int(selected != "NONE"),
                "next": next_step,
            }
        ]
    )

    # Downstream-only benchmark after the football disposition is frozen.
    market = one(a.market_root, "rb_market_casebook.csv")
    market_casebook, market_metrics, market_edges = market_audit(pred, market, blocks)

    outputs = {
        "stack6c_coverage.csv": coverage,
        "stack6c_protocol_features.csv": protocol,
        "stack6c_weekly_fits.csv": fits,
        "stack6c_score_table.csv": scores,
        "stack6c_retention_gates.csv": gates,
        "stack6c_disposition.csv": disp,
        "stack6c_market_metrics.csv": market_metrics,
        "stack6c_market_disagreement.csv": market_edges,
        "stack6c_2025_casebook.csv": pred,
        "stack6c_market_casebook.csv": market_casebook,
    }
    for name, df in outputs.items():
        df.to_csv(a.out_dir / name, index=False)

    print("=== STACK6C coverage ===")
    print(coverage.to_string(index=False))
    print("=== STACK6C football scores ===")
    print(
        scores.loc[
            scores.scope.isin(["eligible_w6_18", "eligible_w13_18", "all_rb_w6_18"])
        ].to_string(index=False)
    )
    print("=== STACK6C retention gates ===")
    print(gates.to_string(index=False))
    print("=== STACK6C frozen disposition ===")
    print(disp.to_string(index=False))
    print("=== STACK6C downstream market benchmark ===")
    print(
        market_metrics.loc[
            market_metrics.scope.isin(["all", "eligible", "m95f_nonrisk", "depth2", "depth3plus"])
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
