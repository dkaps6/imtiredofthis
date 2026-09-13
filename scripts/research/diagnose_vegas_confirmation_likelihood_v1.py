#!/usr/bin/env python3
"""Pregame market-geometry confirmation likelihood diagnostic.

Frozen by docs/research/VEGAS_CONFIRMATION_LIKELIHOOD_V1_PLAN.md.
Research only. No production changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts._opponent_map import canon_team
from scripts.backtest.historical_player_logs import build_historical_player_logs
from scripts.research.diagnose_game_script_confirmed_player_usage_v1 import (
    build_analysis_frame,
    compare_by_side,
    games_to_schedule_history,
)
from scripts.research.diagnose_vegas_line_gamescript_calibration_v1 import load_game_outcomes

TRAIN_SEASONS = (2023, 2024)
TEST_SEASON = 2025
KEY_MARGINS = np.array([3.0, 7.0, 10.0, 14.0], dtype=float)
MONEYLINE_COVERAGE_MIN = 0.80
PRIMARY_THRESHOLD = 7.0
SENSITIVITY_THRESHOLD = 3.0
MIN_CLASS_ROWS = 30
MIN_DOWNSTREAM_ROWS_PER_SIDE = 30


def _num(v):
    return pd.to_numeric(v, errors="coerce")


def american_implied_prob(x: float) -> float:
    if not np.isfinite(x) or x == 0:
        return float("nan")
    if x > 0:
        return float(100.0 / (x + 100.0))
    return float((-x) / ((-x) + 100.0))


def load_market_geometry(seasons: list[int]) -> pd.DataFrame:
    """Load only pregame market fields from nflverse schedules."""
    import nflreadpy as nfl

    rows: list[dict] = []
    for season in seasons:
        s = nfl.load_schedules(int(season))
        if hasattr(s, "to_pandas"):
            s = s.to_pandas()
        s = pd.DataFrame(s)
        s.columns = [str(c).strip().lower() for c in s.columns]
        if "game_type" in s.columns:
            s = s.loc[s["game_type"].astype(str).str.upper().eq("REG")].copy()
        required = {"season", "week", "home_team", "away_team", "spread_line", "total_line"}
        missing = required - set(s.columns)
        if missing:
            raise RuntimeError(f"{season} schedule missing required columns: {sorted(missing)}")
        for _, r in s.iterrows():
            rows.append({
                "season": int(r["season"]),
                "week": int(r["week"]),
                "game_id": r.get("game_id"),
                "home_team": canon_team(r["home_team"]),
                "away_team": canon_team(r["away_team"]),
                "home_moneyline": _num(pd.Series([r.get("home_moneyline")])).iloc[0] if "home_moneyline" in s.columns else np.nan,
                "away_moneyline": _num(pd.Series([r.get("away_moneyline")])).iloc[0] if "away_moneyline" in s.columns else np.nan,
            })
    out = pd.DataFrame(rows)
    keys = ["season", "week", "home_team", "away_team"]
    if out.duplicated(keys).any():
        raise RuntimeError("duplicate market geometry rows")
    return out


def add_pregame_features(games: pd.DataFrame, geometry: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keys = ["season", "week", "home_team", "away_team"]
    x = games.merge(geometry, on=keys, how="left", validate="one_to_one", suffixes=("", "_market"))
    x["abs_spread"] = x["predicted_margin_home"].abs()
    x["key_margin_distance"] = x["abs_spread"].map(
        lambda v: float(np.min(np.abs(KEY_MARGINS - float(v)))) if np.isfinite(v) else np.nan
    )

    two_sided = x["home_moneyline"].notna() & x["away_moneyline"].notna()
    train_mask = x["season"].isin(TRAIN_SEASONS)
    test_mask = x["season"].eq(TEST_SEASON)
    train_cov = float(two_sided.loc[train_mask].mean()) if train_mask.any() else 0.0
    test_cov = float(two_sided.loc[test_mask].mean()) if test_mask.any() else 0.0
    use_moneyline = train_cov >= MONEYLINE_COVERAGE_MIN and test_cov >= MONEYLINE_COVERAGE_MIN

    meta = {
        "moneyline_train_coverage": train_cov,
        "moneyline_test_coverage": test_cov,
        "moneyline_features_used": bool(use_moneyline),
    }
    if not use_moneyline:
        return x, meta

    x["home_ml_raw_p"] = x["home_moneyline"].map(american_implied_prob)
    x["away_ml_raw_p"] = x["away_moneyline"].map(american_implied_prob)
    denom = x["home_ml_raw_p"] + x["away_ml_raw_p"]
    x["home_ml_novig_p"] = x["home_ml_raw_p"] / denom
    x["away_ml_novig_p"] = x["away_ml_raw_p"] / denom
    x["favorite_ml_prob"] = x[["home_ml_novig_p", "away_ml_novig_p"]].max(axis=1)

    spread_side_prob = np.where(
        x["predicted_margin_home"] > 0,
        x["home_ml_novig_p"],
        np.where(x["predicted_margin_home"] < 0, x["away_ml_novig_p"], x["favorite_ml_prob"]),
    )
    x["spread_favored_ml_prob"] = spread_side_prob

    fit = x.loc[train_mask & x["spread_favored_ml_prob"].notna() & x["abs_spread"].notna()].copy()
    if len(fit) < 50:
        raise RuntimeError("moneyline block passed coverage gate but has <50 usable train rows")
    mapper = LinearRegression()
    mapper.fit(fit[["abs_spread"]].to_numpy(dtype=float), fit["spread_favored_ml_prob"].to_numpy(dtype=float))
    x["expected_spread_favored_ml_prob"] = mapper.predict(x[["abs_spread"]].to_numpy(dtype=float))
    x["spread_ml_consistency_resid"] = x["spread_favored_ml_prob"] - x["expected_spread_favored_ml_prob"]
    meta["spread_ml_mapper_slope"] = float(mapper.coef_[0])
    meta["spread_ml_mapper_intercept"] = float(mapper.intercept_)
    return x, meta


def total_cutoff_from_parent_frame(games: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    schedule_history = games_to_schedule_history(games)
    logs = build_historical_player_logs(
        seasons=sorted(set(games["season"].astype(int).tolist())),
        schedule_history=schedule_history,
    )
    frame = build_analysis_frame(games, logs)
    cutoff = float(frame["predicted_total"].median())
    return cutoff, frame


def add_labels(x: pd.DataFrame, total_cutoff: float) -> pd.DataFrame:
    out = x.copy()
    margin_same_side = np.sign(out["predicted_margin_home"]) == np.sign(out["actual_margin_home"])
    total_same_side = (out["predicted_total"] >= total_cutoff) == (out["actual_total"] >= total_cutoff)
    margin_err = (out["actual_margin_home"] - out["predicted_margin_home"]).abs()
    total_err = (out["actual_total"] - out["predicted_total"]).abs()
    for t in [PRIMARY_THRESHOLD, SENSITIVITY_THRESHOLD]:
        suffix = int(t)
        out[f"margin_confirmed_{suffix}"] = (margin_same_side & (margin_err <= t)).astype(int)
        out[f"total_confirmed_{suffix}"] = (total_same_side & (total_err <= t)).astype(int)
    return out


def feature_columns(use_moneyline: bool) -> list[str]:
    cols = ["predicted_margin_home", "abs_spread", "predicted_total", "key_margin_distance"]
    if use_moneyline:
        cols.extend(["favorite_ml_prob", "spread_ml_consistency_resid"])
    return cols


def fit_label(
    x: pd.DataFrame,
    *,
    label_col: str,
    features: list[str],
    hypothesis: str,
    threshold: float,
) -> tuple[dict, pd.DataFrame]:
    cohort = x.copy()
    if hypothesis == "margin":
        cohort = cohort.loc[~cohort["predicted_margin_home"].eq(0.0)].copy()
    cohort = cohort.dropna(subset=features + [label_col]).copy()
    train = cohort.loc[cohort["season"].isin(TRAIN_SEASONS)].copy()
    test = cohort.loc[cohort["season"].eq(TEST_SEASON)].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"empty train/test for {label_col}")
    y_train = train[label_col].astype(int).to_numpy()
    y_test = test[label_col].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise RuntimeError(f"single-class train/test for {label_col}")

    model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, class_weight=None, random_state=42)),
    ])
    model.fit(train[features], y_train)
    train_p = model.predict_proba(train[features])[:, 1]
    test_p = model.predict_proba(test[features])[:, 1]
    selector_cutoff = float(np.quantile(train_p, 0.75))
    test_selected = test_p >= selector_cutoff

    prevalence = float(np.mean(y_train))
    auc = float(roc_auc_score(y_test, test_p))
    brier = float(brier_score_loss(y_test, test_p))
    baseline_brier = float(brier_score_loss(y_test, np.full(len(y_test), prevalence)))
    all_rate = float(np.mean(y_test))
    selected_n = int(np.sum(test_selected))
    selected_rate = float(np.mean(y_test[test_selected])) if selected_n else float("nan")
    lift = selected_rate - all_rate if selected_n else float("nan")
    positives = int(np.sum(y_test == 1))
    negatives = int(np.sum(y_test == 0))

    if np.isclose(threshold, PRIMARY_THRESHOLD):
        gates = {
            "auc_gt_055": auc > 0.55,
            "brier_better_than_prevalence": brier < baseline_brier,
            "selected_lift_ge_010": np.isfinite(lift) and lift >= 0.10,
            "test_class_support": positives >= MIN_CLASS_ROWS and negatives >= MIN_CLASS_ROWS,
        }
    else:
        gates = {
            "auc_gt_050": auc > 0.50,
            "selected_lift_gt_000": np.isfinite(lift) and lift > 0.0,
        }

    logit = model.named_steps["logit"]
    row = {
        "hypothesis": hypothesis,
        "label": label_col,
        "confirm_threshold": threshold,
        "features": "|".join(features),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_prevalence": prevalence,
        "test_prevalence": all_rate,
        "test_positive_rows": positives,
        "test_negative_rows": negatives,
        "auc": auc,
        "brier": brier,
        "prevalence_baseline_brier": baseline_brier,
        "selector_cutoff_train_q75": selector_cutoff,
        "selected_test_rows": selected_n,
        "selected_confirmation_rate": selected_rate,
        "selected_confirmation_lift": lift,
        "classification_gates_pass": bool(all(gates.values())),
        "gates_json": json.dumps(gates, sort_keys=True),
        "logit_intercept": float(logit.intercept_[0]),
        "logit_coefficients_json": json.dumps({f: float(v) for f, v in zip(features, logit.coef_[0])}, sort_keys=True),
    }

    pred = test[["season", "week", "game_id", "home_team", "away_team", label_col]].copy()
    pred["hypothesis"] = hypothesis
    pred["confirm_threshold"] = threshold
    pred["predicted_confirmation_probability"] = test_p
    pred["selector_cutoff_train_q75"] = selector_cutoff
    pred["selected"] = test_selected.astype(int)
    return row, pred


def effect_row(
    frame: pd.DataFrame,
    *,
    hypothesis: str,
    metric: str,
    total_cutoff: float,
    selected_game_ids: set[str] | None,
    eligible_game_ids: set[str],
    arm: str,
) -> dict:
    g = frame.loc[frame["season"].eq(TEST_SEASON)].copy()
    g = g.loc[g["game_id"].astype(str).isin(eligible_game_ids)].copy()
    if selected_game_ids is not None:
        g = g.loc[g["game_id"].astype(str).isin(selected_game_ids)].copy()

    if hypothesis == "margin":
        g = g.loc[~g["predicted_team_margin"].eq(0.0)].copy()
        if arm == "ground_truth":
            g = g.loc[~g["actual_team_margin"].eq(0.0)].copy()
            g["side"] = g["actual_team_margin"] > 0
        else:
            g["side"] = g["predicted_team_margin"] > 0
    elif hypothesis == "total":
        g["side"] = g["actual_total"] >= total_cutoff if arm == "ground_truth" else g["predicted_total"] >= total_cutoff
    else:
        raise ValueError(hypothesis)

    result = compare_by_side(g, side_col="side", metric_col=metric)
    return {"hypothesis": hypothesis, "metric": metric, "arm": arm, **result}


def downstream_summary(
    parent_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    class_summary: pd.DataFrame,
    total_cutoff: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    specs = [
        ("margin", "rb_rush_att", "rb_rush_yards"),
        ("total", "wrte_targets", "wrte_rec_yards"),
    ]
    for hypothesis, primary_metric, secondary_metric in specs:
        primary_class = class_summary.loc[
            class_summary["hypothesis"].eq(hypothesis) & class_summary["confirm_threshold"].eq(PRIMARY_THRESHOLD)
        ]
        if len(primary_class) != 1:
            raise RuntimeError(f"missing primary class row for {hypothesis}")
        class_pass = bool(primary_class.iloc[0]["classification_gates_pass"])

        p = predictions.loc[
            predictions["hypothesis"].eq(hypothesis) & predictions["confirm_threshold"].eq(PRIMARY_THRESHOLD)
        ].copy()
        eligible_ids = set(p["game_id"].astype(str))
        selected_ids = set(p.loc[p["selected"].eq(1), "game_id"].astype(str))

        for metric in [primary_metric, secondary_metric]:
            uncond = effect_row(parent_frame, hypothesis=hypothesis, metric=metric, total_cutoff=total_cutoff, selected_game_ids=None, eligible_game_ids=eligible_ids, arm="unconditional_vegas")
            selected = effect_row(parent_frame, hypothesis=hypothesis, metric=metric, total_cutoff=total_cutoff, selected_game_ids=selected_ids, eligible_game_ids=eligible_ids, arm="selected_high_confirmation")
            ground = effect_row(parent_frame, hypothesis=hypothesis, metric=metric, total_cutoff=total_cutoff, selected_game_ids=None, eligible_game_ids=eligible_ids, arm="ground_truth")

            uncond_d = uncond.get("cohens_d", np.nan)
            selected_d = selected.get("cohens_d", np.nan)
            selected_support = (
                selected.get("status") == "OK"
                and int(selected.get("n_high", 0)) >= MIN_DOWNSTREAM_ROWS_PER_SIDE
                and int(selected.get("n_low", 0)) >= MIN_DOWNSTREAM_ROWS_PER_SIDE
            )
            sharpen = np.isfinite(uncond_d) and np.isfinite(selected_d) and float(selected_d) > float(uncond_d)
            primary_gate = bool(class_pass and selected_support and sharpen) if metric == primary_metric else None
            for row in [uncond, selected, ground]:
                rows.append({
                    **row,
                    "is_primary_metric": metric == primary_metric,
                    "classification_pass": class_pass,
                    "selected_support_pass": selected_support,
                    "usage_sharpen_pass": sharpen,
                    "downstream_primary_gate_pass": primary_gate,
                })
    return pd.DataFrame(rows)


def disposition(class_summary: pd.DataFrame, downstream: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    for hypothesis in ["margin", "total"]:
        c7 = class_summary.loc[class_summary["hypothesis"].eq(hypothesis) & class_summary["confirm_threshold"].eq(PRIMARY_THRESHOLD)].iloc[0]
        c3 = class_summary.loc[class_summary["hypothesis"].eq(hypothesis) & class_summary["confirm_threshold"].eq(SENSITIVITY_THRESHOLD)].iloc[0]
        d = downstream.loc[
            downstream["hypothesis"].eq(hypothesis)
            & downstream["is_primary_metric"].eq(True)
            & downstream["arm"].eq("selected_high_confirmation")
        ].iloc[0]

        class_primary = bool(c7["classification_gates_pass"])
        sensitivity = bool(c3["classification_gates_pass"])
        downstream_pass = bool(d["downstream_primary_gate_pass"]) if pd.notna(d["downstream_primary_gate_pass"]) else False
        if not class_primary or not sensitivity:
            status = "NO_ACTIONABLE_PREGAME_CONFIRMATION_STATE"
        elif not downstream_pass:
            status = "PREDICTABLE_BUT_NOT_USEFUL_FOR_PLAYER_VOLUME"
        else:
            status = "QUALIFIED_PREGAME_CONFIRMATION_CANDIDATE"
        out[hypothesis] = {
            "status": status,
            "primary_classification_pass": class_primary,
            "t3_sensitivity_pass": sensitivity,
            "downstream_primary_pass": downstream_pass,
        }

    out["overall"] = {
        "status": "QUALIFIED_PREGAME_CONFIRMATION_CANDIDATE"
        if all(out[h]["status"] == "QUALIFIED_PREGAME_CONFIRMATION_CANDIDATE" for h in ["margin", "total"])
        else "NO_COMBINED_PROMOTION"
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    seasons = [*TRAIN_SEASONS, TEST_SEASON]
    games = load_game_outcomes(seasons)
    total_cutoff, parent_frame = total_cutoff_from_parent_frame(games)
    geometry = load_market_geometry(seasons)
    x, feature_meta = add_pregame_features(games, geometry)
    x = add_labels(x, total_cutoff)
    features = feature_columns(feature_meta["moneyline_features_used"])

    summaries: list[dict] = []
    predictions: list[pd.DataFrame] = []
    for hypothesis in ["margin", "total"]:
        for threshold in [PRIMARY_THRESHOLD, SENSITIVITY_THRESHOLD]:
            label = f"{hypothesis}_confirmed_{int(threshold)}"
            row, pred = fit_label(x, label_col=label, features=features, hypothesis=hypothesis, threshold=threshold)
            summaries.append(row)
            predictions.append(pred)

    class_summary = pd.DataFrame(summaries)
    pred_detail = pd.concat(predictions, ignore_index=True)
    downstream = downstream_summary(parent_frame, pred_detail, class_summary, total_cutoff)
    disp = disposition(class_summary, downstream)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "vegas_confirmation_likelihood_game_frame.csv", index=False)
    class_summary.to_csv(a.out_dir / "vegas_confirmation_likelihood_classification.csv", index=False)
    pred_detail.to_csv(a.out_dir / "vegas_confirmation_likelihood_2025_predictions.csv", index=False)
    downstream.to_csv(a.out_dir / "vegas_confirmation_likelihood_downstream.csv", index=False)
    result = {
        "train_seasons": list(TRAIN_SEASONS),
        "test_season": TEST_SEASON,
        "total_cutoff": total_cutoff,
        "features": features,
        "feature_meta": feature_meta,
        "disposition": disp,
    }
    (a.out_dir / "vegas_confirmation_likelihood_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("=== VEGAS_CONFIRMATION_LIKELIHOOD_V1 ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("\n-- classification --")
    print(class_summary.to_string(index=False))
    print("\n-- downstream --")
    print(downstream.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
