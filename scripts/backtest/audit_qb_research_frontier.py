#!/usr/bin/env python3
"""Migration 66: QB research frontier / model-diversity audit.

Diagnostic only. This migration does not create or promote a new football model.
It asks whether the information already produced by the QB research program is
redundant, complementary, or still contains leakage-safe residual signal.

Frozen evaluation policy:
- 2024 and 2025 are the only scored target seasons.
- 2023/2022 are not rescored.
- learned ensemble/residual models fit on 2024 only and evaluate on 2025 only.
- no sportsbook player-prop line is a feature.
- no post-hoc candidate/subset search.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest.audit_qb_extreme_error_regimes import (
    ATTEMPT_HISTORY_FEATURES,
    RAW_MODEL_FEATURES,
    SAFE_MARKET_FEATURES,
    SITUATION_FEATURES,
)
from scripts.backtest.validate_qb_dropback_state_occupancy import FEATURES as M65_FEATURES

# Representative, materially distinct point forecasts from the dedicated QB
# program that are available on one paired canonical trace. Raw appears once.
CANDIDATE_LIBRARY = {
    "joint_cap_shrink": "mc_proj_joint_cap_shrink",
    "raw_attempts": "mc_proj_attempts_raw_only",
    "raw_ypa": "mc_proj_ypa_raw_only",
    "both_raw": "mc_proj_both_raw",
    "raw_point_product": "m64_pass_raw_point_product",
    "m64_generative_neutral": "m64_pass_generative_neutral",
    "m64_generative_gamescript": "m64_pass_generative_gamescript",
    "m65_state_formula": "m65_pass_state_formula",
    "m65_state_ridge": "m65_pass_state_ridge",
}

M64_SAFE_FEATURES = [
    "m64_pred_drives",
    "m64_pred_plays_per_drive",
    "m64_pred_dropback_rate_neutral",
    "m64_pred_dropback_rate_gamescript",
    "m64_pred_attempt_conversion",
    "m64_pred_qb_attempt_share",
    "m64_market_implied_win_prob",
    "m64_team_recent_drives",
    "m64_opp_recent_drives",
    "m64_team_def_drives_allowed",
    "m64_opp_def_drives_allowed",
    "m64_team_recent_no_huddle",
    "m64_team_recent_seconds_between_plays",
    "m64_opp_recent_scoring_drive_rate",
]
M65_SAFE_PREDICTIONS = [
    "m65_pred_neutral_share",
    "m65_pred_trailing_share",
    "m65_pred_leading_share",
    "m65_formula_neutral_share",
    "m65_formula_trailing_share",
    "m65_formula_leading_share",
    "m65_pred_neutral_dropback_rate",
    "m65_pred_trailing_dropback_rate",
    "m65_pred_leading_dropback_rate",
    "m65_pred_dropback_rate",
    "m65_formula_dropback_rate",
]
INVALID_INFERENCE_FEATURES = {"coach_change", "coach_tenure_games"}


def num(v):
    return pd.to_numeric(v, errors="coerce")


def read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def metrics(actual, pred) -> dict:
    z = pd.DataFrame({"a": num(actual), "p": num(pred)}).dropna()
    if z.empty:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "corr": np.nan,
            "miss_75plus": 0,
            "miss_100plus": 0,
        }
    e = z.p - z.a
    return {
        "n": int(len(z)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "bias": float(e.mean()),
        "corr": float(z.a.corr(z.p)) if len(z) >= 2 else np.nan,
        "miss_75plus": int(e.abs().ge(75).sum()),
        "miss_100plus": int(e.abs().ge(100).sum()),
    }


def paired_library(g: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    present = {name: col for name, col in CANDIDATE_LIBRARY.items() if col in g.columns}
    if len(present) < 6:
        raise RuntimeError(f"M66 expected >=6 paired QB candidates; found {present}")
    keep = ["actual", "season", "week", "team", "player_clean_key"] + list(present.values())
    x = g[keep].copy()
    for c in ["actual"] + list(present.values()):
        x[c] = num(x[c])
    x = x.dropna(subset=["actual"] + list(present.values())).copy()
    if len(x) < 500:
        raise RuntimeError(f"M66 paired model library unexpectedly small: {len(x)}")
    return x, present


def library_metrics(x: pd.DataFrame, library: dict[str, str]) -> pd.DataFrame:
    rows = []
    for season_label, q in [("2024", x[num(x.season).eq(2024)]), ("2025", x[num(x.season).eq(2025)]), ("combined", x)]:
        for name, col in library.items():
            rows.append({"season": season_label, "candidate": name, **metrics(q.actual, q[col])})
    return pd.DataFrame(rows)


def corr_outputs(x: pd.DataFrame, library: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pd.DataFrame({name: num(x[col]) for name, col in library.items()})
    residual = pd.DataFrame({name: num(x[col]) - num(x.actual) for name, col in library.items()})
    pc = pred.corr()
    rc = residual.corr()
    vals = []
    names = list(rc.columns)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            vals.append({"candidate_a": a, "candidate_b": b, "residual_corr": float(rc.loc[a, b]), "abs_residual_corr": abs(float(rc.loc[a, b]))})
    pairs = pd.DataFrame(vals)
    return pc, rc, pairs


def oracle_audit(x: pd.DataFrame, library: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preds = pd.DataFrame({name: num(x[col]) for name, col in library.items()}, index=x.index)
    abs_err = preds.sub(num(x.actual), axis=0).abs()
    winner = abs_err.idxmin(axis=1)
    oracle = pd.Series([preds.loc[i, winner.loc[i]] for i in x.index], index=x.index)
    base = num(x[library["raw_attempts"]])
    out = x[["season", "week", "team", "player_clean_key", "actual"]].copy()
    out["oracle_candidate"] = winner
    out["oracle_projection"] = oracle
    out["raw_projection"] = base
    out["library_disagreement_sd"] = preds.std(axis=1, ddof=0)
    out["raw_abs_error"] = (base - num(x.actual)).abs()
    out["oracle_abs_error"] = (oracle - num(x.actual)).abs()

    rows = []
    for season_label, q in [("2024", out[num(out.season).eq(2024)]), ("2025", out[num(out.season).eq(2025)]), ("combined", out)]:
        om = metrics(q.actual, q.oracle_projection)
        rm = metrics(q.actual, q.raw_projection)
        rows.append({
            "season": season_label,
            "oracle_mae": om["mae"],
            "oracle_rmse": om["rmse"],
            "oracle_corr": om["corr"],
            "oracle_100plus": om["miss_100plus"],
            "raw_mae": rm["mae"],
            "raw_corr": rm["corr"],
            "oracle_mae_headroom_vs_raw": rm["mae"] - om["mae"],
        })
    summary = pd.DataFrame(rows)
    winners = out.groupby(["season", "oracle_candidate"], dropna=False).size().rename("wins").reset_index()

    # Model disagreement as a prospective uncertainty clue; quintiles are diagnostic only.
    rank = out.library_disagreement_sd.rank(method="first")
    out["disagreement_quintile"] = pd.qcut(rank, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    disagreement = out.groupby("disagreement_quintile", observed=True).agg(
        n=("actual", "size"),
        mean_disagreement=("library_disagreement_sd", "mean"),
        raw_mae=("raw_abs_error", "mean"),
        raw_100plus_rate=("raw_abs_error", lambda s: float((s >= 100).mean())),
        oracle_mae=("oracle_abs_error", "mean"),
    ).reset_index()
    return summary, winners, disagreement


def fit_convex_mae(train: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = train[cols].to_numpy(dtype=float)
    y = num(train.actual).to_numpy(dtype=float)
    n, k = X.shape
    # variables = k nonnegative weights + n absolute-error slack variables.
    c = np.r_[np.zeros(k), np.ones(n)]
    # Xw - t <= y ; -Xw - t <= -y
    A_ub = np.vstack([
        np.c_[X, -np.eye(n)],
        np.c_[-X, -np.eye(n)],
    ])
    b_ub = np.r_[y, -y]
    A_eq = np.zeros((1, k + n)); A_eq[0, :k] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * k + [(0.0, None)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"M66 convex MAE blend failed: {res.message}")
    return np.asarray(res.x[:k], dtype=float)


def ensemble_audit(x: pd.DataFrame, library: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = list(library)
    cols = [library[n] for n in names]
    P = x[cols].astype(float)
    static = pd.DataFrame(index=x.index)
    static["mean"] = P.mean(axis=1)
    static["median"] = P.median(axis=1)
    if len(cols) >= 5:
        arr = np.sort(P.to_numpy(dtype=float), axis=1)
        static["trimmed_mean"] = arr[:, 1:-1].mean(axis=1)
    else:
        static["trimmed_mean"] = static["mean"]

    rows = []
    for season_label, qidx in [("2024", x.index[num(x.season).eq(2024)]), ("2025", x.index[num(x.season).eq(2025)]), ("combined", x.index)]:
        for name in static.columns:
            rows.append({"season": season_label, "ensemble": name, "training": "none", **metrics(x.loc[qidx, "actual"], static.loc[qidx, name])})

    train = x[num(x.season).eq(2024)].copy()
    test = x[num(x.season).eq(2025)].copy()
    weights = fit_convex_mae(train, cols)
    convex_train = train[cols].to_numpy(dtype=float) @ weights
    convex_test = test[cols].to_numpy(dtype=float) @ weights
    rows.append({"season": "2024_train", "ensemble": "convex_mae_2024fit", "training": "2024", **metrics(train.actual, convex_train)})
    rows.append({"season": "2025_test", "ensemble": "convex_mae_2024fit", "training": "2024", **metrics(test.actual, convex_test)})

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    ridge.fit(train[cols], num(train.actual))
    ridge_train = ridge.predict(train[cols])
    ridge_test = ridge.predict(test[cols])
    rows.append({"season": "2024_train", "ensemble": "ridge_stack_2024fit", "training": "2024", **metrics(train.actual, ridge_train)})
    rows.append({"season": "2025_test", "ensemble": "ridge_stack_2024fit", "training": "2024", **metrics(test.actual, ridge_test)})

    weight_table = pd.DataFrame({"candidate": names, "convex_mae_weight": weights})
    predictions = test[["season", "week", "team", "player_clean_key", "actual"]].copy()
    predictions["raw_attempts"] = num(test[library["raw_attempts"]])
    predictions["convex_mae_2024fit"] = convex_test
    predictions["ridge_stack_2024fit"] = ridge_test
    return pd.DataFrame(rows), weight_table, predictions


def merge_feature_universe(g: pd.DataFrame, state_features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    sf = state_features.copy()
    sf["season"] = num(sf.season).astype(int)
    sf["week"] = num(sf.week).astype(int)
    keep = ["season", "week", "team"] + [c for c in M65_FEATURES if c in sf]
    sf = sf[keep].drop_duplicates(["season", "week", "team"])
    rename = {c: f"m65f_{c}" for c in keep if c not in {"season", "week", "team"}}
    sf = sf.rename(columns=rename)
    x = g.merge(sf, on=["season", "week", "team"], how="left", validate="many_to_one")

    base = []
    for c in SAFE_MARKET_FEATURES + RAW_MODEL_FEATURES + ATTEMPT_HISTORY_FEATURES + SITUATION_FEATURES + M64_SAFE_FEATURES + M65_SAFE_PREDICTIONS:
        if c in x.columns and c not in INVALID_INFERENCE_FEATURES and c not in base:
            base.append(c)
    base.extend([c for c in x.columns if c.startswith("m65f_")])
    # Candidate predictions are legitimate pregame outputs and are intentionally
    # included: this is an information-frontier audit, not a new production fit.
    base.extend([col for col in CANDIDATE_LIBRARY.values() if col in x.columns and col not in base])
    usable = []
    for c in base:
        x[c] = num(x[c])
        tr = x[num(x.season).eq(2024)][c]
        if tr.notna().sum() >= 100 and tr.nunique(dropna=True) > 1:
            usable.append(c)
    return x, usable


def residual_audit(g: pd.DataFrame, features: list[str], raw_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = g[num(g.season).eq(2024)].copy()
    test = g[num(g.season).eq(2025)].copy()
    train["target_residual"] = num(train.actual) - num(train[raw_col])
    test["target_residual"] = num(test.actual) - num(test[raw_col])

    models = {
        "ridge_residual": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=50.0),
        ),
        "histgb_residual": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                loss="absolute_error",
                max_iter=150,
                learning_rate=0.04,
                max_depth=2,
                min_samples_leaf=15,
                l2_regularization=5.0,
                random_state=66,
            ),
        ),
    }
    rows = []
    preds = test[["season", "week", "team", "player_clean_key", "actual"]].copy()
    preds["raw_projection"] = num(test[raw_col])
    raw_m = metrics(test.actual, test[raw_col])
    for name, model in models.items():
        model.fit(train[features], train.target_residual)
        pred_resid = pd.Series(model.predict(test[features]), index=test.index)
        corrected = num(test[raw_col]) + pred_resid
        resid_corr = float(pred_resid.corr(test.target_residual)) if len(test) >= 2 else np.nan
        resid_mae = float((pred_resid - test.target_residual).abs().mean())
        cm = metrics(test.actual, corrected)
        rows.append({
            "model": name,
            "train_season": 2024,
            "test_season": 2025,
            "feature_count": len(features),
            "residual_corr": resid_corr,
            "residual_mae": resid_mae,
            "corrected_mae": cm["mae"],
            "corrected_rmse": cm["rmse"],
            "corrected_bias": cm["bias"],
            "corrected_corr": cm["corr"],
            "corrected_100plus": cm["miss_100plus"],
            "raw_mae": raw_m["mae"],
            "raw_corr": raw_m["corr"],
            "mae_gain_vs_raw": raw_m["mae"] - cm["mae"],
            "corr_gain_vs_raw": cm["corr"] - raw_m["corr"],
        })
        preds[f"{name}_residual"] = pred_resid.to_numpy()
        preds[f"{name}_corrected"] = corrected.to_numpy()
    return pd.DataFrame(rows), preds


def mean_median_metrics(files: list[Path]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    z = pd.concat([read(p) for p in files], ignore_index=True)
    rows = []
    for season_label, q in [("2024", z[num(z.season).eq(2024)]), ("2025", z[num(z.season).eq(2025)]), ("combined", z)]:
        for mode in ("current", "raw"):
            for stat in ("mean", "median"):
                col = f"{mode}_{stat}"
                if col in q:
                    rows.append({"season": season_label, "mode": mode, "point_stat": stat, **metrics(q.actual, q[col])})
    return pd.DataFrame(rows)


def interpretation(
    library_metrics_df: pd.DataFrame,
    pairs: pd.DataFrame,
    oracle: pd.DataFrame,
    ensembles: pd.DataFrame,
    residuals: pd.DataFrame,
    meanmedian: pd.DataFrame,
) -> pd.DataFrame:
    raw = library_metrics_df[(library_metrics_df.season.eq("2025")) & library_metrics_df.candidate.eq("raw_attempts")].iloc[0]
    median_abs_resid_corr = float(pairs.abs_residual_corr.median()) if len(pairs) else np.nan
    oracle_combined = oracle[oracle.season.eq("combined")].iloc[0]
    models_materially_diverse = bool(np.isfinite(median_abs_resid_corr) and median_abs_resid_corr <= 0.85)
    oracle_headroom = float(oracle_combined.oracle_mae_headroom_vs_raw)

    eligible_ensembles = []
    for _, r in ensembles.iterrows():
        if str(r.season) not in {"2025", "2025_test"}:
            continue
        if (
            float(raw.mae - r.mae) >= 1.0
            and float(r["corr"] - raw["corr"]) >= 0.03
            and float(r.rmse) <= float(raw.rmse) + 1e-12
            and int(r.miss_100plus) <= int(np.floor(float(raw.miss_100plus) * 0.95))
        ):
            eligible_ensembles.append(str(r.ensemble))

    residual_signal_models = []
    for _, r in residuals.iterrows():
        if (
            float(r.residual_corr) >= 0.20
            and float(r.mae_gain_vs_raw) >= 1.0
            and float(r.corr_gain_vs_raw) >= 0.03
        ):
            residual_signal_models.append(str(r.model))

    mean_median_signal = False
    raw_median_gain = np.nan
    if not meanmedian.empty:
        mm = meanmedian[(meanmedian.season.eq("2025")) & meanmedian.mode.eq("raw")]
        if {"mean", "median"}.issubset(set(mm.point_stat)):
            a = mm[mm.point_stat.eq("mean")].iloc[0]
            b = mm[mm.point_stat.eq("median")].iloc[0]
            raw_median_gain = float(a.mae - b.mae)
            mean_median_signal = raw_median_gain >= 0.50

    existing_combination_signal = bool(eligible_ensembles or residual_signal_models or mean_median_signal)
    if existing_combination_signal:
        verdict = "existing_information_combination_followup"
    elif (not models_materially_diverse) and not residual_signal_models:
        verdict = "current_library_redundant_seek_new_information"
    else:
        verdict = "mixed_frontier_seek_new_information_and_selective_combination"

    return pd.DataFrame([{
        "paired_candidates": int(library_metrics_df[library_metrics_df.season.eq("combined")].candidate.nunique()),
        "median_abs_pairwise_residual_corr": median_abs_resid_corr,
        "models_materially_diverse_le_0_85": models_materially_diverse,
        "oracle_mae_headroom_vs_raw": oracle_headroom,
        "oracle_headroom_ge_8yd": bool(oracle_headroom >= 8.0),
        "ensemble_followup_eligible": bool(eligible_ensembles),
        "eligible_ensembles": "|".join(eligible_ensembles),
        "residual_information_signal": bool(residual_signal_models),
        "residual_signal_models": "|".join(residual_signal_models),
        "raw_median_mae_gain_2025": raw_median_gain,
        "mean_median_signal_ge_0_50": mean_median_signal,
        "m66_interpretation": verdict,
    }])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m65-game-level", type=Path, required=True)
    ap.add_argument("--m65-state-features", type=Path, required=True)
    ap.add_argument("--mean-median-file", type=Path, action="append", default=[])
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    g = read(a.m65_game_level)
    g["season"] = num(g.season).astype(int)
    g["week"] = num(g.week).astype(int)
    state_features = read(a.m65_state_features)

    paired, library = paired_library(g)
    lib_metrics = library_metrics(paired, library)
    pcorr, rcorr, pairs = corr_outputs(paired, library)
    oracle, winners, disagreement = oracle_audit(paired, library)
    ensembles, weights, ensemble_preds = ensemble_audit(paired, library)

    full, features = merge_feature_universe(g, state_features)
    # Keep the same complete paired population for residual evaluation.
    keys = paired[["season", "week", "team", "player_clean_key"]].copy()
    full = full.merge(keys.assign(_m66_keep=1), on=["season", "week", "team", "player_clean_key"], how="inner")
    residuals, residual_preds = residual_audit(full, features, CANDIDATE_LIBRARY["raw_attempts"])
    mm = mean_median_metrics(a.mean_median_file)
    verdict = interpretation(lib_metrics, pairs, oracle, ensembles, residuals, mm)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    lib_metrics.to_csv(a.out_dir / "m66_candidate_library_metrics.csv", index=False)
    pcorr.to_csv(a.out_dir / "m66_prediction_correlation_matrix.csv")
    rcorr.to_csv(a.out_dir / "m66_residual_correlation_matrix.csv")
    pairs.to_csv(a.out_dir / "m66_residual_pairwise.csv", index=False)
    oracle.to_csv(a.out_dir / "m66_oracle_library_summary.csv", index=False)
    winners.to_csv(a.out_dir / "m66_oracle_winner_counts.csv", index=False)
    disagreement.to_csv(a.out_dir / "m66_model_disagreement_quintiles.csv", index=False)
    ensembles.to_csv(a.out_dir / "m66_ensemble_metrics.csv", index=False)
    weights.to_csv(a.out_dir / "m66_convex_ensemble_weights.csv", index=False)
    ensemble_preds.to_csv(a.out_dir / "m66_ensemble_2025_predictions.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(a.out_dir / "m66_residual_feature_manifest.csv", index=False)
    residuals.to_csv(a.out_dir / "m66_residual_predictability.csv", index=False)
    residual_preds.to_csv(a.out_dir / "m66_residual_2025_predictions.csv", index=False)
    mm.to_csv(a.out_dir / "m66_mean_median_metrics.csv", index=False)
    verdict.to_csv(a.out_dir / "m66_precommitted_interpretation.csv", index=False)

    print("=== M66 PRECOMMITTED INTERPRETATION ===")
    print(verdict.to_string(index=False))
    print("\n=== CANDIDATE LIBRARY ===")
    print(lib_metrics.to_string(index=False))
    print("\n=== RESIDUAL DIVERSITY SUMMARY ===")
    print(pairs.sort_values("abs_residual_corr").to_string(index=False))
    print("\n=== ORACLE LIBRARY ===")
    print(oracle.to_string(index=False))
    print("\n=== ENSEMBLES ===")
    print(ensembles.to_string(index=False))
    print("\n=== RESIDUAL PREDICTABILITY ===")
    print(residuals.to_string(index=False))
    print("\n=== MEAN VS MEDIAN ===")
    print(mm.to_string(index=False) if not mm.empty else "not supplied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
