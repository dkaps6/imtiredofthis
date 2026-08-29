#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXCLUDED_FEATURES = {
    "coach_change",
    "coach_tenure_games",
}

TARGETS = {
    "high_volume_surprise": {
        "direction": 1,
        "paired_tail": "cat_under_100plus",
        "description": "actual attempts >= raw-attempt projection + 8",
    },
    "low_volume_surprise": {
        "direction": -1,
        "paired_tail": "cat_over_100plus",
        "description": "actual attempts <= raw-attempt projection - 8",
    },
}


def num(v):
    return pd.to_numeric(v, errors="coerce")


def read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def model_predictions(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> dict[str, np.ndarray]:
    Xtr = train[features]
    Xte = test[features]
    ytr = num(train[target]).astype(int)

    logistic = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=0.35,
            class_weight="balanced",
            max_iter=3000,
            random_state=63,
        ),
    )
    logistic.fit(Xtr, ytr)
    p_log = logistic.predict_proba(Xte)[:, 1]

    imp = SimpleImputer(strategy="median")
    A = imp.fit_transform(Xtr)
    B = imp.transform(Xte)
    pos = int(ytr.eq(1).sum())
    neg = int(ytr.eq(0).sum())
    weights = np.where(ytr.eq(1), max(neg / max(pos, 1), 1.0), 1.0)
    hist = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=180,
        max_leaf_nodes=12,
        min_samples_leaf=18,
        l2_regularization=4.0,
        random_state=63,
    )
    hist.fit(A, ytr, sample_weight=weights)
    p_hist = hist.predict_proba(B)[:, 1]
    return {"logistic": p_log, "hist_gb": p_hist}


def bootstrap_auc_ci(y, p, *, n_boot: int = 1000, seed: int = 63) -> tuple[float, float]:
    z = pd.DataFrame({"y": num(y), "p": num(p)}).dropna().reset_index(drop=True)
    if len(z) < 30 or z.y.nunique() < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    arr_y = z.y.to_numpy(int)
    arr_p = z.p.to_numpy(float)
    n = len(z)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = arr_y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(float(roc_auc_score(yy, arr_p[idx])))
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def metric_row(target: str, model: str, y, p) -> dict:
    z = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    if len(z) < 2 or z.y.nunique() < 2:
        return {
            "target": target,
            "model": model,
            "n": len(z),
            "roc_auc": np.nan,
            "auc_ci_low": np.nan,
            "auc_ci_high": np.nan,
            "average_precision": np.nan,
            "brier": np.nan,
        }
    auc = float(roc_auc_score(z.y, z.p))
    lo, hi = bootstrap_auc_ci(z.y, z.p)
    return {
        "target": target,
        "model": model,
        "n": len(z),
        "roc_auc": auc,
        "auc_ci_low": lo,
        "auc_ci_high": hi,
        "average_precision": float(average_precision_score(z.y, z.p)),
        "brier": float(brier_score_loss(z.y, z.p)),
    }


def bucket_table(test: pd.DataFrame, pred: np.ndarray, target: str, model: str) -> pd.DataFrame:
    z = test.copy()
    z["score"] = pred
    try:
        z["bucket"] = pd.qcut(
            z.score,
            5,
            labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
            duplicates="drop",
        )
    except Exception:
        z["bucket"] = "all"
    rows = []
    paired = TARGETS[target]["paired_tail"]
    for b, g in z.groupby("bucket", observed=True):
        rows.append(
            {
                "target": target,
                "model": model,
                "bucket": str(b),
                "n": len(g),
                "mean_score": float(g.score.mean()),
                "target_rate": float(num(g[target]).mean()),
                "mean_attempt_delta": float(num(g.attempt_delta).mean()),
                "mean_abs_attempt_error": float(num(g.attempt_abs_error).mean()),
                "pass_mae": float(num(g.abs_pass_error).mean()),
                "mean_signed_pass_error": float(num(g.pass_error).mean()),
                "cat_under_100plus_rate": float(num(g.cat_under_100plus).mean()),
                "cat_over_100plus_rate": float(num(g.cat_over_100plus).mean()),
                "paired_100plus_tail_rate": float(num(g[paired]).mean()),
            }
        )
    return pd.DataFrame(rows)


def directional_univariate_screen(games: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        base = {
            "feature": f,
            "coverage": float(num(games[f]).notna().mean()),
            "nunique": int(num(games[f]).nunique(dropna=True)),
        }
        for target in TARGETS:
            z = pd.DataFrame({"x": num(games[f]), "y": num(games[target])}).dropna()
            if len(z) >= 30 and z.x.nunique() >= 2 and z.y.nunique() == 2:
                a = float(roc_auc_score(z.y, z.x))
                auc = a if a >= 0.5 else 1.0 - a
                direction = "higher" if a >= 0.5 else "lower"
            else:
                auc, direction = np.nan, "none"
            base[f"{target}_auc"] = auc
            base[f"{target}_direction"] = direction
            diffs = []
            for s in (2024, 2025):
                q = games[games.season.eq(s)]
                pos = num(q.loc[q[target].eq(1), f]).mean()
                neg = num(q.loc[q[target].eq(0), f]).mean()
                d = pos - neg
                base[f"{target}_mean_diff_{s}"] = d
                diffs.append(d)
            replicate = bool(
                np.isfinite(diffs[0])
                and np.isfinite(diffs[1])
                and np.sign(diffs[0]) == np.sign(diffs[1])
                and np.sign(diffs[0]) != 0
            )
            base[f"{target}_direction_replicates"] = replicate
            base[f"{target}_strong_replicated"] = bool(replicate and np.isfinite(auc) and auc >= 0.58)
        rows.append(base)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-file", action="append", type=Path, required=True)
    ap.add_argument("--manifest-file", action="append", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    games = pd.concat([read(p) for p in args.season_file], ignore_index=True, sort=False)
    games["season"] = num(games.season).astype(int)
    games["actual_pass_att"] = num(games.actual_pass_att)
    games["attempts_raw"] = num(games.attempts_raw)
    games["attempt_delta"] = games.actual_pass_att - games.attempts_raw
    games["attempt_abs_error"] = games.attempt_delta.abs()
    games["high_volume_surprise"] = games.attempt_delta.ge(8).astype(int)
    games["low_volume_surprise"] = games.attempt_delta.le(-8).astype(int)
    games["high_volume_10plus"] = games.attempt_delta.ge(10).astype(int)
    games["low_volume_10plus"] = games.attempt_delta.le(-10).astype(int)

    if "cat_under_100plus" not in games:
        games["cat_under_100plus"] = num(games.pass_error).le(-100).astype(int)
    if "cat_over_100plus" not in games:
        games["cat_over_100plus"] = num(games.pass_error).ge(100).astype(int)

    manifests = [set(read(p).feature.astype(str)) for p in args.manifest_file]
    features = sorted(set.intersection(*manifests))
    features = [
        f
        for f in features
        if f in games
        and f not in EXCLUDED_FEATURES
        and num(games[f]).notna().mean() >= 0.50
        and num(games[f]).nunique(dropna=True) >= 2
    ]

    train = games[games.season.eq(2024)].copy()
    test = games[games.season.eq(2025)].copy()

    metrics_rows: list[dict] = []
    bucket_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    verdict_rows: list[dict] = []

    for target, spec in TARGETS.items():
        preds = model_predictions(train, test, features, target)
        base_rate = float(num(test[target]).mean())
        paired_base = float(num(test[spec["paired_tail"]]).mean())
        for model, p in preds.items():
            mr = metric_row(target, model, test[target], p)
            metrics_rows.append(mr)
            bt = bucket_table(test, p, target, model)
            bucket_frames.append(bt)
            hi = bt.sort_values("mean_score").iloc[-1]
            lo = bt.sort_values("mean_score").iloc[0]
            separation = float(hi.mean_attempt_delta - lo.mean_attempt_delta)
            directional_separation = separation * int(spec["direction"])
            top_enrichment = float(hi.target_rate / base_rate) if base_rate > 0 else np.nan
            paired_tail_enrichment = float(hi.paired_100plus_tail_rate / paired_base) if paired_base > 0 else np.nan
            actionable = bool(
                mr["roc_auc"] >= 0.60
                and mr["auc_ci_low"] > 0.50
                and top_enrichment >= 1.50
                and directional_separation >= 4.0
            )
            bridge = bool(np.isfinite(paired_tail_enrichment) and paired_tail_enrichment >= 1.25)
            verdict_rows.append(
                {
                    "target": target,
                    "model": model,
                    "description": spec["description"],
                    "2025_base_rate": base_rate,
                    "roc_auc": mr["roc_auc"],
                    "auc_ci_low": mr["auc_ci_low"],
                    "auc_ci_high": mr["auc_ci_high"],
                    "top_quintile_rate": float(hi.target_rate),
                    "top_quintile_enrichment": top_enrichment,
                    "q5_minus_q1_mean_attempt_delta": separation,
                    "directional_attempt_separation": directional_separation,
                    "paired_100plus_tail": spec["paired_tail"],
                    "paired_100plus_base_rate": paired_base,
                    "top_quintile_paired_100plus_rate": float(hi.paired_100plus_tail_rate),
                    "paired_100plus_enrichment": paired_tail_enrichment,
                    "passing_tail_bridge": bridge,
                    "m63_directional_regime_actionable": actionable,
                }
            )
            sc = test[
                [
                    "season",
                    "week",
                    "team",
                    "player_clean_key",
                    "actual",
                    "raw_projection",
                    "actual_pass_att",
                    "attempts_raw",
                    "attempt_delta",
                    "pass_error",
                    "abs_pass_error",
                    "cat_under_100plus",
                    "cat_over_100plus",
                    target,
                ]
            ].copy()
            sc["target"] = target
            sc["model"] = model
            sc["score"] = p
            score_frames.append(sc)

    metrics = pd.DataFrame(metrics_rows)
    buckets = pd.concat(bucket_frames, ignore_index=True)
    scores = pd.concat(score_frames, ignore_index=True)
    verdicts = pd.DataFrame(verdict_rows)

    screen = directional_univariate_screen(games, features)
    strong_cols = [f"{t}_strong_replicated" for t in TARGETS]
    replicated = screen[screen[strong_cols].any(axis=1)].copy()

    regime_counts = []
    for s, g in games.groupby("season"):
        regime_counts.append(
            {
                "season": int(s),
                "n": len(g),
                "high_volume_surprise_rate": float(g.high_volume_surprise.mean()),
                "low_volume_surprise_rate": float(g.low_volume_surprise.mean()),
                "high_volume_10plus_rate": float(g.high_volume_10plus.mean()),
                "low_volume_10plus_rate": float(g.low_volume_10plus.mean()),
                "mean_attempt_delta": float(g.attempt_delta.mean()),
                "attempt_delta_sd": float(g.attempt_delta.std(ddof=1)),
                "cat_under_100plus_rate": float(g.cat_under_100plus.mean()),
                "cat_over_100plus_rate": float(g.cat_over_100plus.mean()),
            }
        )
    regime_counts = pd.DataFrame(regime_counts)

    overall = pd.DataFrame(
        [
            {
                "any_directional_regime_actionable": bool(verdicts.m63_directional_regime_actionable.any()),
                "any_passing_tail_bridge": bool(verdicts.passing_tail_bridge.any()),
                "actionable_high_volume_models": int(
                    verdicts[
                        verdicts.target.eq("high_volume_surprise")
                        & verdicts.m63_directional_regime_actionable
                    ].shape[0]
                ),
                "actionable_low_volume_models": int(
                    verdicts[
                        verdicts.target.eq("low_volume_surprise")
                        & verdicts.m63_directional_regime_actionable
                    ].shape[0]
                ),
                "strong_replicated_directional_features": int(len(replicated)),
                "interpretation": (
                    "eligible_for_mixture_regime_prototype"
                    if verdicts.m63_directional_regime_actionable.any()
                    else "hold_directional_classifier_seek_new_information"
                ),
            }
        ]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    games.to_csv(args.out_dir / "m63_combined_directional_games.csv", index=False)
    regime_counts.to_csv(args.out_dir / "m63_regime_base_rates.csv", index=False)
    metrics.to_csv(args.out_dir / "m63_2024_to_2025_classifier_metrics.csv", index=False)
    buckets.to_csv(args.out_dir / "m63_2025_directional_quintiles.csv", index=False)
    scores.to_csv(args.out_dir / "m63_2025_oos_directional_scores.csv", index=False)
    screen.to_csv(args.out_dir / "m63_directional_feature_screen.csv", index=False)
    replicated.to_csv(args.out_dir / "m63_replicated_directional_features.csv", index=False)
    verdicts.to_csv(args.out_dir / "m63_precommitted_model_verdicts.csv", index=False)
    overall.to_csv(args.out_dir / "m63_precommitted_interpretation.csv", index=False)

    print("=== M63 REGIME BASE RATES ===")
    print(regime_counts.to_string(index=False))
    print("\n=== M63 2024->2025 CLASSIFIERS ===")
    print(metrics.to_string(index=False))
    print("\n=== M63 MODEL VERDICTS ===")
    print(verdicts.to_string(index=False))
    print("\n=== M63 TOP REPLICATED DIRECTIONAL FEATURES ===")
    if replicated.empty:
        print("none")
    else:
        print(replicated.head(30).to_string(index=False))
    print("\n=== M63 INTERPRETATION ===")
    print(overall.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
