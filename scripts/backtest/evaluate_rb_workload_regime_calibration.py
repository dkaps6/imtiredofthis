"""M95F: calibrated RB workload-regime / hurdle distribution.

Research-only. Preserve M94C as the central carry estimate. M95E showed that
20+/25+ workload-state ranking is strong, but its class-balanced raw scores and
beta-binomial distribution are badly over-broad. M95F therefore:

1. freezes the M95E tail scorer family (logit);
2. learns leakage-safe calibration from temporal 2024 out-of-fold scores;
3. selects compact score-only vs football-aware calibration on 2024 W13-18;
4. refits the chosen calibration on all eligible 2024 temporal OOF scores;
5. validates exactly once on untouched 2025;
6. expresses tail mass through an empirical hurdle mixture while leaving M94C
   as the official central carry estimate.

No sportsbook input and no production code changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import scripts.backtest.evaluate_rb_absolute_workload_distribution_v3 as v3

m = v3.m

SEED = 9591
RAW_FAMILY = "logit"
OOF_START_WEEK = 5
CAL_FAMILIES = ("platt", "beta", "football")
THRESHOLDS = (
    0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10,
    0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
)
MAX_FLAG_MULTIPLE = 3.0
N_SIM = 4000

FOOTBALL_CAL_FEATURES = [
    "_score_logit",
    "prior_expected_carries",
    "prior_abs_share",
    "role_is_workhorse",
    "role_is_starter_plus",
    "carry_trend_1v5",
    "share_trend_1v5",
    "team_rb_used_avg3",
    "team_top1_share_avg3",
    "team_qb_rush_share_avg3",
    "team_pbp_neutral_rush_rate_avg3",
    "def_rb_carries_allowed_avg3",
    "late_week_17plus",
    "week18",
    "rb_history_missing",
]


def num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def col(x: pd.DataFrame, name: str) -> pd.Series:
    if name in x.columns:
        return num(x[name])
    return pd.Series(np.nan, index=x.index, dtype=float)


def add_cal_features(x: pd.DataFrame, score_col: str = "raw_score") -> pd.DataFrame:
    z = x.copy()
    p = col(z, score_col).clip(1e-5, 1 - 1e-5)
    z["_score_logit"] = np.log(p / (1.0 - p))
    z["_score_log"] = np.log(p)
    z["_one_minus_score_log"] = -np.log(1.0 - p)
    z["late_week_17plus"] = col(z, "week").ge(17).astype(int)
    z["week18"] = col(z, "week").eq(18).astype(int)
    z["carry_trend_1v5"] = col(z, "rb_carries_avg1") - col(z, "rb_carries_avg5")
    z["share_trend_1v5"] = col(z, "rb_rb_share_avg1") - col(z, "rb_rb_share_avg5")
    z["rb_history_missing"] = col(z, "rb_games_before").isna().astype(int)
    return z


def raw_tail_score(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    feats = m.available(train, m.TAIL_FEATURES)
    if not feats:
        raise RuntimeError("M95F tail feature set empty")
    fit = train.loc[train[target].notna()].copy()
    if fit[target].nunique() < 2:
        raise RuntimeError(f"M95F {target} training set has one class")
    model = m.classifiers(SEED)[RAW_FAMILY]
    model.fit(fit[feats], fit[target].astype(int))
    return model.predict_proba(test[feats])[:, 1]


def temporal_oof_2024(trace: pd.DataFrame, target: str, end_week: int = 18) -> pd.DataFrame:
    pieces = []
    for week in range(OOF_START_WEEK, end_week + 1):
        train = trace.loc[
            trace["season"].eq(2023)
            | (trace["season"].eq(2024) & num(trace["week"]).lt(week))
        ].copy()
        test = trace.loc[
            trace["season"].eq(2024) & num(trace["week"]).eq(week)
        ].copy()
        if test.empty or train[target].nunique() < 2:
            continue
        out = test.copy()
        out["raw_score"] = raw_tail_score(train, test, target)
        out["target"] = target
        pieces.append(out)
    if not pieces:
        raise RuntimeError(f"M95F could not generate temporal OOF scores for {target}")
    return pd.concat(pieces, ignore_index=True)


def calibration_pipeline(family: str) -> Pipeline:
    c = 1.0 if family in {"platt", "beta"} else 0.12
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=2500, random_state=SEED)),
    ])


def calibration_columns(family: str) -> list[str]:
    if family == "platt":
        return ["_score_logit"]
    if family == "beta":
        return ["_score_log", "_one_minus_score_log"]
    if family == "football":
        return FOOTBALL_CAL_FEATURES
    raise ValueError(family)


def fit_calibrator(train: pd.DataFrame, family: str) -> Pipeline:
    z = add_cal_features(train)
    y = z["actual_label"].astype(int)
    if y.nunique() < 2:
        raise RuntimeError("M95F calibrator training set has one class")
    pipe = calibration_pipeline(family)
    pipe.fit(z[calibration_columns(family)], y)
    return pipe


def apply_calibrator(model: Pipeline, x: pd.DataFrame, family: str) -> np.ndarray:
    z = add_cal_features(x)
    return np.clip(
        model.predict_proba(z[calibration_columns(family)])[:, 1],
        1e-6,
        1 - 1e-6,
    )


def ece(y, p, bins: int = 10) -> float:
    z = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    if z.empty:
        return np.nan
    try:
        z["bin"] = pd.qcut(z["p"], bins, duplicates="drop")
    except ValueError:
        return abs(float(z["p"].mean() - z["y"].mean()))
    total = len(z)
    out = 0.0
    for _, g in z.groupby("bin", observed=True):
        out += len(g) / total * abs(float(g["p"].mean() - g["y"].mean()))
    return float(out)


def prob_metrics(y, p) -> dict[str, float | int]:
    z = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    if z.empty:
        return {
            "n": 0, "base_rate": np.nan, "mean_prob": np.nan,
            "auc": np.nan, "brier": np.nan, "logloss": np.nan, "ece": np.nan,
        }
    yy = z["y"].astype(int)
    pp = z["p"].clip(1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(yy, pp)) if yy.nunique() > 1 else np.nan
    return {
        "n": int(len(z)),
        "base_rate": float(yy.mean()),
        "mean_prob": float(pp.mean()),
        "auc": auc,
        "brier": float(np.mean(np.square(pp - yy))),
        "logloss": float(log_loss(yy, pp, labels=[0, 1])),
        "ece": ece(yy, pp),
    }


def threshold_grid(y, p) -> tuple[float, pd.DataFrame]:
    yy = num(y).astype(int)
    actual_pos = int(yy.sum())
    rows = []
    for t in THRESHOLDS:
        pred = num(pd.Series(p, index=yy.index)).ge(t)
        truth = yy.eq(1)
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        flagged = int(pred.sum())
        limit = max(int(np.ceil(MAX_FLAG_MULTIPLE * actual_pos)), actual_pos + 5)
        rows.append({
            "threshold": t,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_positive": flagged,
            "actual_positive": actual_pos,
            "flag_multiple": flagged / max(actual_pos, 1),
            "eligible": int(flagged <= limit),
        })
    grid = pd.DataFrame(rows)
    pool = grid.loc[grid["eligible"].eq(1)].copy()
    if pool.empty:
        pool = grid.copy()
    pool = pool.sort_values(
        ["f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return float(pool.iloc[0]["threshold"]), grid


def holdout_score_frame(
    trace: pd.DataFrame,
    hold: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = num(trace["week"])
    train = trace.loc[
        trace["season"].eq(2023)
        | (trace["season"].eq(2024) & w.le(12))
    ].copy()
    out = hold.copy()
    out["raw_score"] = raw_tail_score(train, out, target)
    out["actual_label"] = out[target].astype(int)

    oof = temporal_oof_2024(trace, target, end_week=12)
    oof["actual_label"] = oof[target].astype(int)
    return oof, out


def select_calibrator(
    oof_train: pd.DataFrame,
    hold: pd.DataFrame,
    target: str,
) -> tuple[str, pd.DataFrame, float]:
    rows = []
    calibrated: dict[str, np.ndarray] = {}
    rows.append({
        "target": target,
        "family": "raw_class_balanced_score",
        **prob_metrics(hold["actual_label"], hold["raw_score"]),
    })
    const = np.full(len(hold), float(oof_train["actual_label"].mean()))
    rows.append({
        "target": target,
        "family": "constant_prior",
        **prob_metrics(hold["actual_label"], const),
    })

    for family in CAL_FAMILIES:
        model = fit_calibrator(oof_train, family)
        p = apply_calibrator(model, hold, family)
        calibrated[family] = p
        rows.append({
            "target": target,
            "family": family,
            **prob_metrics(hold["actual_label"], p),
        })

    audit = pd.DataFrame(rows)
    candidates = audit.loc[audit["family"].isin(CAL_FAMILIES)].copy()
    candidates = candidates.sort_values(
        ["brier", "ece", "logloss", "family"]
    ).reset_index(drop=True)
    chosen = str(candidates.iloc[0]["family"])
    threshold, grid = threshold_grid(hold["actual_label"], calibrated[chosen])
    grid.insert(0, "target", target)
    grid.insert(1, "family", chosen)
    return chosen, pd.concat([audit, grid], ignore_index=True, sort=False), threshold


def final_calibrated_probs(
    trace: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    family: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    train = trace.loc[trace["season"].isin([2023, 2024])].copy()
    raw = raw_tail_score(train, test, target)

    oof_all = temporal_oof_2024(trace, target, end_week=18)
    oof_all["actual_label"] = oof_all[target].astype(int)
    cal = fit_calibrator(oof_all, family)
    z = test.copy()
    z["raw_score"] = raw
    z["actual_label"] = z[target].astype(int)
    calibrated = apply_calibrator(cal, z, family)
    return raw, calibrated, oof_all


def empirical_state_pools(train: pd.DataFrame) -> dict[str, np.ndarray]:
    a = num(train["actual_carries"])
    prior = num(train["prior_expected_carries"])
    normal = a.lt(20) & a.notna() & prior.notna()
    residual = (a.loc[normal] - prior.loc[normal]).to_numpy(dtype=float)
    if residual.size < 30:
        residual = np.array([0.0])
    residual = residual - np.nanmean(residual)

    high = a.loc[a.between(20, 24)].dropna().to_numpy(dtype=float)
    extreme = a.loc[a.ge(25)].dropna().to_numpy(dtype=float)
    if high.size < 10 or extreme.size < 5:
        raise RuntimeError("M95F state pools too small")
    return {
        "normal_residual": residual,
        "high20_24": high,
        "extreme25": extreme,
    }


def simulate_hurdle(
    x: pd.DataFrame,
    pools: dict[str, np.ndarray],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in x.iterrows():
        p20 = float(np.clip(r["cal_prob_20"], 0, 1))
        p25 = float(np.clip(min(r["cal_prob_25"], p20), 0, 1))
        p_mid = max(p20 - p25, 0.0)
        base = float(r["m94c_rush_att"])

        u = rng.random(N_SIM)
        draws = np.empty(N_SIM, dtype=float)
        extreme = u < p25
        high = (u >= p25) & (u < p25 + p_mid)
        normal = ~(extreme | high)

        if normal.any():
            resid = rng.choice(
                pools["normal_residual"], size=int(normal.sum()), replace=True
            )
            draws[normal] = np.clip(base + resid, 0.0, 19.99)
        if high.any():
            draws[high] = rng.choice(
                pools["high20_24"], size=int(high.sum()), replace=True
            )
        if extreme.any():
            draws[extreme] = rng.choice(
                pools["extreme25"], size=int(extreme.sum()), replace=True
            )

        rows.append({
            **{k: r[k] for k in m.PLAYER_KEYS},
            "m95f_mix_mean": float(draws.mean()),
            "m95f_p50": float(np.quantile(draws, 0.50)),
            "m95f_p75": float(np.quantile(draws, 0.75)),
            "m95f_p90": float(np.quantile(draws, 0.90)),
            "m95f_p95": float(np.quantile(draws, 0.95)),
            "m95f_sim_prob_20": float(np.mean(draws >= 20)),
            "m95f_sim_prob_25": float(np.mean(draws >= 25)),
        })
    return pd.DataFrame(rows)


def carry_comparison(x: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    for name, mask in m.slice_masks(x).items():
        g = x.loc[mask]
        base = m.metrics(g["actual_carries"], g["m94c_rush_att"])
        mix = m.metrics(g["actual_carries"], g["m95f_mix_mean"])
        rows.append({
            "scope": scope,
            "slice": name,
            "n": base["n"],
            "m94c_mae": base["mae"],
            "m95f_mix_mae": mix["mae"],
            "mix_mae_gain": base["mae"] - mix["mae"],
            "m94c_bias": base["bias"],
            "m95f_mix_bias": mix["bias"],
            "m94c_corr": base["corr"],
            "m95f_mix_corr": mix["corr"],
        })
    return pd.DataFrame(rows)


def distribution_summary(x: pd.DataFrame, scope: str) -> pd.DataFrame:
    actual = num(x["actual_carries"])
    rows = []
    for q, c in [
        (0.50, "m95f_p50"),
        (0.75, "m95f_p75"),
        (0.90, "m95f_p90"),
        (0.95, "m95f_p95"),
    ]:
        rows.append({
            "scope": scope,
            "measure": f"p{int(q * 100)}_coverage",
            "value": float((actual <= num(x[c])).mean()),
            "secondary": float(num(x[c]).mean()),
        })
    for threshold, pcol in [
        (20, "m95f_sim_prob_20"),
        (25, "m95f_sim_prob_25"),
    ]:
        rows.append({
            "scope": scope,
            "measure": f"sim_{threshold}plus_auc",
            "value": m.auc_safe(actual.ge(threshold).astype(int), x[pcol]),
            "secondary": float(num(x[pcol]).mean()),
        })
        for qcol, qlabel in [("m95f_p90", "p90"), ("m95f_p95", "p95")]:
            rows.append({
                "scope": scope,
                "measure": f"{qlabel}_{threshold}plus_count",
                "value": float(num(x[qcol]).ge(threshold).sum()),
                "secondary": float(actual.ge(threshold).sum()),
            })
    return pd.DataFrame(rows)


def tail_diagnostics(x: pd.DataFrame, scope: str) -> pd.DataFrame:
    actual = num(x["actual_carries"])
    rows = []
    for name, mask in {
        "all_rb": pd.Series(True, index=x.index),
        "actual_20_plus": actual.ge(20),
        "actual_25_plus": actual.ge(25),
    }.items():
        g = x.loc[mask]
        rows.append({
            "scope": scope,
            "slice": name,
            "n": len(g),
            "actual_mean": float(num(g["actual_carries"]).mean()),
            "m94c_mean": float(num(g["m94c_rush_att"]).mean()),
            "mix_mean": float(num(g["m95f_mix_mean"]).mean()),
            "p90_mean": float(num(g["m95f_p90"]).mean()),
            "p95_mean": float(num(g["m95f_p95"]).mean()),
            "cal_prob_20_mean": float(num(g["cal_prob_20"]).mean()),
            "cal_prob_25_mean": float(num(g["cal_prob_25"]).mean()),
        })
    return pd.DataFrame(rows)


def probability_audit(
    x: pd.DataFrame,
    scope: str,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for target, suffix in [("actual_20plus", "20"), ("actual_25plus", "25")]:
        for label, pcol in [
            ("raw", f"raw_prob_{suffix}"),
            ("calibrated", f"cal_prob_{suffix}"),
        ]:
            rows.append({
                "scope": scope,
                "target": target,
                "probability": label,
                **prob_metrics(x[target], x[pcol]),
            })

        threshold = thresholds[target]
        grid = threshold_grid(x[target], x[f"cal_prob_{suffix}"])[1]
        q = grid.loc[np.isclose(grid["threshold"], threshold)].iloc[0].to_dict()
        rows.append({
            "scope": scope,
            "target": target,
            "probability": "operating_point",
            "n": len(x),
            "base_rate": float(num(x[target]).mean()),
            "mean_prob": np.nan,
            "auc": np.nan,
            "brier": np.nan,
            "logloss": np.nan,
            "ece": np.nan,
            **{f"op_{k}": v for k, v in q.items()},
        })
    return pd.DataFrame(rows)


def calibration_bins(x: pd.DataFrame, target: str, pcol: str, scope: str) -> pd.DataFrame:
    z = x[[target, pcol]].copy().dropna()
    if len(z) < 20:
        return pd.DataFrame()
    try:
        z["bin"] = pd.qcut(z[pcol], 5, labels=False, duplicates="drop") + 1
    except ValueError:
        z["bin"] = 1
    out = z.groupby("bin", as_index=False).agg(
        n=(target, "size"),
        predicted=(pcol, "mean"),
        actual=(target, "mean"),
    )
    out.insert(0, "scope", scope)
    out.insert(1, "target", target)
    return out


def stability_audit(x: pd.DataFrame, scope: str) -> pd.DataFrame:
    z = add_cal_features(x, "raw_prob_20")
    groups = {
        "all": pd.Series(True, index=z.index),
        "week17_18": z["late_week_17plus"].eq(1),
        "week18": z["week18"].eq(1),
        "share_drop_15pct": z["share_trend_1v5"].le(-0.15),
        "carry_drop_5plus": z["carry_trend_1v5"].le(-5),
        "stable_workhorse": col(z, "role_is_workhorse").eq(1)
            & z["share_trend_1v5"].ge(-0.10),
    }
    rows = []
    for name, mask in groups.items():
        g = z.loc[mask]
        if g.empty:
            continue
        rows.append({
            "scope": scope,
            "slice": name,
            "n": len(g),
            "actual_20_rate": float(num(g["actual_20plus"]).mean()),
            "pred_20": float(num(g["cal_prob_20"]).mean()),
            "actual_25_rate": float(num(g["actual_25plus"]).mean()),
            "pred_25": float(num(g["cal_prob_25"]).mean()),
        })
    return pd.DataFrame(rows)


def tail_examples(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = m.PLAYER_KEYS + [
        "actual_carries", "m94c_rush_att", "m95f_mix_mean",
        "cal_prob_20", "cal_prob_25", "m95f_p90", "m95f_p95",
        "prior_expected_carries", "rb_carries_avg1", "rb_carries_avg5",
        "rb_rb_share_avg1", "rb_rb_share_avg5",
    ]
    keep = [c for c in keep if c in x.columns]
    fp = x.loc[num(x["actual_carries"]).le(14), keep].copy()
    fp["risk"] = num(fp["cal_prob_20"])
    fp = fp.sort_values("risk", ascending=False).head(30)

    fn = x.loc[num(x["actual_carries"]).ge(25), keep].copy()
    fn["miss_gap"] = num(fn["actual_carries"]) - num(fn["m95f_mix_mean"])
    fn = fn.sort_values("miss_gap", ascending=False).head(30)
    return fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m95b-root", type=Path, required=True)
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--m94d-root", type=Path, required=True)
    ap.add_argument("--pbp-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_m95f"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trace0 = m.add_priors(m.load_trace(args.m95b_root))
    pbp = m.read_pbp_totals(args.pbp_root)
    trace, _ = m.add_targets(trace0, pbp)
    trace = m.add_priors(trace)

    team24, team25 = m.load_m94c_team(args.m94c_root)
    rb24, rb25 = m.load_m94d_rb(args.m94d_root)

    w = num(trace["week"])
    dev_train = trace.loc[
        trace["season"].eq(2023)
        | (trace["season"].eq(2024) & w.le(12))
    ].copy()

    hold_feat = trace.loc[trace["season"].eq(2024) & w.ge(13)].copy()
    hold = m.add_priors(m.prepare_eval_frame(hold_feat, rb24, team24))

    test_feat = trace.loc[trace["season"].eq(2025)].copy()
    test = m.add_priors(m.prepare_eval_frame(test_feat, rb25, team25))

    cal_choices: dict[str, str] = {}
    operating_thresholds: dict[str, float] = {}
    dev_audits = []
    hold_scores = []

    for target, suffix in [("actual_20plus", "20"), ("actual_25plus", "25")]:
        oof_train, hold_target = holdout_score_frame(trace, hold, target)
        family, audit, threshold = select_calibrator(oof_train, hold_target, target)
        cal_choices[target] = family
        operating_thresholds[target] = threshold
        dev_audits.append(audit)

        model = fit_calibrator(oof_train, family)
        hold_target[f"raw_prob_{suffix}"] = hold_target["raw_score"]
        hold_target[f"cal_prob_{suffix}"] = apply_calibrator(
            model, hold_target, family
        )
        hold_scores.append(
            hold_target[m.PLAYER_KEYS + [f"raw_prob_{suffix}", f"cal_prob_{suffix}"]]
        )

    hold_final = hold.copy()
    for h in hold_scores:
        hold_final = hold_final.merge(h, on=m.PLAYER_KEYS, how="left", validate="one_to_one")
    hold_final["cal_prob_25"] = np.minimum(
        num(hold_final["cal_prob_25"]), num(hold_final["cal_prob_20"])
    )

    final_oof_audits = []
    for target, suffix in [("actual_20plus", "20"), ("actual_25plus", "25")]:
        raw, calibrated, oof_all = final_calibrated_probs(
            trace, test, target, cal_choices[target]
        )
        test[f"raw_prob_{suffix}"] = raw
        test[f"cal_prob_{suffix}"] = calibrated
        oof_all["selected_family"] = cal_choices[target]
        final_oof_audits.append(
            oof_all[m.PLAYER_KEYS + ["raw_score", target, "selected_family"]]
        )
    test["cal_prob_25"] = np.minimum(
        num(test["cal_prob_25"]), num(test["cal_prob_20"])
    )

    pools24 = empirical_state_pools(dev_train)
    sim24 = simulate_hurdle(hold_final, pools24, SEED + 1)
    hold_final = hold_final.merge(
        sim24, on=m.PLAYER_KEYS, how="left", validate="one_to_one"
    )

    train_all = trace.loc[trace["season"].isin([2023, 2024])].copy()
    pools25 = empirical_state_pools(train_all)
    sim25 = simulate_hurdle(test, pools25, SEED + 2)
    test = test.merge(sim25, on=m.PLAYER_KEYS, how="left", validate="one_to_one")

    cmp24 = carry_comparison(hold_final, "2024_w13_18_architecture_holdout")
    cmp25 = carry_comparison(test, "2025_untouched_validation")
    carry = pd.concat([cmp24, cmp25], ignore_index=True)

    dist = pd.concat([
        distribution_summary(hold_final, "2024_w13_18_architecture_holdout"),
        distribution_summary(test, "2025_untouched_validation"),
    ], ignore_index=True)

    probs = pd.concat([
        probability_audit(
            hold_final, "2024_w13_18_architecture_holdout", operating_thresholds
        ),
        probability_audit(test, "2025_untouched_validation", operating_thresholds),
    ], ignore_index=True, sort=False)

    tails = pd.concat([
        tail_diagnostics(hold_final, "2024_w13_18_architecture_holdout"),
        tail_diagnostics(test, "2025_untouched_validation"),
    ], ignore_index=True)

    cal_bins = pd.concat([
        calibration_bins(test, "actual_20plus", "cal_prob_20", "2025_20plus"),
        calibration_bins(test, "actual_25plus", "cal_prob_25", "2025_25plus"),
    ], ignore_index=True, sort=False)

    stability = pd.concat([
        stability_audit(hold_final, "2024_w13_18_architecture_holdout"),
        stability_audit(test, "2025_untouched_validation"),
    ], ignore_index=True)

    fp, fn = tail_examples(test)

    def prob_row(scope: str, target: str, kind: str) -> pd.Series:
        q = probs.loc[
            probs["scope"].eq(scope)
            & probs["target"].eq(target)
            & probs["probability"].eq(kind)
        ]
        if q.empty:
            raise RuntimeError(f"missing M95F probability row {scope} {target} {kind}")
        return q.iloc[0]

    def dval(scope: str, measure: str) -> float:
        q = dist.loc[
            dist["scope"].eq(scope) & dist["measure"].eq(measure), "value"
        ]
        return float(q.iloc[0]) if len(q) else np.nan

    h20_raw = prob_row("2024_w13_18_architecture_holdout", "actual_20plus", "raw")
    h20_cal = prob_row("2024_w13_18_architecture_holdout", "actual_20plus", "calibrated")
    h25_raw = prob_row("2024_w13_18_architecture_holdout", "actual_25plus", "raw")
    h25_cal = prob_row("2024_w13_18_architecture_holdout", "actual_25plus", "calibrated")
    v20_raw = prob_row("2025_untouched_validation", "actual_20plus", "raw")
    v20_cal = prob_row("2025_untouched_validation", "actual_20plus", "calibrated")
    v25_raw = prob_row("2025_untouched_validation", "actual_25plus", "raw")
    v25_cal = prob_row("2025_untouched_validation", "actual_25plus", "calibrated")

    actual25 = int(num(test["actual_carries"]).ge(25).sum())
    calibration_pass = int(
        h20_cal["brier"] < h20_raw["brier"]
        and h25_cal["brier"] < h25_raw["brier"]
        and v20_cal["brier"] < v20_raw["brier"]
        and v25_cal["brier"] < v25_raw["brier"]
        and abs(v20_cal["mean_prob"] - v20_cal["base_rate"]) <= 0.04
        and abs(v25_cal["mean_prob"] - v25_cal["base_rate"]) <= 0.02
    )
    coverage_pass = int(
        0.87 <= dval("2025_untouched_validation", "p90_coverage") <= 0.95
        and 0.92 <= dval("2025_untouched_validation", "p95_coverage") <= 0.99
        and dval("2025_untouched_validation", "p90_25plus_count")
        <= 5.0 * max(actual25, 1)
    )
    validation_pass = int(calibration_pass and coverage_pass)
    disposition = (
        "ADVANCE_M95F_REGIME_DISTRIBUTION_FOR_INTEGRATION_REVIEW"
        if validation_pass
        else "RETAIN_M95F_AS_DIAGNOSTIC_DO_NOT_PROMOTE"
    )

    disp = pd.DataFrame([{
        "raw_tail_family": RAW_FAMILY,
        "calibrator_20": cal_choices["actual_20plus"],
        "calibrator_25": cal_choices["actual_25plus"],
        "threshold_20": operating_thresholds["actual_20plus"],
        "threshold_25": operating_thresholds["actual_25plus"],
        "2024_20_brier_raw": h20_raw["brier"],
        "2024_20_brier_cal": h20_cal["brier"],
        "2024_25_brier_raw": h25_raw["brier"],
        "2024_25_brier_cal": h25_cal["brier"],
        "2025_20_brier_raw": v20_raw["brier"],
        "2025_20_brier_cal": v20_cal["brier"],
        "2025_25_brier_raw": v25_raw["brier"],
        "2025_25_brier_cal": v25_cal["brier"],
        "2025_20_base_rate": v20_cal["base_rate"],
        "2025_20_mean_probability": v20_cal["mean_prob"],
        "2025_25_base_rate": v25_cal["base_rate"],
        "2025_25_mean_probability": v25_cal["mean_prob"],
        "2025_p90_coverage": dval("2025_untouched_validation", "p90_coverage"),
        "2025_p95_coverage": dval("2025_untouched_validation", "p95_coverage"),
        "2025_p90_25plus_count": dval("2025_untouched_validation", "p90_25plus_count"),
        "actual_2025_25plus_count": actual25,
        "calibration_pass": calibration_pass,
        "coverage_pass": coverage_pass,
        "validation_pass": validation_pass,
        "m94c_central_mean_preserved": 1,
        "disposition": disposition,
        "production_change": 0,
    }])

    pd.concat(dev_audits, ignore_index=True, sort=False).to_csv(
        args.out_dir / "m95f_2024_calibrator_selection.csv", index=False
    )
    pd.concat(final_oof_audits, ignore_index=True, sort=False).to_csv(
        args.out_dir / "m95f_2024_temporal_oof_scores.csv", index=False
    )
    carry.to_csv(args.out_dir / "m95f_carry_comparison.csv", index=False)
    probs.to_csv(args.out_dir / "m95f_probability_audit.csv", index=False)
    dist.to_csv(args.out_dir / "m95f_distribution_summary.csv", index=False)
    tails.to_csv(args.out_dir / "m95f_tail_diagnostics.csv", index=False)
    cal_bins.to_csv(args.out_dir / "m95f_2025_probability_calibration_bins.csv", index=False)
    stability.to_csv(args.out_dir / "m95f_stability_audit.csv", index=False)
    hold_final.to_csv(args.out_dir / "m95f_2024_holdout_trace.csv", index=False)
    test.to_csv(args.out_dir / "m95f_2025_rb_trace.csv", index=False)
    fp.to_csv(args.out_dir / "m95f_2025_false_positive_examples.csv", index=False)
    fn.to_csv(args.out_dir / "m95f_2025_false_negative_25plus.csv", index=False)
    disp.to_csv(args.out_dir / "m95f_disposition.csv", index=False)

    print("[m95f] disposition")
    print(disp.to_string(index=False))
    print("\n[m95f] probability audit")
    print(probs.to_string(index=False))
    print("\n[m95f] carry comparison")
    print(carry.to_string(index=False))
    print("\n[m95f] tail diagnostics")
    print(tails.to_string(index=False))
    print("\n[m95f] distribution")
    print(dist.to_string(index=False))
    print("\n[m95f] stability audit")
    print(stability.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
