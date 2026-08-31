"""M95I: calibrated deep-concentration + workload-tail integration.

Research-only. M95H validated one recipient-specific entitlement signal:
P(player receives >=70% of team RB carries). M95F validated calibrated 20+/25+
workload-state probabilities, while M94C remains the best central carry mean.

M95I asks whether those signals can be integrated selectively so the carry tail
expands for the right RBs without broadly lifting ordinary games.

Protocol
--------
1. Freeze M95H >=70% architecture (entitlement_competition, C=.03).
2. Generate temporal 2024 OOF >=70% probabilities for W9-12; fit a separate
   incumbent/vacancy calibration mapping using only those OOF rows.
3. Use M95H W13-18 probabilities (trained only through W12) plus frozen M95F
   calibrated 20+/25+ probabilities.
4. Fit/select compact tail-integration meta models with 2024 only. Meta models
   train on W13-15 and are selected on W16-18.
5. Select one pre-specified selective tail transformation on W16-18.
6. Freeze architecture, refit the selected meta model on 2024 W13-18, and
   evaluate exactly once on untouched 2025. The authoritative frozen M95H 2025
   trace supplies the >=70% probability so source drift cannot alter validation.
7. No sportsbook input. No production changes. M94C remains the reference
   central estimate unless the pre-specified selective transform is being
   scored as a research candidate.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Apply pandas-1.5 compatibility patch before importing the M95H module.
import scripts.backtest.evaluate_rb_lead_role_entitlement_v2  # noqa: F401
import scripts.backtest.evaluate_rb_lead_role_entitlement as h
import scripts.backtest.evaluate_rb_role_availability as g

SEED = 95109
PLAYER_KEYS = h.PLAYER_KEYS
TEAM_KEYS = h.TEAM_KEYS
SHARE_SPEC = "entitlement_competition"
SHARE_C = 0.03
CAL_SHRINK_GRID = (10.0, 25.0, 50.0)
META_C_GRID = (0.03, 0.10, 0.30)
THRESHOLDS = (
    0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10,
    0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
)
MAX_FLAG_MULTIPLE = 3.0

META_SPECS = {
    "tail_only": ["tail_logit"],
    "tail_share70": ["tail_logit", "share70_logit"],
    "tail_share70_opportunity": [
        "tail_logit", "share70_logit", "m94c_rush_att", "candidate_team_rush_att",
    ],
    "tail_share70_regime": [
        "tail_logit", "share70_logit", "m94c_rush_att", "candidate_team_rush_att",
        "vacancy", "ix_tail_x_share", "ix_share_x_vacancy", "ix_tail_x_vacancy",
    ],
}

# Tail transforms do not invent an arbitrary carry target. For an eligible row,
# they use the expected value of the calibrated normal/high/extreme state mixture,
# then cap the positive uplift. All gates/caps are fixed before 2025 is observed.
TRANSFORMS = (
    {"name": "central_only", "q70": 2.0, "p20": 2.0, "p25": 2.0, "team": 999.0, "cap": 0.0, "mode": "none"},
    {"name": "share60", "q70": 0.60, "p20": 0.00, "p25": 0.00, "team": 0.0, "cap": 3.0, "mode": "or"},
    {"name": "share65_p20", "q70": 0.65, "p20": 0.15, "p25": 0.04, "team": 0.0, "cap": 3.0, "mode": "or"},
    {"name": "share65_env", "q70": 0.65, "p20": 0.15, "p25": 0.04, "team": 27.0, "cap": 3.0, "mode": "or"},
    {"name": "share70_env", "q70": 0.70, "p20": 0.18, "p25": 0.05, "team": 27.0, "cap": 3.5, "mode": "or"},
    {"name": "extreme_only", "q70": 0.75, "p20": 2.0, "p25": 0.05, "team": 26.0, "cap": 4.0, "mode": "p25"},
    {"name": "high_confidence", "q70": 0.80, "p20": 0.25, "p25": 0.07, "team": 28.0, "cap": 4.0, "mode": "or"},
)


def num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} under {root}; found {len(hits)}")
    return hits[0]


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def binary_metrics(y, p) -> dict:
    z = pd.DataFrame({"y": num(y), "p": num(pd.Series(p, index=y.index if hasattr(y, "index") else None))}).dropna()
    if z.empty:
        return {"n": 0, "base_rate": np.nan, "mean_prob": np.nan, "auc": np.nan, "brier": np.nan, "logloss": np.nan}
    yy = z["y"].astype(int)
    pp = z["p"].clip(1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(yy, pp)) if yy.nunique() > 1 else np.nan
    return {
        "n": int(len(z)), "base_rate": float(yy.mean()), "mean_prob": float(pp.mean()),
        "auc": auc, "brier": float(np.mean((pp - yy) ** 2)),
        "logloss": float(log_loss(yy, pp, labels=[0, 1])),
    }


def carry_metrics(actual, pred) -> dict:
    z = pd.DataFrame({"a": num(actual), "p": num(pred)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "bias": np.nan, "corr": np.nan}
    err = z["p"] - z["a"]
    corr = float(z["a"].corr(z["p"])) if len(z) >= 3 and z["a"].nunique() > 1 and z["p"].nunique() > 1 else np.nan
    return {"n": int(len(z)), "mae": float(np.abs(err).mean()), "bias": float(err.mean()), "corr": corr}


def carry_slices(df: pd.DataFrame) -> dict[str, pd.Series]:
    a = num(df["actual_carries"])
    out = {
        "all_rb": pd.Series(True, index=df.index),
        "actual_0_5": a.between(0, 5),
        "actual_6_10": a.between(6, 10),
        "actual_11_14": a.between(11, 14),
        "actual_15_plus": a.ge(15),
        "actual_20_plus": a.ge(20),
        "actual_25_plus": a.ge(25),
    }
    if "bellcow_60" in df.columns:
        out["bellcow60"] = df["bellcow_60"].astype(bool)
    return out


def threshold_stats(y, p, threshold: float) -> dict:
    yy = num(y).fillna(0).astype(int)
    pp = num(pd.Series(p, index=yy.index))
    pred = pp.ge(threshold)
    truth = yy.eq(1)
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "flags": int(pred.sum()), "actual_positive": int(truth.sum()),
        "flag_multiple": float(pred.sum() / max(int(truth.sum()), 1)),
    }


def choose_threshold(y, p) -> tuple[float, pd.DataFrame]:
    rows = []
    actual_pos = int(num(y).fillna(0).astype(int).sum())
    limit = max(int(math.ceil(MAX_FLAG_MULTIPLE * actual_pos)), actual_pos + 5)
    for t in THRESHOLDS:
        r = threshold_stats(y, p, t)
        r["eligible"] = int(r["flags"] <= limit)
        rows.append(r)
    grid = pd.DataFrame(rows)
    pool = grid.loc[grid["eligible"].eq(1)].copy()
    if pool.empty:
        pool = grid.copy()
    pool = pool.sort_values(["f1", "recall", "precision", "threshold"], ascending=[False, False, False, False])
    return float(pool.iloc[0]["threshold"]), grid


def load_frozen_2025_m95h(root: Path) -> pd.DataFrame:
    x = pd.read_csv(find_one(root, "m95h_2025_trace.csv"), low_memory=False)
    x.columns = [str(c).lower() for c in x.columns]
    for c in ["season", "week"]:
        x[c] = num(x[c]).astype(int)
    x["team"] = x["team"].map(g.canon)
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    keep = PLAYER_KEYS + [
        "p_share70_m95h", "p_share70_history", "prior_top1_unavailable",
        "depth_rank", "effective_available", "actual_share70", "actual_rb_share",
    ]
    return x[[c for c in keep if c in x.columns]].drop_duplicates(PLAYER_KEYS)


def build_2024_entitlement_frames(m95f_root: Path, m95b_root: Path):
    oof, hold, _, trace = h.load_inputs(m95f_root, m95b_root)
    rosters, injuries, depth, source_audit = g.load_provider_sources([2024])
    rosters = g.add_roster_transition_features(rosters)
    depth = g.add_depth_transition_features(depth)
    oof = h.add_entitlement_truth(oof, trace)
    hold = h.add_entitlement_truth(hold, trace)
    oof_e = h.enrich(oof, trace, rosters, injuries, depth)
    hold_e = h.enrich(hold, trace, rosters, injuries, depth)
    return oof_e, hold_e, trace, source_audit


def temporal_share70_oof(oof_e: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    folds = [(8, 9, 10), (10, 11, 12)]
    for train_end, test_start, test_end in folds:
        tr = oof_e.loc[num(oof_e["week"]).between(5, train_end)].copy()
        te = oof_e.loc[num(oof_e["week"]).between(test_start, test_end)].copy()
        if tr.empty or te.empty or num(tr["actual_share70"]).nunique() < 2:
            continue
        p, _, _ = h.fit_predict(tr, te, "actual_share70", SHARE_SPEC, SHARE_C)
        q = te[PLAYER_KEYS + ["actual_share70", "prior_top1_unavailable"]].copy()
        q["p_share70_raw"] = p
        pieces.append(q)
    if not pieces:
        raise RuntimeError("M95I could not generate temporal share70 OOF")
    return pd.concat(pieces, ignore_index=True).drop_duplicates(PLAYER_KEYS)


def share70_hold_predictions(oof_e: pd.DataFrame, hold_e: pd.DataFrame) -> pd.DataFrame:
    tr = oof_e.loc[num(oof_e["week"]).between(5, 12)].copy()
    te = hold_e.loc[num(hold_e["week"]).between(13, 18)].copy()
    p, _, _ = h.fit_predict(tr, te, "actual_share70", SHARE_SPEC, SHARE_C)
    keep = PLAYER_KEYS + [
        "actual_share70", "prior_top1_unavailable", "actual_carries_m95h",
        "actual_rb_share", "rb_rb_share_avg1", "rb_rb_share_avg5",
    ]
    q = te[[c for c in keep if c in te.columns]].copy()
    q["p_share70_raw"] = p
    return q


def solve_intercept_delta(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    target = float(np.mean(y))
    if target <= 0:
        return -8.0
    if target >= 1:
        return 8.0
    lp = logit(p)
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        meanp = float(sigmoid(lp + mid).mean())
        if meanp < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass
class RegimeCalibrator:
    shrink_k: float
    global_delta: float
    regime_delta: dict[str, float]

    def apply(self, df: pd.DataFrame, p_col: str = "p_share70_raw") -> np.ndarray:
        p = np.clip(num(df[p_col]).to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        vac = num(df["prior_top1_unavailable"]).fillna(0).eq(1).to_numpy()
        regs = np.where(vac, "vacancy", "incumbent")
        delta = np.array([self.regime_delta.get(r, self.global_delta) for r in regs], dtype=float)
        return np.clip(sigmoid(logit(p) + delta), 1e-6, 1 - 1e-6)


def fit_regime_calibrator(df: pd.DataFrame, shrink_k: float) -> RegimeCalibrator:
    z = df[["p_share70_raw", "actual_share70", "prior_top1_unavailable"]].dropna(subset=["p_share70_raw", "actual_share70"]).copy()
    global_raw = solve_intercept_delta(num(z["p_share70_raw"]), num(z["actual_share70"]))
    global_shrink = len(z) / (len(z) + shrink_k)
    global_delta = float(global_raw * global_shrink)
    deltas = {}
    for name, mask in {
        "incumbent": num(z["prior_top1_unavailable"]).fillna(0).eq(0),
        "vacancy": num(z["prior_top1_unavailable"]).fillna(0).eq(1),
    }.items():
        q = z.loc[mask]
        if len(q) < 15 or num(q["actual_share70"]).nunique() < 2:
            deltas[name] = global_delta
            continue
        raw = solve_intercept_delta(num(q["p_share70_raw"]), num(q["actual_share70"]))
        shrink = len(q) / (len(q) + shrink_k)
        deltas[name] = float(raw * shrink)
    return RegimeCalibrator(shrink_k=shrink_k, global_delta=global_delta, regime_delta=deltas)


def regime_calibration_rows(df: pd.DataFrame, raw_col: str, cal_col: str, scope: str) -> list[dict]:
    rows = []
    for regime, mask in {
        "all": pd.Series(True, index=df.index),
        "incumbent": num(df["prior_top1_unavailable"]).fillna(0).eq(0),
        "vacancy": num(df["prior_top1_unavailable"]).fillna(0).eq(1),
    }.items():
        q = df.loc[mask]
        if len(q) < 10:
            continue
        for label, col in [("raw", raw_col), ("calibrated", cal_col)]:
            rows.append({"scope": scope, "regime": regime, "model": label, **binary_metrics(q["actual_share70"], q[col])})
    return rows


def choose_share_calibration(oof_share: pd.DataFrame, hold_share: pd.DataFrame):
    dev = hold_share.loc[num(hold_share["week"]).between(13, 15)].copy()
    rows = []
    models = {}
    for k in CAL_SHRINK_GRID:
        cal = fit_regime_calibrator(oof_share, k)
        models[k] = cal
        p = cal.apply(dev)
        m = binary_metrics(dev["actual_share70"], p)
        vac = dev.loc[num(dev["prior_top1_unavailable"]).fillna(0).eq(1)].copy()
        if len(vac) >= 10 and num(vac["actual_share70"]).nunique() >= 2:
            pv = cal.apply(vac)
            mv = binary_metrics(vac["actual_share70"], pv)
            vac_brier = mv["brier"]
            vac_gap = abs(mv["mean_prob"] - mv["base_rate"])
        else:
            vac_brier = m["brier"]
            vac_gap = abs(m["mean_prob"] - m["base_rate"])
        score = float(m["brier"] + 0.50 * vac_brier + 0.20 * vac_gap)
        rows.append({"shrink_k": k, **m, "vacancy_brier": vac_brier, "vacancy_abs_cal_gap": vac_gap, "selection_score": score})
    grid = pd.DataFrame(rows).sort_values(["selection_score", "brier", "shrink_k"])
    chosen_k = float(grid.iloc[0]["shrink_k"])
    return models[chosen_k], grid


def add_meta_features(df: pd.DataFrame, tail_col: str) -> pd.DataFrame:
    z = df.copy()
    z["tail_logit"] = logit(num(z[tail_col]).clip(1e-6, 1 - 1e-6))
    z["share70_logit"] = logit(num(z["p_share70_cal"]).clip(1e-6, 1 - 1e-6))
    z["vacancy"] = num(z["prior_top1_unavailable"]).fillna(0).eq(1).astype(int)
    z["m94c_rush_att"] = num(z["m94c_rush_att"])
    z["candidate_team_rush_att"] = num(z["candidate_team_rush_att"])
    z["ix_tail_x_share"] = z["tail_logit"] * z["share70_logit"]
    z["ix_share_x_vacancy"] = z["share70_logit"] * z["vacancy"]
    z["ix_tail_x_vacancy"] = z["tail_logit"] * z["vacancy"]
    return z


def meta_pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=3000, random_state=SEED)),
    ])


def fit_meta(train: pd.DataFrame, test: pd.DataFrame, target: str, tail_col: str, spec: str, c: float):
    tr = add_meta_features(train, tail_col); te = add_meta_features(test, tail_col)
    feats = [f for f in META_SPECS[spec] if f in tr.columns and num(tr[f]).notna().sum() >= 20 and num(tr[f]).nunique(dropna=True) > 1]
    if not feats:
        raise RuntimeError(f"no M95I meta features for {spec}")
    y = num(tr[target]).astype(int)
    if y.nunique() < 2:
        raise RuntimeError(f"one-class M95I meta train for {target}")
    model = meta_pipeline(c)
    model.fit(tr[feats], y)
    p = np.clip(model.predict_proba(te[feats])[:, 1], 1e-6, 1 - 1e-6)
    return p, model, feats


def select_meta(train: pd.DataFrame, select: pd.DataFrame, target: str, tail_col: str):
    base = binary_metrics(select[target], select[tail_col])
    rows = []
    preds = {}
    for spec in META_SPECS:
        for c in META_C_GRID:
            try:
                p, _, feats = fit_meta(train, select, target, tail_col, spec, c)
            except Exception as exc:
                rows.append({"target": target, "spec": spec, "C": c, "eligible": 0, "error": f"{type(exc).__name__}:{exc}"})
                continue
            preds[(spec, c)] = p
            m = binary_metrics(select[target], p)
            auc_gain = (m["auc"] - base["auc"]) if np.isfinite(m["auc"]) and np.isfinite(base["auc"]) else 0.0
            brier_gain = base["brier"] - m["brier"]
            eligible = int(m["brier"] <= base["brier"] + 0.001 and (not np.isfinite(base["auc"]) or not np.isfinite(m["auc"]) or m["auc"] >= base["auc"] - 0.005))
            score = 4.0 * brier_gain + auc_gain
            rows.append({
                "target": target, "spec": spec, "C": c, "feature_count": len(feats), **m,
                "baseline_auc": base["auc"], "baseline_brier": base["brier"],
                "auc_gain": auc_gain, "brier_gain": brier_gain, "eligible": eligible, "selection_score": score,
            })
    grid = pd.DataFrame(rows)
    pool = grid.loc[num(grid["eligible"]).eq(1)].copy()
    if pool.empty:
        pool = grid.loc[grid["selection_score"].notna()].copy()
    if pool.empty:
        raise RuntimeError(f"M95I no valid meta candidate for {target}")
    chosen = pool.sort_values(["selection_score", "brier", "spec", "C"], ascending=[False, True, True, True]).iloc[0].to_dict()
    key = (str(chosen["spec"]), float(chosen["C"]))
    return chosen, grid, preds[key]


def transform_carries(df: pd.DataFrame, spec: dict, high_mean: float, extreme_mean: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mu = num(df["m94c_rush_att"]).clip(lower=0)
    p20 = num(df["p20_joint"]).clip(0, 1)
    p25 = np.minimum(num(df["p25_joint"]).clip(0, 1), p20)
    q70 = num(df["p_share70_cal"]).clip(0, 1)
    team = num(df["candidate_team_rush_att"]).fillna(0)
    if spec["mode"] == "none":
        eligible = pd.Series(False, index=df.index)
    else:
        if spec["mode"] == "p25":
            tail_gate = p25.ge(spec["p25"])
        else:
            tail_gate = p20.ge(spec["p20"]) | p25.ge(spec["p25"])
        eligible = q70.ge(spec["q70"]) & tail_gate & team.ge(spec["team"])
    p_mid = (p20 - p25).clip(lower=0)
    mix_expected = (1 - p20) * mu + p_mid * high_mean + p25 * extreme_mean
    uplift = (mix_expected - mu).clip(lower=0, upper=float(spec["cap"]))
    uplift = uplift.where(eligible, 0.0)
    return mu + uplift, uplift, eligible.astype(int)


def score_transform_grid(select: pd.DataFrame, high_mean: float, extreme_mean: float):
    base_by_slice = {name: carry_metrics(select.loc[mask, "actual_carries"], select.loc[mask, "m94c_rush_att"]) for name, mask in carry_slices(select).items()}
    rows = []
    for spec in TRANSFORMS:
        pred, uplift, eligible = transform_carries(select, spec, high_mean, extreme_mean)
        z = select.copy(); z["candidate"] = pred; z["uplift"] = uplift; z["eligible_tail"] = eligible
        vals = {}
        for name, mask in carry_slices(z).items():
            cm = carry_metrics(z.loc[mask, "actual_carries"], z.loc[mask, "candidate"])
            bm = base_by_slice[name]
            vals[f"{name}_n"] = cm["n"]
            vals[f"{name}_mae"] = cm["mae"]
            vals[f"{name}_gain"] = bm["mae"] - cm["mae"]
        all_damage = max(0.0, -vals.get("all_rb_gain", 0.0))
        ordinary_damage = sum(max(0.0, -vals.get(f"{s}_gain", 0.0)) for s in ["actual_0_5", "actual_6_10", "actual_11_14"])
        gain20 = vals.get("actual_20_plus_gain", 0.0); gain25 = vals.get("actual_25_plus_gain", 0.0)
        eligible_dev = int(
            all_damage <= 0.10
            and all(-vals.get(f"{s}_gain", 0.0) <= 0.15 for s in ["actual_0_5", "actual_6_10", "actual_11_14"])
            and gain20 >= 0.0 and gain25 >= 0.0
            and max(gain20, gain25) >= 0.15
        ) if spec["name"] != "central_only" else 0
        score = gain20 + 1.5 * gain25 - 2.0 * all_damage - ordinary_damage
        rows.append({
            **spec, **vals, "mean_uplift": float(uplift.mean()), "max_uplift": float(uplift.max()),
            "flagged_rows": int(eligible.sum()), "development_eligible": eligible_dev, "selection_score": score,
        })
    grid = pd.DataFrame(rows)
    pool = grid.loc[grid["development_eligible"].eq(1)].copy()
    if pool.empty:
        noncentral = grid.loc[grid["name"].ne("central_only")].copy()
        chosen = noncentral.sort_values(["selection_score", "all_rb_mae"], ascending=[False, True]).iloc[0].to_dict()
        chosen["development_eligible"] = 0
    else:
        chosen = pool.sort_values(["selection_score", "all_rb_mae"], ascending=[False, True]).iloc[0].to_dict()
    return chosen, grid


def projection_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col in [("m94c", "m94c_rush_att"), ("m95i", "m95i_rush_att")]:
        p = num(df[col]).dropna()
        rows.append({
            "model": label, "n": len(p), "mean": float(p.mean()), "max": float(p.max()),
            "p50": float(p.quantile(.50)), "p75": float(p.quantile(.75)), "p90": float(p.quantile(.90)),
            "p95": float(p.quantile(.95)), "p99": float(p.quantile(.99)),
            "count_ge18": int(p.ge(18).sum()), "count_ge20": int(p.ge(20).sum()),
            "count_ge22": int(p.ge(22).sum()), "count_ge25": int(p.ge(25).sum()),
        })
    return pd.DataFrame(rows)


def carry_metric_table(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    for name, mask in carry_slices(df).items():
        b = carry_metrics(df.loc[mask, "actual_carries"], df.loc[mask, "m94c_rush_att"])
        c = carry_metrics(df.loc[mask, "actual_carries"], df.loc[mask, "m95i_rush_att"])
        rows.append({
            "scope": scope, "slice": name, "n": b["n"],
            "m94c_mae": b["mae"], "m95i_mae": c["mae"], "mae_gain": b["mae"] - c["mae"],
            "m94c_bias": b["bias"], "m95i_bias": c["bias"],
            "m94c_corr": b["corr"], "m95i_corr": c["corr"],
            "actual_mean": float(num(df.loc[mask, "actual_carries"]).mean()) if mask.any() else np.nan,
            "m94c_mean": float(num(df.loc[mask, "m94c_rush_att"]).mean()) if mask.any() else np.nan,
            "m95i_mean": float(num(df.loc[mask, "m95i_rush_att"]).mean()) if mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def probability_table(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    share_trend = num(df.get("rb_rb_share_avg1", pd.Series(np.nan, index=df.index))) - num(df.get("rb_rb_share_avg5", pd.Series(np.nan, index=df.index)))
    stable = num(df.get("role_is_workhorse", pd.Series(0, index=df.index))).eq(1) & share_trend.ge(-0.10)
    masks = {
        "all": pd.Series(True, index=df.index),
        "incumbent": num(df["prior_top1_unavailable"]).fillna(0).eq(0),
        "vacancy": num(df["prior_top1_unavailable"]).fillna(0).eq(1),
        "stable_workhorse": stable,
    }
    for target, bcol, jcol in [
        ("actual_20plus", "cal_prob_20", "p20_joint"),
        ("actual_25plus", "cal_prob_25", "p25_joint"),
    ]:
        for sl, mask in masks.items():
            q = df.loc[mask]
            if len(q) < 10 or num(q[target]).nunique() < 2:
                continue
            for model, col in [("m95f", bcol), ("m95i_joint", jcol)]:
                rows.append({"scope": scope, "target": target, "slice": sl, "model": model, **binary_metrics(q[target], q[col])})
    return pd.DataFrame(rows)


def downstream_yard_sensitivity(val: pd.DataFrame, m94c_root: Path) -> pd.DataFrame:
    rb = pd.read_csv(find_one(m94c_root, "m94c_2025_rb_trace.csv"), low_memory=False)
    rb.columns = [str(c).lower() for c in rb.columns]
    rb["season"] = num(rb["season"]).astype(int); rb["week"] = num(rb["week"]).astype(int)
    rb["team"] = rb["team"].map(g.canon); rb["player_clean_key"] = rb["player_clean_key"].astype(str)
    keep = PLAYER_KEYS + ["candidate_rush_att", "candidate_rush_yards", "candidate_rush_rec_yards", "actual_rush_yards", "actual_rush_rec_yards"]
    z = val.merge(rb[[c for c in keep if c in rb.columns]], on=PLAYER_KEYS, how="inner", suffixes=("", "_m94c"))
    if z.empty:
        return pd.DataFrame()
    base_att = num(z["candidate_rush_att"]).clip(lower=0.5)
    ypc = (num(z["candidate_rush_yards"]) / base_att).clip(lower=2.0, upper=7.0)
    delta_att = num(z["m95i_rush_att"]) - num(z["m94c_rush_att"])
    z["m95i_rush_yards_sensitivity"] = num(z["candidate_rush_yards"]) + delta_att * ypc
    rec_component = num(z["candidate_rush_rec_yards"]) - num(z["candidate_rush_yards"])
    z["m95i_rush_rec_sensitivity"] = z["m95i_rush_yards_sensitivity"] + rec_component
    rows = []
    for sl, mask in carry_slices(z).items():
        q = z.loc[mask]
        if q.empty:
            continue
        for target, base_col, new_col in [
            ("rush_yards", "candidate_rush_yards", "m95i_rush_yards_sensitivity"),
            ("rush_rec_yards", "candidate_rush_rec_yards", "m95i_rush_rec_sensitivity"),
        ]:
            actual_col = "actual_rush_yards" if target == "rush_yards" else "actual_rush_rec_yards"
            b = carry_metrics(q[actual_col], q[base_col]); n = carry_metrics(q[actual_col], q[new_col])
            rows.append({"slice": sl, "target": target, "n": b["n"], "m94c_mae": b["mae"], "m95i_sensitivity_mae": n["mae"], "mae_gain": b["mae"] - n["mae"]})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m95f-root", type=Path, required=True)
    ap.add_argument("--m95b-root", type=Path, required=True)
    ap.add_argument("--m95h-root", type=Path, required=True)
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    # Frozen M95F traces provide M94C central carries and calibrated 20+/25+ probabilities.
    hold_f = pd.read_csv(find_one(args.m95f_root, "m95f_2024_holdout_trace.csv"), low_memory=False)
    val_f = pd.read_csv(find_one(args.m95f_root, "m95f_2025_rb_trace.csv"), low_memory=False)
    for x in (hold_f, val_f):
        x.columns = [str(c).lower() for c in x.columns]
        x["season"] = num(x["season"]).astype(int); x["week"] = num(x["week"]).astype(int)
        x["team"] = x["team"].map(g.canon); x["player_clean_key"] = x["player_clean_key"].astype(str)

    oof_e, hold_e, _, source_audit = build_2024_entitlement_frames(args.m95f_root, args.m95b_root)
    share_oof = temporal_share70_oof(oof_e)
    share_hold = share70_hold_predictions(oof_e, hold_e)

    # Select incumbent/vacancy calibration with 2024 only (fit W9-12, score W13-15).
    share_cal, cal_grid = choose_share_calibration(share_oof, share_hold)
    share_hold["p_share70_cal"] = share_cal.apply(share_hold)
    share_oof["p_share70_cal"] = share_cal.apply(share_oof)

    hold = hold_f.merge(
        share_hold[PLAYER_KEYS + ["p_share70_raw", "p_share70_cal", "prior_top1_unavailable", "actual_share70"]],
        on=PLAYER_KEYS, how="inner", validate="one_to_one",
    )
    if hold.empty:
        raise RuntimeError("M95I empty 2024 holdout after share merge")

    # Meta integration: train W13-15, select W16-18.
    meta_train = hold.loc[num(hold["week"]).between(13, 15)].copy()
    meta_select = hold.loc[num(hold["week"]).between(16, 18)].copy()
    selected_meta_rows = []
    meta_grids = []
    selected_predictions = {}
    for target, tail_col, outcol in [
        ("actual_20plus", "cal_prob_20", "p20_joint"),
        ("actual_25plus", "cal_prob_25", "p25_joint"),
    ]:
        chosen, grid, psel = select_meta(meta_train, meta_select, target, tail_col)
        selected_meta_rows.append({"target": target, "tail_col": tail_col, "spec": chosen["spec"], "C": chosen["C"], "selection_auc": chosen.get("auc", np.nan), "selection_brier": chosen.get("brier", np.nan), "auc_gain": chosen.get("auc_gain", np.nan), "brier_gain": chosen.get("brier_gain", np.nan), "development_eligible": chosen.get("eligible", 0)})
        meta_grids.append(grid)
        selected_predictions[outcol] = psel
    selected_meta = pd.DataFrame(selected_meta_rows)
    meta_select = meta_select.copy()
    meta_select["p20_joint"] = selected_predictions["p20_joint"]
    meta_select["p25_joint"] = np.minimum(selected_predictions["p25_joint"], selected_predictions["p20_joint"])

    # State means use only data available before the transform-selection weeks.
    state_source = pd.concat([
        oof_e.loc[num(oof_e["week"]).between(5, 12), ["actual_carries_m95h"]].rename(columns={"actual_carries_m95h": "actual_carries"}),
        hold.loc[num(hold["week"]).between(13, 15), ["actual_carries"]],
    ], ignore_index=True)
    ac = num(state_source["actual_carries"])
    high_mean = float(ac.loc[ac.between(20, 24)].mean())
    extreme_mean = float(ac.loc[ac.ge(25)].mean())
    if not np.isfinite(high_mean) or not np.isfinite(extreme_mean):
        raise RuntimeError("M95I insufficient 2024 high-state observations")

    chosen_transform, transform_grid = score_transform_grid(meta_select, high_mean, extreme_mean)
    chosen_spec = next(d for d in TRANSFORMS if d["name"] == chosen_transform["name"])

    # Freeze operating thresholds on W16-18 integrated probabilities.
    th20, th20_grid = choose_threshold(meta_select["actual_20plus"], meta_select["p20_joint"])
    th25, th25_grid = choose_threshold(meta_select["actual_25plus"], meta_select["p25_joint"])
    th20_grid.insert(0, "target", "actual_20plus"); th25_grid.insert(0, "target", "actual_25plus")

    # Untouched 2025: use authoritative frozen M95H trace for share70 probability.
    h25 = load_frozen_2025_m95h(args.m95h_root)
    val = val_f.merge(h25, on=PLAYER_KEYS, how="inner", validate="one_to_one")
    val = val.rename(columns={"p_share70_m95h": "p_share70_raw"})
    val["p_share70_cal"] = share_cal.apply(val)
    if val.empty:
        raise RuntimeError("M95I empty 2025 validation after frozen M95H merge")

    # Refit selected tail meta models on all 2024 W13-18, then validate once on 2025.
    for target, tail_col, outcol in [
        ("actual_20plus", "cal_prob_20", "p20_joint"),
        ("actual_25plus", "cal_prob_25", "p25_joint"),
    ]:
        row = selected_meta.loc[selected_meta["target"].eq(target)].iloc[0]
        p, _, _ = fit_meta(hold, val, target, tail_col, str(row["spec"]), float(row["C"]))
        val[outcol] = p
    val["p25_joint"] = np.minimum(num(val["p25_joint"]), num(val["p20_joint"]))

    pred, uplift, eligible = transform_carries(val, chosen_spec, high_mean, extreme_mean)
    val["m95i_rush_att"] = pred
    val["m95i_tail_uplift"] = uplift
    val["m95i_tail_eligible"] = eligible

    carry_table = carry_metric_table(val, "2025_untouched_validation")
    prob_table = probability_table(val, "2025_untouched_validation")
    distribution = projection_distribution(val)

    # Operating-point audit versus frozen M95F thresholds (.20 / .10) and M95I thresholds.
    threshold_rows = []
    for target, bcol, jcol, base_th, new_th in [
        ("actual_20plus", "cal_prob_20", "p20_joint", 0.20, th20),
        ("actual_25plus", "cal_prob_25", "p25_joint", 0.10, th25),
    ]:
        for model, col, t in [("m95f", bcol, base_th), ("m95i_joint", jcol, new_th)]:
            threshold_rows.append({"target": target, "model": model, **threshold_stats(val[target], val[col], t)})
    threshold_table = pd.DataFrame(threshold_rows)

    # Stable workhorse / regime audit.
    share_trend = num(val.get("rb_rb_share_avg1", pd.Series(np.nan, index=val.index))) - num(val.get("rb_rb_share_avg5", pd.Series(np.nan, index=val.index)))
    stable = num(val.get("role_is_workhorse", pd.Series(0, index=val.index))).eq(1) & share_trend.ge(-0.10)
    stability_rows = []
    for sl, mask in {
        "all": pd.Series(True, index=val.index), "stable_workhorse": stable,
        "vacancy": num(val["prior_top1_unavailable"]).fillna(0).eq(1),
        "incumbent": num(val["prior_top1_unavailable"]).fillna(0).eq(0),
    }.items():
        q = val.loc[mask]
        if q.empty:
            continue
        stability_rows.append({
            "slice": sl, "n": len(q),
            "actual20": float(num(q["actual_20plus"]).mean()), "m95f20": float(num(q["cal_prob_20"]).mean()), "m95i20": float(num(q["p20_joint"]).mean()),
            "actual25": float(num(q["actual_25plus"]).mean()), "m95f25": float(num(q["cal_prob_25"]).mean()), "m95i25": float(num(q["p25_joint"]).mean()),
        })
    stability = pd.DataFrame(stability_rows)

    q70_audit_rows = []
    hold_a = share_hold.copy(); hold_a["p_share70_cal"] = share_cal.apply(hold_a)
    q70_audit_rows.extend(regime_calibration_rows(hold_a.loc[num(hold_a["week"]).between(13, 18)], "p_share70_raw", "p_share70_cal", "2024_w13_18"))
    val_q = val.copy(); val_q["actual_share70"] = num(val_q.get("actual_share70", pd.Series(np.nan, index=val_q.index)))
    q70_audit_rows.extend(regime_calibration_rows(val_q, "p_share70_raw", "p_share70_cal", "2025_untouched_validation"))
    q70_audit = pd.DataFrame(q70_audit_rows)

    # False-positive / false-negative examples for tail diagnosis.
    keep = [c for c in PLAYER_KEYS + [
        "actual_carries", "m94c_rush_att", "m95i_rush_att", "m95i_tail_uplift",
        "p_share70_raw", "p_share70_cal", "cal_prob_20", "p20_joint", "cal_prob_25", "p25_joint",
        "candidate_team_rush_att", "prior_top1_unavailable", "role_is_workhorse",
    ] if c in val.columns]
    fp = val.loc[num(val["actual_carries"]).le(14), keep].copy()
    fp["risk"] = num(fp["p20_joint"])
    fp = fp.sort_values("risk", ascending=False).head(40)
    fn = val.loc[num(val["actual_carries"]).ge(25), keep].copy()
    fn["miss_gap"] = num(fn["actual_carries"]) - num(fn["m95i_rush_att"])
    fn = fn.sort_values("miss_gap", ascending=False).head(40)

    downstream = downstream_yard_sensitivity(val, args.m94c_root)
    legacy = pd.read_csv(find_one(args.m94c_root, "m94c_legacy_guard.csv"))
    legacy["m95i_production_rush_yard_change"] = 0
    legacy["m95i_note"] = "M95I changes carry-tail research only; all-player rush-yard production guard is inherited unchanged from M94C. Downstream yard table is sensitivity-only."

    # Final gates: research signal can advance even though production remains blocked by the inherited M94C guard.
    def slice_gain(name: str) -> float:
        q = carry_table.loc[carry_table["slice"].eq(name)]
        return float(q.iloc[0]["mae_gain"]) if not q.empty else np.nan
    all_gain = slice_gain("all_rb"); g05 = slice_gain("actual_0_5"); g610 = slice_gain("actual_6_10"); g1114 = slice_gain("actual_11_14")
    g20 = slice_gain("actual_20_plus"); g25 = slice_gain("actual_25_plus")
    carry_pass = int(
        np.isfinite(all_gain) and all_gain >= -0.10
        and g05 >= -0.15 and g610 >= -0.15 and g1114 >= -0.15
        and g20 >= 0.20 and g25 >= 0.30
    )

    def pm(target, model, sl="all"):
        q = prob_table.loc[(prob_table["target"] == target) & (prob_table["model"] == model) & (prob_table["slice"] == sl)]
        return q.iloc[0] if not q.empty else pd.Series(dtype=float)
    event_passes = []
    for target in ["actual_20plus", "actual_25plus"]:
        b = pm(target, "m95f"); n = pm(target, "m95i_joint")
        event_passes.append(
            np.isfinite(float(b.get("brier", np.nan))) and np.isfinite(float(n.get("brier", np.nan)))
            and float(n.get("brier")) <= float(b.get("brier"))
            and (not np.isfinite(float(b.get("auc", np.nan))) or float(n.get("auc", -np.inf)) >= float(b.get("auc")) - 0.005)
        )
    event_pass = int(all(event_passes))

    sw = stability.loc[stability["slice"].eq("stable_workhorse")]
    if not sw.empty:
        r = sw.iloc[0]
        old_gap = abs(float(r["m95f25"]) - float(r["actual25"]))
        new_gap = abs(float(r["m95i25"]) - float(r["actual25"]))
        stable_pass = int(new_gap <= old_gap * 0.90)
        stable_gap_reduction = old_gap - new_gap
    else:
        stable_pass = 0; stable_gap_reduction = np.nan

    dev_transform_pass = int(chosen_transform.get("development_eligible", 0))
    core_scientific_pass = int(dev_transform_pass and carry_pass and event_pass and stable_pass)
    legacy_gain = float(num(legacy["mae_gain"]).iloc[0]) if "mae_gain" in legacy.columns else np.nan
    production_gate_pass = int(core_scientific_pass and np.isfinite(legacy_gain) and legacy_gain >= 0)
    if core_scientific_pass:
        disposition = "ADVANCE_M95I_SELECTIVE_TAIL_SIGNAL_NOT_PRODUCTION"
    else:
        disposition = "RETAIN_M95I_AS_DIAGNOSTIC_DO_NOT_PROMOTE"

    disposition_df = pd.DataFrame([{
        "share70_spec": SHARE_SPEC, "share70_C": SHARE_C, "share_calibration_shrink_k": share_cal.shrink_k,
        "selected_meta20": selected_meta.loc[selected_meta.target.eq("actual_20plus"), "spec"].iloc[0],
        "selected_meta25": selected_meta.loc[selected_meta.target.eq("actual_25plus"), "spec"].iloc[0],
        "selected_transform": chosen_spec["name"], "high_state_mean_2024": high_mean, "extreme_state_mean_2024": extreme_mean,
        "threshold20": th20, "threshold25": th25,
        "2025_all_mae_gain": all_gain, "2025_0_5_gain": g05, "2025_6_10_gain": g610, "2025_11_14_gain": g1114,
        "2025_20plus_gain": g20, "2025_25plus_gain": g25,
        "stable_workhorse_25_gap_reduction": stable_gap_reduction,
        "development_transform_pass": dev_transform_pass, "carry_pass": carry_pass, "event_pass": event_pass,
        "stable_workhorse_pass": stable_pass, "core_scientific_pass": core_scientific_pass,
        "legacy_guard_gain_inherited": legacy_gain, "production_gate_pass": production_gate_pass,
        "m94c_central_reference_preserved": 1, "sportsbook_inputs": 0, "production_change": 0,
        "disposition": disposition,
    }])

    selected_arch = selected_meta.copy()
    selected_arch["share70_calibration_shrink_k"] = share_cal.shrink_k
    selected_arch["tail_transform"] = chosen_spec["name"]
    selected_arch["high_state_mean_2024"] = high_mean
    selected_arch["extreme_state_mean_2024"] = extreme_mean

    # Persist artifacts.
    source_audit.to_csv(args.out_dir / "m95i_source_audit.csv", index=False)
    cal_grid.to_csv(args.out_dir / "m95i_share70_calibration_selection.csv", index=False)
    q70_audit.to_csv(args.out_dir / "m95i_share70_calibration_audit.csv", index=False)
    pd.concat(meta_grids, ignore_index=True, sort=False).to_csv(args.out_dir / "m95i_meta_candidate_grid.csv", index=False)
    selected_arch.to_csv(args.out_dir / "m95i_selected_architecture.csv", index=False)
    transform_grid.to_csv(args.out_dir / "m95i_2024_transform_grid.csv", index=False)
    pd.concat([th20_grid, th25_grid], ignore_index=True, sort=False).to_csv(args.out_dir / "m95i_2024_threshold_grid.csv", index=False)
    carry_table.to_csv(args.out_dir / "m95i_2025_carry_metrics.csv", index=False)
    prob_table.to_csv(args.out_dir / "m95i_2025_probability_metrics.csv", index=False)
    threshold_table.to_csv(args.out_dir / "m95i_2025_threshold_metrics.csv", index=False)
    distribution.to_csv(args.out_dir / "m95i_2025_projection_distribution.csv", index=False)
    stability.to_csv(args.out_dir / "m95i_2025_stability_audit.csv", index=False)
    fp.to_csv(args.out_dir / "m95i_2025_false_positive_examples.csv", index=False)
    fn.to_csv(args.out_dir / "m95i_2025_false_negative_examples.csv", index=False)
    downstream.to_csv(args.out_dir / "m95i_downstream_yard_sensitivity.csv", index=False)
    legacy.to_csv(args.out_dir / "m95i_legacy_guard.csv", index=False)
    disposition_df.to_csv(args.out_dir / "m95i_disposition.csv", index=False)

    trace_keep = [c for c in PLAYER_KEYS + [
        "actual_carries", "m94c_rush_att", "m95i_rush_att", "m95i_tail_uplift", "m95i_tail_eligible",
        "p_share70_raw", "p_share70_cal", "prior_top1_unavailable", "cal_prob_20", "p20_joint",
        "cal_prob_25", "p25_joint", "candidate_team_rush_att", "role_is_workhorse",
        "rb_rb_share_avg1", "rb_rb_share_avg5", "actual_rush_yards", "actual_rush_rec_yards",
    ] if c in val.columns]
    val[trace_keep].to_csv(args.out_dir / "m95i_2025_trace.csv", index=False)

    print("[m95i] disposition")
    print(disposition_df.to_string(index=False))
    print("\n[m95i] selected architecture")
    print(selected_arch.to_string(index=False))
    print("\n[m95i] 2025 carry metrics")
    print(carry_table.to_string(index=False))
    print("\n[m95i] 2025 probability metrics")
    print(prob_table.to_string(index=False))
    print("\n[m95i] projection distribution")
    print(distribution.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
