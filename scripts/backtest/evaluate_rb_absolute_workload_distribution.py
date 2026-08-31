"""M95E: absolute RB workload decomposition and distribution engine.

Research-only. M94D showed that sharpening a back's share of a fixed RB pool
cannot solve 25+ carry compression. M95E explicitly models the missing bridge:

    full team rush attempts
      x RB-room share of all team rushes
      x player share of the RB room
      = absolute player carries

The upstream full-team rush expectation is frozen from M94C's football-only
structured game-environment model. Exact M94C RB projections are imported from
the frozen M94D artifact only as a comparison baseline. No sportsbook input is
used and no production code is changed.

Protocol:
- train workload/share models on 2023 through 2024 W12
- select one pre-specified architecture on 2024 W13-18 only
- freeze architecture/blend/tail classifier/distribution calibration
- refit share/tail models on all 2023-2024
- untouched 2025 validation
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEAM_KEYS = ["season", "week", "team"]
PLAYER_KEYS = ["season", "week", "team", "player_clean_key"]
SEED = 9517
BLENDS = (0.50, 0.75, 1.00)
TEAM_MAP = {"ARZ":"ARI","JAC":"JAX","LA":"LAR","STL":"LAR","OAK":"LV","SD":"LAC","WSH":"WAS"}

TEAM_ROOM_FEATURES = [
    "team_rb_pool_avg3", "team_rb_pool_avg5",
    "team_total_rush_avg3", "team_total_rush_avg5",
    "team_top1_share_avg3", "team_top1_share_avg5",
    "team_rb_used_avg3", "team_rb_used_avg5",
    "team_qb_rush_share_avg3", "team_qb_rush_share_avg5",
    "team_pbp_plays_avg3", "team_pbp_plays_avg5",
    "team_pbp_rush_rate_avg3", "team_pbp_rush_rate_avg5",
    "team_pbp_early_down_rush_rate_avg3", "team_pbp_early_down_rush_rate_avg5",
    "team_pbp_neutral_rush_rate_avg3", "team_pbp_neutral_rush_rate_avg5",
    "team_pbp_qb_scramble_share_avg3", "team_pbp_qb_scramble_share_avg5",
    "def_rush_att_allowed_avg3", "def_rush_att_allowed_avg5",
    "def_rb_carries_allowed_avg3", "def_rb_carries_allowed_avg5",
    "home",
]

PLAYER_SHARE_FEATURES = [
    "rb_games_before",
    "rb_carries_avg1", "rb_carries_avg3", "rb_carries_avg5",
    "rb_rb_share_avg1", "rb_rb_share_avg3", "rb_rb_share_avg5",
    "rb_15plus_rate3", "rb_15plus_rate5",
    "rb_20plus_rate3", "rb_20plus_rate5",
    "role_is_workhorse", "role_is_starter_plus",
    "off_role_opportunity_score",
    "team_rb_pool_avg3", "team_rb_pool_avg5",
    "team_total_rush_avg3", "team_total_rush_avg5",
    "team_top1_share_avg3", "team_top1_share_avg5",
    "team_rb_used_avg3", "team_rb_used_avg5",
    "team_qb_rush_share_avg3", "team_qb_rush_share_avg5",
    "team_pbp_plays_avg3", "team_pbp_plays_avg5",
    "team_pbp_rush_rate_avg3", "team_pbp_rush_rate_avg5",
    "team_pbp_neutral_rush_rate_avg3", "team_pbp_neutral_rush_rate_avg5",
    "team_pbp_early_down_rush_rate_avg3", "team_pbp_early_down_rush_rate_avg5",
    "def_rb_carries_allowed_avg3", "def_rb_carries_allowed_avg5",
    "def_top_rb_carries_allowed_avg3", "def_top_rb_carries_allowed_avg5",
    "def_rb_20plus_carry_rate_allowed_avg3", "def_rb_20plus_carry_rate_allowed_avg5",
    "home",
]

TAIL_FEATURES = PLAYER_SHARE_FEATURES + [
    "prior_room_share", "prior_player_share_norm", "prior_abs_share",
    "prior_expected_carries",
]


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return hits[0]


def load_trace(root: Path) -> pd.DataFrame:
    x = lower(pd.read_csv(find_one(root, "m95b_rb_matchup_trace.csv"), low_memory=False))
    x["season"] = num(x["season"]).astype(int)
    x["week"] = num(x["week"]).astype(int)
    x = x.loc[x["season"].isin([2023, 2024, 2025]) & x["week"].between(1, 18)].copy()
    if "player_clean_key" not in x:
        raise RuntimeError("M95B trace missing player_clean_key")
    x["actual_carries"] = num(x["actual_carries"])
    return x.reset_index(drop=True)


def load_m94c_team(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = lower(pd.read_csv(find_one(root, "m94c_2024_holdout_trace.csv"), low_memory=False))
    v = lower(pd.read_csv(find_one(root, "m94c_2025_team_trace.csv"), low_memory=False))
    need = TEAM_KEYS + [
        "structured_team_rush_att", "candidate_team_rush_att",
        "actual_rush_att_pbp", "actual_team_rush_att",
    ]
    for frame, label in [(h, "2024"), (v, "2025")]:
        miss = [c for c in need if c not in frame]
        if miss:
            raise RuntimeError(f"M94C {label} team trace missing {miss}")
    return h, v


def load_m94d_rb(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = lower(pd.read_csv(find_one(root, "m94d_2024_holdout_rb_trace.csv"), low_memory=False))
    v = lower(pd.read_csv(find_one(root, "m94d_2025_rb_trace.csv"), low_memory=False))
    need = PLAYER_KEYS + ["actual_rush_att", "m94c_rush_att", "bellcow_60"]
    for frame, label in [(h, "2024"), (v, "2025")]:
        miss = [c for c in need if c not in frame]
        if miss:
            raise RuntimeError(f"M94D {label} RB trace missing {miss}")
    return h, v


def read_pbp_totals(root: Path) -> pd.DataFrame:
    frames = []
    for season in (2023, 2024, 2025):
        p = root / f"play_by_play_{season}.parquet"
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"missing PBP {p}")
        schema = pd.read_parquet(p, engine="pyarrow").columns.tolist()
        use = [c for c in ["season", "week", "posteam", "rush_attempt", "qb_kneel"] if c in schema]
        z = lower(pd.read_parquet(p, columns=use, engine="pyarrow"))
        for c in ["season", "week", "rush_attempt", "qb_kneel"]:
            z[c] = num(z.get(c, pd.Series(index=z.index, dtype=float)))
        z = z.loc[z["rush_attempt"].eq(1) & z["qb_kneel"].fillna(0).ne(1) & z["posteam"].notna()].copy()
        z["posteam"] = z["posteam"].astype(str).str.upper().str.strip().replace(TEAM_MAP)
        g = z.groupby(["season", "week", "posteam"], as_index=False).size().rename(
            columns={"posteam": "team", "size": "actual_total_team_rush_pbp"}
        )
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["season"] = num(out["season"]).astype(int)
    out["week"] = num(out["week"]).astype(int)
    return out


def add_targets(trace: pd.DataFrame, pbp_totals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    room = trace.groupby(TEAM_KEYS, as_index=False).agg(
        actual_rb_pool=("actual_carries", lambda s: num(s).sum(min_count=1)),
        rb_rows=("player_clean_key", "size"),
    )
    team = room.merge(pbp_totals, on=TEAM_KEYS, how="left", validate="one_to_one")
    if team["actual_total_team_rush_pbp"].isna().mean() > 0.01:
        raise RuntimeError("PBP target join incomplete")
    team["actual_room_share"] = (
        team["actual_rb_pool"] / team["actual_total_team_rush_pbp"].replace(0, np.nan)
    ).clip(0, 1)

    x = trace.merge(
        team[TEAM_KEYS + ["actual_rb_pool", "actual_total_team_rush_pbp", "actual_room_share"]],
        on=TEAM_KEYS, how="left", validate="many_to_one",
    )
    x["actual_player_rb_share"] = x["actual_carries"] / x["actual_rb_pool"].replace(0, np.nan)
    x["actual_abs_team_share"] = x["actual_carries"] / x["actual_total_team_rush_pbp"].replace(0, np.nan)
    x["actual_20plus"] = x["actual_carries"].ge(20).astype(int)
    x["actual_25plus"] = x["actual_carries"].ge(25).astype(int)
    return x, team


def _coalesce(x: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=x.index, dtype=float)
    for c in cols:
        if c in x:
            out = out.fillna(num(x[c]))
    return out


def add_priors(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    pool = _coalesce(z, ["team_rb_pool_avg5", "team_rb_pool_avg3", "team_rb_pool_avg1"])
    total = _coalesce(z, ["team_total_rush_avg5", "team_total_rush_avg3", "team_total_rush_avg1"])
    z["prior_room_share"] = (pool / total.replace(0, np.nan)).clip(0.35, 0.98)

    z["prior_player_share"] = _coalesce(z, ["rb_rb_share_avg5", "rb_rb_share_avg3", "rb_rb_share_avg1"]).clip(0.005, 0.98)
    c = _coalesce(z, ["rb_carries_avg5", "rb_carries_avg3", "rb_carries_avg1"])
    z["prior_player_share"] = z["prior_player_share"].fillna((c / pool.replace(0, np.nan)).clip(0.005, 0.98))
    z["prior_player_share"] = z["prior_player_share"].fillna(0.20)

    denom = z.groupby(TEAM_KEYS)["prior_player_share"].transform("sum").replace(0, np.nan)
    z["prior_player_share_norm"] = (z["prior_player_share"] / denom).fillna(
        1.0 / z.groupby(TEAM_KEYS)["player_clean_key"].transform("count").clip(lower=1)
    )
    z["prior_room_share"] = z["prior_room_share"].fillna(0.78)
    z["prior_abs_share"] = (z["prior_room_share"] * z["prior_player_share_norm"]).clip(0.001, 0.98)
    z["prior_expected_carries"] = (
        _coalesce(z, ["team_total_rush_avg5", "team_total_rush_avg3", "team_total_rush_avg1"]).fillna(25)
        * z["prior_abs_share"]
    )
    return z


def available(x: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in x and num(x[c]).notna().sum() >= 20]


def regressors(seed: int) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=12.0)),
        ]),
        "gbr": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                random_state=seed, n_estimators=180, learning_rate=0.03,
                max_depth=2, min_samples_leaf=10, loss="huber",
            )),
        ]),
        "rf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                random_state=seed, n_estimators=350, max_depth=5,
                min_samples_leaf=8, max_features=0.70, n_jobs=-1,
            )),
        ]),
    }


def classifiers(seed: int) -> dict[str, Pipeline]:
    return {
        "logit": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=0.30, class_weight="balanced", max_iter=2500, random_state=seed
            )),
        ]),
        "rf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                random_state=seed, n_estimators=450, max_depth=5,
                min_samples_leaf=8, max_features=0.70, class_weight="balanced_subsample",
                n_jobs=-1,
            )),
        ]),
    }


def fit_component_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    family: str,
) -> pd.DataFrame:
    team_feats = available(train, TEAM_ROOM_FEATURES)
    player_feats = available(train, PLAYER_SHARE_FEATURES)
    if not team_feats or not player_feats:
        raise RuntimeError("M95E feature sets empty")

    train_team = train.drop_duplicates(TEAM_KEYS).copy()
    test_team = test.drop_duplicates(TEAM_KEYS).copy()
    room_model = regressors(SEED + 1)[family]
    room_fit = train_team.loc[train_team["actual_room_share"].notna()].copy()
    room_model.fit(room_fit[team_feats], room_fit["actual_room_share"])
    room_pred = np.clip(room_model.predict(test_team[team_feats]), 0.35, 0.98)
    room_map = test_team[TEAM_KEYS].copy()
    room_map["model_room_share"] = room_pred

    pfit = train.loc[train["actual_player_rb_share"].notna()].copy()
    share_model = regressors(SEED + 2)[family]
    share_model.fit(pfit[player_feats], pfit["actual_player_rb_share"])
    player_pred = np.clip(share_model.predict(test[player_feats]), 0.002, 0.995)

    afit = train.loc[train["actual_abs_team_share"].notna()].copy()
    abs_model = regressors(SEED + 3)[family]
    abs_model.fit(afit[player_feats], afit["actual_abs_team_share"])
    abs_pred = np.clip(abs_model.predict(test[player_feats]), 0.001, 0.98)

    out = test[PLAYER_KEYS].copy()
    out["model_player_share"] = player_pred
    out["model_abs_share"] = abs_pred
    out = out.merge(room_map, on=TEAM_KEYS, how="left", validate="many_to_one")
    return out


def normalize_player_share(x: pd.DataFrame, col: str) -> pd.Series:
    v = num(x[col]).clip(0.002, 0.995)
    den = v.groupby([x[k] for k in TEAM_KEYS]).transform("sum").replace(0, np.nan)
    n = v / den
    count = x.groupby(TEAM_KEYS)["player_clean_key"].transform("count").clip(lower=1)
    return n.fillna(1.0 / count)


def prepare_eval_frame(
    feature_trace: pd.DataFrame,
    rb_baseline: pd.DataFrame,
    team_env: pd.DataFrame,
) -> pd.DataFrame:
    cols = PLAYER_KEYS + ["actual_rush_att", "m94c_rush_att", "bellcow_60"]
    base = rb_baseline[cols].copy()
    feat_cols = [c for c in feature_trace.columns if c not in {"actual_carries"}]
    out = base.merge(feature_trace[feat_cols], on=PLAYER_KEYS, how="left", validate="one_to_one")
    tcols = TEAM_KEYS + [
        "structured_team_rush_att", "actual_rush_att_pbp", "candidate_team_rush_att"
    ]
    out = out.merge(team_env[tcols], on=TEAM_KEYS, how="left", validate="many_to_one")
    if out["structured_team_rush_att"].isna().any():
        raise RuntimeError("M94C structured team rush join incomplete")
    out["actual_carries"] = num(out["actual_rush_att"])
    return out


def candidate_from_components(
    base: pd.DataFrame,
    pred: pd.DataFrame,
    mode: str,
    blend: float,
) -> pd.DataFrame:
    x = base.copy()
    x = x.merge(pred, on=PLAYER_KEYS, how="left", validate="one_to_one")
    room = ((1 - blend) * x["prior_room_share"] + blend * x["model_room_share"]).clip(0.35, 0.98)
    p_raw = ((1 - blend) * x["prior_player_share_norm"] + blend * x["model_player_share"]).clip(0.002, 0.995)
    x["_blend_player_raw"] = p_raw
    p_norm = normalize_player_share(x, "_blend_player_raw")
    if mode == "decomp":
        abs_share = (room * p_norm).clip(0.001, 0.98)
    elif mode == "direct":
        abs_share = ((1 - blend) * x["prior_abs_share"] + blend * x["model_abs_share"]).clip(0.001, 0.98)
        sums = abs_share.groupby([x[k] for k in TEAM_KEYS]).transform("sum")
        scale = np.maximum(sums / 0.98, 1.0)
        abs_share = abs_share / scale
    else:
        raise ValueError(mode)
    x["m95e_room_share"] = room
    x["m95e_player_rb_share"] = p_norm
    x["m95e_abs_share"] = abs_share
    x["m95e_carries"] = (
        num(x["structured_team_rush_att"]) * x["m95e_abs_share"]
    ).clip(0, 45)
    return x.drop(columns=["_blend_player_raw"])


def metrics(actual, pred) -> dict[str, float | int]:
    z = pd.DataFrame({"a": num(actual), "p": num(pred)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
    e = z["p"] - z["a"]
    corr = float(z["a"].corr(z["p"])) if z["a"].nunique() > 1 and z["p"].nunique() > 1 else np.nan
    return {
        "n": int(len(z)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.square(e).mean())),
        "bias": float(e.mean()),
        "corr": corr,
    }


def slice_masks(x: pd.DataFrame) -> dict[str, pd.Series]:
    a = num(x["actual_carries"])
    return {
        "all_rb": pd.Series(True, index=x.index),
        "actual_0_5": a.le(5),
        "actual_6_10": a.between(6, 10),
        "actual_11_14": a.between(11, 14),
        "actual_15_plus": a.ge(15),
        "actual_20_plus": a.ge(20),
        "actual_25_plus": a.ge(25),
        "bellcow_60": x.get("bellcow_60", pd.Series(False, index=x.index)).fillna(False).astype(bool),
    }


def comparison(x: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    for name, mask in slice_masks(x).items():
        g = x.loc[mask]
        b = metrics(g["actual_carries"], g["m94c_rush_att"])
        c = metrics(g["actual_carries"], g["m95e_carries"])
        rows.append({
            "scope": scope, "slice": name, "n": b["n"],
            "m94c_mae": b["mae"], "m95e_mae": c["mae"], "mae_gain": b["mae"] - c["mae"],
            "m94c_rmse": b["rmse"], "m95e_rmse": c["rmse"],
            "m94c_bias": b["bias"], "m95e_bias": c["bias"],
            "m94c_corr": b["corr"], "m95e_corr": c["corr"],
        })
    return pd.DataFrame(rows)


def grid_row(x: pd.DataFrame, family: str, mode: str, blend: float) -> dict:
    cmp = comparison(x, "2024_w13_18_architecture_holdout")
    def g(s):
        q = cmp.loc[cmp["slice"].eq(s), "mae_gain"]
        return float(q.iloc[0]) if len(q) else np.nan
    row = {
        "family": family, "mode": mode, "blend": blend,
        "all_gain": g("all_rb"),
        "gain_0_5": g("actual_0_5"),
        "gain_6_10": g("actual_6_10"),
        "gain_11_14": g("actual_11_14"),
        "gain_15_plus": g("actual_15_plus"),
        "gain_20_plus": g("actual_20_plus"),
        "gain_25_plus": g("actual_25_plus"),
    }
    row["eligible"] = int(
        np.isfinite(row["all_gain"])
        and row["all_gain"] >= -0.05
        and row["gain_6_10"] >= -0.10
        and row["gain_11_14"] >= -0.10
        and row["gain_20_plus"] > 0
        and row["gain_25_plus"] > 0
    )
    row["selection_score"] = (
        row["gain_25_plus"] + row["gain_20_plus"] + 0.50 * row["all_gain"]
        + 0.20 * row["gain_6_10"] + 0.20 * row["gain_11_14"]
    )
    row["projected_20_plus"] = int(num(x["m95e_carries"]).ge(20).sum())
    row["projected_25_plus"] = int(num(x["m95e_carries"]).ge(25).sum())
    row["actual_20_plus"] = int(num(x["actual_carries"]).ge(20).sum())
    row["actual_25_plus"] = int(num(x["actual_carries"]).ge(25).sum())
    return row


def select_mean_architecture(train: pd.DataFrame, hold: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    rows = []
    candidates: dict[tuple[str, str, float], pd.DataFrame] = {}
    for family in regressors(SEED):
        pred = fit_component_predictions(train, hold, family)
        for mode in ("decomp", "direct"):
            for blend in BLENDS:
                cand = candidate_from_components(hold, pred, mode, blend)
                rows.append(grid_row(cand, family, mode, blend))
                candidates[(family, mode, blend)] = cand
    grid = pd.DataFrame(rows)
    elig = grid.loc[grid["eligible"].eq(1)].copy()
    pool = elig if len(elig) else grid
    pool = pool.sort_values(
        ["selection_score", "gain_25_plus", "gain_20_plus", "all_gain"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    chosen = pool.iloc[0].to_dict()
    key = (str(chosen["family"]), str(chosen["mode"]), float(chosen["blend"]))
    return chosen, grid.sort_values("selection_score", ascending=False), candidates[key]


def classifier_metrics(y, p, threshold: float | None = None) -> dict:
    y = num(y)
    p = num(pd.Series(p, index=y.index))
    z = pd.DataFrame({"y": y, "p": p}).dropna()
    auc = float(roc_auc_score(z["y"], z["p"])) if z["y"].nunique() > 1 else np.nan
    out = {"n": int(len(z)), "auc": auc}
    if threshold is not None:
        pred = z["p"].ge(threshold)
        truth = z["y"].eq(1)
        tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.update({
            "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "predicted_positive": int(pred.sum()), "actual_positive": int(truth.sum()),
        })
    return out


def choose_threshold(y, p) -> tuple[float, pd.DataFrame]:
    rows = []
    for t in (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        rows.append(classifier_metrics(y, p, t))
    grid = pd.DataFrame(rows).sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).reset_index(drop=True)
    return float(grid.iloc[0]["threshold"]), grid


def select_tail_classifier(train: pd.DataFrame, hold: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    feats = available(train, TAIL_FEATURES)
    rows = []
    probs = {}
    for family, model0 in classifiers(SEED + 30).items():
        fam_probs = {}
        aucs = []
        for label in ("actual_20plus", "actual_25plus"):
            fit = train.loc[train[label].notna()].copy()
            model = classifiers(SEED + 30)[family]
            model.fit(fit[feats], fit[label].astype(int))
            p = model.predict_proba(hold[feats])[:, 1]
            fam_probs[label] = p
            m = classifier_metrics(hold[label], p)
            aucs.append(m["auc"])
            rows.append({"family": family, "target": label, "auc": m["auc"], "n": m["n"]})
        probs[family] = fam_probs
        rows.append({"family": family, "target": "mean_auc", "auc": float(np.nanmean(aucs)), "n": len(hold)})
    audit = pd.DataFrame(rows)
    means = audit.loc[audit["target"].eq("mean_auc")].sort_values(["auc", "family"], ascending=[False, True])
    chosen_family = str(means.iloc[0]["family"])
    out = {"family": chosen_family}
    threshold_frames = []
    for label in ("actual_20plus", "actual_25plus"):
        th, grid = choose_threshold(hold[label], probs[chosen_family][label])
        out[f"{label}_threshold"] = th
        grid.insert(0, "target", label)
        threshold_frames.append(grid)
    return out, pd.concat([audit, pd.concat(threshold_frames, ignore_index=True)], ignore_index=True, sort=False)


def fit_tail_probs(train: pd.DataFrame, test: pd.DataFrame, family: str) -> pd.DataFrame:
    feats = available(train, TAIL_FEATURES)
    out = test[PLAYER_KEYS].copy()
    for label, suffix in [("actual_20plus", "20"), ("actual_25plus", "25")]:
        model = classifiers(SEED + 40)[family]
        model.fit(train[feats], train[label].astype(int))
        out[f"tail_prob_{suffix}"] = model.predict_proba(test[feats])[:, 1]
    return out


def auc_safe(y, p) -> float:
    z = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    return float(roc_auc_score(z["y"], z["p"])) if z["y"].nunique() > 1 else np.nan


def calibrate_distribution(hold: pd.DataFrame, team_hold: pd.DataFrame) -> dict[str, float]:
    team = team_hold.copy()
    e = num(team["actual_rush_att_pbp"]) - num(team["structured_team_rush_att"])
    sigma_team = float(np.sqrt(np.nanmean(np.square(e))))

    p = num(hold["m95e_abs_share"]).clip(0.005, 0.95)
    y = (num(hold["actual_carries"]) / num(hold["actual_rush_att_pbp"]).replace(0, np.nan)).clip(0, 1)
    mse = float(np.nanmean(np.square(y - p)))
    numer = float(np.nanmean(p * (1 - p)))
    k = numer / max(mse, 1e-6) - 1.0
    k = float(np.clip(k, 2.0, 120.0))
    return {"team_rush_sigma": sigma_team, "abs_share_concentration": k}


def simulate_distribution(
    x: pd.DataFrame, cal: dict[str, float], draws: int = 2500
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 99)
    rows = []
    sigma = float(cal["team_rush_sigma"])
    k = float(cal["abs_share_concentration"])
    for r in x.itertuples(index=False):
        mu_team = float(getattr(r, "structured_team_rush_att"))
        p = float(np.clip(getattr(r, "m95e_abs_share"), 0.002, 0.98))
        n = np.rint(rng.normal(mu_team, sigma, draws)).astype(int)
        n = np.clip(n, 8, 55)
        a = max(p * k, 0.05); b = max((1 - p) * k, 0.05)
        pd_ = rng.beta(a, b, draws)
        c = rng.binomial(n, pd_)
        rows.append({
            **{q: getattr(r, q) for q in PLAYER_KEYS},
            "carry_p50": float(np.quantile(c, 0.50)),
            "carry_p75": float(np.quantile(c, 0.75)),
            "carry_p90": float(np.quantile(c, 0.90)),
            "carry_p95": float(np.quantile(c, 0.95)),
            "sim_prob_20plus": float(np.mean(c >= 20)),
            "sim_prob_25plus": float(np.mean(c >= 25)),
        })
    return pd.DataFrame(rows)


def tail_diagnostics(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a = num(x["actual_carries"])
    for name, mask in {
        "all_rb": pd.Series(True, index=x.index),
        "actual_20_plus": a.ge(20),
        "actual_25_plus": a.ge(25),
    }.items():
        g = x.loc[mask]
        row = {"slice": name, "n": int(len(g))}
        for label, col in [("actual", "actual_carries"), ("m94c", "m94c_rush_att"), ("m95e", "m95e_carries"),
                           ("p50", "carry_p50"), ("p90", "carry_p90"), ("p95", "carry_p95")]:
            v = num(g[col]).dropna()
            row[f"{label}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{label}_median"] = float(v.median()) if len(v) else np.nan
            row[f"{label}_max"] = float(v.max()) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def projection_counts(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in (18, 20, 22, 25):
        rows.append({
            "threshold": t,
            "actual_count": int(num(x["actual_carries"]).ge(t).sum()),
            "m94c_count": int(num(x["m94c_rush_att"]).ge(t).sum()),
            "m95e_mean_count": int(num(x["m95e_carries"]).ge(t).sum()),
            "m95e_p50_count": int(num(x["carry_p50"]).ge(t).sum()),
            "m95e_p90_count": int(num(x["carry_p90"]).ge(t).sum()),
            "m95e_p95_count": int(num(x["carry_p95"]).ge(t).sum()),
        })
    return pd.DataFrame(rows)


def distribution_summary(x: pd.DataFrame) -> pd.DataFrame:
    a = num(x["actual_carries"])
    rows = []
    for q, col in [(0.50, "carry_p50"), (0.75, "carry_p75"), (0.90, "carry_p90"), (0.95, "carry_p95")]:
        rows.append({
            "quantile": q, "coverage": float((a <= num(x[col])).mean()),
            "mean_projected_quantile": float(num(x[col]).mean()),
        })
    rows.append({"quantile": 20, "coverage": auc_safe(a.ge(20).astype(int), x["sim_prob_20plus"]),
                 "mean_projected_quantile": float(num(x["sim_prob_20plus"]).mean())})
    rows.append({"quantile": 25, "coverage": auc_safe(a.ge(25).astype(int), x["sim_prob_25plus"]),
                 "mean_projected_quantile": float(num(x["sim_prob_25plus"]).mean())})
    return pd.DataFrame(rows)


def team_pool_summary(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = x.groupby(TEAM_KEYS, as_index=False).agg(
        actual_rb_pool=("actual_carries", "sum"),
        m94c_rb_pool=("m94c_rush_att", "sum"),
        m95e_rb_pool=("m95e_carries", "sum"),
        structured_team_rush_att=("structured_team_rush_att", "first"),
        actual_total_team_rush_pbp=("actual_rush_att_pbp", "first"),
        predicted_room_share=("m95e_room_share", "first"),
    )
    g["actual_room_share"] = g["actual_rb_pool"] / g["actual_total_team_rush_pbp"].replace(0, np.nan)
    rows = []
    for name, mask in {
        "all_team_games": pd.Series(True, index=g.index),
        "actual_rb_pool_20_plus": g["actual_rb_pool"].ge(20),
        "actual_rb_pool_25_plus": g["actual_rb_pool"].ge(25),
    }.items():
        z = g.loc[mask]
        b = metrics(z["actual_rb_pool"], z["m94c_rb_pool"])
        c = metrics(z["actual_rb_pool"], z["m95e_rb_pool"])
        rows.append({
            "slice": name, "n": len(z),
            "m94c_rb_pool_mae": b["mae"], "m95e_rb_pool_mae": c["mae"],
            "rb_pool_mae_gain": b["mae"] - c["mae"],
            "m94c_rb_pool_bias": b["bias"], "m95e_rb_pool_bias": c["bias"],
        })
    return pd.DataFrame(rows), g


def calibration_bins(frame: pd.DataFrame, score: str, actual: str, label: str) -> pd.DataFrame:
    z = frame[[score, actual]].copy().dropna()
    if len(z) < 10:
        return pd.DataFrame()
    try:
        z["bin"] = pd.qcut(z[score], 5, labels=False, duplicates="drop") + 1
    except ValueError:
        z["bin"] = 1
    out = z.groupby("bin", as_index=False).agg(
        n=(actual, "size"), predicted=(score, "mean"), actual=(actual, "mean")
    )
    out.insert(0, "calibration", label)
    return out


def false_examples(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = PLAYER_KEYS + ["actual_carries", "m94c_rush_att", "m95e_carries", "tail_prob_20", "tail_prob_25", "carry_p90", "carry_p95"]
    fp = x.loc[(num(x["m95e_carries"]).ge(18)) & (num(x["actual_carries"]).le(14)), cols].copy()
    fp["over_error"] = num(fp["m95e_carries"]) - num(fp["actual_carries"])
    fp = fp.sort_values("over_error", ascending=False).head(30)
    fn = x.loc[num(x["actual_carries"]).ge(25), cols].copy()
    fn["under_error"] = num(fn["actual_carries"]) - num(fn["m95e_carries"])
    fn = fn.sort_values("under_error", ascending=False).head(30)
    return fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m95b-root", type=Path, required=True)
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--m94d-root", type=Path, required=True)
    ap.add_argument("--pbp-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_m95e"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trace0 = add_priors(load_trace(args.m95b_root))
    pbp = read_pbp_totals(args.pbp_root)
    trace, team_targets = add_targets(trace0, pbp)
    trace = add_priors(trace)

    team24, team25 = load_m94c_team(args.m94c_root)
    rb24, rb25 = load_m94d_rb(args.m94d_root)

    w = num(trace["week"])
    dev_train = trace.loc[(trace["season"].eq(2023)) | (trace["season"].eq(2024) & w.le(12))].copy()
    hold_feat = trace.loc[trace["season"].eq(2024) & w.ge(13)].copy()
    hold = prepare_eval_frame(hold_feat, rb24, team24)
    hold = add_priors(hold)

    chosen, grid, chosen_hold = select_mean_architecture(dev_train, hold)
    tail_choice, tail_dev_audit = select_tail_classifier(dev_train, hold)

    final_train = trace.loc[trace["season"].isin([2023, 2024])].copy()
    test_feat = trace.loc[trace["season"].eq(2025)].copy()
    test = add_priors(prepare_eval_frame(test_feat, rb25, team25))
    pred25 = fit_component_predictions(final_train, test, str(chosen["family"]))
    final25 = candidate_from_components(
        test, pred25, str(chosen["mode"]), float(chosen["blend"])
    )

    tailp = fit_tail_probs(final_train, final25, str(tail_choice["family"]))
    final25 = final25.merge(tailp, on=PLAYER_KEYS, how="left", validate="one_to_one")

    if "actual_rush_att_pbp" not in chosen_hold:
        chosen_hold = chosen_hold.merge(
            team24[TEAM_KEYS + ["actual_rush_att_pbp"]],
            on=TEAM_KEYS, how="left", validate="many_to_one",
        )
    cal = calibrate_distribution(chosen_hold, team24)
    sim = simulate_distribution(final25, cal)
    final25 = final25.merge(sim, on=PLAYER_KEYS, how="left", validate="one_to_one")

    cmp24 = comparison(chosen_hold, "2024_w13_18_architecture_holdout")
    cmp25 = comparison(final25, "2025_untouched_validation")

    tail_rows = []
    for target, suffix in [("actual_20plus", "20"), ("actual_25plus", "25")]:
        feats = available(dev_train, TAIL_FEATURES)
        model = classifiers(SEED + 30)[str(tail_choice["family"])]
        model.fit(dev_train[feats], dev_train[target].astype(int))
        hp = model.predict_proba(hold[feats])[:, 1]
        th = float(tail_choice[f"{target}_threshold"])
        tail_rows.append({"scope": "2024_w13_18_architecture_holdout", "target": target,
                          **classifier_metrics(hold[target], hp, th)})
        y25 = final25[target].astype(int)
        tail_rows.append({"scope": "2025_untouched_validation", "target": target,
                          **classifier_metrics(y25, final25[f"tail_prob_{suffix}"], th)})
    tail_metrics = pd.DataFrame(tail_rows)

    pool_summary, pool_games = team_pool_summary(final25)
    room_cal = calibration_bins(pool_games, "predicted_room_share", "actual_room_share", "rb_room_share")
    lead25 = final25.sort_values(["season", "week", "team", "m95e_carries"], ascending=[True, True, True, False]).drop_duplicates(TEAM_KEYS)
    lead25["actual_abs_share"] = lead25["actual_carries"] / lead25["actual_rush_att_pbp"].replace(0, np.nan)
    abs_cal = calibration_bins(lead25, "m95e_abs_share", "actual_abs_share", "lead_rb_absolute_team_rush_share")
    calibration = pd.concat([room_cal, abs_cal], ignore_index=True, sort=False)

    fp, fn = false_examples(final25)
    tail_diag = tail_diagnostics(final25)
    counts = projection_counts(final25)
    dist = distribution_summary(final25)

    def gain(table, slice_name):
        q = table.loc[table["slice"].eq(slice_name), "mae_gain"]
        return float(q.iloc[0]) if len(q) else np.nan

    all_gain = gain(cmp25, "all_rb")
    g6 = gain(cmp25, "actual_6_10")
    g11 = gain(cmp25, "actual_11_14")
    g20 = gain(cmp25, "actual_20_plus")
    g25 = gain(cmp25, "actual_25_plus")
    dev_eligible = int(chosen.get("eligible", 0))
    validation_pass = int(
        dev_eligible == 1
        and all_gain > 0
        and g6 >= -0.10
        and g11 >= -0.10
        and g20 > 0
        and g25 > 0
    )
    disposition = (
        "ADVANCE_M95E_COMPONENT_FOR_INTEGRATION_REVIEW"
        if validation_pass
        else "RETAIN_M95E_AS_DIAGNOSTIC_DO_NOT_PROMOTE"
    )
    disp = pd.DataFrame([{
        "selected_family": chosen["family"],
        "selected_mode": chosen["mode"],
        "selected_blend": chosen["blend"],
        "development_eligible": dev_eligible,
        "2025_all_gain_vs_m94c": all_gain,
        "2025_6_10_gain_vs_m94c": g6,
        "2025_11_14_gain_vs_m94c": g11,
        "2025_20plus_gain_vs_m94c": g20,
        "2025_25plus_gain_vs_m94c": g25,
        "tail_classifier_family": tail_choice["family"],
        "tail20_threshold": tail_choice["actual_20plus_threshold"],
        "tail25_threshold": tail_choice["actual_25plus_threshold"],
        "team_rush_sigma": cal["team_rush_sigma"],
        "abs_share_concentration": cal["abs_share_concentration"],
        "validation_pass": validation_pass,
        "disposition": disposition,
        "production_change": 0,
    }])

    source_audit = pd.DataFrame([
        {"source": "m95b_frozen_trace", "rows": len(trace), "status": "ok"},
        {"source": "m94c_2024_team_holdout", "rows": len(team24), "status": "ok"},
        {"source": "m94c_2025_team_validation", "rows": len(team25), "status": "ok"},
        {"source": "m94d_2024_rb_holdout", "rows": len(rb24), "status": "ok"},
        {"source": "m94d_2025_rb_validation", "rows": len(rb25), "status": "ok"},
        {"source": "pbp_team_week_targets", "rows": len(pbp), "status": "ok"},
        {"source": "team_room_features", "rows": len(available(dev_train, TEAM_ROOM_FEATURES)), "status": "feature_count"},
        {"source": "player_share_features", "rows": len(available(dev_train, PLAYER_SHARE_FEATURES)), "status": "feature_count"},
    ])

    grid.to_csv(args.out_dir / "m95e_2024_architecture_grid.csv", index=False)
    pd.concat([cmp24, cmp25], ignore_index=True).to_csv(args.out_dir / "m95e_carry_comparison.csv", index=False)
    final25.to_csv(args.out_dir / "m95e_2025_rb_trace.csv", index=False)
    pool_summary.to_csv(args.out_dir / "m95e_2025_rb_pool_summary.csv", index=False)
    pool_games.to_csv(args.out_dir / "m95e_2025_team_pool_trace.csv", index=False)
    calibration.to_csv(args.out_dir / "m95e_2025_share_calibration.csv", index=False)
    tail_dev_audit.to_csv(args.out_dir / "m95e_2024_tail_model_audit.csv", index=False)
    tail_metrics.to_csv(args.out_dir / "m95e_tail_classification.csv", index=False)
    tail_diag.to_csv(args.out_dir / "m95e_2025_tail_diagnostics.csv", index=False)
    counts.to_csv(args.out_dir / "m95e_2025_projection_counts.csv", index=False)
    dist.to_csv(args.out_dir / "m95e_2025_distribution_summary.csv", index=False)
    fp.to_csv(args.out_dir / "m95e_2025_false_positive_examples.csv", index=False)
    fn.to_csv(args.out_dir / "m95e_2025_false_negative_25plus.csv", index=False)
    source_audit.to_csv(args.out_dir / "m95e_source_audit.csv", index=False)
    disp.to_csv(args.out_dir / "m95e_disposition.csv", index=False)

    print("[m95e] disposition")
    print(disp.to_string(index=False))
    print("\n[m95e] carry comparison")
    print(pd.concat([cmp24, cmp25], ignore_index=True).to_string(index=False))
    print("\n[m95e] tail diagnostics")
    print(tail_diag.to_string(index=False))
    print("\n[m95e] tail classification")
    print(tail_metrics.to_string(index=False))
    print("\n[m95e] projection counts")
    print(counts.to_string(index=False))
    print("\n[m95e] RB-pool summary")
    print(pool_summary.to_string(index=False))
    print("\n[m95e] distribution")
    print(dist.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
