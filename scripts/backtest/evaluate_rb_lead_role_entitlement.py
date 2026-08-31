"""M95H: recipient-specific RB lead-role entitlement model.

Research-only. M95G established that current-week roster/depth/availability
information contains incremental signal for 20+ workloads, but its generic
"vacated lead role" feature applied to multiple surviving RBs. M95H asks a
more precise football question: which specific RB is entitled to the upcoming
backfield workload?

Targets (all strictly outcome labels, never model inputs):
- actual_lead_rb: deterministic team RB carry leader for the game;
- actual_share60: player receives >=60% of team RB carries;
- actual_share70: player receives >=70% of team RB carries.

Protocol:
1. Build leakage-safe current-week role/availability features using the frozen
   M95G v5 source contracts.
2. Fit on temporal 2024 OOF rows from Weeks 5-12.
3. Select target-specific architecture on 2024 Weeks 13-18 only.
4. Freeze architecture, refit on all eligible 2024 temporal OOF rows, and
   evaluate once on untouched 2025.
5. M94C remains the official central carry mean. No sportsbook inputs and no
   production changes are made in M95H.
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

import scripts.backtest.evaluate_rb_role_availability_v5  # noqa: F401
import scripts.backtest.evaluate_rb_role_availability as g

SEED = 95108
PLAYER_KEYS = g.PLAYER_KEYS
TEAM_KEYS = g.TEAM_KEYS
TARGETS = ("actual_lead_rb", "actual_share60", "actual_share70")
C_GRID = (0.03, 0.08, 0.20, 0.50)

HISTORY_FEATURES = [
    "rb_games_before",
    "rb_carries_avg1", "rb_carries_avg3", "rb_carries_avg5",
    "rb_rb_share_avg1", "rb_rb_share_avg3", "rb_rb_share_avg5",
    "rb_targets_avg1", "rb_targets_avg3", "rb_targets_avg5",
    "role_is_workhorse", "role_is_starter_plus",
    "target_was_prior_top1", "target_was_prior_top2",
    "prior_top1_carries", "prior_top2_carries",
    "hist_share_rank", "hist_carry_rank", "hist_target_rank",
    "hist_best_share", "hist_best_carries",
    "team_top1_share_avg1", "team_top1_share_avg3", "team_top1_share_avg5",
]

CURRENT_ROLE_FEATURES = [
    "self_roster_present", "self_roster_unavailable", "self_roster_active",
    "self_injury_reported", "self_inj_out", "self_inj_doubtful",
    "self_inj_questionable", "self_practice_dnp", "self_practice_limited",
    "new_roster_entry", "returned_active", "team_change_recent",
    "reappeared_after_gap",
    "team_rb_roster_count", "team_rb_available_count", "other_rb_available_count",
    "team_rb_injury_count", "team_rb_out_count", "team_rb_doubtful_count",
    "team_rb_questionable_count", "other_rb_out_count", "other_rb_doubtful_count",
    "other_rb_questionable_count",
    "prior_top1_unavailable", "prior_top2_unavailable",
    "vacated_lead_role", "vacated_top2_role",
    "depth_rank", "depth_is_rb1", "depth_promotion",
    "effective_available", "available_depth_ordinal", "best_available_depth",
    "better_depth_available_count", "depth_gap_from_best_available",
    "available_hist_share_ordinal", "best_available_hist_share",
    "available_hist_carry_ordinal", "best_available_hist_carries",
    "successor_depth_candidate", "successor_history_candidate",
]

COMPETITION_FEATURES = [
    "competitor_max_share_avg1", "competitor_max_share_avg3", "competitor_max_share_avg5",
    "competitor_max_carries_avg1", "competitor_max_carries_avg3", "competitor_max_carries_avg5",
    "competitor_max_targets_avg3", "competitor_max_targets_avg5",
    "competitor_sum_share_avg3", "competitor_sum_share_avg5",
    "strong_competitor_count", "competition_scarcity",
    "share_edge_vs_best_competitor", "carry_edge_vs_best_competitor",
    "depth_edge_vs_best_competitor",
]

INTERACTION_FEATURES = [
    "ix_vacancy_x_best_depth", "ix_vacancy_x_best_history",
    "ix_vacancy_x_prior_top2", "ix_depth1_x_hist_best",
    "ix_out_x_history", "ix_questionable_x_history",
    "ix_promotion_x_history", "ix_scarcity_x_history",
    "ix_new_entry_x_best_depth",
]

SPECS = {
    "history_only": HISTORY_FEATURES,
    "entitlement_basic": HISTORY_FEATURES + CURRENT_ROLE_FEATURES,
    "entitlement_competition": HISTORY_FEATURES + CURRENT_ROLE_FEATURES + COMPETITION_FEATURES,
    "entitlement_interactions": HISTORY_FEATURES + CURRENT_ROLE_FEATURES + COMPETITION_FEATURES + INTERACTION_FEATURES,
}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return hits[0]


def load_inputs(m95f_root: Path, m95b_root: Path):
    oof = lower(pd.read_csv(find_one(m95f_root, "m95f_2024_temporal_oof_scores.csv"), low_memory=False))
    hold = lower(pd.read_csv(find_one(m95f_root, "m95f_2024_holdout_trace.csv"), low_memory=False))
    val = lower(pd.read_csv(find_one(m95f_root, "m95f_2025_rb_trace.csv"), low_memory=False))
    trace = lower(pd.read_csv(find_one(m95b_root, "m95b_rb_matchup_trace.csv"), low_memory=False))
    for x in (oof, hold, val, trace):
        x["season"] = num(x["season"]).astype(int)
        x["week"] = num(x["week"]).astype(int)
        x["team"] = x["team"].map(g.canon)
        x["player_clean_key"] = x["player_clean_key"].astype(str)
    oof = oof.drop_duplicates(PLAYER_KEYS, keep="first").reset_index(drop=True)
    return oof, hold, val, trace


def attach_actual_carries(base: pd.DataFrame, trace: pd.DataFrame) -> pd.DataFrame:
    z = base.copy()
    carry_col = next((c for c in ["actual_carries", "actual_rush_att"] if c in z.columns), None)
    if carry_col is not None:
        z["actual_carries_m95h"] = num(z[carry_col])
    else:
        if "actual_carries" not in trace.columns:
            raise RuntimeError("M95H trace missing actual_carries")
        t = trace[PLAYER_KEYS + ["actual_carries"]].drop_duplicates(PLAYER_KEYS)
        z = z.merge(t, on=PLAYER_KEYS, how="left", validate="one_to_one")
        z["actual_carries_m95h"] = num(z["actual_carries"])
    return z


def add_entitlement_truth(base: pd.DataFrame, trace: pd.DataFrame) -> pd.DataFrame:
    z = attach_actual_carries(base, trace)
    z = z.loc[z["actual_carries_m95h"].notna()].copy()
    z["actual_carries_m95h"] = num(z["actual_carries_m95h"]).clip(lower=0)
    team = z.groupby(TEAM_KEYS, as_index=False).agg(
        team_rb_actual_carries=("actual_carries_m95h", "sum"),
        team_rb_candidate_count=("player_clean_key", "nunique"),
        team_rb_max_carries=("actual_carries_m95h", "max"),
    )
    z = z.merge(team, on=TEAM_KEYS, how="left", validate="many_to_one")
    z["actual_rb_share"] = np.where(
        num(z["team_rb_actual_carries"]).gt(0),
        num(z["actual_carries_m95h"]) / num(z["team_rb_actual_carries"]),
        np.nan,
    )
    z["actual_share60"] = z["actual_rb_share"].ge(0.60).astype(int)
    z["actual_share70"] = z["actual_rb_share"].ge(0.70).astype(int)
    lead = (
        z.sort_values(TEAM_KEYS + ["actual_carries_m95h", "player_clean_key"], ascending=[True, True, True, False, True])
        .drop_duplicates(TEAM_KEYS, keep="first")[TEAM_KEYS + ["player_clean_key"]]
        .rename(columns={"player_clean_key": "actual_lead_key"})
    )
    z = z.merge(lead, on=TEAM_KEYS, how="left", validate="many_to_one")
    z["actual_lead_rb"] = z["player_clean_key"].eq(z["actual_lead_key"]).astype(int)
    return z


def _team_rank(z: pd.DataFrame, value_col: str, ascending: bool, mask_col: str | None = None) -> pd.Series:
    vals = num(z[value_col])
    if mask_col is not None:
        vals = vals.where(num(z[mask_col]).eq(1))
    return vals.groupby([z[c] for c in TEAM_KEYS]).rank(method="min", ascending=ascending)


def add_recipient_features(base: pd.DataFrame) -> pd.DataFrame:
    z = base.copy()
    z["hist_share_rank"] = _team_rank(z, "rb_rb_share_avg3", ascending=False)
    z["hist_carry_rank"] = _team_rank(z, "rb_carries_avg3", ascending=False)
    z["hist_target_rank"] = _team_rank(z, "rb_targets_avg3", ascending=False)
    z["hist_best_share"] = z["hist_share_rank"].eq(1).astype(int)
    z["hist_best_carries"] = z["hist_carry_rank"].eq(1).astype(int)
    self_unavailable = (
        num(z["self_roster_unavailable"]).fillna(0).eq(1)
        | num(z["self_inj_out"]).fillna(0).eq(1)
        | num(z["self_inj_doubtful"]).fillna(0).eq(1)
    )
    z["effective_available"] = (~self_unavailable).astype(int)
    z["available_depth_ordinal"] = _team_rank(z, "depth_rank", ascending=True, mask_col="effective_available")
    z["best_available_depth"] = z["available_depth_ordinal"].eq(1).astype(int)
    z["available_hist_share_ordinal"] = _team_rank(z, "rb_rb_share_avg3", ascending=False, mask_col="effective_available")
    z["best_available_hist_share"] = z["available_hist_share_ordinal"].eq(1).astype(int)
    z["available_hist_carry_ordinal"] = _team_rank(z, "rb_carries_avg3", ascending=False, mask_col="effective_available")
    z["best_available_hist_carries"] = z["available_hist_carry_ordinal"].eq(1).astype(int)

    for c in [
        "better_depth_available_count", "strong_competitor_count",
    ]:
        z[c] = 0.0
    for c in [
        "depth_gap_from_best_available", "depth_edge_vs_best_competitor",
        "competitor_max_share_avg1", "competitor_max_share_avg3", "competitor_max_share_avg5",
        "competitor_max_carries_avg1", "competitor_max_carries_avg3", "competitor_max_carries_avg5",
        "competitor_max_targets_avg3", "competitor_max_targets_avg5",
        "competitor_sum_share_avg3", "competitor_sum_share_avg5",
    ]:
        z[c] = np.nan

    for _, idx in z.groupby(TEAM_KEYS, sort=False).groups.items():
        ids = list(idx)
        q = z.loc[ids]
        avail = q.loc[num(q["effective_available"]).eq(1)]
        best_depth = num(avail["depth_rank"]).min() if not avail.empty else np.nan
        for i in ids:
            row = z.loc[i]
            competitors = q.loc[q.index.ne(i) & num(q["effective_available"]).eq(1)]
            my_depth = num(pd.Series([row.get("depth_rank", np.nan)])).iloc[0]
            if pd.notna(my_depth):
                z.at[i, "better_depth_available_count"] = float((num(competitors["depth_rank"]) < my_depth).sum())
                if pd.notna(best_depth):
                    z.at[i, "depth_gap_from_best_available"] = float(my_depth - best_depth)
                comp_depth = num(competitors["depth_rank"]).min() if not competitors.empty else np.nan
                if pd.notna(comp_depth):
                    z.at[i, "depth_edge_vs_best_competitor"] = float(comp_depth - my_depth)
            specs = [
                ("rb_rb_share_avg1", "competitor_max_share_avg1", "max"),
                ("rb_rb_share_avg3", "competitor_max_share_avg3", "max"),
                ("rb_rb_share_avg5", "competitor_max_share_avg5", "max"),
                ("rb_carries_avg1", "competitor_max_carries_avg1", "max"),
                ("rb_carries_avg3", "competitor_max_carries_avg3", "max"),
                ("rb_carries_avg5", "competitor_max_carries_avg5", "max"),
                ("rb_targets_avg3", "competitor_max_targets_avg3", "max"),
                ("rb_targets_avg5", "competitor_max_targets_avg5", "max"),
                ("rb_rb_share_avg3", "competitor_sum_share_avg3", "sum"),
                ("rb_rb_share_avg5", "competitor_sum_share_avg5", "sum"),
            ]
            for src, dst, how in specs:
                s = num(competitors[src]).dropna() if src in competitors.columns else pd.Series(dtype=float)
                if not s.empty:
                    z.at[i, dst] = float(s.max() if how == "max" else s.sum())
            if not competitors.empty:
                strong = num(competitors["rb_rb_share_avg3"]).fillna(0).ge(0.30) | num(competitors["rb_carries_avg3"]).fillna(0).ge(7.0)
                z.at[i, "strong_competitor_count"] = float(strong.sum())

    z["share_edge_vs_best_competitor"] = num(z["rb_rb_share_avg3"]) - num(z["competitor_max_share_avg3"])
    z["carry_edge_vs_best_competitor"] = num(z["rb_carries_avg3"]) - num(z["competitor_max_carries_avg3"])
    z["competition_scarcity"] = 1.0 / (1.0 + num(z["other_rb_available_count"]).fillna(2.0))
    z["successor_depth_candidate"] = (
        num(z["prior_top1_unavailable"]).fillna(0).eq(1)
        & num(z["best_available_depth"]).eq(1)
        & num(z["effective_available"]).eq(1)
    ).astype(int)
    z["successor_history_candidate"] = (
        num(z["prior_top1_unavailable"]).fillna(0).eq(1)
        & num(z["best_available_hist_share"]).eq(1)
        & num(z["effective_available"]).eq(1)
    ).astype(int)
    hist = num(z["rb_rb_share_avg3"]).fillna(0)
    z["ix_vacancy_x_best_depth"] = num(z["prior_top1_unavailable"]).fillna(0) * num(z["best_available_depth"]).fillna(0)
    z["ix_vacancy_x_best_history"] = num(z["prior_top1_unavailable"]).fillna(0) * num(z["best_available_hist_share"]).fillna(0)
    z["ix_vacancy_x_prior_top2"] = num(z["prior_top1_unavailable"]).fillna(0) * num(z["target_was_prior_top2"]).fillna(0)
    z["ix_depth1_x_hist_best"] = num(z["best_available_depth"]).fillna(0) * num(z["hist_best_share"]).fillna(0)
    z["ix_out_x_history"] = num(z["self_inj_out"]).fillna(0) * hist
    z["ix_questionable_x_history"] = num(z["self_inj_questionable"]).fillna(0) * hist
    z["ix_promotion_x_history"] = num(z["depth_promotion"]).fillna(0) * hist
    z["ix_scarcity_x_history"] = num(z["competition_scarcity"]).fillna(0) * hist
    z["ix_new_entry_x_best_depth"] = num(z["new_roster_entry"]).fillna(0) * num(z["best_available_depth"]).fillna(0)
    return z


def enrich(base: pd.DataFrame, trace: pd.DataFrame, rosters: pd.DataFrame, injuries: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    z = g.enrich_base(base, trace, rosters, injuries, depth)
    extras = [
        "rb_targets_avg1", "rb_targets_avg3", "rb_targets_avg5",
        "team_top1_share_avg1", "team_top1_share_avg3", "team_top1_share_avg5",
    ]
    available = [c for c in extras if c in trace.columns]
    if available:
        t = trace[PLAYER_KEYS + available].drop_duplicates(PLAYER_KEYS).copy()
        t = t.rename(columns={c: f"_trace_{c}" for c in available})
        z = z.merge(t, on=PLAYER_KEYS, how="left", validate="one_to_one")
        for c in available:
            tc = f"_trace_{c}"
            if c in z.columns:
                z[c] = num(z[c]).combine_first(num(z[tc]))
            else:
                z[c] = num(z[tc])
            z = z.drop(columns=[tc])
    return add_recipient_features(z)


def pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=4000, random_state=SEED)),
    ])


def valid_features(df: pd.DataFrame, spec: str) -> list[str]:
    feats = []
    for c in SPECS[spec]:
        if c in df.columns and num(df[c]).notna().sum() >= 20 and num(df[c]).nunique(dropna=True) > 1:
            feats.append(c)
    if not feats:
        raise RuntimeError(f"M95H no usable features for {spec}")
    return feats


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, spec: str, c: float):
    feats = valid_features(train, spec)
    y = num(train[target]).astype(int)
    if y.nunique() < 2:
        raise RuntimeError(f"M95H one-class training target {target}")
    model = pipeline(c)
    model.fit(train[feats], y)
    p = np.clip(model.predict_proba(test[feats])[:, 1], 1e-6, 1 - 1e-6)
    return p, model, feats


def metrics(y, p) -> dict:
    yy = np.asarray(num(y), dtype=float)
    pp = np.asarray(p, dtype=float)
    mask = np.isfinite(yy) & np.isfinite(pp)
    yy = yy[mask].astype(int); pp = pp[mask]
    auc = float(roc_auc_score(yy, pp)) if len(np.unique(yy)) >= 2 else np.nan
    return {
        "n": int(len(yy)), "base_rate": float(np.mean(yy)) if len(yy) else np.nan,
        "mean_prob": float(np.mean(pp)) if len(pp) else np.nan, "auc": auc,
        "brier": float(np.mean((yy - pp) ** 2)) if len(yy) else np.nan,
        "logloss": float(log_loss(yy, pp, labels=[0, 1])) if len(yy) else np.nan,
    }


def top1_summary(df: pd.DataFrame, prob_col: str, slice_mask: pd.Series | None = None) -> dict:
    z = df.copy()
    if slice_mask is not None:
        z = z.loc[slice_mask.reindex(z.index, fill_value=False)].copy()
    z = z.loc[num(z["team_rb_candidate_count"]).ge(2)].copy()
    if z.empty:
        return {"team_games": 0, "top1_accuracy": np.nan, "mean_true_lead_prob": np.nan, "mean_true_lead_rank": np.nan}
    rows = []
    for _, q in z.groupby(TEAM_KEYS, sort=False):
        q = q.copy()
        q["_p"] = num(q[prob_col]).fillna(-1)
        q = q.sort_values(["_p", "player_clean_key"], ascending=[False, True])
        pred = str(q.iloc[0]["player_clean_key"])
        actual = str(q.iloc[0]["actual_lead_key"])
        tq = q.loc[q["player_clean_key"].eq(actual)]
        true_prob = float(num(tq[prob_col]).iloc[0]) if not tq.empty else np.nan
        q["_rank"] = q["_p"].rank(method="min", ascending=False)
        true_rank = float(q.loc[q["player_clean_key"].eq(actual), "_rank"].iloc[0]) if not tq.empty else np.nan
        rows.append((pred == actual, true_prob, true_rank))
    arr = pd.DataFrame(rows, columns=["correct", "true_prob", "true_rank"])
    return {
        "team_games": int(len(arr)), "top1_accuracy": float(arr["correct"].mean()),
        "mean_true_lead_prob": float(arr["true_prob"].mean()), "mean_true_lead_rank": float(arr["true_rank"].mean()),
    }


def team_slice_mask(df: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "vacancy":
        return num(df["prior_top1_unavailable"]).fillna(0).eq(1)
    if kind == "incumbent_available":
        return num(df["prior_top1_unavailable"]).fillna(0).eq(0) & df["prior_top1_key"].notna()
    if kind == "late_week":
        return num(df["week"]).ge(17)
    return pd.Series(True, index=df.index)


def evaluate_model_frame(df: pd.DataFrame, prob_col: str, target: str, model_name: str, scope: str) -> list[dict]:
    out = [{"scope": scope, "target": target, "model": model_name, "slice": "all", **metrics(df[target], df[prob_col])}]
    for s in ["vacancy", "incumbent_available", "late_week"]:
        q = df.loc[team_slice_mask(df, s)].copy()
        if len(q) >= 10 and num(q[target]).nunique() >= 2:
            out.append({"scope": scope, "target": target, "model": model_name, "slice": s, **metrics(q[target], q[prob_col])})
    return out


def candidate_selection(train: pd.DataFrame, hold: pd.DataFrame, target: str):
    base_p, _, base_feats = fit_predict(train, hold, target, "history_only", 0.08)
    hb = hold.copy(); hb["_base"] = base_p
    base_m = metrics(hb[target], hb["_base"])
    base_top = top1_summary(hb, "_base") if target == "actual_lead_rb" else None
    base_vac = top1_summary(hb, "_base", team_slice_mask(hb, "vacancy")) if target == "actual_lead_rb" else None
    rows = []
    for spec in ["entitlement_basic", "entitlement_competition", "entitlement_interactions"]:
        for c in C_GRID:
            try:
                p, _, feats = fit_predict(train, hold, target, spec, c)
            except Exception as exc:
                rows.append({"target": target, "spec": spec, "C": c, "eligible": 0, "error": f"{type(exc).__name__}:{exc}"})
                continue
            hh = hold.copy(); hh["_p"] = p
            m = metrics(hh[target], hh["_p"])
            row = {
                "target": target, "spec": spec, "C": c, "feature_count": len(feats), **m,
                "baseline_auc": base_m["auc"], "baseline_brier": base_m["brier"], "baseline_logloss": base_m["logloss"],
            }
            if target == "actual_lead_rb":
                tt = top1_summary(hh, "_p")
                tv = top1_summary(hh, "_p", team_slice_mask(hh, "vacancy"))
                row.update({
                    "top1_accuracy": tt["top1_accuracy"], "vacancy_top1_accuracy": tv["top1_accuracy"],
                    "baseline_top1_accuracy": base_top["top1_accuracy"], "baseline_vacancy_top1_accuracy": base_vac["top1_accuracy"],
                })
                row["eligible"] = int(
                    np.isfinite(m["brier"]) and m["brier"] <= base_m["brier"] + 0.002
                    and np.isfinite(tt["top1_accuracy"]) and tt["top1_accuracy"] >= base_top["top1_accuracy"] - 0.01
                )
                row["selection_score"] = (
                    2.0 * (tv["top1_accuracy"] - base_vac["top1_accuracy"] if np.isfinite(tv["top1_accuracy"]) and np.isfinite(base_vac["top1_accuracy"]) else -1)
                    + (tt["top1_accuracy"] - base_top["top1_accuracy"] if np.isfinite(tt["top1_accuracy"]) else -1)
                    + 2.0 * (base_m["brier"] - m["brier"])
                )
            else:
                row["eligible"] = int(
                    np.isfinite(m["brier"]) and m["brier"] <= base_m["brier"] + 0.001
                    and (not np.isfinite(base_m["auc"]) or not np.isfinite(m["auc"]) or m["auc"] >= base_m["auc"] - 0.005)
                )
                row["selection_score"] = 3.0 * (base_m["brier"] - m["brier"]) + (m["auc"] - base_m["auc"] if np.isfinite(m["auc"]) and np.isfinite(base_m["auc"]) else 0)
            rows.append(row)
    grid = pd.DataFrame(rows)
    eligible = grid.loc[num(grid["eligible"]).eq(1)].copy()
    if eligible.empty:
        eligible = grid.loc[grid["selection_score"].notna()].copy()
    if eligible.empty:
        raise RuntimeError(f"M95H no valid candidates for {target}")
    chosen = eligible.sort_values(["selection_score", "brier"], ascending=[False, True]).iloc[0].to_dict()
    chosen["baseline_feature_count"] = len(base_feats)
    return chosen, grid


def probability_bins(df: pd.DataFrame, target: str, prob_col: str, label: str) -> pd.DataFrame:
    z = df[[target, prob_col]].copy().dropna()
    if z.empty:
        return pd.DataFrame()
    try:
        z["bin"] = pd.qcut(num(z[prob_col]), q=10, duplicates="drop")
    except Exception:
        return pd.DataFrame()
    out = z.groupby("bin", observed=True).agg(n=(target, "size"), actual_rate=(target, "mean"), mean_prob=(prob_col, "mean")).reset_index()
    out["bin"] = out["bin"].astype(str); out["target"] = target; out["model"] = label
    return out


def successor_examples(val: pd.DataFrame) -> pd.DataFrame:
    z = val.loc[num(val["prior_top1_unavailable"]).fillna(0).eq(1) & num(val["team_rb_candidate_count"]).ge(2)].copy()
    rows = []
    for keys, q in z.groupby(TEAM_KEYS, sort=False):
        actual = str(q.iloc[0]["actual_lead_key"])
        base = q.sort_values(["p_lead_history", "player_clean_key"], ascending=[False, True]).iloc[0]
        cand = q.sort_values(["p_lead_m95h", "player_clean_key"], ascending=[False, True]).iloc[0]
        aq = q.loc[q["player_clean_key"].eq(actual)].iloc[0]
        rows.append({
            "season": keys[0], "week": keys[1], "team": keys[2], "prior_top1_key": aq.get("prior_top1_key", ""),
            "actual_lead_key": actual, "actual_lead_carries": aq.get("actual_carries_m95h", np.nan),
            "actual_lead_share": aq.get("actual_rb_share", np.nan), "history_pick": base["player_clean_key"],
            "m95h_pick": cand["player_clean_key"], "history_correct": int(str(base["player_clean_key"]) == actual),
            "m95h_correct": int(str(cand["player_clean_key"]) == actual), "actual_lead_p_history": aq.get("p_lead_history", np.nan),
            "actual_lead_p_m95h": aq.get("p_lead_m95h", np.nan), "actual_lead_depth_rank": aq.get("depth_rank", np.nan),
            "actual_lead_best_available_depth": aq.get("best_available_depth", np.nan),
            "actual_lead_prior_top2": aq.get("target_was_prior_top2", np.nan),
            "actual_lead_hist_share_avg3": aq.get("rb_rb_share_avg3", np.nan),
        })
    return pd.DataFrame(rows).sort_values(["week", "team"]) if rows else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m95f-root", type=Path, required=True)
    ap.add_argument("--m95b-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    oof, hold, val, trace = load_inputs(args.m95f_root, args.m95b_root)
    rosters, injuries, depth, source_audit = g.load_provider_sources([2024, 2025])
    rosters = g.add_roster_transition_features(rosters)
    depth = g.add_depth_transition_features(depth)
    oof = add_entitlement_truth(oof, trace); hold = add_entitlement_truth(hold, trace); val = add_entitlement_truth(val, trace)
    oof_e = enrich(oof, trace, rosters, injuries, depth)
    hold_e = enrich(hold, trace, rosters, injuries, depth)
    val_e = enrich(val, trace, rosters, injuries, depth)
    train = oof_e.loc[num(oof_e["week"]).between(5, 12)].copy()
    hold_dev = hold_e.loc[num(hold_e["week"]).between(13, 18)].copy()
    train_all = pd.concat([train, hold_dev], ignore_index=True, sort=False).drop_duplicates(PLAYER_KEYS, keep="last")
    if train.empty or hold_dev.empty or train_all.empty or val_e.empty:
        raise RuntimeError("M95H empty temporal split")

    selected_rows, grids, hold_metric_rows, val_metric_rows, feature_rows = [], [], [], [], []
    for target in TARGETS:
        chosen, grid = candidate_selection(train, hold_dev, target)
        grids.append(grid)
        selected_rows.append({
            "target": target, "spec": chosen["spec"], "C": chosen["C"], "holdout_auc": chosen.get("auc", np.nan),
            "holdout_brier": chosen.get("brier", np.nan), "holdout_logloss": chosen.get("logloss", np.nan),
            "holdout_top1_accuracy": chosen.get("top1_accuracy", np.nan),
            "holdout_vacancy_top1_accuracy": chosen.get("vacancy_top1_accuracy", np.nan),
            "selection_score": chosen.get("selection_score", np.nan), "development_eligible": chosen.get("eligible", 0),
        })
        pbh, _, _ = fit_predict(train, hold_dev, target, "history_only", 0.08)
        pch, _, _ = fit_predict(train, hold_dev, target, str(chosen["spec"]), float(chosen["C"]))
        h = hold_dev.copy(); h["_base"] = pbh; h["_cand"] = pch
        hold_metric_rows.extend(evaluate_model_frame(h, "_base", target, "history", "2024_w13_18_holdout"))
        hold_metric_rows.extend(evaluate_model_frame(h, "_cand", target, "m95h", "2024_w13_18_holdout"))
        pb, _, _ = fit_predict(train_all, val_e, target, "history_only", 0.08)
        pc, _, feats = fit_predict(train_all, val_e, target, str(chosen["spec"]), float(chosen["C"]))
        base_col = {"actual_lead_rb": "p_lead_history", "actual_share60": "p_share60_history", "actual_share70": "p_share70_history"}[target]
        cand_col = {"actual_lead_rb": "p_lead_m95h", "actual_share60": "p_share60_m95h", "actual_share70": "p_share70_m95h"}[target]
        val_e[base_col] = pb; val_e[cand_col] = pc
        val_metric_rows.extend(evaluate_model_frame(val_e, base_col, target, "history", "2025_untouched_validation"))
        val_metric_rows.extend(evaluate_model_frame(val_e, cand_col, target, "m95h", "2025_untouched_validation"))
        feature_rows.extend({"target": target, "spec": chosen["spec"], "feature": f} for f in feats)

    selected = pd.DataFrame(selected_rows); grid_all = pd.concat(grids, ignore_index=True, sort=False)
    hold_metrics = pd.DataFrame(hold_metric_rows); val_metrics = pd.DataFrame(val_metric_rows)
    team_rows = []
    for sl in ["all", "vacancy", "incumbent_available", "late_week"]:
        mask = None if sl == "all" else team_slice_mask(val_e, sl)
        for model_name, col in [("history", "p_lead_history"), ("m95h", "p_lead_m95h")]:
            team_rows.append({"scope": "2025_untouched_validation", "slice": sl, "model": model_name, **top1_summary(val_e, col, mask)})
    team_audit = pd.DataFrame(team_rows)

    def mr(target, model):
        q = val_metrics.loc[(val_metrics["target"] == target) & (val_metrics["model"] == model) & (val_metrics["slice"] == "all")]
        return q.iloc[0] if not q.empty else pd.Series(dtype=float)
    def ta(sl, model):
        q = team_audit.loc[(team_audit["slice"] == sl) & (team_audit["model"] == model)]
        return float(q.iloc[0]["top1_accuracy"]) if not q.empty else np.nan

    lead_hist, lead_new = mr("actual_lead_rb", "history"), mr("actual_lead_rb", "m95h")
    s60_hist, s60_new = mr("actual_share60", "history"), mr("actual_share60", "m95h")
    s70_hist, s70_new = mr("actual_share70", "history"), mr("actual_share70", "m95h")
    overall_hist, overall_new = ta("all", "history"), ta("all", "m95h")
    vac_hist, vac_new = ta("vacancy", "history"), ta("vacancy", "m95h")
    inc_hist, inc_new = ta("incumbent_available", "history"), ta("incumbent_available", "m95h")
    lead_pass = int(
        np.isfinite(overall_hist) and np.isfinite(overall_new) and overall_new >= overall_hist + 0.02
        and np.isfinite(vac_hist) and np.isfinite(vac_new) and vac_new >= vac_hist + 0.05
        and float(lead_new.get("brier", np.inf)) <= float(lead_hist.get("brier", -np.inf))
    )
    share60_pass = int(
        float(s60_new.get("brier", np.inf)) < float(s60_hist.get("brier", -np.inf))
        and (not np.isfinite(float(s60_hist.get("auc", np.nan))) or float(s60_new.get("auc", -np.inf)) >= float(s60_hist.get("auc", np.nan)) - 0.005)
    )
    share70_pass = int(
        float(s70_new.get("brier", np.inf)) < float(s70_hist.get("brier", -np.inf))
        and (not np.isfinite(float(s70_hist.get("auc", np.nan))) or float(s70_new.get("auc", -np.inf)) >= float(s70_hist.get("auc", np.nan)) - 0.005)
    )
    incumbent_guard = int((not np.isfinite(inc_hist)) or (not np.isfinite(inc_new)) or inc_new >= inc_hist - 0.02)
    validation_pass = int(lead_pass and share60_pass and share70_pass and incumbent_guard)
    disposition = "ADVANCE_M95H_ENTITLEMENT_SIGNAL_TO_INTEGRATION_NOT_PRODUCTION" if validation_pass else "RETAIN_M95H_AS_DIAGNOSTIC_DO_NOT_PROMOTE"
    disposition_df = pd.DataFrame([{
        "selected_lead": selected.loc[selected.target.eq("actual_lead_rb"), "spec"].iloc[0],
        "selected_share60": selected.loc[selected.target.eq("actual_share60"), "spec"].iloc[0],
        "selected_share70": selected.loc[selected.target.eq("actual_share70"), "spec"].iloc[0],
        "2025_lead_top1_history": overall_hist, "2025_lead_top1_m95h": overall_new,
        "2025_vacancy_top1_history": vac_hist, "2025_vacancy_top1_m95h": vac_new,
        "2025_incumbent_top1_history": inc_hist, "2025_incumbent_top1_m95h": inc_new,
        "2025_lead_brier_history": lead_hist.get("brier", np.nan), "2025_lead_brier_m95h": lead_new.get("brier", np.nan),
        "2025_share60_brier_history": s60_hist.get("brier", np.nan), "2025_share60_brier_m95h": s60_new.get("brier", np.nan),
        "2025_share70_brier_history": s70_hist.get("brier", np.nan), "2025_share70_brier_m95h": s70_new.get("brier", np.nan),
        "lead_pass": lead_pass, "share60_pass": share60_pass, "share70_pass": share70_pass,
        "incumbent_guard": incumbent_guard, "validation_pass": validation_pass,
        "m94c_central_mean_preserved": 1, "sportsbook_inputs": 0, "production_change": 0, "disposition": disposition,
    }])

    bins = []
    for target, bcol, ccol in [
        ("actual_lead_rb", "p_lead_history", "p_lead_m95h"),
        ("actual_share60", "p_share60_history", "p_share60_m95h"),
        ("actual_share70", "p_share70_history", "p_share70_m95h"),
    ]:
        bins.extend([probability_bins(val_e, target, bcol, "history"), probability_bins(val_e, target, ccol, "m95h")])
    bins_df = pd.concat([x for x in bins if not x.empty], ignore_index=True, sort=False) if any(not x.empty for x in bins) else pd.DataFrame()
    examples = successor_examples(val_e)

    source_audit.to_csv(args.out_dir / "m95h_source_audit.csv", index=False)
    selected.to_csv(args.out_dir / "m95h_selected_architecture.csv", index=False)
    grid_all.to_csv(args.out_dir / "m95h_2024_candidate_grid.csv", index=False)
    hold_metrics.to_csv(args.out_dir / "m95h_2024_holdout_metrics.csv", index=False)
    val_metrics.to_csv(args.out_dir / "m95h_2025_metrics.csv", index=False)
    team_audit.to_csv(args.out_dir / "m95h_team_recipient_audit.csv", index=False)
    bins_df.to_csv(args.out_dir / "m95h_probability_bins.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(args.out_dir / "m95h_feature_audit.csv", index=False)
    examples.to_csv(args.out_dir / "m95h_2025_successor_examples.csv", index=False)
    disposition_df.to_csv(args.out_dir / "m95h_disposition.csv", index=False)
    keep = PLAYER_KEYS + [
        "actual_carries_m95h", "team_rb_actual_carries", "team_rb_candidate_count", "actual_rb_share",
        "actual_lead_key", "actual_lead_rb", "actual_share60", "actual_share70",
        "prior_top1_key", "prior_top1_unavailable", "prior_top2_key", "prior_top2_unavailable",
        "target_was_prior_top1", "target_was_prior_top2", "depth_rank", "depth_is_rb1", "depth_promotion",
        "effective_available", "available_depth_ordinal", "best_available_depth",
        "available_hist_share_ordinal", "best_available_hist_share",
        "rb_carries_avg1", "rb_carries_avg3", "rb_carries_avg5",
        "rb_rb_share_avg1", "rb_rb_share_avg3", "rb_rb_share_avg5",
        "rb_targets_avg1", "rb_targets_avg3", "rb_targets_avg5",
        "p_lead_history", "p_lead_m95h", "p_share60_history", "p_share60_m95h", "p_share70_history", "p_share70_m95h",
    ]
    val_e[[c for c in keep if c in val_e.columns]].to_csv(args.out_dir / "m95h_2025_trace.csv", index=False)

    print("[m95h] disposition"); print(disposition_df.to_string(index=False))
    print("\n[m95h] selected architectures"); print(selected.to_string(index=False))
    print("\n[m95h] team recipient audit"); print(team_audit.to_string(index=False))
    print("\n[m95h] 2025 target metrics"); print(val_metrics.loc[val_metrics["slice"].eq("all")].to_string(index=False))
    print("\n[m95h] source audit"); print(source_audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
