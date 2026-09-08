#!/usr/bin/env python3
"""RB-R16 diagnostic: can strict-prior state/identity predict upside receiving tails?

R13-R15 found that simple pregame YPT mean signals are too weak. R16 therefore
changes the question instead of tuning the failed mean family: keep the frozen
baseline receiving mean untouched and test whether pregame opportunity/state/
identity information predicts the *probability* of an upside tail.

Primary label: baseline underprojection by >=30 receiving yards.
Secondary labels: >=50-yard underprojection, 40+/60+ actual receiving yards, and an
actual >=20-yard reception from PBP. Current-game outcomes/PBP are labels only.

This is diagnostic-only. No target entitlement, team RB pool, production mean,
Monte Carlo parameter, sportsbook input, or production parameter is changed.
2023-2025 are research-visible; folds are 2023 -> 2024 and 2023-24 -> 2025.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts._opponent_map import canon_team
from scripts.backtest.audit_rb_receiving_identity_v1 import _load_logs
from scripts.backtest.diagnose_rb_r14_pbp_efficiency_role_v1 import _pbp_rb_games

FEATURES = [
    "baseline_pred_targets",
    "baseline_pred_rec_yards",
    "state_probability",
    "prior_rb_room_share",
    "r9_raw_r8_residual",
    "frozen_ypt",
    "identity_top20",
]
MEAN_ONLY = ["baseline_pred_rec_yards"]
TARGET_ONLY = ["baseline_pred_targets"]
FOLDS = [((2023,), 2024), ((2023, 2024), 2025)]

# Frozen before executing R16.
MIN_PRIMARY_COMBINED_AUC = 0.65
MIN_PRIMARY_FOLD_AUC = 0.58
MIN_PRIMARY_FOLDS_AT_THRESHOLD = 2
MIN_PRIMARY_TOP_QUINTILE_LIFT = 1.50
MIN_PRIMARY_TOP_QUINTILE_CAPTURE = 0.30
MIN_PRIMARY_BRIER_GAIN_VS_POOLED = 0.0
MIN_PBP_EXACT_TARGET_RATE = 0.90


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _auc(y: pd.Series, score: pd.Series) -> float:
    y = _num(y); score = _num(score)
    ok = y.notna() & score.notna()
    y = y.loc[ok].astype(int); score = score.loc[ok]
    n1 = int(y.sum()); n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = score.rank(method="average")
    return float((ranks.loc[y.eq(1)].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _brier(y, p) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def _logloss(y, p) -> float:
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], label: str) -> np.ndarray:
    if train[label].nunique() < 2:
        raise RuntimeError(f"R16 training has one class for {label}")
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=3000, random_state=916),
    )
    m.fit(train[cols], train[label].astype(int))
    return m.predict_proba(test[cols])[:, 1]


def _attach_explosive_labels(x: pd.DataFrame, pbp_start: int) -> tuple[pd.DataFrame, dict]:
    seasons = sorted(set(int(s) for s in _num(x.season).dropna().astype(int)))
    logs = _load_logs(list(range(int(pbp_start), max(seasons) + 1)))
    games, lineage = _pbp_rb_games(logs, list(range(int(pbp_start), max(seasons) + 1)))
    games = games.copy()
    games["team"] = games.team.map(canon_team)
    g = games.loc[games.season.isin(seasons), ["season","week","team","player_clean_key","targets","explosive_targets"]].copy()
    g = g.groupby(["season","week","team","player_clean_key"], as_index=False).agg(
        pbp_targets=("targets","sum"), explosive_receptions=("explosive_targets","sum")
    )
    z = x.merge(g, on=["season","week","team","player_clean_key"], how="left", validate="one_to_one")
    z["pbp_targets"] = _num(z.pbp_targets).fillna(0.0)
    z["explosive_receptions"] = _num(z.explosive_receptions).fillna(0.0)
    z["pbp_target_exact"] = np.isclose(z.pbp_targets, z.actual_targets, atol=1e-9)
    exact_positive = z.loc[z.actual_targets.gt(0), "pbp_target_exact"]
    audit = {
        "query_rows": int(len(z)),
        "positive_target_rows": int(z.actual_targets.gt(0).sum()),
        "exact_pbp_target_rows_positive_targets": int(exact_positive.sum()),
        "exact_pbp_target_rate_positive_targets": float(exact_positive.mean()) if len(exact_positive) else np.nan,
        "pbp_lineage": lineage,
    }
    z["explosive20"] = np.where(z.pbp_target_exact, z.explosive_receptions.ge(1).astype(float), np.nan)
    return z, audit


def _metric_row(label: str, variant: str, y: pd.Series, p: pd.Series) -> dict:
    y = _num(y); p = _num(p)
    ok = y.notna() & p.notna(); y = y.loc[ok].astype(int); p = p.loc[ok]
    rate = float(y.mean())
    pct = p.rank(pct=True, method="average")
    top = pct.gt(.80)
    top_rate = float(y.loc[top].mean()) if top.any() else np.nan
    return {
        "label": label, "variant": variant, "n": int(len(y)), "events": int(y.sum()), "event_rate": rate,
        "auc": _auc(y, p), "brier": _brier(y, p), "log_loss": _logloss(y, p),
        "top_quintile_n": int(top.sum()), "top_quintile_event_rate": top_rate,
        "top_quintile_lift": float(top_rate / rate) if pd.notna(top_rate) and rate > 0 else np.nan,
        "top_quintile_capture": float(y.loc[top].sum() / y.sum()) if y.sum() else np.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--pbp-start", type=int, default=2018)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    x = pd.read_csv(a.predictions, low_memory=False)
    req = {
        "season","week","team","player_clean_key","actual_targets","actual_rec_yards",
        "baseline_pred_targets","baseline_pred_rec_yards","state_probability","prior_rb_room_share",
        "r9_raw_r8_residual","frozen_ypt","identity_bucket",
    }
    missing = sorted(req - set(x.columns))
    if missing:
        raise RuntimeError(f"R16 predictions missing columns: {missing}")
    if x.duplicated(["season","week","team","player_clean_key"]).any():
        raise RuntimeError("R16 requires one row per RB player-game")

    for c in ["actual_targets","actual_rec_yards","baseline_pred_targets","baseline_pred_rec_yards","state_probability","prior_rb_room_share","r9_raw_r8_residual","frozen_ypt"]:
        x[c] = _num(x[c])
    x["identity_top20"] = x.identity_bucket.eq("TOP20").astype(float)
    x["baseline_signed_under"] = x.actual_rec_yards - x.baseline_pred_rec_yards
    x["cat30_under"] = x.baseline_signed_under.ge(30).astype(float)
    x["cat50_under"] = x.baseline_signed_under.ge(50).astype(float)
    x["rec40"] = x.actual_rec_yards.ge(40).astype(float)
    x["rec60"] = x.actual_rec_yards.ge(60).astype(float)
    x, pbp_audit = _attach_explosive_labels(x, a.pbp_start)

    x = x.dropna(subset=FEATURES + ["cat30_under","cat50_under","rec40","rec60"]).reset_index(drop=True)
    labels = ["cat30_under","cat50_under","rec40","rec60","explosive20"]
    pred_parts = []
    fold_rows = []
    for train_seasons, test_season in FOLDS:
        tr0 = x.loc[x.season.isin(train_seasons)].copy()
        te0 = x.loc[x.season.eq(test_season)].copy()
        for label in labels:
            tr = tr0.loc[tr0[label].notna()].copy()
            te = te0.loc[te0[label].notna()].copy()
            if len(tr) < 100 or len(te) < 100 or tr[label].nunique() < 2 or te[label].nunique() < 2:
                continue
            p_full = _fit_predict(tr, te, FEATURES, label)
            p_mean = _fit_predict(tr, te, MEAN_ONLY, label)
            p_target = _fit_predict(tr, te, TARGET_ONLY, label)
            p_pool = np.full(len(te), float(tr[label].mean()))
            out = te[["season","week","team","player_clean_key","identity_bucket",label]].copy()
            out["label"] = label
            out["p_full"] = p_full; out["p_mean_only"] = p_mean; out["p_target_only"] = p_target; out["p_pooled"] = p_pool
            pred_parts.append(out)
            for variant, p in [("FULL_STATE_IDENTITY",p_full),("MEAN_ONLY",p_mean),("TARGET_ONLY",p_target),("POOLED_BASE",p_pool)]:
                r = _metric_row(label, variant, te[label], pd.Series(p, index=te.index))
                r.update({"train_seasons": ",".join(map(str,train_seasons)), "test_season": int(test_season), "train_event_rate": float(tr[label].mean())})
                fold_rows.append(r)

    if not pred_parts:
        raise RuntimeError("R16 produced no evaluable folds")
    preds = pd.concat(pred_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)

    combined_rows = []
    for label, g in preds.groupby("label"):
        y = _num(g[label])
        for variant, pc in [("FULL_STATE_IDENTITY","p_full"),("MEAN_ONLY","p_mean_only"),("TARGET_ONLY","p_target_only"),("POOLED_BASE","p_pooled")]:
            combined_rows.append(_metric_row(label, variant, y, _num(g[pc])))
    combined = pd.DataFrame(combined_rows)

    def one(label, variant):
        q = combined.loc[combined.label.eq(label) & combined.variant.eq(variant)]
        if len(q) != 1:
            raise RuntimeError(f"R16 combined lookup failed {label=} {variant=}")
        return q.iloc[0]

    primary = one("cat30_under","FULL_STATE_IDENTITY")
    pool = one("cat30_under","POOLED_BASE")
    mean = one("cat30_under","MEAN_ONLY")
    ff = folds.loc[folds.label.eq("cat30_under") & folds.variant.eq("FULL_STATE_IDENTITY")]
    folds_at = int((_num(ff.auc) >= MIN_PRIMARY_FOLD_AUC).sum())
    brier_gain_pool = float(pool.brier - primary.brier)
    auc_gain_mean = float(primary.auc - mean.auc)
    gates = {
        "primary_combined_auc": bool(float(primary.auc) >= MIN_PRIMARY_COMBINED_AUC),
        "primary_fold_auc_consistency": bool(folds_at >= MIN_PRIMARY_FOLDS_AT_THRESHOLD),
        "primary_top_quintile_lift": bool(float(primary.top_quintile_lift) >= MIN_PRIMARY_TOP_QUINTILE_LIFT),
        "primary_top_quintile_capture": bool(float(primary.top_quintile_capture) >= MIN_PRIMARY_TOP_QUINTILE_CAPTURE),
        "primary_brier_better_than_pooled": bool(brier_gain_pool > MIN_PRIMARY_BRIER_GAIN_VS_POOLED),
        "pbp_label_integrity": bool(pbp_audit["exact_pbp_target_rate_positive_targets"] >= MIN_PBP_EXACT_TARGET_RATE),
        "sportsbook_zero": True,
    }
    tail_supported = all(gates.values())

    secondary = {}
    for label in ["cat50_under","rec40","rec60","explosive20"]:
        q = one(label,"FULL_STATE_IDENTITY")
        secondary[label] = {
            "n": int(q.n), "events": int(q.events), "event_rate": float(q.event_rate),
            "auc": float(q.auc), "brier": float(q.brier), "top_quintile_lift": float(q.top_quintile_lift),
            "top_quintile_capture": float(q.top_quintile_capture),
        }

    result = {
        "diagnostic": "RB_R16_UPSIDE_TAIL_STATE_V1",
        "disposition": "RB_R16_UPSIDE_TAIL_SIGNAL_SUPPORTED_DIAGNOSTIC_ONLY" if tail_supported else "RB_R16_UPSIDE_TAIL_SIGNAL_NOT_SUPPORTED_DIAGNOSTIC_ONLY",
        "upside_tail_signal_supported": bool(tail_supported),
        "mean_policy": "FROZEN_BASELINE_RECEIVING_MEAN_UNCHANGED",
        "primary_label": "baseline actual_minus_prediction >= 30 receiving yards",
        "features": FEATURES,
        "primary": {
            "n": int(primary.n), "events": int(primary.events), "event_rate": float(primary.event_rate),
            "combined_auc": float(primary.auc), "mean_only_auc": float(mean.auc), "auc_gain_vs_mean_only": auc_gain_mean,
            "combined_brier": float(primary.brier), "pooled_brier": float(pool.brier), "brier_gain_vs_pooled": brier_gain_pool,
            "top_quintile_lift": float(primary.top_quintile_lift), "top_quintile_capture": float(primary.top_quintile_capture),
            "folds_auc_at_least_threshold": folds_at,
        },
        "secondary": secondary,
        "gates": gates,
        "thresholds": {
            "min_primary_combined_auc": MIN_PRIMARY_COMBINED_AUC,
            "min_primary_fold_auc": MIN_PRIMARY_FOLD_AUC,
            "min_primary_folds_at_threshold": MIN_PRIMARY_FOLDS_AT_THRESHOLD,
            "min_primary_top_quintile_lift": MIN_PRIMARY_TOP_QUINTILE_LIFT,
            "min_primary_top_quintile_capture": MIN_PRIMARY_TOP_QUINTILE_CAPTURE,
            "min_primary_brier_gain_vs_pooled": MIN_PRIMARY_BRIER_GAIN_VS_POOLED,
            "min_pbp_exact_target_rate": MIN_PBP_EXACT_TARGET_RATE,
        },
        "pbp_label_audit": pbp_audit,
        "sportsbook_inputs_added": 0,
        "production_parameters_changed": 0,
        "governance_note": "2023-2025 are research-visible. Support authorizes only a separately frozen distribution/MC candidate; it does not promote an RB receiving mean or R12.",
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(a.out_dir / "rb_r16_tail_predictions.csv", index=False)
    folds.to_csv(a.out_dir / "rb_r16_fold_summary.csv", index=False)
    combined.to_csv(a.out_dir / "rb_r16_combined_summary.csv", index=False)
    (a.out_dir / "rb_r16_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("\n=== combined ===\n", combined.to_string(index=False))
    print("\n=== folds ===\n", folds.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
