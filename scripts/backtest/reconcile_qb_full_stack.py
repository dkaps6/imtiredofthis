#!/usr/bin/env python3
"""M82: reconcile current full-stack QB predictions on canonical-v3 identities.

Diagnostic only. This script never fits football features. It combines already-OOS
MC/ML/State walk-forward predictions, builds an explicitly OOS ensemble, and
scores every model on the exact canonical-v3 football-only QB cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.ensemble_v2 import apply_ensemble, fit_market_weights

EXPECTED_CANONICAL_SHA256 = "c4a481a760657bb7516d52b9aa9ba84af096063a4e78660b12a541f59fd7b742"
EXPECTED_SEASON_ROWS = {2024: 444, 2025: 440}
EXPECTED_ROWS = 884
MIN_ENSEMBLE_ROWS = 40
MODEL_COLS = {
    "canonical_v3": "pred_pass_yards",
    "current_mc": "mc_proj",
    "current_ml": "ml_proj",
    "current_state": "state_proj",
    "oos_ensemble": "ensemble_proj",
}
MARKET_TOKENS = ("sportsbook", "prop_line", "moneyline", "game_total", "implied_total", "vegas")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lower(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def load_canonical(path: Path) -> pd.DataFrame:
    digest = _sha256(path)
    if digest != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(f"canonical-v3 SHA drift: {digest}")
    x = _lower(pd.read_csv(path, low_memory=False))
    if len(x) != EXPECTED_ROWS:
        raise RuntimeError(f"canonical-v3 row drift: {len(x)}")
    counts = {int(k): int(v) for k, v in pd.to_numeric(x["season"], errors="coerce").value_counts().to_dict().items()}
    if counts != EXPECTED_SEASON_ROWS:
        raise RuntimeError(f"canonical-v3 season row drift: {counts}")
    required = {"season", "week", "team", "player_clean_key", "actual_pass_yards", "pred_pass_yards"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"canonical-v3 missing columns: {sorted(missing)}")
    bad = [c for c in x.columns if any(tok in c for tok in MARKET_TOKENS)]
    if bad:
        raise RuntimeError(f"canonical-v3 market boundary violation: {bad}")
    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
    x["team"] = x["team"].astype(str).str.upper().str.strip()
    x["player_clean_key"] = x["player_clean_key"].astype(str).str.strip()
    return x


def load_components(path: Path, season: int) -> pd.DataFrame:
    x = _lower(pd.read_csv(path, low_memory=False))
    required = {"season", "week", "team", "player_clean_key", "market", "actual", "mc_proj", "ml_proj", "state_proj"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"component predictions missing columns {sorted(missing)}: {path}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].eq(int(season))].copy()
    if x.empty:
        raise RuntimeError(f"component predictions contain no {season} rows: {path}")
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x["team"] = x["team"].astype(str).str.upper().str.strip()
    x["player_clean_key"] = x["player_clean_key"].astype(str).str.strip()
    return x


def build_oos_ensemble(p24: pd.DataFrame, p25: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create true OOS ensemble predictions under the frozen M82 contract."""
    q24 = p24.loc[p24["market"].astype(str).str.lower().eq("pass_yards")].copy()
    q25 = p25.loc[p25["market"].astype(str).str.lower().eq("pass_yards")].copy()

    out24 = []
    weight_rows = []
    for week in sorted(q24["week"].dropna().astype(int).unique()):
        target = q24.loc[q24["week"].eq(week)].copy()
        history = q24.loc[q24["week"].lt(week)].copy()
        usable = history.dropna(subset=["actual", "mc_proj", "ml_proj", "state_proj"])
        if len(usable) >= MIN_ENSEMBLE_ROWS:
            weights = fit_market_weights(usable, min_rows=MIN_ENSEMBLE_ROWS)
            applied = apply_ensemble(target, weights=weights)
            if not weights.empty:
                wr = weights.loc[weights["market"].astype(str).str.lower().eq("pass_yards")].iloc[0].to_dict()
                wr.update({"target_season": 2024, "target_week": int(week), "fit_scope": "earlier_2024_oos"})
                weight_rows.append(wr)
        else:
            applied = apply_ensemble(target, weights=pd.DataFrame())
            weight_rows.append({
                "market": "pass_yards", "mc_weight": 1.0, "ml_weight": 0.0, "state_weight": 0.0,
                "calibration_rows": int(len(usable)), "method": "mc_fallback_insufficient_prior_oos",
                "target_season": 2024, "target_week": int(week), "fit_scope": "earlier_2024_oos",
            })
        out24.append(applied)
    e24 = pd.concat(out24, ignore_index=True) if out24 else pd.DataFrame()

    usable24 = q24.dropna(subset=["actual", "mc_proj", "ml_proj", "state_proj"])
    if len(usable24) < MIN_ENSEMBLE_ROWS:
        raise RuntimeError(f"insufficient complete 2024 pass-yard rows for frozen 2025 ensemble: {len(usable24)}")
    weights25 = fit_market_weights(usable24, min_rows=MIN_ENSEMBLE_ROWS)
    if weights25.empty:
        raise RuntimeError("2024 OOS rows failed to produce 2025 ensemble weights")
    e25 = apply_ensemble(q25, weights=weights25)
    wr = weights25.loc[weights25["market"].astype(str).str.lower().eq("pass_yards")].iloc[0].to_dict()
    wr.update({"target_season": 2025, "target_week": 0, "fit_scope": "all_2024_oos_frozen_for_2025"})
    weight_rows.append(wr)

    return pd.concat([e24, e25], ignore_index=True), pd.DataFrame(weight_rows)


def reconcile(canonical: pd.DataFrame, ensemble_rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "week", "team", "player_clean_key"]
    q = ensemble_rows.copy()
    if q.duplicated(keys).any():
        dup = q.loc[q.duplicated(keys, keep=False), keys].head(10).to_dict(orient="records")
        raise RuntimeError(f"duplicate pass-yard component identities: {dup}")
    keep = keys + ["actual", "mc_proj", "ml_proj", "state_proj", "ensemble_proj", "ensemble_status",
                   "ensemble_weight_mc", "ensemble_weight_ml", "ensemble_weight_state"]
    q = q[keep]
    out = canonical.merge(q, on=keys, how="left", validate="one_to_one", indicator=True)
    missing = out["_merge"].ne("both")
    if missing.any():
        sample = out.loc[missing, keys].head(15).to_dict(orient="records")
        raise RuntimeError(f"M82 canonical common-cohort identity mismatch: missing={int(missing.sum())} sample={sample}")
    out.drop(columns=["_merge"], inplace=True)
    actual_component = pd.to_numeric(out["actual"], errors="coerce")
    actual_canonical = pd.to_numeric(out["actual_pass_yards"], errors="coerce")
    mismatch = (actual_component - actual_canonical).abs() > 1e-9
    if mismatch.any():
        raise RuntimeError(f"M82 actual-yard mismatch on {int(mismatch.sum())} canonical rows")
    return out


def metric_row(frame: pd.DataFrame, model: str, col: str, season_label: str) -> dict:
    actual = pd.to_numeric(frame["actual_pass_yards"], errors="coerce")
    pred = pd.to_numeric(frame[col], errors="coerce")
    mask = actual.notna() & pred.notna() & np.isfinite(actual) & np.isfinite(pred)
    a = actual.loc[mask].to_numpy(float)
    p = pred.loc[mask].to_numpy(float)
    n_total = len(frame)
    n = len(a)
    if n == 0:
        return {"season": season_label, "model": model, "n": 0, "coverage": 0.0}
    err = p - a
    corr = float(np.corrcoef(p, a)[0, 1]) if n >= 2 and np.std(p) > 0 and np.std(a) > 0 else np.nan
    return {
        "season": season_label, "model": model, "n": n, "coverage": n / n_total,
        "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)), "correlation": corr, "tails100": int((np.abs(err) >= 100).sum()),
    }


def build_scoreboard(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("2024", x.loc[x["season"].eq(2024)]), ("2025", x.loc[x["season"].eq(2025)]), ("COMBINED", x)]
    for label, part in scopes:
        for model, col in MODEL_COLS.items():
            rows.append(metric_row(part, model, col, label))
    return pd.DataFrame(rows)


def pairwise_correlations(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_rows, resid_rows = [], []
    actual = pd.to_numeric(x["actual_pass_yards"], errors="coerce")
    for (m1, c1), (m2, c2) in combinations(MODEL_COLS.items(), 2):
        p1 = pd.to_numeric(x[c1], errors="coerce")
        p2 = pd.to_numeric(x[c2], errors="coerce")
        mask = actual.notna() & p1.notna() & p2.notna()
        n = int(mask.sum())
        if n < 2:
            pc = rc = np.nan
        else:
            pc = float(p1.loc[mask].corr(p2.loc[mask]))
            r1 = p1.loc[mask] - actual.loc[mask]
            r2 = p2.loc[mask] - actual.loc[mask]
            rc = float(r1.corr(r2))
        pred_rows.append({"model_1": m1, "model_2": m2, "n": n, "prediction_correlation": pc})
        resid_rows.append({"model_1": m1, "model_2": m2, "n": n, "residual_correlation": rc})
    return pd.DataFrame(pred_rows), pd.DataFrame(resid_rows)


def oracle_summary(x: pd.DataFrame, scoreboard: pd.DataFrame) -> pd.DataFrame:
    actual = pd.to_numeric(x["actual_pass_yards"], errors="coerce").to_numpy(float)
    cols = [c for c in MODEL_COLS.values()]
    preds = np.column_stack([pd.to_numeric(x[c], errors="coerce").to_numpy(float) for c in cols])
    abs_err = np.abs(preds - actual[:, None])
    abs_err[~np.isfinite(abs_err)] = np.inf
    best_err = np.min(abs_err, axis=1)
    valid = np.isfinite(best_err)
    oracle_mae = float(best_err[valid].mean()) if valid.any() else np.nan
    combined = scoreboard.loc[scoreboard["season"].eq("COMBINED") & scoreboard["mae"].notna()].sort_values("mae")
    best_model = str(combined.iloc[0]["model"]) if not combined.empty else ""
    best_mae = float(combined.iloc[0]["mae"]) if not combined.empty else np.nan
    return pd.DataFrame([{
        "oracle_scope": "canonical+current_full_stack_hindsight_best_per_game",
        "n": int(valid.sum()), "oracle_mae": oracle_mae,
        "best_single_model": best_model, "best_single_mae": best_mae,
        "hindsight_headroom_yards": float(best_mae - oracle_mae) if np.isfinite(best_mae) and np.isfinite(oracle_mae) else np.nan,
        "deployable": False,
    }])


def disagreement_table(x: pd.DataFrame) -> pd.DataFrame:
    comp = x[["mc_proj", "ml_proj", "state_proj"]].apply(pd.to_numeric, errors="coerce")
    x = x.copy()
    x["component_disagreement_sd"] = comp.std(axis=1, skipna=True)
    x["canonical_abs_error"] = (pd.to_numeric(x["pred_pass_yards"], errors="coerce") - pd.to_numeric(x["actual_pass_yards"], errors="coerce")).abs()
    x["ensemble_abs_error"] = (pd.to_numeric(x["ensemble_proj"], errors="coerce") - pd.to_numeric(x["actual_pass_yards"], errors="coerce")).abs()
    valid = x["component_disagreement_sd"].notna()
    if valid.sum() >= 4:
        x.loc[valid, "disagreement_bucket"] = pd.qcut(x.loc[valid, "component_disagreement_sd"], q=4, labels=["Q1_LOW", "Q2", "Q3", "Q4_HIGH"], duplicates="drop").astype(str)
    else:
        x["disagreement_bucket"] = "UNAVAILABLE"
    return x.groupby("disagreement_bucket", dropna=False).agg(
        n=("actual_pass_yards", "size"),
        mean_component_disagreement=("component_disagreement_sd", "mean"),
        canonical_mae=("canonical_abs_error", "mean"),
        ensemble_mae=("ensemble_abs_error", "mean"),
    ).reset_index()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--predictions-2024", type=Path, required=True)
    p.add_argument("--predictions-2025", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    canonical = load_canonical(args.canonical)
    p24 = load_components(args.predictions_2024, 2024)
    p25 = load_components(args.predictions_2025, 2025)
    ensemble, weights = build_oos_ensemble(p24, p25)
    common = reconcile(canonical, ensemble)

    scoreboard = build_scoreboard(common)
    pred_corr, resid_corr = pairwise_correlations(common)
    oracle = oracle_summary(common, scoreboard)
    disagreement = disagreement_table(common)

    # Sanity-check the canonical metric against the frozen M75/M80 baseline.
    frozen = scoreboard.loc[(scoreboard["season"].eq("COMBINED")) & (scoreboard["model"].eq("canonical_v3"))].iloc[0]
    if abs(float(frozen["mae"]) - 58.505044) > 1e-5:
        raise RuntimeError(f"canonical-v3 MAE drift inside M82: {float(frozen['mae'])}")

    common.to_csv(args.out_dir / "m82_qb_common_cohort_trace.csv", index=False)
    scoreboard.to_csv(args.out_dir / "m82_qb_full_stack_scoreboard.csv", index=False)
    weights.to_csv(args.out_dir / "m82_oos_ensemble_weights.csv", index=False)
    pred_corr.to_csv(args.out_dir / "m82_prediction_correlations.csv", index=False)
    resid_corr.to_csv(args.out_dir / "m82_residual_correlations.csv", index=False)
    oracle.to_csv(args.out_dir / "m82_hindsight_library_oracle.csv", index=False)
    disagreement.to_csv(args.out_dir / "m82_model_disagreement_buckets.csv", index=False)

    combined = scoreboard.loc[scoreboard["season"].eq("COMBINED") & scoreboard["mae"].notna()].copy()
    eligible = combined.loc[combined["coverage"].ge(0.95)].sort_values(["mae", "rmse", "correlation"], ascending=[True, True, False])
    best = eligible.iloc[0].to_dict() if not eligible.empty else {}
    component_resid = resid_corr.loc[resid_corr["model_1"].isin(["current_mc", "current_ml", "current_state"]) & resid_corr["model_2"].isin(["current_mc", "current_ml", "current_state"]), "residual_correlation"].abs().dropna()
    contract = {
        "migration": "M82",
        "production_actionable": False,
        "canonical_sha256": EXPECTED_CANONICAL_SHA256,
        "canonical_rows": len(common),
        "season_rows": {str(k): int(v) for k, v in common["season"].value_counts().sort_index().to_dict().items()},
        "sportsbook_features_used": False,
        "mc_iterations": 2000,
        "ensemble_2024": "expanding earlier-week OOS; MC fallback until >=40 complete rows",
        "ensemble_2025": "frozen weights fit on 2024 OOS only",
        "lowest_combined_mae_model_coverage_ge_95pct": best,
        "median_abs_residual_corr_mc_ml_state": float(component_resid.median()) if not component_resid.empty else None,
        "hindsight_oracle": oracle.iloc[0].to_dict(),
    }
    (args.out_dir / "m82_reconciliation_contract.json").write_text(json.dumps(contract, indent=2, default=str) + "\n")

    print("[m82_scoreboard]")
    print(scoreboard.to_string(index=False))
    print("[m82_component_residual_correlations]")
    print(resid_corr.to_string(index=False))
    print("[m82_oracle]")
    print(oracle.to_string(index=False))
    print("[m82_contract]", json.dumps(contract, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
