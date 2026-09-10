#!/usr/bin/env python3
"""Aggregate R27 folds and apply the 27 frozen structural/scientific gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "baseline"
CAND = "candidate"
EXPECTED_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing R27 evidence: {path}")
    return pd.read_csv(path, low_memory=False)


def _metric(actual: pd.Series, pred: pd.Series, market: str) -> dict[str, float | int]:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(float)
    p = pd.to_numeric(pred, errors="coerce").to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    if not len(a):
        return {
            "n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan,
            "median_abs_error": np.nan, "p75_abs_error": np.nan,
            "p90_abs_error": np.nan, "pearson": np.nan, "spearman": np.nan,
            "large_error_30_rate": np.nan,
        }
    err = p - a
    ae = np.abs(err)
    pearson = float(np.corrcoef(a, p)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(p) > 0 else np.nan
    spearman = float(pd.Series(a).corr(pd.Series(p), method="spearman")) if len(a) > 1 else np.nan
    return {
        "n": int(len(a)), "mae": float(np.mean(ae)), "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)), "median_abs_error": float(np.quantile(ae, .50)),
        "p75_abs_error": float(np.quantile(ae, .75)), "p90_abs_error": float(np.quantile(ae, .90)),
        "pearson": pearson, "spearman": spearman,
        "large_error_30_rate": float(np.mean(ae >= 30.0)) if market == "rec_yards" else np.nan,
    }


def _cohorts(pred: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "ALL": pd.Series(True, index=pred.index),
        "VACANCY_ACTIVE": pred["vacancy_active"].eq(1),
        "VACANCY_INCUMBENT": pred["vacancy_incumbent"].eq(1),
        "VACANCY_RB1_INCUMBENT": pred["vacancy_incumbent"].eq(1) & pred["role"].eq("RB1"),
        "VACANCY_RB2PLUS_INCUMBENT": pred["vacancy_incumbent"].eq(1) & pred["role"].eq("RB2+"),
        "VACANCY_NEW_VETERAN": pred["vacancy_new_veteran"].eq(1),
        "VACANCY_NO_PRIOR_NFL": pred["vacancy_no_prior_nfl"].eq(1),
        "WEEK1": pred["week"].eq(1),
        "WEEKS2PLUS": pred["week"].ge(2),
    }


def _all_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, mask in _cohorts(pred).items():
        g = pred.loc[mask].copy()
        for variant in (BASE, CAND):
            spec = {
                "targets": ("actual_targets", f"{variant}_targets"),
                "receptions": ("actual_receptions", f"{variant}_receptions"),
                "rec_yards": ("actual_rec_yards", "r27_baseline_rec_yards" if variant == BASE else "r27_candidate_rec_yards"),
            }
            for market, (actual_col, pred_col) in spec.items():
                rec = {"cohort": cohort, "variant": variant, "market": market}
                rec.update(_metric(g[actual_col], g[pred_col], market))
                rows.append(rec)
    return pd.DataFrame(rows)


def _season_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in EXPECTED_SEASONS:
        g0 = pred.loc[pd.to_numeric(pred["season"], errors="coerce").eq(season)].copy()
        for cohort, mask in _cohorts(g0).items():
            g = g0.loc[mask].copy()
            for variant in (BASE, CAND):
                spec = {
                    "targets": ("actual_targets", f"{variant}_targets"),
                    "receptions": ("actual_receptions", f"{variant}_receptions"),
                    "rec_yards": ("actual_rec_yards", "r27_baseline_rec_yards" if variant == BASE else "r27_candidate_rec_yards"),
                }
                for market, (actual_col, pred_col) in spec.items():
                    rec = {"season": season, "cohort": cohort, "variant": variant, "market": market}
                    rec.update(_metric(g[actual_col], g[pred_col], market))
                    rows.append(rec)
    return pd.DataFrame(rows)


def _row(metrics: pd.DataFrame, cohort: str, variant: str, market: str) -> pd.Series:
    x = metrics.loc[
        metrics["cohort"].eq(cohort) & metrics["variant"].eq(variant) & metrics["market"].eq(market)
    ]
    if len(x) != 1:
        raise RuntimeError(f"R27 expected one metric row: {cohort} {variant} {market}, got {len(x)}")
    return x.iloc[0]


def _pct(candidate: float, baseline: float) -> float:
    if not np.isfinite(candidate) or not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return np.nan
    return 100.0 * (candidate / baseline - 1.0)


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--protected-clean-marker", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if not args.protected_clean_marker.exists() or args.protected_clean_marker.read_text(encoding="utf-8").strip() != "PASS":
        raise RuntimeError("R27 protected production boundary marker missing/not PASS")

    pred_parts = []
    audit_parts = []
    metas = []
    provenance_rows = []
    for d in args.dirs:
        p = d / "r27_player_predictions.csv"
        a = d / "r27_structural_audit.csv"
        m = d / "r27_fold_metadata.json"
        pred_parts.append(_read(p))
        audit_parts.append(_read(a))
        metas.append(json.loads(m.read_text(encoding="utf-8")))
        for f in sorted(d.glob("*")):
            if f.is_file():
                provenance_rows.append({"path": str(f), "sha256": _hash(f), "bytes": int(f.stat().st_size)})

    pred = pd.concat(pred_parts, ignore_index=True)
    audit = pd.concat(audit_parts, ignore_index=True)
    seasons = sorted(pd.to_numeric(pred["season"], errors="coerce").dropna().astype(int).unique().tolist())
    metrics = _all_metrics(pred)
    season_metrics = _season_metrics(pred)

    # Core metric aliases.
    va_b = _row(metrics, "VACANCY_ACTIVE", BASE, "rec_yards")
    va_c = _row(metrics, "VACANCY_ACTIVE", CAND, "rec_yards")
    vi_b = _row(metrics, "VACANCY_INCUMBENT", BASE, "rec_yards")
    vi_c = _row(metrics, "VACANCY_INCUMBENT", CAND, "rec_yards")
    rb1_b = _row(metrics, "VACANCY_RB1_INCUMBENT", BASE, "rec_yards")
    rb1_c = _row(metrics, "VACANCY_RB1_INCUMBENT", CAND, "rec_yards")
    rb2_b = _row(metrics, "VACANCY_RB2PLUS_INCUMBENT", BASE, "rec_yards")
    rb2_c = _row(metrics, "VACANCY_RB2PLUS_INCUMBENT", CAND, "rec_yards")
    all_b = _row(metrics, "ALL", BASE, "rec_yards")
    all_c = _row(metrics, "ALL", CAND, "rec_yards")
    w1_b = _row(metrics, "WEEK1", BASE, "rec_yards")
    w1_c = _row(metrics, "WEEK1", CAND, "rec_yards")
    vat_b = _row(metrics, "VACANCY_ACTIVE", BASE, "targets")
    vat_c = _row(metrics, "VACANCY_ACTIVE", CAND, "targets")
    var_b = _row(metrics, "VACANCY_ACTIVE", BASE, "receptions")
    var_c = _row(metrics, "VACANCY_ACTIVE", CAND, "receptions")

    season_va = season_metrics.loc[
        season_metrics["cohort"].eq("VACANCY_ACTIVE") & season_metrics["market"].eq("rec_yards")
    ].copy()
    pivot = season_va.pivot(index="season", columns="variant", values="mae")
    if any(s not in pivot.index for s in EXPECTED_SEASONS) or BASE not in pivot.columns or CAND not in pivot.columns:
        raise RuntimeError("R27 incomplete season VACANCY_ACTIVE metric matrix")
    season_changes = {int(s): _pct(float(pivot.loc[s, CAND]), float(pivot.loc[s, BASE])) for s in EXPECTED_SEASONS}
    season_improved = sum(np.isfinite(v) and v < 0.0 for v in season_changes.values())
    max_season_worsen = max([v for v in season_changes.values() if np.isfinite(v)], default=np.inf)

    max_room_gap = float(pd.to_numeric(audit["max_rb_room_mass_gap"], errors="coerce").abs().max())
    max_nonrb = float(pd.to_numeric(audit["max_non_rb_entitlement_delta"], errors="coerce").abs().max())
    max_nonvac = float(pd.to_numeric(audit["max_nonvacancy_target_delta"], errors="coerce").abs().max())
    max_bridge = float(pd.to_numeric(audit["max_reception_bridge_gap"], errors="coerce").abs().max())
    max_base_parity = float(pd.to_numeric(audit["max_r26_baseline_mean_parity_gap"], errors="coerce").abs().max())
    strict_prior_all = bool(pd.Series([m.get("strict_prior_fit") for m in metas]).eq(True).all())
    sports_max = int(pd.to_numeric(audit["sportsbook_inputs_upstream"], errors="coerce").fillna(1).max())
    future_max = int(pd.to_numeric(audit["future_outcomes_used_in_features"], errors="coerce").fillna(1).max())
    vacancy_gate_all = bool(audit["vacancy_gate_exact_room_exits_ge_1"].astype(bool).all())
    no_actual_eff = bool((~audit["actual_efficiency_used_in_prediction"].astype(bool)).all())
    no_r22_mean = bool((~audit["r22_used_as_upstream_mean_correction"].astype(bool)).all())
    no_prod_change = bool((~audit["production_files_changed"].astype(bool)).all())

    gates: dict[str, bool] = {
        "01_sportsbook_inputs_upstream_zero": sports_max == 0,
        "02_future_outcomes_features_zero": future_max == 0,
        "03_exact_r26_strict_prior_contract": strict_prior_all,
        "04_vacancy_gate_room_exits_ge_1_unchanged": vacancy_gate_all,
        "05_rb_room_target_mass_conservation": max_room_gap < 1e-10,
        "06_non_rb_entitlement_exact": max_nonrb < 1e-12,
        "07_protected_production_unchanged": no_prod_change,
        "08_r22_not_used_as_mean_correction": no_r22_mean,
        "09_nonvacancy_opportunity_exact_baseline": max_nonvac < 1e-12,
        "10_reception_path_identity_exact": max_bridge < 1e-10 and max_base_parity < 1e-10,
        "11_actual_efficiency_not_used_in_prediction": no_actual_eff,
        "12_all_2020_2025_folds_complete": seasons == EXPECTED_SEASONS,
        "13_vacancy_active_rec_yard_mae_improves": float(va_c.mae) < float(va_b.mae),
        "14_vacancy_incumbent_rec_yard_mae_improves": float(vi_c.mae) < float(vi_b.mae),
        "15_vacancy_active_rmse_nonworse": float(va_c.rmse) <= float(va_b.rmse),
        "16_vacancy_active_abs_bias_nonworse": abs(float(va_c.bias)) <= abs(float(va_b.bias)),
        "17_vacancy_active_p90_worsens_no_more_than_2pct": _pct(float(va_c.p90_abs_error), float(va_b.p90_abs_error)) <= 2.0,
        "18_at_least_4_of_6_seasons_vacancy_mae_improve": season_improved >= 4,
        "19_no_season_vacancy_mae_worsens_over_3pct": max_season_worsen <= 3.0,
        "20_rb1_and_rb2plus_each_worsen_no_more_than_1_5pct": max(_pct(float(rb1_c.mae), float(rb1_b.mae)), _pct(float(rb2_c.mae), float(rb2_b.mae))) <= 1.5,
        "21_at_least_one_vacancy_role_cohort_improves": min(_pct(float(rb1_c.mae), float(rb1_b.mae)), _pct(float(rb2_c.mae), float(rb2_b.mae))) < 0.0,
        "22_all_rb_mae_worsens_no_more_than_0_25pct": _pct(float(all_c.mae), float(all_b.mae)) <= 0.25,
        "23_all_rb_rmse_worsens_no_more_than_0_25pct": _pct(float(all_c.rmse), float(all_b.rmse)) <= 0.25,
        "24_week1_mae_worsens_no_more_than_0_50pct": _pct(float(w1_c.mae), float(w1_b.mae)) <= 0.50,
        "25_vacancy_target_mae_improves_or_within_0_10pct": _pct(float(vat_c.mae), float(vat_b.mae)) <= 0.10,
        "26_vacancy_reception_mae_improves_or_within_0_10pct": _pct(float(var_c.mae), float(var_b.mae)) <= 0.10,
        "27_vacancy_rec_yard_mae_improves_at_least_0_50pct": _pct(float(va_c.mae), float(va_b.mae)) <= -0.50,
    }

    first_26 = all(gates[f"{i:02d}_" + next(k.split("_", 1)[1] for k in gates if k.startswith(f"{i:02d}_"))] for i in range(1, 27))
    # Simpler defensive equivalent to the expression above; assert no numbering gaps.
    numbered = {int(k[:2]): v for k, v in gates.items()}
    if sorted(numbered) != list(range(1, 28)):
        raise RuntimeError(f"R27 gate numbering invalid: {sorted(numbered)}")
    first_26 = all(numbered[i] for i in range(1, 27))
    if first_26 and numbered[27]:
        disposition = "R27_R26_OPPORTUNITY_REC_YARD_MEAN_SUPPORT_READY_FOR_INTEGRATION_DESIGN"
    elif first_26:
        disposition = "R27_R26_OPPORTUNITY_SAFE_BUT_EFFICIENCY_WORK_REQUIRED"
    else:
        disposition = "R27_R26_OPPORTUNITY_TRANSLATION_MIXED_OR_FAIL"

    comparison = {
        "vacancy_active": {
            "n": int(va_b.n),
            "baseline_rec_yard_mae": float(va_b.mae), "candidate_rec_yard_mae": float(va_c.mae),
            "mae_pct_change": _pct(float(va_c.mae), float(va_b.mae)),
            "baseline_rmse": float(va_b.rmse), "candidate_rmse": float(va_c.rmse),
            "rmse_pct_change": _pct(float(va_c.rmse), float(va_b.rmse)),
            "baseline_bias": float(va_b.bias), "candidate_bias": float(va_c.bias),
            "baseline_p90": float(va_b.p90_abs_error), "candidate_p90": float(va_c.p90_abs_error),
            "p90_pct_change": _pct(float(va_c.p90_abs_error), float(va_b.p90_abs_error)),
            "target_mae_pct_change": _pct(float(vat_c.mae), float(vat_b.mae)),
            "reception_mae_pct_change": _pct(float(var_c.mae), float(var_b.mae)),
        },
        "vacancy_incumbent_mae_pct_change": _pct(float(vi_c.mae), float(vi_b.mae)),
        "vacancy_rb1_mae_pct_change": _pct(float(rb1_c.mae), float(rb1_b.mae)),
        "vacancy_rb2plus_mae_pct_change": _pct(float(rb2_c.mae), float(rb2_b.mae)),
        "all_rb_mae_pct_change": _pct(float(all_c.mae), float(all_b.mae)),
        "all_rb_rmse_pct_change": _pct(float(all_c.rmse), float(all_b.rmse)),
        "week1_mae_pct_change": _pct(float(w1_c.mae), float(w1_b.mae)),
        "season_vacancy_active_mae_pct_change": {str(k): float(v) for k, v in season_changes.items()},
        "season_vacancy_active_improved_count": int(season_improved),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_dir / "r27_player_predictions.csv", index=False)
    metrics.to_csv(args.out_dir / "r27_metrics_by_cohort.csv", index=False)
    season_metrics.to_csv(args.out_dir / "r27_metrics_by_season.csv", index=False)
    audit.to_csv(args.out_dir / "r27_structural_audit.csv", index=False)
    pd.DataFrame([{"gate": k, "pass": bool(v)} for k, v in gates.items()]).to_csv(args.out_dir / "r27_gate_matrix.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(args.out_dir / "r27_provenance_hash_manifest.csv", index=False)

    disposition_doc = {
        "study": "RB_R27_RECEIVING_YARD_MEAN_DECOMPOSITION_V1",
        "disposition": disposition,
        "gate_count": 27,
        "gate_pass_count": int(sum(bool(v) for v in gates.values())),
        "all_gates_pass": bool(all(gates.values())),
        "first_26_support_safety_gates_pass": bool(first_26),
        "material_gate_27_pass": bool(numbered[27]),
        "seasons": EXPECTED_SEASONS,
        "candidate": "EXACT_R26_OPPORTUNITY_X_EXISTING_PRODUCTION_YPT",
        "new_efficiency_model_fit": False,
        "r22_changed": False,
        "production_changed": False,
        "sportsbook_inputs_upstream": sports_max,
        "future_outcomes_used_in_features": future_max,
        "comparison": comparison,
        "structural_maxima": {
            "max_rb_room_mass_gap": max_room_gap,
            "max_non_rb_entitlement_delta": max_nonrb,
            "max_nonvacancy_target_delta": max_nonvac,
            "max_reception_bridge_gap": max_bridge,
            "max_r26_baseline_mean_parity_gap": max_base_parity,
        },
        "gates": gates,
    }
    (args.out_dir / "r27_disposition.json").write_text(json.dumps(disposition_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(disposition_doc, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
