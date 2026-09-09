#!/usr/bin/env python3
"""Qualify the frozen R26 Week-1 component from immutable parent predictions.

No refit. No regenerated predictions. No sportsbook input. No production writes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing required artifact: {path}")
    return pd.read_csv(path, low_memory=False)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def metric(actual: pd.Series, pred: pd.Series) -> dict:
    a = num(actual)
    p = num(pred)
    m = a.notna() & p.notna() & np.isfinite(a) & np.isfinite(p)
    a = a.loc[m].to_numpy(float)
    p = p.loc[m].to_numpy(float)
    if len(a) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "p90_abs_error": np.nan}
    e = p - a
    ae = np.abs(e)
    return {
        "n": int(len(a)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(np.mean(e)),
        "p90_abs_error": float(np.quantile(ae, 0.90)),
    }


def rel_worsen(base: float, cand: float) -> float:
    if not np.isfinite(base) or not np.isfinite(cand):
        return np.inf
    if abs(base) < 1e-12:
        return 0.0 if abs(cand) < 1e-12 else np.inf
    return float(cand / base - 1.0)


def improves(base: float, cand: float) -> bool:
    return bool(np.isfinite(base) and np.isfinite(cand) and cand < base)


def m(g: pd.DataFrame, variant: str, market: str) -> dict:
    return metric(g[f"actual_{market}"], g[f"{variant}_{market}"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--protected-clean-marker", type=Path, required=True)
    a = ap.parse_args()

    if not a.protected_clean_marker.exists() or a.protected_clean_marker.read_text().strip() != "PASS":
        raise RuntimeError("R26E protected-production clean marker missing")

    preds = []
    audits = []
    fits = []
    for season in SEASONS:
        d = a.parent_root / "data" / "backtests" / f"r26_vacancy_gated_r9_{season}"
        p = read(d / "r26_predictions.csv")
        p["expected_test_season"] = season
        preds.append(p)
        audits.append(read(d / "r26_structural_audit.csv"))
        fits.append(json.loads((d / "r26_fit_metadata.json").read_text()))

    pred = pd.concat(preds, ignore_index=True)
    audit = pd.concat(audits, ignore_index=True)
    pred["season"] = num(pred.season).astype(int)
    pred["week"] = num(pred.week).astype(int)
    pred["vacancy_active"] = num(pred.vacancy_active).fillna(0).astype(int)
    pred["continuing_same_team"] = num(pred.continuing_same_team).fillna(0).astype(int)
    pred["rb_rank"] = num(pred.rb_rank).fillna(99).astype(int)

    w1 = pred.loc[pred.week.eq(1)].copy()
    inc = w1.loc[w1.vacancy_active.eq(1) & w1.continuing_same_team.eq(1)].copy()
    rb1 = inc.loc[inc.rb_rank.eq(1)].copy()
    rb2 = inc.loc[inc.rb_rank.ge(2)].copy()
    if w1.empty or inc.empty:
        raise RuntimeError("R26E Week-1 qualification population is empty")
    if not w1.week.eq(1).all() or not inc.week.eq(1).all():
        raise RuntimeError("non-Week-1 rows entered R26E scoring")

    b_inc_rec, c_inc_rec = m(inc, "baseline", "receptions"), m(inc, "candidate", "receptions")
    b_inc_tgt, c_inc_tgt = m(inc, "baseline", "targets"), m(inc, "candidate", "targets")
    b_all_rec, c_all_rec = m(w1, "baseline", "receptions"), m(w1, "candidate", "receptions")
    b_rb1, c_rb1 = m(rb1, "baseline", "receptions"), m(rb1, "candidate", "receptions")
    b_rb2, c_rb2 = m(rb2, "baseline", "receptions"), m(rb2, "candidate", "receptions")

    season_rows = []
    season_improve_count = 0
    season_max_worsen = -np.inf
    for season in SEASONS:
        g = inc.loc[inc.season.eq(season)].copy()
        b = m(g, "baseline", "receptions")
        c = m(g, "candidate", "receptions")
        rw = rel_worsen(b["mae"], c["mae"])
        if improves(b["mae"], c["mae"]):
            season_improve_count += 1
        season_max_worsen = max(season_max_worsen, rw)
        season_rows.append({
            "season": season,
            "n": int(b["n"]),
            "baseline_receptions_mae": float(b["mae"]),
            "candidate_receptions_mae": float(c["mae"]),
            "relative_mae_change": float(rw),
        })

    sportsbook = int(num(pred.get("sportsbook_inputs_used", pd.Series(0, index=pred.index))).fillna(0).sum())
    future_pred = int(num(pred.get("future_outcomes_used", pd.Series(0, index=pred.index))).fillna(0).sum())
    future_audit = int(num(audit.get("future_outcomes_used", pd.Series(0, index=audit.index))).fillna(0).sum())
    max_ry = float(num(audit.max_receiving_yard_mean_delta).abs().max())
    max_r22 = float(num(audit.r22_authority_delta).abs().max())
    strict_prior = bool(all(bool(x.get("strict_prior_fit")) for x in fits))

    gates = {
        "01_parent_artifact_digest_verified_by_workflow": True,
        "02_week1_rows_only": bool(w1.week.eq(1).all() and inc.week.eq(1).all()),
        "03_sportsbook_inputs_zero": sportsbook == 0,
        "04_future_outcome_features_zero": (future_pred + future_audit) == 0 and strict_prior,
        "05_receiving_yard_mean_exact": max_ry == 0.0,
        "06_r22_authority_exact": max_r22 == 0.0,
        "07_protected_production_files_clean": True,
        "08_pooled_w1_vacancy_incumbent_receptions_mae_improves": improves(b_inc_rec["mae"], c_inc_rec["mae"]),
        "09_pooled_w1_vacancy_incumbent_receptions_rmse_nonworse": c_inc_rec["rmse"] <= b_inc_rec["rmse"],
        "10_pooled_w1_vacancy_incumbent_abs_bias_nonworse": abs(c_inc_rec["bias"]) <= abs(b_inc_rec["bias"]),
        "11_pooled_w1_vacancy_incumbent_p90_worsen_le_2pct": rel_worsen(b_inc_rec["p90_abs_error"], c_inc_rec["p90_abs_error"]) <= 0.02,
        "12_pooled_w1_vacancy_incumbent_target_mae_improves": improves(b_inc_tgt["mae"], c_inc_tgt["mae"]),
        "13_at_least_4_of_6_w1_seasons_improve": season_improve_count >= 4,
        "14_no_w1_season_worsens_more_than_5pct": season_max_worsen <= 0.05,
        "15_w1_vacancy_rb1_mae_worsen_le_1pct": rel_worsen(b_rb1["mae"], c_rb1["mae"]) <= 0.01,
        "16_w1_vacancy_rb2plus_mae_worsen_le_1pct": rel_worsen(b_rb2["mae"], c_rb2["mae"]) <= 0.01,
        "17_at_least_one_w1_role_improves": improves(b_rb1["mae"], c_rb1["mae"]) or improves(b_rb2["mae"], c_rb2["mae"]),
        "18_all_rb_w1_receptions_mae_improves": improves(b_all_rec["mae"], c_all_rec["mae"]),
        "19_all_rb_w1_receptions_rmse_worsen_le_0p25pct": rel_worsen(b_all_rec["rmse"], c_all_rec["rmse"]) <= 0.0025,
        "20_all_rb_w1_receptions_p90_worsen_le_1pct": rel_worsen(b_all_rec["p90_abs_error"], c_all_rec["p90_abs_error"]) <= 0.01,
    }

    all_pass = bool(all(gates.values()))
    disposition = (
        "WEEK1_COMPONENT_QUALIFIED_FOR_2026_PROSPECTIVE_SHADOW"
        if all_pass else "WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW"
    )
    result = {
        "candidate": "RB_R26E_WEEK1_COMPONENT_QUALIFICATION_V1",
        "parent_candidate": "RB_R26_VACANCY_GATED_R9_RETROSPECTIVE_V1",
        "scientific_label": "RETROSPECTIVE_WEEK1_COMPONENT_QUALIFICATION",
        "disposition": disposition,
        "all_frozen_gates_pass": all_pass,
        "gates": gates,
        "temporal": {
            "seasons_improved": int(season_improve_count),
            "max_single_season_relative_mae_worsening": float(season_max_worsen),
            "season_rows": season_rows,
        },
        "primary": {
            "rows": int(len(inc)),
            "baseline_receptions": b_inc_rec,
            "candidate_receptions": c_inc_rec,
            "baseline_targets": b_inc_tgt,
            "candidate_targets": c_inc_tgt,
        },
        "role": {
            "RB1": {"rows": int(len(rb1)), "baseline_receptions": b_rb1, "candidate_receptions": c_rb1},
            "RB2PLUS": {"rows": int(len(rb2)), "baseline_receptions": b_rb2, "candidate_receptions": c_rb2},
        },
        "global_week1": {
            "rows": int(len(w1)),
            "baseline_receptions": b_all_rec,
            "candidate_receptions": c_all_rec,
        },
        "structural": {
            "sportsbook_inputs_used": sportsbook,
            "future_outcomes_used": future_pred + future_audit,
            "strict_prior_fit": strict_prior,
            "max_receiving_yard_mean_delta": max_ry,
            "max_r22_authority_delta": max_r22,
        },
        "production_promotion_authorized": False,
        "prospective_2026_week1_shadow_authorized": all_pass,
        "r26_full_candidate_failure_preserved": True,
        "r26d_mixed_router_failure_preserved": True,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "r26e_week1_qualification_disposition.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(season_rows).to_csv(a.out_dir / "r26e_week1_season_metrics.csv", index=False)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
