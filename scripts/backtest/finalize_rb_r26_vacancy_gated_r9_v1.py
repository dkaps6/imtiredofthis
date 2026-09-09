#!/usr/bin/env python3
"""Apply frozen RB R26 vacancy-gated R9 retrospective support gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.evaluate_rb_r25_receptions_specialist_v1 import metric


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing required artifact: {path}")
    return pd.read_csv(path, low_memory=False)


def m(g: pd.DataFrame, variant: str, market: str) -> dict:
    actual = f"actual_{market}"
    pred = f"{variant}_{market}"
    return metric(g[actual], g[pred])


def safe_improve(base: float, cand: float) -> bool:
    return bool(np.isfinite(base) and np.isfinite(cand) and cand < base)


def rel_worsen(base: float, cand: float) -> float:
    if not np.isfinite(base) or not np.isfinite(cand):
        return np.inf
    if abs(base) < 1e-12:
        return 0.0 if abs(cand) < 1e-12 else np.inf
    return float(cand / base - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--protected-clean-marker", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    if not a.protected_clean_marker.exists() or a.protected_clean_marker.read_text().strip() != "PASS":
        raise RuntimeError("R26 protected-production clean marker missing")

    preds = []
    audits = []
    fits = []
    for d in a.dirs:
        preds.append(read(d / "r26_predictions.csv"))
        audits.append(read(d / "r26_structural_audit.csv"))
        fits.append(json.loads((d / "r26_fit_metadata.json").read_text()))
    pred = pd.concat(preds, ignore_index=True)
    audit = pd.concat(audits, ignore_index=True)

    pred["season"] = pd.to_numeric(pred.season, errors="coerce").astype(int)
    pred["week"] = pd.to_numeric(pred.week, errors="coerce").astype(int)
    pred["vacancy_active"] = pd.to_numeric(pred.vacancy_active, errors="coerce").fillna(0).astype(int)
    pred["continuing_same_team"] = pd.to_numeric(pred.continuing_same_team, errors="coerce").fillna(0).astype(int)
    pred["rb_rank"] = pd.to_numeric(pred.rb_rank, errors="coerce").fillna(99).astype(int)

    inc = pred.loc[pred.vacancy_active.eq(1) & pred.continuing_same_team.eq(1)].copy()
    rb1 = inc.loc[inc.rb_rank.eq(1)].copy()
    rb2 = inc.loc[inc.rb_rank.ge(2)].copy()
    w1 = pred.loc[pred.week.eq(1)].copy()

    base_inc_rec = m(inc, "baseline", "receptions")
    cand_inc_rec = m(inc, "candidate", "receptions")
    base_inc_tgt = m(inc, "baseline", "targets")
    cand_inc_tgt = m(inc, "candidate", "targets")
    base_all_rec = m(pred, "baseline", "receptions")
    cand_all_rec = m(pred, "candidate", "receptions")
    base_w1_rec = m(w1, "baseline", "receptions")
    cand_w1_rec = m(w1, "candidate", "receptions")
    base_rb1_rec = m(rb1, "baseline", "receptions")
    cand_rb1_rec = m(rb1, "candidate", "receptions")
    base_rb2_rec = m(rb2, "baseline", "receptions")
    cand_rb2_rec = m(rb2, "candidate", "receptions")

    season_rows = []
    season_improve_count = 0
    season_max_worsen = -np.inf
    for season in sorted(pred.season.unique()):
        g = inc.loc[inc.season.eq(int(season))]
        b = m(g, "baseline", "receptions")
        c = m(g, "candidate", "receptions")
        worsen = rel_worsen(float(b["mae"]), float(c["mae"]))
        if safe_improve(float(b["mae"]), float(c["mae"])):
            season_improve_count += 1
        season_max_worsen = max(season_max_worsen, worsen)
        season_rows.append({
            "season": int(season),
            "n": int(b["n"]),
            "baseline_receptions_mae": float(b["mae"]),
            "candidate_receptions_mae": float(c["mae"]),
            "relative_mae_change": worsen,
        })

    max_room_gap = float(pd.to_numeric(audit.room_mass_gap, errors="coerce").abs().max())
    max_team_gap = float(pd.to_numeric(audit.team_entitlement_gap, errors="coerce").abs().max())
    max_non_rb = float(pd.to_numeric(audit.max_non_rb_entitlement_delta, errors="coerce").abs().max())
    max_ry = float(pd.to_numeric(audit.max_receiving_yard_mean_delta, errors="coerce").abs().max())
    max_r22 = float(pd.to_numeric(audit.r22_authority_delta, errors="coerce").abs().max())
    sportsbook = int(pd.to_numeric(audit.sportsbook_inputs_used, errors="coerce").fillna(0).sum())
    future = int(pd.to_numeric(audit.future_outcomes_used, errors="coerce").fillna(0).sum())
    strict_prior = bool(all(bool(x.get("strict_prior_fit")) for x in fits))

    gates = {
        "01_sportsbook_inputs_zero": sportsbook == 0,
        "02_future_outcome_features_zero": future == 0,
        "03_strict_prior_state_and_fit": strict_prior,
        "04_rb_room_mass_gap_lt_1e10": max_room_gap < 1e-10,
        "05_non_rb_entitlement_delta_lt_1e12": max_non_rb < 1e-12,
        "06_receiving_yard_mean_exact": max_ry == 0.0,
        "07_r22_authority_exact": max_r22 == 0.0,
        "08_protected_production_files_clean": True,
        "09_vacancy_incumbent_receptions_mae_improves": safe_improve(base_inc_rec["mae"], cand_inc_rec["mae"]),
        "10_vacancy_incumbent_receptions_rmse_nonworse": cand_inc_rec["rmse"] <= base_inc_rec["rmse"],
        "11_vacancy_incumbent_abs_bias_nonworse": abs(cand_inc_rec["bias"]) <= abs(base_inc_rec["bias"]),
        "12_vacancy_incumbent_receptions_p90_worsen_le_2pct": rel_worsen(base_inc_rec["p90_abs_error"], cand_inc_rec["p90_abs_error"]) <= 0.02,
        "13_vacancy_incumbent_target_mae_improves": safe_improve(base_inc_tgt["mae"], cand_inc_tgt["mae"]),
        "14_at_least_4_of_6_seasons_improve": season_improve_count >= 4,
        "15_no_season_worsens_more_than_2pct": season_max_worsen <= 0.02,
        "16_neither_role_worsens_more_than_1pct": (
            rel_worsen(base_rb1_rec["mae"], cand_rb1_rec["mae"]) <= 0.01
            and rel_worsen(base_rb2_rec["mae"], cand_rb2_rec["mae"]) <= 0.01
        ),
        "17_at_least_one_role_improves": (
            safe_improve(base_rb1_rec["mae"], cand_rb1_rec["mae"])
            or safe_improve(base_rb2_rec["mae"], cand_rb2_rec["mae"])
        ),
        "18_all_rb_receptions_mae_worsen_le_0p25pct": rel_worsen(base_all_rec["mae"], cand_all_rec["mae"]) <= 0.0025,
        "19_all_rb_receptions_rmse_worsen_le_0p25pct": rel_worsen(base_all_rec["rmse"], cand_all_rec["rmse"]) <= 0.0025,
        "20_week1_receptions_mae_worsen_le_0p50pct": rel_worsen(base_w1_rec["mae"], cand_w1_rec["mae"]) <= 0.005,
    }

    all_pass = bool(all(gates.values()))
    disposition = (
        "RETROSPECTIVE_SUPPORT_FOR_2026_PROSPECTIVE_SHADOW"
        if all_pass else "RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW"
    )

    result = {
        "candidate": "RB_R26_VACANCY_GATED_R9_RETROSPECTIVE_V1",
        "scientific_label": "PREDECLARED_RETROSPECTIVE_MECHANISM_TEST",
        "disposition": disposition,
        "all_frozen_gates_pass": all_pass,
        "gates": gates,
        "structural": {
            "max_rb_room_mass_gap": max_room_gap,
            "max_team_entitlement_gap": max_team_gap,
            "max_non_rb_entitlement_delta": max_non_rb,
            "max_receiving_yard_mean_delta": max_ry,
            "max_r22_authority_delta": max_r22,
            "sportsbook_inputs_used": sportsbook,
            "future_outcomes_used_in_features": future,
        },
        "coverage": {
            "min_room_state_coverage": float(min(float(x.get("room_state_coverage", 0.0)) for x in fits)),
            "min_player_state_coverage": float(min(float(x.get("player_state_coverage", 0.0)) for x in fits)),
        },
        "vacancy_incumbent": {
            "rows": int(len(inc)),
            "baseline_receptions": base_inc_rec,
            "candidate_receptions": cand_inc_rec,
            "baseline_targets": base_inc_tgt,
            "candidate_targets": cand_inc_tgt,
        },
        "role": {
            "RB1": {"baseline_receptions": base_rb1_rec, "candidate_receptions": cand_rb1_rec},
            "RB2PLUS": {"baseline_receptions": base_rb2_rec, "candidate_receptions": cand_rb2_rec},
        },
        "global": {
            "baseline_receptions": base_all_rec,
            "candidate_receptions": cand_all_rec,
            "week1_baseline_receptions": base_w1_rec,
            "week1_candidate_receptions": cand_w1_rec,
        },
        "temporal": {
            "seasons_improved": int(season_improve_count),
            "max_single_season_relative_mae_worsening": float(season_max_worsen),
            "season_rows": season_rows,
        },
        "fit_metadata": fits,
        "production_promotion_authorized": False,
        "prospective_2026_shadow_authorized": all_pass,
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(season_rows).to_csv(a.out.with_name("r26_season_gate_metrics.csv"), index=False)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
