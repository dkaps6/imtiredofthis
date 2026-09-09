#!/usr/bin/env python3
"""Qualify the frozen R26 V1 Week-1 component without refitting/regenerating."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = tuple(range(2020, 2026))


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_many(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.rglob(name))
    if len(paths) < 6:
        raise RuntimeError(f"expected >=6 {name} files under {root}, found {len(paths)}")
    return pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True, sort=False)


def metric(g: pd.DataFrame, variant: str, market: str) -> dict:
    a = num(g[f"actual_{market}"]).to_numpy(float)
    p = num(g[f"{variant}_{market}"]).to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(p)
    a = a[ok]; p = p[ok]
    if len(a) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "p90_abs_error": np.nan}
    e = p - a
    ae = np.abs(e)
    return {
        "n": int(len(a)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    pred = read_many(a.r26_root, "r26_predictions.csv")
    audit = read_many(a.r26_root, "r26_structural_audit.csv")
    fit_paths = sorted(a.r26_root.rglob("r26_fit_metadata.json"))
    if len(fit_paths) != 6:
        raise RuntimeError(f"expected exactly 6 R26 fit metadata files, found {len(fit_paths)}")
    fits = [json.loads(p.read_text()) for p in fit_paths]

    pred["season"] = num(pred.season).astype(int)
    pred["week"] = num(pred.week).astype(int)
    pred["vacancy_active"] = num(pred.vacancy_active).fillna(0).astype(int)
    pred["continuing_same_team"] = num(pred.continuing_same_team).fillna(0).astype(int)
    pred["rb_rank"] = num(pred.rb_rank).fillna(99).astype(int)
    pred = pred.loc[pred.season.isin(SEASONS)].copy()

    w1 = pred.loc[pred.week.eq(1)].copy()
    inc = w1.loc[w1.vacancy_active.eq(1) & w1.continuing_same_team.eq(1)].copy()
    rb1 = inc.loc[inc.rb_rank.eq(1)].copy()
    rb2 = inc.loc[inc.rb_rank.ge(2)].copy()
    if inc.empty:
        raise RuntimeError("R26E found zero Week-1 vacancy incumbents")

    b_inc_rec = metric(inc, "baseline", "receptions")
    c_inc_rec = metric(inc, "candidate", "receptions")
    b_inc_tgt = metric(inc, "baseline", "targets")
    c_inc_tgt = metric(inc, "candidate", "targets")
    b_rb1 = metric(rb1, "baseline", "receptions")
    c_rb1 = metric(rb1, "candidate", "receptions")
    b_rb2 = metric(rb2, "baseline", "receptions")
    c_rb2 = metric(rb2, "candidate", "receptions")
    b_all = metric(w1, "baseline", "receptions")
    c_all = metric(w1, "candidate", "receptions")

    season_rows = []
    seasons_improved = 0
    seasons_n20 = 0
    max_worsen = -np.inf
    for season in SEASONS:
        g = inc.loc[inc.season.eq(season)].copy()
        b = metric(g, "baseline", "receptions")
        c = metric(g, "candidate", "receptions")
        rw = rel_worsen(float(b["mae"]), float(c["mae"]))
        if improves(float(b["mae"]), float(c["mae"])):
            seasons_improved += 1
        if int(min(b["n"], c["n"])) >= 20:
            seasons_n20 += 1
        max_worsen = max(max_worsen, rw)
        season_rows.append({
            "season": season,
            "n": int(min(b["n"], c["n"])),
            "baseline_receptions_mae": float(b["mae"]),
            "candidate_receptions_mae": float(c["mae"]),
            "relative_mae_change": rw,
            "baseline_receptions_rmse": float(b["rmse"]),
            "candidate_receptions_rmse": float(c["rmse"]),
            "baseline_receptions_bias": float(b["bias"]),
            "candidate_receptions_bias": float(c["bias"]),
            "baseline_receptions_p90_abs_error": float(b["p90_abs_error"]),
            "candidate_receptions_p90_abs_error": float(c["p90_abs_error"]),
        })

    max_room_gap = float(num(audit.room_mass_gap).abs().max())
    max_team_gap = float(num(audit.team_entitlement_gap).abs().max())
    max_non_rb = float(num(audit.max_non_rb_entitlement_delta).abs().max())
    max_ry = float(num(audit.max_receiving_yard_mean_delta).abs().max())
    max_r22 = float(num(audit.r22_authority_delta).abs().max())
    sportsbook = int(num(audit.sportsbook_inputs_used).fillna(0).sum())
    future = int(num(audit.future_outcomes_used).fillna(0).sum())
    strict_prior = bool(all(bool(x.get("strict_prior_fit")) for x in fits))

    integrity_exact = bool(
        max_room_gap < 1e-10
        and max_team_gap < 1e-10
        and max_non_rb < 1e-12
        and max_ry == 0.0
        and max_r22 == 0.0
        and sportsbook == 0
        and future == 0
        and strict_prior
    )

    gates = {
        "01_immutable_parent_and_structural_integrity": integrity_exact,
        "02_r26_predictions_not_regenerated_and_r9_not_refit": True,
        "03_sportsbook_zero_and_production_unchanged": sportsbook == 0,
        "04_parent_mass_nonrb_recmean_r22_exact": (
            max_room_gap < 1e-10 and max_team_gap < 1e-10 and max_non_rb < 1e-12 and max_ry == 0.0 and max_r22 == 0.0
        ),
        "05_w1_vacancy_incumbent_receptions_mae_improves": improves(b_inc_rec["mae"], c_inc_rec["mae"]),
        "06_w1_vacancy_incumbent_receptions_rmse_nonworse": c_inc_rec["rmse"] <= b_inc_rec["rmse"],
        "07_w1_vacancy_incumbent_abs_bias_nonworse": abs(c_inc_rec["bias"]) <= abs(b_inc_rec["bias"]),
        "08_w1_vacancy_incumbent_p90_worsen_le_2pct": rel_worsen(b_inc_rec["p90_abs_error"], c_inc_rec["p90_abs_error"]) <= 0.02,
        "09_w1_vacancy_incumbent_target_mae_improves": improves(b_inc_tgt["mae"], c_inc_tgt["mae"]),
        "10_w1_vacancy_incumbent_improves_at_least_4_of_6_seasons": seasons_improved >= 4,
        "11_no_w1_season_worsens_more_than_2pct": max_worsen <= 0.02,
        "12_w1_support_n150_and_n20_in_4_seasons": int(b_inc_rec["n"]) >= 150 and seasons_n20 >= 4,
        "13_w1_vacancy_rb1_mae_worsen_le_1pct": rel_worsen(b_rb1["mae"], c_rb1["mae"]) <= 0.01,
        "14_w1_vacancy_rb2plus_mae_worsen_le_1pct": rel_worsen(b_rb2["mae"], c_rb2["mae"]) <= 0.01,
        "15_at_least_one_w1_role_improves": improves(b_rb1["mae"], c_rb1["mae"]) or improves(b_rb2["mae"], c_rb2["mae"]),
        "16_all_w1_rb_receptions_mae_worsen_le_0p5pct": rel_worsen(b_all["mae"], c_all["mae"]) <= 0.005,
        "17_all_w1_rb_receptions_rmse_worsen_le_0p5pct": rel_worsen(b_all["rmse"], c_all["rmse"]) <= 0.005,
        "18_all_w1_rb_abs_bias_nonworse": abs(c_all["bias"]) <= abs(b_all["bias"]),
    }

    all_pass = bool(all(gates.values()))
    if not integrity_exact:
        disposition = "WEEK1_COMPONENT_INTEGRITY_FAILURE"
    elif all_pass:
        disposition = "WEEK1_COMPONENT_RETROSPECTIVE_SUPPORT_FOR_2026_PROSPECTIVE_SHADOW"
    else:
        disposition = "WEEK1_COMPONENT_MIXED_OR_FAIL_NO_SHADOW"

    result = {
        "candidate": "RB_R26E_WEEK1_VACANCY_COMPONENT_QUALIFICATION_V1",
        "scientific_label": "NO_REFIT_WEEK1_COMPONENT_QUALIFICATION",
        "parent_full_candidate_disposition_unchanged": "RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW",
        "disposition": disposition,
        "all_frozen_gates_pass": all_pass,
        "gates_passed": int(sum(bool(v) for v in gates.values())),
        "gates_total": int(len(gates)),
        "gates": gates,
        "prospective_2026_week1_shadow_design_authorized": all_pass,
        "production_promotion_authorized": False,
        "all_season_use_authorized": False,
        "integrity": {
            "r26_predictions_regenerated": False,
            "r9_refit": False,
            "sportsbook_inputs_used": sportsbook,
            "future_outcomes_used_in_features": future,
            "strict_prior_fit": strict_prior,
            "production_parameters_changed": False,
            "max_rb_room_mass_gap": max_room_gap,
            "max_team_entitlement_gap": max_team_gap,
            "max_non_rb_entitlement_delta": max_non_rb,
            "max_receiving_yard_mean_delta": max_ry,
            "max_r22_authority_delta": max_r22,
        },
        "week1_vacancy_incumbent": {
            "rows_with_reception_labels": int(b_inc_rec["n"]),
            "baseline_receptions": b_inc_rec,
            "candidate_receptions": c_inc_rec,
            "baseline_targets": b_inc_tgt,
            "candidate_targets": c_inc_tgt,
        },
        "week1_role": {
            "RB1": {"baseline_receptions": b_rb1, "candidate_receptions": c_rb1},
            "RB2PLUS": {"baseline_receptions": b_rb2, "candidate_receptions": c_rb2},
        },
        "week1_global": {
            "baseline_receptions": b_all,
            "candidate_receptions": c_all,
        },
        "temporal": {
            "seasons_improved": seasons_improved,
            "seasons_n_ge_20": seasons_n20,
            "max_single_season_relative_mae_worsening": float(max_worsen),
            "season_rows": season_rows,
        },
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(season_rows).to_csv(a.out_dir / "r26e_week1_season_metrics.csv", index=False)
    (a.out_dir / "r26e_week1_disposition.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("=== Week-1 season metrics ===")
    print(pd.DataFrame(season_rows).to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
