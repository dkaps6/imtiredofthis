#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-6
CORR_TOL = 1e-9
KEYS = ["season", "week", "team", "player_clean_key"]
PRIMARY_REF = {
    "pearson": 0.6973048915041206,
    "spearman": 0.670648420892482,
    "same_sign": 0.7318181818181818,
}
SECONDARY_REF = {
    "pearson": 0.5238926874684945,
    "spearman": 0.49940402459076466,
    "same_sign": 0.7092760180995475,
}


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def clean_keys(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["season"] = num(x["season"])
    x["week"] = num(x["week"])
    x["team"] = x["team"].fillna("").astype(str).str.upper().str.strip()
    x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str).str.strip()
    return x


def metrics(actual, pred, miss_levels=()):
    a = num(actual).to_numpy(float)
    p = num(pred).to_numpy(float)
    if len(a) == 0 or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise RuntimeError("invalid metric arrays")
    e = p - a
    out = {
        "n": int(len(a)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "bias": float(np.mean(e)),
        "corr": float(np.corrcoef(a, p)[0, 1]) if len(a) >= 2 else np.nan,
        "p90_abs_error": float(np.quantile(np.abs(e), 0.90)),
    }
    for level in miss_levels:
        out[f"miss_{int(level)}_plus_rate"] = float(np.mean(np.abs(e) >= level))
    return out


def corr_metrics(x, y):
    z = pd.DataFrame({"x": num(x), "y": num(y)}).dropna()
    if len(z) < 3:
        return {"n": int(len(z)), "pearson": np.nan, "spearman": np.nan, "same_sign": np.nan}
    return {
        "n": int(len(z)),
        "pearson": float(z.x.corr(z.y, method="pearson")),
        "spearman": float(z.x.corr(z.y, method="spearman")),
        "same_sign": float((np.sign(z.x) == np.sign(z.y)).mean()),
    }


def q4_q1_gap(rank, y):
    z = pd.DataFrame({"r": num(rank), "y": num(y)}).dropna()
    q1 = z.r.quantile(0.25)
    q4 = z.r.quantile(0.75)
    return float(z.loc[z.r.ge(q4), "y"].mean() - z.loc[z.r.le(q1), "y"].mean())


def load_chain(root: Path) -> pd.DataFrame:
    cols = KEYS + [
        "actual_pass_yards", "actual_attempts", "pred_attempts", "pred_ypa",
        "football_synthesis", "pred_D", "pred_C", "pred_S", "actual_D",
    ]
    x = pd.read_csv(one(root, "qb_opportunity_chain_casebook.csv"), usecols=cols, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    x = clean_keys(x)
    for c in ["actual_pass_yards", "actual_attempts", "pred_attempts", "pred_ypa", "football_synthesis", "pred_d", "pred_c", "pred_s", "actual_d"]:
        x[c] = num(x[c])
    if len(x) != 884:
        raise RuntimeError(f"chain row drift {len(x)}")
    if x.duplicated(KEYS).any():
        raise RuntimeError("duplicate chain keys")
    if (x.pred_ypa <= 0).any() or ((x.pred_c * x.pred_s) <= 0).any():
        raise RuntimeError("nonpositive implied-volume denominator")
    return x


def add_implied(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    z["baseline_attempt_identity"] = z.pred_d * z.pred_c * z.pred_s
    z["implied_attempts"] = z.football_synthesis / z.pred_ypa
    z["implied_d"] = z.implied_attempts / (z.pred_c * z.pred_s)
    z["implied_attempt_correction"] = z.implied_attempts - z.pred_attempts
    z["implied_d_correction"] = z.implied_d - z.pred_d
    z["actual_attempt_residual"] = z.actual_attempts - z.pred_attempts
    z["implied_mean_identity"] = z.implied_attempts * z.pred_ypa
    z["implied_d_identity"] = z.implied_d * z.pred_c * z.pred_s
    return z


def season_summary(z: pd.DataFrame) -> dict:
    out = {}
    for label, g in [
        ("2024", z.loc[z.season.eq(2024)]),
        ("2025", z.loc[z.season.eq(2025)]),
        ("POOLED_2024_2025", z),
    ]:
        base_a = metrics(g.actual_attempts, g.pred_attempts, miss_levels=(8, 10))
        imp_a = metrics(g.actual_attempts, g.implied_attempts, miss_levels=(8, 10))
        base_d = metrics(g.actual_d, g.pred_d)
        imp_d = metrics(g.actual_d, g.implied_d)
        cm = corr_metrics(g.implied_attempt_correction, g.actual_attempt_residual)
        cm.update({
            "mean_correction": float(g.implied_attempt_correction.mean()),
            "mean_abs_correction": float(g.implied_attempt_correction.abs().mean()),
            "p90_abs_correction": float(g.implied_attempt_correction.abs().quantile(0.90)),
            "min_correction": float(g.implied_attempt_correction.min()),
            "max_correction": float(g.implied_attempt_correction.max()),
        })
        out[label] = {
            "qb_attempts": {"baseline": base_a, "implied": imp_a, "mae_gain": base_a["mae"] - imp_a["mae"]},
            "team_pass_opportunity": {"baseline": base_d, "implied": imp_d, "mae_gain": base_d["mae"] - imp_d["mae"]},
            "correction_vs_actual_attempt_residual": cm,
        }
    return out


def shared_attribution(z: pd.DataFrame, root: Path):
    p = pd.read_csv(one(root, "qb_wr_shared_pass_volume_primary_2025.csv"), low_memory=False)
    s = pd.read_csv(one(root, "qb_wr_shared_pass_volume_secondary_2024_2025.csv"), low_memory=False)
    p.columns = [str(c).strip().lower() for c in p.columns]
    s.columns = [str(c).strip().lower() for c in s.columns]
    p, s = clean_keys(p), clean_keys(s)
    if len(p) != 440 or len(s) != 884:
        raise RuntimeError(f"shared row drift primary={len(p)} secondary={len(s)}")
    if p.duplicated(KEYS).any() or s.duplicated(KEYS).any():
        raise RuntimeError("duplicate shared keys")

    keep = KEYS + ["actual_attempt_residual", "implied_attempt_correction"]
    p = p.merge(z[keep], on=KEYS, how="left", validate="one_to_one")
    s = s.merge(z[keep], on=KEYS, how="left", validate="one_to_one")
    if p["implied_attempt_correction"].isna().any() or s["implied_attempt_correction"].isna().any():
        raise RuntimeError("shared attribution alignment missing")

    p_resid_gap = float((num(p.qb_attempt_residual) - p.actual_attempt_residual).abs().max())
    s_resid_gap = float((num(s.qb_attempt_residual) - s.actual_attempt_residual).abs().max())
    p_ref = corr_metrics(p.qb_attempt_residual, p.wr_target_mass_residual)
    s_ref = corr_metrics(s.qb_attempt_residual, s.wr_reception_mass_residual)

    rows = []
    for view, g, ycol, season_views in [
        ("PRIMARY_WR_TARGET_MASS", p, "wr_target_mass_residual", [("2025", p)]),
        ("SECONDARY_WR_RECEPTION_MASS", s, "wr_reception_mass_residual", [
            ("POOLED_2024_2025", s),
            ("2024", s.loc[s.season.eq(2024)]),
            ("2025", s.loc[s.season.eq(2025)]),
        ]),
    ]:
        for season_label, g2 in season_views:
            m = corr_metrics(g2.implied_attempt_correction, g2[ycol])
            rows.append({
                "view": view,
                "season": season_label,
                **m,
                "signed_correction_q4_minus_q1_wr_residual_gap": q4_q1_gap(g2.implied_attempt_correction, g2[ycol]),
            })
    return pd.DataFrame(rows), p_ref, s_ref, p_resid_gap, s_resid_gap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-root", type=Path, required=True)
    ap.add_argument("--shared-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    z = add_implied(load_chain(a.chain_root))
    summaries = season_summary(z)
    shared, p_ref, s_ref, p_gap, s_gap = shared_attribution(z, a.shared_root)

    identity = {
        "current_attempt_identity_max_abs": float((z.baseline_attempt_identity - z.pred_attempts).abs().max()),
        "implied_mean_identity_max_abs": float((z.implied_mean_identity - z.football_synthesis).abs().max()),
        "implied_d_identity_max_abs": float((z.implied_d_identity - z.implied_attempts).abs().max()),
        "primary_shared_attempt_residual_max_abs": p_gap,
        "secondary_shared_attempt_residual_max_abs": s_gap,
    }
    corr_repro = (
        abs(p_ref["pearson"] - PRIMARY_REF["pearson"]) <= CORR_TOL
        and abs(p_ref["spearman"] - PRIMARY_REF["spearman"]) <= CORR_TOL
        and abs(p_ref["same_sign"] - PRIMARY_REF["same_sign"]) <= CORR_TOL
        and abs(s_ref["pearson"] - SECONDARY_REF["pearson"]) <= CORR_TOL
        and abs(s_ref["spearman"] - SECONDARY_REF["spearman"]) <= CORR_TOL
        and abs(s_ref["same_sign"] - SECONDARY_REF["same_sign"]) <= CORR_TOL
        and p_gap <= TOL and s_gap <= TOL
    )
    integrity_gates = {
        "exact_884_chain_rows": len(z) == 884,
        "exact_440_and_884_shared_rows": True,
        "zero_sportsbook_inputs": True,
        "zero_model_fitting": True,
        "no_production_change": True,
        "positive_denominators": bool((z.pred_ypa > 0).all() and ((z.pred_c * z.pred_s) > 0).all()),
        "current_attempt_identity": identity["current_attempt_identity_max_abs"] <= TOL,
        "implied_mean_identity": identity["implied_mean_identity_max_abs"] <= TOL,
        "implied_d_identity": identity["implied_d_identity_max_abs"] <= TOL,
        "shared_source_correlations_reproduce": corr_repro,
        "target_outcomes_diagnostic_only": True,
    }
    all_integrity = all(integrity_gates.values())

    pooled = summaries["POOLED_2024_2025"]
    y24 = summaries["2024"]
    y25 = summaries["2025"]
    primary_row = shared.loc[(shared.view == "PRIMARY_WR_TARGET_MASS") & (shared.season == "2025")].iloc[0]
    secondary_pool = shared.loc[(shared.view == "SECONDARY_WR_RECEPTION_MASS") & (shared.season == "POOLED_2024_2025")].iloc[0]

    support_gates = {
        "pooled_attempt_mae_gain_ge_0_25": pooled["qb_attempts"]["mae_gain"] >= 0.25,
        "attempt_mae_nonworse_2024": y24["qb_attempts"]["mae_gain"] >= -1e-12,
        "attempt_mae_nonworse_2025": y25["qb_attempts"]["mae_gain"] >= -1e-12,
        "pooled_team_pass_opportunity_mae_gain_ge_0_25": pooled["team_pass_opportunity"]["mae_gain"] >= 0.25,
        "team_pass_opportunity_mae_nonworse_2024": y24["team_pass_opportunity"]["mae_gain"] >= -1e-12,
        "team_pass_opportunity_mae_nonworse_2025": y25["team_pass_opportunity"]["mae_gain"] >= -1e-12,
        "pooled_correction_vs_attempt_residual_spearman_ge_0_20": pooled["correction_vs_actual_attempt_residual"]["spearman"] >= 0.20,
        "correction_vs_attempt_residual_spearman_2024_ge_0_10": y24["correction_vs_actual_attempt_residual"]["spearman"] >= 0.10,
        "correction_vs_attempt_residual_spearman_2025_ge_0_10": y25["correction_vs_actual_attempt_residual"]["spearman"] >= 0.10,
        "wr_target_2025_spearman_ge_0_20": float(primary_row.spearman) >= 0.20,
        "wr_reception_pooled_spearman_ge_0_15": float(secondary_pool.spearman) >= 0.15,
        "pooled_10_plus_attempt_miss_nonworse": pooled["qb_attempts"]["implied"]["miss_10_plus_rate"] <= pooled["qb_attempts"]["baseline"]["miss_10_plus_rate"] + 1e-12,
        "qb_mean_preservation": identity["implied_mean_identity_max_abs"] <= TOL,
        "all_integrity_gates_pass": all_integrity,
    }
    supported = all(support_gates.values())
    disposition = (
        "MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE" if not all_integrity
        else "M89_SYNTHESIS_CONTAINS_REUSABLE_OPPORTUNITY_SIGNAL" if supported
        else "M89_SYNTHESIS_OPPORTUNITY_REALLOCATION_NOT_SUPPORTED"
    )

    result = {
        "migration": "QB_SYNTHESIS_OPPORTUNITY_REPARAMETERIZATION_A1",
        "disposition": disposition,
        "production_actionable": False,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "identity": identity,
        "integrity_gates": integrity_gates,
        "support_gates": support_gates,
        "season_summary": summaries,
        "shared_reference_reproduction": {"primary": p_ref, "secondary": s_ref},
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    z.to_csv(a.out_dir / "qb_synthesis_opportunity_reparameterization_casebook.csv", index=False)
    shared.to_csv(a.out_dir / "qb_synthesis_opportunity_reparameterization_shared.csv", index=False)
    (a.out_dir / "qb_synthesis_opportunity_reparameterization_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all_integrity else 2


if __name__ == "__main__":
    raise SystemExit(main())
