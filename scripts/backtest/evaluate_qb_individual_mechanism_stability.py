#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}, got {len(hits)}")
    return hits[0]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def corr(a, b, method="pearson"):
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    return float(z.a.corr(z.b, method=method)) if len(z) > 2 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m89-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    r = pd.read_csv(one(a.m89_root, "m89_corrected_qb_common_trace.csv"), low_memory=False)
    s = pd.read_csv(one(a.m89_root, "m89_2024_2025_synthesis_trace.csv"), low_memory=False)
    r.columns = [str(c).strip().lower() for c in r.columns]
    s.columns = [str(c).strip().lower() for c in s.columns]

    k = ["season", "week", "team", "player_clean_key"]
    cols = k + ["actual_attempts", "actual_ypa", "actual_pass_yards", "pred_attempts", "pred_ypa"]
    x = s[k + ["base_proj", "football_synthesis", "actual_pass_yards"]].rename(
        columns={"actual_pass_yards": "synth_actual"}
    ).merge(r[cols], on=k, how="inner", validate="one_to_one")
    if len(x) != 884:
        raise RuntimeError(f"aligned row drift {len(x)}")

    for c in ["actual_attempts", "actual_ypa", "actual_pass_yards", "pred_attempts", "pred_ypa", "base_proj", "football_synthesis", "synth_actual"]:
        x[c] = num(x[c])
    x["mechanics_proj"] = x["pred_attempts"] * x["pred_ypa"]
    x["attempt_component"] = (x["actual_attempts"] - x["pred_attempts"]) * (x["actual_ypa"] + x["pred_ypa"]) / 2
    x["ypa_component"] = (x["actual_ypa"] - x["pred_ypa"]) * (x["actual_attempts"] + x["pred_attempts"]) / 2
    x["stack_adjustment"] = x["base_proj"] - x["mechanics_proj"]
    x["synthesis_adjustment"] = x["football_synthesis"] - x["base_proj"]
    x["final_residual"] = x["actual_pass_yards"] - x["football_synthesis"]
    x["decomp_reconstructed"] = x["attempt_component"] + x["ypa_component"] - x["stack_adjustment"] - x["synthesis_adjustment"]
    decomp_err = float((x["final_residual"] - x["decomp_reconstructed"]).abs().max())
    if decomp_err > 1e-6:
        raise RuntimeError(f"decomposition drift {decomp_err}")
    x["attempt_residual"] = x["actual_attempts"] - x["pred_attempts"]
    x["ypa_residual"] = x["actual_ypa"] - x["pred_ypa"]

    season_profiles = []
    for (pk, season), g in x.groupby(["player_clean_key", "season"]):
        if len(g) < 6:
            continue
        vals = {
            "ATTEMPTS": float(g["attempt_component"].abs().mean()),
            "YPA": float(g["ypa_component"].abs().mean()),
            "STACK": float(g["stack_adjustment"].abs().mean()),
            "SYNTHESIS": float(g["synthesis_adjustment"].abs().mean()),
        }
        total = sum(vals.values())
        season_profiles.append({
            "player_key": pk,
            "season": int(season),
            "games": int(len(g)),
            "yard_mae": float(g["final_residual"].abs().mean()),
            "yard_bias": float(g["final_residual"].mean()),
            "attempt_residual_mean": float(g["attempt_residual"].mean()),
            "attempt_residual_mae": float(g["attempt_residual"].abs().mean()),
            "ypa_residual_mean": float(g["ypa_residual"].mean()),
            "ypa_residual_mae": float(g["ypa_residual"].abs().mean()),
            "attempt_abs": vals["ATTEMPTS"],
            "ypa_abs": vals["YPA"],
            "stack_abs": vals["STACK"],
            "synthesis_abs": vals["SYNTHESIS"],
            "attempt_share": vals["ATTEMPTS"] / total if total else np.nan,
            "ypa_share": vals["YPA"] / total if total else np.nan,
            "stack_share": vals["STACK"] / total if total else np.nan,
            "synthesis_share": vals["SYNTHESIS"] / total if total else np.nan,
            "dominant_component": max(vals, key=vals.get),
        })
    ps = pd.DataFrame(season_profiles)

    p24 = ps.loc[ps["season"].eq(2024)].copy()
    p25 = ps.loc[ps["season"].eq(2025)].copy()
    wide = p24.merge(p25, on="player_key", suffixes=("_2024", "_2025"), how="inner", validate="one_to_one")
    wide = wide.loc[(wide["games_2024"] + wide["games_2025"]).ge(16)].copy()

    same_dom = float((wide["dominant_component_2024"] == wide["dominant_component_2025"]).mean()) if len(wide) else np.nan
    att_sign = float((np.sign(wide["attempt_residual_mean_2024"]) == np.sign(wide["attempt_residual_mean_2025"])).mean()) if len(wide) else np.nan
    ypa_sign = float((np.sign(wide["ypa_residual_mean_2024"]) == np.sign(wide["ypa_residual_mean_2025"])).mean()) if len(wide) else np.nan

    metrics = {
        "qualifying_qbs": int(len(wide)),
        "same_dominant_component_rate": same_dom,
        "attempt_share_pearson": corr(wide["attempt_share_2024"], wide["attempt_share_2025"], "pearson"),
        "attempt_share_spearman": corr(wide["attempt_share_2024"], wide["attempt_share_2025"], "spearman"),
        "ypa_share_pearson": corr(wide["ypa_share_2024"], wide["ypa_share_2025"], "pearson"),
        "ypa_share_spearman": corr(wide["ypa_share_2024"], wide["ypa_share_2025"], "spearman"),
        "yard_mae_pearson": corr(wide["yard_mae_2024"], wide["yard_mae_2025"], "pearson"),
        "yard_mae_spearman": corr(wide["yard_mae_2024"], wide["yard_mae_2025"], "spearman"),
        "attempt_bias_sign_persistence": att_sign,
        "ypa_bias_sign_persistence": ypa_sign,
    }
    gates = {
        "qualifying_qbs_ge18": bool(metrics["qualifying_qbs"] >= 18),
        "same_dominant_ge0_55": bool(same_dom >= 0.55),
        "attempt_share_spearman_ge0_30": bool(metrics["attempt_share_spearman"] >= 0.30),
        "ypa_share_spearman_ge0_30": bool(metrics["ypa_share_spearman"] >= 0.30),
        "one_bias_sign_persistence_ge0_60": bool(max(att_sign, ypa_sign) >= 0.60),
    }
    disposition = "QB_PLAYER_MECHANISMS_SHOW_USEFUL_STABILITY" if all(gates.values()) else "QB_PLAYER_MECHANISMS_REQUIRE_CONTEXT_REGIMES"

    result = {
        "migration": "QB_INDIVIDUAL_MECHANISM_STABILITY",
        "rows": int(len(x)),
        "decomposition_max_abs_error": decomp_err,
        "season_profiles": int(len(ps)),
        "metrics": metrics,
        "gates": gates,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    ps.to_csv(a.out_dir / "qb_individual_mechanism_season_profiles.csv", index=False)
    wide.to_csv(a.out_dir / "qb_individual_mechanism_stability_pairs.csv", index=False)
    (a.out_dir / "qb_individual_mechanism_stability_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if len(wide):
        show = ["player_key", "games_2024", "games_2025", "dominant_component_2024", "dominant_component_2025", "attempt_share_2024", "attempt_share_2025", "ypa_share_2024", "ypa_share_2025", "yard_mae_2024", "yard_mae_2025"]
        print(wide[show].sort_values("yard_mae_2025", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
