#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TEAM_ALIAS = {"JAC": "JAX", "JAX": "JAX", "LA": "LAR", "LAR": "LAR"}


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}, got {len(hits)}")
    return hits[0]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def team(v):
    raw = str(v or "").strip().upper()
    return TEAM_ALIAS.get(raw, raw)


def corr(a, b, method="pearson"):
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    return float(z.a.corr(z.b, method=method)) if len(z) > 2 else np.nan


def score(frame: pd.DataFrame, signal: str, target: str) -> dict:
    g = frame.loc[num(frame[signal]).notna() & num(frame[target]).notna()].copy()
    s = num(g[signal])
    y = num(g[target])
    q1 = float(s.quantile(.25))
    q4 = float(s.quantile(.75))
    nz = s.ne(0) & y.ne(0)
    same = float((np.sign(s.loc[nz]) == np.sign(y.loc[nz])).mean()) if nz.any() else np.nan
    return {
        "n": int(len(g)),
        "pearson": corr(s, y, "pearson"),
        "spearman": corr(s, y, "spearman"),
        "same_sign_rate": same,
        "signal_q1": q1,
        "signal_q4": q4,
        "target_q4_minus_q1_gap": float(y.loc[s.ge(q4)].mean() - y.loc[s.le(q1)].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m89-root", type=Path, required=True)
    ap.add_argument("--nd5-root", type=Path, required=True)
    ap.add_argument("--wr-r1-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    qb = pd.read_csv(one(a.m89_root, "m89_corrected_qb_common_trace.csv"), low_memory=False)
    qb.columns = [str(c).strip().lower() for c in qb.columns]
    qb = qb.loc[num(qb["season"]).isin([2024, 2025])].copy()
    qb["season"] = num(qb["season"]).astype(int)
    qb["week"] = num(qb["week"]).astype(int)
    qb["team"] = qb["team"].map(team)
    qb["actual_attempts"] = num(qb["actual_attempts"])
    qb["pred_attempts"] = num(qb["pred_attempts"])
    qb["qb_attempt_residual"] = qb["actual_attempts"] - qb["pred_attempts"]
    qb = qb[["season", "week", "team", "player_clean_key", "actual_attempts", "pred_attempts", "qb_attempt_residual"]]
    if len(qb) != 884 or qb.duplicated(["season", "week", "team"]).any():
        raise RuntimeError(f"QB cohort drift rows={len(qb)} duplicates={int(qb.duplicated(['season','week','team']).sum())}")

    nd = pd.read_csv(one(a.nd5_root, "wr_nd5_casebook.csv"), low_memory=False)
    nd.columns = [str(c).strip().lower() for c in nd.columns]
    if len(nd) != 2130:
        raise RuntimeError(f"ND5 row drift {len(nd)}")
    nd["season"] = num(nd["season"]).astype(int)
    nd["week"] = num(nd["week"]).astype(int)
    nd["team"] = nd["team"].map(team)
    nd["actual_targets"] = num(nd["actual_targets"])
    nd["pred_targets"] = num(nd["pred_targets"])
    tgt = nd.groupby(["season", "week", "team"], as_index=False).agg(
        wr_actual_targets=("actual_targets", "sum"),
        wr_pred_targets=("pred_targets", "sum"),
        wr_count=("player_clean_key", "size"),
    )
    tgt["wr_target_mass_residual"] = tgt["wr_actual_targets"] - tgt["wr_pred_targets"]

    primary = qb.loc[qb["season"].eq(2025)].merge(tgt, on=["season", "week", "team"], how="inner", validate="one_to_one")
    p = score(primary, "wr_target_mass_residual", "qb_attempt_residual")
    p_w2 = score(primary.loc[primary["week"].between(2, 18)], "wr_target_mass_residual", "qb_attempt_residual")
    p_w13 = score(primary.loc[primary["week"].between(13, 18)], "wr_target_mass_residual", "qb_attempt_residual")

    paired = pd.read_csv(one(a.wr_r1_root, "wr_r1_paired_wr_casebook.csv"), low_memory=False)
    paired.columns = [str(c).strip().lower() for c in paired.columns]
    w = paired.loc[
        paired["market"].astype(str).str.lower().eq("receptions")
        & paired["position"].astype(str).str.upper().eq("WR")
        & num(paired["season"]).isin([2024, 2025])
    ].copy()
    w["season"] = num(w["season"]).astype(int)
    w["week"] = num(w["week"]).astype(int)
    w["team"] = w["team"].map(team)
    w["actual_rec"] = num(w["actual_m38"])
    w["proj_rec"] = num(w["mc_proj_m38"])
    rec = w.groupby(["season", "week", "team"], as_index=False).agg(
        wr_actual_receptions=("actual_rec", "sum"),
        wr_proj_receptions=("proj_rec", "sum"),
        wr_count=("player_clean_key", "size"),
    )
    rec["wr_reception_mass_residual"] = rec["wr_actual_receptions"] - rec["wr_proj_receptions"]
    secondary = qb.merge(rec, on=["season", "week", "team"], how="inner", validate="one_to_one")
    s_all = score(secondary, "wr_reception_mass_residual", "qb_attempt_residual")
    s_2024 = score(secondary.loc[secondary["season"].eq(2024)], "wr_reception_mass_residual", "qb_attempt_residual")
    s_2025 = score(secondary.loc[secondary["season"].eq(2025)], "wr_reception_mass_residual", "qb_attempt_residual")

    gates = {
        "aligned_2025_ge400": bool(p["n"] >= 400),
        "pearson_ge_0_50": bool(p["pearson"] >= 0.50),
        "spearman_ge_0_45": bool(p["spearman"] >= 0.45),
        "same_sign_ge_0_65": bool(p["same_sign_rate"] >= 0.65),
        "quartile_gap_ge5_attempts": bool(p["target_q4_minus_q1_gap"] >= 5.0),
        "w2_18_pearson_positive": bool(p_w2["pearson"] > 0),
        "w13_18_pearson_positive": bool(p_w13["pearson"] > 0),
    }
    disposition = "STRONG_SHARED_PASS_VOLUME_MECHANISM" if all(gates.values()) else "NO_STRONG_SHARED_PASS_VOLUME_MECHANISM"

    result = {
        "migration": "QB_WR_SHARED_PASS_VOLUME_MECHANISM",
        "qb_rows": int(len(qb)),
        "nd5_rows": int(len(nd)),
        "primary_target_mass": p,
        "primary_w2_18": p_w2,
        "primary_w13_18": p_w13,
        "secondary_reception_mass_all": s_all,
        "secondary_reception_mass_2024": s_2024,
        "secondary_reception_mass_2025": s_2025,
        "gates": gates,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    primary.to_csv(a.out_dir / "qb_wr_shared_pass_volume_primary_2025.csv", index=False)
    secondary.to_csv(a.out_dir / "qb_wr_shared_pass_volume_secondary_2024_2025.csv", index=False)
    (a.out_dir / "qb_wr_shared_pass_volume_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
