#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 2130


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}, got {len(hits)}")
    return hits[0]


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def corr(a: pd.Series, b: pd.Series, method: str) -> float:
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    if len(z) <= 2 or z["a"].nunique() <= 1 or z["b"].nunique() <= 1:
        return np.nan
    return float(z["a"].corr(z["b"], method=method))


def score(frame: pd.DataFrame, signal: str) -> dict:
    g = frame.loc[num(frame[signal]).notna() & num(frame["allocation_residual"]).notna()].copy()
    if g.empty:
        return {
            "n": 0, "pearson": np.nan, "spearman": np.nan, "same_sign_rate": np.nan,
            "q1": np.nan, "q4": np.nan, "allocation_q4_minus_q1_gap": np.nan,
            "raw_target_q4_minus_q1_gap": np.nan,
        }
    s = num(g[signal])
    alloc = num(g["allocation_residual"])
    raw = num(g["raw_target_residual"])
    q1 = float(s.quantile(0.25))
    q4 = float(s.quantile(0.75))
    nz = s.ne(0) & alloc.ne(0)
    same = float((np.sign(s.loc[nz]) == np.sign(alloc.loc[nz])).mean()) if nz.any() else np.nan
    hi = s.ge(q4)
    lo = s.le(q1)
    return {
        "n": int(len(g)),
        "pearson": corr(s, alloc, "pearson"),
        "spearman": corr(s, alloc, "spearman"),
        "same_sign_rate": same,
        "q1": q1,
        "q4": q4,
        "allocation_q4_minus_q1_gap": float(alloc.loc[hi].mean() - alloc.loc[lo].mean()) if hi.any() and lo.any() else np.nan,
        "raw_target_q4_minus_q1_gap": float(raw.loc[hi].mean() - raw.loc[lo].mean()) if hi.any() and lo.any() else np.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-r5-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    x = pd.read_csv(one(a.wr_r5_root, "wr_r5_mechanism_casebook.csv"), low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if len(x) != EXPECTED_ROWS:
        raise RuntimeError(f"WR-R5 casebook drift expected={EXPECTED_ROWS} got={len(x)}")

    required = {"season", "week", "team", "player_clean_key", "m38_wr_role", "pred_targets", "actual_targets"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"WR-R5 casebook missing {sorted(missing)}")

    x["season"] = num(x["season"]).astype(int)
    x["week"] = num(x["week"]).astype(int)
    x["pred_targets"] = num(x["pred_targets"])
    x["actual_targets"] = num(x["actual_targets"])
    x["m38_wr_role"] = x["m38_wr_role"].fillna("").astype(str)

    keys = ["season", "week", "team"]
    x["pred_wr_target_mass"] = x.groupby(keys)["pred_targets"].transform("sum")
    x["actual_wr_target_mass"] = x.groupby(keys)["actual_targets"].transform("sum")
    x["pred_wr_share"] = np.where(x["pred_wr_target_mass"].gt(0), x["pred_targets"] / x["pred_wr_target_mass"], np.nan)
    x["actual_wr_share"] = np.where(x["actual_wr_target_mass"].gt(0), x["actual_targets"] / x["actual_wr_target_mass"], np.nan)
    x["allocation_residual"] = x["actual_wr_share"] - x["pred_wr_share"]
    x["raw_target_residual"] = x["actual_targets"] - x["pred_targets"]

    x = x.sort_values(["season", "player_clean_key", "week", "team"], kind="stable").reset_index(drop=True)
    gp = x.groupby(["season", "player_clean_key"], sort=False)
    x["prior4_wr_share_resid"] = gp["allocation_residual"].transform(lambda s: s.shift(1).rolling(4, min_periods=3).mean())
    x["prior4_raw_target_resid"] = gp["raw_target_residual"].transform(lambda s: s.shift(1).rolling(4, min_periods=3).mean())
    x["previous_m38_wr_role"] = gp["m38_wr_role"].shift(1)
    x["stable_role"] = x["previous_m38_wr_role"].notna() & x["m38_wr_role"].eq(x["previous_m38_wr_role"])
    x["role_change"] = x["previous_m38_wr_role"].notna() & ~x["m38_wr_role"].eq(x["previous_m38_wr_role"])

    coverage = float(x["prior4_wr_share_resid"].notna().mean())
    stable = x.loc[x["stable_role"] & x["prior4_wr_share_resid"].notna()].copy()
    changed = x.loc[x["role_change"] & x["prior4_wr_share_resid"].notna()].copy()

    primary = score(stable, "prior4_wr_share_resid")
    secondary = score(stable, "prior4_raw_target_resid")
    role_change = score(changed, "prior4_wr_share_resid")
    w2 = score(stable.loc[stable["week"].between(2, 18)], "prior4_wr_share_resid")
    w13 = score(stable.loc[stable["week"].between(13, 18)], "prior4_wr_share_resid")

    role_scores = {}
    positive_roles = 0
    for role in ["WR1", "WR2", "WR3"]:
        rs = score(stable.loc[stable["m38_wr_role"].eq(role)], "prior4_wr_share_resid")
        role_scores[role] = rs
        if np.isfinite(rs["spearman"]) and rs["spearman"] > 0:
            positive_roles += 1

    gates = {
        "coverage_ge_0_65": bool(coverage >= 0.65),
        "stable_role_n_ge_700": bool(primary["n"] >= 700),
        "stable_role_spearman_ge_0_10": bool(np.isfinite(primary["spearman"]) and primary["spearman"] >= 0.10),
        "stable_role_allocation_gap_ge_0_025": bool(np.isfinite(primary["allocation_q4_minus_q1_gap"]) and primary["allocation_q4_minus_q1_gap"] >= 0.025),
        "stable_role_raw_target_gap_ge_0_75": bool(np.isfinite(primary["raw_target_q4_minus_q1_gap"]) and primary["raw_target_q4_minus_q1_gap"] >= 0.75),
        "stable_role_same_sign_ge_0_58": bool(np.isfinite(primary["same_sign_rate"]) and primary["same_sign_rate"] >= 0.58),
        "w2_18_spearman_positive": bool(np.isfinite(w2["spearman"]) and w2["spearman"] > 0),
        "w13_18_spearman_positive": bool(np.isfinite(w13["spearman"]) and w13["spearman"] > 0),
        "two_of_three_wr_roles_positive": bool(positive_roles >= 2),
    }
    passed = all(gates.values())
    disposition = "WR_PLAYER_TARGET_PERSISTENCE_DISCOVERY_PASS" if passed else "NO_ACTIONABLE_WR_PLAYER_TARGET_PERSISTENCE_2025"

    result = {
        "migration": "WR_R6_PLAYER_TARGET_RESIDUAL_PERSISTENCE",
        "rows": int(len(x)),
        "signal_coverage": coverage,
        "stable_role_rows_with_signal": int(len(stable)),
        "role_change_rows_with_signal": int(len(changed)),
        "primary_prior4_wr_share_resid_stable": primary,
        "secondary_prior4_raw_target_resid_stable": secondary,
        "role_change_contrast": role_change,
        "primary_w2_18": w2,
        "primary_w13_18": w13,
        "role_slices": role_scores,
        "positive_wr_role_slices": int(positive_roles),
        "gates": gates,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "multiseason_replication_required_if_pass": True,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "wr_r6_player_target_persistence_casebook.csv", index=False)
    pd.DataFrame([
        {"slice": "STABLE_PRIMARY_SHARE", **primary},
        {"slice": "STABLE_SECONDARY_RAW", **secondary},
        {"slice": "ROLE_CHANGE_PRIMARY_SHARE", **role_change},
        {"slice": "STABLE_W2_18", **w2},
        {"slice": "STABLE_W13_18", **w13},
        *[{"slice": f"STABLE_{role}", **role_scores[role]} for role in ["WR1", "WR2", "WR3"]],
    ]).to_csv(a.out_dir / "wr_r6_score_summary.csv", index=False)
    (a.out_dir / "wr_r6_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
