"""M96A: RB opportunity vs efficiency attribution audit.

Research-only, no fitting/search.  This audit combines only already-frozen
M94C/M95F/M95I outputs and asks how much rushing-yard error is recoverable from
perfect opportunity versus perfect rushing efficiency.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PLAYER_KEYS = ["season", "week", "team", "player_join_key"]
ALIASES = {
    # Historical source normalization used by M95F drops the terminal accented e;
    # M94C retained it. This is an identity-only bridge, already exposed by source
    # comparison before scientific scoring.
    "audricestime": "audricestim",
}


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return hits[0]


def canon_team(x) -> str:
    s = "" if pd.isna(x) else str(x).upper().strip()
    return {"OAK": "LV", "SD": "LAC", "STL": "LA", "JAX": "JAC"}.get(s, s)


def clean_player_key(x) -> str:
    s = "" if pd.isna(x) else str(x)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return ALIASES.get(s, s)


def prep(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    z.columns = [str(c).lower() for c in z.columns]
    z["season"] = num(z["season"]).astype(int)
    z["week"] = num(z["week"]).astype(int)
    z["team"] = z["team"].map(canon_team)
    z["player_join_key"] = z["player_clean_key"].map(clean_player_key)
    return z


def point_metrics(actual, pred) -> dict:
    z = pd.DataFrame({"actual": num(actual), "pred": num(pred)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan,
                "actual_mean": np.nan, "pred_mean": np.nan}
    err = z["pred"] - z["actual"]
    corr = z["actual"].corr(z["pred"]) if len(z) >= 3 and z["actual"].nunique() > 1 and z["pred"].nunique() > 1 else np.nan
    return {
        "n": int(len(z)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()),
        "corr": float(corr) if pd.notna(corr) else np.nan,
        "actual_mean": float(z["actual"].mean()),
        "pred_mean": float(z["pred"].mean()),
    }


def auc_safe(y, score) -> float:
    z = pd.DataFrame({"y": num(y), "s": num(score)}).dropna()
    if z.empty or z["y"].nunique() < 2:
        return np.nan
    return float(roc_auc_score(z["y"].astype(int), z["s"]))


def slices(df: pd.DataFrame) -> dict[str, pd.Series]:
    a = num(df["actual_carries"])
    vacancy = num(df["prior_top1_unavailable"]).fillna(0).eq(1)
    share_trend = num(df.get("rb_rb_share_avg1", pd.Series(np.nan, index=df.index))) - num(
        df.get("rb_rb_share_avg5", pd.Series(np.nan, index=df.index))
    )
    stable = num(df.get("role_is_workhorse", pd.Series(0, index=df.index))).fillna(0).eq(1) & share_trend.ge(-0.10)
    return {
        "all_rb": pd.Series(True, index=df.index),
        "actual_0_5": a.between(0, 5),
        "actual_6_10": a.between(6, 10),
        "actual_11_14": a.between(11, 14),
        "actual_15_19": a.between(15, 19),
        "actual_20_plus": a.ge(20),
        "actual_25_plus": a.ge(25),
        "incumbent": ~vacancy,
        "vacancy": vacancy,
        "stable_workhorse": stable,
    }


def load_inputs(m94c_root: Path, m95f_root: Path, m95i_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    m94 = prep(pd.read_csv(find_one(m94c_root, "m94c_2025_rb_trace.csv"), low_memory=False))
    f = prep(pd.read_csv(find_one(m95f_root, "m95f_2025_rb_trace.csv"), low_memory=False))
    i = prep(pd.read_csv(find_one(m95i_root, "m95i_2025_trace.csv"), low_memory=False))

    for name, x in [("m94c", m94), ("m95f", f), ("m95i", i)]:
        if x.duplicated(PLAYER_KEYS).any():
            raise RuntimeError(f"{name} has duplicate M96A player-game keys")

    f_keep = PLAYER_KEYS + [
        "player_clean_key", "actual_carries", "actual_rush_yards", "m94c_rush_att",
        "m95f_mix_mean", "m95f_p50", "m95f_p75", "m95f_p90", "m95f_p95",
        "cal_prob_20", "cal_prob_25", "role_is_workhorse", "rb_rb_share_avg1", "rb_rb_share_avg5",
    ]
    i_keep = PLAYER_KEYS + [
        "m95i_rush_att", "m95i_tail_uplift", "m95i_tail_eligible",
        "prior_top1_unavailable", "p20_joint", "p25_joint",
    ]
    m_keep = PLAYER_KEYS + ["candidate_rush_att", "candidate_rush_yards", "actual_rush_att", "actual_rush_yards"]

    f2 = f[[c for c in f_keep if c in f.columns]].copy()
    i2 = i[[c for c in i_keep if c in i.columns]].copy()
    m2 = m94[[c for c in m_keep if c in m94.columns]].copy().rename(
        columns={"actual_rush_yards": "actual_rush_yards_m94c"}
    )

    fi = f2.merge(i2, on=PLAYER_KEYS, how="inner", validate="one_to_one")
    all3 = fi.merge(m2, on=PLAYER_KEYS, how="inner", validate="one_to_one")

    source_rows = []
    base_n = len(f2)
    for name, x in [("m95f", f2), ("m95i", i2), ("m94c", m2), ("m95f_m95i", fi), ("all_three", all3)]:
        source_rows.append({
            "source": name,
            "rows": int(len(x)),
            "coverage_vs_m95f": float(len(x) / base_n) if base_n else np.nan,
        })
    source = pd.DataFrame(source_rows)

    if len(fi) / max(base_n, 1) < 0.995 or len(all3) / max(base_n, 1) < 0.995:
        raise RuntimeError(f"M96A source join coverage below 99.5%: {source.to_dict(orient='records')}")

    # Exact truth and central-carry parity gates.
    carry_diff = np.abs(num(all3["m94c_rush_att"]) - num(all3["candidate_rush_att"]))
    actual_carry_diff = np.abs(num(all3["actual_carries"]) - num(all3["actual_rush_att"]))
    yard_diff = np.abs(num(all3["actual_rush_yards"]) - num(all3["actual_rush_yards_m94c"]))
    source["max_m94c_carry_parity_diff"] = float(carry_diff.max())
    source["max_actual_carry_parity_diff"] = float(actual_carry_diff.max())
    source["max_actual_yard_parity_diff"] = float(yard_diff.max())
    if carry_diff.max() > 1e-6 or actual_carry_diff.max() > 1e-6 or yard_diff.max() > 1e-6:
        raise RuntimeError("M96A frozen trace parity gate failed")

    return all3, source


def add_arms(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    att = num(z["candidate_rush_att"])
    yards = num(z["candidate_rush_yards"])
    z["pred_ypc"] = np.where(att.abs() > 1e-9, yards / att, np.nan)
    if not np.isfinite(z["pred_ypc"].dropna()).all():
        raise RuntimeError("M96A non-finite M94C implied YPC")

    z["pred_ypc_clamped_sensitivity"] = num(z["pred_ypc"]).clip(2.0, 7.0)
    z["arm_m94c_carries"] = num(z["m94c_rush_att"])
    z["arm_m95f_carries"] = num(z["m95f_mix_mean"])
    vacancy = num(z["prior_top1_unavailable"]).fillna(0).eq(1)
    z["arm_hybrid_carries"] = np.where(vacancy, num(z["m95i_rush_att"]), num(z["m95f_mix_mean"]))

    for arm, c in [
        ("m94c", "arm_m94c_carries"),
        ("m95f_distribution", "arm_m95f_carries"),
        ("m95f_plus_m95i_vacancy", "arm_hybrid_carries"),
    ]:
        z[f"pred_yards_{arm}"] = num(z[c]) * num(z["pred_ypc"])
        z[f"pred_yards_{arm}_clamped_sensitivity"] = num(z[c]) * num(z["pred_ypc_clamped_sensitivity"])

    # M94C must reconstruct its own frozen yard prediction exactly in the primary calculation.
    exact_diff = np.abs(num(z["pred_yards_m94c"]) - num(z["candidate_rush_yards"]))
    if exact_diff.max() > 1e-6:
        raise RuntimeError(f"M96A M94C yard reconstruction failed: max diff {exact_diff.max()}")

    actual_c = num(z["actual_carries"])
    actual_y = num(z["actual_rush_yards"])
    z["actual_ypc_for_attribution"] = np.where(
        actual_c.gt(0), actual_y / actual_c, num(z["pred_ypc"])
    )
    return z


def point_table(z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    arm_cols = {
        "m94c": "pred_yards_m94c",
        "m95f_distribution": "pred_yards_m95f_distribution",
        "m95f_plus_m95i_vacancy": "pred_yards_m95f_plus_m95i_vacancy",
    }
    for sl, mask in slices(z).items():
        q = z.loc[mask]
        for arm, pcol in arm_cols.items():
            rows.append({"slice": sl, "arm": arm, **point_metrics(q["actual_rush_yards"], q[pcol])})
    return pd.DataFrame(rows)


def oracle_table(z: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    comp_rows = []
    arms = {
        "m94c": "arm_m94c_carries",
        "m95f_distribution": "arm_m95f_carries",
        "m95f_plus_m95i_vacancy": "arm_hybrid_carries",
    }
    actual_c = num(z["actual_carries"])
    actual_y = num(z["actual_rush_yards"])
    pred_ypc = num(z["pred_ypc"])
    actual_ypc = num(z["actual_ypc_for_attribution"])

    for arm, ccol in arms.items():
        carries = num(z[ccol])
        pred = carries * pred_ypc
        oracle_opp = actual_c * pred_ypc
        oracle_eff = carries * actual_ypc
        opp_comp = (actual_c - carries) * pred_ypc
        eff_comp = actual_c * (actual_ypc - pred_ypc)
        residual = actual_y - pred
        identity_err = np.abs(residual - (opp_comp + eff_comp))
        if np.nanmax(identity_err) > 1e-6:
            raise RuntimeError(f"M96A attribution identity failed for {arm}: {np.nanmax(identity_err)}")

        zz = z.copy()
        zz["_pred"] = pred; zz["_opp"] = oracle_opp; zz["_eff"] = oracle_eff
        zz["_opp_comp"] = opp_comp; zz["_eff_comp"] = eff_comp; zz["_resid"] = residual
        for sl, mask in slices(zz).items():
            q = zz.loc[mask]
            p = point_metrics(q["actual_rush_yards"], q["_pred"])
            o = point_metrics(q["actual_rush_yards"], q["_opp"])
            e = point_metrics(q["actual_rush_yards"], q["_eff"])
            rows.append({
                "slice": sl, "arm": arm, "n": p["n"],
                "pregame_mae": p["mae"], "perfect_opportunity_mae": o["mae"],
                "perfect_efficiency_mae": e["mae"],
                "opportunity_mae_recovery": p["mae"] - o["mae"],
                "efficiency_mae_recovery": p["mae"] - e["mae"],
                "pregame_rmse": p["rmse"], "perfect_opportunity_rmse": o["rmse"],
                "perfect_efficiency_rmse": e["rmse"],
            })

            qa = q[["_opp_comp", "_eff_comp", "_resid"]].replace([np.inf, -np.inf], np.nan).dropna()
            if qa.empty:
                continue
            abs_o = qa["_opp_comp"].abs(); abs_e = qa["_eff_comp"].abs()
            comp_rows.append({
                "slice": sl, "arm": arm, "n": int(len(qa)),
                "mean_abs_opportunity_component": float(abs_o.mean()),
                "median_abs_opportunity_component": float(abs_o.median()),
                "mean_abs_efficiency_component": float(abs_e.mean()),
                "median_abs_efficiency_component": float(abs_e.median()),
                "opportunity_dominant_share": float((abs_o > abs_e).mean()),
                "efficiency_dominant_share": float((abs_e > abs_o).mean()),
                "tie_share": float(np.isclose(abs_o, abs_e, atol=1e-12).mean()),
                "opportunity_component_residual_corr": float(qa["_opp_comp"].corr(qa["_resid"])) if len(qa) >= 3 else np.nan,
                "efficiency_component_residual_corr": float(qa["_eff_comp"].corr(qa["_resid"])) if len(qa) >= 3 else np.nan,
                "max_identity_error": float(identity_err.loc[qa.index].max()),
            })
    return pd.DataFrame(rows), pd.DataFrame(comp_rows)


def tail_table(z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [75, 100]:
        truth = num(z["actual_rush_yards"]).ge(threshold).astype(int)
        for arm, pcol in [
            ("m94c", "pred_yards_m94c"),
            ("m95f_distribution", "pred_yards_m95f_distribution"),
            ("m95f_plus_m95i_vacancy", "pred_yards_m95f_plus_m95i_vacancy"),
        ]:
            rows.append({
                "threshold": threshold, "arm": arm, "n": int(len(z)),
                "events": int(truth.sum()), "event_rate": float(truth.mean()),
                "auc": auc_safe(truth, z[pcol]), "mean_score_yards": float(num(z[pcol]).mean()),
            })
    return pd.DataFrame(rows)


def quantile_table(z: pd.DataFrame) -> pd.DataFrame:
    actual = num(z["actual_rush_yards"])
    rows = []
    for q, c in [(0.50, "m95f_p50"), (0.75, "m95f_p75"), (0.90, "m95f_p90"), (0.95, "m95f_p95")]:
        yard_q = num(z[c]) * num(z["pred_ypc"])
        rows.append({
            "quantile": q, "n": int(yard_q.notna().sum()),
            "coverage_actual_yards_le_quantile": float((actual <= yard_q).mean()),
            "nominal": q, "coverage_gap": float((actual <= yard_q).mean() - q),
            "mean_predicted_yard_quantile": float(yard_q.mean()),
        })
    return pd.DataFrame(rows)


def clamp_sensitivity(z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    changed = ~np.isclose(num(z["pred_ypc"]), num(z["pred_ypc_clamped_sensitivity"]), atol=1e-12)
    for arm in ["m94c", "m95f_distribution", "m95f_plus_m95i_vacancy"]:
        raw = point_metrics(z["actual_rush_yards"], z[f"pred_yards_{arm}"])
        clp = point_metrics(z["actual_rush_yards"], z[f"pred_yards_{arm}_clamped_sensitivity"])
        rows.append({
            "arm": arm, "changed_rows": int(changed.sum()),
            "raw_mae": raw["mae"], "clamped_mae": clp["mae"], "clamped_minus_raw_mae": clp["mae"] - raw["mae"],
        })
    return pd.DataFrame(rows)


def choose_route(oracle: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    o = oracle.loc[(oracle["slice"] == "all_rb") & (oracle["arm"] == "m94c")].iloc[0]
    c = components.loc[(components["slice"] == "all_rb") & (components["arm"] == "m94c")].iloc[0]
    og = float(o["opportunity_mae_recovery"]); eg = float(o["efficiency_mae_recovery"])
    os = float(c["opportunity_dominant_share"]); es = float(c["efficiency_dominant_share"])
    if og >= eg + 1.0 and os >= 0.55:
        route = "OPPORTUNITY_DOMINANT_RETURN_TO_TARGETED_OPPORTUNITY"
    elif eg >= og + 1.0 and es >= 0.55:
        route = "EFFICIENCY_DOMINANT_ADVANCE_M96B_ENVIRONMENT_EFFICIENCY"
    else:
        route = "JOINT_ADVANCE_M96B_SEPARATE_WORKLOAD_AND_EFFICIENCY_DISTRIBUTIONS"
    return pd.DataFrame([{
        "m94c_pregame_mae": float(o["pregame_mae"]),
        "perfect_opportunity_mae": float(o["perfect_opportunity_mae"]),
        "perfect_efficiency_mae": float(o["perfect_efficiency_mae"]),
        "opportunity_mae_recovery": og,
        "efficiency_mae_recovery": eg,
        "opportunity_dominant_share": os,
        "efficiency_dominant_share": es,
        "route_margin_yards": abs(og - eg),
        "routing_rule_yard_margin": 1.0,
        "routing_rule_component_share": 0.55,
        "disposition": route,
        "model_fit": 0, "feature_search": 0, "coefficient_search": 0,
        "sportsbook_inputs": 0, "production_change": 0,
    }])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--m95f-root", type=Path, required=True)
    ap.add_argument("--m95i-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    z, source = load_inputs(args.m94c_root, args.m95f_root, args.m95i_root)
    z = add_arms(z)
    points = point_table(z)
    oracle, components = oracle_table(z)
    tails = tail_table(z)
    quantiles = quantile_table(z)
    clamp = clamp_sensitivity(z)
    disposition = choose_route(oracle, components)

    source.to_csv(args.out_dir / "m96a_source_parity.csv", index=False)
    points.to_csv(args.out_dir / "m96a_point_metrics.csv", index=False)
    oracle.to_csv(args.out_dir / "m96a_oracle_attribution.csv", index=False)
    components.to_csv(args.out_dir / "m96a_residual_components.csv", index=False)
    tails.to_csv(args.out_dir / "m96a_tail_discrimination.csv", index=False)
    quantiles.to_csv(args.out_dir / "m96a_m95f_yard_quantile_coverage.csv", index=False)
    clamp.to_csv(args.out_dir / "m96a_ypc_clamp_sensitivity.csv", index=False)
    disposition.to_csv(args.out_dir / "m96a_disposition.csv", index=False)

    keep = [
        "season", "week", "team", "player_clean_key", "actual_carries", "actual_rush_yards",
        "pred_ypc", "arm_m94c_carries", "arm_m95f_carries", "arm_hybrid_carries",
        "pred_yards_m94c", "pred_yards_m95f_distribution", "pred_yards_m95f_plus_m95i_vacancy",
        "m95f_p50", "m95f_p75", "m95f_p90", "m95f_p95", "cal_prob_20", "cal_prob_25",
        "prior_top1_unavailable", "m95i_rush_att", "p20_joint", "p25_joint", "role_is_workhorse",
    ]
    z[[c for c in keep if c in z.columns]].to_csv(args.out_dir / "m96a_2025_trace.csv", index=False)

    print("=== M96A source parity ===")
    print(source.to_string(index=False))
    print("=== M96A disposition ===")
    print(disposition.to_string(index=False))
    print("=== M96A all-RB point metrics ===")
    print(points.loc[points["slice"].eq("all_rb")].to_string(index=False))
    print("=== M96A all-RB oracle attribution ===")
    print(oracle.loc[oracle["slice"].eq("all_rb")].to_string(index=False))
    print("=== M96A M94C all-RB residual components ===")
    print(components.loc[(components["slice"].eq("all_rb")) & (components["arm"].eq("m94c"))].to_string(index=False))
    print("=== M96A tail discrimination ===")
    print(tails.to_string(index=False))
    print("=== M96A M95F yard quantile coverage ===")
    print(quantiles.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
