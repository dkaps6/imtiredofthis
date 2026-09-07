#!/usr/bin/env python3
"""Cross-position catastrophic casebook V1, Phase A artifact decomposition.

Consumes exact frozen research artifacts only. No underlying model is refit.
Postgame play-level forensics are a separate Phase B; this phase isolates the
mathematical opportunity/share/conversion/efficiency layers on the frozen rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def qcut4(s: pd.Series) -> pd.Series:
    return pd.qcut(pd.to_numeric(s, errors="coerce"), 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")


def dominant(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").abs() for c in cols}, index=df.index).idxmax(axis=1)


def qb_frame(root: Path) -> pd.DataFrame:
    syn = pd.read_csv(root / "synthesis" / "m89_2024_2025_synthesis_trace.csv", low_memory=False)
    corr = pd.read_csv(root / "reconciliation" / "m89_corrected_qb_common_trace.csv", low_memory=False)
    keys = ["season", "week", "team", "player_clean_key"]
    syn = syn.merge(corr[keys + ["actual_attempts", "actual_ypa"]].drop_duplicates(keys), on=keys, how="left", validate="one_to_one")
    x = syn.copy()
    x["position"] = "QB"
    x["player"] = x["player_clean_key"]
    x["pred"] = pd.to_numeric(x["football_synthesis"], errors="coerce")
    x["actual"] = pd.to_numeric(x["actual_pass_yards"], errors="coerce")
    x["pred_opportunity"] = pd.to_numeric(x["pred_attempts"], errors="coerce")
    x["actual_opportunity"] = pd.to_numeric(x["actual_attempts"], errors="coerce")
    x["pred_efficiency"] = pd.to_numeric(x["pred_ypa"], errors="coerce")
    x["actual_efficiency"] = pd.to_numeric(x["actual_ypa"], errors="coerce")
    x["team_pool_contrib"] = (x["actual_opportunity"] - x["pred_opportunity"]) * x["pred_efficiency"]
    x["share_contrib"] = 0.0
    x["conversion_contrib"] = 0.0
    x["efficiency_contrib"] = x["actual_opportunity"] * (x["actual_efficiency"] - x["pred_efficiency"])
    x["other_contrib"] = (x["actual"] - pd.to_numeric(x["base_proj"], errors="coerce")) - x["team_pool_contrib"] - x["efficiency_contrib"]
    x["synthesis_shift"] = pd.to_numeric(x["football_synthesis"], errors="coerce") - pd.to_numeric(x["base_proj"], errors="coerce")
    x["threshold"] = 100.0
    x["source_model"] = "QB_PASS_SYNTHESIS_V1_M89_M90"
    x["dominant_mechanism"] = dominant(x, ["team_pool_contrib", "efficiency_contrib", "other_contrib"])
    x["dominant_mechanism"] = x["dominant_mechanism"].replace({
        "team_pool_contrib": "TEAM_OPPORTUNITY_MISS",
        "efficiency_contrib": "EFFICIENCY",
        "other_contrib": "MIXED_OR_MODEL_RESIDUAL",
    })
    return x


def wr_frame(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x = x.loc[x["position"].astype(str).str.upper().eq("WR")].copy()
    x["pred"] = pd.to_numeric(x["b0_rec_yards"], errors="coerce")
    x["actual"] = pd.to_numeric(x["rec_yards"], errors="coerce")
    x["pred_opportunity"] = pd.to_numeric(x["b0_expected_targets"], errors="coerce")
    x["actual_opportunity"] = pd.to_numeric(x["targets"], errors="coerce")
    x["pred_rec"] = pd.to_numeric(x["b0_receptions"], errors="coerce")
    x["actual_rec"] = pd.to_numeric(x["receptions"], errors="coerce")
    x["pred_cr"] = np.where(x["pred_opportunity"] > 0, x["pred_rec"] / x["pred_opportunity"], np.nan)
    x["pred_ypr"] = np.where(x["pred_rec"] > 0, x["pred"] / x["pred_rec"], np.nan)
    x["actual_cr"] = np.where(x["actual_opportunity"] > 0, x["actual_rec"] / x["actual_opportunity"], np.nan)
    x["actual_ypr"] = np.where(x["actual_rec"] > 0, x["actual"] / x["actual_rec"], np.nan)
    g = x.groupby(["season", "week", "team"], as_index=False).agg(pred_pool=("pred_opportunity", "sum"), actual_pool=("actual_opportunity", "sum"))
    x = x.merge(g, on=["season", "week", "team"], how="left")
    x["pred_share"] = np.where(x["pred_pool"] > 0, x["pred_opportunity"] / x["pred_pool"], np.nan)
    x["actual_share"] = np.where(x["actual_pool"] > 0, x["actual_opportunity"] / x["actual_pool"], np.nan)
    x["pred_ypt"] = np.where(x["pred_opportunity"] > 0, x["pred"] / x["pred_opportunity"], np.nan)
    x["team_pool_contrib"] = (x["actual_pool"] - x["pred_pool"]) * x["pred_share"] * x["pred_ypt"]
    x["share_contrib"] = x["actual_pool"] * (x["actual_share"] - x["pred_share"]) * x["pred_ypt"]
    x["conversion_contrib"] = x["actual_opportunity"] * (x["actual_cr"] - x["pred_cr"]) * x["pred_ypr"]
    x["efficiency_contrib"] = x["actual_rec"] * (x["actual_ypr"] - x["pred_ypr"])
    x["other_contrib"] = (x["actual"] - x["pred"]) - x[["team_pool_contrib", "share_contrib", "conversion_contrib", "efficiency_contrib"]].sum(axis=1)
    x["pred_efficiency"] = x["pred_ypr"]
    x["actual_efficiency"] = x["actual_ypr"]
    x["threshold"] = 50.0
    x["source_model"] = "M38_B0_RECEIVING_MEAN"
    x["dominant_mechanism"] = dominant(x, ["team_pool_contrib", "share_contrib", "conversion_contrib", "efficiency_contrib"])
    x["dominant_mechanism"] = x["dominant_mechanism"].replace({
        "team_pool_contrib": "TEAM_OPPORTUNITY_MISS", "share_contrib": "PLAYER_ENTITLEMENT_MISS",
        "conversion_contrib": "CONVERSION_MISS", "efficiency_contrib": "EFFICIENCY",
    })
    return x


def te_frame(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x["pred"] = pd.to_numeric(x["candidate_rec_yards_r5"], errors="coerce")
    x["actual"] = pd.to_numeric(x["rec_yards"], errors="coerce")
    x["pred_opportunity"] = pd.to_numeric(x["candidate_targets_r5"], errors="coerce")
    x["actual_opportunity"] = pd.to_numeric(x["targets"], errors="coerce")
    x["pred_rec"] = pd.to_numeric(x["candidate_receptions_r5"], errors="coerce")
    x["actual_rec"] = pd.to_numeric(x["receptions"], errors="coerce")
    x["pred_cr"] = pd.to_numeric(x["b0_rec_per_target"], errors="coerce")
    x["pred_ypt"] = pd.to_numeric(x["b0_rec_yards_per_target"], errors="coerce")
    x["pred_ypr"] = np.where(x["pred_cr"] > 0, x["pred_ypt"] / x["pred_cr"], np.nan)
    x["actual_cr"] = np.where(x["actual_opportunity"] > 0, x["actual_rec"] / x["actual_opportunity"], np.nan)
    x["actual_ypr"] = np.where(x["actual_rec"] > 0, x["actual"] / x["actual_rec"], np.nan)
    x["team_pool_contrib"] = (pd.to_numeric(x["actual_te_pool"], errors="coerce") - pd.to_numeric(x["candidate_te_pool"], errors="coerce")) * pd.to_numeric(x["candidate_room_share"], errors="coerce") * x["pred_ypt"]
    x["share_contrib"] = pd.to_numeric(x["actual_te_pool"], errors="coerce") * (pd.to_numeric(x["actual_room_share"], errors="coerce") - pd.to_numeric(x["candidate_room_share"], errors="coerce")) * x["pred_ypt"]
    x["conversion_contrib"] = x["actual_opportunity"] * (x["actual_cr"] - x["pred_cr"]) * x["pred_ypr"]
    x["efficiency_contrib"] = x["actual_rec"] * (x["actual_ypr"] - x["pred_ypr"])
    x["other_contrib"] = (x["actual"] - x["pred"]) - x[["team_pool_contrib", "share_contrib", "conversion_contrib", "efficiency_contrib"]].sum(axis=1)
    x["pred_efficiency"] = x["pred_ypr"]
    x["actual_efficiency"] = x["actual_ypr"]
    x["position"] = "TE"
    x["threshold"] = 40.0
    x["source_model"] = "TE_R5_PARTICIPATION_ENTITLEMENT"
    x["dominant_mechanism"] = dominant(x, ["team_pool_contrib", "share_contrib", "conversion_contrib", "efficiency_contrib"])
    x["dominant_mechanism"] = x["dominant_mechanism"].replace({
        "team_pool_contrib": "TEAM_OPPORTUNITY_MISS", "share_contrib": "PLAYER_ENTITLEMENT_MISS",
        "conversion_contrib": "CONVERSION_MISS", "efficiency_contrib": "EFFICIENCY",
    })
    return x


def rb_frame(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x["pred"] = np.where(x["week"].eq(1), pd.to_numeric(x["arm_week1_stack"], errors="coerce"), pd.to_numeric(x["arm_stack2_parent"], errors="coerce"))
    x["pred_opportunity"] = np.where(x["week"].eq(1), pd.to_numeric(x["base_rush_att"], errors="coerce"), pd.to_numeric(x["enriched_att"], errors="coerce"))
    x["actual"] = pd.to_numeric(x["actual_rush_yards"], errors="coerce")
    x["actual_opportunity"] = pd.to_numeric(x["actual_rush_att"], errors="coerce")
    x["pred_efficiency"] = np.where(x["pred_opportunity"] > 0, x["pred"] / x["pred_opportunity"], np.nan)
    x["actual_efficiency"] = np.where(x["actual_opportunity"] > 0, x["actual"] / x["actual_opportunity"], np.nan)
    x["team_pool_contrib"] = (x["actual_opportunity"] - x["pred_opportunity"]) * x["pred_efficiency"]
    x["share_contrib"] = 0.0
    x["conversion_contrib"] = 0.0
    x["efficiency_contrib"] = x["actual_opportunity"] * (x["actual_efficiency"] - x["pred_efficiency"])
    x["other_contrib"] = (x["actual"] - x["pred"]) - x["team_pool_contrib"] - x["efficiency_contrib"]
    x["threshold"] = 40.0
    x["source_model"] = "RB_P3_SYNTHESIS_V1"
    x["dominant_mechanism"] = dominant(x, ["team_pool_contrib", "efficiency_contrib"])
    x["dominant_mechanism"] = x["dominant_mechanism"].replace({"team_pool_contrib": "OPPORTUNITY_MISS", "efficiency_contrib": "EFFICIENCY"})
    return x


def standardize(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["abs_error"] = (pd.to_numeric(x["actual"], errors="coerce") - pd.to_numeric(x["pred"], errors="coerce")).abs()
    x["signed_error_actual_minus_pred"] = pd.to_numeric(x["actual"], errors="coerce") - pd.to_numeric(x["pred"], errors="coerce")
    x["direction"] = np.where(x["signed_error_actual_minus_pred"] > 0, "UNDERPROJECTED", "OVERPROJECTED")
    x["opportunity_quartile"] = qcut4(x["pred_opportunity"])
    x["catastrophic"] = x["abs_error"] >= pd.to_numeric(x["threshold"], errors="coerce")
    return x


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--qb-root", type=Path, required=True)
    p.add_argument("--wr-casebook", type=Path, required=True)
    p.add_argument("--te-casebook", type=Path, required=True)
    p.add_argument("--rb-casebook", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    frames = [standardize(qb_frame(a.qb_root)), standardize(wr_frame(a.wr_casebook)), standardize(te_frame(a.te_casebook)), standardize(rb_frame(a.rb_casebook))]
    keep = ["position", "season", "week", "team", "player", "player_clean_key", "source_model", "pred", "actual", "abs_error", "signed_error_actual_minus_pred", "direction", "threshold", "catastrophic", "pred_opportunity", "actual_opportunity", "pred_efficiency", "actual_efficiency", "opportunity_quartile", "team_pool_contrib", "share_contrib", "conversion_contrib", "efficiency_contrib", "other_contrib", "dominant_mechanism"]
    all_rows = pd.concat([f[[c for c in keep if c in f.columns]] for f in frames], ignore_index=True, sort=False)
    cats = all_rows.loc[all_rows["catastrophic"]].copy()

    summary = []
    for pos, g in all_rows.groupby("position"):
        c = cats.loc[cats["position"].eq(pos)]
        summary.append({"position": pos, "rows": len(g), "catastrophic_rows": len(c), "catastrophic_rate": len(c) / len(g) if len(g) else np.nan, "mae": float(g["abs_error"].mean()), "catastrophic_abs_error_mass": float(c["abs_error"].sum())})
    summary = pd.DataFrame(summary)

    mech = (cats.groupby(["position", "dominant_mechanism"], as_index=False).agg(n=("abs_error", "size"), abs_error_mass=("abs_error", "sum")))
    mech["case_share"] = mech["n"] / mech.groupby("position")["n"].transform("sum")
    mech["error_mass_share"] = mech["abs_error_mass"] / mech.groupby("position")["abs_error_mass"].transform("sum")

    q4 = all_rows.loc[all_rows["opportunity_quartile"].astype(str).eq("Q4")].copy()
    q4s = q4.groupby("position", as_index=False).agg(rows=("abs_error", "size"), catastrophic_rows=("catastrophic", "sum"), mae=("abs_error", "mean"), p90=("abs_error", lambda s: s.quantile(.9)))
    q4s["catastrophic_rate"] = q4s["catastrophic_rows"] / q4s["rows"]
    q4cats = q4.loc[q4["catastrophic"]].copy()
    q4dir = q4cats.groupby(["position", "direction", "dominant_mechanism"], as_index=False).agg(n=("abs_error", "size"), abs_error_mass=("abs_error", "sum"))
    q4dir["direction_error_mass_share"] = q4dir["abs_error_mass"] / q4dir.groupby(["position", "direction"])["abs_error_mass"].transform("sum")

    repeated = cats.groupby(["position", "player"], as_index=False).agg(cases=("abs_error", "size"), abs_error_mass=("abs_error", "sum"), mean_abs_error=("abs_error", "mean"))
    repeated = repeated.sort_values(["position", "cases", "abs_error_mass"], ascending=[True, False, False])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(a.out_dir / "cross_position_all_rows.csv", index=False)
    cats.sort_values(["position", "abs_error"], ascending=[True, False]).to_csv(a.out_dir / "cross_position_catastrophic_casebook.csv", index=False)
    summary.to_csv(a.out_dir / "cross_position_summary.csv", index=False)
    mech.to_csv(a.out_dir / "cross_position_mechanism_summary.csv", index=False)
    q4s.to_csv(a.out_dir / "cross_position_q4_summary.csv", index=False)
    q4dir.to_csv(a.out_dir / "cross_position_q4_direction_mechanisms.csv", index=False)
    repeated.to_csv(a.out_dir / "cross_position_repeated_players.csv", index=False)

    result = {
        "disposition": "PHASE_A_ARTIFACT_DECOMPOSITION_COMPLETE",
        "postgame_forensic_fields_used_for_prediction": False,
        "sportsbook_features_used_for_cause_classification": False,
        "rows": int(len(all_rows)),
        "catastrophic_rows": int(len(cats)),
        "position_summary": summary.to_dict("records"),
    }
    (a.out_dir / "cross_position_phase_a_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(summary.to_string(index=False))
    print("\n=== MECHANISMS ===")
    print(mech.sort_values(["position", "error_mass_share"], ascending=[True, False]).to_string(index=False))
    print("\n=== Q4 ===")
    print(q4s.to_string(index=False))
    print("\n=== Q4 DIRECTION / MECHANISMS ===")
    print(q4dir.sort_values(["position", "direction", "direction_error_mass_share"], ascending=[True, True, False]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
