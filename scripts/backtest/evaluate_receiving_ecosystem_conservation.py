#!/usr/bin/env python3
"""Evaluate the frozen WR/TE/RB/FB receiving ecosystem + QB conservation audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
PASS_GROUPS = ["WR", "TE", "RB", "FB"]
REPORT_GROUPS = ["WR", "TE", "RB", "FB", "RB_FB", "OTHER", "ALL"]
TEAM_ALIAS = {"JAC": "JAX", "JAX": "JAX", "LA": "LAR", "LAR": "LAR"}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def team(v):
    raw = str(v or "").strip().upper()
    return TEAM_ALIAS.get(raw, raw)


def metric(actual, projected) -> dict:
    z = pd.DataFrame({"actual": num(actual), "projected": num(projected)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan}
    e = z["projected"] - z["actual"]
    return {
        "n": int(len(z)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "bias": float(e.mean()),
        "correlation": float(z["projected"].corr(z["actual"])) if len(z) > 1 else np.nan,
    }


def corr(a, b, method="pearson"):
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    return float(z.a.corr(z.b, method=method)) if len(z) > 2 else np.nan


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, got {len(hits)}")
    return hits[0]


def load_season(root: Path, season: int, name: str) -> pd.DataFrame:
    path = root / str(season) / "trace" / name
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    out["season"] = num(out["season"]).astype(int)
    out["week"] = num(out["week"]).astype(int)
    out["team"] = out["team"].map(team)
    return out


def combined_group_rows(frame: pd.DataFrame, value_cols: list[str], source_group_col: str = "position_group") -> pd.DataFrame:
    keys = ["season", "week", "team"]
    base = frame.copy()
    base[source_group_col] = base[source_group_col].fillna("OTHER").astype(str).str.upper()
    base.loc[~base[source_group_col].isin(PASS_GROUPS), source_group_col] = "OTHER"
    grouped = base.groupby(keys + [source_group_col], as_index=False)[value_cols].sum(min_count=1)
    grouped = grouped.rename(columns={source_group_col: "group"})

    rbfb = (
        grouped.loc[grouped["group"].isin(["RB", "FB"])]
        .groupby(keys, as_index=False)[value_cols].sum(min_count=1)
    )
    rbfb["group"] = "RB_FB"
    all_rows = grouped.groupby(keys, as_index=False)[value_cols].sum(min_count=1)
    all_rows["group"] = "ALL"
    return pd.concat([grouped, rbfb, all_rows], ignore_index=True, sort=False)


def make_team_ledger(proj: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    pass_proj = proj.loc[proj["position_group"].isin(PASS_GROUPS)].copy()
    pvals = ["det_expected_targets", "target_share_within_modeled_receivers", "proj_receptions", "proj_rec_yards"]
    p = combined_group_rows(pass_proj, pvals)
    # Projection has no explicit OTHER receiver class; keep zero OTHER rows after the outer merge.

    avals = ["targets", "receptions", "rec_yards"]
    a = combined_group_rows(actual, avals)

    ledger = a.merge(p, on=["season", "week", "team", "group"], how="outer")
    for c in avals + pvals:
        ledger[c] = num(ledger[c]).fillna(0.0)

    # Actual shares use the complete observed receiver target/reception/yard totals.
    all_actual = (
        a.loc[a["group"].eq("ALL"), ["season", "week", "team", "targets", "receptions", "rec_yards"]]
        .rename(columns={"targets": "actual_team_targets", "receptions": "actual_team_receptions", "rec_yards": "actual_team_rec_yards"})
    )
    # Projected reception/yard shares are conditioned on the modeled receiving pool.
    all_proj = (
        p.loc[p["group"].eq("ALL"), ["season", "week", "team", "proj_receptions", "proj_rec_yards"]]
        .rename(columns={"proj_receptions": "proj_team_receptions", "proj_rec_yards": "proj_team_rec_yards"})
    )
    ledger = ledger.merge(all_actual, on=["season", "week", "team"], how="left", validate="many_to_one")
    ledger = ledger.merge(all_proj, on=["season", "week", "team"], how="left", validate="many_to_one")
    ledger["actual_target_share"] = np.where(ledger.actual_team_targets > 0, ledger.targets / ledger.actual_team_targets, np.nan)
    ledger["proj_target_share"] = ledger["target_share_within_modeled_receivers"]
    ledger["actual_reception_share"] = np.where(ledger.actual_team_receptions > 0, ledger.receptions / ledger.actual_team_receptions, np.nan)
    ledger["proj_reception_share"] = np.where(ledger.proj_team_receptions > 0, ledger.proj_receptions / ledger.proj_team_receptions, np.nan)
    ledger["actual_rec_yard_share"] = np.where(ledger.actual_team_rec_yards != 0, ledger.rec_yards / ledger.actual_team_rec_yards, np.nan)
    ledger["proj_rec_yard_share"] = np.where(ledger.proj_team_rec_yards > 0, ledger.proj_rec_yards / ledger.proj_team_rec_yards, np.nan)

    for stem in ["target_share", "reception_share", "rec_yard_share"]:
        ledger[f"{stem}_residual_actual_minus_proj"] = ledger[f"actual_{stem}"] - ledger[f"proj_{stem}"]
    ledger["receptions_residual_actual_minus_proj"] = ledger.receptions - ledger.proj_receptions
    ledger["rec_yards_residual_actual_minus_proj"] = ledger.rec_yards - ledger.proj_rec_yards
    return ledger.sort_values(["season", "week", "team", "group"]).reset_index(drop=True)


def team_metric_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("target_share", "actual_target_share", "proj_target_share"),
        ("receptions", "receptions", "proj_receptions"),
        ("reception_share", "actual_reception_share", "proj_reception_share"),
        ("rec_yards", "rec_yards", "proj_rec_yards"),
        ("rec_yard_share", "actual_rec_yard_share", "proj_rec_yard_share"),
    ]
    rows = []
    for group_name in REPORT_GROUPS:
        g0 = ledger.loc[ledger.group.eq(group_name)]
        for season_label, g in [("POOLED", g0)] + [(str(s), g0.loc[g0.season.eq(s)]) for s in SEASONS]:
            for measure, ac, pc in specs:
                rows.append({"season": season_label, "group": group_name, "measure": measure, **metric(g[ac], g[pc])})
    return pd.DataFrame(rows)


def displacement_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [("WR", "TE"), ("WR", "RB_FB"), ("TE", "RB_FB")]
    measures = {
        "target_share": "target_share_residual_actual_minus_proj",
        "receptions": "receptions_residual_actual_minus_proj",
        "rec_yards": "rec_yards_residual_actual_minus_proj",
    }
    for label, col in measures.items():
        wide = ledger.pivot_table(index=["season", "week", "team"], columns="group", values=col, aggfunc="first").reset_index()
        for left, right in pairs:
            for season_label, g in [("POOLED", wide)] + [(str(s), wide.loc[wide.season.eq(s)]) for s in SEASONS]:
                z = g[[left, right]].dropna() if left in g.columns and right in g.columns else pd.DataFrame()
                if z.empty:
                    pearson = spearman = same = np.nan
                    n = 0
                else:
                    pearson = corr(z[left], z[right], "pearson")
                    spearman = corr(z[left], z[right], "spearman")
                    nz = z[left].ne(0) & z[right].ne(0)
                    same = float((np.sign(z.loc[nz, left]) == np.sign(z.loc[nz, right])).mean()) if nz.any() else np.nan
                    n = len(z)
                rows.append({
                    "season": season_label,
                    "measure": label,
                    "left_group": left,
                    "right_group": right,
                    "n": int(n),
                    "pearson": pearson,
                    "spearman": spearman,
                    "same_sign_rate": same,
                })
    return pd.DataFrame(rows)


def player_baseline(proj: pd.DataFrame, actual: pd.DataFrame, position_group: str, include_rushing: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = proj.loc[proj.position_group.eq(position_group)].copy()
    a = actual.loc[actual.position_group.eq(position_group)].copy()
    pkeep = ["season", "week", "team", "join_key", "player", "player_clean_key", "det_expected_targets", "proj_receptions", "proj_rec_yards"]
    if include_rushing:
        pkeep += ["proj_rush_yards", "proj_rush_rec_yards"]
    akeep = ["season", "week", "team", "join_key", "player", "player_clean_key", "targets", "receptions", "rec_yards"]
    if include_rushing:
        akeep += ["rush_yards", "rush_rec_yards"]
    p = p[pkeep].rename(columns={"player": "proj_player", "player_clean_key": "proj_player_clean_key"})
    a = a[akeep].rename(columns={"player": "actual_player", "player_clean_key": "actual_player_clean_key"})
    x = p.merge(a, on=["season", "week", "team", "join_key"], how="outer", indicator=True, validate="one_to_one")
    pairs = [("targets", "targets", "det_expected_targets"), ("receptions", "receptions", "proj_receptions"), ("rec_yards", "rec_yards", "proj_rec_yards")]
    if include_rushing:
        pairs += [("rush_yards", "rush_yards", "proj_rush_yards"), ("rush_rec_yards", "rush_rec_yards", "proj_rush_rec_yards")]
    for _, ac, pc in pairs:
        x[ac] = num(x[ac]).fillna(0.0)
        x[pc] = num(x[pc]).fillna(0.0)
    summaries = []
    for season_label, g in [("POOLED", x)] + [(str(s), x.loc[x.season.eq(s)]) for s in SEASONS]:
        for market, ac, pc in pairs:
            summaries.append({"season": season_label, "position_group": position_group, "market": market, **metric(g[ac], g[pc])})
    return x, pd.DataFrame(summaries)


def historical_accounting(actual: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = actual.groupby(["season", "week", "team"], as_index=False).agg(
        all_passer_pass_yards=("pass_yards", "sum"),
        all_receiver_rec_yards=("rec_yards", "sum"),
        team_pass_attempts=("pass_att", "sum"),
        team_targets=("targets", "sum"),
    )
    x = x.loc[(x.team_pass_attempts > 0) | (x.team_targets > 0)].copy()
    x["signed_gap_pass_minus_rec"] = x.all_passer_pass_yards - x.all_receiver_rec_yards
    x["abs_gap"] = x.signed_gap_pass_minus_rec.abs()
    out = {
        "n": int(len(x)),
        "mae": float(x.abs_gap.mean()) if len(x) else np.nan,
        "median_abs_gap": float(x.abs_gap.median()) if len(x) else np.nan,
        "p95_abs_gap": float(x.abs_gap.quantile(.95)) if len(x) else np.nan,
        "max_abs_gap": float(x.abs_gap.max()) if len(x) else np.nan,
        "rows_abs_gap_gt_1": int(x.abs_gap.gt(1.0).sum()),
    }
    return x, out


def promoted_conservation(proj: pd.DataFrame, m89_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    qb = pd.read_csv(one(m89_root, "m89_2024_2025_synthesis_trace.csv"), low_memory=False)
    qb.columns = [str(c).strip().lower() for c in qb.columns]
    qb = qb.loc[num(qb.season).isin([2024, 2025])].copy()
    qb["season"] = num(qb.season).astype(int)
    qb["week"] = num(qb.week).astype(int)
    qb["team"] = qb.team.map(team)
    if qb.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate M89 team-week rows")
    rec = (
        proj.loc[proj.position_group.isin(PASS_GROUPS)]
        .groupby(["season", "week", "team"], as_index=False)
        .agg(projected_modeled_receiver_yards=("proj_rec_yards", "sum"))
    )
    x = qb.merge(rec, on=["season", "week", "team"], how="inner", validate="one_to_one")
    x["football_synthesis"] = num(x["football_synthesis"])
    x["promoted_qb_minus_receiver_sum"] = x.football_synthesis - x.projected_modeled_receiver_yards
    x["abs_gap"] = x.promoted_qb_minus_receiver_sum.abs()
    summaries = []
    for season_label, g in [("POOLED", x), ("2024", x.loc[x.season.eq(2024)]), ("2025", x.loc[x.season.eq(2025)])]:
        summaries.append({
            "season": season_label,
            "n": int(len(g)),
            "mean_signed_gap": float(g.promoted_qb_minus_receiver_sum.mean()) if len(g) else np.nan,
            "mae": float(g.abs_gap.mean()) if len(g) else np.nan,
            "median_abs_gap": float(g.abs_gap.median()) if len(g) else np.nan,
            "p90_abs_gap": float(g.abs_gap.quantile(.90)) if len(g) else np.nan,
            "p95_abs_gap": float(g.abs_gap.quantile(.95)) if len(g) else np.nan,
            "pct_gt_10": float(g.abs_gap.gt(10).mean()) if len(g) else np.nan,
            "pct_gt_25": float(g.abs_gap.gt(25).mean()) if len(g) else np.nan,
            "pct_gt_50": float(g.abs_gap.gt(50).mean()) if len(g) else np.nan,
            "correlation": corr(g.football_synthesis, g.projected_modeled_receiver_yards, "pearson"),
        })
    pooled = summaries[0]
    gate = {
        "median_abs_gap_ge_10": bool(pooled["median_abs_gap"] >= 10.0),
        "p90_abs_gap_ge_25": bool(pooled["p90_abs_gap"] >= 25.0),
    }
    return x, pd.DataFrame(summaries), gate


def qb_ecosystem_coupling(ledger: pd.DataFrame, m89_root: Path) -> pd.DataFrame:
    qb = pd.read_csv(one(m89_root, "m89_2024_2025_synthesis_trace.csv"), low_memory=False)
    qb.columns = [str(c).strip().lower() for c in qb.columns]
    qb = qb.loc[num(qb.season).isin([2024, 2025])].copy()
    qb["season"] = num(qb.season).astype(int)
    qb["week"] = num(qb.week).astype(int)
    qb["team"] = qb.team.map(team)
    qb["qb_residual"] = num(qb.actual_pass_yards) - num(qb.football_synthesis)
    wide = ledger.loc[ledger.group.isin(["ALL", "WR", "TE", "RB_FB"])].pivot_table(
        index=["season", "week", "team"], columns="group", values="rec_yards_residual_actual_minus_proj", aggfunc="first"
    ).reset_index()
    x = qb.merge(wide, on=["season", "week", "team"], how="inner", validate="one_to_one")
    rows = []
    for group_name in ["ALL", "WR", "TE", "RB_FB"]:
        if group_name not in x.columns:
            continue
        for season_label, g in [("POOLED", x)] + [(str(s), x.loc[x.season.eq(s)]) for s in [2024, 2025]]:
            z = g[["qb_residual", group_name]].dropna()
            rows.append({
                "season": season_label,
                "receiver_group": group_name,
                "n": int(len(z)),
                "pearson": corr(z[group_name], z.qb_residual, "pearson"),
                "spearman": corr(z[group_name], z.qb_residual, "spearman"),
            })
    return pd.DataFrame(rows)


def sign_consistency(summary: pd.DataFrame, group_name: str) -> dict:
    x = summary.loc[(summary.group.eq(group_name)) & (summary.measure.eq("target_share"))].copy()
    pooled = x.loc[x.season.eq("POOLED")]
    pooled_bias = float(pooled.bias.iloc[0]) if len(pooled) else np.nan
    sign = np.sign(pooled_bias) if np.isfinite(pooled_bias) and pooled_bias != 0 else 0
    annual = x.loc[x.season.isin([str(s) for s in SEASONS])]
    same = int((np.sign(num(annual.bias)) == sign).sum()) if sign != 0 else 0
    return {"pooled_bias": pooled_bias, "same_sign_seasons": same}


def displacement_gate(displacement: pd.DataFrame, right_group: str) -> dict:
    x = displacement.loc[
        displacement.measure.eq("target_share")
        & displacement.left_group.eq("WR")
        & displacement.right_group.eq(right_group)
    ].copy()
    pooled = x.loc[x.season.eq("POOLED")]
    pearson = float(pooled.pearson.iloc[0]) if len(pooled) else np.nan
    annual = x.loc[x.season.isin([str(s) for s in SEASONS])]
    negative_seasons = int((num(annual.pearson) < 0).sum())
    return {"pooled_pearson": pearson, "negative_seasons": negative_seasons}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--m89-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    projections = pd.concat([load_season(args.root, s, "receiving_ecosystem_projection_trace.csv") for s in SEASONS], ignore_index=True)
    actuals = pd.concat([load_season(args.root, s, "receiving_ecosystem_actual_usage.csv") for s in SEASONS], ignore_index=True)
    raw_cons = pd.concat([load_season(args.root, s, "raw_sim_pass_receiving_conservation.csv") for s in SEASONS], ignore_index=True)

    ledger = make_team_ledger(projections, actuals)
    mass_summary = team_metric_summary(ledger)
    displacement = displacement_summary(ledger)
    rb_casebook, rb_summary = player_baseline(projections, actuals, "RB", include_rushing=True)
    te_casebook, te_summary = player_baseline(projections, actuals, "TE", include_rushing=False)
    accounting_casebook, accounting = historical_accounting(actuals)
    promoted_casebook, promoted_summary, conservation_gate = promoted_conservation(projections, args.m89_root)
    coupling = qb_ecosystem_coupling(ledger, args.m89_root)

    te_bias = sign_consistency(mass_summary, "TE")
    rbfb_bias = sign_consistency(mass_summary, "RB_FB")
    wr_te = displacement_gate(displacement, "TE")
    wr_rbfb = displacement_gate(displacement, "RB_FB")

    mass_gate_details = {
        "te_abs_bias_ge_1_5pp_and_same_sign_4of6": bool(abs(te_bias["pooled_bias"]) >= .015 and te_bias["same_sign_seasons"] >= 4),
        "rbfb_abs_bias_ge_1_5pp_and_same_sign_4of6": bool(abs(rbfb_bias["pooled_bias"]) >= .015 and rbfb_bias["same_sign_seasons"] >= 4),
        "wr_te_pearson_le_neg_0_15_and_negative_4of6": bool(wr_te["pooled_pearson"] <= -.15 and wr_te["negative_seasons"] >= 4),
        "wr_rbfb_pearson_le_neg_0_15_and_negative_4of6": bool(wr_rbfb["pooled_pearson"] <= -.15 and wr_rbfb["negative_seasons"] >= 4),
    }
    position_disposition = (
        "POSITION_GROUP_MISALLOCATION_SUPPORTED"
        if any(mass_gate_details.values())
        else "NO_MATERIAL_POSITION_GROUP_MASS_MISALLOCATION"
    )
    conservation_disposition = (
        "MATERIAL_PASS_RECEIVING_CONSERVATION_GAP"
        if any(conservation_gate.values())
        else "CONSERVATION_GAP_SMALL_AT_CURRENT_RESOLUTION"
    )
    integrity_ok = bool(np.isfinite(accounting["mae"]) and accounting["mae"] <= 1.0)

    raw_summary = {
        "n_team_games": int(len(raw_cons)),
        "mean_team_game_raw_sim_mae": float(num(raw_cons.mean_abs_gap).mean()) if len(raw_cons) else np.nan,
        "median_team_game_raw_sim_median_abs_gap": float(num(raw_cons.median_abs_gap).median()) if len(raw_cons) else np.nan,
        "p90_team_game_raw_sim_p90_abs_gap": float(num(raw_cons.p90_abs_gap).quantile(.90)) if len(raw_cons) else np.nan,
    }

    result = {
        "migration": "RECEIVING_ECOSYSTEM_CONSERVATION_AUDIT",
        "seasons": SEASONS,
        "projection_rows": int(len(projections)),
        "actual_usage_rows": int(len(actuals)),
        "team_group_ledger_rows": int(len(ledger)),
        "rb_casebook_rows": int(len(rb_casebook)),
        "te_casebook_rows": int(len(te_casebook)),
        "historical_accounting": accounting,
        "historical_accounting_gate_mae_le_1": integrity_ok,
        "target_share_bias": {"TE": te_bias, "RB_FB": rbfb_bias},
        "target_share_displacement": {"WR_TE": wr_te, "WR_RB_FB": wr_rbfb},
        "position_group_gates": mass_gate_details,
        "position_group_disposition": position_disposition if integrity_ok else "INTEGRITY_FAILURE_NO_SCIENTIFIC_DISPOSITION",
        "promoted_qb_receiver_conservation_gates": conservation_gate,
        "promoted_qb_receiver_conservation_disposition": conservation_disposition if integrity_ok else "INTEGRITY_FAILURE_NO_SCIENTIFIC_DISPOSITION",
        "raw_sim_conservation_diagnostic": raw_summary,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(args.out_dir / "receiving_position_group_team_ledger.csv", index=False)
    mass_summary.to_csv(args.out_dir / "receiving_position_group_metric_summary.csv", index=False)
    displacement.to_csv(args.out_dir / "receiving_cross_position_displacement.csv", index=False)
    rb_casebook.to_csv(args.out_dir / "rb_receiving_player_baseline_casebook.csv", index=False)
    rb_summary.to_csv(args.out_dir / "rb_receiving_player_baseline_summary.csv", index=False)
    te_casebook.to_csv(args.out_dir / "te_receiving_player_baseline_casebook.csv", index=False)
    te_summary.to_csv(args.out_dir / "te_receiving_player_baseline_summary.csv", index=False)
    accounting_casebook.to_csv(args.out_dir / "historical_pass_receiving_accounting.csv", index=False)
    promoted_casebook.to_csv(args.out_dir / "promoted_qb_receiver_conservation_casebook.csv", index=False)
    promoted_summary.to_csv(args.out_dir / "promoted_qb_receiver_conservation_summary.csv", index=False)
    coupling.to_csv(args.out_dir / "qb_full_receiving_ecosystem_residual_coupling.csv", index=False)
    raw_cons.to_csv(args.out_dir / "raw_sim_pass_receiving_conservation_all_seasons.csv", index=False)
    (args.out_dir / "receiving_ecosystem_conservation_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    print("\n--- RB baseline ---")
    print(rb_summary.to_string(index=False))
    print("\n--- TE baseline ---")
    print(te_summary.to_string(index=False))
    print("\n--- promoted conservation ---")
    print(promoted_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
