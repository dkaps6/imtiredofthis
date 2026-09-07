#!/usr/bin/env python3
"""Evaluate the frozen B0/C1/C2/C3 Joint Pass/Receiving Conservation V1 experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
POSITIONS = ["WR", "TE", "RB"]
GROUPS = ["WR", "TE", "RB_FB"]
VARIANTS = ["b0", "c1", "c2", "c3"]
EXPECTED_2025_ALL_REC_N = 4647
EXPECTED_2025_ALL_REC_MAE = 17.099904733366


def num(s): return pd.to_numeric(s, errors="coerce")


def metric(actual, projected) -> dict:
    z = pd.DataFrame({"a": num(actual), "p": num(projected)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan}
    e = z.p - z.a
    return {
        "n": int(len(z)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "bias": float(e.mean()),
        "correlation": float(z.p.corr(z.a)) if len(z) > 2 else np.nan,
    }


def load(root: Path, season: int, name: str) -> pd.DataFrame:
    p = root / str(season) / "trace" / name
    if not p.exists() or not p.stat().st_size:
        raise RuntimeError(f"missing {p}")
    x = pd.read_csv(p, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if "season" in x.columns: x["season"] = num(x.season).astype(int)
    if "week" in x.columns: x["week"] = num(x.week).astype(int)
    return x


def player_casebook(root: Path) -> pd.DataFrame:
    frames = []
    for s in SEASONS:
        p = load(root, s, "joint_v1_player_projection_trace.csv")
        a = load(root, s, "joint_v1_actual_usage.csv")
        p["team"] = p.team.astype(str)
        a["team"] = a.team.astype(str)
        keep = ["season", "week", "team", "join_key", "targets", "receptions", "rec_yards", "rush_yards", "rush_rec_yards"]
        x = p.merge(a[keep], on=["season", "week", "team", "join_key"], how="inner", validate="one_to_one")
        frames.append(x)
    return pd.concat(frames, ignore_index=True, sort=False)


def player_metrics(casebook: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = {
        "targets": "expected_targets",
        "receptions": "receptions",
        "rec_yards": "rec_yards",
        "rush_yards": "rush_yards",
        "rush_rec_yards": "rush_rec_yards",
    }
    for pos in POSITIONS:
        base = casebook.loc[casebook.position_group.astype(str).str.upper().eq(pos)].copy()
        for slabel, g in [("POOLED", base)] + [(str(s), base.loc[base.season.eq(s)]) for s in SEASONS] + [("2024_2025", base.loc[base.season.isin([2024, 2025])])]:
            for v in VARIANTS:
                for market, suffix in specs.items():
                    if market in {"rush_yards", "rush_rec_yards"} and pos != "RB":
                        continue
                    col = f"{v}_{suffix}"
                    if col not in g.columns:
                        continue
                    rows.append({"season": slabel, "position_group": pos, "variant": v.upper(), "market": market, **metric(g[market], g[col])})
    return pd.DataFrame(rows)


def group_casebook(root: Path) -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        p = load(root, s, "joint_v1_player_projection_trace.csv")
        a = load(root, s, "joint_v1_actual_usage.csv")
        # Actual composition denominator is WR+TE+RB+FB only, frozen in implementation spec.
        aa = a.loc[a.mass_group.astype(str).isin(GROUPS)].groupby(["season", "week", "team", "mass_group"], as_index=False).targets.sum()
        at = aa.groupby(["season", "week", "team"], as_index=False).targets.sum().rename(columns={"targets": "actual_modeled_group_targets"})
        aa = aa.merge(at, on=["season", "week", "team"], how="left")
        aa["actual_group_share"] = np.where(aa.actual_modeled_group_targets > 0, aa.targets / aa.actual_modeled_group_targets, np.nan)
        pp = p.loc[p.mass_group.astype(str).isin(GROUPS)].copy()
        for v in ["b0", "c1", "c3"]:
            g = pp.groupby(["season", "week", "team", "mass_group"], as_index=False)[f"{v}_target_probability"].sum()
            den = g.groupby(["season", "week", "team"], as_index=False)[f"{v}_target_probability"].sum().rename(columns={f"{v}_target_probability": "den"})
            g = g.merge(den, on=["season", "week", "team"], how="left")
            g[f"{v}_group_share"] = np.where(g.den > 0, g[f"{v}_target_probability"] / g.den, np.nan)
            g = g.drop(columns=[f"{v}_target_probability", "den"])
            aa = aa.merge(g, on=["season", "week", "team", "mass_group"], how="outer")
        rows.append(aa)
    return pd.concat(rows, ignore_index=True, sort=False)


def group_metrics(gcase: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        base = gcase.loc[gcase.mass_group.eq(group)].copy()
        slices = [("POOLED", base)] + [(str(s), base.loc[base.season.eq(s)]) for s in SEASONS] + [("2024_2025", base.loc[base.season.isin([2024, 2025])])]
        for slabel, g in slices:
            for v in ["b0", "c1", "c3"]:
                rows.append({"season": slabel, "group": group, "variant": v.upper(), **metric(g.actual_group_share, g[f"{v}_group_share"])})
    return pd.DataFrame(rows)


def macro_group_mae(gmetrics: pd.DataFrame, season: str, variant: str) -> float:
    g = gmetrics.loc[(gmetrics.season.astype(str).eq(str(season))) & (gmetrics.variant.eq(variant)) & (gmetrics.group.isin(GROUPS))]
    return float(g.mae.mean()) if len(g) == len(GROUPS) else np.nan


def get_mae(pmetrics: pd.DataFrame, season: str, pos: str, variant: str, market: str) -> float:
    g = pmetrics.loc[(pmetrics.season.astype(str).eq(str(season))) & pmetrics.position_group.eq(pos) & pmetrics.variant.eq(variant) & pmetrics.market.eq(market)]
    return float(g.iloc[0].mae) if len(g) == 1 else np.nan


def qb_casebook(root: Path) -> pd.DataFrame:
    return pd.concat([load(root, s, "joint_v1_qb_distribution_trace.csv") for s in [2024, 2025]], ignore_index=True, sort=False)


def bootstrap_prob_improve(diff: np.ndarray, seed: int = 5601, nboot: int = 10000) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if not len(d): return np.nan
    rng = np.random.default_rng(seed)
    wins = 0
    # Chunked to avoid a large nboot x n matrix.
    for _ in range(nboot):
        idx = rng.integers(0, len(d), len(d))
        wins += int(float(np.mean(d[idx])) < 0.0)
    return float(wins / nboot)


def qb_summary(qb: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows = []
    extra = {}
    anchor = metric(qb.actual_pass_yards, qb.football_synthesis)
    for v in ["b0", "c2", "c3"]:
        mean_col = f"{v}_mean"
        m = metric(qb.actual_pass_yards, qb[mean_col])
        crps = num(qb[f"{v}_crps"])
        cover50 = float(num(qb[f"{v}_cover50"]).mean())
        cover80 = float(num(qb[f"{v}_cover80"]).mean())
        cover90 = float(num(qb[f"{v}_cover90"]).mean())
        miss100 = int((num(qb[mean_col]) - num(qb.actual_pass_yards)).abs().ge(100.0).sum())
        rows.append({
            "variant": v.upper(), **m,
            "mean_crps": float(crps.mean()),
            "coverage50": cover50, "coverage80": cover80, "coverage90": cover90,
            "coverage50_abs_error": abs(cover50 - 0.50),
            "coverage80_abs_error": abs(cover80 - 0.80),
            "coverage90_abs_error": abs(cover90 - 0.90),
            "miss100_count": miss100,
            "max_anchor_mean_abs_diff": float((num(qb[mean_col]) - num(qb.football_synthesis)).abs().max()),
        })
    out = pd.DataFrame(rows)
    for v in ["c2", "c3"]:
        diff = num(qb[f"{v}_crps"]) - num(qb["b0_crps"])
        extra[v.upper()] = {
            "mean_crps_delta_vs_b0": float(diff.mean()),
            "bootstrap_prob_crps_improves": bootstrap_prob_improve(diff.to_numpy()),
            "qb_mean_mae_delta_vs_anchor": float(metric(qb.actual_pass_yards, qb[f"{v}_mean"])["mae"] - anchor["mae"]),
        }
    return out, extra


def conservation_summary(root: Path) -> pd.DataFrame:
    x = pd.concat([load(root, s, "joint_v1_conservation_trace.csv") for s in SEASONS], ignore_index=True, sort=False)
    rows = []
    for v in ["C2", "C3"]:
        g = x.loc[x.variant.eq(v)]
        rows.append({
            "variant": v, "n": int(len(g)),
            "max_abs_gap": float(num(g.max_abs_gap).max()),
            "mean_abs_gap": float(num(g.mean_abs_gap).mean()),
            "median_abs_gap": float(num(g.median_abs_gap).median()),
            "p90_abs_gap": float(num(g.p90_abs_gap).quantile(.90)),
            "p95_abs_gap": float(num(g.p95_abs_gap).quantile(.95)),
            "pct_gap_gt_0_01": float(num(g.pct_gap_gt_0_01).mean()),
            "pct_gap_gt_1": float(num(g.pct_gap_gt_1).mean()),
            "pct_gap_gt_10": float(num(g.pct_gap_gt_10).mean()),
        })
    return pd.DataFrame(rows)


def integrity_counts(root: Path) -> dict:
    x = pd.concat([load(root, s, "joint_v1_integrity_counts.csv") for s in SEASONS], ignore_index=True)
    return {
        "c2_zero_rec_positive_yards": int(num(x.c2_zero_rec_positive_yards).sum()),
        "c3_zero_rec_positive_yards": int(num(x.c3_zero_rec_positive_yards).sum()),
    }


def group_gate(gm: pd.DataFrame, pm: pd.DataFrame, variant: str) -> tuple[dict, bool]:
    b_macro = macro_group_mae(gm, "POOLED", "B0")
    v_macro = macro_group_mae(gm, "POOLED", variant)
    group_deltas = {}
    for g in GROUPS:
        b = float(gm.loc[(gm.season.eq("POOLED")) & gm.group.eq(g) & gm.variant.eq("B0"), "mae"].iloc[0])
        c = float(gm.loc[(gm.season.eq("POOLED")) & gm.group.eq(g) & gm.variant.eq(variant), "mae"].iloc[0])
        group_deltas[g] = c - b
    year_improve = sum(macro_group_mae(gm, str(s), variant) < macro_group_mae(gm, str(s), "B0") for s in SEASONS)
    player_target_ok = all(get_mae(pm, "POOLED", p, variant, "targets") - get_mae(pm, "POOLED", p, "B0", "targets") <= .03 for p in POSITIONS)
    player_rec_ok = all(get_mae(pm, "POOLED", p, variant, "receptions") - get_mae(pm, "POOLED", p, "B0", "receptions") <= .03 for p in POSITIONS)
    player_yard_ok = all(get_mae(pm, "POOLED", p, variant, "rec_yards") - get_mae(pm, "POOLED", p, "B0", "rec_yards") <= .50 for p in POSITIONS)
    gates = {
        "macro_target_share_mae_improvement_ge_0_005": bool(b_macro - v_macro >= .005),
        "no_group_target_share_mae_regression_gt_0_0025": bool(max(group_deltas.values()) <= .0025),
        "macro_target_share_improves_ge4_of6_seasons": bool(year_improve >= 4),
        "latest_2024_2025_macro_target_share_improves": bool(macro_group_mae(gm, "2024_2025", variant) < macro_group_mae(gm, "2024_2025", "B0")),
        "player_target_mae_nonworse_0_03_all_positions": bool(player_target_ok),
        "player_reception_mae_nonworse_0_03_all_positions": bool(player_rec_ok),
        "player_rec_yard_mae_nonworse_0_50_all_positions": bool(player_yard_ok),
    }
    gates["yearly_macro_improve_count"] = int(year_improve)
    gates["pooled_macro_b0"] = b_macro
    gates["pooled_macro_candidate"] = v_macro
    gates["group_mae_deltas"] = group_deltas
    return gates, all(bool(v) for k, v in gates.items() if k not in {"yearly_macro_improve_count", "pooled_macro_b0", "pooled_macro_candidate", "group_mae_deltas"})


def conservation_gate(pm: pd.DataFrame, qsum: pd.DataFrame, qextra: dict, csum: pd.DataFrame, integ: dict, variant: str) -> tuple[dict, bool]:
    row = qsum.loc[qsum.variant.eq(variant)].iloc[0]
    b0 = qsum.loc[qsum.variant.eq("B0")].iloc[0]
    cr = csum.loc[csum.variant.eq(variant)].iloc[0]
    vlow = variant.lower()
    pos_yard_ok = all(get_mae(pm, "POOLED", p, variant, "rec_yards") - get_mae(pm, "POOLED", p, "B0", "rec_yards") <= .50 for p in POSITIONS)
    bmacro = float(np.mean([get_mae(pm, "POOLED", p, "B0", "rec_yards") for p in POSITIONS]))
    vmacro = float(np.mean([get_mae(pm, "POOLED", p, variant, "rec_yards") for p in POSITIONS]))
    zero_key = f"{vlow}_zero_rec_positive_yards"
    gates = {
        "iteration_max_accounting_gap_le_1e_6": bool(float(cr.max_abs_gap) <= 1e-6),
        "max_qb_mean_anchor_diff_le_0_01": bool(float(row.max_anchor_mean_abs_diff) <= .01),
        "qb_mean_mae_anchor_delta_abs_le_0_01": bool(abs(float(qextra[variant]["qb_mean_mae_delta_vs_anchor"])) <= .01),
        "mean_qb_crps_improves_ge_0_25": bool(float(qextra[variant]["mean_crps_delta_vs_b0"]) <= -.25),
        "bootstrap_prob_crps_improves_ge_0_90": bool(float(qextra[variant]["bootstrap_prob_crps_improves"]) >= .90),
        "coverage80_abs_error_nonworse_0_02": bool(float(row.coverage80_abs_error) <= float(b0.coverage80_abs_error) + .02),
        "player_rec_yard_mae_nonworse_0_50_all_positions": bool(pos_yard_ok),
        "macro_player_rec_yard_mae_nonworse": bool(vmacro <= bmacro),
        "zero_reception_positive_yards_zero": bool(int(integ[zero_key]) == 0),
    }
    gates["macro_b0_rec_yard_mae"] = bmacro
    gates["macro_candidate_rec_yard_mae"] = vmacro
    return gates, all(bool(v) for k, v in gates.items() if k not in {"macro_b0_rec_yard_mae", "macro_candidate_rec_yard_mae"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    case = player_casebook(a.root)
    pm = player_metrics(case)
    gc = group_casebook(a.root)
    gm = group_metrics(gc)
    qb = qb_casebook(a.root)
    qsum, qextra = qb_summary(qb)
    csum = conservation_summary(a.root)
    integ = integrity_counts(a.root)

    # Gate 0 exact current-baseline parity on the inner matched receiving cohort.
    p25 = case.loc[case.season.eq(2025)].copy()
    parity = metric(p25.rec_yards, p25.b0_rec_yards)
    parity_ok = parity["n"] == EXPECTED_2025_ALL_REC_N and abs(parity["mae"] - EXPECTED_2025_ALL_REC_MAE) <= .05
    qb_unique = not qb.duplicated(["season", "week", "team"]).any()
    baseline_gate = {"2025_all_receiver_n_exact": bool(parity["n"] == EXPECTED_2025_ALL_REC_N), "2025_all_receiver_mae_within_0_05": bool(abs(parity["mae"] - EXPECTED_2025_ALL_REC_MAE) <= .05), "m89_team_week_unique": bool(qb_unique)}
    baseline_ok = parity_ok and qb_unique

    c1_gates, c1_pass = group_gate(gm, pm, "C1")
    c2_gates, c2_pass = conservation_gate(pm, qsum, qextra, csum, integ, "C2")
    c3_group_gates, c3_group_pass = group_gate(gm, pm, "C3")
    c3_cons_gates, c3_cons_pass = conservation_gate(pm, qsum, qextra, csum, integ, "C3")

    bmacro = float(np.mean([get_mae(pm, "POOLED", p, "B0", "rec_yards") for p in POSITIONS]))
    c3macro = float(np.mean([get_mae(pm, "POOLED", p, "C3", "rec_yards") for p in POSITIONS]))
    rb_rr_delta = get_mae(pm, "POOLED", "RB", "C3", "rush_rec_yards") - get_mae(pm, "POOLED", "RB", "B0", "rush_rec_yards")
    season_pos_regressions = []
    for s in SEASONS:
        for p in POSITIONS:
            season_pos_regressions.append(get_mae(pm, str(s), p, "C3", "rec_yards") - get_mae(pm, str(s), p, "B0", "rec_yards"))
    latest_b = float(np.mean([get_mae(pm, "2024_2025", p, "B0", "rec_yards") for p in POSITIONS]))
    latest_c3 = float(np.mean([get_mae(pm, "2024_2025", p, "C3", "rec_yards") for p in POSITIONS]))
    joint_extra = {
        "c3_satisfies_group_gate": bool(c3_group_pass),
        "c3_satisfies_conservation_gate": bool(c3_cons_pass),
        "macro_rec_yard_mae_improves_ge_0_25": bool(bmacro - c3macro >= .25),
        "rb_rush_rec_mae_nonworse_0_50": bool(rb_rr_delta <= .50),
        "no_season_position_rec_yard_regression_gt_1_50": bool(max(season_pos_regressions) <= 1.50),
        "latest_2024_2025_macro_rec_yard_mae_improves": bool(latest_c3 < latest_b),
    }
    joint_pass = all(joint_extra.values())

    if not baseline_ok:
        disposition = "BASELINE_PARITY_FAIL"
    elif joint_pass:
        disposition = "JOINT_ARCHITECTURE_CANDIDATE_PASS"
    elif c1_pass and c2_pass:
        disposition = "BOTH_COMPONENTS_PASS_JOINT_FAIL"
    elif c1_pass:
        disposition = "GROUP_MASS_ONLY_SUPPORTED"
    elif c2_pass:
        disposition = "CONSERVATION_ONLY_SUPPORTED"
    else:
        disposition = "NO_V1_CANDIDATE_PASS"

    result = {
        "migration": "JOINT_PASS_RECEIVING_CONSERVATION_V1",
        "target_seasons": SEASONS,
        "qb_distribution_seasons": [2024, 2025],
        "iterations": 2000,
        "sportsbook_inputs_used": False,
        "production_changed": False,
        "baseline_parity": parity,
        "baseline_gates": baseline_gate,
        "c1_group_gates": c1_gates,
        "c1_group_pass": bool(c1_pass and baseline_ok),
        "c2_conservation_gates": c2_gates,
        "c2_conservation_pass": bool(c2_pass and baseline_ok),
        "c3_group_gates": c3_group_gates,
        "c3_conservation_gates": c3_cons_gates,
        "c3_joint_extra_gates": joint_extra,
        "c3_joint_pass": bool(joint_pass and baseline_ok),
        "qb_extra": qextra,
        "integrity_counts": integ,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    case.to_csv(a.out_dir / "joint_v1_paired_player_casebook.csv", index=False)
    pm.to_csv(a.out_dir / "joint_v1_player_metric_summary.csv", index=False)
    gc.to_csv(a.out_dir / "joint_v1_group_casebook.csv", index=False)
    gm.to_csv(a.out_dir / "joint_v1_group_metric_summary.csv", index=False)
    qb.to_csv(a.out_dir / "joint_v1_qb_casebook.csv", index=False)
    qsum.to_csv(a.out_dir / "joint_v1_qb_distribution_summary.csv", index=False)
    csum.to_csv(a.out_dir / "joint_v1_conservation_summary.csv", index=False)
    (a.out_dir / "joint_v1_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("--- QB summary ---")
    print(qsum.to_string(index=False))
    print("--- group summary pooled ---")
    print(gm.loc[gm.season.eq("POOLED")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
