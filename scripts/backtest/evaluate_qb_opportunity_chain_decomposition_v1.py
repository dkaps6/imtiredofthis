#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_M89_ROWS = 884
EXPECTED_PRIMARY_ROWS = 440
EXPECTED_SECONDARY_ROWS = 884
TOL = 1e-6
CORR_TOL = 1e-9

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
COMPONENTS = ["TEAM_PASS_OPPORTUNITY", "ATTEMPT_CONVERSION", "QB_SHARE"]
FACTOR_KEYS = ["D", "C", "S"]


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def canon_team(s):
    return s.fillna("").astype(str).str.upper().str.strip()


def canon_key(s):
    return s.fillna("").astype(str).str.strip()


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


def shapley_three(pred: dict[str, float], actual: dict[str, float]) -> dict[str, float]:
    contrib = {k: 0.0 for k in FACTOR_KEYS}
    perms = list(itertools.permutations(FACTOR_KEYS))
    for perm in perms:
        state = dict(pred)
        prev = float(state["D"] * state["C"] * state["S"])
        for k in perm:
            state[k] = actual[k]
            cur = float(state["D"] * state["C"] * state["S"])
            contrib[k] += cur - prev
            prev = cur
    return {k: float(v / len(perms)) for k, v in contrib.items()}


def load_m89(m89_root: Path) -> pd.DataFrame:
    common = pd.read_csv(one(m89_root, "m89_corrected_qb_common_trace.csv"), low_memory=False)
    synth = pd.read_csv(one(m89_root, "m89_2024_2025_synthesis_trace.csv"), low_memory=False)
    for d in (common, synth):
        d.columns = [str(c).strip().lower() for c in d.columns]
        d["season"] = num(d["season"])
        d["week"] = num(d["week"])
        d["team"] = canon_team(d["team"])
        d["player_clean_key"] = canon_key(d["player_clean_key"])

    keys = ["season", "week", "team", "player_clean_key"]
    if len(common) != EXPECTED_M89_ROWS or len(synth) != EXPECTED_M89_ROWS:
        raise RuntimeError(f"M89 row drift common={len(common)} synth={len(synth)}")
    if common.duplicated(keys).any() or synth.duplicated(keys).any():
        raise RuntimeError("duplicate M89 QB keys")

    need_common = keys + [
        "actual_pass_yards", "actual_attempts", "pred_attempts", "pred_ypa",
        "actual_qb_attempt_share", "actual_ypa",
    ]
    need_synth = keys + ["base_proj", "football_synthesis"]
    for c in need_common:
        if c not in common.columns:
            raise RuntimeError(f"M89 common trace missing {c}")
    for c in need_synth:
        if c not in synth.columns:
            raise RuntimeError(f"M89 synthesis trace missing {c}")

    x = common[need_common].merge(
        synth[need_synth], on=keys, how="inner", validate="one_to_one"
    )
    if len(x) != EXPECTED_M89_ROWS:
        raise RuntimeError(f"M89 common/synthesis alignment row drift {len(x)}")

    cp_parts = []
    obs_parts = []
    for season in [2024, 2025]:
        cp_hits = list(m89_root.rglob(f"{season}/component_predictions.csv"))
        obs_hits = list(m89_root.rglob(f"{season}/m89_corrected_team_observations.csv"))
        if len(cp_hits) != 1 or len(obs_hits) != 1:
            raise RuntimeError(
                f"expected one season source season={season} cp={len(cp_hits)} obs={len(obs_hits)}"
            )
        cp = pd.read_csv(cp_hits[0], low_memory=False)
        cp.columns = [str(c).strip().lower() for c in cp.columns]
        cp["season"] = num(cp["season"])
        cp["week"] = num(cp["week"])
        cp["team"] = canon_team(cp["team"])
        cp["player_clean_key"] = canon_key(cp["player_clean_key"])
        if "market" not in cp.columns:
            raise RuntimeError("component_predictions missing market")
        cp = cp.loc[
            cp["season"].eq(season) & cp["market"].astype(str).eq("pass_yards")
        ].copy()
        cp_need = keys + [
            "mc_team_expected_dropbacks", "mc_pass_attempts_per_dropback",
            "mc_qb_pass_att_share", "mc_expected_pass_attempts",
        ]
        for c in cp_need:
            if c not in cp.columns:
                raise RuntimeError(f"component_predictions missing {c}")
        if cp.duplicated(keys).any():
            raise RuntimeError(f"duplicate pass_yards component keys season={season}")
        cp_parts.append(cp[cp_need])

        obs = pd.read_csv(obs_hits[0], low_memory=False)
        obs.columns = [str(c).strip().lower() for c in obs.columns]
        obs["season"] = num(obs["season"])
        obs["week"] = num(obs["week"])
        obs["team"] = canon_team(obs["team"])
        obs = obs.loc[obs["season"].eq(season)].copy()
        obs_need = [
            "season", "week", "team", "pass_opportunities", "official_pass_attempts",
            "pbp_sacks", "pbp_qb_scrambles",
        ]
        for c in obs_need:
            if c not in obs.columns:
                raise RuntimeError(f"team observations missing {c}")
        if obs.duplicated(["season", "week", "team"]).any():
            raise RuntimeError(f"duplicate team observation keys season={season}")
        obs_parts.append(obs[obs_need])

    cp = pd.concat(cp_parts, ignore_index=True)
    obs = pd.concat(obs_parts, ignore_index=True)
    x = x.merge(cp, on=keys, how="left", validate="one_to_one")
    x = x.merge(obs, on=["season", "week", "team"], how="left", validate="many_to_one")
    if x[[
        "mc_team_expected_dropbacks", "mc_pass_attempts_per_dropback", "mc_qb_pass_att_share",
        "mc_expected_pass_attempts", "pass_opportunities", "official_pass_attempts",
        "actual_qb_attempt_share"
    ]].isna().any().any():
        raise RuntimeError("missing aligned opportunity factors")

    numeric_cols = [
        "actual_pass_yards", "actual_attempts", "pred_attempts", "pred_ypa",
        "actual_qb_attempt_share", "actual_ypa", "base_proj", "football_synthesis",
        "mc_team_expected_dropbacks", "mc_pass_attempts_per_dropback",
        "mc_qb_pass_att_share", "mc_expected_pass_attempts", "pass_opportunities",
        "official_pass_attempts", "pbp_sacks", "pbp_qb_scrambles",
    ]
    for c in numeric_cols:
        x[c] = num(x[c])
    return x


def add_decomposition(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    z["pred_D"] = z["mc_team_expected_dropbacks"]
    z["pred_C"] = z["mc_pass_attempts_per_dropback"]
    z["pred_S"] = z["mc_qb_pass_att_share"]
    z["actual_D"] = z["pass_opportunities"]
    z["actual_C"] = z["official_pass_attempts"] / z["pass_opportunities"].replace(0, np.nan)
    z["actual_S"] = z["actual_qb_attempt_share"]

    z["pred_attempt_product"] = z.pred_D * z.pred_C * z.pred_S
    z["actual_attempt_product"] = z.actual_D * z.actual_C * z.actual_S
    z["attempt_residual"] = z.actual_attempts - z.pred_attempts

    rows = []
    for r in z.itertuples(index=False):
        pred = {"D": float(r.pred_D), "C": float(r.pred_C), "S": float(r.pred_S)}
        actual = {"D": float(r.actual_D), "C": float(r.actual_C), "S": float(r.actual_S)}
        rows.append(shapley_three(pred, actual))
    sh = pd.DataFrame(rows, index=z.index)
    z["team_pass_opportunity_attempts"] = sh["D"]
    z["attempt_conversion_attempts"] = sh["C"]
    z["qb_share_attempts"] = sh["S"]
    z["shapley_attempt_sum"] = (
        z.team_pass_opportunity_attempts + z.attempt_conversion_attempts + z.qb_share_attempts
    )

    z["avg_ypa"] = (z.actual_ypa + z.pred_ypa) / 2.0
    z["attempt_component"] = (z.actual_attempts - z.pred_attempts) * z.avg_ypa
    z["ypa_component"] = (z.actual_ypa - z.pred_ypa) * (z.actual_attempts + z.pred_attempts) / 2.0
    z["mechanics_proj"] = z.pred_attempts * z.pred_ypa
    z["stack_adjustment"] = z.base_proj - z.mechanics_proj
    z["synthesis_adjustment"] = z.football_synthesis - z.base_proj
    z["final_residual"] = z.actual_pass_yards - z.football_synthesis

    for pfx in ["team_pass_opportunity", "attempt_conversion", "qb_share"]:
        z[f"{pfx}_yards"] = z[f"{pfx}_attempts"] * z.avg_ypa
    z["opportunity_chain_yards"] = (
        z.team_pass_opportunity_yards + z.attempt_conversion_yards + z.qb_share_yards
    )
    z["full_reconstructed_residual"] = (
        z.opportunity_chain_yards + z.ypa_component - z.stack_adjustment - z.synthesis_adjustment
    )

    z["mechanism_state"] = "MIXED"
    z.loc[
        z.attempt_component.abs().ge(1.25 * z.ypa_component.abs()), "mechanism_state"
    ] = "ATTEMPTS_DOMINANT"
    z.loc[
        z.ypa_component.abs().ge(1.25 * z.attempt_component.abs()), "mechanism_state"
    ] = "YPA_DOMINANT"

    abs_cols = [
        "team_pass_opportunity_attempts", "attempt_conversion_attempts", "qb_share_attempts"
    ]
    abs_frame = z[abs_cols].abs()
    labels = np.array(COMPONENTS, dtype=object)
    z["dominant_opportunity_component"] = labels[abs_frame.to_numpy().argmax(axis=1)]
    return z


def summary_rows(x: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("ALL", pd.Series(True, index=x.index)),
        ("ATTEMPTS_DOMINANT", x.mechanism_state.eq("ATTEMPTS_DOMINANT")),
        ("UNDERPROJECTED_ATTEMPTS", x.attempt_residual.lt(0)),
        ("OVERPROJECTED_ATTEMPTS", x.attempt_residual.gt(0)),
        ("ABS_ATTEMPT_MISS_8_PLUS", x.attempt_residual.abs().ge(8)),
        ("ABS_ATTEMPT_MISS_10_PLUS", x.attempt_residual.abs().ge(10)),
    ]
    mapping = {
        "TEAM_PASS_OPPORTUNITY": "team_pass_opportunity_attempts",
        "ATTEMPT_CONVERSION": "attempt_conversion_attempts",
        "QB_SHARE": "qb_share_attempts",
    }
    rows = []
    for season_label, season_mask in [
        ("2024", x.season.eq(2024)),
        ("2025", x.season.eq(2025)),
        ("POOLED_2024_2025", pd.Series(True, index=x.index)),
    ]:
        for cohort, mask in specs:
            g = x.loc[season_mask & mask].copy()
            if g.empty:
                continue
            mean_abs = {c: float(g[col].abs().mean()) for c, col in mapping.items()}
            denom = sum(mean_abs.values())
            for comp, col in mapping.items():
                ss = g[col]
                rows.append({
                    "season": season_label,
                    "cohort": cohort,
                    "component": comp,
                    "n": int(len(g)),
                    "mean_attempt_contribution": float(ss.mean()),
                    "mean_abs_attempt_contribution": mean_abs[comp],
                    "abs_chain_mass_share": float(mean_abs[comp] / denom) if denom > 0 else np.nan,
                    "sign_agreement_with_total_attempt_residual": float(
                        (np.sign(ss) == np.sign(g.attempt_residual)).mean()
                    ),
                    "dominant_row_rate": float(g.dominant_opportunity_component.eq(comp).mean()),
                    "p50_abs_attempt_contribution": float(ss.abs().quantile(0.50)),
                    "p75_abs_attempt_contribution": float(ss.abs().quantile(0.75)),
                    "p90_abs_attempt_contribution": float(ss.abs().quantile(0.90)),
                })
    return pd.DataFrame(rows)


def q4_q1_gap(rank, y):
    z = pd.DataFrame({"rank": num(rank), "y": num(y)}).dropna()
    if len(z) < 8:
        return np.nan
    q1 = float(z["rank"].quantile(0.25))
    q4 = float(z["rank"].quantile(0.75))
    lo = z["rank"].le(q1)
    hi = z["rank"].ge(q4)
    return float(z.loc[hi, "y"].mean() - z.loc[lo, "y"].mean())


def cross_position(chain: pd.DataFrame, shared_root: Path):
    primary = pd.read_csv(one(shared_root, "qb_wr_shared_pass_volume_primary_2025.csv"), low_memory=False)
    secondary = pd.read_csv(one(shared_root, "qb_wr_shared_pass_volume_secondary_2024_2025.csv"), low_memory=False)
    for d in (primary, secondary):
        d.columns = [str(c).strip().lower() for c in d.columns]
        d["season"] = num(d["season"])
        d["week"] = num(d["week"])
        d["team"] = canon_team(d["team"])
        d["player_clean_key"] = canon_key(d["player_clean_key"])

    if len(primary) != EXPECTED_PRIMARY_ROWS or len(secondary) != EXPECTED_SECONDARY_ROWS:
        raise RuntimeError(
            f"shared cohort drift primary={len(primary)} secondary={len(secondary)}"
        )
    keys = ["season", "week", "team", "player_clean_key"]
    if primary.duplicated(keys).any() or secondary.duplicated(keys).any():
        raise RuntimeError("duplicate shared-volume keys")

    keep = keys + [
        "attempt_residual", "team_pass_opportunity_attempts",
        "attempt_conversion_attempts", "qb_share_attempts"
    ]
    p = primary.merge(chain[keep], on=keys, how="left", validate="one_to_one")
    q = secondary.merge(chain[keep], on=keys, how="left", validate="one_to_one")
    if len(p) != EXPECTED_PRIMARY_ROWS or len(q) != EXPECTED_SECONDARY_ROWS:
        raise RuntimeError("shared-volume alignment row drift")
    if p[keep[4:]].isna().any().any() or q[keep[4:]].isna().any().any():
        raise RuntimeError("shared-volume opportunity component alignment missing")

    primary_resid_gap = float((num(p.qb_attempt_residual) - p.attempt_residual).abs().max())
    secondary_resid_gap = float((num(q.qb_attempt_residual) - q.attempt_residual).abs().max())

    pbase = corr_metrics(p.qb_attempt_residual, p.wr_target_mass_residual)
    sbase = corr_metrics(q.qb_attempt_residual, q.wr_reception_mass_residual)

    rows = []
    component_map = {
        "TOTAL_QB_ATTEMPT_RESIDUAL": "qb_attempt_residual",
        "TEAM_PASS_OPPORTUNITY": "team_pass_opportunity_attempts",
        "ATTEMPT_CONVERSION": "attempt_conversion_attempts",
        "QB_SHARE": "qb_share_attempts",
    }
    for label, frame, ycol, season_views in [
        ("PRIMARY_WR_TARGET_MASS", p, "wr_target_mass_residual", [("2025", p)]),
        (
            "SECONDARY_WR_RECEPTION_MASS",
            q,
            "wr_reception_mass_residual",
            [
                ("POOLED_2024_2025", q),
                ("2024", q.loc[q.season.eq(2024)]),
                ("2025", q.loc[q.season.eq(2025)]),
            ],
        ),
    ]:
        for season_label, g in season_views:
            for comp, col in component_map.items():
                met = corr_metrics(g[col], g[ycol])
                rows.append({
                    "view": label,
                    "season": season_label,
                    "component": comp,
                    **met,
                    "signed_component_q4_minus_q1_wr_residual_gap": q4_q1_gap(g[col], g[ycol]),
                })
    return pd.DataFrame(rows), pbase, sbase, primary_resid_gap, secondary_resid_gap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m89-root", type=Path, required=True)
    ap.add_argument("--shared-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    raw = load_m89(a.m89_root)
    x = add_decomposition(raw)
    summaries = summary_rows(x)
    cross, pbase, sbase, p_resid_gap, s_resid_gap = cross_position(x, a.shared_root)

    integrity_values = {
        "m89_rows": int(len(x)),
        "primary_shared_rows": EXPECTED_PRIMARY_ROWS,
        "secondary_shared_rows": EXPECTED_SECONDARY_ROWS,
        "predicted_product_vs_mc_expected_max_abs": float(
            (x.pred_attempt_product - x.mc_expected_pass_attempts).abs().max()
        ),
        "mc_expected_vs_m89_pred_attempts_max_abs": float(
            (x.mc_expected_pass_attempts - x.pred_attempts).abs().max()
        ),
        "realized_product_vs_actual_attempts_max_abs": float(
            (x.actual_attempt_product - x.actual_attempts).abs().max()
        ),
        "shapley_attempt_identity_max_abs": float(
            (x.shapley_attempt_sum - x.attempt_residual).abs().max()
        ),
        "opportunity_chain_yards_vs_attempt_component_max_abs": float(
            (x.opportunity_chain_yards - x.attempt_component).abs().max()
        ),
        "full_final_residual_identity_max_abs": float(
            (x.full_reconstructed_residual - x.final_residual).abs().max()
        ),
        "primary_source_vs_chain_attempt_residual_max_abs": p_resid_gap,
        "secondary_source_vs_chain_attempt_residual_max_abs": s_resid_gap,
        "primary_total_pearson_reproduced": pbase["pearson"],
        "primary_total_spearman_reproduced": pbase["spearman"],
        "primary_total_same_sign_reproduced": pbase["same_sign"],
        "secondary_total_pearson_reproduced": sbase["pearson"],
        "secondary_total_spearman_reproduced": sbase["spearman"],
        "secondary_total_same_sign_reproduced": sbase["same_sign"],
    }

    gates = {
        "m89_rows_exact_884": len(x) == EXPECTED_M89_ROWS,
        "shared_rows_exact_440_and_884": True,
        "zero_sportsbook_inputs": True,
        "no_production_change": True,
        "predicted_attempt_product_reconciles": (
            integrity_values["predicted_product_vs_mc_expected_max_abs"] <= TOL
            and integrity_values["mc_expected_vs_m89_pred_attempts_max_abs"] <= TOL
        ),
        "realized_attempt_product_reconciles": (
            integrity_values["realized_product_vs_actual_attempts_max_abs"] <= TOL
        ),
        "shapley_attempt_identity_reconciles": (
            integrity_values["shapley_attempt_identity_max_abs"] <= TOL
        ),
        "opportunity_chain_yards_reconcile": (
            integrity_values["opportunity_chain_yards_vs_attempt_component_max_abs"] <= TOL
        ),
        "full_final_residual_reconciles": (
            integrity_values["full_final_residual_identity_max_abs"] <= TOL
        ),
        "keys_unique_and_fully_aligned": True,
        "target_outcomes_diagnostic_only": True,
        "shared_total_correlations_reproduce": (
            abs(pbase["pearson"] - PRIMARY_REF["pearson"]) <= CORR_TOL
            and abs(pbase["spearman"] - PRIMARY_REF["spearman"]) <= CORR_TOL
            and abs(pbase["same_sign"] - PRIMARY_REF["same_sign"]) <= CORR_TOL
            and abs(sbase["pearson"] - SECONDARY_REF["pearson"]) <= CORR_TOL
            and abs(sbase["spearman"] - SECONDARY_REF["spearman"]) <= CORR_TOL
            and abs(sbase["same_sign"] - SECONDARY_REF["same_sign"]) <= CORR_TOL
            and p_resid_gap <= TOL
            and s_resid_gap <= TOL
        ),
    }
    all_integrity = all(bool(v) for v in gates.values())

    pooled = summaries.loc[
        summaries.season.eq("POOLED_2024_2025") & summaries.cohort.eq("ALL")
    ].set_index("component")
    yr24 = summaries.loc[summaries.season.eq("2024") & summaries.cohort.eq("ALL")].set_index("component")
    yr25 = summaries.loc[summaries.season.eq("2025") & summaries.cohort.eq("ALL")].set_index("component")
    target = cross.loc[
        cross.view.eq("PRIMARY_WR_TARGET_MASS") & cross.season.eq("2025")
        & cross.component.isin(COMPONENTS)
    ].set_index("component")

    routing = {}
    for comp in COMPONENTS:
        largest_pooled = (
            pooled.loc[comp, "mean_abs_attempt_contribution"]
            == pooled["mean_abs_attempt_contribution"].max()
        )
        max24 = float(yr24["mean_abs_attempt_contribution"].max())
        max25 = float(yr25["mean_abs_attempt_contribution"].max())
        v24 = float(yr24.loc[comp, "mean_abs_attempt_contribution"])
        v25 = float(yr25.loc[comp, "mean_abs_attempt_contribution"])
        largest24 = v24 == max24
        largest25 = v25 == max25
        stable_largest = bool(
            (largest24 and largest25)
            or (largest24 and v25 >= 0.90 * max25)
            or (largest25 and v24 >= 0.90 * max24)
        )
        spearman = abs(float(target.loc[comp, "spearman"]))
        other = [abs(float(target.loc[o, "spearman"])) for o in COMPONENTS if o != comp]
        routing[comp] = {
            "largest_pooled_mean_abs": bool(largest_pooled),
            "season_stability_gate": stable_largest,
            "wr_target_abs_spearman_ge_0_30": bool(spearman >= 0.30),
            "wr_target_abs_spearman_lead_ge_0_10": bool(
                all(spearman >= v + 0.10 for v in other)
            ),
            "pooled_mean_abs_attempt_contribution": float(
                pooled.loc[comp, "mean_abs_attempt_contribution"]
            ),
            "2024_mean_abs_attempt_contribution": v24,
            "2025_mean_abs_attempt_contribution": v25,
            "2025_wr_target_spearman": float(target.loc[comp, "spearman"]),
        }

    qualifying = [
        comp for comp, g in routing.items()
        if all([
            g["largest_pooled_mean_abs"],
            g["season_stability_gate"],
            g["wr_target_abs_spearman_ge_0_30"],
            g["wr_target_abs_spearman_lead_ge_0_10"],
        ])
    ]
    if not all_integrity:
        disposition = "MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE"
    elif len(qualifying) == 1:
        disposition = f"{qualifying[0]}_PRIMARY_DIAGNOSTIC"
    else:
        disposition = "MIXED_OPPORTUNITY_CHAIN_NO_SINGLE_PRIMARY"

    result = {
        "migration": "QB_OPPORTUNITY_CHAIN_DECOMPOSITION_V1",
        "m89_source_run": 33331073376,
        "shared_volume_source_run": 34066549394,
        "rows": int(len(x)),
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "integrity_values": integrity_values,
        "integrity_gates": gates,
        "all_integrity_gates_pass": all_integrity,
        "routing_gates": routing,
        "qualifying_primary_components": qualifying,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "qb_opportunity_chain_casebook.csv", index=False)
    summaries.to_csv(a.out_dir / "qb_opportunity_chain_summary.csv", index=False)
    cross.to_csv(a.out_dir / "qb_opportunity_chain_cross_position.csv", index=False)
    (a.out_dir / "qb_opportunity_chain_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("\nPOOLED ALL-GAME SUMMARY")
    print(
        summaries.loc[
            summaries.season.eq("POOLED_2024_2025") & summaries.cohort.eq("ALL")
        ].to_string(index=False)
    )
    print("\n2025 WR TARGET ATTRIBUTION")
    print(
        cross.loc[
            cross.view.eq("PRIMARY_WR_TARGET_MASS") & cross.season.eq("2025")
        ].to_string(index=False)
    )
    return 0 if all_integrity else 2


if __name__ == "__main__":
    raise SystemExit(main())
