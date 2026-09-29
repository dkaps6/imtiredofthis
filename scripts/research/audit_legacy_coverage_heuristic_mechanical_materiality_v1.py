#!/usr/bin/env python3
"""Audit mechanical materiality of the legacy WR coverage heuristic.

This audit is outcome-free. It consumes one preserved production artifact and
reconstructs only the deterministic pre-simulation target-entitlement seams.

Counterfactual:
- remove coverage_penalty() only;
- leave every other football input/model/threshold untouched;
- do not price or grade bets.

The script fails closed if it cannot reproduce the frozen current M38 and
WR-R15 states to numerical tolerance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WR_POSITIONS = {"WR", "LWR", "RWR", "SWR"}
WR_MULT = (1.40, 1.14, 0.91, 0.78)
TARGET_CAP = 0.95
SAFE_CAP = float(np.nextafter(TARGET_CAP, 0.0))
TOL = 1e-10


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _sharpen(group: pd.DataFrame, shares: np.ndarray) -> np.ndarray:
    clean = np.clip(np.nan_to_num(np.asarray(shares, dtype=float)), 0.0, TARGET_CAP)
    pos = group["position"].astype("string").fillna("").str.upper().to_numpy()
    wr_idx = np.flatnonzero(np.isin(pos, list(WR_POSITIONS)))
    if len(wr_idx) <= 1:
        return clean
    wr = clean[wr_idx].copy()
    total = float(wr.sum())
    if total <= 0:
        return clean
    order = np.argsort(-wr, kind="stable")
    mult = np.ones(len(wr), dtype=float)
    for rank, idx in enumerate(order):
        mult[idx] = WR_MULT[min(rank, len(WR_MULT) - 1)]
    sharpened = wr * mult
    if sharpened.sum() <= 0:
        return clean
    sharpened *= total / float(sharpened.sum())
    out = clean.copy()
    out[wr_idx] = sharpened
    return out


def _materialize(frame: pd.DataFrame, share_col: str) -> pd.Series:
    out = pd.Series(index=frame.index, dtype=float)
    for (_, _), idx in frame.groupby(["event_id", "team"], sort=False).groups.items():
        g = frame.loc[idx].sort_values(["event_id", "team", "player_clean_key"]).copy()
        raw = _num(g[share_col]).fillna(0.0).to_numpy(float)
        raw = np.clip(np.nan_to_num(raw), 0.0, TARGET_CAP)
        sh = _sharpen(g, raw)
        total = float(sh.sum())
        scale = SAFE_CAP / total if total > TARGET_CAP else 1.0
        out.loc[g.index] = sh * scale
    return out


def _wr_r15_score(frame: pd.DataFrame, model: dict, baseline_col: str) -> pd.DataFrame:
    x = frame.copy()
    x["_pool"] = x.groupby(["event_id", "team"])[baseline_col].transform("sum")
    x["_room"] = np.where(x["_pool"].gt(0), x[baseline_col] / x["_pool"], 0.0)
    x["_log_pool"] = np.log1p(x["_pool"].clip(lower=0.0))
    x["_prior1_same"] = (_num(x["prior_count_same_team"]).fillna(0) >= 1).astype(float)
    x["_prior3_same"] = (_num(x["prior_count_same_team"]).fillna(0) >= 3).astype(float)
    x["_log_same"] = np.log1p(_num(x["prior_count_same_team"]).fillna(0).clip(lower=0))
    x["_log_any"] = np.log1p(_num(x["prior_count_anyteam"]).fillna(0).clip(lower=0))

    source = {
        "b0_secondary_room_share": "_room",
        "log_b0_secondary_pool": "_log_pool",
        "secondary_room_size": "secondary_room_size",
        "prior1_same_team_offense_pct": "prior1_same_team_offense_pct",
        "prior1_same_team_offense_snaps": "prior1_same_team_offense_snaps",
        "prior1_anyteam_offense_pct": "prior1_anyteam_offense_pct",
        "prior3_anyteam_offense_pct": "prior3_anyteam_offense_pct",
        "prior1_anyteam_offense_snaps": "prior1_anyteam_offense_snaps",
        "prior3_anyteam_offense_snaps": "prior3_anyteam_offense_snaps",
        "log1p_prior_count_same_team": "_log_same",
        "log1p_prior_count_anyteam": "_log_any",
        "prior1_same_team_available": "_prior1_same",
        "prior3_same_team_available": "_prior3_same",
        "secondary_snap_share_prior1_same_team": "secondary_snap_share_prior1_same_team",
        "secondary_snap_share_prior3_anyteam": "secondary_snap_share_prior3_anyteam",
    }

    matrix = np.column_stack([
        _num(x[source[f]]).fillna(0.0).to_numpy(float) for f in model["features"]
    ])
    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    coef = np.asarray(model["ridge_coef"], dtype=float)
    residual = ((matrix - mean) / scale) @ coef + float(model["ridge_intercept"])
    lo, hi = [float(v) for v in model["prediction_clip"]]
    residual = np.clip(residual, lo, hi)
    score = np.log(x["_room"].to_numpy(float) + float(model["eps"])) + residual

    x["_residual"] = residual
    x["_score"] = score
    x["_final"] = x[baseline_col].astype(float)
    x["_room_final"] = 0.0

    for (_, _), idx in x.groupby(["event_id", "team"], sort=False).groups.items():
        pool = float(x.loc[idx, "_pool"].iloc[0])
        score_arr = x.loc[idx, "_score"].to_numpy(float)
        if pool <= 0:
            room = np.zeros(len(idx), dtype=float)
            candidate = np.zeros(len(idx), dtype=float)
        else:
            stable = score_arr - float(np.max(score_arr))
            weight = np.exp(stable)
            room = weight / float(weight.sum())
            candidate = pool * room
            candidate[int(np.argmax(room))] += pool - float(candidate.sum())
        x.loc[idx, "_room_final"] = room
        x.loc[idx, "_final"] = candidate
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    data = args.artifact_root / "data"
    outputs = args.artifact_root / "outputs"
    trace = pd.read_csv(data / "target_entitlement_v1_trace.csv")
    rules = pd.read_csv(data / "model_rule_simulation_inputs.csv")
    wr_trace = pd.read_csv(data / "wr_r15_full_slate_entitlement_trace.csv")
    team_cov = pd.read_csv(data / "cb_coverage_team.csv")
    props = pd.read_csv(outputs / "props_priced_clean.csv")
    model = json.loads(
        (data / "models/wr_r15_production_model_v1/wr_r15_production_model_v1.json")
        .read_text(encoding="utf-8")
    )

    if not team_cov["coverage_available"].eq(1).all():
        raise RuntimeError("coverage availability is not complete in source artifact")

    keys = ["team", "player_clean_key"]
    if rules.duplicated(keys).any():
        # Pricing metrics repeat one football player across markets. Every repeated
        # row must agree on the protected rule state before collapsing.
        protected = [
            "position", "primary_cb", "coverage_man_rate_opp_y",
            "coverage_zone_rate_opp_y", "rules_ypt", "rules_tgt_share",
        ]
        for c in protected:
            if rules.groupby(keys)[c].nunique(dropna=False).max() > 1:
                raise RuntimeError(f"rule input disagrees across repeated market rows: {c}")
    rules = rules.drop_duplicates(keys).copy()

    keep = [
        "team", "player_clean_key", "primary_cb", "coverage_man_rate_opp_y",
        "coverage_zone_rate_opp_y", "rules_ypt",
    ]
    z = trace.merge(rules[keep], on=keys, how="left", validate="one_to_one")
    pos = z["position"].astype("string").fillna("").str.upper()
    is_wr = pos.isin(WR_POSITIONS)

    # coverage_penalty only runs when both target share and YPT are finite.
    ypt_finite = _num(z["rules_ypt"]).notna()
    shadow = z["primary_cb"].astype("string").fillna("").str.strip().ne("")
    zone = _num(z["coverage_zone_rate_opp_y"]).ge(0.60)
    man = _num(z["coverage_man_rate_opp_y"]).ge(0.50)
    tough = is_wr & ypt_finite & shadow
    heavy_man = is_wr & ypt_finite & shadow & man
    heavy_zone = is_wr & ypt_finite & (~shadow) & zone

    z["coverage_state"] = np.select(
        [tough | heavy_man, heavy_zone],
        ["TOUGH_SHADOW_OR_HEAVY_MAN", "HEAVY_ZONE"],
        default="NONE",
    )
    z["target_factor"] = np.select(
        [z["coverage_state"].eq("TOUGH_SHADOW_OR_HEAVY_MAN"),
         z["coverage_state"].eq("HEAVY_ZONE")],
        [0.92, 1.06],
        default=1.0,
    )
    z["ypt_factor"] = np.select(
        [z["coverage_state"].eq("TOUGH_SHADOW_OR_HEAVY_MAN"),
         z["coverage_state"].eq("HEAVY_ZONE")],
        [0.94, 1.04],
        default=1.0,
    )

    z["current_rule_share"] = _num(z["rules_tgt_share"])
    z["counterfactual_rule_share"] = z["current_rule_share"] / z["target_factor"]
    z["reconstructed_current_baseline"] = _materialize(z, "current_rule_share")
    z["counterfactual_baseline"] = _materialize(z, "counterfactual_rule_share")

    frozen = _num(z["m38_explicit_entitlement_tgt_share"])
    gap = (z["reconstructed_current_baseline"] - frozen).abs()
    if float(gap.max()) > TOL:
        raise RuntimeError(f"M38/entitlement current-state reproduction failed: {gap.max()}")

    # WR-R15 anchors must remain stable; otherwise the compact trace lacks the
    # former anchor's secondary-only feature record and the audit fails closed.
    current_wr = z.loc[is_wr].copy()
    cur_anchor = current_wr.loc[
        current_wr.groupby(["event_id", "team"])["m38_explicit_entitlement_tgt_share"].idxmax(),
        ["event_id", "team", "player_clean_key"],
    ].set_index(["event_id", "team"])["player_clean_key"]
    cf_anchor = current_wr.loc[
        current_wr.groupby(["event_id", "team"])["counterfactual_baseline"].idxmax(),
        ["event_id", "team", "player_clean_key"],
    ].set_index(["event_id", "team"])["player_clean_key"]
    if not cur_anchor.equals(cf_anchor):
        raise RuntimeError("coverage removal changes WR-R15 anchor identity; compact replay insufficient")

    cf_map = z.set_index(["event_id", "team", "player_clean_key"])["counterfactual_baseline"].to_dict()
    wr_trace["counterfactual_baseline"] = [
        cf_map[(r.event_id, r.team, r.player_clean_key)] for r in wr_trace.itertuples(index=False)
    ]

    control = _wr_r15_score(wr_trace, model, "wr_r15_baseline_entitlement_tgt_share")
    control_gap = (control["_final"] - wr_trace["wr_r15_entitlement_tgt_share"]).abs()
    residual_gap = (control["_residual"] - wr_trace["wr_r15_residual"]).abs()
    if float(control_gap.max()) > TOL or float(residual_gap.max()) > TOL:
        raise RuntimeError(
            f"WR-R15 control reproduction failed final={control_gap.max()} residual={residual_gap.max()}"
        )

    cf_secondary = _wr_r15_score(wr_trace, model, "counterfactual_baseline")
    sec_final = cf_secondary.set_index(["event_id", "team", "player_clean_key"])["_final"].to_dict()

    z["counterfactual_final_entitlement"] = _num(z["entitlement_tgt_share"])
    for idx, r in z.loc[is_wr].iterrows():
        key = (r["event_id"], r["team"], r["player_clean_key"])
        z.at[idx, "counterfactual_final_entitlement"] = sec_final.get(
            key, float(r["counterfactual_baseline"])
        )

    # TE-R5P and other non-WR entitlement consumers preserve their position/team
    # pool. Coverage removal changes only the team conservation scale for them,
    # so their current final entitlement scales by exact baseline ratio.
    non_wr = ~is_wr
    ratio = np.divide(
        z.loc[non_wr, "counterfactual_baseline"].to_numpy(float),
        frozen.loc[non_wr].to_numpy(float),
        out=np.ones(int(non_wr.sum()), dtype=float),
        where=frozen.loc[non_wr].to_numpy(float) > 1e-15,
    )
    z.loc[non_wr, "counterfactual_final_entitlement"] = (
        _num(z.loc[non_wr, "entitlement_tgt_share"]).to_numpy(float) * ratio
    )

    z["current_ypt"] = _num(z["rules_ypt"])
    z["counterfactual_ypt"] = z["current_ypt"] / z["ypt_factor"]
    z["current_kernel"] = _num(z["entitlement_tgt_share"]) * z["current_ypt"]
    z["counterfactual_kernel"] = z["counterfactual_final_entitlement"] * z["counterfactual_ypt"]
    z["kernel_pct_delta"] = (z["counterfactual_kernel"] / z["current_kernel"] - 1.0) * 100.0
    z["entitlement_pct_delta"] = (
        z["counterfactual_final_entitlement"] / _num(z["entitlement_tgt_share"]) - 1.0
    ) * 100.0

    affected_wr = is_wr & z["coverage_state"].ne("NONE") & z["current_ypt"].notna()
    affected_teams = set(
        map(tuple, z.loc[affected_wr, ["event_id", "team"]].drop_duplicates().itertuples(index=False, name=None))
    )
    z["affected_team"] = [
        (e, t) in affected_teams for e, t in zip(z["event_id"], z["team"])
    ]

    pool = (
        z.groupby(["event_id", "team", pos.rename("position")])[
            ["entitlement_tgt_share", "counterfactual_final_entitlement"]
        ].sum()
    )
    pool["pct_delta"] = (
        pool["counterfactual_final_entitlement"] / pool["entitlement_tgt_share"] - 1.0
    ) * 100.0
    pool = pool.reset_index()

    rec_ids = (
        props.loc[props["market"].eq("rec_yards"), ["team", "player_clean_key"]]
        .drop_duplicates()
        .assign(priced_rec_yards=True)
    )
    wr_rows = z.loc[is_wr].merge(rec_ids, on=["team", "player_clean_key"], how="left")
    wr_rows["priced_rec_yards"] = wr_rows["priced_rec_yards"].fillna(False)

    aw = z.loc[affected_wr].copy()
    summary = {
        "source_artifact_week": 3,
        "outcomes_used": False,
        "sportsbook_used_as_football_input": False,
        "wr_rows": int(is_wr.sum()),
        "affected_wr_rows": int(affected_wr.sum()),
        "affected_wr_fraction": float(affected_wr.mean() / is_wr.mean()),
        "affected_teams": int(len(affected_teams)),
        "wr_teams": int(z.loc[is_wr, "team"].nunique()),
        "wr_primary_cb_nonmissing": int((is_wr & shadow).sum()),
        "coverage_state_counts": z.loc[is_wr, "coverage_state"].value_counts().to_dict(),
        "m38_control_max_gap": float(gap.max()),
        "wr_r15_control_max_final_gap": float(control_gap.max()),
        "wr_r15_control_max_residual_gap": float(residual_gap.max()),
        "affected_wr_entitlement_pct_delta_mean": float(aw["entitlement_pct_delta"].mean()),
        "affected_wr_entitlement_pct_delta_median": float(aw["entitlement_pct_delta"].median()),
        "affected_wr_kernel_pct_delta_mean": float(aw["kernel_pct_delta"].mean()),
        "affected_wr_kernel_pct_delta_median": float(aw["kernel_pct_delta"].median()),
        "affected_wr_kernel_pct_delta_min": float(aw["kernel_pct_delta"].min()),
        "affected_wr_kernel_pct_delta_max": float(aw["kernel_pct_delta"].max()),
        "priced_rec_yards_wr_identities": int(wr_rows["priced_rec_yards"].sum()),
        "affected_priced_rec_yards_wr_identities": int(
            (wr_rows["priced_rec_yards"] & wr_rows["coverage_state"].ne("NONE")).sum()
        ),
    }

    for p in ("WR", "TE", "RB", "QB"):
        q = pool.loc[pool["position"].eq(p)]
        q = q.loc[
            [(e, t) in affected_teams for e, t in zip(q["event_id"], q["team"])]
        ]
        if len(q):
            summary[f"affected_team_{p.lower()}_pool_pct_delta_mean"] = float(q["pct_delta"].mean())
            summary[f"affected_team_{p.lower()}_pool_pct_delta_median"] = float(q["pct_delta"].median())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    z.to_csv(args.out_dir / "coverage_heuristic_mechanical_row_detail.csv", index=False)
    pool.to_csv(args.out_dir / "coverage_heuristic_position_pool_detail.csv", index=False)
    (args.out_dir / "coverage_heuristic_mechanical_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
