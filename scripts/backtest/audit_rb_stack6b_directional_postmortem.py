#!/usr/bin/env python3
"""RB STACK6B no-fit directional/depth failure postmortem.

Consumes only the frozen STACK6B casebook. No model is fit, no threshold is
searched, and sportsbook data is intentionally excluded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ["COMPACT_ROLE", "AGG_PLUS_COMPACT"]
TOL = 1e-12


def num(s):
    return pd.to_numeric(s, errors="coerce")


def one(root: Path, name: str):
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} under {root}; found {len(hits)}")
    x = pd.read_csv(hits[0], low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    n = num(s)
    out = n.eq(1)
    unresolved = n.isna() & s.notna()
    if unresolved.any():
        text = s.astype(str).str.strip().str.lower()
        out = out.where(~unresolved, text.isin({"true", "t", "yes", "y"}))
    return out.fillna(False).astype(bool)


def metric(y, p):
    q = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    if q.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
    e = q.p - q.y
    corr = q.y.corr(q.p) if len(q) >= 3 and q.y.nunique() > 1 and q.p.nunique() > 1 else np.nan
    return {
        "n": int(len(q)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.square(e).mean())),
        "bias": float(e.mean()),
        "corr": float(corr) if pd.notna(corr) else np.nan,
    }


def validate(x):
    z = x.copy()
    required = [
        "week",
        "depth_rank",
        "parent_att",
        "parent_yards",
        "parent_ypc",
        "actual_rush_att",
        "actual_rush_yards",
        "stack6b_model_eligible",
    ]
    for arm in ARMS:
        required += [f"delta_{arm}".lower(), f"pred_att_{arm}".lower(), f"pred_yards_{arm}".lower()]
    missing = [c for c in required if c not in z]
    if missing:
        raise RuntimeError(f"STACK6B postmortem missing frozen fields: {missing}")

    z["week"] = num(z.week).astype(int)
    z["depth_rank"] = num(z.depth_rank)
    z["parent_att"] = num(z.parent_att)
    z["parent_yards"] = num(z.parent_yards)
    z["parent_ypc"] = num(z.parent_ypc)
    z["actual_rush_att"] = num(z.actual_rush_att)
    z["actual_rush_yards"] = num(z.actual_rush_yards)
    z["stack6b_model_eligible"] = bool_series(z.stack6b_model_eligible)
    z["actual_carry_residual"] = z.actual_rush_att - z.parent_att

    for arm in ARMS:
        a = arm.lower()
        z[f"delta_{a}"] = num(z[f"delta_{a}"])
        z[f"pred_att_{a}"] = num(z[f"pred_att_{a}"])
        z[f"pred_yards_{a}"] = num(z[f"pred_yards_{a}"])
    return z


def direction(delta):
    d = num(delta)
    return np.select(
        [d.lt(-TOL), d.gt(TOL)],
        ["CONTRACT", "EXPAND"],
        default="NO_CHANGE",
    )


def phase_masks(z):
    return {
        "W6_18": z.stack6b_model_eligible,
        "W6_12": z.stack6b_model_eligible & z.week.between(6, 12),
        "W13_18": z.stack6b_model_eligible & z.week.between(13, 18),
    }


def depth_masks(z):
    return {
        "DEPTH2PLUS": z.stack6b_model_eligible,
        "DEPTH2": z.stack6b_model_eligible & z.depth_rank.eq(2),
        "DEPTH3PLUS": z.stack6b_model_eligible & z.depth_rank.ge(3),
    }


def directional_table(z):
    rows = []
    phases = phase_masks(z)
    depths = depth_masks(z)

    for arm in ARMS:
        a = arm.lower()
        dcol = f"delta_{a}"
        acol = f"pred_att_{a}"
        ycol = f"pred_yards_{a}"
        dirs = pd.Series(direction(z[dcol]), index=z.index)

        for phase, pmask in phases.items():
            for depth, dmask in depths.items():
                base_mask = pmask & dmask
                for label in ["ALL_DIRECTIONS", "CONTRACT", "EXPAND", "NO_CHANGE"]:
                    mask = base_mask if label == "ALL_DIRECTIONS" else base_mask & dirs.eq(label)
                    g = z.loc[mask].copy()
                    if g.empty:
                        continue

                    pc = metric(g.actual_rush_att, g.parent_att)
                    mc = metric(g.actual_rush_att, g[acol])
                    py = metric(g.actual_rush_yards, g.parent_yards)
                    my = metric(g.actual_rush_yards, g[ycol])
                    carry_imp = (g.parent_att - g.actual_rush_att).abs() - (g[acol] - g.actual_rush_att).abs()
                    yard_imp = (g.parent_yards - g.actual_rush_yards).abs() - (g[ycol] - g.actual_rush_yards).abs()

                    nonzero = g[dcol].abs().gt(TOL)
                    aligned = (np.sign(g.loc[nonzero, dcol]) == np.sign(g.loc[nonzero, "actual_carry_residual"]))
                    rows.append(
                        {
                            "arm": arm,
                            "phase": phase,
                            "depth": depth,
                            "direction": label,
                            "n": int(len(g)),
                            "mean_delta": float(g[dcol].mean()),
                            "mean_abs_delta": float(g[dcol].abs().mean()),
                            "mean_actual_carry_residual": float(g.actual_carry_residual.mean()),
                            "parent_carry_mae": pc["mae"],
                            "model_carry_mae": mc["mae"],
                            "carry_mae_gain": float(pc["mae"] - mc["mae"]),
                            "parent_yard_mae": py["mae"],
                            "model_yard_mae": my["mae"],
                            "yard_mae_gain": float(py["mae"] - my["mae"]),
                            "parent_carry_bias": pc["bias"],
                            "model_carry_bias": mc["bias"],
                            "directional_alignment_rate": float(aligned.mean()) if len(aligned) else np.nan,
                            "carry_error_improved_rate": float(carry_imp.gt(0).mean()),
                            "yard_error_improved_rate": float(yard_imp.gt(0).mean()),
                            "mean_carry_abs_error_recovery": float(carry_imp.mean()),
                            "mean_yard_abs_error_recovery": float(yard_imp.mean()),
                        }
                    )
    return pd.DataFrame(rows)


def counterfactual_predictions(z, arm, mode):
    a = arm.lower()
    delta = num(z[f"delta_{a}"]).fillna(0.0)
    if mode == "FROZEN_ARM":
        used = delta
    elif mode == "CONTRACTION_ONLY":
        used = delta.clip(upper=0.0)
    elif mode == "EXPANSION_ONLY":
        used = delta.clip(lower=0.0)
    elif mode == "P3_PARENT":
        used = pd.Series(0.0, index=z.index)
    else:
        raise RuntimeError(f"unknown diagnostic counterfactual {mode}")

    att = (num(z.parent_att) + used).clip(lower=0.0)
    ypc = num(z.parent_ypc)
    yards = pd.Series(np.where(ypc.notna(), att * ypc, num(z.parent_yards)), index=z.index)
    return used, att, yards


def counterfactual_table(z):
    rows = []
    scopes = {}
    for phase, pmask in phase_masks(z).items():
        for depth, dmask in depth_masks(z).items():
            scopes[f"{phase}__{depth}"] = pmask & dmask

    for arm in ARMS:
        for mode in ["P3_PARENT", "FROZEN_ARM", "CONTRACTION_ONLY", "EXPANSION_ONLY"]:
            used, att, yards = counterfactual_predictions(z, arm, mode)
            for scope, mask in scopes.items():
                g = z.loc[mask]
                if g.empty:
                    continue
                cm = metric(g.actual_rush_att, att.loc[g.index])
                ym = metric(g.actual_rush_yards, yards.loc[g.index])
                rows.append(
                    {
                        "arm_source": arm,
                        "diagnostic_mode": mode,
                        "scope": scope,
                        "n": int(len(g)),
                        "mean_used_delta": float(used.loc[g.index].mean()),
                        "mean_abs_used_delta": float(used.loc[g.index].abs().mean()),
                        "carry_mae": cm["mae"],
                        "carry_rmse": cm["rmse"],
                        "carry_bias": cm["bias"],
                        "yard_mae": ym["mae"],
                        "yard_rmse": ym["rmse"],
                        "yard_bias": ym["bias"],
                    }
                )
    out = pd.DataFrame(rows)

    # Add gains vs the arm-specific P3_PARENT diagnostic baseline.
    gain_rows = []
    for (arm, scope), g in out.groupby(["arm_source", "scope"], sort=False):
        base = g.loc[g.diagnostic_mode.eq("P3_PARENT")]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, r in g.iterrows():
            rec = r.to_dict()
            rec["carry_mae_gain_vs_parent"] = float(b.carry_mae - r.carry_mae)
            rec["yard_mae_gain_vs_parent"] = float(b.yard_mae - r.yard_mae)
            gain_rows.append(rec)
    return pd.DataFrame(gain_rows)


def contribution_table(z):
    rows = []
    q = z.loc[z.stack6b_model_eligible].copy()
    for arm in ARMS:
        a = arm.lower()
        delta = num(q[f"delta_{a}"])
        carry_gain = (q.parent_att - q.actual_rush_att).abs() - (q[f"pred_att_{a}"] - q.actual_rush_att).abs()
        yard_gain = (q.parent_yards - q.actual_rush_yards).abs() - (q[f"pred_yards_{a}"] - q.actual_rush_yards).abs()
        for label, mask in {
            "CONTRACT": delta.lt(-TOL),
            "EXPAND": delta.gt(TOL),
            "NO_CHANGE": delta.abs().le(TOL),
        }.items():
            if not mask.any():
                continue
            rows.append(
                {
                    "arm": arm,
                    "direction": label,
                    "n": int(mask.sum()),
                    "share_of_eligible": float(mask.mean()),
                    "sum_carry_abs_error_recovery": float(carry_gain.loc[mask].sum()),
                    "sum_yard_abs_error_recovery": float(yard_gain.loc[mask].sum()),
                    "mean_carry_abs_error_recovery": float(carry_gain.loc[mask].mean()),
                    "mean_yard_abs_error_recovery": float(yard_gain.loc[mask].mean()),
                    "positive_carry_recovery_rate": float(carry_gain.loc[mask].gt(0).mean()),
                    "positive_yard_recovery_rate": float(yard_gain.loc[mask].gt(0).mean()),
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack6b-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    z = validate(one(a.stack6b_root, "stack6b_2025_casebook.csv"))
    directional = directional_table(z)
    counterfactual = counterfactual_table(z)
    contributions = contribution_table(z)

    protocol = pd.DataFrame(
        [
            {
                "input_rows": int(len(z)),
                "eligible_rows": int(z.stack6b_model_eligible.sum()),
                "model_fit": 0,
                "sportsbook_used": 0,
                "threshold_search": 0,
                "feature_search": 0,
                "hyperparameter_search": 0,
                "weight_search": 0,
                "production_change": 0,
                "counterfactuals_eligible_for_retention": 0,
                "purpose": "FAILURE_MECHANISM_ONLY",
            }
        ]
    )

    outputs = {
        "stack6b_postmortem_protocol.csv": protocol,
        "stack6b_postmortem_directional.csv": directional,
        "stack6b_postmortem_counterfactuals.csv": counterfactual,
        "stack6b_postmortem_contributions.csv": contributions,
    }
    for name, df in outputs.items():
        df.to_csv(a.out_dir / name, index=False)

    focus = counterfactual.loc[
        counterfactual.scope.isin(
            [
                "W6_18__DEPTH2PLUS",
                "W13_18__DEPTH2PLUS",
                "W6_18__DEPTH2",
                "W6_18__DEPTH3PLUS",
                "W13_18__DEPTH2",
                "W13_18__DEPTH3PLUS",
            ]
        )
    ]
    print("=== protocol ===")
    print(protocol.to_string(index=False))
    print("=== diagnostic counterfactual focus ===")
    print(focus.to_string(index=False))
    print("=== direction contribution ===")
    print(contributions.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
