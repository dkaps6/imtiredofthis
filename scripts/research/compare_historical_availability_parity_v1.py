#!/usr/bin/env python3
"""Compare baseline ACT+INA opportunity allocation to ACT-only reconstruction."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HIGH_BIN = {
    ("QB", "pass_attempts"): "41_PLUS",
    ("RB", "carries"): "15_PLUS",
    ("FB", "carries"): "15_PLUS",
    ("RB", "targets"): "09_PLUS",
    ("FB", "targets"): "09_PLUS",
    ("WR", "targets"): "09_PLUS",
    ("TE", "targets"): "09_PLUS",
}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _pkey(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def _metrics(g: pd.DataFrame, prefix: str) -> dict:
    pred = pd.to_numeric(g[f"{prefix}_predicted_opportunities"], errors="coerce")
    actual = pd.to_numeric(g["actual_opportunities"], errors="coerce")
    err = pred - actual
    active = actual.gt(0)
    zero = actual.eq(0)
    return {
        f"{prefix}_mae": float(err.abs().mean()),
        f"{prefix}_bias": float(err.mean()),
        f"{prefix}_pearson": float(pred.corr(actual)) if pred.nunique() > 1 and actual.nunique() > 1 else np.nan,
        f"{prefix}_spearman": float(pred.rank(method="average").corr(actual.rank(method="average"))) if pred.nunique() > 1 and actual.nunique() > 1 else np.nan,
        f"{prefix}_mean_prediction": float(pred.mean()),
        f"{prefix}_zero_mean_prediction": float(pred.loc[zero].mean()) if zero.any() else np.nan,
        f"{prefix}_active_bias": float(err.loc[active].mean()) if active.any() else np.nan,
        f"{prefix}_active_mae": float(err.loc[active].abs().mean()) if active.any() else np.nan,
    }


def compare(
    *,
    baseline_rows_path: Path,
    act_rows_path: Path,
    status_map_path: Path,
    group_out: Path,
    mass_out: Path,
    summary_out: Path,
) -> dict:
    b = _read(baseline_rows_path, "baseline opportunity rows")
    a = _read(act_rows_path, "ACT-only opportunity rows")
    status = _read(status_map_path, "historical availability status map")

    for x in (b, a):
        x["player_join"] = x["player"].map(_pkey)
    status["player_join"] = status["player_join"].astype(str)

    status_key = ["season", "week", "team", "player_join"]
    if status.duplicated(status_key).any():
        raise RuntimeError("status map duplicate identity")
    b = b.merge(
        status[status_key + ["status"]],
        on=status_key,
        how="left",
        validate="many_to_one",
    )
    if b["status"].isna().any():
        bad = b.loc[b["status"].isna(), ["week", "team", "player"]].head(20)
        raise RuntimeError(f"baseline audit rows missing status: {bad.to_dict('records')}")
    if not b["status"].isin({"ACT", "INA"}).all():
        raise RuntimeError(f"unexpected baseline audit status: {b['status'].value_counts().to_dict()}")

    # Baseline modeled mass assigned to explicit inactive rows.
    mass_rows = []
    for (pos, opp), g in b.groupby(["position_family", "opportunity_type"], dropna=False):
        total = float(pd.to_numeric(g["predicted_opportunities"], errors="coerce").sum())
        ina = g.loc[g["status"].eq("INA")]
        ina_mass = float(pd.to_numeric(ina["predicted_opportunities"], errors="coerce").sum())
        teamweek = (
            ina.groupby(["week", "team"])["predicted_opportunities"].sum()
            if not ina.empty else pd.Series(dtype=float)
        )
        mass_rows.append({
            "position_family": str(pos),
            "opportunity_type": str(opp),
            "baseline_rows": int(len(g)),
            "ina_rows": int(len(ina)),
            "baseline_predicted_opportunity_mass": total,
            "ina_predicted_opportunity_mass": ina_mass,
            "ina_share_of_modeled_player_opportunity": float(ina_mass / total) if total > 0 else np.nan,
            "team_weeks_with_positive_ina_mass": int((teamweek > 0).sum()) if len(teamweek) else 0,
        })
    mass = pd.DataFrame(mass_rows)

    # Pair exact ACT player identities across variants.
    keys = ["season", "week", "event_id", "team", "player_join", "opportunity_type"]
    b_act = b.loc[b["status"].eq("ACT")].copy()
    keep_b = keys + [
        "player", "player_clean_key", "position_family", "actual_opportunities",
        "actual_opportunity_bin", "predicted_opportunities", "linked_yards_error",
        "linked_count_error",
    ]
    keep_a = keys + [
        "player", "player_clean_key", "position_family", "actual_opportunities",
        "actual_opportunity_bin", "predicted_opportunities", "linked_yards_error",
        "linked_count_error",
    ]
    if b_act.duplicated(keys).any() or a.duplicated(keys).any():
        raise RuntimeError("duplicate pair identity in parity inputs")

    paired = b_act[keep_b].merge(
        a[keep_a],
        on=keys,
        how="inner",
        suffixes=("_baseline", "_act"),
        validate="one_to_one",
    )
    if paired.empty:
        raise RuntimeError("zero paired ACT player rows")
    if not np.allclose(
        pd.to_numeric(paired["actual_opportunities_baseline"], errors="coerce"),
        pd.to_numeric(paired["actual_opportunities_act"], errors="coerce"),
        rtol=0,
        atol=0,
        equal_nan=True,
    ):
        raise RuntimeError("actual opportunities changed across diagnostic variants")

    paired["actual_opportunities"] = pd.to_numeric(paired["actual_opportunities_baseline"], errors="coerce")
    paired["actual_opportunity_bin"] = paired["actual_opportunity_bin_baseline"].astype(str)
    paired["position_family"] = paired["position_family_baseline"].astype(str)
    paired["baseline_predicted_opportunities"] = pd.to_numeric(paired["predicted_opportunities_baseline"], errors="coerce")
    paired["act_predicted_opportunities"] = pd.to_numeric(paired["predicted_opportunities_act"], errors="coerce")
    paired["prediction_delta_act_minus_baseline"] = (
        paired["act_predicted_opportunities"] - paired["baseline_predicted_opportunities"]
    )

    group_rows = []
    for (pos, opp), g in paired.groupby(["position_family", "opportunity_type"], dropna=False):
        rec = {
            "position_family": str(pos),
            "opportunity_type": str(opp),
            "paired_rows": int(len(g)),
            "actual_zero_rate": float(g["actual_opportunities"].eq(0).mean()),
            "mean_prediction_delta_act_minus_baseline": float(g["prediction_delta_act_minus_baseline"].mean()),
        }
        rec.update(_metrics(g, "baseline"))
        rec.update(_metrics(g, "act"))
        rec["mae_improvement_baseline_minus_act"] = rec["baseline_mae"] - rec["act_mae"]
        rec["active_mae_improvement_baseline_minus_act"] = rec["baseline_active_mae"] - rec["act_active_mae"]

        h = g.loc[g["actual_opportunity_bin"].eq(HIGH_BIN.get((str(pos), str(opp)), "__NONE__"))].copy()
        rec["high_bin"] = HIGH_BIN.get((str(pos), str(opp)), "")
        rec["high_bin_rows"] = int(len(h))
        if len(h):
            actual = pd.to_numeric(h["actual_opportunities"], errors="coerce")
            rec["baseline_high_bin_bias"] = float((h["baseline_predicted_opportunities"] - actual).mean())
            rec["act_high_bin_bias"] = float((h["act_predicted_opportunities"] - actual).mean())
        else:
            rec["baseline_high_bin_bias"] = np.nan
            rec["act_high_bin_bias"] = np.nan

        linked_y = pd.to_numeric(g["linked_yards_error_baseline"], errors="coerce")
        linked_c = pd.to_numeric(g["linked_count_error_baseline"], errors="coerce")
        berr = g["baseline_predicted_opportunities"] - g["actual_opportunities"]
        aerr = g["act_predicted_opportunities"] - g["actual_opportunities"]
        rec["baseline_opportunity_error_vs_linked_yards_error_pearson"] = (
            float(berr.corr(linked_y)) if linked_y.notna().sum() > 2 else np.nan
        )
        rec["act_opportunity_error_vs_linked_yards_error_pearson"] = (
            float(aerr.corr(linked_y)) if linked_y.notna().sum() > 2 else np.nan
        )
        rec["baseline_opportunity_error_vs_linked_count_error_pearson"] = (
            float(berr.corr(linked_c)) if linked_c.notna().sum() > 2 else np.nan
        )
        rec["act_opportunity_error_vs_linked_count_error_pearson"] = (
            float(aerr.corr(linked_c)) if linked_c.notna().sum() > 2 else np.nan
        )
        group_rows.append(rec)

    group = pd.DataFrame(group_rows)

    # Identity changes / coverage.
    base_keys = set(map(tuple, b[keys].astype(str).to_numpy()))
    act_keys = set(map(tuple, a[keys].astype(str).to_numpy()))
    new_act = act_keys - base_keys
    dropped = base_keys - act_keys

    qb_b = b.loc[b["position_family"].eq("QB") & b["opportunity_type"].eq("pass_attempts")].copy()
    qb_a = a.loc[a["position_family"].eq("QB") & a["opportunity_type"].eq("pass_attempts")].copy()
    qb_map_b = {(int(r.week), str(r.team)): str(r.player) for r in qb_b.itertuples(index=False)}
    qb_map_a = {(int(r.week), str(r.team)): str(r.player) for r in qb_a.itertuples(index=False)}
    qb_pairs = sorted(set(qb_map_b) | set(qb_map_a))
    qb_changes = [
        {
            "week": w, "team": t,
            "baseline_qb": qb_map_b.get((w, t), ""),
            "act_only_qb": qb_map_a.get((w, t), ""),
        }
        for w, t in qb_pairs
        if qb_map_b.get((w, t), "") != qb_map_a.get((w, t), "")
    ]

    group_out.parent.mkdir(parents=True, exist_ok=True)
    group.to_csv(group_out, index=False)
    mass.to_csv(mass_out, index=False)

    payload = {
        "version": "HISTORICAL_AVAILABILITY_PARITY_DIAGNOSTIC_V1",
        "baseline_rows": int(len(b)),
        "act_only_rows": int(len(a)),
        "baseline_act_rows": int(b["status"].eq("ACT").sum()),
        "baseline_ina_rows": int(b["status"].eq("INA").sum()),
        "paired_act_rows": int(len(paired)),
        "act_only_new_identity_rows": int(len(new_act)),
        "baseline_dropped_identity_rows": int(len(dropped)),
        "qb_identity_change_count": int(len(qb_changes)),
        "qb_identity_changes": qb_changes,
        "group_comparison": group.to_dict("records"),
        "inactive_mass_audit": mass.to_dict("records"),
        "parameters_fit": 0,
        "automatic_promotion": False,
        "sportsbook_inputs_used_upstream": False,
        "paid_odds_api_used": False,
        "disposition": "DIAGNOSTIC_COMPLETE_RAW_RESULT_REQUIRES_INTERPRETATION",
    }
    summary_out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-rows", type=Path, required=True)
    p.add_argument("--act-rows", type=Path, required=True)
    p.add_argument("--status-map", type=Path, required=True)
    p.add_argument("--group-out", type=Path, required=True)
    p.add_argument("--mass-out", type=Path, required=True)
    p.add_argument("--summary-out", type=Path, required=True)
    a = p.parse_args()
    compare(
        baseline_rows_path=a.baseline_rows,
        act_rows_path=a.act_rows,
        status_map_path=a.status_map,
        group_out=a.group_out,
        mass_out=a.mass_out,
        summary_out=a.summary_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
