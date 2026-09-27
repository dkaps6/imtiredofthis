#!/usr/bin/env python3
"""Diagnostic-only post-M89/M90 QB passing error decomposition.

Decomposes the promoted football-only QB passing-yards projection error into:
- passing opportunity (attempts),
- passing efficiency (YPA), and
- the non-factor residual between M89/M90 and attempts*YPA.

No candidate is fit. No sportsbook input is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

TARGET_SEASONS = (2024, 2025)
TOL = 1e-8


def _key(value) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _metric(actual: pd.Series, pred: pd.Series) -> dict:
    z = pd.DataFrame({
        "actual": pd.to_numeric(actual, errors="coerce"),
        "pred": pd.to_numeric(pred, errors="coerce"),
    }).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan}
    e = z["pred"] - z["actual"]
    corr = (
        float(z["pred"].corr(z["actual"]))
        if len(z) > 1 and z["pred"].nunique() > 1 and z["actual"].nunique() > 1
        else np.nan
    )
    return {
        "n": int(len(z)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "bias": float(e.mean()),
        "correlation": corr,
    }


def _actual_log_frame(logs: pd.DataFrame) -> pd.DataFrame:
    x = _lower(logs)
    required = {"season", "week", "team"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"player logs missing columns: {sorted(missing)}")

    if "player_clean_key" in x.columns:
        x["player_clean_key"] = x["player_clean_key"].map(_key)
    elif "player" in x.columns:
        x["player_clean_key"] = x["player"].map(_key)
    else:
        raise RuntimeError("player logs missing player identity")

    att_col = next((c for c in ("pass_att", "attempts", "passing_attempts") if c in x.columns), None)
    yd_col = next((c for c in ("pass_yards", "passing_yards") if c in x.columns), None)
    if not att_col or not yd_col:
        raise RuntimeError("player logs require pass attempts and pass yards")

    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["actual_pass_attempts"] = pd.to_numeric(x[att_col], errors="coerce")
    x["actual_pass_yards_logs"] = pd.to_numeric(x[yd_col], errors="coerce")
    x = x.loc[x["season"].isin(TARGET_SEASONS)].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)

    q = x.loc[x["actual_pass_attempts"].gt(0)].copy()
    q["actual_ypa"] = q["actual_pass_yards_logs"] / q["actual_pass_attempts"]
    keep = [
        "season", "week", "team", "player_clean_key",
        "actual_pass_attempts", "actual_pass_yards_logs", "actual_ypa",
    ]
    q = q[keep].drop_duplicates(["season", "week", "team", "player_clean_key"])
    if q.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("duplicate actual QB identity rows")
    return q


def build_decomposition(trace: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    t = _lower(trace)
    required = {
        "season", "week", "team", "player_clean_key",
        "actual_pass_yards", "football_synthesis", "pred_attempts", "pred_ypa",
    }
    missing = required - set(t.columns)
    if missing:
        raise RuntimeError(f"M89 trace missing columns: {sorted(missing)}")

    t["season"] = pd.to_numeric(t["season"], errors="coerce")
    t["week"] = pd.to_numeric(t["week"], errors="coerce")
    t = t.loc[t["season"].isin(TARGET_SEASONS)].copy()
    if set(t["season"].dropna().astype(int).unique()) != set(TARGET_SEASONS):
        raise RuntimeError("evaluation seasons are not exactly 2024 and 2025")
    t["season"] = t["season"].astype(int)
    t["week"] = t["week"].astype(int)
    t["team"] = t["team"].map(canon_team)
    t["player_clean_key"] = t["player_clean_key"].map(_key)
    identity = ["season", "week", "team", "player_clean_key"]
    if t.duplicated(identity).any():
        raise RuntimeError("duplicate M89 evaluation identity rows")

    for c in ("actual_pass_yards", "football_synthesis", "pred_attempts", "pred_ypa"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    if t[list(("actual_pass_yards", "football_synthesis", "pred_attempts", "pred_ypa"))].isna().any().any():
        bad = t.loc[
            t[list(("actual_pass_yards", "football_synthesis", "pred_attempts", "pred_ypa"))]
            .isna().any(axis=1),
            identity,
        ]
        raise RuntimeError(f"non-finite M89 decomposition primitives rows={len(bad)} sample={bad.head().to_dict('records')}")

    a = _actual_log_frame(logs)
    x = t.merge(a, on=identity, how="left", validate="one_to_one")
    missing_actual = x["actual_pass_attempts"].isna() | x["actual_ypa"].isna()
    if missing_actual.any():
        sample = x.loc[missing_actual, identity].head().to_dict("records")
        raise RuntimeError(f"missing actual attempts/YPA for M89 rows n={int(missing_actual.sum())} sample={sample}")
    if not x["actual_pass_attempts"].gt(0).all():
        raise RuntimeError("non-positive actual pass attempts in evaluated M89 rows")

    actual_gap = (x["actual_pass_yards"] - x["actual_pass_yards_logs"]).abs()
    if float(actual_gap.max()) > TOL:
        raise RuntimeError(f"trace/log actual passing-yard mismatch max_gap={float(actual_gap.max())}")

    P = x["pred_attempts"].to_numpy(float)
    A = x["actual_pass_attempts"].to_numpy(float)
    Yp = x["pred_ypa"].to_numpy(float)
    Ya = x["actual_ypa"].to_numpy(float)
    M = x["football_synthesis"].to_numpy(float)
    actual = x["actual_pass_yards"].to_numpy(float)

    primitive = P * Yp
    residual = M - primitive
    opp = (P - A) * (Yp + Ya) / 2.0
    eff = (Yp - Ya) * (P + A) / 2.0
    total = M - actual
    identity_gap = total - (opp + eff + residual)

    x["primitive_pred_yards"] = primitive
    x["nonfactor_residual"] = residual
    x["opportunity_contribution"] = opp
    x["efficiency_contribution"] = eff
    x["total_error"] = total
    x["decomposition_identity_gap"] = identity_gap
    x["opportunity_oracle_proj"] = A * Yp + residual
    x["efficiency_oracle_proj"] = P * Ya + residual
    x["both_primitives_oracle_proj"] = A * Ya + residual

    max_gap = float(np.max(np.abs(identity_gap))) if len(x) else np.nan
    if not np.isfinite(max_gap) or max_gap > TOL:
        raise RuntimeError(f"decomposition identity failed max_gap={max_gap}")

    both_gap = (
        x["both_primitives_oracle_proj"]
        - (x["actual_pass_yards"] + x["nonfactor_residual"])
    ).abs()
    if float(both_gap.max()) > TOL:
        raise RuntimeError(f"both-primitives oracle identity failed max_gap={float(both_gap.max())}")

    comps = x[[
        "opportunity_contribution", "efficiency_contribution", "nonfactor_residual"
    ]].abs()
    x["dominant_component"] = comps.idxmax(axis=1).map({
        "opportunity_contribution": "opportunity",
        "efficiency_contribution": "efficiency",
        "nonfactor_residual": "nonfactor_residual",
    })
    x["absolute_component_sum"] = comps.sum(axis=1)
    x["cancellation_present"] = (
        x["absolute_component_sum"] - x["total_error"].abs()
    ).gt(TOL).astype(int)
    x["error_bucket"] = pd.cut(
        x["total_error"].abs(),
        bins=[-np.inf, 25, 50, 75, 100, np.inf],
        right=False,
        labels=["LT25", "25_49", "50_74", "75_99", "GE100"],
    ).astype(str)

    x["attempt_quartile"] = pd.qcut(
        x["pred_attempts"].rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    x["ypa_quartile"] = pd.qcut(
        x["pred_ypa"].rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    return x


def _summary_rows(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    slices = [("POOLED", "ALL", x)]
    for season, g in x.groupby("season", sort=True):
        slices.append(("SEASON", str(int(season)), g))
    for bucket, g in x.groupby("error_bucket", sort=False):
        slices.append(("ERROR_BUCKET", str(bucket), g))
    for bucket, g in x.groupby("attempt_quartile", sort=True):
        slices.append(("PRED_ATTEMPT_QUARTILE", str(bucket), g))
    for bucket, g in x.groupby("ypa_quartile", sort=True):
        slices.append(("PRED_YPA_QUARTILE", str(bucket), g))

    for dim, bucket, g in slices:
        base = _metric(g["actual_pass_yards"], g["football_synthesis"])
        opp = _metric(g["actual_pass_yards"], g["opportunity_oracle_proj"])
        eff = _metric(g["actual_pass_yards"], g["efficiency_oracle_proj"])
        both = _metric(g["actual_pass_yards"], g["both_primitives_oracle_proj"])
        abs_mass = {
            "opportunity": float(g["opportunity_contribution"].abs().sum()),
            "efficiency": float(g["efficiency_contribution"].abs().sum()),
            "nonfactor_residual": float(g["nonfactor_residual"].abs().sum()),
        }
        denom = sum(abs_mass.values())
        dom = g["dominant_component"].value_counts().to_dict()
        rows.append({
            "dimension": dim,
            "bucket": bucket,
            "n": int(len(g)),
            "m89_mae": base["mae"],
            "m89_rmse": base["rmse"],
            "m89_bias": base["bias"],
            "opportunity_oracle_mae": opp["mae"],
            "opportunity_mae_recovery": base["mae"] - opp["mae"],
            "efficiency_oracle_mae": eff["mae"],
            "efficiency_mae_recovery": base["mae"] - eff["mae"],
            "both_primitives_oracle_mae": both["mae"],
            "both_primitives_mae_recovery": base["mae"] - both["mae"],
            "opportunity_abs_mass_share": abs_mass["opportunity"] / denom if denom else np.nan,
            "efficiency_abs_mass_share": abs_mass["efficiency"] / denom if denom else np.nan,
            "nonfactor_abs_mass_share": abs_mass["nonfactor_residual"] / denom if denom else np.nan,
            "dominant_opportunity_rows": int(dom.get("opportunity", 0)),
            "dominant_efficiency_rows": int(dom.get("efficiency", 0)),
            "dominant_nonfactor_rows": int(dom.get("nonfactor_residual", 0)),
            "cancellation_rate": float(g["cancellation_present"].mean()) if len(g) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    trace = pd.read_csv(a.trace, low_memory=False)
    logs = pd.read_csv(a.player_logs, low_memory=False)
    x = build_decomposition(trace, logs)
    summary = _summary_rows(x)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "qb_m89_opp_eff_player_trace.csv", index=False)
    summary.to_csv(a.out_dir / "qb_m89_opp_eff_summary.csv", index=False)

    pooled = summary.loc[
        summary["dimension"].eq("POOLED") & summary["bucket"].eq("ALL")
    ].iloc[0].to_dict()
    tail75 = x.loc[x["total_error"].abs().ge(75)]
    tail100 = x.loc[x["total_error"].abs().ge(100)]
    payload = {
        "disposition": "QB_M89_OPP_EFF_DECOMPOSITION_COMPLETE",
        "evaluation_seasons": list(TARGET_SEASONS),
        "rows": int(len(x)),
        "players": int(x["player_clean_key"].nunique()),
        "max_decomposition_identity_gap": float(x["decomposition_identity_gap"].abs().max()),
        "max_actual_trace_log_gap": float((x["actual_pass_yards"] - x["actual_pass_yards_logs"]).abs().max()),
        "sportsbook_inputs_used": False,
        "candidate_models_fit": 0,
        "production_changed": False,
        "pooled": pooled,
        "tail_75_rows": int(len(tail75)),
        "tail_75_dominant_components": tail75["dominant_component"].value_counts().to_dict(),
        "tail_100_rows": int(len(tail100)),
        "tail_100_dominant_components": tail100["dominant_component"].value_counts().to_dict(),
        "next_candidate_authorized": False,
        "anti_reinvention_ledger_binding": True,
    }
    (a.out_dir / "qb_m89_opp_eff_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
