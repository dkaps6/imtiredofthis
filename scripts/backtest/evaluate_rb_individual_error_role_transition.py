#!/usr/bin/env python3
"""Frozen RB individual-error / role-transition diagnostic.

No model is fit. No sportsbook data is read. Exact STACK1 projections/outcomes
are merged to timestamp-safe STACK2 metadata using only canonical team aliases.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 1393
EXPECTED_DEPTH_COVERAGE = 0.949748743718593
TEAM_ALIAS = {"JAC": "JAX", "JAX": "JAX", "LA": "LAR", "LAR": "LAR"}
RB_TRUE = {"RB", "HB"}


def _read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if x.empty:
        raise RuntimeError(f"empty input {path}")
    return x


def _one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def _team(v) -> str:
    raw = str(v or "").strip().upper()
    return TEAM_ALIAS.get(raw, raw)


def _key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _metric(actual: pd.Series, pred: pd.Series) -> dict:
    a = _num(actual); p = _num(pred)
    ok = a.notna() & p.notna()
    a = a.loc[ok].astype(float); p = p.loc[ok].astype(float)
    e = p - a
    return {
        "n": int(len(e)),
        "mae": float(e.abs().mean()) if len(e) else np.nan,
        "rmse": float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan,
        "bias": float(e.mean()) if len(e) else np.nan,
    }


def build_stack1_wide(s: pd.DataFrame) -> pd.DataFrame:
    q = s.loc[
        _num(s["season"]).eq(2025)
        & _num(s["week"]).between(1, 18)
        & s["market"].astype(str).str.lower().isin(["rush_att", "rush_yards"])
    ].copy()
    q["team"] = q["team"].map(_team)
    q["player_key"] = q.get("player_clean_key", q.get("player", "")).map(_key)
    keys = ["season", "week", "team", "player_key"]
    rows = []
    for key, g in q.groupby(keys, sort=False, dropna=False):
        out = dict(zip(keys, key))
        out["player"] = g.iloc[0].get("player", "")
        out["position"] = str(g.iloc[0].get("position", "")).upper().strip()
        for market, suffix in [("rush_att", "att"), ("rush_yards", "yards")]:
            r = g.loc[g["market"].astype(str).str.lower().eq(market)]
            if len(r) != 1:
                raise RuntimeError(f"STACK1 duplicate/missing {market} for {key}: {len(r)}")
            rr = r.iloc[0]
            out[f"pred_{suffix}"] = float(pd.to_numeric(pd.Series([rr.get("ensemble_2024_frozen")]), errors="coerce").iloc[0])
            out[f"actual_{suffix}"] = float(pd.to_numeric(pd.Series([rr.get("actual")]), errors="coerce").iloc[0])
        rows.append(out)
    x = pd.DataFrame(rows)
    if len(x) != EXPECTED_ROWS:
        raise RuntimeError(f"STACK1 row drift expected={EXPECTED_ROWS} got={len(x)}")
    return x


def attach_metadata(stack: pd.DataFrame, s2: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    d = s2.loc[_num(s2["season"]).eq(2025) & _num(s2["week"]).between(1, 18)].copy()
    d["team"] = d["team"].map(_team)
    d["player_key"] = d.get("player_clean_key", d.get("player", "")).map(_key)
    meta_cols = [
        "season", "week", "team", "player_key", "depth_rank", "depth_present", "depth_slot",
        "prior_games", "same_team_last_game", "rookie_flag", "injured_comp_count",
        "injury_out_doubtful", "injury_questionable", "practice_dnp", "practice_limited",
        "roster_active", "roster_inactive",
    ]
    meta_cols = [c for c in meta_cols if c in d.columns]
    d = d[meta_cols].drop_duplicates(["season", "week", "team", "player_key"], keep="last")
    x = stack.merge(d, on=["season", "week", "team", "player_key"], how="left", validate="one_to_one", indicator=True)
    if len(x) != EXPECTED_ROWS or not x["_merge"].eq("both").all():
        raise RuntimeError(f"STACK1/STACK2 identity parity failed rows={len(x)} matched={int(x['_merge'].eq('both').sum())}")
    x = x.drop(columns=["_merge"])
    depth_cov = float(_num(x["depth_present"]).fillna(0).gt(0).mean())
    inherited = float(_num(coverage["depth_coverage"]).iloc[0])
    if abs(inherited - EXPECTED_DEPTH_COVERAGE) > 1e-9 or abs(depth_cov - inherited) > 1e-9:
        raise RuntimeError(f"depth coverage drift inherited={inherited} merged={depth_cov}")
    return x


def add_states(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    out["depth_rank"] = _num(out.get("depth_rank"))
    out["prior_games"] = _num(out.get("prior_games")).fillna(0)
    out["same_team_last_game"] = _num(out.get("same_team_last_game")).fillna(0)
    out["rookie_flag"] = _num(out.get("rookie_flag")).fillna(0)
    out["depth_present"] = _num(out.get("depth_present")).fillna(0)
    for c in ["injured_comp_count", "injury_out_doubtful", "injury_questionable", "practice_dnp", "practice_limited"]:
        out[c] = _num(out.get(c)).fillna(0)

    out["projected_carry_rank"] = np.nan
    for _, g in out.groupby(["season", "week", "team"], sort=False):
        rb = g.loc[g["position"].isin(RB_TRUE)].copy()
        if rb.empty:
            continue
        rb = rb.sort_values(["pred_att", "player_key"], ascending=[False, True], kind="stable")
        out.loc[rb.index, "projected_carry_rank"] = np.arange(1, len(rb) + 1, dtype=float)

    out["state_depth_vs_carry_order_mismatch"] = (
        out["position"].isin(RB_TRUE)
        & out["depth_rank"].notna()
        & out["projected_carry_rank"].notna()
        & out["depth_rank"].ne(out["projected_carry_rank"])
    ).astype(int)
    out["state_limited_prior_history"] = out["prior_games"].le(2).astype(int)
    out["state_no_prior_same_team_game"] = out["same_team_last_game"].ne(1).astype(int)
    out["state_rookie"] = out["rookie_flag"].eq(1).astype(int)
    out["state_injury_created_context"] = (
        out[["injured_comp_count", "injury_out_doubtful", "injury_questionable", "practice_dnp", "practice_limited"]]
        .gt(0).any(axis=1)
    ).astype(int)
    out["state_depth_present"] = out["depth_present"].gt(0).astype(int)
    out["state_mismatch_and_limited_prior"] = (
        out["state_depth_vs_carry_order_mismatch"].eq(1) & out["state_limited_prior_history"].eq(1)
    ).astype(int)
    out["state_mismatch_and_no_same_team"] = (
        out["state_depth_vs_carry_order_mismatch"].eq(1) & out["state_no_prior_same_team_game"].eq(1)
    ).astype(int)
    out["state_mismatch_and_rookie"] = (
        out["state_depth_vs_carry_order_mismatch"].eq(1) & out["state_rookie"].eq(1)
    ).astype(int)
    return out


def player_profiles(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (k, player), g in x.groupby(["player_key", "player"], sort=True):
        ea = _num(g["pred_att"]) - _num(g["actual_att"])
        ey = _num(g["pred_yards"]) - _num(g["actual_yards"])
        rows.append({
            "player_key": k,
            "player": player,
            "games": int(len(g)),
            "carry_mae": float(ea.abs().mean()),
            "carry_bias_projection_minus_actual": float(ea.mean()),
            "carry_median_abs_error": float(ea.abs().median()),
            "carry_p90_abs_error": float(np.quantile(ea.abs(), .90)),
            "rush_yards_mae": float(ey.abs().mean()),
            "rush_yards_bias_projection_minus_actual": float(ey.mean()),
            "rush_yards_miss30_rate": float(ey.abs().ge(30).mean()),
            "rush_yards_miss50_rate": float(ey.abs().ge(50).mean()),
        })
    return pd.DataFrame(rows)


def state_metrics(x: pd.DataFrame) -> pd.DataFrame:
    states = [c for c in x.columns if c.startswith("state_")]
    rows = []
    for state in states:
        for value in [1, 0]:
            g = x.loc[_num(x[state]).eq(value)].copy()
            att = _metric(g["actual_att"], g["pred_att"])
            yds = _metric(g["actual_yards"], g["pred_yards"])
            att_err = (_num(g["pred_att"]) - _num(g["actual_att"])).abs()
            yd_err = (_num(g["pred_yards"]) - _num(g["actual_yards"])).abs()
            rows.append({
                "state": state,
                "state_value": value,
                "rows": int(len(g)),
                "carry_mae": att["mae"],
                "carry_rmse": att["rmse"],
                "carry_bias_projection_minus_actual": att["bias"],
                "rush_yards_mae": yds["mae"],
                "rush_yards_rmse": yds["rmse"],
                "rush_yards_bias_projection_minus_actual": yds["bias"],
                "carry_abs_miss5_rate": float(att_err.ge(5).mean()) if len(g) else np.nan,
                "carry_abs_miss10_rate": float(att_err.ge(10).mean()) if len(g) else np.nan,
                "rush_yards_abs_miss20_rate": float(yd_err.ge(20).mean()) if len(g) else np.nan,
                "rush_yards_abs_miss40_rate": float(yd_err.ge(40).mean()) if len(g) else np.nan,
                "carry_residual_actual_minus_projection": float((_num(g["actual_att"]) - _num(g["pred_att"])).mean()) if len(g) else np.nan,
                "rush_yards_residual_actual_minus_projection": float((_num(g["actual_yards"]) - _num(g["pred_yards"])).mean()) if len(g) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack1-root", type=Path, required=True)
    ap.add_argument("--stack2-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_individual_error_role_transition"))
    a = ap.parse_args()

    s1 = _read(_one(a.stack1_root, "stack1_2025_rb_trace.csv"))
    s2 = _read(_one(a.stack2_root, "stack2_2025_casebook.csv"))
    coverage = _read(_one(a.stack2_root, "stack2_coverage.csv"))
    wide = build_stack1_wide(s1)
    x = attach_metadata(wide, s2, coverage)
    x = add_states(x)
    profiles = player_profiles(x)
    states = state_metrics(x)

    summary = {
        "migration": "RB_INDIVIDUAL_ERROR_ROLE_TRANSITION",
        "rows": int(len(x)),
        "players": int(len(profiles)),
        "depth_coverage": float(_num(x["depth_present"]).fillna(0).gt(0).mean()),
        "depth_vs_carry_order_mismatch_rows": int(x["state_depth_vs_carry_order_mismatch"].sum()),
        "limited_prior_history_rows": int(x["state_limited_prior_history"].sum()),
        "no_prior_same_team_rows": int(x["state_no_prior_same_team_game"].sum()),
        "rookie_rows": int(x["state_rookie"].sum()),
        "injury_created_context_rows": int(x["state_injury_created_context"].sum()),
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": "DIAGNOSTIC_ONLY_NO_PRODUCTION_CHANGE",
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "rb_role_transition_casebook.csv", index=False)
    profiles.sort_values(["games", "carry_mae"], ascending=[False, False]).to_csv(a.out_dir / "rb_individual_error_profiles.csv", index=False)
    states.to_csv(a.out_dir / "rb_role_transition_state_metrics.csv", index=False)
    (a.out_dir / "rb_individual_error_role_transition_result.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("=== RB INDIVIDUAL / ROLE-TRANSITION SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("=== STATE METRICS ===")
    print(states.to_string(index=False))
    print("=== HIGHEST-GAME INDIVIDUAL PROFILES ===")
    print(profiles.sort_values(["games", "carry_mae"], ascending=[False, False]).head(60).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
