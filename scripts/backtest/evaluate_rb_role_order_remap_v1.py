#!/usr/bin/env python3
"""Frozen RB role-order remap v1 diagnostic.

No model is fit. No sportsbook data is read. The candidate only reassigns the
already-existing STACK1 player carry means among timestamp-safe current RB/HB
depth ranks while preserving exact team-week opportunity mass.

Frozen protocol: docs/migrations/RB_ROLE_ORDER_REMAP_V1_PLAN.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 1393
EXPECTED_ATT = {
    "n": 1393,
    "mae": 3.482575936421331,
    "rmse": 4.741739937918672,
    "bias": -0.8627247216838725,
    "corr": 0.7353960735701257,
}
EXPECTED_YARDS = {
    "n": 1393,
    "mae": 20.4241632527228,
    "rmse": 30.06980647091156,
    "bias": -5.537127660834512,
    "corr": 0.6168472895165802,
}
EXPECTED_W1_YARDS_MAE = 20.09082962816558
EXPECTED_DEPTH_COVERAGE = 0.949749
RB_TRUE = {"RB", "HB"}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if x.empty:
        raise RuntimeError(f"zero-row {label}: {path}")
    return x


def _one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} below {root}, found {len(hits)}")
    return hits[0]


def _key(v) -> str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def _metric(y: pd.Series, p: pd.Series) -> dict:
    y = pd.to_numeric(y, errors="coerce")
    p = pd.to_numeric(p, errors="coerce")
    ok = y.notna() & p.notna()
    y = y.loc[ok].astype(float)
    p = p.loc[ok].astype(float)
    if not len(y):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
    e = p - y
    corr = float(p.corr(y)) if len(y) > 1 and p.nunique() > 1 and y.nunique() > 1 else np.nan
    return {
        "n": int(len(y)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "bias": float(e.mean()),
        "corr": corr,
    }


def _assert_close(name: str, got: float, expected: float, tol: float = 1e-9) -> None:
    if not np.isfinite(got) or abs(float(got) - float(expected)) > tol:
        raise RuntimeError(f"parent parity drift {name}: got={got} expected={expected}")


def _stack1_wide(stack1: pd.DataFrame) -> pd.DataFrame:
    s = stack1.copy()
    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s["week"] = pd.to_numeric(s["week"], errors="coerce")
    s = s.loc[s["season"].eq(2025) & s["week"].between(1, 18)].copy()
    s["player_clean_key"] = s.get("player_clean_key", s.get("player", "")).map(_key)
    s["position"] = s.get("position", "").fillna("").astype(str).str.upper().str.strip()
    s = s.loc[s["market"].astype(str).str.lower().isin(["rush_att", "rush_yards"])].copy()
    keys = ["season", "week", "team", "player_clean_key"]
    rows = []
    for key, g in s.groupby(keys, dropna=False, sort=False):
        out = dict(zip(keys, key))
        first = g.iloc[0]
        out["player"] = first.get("player", "")
        out["position"] = first.get("position", "")
        for market, prefix in [("rush_att", "att"), ("rush_yards", "yards")]:
            q = g.loc[g["market"].astype(str).str.lower().eq(market)]
            if len(q) != 1:
                raise RuntimeError(f"STACK1 duplicate/missing {market} row for {key}: {len(q)}")
            r = q.iloc[0]
            out[f"stack_{prefix}"] = pd.to_numeric(pd.Series([r.get("ensemble_2024_frozen")]), errors="coerce").iloc[0]
            out[f"actual_{prefix}"] = pd.to_numeric(pd.Series([r.get("actual")]), errors="coerce").iloc[0]
        rows.append(out)
    x = pd.DataFrame(rows)
    if len(x) != EXPECTED_ROWS:
        raise RuntimeError(f"STACK1 wide row drift: expected={EXPECTED_ROWS} got={len(x)}")
    if x.duplicated(keys).any():
        raise RuntimeError("STACK1 wide contains duplicate player-team-week rows")
    return x


def _parent_parity(x: pd.DataFrame) -> dict:
    att = _metric(x["actual_att"], x["stack_att"])
    yards = _metric(x["actual_yards"], x["stack_yards"])
    for k in ["mae", "rmse", "bias", "corr"]:
        _assert_close(f"rush_att_{k}", att[k], EXPECTED_ATT[k])
        _assert_close(f"rush_yards_{k}", yards[k], EXPECTED_YARDS[k])
    if att["n"] != EXPECTED_ATT["n"] or yards["n"] != EXPECTED_YARDS["n"]:
        raise RuntimeError(f"STACK1 parent n drift att={att['n']} yards={yards['n']}")
    w1 = x.loc[x["week"].eq(1)]
    w1_y = _metric(w1["actual_yards"], w1["stack_yards"])
    _assert_close("week1_rush_yards_mae", w1_y["mae"], EXPECTED_W1_YARDS_MAE)
    return {"rush_att": att, "rush_yards": yards, "week1_rush_yards": w1_y}


def _merge_depth(stack: pd.DataFrame, stack2: pd.DataFrame, coverage: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = stack2.copy()
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["week"] = pd.to_numeric(d["week"], errors="coerce")
    d = d.loc[d["season"].eq(2025) & d["week"].between(1, 18)].copy()
    d["player_clean_key"] = d.get("player_clean_key", d.get("player", "")).map(_key)
    keep = [
        "season", "week", "team", "player_clean_key", "depth_rank", "depth_slot",
        "depth_present", "depth_slot_rb", "depth_slot_fb", "stack_att", "stack_yards",
    ]
    keep = [c for c in keep if c in d.columns]
    d = d[keep].drop_duplicates(["season", "week", "team", "player_clean_key"], keep="last")
    x = stack.merge(d, on=["season", "week", "team", "player_clean_key"], how="left", validate="one_to_one", suffixes=("", "_stack2"))
    if len(x) != EXPECTED_ROWS:
        raise RuntimeError(f"depth merge row drift: {len(x)}")
    for c in ["stack_att_stack2", "stack_yards_stack2"]:
        if c in x.columns:
            parent = "stack_att" if "att" in c else "stack_yards"
            a = pd.to_numeric(x[parent], errors="coerce")
            b = pd.to_numeric(x[c], errors="coerce")
            ok = a.notna() & b.notna()
            max_diff = float((a.loc[ok] - b.loc[ok]).abs().max()) if int(ok.sum()) else np.inf
            if max_diff > 1e-9 or int(ok.sum()) != EXPECTED_ROWS:
                raise RuntimeError(f"STACK2/STACK1 row parity drift {c}: matched={int(ok.sum())} max_diff={max_diff}")
    depth_present = pd.to_numeric(x.get("depth_present"), errors="coerce").fillna(0.0)
    depth_coverage = float(depth_present.gt(0).mean())
    if "depth_coverage" not in coverage.columns:
        raise RuntimeError("STACK2 coverage artifact missing depth_coverage")
    inherited = float(pd.to_numeric(coverage["depth_coverage"], errors="coerce").iloc[0])
    _assert_close("inherited_depth_coverage", inherited, EXPECTED_DEPTH_COVERAGE, tol=1e-6)
    _assert_close("merged_depth_coverage", depth_coverage, inherited, tol=1e-6)
    return x, {
        "depth_coverage": depth_coverage,
        "inherited_timestamp_contract": "STRICT_PRE_KICKOFF_CANONICAL_STACK2_2025",
        "timestamp_violations": 0,
        "stack2_stack1_row_parity": True,
    }


def _apply_role_order(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = x.copy()
    out["position"] = out["position"].fillna("").astype(str).str.upper().str.strip()
    out["depth_rank"] = pd.to_numeric(out.get("depth_rank"), errors="coerce")
    out["role_order_att"] = pd.to_numeric(out["stack_att"], errors="coerce")
    out["role_order_changed"] = 0
    team_rows = []

    for (season, week, team), g in out.groupby(["season", "week", "team"], sort=True):
        idx = g.index
        eligible = g["position"].isin(RB_TRUE) & g["depth_rank"].notna()
        e = g.loc[eligible].copy()
        before_sum = float(pd.to_numeric(g["stack_att"], errors="coerce").sum())
        eligible_before = float(pd.to_numeric(e["stack_att"], errors="coerce").sum()) if len(e) else 0.0
        changed = 0
        if len(e) >= 2:
            e["_stack"] = pd.to_numeric(e["stack_att"], errors="coerce")
            e["_key"] = e["player_clean_key"].astype(str)
            order = e.sort_values(["depth_rank", "_stack", "_key"], ascending=[True, False, True], kind="stable").index
            values = np.sort(e["_stack"].to_numpy(dtype=float))[::-1]
            out.loc[order, "role_order_att"] = values
            delta = np.abs(out.loc[order, "role_order_att"].to_numpy(dtype=float) - e.loc[order, "_stack"].to_numpy(dtype=float))
            out.loc[order, "role_order_changed"] = (delta > 1e-12).astype(int)
            changed = int((delta > 1e-12).sum())
        after_sum = float(pd.to_numeric(out.loc[idx, "role_order_att"], errors="coerce").sum())
        team_rows.append({
            "season": int(season),
            "week": int(week),
            "team": team,
            "rows": int(len(g)),
            "eligible_rb_hb_rows": int(eligible.sum()),
            "changed_rows": changed,
            "baseline_team_player_carries": before_sum,
            "candidate_team_player_carries": after_sum,
            "team_mass_delta": after_sum - before_sum,
            "eligible_mass_before": eligible_before,
        })

    out["stack_ypc"] = np.where(
        pd.to_numeric(out["stack_att"], errors="coerce").gt(0.20),
        pd.to_numeric(out["stack_yards"], errors="coerce") / pd.to_numeric(out["stack_att"], errors="coerce"),
        np.nan,
    )
    usable_eff = pd.to_numeric(out["stack_ypc"], errors="coerce").notna() & np.isfinite(pd.to_numeric(out["stack_ypc"], errors="coerce"))
    out["role_order_yards"] = pd.to_numeric(out["stack_yards"], errors="coerce")
    out.loc[usable_eff, "role_order_yards"] = (
        pd.to_numeric(out.loc[usable_eff, "role_order_att"], errors="coerce")
        * pd.to_numeric(out.loc[usable_eff, "stack_ypc"], errors="coerce")
    )
    out["ypc_original_retained"] = (~usable_eff).astype(int)
    return out, pd.DataFrame(team_rows)


def _rows_for_slice(x: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "ALL_RB":
        return x
    if name == "WEEK1":
        return x.loc[x["week"].eq(1)]
    if name == "W2_18":
        return x.loc[x["week"].ge(2)]
    if name == "W13_18":
        return x.loc[x["week"].ge(13)]
    if name in {"RB1", "RB2", "RB3"}:
        rank = int(name[-1])
        return x.loc[x["position"].isin(RB_TRUE) & x["depth_rank"].eq(rank)]
    if name == "BASELINE_ROLE_MISALIGNED":
        return x.loc[x["baseline_role_aligned"].eq(0)]
    if name == "BASELINE_ROLE_ALIGNED":
        return x.loc[x["baseline_role_aligned"].eq(1)]
    raise RuntimeError(f"unknown slice {name}")


def _add_alignment(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    out["baseline_role_aligned"] = np.nan
    for _, g in out.groupby(["season", "week", "team"], sort=False):
        rb = g.loc[g["position"].isin(RB_TRUE) & g["depth_rank"].notna()].copy()
        if rb.empty or not rb["depth_rank"].eq(1).any():
            continue
        max_att = pd.to_numeric(rb["stack_att"], errors="coerce").max()
        leaders = rb.loc[np.isclose(pd.to_numeric(rb["stack_att"], errors="coerce"), max_att)]
        aligned = int(leaders["depth_rank"].eq(1).any())
        out.loc[g.index, "baseline_role_aligned"] = aligned
    return out


def _score(x: pd.DataFrame, team_trace: pd.DataFrame, source: dict) -> tuple[pd.DataFrame, dict]:
    x = _add_alignment(x)
    rows = []
    slices = ["ALL_RB", "WEEK1", "W2_18", "W13_18", "RB1", "RB2", "RB3", "BASELINE_ROLE_MISALIGNED", "BASELINE_ROLE_ALIGNED"]
    for name in slices:
        g = _rows_for_slice(x, name)
        for market, actual, base, cand in [
            ("rush_att", "actual_att", "stack_att", "role_order_att"),
            ("rush_yards", "actual_yards", "stack_yards", "role_order_yards"),
        ]:
            b = _metric(g[actual], g[base])
            c = _metric(g[actual], g[cand])
            rows.append({
                "slice": name,
                "market": market,
                "n": b["n"],
                "baseline_mae": b["mae"],
                "candidate_mae": c["mae"],
                "mae_delta_candidate_minus_baseline": c["mae"] - b["mae"],
                "mae_relative_improvement": (b["mae"] - c["mae"]) / b["mae"] if np.isfinite(b["mae"]) and b["mae"] > 0 else np.nan,
                "baseline_rmse": b["rmse"],
                "candidate_rmse": c["rmse"],
                "baseline_bias": b["bias"],
                "candidate_bias": c["bias"],
                "baseline_corr": b["corr"],
                "candidate_corr": c["corr"],
            })
    metrics = pd.DataFrame(rows)

    def row(slice_name: str, market: str) -> pd.Series:
        q = metrics.loc[metrics["slice"].eq(slice_name) & metrics["market"].eq(market)]
        if len(q) != 1:
            raise RuntimeError(f"missing metric row {slice_name}/{market}")
        return q.iloc[0]

    all_att = row("ALL_RB", "rush_att")
    w1_att = row("WEEK1", "rush_att")
    w2_att = row("W2_18", "rush_att")
    w13_att = row("W13_18", "rush_att")
    all_y = row("ALL_RB", "rush_yards")
    w1_y = row("WEEK1", "rush_yards")
    role_rows = [row(r, "rush_att") for r in ["RB1", "RB2", "RB3"]]
    role_improved = int(sum(float(r["candidate_mae"]) < float(r["baseline_mae"]) for r in role_rows))
    role_worst_delta = max(float(r["mae_delta_candidate_minus_baseline"]) for r in role_rows)
    max_mass_delta = float(team_trace["team_mass_delta"].abs().max()) if len(team_trace) else np.inf

    integrity = bool(
        len(x) == EXPECTED_ROWS
        and source["depth_coverage"] >= 0.90
        and int(source["timestamp_violations"]) == 0
        and max_mass_delta <= 1e-10
    )
    gates = {
        "integrity_source_gate": integrity,
        "team_mass_gate": max_mass_delta <= 1e-10,
        "overall_att_mae_improve_ge_1pct": float(all_att["mae_relative_improvement"]) >= 0.01,
        "week1_att_mae_improves": float(w1_att["candidate_mae"]) < float(w1_att["baseline_mae"]),
        "w2_18_att_mae_improves": float(w2_att["candidate_mae"]) < float(w2_att["baseline_mae"]),
        "w13_18_att_mae_improves": float(w13_att["candidate_mae"]) < float(w13_att["baseline_mae"]),
        "at_least_two_rb_role_slices_improve": role_improved >= 2,
        "no_rb1_rb2_rb3_slice_worsens_gt_0p10": role_worst_delta <= 0.10,
        "overall_yards_mae_improves": float(all_y["candidate_mae"]) < float(all_y["baseline_mae"]),
        "week1_yards_noninferior_0p10": float(w1_y["candidate_mae"]) <= float(w1_y["baseline_mae"]) + 0.10,
        "overall_abs_att_bias_noninferior_0p10": abs(float(all_att["candidate_bias"])) <= abs(float(all_att["baseline_bias"])) + 0.10,
    }
    if not integrity:
        disposition = "MECHANICAL_OR_SOURCE_FAILURE"
    elif all(gates.values()):
        disposition = "ROLE_ORDER_REMAP_V1_ACTIONABLE"
    else:
        disposition = "ROLE_ORDER_REMAP_V1_NOT_ACTIONABLE"

    summary = {
        "migration": "RB_ROLE_ORDER_REMAP_V1",
        "candidate": "ROLE_ORDER_REMAP_V1",
        "season": 2025,
        "rows": int(len(x)),
        "depth_coverage": float(source["depth_coverage"]),
        "timestamp_violations": int(source["timestamp_violations"]),
        "max_abs_team_week_carry_mass_delta": max_mass_delta,
        "changed_player_rows": int(pd.to_numeric(x["role_order_changed"], errors="coerce").fillna(0).sum()),
        "ypc_original_retained_rows": int(pd.to_numeric(x["ypc_original_retained"], errors="coerce").fillna(0).sum()),
        "rb1_rb2_rb3_improved_count": role_improved,
        "rb1_rb2_rb3_worst_mae_delta": role_worst_delta,
        "model_fitting_used": False,
        "sportsbook_inputs_used": False,
        "production_changed": False,
        "gates": gates,
        "disposition": disposition,
    }
    return metrics, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack1-root", type=Path, required=True)
    ap.add_argument("--stack2-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_role_order_remap_v1"))
    a = ap.parse_args()

    stack1 = _read(_one(a.stack1_root, "stack1_2025_rb_trace.csv"), "STACK1 trace")
    stack2 = _read(_one(a.stack2_root, "stack2_2025_casebook.csv"), "STACK2 casebook")
    coverage = _read(_one(a.stack2_root, "stack2_coverage.csv"), "STACK2 coverage")

    wide = _stack1_wide(stack1)
    parent = _parent_parity(wide)
    merged, source = _merge_depth(wide, stack2, coverage)
    candidate, team_trace = _apply_role_order(merged)
    metrics, summary = _score(candidate, team_trace, source)
    summary["parent_check"] = parent

    a.out_dir.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(a.out_dir / "rb_role_order_remap_v1_predictions.csv", index=False)
    team_trace.to_csv(a.out_dir / "rb_role_order_remap_v1_team_trace.csv", index=False)
    metrics.to_csv(a.out_dir / "rb_role_order_remap_v1_metrics.csv", index=False)
    (a.out_dir / "rb_role_order_remap_v1_result.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("=== RB ROLE ORDER REMAP V1 METRICS ===")
    print(metrics.to_string(index=False))
    print("=== RB ROLE ORDER REMAP V1 RESULT ===")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["disposition"] == "MECHANICAL_OR_SOURCE_FAILURE":
        raise RuntimeError("RB role-order remap v1 mechanical/source integrity failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
