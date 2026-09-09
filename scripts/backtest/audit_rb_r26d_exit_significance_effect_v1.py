#!/usr/bin/env python3
"""R26D no-refit diagnostic: departed-RB significance x frozen R26 effect.

This script does not fit or regenerate predictions. It joins immutable R26 V1
prediction rows to immutable R26C strict-prior exited-player source state and
applies the classifier/gates frozen in the R26D plan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = tuple(range(2020, 2026))
MEANINGFUL_TARGETS_PG = 1.0
MEANINGFUL_ROOM_SHARE = 0.25


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_one(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(paths)}: {paths}")
    return pd.read_csv(paths[0], low_memory=False)


def read_predictions(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("r26_predictions.csv"))
    if not paths:
        raise RuntimeError(f"found zero r26_predictions.csv under {root}")
    parts = [pd.read_csv(p, low_memory=False) for p in paths]
    out = pd.concat(parts, ignore_index=True, sort=False)
    keys = ["season", "week", "team", "player_clean_key"]
    missing = [c for c in keys if c not in out.columns]
    if missing:
        raise RuntimeError(f"R26 prediction schema missing keys: {missing}")
    dup = out.duplicated(keys, keep=False)
    if dup.any():
        d = out.loc[dup, keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate R26 prediction keys across parent artifact: {d}")
    return out


def metric(actual: pd.Series, pred: pd.Series) -> dict:
    a = num(actual).to_numpy(float)
    p = num(pred).to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(p)
    a = a[ok]; p = p[ok]
    if len(a) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "p90_abs_error": np.nan}
    e = p - a
    ae = np.abs(e)
    return {
        "n": int(len(a)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(np.mean(e)),
        "p90_abs_error": float(np.quantile(ae, 0.90)),
    }


def effect_metrics(g: pd.DataFrame, market: str) -> dict:
    b = metric(g[f"actual_{market}"], g[f"baseline_{market}"])
    c = metric(g[f"actual_{market}"], g[f"candidate_{market}"])
    delta = float(c["mae"] - b["mae"]) if np.isfinite(b["mae"]) and np.isfinite(c["mae"]) else np.nan
    rel = float(c["mae"] / b["mae"] - 1.0) if np.isfinite(b["mae"]) and b["mae"] > 1e-12 else np.nan
    return {
        "n": int(min(b["n"], c["n"])),
        "baseline_mae": b["mae"], "candidate_mae": c["mae"], "mae_delta": delta, "relative_mae_change": rel,
        "baseline_rmse": b["rmse"], "candidate_rmse": c["rmse"],
        "baseline_bias": b["bias"], "candidate_bias": c["bias"],
        "baseline_p90_abs_error": b["p90_abs_error"], "candidate_p90_abs_error": c["p90_abs_error"],
    }


def target_bin(v: float, history: bool) -> str:
    if not history or not np.isfinite(v):
        return "NO_PRIOR_HISTORY"
    if v == 0:
        return "0"
    if v <= 1:
        return "0_to_1"
    if v <= 2:
        return "1_to_2"
    return ">2"


def share_bin(v: float, history: bool) -> str:
    if not history or not np.isfinite(v):
        return "NO_PRIOR_HISTORY"
    if v < .10:
        return "<0.10"
    if v < .25:
        return "0.10_to_0.25"
    if v < .50:
        return "0.25_to_0.50"
    return ">=0.50"


def summarize(g: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    if not dimensions:
        groups = [((), g)]
    else:
        groups = g.groupby(dimensions, dropna=False, sort=True)
    for key, x in groups:
        if not isinstance(key, tuple):
            key = (key,)
        prefix = dict(zip(dimensions, key))
        for market in ("receptions", "targets"):
            rows.append({**prefix, "market": market, **effect_metrics(x, market)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26-root", type=Path, required=True)
    ap.add_argument("--r26c-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    pred = read_predictions(a.r26_root)
    src = read_one(a.r26c_root, "r26c_vacancy_teamweek_aggregate.csv")
    source_disp = json.loads(next(a.r26c_root.rglob("r26c_source_disposition.json")).read_text())
    if source_disp.get("disposition") != "EXIT_SIGNIFICANCE_SOURCE_READY":
        raise RuntimeError(f"R26C source is not READY: {source_disp.get('disposition')}")

    for c in ("season", "week"):
        pred[c] = num(pred[c]).astype(int)
        src[c] = num(src[c]).astype(int)
    pred["team"] = pred.team.astype(str)
    src["team"] = src.team.astype(str)
    pred = pred.loc[pred.season.isin(SEASONS)].copy()
    src = src.loc[src.season.isin(SEASONS)].copy()

    # Frozen R26D primary population: vacancy-active same-team incumbents only.
    inc = pred.loc[num(pred.vacancy_active).eq(1) & num(pred.continuing_same_team).eq(1)].copy()
    if inc.empty:
        raise RuntimeError("R26D parent contains zero vacancy incumbents")

    src_keys = ["season", "week", "team"]
    if src.duplicated(src_keys).any():
        raise RuntimeError("R26C team-week aggregate is not unique")
    joined = inc.merge(src, on=src_keys, how="left", validate="many_to_one", indicator=True, suffixes=("", "_src"))
    join_rate = float(joined._merge.eq("both").mean())
    joined = joined.loc[joined._merge.eq("both")].drop(columns=["_merge"]).copy()

    hist = num(joined.any_exit_positive_history).fillna(0).eq(1)
    mt = num(joined.max_exit_prior_targets_pg)
    ms = num(joined.max_exit_prior_rb_room_share)
    meaningful = hist & (mt.gt(MEANINGFUL_TARGETS_PG) | ms.ge(MEANINGFUL_ROOM_SHARE))
    low = hist & ~meaningful
    joined["exit_significance_class"] = np.select(
        [meaningful, low], ["MEANINGFUL_RECEIVING_EXIT", "LOW_RECEIVING_EXIT"], default="UNKNOWN_EXIT_HISTORY"
    )
    joined["role"] = np.where(num(joined.rb_rank).eq(1), "RB1", "RB2+")
    joined["phase"] = np.where(num(joined.week).eq(1), "W1", "W2+")
    joined["max_exit_prior_targets_pg_bin"] = [target_bin(v, h) for v, h in zip(mt, hist)]
    joined["max_exit_prior_rb_room_share_bin"] = [share_bin(v, h) for v, h in zip(ms, hist)]
    last8 = num(joined.max_exit_last8_targets_pg)
    joined["max_exit_last8_targets_pg_bin"] = [target_bin(v, h) for v, h in zip(last8, hist)]

    # Persist player-level deltas for auditability.
    for market in ("receptions", "targets"):
        joined[f"baseline_{market}_abs_error"] = (num(joined[f"baseline_{market}"]) - num(joined[f"actual_{market}"])).abs()
        joined[f"candidate_{market}_abs_error"] = (num(joined[f"candidate_{market}"]) - num(joined[f"actual_{market}"])).abs()
        joined[f"{market}_effect_delta_abs_error"] = joined[f"candidate_{market}_abs_error"] - joined[f"baseline_{market}_abs_error"]

    pooled = summarize(joined, ["exit_significance_class"])
    season = summarize(joined, ["season", "exit_significance_class"])
    role = summarize(joined, ["role", "exit_significance_class"])
    phase = summarize(joined, ["phase", "exit_significance_class"])
    fine_targets = summarize(joined, ["max_exit_prior_targets_pg_bin"])
    fine_share = summarize(joined, ["max_exit_prior_rb_room_share_bin"])
    fine_last8 = summarize(joined, ["max_exit_last8_targets_pg_bin"])

    def row(df: pd.DataFrame, **conds) -> pd.Series | None:
        q = df.copy()
        for k, v in conds.items():
            q = q.loc[q[k].eq(v)]
        return None if q.empty else q.iloc[0]

    m_pool = row(pooled, exit_significance_class="MEANINGFUL_RECEIVING_EXIT", market="receptions")
    l_pool = row(pooled, exit_significance_class="LOW_RECEIVING_EXIT", market="receptions")
    if m_pool is None or l_pool is None:
        core_support = False
        meaningful_n = 0 if m_pool is None else int(m_pool.n)
    else:
        meaningful_n = int(m_pool.n)
        core_support = True

    season_improve = 0
    season_n20 = 0
    ordering = 0
    qualified_ordering = 0
    early_order = False
    late_order = False
    season_gate_rows = []
    for s in SEASONS:
        mr = row(season, season=s, exit_significance_class="MEANINGFUL_RECEIVING_EXIT", market="receptions")
        lr = row(season, season=s, exit_significance_class="LOW_RECEIVING_EXIT", market="receptions")
        if mr is not None and int(mr.n) >= 20:
            season_n20 += 1
        if mr is not None and np.isfinite(mr.mae_delta) and float(mr.mae_delta) < 0:
            season_improve += 1
        qualified = mr is not None and lr is not None and int(mr.n) >= 15 and int(lr.n) >= 15
        ord_ok = bool(qualified and float(mr.mae_delta) < float(lr.mae_delta))
        if qualified:
            qualified_ordering += 1
            if ord_ok:
                ordering += 1
                if s <= 2022:
                    early_order = True
                if s >= 2024:
                    late_order = True
        season_gate_rows.append({
            "season": s,
            "meaningful_n": 0 if mr is None else int(mr.n),
            "meaningful_mae_delta": np.nan if mr is None else float(mr.mae_delta),
            "low_n": 0 if lr is None else int(lr.n),
            "low_mae_delta": np.nan if lr is None else float(lr.mae_delta),
            "qualified_ordering": bool(qualified),
            "meaningful_better_than_low": ord_ok,
        })

    rb1 = row(role, role="RB1", exit_significance_class="MEANINGFUL_RECEIVING_EXIT", market="receptions")
    rb2 = row(role, role="RB2+", exit_significance_class="MEANINGFUL_RECEIVING_EXIT", market="receptions")
    w1 = row(phase, phase="W1", exit_significance_class="MEANINGFUL_RECEIVING_EXIT", market="receptions")

    gates = {
        "01_meaningful_pooled_receptions_mae_improves": bool(core_support and float(m_pool.mae_delta) < 0),
        "02_meaningful_improves_at_least_4_of_6_seasons": season_improve >= 4,
        "03_meaningful_support_n200_and_n20_in_4_seasons": meaningful_n >= 200 and season_n20 >= 4,
        "04_low_is_less_favorable_than_meaningful_pooled": bool(core_support and float(l_pool.mae_delta) > float(m_pool.mae_delta)),
        "05_meaningful_vs_low_ordering_in_at_least_3_seasons": ordering >= 3,
        "06_ordering_not_2023_only_early_and_late_replication": early_order and late_order,
        "07_neither_role_meaningful_mae_worsens_over_1pct": bool(
            rb1 is not None and rb2 is not None
            and float(rb1.relative_mae_change) <= 0.01
            and float(rb2.relative_mae_change) <= 0.01
        ),
        "08_week1_meaningful_mae_worsens_no_more_than_0p5pct": bool(w1 is not None and float(w1.relative_mae_change) <= 0.005),
    }

    integrity = {
        "r26_prediction_rows": int(len(pred)),
        "vacancy_incumbent_parent_rows": int(len(inc)),
        "joined_vacancy_incumbent_rows": int(len(joined)),
        "join_rate": join_rate,
        "source_disposition": source_disp.get("disposition"),
        "r26_predictions_regenerated": False,
        "r9_refit": False,
        "sportsbook_inputs_added": 0,
        "production_parameters_changed": False,
        "receiving_yard_means_changed": False,
        "r22_changed": False,
        "primary_targets_pg_threshold": MEANINGFUL_TARGETS_PG,
        "primary_room_share_threshold": MEANINGFUL_ROOM_SHARE,
    }

    if join_rate < 0.995 or meaningful_n < 50 or l_pool is None:
        disposition = "EXIT_SIGNIFICANCE_EFFECT_INSUFFICIENT"
    elif all(gates.values()):
        disposition = "SIGNIFICANCE_ROUTER_HYPOTHESIS_SUPPORTED"
    else:
        disposition = "EXIT_SIGNIFICANCE_EFFECT_MIXED_NO_ROUTER"

    result = {
        "candidate": "RB_R26D_EXIT_SIGNIFICANCE_EFFECT_ATLAS_V1",
        "scientific_label": "NO_REFIT_RETROSPECTIVE_DIAGNOSTIC",
        "disposition": disposition,
        "child_candidate_design_authorized": disposition == "SIGNIFICANCE_ROUTER_HYPOTHESIS_SUPPORTED",
        "production_promotion_authorized": False,
        "prospective_shadow_authorized": False,
        "integrity": integrity,
        "replication_gates": gates,
        "replication_counts": {
            "meaningful_seasons_improved": season_improve,
            "meaningful_seasons_n_ge_20": season_n20,
            "qualified_meaningful_vs_low_season_comparisons": qualified_ordering,
            "meaningful_better_than_low_seasons": ordering,
            "early_ordering_present_2020_2022": early_order,
            "late_ordering_present_2024_2025": late_order,
        },
        "primary_pooled_receptions": {
            "meaningful": None if m_pool is None else m_pool.to_dict(),
            "low": None if l_pool is None else l_pool.to_dict(),
        },
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(a.out_dir / "r26d_joined_player_effects.csv", index=False)
    pooled.to_csv(a.out_dir / "r26d_pooled_significance_effect.csv", index=False)
    season.to_csv(a.out_dir / "r26d_season_significance_effect.csv", index=False)
    role.to_csv(a.out_dir / "r26d_role_significance_effect.csv", index=False)
    phase.to_csv(a.out_dir / "r26d_phase_significance_effect.csv", index=False)
    fine_targets.to_csv(a.out_dir / "r26d_prior_targets_bins.csv", index=False)
    fine_share.to_csv(a.out_dir / "r26d_prior_room_share_bins.csv", index=False)
    fine_last8.to_csv(a.out_dir / "r26d_last8_targets_bins.csv", index=False)
    pd.DataFrame(season_gate_rows).to_csv(a.out_dir / "r26d_season_replication_gates.csv", index=False)
    (a.out_dir / "r26d_disposition.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("=== pooled significance ===")
    print(pooled.to_csv(index=False))
    print("=== season replication ===")
    print(pd.DataFrame(season_gate_rows).to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
