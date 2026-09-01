#!/usr/bin/env python3
"""RB-ENV1: no-fit pregame environment x runner-quality outcome atlas.

This is a diagnostic only:
- uses frozen M95B pregame feature trace,
- uses frozen M95C out-of-sample predictions for 2024/2025,
- fits no new model and searches no thresholds,
- postgame outcomes are used only to measure correlation/exception cases.

Primary environment grade:
    role_plus_environment rush-yard prediction - role_baseline prediction
from the frozen M95C temporal test. This isolates the incremental pregame
environment contribution from the role baseline without inventing new weights.

Primary runner-quality grade:
    mean within-season percentile of the frozen M95B
    off_player_efficiency_score and off_player_explosive_score.
These are built only from information available before the target game.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} under {root}; found {len(hits)}")
    return hits[0]


def _corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    z = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"),
                      "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(z) < 3 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return np.nan
    return float(z.a.corr(z.b, method=method))


def _bucket_quartile(pct: pd.Series, low: str, mid: str, high: str) -> pd.Series:
    return pd.cut(
        pct,
        bins=[-1e-9, 0.25, 0.75, 1.0000001],
        labels=[low, mid, high],
        include_lowest=True,
    )


def _summarize(g: pd.DataFrame) -> dict:
    ypc8 = g.loc[pd.to_numeric(g.actual_carries, errors="coerce").ge(8), "actual_ypc"]
    return {
        "n": int(len(g)),
        "actual_carries_mean": float(pd.to_numeric(g.actual_carries, errors="coerce").mean()),
        "actual_rush_yards_mean": float(pd.to_numeric(g.actual_rush_yards, errors="coerce").mean()),
        "actual_ypc_8plus_mean": float(pd.to_numeric(ypc8, errors="coerce").mean()) if len(ypc8) else np.nan,
        "role_baseline_mean": float(pd.to_numeric(g.role_baseline, errors="coerce").mean()),
        "baseline_residual_mean": float(pd.to_numeric(g.baseline_residual, errors="coerce").mean()),
        "rush_75plus_rate": float(g.actual_75plus.mean()),
        "rush_100plus_rate": float(g.actual_100plus.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m95b-root", type=Path, required=True)
    ap.add_argument("--m95c-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trace = pd.read_csv(_find_one(args.m95b_root, "m95b_rb_matchup_trace.csv"), low_memory=False)
    pred = pd.read_csv(_find_one(args.m95c_root, "m95c_prediction_trace.csv"), low_memory=False)

    need_trace = {
        "season","week","team","opponent","player","actual_carries","actual_rush_yards",
        "off_player_efficiency_score","off_player_explosive_score",
    }
    missing = sorted(need_trace - set(trace.columns))
    if missing:
        raise RuntimeError(f"M95B trace missing required columns: {missing}")

    ry = pred.loc[pred.target.astype(str).eq("rush_yards")].copy()
    fams = {"role_baseline","role_plus_environment"}
    ry = ry.loc[ry.family.astype(str).isin(fams)].copy()
    wide = ry.pivot_table(
        index=["row_index","test_season"], columns="family", values="prediction", aggfunc="first"
    ).reset_index()
    missing_fam = fams - set(wide.columns)
    if missing_fam:
        raise RuntimeError(f"M95C prediction trace missing families: {sorted(missing_fam)}")

    t = trace.reset_index().rename(columns={"index":"row_index"})
    z = wide.merge(t, on="row_index", how="left", validate="one_to_one")
    z = z.loc[z.test_season.isin([2024,2025])].copy()
    if len(z) != 2580:
        raise RuntimeError(f"expected 2580 frozen 2024/2025 OOF rows, got {len(z)}")
    if not (pd.to_numeric(z.season, errors="coerce") == pd.to_numeric(z.test_season, errors="coerce")).all():
        raise RuntimeError("row_index/test_season mismatch between frozen M95B/M95C sources")

    z["env_delta_yards"] = pd.to_numeric(z.role_plus_environment, errors="coerce") - pd.to_numeric(z.role_baseline, errors="coerce")
    z["baseline_residual"] = pd.to_numeric(z.actual_rush_yards, errors="coerce") - pd.to_numeric(z.role_baseline, errors="coerce")
    z["actual_ypc"] = np.where(
        pd.to_numeric(z.actual_carries, errors="coerce").gt(0),
        pd.to_numeric(z.actual_rush_yards, errors="coerce") / pd.to_numeric(z.actual_carries, errors="coerce"),
        np.nan,
    )
    z["actual_75plus"] = pd.to_numeric(z.actual_rush_yards, errors="coerce").ge(75)
    z["actual_100plus"] = pd.to_numeric(z.actual_rush_yards, errors="coerce").ge(100)

    # Within-test-season ranking prevents cross-season scale drift from defining the categories.
    z["env_pct"] = z.groupby("test_season")["env_delta_yards"].rank(pct=True, method="average")
    z["eff_pct"] = z.groupby("test_season")["off_player_efficiency_score"].rank(pct=True, method="average")
    z["explosive_pct"] = z.groupby("test_season")["off_player_explosive_score"].rank(pct=True, method="average")
    z["player_quality_pct"] = z[["eff_pct","explosive_pct"]].mean(axis=1, skipna=True)

    z["environment_bucket"] = _bucket_quartile(z.env_pct, "BAD_SPOT", "NEUTRAL_SPOT", "GOOD_SPOT")
    z["runner_quality_bucket"] = _bucket_quartile(z.player_quality_pct, "WEAK_RB", "MID_RB", "STRONG_RB")

    # Correlation atlas: raw yards, residual beyond role baseline, and efficiency for meaningful carry samples.
    corr_rows = []
    for season_label, g in [("POOLED_2024_2025", z)] + [(str(s), q) for s,q in z.groupby("test_season")]:
        for signal in ["env_delta_yards","player_quality_pct","off_player_efficiency_score","off_player_explosive_score"]:
            for outcome in ["actual_rush_yards","baseline_residual"]:
                corr_rows.append({
                    "season": season_label, "signal": signal, "outcome": outcome,
                    "n": int(pd.DataFrame({"x":g[signal],"y":g[outcome]}).dropna().shape[0]),
                    "pearson": _corr(g[signal], g[outcome], "pearson"),
                    "spearman": _corr(g[signal], g[outcome], "spearman"),
                })
            q = g.loc[pd.to_numeric(g.actual_carries, errors="coerce").ge(8)]
            corr_rows.append({
                "season": season_label, "signal": signal, "outcome": "actual_ypc_8plus",
                "n": int(pd.DataFrame({"x":q[signal],"y":q.actual_ypc}).dropna().shape[0]),
                "pearson": _corr(q[signal], q.actual_ypc, "pearson"),
                "spearman": _corr(q[signal], q.actual_ypc, "spearman"),
            })
    correlations = pd.DataFrame(corr_rows)

    # Environment alone.
    env_rows = []
    for season_label, g in [("POOLED_2024_2025", z)] + [(str(s), q) for s,q in z.groupby("test_season")]:
        for bucket, q in g.groupby("environment_bucket", observed=False):
            env_rows.append({"season":season_label, "environment_bucket":str(bucket), **_summarize(q)})
    env_summary = pd.DataFrame(env_rows)

    # Environment x quality quadrants.
    quad_rows = []
    for season_label, g in [("POOLED_2024_2025", z)] + [(str(s), q) for s,q in z.groupby("test_season")]:
        for (eb,qb), q in g.groupby(["environment_bucket","runner_quality_bucket"], observed=False):
            quad_rows.append({
                "season":season_label, "environment_bucket":str(eb), "runner_quality_bucket":str(qb),
                **_summarize(q),
            })
    quadrants = pd.DataFrame(quad_rows)

    # Postgame workload diagnostic: does environment still line up with output within similar realized workload?
    c = pd.to_numeric(z.actual_carries, errors="coerce")
    z["actual_carry_bucket"] = pd.cut(
        c, bins=[-0.1,5,10,14,19,np.inf],
        labels=["0_5","6_10","11_14","15_19","20_plus"], include_lowest=True
    )
    carry_rows = []
    for season_label, g in [("POOLED_2024_2025", z)] + [(str(s), q) for s,q in z.groupby("test_season")]:
        for (cb,eb), q in g.groupby(["actual_carry_bucket","environment_bucket"], observed=False):
            if q.empty:
                continue
            carry_rows.append({
                "season":season_label, "actual_carry_bucket":str(cb), "environment_bucket":str(eb),
                **_summarize(q),
            })
    carry_summary = pd.DataFrame(carry_rows)

    # Exceptions. Thresholds are descriptive and fixed before looking at this run.
    def label_exception(r):
        if r.environment_bucket == "BAD_SPOT" and r.actual_rush_yards >= 75:
            return "BAD_SPOT_75PLUS"
        if r.environment_bucket == "BAD_SPOT" and r.baseline_residual >= 30:
            return "BAD_SPOT_BIG_OVER"
        if r.environment_bucket == "GOOD_SPOT" and r.baseline_residual <= -30:
            return "GOOD_SPOT_BIG_UNDER"
        if r.environment_bucket == "GOOD_SPOT" and r.actual_rush_yards <= 40:
            return "GOOD_SPOT_40_OR_LESS"
        return ""
    z["exception_type"] = z.apply(label_exception, axis=1)
    case_cols = [
        "test_season","week","team","opponent","player","actual_carries","actual_rush_yards","actual_ypc",
        "role_baseline","role_plus_environment","env_delta_yards","env_pct","environment_bucket",
        "off_player_efficiency_score","off_player_explosive_score","player_quality_pct","runner_quality_bucket",
        "baseline_residual","exception_type",
    ]
    cases = z.loc[z.exception_type.ne(""), case_cols].copy()
    cases["abs_baseline_residual"] = cases.baseline_residual.abs()
    cases = cases.sort_values(["test_season","exception_type","abs_baseline_residual"], ascending=[True,True,False])

    # Repeat offenders / repeat environment-beaters by player, descriptive only.
    player_rows = []
    for (season, player), g in z.groupby(["test_season","player"]):
        bad = g.loc[g.environment_bucket.eq("BAD_SPOT")]
        good = g.loc[g.environment_bucket.eq("GOOD_SPOT")]
        if len(bad) >= 2:
            player_rows.append({
                "test_season":season,"player":player,"profile":"BAD_SPOT",
                "n":len(bad),"avg_yards":bad.actual_rush_yards.mean(),
                "avg_residual":bad.baseline_residual.mean(),"75plus_rate":bad.actual_75plus.mean(),
                "avg_quality_pct":bad.player_quality_pct.mean(),
            })
        if len(good) >= 2:
            player_rows.append({
                "test_season":season,"player":player,"profile":"GOOD_SPOT",
                "n":len(good),"avg_yards":good.actual_rush_yards.mean(),
                "avg_residual":good.baseline_residual.mean(),"75plus_rate":good.actual_75plus.mean(),
                "avg_quality_pct":good.player_quality_pct.mean(),
            })
    player_profiles = pd.DataFrame(player_rows)

    # Headline compact output.
    head = []
    for season_label, g in [("2024",z.loc[z.test_season.eq(2024)]),("2025",z.loc[z.test_season.eq(2025)]),("POOLED",z)]:
        bad = g.loc[g.environment_bucket.eq("BAD_SPOT")]
        good = g.loc[g.environment_bucket.eq("GOOD_SPOT")]
        head.append({
            "season":season_label,
            "n":len(g),
            "env_vs_actual_yards_pearson":_corr(g.env_delta_yards,g.actual_rush_yards),
            "env_vs_baseline_residual_pearson":_corr(g.env_delta_yards,g.baseline_residual),
            "env_vs_ypc8_pearson":_corr(g.loc[g.actual_carries.ge(8),"env_delta_yards"],g.loc[g.actual_carries.ge(8),"actual_ypc"]),
            "bad_spot_avg_yards":bad.actual_rush_yards.mean(),
            "good_spot_avg_yards":good.actual_rush_yards.mean(),
            "good_minus_bad_avg_yards":good.actual_rush_yards.mean()-bad.actual_rush_yards.mean(),
            "bad_spot_75_rate":bad.actual_75plus.mean(),
            "good_spot_75_rate":good.actual_75plus.mean(),
            "bad_spot_100_rate":bad.actual_100plus.mean(),
            "good_spot_100_rate":good.actual_100plus.mean(),
        })
    headline = pd.DataFrame(head)

    # Integrity.
    integrity = pd.DataFrame([{
        "fitted_models":0,"feature_search":0,"threshold_search":0,"sportsbook_inputs":0,
        "postgame_used_for_grade_definition":0,
        "environment_grade":"frozen M95C role_plus_environment minus role_baseline OOF rush-yard prediction",
        "runner_quality_grade":"within-season mean percentile of frozen pregame M95B efficiency+explosive scores",
        "rows_2024":int(z.test_season.eq(2024).sum()),
        "rows_2025":int(z.test_season.eq(2025).sum()),
    }])

    z[case_cols + ["actual_75plus","actual_100plus"]].to_csv(args.out_dir/"env1_full_trace.csv", index=False)
    headline.to_csv(args.out_dir/"env1_headline.csv", index=False)
    correlations.to_csv(args.out_dir/"env1_correlations.csv", index=False)
    env_summary.to_csv(args.out_dir/"env1_environment_summary.csv", index=False)
    quadrants.to_csv(args.out_dir/"env1_environment_quality_quadrants.csv", index=False)
    carry_summary.to_csv(args.out_dir/"env1_environment_by_actual_carry_bucket.csv", index=False)
    cases.to_csv(args.out_dir/"env1_exception_casebook.csv", index=False)
    player_profiles.to_csv(args.out_dir/"env1_player_profiles.csv", index=False)
    integrity.to_csv(args.out_dir/"env1_integrity.csv", index=False)

    print("=== headline ===")
    print(headline.to_string(index=False))
    print("=== 2025 quadrants ===")
    print(quadrants.loc[quadrants.season.eq("2025")].to_string(index=False))
    print("=== 2025 largest bad-spot successes ===")
    print(cases.loc[(cases.test_season.eq(2025)) & cases.exception_type.str.startswith("BAD_SPOT")]
          .sort_values("actual_rush_yards",ascending=False).head(20).to_string(index=False))
    print("=== 2025 largest good-spot failures ===")
    print(cases.loc[(cases.test_season.eq(2025)) & cases.exception_type.str.startswith("GOOD_SPOT")]
          .sort_values("baseline_residual").head(20).to_string(index=False))


if __name__ == "__main__":
    main()
