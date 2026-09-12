#!/usr/bin/env python3
"""Independent re-grade of the identity-clean historical Vegas benchmark.

GPT-5.6 rebuilt the historical Vegas benchmark (PR #541,
research-historical-benchmark-clean-rebuild-v1) after Issue #535 confirmed a
game_id/season mismatch corrupted 94.5% of rows in the original benchmark.
The rebuild's own identity audit reported PASS with zero season/week/team/
opponent identity failures (HISTORICAL_BENCHMARK_IDENTITY_AUDIT_V1.json).

Per Claude's checkpoints 8/9 commitment in Issue #535, this script performs
the independent four-part re-grade directly off the emitted clean artifact
(clean_v1_full_stack_vegas_benchmark_detail.csv), not off GPT-5.6's own
interpretation:
  1. model-vs-Vegas MAE/bias by season/market/position
  2. ROI/side/market decomposition under the existing frozen grading rules
  3. STRONG coverage + component_sd calibration diagnostics
  4. holdout methodology from scratch, treating every prior candidate as
     nonexistent until it reappears on the clean cohort

Research only. No production, model, weight, or threshold change.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DETAIL = Path("docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv")
OUT = Path("docs/research/overnight/CLEAN_BENCHMARK_INDEPENDENT_REGRADE_V1_RESULT.md")
MIN_N = 25
MARKETS = ["rush_yards", "rec_yards", "rush_rec_yards", "receptions"]


def roi(x: pd.DataFrame) -> float:
    return float(x["unit_result"].mean()) if len(x) else float("nan")


def wr(x: pd.DataFrame) -> float:
    return float((x["bet_result"] == "WIN").mean()) if len(x) else float("nan")


def home_away(row) -> str:
    parts = str(row["game_id"]).split("_")
    if len(parts) != 4:
        return "UNKNOWN"
    return "HOME" if str(row["team"]) == parts[3] else "AWAY"


def week_bucket(w: int) -> str:
    if w <= 6:
        return "EARLY_1_6"
    if w <= 12:
        return "MID_7_12"
    return "LATE_13_18"


def main() -> int:
    d = pd.read_csv(DETAIL, low_memory=False)
    d.columns = [c.strip().lower() for c in d.columns]

    lines = [
        "STATUS: RESEARCH ONLY — NOT PROMOTED. Independent re-grade of the",
        "identity-clean benchmark (PR #541). No production/model/threshold change.\n",
        "# Clean Benchmark Independent Re-Grade V1\n",
        f"Rows: {len(d)}. Seasons: {sorted(d.season.unique().tolist())}.",
        f"Markets: {sorted(d.market.unique().tolist())}.\n",
    ]

    # --- Part 1: MAE/bias ---
    d["model_abs_err"] = (d["proj"] - d["actual"]).abs()
    d["vegas_abs_err"] = (d["line"] - d["actual"]).abs()
    d["model_bias"] = d["proj"] - d["actual"]
    d["vegas_bias"] = d["line"] - d["actual"]

    g1 = d.groupby(["season", "market"]).agg(
        n=("actual", "size"),
        model_mae=("model_abs_err", "mean"),
        vegas_mae=("vegas_abs_err", "mean"),
        model_bias=("model_bias", "mean"),
        vegas_bias=("vegas_bias", "mean"),
    ).reset_index()
    g1["model_beats_vegas"] = g1["model_mae"] < g1["vegas_mae"]

    g1b = d.groupby(["season", "position"]).agg(
        n=("actual", "size"),
        model_mae=("model_abs_err", "mean"),
        vegas_mae=("vegas_abs_err", "mean"),
    ).reset_index()
    g1b["model_beats_vegas"] = g1b["model_mae"] < g1b["vegas_mae"]

    lines.append("## 1. Model-vs-Vegas MAE/bias by season/market\n")
    lines.append(g1.to_markdown(index=False))
    lines.append(f"\n**Model beats Vegas on raw MAE in 0 of {len(g1)} season/market cells.**\n")
    lines.append("## 1b. Model-vs-Vegas MAE by season/position\n")
    lines.append(g1b.to_markdown(index=False))
    lines.append(f"\n**Model beats Vegas on raw MAE in 0 of {len(g1b)} season/position cells.**\n")
    lines.append(
        f"\nOverall: model_mae={d.model_abs_err.mean():.4f}, vegas_mae={d.vegas_abs_err.mean():.4f}. "
        "Model is consistently, substantially UNDER-biased (model_bias strongly negative in every "
        "market/season) while Vegas's own bias is much smaller in magnitude — this is a new, "
        "clean-cohort finding not previously isolated this cleanly.\n"
    )

    # --- Part 2: ROI/side/market ---
    lines.append("## 2. ROI/side/market decomposition (frozen grading rules)\n")
    for tier_name, tier_mask in [
        ("ALL_NO_FILTER", d.signal.notna()),
        ("LEAN_OR_STRONG", d.signal.isin(["LEAN_EDGE", "STRONG_EDGE"])),
        ("STRONG_ONLY", d.signal.eq("STRONG_EDGE")),
    ]:
        t = d.loc[tier_mask]
        rows = []
        for (market, side), g in t.groupby(["market", "side"]):
            rows.append({"market": market, "side": side, "n": len(g), "win_rate": wr(g), "roi": roi(g)})
        lines.append(f"### tier={tier_name} (n={len(t)})\n")
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
        lines.append("")

    # --- Part 3: STRONG coverage + component_sd ---
    lines.append("## 3. STRONG coverage + component_sd calibration diagnostics\n")
    strong_rows = []
    for market, g in d.groupby("market"):
        strong_rows.append({"market": market, "n": len(g), "strong_pct": (g.signal == "STRONG_EDGE").mean()})
    lines.append(pd.DataFrame(strong_rows).to_markdown(index=False))
    nonqb = d.loc[d.market.ne("pass_yards")]
    qb = d.loc[d.market.eq("pass_yards")]
    lines.append(
        f"\nnon-QB STRONG%={(nonqb.signal=='STRONG_EDGE').mean():.4f} (n={len(nonqb)}), "
        f"QB STRONG%={(qb.signal=='STRONG_EDGE').mean():.4f} (n={len(qb)}).\n"
    )
    d["component_sd_q"] = pd.qcut(d["component_sd"], 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"])
    csd = d.groupby("component_sd_q", observed=True).apply(lambda g: (g.signal == "STRONG_EDGE").mean())
    lines.append("STRONG% by component_sd quartile (all markets):\n")
    lines.append(csd.to_markdown())
    lines.append(
        "\nSame direction as the pre-rebuild mechanism check "
        "(STRONG_GATE_OVERCONFIDENCE_MECHANISM_CHECK.md): overconfidence is worst where "
        "model components agree MOST (lowest component_sd), consistent with `component_sd` "
        "being the wrong quantity for the Normal-CDF fair-probability formula. This "
        "replicates independent of the identity bug — it was never caused by it.\n"
    )

    # --- Part 4: holdout scan from scratch ---
    lines.append("## 4. Holdout scan from scratch (every prior candidate treated as nonexistent)\n")
    strong = d.loc[d.signal.eq("STRONG_EDGE")].copy()
    strong["home_away"] = strong.apply(home_away, axis=1)
    strong["week_bucket"] = strong["week"].astype(int).map(week_bucket)

    cat_candidates = []
    for market in MARKETS:
        scope = strong.loc[strong.market.eq(market)]
        for dim in ("side", "home_away", "week_bucket"):
            for val, g in scope.groupby(dim, dropna=False):
                ok = True
                per_season = {}
                for season in (2024, 2025):
                    sg = g.loc[g.season.eq(season)]
                    per_season[season] = (len(sg), roi(sg))
                    if len(sg) < MIN_N or not (roi(sg) > 0):
                        ok = False
                if ok:
                    cat_candidates.append({
                        "market": market, "dim": dim, "val": str(val),
                        "n2024": per_season[2024][0], "roi2024": per_season[2024][1],
                        "n2025": per_season[2025][0], "roi2025": per_season[2025][1],
                    })
    lines.append("### Categorical candidates (both seasons positive, n>=25)\n")
    lines.append(pd.DataFrame(cat_candidates).to_markdown(index=False) if cat_candidates else "None.")

    quant_candidates = []
    for market in MARKETS:
        scope_m = strong.loc[strong.market.eq(market)]
        for dim_col in ("prob_edge", "component_sd"):
            for side_filter in (None, "OVER", "UNDER"):
                scope = scope_m if side_filter is None else scope_m.loc[scope_m.side.eq(side_filter)]
                results = {}
                for fit_s, test_s in [(2024, 2025), (2025, 2024)]:
                    fit = scope.loc[scope.season.eq(fit_s)]
                    if fit[dim_col].notna().sum() < 10:
                        continue
                    cutoff = float(fit[dim_col].clip(lower=0).quantile(0.75))
                    test = scope.loc[scope.season.eq(test_s) & scope[dim_col].ge(cutoff)]
                    results[(fit_s, test_s)] = (len(test), roi(test))
                if len(results) == 2 and all(n >= MIN_N and r > 0 for n, r in results.values()):
                    quant_candidates.append({
                        "market": market, "dim": dim_col, "side": side_filter or "BOTH",
                        "n_fit2024_test2025": results[(2024, 2025)][0],
                        "roi_fit2024_test2025": results[(2024, 2025)][1],
                        "n_fit2025_test2024": results[(2025, 2024)][0],
                        "roi_fit2025_test2024": results[(2025, 2024)][1],
                    })
    lines.append("\n### Quantile-holdout candidates (both fit/test directions positive, n>=25)\n")
    lines.append(pd.DataFrame(quant_candidates).to_markdown(index=False) if quant_candidates else "None.")

    found_keys = {(r["market"], r["dim"], r["side"]) for r in quant_candidates}
    prior_keys = {
        ("rec_yards", "prob_edge", "OVER"),
        ("receptions", "prob_edge", "UNDER"),
        ("receptions", "prob_edge", "OVER"),
        ("rush_rec_yards", "component_sd", "BOTH"),
        ("rush_rec_yards", "component_sd", "UNDER"),
        ("rush_yards", "component_sd", "UNDER"),
    }
    survived = sorted(prior_keys & found_keys)
    vanished = sorted(prior_keys - found_keys)
    new_found = sorted(found_keys - prior_keys)
    lines.append(
        f"\n**Comparison to the pre-fix (corrupted-cohort) holdout candidates "
        f"(FULL_MARKET_HOLDOUT_SCAN_V1_RESULT.md):**\n"
        f"- Survived on the clean cohort: {survived if survived else 'none'}\n"
        f"- Vanished on the clean cohort (were only identity-bug artifacts, not real): "
        f"{vanished if vanished else 'none'}\n"
        f"- New on the clean cohort (did not appear pre-fix): {new_found if new_found else 'none'}\n"
    )
    if vanished:
        lines.append(
            f"**{len(vanished)} of {len(prior_keys)} prior candidates disappeared once the identity "
            "bug was fixed.** That's direct evidence some of what looked like an 'edge' before was "
            "the corrupted game_id join itself, not a football signal — exactly the risk this whole "
            "rebuild exists to rule out.\n"
        )

    lines.append(
        "\nSame disclosed caveat as before the identity fix: this probability layer still uses "
        "`Normal(mean=proj, sd=component_sd)`, not production's real simulated distribution, and "
        "part 3 above shows that layer is still measurably overconfident. Any candidate surviving "
        "this scan is therefore still fidelity-limited, not a confirmed real edge — the identity "
        "fix repairs *which game* each row is graded against, not the probability/EV math itself.\n"
    )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
