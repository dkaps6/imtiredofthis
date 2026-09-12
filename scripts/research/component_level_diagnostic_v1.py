#!/usr/bin/env python3
"""Component-level diagnostic: does any single component (MC/ML/State) beat
Vegas on its own, or are all three individually mediocre and the blend is
just averaging weak pieces together?

Answers a question the clean-cohort re-grade (checkpoint 10, PR #542) left
open: it showed the BLENDED ensemble loses to Vegas everywhere, but never
isolated whether that's because no individual component is competitive, or
because a competitive component is being diluted by weak ones in the blend.

Runs entirely off the already-committed identity-clean cohort
(clean_v1_full_stack_vegas_benchmark_detail.csv) -- no new CI run, no new
data build. Also inspects the ensemble_weight_mc/ml/state columns already
recorded per row to see whether weighting is close to uniform or genuinely
differentiated, and computes an in-sample oracle ceiling (best possible
static per-market weighting in hindsight) purely as a diagnostic bound, not
a proposed change.

Research only. No production, model, weight, or threshold change.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DETAIL = Path("docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv")
OUT = Path("docs/research/overnight/COMPONENT_LEVEL_DIAGNOSTIC_V1_RESULT.md")
COMPONENTS = ["mc_proj", "ml_proj", "state_proj"]


def mae(pred: pd.Series, actual: pd.Series) -> float:
    return float((pred - actual).abs().mean())


def bias(pred: pd.Series, actual: pd.Series) -> float:
    return float((pred - actual).mean())


def oracle_static_weight_mae(d: pd.DataFrame) -> tuple[float, np.ndarray, int]:
    """Best possible static (non-negative, sum-to-1) blend of the three
    components fit in-sample on this exact data -- an upper bound on what
    reweighting alone could ever achieve, not a real out-of-sample estimate.
    """
    clean = d.dropna(subset=COMPONENTS + ["actual"])
    dropped = len(d) - len(clean)
    X = clean[COMPONENTS].to_numpy()
    y = clean["actual"].to_numpy()
    best_mae, best_w = np.inf, None
    # Coarse grid search over the simplex (weights >= 0, sum to 1) -- exact
    # enough for a diagnostic bound, not meant to be a fitted model.
    step = 0.05
    grid = np.arange(0, 1 + step, step)
    for w_mc in grid:
        for w_ml in grid:
            w_state = 1 - w_mc - w_ml
            if w_state < -1e-9 or w_state > 1 + 1e-9:
                continue
            w_state = max(0.0, w_state)
            pred = w_mc * X[:, 0] + w_ml * X[:, 1] + w_state * X[:, 2]
            m = float(np.abs(pred - y).mean())
            if m < best_mae:
                best_mae, best_w = m, np.array([w_mc, w_ml, w_state])
    return best_mae, best_w, dropped


def main() -> int:
    d = pd.read_csv(DETAIL, low_memory=False)
    d.columns = [c.strip().lower() for c in d.columns]

    lines = [
        "STATUS: RESEARCH ONLY — NOT PROMOTED. No production/model/weight change.\n",
        "# Component-Level Diagnostic V1\n",
        "Does MC, ML, or State individually beat Vegas, or are all three",
        "individually mediocre and the blend is just averaging weak pieces?",
        "Runs on the already-committed identity-clean non-QB cohort",
        "(clean_v1_full_stack_vegas_benchmark_detail.csv, PR #541/#542) -- no new",
        "CI run, no new data build.\n",
    ]

    # --- Part 1: MAE/bias per component, per market/season ---
    rows = []
    for (season, market), g in d.groupby(["season", "market"]):
        row = {"season": season, "market": market, "n": len(g)}
        for c in COMPONENTS + ["ensemble_proj", "line"]:
            row[f"{c}_mae"] = mae(g[c], g["actual"])
            row[f"{c}_bias"] = bias(g[c], g["actual"])
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    lines.append("## 1. MAE by component, market, season\n")
    show_cols = ["season", "market", "n", "mc_proj_mae", "ml_proj_mae", "state_proj_mae", "ensemble_proj_mae", "line_mae"]
    lines.append(comp_df[show_cols].round(3).to_markdown(index=False))

    overall = {}
    for c in COMPONENTS + ["ensemble_proj", "line"]:
        overall[c] = mae(d[c], d["actual"])
    best_component = min(COMPONENTS, key=lambda c: overall[c])
    lines.append(f"\n**Overall MAE**: mc_proj={overall['mc_proj']:.3f}, ml_proj={overall['ml_proj']:.3f}, "
                 f"state_proj={overall['state_proj']:.3f}, ensemble_proj={overall['ensemble_proj']:.3f}, "
                 f"vegas_line={overall['line']:.3f}.\n")
    lines.append(f"**Best individual component**: `{best_component}` (MAE={overall[best_component]:.3f}). "
                 f"{'Beats' if overall[best_component] < overall['line'] else 'Still loses to'} Vegas "
                 f"({'<' if overall[best_component] < overall['line'] else '>'} {overall['line']:.3f}).\n")
    lines.append(f"**Does the ensemble beat its own best component?** "
                 f"{'YES' if overall['ensemble_proj'] < overall[best_component] else 'NO'} "
                 f"(ensemble={overall['ensemble_proj']:.3f} vs best component={overall[best_component]:.3f}).\n")

    # --- Part 2: actual weights used ---
    lines.append("## 2. Ensemble weights actually applied\n")
    wcols = ["ensemble_weight_mc", "ensemble_weight_ml", "ensemble_weight_state"]
    if all(c in d.columns for c in wcols):
        wstats = d[wcols].describe().round(4)
        lines.append(wstats.to_markdown())
        wstd = d[wcols].std().round(4).to_dict()
        lines.append(f"\nStd dev of weights across rows: {wstd}. "
                     f"{'Weights vary meaningfully row to row.' if max(wstd.values()) > 0.02 else 'Weights are essentially FIXED/uniform across rows -- not row-adaptive.'}\n")
        by_market_w = d.groupby("market")[wcols].mean().round(4)
        lines.append("Mean weight by market:\n")
        lines.append(by_market_w.to_markdown())
    else:
        lines.append("Weight columns not found in detail file.\n")

    if "ensemble_status" in d.columns and "ensemble_method" in d.columns:
        lines.append("\n### Root cause traced\n")
        status = d.groupby("market")[["ensemble_status", "ensemble_method"]].agg(lambda x: x.value_counts().to_dict())
        lines.append(status.to_markdown())
        weights_path = Path("data/model_ensemble_weights.csv")
        if weights_path.exists():
            wf = pd.read_csv(weights_path)
            fitted_markets = sorted(wf["market"].unique().tolist())
            all_markets = sorted(d["market"].unique().tolist())
            missing = sorted(set(all_markets) - set(fitted_markets))
            lines.append(
                f"\n`data/model_ensemble_weights.csv` has fitted weights for: {fitted_markets}. "
                f"It has **no entry at all** for: {missing if missing else 'none -- all markets covered'}. "
                "For those missing markets, `apply_ensemble()` correctly and safely falls back to "
                "MC-only by explicit design (`data/backtests/component_predictions.csv`, the "
                "calibration accumulation file `fit_market_weights()` reads from, does not exist in "
                "this repo at all) -- this is not a modeling bug, it's an incomplete pipeline step: "
                "nobody has ever run weight-fitting for these markets. The code to do it already "
                "exists and works (proven by pass_yards/rush_att/rush_yards). Completing it is "
                "closing an existing gap, not building something new.\n"
            )

    # --- Part 3: component correlation ---
    lines.append("\n## 3. Component correlation (redundancy check)\n")
    corr = d[COMPONENTS + ["actual"]].corr().round(3)
    lines.append(corr.to_markdown())
    lines.append(
        "\nHigh pairwise correlation among mc_proj/ml_proj/state_proj (independent of their "
        "correlation with actual) would mean the three components are largely measuring the "
        "same thing -- blending them adds little regardless of how the weights are set.\n"
    )

    # --- Part 4: in-sample oracle ceiling ---
    lines.append("## 4. In-sample oracle ceiling (diagnostic bound, not a proposal)\n")
    oracle_mae, oracle_w, dropped = oracle_static_weight_mae(d)
    lines.append(
        f"Best possible STATIC (non-negative, sum-to-1) blend fit in-sample on this exact "
        f"data ({dropped} rows with missing state_proj dropped for this computation only): "
        f"MAE={oracle_mae:.3f} at weights mc={oracle_w[0]:.2f}/ml={oracle_w[1]:.2f}/"
        f"state={oracle_w[2]:.2f}, vs the realized frozen-ensemble MAE={overall['ensemble_proj']:.3f} "
        f"and Vegas MAE={overall['line']:.3f}.\n"
    )
    lines.append(
        "This is an UPPER BOUND on what reweighting alone could ever achieve (fit and evaluated "
        "on the same data -- classic in-sample overfitting, not a real out-of-sample estimate). "
        "If this ceiling still doesn't approach Vegas, reweighting cannot be the fix by itself. "
        "If it does approach or beat Vegas, that's a signal the components carry more real signal "
        "than the current static weighting extracts -- worth a genuine held-out weight-fitting "
        "study, not a reason to change production weights from this number directly.\n"
    )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"\nOverall MAE: mc={overall['mc_proj']:.3f} ml={overall['ml_proj']:.3f} state={overall['state_proj']:.3f} "
          f"ensemble={overall['ensemble_proj']:.3f} vegas={overall['line']:.3f}")
    print(f"Oracle static-weight MAE: {oracle_mae:.3f} at weights {oracle_w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
