"""Close the two unpriceable markets in the alt-line decision layer.

pass_yards  -> POOLED fraction-of-line curve (no quartile stratification).
receptions  -> ABSOLUTE integer thresholds by main line.

Design stated before scoring (PRISM_ALT_LINE_ANCHOR_RESULTS_V1 is the parent):

pass_yards failed the support floor only because of stratification: 36 players
in the 2024 fit is ~9 per quartile against a 40-player bar. The pooled
structure is arm B0 of the parent study, which was already validated
out-of-sample on 2025 -- it beat the projection anchor (-0.00567, CI excluding
0), stratification added nothing for this market (S2 = +0.00049
[-0.00028,+0.00131], ns and point-NEGATIVE), and pooled passed decile
calibration at 2.98pp where stratified failed at 3.38pp. So pooled is the
measured-better model here, not a fallback. Refitting that validated structure
on both seasons gives 812 rows / 43 players and clears the bar.

receptions is an integer market, so a fraction of the line is the wrong
parameterisation: 0.4x, 0.5x, 0.6x and 0.7x of a 2.5 line all mean ">= 2". The
fix is absolute thresholds, which is also how books actually quote it ("3+
receptions"). That is a NEW parameterisation, so it must clear its own bars,
fixed here before anything is computed:
  * support   -- every emitted cell >= 100 rows AND >= 40 players
  * calibration -- fit 2024, predict 2025, |predicted - realized| <= 3pp
                   across deciles of predicted probability
  * monotonicity -- P(>= k) non-increasing in k within a line
If calibration or monotonicity fails, the table is NOT emitted.

Final tables are refit on both seasons once the structure is validated, which
is the usual validate-then-refit pattern and is stated rather than hidden.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20261104)
ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv"
OUT_PASS = ROOT / "data/research/alt_line_pass_yards_pooled_v1.csv"
OUT_RECP = ROOT / "data/research/alt_line_receptions_absolute_v1.csv"
OUT_JSON = ROOT / "data/research/alt_tables_passyards_receptions_v1_results.json"

FINE = np.round(np.arange(0.20, 2.0001, 0.05), 4)
GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
MIN_ROWS, MIN_PLAYERS = 100, 40


def load() -> pd.DataFrame:
    h = pd.read_csv(HIST, low_memory=False)
    opp = pd.to_numeric(h.get("actual_opportunities"), errors="coerce")
    h = h[~((opp.fillna(0) <= 0) & (h["actual"].fillna(0) <= 0))]
    h = h.dropna(subset=["line", "actual"])
    h = h[h["line"] > 0]
    h["player"] = h["player_clean_key"].astype(str)
    return h


def _ci_lo(p: float, n_players: int) -> float:
    se = np.sqrt(max(p * (1 - p), 1e-9) / max(n_players, 1))
    return float(np.clip(p - 1.96 * se, 1e-4, 1 - 1e-4))


def _breakeven(p: float) -> int:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return int(round(-100 * p / (1 - p) if p > 0.5 else 100 * (1 - p) / p))


# ---------------------------------------------------------------- pass_yards
def build_pass(h: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = h[h["market"] == "pass_yards"]
    n, npl = len(g), int(g["player"].nunique())
    ok = n >= MIN_ROWS and npl >= MIN_PLAYERS
    rows = []
    if ok:
        a, y = g["line"].to_numpy(float), g["actual"].to_numpy(float)
        for f in GRID:
            c = float(np.mean(y > f * a))
            lo = _ci_lo(c, npl)
            rows.append({"market": "pass_yards", "frac_of_LINE": f,
                         "line_median": float(g["line"].median()),
                         "n": n, "players": npl,
                         "cleared": round(c, 4), "cleared_ci_lo": round(lo, 4),
                         "breakeven_american": _breakeven(c),
                         "breakeven_american_conservative": _breakeven(lo)})
    return pd.DataFrame(rows), {"rows": n, "players": npl, "support_ok": ok}


# ---------------------------------------------------------------- receptions
def _recp_cells(g: pd.DataFrame) -> dict[tuple[float, int], tuple[float, int, int]]:
    out = {}
    for L, part in g.groupby("line"):
        y = part["actual"].to_numpy(float)
        npl = int(part["player"].nunique())
        for k in range(1, 13):
            p = float(np.mean(y >= k))
            out[(float(L), k)] = (p, len(part), npl)
    return out


def build_recp(h: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = h[h["market"] == "receptions"]
    fit, val = g[g["season"] == 2024], g[g["season"] == 2025]

    # --- calibration: fit 2024, predict 2025
    cells = _recp_cells(fit)
    preds, obs = [], []
    for L, part in val.groupby("line"):
        y = part["actual"].to_numpy(float)
        for k in range(1, 13):
            c = cells.get((float(L), k))
            if c is None or c[1] < MIN_ROWS or c[2] < MIN_PLAYERS:
                continue
            preds.append(c[0])
            obs.append(float(np.mean(y >= k)))
    cal = pd.DataFrame({"p": preds, "y": obs})
    max_gap = np.nan
    if len(cal) >= 10:
        cal["d"] = pd.qcut(cal["p"], min(10, cal["p"].nunique()), labels=False, duplicates="drop")
        t = cal.groupby("d").agg(pred=("p", "mean"), real=("y", "mean"))
        max_gap = float((t["real"] - t["pred"]).abs().max())
    cal_ok = bool(np.isfinite(max_gap) and max_gap <= 0.03)

    # --- monotonicity on the full sample
    full = _recp_cells(g)
    mono_ok = True
    viol = []
    for L in sorted({l for l, _ in full}):
        seq = [full[(L, k)][0] for k in range(1, 13) if (L, k) in full]
        for i in range(1, len(seq)):
            if seq[i] > seq[i - 1] + 1e-12:
                mono_ok = False
                viol.append((L, i + 1))
    rows = []
    if cal_ok and mono_ok:
        for (L, k), (p, n, npl) in sorted(full.items()):
            if n < MIN_ROWS or npl < MIN_PLAYERS:
                continue
            if p <= 1e-4 or p >= 1 - 1e-4:
                continue
            lo = _ci_lo(p, npl)
            rows.append({"market": "receptions", "main_line": L, "threshold_at_least": k,
                         "n": n, "players": npl,
                         "cleared": round(p, 4), "cleared_ci_lo": round(lo, 4),
                         "breakeven_american": _breakeven(p),
                         "breakeven_american_conservative": _breakeven(lo)})
    return pd.DataFrame(rows), {"calibration_max_gap": max_gap, "calibration_ok": cal_ok,
                                "monotonic": mono_ok, "monotonicity_violations": viol[:10],
                                "calibration_cells": int(len(cal))}


def main() -> int:
    h = load()
    pass_df, pass_meta = build_pass(h)
    recp_df, recp_meta = build_recp(h)

    print("=== pass_yards (pooled, both seasons) ===")
    print(f"  support: {pass_meta['rows']} rows / {pass_meta['players']} players "
          f"-> {'PASS' if pass_meta['support_ok'] else 'FAIL'} (bars {MIN_ROWS}/{MIN_PLAYERS})")
    if not pass_df.empty:
        print(pass_df[["frac_of_LINE", "cleared", "cleared_ci_lo",
                       "breakeven_american", "breakeven_american_conservative"]].to_string(index=False))

    print("\n=== receptions (absolute thresholds) ===")
    print(f"  calibration max |gap| = {recp_meta['calibration_max_gap']:.4f} over "
          f"{recp_meta['calibration_cells']} cells -> {'PASS' if recp_meta['calibration_ok'] else 'FAIL'} (bar 0.03)")
    print(f"  monotonic in k: {'PASS' if recp_meta['monotonic'] else 'FAIL ' + str(recp_meta['monotonicity_violations'])}")
    if not recp_df.empty:
        piv = recp_df.pivot_table(index="threshold_at_least", columns="main_line", values="cleared")
        print("  P(receptions >= k) by main line:")
        print(piv.round(3).to_string())

    if not pass_df.empty:
        pass_df.to_csv(OUT_PASS, index=False)
        print(f"\nwrote {OUT_PASS} ({len(pass_df)} rows)")
    else:
        print("\npass_yards NOT emitted")
    if not recp_df.empty:
        recp_df.to_csv(OUT_RECP, index=False)
        print(f"wrote {OUT_RECP} ({len(recp_df)} rows)")
    else:
        print("receptions NOT emitted (a bar failed)")

    OUT_JSON.write_text(json.dumps({"pass_yards": pass_meta, "receptions": recp_meta},
                                   indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
