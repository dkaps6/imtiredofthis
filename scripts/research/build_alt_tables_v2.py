"""Alt-line clearance tables v2: drift-aware, and with the two missing markets.

Three changes over v1, all of them things v1 got wrong or left out.

1. SEASON DRIFT IS PRICED IN. v1's conservative column was a 1.96-sigma
   sampling bound and nothing else. But realized clearance moves between
   seasons by more than sampling error in at least one market: measured
   2024 vs 2025 at the same fraction of line, rush_rec_yards moves 0.0632,
   against a 3pp calibration bar -- and rush_rec_yards is the market carrying
   v1's headline "94.6% clears" row. A bound that ignores regime shift
   overstates how safe the safe tier is. v2's conservative figure subtracts
   BOTH the sampling bound and the measured season drift.

2. pass_yards is included, POOLED. It failed v1's support floor only because
   of quartile stratification (36 players in the 2024 fit is ~9 per quartile
   against a 40-player bar). The parent study measured that stratification
   does not help this market (S2 ns and point-negative) and that pooled passed
   decile calibration where stratified failed, so pooled is the better model
   here, not a fallback. Both seasons give 812 rows / 43 players.

3. receptions uses ABSOLUTE integer thresholds. A fraction of an integer line
   is the wrong parameterisation: 0.4x, 0.5x, 0.6x and 0.7x of a 2.5 line all
   mean ">= 2". Books quote it as "3+ receptions" and the table now matches.

Receptions, and the line cap, which is NO LONGER post-hoc. The absolute table
was preregistered to need decile calibration within 3pp on a 2024->2025 split
and FAILED at 0.0525, concentrated entirely in main line 5.5. The cap at 4.5
was initially chosen after seeing that, which made it post-hoc.

It has since been validated on a THIRD, fully independent sample: 2026 weeks
1/2/6/7/8 board lines joined to nflverse actuals, 583 rows / 215 players. Line
5.5 is again the worst by a wide margin -- max cell error 0.1059 against
0.051-0.059 for every other line, on only 29 rows -- so two independent
holdouts agree it is unstable and the cap stands on evidence rather than on
hindsight.

What 2026 also showed is that the surviving lines carry no systematic drift
(mean gaps +0.017, +0.000, -0.012, -0.002) but do carry about 5pp of cell-level
noise, which is above the 3pp bar. That error is priced into the conservative
bound via RECP_OOS_CELL_ERROR rather than wished away.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv"
OUT_FRAC = ROOT / "data/research/alt_line_clearance_v2.csv"
OUT_RECP = ROOT / "data/research/alt_line_receptions_absolute_v2.csv"
OUT_JSON = ROOT / "data/research/alt_tables_v2_results.json"

GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
MIN_ROWS, MIN_PLAYERS = 100, 40
STRATIFIED = ["rec_yards", "rush_yards", "rush_rec_yards"]   # volume stratification helps
POOLED = ["pass_yards"]                                       # measured: it does not
RECP_MAX_LINE = 4.5   # validated on 2026 out-of-sample, see module docstring
# Measured cell-level error of the receptions table against a THIRD, fully
# independent sample (2026 weeks 1/2/6/7/8 board lines joined to nflverse
# actuals, 583 rows / 215 players). Mean gaps per line were +0.017, +0.000,
# -0.012, -0.002 -- no systematic drift -- but worst-cell error reached 0.059.
# That is above the 3pp bar, so it is priced into the conservative bound
# rather than wished away.
RECP_OOS_CELL_ERROR = 0.059


def load() -> pd.DataFrame:
    h = pd.read_csv(HIST, low_memory=False)
    opp = pd.to_numeric(h.get("actual_opportunities"), errors="coerce")
    h = h[~((opp.fillna(0) <= 0) & (h["actual"].fillna(0) <= 0))]
    h = h.dropna(subset=["line", "actual"])
    h = h[h["line"] > 0]
    h["player"] = h["player_clean_key"].astype(str)
    return h


def _bound(p: float, npl: int, drift: float) -> float:
    """Sampling bound AND measured regime shift, not just sampling."""
    se = np.sqrt(max(p * (1 - p), 1e-9) / max(npl, 1))
    return float(np.clip(p - 1.96 * se - abs(drift), 1e-4, 1 - 1e-4))


def _be(p: float) -> int:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return int(round(-100 * p / (1 - p) if p > 0.5 else 100 * (1 - p) / p))


def _drift_frac(g: pd.DataFrame, f: float) -> float:
    a, b = g[g.season == 2024], g[g.season == 2025]
    if len(a) < 30 or len(b) < 30:
        return 0.0
    c24 = float(np.mean(b24 := (a["actual"].to_numpy(float) > f * a["line"].to_numpy(float))))
    c25 = float(np.mean(b["actual"].to_numpy(float) > f * b["line"].to_numpy(float)))
    return c25 - c24


def build_fraction(h: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in STRATIFIED + POOLED:
        g = h[h["market"] == m]
        if g.empty:
            continue
        if m in POOLED:
            groups = [(-1, g)]
        else:
            q = np.quantile(g[g.season == 2024]["line"], [.25, .5, .75])
            qt = np.digitize(g["line"].to_numpy(float), q)
            groups = [(int(k), g[qt == k]) for k in sorted(set(qt))]
        for qk, part in groups:
            n, npl = len(part), int(part["player"].nunique())
            if n < MIN_ROWS or npl < MIN_PLAYERS:
                continue
            a, y = part["line"].to_numpy(float), part["actual"].to_numpy(float)
            for f in GRID:
                c = float(np.mean(y > f * a))
                d = _drift_frac(part, f)
                lo = _bound(c, npl, d)
                rows.append({"market": m, "line_quartile": qk, "frac_of_LINE": f,
                             "pooled": qk == -1,
                             "line_median": float(part["line"].median()),
                             "n": n, "players": npl,
                             "cleared": round(c, 4),
                             "season_drift": round(d, 4),
                             "cleared_conservative": round(lo, 4),
                             "breakeven_american": _be(c),
                             "breakeven_american_conservative": _be(lo)})
    return pd.DataFrame(rows)


def build_receptions(h: pd.DataFrame) -> pd.DataFrame:
    g = h[(h["market"] == "receptions") & (h["line"] <= RECP_MAX_LINE)]
    rows = []
    for L, part in g.groupby("line"):
        n, npl = len(part), int(part["player"].nunique())
        if n < MIN_ROWS or npl < MIN_PLAYERS:
            continue
        y = part["actual"].to_numpy(float)
        a24 = part[part.season == 2024]["actual"].to_numpy(float)
        a25 = part[part.season == 2025]["actual"].to_numpy(float)
        for k in range(1, 13):
            p = float(np.mean(y >= k))
            if p <= 1e-4 or p >= 1 - 1e-4:
                continue
            d = (float(np.mean(a25 >= k)) - float(np.mean(a24 >= k))
                 if len(a24) >= 30 and len(a25) >= 30 else 0.0)
            # Widen by whichever is larger: the season drift seen inside the fit
            # sample, or the cell error measured against the independent 2026 set.
            lo = _bound(p, npl, max(abs(d), RECP_OOS_CELL_ERROR))
            rows.append({"market": "receptions", "main_line": float(L),
                         "threshold_at_least": k, "n": n, "players": npl,
                         "cleared": round(p, 4), "season_drift": round(d, 4),
                         "cleared_conservative": round(lo, 4),
                         "breakeven_american": _be(p),
                         "breakeven_american_conservative": _be(lo),
                         "post_hoc_line_restriction": True})
    return pd.DataFrame(rows)


def main() -> int:
    h = load()
    frac, recp = build_fraction(h), build_receptions(h)
    frac.to_csv(OUT_FRAC, index=False)
    recp.to_csv(OUT_RECP, index=False)

    print(f"wrote {OUT_FRAC} rows={len(frac)} markets={sorted(frac['market'].unique())}")
    print(f"wrote {OUT_RECP} rows={len(recp)} lines={sorted(recp['main_line'].unique())}")
    print("\n=== what the drift adjustment costs the headline rows ===")
    show = frac[(frac.frac_of_LINE.isin([0.4, 0.5, 0.6])) &
                (frac.market.isin(["rush_rec_yards", "pass_yards"]))]
    print(show[["market", "line_quartile", "frac_of_LINE", "cleared", "season_drift",
                "cleared_conservative", "breakeven_american",
                "breakeven_american_conservative"]].to_string(index=False))
    print("\n=== receptions, absolute thresholds ===")
    print(recp.pivot_table(index="threshold_at_least", columns="main_line",
                           values="cleared").round(3).to_string())
    OUT_JSON.write_text(json.dumps(
        {"fraction_rows": len(frac), "receptions_rows": len(recp),
         "receptions_line_cap": RECP_MAX_LINE,
         "receptions_restriction_is_post_hoc": True,
         "max_abs_drift_by_market": {m: float(frac[frac.market == m]["season_drift"].abs().max())
                                     for m in frac["market"].unique()}},
        indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
