"""Receiving mass bias V2: leave-one-week-out cross-fitted arms.
Preregistration: docs/research/PRISM_RECEIVING_MASS_PREREGISTRATION_V2.md (frozen first).
Primary metric is PER-PLAYER bias. Pooled MAE is a guardrail only (AGENTS.md).
"""
from __future__ import annotations
import numpy as np, pandas as pd

ROWS = "data/research/live_2026/backtest_2026_live_calibration_v1_rows.csv"
WEEKS = [1, 2, 6, 7, 8]
GRID = np.arange(0, 1.0001, 0.05)
MIN_PLAYERS, MIN_WEEKS = 100, 3


def run(market: str) -> pd.DataFrame:
    g = pd.read_csv(ROWS)
    d = g[g.market == market].dropna(subset=["actual", "mc_proj", "ml_proj", "ensemble_proj"]).copy()
    d["C1"] = d.ensemble_proj
    d["D1"] = d.ml_proj
    d["D0"] = np.nan
    d["D2"] = np.nan
    for W in WEEKS:
        hi = d[d._board_week != W]
        ho = d._board_week == W
        if len(hi) == 0 or ho.sum() == 0:
            continue
        delta = float((hi.actual - hi.C1).mean())
        best = min(GRID, key=lambda w: abs(float(((w * hi.ml_proj + (1 - w) * hi.mc_proj) - hi.actual).mean())))
        d.loc[ho, "D0"] = d.loc[ho, "C1"] + delta
        d.loc[ho, "D2"] = best * d.loc[ho, "ml_proj"] + (1 - best) * d.loc[ho, "mc_proj"]
    return d.dropna(subset=["D0", "D2"])


def player_bias(d: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for a in ("C1", "D0", "D1", "D2"):
        s = d.assign(e=d[a] - d.actual).groupby("player").e.agg(["mean", "size"])
        out[a] = s["mean"]
        n = s["size"]
    pl = pd.DataFrame(out)
    pl["n"] = n
    return pl[pl.n >= MIN_WEEKS]


def boot(df, fn, iters=10000, seed=31):
    rng = np.random.default_rng(seed)
    return np.array([fn(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(iters)])
