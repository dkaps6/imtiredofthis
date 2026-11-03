"""NGS receiving signal. Preregistration:
docs/research/PRISM_NGS_RECEIVING_PREREGISTRATION_V1.md (frozen first).
Walk-forward within season; global coefficients fitted leave-one-season-out.
Target-week NGS supplies the OUTCOME only. No sportsbook data.
"""
from __future__ import annotations
import numpy as np, pandas as pd, nflreadpy as nfl

SEASONS = [2023, 2024, 2025]
MIN_PRIOR, MIN_WEEKS = 4, 3


def load() -> pd.DataFrame:
    fr = []
    for s in SEASONS:
        d = nfl.load_nextgen_stats(seasons=[s], stat_type="receiving")
        df = d.to_pandas() if hasattr(d, "to_pandas") else d
        df = df[pd.to_numeric(df.week, errors="coerce") > 0].copy()
        df["season"] = s
        fr.append(df)
    a = pd.concat(fr, ignore_index=True)
    for c in ("avg_intended_air_yards", "percent_share_of_intended_air_yards",
              "avg_separation", "yards", "targets", "week"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    return a.dropna(subset=["week", "player_gsis_id"])


def build() -> pd.DataFrame:
    a = load()
    rows = []
    for s in SEASONS:
        d = a[a.season == s]
        for W in range(2, int(d.week.max()) + 1):
            hist = d[d.week < W]
            g = hist.groupby("player_gsis_id").agg(
                B=("yards", "mean"), AY=("avg_intended_air_yards", "mean"),
                SH=("percent_share_of_intended_air_yards", "mean"),
                SEP=("avg_separation", "mean"), TG=("targets", "mean"),
                ng=("week", "nunique")).reset_index()
            g = g[g.ng >= MIN_PRIOR]
            # league prior yards-per-target for the naive control
            ypt = hist.yards.sum() / hist.targets.sum() if hist.targets.sum() > 0 else np.nan
            t = d[(d.week == W) & (d.targets >= 1)][
                ["player_gsis_id", "player_display_name", "week", "yards"]].rename(columns={"yards": "actual"})
            t = t.merge(g, on="player_gsis_id", how="inner")
            t["season"] = s
            t["C"] = t.TG * ypt
            rows.append(t)
    return pd.concat(rows, ignore_index=True).dropna(subset=["actual", "B", "AY", "SH", "SEP"])


def fit(tr: pd.DataFrame, x: str, resid: bool) -> tuple[float, float]:
    """Return (beta, centre). resid=True fits on (actual - B); else on actual."""
    c = tr[x].mean()
    xv = (tr[x] - c).to_numpy()
    yv = (tr.actual - tr.B).to_numpy() if resid else tr.actual.to_numpy()
    ok = np.isfinite(xv) & np.isfinite(yv)
    return (float(np.polyfit(xv[ok], yv[ok], 1)[0]) if ok.sum() > 50 else 0.0), float(c)


def apply_loso(a: pd.DataFrame) -> pd.DataFrame:
    a = a.copy()
    for arm in ("N1", "N2", "N3"): a[arm] = np.nan
    for s in SEASONS:
        tr, te = a[a.season != s], a.season == s
        # N1 replaces the baseline: regress actual on AY (with intercept)
        ok = np.isfinite(tr.AY) & np.isfinite(tr.actual)
        b1, a1 = np.polyfit(tr.AY[ok], tr.actual[ok], 1)
        a.loc[te, "N1"] = a1 + b1 * a.loc[te, "AY"]
        for arm, col in (("N2", "SEP"), ("N3", "SH")):
            b, c = fit(tr, col, resid=True)
            a.loc[te, arm] = a.loc[te, "B"] + b * (a.loc[te, col] - c)
    return a


def boot(df, fn, iters=10000, seed=91):
    rng = np.random.default_rng(seed)
    return np.array([fn(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(iters)])
