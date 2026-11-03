"""NGS as an opportunity signal. Preregistration:
docs/research/PRISM_NGS_OPPORTUNITY_PREREGISTRATION_V1.md (frozen first).
Part A: NGS receiving -> target share.  Part B: rushing NGS -> rush yards.
Walk-forward, global coefficients, leave-one-season-out. No sportsbook data.
"""
from __future__ import annotations
import numpy as np, pandas as pd, nflreadpy as nfl

SEASONS = [2023, 2024, 2025]
MIN_PRIOR, MIN_WEEKS, LEAGUE_TEAM_TGT = 4, 3, 35.0


def _num(df, cols):
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _ps(s):
    d = nfl.load_player_stats(seasons=[s], summary_level="week")
    df = d.to_pandas() if hasattr(d, "to_pandas") else d
    if "season_type" in df.columns: df = df[df.season_type == "REG"]
    return _num(df, ["targets", "carries", "rushing_yards", "week"])


def _ngs(s, kind):
    d = nfl.load_nextgen_stats(seasons=[s], stat_type=kind)
    df = d.to_pandas() if hasattr(d, "to_pandas") else d
    df = _num(df, ["week", "percent_share_of_intended_air_yards",
                   "rush_yards_over_expected_per_att", "efficiency",
                   "percent_attempts_gte_eight_defenders", "rush_attempts", "rush_yards"])
    return df[df.week > 0]


def build_A() -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        ps, ng = _ps(s), _ngs(s, "receiving")
        tt = ps.groupby(["team", "week"], as_index=False).targets.sum().rename(columns={"targets": "team_tgt"})
        p = ps.merge(tt, on=["team", "week"], how="left")
        p["tshare"] = np.where(p.team_tgt >= 10, p.targets / p.team_tgt, np.nan)
        j = ng[["player_gsis_id", "week", "percent_share_of_intended_air_yards"]].rename(
            columns={"player_gsis_id": "player_id", "percent_share_of_intended_air_yards": "sh"})
        j["sh"] = j.sh / 100.0
        d = p.merge(j, on=["player_id", "week"], how="inner")
        for W in range(2, int(d.week.max()) + 1):
            h = d[(d.week < W) & d.tshare.notna()]
            g = h.groupby("player_id").agg(B=("tshare", "mean"), SH=("sh", "mean"),
                                           TG=("targets", "mean"), ng=("week", "nunique")).reset_index()
            g = g[g.ng >= MIN_PRIOR]
            t = d[(d.week == W) & (d.targets >= 1) & d.tshare.notna()][
                ["player_id", "player_display_name", "week", "tshare"]].rename(columns={"tshare": "actual"})
            t = t.merge(g, on="player_id", how="inner")
            t["season"] = s; t["C"] = t.TG / LEAGUE_TEAM_TGT
            rows.append(t)
    return pd.concat(rows, ignore_index=True).dropna(subset=["actual", "B", "SH"])


def build_B() -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        ps, ng = _ps(s), _ngs(s, "rushing")
        j = ng[["player_gsis_id", "week", "rush_yards_over_expected_per_att", "efficiency",
                "percent_attempts_gte_eight_defenders"]].rename(columns={"player_gsis_id": "player_id"})
        d = ps.merge(j, on=["player_id", "week"], how="inner")
        for W in range(2, int(d.week.max()) + 1):
            h = d[(d.week < W) & (d.carries >= 1)]
            if len(h) == 0: continue
            ypc = h.rushing_yards.sum() / max(1.0, h.carries.sum())
            g = h.groupby("player_id").agg(B=("rushing_yards", "mean"), CAR=("carries", "mean"),
                                           ROE=("rush_yards_over_expected_per_att", "mean"),
                                           EFF=("efficiency", "mean"), ng=("week", "nunique")).reset_index()
            g = g[g.ng >= MIN_PRIOR]
            t = d[(d.week == W) & (d.carries >= 1)][
                ["player_id", "player_display_name", "week", "rushing_yards"]].rename(columns={"rushing_yards": "actual"})
            t = t.merge(g, on="player_id", how="inner")
            t["season"] = s; t["ypc"] = ypc; t["C"] = t.CAR * ypc
            rows.append(t)
    return pd.concat(rows, ignore_index=True).dropna(subset=["actual", "B", "ROE"])


def _beta(tr, x, resid=True):
    c = tr[x].mean(); xv = (tr[x] - c).to_numpy()
    yv = (tr.actual - tr.B).to_numpy() if resid else tr.actual.to_numpy()
    ok = np.isfinite(xv) & np.isfinite(yv)
    return (float(np.polyfit(xv[ok], yv[ok], 1)[0]) if ok.sum() > 40 else 0.0), float(c)


def apply_A(a):
    a = a.copy(); a["A1"] = np.nan; a["A2"] = np.nan
    for s in SEASONS:
        tr, te = a[a.season != s], a.season == s
        ok = np.isfinite(tr.SH) & np.isfinite(tr.actual)
        b, i = np.polyfit(tr.SH[ok], tr.actual[ok], 1)
        a.loc[te, "A1"] = i + b * a.loc[te, "SH"]
        tr2 = tr.assign(D=tr.SH - tr.B); bb, cc = _beta(tr2.rename(columns={"D": "x"}), "x")
        a.loc[te, "A2"] = a.loc[te, "B"] + bb * ((a.loc[te, "SH"] - a.loc[te, "B"]) - cc)
    return a


def apply_B(a):
    a = a.copy(); a["R1"] = np.nan; a["R2"] = np.nan
    for s in SEASONS:
        tr, te = a[a.season != s], a.season == s
        b, c = _beta(tr, "ROE")
        a.loc[te, "R1"] = a.loc[te, "B"] + b * (a.loc[te, "ROE"] - c)
        a.loc[te, "R2"] = a.loc[te, "CAR"] * (a.loc[te, "ypc"] + b / max(1e-9, tr.CAR.mean()) * (a.loc[te, "ROE"] - c))
    return a


def boot(df, fn, iters=10000, seed=101):
    rng = np.random.default_rng(seed)
    return np.array([fn(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(iters)])
