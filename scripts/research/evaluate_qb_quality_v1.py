"""QB quality -> receiver production. Preregistration:
docs/research/PRISM_QB_QUALITY_PREREGISTRATION_V1.md (frozen first).
Walk-forward within season; one GLOBAL coefficient, fitted leave-one-season-out.
No sportsbook data.
"""
from __future__ import annotations
import numpy as np, pandas as pd, nflreadpy as nfl

SEASONS = [2023, 2024, 2025]
MIN_PRIOR_REC, MIN_PRIOR_QB, MIN_QB_ATT = 4, 4, 10
_c: dict[int, pd.DataFrame] = {}


def load(s: int) -> pd.DataFrame:
    if s in _c: return _c[s]
    d = nfl.load_player_stats(seasons=[s], summary_level="week")
    df = d.to_pandas() if hasattr(d, "to_pandas") else d
    if "season_type" in df.columns: df = df[df.season_type == "REG"]
    for col in ("targets", "receiving_yards", "attempts", "passing_cpoe", "passing_epa", "passing_yards"):
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    df["week"] = pd.to_numeric(df.week, errors="coerce")
    _c[s] = df
    return df


def team_points(s: int) -> pd.DataFrame:
    sch = nfl.load_schedules(seasons=[s])
    sc = sch.to_pandas() if hasattr(sch, "to_pandas") else sch
    a = sc[["season", "week", "home_team", "home_score"]].rename(columns={"home_team": "team", "home_score": "pts"})
    b = sc[["season", "week", "away_team", "away_score"]].rename(columns={"away_team": "team", "away_score": "pts"})
    out = pd.concat([a, b], ignore_index=True)
    out["week"] = pd.to_numeric(out.week, errors="coerce")
    out["pts"] = pd.to_numeric(out.pts, errors="coerce")
    return out.dropna(subset=["week", "pts"])


def build_season(s: int) -> pd.DataFrame:
    df = load(s); pts = team_points(s)
    qbs = df[(df.position == "QB") & (df.attempts >= MIN_QB_ATT)]
    rows = []
    for W in range(2, int(df.week.max()) + 1):
        hist = df[df.week < W]
        # receiver baseline
        rh = hist[hist.position.isin(["WR", "TE", "RB"]) & (hist.targets >= 1)]
        base = rh.groupby("player_id").agg(B=("receiving_yards", "mean"), ng=("week", "nunique")).reset_index()
        base = base[base.ng >= MIN_PRIOR_REC]
        # primary QB of week W per team, and his prior quality
        wk = df[df.week == W]
        prim = wk[wk.position == "QB"].sort_values("attempts", ascending=False).groupby("team").first().reset_index()
        qh = qbs[qbs.week < W].groupby(["player_id"]).agg(
            cpoe=("passing_cpoe", "mean"), epa=("passing_epa", "sum"), att=("attempts", "sum"),
            ngq=("week", "nunique")).reset_index()
        qh["epa_att"] = qh.epa / qh.att.replace(0, np.nan)
        qh = qh[qh.ngq >= MIN_PRIOR_QB]
        prim = prim[["team", "player_id"]].rename(columns={"player_id": "qb_id"}).merge(
            qh.rename(columns={"player_id": "qb_id"}), on="qb_id", how="inner")
        # prior primary QB (week W-1) for the change flag
        pw = df[(df.week == W - 1) & (df.position == "QB")].sort_values("attempts", ascending=False)
        prev = pw.groupby("team").first().reset_index()[["team", "player_id"]].rename(columns={"player_id": "prev_qb"})
        prim = prim.merge(prev, on="team", how="left")
        prim["qb_changed"] = prim.prev_qb.notna() & (prim.qb_id != prim.prev_qb)
        # team prior points
        tp = pts[(pts.season == s) & (pts.week < W)].groupby("team").pts.mean().rename("team_pts").reset_index()
        tgt = wk[wk.position.isin(["WR", "TE", "RB"]) & (wk.targets >= 1)]
        t = (tgt[["player_id", "player_display_name", "position", "team", "receiving_yards"]]
             .merge(base[["player_id", "B"]], on="player_id", how="inner")
             .merge(prim[["team", "cpoe", "epa_att", "qb_changed"]], on="team", how="inner")
             .merge(tp, on="team", how="left"))
        t["season"], t["week"] = s, W
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def build() -> pd.DataFrame:
    a = pd.concat([build_season(s) for s in SEASONS], ignore_index=True)
    return a.dropna(subset=["receiving_yards", "B", "cpoe"])


def fit_beta(tr: pd.DataFrame, xcol: str) -> float:
    x = (tr[xcol] - tr[xcol].mean()).to_numpy()
    y = (tr.receiving_yards - tr.B).to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() > 50 else 0.0


def apply_loso(a: pd.DataFrame) -> pd.DataFrame:
    a = a.copy()
    for col, arm in (("cpoe", "Q"), ("team_pts", "T"), ("epa_att", "Q_epa")):
        a[arm] = np.nan
        for s in SEASONS:
            tr, te = a[a.season != s], a.season == s
            if a.loc[te, col].notna().sum() == 0: continue
            b = fit_beta(tr.dropna(subset=[col]), col)
            a.loc[te, arm] = a.loc[te, "B"] + b * (a.loc[te, col] - tr[col].mean())
    return a


def boot(df, fn, iters=10000, seed=73):
    rng = np.random.default_rng(seed)
    return np.array([fn(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(iters)])
