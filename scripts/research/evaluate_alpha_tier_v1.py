"""Alpha-tier usage share: walk-forward evaluation.
Preregistration: docs/research/PRISM_ALPHA_TIER_PREREGISTRATION_V1.md (frozen first).
No sportsbook data in any arm (AGENTS.md 3).
"""
from __future__ import annotations
import numpy as np, pandas as pd, nflreadpy as nfl

G, CAP = 3.0, 6.0
DEF = {"QB": 0.0, "RB": 0.08, "WR": 0.16, "TE": 0.12, "OTHER": 0.06}
MIN_TEAM, MIN_PRIOR_GAMES, ALPHA_CUT, KAPPA = 10, 3, 0.20, 100.0
PAIRS = [(2024, 2023), (2025, 2024), (2026, 2025)]
_cache: dict[int, pd.DataFrame] = {}


def load(season: int) -> pd.DataFrame:
    if season in _cache: return _cache[season]
    d = nfl.load_player_stats(seasons=[season], summary_level="week")
    df = d.to_pandas() if hasattr(d, "to_pandas") else d
    if "season_type" in df.columns: df = df[df["season_type"] == "REG"]
    df = df[["player_id", "player_display_name", "position", "team", "season", "week", "targets"]].copy()
    df["targets"] = pd.to_numeric(df["targets"], errors="coerce").fillna(0.0)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["week", "player_id", "team"])
    tot = df.groupby(["season", "week", "team"], as_index=False)["targets"].sum().rename(columns={"targets": "team_tot"})
    df = df.merge(tot, on=["season", "week", "team"], how="left")
    df["share"] = np.where(df["team_tot"] >= MIN_TEAM, df["targets"] / df["team_tot"], np.nan)
    _cache[season] = df
    return df


def _lg(p, eps=1e-4):
    p = min(max(float(p), eps), 1 - eps); return np.log(p / (1 - p))


def _inv(z): return 1.0 / (1.0 + np.exp(-z))


def arms(g, P, m, C, n, ptgt, ttgt):
    """Return dict of arm -> prediction. C1 is production as shipped."""
    out = {"C0": C}
    w, v = [G], [g]
    if np.isfinite(P) and m > 0: w.append(min(float(m), CAP)); v.append(float(P))
    if np.isfinite(C) and n > 0: w.append(float(n)); v.append(float(C))
    out["C1"] = float(np.average(v, weights=w))
    # B1: tier-aware group weight
    Ge = G * min(1.0, g / C) if (np.isfinite(C) and C > g > 0) else G
    w1, v1 = [Ge], [g]
    if np.isfinite(P) and m > 0: w1.append(min(float(m), CAP)); v1.append(float(P))
    if np.isfinite(C) and n > 0: w1.append(float(n)); v1.append(float(C))
    out["B1"] = float(np.average(v1, weights=w1))
    # B2: same weights, logit space
    out["B2"] = float(_inv(np.average([_lg(x) for x in v], weights=w)))
    # B3: Beta-Binomial; prior mean = C1 WITHOUT the current term
    w0, v0 = [G], [g]
    if np.isfinite(P) and m > 0: w0.append(min(float(m), CAP)); v0.append(float(P))
    mu0 = float(np.average(v0, weights=w0))
    out["B3"] = float((mu0 * KAPPA + ptgt) / (KAPPA + ttgt)) if ttgt > 0 else mu0
    return out


def build() -> pd.DataFrame:
    rows = []
    for cur_s, pri_s in PAIRS:
        cur, pri = load(cur_s), load(pri_s)
        pv = pri.dropna(subset=["share"]).groupby("player_id").agg(
            ptg=("targets", "sum"), ptt=("team_tot", "sum"), m=("week", "nunique")).reset_index()
        pv["P"] = np.where(pv.ptt > 0, pv.ptg / pv.ptt, np.nan)
        for W in range(2, int(cur.week.max()) + 1):
            h = cur[(cur.week < W) & cur.share.notna()]
            agg = h.groupby("player_id").agg(ctg=("targets", "sum"), ctt=("team_tot", "sum"),
                                             n=("week", "nunique")).reset_index()
            agg["C"] = np.where(agg.ctt > 0, agg.ctg / agg.ctt, np.nan)
            agg = agg[agg.n >= MIN_PRIOR_GAMES]
            t = cur[(cur.week == W) & cur.share.notna() & (cur.targets >= 1)
                    & cur.position.isin(["WR", "TE", "RB"])]
            t = t.merge(agg, on="player_id", how="inner").merge(pv[["player_id", "P", "m"]], on="player_id", how="left")
            for r in t.itertuples(index=False):
                if not np.isfinite(r.C): continue
                g = DEF.get(str(r.position).upper(), DEF["OTHER"])
                P, m = (r.P, r.m) if np.isfinite(r.P) else (np.nan, 0)
                a = arms(g, P, m, r.C, r.n, r.ctg, r.ctt)
                row = dict(season=cur_s, week=W, player_id=r.player_id, player=r.player_display_name,
                           position=r.position, team=r.team, actual=r.share, C=r.C, n=r.n, P=P, m=m)
                for k, val in a.items():
                    row[k] = val; row["err_" + k] = abs(val - r.share); row["sgn_" + k] = val - r.share
                rows.append(row)
    d = pd.DataFrame(rows)
    d["ALPHA"] = d["C1"] >= ALPHA_CUT
    return d


def boot(df, col, iters=10000, seed=13):
    rng = np.random.default_rng(seed)
    x = df[col].to_numpy(); pid = df["player_id"].to_numpy()
    uniq, inv = np.unique(pid, return_inverse=True)
    b = [np.where(inv == i)[0] for i in range(len(uniq))]
    st = np.empty(iters)
    for i in range(iters):
        idx = np.concatenate([b[j] for j in rng.integers(0, len(b), len(b))])
        st[i] = x[idx].mean()
    lo, hi = np.percentile(st, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)
