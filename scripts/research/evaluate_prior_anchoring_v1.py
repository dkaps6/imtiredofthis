"""Prior-anchoring fix: walk-forward evaluation of four posterior-weight arms.

Preregistration: docs/research/PRISM_PRIOR_ANCHORING_PREREGISTRATION_V1.md (frozen first).
Predicts 2026 week-W player usage share from 2025 prior + 2026 weeks 1..W-1 only.
No sportsbook data in any arm (AGENTS.md 3).
"""
from __future__ import annotations
import numpy as np, pandas as pd, nflreadpy as nfl

GROUP_STRENGTH = {"tgt_share": 3.0, "rush_share": 3.0}
PRIOR_CAP = {"tgt_share": 6.0, "rush_share": 6.0}
DEFAULTS = {
    "tgt_share": {"QB": 0.0, "RB": 0.08, "WR": 0.16, "TE": 0.12, "OTHER": 0.06},
    "rush_share": {"QB": 0.14, "RB": 0.32, "WR": 0.02, "TE": 0.00, "OTHER": 0.02},
}
NUM = {"tgt_share": "targets", "rush_share": "carries"}
MIN_TEAM = 10
S_PREREG, D_PREREG = 0.05, 8.0
DIVERGENCE_CUT = 0.05


def load(season: int) -> pd.DataFrame:
    d = nfl.load_player_stats(seasons=[season])
    df = d.to_pandas() if hasattr(d, "to_pandas") else d
    df = df[df["season_type"] == "REG"] if "season_type" in df.columns else df
    keep = ["player_id", "player_display_name", "position", "team", "season", "week", "targets", "carries"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for c in ("targets", "carries"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    return df.dropna(subset=["week", "player_id", "team"])


def shares(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    num = NUM[metric]
    tot = df.groupby(["season", "week", "team"], as_index=False)[num].sum().rename(columns={num: "team_tot"})
    out = df.merge(tot, on=["season", "week", "team"], how="left")
    out["share"] = np.where(out["team_tot"] >= MIN_TEAM, out[num] / out["team_tot"], np.nan)
    out["opp"] = out[num]
    return out


def pos_default(pos: str, metric: str) -> float:
    p = str(pos or "").upper()
    return DEFAULTS[metric].get(p, DEFAULTS[metric]["OTHER"])


def posterior(g, P, m, C, n, metric, arm, s=S_PREREG, D=D_PREREG):
    G, CAP = GROUP_STRENGTH[metric], PRIOR_CAP[metric]
    if arm == "C0":
        return C
    w, v = [G], [g]
    if np.isfinite(P) and m > 0:
        pw = min(float(m), CAP)
        if arm == "A1":
            pw *= max(0.0, 1.0 - n / D)
        elif arm == "A2":
            pw /= 1.0 + (abs(C - P) / s) ** 2 if np.isfinite(C) else 1.0
        if pw > 0:
            w.append(pw); v.append(float(P))
    if np.isfinite(C) and n > 0:
        w.append(float(n)); v.append(float(C))
    return float(np.average(np.asarray(v), weights=np.asarray(w)))


def build(metric: str, s=S_PREREG, D=D_PREREG) -> pd.DataFrame:
    prior = shares(load(2025), metric)
    cur = shares(load(2026), metric)
    pr = (prior.dropna(subset=["share"]).groupby("player_id")
          .agg(P=("share", "mean"), m=("share", "size")).reset_index())
    rows = []
    for W in range(2, int(cur["week"].max()) + 1):
        hist = cur[cur["week"] < W].dropna(subset=["share"])
        agg = hist.groupby("player_id").agg(C=("share", "mean"), n=("share", "size")).reset_index()
        tgt = cur[(cur["week"] == W) & cur["share"].notna() & (cur["opp"] >= 1)].copy()
        t = tgt.merge(agg, on="player_id", how="inner").merge(pr, on="player_id", how="left")
        for r in t.itertuples(index=False):
            g = pos_default(r.position, metric)
            P, m = (r.P, r.m) if np.isfinite(r.P) else (np.nan, 0)
            base = dict(week=W, player_id=r.player_id, player=r.player_display_name,
                        position=r.position, team=r.team, actual=r.share, C=r.C, n=r.n,
                        P=P, m=m, has_prior=bool(m > 0),
                        divergence=abs(r.C - P) if np.isfinite(P) else np.nan)
            for arm in ("C0", "C1", "A1", "A2"):
                base[arm] = posterior(g, P, m, r.C, r.n, metric, arm, s, D)
                base[f"err_{arm}"] = abs(base[arm] - r.share)
            rows.append(base)
    return pd.DataFrame(rows)


def boot_by_player(df, a, b, iters=10000, seed=7):
    """Paired bootstrap on mean(err_b - err_a), resampled BY PLAYER."""
    rng = np.random.default_rng(seed)
    d = df[f"err_{b}"].to_numpy() - df[f"err_{a}"].to_numpy()
    pids = df["player_id"].to_numpy()
    uniq, inv = np.unique(pids, return_inverse=True)
    buckets = [np.where(inv == i)[0] for i in range(len(uniq))]
    obs = float(d.mean())
    stats = np.empty(iters)
    for i in range(iters):
        pick = rng.integers(0, len(buckets), len(buckets))
        idx = np.concatenate([buckets[j] for j in pick])
        stats[i] = d[idx].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return obs, float(lo), float(hi)
