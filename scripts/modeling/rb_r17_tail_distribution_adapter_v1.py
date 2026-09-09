from __future__ import annotations
import copy, hashlib
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ResidualPools:
    non_tail: np.ndarray
    tail_30_49: np.ndarray
    tail_50_plus: np.ndarray


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join([str(int(base_seed)), *[str(p) for p in parts]]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32 - 1)


def mean_preserve(mu: float, draws: np.ndarray) -> np.ndarray:
    mu = max(0.0, float(mu))
    x = np.clip(np.asarray(draws, dtype=float), 0.0, None)
    if mu <= 0.0:
        return np.zeros_like(x)
    m = float(x.mean())
    if not np.isfinite(m) or m <= 0.0:
        return np.full_like(x, mu)
    x = x * (mu / m)
    m2 = float(x.mean())
    if m2 > 0.0:
        x *= mu / m2
    return x


def nested_tail_residuals(rng: np.random.Generator, n: int, p30: float, p50: float, pools: ResidualPools) -> np.ndarray:
    p30 = float(np.clip(p30, 0.0, 1.0)); p50 = float(np.clip(p50, 0.0, 1.0))
    w50 = min(p50, p30); w30 = max(p30 - w50, 0.0); wnon = max(0.0, 1.0 - p30)
    total = wnon + w30 + w50
    if total <= 0.0: wnon, w30, w50 = 1.0, 0.0, 0.0
    else: wnon, w30, w50 = wnon/total, w30/total, w50/total
    comp = rng.choice(3, size=int(n), p=[wnon, w30, w50])
    out = np.empty(int(n), dtype=float)
    for k, pool in [(0,pools.non_tail),(1,pools.tail_30_49),(2,pools.tail_50_plus)]:
        mask = comp == k
        if mask.any(): out[mask] = rng.choice(np.asarray(pool,float), size=int(mask.sum()), replace=True)
    return out


def target_tail_draws(mu: float, n: int, p30: float, p50: float, pools: ResidualPools, rng: np.random.Generator) -> np.ndarray:
    residuals = nested_tail_residuals(rng, n, p30, p50, pools)
    return mean_preserve(mu, float(mu) + residuals)


def rank_preserving_quantile_map(canonical: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = np.asarray(canonical, dtype=float); t = np.sort(np.asarray(target, dtype=float))
    if len(t) > 1:
        eps = max(1.0, abs(float(x.mean()))) * 1e-10
        t = t + eps * np.arange(len(t), dtype=float)
    if len(x) != len(t): raise ValueError("canonical and target draw lengths differ")
    if len(x) == 0: return x.copy()
    if np.allclose(x, x[0], atol=0.0, rtol=0.0): return mean_preserve(float(x.mean()), t)
    ranks = pd.Series(x).rank(method="average", pct=True).to_numpy(float)
    q = (np.arange(len(t), dtype=float) + 0.5) / len(t)
    mapped = np.interp(ranks, q, t, left=t[0], right=t[-1])
    return mean_preserve(float(x.mean()), mapped)


def adapt_rb_receiving_tail(result, metrics: pd.DataFrame, risk: pd.DataFrame, pools: ResidualPools, *, seed: int = 918):
    frame = metrics.copy()
    if "player_clean_key" not in frame.columns: raise ValueError("metrics missing player_clean_key")
    if "event_id" not in frame.columns: raise ValueError("metrics missing event_id")
    pos = frame.get("position_family", frame.get("position", pd.Series("", index=frame.index))).fillna("").astype(str).str.upper()
    frame = frame.assign(_is_rb=pos.eq("RB"))
    risk_req = {"event_id","player_clean_key","p30","p50"}
    if risk_req - set(risk.columns): raise ValueError(f"risk missing {sorted(risk_req-set(risk.columns))}")
    rmap = {(str(r.event_id), str(r.player_clean_key)):(float(r.p30),float(r.p50)) for r in risk.itertuples(index=False)}
    rbkeys = {(str(r.event_id), str(r.player_clean_key)) for r in frame.loc[frame._is_rb].itertuples(index=False)}
    out = copy.copy(result); out.values = {k: np.asarray(v).copy() for k,v in result.values.items()}
    audit=[]
    for game,pkey in sorted(rbkeys):
        key=(game,pkey,"rec_yards")
        if key not in out.values or (game,pkey) not in rmap: continue
        canonical = np.asarray(result.values[key], dtype=float)
        p30,p50 = rmap[(game,pkey)]
        rng=np.random.default_rng(stable_seed(seed,game,pkey,"adapter"))
        target=target_tail_draws(float(canonical.mean()), len(canonical), p30,p50,pools,rng)
        adapted=rank_preserving_quantile_map(canonical,target)
        out.values[key]=adapted
        rush_key=(game,pkey,"rush_yards"); combo_key=(game,pkey,"rush_rec_yards")
        if rush_key in out.values and combo_key in out.values:
            out.values[combo_key]=np.asarray(out.values[rush_key],float)+adapted
        audit.append({"event_id":game,"player_clean_key":pkey,"p30":p30,"p50":p50,"canonical_mean":float(canonical.mean()),"adapted_mean":float(adapted.mean())})
    return out, pd.DataFrame(audit)
