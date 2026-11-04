"""Is the pass_yards EV edge real? Cluster-bootstrapped over quarterbacks.
Preregistration: docs/research/PRISM_PASS_YARDS_EDGE_DURABILITY_PREREGISTRATION_V1.md.
No odds fetched; prices come from the committed boards. No bet is authorized by this script.
"""
from __future__ import annotations
import re, glob
import numpy as np, pandas as pd, nflreadpy as nfl
def norm(s): return re.sub(r"[^a-z]","",str(s).lower())
def payout(o): return np.where(o>0,o/100.0,100.0/np.abs(o))


def load():
    rows=[]
    for f in sorted(glob.glob("data/market_track_record/boards/*.csv")):
        wk=int(re.search(r"wk(\d+)",f).group(1))
        d=pd.read_csv(f,low_memory=False); d["wk"]=wk; rows.append(d)
    b=pd.concat(rows,ignore_index=True)
    b=b[b.market=="pass_yards"].copy(); b["k"]=b.player.map(norm)
    for c in ("vegas_line","model_proj","vegas_over_odds","vegas_under_odds"):
        b[c]=pd.to_numeric(b[c],errors="coerce")
    d=nfl.load_player_stats(seasons=[2026],summary_level="week")
    ps=d.to_pandas() if hasattr(d,"to_pandas") else d
    if "season_type" in ps.columns: ps=ps[ps.season_type=="REG"]
    ps=ps.copy(); ps["k"]=ps.player_display_name.map(norm)
    ps["wk"]=pd.to_numeric(ps.week,errors="coerce")
    ps["passing_yards"]=pd.to_numeric(ps.get("passing_yards"),errors="coerce").fillna(0.0)
    a=ps.groupby(["k","wk"]).passing_yards.sum().rename("actual").reset_index()
    b=b.merge(a,on=["k","wk"],how="inner")
    b=b[b.actual!=b.vegas_line].copy()
    b["over"]=b.model_proj>b.vegas_line
    b["win"]=(b.over==(b.actual>b.vegas_line))
    b["odds"]=np.where(b.over,b.vegas_over_odds,b.vegas_under_odds)
    b=b.dropna(subset=["odds"]).copy()
    b["profit"]=np.where(b.win,payout(b.odds),-1.0)
    b["inv_profit"]=np.where(~b.win,payout(np.where(b.over,b.vegas_under_odds,b.vegas_over_odds)),-1.0)
    return b


def cluster_ci(df, col="profit", iters=20000, seed=5):
    """Resample QUARTERBACKS (clusters), not rows."""
    rng=np.random.default_rng(seed)
    gs=[g[col].to_numpy() for _,g in df.groupby("k")]
    n=len(gs)
    out=np.empty(iters)
    for i in range(iters):
        pick=rng.integers(0,n,n)
        out[i]=np.concatenate([gs[j] for j in pick]).mean()
    return out.mean(), np.percentile(out,[2.5,97.5]), n


def row_ci(x, iters=20000, seed=5):
    rng=np.random.default_rng(seed)
    d=rng.integers(0,len(x),(iters,len(x)))
    return np.percentile(x[d].mean(1),[2.5,97.5])


def main():
    b=load()
    print(f"pass_yards graded rows with odds: {len(b)}   quarterbacks: {b.k.nunique()}   "
          f"weeks: {sorted(b.wk.unique())}")
    print(f"rows per QB: median {b.groupby('k').size().median():.0f}  "
          f"max {b.groupby('k').size().max()}")

    print("\n[T1] HEADLINE -- EV per $1, bootstrapped over QUARTERBACKS")
    m,(lo,hi),nq=cluster_ci(b)
    rlo,rhi=row_ci(b.profit.to_numpy())
    print(f"  EV {m:+.4f}   cluster 95% CI [{lo:+.4f}, {hi:+.4f}]   (clusters={nq})")
    print(f"               row-level 95% CI [{rlo:+.4f}, {rhi:+.4f}]  <- what the earlier study used")
    print(f"  G1 (cluster CI low > 0): {'PASS' if lo>0 else 'FAIL -> K1'}")
    print(f"  cluster interval is {(hi-lo)/(rhi-rlo):.2f}x the width of the row interval")

    print("\n[T2] week stability")
    pos=0
    print(f"  {'wk':>3s} {'n':>4s} {'win%':>7s} {'EV/$1':>8s}")
    for w,s in b.groupby("wk"):
        ev=s.profit.mean(); pos+= ev>0
        print(f"  {w:3d} {len(s):4d} {s.win.mean()*100:6.2f}% {ev:+8.4f}")
    print(f"  G2 (>=4 of 5 weeks positive): {pos}/5  {'PASS' if pos>=4 else 'FAIL -> K2'}")

    print("\n[T3] concentration by quarterback (top and bottom 5 by total profit)")
    qb=b.groupby("k").agg(n=("profit","size"),ev=("profit","mean"),tot=("profit","sum")).sort_values("tot")
    for lab,sl in (("worst",qb.head(5)),("best",qb.tail(5))):
        print(f"  {lab}:")
        for k,r in sl.iterrows():
            print(f"     {k[:22]:22s} n={int(r.n):3d}  EV {r.ev:+.4f}  total {r.tot:+7.2f}")
    best=qb.index[-1]
    b2=b[b.k!=best]
    m2,(lo2,hi2),nq2=cluster_ci(b2,seed=9)
    print(f"  dropping the single best QB ({best[:20]}): EV {m2:+.4f} CI [{lo2:+.4f}, {hi2:+.4f}]")
    print(f"  G3 (survives dropping best QB): {'PASS' if lo2>0 else 'FAIL -> K3'}")
    top3=qb.tail(3).tot.sum(); tot=qb.tot.sum()
    print(f"  top 3 QBs contribute {top3:+.2f} of {tot:+.2f} total profit "
          f"({top3/tot*100 if tot!=0 else float('nan'):.0f}%)")

    print("\n[T4] side")
    for lab,m_ in (("model says OVER",b.over),("model says UNDER",~b.over)):
        s=b[m_]
        if len(s)<20: continue
        print(f"  {lab:18s} n={len(s):4d} win {s.win.mean()*100:6.2f}%  EV {s.profit.mean():+.4f}")

    print("\n[T5] price bands -- DESCRIPTIVE ONLY (K5 forbids deriving a rule here)")
    b["band"]=pd.cut(b.odds,[-100000,-130,-120,-110,-100,100000],
                     labels=["worse than -130","-130..-120","-120..-110","-110..-100","plus"])
    for t,s in b.groupby("band",observed=True):
        print(f"  {str(t):16s} n={len(s):4d} win {s.win.mean()*100:6.2f}%  EV {s.profit.mean():+.4f}")

    print("\n[T6] naive controls")
    mi,(lio,hio),_=cluster_ci(b,col="inv_profit",seed=11)
    print(f"  model's side      EV {b.profit.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  INVERSE side      EV {mi:+.4f}  CI [{lio:+.4f}, {hio:+.4f}]")
    print(f"  G4 (model clearly beats inverse): "
          f"{'PASS' if b.profit.mean()>mi and lo>hio else 'fail' }")
    b.to_csv("data/research/pass_yards_edge_durability_v1.csv",index=False)


if __name__=="__main__":
    main()
